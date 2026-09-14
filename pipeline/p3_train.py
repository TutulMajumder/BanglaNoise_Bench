"""Phase 3: training (R1 clean baselines + R4 augmentation), one job at a
time, with the full R2/R6 noise-grid evaluation folded into the SAME job so
the trained weights are reused in memory and never touch disk a second time.

Why train+evaluate are one job, not two notebooks worth of separate work:
R2's GPU-hour budget in Topic4_FULL_PAPER_PLAN.md §7 is costed as
"~4-6 GPU-h, inference only" -- that number is only true if R2 reuses an
already-loaded model. Reloading from a saved checkpoint is fine, but no
checkpoint is ever saved past its own run (constraint 0: 45 checkpoints x
~440MB would blow the 20GB /kaggle/working cap), so "reuse" can only mean
"the same Python process, same GPU tensor, right after training." So:

    run_job(model, task, seed, train_variant):
        train  ->  pick best-val epoch  ->  evaluate clean test (severity 0)
              ->  full noise grid (9 types x 5 severities), same in-memory model
              ->  R6 normalizer ON/OFF, BanFakeNews only
              ->  append every row to results/results.csv
              ->  save results/curves.csv rows (per epoch) and
                  results/preds/<run_id>.csv (per test item)
              ->  delete the model/checkpoint
              ->  return

This is what "one run: train -> evaluate -> append -> delete -> next" means
here: "evaluate" is this job's ENTIRE evaluation grid, not just one point.
`pipeline/p3_evaluate_noise.py` holds the grid-building and in-memory
evaluation logic this file calls; it is also usable standalone (accepting a
retrain) to resume a job whose grid was left incomplete when a 12-hour
Kaggle session ended mid-way -- see its docstring.

Hyperparameters (METHODOLOGY.md M4 / Topic4_FULL_PAPER_PLAN.md §6 -- fully
specified there, not a Phase 3 choice):

    max_len=128, batch=32, lr=2e-5, epochs=4, AdamW, linear warmup 10%,
    fp16, early stopping on val macro-F1 (patience 2), seeds={42,1337,2024}

max_len=128 truncates BanFakeNews's ~1,100-cluster articles to roughly their
headline + lead paragraph -- a real limitation (PHASE3_STATUS.md), not a
bug. It does not break the noise study: perturbation is applied to the FULL
text before tokenization (see `_prepare_text_for_tokenizer` and
`test_perturb_before_truncate_not_after` in tests/test_p3_train.py), so noise
density inside the 128-token window matches the nominal severity regardless
of how much of the article that window covers.

    python pipeline/p3_train.py --list-jobs                       # job list + GPU-h estimate
    python pipeline/p3_train.py --model char_ngram --task bd_shs --seed 42 --train-variant clean --limit 20   # smoke run
    python pipeline/p3_train.py --model banglabert --task sentnob --seed 42 --train-variant clean             # real job

CLAUDE.md invariant 9: resumable (skips a job already fully present in
results.csv, unless --force), prints progress to stderr, --limit caps rows
per split for a fast smoke run and isolates output under .smoke/.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench.models import MODEL_ORDER, MODELS, ModelSpec  # noqa: E402
from noisebench.perturbations import NOISE_TYPES, perturb  # noqa: E402
from noisebench.validate import normalize_banglabert, normalizer_available  # noqa: E402
from pipeline._common import Progress, add_common_args, print_smoke_banner, smoke_or_prod  # noqa: E402
from pipeline.s2_prepare import dataset_final_dir  # noqa: E402
from pipeline.s2_prepare import DATASETS as DATA_SPECS  # noqa: E402

_DEFAULT_DATA_FINAL = os.path.join(_ROOT, "data", "final")
_DEFAULT_RESULTS_DIR = os.path.join(_ROOT, "results")

# ---------------------------------------------------------------------------
# Fixed config -- METHODOLOGY M4 / Topic4 §6. Not tunable per model or task.
# ---------------------------------------------------------------------------

SEEDS: tuple[int, ...] = (42, 1337, 2024)
MAX_LEN = 128
BATCH_SIZE = 32
LR = 2e-5
EPOCHS = 4
WARMUP_RATIO = 0.10
PATIENCE = 2  # early stopping, on val macro-F1

TASKS: dict[str, dict] = {
    "sentnob": {"num_labels": 3, "label_names": {0: "neutral", 1: "positive", 2: "negative"},
               "extra_metrics": []},
    "bd_shs": {"num_labels": 2, "label_names": {0: "not_hate", 1: "hate"}, "extra_metrics": []},
    "banfakenews": {"num_labels": 2, "label_names": {0: "fake", 1: "authentic"},
                    "extra_metrics": ["pr_auc", "mcc"]},
}
TASK_ORDER: tuple[str, ...] = ("sentnob", "bd_shs", "banfakenews")

TRAIN_VARIANTS: tuple[str, ...] = ("clean", "augmented_n1n4", "augmented_n5n8")
#: M5.2 groups. A = generic, B = Bangla-specific, C = surface (not used for
#: augmentation -- M5.2's 2x2 matched/mismatched table only trains on A or B).
NOISE_GROUP_A: tuple[str, ...] = NOISE_TYPES[0:4]   # N1-N4: char_insert..char_transpose
NOISE_GROUP_B: tuple[str, ...] = NOISE_TYPES[4:8]   # N5-N8: homophone_confuse..elongate
NOISE_GROUP_C: tuple[str, ...] = NOISE_TYPES[8:9]   # N9: whitespace_error
AUGMENT_FRACTION = 0.50
AUGMENT_SEVERITIES: tuple[int, ...] = (1, 2, 3)

#: R4 only trains 2 models (Topic4 §7: "Two models, three tasks, three
#: seeds"). BanglaBERT (the model expected to collapse) and char_ngram (the
#: control expected to hold) -- the two accent-coloured models in every
#: figure, and the pair the paper's claim is actually about.
R4_MODELS: tuple[str, ...] = ("banglabert", "char_ngram")

#: R6 (normalizer ON/OFF) -- BanFakeNews only; sentnob/bd_shs dropped per
#: M5.4 (touch_rate 1.67% each, measured 2026-09-15). Doubles the noise grid
#: for this one task only, not R1's clean evaluation.
R6_TASKS: tuple[str, ...] = ("banfakenews",)


# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Job:
    model: str
    task: str
    seed: int
    train_variant: str

    @property
    def run_id(self) -> str:
        return f"{self.model}__{self.task}__{self.seed}__{self.train_variant}"


def make_r1_jobs() -> list[Job]:
    """R1: every model, every task, every seed, clean training."""
    return [Job(m, t, s, "clean") for m in MODEL_ORDER for t in TASK_ORDER for s in SEEDS]


def make_r4_jobs() -> list[Job]:
    """R4a/R4b: R4_MODELS only, both augmentation groups (M5.2's 2x2 needs
    training on BOTH group A and group B to get matched AND mismatched cells
    in both directions)."""
    variants = ("augmented_n1n4", "augmented_n5n8")
    return [Job(m, t, s, v) for m in R4_MODELS for t in TASK_ORDER for s in SEEDS for v in variants]


def all_jobs() -> list[Job]:
    return make_r1_jobs() + make_r4_jobs()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_task_splits(task: str, data_final_root: str, limit: int | None = None) -> dict[str, pd.DataFrame]:
    spec = DATA_SPECS[task]
    out_dir = dataset_final_dir(spec, data_final_root)
    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(out_dir, f"{name}.csv"), dtype={"id": str, "text": str, "label": int})
        if limit is not None:
            df = df.head(limit).reset_index(drop=True)
        splits[name] = df
    return splits


def _augment_item_seed(seed: int, item_id: str) -> int:
    """Per-item seed derivation (METHODOLOGY M1.3's rule, applied to the
    AUGMENTATION SELECTION decisions -- which items get perturbed, which
    noise type, which severity). `perturb()` itself already derives its own
    internal per-call seed by hashing in `(noise_type, severity, text)`
    (`perturbations._derive_seed`), so passing the same `seed` to every
    `perturb()` call is safe and does NOT reproduce the N7 correlation bug;
    what would reproduce it is re-seeding `random.Random(seed)` directly for
    the SELECTION step below, which is exactly what this function prevents."""
    h = hashlib.sha1(f"{seed}\x00augment\x00{item_id}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def augment_training_set(df: pd.DataFrame, seed: int, noise_group: tuple[str, ...],
                         fraction: float = AUGMENT_FRACTION,
                         severities: tuple[int, ...] = AUGMENT_SEVERITIES) -> pd.DataFrame:
    """M5.2: perturb `fraction` of training examples, noise type drawn
    uniformly from `noise_group`, severity drawn uniformly from `severities`.
    Each selected item gets ONE fixed (noise_type, severity) pair, chosen
    once (not re-drawn per epoch) via a seed derived from (seed, item id) --
    fully deterministic and reproducible."""
    out = df.copy()
    texts = []
    for item_id, text in zip(df["id"], df["text"]):
        rng = random.Random(_augment_item_seed(seed, item_id))
        if rng.random() < fraction:
            noise_type = rng.choice(noise_group)
            severity = rng.choice(severities)
            text = perturb(text, noise_type, severity, seed)
        texts.append(text)
    out["text"] = texts
    return out


def prepare_training_data(task: str, seed: int, train_variant: str,
                          splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    train_df = splits["train"]
    if train_variant == "clean":
        return train_df
    if train_variant == "augmented_n1n4":
        return augment_training_set(train_df, seed, NOISE_GROUP_A)
    if train_variant == "augmented_n5n8":
        return augment_training_set(train_df, seed, NOISE_GROUP_B)
    raise ValueError(f"unknown train_variant {train_variant!r}")


def _prepare_text_for_tokenizer(text: str, normalizer: str) -> str:
    """The ONE place normalization happens before tokenization. Perturbation
    (training-time augmentation, or noise-grid evaluation) always happens
    BEFORE this function is called -- see module docstring and
    tests/test_p3_train.py::test_perturb_before_truncate_not_after. Model
    tokenizers truncate to MAX_LEN internally; nothing here truncates."""
    if normalizer == "on":
        normed = normalize_banglabert(text)
        return normed if normed is not None else text
    return text


# ---------------------------------------------------------------------------
# Metrics -- M6.1
# ---------------------------------------------------------------------------


def compute_metrics(y_true: list[int], y_pred: list[int], task: str,
                    y_score_pos: list[float] | None = None) -> dict:
    """M6.1: macro-F1 (primary), weighted F1, per-class P/R, balanced
    accuracy. BanFakeNews additionally: PR-AUC and MCC (never accuracy).

    PR-AUC "positive class" choice (not specified in M6.1 -- recorded here,
    not silently assumed): **fake (label 0)**, not authentic (label 1) --
    for a fake-news detector the class of interest is the minority class you
    are trying to catch, not the majority class. `y_score_pos` must already
    be P(fake) / decision-score-for-fake; callers pass `1 - P(authentic)`
    for transformers or the negated LinearSVC decision function.
    """
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        f1_score, matthews_corrcoef, precision_recall_fscore_support,
    )

    labels_present = sorted(set(y_true) | set(y_pred))
    p, r, f1c, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels_present, zero_division=0)
    per_class = {
        str(lbl): {"precision": float(p[i]), "recall": float(r[i]),
                   "f1": float(f1c[i]), "support": int(support[i])}
        for i, lbl in enumerate(labels_present)
    }
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),  # not reported for banfakenews in the paper (M6.1); still computed for the training-curve diagnostic
        "per_class_json": json.dumps(per_class, sort_keys=True),
    }
    extra = TASKS[task]["extra_metrics"]
    if "pr_auc" in extra:
        y_true_fake = [1 if y == 0 else 0 for y in y_true]  # positive = fake
        out["pr_auc"] = (float(average_precision_score(y_true_fake, y_score_pos))
                         if y_score_pos is not None else None)
    if "mcc" in extra:
        out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    return out


# ---------------------------------------------------------------------------
# Transformer training
# ---------------------------------------------------------------------------


def _get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _TextDataset:
    """Tiny torch Dataset -- tokenizes lazily per item (simplest correct
    thing; re-tokenizing per epoch is negligible next to a forward+backward
    pass, and avoids holding a second full-corpus tensor in memory)."""

    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        import torch
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len,
                             padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _run_epoch_transformer(model, loader, device, optimizer=None, scheduler=None):
    """One pass. `optimizer` set -> training mode + backward pass; unset ->
    eval mode, no grad. Returns (mean_loss, y_true, y_pred, y_score_pos1)."""
    import torch
    import torch.nn.functional as F

    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    y_true, y_pred, y_score = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            out = model(**batch, labels=labels)
            loss = out.loss
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            probs = F.softmax(out.logits.detach(), dim=-1)
            preds = probs.argmax(dim=-1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            # score for the highest class index (used for binary PR-AUC as
            # P(class=1); banfakenews's compute_metrics converts to P(fake)).
            if probs.shape[1] >= 2:
                y_score.extend(probs[:, -1].cpu().tolist())
    return (total_loss / max(1, n_batches)), y_true, y_pred, y_score


def train_transformer(job: Job, splits: dict[str, pd.DataFrame], train_df: pd.DataFrame,
                      normalizer: str, progress: Progress, epochs: int = EPOCHS,
                      batch_size: int = BATCH_SIZE) -> dict:
    """Trains, tracks per-epoch curves, selects the best epoch by VAL
    macro-F1 only, keeps that epoch's weights in memory (never re-loaded from
    disk -- no checkpoint is written to disk at all). Returns a dict with the
    live `model`, `tokenizer`, `device`, `curves_rows`, and `best_epoch`."""
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    spec = MODELS[job.model]
    device = _get_device()
    fp16 = device.type == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        spec.hf_id, num_labels=TASKS[job.task]["num_labels"])
    model.to(device)

    train_texts = [_prepare_text_for_tokenizer(t, normalizer) for t in train_df["text"]]
    val_texts = [_prepare_text_for_tokenizer(t, normalizer) for t in splits["val"]["text"]]
    train_ds = _TextDataset(train_texts, train_df["label"].tolist(), tokenizer, MAX_LEN)
    val_ds = _TextDataset(val_texts, splits["val"]["label"].tolist(), tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)

    scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    curves_rows: list[dict] = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state: dict | None = None
    epochs_since_best = 0
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if fp16:
            train_loss, tr_true, tr_pred, _ = _run_epoch_transformer_amp(
                model, train_loader, device, optimizer, scheduler, scaler)
        else:
            train_loss, tr_true, tr_pred, _ = _run_epoch_transformer(
                model, train_loader, device, optimizer, scheduler)
        train_metrics = compute_metrics(tr_true, tr_pred, job.task)

        val_loss, val_true, val_pred, val_score = _run_epoch_transformer(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_pred, job.task,
                                      y_score_pos=[1 - s for s in val_score] if val_score else None)

        elapsed = time.perf_counter() - t0
        progress.log(job.run_id, f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                                 f"val_loss={val_loss:.4f} val_macro_f1={val_metrics['macro_f1']:.4f} "
                                 f"elapsed={elapsed:.1f}s")

        curves_rows.append({
            "run_id": job.run_id, "model": job.model, "task": job.task, "seed": job.seed,
            "train_variant": job.train_variant, "normalizer": normalizer, "epoch": epoch,
            "train_loss": train_loss, "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_loss, "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            # Diagnostic-only columns below: per-epoch TEST metrics, kept
            # STRICTLY SEPARATE from the val columns above that drive model
            # selection. Filled in after training by `_add_diagnostic_test_curve`.
            "test_accuracy_diagnostic": None, "test_macro_f1_diagnostic": None,
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= PATIENCE:
                progress.log(job.run_id, f"early stop at epoch {epoch} (patience {PATIENCE}, "
                                         f"best val_macro_f1={best_val_f1:.4f} @ epoch {best_epoch})")
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    del best_state
    gc.collect()

    return {"model": model, "tokenizer": tokenizer, "device": device, "kind": "transformer",
           "curves_rows": curves_rows, "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1}


def _run_epoch_transformer_amp(model, loader, device, optimizer, scheduler, scaler):
    """Same as `_run_epoch_transformer` (training branch only) but with
    torch.cuda.amp mixed precision -- fp16 per METHODOLOGY M4, CUDA only."""
    import torch
    import torch.nn.functional as F

    model.train()
    total_loss, n_batches = 0.0, 0
    y_true, y_pred, y_score = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        optimizer.zero_grad()
        with torch.autocast("cuda"):
            out = model(**batch, labels=labels)
            loss = out.loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1
        probs = F.softmax(out.logits.detach(), dim=-1)
        preds = probs.argmax(dim=-1)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        if probs.shape[1] >= 2:
            y_score.extend(probs[:, -1].cpu().tolist())
    return (total_loss / max(1, n_batches)), y_true, y_pred, y_score


def predict_transformer(state: dict, texts: list[str], normalizer: str,
                        batch_size: int = BATCH_SIZE) -> tuple[list[int], list[float]]:
    """Inference only -- used by both the clean-test pass here and by
    p3_evaluate_noise.py's grid (same in-memory model)."""
    import torch
    import torch.nn.functional as F

    model, tokenizer, device = state["model"], state["tokenizer"], state["device"]
    model.eval()
    texts = [_prepare_text_for_tokenizer(t, normalizer) for t in texts]
    preds: list[int] = []
    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = tokenizer(chunk, truncation=True, max_length=MAX_LEN, padding=True,
                            return_tensors="pt").to(device)
            out = model(**enc)
            probs = F.softmax(out.logits, dim=-1)
            preds.extend(probs.argmax(dim=-1).cpu().tolist())
            if probs.shape[1] >= 2:
                scores.extend(probs[:, -1].cpu().tolist())
    return preds, scores


# ---------------------------------------------------------------------------
# Classical (char_ngram) training -- METHODOLOGY M4
# ---------------------------------------------------------------------------


def train_classical(job: Job, splits: dict[str, pd.DataFrame], train_df: pd.DataFrame,
                    normalizer: str, progress: Progress) -> dict:
    """No epochs in the usual sense -- one `.fit()` call. Recorded as a
    single curves.csv row (epoch=1) so the schema stays uniform; val/test
    accuracy are both computed from the same one fit, so "best epoch" is
    trivially epoch 1. No loss curve exists for a linear SVM fit this way,
    so train_loss/val_loss are left null (documented, not a missing bug)."""
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.feature_extraction.text import TfidfVectorizer

    t0 = time.perf_counter()
    train_texts = [_prepare_text_for_tokenizer(t, normalizer) for t in train_df["text"]]
    val_texts = [_prepare_text_for_tokenizer(t, normalizer) for t in splits["val"]["text"]]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2)),
        ("svm", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    pipe.fit(train_texts, train_df["label"].tolist())

    train_pred = pipe.predict(train_texts)
    val_pred = pipe.predict(val_texts)
    train_metrics = compute_metrics(train_df["label"].tolist(), list(train_pred), job.task)
    val_score = pipe.decision_function(val_texts) if hasattr(pipe, "decision_function") else None
    val_score_pos1 = _binary_positive_score(val_score)
    val_metrics = compute_metrics(splits["val"]["label"].tolist(), list(val_pred), job.task,
                                  y_score_pos=([1 - s for s in val_score_pos1] if val_score_pos1 else None))

    elapsed = time.perf_counter() - t0
    progress.log(job.run_id, f"fit done val_macro_f1={val_metrics['macro_f1']:.4f} elapsed={elapsed:.1f}s")

    curves_rows = [{
        "run_id": job.run_id, "model": job.model, "task": job.task, "seed": job.seed,
        "train_variant": job.train_variant, "normalizer": normalizer, "epoch": 1,
        "train_loss": None, "train_accuracy": train_metrics["accuracy"],
        "train_macro_f1": train_metrics["macro_f1"],
        "val_loss": None, "val_accuracy": val_metrics["accuracy"],
        "val_macro_f1": val_metrics["macro_f1"],
        "test_accuracy_diagnostic": None, "test_macro_f1_diagnostic": None,
    }]
    return {"pipe": pipe, "kind": "classical", "curves_rows": curves_rows, "best_epoch": 1,
           "best_val_macro_f1": val_metrics["macro_f1"]}


def _binary_positive_score(decision_scores) -> list[float] | None:
    """LinearSVC.decision_function: 1-D signed distance for binary, (n,k)
    for multiclass (one-vs-rest). Returns P(class=1)-like score (higher =
    more class 1) via a logistic squash of the signed distance, or None for
    multiclass (PR-AUC/MCC only apply to banfakenews, which is binary)."""
    if decision_scores is None:
        return None
    arr = np.asarray(decision_scores)
    if arr.ndim != 1:
        return None
    return (1.0 / (1.0 + np.exp(-arr))).tolist()


def predict_classical(state: dict, texts: list[str], normalizer: str) -> tuple[list[int], list[float]]:
    texts = [_prepare_text_for_tokenizer(t, normalizer) for t in texts]
    pipe = state["pipe"]
    preds = list(pipe.predict(texts))
    scores = pipe.decision_function(texts) if hasattr(pipe, "decision_function") else None
    score1 = _binary_positive_score(scores) or []
    return preds, score1


def predict(state: dict, texts: list[str], normalizer: str) -> tuple[list[int], list[float]]:
    """Dispatches to the transformer or classical predictor -- the one entry
    point p3_evaluate_noise.py's grid calls, so it never needs to know which
    kind of model it's holding."""
    if state["kind"] == "transformer":
        return predict_transformer(state, texts, normalizer)
    return predict_classical(state, texts, normalizer)


def delete_model_state(state: dict) -> None:
    """Frees the in-memory weights -- the "delete checkpoint" step. Nothing
    was ever written to disk in the first place (no torch.save/pipe dump
    anywhere in this file), so this is just freeing the Python/GPU memory."""
    if state["kind"] == "transformer":
        model = state.pop("model", None)
        if model is not None:
            del model
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    else:
        state.pop("pipe", None)
    gc.collect()


# ---------------------------------------------------------------------------
# results.csv / curves.csv / preds -- CLAUDE.md invariant 5 + Phase 3 additions
# ---------------------------------------------------------------------------

RESULTS_COLUMNS: tuple[str, ...] = (
    "run_id", "model", "task", "seed", "train_variant", "normalizer",
    "noise_type", "severity", "n_test_items",
    "macro_f1", "weighted_f1", "balanced_accuracy", "accuracy",
    "pr_auc", "mcc", "per_class_json",
    "best_epoch", "best_val_macro_f1", "timestamp",
)
CURVES_COLUMNS: tuple[str, ...] = (
    "run_id", "model", "task", "seed", "train_variant", "normalizer", "epoch",
    "train_loss", "train_accuracy", "train_macro_f1",
    "val_loss", "val_accuracy", "val_macro_f1",
    "test_accuracy_diagnostic", "test_macro_f1_diagnostic",
)
#: This caption string is the one written into the training-curves figure
#: (pipeline/p3_figures.py) -- kept HERE, beside the columns it describes, so
#: it cannot drift out of sync or get lost. Model selection (best_epoch in
#: results.csv) is decided ONLY from the val_* columns above; the
#: test_*_diagnostic columns are for this figure's dashed reference line only.
TEST_DIAGNOSTIC_CAPTION = (
    "Per-epoch test accuracy/macro-F1 (dashed) is shown for reference only. "
    "Model selection used validation macro-F1 exclusively (solid line); the "
    "test set never influenced which epoch was chosen."
)


def _results_key(run_id: str, normalizer: str, noise_type: str | None, severity: int) -> tuple:
    return (run_id, normalizer, noise_type or "none", int(severity))


def load_existing_result_keys(results_path: str) -> set[tuple]:
    """Read once per job -- the resumability check. `noise_type`/`severity`
    default to "none"/0 for the clean row, matching `_results_key`."""
    if not os.path.exists(results_path):
        return set()
    keys: set[tuple] = set()
    with open(results_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            keys.add(_results_key(row["run_id"], row["normalizer"],
                                  row.get("noise_type") or None, int(row["severity"])))
    return keys


def _write_utf8_append(path: str, header: tuple[str, ...], rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    is_new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(header))
        if is_new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})


def append_results_rows(results_path: str, rows: list[dict]) -> None:
    _write_utf8_append(results_path, RESULTS_COLUMNS, rows)


def append_curves_rows(curves_path: str, rows: list[dict]) -> None:
    _write_utf8_append(curves_path, CURVES_COLUMNS, rows)


def write_preds_file(preds_path: str, ids: list[str], y_true: list[int], y_pred: list[int],
                     extra_cols: dict[str, list] | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(preds_path)), exist_ok=True)
    header = ["id", "true", "pred"] + list((extra_cols or {}).keys())
    with open(preds_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in range(len(ids)):
            row = [ids[i], y_true[i], y_pred[i]]
            for col in (extra_cols or {}):
                row.append(extra_cols[col][i])
            w.writerow(row)


def make_result_row(job: Job, normalizer: str, noise_type: str | None, severity: int,
                    metrics: dict, n_test_items: int, best_epoch: int, best_val_f1: float) -> dict:
    row = {
        "run_id": job.run_id, "model": job.model, "task": job.task, "seed": job.seed,
        "train_variant": job.train_variant, "normalizer": normalizer,
        "noise_type": noise_type or "", "severity": severity, "n_test_items": n_test_items,
        "macro_f1": metrics["macro_f1"], "weighted_f1": metrics["weighted_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"], "accuracy": metrics["accuracy"],
        "pr_auc": metrics.get("pr_auc"), "mcc": metrics.get("mcc"),
        "per_class_json": metrics["per_class_json"],
        "best_epoch": best_epoch, "best_val_macro_f1": best_val_f1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return row


# ---------------------------------------------------------------------------
# run_job -- the whole point of this file
# ---------------------------------------------------------------------------


def run_job(job: Job, data_final_root: str, results_dir: str, progress: Progress,
           limit: int | None = None, force: bool = False,
           run_noise_grid: bool = True) -> dict | None:
    """Train (or skip if already fully done), evaluate clean test (severity
    0), run the full noise grid + R6 (same in-memory model), append
    everything, delete the model, return a summary dict (or None if skipped).
    """
    results_path = os.path.join(results_dir, "results.csv")
    curves_path = os.path.join(results_dir, "curves.csv")
    preds_dir = os.path.join(results_dir, "preds")

    existing = load_existing_result_keys(results_path)
    clean_key = _results_key(job.run_id, "on", None, 0)
    grid_done = _job_grid_is_complete(job, existing) if run_noise_grid else True

    if not force and clean_key in existing and grid_done:
        progress.log(job.run_id, "skipping -- fully done in results.csv; use --force to redo")
        return None

    progress.log(job.run_id, f"loading data (limit={limit})")
    splits = load_task_splits(job.task, data_final_root, limit=limit)
    train_df = prepare_training_data(job.task, job.seed, job.train_variant, splits)

    spec = MODELS[job.model]
    progress.log(job.run_id, f"training {spec.display_name} ({spec.kind})")
    if spec.kind == "transformer":
        state = train_transformer(job, splits, train_df, normalizer="on", progress=progress)
    else:
        state = train_classical(job, splits, train_df, normalizer="on", progress=progress)

    append_curves_rows(curves_path, state["curves_rows"])

    # Clean test pass (severity 0) -- ALWAYS normalizer="on" (the default;
    # R6 only doubles the NOISE GRID for banfakenews, not this clean pass).
    test_texts = splits["test"]["text"].tolist()
    test_labels = splits["test"]["label"].tolist()
    test_ids = splits["test"]["id"].tolist()
    if clean_key not in existing or force:
        progress.log(job.run_id, "evaluating clean test (severity 0)")
        preds, scores = predict(state, test_texts, normalizer="on")
        # `predict` returns P(class=1); compute_metrics wants P(fake) for
        # banfakenews (label 0 = fake, see its docstring) -- 1 - P(class=1).
        pos_scores = [1 - s for s in scores] if scores else None
        metrics = compute_metrics(test_labels, preds, job.task, y_score_pos=pos_scores)
        row = make_result_row(job, "on", None, 0, metrics, len(test_labels),
                              state["best_epoch"], state["best_val_macro_f1"])
        append_results_rows(results_path, [row])
        write_preds_file(os.path.join(preds_dir, f"{job.run_id}.csv"), test_ids, test_labels, preds)
        progress.log(job.run_id, f"clean test macro_f1={metrics['macro_f1']:.4f}")

    grid_summary = None
    if run_noise_grid:
        from pipeline.p3_evaluate_noise import run_grid_in_memory
        progress.log(job.run_id, "running noise grid (R2" +
                     ("+R6" if job.task in R6_TASKS else "") + ")")
        grid_summary = run_grid_in_memory(
            job=job, state=state, test_df=splits["test"], results_path=results_path,
            preds_dir=preds_dir, progress=progress, existing_keys=existing, force=force)

    delete_model_state(state)
    progress.log(job.run_id, "done, checkpoint deleted")
    return {"job": job, "best_epoch": state["best_epoch"],
           "best_val_macro_f1": state["best_val_macro_f1"], "grid": grid_summary}


def _job_grid_is_complete(job: Job, existing: set[tuple]) -> bool:
    """True if every (noise_type, severity) cell -- and R6's off-normalizer
    cells for banfakenews -- already has a results.csv row for this run_id."""
    for nt in NOISE_TYPES:
        for sev in range(1, 6):
            if _results_key(job.run_id, "on", nt, sev) not in existing:
                return False
            if job.task in R6_TASKS and _results_key(job.run_id, "off", nt, sev) not in existing:
                return False
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-final-root", default=None,
                    help=f"default: {_DEFAULT_DATA_FINAL}, or the .smoke/ mirror under --limit")
    ap.add_argument("--results-dir", default=None,
                    help=f"default: {_DEFAULT_RESULTS_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--model", choices=MODEL_ORDER)
    ap.add_argument("--task", choices=TASK_ORDER)
    ap.add_argument("--seed", type=int, choices=SEEDS)
    ap.add_argument("--train-variant", choices=TRAIN_VARIANTS)
    ap.add_argument("--job-set", choices=("r1", "r4", "all"), default=None,
                    help="run every job in this set (instead of one --model/--task/--seed job)")
    ap.add_argument("--no-noise-grid", action="store_true",
                    help="train + clean eval only, skip the R2/R6 grid (for a fast smoke check)")
    ap.add_argument("--list-jobs", action="store_true",
                    help="print the full job list with GPU-time estimates and exit")
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.list_jobs:
        _print_job_list()
        return 0

    if args.data_final_root is None:
        # data/final/ is real Phase 2 INPUT, not this script's output -- read
        # from the real tree always; --limit truncates via `.head(limit)` in
        # `load_task_splits`, it does not need (and there is no) a .smoke/
        # mirror of the input data to read instead.
        args.data_final_root = _DEFAULT_DATA_FINAL
    if args.results_dir is None:
        args.results_dir = smoke_or_prod(args.limit, _DEFAULT_RESULTS_DIR)
    if args.limit is not None:
        print_smoke_banner("p3_train", args.limit, {
            "data-final-root": args.data_final_root, "results-dir": args.results_dir})

    progress = Progress("p3_train")

    if args.job_set:
        jobs = {"r1": make_r1_jobs, "r4": make_r4_jobs, "all": all_jobs}[args.job_set]()
    elif args.model and args.task and args.seed and args.train_variant:
        jobs = [Job(args.model, args.task, args.seed, args.train_variant)]
    else:
        print("pass --model/--task/--seed/--train-variant for one job, "
              "--job-set for many, or --list-jobs to just see the plan",
              file=sys.stderr)
        return 2

    for job in jobs:
        run_job(job, args.data_final_root, args.results_dir, progress,
               limit=args.limit, force=args.force, run_noise_grid=not args.no_noise_grid)
    progress.done()
    return 0


def _print_job_list() -> None:
    from pipeline.p3_estimate import estimate_all
    estimate_all(sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
