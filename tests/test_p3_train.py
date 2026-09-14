"""Tests for pipeline/p3_train.py + pipeline/p3_evaluate_noise.py.

No real datasets or downloaded models required -- these exercise pure
functions (metrics, seeding, augmentation, resumability bookkeeping) and the
one invariant explicitly required by the Phase 3 spec: perturbation is
applied to the FULL text before any truncation happens, so noise density
inside BanFakeNews's 128-token window matches the nominal severity (see
p3_train.py's module docstring and the max_len=128 limitation it records).
"""

from __future__ import annotations

import importlib
import os
import sys

import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.bangla_samples import SENTENCES

p3_train = importlib.import_module("pipeline.p3_train")
p3_evaluate_noise = importlib.import_module("pipeline.p3_evaluate_noise")


def _long_text(n_repeats: int = 40) -> str:
    """A text long enough that its tokenized length clears MAX_LEN=128 by a
    wide margin -- concatenating real Bangla sentences (bangla_samples.py),
    never inventing text."""
    base = list(dict.fromkeys(SENTENCES))
    return " ".join((base * ((n_repeats // len(base)) + 1))[:n_repeats])


# ---------------------------------------------------------------------------
# The explicitly-required test: perturb() sees the FULL text, not a
# pre-truncated slice -- truncation only ever happens inside the tokenizer,
# downstream of perturbation.
# ---------------------------------------------------------------------------


def test_perturb_before_truncate_not_after_in_augmentation(monkeypatch):
    """augment_training_set (training-time M5.2 augmentation): every call
    into perturb() must receive the ITEM'S FULL, UNTRUNCATED text."""
    long_text = _long_text()
    assert len(long_text) > 600, "fixture text must be long enough to matter for truncation"

    df = pd.DataFrame({
        "id": [f"item{i}" for i in range(20)],
        "text": [long_text] * 20,
        "label": [0] * 20,
    })

    seen_lengths: list[int] = []

    def _spy_perturb(text, noise_type, severity, seed):
        seen_lengths.append(len(text))
        return text  # identity -- only checking what it was called WITH

    monkeypatch.setattr(p3_train, "perturb", _spy_perturb)
    out = p3_train.augment_training_set(df, seed=42, noise_group=p3_train.NOISE_GROUP_A)

    assert len(out) == len(df)
    assert seen_lengths, "fixture's 50% augment fraction should have selected at least one item"
    for n in seen_lengths:
        assert n == len(long_text), (
            f"perturb() was called with {n} chars, expected the full "
            f"{len(long_text)}-char text -- augmentation must run before any "
            "truncation, not on an already-truncated string")


def test_perturb_before_truncate_not_after_in_noise_grid(monkeypatch):
    """run_grid_in_memory (R2/R6 noise-grid evaluation): same requirement for
    the TEST-time noise grid -- perturb() must see the full test item text,
    and _prepare_text_for_tokenizer must not shorten it before the tokenizer
    does (MAX_LEN=128 truncation happens inside predict(), never before)."""
    long_text = _long_text()

    class _FakeClassicalState(dict):
        pass

    # A trivial classical predictor: predicts label 0 for everything, no
    # real model needed since this test only checks what text perturb() saw.
    class _StubPipe:
        def predict(self, texts):
            return [0] * len(texts)

        def decision_function(self, texts):
            return [0.0] * len(texts)

    state = {"kind": "classical", "pipe": _StubPipe(), "best_epoch": 1, "best_val_macro_f1": 0.5}
    test_df = pd.DataFrame({
        "id": ["a", "b", "c"],
        "text": [long_text, long_text, long_text],
        "label": [0, 1, 0],
    })

    seen_lengths: list[int] = []

    def _spy_perturb(text, noise_type, severity, seed):
        seen_lengths.append(len(text))
        return text

    monkeypatch.setattr(p3_evaluate_noise, "perturb", _spy_perturb)

    job = p3_train.Job(model="char_ngram", task="banfakenews", seed=42, train_variant="clean")
    results_path = "/dev/null/unused.csv"  # never actually written by this test's stub path

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        results_path = os.path.join(tmp, "results.csv")
        preds_dir = os.path.join(tmp, "preds")
        progress = p3_train.Progress("test")
        summary = p3_evaluate_noise.run_grid_in_memory(
            job=job, state=state, test_df=test_df, results_path=results_path,
            preds_dir=preds_dir, progress=progress, existing_keys=set(), force=False)

    # banfakenews is an R6_TASKS member -> normalizer on AND off, so every
    # one of the 9*5*2=90 grid cells calls perturb() once per test item.
    assert summary["n_computed"] == 90
    assert seen_lengths, "expected perturb() to be called for the noise grid"
    for n in seen_lengths:
        assert n == len(long_text), (
            f"perturb() in the noise grid was called with {n} chars, "
            f"expected the full {len(long_text)}-char item -- truncation to "
            "MAX_LEN must happen only inside predict()'s tokenizer call, "
            "after perturbation, never before it")


def test_prepare_text_for_tokenizer_never_truncates():
    """_prepare_text_for_tokenizer is the one non-tokenizer text-prep step in
    the whole pipeline -- it must not itself shorten text (that would be
    truncation happening outside, and possibly before, the tokenizer)."""
    long_text = _long_text()
    out_on = p3_train._prepare_text_for_tokenizer(long_text, normalizer="off")
    assert len(out_on) == len(long_text)


# ---------------------------------------------------------------------------
# M1.3 per-item seeding: augmentation selection must not reproduce the N7
# seed-correlation bug (re-seeding random.Random(seed) directly per item).
# ---------------------------------------------------------------------------


def test_augment_training_set_deterministic_and_decorrelated():
    df = pd.DataFrame({
        "id": [f"item{i}" for i in range(200)],
        "text": [SENTENCES[i % len(SENTENCES)] for i in range(200)],
        "label": [0] * 200,
    })
    out1 = p3_train.augment_training_set(df, seed=42, noise_group=p3_train.NOISE_GROUP_A)
    out2 = p3_train.augment_training_set(df, seed=42, noise_group=p3_train.NOISE_GROUP_A)
    assert list(out1["text"]) == list(out2["text"]), "same seed must reproduce identical augmentation"

    n_changed = sum(1 for a, b in zip(df["text"], out1["text"]) if a != b)
    frac = n_changed / len(df)
    print(f"augment_training_set: {n_changed}/{len(df)} items changed (fraction={frac:.3f}, "
         f"target={p3_train.AUGMENT_FRACTION})")
    # Not an exact 50% (each item's inclusion is its own Bernoulli draw), but
    # with 200 items it must not be all-or-none -- the N7 bug's signature.
    assert 0.30 < frac < 0.70, (
        f"augmented fraction {frac:.3f} is far from the target "
        f"{p3_train.AUGMENT_FRACTION} -- looks like correlated (all-or-none) "
        "per-item draws, the N7 bug augment_training_set must avoid")

    out_seed2 = p3_train.augment_training_set(df, seed=1337, noise_group=p3_train.NOISE_GROUP_A)
    assert list(out1["text"]) != list(out_seed2["text"]), "different seed must select a different subset"


# ---------------------------------------------------------------------------
# compute_metrics -- M6.1 metric set + the PR-AUC positive-class convention
# ---------------------------------------------------------------------------


def test_compute_metrics_macro_f1_and_banfakenews_extras():
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]  # 0=fake (minority), 1=authentic
    y_pred = [0, 0, 1, 1, 1, 1, 1, 0]
    # scores must be P(fake) per compute_metrics' documented convention
    y_score_pos_fake = [0.9, 0.8, 0.4, 0.3, 0.2, 0.1, 0.2, 0.6]
    m = p3_train.compute_metrics(y_true, y_pred, "banfakenews", y_score_pos=y_score_pos_fake)
    print("compute_metrics(banfakenews):", {k: v for k, v in m.items() if k != "per_class_json"})
    assert set(["macro_f1", "weighted_f1", "balanced_accuracy", "accuracy",
               "pr_auc", "mcc", "per_class_json"]) <= set(m)
    assert 0.0 <= m["macro_f1"] <= 1.0
    assert m["pr_auc"] is not None and 0.0 <= m["pr_auc"] <= 1.0
    assert -1.0 <= m["mcc"] <= 1.0

    # Flipping which class the score favors should change PR-AUC -- proves
    # the function actually uses the passed-in "positive=fake" scores rather
    # than silently defaulting to sklearn's positive=1 convention.
    y_score_pos_authentic = [1 - s for s in y_score_pos_fake]
    m_flipped = p3_train.compute_metrics(y_true, y_pred, "banfakenews", y_score_pos=y_score_pos_authentic)
    print(f"pr_auc with correct (fake-positive) scores={m['pr_auc']:.4f}, "
         f"with flipped scores={m_flipped['pr_auc']:.4f}")
    assert m["pr_auc"] != pytest.approx(m_flipped["pr_auc"], abs=1e-9)


def test_compute_metrics_sentnob_has_no_banfakenews_extras():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 2, 2]
    m = p3_train.compute_metrics(y_true, y_pred, "sentnob")
    assert "pr_auc" not in m and "mcc" not in m
    print("compute_metrics(sentnob) macro_f1=", m["macro_f1"])


# ---------------------------------------------------------------------------
# Resumability bookkeeping -- CLAUDE.md invariant 6 / Phase 3 constraint 0
# ---------------------------------------------------------------------------


def test_results_key_and_job_grid_complete(tmp_path):
    job = p3_train.Job(model="banglabert", task="bd_shs", seed=42, train_variant="clean")
    existing: set[tuple] = set()
    assert not p3_train._job_grid_is_complete(job, existing)

    for nt in p3_train.NOISE_TYPES:
        for sev in range(1, 6):
            existing.add(p3_train._results_key(job.run_id, "on", nt, sev))
    assert p3_train._job_grid_is_complete(job, existing), (
        "bd_shs is not an R6 task -- the on-normalizer grid alone must be sufficient")

    fake_job = p3_train.Job(model="banglabert", task="banfakenews", seed=42, train_variant="clean")
    assert not p3_train._job_grid_is_complete(fake_job, existing), (
        "banfakenews IS an R6 task -- normalizer=on alone must not count as complete")


def test_load_existing_result_keys_roundtrip(tmp_path):
    job = p3_train.Job(model="char_ngram", task="sentnob", seed=42, train_variant="clean")
    row = p3_train.make_result_row(
        job, "on", "char_insert", 3,
        p3_train.compute_metrics([0, 1, 2], [0, 1, 1], "sentnob"),
        n_test_items=3, best_epoch=1, best_val_f1=0.5)
    results_path = str(tmp_path / "results.csv")
    p3_train.append_results_rows(results_path, [row])

    keys = p3_train.load_existing_result_keys(results_path)
    print("keys loaded from results.csv:", keys)
    assert p3_train._results_key(job.run_id, "on", "char_insert", 3) in keys


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
