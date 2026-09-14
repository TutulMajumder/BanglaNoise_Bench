"""Phase 3 R2/R6: the noise-grid evaluation that runs INSIDE `p3_train.py`'s
`run_job()`, reusing its already-trained, already-in-memory model -- see that
file's module docstring for why this must happen in the same process rather
than as a second notebook that reloads a checkpoint (no checkpoint is ever
saved).

Grid: 9 noise types (N1-N9, `noisebench.perturbations.NOISE_TYPES` -- N10 is
excluded per METHODOLOGY M5.1) x 5 severities x this job's one (model, task,
seed). Severity 0 (clean) is NOT here -- `p3_train.run_job` evaluates it
itself before this module is even called, since it needs no perturbation.

R6 (METHODOLOGY M5.4): BanFakeNews only doubles this grid with
normalizer="off" alongside the default normalizer="on" -- sentnob/bd_shs are
dropped (their measured normalizer touch_rate is 1.67%, see p3_train.R6_TASKS)
and simply never get an "off" row; `p3_figures.py` reports that as the null
result the docs call for, not a missing cell.

Perturbation always happens on the FULL test text, before
`p3_train._prepare_text_for_tokenizer` (which only normalizes/hands off to the
tokenizer's own truncation) ever sees it -- this is what keeps the severity
scale meaningful on BanFakeNews's max_len=128 window even though that window
only covers the article's headline and lead; see
tests/test_p3_train.py::test_perturb_before_truncate_not_after.

Standalone use (resuming a job whose grid was left incomplete mid-Kaggle-
session): this module has no CLI of its own on purpose -- resuming means
re-running the SAME `p3_train.py` job command. `run_job` retrains (there is
nothing else to reuse; the checkpoint was deleted) and then calls back in
here, which skips every grid cell already in results.csv via `existing_keys`
and only computes what's missing. Retraining a few extra seconds/minutes to
resume a few missing grid cells is the accepted cost of never persisting a
checkpoint (constraint 0).

McNemar prediction pooling (M6.2: BanglaBERT vs char_ngram, severity 3 and 5,
per task -- 6 pre-registered tests total, Holm-corrected): saving per-item
predictions for the full 45-cell grid x every model would blow past
reasonable disk use for no purpose (only these two models at these two
severities are ever compared). So this module writes per-item predictions
ONLY for `job.model in P3_TRAIN.R4_MODELS` (banglabert, char_ngram) at
severity in `MCNEMAR_SEVERITIES` (3, 5), pooling all 9 noise types into one
file per (run_id, severity) -- `results/preds/<run_id>__noise_s{severity}.csv`
-- since M6.2 compares the two models' correctness on the same noised test
items, not noise-type by noise-type.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench.perturbations import NOISE_TYPES, perturb  # noqa: E402
from pipeline import p3_train  # noqa: E402
from pipeline._common import Progress  # noqa: E402

SEVERITIES: tuple[int, ...] = (1, 2, 3, 4, 5)

#: M6.2's pre-registered severities -- pooled per-item predictions are only
#: written at these two, and only for the two R4_MODELS.
MCNEMAR_SEVERITIES: tuple[int, ...] = (3, 5)


def _positive_scores_for(job_task: str, scores: list[float]) -> list[float] | None:
    """Same P(class=1) -> P(fake) conversion as `p3_train.run_job`'s clean
    pass (banfakenews label 0 = fake; `predict` always returns P(class=1))."""
    if not scores:
        return None
    return [1 - s for s in scores]


def run_grid_in_memory(job, state: dict, test_df, results_path: str, preds_dir: str,
                       progress: Progress, existing_keys: set[tuple], force: bool = False) -> dict:
    """The R2/R6 grid for one job, called from `p3_train.run_job` right after
    training, before the model is deleted. Appends rows to `results.csv` as
    they're computed (not batched to the end) so a session killed mid-grid
    loses at most the in-flight cell -- `existing_keys` was read once by the
    caller before this started, so this function's own writes this run are
    never re-checked against themselves (fine: every (run_id, normalizer,
    noise_type, severity) triple below is visited at most once)."""
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].tolist()
    test_ids = test_df["id"].tolist()

    normalizers = ("on", "off") if job.task in p3_train.R6_TASKS else ("on",)
    pool_preds = job.model in p3_train.R4_MODELS

    n_total = len(NOISE_TYPES) * len(SEVERITIES) * len(normalizers)
    n_done = 0
    n_skipped = 0

    # normalizer -> severity -> {"ids": [...], "true": [...], "pred": [...], "noise_type": [...]}
    pooled: dict[str, dict[int, dict[str, list]]] = {
        norm: {sev: {"ids": [], "true": [], "pred": [], "noise_type": []} for sev in MCNEMAR_SEVERITIES}
        for norm in normalizers
    }

    for normalizer in normalizers:
        for noise_type in NOISE_TYPES:
            for severity in SEVERITIES:
                key = p3_train._results_key(job.run_id, normalizer, noise_type, severity)
                if key in existing_keys and not force:
                    n_skipped += 1
                    continue

                # Perturb the FULL text first, THEN hand to predict() (which
                # normalizes/truncates) -- see module docstring.
                noised_texts = [perturb(t, noise_type, severity, job.seed) for t in test_texts]
                preds, scores = p3_train.predict(state, noised_texts, normalizer)
                pos_scores = _positive_scores_for(job.task, scores)
                metrics = p3_train.compute_metrics(test_labels, preds, job.task, y_score_pos=pos_scores)

                row = p3_train.make_result_row(
                    job, normalizer, noise_type, severity, metrics, len(test_labels),
                    state["best_epoch"], state["best_val_macro_f1"])
                p3_train.append_results_rows(results_path, [row])
                existing_keys.add(key)
                n_done += 1

                if pool_preds and severity in MCNEMAR_SEVERITIES:
                    bucket = pooled[normalizer][severity]
                    bucket["ids"].extend(test_ids)
                    bucket["true"].extend(test_labels)
                    bucket["pred"].extend(preds)
                    bucket["noise_type"].extend([noise_type] * len(test_ids))

                if n_done % 15 == 0 or n_done == n_total - n_skipped:
                    progress.log(job.run_id,
                                f"noise grid {n_done + n_skipped}/{n_total} cells "
                                f"({normalizer}, {noise_type}, sev={severity}) "
                                f"macro_f1={metrics['macro_f1']:.4f}")

    if pool_preds:
        for normalizer in normalizers:
            for severity in MCNEMAR_SEVERITIES:
                bucket = pooled[normalizer][severity]
                if not bucket["ids"]:
                    continue  # every cell already existed from a prior partial session
                suffix = "" if normalizer == "on" else "__normoff"
                path = os.path.join(preds_dir, f"{job.run_id}__noise_s{severity}{suffix}.csv")
                p3_train.write_preds_file(
                    path, bucket["ids"], bucket["true"], bucket["pred"],
                    extra_cols={"noise_type": bucket["noise_type"]})

    progress.log(job.run_id, f"noise grid done: {n_done} computed, {n_skipped} already in results.csv")
    return {"n_computed": n_done, "n_skipped": n_skipped, "n_total": n_total}
