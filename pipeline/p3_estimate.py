"""Phase 3 job-list + GPU-time estimator. `python pipeline/p3_train.py
--list-jobs` calls `estimate_all` here. Pure arithmetic over
`noisebench/models.py` + `pipeline/p3_train.py`'s job list + real row counts
read from `data/final/`; no training happens here.

**These are ESTIMATES, not measurements** -- there is no GPU in this dev
environment to benchmark against, so throughput is anchored to commonly
reported BERT-base fine-tuning speeds on a Kaggle T4 (fp16, batch=32,
seq_len=128): roughly 60-70 examples/sec train, ~3x that for inference-only.
`ASSUMED_TRAIN_THROUGHPUT`/`ASSUMED_INFER_THROUGHPUT` below are the only
numbers that matter here -- **the first real Kaggle run must have its actual
per-epoch `elapsed=` (already logged by `p3_train.train_transformer`) compared
against this estimate, and this file corrected**. Flagged again in
PHASE3_STATUS.md; do not treat the totals this file prints as a GPU-hour
budget guarantee before that calibration happens.

char_ngram (classical, CPU, `noisebench.models.MODELS["char_ngram"].kind ==
"classical"`) costs 0 GPU-hours by definition -- estimated here only as wall-
clock minutes, for session planning, not against the 30 GPU-h/week quota.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench.models import MODELS  # noqa: E402
from noisebench.perturbations import NOISE_TYPES  # noqa: E402
from pipeline import p3_train  # noqa: E402

# ---------------------------------------------------------------------------
# Assumptions -- see module docstring. Anchor: Kaggle T4, fp16, batch=32,
# seq_len=128, BERT-base-sized (12 layers/768 hidden) encoder.
# ---------------------------------------------------------------------------

ASSUMED_TRAIN_THROUGHPUT = 65.0   # examples/sec, forward+backward, any of the 4 transformers
ASSUMED_INFER_THROUGHPUT = 200.0  # examples/sec, forward only
ASSUMED_CLASSICAL_FIT_RATE = 4000.0  # train examples/sec, TfidfVectorizer(char_wb,2-5)+LinearSVC.fit, CPU
ASSUMED_CLASSICAL_INFER_RATE = 20000.0  # examples/sec, CPU inference
VAL_EVAL_OVERHEAD = 0.20  # extra fraction of train-set-equivalent time per epoch, for the val pass

#: Conservative: budget the full EPOCHS (early stopping usually saves some of
#: this, but a budget should not assume the saving).
EPOCHS_FOR_ESTIMATE = p3_train.EPOCHS

_ROW_COUNT_CACHE: dict[tuple[str, str], int] = {}


def _row_count(task: str, split: str, data_final_root: str) -> int:
    key = (task, split)
    if key not in _ROW_COUNT_CACHE:
        import pandas as pd
        spec = p3_train.DATA_SPECS[task]
        path = os.path.join(p3_train.dataset_final_dir(spec, data_final_root), f"{split}.csv")
        _ROW_COUNT_CACHE[key] = sum(1 for _ in open(path, encoding="utf-8")) - 1 if os.path.exists(path) else 0
    return _ROW_COUNT_CACHE[key]


@dataclass(frozen=True)
class JobEstimate:
    job: "p3_train.Job"
    train_seconds: float
    grid_seconds: float
    gpu_hours: float  # 0.0 for classical (char_ngram)

    @property
    def total_seconds(self) -> float:
        return self.train_seconds + self.grid_seconds


def estimate_job(job, data_final_root: str) -> JobEstimate:
    spec = MODELS[job.model]
    n_train = _row_count(job.task, "train", data_final_root)
    n_test = _row_count(job.task, "test", data_final_root)

    n_grid_cells = len(NOISE_TYPES) * 5  # 9 noise types x 5 severities
    if job.task in p3_train.R6_TASKS:
        n_grid_cells *= 2  # normalizer on/off, banfakenews only
    n_grid_predictions = n_grid_cells * n_test + n_test  # + the clean (severity-0) pass

    if spec.kind == "classical":
        train_seconds = n_train / ASSUMED_CLASSICAL_FIT_RATE  # single .fit(), not per-epoch
        grid_seconds = n_grid_predictions / ASSUMED_CLASSICAL_INFER_RATE
        return JobEstimate(job, train_seconds, grid_seconds, gpu_hours=0.0)

    per_epoch = (n_train / ASSUMED_TRAIN_THROUGHPUT) * (1 + VAL_EVAL_OVERHEAD)
    train_seconds = per_epoch * EPOCHS_FOR_ESTIMATE
    grid_seconds = n_grid_predictions / ASSUMED_INFER_THROUGHPUT
    total_seconds = train_seconds + grid_seconds
    return JobEstimate(job, train_seconds, grid_seconds, gpu_hours=total_seconds / 3600.0)


def estimate_all_jobs(data_final_root: str | None = None) -> list[JobEstimate]:
    data_final_root = data_final_root or p3_train._DEFAULT_DATA_FINAL
    return [estimate_job(j, data_final_root) for j in p3_train.all_jobs()]


# ---------------------------------------------------------------------------
# Session packing -- greedy bin-pack GPU jobs into <12h Kaggle sessions,
# respecting the 30 GPU-h/week quota. char_ngram jobs are packed separately
# (CPU wall-clock only, no GPU-hour cost) since they should run in a
# CPU-only Kaggle session and not occupy GPU quota at all.
# ---------------------------------------------------------------------------

SESSION_CAP_HOURS = 11.0    # budget under the 12h hard cap, not up to it
WEEKLY_GPU_CAP_HOURS = 30.0


def pack_sessions(estimates: list[JobEstimate]) -> list[list[JobEstimate]]:
    """Greedy bin-pack, largest-first, into `SESSION_CAP_HOURS` sessions."""
    gpu_jobs = sorted((e for e in estimates if e.gpu_hours > 0), key=lambda e: -e.gpu_hours)
    sessions: list[list[JobEstimate]] = []
    session_totals: list[float] = []
    for e in gpu_jobs:
        placed = False
        for i, total in enumerate(session_totals):
            if total + e.gpu_hours <= SESSION_CAP_HOURS:
                sessions[i].append(e)
                session_totals[i] += e.gpu_hours
                placed = True
                break
        if not placed:
            sessions.append([e])
            session_totals.append(e.gpu_hours)
    return sessions


def pack_weeks(sessions: list[list[JobEstimate]]) -> list[list[list[JobEstimate]]]:
    """Greedy-pack whole sessions into weeks under `WEEKLY_GPU_CAP_HOURS`."""
    weeks: list[list[list[JobEstimate]]] = []
    week_totals: list[float] = []
    for sess in sessions:
        sess_hours = sum(e.gpu_hours for e in sess)
        placed = False
        for i, total in enumerate(week_totals):
            if total + sess_hours <= WEEKLY_GPU_CAP_HOURS:
                weeks[i].append(sess)
                week_totals[i] += sess_hours
                placed = True
                break
        if not placed:
            weeks.append([sess])
            week_totals.append(sess_hours)
    return weeks


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def estimate_all(fh=sys.stdout, data_final_root: str | None = None) -> None:
    data_final_root = data_final_root or p3_train._DEFAULT_DATA_FINAL
    estimates = estimate_all_jobs(data_final_root)

    print("Phase 3 job list -- ESTIMATES, not measurements (see module docstring "
         "for the throughput assumptions; calibrate against the first real run's "
         "logged epoch elapsed= times).", file=fh)
    print(f"{'run_id':55s} {'kind':11s} {'train_s':>8s} {'grid_s':>8s} {'gpu_h':>7s}", file=fh)
    total_gpu_h = 0.0
    total_cpu_jobs = 0
    for e in estimates:
        kind = MODELS[e.job.model].kind
        print(f"{e.job.run_id:55s} {kind:11s} {e.train_seconds:8.0f} "
             f"{e.grid_seconds:8.0f} {e.gpu_hours:7.2f}", file=fh)
        total_gpu_h += e.gpu_hours
        if kind == "classical":
            total_cpu_jobs += 1

    n_gpu_jobs = len(estimates) - total_cpu_jobs
    print(f"\n{len(estimates)} jobs total: {n_gpu_jobs} GPU (transformer) + "
         f"{total_cpu_jobs} CPU (char_ngram, 0 GPU-h)", file=fh)
    print(f"Estimated total GPU-hours: {total_gpu_h:.1f} "
         f"(quota: {WEEKLY_GPU_CAP_HOURS:.0f}/week)", file=fh)

    sessions = pack_sessions(estimates)
    weeks = pack_weeks(sessions)
    print(f"\nProposed session plan: {len(sessions)} GPU sessions "
         f"(<= {SESSION_CAP_HOURS:.0f}h each), packed into {len(weeks)} week(s):", file=fh)
    for wi, week in enumerate(weeks, 1):
        week_hours = sum(e.gpu_hours for sess in week for e in sess)
        print(f"  Week {wi}: {len(week)} session(s), {week_hours:.1f} GPU-h total", file=fh)
        for si, sess in enumerate(week, 1):
            sess_hours = sum(e.gpu_hours for e in sess)
            print(f"    Session {si}: {len(sess)} jobs, {sess_hours:.1f} GPU-h", file=fh)
            for e in sess:
                print(f"      {e.job.run_id}", file=fh)
    print(f"\nchar_ngram jobs ({total_cpu_jobs}): run in a separate CPU-only Kaggle "
         "session, any order, no GPU quota impact.", file=fh)


if __name__ == "__main__":
    estimate_all()
