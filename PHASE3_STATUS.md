# Phase 3 status -- training + evaluation suite

**Everything below is written, smoke-tested, and nothing real has been run.**
No training job, noise-grid evaluation, or figure was produced from real
experiment data by this build pass -- only from tiny `--limit`-truncated
smoke runs (isolated under `.smoke/`, gitignored) and one synthetic figure
fixture. Real Kaggle runs are the next step, gated on the sign-off items in
[Section 3](#3-hyperparameters--confirmed-not-chosen) and
[Section 4](#4-design-decisions-made-without-prior-confirmation--please-sign-off).

## 0. What's built

| File | Purpose |
|---|---|
| `noisebench/models.py` | Model registry (5 models: banglabert, banglishbert, xlmr, mbert, char_ngram) -- one source of truth for the paper's model table |
| `noisebench/plotstyle.py` | Extended with the Phase 3 model-emphasis `STYLE` dict + `MODEL_SHORT`/`TASK_LABEL`/`NOISE_LABEL` (was Phase 2's dataset-only palette) |
| `pipeline/p3_train.py` | R1/R4 job list + `run_job()`: train -> clean eval -> noise grid -> append -> delete checkpoint |
| `pipeline/p3_evaluate_noise.py` | R2/R6 noise-grid evaluation, called in-memory from `run_job()` |
| `pipeline/p3_estimate.py` | Job-list GPU-hour estimator + session/week packer (`--list-jobs`) |
| `pipeline/p3_figures.py` | All 8 figure types, reads `results.csv`/`curves.csv` only, CPU |
| `tests/test_p3_train.py` | 8 tests, incl. the required perturb-before-truncate test |
| `notebooks/p3_01_train_clean.ipynb` | R1: 45 clean-training jobs |
| `notebooks/p3_02_noise_grid.ipynb` | Grid-completion/resume pass (see 4.6) |
| `notebooks/p3_03_train_augmented.ipynb` | R4: 36 augmented-training jobs |
| `notebooks/p3_04_analysis.ipynb` | Figures + the 6 pre-registered M6.2 McNemar tests |
| `requirements-phase3.txt` | transformers/scikit-learn/statsmodels/matplotlib; torch deliberately unpinned (see file) |

## 1. MuRIL -- proposed and rejected

A sixth model, MuRIL, was named in the original Phase 3 task description
without first checking METHODOLOGY.md M4 / Topic4_FULL_PAPER_PLAN.md §6,
both of which list exactly five models. Confirmed via `grep -in "muril"
docs/*.md CLAUDE.md` -> no matches anywhere. Raised to the user, who
confirmed it was their own error and gave the reason for the record:

> Redundant with XLM-R/mBERT on the multilingual axis; ~3-4 GPU-h not
> justified against the 30 Sep deadline.

The registry (`noisebench/models.py`) documents this rejection in its module
docstring. **No doc changes were made** -- METHODOLOGY.md and Topic4 already
correctly listed five models; only the erroneous task instruction was wrong.

## 2. Hyperparameters -- confirmed, not chosen

Same class of error as MuRIL: fallback hyperparameter values were proposed
without checking that M4 already fully specifies them. The user confirmed
**M4's values govern exactly**:

```
max_len=128, batch=32, lr=2e-5, epochs=4, AdamW, linear warmup 10%,
fp16 (CUDA only), early stopping on val macro-F1 (patience 2),
seeds = {42, 1337, 2024}
```

All in `pipeline/p3_train.py`'s `SEEDS`/`MAX_LEN`/`BATCH_SIZE`/`LR`/`EPOCHS`/
`WARMUP_RATIO`/`PATIENCE` constants -- uniform across every model and every
task (sentnob/bd_shs/banfakenews all use `max_len=128`, not a 128/512 split).

**Nothing here is a new choice needing sign-off** -- it's a transcription of
M4. Flagged anyway per the task instruction.

### 2.1 The `max_len=128` / BanFakeNews limitation (for §VI)

BanFakeNews articles average ~1,100 grapheme clusters (`cer_truncation_impact.csv`,
Phase 2); `max_len=128` truncates the model's view of most articles to
roughly the headline and lead paragraph -- **effectively headline-and-lead
classification, not full-document**. This is a real, defensible-but-must-be-
stated limitation. **Action needed: one sentence in the paper's Limitations
section (§VI)** -- not yet written (out of scope for this code pass).

**It does not break the noise study.** Perturbation (`perturb()`) is applied
to the FULL, untruncated article text; only the tokenizer's own
`truncation=True` shortens it afterward, inside `predict_transformer`/
`train_transformer`. So the noise density inside the visible 128-token window
matches the nominal severity regardless of how much of the article that
window covers -- the severity scale stays interpretable. This is asserted,
not just claimed: `tests/test_p3_train.py::
test_perturb_before_truncate_not_after_in_augmentation` and
`::test_perturb_before_truncate_not_after_in_noise_grid` monkeypatch
`perturb()` and assert it is always called with the item's full, untruncated
text (both for M5.2 training-time augmentation and for the R2/R6 noise
grid) -- both pass. A third test,
`test_prepare_text_for_tokenizer_never_truncates`, confirms the one
non-tokenizer text-prep step (`_prepare_text_for_tokenizer`, normalization)
never itself shortens text either.

## 3. Design decisions made without prior confirmation -- please sign off

These were necessary to make the spec's pieces fit together and are
implemented, but were not explicitly asked for. Flagging each for a yes/no.

### 3.1 Train + evaluate are ONE job, not two notebook passes

Topic4 §7 costs R2 as "~4-6 GPU-h, inference only" -- achievable ONLY if R2
reuses an already-loaded model. Constraint 0 forbids ever saving a
checkpoint past its own run (45 x ~440MB > 20GB cap). So `run_job()` does
train -> pick best-val epoch -> evaluate clean test -> full R2/R6 grid ->
append everything -> delete the model, all in one Python process, one GPU
tensor, never reloaded from disk. "Evaluate" in "train -> evaluate -> append
-> delete -> next" means this job's ENTIRE evaluation grid, not one point.

**Consequence for the notebooks:** `p3_01_train_clean.ipynb` (named for R1's
*clean training*) already produces that job's full noise grid too, since
that's what `run_job` does. `p3_02_noise_grid.ipynb` is therefore not an
independent second evaluation pass -- see 3.6.

### 3.2 A third `train_variant`, `augmented_n5n8`

The task literally said `train_variant ∈ {clean, augmented_n1n4}`. M5.2's
matched/mismatched 2x2 table needs training on BOTH noise groups (A and B) to
populate all four cells (A-train/A-eval, A-train/B-eval, B-train/A-eval,
B-train/B-eval), so `TRAIN_VARIANTS` was extended to
`{clean, augmented_n1n4, augmented_n5n8}`. `make_r4_jobs()` trains both.

### 3.3 McNemar prediction pooling (M6.2)

Saving per-item predictions for the full 45-cell grid x every model would be
needless disk use -- M6.2 only ever compares banglabert vs char_ngram, at
severity 3 and 5. `p3_evaluate_noise.run_grid_in_memory` writes pooled
per-item prediction files ONLY for `model in R4_MODELS` at
`severity in (3, 5)`, pooling all 9 noise types (and, at analysis time, all 3
seeds) into one file:
`results/preds/<run_id>__noise_s{severity}[__normoff].csv`.

**A real bug this caught during smoke testing:** the pooled file repeats
every test-set `id` once per noise type (9x). `p3_04_analysis.ipynb`'s first
draft merged BanglaBERT's and char_ngram's pooled predictions on `id` alone,
which silently cross-joined mismatched noise types (BanglaBERT's
`char_insert` prediction paired against char_ngram's `char_delete`
prediction for the "same" id) -- inflating the contingency table ~9x with
nonsense pairs. Caught by actually running the notebook's McNemar cell
against a tiny real (`--limit 15`) fixture before shipping it: row counts
were ~9x too large. Fixed by merging on `["id", "noise_type"]`. The 6
pre-registered tests pool across noise types AND seeds into one 2x2 table per
(task, severity) -- documented in the notebook cell as a deliberate
simplification (paired observations are not fully independent across seeds,
which makes the test conservative, not anti-conservative).

### 3.4 PR-AUC positive class = fake (label 0)

M6.1 requires PR-AUC for BanFakeNews but doesn't name the positive class.
Chosen: **fake**, the minority/target class a detector exists to catch, not
sklearn's default positive=1. Documented in `compute_metrics`'s docstring;
`run_job`/`run_grid_in_memory` convert `predict()`'s raw P(class=1) to
P(fake) = `1 - P(class=1)` before calling it, since label 0 = fake.

### 3.5 R4b matched/mismatched definition

Not spelled out operationally anywhere in the docs. Defined as: **matched**
= mean over {(train=augmented_n1n4, eval noise in group A), (train=
augmented_n5n8, eval noise in group B)}; **mismatched** = the two cross
pairs. The unaugmented (dashed) baseline in both panels is the same
clean-trained run, evaluated on the matched/mismatched panel's noise types.

### 3.6 What `p3_02_noise_grid.ipynb` actually is

Given 3.1, p3_02 cannot be an independent "run R2 over saved checkpoints"
pass -- nothing is saved. It is the **grid-completion/resume notebook**: if a
12-hour session died mid-grid, re-running `run_job` for that job retrains
(the only option) and then skips every cell already in `results.csv`,
computing only what's missing. For any job already fully done, its loop is a
free no-op. Its markdown cell says this explicitly so it isn't mistaken for
a second independent evaluation stage.

### 3.7 GPU-hour estimates are unmeasured assumptions

No GPU is available in this dev environment. `pipeline/p3_estimate.py`
anchors throughput to commonly reported BERT-base fine-tuning speed on a
Kaggle T4 (fp16, batch=32, seq_len=128): ~65 examples/sec train, ~200/sec
inference (`ASSUMED_TRAIN_THROUGHPUT`/`ASSUMED_INFER_THROUGHPUT`). **Compare
these against the real `elapsed=` values `train_transformer` logs to stderr
on the first real Kaggle run, and correct the constants** -- the session plan
in Section 5 is only as good as this calibration.

## 4. Perturb-before-truncate -- asserted, not assumed

Per the user's explicit follow-up request: this is enforced by
`_prepare_text_for_tokenizer`'s placement (only normalizes; truncation
happens only inside the tokenizer call, downstream of every perturbation
call site) and verified by the two tests named in Section 2.1. Both pass:

```
tests/test_p3_train.py::test_perturb_before_truncate_not_after_in_augmentation PASSED
tests/test_p3_train.py::test_perturb_before_truncate_not_after_in_noise_grid PASSED
tests/test_p3_train.py::test_prepare_text_for_tokenizer_never_truncates PASSED
```

## 5. Job list, GPU-hour estimate, and session plan

`python pipeline/p3_train.py --list-jobs` prints the full 81-job list (45 R1
+ 36 R4) with per-job estimates. Summary (see 3.7 -- these are estimates):

- **81 jobs total: 54 GPU (transformer) jobs + 27 CPU (char_ngram) jobs.**
- **Estimated total: 36.5 GPU-hours** across the 54 transformer jobs (quota:
  30 GPU-h/week -- this does not fit in one week).
- char_ngram's 27 jobs cost **0 GPU-hours** by construction (CPU-only, no
  torch/GPU code path) -- run them in a separate CPU-only Kaggle session,
  any order; they're fast enough (seconds-minutes each locally) that session
  packing for them isn't worth automating.

Proposed GPU session plan (`pack_sessions`/`pack_weeks`, greedy bin-pack,
11h budget under the 12h hard cap, 30 GPU-h/week cap):

| Week | Sessions | GPU-h |
|---|---|---|
| 1 | 3 sessions (~11h, ~10.9h, ~4.0h) | 25.7 |
| 2 | 1 session (~10.8h) | 10.8 |

4 GPU sessions total across 2 weeks, plus 1 (or more, split as convenient)
CPU-only session for the 27 char_ngram jobs. **Run `--list-jobs` again after
the first real session** to get a corrected plan once 3.7's calibration is
in.

## 6. Creating the two private Kaggle Datasets

**SentNoB is CC-BY-ND-4.0 -- the Dataset holding `data/final/` MUST be
private.** Do not make it public; ND forbids redistributing derivatives, and
`data/final/` is a derivative (cleaned/split) of SentNoB.

1. **`bangla-noisebench-data-final`** (private): zip `data/final/` (contains
   `sentnob/`, `bd_shs/`, `banfakenews/`, each with `train.csv`/`val.csv`/
   `test.csv`/`manifest.json`) and create a new Kaggle Dataset from it via
   the Kaggle web UI (New Dataset -> upload the zip) or `kaggle datasets
   create -p data/final --dir-mode zip` after `kaggle datasets init`. **Set
   visibility to Private** at creation (default is private; do not toggle
   it public).
2. **`bangla-noisebench-results`** (private -- holds per-item predictions,
   also a SentNoB-derived artifact via item ids/labels): start it EMPTY (an
   empty `results.csv`/`curves.csv`/empty `preds/` dir is fine, or omit the
   files entirely -- the notebooks' bootstrap cell handles a missing prior
   version gracefully) so the first notebook run has something to attach and
   later version. Private for the same CC-BY-ND-4.0 reason as above.
3. **`bangla-noisebench-code`** (private is simplest, public is fine too --
   no data-license issue here): zip this repository (or connect it as a
   Kaggle "Notebook output" / GitHub-linked Dataset, whichever this account
   supports) so `pipeline`/`noisebench` are importable inside the notebook
   at `/kaggle/input/bangla-noisebench-code/bangla-noisebench`.

Attach all three to each notebook via **Add Data** before running. If your
slugs differ from the three assumed in `notebooks/*.ipynb`'s bootstrap cell
(`DATA_DATASET_SLUG`/`RESULTS_DATASET_SLUG`/`CODE_DATASET_SLUG`), edit those
three lines.

The re-upload cell (last cell, every notebook) calls `kaggle datasets
version` against `bangla-noisebench-results`, which requires a `kaggle.json`
API token -- present by default in a Kaggle notebook environment with
"Internet" enabled. It writes a placeholder `dataset-metadata.json` with
`YOUR_KAGGLE_USERNAME` the first time it runs if one doesn't already exist;
edit that once, or run `kaggle datasets init` by hand first.

## 7. Running order

1. `p3_01_train_clean.ipynb` -- R1, 45 jobs. Slice via `JOB_SLICE` to fit a
   session (Section 5's plan; note R1 and R4 jobs interleave in the packed
   sessions above, so a literal `JOB_SLICE` over `make_r1_jobs()` alone won't
   match the table exactly -- use it as a budget guide, not a literal index
   range, or extend `p3_estimate.py` to emit index ranges if that's wanted).
2. `p3_02_noise_grid.ipynb` -- run after ANY session (p3_01 or p3_03) that
   hit the 12h cutoff, with `JOB_FAMILY` set to whichever family was
   interrupted. Free no-op for already-finished jobs.
3. `p3_03_train_augmented.ipynb` -- R4, 36 jobs, after R1 is far enough along
   that the clean baselines exist for comparison (not a hard code
   dependency, just makes the R4b figure meaningful sooner).
4. `p3_04_analysis.ipynb` -- any time, CPU-only, cheap. Re-run freely as more
   results land; every figure degrades gracefully (prints "SKIPPED -- ..."
   to stderr) if its required rows don't exist yet rather than crashing.

## 8. Smoke testing performed (nothing else has been run)

All under `.smoke/` (gitignored), deleted after each check:

- `pipeline/p3_train.py`: classical (`char_ngram`) full job end-to-end
  (train -> clean eval -> 45-cell grid -> resumability skip verified);
  transformer (`banglabert`) full job end-to-end on CPU (train with early
  stopping -> clean eval -> 45-cell grid); R6 doubling verified on
  `banfakenews` (90-cell grid, both normalizer on/off, pooled McNemar preds
  files for both).
- `pipeline/p3_figures.py --limit N`: writes a synthetic (clearly-labeled,
  never-real) fixture and renders all 8 figure types successfully; spot
  checked `fig1_degradation.png` and `fig_heatmap_sev3.png` and
  `fig5_r4b_augmentation.png` visually against Figure_Plan_Evaluation.md's
  spec (emphasis colours, greyscale-safe line styles/markers, annotated
  heatmap cells, shaded matched/mismatched recovery gap) -- all match.
- `pipeline/p3_estimate.py` / `--list-jobs`: runs against real `data/final/`
  row counts, produces the Section 5 table.
- Notebooks: not executable end-to-end here (no Kaggle environment, no real
  GPU). Verified instead: all 4 are valid nbformat-4 JSON; every code cell
  compiles (`py_compile`-equivalent check across all cells, including
  multi-line `%pip` magics); the bootstrap + parameters + job-count cells of
  `p3_01` were actually EXECUTED against the real local repo (non-Kaggle
  branch) and produced correct output; `p3_04`'s McNemar cell was actually
  EXECUTED against real (if tiny, `--limit 15`) pooled prediction files
  produced by a real `p3_01`-equivalent smoke run for both `banglabert` and
  `char_ngram` on `bd_shs`, seeds 42/1337/2024 -- this is what caught the
  `id`-only-merge bug in 3.3.
- `tests/test_p3_train.py`: 8 new tests, all passing with real printed
  numbers (not just "N passed") per CLAUDE.md's Definition of Done. Full
  suite (`tests/`): 77 passed.

## 9. Open items / uncertainties

- `noisebench/models.py`'s `pretraining_corpus` field for `banglabert` and
  `banglishbert` carries `TODO(verify)` comments -- the exact corpus
  size/composition figure needs checking against the BanglaBERT paper/model
  card before it goes in the paper's model table. xlmr/mbert's corpus
  descriptions are stated confidently (well-known public facts, Conneau et
  al. 2020 / Devlin et al. 2019).
- Section 2.1's one-sentence Limitations note is not yet written into any
  paper draft -- code confirms the claim, the sentence itself is a writing
  task, not a code task.
- Section 3.7's throughput constants are unmeasured; the whole session plan
  in Section 5 should be re-derived after the first real session.
- `p3_figures.fig_cer_vs_drop` reads Phase 1's per-noise-type mean CER table
  (`outputs/s2_preparation/tables/m2_4_cer.csv`, corpus-level, not
  per-dataset) and compares it against Phase 3's per-model accuracy drop
  averaged across all models/tasks -- reasonable given CER is a property of
  the perturbation function, not the classifier, but flagging the join
  in case a task-specific CER re-measurement is later preferred.
