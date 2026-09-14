# bangla-noisebench — Phase 2 status report

Generated 2026-09-15 (see §J for today's changes; §0 through §I are the
2026-09-14 restructure report, kept for the record). The real, full-scale
run is now authorized and executed — see §K for the actual numbers.

---

## J. 2026-09-15 — scope cuts + a real bug fix, before the real run

### J.1 `banfakenews_full` cut entirely

**Removed** from the registry (`pipeline/s2_prepare.py::DATASETS`), from
`noisebench/plotstyle.py::DATASET_STYLE`, from the test fixtures
(`tests/conftest.py`), and from every table/figure/report — all of s1/s3/s4/s5
iterate `DATASETS` generically, so removing the registry entry cascades
through the whole pipeline with no per-stage code left referencing it (every
stray docstring/comment mention was updated too; grepped to confirm).

**Reason, for the record:** the full 48K/1K imbalanced pair was implemented
as a secondary condition supporting no claim in the paper, and it consumed
**72% of s1's real-scale runtime (755s of 1042s)**. Three datasets remain:
**sentnob, bd_shs, banfakenews.** Also marked in
`docs/Topic4_FULL_PAPER_PLAN.md` (§5 dataset table, §11 cut list) and
`docs/METHODOLOGY.md` (M2.2).

Nothing existed yet at `data/final/banfakenews/full/` (s2 had never
completed a real run), so there was nothing to delete there.

### J.2 s1's BanFakeNews label bug — fixed with a shared function

**Bug:** s1 reported BanFakeNews as `{'1': 8501}` — every row "authentic."
Correct (and what s2 always got right): 7,202 authentic / 1,299 fake.

**Root cause:** s1 re-derived which `SourceFile` (`src`) a file came from via
`next(s for s in spec.sources if (s.split or file_stem) == split_label)`.
For BanFakeNews, both sources have `split=None`, so the condition
`(None or file_stem) == split_label` is trivially True for *every* source
being tested — `next()` therefore always returned the *first* source in
`spec.sources` (`LabeledAuthentic-7K`, `label_const=1`), regardless of which
file was actually being processed. Every BanFakeNews row got `label_const=1`.

**Fix:** added `resolve_labels(df, spec, src, path)` to
`pipeline/s2_prepare.py` (the registry module) — the one place file-level
(`label_const`) vs. column-mapped (`label_col`/`label_map`) labels are
resolved. `s2_prepare.py::load_raw` and `s1_inspect_raw.py::inspect_dataset`
both call it now. s1's `_resolve_sources` also now returns the matched `src`
object directly instead of re-deriving it — the buggy lookup is gone, not
patched.

**Regression test:** `tests/test_pipeline.py::test_s1_s2_agree_on_raw_class_distribution`
— for every registered dataset, asserts s1's raw class distribution equals
s2's (via `load_raw`) on the same fixture input, plus a BanFakeNews-specific
check against the fixture's deliberately asymmetric 32/16 split (a label
swap changes the *count*, not just the name, so this can't pass by
accident). Verified against real data too: `s1_inspect_raw.py --limit 50`
now reports BanFakeNews as `{'1': 50, '0': 50}` (50 rows from each of the
two source files), not all-authentic.

### J.3 N10 (emoji) cut from evaluation — decided, not contingent

- N10 **stays defined** in the taxonomy and `docs/METHODOLOGY.md` (M0/M1.4)
  — a real phenomenon, unchanged.
- N10 **excluded from the R2 grid** (`docs/METHODOLOGY.md` M5.1;
  `docs/Topic4_FULL_PAPER_PLAN.md` §7) — the CER tables already only ever
  reported 9 noise types, so this makes the documented grid and the actual
  code/tables agree explicitly rather than incidentally. No code change was
  needed here (N10 was never in the CER grid); this was a documentation gap.
- **R5 (emoji training arm) dropped** — 18 fine-tunes, ~4 GPU-hours freed.
  Marked cut in `docs/Topic4_FULL_PAPER_PLAN.md` §7 (R-matrix) and §11 (cut
  list, moved from contingent to decided) and `docs/METHODOLOGY.md` M5.3,
  with the reason and numbers in both places.
- **Reason:** s1 raw inspection counts 266 emoji-bearing texts in SentNoB
  (1.7%) and 288 in BD-SHS (0.57%) — roughly 27 and 57 items on the test
  splits alone. Not enough to support an emoji-noise evaluation regardless
  of GPU budget. Limitations-statement text (for §VI, not written by this
  pipeline — that's the paper's, not code's, to say):

  > *"Available Bangla classification corpora carry too little emoji
  > content to support an emoji-noise evaluation — itself a finding about
  > the state of Bangla datasets."*

### J.4 Note for you (per "write it as a note and move on")

`resolve_labels`'s fix changes behavior for any *future* dataset added to
the registry whose label column has genuinely unmappable values: it now
hard-exits (matching s2's long-standing behavior) instead of s1's old
silent `f"UNMAPPED:{v}"` tag. This is intentional (the whole point is that
s1 and s2 can't diverge again) but is a stricter failure mode than s1 had
before — flagging in case a future raw-file schema change surfaces this
where the old code would have limped along with a wrong-but-non-fatal label.

No other architectural changes made or considered beyond what was asked.

---

## K. The real run — executed 2026-09-15

All five stages, `--force`, in order, no `--limit`. **Total wall time 504.1s
(~8.4 min)** — under the 12–18 min estimate, no stage near the 10-minute
invariant-9 threshold.

| stage | wall time |
|---|--:|
| s1 raw inspection | 188.2s |
| s2 prepare | 16.8s |
| s3 describe | 202.4s |
| s4 measure | 92.0s (cer=81.3s, n5=2.2s, emoji=3.5s, normalizer=1.1s, n10=3.6s) |
| s5 figures | 4.7s |

### Headline numbers

| dataset | raw | final | dupes | short | zero-Bangla | class distribution (final) |
|---|--:|--:|--:|--:|--:|---|
| sentnob | 15,728 | 14,353 | 1,369 | 0 | 6 | neutral=3,361, positive=5,921, negative=5,071 |
| bd_shs | 50,281 | 50,127 | 18 | 133 | 3 | not_hate=25,999, hate=24,128 |
| banfakenews | 8,501 | 8,489 | 10 | 0 | 2 | authentic=7,194, fake=1,295 |

BanFakeNews's 7,194/1,295 (vs. the pre-fix bug's 8,489/0) is the direct
confirmation the J.2 fix holds at real scale, not just on the fixture.

**SentNoB leakage** (raw-text → final-text, both now in `manifest.json`):

| pair | raw | final | % of split (final) |
|---|--:|--:|--:|
| test_in_train | 145 | 361 | 23.01% of 1,569 |
| val_in_train | 142 | 351 | 22.54% of 1,557 |
| val_in_test | 16 | 54 | 3.47% of 1,557 |

**BD-SHS U+FFFD × hate-speech label (2×2):** flagged (n=2,545) hate=38.11%/
not_hate=61.89%; unflagged (n=47,582) hate=48.67%/not_hate=51.33% — same
material gap as before, now on the corrected 3-dataset run.

**N10, full test splits (exact counts, not sampled):**

| dataset | n test | emoji-bearing | % |
|---|--:|--:|--:|
| sentnob | 1,569 | 29 | 1.85% |
| bd_shs | 10,025 | 49 | 0.49% |
| banfakenews | 1,698 | 5 | 0.29% |

(s1's dataset-wide counts, cited in §J.3, were 266/288 across the whole
corpus, not just test — both consistent with "not enough," which is the
point; test-split counts are what actually matters for R5 and both are
even smaller.)

**R6, per dataset (M5.4, 5% threshold), `banfakenews_full` no longer in the pool:**

| dataset | touch_rate | verdict |
|---|--:|---|
| sentnob | 1.67% | DROP |
| bd_shs | 1.67% | DROP |
| banfakenews | 54.00% | KEEP |

### Publication hygiene — verified on the real run

`python pipeline/verify_hygiene.py` → exit 1 (as designed — unexplained
codepoints always require a look, never silently pass):

```
Allowlisted: 2,356 (outputs/s0_noise_suite/, unchanged)
UNEXPLAINED: 124 (outputs/s2_preparation/m2_4_summary.md: 64,
                   outputs/s2_preparation/tables/m2_4_n5.csv: 60)
```

Enumerated the 124 directly (not assumed): all of them are exactly the 8
homophone-set glyph labels (শ/ষ/স, ন/ণ, ই/ঈ, উ/ঊ, ি/ী, ু/ূ, র/ড়/ঢ়, জ/য),
3–4 occurrences each across the 3 datasets. No corpus text. Confirmed clean.

### Tests

`python -m pytest -q` → **69 passed**, including
`test_s1_s2_agree_on_raw_class_distribution` against the real fix.

No git commands were run.

---

## 0. Answers to 1–3 (report before the real run — no s1/downstream run yet)

### 1. `constants_sheet.*` Bangla-glyph exemption

**Before this turn, the check was not built at all** — the one-liner I put
in §D previously was never run against the real `outputs/` tree (only
against scratch smoke output, which doesn't include `s0_noise_suite/`). Run
for real just now, it correctly **flagged** 2,356 Bengali-block codepoints
across `constants_sheet.{md,html}` and `phase1_report.{md,html}` — it did
not "pass for the wrong reason," it simply hadn't been run. So: neither of
your two options — I built the allowlist properly instead of doing either.

New file `pipeline/verify_hygiene.py`: scans `outputs/**/*.{md,html,csv,json}`,
counts U+0980–U+09FF codepoints per file, and separates hits into two
buckets — **allowlisted** (path-prefix, each with a written reason in the
module docstring and inline) and **unexplained** (must be investigated).
`outputs/s0_noise_suite/` is the one allowlisted entry, with the reason
being the empirical proof in §0.2 below, not just an assertion. Verified
just now against the real tree:

```
Allowlisted (expected) Bengali-block codepoints:
     460  s0_noise_suite/constants_sheet.html
     460  s0_noise_suite/constants_sheet.md
     718  s0_noise_suite/phase1_report.html
     718  s0_noise_suite/phase1_report.md

UNEXPLAINED Bengali-block codepoints (investigate before committing):
  (none -- clean)

Total allowlisted: 2356   Total UNEXPLAINED: 0
```

Exit code 0 (0 unexplained). This replaces the inline one-liner in the old
§D step 3 — see the updated §D below. Note deliberately NOT allowlisted:
the N5 homophone-set glyph labels that will appear in
`outputs/s2_preparation/` once you run s4 for real (~80–164 codepoints in
the prior session's smoke data). Those stay in the "unexplained" bucket on
purpose — small, expected, but worth eyeballing every run rather than
silently filtered, per your framing ("not by weakening the check").

### 2. `phase1_report.*` fixture sentences — verified, not assumed

Structural check: `scripts/phase1_report.py::_make_corpus()` samples only
from `tests.bangla_samples.SENTENCES` (58 sentences) — it has no code path
that reads `data/raw/`.

Empirical check (the one that matters, since wording could coincidentally
match): every one of the 58 fixture sentences checked for **exact substring
containment** against **every raw row in all 10 CSVs** (124,487 rows total
across sentnob, bd_shs, banfakenews, banfakenews_full).

**Result: 0 matches.** None of the 58 fixture sentences appear verbatim
anywhere in any of the three real corpora. Confirmed empirically, per your
instruction — not assumed from the file's own docstring claim.

### 3. The ✈ row, against `data/raw/`, and the CSV-vs-JSONL question

Found in `data/raw/bd_shs/train.csv`, row index 2 (0-indexed, header excluded):

```
'✈✈✈✈��� মুরগি চোরের পাছায় ডুকবি আর মারবি।।।'
```

Full content as requested. **Correction to the recollection**: this row is
not Bangla-free — it has **27 Bengali-block characters** after the
emoji/replacement-character prefix ("মুরগি চোরের পাছায় ডুকবি আর মারবি।।।" —
chicken-thief-themed abuse, consistent with bd_shs's content). No `\n` or
`\r` anywhere in the field. (There's a second, unrelated ✈-bearing row at
index 2823, also newline-free.)

**Embedded newline/CR census, all 10 raw files:**

| dataset | raw rows | rows with `\n` | rows with `\r` |
|---|--:|--:|--:|
| sentnob | 15,728 | 0 (0.000%) | 0 |
| bd_shs | 50,281 | **23 (0.046%)** | 0 |
| banfakenews | 8,501 | 0 (0.000%) | 0 |
| banfakenews_full | 49,977 | 0 (0.000%) | 0 |

Only bd_shs has any (23 rows, all `train.csv`/`val.csv`/`test.csv`, e.g.
`'সাব্বির কে গুলি করে হত্যা করা উচিত।...\n'` — a trailing newline, not an
embedded mid-field one).

**But this doesn't gate the output format, because it doesn't reach
`data/final/`.** `preprocess_text`'s step 3 (`_WS_RE.sub(" ", s).strip()`,
`_WS_RE = re.compile(r"\s+")`) collapses every whitespace run — `\n` and
`\r` included, since Python's `\s` matches them — into a single space,
*before* the text is ever written out. I ran `preprocess_text` on all 23
raw newline-bearing rows directly: **0 of 23 retain a `\n` or `\r` after
cleaning.** So every row that reaches `data/final/*.csv` is guaranteed
newline-free by construction, not by a format choice made now.

**Recommendation: keep CSV**, not JSONL, for `data/final/`. The concern
("naive readers with pandas defaults") doesn't apply here specifically
*because* the field that would need multi-line quoting never survives
M2.1 — this is a proof about the existing pipeline behavior, not a new
design decision. If you'd still rather have JSONL for other reasons
(schema evolution, streaming, avoiding CSV's quoting rules generally even
though they're not exercised here), say so and I'll switch it — but the
newline-safety argument specifically doesn't require it.

---

## A. Stale-state audit (done first, as asked)

Checked actual file timestamps and content, not memory of the session.

| path | status | evidence |
|---|---|---|
| `data/clean/*.csv`, `*.prep.json` | **stale**, now deleted | `sentnob.prep.json`'s `class_dist` still had `"neutral?"` etc. — the pre-fix labels — even though the registry had already been corrected to drop the `?` |
| `data/splits/*.txt` | same run, same staleness, now deleted | — |
| `results/table2.md/.html` | **stale**, now deleted | inherited the `neutral?` labels from the stale prep.json |
| `results/m2_4.md/.html/.json` | **stale**, now deleted | contained 25 occurrences of the buggy `has-digit` kinds tag and the old pooled `Overall touch_rate: 27.92%` framing — both fixed in code *after* this file was written |
| `data/prepare.py`, `scripts/table2_report.py` | already gone | confirmed deleted in a prior turn |
| `data/__pycache__/`, `scripts/__pycache__/` | dead bytecode | gitignored, removed anyway |
| `outputs/`, `pipeline/`, `docs/` | didn't exist | nothing to audit |

Everything else in the (now-deleted) `data/clean/` was **not** stale — bd_shs's
`n_raw=50281` confirmed the 3-file fix was already applied, and
`non_bangla_removed` was present and correct for all 4 datasets. Only the
label-string cosmetics and the two m2_4 reporting bugs had drifted.

**Deletion list — approved, with one amendment:** you chose to move Phase 1's
`results/constants_sheet.*` and `results/phase1_report.*` into
`outputs/s0_noise_suite/` rather than delete them (they were current, not
superseded by anything). Everything else in the original list was deleted as
proposed.

---

## B. What changed — restructure summary

- `noisebench/` gained `corpus_stats.py` (shared s1/s3 measurement functions,
  stdlib+regex only) and `plotstyle.py` (palette/IEEE-width constants, pure
  data, no imports).
- All runnable code moved to `pipeline/`: `s1_inspect_raw.py`,
  `s2_prepare.py`, `s3_describe.py`, `s4_measure.py`, `s5_figures.py`,
  `run_all.py`, `_common.py`. The old `scripts/prepare_data.py`,
  `make_table2.py`, `m2_4_measurements.py`, `stage_a_inspect.py`, `_common.py`
  are deleted. `scripts/` now holds only the two Phase 1 (noise-suite
  validation) scripts, which predate Phase 2 and aren't part of this
  pipeline — `constants_sheet.py`, `phase1_report.py`.
- `data/clean/` + `data/splits/` → `data/final/<dataset>/{train,val,test}.csv`
  + `manifest.json`. `banfakenews_full` now lives at
  `data/final/banfakenews/full/` (a condition on banfakenews, not a fifth
  corpus) — matches your target tree exactly.
- `METHODOLOGY.md`, `Topic4_FULL_PAPER_PLAN.md`, `Figure_Plan_Evaluation.md`
  moved to `docs/`. `CLAUDE.md` references updated.
- `.gitignore` rewritten and verified with real `git check-ignore -v` calls
  (§F).
- `tests/test_phase2_pipeline.py` → `tests/test_pipeline.py`, rewritten for
  the new APIs. Added `tests/test_corpus_stats.py` (11 tests) in the prior
  turn. **56 tests pass.**

---

## C. Final tree

```
bangla-noisebench/
  CLAUDE.md                         invariants 9 (no long-running analysis)
                                     and 10 (publication hygiene) added
  PHASE1_STATUS.md  PHASE2_STATUS.md  README.md
  requirements.txt  requirements-phase2.txt

  noisebench/                       library: pure, no I/O, no scripts
    __init__.py  bangla_maps.py  perturbations.py  severity.py
    validate.py  corpus_stats.py  plotstyle.py

  pipeline/                         the runnable pipeline, in execution order
    _common.py         Progress + --limit/--force plumbing (invariant 9)
    s1_inspect_raw.py  raw inspection (read-only on data/raw/)
    s2_prepare.py       M2.1 + M2.2 -> data/final/
    s3_describe.py      Table II (M2.3) + raw->final delta + eligible units
    s4_measure.py        M2.4 (CER/N5/emoji/normalizer) + N10 report
    s5_figures.py        renders every figure from tables/*.csv only
    run_all.py            orchestrator, --stage {1,2,3,4,5,all}

  data/
    README.md            tracked
    raw/                 gitignored — you already have this populated
    final/                gitignored except manifest.json (see §F)
      sentnob/  bd_shs/  banfakenews/{,/full}/
        train.csv val.csv test.csv manifest.json

  outputs/                          tracked — paper material
    s0_noise_suite/       Phase 1 (moved from results/, your call)
    s1_raw_inspection/     tables/  figures/  report.md
    s2_preparation/        tables/  figures/  report.md  provenance.json
                            (+ M2.4's tables/figures — see §H design note)
    s3_analysis/            tables/  figures/  report.md

  scripts/                          Phase 1 only, predates the pipeline
    constants_sheet.py  phase1_report.py

  tests/
    test_perturbations.py  test_corpus_stats.py  test_pipeline.py
    bangla_samples.py  __init__.py

  docs/
    METHODOLOGY.md  Topic4_FULL_PAPER_PLAN.md  Figure_Plan_Evaluation.md

  results/                          Phase 5+ only now (results.csv), empty
```

---

## D. Exact commands, in order, with wall time

**Run s1 first and look at `outputs/s1_raw_inspection/report.md` before
running s2** — that's the whole point of separating them (per your
instruction). Nothing below has been run at this scale by me; wall times for
s1/s3 are **estimates** (see the uncertainty flagged in §H.1) — s4's are
**measured**, carried over from real timing data this session.

```bash
# 1. Smoke-test everything first (already done by me, but you can re-verify):
python pipeline/run_all.py --stage all --limit 20 --data-root data/raw

# 2. Real run, stage by stage:
python pipeline/s1_inspect_raw.py --data-root data/raw
#    -> look at outputs/s1_raw_inspection/report.md and tables/ before continuing
#    est. 3-10 min (UNVERIFIED at full scale -- see §H.1; watch stderr, each
#    dataset logs progress every file/split; if one dataset alone exceeds
#    ~10 min, stop and tell me rather than waiting, per invariant 9)

python pipeline/s2_prepare.py --data-root data/raw
#    est. 1-2 min (measured historically at this data volume, pre-restructure)

python pipeline/s3_describe.py
#    est. 5-10 min (UNVERIFIED at full scale -- same character-classification
#    cost as s1, plus eligible-units sampling; see §H.1)

python pipeline/s4_measure.py
#    est. 3-6 min -- MEASURED: CER 132-316s (rapidfuzz-backed, verified 0
#    mismatches vs pure-Python on the full Phase 1 table), N5 ~3-5s,
#    emoji ~4-7s, normalizer ~2s, N10 ~0s. Watch [s4_measure] stderr; if CER
#    exceeds ~10 min even now, stop and tell me -- do not let it run.

python pipeline/s5_figures.py
#    est. <1 min

# 3. Publication-hygiene check (run after step 2 finishes):
python pipeline/verify_hygiene.py
#    Exit code 0 = clean. Expect the s0_noise_suite/ allowlisted total
#    (2,356, already verified this session) PLUS a small "UNEXPLAINED"
#    entry for outputs/s2_preparation/'s N5 homophone-set glyph labels --
#    expected there, deliberately left unexplained rather than allowlisted
#    so you always see the count and can eyeball it (prior smoke data:
#    ~80-164 total). If UNEXPLAINED shows anything else, or that number
#    jumps by an order of magnitude, stop and show me before committing.
```

Equivalent one-shot form once you trust it: `python pipeline/run_all.py --stage all --data-root data/raw`. I'd still run s1 alone first the very first time, per your own instruction.

---

## E. Corrections folded in (from your message)

1. **R6 revised, per dataset (M5.4), not pooled.** `s4_measure.py`'s
   `measure_normalizer` now sets `recommend_drop_r6` per dataset against the
   5% threshold (`None` for the pooled `"ALL"` pseudo-dataset — that pool's
   composition is a registry artefact, not a population, so it gets no
   verdict of its own). Historical numbers from this session (will be
   re-measured on `data/final/` by your real run): sentnob 4.33% → DROP,
   bd_shs 1.33% → DROP, banfakenews 53.00% → KEEP, banfakenews_full 46.67% →
   KEEP. Report text explicitly notes the BanFakeNews examples are
   curly-quote→straight-quote substitutions (editorial typography). **No
   normalizer ON/OFF arm is scheduled anywhere in this codebase** — that's a
   Phase 3 grid-design decision, out of scope here, but nothing here
   pre-builds it either.
2. **Kinds classifier fixed**, `_classify_touch`/`_diff_char_classes` in
   `s4_measure.py`: now diffs `Counter(before)` vs `Counter(after)` and
   classifies only the characters whose count actually changed (via
   `corpus_stats.classify_char`), instead of the old co-occurrence check
   (`any(c.isdigit() for c in before)`, which fired whenever a digit
   appeared **anywhere** in the whole input, unrelated to what changed).
3. **Table II**: SentNoB labels are `neutral`/`positive`/`negative` (no
   `?`), with the lexicon-probe verification method footnoted directly under
   the table (`s3_describe.py::SENTNOB_LABEL_FOOTNOTE`).
4. **conjunct_split**: `_cer_block_md`'s "flag" column is now a stated rule,
   not a bare ⚠. The pass/fail check excludes s2/s1 (reported, never
   scored) and evaluates s3/s2, s4/s3, s5/s4 only, with `RATIO_RULE_NOTE`
   printed once at the top of every CER block explaining why, pointing at
   `s3_describe.py`'s `eligible_units_per_noise_type.csv` / figure as the
   evidence.
5. **N10**: `measure_n10_test_split` reports emoji-bearing counts on the
   **full** `test.csv` per dataset (not the 300-sample, which can't answer
   the question for splits bigger than 300). Numbers only, printed as its
   own report section — no drop/keep decision is made in code.

---

## F. Publication hygiene — verified

- **`.gitignore`**, verified with real `git check-ignore -v` calls (not
  assumed): `data/final/*/train.csv` and `data/final/*/full/train.csv` →
  **ignored**; `data/final/*/manifest.json` and
  `data/final/*/full/manifest.json` → **not ignored** (tracked, ids +
  seed + source SHA-256 only, no text); `data/raw/**` → **ignored**;
  `outputs/**` → **not ignored** (tracked). All four checked directly, output
  pasted into this session, not inferred.
- **No verbatim corpus text in `outputs/`**: `s4_measure.py`'s tracked
  summary (`m2_4_summary.md`) replaces the normalizer's before/after example
  text with `_structural_description()` (edit kind, offset of first
  difference, length before, length delta) — verified in this session's
  smoke test: 84 Bengali-block codepoints in the tracked file, **all** of
  them the deliberate homophone-set glyph labels (শ/ষ/স, ই/ঈ, উ/ঊ, etc. —
  counted and enumerated, not assumed). The verbatim-text version stays
  under `results/m2_4.md`, which is git-ignored.
- **No absolute paths**: grepped the whole tree (`.py`/`.md`, excluding
  `.git` and `data/raw`) for `G:\Icrest_codebase` and `C:\Users\tutul` —
  zero hits.
- **§D's step 3** is the same check applied to the real, full-scale
  `outputs/` tree — I could only run it against smoke output (see above);
  you need to run it for real once §D's steps finish.

---

## G. BD-SHS content note

Per your CLAUDE.md invariant 10: BD-SHS contains obscene/abusive text by
design (it's a hate-speech corpus) and stays in `data/final/bd_shs/*.csv`
(git-ignored). Nothing in `outputs/` or any figure can contain a line of it —
the hygiene check in §D/§F is exactly what proves that, every run.

---

## H. Design decisions I was unsure about

1. **s1/s3 wall time is genuinely unverified at full scale.**
   `corpus_stats.char_class_composition` and `latin_share_per_text` each do
   an independent O(n) pass over every character via `classify_char`
   (multiple `regex.match` calls per character). On ~71M total raw
   characters across the 4 datasets, this could plausibly take several
   minutes per pass — I estimated 3-10 minutes for s1 and 5-10 for s3, but
   I have not measured it (invariant 9 forbids it) and could be wrong by a
   large factor if regex-per-character turns out slower than I think. If s1
   stalls, the fix is the same shape as the CER incident: merge the two
   per-character passes into one, or move the hot loop to something
   C-backed. Tell me if it exceeds ~10 minutes on any single dataset and I'll
   fix it the same way I fixed the CER measurement, rather than letting it run.
2. **Output-folder-to-pipeline-stage mapping.** Your target tree lists only
   3 output folders (`s1_raw_inspection`, `s2_preparation`, `s3_analysis`)
   for 5 pipeline stages. I mapped s4 (M2.4 measurements) into
   `outputs/s2_preparation/` alongside s2's `provenance.json`, reasoning
   that M2.1-M2.4 are all under METHODOLOGY's M2 "preparation" umbrella and
   there's no 4th folder. If you intended s4 to get its own folder, that's a
   rename, not a rebuild — say the word.
3. ~~`data/final/*/manifest.json` tracked, not ignored.~~ **Resolved —
   confirmed correct.** Kept as implemented.
4. ~~SentNoB leakage numbers — discrepancy.~~ **Resolved.**
   `compute_leakage` now reports **both** representations, labelled, per
   pair, over the same final row population: `raw_text` (as downloaded) and
   `final_text` (post-M2.1). Each pair also gets `pct_of_split` (percentage
   of that split's size, for both representations) — so
   `data/final/sentnob/manifest.json`'s `leakage` block now looks like:

   ```json
   "leakage": {
     "raw_text": {"test_in_train": [...ids], "val_in_train": [...], "val_in_test": [...]},
     "final_text": {"test_in_train": [...ids], "val_in_train": [...], "val_in_test": [...]},
     "pct_of_split": {
       "test_in_train": {"raw_text_pct": .., "final_text_pct": .., "split_size": ..},
       "val_in_train": {...}, "val_in_test": {...}
     }
   }
   ```

   Nothing is dropped or re-split — exactly as instructed; the ids are
   there for Phase 4 to compute the degradation curve both ways.
   `outputs/s2_preparation/report.md` renders both counts and both
   percentages in a table per pair, so the rate is visible without opening
   the manifest. I proved (not assumed) that `final_text` leakage can only
   be **≥** `raw_text` leakage — `preprocess_text` is a deterministic
   function of the raw text, so two raw-identical rows are always
   final-identical too, meaning every raw-leaked id-set is a **subset** of
   the corresponding final-leaked id-set. This is now a regression test
   (`tests/test_pipeline.py::test_prepare_all_four`, subset assertion on all
   three pairs). On the real data, expect `final_text.test_in_train` to
   land near 361 (last real measurement, on the now-deleted `data/clean/` —
   will be re-measured fresh by your real s2 run) against a 1,569-row test
   split, i.e. ~23% — your point about what that means for the clean
   baseline / memorisation-vs-generalisation is noted for Phase 4, not
   something this script asserts (it measures; the report states the
   percentage plainly, it doesn't draw the inference itself, per CLAUDE.md's
   "never interpret results" rule for pipeline output — that reasoning is
   yours, quoted back correctly rather than baked into generated text as if
   it were the pipeline's own conclusion).
5. **`s5_figures.py` builds 11 figures, not the full Figure_Plan_Evaluation.md
   list** (that plan is written for Phase 5 model-comparison figures, not
   Phase 2 data description — F1/F3/F4/F5 etc. don't apply here; there are no
   models yet). What I built maps to your Stage-C figure asks: length ECDF
   (raw + final, with the CER-cap line), character-class stacked bar
   (raw + final), code-mixing ECDF (raw + final), class balance per split,
   length-by-class boxplot, the BD-SHS U+FFFD 2×2, eligible-units-per-noise-
   type boxplot, and the N5 homophone-firing bar chart (the one figure that
   actually needs Bangla glyphs, correctly suffixed `_bn`). The length-ECDF
   figure's "CER cap" annotation text currently overlaps the legend slightly
   at smoke-test size — cosmetic, will look different with real data volume;
   flagging rather than polishing blind.
6. **Eligible-units sampling.** `s3_describe.py` samples 300 texts/dataset
   (`ELIGIBLE_UNITS_SAMPLE`) for the eligible-units-per-noise-type figure,
   matching M2.4's sample size, rather than computing it over the full
   corpus. `n_eligible` is a structural property of the text (proven
   severity/seed-independent, regression-tested in `test_corpus_stats.py`),
   so a sample is representative; full-corpus would just be slower for no
   statistical benefit at this scale.
7. **`--data-root` for `run_all.py --stage all`** is forwarded only to s1
   and s2 (the two stages that touch raw data); s3/s4/s5 don't take it. If
   you pass other stage-specific flags to `run_all.py`, they're forwarded to
   *every* stage via `parse_known_args`, which will error on a stage that
   doesn't define that flag — fine for the plain `--stage all` case above,
   but don't mix in flags like `--out-dir` when running `all`.

---

## I. Git

No git commands were run. Proposed commit message when you're ready:

```
phase-2: restructure into pipeline/ (s1-s5) + data/final/ + outputs/

- delete stale data/clean/, data/splits/, results/table2.*, results/m2_4.*
  (predated two label/kinds-classifier fixes; verified stale via timestamps
  and content, not assumed)
- noisebench/corpus_stats.py + plotstyle.py: shared s1/s3 measurement
  module and validated palette, single source of truth
- pipeline/{s1_inspect_raw,s2_prepare,s3_describe,s4_measure,s5_figures,
  run_all}.py: five-stage pipeline, every script resumable with --limit/
  --force progress-logged (CLAUDE.md invariant 9)
- data/final/<dataset>/{train,val,test}.csv + manifest.json replaces
  data/clean + data/splits; banfakenews_full nests under banfakenews/full/
- corrections: R6 decided per dataset not pooled, kinds classifier now
  diffs actual character classes changed, Table II labels drop "?",
  conjunct_split gets a stated ratio rule instead of a bare flag, N10
  reports full-test-split emoji counts
- outputs/ tracked, publication-hygiene verified (no verbatim corpus text,
  no absolute paths, .gitignore checked with git check-ignore)
- pipeline/verify_hygiene.py: documented, named allowlist for the
  U+0980-U+09FF sweep (outputs/s0_noise_suite/ only -- Phase 1's authored
  fixtures, verified with 0 exact-substring matches against all 3 real
  corpora); N5 glyph labels deliberately left unexplained, not allowlisted,
  so the count stays visible every run
- SentNoB leakage now reported on BOTH raw-text and post-preprocessing text,
  labelled, per pair, with % of split, over the same final row population;
  ids for both preserved in manifest.json for Phase 4's leakage-free
  sensitivity check; raw-subset-of-final proven and regression-tested
- docs/ holds METHODOLOGY.md + the two plan docs; CLAUDE.md invariants 9
  (no long-running analysis) and 10 (publication hygiene) added
- Phase 1 outputs moved results/ -> outputs/s0_noise_suite/
- tests/test_pipeline.py replaces test_phase2_pipeline.py; 56 tests pass
```
