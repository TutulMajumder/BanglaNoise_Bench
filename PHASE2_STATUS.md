# bangla-noisebench — Phase 2 status report

Generated 2026-09-02. Commit `96e4a43` (`phase-2: data-preparation code`).
Same format as `PHASE1_STATUS.md`. Numbers, not prose.

> **Phase 2 cannot be completed in this workspace: the three raw corpora
> (SentNoB, BD-SHS, BanFakeNews) are not on disk.** Every Phase 2 acceptance
> item that needs real numbers — Table II, real-corpus CER, the four M2.4
> measurements, the R6 recommendation — is **PENDING data**. The full pipeline
> is written, committed, and smoke-tested on a synthetic fixture; drop the CSVs
> in and it runs. See §B and §L.

---

## A. Phase 1 carry-overs — both done

| item | status |
|------|--------|
| Delete `CLAUDE_md_for_repo.md` (content already in `CLAUDE.md`) | **done** — you removed it in `be81c8c`; confirmed absent from `git ls-files` and disk |
| Verify the `data/splits` gitignore negation actually works | **verified working.** `mkdir -p data/splits && echo test > data/splits/_probe.txt` → `git check-ignore -v data/splits/_probe.txt` prints nothing (exit 1, not ignored); `git status` lists the file. The earlier inline-comment form silently failed; the pattern is now `data/*` then `!data/splits` on its own line (`.gitignore` has no inline comments). Also verified: `data/prepare.py` tracked, `data/clean/*` and `data/raw/*` ignored. |

---

## B. BLOCKER — datasets not on disk

Searched `G:\Icrest_codebase`, `G:\ICREST`, and `G:\` to depth 6. Found: a
`fine_tuned_banglabert_model/` from an unrelated thesis, and many non-Bangla
CSVs. **No SentNoB / BD-SHS / BanFakeNews files anywhere**, no archives.

The task says "assume CSVs are already on disk" — they are not in any location
visible to this session (they may be Kaggle-only). So the pipeline is built and
tested, but the real-number deliverables cannot be produced here.

Sources (from `Topic4_FULL_PAPER_PLAN.md`, for whoever places the files):

| dataset | source | note |
|---------|--------|------|
| BD-SHS | github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media · Kaggle `naurosromim/bdshs` | ~50,200+ rows; record class balance |
| SentNoB | github.com/KhondokerIslam/SentNoB · Kaggle `cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts` | split sizes undocumented online — we report ours |
| BanFakeNews | github.com/Rowan1697/FakeNews · Kaggle `cryptexcode/banfakenews` | labelled balanced subset (~7K/1K) for main runs |

---

## C. Code delivered

| bytes | path | tracked | purpose |
|------:|------|:------:|---------|
| 21119 | `data/prepare.py` | yes | M2.1 preprocessing + M2.2 splitting |
| 2655 | `data/README.md` | yes | expected raw layout + schema table |
| 9870 | `scripts/table2_report.py` | yes | M2.3 Table II → `results/table2.{md,html}` |
| 20091 | `scripts/m2_4_measurements.py` | yes | the four M2.4 measurements → `results/m2_4.{md,html,json}` |
| 5820 | `tests/test_phase2_pipeline.py` | yes | 4 smoke tests |
| 388 | `requirements-phase2.txt` | yes | `-r requirements.txt` + `pandas>=2.0` |

`noisebench/` was **not modified** (checked: `git diff be81c8c..HEAD -- noisebench/`
is empty). The two Phase 2 scripts import three private helpers for reading only
— `_is_emoji_codepoint` (M2.3 emoji rate), `_homophone_occurrences` and
`_HOMOPHONE_MEMBER_TO_SET` (M2.4(2) grouping pass) — which M2.4 explicitly calls for.

---

## D. Tests

`python -m pytest -q` → **44 passed** (40 Phase 1 + 4 Phase 2). Phase 2:

| test | checks |
|------|--------|
| `test_prepare_all_three` | all 3 datasets clean; `clean/<ds>.csv` has exactly `id,text,label,split`; manifests reload and partition the id set with no overlap; SentNoB keeps provided splits; BD-SHS/BanFakeNews land in 60–80 / 3–18 / 12–30 % |
| `test_split_is_deterministic` | two runs of `prepare_one` produce byte-identical manifests |
| `test_preprocess_order` | `"  See   https://a.b/c?d=1  @Ripon   ABC  "` → `"See <URL> <USER> ABC"` (replace not remove, whitespace collapsed, no lowercasing) |
| `test_table2_and_m2_4_run` | Table II builds, N10 host picked; `measure_n5` returns the ই/ঈ + উ/ঊ review keys; `measure_emoji` / `measure_cer` return the right shapes (9×5 per dataset) |

Fixture CSVs are built from `tests/bangla_samples.SENTENCES` (already in the
repo) — no invented Bangla.

---

## E. `data/prepare.py` — behaviour

**M2.1, in this exact order** (`preprocess_text`):
1. `unicodedata.normalize("NFC", …)`
2. URLs → `<URL>` (regex, before mentions), `@mentions` → `<USER>` — replace, never remove
3. collapse `\s+` → single space, strip ends
4. no `.lower()`
5. no emoji handling (that is N10)

then: 6. drop exact duplicates on cleaned text (**within each provided split**
for SentNoB, else global) — count recorded; 7. drop texts with `< 3` grapheme
clusters (`regex \X` after NFC) — count recorded. Truncation to 128 tokens is
left to model tokenization.

**M2.2 splits:**
- SentNoB → `src_split` used as given. Cross-split duplicate texts are counted
  and surfaced as a warning, never reassigned.
- BD-SHS, BanFakeNews → `stratified_split(seed=42)`: per class, ids sorted then
  `random.Random(42).shuffle`, sliced 70/10/20. Order-independent ⇒ reproducible.
- id = `sha1(dataset · src_split? · cleaned_text)[:16]`. Manifests:
  `data/splits/<ds>.{train,val,test}.txt`, one id per line, sorted.

**Schema registry** (`DATASETS`) — every field `# TODO(verify)`:

| dataset | text col (tried in order) | label | mapping — **confirm** |
|---------|--------------------------|-------|-----------------------|
| `sentnob` | `Data, data, text, sentence` | `Label` | `0/1/2` → names **unconfirmed** (Topic4: 0=neutral,1=positive,2=negative; other sources flip two) |
| `bd_shs` | `sentence, text, Sentence, comment` | `hate` | `0`=not_hate, `1`=hate |
| `banfakenews` | `content, Content, articleContent, text` | *(file-level)* | `LabeledAuthentic-7K.csv`→1, `LabeledFake-1K.csv`→0 |

Missing text/label column → prints the columns found, exits 2. Override with
`--text-col` / `--label-col`.

---

## F. Smoke-test output (synthetic fixture — NOT real data)

Fixture = the ~50 distinct `bangla_samples` sentences, replicated and labelled
round-robin, laid out per each dataset's expected schema. Proves shapes and
plumbing only; **do not read any number here as a result.**

**Table II (fixture):**

| dataset | n | class dist | mean/median len | emoji rate | dupes rm | short rm |
|---------|--:|-----------|-----------------|-----------:|---------:|---------:|
| banfakenews | 58 | authentic=58 | 25.3 / 25 | 8.6% (5/58) | 162 | 0 |
| bd_shs | 58 | hate=29, not_hate=29 | 25.3 / 25 | 8.6% (5/58) | 162 | 0 |
| sentnob | 128 | neg=42, neu=44, pos=42 | 25.4 / 25 | 7.8% (10/128) | 92 | 0 |

Splits: sentnob 58/25/45 (provided); bd_shs 40/6/12, banfakenews 41/6/11
(stratified). N10 host picked = highest emoji rate. Cross-split-duplicate
warning fired for sentnob (expected — the fixture reuses sentences).

**M2.4 (fixture, sample 50/ds, seeds {42,1337}):**
- (1) 9×5 CER: synthetic + one table per dataset + ALL, each with s2/s1…s5/s4
  and a `⚠` flag when a ratio is >10% off `2.00/2.00/1.75/1.43`. On the tiny
  fixture `conjunct_split` flags at low severity (few junctions → noisy) —
  exactly the M2.4(1) caveat about N7 shifting on real text.
- (2) N5 per-set firing table per dataset + the ই/ঈ, উ/ঊ review block with a
  "< 1 eligible per 1000 texts ⇒ recommend DROP" rule (not auto-applied).
- (3) `emoji_range_coverage` per dataset + ALL; proposes (does not apply) range
  widening if any dataset with ≥20 occurrences is < 90%.
- (4) `normalizer touch_rate` per dataset + overall, with a `kinds` breakdown
  and before/after examples; prints the R6 recommendation.

---

## G. Table II (M2.3) — PENDING data

Will report per dataset: n after cleaning, class distribution (+%), mean and
median grapheme-cluster length, emoji rate (fraction of texts with ≥1 emoji),
duplicates removed, short texts removed, and the three split sizes. The highest
emoji rate names the N10 host. `results/table2.{md,html}`.

---

## H. M2.4 measurements — PENDING data

1. **9×5 CER, real vs synthetic**, ratio columns vs `2.00/2.00/1.75/1.43`,
   per-row `⚠` when >10% off. Acceptance: real-corpus ratios still approximate
   the target (N7/N6 may shift with real conjunct/matra density — reported, not
   suppressed).
2. **N5 per-set firing**: `n_eligible` (grouping pass over
   `_homophone_occurrences`) and `n_applied` (sum of `PerturbStats.submodes`,
   severity 3) per `HOMOPHONE_SET` per dataset. **ই/ঈ and উ/ঊ**: if either is
   `< 1` eligible occurrence per 1000 texts across the real corpora, the report
   flags it for removal **with that number as the evidence**; not removed by
   this script.
3. **Emoji coverage** per dataset via `validate.emoji_range_coverage` —
   reported as an **upper bound against `emoji` 1.4.2 ≈ Unicode 13 (2021)**. If
   any dataset (≥20 occurrences) is `< ~90%`, the report lists the uncaught
   code points and a proposed range widening **to be applied later in a
   reviewed change** — `noisebench/` is not touched in Phase 2.
4. **Normalizer touch_rate** per dataset and overall, plus what it touched
   (kind breakdown + examples). Feeds §I.

---

## I. R6 (normalizer ON/OFF) recommendation — DEFERRED, rule fixed

**Cannot recommend yet — needs the real-corpus touch_rate.**

The decision rule is fixed now, before any result is seen (METHODOLOGY M5.4):

> Compute `normalize_banglabert` touch_rate over the cleaned real corpora.
> **If overall touch_rate < ~5% → DROP R6**, run `--normalizer on` only, and
> report the null: *"the normalizer bundled with the standard Bangla encoder
> pipeline does not mitigate orthographic noise; it operates on URLs and
> punctuation, not Bangla orthography."* Otherwise keep R6 as a full
> ON/OFF condition and reclaim nothing.

Prior evidence pointing toward DROP (not sufficient on its own): on the Phase 1
**synthetic** corpus touch_rate was **0.0%** (540/540 calls unchanged) —
`normalize()` with defaults acts on URLs, emails, punctuation spacing and NFKC,
none of which the clean synthetic sentences contain. The real corpora are
messier (social-media text, URLs, mixed English), so the real touch_rate will
be higher than 0% — the question is whether it clears 5%. `scripts/m2_4_measurements.py`
prints `recommend_drop_r6` directly once data is present.

---

## J. Dependencies

Phase 2 adds **pandas** only. `requirements-phase2.txt`:

```
-r requirements.txt
pandas>=2.0
```

Resolved locally: pandas 2.2.3, Python 3.12.5. Kaggle ships pandas 2.2.x +
Python 3.12.13. `noisebench/` itself still imports only `regex`. No pyarrow
(clean output is CSV, UTF-8, `\n`).

---

## K. Git

| field | value |
|-------|-------|
| HEAD | `96e4a43` `phase-2: data-preparation code (METHODOLOGY M2)` |
| history | `81ae763` (tag `phase-1`) → `2f47f8c` → `be81c8c` (your cleanup) → `96e4a43` |
| new tracked | `data/prepare.py`, `data/README.md`, `scripts/table2_report.py`, `scripts/m2_4_measurements.py`, `tests/test_phase2_pipeline.py`, `requirements-phase2.txt` |
| still not tracked | verified no `data/raw/`, `data/clean/`, `results/`, `*.csv`, model files in `git ls-files` |
| `data/splits/` | tracked as a path; **empty until real data is processed** (git stores no empty dir — the manifests appear on the first real `prepare.py` run) |
| `noisebench/` diff since `phase-1` | none |

---

## L. To finish Phase 2 (what's needed)

1. **Place the three corpora** under `data/raw/{sentnob,bd_shs,banfakenews}/`
   (or run with `--data-root /kaggle/input`).
2. **Confirm the schema** in `data/prepare.py::DATASETS` against the actual
   files — especially the **SentNoB label→class-name mapping** (0/1/2), which
   two independent sources disagree on. Wrong mapping = wrong class names in
   Table II, nothing else.
3. Run:
   ```
   pip install -r requirements-phase2.txt
   python data/prepare.py --data-root data/raw --dataset all
   python scripts/table2_report.py
   python scripts/m2_4_measurements.py
   git add data/splits && git commit -m "phase-2: split manifests"
   ```
4. Fill Table II (§G), the four M2.4 blocks (§H), and the R6 recommendation
   (§I) into a completed PHASE2_STATUS.md; check the real-corpus CER ratios
   against `2.00/2.00/1.75/1.43` and act on any near-zero N5 set.

Carried further (Phase 3+, unchanged from PHASE1_STATUS §L): M6.3
native-speaker plausibility rating; per-`# TODO(verify)` native-speaker review
of `bangla_maps.py`; the R2 noise grid; subword-token-count-delta metric (M6.1).

---

## M. Deviations from METHODOLOGY.md

1. **SentNoB dedup is per-provided-split**, not global, so the shipped Val/Test
   sets are preserved exactly ("use as given", M2.2). Cross-split duplicate
   texts are counted and warned, not dropped or reassigned. M2.1 step 6 says
   "within each dataset before splitting"; for a dataset whose splits are fixed
   this is the faithful reading. Noted here for reconciliation.
2. **`requirements-phase2.txt`** is a new file (pandas); the released-artefact
   `requirements.txt` is unchanged.
3. Table II / M2.4 / R6 numbers are **absent** — data blocker, not a design
   choice.

Nothing contradicts a locked M0 decision, the severity scale, the taxonomy, or
the metric set. `noisebench/` is byte-identical to tag `phase-1`.
