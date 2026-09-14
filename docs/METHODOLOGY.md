# METHODOLOGY — bangla-noisebench
**The precise spec the code is written against.** Task-ordered, not date-ordered. Work through M1 → M7; each module has an acceptance test. Roughly 70% of this becomes §III and §IV of the paper.

> Companion to `Topic4_FULL_PAPER_PLAN.md` (experiment design) and `Figure_Plan_Evaluation.md` (figures). Where those disagree with this file, **this file wins for implementation detail**; the plan wins for scope.

> **RESTORED 2026-09-02.** The original was lost because it lived outside the git repo. This version folds in every decision made since it was written. **Keep this file inside `bangla-noisebench/` and commit it** — see the note at the end.

---

# M0. RESOLVED DECISIONS

## M0.1 N3 — no keyboard layout

A guessed Bijoy layout is unverifiable and would discredit the suite. **N3 is in-class substitution**: a character is replaced by another drawn uniformly from its own orthographic class — consonant → consonant, independent vowel → independent vowel, digit → digit. No layout dependency.

Paper wording: *"N3 substitutes a character with another of the same orthographic class. We deliberately avoid keyboard-layout-specific adjacency because Bangla input is split across incompatible layouts (Bijoy, Avro phonetic, mobile phonetic), so no single adjacency model is representative. Layout-conditioned substitution is left to future work."*

`build_adjacency_from_layout()` is retained as an unused helper for an optional N3b path. `KEYBOARD_ADJACENCY` and the candidate layout grids are **not** part of the shipped suite.

## M0.2 Answers to the agent's original ten questions

| # | Question | Decision |
|---|---|---|
| Q2 | N1 never inserts a combining mark | **Approved.** Inserting a matra or hasanta at an arbitrary position produces invalid sequences. N6 owns matra manipulation. |
| Q3 | N5: merge the matra forms? | **Yes.** In running Bangla, ই/ঈ and উ/ঊ appear far more often as ি/ী and ু/ূ. ি↔ী and ু↔ূ are first-class sets. |
| Q4 | N6 omits া, ৃ, ৈ? | **No — include all** dependent vowel signs. া is the most frequent matra in Bangla. Hasanta and chandrabindu are separate drop targets. |
| Q5 | N6 "alter" targets | **`N6_ALTER_PAIRS = {ি↔ী, ু↔ূ}` only** *(revised 2026-09-01)*. ে↔ৈ and ো↔ৌ were removed: alteration models *phonetic* confusion, and Bangla has no vowel-length distinction in speech, so the i/u length pairs are genuinely confused. ে/ৈ and ো/ৌ are phonetically distinct and only visually similar — stroke loss/gain belongs to N1/N2. Everything else is drop-only. |
| Q6 | N7: drop hasanta or insert ZWNJ? | **drop_hasanta only** *(revised 2026-09-01)*. N7 is single-mechanism. ZWNJ insertion was removed: N7 models *failure to form a conjunct*, whereas ZWNJ insertion is a deliberate rendering choice producing a correct non-ligated form — a different construct, and one subject to normalization. Triple conjuncts (স্ত্র) remain excluded; state in Limitations. |
| Q7 | N8 multiplicity | **Per submode** *(revised 2026-09-01)*: `insert_vowel` adds **exactly +1** (a distinct real spelling, not a repetition); `repeat_vowel` and `repeat_final` produce **2–4 copies total**. Cluster-level CER scores repeated matras as a single substitution because they stay one grapheme cluster — a correct consequence of M1.5, noted for §III. |
| Q8 | ঋ included, archaic excluded | **Approved.** Document the exclusion of ঌ/ৡ. |
| Q9 | Emoji range coverage | **Measure it in M2.** `normalizer` pulls in the `emoji` library transitively, so coverage can be measured for free: compare `EMOJI_CODEPOINT_RANGES` against the library's list on the real corpora (`validate.emoji_range_coverage`). **Detection in `perturbations.py` stays hand-rolled** — the released artefact must not depend on `emoji`; only the *measurement* uses it. Widen the ranges if coverage is under ~90%. **Caveat:** the pinned `emoji` is **1.4.2** (the version `csebuetnlp/normalizer` pins transitively), whose tables track roughly **Unicode 13 (2021)**. Emoji added after that fall outside both the reference list and our ranges, so the coverage figure is an **upper bound against a 2021-era reference**, not against current Unicode. State this in Limitations. |
| Q10 | ো/ৌ decomposition | **Handled.** NFC at entry and exit of every perturbation. `validate.py` tests that ো and ৌ survive both an NFC round trip and a `csebuetnlp/normalizer` round trip. |

## M0.3 Plausibility-review decisions (2026-09-01)

Logged for the paper trail after reviewing the Table I output.

| # | Change | Rationale |
|---|---|---|
| a | **Digits removed from `INSERTABLE_CHARS`** | With keyboard adjacency removed, no mechanism produces a digit inserted mid-word; ~17% of insertions were digits. Digits remain eligible for N2 and N3. |
| b | **ব/ভ dropped from `HOMOPHONE_SETS`** | ব/ভ is one of ten aspirated/unaspirated pairs (ক/খ, গ/ঘ, চ/ছ, জ/ঝ, ট/ঠ, ড/ঢ, ত/থ, দ/ধ, প/ফ, ব/ভ). Aspiration is phonemic in Bangla, so speakers do not confuse these by ear; including only ব/ভ would be arbitrary. **All ten or none — we take none.** Moved to `HOMOPHONE_SETS_REJECTED` with the rationale. |
| c | **N7 `insert_zwnj` removed** | See Q6. |
| d | **`MATRA_TO_INDEPENDENT_VOWEL` restricted** to {ু→উ, ূ→ঊ, ি→ই, ী→ঈ, ো→ও} | The vowels Bangla speakers lengthen for emphasis online are i, u, o (খুউব, ভালোওও, আমিইই). ে→এ, ৈ→ঐ, ৃ→ঋ, া→আ are unattested as emphatic spellings. |
| e | **N10 `remove` collapses whitespace and strips** | Otherwise `remove` differs from `keep` by more than the emoji, confounding the three-way comparison. |

**Retained sets:** শ/ষ/স · ন/ণ · ই/ঈ · ি/ী · উ/ঊ · ু/ূ · র/ড়/ঢ় · জ/য. `ই/ঈ` and `উ/ঊ` are retained on the evidence of ঈদ/ইদ and ঊষা/উষা, ঊর্ধ্ব/উর্ধ্ব; both are low-frequency in running text and their retention is **subject to the M2.4 firing-count check**.

---

# M1. CORE DEFINITIONS

## M1.1 The text unit

All character-level operations work on **Unicode extended grapheme clusters**, obtained with `regex` `\X`, after NFC normalization. A "character" in this spec always means one grapheme cluster (`কি` is one unit).

```python
import regex, unicodedata
def clusters(text: str) -> list[str]:
    return regex.findall(r"\X", unicodedata.normalize("NFC", text))
```

**Note on ড় ঢ় য়** (U+09DC/DD/DF): these are Unicode composition exclusions, so NFC decomposes them to base + nukta. They remain **one grapheme cluster**, so cluster-level operations are unaffected — but character-level membership tests must use the decomposed form. `bangla_maps.py` keeps literals precomposed for readability and NFC-normalizes them at module load.

## M1.2 Eligible units per noise type

Severity is a fraction **of eligible units**, and "eligible" differs per type. Normative.

| Noise | Eligible unit | Pool definition |
|---|---|---|
| N1 char_insert | inter-cluster boundary | all boundaries inside tokens of ≥2 clusters |
| N2 char_delete | grapheme cluster | clusters in tokens of ≥2 clusters (never delete a whole single-cluster token) |
| N3 char_substitute | grapheme cluster | clusters whose base character has a defined in-class pool |
| N4 char_transpose | adjacent cluster pair | pairs within a single token |
| N5 homophone_confuse | character occurrence | occurrences of any character in `HOMOPHONE_SETS` (base letters and matra forms) |
| N6 matra_perturb | sign occurrence | any dependent vowel sign, hasanta, or chandrabindu |
| N7 conjunct_split | conjunct junction | each `C + HASANTA + C` junction |
| N8 elongate | vowel-bearing cluster | clusters containing a matra or independent vowel |
| N9 whitespace_error | word-boundary event | one pool combining split points and merge points |
| N10 emoji_transform | — | not severity-graded; see M1.4 |

## M1.3 How many units to perturb

Naive rounding breaks short texts: at severity 1 a 12-cluster text yields `0.05 × 12 = 0.6 → 0`, so severity 1 becomes a silent no-op. Use deterministic floor plus a seeded Bernoulli remainder, which keeps the expected rate exactly `frac`:

```python
def n_to_perturb(n_eligible: int, frac: float, rng: random.Random) -> int:
    if n_eligible == 0:
        return 0
    exact = n_eligible * frac
    k = int(exact)                      # floor
    if rng.random() < (exact - k):      # seeded remainder
        k += 1
    return k
```

Selection is `rng.sample(eligible, k)` — without replacement.

> ### ⚠ The per-item seeding rule
> The per-call RNG **must** be derived by hashing `(seed, noise_type, severity, text)`, not by seeding `random.Random(seed)` directly. Seeding directly makes the first `.random()` call identical for every text, so the Bernoulli remainder fires all-or-none across a corpus. This was invisible for types with 20–50 eligible units (floor dominates) and fatal for N7 (≈4.7 eligible units, where the Bernoulli *is* the signal). See `perturbations._derive_seed` and its regression test.

### Achieved rate vs. nominal fraction

The count rule is unbiased: `mean n_requested / (n_eligible · frac) ≈ 1.00` for every noise type at every severity. For eight of nine types `n_applied = n_requested`, so achieved equals nominal within ±1%. **One type — N2 — applies fewer units than requested**, because of a guard no implementation can avoid. (N4 was the second such type until the disjoint-pair rewrite below; it now reaches nominal.)

**N2 char_delete — documented, not fixed.** Its guard forbids deleting every cluster of a token (a one-cluster token must never disappear). The deficit is small (−9% at s5) and mathematically unavoidable. **Report the achieved fraction:**

| | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| nominal | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 |
| N2 achieved | 0.050 | 0.099 | 0.195 | 0.332 | 0.453 |

**N4 char_transpose — fixed.** The original implementation made *k* global picks then skipped overlaps, giving a −22% deficit at s5 (achieved 0.392 vs nominal 0.50). That is large enough to make "severity 5" mean materially less for N4 than for the other eight types, which breaks cross-type comparability in the per-noise small multiples. N4 now **draws *k* disjoint adjacent pairs from a pool that shrinks as pairs are taken**: pick a random surviving pair, apply the swap, remove that pair and its two neighbours from the pool, repeat. This is the correct construction for *k* independent non-overlapping adjacent transpositions; semantics unchanged. `n_applied` falls below *k* only if the pool empties first (a genuine limit on how many disjoint swaps a short token set admits), which on real text is negligible — measured `applied/requested` = 1.000 at s1–s4 and 0.999 at s5. Achieved fraction:

| | s1 | s2 | s3 | s4 | s5 |
|---|---|---|---|---|---|
| nominal | 0.05 | 0.10 | 0.20 | 0.35 | 0.50 |
| N4 achieved | 0.051 | 0.100 | 0.200 | 0.350 | 0.500 |

N2's guard rationale and achieved-rate table appear in the paper; the N4 rewrite is noted in Limitations as a fixed deviation. Measured over a 200-text corpus × seeds {42, 1337, 2024}, produced by `scripts/phase1_report.py` ("Achieved vs nominal perturbation rate").

## M1.4 The stats contract

```python
@dataclass
class PerturbStats:
    noise_type: str
    severity: int | None        # None for N10
    n_eligible: int
    n_requested: int            # what the M1.3 count rule selected
    n_applied: int              # what actually landed after guards
    submodes: dict[str, int]    # N7/N8 mechanisms; for N5, keyed per homophone set
    unchanged: bool

def perturb_with_stats(text, noise_type, severity, seed) -> tuple[str, PerturbStats]: ...
def perturb(text, noise_type, severity, seed) -> str:   # thin wrapper
```

`n_requested` vs `n_applied` is what exposes guard-induced shortfall. **N5's `submodes` are keyed per homophone set** (e.g. `{"শ/ষ/স": 2, "জ/য": 1}`) so M2.4 can be computed by summation with no new instrumentation.

## M1.5 CER

**Normalized Levenshtein distance over NFC-normalized grapheme-cluster sequences**, not code points.

```
CER(a, b) = levenshtein(clusters(a), clusters(b)) / max(1, len(clusters(a)))
```

Computing over code points would inflate N6 and N7 specifically. Corpus CER is the arithmetic mean over texts.

**Interpretation caveat for §III:** CER measures *whether the perturbation applied and scales monotonically*. It is **not** a measure of model-facing difficulty, and it is **not comparable across noise types** — a transposition costs 2 edits, a substitution 1, and repeated matras stay one cluster so N8 scores as a single substitution regardless of how many were added. Difficulty is measured by F1 drop. Say this explicitly or a reviewer will read the CER table as a damage ranking.

---

# M2. DATA PREPARATION

## M2.1 Preprocessing (once, before any perturbation)

Apply in exactly this order:

1. NFC normalization.
2. Replace URLs with `<URL>`, `@mentions` with `<USER>` — **replace, never remove** (removal changes length distributions between datasets).
3. Collapse whitespace runs to a single space; strip.
4. **Do not lowercase** (Bangla has no case; it would only affect embedded English — an uncontrolled variable).
5. **Do not strip emoji** — that is experimental variable N10.
6. Drop exact duplicates *within* each dataset before splitting. Record the count.
7. Drop texts with fewer than 3 grapheme clusters. Record the count.

Truncation to 128 tokens happens at model tokenization, **not** here.

## M2.2 Splits

- **SentNoB** ships Train/Val/Test — **use as given.**
- **BD-SHS** and **BanFakeNews**: seeded stratified **70/10/20**, `split_seed=42`. Row ids, the seed, and source-file SHA-256 are committed in `data/final/<dataset>/manifest.json` (text itself stays git-ignored).
- **BanFakeNews** main runs use the labelled balanced subsets (≈7K authentic / 1K fake). The full 48K/1K set (`banfakenews_full`) was implemented as a secondary condition, then **cut entirely 2026-09-15** — it supported no paper claim and cost 72% of s1's real-scale runtime. Three datasets remain: SentNoB, BD-SHS, BanFakeNews. See `PHASE2_STATUS.md`.

## M2.3 Statistics for Table II

Per dataset: n after cleaning, class distribution, mean/median cluster length, **emoji rate**, duplicates removed, short texts removed. (Emoji rate no longer selects an N10 host task — R5 is cut, 2026-09-15; see M5.3. Still reported, since it's the evidence for that cut.)

## M2.4 Real-corpus regeneration and firing counts

1. **Regenerate the 9×5 CER table on real corpus samples.** That version goes in the paper; the synthetic one was a development check. Report both. **Each sampled text is capped to its first 300 grapheme clusters before perturbing for this measurement only** (`scripts/m2_4_measurements.py::CER_TRUNCATE_CLUSTERS`) — `validate.levenshtein` is pure-Python O(n²), and BanFakeNews articles average ~1,100 clusters (some far more) versus ~60 for the Phase 1 fixture, which made the full-length real-corpus table unrunnable (a real run hung 17h). This measurement checks that the severity-ratio structure survives on real text, not a document-level CER claim; CER is normalized by reference length and the M1.3 count rule is per-eligible-unit, so the cap does not bias the ratios. N5/emoji/normalizer measurements below are unaffected — they see full-length text.
2. **N5 per-set firing counts:** `n_eligible` and `n_applied` per `HOMOPHONE_SET` per dataset. `n_applied` sums directly off `PerturbStats.submodes`; `n_eligible` needs a grouping pass over `_homophone_occurrences`. Any set eligible near-zero times on real text is **dropped with evidence** and noted in Limitations — not guessed at. `ই/ঈ` and `উ/ঊ` are the two under review.
3. **Emoji coverage:** fraction of emoji occurrences caught by `EMOJI_CODEPOINT_RANGES` versus the `emoji` library, measured in the validation layer only (`validate.emoji_range_coverage`). The reference is `emoji` 1.4.2 ≈ Unicode 13 (2021); report the number as an upper bound against that era, not current Unicode (see M0.2 Q9).
4. **Normalizer touch_rate on real corpora** — see M5.4.

**Acceptance:** Table II printed with real numbers; split files reload; real-corpus CER ratios still near 2.00 / 2.00 / 1.75 / 1.43.

---

# M3. THE NOISE SUITE — ACCEPTANCE

1. **Determinism:** identical output across 100 texts × 3 repeat calls.
2. **Monotonicity:** mean corpus CER strictly increases across severity 1→5 for every noise type. **Print the 9×5 table with real numbers.** Ratio structure must approximate 2.00 / 2.00 / 1.75 / 1.43.
3. **Validity:** NFC-valid, no U+FFFD, no orphan nukta; ো and ৌ survive both an NFC round trip and a `csebuetnlp/normalizer` round trip.
4. **Non-identity:** at severity 5, ≥95% of texts with ≥1 eligible unit change.

Plus the 25 structural assertions on `bangla_maps.py` (pool sizes 35/11/10; no matra, hasanta or chandrabindu in any `IN_CLASS_POOL` or in `INSERTABLE_CHARS`; no digits in `INSERTABLE_CHARS`; pools mutually disjoint; `N6_ALTER_PAIRS` exactly {ি↔ী, ু↔ূ}; `MATRA_TO_INDEPENDENT_VOWEL` exactly {ু→উ, ূ→ঊ, ি→ই, ী→ঈ, ো→ও}; ব/ভ absent from `HOMOPHONE_SETS`; `HOMOPHONE_SETS_REJECTED` holds the ten aspiration pairs and is disjoint from `HOMOPHONE_SETS`; every string constant NFC-idempotent). Run by `scripts/constants_sheet.py` and in `pytest` (`test_bangla_maps_structural_assertions`).

---

# M4. MODELS AND TRAINING

Fixed and identical across models:

```
max_len=128, batch=32, lr=2e-5, epochs=4, AdamW,
linear warmup 10%, fp16, early stopping on val macro-F1 (patience 2),
seeds = {42, 1337, 2024}
```

Models: `csebuetnlp/banglabert`, `csebuetnlp/banglishbert`, `xlm-roberta-base`, `bert-base-multilingual-cased`, and TF-IDF `char_wb` 2–5-gram + `LinearSVC` (`C=1.0`, `class_weight='balanced'`, `min_df=2`).

**Normalizer.** `--normalizer {on,off}`, default `on`, recorded as a column in every results row. It **must** call `csebuetnlp/normalizer`'s `normalize(text)` with default arguments — the same package and call the Phase 1 survival test validated. `bnunicodenormalizer` is a different project (OCR/ASR) and must not be substituted. Pinned to a git commit (no PyPI release exists):

```
normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9
```

**Acceptance:** clean BD-SHS macro-F1 lands in the published range (≈0.91). If not, stop and fix the pipeline.

**Environment (verified 2026-09-01):** Kaggle runs Python **3.12.13**; the git install of `normalizer` succeeds there. The code targets **3.10+** anyway, as a portability choice for the released artefact — not a Kaggle constraint. `normalizer` pulls in `emoji` and `ftfy` transitively; record their resolved versions, since both can affect reported numbers.

---

# M5. THE EXPERIMENT GRID

## M5.1 Robustness (R2)

Perturbation applied to **test sets only**. Every (model × task × seed × noise_type × severity × normalizer) appended to `results/results.csv`. Inference only. Resumable — skip rows already present. `noise_type` ranges over the 9 severity-graded types (N1–N9) only — **N10 is deliberately excluded from this grid** (decided 2026-09-15; see M5.3 and `Topic4_FULL_PAPER_PLAN.md` §4/§7/§11). This matches what the CER tables already reported, so code and tables now agree explicitly rather than incidentally.

## M5.2 Augmentation (R4)

Perturbation applied to **training sets only**:

- **50%** of training examples perturbed.
- Severity drawn uniformly from **{1, 2, 3}** — never 4–5, which degrades clean performance and confounds the result.
- Noise type drawn uniformly from the **training group**.

**Groups:** A = {N1,N2,N3,N4} generic · B = {N5,N6,N7,N8} Bangla-specific · C = {N9}

| Train on | Test on A | Test on B |
|---|---|---|
| **A** | matched | **mismatched** |
| **B** | **mismatched** | matched |

Two models, three tasks, three seeds. Matched-vs-mismatched is what makes "augmentation helps" non-trivial.

**Apply the M1.3 per-item seeding rule here** — severity and noise type are drawn per training example, which is exactly the situation that produced the N7 seed-correlation bug.

## M5.3 Emoji (R5) — **CUT, 2026-09-15**

Three conditions — `keep` / `remove` / `replace_with_text` — applied identically at train and test time, on the highest-emoji-rate task. Bangla descriptions hand-mapped for the top ~50 emoji by corpus frequency; removal as fallback. Report mapping coverage.

**This run is cut, not deferred.** s1 raw inspection: 266 emoji-bearing texts in SentNoB (1.7%), 288 in BD-SHS (0.57%) — roughly 27 and 57 items on the test splits alone. Not enough to measure anything, independent of GPU budget (contrast M5.4, which is conditional on a *measured* touch_rate; this is conditional on nothing — the corpora simply don't carry enough emoji). The spec above stays for the record (N10 itself is a real, defined phenomenon — see M0/M1.4), but no R5 run is scheduled. Report as a limitations finding:

> *"Available Bangla classification corpora carry too little emoji content to support an emoji-noise evaluation — itself a finding about the state of Bangla datasets."*

## M5.4 Normalizer ON/OFF (R6) — **conditional**

On the Phase 1 synthetic corpus, `csebuetnlp/normalizer` was a **complete no-op**: 540/540 calls unchanged, touch_rate 0.0%. It acts on URLs, emails, punctuation spacing and NFKC — not Bangla orthography. So the post-normalizer CER figures prove nothing about survival; they are identical to the plain CER row, and the report says so rather than implying a guarantee.

**Decision rule:** measure touch_rate on the real corpora in M2. If it is **under ~5%**, drop R6 and reclaim the ~4 GPU-hours. Either way the finding is reportable, and the null is the better sentence:

> *"The normalizer bundled with the standard Bangla encoder pipeline does not mitigate orthographic noise; it operates on URLs and punctuation, not Bangla orthography."*

---

# M6. METRICS AND STATISTICS

## M6.1 Metrics

Primary **macro-F1**. Additionally weighted F1, per-class P/R, balanced accuracy. **BanFakeNews also reports PR-AUC and MCC**; accuracy is never reported for it.

Derived: **Mean Noise F1**; **degradation slope** (OLS of macro-F1 on severity index 0–5); **rank-inversion indicator** (Spearman ρ between clean-F1 ranking and Mean-Noise-F1 ranking). ECE secondary.

**Phase 3 addition:** alongside CER, report **mean subword-token count before vs after** perturbation, per noise type per severity, under the BanglaBERT tokenizer. Cluster-CER understates N8 because খুশিিিি stays one grapheme cluster while the tokenizer sees a different token entirely.

## M6.2 Multiple comparisons — fixed before any results are seen

The grid is 5 models × 3 tasks × 9 noise types × 5 severities. Testing every cell manufactures false positives.

**Six pre-registered confirmatory tests:** for each of the three tasks, McNemar comparing **BanglaBERT vs the character n-gram baseline** at **severity 3** and **severity 5**. **Holm-corrected** across the family.

Everything else is **descriptive**: means, standard deviations over three seeds, bootstrap 95% CIs — and never a p-value or the word "significant". State this in §IV verbatim.

## M6.3 Native-speaker plausibility check

100 perturbed samples at severity 3, stratified **10 per noise type**, plus **10 unperturbed controls mixed in blind**. Binary rating: *"could a real person plausibly have typed this?"* Report the rate per noise type **and the control rate** — if clean text is rated implausible, the instrument is unreliable and must be redone.

Run on **real corpus text**, not the synthetic fixtures. The informal Table I review done on 2026-09-01 was a design check, not this measurement; do not conflate them in the paper.

---

# M7. WHAT THE PAPER REPORTS

- **§III** ← M0 (design decisions), M1 (definitions, incl. the CER caveat and achieved-rate table), M3 (validation), M6.3
- **§IV** ← M2 (data), M4 (models, environment), M5 (grid), M6.1–6.2 (metrics, statistics)
- **§V** ← results, per `Figure_Plan_Evaluation.md`

**Limitations must state explicitly:** layout-free N3 and why; triple conjuncts excluded; N7 single-mechanism (ZWNJ excluded) and why; N2's achieved rate below nominal at high severity; the N4 selection change (shrinking-pool disjoint-pair sampling replacing skip-on-overlap) as a fixed deviation from the original approach; any homophone set dropped on M2.4 evidence; emoji mapping coverage, **and that emoji-range coverage is measured against `emoji` 1.4.2 ≈ Unicode 13 (2021), so post-2021 emoji are outside both the reference and our ranges — the figure is an upper bound against a 2021-era reference**; synthetic noise validated against one naturally noisy corpus; single language.

---

# ⚠ FILE LOCATION — the reason this file was lost once

`METHODOLOGY.md`, `Topic4_FULL_PAPER_PLAN.md` and `Figure_Plan_Evaluation.md` must live **inside `bangla-noisebench/`** and be **committed to git**. The original was lost because it sat at the workspace root, outside the repository, so there was no version to restore.

```
bangla-noisebench/
├── METHODOLOGY.md              ← here, tracked
├── Topic4_FULL_PAPER_PLAN.md   ← here, tracked
├── Figure_Plan_Evaluation.md   ← here, tracked
├── CLAUDE.md
└── ...
```

These are specification documents. If the code is worth version control, the spec it is written against is too.
