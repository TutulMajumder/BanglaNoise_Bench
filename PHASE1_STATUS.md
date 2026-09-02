# bangla-noisebench — Phase 1 verification report

Generated 2026-09-02. Commit `81ae763` (`phase-1: noise suite complete`), tag `phase-1`.
Numbers, not prose. Regenerate the underlying reports with
`python scripts/phase1_report.py` and `python scripts/constants_sheet.py`.

---

## A. File inventory

All 19 tracked files (`git ls-files`). No data, results, notebooks, caches or
model artefacts are tracked. `results/`, `__pycache__/`, `.pytest_cache/` exist
on disk and are git-ignored.

| bytes | path | tracked |
|------:|------|:------:|
| 331 | `.gitattributes` | yes |
| 597 | `.gitignore` | yes |
| 5434 | `CLAUDE.md` | yes |
| 3358 | `CLAUDE_md_for_repo.md` | yes |
| 15184 | `Figure_Plan_Evaluation.md` | yes |
| 23727 | `METHODOLOGY.md` | yes |
| 8471 | `README.md` | yes |
| 21762 | `Topic4_FULL_PAPER_PLAN.md` | yes |
| 1082 | `noisebench/__init__.py` | yes |
| 27676 | `noisebench/bangla_maps.py` | yes |
| 32244 | `noisebench/perturbations.py` | yes |
| 2632 | `noisebench/severity.py` | yes |
| 20758 | `noisebench/validate.py` | yes |
| 1898 | `requirements.txt` | yes |
| 18757 | `scripts/constants_sheet.py` | yes |
| 34643 | `scripts/phase1_report.py` | yes |
| 144 | `tests/__init__.py` | yes |
| 8265 | `tests/bangla_samples.py` | yes |
| 21302 | `tests/test_perturbations.py` | yes |
| — | `PHASE1_STATUS.md` | yes (this file, committed separately) |

Untracked + git-ignored on disk: `results/` (regenerable reports),
`noisebench/__pycache__/`, `scripts/__pycache__/`, `tests/__pycache__/`,
`.pytest_cache/`.

---

## B. Tests

`python -m pytest -q` → **40 passed, 0 failed** (~51 s). 35 test functions;
40 cases after parametrisation.

Regression guards:

| guard | test |
|-------|------|
| seed decorrelation (N7 Bernoulli remainder independent across texts) | `test_bernoulli_remainder_is_decorrelated_across_texts` |
| seed actually changes output | `test_determinism_seed_actually_matters` |
| N8 lengthens vowels/matras only, never consonants | `test_n8_elongation_only_lengthens_vowels` |
| 25 structural assertions on `bangla_maps.py` | `test_bangla_maps_structural_assertions` |
| nukta letters ড়/ঢ়/য় never torn at U+09BC | `test_nukta_letters_never_torn_apart` |
| whole-cluster ops keep nukta-letter count intact | `test_nukta_whole_cluster_ops_keep_letters_intact` |
| N3 substitutes a nukta letter as a whole unit | `test_nukta_letter_is_substituted_whole` |
| dropped ব/ভ homophone set never fires | `test_n5_dropped_b_bh_set_never_fires` |
| `0 ≤ n_applied ≤ n_requested ≤ n_eligible`; equality except N2/N4 | `test_n_requested_vs_applied_contract` |
| achieved fraction ≈ nominal for every type except N2 | `test_achieved_rate_matches_nominal_for_unguarded_types` |
| N8 submodes cover all three mechanisms; N7 is `{drop_hasanta}` | `test_stats_submodes_cover_mechanisms` |
| `perturb` output == `perturb_with_stats` output | `test_perturb_wrapper_matches_stats_output` |
| count rule is not a silent no-op on short text | `test_n_to_perturb_is_not_a_silent_noop_on_short_text` |

---

## C. 9×5 CER table (post N4 fix)

Mean corpus CER, corpus n=200 (random 3-sentence texts), seeds {42, 1337, 2024}.
Every row strictly increasing (M3 criterion 2 = PASS).

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 |
|------------|-----:|-----:|-----:|-----:|-----:|:----:|:----:|:----:|:----:|
| char_insert       | 0.0255 | 0.0509 | 0.1024 | 0.1785 | 0.2554 | 2.00 | 2.01 | 1.74 | 1.43 |
| char_delete       | 0.0373 | 0.0748 | 0.1466 | 0.2500 | 0.3415 | 2.01 | 1.96 | 1.71 | 1.37 |
| char_substitute   | 0.0353 | 0.0688 | 0.1387 | 0.2432 | 0.3471 | 1.95 | 2.02 | 1.75 | 1.43 |
| char_transpose    | 0.0512 | 0.1006 | 0.1984 | 0.3404 | 0.4685 | 1.96 | 1.97 | 1.72 | 1.38 |
| homophone_confuse | 0.0196 | 0.0393 | 0.0777 | 0.1329 | 0.1858 | 2.01 | 1.98 | 1.71 | 1.40 |
| matra_perturb     | 0.0246 | 0.0492 | 0.0962 | 0.1665 | 0.2363 | 2.00 | 1.96 | 1.73 | 1.42 |
| conjunct_split    | 0.0061 | 0.0119 | 0.0241 | 0.0421 | 0.0606 | 1.95 | 2.03 | 1.75 | 1.44 |
| elongate          | 0.0228 | 0.0469 | 0.0918 | 0.1613 | 0.2308 | 2.06 | 1.96 | 1.76 | 1.43 |
| whitespace_error  | 0.0375 | 0.0750 | 0.1498 | 0.2622 | 0.3744 | 2.00 | 2.00 | 1.75 | 1.43 |

Targets: **2.00 / 2.00 / 1.75 / 1.43**. Tolerance ±10% →
s2/s1 ∈ [1.80, 2.20], s3/s2 ∈ [1.80, 2.20], s4/s3 ∈ [1.575, 1.925], s5/s4 ∈ [1.287, 1.573].

**No row deviates > 10% on any ratio.** The largest gaps are the two
edit-heavy types at the top severities — char_delete s5/s4 = 1.37 (−4.2%) and
char_transpose s5/s4 = 1.38 (−3.5%) — the expected CER saturation as edits
start to overlap within a token (documented in M1.5's interpretation caveat).

char_transpose row before the N4 fix (for reference): 0.0492 / 0.0973 / 0.1802 /
0.2894 / 0.3726, s5/s4 = 1.29. The fix lifts it into the pack.

---

## D. Achieved vs nominal perturbation rate (post N4 fix)

corpus n=200 × seeds {42, 1337, 2024}, cells with ≥1 eligible unit.
`req/nom` = mean `n_requested / (n_eligible·frac)` (count-rule bias).
`app/req` = mean `n_applied / n_requested` (guard-induced loss).
`achieved` = mean `n_applied / n_eligible`.

| noise_type | s | nominal | req/nom | app/req | achieved |
|------------|--:|--------:|--------:|--------:|---------:|
| char_insert | 1–5 | 0.05–0.50 | 0.997–1.003 | 1.000 | 0.0500 / 0.0997 / 0.2005 / 0.3495 / 0.5001 |
| **char_delete** | 1 | 0.05 | 0.991 | 0.998 | 0.0495 |
| **char_delete** | 2 | 0.10 | 1.002 | 0.991 | 0.0993 |
| **char_delete** | 3 | 0.20 | 0.997 | 0.975 | 0.1945 |
| **char_delete** | 4 | 0.35 | 1.000 | **0.947** | **0.3318** (−5.2%) |
| **char_delete** | 5 | 0.50 | 0.999 | **0.906** | **0.4532** (−9.4%) |
| char_substitute | 1–5 | 0.05–0.50 | 0.992–1.016 | 1.000 | 0.0508 / 0.0992 / 0.1999 / 0.3503 / 0.5001 |
| char_transpose | 1 | 0.05 | 1.011 | 1.000 | 0.0506 |
| char_transpose | 2 | 0.10 | 0.998 | 1.000 | 0.0998 |
| char_transpose | 3 | 0.20 | 0.998 | 1.000 | 0.1997 |
| char_transpose | 4 | 0.35 | 1.000 | 1.000 | 0.3501 |
| char_transpose | 5 | 0.50 | 1.000 | **1.000** | **0.4996** (−0.1%) |
| homophone_confuse | 1–5 | 0.05–0.50 | 0.989–1.003 | 1.000 | 0.0494 / 0.0999 / 0.2003 / 0.3510 / 0.5004 |
| matra_perturb | 1–5 | 0.05–0.50 | 0.996–1.012 | 1.000 | 0.0506 / 0.0996 / 0.1993 / 0.3497 / 0.5000 |
| conjunct_split | 1–5 | 0.05–0.50 | 0.988–1.020 | 1.000 | 0.0510 / 0.0988 / 0.1994 / 0.3485 / 0.5011 |
| elongate | 1–5 | 0.05–0.50 | 0.988–1.002 | 1.000 | 0.0494 / 0.1002 / 0.1992 / 0.3501 / 0.5001 |
| whitespace_error | 1–5 | 0.05–0.50 | 1.000–1.002 | 1.000 | 0.0501 / 0.1002 / 0.2000 / 0.3501 / 0.5001 |

- **Count rule unbiased everywhere:** `req/nom` ∈ [0.988, 1.020] across all 45 cells.
- **N4 char_transpose now hits nominal:** `app/req` = 1.000 at s1–s4, 0.999 at
  s5 (achieved 0.4996 vs 0.50). Confirmed. The shrinking-pool selection empties
  before *k* only in a vanishing fraction of texts.
- **N2 char_delete still short**, as documented in M1.3: `app/req` falls to
  0.906 at s5 (achieved 0.4532). Unavoidable — the guard forbids deleting every
  cluster of a one-cluster token, and any run where all of a token's clusters
  are selected must drop one. Reported as the achieved fraction, not fixed.

---

## E. M3 acceptance criteria — with numbers

| # | criterion | result | evidence |
|---|-----------|:------:|----------|
| 1 | Determinism | **PASS** | 50 texts × 9 types × 5 severities × 2 repeat calls, all byte-identical. Also `test_determinism_seed_actually_matters`: seed 7 ≠ seed 8 output. |
| 2 | Monotonicity | **PASS** | all 9 CER rows strictly increasing (§C). |
| 3 | Validity | **PASS** | 80 texts × 9 types × {s1,s3,s5} × 3 seeds → `is_valid_output` true for every output (NFC-idempotent, no U+FFFD, no orphan nukta, no leading combining mark). |
| 3 | ো/ৌ survive NFC round trip | **PASS** | 12 probe texts, ো/ৌ cluster counts unchanged. |
| 3 | ো/ৌ survive BanglaBERT-normalizer round trip | **PASS** | same 12 probes through `normalize()`. |
| 4 | Non-identity at severity 5 (≥95% of eligible texts change) | **PASS** | see below. |

Non-identity at s5 — changed / eligible (corpus n=200 × 3 seeds):

| noise_type | changed / eligible | rate |
|------------|-------------------:|-----:|
| char_insert | 600 / 600 | 100.0% |
| char_delete | 600 / 600 | 100.0% |
| char_substitute | 600 / 600 | 100.0% |
| char_transpose | 600 / 600 | 100.0% |
| homophone_confuse | 600 / 600 | 100.0% |
| matra_perturb | 600 / 600 | 100.0% |
| conjunct_split | 570 / 591 | 96.4% |
| elongate | 600 / 600 | 100.0% |
| whitespace_error | 600 / 600 | 100.0% |

conjunct_split is the floor: 9 of 200 texts have no `C+HASANTA+C` junction
(eligible 591 not 600), and at s5 with ~4–5 junctions per eligible text a
handful land on identical-after-split output. 96.4% > 95%.

---

## F. Structural assertions

`scripts/constants_sheet.py` → **25 / 25 PASS** (also run in
`test_bangla_maps_structural_assertions`). Pool sizes explicit:
**BANGLA_CONSONANTS = 35, BANGLA_INDEPENDENT_VOWELS = 11, BANGLA_DIGITS = 10.**

| # | check | result |
|--:|-------|:------:|
| 1 | consonant pool size == 35 | PASS (got 35) |
| 2 | independent_vowel pool size == 11 | PASS (got 11) |
| 3 | digit pool size == 10 | PASS (got 10) |
| 4 | IN_CLASS_POOLS match BANGLA_* inventories | PASS |
| 5 | no matra / hasanta in any IN_CLASS_POOL | PASS |
| 6 | no matra / hasanta in INSERTABLE_CHARS | PASS |
| 7 | no chandrabindu (U+0981) in INSERTABLE_CHARS | PASS |
| 8 | no digits in INSERTABLE_CHARS | PASS |
| 9 | IN_CLASS_POOLS mutually disjoint | PASS |
| 10 | every IN_CLASS_POOL has ≥ 2 members | PASS |
| 11 | HOMOPHONE_SETS members distinct across sets | PASS |
| 12 | every HOMOPHONE set has ≥ 2 members | PASS |
| 13 | ি/ী and ু/ূ present as HOMOPHONE sets (M0 Q3) | PASS |
| 14 | ই/ঈ, উ/ঊ, র/ড়/ঢ় kept (author-confirmed) | PASS |
| 15 | ব/ভ removed from HOMOPHONE_SETS | PASS |
| 16 | N6_DROP_TARGETS == all matras + hasanta + chandrabindu | PASS |
| 17 | N6_ALTER_PAIRS chars ⊆ N6_DROP_TARGETS | PASS |
| 18 | N6_ALTER_PAIRS == {ি↔ী, ু↔ূ} only (M0 Q5 revised) | PASS |
| 19 | MATRA_TO_INDEPENDENT_VOWEL keys are all dependent vowel signs | PASS |
| 20 | MATRA_TO_INDEPENDENT_VOWEL values are all independent vowels | PASS |
| 21 | MATRA_TO_INDEPENDENT_VOWEL == {ু→উ, ূ→ঊ, ি→ই, ী→ঈ, ো→ও} only | PASS |
| 22 | HOMOPHONE_SETS_REJECTED includes the ten aspiration pairs | PASS |
| 23 | HOMOPHONE_SETS and HOMOPHONE_SETS_REJECTED disjoint | PASS |
| 24 | every COMMON_CONJUNCTS entry is `<consonant> HASANTA <consonant>` | PASS |
| 25 | every string constant NFC-idempotent | PASS |

(METHODOLOGY M3 previously said "19"; updated to 25 in this commit — the count
grew as decisions were locked in, no assertions were removed.)

---

## G. Normalizer

| field | value |
|-------|-------|
| package | `csebuetnlp/normalizer` (BanglaBERT's own preprocessing pipeline) |
| **not** | `bnunicodenormalizer` (different BUET project, OCR/ASR) |
| import | `from normalizer import normalize` |
| pin | `normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9` |
| installed | version `0.0.1` at commit `d405944` (verified via `direct_url.json`) |
| call Phase 3 MUST use | `noisebench.validate.normalize_banglabert(text)` → `normalize(text)` with **all defaults** |
| optional at import | yes — `normalizer_available()` gates the two survival helpers; `requirements.txt` still pins it (affects reported numbers, not test-only) |

Synthetic-corpus survival (severity 3, corpus n=200 × 3 seeds):

| noise_type | post-normalizer CER | touch_rate |
|------------|--------------------:|-----------:|
| char_insert | 0.1024 | 0.0% |
| char_delete | 0.1466 | 0.0% |
| char_substitute | 0.1387 | 0.0% |
| char_transpose | 0.1984 | 0.0% |
| homophone_confuse | 0.0777 | 0.0% |
| matra_perturb | 0.0962 | 0.0% |
| conjunct_split | 0.0241 | 0.0% |
| elongate | 0.0918 | 0.0% |
| whitespace_error | 0.1498 | 0.0% |

**Caveat (carried in the report and M5.4):** touch_rate = **0.0%** on every
type — `normalize()` with defaults is a complete no-op on clean synthetic
Bangla (no URLs, emails, or punctuation-spacing issues). The post-normalizer
CER row is therefore identical to the plain CER row and proves nothing about
survival *yet*. It becomes meaningful only on the real, messier Phase 2
corpora, where M5.4's decision rule applies: if touch_rate stays < ~5%, drop
the R6 normalizer ON/OFF condition and report the null.

---

## H. Emoji

| field | value |
|-------|-------|
| detection (ships) | hand-rolled `bangla_maps.EMOJI_CODEPOINT_RANGES` (9 inclusive ranges); `noisebench/` never imports `emoji` |
| coverage measurement | `noisebench.validate.emoji_range_coverage(texts)` — optional `emoji` import, version-tolerant (`emoji_list` / `emoji_lis` / `get_emoji_regexp`) |
| wired into | `scripts/phase1_report.py` (provisional section); intended target is Phase 2 real corpora (M0.2 Q9, M2.4 item 3) |
| `emoji` version | **1.4.2** (pinned transitively by the normalizer; ≈ Unicode 13 / 2021) |
| provisional number | on the 58 sample sentences: **11 emoji occurrences, 100.0% coverage** (any and full), **0** over-match code points |

**Caveat:** the reference list is `emoji` 1.4.2 (Unicode ~13, 2021). Emoji
added after 2021 are absent from *both* the reference and our ranges, so the
coverage figure is an **upper bound against a 2021-era reference**, not against
current Unicode. Sample size here (11 occurrences) is far too small to act on —
the real figure comes from Phase 2.

---

## I. Dependencies

`requirements.txt` verbatim (comments trimmed to the pins):

```
regex==2024.11.6
normalizer @ git+https://github.com/csebuetnlp/normalizer@d405944dde5ceeacb7c2fd3245ae2a9dea5f35c9
pytest>=7.4
```

Resolved in the local dev environment (Windows, Python 3.12.5):

| package | resolved | how pinned |
|---------|----------|------------|
| `regex` | 2024.11.6 | `requirements.txt` `==` |
| `normalizer` | `0.0.1` @ `d405944` | `requirements.txt` git commit |
| `emoji` | **1.4.2** | transitive; `normalizer` metadata `emoji==1.4.2` |
| `ftfy` | **6.0.3** | transitive; `normalizer` metadata `ftfy==6.0.3` |
| `pytest` | 9.1.1 | `requirements.txt` `>=7.4` |

`normalizer` metadata `requires` = `['regex', 'emoji==1.4.2', 'ftfy==6.0.3']`
(verified via `importlib.metadata`). Kaggle runs Python **3.12.13** (verified
2026-09-01); the 3.10+ target is a portability choice, not a Kaggle
constraint. **To confirm on Kaggle:** `pip freeze | grep -Ei '^(regex|emoji|ftfy|pytest)=='`
and record in README "Resolved dependency versions".

---

## J. Git

| field | value |
|-------|-------|
| repo root | `bangla-noisebench/` (branch `main`) |
| commits | `9192b86` phase-1: noise suite, validation, reports, spec docs · `81ae763` phase-1: noise suite complete |
| tag | **`phase-1` → `81ae763`** |
| this report | committed separately on top of `81ae763` |
| tracked | 19 source/spec files + this report |
| **not tracked** | verified: no match for `results/`, `*.csv`, `*.pt`, `*.bin`, `*.safetensors`, `data/` in `git ls-files` |
| `.gitignore` | `data/*` with `!data/splits` (M2.2 split manifests stay tracked); `results/`, `results.csv`, model-artefact extensions, `__pycache__/`, `.pytest_cache/` |
| `.gitattributes` | `* text=auto eol=lf` (system `core.autocrlf=true` would otherwise CRLF-ify on checkout) |
| spec docs in git | `METHODOLOGY.md`, `Topic4_FULL_PAPER_PLAN.md`, `Figure_Plan_Evaluation.md` — all tracked, all inside `bangla-noisebench/` |

---

## K. Deviations from METHODOLOGY.md (as restored)

1. **N4 selection mechanism.** M1.3's general line says "Selection is
   `rng.sample(eligible, k)` — without replacement." N4 does **not** use that;
   it draws *k* disjoint adjacent pairs from a shrinking pool (M1.3's own N4
   paragraph now describes this exactly). This is deliberate and is the point
   of the N4 fix — `rng.sample` of *k* pair positions is what produced the
   overlap collisions. No other type deviates from `rng.sample`.
2. **Structural-assertion count.** Was "19" in M3; now 25. Updated in this
   commit. No assertions removed — the count grew as M0.3 decisions were
   locked in. (§F.)
3. **`emoji_range_coverage` is new code** in `validate.py`, not named in the
   restored M1/M2 text beyond M0.2 Q9's "measure it". It is measurement-only
   and additive; M0.2 Q9 and M2.4 item 3 now reference it by name (edited in
   this commit).
4. **`.gitattributes`** added (LF normalization) — infra, not covered by
   METHODOLOGY.
5. **N2 achieved-rate table** in M1.3 and the **emoji Unicode-13 caveat** in
   M0.2 Q9 / M2.4 / M7 were added/edited in this commit at your instruction;
   listed here so you can reconcile against your pristine copy.

Nothing in the code contradicts a locked M0 decision, the severity scale, the
noise taxonomy, or the metric set.

---

## L. Carried into Phase 2

| item | METHODOLOGY ref | note |
|------|-----------------|------|
| Regenerate the 9×5 CER table on **real corpus samples**; report both synthetic and real | M2.4 (1), M3 §2 | the synthetic table here is a development check |
| **N5 per-set firing counts** — `n_eligible` and `n_applied` per `HOMOPHONE_SET` per dataset (BD-SHS / SentNoB / BanFakeNews) | M2.4 (2) | `n_applied` sums off `PerturbStats.submodes` (keyed per set); `n_eligible` needs a grouping pass over `_homophone_occurrences`. **ই/ঈ and উ/ঊ are the two sets under review** — drop with evidence if eligible ≈ 0 on real text, note in Limitations |
| **Emoji coverage on real corpora** — `EMOJI_CODEPOINT_RANGES` vs `emoji` 1.4.2 | M0.2 Q9, M2.4 (3) | `validate.emoji_range_coverage` is wired and tested; widen ranges if coverage < ~90%. Report as an upper bound vs a 2021-era reference |
| **Normalizer touch_rate on real corpora → R6 go/no-go** | M5.4, M2.4 (4) | 0.0% on synthetic. If < ~5% on real data, drop the normalizer ON/OFF condition and report the null; reclaim ~4 GPU-h |
| **M6.3 native-speaker plausibility rating** | M6.3, M7 | 100 samples at s3 (10/noise type) + 10 blind clean controls; on real corpus text, not fixtures |
| Table II dataset statistics (n, class balance, cluster length, emoji rate, dupes/short removed) | M2.3 | emoji rate decides which task hosts N10 |
| Split manifests committed to `data/splits/` | M2.2 | `.gitignore` now permits this path |
| Native-speaker review of every `# TODO(verify):` constant in `bangla_maps.py` | CLAUDE.md, README | still pending; blocks nothing in code but gates the linguistic claims |
| Limitations paragraph: layout-free N3; triple conjuncts excluded; N7 single-mechanism; N2 achieved rate; **N4 selection change**; dropped homophone sets; emoji coverage + Unicode-13 caveat; one naturally-noisy corpus; single language | M7 | assembled from the above |

**Phase 2 not started.** This report is the stopping point.
