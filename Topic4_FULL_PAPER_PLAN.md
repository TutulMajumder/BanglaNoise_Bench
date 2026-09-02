# FULL PAPER PLAN — Bangla Text Classification Under Realistic Noise
**Target: ICREST'27 (AIUB), submission 30 September 2026 · Plan written 30 August 2026 · 31 days remaining**

---

## 0. THE ONE-SENTENCE CLAIM

> Bangla text classifiers are evaluated on cleaned text; under the orthographic and typographic noise that real Bangla input actually contains, their performance degrades substantially, the ranking between models changes, and the cheap mitigations practitioners apply have never been compared — so we build a severity-graded noise benchmark, measure the degradation across three tasks, compare the mitigations, and validate the synthetic noise against a naturally noisy corpus.

Everything in the paper serves that sentence. If an experiment doesn't support it, cut it.

**Final title:** *"Bangla Text Classifiers Under Realistic Noise: A Multi-Task Robustness Benchmark and an Evaluation of Low-Cost Mitigations"*

**Repo name:** `bangla-noisebench`

---

## 1. ⚠️ THE METHODOLOGICAL TRAP — READ THIS FIRST

BanglaBERT (csebuetnlp) **requires a text normalizer** as part of its official preprocessing pipeline (the `normalizer` package; related tools include `bnunicodenormalizer` and `banglanlptoolkit`). That normalizer performs Unicode normalization and canonicalisation.

**The danger:** the normalizer may silently *undo* some of the noise you inject — particularly Unicode-level perturbations, conjunct decomposition and some diacritic manipulations. If you inject noise and then normalize, you may measure a much smaller degradation than really exists, and your headline result would be an artefact of your own pipeline.

**How to handle it — this becomes a contribution, not just a fix:**

1. Run every noise evaluation in **two conditions**: `normalizer ON` and `normalizer OFF`.
2. Report the gap between them explicitly as its own small result: *"how much of Bangla model robustness is actually contributed by the preprocessing normalizer rather than the model?"*
3. In the noise-suite unit tests, assert that for each noise type, applying the normalizer does **not** restore the text to the original (measure post-normalizer character error rate > 0). Any noise type the normalizer fully reverses must be flagged in the paper.

This is one of the most defensible things in the paper. No prior Bangla robustness work reports it, and it directly extends the BLP-2025 study.

---

## 2. PAPER STRUCTURE (IEEE two-column, 6 pages)

| § | Section | Pages | Content |
|---|---|---|---|
| I | Introduction | 0.75 | Real Bangla is noisy → classifiers evaluated clean → three unanswered questions → our four contributions, listed as bullets |
| II | Related Work | 0.75 | (a) Bangla text classification & encoders; (b) robustness/noise in NLP; (c) **the gap**: BLP-2025's stated exclusions + Tisha et al.'s emoji finding + SentNoB's classical-beats-neural result |
| III | Noise Taxonomy & Benchmark | 1.0 | **The contribution.** Ten noise types with Bangla examples (Table I), severity definition, determinism/seeding, normalizer interaction, unit-test validation |
| IV | Experimental Setup | 0.6 | Datasets (Table II), models, training config, metrics, statistics protocol |
| V | Results | 1.75 | R1 clean reproduction → R2 degradation curves → R3 noise-type ranking → R4 mitigations → R5 synthetic-vs-natural validation → R6 normalizer effect |
| VI | Discussion & Limitations | 0.4 | What the results mean for practitioners; honest limitations |
| VII | Conclusion | 0.2 | Three sentences + artefact release |
| — | References | 0.5 | 20–25 refs |

**Write §III before §V.** The taxonomy table is the paper's identity and it forces you to finalise the design.

---

## 3. FIGURES AND TABLES (final list — do not add more)

| ID | Type | Content | Why it earns space |
|---|---|---|---|
| **Table I** | Table | Noise taxonomy: 10 types, definition, **real Bangla example (original → perturbed)** | Reviewers remember this table; it makes the contribution concrete |
| **Table II** | Table | Dataset statistics: size, splits, classes, class balance, emoji rate, mean length |
| **Table III** | Table | Clean-data baselines, 3 tasks × 5 models, mean ± std over 3 seeds | Proves your pipeline reproduces published numbers |
| **Figure 1** | 3-panel line plot | Macro-F1 vs severity (1–5), one panel per task, one line per model | The headline visual |
| **Figure 2** | Heatmap | Degradation Δ per (noise type × model), averaged over tasks | Answers "which noise matters" at a glance |
| **Table IV** | Table | Mean noise F1 + fitted degradation slope, ranked — with clean rank shown beside robustness rank | **Where a rank inversion becomes visible** |
| **Table V** | Table | Mitigation comparison: none / noise-aug (matched) / noise-aug (mismatched) / char n-gram / emoji-aware | The practical payoff |
| **Figure 3** | Scatter + Spearman ρ | Model robustness rank on synthetic noise vs rank on natural SentNoB noise | Validates the benchmark; pre-empts the main objection |

Nine artefacts in six pages is already dense. If space runs out, merge Table III into Figure 1's caption.

---

## 4. THE NOISE TAXONOMY (full specification)

Ten types, five severity levels each. Severity = **fraction of eligible units perturbed**: `s1=5%, s2=10%, s3=20%, s4=35%, s5=50%`.

### Group A — Generic character-level (4 types)
| # | Type | Operation |
|---|---|---|
| N1 | Character insertion | Insert a random Bangla character at a random position |
| N2 | Character deletion | Delete a random character |
| N3 | Character substitution | Replace with a **keyboard-adjacent** character (Avro/phonetic layout or Bijoy — pick one, document it) |
| N4 | Character transposition | Swap two adjacent characters |

### Group B — Bangla-specific orthographic (4 types) ← **this is what makes the paper Bangla, not generic**
| # | Type | Operation | Example |
|---|---|---|---|
| N5 | **Homophone confusion** | Swap within confusable sets — the classic Bangla misspelling classes: `শ/ষ/স`, `ন/ণ`, `ই/ঈ`, `উ/ঊ`, `র/ড়/ঢ়`, `জ/য`, `ব/ভ` | কষ্ট → কস্ট |
| N6 | **Matra (diacritic) drop/change** | Remove or alter a vowel sign: `ে ি ী ু ূ ো ৌ ্` | ভালো → ভাল |
| N7 | **Conjunct (juktakkhor) decomposition** | Split a conjunct into components + hasanta, or remove hasanta | ক্ষ → ক্‌ষ / কষ |
| N8 | **Character elongation / repetition** | Repeat a character 2–4× for emphasis | খুব → খুউব, ভালো → ভালোওও |

*Group B is the most defensible part of the suite. These are not arbitrary corruptions — they are the actual error classes Bangla writers produce, and no prior work has measured their effect.*

### Group C — Surface-level (2 types)
| # | Type | Operation |
|---|---|---|
| N9 | Whitespace errors | Split a word in two, or merge two adjacent words |
| N10 | **Emoji manipulation** | Three conditions: `keep` (control) / `remove` / `replace with Bangla textual description` |

**N10 is handled differently** — it is not a severity-graded corruption but a three-way preprocessing condition, run as its own experiment (motivated by Tisha et al.'s ~12% emoji finding).

### Implementation requirements (non-negotiable)
- **Deterministic and seeded.** `perturb(text, noise_type, severity, seed)` must be reproducible bit-for-bit.
- **Unit-tested monotonicity:** mean character error rate (CER) vs original must increase strictly with severity for every noise type. Write this test on day 2.
- **Normalizer-survival test:** for each noise type, assert CER > 0 *after* applying the BanglaBERT normalizer (see §1).
- **Native-speaker validation:** sample 100 perturbed examples at severity 3, have a native speaker (you) rate each as "plausible real-world error" yes/no. **Report that percentage in the paper.** This single number kills most "your noise is unrealistic" objections.
- Applied to **test sets only** for robustness measurement; **training sets only** for the augmentation arm. Never both in the same run.

---

## 5. DATASETS — acquisition and day-1 checks

| Dataset | Task | Source | Day-1 check |
|---|---|---|---|
| **BD-SHS** | Hate speech (binary) | github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media ; Kaggle `naurosromim/bdshs` | Confirm 50,200+; record class balance |
| **SentNoB** | Sentiment (3-class: 0=neutral, 1=positive, 2=negative) | github.com/KhondokerIslam/SentNoB ; Kaggle `cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts` | **Exact split sizes are not documented online — record Train/Val/Test counts yourself and report them.** Check licence on the Kaggle page |
| **BanFakeNews** | Fake news (binary) | github.com/Rowan1697/FakeNews ; Kaggle `cryptexcode/banfakenews` | 48K authentic / 1K fake — **use the labelled balanced subsets (7K auth / 1K fake) for main runs**, full set as secondary |

**Also compute on day 1 and put in Table II:** emoji rate per dataset (fraction of samples containing ≥1 emoji), mean token length, and duplicate rate. The emoji rate determines which task the N10 experiment runs on — pick the highest.

---

## 6. MODELS AND TRAINING CONFIGURATION

| Model | HF identifier / method | Role |
|---|---|---|
| BanglaBERT | `csebuetnlp/banglabert` | Primary subject; the model shown to lose ~60% F1 under perturbation |
| BanglishBERT | `csebuetnlp/banglishbert` | Code-mixed-aware competitor |
| XLM-R base | `xlm-roberta-base` | Multilingual arm |
| mBERT | `bert-base-multilingual-cased` | Second multilingual reference |
| **Char n-gram + LinearSVC / LogReg** | scikit-learn TF-IDF, `analyzer='char_wb'`, ngram 2–5 | Expected most robust — SentNoB precedent |

**Fixed training config for all transformers** (document it; identical across models is what makes comparison fair):
`max_len=128, batch=32, lr=2e-5, epochs=4, AdamW, linear warmup 10%, fp16, early stopping on val macro-F1, seeds {42, 1337, 2024}`

---

## 7. EXPERIMENT MATRIX

| ID | Experiment | Runs | Cost |
|---|---|---|---|
| **R1** | Clean baselines | 4 transformers × 3 tasks × 3 seeds = **36 fine-tunes** + classical (CPU) | ~10–14 GPU-h |
| **R2** | Noise grid (inference only) | 5 models × 3 tasks × 9 noise types × 5 severities × 3 seeds ≈ 2,000 eval passes | ~4–6 GPU-h |
| **R3** | Noise-type ranking | Derived from R2 — no new runs | 0 |
| **R4a** | Noise-augmented training (matched) | 2 best models × 3 tasks × 3 seeds = **18 fine-tunes** | ~5 GPU-h |
| **R4b** | Noise-augmented (mismatched: train on N1–N4, test on N5–N8) | Reuses R4a checkpoints — inference only | ~1 GPU-h |
| **R5** | Emoji condition (N10) | 3 conditions × 2 models × 1 task × 3 seeds = **18 fine-tunes** | ~4 GPU-h |
| **R6** | Normalizer ON/OFF | Reuses R2 checkpoints — inference only, doubles R2 eval | ~4 GPU-h |
| **R7** | Synthetic vs natural validation | Derived — rank correlation between R2 rankings and clean-SentNoB rankings | 0 |

**Total ≈ 30–35 GPU-hours.** Above my earlier 10–15h estimate because of the normalizer condition and the emoji arm. Still inside Kaggle's 30h/week quota across two weeks. If you must cut, drop R6 to a single task.

---

## 8. METRICS AND STATISTICS

- **Primary:** macro-F1 (all tasks)
- BanFakeNews additionally: **PR-AUC and MCC** (48:1 imbalance makes ROC-AUC optimistic and accuracy meaningless)
- Per-class precision/recall; balanced accuracy
- **Mean Noise F1 (MNF):** average macro-F1 across all noise types and severities — one number per (model, task)
- **Degradation slope:** fitted linear coefficient of macro-F1 vs severity index
- **Rank inversion indicator:** Spearman ρ between clean-F1 ranking and MNF ranking. ρ < 1 is the finding.
- ECE under noise (secondary)
- Efficiency: params, model size (MB), inference latency ms/sample (CPU and GPU)

**Statistics protocol:** 3 seeds everywhere; report mean ± std in every table; **McNemar's test** for paired model comparisons on the same test set; bootstrap 95% CIs for the headline degradation numbers. State explicitly that differences within one standard deviation are not claimed as findings.

---

## 9. CODE ARCHITECTURE

```
bangla-noisebench/
├── README.md
├── requirements.txt
├── noisebench/
│   ├── __init__.py
│   ├── perturbations.py      # N1–N10, deterministic, seeded  ← THE ARTEFACT
│   ├── bangla_maps.py        # homophone sets, matra list, conjuncts, keyboard adjacency
│   ├── severity.py           # severity → fraction mapping
│   └── validate.py           # monotonicity + normalizer-survival tests
├── tests/
│   └── test_perturbations.py # unit tests — write these on day 2
├── data/
│   ├── prepare.py            # download, clean, split (seeded), stats for Table II
│   └── splits/               # committed split files
├── experiments/
│   ├── train.py              # args: --model --task --seed --augment
│   ├── evaluate_noise.py     # R2/R6 grid, writes results.csv
│   ├── mitigations.py        # R4, R5
│   └── configs/
├── analysis/
│   ├── tables.py             # emits LaTeX for Tables I–V
│   ├── figures.py            # emits Figures 1–3
│   └── stats.py              # McNemar, bootstrap, Spearman
└── results/
    └── results.csv           # one row per (model, task, seed, noise, severity, normalizer, metric)
```

**One flat `results.csv` with one row per evaluation.** Every table and figure is a groupby on it. Do not scatter results across notebooks — that is how students lose two days in week four.

---

## 10. DAY-BY-DAY SCHEDULE (30 Aug – 30 Sep 2026)

### Week 1 — Foundation (30 Aug – 5 Sep)
| Day | Date | Task | Done when |
|---|---|---|---|
| 1 | 30 Aug | Download all 3 datasets; record exact sizes, splits, class balance, emoji rate; check licences | Table II drafted with real numbers |
| 2 | 31 Aug | Write `perturbations.py` for N1–N4 + unit tests (monotonic CER) | Tests pass |
| 3 | 1 Sep | Write N5–N8 (Bangla-specific) + `bangla_maps.py` | Tests pass; 20 examples printed for Table I |
| 4 | 2 Sep | N9, N10; **normalizer-survival test**; native-speaker plausibility rating of 100 samples | Plausibility % recorded |
| 5 | 3 Sep | `train.py` + `prepare.py`; seeded splits committed | One end-to-end run completes |
| 6 | 4 Sep | **FREEZE EXPERIMENT LIST.** Launch R1 clean baselines | R1 running |
| 7 | 5 Sep | R1 continues; draft §III (Noise Taxonomy) | §III first draft done |

### Week 2 — Core results (6–12 Sep)
| Day | Date | Task | Done when |
|---|---|---|---|
| 8 | 6 Sep | R1 finishes; verify BD-SHS ≈ published F1 range | Table III drafted |
| 9 | 7 Sep | Write `evaluate_noise.py`; smoke-test on one model | Grid runs end-to-end |
| 10–11 | 8–9 Sep | **Run R2 full noise grid** | `results.csv` populated |
| 12 | 10 Sep | Figure 1 + Figure 2 first versions | Headline visual exists |
| 13 | 11 Sep | R3 analysis; Table IV; **check for rank inversion** | Know whether you have the key result |
| 14 | 12 Sep | Draft §I Introduction + §II Related Work | Half the prose exists |

### Week 3 — Mitigations & validation (13–19 Sep)
| Day | Date | Task |
|---|---|---|
| 15–16 | 13–14 Sep | R4a + R4b (noise-augmented, matched/mismatched) |
| 17 | 15 Sep | R5 emoji conditions |
| 18 | 16 Sep | R6 normalizer ON/OFF |
| 19 | 17 Sep | R7 synthetic-vs-natural; Figure 3 |
| 20 | 18 Sep | Table V; efficiency measurements |
| 21 | 19 Sep | **All experiments complete.** Buffer day for reruns |

### Week 4 — Writing (20–26 Sep)
| Day | Date | Task |
|---|---|---|
| 22 | 20 Sep | Finalise all figures/tables to publication quality |
| 23 | 21 Sep | Write §V Results in full |
| 24 | 22 Sep | Write §IV Setup + §VI Discussion/Limitations |
| 25 | 23 Sep | Write §VII Conclusion + abstract; complete bibliography |
| 26 | 24 Sep | Full read-through; fix §I to match actual findings |
| 27 | 25 Sep | **Send complete draft to supervisor** |
| 28 | 26 Sep | Clean the repo; write README; prepare artefact release |

### Final days (27–30 Sep)
| Day | Date | Task |
|---|---|---|
| 29 | 27 Sep | Supervisor feedback arrives; start revisions |
| 30 | 28 Sep | Revisions; IEEE template compliance; page-limit trim |
| 31 | 29 Sep | Final proofread; plagiarism/format check; **submit** |
| 32 | 30 Sep | **Deadline — buffer only. Do not plan to work today.** |

**Two hard gates:** experiment list frozen 4 Sep; all experiments done 19 Sep. Miss either and start cutting from §11.

---

## 11. CUT LIST (in order, if you fall behind)

1. **R6 normalizer ON/OFF** → reduce to one task instead of three
2. **BanFakeNews** → drop the task entirely (least noisy domain, worst imbalance); paper survives on two tasks
3. **R5 emoji arm** → report as future work
4. **mBERT** → drop one multilingual model
5. **R4b mismatched augmentation** → drop, but only last; it is what makes R4 non-trivial

**Never cut:** the noise suite itself, R1, R2, or R7 (synthetic-vs-natural validation). Those four are the paper.

---

## 12. RISKS

| Risk | Mitigation |
|---|---|
| **Normalizer erases your noise** (§1) | Detected by the normalizer-survival unit test on day 4, not in week 3. Any reversed noise type is flagged and reported rather than silently weakening results. |
| **"Synthetic noise isn't real"** | Two defences already in the design: the native-speaker plausibility percentage (day 4) and the SentNoB rank-correlation validation (R7). |
| **Degradation is uniform and uninteresting** | Then the finding is in R4b (does augmentation generalise across noise types?) and R6 (how much robustness comes from preprocessing rather than the model). Both are independent of the size of the degradation. |
| **BanFakeNews imbalance distorts everything** | Use the balanced labelled subsets for main runs; PR-AUC and MCC, never accuracy; it is also cut #2. |
| **Conjunct/matra manipulation produces invalid Unicode** | Test on day 3 with a rendering check; if a specific operation produces garbage, restrict it to a validated subset and document the restriction. |

---

## 13. REPRODUCIBILITY CHECKLIST (do these; they cost hours and buy credibility)

- [ ] Split files committed to the repo, seeded
- [ ] `requirements.txt` with pinned versions
- [ ] All 3 seeds reported, never a single run
- [ ] `results.csv` released with the paper
- [ ] Noise suite released under a permissive licence (MIT)
- [ ] Dataset versions and download dates recorded
- [ ] Exact model checkpoints (HF revision hashes) recorded
- [ ] Hardware and runtime reported in §IV

---

## 14. BIBLIOGRAPHY SEED (~20 refs; verify each at its DOI before submitting)

**Core gap (must cite):**
1. Haider et al., *Robustness of LLMs to Transliteration Perturbations in Bangla*, BLP-2025 — `aclanthology.org/2025.banglalp-1.27`
2. Islam, Kar, Islam & Amin, *SentNoB*, Findings of EMNLP 2021 — `aclanthology.org/2021.findings-emnlp.278`
3. Tisha, Tabassum, Kibria, Nahiduzzaman & Ahsan, arXiv:2607.11597 (2026) — **cite as preprint; check whether it has since appeared at a venue**

**Datasets & models:**
4. Romim et al., *BD-SHS*, LREC 2022 — `aclanthology.org/2022.lrec-1.552`
5. Hossain, Rahman, Islam & Kar, *BanFakeNews*, LREC 2020 — `aclanthology.org/2020.lrec-1.349`
6. Bhattacharjee et al., *BanglaBERT*, Findings of NAACL 2022
7. Conneau et al., *XLM-R* (Unsupervised Cross-lingual Representation Learning at Scale), ACL 2020
8. Devlin et al., *BERT*, NAACL 2019

**Bangla NLP context:**
9. *Overview of BLP-2025 Task 1: Bangla Hate Speech Detection* — `aclanthology.org/2025.banglalp-1.32`
10. *Breaking the Curse of Class Imbalance: Bangla Text Classification*, ACM TALLIP, DOI 10.1145/3511601
11. Raihan et al., *TB-OLID*, BLP@EMNLP 2023 — `aclanthology.org/2023.banglalp-1.1`
12. Ishmam et al., *BanglaTLit*, Findings of EMNLP 2024
13. Haider, Shifat, Ishmam et al., *BanTH*, Findings of NAACL 2025

**Robustness methodology (general NLP — for §II framing):**
14–20. Adversarial/noise robustness in text classification, character-level attacks, typo robustness, and data-augmentation-for-robustness references. **Find and verify 5–7 of these yourself during days 1–7** — I have not verified specific ones for you and will not fabricate citations.

---

## 15. HOW TO WRITE EACH SECTION FAST

- **§III first** (day 7). It is the contribution and forces design decisions.
- **§II second** (day 14). Structure it as three short paragraphs, ending each with what is *missing* — the third paragraph ends with your gap sentence, quoting BLP-2025's own limitation.
- **§V third** (day 23), one subsection per result R1–R7, each opening with the finding in one sentence, then the evidence.
- **§I last-but-one** (day 24 revision). Write the Introduction *after* you know the results, so the promises match the delivery.
- **Abstract last** (day 25). Six sentences: context, problem, gap, what we did, what we found, what we release.

**Rule:** never write "we propose a novel..." — you are not proposing a method. Write "we construct", "we measure", "we release".

---

## 16. FIRST THREE ACTIONS (today)

1. Create the repo `bangla-noisebench` with the structure in §9.
2. Download BD-SHS, SentNoB and BanFakeNews; record exact counts, splits, class balance, emoji rate; verify each licence. **SentNoB's split sizes are not documented online — you will be the one reporting them.**
3. Write `perturbations.py` for N1 (insertion) plus its monotonicity unit test. One noise type working end-to-end today makes the other nine mechanical.
