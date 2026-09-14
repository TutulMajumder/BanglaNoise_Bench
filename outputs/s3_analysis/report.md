# Phase 2, s3 — post-preparation description

## Table II — dataset statistics after cleaning (METHODOLOGY M2.3)

> SentNoB 0=neutral, 1=positive, 2=negative is not printed on the shipped CSVs; confirmed empirically via a lexicon probe (common Bangla sentiment words checked for co-occurrence with each label) -- see PHASE2_STATUS.md for the method and numbers.

| dataset | task | n (final) | class distribution | mean len | median len | emoji rate | dupes removed | short removed | zero-Bangla removed |
|---|---|--:|---|--:|--:|--:|--:|--:|--:|
| sentnob | sentiment (3-class) | 14353 | negative=5071, positive=5921, neutral=3361 | 53.5 | 43 | 1.770% | 1369 | 0 | 6 |
| bd_shs | hate speech (binary) | 50127 | not_hate=25999, hate=24128 | 47.9 | 29 | 0.575% | 18 | 133 | 3 |
| banfakenews | fake news (binary) | 8489 | authentic=7194, fake=1295 | 1107.0 | 867 | 0.212% | 10 | 0 | 2 |

### BD-SHS: U+FFFD vs. hate-speech label (2x2)

Kept per METHODOLOGY decision (2026-09-12): U+FFFD is authentic user-generated noise (likely mangled emoji from the original scrape), not a reading artefact -- recorded to check whether its presence is label-correlated.

| group | n | hate | not_hate |
|---|--:|--:|--:|
| flagged | 2545 | 38.11% | 61.89% |
| unflagged | 47582 | 48.67% | 51.33% |

**N10 (emoji) experiment host: `sentnob`** — highest emoji rate (1.770%).


## Raw → final delta

See `tables/raw_final_delta.csv` for the full per-dataset, per-metric comparison against s1's raw-corpus numbers -- the defence of every cleaning decision. Also: `tables/class_balance_per_split.csv`, `tables/length_by_class.csv`, `tables/eligible_units_per_noise_type.csv` (explains conjunct_split's severity-1 behaviour), and `tables/cer_truncation_impact.csv` (fraction of texts affected by the 300-cluster CER cap, per dataset).

