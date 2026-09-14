# Phase 2, s2 — preparation (M2.1 + M2.2)

| dataset | raw | dupes | short | zero-Bangla | final | split source | train/val/test |
|---|--:|--:|--:|--:|--:|---|---|
| sentnob | 15728 | 1369 | 0 | 6 | 14353 | dataset-provided | 11227/1557/1569 |
| bd_shs | 50281 | 18 | 133 | 3 | 50127 | stratified-70/10/20 | 35089/5013/10025 |
| banfakenews | 8501 | 10 | 0 | 2 | 8489 | stratified-70/10/20 | 5942/849/1698 |

**sentnob warnings:**
- 1511 rows share cleaned text across provided splits (kept as shipped, not deduplicated across splits)

**sentnob leakage** — shipped splits used as given (M2.2), never reassigned; measured on the same final row population, two text representations: raw (as downloaded) vs. final (post-M2.1 masking).

| pair | split size | n raw-text | % raw | n final-text | % final |
|---|--:|--:|--:|--:|--:|
| test_in_train | 1569 | 145 | 9.24% | 361 | 23.01% |
| val_in_train | 1557 | 142 | 9.12% | 351 | 22.54% |
| val_in_test | 1557 | 16 | 1.03% | 54 | 3.47% |

