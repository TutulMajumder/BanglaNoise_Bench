# METHODOLOGY M2.4 — real-corpus measurements (tracked summary, no verbatim text)

Sample: 300 texts/dataset (seeded), {42, 1337, 2024}; datasets: sentnob, bd_shs, banfakenews.

_Normalizer "what it touched" examples below are structural descriptions (edit kind, offset of first difference, length delta) -- no verbatim corpus text. See `results/m2_4.md` (git-ignored) for the verbatim before/after.

## (1) 9×5 CER — real vs synthetic  (seeds [42, 1337, 2024])

Target severity ratios s2/s1, s3/s2, s4/s3, s5/s4 = **2.00 / 2.00 / 1.75 / 1.43**.

Ratio-structure rule: s3/s2, s4/s3, s5/s4 must each be within 10% of their targets (2.00 / 1.75 / 1.43). s2/s1 is reported but excluded from the pass/fail rule, because low eligible-unit counts at severity 1 make it noisy on short-text corpora -- see s3_describe.py's eligible-units-per-noise-type figure for the eligible-count distribution behind this.

**Each text capped to its first 300 grapheme clusters before perturbing** (real corpora only average far more than the synthetic dev corpus; `levenshtein` is O(n²) and this measurement checks the ratio structure, not document-level CER — see METHODOLOGY M2.4). CER is reference-length-normalized and the M1.3 count rule is per-eligible-unit, so this does not bias the ratios.

### synthetic (dev corpus)

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| char_insert | 0.0255 | 0.0509 | 0.1024 | 0.1785 | 0.2554 | 2.00 | 2.01 | 1.74 | 1.43 | holds (s3+) |
| char_delete | 0.0373 | 0.0748 | 0.1466 | 0.2500 | 0.3415 | 2.01 | 1.96 | 1.71 | 1.37 | holds (s3+) |
| char_substitute | 0.0353 | 0.0688 | 0.1387 | 0.2432 | 0.3471 | 1.95 | 2.02 | 1.75 | 1.43 | holds (s3+) |
| char_transpose | 0.0512 | 0.1006 | 0.1984 | 0.3404 | 0.4685 | 1.96 | 1.97 | 1.72 | 1.38 | holds (s3+) |
| homophone_confuse | 0.0196 | 0.0393 | 0.0777 | 0.1329 | 0.1858 | 2.00 | 1.98 | 1.71 | 1.40 | holds (s3+) |
| matra_perturb | 0.0246 | 0.0492 | 0.0962 | 0.1665 | 0.2363 | 2.00 | 1.96 | 1.73 | 1.42 | holds (s3+) |
| conjunct_split | 0.0061 | 0.0119 | 0.0241 | 0.0421 | 0.0606 | 1.96 | 2.02 | 1.74 | 1.44 | holds (s3+) |
| elongate | 0.0228 | 0.0469 | 0.0918 | 0.1613 | 0.2308 | 2.06 | 1.96 | 1.76 | 1.43 | holds (s3+) |
| whitespace_error | 0.0375 | 0.0750 | 0.1498 | 0.2622 | 0.3744 | 2.00 | 2.00 | 1.75 | 1.43 | holds (s3+) |

### real: sentnob

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| char_insert | 0.0221 | 0.0444 | 0.0898 | 0.1575 | 0.2258 | 2.01 | 2.02 | 1.75 | 1.43 | holds (s3+) |
| char_delete | 0.0345 | 0.0693 | 0.1360 | 0.2281 | 0.3114 | 2.01 | 1.96 | 1.68 | 1.36 | holds (s3+) |
| char_substitute | 0.0355 | 0.0707 | 0.1432 | 0.2501 | 0.3579 | 1.99 | 2.03 | 1.75 | 1.43 | holds (s3+) |
| char_transpose | 0.0440 | 0.0905 | 0.1773 | 0.3063 | 0.4246 | 2.06 | 1.96 | 1.73 | 1.39 | holds (s3+) |
| homophone_confuse | 0.0174 | 0.0355 | 0.0707 | 0.1220 | 0.1697 | 2.05 | 1.99 | 1.72 | 1.39 | holds (s3+) |
| matra_perturb | 0.0230 | 0.0469 | 0.0940 | 0.1637 | 0.2306 | 2.04 | 2.00 | 1.74 | 1.41 | holds (s3+) |
| conjunct_split | 0.0033 | 0.0072 | 0.0147 | 0.0259 | 0.0361 | 2.18 | 2.04 | 1.77 | 1.39 | holds (s3+) |
| elongate | 0.0281 | 0.0572 | 0.1143 | 0.1966 | 0.2823 | 2.03 | 2.00 | 1.72 | 1.44 | holds (s3+) |
| whitespace_error | 0.0359 | 0.0718 | 0.1423 | 0.2492 | 0.3556 | 2.00 | 1.98 | 1.75 | 1.43 | holds (s3+) |

### real: bd_shs

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| char_insert | 0.0238 | 0.0483 | 0.0963 | 0.1652 | 0.2374 | 2.03 | 2.00 | 1.72 | 1.44 | holds (s3+) |
| char_delete | 0.0363 | 0.0741 | 0.1423 | 0.2414 | 0.3290 | 2.04 | 1.92 | 1.70 | 1.36 | holds (s3+) |
| char_substitute | 0.0372 | 0.0738 | 0.1476 | 0.2600 | 0.3713 | 1.98 | 2.00 | 1.76 | 1.43 | holds (s3+) |
| char_transpose | 0.0468 | 0.0949 | 0.1841 | 0.3159 | 0.4428 | 2.03 | 1.94 | 1.72 | 1.40 | holds (s3+) |
| homophone_confuse | 0.0183 | 0.0361 | 0.0731 | 0.1244 | 0.1778 | 1.97 | 2.02 | 1.70 | 1.43 | holds (s3+) |
| matra_perturb | 0.0250 | 0.0481 | 0.0985 | 0.1718 | 0.2450 | 1.93 | 2.05 | 1.74 | 1.43 | holds (s3+) |
| conjunct_split | 0.0035 | 0.0076 | 0.0121 | 0.0237 | 0.0342 | 2.16 | 1.59 | 1.96 | 1.44 | ⚠ off (s3+) |
| elongate | 0.0288 | 0.0582 | 0.1123 | 0.2011 | 0.2877 | 2.02 | 1.93 | 1.79 | 1.43 | holds (s3+) |
| whitespace_error | 0.0353 | 0.0719 | 0.1418 | 0.2501 | 0.3556 | 2.03 | 1.97 | 1.76 | 1.42 | holds (s3+) |

### real: banfakenews

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| char_insert | 0.0267 | 0.0535 | 0.1071 | 0.1874 | 0.2677 | 2.00 | 2.00 | 1.75 | 1.43 | holds (s3+) |
| char_delete | 0.0379 | 0.0753 | 0.1489 | 0.2542 | 0.3517 | 1.99 | 1.98 | 1.71 | 1.38 | holds (s3+) |
| char_substitute | 0.0368 | 0.0736 | 0.1472 | 0.2576 | 0.3680 | 2.00 | 2.00 | 1.75 | 1.43 | holds (s3+) |
| char_transpose | 0.0529 | 0.1048 | 0.2067 | 0.3506 | 0.4814 | 1.98 | 1.97 | 1.70 | 1.37 | holds (s3+) |
| homophone_confuse | 0.0212 | 0.0420 | 0.0831 | 0.1423 | 0.1989 | 1.98 | 1.98 | 1.71 | 1.40 | holds (s3+) |
| matra_perturb | 0.0269 | 0.0534 | 0.1061 | 0.1823 | 0.2571 | 1.99 | 1.98 | 1.72 | 1.41 | holds (s3+) |
| conjunct_split | 0.0070 | 0.0140 | 0.0280 | 0.0491 | 0.0701 | 2.00 | 2.00 | 1.75 | 1.43 | holds (s3+) |
| elongate | 0.0251 | 0.0504 | 0.1011 | 0.1759 | 0.2519 | 2.01 | 2.01 | 1.74 | 1.43 | holds (s3+) |
| whitespace_error | 0.0382 | 0.0765 | 0.1531 | 0.2679 | 0.3825 | 2.00 | 2.00 | 1.75 | 1.43 | holds (s3+) |

### real: ALL

| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |
|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| char_insert | 0.0239 | 0.0484 | 0.0967 | 0.1702 | 0.2431 | 2.02 | 2.00 | 1.76 | 1.43 | holds (s3+) |
| char_delete | 0.0362 | 0.0723 | 0.1423 | 0.2411 | 0.3295 | 2.00 | 1.97 | 1.69 | 1.37 | holds (s3+) |
| char_substitute | 0.0377 | 0.0729 | 0.1455 | 0.2548 | 0.3645 | 1.93 | 1.99 | 1.75 | 1.43 | holds (s3+) |
| char_transpose | 0.0463 | 0.0964 | 0.1882 | 0.3242 | 0.4485 | 2.08 | 1.95 | 1.72 | 1.38 | holds (s3+) |
| homophone_confuse | 0.0190 | 0.0370 | 0.0740 | 0.1278 | 0.1791 | 1.95 | 2.00 | 1.73 | 1.40 | holds (s3+) |
| matra_perturb | 0.0247 | 0.0499 | 0.0995 | 0.1720 | 0.2436 | 2.02 | 1.99 | 1.73 | 1.42 | holds (s3+) |
| conjunct_split | 0.0045 | 0.0088 | 0.0191 | 0.0326 | 0.0458 | 1.95 | 2.18 | 1.71 | 1.40 | holds (s3+) |
| elongate | 0.0270 | 0.0555 | 0.1084 | 0.1890 | 0.2719 | 2.05 | 1.95 | 1.74 | 1.44 | holds (s3+) |
| whitespace_error | 0.0362 | 0.0736 | 0.1454 | 0.2554 | 0.3649 | 2.03 | 1.98 | 1.76 | 1.43 | holds (s3+) |

## (2) N5 homophone-set firing counts

`n_eligible` = member-occurrence count (grouping pass over `_homophone_occurrences`, severity-independent). `n_applied` = sum of `PerturbStats.submodes` at severity 3 over the listed seeds. `per 1k` = eligible occurrences per 1000 sampled texts.

### sentnob  (n=300 texts, applied@s3)

| homophone set | n_eligible | per 1k texts | texts ≥1 | n_applied |
|---|--:|--:|--:|--:|
| শ/ষ/স | 684 | 2280.0 | 248 | 409 |
| ন/ণ | 1053 | 3510.0 | 278 | 582 |
| ই/ঈ | 397 | 1323.3 | 208 | 257 |
| উ/ঊ | 64 | 213.3 | 52 | 33 |
| ি/ী | 1048 | 3493.3 | 272 | 647 |
| ু/ূ | 424 | 1413.3 | 194 | 244 |
| র/ড়/ঢ় | 1358 | 4526.7 | 281 | 818 |
| জ/য | 490 | 1633.3 | 201 | 322 |

### bd_shs  (n=300 texts, applied@s3)

| homophone set | n_eligible | per 1k texts | texts ≥1 | n_applied |
|---|--:|--:|--:|--:|
| শ/ষ/স | 634 | 2113.3 | 216 | 374 |
| ন/ণ | 987 | 3290.0 | 231 | 615 |
| ই/ঈ | 314 | 1046.7 | 151 | 184 |
| উ/ঊ | 49 | 163.3 | 46 | 37 |
| ি/ী | 971 | 3236.7 | 241 | 588 |
| ু/ূ | 481 | 1603.3 | 202 | 288 |
| র/ড়/ঢ় | 1378 | 4593.3 | 270 | 800 |
| জ/য | 455 | 1516.7 | 184 | 271 |

### banfakenews  (n=300 texts, applied@s3)

| homophone set | n_eligible | per 1k texts | texts ≥1 | n_applied |
|---|--:|--:|--:|--:|
| শ/ষ/স | 21814 | 72713.3 | 300 | 12978 |
| ন/ণ | 25672 | 85573.3 | 300 | 15378 |
| ই/ঈ | 4599 | 15330.0 | 298 | 2780 |
| উ/ঊ | 1909 | 6363.3 | 278 | 1175 |
| ি/ী | 29117 | 97056.7 | 300 | 17450 |
| ু/ূ | 8665 | 28883.3 | 300 | 5210 |
| র/ড়/ঢ় | 38036 | 126786.7 | 300 | 22834 |
| জ/য | 12611 | 42036.7 | 300 | 7655 |

### Sets under review (M2.4)

| set | total eligible | per 1k texts | recommendation |
|---|--:|--:|---|
| ই/ঈ | 5310 | 5900.00 | keep |
| উ/ঊ | 2022 | 2246.67 | keep |

Threshold: < 1 eligible occurrence per 1000 texts ⇒ recommend removal with this evidence, per METHODOLOGY M2.4(2). Not removed automatically.

## (3) Emoji-range coverage

emoji 1.4.2 (~Unicode 13 / 2021) -- coverage is an UPPER BOUND against this era, not current Unicode

| dataset | emoji lib | occurrences | coverage (any) | coverage (full) | over-match cps |
|---|---|--:|--:|--:|--:|
| sentnob | 1.4.2 | 14 | 100.0% | 100.0% | 0 |
| bd_shs | 1.4.2 | 2 | 100.0% | 100.0% | 0 |
| banfakenews | 1.4.2 | 1 | 100.0% | 100.0% | 0 |
| ALL | 1.4.2 | 3 | 100.0% | 100.0% | 0 |

All datasets with ≥20 emoji occurrences are at ≥90% any-coverage; no range change proposed.

## (4) Normalizer touch_rate — R6 go/no-go (M5.4, PER DATASET)

M5.4's 5% threshold is applied **per dataset**, not pooled.

| dataset | n texts | touched | touch_rate | R6 (this dataset) | kinds |
|---|--:|--:|--:|---|---|
| sentnob | 300 | 5 | 1.67% | DROP (<5%) | bengali, len 123->121 (1), punctuation (1), bengali, len 104->103 (1), bengali, len 108->105 (1) |
| bd_shs | 300 | 5 | 1.67% | DROP (<5%) | bengali, punctuation (3), bengali, len 256->249 (1), punctuation (1) |
| banfakenews | 300 | 162 | 54.00% | KEEP (≥5%) | punctuation (143), punctuation, other, len 3235->3233 (1), punctuation, len 1032->1036 (1), punctuation, whitespace, len 9558->9562 (1) |
| ALL | 300 | 55 | 18.33% | N/A (pooled, not a decision unit) | punctuation (48), punctuation, other, len 3029->3028 (1), punctuation, len 2409->2415 (1), punctuation, bengali, len 2391->2395 (1) |

### What it touched — structural description (no verbatim text)

**sentnob:**

| kind | offset of first diff | length before | length delta |
|---|--:|--:|--:|
| bengali, len 123->121 | 77 | 123 | -2 |
| punctuation | 28 | 31 | +0 |
| bengali, len 104->103 | 55 | 104 | -1 |
| bengali, len 108->105 | 0 | 108 | -3 |
| bengali, len 49->48 | 0 | 49 | -1 |

**bd_shs:**

| kind | offset of first diff | length before | length delta |
|---|--:|--:|--:|
| bengali, punctuation | 32 | 144 | +0 |
| bengali, punctuation | 73 | 145 | +0 |
| bengali, len 256->249 | 0 | 160 | +0 |
| punctuation | 27 | 160 | +0 |
| bengali, punctuation | 11 | 45 | +0 |

**banfakenews:**

| kind | offset of first diff | length before | length delta |
|---|--:|--:|--:|
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 46 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |

**ALL:**

| kind | offset of first diff | length before | length delta |
|---|--:|--:|--:|
| punctuation, other, len 3029->3028 | 0 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |
| punctuation | 13 | 160 | +0 |
| punctuation, len 2409->2415 | 160 | 160 | +0 |
| punctuation | 160 | 160 | +0 |

## N10 — emoji-bearing text counts, full test split (report only)

Not a decision. A 5-point severity curve needs enough emoji-bearing texts in the TEST split specifically -- the 300-sample used above cannot answer this for splits larger than 300.

| dataset | n test texts | emoji-bearing | % |
|---|--:|--:|--:|
| sentnob | 1569 | 29 | 1.85% |
| bd_shs | 10025 | 49 | 0.49% |
| banfakenews | 1698 | 5 | 0.29% |

