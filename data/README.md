# data/

Only data lives here now — all Phase 2 pipeline code is under `pipeline/`
(see `pipeline/run_all.py`). **Raw and final corpora are git-ignored** except
each dataset's `manifest.json` (ids + stratification seed + source SHA-256,
no text — `.gitignore`: `data/*` with `!data/README.md`, `!data/final`, then
`data/final/*/*.csv` and `data/final/*/*/*.csv` re-excluded).

## Layout

```
data/
├── README.md             tracked  — this file
├── raw/                  IGNORED  — put the downloaded CSVs here (or use
│   ├── sentnob/                    --data-root to point elsewhere, e.g.
│   ├── bd_shs/                     /kaggle/input)
│   └── banfakenews/
└── final/                written by pipeline/s2_prepare.py
    ├── sentnob/
    │   ├── train.csv      IGNORED  — id, text, label
    │   ├── val.csv        IGNORED
    │   ├── test.csv       IGNORED
    │   └── manifest.json  tracked  — ids/split, seed, source SHA-256, leakage
    ├── bd_shs/            (same 4 files)
    └── banfakenews/
        ├── train.csv val.csv test.csv manifest.json   (same 4 files — primary)
        └── full/
            └── train.csv val.csv test.csv manifest.json  (secondary condition)
```

`banfakenews_full` lives at `data/final/banfakenews/full/`, not as a sibling
top-level dataset — it is a condition ON banfakenews, not a fifth corpus.

## Expected raw files (verified 2026-09-12 against the shipped CSVs)

| dataset | folder | files | text col | label col | labels |
|---------|--------|-------|----------|-----------|--------|
| `sentnob` | `sentnob/` | `Train.csv`, `Val.csv`, `Test.csv` | `Data` | `Label` | `0/1/2` — **confirmed empirically** (lexicon-probe against the CSVs, see `PHASE2_STATUS.md`): 0=neutral, 1=positive, 2=negative |
| `bd_shs` | `bd_shs/` | `train.csv`, `val.csv`, `test.csv` (all three pooled, re-split) | `sentence` | `hate speech` (note the space — not `hate`) | `0` not-hate / `1` hate |
| `banfakenews` | `banfakenews/` | `LabeledAuthentic-7K.csv`, `LabeledFake-1K.csv` | `content` | *(file-level: authentic=1, fake=0)* | balanced labelled subset — the **primary** condition |
| `banfakenews_full` | `banfakenews/` (same dir) | `Authentic-48K.csv`, `Fake-1K.csv` | `content` | *(file-level: authentic=1, fake=0)* | full imbalanced pair — **secondary** condition, same registry/pipeline as `banfakenews`, reported in Table II only |

If a column is missing the script prints the columns it found and exits 2 — it
never guesses. Override with `--text-col` / `--label-col`.

Rows with zero characters in the Bengali Unicode block (U+0980–U+09FF) are
dropped in M2.1 (not a percentage threshold — code-mixed Bangla/English is
kept). Counts and dropped ids are in `outputs/s2_preparation/provenance.json`
and `PHASE2_STATUS.md`.

## Run

See `pipeline/run_all.py` and `PHASE2_STATUS.md` for the exact commands and
order (s1 raw inspection → look at the report → s2 prepare → s3 describe →
s4 measure → s5 figures). On Kaggle: `--data-root /kaggle/input` (the loader
searches recursively).

## Splits (METHODOLOGY M2.2)

- **SentNoB**: shipped Train/Val/Test used as given. Exact duplicates are dropped
  *within each provided split*; texts that recur across splits are counted and
  recorded in `manifest.json`'s `leakage` block (`test_in_train`, `val_in_train`,
  `val_in_test`), never reassigned.
- **BD-SHS, BanFakeNews(_full)**: seeded stratified 70/10/20, `split_seed=42`,
  order-independent (ids sorted per class before the seeded shuffle), so
  `manifest.json` reproduces exactly regardless of raw row order.
