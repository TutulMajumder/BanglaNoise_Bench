# data/

Phase 2 data-preparation code and the seeded split manifests. **Raw corpora and
cleaned outputs are git-ignored** (`.gitignore`: `data/*` with `!data/*.py`,
`!data/README.md`, `!data/splits`).

## Layout

```
data/
├── prepare.py            tracked  — M2.1 preprocessing + M2.2 splitting
├── README.md             tracked  — this file
├── splits/               tracked  — id-list manifests, one id per line
│   ├── <dataset>.train.txt
│   ├── <dataset>.val.txt
│   └── <dataset>.test.txt
├── raw/                  IGNORED  — put the downloaded CSVs here (or use
│   ├── sentnob/                    --data-root to point elsewhere, e.g.
│   ├── bd_shs/                     /kaggle/input)
│   └── banfakenews/
└── clean/               IGNORED  — written by prepare.py
    ├── <dataset>.csv               id, text, label, split
    └── <dataset>.prep.json         every M2.1 count + split sizes
```

## Expected raw files (TODO(verify) — confirm against what you download)

| dataset | folder | files | text col | label col | labels |
|---------|--------|-------|----------|-----------|--------|
| `sentnob` | `sentnob/` | `train.csv`, `val.csv`, `test.csv` | `Data` | `Label` | `0/1/2` — **class-name mapping unconfirmed** (Topic4 plan: 0=neutral, 1=positive, 2=negative; other sources differ) |
| `bd_shs` | `bd_shs/` | any `*.csv` (all rows pooled, re-split) | `sentence` | `hate` | `0` not-hate / `1` hate |
| `banfakenews` | `banfakenews/` | `LabeledAuthentic-7K.csv`, `LabeledFake-1K.csv` | `content` | *(file-level: authentic=1, fake=0)* | balanced labelled subset; full 48K/1K is secondary |

Every one of these is marked `# TODO(verify)` in `prepare.py::DATASETS`. If a
column is missing the script prints the columns it found and exits 2 — it never
guesses. Override with `--text-col` / `--label-col`.

## Run

```bash
python data/prepare.py --data-root data/raw --dataset all
python scripts/table2_report.py           # -> results/table2.{md,html}
python scripts/m2_4_measurements.py        # -> results/m2_4.{md,html,json}
```

On Kaggle: `--data-root /kaggle/input` (the loader searches recursively).

## Splits (METHODOLOGY M2.2)

- **SentNoB**: shipped Train/Val/Test used as given. Exact duplicates are dropped
  *within each provided split*; texts that recur across splits are counted and
  reported, never reassigned.
- **BD-SHS, BanFakeNews**: seeded stratified 70/10/20, `split_seed=42`,
  order-independent (ids sorted per class before the seeded shuffle), so the
  manifests reproduce exactly regardless of raw row order.
