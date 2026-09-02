"""Phase 2 data preparation (METHODOLOGY M2.1 + M2.2).

Reads the three raw dataset CSVs **already on disk** (this script downloads
nothing), applies the M2.1 preprocessing pipeline in the exact required order,
drops within-dataset duplicates and <3-grapheme-cluster texts (recording both
counts), then writes:

    data/clean/<dataset>.csv        id, text, label, split   (UTF-8, \\n)
    data/clean/<dataset>.prep.json  every count from M2.1 + the split sizes
    data/splits/<dataset>.{train,val,test}.txt   id-list manifests (tracked)

Splits (M2.2):
  * SentNoB ships Train/Val/Test -> used as given, never re-split.
  * BD-SHS and BanFakeNews -> seeded stratified 70/10/20, split_seed=42.

Usage:
    python data/prepare.py --data-root /kaggle/input --dataset all
    python data/prepare.py --data-root ./data/raw --dataset bd_shs
    python data/prepare.py --data-root ./data/raw --dataset sentnob \\
        --text-col Data --label-col Label        # schema overrides

Every schema fact below is marked ``# TODO(verify)`` -- confirm each column
name against the CSV actually shipped before trusting the output. If an
expected column is missing the script prints the columns it *did* find and
exits 2 rather than guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import regex  # noqa: E402  (same pin as noisebench/)

_GRAPHEME_RE = regex.compile(r"\X")

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)  # train / val / test (METHODOLOGY M2.2)


# ---------------------------------------------------------------------------
# Dataset registry -- EVERY field is TODO(verify) against the shipped CSV
# ---------------------------------------------------------------------------


@dataclass
class SourceFile:
    """One raw CSV. ``split`` is set only when the dataset ships its own."""

    # TODO(verify): filename(s) as shipped. Globs allowed; first match wins.
    patterns: tuple[str, ...]
    split: str | None = None          # "train" / "val" / "test" or None
    label_const: int | None = None    # for files that are single-class (BanFakeNews)


@dataclass
class DatasetSpec:
    name: str
    task: str
    # TODO(verify): the text column, and fallbacks tried in order.
    text_cols: tuple[str, ...]
    # TODO(verify): the label column. Ignored when every SourceFile sets label_const.
    label_col: str | None
    # TODO(verify): raw label value -> canonical small int. Keys compared as str.
    label_map: dict[str, int]
    label_names: dict[int, str]
    sources: tuple[SourceFile, ...]
    ships_own_splits: bool
    subdir: str = ""                  # optional folder under --data-root
    notes: str = ""


DATASETS: dict[str, DatasetSpec] = {
    # -- SentNoB ----------------------------------------------------------
    # github.com/KhondokerIslam/SentNoB ;
    # Kaggle cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts
    "sentnob": DatasetSpec(
        name="sentnob",
        task="sentiment (3-class)",
        text_cols=("Data", "data", "text", "sentence"),          # TODO(verify)
        label_col="Label",                                        # TODO(verify)
        # TODO(verify): Topic4 plan says 0=neutral, 1=positive, 2=negative.
        # Other sources list 0=negative/1=neutral/2=positive -- CONFIRM before
        # reporting the class distribution, the mapping flips two class names.
        label_map={"0": 0, "1": 1, "2": 2},
        label_names={0: "neutral?", 1: "positive?", 2: "negative?"},  # TODO(verify)
        sources=(
            SourceFile(("train.csv", "Train.csv", "*train*.csv"), split="train"),
            SourceFile(("val.csv", "Val.csv", "dev.csv", "*val*.csv"), split="val"),
            SourceFile(("test.csv", "Test.csv", "*test*.csv"), split="test"),
        ),
        ships_own_splits=True,
        subdir="sentnob",
        notes="Split sizes are not documented online; we report ours (Topic4 plan).",
    ),
    # -- BD-SHS ---------------------------------------------------------------
    # github.com/naurosromim/hate-speech-dataset-for-Bengali-social-media ;
    # Kaggle naurosromim/bdshs
    "bd_shs": DatasetSpec(
        name="bd_shs",
        task="hate speech (binary)",
        text_cols=("sentence", "text", "Sentence", "comment"),   # TODO(verify)
        label_col="hate",                                         # TODO(verify)
        label_map={"0": 0, "1": 1, "0.0": 0, "1.0": 1},          # TODO(verify)
        label_names={0: "not_hate", 1: "hate"},                   # TODO(verify)
        sources=(
            # The repo ships train/val/test; METHODOLOGY M2.2 re-splits anyway,
            # so we pull every row from whichever files exist and ignore any
            # source split column.
            SourceFile(("*.csv",)),
        ),
        ships_own_splits=False,
        subdir="bd_shs",
    ),
    # -- BanFakeNews --------------------------------------------------------
    # github.com/Rowan1697/FakeNews ; Kaggle cryptexcode/banfakenews
    "banfakenews": DatasetSpec(
        name="banfakenews",
        task="fake news (binary)",
        text_cols=("content", "Content", "articleContent", "text"),  # TODO(verify)
        label_col="label",                                            # TODO(verify)
        label_map={"1": 1, "0": 0, "authentic": 1, "fake": 0},        # TODO(verify)
        label_names={1: "authentic", 0: "fake"},                      # TODO(verify)
        sources=(
            # MAIN runs = balanced labelled subsets (~7K authentic / 1K fake).
            # TODO(verify): filenames "LabeledAuthentic-7K.csv" / "LabeledFake-1K.csv".
            SourceFile(("LabeledAuthentic-7K.csv", "*labeled*authentic*7*.csv"),
                       label_const=1),
            SourceFile(("LabeledFake-1K.csv", "*labeled*fake*1*.csv"),
                       label_const=0),
        ),
        ships_own_splits=False,
        subdir="banfakenews",
        notes="Full 48K/1K set is a secondary condition, not built here.",
    ),
}


# ---------------------------------------------------------------------------
# M2.1 preprocessing -- APPLY IN THIS EXACT ORDER
# ---------------------------------------------------------------------------

# Step 2 patterns. URLs first (they may contain @), then @mentions.
_URL_RE = re.compile(r"""(?xi)
    \b(
        (?:https?://|www\.)\S+
    )
""")
# @handle: ASCII/Bangla word chars after @, not an email local-part.
_MENTION_RE = re.compile(r"(?<![\w@])@[\wঀ-৿]+")
_WS_RE = re.compile(r"\s+")


def preprocess_text(s: str) -> str:
    """M2.1 steps 1-5 on one string. Order is normative.

    1. NFC normalization.
    2. URLs -> ``<URL>``, @mentions -> ``<USER>`` (replace, never remove).
    3. Collapse whitespace runs to one space; strip ends.
    4. No lowercasing.
    5. No emoji stripping (that is N10).
    """
    s = unicodedata.normalize("NFC", str(s))
    s = _URL_RE.sub("<URL>", s)
    s = _MENTION_RE.sub("<USER>", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def n_clusters(s: str) -> int:
    """Grapheme-cluster count (NFC + ``\\X``), matching noisebench M1.1."""
    return len(_GRAPHEME_RE.findall(unicodedata.normalize("NFC", s)))


def _row_id(dataset: str, text: str, src_split: str = "") -> str:
    """Stable id from the cleaned text -- order-independent, so splits are
    reproducible regardless of how the raw rows were ordered. ``src_split`` is
    folded in only for datasets used with their shipped splits, so a text that
    legitimately appears in two provided splits keeps two distinct ids."""
    h = hashlib.sha1(f"{dataset}\x00{src_split}\x00{text}".encode("utf-8")).hexdigest()
    return f"{dataset}-{h[:16]}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass
class LoadReport:
    dataset: str
    source_files: list[str] = field(default_factory=list)
    schema_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _find_file(root: str, patterns: tuple[str, ...]) -> str | None:
    import glob

    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(root, pat)))
        hits += sorted(glob.glob(os.path.join(root, "**", pat), recursive=True))
        if hits:
            return hits[0]
    return None


def _pick_column(df: pd.DataFrame, candidates: tuple[str, ...], role: str,
                 path: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    print(
        f"\nERROR: no {role} column in {path}\n"
        f"  tried: {list(candidates)}\n"
        f"  found: {list(df.columns)}\n"
        f"  pass --{role.replace('label', 'label-').replace('text', 'text-')}col "
        f"or fix DATASETS['...'] in data/prepare.py",
        file=sys.stderr,
    )
    raise SystemExit(2)


def load_raw(
    spec: DatasetSpec,
    data_root: str,
    text_col_override: str | None = None,
    label_col_override: str | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Return a frame with columns ``raw_text, label, src_split`` (pre-cleaning)."""
    rep = LoadReport(dataset=spec.name)
    search_roots = [data_root]
    if spec.subdir:
        search_roots.insert(0, os.path.join(data_root, spec.subdir))

    frames: list[pd.DataFrame] = []
    for src in spec.sources:
        path = None
        for r in search_roots:
            path = _find_file(r, src.patterns)
            if path:
                break
        if not path:
            if src.label_const is not None or src.split is not None:
                rep.warnings.append(
                    f"source file not found for patterns {src.patterns} "
                    f"under {search_roots}"
                )
            continue
        rep.source_files.append(_safe_relpath(path, data_root))
        df = pd.read_csv(path, dtype=str, keep_default_na=False)

        tcol = text_col_override or _pick_column(df, spec.text_cols, "text", path)
        rep.schema_used["text_col"] = tcol
        sub = pd.DataFrame({"raw_text": df[tcol].astype(str)})

        if src.label_const is not None:
            sub["label"] = src.label_const
        else:
            lcol = label_col_override or spec.label_col
            if lcol not in df.columns:
                _pick_column(df, (lcol,) if lcol else (), "label", path)
            rep.schema_used["label_col"] = lcol
            mapped = df[lcol].astype(str).str.strip().map(spec.label_map)
            if mapped.isna().any():
                bad = sorted(set(df[lcol].astype(str)[mapped.isna()]))[:10]
                print(
                    f"\nERROR: unmapped label values in {path}: {bad}\n"
                    f"  label_map = {spec.label_map}\n"
                    f"  fix DATASETS['{spec.name}'].label_map",
                    file=sys.stderr,
                )
                raise SystemExit(2)
            sub["label"] = mapped.astype(int)

        sub["src_split"] = src.split if src.split else ""
        frames.append(sub)

    if not frames:
        print(f"\nERROR: no source files found for {spec.name} under "
              f"{search_roots}\n", file=sys.stderr)
        raise SystemExit(2)

    return pd.concat(frames, ignore_index=True), rep


# ---------------------------------------------------------------------------
# M2.1 cleaning + M2.2 splitting
# ---------------------------------------------------------------------------


@dataclass
class PrepStats:
    dataset: str
    n_raw: int = 0
    dupes_removed: int = 0
    short_removed: int = 0
    n_final: int = 0
    class_dist: dict[str, int] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    split_seed: int = SPLIT_SEED
    split_source: str = ""            # "dataset-provided" or "stratified-70/10/20"
    source_files: list[str] = field(default_factory=list)
    schema_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def clean(raw: pd.DataFrame, spec: DatasetSpec, rep: LoadReport) -> tuple[pd.DataFrame, PrepStats]:
    """M2.1: preprocess -> drop exact dupes -> drop <3-cluster texts."""
    st = PrepStats(dataset=spec.name, n_raw=len(raw),
                   source_files=rep.source_files, schema_used=rep.schema_used,
                   warnings=list(rep.warnings))

    df = raw.copy()
    df["text"] = df["raw_text"].map(preprocess_text)

    # 6. exact duplicates on the cleaned text. Keep first.
    #    * datasets we re-split: dedup globally.
    #    * datasets whose splits we "use as given" (SentNoB): dedup WITHIN each
    #      provided split, so the official Val/Test sets stay exactly as shipped;
    #      count cross-split duplicates separately and report them, never
    #      silently reassign a row's split.
    before = len(df)
    if spec.ships_own_splits:
        df = df.drop_duplicates(subset=["src_split", "text"], keep="first")
        df = df.reset_index(drop=True)
        st.dupes_removed = before - len(df)
        x = df["text"].duplicated(keep=False).sum()
        if x:
            st.warnings.append(
                f"{int(x)} rows share cleaned text across provided splits "
                f"(kept as shipped, not deduplicated across splits)"
            )
    else:
        df = df.drop_duplicates(subset="text", keep="first").reset_index(drop=True)
        st.dupes_removed = before - len(df)

    # 7. drop texts with < 3 grapheme clusters.
    before = len(df)
    df["n_clusters"] = df["text"].map(n_clusters)
    df = df[df["n_clusters"] >= 3].reset_index(drop=True)
    st.short_removed = before - len(df)

    id_split = df["src_split"] if spec.ships_own_splits else ""
    df["id"] = [
        _row_id(spec.name, t, s)
        for t, s in zip(df["text"], id_split if spec.ships_own_splits
                        else [""] * len(df))
    ]
    df = df.drop_duplicates(subset="id", keep="first").reset_index(drop=True)

    st.n_final = len(df)
    st.class_dist = {
        spec.label_names.get(int(k), str(k)): int(v)
        for k, v in df["label"].value_counts().sort_index().items()
    }
    return df, st


def stratified_split(df: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.DataFrame:
    """Seeded stratified 70/10/20 (METHODOLOGY M2.2). Order-independent:
    ids are sorted within each class before the seeded shuffle."""
    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    tr, va, _te = SPLIT_FRACTIONS
    for label, grp in df.groupby("label"):
        ids = sorted(grp["id"].tolist())
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * tr))
        n_va = int(round(n * va))
        for i, _id in enumerate(ids):
            split_of[_id] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
    out = df.copy()
    out["split"] = out["id"].map(split_of)
    return out


def assign_splits(df: pd.DataFrame, spec: DatasetSpec, st: PrepStats) -> pd.DataFrame:
    if spec.ships_own_splits:
        if (df["src_split"] == "").any():
            st.warnings.append("some rows had no source split; those are dropped")
            df = df[df["src_split"] != ""].reset_index(drop=True)
        df = df.copy()
        df["split"] = df["src_split"]
        st.split_source = "dataset-provided"
    else:
        df = stratified_split(df, seed=st.split_seed)
        st.split_source = "stratified-70/10/20"
    st.split_sizes = {
        k: int(v) for k, v in df["split"].value_counts().reindex(
            ["train", "val", "test"]).fillna(0).astype(int).items()
    }
    return df


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_utf8(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _safe_relpath(path: str, start: str) -> str:
    """``os.path.relpath`` but tolerant of a different drive (Windows / Kaggle
    ``/kaggle/input`` vs the repo on another mount)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)


def write_outputs(df: pd.DataFrame, st: PrepStats, out_root: str) -> None:
    clean_dir = os.path.join(out_root, "clean")
    splits_dir = os.path.join(out_root, "splits")

    cols = ["id", "text", "label", "split"]
    csv_path = os.path.join(clean_dir, f"{st.dataset}.csv")
    os.makedirs(clean_dir, exist_ok=True)
    df[cols].to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")

    _write_utf8(
        os.path.join(clean_dir, f"{st.dataset}.prep.json"),
        json.dumps(st.__dict__, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    for split in ("train", "val", "test"):
        ids = sorted(df.loc[df["split"] == split, "id"].tolist())
        _write_utf8(
            os.path.join(splits_dir, f"{st.dataset}.{split}.txt"),
            "".join(f"{i}\n" for i in ids),
        )


def reload_manifest(dataset: str, out_root: str) -> dict[str, list[str]]:
    """Read back the id-list manifests -- used by the acceptance test."""
    splits_dir = os.path.join(out_root, "splits")
    out: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(splits_dir, f"{dataset}.{split}.txt")
        with open(path, encoding="utf-8") as fh:
            out[split] = [ln.strip() for ln in fh if ln.strip()]
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def prepare_one(
    name: str,
    data_root: str,
    out_root: str,
    text_col: str | None = None,
    label_col: str | None = None,
) -> PrepStats:
    spec = DATASETS[name]
    raw, rep = load_raw(spec, data_root, text_col, label_col)
    df, st = clean(raw, spec, rep)
    df = assign_splits(df, spec, st)
    write_outputs(df, st, out_root)
    return st


def _print_summary(st: PrepStats) -> None:
    print(f"\n[{st.dataset}]  source: {', '.join(st.source_files) or '(none)'}")
    print(f"  raw rows              {st.n_raw}")
    print(f"  duplicates removed    {st.dupes_removed}")
    print(f"  short (<3 clusters)   {st.short_removed}")
    print(f"  final n               {st.n_final}")
    print(f"  class distribution    {st.class_dist}")
    print(f"  split ({st.split_source}) {st.split_sizes}")
    for w in st.warnings:
        print(f"  ! {w}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="folder holding the raw CSVs (or /kaggle/input)")
    ap.add_argument("--dataset", default="all",
                    choices=[*DATASETS, "all"])
    ap.add_argument("--out-root", default=os.path.join(_ROOT, "data"),
                    help="where data/clean and data/splits are written")
    ap.add_argument("--text-col", default=None, help="override the text column")
    ap.add_argument("--label-col", default=None, help="override the label column")
    ap.add_argument("--quiet", action="store_true", help="write files, minimal stdout")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    stats = []
    for name in names:
        st = prepare_one(name, args.data_root, args.out_root,
                         args.text_col, args.label_col)
        if not args.quiet:
            _print_summary(st)
        stats.append(st)

    print(f"[written] {_safe_relpath(os.path.join(args.out_root, 'clean'), _ROOT)}/")
    print(f"[written] {_safe_relpath(os.path.join(args.out_root, 'splits'), _ROOT)}/")
    if not args.quiet:
        print("\nNext: python scripts/table2_report.py   and   "
              "python scripts/m2_4_measurements.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
