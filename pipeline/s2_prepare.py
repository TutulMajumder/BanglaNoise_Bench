"""Phase 2, pipeline stage 2: M2.1 preprocessing + M2.2 splitting.

Reads the raw dataset CSVs **already on disk** (downloads nothing), applies
the M2.1 preprocessing pipeline in the exact required order, drops
within-dataset duplicates, <3-grapheme-cluster texts, and texts with zero
Bengali-block (U+0980-U+09FF) characters (recording all three counts and the
dropped ids), then writes:

    data/final/<dataset>/{train,val,test}.csv   id, text, label (UTF-8, \\n)
    data/final/<dataset>/manifest.json          ids/split, seed, source SHA-256
    outputs/s2_preparation/provenance.json      the full audit record, all datasets
    outputs/s2_preparation/report.md            human-readable summary

Three datasets: sentnob, bd_shs, banfakenews. (banfakenews_full -- the full
48K/1K imbalanced pair, a secondary condition -- was cut 2026-09-15: it
supported no claim in the paper and consumed 72% of s1's real-scale runtime.
See PHASE2_STATUS.md and docs/Topic4_FULL_PAPER_PLAN.md.)

Splits (M2.2):
  * SentNoB ships Train/Val/Test -> used as given, never re-split. Cross-split
    duplicate ids are recorded as a leakage block in its manifest.json
    (test-in-train, val-in-train, val-in-test) -- kept as shipped, available
    later for a leakage-free sensitivity check, never reassigned.
  * BD-SHS and BanFakeNews -> seeded stratified 70/10/20, split_seed=42.

Usage:
    python pipeline/s2_prepare.py --data-root data/raw --dataset all --limit 20   # smoke
    python pipeline/s2_prepare.py --data-root data/raw --dataset all              # real run
    python pipeline/s2_prepare.py --data-root data/raw --dataset bd_shs --force   # redo one

CLAUDE.md invariant 9: resumable (skips a dataset whose train/val/test.csv +
manifest.json already exist, unless --force) and prints progress to stderr.

Every schema fact below is marked verified-or-TODO against the shipped CSV.
If an expected column is missing the script prints the columns it *did* find
and exits 2 rather than guessing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import regex  # noqa: E402  (same pin as noisebench/)

from pipeline._common import (  # noqa: E402
    Progress,
    add_common_args,
    print_smoke_banner,
    should_skip,
    smoke_or_prod,
    write_run_meta,
)

_GRAPHEME_RE = regex.compile(r"\X")

SPLIT_SEED = 42
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)  # train / val / test (METHODOLOGY M2.2)
_DEFAULT_DATA_FINAL = os.path.join(_ROOT, "data", "final")
_OUT_DIR = os.path.join(_ROOT, "outputs", "s2_preparation")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class SourceFile:
    """One raw CSV. ``split`` is set only when the dataset ships its own."""

    patterns: tuple[str, ...]
    split: str | None = None          # "train" / "val" / "test" or None
    label_const: int | None = None    # for files that are single-class (BanFakeNews)


@dataclass
class DatasetSpec:
    name: str
    task: str
    text_cols: tuple[str, ...]
    label_col: str | None
    label_map: dict[str, int]
    label_names: dict[int, str]
    sources: tuple[SourceFile, ...]
    ships_own_splits: bool
    subdir: str = ""                  # optional folder under --data-root
    out_subdir: str = ""              # folder under data/final/ (default: name)
    notes: str = ""

    def __post_init__(self):
        if not self.out_subdir:
            self.out_subdir = self.name


DATASETS: dict[str, DatasetSpec] = {
    # -- SentNoB ----------------------------------------------------------
    # github.com/KhondokerIslam/SentNoB ;
    # Kaggle cryptexcode/sentnob-sentiment-analysis-in-noisy-bangla-texts
    "sentnob": DatasetSpec(
        name="sentnob",
        task="sentiment (3-class)",
        text_cols=("Data", "data", "text", "sentence"),
        label_col="Label",
        label_map={"0": 0, "1": 1, "2": 2},
        # Confirmed empirically 2026-09-12 (lexicon-probe against the real
        # CSVs -- see PHASE2_STATUS.md): matches the HuggingFace dataset card.
        label_names={0: "neutral", 1: "positive", 2: "negative"},
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
        text_cols=("sentence", "text", "Sentence", "comment"),
        label_col="hate speech",   # verified 2026-09-12 -- note the space, not "hate"
        label_map={"0": 0, "1": 1, "0.0": 0, "1.0": 1},
        label_names={0: "not_hate", 1: "hate"},
        sources=(
            # The repo ships train/val/test; METHODOLOGY M2.2 re-splits anyway,
            # so we pull every row from all three and ignore the source split
            # (split=None on each -> src_split="" -> stratified_split reshuffles
            # the pooled rows). Three explicit SourceFiles, not one "*.csv"
            # pattern: _find_file returns only the first glob hit, so a single
            # wildcard entry silently loaded test.csv alone and dropped
            # train.csv/val.csv (45,252 rows) -- caught running on real data.
            SourceFile(("train.csv",)),
            SourceFile(("val.csv",)),
            SourceFile(("test.csv",)),
        ),
        ships_own_splits=False,
        subdir="bd_shs",
    ),
    # -- BanFakeNews --------------------------------------------------------
    # github.com/Rowan1697/FakeNews ; Kaggle cryptexcode/banfakenews
    "banfakenews": DatasetSpec(
        name="banfakenews",
        task="fake news (binary)",
        text_cols=("content", "Content", "articleContent", "text"),
        label_col="label",
        # "1.0"/"0.0": LabeledAuthentic-7K.csv ships its own `label` column as
        # the string "1.0" (float-formatted); label_const bypasses it for the
        # two sources below, but the map stays float-tolerant like bd_shs's.
        label_map={"1": 1, "0": 0, "1.0": 1, "0.0": 0, "authentic": 1, "fake": 0},
        label_names={1: "authentic", 0: "fake"},
        sources=(
            # MAIN runs = balanced labelled subsets (~7K authentic / 1K fake).
            SourceFile(("LabeledAuthentic-7K.csv",), label_const=1),
            SourceFile(("LabeledFake-1K.csv",), label_const=0),
        ),
        ships_own_splits=False,
        subdir="banfakenews",
        notes="The only BanFakeNews condition (2026-09-15: banfakenews_full "
              "cut -- see PHASE2_STATUS.md).",
    ),
    # banfakenews_full (the full 48K/1K imbalanced pair, a secondary
    # condition) was cut 2026-09-15: it supported no claim in the paper and
    # consumed 72% of s1's real-scale runtime (755s of 1042s). See
    # PHASE2_STATUS.md and docs/Topic4_FULL_PAPER_PLAN.md for the record.
    # Three datasets remain: sentnob, bd_shs, banfakenews.
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


# Bengali Unicode block only (U+0980-U+09FF) -- deliberately not a percentage
# threshold. Code-mixed Bangla/English is common and realistic and must be
# kept; a text with ZERO Bengali-block characters has zero eligible units for
# N5/N6/N7 (all defined over Bangla letters/matras), so it would contribute a
# guaranteed CER of 0.0 to those noise types and silently flatten the
# severity curve if left in.
_BENGALI_BLOCK_RE = re.compile("[ঀ-৿]")


def n_bengali_chars(s: str) -> int:
    """Count of characters in the Bengali Unicode block (U+0980-U+09FF)."""
    return len(_BENGALI_BLOCK_RE.findall(s))


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
    source_sha256: dict[str, str] = field(default_factory=dict)
    schema_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
        f"or fix DATASETS['...'] in pipeline/s2_prepare.py",
        file=sys.stderr,
    )
    raise SystemExit(2)


def resolve_labels(
    df: pd.DataFrame,
    spec: DatasetSpec,
    src: SourceFile,
    path: str,
    label_col_override: str | None = None,
) -> pd.Series:
    """THE one place a source file's labels are resolved -- file-level
    (``src.label_const``, e.g. BanFakeNews: LabeledAuthentic-7K -> authentic,
    LabeledFake-1K -> fake) or column-mapped (``spec.label_col`` through
    ``spec.label_map``). Both s2_prepare.py::load_raw and
    s1_inspect_raw.py::inspect_dataset call this exact function so they can
    never derive labels differently again.

    2026-09-14 bug this replaces: s1 re-derived `src` from `spec.sources` via
    a lookup keyed on ``s.split or file_stem``, which is trivially True for
    EVERY source in a dataset whose sources all have ``split=None`` (e.g.
    BanFakeNews's two label_const sources) -- so `next()` always returned the
    FIRST such source regardless of which file was actually being processed,
    and every BanFakeNews row got label_const=1 (authentic). s1 now receives
    the correct `src` directly instead of re-deriving it (see
    ``_resolve_sources`` in s1_inspect_raw.py), and both stages call this.

    Returns an int Series aligned with `df`'s index. Exits 2 (with a message
    naming the bad values) if `label_col` is missing or a value doesn't map.
    """
    if src.label_const is not None:
        return pd.Series([src.label_const] * len(df), index=df.index, dtype=int)

    lcol = label_col_override or spec.label_col
    if lcol not in df.columns:
        _pick_column(df, (lcol,) if lcol else (), "label", path)
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
    return mapped.astype(int)


def load_raw(
    spec: DatasetSpec,
    data_root: str,
    text_col_override: str | None = None,
    label_col_override: str | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Return a frame with columns ``raw_text, label, src_split`` (pre-cleaning).

    ``limit``: truncate EACH source file to its first N rows before
    concatenation -- a smoke run (CLAUDE.md invariant 9), not a sample of the
    final corpus. SHA-256 is always computed over the FULL file regardless of
    ``limit`` (it identifies the source, not the sample).
    """
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
        rel = os.path.relpath(path, data_root)
        rep.source_files.append(rel)
        rep.source_sha256[rel] = _sha256_file(path)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if limit is not None:
            df = df.head(limit)

        tcol = text_col_override or _pick_column(df, spec.text_cols, "text", path)
        rep.schema_used["text_col"] = tcol
        sub = pd.DataFrame({"raw_text": df[tcol].astype(str)})

        sub["label"] = resolve_labels(df, spec, src, path, label_col_override)
        if src.label_const is None:
            rep.schema_used["label_col"] = label_col_override or spec.label_col

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
    non_bangla_removed: int = 0
    non_bangla_dropped_ids: list[str] = field(default_factory=list)
    n_final: int = 0
    class_dist: dict[str, int] = field(default_factory=dict)
    split_sizes: dict[str, int] = field(default_factory=dict)
    split_seed: int = SPLIT_SEED
    split_source: str = ""            # "dataset-provided" or "stratified-70/10/20"
    source_files: list[str] = field(default_factory=list)
    source_sha256: dict[str, str] = field(default_factory=dict)
    schema_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    leakage: dict = field(default_factory=dict)  # {"raw_text":.., "final_text":.., "pct_of_split":..}


def clean(raw: pd.DataFrame, spec: DatasetSpec, rep: LoadReport) -> tuple[pd.DataFrame, PrepStats]:
    """M2.1: preprocess -> drop exact dupes -> drop <3-cluster texts -> drop
    zero-Bangla texts."""
    st = PrepStats(dataset=spec.name, n_raw=len(raw),
                   source_files=rep.source_files, source_sha256=rep.source_sha256,
                   schema_used=rep.schema_used, warnings=list(rep.warnings))

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

    # 8. drop rows with zero Bengali-block characters (see n_bengali_chars).
    #    Code-mixed rows (Bangla + English/digits) are kept -- only an
    #    ABSENCE of any Bengali-block character is dropped.
    before = len(df)
    is_non_bangla = df["text"].map(n_bengali_chars) == 0
    st.non_bangla_dropped_ids = sorted(df.loc[is_non_bangla, "id"].tolist())
    df = df[~is_non_bangla].reset_index(drop=True)
    st.non_bangla_removed = before - len(df)

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


_LEAKAGE_PAIRS = (("test_in_train", "test", "train"),
                  ("val_in_train", "val", "train"),
                  ("val_in_test", "val", "test"))


def _leakage_on_column(df: pd.DataFrame, text_col: str) -> dict[str, list[str]]:
    texts = {s: set(df.loc[df["split"] == s, text_col]) for s in ("train", "val", "test")}

    def overlap_ids(split_name: str, other_texts: set[str]) -> list[str]:
        mask = (df["split"] == split_name) & df[text_col].isin(other_texts)
        return sorted(df.loc[mask, "id"].tolist())

    return {label: overlap_ids(a, texts[b]) for label, a, b in _LEAKAGE_PAIRS}


def compute_leakage(df: pd.DataFrame) -> dict:
    """For a dataset with shipped splits (SentNoB): which val/test ids share
    text with a train id (and val with test), reported on BOTH text
    representations over the SAME final row population --

      * "raw_text": the text exactly as downloaded, before M2.1.
      * "final_text": after M2.1 (URL/mention masking, NFC, whitespace
        collapse) -- what actually ships in data/final/*.csv.

    Both use ``df["id"]`` (assigned from the final cleaned text) so the ids
    are directly usable against data/final/ either way; only the equality
    test changes. The final_text count is expected to be >= the raw_text
    count: masking can make two previously-different raw texts identical
    (e.g. two different URLs both become "<URL>"), never the reverse -- so
    preprocessing can only ADD apparent leakage, never remove real leakage.
    Nothing is reassigned or dropped; this measures, it does not decide.
    """
    raw = _leakage_on_column(df, "raw_text")
    final = _leakage_on_column(df, "text")
    split_sizes = {s: int((df["split"] == s).sum()) for s in ("train", "val", "test")}
    pct_of_split = {}
    for label, a, _b in _LEAKAGE_PAIRS:
        denom = split_sizes[a] or 1
        pct_of_split[label] = {
            "raw_text_pct": round(100.0 * len(raw[label]) / denom, 3),
            "final_text_pct": round(100.0 * len(final[label]) / denom, 3),
            "split_size": split_sizes[a],
        }
    return {"raw_text": raw, "final_text": final, "pct_of_split": pct_of_split}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _write_utf8(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _safe_relpath(path: str, start: str) -> str:
    """``os.path.relpath`` but tolerant of a different drive (Windows / Kaggle
    ``/kaggle/input`` vs the repo on another mount, or a scratch --out-dir)."""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        return os.path.abspath(path)


def dataset_final_dir(spec: DatasetSpec, data_final_root: str) -> str:
    return os.path.join(data_final_root, spec.out_subdir)


def output_paths(spec: DatasetSpec, data_final_root: str) -> list[str]:
    """Paths that mark a dataset as already prepared -- used for resumability."""
    d = dataset_final_dir(spec, data_final_root)
    return [os.path.join(d, f"{s}.csv") for s in ("train", "val", "test")] + \
           [os.path.join(d, "manifest.json")]


def write_outputs(df: pd.DataFrame, spec: DatasetSpec, st: PrepStats, data_final_root: str) -> None:
    out_dir = dataset_final_dir(spec, data_final_root)
    os.makedirs(out_dir, exist_ok=True)

    ids_by_split: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        sub = df[df["split"] == split][["id", "text", "label"]].copy()
        sub = sub.sort_values("id")
        sub.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False,
                   encoding="utf-8", lineterminator="\n")
        ids_by_split[split] = sub["id"].tolist()

    manifest = {
        "dataset": spec.name,
        "out_subdir": spec.out_subdir,
        "split_source": st.split_source,
        "split_seed": st.split_seed if st.split_source.startswith("stratified") else None,
        "split_sizes": st.split_sizes,
        "source_sha256": st.source_sha256,
        "ids": ids_by_split,
    }
    if st.dataset == "sentnob":
        manifest["leakage"] = st.leakage

    _write_utf8(os.path.join(out_dir, "manifest.json"),
               json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown (git rev-parse failed)"


def write_provenance(all_stats: dict[str, PrepStats], out_path: str) -> None:
    provenance = {
        "stage": "s2 (M2.1 preprocessing + M2.2 splitting)",
        "git_rev": _git_rev(),
        "datasets": {
            name: {
                "source_files": st.source_files,
                "source_sha256": st.source_sha256,
                "n_raw": st.n_raw,
                "rows_removed": {
                    "exact_duplicates": st.dupes_removed,
                    "short_lt_3_clusters": st.short_removed,
                    "zero_bangla_chars": st.non_bangla_removed,
                },
                "n_final": st.n_final,
                "class_dist": st.class_dist,
                "split_source": st.split_source,
                "split_seed": st.split_seed if st.split_source.startswith("stratified") else None,
                "split_sizes": st.split_sizes,
                "warnings": st.warnings,
                "leakage_summary": {
                    "n_raw_text": {k: len(v) for k, v in st.leakage["raw_text"].items()},
                    "n_final_text": {k: len(v) for k, v in st.leakage["final_text"].items()},
                    "pct_of_split": st.leakage["pct_of_split"],
                } if st.leakage else {},
            }
            for name, st in all_stats.items()
        },
    }
    _write_utf8(out_path, json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_report(all_stats: dict[str, PrepStats], out_path: str) -> None:
    L = ["# Phase 2, s2 — preparation (M2.1 + M2.2)", ""]
    L.append("| dataset | raw | dupes | short | zero-Bangla | final | split source | train/val/test |")
    L.append("|---|--:|--:|--:|--:|--:|---|---|")
    for name, st in all_stats.items():
        ss = st.split_sizes
        L.append(f"| {name} | {st.n_raw} | {st.dupes_removed} | {st.short_removed} | "
                 f"{st.non_bangla_removed} | {st.n_final} | {st.split_source} | "
                 f"{ss.get('train','?')}/{ss.get('val','?')}/{ss.get('test','?')} |")
    L.append("")
    for name, st in all_stats.items():
        if st.warnings:
            L.append(f"**{name} warnings:**")
            for w in st.warnings:
                L.append(f"- {w}")
            L.append("")
        if st.leakage:
            L.append(f"**{name} leakage** — shipped splits used as given (M2.2), never "
                     f"reassigned; measured on the same final row population, two text "
                     f"representations: raw (as downloaded) vs. final (post-M2.1 masking).")
            L.append("")
            L.append("| pair | split size | n raw-text | % raw | n final-text | % final |")
            L.append("|---|--:|--:|--:|--:|--:|")
            for label, _a, _b in _LEAKAGE_PAIRS:
                p = st.leakage["pct_of_split"][label]
                n_raw = len(st.leakage["raw_text"][label])
                n_final = len(st.leakage["final_text"][label])
                L.append(f"| {label} | {p['split_size']} | {n_raw} | {p['raw_text_pct']:.2f}% "
                         f"| {n_final} | {p['final_text_pct']:.2f}% |")
            L.append("")
    _write_utf8(out_path, "\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def prepare_one(
    name: str,
    data_root: str,
    data_final_root: str,
    text_col: str | None = None,
    label_col: str | None = None,
    limit: int | None = None,
    force: bool = False,
    progress: Progress | None = None,
) -> PrepStats | None:
    """Returns None (and skips work) if outputs already exist and `force` is
    False -- CLAUDE.md invariant 9 resumability. `limit` truncates each
    source file to its first N rows (smoke run only, not a sample)."""
    progress = progress or Progress("s2_prepare")
    spec = DATASETS[name]
    out_dir = dataset_final_dir(spec, data_final_root)
    if should_skip(out_dir, output_paths(spec, data_final_root), limit, force, progress, name):
        return None

    progress.log(name, f"loading raw ({'limit=' + str(limit) if limit else 'full'})")
    raw, rep = load_raw(spec, data_root, text_col, label_col, limit=limit)
    progress.log(name, f"loaded {len(raw)} raw rows; cleaning (M2.1)")
    df, st = clean(raw, spec, rep)
    progress.log(name, f"{st.n_final} rows after cleaning; splitting (M2.2)")
    df = assign_splits(df, spec, st)
    if name == "sentnob":
        st.leakage = compute_leakage(df)
        progress.log(name, "leakage (raw_text -> final_text): " + ", ".join(
            f"{label}={len(st.leakage['raw_text'][label])}->{len(st.leakage['final_text'][label])}"
            f" ({st.leakage['pct_of_split'][label]['final_text_pct']:.2f}% of split)"
            for label, _a, _b in _LEAKAGE_PAIRS
        ))
    progress.log(name, f"split {st.split_sizes}; writing outputs")
    write_outputs(df, spec, st, data_final_root)
    write_run_meta(out_dir, limited=limit is not None, limit=limit,
                   n_rows_processed=st.n_final)
    progress.log(name, "done")
    return st


def _print_summary(st: PrepStats) -> None:
    print(f"\n[{st.dataset}]  source: {', '.join(st.source_files) or '(none)'}")
    print(f"  raw rows              {st.n_raw}")
    print(f"  duplicates removed    {st.dupes_removed}")
    print(f"  short (<3 clusters)   {st.short_removed}")
    print(f"  zero-Bangla removed   {st.non_bangla_removed}")
    print(f"  final n               {st.n_final}")
    print(f"  class distribution    {st.class_dist}")
    print(f"  split ({st.split_source}) {st.split_sizes}")
    for w in st.warnings:
        print(f"  ! {w}")
    if st.leakage:
        print("  leakage (raw_text -> final_text, % of split):")
        for label, _a, _b in _LEAKAGE_PAIRS:
            n_raw = len(st.leakage["raw_text"][label])
            n_final = len(st.leakage["final_text"][label])
            p = st.leakage["pct_of_split"][label]
            print(f"    {label}: {n_raw} -> {n_final}  "
                 f"({p['raw_text_pct']:.2f}% -> {p['final_text_pct']:.2f}% of {p['split_size']})")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True,
                    help="folder holding the raw CSVs (or /kaggle/input)")
    ap.add_argument("--dataset", default="all",
                    choices=[*DATASETS, "all"])
    ap.add_argument("--data-final-root", default=None,
                    help=f"default: {_DEFAULT_DATA_FINAL}, or the .smoke/ mirror under --limit")
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {_OUT_DIR}, or the .smoke/ mirror under --limit "
                         "(where provenance.json + report.md go)")
    ap.add_argument("--text-col", default=None, help="override the text column")
    ap.add_argument("--label-col", default=None, help="override the label column")
    ap.add_argument("--quiet", action="store_true", help="write files, minimal stdout")
    add_common_args(ap)
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    if args.data_final_root is None:
        args.data_final_root = smoke_or_prod(args.limit, _DEFAULT_DATA_FINAL)
    if args.out_dir is None:
        args.out_dir = smoke_or_prod(args.limit, _OUT_DIR)
    if args.limit is not None:
        print_smoke_banner("s2_prepare", args.limit,
                           {"data-final-root": args.data_final_root, "out-dir": args.out_dir})

    progress = Progress("s2_prepare")
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_stats: dict[str, PrepStats] = {}
    for name in names:
        st = prepare_one(name, args.data_root, args.data_final_root,
                         args.text_col, args.label_col,
                         limit=args.limit, force=args.force, progress=progress)
        if st is not None and not args.quiet:
            _print_summary(st)
        if st is not None:
            all_stats[name] = st
    progress.done()

    if all_stats:
        os.makedirs(args.out_dir, exist_ok=True)
        write_provenance(all_stats, os.path.join(args.out_dir, "provenance.json"))
        write_report(all_stats, os.path.join(args.out_dir, "report.md"))
        print(f"[written] {_safe_relpath(args.out_dir, _ROOT)}/provenance.json")
        print(f"[written] {_safe_relpath(args.out_dir, _ROOT)}/report.md")
    print(f"[written] {_safe_relpath(args.data_final_root, _ROOT)}/")
    if not args.quiet and not args.orchestrated:
        print("\nNext: python pipeline/s3_describe.py   and   "
              "python pipeline/s4_measure.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
