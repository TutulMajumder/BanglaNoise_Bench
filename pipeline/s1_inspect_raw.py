"""Phase 2, pipeline stage 1: raw inspection.

Runs on ``data/raw/`` exactly as downloaded -- BEFORE any M2.1/M2.2 cleaning.
Read-only: writes nothing under ``data/``. This is paper material: it
characterises the corpora as published, so the Methodology section can state
what we received before stating what we changed.

Uses ``noisebench/corpus_stats.py`` for every metric -- the SAME functions
s3_describe.py calls on ``data/final/``, so the raw-vs-final delta table
compares identical measurements on different inputs, never two
implementations that happen to compute similar things.

Writes, under ``outputs/s1_raw_inspection/``:

    tables/file_inventory.csv            rows, columns, dtypes, SHA-256/file
    tables/class_distribution.csv        per dataset, per file/provided-split
    tables/length_distribution.csv       grapheme-cluster length, per text
    tables/char_class_composition.csv    Bengali/Latin/digit/.../char, counts
    tables/code_mixing_latin_share.csv   per-text Latin-character share
    tables/data_quality_inventory.csv    dup/empty/zero-Bangla/FFFD/emoji counts
    tables/cross_split_duplicates.csv    exact-duplicate texts across files
    tables/emoji_bearing_per_split.csv   emoji-bearing counts, per file/split
    figures/                             populated later by s5_figures.py
    report.md

    python pipeline/s1_inspect_raw.py --limit 20     # smoke run
    python pipeline/s1_inspect_raw.py                # real run (reads data/raw/)

CLAUDE.md invariant 9: resumable (skips if tables/file_inventory.csv exists,
unless --force), prints progress to stderr, --limit caps rows read per file.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench import corpus_stats as cs  # noqa: E402
from pipeline._common import (  # noqa: E402
    Progress,
    add_common_args,
    print_smoke_banner,
    should_skip,
    smoke_or_prod,
    write_run_meta,
)
from pipeline.s2_prepare import (  # noqa: E402
    DATASETS,
    DatasetSpec,
    SourceFile,
    _find_file,
    _pick_column,
    _safe_relpath,
    _write_utf8,
    resolve_labels,
)

_DEFAULT_RAW = os.path.join(_ROOT, "data", "raw")
_OUT_DIR = os.path.join(_ROOT, "outputs", "s1_raw_inspection")


def _resolve_sources(spec: DatasetSpec, data_root: str) -> list[tuple[str, str, SourceFile]]:
    """[(split_label, path, src), ...] for every source file that exists.
    `split_label` is `src.split` when the dataset ships one (SentNoB), else
    the file's stem (banfakenews's by-file label; bd_shs's own train/val/test
    filenames -- meaningful groupings to check for cross-file duplication
    even though M2.2 will pool and re-split bd_shs/banfakenews anyway).

    Returns the matched `src` (the actual `SourceFile` this path came from)
    directly, rather than making the caller re-derive it later: re-deriving
    it from `spec.sources` via a lookup keyed on `s.split or file_stem` was
    the 2026-09-14 bug -- trivially True for every source in a dataset whose
    sources all have `split=None` (BanFakeNews's two label_const sources), so
    `next()` always returned the FIRST one regardless of which file was
    actually being processed, and every BanFakeNews row got label_const=1."""
    search_roots = [data_root]
    if spec.subdir:
        search_roots.insert(0, os.path.join(data_root, spec.subdir))
    out = []
    for src in spec.sources:
        path = None
        for r in search_roots:
            path = _find_file(r, src.patterns)
            if path:
                break
        if not path:
            continue
        label = src.split or os.path.splitext(os.path.basename(path))[0]
        out.append((label, path, src))
    return out


def inspect_dataset(name: str, spec: DatasetSpec, data_root: str,
                     limit: int | None, progress: Progress) -> dict:
    sources = _resolve_sources(spec, data_root)
    file_inventory: list[dict] = []
    split_to_texts: dict[str, list[str]] = {}
    split_to_labels: dict[str, list] = {}

    for split_label, path, src in sources:
        progress.log(name, f"reading {os.path.basename(path)}")
        sha = cs.sha256_file(path)
        n_decode_fail = cs.utf8_decode_failures(path)
        df_full = pd.read_csv(path, dtype=str, keep_default_na=False)
        n_rows_full = len(df_full)
        df = df_full.head(limit) if limit is not None else df_full

        tcol = _pick_column(df, spec.text_cols, "text", path)
        texts = df[tcol].astype(str).tolist()

        # Same function s2_prepare.py::load_raw calls -- see resolve_labels's
        # docstring for the bug this replaced.
        labels = resolve_labels(df, spec, src, path).tolist()

        file_inventory.append({
            "dataset": name,
            "file": _safe_relpath(path, data_root),
            "sha256": sha,
            "n_rows": n_rows_full,
            "n_rows_inspected": len(df),
            "n_columns": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {c: str(df_full[c].dtype) for c in df_full.columns},
            "utf8_decode_failures": n_decode_fail,
        })
        split_to_texts[split_label] = texts
        split_to_labels[split_label] = labels
        progress.log(name, f"{split_label}: {len(texts)} rows, "
                            f"{n_decode_fail} UTF-8 decode failures")

    all_texts = [t for texts in split_to_texts.values() for t in texts]
    all_labels = [lab for labels in split_to_labels.values() for lab in labels]
    quality = cs.quality_inventory(split_to_texts)

    progress.log(name, f"{len(all_texts)} total rows; computing corpus_stats metrics")
    return {
        "dataset": name,
        "task": spec.task,
        "file_inventory": file_inventory,
        "class_distribution_overall": cs.class_distribution(all_labels),
        "class_distribution_per_split": {s: cs.class_distribution(lab)
                                         for s, lab in split_to_labels.items()},
        "length_distribution_clusters": cs.grapheme_lengths(all_texts),
        "char_class_composition": cs.char_class_composition(all_texts),
        "latin_share_per_text": cs.latin_share_per_text(all_texts),
        "quality_inventory": quality.__dict__,
        "emoji_bearing_overall": cs.emoji_bearing_count(all_texts),
        "emoji_bearing_per_split": {s: {
            "n_texts": len(t),
            "n_emoji_bearing": cs.emoji_bearing_count(t),
            "pct_emoji_bearing": (100.0 * cs.emoji_bearing_count(t) / len(t)) if t else 0.0,
        } for s, t in split_to_texts.items()},
        "n_total_rows": len(all_texts),
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_tables(results: dict[str, dict], tables_dir: str) -> None:
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, "file_inventory.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "file", "sha256", "n_rows", "n_rows_inspected",
                    "n_columns", "columns", "utf8_decode_failures"])
        for r in results.values():
            for fi in r["file_inventory"]:
                w.writerow([fi["dataset"], fi["file"], fi["sha256"], fi["n_rows"],
                           fi["n_rows_inspected"], fi["n_columns"],
                           "|".join(fi["columns"]), fi["utf8_decode_failures"]])

    with open(os.path.join(tables_dir, "class_distribution.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split_or_file", "label", "count"])
        for name, r in results.items():
            for split, dist in r["class_distribution_per_split"].items():
                for label, count in dist.items():
                    w.writerow([name, split, label, count])
            for label, count in r["class_distribution_overall"].items():
                w.writerow([name, "ALL", label, count])

    with open(os.path.join(tables_dir, "length_distribution.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "n_grapheme_clusters"])
        for name, r in results.items():
            for n in r["length_distribution_clusters"]:
                w.writerow([name, n])

    with open(os.path.join(tables_dir, "char_class_composition.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "char_class", "count"])
        for name, r in results.items():
            for cls, count in r["char_class_composition"].items():
                w.writerow([name, cls, count])

    with open(os.path.join(tables_dir, "code_mixing_latin_share.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "latin_share"])
        for name, r in results.items():
            for share in r["latin_share_per_text"]:
                w.writerow([name, f"{share:.5f}"])

    with open(os.path.join(tables_dir, "data_quality_inventory.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "n_rows", "n_unique", "n_duplicate_rows",
                    "n_empty_or_whitespace", "n_zero_bangla", "n_replacement_char",
                    "n_emoji_bearing"])
        for name, r in results.items():
            q = r["quality_inventory"]
            w.writerow([name, q["n_rows"], q["n_unique"], q["n_duplicate_rows"],
                       q["n_empty_or_whitespace"], q["n_zero_bangla"],
                       q["n_replacement_char"], q["n_emoji_bearing"]])

    with open(os.path.join(tables_dir, "cross_split_duplicates.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split_pair", "n_identical_texts"])
        for name, r in results.items():
            for pair, count in r["quality_inventory"]["cross_split_duplicates"].items():
                w.writerow([name, pair, count])

    with open(os.path.join(tables_dir, "emoji_bearing_per_split.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split_or_file", "n_texts", "n_emoji_bearing", "pct_emoji_bearing"])
        for name, r in results.items():
            for split, d in r["emoji_bearing_per_split"].items():
                w.writerow([name, split, d["n_texts"], d["n_emoji_bearing"],
                           f"{d['pct_emoji_bearing']:.3f}"])


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown (git rev-parse failed)"


def write_report(results: dict[str, dict], data_root: str, report_path: str) -> None:
    lines = ["# Phase 2, s1 — raw inspection", "",
             "Read-only characterisation of `data/raw/` exactly as downloaded, "
             "before any M2.1/M2.2 cleaning. No verbatim corpus text below.", ""]
    lines.append(f"git rev: `{_git_rev()}`  ·  data root: `{_safe_relpath(data_root, _ROOT)}`")
    lines.append("")
    lines.append("| dataset | files | total rows | UTF-8 decode failures |")
    lines.append("|---|--:|--:|--:|")
    for name, r in results.items():
        n_files = len(r["file_inventory"])
        n_fail = sum(fi["utf8_decode_failures"] for fi in r["file_inventory"])
        lines.append(f"| {name} | {n_files} | {r['n_total_rows']} | {n_fail} |")
    lines.append("")
    lines.append("SHA-256 per source file is in `tables/file_inventory.csv`. See the "
                 "other `tables/*.csv` for class distribution, length distribution, "
                 "character-class composition, code-mixing, data-quality inventory, "
                 "cross-split duplicates, and emoji-bearing counts.")
    _write_utf8(report_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=_DEFAULT_RAW)
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {_OUT_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.out_dir is None:
        args.out_dir = smoke_or_prod(args.limit, _OUT_DIR)
    if args.limit is not None:
        print_smoke_banner("s1_inspect", args.limit, {"out-dir": args.out_dir})

    progress = Progress("s1_inspect")
    tables_dir = os.path.join(args.out_dir, "tables")
    marker = os.path.join(tables_dir, "file_inventory.csv")
    if should_skip(args.out_dir, [marker], args.limit, args.force, progress, "s1_inspect"):
        return 0

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results: dict[str, dict] = {}
    for name in names:
        results[name] = inspect_dataset(name, DATASETS[name], args.data_root,
                                        args.limit, progress)
        progress.log(name, "done")

    progress.log("s1_inspect", "writing tables/")
    write_tables(results, tables_dir)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    write_report(results, args.data_root, os.path.join(args.out_dir, "report.md"))
    write_run_meta(args.out_dir, limited=args.limit is not None, limit=args.limit,
                   n_rows_processed={name: r["n_total_rows"] for name, r in results.items()})
    progress.done()

    for name, r in results.items():
        print(f"\n[{name}] task={r['task']}  n_rows={r['n_total_rows']}")
        print(f"  class_distribution: {r['class_distribution_overall']}")
        q = r["quality_inventory"]
        print(f"  quality: dup_rows={q['n_duplicate_rows']} empty_ws={q['n_empty_or_whitespace']} "
              f"zero_bangla={q['n_zero_bangla']} fffd={q['n_replacement_char']} "
              f"emoji_bearing={q['n_emoji_bearing']}")
        if q["cross_split_duplicates"]:
            print(f"  cross-split/file duplicates: {q['cross_split_duplicates']}")
    print(f"\n[written] {_safe_relpath(tables_dir, _ROOT)}/")
    print(f"[written] {_safe_relpath(os.path.join(args.out_dir, 'report.md'), _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
