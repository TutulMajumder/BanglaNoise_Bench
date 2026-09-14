"""Phase 2, pipeline stage 3: post-preparation description.

Table II (M2.3) plus every s1 measurement recomputed on ``data/final/``,
plus a raw-to-final delta table showing exactly what preparation removed per
dataset per metric. Shares ``noisebench/corpus_stats.py`` with
``s1_inspect_raw.py`` -- the SAME functions, never two implementations that
happen to compute similar things, so the delta is a real comparison.

Also computes what s1 cannot: class balance per split after stratification,
length by class (the box-plot figure), U+FFFD vs. label for bd_shs, the
eligible-unit count per noise type per dataset (the figure that explains
conjunct_split's severity-1 behaviour), and the CER-truncation-cap impact on
the length distribution.

Writes, under ``outputs/s3_analysis/``:

    tables/table2.csv                    Table II main stats, M2.3
    tables/table2_splits.csv
    tables/class_distribution.csv        s1-equivalent, on data/final/
    tables/length_distribution.csv
    tables/char_class_composition.csv
    tables/code_mixing_latin_share.csv
    tables/data_quality_inventory.csv
    tables/emoji_bearing_per_split.csv
    tables/raw_final_delta.csv           the third output: what changed
    tables/class_balance_per_split.csv
    tables/length_by_class.csv
    tables/bdshs_fffd_2x2.csv
    tables/eligible_units_per_noise_type.csv
    tables/cer_truncation_impact.csv
    figures/                             populated later by s5_figures.py
    report.md

    python pipeline/s3_describe.py --limit 20     # smoke run
    python pipeline/s3_describe.py                # real run

CLAUDE.md invariant 9: resumable (skips if tables/table2.csv exists, unless
--force), prints progress to stderr.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
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
    _safe_relpath,
    _write_utf8,
    dataset_final_dir,
)

_DEFAULT_DATA_FINAL = os.path.join(_ROOT, "data", "final")
_S1_DIR = os.path.join(_ROOT, "outputs", "s1_raw_inspection")
_S2_DIR = os.path.join(_ROOT, "outputs", "s2_preparation")
_OUT_DIR = os.path.join(_ROOT, "outputs", "s3_analysis")
ELIGIBLE_UNITS_SAMPLE = 300

#: Footnote for the record: SentNoB's 0/1/2 -> neutral/positive/negative
#: mapping is not printed on the shipped CSVs. Confirmed empirically
#: (2026-09-12) via a lexicon probe -- common, unambiguous Bangla sentiment
#: words (ভালো/সুন্দর/চমৎকার... vs. খারাপ/বাজে/জঘন্য...) checked for
#: co-occurrence with each numeric label. Label 1 shows 4-12x the
#: positive-word hit rate of the other two; label 2 shows the highest
#: negative-word hit rate and the lowest positive/negative ratio. Matches the
#: HuggingFace dataset card; inconsistent with the alternative mapping some
#: other sources cite. See PHASE2_STATUS.md for the full method and numbers.
SENTNOB_LABEL_FOOTNOTE = (
    "SentNoB 0=neutral, 1=positive, 2=negative is not printed on the shipped "
    "CSVs; confirmed empirically via a lexicon probe (common Bangla sentiment "
    "words checked for co-occurrence with each label) -- see PHASE2_STATUS.md "
    "for the method and numbers."
)


# ---------------------------------------------------------------------------
# load data/final/
# ---------------------------------------------------------------------------


def load_final(spec: DatasetSpec, data_final_root: str,
               limit: int | None) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    out_dir = dataset_final_dir(spec, data_final_root)
    split_to_texts: dict[str, list[str]] = {}
    split_to_labels: dict[str, list[str]] = {}
    for split in ("train", "val", "test"):
        path = os.path.join(out_dir, f"{split}.csv")
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if limit is not None:
            df = df.head(limit)
        split_to_texts[split] = df["text"].tolist()
        split_to_labels[split] = df["label"].tolist()
    return split_to_texts, split_to_labels


def describe_dataset(name: str, spec: DatasetSpec, data_final_root: str,
                     limit: int | None, progress: Progress) -> dict:
    progress.log(name, "reading data/final/")
    split_to_texts, split_to_labels = load_final(spec, data_final_root, limit)
    all_texts = [t for texts in split_to_texts.values() for t in texts]
    all_labels_raw = [lab for labels in split_to_labels.values() for lab in labels]
    # data/final labels are already the small canonical int (as strings);
    # map to display names via the registry for Table II / class_distribution.
    all_labels_named = [spec.label_names.get(int(lab), lab) for lab in all_labels_raw]

    quality = cs.quality_inventory(split_to_texts)
    lengths = cs.grapheme_lengths(all_texts)
    n_emoji = cs.emoji_bearing_count(all_texts)
    progress.log(name, f"{len(all_texts)} rows; computing corpus_stats metrics")

    length_by_label: dict[str, list[int]] = {}
    for lab in sorted(set(all_labels_named)):
        texts_for_label = [t for t, x in zip(all_texts, all_labels_named) if x == lab]
        length_by_label[lab] = cs.grapheme_lengths(texts_for_label)

    return {
        "dataset": name,
        "task": spec.task,
        "n_total_rows": len(all_texts),
        "class_distribution_overall": cs.class_distribution(all_labels_named),
        "class_distribution_per_split": {s: cs.class_distribution(
            [spec.label_names.get(int(x), x) for x in labs])
            for s, labs in split_to_labels.items()},
        "length_distribution_clusters": lengths,
        "length_by_label": length_by_label,
        "char_class_composition": cs.char_class_composition(all_texts),
        "latin_share_per_text": cs.latin_share_per_text(all_texts),
        "quality_inventory": quality.__dict__,
        "emoji_bearing_overall": n_emoji,
        "emoji_bearing_per_split": {s: {
            "n_texts": len(t),
            "n_emoji_bearing": cs.emoji_bearing_count(t),
            "pct_emoji_bearing": (100.0 * cs.emoji_bearing_count(t) / len(t)) if t else 0.0,
        } for s, t in split_to_texts.items()},
        "pct_texts_over_cer_cap": (100.0 * sum(1 for n in lengths if n > cs.CER_TRUNCATE_CLUSTERS)
                                   / len(lengths)) if lengths else 0.0,
        "split_sizes": {s: len(t) for s, t in split_to_texts.items()},
        "_split_to_texts": split_to_texts,
        "_split_to_labels_raw": split_to_labels,
        "_all_texts": all_texts,
    }


# ---------------------------------------------------------------------------
# eligible units per noise type + bd_shs U+FFFD 2x2
# ---------------------------------------------------------------------------


def sample_texts(texts: list[str], k: int, seed_key: str) -> list[str]:
    if len(texts) <= k:
        return texts
    rng = random.Random(seed_key)
    return rng.sample(texts, k)


def compute_eligible_units(name: str, all_texts: list[str], progress: Progress) -> dict[str, list[int]]:
    sample = sample_texts(all_texts, ELIGIBLE_UNITS_SAMPLE, f"s3-eligible\x00{name}")
    progress.log(name, f"eligible units per noise type on {len(sample)} texts")
    return cs.eligible_units_per_noise_type(sample)


def compute_bdshs_fffd_2x2(spec: DatasetSpec, all_texts: list[str],
                           all_labels_raw: list[str]) -> dict | None:
    if spec.name != "bd_shs":
        return None
    label_names = spec.label_names
    return cs.flag_vs_label_table(
        all_texts, [int(x) for x in all_labels_raw],
        lambda t: "�" in t, label_names=label_names,
    )


# ---------------------------------------------------------------------------
# raw -> final delta (reads s1's tables, compares to what we just computed)
# ---------------------------------------------------------------------------


def _read_s1_quality(s1_dir: str) -> dict[str, dict]:
    path = os.path.join(s1_dir, "tables", "data_quality_inventory.csv")
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["dataset"]] = row
    return out


def _read_s1_char_class(s1_dir: str) -> dict[str, dict[str, int]]:
    path = os.path.join(s1_dir, "tables", "char_class_composition.csv")
    out: dict[str, dict[str, int]] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.setdefault(row["dataset"], {})[row["char_class"]] = int(row["count"])
    return out


def build_delta(results: dict[str, dict], s1_dir: str) -> list[dict]:
    """Third output: what preparation removed, per dataset per metric. Reads
    s1's already-computed raw-side tables rather than recomputing raw stats
    (s1 already ran); computes the final-side fresh via corpus_stats on
    data/final/ (done in describe_dataset above)."""
    raw_quality = _read_s1_quality(s1_dir)
    raw_charclass = _read_s1_char_class(s1_dir)
    rows: list[dict] = []
    metric_keys = ["n_rows", "n_duplicate_rows", "n_empty_or_whitespace",
                   "n_zero_bangla", "n_replacement_char", "n_emoji_bearing"]
    for name, r in results.items():
        rq = raw_quality.get(name)
        if rq is None:
            continue
        fq = r["quality_inventory"]
        for m in metric_keys:
            raw_v = float(rq[m])
            final_v = float(fq[m])
            rows.append({
                "dataset": name, "metric": m, "raw_value": raw_v, "final_value": final_v,
                "delta": final_v - raw_v,
                "pct_of_raw": (100.0 * final_v / raw_v) if raw_v else 0.0,
            })
        raw_cc = raw_charclass.get(name, {})
        raw_total = sum(raw_cc.values()) or 1
        final_cc = r["char_class_composition"]
        final_total = sum(final_cc.values()) or 1
        for cls in cs.CHAR_CLASSES:
            raw_pct = 100.0 * raw_cc.get(cls, 0) / raw_total
            final_pct = 100.0 * final_cc.get(cls, 0) / final_total
            rows.append({
                "dataset": name, "metric": f"pct_char_{cls}", "raw_value": raw_pct,
                "final_value": final_pct, "delta": final_pct - raw_pct,
                "pct_of_raw": (100.0 * final_pct / raw_pct) if raw_pct else 0.0,
            })
    return rows


# ---------------------------------------------------------------------------
# Table II rendering (M2.3)
# ---------------------------------------------------------------------------


def _fmt_dist(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def render_table2_markdown(results: dict[str, dict], provenance: dict,
                           bdshs_fffd: dict | None) -> str:
    L: list[str] = ["## Table II — dataset statistics after cleaning (METHODOLOGY M2.3)", ""]
    L.append(f"> {SENTNOB_LABEL_FOOTNOTE}")
    L.append("")
    L.append("| dataset | task | n (final) | class distribution | mean len | median len | "
             "emoji rate | dupes removed | short removed | zero-Bangla removed |")
    L.append("|---|---|--:|---|--:|--:|--:|--:|--:|--:|")
    for name, r in results.items():
        prov = provenance.get("datasets", {}).get(name, {})
        rows_removed = prov.get("rows_removed", {})
        lengths = r["length_distribution_clusters"]
        mean_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
        emoji_rate = r["emoji_bearing_overall"] / r["n_total_rows"] if r["n_total_rows"] else 0.0
        L.append(f"| {name} | {r['task']} | {r['n_total_rows']} | "
                 f"{_fmt_dist(r['class_distribution_overall'])} | {mean_len:.1f} | "
                 f"{median_len} | {emoji_rate:.3%} | "
                 f"{rows_removed.get('exact_duplicates', '?')} | "
                 f"{rows_removed.get('short_lt_3_clusters', '?')} | "
                 f"{rows_removed.get('zero_bangla_chars', '?')} |")
    L.append("")
    if bdshs_fffd:
        L.append("### BD-SHS: U+FFFD vs. hate-speech label (2x2)")
        L.append("")
        L.append("Kept per METHODOLOGY decision (2026-09-12): U+FFFD is authentic "
                 "user-generated noise (likely mangled emoji from the original "
                 "scrape), not a reading artefact -- recorded to check whether its "
                 "presence is label-correlated.")
        L.append("")
        label_cols = sorted({k for g in bdshs_fffd.values() for k in g["label_dist_pct"]})
        L.append("| group | n | " + " | ".join(label_cols) + " |")
        L.append("|---|--:|" + "--:|" * len(label_cols))
        for key, g in bdshs_fffd.items():
            cells = " | ".join(f"{g['label_dist_pct'].get(c, 0.0):.2f}%" for c in label_cols)
            L.append(f"| {key} | {g['n']} | {cells} |")
        L.append("")
    host = max(results, key=lambda n: results[n]["emoji_bearing_overall"] / max(1, results[n]["n_total_rows"])) \
        if results else None
    if host:
        rate = results[host]["emoji_bearing_overall"] / max(1, results[host]["n_total_rows"])
        L.append(f"**N10 (emoji) experiment host: `{host}`** — highest emoji rate ({rate:.3%}).")
        L.append("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------


def write_tables(results: dict[str, dict], provenance: dict, delta_rows: list[dict],
                 eligible: dict[str, dict[str, list[int]]], bdshs_fffd: dict | None,
                 tables_dir: str) -> None:
    os.makedirs(tables_dir, exist_ok=True)

    with open(os.path.join(tables_dir, "table2.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "task", "n", "class_distribution", "mean_len", "median_len",
                    "emoji_rate", "dupes_removed", "short_removed", "non_bangla_removed"])
        for name, r in results.items():
            prov = provenance.get("datasets", {}).get(name, {})
            rows_removed = prov.get("rows_removed", {})
            lengths = r["length_distribution_clusters"]
            mean_len = (sum(lengths) / len(lengths)) if lengths else 0.0
            median_len = sorted(lengths)[len(lengths) // 2] if lengths else 0
            emoji_rate = r["emoji_bearing_overall"] / r["n_total_rows"] if r["n_total_rows"] else 0.0
            w.writerow([name, r["task"], r["n_total_rows"], _fmt_dist(r["class_distribution_overall"]),
                       f"{mean_len:.2f}", median_len, f"{emoji_rate:.5f}",
                       rows_removed.get("exact_duplicates"), rows_removed.get("short_lt_3_clusters"),
                       rows_removed.get("zero_bangla_chars")])

    with open(os.path.join(tables_dir, "table2_splits.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split_source", "train", "val", "test"])
        for name, r in results.items():
            prov = provenance.get("datasets", {}).get(name, {})
            ss = r["split_sizes"]
            w.writerow([name, prov.get("split_source"), ss.get("train"), ss.get("val"), ss.get("test")])

    with open(os.path.join(tables_dir, "class_distribution.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split", "label", "count"])
        for name, r in results.items():
            for split, dist in r["class_distribution_per_split"].items():
                for label, count in dist.items():
                    w.writerow([name, split, label, count])

    with open(os.path.join(tables_dir, "class_balance_per_split.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split", "label", "count", "pct_within_split"])
        for name, r in results.items():
            for split, dist in r["class_distribution_per_split"].items():
                total = sum(dist.values()) or 1
                for label, count in dist.items():
                    w.writerow([name, split, label, count, f"{100.0 * count / total:.2f}"])

    with open(os.path.join(tables_dir, "length_distribution.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "n_grapheme_clusters"])
        for name, r in results.items():
            for n in r["length_distribution_clusters"]:
                w.writerow([name, n])

    with open(os.path.join(tables_dir, "length_by_class.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "label", "n_grapheme_clusters"])
        for name, r in results.items():
            for label, lengths in r["length_by_label"].items():
                for n in lengths:
                    w.writerow([name, label, n])

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

    with open(os.path.join(tables_dir, "emoji_bearing_per_split.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "split", "n_texts", "n_emoji_bearing", "pct_emoji_bearing"])
        for name, r in results.items():
            for split, d in r["emoji_bearing_per_split"].items():
                w.writerow([name, split, d["n_texts"], d["n_emoji_bearing"], f"{d['pct_emoji_bearing']:.3f}"])

    with open(os.path.join(tables_dir, "raw_final_delta.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "metric", "raw_value", "final_value", "delta", "pct_of_raw"])
        for row in delta_rows:
            w.writerow([row["dataset"], row["metric"], f"{row['raw_value']:.4f}",
                       f"{row['final_value']:.4f}", f"{row['delta']:.4f}", f"{row['pct_of_raw']:.2f}"])

    with open(os.path.join(tables_dir, "eligible_units_per_noise_type.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "noise_type", "n_eligible"])
        for name, per_nt in eligible.items():
            for nt, values in per_nt.items():
                for v in values:
                    w.writerow([name, nt, v])

    with open(os.path.join(tables_dir, "cer_truncation_impact.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "cap_grapheme_clusters", "n_texts", "pct_texts_over_cap"])
        for name, r in results.items():
            w.writerow([name, cs.CER_TRUNCATE_CLUSTERS, r["n_total_rows"],
                       f"{r['pct_texts_over_cer_cap']:.3f}"])

    if bdshs_fffd:
        with open(os.path.join(tables_dir, "bdshs_fffd_2x2.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            label_cols = sorted({k for g in bdshs_fffd.values() for k in g["label_dist_pct"]})
            w.writerow(["group", "n", *label_cols])
            for key, g in bdshs_fffd.items():
                w.writerow([key, g["n"], *[f"{g['label_dist_pct'].get(c, 0.0):.3f}" for c in label_cols]])


def write_report(results: dict[str, dict], provenance: dict, bdshs_fffd: dict | None,
                 report_path: str) -> None:
    lines = ["# Phase 2, s3 — post-preparation description", ""]
    lines.append(render_table2_markdown(results, provenance, bdshs_fffd))
    lines.append("## Raw → final delta")
    lines.append("")
    lines.append("See `tables/raw_final_delta.csv` for the full per-dataset, per-metric "
                 "comparison against s1's raw-corpus numbers -- the defence of every "
                 "cleaning decision. Also: `tables/class_balance_per_split.csv`, "
                 "`tables/length_by_class.csv`, `tables/eligible_units_per_noise_type.csv` "
                 "(explains conjunct_split's severity-1 behaviour), and "
                 "`tables/cer_truncation_impact.csv` (fraction of texts affected by the "
                 "300-cluster CER cap, per dataset).")
    lines.append("")
    _write_utf8(report_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-final-root", default=None,
                    help=f"default: {_DEFAULT_DATA_FINAL}, or the .smoke/ mirror under --limit")
    ap.add_argument("--s1-dir", default=None,
                    help=f"default: {_S1_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--s2-dir", default=None,
                    help=f"default: {_S2_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {_OUT_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--dataset", default="all", choices=[*DATASETS, "all"])
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.data_final_root is None:
        args.data_final_root = smoke_or_prod(args.limit, _DEFAULT_DATA_FINAL)
    if args.s1_dir is None:
        args.s1_dir = smoke_or_prod(args.limit, _S1_DIR)
    if args.s2_dir is None:
        args.s2_dir = smoke_or_prod(args.limit, _S2_DIR)
    if args.out_dir is None:
        args.out_dir = smoke_or_prod(args.limit, _OUT_DIR)
    if args.limit is not None:
        print_smoke_banner("s3_describe", args.limit, {
            "data-final-root": args.data_final_root, "s1-dir": args.s1_dir,
            "s2-dir": args.s2_dir, "out-dir": args.out_dir,
        })

    progress = Progress("s3_describe")
    tables_dir = os.path.join(args.out_dir, "tables")
    marker = os.path.join(tables_dir, "table2.csv")
    if should_skip(args.out_dir, [marker], args.limit, args.force, progress, "s3_describe"):
        return 0

    import json
    prov_path = os.path.join(args.s2_dir, "provenance.json")
    if not os.path.exists(prov_path):
        print(f"no {prov_path} — run pipeline/s2_prepare.py first", file=sys.stderr)
        return 2
    provenance = json.load(open(prov_path, encoding="utf-8"))

    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results: dict[str, dict] = {}
    eligible: dict[str, dict] = {}
    bdshs_fffd = None
    for name in names:
        spec = DATASETS[name]
        r = describe_dataset(name, spec, args.data_final_root, args.limit, progress)
        results[name] = r
        eligible[name] = compute_eligible_units(name, r["_all_texts"], progress)
        fffd = compute_bdshs_fffd_2x2(spec, r["_all_texts"],
                                      [x for labs in r["_split_to_labels_raw"].values() for x in labs])
        if fffd is not None:
            bdshs_fffd = fffd
        progress.log(name, "done")

    progress.log("s3_describe", "building raw->final delta")
    delta_rows = build_delta(results, args.s1_dir)

    progress.log("s3_describe", "writing tables/")
    write_tables(results, provenance, delta_rows, eligible, bdshs_fffd, tables_dir)
    os.makedirs(os.path.join(args.out_dir, "figures"), exist_ok=True)
    write_report(results, provenance, bdshs_fffd, os.path.join(args.out_dir, "report.md"))
    write_run_meta(args.out_dir, limited=args.limit is not None, limit=args.limit,
                   n_rows_processed={name: r["n_total_rows"] for name, r in results.items()})
    progress.done()

    print(render_table2_markdown(results, provenance, bdshs_fffd))
    print(f"[written] {_safe_relpath(tables_dir, _ROOT)}/")
    print(f"[written] {_safe_relpath(os.path.join(args.out_dir, 'report.md'), _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
