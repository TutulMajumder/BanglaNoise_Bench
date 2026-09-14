"""Phase 2, pipeline stage 5: figures.

Reads ONLY the ``.csv`` files under ``outputs/*/tables/`` -- never touches a
corpus, so figures can be restyled without re-running anything (this is why
s1/s3/s4 write CSVs at all, not just markdown).

Bangla font: matplotlib's default fonts have no Bengali glyphs -- Bangla
labels render as empty boxes. Resolved ONCE at import via
``matplotlib.font_manager``, in this order: Noto Sans Bengali, Nirmala UI,
Kalpurush. If none is found, raises -- never falls back silently. The
resolved font is printed to stderr on every run. ASCII labels are used
wherever meaning survives (noise types already have English names); Bangla
is reserved for where the character IS the data (homophone sets, matra
pairs), and those figures are suffixed ``_bn``.

Every figure: ``.png`` (300dpi) + ``.pdf`` (vector), same basename. Palette
and IEEE column widths from ``noisebench/plotstyle.py`` (validated in
``docs/Figure_Plan_Evaluation.md``) -- no new colours invented here.

    python pipeline/s5_figures.py --limit 20     # smoke run (tiny tables)
    python pipeline/s5_figures.py                # real run

CLAUDE.md invariant 9: resumable (skips a figure whose .pdf already exists
AND was confirmed produced by a non-limited run, unless --force), prints
progress to stderr. ``--limit`` redirects the default ``--s1-dir``/
``--s2-dir``/``--s3-dir`` under ``.smoke/`` (mirroring whatever s1-s4 wrote
there under the same ``--limit``) -- figure GENERATION itself is cheap
regardless of table size, but reading real production tables during a
"smoke" run would defeat the point and risks writing figures into
production ``figures/`` folders.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench import plotstyle as ps  # noqa: E402
from pipeline._common import (  # noqa: E402
    Progress,
    add_common_args,
    print_smoke_banner,
    read_run_meta,
    smoke_or_prod,
    write_run_meta,
)

_S1_DIR = os.path.join(_ROOT, "outputs", "s1_raw_inspection")
_S2_DIR = os.path.join(_ROOT, "outputs", "s2_preparation")
_S3_DIR = os.path.join(_ROOT, "outputs", "s3_analysis")

_BENGALI_FONT_CANDIDATES = ("Noto Sans Bengali", "Nirmala UI", "Kalpurush")


def resolve_bengali_font() -> str:
    """Never falls back silently -- raises naming the fonts to install."""
    available = {f.name for f in fm.fontManager.ttflist}
    for name in _BENGALI_FONT_CANDIDATES:
        if name in available:
            return name
    raise RuntimeError(
        "No Bengali-capable font found (tried: " + ", ".join(_BENGALI_FONT_CANDIDATES) +
        "). Bangla glyph labels (homophone sets, matra pairs) would render as "
        "empty boxes. Install one: Windows ships Nirmala UI by default (if this "
        "still fails, the matplotlib font cache is stale -- delete "
        "~/.cache/matplotlib and retry); Linux: apt install fonts-noto-bengali "
        "or fonts-beng; Kalpurush: https://www.omicronlab.com/bangla-fonts.html."
    )


def _apply_base_style() -> None:
    plt.rcParams.update(ps.MPL_RCPARAMS)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def _save(fig, out_dir: str, basename: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    png = os.path.join(out_dir, f"{basename}.png")
    pdf = os.path.join(out_dir, f"{basename}.pdf")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _skip(paths: list[str], limit: int | None, force: bool, progress: Progress, label: str) -> bool:
    """Mirrors pipeline._common.should_skip's provenance check: a figure that
    already exists is only trusted as "done" if it wasn't produced by a
    limited run (or of unknown provenance -- predates this fix)."""
    if force:
        return False
    if not all(os.path.exists(p) for p in paths):
        return False
    if limit is not None:
        return False
    out_dir = os.path.dirname(paths[0])
    meta = read_run_meta(out_dir)
    if meta is None or meta.get("limited"):
        progress.log(label, f"existing figure(s) in {out_dir} came from a "
                            "limited run (or unknown provenance) -- regenerating")
        return False
    progress.log(label, f"skipping -- {paths[0]} exists; use --force to redo")
    return True


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_length_ecdf(tables_dir: str, out_dir: str, basename: str, title: str,
                    cap: int | None, limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "length_distribution.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(ps.COL1, 2.6))
    for name in sorted(df["dataset"].unique()):
        vals = sorted(df.loc[df["dataset"] == name, "n_grapheme_clusters"])
        n = len(vals)
        if n == 0:
            continue
        y = [(i + 1) / n for i in range(n)]
        st = ps.DATASET_STYLE.get(name, dict(color=ps.GRAY, ls="-", marker=None))
        ax.plot(vals, y, color=st["color"], ls=st["ls"], lw=1.4, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Text length (grapheme clusters, log scale)")
    ax.set_ylabel("ECDF")
    ax.set_title(title, pad=4)
    ax.grid(axis="both", alpha=0.5)
    if cap is not None:
        ax.axvline(cap, color=ps.AXIS, lw=0.8, ls="--", zorder=1)
        ax.annotate(f"CER cap = {cap}", (cap, 0.05), xytext=(4, 0),
                    textcoords="offset points", fontsize=6, color=ps.INK2)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_char_class_stacked_bar(tables_dir: str, out_dir: str, basename: str,
                               limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "char_class_composition.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    pivot = df.pivot(index="dataset", columns="char_class", values="count").fillna(0)
    pivot = pivot.div(pivot.sum(axis=1), axis=0) * 100.0
    order = [c for c in cs_char_classes() if c in pivot.columns]
    pivot = pivot[order]

    fig, ax = plt.subplots(figsize=(ps.COL1, 2.6))
    bottom = pd.Series(0.0, index=pivot.index)
    palette = [ps.BLUE, ps.ORANGE, ps.AQUA, ps.GRAY, "#c3c2b7", "#8ec1e8", "#f2b896", "#0b0b0b"]
    for i, cls in enumerate(order):
        ax.barh(pivot.index, pivot[cls], left=bottom, color=palette[i % len(palette)],
               label=cls, height=0.6)
        bottom += pivot[cls]
    ax.set_xlabel("Share of all characters (%)")
    ax.set_title("Character-class composition", pad=4)
    ax.legend(frameon=False, fontsize=6, loc="upper center",
             bbox_to_anchor=(0.5, -0.18), ncol=4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def cs_char_classes() -> tuple[str, ...]:
    from noisebench.corpus_stats import CHAR_CLASSES
    return CHAR_CLASSES


def fig_code_mixing_ecdf(tables_dir: str, out_dir: str, basename: str,
                         limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "code_mixing_latin_share.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(ps.COL1, 2.6))
    for name in sorted(df["dataset"].unique()):
        vals = sorted(df.loc[df["dataset"] == name, "latin_share"])
        n = len(vals)
        if n == 0:
            continue
        y = [(i + 1) / n for i in range(n)]
        st = ps.DATASET_STYLE.get(name, dict(color=ps.GRAY, ls="-", marker=None))
        ax.plot(vals, y, color=st["color"], ls=st["ls"], lw=1.4, label=name)
    ax.set_xlabel("Per-text Latin-character share")
    ax.set_ylabel("ECDF")
    ax.set_title("Code-mixing (Latin-character share)", pad=4)
    ax.grid(axis="both", alpha=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_class_balance_per_split(tables_dir: str, out_dir: str, basename: str,
                                limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "class_balance_per_split.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(ps.COL2, 2.2), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        splits = ["train", "val", "test"]
        labels = sorted(sub["label"].unique())
        bottom = [0.0] * len(splits)
        palette = [ps.BLUE, ps.ORANGE, ps.AQUA, ps.GRAY]
        for i, lab in enumerate(labels):
            vals = [sub[(sub["split"] == s) & (sub["label"] == lab)]["pct_within_split"].sum()
                   for s in splits]
            ax.bar(splits, vals, bottom=bottom, color=palette[i % len(palette)], label=str(lab))
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_title(name, pad=4, fontsize=7)
        ax.grid(axis="y", alpha=0.5)
    axes[0].set_ylabel("% within split")
    handles, labs = axes[0].get_legend_handles_labels()
    fig.legend(handles, labs, frameon=False, fontsize=6, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=4)
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_length_by_class_boxplot(tables_dir: str, out_dir: str, basename: str,
                                limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "length_by_class.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    datasets = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(ps.COL2, 2.4), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, name in zip(axes, datasets):
        sub = df[df["dataset"] == name]
        labels = sorted(sub["label"].unique())
        data = [sub[sub["label"] == lab]["n_grapheme_clusters"].values for lab in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False,
                        widths=0.5)
        for patch in bp["boxes"]:
            patch.set_facecolor(ps.GRAY)
            patch.set_alpha(0.3)
            patch.set_edgecolor(ps.INK2)
        for med in bp["medians"]:
            med.set_color(ps.ORANGE)
        ax.set_title(name, pad=4, fontsize=7)
        ax.tick_params(axis="x", labelrotation=30, labelsize=6)
        ax.grid(axis="y", alpha=0.5)
    axes[0].set_ylabel("Grapheme clusters")
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_bdshs_fffd_2x2(tables_dir: str, out_dir: str, basename: str,
                       limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "bdshs_fffd_2x2.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    label_cols = [c for c in df.columns if c not in ("group", "n")]
    fig, ax = plt.subplots(figsize=(ps.COL1, 2.2))
    x = range(len(df))
    bottom = [0.0] * len(df)
    palette = [ps.BLUE, ps.ORANGE, ps.AQUA, ps.GRAY]
    for i, col in enumerate(label_cols):
        vals = (df[col] * 100.0).tolist()
        ax.bar(x, vals, bottom=bottom, color=palette[i % len(palette)], label=col, width=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["group"].tolist(), fontsize=7)
    ax.set_ylabel("% of group")
    ax.set_title("BD-SHS: U+FFFD vs. hate-speech label", pad=4, fontsize=8)
    ax.legend(frameon=False, fontsize=6)
    ax.grid(axis="y", alpha=0.5)
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_eligible_units_per_noise_type(tables_dir: str, out_dir: str, basename: str,
                                      limit: int | None, force: bool, progress: Progress) -> None:
    csv_path = os.path.join(tables_dir, "eligible_units_per_noise_type.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    noise_types = sorted(df["noise_type"].unique(),
                         key=lambda nt: -df[df["noise_type"] == nt]["n_eligible"].mean())
    fig, ax = plt.subplots(figsize=(ps.COL2, 2.8))
    data = [df[df["noise_type"] == nt]["n_eligible"].values for nt in noise_types]
    bp = ax.boxplot(data, tick_labels=noise_types, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(ps.GRAY)
        patch.set_alpha(0.3)
        patch.set_edgecolor(ps.INK2)
    for med in bp["medians"]:
        med.set_color(ps.ORANGE)
    ax.set_ylabel("Eligible units per text")
    ax.set_title("Eligible units per noise type (all datasets pooled)", pad=4)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6)
    ax.grid(axis="y", alpha=0.5)
    _save(fig, out_dir, basename)
    progress.log(basename, "written")


def fig_n5_homophone_firing_bn(tables_dir: str, out_dir: str, basename: str,
                               bengali_font: str, limit: int | None, force: bool, progress: Progress) -> None:
    """The character IS the data here (homophone set names) -- Bangla glyphs,
    per METHODOLOGY plotting rules. Suffixed _bn."""
    csv_path = os.path.join(tables_dir, "m2_4_n5.csv")
    if not os.path.exists(csv_path):
        return
    if _skip([os.path.join(out_dir, f"{basename}.pdf")], limit, force, progress, basename):
        return
    df = pd.read_csv(csv_path)
    sets = df["homophone_set"].unique().tolist()
    datasets = sorted(df["dataset"].unique())
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": [bengali_font]}):
        fig, ax = plt.subplots(figsize=(ps.COL2, 2.6))
        width = 0.8 / max(1, len(datasets))
        palette = [ps.BLUE, ps.ORANGE, ps.AQUA, ps.GRAY]
        for i, name in enumerate(datasets):
            sub = df[df["dataset"] == name].set_index("homophone_set").reindex(sets)
            xs = [j + i * width for j in range(len(sets))]
            ax.bar(xs, sub["eligible_per_1k_texts"], width=width,
                  color=palette[i % len(palette)], label=name)
        ax.set_xticks([j + width * (len(datasets) - 1) / 2 for j in range(len(sets))])
        ax.set_xticklabels(sets, fontsize=9)
        ax.set_ylabel("Eligible occurrences per 1,000 texts")
        ax.set_title("N5 homophone-set firing rate", pad=4)
        ax.legend(frameon=False, fontsize=6)
        ax.grid(axis="y", alpha=0.5)
        _save(fig, out_dir, basename)
    progress.log(basename, "written")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s1-dir", default=None,
                    help=f"default: {_S1_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--s2-dir", default=None,
                    help=f"default: {_S2_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--s3-dir", default=None,
                    help=f"default: {_S3_DIR}, or the .smoke/ mirror under --limit")
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.s1_dir is None:
        args.s1_dir = smoke_or_prod(args.limit, _S1_DIR)
    if args.s2_dir is None:
        args.s2_dir = smoke_or_prod(args.limit, _S2_DIR)
    if args.s3_dir is None:
        args.s3_dir = smoke_or_prod(args.limit, _S3_DIR)
    if args.limit is not None:
        print_smoke_banner("s5_figures", args.limit, {
            "s1-dir": args.s1_dir, "s2-dir": args.s2_dir, "s3-dir": args.s3_dir,
        })

    progress = Progress("s5_figures")
    _apply_base_style()

    bengali_font = resolve_bengali_font()
    print(f"[s5_figures] resolved Bengali font: {bengali_font}", file=sys.stderr)

    s1_tables = os.path.join(args.s1_dir, "tables")
    s1_figs = os.path.join(args.s1_dir, "figures")
    s2_tables = os.path.join(args.s2_dir, "tables")
    s2_figs = os.path.join(args.s2_dir, "figures")
    s3_tables = os.path.join(args.s3_dir, "tables")
    s3_figs = os.path.join(args.s3_dir, "figures")

    fig_length_ecdf(s1_tables, s1_figs, "length_ecdf_raw", "Length ECDF (raw)",
                    None, args.limit, args.force, progress)
    fig_char_class_stacked_bar(s1_tables, s1_figs, "char_class_composition_raw",
                               args.limit, args.force, progress)
    fig_code_mixing_ecdf(s1_tables, s1_figs, "code_mixing_latin_ecdf_raw",
                         args.limit, args.force, progress)
    write_run_meta(s1_figs, limited=args.limit is not None, limit=args.limit, n_rows_processed=None)

    fig_n5_homophone_firing_bn(s2_tables, s2_figs, "n5_homophone_firing_bn",
                               bengali_font, args.limit, args.force, progress)
    write_run_meta(s2_figs, limited=args.limit is not None, limit=args.limit, n_rows_processed=None)

    fig_length_ecdf(s3_tables, s3_figs, "length_ecdf_final", "Length ECDF (final)",
                    300, args.limit, args.force, progress)
    fig_char_class_stacked_bar(s3_tables, s3_figs, "char_class_composition_final",
                               args.limit, args.force, progress)
    fig_code_mixing_ecdf(s3_tables, s3_figs, "code_mixing_latin_ecdf_final",
                         args.limit, args.force, progress)
    fig_class_balance_per_split(s3_tables, s3_figs, "class_balance_per_split",
                                args.limit, args.force, progress)
    fig_length_by_class_boxplot(s3_tables, s3_figs, "length_by_class",
                                args.limit, args.force, progress)
    fig_bdshs_fffd_2x2(s3_tables, s3_figs, "bdshs_fffd_2x2",
                       args.limit, args.force, progress)
    fig_eligible_units_per_noise_type(s3_tables, s3_figs, "eligible_units_per_noise_type",
                                      args.limit, args.force, progress)
    write_run_meta(s3_figs, limited=args.limit is not None, limit=args.limit, n_rows_processed=None)

    progress.done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
