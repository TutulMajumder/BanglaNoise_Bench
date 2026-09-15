#!/usr/bin/env python3
"""
p4_analysis.py -- bangla-noisebench R1/R2/R6 analysis.

Reads results/results.csv and results/curves.csv, writes publication figures
(PDF + PNG) and LaTeX tables.

Usage:
    python pipeline/p4_analysis.py --results results --out analysis
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# --------------------------------------------------------------------------
# Presentation constants
# --------------------------------------------------------------------------

# Categorical palette validated for CVD separation (Okabe-Ito subset,
# ordered so every adjacent pair clears delta-E 10 under protan/deutan/tritan).
MODEL_ORDER = ["banglabert", "banglishbert", "char_ngram", "mbert", "xlmr"]
MODEL_LABEL = {
    "banglabert": "BanglaBERT",
    "banglishbert": "BanglishBERT",
    "char_ngram": "char n-gram + SVM",
    "mbert": "mBERT",
    "xlmr": "XLM-R",
}
MODEL_COLOR = {
    "banglabert": "#0072B2",
    "banglishbert": "#E69F00",
    "char_ngram": "#009E73",
    "mbert": "#D55E00",
    "xlmr": "#CC79A7",
}
# Secondary encoding so the figures survive greyscale printing.
MODEL_MARKER = {
    "banglabert": "o",
    "banglishbert": "s",
    "char_ngram": "D",
    "mbert": "^",
    "xlmr": "v",
}
MODEL_DASH = {
    "banglabert": (None, None),
    "banglishbert": (5, 2),
    "char_ngram": (None, None),
    "mbert": (1.5, 1.5),
    "xlmr": (6, 2, 1.5, 2),
}

TRANSFORMERS = ["banglabert", "banglishbert", "mbert", "xlmr"]

TASK_ORDER = ["sentnob", "bd_shs", "banfakenews"]
TASK_LABEL = {
    "sentnob": "SentNoB (3-class sentiment)",
    "bd_shs": "BD-SHS (hate speech)",
    "banfakenews": "BanFakeNews (fake news)",
}

NOISE_ORDER = [
    "char_insert",
    "char_delete",
    "char_substitute",
    "char_transpose",
    "homophone_confuse",
    "matra_perturb",
    "conjunct_split",
    "elongate",
    "whitespace_error",
]
NOISE_LABEL = {
    "char_insert": "Character insertion",
    "char_delete": "Character deletion",
    "char_substitute": "Character substitution",
    "char_transpose": "Character transposition",
    "homophone_confuse": "Homophone confusion",
    "matra_perturb": "Matra perturbation",
    "conjunct_split": "Conjunct splitting",
    "elongate": "Elongation",
    "whitespace_error": "Whitespace error",
}

INK = "#1a1a1a"
INK_MUTED = "#6b6b6b"
GRID = "#e3e3e0"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load(results_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = pd.read_csv(os.path.join(results_dir, "results.csv"))
    cur = pd.read_csv(os.path.join(results_dir, "curves.csv"))

    # A job that was re-trained after a session timeout appends a second set of
    # epoch rows under the same run_id. Keep the surviving (last) training run.
    before = len(cur)
    cur = cur.drop_duplicates(subset=["run_id", "epoch"], keep="last")
    if len(cur) != before:
        print(f"[curves] dropped {before - len(cur)} duplicate epoch rows")

    res["is_clean"] = res["noise_type"].isna()
    return res, cur


def check(res: pd.DataFrame) -> None:
    """Fail loudly if the grid is incomplete -- never plot a partial grid."""
    n_runs = res.run_id.nunique()
    assert n_runs == 45, f"expected 45 runs, found {n_runs}"
    assert res.macro_f1.isna().sum() == 0, "NaN macro_f1 present"
    dup = res.duplicated(subset=["run_id", "normalizer", "noise_type", "severity"]).sum()
    assert dup == 0, f"{dup} duplicate cells"
    noisy = res[~res.is_clean]
    counts = noisy.groupby("severity").size()
    assert counts.nunique() == 1, f"ragged severity counts: {counts.to_dict()}"
    print(f"[check] OK -- {len(res)} rows, {n_runs} runs, grid complete")


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def clean_baseline(res: pd.DataFrame) -> pd.DataFrame:
    """Mean/std macro-F1 on unperturbed test data, over seeds."""
    c = res[res.is_clean]
    g = c.groupby(["task", "model"])["macro_f1"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    return out


def curve_table(res: pd.DataFrame, normalizer: str = "on") -> pd.DataFrame:
    """Severity 0..5 macro-F1 per (task, model, noise_type), averaged over seeds.

    Severity 0 is the clean baseline, repeated into every noise_type so each
    panel starts from the same anchor.
    """
    noisy = res[(~res.is_clean) & (res.normalizer == normalizer)]
    agg = (
        noisy.groupby(["task", "model", "noise_type", "severity"])["macro_f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg["std"] = agg["std"].fillna(0.0)

    base = clean_baseline(res).rename(columns={"mean": "mean", "std": "std"})
    rows = []
    for _, r in base.iterrows():
        for nt in NOISE_ORDER:
            rows.append(
                {
                    "task": r.task,
                    "model": r.model,
                    "noise_type": nt,
                    "severity": 0,
                    "mean": r["mean"],
                    "std": r["std"],
                }
            )
    return pd.concat([pd.DataFrame(rows), agg], ignore_index=True)


def robustness_table(res: pd.DataFrame, normalizer: str = "on") -> pd.DataFrame:
    """Absolute drop and relative retention at severity 5, vs clean."""
    base = clean_baseline(res)[["task", "model", "mean"]].rename(
        columns={"mean": "clean_f1"}
    )
    s5 = (
        res[(~res.is_clean) & (res.normalizer == normalizer) & (res.severity == 5)]
        .groupby(["task", "model", "noise_type"])["macro_f1"]
        .mean()
        .reset_index()
        .rename(columns={"macro_f1": "f1_s5"})
    )
    m = s5.merge(base, on=["task", "model"])
    m["abs_drop"] = m["clean_f1"] - m["f1_s5"]
    m["retention"] = m["f1_s5"] / m["clean_f1"]
    return m


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _style_axis(ax, ylim=None):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    if ylim:
        ax.set_ylim(*ylim)


def _model_legend(fig, ncol=5, y=0.0, models=None, kind="line"):
    models = models or MODEL_ORDER
    if kind == "patch":
        handles = [Patch(facecolor=MODEL_COLOR[m], label=MODEL_LABEL[m]) for m in models]
    else:
        handles = [
            Line2D(
                [], [],
                color=MODEL_COLOR[m],
                marker=MODEL_MARKER[m],
                markersize=4.5,
                linewidth=1.6,
                dashes=MODEL_DASH[m] if MODEL_DASH[m][0] else (),
                label=MODEL_LABEL[m],
            )
            for m in models
        ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        columnspacing=1.6,
        handlelength=2.4,
    )


def fig_degradation(ct: pd.DataFrame, task: str, out: str) -> None:
    """3x3 grid of noise types for one dataset; one line per model."""
    sub = ct[ct.task == task]
    ymin = max(0.0, sub["mean"].min() - 0.06)
    ymax = min(1.0, sub["mean"].max() + 0.04)

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, nt in zip(axes.ravel(), NOISE_ORDER):
        for m in MODEL_ORDER:
            s = sub[(sub.model == m) & (sub.noise_type == nt)].sort_values("severity")
            if s.empty:
                continue
            dashes = MODEL_DASH[m]
            ax.plot(
                s.severity,
                s["mean"],
                color=MODEL_COLOR[m],
                marker=MODEL_MARKER[m],
                markersize=3.6,
                markeredgecolor=SURFACE,
                markeredgewidth=0.6,
                linewidth=1.6,
                dashes=dashes if dashes[0] else (),
                zorder=3,
            )
            ax.fill_between(
                s.severity,
                s["mean"] - s["std"],
                s["mean"] + s["std"],
                color=MODEL_COLOR[m],
                alpha=0.13,
                linewidth=0,
                zorder=2,
            )
        ax.set_title(NOISE_LABEL[nt], loc="left", color=INK)
        _style_axis(ax, (ymin, ymax))
        ax.set_xticks(range(6))

    for ax in axes[-1]:
        ax.set_xlabel("Noise severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")

    fig.suptitle(
        f"Robustness degradation — {TASK_LABEL[task]}",
        x=0.012, y=0.997, ha="left", fontsize=10.5, color=INK,
    )
    fig.text(
        0.012, 0.963,
        "Mean over 3 seeds; band = ±1 s.d. Severity 0 is the unperturbed test set.",
        ha="left", fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    _model_legend(fig, y=-0.004)
    _save(fig, out)


def fig_whitespace(ct: pd.DataFrame, out: str) -> None:
    """Headline: whitespace_error across all three datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, task in zip(axes, TASK_ORDER):
        s0 = ct[(ct.task == task) & (ct.noise_type == "whitespace_error")]
        for m in MODEL_ORDER:
            s = s0[s0.model == m].sort_values("severity")
            dashes = MODEL_DASH[m]
            ax.plot(
                s.severity, s["mean"],
                color=MODEL_COLOR[m], marker=MODEL_MARKER[m], markersize=4.2,
                markeredgecolor=SURFACE, markeredgewidth=0.7,
                linewidth=1.9 if m == "char_ngram" else 1.6,
                dashes=dashes if dashes[0] else (), zorder=3,
            )
        ax.set_title(TASK_LABEL[task], loc="left", color=INK)
        ax.set_xlabel("Noise severity")
        ax.set_xticks(range(6))
        _style_axis(ax, (0.0, 1.0))
    axes[0].set_ylabel("Macro-F1")

    # Callout placed in empty plot space rather than on top of the lines.
    tail = ct[(ct.task == "banfakenews") & (ct.noise_type == "whitespace_error") & (ct.severity == 5)]

    def _v(m):
        s = tail[tail.model == m]["mean"]
        return float(s.iloc[0]) if len(s) else float("nan")

    axes[2].text(
        0.18, 0.30,
        f"At severity 5\nchar $n$-gram: {_v('char_ngram'):.2f}\nXLM-R: {_v('xlmr'):.2f}",
        fontsize=7.5, color=INK, linespacing=1.5, va="top",
    )

    fig.suptitle(
        "Whitespace corruption collapses subword models but not character n-grams",
        x=0.012, y=0.97, ha="left", fontsize=10.5, color=INK,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    _model_legend(fig, y=-0.01)
    _save(fig, out)


def fig_retention(rt: pd.DataFrame, out: str) -> None:
    """Grouped bars: retention at severity 5, per noise type, per dataset."""
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    x = np.arange(len(NOISE_ORDER))
    w = 0.16
    for ax, task in zip(axes, TASK_ORDER):
        sub = rt[rt.task == task]
        for i, m in enumerate(MODEL_ORDER):
            vals = [
                float(sub[(sub.model == m) & (sub.noise_type == nt)].retention.iloc[0])
                if len(sub[(sub.model == m) & (sub.noise_type == nt)]) else np.nan
                for nt in NOISE_ORDER
            ]
            ax.bar(
                x + (i - 2) * w, vals, w * 0.88,
                color=MODEL_COLOR[m], zorder=3, linewidth=0,
            )
        ax.axhline(1.0, color=INK_MUTED, linewidth=0.8, dashes=(3, 3), zorder=2)
        ax.set_title(TASK_LABEL[task], loc="left", color=INK)
        ax.set_ylabel("Retention at sev. 5")
        _style_axis(ax, (0.0, 1.12))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([NOISE_LABEL[n] for n in NOISE_ORDER], rotation=30, ha="right")

    fig.suptitle(
        "Fraction of clean macro-F1 retained at maximum noise severity",
        x=0.012, y=0.996, ha="left", fontsize=10.5, color=INK,
    )
    fig.text(
        0.012, 0.967, "1.0 = no degradation. Mean over 3 seeds.",
        ha="left", fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.965])
    _model_legend(fig, y=-0.003, kind="patch")
    _save(fig, out)


def fig_normalizer(res: pd.DataFrame, out: str) -> None:
    """R6: BanFakeNews with the normalizer on vs off, per noise type."""
    sub = res[(res.task == "banfakenews") & (~res.is_clean)]
    agg = (
        sub.groupby(["model", "noise_type", "severity", "normalizer"])["macro_f1"]
        .mean().reset_index()
    )
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True)
    for ax, nt in zip(axes.ravel(), NOISE_ORDER):
        for m in MODEL_ORDER:
            for norm, alpha, dash in (("on", 1.0, ()), ("off", 0.55, (2, 2))):
                s = agg[(agg.model == m) & (agg.noise_type == nt) & (agg.normalizer == norm)]
                s = s.sort_values("severity")
                ax.plot(
                    s.severity, s.macro_f1,
                    color=MODEL_COLOR[m], linewidth=1.5, alpha=alpha,
                    dashes=dash, marker=MODEL_MARKER[m] if norm == "on" else None,
                    markersize=3.2, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    zorder=3,
                )
        ax.set_title(NOISE_LABEL[nt], loc="left", color=INK)
        _style_axis(ax, (0.0, 1.0))
        ax.set_xticks(range(1, 6))
    for ax in axes[-1]:
        ax.set_xlabel("Noise severity")
    for ax in axes[:, 0]:
        ax.set_ylabel("Macro-F1")

    fig.suptitle(
        "Normalizer ablation — BanFakeNews",
        x=0.012, y=0.997, ha="left", fontsize=10.5, color=INK,
    )
    fig.text(
        0.012, 0.963,
        "Solid + markers = normalizer on; dashed = off. No clean-off baseline was run, so severity starts at 1.",
        ha="left", fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    _model_legend(fig, y=-0.004)
    _save(fig, out)


def _pred_rate(js: str, cls: str = "0") -> float:
    """Recover the predicted positive rate for `cls` from per-class metrics.

    predicted_count = TP / precision, TP = recall * support.
    """
    import json

    d = json.loads(js)
    tot = sum(v["support"] for v in d.values())
    c = d[cls]
    if c["precision"] <= 0:
        return 0.0
    return (c["recall"] * c["support"]) / c["precision"] / tot


def fig_collapse(res: pd.DataFrame, out: str) -> None:
    """Prediction collapse on BanFakeNews: share of items labelled 'fake'."""
    sub = res[(res.task == "banfakenews") & (res.normalizer == "on")].copy()
    sub["pred_rate"] = sub.per_class_json.map(_pred_rate)
    base_rate = 259 / 1698

    show = ["char_delete", "char_transpose", "whitespace_error"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for ax, nt in zip(axes, show):
        for m in MODEL_ORDER:
            s = sub[(sub.model == m) & ((sub.noise_type == nt) | (sub.is_clean))]
            g = s.groupby("severity").pred_rate.mean().sort_index()
            dashes = MODEL_DASH[m]
            ax.plot(
                g.index, g.values,
                color=MODEL_COLOR[m], marker=MODEL_MARKER[m], markersize=4,
                markeredgecolor=SURFACE, markeredgewidth=0.7, linewidth=1.6,
                dashes=dashes if dashes[0] else (), zorder=3,
            )
        ax.axhline(base_rate, color=INK_MUTED, linewidth=0.9, dashes=(3, 3), zorder=2)
        ax.set_title(NOISE_LABEL[nt], loc="left", color=INK)
        ax.set_xlabel("Noise severity")
        ax.set_xticks(range(6))
        _style_axis(ax, (0.0, 1.05))
    axes[0].set_ylabel("Share predicted “fake”")
    axes[0].annotate(
        "true base rate 0.15", xy=(2.55, 0.02), fontsize=7, color=INK_MUTED,
    )

    fig.suptitle(
        "Under heavy noise the Bangla-specific models label almost everything “fake”",
        x=0.012, y=1.005, ha="left", fontsize=10.5, color=INK,
    )
    fig.text(
        0.012, 0.915,
        "BanFakeNews; mean over 3 seeds. A flat line at 1.0 is total collapse onto the minority class.",
        ha="left", fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.11, 1, 0.87])
    _model_legend(fig, y=-0.01)
    _save(fig, out)


def fig_curves(cur: pd.DataFrame, out: str) -> None:
    """Learning curves: per-epoch train/val loss and val macro-F1."""
    tr = cur[cur.train_loss.notna()]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for j, task in enumerate(TASK_ORDER):
        ax_l, ax_f = axes[0, j], axes[1, j]
        for m in MODEL_ORDER:
            s = tr[(tr.task == task) & (tr.model == m)]
            if s.empty:
                continue
            g = s.groupby("epoch")[["train_loss", "val_loss", "val_macro_f1"]].mean()
            ax_l.plot(g.index, g.train_loss, color=MODEL_COLOR[m], linewidth=1.5, zorder=3)
            ax_l.plot(g.index, g.val_loss, color=MODEL_COLOR[m], linewidth=1.5,
                      dashes=(2.5, 2), alpha=0.85, zorder=3)
            ax_f.plot(g.index, g.val_macro_f1, color=MODEL_COLOR[m],
                      marker=MODEL_MARKER[m], markersize=3.6, markeredgecolor=SURFACE,
                      markeredgewidth=0.6, linewidth=1.5, zorder=3)
        ax_l.set_title(TASK_LABEL[task], loc="left", color=INK)
        _style_axis(ax_l)
        _style_axis(ax_f)
        ax_f.set_xlabel("Epoch")
        ax_f.set_xticks(range(1, 5))
    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Validation macro-F1")

    fig.suptitle(
        "Training dynamics — mean over 3 seeds (transformers only)",
        x=0.012, y=1.02, ha="left", fontsize=10.5, color=INK,
    )
    fig.text(
        0.012, 0.955, "Top row: solid = training loss, dashed = validation loss.",
        ha="left", fontsize=7.5, color=INK_MUTED,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.925])
    _model_legend(fig, y=-0.012, models=TRANSFORMERS, ncol=4)
    _save(fig, out)


SHOW_TITLES = True


def _suptitle(fig, title, subtitle=None, y=0.99, sy=0.96):
    """In-figure titles are for review; with --no-titles the caption carries them."""
    if not SHOW_TITLES:
        return
    fig.suptitle(title, x=0.012, y=y, ha="left", fontsize=10.5, color=INK)
    if subtitle:
        fig.text(0.012, sy, subtitle, ha="left", fontsize=7.5, color=INK_MUTED)


def _save(fig, path_noext: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_noext}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(path_noext) + ".{pdf,png}")


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------

def tex_clean(base: pd.DataFrame, path: str) -> None:
    piv_m = base.pivot(index="task", columns="model", values="mean")
    piv_s = base.pivot(index="task", columns="model", values="std")
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Clean-test macro-F1 (mean $\pm$ s.d. over three seeds). "
        r"The character $n$-gram baseline is deterministic, so its s.d.\ is zero "
        r"by construction rather than by low variance.}",
        r"\label{tab:clean-baseline}",
        r"\begin{tabular}{l" + "c" * len(MODEL_ORDER) + "}", r"\toprule",
        "Dataset & " + " & ".join(MODEL_LABEL[m] for m in MODEL_ORDER) + r" \\",
        r"\midrule",
    ]
    for t in TASK_ORDER:
        best = max(MODEL_ORDER, key=lambda m: piv_m.loc[t, m])
        cells = []
        for m in MODEL_ORDER:
            v = f"{piv_m.loc[t, m]:.3f} $\\pm$ {piv_s.loc[t, m]:.3f}"
            cells.append(f"\\textbf{{{v}}}" if m == best else v)
        lines.append(TASK_LABEL[t].split(" (")[0] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("\n".join(lines), path)


def tex_retention(rt: pd.DataFrame, path: str) -> None:
    lines = [
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Macro-F1 retained at severity~5 as a fraction of clean "
        r"performance, averaged over three seeds. Lower is worse; values below "
        r"0.5 indicate near-total failure.}",
        r"\label{tab:retention}",
        r"\begin{tabular}{ll" + "c" * len(NOISE_ORDER) + "}", r"\toprule",
        "Dataset & Model & " + " & ".join(
            NOISE_LABEL[n].replace(" ", "~") for n in NOISE_ORDER
        ) + r" \\", r"\midrule",
    ]
    for t in TASK_ORDER:
        for i, m in enumerate(MODEL_ORDER):
            cells = []
            for n in NOISE_ORDER:
                r = rt[(rt.task == t) & (rt.model == m) & (rt.noise_type == n)]
                cells.append(f"{float(r.retention.iloc[0]):.2f}" if len(r) else "--")
            head = TASK_LABEL[t].split(" (")[0] if i == 0 else ""
            lines.append(f"{head} & {MODEL_LABEL[m]} & " + " & ".join(cells) + r" \\")
        lines.append(r"\midrule" if t != TASK_ORDER[-1] else "")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write("\n".join(x for x in lines if x != ""), path)


def normalizer_delta(res: pd.DataFrame) -> pd.DataFrame:
    """R6: per-model effect of the normalizer on BanFakeNews, paired by cell.

    Each (model, noise_type, severity, seed) cell was evaluated twice, so the
    on/off difference is paired -- no cross-seed averaging is needed first.
    """
    d = res[(res.task == "banfakenews") & res.noise_type.notna()]
    p = d.pivot_table(
        index=["model", "noise_type", "severity", "seed"],
        columns="normalizer", values="macro_f1",
    ).reset_index()
    p["delta"] = p["on"] - p["off"]
    out = (
        p.groupby("model")["delta"]
        .agg(mean_delta="mean", sd_delta="std",
             max_abs_delta=lambda x: x.abs().max(),
             n_cells="size")
        .reset_index()
    )
    return out


def tex_normalizer(nd: pd.DataFrame, path: str) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Effect of Unicode normalization on BanFakeNews under "
        r"perturbation (R6). $\Delta = \mathrm{F1}_{\text{on}} - "
        r"\mathrm{F1}_{\text{off}}$, computed per "
        r"(noise type, severity, seed) cell and aggregated over all "
        r"$9\times5\times3=135$ cells per model. Normalization is "
        r"effectively a no-op for every model except mBERT.}",
        r"\label{tab:normalizer}",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Model & mean $\Delta$ & s.d. & max $|\Delta|$ \\",
        r"\midrule",
    ]
    idx = nd.set_index("model")
    for m in MODEL_ORDER:
        if m not in idx.index:
            continue
        r = idx.loc[m]
        cells = f"{r.mean_delta:+.4f} & {r.sd_delta:.4f} & {r.max_abs_delta:.4f}"
        if m == "mbert":
            cells = (f"\\textbf{{{r.mean_delta:+.4f}}} & "
                     f"\\textbf{{{r.sd_delta:.4f}}} & "
                     f"\\textbf{{{r.max_abs_delta:.4f}}}")
        lines.append(f"{MODEL_LABEL[m]} & {cells} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    _write("\n".join(lines), path)


def _write(text: str, path: str) -> None:
    with open(path, "w") as fh:
        fh.write(text + "\n")
    print("wrote", os.path.basename(path))


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--no-titles", action="store_true",
                    help="omit in-figure titles (camera-ready: the LaTeX caption carries them)")
    a = ap.parse_args()
    global SHOW_TITLES
    SHOW_TITLES = not a.no_titles

    figs = os.path.join(a.out, "figures")
    tabs = os.path.join(a.out, "tables")
    os.makedirs(figs, exist_ok=True)
    os.makedirs(tabs, exist_ok=True)

    res, cur = load(a.results)
    check(res)

    base = clean_baseline(res)
    ct = curve_table(res, normalizer="on")
    rt = robustness_table(res, normalizer="on")

    base.to_csv(os.path.join(a.out, "clean_baseline.csv"), index=False)
    rt.to_csv(os.path.join(a.out, "robustness_severity5.csv"), index=False)

    for t in TASK_ORDER:
        fig_degradation(ct, t, os.path.join(figs, f"degradation_{t}"))
    fig_whitespace(ct, os.path.join(figs, "headline_whitespace"))
    fig_retention(rt, os.path.join(figs, "retention_severity5"))
    fig_collapse(res, os.path.join(figs, "prediction_collapse"))
    fig_normalizer(res, os.path.join(figs, "normalizer_ablation"))
    fig_curves(cur, os.path.join(figs, "learning_curves"))

    nd = normalizer_delta(res)
    nd.to_csv(os.path.join(a.out, "normalizer_delta.csv"), index=False)
    print("normalizer effect (BanFakeNews):")
    print(nd.round(4).to_string(index=False))

    tex_clean(base, os.path.join(tabs, "clean_baseline.tex"))
    tex_normalizer(nd, os.path.join(tabs, "normalizer.tex"))
    tex_retention(rt, os.path.join(tabs, "retention.tex"))


if __name__ == "__main__":
    main()
