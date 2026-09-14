"""Phase 3 figures -- reads results.csv + curves.csv (and Phase 1's CER
table) ONLY. Never trains, never runs inference, CPU-only, fast. Palette,
IEEE column widths and rcParams come from noisebench/plotstyle.py, which
mirrors docs/Figure_Plan_Evaluation.md Part 0 exactly -- do not invent a new
colour or size here.

Eight figure types (Phase 3 spec item 5):
  1. fig_degradation        -- accuracy+macro-F1 vs severity (Figure_Plan F1)
  2. fig_training_curves    -- epoch vs loss/accuracy, train+val, mean+-1SD
                                over seeds, with the diagnostic test curve
                                dashed and TEST_DIAGNOSTIC_CAPTION printed
  3. fig_relative_degradation -- each model normalized to its own clean F1
  4. fig_heatmap            -- noise-type x model accuracy drop @ severity 3
                                (Figure_Plan S7)
  5. fig_rank_inversion     -- clean rank vs noise rank slopegraph (F4)
  6. fig_cer_vs_drop        -- mean CER (Phase 1) vs mean accuracy drop
  7. fig_r4b_augmentation   -- matched vs mismatched augmentation (F5)
  8. fig_model_table        -- rendered from noisebench.models registry

Every figure saves THREE files: <name>.png (300dpi), <name>.pdf (vector,
what actually goes in the paper per Figure_Plan Part D), <name>.csv (the
exact aggregated data plotted, for the accessibility checklist item "every
number in a figure also reachable from a table").

A figure whose required CSV rows don't exist yet (no training has run) is
SKIPPED with a stderr message, not a crash -- this file must run correctly
against a results.csv that has 3 rows or 300,000.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench.models import MODEL_ORDER, MODELS  # noqa: E402
from noisebench.perturbations import NOISE_TYPES  # noqa: E402
from noisebench.plotstyle import (  # noqa: E402
    AQUA, AXIS, BLUE, COL1, COL2, GRAY, GRID, INK, INK2, MODEL_SHORT,
    MPL_RCPARAMS, NOISE_LABEL, ORANGE, STYLE, TASK_LABEL,
)
from pipeline._common import Progress, add_common_args, print_smoke_banner, smoke_or_prod  # noqa: E402
from pipeline.p3_train import R4_MODELS, TASK_ORDER, TEST_DIAGNOSTIC_CAPTION  # noqa: E402

_DEFAULT_RESULTS_DIR = os.path.join(_ROOT, "results")
_DEFAULT_OUT_DIR = os.path.join(_ROOT, "outputs", "p3_figures")
_DEFAULT_CER_TABLE = os.path.join(_ROOT, "outputs", "s2_preparation", "tables", "m2_4_cer.csv")


def _apply_style() -> None:
    import matplotlib as mpl
    mpl.rcParams.update(MPL_RCPARAMS)


def load_results(results_path: str) -> pd.DataFrame:
    if not os.path.exists(results_path):
        return pd.DataFrame(columns=[
            "run_id", "model", "task", "seed", "train_variant", "normalizer",
            "noise_type", "severity", "n_test_items", "macro_f1", "weighted_f1",
            "balanced_accuracy", "accuracy", "pr_auc", "mcc", "per_class_json",
            "best_epoch", "best_val_macro_f1", "timestamp"])
    df = pd.read_csv(results_path)
    df["noise_type"] = df["noise_type"].fillna("")
    return df


def load_curves(curves_path: str) -> pd.DataFrame:
    if not os.path.exists(curves_path):
        return pd.DataFrame(columns=[
            "run_id", "model", "task", "seed", "train_variant", "normalizer",
            "epoch", "train_loss", "train_accuracy", "train_macro_f1",
            "val_loss", "val_accuracy", "val_macro_f1",
            "test_accuracy_diagnostic", "test_macro_f1_diagnostic"])
    return pd.read_csv(curves_path)


def _save_figure(fig, out_dir: str, name: str, source_df: pd.DataFrame, progress: Progress) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{name}.png")
    pdf_path = os.path.join(out_dir, f"{name}.pdf")
    csv_path = os.path.join(out_dir, f"{name}.csv")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        source_df.to_csv(fh, index=False)
    import matplotlib.pyplot as plt
    plt.close(fig)
    progress.log(name, f"saved {png_path}, {pdf_path}, {csv_path}")


def _skip(progress: Progress, name: str, reason: str) -> None:
    progress.log(name, f"SKIPPED -- {reason}")


# ---------------------------------------------------------------------------
# 1. Degradation curves -- Figure_Plan F1. x=severity (0=clean..5), y=metric,
#    one small-multiple panel per task, models emphasis-encoded, averaged
#    over the 9 noise types (and all 3 seeds) at each severity.
# ---------------------------------------------------------------------------


def fig_degradation(results: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    name = "fig1_degradation"
    d = results[(results.train_variant == "clean") & (results.normalizer == "on")]
    if d.empty:
        return _skip(progress, name, "no clean-trained, normalizer=on rows in results.csv")

    import matplotlib.pyplot as plt

    rows = []
    for metric in ("accuracy", "macro_f1"):
        for task in TASK_ORDER:
            for model in MODEL_ORDER:
                sub = d[(d.task == task) & (d.model == model)]
                if sub.empty:
                    continue
                g = sub.groupby("severity")[metric]
                for sev, mean_v in g.mean().items():
                    rows.append({"metric": metric, "task": task, "model": model,
                                "severity": sev, "mean": mean_v, "std": g.std().get(sev, 0.0),
                                "n_seeds": int(g.count().get(sev, 0))})
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no rows survived aggregation")

    fig, axes = plt.subplots(2, 3, figsize=(COL2, 4.2), sharex=True)
    for row_i, metric in enumerate(("accuracy", "macro_f1")):
        for col_i, task in enumerate(TASK_ORDER):
            ax = axes[row_i, col_i]
            sub = src[(src.metric == metric) & (src.task == task)]
            for model in MODEL_ORDER:
                ms = sub[sub.model == model].sort_values("severity")
                if ms.empty:
                    continue
                st = STYLE[model]
                ax.plot(ms.severity, ms["mean"], label=MODEL_SHORT[model], **st)
                ax.fill_between(ms.severity, ms["mean"] - ms["std"], ms["mean"] + ms["std"],
                                color=st["color"], alpha=0.12, lw=0, zorder=st["zorder"] - 1)
            if row_i == 0:
                ax.set_title(TASK_LABEL[task], pad=4)
            if row_i == 1:
                ax.set_xlabel("Noise severity")
            ax.set_xticks(range(6))
            ax.grid(axis="y", alpha=0.7)
        axes[row_i, 0].set_ylabel("Accuracy" if metric == "accuracy" else "Macro-F1")
    axes[0, -1].legend(frameon=False, loc="lower left", fontsize=6)
    fig.suptitle("Degradation under synthetic orthographic/typographic noise "
                 "(mean +/-1 SD over 3 seeds; clean = severity 0)", fontsize=8)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 2. Training curves -- from curves.csv. Solid = val (drives model
#    selection), dashed = the diagnostic test curve (never used to select),
#    caption printed alongside so it can't be separated from the figure.
# ---------------------------------------------------------------------------


def fig_training_curves(curves: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    name = "fig_training_curves"
    d = curves[curves.train_variant == "clean"]
    if d.empty:
        return _skip(progress, name, "no clean-training rows in curves.csv")

    import matplotlib.pyplot as plt

    rows = []
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            sub = d[(d.task == task) & (d.model == model)]
            if sub.empty:
                continue
            for epoch, g in sub.groupby("epoch"):
                rows.append({
                    "task": task, "model": model, "epoch": epoch,
                    "train_loss_mean": g.train_loss.mean(), "val_loss_mean": g.val_loss.mean(),
                    "train_acc_mean": g.train_accuracy.mean(), "val_acc_mean": g.val_accuracy.mean(),
                    "val_macro_f1_mean": g.val_macro_f1.mean(), "val_macro_f1_std": g.val_macro_f1.std(),
                    "test_acc_diag_mean": g.test_accuracy_diagnostic.mean(),
                    "test_macro_f1_diag_mean": g.test_macro_f1_diagnostic.mean(),
                    "n_seeds": int(g.epoch.count()),
                })
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no rows survived aggregation")

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(COL2, 2.4), sharex=False)
    for ax, task in zip(np.atleast_1d(axes), TASK_ORDER):
        sub = src[src.task == task]
        for model in R4_MODELS:  # only the two accent models to keep this readable
            ms = sub[sub.model == model].sort_values("epoch")
            if ms.empty:
                continue
            st = STYLE[model]
            ax.plot(ms.epoch, ms.val_macro_f1_mean, color=st["color"], ls="-",
                    marker=st["marker"], label=f"{MODEL_SHORT[model]} val")
            if ms.test_macro_f1_diag_mean.notna().any():
                ax.plot(ms.epoch, ms.test_macro_f1_diag_mean, color=st["color"], ls="--",
                        alpha=0.6, label=f"{MODEL_SHORT[model]} test (diagnostic)")
        ax.set_title(TASK_LABEL[task], pad=4)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("Macro-F1") if hasattr(axes, "__len__") else None
    handles, labels = axes[-1].get_legend_handles_labels() if hasattr(axes, "__len__") else axes.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.08), fontsize=6)
    fig.text(0.5, -0.16, TEST_DIAGNOSTIC_CAPTION, ha="center", fontsize=6, color=INK2, wrap=True)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 3. Relative degradation -- each model's own clean performance = 1.0.
# ---------------------------------------------------------------------------


def fig_relative_degradation(results: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    name = "fig_relative_degradation"
    d = results[(results.train_variant == "clean") & (results.normalizer == "on")]
    if d.empty:
        return _skip(progress, name, "no clean-trained, normalizer=on rows in results.csv")

    import matplotlib.pyplot as plt

    rows = []
    for task in TASK_ORDER:
        for model in MODEL_ORDER:
            sub = d[(d.task == task) & (d.model == model)]
            if sub.empty:
                continue
            clean = sub[sub.severity == 0].macro_f1.mean()
            if not clean or np.isnan(clean) or clean == 0:
                continue
            g = sub.groupby("severity").macro_f1.mean()
            for sev, v in g.items():
                rows.append({"task": task, "model": model, "severity": sev,
                            "relative_macro_f1": v / clean, "clean_macro_f1": clean})
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no model has a nonzero clean baseline")

    fig, axes = plt.subplots(1, len(TASK_ORDER), figsize=(COL2, 2.2), sharey=True)
    for ax, task in zip(np.atleast_1d(axes), TASK_ORDER):
        sub = src[src.task == task]
        for model in MODEL_ORDER:
            ms = sub[sub.model == model].sort_values("severity")
            if ms.empty:
                continue
            ax.plot(ms.severity, ms.relative_macro_f1, **STYLE[model], label=MODEL_SHORT[model])
        ax.axhline(1.0, color=AXIS, lw=0.6, zorder=1)
        ax.set_title(TASK_LABEL[task], pad=4)
        ax.set_xlabel("Noise severity")
        ax.set_xticks(range(6))
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("Macro-F1 / own clean Macro-F1")
    axes[-1].legend(frameon=False, loc="lower left", fontsize=6)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 4. Heatmap -- noise type x model, accuracy drop at severity 3 (Figure_Plan S7)
# ---------------------------------------------------------------------------


def fig_heatmap(results: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    name = "fig_heatmap_sev3"
    d = results[(results.train_variant == "clean") & (results.normalizer == "on")]
    if d.empty:
        return _skip(progress, name, "no clean-trained, normalizer=on rows in results.csv")

    import matplotlib.pyplot as plt

    rows = []
    for model in MODEL_ORDER:
        clean = d[(d.model == model) & (d.severity == 0)]
        clean_acc = {task: clean[clean.task == task].accuracy.mean() for task in TASK_ORDER}
        for nt in NOISE_TYPES:
            sev3 = d[(d.model == model) & (d.noise_type == nt) & (d.severity == 3)]
            if sev3.empty:
                continue
            drops = [clean_acc[task] - sev3[sev3.task == task].accuracy.mean()
                    for task in TASK_ORDER if not np.isnan(clean_acc.get(task, np.nan))
                    and not sev3[sev3.task == task].empty]
            if drops:
                rows.append({"model": model, "noise_type": nt, "mean_accuracy_drop": float(np.mean(drops))})
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no severity-3 rows to compare against a clean baseline")

    pivot = src.pivot(index="noise_type", columns="model", values="mean_accuracy_drop")
    pivot = pivot.reindex(index=[nt for nt in NOISE_TYPES if nt in pivot.index],
                          columns=[m for m in MODEL_ORDER if m in pivot.columns])

    fig, ax = plt.subplots(figsize=(COL1, 3.2))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([MODEL_SHORT[m] for m in pivot.columns], rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([NOISE_LABEL[nt] for nt in pivot.index])
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                       color="white" if v > np.nanmax(pivot.values) * 0.6 else INK)
    fig.colorbar(im, ax=ax, label="Mean accuracy drop @ severity 3", shrink=0.8)
    ax.set_title("Accuracy drop by noise type x model (severity 3, avg over tasks)", fontsize=7)
    _save_figure(fig, out_dir, name, pivot.reset_index(), progress)


# ---------------------------------------------------------------------------
# 5. Rank-inversion slopegraph (Figure_Plan F4) -- one figure per task.
# ---------------------------------------------------------------------------


def fig_rank_inversion(results: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    d = results[(results.train_variant == "clean") & (results.normalizer == "on")]
    if d.empty:
        return _skip(progress, "fig_rank_inversion", "no clean-trained, normalizer=on rows in results.csv")

    import matplotlib.pyplot as plt

    for task in TASK_ORDER:
        name = f"fig4_rank_inversion_{task}"
        sub = d[d.task == task]
        clean_f1 = sub[sub.severity == 0].groupby("model").macro_f1.mean()
        noisy_f1 = sub[sub.severity > 0].groupby("model").macro_f1.mean()
        models = [m for m in MODEL_ORDER if m in clean_f1.index and m in noisy_f1.index]
        if len(models) < 2:
            _skip(progress, name, f"fewer than 2 models with both clean and noisy rows for {task}")
            continue
        clean_rank = clean_f1[models].rank(ascending=False, method="min")
        noise_rank = noisy_f1[models].rank(ascending=False, method="min")
        src = pd.DataFrame({"model": models,
                            "clean_macro_f1": clean_f1[models].values,
                            "mean_noise_macro_f1": noisy_f1[models].values,
                            "clean_rank": clean_rank.values, "noise_rank": noise_rank.values})

        fig, ax = plt.subplots(figsize=(COL1, 2.6))
        any_crosses = False
        for model in models:
            r0, r1 = clean_rank[model], noise_rank[model]
            crosses = r0 != r1
            any_crosses = any_crosses or crosses
            ax.plot([0, 1], [r0, r1], lw=1.6 if crosses else 1.0,
                   color=(BLUE if crosses else GRAY), alpha=1.0 if crosses else 0.5,
                   marker="o", markersize=4, zorder=5 if crosses else 2)
            ax.annotate(MODEL_SHORT[model], (0, r0), xytext=(-6, 0), ha="right",
                       textcoords="offset points", fontsize=7)
            ax.annotate(MODEL_SHORT[model], (1, r1), xytext=(6, 0), ha="left",
                       textcoords="offset points", fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(-0.55, 1.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clean rank", "Robustness rank"])
        for s in ("left", "right", "top"):
            ax.spines[s].set_visible(False)
        ax.set_yticks([])
        ax.set_title(TASK_LABEL[task], pad=4)
        if not any_crosses:
            fig.text(0.5, -0.05, "No rank inversion observed -- ranking is stable under noise.",
                     ha="center", fontsize=6, color=INK2)
        _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 6. CER (Phase 1, per noise type/severity) vs mean accuracy drop (Phase 3,
#    averaged over models/tasks/seeds at that noise type/severity).
# ---------------------------------------------------------------------------


def fig_cer_vs_drop(results: pd.DataFrame, cer_table_path: str, out_dir: str, progress: Progress) -> None:
    name = "fig_cer_vs_drop"
    if not os.path.exists(cer_table_path):
        return _skip(progress, name, f"CER table not found at {cer_table_path} -- run scripts/phase1_report.py")
    d = results[(results.train_variant == "clean") & (results.normalizer == "on")]
    if d.empty:
        return _skip(progress, name, "no clean-trained, normalizer=on rows in results.csv")

    cer = pd.read_csv(cer_table_path)
    cer = cer[cer.group == "synthetic"] if "group" in cer.columns else cer

    rows = []
    for _, cer_row in cer.iterrows():
        nt = cer_row["noise_type"]
        if nt not in NOISE_TYPES:
            continue
        clean_acc = d[d.severity == 0].accuracy.mean()
        for sev in range(1, 6):
            col = f"s{sev}"
            if col not in cer_row or pd.isna(cer_row[col]):
                continue
            sev_acc = d[(d.noise_type == nt) & (d.severity == sev)].accuracy.mean()
            if np.isnan(sev_acc) or np.isnan(clean_acc):
                continue
            rows.append({"noise_type": nt, "severity": sev, "mean_cer": cer_row[col],
                        "mean_accuracy_drop": clean_acc - sev_acc})
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no (noise_type, severity) pair had both a CER value and a matching results.csv drop")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(COL1, 2.6))
    ax.scatter(src.mean_cer, src.mean_accuracy_drop, s=18, color=INK2, edgecolor="white", linewidth=0.6, zorder=3)
    for _, r in src.iterrows():
        if r.severity in (3, 5):
            ax.annotate(f"{NOISE_LABEL[r.noise_type].split(' ', 1)[1]} s{r.severity}",
                       (r.mean_cer, r.mean_accuracy_drop), xytext=(3, 3),
                       textcoords="offset points", fontsize=5)
    ax.set_xlabel("Mean character error rate (Phase 1)")
    ax.set_ylabel("Mean accuracy drop (all models/tasks)")
    ax.grid(alpha=0.7)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 7. R4b matched vs mismatched augmentation (Figure_Plan F5).
#
# Design decision (PHASE3_STATUS.md): "matched" pools the two cells where the
# augmentation group equals the evaluation noise group (A-trained/A-eval,
# B-trained/B-eval); "mismatched" pools the two cross cells
# (A-trained/B-eval, B-trained/A-eval). The unaugmented baseline (dashed) is
# the SAME clean-trained run evaluated on the matched/mismatched noise types
# for that panel -- there is only one "no augmentation" condition, M5.2 does
# not define a separate one per group.
# ---------------------------------------------------------------------------

_GROUP_A = NOISE_TYPES[0:4]
_GROUP_B = NOISE_TYPES[4:8]
_VARIANT_GROUP = {"augmented_n1n4": _GROUP_A, "augmented_n5n8": _GROUP_B}


def fig_r4b_augmentation(results: pd.DataFrame, out_dir: str, progress: Progress) -> None:
    name = "fig5_r4b_augmentation"
    d = results[results.normalizer == "on"]
    aug = d[d.train_variant.isin(_VARIANT_GROUP)]
    clean = d[d.train_variant == "clean"]
    if aug.empty or clean.empty:
        return _skip(progress, name, "no augmented (R4) training rows in results.csv yet")

    rows = []
    for model in R4_MODELS:
        for cond, pairs in (("matched", [("augmented_n1n4", _GROUP_A), ("augmented_n5n8", _GROUP_B)]),
                            ("mismatched", [("augmented_n1n4", _GROUP_B), ("augmented_n5n8", _GROUP_A)])):
            for sev in range(1, 6):
                aug_vals, base_vals = [], []
                for variant, group in pairs:
                    a = aug[(aug.model == model) & (aug.train_variant == variant) &
                           (aug.noise_type.isin(group)) & (aug.severity == sev)]
                    b = clean[(clean.model == model) & (clean.noise_type.isin(group)) & (clean.severity == sev)]
                    if not a.empty:
                        aug_vals.append(a.macro_f1.mean())
                    if not b.empty:
                        base_vals.append(b.macro_f1.mean())
                if aug_vals and base_vals:
                    rows.append({"model": model, "condition": cond, "severity": sev,
                                "augmented_macro_f1": float(np.mean(aug_vals)),
                                "baseline_macro_f1": float(np.mean(base_vals))})
    src = pd.DataFrame(rows)
    if src.empty:
        return _skip(progress, name, "no matched/mismatched cells had both an augmented and a clean-baseline row")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True)
    for ax, cond in zip(axes, ("matched", "mismatched")):
        for model in R4_MODELS:
            ms = src[(src.model == model) & (src.condition == cond)].sort_values("severity")
            if ms.empty:
                continue
            col = STYLE[model]["color"]
            ax.plot(ms.severity, ms.baseline_macro_f1, color=col, ls="--", lw=1.2)
            ax.plot(ms.severity, ms.augmented_macro_f1, color=col, ls="-", lw=1.6, marker="o",
                   label=MODEL_SHORT[model])
            ax.fill_between(ms.severity, ms.baseline_macro_f1, ms.augmented_macro_f1,
                            color=col, alpha=0.10, lw=0)
        ax.set_title(f"{cond.capitalize()} noise", pad=4)
        ax.set_xlabel("Noise severity")
        ax.grid(axis="y", alpha=0.7)
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# 8. Model comparison table -- from the registry, not results.csv.
# ---------------------------------------------------------------------------


def fig_model_table(out_dir: str, progress: Progress) -> None:
    name = "table_model_comparison"
    cols = ["display_name", "kind", "tokenizer_type", "vocab_size", "param_count",
           "hidden_size", "n_layers", "pretraining_corpus", "why_included"]
    rows = [{c: getattr(MODELS[k], c) for c in cols} for k in MODEL_ORDER]
    src = pd.DataFrame(rows)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(COL2, 0.35 * (len(rows) + 1) + 0.3))
    ax.axis("off")
    display_cols = ["display_name", "kind", "tokenizer_type", "vocab_size", "param_count", "pretraining_corpus"]
    cell_text = [[str(r[c]) for c in display_cols] for r in rows]
    tbl = ax.table(cellText=cell_text, colLabels=display_cols, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6)
    tbl.scale(1, 1.4)
    _save_figure(fig, out_dir, name, src, progress)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_all(results_dir: str, out_dir: str, cer_table_path: str, progress: Progress) -> None:
    _apply_style()
    results = load_results(os.path.join(results_dir, "results.csv"))
    curves = load_curves(os.path.join(results_dir, "curves.csv"))
    progress.log("p3_figures", f"loaded {len(results)} results.csv rows, {len(curves)} curves.csv rows")

    fig_degradation(results, out_dir, progress)
    fig_training_curves(curves, out_dir, progress)
    fig_relative_degradation(results, out_dir, progress)
    fig_heatmap(results, out_dir, progress)
    fig_rank_inversion(results, out_dir, progress)
    fig_cer_vs_drop(results, cer_table_path, out_dir, progress)
    fig_r4b_augmentation(results, out_dir, progress)
    fig_model_table(out_dir, progress)


def _write_smoke_fixture(results_dir: str) -> None:
    """Synthetic-but-plausible results.csv/curves.csv so every figure
    function can be exercised without a real (multi-day) training run --
    values are NOT experiment results, never used for anything but proving
    the plotting code runs. Deterministic (seed=0) so the smoke test is
    reproducible."""
    rng = np.random.default_rng(0)
    os.makedirs(results_dir, exist_ok=True)
    results_rows, curves_rows = [], []
    for model in MODEL_ORDER:
        base = 0.55 if model == "char_ngram" else 0.85
        decay = 0.015 if model == "char_ngram" else 0.09
        for task in TASK_ORDER:
            for seed in (42, 1337, 2024):
                run_id = f"{model}__{task}__{seed}__clean"
                for epoch in range(1, 4):
                    curves_rows.append({
                        "run_id": run_id, "model": model, "task": task, "seed": seed,
                        "train_variant": "clean", "normalizer": "on", "epoch": epoch,
                        "train_loss": 1.0 / epoch, "train_accuracy": min(0.99, 0.5 + 0.1 * epoch),
                        "train_macro_f1": min(0.99, 0.5 + 0.1 * epoch),
                        "val_loss": 1.1 / epoch, "val_accuracy": min(0.95, 0.45 + 0.1 * epoch),
                        "val_macro_f1": min(0.95, 0.45 + 0.1 * epoch),
                        "test_accuracy_diagnostic": min(0.95, 0.44 + 0.1 * epoch),
                        "test_macro_f1_diagnostic": min(0.95, 0.44 + 0.1 * epoch),
                    })

                def _metric_row(noise_type, severity):
                    f1 = max(0.05, base - decay * severity + rng.normal(0, 0.01))
                    acc = max(0.05, f1 + 0.03)
                    return {
                        "run_id": run_id, "model": model, "task": task, "seed": seed,
                        "train_variant": "clean", "normalizer": "on",
                        "noise_type": noise_type, "severity": severity, "n_test_items": 500,
                        "macro_f1": f1, "weighted_f1": f1, "balanced_accuracy": acc, "accuracy": acc,
                        "pr_auc": "", "mcc": "", "per_class_json": "{}",
                        "best_epoch": 2, "best_val_macro_f1": f1, "timestamp": "2026-01-01T00:00:00Z",
                    }

                results_rows.append(_metric_row("", 0))
                for nt in NOISE_TYPES:
                    for sev in range(1, 6):
                        results_rows.append(_metric_row(nt, sev))

    # A slice of synthetic R4 (augmented) rows too, so fig_r4b_augmentation
    # has something to plot in the smoke test -- augmentation recovers half
    # of the clean-vs-noisy gap on the group it trained on (matched), a
    # smaller fraction on the other group (mismatched); this is fixture
    # shape only, not a claim about real results.
    for model in R4_MODELS:
        base = 0.55 if model == "char_ngram" else 0.85
        decay = 0.015 if model == "char_ngram" else 0.09
        for task in TASK_ORDER:
            for seed in (42, 1337, 2024):
                for variant, matched_group in _VARIANT_GROUP.items():
                    run_id = f"{model}__{task}__{seed}__{variant}"
                    for nt in NOISE_TYPES:
                        for sev in range(1, 6):
                            clean_f1 = max(0.05, base - decay * sev)
                            recovery = 0.5 if nt in matched_group else 0.15
                            f1 = min(0.99, clean_f1 + recovery * decay * sev + rng.normal(0, 0.01))
                            results_rows.append({
                                "run_id": run_id, "model": model, "task": task, "seed": seed,
                                "train_variant": variant, "normalizer": "on",
                                "noise_type": nt, "severity": sev, "n_test_items": 500,
                                "macro_f1": f1, "weighted_f1": f1, "balanced_accuracy": f1 + 0.03,
                                "accuracy": f1 + 0.03, "pr_auc": "", "mcc": "", "per_class_json": "{}",
                                "best_epoch": 2, "best_val_macro_f1": f1, "timestamp": "2026-01-01T00:00:00Z",
                            })
    pd.DataFrame(results_rows).to_csv(os.path.join(results_dir, "results.csv"), index=False)
    pd.DataFrame(curves_rows).to_csv(os.path.join(results_dir, "curves.csv"), index=False)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--cer-table", default=_DEFAULT_CER_TABLE)
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.results_dir is None:
        args.results_dir = smoke_or_prod(args.limit, _DEFAULT_RESULTS_DIR)
    if args.out_dir is None:
        args.out_dir = smoke_or_prod(args.limit, _DEFAULT_OUT_DIR)

    progress = Progress("p3_figures")
    if args.limit is not None:
        print_smoke_banner("p3_figures", args.limit, {"results-dir": args.results_dir, "out-dir": args.out_dir})
        _write_smoke_fixture(args.results_dir)
        progress.log("p3_figures", "wrote a synthetic smoke fixture results.csv/curves.csv "
                                   "(NOT real results) to exercise every figure function")

    build_all(args.results_dir, args.out_dir, args.cer_table, progress)
    progress.done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
