"""Shared plotting constants -- the validated palette and IEEE sizing from
``Figure_Plan_Evaluation.md``. Do not invent new colours here; if a figure
needs a colour not listed, that is a sign it needs a redesign, not a new hex
code (see the palette-checker rationale in the figure plan).

Pure data (strings, tuples, dicts) -- no third-party imports, so this stays
inside the CLAUDE.md "standard library only" constraint for ``noisebench/``.
Matplotlib itself is a `scripts/`-only dependency (``make_figures.py``); this
module is imported from there, never the reverse.
"""

from __future__ import annotations

# Fixed encoding -- identical in every figure across the whole paper
# (Figure_Plan_Evaluation.md Part 0). BLUE/ORANGE are the emphasis pair for
# model comparisons in Phase 5; Phase 2 has no models yet, so its figures
# reuse the same four validated colours as dataset identity instead of
# inventing new ones (see DATASET_STYLE below).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRAY = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"

# IEEE single- and double-column widths, inches.
COL1 = 3.5
COL2 = 7.16

# Base rcParams, applied once by scripts/make_figures.py before any figure is
# built (Figure_Plan_Evaluation.md Part 0 "Matplotlib setup").
MPL_RCPARAMS: dict = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": AXIS,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",  # solid, never dashed
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}

# Phase 2 has no models to emphasise -- every dataset is context, so each gets
# a distinct colour + line style + marker from the SAME validated set, applied
# consistently across every Phase 2 figure (Figure_Plan_Evaluation.md Part D:
# "each model/series has the same colour + line style + marker in every
# figure"). Order matches the three datasets in pipeline/s2_prepare.py
# (banfakenews_full cut 2026-09-15 -- see PHASE2_STATUS.md).
DATASET_STYLE: dict[str, dict] = {
    "sentnob": dict(color=BLUE, ls="-", marker="o"),
    "bd_shs": dict(color=ORANGE, ls="-", marker="s"),
    "banfakenews": dict(color=AQUA, ls="--", marker="^"),
}
