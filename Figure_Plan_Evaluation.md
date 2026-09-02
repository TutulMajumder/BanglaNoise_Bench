# Figure Plan — Evaluation Section
**`bangla-noisebench` · ICREST'27 · 31 Aug 2026**

Every figure below answers a question a reviewer will actually ask. If a figure doesn't defend a claim, it isn't here.

---

## 0. THE DESIGN RULES (apply to every figure)

### Why not five coloured lines

The obvious design — five models as five coloured lines — **fails accessibility validation**. I ran the palette checker against a white paper background:

| Palette | Result |
|---|---|
| blue + orange + aqua (3 series) | PASS, but aqua is 2.82:1 contrast on white → needs direct labels as relief |
| blue + orange + **green** | **FAIL** — green↔orange ΔE 3.2 under protanopia. Invisible difference for red-green colourblind readers |
| **blue + orange only, rest gray** | **PASS with the widest margin** — CVD ΔE 24.7, normal-vision ΔE 33.6, both ≥3:1 contrast |

So the primary form is **emphasis**: two models in colour (the story), the rest in muted gray (the context). This is better science communication anyway — your claim is "transformers collapse, the character baseline holds," not "here are five equal lines."

### Fixed encoding — identical in every figure

```python
# noisebench/plotstyle.py
BLUE   = "#2a78d6"   # accent 1 — BanglaBERT (the model that collapses)
ORANGE = "#eb6834"   # accent 2 — char n-gram + SVM (the model that holds)
AQUA   = "#1baf7a"   # third slot, only where 3 series are unavoidable
GRAY   = "#898781"   # context models (muted ink)
GRID   = "#e1e0d9"   # hairline gridline
AXIS   = "#c3c2b7"   # baseline / axis
INK    = "#0b0b0b"   # primary text
INK2   = "#52514e"   # secondary text

# Secondary encoding — MANDATORY. IEEE proceedings are often printed greyscale,
# so identity must never depend on colour alone.
STYLE = {
    "banglabert":  dict(color=BLUE,   ls="-",   marker="o", zorder=5),
    "char_ngram":  dict(color=ORANGE, ls="-",   marker="s", zorder=5),
    "banglishbert":dict(color=GRAY,   ls="--",  marker="^", zorder=2),
    "xlmr":        dict(color=GRAY,   ls="-.",  marker="v", zorder=2),
    "mbert":       dict(color=GRAY,   ls=":",   marker="D", zorder=2),
}
```

**A model keeps its colour, line style and marker across every figure in the paper.** A reader who learns "solid blue with circles = BanglaBERT" in Figure 1 must not have to relearn it in Figure 4.

### Matplotlib setup

```python
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.format": "pdf",
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "axes.edgecolor": "#c3c2b7",
    "grid.color": "#e1e0d9", "grid.linewidth": 0.5, "grid.linestyle": "-",  # solid, never dashed
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 3.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL1, COL2 = 3.5, 7.16   # IEEE single- and double-column width, inches
```

Save as **PDF (vector)**, never PNG, and set the figure size in inches so LaTeX never rescales it — rescaling is what makes axis text unreadable in student papers.

### Hard rules

- **Never a dual y-axis.** Two measures of different scale → two panels, or index both to a common base.
- Gridlines and axes are **solid hairlines**, one shade off white. Never dashed grid.
- **No number on every data point.** Direct-label the endpoint of the two accent lines only; the axis carries the rest.
- **A legend is always present** for ≥2 series, placed once for a whole small-multiple grid, not per panel.
- Include **severity 0 = clean** on every degradation curve. Without it the reader can't see the drop.

---

# PART A — CORE FIGURES (in the 6-page paper)

## FIGURE 1 — Degradation curves ★ the headline

**Question it answers:** how much performance is actually lost, and does the loss differ by model?

**Form:** small multiples, 1×3 (one panel per task). x = severity 0–5, y = macro-F1. Lines = models, emphasis encoding. Shaded ±1 std band over 3 seeds.

**Why this and not a bar chart:** a bar chart of clean-vs-noisy shows two states. The curve shows **shape** — whether degradation is gradual or a cliff, and where the cliff is. Shape is the finding.

```python
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.1), sharey=True)
for ax, task in zip(axes, ["BD-SHS", "SentNoB", "BanFakeNews"]):
    d = df[df.task == task]
    for model, st in STYLE.items():
        g = d[d.model == model].groupby("severity").macro_f1
        m, s = g.mean(), g.std()
        ax.plot(m.index, m.values, **st, label=LABEL[model])
        ax.fill_between(m.index, m - s, m + s, color=st["color"],
                        alpha=0.12, lw=0, zorder=st["zorder"] - 1)
    ax.set_title(task, pad=4)
    ax.set_xlabel("Noise severity")
    ax.grid(axis="y", alpha=0.7)
    ax.set_xticks(range(6))
axes[0].set_ylabel("Macro-F1")
# direct-label the two accent lines at the right edge of the LAST panel only
axes[-1].legend(frameon=False, loc="lower left", fontsize=6)
fig.savefig("fig1_degradation.pdf")
```

**What defends the work:** if the blue line falls steeply and the orange stays flat, that single image *is* the paper's argument.

---

## FIGURE 2 — Per-noise-type small multiples ★ which noise matters

**Question:** is all noise equally harmful, or do a few types dominate?

**Form:** 3×3 grid of small multiples, one panel per graded noise type (N1–N9). Same axes as Figure 1, shared y-scale, one shared legend below the grid. Averaged over tasks, or fixed to one task with the other two in the appendix.

**Why not a heatmap alone:** the heatmap gives you one number per cell; the small multiples give you the *curve* in each cell. A noise type that is harmless until severity 4 then collapses looks identical to a mild linear one in a heatmap. Keep the heatmap as a compact summary (Supplementary S7) but let the curves carry the claim.

```python
fig, axes = plt.subplots(3, 3, figsize=(COL2, 5.0), sharex=True, sharey=True)
for ax, noise in zip(axes.ravel(), NOISE_TYPES):
    d = df[df.noise_type == noise]
    for model, st in STYLE.items():
        m = d[d.model == model].groupby("severity").macro_f1.mean()
        ax.plot(m.index, m.values, **st)
    ax.set_title(NOISE_LABEL[noise], pad=3, fontsize=7)
    ax.grid(axis="y", alpha=0.7)
fig.supxlabel("Noise severity", fontsize=8)
fig.supylabel("Macro-F1", fontsize=8)
fig.legend(handles=legend_handles, ncol=5, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.03))
```

**Sort the panels by mean degradation, worst first.** An unordered 3×3 grid makes the reader do the ranking; a sorted one hands it to them.

---

## FIGURE 3 — Clean vs robust dissociation ★ the money figure

**Question:** does clean-benchmark performance predict robustness?

**Form:** scatter, faceted by task. x = clean macro-F1, y = mean noise F1. One labelled point per model. A hairline **y = x diagonal** for reference. **No colour needed** — five labelled dots in one ink, which sidesteps the all-pairs colour cap entirely.

**Why it defends the work:** vertical distance below the diagonal = robustness loss. If the rightmost point (best clean model) is not the highest point (most robust model), the dissociation is visible in one glance — and that is the claim that makes model selection on clean benchmarks look unsafe.

```python
fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3), sharex=True, sharey=True)
for ax, task in zip(axes, TASKS):
    d = agg[agg.task == task]
    lo, hi = 0.3, 1.0
    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.6, zorder=1)   # y = x
    ax.scatter(d.clean_f1, d.mean_noise_f1, s=22, color=INK2,
               edgecolor="white", linewidth=0.8, zorder=3)
    for _, r in d.iterrows():
        ax.annotate(SHORT[r.model], (r.clean_f1, r.mean_noise_f1),
                    xytext=(3, 3), textcoords="offset points", fontsize=6)
    ax.set_title(task, pad=4); ax.grid(alpha=0.7)
axes[0].set_ylabel("Mean noise macro-F1")
fig.supxlabel("Clean macro-F1", fontsize=8)
```

Annotate the diagonal once with "no degradation" in muted ink.

---

## FIGURE 4 — Rank-inversion slopegraph ★ the most persuasive single image

**Question:** does the ranking of models change under noise?

**Form:** slopegraph (a "before → after per item" dumbbell). Left column = rank by clean F1; right column = rank by mean noise F1. One line per model, labelled at both ends. **Emphasis:** every line gray except the crossing pair.

**Why:** a table of ranks makes the reader compare two columns by eye. A slopegraph makes crossing lines literally visible — the inversion is the shape.

```python
fig, ax = plt.subplots(figsize=(COL1, 2.6))
for model in MODELS:
    r0, r1 = clean_rank[model], noise_rank[model]
    crosses = (r0 - r1) != 0
    ax.plot([0, 1], [r0, r1], lw=1.6 if crosses else 1.0,
            color=(BLUE if crosses else GRAY), alpha=1.0 if crosses else 0.5,
            marker="o", markersize=4, zorder=5 if crosses else 2)
    ax.annotate(SHORT[model], (0, r0), xytext=(-6, 0), ha="right",
                textcoords="offset points", fontsize=7)
    ax.annotate(SHORT[model], (1, r1), xytext=(6, 0), ha="left",
                textcoords="offset points", fontsize=7)
ax.invert_yaxis(); ax.set_xlim(-0.55, 1.55)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Clean rank", "Robustness rank"])
for s in ["left", "right", "top"]: ax.spines[s].set_visible(False)
ax.set_yticks([])
```

If no line crosses, say so plainly — a null result here is honest and still worth a sentence.

---

## FIGURE 5 — Mitigation: matched vs mismatched augmentation ★ the ablation that matters

**Question:** does noise-augmented training give *general* robustness, or only robustness to the noise it saw?

**Form:** 1×2 panels — left "matched noise", right "mismatched noise". Two models only (so two colours, no cap problem). **Solid = with augmentation, dashed = without.** Shade the area between the two lines to show the recovered gap.

**Why it defends the work:** "augmentation helps" is unsurprising and weak. "Augmentation helps on the noise it trained on but transfers poorly to unseen noise types" is a real finding with a practical consequence, and this figure shows both panels side by side so the contrast is unmissable.

```python
fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.2), sharey=True)
for ax, cond in zip(axes, ["matched", "mismatched"]):
    for model, col in [("banglabert", BLUE), ("char_ngram", ORANGE)]:
        base = curve(model, cond, augment=False)
        aug  = curve(model, cond, augment=True)
        ax.plot(SEV, base, color=col, ls="--", lw=1.2)
        ax.plot(SEV, aug,  color=col, ls="-",  lw=1.6, marker="o")
        ax.fill_between(SEV, base, aug, color=col, alpha=0.10, lw=0)
    ax.set_title(f"{cond.capitalize()} noise", pad=4)
    ax.set_xlabel("Noise severity"); ax.grid(axis="y", alpha=0.7)
axes[0].set_ylabel("Macro-F1")
```

---

# PART B — SUPPLEMENTARY FIGURES (appendix, arXiv, or the extended journal version)

## S1 — Benchmark validation: CER vs severity (**do this one first**)

Line per noise type, x = severity, y = mean character error rate. **This proves your benchmark is well-constructed** — the curves must rise monotonically. Add a second dashed line per noise type showing CER *after* the BanglaBERT normalizer: any noise type whose post-normalizer CER collapses toward zero is being erased by preprocessing and must be flagged.

Build this in week 1, not week 3. It is your unit test made visible, and it will catch design bugs early.

## S2 — Normalizer ON/OFF delta

Diverging encoding: y = F1(normalizer on) − F1(off), x = severity, **neutral gray zero line**. Blue above zero (normalizer helps), red below (normalizer hurts). Never put a hue at the zero midpoint.

Answers: *how much of Bangla model robustness is the model, and how much is the preprocessing?* No prior Bangla work reports this.

## S3 — Calibration under noise

Two panels: (a) ECE vs severity, one line per model; (b) reliability diagrams at severity 0 and 5, overlaid, with the diagonal. Shows whether confidence degrades faster than correctness — which is what breaks a deployed system that acts on a threshold.

## S4 — Per-class F1 vs severity

Emphasis form: minority class in accent colour, majority classes in gray. On BanFakeNews (48:1) this likely shows the minority class collapsing well before the macro average does. That is the deployment-relevant failure.

## S5 — Synthetic vs natural validation

Slopegraph again: robustness rank under your synthetic suite (left) vs rank on naturally noisy SentNoB (right), with **Spearman ρ annotated in the corner**. Near-parallel lines validate the benchmark; crossing lines are a finding about synthetic-noise studies generally.

## S6 — Robustness–efficiency Pareto

Scatter: x = inference latency (ms/sample), y = mean noise F1, labelled points, Pareto frontier as a step line. Supports the practical recommendation — "if you need robustness on cheap hardware, use this."

## S7 — Heatmap (noise type × model)

Sequential **single-hue** blue ramp (light → dark), never a rainbow. A compact companion to Figure 2, useful if you need to save space. Annotate cells with values since the ramp alone shouldn't carry the reading.

---

# PART C — PRIORITY AND SPACE BUDGET

| Priority | Figure | If you only build... |
|---|---|---|
| 1 | S1 (CER validation) | Week 1 — it's your unit test, visualised |
| 2 | F1 (degradation curves) | ...three figures, build 1, 3, 4 |
| 3 | F3 (clean vs robust scatter) | |
| 4 | F4 (rank-inversion slopegraph) | |
| 5 | F5 (matched/mismatched) | ...five, add 5 and 2 |
| 6 | F2 (per-noise small multiples) | |
| 7+ | S2–S7 | Appendix / extended version |

A 6-page IEEE paper realistically fits **4–5 figures plus 3–4 tables**. Take F1, F3, F4, F5 into the paper; F2 goes in if you can trim a table; everything else is supplementary.

---

# PART D — PRE-SUBMISSION CHECKLIST

- [ ] Every figure saved as **vector PDF**, sized in inches, never rescaled in LaTeX
- [ ] Print one page in **greyscale** and confirm every series is still identifiable by line style and marker
- [ ] No dual y-axis anywhere
- [ ] Gridlines solid hairlines, not dashed
- [ ] Severity 0 (clean) included on every degradation curve
- [ ] Each model has the same colour + line style + marker in every figure
- [ ] Legend present once per figure; direct labels only on accent series and endpoints
- [ ] ±1 std bands shown wherever 3 seeds exist (never a bare mean line)
- [ ] Axis labels include units; y-axis says "Macro-F1", not "F1" or "Score"
- [ ] Captions state the finding, not the mechanics — "BanglaBERT loses X while the character baseline holds", not "Macro-F1 versus severity"
- [ ] Every number in a figure also reachable from a table (accessibility)
- [ ] Re-run the palette validator if you change any colour:
      `node scripts/validate_palette.js "#2a78d6,#eb6834" --mode light --surface "#ffffff" --pairs all`
