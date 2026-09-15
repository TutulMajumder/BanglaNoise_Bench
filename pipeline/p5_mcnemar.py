#!/usr/bin/env python3
"""
p5_mcnemar.py -- M6.2 paired significance tests, BanglaBERT vs char n-gram.

Six pre-registered tests: {SentNoB, BD-SHS, BanFakeNews} x {severity 3, 5}.
Each test pools per-item predictions over all nine perturbation types, as
pre-registered, and is run separately for every seed. Holm-Bonferroni controls
the family-wise error rate over the six pre-registered tests.

Usage:
    python pipeline/p5_mcnemar.py --preds results/preds --out analysis
"""
from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

MODEL_A = "banglabert"
MODEL_B = "char_ngram"
LABEL = {"banglabert": "BanglaBERT", "char_ngram": "char n-gram + SVM"}

TASKS = ["sentnob", "bd_shs", "banfakenews"]
TASK_LABEL = {
    "sentnob": "SentNoB",
    "bd_shs": "BD-SHS",
    "banfakenews": "BanFakeNews",
}
SEVERITIES = [3, 5]
SEEDS = [42, 1337, 2024]


# --------------------------------------------------------------------------

def mcnemar(b: int, c: int) -> tuple[float, str]:
    """Paired test on the discordant cells.

    b = A right & B wrong, c = A wrong & B right. Exact binomial when the
    discordant count is small (the chi-square approximation is unreliable
    below ~25); otherwise chi-square with continuity correction.
    """
    n = b + c
    if n == 0:
        return 1.0, "exact"
    if n < 25:
        p = float(stats.binomtest(min(b, c), n, 0.5).pvalue)
        return p, "exact"
    chi2 = (abs(b - c) - 1) ** 2 / n
    return float(stats.chi2.sf(chi2, 1)), "chi2-cc"


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def load_pair(preds_dir: str, task: str, seed: int, sev: int):
    """Return the aligned (true, pred_A, pred_B, noise_type) frame, or None."""
    def path(model):
        return os.path.join(
            preds_dir, f"{model}__{task}__{seed}__clean__noise_s{sev}.csv"
        )

    pa, pb = path(MODEL_A), path(MODEL_B)
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    a = pd.read_csv(pa)
    b = pd.read_csv(pb)

    key = ["id", "noise_type"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    # Guard against any silent misalignment between the two prediction files.
    assert len(m) == len(a) == len(b), (
        f"row mismatch for {task}/{seed}/s{sev}: {len(a)} vs {len(b)} -> {len(m)}"
    )
    assert (m.true_a == m.true_b).all(), "gold labels disagree between files"
    return m


def cell_counts(m: pd.DataFrame) -> tuple[int, int, int, int]:
    ok_a = (m.pred_a == m.true_a).to_numpy()
    ok_b = (m.pred_b == m.true_b).to_numpy()
    return (
        int((ok_a & ok_b).sum()),      # a: both right
        int((ok_a & ~ok_b).sum()),     # b: only A right
        int((~ok_a & ok_b).sum()),     # c: only B right
        int((~ok_a & ~ok_b).sum()),    # d: both wrong
    )


# --------------------------------------------------------------------------

def run(preds_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_seed, missing = [], []

    for task, sev, seed in itertools.product(TASKS, SEVERITIES, SEEDS):
        m = load_pair(preds_dir, task, seed, sev)
        if m is None:
            missing.append((task, sev, seed))
            continue
        both, only_a, only_b, neither = cell_counts(m)
        p, method = mcnemar(only_a, only_b)
        n = len(m)
        per_seed.append(
            {
                "task": task, "severity": sev, "seed": seed, "n_items": n,
                "both_correct": both, "only_banglabert": only_a,
                "only_char_ngram": only_b, "neither": neither,
                "acc_banglabert": (both + only_a) / n,
                "acc_char_ngram": (both + only_b) / n,
                "acc_diff": (only_a - only_b) / n,
                # Odds ratio on the discordant cells; +0.5 continuity correction
                # keeps it finite when one cell is empty.
                "odds_ratio": (only_a + 0.5) / (only_b + 0.5),
                "p_value": p, "method": method,
            }
        )

        # Supplementary: the same test within each perturbation type.
        for nt, g in m.groupby("noise_type"):
            bb, oa, ob, nn = cell_counts(g)
            pp, mm = mcnemar(oa, ob)
            per_seed[-1].setdefault("_by_noise", []).append(
                {"noise_type": nt, "only_banglabert": oa,
                 "only_char_ngram": ob, "p_value": pp, "method": mm}
            )

    df = pd.DataFrame(per_seed)
    if missing:
        print("[warn] missing prediction files for:")
        for t, s, sd in missing:
            print(f"        {TASK_LABEL[t]} severity {s} seed {sd}")

    # The six pre-registered tests: one per (task, severity), aggregating the
    # discordant counts over the seeds that are available.
    rows = []
    for task, sev in itertools.product(TASKS, SEVERITIES):
        g = df[(df.task == task) & (df.severity == sev)]
        if g.empty:
            continue
        only_a = int(g.only_banglabert.sum())
        only_b = int(g.only_char_ngram.sum())
        p, method = mcnemar(only_a, only_b)
        rows.append(
            {
                "task": task, "severity": sev, "seeds": len(g),
                "n_items": int(g.n_items.sum()),
                "only_banglabert": only_a, "only_char_ngram": only_b,
                "acc_banglabert": float(g.acc_banglabert.mean()),
                "acc_char_ngram": float(g.acc_char_ngram.mean()),
                "acc_diff": float(g.acc_diff.mean()),
                "odds_ratio": (only_a + 0.5) / (only_b + 0.5),
                "p_value": p, "method": method,
                "winner": LABEL[MODEL_A] if only_a > only_b else LABEL[MODEL_B],
            }
        )
    main = pd.DataFrame(rows)
    main["p_holm"] = holm(main.p_value.tolist())
    main["significant_0.05"] = main.p_holm < 0.05
    return main, df.drop(columns=["_by_noise"], errors="ignore")


def tex(main: pd.DataFrame, path: str) -> None:
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Pre-registered McNemar tests (M6.2): BanglaBERT versus the "
        r"character $n$-gram baseline on per-item predictions pooled over all "
        r"nine perturbation types. $b$ and $c$ are the discordant counts "
        r"(only BanglaBERT correct / only char $n$-gram correct), summed over "
        r"seeds. $p$ values are Holm-Bonferroni adjusted over the six tests.}",
        r"\label{tab:mcnemar}",
        r"\begin{tabular}{llrrrrl}", r"\toprule",
        r"Dataset & Sev. & $b$ & $c$ & OR & $p_{\text{Holm}}$ & Favours \\",
        r"\midrule",
    ]
    for _, r in main.iterrows():
        p = "$<10^{-300}$" if r.p_holm == 0 else f"${r.p_holm:.2e}$".replace("e", r"\mathrm{e}")
        fav = "BanglaBERT" if r.only_banglabert > r.only_char_ngram else r"char $n$-gram"
        lines.append(
            f"{TASK_LABEL[r.task]} & {r.severity} & {r.only_banglabert} & "
            f"{r.only_char_ngram} & {r.odds_ratio:.2f} & {p} & {fav} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", os.path.basename(path))


def main_cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="results/preds")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)

    main, per_seed = run(a.preds)

    pd.set_option("display.width", 200)
    print("\n=== Six pre-registered tests ===")
    cols = ["task", "severity", "seeds", "n_items", "only_banglabert",
            "only_char_ngram", "acc_banglabert", "acc_char_ngram",
            "odds_ratio", "p_holm", "winner"]
    print(main[cols].round(4).to_string(index=False))

    print("\n=== Per seed ===")
    print(per_seed[["task", "severity", "seed", "only_banglabert",
                    "only_char_ngram", "acc_diff", "p_value"]]
          .round(4).to_string(index=False))

    main.to_csv(os.path.join(a.out, "mcnemar_main.csv"), index=False)
    per_seed.to_csv(os.path.join(a.out, "mcnemar_per_seed.csv"), index=False)
    tex(main, os.path.join(a.out, "tables", "mcnemar.tex"))


if __name__ == "__main__":
    main_cli()
