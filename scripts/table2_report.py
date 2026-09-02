"""Table II -- per-dataset statistics after cleaning (METHODOLOGY M2.3).

Reads ``data/clean/<dataset>.csv`` + ``data/clean/<dataset>.prep.json`` produced
by ``data/prepare.py`` and writes:

    results/table2.md
    results/table2.html      (<meta charset="utf-8"> + Bangla font stack)

Per dataset: n after cleaning, class distribution, mean/median grapheme-cluster
length, emoji rate (fraction of texts with >=1 emoji), duplicates removed,
short texts removed. The highest emoji rate names the task that hosts the N10
experiment.

    python scripts/table2_report.py
    python scripts/table2_report.py --clean-dir data/clean --out results/table2.md
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import statistics
import sys
import unicodedata

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import regex  # noqa: E402
# Reading noisebench for the emoji-detection predicate only (same predicate the
# suite ships; METHODOLOGY M2.3 emoji rate feeds the same measurement as M2.4).
from noisebench.perturbations import _is_emoji_codepoint  # noqa: E402,PLC2701

_GRAPHEME_RE = regex.compile(r"\X")

_DEFAULT_CLEAN = os.path.join(_ROOT, "data", "clean")
_DEFAULT_MD = os.path.join(_ROOT, "results", "table2.md")
_DEFAULT_HTML = os.path.join(_ROOT, "results", "table2.html")


def n_clusters(s: str) -> int:
    return len(_GRAPHEME_RE.findall(unicodedata.normalize("NFC", s)))


def has_emoji(s: str) -> bool:
    return any(_is_emoji_codepoint(ch) for ch in s)


def dataset_stats(csv_path: str) -> dict:
    name = os.path.basename(csv_path)[: -len(".csv")]
    df = pd.read_csv(csv_path, dtype={"id": str, "text": str, "label": "Int64",
                                      "split": str}, keep_default_na=False)
    prep_path = csv_path[: -len(".csv")] + ".prep.json"
    prep = json.load(open(prep_path, encoding="utf-8")) if os.path.exists(prep_path) else {}

    lengths = [n_clusters(t) for t in df["text"]]
    n_emoji_texts = sum(has_emoji(t) for t in df["text"])
    n = len(df)

    class_dist = prep.get("class_dist") or {
        str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()
    }

    return {
        "dataset": name,
        "task": _TASK.get(name, prep.get("task", "?")),
        "n": n,
        "class_dist": class_dist,
        "class_dist_pct": {k: (100.0 * v / n if n else 0.0) for k, v in class_dist.items()},
        "mean_len": statistics.fmean(lengths) if lengths else 0.0,
        "median_len": statistics.median(lengths) if lengths else 0.0,
        "emoji_rate": n_emoji_texts / n if n else 0.0,
        "n_emoji_texts": n_emoji_texts,
        "dupes_removed": prep.get("dupes_removed"),
        "short_removed": prep.get("short_removed"),
        "n_raw": prep.get("n_raw"),
        "split_sizes": prep.get("split_sizes"),
        "split_source": prep.get("split_source"),
        "warnings": prep.get("warnings", []),
    }


_TASK = {
    "sentnob": "sentiment (3-class)",
    "bd_shs": "hate speech (binary)",
    "banfakenews": "fake news (binary)",
}


def _fmt_dist(d: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in d.items())


def render_markdown(rows: list[dict], host: str | None) -> str:
    L: list[str] = ["# Table II — dataset statistics after cleaning (METHODOLOGY M2.3)", ""]
    L.append("Grapheme-cluster length = `regex \\X` after NFC. Emoji rate = fraction "
             "of texts with ≥1 code point in `EMOJI_CODEPOINT_RANGES`.")
    L.append("")
    L.append("| dataset | task | n (clean) | class distribution | mean len | median len | emoji rate | dupes removed | short removed |")
    L.append("|---------|------|----------:|--------------------|---------:|-----------:|-----------:|--------------:|--------------:|")
    for r in rows:
        L.append(
            f"| {r['dataset']} | {r['task']} | {r['n']} | "
            f"{_fmt_dist(r['class_dist'])} | {r['mean_len']:.1f} | "
            f"{r['median_len']:.0f} | {r['emoji_rate']:.3%} "
            f"({r['n_emoji_texts']}/{r['n']}) | {r['dupes_removed']} | "
            f"{r['short_removed']} |"
        )
    L.append("")
    L.append("### Splits")
    L.append("")
    L.append("| dataset | source | train | val | test |")
    L.append("|---------|--------|------:|----:|-----:|")
    for r in rows:
        ss = r["split_sizes"] or {}
        L.append(f"| {r['dataset']} | {r['split_source'] or '?'} | "
                 f"{ss.get('train','?')} | {ss.get('val','?')} | {ss.get('test','?')} |")
    L.append("")
    if host:
        hr = next(x for x in rows if x["dataset"] == host)
        L.append(f"**N10 (emoji) experiment host: `{host}`** — highest emoji rate "
                 f"({hr['emoji_rate']:.3%}). METHODOLOGY M2.3.")
    else:
        L.append("_N10 host undetermined — no dataset rows._")
    L.append("")
    warns = [(r["dataset"], w) for r in rows for w in r.get("warnings", [])]
    if warns:
        L.append("### Warnings carried from `data/prepare.py`")
        L.append("")
        for ds, w in warns:
            L.append(f"- `{ds}`: {w}")
        L.append("")
    return "\n".join(L) + "\n"


_HTML_HEAD = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Table II — dataset statistics</title>
<style>
  body { font-family: "Noto Sans Bengali", "Nirmala UI", system-ui, sans-serif;
         line-height: 1.5; margin: 2rem auto; max-width: 1000px; padding: 0 1rem; }
  h1 { font-size: 1.4rem; } h3 { font-size: 1rem; margin-top: 1.5rem; }
  table { border-collapse: collapse; width: 100%; margin: .75rem 0; font-size: .92rem; }
  th, td { border: 1px solid #ccc; padding: .35rem .5rem; text-align: left; }
  th { background: #f2f2f2; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  code { background: #f4f4f4; padding: .05rem .3rem; border-radius: 3px; }
</style>
</head>
<body>
"""


def render_html(rows: list[dict], host: str | None) -> str:
    P: list[str] = [_HTML_HEAD, "<h1>Table II — dataset statistics after cleaning</h1>"]
    P.append("<p>Grapheme-cluster length = <code>regex \\X</code> after NFC. "
             "Emoji rate = fraction of texts with &ge;1 code point in "
             "<code>EMOJI_CODEPOINT_RANGES</code>.</p>")
    P.append("<table><thead><tr><th>dataset</th><th>task</th><th>n (clean)</th>"
             "<th>class distribution</th><th>mean len</th><th>median len</th>"
             "<th>emoji rate</th><th>dupes removed</th><th>short removed</th>"
             "</tr></thead><tbody>")
    for r in rows:
        P.append(
            "<tr>"
            f"<td>{html.escape(r['dataset'])}</td><td>{html.escape(r['task'])}</td>"
            f"<td class='num'>{r['n']}</td><td>{html.escape(_fmt_dist(r['class_dist']))}</td>"
            f"<td class='num'>{r['mean_len']:.1f}</td><td class='num'>{r['median_len']:.0f}</td>"
            f"<td class='num'>{r['emoji_rate']:.3%} ({r['n_emoji_texts']}/{r['n']})</td>"
            f"<td class='num'>{r['dupes_removed']}</td><td class='num'>{r['short_removed']}</td>"
            "</tr>"
        )
    P.append("</tbody></table>")
    P.append("<h3>Splits</h3><table><thead><tr><th>dataset</th><th>source</th>"
             "<th>train</th><th>val</th><th>test</th></tr></thead><tbody>")
    for r in rows:
        ss = r["split_sizes"] or {}
        P.append(f"<tr><td>{html.escape(r['dataset'])}</td>"
                 f"<td>{html.escape(str(r['split_source'] or '?'))}</td>"
                 f"<td class='num'>{ss.get('train','?')}</td>"
                 f"<td class='num'>{ss.get('val','?')}</td>"
                 f"<td class='num'>{ss.get('test','?')}</td></tr>")
    P.append("</tbody></table>")
    if host:
        hr = next(x for x in rows if x["dataset"] == host)
        P.append(f"<p><strong>N10 (emoji) experiment host: <code>{html.escape(host)}</code></strong>"
                 f" — highest emoji rate ({hr['emoji_rate']:.3%}).</p>")
    P.append("</body></html>\n")
    return "\n".join(P)


def build(clean_dir: str) -> tuple[list[dict], str | None]:
    paths = sorted(p for p in glob.glob(os.path.join(clean_dir, "*.csv")))
    rows = [dataset_stats(p) for p in paths]
    host = max(rows, key=lambda r: r["emoji_rate"])["dataset"] if rows else None
    return rows, host


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default=_DEFAULT_CLEAN)
    ap.add_argument("--out", default=_DEFAULT_MD)
    ap.add_argument("--out-html", default=_DEFAULT_HTML)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    rows, host = build(args.clean_dir)
    if not rows:
        print(f"no data/clean/*.csv under {args.clean_dir} — run data/prepare.py first",
              file=sys.stderr)
        return 2

    from data.prepare import _write_utf8, _safe_relpath

    _write_utf8(args.out, render_markdown(rows, host))
    _write_utf8(args.out_html, render_html(rows, host))
    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        print(render_markdown(rows, host))
    print(f"[written] {_safe_relpath(args.out, _ROOT)}")
    print(f"[written] {_safe_relpath(args.out_html, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
