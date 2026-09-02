"""The four METHODOLOGY M2.4 measurements on the real corpora.

    (1) 9x5 CER table on real corpus samples, side by side with the synthetic
        development table; ratio structure vs the 2.00 / 2.00 / 1.75 / 1.43
        target.
    (2) N5 per-set firing counts: n_eligible (grouping pass) and n_applied
        (summed off PerturbStats.submodes) per HOMOPHONE_SET per dataset.
        ই/ঈ and উ/ঊ are explicitly under review.
    (3) Emoji-range coverage per dataset (validate.emoji_range_coverage) --
        an UPPER BOUND against the emoji 1.4.2 / Unicode 13 reference.
    (4) Normalizer touch_rate per dataset and overall -- the R6 go/no-go
        (M5.4): under ~5% => drop R6, report the null.

Reads ``data/clean/<dataset>.csv`` from ``data/prepare.py``. Writes
``results/m2_4.md``, ``results/m2_4.html``, ``results/m2_4.json``.

    python scripts/m2_4_measurements.py
    python scripts/m2_4_measurements.py --sample-per-dataset 300 --seeds 42,1337,2024

Nothing here modifies ``noisebench/``. It imports two private helpers
(``_homophone_occurrences``, ``_HOMOPHONE_MEMBER_TO_SET``) that M2.4(2)
explicitly needs to read.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import random
import statistics
import sys
import unicodedata

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench import NOISE_TYPES, perturb_with_stats  # noqa: E402
from noisebench.bangla_maps import HOMOPHONE_SETS  # noqa: E402
from noisebench.perturbations import (  # noqa: E402,PLC2701
    _HOMOPHONE_MEMBER_TO_SET,
    _homophone_occurrences,
    clusters,
)
from noisebench.validate import (  # noqa: E402
    build_cer_table,
    emoji_range_coverage,
    normalize_banglabert,
    normalizer_available,
)

_DEFAULT_CLEAN = os.path.join(_ROOT, "data", "clean")
_TARGET_RATIOS = (2.00, 2.00, 1.75, 1.43)
SAMPLE_SEED = 20260902


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def load_samples(clean_dir: str, k: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(clean_dir, "*.csv"))):
        name = os.path.basename(path)[: -len(".csv")]
        texts = pd.read_csv(path, dtype=str, keep_default_na=False)["text"].tolist()
        rng = random.Random(f"{SAMPLE_SEED}\x00{name}")
        out[name] = texts if len(texts) <= k else rng.sample(texts, k)
    if len(out) > 1:
        rng = random.Random(f"{SAMPLE_SEED}\x00combined")
        pool = [t for texts in out.values() for t in texts]
        out["ALL"] = pool if len(pool) <= k else rng.sample(pool, k)
    return out


def _synthetic_corpus() -> list[str]:
    from scripts.phase1_report import _make_corpus

    return _make_corpus()


# ---------------------------------------------------------------------------
# (1) CER table
# ---------------------------------------------------------------------------


def _ratios(row: list[float]) -> list[float]:
    return [row[i + 1] / row[i] if row[i] else float("nan") for i in range(len(row) - 1)]


def measure_cer(samples: dict[str, list[str]], seeds: tuple[int, ...]) -> dict:
    synth = build_cer_table(_synthetic_corpus(), seeds=seeds)
    real = {ds: build_cer_table(texts, seeds=seeds) for ds, texts in samples.items()}
    return {"synthetic": synth, "real": real, "seeds": list(seeds)}


# ---------------------------------------------------------------------------
# (2) N5 per-set firing
# ---------------------------------------------------------------------------


def _set_key(homoset: tuple[str, ...]) -> str:
    return "/".join(homoset)


def measure_n5(samples: dict[str, list[str]], seeds: tuple[int, ...],
               severity: int = 3) -> dict:
    keys = [_set_key(s) for s in HOMOPHONE_SETS]
    result: dict[str, dict] = {}
    for ds, texts in samples.items():
        if ds == "ALL":
            continue
        elig = {k: 0 for k in keys}
        applied = {k: 0 for k in keys}
        n_texts_with = {k: 0 for k in keys}
        for t in texts:
            seen = set()
            for _ci, _off, member in _homophone_occurrences(clusters(t)):
                k = _set_key(_HOMOPHONE_MEMBER_TO_SET[member])
                elig[k] += 1
                seen.add(k)
            for k in seen:
                n_texts_with[k] += 1
            for seed in seeds:
                _out, st = perturb_with_stats(t, "homophone_confuse", severity, seed)
                for k, v in st.submodes.items():
                    applied[k] = applied.get(k, 0) + v
        result[ds] = {
            "n_texts": len(texts),
            "severity_for_applied": severity,
            "seeds": list(seeds),
            "per_set": {
                k: {
                    "n_eligible": elig[k],
                    "n_applied": applied.get(k, 0),
                    "texts_with_ge1": n_texts_with[k],
                    "eligible_per_1k_texts": 1000.0 * elig[k] / max(1, len(texts)),
                }
                for k in keys
            },
        }
    # under-review flags, aggregated across datasets
    review = {}
    for target in ("ই/ঈ", "উ/ঊ"):
        tot_elig = sum(result[ds]["per_set"].get(target, {}).get("n_eligible", 0)
                       for ds in result)
        tot_texts = sum(result[ds]["n_texts"] for ds in result)
        per_1k = 1000.0 * tot_elig / max(1, tot_texts)
        review[target] = {
            "total_eligible": tot_elig,
            "eligible_per_1k_texts": per_1k,
            "recommend_drop": per_1k < 1.0,  # < ~1 occurrence per 1000 texts
        }
    return {"datasets": result, "under_review": review, "set_order": keys}


# ---------------------------------------------------------------------------
# (3) emoji coverage
# ---------------------------------------------------------------------------


def measure_emoji(samples: dict[str, list[str]]) -> dict:
    out: dict[str, dict | None] = {}
    for ds, texts in samples.items():
        out[ds] = emoji_range_coverage(texts)
    return {
        "reference": "emoji 1.4.2 (~Unicode 13 / 2021) -- coverage is an UPPER "
                     "BOUND against this era, not current Unicode",
        "per_dataset": out,
    }


# ---------------------------------------------------------------------------
# (4) normalizer touch_rate  -- R6 go/no-go
# ---------------------------------------------------------------------------


def _classify_touch(before: str, after: str) -> str:
    if before == after:
        return "unchanged"
    tags = []
    if len(after) != len(before):
        tags.append(f"len {len(before)}->{len(after)}")
    if "<URL>" in before or "http" in before.lower():
        tags.append("url-ish")
    if any(c.isdigit() for c in before):
        tags.append("has-digit")
    if before.replace(" ", "") == after.replace(" ", ""):
        tags.append("whitespace-only")
    if unicodedata.normalize("NFKC", before) == after:
        tags.append("==NFKC")
    return ", ".join(tags) or "other"


def measure_normalizer(samples: dict[str, list[str]], max_examples: int = 12) -> dict:
    if not normalizer_available():
        return {"available": False}
    per: dict[str, dict] = {}
    for ds, texts in samples.items():
        touched = 0
        examples: list[dict] = []
        cats: dict[str, int] = {}
        for t in texts:
            norm = normalize_banglabert(t)
            if norm is not None and norm != t:
                touched += 1
                cat = _classify_touch(t, norm)
                cats[cat] = cats.get(cat, 0) + 1
                if len(examples) < max_examples:
                    examples.append({"before": t[:160], "after": norm[:160], "kind": cat})
        per[ds] = {
            "n_texts": len(texts),
            "touched": touched,
            "touch_rate": touched / len(texts) if texts else 0.0,
            "kinds": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "examples": examples,
        }
    overall_texts = sum(v["n_texts"] for k, v in per.items() if k != "ALL")
    overall_touched = sum(v["touched"] for k, v in per.items() if k != "ALL")
    overall_rate = overall_touched / overall_texts if overall_texts else 0.0
    return {
        "available": True,
        "per_dataset": per,
        "overall_touch_rate": overall_rate,
        "threshold": 0.05,
        "recommend_drop_r6": overall_rate < 0.05,
    }


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _cer_block_md(cer: dict) -> list[str]:
    L = [f"## (1) 9×5 CER — real vs synthetic  (seeds {cer['seeds']})", ""]
    L.append(f"Target severity ratios s2/s1, s3/s2, s4/s3, s5/s4 = "
             f"**{' / '.join(f'{x:.2f}' for x in _TARGET_RATIOS)}**. "
             f"Flag = any ratio > 10% off target.")
    L.append("")
    tables = [("synthetic (dev corpus)", cer["synthetic"])]
    tables += [(f"real: {ds}", tbl) for ds, tbl in cer["real"].items()]
    for title, tbl in tables:
        L.append(f"### {title}")
        L.append("")
        L.append("| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | flag |")
        L.append("|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|")
        for nt in NOISE_TYPES:
            row = tbl[nt]
            rr = _ratios(row)
            off = any(
                r == r and abs(r - t) > 0.10 * t
                for r, t in zip(rr, _TARGET_RATIOS)
            )
            L.append(
                f"| {nt} | " + " | ".join(f"{v:.4f}" for v in row) + " | "
                + " | ".join(f"{r:.2f}" for r in rr)
                + f" | {'⚠' if off else ''} |"
            )
        L.append("")
    return L


def _n5_block_md(n5: dict) -> list[str]:
    L = ["## (2) N5 homophone-set firing counts", ""]
    L.append("`n_eligible` = member-occurrence count (grouping pass over "
             "`_homophone_occurrences`, severity-independent). `n_applied` = "
             "sum of `PerturbStats.submodes` at severity 3 over the listed seeds. "
             "`per 1k` = eligible occurrences per 1000 sampled texts.")
    L.append("")
    for ds, d in n5["datasets"].items():
        L.append(f"### {ds}  (n={d['n_texts']} texts, applied@s{d['severity_for_applied']})")
        L.append("")
        L.append("| homophone set | n_eligible | per 1k texts | texts ≥1 | n_applied |")
        L.append("|---|--:|--:|--:|--:|")
        for k in n5["set_order"]:
            ps = d["per_set"][k]
            L.append(f"| {k} | {ps['n_eligible']} | {ps['eligible_per_1k_texts']:.1f} "
                     f"| {ps['texts_with_ge1']} | {ps['n_applied']} |")
        L.append("")
    L.append("### Sets under review (M2.4)")
    L.append("")
    L.append("| set | total eligible | per 1k texts | recommendation |")
    L.append("|---|--:|--:|---|")
    for k, v in n5["under_review"].items():
        rec = "**DROP — evidence below threshold**" if v["recommend_drop"] else "keep"
        L.append(f"| {k} | {v['total_eligible']} | {v['eligible_per_1k_texts']:.2f} | {rec} |")
    L.append("")
    L.append("Threshold: < 1 eligible occurrence per 1000 texts ⇒ recommend removal "
             "with this evidence, per METHODOLOGY M2.4(2). Not removed automatically.")
    L.append("")
    return L


def _emoji_block_md(em: dict) -> list[str]:
    L = ["## (3) Emoji-range coverage", "", em["reference"], ""]
    L.append("| dataset | emoji lib | occurrences | coverage (any) | coverage (full) | over-match cps |")
    L.append("|---|---|--:|--:|--:|--:|")
    for ds, c in em["per_dataset"].items():
        if c is None:
            L.append(f"| {ds} | _not installed_ | — | — | — | — |")
            continue
        L.append(f"| {ds} | {c['emoji_lib_version']} | {c['n_occurrences']} | "
                 f"{c['coverage_any']:.1%} | {c['coverage_full']:.1%} | "
                 f"{c['n_false_positive_codepoints']} |")
    L.append("")
    worst = [
        (ds, c) for ds, c in em["per_dataset"].items()
        if c and c["n_occurrences"] >= 20 and c["coverage_any"] < 0.90
    ]
    if worst:
        L.append("**Coverage < ~90% on ≥20 occurrences — proposed (NOT applied) range widening:**")
        L.append("")
        for ds, c in worst:
            ucp = sorted({ord(ch) for e in c["uncaught"] for ch in e})
            blocks = sorted({f"U+{cp:04X}" for cp in ucp})
            L.append(f"- `{ds}`: {c['coverage_any']:.1%} any-coverage. Uncaught code "
                     f"points: {', '.join(blocks[:20])}"
                     + (" …" if len(blocks) > 20 else ""))
        L.append("")
        L.append("Add the enclosing Unicode blocks to `EMOJI_CODEPOINT_RANGES` in a "
                 "later, reviewed change — do not touch `noisebench/` in Phase 2.")
    else:
        L.append("All datasets with ≥20 emoji occurrences are at ≥90% any-coverage; "
                 "no range change proposed.")
    L.append("")
    return L


def _norm_block_md(nm: dict) -> list[str]:
    L = ["## (4) Normalizer touch_rate — R6 go/no-go (M5.4)", ""]
    if not nm.get("available"):
        L.append("_`csebuetnlp/normalizer` not installed — measurement skipped. "
                 "Install it (`requirements.txt`) and re-run before deciding R6._")
        return L + [""]
    L.append("| dataset | n texts | touched | touch_rate | kinds |")
    L.append("|---|--:|--:|--:|---|")
    for ds, d in nm["per_dataset"].items():
        kinds = ", ".join(f"{k} ({v})" for k, v in list(d["kinds"].items())[:4]) or "—"
        L.append(f"| {ds} | {d['n_texts']} | {d['touched']} | {d['touch_rate']:.2%} | {kinds} |")
    L.append("")
    L.append(f"**Overall touch_rate: {nm['overall_touch_rate']:.2%}** "
             f"(threshold {nm['threshold']:.0%}).")
    L.append("")
    L.append("### What it touched — examples")
    L.append("")
    for ds, d in nm["per_dataset"].items():
        if not d["examples"]:
            continue
        L.append(f"**{ds}:**")
        L.append("")
        for ex in d["examples"][:8]:
            L.append(f"- _{ex['kind']}_")
            L.append(f"  - before: `{ex['before']}`")
            L.append(f"  - after:  `{ex['after']}`")
        L.append("")
    return L


def render_markdown(payload: dict) -> str:
    L = ["# METHODOLOGY M2.4 — real-corpus measurements", ""]
    L.append(f"Sample: {payload['sample_note']}.")
    L.append("")
    L += _cer_block_md(payload["cer"])
    L += _n5_block_md(payload["n5"])
    L += _emoji_block_md(payload["emoji"])
    L += _norm_block_md(payload["normalizer"])
    return "\n".join(L) + "\n"


_HTML_HEAD = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M2.4 — real-corpus measurements</title>
<style>
  body { font-family: "Noto Sans Bengali", "Nirmala UI", system-ui, sans-serif;
         line-height: 1.5; margin: 2rem auto; max-width: 1050px; padding: 0 1rem; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2rem;
       border-bottom: 1px solid #ddd; } h3 { font-size: 1rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .9rem; }
  th, td { border: 1px solid #ccc; padding: .3rem .5rem; text-align: left; }
  th { background: #f2f2f2; }
  code { background: #f4f4f4; padding: .05rem .3rem; border-radius: 3px;
         overflow-wrap: anywhere; }
</style>
</head>
<body>
"""


def render_html(payload: dict) -> str:
    # Markdown is authoritative; the HTML mirror keeps a charset + Bangla font
    # stack per invariant 8. Convert the markdown minimally.
    md = render_markdown(payload)
    body: list[str] = [_HTML_HEAD]
    in_table = False
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if line.strip():
                body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    body.append("</body></html>\n")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", default=_DEFAULT_CLEAN)
    ap.add_argument("--sample-per-dataset", type=int, default=300)
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "m2_4.md"))
    ap.add_argument("--out-html", default=os.path.join(_ROOT, "results", "m2_4.html"))
    ap.add_argument("--out-json", default=os.path.join(_ROOT, "results", "m2_4.json"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    seeds = tuple(int(x) for x in args.seeds.split(","))
    samples = load_samples(args.clean_dir, args.sample_per_dataset)
    if not samples:
        print(f"no data/clean/*.csv under {args.clean_dir} — run data/prepare.py first",
              file=sys.stderr)
        return 2

    payload = {
        "sample_note": f"{args.sample_per_dataset} texts/dataset (seeded), "
                       f"{{{', '.join(map(str, seeds))}}}; datasets: "
                       f"{', '.join(k for k in samples if k != 'ALL')}",
        "cer": measure_cer(samples, seeds),
        "n5": measure_n5(samples, seeds),
        "emoji": measure_emoji(samples),
        "normalizer": measure_normalizer(samples),
    }

    from data.prepare import _write_utf8, _safe_relpath

    _write_utf8(args.out, render_markdown(payload))
    _write_utf8(args.out_html, render_html(payload))
    _write_utf8(args.out_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        print(render_markdown(payload))
    print(f"[written] {_safe_relpath(args.out, _ROOT)}")
    print(f"[written] {_safe_relpath(args.out_html, _ROOT)}")
    print(f"[written] {_safe_relpath(args.out_json, _ROOT)}")

    nm = payload["normalizer"]
    if nm.get("available"):
        print(f"\nR6 recommendation: "
              f"{'DROP (touch_rate < 5%)' if nm['recommend_drop_r6'] else 'KEEP'} "
              f"— overall touch_rate {nm['overall_touch_rate']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
