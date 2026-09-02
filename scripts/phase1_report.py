"""Phase 1 report: Table I examples + the 9x5 CER table + M3 acceptance checks.

Writes UTF-8 files. **Never rely on shell redirection** -- Windows PowerShell's
``>`` reinterprets UTF-8 bytes under the console code page and re-saves as
UTF-16, turning Bangla into mojibake. This script writes the files itself.

    python scripts/phase1_report.py
        -> results/phase1_report.md  and  results/phase1_report.html
           (and prints a plain-text copy to the console)

    python scripts/phase1_report.py --out PATH.md --out-html PATH.html
    python scripts/phase1_report.py --quiet        # files only, no console output

Contents:
  1. Table I  -- for each noise type, one Bangla example
     `original -> perturbed at severity 3` (N10: all three modes).
  2. The 9x5 mean-CER table (noise type x severity), averaged over the sample
     corpus and seeds {42, 1337, 2024}. METHODOLOGY M3 criterion 2: every row
     must be strictly increasing.
  3. Normalizer-survival row: post-BanglaBERT-normalizer CER at severity 3.
  4. PASS/FAIL for all four M3 acceptance criteria.

Nothing here is experiment code; it only exercises `noisebench`.
"""

from __future__ import annotations

import argparse
import html
import os
import random
import sys
from dataclasses import dataclass, field

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench import (  # noqa: E402
    EMOJI_MODES,
    NOISE_TYPES,
    emoji_transform_with_stats,
    perturb_with_stats,
)
from noisebench.validate import (  # noqa: E402
    achieved_rate_table,
    build_cer_table,
    emoji_range_coverage,
    is_strictly_increasing,
    is_valid_output,
    matra_roundtrip_report,
    normalizer_available,
    normalizer_survival,
)
from tests.bangla_samples import (  # noqa: E402
    SENTENCES,
    SENTENCES_EMOJI,
    SENTENCES_GENERAL,
    SENTENCES_NUKTA,
)

SEEDS = (42, 1337, 2024)
CORPUS_N = 200
CORPUS_SENTENCES_PER = 3

_DEFAULT_MD = os.path.join(_ROOT, "results", "phase1_report.md")
_DEFAULT_HTML = os.path.join(_ROOT, "results", "phase1_report.html")

# noise_type -> example sentence (chosen to have many eligible units)
_EXAMPLE_TEXT: dict[str, str] = {
    "char_insert": SENTENCES_GENERAL[0],
    "char_delete": SENTENCES_GENERAL[2],
    "char_substitute": SENTENCES_GENERAL[3],
    "char_transpose": SENTENCES_GENERAL[4],
    "homophone_confuse": SENTENCES_GENERAL[6],
    "matra_perturb": SENTENCES_GENERAL[18],
    "conjunct_split": "বিজ্ঞান ও প্রযুক্তির দ্রুত অগ্রগতিতে রাষ্ট্রের সমস্যা কমছে।",
    "elongate": "সে খুব ভালো আর আমি আজ খুশি।",
    "whitespace_error": SENTENCES_GENERAL[0],
}
# per-type seed chosen so the severity-3 example is legible (>=2 edits; more
# than one N8/N9 submode where possible). Deterministic.
_EXAMPLE_SEED: dict[str, int] = {
    "conjunct_split": 8,
    "elongate": 3,
    "matra_perturb": 2,
    "whitespace_error": 0,
}
_NOISE_LABEL: dict[str, str] = {
    "char_insert": "N1 char_insert",
    "char_delete": "N2 char_delete",
    "char_substitute": "N3 char_substitute",
    "char_transpose": "N4 char_transpose",
    "homophone_confuse": "N5 homophone_confuse",
    "matra_perturb": "N6 matra_perturb",
    "conjunct_split": "N7 conjunct_split",
    "elongate": "N8 elongate",
    "whitespace_error": "N9 whitespace_error",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExampleRow:
    tag: str
    noise: str
    original: str
    perturbed: str
    eligible: int
    applied: int
    submodes: str
    seed: str


@dataclass
class Report:
    corpus_n: int
    seeds: tuple[int, ...]
    examples: list[ExampleRow] = field(default_factory=list)
    cer_table: dict[str, list[float]] = field(default_factory=dict)
    cer_monotonic: dict[str, bool] = field(default_factory=dict)
    rate_table: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    normalizer_survival: dict[str, dict[str, float]] | None = None
    emoji_coverage: dict[str, object] | None = None
    determinism_ok: bool = False
    monotonic_ok: bool = False
    monotonic_bad: list[str] = field(default_factory=list)
    validity_ok: bool = False
    nfc_ok: bool = False
    normalizer_matra_ok: bool | None = None
    nonidentity: list[tuple[str, int, int, float, bool]] = field(default_factory=list)
    nonidentity_ok: bool = False

    @property
    def overall_ok(self) -> bool:
        return (
            self.determinism_ok
            and self.monotonic_ok
            and self.validity_ok
            and self.nfc_ok
            and self.nonidentity_ok
            and self.normalizer_matra_ok is not False
        )


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _make_corpus() -> list[str]:
    rng = random.Random(20260901)
    pool = list(SENTENCES)
    corpus: list[str] = []
    while len(corpus) < CORPUS_N:
        corpus.append(" ".join(rng.sample(pool, k=CORPUS_SENTENCES_PER)))
    return corpus


def _fmt_submodes(submodes: dict[str, int]) -> str:
    return ", ".join(f"{k}={v}" for k, v in submodes.items()) or "-"


def compute_report() -> Report:
    corpus = _make_corpus()
    rep = Report(corpus_n=len(corpus), seeds=SEEDS)

    # 1. Table I
    for noise_type in NOISE_TYPES:
        text = _EXAMPLE_TEXT[noise_type]
        seed = _EXAMPLE_SEED.get(noise_type, 42)
        out, stats = perturb_with_stats(text, noise_type, severity=3, seed=seed)
        rep.examples.append(ExampleRow(
            tag=_NOISE_LABEL[noise_type].split()[0],
            noise=_NOISE_LABEL[noise_type].split(maxsplit=1)[1],
            original=text, perturbed=out,
            eligible=stats.n_eligible, applied=stats.n_applied,
            submodes=_fmt_submodes(stats.submodes), seed=str(seed),
        ))
    etext = SENTENCES_EMOJI[0]
    for mode in EMOJI_MODES:
        out, stats = emoji_transform_with_stats(etext, mode)
        rep.examples.append(ExampleRow(
            tag="N10", noise=f"emoji_transform ({mode})",
            original=etext, perturbed=out,
            eligible=stats.n_eligible, applied=stats.n_applied,
            submodes=_fmt_submodes(stats.submodes), seed="-",
        ))

    # 2. CER table
    rep.cer_table = build_cer_table(corpus, seeds=SEEDS)
    rep.cer_monotonic = {
        nt: is_strictly_increasing(rep.cer_table[nt]) for nt in NOISE_TYPES
    }
    rep.monotonic_bad = [nt for nt, ok in rep.cer_monotonic.items() if not ok]
    rep.monotonic_ok = not rep.monotonic_bad

    # 2b. Achieved vs nominal perturbation rate
    rep.rate_table = achieved_rate_table(corpus, seeds=SEEDS)

    # 3. Normalizer survival
    rep.normalizer_survival = normalizer_survival(corpus, severity=3, seeds=SEEDS)

    # 3b. Emoji-range coverage (M0 Q9) -- provisional; the real number comes from
    # the Phase 2 corpora. Here it only exercises the wiring on the emoji/mixed
    # sample sentences.
    rep.emoji_coverage = emoji_range_coverage(list(SENTENCES))

    # 4a. Determinism
    det_ok = True
    for text in corpus[:50]:
        for noise_type in NOISE_TYPES:
            for sev in range(1, 6):
                a = perturb_with_stats(text, noise_type, sev, 7)[0]
                b = perturb_with_stats(text, noise_type, sev, 7)[0]
                det_ok = det_ok and (a == b)
    rep.determinism_ok = det_ok

    # 4b. Validity + matra survival
    val_ok = True
    for text in corpus[:80]:
        for noise_type in NOISE_TYPES:
            for sev in (1, 3, 5):
                for seed in SEEDS:
                    out = perturb_with_stats(text, noise_type, sev, seed)[0]
                    val_ok = val_ok and is_valid_output(out)
    rep.validity_ok = val_ok
    probes = SENTENCES_NUKTA + [
        "সে ভালো ছেলে", "নৌকা চলে যায়", "মৌমাছি মধু আনে", "শৌখিন মানুষ",
    ]
    rt = matra_roundtrip_report(probes)
    rep.nfc_ok = bool(rt["nfc_ok"])
    rep.normalizer_matra_ok = rt["normalizer_ok"]

    # 4c. Non-identity at severity 5
    all_ok = True
    for noise_type in NOISE_TYPES:
        n_elig = n_changed = 0
        for text in corpus:
            for seed in SEEDS:
                out, stats = perturb_with_stats(text, noise_type, 5, seed)
                if stats.n_eligible > 0:
                    n_elig += 1
                    n_changed += out != text
        rate = n_changed / n_elig if n_elig else 0.0
        ok = rate >= 0.95
        all_ok = all_ok and ok
        rep.nonidentity.append((noise_type, n_changed, n_elig, rate, ok))
    rep.nonidentity_ok = all_ok

    return rep


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

_PASS = {True: "PASS", False: "FAIL", None: "SKIP"}


def _md_escape(cell: str) -> str:
    return cell.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(rep: Report) -> str:
    L: list[str] = []
    L.append("# bangla-noisebench — Phase 1 report")
    L.append("")
    L.append(f"Generated by `scripts/phase1_report.py`. Corpus n={rep.corpus_n} "
             f"(random {CORPUS_SENTENCES_PER}-sentence texts), seeds "
             f"{{{', '.join(map(str, rep.seeds))}}}.")
    L.append("")
    L.append(f"**OVERALL: {'ALL CHECKS PASS' if rep.overall_ok else 'SOME CHECKS FAILED'}**")
    L.append("")

    # Table I
    L.append("## Table I — noise taxonomy (one Bangla example each, severity 3)")
    L.append("")
    L.append("| # | noise | original | perturbed | eligible | applied | submodes | seed |")
    L.append("|---|-------|----------|-----------|---------:|--------:|----------|------|")
    for r in rep.examples:
        L.append("| " + " | ".join(_md_escape(c) for c in [
            r.tag, r.noise, r.original, r.perturbed,
            str(r.eligible), str(r.applied), r.submodes, r.seed,
        ]) + " |")
    L.append("")

    # CER table
    L.append("## 9×5 mean CER table (noise type × severity)")
    L.append("")
    L.append("METHODOLOGY M3 criterion 2: every row must be strictly increasing.")
    L.append("")
    L.append("| noise_type | s1 | s2 | s3 | s4 | s5 | monotonic |")
    L.append("|------------|---:|---:|---:|---:|---:|:---------:|")
    for nt in NOISE_TYPES:
        row = rep.cer_table[nt]
        cells = " | ".join(f"{v:.4f}" for v in row)
        mono = "yes" if rep.cer_monotonic[nt] else "**NO**"
        L.append(f"| {nt} | {cells} | {mono} |")
    L.append("")

    # Achieved vs nominal rate
    L.append("## Achieved vs nominal perturbation rate")
    L.append("")
    L.append("Does *severity = fraction of eligible units* actually hold? For each "
             "cell, over corpus × seeds (texts with ≥1 eligible unit): "
             "`applied/requested` = mean `n_applied / n_requested` (<1.0 ⇒ a "
             "per-type guard is dropping selected units); `achieved` = mean "
             "`n_applied / n_eligible`, the rate the paper can actually claim, "
             "vs the nominal fraction.")
    L.append("")
    L.append("| noise_type | s | nominal | requested/nominal | applied/requested | **achieved** |")
    L.append("|------------|--:|--------:|------------------:|------------------:|-------------:|")
    short: list[str] = []
    for nt in NOISE_TYPES:
        for s, cell in enumerate(rep.rate_table[nt], start=1):
            nominal = cell["nominal_fraction"]
            achieved = cell["achieved_fraction"]
            aor = cell["applied_over_requested"]
            deficit = (nominal - achieved) / nominal if nominal else 0.0
            mark = "  ← short" if deficit >= 0.05 else ""
            if deficit >= 0.05:
                short.append(f"{nt} s{s} ({achieved:.3f} vs {nominal:.2f}, "
                             f"−{deficit:.0%})")
            L.append(f"| {nt} | {s} | {nominal:.2f} | "
                     f"{cell['requested_over_nominal']:.3f} | {aor:.3f} | "
                     f"{achieved:.4f}{mark} |")
    L.append("")
    if short:
        L.append("**Materially short of nominal (≥5%):** " + "; ".join(short) + ".")
        L.append("")
        L.append("The seeded-Bernoulli count rule itself is unbiased everywhere "
                 "(`requested/nominal` ≈ 1.00). The remaining shortfall is "
                 "**N2** (`char_delete`): its don't-empty-a-token guard drops a "
                 "selection whenever every cluster of a one-cluster token is "
                 "picked, and no implementation can reach the nominal fraction "
                 "without violating that guard — so METHODOLOGY M1.3 reports N2's "
                 "achieved fraction next to the nominal one. (N4 `char_transpose` "
                 "previously fell ~22% short at severity 5 from its overlap-skip "
                 "guard; the disjoint-pair rewrite now hits nominal exactly — "
                 "`applied/requested` = 1.000.)")
    else:
        L.append("All types achieve the nominal fraction within 5%.")
    L.append("")

    # Normalizer survival
    L.append("## Normalizer survival — CER(original, normalize(perturbed)) at severity 3")
    L.append("")
    if rep.normalizer_survival is None:
        L.append("_`normalizer` (csebuetnlp/normalizer) not installed — skipped._")
    else:
        L.append("A value near zero means the BanglaBERT normalizer (partly) undoes "
                 "that noise type and it must be flagged in the paper. `touch_rate` is "
                 "the fraction of calls where the normalizer actually changed anything "
                 "— a high post-normalizer CER at a **low** touch rate mostly proves the "
                 "normalizer didn't run, not that it survived; see the caveat below the "
                 "table.")
        L.append("")
        L.append("| noise_type | post-normalizer CER | touch_rate | flag |")
        L.append("|------------|--------------------:|-----------:|------|")
        for nt in NOISE_TYPES:
            v = rep.normalizer_survival[nt]
            flag = "near zero CER — check" if v["post_normalizer_cer"] < 0.01 else ""
            L.append(f"| {nt} | {v['post_normalizer_cer']:.4f} | {v['touch_rate']:.1%} | {flag} |")
        max_touch = max(v["touch_rate"] for v in rep.normalizer_survival.values())
        if max_touch < 0.05:
            L.append("")
            L.append(f"**Caveat:** touch_rate is <5% for every noise type on this "
                     f"synthetic corpus (max {max_touch:.1%}) — `normalize()` with "
                     f"default arguments does essentially nothing on clean Bangla "
                     f"sentences with no URLs/emails/punctuation-spacing issues. This "
                     f"survival check is not yet a meaningful test on this corpus; it "
                     f"will become one on the real, messier datasets in Phase 2.")
    L.append("")

    # Emoji-range coverage (M0 Q9)
    L.append("## Emoji-range coverage (METHODOLOGY M0 Q9) — provisional")
    L.append("")
    L.append("Scores the shipped hand-rolled `EMOJI_CODEPOINT_RANGES` against the "
             "`emoji` package (an optional, measurement-only dependency — "
             "`noisebench/` never imports it). **This runs on the tiny emoji/mixed "
             "sample sentences only**; the number that goes in the paper comes from "
             "the real Phase 2 corpora (`validate.emoji_range_coverage`). The "
             "reference is `emoji` 1.4.2 ≈ Unicode 13 (2021), so the figure is an "
             "upper bound against a 2021-era reference, not current Unicode.")
    L.append("")
    ec = rep.emoji_coverage
    if ec is None:
        L.append("_`emoji` package not installed — skipped._")
    else:
        L.append(f"- `emoji` version: `{ec['emoji_lib_version']}`")
        L.append(f"- emoji occurrences found: {ec['n_occurrences']}")
        L.append(f"- coverage (any code point flagged): "
                 f"{ec['coverage_any']:.1%} ({ec['n_caught_any']}/{ec['n_occurrences']})")
        L.append(f"- coverage (every non-glue code point flagged): "
                 f"{ec['coverage_full']:.1%} ({ec['n_caught_full']}/{ec['n_occurrences']})")
        L.append(f"- flagged code points the `emoji` package never attributes to "
                 f"an emoji (possible over-match): {ec['n_false_positive_codepoints']}")
        uncaught = ec["uncaught"]  # type: ignore[index]
        if uncaught:
            L.append(f"- uncaught examples: {' '.join(uncaught)}")
    L.append("")

    # Acceptance
    L.append("## M3 acceptance criteria")
    L.append("")
    L.append("| # | criterion | result |")
    L.append("|---|-----------|--------|")
    L.append(f"| 1 | Determinism (repeat calls identical) | {_PASS[rep.determinism_ok]} |")
    mono_detail = "" if rep.monotonic_ok else f" ({', '.join(rep.monotonic_bad)})"
    L.append(f"| 2 | Monotonicity (every CER row strictly increasing) | "
             f"{_PASS[rep.monotonic_ok]}{mono_detail} |")
    L.append(f"| 3 | Validity (NFC, no U+FFFD, no orphan nukta) | {_PASS[rep.validity_ok]} |")
    L.append(f"| 3 | ো/ৌ survive NFC round trip | {_PASS[rep.nfc_ok]} |")
    L.append(f"| 3 | ো/ৌ survive BanglaBERT-normalizer round trip | "
             f"{_PASS[rep.normalizer_matra_ok]} |")
    L.append(f"| 4 | Non-identity at severity 5 (≥95% of eligible texts change) | "
             f"{_PASS[rep.nonidentity_ok]} |")
    L.append("")

    L.append("### Non-identity at severity 5 — detail")
    L.append("")
    L.append("| noise_type | changed / eligible | rate | ok |")
    L.append("|------------|-------------------:|-----:|:--:|")
    for nt, changed, elig, rate, ok in rep.nonidentity:
        L.append(f"| {nt} | {changed} / {elig} | {rate:.1%} | {'ok' if ok else 'LOW'} |")
    L.append("")
    return "\n".join(L) + "\n"


_HTML_HEAD = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bangla-noisebench — Phase 1 report</title>
<style>
  body { font-family: "Noto Sans Bengali", "Nirmala UI", "Kalpurush",
         system-ui, -apple-system, "Segoe UI", sans-serif;
         line-height: 1.5; margin: 2rem auto; max-width: 1100px; padding: 0 1rem;
         color: #1a1a1a; }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.15rem; margin-top: 2rem; border-bottom: 1px solid #ddd;
       padding-bottom: .2rem; }
  h3 { font-size: 1rem; }
  table { border-collapse: collapse; width: 100%; margin: .75rem 0;
          font-size: .92rem; }
  th, td { border: 1px solid #ccc; padding: .35rem .5rem; text-align: left;
           vertical-align: top; }
  th { background: #f2f2f2; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  .bn { font-size: 1.05rem; }
  .pass { color: #0a7d28; font-weight: 600; }
  .fail { color: #b00020; font-weight: 600; }
  .skip { color: #777; }
  code { background: #f4f4f4; padding: .05rem .3rem; border-radius: 3px; }
</style>
</head>
<body>
"""


def _cell(text: str, cls: str = "") -> str:
    attr = f' class="{cls}"' if cls else ""
    return f"<td{attr}>{html.escape(text)}</td>"


def _verdict_span(v: bool | None) -> str:
    label = _PASS[v]
    cls = {"PASS": "pass", "FAIL": "fail", "SKIP": "skip"}[label]
    return f'<span class="{cls}">{label}</span>'


def render_html(rep: Report) -> str:
    P: list[str] = [_HTML_HEAD]
    P.append("<h1>bangla-noisebench — Phase 1 report</h1>")
    P.append(f"<p>Generated by <code>scripts/phase1_report.py</code>. "
             f"Corpus n={rep.corpus_n} (random {CORPUS_SENTENCES_PER}-sentence texts), "
             f"seeds {{{', '.join(map(str, rep.seeds))}}}.</p>")
    overall = ("ALL CHECKS PASS" if rep.overall_ok else "SOME CHECKS FAILED")
    cls = "pass" if rep.overall_ok else "fail"
    P.append(f'<p><strong>OVERALL: <span class="{cls}">{overall}</span></strong></p>')

    # Table I
    P.append("<h2>Table I — noise taxonomy (one Bangla example each, severity 3)</h2>")
    P.append("<table><thead><tr>"
             "<th>#</th><th>noise</th><th>original</th><th>perturbed</th>"
             "<th>eligible</th><th>applied</th><th>submodes</th><th>seed</th>"
             "</tr></thead><tbody>")
    for r in rep.examples:
        P.append("<tr>"
                 + _cell(r.tag) + _cell(r.noise)
                 + _cell(r.original, "bn") + _cell(r.perturbed, "bn")
                 + _cell(str(r.eligible), "num") + _cell(str(r.applied), "num")
                 + _cell(r.submodes) + _cell(r.seed) + "</tr>")
    P.append("</tbody></table>")

    # CER table
    P.append("<h2>9×5 mean CER table (noise type × severity)</h2>")
    P.append("<p>METHODOLOGY M3 criterion 2: every row must be strictly increasing.</p>")
    P.append("<table><thead><tr><th>noise_type</th>"
             + "".join(f"<th>s{s}</th>" for s in range(1, 6))
             + "<th>monotonic</th></tr></thead><tbody>")
    for nt in NOISE_TYPES:
        row = rep.cer_table[nt]
        tds = "".join(f'<td class="num">{v:.4f}</td>' for v in row)
        mono = rep.cer_monotonic[nt]
        mono_html = '<td class="pass">yes</td>' if mono else '<td class="fail">NO</td>'
        P.append(f"<tr>{_cell(nt)}{tds}{mono_html}</tr>")
    P.append("</tbody></table>")

    # Achieved vs nominal rate
    P.append("<h2>Achieved vs nominal perturbation rate</h2>")
    P.append("<p>Does <em>severity = fraction of eligible units</em> actually hold? "
             "<code>applied/requested</code> &lt; 1.0 means a per-type guard is "
             "dropping selected units; <code>achieved</code> = mean "
             "<code>n_applied / n_eligible</code> is the rate the paper can claim.</p>")
    P.append("<table><thead><tr><th>noise_type</th><th>s</th><th>nominal</th>"
             "<th>requested/nominal</th><th>applied/requested</th>"
             "<th>achieved</th></tr></thead><tbody>")
    short_html: list[str] = []
    for nt in NOISE_TYPES:
        for s, cell in enumerate(rep.rate_table[nt], start=1):
            nominal = cell["nominal_fraction"]
            achieved = cell["achieved_fraction"]
            deficit = (nominal - achieved) / nominal if nominal else 0.0
            cls = ' class="fail"' if deficit >= 0.05 else ' class="num"'
            if deficit >= 0.05:
                short_html.append(f"{nt} s{s} ({achieved:.3f} vs {nominal:.2f}, "
                                  f"&minus;{deficit:.0%})")
            P.append(f"<tr>{_cell(nt)}<td class='num'>{s}</td>"
                     f"<td class='num'>{nominal:.2f}</td>"
                     f"<td class='num'>{cell['requested_over_nominal']:.3f}</td>"
                     f"<td class='num'>{cell['applied_over_requested']:.3f}</td>"
                     f"<td{cls}>{achieved:.4f}</td></tr>")
    P.append("</tbody></table>")
    if short_html:
        P.append("<p><strong>Materially short of nominal (&ge;5%):</strong> "
                 + "; ".join(short_html) + ". The count rule is unbiased "
                 "(<code>requested/nominal</code> &asymp; 1.00); the remaining "
                 "shortfall is <strong>N2</strong>'s don't-empty-a-token guard, "
                 "which no implementation can avoid — reported next to nominal in "
                 "METHODOLOGY M1.3. N4's earlier overlap-skip shortfall is gone "
                 "since the disjoint-pair rewrite (<code>applied/requested</code> "
                 "= 1.000).</p>")

    # Normalizer survival
    P.append("<h2>Normalizer survival — CER(original, normalize(perturbed)) at severity 3</h2>")
    if rep.normalizer_survival is None:
        P.append("<p><em><code>normalizer</code> (csebuetnlp/normalizer) not installed — skipped.</em></p>")
    else:
        P.append("<p>A value near zero means the BanglaBERT normalizer (partly) undoes "
                 "that noise type and it must be flagged in the paper. "
                 "<code>touch_rate</code> is the fraction of calls where the normalizer "
                 "actually changed anything — a high post-normalizer CER at a "
                 "<strong>low</strong> touch rate mostly proves the normalizer didn't "
                 "run, not that it survived.</p>")
        P.append("<table><thead><tr><th>noise_type</th>"
                 "<th>post-normalizer CER</th><th>touch_rate</th><th>flag</th></tr></thead><tbody>")
        for nt in NOISE_TYPES:
            v = rep.normalizer_survival[nt]
            flag = "near zero CER — check" if v["post_normalizer_cer"] < 0.01 else ""
            P.append(f'<tr>{_cell(nt)}<td class="num">{v["post_normalizer_cer"]:.4f}</td>'
                     f'<td class="num">{v["touch_rate"]:.1%}</td>{_cell(flag)}</tr>')
        P.append("</tbody></table>")
        max_touch = max(v["touch_rate"] for v in rep.normalizer_survival.values())
        if max_touch < 0.05:
            P.append(f'<p><strong>Caveat:</strong> touch_rate is &lt;5% for every noise '
                     f'type on this synthetic corpus (max {max_touch:.1%}) — '
                     f'<code>normalize()</code> with default arguments does essentially '
                     f'nothing on clean Bangla sentences with no URLs/emails/'
                     f'punctuation-spacing issues. This survival check is not yet a '
                     f'meaningful test on this corpus; it will become one on the real, '
                     f'messier datasets in Phase 2.</p>')

    # Emoji-range coverage (M0 Q9)
    P.append("<h2>Emoji-range coverage (METHODOLOGY M0 Q9) — provisional</h2>")
    P.append("<p>Scores the shipped hand-rolled <code>EMOJI_CODEPOINT_RANGES</code> "
             "against the <code>emoji</code> package (optional, measurement-only — "
             "<code>noisebench/</code> never imports it). <strong>Runs on the tiny "
             "emoji/mixed sample sentences only</strong>; the paper number comes from "
             "the real Phase 2 corpora.</p>")
    ec = rep.emoji_coverage
    if ec is None:
        P.append("<p><em><code>emoji</code> package not installed — skipped.</em></p>")
    else:
        uncaught = ec["uncaught"]  # type: ignore[index]
        P.append("<ul>")
        P.append(f"<li><code>emoji</code> version: <code>{html.escape(str(ec['emoji_lib_version']))}</code></li>")
        P.append(f"<li>emoji occurrences found: {ec['n_occurrences']}</li>")
        P.append(f"<li>coverage (any code point flagged): {ec['coverage_any']:.1%} "
                 f"({ec['n_caught_any']}/{ec['n_occurrences']})</li>")
        P.append(f"<li>coverage (every non-glue code point flagged): {ec['coverage_full']:.1%} "
                 f"({ec['n_caught_full']}/{ec['n_occurrences']})</li>")
        P.append(f"<li>flagged code points the <code>emoji</code> package never "
                 f"attributes to an emoji: {ec['n_false_positive_codepoints']}</li>")
        if uncaught:
            P.append(f"<li>uncaught examples: {html.escape(' '.join(uncaught))}</li>")
        P.append("</ul>")

    # Acceptance
    P.append("<h2>M3 acceptance criteria</h2>")
    P.append("<table><thead><tr><th>#</th><th>criterion</th><th>result</th></tr>"
             "</thead><tbody>")
    rows = [
        ("1", "Determinism (repeat calls identical)", _verdict_span(rep.determinism_ok)),
        ("2", "Monotonicity (every CER row strictly increasing)"
              + ("" if rep.monotonic_ok else f" ({', '.join(rep.monotonic_bad)})"),
         _verdict_span(rep.monotonic_ok)),
        ("3", "Validity (NFC, no U+FFFD, no orphan nukta)", _verdict_span(rep.validity_ok)),
        ("3", "ো/ৌ survive NFC round trip", _verdict_span(rep.nfc_ok)),
        ("3", "ো/ৌ survive BanglaBERT-normalizer round trip",
         _verdict_span(rep.normalizer_matra_ok)),
        ("4", "Non-identity at severity 5 (≥95% of eligible texts change)",
         _verdict_span(rep.nonidentity_ok)),
    ]
    for num, crit, verdict in rows:
        P.append(f"<tr>{_cell(num)}<td>{html.escape(crit)}</td><td>{verdict}</td></tr>")
    P.append("</tbody></table>")

    P.append("<h3>Non-identity at severity 5 — detail</h3>")
    P.append("<table><thead><tr><th>noise_type</th><th>changed / eligible</th>"
             "<th>rate</th><th>ok</th></tr></thead><tbody>")
    for nt, changed, elig, rate, ok in rep.nonidentity:
        okhtml = '<td class="pass">ok</td>' if ok else '<td class="fail">LOW</td>'
        P.append(f'<tr>{_cell(nt)}{_cell(f"{changed} / {elig}")}'
                 f'<td class="num">{rate:.1%}</td>{okhtml}</tr>')
    P.append("</tbody></table>")

    P.append("</body></html>\n")
    return "\n".join(P)


def render_text(rep: Report) -> str:
    """Plain-text console version (also written nowhere; console only)."""
    out: list[str] = []
    w = out.append
    w("=" * 78)
    w(f"bangla-noisebench Phase 1 report   corpus n={rep.corpus_n}  seeds={rep.seeds}")
    w("=" * 78)
    w("")
    w("TABLE I  (severity 3)")
    for r in rep.examples:
        w(f"\n{r.tag} {r.noise}  (seed {r.seed}; eligible={r.eligible}, "
          f"applied={r.applied}, submodes={r.submodes})")
        w(f"  original :  {r.original}")
        w(f"  perturbed:  {r.perturbed}")
    w("")
    w("9x5 MEAN CER TABLE")
    w(f"  {'noise_type':<20}" + "".join(f"s{s}".ljust(10) for s in range(1, 6)) + "monotonic")
    for nt in NOISE_TYPES:
        row = rep.cer_table[nt]
        w(f"  {nt:<20}" + "".join(f"{v:.4f}".ljust(10) for v in row)
          + ("yes" if rep.cer_monotonic[nt] else "NO <-- FAIL"))
    w("")
    w("ACHIEVED vs NOMINAL RATE  (achieved = mean n_applied/n_eligible)")
    w(f"  {'noise_type':<18}{'s':>2}  {'nominal':>8}{'req/nom':>9}{'app/req':>9}{'achieved':>10}")
    short_txt: list[str] = []
    for nt in NOISE_TYPES:
        for s, cell in enumerate(rep.rate_table[nt], start=1):
            nom = cell["nominal_fraction"]
            ach = cell["achieved_fraction"]
            deficit = (nom - ach) / nom if nom else 0.0
            mark = "  <-- short" if deficit >= 0.05 else ""
            if deficit >= 0.05:
                short_txt.append(f"{nt} s{s} ({ach:.3f} vs {nom:.2f})")
            w(f"  {nt:<18}{s:>2}  {nom:>8.2f}{cell['requested_over_nominal']:>9.3f}"
              f"{cell['applied_over_requested']:>9.3f}{ach:>10.4f}{mark}")
    if short_txt:
        w("  MATERIALLY SHORT (>=5%): " + "; ".join(short_txt))
        w("  -> N2 don't-empty-a-token guard (unavoidable); documented in M1.3. "
          "N4 now hits nominal after the disjoint-pair rewrite.")
    w("")
    w("NORMALIZER SURVIVAL (CER original vs normalized-perturbed, severity 3)")
    if rep.normalizer_survival is None:
        w("  normalizer (csebuetnlp/normalizer) not installed -- skipped")
    else:
        for nt in NOISE_TYPES:
            v = rep.normalizer_survival[nt]
            w(f"  {nt:<20} cer={v['post_normalizer_cer']:.4f}  "
              f"touch_rate={v['touch_rate']:.1%}"
              + ("  <-- near zero CER" if v["post_normalizer_cer"] < 0.01 else ""))
        max_touch = max(v["touch_rate"] for v in rep.normalizer_survival.values())
        if max_touch < 0.05:
            w(f"  CAVEAT: touch_rate <5% everywhere (max {max_touch:.1%}) -- the "
              f"normalizer barely ran on this clean synthetic corpus; see the .md/.html "
              f"for the full caveat.")
    w("")
    w("EMOJI-RANGE COVERAGE (M0 Q9; provisional -- sample sentences only)")
    ec = rep.emoji_coverage
    if ec is None:
        w("  emoji package not installed -- skipped")
    else:
        w(f"  emoji version {ec['emoji_lib_version']}; {ec['n_occurrences']} occurrences")
        w(f"  coverage any={ec['coverage_any']:.1%}  full={ec['coverage_full']:.1%}"
          f"  false-positive codepoints={ec['n_false_positive_codepoints']}")
        if ec["uncaught"]:
            w(f"  uncaught: {' '.join(ec['uncaught'])}")
    w("")
    w("M3 ACCEPTANCE CRITERIA")
    w(f"  1. Determinism ................................... {_PASS[rep.determinism_ok]}")
    w(f"  2. Monotonicity ................................. {_PASS[rep.monotonic_ok]}"
      + ("" if rep.monotonic_ok else f"  ({', '.join(rep.monotonic_bad)})"))
    w(f"  3. Validity / no U+FFFD / no orphan nukta ........ {_PASS[rep.validity_ok]}")
    w(f"     matra survives NFC round trip ................. {_PASS[rep.nfc_ok]}")
    w(f"     matra survives BanglaBERT normalizer ......... {_PASS[rep.normalizer_matra_ok]}")
    w(f"  4. Non-identity at severity 5 (>=95%) ............ {_PASS[rep.nonidentity_ok]}")
    for nt, changed, elig, rate, ok in rep.nonidentity:
        w(f"       {nt:<20} {rate:6.1%} of {elig:5d}  {'ok' if ok else 'LOW'}")
    w("")
    w("=" * 78)
    w(f"OVERALL: {'ALL CHECKS PASS' if rep.overall_ok else 'SOME CHECKS FAILED'}")
    w("=" * 78)
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _write_utf8(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=_DEFAULT_MD,
                        help="path for the Markdown report (UTF-8)")
    parser.add_argument("--out-html", default=_DEFAULT_HTML,
                        help="path for the HTML report (UTF-8)")
    parser.add_argument("--quiet", action="store_true",
                        help="write files only; no console output")
    args = parser.parse_args(argv)

    rep = compute_report()

    _write_utf8(args.out, render_markdown(rep))
    _write_utf8(args.out_html, render_html(rep))

    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        print(render_text(rep))
    # Always report where the (authoritative, UTF-8) files went.
    print(f"[written] {os.path.relpath(args.out, _ROOT)}")
    print(f"[written] {os.path.relpath(args.out_html, _ROOT)}")

    if not rep.overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
