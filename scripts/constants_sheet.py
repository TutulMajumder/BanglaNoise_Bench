"""Rendered-constants sheet for `noisebench/bangla_maps.py`.

Renders every linguistic constant as ACTUAL Bangla (glyph + code points +
Unicode names) so a native speaker can review it without reading Python, and
runs a block of structural assertions:

  * consonant / independent-vowel / digit pool sizes;
  * NO matra and NO hasanta anywhere in any `IN_CLASS_POOL` or in
    `INSERTABLE_CHARS`;
  * the pools are mutually disjoint; the derived maps are internally consistent;
  * every constant is NFC-idempotent.

Writes UTF-8 files itself (CLAUDE.md invariant 8 -- never shell redirection):

    python scripts/constants_sheet.py
        -> results/constants_sheet.md  and  results/constants_sheet.html

    python scripts/constants_sheet.py --out PATH.md --out-html PATH.html --quiet

Exit code is non-zero if any assertion fails.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import unicodedata

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from noisebench import bangla_maps as bm  # noqa: E402

_DEFAULT_MD = os.path.join(_ROOT, "outputs", "s0_noise_suite", "constants_sheet.md")
_DEFAULT_HTML = os.path.join(_ROOT, "outputs", "s0_noise_suite", "constants_sheet.html")


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def uname(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return "<unnamed>"


def cps(s: str) -> str:
    return " ".join(f"U+{ord(c):04X}" for c in s)


def names(s: str) -> str:
    return " + ".join(uname(c) for c in s)


def is_nfc(s: str) -> bool:
    return unicodedata.normalize("NFC", s) == s


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------

MATRA_SET = bm.MATRA_SET
HASANTA = bm.HASANTA


def _has_matra_or_hasanta(s: str) -> bool:
    return any(c in MATRA_SET or c == HASANTA for c in s)


def run_assertions() -> tuple[list[tuple[str, bool, str]], bool]:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        checks.append((name, bool(cond), detail))

    # --- pool sizes (report + sanity) --------------------------------------
    n_cons = len(bm.IN_CLASS_POOLS["consonant"])
    n_vow = len(bm.IN_CLASS_POOLS["independent_vowel"])
    n_dig = len(bm.IN_CLASS_POOLS["digit"])
    check("consonant pool size == 35", n_cons == 35, f"got {n_cons}")
    check("independent_vowel pool size == 11", n_vow == 11, f"got {n_vow}")
    check("digit pool size == 10", n_dig == 10, f"got {n_dig}")
    check(
        "IN_CLASS_POOLS match BANGLA_* inventories",
        bm.IN_CLASS_POOLS["consonant"] == bm.BANGLA_CONSONANTS
        and bm.IN_CLASS_POOLS["independent_vowel"] == bm.BANGLA_INDEPENDENT_VOWELS
        and bm.IN_CLASS_POOLS["digit"] == bm.BANGLA_DIGITS,
    )

    # --- NO matra / hasanta in any pool ----------------------------------
    pool_offenders = {
        cls: [ch for ch in chars if _has_matra_or_hasanta(ch)]
        for cls, chars in bm.IN_CLASS_POOLS.items()
    }
    bad = {k: v for k, v in pool_offenders.items() if v}
    check(
        "no matra / hasanta in any IN_CLASS_POOL",
        not bad,
        "" if not bad else f"offenders: {bad}",
    )

    ins_offenders = [ch for ch in bm.INSERTABLE_CHARS if _has_matra_or_hasanta(ch)]
    check(
        "no matra / hasanta in INSERTABLE_CHARS",
        not ins_offenders,
        "" if not ins_offenders else f"offenders: {[cps(c) for c in ins_offenders]}",
    )
    check(
        "no chandrabindu (U+0981) in INSERTABLE_CHARS",
        bm.CHANDRABINDU not in bm.INSERTABLE_CHARS,
    )
    check(
        "no digits in INSERTABLE_CHARS (removed 2026-09-01: no plausible "
        "mechanism for a digit-in-word insertion)",
        not any(d in bm.INSERTABLE_CHARS for d in bm.BANGLA_DIGITS),
    )

    # --- pools mutually disjoint ----------------------------------------
    seen: dict[str, str] = {}
    overlap: list[str] = []
    for cls, chars in bm.IN_CLASS_POOLS.items():
        for ch in chars:
            if ch in seen:
                overlap.append(f"{ch!r} in {seen[ch]} and {cls}")
            seen[ch] = cls
    check("IN_CLASS_POOLS are mutually disjoint", not overlap, "; ".join(overlap))

    # --- N3 never produces an identity swap possible? pool has >=2 members --
    check(
        "every IN_CLASS_POOL has >= 2 members (N3 can always pick a different char)",
        all(len(v) >= 2 for v in bm.IN_CLASS_POOLS.values()),
    )

    # --- homophone sets ------------------------------------------------
    homo_flat = [m for s in bm.HOMOPHONE_SETS for m in s]
    check(
        "HOMOPHONE_SETS members are distinct across sets",
        len(homo_flat) == len(set(homo_flat)),
    )
    check(
        "every HOMOPHONE set has >= 2 members",
        all(len(s) >= 2 for s in bm.HOMOPHONE_SETS),
    )
    check(
        "ি/ী and ু/ূ are present as HOMOPHONE sets (M0 Q3)",
        ("ি", "ী") in bm.HOMOPHONE_SETS and ("ু", "ূ") in bm.HOMOPHONE_SETS,
    )
    check(
        "ই/ঈ, উ/ঊ, র/ড়/ঢ় kept (author-confirmed)",
        ("ই", "ঈ") in bm.HOMOPHONE_SETS and ("উ", "ঊ") in bm.HOMOPHONE_SETS
        and ("র", "ড়", "ঢ়") in bm.HOMOPHONE_SETS,
    )
    check(
        "ব/ভ removed from HOMOPHONE_SETS (one of ten aspiration pairs; all-or-none)",
        ("ব", "ভ") not in bm.HOMOPHONE_SETS
        and ("ব", "ভ") in bm.HOMOPHONE_SETS_REJECTED,
    )

    # --- N6 targets / alter pairs -------------------------------------
    check(
        "N6_DROP_TARGETS == all matras + hasanta + chandrabindu",
        set(bm.N6_DROP_TARGETS) == set(bm.MATRA_SIGNS) | {HASANTA, bm.CHANDRABINDU},
    )
    alter_chars = {c for pair in bm.N6_ALTER_PAIRS for c in pair}
    check(
        "N6_ALTER_PAIRS chars are a subset of N6_DROP_TARGETS",
        alter_chars <= set(bm.N6_DROP_TARGETS),
    )
    check(
        "N6_ALTER_PAIRS == {ি<->ী, ু<->ূ} only (M0 Q5 revised)",
        set(map(frozenset, bm.N6_ALTER_PAIRS))
        == {frozenset(("ি", "ী")), frozenset(("ু", "ূ"))},
    )

    # --- N8 matra -> independent vowel map ---------------------------
    check(
        "MATRA_TO_INDEPENDENT_VOWEL keys are all dependent vowel signs",
        set(bm.MATRA_TO_INDEPENDENT_VOWEL) <= set(bm.MATRA_SIGNS),
    )
    check(
        "MATRA_TO_INDEPENDENT_VOWEL values are all independent vowels",
        set(bm.MATRA_TO_INDEPENDENT_VOWEL.values()) <= set(bm.BANGLA_INDEPENDENT_VOWELS),
    )
    check(
        "MATRA_TO_INDEPENDENT_VOWEL == {ু->উ, ূ->ঊ, ি->ই, ী->ঈ, ো->ও} only "
        "(restricted 2026-09-01)",
        bm.MATRA_TO_INDEPENDENT_VOWEL
        == {"ু": "উ", "ূ": "ঊ", "ি": "ই", "ী": "ঈ", "ো": "ও"},
    )

    # --- N5 rejected sets ------------------------------------------
    check(
        "HOMOPHONE_SETS_REJECTED includes the ten aspirated/unaspirated pairs",
        {frozenset(p) for p in bm.HOMOPHONE_SETS_REJECTED}
        >= {frozenset(p) for p in [
            ("ক", "খ"), ("গ", "ঘ"), ("চ", "ছ"), ("জ", "ঝ"), ("ট", "ঠ"),
            ("ড", "ঢ"), ("ত", "থ"), ("দ", "ধ"), ("প", "ফ"), ("ব", "ভ"),
        ]},
    )
    check(
        "HOMOPHONE_SETS and HOMOPHONE_SETS_REJECTED are disjoint",
        {frozenset(s) for s in bm.HOMOPHONE_SETS}.isdisjoint(
            {frozenset(s) for s in bm.HOMOPHONE_SETS_REJECTED}
        ),
    )

    # --- conjunct fixtures -----------------------------------------
    check(
        "every COMMON_CONJUNCTS entry is <consonant> HASANTA <consonant>",
        all(
            len(c) == 3 and c[1] == HASANTA
            and c[0] in {x for x in bm.CONSONANT_SET if len(x) == 1}
            and c[2] in {x for x in bm.CONSONANT_SET if len(x) == 1}
            for c in bm.COMMON_CONJUNCTS
        ),
    )

    # --- NFC idempotence of every string constant -----------------
    non_nfc: list[str] = []
    for group in (
        bm.BANGLA_CONSONANTS, bm.BANGLA_INDEPENDENT_VOWELS, bm.MATRA_SIGNS,
        bm.BANGLA_SIGNS, bm.BANGLA_DIGITS, bm.INSERTABLE_CHARS, bm.N6_DROP_TARGETS,
        bm.COMMON_CONJUNCTS, homo_flat,
    ):
        non_nfc += [s for s in group if not is_nfc(s)]
    check("every string constant is NFC-idempotent", not non_nfc,
          "" if not non_nfc else f"non-NFC: {[cps(s) for s in non_nfc]}")

    all_ok = all(ok for _, ok, _ in checks)
    return checks, all_ok


# ---------------------------------------------------------------------------
# section builders (return list of (title, header, rows))
# ---------------------------------------------------------------------------

Section = tuple[str, list[str], list[list[str]]]


def _char_rows(chars: tuple[str, ...]) -> list[list[str]]:
    return [[ch, cps(ch), names(ch)] for ch in chars]


def build_sections() -> list[Section]:
    S: list[Section] = []
    hdr = ["glyph", "code point(s)", "Unicode name(s)"]

    S.append(("Structural code points", hdr, _char_rows(
        (bm.HASANTA, bm.ZWNJ, bm.ZWJ, bm.CHANDRABINDU))))

    S.append((f"BANGLA_CONSONANTS ({len(bm.BANGLA_CONSONANTS)})", hdr,
              _char_rows(bm.BANGLA_CONSONANTS)))
    S.append((f"BANGLA_INDEPENDENT_VOWELS ({len(bm.BANGLA_INDEPENDENT_VOWELS)})", hdr,
              _char_rows(bm.BANGLA_INDEPENDENT_VOWELS)))
    S.append((f"MATRA_SIGNS ({len(bm.MATRA_SIGNS)})", hdr, _char_rows(bm.MATRA_SIGNS)))
    S.append((f"BANGLA_SIGNS ({len(bm.BANGLA_SIGNS)})", hdr, _char_rows(bm.BANGLA_SIGNS)))
    S.append((f"BANGLA_DIGITS ({len(bm.BANGLA_DIGITS)})", hdr, _char_rows(bm.BANGLA_DIGITS)))

    S.append((f"INSERTABLE_CHARS — N1 pool ({len(bm.INSERTABLE_CHARS)})",
              ["glyph", "code point(s)", "class"],
              [[ch, cps(ch), bm.CHAR_TO_CLASS.get(ch, "sign")]
               for ch in bm.INSERTABLE_CHARS]))

    S.append(("IN_CLASS_POOLS — N3 substitution pools",
              ["class", "size", "members"],
              [[cls, str(len(chars)), " ".join(chars)]
               for cls, chars in bm.IN_CLASS_POOLS.items()]))

    S.append((f"HOMOPHONE_SETS — N5 ({len(bm.HOMOPHONE_SETS)} sets)",
              ["#", "members", "code points", "note"],
              [[str(i + 1), " / ".join(s), " , ".join(cps(m) for m in s),
                "dependent signs" if all(len(m) == 1 and m in MATRA_SET for m in s)
                else ("nukta letters" if any(len(m) == 2 for m in s) else "letters")]
               for i, s in enumerate(bm.HOMOPHONE_SETS)]))

    S.append((f"HOMOPHONE_SETS_REJECTED — considered and rejected "
              f"({len(bm.HOMOPHONE_SETS_REJECTED)})",
              ["members", "code points"],
              [[" / ".join(s), " , ".join(cps(m) for m in s)]
               for s in bm.HOMOPHONE_SETS_REJECTED]))

    S.append((f"N6_DROP_TARGETS — N6 drop set ({len(bm.N6_DROP_TARGETS)})", hdr,
              _char_rows(bm.N6_DROP_TARGETS)))
    S.append(("N6_ALTER_PAIRS — N6 may alter (else drop-only)",
              ["a", "b", "code points"],
              [[a, b, f"{cps(a)} <-> {cps(b)}"] for a, b in bm.N6_ALTER_PAIRS]))

    S.append(("MATRA_TO_INDEPENDENT_VOWEL — N8 insert_vowel map",
              ["matra", "-> independent vowel", "code points"],
              [[k, v, f"{cps(k)} -> {cps(v)}"]
               for k, v in bm.MATRA_TO_INDEPENDENT_VOWEL.items()]))

    S.append((f"COMMON_CONJUNCTS — N7 fixtures ({len(bm.COMMON_CONJUNCTS)})",
              ["glyph", "components", "code points"],
              [[c, f"{c[0]} + ্ + {c[2]}", cps(c)] for c in bm.COMMON_CONJUNCTS]))
    S.append((f"CANDIDATE_CONJUNCTS_TRIPLE — excluded from N7 "
              f"({len(bm.CANDIDATE_CONJUNCTS_TRIPLE)})",
              ["glyph", "code points"],
              [[c, cps(c)] for c in bm.CANDIDATE_CONJUNCTS_TRIPLE]))

    S.append(("EMOJI_CODEPOINT_RANGES — N10 detection",
              ["range", "size", "sample"],
              [[f"U+{lo:04X}–U+{hi:04X}", str(hi - lo + 1),
                next((uname(chr(cp)) for cp in range(lo, hi + 1)
                      if uname(chr(cp)) != "<unnamed>"), "-")]
               for lo, hi in bm.EMOJI_CODEPOINT_RANGES]))
    S.append(("EMOJI_SEQUENCE_GLUE — stripped with emoji in 'remove'", hdr,
              _char_rows(bm.EMOJI_SEQUENCE_GLUE)))

    S.append(("Derived lookups (sizes)",
              ["name", "size", "note"],
              [["CONSONANT_SET", str(len(bm.CONSONANT_SET)), ""],
               ["INDEPENDENT_VOWEL_SET", str(len(bm.INDEPENDENT_VOWEL_SET)), ""],
               ["MATRA_SET", str(len(bm.MATRA_SET)), ""],
               ["DIGIT_SET", str(len(bm.DIGIT_SET)), ""],
               ["CHAR_TO_CLASS", str(len(bm.CHAR_TO_CLASS)),
                "multi-codepoint keys: "
                + " ".join(k for k in bm.CHAR_TO_CLASS if len(k) > 1)],
               ["COMBINING_MARKS", str(len(bm.COMBINING_MARKS)), ""]]))
    return S


# ---------------------------------------------------------------------------
# renderers
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "Rendered from `noisebench/bangla_maps.py` by `scripts/constants_sheet.py`. "
    "Every `# TODO(verify):` block in that file still needs a native-speaker "
    "pass; this sheet only makes the values readable and checks their structure."
)


def _md_cell(s: str) -> str:
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def render_markdown(sections: list[Section], checks, all_ok: bool) -> str:
    L = ["# bangla-noisebench — rendered constants sheet", "", _PREAMBLE, "",
         f"**Structural assertions: {'ALL PASS' if all_ok else 'SOME FAILED'}**", ""]
    L.append("## Assertions")
    L.append("")
    L.append("| check | result | detail |")
    L.append("|-------|--------|--------|")
    for name, ok, detail in checks:
        L.append(f"| {_md_cell(name)} | {'PASS' if ok else '**FAIL**'} | {_md_cell(detail)} |")
    L.append("")
    for title, header, rows in sections:
        L.append(f"## {title}")
        L.append("")
        L.append("| " + " | ".join(header) + " |")
        L.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows:
            L.append("| " + " | ".join(_md_cell(c) for c in row) + " |")
        L.append("")
    return "\n".join(L) + "\n"


_HTML_HEAD = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>bangla-noisebench — rendered constants sheet</title>
<style>
  body { font-family: "Noto Sans Bengali", "Nirmala UI", "Kalpurush",
         system-ui, -apple-system, "Segoe UI", sans-serif;
         line-height: 1.5; margin: 2rem auto; max-width: 1100px; padding: 0 1rem;
         color: #1a1a1a; }
  h1 { font-size: 1.5rem; }
  h2 { font-size: 1.1rem; margin-top: 1.8rem; border-bottom: 1px solid #ddd; }
  table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .9rem; }
  th, td { border: 1px solid #ccc; padding: .3rem .5rem; text-align: left;
           vertical-align: top; }
  th { background: #f2f2f2; }
  td.g { font-size: 1.3rem; }
  .pass { color: #0a7d28; font-weight: 600; }
  .fail { color: #b00020; font-weight: 600; }
  code { background: #f4f4f4; padding: .05rem .3rem; border-radius: 3px; }
</style>
</head>
<body>
"""


def render_html(sections: list[Section], checks, all_ok: bool) -> str:
    P = [_HTML_HEAD, "<h1>bangla-noisebench — rendered constants sheet</h1>",
         f"<p>{html.escape(_PREAMBLE)}</p>"]
    cls = "pass" if all_ok else "fail"
    P.append(f'<p><strong>Structural assertions: '
             f'<span class="{cls}">{"ALL PASS" if all_ok else "SOME FAILED"}</span>'
             f'</strong></p>')
    P.append("<h2>Assertions</h2>")
    P.append("<table><thead><tr><th>check</th><th>result</th><th>detail</th></tr>"
             "</thead><tbody>")
    for name, ok, detail in checks:
        v = f'<span class="{"pass" if ok else "fail"}">{"PASS" if ok else "FAIL"}</span>'
        P.append(f"<tr><td>{html.escape(name)}</td><td>{v}</td>"
                 f"<td>{html.escape(detail)}</td></tr>")
    P.append("</tbody></table>")
    for title, header, rows in sections:
        P.append(f"<h2>{html.escape(title)}</h2>")
        P.append("<table><thead><tr>"
                 + "".join(f"<th>{html.escape(h)}</th>" for h in header)
                 + "</tr></thead><tbody>")
        for row in rows:
            tds = []
            for i, c in enumerate(row):
                klass = ' class="g"' if header[i] == "glyph" else ""
                tds.append(f"<td{klass}>{html.escape(c)}</td>")
            P.append("<tr>" + "".join(tds) + "</tr>")
        P.append("</tbody></table>")
    P.append("</body></html>\n")
    return "\n".join(P)


def render_text(checks, all_ok: bool) -> str:
    out = ["bangla-noisebench constants sheet — assertion summary", "-" * 60]
    for name, ok, detail in checks:
        out.append(f"  [{'PASS' if ok else 'FAIL'}] {name}"
                   + (f"   ({detail})" if detail else ""))
    out.append("-" * 60)
    out.append(f"  {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _write_utf8(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(content)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=_DEFAULT_MD)
    parser.add_argument("--out-html", default=_DEFAULT_HTML)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    sections = build_sections()
    checks, all_ok = run_assertions()

    _write_utf8(args.out, render_markdown(sections, checks, all_ok))
    _write_utf8(args.out_html, render_html(sections, checks, all_ok))

    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        print(render_text(checks, all_ok))
    print(f"[written] {os.path.relpath(args.out, _ROOT)}")
    print(f"[written] {os.path.relpath(args.out_html, _ROOT)}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
