"""Publication hygiene check (CLAUDE.md invariant 10): count codepoints in
the Bengali Unicode block (U+0980-U+09FF) across every tracked file under
``outputs/``, and report the total. Run this after any real pipeline run,
before committing.

Expected: near-zero, with exactly two named, DOCUMENTED exceptions -- not a
weakened range check, and not a magic path string dropped in without
explanation:

  1. Deliberate glyph labels inside Phase 2 tables/figures (homophone-set
     names like শ/ষ/স, matra pairs like ি/ী) -- sourced from
     ``noisebench.bangla_maps.HOMOPHONE_SETS``, printed by
     ``pipeline/s4_measure.py`` and ``pipeline/s5_figures.py``. A handful of
     occurrences per table/figure; counted, not filtered out, so a sudden
     jump is still visible.
  2. ``outputs/s0_noise_suite/`` -- Phase 1's noise-suite validation report
     and constants sheet, built entirely from ``tests/bangla_samples.py``:
     sentences WE authored to exercise the perturbation code (conjuncts,
     matras, nukta letters ড়/ঢ়/য়, homophone-prone letters, emoji), never
     downloaded from any corpus. Verified empirically, not assumed: all 58
     fixture sentences checked for exact-substring matches against all
     124,487 raw rows across the three real corpora -- 0 matches (2026-09-14,
     see PHASE2_STATUS.md). Allowlisted by path prefix below.

Any Bengali-block codepoints found in a file OUTSIDE these two exceptions
must be treated as a likely corpus-text leak and investigated before
committing -- this script does not decide that for you, it just counts and
shows you where.

Deliberately standalone (not wired into pipeline/run_all.py): this is a
manual gate you run before committing, not a data-producing pipeline stage.

    python pipeline/verify_hygiene.py
    python pipeline/verify_hygiene.py --dir outputs
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Path prefixes (relative to the scanned root) where Bengali-block
#: codepoints are expected and allowlisted, WITH a reason -- see module
#: docstring for the full justification of each.
ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("s0_noise_suite/", "Phase 1: authored fixture sentences (tests/bangla_samples.py), "
                        "verified empirically to have 0 exact-substring matches against "
                        "any of the three raw corpora -- not corpus text."),
)

_SCAN_EXTS = (".md", ".html", ".csv", ".json")


def scan(root: str) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """Returns (allowlisted_hits, unexplained_hits), each a list of
    (relative_path, codepoint_count)."""
    allowlisted: list[tuple[str, int]] = []
    unexplained: list[tuple[str, int]] = []
    paths = []
    for ext in _SCAN_EXTS:
        paths += glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)
    for path in sorted(paths):
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        n = sum(1 for ch in text if "ঀ" <= ch <= "৿")
        if n == 0:
            continue
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        reason = next((why for prefix, why in ALLOWLIST if rel.startswith(prefix)), None)
        if reason is not None:
            allowlisted.append((rel, n))
        else:
            unexplained.append((rel, n))
    return allowlisted, unexplained


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(_ROOT, "outputs"))
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass

    allowlisted, unexplained = scan(args.dir)

    print("Allowlisted (expected) Bengali-block codepoints:")
    if allowlisted:
        for rel, n in allowlisted:
            reason = next(why for prefix, why in ALLOWLIST if rel.startswith(prefix))
            print(f"  {n:>6}  {rel}   [{reason.split('.')[0]}...]")
    else:
        print("  (none found)")
    total_allowlisted = sum(n for _, n in allowlisted)

    print("\nUNEXPLAINED Bengali-block codepoints (investigate before committing):")
    if unexplained:
        for rel, n in unexplained:
            print(f"  {n:>6}  {rel}")
    else:
        print("  (none -- clean)")
    total_unexplained = sum(n for _, n in unexplained)

    print(f"\nTotal allowlisted: {total_allowlisted}   Total UNEXPLAINED: {total_unexplained}")
    return 1 if total_unexplained else 0


if __name__ == "__main__":
    raise SystemExit(main())
