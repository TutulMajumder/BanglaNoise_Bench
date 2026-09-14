"""Phase 2 pipeline orchestrator -- runs s1 through s5 in order.

    python pipeline/run_all.py --stage all --data-root data/raw --limit 20   # smoke
    python pipeline/run_all.py --stage all --data-root data/raw              # real run
    python pipeline/run_all.py --stage 1                                     # just s1
    python pipeline/run_all.py --stage 1,2                                   # s1 then s2

Runs each stage's ``main()`` in-process (not a subprocess) so a failure in
stage N stops the run before stage N+1, and stderr progress lines from every
stage interleave in one stream. CLAUDE.md invariant 9 applies transitively:
every stage is independently resumable, so re-running ``run_all.py`` after a
partial failure skips whatever already completed -- but ONLY work confirmed
to be from a real (non-``--limit``) run; see ``pipeline/_common.py``.

**``--limit`` isolates itself.** Every stage redirects its default output
(and, for s3/s5, input) paths under ``.smoke/`` instead of ``outputs/``,
``data/final/`` and ``results/`` whenever ``--limit`` is set -- a smoke run
can never write, or be mistaken by a later full run for, real output. This
followed a real incident: an earlier version wrote ``--limit 20`` output
straight into production paths, and a later full run's resume check then
skipped everything because *something* existed there.

s1 is deliberately not gated on s2 completing -- inspect the
raw corpora, look at the report, THEN decide whether to run preparation. Do
not run ``--stage all`` unattended the first time on a new machine; run 1,
look at outputs/s1_raw_inspection/report.md, then continue.
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline import s1_inspect_raw, s2_prepare, s3_describe, s4_measure, s5_figures  # noqa: E402

STAGES = {
    "1": ("s1_inspect_raw", s1_inspect_raw),
    "2": ("s2_prepare", s2_prepare),
    "3": ("s3_describe", s3_describe),
    "4": ("s4_measure", s4_measure),
    "5": ("s5_figures", s5_figures),
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    help="comma list of stages to run (1-5), or 'all'")
    ap.add_argument("--data-root", default=os.path.join(_ROOT, "data", "raw"),
                    help="passed to s1/s2 as --data-root")
    ap.add_argument("--limit", type=int, default=None, help="smoke run, passed to every stage")
    ap.add_argument("--force", action="store_true", help="passed to every stage")
    args, extra = ap.parse_known_args(argv)

    stages = list(STAGES) if args.stage == "all" else args.stage.split(",")
    for s in stages:
        if s not in STAGES:
            print(f"unknown stage {s!r}; choose from {list(STAGES)} or 'all'", file=sys.stderr)
            return 2

    for s in stages:
        name, module = STAGES[s]
        print(f"\n=== run_all: stage {s} ({name}) ===", file=sys.stderr)
        stage_argv: list[str] = list(extra)
        if args.limit is not None:
            stage_argv += ["--limit", str(args.limit)]
        if args.force:
            stage_argv += ["--force"]
        if s in ("1", "2"):
            stage_argv += ["--data-root", args.data_root]
        # Suppresses each stage's own "Next: run X" hint -- misleading when
        # run_all.py is already about to run X itself (this is why s2 used
        # to print "Next: python pipeline/s3_describe.py" mid--stage-all-run).
        stage_argv += ["--orchestrated"]
        rc = module.main(stage_argv)
        if rc != 0:
            print(f"\nrun_all: stage {s} ({name}) exited {rc}; stopping.", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
