"""Shared plumbing for every pipeline stage script -- CLAUDE.md invariant 9:
every script prints progress to stderr, accepts ``--limit N``, and is
resumable (skip existing output unless ``--force``).

Also the fix for a real incident (2026-09-14): ``python pipeline/run_all.py
--stage all --limit 20`` wrote smoke output straight into the production
``outputs/``, ``data/final/`` and ``results/`` trees (no isolation), and the
resume check could then see those files and skip a later real run -- so a
full-scale run silently became a no-op on 20-row data. Two independent
guards fix this:

  1. **Isolation.** ``smoke_or_prod`` redirects every DEFAULT output/input
     root to ``<repo_root>/.smoke/<same relative path>`` whenever ``--limit``
     is active, mirroring the production tree exactly. An explicit
     ``--out-dir`` (etc.) the caller passes is still honoured -- this
     redirects defaults, not deliberate overrides -- but nothing defaults
     into a production path anymore while ``--limit`` is set. ``.smoke/`` is
     gitignored.
  2. **Provenance-checked resume.** ``write_run_meta``/``should_skip``: every
     stage records whether ITS OWN run was limited (and by how much) beside
     its output. A later full run does not trust file existence alone --
     if the existing output's metadata says it came from a limited run (or
     there is no metadata at all, e.g. predating this fix), the full run
     regenerates and logs why, rather than skipping faked-by-omission work.

Not part of ``noisebench/`` (uses no package-specific logic); imported by the
pipeline stage scripts and run_all.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN_META_NAME = ".run_meta.json"


class Progress:
    """Prints ``[stage] dataset: message (elapsed Xs)`` to stderr as a script
    runs, so a stall is visible within a minute without reading any output
    file. Never buffers -- ``flush=True`` on every line."""

    def __init__(self, stage: str):
        self.stage = stage
        self._t0 = time.perf_counter()

    def log(self, dataset: str, message: str) -> None:
        elapsed = time.perf_counter() - self._t0
        print(f"[{self.stage}] {dataset}: {message} ({elapsed:.1f}s)",
              file=sys.stderr, flush=True)

    def done(self, message: str = "all done") -> None:
        elapsed = time.perf_counter() - self._t0
        print(f"[{self.stage}] {message} ({elapsed:.1f}s total)",
              file=sys.stderr, flush=True)


def add_common_args(ap: argparse.ArgumentParser) -> None:
    """``--limit`` and ``--force``, with the standard help text -- add to
    every stage script's parser so the CLI contract is identical everywhere."""
    ap.add_argument("--limit", type=int, default=None,
                     help="process at most N rows per dataset (smoke run) -- "
                          "automatically isolates all default output/input "
                          "paths under .smoke/ instead of outputs/, "
                          "data/final/, results/ (never a production path)")
    ap.add_argument("--force", action="store_true",
                     help="recompute and overwrite outputs that already exist "
                          "(default: skip them, so a killed run can resume)")
    ap.add_argument("--orchestrated", action="store_true", help=argparse.SUPPRESS)
    # Set by run_all.py on every stage it invokes -- suppresses a stage's own
    # "Next: run X" hint, which is misleading when run_all.py is already
    # about to run X itself. Not meant for interactive use (hence SUPPRESS).


# ---------------------------------------------------------------------------
# 1. Isolation -- --limit can never default into a production path
# ---------------------------------------------------------------------------


def smoke_or_prod(limit: int | None, prod_path: str, root: str = _ROOT) -> str:
    """If `limit` is set, redirect `prod_path` (an absolute path under
    `root`) to the mirrored location under `root/.smoke/`, preserving the
    same relative structure. If `limit` is None, returns `prod_path`
    unchanged.

    Callers use this ONLY to compute a DEFAULT -- an explicit CLI path the
    user passed should never be run through this function, so a deliberate
    override is still honoured."""
    if limit is None:
        return prod_path
    rel = os.path.relpath(prod_path, root)
    return os.path.join(root, ".smoke", rel)


def print_smoke_banner(stage: str, limit: int, resolved: dict[str, str]) -> None:
    """Call once at the top of main() whenever args.limit is not None."""
    print(f"[{stage}] SMOKE RUN (--limit={limit}) -- writing to .smoke/, not "
         f"outputs/ / data/final/ / results/", file=sys.stderr, flush=True)
    for label, path in resolved.items():
        print(f"[{stage}]   {label}: {path}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# 2. Provenance-checked resume -- a limited run can never satisfy a full run
# ---------------------------------------------------------------------------


def write_run_meta(out_dir: str, limited: bool, limit: int | None, n_rows_processed) -> None:
    """Sidecar `.run_meta.json` beside a stage's output, recording whether
    THIS run was limited and how much it processed. Read back by
    `should_skip` so a later full run never trusts file-existence alone."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _RUN_META_NAME)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"limited": limited, "limit": limit,
                   "n_rows_processed": n_rows_processed}, fh, indent=2)
        fh.write("\n")


def read_run_meta(out_dir: str) -> dict | None:
    path = os.path.join(out_dir, _RUN_META_NAME)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def should_skip(out_dir: str, marker_paths: list[str], limit: int | None, force: bool,
               progress: Progress, label: str) -> bool:
    """Unified resumability check for every stage.

      * `force` -> never skip.
      * any `marker_paths` missing -> nothing to skip.
      * THIS run is limited (`limit is not None`) -> never skip. It writes
        under its own isolated `.smoke/` root (see `smoke_or_prod`), so
        re-running a smoke sample is cheap and should always reflect the
        current code, not a stale smoke artefact.
      * existing output's `.run_meta.json` says IT was limited, or is
        missing entirely (predates this fix / unknown provenance) -> do NOT
        skip; regenerate for real and say why. Resume only skips work that
        is genuinely done, never work that a smoke run faked.
      * otherwise (existing output is confirmed from a prior full run) ->
        skip, as before.
    """
    if force:
        return False
    if not marker_paths or not all(os.path.exists(p) for p in marker_paths):
        return False
    if limit is not None:
        return False
    meta = read_run_meta(out_dir)
    if meta is None:
        progress.log(label, f"existing output at {out_dir} has no run metadata "
                            "(predates the smoke-isolation fix, or was produced "
                            "some other way) -- cannot confirm it's a real full "
                            "run; regenerating rather than trusting it")
        return False
    if meta.get("limited"):
        progress.log(label, f"existing output at {out_dir} was produced by a "
                            f"LIMITED run (--limit={meta.get('limit')}) -- "
                            "regenerating for real, not skipping")
        return False
    progress.log(label, f"skipping -- {out_dir} has confirmed full-run output; "
                        "use --force to redo")
    return True
