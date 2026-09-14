"""Regression tests for the 2026-09-14 incident: ``python pipeline/run_all.py
--stage all --limit 20`` wrote smoke output straight into the production
``outputs/``, ``data/final/`` and ``results/`` trees, and a later full run's
resume check then skipped everything because *something* existed there.

Two things are tested:

  1. Isolation: a ``--limit`` run, using ONLY its default paths (no explicit
     ``--out-dir`` etc.), must never write under the real repo's
     ``outputs/``, ``data/final/`` or ``results/`` -- it must land under
     ``.smoke/`` instead. Verified against the REAL repo root (not a fake
     one), since that's the exact thing that broke; the real ``outputs/``
     etc. are snapshotted before and after and asserted unchanged, and
     ``.smoke/`` is cleaned up afterwards (it's gitignored scratch either way).
  2. Provenance-checked resume: an existing output with no run metadata (or
     metadata saying it came from a limited run) must never be trusted by a
     later full (non-``--limit``) run -- it must regenerate, not skip. Uses
     an explicit, disposable ``tmp_path`` output directory (not the real
     repo) so this never depends on or risks the real corpora.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

common = importlib.import_module("pipeline._common")
s1 = importlib.import_module("pipeline.s1_inspect_raw")

_REAL_PRODUCTION_DIRS = [
    os.path.join(_ROOT, "outputs"),
    os.path.join(_ROOT, "data", "final"),
    os.path.join(_ROOT, "results"),
]
_SMOKE_ROOT = os.path.join(_ROOT, ".smoke")


def _snapshot(paths: list[str]) -> dict[str, float]:
    """{relpath: mtime} for every file under any of `paths` that exists."""
    snap: dict[str, float] = {}
    for root in paths:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for fn in files:
                p = os.path.join(dirpath, fn)
                snap[os.path.relpath(p, _ROOT)] = os.path.getmtime(p)
    return snap


# ---------------------------------------------------------------------------
# 1. smoke_or_prod / print_smoke_banner -- unit level
# ---------------------------------------------------------------------------


def test_smoke_or_prod_unaffected_when_no_limit():
    prod = os.path.join(_ROOT, "outputs", "s1_raw_inspection")
    assert common.smoke_or_prod(None, prod) == prod


def test_smoke_or_prod_redirects_under_limit():
    prod = os.path.join(_ROOT, "outputs", "s1_raw_inspection")
    redirected = common.smoke_or_prod(20, prod)
    assert redirected == os.path.join(_ROOT, ".smoke", "outputs", "s1_raw_inspection")
    assert redirected != prod


def test_smoke_or_prod_never_returns_a_production_path_for_any_limit():
    prod_paths = [
        os.path.join(_ROOT, "outputs", "s1_raw_inspection"),
        os.path.join(_ROOT, "data", "final"),
        os.path.join(_ROOT, "results"),
    ]
    for limit in (1, 20, 300, 999999):
        for prod in prod_paths:
            redirected = common.smoke_or_prod(limit, prod)
            assert redirected.startswith(os.path.join(_ROOT, ".smoke")), (
                f"limit={limit} prod={prod} redirected to {redirected} -- "
                "a limited run must never resolve to a production path"
            )


# ---------------------------------------------------------------------------
# 2. should_skip -- unit level, every branch
# ---------------------------------------------------------------------------


class _FakeProgress:
    def __init__(self):
        self.messages: list[str] = []

    def log(self, dataset, message):
        self.messages.append(f"{dataset}: {message}")


def test_should_skip_force_never_skips(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"
    marker.write_text("x", encoding="utf-8")
    common.write_run_meta(str(out_dir), limited=False, limit=None, n_rows_processed=100)
    assert common.should_skip(str(out_dir), [str(marker)], None, True,
                              _FakeProgress(), "test") is False


def test_should_skip_missing_marker_never_skips(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"  # never created
    assert common.should_skip(str(out_dir), [str(marker)], None, False,
                              _FakeProgress(), "test") is False


def test_should_skip_current_run_limited_never_skips(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"
    marker.write_text("x", encoding="utf-8")
    common.write_run_meta(str(out_dir), limited=False, limit=None, n_rows_processed=100)
    # Even though this looks like confirmed full-run output, the CURRENT
    # invocation is itself limited -- it writes to its own isolated .smoke/
    # root and must always regenerate there.
    assert common.should_skip(str(out_dir), [str(marker)], 20, False,
                              _FakeProgress(), "test") is False


def test_should_skip_no_metadata_does_not_skip(tmp_path):
    """The exact incident: existing output predates this fix (no
    .run_meta.json at all) -- must not be trusted."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"
    marker.write_text("x", encoding="utf-8")
    progress = _FakeProgress()
    assert common.should_skip(str(out_dir), [str(marker)], None, False,
                              progress, "test") is False
    assert any("no run metadata" in m for m in progress.messages)


def test_should_skip_limited_metadata_does_not_skip(tmp_path):
    """Existing output IS marked as having come from a --limit run -- a
    later full run must regenerate, not skip."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"
    marker.write_text("x", encoding="utf-8")
    common.write_run_meta(str(out_dir), limited=True, limit=20, n_rows_processed=20)
    progress = _FakeProgress()
    assert common.should_skip(str(out_dir), [str(marker)], None, False,
                              progress, "test") is False
    assert any("LIMITED run" in m for m in progress.messages)


def test_should_skip_confirmed_full_run_skips(tmp_path):
    """Only a confirmed full-run output, with no --force and a non-limited
    current invocation, actually gets skipped."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker = out_dir / "marker.csv"
    marker.write_text("x", encoding="utf-8")
    common.write_run_meta(str(out_dir), limited=False, limit=None, n_rows_processed=12575)
    assert common.should_skip(str(out_dir), [str(marker)], None, False,
                              _FakeProgress(), "test") is True


# ---------------------------------------------------------------------------
# 3. Integration: --limit, using DEFAULT paths, never touches the real repo
# ---------------------------------------------------------------------------


def test_limit_run_with_default_paths_never_touches_production(raw_root):
    before = {p: _snapshot([p]) for p in _REAL_PRODUCTION_DIRS}
    if os.path.isdir(_SMOKE_ROOT):
        shutil.rmtree(_SMOKE_ROOT)  # start clean so this test is self-contained
    try:
        rc = s1.main(["--data-root", str(raw_root), "--limit", "5"])
        assert rc == 0

        for p in _REAL_PRODUCTION_DIRS:
            after = _snapshot([p])
            assert after == before[p], (
                f"a --limit run touched production path {p} -- "
                "isolation regression (the exact 2026-09-14 incident)"
            )

        smoke_out = os.path.join(_SMOKE_ROOT, "outputs", "s1_raw_inspection")
        assert os.path.exists(os.path.join(smoke_out, "tables", "file_inventory.csv")), (
            ".smoke/ mirror was not created -- --limit produced no output at all"
        )
        meta = common.read_run_meta(smoke_out)
        assert meta is not None and meta["limited"] is True and meta["limit"] == 5
    finally:
        if os.path.isdir(_SMOKE_ROOT):
            shutil.rmtree(_SMOKE_ROOT)


# ---------------------------------------------------------------------------
# 4. Integration: a full run never skips stale/limited-origin output
# ---------------------------------------------------------------------------


def test_full_run_regenerates_stale_limited_output_instead_of_skipping(raw_root, tmp_path):
    """Simulates the exact failure mode: `out_dir` already has output that
    looks done (the marker file exists) but was produced by a limited run.
    A later full run against the SAME (explicit, disposable) out_dir must
    not skip it."""
    fake_prod = tmp_path / "fake_outputs" / "s1_raw_inspection"
    fake_prod.mkdir(parents=True)
    (fake_prod / "tables").mkdir()
    stale_marker = fake_prod / "tables" / "file_inventory.csv"
    stale_marker.write_text("dataset,file,sha256,n_rows\nFAKE,fake.csv,deadbeef,999999\n",
                            encoding="utf-8")
    common.write_run_meta(str(fake_prod), limited=True, limit=20, n_rows_processed={"fake": 20})

    rc = s1.main(["--data-root", str(raw_root), "--out-dir", str(fake_prod)])
    assert rc == 0

    # Regenerated for real: the fake placeholder row is gone, replaced by
    # the actual fixture data's file inventory.
    content = stale_marker.read_text(encoding="utf-8")
    assert "FAKE" not in content
    assert "sentnob" in content

    meta = common.read_run_meta(str(fake_prod))
    assert meta is not None and meta["limited"] is False and meta["limit"] is None


def test_full_run_skips_confirmed_full_output(raw_root, tmp_path):
    """Sanity check on the other side: a genuinely confirmed full-run output
    IS skipped on a second full run (without --force)."""
    out_dir = tmp_path / "fake_outputs" / "s1_raw_inspection"
    rc1 = s1.main(["--data-root", str(raw_root), "--out-dir", str(out_dir)])
    assert rc1 == 0
    marker = out_dir / "tables" / "file_inventory.csv"
    mtime_1 = marker.stat().st_mtime

    rc2 = s1.main(["--data-root", str(raw_root), "--out-dir", str(out_dir)])
    assert rc2 == 0
    assert marker.stat().st_mtime == mtime_1  # untouched -- skipped, as expected
