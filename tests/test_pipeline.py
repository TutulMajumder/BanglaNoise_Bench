"""Smoke test for the Phase 2 pipeline (pipeline/s2_prepare.py + s3_describe.py
+ s4_measure.py). Builds tiny fixture CSVs from the existing Bangla sample
sentences -- no invented text -- in a temp dir, runs the pipeline end to end,
and checks the structural invariants METHODOLOGY M2 / the Phase 2 acceptance
list require.

Replaces tests/test_phase2_pipeline.py (pre-restructure; scripts/prepare_data.py
etc. no longer exist). This does NOT exercise the real datasets (not on
disk); it proves the code runs, manifests reload, and split ratios are right.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# `raw_root` fixture moved to tests/conftest.py (shared with test_smoke_isolation.py)

s2 = importlib.import_module("pipeline.s2_prepare")


def test_prepare_all_three(raw_root, tmp_path):
    final_root = tmp_path / "final"
    stats = {name: s2.prepare_one(name, str(raw_root), str(final_root))
             for name in s2.DATASETS}

    for name, st in stats.items():
        assert st.n_final > 0
        assert st.dupes_removed >= 0 and st.short_removed >= 0
        spec = s2.DATASETS[name]
        out_dir = s2.dataset_final_dir(spec, str(final_root))
        manifest = json.load(open(os.path.join(out_dir, "manifest.json"), encoding="utf-8"))
        all_ids_from_csv: set[str] = set()
        for split in ("train", "val", "test"):
            df = pd.read_csv(os.path.join(out_dir, f"{split}.csv"), dtype=str, keep_default_na=False)
            assert list(df.columns) == ["id", "text", "label"]
            assert set(df["id"]) == set(manifest["ids"][split])
            all_ids_from_csv |= set(df["id"])
        assert len(all_ids_from_csv) == st.n_final
        # no id in two splits
        assert not (set(manifest["ids"]["train"]) & set(manifest["ids"]["val"]))
        assert not (set(manifest["ids"]["val"]) & set(manifest["ids"]["test"]))
        # source SHA-256 recorded for every source file
        assert set(manifest["source_sha256"]) == set(st.source_files)

    assert set(s2.DATASETS) == {"sentnob", "bd_shs", "banfakenews"}  # banfakenews_full cut

    # sentnob keeps its provided splits + carries a dual (raw/final-text)
    # leakage block, both on the same final row population
    assert stats["sentnob"].split_source == "dataset-provided"
    manifest = json.load(open(final_root / "sentnob" / "manifest.json", encoding="utf-8"))
    leakage = manifest["leakage"]
    assert set(leakage) == {"raw_text", "final_text", "pct_of_split"}
    pairs = {"test_in_train", "val_in_train", "val_in_test"}
    assert set(leakage["raw_text"]) == pairs
    assert set(leakage["final_text"]) == pairs
    assert set(leakage["pct_of_split"]) == pairs
    for label in pairs:
        p = leakage["pct_of_split"][label]
        assert set(p) == {"raw_text_pct", "final_text_pct", "split_size"}
        # preprocess_text is a deterministic function of raw text, so two
        # raw-identical rows are always final-identical too -- every
        # raw-text-leaked id must also appear in final-text-leaked ids.
        # Masking can only ADD apparent leakage (merge distinct raw texts
        # into the same final text), never remove real leakage.
        assert set(leakage["raw_text"][label]) <= set(leakage["final_text"][label])

    # bd_shs / banfakenews are stratified ~70/10/20
    for name in ("bd_shs", "banfakenews"):
        ss = stats[name].split_sizes
        n = sum(ss.values())
        assert 0.60 <= ss["train"] / n <= 0.80
        assert 0.03 <= ss["val"] / n <= 0.18
        assert 0.12 <= ss["test"] / n <= 0.30
        assert stats[name].split_source == "stratified-70/10/20"


def test_split_is_deterministic(raw_root, tmp_path):
    a = s2.prepare_one("bd_shs", str(raw_root), str(tmp_path / "a"))
    b = s2.prepare_one("bd_shs", str(raw_root), str(tmp_path / "b"))
    manifest_a = json.load(open(tmp_path / "a" / "bd_shs" / "manifest.json", encoding="utf-8"))
    manifest_b = json.load(open(tmp_path / "b" / "bd_shs" / "manifest.json", encoding="utf-8"))
    assert manifest_a["ids"] == manifest_b["ids"]
    assert a.split_sizes == b.split_sizes


def test_s1_s2_agree_on_raw_class_distribution(raw_root):
    """Regression test for the 2026-09-14 bug: s1 reported BanFakeNews as
    100% authentic (it re-derived a source file's `src` via a lookup that
    was trivially True for every BanFakeNews source, since both have
    `split=None`, so `next()` always returned the first one -- label_const=1
    -- regardless of which file was actually being read). Both stages now
    call the same `resolve_labels` function; this asserts they can never
    silently disagree again, for every registered dataset, not just
    BanFakeNews."""
    from noisebench import corpus_stats as cs
    s1 = importlib.import_module("pipeline.s1_inspect_raw")
    progress = s1.Progress("test")

    for name, spec in s2.DATASETS.items():
        r1 = s1.inspect_dataset(name, spec, str(raw_root), None, progress)
        raw_df, _rep = s2.load_raw(spec, str(raw_root))
        s2_raw_dist = cs.class_distribution(raw_df["label"].tolist())
        assert r1["class_distribution_overall"] == s2_raw_dist, (
            f"{name}: s1={r1['class_distribution_overall']} vs "
            f"s2={s2_raw_dist} -- label resolution has drifted apart again"
        )

    # BanFakeNews specifically: the fixture is 32 authentic / 16 fake
    # (asymmetric on purpose -- a label swap changes the COUNT, not just
    # the name, so this can't pass by accident).
    bf_spec = s2.DATASETS["banfakenews"]
    r1 = s1.inspect_dataset("banfakenews", bf_spec, str(raw_root), None, progress)
    assert r1["class_distribution_overall"] == {"1": 32, "0": 16}


def test_preprocess_order():
    # URL and mention replaced (not removed), whitespace collapsed, no lowercasing
    out = s2.preprocess_text("  See   https://a.b/c?d=1  @Ripon   ABC  ")
    assert out == "See <URL> <USER> ABC"
    # < 3 clusters filtered downstream, not here
    assert s2.n_clusters("কি") == 1


def test_resumability_skips_without_force(raw_root, tmp_path):
    final_root = tmp_path / "final"
    st1 = s2.prepare_one("sentnob", str(raw_root), str(final_root))
    assert st1 is not None
    st2 = s2.prepare_one("sentnob", str(raw_root), str(final_root))
    assert st2 is None  # skipped -- outputs already exist
    st3 = s2.prepare_one("sentnob", str(raw_root), str(final_root), force=True)
    assert st3 is not None


def test_describe_and_measure_run(raw_root, tmp_path):
    final_root = tmp_path / "final"
    all_stats = {name: s2.prepare_one(name, str(raw_root), str(final_root))
                for name in s2.DATASETS}
    s2_out = tmp_path / "s2_out"
    s2.write_provenance(all_stats, str(s2_out / "provenance.json"))

    s3 = importlib.import_module("pipeline.s3_describe")
    results = {}
    for name in s2.DATASETS:
        results[name] = s3.describe_dataset(name, s2.DATASETS[name], str(final_root), None,
                                            s3.Progress("test"))
    assert all(r["n_total_rows"] > 0 for r in results.values())

    provenance = json.load(open(s2_out / "provenance.json", encoding="utf-8"))
    md = s3.render_table2_markdown(results, provenance, bdshs_fffd=None)
    assert "Table II" in md and "N10" in md

    s4 = importlib.import_module("pipeline.s4_measure")
    samples = s4.load_samples(str(final_root), k=20)
    assert "ALL" in samples
    n5 = s4.measure_n5(samples, seeds=(42,))
    assert set(n5["under_review"]) == {"ই/ঈ", "উ/ঊ"}
    em = s4.measure_emoji(samples)
    assert "per_dataset" in em
    cer = s4.measure_cer({k: v for k, v in samples.items() if k != "ALL"}, seeds=(42,))
    assert set(cer["real"]) == {"sentnob", "bd_shs", "banfakenews"}
    for tbl in cer["real"].values():
        assert all(len(tbl[nt]) == 5 for nt in tbl)
