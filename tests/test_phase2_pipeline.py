"""Smoke test for the Phase 2 data pipeline (data/prepare.py + the two report
scripts). Builds tiny fixture CSVs from the existing Bangla sample sentences --
no invented text -- in a temp dir, runs the pipeline end to end, and checks the
structural invariants METHODOLOGY M2 / the Phase 2 acceptance list require.

This does NOT exercise the real datasets (not on disk); it proves the code
runs, the manifests reload, and the split ratios are right.
"""

from __future__ import annotations

import csv
import importlib
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.bangla_samples import SENTENCES

prepare = importlib.import_module("data.prepare")


def _rows(n: int) -> list[str]:
    base = list(dict.fromkeys(SENTENCES))  # de-dupe, keep order
    assert len(base) >= 20, "need a few dozen distinct sample sentences"
    return (base * ((n // len(base)) + 1))[:n]


@pytest.fixture()
def raw_root(tmp_path):
    """Lay out fixture CSVs matching each DATASETS spec's expected schema."""
    texts = _rows(48)

    # sentnob: ships train/val/test, columns Data,Label (0/1/2)
    sdir = tmp_path / "sentnob"
    sdir.mkdir()
    for split, chunk in (("train", texts[:30]), ("val", texts[30:36]), ("test", texts[36:48])):
        with open(sdir / f"{split}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["Data", "Label"])
            for i, t in enumerate(chunk):
                w.writerow([t, i % 3])

    # bd_shs: single csv, columns sentence,hate (0/1)
    bdir = tmp_path / "bd_shs"
    bdir.mkdir()
    with open(bdir / "bdshs.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence", "hate"])
        for i, t in enumerate(texts):
            w.writerow([t, i % 2])

    # banfakenews: two single-class files, text col content, no label col
    fdir = tmp_path / "banfakenews"
    fdir.mkdir()
    for fname, chunk in (("LabeledAuthentic-7K.csv", texts[:32]),
                         ("LabeledFake-1K.csv", texts[32:48])):
        with open(fdir / fname, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["articleID", "domain", "content"])
            for i, t in enumerate(chunk):
                w.writerow([i, "x.com", t])

    return tmp_path


def test_prepare_all_three(raw_root, tmp_path):
    out_root = tmp_path / "out"
    stats = {name: prepare.prepare_one(name, str(raw_root), str(out_root))
             for name in prepare.DATASETS}

    for name, st in stats.items():
        assert st.n_final > 0
        assert st.dupes_removed >= 0 and st.short_removed >= 0
        # clean csv exists and reloads with the 4 canonical columns
        import pandas as pd
        df = pd.read_csv(out_root / "clean" / f"{name}.csv", dtype=str, keep_default_na=False)
        assert list(df.columns) == ["id", "text", "label", "split"]
        assert len(df) == st.n_final
        # manifests reload and partition the ids exactly
        man = prepare.reload_manifest(name, str(out_root))
        allids = set(df["id"])
        assert set().union(*man.values()) == allids
        assert sum(len(v) for v in man.values()) == len(allids)
        # no id in two splits
        assert not (set(man["train"]) & set(man["val"]))
        assert not (set(man["val"]) & set(man["test"]))

    # sentnob keeps its provided splits
    assert stats["sentnob"].split_source == "dataset-provided"
    # bd_shs / banfakenews are stratified ~70/10/20
    for name in ("bd_shs", "banfakenews"):
        ss = stats[name].split_sizes
        n = sum(ss.values())
        assert 0.60 <= ss["train"] / n <= 0.80
        assert 0.03 <= ss["val"] / n <= 0.18
        assert 0.12 <= ss["test"] / n <= 0.30
        assert stats[name].split_source == "stratified-70/10/20"


def test_split_is_deterministic(raw_root, tmp_path):
    a = prepare.prepare_one("bd_shs", str(raw_root), str(tmp_path / "a"))
    b = prepare.prepare_one("bd_shs", str(raw_root), str(tmp_path / "b"))
    assert prepare.reload_manifest("bd_shs", str(tmp_path / "a")) == \
           prepare.reload_manifest("bd_shs", str(tmp_path / "b"))
    assert a.split_sizes == b.split_sizes


def test_preprocess_order():
    # URL and mention replaced (not removed), whitespace collapsed, no lowercasing
    out = prepare.preprocess_text("  See   https://a.b/c?d=1  @Ripon   ABC  ")
    assert out == "See <URL> <USER> ABC"
    # < 3 clusters filtered downstream, not here
    assert prepare.n_clusters("কি") == 1


def test_table2_and_m2_4_run(raw_root, tmp_path):
    out_root = tmp_path / "out"
    for name in prepare.DATASETS:
        prepare.prepare_one(name, str(raw_root), str(out_root))

    t2 = importlib.import_module("scripts.table2_report")
    rows, host = t2.build(str(out_root / "clean"))
    assert len(rows) == 3
    assert host in {r["dataset"] for r in rows}
    assert all(r["n"] > 0 and 0.0 <= r["emoji_rate"] <= 1.0 for r in rows)
    md = t2.render_markdown(rows, host)
    assert "Table II" in md and "N10" in md

    m24 = importlib.import_module("scripts.m2_4_measurements")
    samples = m24.load_samples(str(out_root / "clean"), k=20)
    assert "ALL" in samples
    n5 = m24.measure_n5(samples, seeds=(42,))
    assert set(n5["under_review"]) == {"ই/ঈ", "উ/ঊ"}
    em = m24.measure_emoji(samples)
    assert "per_dataset" in em
    # CER block on a tiny sample -- just assert it produces a 9x5 per dataset
    cer = m24.measure_cer({k: v for k, v in samples.items() if k != "ALL"}, seeds=(42,))
    assert set(cer["real"]) == {"sentnob", "bd_shs", "banfakenews"}
    for tbl in cer["real"].values():
        assert all(len(tbl[nt]) == 5 for nt in tbl)
