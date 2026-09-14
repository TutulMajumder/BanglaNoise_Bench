"""Shared pytest fixtures for the Phase 2 pipeline tests."""

from __future__ import annotations

import csv
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.bangla_samples import SENTENCES


def _rows(n: int) -> list[str]:
    base = list(dict.fromkeys(SENTENCES))  # de-dupe, keep order
    assert len(base) >= 20, "need a few dozen distinct sample sentences"
    return (base * ((n // len(base)) + 1))[:n]


@pytest.fixture()
def raw_root(tmp_path):
    """Lay out fixture CSVs matching each DATASETS spec's expected schema --
    no invented text, sourced from tests/bangla_samples.py."""
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

    # bd_shs: ships its own train/val/test.csv (all pulled and re-split,
    # METHODOLOGY M2.2), columns sentence,"hate speech" (0/1) -- the shipped
    # label column has a space in its name, verified against the real CSV.
    bdir = tmp_path / "bd_shs"
    bdir.mkdir()
    for split, chunk in (("train", texts[:30]), ("val", texts[30:36]), ("test", texts[36:48])):
        with open(bdir / f"{split}.csv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["sentence", "hate speech"])
            for i, t in enumerate(chunk):
                w.writerow([t, i % 2])

    # banfakenews: two single-class files, text col content, no label col
    # (primary, balanced) + the full imbalanced pair (secondary condition,
    # same directory, different filenames -- banfakenews_full).
    # banfakenews_full (Authentic-48K.csv/Fake-1K.csv, the secondary/imbalanced
    # condition) was cut 2026-09-15 -- see PHASE2_STATUS.md. Deliberately
    # asymmetric chunk sizes below (32 vs 16) so a label-assignment bug that
    # swaps the two files' labels would show up as a wrong CLASS COUNT, not
    # just a wrong class NAME -- this is the fixture the 2026-09-14 s1/s2
    # label-mismatch bug (test_s1_s2_agree_on_raw_class_distribution) needs.
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
