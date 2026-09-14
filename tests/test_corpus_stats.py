"""Tests for noisebench/corpus_stats.py -- the shared s1 / s3
measurement module. Uses the existing bangla_samples fixture; no invented text.
"""

from __future__ import annotations

from noisebench import corpus_stats as cs
from noisebench.perturbations import NOISE_TYPES
from tests.bangla_samples import SENTENCES


def test_char_class_composition_sums_to_total_chars():
    texts = SENTENCES[:10]
    counts = cs.char_class_composition(texts)
    assert sum(counts.values()) == sum(len(t) for t in texts)
    assert counts["bengali"] > 0
    assert set(counts) == set(cs.CHAR_CLASSES)


def test_char_class_composition_replacement_and_emoji():
    counts = cs.char_class_composition(["bad�text", "fire\U0001F525"])
    assert counts["replacement"] == 1
    assert counts["emoji"] == 1


def test_latin_share_per_text():
    shares = cs.latin_share_per_text(["hello", "আমি", "আমি hello", ""])
    assert shares[0] == 1.0
    assert shares[1] == 0.0
    assert 0.0 < shares[2] < 1.0
    assert shares[3] == 0.0


def test_grapheme_lengths_matches_clusters():
    from noisebench.perturbations import clusters
    texts = SENTENCES[:5]
    assert cs.grapheme_lengths(texts) == [len(clusters(t)) for t in texts]


def test_duplicate_count():
    d = cs.duplicate_count(["a", "b", "a", "c", "a"])
    assert d == {"n_rows": 5, "n_unique": 3, "n_duplicate_rows": 2}


def test_cross_split_duplicate_counts():
    out = cs.cross_split_duplicate_counts({
        "train": ["a", "b", "c"],
        "val": ["b", "d"],
        "test": ["c", "d", "e"],
    })
    # keys are sorted alphabetically: test, train, val
    assert out == {"test|train": 1, "test|val": 1, "train|val": 1}


def test_quality_inventory_zero_bangla_and_empty():
    inv = cs.quality_inventory({"all": ["আমি ভালো", "hello", "", "  ", "bad�"]})
    assert inv.n_rows == 5
    assert inv.n_empty_or_whitespace == 2
    assert inv.n_zero_bangla == 4  # "hello", "", "  ", "bad�" all have zero Bengali-block chars
    assert inv.n_replacement_char == 1


def test_flag_vs_label_table():
    texts = ["a�", "b", "c�", "d"]
    labels = [1, 0, 1, 0]
    out = cs.flag_vs_label_table(texts, labels, lambda t: "�" in t,
                                  label_names={0: "no", 1: "yes"})
    assert out["flagged"]["n"] == 2
    assert out["flagged"]["label_dist_pct"] == {"yes": 100.0}
    assert out["unflagged"]["n"] == 2
    assert out["unflagged"]["label_dist_pct"] == {"no": 100.0}


def test_eligible_units_independent_of_severity_and_seed():
    """n_eligible is a structural property of the text, not of severity/seed
    -- this is the invariant scripts/stage_c_eda.py's eligible-units figure
    relies on to call perturb_with_stats once per (text, noise_type)."""
    texts = SENTENCES[:8]
    a = cs.eligible_units_per_noise_type(texts, severity=1, seed=42)
    b = cs.eligible_units_per_noise_type(texts, severity=5, seed=999)
    assert a.keys() == b.keys() == set(NOISE_TYPES)
    for nt in NOISE_TYPES:
        assert a[nt] == b[nt], f"{nt}: n_eligible depends on severity/seed"
        assert all(n >= 0 for n in a[nt])


def test_sha256_file(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello", encoding="utf-8")
    import hashlib
    assert cs.sha256_file(str(p)) == hashlib.sha256(b"hello").hexdigest()


def test_utf8_decode_failures(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"good line\n" + b"\xff\xfe bad line\n" + "ভালো\n".encode("utf-8"))
    assert cs.utf8_decode_failures(str(p)) == 1
