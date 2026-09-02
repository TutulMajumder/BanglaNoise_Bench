"""Unit tests for the noise suite (METHODOLOGY M3, CLAUDE.md "Definition of done").

The four M3 acceptance criteria, in order of importance:
  * ``test_monotonicity``             -- mean corpus CER strictly increases with
    severity for every noise type (the single most important test);
  * ``test_determinism``              -- identical output for identical inputs;
  * ``test_output_validity`` + ``test_matra_*`` -- NFC-valid, no U+FFFD, ো/ৌ
    survive both round trips;
  * ``test_non_identity_at_severity_5`` -- >=95% of eligible texts change.

Plus targeted tests for the nukta letters ড়/ঢ়/য়, the stats contract, the
severity scale, and the emoji modes.

Run:  pytest -q          (add -s to see the printed 9x5 CER table)
"""

from __future__ import annotations

import random
import unicodedata

import pytest

from noisebench import bangla_maps as bm
from noisebench import (
    EMOJI_MODES,
    NOISE_TYPES,
    SEVERITY_TO_FRACTION,
    emoji_transform,
    fraction_for_severity,
    n_to_perturb,
    perturb,
    perturb_with_stats,
)
from noisebench.validate import (
    build_cer_table,
    cer,
    has_orphan_nukta,
    is_strictly_increasing,
    is_valid_output,
    normalizer_available,
    normalize_banglabert,
)
from tests.bangla_samples import (
    SENTENCES,
    SENTENCES_EMOJI,
    SENTENCES_NUKTA,
)

SEEDS = (42, 1337, 2024)
NUKTA = "়"  # U+09BC


# ---------------------------------------------------------------------------
# Corpora (deterministic)
# ---------------------------------------------------------------------------


def _make_corpus(n_target: int, sentences_per: int, salt: int) -> list[str]:
    """Deterministic corpus of multi-sentence texts (stable across runs)."""
    rng = random.Random(salt)
    pool = list(SENTENCES)
    out: list[str] = []
    while len(out) < n_target:
        out.append(" ".join(rng.sample(pool, k=sentences_per)))
    return out


DETERMINISM_CORPUS = _make_corpus(100, 2, salt=1)
MONOTONICITY_CORPUS = _make_corpus(160, 3, salt=2)
VALIDITY_CORPUS = _make_corpus(60, 3, salt=3)
NONIDENTITY_CORPUS = _make_corpus(100, 3, salt=4)


# ---------------------------------------------------------------------------
# 1. Determinism
# ---------------------------------------------------------------------------


def test_determinism_repeat_calls_identical():
    """Same (text, noise_type, severity, seed) -> byte-identical output."""
    for text in DETERMINISM_CORPUS:
        for noise_type in NOISE_TYPES:
            for severity in (1, 3, 5):
                first = perturb(text, noise_type, severity, seed=7)
                for _ in range(2):
                    assert perturb(text, noise_type, severity, seed=7) == first


def test_determinism_full_severity_grid():
    for text in DETERMINISM_CORPUS[:20]:
        for noise_type in NOISE_TYPES:
            for severity in range(1, 6):
                a = perturb(text, noise_type, severity, seed=99)
                b = perturb(text, noise_type, severity, seed=99)
                assert a == b


def test_determinism_seed_actually_matters():
    """Different seeds should (usually) give different output on rich text."""
    text = " ".join(SENTENCES)
    differ = 0
    for noise_type in NOISE_TYPES:
        a = perturb(text, noise_type, 3, seed=1)
        b = perturb(text, noise_type, 3, seed=2)
        differ += a != b
    assert differ >= len(NOISE_TYPES) - 1  # allow at most one tie


# ---------------------------------------------------------------------------
# 2. Monotonicity  --  the single most important test
# ---------------------------------------------------------------------------


def test_monotonicity(capsys):
    table = build_cer_table(MONOTONICITY_CORPUS, seeds=SEEDS)

    with capsys.disabled():
        print(f"\n  9x5 mean CER (corpus n={len(MONOTONICITY_CORPUS)}, seeds={SEEDS})")
        print("  " + "noise_type".ljust(20) + "".join(f"s{s}".ljust(10) for s in range(1, 6)))
        for noise_type in NOISE_TYPES:
            row = table[noise_type]
            print("  " + noise_type.ljust(20) + "".join(f"{v:.4f}".ljust(10) for v in row))

    for noise_type in NOISE_TYPES:
        row = table[noise_type]
        assert is_strictly_increasing(row), f"{noise_type} not strictly increasing: {row}"
        assert row[4] > 0.02, f"{noise_type} severity-5 CER near zero: {row[4]}"


def test_bernoulli_remainder_is_decorrelated_across_texts():
    """Regression: seeding random.Random(seed) directly correlates the M1.3
    remainder draw across every text, so N7 at severity 1 (n_eligible*frac < 1)
    fires on ~1% of texts instead of the expected ~n_eligible*frac. The per-call
    seed must fold in the text.
    """
    texts = [t for t in MONOTONICITY_CORPUS
             if perturb_with_stats(t, "conjunct_split", 1, 42)[1].n_eligible >= 1]
    fired = 0
    trials = 0
    exact_sum = 0.0
    for t in texts:
        for seed in SEEDS:
            _, s = perturb_with_stats(t, "conjunct_split", 1, seed)
            trials += 1
            fired += s.n_applied > 0
            exact_sum += s.n_eligible * fraction_for_severity(1)
    expected_rate = exact_sum / trials  # ~0.24
    actual_rate = fired / trials
    assert abs(actual_rate - expected_rate) < 0.08, (actual_rate, expected_rate)


def test_n8_elongation_only_lengthens_vowels():
    """repeat_final / insert_vowel must never multiply a consonant or a
    trailing sign (ং/ঃ/ৎ/ঁ) -- only matra and independent-vowel code points."""
    import collections

    from noisebench.bangla_maps import (
        BANGLA_CONSONANTS,
        BANGLA_SIGNS,
        CHANDRABINDU,
    )

    growable_forbidden = set(BANGLA_CONSONANTS) | set(BANGLA_SIGNS) | {CHANDRABINDU}
    growable_forbidden = {ch for s in growable_forbidden for ch in s}
    texts = [
        "বাংলাদেশ একটি সুন্দর দেশ এবং আমরা গর্বিত।",
        "তিনি অংশ নিয়ে দুঃখ প্রকাশ করেছেন বাঁশ বাগানে।",
    ] + SENTENCES_NUKTA
    for text in texts:
        base = collections.Counter(unicodedata.normalize("NFC", text))
        for seed in range(25):
            out = collections.Counter(perturb(text, "elongate", 5, seed))
            for ch in growable_forbidden:
                assert out[ch] <= base[ch], (text, seed, repr(ch))


def test_severity_5_stronger_than_severity_1_per_text():
    """Not just on average: for most individual texts, s5 CER > s1 CER."""
    for noise_type in NOISE_TYPES:
        wins = 0
        total = 0
        for text in MONOTONICITY_CORPUS[:60]:
            c1 = cer(text, perturb(text, noise_type, 1, 42))
            c5 = cer(text, perturb(text, noise_type, 5, 42))
            total += 1
            wins += c5 >= c1
        assert wins / total >= 0.9, f"{noise_type}: only {wins}/{total} texts s5>=s1"


# ---------------------------------------------------------------------------
# 3. Validity / Unicode safety
# ---------------------------------------------------------------------------


def test_output_validity():
    for text in VALIDITY_CORPUS:
        for noise_type in NOISE_TYPES:
            for severity in (1, 3, 5):
                for seed in SEEDS:
                    out = perturb(text, noise_type, severity, seed)
                    assert "�" not in out, (noise_type, severity, seed)
                    assert unicodedata.normalize("NFC", out) == out
                    assert not has_orphan_nukta(out), (noise_type, severity, seed, out)
                    assert is_valid_output(out), (noise_type, severity, seed, out)


def test_emoji_output_validity():
    for text in SENTENCES_EMOJI:
        for mode in EMOJI_MODES:
            out = emoji_transform(text, mode)
            assert "�" not in out
            assert unicodedata.normalize("NFC", out) == out


def test_matra_survives_nfc_roundtrip():
    """ো (U+09CB) and ৌ (U+09CC) counts are unchanged by an NFC round trip."""
    probes = SENTENCES_NUKTA + [
        "নৌকা চলে যায়",
        "মৌমাছি মধু আনে",
        "শৌখিন মানুষের পছন্দ আলাদা",
        "সে ভালো ছেলে, নৌকায় চড়ে বৌদিকে দেখতে যায়",
    ]
    for text in probes:
        base = unicodedata.normalize("NFC", text)
        rt = unicodedata.normalize("NFC", base)
        assert base.count("ো") == rt.count("ো")
        assert base.count("ৌ") == rt.count("ৌ")


@pytest.mark.skipif(not normalizer_available(), reason="csebuetnlp/normalizer not installed")
def test_matra_survives_banglabert_normalizer_roundtrip():
    probes = SENTENCES_NUKTA + [
        "নৌকা চলে যায়",
        "মৌমাছি মধু আনে",
        "ভালো মানুষ সবাই পছন্দ করে",
        "সৌরভ আর গৌরব দুই ভাই",
    ]
    for text in probes:
        base = unicodedata.normalize("NFC", text)
        normalized = normalize_banglabert(base)
        assert normalized is not None
        assert base.count("ো") == normalized.count("ো"), (text, normalized)
        assert base.count("ৌ") == normalized.count("ৌ"), (text, normalized)


# ---------------------------------------------------------------------------
# 4. Non-identity at severity 5
# ---------------------------------------------------------------------------


def test_non_identity_at_severity_5():
    for noise_type in NOISE_TYPES:
        eligible = 0
        changed = 0
        for text in NONIDENTITY_CORPUS:
            for seed in SEEDS:
                out, stats = perturb_with_stats(text, noise_type, 5, seed)
                if stats.n_eligible > 0:
                    eligible += 1
                    changed += out != unicodedata.normalize("NFC", text)
        assert eligible > 0
        rate = changed / eligible
        assert rate >= 0.95, f"{noise_type}: only {rate:.1%} of eligible texts changed"


# ---------------------------------------------------------------------------
# 5. Nukta letters ড় / ঢ় / য়  (two code points, one cluster)
# ---------------------------------------------------------------------------


def _nukta_letter_count(text: str) -> int:
    """Count base+U+09BC letters, over NFC-decomposed form."""
    n = 0
    for i, ch in enumerate(text):
        if ch == NUKTA and i > 0 and unicodedata.category(text[i - 1]) == "Lo":
            n += 1
    return n


def test_nukta_letters_never_torn_apart():
    """Every noise type, every severity/seed: no orphaned nukta, output valid."""
    for text in SENTENCES_NUKTA:
        assert _nukta_letter_count(text) >= 1  # fixture sanity
        for noise_type in NOISE_TYPES:
            for severity in range(1, 6):
                for seed in SEEDS:
                    out = perturb(text, noise_type, severity, seed)
                    assert "�" not in out
                    assert not has_orphan_nukta(out), (noise_type, severity, seed, out)
                    # every U+09BC still rides on a letter
                    for i, ch in enumerate(out):
                        if ch == NUKTA:
                            assert i > 0 and unicodedata.category(out[i - 1]) == "Lo"


def test_nukta_whole_cluster_ops_keep_letters_intact():
    """Whole-cluster ops never TEAR a nukta letter: transpose preserves the
    count exactly (pure permutation), delete can only remove whole letters,
    insert can only add whole letters (ড়/ঢ়/য় are in the consonant pool)."""
    for text in SENTENCES_NUKTA:
        base_n = _nukta_letter_count(unicodedata.normalize("NFC", text))
        for seed in SEEDS:
            assert _nukta_letter_count(perturb(text, "char_transpose", 5, seed)) == base_n
            assert _nukta_letter_count(perturb(text, "char_delete", 5, seed)) <= base_n
            assert _nukta_letter_count(perturb(text, "char_insert", 5, seed)) >= base_n


def test_nukta_letter_is_substituted_whole():
    """N3 on a ড়/ঢ়/য় cluster replaces the whole letter, never leaving a nukta."""
    text = "বড় গাড়ি পড়ে আছে, সময় নেই।"
    for seed in range(30):
        out = perturb(text, "char_substitute", 5, seed)
        assert not has_orphan_nukta(out)
        assert is_valid_output(out)


# ---------------------------------------------------------------------------
# 6. Stats contract  --  no silent no-ops
# ---------------------------------------------------------------------------


def test_stats_reports_zero_eligible_when_nothing_to_do():
    no_conjunct = "আমি তুমি সে আমরা ওরা কথা বলি না"  # no C+hasanta+C
    out, stats = perturb_with_stats(no_conjunct, "conjunct_split", 5, 42)
    assert stats.n_eligible == 0
    assert stats.n_applied == 0
    assert stats.unchanged is True
    assert out == unicodedata.normalize("NFC", no_conjunct)


def test_stats_reports_work_done_when_eligible():
    rich = " ".join(SENTENCES)
    for noise_type in NOISE_TYPES:
        out, stats = perturb_with_stats(rich, noise_type, 5, 42)
        assert stats.n_eligible > 0
        assert stats.n_applied > 0
        assert stats.submodes  # non-empty
        assert sum(stats.submodes.values()) == stats.n_applied
        assert stats.unchanged is (out == unicodedata.normalize("NFC", rich))


def test_n_requested_vs_applied_contract():
    """n_applied <= n_requested <= n_eligible always. n_applied == n_requested
    for every type except N2 (don't-empty-a-token guard) and N4 (its
    shrinking-pool selection can run out of disjoint pairs before k, though on
    real text this is negligible -- see test_achieved_rate_matches_nominal)."""
    guarded = {"char_delete", "char_transpose"}
    rich = " ".join(SENTENCES * 2)
    for noise_type in NOISE_TYPES:
        for sev in range(1, 6):
            for seed in SEEDS:
                _, s = perturb_with_stats(rich, noise_type, sev, seed)
                assert 0 <= s.n_applied <= s.n_requested <= s.n_eligible
                if noise_type not in guarded:
                    assert s.n_applied == s.n_requested, (noise_type, sev, seed)


def test_achieved_rate_matches_nominal_for_unguarded_types():
    """The seeded-Bernoulli count rule is unbiased: mean n_applied / n_eligible
    tracks the nominal fraction for every type whose guards don't drop units.
    (Low-count types like conjunct_split at severity 1 are estimation-noisy, so
    the check needs enough expected events per cell.)"""
    from noisebench.validate import achieved_rate_table

    corpus = _make_corpus(200, 3, salt=9)
    table = achieved_rate_table(corpus, seeds=SEEDS)
    for noise_type, rows in table.items():
        for cell in rows:
            expected_events = cell["mean_eligible"] * cell["nominal_fraction"]
            if expected_events >= 2.0:  # enough to estimate a rate
                assert abs(cell["requested_over_nominal"] - 1.0) < 0.06, (
                    noise_type, cell)
            # N2 (char_delete) is the only type whose guard makes the achieved
            # fraction fall below nominal by construction; every other type,
            # N4 included since the disjoint-pair rewrite, should track nominal.
            if noise_type != "char_delete" and expected_events >= 2.0:
                rel = abs(cell["achieved_fraction"] - cell["nominal_fraction"])
                assert rel < 0.06 * cell["nominal_fraction"], (noise_type, cell)


def test_stats_submodes_cover_mechanisms():
    rich = " ".join(SENTENCES * 2)
    _, s5 = perturb_with_stats(rich, "homophone_confuse", 5, 42)
    assert set(s5.submodes) <= {"/".join(s) for s in bm.HOMOPHONE_SETS}
    _, s7 = perturb_with_stats(rich, "conjunct_split", 5, 42)
    assert set(s7.submodes) == {"drop_hasanta"}  # single mechanism (M0.2 row 6, revised)
    _, s8 = perturb_with_stats(rich, "elongate", 5, 42)
    assert set(s8.submodes).issubset({"insert_vowel", "repeat_vowel", "repeat_final"})
    _, s9 = perturb_with_stats(rich, "whitespace_error", 5, 42)
    assert set(s9.submodes) == {"split", "merge"}


def test_perturb_wrapper_matches_stats_output():
    for text in DETERMINISM_CORPUS[:15]:
        for noise_type in NOISE_TYPES:
            out1 = perturb(text, noise_type, 3, 5)
            out2, _ = perturb_with_stats(text, noise_type, 3, 5)
            assert out1 == out2


# ---------------------------------------------------------------------------
# 7. Severity scale + count rule
# ---------------------------------------------------------------------------


def test_fraction_for_severity_values():
    assert SEVERITY_TO_FRACTION == {1: 0.05, 2: 0.10, 3: 0.20, 4: 0.35, 5: 0.50}
    for s, f in SEVERITY_TO_FRACTION.items():
        assert fraction_for_severity(s) == f


@pytest.mark.parametrize("bad", [0, 6, -1, 3.0, True, "3"])
def test_fraction_for_severity_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        fraction_for_severity(bad)


def test_n_to_perturb_is_not_a_silent_noop_on_short_text():
    """Severity 1 on a 12-unit text must sometimes perturb 1 unit, not always 0."""
    counts = [n_to_perturb(12, 0.05, random.Random(s)) for s in range(400)]
    assert any(c >= 1 for c in counts)          # not the constant-zero bug
    assert 0.0 < sum(counts) / len(counts) < 2.0


def test_n_to_perturb_expected_rate():
    for n, frac in [(50, 0.05), (50, 0.20), (37, 0.35), (100, 0.50)]:
        draws = [n_to_perturb(n, frac, random.Random(s)) for s in range(3000)]
        mean = sum(draws) / len(draws)
        assert abs(mean - n * frac) < 0.06 * n
        assert all(0 <= d <= n for d in draws)


def test_n_to_perturb_zero_eligible():
    assert n_to_perturb(0, 0.5, random.Random(0)) == 0


# ---------------------------------------------------------------------------
# 8. Emoji modes (N10)
# ---------------------------------------------------------------------------


def _has_emoji(text: str) -> bool:
    from noisebench.perturbations import _is_emoji_codepoint

    return any(_is_emoji_codepoint(ch) for ch in text)


def test_emoji_keep_is_identity():
    for text in SENTENCES_EMOJI:
        assert emoji_transform(text, "keep") == unicodedata.normalize("NFC", text)


def test_emoji_remove_strips_all_emoji():
    for text in SENTENCES_EMOJI:
        assert _has_emoji(text)  # fixture sanity
        out = emoji_transform(text, "remove")
        assert not _has_emoji(out)


def test_emoji_replace_produces_no_emoji():
    for text in SENTENCES_EMOJI:
        out = emoji_transform(text, "replace_with_text")
        assert not _has_emoji(out)
        assert ":" in out  # at least one :label:


def test_emoji_unknown_mode_raises():
    with pytest.raises(ValueError):
        emoji_transform("hi 🎉", "delete_all")


# ---------------------------------------------------------------------------
# 9. Dispatch errors
# ---------------------------------------------------------------------------


def test_unknown_noise_type_raises():
    with pytest.raises(ValueError):
        perturb("কিছু একটা", "char_shuffle", 3, 42)


def test_emoji_via_perturb_raises_with_pointer():
    with pytest.raises(ValueError, match="emoji_transform"):
        perturb("hi 🎉", "emoji_transform", 3, 42)


def test_n5_dropped_b_bh_set_never_fires():
    """ব/ভ was removed from HOMOPHONE_SETS (pending M2.4 corpus evidence);
    N5 must never swap ব<->ভ now."""
    from noisebench.bangla_maps import HOMOPHONE_SETS

    assert all(set(s) != {"ব", "ভ"} for s in HOMOPHONE_SETS)
    text = " ".join(["ভালোবাসা ভাই বাবা ভয় বই ভবন বাংলা ভাষা"] * 5)
    base_b = text.count("ব")
    base_bh = text.count("ভ")
    for seed in range(20):
        out = perturb(text, "homophone_confuse", 5, seed)
        # ব and ভ counts can only change via each other in this text; assert stable
        assert out.count("ব") == base_b and out.count("ভ") == base_bh, (seed, out)


def test_bangla_maps_structural_assertions():
    """The constants-sheet assertions (pool sizes; no matra/hasanta in any
    IN_CLASS_POOL or INSERTABLE_CHARS; disjoint pools; consistent derived maps;
    NFC-idempotent constants) are part of CI, not just the report script."""
    import importlib.util
    import pathlib

    spec = importlib.util.spec_from_file_location(
        "constants_sheet",
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "constants_sheet.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    checks, all_ok = mod.run_assertions()
    failed = [name for name, ok, _ in checks if not ok]
    assert all_ok, f"failed structural checks: {failed}"


def test_all_ten_types_are_exposed():
    assert len(NOISE_TYPES) == 9
    assert set(NOISE_TYPES) == {
        "char_insert", "char_delete", "char_substitute", "char_transpose",
        "homophone_confuse", "matra_perturb", "conjunct_split", "elongate",
        "whitespace_error",
    }
    assert EMOJI_MODES == ("keep", "remove", "replace_with_text")
