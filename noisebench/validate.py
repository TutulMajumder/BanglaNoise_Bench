"""Validation utilities for the noise suite (METHODOLOGY M1.5 + M3).

Two jobs:

1. **CER** -- normalized Levenshtein distance over NFC grapheme-cluster
   sequences (``cer`` / ``corpus_cer``). This is the metric the whole benchmark
   rests on; computing it over code points instead would inflate exactly the
   Bangla-specific noise types (N6, N7).

2. **Acceptance checks** for M3:
   * ``build_cer_table``      -- the 9x5 mean-CER table (must be strictly
     increasing along each row);
   * ``is_valid_output``      -- NFC-valid, no U+FFFD, no orphaned combining
     mark;
   * ``matra_roundtrip_report`` -- ো / ৌ survive an NFC round trip and a
     BanglaBERT-normalizer round trip;
   * ``normalizer_survival``  -- the normalizer does NOT undo the noise
     (post-normalizer CER > 0).

3. **Emoji-range coverage** (``emoji_range_coverage``, METHODOLOGY M0 Q9) --
   scores the shipped hand-rolled ``EMOJI_CODEPOINT_RANGES`` against the
   ``emoji`` package (an optional, measurement-only dependency; the perturbation
   code never imports it).

THE NORMALIZER IS ``csebuetnlp/normalizer`` (import name ``normalizer``,
``from normalizer import normalize``) -- the exact package BanglaBERT's own
preprocessing pipeline uses (its model card / training code call
``normalize(text)`` with defaults). This is NOT the same package as
``bnunicodenormalizer`` (a different BUET project, used for OCR/ASR text, not
BanglaBERT). Using the wrong one here would validate a guarantee that Phase 3's
``--normalizer on`` flag does not actually provide -- so Phase 3 MUST call
``noisebench.validate.normalize_banglabert`` (or the equivalent
``normalizer.normalize`` call with the same arguments), never a different
normalizer. It is an *optional* import here regardless: if it is missing, the
two normalizer-based helpers return ``None`` and the caller is expected to skip
that check with a warning -- but `requirements.txt` pins it because it affects
reported results, not just tests.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

from .bangla_maps import COMBINING_MARKS, EMOJI_SEQUENCE_GLUE
from .perturbations import (
    NOISE_TYPES,
    clusters,
    emoji_transform,
    perturb,
    perturb_with_stats,
)
# The hand-rolled emoji detector that actually ships (bangla_maps.EMOJI_CODEPOINT_
# RANGES). Imported here as the single source of truth for the M0 Q9 coverage
# measurement below -- the measurement compares THIS against the `emoji` package.
from .perturbations import _is_emoji_codepoint  # noqa: PLC2701
from .severity import MAX_SEVERITY, MIN_SEVERITY, fraction_for_severity

__all__ = [
    "cer",
    "corpus_cer",
    "achieved_rate_table",
    "levenshtein",
    "is_valid_output",
    "has_orphan_nukta",
    "build_cer_table",
    "is_strictly_increasing",
    "normalizer_available",
    "normalize_banglabert",
    "matra_roundtrip_report",
    "normalizer_survival",
    "emoji_lib_available",
    "emoji_range_coverage",
]

NUKTA = "়"
_EMOJI_GLUE_SET = frozenset(EMOJI_SEQUENCE_GLUE)


# ---------------------------------------------------------------------------
# CER
# ---------------------------------------------------------------------------


def levenshtein(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two sequences (unit cost).

    Operates on lists of tokens (here: grapheme clusters), not characters.
    O(len(a) * len(b)) time, O(len(b)) space.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """Character error rate: cluster-level Levenshtein, normalized by reference.

    ``CER(a, b) = levenshtein(clusters(a), clusters(b)) / max(1, len(clusters(a)))``
    (METHODOLOGY M1.5). Both strings are NFC-normalized and split into extended
    grapheme clusters first, so ``কি`` counts as one unit and ``ড়`` as one unit.

    Args:
        reference: the original (clean) text -- the denominator.
        hypothesis: the perturbed text.

    Returns:
        Non-negative float. ``0.0`` iff the cluster sequences are identical.
        Can exceed ``1.0`` when the hypothesis is much longer than the
        reference (e.g. heavy N8 elongation).
    """
    ref_clusters = clusters(reference)
    hyp_clusters = clusters(hypothesis)
    return levenshtein(ref_clusters, hyp_clusters) / max(1, len(ref_clusters))


def corpus_cer(pairs: list[tuple[str, str]]) -> float:
    """Arithmetic mean of ``cer`` over ``(reference, hypothesis)`` pairs.

    Returns ``0.0`` for an empty list.
    """
    if not pairs:
        return 0.0
    return sum(cer(ref, hyp) for ref, hyp in pairs) / len(pairs)


# ---------------------------------------------------------------------------
# Output validity
# ---------------------------------------------------------------------------


def has_orphan_nukta(text: str) -> bool:
    """True if a nukta (U+09BC) is not immediately preceded by a base letter.

    ``ড়``/``ঢ়``/``য়`` decompose under NFC to ``<base, U+09BC>``. A code-point
    level slip (splitting or reordering inside the cluster) would leave the
    nukta stranded -- this catches that.
    """
    for i, ch in enumerate(text):
        if ch == NUKTA:
            if i == 0:
                return True
            prev = text[i - 1]
            if not (unicodedata.category(prev) == "Lo"):
                return True
    return False


def is_valid_output(text: str) -> bool:
    """True if ``text`` is a well-formed perturbation output.

    Checks (METHODOLOGY M3 criterion 3):
      * no U+FFFD replacement character;
      * NFC-idempotent (``normalize("NFC", text) == text``);
      * no orphaned nukta;
      * no combining mark at string start or directly after whitespace
        (a matra/hasanta with nothing to attach to).
    """
    if "�" in text:
        return False
    if unicodedata.normalize("NFC", text) != text:
        return False
    if has_orphan_nukta(text):
        return False
    prev = ""
    for ch in text:
        at_boundary = prev == "" or prev.isspace()
        if at_boundary and (ch in COMBINING_MARKS or unicodedata.category(ch).startswith("M")):
            return False
        prev = ch
    return True


# ---------------------------------------------------------------------------
# The 9x5 CER table (M3 criterion 2)
# ---------------------------------------------------------------------------


def build_cer_table(
    texts: list[str],
    seeds: tuple[int, ...] = (42, 1337, 2024),
    noise_types: tuple[str, ...] = NOISE_TYPES,
) -> dict[str, list[float]]:
    """Mean corpus CER for every noise type at every severity.

    For each ``(noise_type, severity)`` cell, CER is averaged over
    ``texts x seeds``.

    Returns:
        ``{noise_type: [cer_s1, cer_s2, cer_s3, cer_s4, cer_s5]}``.
    """
    table: dict[str, list[float]] = {}
    severities = list(range(MIN_SEVERITY, MAX_SEVERITY + 1))
    for noise_type in noise_types:
        row: list[float] = []
        for severity in severities:
            pairs: list[tuple[str, str]] = []
            for text in texts:
                for seed in seeds:
                    pairs.append((text, perturb(text, noise_type, severity, seed)))
            row.append(corpus_cer(pairs))
        table[noise_type] = row
    return table


def is_strictly_increasing(values: list[float], tol: float = 0.0) -> bool:
    """True if each value exceeds the previous by more than ``tol``."""
    return all(b - a > tol for a, b in zip(values, values[1:]))


def achieved_rate_table(
    texts: list[str],
    seeds: tuple[int, ...] = (42, 1337, 2024),
    noise_types: tuple[str, ...] = NOISE_TYPES,
) -> dict[str, list[dict[str, float]]]:
    """Does "severity = fraction of eligible units" actually hold, per type?

    For each ``(noise_type, severity)`` cell, over ``texts x seeds`` (counting
    only texts with ``n_eligible > 0``), report:

      * ``mean_eligible``   -- mean ``n_eligible``;
      * ``mean_requested``  -- mean ``n_requested`` (what the M1.3 count rule
        selected);
      * ``mean_applied``    -- mean ``n_applied`` (what actually happened after
        the per-type guards);
      * ``applied_over_requested`` -- mean of ``n_applied / n_requested`` over
        cases with ``n_requested > 0``. **< 1.0 means a guard is dropping
        selected units** (N2's don't-empty-a-token, N4's overlap-skip);
      * ``requested_over_nominal`` -- ``mean_requested / (mean_eligible *
        frac)``; confirms the Bernoulli count rule itself is unbiased (~1.0);
      * ``achieved_fraction`` -- ``mean_applied / mean_eligible``, the rate the
        paper can actually claim, vs. the nominal ``frac``.

    Returns ``{noise_type: [cell_s1, ..., cell_s5]}``.
    """
    severities = list(range(MIN_SEVERITY, MAX_SEVERITY + 1))
    out: dict[str, list[dict[str, float]]] = {}
    for noise_type in noise_types:
        rows: list[dict[str, float]] = []
        for severity in severities:
            frac = fraction_for_severity(severity)
            n = 0
            sum_elig = sum_req = sum_app = 0
            ratio_sum = 0.0
            ratio_n = 0
            for text in texts:
                for seed in seeds:
                    st = perturb_with_stats(text, noise_type, severity, seed)[1]
                    if st.n_eligible == 0:
                        continue
                    n += 1
                    sum_elig += st.n_eligible
                    sum_req += st.n_requested
                    sum_app += st.n_applied
                    if st.n_requested > 0:
                        ratio_sum += st.n_applied / st.n_requested
                        ratio_n += 1
            mean_elig = sum_elig / n if n else 0.0
            mean_req = sum_req / n if n else 0.0
            mean_app = sum_app / n if n else 0.0
            rows.append({
                "mean_eligible": mean_elig,
                "mean_requested": mean_req,
                "mean_applied": mean_app,
                "applied_over_requested": ratio_sum / ratio_n if ratio_n else 1.0,
                "requested_over_nominal": (
                    mean_req / (mean_elig * frac) if mean_elig * frac else 1.0
                ),
                "achieved_fraction": mean_app / mean_elig if mean_elig else 0.0,
                "nominal_fraction": frac,
            })
        out[noise_type] = rows
    return out


# ---------------------------------------------------------------------------
# BanglaBERT normalizer interaction (optional dependency, pinned in
# requirements.txt because it affects reported results -- see module docstring)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_normalize_fn():
    """Return the cached ``csebuetnlp/normalizer`` ``normalize`` function, or
    ``None`` if the package is not installed."""
    try:
        from normalizer import normalize
    except ImportError:
        return None
    return normalize


def normalizer_available() -> bool:
    """True if ``csebuetnlp/normalizer`` (import name ``normalizer``) is importable."""
    return _get_normalize_fn() is not None


def normalize_banglabert(text: str) -> str | None:
    """Apply the BanglaBERT normalizer -- ``csebuetnlp/normalizer``'s ``normalize``.

    Called with ALL DEFAULTS (``normalize(text)``), matching how BanglaBERT's
    own preprocessing pipeline calls it. Phase 3's ``--normalizer on`` flag
    must call it the same way, or the survival guarantee measured here does not
    transfer.

    Returns ``None`` if the package is not installed.
    """
    normalize = _get_normalize_fn()
    if normalize is None:
        return None
    return normalize(text)


def matra_roundtrip_report(probe_texts: list[str]) -> dict[str, object]:
    """Check that ো (U+09CB) and ৌ (U+09CC) survive both round trips.

    "Survive" = the count of ো and of ৌ, measured over grapheme clusters, is
    unchanged after (a) an NFC round trip and (b) a BanglaBERT-normalizer round
    trip.

    Returns a dict with ``nfc_ok`` (bool), ``normalizer_ok`` (bool or ``None``
    if the normalizer is unavailable), and a list of any failing probes.
    """

    def sign_counts(s: str) -> tuple[int, int]:
        joined = "".join(clusters(s))
        return joined.count("ো"), joined.count("ৌ")

    nfc_failures: list[str] = []
    norm_failures: list[str] = []
    normalizer_ok: bool | None = None if not normalizer_available() else True

    for text in probe_texts:
        base = unicodedata.normalize("NFC", text)
        if sign_counts(base) != sign_counts(unicodedata.normalize("NFC", base)):
            nfc_failures.append(text)
        if normalizer_available():
            normalized = normalize_banglabert(base)
            if normalized is not None and sign_counts(base) != sign_counts(normalized):
                norm_failures.append(text)

    if normalizer_ok is not None:
        normalizer_ok = not norm_failures

    return {
        "nfc_ok": not nfc_failures,
        "normalizer_ok": normalizer_ok,
        "nfc_failures": nfc_failures,
        "normalizer_failures": norm_failures,
    }


def normalizer_survival(
    texts: list[str],
    severity: int = 3,
    seeds: tuple[int, ...] = (42, 1337, 2024),
    noise_types: tuple[str, ...] = NOISE_TYPES,
) -> dict[str, dict[str, float]] | None:
    """Mean CER between the ORIGINAL text and the *normalized* perturbed text --
    and how often the normalizer actually touched anything.

    If the BanglaBERT normalizer silently undoes a noise type, that type's
    ``post_normalizer_cer`` collapses toward ``0`` and it must be flagged in the
    paper (METHODOLOGY M0 Q10, Topic4 plan section 1). But a high CER is only
    reassuring if the normalizer actually ran on the text: on clean synthetic
    sentences (no URLs, emails, or punctuation-spacing issues) ``normalize()``
    can be a complete no-op, in which case ``post_normalizer_cer ==`` the plain
    CER trivially and proves nothing about the normalizer. ``touch_rate`` (the
    fraction of calls where ``normalize(perturbed) != perturbed``) makes that
    visible instead of hiding it.

    Returns ``None`` if the normalizer is unavailable, else
    ``{noise_type: {"post_normalizer_cer": ..., "touch_rate": ...}}``.
    """
    if not normalizer_available():
        return None
    result: dict[str, dict[str, float]] = {}
    for noise_type in noise_types:
        pairs: list[tuple[str, str]] = []
        touched = 0
        total = 0
        for text in texts:
            base = unicodedata.normalize("NFC", text)
            for seed in seeds:
                noisy = perturb(base, noise_type, severity, seed)
                normalized = normalize_banglabert(noisy)
                total += 1
                touched += normalized is not None and normalized != noisy
                pairs.append((base, normalized if normalized is not None else noisy))
        result[noise_type] = {
            "post_normalizer_cer": corpus_cer(pairs),
            "touch_rate": touched / total if total else 0.0,
        }
    return result


# ---------------------------------------------------------------------------
# Emoji-detection coverage (METHODOLOGY M0 Q9) -- MEASUREMENT ONLY
#
# ``noisebench/perturbations.py`` detects emoji with a hand-rolled 9-range table
# (``bangla_maps.EMOJI_CODEPOINT_RANGES``) and deliberately does NOT import the
# ``emoji`` package: the released artefact keeps its dependency surface minimal
# and its detection auditable. ``emoji`` is nonetheless present in the pinned
# environment -- ``csebuetnlp/normalizer`` pulls it transitively -- so this
# helper uses it (when importable) purely to *score* the hand-rolled ranges
# against a maintained reference emoji list, on the Phase 2 corpora. It is an
# optional import: absent ``emoji``, the helper returns ``None``.
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_emoji_module():
    """Return the cached ``emoji`` module, or ``None`` if it is not installed."""
    try:
        import emoji
    except ImportError:
        return None
    return emoji


def emoji_lib_available() -> bool:
    """True if the ``emoji`` package (transitive dep of the normalizer) imports."""
    return _get_emoji_module() is not None


def _emoji_occurrences(text: str) -> list[str] | None:
    """Every emoji substring in ``text`` per the ``emoji`` package.

    Version-tolerant: ``emoji_list`` (>=2.0), ``emoji_lis`` (1.x), or the
    ``get_emoji_regexp`` fallback (older). Returns ``None`` if ``emoji`` is
    unavailable or exposes none of those APIs.
    """
    mod = _get_emoji_module()
    if mod is None:
        return None
    if hasattr(mod, "emoji_list"):
        return [d["emoji"] for d in mod.emoji_list(text)]
    if hasattr(mod, "emoji_lis"):
        return [d["emoji"] for d in mod.emoji_lis(text)]
    if hasattr(mod, "get_emoji_regexp"):
        return [m.group() for m in mod.get_emoji_regexp().finditer(text)]
    return None


def emoji_range_coverage(
    texts: list[str],
    max_examples: int = 25,
) -> dict[str, object] | None:
    """Score ``EMOJI_CODEPOINT_RANGES`` against the ``emoji`` package (M0 Q9).

    For every emoji occurrence the ``emoji`` package finds in ``texts``, check
    whether the shipped detector (:func:`perturbations._is_emoji_codepoint`)
    flags its code points. "Glue" code points -- ZWJ, variation selectors,
    enclosing keycap (``EMOJI_SEQUENCE_GLUE``) -- are excluded from the check:
    the ranges are not meant to match them and ``emoji_transform`` strips them
    on their own.

    Returns ``None`` if the ``emoji`` package is not installed, else a dict:

      * ``emoji_lib_version``           -- ``emoji.__version__`` (or ``None``);
      * ``n_occurrences``               -- emoji occurrences the package found;
      * ``n_caught_any`` / ``coverage_any``   -- occurrences (fraction) with
        >=1 non-glue code point flagged (the emoji is at least partly stripped
        by N10 "remove");
      * ``n_caught_full`` / ``coverage_full`` -- occurrences (fraction) with
        *every* non-glue code point flagged (stripped cleanly, nothing left for
        :func:`is_valid_output` to reject);
      * ``n_false_positive_codepoints`` -- distinct code points in ``texts``
        that the ranges flag but the ``emoji`` package never attributes to an
        emoji (possible over-matching of ordinary symbols);
      * ``uncaught``                    -- up to ``max_examples`` distinct emoji
        with no non-glue code point flagged.

    Caveat: the pinned ``emoji`` is 1.4.2 (pinned transitively by
    ``csebuetnlp/normalizer``), whose data tracks roughly Unicode 13 (2021).
    Emoji added after that are absent from the reference too, so the coverage
    figure is an upper bound against a 2021-era reference, not current Unicode.
    """
    mod = _get_emoji_module()
    if mod is None:
        return None

    n_occ = n_any = n_full = 0
    uncaught: list[str] = []
    uncaught_seen: set[str] = set()
    emoji_codepoints: set[str] = set()

    for text in texts:
        for occurrence in _emoji_occurrences(text) or []:
            n_occ += 1
            core = [ch for ch in occurrence if ch not in _EMOJI_GLUE_SET]
            emoji_codepoints.update(core)
            flags = [_is_emoji_codepoint(ch) for ch in core]
            if any(flags):
                n_any += 1
            if core and all(flags):
                n_full += 1
            if core and not any(flags) and occurrence not in uncaught_seen:
                uncaught_seen.add(occurrence)
                if len(uncaught) < max_examples:
                    uncaught.append(occurrence)

    corpus_codepoints = {ch for text in texts for ch in text}
    false_positives = sum(
        1
        for ch in corpus_codepoints
        if _is_emoji_codepoint(ch) and ch not in emoji_codepoints
    )

    return {
        "emoji_lib_version": getattr(mod, "__version__", None),
        "n_occurrences": n_occ,
        "n_caught_any": n_any,
        "n_caught_full": n_full,
        "coverage_any": n_any / n_occ if n_occ else 0.0,
        "coverage_full": n_full / n_occ if n_occ else 0.0,
        "n_false_positive_codepoints": false_positives,
        "uncaught": uncaught,
    }
