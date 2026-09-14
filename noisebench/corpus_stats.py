"""Shared corpus-measurement functions -- the single source of truth for
pipeline stage s1 (raw inspection) and s3 (post-preparation description).

Every metric here is computed identically regardless of whether the input is
`data/raw/` text or `data/final/` text; the two stages differ only in *which*
texts they pass in. This is what makes s3's raw-vs-final delta table
meaningful -- it is never comparing two different implementations that happen
to compute "similar" things.

Standard library + ``regex`` only (same constraint as the rest of
``noisebench/`` -- CLAUDE.md). No pandas, no matplotlib: those live in
`pipeline/`, which calls into this module and then shapes/plots the results.
"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass, field

import regex

from .perturbations import NOISE_TYPES, _is_emoji_codepoint, clusters, perturb_with_stats
from .severity import MIN_SEVERITY

_BENGALI_RE = regex.compile(r"\p{Script=Bengali}")
_LATIN_RE = regex.compile(r"\p{Script=Latin}")

#: Grapheme-cluster cap for the M2.4 CER measurement (pipeline/s4_measure.py)
#: -- shared here so s3_describe.py's length-ECDF figure can mark the same
#: cap and report the fraction of texts it affects, per METHODOLOGY M2.4.
#: `validate.levenshtein` is O(n^2); article-length real text (BanFakeNews
#: averages ~1,100 clusters) made the uncapped real-corpus CER table
#: unrunnable (a real run hung 17h). CER is reference-length-normalized and
#: the M1.3 count rule is per-eligible-unit, so the cap does not bias ratios.
CER_TRUNCATE_CLUSTERS = 300

#: Mutually-exclusive character classes, in priority order. A character is
#: assigned to the first class it matches; classes are counted once each so
#: the totals sum to the total character count (stacked-bar-ready).
CHAR_CLASSES: tuple[str, ...] = (
    "replacement", "emoji", "bengali", "latin", "digit", "punctuation",
    "whitespace", "other",
)


# ---------------------------------------------------------------------------
# File-level
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """SHA-256 of a file's raw bytes, for the provenance record."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utf8_decode_failures(path: str) -> int:
    """Count of lines in a file that do not decode as UTF-8.

    Reads raw bytes and tries to decode line by line, since a single bad byte
    would otherwise fail the whole file. Returns 0 for a fully clean file.
    """
    n_fail = 0
    with open(path, "rb") as fh:
        for line in fh:
            try:
                line.decode("utf-8")
            except UnicodeDecodeError:
                n_fail += 1
    return n_fail


# ---------------------------------------------------------------------------
# Per-text classification
# ---------------------------------------------------------------------------


def classify_char(ch: str) -> str:
    """Which of ``CHAR_CLASSES`` a single character belongs to."""
    if ch == "�":
        return "replacement"
    if _is_emoji_codepoint(ch):
        return "emoji"
    if _BENGALI_RE.match(ch):
        return "bengali"
    if _LATIN_RE.match(ch):
        return "latin"
    if ch.isdigit():
        return "digit"
    if unicodedata.category(ch).startswith("P"):
        return "punctuation"
    if ch.isspace():
        return "whitespace"
    return "other"


def char_class_composition(texts: list[str]) -> dict[str, int]:
    """Character count per class (see ``CHAR_CLASSES``), summed over all texts.

    Denominator for a "share of all characters" stacked bar is
    ``sum(char_class_composition(texts).values())``.
    """
    counts: dict[str, int] = {c: 0 for c in CHAR_CLASSES}
    for text in texts:
        for ch in text:
            counts[classify_char(ch)] += 1
    return counts


def latin_share_per_text(texts: list[str]) -> list[float]:
    """Per-text fraction of characters classified ``latin`` (code-mixing ECDF
    input). ``0.0`` for an empty text."""
    out = []
    for text in texts:
        n = len(text)
        if n == 0:
            out.append(0.0)
            continue
        n_latin = sum(1 for ch in text if classify_char(ch) == "latin")
        out.append(n_latin / n)
    return out


def grapheme_lengths(texts: list[str]) -> list[int]:
    """Grapheme-cluster count per text (NFC + ``regex \\X``)."""
    return [len(clusters(t)) for t in texts]


# ---------------------------------------------------------------------------
# Data-quality inventory
# ---------------------------------------------------------------------------


def class_distribution(labels: list) -> dict[str, int]:
    """Value counts, stringified keys, insertion order not guaranteed."""
    counts: dict[str, int] = {}
    for label in labels:
        key = str(label)
        counts[key] = counts.get(key, 0) + 1
    return counts


def duplicate_count(texts: list[str]) -> dict[str, int]:
    """Exact-duplicate inventory within one list of texts."""
    seen: dict[str, int] = {}
    for t in texts:
        seen[t] = seen.get(t, 0) + 1
    n_unique = len(seen)
    n_dup_rows = len(texts) - n_unique
    return {"n_rows": len(texts), "n_unique": n_unique, "n_duplicate_rows": n_dup_rows}


def cross_split_duplicate_counts(split_to_texts: dict[str, list[str]]) -> dict[str, int]:
    """For every unordered pair of splits, count of texts appearing (exactly)
    in both. Keys are ``"<a>|<b>"`` with ``a < b`` lexicographically."""
    names = sorted(split_to_texts)
    sets = {name: set(texts) for name, texts in split_to_texts.items()}
    out: dict[str, int] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            out[f"{a}|{b}"] = len(sets[a] & sets[b])
    return out


def empty_or_whitespace_count(texts: list[str]) -> int:
    return sum(1 for t in texts if t.strip() == "")


def zero_bangla_count(texts: list[str]) -> int:
    """Rows with zero characters in the Bengali Unicode block (U+0980-U+09FF)
    -- same rule ``pipeline/s2_prepare.py`` uses to drop rows in M2.1."""
    return sum(1 for t in texts if not _BENGALI_RE.search(t))


def replacement_char_count(texts: list[str]) -> int:
    return sum(1 for t in texts if "�" in t)


def emoji_bearing_count(texts: list[str]) -> int:
    return sum(1 for t in texts if any(_is_emoji_codepoint(ch) for ch in t))


@dataclass
class QualityInventory:
    n_rows: int = 0
    n_unique: int = 0
    n_duplicate_rows: int = 0
    n_empty_or_whitespace: int = 0
    n_zero_bangla: int = 0
    n_replacement_char: int = 0
    n_emoji_bearing: int = 0
    cross_split_duplicates: dict[str, int] = field(default_factory=dict)


def quality_inventory(
    split_to_texts: dict[str, list[str]],
) -> QualityInventory:
    """The full data-quality inventory (s1 raw inspection / s3 recompute),
    over a dataset's texts grouped by split (pass ``{"all": texts}`` if there
    is no split structure to check)."""
    all_texts = [t for texts in split_to_texts.values() for t in texts]
    dup = duplicate_count(all_texts)
    return QualityInventory(
        n_rows=dup["n_rows"],
        n_unique=dup["n_unique"],
        n_duplicate_rows=dup["n_duplicate_rows"],
        n_empty_or_whitespace=empty_or_whitespace_count(all_texts),
        n_zero_bangla=zero_bangla_count(all_texts),
        n_replacement_char=replacement_char_count(all_texts),
        n_emoji_bearing=emoji_bearing_count(all_texts),
        cross_split_duplicates=cross_split_duplicate_counts(split_to_texts)
        if len(split_to_texts) > 1 else {},
    )


# ---------------------------------------------------------------------------
# Label-conditioned checks (e.g. the bd_shs U+FFFD x hate-speech 2x2)
# ---------------------------------------------------------------------------


def flag_vs_label_table(
    texts: list[str],
    labels: list,
    flag_fn,
    label_names: dict | None = None,
) -> dict:
    """Generic 2x2 (or 2xK): label distribution among texts where
    ``flag_fn(text)`` is True vs. False. ``label_names`` maps raw label value
    -> display name (falls back to ``str(label)``)."""
    label_names = label_names or {}
    groups: dict[str, list] = {"flagged": [], "unflagged": []}
    for text, label in zip(texts, labels):
        groups["flagged" if flag_fn(text) else "unflagged"].append(label)
    out = {}
    for key, group_labels in groups.items():
        n = len(group_labels)
        dist = class_distribution(group_labels)
        out[key] = {
            "n": n,
            "label_dist_pct": {
                label_names.get(_coerce_key(k, label_names), k): (100.0 * v / n if n else 0.0)
                for k, v in dist.items()
            },
        }
    return out


def _coerce_key(k: str, label_names: dict):
    """class_distribution stringifies keys; try to recover the original type
    so it matches label_names' keys (usually int)."""
    for candidate in label_names:
        if str(candidate) == k:
            return candidate
    return k


# ---------------------------------------------------------------------------
# Eligible units per noise type (s3 figure: explains conjunct_split etc.)
# ---------------------------------------------------------------------------


def eligible_units_per_noise_type(
    texts: list[str],
    noise_types: tuple[str, ...] = NOISE_TYPES,
    severity: int = MIN_SEVERITY,
    seed: int = 42,
) -> dict[str, list[int]]:
    """Per-text ``n_eligible`` for every graded noise type.

    ``n_eligible`` is a property of the input text's structure (how many
    units of that noise type's kind exist to act on), not of severity or
    seed -- severity only controls how many of the eligible units get
    perturbed. A fixed ``(severity, seed)`` is used purely to drive one
    ``perturb_with_stats`` call per text; the returned counts do not depend
    on which severity/seed was passed (verified in
    ``tests/test_corpus_stats.py``).
    """
    out: dict[str, list[int]] = {nt: [] for nt in noise_types}
    for text in texts:
        for nt in noise_types:
            _, stats = perturb_with_stats(text, nt, severity, seed)
            out[nt].append(stats.n_eligible)
    return out
