"""The ten noise functions for the Bangla noise benchmark.

Public API (METHODOLOGY M1.4):

    from noisebench import perturb, perturb_with_stats

    perturb(text, noise_type, severity, seed) -> str
    perturb_with_stats(text, noise_type, severity, seed) -> (str, PerturbStats)

``noise_type`` is one of ``NOISE_TYPES`` (nine severity-graded types, N1..N9).

N10 (emoji) is **not** severity-graded -- it is a three-way preprocessing
condition -- so it has its own signature:

    emoji_transform(text, mode) -> str
    emoji_transform_with_stats(text, mode) -> (str, PerturbStats)

with ``mode in {"keep", "remove", "replace_with_text"}``.

Design invariants (CLAUDE.md):
  * DETERMINISM. Every function derives all randomness from a local
    ``random.Random`` instance, seeded (via ``_derive_seed``) from *all four*
    arguments ``(text, noise_type, severity, seed)`` so that per-text random
    draws are independent across a corpus. The global ``random`` module is
    never touched.
  * UNICODE SAFETY. All character-level work is on NFC-normalized extended
    grapheme clusters (``regex`` ``\\X``); a whole cluster is the smallest unit
    that is ever moved, deleted or duplicated, so a multi-code-point cluster
    (e.g. ``ড়`` = U+09A1 U+09BC) is never torn in half. The few operations that
    edit *inside* a cluster (N3 base swap, N5, N6) only ever touch a single
    base letter or a single sign, and re-normalize on exit. Every guard that
    skips a unit to avoid emitting an invalid sequence is commented ``# GUARD``.
  * NO SILENT NO-OPS. ``perturb_with_stats`` always reports ``n_eligible``,
    ``n_applied`` and ``unchanged`` so a caller can see when nothing happened.
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

import regex

from .bangla_maps import (
    CHANDRABINDU,
    CHAR_TO_CLASS,
    COMBINING_MARKS,
    CONSONANT_SET,
    EMOJI_CODEPOINT_RANGES,
    EMOJI_REPLACEMENT_FALLBACK,
    EMOJI_SEQUENCE_GLUE,
    HASANTA,
    HOMOPHONE_SETS,
    INDEPENDENT_VOWEL_SET,
    INSERTABLE_CHARS,
    IN_CLASS_POOLS,
    MATRA_SET,
    MATRA_TO_INDEPENDENT_VOWEL,
    N6_ALTER_PAIRS,
    N6_DROP_TARGETS,
)
from .severity import fraction_for_severity, n_to_perturb

__all__ = [
    "PerturbStats",
    "NOISE_TYPES",
    "EMOJI_MODES",
    "clusters",
    "perturb",
    "perturb_with_stats",
    "emoji_transform",
    "emoji_transform_with_stats",
]

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

_GRAPHEME_RE = regex.compile(r"\X")


def _nfc(text: str) -> str:
    """Return the NFC (canonical composition) form of ``text``."""
    return unicodedata.normalize("NFC", text)


def clusters(text: str) -> list[str]:
    """Split ``text`` into Unicode extended grapheme clusters, after NFC.

    This is *the* text-unit function for the whole benchmark (METHODOLOGY
    M1.1). ``কি`` is one cluster, ``ড়`` is one cluster, an emoji ZWJ sequence
    is one cluster.

    Args:
        text: arbitrary input string.

    Returns:
        List of grapheme-cluster strings whose concatenation is ``_nfc(text)``.
    """
    return _GRAPHEME_RE.findall(_nfc(text))


def _is_ws_cluster(cluster: str) -> bool:
    """True if the cluster is whitespace (used to find token boundaries)."""
    return bool(cluster) and all(ch.isspace() for ch in cluster)


def _token_spans(cl: list[str]) -> list[tuple[int, int]]:
    """Return ``(start, end)`` index ranges of maximal non-whitespace runs.

    A "token" in METHODOLOGY M1.2 is one such run of grapheme clusters.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i, c in enumerate(cl):
        if _is_ws_cluster(c):
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(cl)))
    return spans


def _sample_indices(n: int, k: int, rng: random.Random) -> set[int]:
    """Return a set of ``k`` distinct indices from ``range(n)`` (seeded)."""
    if k <= 0:
        return set()
    return set(rng.sample(range(n), k))


def _derive_seed(seed: int, noise_type: str, severity: int | None, text: str) -> int:
    """Fold every argument into the RNG seed.

    ``perturb`` must be a pure function of ``(text, noise_type, severity,
    seed)`` -- but seeding ``random.Random(seed)`` directly makes the FIRST
    random draw (the Bernoulli remainder in ``n_to_perturb``) identical for
    every text at a given seed. The draws are then perfectly correlated across
    a corpus, and M1.3's "expected count = n_eligible * frac over many texts"
    fails badly whenever ``n_eligible * frac < 1`` (most visibly N7 at low
    severity). Hashing the text (and noise_type/severity) into the seed keeps
    determinism while making each text's draw independent.
    """
    payload = f"{seed}\x00{noise_type}\x00{severity}\x00{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Stats contract (METHODOLOGY M1.4)
# ---------------------------------------------------------------------------


@dataclass
class PerturbStats:
    """What actually happened during one ``perturb`` call.

    Attributes:
        noise_type: the noise type applied (``"char_insert"`` ... or
            ``"emoji_transform"``).
        severity: severity level 1..5, or ``None`` for ``emoji_transform``.
        n_eligible: number of eligible units found in the input
            (METHODOLOGY M1.2). ``0`` means the text had nothing this noise
            type could act on.
        n_requested: how many units the severity fraction asked for --
            ``n_to_perturb(n_eligible, frac, rng)`` (METHODOLOGY M1.3), capped
            for N4 at the token set's disjoint-pair capacity. It is the number
            of units the call actually set out to perturb.
        n_applied: how many units were *actually* perturbed. Equals
            ``n_requested`` for every noise type except N2, where a selection is
            dropped rather than empty a single-cluster token -- a shortfall that
            is mathematically unavoidable given that guard, not an
            implementation choice (see METHODOLOGY M1.3). The
            ``n_applied / n_requested`` ratio, together with
            ``n_requested / (n_eligible * frac)``, tells you whether "severity =
            fraction of eligible units" holds for a given type; see
            ``validate.achieved_rate_table``.
        submodes: per-mechanism counts, e.g. ``{"insert_vowel": 2,
            "repeat_final": 1}`` for N8. Lets the three mechanisms of N8 (and
            the split/merge of N9) be separated in analysis. Single-mechanism
            types still report one key (e.g. ``{"drop_hasanta": n}`` for N7).
        unchanged: ``True`` iff the output string equals the NFC-normalized
            input byte-for-byte. Can be ``True`` even when ``n_applied > 0``
            (e.g. transposing two identical clusters) -- that is a real,
            observable no-op, not a hidden one.
    """

    noise_type: str
    severity: int | None
    n_eligible: int
    n_requested: int
    n_applied: int
    submodes: dict[str, int] = field(default_factory=dict)
    unchanged: bool = True


# ---------------------------------------------------------------------------
# Group A -- generic character-level (N1..N4)
# ---------------------------------------------------------------------------


def _n1_char_insert(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N1: insert a random standalone Bangla character at a cluster boundary.

    Eligible unit: an inter-cluster boundary strictly inside a token of >= 2
    clusters (never at a token edge -- see METHODOLOGY M1.2). Inserted
    characters come from ``INSERTABLE_CHARS`` (consonants, independent vowels,
    digits, ং/ঃ/ৎ); never a matra or hasanta (GUARD: a combining mark at a
    random boundary would create a degenerate sequence).
    """
    cl = clusters(text)
    eligible: list[int] = []  # insert *after* cluster index p
    for start, end in _token_spans(cl):
        if end - start >= 2:
            eligible.extend(range(start, end - 1))
    k = n_to_perturb(len(eligible), frac, rng)
    chosen = {eligible[i] for i in _sample_indices(len(eligible), k, rng)}

    out: list[str] = []
    applied = 0
    for i, c in enumerate(cl):
        out.append(c)
        if i in chosen:
            out.append(rng.choice(INSERTABLE_CHARS))
            applied += 1
    return "".join(out), len(eligible), k, applied, {"insert": applied}


def _n2_char_delete(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N2: delete a grapheme cluster.

    Eligible unit: a cluster inside a token of >= 2 clusters. GUARD: a token is
    never emptied -- if every cluster of a token is selected, one is randomly
    kept (so a single-cluster token can never vanish, per METHODOLOGY M1.2).
    """
    cl = clusters(text)
    spans = _token_spans(cl)
    eligible = [i for (s, e) in spans if e - s >= 2 for i in range(s, e)]
    k = n_to_perturb(len(eligible), frac, rng)
    chosen = {eligible[i] for i in _sample_indices(len(eligible), k, rng)}

    for s, e in spans:
        in_tok = [i for i in range(s, e) if i in chosen]
        if e - s >= 2 and len(in_tok) == e - s:  # GUARD: would empty the token
            chosen.discard(rng.choice(in_tok))

    out = [c for i, c in enumerate(cl) if i not in chosen]
    applied = len(chosen)
    return "".join(out), len(eligible), k, applied, {"delete": applied}


def _n3_base_class(cluster: str) -> tuple[int, str] | None:
    """Return ``(base_len, class_name)`` for an N3-eligible cluster, else ``None``.

    ``base_len`` is the length (in code points) of the leading base letter that
    N3 will swap: ``2`` for the nukta letters ড়/ঢ়/য় (base + U+09BC, which NFC
    keeps decomposed), otherwise ``1``. GUARD: taking the 2-code-point prefix
    for nukta letters is what stops N3 from replacing only ``ড`` and leaving an
    orphaned ``়`` on the wrong consonant.
    """
    for base_len in (2, 1):
        if cluster[:base_len] in CHAR_TO_CLASS:
            return base_len, CHAR_TO_CLASS[cluster[:base_len]]
    return None


def _n3_char_substitute(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N3: replace a cluster's base character with another of the same class.

    In-class substitution (METHODOLOGY M0.1): consonant->consonant,
    independent-vowel->independent-vowel, digit->digit, drawn uniformly. No
    keyboard-layout dependency. Matras, hasanta and signs have no class and are
    not eligible.
    """
    cl = clusters(text)
    eligible = [i for i, c in enumerate(cl) if _n3_base_class(c) is not None]
    k = n_to_perturb(len(eligible), frac, rng)
    chosen = {eligible[i] for i in _sample_indices(len(eligible), k, rng)}

    out = list(cl)
    applied = 0
    for i in sorted(chosen):
        base_len, class_name = _n3_base_class(cl[i])  # type: ignore[misc]
        current = cl[i][:base_len]
        pool = [ch for ch in IN_CLASS_POOLS[class_name] if ch != current]
        repl = rng.choice(pool)
        out[i] = repl + cl[i][base_len:]
        applied += 1
    return "".join(out), len(eligible), k, applied, {"substitute": applied}


def _n4_char_transpose(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N4: swap two adjacent grapheme clusters within one token.

    Eligible unit: an adjacent cluster pair inside a token, identified by its
    left index (there are ``token_len - 1`` per token). The count rule picks
    ``k``; N4 then draws ``k`` *pairwise-disjoint* pairs from a pool that
    shrinks as pairs are taken -- pick a random surviving pair, apply the swap,
    then remove that pair and its two neighbours from the pool. This is the
    correct construction for ``k`` independent non-overlapping adjacent
    transpositions and, unlike the older "``k`` global picks, skip overlaps"
    selection (which fell ~22% short of nominal at severity 5 because
    independent picks collide), it reaches the nominal fraction.

    GUARD: no cluster is moved twice (the neighbour removal enforces it).
    ``n_applied`` can fall short of ``k`` only if the pool empties first -- a
    real limit on how many disjoint swaps a short token set admits, reported
    honestly in the stats, never a silent no-op.
    """
    cl = clusters(text)
    eligible = [
        i for (s, e) in _token_spans(cl) if e - s >= 2 for i in range(s, e - 1)
    ]
    k = n_to_perturb(len(eligible), frac, rng)

    pool = set(eligible)
    out = list(cl)
    applied = 0
    while pool and applied < k:
        i = rng.choice(sorted(pool))
        pool.difference_update((i - 1, i, i + 1))  # drop the pair and its neighbours
        out[i], out[i + 1] = out[i + 1], out[i]
        applied += 1
    return "".join(out), len(eligible), k, applied, {"transpose": applied}


# ---------------------------------------------------------------------------
# Group B -- Bangla-specific orthographic (N5..N8)
# ---------------------------------------------------------------------------

# member string -> the confusable set it belongs to
_HOMOPHONE_MEMBER_TO_SET: dict[str, tuple[str, ...]] = {}
for _s in HOMOPHONE_SETS:
    for _m in _s:
        _HOMOPHONE_MEMBER_TO_SET[_m] = _s

_NUKTA = "়"  # U+09BC


def _homophone_occurrences(cl: list[str]) -> list[tuple[int, int, str]]:
    """Find non-overlapping homophone-member occurrences.

    Returns ``(cluster_index, offset_in_cluster, member_string)`` tuples in
    reading order. GUARD: a base letter directly followed by U+09BC is a
    *distinct* nukta letter (ড় ঢ় য়), so it is only ever matched as the whole
    2-code-point unit -- a bare ``য`` inside ``য়`` is never treated as জ/য.
    """
    occ: list[tuple[int, int, str]] = []
    for ci, c in enumerate(cl):
        j = 0
        while j < len(c):
            if c[j + 1 : j + 2] == _NUKTA:  # base + nukta = one letter
                pair = c[j : j + 2]
                if pair in _HOMOPHONE_MEMBER_TO_SET:
                    occ.append((ci, j, pair))
                j += 2
                continue
            if c[j] in _HOMOPHONE_MEMBER_TO_SET:
                occ.append((ci, j, c[j]))
            j += 1
    return occ


def _n5_homophone_confuse(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N5: swap a character for another member of its confusable set.

    Sets are in ``HOMOPHONE_SETS`` (শ/ষ/স, ন/ণ, ই/ঈ, উ/ঊ, ি/ী, ু/ূ, র/ড়/ঢ়,
    জ/য). Eligible unit: one member occurrence. Occurrences inside the same
    cluster are rewritten right-to-left so earlier offsets stay valid.

    ``submodes`` is keyed **per confusable set** (e.g. ``"শ/ষ/স": 2``), not by
    a single ``"homophone"`` key -- this is what lets METHODOLOGY M2.4's
    per-set firing report (which sets actually fire on real corpora) read
    straight off ``PerturbStats`` with no extra instrumentation.
    """
    cl = clusters(text)
    occ = _homophone_occurrences(cl)
    k = n_to_perturb(len(occ), frac, rng)
    chosen = _sample_indices(len(occ), k, rng)

    out = list(cl)
    applied = 0
    submodes: dict[str, int] = defaultdict(int)
    for oi in sorted(chosen, reverse=True):  # descending -> right-to-left in cluster
        ci, j, member = occ[oi]
        homoset = _HOMOPHONE_MEMBER_TO_SET[member]
        pool = [ch for ch in homoset if ch != member]
        repl = rng.choice(pool)
        out[ci] = out[ci][:j] + repl + out[ci][j + len(member) :]
        submodes["/".join(homoset)] += 1
        applied += 1
    return "".join(out), len(occ), k, applied, dict(submodes)


_N6_ALTER: dict[str, str] = {}
for _a, _b in N6_ALTER_PAIRS:
    _N6_ALTER[_a] = _b
    _N6_ALTER[_b] = _a
_N6_TARGET_SET = frozenset(N6_DROP_TARGETS)


def _n6_matra_perturb(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N6: drop or alter a dependent vowel sign, hasanta, or chandrabindu.

    Eligible unit: one occurrence of any character in ``N6_DROP_TARGETS``
    (all 10 matras + ্ + ঁ). Default action is to DROP it; for the length
    pairs ি/ী and ু/ূ the RNG instead ALTERs it (50/50). Everything else is
    drop-only (METHODOLOGY M0 Q5 / decision 3). GUARD: if the target is the
    whole cluster (malformed leading mark in the input), dropping just removes
    that cluster -- it can never leave an orphaned combining mark.
    """
    cl = clusters(text)
    occ = [
        (ci, j, ch)
        for ci, c in enumerate(cl)
        for j, ch in enumerate(c)
        if ch in _N6_TARGET_SET
    ]
    k = n_to_perturb(len(occ), frac, rng)
    chosen = _sample_indices(len(occ), k, rng)

    out = list(cl)
    submodes: dict[str, int] = defaultdict(int)
    applied = 0
    for oi in sorted(chosen, reverse=True):
        ci, j, ch = occ[oi]
        if ch in _N6_ALTER and rng.random() < 0.5:
            out[ci] = out[ci][:j] + _N6_ALTER[ch] + out[ci][j + 1 :]
            submodes["alter"] += 1
        else:
            out[ci] = out[ci][:j] + out[ci][j + 1 :]
            if ch == HASANTA:
                submodes["drop_hasanta"] += 1
            elif ch == CHANDRABINDU:
                submodes["drop_chandrabindu"] += 1
            else:
                submodes["drop_matra"] += 1
        applied += 1
    return "".join(out), len(occ), k, applied, dict(submodes)


_CONSONANT_CP = frozenset(ch for ch in CONSONANT_SET if len(ch) == 1)


def _n7_conjunct_split(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N7: break a two-consonant conjunct (juktakkhor) by dropping its hasanta.

    Eligible unit: a ``C + HASANTA + C`` junction that is the *only* hasanta in
    its cluster (GUARD: triple+ conjuncts such as স্ত্র are skipped entirely --
    METHODOLOGY M0 Q6). Single mechanism (revised 2026-09-01, M0 Q6): remove
    the hasanta so the two consonants no longer join (ক্ষ -> কষ, বিদ্যা ->
    বিদযা). ZWNJ insertion was dropped -- it is a deliberate rendering choice,
    not a failure to form the conjunct, which is what N7 models.
    """
    cl = clusters(text)
    junctions: list[tuple[int, int]] = []
    for ci, c in enumerate(cl):
        hpos = [j for j, ch in enumerate(c) if ch == HASANTA]
        if len(hpos) != 1:
            continue  # GUARD: 0 hasanta = not a conjunct; 2+ = triple, excluded
        j = hpos[0]
        if 0 < j < len(c) - 1 and c[j - 1] in _CONSONANT_CP and c[j + 1] in _CONSONANT_CP:
            junctions.append((ci, j))

    k = n_to_perturb(len(junctions), frac, rng)
    chosen = _sample_indices(len(junctions), k, rng)

    out = list(cl)
    applied = 0
    for idx in sorted(chosen, reverse=True):
        ci, j = junctions[idx]
        c = out[ci]
        out[ci] = c[:j] + c[j + 1 :]
        applied += 1
    return "".join(out), len(junctions), k, applied, {"drop_hasanta": applied}


def _n8_modes_for(cluster: str) -> list[str]:
    """Which N8 submodes can act on this cluster (order fixed for determinism)."""
    modes: list[str] = []
    if any(ch in MATRA_TO_INDEPENDENT_VOWEL for ch in cluster):
        modes.append("insert_vowel")
    if cluster in INDEPENDENT_VOWEL_SET:
        modes.append("repeat_vowel")
    if any(ch in MATRA_SET for ch in cluster) or any(
        ch in INDEPENDENT_VOWEL_SET for ch in cluster
    ):
        modes.append("repeat_final")
    return modes


def _n8_elongate(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N8: lengthen a vowel for emphasis (খুব -> খুউব, ভালো -> ভালোওও).

    Eligible unit: a cluster carrying a matra or an independent vowel. Three
    RNG-chosen submodes (METHODOLOGY M0 Q7):
      * ``insert_vowel``  -- insert, ONCE, the independent vowel that matches a
        matra, right after that matra (ু -> উ), so খু -> খুউ. This is a
        distinct real spelling (খুউব for খুব), not a repetition, so it adds
        exactly one vowel.
      * ``repeat_vowel``  -- repeat an independent-vowel cluster 2-4x total
        (আ -> আআ / আআআ / আআআআ).
      * ``repeat_final``  -- repeat the cluster's LAST vowel/matra code point
        2-4x total (খু -> খুু / খুুু / খুুুু). GUARD: only a matra or
        independent-vowel code point is ever repeated -- never a trailing
        consonant or sign (ং/ঃ/ৎ), which would not be a real elongation.

    Repeating a matra keeps the grapheme cluster intact, so at the cluster
    level (the CER unit, M1.5) ``repeat_final`` and ``insert_vowel`` are a
    one-cluster edit and ``repeat_vowel`` is a 1-3 cluster edit. That is a
    correct, deliberate consequence of measuring CER over clusters.
    """
    cl = clusters(text)
    modes_by_index = {i: _n8_modes_for(c) for i, c in enumerate(cl)}
    eligible = [i for i, m in modes_by_index.items() if m]
    k = n_to_perturb(len(eligible), frac, rng)
    chosen = {eligible[i] for i in _sample_indices(len(eligible), k, rng)}

    out: list[str] = []
    submodes: dict[str, int] = defaultdict(int)
    applied = 0
    for i, c in enumerate(cl):
        if i not in chosen:
            out.append(c)
            continue
        mode = rng.choice(modes_by_index[i])
        if mode == "insert_vowel":
            mpos = next(
                idx for idx, ch in enumerate(c) if ch in MATRA_TO_INDEPENDENT_VOWEL
            )
            out.append(
                c[: mpos + 1] + MATRA_TO_INDEPENDENT_VOWEL[c[mpos]] + c[mpos + 1 :]
            )
        elif mode == "repeat_vowel":
            out.append(c * rng.randint(2, 4))
        else:  # repeat_final
            vpos = max(
                idx
                for idx, ch in enumerate(c)
                if ch in MATRA_SET or ch in INDEPENDENT_VOWEL_SET
            )
            extra = c[vpos] * (rng.randint(2, 4) - 1)
            out.append(c[: vpos + 1] + extra + c[vpos + 1 :])
        submodes[mode] += 1
        applied += 1
    return "".join(out), len(eligible), k, applied, dict(submodes)


# ---------------------------------------------------------------------------
# Group C -- surface-level (N9); N10 is separate, below
# ---------------------------------------------------------------------------

_WS_SPLIT_RE = regex.compile(r"(\s+)")


def _n9_whitespace_error(
    text: str, frac: float, rng: random.Random
) -> tuple[str, int, int, dict[str, int]]:
    """N9: split one word into two, or merge two adjacent words.

    Eligible unit (one combined pool, METHODOLOGY M1.2):
      * a split point -- an internal cluster boundary of a token of >= 2
        clusters (GUARD: a boundary is skipped if the right-hand piece would
        start with a combining mark);
      * a merge point -- the whitespace between two non-empty tokens.
    """
    parts = _WS_SPLIT_RE.split(text)
    tokens = parts[0::2]
    seps = parts[1::2]
    token_clusters = [clusters(t) for t in tokens]

    split_events: list[tuple[int, int]] = []
    for ti, tc in enumerate(token_clusters):
        for pos in range(1, len(tc)):
            if tc[pos][:1] in COMBINING_MARKS:  # GUARD
                continue
            split_events.append((ti, pos))
    merge_events = [
        si
        for si in range(len(seps))
        if tokens[si] != "" and si + 1 < len(tokens) and tokens[si + 1] != ""
    ]

    eligible: list[tuple[str, object]] = [("split", e) for e in split_events]
    eligible += [("merge", m) for m in merge_events]
    k = n_to_perturb(len(eligible), frac, rng)
    chosen = _sample_indices(len(eligible), k, rng)

    splits_by_token: dict[int, list[int]] = defaultdict(list)
    merges: set[int] = set()
    for idx in chosen:
        kind, payload = eligible[idx]
        if kind == "split":
            ti, pos = payload  # type: ignore[misc]
            splits_by_token[ti].append(pos)
        else:
            merges.add(payload)  # type: ignore[arg-type]

    rebuilt: list[str] = []
    for ti, tok in enumerate(tokens):
        positions = sorted(splits_by_token.get(ti, []))
        if positions:
            tc = token_clusters[ti]
            pieces = []
            last = 0
            for pos in positions:
                pieces.append("".join(tc[last:pos]))
                last = pos
            pieces.append("".join(tc[last:]))
            rebuilt.append(" ".join(pieces))
        else:
            rebuilt.append(tok)
        if ti < len(seps):
            rebuilt.append("" if ti in merges else seps[ti])

    n_split = sum(len(v) for v in splits_by_token.values())
    submodes = {"split": n_split, "merge": len(merges)}
    return "".join(rebuilt), len(eligible), k, n_split + len(merges), submodes


# ---------------------------------------------------------------------------
# Registry + public dispatch
# ---------------------------------------------------------------------------

_NOISE_FUNCS: dict[str, object] = {
    "char_insert": _n1_char_insert,
    "char_delete": _n2_char_delete,
    "char_substitute": _n3_char_substitute,
    "char_transpose": _n4_char_transpose,
    "homophone_confuse": _n5_homophone_confuse,
    "matra_perturb": _n6_matra_perturb,
    "conjunct_split": _n7_conjunct_split,
    "elongate": _n8_elongate,
    "whitespace_error": _n9_whitespace_error,
}

#: The nine severity-graded noise types, in canonical order (N1..N9).
NOISE_TYPES: tuple[str, ...] = tuple(_NOISE_FUNCS)

#: The three emoji-transform modes (N10).
EMOJI_MODES: tuple[str, ...] = ("keep", "remove", "replace_with_text")


def perturb_with_stats(
    text: str, noise_type: str, severity: int, seed: int
) -> tuple[str, PerturbStats]:
    """Apply one severity-graded noise type and report what happened.

    Args:
        text: input string (any Unicode; it is NFC-normalized on entry).
        noise_type: one of ``NOISE_TYPES``.
        severity: integer 1..5 (maps to a fraction via ``severity.py``).
        seed: integer seed; the entire call is a pure function of
            ``(text, noise_type, severity, seed)``.

    Returns:
        ``(output_text, stats)``. ``output_text`` is NFC-normalized.

    Raises:
        ValueError: if ``noise_type`` is unknown (including
            ``"emoji_transform"``, which is not severity-graded -- use
            ``emoji_transform``), or if ``severity`` is not in 1..5.
    """
    if noise_type == "emoji_transform":
        raise ValueError(
            "emoji_transform is not severity-graded; call "
            "emoji_transform(text, mode) / emoji_transform_with_stats(text, mode)"
        )
    if noise_type not in _NOISE_FUNCS:
        raise ValueError(
            f"unknown noise_type {noise_type!r}; expected one of {NOISE_TYPES}"
        )
    frac = fraction_for_severity(severity)
    base = _nfc(text)
    rng = random.Random(_derive_seed(seed, noise_type, severity, base))
    func = _NOISE_FUNCS[noise_type]
    out, n_eligible, n_requested, n_applied, submodes = func(base, frac, rng)  # type: ignore[operator]
    out = _nfc(out)
    stats = PerturbStats(
        noise_type=noise_type,
        severity=severity,
        n_eligible=n_eligible,
        n_requested=n_requested,
        n_applied=n_applied,
        submodes=submodes,
        unchanged=(out == base),
    )
    return out, stats


def perturb(text: str, noise_type: str, severity: int, seed: int) -> str:
    """Apply one severity-graded noise type; return only the perturbed text.

    Thin wrapper over :func:`perturb_with_stats` -- see it for argument
    semantics and the determinism guarantee.
    """
    return perturb_with_stats(text, noise_type, severity, seed)[0]


# ---------------------------------------------------------------------------
# N10 -- emoji_transform (NOT severity-graded; own signature)
# ---------------------------------------------------------------------------
#
# N10 is a three-way *preprocessing condition*, not a graded corruption: a
# practitioner picks ONE of keep / remove / replace_with_text and applies it to
# the whole corpus. There is no "fraction of emoji to remove" that anyone would
# actually use, so N10 takes ``mode`` instead of ``(severity, seed)`` and needs
# no RNG -- it is fully deterministic by construction.

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_EMOJI_GLUE_SET = frozenset(EMOJI_SEQUENCE_GLUE)


def _is_emoji_codepoint(ch: str) -> bool:
    """True if ``ch`` falls in one of ``EMOJI_CODEPOINT_RANGES``."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in EMOJI_CODEPOINT_RANGES)


def _emoji_label(ch: str) -> str:
    """``😀`` -> ``:grinning_face:`` via ``unicodedata.name`` (stdlib only)."""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return EMOJI_REPLACEMENT_FALLBACK
    slug = _NON_ALNUM_RE.sub("_", name.lower()).strip("_")
    return f":{slug}:" if slug else EMOJI_REPLACEMENT_FALLBACK


def emoji_transform_with_stats(text: str, mode: str) -> tuple[str, PerturbStats]:
    """Apply one emoji preprocessing condition (N10) and report what happened.

    Args:
        text: input string (NFC-normalized on entry and exit).
        mode: one of ``EMOJI_MODES``:
            * ``"keep"``               -- identity (control condition);
            * ``"remove"``             -- strip every emoji code point and its
              sequence glue (ZWJ, variation selectors, enclosing keycap), then
              collapse the whitespace runs this leaves behind and strip the
              ends, so ``remove`` differs from ``keep`` only by the emoji
              (not by stray double spaces where an emoji used to sit);
            * ``"replace_with_text"``  -- replace each emoji code point with a
              ``:name:`` label from the Unicode database; glue is dropped.

    Returns:
        ``(output_text, stats)`` with ``stats.severity is None`` and
        ``stats.submodes == {mode: n_applied}``.

    Raises:
        ValueError: if ``mode`` is not in ``EMOJI_MODES``.
    """
    if mode not in EMOJI_MODES:
        raise ValueError(f"unknown emoji mode {mode!r}; expected one of {EMOJI_MODES}")

    base = _nfc(text)
    chars = list(base)
    n_emoji = sum(1 for ch in chars if _is_emoji_codepoint(ch))

    if mode == "keep":
        out = base
        applied = 0
    elif mode == "remove":
        stripped = "".join(
            ch
            for ch in chars
            if not _is_emoji_codepoint(ch) and ch not in _EMOJI_GLUE_SET
        )
        # GUARD/decision (2026-09-01): collapse the whitespace runs left behind
        # by removed emoji, so `remove` differs from `keep` only by the emoji.
        out = re.sub(r"\s+", " ", stripped).strip()
        applied = n_emoji
    else:  # replace_with_text
        buf: list[str] = []
        for ch in chars:
            if _is_emoji_codepoint(ch):
                buf.append(_emoji_label(ch))
            elif ch in _EMOJI_GLUE_SET:
                continue
            else:
                buf.append(ch)
        out = "".join(buf)
        applied = n_emoji

    out = _nfc(out)
    stats = PerturbStats(
        noise_type="emoji_transform",
        severity=None,
        n_eligible=n_emoji,
        n_requested=applied,  # N10 has no sampling step: request == apply
        n_applied=applied,
        submodes={mode: applied},
        unchanged=(out == base),
    )
    return out, stats


def emoji_transform(text: str, mode: str) -> str:
    """Apply one emoji preprocessing condition (N10); return only the text.

    Thin wrapper over :func:`emoji_transform_with_stats`.
    """
    return emoji_transform_with_stats(text, mode)[0]
