"""Linguistic resources for the Bangla noise benchmark.

This module contains **only data** (plus one small pure helper,
``build_adjacency_from_layout``, kept for the optional layout-conditioned path
described in METHODOLOGY M0.1 — it is not used by the shipped noise suite).
No perturbation logic lives here.

Every constant below is annotated with a ``# TODO(verify):`` block describing:
  * what the constant is supposed to contain,
  * where the values came from,
  * exactly how you (or a native speaker) should check it.

Anything that is a *guess* rather than a transcription of a source you gave me
is named with a ``CANDIDATE_`` prefix and must be confirmed before use.

Unicode conventions used throughout:
  * Characters are written literally in the source (the file is UTF-8) with the
    code point in a trailing comment, e.g. ``"ক"  # U+0995 KA``.
  * "matra" / "kar" = dependent vowel sign (combining).
  * "hasanta" / "hosonto" / virama = U+09CD, the conjunct-forming joiner.


# =========================================================================
# M0 — RESOLVED DECISIONS (from METHODOLOGY.md §M0; do not revisit)
# =========================================================================
#
# M0.1  N3 is IN-CLASS substitution, not keyboard-adjacency. No Bijoy grid
#       ships. A character is replaced by another uniformly drawn from its own
#       orthographic class (see IN_CLASS_POOLS). Layout-conditioned N3b is an
#       optional future add-on; build_adjacency_from_layout() is kept for it.
# Q2 :  N1 never inserts a combining mark (matra/hasanta). N6 owns matras.
# Q3 :  N5 merges the matra forms — ি↔ী and ু↔ূ are first-class HOMOPHONE_SETS
#       entries, because the dependent forms are far more frequent in running
#       text than the independent letters ই/ঈ, উ/ঊ.
# Q4 :  N6 drop targets = ALL dependent vowel signs (া ৃ ৈ included — া is the
#       most frequent matra) plus hasanta, plus ঁ chandrabindu (dropping it,
#       চাঁদ → চাদ, is one of the most common real Bangla omissions).
# Q5 :  N6 may only *alter* within N6_ALTER_PAIRS = {ি↔ী, ু↔ূ}. These model
#       phonetic confusion — Bangla speech has no vowel-length distinction, so
#       the i/u length pairs are genuinely confused. ে/ৈ and ো/ৌ are
#       phonetically distinct (only visually similar), so their stroke loss/gain
#       is left to N1/N2; ে ো and every other target are drop-only.
# Q6 :  N7 does both — drop hasanta OR insert ZWNJ — RNG-chosen, recorded in
#       PerturbStats.submodes. Triple conjuncts (স্ত্র) excluded for now.
# Q7 :  N8 has three submodes — insert independent vowel after a matra, repeat
#       an independent vowel, repeat the final character — RNG-chosen, recorded.
# Q8 :  ঋ is included; archaic vocalic letters ঌ (U+098C) and ৡ (U+09E1) are
#       excluded by design.
# Q9 :  Emoji ranges are provisional; M2 measures their coverage vs a reference
#       list and widens them if coverage < ~90%.
# Q10:  NFC normalization is applied at entry and exit of every perturbation.
#       validate.py explicitly tests that ো and ৌ survive an NFC round trip and
#       a BanglaBERT-normalizer round trip.
# =========================================================================
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 0. Invisible / structural code points
# ---------------------------------------------------------------------------

# TODO(verify): these three are Unicode infrastructure, not Bangla letters.
#   HASANTA (U+09CD) is the virama: <consonant, HASANTA, consonant> is how a
#   conjunct (juktakkhor) is encoded. ZWNJ/ZWJ control whether that sequence
#   renders as a ligature. Check: U+09CD is named "BENGALI SIGN VIRAMA" in the
#   Unicode charts; ZWNJ = U+200C, ZWJ = U+200D.
HASANTA = "্"          # U+09CD BENGALI SIGN VIRAMA (hasanta / hosonto)
ZWNJ = "‌"             # U+200C ZERO WIDTH NON-JOINER
ZWJ = "‍"              # U+200D ZERO WIDTH JOINER


# ---------------------------------------------------------------------------
# 1. Character inventory
# ---------------------------------------------------------------------------

# TODO(verify): the 39 basic Bangla consonant letters, in traditional order,
#   plus the three "dotted" letters formed with nukta (down, drho, yo). Check
#   against the Unicode Bengali block (U+0985..U+09FF) or any Bangla varnamala
#   chart. Note: ৱ (U+09F1) and other Assamese/regional letters are NOT here on
#   purpose -- this benchmark is Bangla (Bangladesh standard). Flag if you want
#   ৰ/ৱ included.
BANGLA_CONSONANTS: tuple[str, ...] = (
    "ক",  # U+0995 KA    ক
    "খ",  # U+0996 KHA   খ
    "গ",  # U+0997 GA    গ
    "ঘ",  # U+0998 GHA   ঘ
    "ঙ",  # U+0999 NGA   ঙ
    "চ",  # U+099A CA    চ
    "ছ",  # U+099B CHA   ছ
    "জ",  # U+099C JA    জ
    "ঝ",  # U+099D JHA   ঝ
    "ঞ",  # U+099E NYA   ঞ
    "ট",  # U+099F TTA   ট
    "ঠ",  # U+09A0 TTHA  ঠ
    "ড",  # U+09A1 DDA   ড
    "ঢ",  # U+09A2 DDHA  ঢ
    "ণ",  # U+09A3 NNA   ণ
    "ত",  # U+09A4 TA    ত
    "থ",  # U+09A5 THA   থ
    "দ",  # U+09A6 DA    দ
    "ধ",  # U+09A7 DHA   ধ
    "ন",  # U+09A8 NA    ন
    "প",  # U+09AA PA    প
    "ফ",  # U+09AB PHA   ফ
    "ব",  # U+09AC BA    ব  (also used for VA / w-sound)
    "ভ",  # U+09AD BHA   ভ
    "ম",  # U+09AE MA    ম
    "য",  # U+09AF YA    য  (ôntôsthô jô)
    "র",  # U+09B0 RA    র
    "ল",  # U+09B2 LA    ল
    "শ",  # U+09B6 SHA   শ  (talobyo shô)
    "ষ",  # U+09B7 SSA   ষ  (murdhonyo shô)
    "স",  # U+09B8 SA    স  (dontyo shô)
    "হ",  # U+09B9 HA    হ
    "ড়",  # U+09DC RRA   ড়  (= U+09A1 + U+09BC nukta, precomposed)
    "ঢ়",  # U+09DD RHA   ঢ়  (= U+09A2 + U+09BC nukta, precomposed)
    "য়",  # U+09DF YYA   য়  (= U+09AF + U+09BC nukta, precomposed)
)

# TODO(verify): the 11 independent (full) vowel letters used in modern Bangla.
#   Check: U+098B ঋ is included (appears in tatsama words like ঋণ, ঋতু).
#   M0 Q8: archaic vocalic letters ঌ (U+098C) and ৡ (U+09E1), and vocalic
#   RR (U+09E0), are excluded by design.
BANGLA_INDEPENDENT_VOWELS: tuple[str, ...] = (
    "অ",  # U+0985 A    অ
    "আ",  # U+0986 AA   আ
    "ই",  # U+0987 I    ই
    "ঈ",  # U+0988 II   ঈ
    "উ",  # U+0989 U    উ
    "ঊ",  # U+098A UU   ঊ
    "ঋ",  # U+098B VOCALIC R  ঋ
    "এ",  # U+098F E    এ
    "ঐ",  # U+0990 AI   ঐ
    "ও",  # U+0993 O    ও
    "ঔ",  # U+0994 AU   ঔ
)

# TODO(verify): dependent vowel signs (matra / kar). Order follows the Unicode
#   block. Check each renders as a combining sign, not a standalone letter.
#   IMPORTANT ENCODING NOTE: ো (U+09CB) and ৌ (U+09CC) are single code points
#   here, but a normalizer may decompose them to <U+09C7, U+09BE> and
#   <U+09C7, U+09D7> respectively. validate.py must handle both forms (M0 Q10).
MATRA_SIGNS: tuple[str, ...] = (
    "া",  # U+09BE AA sign   া
    "ি",  # U+09BF I sign    ি   (renders to the LEFT of its consonant)
    "ী",  # U+09C0 II sign   ী
    "ু",  # U+09C1 U sign    ু
    "ূ",  # U+09C2 UU sign   ূ
    "ৃ",  # U+09C3 VOCALIC R sign  ৃ
    "ে",  # U+09C7 E sign    ে   (renders to the LEFT)
    "ৈ",  # U+09C8 AI sign   ৈ   (renders to the LEFT)
    "ো",  # U+09CB O sign    ো   (wraps: left + right)
    "ৌ",  # U+09CC AU sign   ৌ   (wraps: left + right)
)

# TODO(verify): non-vowel signs that attach to / follow a consonant.
#   ং anusvara, ঃ visarga, ঁ chandrabindu, ৎ khanda ta.
#   These are relevant to N1/N2 (they are deletable/insertable units) and to
#   the validator (a matra must not attach directly to most of them).
BANGLA_SIGNS: tuple[str, ...] = (
    "ঁ",  # U+0981 CHANDRABINDU  ঁ
    "ং",  # U+0982 ANUSVARA      ং
    "ঃ",  # U+0983 VISARGA       ঃ
    "ৎ",  # U+09CE KHANDA TA     ৎ
)

# TODO(verify): Bangla digits ০-৯ (U+09E6..U+09EF), in value order 0..9.
BANGLA_DIGITS: tuple[str, ...] = (
    "০", "১", "২", "৩", "৪",  # ০ ১ ২ ৩ ৪
    "৫", "৬", "৭", "৮", "৯",  # ৫ ৬ ৭ ৮ ৯
)

# TODO(verify): the pool N1 (char_insert) draws from. M0 Q2: insert only
#   spacing "standalone" characters -- consonants, independent vowels, and the
#   spacing post-base signs ং ঃ ৎ -- but NOT bare matras, hasanta, or
#   ঁ chandrabindu, because inserting a non-spacing combining mark at a random
#   position frequently produces an invalid/degenerate sequence (matra after a
#   space, doubled matra). N6 is the noise type that manipulates matras.
#   DIGITS (২০২৪ …) are ALSO excluded (decision 2026-09-01): with the
#   keyboard-adjacency model gone there is no mechanism that would drop a digit
#   inside a word, and প্রতিদিন -> প্রতি৬দিন is not a plausible slip. Digits stay
#   eligible for N2 (delete) and N3 (digit -> digit).
#   Note: ং/ঃ still attach to whatever precedes them, but N1 only ever inserts
#   at a token-internal boundary, so a base character always precedes.
_INSERTABLE_SIGNS: tuple[str, ...] = ("ং", "ঃ", "ৎ")  # U+0982, U+0983, U+09CE
INSERTABLE_CHARS: tuple[str, ...] = (
    BANGLA_CONSONANTS + BANGLA_INDEPENDENT_VOWELS + _INSERTABLE_SIGNS
)


# ---------------------------------------------------------------------------
# 2. N3 char_substitute -- in-class substitution pools (M0.1)
# ---------------------------------------------------------------------------

# TODO(verify): N3 replaces a grapheme cluster's base character with another
#   character drawn uniformly from the SAME class. Classes are mutually
#   exclusive and cover only characters that have a meaningful same-class peer:
#     * "consonant"        -> any other consonant
#     * "independent_vowel"-> any other independent vowel
#     * "digit"            -> any other digit
#   Matras, hasanta, signs, punctuation and non-Bangla characters have NO pool
#   and are therefore not eligible for N3 (see METHODOLOGY M1.2). Check that
#   this partition matches how you want N3 to behave; in particular confirm you
#   do NOT want matra->matra substitution here (that lives in N6 as N6_ALTER).
IN_CLASS_POOLS: dict[str, tuple[str, ...]] = {
    "consonant": BANGLA_CONSONANTS,
    "independent_vowel": BANGLA_INDEPENDENT_VOWELS,
    "digit": BANGLA_DIGITS,
}


# ---------------------------------------------------------------------------
# 3. N5 homophone_confuse -- confusable sets
# ---------------------------------------------------------------------------

# TODO(verify): confusable sets for N5. A perturbation replaces a character
#   with another member of the same set (seeded RNG); membership is symmetric.
#   Check:
#     * শ/ষ/স  -- the three sibilants ("shô"), classic spelling-error source.
#     * ন/ণ    -- dontyo nô vs murdhonyo nô.
#     * ই/ঈ    -- hrôssô i vs dirghô i, INDEPENDENT letters. KEEP (author-confirmed).
#     * উ/ঊ    -- hrôssô u vs dirghô u, INDEPENDENT letters. KEEP (author-confirmed).
#     * ি/ী   -- i-kar vs ii-kar, DEPENDENT signs (M0 Q3: most common real
#                 spelling error; e.g. কি vs কী, দিন vs দীন).
#     * ু/ূ   -- u-kar vs uu-kar, DEPENDENT signs (M0 Q3; e.g. তুমি vs তূমি).
#     * র/ড়/ঢ় -- confusable in fast handwriting / phonetically. KEEP (author-confirmed).
#     * জ/য    -- borgiyo jô vs ôntôsthô jô ("jô"/"yô").
#   The KEPT sets are still checked against DATA: METHODOLOGY M2.4 produces a
#   per-set n_eligible / n_applied report over BD-SHS / SentNoB / BanFakeNews,
#   and any set eligible near-zero times is dropped with those numbers cited in
#   Limitations (evidence, not intuition).
HOMOPHONE_SETS: tuple[tuple[str, ...], ...] = (
    ("শ", "ষ", "স"),            # শ ষ স
    ("ন", "ণ"),                      # ন ণ
    ("ই", "ঈ"),                      # ই ঈ   (independent letters; KEEP)
    ("উ", "ঊ"),                      # উ ঊ   (independent letters; KEEP)
    ("ি", "ী"),                      # ি ী   (dependent signs; added M0 Q3)
    ("ু", "ূ"),                      # ু ূ   (dependent signs; added M0 Q3)
    ("র", "ড়", "ঢ়"),            # র ড় ঢ়   (KEEP)
    ("জ", "য"),                      # জ য
)

# Considered and REJECTED as N5 sets, with the reasons to cite in Limitations:
#   * অ/আ, এ/ঐ, ও/ঔ  -- distinct vowels, not a spelling confusion.
#   * ং/ঁ             -- anusvara vs chandrabindu; ঁ is handled by N6 (drop).
#   * ব/ভ  (removed 2026-09-01) -- this is one of TEN aspirated/unaspirated
#     stop pairs (ক/খ, গ/ঘ, চ/ছ, জ/ঝ, ট/ঠ, ড/ঢ, ত/থ, দ/ধ, প/ফ, ব/ভ).
#     Aspiration is phonemic in Bangla, so speakers do not confuse these by
#     ear; including only ব/ভ would be arbitrary. Either all ten or none --
#     we take none.
HOMOPHONE_SETS_REJECTED: tuple[tuple[str, ...], ...] = (
    ("অ", "আ"), ("এ", "ঐ"), ("ও", "ঔ"), ("ং", "ঁ"),
    ("ক", "খ"), ("গ", "ঘ"), ("চ", "ছ"), ("জ", "ঝ"), ("ট", "ঠ"),
    ("ড", "ঢ"), ("ত", "থ"), ("দ", "ধ"), ("প", "ফ"), ("ব", "ভ"),
)


# ---------------------------------------------------------------------------
# 4. N6 matra_perturb -- drop targets and alteration pairs
# ---------------------------------------------------------------------------

CHANDRABINDU = "ঁ"  # U+0981 BENGALI SIGN CANDRABINDU (nasalisation)

# TODO(verify): M0 Q4 + decision 5 -- every occurrence of one of these
#   characters is an eligible unit for N6, and the default action is to DROP it:
#     * all 10 dependent vowel signs, া ৃ ৈ included (া is the most frequent
#       matra in Bangla -- omitting it would cripple N6),
#     * HASANTA ্ (dropping it de-conjuncts: বিদ্যা -> বিদযা),
#     * CHANDRABINDU ঁ (dropping it, চাঁদ -> চাদ / বাঁশ -> বাশ, is one of the
#       most common real Bangla typing omissions and is otherwise uncovered).
#   Confirm this list. N7 also removes hasanta, but at conjunct junctions
#   specifically and with a different mechanism set; N6 removes any hasanta.
N6_DROP_TARGETS: tuple[str, ...] = MATRA_SIGNS + (HASANTA, CHANDRABINDU)

# Back-compat / eligibility alias: the full set of characters N6 may act on.
# Identical to N6_DROP_TARGETS (everything droppable is eligible; a subset is
# also alterable). Kept as a separate name because METHODOLOGY M1.2 refers to
# "N6 targets".
N6_TARGETS: tuple[str, ...] = N6_DROP_TARGETS

# TODO(verify): M0 Q5 (narrowed by decision 3) -- the ONLY matra->matra
#   substitutions N6 may make, instead of dropping. Both are vowel-LENGTH pairs:
#   Bangla speech has no phonemic vowel length, so hrôssô/dirghô i and u are
#   genuinely confused by writers. ে/ৈ and ো/ৌ were considered and REJECTED --
#   they are phonetically distinct and only look similar, so their stroke
#   loss/gain belongs to N1/N2, not here. Each pair is bidirectional.
N6_ALTER_PAIRS: tuple[tuple[str, str], ...] = (
    ("ি", "ী"),  # ি <-> ী   (i-kar / ii-kar)
    ("ু", "ূ"),  # ু <-> ূ   (u-kar / uu-kar)
)


# ---------------------------------------------------------------------------
# 5. N8 elongate -- matra -> full vowel letter
# ---------------------------------------------------------------------------

# TODO(verify): drives N8 submode "insert_vowel": খুব -> খুউব inserts the
#   independent vowel matching a matra (ু -> উ) right after it. Restricted
#   2026-09-01 to the five matras whose stretched form is actually written:
#     ু->উ  ূ->ঊ  ি->ই  ী->ঈ  ো->ও   (খুউব, ভালোওও, দীঈন-style emphasis)
#   DROPPED: ে->এ (সেএ reads unnaturally), ৈ->ঐ, ৃ->ঋ, া->আ -- their stretched
#   spelling is not attested. Those matras still get N8's other two submodes
#   ("repeat_vowel", "repeat_final"). The other two submodes need no map.
MATRA_TO_INDEPENDENT_VOWEL: dict[str, str] = {
    "ু": "উ",  # ু -> উ
    "ূ": "ঊ",  # ূ -> ঊ
    "ি": "ই",  # ি -> ই
    "ী": "ঈ",  # ী -> ঈ
    "ো": "ও",  # ো -> ও
}


# ---------------------------------------------------------------------------
# 6. N7 conjunct_split -- common conjuncts (juktakkhor)
# ---------------------------------------------------------------------------

# TODO(verify): N7 does NOT actually need this list to run -- it finds
#   conjuncts structurally with the regex <consonant><HASANTA><consonant> and
#   either (a) drops the hasanta, splitting into two full consonants
#   (ক্ষ -> কষ), or (b) inserts ZWNJ after the hasanta to break the ligature
#   without changing letters (ক্ষ -> ক্‌ষ). Both, RNG-chosen (M0 Q6).
#   This list is used ONLY as:
#     * test fixtures (tests/test_perturbations.py), and
#     * the validator's "known real conjuncts" reference.
#   Please check these are all genuine, frequently-occurring Bangla conjuncts
#   and add any common ones I have missed. Each string is
#   <consonant><HASANTA><consonant>.
COMMON_CONJUNCTS: tuple[str, ...] = (
    "ক্ষ",          # ক্ষ  (KA + HASANTA + SSA)
    "জ্ঞ",          # জ্ঞ  ("gg")
    "ঞ্চ",          # ঞ্চ
    "ঞ্জ",          # ঞ্জ
    "ণ্ড",          # ণ্ড
    "ত্ত",          # ত্ত
    "ত্র",          # ত্র
    "দ্ধ",          # দ্ধ
    "দ্ব",          # দ্ব
    "ন্ত",          # ন্ত
    "ন্দ",          # ন্দ
    "ন্ধ",          # ন্ধ
    "প্র",          # প্র
    "ব্দ",          # ব্দ
    "স্ত",          # স্ত
    "স্থ",          # স্থ
    "স্প",          # স্প
    "শ্র",          # শ্র
    "ষ্ণ",          # ষ্ণ
    "ঙ্গ",          # ঙ্গ
    "ক্র",          # ক্র
    "ক্ত",          # ক্ত
    "ল্ল",          # ল্ল
    "ম্প",          # ম্প
    "ম্ব",          # ম্ব
)

# TODO(verify): CANDIDATE -- three-consonant conjuncts. M0 Q6: EXCLUDED from N7
#   for now and stated in Limitations. Kept here only so a future version can
#   split exactly one junction per occurrence. Review whether the list is right.
CANDIDATE_CONJUNCTS_TRIPLE: tuple[str, ...] = (
    "স্ত্র",  # স্ত্র
    "ন্ত্র",  # ন্ত্র
    "ক্ষ্ম",  # ক্ষ্ম
)


# ---------------------------------------------------------------------------
# 7. Optional layout-conditioned path (N3b) -- NOT used by the shipped suite
# ---------------------------------------------------------------------------

# M0.1: no Bijoy grid ships (a guessed layout is unverifiable and would
# discredit the suite). N3 is in-class substitution (section 2 above). The
# helper below is retained so that, IF an official Bijoy Bayanno chart is
# obtained later, an ADDITIONAL noise type `N3b char_substitute_bijoy` can be
# built from it without touching N3. Until then nothing calls this.

# Physical key positions on a standard ANSI keyboard, used to compute which
# keys are neighbours. (Layout-independent and reliable.)
QWERTY_PHYSICAL_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (row_id, keys left-to-right).  x-offsets: row1=0.0, row2=0.5, row3=1.5
    ("row1", ("q", "w", "e", "r", "t", "y", "u", "i", "o", "p")),
    ("row2", ("a", "s", "d", "f", "g", "h", "j", "k", "l")),
    ("row3", ("z", "x", "c", "v", "b", "n", "m")),
)
QWERTY_ROW_XOFFSET: dict[str, float] = {"row1": 0.0, "row2": 0.5, "row3": 1.5}


def build_adjacency_from_layout(
    normal_rows: tuple[str, ...],
    shift_rows: tuple[str, ...] | None = None,
    *,
    include_diagonals: bool = True,
) -> dict[str, tuple[str, ...]]:
    """Turn keyboard-layout grids into a symmetric character-adjacency map.

    Unused by the shipped noise suite (see M0.1). Retained for the optional
    layout-conditioned noise type N3b, to be enabled only with a verified Bijoy
    Bayanno chart.

    Two Bangla characters are "adjacent" if the physical keys that produce them
    (on either the normal or shift layer) are neighbours on a standard staggered
    QWERTY keyboard: immediately left/right in the same row, or the one or two
    nearest keys in the row above/below.

    Args:
        normal_rows: one space-separated string per physical row (see
            ``QWERTY_PHYSICAL_ROWS`` for the key count per row). Use ``"-"`` for
            a key that does not emit a single substitutable Bangla character.
        shift_rows: optional shift-layer grid, same shape.
        include_diagonals: if True, count the nearest keys in adjacent rows as
            neighbours (recommended -- real slips are often diagonal).

    Returns:
        Mapping from a single Bangla character to a tuple of distinct
        adjacent characters, sorted for determinism. Only length-1 keys are
        returned.

    Note:
        Pure function, no RNG, no global state.
    """
    def parse(rows: tuple[str, ...]) -> dict[str, tuple[float, float]]:
        """key-string -> (x, y) physical coordinate."""
        pos: dict[str, tuple[float, float]] = {}
        for y, (row_meta, row_str) in enumerate(zip(QWERTY_PHYSICAL_ROWS, rows)):
            row_id, _keys = row_meta
            x0 = QWERTY_ROW_XOFFSET[row_id]
            entries = row_str.split(" ")
            for x, ch in enumerate(entries):
                if ch and ch != "-":
                    pos[ch] = (x0 + x, float(y))
        return pos

    layers = [parse(normal_rows)]
    if shift_rows is not None:
        layers.append(parse(shift_rows))

    adj: dict[str, set[str]] = {}
    for layer in layers:
        items = list(layer.items())
        for ch_a, (xa, ya) in items:
            for ch_b, (xb, yb) in items:
                if ch_a == ch_b:
                    continue
                dx, dy = abs(xa - xb), abs(ya - yb)
                same_row_adjacent = dy == 0 and dx <= 1.01
                diag_adjacent = include_diagonals and dy == 1 and dx <= 1.01
                if same_row_adjacent or diag_adjacent:
                    adj.setdefault(ch_a, set()).add(ch_b)
                    adj.setdefault(ch_b, set()).add(ch_a)

    return {
        k: tuple(sorted(v))
        for k, v in sorted(adj.items())
        if len(k) == 1
    }


# ---------------------------------------------------------------------------
# 8. N10 emoji_transform -- detection ranges (stdlib only)
# ---------------------------------------------------------------------------

# TODO(verify): inclusive code-point ranges treated as "emoji" for detection.
#   Chosen to catch the common pictographic emoji while NOT sweeping up
#   ordinary symbols/punctuation. "replace_with_text" mode uses
#   unicodedata.name(ch) to produce a label like ":rolling_on_the_floor..:".
#   M0 Q9: provisional. `validate.emoji_range_coverage` scores these ranges
#   against the `emoji` package (measurement only -- this module never imports
#   `emoji`) on the Phase 2 corpora; widen them if coverage < ~90%. Known
#   deliberate omissions: plain arrows (U+2190..U+21FF), most of Letterlike
#   Symbols, CJK symbols.
EMOJI_CODEPOINT_RANGES: tuple[tuple[int, int], ...] = (
    (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
    (0x1F600, 0x1F64F),  # Emoticons
    (0x1F680, 0x1F6FF),  # Transport and Map Symbols
    (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
    (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
    (0x2600, 0x26FF),    # Misc Symbols (sun, umbrella, ball ...)
    (0x2700, 0x27BF),    # Dingbats (scissors, check mark, sparkles ...)
    (0x1F1E6, 0x1F1FF),  # Regional Indicator Symbols (flag halves)
    (0x1F000, 0x1F02F),  # Mahjong / dominoes / playing cards start
)

# TODO(verify): "glue" code points that are part of an emoji sequence and must
#   be stripped along with the emoji in "remove" mode (otherwise a lone ZWJ or
#   variation selector is left behind, which the validator would flag).
EMOJI_SEQUENCE_GLUE: tuple[str, ...] = (
    "‍",  # ZWJ (family / profession sequences)
    "️",  # VARIATION SELECTOR-16 (emoji presentation)
    "︎",  # VARIATION SELECTOR-15 (text presentation)
    "⃣",  # COMBINING ENCLOSING KEYCAP (0..9,#,* -> keycap)
)

# TODO(verify): fallback text when unicodedata has no name for an emoji
#   (unassigned in this Python's Unicode DB). Deterministic sentinel.
EMOJI_REPLACEMENT_FALLBACK = ":emoji:"


# ---------------------------------------------------------------------------
# 9. NFC normalization of all string constants (M0 Q10)
# ---------------------------------------------------------------------------

# The whole pipeline operates on NFC-normalized text. Three consonants --
# ড় (U+09DC), ঢ় (U+09DD), য় (U+09DF) -- are Unicode *composition exclusions*:
# NFC DECOMPOSES them to <base + U+09BC nukta> and does not recombine. The
# literals above are written precomposed for readability, so we normalize every
# constant here to guarantee it matches NFC-normalized input. (ো/ৌ are NOT
# exclusions; NFC keeps them composed. The BanglaBERT normalizer may still
# decompose them -- that is what validate.py's round-trip test checks.)

import unicodedata as _ud


def _nfc(s: str) -> str:
    return _ud.normalize("NFC", s)


def _nfc_seq(seq: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_nfc(x) for x in seq)


HASANTA = _nfc(HASANTA)
ZWNJ = _nfc(ZWNJ)
ZWJ = _nfc(ZWJ)
CHANDRABINDU = _nfc(CHANDRABINDU)
BANGLA_CONSONANTS = _nfc_seq(BANGLA_CONSONANTS)
BANGLA_INDEPENDENT_VOWELS = _nfc_seq(BANGLA_INDEPENDENT_VOWELS)
MATRA_SIGNS = _nfc_seq(MATRA_SIGNS)
BANGLA_SIGNS = _nfc_seq(BANGLA_SIGNS)
BANGLA_DIGITS = _nfc_seq(BANGLA_DIGITS)
INSERTABLE_CHARS = _nfc_seq(INSERTABLE_CHARS)
IN_CLASS_POOLS = {k: _nfc_seq(v) for k, v in IN_CLASS_POOLS.items()}
HOMOPHONE_SETS = tuple(_nfc_seq(s) for s in HOMOPHONE_SETS)
HOMOPHONE_SETS_REJECTED = tuple(_nfc_seq(s) for s in HOMOPHONE_SETS_REJECTED)
N6_DROP_TARGETS = _nfc_seq(N6_DROP_TARGETS)
N6_TARGETS = _nfc_seq(N6_TARGETS)
N6_ALTER_PAIRS = tuple((_nfc(a), _nfc(b)) for a, b in N6_ALTER_PAIRS)
MATRA_TO_INDEPENDENT_VOWEL = {
    _nfc(k): _nfc(v) for k, v in MATRA_TO_INDEPENDENT_VOWEL.items()
}
COMMON_CONJUNCTS = _nfc_seq(COMMON_CONJUNCTS)
CANDIDATE_CONJUNCTS_TRIPLE = _nfc_seq(CANDIDATE_CONJUNCTS_TRIPLE)


# ---------------------------------------------------------------------------
# 10. Convenience lookups (derived; do not edit)
# ---------------------------------------------------------------------------

CONSONANT_SET = frozenset(BANGLA_CONSONANTS)
INDEPENDENT_VOWEL_SET = frozenset(BANGLA_INDEPENDENT_VOWELS)
MATRA_SET = frozenset(MATRA_SIGNS)
SIGN_SET = frozenset(BANGLA_SIGNS)
DIGIT_SET = frozenset(BANGLA_DIGITS)

# Reverse index: base character -> its IN_CLASS_POOLS class name (or None).
CHAR_TO_CLASS: dict[str, str] = {
    ch: cls for cls, chars in IN_CLASS_POOLS.items() for ch in chars
}

# All combining marks that must never appear at the start of a cluster / after
# whitespace. Used by validate.py.  ় = U+09BC nukta, ৗ = U+09D7 au length mark.
COMBINING_MARKS = frozenset(MATRA_SIGNS) | {HASANTA, "়", "ৗ", "ঁ", "ং", "ঃ"}
