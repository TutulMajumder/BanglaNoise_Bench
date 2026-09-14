"""Phase 2, pipeline stage 4: the METHODOLOGY M2.4 measurements on the real
corpora, plus an N10 report (not a decision).

    (1) 9x5 CER table on real corpus samples, side by side with the synthetic
        development table; ratio structure vs the 2.00 / 2.00 / 1.75 / 1.43
        target -- evaluated from s3/s2 upward (see the conjunct_split note).
    (2) N5 per-set firing counts: n_eligible (grouping pass) and n_applied
        (summed off PerturbStats.submodes) per HOMOPHONE_SET per dataset.
        ই/ঈ and উ/ঊ are explicitly under review.
    (3) Emoji-range coverage per dataset (validate.emoji_range_coverage) --
        an UPPER BOUND against the emoji 1.4.2 / Unicode 13 reference.
    (4) Normalizer touch_rate per dataset -- the R6 go/no-go (M5.4), decided
        PER DATASET against the 5% threshold, not on an arbitrarily pooled
        "ALL". See the R6 note below.
    N10: emoji-bearing text counts per dataset, per TEST split (full split,
        not the 300-sample) -- report only, no drop/keep decision made here.

Reads ``data/final/<dataset>/{train,val,test}.csv`` from
``pipeline/s2_prepare.py``. Writes TWO copies:

    results/m2_4.{md,html,json}         full detail, WITH verbatim corpus
                                         text in normalizer examples --
                                         results/ is git-ignored.
    outputs/s2_preparation/tables/*.csv tracked, paper-facing (M2.4 is part
                                         of METHODOLOGY's M2 umbrella, same
                                         output folder as s2_prepare.py).
    outputs/s2_preparation/m2_4_summary.md  tracked; normalizer examples are
                                         STRUCTURAL DESCRIPTIONS (edit kind,
                                         character classes changed, offset,
                                         length delta) -- no verbatim text,
                                         since SentNoB is CC-BY-ND-4.0.

    python pipeline/s4_measure.py --limit 20              # smoke run
    python pipeline/s4_measure.py                         # real run
    python pipeline/s4_measure.py --measurements 4        # just R6

CLAUDE.md invariant 9: resumable (skips if the tracked CSVs already exist,
unless --force), prints progress to stderr, and --limit caps the sample size.

Nothing here modifies ``noisebench/``. It imports two private helpers
(``_homophone_occurrences``, ``_HOMOPHONE_MEMBER_TO_SET``) that M2.4(2)
explicitly needs to read.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import sys
import time
import unicodedata
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rapidfuzz.distance import Levenshtein as _rf_levenshtein  # noqa: E402

from noisebench import NOISE_TYPES, perturb, perturb_with_stats  # noqa: E402
from noisebench.bangla_maps import HOMOPHONE_SETS  # noqa: E402
from noisebench.corpus_stats import CER_TRUNCATE_CLUSTERS, classify_char, emoji_bearing_count  # noqa: E402
from noisebench.perturbations import (  # noqa: E402,PLC2701
    _HOMOPHONE_MEMBER_TO_SET,
    _homophone_occurrences,
    clusters,
)
from noisebench.severity import MAX_SEVERITY, MIN_SEVERITY  # noqa: E402
from noisebench.validate import (  # noqa: E402
    emoji_range_coverage,
    normalize_banglabert,
    normalizer_available,
)
from pipeline._common import (  # noqa: E402
    Progress,
    add_common_args,
    print_smoke_banner,
    should_skip,
    smoke_or_prod,
    write_run_meta,
)
from pipeline.s2_prepare import DATASETS, dataset_final_dir  # noqa: E402

_DEFAULT_DATA_FINAL = os.path.join(_ROOT, "data", "final")
_RESULTS_DIR = os.path.join(_ROOT, "results")
_OUT_DIR = os.path.join(_ROOT, "outputs", "s2_preparation")
_TARGET_RATIOS = (2.00, 2.00, 1.75, 1.43)
SAMPLE_SEED = 20260902
NORMALIZER_R6_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------


def _reservoir_sample_rows(paths: list[str], column: str, k: int, seed_key: str) -> list[str]:
    """Reservoir-sample (Algorithm R) `k` values from `column`, streaming
    across one or more CSVs (a dataset's train+val+test.csv) in one pass,
    never holding more than `k` rows in memory. `banfakenews/full/*.csv`
    alone is ~250MB of article text; loading it whole (the pre-restructure
    approach, via pandas) to keep 300 rows was the real memory cost, not the
    sampling."""
    rng = random.Random(seed_key)
    reservoir: list[str] = []
    n_seen = 0
    for path in paths:
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                n_seen += 1
                val = row[column]
                if len(reservoir) < k:
                    reservoir.append(val)
                else:
                    j = rng.randrange(n_seen)
                    if j < k:
                        reservoir[j] = val
    return reservoir


def load_samples(data_final_root: str, k: int) -> dict[str, list[str]]:
    """Sample `k` texts per dataset from ALL of its train+val+test.csv
    (pooled) -- M2.4's real-corpus checks are about the prepared corpus as a
    whole, not any one split."""
    out: dict[str, list[str]] = {}
    for name, spec in DATASETS.items():
        d = dataset_final_dir(spec, data_final_root)
        paths = [os.path.join(d, f"{s}.csv") for s in ("train", "val", "test")
                if os.path.exists(os.path.join(d, f"{s}.csv"))]
        if not paths:
            continue
        out[name] = _reservoir_sample_rows(paths, "text", k, f"{SAMPLE_SEED}\x00{name}")
    if len(out) > 1:
        # Built from the already-sampled (<=k each) per-dataset lists above,
        # never from a fresh concatenation of the full corpora.
        rng = random.Random(f"{SAMPLE_SEED}\x00combined")
        pool = [t for texts in out.values() for t in texts]
        out["ALL"] = pool if len(pool) <= k else rng.sample(pool, k)
    return out


def _synthetic_corpus() -> list[str]:
    from scripts.phase1_report import _make_corpus

    return _make_corpus()


# ---------------------------------------------------------------------------
# (1) CER table
# ---------------------------------------------------------------------------


def _ratios(row: list[float]) -> list[float]:
    return [row[i + 1] / row[i] if row[i] else float("nan") for i in range(len(row) - 1)]


# `levenshtein` (noisebench/validate.py) is pure-Python O(n^2). BanFakeNews
# articles average ~1,100 grapheme clusters (p95 far higher) vs ~60 for the
# Phase 1 fixture -- at 9 noise types x 5 severities x 3 seeds x 300 texts
# per dataset group, that is unrunnable (a real run hung for 17h). M2.4(1)
# exists to check the severity-ratio structure (2.00/2.00/1.75/1.43) survives
# on real text, which is not a document-level CER claim: CER is normalized by
# reference length and the perturbation count rule (M1.3) is per-eligible-unit,
# so capping the input length does not bias the ratios. Applied ONLY here --
# N5/emoji/normalizer measurements below see full-length text. CAP is defined
# once in noisebench/corpus_stats.py (shared with s3_describe.py's length-ECDF
# figure). See METHODOLOGY M2.4 and PHASE2_STATUS.md for the recorded cap.


def _truncate_for_cer(text: str, cap: int = CER_TRUNCATE_CLUSTERS) -> str:
    cl = clusters(text)
    return "".join(cl[:cap]) if len(cl) > cap else text


def _cer_fast(reference: str, hypothesis: str) -> float:
    """Same definition as noisebench.validate.cer (cluster-level Levenshtein
    normalized by reference length), but backed by rapidfuzz's C++
    implementation instead of the pure-Python one. Verified to agree EXACTLY
    (0 mismatches) with noisebench.validate.cer over the full Phase 1 9x5 CER
    table (all noise types, severities, seeds) before use -- see
    PHASE2_STATUS.md. pipeline/-only: noisebench/ is untouched and still
    stdlib+regex only (CLAUDE.md invariant)."""
    ref_clusters = clusters(reference)
    hyp_clusters = clusters(hypothesis)
    return _rf_levenshtein.distance(ref_clusters, hyp_clusters) / max(1, len(ref_clusters))


def _corpus_cer_fast(pairs: list[tuple[str, str]]) -> float:
    if not pairs:
        return 0.0
    return sum(_cer_fast(ref, hyp) for ref, hyp in pairs) / len(pairs)


def _build_cer_table_fast(
    texts: list[str],
    seeds: tuple[int, ...],
    noise_types: tuple[str, ...] = NOISE_TYPES,
) -> dict[str, list[float]]:
    """Same contract as noisebench.validate.build_cer_table (mean corpus CER
    per noise type per severity), rapidfuzz-backed. See _cer_fast."""
    table: dict[str, list[float]] = {}
    severities = list(range(MIN_SEVERITY, MAX_SEVERITY + 1))
    for noise_type in noise_types:
        row: list[float] = []
        for severity in severities:
            pairs = [
                (text, perturb(text, noise_type, severity, seed))
                for text in texts
                for seed in seeds
            ]
            row.append(_corpus_cer_fast(pairs))
        table[noise_type] = row
    return table


def measure_cer(samples: dict[str, list[str]], seeds: tuple[int, ...]) -> dict:
    synth = _build_cer_table_fast([_truncate_for_cer(t) for t in _synthetic_corpus()], seeds=seeds)
    real = {
        ds: _build_cer_table_fast([_truncate_for_cer(t) for t in texts], seeds=seeds)
        for ds, texts in samples.items()
    }
    return {
        "synthetic": synth,
        "real": real,
        "seeds": list(seeds),
        "truncate_clusters": CER_TRUNCATE_CLUSTERS,
    }


# ---------------------------------------------------------------------------
# (2) N5 per-set firing
# ---------------------------------------------------------------------------


def _set_key(homoset: tuple[str, ...]) -> str:
    return "/".join(homoset)


def measure_n5(samples: dict[str, list[str]], seeds: tuple[int, ...],
               severity: int = 3) -> dict:
    keys = [_set_key(s) for s in HOMOPHONE_SETS]
    result: dict[str, dict] = {}
    for ds, texts in samples.items():
        if ds == "ALL":
            continue
        elig = {k: 0 for k in keys}
        applied = {k: 0 for k in keys}
        n_texts_with = {k: 0 for k in keys}
        for t in texts:
            seen = set()
            for _ci, _off, member in _homophone_occurrences(clusters(t)):
                k = _set_key(_HOMOPHONE_MEMBER_TO_SET[member])
                elig[k] += 1
                seen.add(k)
            for k in seen:
                n_texts_with[k] += 1
            for seed in seeds:
                _out, st = perturb_with_stats(t, "homophone_confuse", severity, seed)
                for k, v in st.submodes.items():
                    applied[k] = applied.get(k, 0) + v
        result[ds] = {
            "n_texts": len(texts),
            "severity_for_applied": severity,
            "seeds": list(seeds),
            "per_set": {
                k: {
                    "n_eligible": elig[k],
                    "n_applied": applied.get(k, 0),
                    "texts_with_ge1": n_texts_with[k],
                    "eligible_per_1k_texts": 1000.0 * elig[k] / max(1, len(texts)),
                }
                for k in keys
            },
        }
    # under-review flags, aggregated across datasets
    review = {}
    for target in ("ই/ঈ", "উ/ঊ"):
        tot_elig = sum(result[ds]["per_set"].get(target, {}).get("n_eligible", 0)
                       for ds in result)
        tot_texts = sum(result[ds]["n_texts"] for ds in result)
        per_1k = 1000.0 * tot_elig / max(1, tot_texts)
        review[target] = {
            "total_eligible": tot_elig,
            "eligible_per_1k_texts": per_1k,
            "recommend_drop": per_1k < 1.0,  # < ~1 occurrence per 1000 texts
        }
    return {"datasets": result, "under_review": review, "set_order": keys}


# ---------------------------------------------------------------------------
# (3) emoji coverage
# ---------------------------------------------------------------------------


def measure_emoji(samples: dict[str, list[str]]) -> dict:
    out: dict[str, dict | None] = {}
    for ds, texts in samples.items():
        out[ds] = emoji_range_coverage(texts)
    return {
        "reference": "emoji 1.4.2 (~Unicode 13 / 2021) -- coverage is an UPPER "
                     "BOUND against this era, not current Unicode",
        "per_dataset": out,
    }


# ---------------------------------------------------------------------------
# N10 -- report, don't decide: emoji-bearing counts on the FULL test split
# ---------------------------------------------------------------------------


def measure_n10_test_split(data_final_root: str, limit: int | None = None) -> dict:
    """Emoji-bearing text counts per dataset, over the FULL test.csv (not
    the 300-sample) -- the 300-sample cannot tell you whether a test split
    the size of BanFakeNews's (1,698) or SentNoB's (1,569) has enough
    emoji-bearing texts to support a 5-point severity curve. Report only; no
    threshold/decision is applied here -- that call is not this script's."""
    out: dict[str, dict] = {}
    for name, spec in DATASETS.items():
        path = os.path.join(dataset_final_dir(spec, data_final_root), "test.csv")
        if not os.path.exists(path):
            continue
        test_texts: list[str] = []
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                test_texts.append(row["text"])
                if limit is not None and len(test_texts) >= limit:
                    break
        n_emoji = emoji_bearing_count(test_texts)
        out[name] = {
            "n_test_texts": len(test_texts),
            "n_emoji_bearing": n_emoji,
            "pct_emoji_bearing": (100.0 * n_emoji / len(test_texts)) if test_texts else 0.0,
        }
    return out


# ---------------------------------------------------------------------------
# (4) normalizer touch_rate  -- R6 go/no-go, decided PER DATASET
# ---------------------------------------------------------------------------


def _diff_char_classes(before: str, after: str) -> dict[str, int]:
    """Character classes whose count differs between `before` and `after`,
    with the net count of characters removed+added in that class.

    This is a real diff, not a co-occurrence flag: the previous classifier's
    "has-digit" tag fired whenever ANY digit appeared ANYWHERE in `before`
    (e.g. a date in an unrelated sentence of a long article), regardless of
    whether normalization touched a digit at all -- on article-length text
    that fires on nearly everything and says nothing about what changed.
    """
    cb, ca = Counter(before), Counter(after)
    changed = (cb - ca) + (ca - cb)  # chars whose count differs, either direction
    classes: dict[str, int] = {}
    for ch, n in changed.items():
        cls = classify_char(ch)
        classes[cls] = classes.get(cls, 0) + n
    return classes


def _r6_verdict_label(recommend_drop: bool | None) -> str:
    if recommend_drop is None:
        return "N/A (pooled, not a decision unit)"
    return "DROP (<5%)" if recommend_drop else "KEEP (≥5%)"


def _classify_touch(before: str, after: str) -> str:
    """The actual character classes normalize_banglabert changed, plus
    length-delta and NFKC-equivalence diagnostics. Replaces the old
    co-occurrence tags ("has-digit", "url-ish") which described the whole
    input text, not the edit."""
    if before == after:
        return "unchanged"
    classes = _diff_char_classes(before, after)
    tags = sorted(classes, key=lambda c: -classes[c])
    if len(after) != len(before):
        tags.append(f"len {len(before)}->{len(after)}")
    if unicodedata.normalize("NFKC", before) == after:
        tags.append("==NFKC")
    return ", ".join(tags) or "other"


def _structural_description(before: str, after: str, kind: str) -> dict:
    """No verbatim corpus text -- describes the edit instead (METHODOLOGY:
    SentNoB is CC-BY-ND-4.0 and cannot be redistributed; the others are kept
    out on principle). Used for the tracked outputs/ copy only; results/
    (git-ignored) keeps the verbatim before/after for local inspection."""
    offset = 0
    for a, b in zip(before, after):
        if a != b:
            break
        offset += 1
    return {
        "kind": kind,
        "offset_first_diff": offset,
        "length_before": len(before),
        "length_delta": len(after) - len(before),
    }


def measure_normalizer(samples: dict[str, list[str]], max_examples: int = 12) -> dict:
    if not normalizer_available():
        return {"available": False}
    per: dict[str, dict] = {}
    for ds, texts in samples.items():
        touched = 0
        examples: list[dict] = []
        cats: dict[str, int] = {}
        for t in texts:
            norm = normalize_banglabert(t)
            if norm is not None and norm != t:
                touched += 1
                cat = _classify_touch(t, norm)
                cats[cat] = cats.get(cat, 0) + 1
                if len(examples) < max_examples:
                    examples.append({"before": t[:160], "after": norm[:160], "kind": cat})
        touch_rate = touched / len(texts) if texts else 0.0
        per[ds] = {
            "n_texts": len(texts),
            "touched": touched,
            "touch_rate": touch_rate,
            "kinds": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
            "examples": examples,
            # "ALL" is the pooled sample across every dataset -- exactly the
            # arbitrary-composition pool this per-dataset rule replaces, so it
            # gets no R6 verdict of its own (touch_rate is still reported).
            "recommend_drop_r6": (touch_rate < NORMALIZER_R6_THRESHOLD) if ds != "ALL" else None,
        }
    overall_texts = sum(v["n_texts"] for k, v in per.items() if k != "ALL")
    overall_touched = sum(v["touched"] for k, v in per.items() if k != "ALL")
    overall_rate = overall_touched / overall_texts if overall_texts else 0.0
    return {
        "available": True,
        "per_dataset": per,
        # Kept for reference only -- NOT the R6 decision rule. M5.4's 5%
        # threshold is applied PER DATASET (per["<name>"]["recommend_drop_r6"]),
        # not to this pool, whose dataset composition (2 BanFakeNews variants,
        # 1 SentNoB, 1 bd_shs) is an artefact of the registry, not a
        # meaningful population to average touch_rate over.
        "overall_touch_rate_DO_NOT_USE_FOR_R6": overall_rate,
        "threshold": NORMALIZER_R6_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# rendering (full detail, WITH verbatim text -- results/, git-ignored)
# ---------------------------------------------------------------------------


#: conjunct_split (and any noise type) can show a noisy s2/s1 ratio on
#: short-text corpora at severity 1 -- few eligible junctions means the
#: floor+Bernoulli count rule (M1.3) is dominated by the Bernoulli remainder.
#: The ratio-structure claim is evaluated from s3/s2 upward; s2/s1 is reported
#: but not part of the pass/fail rule. See s3_describe.py's eligible-units-
#: per-noise-type figure for the distribution that explains this (not a
#: guess: it's the same n_eligible this rule is about).
RATIO_RULE_NOTE = (
    "Ratio-structure rule: s3/s2, s4/s3, s5/s4 must each be within 10% of "
    "their targets (2.00 / 1.75 / 1.43). s2/s1 is reported but excluded from "
    "the pass/fail rule, because low eligible-unit counts at severity 1 make "
    "it noisy on short-text corpora -- see s3_describe.py's eligible-units-per-"
    "noise-type figure for the eligible-count distribution behind this."
)


def _cer_block_md(cer: dict) -> list[str]:
    L = [f"## (1) 9×5 CER — real vs synthetic  (seeds {cer['seeds']})", ""]
    L.append(f"Target severity ratios s2/s1, s3/s2, s4/s3, s5/s4 = "
             f"**{' / '.join(f'{x:.2f}' for x in _TARGET_RATIOS)}**.")
    L.append("")
    L.append(RATIO_RULE_NOTE)
    L.append("")
    L.append(f"**Each text capped to its first {cer['truncate_clusters']} grapheme "
             f"clusters before perturbing** (real corpora only average far more "
             f"than the synthetic dev corpus; `levenshtein` is O(n²) and this "
             f"measurement checks the ratio structure, not document-level CER — "
             f"see METHODOLOGY M2.4). CER is reference-length-normalized and the "
             f"M1.3 count rule is per-eligible-unit, so this does not bias the ratios.")
    L.append("")
    tables = [("synthetic (dev corpus)", cer["synthetic"])]
    tables += [(f"real: {ds}", tbl) for ds, tbl in cer["real"].items()]
    for title, tbl in tables:
        L.append(f"### {title}")
        L.append("")
        L.append("| noise_type | s1 | s2 | s3 | s4 | s5 | s2/s1 | s3/s2 | s4/s3 | s5/s4 | rule |")
        L.append("|---|--:|--:|--:|--:|--:|:--:|:--:|:--:|:--:|:--:|")
        for nt in NOISE_TYPES:
            row = tbl[nt]
            rr = _ratios(row)
            scored = rr[1:]  # exclude s2/s1 from the rule, per RATIO_RULE_NOTE
            scored_targets = _TARGET_RATIOS[1:]
            off = any(
                r == r and abs(r - t) > 0.10 * t
                for r, t in zip(scored, scored_targets)
            )
            rule = "holds (s3+)" if not off else "⚠ off (s3+)"
            L.append(
                f"| {nt} | " + " | ".join(f"{v:.4f}" for v in row) + " | "
                + " | ".join(f"{r:.2f}" for r in rr)
                + f" | {rule} |"
            )
        L.append("")
    return L


def _n5_block_md(n5: dict) -> list[str]:
    L = ["## (2) N5 homophone-set firing counts", ""]
    L.append("`n_eligible` = member-occurrence count (grouping pass over "
             "`_homophone_occurrences`, severity-independent). `n_applied` = "
             "sum of `PerturbStats.submodes` at severity 3 over the listed seeds. "
             "`per 1k` = eligible occurrences per 1000 sampled texts.")
    L.append("")
    for ds, d in n5["datasets"].items():
        L.append(f"### {ds}  (n={d['n_texts']} texts, applied@s{d['severity_for_applied']})")
        L.append("")
        L.append("| homophone set | n_eligible | per 1k texts | texts ≥1 | n_applied |")
        L.append("|---|--:|--:|--:|--:|")
        for k in n5["set_order"]:
            ps = d["per_set"][k]
            L.append(f"| {k} | {ps['n_eligible']} | {ps['eligible_per_1k_texts']:.1f} "
                     f"| {ps['texts_with_ge1']} | {ps['n_applied']} |")
        L.append("")
    L.append("### Sets under review (M2.4)")
    L.append("")
    L.append("| set | total eligible | per 1k texts | recommendation |")
    L.append("|---|--:|--:|---|")
    for k, v in n5["under_review"].items():
        rec = "**DROP — evidence below threshold**" if v["recommend_drop"] else "keep"
        L.append(f"| {k} | {v['total_eligible']} | {v['eligible_per_1k_texts']:.2f} | {rec} |")
    L.append("")
    L.append("Threshold: < 1 eligible occurrence per 1000 texts ⇒ recommend removal "
             "with this evidence, per METHODOLOGY M2.4(2). Not removed automatically.")
    L.append("")
    return L


def _emoji_block_md(em: dict) -> list[str]:
    L = ["## (3) Emoji-range coverage", "", em["reference"], ""]
    L.append("| dataset | emoji lib | occurrences | coverage (any) | coverage (full) | over-match cps |")
    L.append("|---|---|--:|--:|--:|--:|")
    for ds, c in em["per_dataset"].items():
        if c is None:
            L.append(f"| {ds} | _not installed_ | — | — | — | — |")
            continue
        L.append(f"| {ds} | {c['emoji_lib_version']} | {c['n_occurrences']} | "
                 f"{c['coverage_any']:.1%} | {c['coverage_full']:.1%} | "
                 f"{c['n_false_positive_codepoints']} |")
    L.append("")
    worst = [
        (ds, c) for ds, c in em["per_dataset"].items()
        if c and c["n_occurrences"] >= 20 and c["coverage_any"] < 0.90
    ]
    if worst:
        L.append("**Coverage < ~90% on ≥20 occurrences — proposed (NOT applied) range widening:**")
        L.append("")
        for ds, c in worst:
            ucp = sorted({ord(ch) for e in c["uncaught"] for ch in e})
            blocks = sorted({f"U+{cp:04X}" for cp in ucp})
            L.append(f"- `{ds}`: {c['coverage_any']:.1%} any-coverage. Uncaught code "
                     f"points: {', '.join(blocks[:20])}"
                     + (" …" if len(blocks) > 20 else ""))
        L.append("")
        L.append("Add the enclosing Unicode blocks to `EMOJI_CODEPOINT_RANGES` in a "
                 "later, reviewed change — do not touch `noisebench/` in Phase 2.")
    else:
        L.append("All datasets with ≥20 emoji occurrences are at ≥90% any-coverage; "
                 "no range change proposed.")
    L.append("")
    return L


def _n10_block_md(n10: dict) -> list[str]:
    L = ["## N10 — emoji-bearing text counts, full test split (report only)", ""]
    L.append("Not a decision. A 5-point severity curve needs enough "
             "emoji-bearing texts in the TEST split specifically -- the "
             "300-sample used above cannot answer this for splits larger "
             "than 300.")
    L.append("")
    L.append("| dataset | n test texts | emoji-bearing | % |")
    L.append("|---|--:|--:|--:|")
    for ds, d in n10.items():
        L.append(f"| {ds} | {d['n_test_texts']} | {d['n_emoji_bearing']} | "
                 f"{d['pct_emoji_bearing']:.2f}% |")
    L.append("")
    return L


def _norm_block_md(nm: dict) -> list[str]:
    L = ["## (4) Normalizer touch_rate — R6 go/no-go (M5.4, PER DATASET)", ""]
    if not nm.get("available"):
        L.append("_`csebuetnlp/normalizer` not installed — measurement skipped. "
                 "Install it (`requirements.txt`) and re-run before deciding R6._")
        return L + [""]
    L.append("M5.4's 5% threshold is applied **per dataset**, not to a pool "
             "whose composition (2 BanFakeNews variants + 1 SentNoB + 1 "
             "bd_shs) is a registry artefact, not a population.")
    L.append("")
    L.append("| dataset | n texts | touched | touch_rate | R6 (this dataset) | kinds |")
    L.append("|---|--:|--:|--:|---|---|")
    for ds, d in nm["per_dataset"].items():
        kinds = ", ".join(f"{k} ({v})" for k, v in list(d["kinds"].items())[:4]) or "—"
        verdict = _r6_verdict_label(d["recommend_drop_r6"])
        L.append(f"| {ds} | {d['n_texts']} | {d['touched']} | {d['touch_rate']:.2%} "
                 f"| {verdict} | {kinds} |")
    L.append("")
    L.append("### What it touched — examples")
    L.append("")
    for ds, d in nm["per_dataset"].items():
        if not d["examples"]:
            continue
        L.append(f"**{ds}:**")
        L.append("")
        for ex in d["examples"][:8]:
            L.append(f"- _{ex['kind']}_")
            L.append(f"  - before: `{ex['before']}`")
            L.append(f"  - after:  `{ex['after']}`")
        L.append("")
    return L


def render_markdown(payload: dict) -> str:
    L = ["# METHODOLOGY M2.4 — real-corpus measurements (full detail, verbatim text)", ""]
    L.append(f"Sample: {payload['sample_note']}.")
    L.append("")
    if "timing" in payload:
        L.append("Wall time: " + ", ".join(
            f"({n}) {t:.1f}s" for n, t in payload["timing"].items()
        ))
        L.append("")
    if "cer" in payload:
        L += _cer_block_md(payload["cer"])
    else:
        L += ["## (1) 9×5 CER — real vs synthetic", "", "_skipped this run (`--measurements`)._", ""]
    if "n5" in payload:
        L += _n5_block_md(payload["n5"])
    else:
        L += ["## (2) N5 homophone-set firing counts", "", "_skipped this run (`--measurements`)._", ""]
    if "emoji" in payload:
        L += _emoji_block_md(payload["emoji"])
    else:
        L += ["## (3) Emoji-range coverage", "", "_skipped this run (`--measurements`)._", ""]
    if "normalizer" in payload:
        L += _norm_block_md(payload["normalizer"])
    else:
        L += ["## (4) Normalizer touch_rate — R6 go/no-go (M5.4)", "", "_skipped this run (`--measurements`)._", ""]
    if "n10" in payload:
        L += _n10_block_md(payload["n10"])
    return "\n".join(L) + "\n"


_HTML_HEAD = """<!doctype html>
<html lang="bn">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>M2.4 — real-corpus measurements</title>
<style>
  body { font-family: "Noto Sans Bengali", "Nirmala UI", system-ui, sans-serif;
         line-height: 1.5; margin: 2rem auto; max-width: 1050px; padding: 0 1rem; }
  h1 { font-size: 1.4rem; } h2 { font-size: 1.15rem; margin-top: 2rem;
       border-bottom: 1px solid #ddd; } h3 { font-size: 1rem; }
  table { border-collapse: collapse; width: 100%; margin: .6rem 0; font-size: .9rem; }
  th, td { border: 1px solid #ccc; padding: .3rem .5rem; text-align: left; }
  th { background: #f2f2f2; }
  code { background: #f4f4f4; padding: .05rem .3rem; border-radius: 3px;
         overflow-wrap: anywhere; }
</style>
</head>
<body>
"""


def render_html(payload: dict) -> str:
    # Markdown is authoritative; the HTML mirror keeps a charset + Bangla font
    # stack per invariant 8. Convert the markdown minimally.
    md = render_markdown(payload)
    body: list[str] = [_HTML_HEAD]
    in_table = False
    for line in md.splitlines():
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                tag = "th"
            else:
                tag = "td"
            body.append("<tr>" + "".join(f"<{tag}>{html.escape(c)}</{tag}>" for c in cells) + "</tr>")
        else:
            if in_table:
                body.append("</table>")
                in_table = False
            if line.strip():
                body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    body.append("</body></html>\n")
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Tracked, text-stripped outputs (outputs/s2_preparation/)
# ---------------------------------------------------------------------------


def write_tracked_csv_tables(payload: dict, tables_dir: str) -> None:
    """CSV tables with NO verbatim corpus text -- safe to commit (SentNoB is
    CC-BY-ND-4.0; the others are kept out on principle too)."""
    os.makedirs(tables_dir, exist_ok=True)

    if "cer" in payload:
        with open(os.path.join(tables_dir, "m2_4_cer.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["group", "noise_type", "s1", "s2", "s3", "s4", "s5",
                        "s2_s1", "s3_s2", "s4_s3", "s5_s4", "rule_holds_s3plus"])
            groups = [("synthetic", payload["cer"]["synthetic"])]
            groups += [(f"real_{ds}", tbl) for ds, tbl in payload["cer"]["real"].items()]
            for group, tbl in groups:
                for nt in NOISE_TYPES:
                    row = tbl[nt]
                    rr = _ratios(row)
                    scored = rr[1:]
                    off = any(r == r and abs(r - t) > 0.10 * t
                             for r, t in zip(scored, _TARGET_RATIOS[1:]))
                    w.writerow([group, nt, *[f"{v:.5f}" for v in row],
                               *[f"{r:.3f}" for r in rr], not off])

    if "n5" in payload:
        with open(os.path.join(tables_dir, "m2_4_n5.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "homophone_set", "n_eligible", "eligible_per_1k_texts",
                        "texts_with_ge1", "n_applied"])
            for ds, d in payload["n5"]["datasets"].items():
                for k in payload["n5"]["set_order"]:
                    ps = d["per_set"][k]
                    w.writerow([ds, k, ps["n_eligible"], f"{ps['eligible_per_1k_texts']:.3f}",
                               ps["texts_with_ge1"], ps["n_applied"]])

    if "emoji" in payload:
        with open(os.path.join(tables_dir, "m2_4_emoji.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "emoji_lib_version", "n_occurrences", "coverage_any",
                        "coverage_full", "n_false_positive_codepoints"])
            for ds, c in payload["emoji"]["per_dataset"].items():
                if c is None:
                    w.writerow([ds, "not_installed", "", "", "", ""])
                else:
                    w.writerow([ds, c["emoji_lib_version"], c["n_occurrences"],
                               f"{c['coverage_any']:.4f}", f"{c['coverage_full']:.4f}",
                               c["n_false_positive_codepoints"]])

    if "n10" in payload:
        with open(os.path.join(tables_dir, "m2_4_n10_test_split.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "n_test_texts", "n_emoji_bearing", "pct_emoji_bearing"])
            for ds, d in payload["n10"].items():
                w.writerow([ds, d["n_test_texts"], d["n_emoji_bearing"], f"{d['pct_emoji_bearing']:.3f}"])

    if "normalizer" in payload and payload["normalizer"].get("available"):
        with open(os.path.join(tables_dir, "m2_4_normalizer.csv"), "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "n_texts", "touched", "touch_rate", "recommend_drop_r6", "top_kinds"])
            for ds, d in payload["normalizer"]["per_dataset"].items():
                kinds = "; ".join(f"{k} ({v})" for k, v in list(d["kinds"].items())[:6])
                w.writerow([ds, d["n_texts"], d["touched"], f"{d['touch_rate']:.4f}",
                           d["recommend_drop_r6"], kinds])


def render_tracked_markdown(payload: dict) -> str:
    """Same content as render_markdown, but normalizer examples are replaced
    with structural descriptions -- no verbatim text."""
    scrubbed = dict(payload)
    if "normalizer" in payload and payload["normalizer"].get("available"):
        scrubbed_norm = json.loads(json.dumps(payload["normalizer"]))  # deep copy
        for ds, d in scrubbed_norm["per_dataset"].items():
            d["examples"] = [
                _structural_description(ex["before"], ex["after"], ex["kind"])
                for ex in payload["normalizer"]["per_dataset"][ds]["examples"]
            ]
        scrubbed["normalizer"] = scrubbed_norm

    L = ["# METHODOLOGY M2.4 — real-corpus measurements (tracked summary, no verbatim text)", ""]
    L.append(f"Sample: {payload['sample_note']}.")
    L.append("")
    L.append("_Normalizer \"what it touched\" examples below are structural "
             "descriptions (edit kind, offset of first difference, length "
             "delta) -- no verbatim corpus text. See `results/m2_4.md` "
             "(git-ignored) for the verbatim before/after.")
    L.append("")
    if "cer" in payload:
        L += _cer_block_md(payload["cer"])
    if "n5" in payload:
        L += _n5_block_md(payload["n5"])
    if "emoji" in payload:
        L += _emoji_block_md(payload["emoji"])
    if "normalizer" in scrubbed:
        nm = scrubbed["normalizer"]
        L += ["## (4) Normalizer touch_rate — R6 go/no-go (M5.4, PER DATASET)", ""]
        if not nm.get("available"):
            L += ["_not installed — measurement skipped._", ""]
        else:
            L.append("M5.4's 5% threshold is applied **per dataset**, not pooled.")
            L.append("")
            L.append("| dataset | n texts | touched | touch_rate | R6 (this dataset) | kinds |")
            L.append("|---|--:|--:|--:|---|---|")
            for ds, d in nm["per_dataset"].items():
                kinds = ", ".join(f"{k} ({v})" for k, v in list(d["kinds"].items())[:4]) or "—"
                verdict = _r6_verdict_label(d["recommend_drop_r6"])
                L.append(f"| {ds} | {d['n_texts']} | {d['touched']} | {d['touch_rate']:.2%} "
                         f"| {verdict} | {kinds} |")
            L.append("")
            L.append("### What it touched — structural description (no verbatim text)")
            L.append("")
            for ds, d in nm["per_dataset"].items():
                if not d["examples"]:
                    continue
                L.append(f"**{ds}:**")
                L.append("")
                L.append("| kind | offset of first diff | length before | length delta |")
                L.append("|---|--:|--:|--:|")
                for ex in d["examples"][:8]:
                    L.append(f"| {ex['kind']} | {ex['offset_first_diff']} | "
                             f"{ex['length_before']} | {ex['length_delta']:+d} |")
                L.append("")
    if "n10" in payload:
        L += _n10_block_md(payload["n10"])
    return "\n".join(L) + "\n"


def count_bengali_codepoints_in_file(path: str) -> int:
    """Sanity check (per instruction): count of characters in U+0980-U+09FF
    in a tracked file -- should be zero outside deliberate glyph labels
    (homophone-set names like শ/ষ/স, matra pairs like ি/ী)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    return sum(1 for ch in text if "ঀ" <= ch <= "৿")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-final-root", default=None,
                    help=f"default: {_DEFAULT_DATA_FINAL}, or the .smoke/ mirror under --limit")
    ap.add_argument("--sample-per-dataset", type=int, default=300)
    ap.add_argument("--seeds", default="42,1337,2024")
    ap.add_argument("--out-results-dir", default=None,
                    help=f"default: {_RESULTS_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--out-dir", default=None,
                    help=f"default: {_OUT_DIR}, or the .smoke/ mirror under --limit")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--measurements", default="1,2,3,4",
                    help="comma list of measurements to run: "
                         "1=CER (slow on real text -- see METHODOLOGY M2.4 "
                         "truncation), 2=N5 firing, 3=emoji coverage, "
                         "4=normalizer touch_rate (+ N10 report, always run "
                         "if data/final/ exists). Default: all four.")
    add_common_args(ap)
    args = ap.parse_args(argv)

    if args.data_final_root is None:
        args.data_final_root = smoke_or_prod(args.limit, _DEFAULT_DATA_FINAL)
    if args.out_results_dir is None:
        args.out_results_dir = smoke_or_prod(args.limit, _RESULTS_DIR)
    if args.out_dir is None:
        args.out_dir = smoke_or_prod(args.limit, _OUT_DIR)
    if args.limit is not None:
        print_smoke_banner("s4_measure", args.limit, {
            "data-final-root": args.data_final_root,
            "out-results-dir": args.out_results_dir, "out-dir": args.out_dir,
        })

    progress = Progress("s4_measure")
    tables_dir = os.path.join(args.out_dir, "tables")
    marker = os.path.join(tables_dir, "m2_4_cer.csv")
    if should_skip(args.out_dir, [marker], args.limit, args.force, progress, "s4_measure"):
        return 0

    seeds = tuple(int(x) for x in args.seeds.split(","))
    want = {int(x) for x in args.measurements.split(",")}
    sample_k = args.limit if args.limit is not None else args.sample_per_dataset
    progress.log("s4_measure", f"sampling {sample_k}/dataset from {args.data_final_root}")
    samples = load_samples(args.data_final_root, sample_k)
    if not samples:
        print(f"no data/final/*/{{train,val,test}}.csv under {args.data_final_root} "
              f"— run pipeline/s2_prepare.py first", file=sys.stderr)
        return 2

    payload: dict = {
        "sample_note": f"{sample_k} texts/dataset (seeded), "
                       f"{{{', '.join(map(str, seeds))}}}; datasets: "
                       f"{', '.join(k for k in samples if k != 'ALL')}",
    }
    timing: dict[str, float] = {}

    def _timed(label: str, fn, *fn_args):
        t0 = time.perf_counter()
        result = fn(*fn_args)
        elapsed = time.perf_counter() - t0
        timing[label] = elapsed
        progress.log("m2_4", f"({label}) done in {elapsed:.1f}s")
        return result

    if 1 in want:
        payload["cer"] = _timed("cer", measure_cer, samples, seeds)
    if 2 in want:
        payload["n5"] = _timed("n5", measure_n5, samples, seeds)
    if 3 in want:
        payload["emoji"] = _timed("emoji", measure_emoji, samples)
    if 4 in want:
        payload["normalizer"] = _timed("normalizer", measure_normalizer, samples)
        payload["n10"] = _timed("n10", measure_n10_test_split, args.data_final_root, args.limit)
    payload["timing"] = timing

    from pipeline.s2_prepare import _safe_relpath, _write_utf8

    progress.log("m2_4", "writing results/ (full detail, git-ignored)")
    out_md = os.path.join(args.out_results_dir, "m2_4.md")
    out_html = os.path.join(args.out_results_dir, "m2_4.html")
    out_json = os.path.join(args.out_results_dir, "m2_4.json")
    _write_utf8(out_md, render_markdown(payload))
    _write_utf8(out_html, render_html(payload))
    _write_utf8(out_json, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    progress.log("m2_4", "writing outputs/ (tracked, no verbatim text)")
    write_tracked_csv_tables(payload, tables_dir)
    tracked_md = os.path.join(args.out_dir, "m2_4_summary.md")
    _write_utf8(tracked_md, render_tracked_markdown(payload))
    n_bn = count_bengali_codepoints_in_file(tracked_md)
    progress.log("m2_4", f"Bengali-block codepoints in tracked summary: {n_bn} "
                         "(expected: only homophone-set/matra glyph labels)")

    if not args.quiet:
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass
        print(render_markdown(payload))
    print(f"[written] {_safe_relpath(out_md, _ROOT)}")
    print(f"[written] {_safe_relpath(out_html, _ROOT)}")
    print(f"[written] {_safe_relpath(out_json, _ROOT)}")
    print(f"[written] {_safe_relpath(tables_dir, _ROOT)}/")
    print(f"[written] {_safe_relpath(tracked_md, _ROOT)}")

    write_run_meta(args.out_dir, limited=args.limit is not None, limit=args.limit,
                   n_rows_processed=sample_k)
    progress.done()
    print("\nWall time per measurement: " +
          ", ".join(f"{k}={v:.1f}s" for k, v in timing.items()))

    nm = payload.get("normalizer")
    if nm and nm.get("available"):
        print("\nR6 recommendation, PER DATASET (M5.4, 5% threshold):")
        for ds, d in nm["per_dataset"].items():
            verdict = _r6_verdict_label(d["recommend_drop_r6"])
            print(f"  {ds}: {verdict} (touch_rate {d['touch_rate']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
