# bangla-noisebench

A severity-graded benchmark of realistic orthographic and typographic noise for
Bangla text, and the released artefact of an IEEE conference paper on the
robustness of Bangla text classifiers to that noise.

**Phase 1 (this release): the perturbation module only.** No training code, model
code, or data loaders yet.

## Install

```bash
pip install -r requirements.txt   # regex + csebuetnlp/normalizer (runtime), pytest (tests)
```

Python 3.10+ is the support target (a portability choice). Kaggle, where the
experiments run, is on Python 3.12.13 (verified 2026-09-01).

### Resolved dependency versions (reproducibility)

`requirements.txt` pins `regex` exactly and the BanglaBERT normalizer to a git
commit. The normalizer pulls two more packages transitively — `emoji` and
`ftfy` — and both can move reported numbers (`ftfy` via the normalizer's
mojibake fixing inside `normalize()`; `emoji` via the M0 Q9 coverage
measurement in `noisebench.validate.emoji_range_coverage`). The normalizer's
own metadata at commit `d405944` **pins them exactly**, so they are effectively
transitively pinned — but confirm the resolved versions in the run environment
and record them here:

| package | role | how pinned | resolved version |
|---------|------|------------|------------------|
| `regex` | grapheme clusters (`\X`) | `requirements.txt` `==2024.11.6` | 2024.11.6 |
| `normalizer` | BanglaBERT normalizer | `requirements.txt` git commit | `d405944` |
| `emoji` | transitive; coverage-measurement reference list only (1.4.2 ≈ Unicode 13 / 2021 — coverage is an upper bound against that vintage) | `normalizer` metadata `==1.4.2` | 1.4.2 _(local; confirm on Kaggle)_ |
| `ftfy` | transitive; runs inside `normalize()` | `normalizer` metadata `==6.0.3` | 6.0.3 _(local; confirm on Kaggle)_ |
| `pytest` | tests only | `requirements.txt` `>=7.4` | _record if reporting test runs_ |

To confirm in the target environment (e.g. a Kaggle notebook cell):

```bash
pip freeze | grep -Ei "^(regex|emoji|ftfy|pytest)=="
python -c "import importlib.metadata as m; print(m.distribution('normalizer').requires)"
python -c "import emoji, ftfy; print(emoji.__version__, ftfy.__version__)"
```

## The API

```python
from noisebench import perturb, perturb_with_stats

perturb(
    text: str,          # any Unicode; NFC-normalized on entry and exit
    noise_type: str,    # one of noisebench.NOISE_TYPES (see table below)
    severity: int,      # 1..5
    seed: int,          # the call is a pure function of these four arguments
) -> str
```

`perturb_with_stats(...)` returns `(text, PerturbStats)` where `PerturbStats`
carries `n_eligible`, `n_applied`, `submodes` (per-mechanism counts) and
`unchanged`. Use it whenever you need to know whether anything actually
happened — the plain `perturb` never signals a no-op.

**Determinism.** Same `(text, noise_type, severity, seed)` ⇒ byte-identical
output, always. The local `random.Random` is seeded from a hash of *all four*
arguments (not `seed` alone), so the per-unit random draws — in particular the
Bernoulli remainder of the count rule below — are **independent across texts**.
Seeding on `seed` alone correlates that draw across a whole corpus and breaks
the expected-rate guarantee for noise types with few eligible units (N7).

### The ten noise types

| id | `noise_type` | what it does |
|----|--------------|--------------|
| N1 | `char_insert` | insert a standalone Bangla character at a cluster boundary inside a word |
| N2 | `char_delete` | delete a grapheme cluster (never the whole of a one-cluster word) |
| N3 | `char_substitute` | replace a character with another of the **same orthographic class** (consonant/independent-vowel/digit), drawn uniformly — no keyboard-layout assumption |
| N4 | `char_transpose` | swap two adjacent grapheme clusters within a word |
| N5 | `homophone_confuse` | swap within a confusable set: শ/ষ/স, ন/ণ, ই/ঈ, উ/ঊ, ি/ী, ু/ূ, র/ড়/ঢ়, জ/য (ব/ভ removed pending corpus evidence — METHODOLOGY M2.4) |
| N6 | `matra_perturb` | drop a dependent vowel sign / hasanta / chandrabindu; or alter within ি↔ী, ু↔ূ |
| N7 | `conjunct_split` | break a two-consonant conjunct by dropping the hasanta (ক্ষ→কষ). ZWNJ insertion was considered and dropped — it's a rendering choice, not an error. |
| N8 | `elongate` | lengthen a vowel: insert the matching independent vowel once, for ু→উ, ূ→ঊ, ি→ই, ী→ঈ, ো→ও only (খুব→খুউব); repeat an independent vowel 2–4× (আ→আআ); or repeat the cluster's last vowel sign 2–4× (খুশি→খুশিিি). Repeating a matra keeps the cluster intact, so it is a one-cluster edit under the CER metric. |
| N9 | `whitespace_error` | split one word in two, or merge two adjacent words |
| N10 | `emoji_transform` | **not severity-graded** — see below |

`conjunct_split` excludes triple conjuncts (স্ত্র) in this release.

### N10 — emoji, a separate signature

N10 is a **three-way preprocessing condition**, not a graded corruption: a
practitioner picks one policy and applies it to the whole corpus. There is no
"fraction of emoji to corrupt" anyone would use, so it takes a `mode` instead of
`(severity, seed)` and is fully deterministic by construction.

```python
from noisebench import emoji_transform

emoji_transform(text, "keep")               # identity (control)
emoji_transform(text, "remove")             # strip emoji + glue, collapse whitespace
emoji_transform(text, "replace_with_text")  # 😀 -> :grinning_face:  (via unicodedata)
```

## The severity scale

Severity maps to the **fraction of eligible units** perturbed. "Eligible unit"
is defined per noise type (METHODOLOGY M1.2) — e.g. a cluster boundary for N1, a
conjunct junction for N7, a matra occurrence for N6.

| severity | fraction |
|----------|----------|
| 1 | 0.05 |
| 2 | 0.10 |
| 3 | 0.20 |
| 4 | 0.35 |
| 5 | 0.50 |

The number of units actually perturbed is `floor(n_eligible × fraction)` plus one
more with probability equal to the fractional remainder (seeded). This keeps the
expected rate exactly `fraction` **and** stops severity 1 from becoming a silent
no-op on short texts, which plain rounding would cause.

Units are chosen with `random.Random(seed).sample`, i.e. without replacement, so
one unit is never perturbed twice in a call.

## Text unit and metric

Every character-level operation works on **Unicode extended grapheme clusters**
(`regex` `\X`) after NFC normalization, so `কি` is one unit and `ড়` is one unit.
Character error rate (CER) is normalized Levenshtein distance over
cluster sequences — `noisebench.validate.cer` / `corpus_cer`.

## Reproducing the Phase 1 report

```bash
python scripts/phase1_report.py
```

writes `results/phase1_report.md` and `results/phase1_report.html` (both UTF-8;
`results/` is gitignored, regenerate as needed) and prints a plain-text copy.
`--out PATH.md`, `--out-html PATH.html`, `--quiet` override the defaults.

The report contains: Table I (one Bangla example per noise type at severity 3,
plus all three N10 modes), the 9×5 mean CER table, the normalizer-survival row,
and PASS/FAIL for the four M3 acceptance criteria.

> The script writes the files itself and never relies on shell redirection —
> PowerShell's `>` corrupts UTF-8 Bangla to mojibake. Open the `.html` in a
> browser for the most reliable Bangla rendering.

## Tests

```bash
pytest -q            # 38 tests
pytest -q -s         # also prints the 9×5 CER table
```

Covers determinism, monotonicity of CER vs severity (the key test), Unicode
validity, ো/ৌ survival through NFC and the BanglaBERT normalizer, non-identity at
severity 5, the nukta letters ড়/ঢ়/য়, the stats contract, and the emoji modes.

## Linguistic content

All linguistic data lives in [`noisebench/bangla_maps.py`](noisebench/bangla_maps.py)
as commented constants, each with a `# TODO(verify):` block. It is pending a
native-speaker review.

```bash
python scripts/constants_sheet.py    # -> results/constants_sheet.{md,html}
```

renders every constant as actual Bangla (glyph + code points + Unicode names)
for that review, and runs structural assertions (pool sizes; no matra/hasanta
in any `IN_CLASS_POOL` or `INSERTABLE_CHARS`; disjoint pools; NFC-idempotence).
The same assertions run in `pytest` (`test_bangla_maps_structural_assertions`).
