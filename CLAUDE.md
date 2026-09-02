# Project rules — bangla-noisebench

Research code for an IEEE conference paper on the robustness of Bangla text
classifiers to orthographic and typographic noise. Deadline: 30 September 2026.
Optimise for correct, simple and testable. Never for clever.

## Source of truth

`METHODOLOGY.md` is the implementation spec (task-ordered M0–M7).
`Topic4_FULL_PAPER_PLAN.md` defines the experiment design and
`Figure_Plan_Evaluation.md` defines the figures. **Where METHODOLOGY.md
disagrees with the plan on an implementation detail, METHODOLOGY.md wins; the
plan wins for scope.** If a task instruction contradicts all of those, stop and
ask — do not silently pick one.

## Settled decisions — do not revisit

- **Python 3.10+ target** for the released artefact — a portability choice, not
  a Kaggle constraint. Kaggle itself runs **Python 3.12.13** (verified
  2026-09-01), so nothing here *needs* to avoid 3.11+ syntax to run there; we
  still do (`from __future__ import annotations` for the `X | None` hints, no
  `match`, etc.) so the shipped package installs on older interpreters.
- `regex` is a runtime dependency, pinned with `==`. Use `\X` for grapheme
  clusters. Everything else in `noisebench/`: standard library only.
- The BanglaBERT normalizer is **`csebuetnlp/normalizer`** (import name
  `normalizer`), pinned to a git commit in `requirements.txt` (not `==` since
  it has no PyPI release, but the commit hash is the equivalent). It is NOT
  `bnunicodenormalizer` — a different package used for OCR/ASR text, not by
  BanglaBERT. `validate.py`'s normalizer-survival check validates this exact
  package; Phase 3's `--normalizer on` flag MUST call the same one
  (`noisebench.validate.normalize_banglabert`) or the survival guarantee does
  not transfer. Pinned because it affects reported results, not test-only.
- N3 is **in-class substitution**, not keyboard adjacency (METHODOLOGY M0.1).
  No Bijoy grid ships. Phonetic-input errors are modelled by N5.
- Severity scale: 1→0.05, 2→0.10, 3→0.20, 4→0.35, 5→0.50. Count rule is
  floor + seeded Bernoulli (M1.3), never plain rounding.
- N10 (emoji) is a 3-way preprocessing condition, not severity-graded.

## Invariants

1. **Determinism.** Every stochastic function takes an explicit `seed` and uses a
   local `random.Random(...)`. Never the global `random`, never global numpy state.
   **Never re-seed an RNG per item with the same seed** — the draws become
   correlated across items (e.g. the first `.random()` call is identical for
   every text, so a per-text Bernoulli fires all-or-none). Use one stream per
   run, or derive the per-item seed by hashing `(seed, item identity)`. See the
   N7 seed-correlation bug: `perturbations._derive_seed`, and its regression
   test `test_bernoulli_remainder_is_decorrelated_across_texts`.
2. **Unicode safety.** Operations on matras and conjuncts must never emit invalid
   sequences. Validate and skip rather than emit broken text. Every guard is
   commented `# GUARD`.
3. **No silent no-ops.** If a perturbation cannot be applied, say so in the returned
   stats (`n_eligible`, `n_applied`, `unchanged`). Never return the input silently.
4. **Seeds are {42, 1337, 2024}** for every experiment. Never a single run.
5. **One flat results file.** All evaluation output appends one row per evaluation to
   `results/results.csv`. Never a second results format, never per-notebook outputs.
6. **Resumability.** Any script that loops over a grid must skip work already present
   in `results.csv`. Kaggle sessions cap at 12 hours.
7. Type hints and docstrings on every public function. This code ships with the paper.
8. **Any script that emits Bangla must write a UTF-8 file itself** (`open(path,
   "w", encoding="utf-8", newline="\n")`), never rely on shell redirection.
   Windows PowerShell's `>` reinterprets UTF-8 under the console code page and
   re-saves as UTF-16, corrupting Bangla to mojibake. Console printing is fine
   as an *extra*, but the file is the authoritative artefact. Prefer Markdown
   (renders in the VS Code preview) plus an HTML variant with
   `<meta charset="utf-8">` and a Bangla font stack
   (`"Noto Sans Bengali", "Nirmala UI", sans-serif`).

## Never do these

- Never invent Bangla words, spellings or orthographic rules. Uncertain linguistic
  content goes in a `CANDIDATE_` constant with a `# TODO(verify):` comment for me.
- Never commit data, model weights or `results.csv` (see `.gitignore`).
- Never change the severity scale, the noise taxonomy or the metric set without asking.
- Never interpret experimental results or write claims about what they mean. Produce
  numbers and figures; the conclusions are mine.
- Never mark a task done because tests pass. See below.

## Definition of done

A task is done when: tests pass, **and** you have printed the actual numbers the
tests are asserting on (not just "5 passed"), **and** you have listed every design
decision you were uncertain about.

For the noise suite specifically, "done" means showing me the 9×5 table of mean
character error rate per noise type per severity. If any noise type is flat across
severities, or near zero at severity 5, the implementation is wrong even if the
test passed. Run `python scripts/phase1_report.py` to produce it.

## Git — the human commits, never the agent

The agent must **never** run `git commit`, `git tag`, `git push`, `git rebase`,
`git reset`, or any history-rewriting command. Read-only git is fine (`status`,
`log`, `diff`, `check-ignore`); `git add` is fine when asked. When work is ready,
the agent says so and proposes a commit message; **I run the commit.** This is a
research artefact published alongside a paper — every commit must carry my
authorship.

Commit at the end of each phase with `phase-N: <what changed>`. Do not amend or
rebase previous commits — I need the history to roll back.
