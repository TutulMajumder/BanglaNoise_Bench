"""Severity scale for the Bangla noise benchmark.

A severity level (1..5) maps to a **fraction of eligible units** to perturb.
"Eligible units" are defined per noise type in ``perturbations.py`` /
METHODOLOGY M1.2.

The scale is fixed by the paper and must not be changed without sign-off
(see CLAUDE.md "Settled decisions"):

    1 -> 0.05    2 -> 0.10    3 -> 0.20    4 -> 0.35    5 -> 0.50
"""

from __future__ import annotations

import random

# METHODOLOGY / Topic4 plan: severity = fraction of eligible units perturbed.
SEVERITY_TO_FRACTION: dict[int, float] = {
    1: 0.05,
    2: 0.10,
    3: 0.20,
    4: 0.35,
    5: 0.50,
}

MIN_SEVERITY = 1
MAX_SEVERITY = 5


def fraction_for_severity(severity: int) -> float:
    """Return the perturbation fraction for a severity level.

    Args:
        severity: integer in ``[1, 5]``.

    Returns:
        The fraction of eligible units to perturb (e.g. ``0.20`` for severity 3).

    Raises:
        ValueError: if ``severity`` is not an integer in ``[1, 5]``.
    """
    if not isinstance(severity, int) or isinstance(severity, bool):
        raise ValueError(f"severity must be an int, got {type(severity).__name__}")
    if severity not in SEVERITY_TO_FRACTION:
        raise ValueError(
            f"severity must be one of {sorted(SEVERITY_TO_FRACTION)}, got {severity}"
        )
    return SEVERITY_TO_FRACTION[severity]


def n_to_perturb(n_eligible: int, frac: float, rng: random.Random) -> int:
    """Decide how many eligible units to perturb, without naive rounding.

    Naive rounding (``round(n * frac)``) makes severity 1 a silent no-op on
    short texts: a 12-cluster comment gives ``0.05 * 12 = 0.6 -> 1`` with
    ``round`` but ``0`` with ``int``/truncation, and either way the *expected*
    rate is wrong. Instead we take the deterministic floor and add one more
    unit with probability equal to the fractional remainder, drawn from the
    caller's seeded RNG. Over many texts the expected count is exactly
    ``n_eligible * frac`` (METHODOLOGY M1.3).

    Args:
        n_eligible: number of eligible units in this text (>= 0).
        frac: target fraction in ``[0, 1]`` (from ``fraction_for_severity``).
        rng: a seeded ``random.Random`` instance. Never the global ``random``
            module (determinism invariant).

    Returns:
        Integer count ``k`` with ``0 <= k <= n_eligible``.
    """
    if n_eligible <= 0:
        return 0
    exact = n_eligible * frac
    k = int(exact)  # deterministic floor
    remainder = exact - k
    if remainder > 0.0 and rng.random() < remainder:
        k += 1
    return min(k, n_eligible)
