"""noisebench -- a severity-graded orthographic/typographic noise benchmark for Bangla.

Phase 1 public surface:

    from noisebench import perturb, perturb_with_stats, PerturbStats
    from noisebench import emoji_transform, emoji_transform_with_stats
    from noisebench import NOISE_TYPES, EMOJI_MODES, SEVERITY_TO_FRACTION

See ``perturbations.py`` for the API contract and ``METHODOLOGY.md`` sections
M0-M3 for the spec this is written against.
"""

from __future__ import annotations

from .perturbations import (
    EMOJI_MODES,
    NOISE_TYPES,
    PerturbStats,
    clusters,
    emoji_transform,
    emoji_transform_with_stats,
    perturb,
    perturb_with_stats,
)
from .severity import (
    SEVERITY_TO_FRACTION,
    fraction_for_severity,
    n_to_perturb,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "perturb",
    "perturb_with_stats",
    "PerturbStats",
    "emoji_transform",
    "emoji_transform_with_stats",
    "clusters",
    "NOISE_TYPES",
    "EMOJI_MODES",
    "SEVERITY_TO_FRACTION",
    "fraction_for_severity",
    "n_to_perturb",
]
