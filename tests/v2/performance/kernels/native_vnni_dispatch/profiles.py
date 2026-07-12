"""Measurement profiles and hard promotion-quality gates.

Workflow smokes are useful for validating a harness, but they are not policy
evidence.  Keeping that distinction in executable code prevents a short run
from becoming installable merely because its CSV happens to be complete.
"""

from __future__ import annotations

from enum import Enum

from .corpus import ObservationCorpus
from .schema import NativeVNNIObservation


MIN_PROMOTION_WARMUPS = 5
MIN_PROMOTION_SAMPLES = 30


class MeasurementProfile(str, Enum):
    """Supported trainer profiles from schema smoke through full production."""

    QUICK = "quick"
    FAMILY_SMOKE = "family-smoke"
    PARTIAL_PRODUCTION = "partial-production"
    PRODUCTION = "production"

    @property
    def installable(self) -> bool:
        """Return whether this profile may contribute to an installed bundle."""

        return self in {self.PARTIAL_PRODUCTION, self.PRODUCTION}


def observation_has_promotion_timing(observation: NativeVNNIObservation) -> bool:
    """Return whether one row satisfies the default robust timing protocol."""

    return (
        observation.warmup_count >= MIN_PROMOTION_WARMUPS
        and observation.sample_count >= MIN_PROMOTION_SAMPLES
        and bool(observation.timing_sample_hash.strip())
    )


def require_promotion_timing(corpus: ObservationCorpus, *, label: str) -> None:
    """Reject the first row measured with a smoke-only timing budget."""

    for observation in corpus:
        if observation_has_promotion_timing(observation):
            continue
        raise ValueError(
            f"{label}: non-promotable timing for {observation.candidate_id} "
            f"at {observation.shape_name}/M={observation.m}: "
            f"warmups={observation.warmup_count} samples={observation.sample_count}; "
            f"required>={MIN_PROMOTION_WARMUPS}/{MIN_PROMOTION_SAMPLES}"
        )
