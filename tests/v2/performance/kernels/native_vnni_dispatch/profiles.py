"""Measurement profiles and hard promotion-quality gates.

Workflow smokes are useful for validating a harness, but they are not policy
evidence.  Keeping that distinction in executable code prevents a short run
from becoming installable merely because its CSV happens to be complete.
"""

from __future__ import annotations

import math
from enum import Enum

from .corpus import ObservationCorpus
from .schema import NativeVNNIObservation


MIN_PROMOTION_WARMUPS = 5
MIN_PROMOTION_SAMPLES = 30
LEGACY_ADAPTIVE_TIMING_PROTOCOL = "elapsed-stability-interleaved-v13"
ADAPTIVE_TIMING_PROTOCOL = "elapsed-stability-interleaved-v16"
SUPPORTED_ADAPTIVE_TIMING_PROTOCOLS = frozenset({
    LEGACY_ADAPTIVE_TIMING_PROTOCOL,
    ADAPTIVE_TIMING_PROTOCOL,
})
LEGACY_ADAPTIVE_TIMING_CEILING_POLICY = "samples-and-elapsed-v1"
ADAPTIVE_TIMING_CEILING_POLICY = "samples-and-four-elapsed-v3"
ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER = 4.0
MIN_PROMOTION_WARMUP_DURATION_US = 50_000.0
MIN_PROMOTION_TRANSITION_WARMUP_DURATION_US = 50_000.0
MIN_PROMOTION_TRANSITION_WARMUP_LATENCY_MULTIPLIER = 60.0
MIN_PROMOTION_TRANSITION_WARMUP_BUDGET_CEILING_US = 500_000.0
MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US = 30_000_000.0
CPU_PREFILL_DYNAMIC_WATCHDOG_MULTIPLIER = 4.0
CPU_PREFILL_DYNAMIC_WATCHDOG_PADDING_US = 1_000_000.0
MIN_PROMOTION_MAXIMUM_SAMPLES = 180
MIN_PROMOTION_ADAPTIVE_SAMPLES = 5
MIN_PROMOTION_TIMED_DURATION_US = 100_000.0
MAX_PROMOTION_MEDIAN_RELATIVE_DRIFT = 0.02
MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS = 3

# A candidate-expansion cohort is measured in complete rounds with an existing
# source candidate as a contemporaneous anchor. The anchor removes the
# cross-session clock axis before economy fitting, so the incremental cohort
# can use its two byte-parity route launches directly before acquisition rather
# than paying for a second, discarded warmup phase. Its 100 ms floor is paired
# with a launch-population floor and shuffled complete rounds, so fast kernels
# contribute many observations while expensive kernels cannot pass on a chance
# two-launch agreement. Raw half-window drift remains authenticated diagnostic
# evidence; it is not seven independent gates over one shared epoch. Historical
# expansion shards requested a 60-sample recovery ceiling and remain valid
# immutable evidence. Current collection requests 180, but neither cohort is
# promotable until the contemporaneous anchor proof below removes the session
# clock axis. This bounded protocol is never valid for a standalone corpus.
CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA = (
    "cpu-prefill-candidate-expansion-anchor-normalization-v1"
)
CANDIDATE_EXPANSION_EVIDENCE_KEY = "candidate_expansion_normalization"
CANDIDATE_EXPANSION_TIMING_STOP_REASON = "anchored_complete_rounds"
MIN_CANDIDATE_EXPANSION_WARMUPS = 2
MIN_CANDIDATE_EXPANSION_WARMUP_DURATION_US = 0.0
MIN_CANDIDATE_EXPANSION_TRANSITION_WARMUP_DURATION_US = 0.0
MIN_CANDIDATE_EXPANSION_TIMED_DURATION_US = 100_000.0
MIN_CANDIDATE_EXPANSION_ADAPTIVE_SAMPLES = 5
MIN_CANDIDATE_EXPANSION_MAXIMUM_SAMPLES = 60


LEGACY_ADAPTIVE_PROMOTION_EVIDENCE_FIELDS = frozenset({
    "coordination_policy",
    "mpi_world_size",
    "preconditioning_policy",
    "preconditioning_m",
    "preconditioning_budget_us",
    "preconditioning_duration_us",
    "warmup_budget_policy",
    "warmup_budget_us",
    "warmup_duration_us",
    "warmup_latency_multiplier",
    "warmup_budget_ceiling_us",
    "warmup_round_timeout_us",
    "timing_protocol",
    "timing_ceiling_policy",
    "minimum_samples",
    "stable_samples",
    "maximum_samples",
    "timing_budget_us",
    "timed_duration_us",
    "median_stability_limit",
    "median_relative_drift",
    "timing_converged",
    "stop_reason",
})
ADAPTIVE_PROMOTION_EVIDENCE_FIELDS = frozenset({
    *LEGACY_ADAPTIVE_PROMOTION_EVIDENCE_FIELDS,
    "stationary_sample_begin",
    "stationary_sample_count",
    "stationary_duration_us",
})


def adaptive_timing_evidence_is_promotable(
    evidence: dict[str, object],
    *,
    warmup_count: int,
    sample_count: int,
) -> bool:
    """Recompute production acceptance from persisted adaptive diagnostics.

    The backend adapter authenticates every field against raw timing samples.
    Persisting the reviewed subset here lets a later compiler re-establish the
    promotion decision without trusting a transient in-memory adapter result.
    """

    normalized_evidence = dict(evidence)
    expansion = normalized_evidence.pop(
        CANDIDATE_EXPANSION_EVIDENCE_KEY,
        None,
    )
    try:
        complete_round_probe_duration_us = float(
            normalized_evidence.pop("complete_round_probe_duration_us", 0.0)
        )
        timing_protocol = str(normalized_evidence["timing_protocol"])
        expected_fields = (
            ADAPTIVE_PROMOTION_EVIDENCE_FIELDS
            if timing_protocol == ADAPTIVE_TIMING_PROTOCOL
            else LEGACY_ADAPTIVE_PROMOTION_EVIDENCE_FIELDS
        )
        if (
            timing_protocol not in SUPPORTED_ADAPTIVE_TIMING_PROTOCOLS
            or set(normalized_evidence) != expected_fields
        ):
            return False
        warmup_floor = MIN_PROMOTION_WARMUP_DURATION_US
        transition_floor = MIN_PROMOTION_TRANSITION_WARMUP_DURATION_US
        timing_floor = MIN_PROMOTION_TIMED_DURATION_US
        adaptive_sample_floor = MIN_PROMOTION_ADAPTIVE_SAMPLES
        if expansion is not None:
            if not isinstance(expansion, dict) or set(expansion) != {
                "schema_version",
                "plan_digest",
                "status",
                "anchor_candidate_id",
                "source_anchor_median_us_hex",
                "expansion_anchor_median_us_hex",
                "scale_hex",
            }:
                return False
            if (
                expansion["schema_version"]
                != CANDIDATE_EXPANSION_NORMALIZATION_SCHEMA
                or expansion["status"]
                != "scaled_to_immutable_source_anchor"
                or not str(expansion["plan_digest"]).startswith("sha256:")
                or not str(expansion["anchor_candidate_id"]).strip()
            ):
                return False
            source_anchor = float.fromhex(
                str(expansion["source_anchor_median_us_hex"])
            )
            current_anchor = float.fromhex(
                str(expansion["expansion_anchor_median_us_hex"])
            )
            scale = float.fromhex(str(expansion["scale_hex"]))
            if (
                not all(math.isfinite(value) for value in (
                    source_anchor, current_anchor, scale
                ))
                or source_anchor <= 0.0
                or current_anchor <= 0.0
                or scale <= 0.0
                or abs(scale - source_anchor / current_anchor)
                > 1.0e-15 * max(1.0, scale)
            ):
                return False
            warmup_floor = MIN_CANDIDATE_EXPANSION_WARMUP_DURATION_US
            transition_floor = (
                MIN_CANDIDATE_EXPANSION_TRANSITION_WARMUP_DURATION_US
            )
            timing_floor = MIN_CANDIDATE_EXPANSION_TIMED_DURATION_US
            adaptive_sample_floor = MIN_CANDIDATE_EXPANSION_ADAPTIVE_SAMPLES

        minimum_samples = int(normalized_evidence["minimum_samples"])
        stable_samples = int(normalized_evidence["stable_samples"])
        maximum_samples = int(normalized_evidence["maximum_samples"])
        timing_budget_us = float(normalized_evidence["timing_budget_us"])
        timed_duration_us = float(normalized_evidence["timed_duration_us"])
        stability_limit = float(normalized_evidence["median_stability_limit"])
        timing_converged = normalized_evidence["timing_converged"] is True
        stop_reason = str(normalized_evidence["stop_reason"])
        if timing_protocol == ADAPTIVE_TIMING_PROTOCOL:
            stationary_begin = int(
                normalized_evidence["stationary_sample_begin"]
            )
            stationary_count = int(
                normalized_evidence["stationary_sample_count"]
            )
            stationary_duration_us = float(
                normalized_evidence["stationary_duration_us"]
            )
            timing_stop_complete = (
                (
                    stop_reason == "stationary_window"
                    and stationary_begin >= 0
                    and stationary_count >= stable_samples
                    and stationary_begin + stationary_count == sample_count
                    and stationary_duration_us >= timing_budget_us
                    and stationary_duration_us <= timed_duration_us + 5.1e-7
                )
                or (
                    expansion is not None
                    and stop_reason == CANDIDATE_EXPANSION_TIMING_STOP_REASON
                    and stationary_begin == 0
                    and stationary_count == sample_count
                    and stationary_count >= stable_samples
                    and stationary_duration_us >= timing_budget_us
                    and abs(stationary_duration_us - timed_duration_us)
                    <= 5.1e-7
                )
            )
            adaptive_sample_complete = (
                stationary_count >= adaptive_sample_floor
            )
        else:
            timing_stop_complete = (
                (
                    stop_reason == "elapsed_stable"
                    and sample_count >= minimum_samples
                    and timed_duration_us >= timing_budget_us
                )
                or (
                    stop_reason == "stable_sample_floor"
                    and sample_count >= stable_samples
                    and stable_samples >= MIN_PROMOTION_SAMPLES
                )
            )
            adaptive_sample_complete = (
                minimum_samples >= adaptive_sample_floor
            )
        required_warmups = (
            MIN_CANDIDATE_EXPANSION_WARMUPS
            if expansion is not None
            else MIN_PROMOTION_WARMUPS
        )
        preconditioning_complete = (
            str(normalized_evidence["preconditioning_policy"])
            == "source-complete-round-v1"
            and int(normalized_evidence["preconditioning_m"]) >= 2
            and float(normalized_evidence["preconditioning_budget_us"])
            >= warmup_floor
            and float(normalized_evidence["preconditioning_duration_us"])
            >= float(normalized_evidence["preconditioning_budget_us"])
        )
        if expansion is not None:
            preconditioning_complete = preconditioning_complete or (
                str(normalized_evidence["preconditioning_policy"]) == "disabled"
                and int(normalized_evidence["preconditioning_m"]) == 0
                and float(normalized_evidence["preconditioning_budget_us"]) == 0.0
                and float(normalized_evidence["preconditioning_duration_us"]) == 0.0
            )
        warmup_policy_complete = (
            str(normalized_evidence["warmup_budget_policy"])
            == "source-fixed-v1"
            or (
                expansion is not None
                and str(normalized_evidence["warmup_budget_policy"])
                == "transition-fixed-v1"
                and float(normalized_evidence["warmup_latency_multiplier"]) == 0.0
                and float(normalized_evidence["warmup_budget_ceiling_us"]) == 0.0
            )
            or (
                float(normalized_evidence["warmup_latency_multiplier"])
                >= MIN_PROMOTION_TRANSITION_WARMUP_LATENCY_MULTIPLIER
                and float(normalized_evidence["warmup_budget_ceiling_us"])
                >= MIN_PROMOTION_TRANSITION_WARMUP_BUDGET_CEILING_US
            )
        )
        if expansion is not None:
            warmup_policy_complete = warmup_policy_complete or (
                str(normalized_evidence["warmup_budget_policy"]) == "disabled"
                and float(normalized_evidence["warmup_budget_us"]) == 0.0
                and float(normalized_evidence["warmup_duration_us"]) == 0.0
                and float(normalized_evidence["warmup_latency_multiplier"]) == 0.0
                and float(normalized_evidence["warmup_budget_ceiling_us"]) == 0.0
            )
        warmup_round_timeout_us = float(
            normalized_evidence["warmup_round_timeout_us"]
        )
        if complete_round_probe_duration_us > 0.0:
            expected_round_timeout_us = max(
                MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US,
                complete_round_probe_duration_us
                * CPU_PREFILL_DYNAMIC_WATCHDOG_MULTIPLIER
                + CPU_PREFILL_DYNAMIC_WATCHDOG_PADDING_US,
            )
            watchdog_complete = math.isclose(
                warmup_round_timeout_us,
                expected_round_timeout_us,
                rel_tol=0.0,
                abs_tol=5.1e-7,
            )
        else:
            watchdog_complete = (
                warmup_round_timeout_us
                <= MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US
            )
        return (
            warmup_count >= required_warmups
            and (
                timing_protocol == ADAPTIVE_TIMING_PROTOCOL
                or (
                    expansion is None
                    and timing_protocol == LEGACY_ADAPTIVE_TIMING_PROTOCOL
                )
            )
            and str(normalized_evidence["timing_ceiling_policy"])
            == (
                ADAPTIVE_TIMING_CEILING_POLICY
                if timing_protocol == ADAPTIVE_TIMING_PROTOCOL
                else LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
            )
            and str(normalized_evidence["coordination_policy"])
            == "mpi-complete-round-v1"
            and 1 <= int(normalized_evidence["mpi_world_size"]) <= 2
            and preconditioning_complete
            and float(normalized_evidence["warmup_budget_us"])
            >= transition_floor
            and warmup_policy_complete
            and float(normalized_evidence["warmup_duration_us"])
            >= float(normalized_evidence["warmup_budget_us"])
            and watchdog_complete
            # Current evidence carries an authenticated stationary suffix.  A
            # later policy may therefore raise its admission floor and reuse a
            # shard whose *observed* suffix is already strong enough even when
            # the historical requested minimum was lower.  Legacy protocols
            # lack that proof and must continue to satisfy their configured
            # minimum directly.
            and adaptive_sample_complete
            and maximum_samples >= (
                MIN_CANDIDATE_EXPANSION_MAXIMUM_SAMPLES
                if expansion is not None
                else MIN_PROMOTION_MAXIMUM_SAMPLES
            )
            and sample_count >= minimum_samples
            and timing_budget_us >= timing_floor
            and stability_limit <= MAX_PROMOTION_MEDIAN_RELATIVE_DRIFT
            and (
                (
                    float(normalized_evidence["median_relative_drift"])
                    <= stability_limit
                    and timing_converged
                )
                or (
                    expansion is not None
                    and stop_reason == CANDIDATE_EXPANSION_TIMING_STOP_REASON
                )
            )
            and timing_stop_complete
        )
    except (TypeError, ValueError):
        return False


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


def observation_has_promotion_timing(
    observation: NativeVNNIObservation,
    *,
    minimum_warmups: int = MIN_PROMOTION_WARMUPS,
    minimum_samples: int = MIN_PROMOTION_SAMPLES,
) -> bool:
    """Return whether one row satisfies its reviewed timing protocol floor."""

    if not observation.timing_sample_hash.strip():
        return False
    if observation.adaptive_timing_evidence:
        return adaptive_timing_evidence_is_promotable(
            observation.adaptive_timing_evidence,
            warmup_count=observation.warmup_count,
            sample_count=observation.sample_count,
        )
    return (
        observation.warmup_count >= minimum_warmups
        and observation.sample_count >= minimum_samples
    )


def require_promotion_timing(
    corpus: ObservationCorpus,
    *,
    label: str,
    minimum_warmups: int = MIN_PROMOTION_WARMUPS,
    minimum_samples: int = MIN_PROMOTION_SAMPLES,
) -> None:
    """Reject the first row measured with a smoke-only timing budget."""

    if minimum_warmups < 1 or minimum_samples < 1:
        raise ValueError("promotion timing floors must be positive")

    for observation in corpus:
        if not observation.supported or not observation.forced_route_ok:
            continue
        if observation_has_promotion_timing(
            observation,
            minimum_warmups=minimum_warmups,
            minimum_samples=minimum_samples,
        ):
            continue
        raise ValueError(
            f"{label}: non-promotable timing for {observation.candidate_id} "
            f"at {observation.shape_name}/M={observation.m}: "
            f"warmups={observation.warmup_count} samples={observation.sample_count}; "
            f"required>={minimum_warmups}/{minimum_samples}"
        )
