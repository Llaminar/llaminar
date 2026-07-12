"""End-to-end common policy compiler with frozen generic sealed evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .certification import CertificationReport, certify_generic_policy
from .corpus import ObservationCorpus, RuntimeKey
from .exact_oracle import build_exact_winners
from .format_registry import registry_digest
from .policy_ir import PolicyIR, make_policy_ir
from .profiles import require_promotion_timing
from .segmented_policy import fit_generic_policy
from .splits import SealedPartition


@dataclass(frozen=True)
class CompiledPolicy:
    """Final common IR and the certificate bound to its generic digest."""

    policy_ir: PolicyIR
    certification: CertificationReport


def compile_policy(
    development: ObservationCorpus,
    sealed_partition: SealedPartition,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    max_segments: int = 3,
    min_shape_groups_per_leaf: int = 2,
    require_promotable: bool = True,
) -> CompiledPolicy:
    """Fit, freeze, open sealed evidence, certify, and emit without refitting."""

    if require_promotable:
        # A complete smoke CSV is still only a smoke CSV. Enforce the robust
        # timing budget before development evidence can influence an
        # installable policy artifact.
        require_promotion_timing(development, label="development corpus")

    development_exact = build_exact_winners(
        development, serial_m1_hashes=serial_m1_hashes
    )
    generic = fit_generic_policy(
        development,
        serial_m1_hashes=serial_m1_hashes,
        max_segments=max_segments,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
    )
    common_metadata = {
        "development_corpus_digest": development.digest(),
        "sealed_commitment": sealed_partition.commitment,
        "split_manifest_digest": sealed_partition.manifest_digest,
        "format_registry_digest": registry_digest(),
        **dict(metadata or {}),
    }
    frozen_ir = make_policy_ir(
        development_exact,
        generic,
        metadata=common_metadata,
    )
    frozen_generic_digest = frozen_ir.digest(generic_only=True)
    sealed = sealed_partition.open(frozen_generic_digest)
    if require_promotable:
        require_promotion_timing(sealed, label="sealed corpus")
    certification = certify_generic_policy(
        frozen_ir,
        sealed,
        serial_m1_hashes=serial_m1_hashes,
    )
    if certification.frozen_generic_policy_digest != frozen_generic_digest:
        raise ValueError("sealed certificate does not bind the frozen generic IR")
    if require_promotable:
        certification.require_promotable()

    # Exact overlays may use accepted sealed exact-shape measurements, but the
    # generic rules are copied byte-for-byte from the pre-open frozen policy.
    combined = ObservationCorpus((*development.observations, *sealed.observations))
    final_exact = build_exact_winners(combined, serial_m1_hashes=serial_m1_hashes)
    final_ir = make_policy_ir(
        final_exact,
        generic,
        metadata={
            **common_metadata,
            "sealed_corpus_digest": sealed.digest(),
            "frozen_generic_policy_digest": frozen_generic_digest,
            "sealed_max_regret": certification.max_observed_regret,
            "sealed_max_simultaneous_ucb": (
                certification.max_simultaneous_95pct_upper_regret
            ),
        },
    )
    if final_ir.digest(generic_only=True) != frozen_generic_digest:
        raise ValueError("generic policy changed after sealed certification")
    return CompiledPolicy(policy_ir=final_ir, certification=certification)
