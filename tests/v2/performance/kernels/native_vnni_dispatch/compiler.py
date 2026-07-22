"""End-to-end common policy compiler with frozen generic sealed evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .certification import CertificationReport, certify_generic_policy
from .corpus import GenericDomain, ObservationCorpus, RuntimeKey
from .exact_oracle import build_exact_winners
from .format_registry import registry_digest
from .paired_confirmation import PairedCellKey, PairedTimingComparison
from .policy_ir import PolicyIR, make_policy_ir
from .profiles import (
    MIN_PROMOTION_SAMPLES,
    MIN_PROMOTION_WARMUPS,
    require_promotion_timing,
)
from .profiler_model import ProfilerFeatureCatalog
from .schema import MINIMUM_PASSING_DOMAIN_FRACTION, P95_REGRET_BUDGET
from .segmented_policy import (
    CandidatePointCost,
    DEFAULT_TREE_LEAVES,
    DomainPromotionDiagnostic,
    GenericPolicy,
    PolicyFitCache,
    domain_promotion_quota_is_satisfied,
    fit_generic_policy,
)
from .splits import SealedPartition


@dataclass(frozen=True)
class CompiledPolicy:
    """Final common IR and the certificate bound to its generic digest."""

    policy_ir: PolicyIR
    certification: CertificationReport


@dataclass(frozen=True)
class FrozenPolicy:
    """Development-fitted IR whose generic digest predates sealed access."""

    policy_ir: PolicyIR
    promotion_diagnostics: tuple[DomainPromotionDiagnostic, ...] = ()

    @property
    def generic_digest(self) -> str:
        """Return the immutable identity consumed by sealed certification."""

        return self.policy_ir.digest(generic_only=True)


def _require_certificate_promotion_criteria(
    frozen: FrozenPolicy,
    certification: CertificationReport,
) -> None:
    """Bind sealed decisions to the criteria frozen before evidence opened."""

    expected_p95 = frozen.policy_ir.metadata.get(
        "promotion_p95_regret_budget"
    )
    expected_fraction = frozen.policy_ir.metadata.get(
        "promotion_minimum_passing_domain_fraction"
    )
    if (
        expected_p95 != certification.p95_regret_budget
        or expected_fraction != certification.minimum_passing_domain_fraction
    ):
        raise ValueError(
            "sealed certification promotion criteria differ from the "
            "development-frozen policy"
        )


def freeze_policy(
    development: ObservationCorpus,
    *,
    sealed_commitment: str,
    split_manifest_digest: str,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    paired_development_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
    supplemental_development_costs: Mapping[
        GenericDomain, tuple[CandidatePointCost, ...]
    ] | None = None,
    metadata: Mapping[str, object] | None = None,
    max_leaves: int = DEFAULT_TREE_LEAVES,
    min_shape_groups_per_leaf: int = 2,
    require_promotable: bool = True,
    minimum_promotion_warmups: int = MIN_PROMOTION_WARMUPS,
    minimum_promotion_samples: int = MIN_PROMOTION_SAMPLES,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
    fit_cache: PolicyFitCache | None = None,
) -> FrozenPolicy:
    """Fit and freeze one policy without receiving sealed observations.

    The function signature deliberately has no sealed-corpus parameter. This
    makes the development/sealed lifetime mechanically reviewable: a caller
    must publish ``generic_digest`` before a separate certification call can
    consume any held-out measurements.
    """

    if not sealed_commitment.strip() or not split_manifest_digest.strip():
        raise ValueError("freezing requires sealed and split commitments")
    if require_promotable:
        require_promotion_timing(
            development,
            label="development corpus",
            minimum_warmups=minimum_promotion_warmups,
            minimum_samples=minimum_promotion_samples,
        )

    development_exact = build_exact_winners(
        development, serial_m1_hashes=serial_m1_hashes
    )
    generic = fit_generic_policy(
        development,
        serial_m1_hashes=serial_m1_hashes,
        paired_comparisons=paired_development_comparisons,
        supplemental_development_costs=supplemental_development_costs,
        max_leaves=max_leaves,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        profiler_feature_catalog=profiler_feature_catalog,
        fit_cache=fit_cache,
    )
    development_required_domain_count = len(generic.cross_validation)
    development_rejected_domain_count = len(generic.promotion_diagnostics)
    development_passing_domain_count = (
        development_required_domain_count - development_rejected_domain_count
    )
    development_passing_domain_fraction = (
        float(development_passing_domain_count)
        / float(development_required_domain_count)
        if development_required_domain_count
        else 0.0
    )
    common_metadata = {
        "development_corpus_digest": development.digest(),
        "sealed_commitment": sealed_commitment,
        "split_manifest_digest": split_manifest_digest,
        "format_registry_digest": registry_digest(),
        "profiler_feature_catalog_digest": (
            profiler_feature_catalog.digest
            if profiler_feature_catalog is not None
            else None
        ),
        "development_required_domain_count": (
            development_required_domain_count
        ),
        "development_passing_domain_count": development_passing_domain_count,
        "development_passing_domain_fraction": (
            development_passing_domain_fraction
        ),
        "minimum_promotion_warmups": minimum_promotion_warmups,
        "minimum_promotion_samples": minimum_promotion_samples,
        "promotion_p95_regret_budget": P95_REGRET_BUDGET,
        "promotion_minimum_passing_domain_fraction": (
            MINIMUM_PASSING_DOMAIN_FRACTION
        ),
        "development_domain_promotion_quota_satisfied": (
            domain_promotion_quota_is_satisfied(
                development_passing_domain_count,
                development_required_domain_count,
                minimum_passing_fraction=(
                    MINIMUM_PASSING_DOMAIN_FRACTION
                ),
            )
        ),
        "development_over_budget_domains": [
            diagnostic.domain
            for diagnostic in generic.promotion_diagnostics
            if diagnostic.rejection_stage not in {
                "cross_validation_missing",
                "cross_validation_coverage",
                "final_fit_empty",
            }
        ],
        **dict(metadata or {}),
    }
    return FrozenPolicy(
        make_policy_ir(
            development_exact,
            generic,
            metadata=common_metadata,
        ),
        promotion_diagnostics=generic.promotion_diagnostics,
    )


def certify_frozen_policy(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    sealed: ObservationCorpus,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    require_promotable: bool = True,
) -> CompiledPolicy:
    """Certify a frozen generic IR and add exact entries without refitting."""

    if development.digest() != frozen.policy_ir.metadata.get(
        "development_corpus_digest"
    ):
        raise ValueError("development corpus changed after generic freeze")
    if require_promotable:
        require_promotion_timing(sealed, label="sealed corpus")

    certification = certify_generic_policy(
        frozen.policy_ir,
        sealed,
        serial_m1_hashes=serial_m1_hashes,
    )
    if certification.frozen_generic_policy_digest != frozen.generic_digest:
        raise ValueError("sealed certificate does not bind the frozen generic IR")
    _require_certificate_promotion_criteria(frozen, certification)
    if require_promotable:
        certification.require_promotable()

    if (
        development.distinguishes_execution_mode
        != sealed.distinguishes_execution_mode
        or development.distinguishes_aspect_bucket
        != sealed.distinguishes_aspect_bucket
    ):
        raise ValueError(
            "development and sealed corpora use different runtime projections"
        )
    combined = ObservationCorpus(
        (
            *development.observations,
            *sealed.observations,
        ),
        distinguish_execution_mode=development.distinguishes_execution_mode,
        distinguish_aspect_bucket=development.distinguishes_aspect_bucket,
    )
    final_exact = build_exact_winners(
        combined, serial_m1_hashes=serial_m1_hashes
    )
    # Reconstructing this immutable wrapper copies the already-frozen rules;
    # no cost matrix or learner is invoked after sealed evidence is visible.
    generic = GenericPolicy(
        rules=frozen.policy_ir.generic_rules,
        unpromoted_domains=frozen.policy_ir.unpromoted_domains,
        cross_validation=frozen.policy_ir.cross_validation,
    )
    final_ir = make_policy_ir(
        final_exact,
        generic,
        metadata={
            **dict(frozen.policy_ir.metadata),
            "sealed_corpus_digest": sealed.digest(),
            "frozen_generic_policy_digest": frozen.generic_digest,
            "sealed_p95_regret": certification.p95_observed_regret,
            "sealed_p95_simultaneous_ucb": (
                certification.p95_simultaneous_95pct_upper_regret
            ),
            "sealed_required_domain_count": (
                certification.required_domain_count
            ),
            "sealed_passing_domain_count": (
                certification.passing_domain_count()
            ),
            "sealed_passing_domain_fraction": (
                certification.passing_domain_fraction
            ),
            # Global p95 and worst-cell values are diagnostics; installation
            # is owned by the frozen p95 and passing-domain criteria.
            "sealed_max_regret": certification.max_observed_regret,
            "sealed_max_simultaneous_ucb": (
                certification.max_simultaneous_95pct_upper_regret
            ),
        },
    )
    if final_ir.digest(generic_only=True) != frozen.generic_digest:
        raise ValueError("generic policy changed after sealed certification")
    return CompiledPolicy(final_ir, certification)


def finalize_frozen_policy_certificate(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    certification: CertificationReport,
    *,
    sealed_evidence_digest: str,
) -> CompiledPolicy:
    """Attach an independently produced sealed certificate without refitting.

    Some production surfaces use a post-freeze paired tournament instead of a
    broad sealed :class:`ObservationCorpus`.  The paired collector measures
    only generated certification geometries, so those shapes must not become
    exact production overlays.  This function therefore preserves the frozen
    development exact entries and generic IR verbatim while binding the final
    artifact to the paired evidence digest and complete certificate.
    """

    if development.digest() != frozen.policy_ir.metadata.get(
        "development_corpus_digest"
    ):
        raise ValueError("development corpus changed after generic freeze")
    if certification.frozen_generic_policy_digest != frozen.generic_digest:
        raise ValueError("sealed certificate does not bind the frozen generic IR")
    _require_certificate_promotion_criteria(frozen, certification)
    if not sealed_evidence_digest.startswith("sha256:"):
        raise ValueError("sealed paired evidence lacks a content digest")
    # The paired seal is generic-only evidence. Generated seal geometries must
    # never become exact overlays, and recreating exact winners would discard
    # development-only selection metadata. Preserve every immutable IR field
    # directly and replace only the certificate-bearing metadata mapping.
    final_ir = PolicyIR(
        policy_abi=frozen.policy_ir.policy_abi,
        learner_version=frozen.policy_ir.learner_version,
        feature_schema_version=frozen.policy_ir.feature_schema_version,
        exact_entries=frozen.policy_ir.exact_entries,
        generic_rules=frozen.policy_ir.generic_rules,
        unpromoted_domains=frozen.policy_ir.unpromoted_domains,
        cross_validation=frozen.policy_ir.cross_validation,
        metadata={
            **dict(frozen.policy_ir.metadata),
            "sealed_evidence_digest": sealed_evidence_digest,
            "sealed_evidence_kind": "paired-frozen-leaf-tournament",
            "frozen_generic_policy_digest": frozen.generic_digest,
            "sealed_p95_regret": certification.p95_observed_regret,
            "sealed_p95_simultaneous_ucb": (
                certification.p95_simultaneous_95pct_upper_regret
            ),
            "sealed_required_domain_count": certification.required_domain_count,
            "sealed_passing_domain_count": certification.passing_domain_count(),
            "sealed_passing_domain_fraction": certification.passing_domain_fraction,
            "sealed_max_regret": certification.max_observed_regret,
            "sealed_max_simultaneous_ucb": (
                certification.max_simultaneous_95pct_upper_regret
            ),
        },
    )
    if final_ir.digest(generic_only=True) != frozen.generic_digest:
        raise ValueError("generic policy changed after paired sealed certification")
    return CompiledPolicy(final_ir, certification)


def compile_policy(
    development: ObservationCorpus,
    sealed_partition: SealedPartition,
    *,
    serial_m1_hashes: Mapping[RuntimeKey, str] | None = None,
    paired_development_comparisons: Mapping[
        PairedCellKey, tuple[PairedTimingComparison, ...]
    ] | None = None,
    supplemental_development_costs: Mapping[
        GenericDomain, tuple[CandidatePointCost, ...]
    ] | None = None,
    metadata: Mapping[str, object] | None = None,
    max_leaves: int = DEFAULT_TREE_LEAVES,
    min_shape_groups_per_leaf: int = 2,
    require_promotable: bool = True,
) -> CompiledPolicy:
    """Fit, freeze, open sealed evidence, certify, and emit without refitting."""

    frozen = freeze_policy(
        development,
        sealed_commitment=sealed_partition.commitment,
        split_manifest_digest=sealed_partition.manifest_digest,
        serial_m1_hashes=serial_m1_hashes,
        paired_development_comparisons=paired_development_comparisons,
        supplemental_development_costs=supplemental_development_costs,
        metadata=metadata,
        max_leaves=max_leaves,
        min_shape_groups_per_leaf=min_shape_groups_per_leaf,
        require_promotable=require_promotable,
    )
    sealed = sealed_partition.open(frozen.generic_digest)
    return certify_frozen_policy(
        frozen,
        development,
        sealed,
        serial_m1_hashes=serial_m1_hashes,
        require_promotable=require_promotable,
    )
