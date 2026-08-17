"""Plan concrete paired timings from p95-failing cross-validation domains.

Broad all-candidate timing is deliberately provisional. It identifies a
bounded generic policy and the domains whose held-out p95 regret is not below
five percent. Within those domains, cells over the same budget identify the
candidate edges that can improve the fit. This module closes the loop without
hand-transcribing candidate IDs: CUDA formulas resolve to their concrete
exact-KB launch, CPU schedules retain their build/runtime/thread class, and the
accumulated paired tournament emits only measurements that can still change a
decision.

The emitted JSON is consumed directly by the backend performance trainer. It
is a development-only measurement plan, never a production dispatch artifact.
An inspected CPU seal may re-enter only as generic development costs for the
next policy generation; its points are already exhaustive paired contests and
are therefore never requested again by this planner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_observation import read_observation_rows
from .candidate_registry import cuda_native_vnni_gemv_registry
from .corpus import GenericDomain, ObservationCorpus
from .cuda_shape_resolved import (
    CUDA_SHAPE_RESOLVED_PROJECTION_VERSION,
    project_cuda_shape_resolved_candidates,
    resolve_cuda_concrete_candidate_id,
)
from .exact_oracle import candidate_is_eligible
from .paired_confirmation import (
    PairedCellEvidence,
    PairedCellKey,
    PairedTimingComparison,
    paired_timing_comparisons,
    promotion_eligible_comparisons,
    read_paired_confirmation_csv,
)
from .profiler_model import load_profiler_feature_catalog
from .schema import (
    Backend,
    FEATURE_SCHEMA_VERSION,
    LEARNER_VERSION,
    P95_REGRET_BUDGET,
    SemanticContract,
)
from .segmented_policy import (
    CrossValidationCell,
    GenericPolicy,
    PolicyFitCache,
    _paired_effective_latencies,
    _runtime_paired_comparisons,
    domain_promotion_quota_is_satisfied,
    fit_generic_policy,
    generic_domain_corpus_digest,
    load_cached_cross_validations,
)
from .segmented_policy import DEFAULT_TREE_LEAVES, MAX_TREE_LEAVES
from .shape_manifest import (
    MANIFEST_PATH,
    ShapePartition,
    load_shape_manifest,
    partition_assignments,
)


PAIRED_REQUEST_SCHEMA_VERSION = "native-vnni-paired-request-v2"
PROJECTED_CORPUS_IDENTITY_SCHEMA_VERSION = (
    "native-vnni-paired-projected-corpus-identity-v1"
)
DEFAULT_MAX_REGRET = P95_REGRET_BUDGET
DECODE_M1_SURFACE = "decode-m1"
GROUPED_VERIFIER_SURFACE = "grouped-verifier"


@dataclass(frozen=True, order=True)
class PairedTimingRequest:
    """One concrete selected/reference edge for one backend trainer."""

    request_id: str
    source_format: str
    source_codebook: int
    execution_codebook: int
    architecture_class: str
    shape: str
    shape_group_id: str
    execution_mode: str
    m: int
    n: int
    k: int
    selected_candidate_id: str
    exact_candidate_id: str
    observed_cv_regret: float
    reason: str


@dataclass(frozen=True, order=True)
class PairedPlanIssue:
    """A measured over-budget or internally inconsistent tournament edge."""

    source_format: str
    source_codebook: int
    execution_codebook: int
    architecture_class: str
    shape: str
    shape_group_id: str
    execution_mode: str
    m: int
    n: int
    k: int
    selected_candidate_id: str
    exact_candidate_id: str
    observed_cv_regret: float
    direct_paired_regret: float | None
    reason: str


@dataclass(frozen=True, order=True)
class PairedDomainSummary:
    """Compact CV diagnostics for one mode/codebook/aspect policy domain."""

    backend: str
    architecture_class: str
    semantic_contract: str
    operation_kind: str
    bundle_signature: str
    prepared_family_id: str
    packing_abi: str
    runtime_codebook: int
    execution_mode: str
    m: int
    aspect_bucket: str
    feature_policy: str
    profiler_influence: str
    selected_max_leaves: int
    boundary_placement: str
    max_regret: float
    p95_regret: float
    mean_regret: float
    failed_cell_count: int
    cell_count: int


@dataclass(frozen=True, order=True)
class PairedPromotionDiagnostic:
    """Structured publication failure retained in the planner report."""

    backend: str
    architecture_class: str
    bundle_signature: str
    runtime_codebook: int
    execution_mode: str
    m: int
    blocking: bool
    rejection_stage: str
    selected_feature_policy: str
    selected_profiler_influence: str
    selected_max_leaves: int | None
    cv_required_point_count: int
    cv_covered_point_count: int
    cv_p95_regret: float | None
    cv_max_regret: float | None
    final_cross_fitted_p95_regret: float | None
    final_rule_count: int
    final_worst_p95_regret: float | None
    final_worst_max_regret: float | None
    final_worst_shape_group_id: str
    final_worst_aggregate_n: int | None
    final_worst_k: int | None
    final_worst_candidate_id: str


@dataclass(frozen=True)
class PairedRequestPlan:
    """Complete development-CV disposition for one planner iteration."""

    backend: str
    development_corpus_digest: str
    paired_evidence_digest: str
    max_regret: float
    cross_validation_domain_count: int
    failed_cv_cell_count: int
    passing_cv_cell_count: int
    equivalent_effective_surface_count: int
    tournament_completion_request_count: int
    domains: tuple[PairedDomainSummary, ...]
    requests: tuple[PairedTimingRequest, ...]
    confirmed_cv_misses: tuple[PairedPlanIssue, ...]
    conflicts: tuple[PairedPlanIssue, ...]
    promotion_diagnostics: tuple[PairedPromotionDiagnostic, ...]
    unvalidated_domains: tuple[str, ...]

    @property
    def status(self) -> str:
        """Return the single state consumed by the unattended refresh loop."""

        # A directly measured miss describes the current provisional tree. Other
        # unresolved edges can still change its boundaries or selected leaf
        # candidates, so collect every actionable request before declaring the
        # development policy terminally over budget.
        if self.requests:
            return "pending"
        if self.conflicts:
            return "evidence_conflict"
        if self.unvalidated_domains:
            return "insufficient_cross_validation"
        # Do not spend the untouched sealed partition on a model class that has
        # already failed its development holdouts under direct paired timing.
        # A denser development corpus or a revised generic feature policy must
        # first make every development CV cell economical.
        if self.confirmed_cv_misses:
            return "confirmed_failure"
        return "green"

    def request_manifest_mapping(self) -> dict[str, object]:
        """Return the strict JSON contract consumed by the CUDA trainer."""

        return {
            "schema_version": PAIRED_REQUEST_SCHEMA_VERSION,
            "backend": self.backend,
            "development_corpus_digest": self.development_corpus_digest,
            "paired_evidence_digest": self.paired_evidence_digest,
            "max_regret": self.max_regret,
            "request_count": len(self.requests),
            "requests": [asdict(request) for request in self.requests],
        }

    def report_mapping(self) -> dict[str, object]:
        """Return a human-auditable planner report with terminal failures."""

        return {
            **self.request_manifest_mapping(),
            "learner_version": LEARNER_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "status": self.status,
            "cross_validation_domain_count": self.cross_validation_domain_count,
            "failed_cv_cell_count": self.failed_cv_cell_count,
            "passing_cv_cell_count": self.passing_cv_cell_count,
            "equivalent_effective_surface_count": (
                self.equivalent_effective_surface_count
            ),
            "tournament_completion_request_count": (
                self.tournament_completion_request_count
            ),
            "domains": [asdict(domain) for domain in self.domains],
            "confirmed_cv_misses": [
                asdict(issue) for issue in self.confirmed_cv_misses
            ],
            "conflicts": [asdict(issue) for issue in self.conflicts],
            "promotion_diagnostics": [
                asdict(diagnostic) for diagnostic in self.promotion_diagnostics
            ],
            "unvalidated_domains": list(self.unvalidated_domains),
        }


def paired_comparison_digest(
    comparisons: Mapping[PairedCellKey, tuple[PairedTimingComparison, ...]],
) -> str:
    """Hash the exact tournament graph independently of input-file ordering."""

    rows = []
    for key, edges in sorted(comparisons.items()):
        for edge in sorted(
            edges,
            key=lambda item: (
                item.selected_effective_candidate_id,
                item.exact_effective_candidate_id,
                item.selected_to_exact_median_ratio,
                item.pair_count,
                item.timing_scope,
                item.mpi_world_size,
            ),
        ):
            rows.append({
                "key": asdict(key),
                "selected_effective_candidate_id": (
                    edge.selected_effective_candidate_id
                ),
                "exact_effective_candidate_id": edge.exact_effective_candidate_id,
                "selected_to_exact_median_ratio_hex": (
                    edge.selected_to_exact_median_ratio.hex()
                ),
                "pair_count": edge.pair_count,
                "timing_scope": edge.timing_scope,
                "mpi_world_size": edge.mpi_world_size,
            })
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def combined_development_evidence_digest(
    comparisons: Mapping[PairedCellKey, tuple[PairedTimingComparison, ...]],
    supplemental_evidence_digests: Iterable[str],
) -> str:
    """Bind request identities to direct pairs and inspected seal evidence."""

    supplemental = tuple(sorted(supplemental_evidence_digests))
    if not supplemental:
        return paired_comparison_digest(comparisons)
    if any(not digest.startswith("sha256:") for digest in supplemental):
        raise ValueError("supplemental development evidence lacks a digest")
    payload = {
        "paired_comparison_digest": paired_comparison_digest(comparisons),
        "supplemental_evidence_digests": supplemental,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def input_file_digest(paths: Iterable[Path]) -> str:
    """Hash common-observation shards without reserializing projected formulas."""

    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def projected_domain_cache_key(
    development: ObservationCorpus,
    domain: GenericDomain,
) -> str:
    """Identify one lazy CUDA formula projection without materializing it.

    The directly measured rows already encode the development partition and
    exact shape inventory for this domain. Their domain-local digest, combined
    with the formula and candidate-registry versions, identifies every input to
    the deterministic shape-resolved projection while preserving cache entries
    for unrelated domains.
    """

    payload = {
        "schema_version": PROJECTED_CORPUS_IDENTITY_SCHEMA_VERSION,
        "domain_corpus_digest": generic_domain_corpus_digest(
            development, domain
        ),
        "projection_version": CUDA_SHAPE_RESOLVED_PROJECTION_VERSION,
        "candidate_registry_digest": (
            cuda_native_vnni_gemv_registry().digest()
        ),
        "learner_version": LEARNER_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "semantic_contract": SemanticContract.FAST.value,
        "m": 1,
        "partition": ShapePartition.DEVELOPMENT.value,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


_PARALLEL_PROJECTED_CACHE_CORPUS: ObservationCorpus | None = None
_PARALLEL_PROJECTED_CACHE_DOMAINS: tuple[GenericDomain, ...] = ()


def _projected_domain_cache_key_at(index: int) -> tuple[GenericDomain, str]:
    """Compute one projected CUDA domain identity in a fork worker."""

    if _PARALLEL_PROJECTED_CACHE_CORPUS is None:
        raise RuntimeError("parallel projected CUDA corpus is unavailable")
    domain = _PARALLEL_PROJECTED_CACHE_DOMAINS[index]
    return domain, projected_domain_cache_key(
        _PARALLEL_PROJECTED_CACHE_CORPUS,
        domain,
    )


def projected_domain_cache_keys(
    development: ObservationCorpus,
) -> dict[GenericDomain, str]:
    """Precompute CUDA projected-corpus keys across physical CPU cores.

    CUDA lazily projects shape-resolved formula candidates per generic domain.
    Supplying a domain provider disables the common fitter's aggregate cache-key
    prepass, so computing these identities inside its serial domain loop used to
    JSON-encode the entire development corpus on one core. Fork workers inherit
    immutable observations and return keys in canonical domain order.
    """

    domains = development.generic_domains()
    worker_count = min(_physical_core_count(), len(domains))
    global _PARALLEL_PROJECTED_CACHE_CORPUS
    global _PARALLEL_PROJECTED_CACHE_DOMAINS
    _PARALLEL_PROJECTED_CACHE_CORPUS = development
    _PARALLEL_PROJECTED_CACHE_DOMAINS = domains
    try:
        if worker_count > 1 and len(domains) >= 4:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                pairs = tuple(executor.map(
                    _projected_domain_cache_key_at,
                    range(len(domains)),
                ))
        else:
            pairs = tuple(
                _projected_domain_cache_key_at(index)
                for index in range(len(domains))
            )
    finally:
        _PARALLEL_PROJECTED_CACHE_CORPUS = None
        _PARALLEL_PROJECTED_CACHE_DOMAINS = ()
    return dict(pairs)


def projected_cuda_domain_corpora(
    development: ObservationCorpus,
) -> dict[GenericDomain, ObservationCorpus]:
    """Project each CUDA policy domain without building a giant aggregate.

    Profiler-informed fitting consumes every generic domain to construct its
    cross-format observation index. The earlier batched implementation first
    materialized and fully indexed a multi-million-row aggregate corpus, then
    indexed every row again while partitioning it. Formula resolution is tiny
    and memoized by geometry; constructing each domain directly preserves the
    same rows and source order while paying for exactly one corpus index.
    """

    domains = development.generic_domains()
    projected = {
        domain: project_cuda_shape_resolved_candidates(
            development.rows_for_generic_domain(domain),
            known_generic_domain=domain,
        )
        for domain in domains
    }
    changed = tuple(
        domain
        for domain, corpus in projected.items()
        if corpus.generic_domains() != (domain,)
    )
    if changed:
        raise ValueError(
            "domain-local CUDA formula projection changed the generic domain "
            f"inventory for {changed!r}"
        )
    return projected


def _request_id(
    fields: Mapping[str, object],
    paired_evidence_digest: str,
) -> str:
    """Bind a request identity to both its edge and tournament generation."""

    encoded = json.dumps(
        {
            "fields": dict(fields),
            "paired_evidence_digest": paired_evidence_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "nvnni-pair-" + hashlib.sha256(encoded).hexdigest()[:24]


def _effective_candidate(
    rows: Iterable,
    candidate_id: str,
) -> str:
    """Resolve one nominal policy candidate from directly backed observations.

    The planner deliberately keeps the compact directly measured corpus in
    memory. A parameterized KPAR formula is resolved from the reviewed registry
    at this exact N/K and then authenticated against its concrete measured row;
    materializing every formula alias across the complete corpus is unnecessary.
    """

    rows = tuple(rows)
    effective = {
        row.effective_candidate_id
        for row in rows
        if row.candidate_id == candidate_id
        and row.generic_eligible
        and candidate_is_eligible(row, None)
    }
    if not effective:
        if not rows:
            raise ValueError("cannot resolve a candidate from an empty surface")
        if rows[0].backend != Backend.CUDA:
            raise ValueError(
                f"{candidate_id}: backend candidate has no forceable observation"
            )
        dimensions = {(row.aggregate_n, row.k) for row in rows}
        if len(dimensions) != 1:
            raise ValueError("candidate surface changed N/K dimensions")
        n, k = next(iter(dimensions))
        candidate = cuda_native_vnni_gemv_registry().resolve(candidate_id)
        concrete_id = resolve_cuda_concrete_candidate_id(candidate, n, k)
        effective = {
            row.effective_candidate_id
            for row in rows
            if row.effective_candidate_id == concrete_id
            and candidate_is_eligible(row, None)
        }
    if len(effective) != 1:
        raise ValueError(
            f"{candidate_id}: expected one forceable effective candidate, "
            f"got {sorted(effective)}"
        )
    return next(iter(effective))


def _direct_paired_regret(
    edges: Iterable[PairedTimingComparison],
    selected_candidate_id: str,
    exact_candidate_id: str,
) -> float | None:
    """Return the weighted direct-edge log ratio in selected/exact orientation."""

    weighted_log_sum = 0.0
    total_pairs = 0
    for edge in edges:
        selected = edge.selected_effective_candidate_id
        exact = edge.exact_effective_candidate_id
        if selected == selected_candidate_id and exact == exact_candidate_id:
            ratio = edge.selected_to_exact_median_ratio
        elif selected == exact_candidate_id and exact == selected_candidate_id:
            ratio = 1.0 / edge.selected_to_exact_median_ratio
        else:
            continue
        weighted_log_sum += edge.pair_count * math.log(ratio)
        total_pairs += edge.pair_count
    if total_pairs == 0:
        return None
    return math.exp(weighted_log_sum / total_pairs) - 1.0


def _directed_paired_ratio(
    edges: Iterable[PairedTimingComparison],
    selected_candidate_id: str,
    exact_candidate_id: str,
) -> float | None:
    """Return one role-preserving direct ratio from a single star direction."""

    weighted_log_sum = 0.0
    total_pairs = 0
    for edge in edges:
        if (
            edge.selected_effective_candidate_id != selected_candidate_id
            or edge.exact_effective_candidate_id != exact_candidate_id
        ):
            continue
        weighted_log_sum += edge.pair_count * math.log(
            edge.selected_to_exact_median_ratio
        )
        total_pairs += edge.pair_count
    if total_pairs == 0:
        return None
    return math.exp(weighted_log_sum / total_pairs)


def _complete_directional_star_latencies(
    edges: Iterable[PairedTimingComparison],
    candidates: Iterable[str],
) -> tuple[str, dict[str, float]] | None:
    """Recover relative latencies from one complete role-preserving star.

    Every edge emitted for a cell in one generation names the same selected
    anchor. Requiring that direction distinguishes a coherent atomic star from
    an undirected graph pieced together over unrelated process histories. The
    anchor latency is normalized to one; an ``anchor / candidate`` measurement
    therefore places the candidate at the reciprocal ratio.
    """

    edges = tuple(edges)
    candidates = tuple(sorted(set(candidates)))
    complete = []
    for anchor in candidates:
        relative_latencies = {anchor: 1.0}
        for candidate in candidates:
            if candidate == anchor:
                continue
            ratio = _directed_paired_ratio(edges, anchor, candidate)
            if ratio is None:
                break
            relative_latencies[candidate] = 1.0 / ratio
        else:
            complete.append((anchor, relative_latencies))
    if len(complete) > 1:
        raise ValueError(
            "paired runtime cell contains multiple complete directional stars"
        )
    return complete[0] if complete else None


def _surface_rows(
    development: ObservationCorpus,
    cell: CrossValidationCell,
) -> dict[str, tuple]:
    """Return each source alias represented by one held-out runtime point."""

    rows = tuple(
        row
        for row in development.rows_for_runtime_key(cell.runtime_key)
        if row.shape_group_id == cell.shape_group_id
    )
    grouped: dict[str, list] = {}
    for row in rows:
        grouped.setdefault(row.source_format, []).append(row)
    if not grouped:
        raise ValueError(
            f"cross-validation cell has no development observations: {cell}"
        )
    return {name: tuple(values) for name, values in grouped.items()}


def _paired_key(rows: tuple, cell: CrossValidationCell) -> PairedCellKey:
    """Build the exact confirmation key and reject alias metadata drift."""

    shape_names = {row.shape_name for row in rows}
    source_codebooks = {row.source_codebook_id for row in rows}
    runtime_codebooks = {row.runtime_codebook_id for row in rows}
    if len(shape_names) != 1 or len(source_codebooks) != 1 or len(runtime_codebooks) != 1:
        raise ValueError("one source alias changed shape or codebook identity")
    key = cell.runtime_key
    return PairedCellKey(
        backend=key.backend.value,
        source_format=rows[0].source_format,
        source_codebook=next(iter(source_codebooks)),
        execution_codebook=next(iter(runtime_codebooks)),
        shape=next(iter(shape_names)),
        execution_mode=key.execution_mode.value,
        m=key.m,
        n=key.aggregate_n,
        k=key.k,
        architecture_class=(
            key.architecture_class if key.backend == Backend.CPU else ""
        ),
    )


def _request_architecture_class(rows: tuple) -> str:
    """Return the producer architecture even when legacy CUDA pairs omit it."""

    architectures = {row.architecture_class for row in rows}
    if len(architectures) != 1 or not next(iter(architectures)):
        raise ValueError("one source alias changed or omitted architecture identity")
    return next(iter(architectures))


def _source_alias_regret_witness(
    aliases: Iterable[tuple[str, tuple]],
    selected_effective_candidate_id: str,
    comparisons: Iterable[PairedTimingComparison],
) -> str | None:
    """Return the physical candidate that witnesses selected's worst regret.

    ``CrossValidationCell.exact_candidate_id`` is the best minimax compromise
    across source aliases. It is not necessarily the candidate that establishes
    the selected candidate's maximum per-alias regret. Separate aliases can
    name different winners even though they normalize to one production
    ``RuntimeKey``. Reconstruct the same paired-corrected effective timings used
    by cost construction, then select the candidate with the largest advantage
    over the held-out choice. That concrete edge is the one paired timing can
    actually confirm or repair.
    """

    competitors: list[tuple[float, str]] = []
    for _, rows in aliases:
        latencies: dict[str, list[float]] = {}
        for row in rows:
            if not row.generic_eligible or not candidate_is_eligible(row, None):
                continue
            latencies.setdefault(row.effective_candidate_id, []).append(
                row.median_us
            )
        medians = {
            candidate: statistics.median(values)
            for candidate, values in latencies.items()
        }
        if comparisons:
            medians.update(_paired_effective_latencies(medians, comparisons))
        selected_latency = medians.get(selected_effective_candidate_id)
        if selected_latency is None:
            raise ValueError(
                "selected effective candidate is absent from one source alias"
            )
        for candidate, latency in medians.items():
            if candidate == selected_effective_candidate_id:
                continue
            advantage = selected_latency / latency - 1.0
            if advantage > 0.0:
                competitors.append((advantage, candidate))
    if not competitors:
        return None
    return max(competitors, key=lambda item: (item[0], item[1]))[1]


def _source_alias_effective_candidates(
    aliases: Iterable[tuple[str, tuple]],
) -> tuple[str, ...]:
    """Return forceable physical launches shared by every source alias.

    A production runtime key has no source-format discriminator.  Its paired
    tournament must therefore compare only concrete launches represented by
    every source alias that contributed to the held-out point.  Returning the
    intersection also collapses nominal formula aliases that resolve to the
    same physical launch.
    """

    candidate_sets = []
    for _, rows in aliases:
        candidates = {
            row.effective_candidate_id
            for row in rows
            if row.generic_eligible and candidate_is_eligible(row, None)
        }
        if not candidates:
            raise ValueError("source alias has no forceable paired candidates")
        candidate_sets.append(candidates)
    shared = set.intersection(*candidate_sets)
    if not shared:
        raise ValueError("source aliases share no forceable paired candidates")
    return tuple(sorted(shared))


def build_paired_request_plan(
    development: ObservationCorpus,
    policy: GenericPolicy,
    comparisons: Mapping[PairedCellKey, tuple[PairedTimingComparison, ...]],
    *,
    max_regret: float = DEFAULT_MAX_REGRET,
    development_corpus_digest: str | None = None,
    supplemental_shape_group_ids: frozenset[str] = frozenset(),
    supplemental_evidence_digests: tuple[str, ...] = (),
) -> PairedRequestPlan:
    """Classify p95-failing CV domains and emit actionable pair timings.

    Maximum cell regret remains visible in the report, but an isolated tail
    cell cannot open a paired-refinement transaction after its complete domain
    already satisfies the installation p95 threshold.
    """

    if not 0.0 < max_regret <= 1.0:
        raise ValueError("max_regret must be in (0, 1]")
    backends = {row.backend for row in development}
    if len(backends) != 1 or next(iter(backends)) not in {
        Backend.CPU,
        Backend.CUDA,
        Backend.ROCM,
    }:
        raise ValueError("paired planning requires one supported backend")
    backend = next(iter(backends))

    evidence_digest = combined_development_evidence_digest(
        comparisons, supplemental_evidence_digests
    )
    runtime_comparisons = _runtime_paired_comparisons(
        development,
        comparisons,
    )
    requests: list[PairedTimingRequest] = []
    failures: list[PairedPlanIssue] = []
    conflicts: list[PairedPlanIssue] = []
    equivalent_surfaces = 0
    domains = tuple(sorted(
        PairedDomainSummary(
            backend=validation.domain.backend.value,
            architecture_class=validation.domain.architecture_class,
            semantic_contract=validation.domain.semantic_contract.value,
            operation_kind=validation.domain.operation_kind,
            bundle_signature=validation.domain.bundle_signature,
            prepared_family_id=validation.domain.prepared_family_id,
            packing_abi=validation.domain.packing_abi,
            runtime_codebook=validation.domain.runtime_codebook_id,
            execution_mode=validation.domain.execution_mode.value,
            m=validation.domain.m,
            aspect_bucket=validation.domain.aspect_bucket.value,
            feature_policy=validation.selected_feature_policy.value,
            profiler_influence=(
                validation.selected_profiler_influence.value
            ),
            selected_max_leaves=validation.selected_max_leaves,
            boundary_placement=validation.selected_boundary_placement.value,
            max_regret=validation.max_regret,
            p95_regret=validation.p95_regret,
            mean_regret=validation.mean_regret,
            failed_cell_count=sum(
                cell.observed_broad_regret >= max_regret
                for cell in validation.cells
            ),
            cell_count=len(validation.cells),
        )
        for validation in policy.cross_validation
    ))

    validated_domains = {result.domain for result in policy.cross_validation}
    final_publication_failures = {
        diagnostic.domain
        for diagnostic in policy.promotion_diagnostics
        if diagnostic.rejection_stage.startswith("final_fit")
    }
    validation_by_domain = {
        validation.domain: validation
        for validation in policy.cross_validation
    }
    blocking_domains = set(policy.unpromoted_domains)
    promotion_diagnostics = tuple(sorted(
        PairedPromotionDiagnostic(
            backend=diagnostic.domain.backend.value,
            architecture_class=diagnostic.domain.architecture_class,
            bundle_signature=diagnostic.domain.bundle_signature,
            runtime_codebook=diagnostic.domain.runtime_codebook_id,
            execution_mode=diagnostic.domain.execution_mode.value,
            m=diagnostic.domain.m,
            blocking=diagnostic.domain in blocking_domains,
            rejection_stage=diagnostic.rejection_stage,
            selected_feature_policy=(
                validation_by_domain[diagnostic.domain].selected_feature_policy.value
                if diagnostic.domain in validation_by_domain
                else ""
            ),
            selected_profiler_influence=(
                validation_by_domain[
                    diagnostic.domain
                ].selected_profiler_influence.value
                if diagnostic.domain in validation_by_domain
                else ""
            ),
            selected_max_leaves=(
                validation_by_domain[diagnostic.domain].selected_max_leaves
                if diagnostic.domain in validation_by_domain
                else None
            ),
            cv_required_point_count=diagnostic.cv_required_point_count,
            cv_covered_point_count=diagnostic.cv_covered_point_count,
            cv_p95_regret=diagnostic.cv_p95_regret,
            cv_max_regret=diagnostic.cv_max_regret,
            final_cross_fitted_p95_regret=(
                diagnostic.final_cross_fitted_p95_regret
            ),
            final_rule_count=diagnostic.final_rule_count,
            final_worst_p95_regret=diagnostic.final_worst_p95_regret,
            final_worst_max_regret=diagnostic.final_worst_max_regret,
            final_worst_shape_group_id=(
                diagnostic.final_worst_shape_group_id
            ),
            final_worst_aggregate_n=diagnostic.final_worst_aggregate_n,
            final_worst_k=diagnostic.final_worst_k,
            final_worst_candidate_id=diagnostic.final_worst_candidate_id,
        )
        for diagnostic in policy.promotion_diagnostics
    ))
    # Ordinary CV failures remain repairable by the paired tournament below.
    # A final publication failure is different: CV exists, but no installable
    # total tree was produced. Surface that state before sealed data is opened.
    unvalidated = tuple(
        repr(domain)
        for domain in policy.unpromoted_domains
        if domain not in validated_domains
        or domain in final_publication_failures
    )
    passing_domain_count = sum(
        validation.p95_regret < max_regret
        for validation in policy.cross_validation
    )
    quota_allows_performance_exceptions = (
        not unvalidated
        and domain_promotion_quota_is_satisfied(
            passing_domain_count,
            len(policy.cross_validation),
        )
    )
    if not quota_allows_performance_exceptions:
        supplemental_blockers = tuple(
            "supplemental-development:" + repr(validation.domain)
            for validation in policy.cross_validation
            if validation.p95_regret >= max_regret
            and any(
                cell.observed_broad_regret >= max_regret
                and cell.shape_group_id in supplemental_shape_group_ids
                for cell in validation.cells
            )
        )
        unvalidated = tuple(sorted((*unvalidated, *supplemental_blockers)))
    failed_cells = sum(
        sum(
            cell.observed_broad_regret >= max_regret
            for cell in validation.cells
        )
        for validation in policy.cross_validation
        if validation.p95_regret >= max_regret
    )
    passing_cells = sum(
        len(validation.cells)
        if validation.p95_regret < max_regret
        else sum(
            cell.observed_broad_regret < max_regret
            for cell in validation.cells
        )
        for validation in policy.cross_validation
    )

    tournament_completion_request_count = 0
    requested_edges = set()
    for validation in policy.cross_validation:
        if (
            validation.p95_regret < max_regret
            or quota_allows_performance_exceptions
        ):
            continue
        for cell in validation.cells:
            if (
                cell.observed_broad_regret < max_regret
                or cell.shape_group_id in supplemental_shape_group_ids
            ):
                continue
            aliases = sorted(_surface_rows(development, cell).items())
            resolved = tuple(
                (
                    source_format,
                    rows,
                    _paired_key(rows, cell),
                    _effective_candidate(rows, cell.selected_candidate_id),
                )
                for source_format, rows in aliases
            )
            selected_launches = {selected for _, _, _, selected in resolved}
            if len(selected_launches) != 1:
                raise ValueError(
                    "source aliases of one runtime surface resolve different "
                    f"selected launches: {sorted(selected_launches)!r}"
                )
            source_format, rows, key, selected = resolved[0]
            shared_candidates = _source_alias_effective_candidates(aliases)
            if selected not in shared_candidates:
                raise ValueError(
                    "selected launch is not forceable across every source alias"
                )
            runtime_edges = runtime_comparisons.get(cell.runtime_key, ())
            # Historical CPU v3 rows did not identify their co-running socket
            # workload. They may seed a development fit, but only isolated v4
            # edges can close a tournament, confirm a miss, or suppress a fresh
            # request. GPU evidence remains promotion eligible unchanged.
            authoritative_runtime_edges = promotion_eligible_comparisons(
                runtime_edges
            )
            competitor = _source_alias_regret_witness(
                aliases,
                selected,
                runtime_edges,
            )
            if competitor is None:
                equivalent_surfaces += 1
                continue

            complete_star = _complete_directional_star_latencies(
                authoritative_runtime_edges,
                shared_candidates,
            )
            if complete_star is not None:
                _, relative_latencies = complete_star
                star_regret = (
                    relative_latencies[selected]
                    / relative_latencies[competitor]
                    - 1.0
                )
                issue_fields = {
                    "source_format": source_format,
                    "source_codebook": key.source_codebook,
                    "execution_codebook": key.execution_codebook,
                    "architecture_class": _request_architecture_class(rows),
                    "shape": key.shape,
                    "shape_group_id": cell.shape_group_id,
                    "execution_mode": key.execution_mode,
                    "m": key.m,
                    "n": key.n,
                    "k": key.k,
                    "selected_candidate_id": selected,
                    "exact_candidate_id": competitor,
                    "observed_cv_regret": cell.observed_broad_regret,
                    "direct_paired_regret": star_regret,
                }
                destination = (
                    failures if star_regret >= max_regret else conflicts
                )
                destination.append(PairedPlanIssue(
                    **issue_fields,
                    reason=(
                        "complete_tournament_star_confirms_over_budget"
                        if star_regret >= max_regret
                        else "tournament_fit_conflicts_with_complete_star"
                    ),
                ))
                continue

            direct_regret = _direct_paired_regret(
                authoritative_runtime_edges, selected, competitor
            )
            issue_fields = {
                "source_format": source_format,
                "source_codebook": key.source_codebook,
                "execution_codebook": key.execution_codebook,
                "architecture_class": _request_architecture_class(rows),
                "shape": key.shape,
                "shape_group_id": cell.shape_group_id,
                "execution_mode": key.execution_mode,
                "m": key.m,
                "n": key.n,
                "k": key.k,
                "selected_candidate_id": selected,
                "exact_candidate_id": competitor,
                "observed_cv_regret": cell.observed_broad_regret,
                "direct_paired_regret": direct_regret,
            }
            if direct_regret is not None:
                destination = (
                    failures if direct_regret >= max_regret else conflicts
                )
                destination.append(PairedPlanIssue(
                    **issue_fields,
                    reason=(
                        "direct_pair_confirms_over_budget"
                        if direct_regret >= max_regret
                        else "tournament_fit_conflicts_with_direct_edge"
                    ),
                ))

            for exact in shared_candidates:
                if exact == selected:
                    continue
                if _directed_paired_ratio(
                    authoritative_runtime_edges, selected, exact
                ) is not None:
                    continue
                edge = (
                    key.execution_codebook,
                    _request_architecture_class(rows),
                    key.shape,
                    cell.shape_group_id,
                    key.execution_mode,
                    key.m,
                    key.n,
                    key.k,
                    frozenset((selected, exact)),
                )
                if edge in requested_edges:
                    continue
                request_fields = {
                    "source_format": source_format,
                    "source_codebook": key.source_codebook,
                    "execution_codebook": key.execution_codebook,
                    "architecture_class": _request_architecture_class(rows),
                    "shape": key.shape,
                    "shape_group_id": cell.shape_group_id,
                    "execution_mode": key.execution_mode,
                    "m": key.m,
                    "n": key.n,
                    "k": key.k,
                    "selected_candidate_id": selected,
                    "exact_candidate_id": exact,
                    "observed_cv_regret": cell.observed_broad_regret,
                }
                requests.append(PairedTimingRequest(
                    request_id=_request_id(request_fields, evidence_digest),
                    **request_fields,
                    reason=(
                        "missing_direct_tournament_edge"
                        if exact == competitor
                        else "complete_tournament_star_edge"
                    ),
                ))
                requested_edges.add(edge)
                if exact != competitor:
                    tournament_completion_request_count += 1

    duplicate_ids = {
        request.request_id
        for request in requests
        if sum(item.request_id == request.request_id for item in requests) > 1
    }
    if duplicate_ids:
        raise ValueError(f"duplicate paired request IDs: {sorted(duplicate_ids)}")

    return PairedRequestPlan(
        backend=backend.value,
        development_corpus_digest=(
            development_corpus_digest or development.digest()
        ),
        paired_evidence_digest=evidence_digest,
        max_regret=max_regret,
        cross_validation_domain_count=len(policy.cross_validation),
        failed_cv_cell_count=failed_cells,
        passing_cv_cell_count=passing_cells,
        equivalent_effective_surface_count=equivalent_surfaces,
        tournament_completion_request_count=(
            tournament_completion_request_count
        ),
        domains=domains,
        requests=tuple(sorted(requests)),
        confirmed_cv_misses=tuple(sorted(failures)),
        conflicts=tuple(sorted(conflicts)),
        promotion_diagnostics=promotion_diagnostics,
        unvalidated_domains=unvalidated,
    )


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Atomically publish one deterministic planner artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_request_shards(
    directory: Path,
    plan: PairedRequestPlan,
    *,
    max_requests_per_shard: int,
) -> tuple[Path, ...]:
    """Publish deterministic homogeneous manifests for resumable producers.

    A CPU process has one compiled ISA and one runtime dispatch request, while a
    planner transaction can contain all three CPU regimes. Shards therefore
    never cross architecture classes. Every edge for one runtime cell remains
    in one process invocation so CPU clock, socket, cache, and thread-runtime
    state cannot turn separately gathered edges into an inconsistent graph.
    ``max_requests_per_shard`` is a packing target; one indivisible cell may
    exceed it when a backend exposes a larger candidate family.
    """

    if max_requests_per_shard < 1:
        raise ValueError("max_requests_per_shard must be positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, str], list[PairedTimingRequest]] = {}
    for request in plan.requests:
        grouped.setdefault(
            (plan.backend, request.architecture_class), []
        ).append(request)

    paths = []
    shard_records = []
    shard_index = 0
    for (backend, architecture_class), requests in sorted(grouped.items()):
        cell_groups: dict[tuple, list[PairedTimingRequest]] = {}
        for request in requests:
            cell_groups.setdefault((
                request.source_format,
                request.source_codebook,
                request.execution_codebook,
                request.shape,
                request.shape_group_id,
                request.execution_mode,
                request.m,
                request.n,
                request.k,
            ), []).append(request)
        batches: list[tuple[PairedTimingRequest, ...]] = []
        current: list[PairedTimingRequest] = []
        for _, cell_requests in sorted(cell_groups.items()):
            ordered_cell = sorted(
                cell_requests,
                key=lambda request: (
                    request.selected_candidate_id,
                    request.exact_candidate_id,
                    request.request_id,
                ),
            )
            if current and (
                len(current) + len(ordered_cell) > max_requests_per_shard
            ):
                batches.append(tuple(current))
                current = []
            current.extend(ordered_cell)
        if current:
            batches.append(tuple(current))
        architecture_digest = hashlib.sha256(
            architecture_class.encode("utf-8")
        ).hexdigest()[:12]
        for selected in batches:
            payload = plan.request_manifest_mapping()
            payload["request_count"] = len(selected)
            payload["requests"] = [asdict(request) for request in selected]
            request_digest = hashlib.sha256(json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")).hexdigest()[:12]
            name = (
                f"shard-{shard_index:04d}.{backend}."
                f"{architecture_digest}.{request_digest}.requests.json"
            )
            path = directory / name
            write_json(path, payload)
            paths.append(path)
            shard_records.append({
                "path": name,
                "backend": backend,
                "architecture_class": architecture_class,
                "request_digest": request_digest,
                "request_count": len(selected),
            })
            shard_index += 1

    expected_names = {path.name for path in paths}
    for stale in directory.glob("shard-*.requests.json"):
        if stale.name not in expected_names:
            stale.unlink()
    write_json(directory / "index.json", {
        "schema_version": "native-vnni-paired-request-shards-v2-content-addressed",
        "backend": plan.backend,
        "development_corpus_digest": plan.development_corpus_digest,
        "paired_evidence_digest": plan.paired_evidence_digest,
        "max_requests_per_shard": max_requests_per_shard,
        "request_count": len(plan.requests),
        "shard_count": len(paths),
        "shards": shard_records,
    })
    return tuple(paths)


def _read_paired_shards(paths: Iterable[Path]) -> tuple[PairedCellEvidence, ...]:
    """Read independent evidence shards across affinity-visible physical cores.

    A refinement generation commonly retains hundreds of immutable CSV files.
    Each file has a self-contained protocol/header/cell validation transaction,
    so process workers can authenticate them independently. ``executor.map``
    preserves canonical path order; the parent still constructs the tournament
    graph and therefore remains the sole owner of cross-shard duplicate checks.
    """

    paths = tuple(sorted(Path(path) for path in paths))
    if not paths:
        return ()
    workers = min(_physical_core_count(), len(paths))
    if workers <= 1 or len(paths) < 4:
        shards = tuple(_read_one_paired_shard(path) for path in paths)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            shards = tuple(executor.map(_read_one_paired_shard, paths))
    return tuple(cell for shard in shards for cell in shard)


def _read_one_paired_shard(path: Path) -> tuple[PairedCellEvidence, ...]:
    """Authenticate one immutable paired CSV in a process worker."""

    return read_paired_confirmation_csv((path,))


def _physical_core_count() -> int:
    """Return physical cores visible through the current affinity mask."""

    logical_cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    physical_cores = set()
    for cpu in logical_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package_id = (topology / "physical_package_id").read_text().strip()
            core_id = (topology / "core_id").read_text().strip()
        except OSError:
            return max(1, len(logical_cpus))
        physical_cores.add((package_id, core_id))
    return max(1, len(physical_cores))


def _development_corpus(
    observations: Iterable,
    manifest_path: Path,
    *,
    surface: str = DECODE_M1_SURFACE,
) -> ObservationCorpus:
    """Retain one compact direct CPU/CUDA development surface.

    Shape-resolved formula aliases are projected lazily for a cost/CV cache-miss
    domain. Keeping them out of this owning corpus avoids expanding a modest
    source dataset into a gigabyte-scale repeated-metadata intermediate.
    """

    rows = tuple(observations)
    manifest = load_shape_manifest(manifest_path)
    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in rows),
        verifier=surface == GROUPED_VERIFIER_SURFACE,
        manifest=manifest,
    )
    if surface == DECODE_M1_SURFACE:
        matches_surface = lambda row: (
            row.semantic_contract == SemanticContract.FAST and row.m == 1
        )
    elif surface == GROUPED_VERIFIER_SURFACE:
        matches_surface = lambda row: (
            row.backend == Backend.CPU
            and row.semantic_contract
            == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
            and row.m > 1
        )
    else:
        raise ValueError(f"unknown paired development surface {surface!r}")
    direct_development = tuple(
        row
        for row in rows
        if matches_surface(row)
        and assignments[row.shape_group_id] == ShapePartition.DEVELOPMENT
    )
    # CPU generic rules deliberately learn aspect ratio inside one tree. Build
    # that collapsed index directly: constructing the default aspect-partitioned
    # corpus and immediately rebuilding it touched all trainer rows twice.
    collapse_aspect_domains = {
        row.backend for row in direct_development
    } == {Backend.CPU}
    return ObservationCorpus._from_validated(
        direct_development,
        distinguish_aspect_bucket=not collapse_aspect_domains,
    )


def _policy_from_cached_validations(
    development: ObservationCorpus,
    validations: Iterable,
) -> GenericPolicy:
    """Authenticate a complete cached CV generation against current rows.

    This is a bootstrap path only. The cache supplies the provisional held-out
    decisions from an already-completed fit; the current corpus still supplies
    every source alias and forceable effective candidate used to form requests.
    A later iteration with paired corrections must run the normal incremental
    fitter so changed candidate costs can select a different tree.
    """

    validations = tuple(validations)
    expected_domains = frozenset(development.generic_domains())
    observed_domains = frozenset(validation.domain for validation in validations)
    if len(observed_domains) != len(validations):
        raise ValueError("cached cross-validation repeats a policy domain")
    if observed_domains != expected_domains:
        missing = sorted(expected_domains - observed_domains)
        foreign = sorted(observed_domains - expected_domains)
        raise ValueError(
            "cached cross-validation does not match the development corpus: "
            f"missing={missing!r}, foreign={foreign!r}"
        )

    for validation in validations:
        cells = (*validation.cells, *validation.competitive_cells)
        for cell in cells:
            rows = tuple(
                row
                for row in development.rows_for_runtime_key(cell.runtime_key)
                if row.shape_group_id == cell.shape_group_id
            )
            if not rows:
                raise ValueError(
                    "cached cross-validation cell is absent from the current "
                    f"development corpus: {cell!r}"
                )
    return GenericPolicy(
        rules=(),
        unpromoted_domains=tuple(sorted(expected_domains)),
        cross_validation=tuple(sorted(validations, key=lambda item: item.domain)),
    )


def _load_cpu_burned_development(
    development: ObservationCorpus,
    surface: str,
    plan_paths: tuple[Path, ...],
    evidence_directories: tuple[Path, ...],
):
    """Load one CPU surface's inspected seals without creating import cycles."""

    if not plan_paths and not evidence_directories:
        return {}, (), {}
    # The CPU seal modules import request schemas from this module. Importing
    # them only after module initialization keeps that dependency acyclic while
    # still sharing the exact plan readers and cost adapters used by freeze.
    from .cpu_sealed_paired import load_cpu_burned_seal_development

    if surface == DECODE_M1_SURFACE:
        from .cpu_decode_sealed_plan import (
            cpu_decode_burned_seal_costs,
            read_cpu_decode_sealed_plan,
        )

        return load_cpu_burned_seal_development(
            development,
            plan_paths,
            evidence_directories,
            surface_name="decode",
            read_plan=read_cpu_decode_sealed_plan,
            build_costs=cpu_decode_burned_seal_costs,
        )
    if surface == GROUPED_VERIFIER_SURFACE:
        from .cpu_grouped_decode_sealed_plan import (
            cpu_grouped_burned_seal_costs,
            read_cpu_grouped_sealed_plan,
        )

        return load_cpu_burned_seal_development(
            development,
            plan_paths,
            evidence_directories,
            surface_name="grouped decode",
            read_plan=read_cpu_grouped_sealed_plan,
            build_costs=cpu_grouped_burned_seal_costs,
        )
    raise ValueError(f"burned CPU evidence cannot target {surface!r}")


def main() -> int:
    """Fit development CV, emit the next paired batch, and report disposition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--surface",
        choices=(DECODE_M1_SURFACE, GROUPED_VERIFIER_SURFACE),
        default=DECODE_M1_SURFACE,
        help="Development semantic/M surface to fit and refine",
    )
    parser.add_argument(
        "--paired-csv",
        action="append",
        type=Path,
        default=[],
        help="Validated paired evidence shard; repeat for every retained file",
    )
    parser.add_argument(
        "--burned-sealed-plan-json",
        action="append",
        type=Path,
        default=[],
        help="Inspected CPU seal plan reused as generic development evidence",
    )
    parser.add_argument(
        "--burned-sealed-paired-dir",
        action="append",
        type=Path,
        default=[],
        help="Paired CSV directory matched positionally to a burned CPU plan",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--request-shard-dir",
        type=Path,
        help="Publish architecture-homogeneous bounded producer manifests",
    )
    parser.add_argument(
        "--max-requests-per-shard",
        type=int,
        default=16,
        help="Maximum requests in one resumable producer process",
    )
    parser.add_argument("--shape-manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--development-profiler-requests",
        type=Path,
        help=(
            "Authenticated isolated profiler requests for a new/incremental "
            "fit; omitted only for cached-CV bootstrap"
        ),
    )
    parser.add_argument(
        "--development-profiler-evidence",
        type=Path,
        help=(
            "Complete per-candidate development profiler evidence; omitted "
            "only for cached-CV bootstrap"
        ),
    )
    parser.add_argument(
        "--development-profiler-observations",
        type=Path,
        help="Original common CSV bound to reusable profiler sidecars",
    )
    parser.add_argument(
        "--fit-cache-dir",
        type=Path,
        help=(
            "Content-addressed domain-cost and domain-CV cache used across "
            "paired refinement iterations"
        ),
    )
    parser.add_argument(
        "--cached-cross-validation-dir",
        type=Path,
        help=(
            "Read one already-completed immutable domain-CV generation to "
            "bootstrap the first paired batch without rerunning tree search"
        ),
    )
    parser.add_argument("--max-regret", type=float, default=DEFAULT_MAX_REGRET)
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=DEFAULT_TREE_LEAVES,
        help=(
            "Cross-validated per-domain leaf budget; production defaults to "
            f"{DEFAULT_TREE_LEAVES} and the feature schema permits at most "
            f"{MAX_TREE_LEAVES}"
        ),
    )
    parser.add_argument("--min-shape-groups-per-leaf", type=int, default=2)
    args = parser.parse_args()
    if args.max_leaves < 1:
        parser.error("--max-leaves must be positive")
    if args.min_shape_groups_per_leaf < 1:
        parser.error("--min-shape-groups-per-leaf must be positive")
    if args.max_requests_per_shard < 1:
        parser.error("--max-requests-per-shard must be positive")
    if len(args.burned_sealed_plan_json) != len(
        args.burned_sealed_paired_dir
    ):
        parser.error(
            "burned sealed plans and paired directories must pair by position"
        )
    if args.request_shard_dir is None and (
        args.max_requests_per_shard != 16
    ):
        parser.error(
            "--max-requests-per-shard requires --request-shard-dir"
        )
    profiler_pair = (
        args.development_profiler_requests is not None,
        args.development_profiler_evidence is not None,
    )
    if profiler_pair[0] != profiler_pair[1]:
        parser.error(
            "development profiler requests and evidence are required together"
        )
    if args.development_profiler_observations and not all(profiler_pair):
        parser.error("profiler observations require requests and evidence")
    if args.cached_cross_validation_dir is not None:
        if args.paired_csv:
            parser.error(
                "cached-CV bootstrap accepts no paired evidence; refit changed "
                "domains through the normal fit-cache path"
            )
        if args.burned_sealed_plan_json:
            parser.error(
                "cached-CV bootstrap cannot add burned evidence; use the "
                "normal content-addressed fit"
            )
    elif not all(profiler_pair):
        parser.error(
            "a new or incremental fit requires development profiler evidence"
        )

    planner_started = time.perf_counter()
    input_paths = tuple(sorted(args.inputs))
    source_digest = input_file_digest(input_paths)
    digest_complete = time.perf_counter()
    observations = read_observation_rows(input_paths)
    read_complete = time.perf_counter()
    development = _development_corpus(
        observations,
        args.shape_manifest,
        surface=args.surface,
    )
    del observations
    backends = {row.backend for row in development}
    if len(backends) != 1:
        parser.error("paired planning requires one homogeneous backend corpus")
    backend = next(iter(backends))
    if args.burned_sealed_plan_json and backend != Backend.CPU:
        parser.error("burned sealed development is currently a CPU transaction")
    development_complete = time.perf_counter()
    supplemental_costs, supplemental_digests, _supplemental_dimensions = (
        _load_cpu_burned_development(
            development,
            args.surface,
            tuple(args.burned_sealed_plan_json),
            tuple(args.burned_sealed_paired_dir),
        )
    )
    profiler_catalog = (
        load_profiler_feature_catalog(
            development,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus_path=args.development_profiler_observations,
            cache_path=(
                args.fit_cache_dir / "profiler_feature_catalog_v1.json"
                if args.fit_cache_dir is not None
                else None
            ),
        )
        if all(profiler_pair)
        else None
    )
    paired_cells = _read_paired_shards(args.paired_csv)
    comparisons = (
        paired_timing_comparisons(paired_cells) if paired_cells else {}
    )
    paired_complete = time.perf_counter()
    if args.cached_cross_validation_dir is not None:
        policy = _policy_from_cached_validations(
            development,
            load_cached_cross_validations(args.cached_cross_validation_dir),
        )
    else:
        fit_arguments = {
            "paired_comparisons": comparisons,
            "supplemental_development_costs": supplemental_costs,
            "max_leaves": args.max_leaves,
            "min_shape_groups_per_leaf": args.min_shape_groups_per_leaf,
            "fit_final_rules": True,
            "fit_cache": (
                PolicyFitCache(args.fit_cache_dir)
                if args.fit_cache_dir is not None
                else None
            ),
            "profiler_feature_catalog": profiler_catalog,
        }
        if backend == Backend.CUDA:
            projected_cache_keys = projected_domain_cache_keys(development)
            fit_arguments.update({
                "domain_corpus_provider": lambda domain: (
                    project_cuda_shape_resolved_candidates(
                        development.rows_for_generic_domain(domain),
                        known_generic_domain=domain,
                    )
                ),
                "domain_corpus_digest_provider": lambda domain: (
                    projected_cache_keys[domain]
                ),
            })
        policy = fit_generic_policy(development, **fit_arguments)
    fit_complete = time.perf_counter()
    plan = build_paired_request_plan(
        development,
        policy,
        comparisons,
        max_regret=args.max_regret,
        development_corpus_digest=source_digest,
        supplemental_shape_group_ids=frozenset(
            cost.shape_group_id
            for domain_costs in supplemental_costs.values()
            for cost in domain_costs
        ),
        supplemental_evidence_digests=supplemental_digests,
    )
    plan_complete = time.perf_counter()
    write_json(args.output, plan.request_manifest_mapping())
    request_shards = (
        write_request_shards(
            args.request_shard_dir,
            plan,
            max_requests_per_shard=args.max_requests_per_shard,
        )
        if args.request_shard_dir is not None
        else ()
    )
    if args.report:
        write_json(args.report, plan.report_mapping())
    write_complete = time.perf_counter()
    print(
        "paired planner timing "
        f"input_digest={digest_complete - planner_started:.3f}s "
        f"read={read_complete - digest_complete:.3f}s "
        f"development_filter={development_complete - read_complete:.3f}s "
        f"paired_evidence={paired_complete - development_complete:.3f}s "
        f"fit={fit_complete - paired_complete:.3f}s "
        f"plan={plan_complete - fit_complete:.3f}s "
        f"write={write_complete - plan_complete:.3f}s "
        f"total={write_complete - planner_started:.3f}s"
    )
    print(
        "paired plan "
        f"status={plan.status} domains={plan.cross_validation_domain_count} "
        f"failed_cells={plan.failed_cv_cell_count} "
        f"requests={len(plan.requests)} "
        f"request_shards={len(request_shards)} "
        f"confirmed_cv_misses={len(plan.confirmed_cv_misses)} "
        f"conflicts={len(plan.conflicts)} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
