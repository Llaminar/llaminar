"""Plan concrete paired timings from p95-failing cross-validation domains.

Broad all-candidate timing is deliberately provisional. It identifies a
bounded generic policy and the domains whose held-out p95 regret is not below
five percent. Within those domains, cells over the same budget identify the
candidate edges that can improve the fit. This module closes the loop without
hand-transcribing candidate IDs: CUDA formulas resolve to their concrete
exact-KB launch, CPU schedules retain their build/runtime/thread class, and the
accumulated paired tournament emits only measurements that can still change a
decision.

The emitted JSON is consumed directly by the CUDA performance trainer. It is a
development-only measurement plan, never a production dispatch artifact.
Sealed observations are intentionally absent from this workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_observation import read_observation_csv, read_observation_rows
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
    runtime_codebook: int
    execution_mode: str
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
    competitive_frontier_request_count: int
    domains: tuple[PairedDomainSummary, ...]
    requests: tuple[PairedTimingRequest, ...]
    confirmed_cv_misses: tuple[PairedPlanIssue, ...]
    conflicts: tuple[PairedPlanIssue, ...]
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
            "competitive_frontier_request_count": (
                self.competitive_frontier_request_count
            ),
            "domains": [asdict(domain) for domain in self.domains],
            "confirmed_cv_misses": [
                asdict(issue) for issue in self.confirmed_cv_misses
            ],
            "conflicts": [asdict(issue) for issue in self.conflicts],
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
            })
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
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


def _paired_candidates_connected(
    edges: Iterable[PairedTimingComparison],
    first_candidate_id: str,
    second_candidate_id: str,
) -> bool:
    """Return whether validated tournament evidence links two candidates.

    Cost construction solves every connected component as one weighted
    log-latency graph. An alternate CV model therefore needs connectivity, not
    a quadratic inventory of direct candidate pairs. The currently selected
    model still receives a direct confirmation edge before terminal promotion
    or failure.
    """

    if first_candidate_id == second_candidate_id:
        return True
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        selected = edge.selected_effective_candidate_id
        exact = edge.exact_effective_candidate_id
        adjacency.setdefault(selected, set()).add(exact)
        adjacency.setdefault(exact, set()).add(selected)
    pending = [first_candidate_id]
    visited = set()
    while pending:
        candidate = pending.pop()
        if candidate == second_candidate_id:
            return True
        if candidate in visited:
            continue
        visited.add(candidate)
        pending.extend(adjacency.get(candidate, set()) - visited)
    return False


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


def build_paired_request_plan(
    development: ObservationCorpus,
    policy: GenericPolicy,
    comparisons: Mapping[PairedCellKey, tuple[PairedTimingComparison, ...]],
    *,
    max_regret: float = DEFAULT_MAX_REGRET,
    development_corpus_digest: str | None = None,
) -> PairedRequestPlan:
    """Classify p95-failing CV domains and emit actionable pair timings.

    Maximum cell regret remains visible in the report, but an isolated tail
    cell cannot open a paired-refinement transaction after its complete domain
    already satisfies the installation p95 threshold.
    """

    if not 0.0 < max_regret < 1.0:
        raise ValueError("max_regret must be strictly between zero and one")
    backends = {row.backend for row in development}
    if len(backends) != 1 or next(iter(backends)) not in {
        Backend.CPU,
        Backend.CUDA,
        Backend.ROCM,
    }:
        raise ValueError("paired planning requires one supported backend")
    backend = next(iter(backends))

    evidence_digest = paired_comparison_digest(comparisons)
    requests: list[PairedTimingRequest] = []
    failures: list[PairedPlanIssue] = []
    conflicts: list[PairedPlanIssue] = []
    failed_cells = 0
    passing_cells = 0
    equivalent_surfaces = 0
    domains = tuple(sorted(
        PairedDomainSummary(
            backend=validation.domain.backend.value,
            runtime_codebook=validation.domain.runtime_codebook_id,
            execution_mode=validation.domain.execution_mode.value,
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

    for validation in policy.cross_validation:
        if validation.p95_regret < max_regret:
            passing_cells += len(validation.cells)
            continue
        for cell in validation.cells:
            if cell.observed_broad_regret < max_regret:
                passing_cells += 1
                continue
            failed_cells += 1
            if cell.runtime_key.semantic_contract != SemanticContract.FAST:
                raise ValueError("paired planner currently accepts only Fast cells")
            if cell.runtime_key.m != 1:
                raise ValueError("paired planner currently accepts only public M1")
            for source_format, rows in sorted(
                _surface_rows(development, cell).items()
            ):
                key = _paired_key(rows, cell)
                selected = _effective_candidate(
                    rows, cell.selected_candidate_id
                )
                exact = _effective_candidate(rows, cell.exact_candidate_id)
                if selected == exact:
                    equivalent_surfaces += 1
                    continue
                direct_regret = _direct_paired_regret(
                    comparisons.get(key, ()), selected, exact
                )
                issue_fields = {
                    "source_format": source_format,
                    "shape": key.shape,
                    "shape_group_id": cell.shape_group_id,
                    "execution_mode": key.execution_mode,
                    "m": key.m,
                    "n": key.n,
                    "k": key.k,
                    "selected_candidate_id": selected,
                    "exact_candidate_id": exact,
                    "observed_cv_regret": cell.observed_broad_regret,
                    "direct_paired_regret": direct_regret,
                }
                if direct_regret is not None and direct_regret >= max_regret:
                    failures.append(PairedPlanIssue(
                        **issue_fields,
                        reason="direct_pair_confirms_over_budget",
                    ))
                    continue

                reason = (
                    "missing_direct_tournament_edge"
                    if direct_regret is None
                    else "tournament_fit_conflicts_with_direct_edge"
                )
                if direct_regret is not None:
                    conflicts.append(PairedPlanIssue(
                        **issue_fields,
                        reason=reason,
                    ))
                request_fields = {
                    key_name: value
                    for key_name, value in issue_fields.items()
                    if key_name != "direct_paired_regret"
                }
                request_fields.update({
                    "source_codebook": key.source_codebook,
                    "execution_codebook": key.execution_codebook,
                    "architecture_class": _request_architecture_class(rows),
                })
                requests.append(PairedTimingRequest(
                    request_id=_request_id(request_fields, evidence_digest),
                    **request_fields,
                    reason=reason,
                ))

    # The selected model is only one point on the retained CV frontier. A new
    # direct timing can promote the next model and expose another candidate
    # edge. Measure missing edges from the bounded competitive frontier in this
    # same batch so refinement closes as a tournament instead of advancing one
    # expensive whole-corpus fit at a time. Frontier edges already represented
    # by direct paired evidence need no further launch, and only selected-model
    # misses can make the policy terminally fail.
    selected_cell_identities = {
        (
            cell.runtime_key,
            cell.shape_group_id,
            cell.selected_candidate_id,
            cell.exact_candidate_id,
        )
        for validation in policy.cross_validation
        for cell in validation.cells
    }
    requested_edges = {
        (
            request.source_format,
            request.source_codebook,
            request.execution_codebook,
            request.architecture_class,
            request.shape,
            request.shape_group_id,
            request.execution_mode,
            request.m,
            request.n,
            request.k,
            request.selected_candidate_id,
            request.exact_candidate_id,
        )
        for request in requests
    }
    frontier_request_count = 0
    for validation in policy.cross_validation:
        if validation.p95_regret < max_regret:
            continue
        for cell in validation.competitive_cells:
            cell_identity = (
                cell.runtime_key,
                cell.shape_group_id,
                cell.selected_candidate_id,
                cell.exact_candidate_id,
            )
            if (
                cell_identity in selected_cell_identities
                or cell.observed_broad_regret < max_regret
            ):
                continue
            for source_format, rows in sorted(
                _surface_rows(development, cell).items()
            ):
                key = _paired_key(rows, cell)
                selected = _effective_candidate(
                    rows, cell.selected_candidate_id
                )
                exact = _effective_candidate(rows, cell.exact_candidate_id)
                if selected == exact:
                    continue
                if _paired_candidates_connected(
                    comparisons.get(key, ()), selected, exact
                ):
                    continue
                edge = (
                    source_format,
                    key.source_codebook,
                    key.execution_codebook,
                    _request_architecture_class(rows),
                    key.shape,
                    cell.shape_group_id,
                    key.execution_mode,
                    key.m,
                    key.n,
                    key.k,
                    selected,
                    exact,
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
                    reason="competitive_model_frontier_edge",
                ))
                requested_edges.add(edge)
                frontier_request_count += 1

    duplicate_ids = {
        request.request_id
        for request in requests
        if sum(item.request_id == request.request_id for item in requests) > 1
    }
    if duplicate_ids:
        raise ValueError(f"duplicate paired request IDs: {sorted(duplicate_ids)}")

    validated_domains = {result.domain for result in policy.cross_validation}
    unvalidated = tuple(
        repr(domain)
        for domain in policy.unpromoted_domains
        if domain not in validated_domains
    )
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
        competitive_frontier_request_count=frontier_request_count,
        domains=domains,
        requests=tuple(sorted(requests)),
        confirmed_cv_misses=tuple(sorted(failures)),
        conflicts=tuple(sorted(conflicts)),
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
    never cross architecture classes. Bounded request counts also keep fixture
    setup and retained work small enough that a process interruption discards
    only one short `.inprogress` output rather than the complete tournament.
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
        requests = sorted(requests)
        architecture_digest = hashlib.sha256(
            architecture_class.encode("utf-8")
        ).hexdigest()[:12]
        for begin in range(0, len(requests), max_requests_per_shard):
            selected = tuple(requests[begin:begin + max_requests_per_shard])
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
    """Read shards independently so legacy v1 files may hold parallel edges."""

    cells = []
    for path in paths:
        cells.extend(read_paired_confirmation_csv((path,)))
    return tuple(cells)


def _development_corpus(
    observations: Iterable,
    manifest_path: Path,
) -> ObservationCorpus:
    """Retain only compact direct Fast-M1 development observations.

    Shape-resolved formula aliases are projected lazily for a cost/CV cache-miss
    domain. Keeping them out of this owning corpus avoids expanding a modest
    source dataset into a gigabyte-scale repeated-metadata intermediate.
    """

    rows = tuple(observations)
    manifest = load_shape_manifest(manifest_path)
    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in rows),
        verifier=False,
        manifest=manifest,
    )
    direct_development = tuple(
        row
        for row in rows
        if row.semantic_contract == SemanticContract.FAST
        and row.m == 1
        and assignments[row.shape_group_id] == ShapePartition.DEVELOPMENT
    )
    return ObservationCorpus._from_validated(direct_development)


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


def main() -> int:
    """Fit development CV, emit the next paired batch, and report disposition."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--paired-csv",
        action="append",
        type=Path,
        default=[],
        help="Validated paired evidence shard; repeat for every retained file",
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
    development = _development_corpus(observations, args.shape_manifest)
    del observations
    backends = {row.backend for row in development}
    if len(backends) != 1:
        parser.error("paired planning requires one homogeneous backend corpus")
    backend = next(iter(backends))
    if backend == Backend.CPU:
        development = development.with_collapsed_aspect_domains()
    development_complete = time.perf_counter()
    profiler_catalog = (
        load_profiler_feature_catalog(
            development,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
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
            "max_leaves": args.max_leaves,
            "min_shape_groups_per_leaf": args.min_shape_groups_per_leaf,
            "fit_final_rules": False,
            "fit_cache": (
                PolicyFitCache(args.fit_cache_dir)
                if args.fit_cache_dir is not None
                else None
            ),
            "profiler_feature_catalog": profiler_catalog,
        }
        if backend == Backend.CUDA:
            fit_arguments.update({
                "domain_corpus_provider": lambda domain: (
                    project_cuda_shape_resolved_candidates(
                        development.rows_for_generic_domain(domain)
                    )
                ),
                "domain_corpus_digest_provider": lambda domain: (
                    projected_domain_cache_key(development, domain)
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
