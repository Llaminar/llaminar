"""Shared frozen-leaf paired certification for CPU NativeVNNI policies.

M=1 decode and grouped verifier dispatch have different forceable candidates,
but their final statistical contract is identical. Each frozen generic leaf
owns one fresh geometry, every source-format alias owns an interleaved timing
cell, and every non-selected forceable candidate is a challenger edge. This
module authenticates those edges, computes a simultaneous challenger bound,
and emits the common :class:`CertificationReport` consumed by installation.

Correctness is intentionally not estimated here. The common paired CSV reader
rejects route failures, repeat instability, or any byte mismatch before a
latency ratio reaches this reduction.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Mapping, Protocol, Sequence

import numpy as np

from .certification import (
    CertificationCell,
    CertificationReport,
    CertificationRuleCoverage,
)
from .corpus import GenericDomain, RuntimeKey
from .paired_confirmation import PairedCellEvidence, read_paired_confirmation_csv
from .paired_requests import (
    PAIRED_REQUEST_SCHEMA_VERSION,
    PairedTimingRequest,
    write_json,
)
from .segmented_policy import CandidatePointCost, GenericDispatchRule


DEFAULT_BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_BATCH = 4_096


def sha256_json(value: object) -> str:
    """Return a stable digest for one JSON-compatible transaction object."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def paired_evidence_digest(
    plan_digest: str,
    evidence_paths: Iterable[Path],
) -> str:
    """Bind a final artifact to every immutable paired CSV byte."""

    if not plan_digest.startswith("sha256:"):
        raise ValueError("paired evidence requires a sealed plan digest")
    content_digests = []
    for path in evidence_paths:
        content_digests.append(
            "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()
        )
    if not content_digests:
        raise ValueError("paired evidence inventory must not be empty")
    return sha256_json({
        "schema_version": "cpu-sealed-paired-evidence-v1",
        "plan_digest": plan_digest,
        "csv_content_digests": sorted(content_digests),
    })


def resolve_sealed_paired_evidence_paths(
    explicit_paths: Iterable[Path],
    evidence_directories: Iterable[Path],
) -> tuple[Path, ...]:
    """Expand bounded CLI inputs into one complete sealed shard inventory.

    A production seal can contain hundreds of CSV shards, so spelling every
    path as a repeated command-line option can exceed the operating system's
    argument-size limit.  Directory expansion remains strict: every request
    manifest must own exactly one final CSV, orphan CSVs are rejected, and an
    in-progress file prevents certification.  The downstream plan validator
    still authenticates every request ID and observation byte.
    """

    resolved = [Path(path) for path in explicit_paths]
    for directory_value in evidence_directories:
        directory = Path(directory_value)
        if not directory.is_dir():
            raise ValueError(
                f"CPU sealed paired evidence directory does not exist: {directory}"
            )
        inprogress = tuple(sorted(directory.glob("shard-*.csv.inprogress")))
        if inprogress:
            raise ValueError(
                "CPU sealed paired evidence directory contains unfinished "
                f"shards: first={inprogress[0]} count={len(inprogress)}"
            )
        manifests = tuple(sorted(directory.glob("shard-*.requests.json")))
        expected = {
            manifest.with_name(
                manifest.name.removesuffix(".requests.json") + ".csv"
            )
            for manifest in manifests
        }
        actual = set(directory.glob("shard-*.csv"))
        missing = sorted(expected - actual)
        orphaned = sorted(actual - expected)
        if not manifests or missing or orphaned:
            raise ValueError(
                "CPU sealed paired evidence directory is incomplete: "
                f"manifests={len(manifests)} csvs={len(actual)} "
                f"missing={len(missing)} orphaned={len(orphaned)}"
            )
        resolved.extend(sorted(actual))

    if not resolved:
        raise ValueError("CPU sealed paired evidence inventory is empty")
    if any(not path.is_file() for path in resolved):
        missing_path = next(path for path in resolved if not path.is_file())
        raise ValueError(f"CPU sealed paired evidence file is missing: {missing_path}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("CPU sealed paired evidence inventory repeats a path")
    return tuple(resolved)


def fresh_geometry_candidates_for_rule(
    rule: GenericDispatchRule,
    dimensions_by_group: Mapping[str, tuple[int, int]],
    forbidden_dimensions: frozenset[tuple[int, int]],
    maximum_weight_elements: int,
    count: int,
    *,
    launch_k_tiles_by_group: Mapping[str, int] | None = None,
    n_quanta: tuple[int, ...] = (32,),
    k_quantum: int = 32,
    stratify_by_work: bool = False,
    minimum_count: int | None = None,
) -> tuple[tuple[int, int], ...]:
    """Find deterministic fresh dimensions inside one frozen leaf.

    The search uses only development geometry anchors and immutable predicates.
    It never reads candidate latency, profiler evidence, or a sealed outcome.

    ``N`` and ``K`` have separate production lattices. NativeVNNI output rows
    permit every positive ``N``, while quantized ``K`` remains aligned to its
    32-value block. Multiple N quanta let a caller combine dense narrow-leaf
    coverage with broad route-forceability probes. Callers commit their chosen
    quanta in the sealed reserve identity; keeping the default at 32 preserves
    older grouped-verifier reserve generations.

    Complete Chebyshev-ring perimeters are searched instead of a handful of
    axis and diagonal rays.  Learned leaves can be narrow in a derived feature
    such as ``N * K``: moving only one axis, or moving both axes by the same
    amount, can immediately leave such a region even though many nearby 2-D
    lattice points remain valid.  Ring enumeration covers that lattice without
    changing the timing-free or deterministic nature of reserve construction.

    ``count`` is the desired reserve width. ``minimum_count`` lets a caller
    distinguish redundant route-probe choices from the structural minimum: a
    decode seal needs one fresh witness in every leaf, even when that leaf's
    complete legal lattice contains fewer than the preferred twelve probes.
    """

    required_count = count if minimum_count is None else minimum_count
    if (
        count < 1
        or required_count < 1
        or required_count > count
        or not n_quanta
        or any(quantum < 1 for quantum in n_quanta)
        or len(set(n_quanta)) != len(n_quanta)
        or k_quantum < 1
        or k_quantum % 32 != 0
    ):
        raise ValueError("CPU sealed reserve has invalid geometry quanta")

    bases = sorted({
        (
            *dimensions_by_group[group],
            max(0, (launch_k_tiles_by_group or {}).get(group, 0)),
        )
        for group in rule.development_shape_groups
        if group in dimensions_by_group
    }, key=lambda item: (item[0] * item[1], item))
    if not bases:
        raise ValueError("CPU sealed rule has no development geometry anchor")
    candidates: set[tuple[int, int]] = set()
    for n_quantum in n_quanta:
        quantum_start_count = len(candidates)
        for n0, k0, launch_k_tiles in bases:
            for radius in range(1, 129):
                # Visit each point on the square perimeter exactly once.  The
                # stable edge order is part of the reserve algorithm even
                # though accepted dimensions receive a canonical sort below.
                ring_steps = []
                for delta_n_steps in range(-radius, radius + 1):
                    ring_steps.append((delta_n_steps, -radius))
                    ring_steps.append((delta_n_steps, radius))
                for delta_k_steps in range(-radius + 1, radius):
                    ring_steps.append((-radius, delta_k_steps))
                    ring_steps.append((radius, delta_k_steps))
                for delta_n_steps, delta_k_steps in ring_steps:
                    delta_n = n_quantum * delta_n_steps
                    delta_k = k_quantum * delta_k_steps
                    n = n0 + delta_n
                    k = k0 + delta_k
                    if (
                        n <= 0
                        or k <= 0
                        or k % 32 != 0
                        or n * k > maximum_weight_elements
                        or (n, k) in forbidden_dimensions
                        or not rule.matches(n, k, launch_k_tiles)
                    ):
                        continue
                    candidates.add((n, k))
                if len(candidates) - quantum_start_count >= count * 2:
                    break
            if len(candidates) - quantum_start_count >= count * 2:
                break
    ordered = tuple(sorted(
        candidates, key=lambda item: (item[0] * item[1], item)
    ))
    if len(ordered) < required_count:
        raise ValueError(
            "CPU sealed reserve cannot provide enough fresh geometry for leaf "
            f"candidate={rule.candidate_id}: "
            f"found={len(ordered)} required={required_count}"
        )
    selected_count = min(count, len(ordered))
    if stratify_by_work and selected_count > 1:
        # A mixed dense/coarse reserve can contain many nearby low-work points.
        # Evenly spaced ranks retain both local narrow-leaf evidence and broad
        # geometries where additional production schedules become forceable.
        return tuple(
            ordered[index * (len(ordered) - 1) // (selected_count - 1)]
            for index in range(selected_count)
        )
    return ordered[:selected_count]


class CPUSealedRuleWitness(Protocol):
    """Structural fields required from either CPU seal planner."""

    rule_index: int
    shape: str
    n: int
    k: int


class CPUSealedRequest(Protocol):
    """Structural association between a frozen rule and one paired edge."""

    rule_index: int
    request: PairedTimingRequest


class CPUSealedPlan(Protocol):
    """Minimum transaction identity shared by the two CPU seal plans."""

    plan_digest: str


def load_cpu_burned_seal_development(
    development: object,
    plan_paths: Sequence[Path],
    evidence_directories: Sequence[Path],
    *,
    surface_name: str,
    read_plan: Callable[[Path], CPUSealedPlan],
    build_costs: Callable[
        [object, CPUSealedPlan, tuple[Path, ...]],
        Mapping[GenericDomain, tuple[CandidatePointCost, ...]],
    ],
) -> tuple[
    dict[GenericDomain, tuple[CandidatePointCost, ...]],
    tuple[str, ...],
    dict[str, tuple[int, int]],
]:
    """Authenticate and merge inspected CPU seals for a later generation.

    Development fitting, incremental paired planning, freeze, and final
    certification must all see the same burned transactions. Centralizing the
    merge prevents one phase from silently omitting a seal, accepting an empty
    evidence directory, changing a point's dimensions, or replaying the same
    transaction twice.
    """

    if len(plan_paths) != len(evidence_directories):
        raise ValueError(
            f"burned CPU {surface_name} plans and evidence directories must "
            "pair by position"
        )
    merged: dict[GenericDomain, list[CandidatePointCost]] = defaultdict(list)
    evidence_digests = []
    dimensions_by_group = {}
    for plan_path, evidence_directory in zip(
        plan_paths, evidence_directories, strict=True
    ):
        plan = read_plan(Path(plan_path))
        evidence = tuple(sorted(Path(evidence_directory).glob("*.csv")))
        if not evidence:
            raise ValueError(
                f"burned CPU {surface_name} evidence directory is empty: "
                f"{evidence_directory}"
            )
        costs = build_costs(development, plan, evidence)
        evidence_digests.append(
            paired_evidence_digest(plan.plan_digest, evidence)
        )
        for domain, domain_costs in costs.items():
            for cost in domain_costs:
                dimensions = (
                    cost.runtime_key.aggregate_n,
                    cost.runtime_key.k,
                )
                previous = dimensions_by_group.setdefault(
                    cost.shape_group_id, dimensions
                )
                if previous != dimensions:
                    raise ValueError(
                        f"burned CPU {surface_name} shape group changed "
                        "dimensions"
                    )
            merged[domain].extend(domain_costs)
    if len(set(evidence_digests)) != len(evidence_digests):
        raise ValueError(
            f"burned CPU {surface_name} evidence transaction is repeated"
        )
    return (
        {domain: tuple(costs) for domain, costs in sorted(merged.items())},
        tuple(evidence_digests),
        dimensions_by_group,
    )


def _validated_cpu_sealed_cells(
    requests: Sequence[CPUSealedRequest],
    evidence_paths: Iterable[Path],
) -> dict[
    tuple[int, str],
    list[tuple[PairedTimingRequest, PairedCellEvidence]],
]:
    """Authenticate paired CSV rows and group complete challenger surfaces.

    Certification and failed-seal adaptation must consume exactly the same
    evidence transaction. Keeping request replay in one helper prevents a
    future development adapter from accepting a relabeled M, geometry, source
    alias, ISA regime, or candidate edge that final certification would reject.
    """

    single_candidate_request_ids = frozenset(
        item.request.request_id
        for item in requests
        if item.request.reason ==
        "grouped_frozen_leaf_single_candidate_witness"
    )
    cells = []
    for path in evidence_paths:
        cells.extend(read_paired_confirmation_csv(
            (Path(path),),
            allow_identical_request_ids=single_candidate_request_ids,
        ))
    by_request = {}
    for cell in cells:
        if cell.request_id in by_request:
            raise ValueError(
                f"CPU paired evidence repeats request {cell.request_id}"
            )
        by_request[cell.request_id] = cell
    expected_ids = {item.request.request_id for item in requests}
    if set(by_request) != expected_ids:
        raise ValueError(
            "CPU sealed paired evidence is incomplete: "
            f"missing={len(expected_ids-set(by_request))} "
            f"unexpected={len(set(by_request)-expected_ids)}"
        )

    grouped: dict[
        tuple[int, str],
        list[tuple[PairedTimingRequest, PairedCellEvidence]],
    ] = defaultdict(list)
    for item in requests:
        request = item.request
        evidence = by_request[request.request_id]
        key = evidence.key
        if (
            key.backend != "cpu"
            or key.source_format != request.source_format.upper()
            or key.source_codebook != request.source_codebook
            or key.execution_codebook != request.execution_codebook
            or key.architecture_class != request.architecture_class
            or key.shape != request.shape
            or key.execution_mode != request.execution_mode
            or (key.m, key.n, key.k) != (request.m, request.n, request.k)
            or evidence.selected_candidate_id != request.selected_candidate_id
            or evidence.exact_candidate_id != request.exact_candidate_id
        ):
            raise ValueError(
                f"sealed paired evidence changed request {request.request_id}"
            )
        is_single_witness = request.request_id in single_candidate_request_ids
        if is_single_witness != (
            request.selected_candidate_id == request.exact_candidate_id
        ):
            raise ValueError(
                "CPU sealed single-candidate witness identity is inconsistent"
            )
        grouped[(item.rule_index, request.source_format)].append(
            (request, evidence)
        )
    return grouped


def cpu_sealed_development_costs(
    plan_digest: str,
    witnesses: Sequence[CPUSealedRuleWitness],
    requests: Sequence[CPUSealedRequest],
    evidence_paths: Iterable[Path],
    domain_for_witness: Callable[[CPUSealedRuleWitness], GenericDomain],
    expected_path: Callable[[GenericDomain], str],
) -> dict[GenericDomain, tuple[CandidatePointCost, ...]]:
    """Convert one inspected seal into generic-only development costs.

    A failed seal is no longer independent evidence for the policy that saw
    its result. Its paired timings are still valuable development evidence for
    a later policy generation. This adapter retains the paired ratio protocol,
    creates no ordinary observation and therefore no exact overlay, and leaves
    profiler predictions absent at the new point. The changed policy must be
    certified against another untouched plan.
    """

    if not plan_digest.startswith("sha256:"):
        raise ValueError("burned CPU seal lacks a plan digest")
    witnesses_by_rule = {item.rule_index: item for item in witnesses}
    if len(witnesses_by_rule) != len(witnesses):
        raise ValueError("burned CPU seal repeats a rule witness")
    grouped = _validated_cpu_sealed_cells(requests, evidence_paths)
    surfaces_by_rule: dict[
        int, dict[str, dict[str, float]]
    ] = defaultdict(dict)
    for (rule_index, source_format), edges in sorted(grouped.items()):
        witness = witnesses_by_rule.get(rule_index)
        if witness is None:
            raise ValueError("burned CPU seal request lacks a rule witness")
        domain = domain_for_witness(witness)
        if any(
            evidence.observed_path != expected_path(domain)
            for _, evidence in edges
        ):
            raise ValueError(
                f"burned CPU seal route disagrees with rule {rule_index}"
            )
        selected_ids = {request.selected_candidate_id for request, _ in edges}
        if len(selected_ids) != 1:
            raise ValueError("burned CPU seal changes its selected candidate")
        selected_id = next(iter(selected_ids))
        relative_latency = {selected_id: 1.0}
        for request, evidence in edges:
            if request.selected_candidate_id == request.exact_candidate_id:
                continue
            challenger_relative = math.exp(
                -statistics.median(evidence.log_latency_ratios)
            )
            previous = relative_latency.setdefault(
                request.exact_candidate_id, challenger_relative
            )
            if previous != challenger_relative:
                raise ValueError("burned CPU seal repeats a challenger edge")
        surfaces_by_rule[rule_index][source_format] = relative_latency

    result: dict[GenericDomain, list[CandidatePointCost]] = defaultdict(list)
    for rule_index, surfaces in sorted(surfaces_by_rule.items()):
        witness = witnesses_by_rule[rule_index]
        domain = domain_for_witness(witness)
        candidate_sets = {frozenset(values) for values in surfaces.values()}
        if len(candidate_sets) != 1:
            raise ValueError(
                "burned CPU seal candidate matrix differs across aliases"
            )
        candidates = sorted(next(iter(candidate_sets)))
        if not candidates:
            raise ValueError("burned CPU seal point has no forceable candidate")
        runtime_key = RuntimeKey(
            backend=domain.backend,
            architecture_class=domain.architecture_class,
            semantic_contract=domain.semantic_contract,
            operation_kind=domain.operation_kind,
            bundle_signature=domain.bundle_signature,
            projection_n_vector=(witness.n,),
            prepared_family_id=domain.prepared_family_id,
            packing_abi=domain.packing_abi,
            runtime_codebook_id=domain.runtime_codebook_id,
            execution_mode=domain.execution_mode,
            m=domain.m,
            aggregate_n=witness.n,
            k=witness.k,
            launch_k_tiles=getattr(witness, "k_tiles", 0),
        )
        best_by_surface = {
            source_format: min(values.values())
            for source_format, values in surfaces.items()
        }
        for candidate in candidates:
            regrets = tuple(
                values[candidate] / best_by_surface[source_format] - 1.0
                for source_format, values in sorted(surfaces.items())
            )
            ordered = sorted(regrets)
            p95_rank = max(0, math.ceil(0.95 * len(ordered)) - 1)
            result[domain].append(CandidatePointCost(
                runtime_key=runtime_key,
                shape_group_id=(
                    f"cpu-burned-seal:{plan_digest[7:19]}:"
                    f"rule-{rule_index}:n{witness.n}:k{witness.k}"
                ),
                candidate_id=candidate,
                max_surface_regret=max(regrets),
                p95_surface_regret=ordered[p95_rank],
                mean_surface_regret=statistics.fmean(regrets),
            ))
    return {
        domain: tuple(costs) for domain, costs in sorted(result.items())
    }


def write_cpu_sealed_request_shards(
    request_shard_directory: Path,
    requests: Sequence[CPUSealedRequest],
    development_corpus_digest: str,
    plan_digest: str,
    index_schema: str,
    *,
    max_requests_per_shard: int = 16,
) -> tuple[Path, ...]:
    """Publish architecture-homogeneous resumable C++ request shards."""

    if max_requests_per_shard < 1:
        raise ValueError("sealed request shard size must be positive")
    grouped: dict[str, list[PairedTimingRequest]] = defaultdict(list)
    for item in requests:
        grouped[item.request.architecture_class].append(item.request)
    request_shard_directory.mkdir(parents=True, exist_ok=True)
    paths = []
    index = []
    shard_index = 0
    for architecture, architecture_requests in sorted(grouped.items()):
        architecture_token = hashlib.sha256(
            architecture.encode()
        ).hexdigest()[:12]
        for begin in range(0, len(architecture_requests), max_requests_per_shard):
            selected = tuple(sorted(architecture_requests))[
                begin:begin + max_requests_per_shard
            ]
            payload = {
                "schema_version": PAIRED_REQUEST_SCHEMA_VERSION,
                "backend": "cpu",
                "development_corpus_digest": development_corpus_digest,
                "paired_evidence_digest": plan_digest,
                "max_regret": 0.05,
                "request_count": len(selected),
                "requests": [asdict(request) for request in selected],
            }
            request_token = hashlib.sha256(json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode()).hexdigest()[:12]
            name = (
                f"shard-{shard_index:04d}.cpu.{architecture_token}."
                f"{request_token}.requests.json"
            )
            shard_path = request_shard_directory / name
            write_json(shard_path, payload)
            paths.append(shard_path)
            index.append({
                "path": name,
                "architecture_class": architecture,
                "request_count": len(selected),
                "request_digest": request_token,
            })
            shard_index += 1
    expected = {item.name for item in paths}
    expected_evidence = {
        item.name.removesuffix(".requests.json") + ".csv"
        for item in paths
    }
    for stale in request_shard_directory.glob("shard-*.requests.json"):
        if stale.name not in expected:
            stale.unlink()
    # A changed frozen policy produces content-addressed request names.  Keep
    # the live directory generation-pure as those manifests turn over: burned
    # evidence loaders intentionally authenticate every CSV in their archived
    # directory, so an orphan from an older plan must never survive beside the
    # current request inventory and later masquerade as additive evidence.
    for stale in request_shard_directory.glob("shard-*.csv"):
        if stale.name not in expected_evidence:
            stale.unlink()
    for stale in request_shard_directory.glob("shard-*.csv.inprogress"):
        if stale.name.removesuffix(".inprogress") not in expected_evidence:
            stale.unlink()
    write_json(request_shard_directory / "index.json", {
        "schema_version": index_schema,
        "plan_digest": plan_digest,
        "request_count": len(requests),
        "shard_count": len(paths),
        "shards": index,
    })
    return tuple(paths)


def _physical_core_count() -> int:
    """Count physical cores in the current affinity mask.

    Bootstrap workers are compute-heavy NumPy processes. Hyperthreads contend
    for the same execution resources and made prior reductions slower, so the
    default deliberately caps process parallelism at physical cores.
    """

    logical = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    cores = set()
    for cpu in logical:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            cores.add((
                (root / "physical_package_id").read_text().strip(),
                (root / "core_id").read_text().strip(),
            ))
        except OSError:
            return max(1, len(logical))
    return max(1, len(cores))


def _bootstrap_cell(
    arguments: tuple[
        int,
        str,
        tuple[tuple[str, tuple[float, ...]], ...],
        int,
    ],
) -> tuple[int, str, float, float]:
    """Compute observed and simultaneous challenger regret for one cell."""

    cell_index, plan_digest, edges, replicates = arguments
    observed_edges = tuple(
        (candidate, math.expm1(float(np.median(np.asarray(values)))))
        for candidate, values in edges
    )
    best_candidate, observed = max(
        observed_edges, key=lambda item: (item[1], item[0])
    )
    observed = max(0.0, observed)
    maxima = np.full(replicates, -np.inf, dtype=np.float64)
    for candidate, values_tuple in edges:
        values = np.asarray(values_tuple, dtype=np.float64)
        seed = int.from_bytes(hashlib.sha256(
            (
                "cpu-sealed-paired-bootstrap-v1:"
                f"{plan_digest}:{cell_index}:{candidate}"
            ).encode()
        ).digest()[:16], "little")
        rng = np.random.Generator(np.random.PCG64(seed))
        for begin in range(0, replicates, BOOTSTRAP_BATCH):
            end = min(replicates, begin + BOOTSTRAP_BATCH)
            indices = rng.integers(
                0,
                values.size,
                size=(end - begin, values.size),
                dtype=np.int32,
            )
            regrets = np.expm1(np.median(values[indices], axis=1))
            np.maximum(maxima[begin:end], regrets, out=maxima[begin:end])
    rank = max(0, math.ceil(0.95 * replicates) - 1)
    upper = max(0.0, float(np.partition(maxima, rank)[rank]))
    if observed == 0.0:
        best_candidate = "selected"
    return cell_index, best_candidate, observed, upper


def certify_cpu_sealed_pairs(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    plan_frozen_generic_policy_digest: str,
    plan_digest: str,
    witnesses: Sequence[CPUSealedRuleWitness],
    requests: Sequence[CPUSealedRequest],
    evidence_paths: Iterable[Path],
    expected_path: Callable[[GenericDispatchRule], str],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    workers: int | None = None,
) -> CertificationReport:
    """Build one generic-only certificate from fresh paired evidence.

    Every request field is replayed against the paired CSV. This prevents a
    correctly formatted timing from being relabeled onto a different M,
    geometry, source alias, ISA regime, or candidate edge.
    """

    if plan_frozen_generic_policy_digest != frozen_generic_policy_digest:
        raise ValueError("CPU paired seal belongs to another frozen policy")
    if not plan_digest.startswith("sha256:"):
        raise ValueError("CPU paired seal lacks a plan digest")
    if bootstrap_replicates < 1_000:
        raise ValueError("CPU sealed bootstrap requires at least 1000 replicates")
    if len(witnesses) != len(rules):
        raise ValueError("CPU paired seal does not witness every frozen rule")

    grouped = _validated_cpu_sealed_cells(requests, evidence_paths)

    tasks = []
    metadata = []
    for cell_index, ((rule_index, source_format), edges) in enumerate(
        sorted(grouped.items())
    ):
        rule = rules[rule_index]
        witness = witnesses[rule_index]
        required_path = expected_path(rule)
        if any(evidence.observed_path != required_path for _, evidence in edges):
            raise ValueError(
                f"CPU sealed route disagrees with rule {rule_index}"
            )
        tasks.append((
            cell_index,
            plan_digest,
            tuple(
                (request.exact_candidate_id, evidence.log_latency_ratios)
                for request, evidence in sorted(
                    edges, key=lambda item: item[0].exact_candidate_id
                )
            ),
            bootstrap_replicates,
        ))
        metadata.append((rule_index, source_format, rule, witness))

    single_candidate_indices = {
        cell_index
        for cell_index, ((_rule_index, _source_format), edges) in enumerate(
            sorted(grouped.items())
        )
        if all(
            request.selected_candidate_id == request.exact_candidate_id
            for request, _evidence in edges
        )
    }
    tasks = [task for task in tasks if task[0] not in single_candidate_indices]
    if tasks:
        worker_count = min(workers or _physical_core_count(), len(tasks))
        if worker_count <= 1:
            results = tuple(_bootstrap_cell(task) for task in tasks)
        else:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count
            ) as executor:
                results = tuple(executor.map(_bootstrap_cell, tasks))
    else:
        results = ()
    by_index = {
        **{index: ("selected", 0.0, 0.0) for index in single_candidate_indices},
        **{result[0]: result[1:] for result in results},
    }

    certificate_cells = []
    rule_hits = defaultdict(int)
    for cell_index, (rule_index, source_format, rule, witness) in enumerate(
        metadata
    ):
        exact_candidate, observed, upper = by_index[cell_index]
        if exact_candidate == "selected":
            exact_candidate = rule.candidate_id
        runtime_key = RuntimeKey(
            backend=rule.domain.backend,
            architecture_class=rule.domain.architecture_class,
            semantic_contract=rule.domain.semantic_contract,
            operation_kind=rule.domain.operation_kind,
            bundle_signature=rule.domain.bundle_signature,
            projection_n_vector=(witness.n,),
            prepared_family_id=rule.domain.prepared_family_id,
            packing_abi=rule.domain.packing_abi,
            runtime_codebook_id=rule.domain.runtime_codebook_id,
            execution_mode=rule.domain.execution_mode,
            m=rule.domain.m,
            aggregate_n=witness.n,
            k=witness.k,
            launch_k_tiles=getattr(witness, "k_tiles", 0),
        )
        certificate_cells.append(CertificationCell(
            runtime_key=runtime_key,
            shape_group_id=(
                f"cpu-sealed:{witness.shape}:n{witness.n}:k{witness.k}:"
                f"source={source_format}"
            ),
            domain=rule.domain,
            selected_candidate_id=rule.candidate_id,
            exact_candidate_id=exact_candidate,
            observed_worst_surface_regret=observed,
            simultaneous_95pct_upper_regret=upper,
            alias_count=1,
            execution_mode_count=1,
        ))
        rule_hits[rule_index] += 1
    coverage = tuple(
        CertificationRuleCoverage(rule, rule_hits[index])
        for index, rule in enumerate(rules)
    )
    return CertificationReport(
        frozen_generic_policy_digest=frozen_generic_policy_digest,
        cells=tuple(certificate_cells),
        sealed_cell_count=len(certificate_cells),
        out_of_scope_cell_count=0,
        required_cell_count=len(certificate_cells),
        covered_cell_count=len(certificate_cells),
        verifier_bitwise_failures=0,
        unexercised_rule_count=sum(
            item.sealed_hit_count == 0 for item in coverage
        ),
        unpromoted_domain_count=0,
        rule_coverage=coverage,
    )
