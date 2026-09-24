"""Fresh frozen-leaf sealing for the CPU NativeVNNI M=1 policy.

The broad development sweep is intentionally unsuitable as a final economy
certificate.  Candidate launches are timed in separate blocks, so scheduler
noise and clock drift make a conservative unpaired confidence interval much
wider than the dispatch differences being measured.  A fixed shape holdout is
also unable to exercise every leaf of a tree whose geometry is learned from
the development corpus.

This module implements a two-stage, post-freeze transaction:

* derive several fresh geometry candidates for every immutable generic leaf
  in deterministic affinity-visible physical-core workers;
* ask the production C++ tile planner whether each geometry uses serial full-K
  or ordered K-part arithmetic on each ISA regime;
* select one matching physical route per leaf without reading timing data;
* compare the frozen selected schedule against every other forceable schedule
  in randomized, interleaved timing pairs; and
* construct per-cell simultaneous challenger bounds before aggregating p95 by
  generic domain.

The generic policy cannot change anywhere in this lifetime.  Route probes are
timing-free, and the paired evidence is useful only when its plan, candidate
registry, route manifests, development corpus, and frozen generic digest all
match byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_registry import (
    cpu_native_vnni_decode_registry,
    resolve_cpu_native_vnni_decode_n_block_chunks,
)
from .certification import CertificationReport
from .corpus import GenericDomain, ObservationCorpus
from .cpu_prefill_route_manifest import (
    CPUPrefillSerialRoute,
    read_cpu_prefill_route_manifests,
)
from .format_registry import format_spec, registry_digest, runtime_aliases
from .paired_requests import (
    PairedTimingRequest,
    write_json,
)
from .cpu_sealed_paired import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    certify_cpu_sealed_pairs,
    cpu_sealed_development_costs,
    fresh_geometry_candidates_for_rule,
    write_cpu_sealed_request_shards,
)
from .segmented_policy import (
    CandidatePointCost,
    GenericDispatchRule,
    _physical_core_worker_count,
)
from .shape_manifest import NativeVNNIShapeManifest


CPU_DECODE_SEALED_PROBE_SCHEMA = "cpu-decode-sealed-route-probe-v3"
CPU_DECODE_SEALED_PLAN_SCHEMA = "cpu-decode-frozen-leaf-paired-seal-v3"
LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA = "cpu-decode-frozen-leaf-paired-seal-v2"
CPU_DECODE_SEALED_RESERVE_SCHEMA = "cpu-decode-sealed-reserve-v7"
CPU_DECODE_SEALED_GEOMETRY_PREFIX = "CPUDecodeAutoSeal_"
SERIAL_FULL_K_BUNDLE = "single-native-vnni-decode:serial-full-k:fp32-output:v1"
SERIAL_KPART_BUNDLE = "single-native-vnni-decode:serial-kpart:fp32-output:v1"
DEFAULT_GEOMETRY_CANDIDATES_PER_RULE = 12


# Frozen-rule reserve searches are independent but each may inspect tens of
# thousands of lattice points. The parent materializes these immutable inputs
# once, then Linux ``fork`` workers inherit their pages without repeatedly
# serializing the development corpus through multiprocessing pipes.
_PARALLEL_PROBE_RULES: tuple[GenericDispatchRule, ...] = ()
_PARALLEL_PROBE_DIMENSIONS_BY_GROUP: Mapping[str, tuple[int, int]] = {}
_PARALLEL_PROBE_FORBIDDEN: frozenset[tuple[int, int]] = frozenset()
_PARALLEL_PROBE_MAXIMUM_WEIGHT_ELEMENTS = 0
_PARALLEL_PROBE_DEVELOPMENT_ROWS_BY_GROUP: Mapping[str, tuple[object, ...]] = {}
_PARALLEL_PROBE_SUPPLEMENTAL_COSTS_BY_GROUP: Mapping[
    str, tuple[CandidatePointCost, ...]
] = {}
_PARALLEL_PLAN_RULES: tuple[GenericDispatchRule, ...] = ()
_PARALLEL_PLAN_ROUTES_BY_DOMAIN: Mapping[
    tuple[int, str, int, str], tuple[CPUPrefillSerialRoute, ...]
] = {}
_PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY: Mapping[
    tuple[int, int, int, int], tuple[str, ...]
] = {}


def _sha256_json(value: object) -> str:
    """Return a stable digest for one JSON-compatible transaction object."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def cpu_decode_sealed_reserve_commitment(
    manifest: NativeVNNIShapeManifest,
) -> str:
    """Commit fitting to the timing-free fresh-geometry reserve algorithm."""

    return _sha256_json({
        "schema_version": CPU_DECODE_SEALED_RESERVE_SCHEMA,
        "shape_manifest_digest": manifest.digest(),
        "candidate_registry_digest": cpu_native_vnni_decode_registry().digest(),
        "format_registry_digest": registry_digest(),
        "geometry_prefix": CPU_DECODE_SEALED_GEOMETRY_PREFIX,
        "candidate_count_per_rule": DEFAULT_GEOMETRY_CANDIDATES_PER_RULE,
        "minimum_candidate_count_per_rule": 1,
        "search_pattern": "complete-chebyshev-ring-perimeter-v1",
        "n_delta_quanta": [1, 32],
        "k_delta_quantum": 32,
        "candidate_selection": "up-to-12-work-quantile-stratified-v2",
        "maximum_delta_steps": 128,
        "runtime_geometry": (
            "development-and-burned-anchor-launch-k-tiles-v2"
        ),
    })


def _fresh_cpu_decode_probe_geometries_at(
    rule_index: int,
) -> tuple[tuple[int, int], ...]:
    """Build one rule's reserve from process-inherited immutable indexes."""

    rule = _PARALLEL_PROBE_RULES[rule_index]
    launch_k_tiles_by_group: dict[str, int] = {}
    for shape_group_id in rule.development_shape_groups:
        for row in _PARALLEL_PROBE_DEVELOPMENT_ROWS_BY_GROUP.get(
            shape_group_id, ()
        ):
            if (
                getattr(
                    row,
                    "architecture_class",
                    rule.domain.architecture_class,
                ) != rule.domain.architecture_class
                or getattr(
                    row,
                    "runtime_codebook_id",
                    rule.domain.runtime_codebook_id,
                ) != rule.domain.runtime_codebook_id
                or getattr(
                    row,
                    "bundle_signature",
                    rule.domain.bundle_signature,
                ) != rule.domain.bundle_signature
                or getattr(
                    row, "operation_kind", rule.domain.operation_kind
                ) != rule.domain.operation_kind
                or getattr(
                    row, "execution_mode", rule.domain.execution_mode
                ) != rule.domain.execution_mode
                or getattr(row, "m", rule.domain.m) != rule.domain.m
            ):
                continue
            launch_k_tiles = int(getattr(row, "launch_k_tiles", 0))
            previous = launch_k_tiles_by_group.setdefault(
                row.shape_group_id, launch_k_tiles
            )
            if previous != launch_k_tiles:
                raise ValueError(
                    "CPU decode shape group changed frozen K-tile geometry"
                )

    # A burned seal becomes legitimate development evidence for the next
    # policy generation. Its fresh shape may consequently become the only
    # anchor owned by a later leaf. Preserve the exact K-tile telemetry carried
    # by that evidence just as broad-sweep observations do; treating it as
    # absent silently rewrites ordered K-part into full-K arithmetic.
    for shape_group_id in rule.development_shape_groups:
        for cost in _PARALLEL_PROBE_SUPPLEMENTAL_COSTS_BY_GROUP.get(
            shape_group_id, ()
        ):
            runtime = cost.runtime_key
            if (
                runtime.architecture_class != rule.domain.architecture_class
                or runtime.runtime_codebook_id
                != rule.domain.runtime_codebook_id
                or runtime.bundle_signature != rule.domain.bundle_signature
                or runtime.operation_kind != rule.domain.operation_kind
                or runtime.execution_mode != rule.domain.execution_mode
                or runtime.m != rule.domain.m
            ):
                continue
            launch_k_tiles = int(runtime.launch_k_tiles)
            previous = launch_k_tiles_by_group.setdefault(
                cost.shape_group_id, launch_k_tiles
            )
            if previous != launch_k_tiles:
                raise ValueError(
                    "CPU decode supplemental shape group changed frozen "
                    "K-tile geometry"
                )

    return fresh_geometry_candidates_for_rule(
        rule,
        _PARALLEL_PROBE_DIMENSIONS_BY_GROUP,
        _PARALLEL_PROBE_FORBIDDEN,
        _PARALLEL_PROBE_MAXIMUM_WEIGHT_ELEMENTS,
        DEFAULT_GEOMETRY_CANDIDATES_PER_RULE,
        launch_k_tiles_by_group=launch_k_tiles_by_group,
        n_quanta=(1, 32),
        k_quantum=32,
        stratify_by_work=True,
        minimum_count=1,
    )


def _rule_regime(rule: GenericDispatchRule) -> tuple[str, int]:
    """Decode the build/runtime/thread identity embedded in one CPU rule."""

    parts = rule.domain.architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError("CPU decode rule has malformed architecture class")
    build = parts[1].removeprefix("build=").lower()
    runtime = parts[2].removeprefix("runtime=").lower()
    threads = int(parts[3].removeprefix("threads="))
    regime = f"{build}-build.{runtime}-runtime"
    if regime not in {
        "avx2-build.avx2-runtime",
        "avx512-build.avx2-runtime",
        "avx512-build.avx512-runtime",
    } or threads <= 0:
        raise ValueError(f"unsupported CPU decode ISA regime {regime}")
    return regime, threads


def _decode_bundle(route: CPUPrefillSerialRoute) -> str:
    """Translate a production tile probe into the decode bundle identity."""

    return SERIAL_KPART_BUNDLE if route.k_tiles > 1 else SERIAL_FULL_K_BUNDLE


def _fresh_shape_name(n: int, k: int) -> str:
    """Name a generated geometry by dimensions so aliases share fixtures."""

    return f"{CPU_DECODE_SEALED_GEOMETRY_PREFIX}N{n}_K{k}"


@dataclass(frozen=True, order=True)
class CPUDecodeSealedProbeShape:
    """One fresh timing-free geometry submitted to the C++ route planner."""

    name: str
    n: int
    k: int


@dataclass(frozen=True)
class CPUDecodeSealedRouteProbe:
    """Frozen-policy-bound candidate geometry pool for route probing."""

    schema_version: str
    frozen_generic_policy_digest: str
    development_corpus_digest: str
    sealed_build_id: str
    reserve_commitment: str
    shapes: tuple[CPUDecodeSealedProbeShape, ...]

    def canonical_mapping(self) -> dict[str, object]:
        """Return the exact JSON consumed by the C++ timing-free probe."""

        return {
            "schema_version": self.schema_version,
            "frozen_generic_policy_digest": self.frozen_generic_policy_digest,
            "development_corpus_digest": self.development_corpus_digest,
            "sealed_build_id": self.sealed_build_id,
            "reserve_commitment": self.reserve_commitment,
            "shapes": [asdict(shape) for shape in self.shapes],
        }

    @property
    def digest(self) -> str:
        """Bind witness selection to the complete candidate geometry pool."""

        return _sha256_json(self.canonical_mapping())


@dataclass(frozen=True, order=True)
class CPUDecodeSealedRuleWitness:
    """One frozen generic leaf and its fresh production-route witness."""

    rule_index: int
    shape: str
    n: int
    k: int
    k_tiles: int
    architecture_class: str
    runtime_codebook: int
    bundle_signature: str
    selected_candidate_id: str
    forceable_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, order=True)
class CPUDecodeSealedRequest:
    """Associate one physical timing edge with its frozen generic leaf."""

    rule_index: int
    request: PairedTimingRequest


@dataclass(frozen=True)
class CPUDecodeSealedPlan:
    """Self-contained leaf witnesses and exhaustive paired challenger edges."""

    schema_version: str
    frozen_generic_policy_digest: str
    development_corpus_digest: str
    sealed_build_id: str
    reserve_commitment: str
    route_probe_digest: str
    route_manifest_digest: str
    candidate_registry_digest: str
    format_registry_digest: str
    plan_digest: str
    rule_witnesses: tuple[CPUDecodeSealedRuleWitness, ...]
    requests: tuple[CPUDecodeSealedRequest, ...]

    def canonical_mapping(self, *, include_plan_digest: bool = True) -> dict[str, object]:
        """Return stable plan JSON, retaining leaf ownership outside C++ shards."""

        result = {
            "schema_version": self.schema_version,
            "frozen_generic_policy_digest": self.frozen_generic_policy_digest,
            "development_corpus_digest": self.development_corpus_digest,
            "sealed_build_id": self.sealed_build_id,
            "reserve_commitment": self.reserve_commitment,
            "route_probe_digest": self.route_probe_digest,
            "route_manifest_digest": self.route_manifest_digest,
            "candidate_registry_digest": self.candidate_registry_digest,
            "format_registry_digest": self.format_registry_digest,
            "rule_witnesses": [asdict(item) for item in self.rule_witnesses],
            "requests": [
                {"rule_index": item.rule_index, **asdict(item.request)}
                for item in self.requests
            ],
        }
        if include_plan_digest:
            result["plan_digest"] = self.plan_digest
        return result


def build_cpu_decode_sealed_route_probe(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    sealed_build_id: str,
    supplemental_dimensions_by_group: Mapping[
        str, tuple[int, int]
    ] | None = None,
    supplemental_development_costs: Mapping[
        GenericDomain, tuple[CandidatePointCost, ...]
    ] | None = None,
) -> CPUDecodeSealedRouteProbe:
    """Create a fresh candidate pool after the generic policy is immutable."""

    if (
        not rules
        or not frozen_generic_policy_digest.startswith("sha256:")
        or not sealed_build_id.startswith("sha256:")
    ):
        raise ValueError("CPU decode sealing requires a frozen generic policy")
    dimensions_by_group: dict[str, tuple[int, int]] = {}
    mutable_development_rows_by_group: dict[str, list[object]] = defaultdict(
        list
    )
    for row in development:
        dimensions = (row.aggregate_n, row.k)
        previous = dimensions_by_group.setdefault(row.shape_group_id, dimensions)
        if previous != dimensions:
            raise ValueError("CPU decode shape group changed dimensions")
        mutable_development_rows_by_group[row.shape_group_id].append(row)
    for shape_group_id, dimensions in (
        supplemental_dimensions_by_group or {}
    ).items():
        previous = dimensions_by_group.setdefault(shape_group_id, dimensions)
        if previous != dimensions:
            raise ValueError(
                "CPU decode supplemental shape group changed dimensions"
            )
    forbidden = frozenset({
        *dimensions_by_group.values(),
        *((shape.n, shape.k) for shape in manifest.shapes),
    })
    supplemental_costs = tuple(
        cost
        for domain_costs in (supplemental_development_costs or {}).values()
        for cost in domain_costs
    )
    mutable_supplemental_costs_by_group: dict[
        str, list[CandidatePointCost]
    ] = defaultdict(list)
    for cost in supplemental_costs:
        mutable_supplemental_costs_by_group[cost.shape_group_id].append(cost)

    development_rows_by_group = {
        group: tuple(rows)
        for group, rows in mutable_development_rows_by_group.items()
    }
    supplemental_costs_by_group = {
        group: tuple(costs)
        for group, costs in mutable_supplemental_costs_by_group.items()
    }
    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
        str(_physical_core_worker_count()),
    ))
    if requested_workers < 1:
        raise ValueError("CPU decode reserve worker count must be positive")
    worker_count = min(
        requested_workers,
        _physical_core_worker_count(),
        len(rules),
    )

    global _PARALLEL_PROBE_RULES
    global _PARALLEL_PROBE_DIMENSIONS_BY_GROUP
    global _PARALLEL_PROBE_FORBIDDEN
    global _PARALLEL_PROBE_MAXIMUM_WEIGHT_ELEMENTS
    global _PARALLEL_PROBE_DEVELOPMENT_ROWS_BY_GROUP
    global _PARALLEL_PROBE_SUPPLEMENTAL_COSTS_BY_GROUP
    _PARALLEL_PROBE_RULES = rules
    _PARALLEL_PROBE_DIMENSIONS_BY_GROUP = dimensions_by_group
    _PARALLEL_PROBE_FORBIDDEN = forbidden
    _PARALLEL_PROBE_MAXIMUM_WEIGHT_ELEMENTS = (
        manifest.maximum_cpu_measurement_weight_elements
    )
    _PARALLEL_PROBE_DEVELOPMENT_ROWS_BY_GROUP = development_rows_by_group
    _PARALLEL_PROBE_SUPPLEMENTAL_COSTS_BY_GROUP = supplemental_costs_by_group
    try:
        # A tiny unit fixture is faster in-process. Production policies own
        # hundreds of independent leaves and therefore use every available
        # physical core without scheduling duplicate SMT workers.
        if worker_count > 1 and len(rules) >= 8:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                candidates_by_rule = tuple(executor.map(
                    _fresh_cpu_decode_probe_geometries_at,
                    range(len(rules)),
                ))
        else:
            candidates_by_rule = tuple(
                _fresh_cpu_decode_probe_geometries_at(rule_index)
                for rule_index in range(len(rules))
            )
    finally:
        _PARALLEL_PROBE_RULES = ()
        _PARALLEL_PROBE_DIMENSIONS_BY_GROUP = {}
        _PARALLEL_PROBE_FORBIDDEN = frozenset()
        _PARALLEL_PROBE_MAXIMUM_WEIGHT_ELEMENTS = 0
        _PARALLEL_PROBE_DEVELOPMENT_ROWS_BY_GROUP = {}
        _PARALLEL_PROBE_SUPPLEMENTAL_COSTS_BY_GROUP = {}

    selected = {
        dimensions
        for candidates in candidates_by_rule
        for dimensions in candidates
    }
    shapes = tuple(
        CPUDecodeSealedProbeShape(_fresh_shape_name(n, k), n, k)
        for n, k in sorted(selected, key=lambda item: (item[0] * item[1], item))
    )
    return CPUDecodeSealedRouteProbe(
        schema_version=CPU_DECODE_SEALED_PROBE_SCHEMA,
        frozen_generic_policy_digest=frozen_generic_policy_digest,
        development_corpus_digest=development.digest(),
        sealed_build_id=sealed_build_id,
        reserve_commitment=cpu_decode_sealed_reserve_commitment(manifest),
        shapes=shapes,
    )


def write_cpu_decode_sealed_route_probe(
    path: Path,
    probe: CPUDecodeSealedRouteProbe,
) -> None:
    """Atomically publish a route-probe request before any sealed timing."""

    write_json(path, probe.canonical_mapping())


def read_cpu_decode_sealed_route_probe(path: Path) -> CPUDecodeSealedRouteProbe:
    """Read and strictly validate one timing-free route-probe request."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "frozen_generic_policy_digest",
        "development_corpus_digest",
        "sealed_build_id",
        "reserve_commitment",
        "shapes",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("CPU decode sealed route-probe fields are invalid")
    shapes = tuple(
        CPUDecodeSealedProbeShape(
            name=str(item["name"]), n=int(item["n"]), k=int(item["k"])
        )
        for item in raw["shapes"]
    )
    probe = CPUDecodeSealedRouteProbe(
        schema_version=str(raw["schema_version"]),
        frozen_generic_policy_digest=str(raw["frozen_generic_policy_digest"]),
        development_corpus_digest=str(raw["development_corpus_digest"]),
        sealed_build_id=str(raw["sealed_build_id"]),
        reserve_commitment=str(raw["reserve_commitment"]),
        shapes=shapes,
    )
    if (
        probe.schema_version != CPU_DECODE_SEALED_PROBE_SCHEMA
        or not probe.sealed_build_id.startswith("sha256:")
        or not shapes
    ):
        raise ValueError("unsupported or empty CPU decode sealed route probe")
    identities = {(shape.name, shape.n, shape.k) for shape in shapes}
    if len(identities) != len(shapes) or any(
        not shape.name.startswith(CPU_DECODE_SEALED_GEOMETRY_PREFIX)
        or shape.n <= 0
        or shape.k <= 0
        or shape.k % 32 != 0
        for shape in shapes
    ):
        raise ValueError("CPU decode route probe contains invalid geometry")
    return probe


def _request_id(fields: Mapping[str, object], plan_seed: str) -> str:
    """Bind one physical challenger edge to its complete sealed generation."""

    return "nvnni-seal-" + _sha256_json({
        "fields": dict(fields), "plan_seed": plan_seed
    }).removeprefix("sha256:")[:24]


def _forceable_candidates(
    n: int,
    k: int,
    k_tiles: int,
    threads: int,
) -> tuple[str, ...]:
    """Return distinct economical schedules at one complete CPU geometry."""

    result = []
    for candidate in cpu_native_vnni_decode_registry().entries:
        chunks = int(candidate.config_json["n_block_chunks"])
        effective = resolve_cpu_native_vnni_decode_n_block_chunks(
            chunks,
            n=n,
            k=k,
            k_tiles=k_tiles,
            threads=threads,
        )
        if chunks == effective:
            result.append(candidate.candidate_id)
    if not result:
        raise ValueError(
            f"N={n} K={k} k_tiles={k_tiles} threads={threads} has no "
            "forceable CPU decode schedule"
        )
    return tuple(result)


def _cpu_decode_sealed_witness_at(
    rule_index: int,
) -> CPUDecodeSealedRuleWitness:
    """Select one fresh route for a rule from process-inherited indexes."""

    rule = _PARALLEL_PLAN_RULES[rule_index]
    regime, threads = _rule_regime(rule)
    domain_key = (
        rule.domain.runtime_codebook_id,
        regime,
        threads,
        rule.domain.bundle_signature,
    )
    eligible = []
    for route in _PARALLEL_PLAN_ROUTES_BY_DOMAIN.get(domain_key, ()):
        forceable = _PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY[
            (route.n, route.k, route.k_tiles, route.threads)
        ]
        if (
            len(forceable) >= 2
            and rule.candidate_id in forceable
            and rule.matches(route.n, route.k, route.k_tiles)
        ):
            eligible.append(route)
    if not eligible:
        raise ValueError(
            "CPU decode route probe cannot exercise frozen rule "
            f"index={rule_index} codebook={rule.domain.runtime_codebook_id}"
        )
    selected_route = min(
        eligible,
        key=lambda route: (
            route.n * route.k,
            route.n,
            route.k,
            route.shape_name,
        ),
    )
    forceable = _PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY[
        (
            selected_route.n,
            selected_route.k,
            selected_route.k_tiles,
            selected_route.threads,
        )
    ]
    return CPUDecodeSealedRuleWitness(
        rule_index=rule_index,
        shape=selected_route.shape_name,
        n=selected_route.n,
        k=selected_route.k,
        k_tiles=selected_route.k_tiles,
        architecture_class=rule.domain.architecture_class,
        runtime_codebook=rule.domain.runtime_codebook_id,
        bundle_signature=rule.domain.bundle_signature,
        selected_candidate_id=rule.candidate_id,
        forceable_candidate_ids=forceable,
    )


def build_cpu_decode_sealed_plan(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    probe: CPUDecodeSealedRouteProbe,
    route_manifest_paths: Iterable[Path],
    sealed_build_id: str,
) -> CPUDecodeSealedPlan:
    """Select one fresh route per frozen leaf and enumerate all challengers."""

    # Authenticate every immutable plan input exactly once.  In production the
    # development corpus contains hundreds of thousands of observations, so
    # recomputing its canonical digest at each use would turn plan assembly into
    # repeated whole-corpus preprocessing.  The local values below are also the
    # sole identities used to seed and publish the plan, making it impossible
    # for validation and publication to accidentally hash different snapshots.
    development_corpus_digest = development.digest()
    route_probe_digest = probe.digest
    candidate_registry_digest = cpu_native_vnni_decode_registry().digest()
    format_registry_digest_value = registry_digest()

    if probe.frozen_generic_policy_digest != frozen_generic_policy_digest:
        raise ValueError("CPU decode route probe belongs to another frozen policy")
    if probe.development_corpus_digest != development_corpus_digest:
        raise ValueError("CPU decode development corpus changed after route probe")
    if probe.sealed_build_id != sealed_build_id:
        raise ValueError("CPU decode route probe belongs to another sealed build")
    if probe.reserve_commitment != cpu_decode_sealed_reserve_commitment(manifest):
        raise ValueError("CPU decode sealed reserve commitment changed")
    routes = read_cpu_prefill_route_manifests(route_manifest_paths)
    probe_names = {shape.name for shape in probe.shapes}
    route_candidates = tuple(
        route for route in routes.routes() if route.shape_name in probe_names
    )
    if not route_candidates:
        raise ValueError("CPU decode route manifests omit generated probe shapes")
    route_manifest_digest = routes.digest()

    mutable_routes_by_domain: dict[
        tuple[int, str, int, str], list[CPUPrefillSerialRoute]
    ] = defaultdict(list)
    forceable_by_geometry: dict[
        tuple[int, int, int, int], tuple[str, ...]
    ] = {}
    for route in route_candidates:
        mutable_routes_by_domain[
            (
                route.execution_codebook,
                route.isa_regime,
                route.threads,
                _decode_bundle(route),
            )
        ].append(route)
        geometry = (route.n, route.k, route.k_tiles, route.threads)
        if geometry not in forceable_by_geometry:
            forceable_by_geometry[geometry] = _forceable_candidates(*geometry)
    routes_by_domain = {
        domain: tuple(routes_for_domain)
        for domain, routes_for_domain in mutable_routes_by_domain.items()
    }

    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_WORKERS",
        str(_physical_core_worker_count()),
    ))
    if requested_workers < 1:
        raise ValueError("CPU decode seal planner worker count must be positive")
    worker_count = min(
        requested_workers,
        _physical_core_worker_count(),
        len(rules),
    )

    global _PARALLEL_PLAN_RULES
    global _PARALLEL_PLAN_ROUTES_BY_DOMAIN
    global _PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY
    _PARALLEL_PLAN_RULES = rules
    _PARALLEL_PLAN_ROUTES_BY_DOMAIN = routes_by_domain
    _PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY = forceable_by_geometry
    try:
        if worker_count > 1 and len(rules) >= 8:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                witnesses = tuple(executor.map(
                    _cpu_decode_sealed_witness_at,
                    range(len(rules)),
                ))
        else:
            witnesses = tuple(
                _cpu_decode_sealed_witness_at(rule_index)
                for rule_index in range(len(rules))
            )
    finally:
        _PARALLEL_PLAN_RULES = ()
        _PARALLEL_PLAN_ROUTES_BY_DOMAIN = {}
        _PARALLEL_PLAN_FORCEABLE_BY_GEOMETRY = {}

    seed_payload = {
        "schema_version": CPU_DECODE_SEALED_PLAN_SCHEMA,
        "frozen_generic_policy_digest": frozen_generic_policy_digest,
        "development_corpus_digest": development_corpus_digest,
        "sealed_build_id": sealed_build_id,
        "reserve_commitment": probe.reserve_commitment,
        "route_probe_digest": route_probe_digest,
        "route_manifest_digest": route_manifest_digest,
        "candidate_registry_digest": candidate_registry_digest,
        "format_registry_digest": format_registry_digest_value,
        "rule_witnesses": [asdict(item) for item in witnesses],
    }
    plan_digest = _sha256_json(seed_payload)
    requests = []
    for witness in witnesses:
        rule = rules[witness.rule_index]
        for source_format in runtime_aliases("cpu", witness.runtime_codebook):
            spec = format_spec(source_format)
            for challenger in witness.forceable_candidate_ids:
                if challenger == witness.selected_candidate_id:
                    continue
                fields = {
                    "rule_index": witness.rule_index,
                    "source_format": source_format,
                    "architecture_class": witness.architecture_class,
                    "shape": witness.shape,
                    "n": witness.n,
                    "k": witness.k,
                    "selected_candidate_id": witness.selected_candidate_id,
                    "exact_candidate_id": challenger,
                }
                request = PairedTimingRequest(
                    request_id=_request_id(fields, plan_digest),
                    source_format=source_format,
                    source_codebook=spec.source_codebook_id,
                    execution_codebook=spec.cpu_execution_codebook_id,
                    architecture_class=witness.architecture_class,
                    shape=witness.shape,
                    shape_group_id=(
                        f"cpu-decode-sealed:{witness.shape}:"
                        f"n{witness.n}:k{witness.k}"
                    ),
                    execution_mode=rule.domain.execution_mode.value,
                    m=1,
                    n=witness.n,
                    k=witness.k,
                    selected_candidate_id=witness.selected_candidate_id,
                    exact_candidate_id=challenger,
                    observed_cv_regret=0.0,
                    reason="sealed_frozen_leaf_exhaustive_challenger",
                )
                requests.append(CPUDecodeSealedRequest(witness.rule_index, request))
    if not requests:
        raise ValueError("CPU decode sealed plan contains no challenger requests")
    return CPUDecodeSealedPlan(
        schema_version=CPU_DECODE_SEALED_PLAN_SCHEMA,
        frozen_generic_policy_digest=frozen_generic_policy_digest,
        development_corpus_digest=development_corpus_digest,
        sealed_build_id=sealed_build_id,
        reserve_commitment=probe.reserve_commitment,
        route_probe_digest=route_probe_digest,
        route_manifest_digest=route_manifest_digest,
        candidate_registry_digest=candidate_registry_digest,
        format_registry_digest=format_registry_digest_value,
        plan_digest=plan_digest,
        rule_witnesses=tuple(witnesses),
        requests=tuple(sorted(requests)),
    )


def write_cpu_decode_sealed_plan(
    path: Path,
    plan: CPUDecodeSealedPlan,
    request_shard_directory: Path,
    *,
    max_requests_per_shard: int = 16,
) -> tuple[Path, ...]:
    """Publish the audit plan plus homogeneous resumable C++ request shards."""

    write_json(path, plan.canonical_mapping())
    return write_cpu_sealed_request_shards(
        request_shard_directory,
        plan.requests,
        plan.development_corpus_digest,
        plan.plan_digest,
        "cpu-decode-sealed-request-shards-v1",
        max_requests_per_shard=max_requests_per_shard,
    )


def read_cpu_decode_sealed_plan(path: Path) -> CPUDecodeSealedPlan:
    """Read a plan and reject stale registry or self-digest identities.

    Version 2 plans predate the exact K-tile witness field.  Their immutable
    payload does, however, authenticate the complete route-manifest digest.
    For that schema only, recover K-tile counts from a retained manifest set
    whose digest matches the plan.  This preserves the original plan and
    request identities while refusing to turn absent launch geometry into a
    fabricated learner feature.
    """

    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version", "frozen_generic_policy_digest",
        "development_corpus_digest", "sealed_build_id", "reserve_commitment",
        "route_probe_digest", "route_manifest_digest",
        "candidate_registry_digest", "format_registry_digest", "plan_digest",
        "rule_witnesses", "requests",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("CPU decode sealed plan fields are invalid")
    schema_version = str(raw["schema_version"])
    if schema_version not in {
        CPU_DECODE_SEALED_PLAN_SCHEMA,
        LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA,
    }:
        raise ValueError("unsupported CPU decode sealed plan schema")

    # Authenticate the exact serialized schema before enriching a legacy
    # witness in memory.  Requests are derived from this digest and therefore
    # cannot participate recursively in its seed.
    seed = dict(raw)
    seed.pop("plan_digest")
    seed.pop("requests")
    if _sha256_json(seed) != str(raw["plan_digest"]):
        raise ValueError("CPU decode sealed plan digest is invalid")

    legacy_k_tiles = (
        _legacy_cpu_decode_k_tiles(path, raw)
        if schema_version == LEGACY_CPU_DECODE_SEALED_PLAN_SCHEMA
        else {}
    )
    witness_fields = {
        "rule_index", "shape", "n", "k", "architecture_class",
        "runtime_codebook", "bundle_signature", "selected_candidate_id",
        "forceable_candidate_ids",
    }
    if schema_version == CPU_DECODE_SEALED_PLAN_SCHEMA:
        witness_fields.add("k_tiles")
    witnesses = tuple(
        CPUDecodeSealedRuleWitness(
            rule_index=int(item["rule_index"]),
            shape=str(item["shape"]),
            n=int(item["n"]),
            k=int(item["k"]),
            k_tiles=(
                int(item["k_tiles"])
                if schema_version == CPU_DECODE_SEALED_PLAN_SCHEMA
                else legacy_k_tiles[int(item["rule_index"])]
            ),
            architecture_class=str(item["architecture_class"]),
            runtime_codebook=int(item["runtime_codebook"]),
            bundle_signature=str(item["bundle_signature"]),
            selected_candidate_id=str(item["selected_candidate_id"]),
            forceable_candidate_ids=tuple(item["forceable_candidate_ids"]),
        )
        for item in raw["rule_witnesses"]
        if _require_exact_fields(
            item, witness_fields, "CPU decode sealed rule witness"
        )
    )
    request_fields = set(PairedTimingRequest.__dataclass_fields__)
    requests = []
    for item in raw["requests"]:
        if set(item) != request_fields | {"rule_index"}:
            raise ValueError("CPU decode sealed request fields are invalid")
        requests.append(CPUDecodeSealedRequest(
            int(item["rule_index"]),
            PairedTimingRequest(**{
                key: item[key] for key in request_fields
            }),
        ))
    plan = CPUDecodeSealedPlan(
        schema_version=schema_version,
        frozen_generic_policy_digest=str(raw["frozen_generic_policy_digest"]),
        development_corpus_digest=str(raw["development_corpus_digest"]),
        sealed_build_id=str(raw["sealed_build_id"]),
        reserve_commitment=str(raw["reserve_commitment"]),
        route_probe_digest=str(raw["route_probe_digest"]),
        route_manifest_digest=str(raw["route_manifest_digest"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        format_registry_digest=str(raw["format_registry_digest"]),
        plan_digest=str(raw["plan_digest"]),
        rule_witnesses=witnesses,
        requests=tuple(requests),
    )
    if not plan.sealed_build_id.startswith("sha256:"):
        raise ValueError("CPU decode sealed plan build identity is invalid")
    if (
        plan.candidate_registry_digest
        != cpu_native_vnni_decode_registry().digest()
    ):
        raise ValueError("CPU decode sealed plan uses a stale candidate registry")
    if plan.format_registry_digest != registry_digest():
        raise ValueError("CPU decode sealed plan uses a stale format registry")
    if tuple(item.rule_index for item in witnesses) != tuple(range(len(witnesses))):
        raise ValueError("CPU decode sealed rule indices are incomplete")
    if len({item.request.request_id for item in requests}) != len(requests):
        raise ValueError("CPU decode sealed plan repeats request IDs")
    return plan


def _require_exact_fields(
    value: object,
    expected: set[str],
    description: str,
) -> bool:
    """Validate one nested JSON record while remaining expression-friendly."""

    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{description} fields are invalid")
    return True


def _legacy_cpu_decode_k_tiles(
    plan_path: Path,
    raw_plan: Mapping[str, object],
) -> dict[int, int]:
    """Recover v2 K-tile counts from its authenticated route transaction.

    Legacy refreshes retained route manifests in the generation root while the
    burned plan and paired requests moved into a child directory.  Candidate
    route generations are content-addressed by a final twelve-hex suffix.  A
    generation is admissible only when its complete canonical route digest is
    the digest committed by the immutable v2 plan.
    """

    plan_name_suffix = ".sealed-plan.json"
    if not plan_path.name.endswith(plan_name_suffix):
        raise ValueError(
            "legacy CPU decode sealed plan has no discoverable route prefix"
        )
    prefix = plan_path.name.removesuffix(plan_name_suffix)
    route_pattern = re.compile(
        rf"^{re.escape(prefix)}\.sealed-route\..*\.([0-9a-f]{{12}})\.csv$"
    )
    generations: dict[str, list[Path]] = defaultdict(list)
    for directory in (plan_path.parent, plan_path.parent.parent):
        for candidate in directory.glob(f"{prefix}.sealed-route.*.csv"):
            match = route_pattern.match(candidate.name)
            if match is not None:
                generations[match.group(1)].append(candidate)

    expected_digest = str(raw_plan["route_manifest_digest"])
    routes = None
    for generation in sorted(generations):
        candidate_routes = read_cpu_prefill_route_manifests(
            sorted(generations[generation])
        )
        if candidate_routes.digest() == expected_digest:
            routes = candidate_routes
            break
    if routes is None:
        raise ValueError(
            "legacy CPU decode sealed plan requires its authenticated route "
            f"manifests: digest={expected_digest}"
        )

    result = {}
    for item in raw_plan["rule_witnesses"]:
        if not isinstance(item, dict):
            raise ValueError("CPU decode sealed rule witness fields are invalid")
        architecture = str(item["architecture_class"])
        parts = architecture.rsplit("|", 3)
        if len(parts) != 4:
            raise ValueError("legacy CPU decode witness architecture is malformed")
        regime = (
            f"{parts[1].removeprefix('build=').lower()}-build."
            f"{parts[2].removeprefix('runtime=').lower()}-runtime"
        )
        route = routes.route_for(
            int(item["runtime_codebook"]), str(item["shape"]), regime
        )
        if (
            (route.n, route.k) != (int(item["n"]), int(item["k"]))
            or _decode_bundle(route) != str(item["bundle_signature"])
        ):
            raise ValueError(
                "legacy CPU decode witness disagrees with authenticated route"
            )
        result[int(item["rule_index"])] = route.k_tiles
    return result


def validate_cpu_decode_sealed_plan(
    plan: CPUDecodeSealedPlan,
    rules: tuple[GenericDispatchRule, ...],
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    sealed_build_id: str,
    supplemental_dimensions_by_group: Mapping[
        str, tuple[int, int]
    ] | None = None,
) -> None:
    """Replay every frozen leaf, fresh-geometry, alias, and challenger invariant."""

    if plan.frozen_generic_policy_digest == "" or len(plan.rule_witnesses) != len(rules):
        raise ValueError("CPU decode sealed plan does not cover every frozen rule")
    if plan.development_corpus_digest != development.digest():
        raise ValueError("CPU decode sealed plan development digest changed")
    if plan.sealed_build_id != sealed_build_id:
        raise ValueError("CPU decode sealed plan belongs to another build")
    if plan.reserve_commitment != cpu_decode_sealed_reserve_commitment(manifest):
        raise ValueError("CPU decode sealed reserve commitment changed")
    development_dimensions = {
        *((row.aggregate_n, row.k) for row in development),
        *(supplemental_dimensions_by_group or {}).values(),
    }
    static_dimensions = {(shape.n, shape.k) for shape in manifest.shapes}
    requests_by_rule: dict[int, list[PairedTimingRequest]] = defaultdict(list)
    for item in plan.requests:
        requests_by_rule[item.rule_index].append(item.request)
    for index, (rule, witness) in enumerate(zip(rules, plan.rule_witnesses, strict=True)):
        if (
            witness.rule_index != index
            or witness.k_tiles < 0
            or not rule.matches(witness.n, witness.k, witness.k_tiles)
        ):
            raise ValueError(f"CPU decode sealed witness does not exercise rule {index}")
        if (
            witness.architecture_class != rule.domain.architecture_class
            or witness.runtime_codebook != rule.domain.runtime_codebook_id
            or witness.bundle_signature != rule.domain.bundle_signature
            or witness.selected_candidate_id != rule.candidate_id
        ):
            raise ValueError(f"CPU decode sealed witness changed rule {index}")
        if (witness.n, witness.k) in development_dimensions | static_dimensions:
            raise ValueError("CPU decode sealed witness geometry was visible to fitting")
        _, threads = _rule_regime(rule)
        if tuple(witness.forceable_candidate_ids) != _forceable_candidates(
            witness.n, witness.k, witness.k_tiles, threads
        ):
            raise ValueError("CPU decode sealed forceable candidate set changed")
        expected = {
            (source_format, challenger)
            for source_format in runtime_aliases("cpu", witness.runtime_codebook)
            for challenger in witness.forceable_candidate_ids
            if challenger != witness.selected_candidate_id
        }
        observed = {
            (request.source_format, request.exact_candidate_id)
            for request in requests_by_rule[index]
        }
        if observed != expected:
            raise ValueError(
                f"CPU decode sealed challenger matrix incomplete for rule {index}"
            )


def certify_cpu_decode_sealed_pairs(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    plan: CPUDecodeSealedPlan,
    evidence_paths: Iterable[Path],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    workers: int | None = None,
) -> CertificationReport:
    """Build the final generic-only certificate from fresh paired evidence."""
    return certify_cpu_sealed_pairs(
        rules,
        frozen_generic_policy_digest,
        plan.frozen_generic_policy_digest,
        plan.plan_digest,
        plan.rule_witnesses,
        plan.requests,
        evidence_paths,
        lambda rule: (
            "serial-kpart"
            if rule.domain.bundle_signature == SERIAL_KPART_BUNDLE
            else "serial-full-k"
        ),
        bootstrap_replicates=bootstrap_replicates,
        workers=workers,
    )


def cpu_decode_burned_seal_costs(
    development: ObservationCorpus,
    plan: CPUDecodeSealedPlan,
    evidence_paths: Iterable[Path],
) -> dict[GenericDomain, tuple[CandidatePointCost, ...]]:
    """Promote one inspected M=1 seal to generic-only development evidence.

    The sealed plan records the development generation that selected its old
    policy, but a burned seal is consumed by a *later* generation. That later
    corpus may add ordinary geometry or another burned transaction, so equality
    with ``plan.development_corpus_digest`` would make additive refinement
    impossible. The plan's self digest plus current candidate/format registry
    identities authenticate what was measured; the domain lookup below then
    proves that each historical point still has one compatible generic owner.
    """

    if plan.candidate_registry_digest != cpu_native_vnni_decode_registry().digest():
        raise ValueError("burned CPU decode seal uses a stale candidate registry")
    if plan.format_registry_digest != registry_digest():
        raise ValueError("burned CPU decode seal uses a stale format registry")
    domains = development.generic_domains()

    def domain_for_witness(
        witness: CPUDecodeSealedRuleWitness,
    ) -> GenericDomain:
        matches = tuple(
            domain
            for domain in domains
            if domain.architecture_class == witness.architecture_class
            and domain.runtime_codebook_id == witness.runtime_codebook
            and domain.bundle_signature == witness.bundle_signature
            and domain.m == 1
        )
        if len(matches) != 1:
            raise ValueError(
                "burned CPU decode witness does not identify one generic domain"
            )
        return matches[0]

    return cpu_sealed_development_costs(
        plan.plan_digest,
        plan.rule_witnesses,
        plan.requests,
        evidence_paths,
        domain_for_witness,
        lambda domain: (
            "serial-kpart"
            if domain.bundle_signature == SERIAL_KPART_BUNDLE
            else "serial-full-k"
        ),
    )
