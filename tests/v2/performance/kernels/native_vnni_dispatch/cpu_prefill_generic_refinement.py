"""Typed boundary-refinement plans for CPU NativeVNNI prefill dispatch.

The generic learner may reject a policy domain when leave-one-shape-out cross
validation exposes an unstable launch boundary. Re-running the original corpus
cannot add information at that boundary. This module instead projects every
unpromoted domain onto nearby, non-overlay development geometries whose serial
arithmetic bundle was observed by the production C++ route probe.

The resulting plan is development evidence only. It never reads or selects a
sealed prefill geometry, and it cannot make a policy installable. A later fit
must still produce total generic dispatch, pass grouped CV at five percent,
freeze, and survive a fresh sealed certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_observation import read_observation_csv
from .cpu_prefill_route_manifest import (
    CPUPrefillSerialRoute,
    CPUPrefillSerialRouteManifest,
)
from .cpu_prefill_route_manifest import (
    CPU_PREFILL_ROUTE_BUNDLES,
    read_cpu_prefill_route_manifests,
)
from .cpu_prefill_split_manifest import (
    CPUPrefillSplitManifest,
    load_cpu_prefill_split_manifest,
)
from .cpu_prefill_training_plan import (
    ISA_REGIMES,
    RUNTIME_ISA_BY_REGIME,
    CPUPrefillSourceTrainingRecord,
)
from .format_registry import runtime_aliases
from .prefill_matrix import (
    CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
    PREFILL_M_BUCKETS,
    cpu_prefill_maximum_weight_elements_for_m,
)
from .profiles import MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
from .shape_manifest import (
    ShapePartition,
    ShapeRole,
    load_declared_shape_manifest,
    load_shape_manifest,
)


CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA = (
    "cpu-prefill-generic-refinement-plan-v3"
)
PREVIOUS_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA = (
    "cpu-prefill-generic-refinement-plan-v2"
)
LEGACY_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA = (
    "cpu-prefill-generic-refinement-plan-v1"
)
DEFAULT_BOUNDARY_NEIGHBORS = 4
CPU_PREFILL_GENERIC_REFINEMENT_PROBE_SCHEMA = (
    "cpu-prefill-generic-refinement-probe-v1"
)
REFINEMENT_RING_STEPS = (64, 128, 256, 512, 1024, 2048, 4096)


def _canonical_json(value: object) -> str:
    """Return the stable JSON encoding used for domain identity."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha256_mapping(value: Mapping[str, object]) -> str:
    """Hash one canonical mapping for provenance checks."""

    return "sha256:" + hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _domain_regime(domain: Mapping[str, object]) -> tuple[str, int]:
    """Decode one CPU build/runtime/thread surface from common policy IR."""

    architecture = str(domain["architecture_class"])
    parts = architecture.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError("CPU prefill refinement domain has malformed architecture")
    build = parts[1].removeprefix("build=").lower()
    runtime = parts[2].removeprefix("runtime=").lower()
    threads = int(parts[3].removeprefix("threads="))
    regime = f"{build}-build.{runtime}-runtime"
    if regime not in ISA_REGIMES or threads <= 0:
        raise ValueError(f"CPU prefill refinement domain has invalid ISA {regime}")
    return regime, threads


@dataclass(frozen=True, order=True)
class CPUPrefillGenericRefinementObligation:
    """One unpromoted generic domain and its selected neighbor geometries."""

    runtime_codebook: int
    m: int
    isa_regime: str
    threads: int
    bundle_signature: str
    cv_max_regret: float | None
    worst_n: int | None
    worst_k: int | None
    selected_shapes: tuple[str, ...]


@dataclass(frozen=True, order=True)
class CPUPrefillGenericRefinementShape:
    """One generated, development-only geometry outside the shared catalog."""

    name: str
    n: int
    k: int

    @property
    def work_items(self) -> int:
        """Return the exact matrix-weight element count."""

        return self.n * self.k


@dataclass(frozen=True)
class CPUPrefillGenericRefinementProbePlan:
    """Immutable candidate ring whose C++ serial routes must be observed."""

    schema_version: str
    source_policy_digest: str
    source_promotion_diagnostics_digest: str
    shapes: tuple[CPUPrefillGenericRefinementShape, ...]

    def canonical_mapping(self) -> dict[str, object]:
        """Return the stable JSON consumed by Python and the zero-kernel probe."""

        return {
            "schema_version": self.schema_version,
            "source_policy_digest": self.source_policy_digest,
            "source_promotion_diagnostics_digest": (
                self.source_promotion_diagnostics_digest
            ),
            "shapes": [
                {"name": shape.name, "n": shape.n, "k": shape.k}
                for shape in self.shapes
            ],
        }

    def digest(self) -> str:
        """Return the exact candidate-ring identity embedded in the final plan."""

        return _sha256_mapping(self.canonical_mapping())


@dataclass(frozen=True)
class CPUPrefillGenericRefinementPlan:
    """Immutable development launch inventory derived from one failed fit."""

    schema_version: str
    source_fit_state: str
    source_policy_digest: str
    source_generic_policy_digest: str
    source_development_corpus_digest: str
    source_promotion_diagnostics_digest: str | None
    route_manifest_digest: str
    split_manifest_digest: str
    shape_manifest_digest: str
    obligations: tuple[CPUPrefillGenericRefinementObligation, ...]
    records: tuple[CPUPrefillSourceTrainingRecord, ...]
    probe_plan_digest: str | None = None
    refinement_shapes: tuple[CPUPrefillGenericRefinementShape, ...] = ()
    refinement_routes: tuple[CPUPrefillSerialRoute, ...] = ()

    def canonical_mapping(self) -> dict[str, object]:
        """Return the complete deterministic JSON representation."""

        result = {
            "schema_version": self.schema_version,
            "source_fit_state": self.source_fit_state,
            "source_policy_digest": self.source_policy_digest,
            "source_generic_policy_digest": self.source_generic_policy_digest,
            "source_development_corpus_digest": (
                self.source_development_corpus_digest
            ),
            "route_manifest_digest": self.route_manifest_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "shape_manifest_digest": self.shape_manifest_digest,
            "obligations": [
                {
                    "runtime_codebook": item.runtime_codebook,
                    "m": item.m,
                    "isa_regime": item.isa_regime,
                    "threads": item.threads,
                    "bundle_signature": item.bundle_signature,
                    "cv_max_regret": item.cv_max_regret,
                    "worst_n": item.worst_n,
                    "worst_k": item.worst_k,
                    "selected_shapes": list(item.selected_shapes),
                }
                for item in self.obligations
            ],
            "records": [
                {
                    "source_format": item.source_format,
                    "shape_name": item.shape_name,
                    "n": item.n,
                    "k": item.k,
                    "isa_regime": item.isa_regime,
                    "runtime_isa": item.runtime_isa,
                    "m_values": list(item.m_values),
                }
                for item in self.records
            ],
        }
        if self.schema_version != LEGACY_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA:
            result["source_promotion_diagnostics_digest"] = (
                self.source_promotion_diagnostics_digest
            )
        if self.schema_version == CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA:
            result["probe_plan_digest"] = self.probe_plan_digest
            result["refinement_shapes"] = [
                {"name": shape.name, "n": shape.n, "k": shape.k}
                for shape in self.refinement_shapes
            ]
            result["refinement_routes"] = [
                {
                    "execution_codebook": route.execution_codebook,
                    "shape_name": route.shape_name,
                    "n": route.n,
                    "k": route.k,
                    "isa_regime": route.isa_regime,
                    "payload_bytes": route.payload_bytes,
                    "threads": route.threads,
                    "k_tiles": route.k_tiles,
                    "bundle_signature": route.bundle_signature,
                }
                for route in self.refinement_routes
            ]
        return result

    def digest(self) -> str:
        """Return the content identity used by collection checkpoints."""

        return _sha256_mapping(self.canonical_mapping())


def _policy_payload(
    path: Path,
    *,
    require_failures: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Read one failed development fit without granting installability.

    New transactions use the explicit non-installable diagnostic state. The
    historical ``frozen_development`` state is accepted only as refinement
    input so the already-spent v7 corpus can identify missing domains without
    an identical multi-hour recollection. The output plan remains development
    evidence and has no publication path.
    """

    raw = json.loads(path.read_text(encoding="utf-8"))
    accepted_states = {
        "development_fit_diagnostic_noninstallable",
        "frozen_development",
    }
    if not isinstance(raw, dict) or raw.get("state") not in accepted_states:
        raise ValueError("CPU prefill refinement requires a development diagnostic")
    policy = raw.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("CPU prefill development diagnostic lacks common policy IR")
    if raw.get("policy_digest") != _sha256_mapping(policy):
        raise ValueError("CPU prefill development diagnostic policy digest changed")
    diagnostics = raw.get("promotion_diagnostics")
    diagnostics_digest = raw.get("promotion_diagnostics_digest")
    if diagnostics is not None or diagnostics_digest is not None:
        if not isinstance(diagnostics, list) or not isinstance(
            diagnostics_digest, str
        ):
            raise ValueError(
                "CPU prefill promotion diagnostics provenance is incomplete"
            )
        expected_diagnostics_digest = _sha256_mapping({
            "promotion_diagnostics": diagnostics,
        })
        if diagnostics_digest != expected_diagnostics_digest:
            raise ValueError(
                "CPU prefill promotion diagnostics changed"
            )
    if require_failures and not policy.get("unpromoted_domains"):
        raise ValueError("CPU prefill refinement source has no failed generic domains")
    return raw, policy


def development_fit_unpromoted_domain_count(path: Path) -> int:
    """Return the failed-domain count from one digest-verified fit artifact."""

    _, policy = _policy_payload(path, require_failures=False)
    domains = policy.get("unpromoted_domains")
    if not isinstance(domains, list):
        raise ValueError("CPU prefill development fit lacks unpromoted-domain IR")
    return len(domains)


def _obligation_sort_key(
    item: CPUPrefillGenericRefinementObligation,
) -> tuple[object, ...]:
    """Return an order that remains stable for optional CV diagnostics."""

    return (
        item.runtime_codebook,
        item.m,
        item.isa_regime,
        item.threads,
        item.bundle_signature,
        item.selected_shapes,
    )


def _record_sort_key(
    item: CPUPrefillSourceTrainingRecord,
) -> tuple[object, ...]:
    """Match the work-first launch order emitted by the plan builder."""

    return (
        item.m_values,
        -(item.n * item.k * max(item.m_values)),
        item.shape_name,
        item.source_format,
        item.isa_regime,
    )


def _validation_by_domain(
    policy: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Index typed CV reports by their complete generic domain mapping."""

    result: dict[str, Mapping[str, object]] = {}
    for raw in policy.get("cross_validation", []):
        if not isinstance(raw, dict) or not isinstance(raw.get("domain"), dict):
            raise ValueError("CPU prefill development diagnostic has invalid CV IR")
        key = _canonical_json(raw["domain"])
        if key in result:
            raise ValueError("CPU prefill development diagnostic repeats a CV domain")
        result[key] = raw
    return result


def _promotion_diagnostic_by_domain(
    artifact: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    """Index optional final-fit diagnostics without breaking older plans."""

    raw_diagnostics = artifact.get("promotion_diagnostics", [])
    if not isinstance(raw_diagnostics, list):
        raise ValueError(
            "CPU prefill development diagnostic has invalid promotion diagnostics"
        )
    result: dict[str, Mapping[str, object]] = {}
    for raw in raw_diagnostics:
        if not isinstance(raw, dict) or not isinstance(raw.get("domain"), dict):
            raise ValueError(
                "CPU prefill development diagnostic has an invalid promotion entry"
            )
        key = _canonical_json(raw["domain"])
        if key in result:
            raise ValueError(
                "CPU prefill development diagnostic repeats a promotion domain"
            )
        result[key] = raw
    return result


def _refinement_target(
    domain: Mapping[str, object],
    validation: Mapping[str, object],
    promotion_diagnostic: Mapping[str, object] | None,
) -> tuple[int, int, float]:
    """Resolve the geometry and regret statistic that rejected one domain."""

    if (
        promotion_diagnostic is not None
        and promotion_diagnostic.get("rejection_stage") == "final_fit_p95"
    ):
        source = promotion_diagnostic
        keys = (
            "final_worst_aggregate_n",
            "final_worst_k",
            "final_worst_p95_regret",
        )
        error_message = "CPU prefill final-fit diagnostic lacks target geometry"
    else:
        source = validation
        keys = ("worst_aggregate_n", "worst_k", "max_regret")
        error_message = "CPU prefill CV report lacks target geometry"
    try:
        n = int(source[keys[0]])
        k = int(source[keys[1]])
        regret = float(source[keys[2]])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(error_message) from error
    if n <= 0 or k <= 0 or k % 32 != 0 or regret < 0.0:
        raise ValueError(
            "CPU prefill refinement target must have positive, block-aligned "
            "dimensions and non-negative regret"
        )
    del domain  # The typed signature makes the caller's domain association explicit.
    return n, k, regret


def _generated_ring_shapes(
    *,
    worst_n: int,
    worst_k: int,
    maximum_work_items: int,
    excluded_dimensions: set[tuple[int, int]],
) -> tuple[CPUPrefillGenericRefinementShape, ...]:
    """Generate deterministic widening rings around one failed geometry.

    Axis points identify an N-only or K-only transition. Diagonal points expose
    interactions between row sharing and reduction depth. Rings widen to 4096
    elements so repeated refinement rounds never depend on a finite hand-written
    shape catalog, while the M-specific work envelope still prevents impractical
    long-prefill measurements.
    """

    generated: dict[tuple[int, int], CPUPrefillGenericRefinementShape] = {}
    for step in REFINEMENT_RING_STEPS:
        offsets = (
            (-step, 0),
            (step, 0),
            (0, -step),
            (0, step),
            (-step, -step),
            (-step, step),
            (step, -step),
            (step, step),
        )
        for delta_n, delta_k in offsets:
            n = worst_n + delta_n
            k = worst_k + delta_k
            dimensions = (n, k)
            if (
                n <= 0
                or k <= 0
                or k % 32 != 0
                or n * k > maximum_work_items
                or dimensions in excluded_dimensions
            ):
                continue
            generated.setdefault(
                dimensions,
                CPUPrefillGenericRefinementShape(
                    name=f"CPUPrefillAutoRefine_N{n}_K{k}",
                    n=n,
                    k=k,
                ),
            )
    return tuple(sorted(generated.values()))


def build_cpu_prefill_generic_refinement_probe_plan(
    development_fit_path: Path,
) -> CPUPrefillGenericRefinementProbePlan:
    """Build fresh route-probe candidates for every failed CV domain."""

    artifact, policy = _policy_payload(development_fit_path)
    validations = _validation_by_domain(policy)
    promotion_diagnostics = _promotion_diagnostic_by_domain(artifact)
    unpromoted = policy.get("unpromoted_domains")
    if not isinstance(unpromoted, list) or not unpromoted:
        raise ValueError("CPU prefill development fit has no refinement obligations")

    declared_dimensions = {
        (shape.n, shape.k) for shape in load_shape_manifest().shapes
    }
    shapes_by_dimensions: dict[
        tuple[int, int], CPUPrefillGenericRefinementShape
    ] = {}
    for domain_raw in unpromoted:
        if not isinstance(domain_raw, dict):
            raise ValueError("CPU prefill unpromoted domain is not a mapping")
        validation = validations.get(_canonical_json(domain_raw))
        if validation is None:
            continue
        cells = validation.get("cells")
        if not isinstance(cells, list):
            raise ValueError("CPU prefill CV report lacks measured cells")
        measured = {
            (
                int(cell["runtime_key"]["aggregate_n"]),
                int(cell["runtime_key"]["k"]),
            )
            for cell in cells
            if isinstance(cell, dict)
            and isinstance(cell.get("runtime_key"), dict)
        }
        worst_n, worst_k, _ = _refinement_target(
            domain_raw,
            validation,
            promotion_diagnostics.get(_canonical_json(domain_raw)),
        )
        m = int(domain_raw["m"])
        excluded = declared_dimensions | measured
        for shape in _generated_ring_shapes(
            worst_n=worst_n,
            worst_k=worst_k,
            maximum_work_items=cpu_prefill_maximum_weight_elements_for_m(m),
            excluded_dimensions=excluded,
        ):
            shapes_by_dimensions.setdefault((shape.n, shape.k), shape)

    if not shapes_by_dimensions:
        raise ValueError("CPU prefill refinement probe has no fresh geometries")
    diagnostics_digest = str(
        artifact.get(
            "promotion_diagnostics_digest",
            _sha256_mapping({"promotion_diagnostics": []}),
        )
    )
    return CPUPrefillGenericRefinementProbePlan(
        schema_version=CPU_PREFILL_GENERIC_REFINEMENT_PROBE_SCHEMA,
        source_policy_digest=str(artifact["policy_digest"]),
        source_promotion_diagnostics_digest=diagnostics_digest,
        shapes=tuple(sorted(shapes_by_dimensions.values())),
    )


def write_cpu_prefill_generic_refinement_probe_plan(
    path: Path,
    plan: CPUPrefillGenericRefinementProbePlan,
) -> None:
    """Atomically publish one source-bound zero-kernel probe inventory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.canonical_mapping(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_cpu_prefill_generic_refinement_probe_plan(
    path: Path,
    development_fit_path: Path | None = None,
) -> CPUPrefillGenericRefinementProbePlan:
    """Read, validate, and optionally reauthenticate a generated probe plan."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_policy_digest",
        "source_promotion_diagnostics_digest",
        "shapes",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("CPU prefill refinement probe root schema is invalid")
    if raw["schema_version"] != CPU_PREFILL_GENERIC_REFINEMENT_PROBE_SCHEMA:
        raise ValueError("unsupported CPU prefill refinement probe schema")
    shapes = tuple(
        CPUPrefillGenericRefinementShape(
            name=str(item["name"]),
            n=int(item["n"]),
            k=int(item["k"]),
        )
        for item in raw["shapes"]
    )
    if not shapes or shapes != tuple(sorted(shapes)):
        raise ValueError("CPU prefill refinement probe shapes are not canonical")
    if len({shape.name for shape in shapes}) != len(shapes) or len({
        (shape.n, shape.k) for shape in shapes
    }) != len(shapes):
        raise ValueError("CPU prefill refinement probe shapes are duplicated")
    if any(
        not shape.name.startswith("CPUPrefillAutoRefine_")
        or shape.n <= 0
        or shape.k <= 0
        or shape.k % 32 != 0
        or shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
        for shape in shapes
    ):
        raise ValueError("CPU prefill refinement probe shape is invalid")
    plan = CPUPrefillGenericRefinementProbePlan(
        schema_version=str(raw["schema_version"]),
        source_policy_digest=str(raw["source_policy_digest"]),
        source_promotion_diagnostics_digest=str(
            raw["source_promotion_diagnostics_digest"]
        ),
        shapes=shapes,
    )
    if not plan.source_policy_digest.startswith("sha256:") or not (
        plan.source_promotion_diagnostics_digest.startswith("sha256:")
    ):
        raise ValueError("CPU prefill refinement probe provenance is incomplete")
    if development_fit_path is not None:
        artifact, _ = _policy_payload(development_fit_path)
        expected_diagnostics = str(
            artifact.get(
                "promotion_diagnostics_digest",
                _sha256_mapping({"promotion_diagnostics": []}),
            )
        )
        if (
            plan.source_policy_digest != artifact["policy_digest"]
            or plan.source_promotion_diagnostics_digest != expected_diagnostics
        ):
            raise ValueError("CPU prefill refinement probe source fit changed")
    return plan


def _boundary_neighbors(
    candidates: list[tuple[str, int, int]],
    *,
    worst_n: int,
    worst_k: int,
    count: int,
) -> tuple[str, ...]:
    """Choose distinct near-boundary points on both sides of N and K."""

    def distance(item: tuple[str, int, int]) -> tuple[float, int, int, str]:
        name, n, k = item
        logarithmic = abs(math.log2(n / worst_n)) + abs(math.log2(k / worst_k))
        return logarithmic, abs(n - worst_n) + abs(k - worst_k), n * k, name

    selected: list[tuple[str, int, int]] = []
    for predicate in (
        lambda item: item[1] < worst_n,
        lambda item: item[1] > worst_n,
        lambda item: item[2] < worst_k,
        lambda item: item[2] > worst_k,
    ):
        eligible = [item for item in candidates if predicate(item)]
        if eligible:
            choice = min(eligible, key=distance)
            if choice not in selected:
                selected.append(choice)
    for item in sorted(candidates, key=distance):
        if item not in selected:
            selected.append(item)
        if len(selected) >= count:
            break
    if len(selected) < count:
        raise ValueError(
            "CPU prefill generic refinement lacks enough route-compatible "
            f"boundary neighbors around N={worst_n} K={worst_k}"
        )
    return tuple(item[0] for item in selected[:count])


def _coverage_anchors(
    candidates: list[tuple[str, int, int]],
) -> tuple[str, ...]:
    """Choose low, middle, and high-work groups for a domain lacking CV."""

    ordered = sorted(candidates, key=lambda item: (item[1] * item[2], *item[1:]))
    required = MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
    if len(ordered) < required:
        raise ValueError("CPU prefill generic domain lacks three refinement groups")
    base_work = ordered[0][1] * ordered[0][2]
    selected_items = []
    for multiplier in (1.0, 1.5, 2.0):
        target = base_work * multiplier
        remaining = [item for item in ordered if item not in selected_items]
        selected_items.append(min(
            remaining,
            key=lambda item: (
                abs(math.log2((item[1] * item[2]) / target)),
                item[1] * item[2],
                item[0],
            ),
        ))
    selected = tuple(item[0] for item in selected_items)
    if len(selected) != required:
        raise ValueError("CPU prefill coverage anchors are not independent shapes")
    return selected


def build_cpu_prefill_generic_refinement_plan(
    development_fit_path: Path,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    *,
    boundary_neighbors: int = DEFAULT_BOUNDARY_NEIGHBORS,
    probe_plan: CPUPrefillGenericRefinementProbePlan | None = None,
    probe_route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> CPUPrefillGenericRefinementPlan:
    """Project every unpromoted generic domain onto fresh development points."""

    if boundary_neighbors < MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS:
        raise ValueError("CPU prefill refinement needs at least three neighbors")
    if (probe_plan is None) != (probe_route_manifest is None):
        raise ValueError(
            "CPU prefill refinement probe plan and route manifest are required together"
        )
    artifact, policy = _policy_payload(development_fit_path)
    if probe_plan is not None:
        expected_diagnostics = str(
            artifact.get(
                "promotion_diagnostics_digest",
                _sha256_mapping({"promotion_diagnostics": []}),
            )
        )
        if (
            probe_plan.source_policy_digest != artifact["policy_digest"]
            or probe_plan.source_promotion_diagnostics_digest
            != expected_diagnostics
        ):
            raise ValueError("CPU prefill refinement probe source fit changed")
    validations = _validation_by_domain(policy)
    promotion_diagnostics = _promotion_diagnostic_by_domain(artifact)
    unpromoted = policy.get("unpromoted_domains")
    if not isinstance(unpromoted, list) or not unpromoted:
        raise ValueError("CPU prefill development fit has no refinement obligations")

    shape_manifest = load_shape_manifest()
    sealed_names = split_manifest.shape_names(sealed=True)
    declared_pool = tuple(
        shape
        for shape in shape_manifest.shapes
        if shape.role == ShapeRole.CERTIFICATION
        and not shape.exact_overlay
        and shape.name not in sealed_names
        and shape.prefill_partition != ShapePartition.SEALED
        and shape.work_items
        <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
    )
    generated_pool = probe_plan.shapes if probe_plan is not None else ()
    pool = (*declared_pool, *generated_pool)
    if not pool:
        raise ValueError("CPU prefill generic refinement pool is empty")
    generated_names = {shape.name for shape in generated_pool}
    shapes_by_name = {shape.name: shape for shape in pool}
    if len(shapes_by_name) != len(pool):
        raise ValueError("CPU prefill refinement probe collides with a declared shape")

    def route_for(
        codebook: int,
        shape_name: str,
        regime: str,
    ) -> CPUPrefillSerialRoute:
        """Resolve declared routes separately from generated probe evidence."""

        source = (
            probe_route_manifest
            if shape_name in generated_names
            else route_manifest
        )
        assert source is not None
        route = source.route_for(codebook, shape_name, regime)
        shape = shapes_by_name[shape_name]
        if (route.n, route.k) != (shape.n, shape.k):
            raise ValueError(
                f"{shape_name}: CPU prefill refinement probe route dimensions changed"
            )
        return route

    obligations = []
    grouped_m: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    for domain_raw in unpromoted:
        if not isinstance(domain_raw, dict):
            raise ValueError("CPU prefill unpromoted domain is not a mapping")
        regime, threads = _domain_regime(domain_raw)
        codebook = int(domain_raw["runtime_codebook_id"])
        m = int(domain_raw["m"])
        bundle = str(domain_raw["bundle_signature"])
        maximum_work_items = cpu_prefill_maximum_weight_elements_for_m(m)
        candidates = []
        for shape in pool:
            if shape.work_items > maximum_work_items:
                continue
            route = route_for(codebook, shape.name, regime)
            if route.threads == threads and route.bundle_signature == bundle:
                candidates.append((shape.name, shape.n, shape.k))

        domain_identity = _canonical_json(domain_raw)
        validation = validations.get(domain_identity)
        promotion_diagnostic = promotion_diagnostics.get(domain_identity)
        if validation is None:
            selected_shapes = _coverage_anchors(candidates)
            cv_max_regret = None
            worst_n = None
            worst_k = None
        else:
            cells = validation.get("cells")
            if not isinstance(cells, list):
                raise ValueError("CPU prefill CV report lacks measured cells")
            measured = {
                (
                    int(cell["runtime_key"]["aggregate_n"]),
                    int(cell["runtime_key"]["k"]),
                )
                for cell in cells
                if isinstance(cell, dict)
                and isinstance(cell.get("runtime_key"), dict)
            }
            candidates = [
                item for item in candidates if (item[1], item[2]) not in measured
            ]
            worst_n, worst_k, cv_max_regret = _refinement_target(
                domain_raw,
                validation,
                promotion_diagnostic,
            )
            selected_shapes = _boundary_neighbors(
                candidates,
                worst_n=worst_n,
                worst_k=worst_k,
                count=boundary_neighbors,
            )

        obligations.append(CPUPrefillGenericRefinementObligation(
            runtime_codebook=codebook,
            m=m,
            isa_regime=regime,
            threads=threads,
            bundle_signature=bundle,
            cv_max_regret=cv_max_regret,
            worst_n=worst_n,
            worst_k=worst_k,
            selected_shapes=selected_shapes,
        ))
        source_format = runtime_aliases("cpu", codebook)[0]
        for shape_name in selected_shapes:
            grouped_m[(source_format, shape_name, regime)].add(m)

    records = []
    for (source_format, shape_name, regime), m_values in grouped_m.items():
        shape = shapes_by_name[shape_name]
        records.append(CPUPrefillSourceTrainingRecord(
            source_format=source_format,
            shape_name=shape_name,
            n=shape.n,
            k=shape.k,
            isa_regime=regime,
            runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
            m_values=tuple(sorted(m_values)),
        ))
    records.sort(key=_record_sort_key)

    metadata = policy.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("CPU prefill development diagnostic lacks metadata")
    corpus_digest = str(metadata.get("development_corpus_digest", ""))
    if not corpus_digest.startswith("sha256:"):
        raise ValueError("CPU prefill diagnostic lacks a development corpus digest")
    selected_generated_names = {
        shape_name
        for obligation in obligations
        for shape_name in obligation.selected_shapes
        if shape_name in generated_names
    }
    selected_generated_shapes = tuple(sorted(
        shapes_by_name[name] for name in selected_generated_names
    ))
    selected_generated_routes = tuple(sorted({
        route_for(
            obligation.runtime_codebook,
            shape_name,
            obligation.isa_regime,
        )
        for obligation in obligations
        for shape_name in obligation.selected_shapes
        if shape_name in generated_names
    }))
    schema_version = (
        CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA
        if probe_plan is not None
        else PREVIOUS_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA
    )
    return CPUPrefillGenericRefinementPlan(
        schema_version=schema_version,
        source_fit_state=str(artifact["state"]),
        source_policy_digest=str(artifact["policy_digest"]),
        source_generic_policy_digest=str(
            artifact["frozen_generic_policy_digest"]
        ),
        source_development_corpus_digest=corpus_digest,
        source_promotion_diagnostics_digest=str(
            artifact.get(
                "promotion_diagnostics_digest",
                _sha256_mapping({"promotion_diagnostics": []}),
            )
        ),
        route_manifest_digest=route_manifest.digest(),
        split_manifest_digest=split_manifest.digest(),
        shape_manifest_digest=shape_manifest.digest(),
        obligations=tuple(sorted(obligations, key=_obligation_sort_key)),
        records=tuple(records),
        probe_plan_digest=(probe_plan.digest() if probe_plan is not None else None),
        refinement_shapes=selected_generated_shapes,
        refinement_routes=selected_generated_routes,
    )


def write_cpu_prefill_generic_refinement_plan(
    path: Path,
    plan: CPUPrefillGenericRefinementPlan,
) -> None:
    """Atomically publish one development-only refinement transaction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.canonical_mapping(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_cpu_prefill_generic_refinement_plan(
    path: Path,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
) -> CPUPrefillGenericRefinementPlan:
    """Read and authenticate a prior development refinement transaction."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    legacy_required = {
        "schema_version",
        "source_fit_state",
        "source_policy_digest",
        "source_generic_policy_digest",
        "source_development_corpus_digest",
        "route_manifest_digest",
        "split_manifest_digest",
        "shape_manifest_digest",
        "obligations",
        "records",
    }
    if not isinstance(raw, dict):
        raise ValueError("CPU prefill refinement plan root schema is invalid")
    schema_version = str(raw.get("schema_version", ""))
    required = set(legacy_required)
    if schema_version in {
        CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA,
        PREVIOUS_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA,
    }:
        required.add("source_promotion_diagnostics_digest")
        if schema_version == CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA:
            required.update({
                "probe_plan_digest",
                "refinement_shapes",
                "refinement_routes",
            })
    elif schema_version != LEGACY_CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA:
        raise ValueError("unsupported CPU prefill refinement plan schema")
    if set(raw) != required:
        raise ValueError("CPU prefill refinement plan root schema is invalid")
    obligations = tuple(
        CPUPrefillGenericRefinementObligation(
            runtime_codebook=int(item["runtime_codebook"]),
            m=int(item["m"]),
            isa_regime=str(item["isa_regime"]),
            threads=int(item["threads"]),
            bundle_signature=str(item["bundle_signature"]),
            cv_max_regret=(
                None
                if item["cv_max_regret"] is None
                else float(item["cv_max_regret"])
            ),
            worst_n=None if item["worst_n"] is None else int(item["worst_n"]),
            worst_k=None if item["worst_k"] is None else int(item["worst_k"]),
            selected_shapes=tuple(str(name) for name in item["selected_shapes"]),
        )
        for item in raw["obligations"]
    )
    records = tuple(
        CPUPrefillSourceTrainingRecord(
            source_format=str(item["source_format"]),
            shape_name=str(item["shape_name"]),
            n=int(item["n"]),
            k=int(item["k"]),
            isa_regime=str(item["isa_regime"]),
            runtime_isa=str(item["runtime_isa"]),
            m_values=tuple(int(value) for value in item["m_values"]),
        )
        for item in raw["records"]
    )
    refinement_shapes = tuple(
        CPUPrefillGenericRefinementShape(
            name=str(item["name"]),
            n=int(item["n"]),
            k=int(item["k"]),
        )
        for item in raw.get("refinement_shapes", [])
    )
    refinement_routes = tuple(
        CPUPrefillSerialRoute(
            execution_codebook=int(item["execution_codebook"]),
            shape_name=str(item["shape_name"]),
            n=int(item["n"]),
            k=int(item["k"]),
            isa_regime=str(item["isa_regime"]),
            payload_bytes=int(item["payload_bytes"]),
            threads=int(item["threads"]),
            k_tiles=int(item["k_tiles"]),
            bundle_signature=str(item["bundle_signature"]),
        )
        for item in raw.get("refinement_routes", [])
    )
    plan = CPUPrefillGenericRefinementPlan(
        schema_version=str(raw["schema_version"]),
        source_fit_state=str(raw["source_fit_state"]),
        source_policy_digest=str(raw["source_policy_digest"]),
        source_generic_policy_digest=str(raw["source_generic_policy_digest"]),
        source_development_corpus_digest=str(
            raw["source_development_corpus_digest"]
        ),
        source_promotion_diagnostics_digest=(
            str(raw["source_promotion_diagnostics_digest"])
            if "source_promotion_diagnostics_digest" in raw
            else None
        ),
        route_manifest_digest=str(raw["route_manifest_digest"]),
        split_manifest_digest=str(raw["split_manifest_digest"]),
        shape_manifest_digest=str(raw["shape_manifest_digest"]),
        obligations=obligations,
        records=records,
        probe_plan_digest=(
            str(raw["probe_plan_digest"])
            if "probe_plan_digest" in raw
            else None
        ),
        refinement_shapes=refinement_shapes,
        refinement_routes=refinement_routes,
    )
    if plan.source_fit_state not in {
        "development_fit_diagnostic_noninstallable",
        "frozen_development",
    }:
        raise ValueError("CPU prefill refinement source state is invalid")
    if plan.route_manifest_digest != route_manifest.digest():
        raise ValueError("CPU prefill refinement plan route manifest changed")
    if plan.split_manifest_digest != split_manifest.digest():
        raise ValueError("CPU prefill refinement plan split manifest changed")
    accepted_shape_manifest_digests = {
        load_shape_manifest().digest(),
        load_declared_shape_manifest().digest(),
    }
    if plan.shape_manifest_digest not in accepted_shape_manifest_digests:
        raise ValueError("CPU prefill refinement plan shape manifest changed")
    if not plan.obligations or not plan.records:
        raise ValueError("CPU prefill refinement plan is empty")
    if any(
        not digest.startswith("sha256:")
        for digest in (
            plan.source_policy_digest,
            plan.source_generic_policy_digest,
            plan.source_development_corpus_digest,
            *(
                (plan.source_promotion_diagnostics_digest,)
                if plan.source_promotion_diagnostics_digest is not None
                else ()
            ),
            *((plan.probe_plan_digest,) if plan.probe_plan_digest is not None else ()),
        )
    ):
        raise ValueError("CPU prefill refinement plan provenance is incomplete")
    _validate_refinement_inventory(plan, route_manifest, split_manifest)
    return plan


def validate_cpu_prefill_generic_refinement_plan_source(
    plan: CPUPrefillGenericRefinementPlan,
    development_fit_path: Path,
    development_observations_path: Path,
) -> None:
    """Prove a resumable plan still names its exact failed-fit corpus.

    A plan authenticates its route, split, and shape manifests when it is read,
    but those identities alone do not prove that a caller supplied the corpus
    which produced the failed fit.  Direct collection of an interrupted round
    therefore reopens both immutable source artifacts: the non-installable fit
    diagnostic proves the policy identities, while the adapted observation CSV
    proves the development-corpus identity after the learner's canonical
    aspect-domain collapse.
    """

    artifact, policy = _policy_payload(development_fit_path)
    metadata = policy.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("CPU prefill development diagnostic lacks metadata")
    expected = {
        "source_fit_state": str(artifact["state"]),
        "source_policy_digest": str(artifact["policy_digest"]),
        "source_generic_policy_digest": str(
            artifact["frozen_generic_policy_digest"]
        ),
        "source_development_corpus_digest": str(
            metadata.get("development_corpus_digest", "")
        ),
        "source_promotion_diagnostics_digest": str(
            artifact.get(
                "promotion_diagnostics_digest",
                _sha256_mapping({"promotion_diagnostics": []}),
            )
        ),
    }
    actual = {
        "source_fit_state": plan.source_fit_state,
        "source_policy_digest": plan.source_policy_digest,
        "source_generic_policy_digest": plan.source_generic_policy_digest,
        "source_development_corpus_digest": (
            plan.source_development_corpus_digest
        ),
    }
    if plan.source_promotion_diagnostics_digest is not None:
        actual["source_promotion_diagnostics_digest"] = (
            plan.source_promotion_diagnostics_digest
        )
    else:
        expected.pop("source_promotion_diagnostics_digest")
    mismatches = [
        name for name, value in actual.items() if value != expected[name]
    ]
    if mismatches:
        raise ValueError(
            "CPU prefill refinement plan source fit changed: "
            + ", ".join(sorted(mismatches))
        )

    observations = read_observation_csv((development_observations_path,))
    observed_digest = observations.with_collapsed_aspect_domains().digest()
    if observed_digest != plan.source_development_corpus_digest:
        raise ValueError(
            "CPU prefill refinement plan source observations changed: "
            f"expected {plan.source_development_corpus_digest}, "
            f"observed {observed_digest}"
        )


def _validate_refinement_inventory(
    plan: CPUPrefillGenericRefinementPlan,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
) -> None:
    """Prove records are the exact projection of current route obligations."""

    if tuple(sorted(plan.obligations, key=_obligation_sort_key)) != plan.obligations:
        raise ValueError("CPU prefill refinement obligations are not canonical")
    shape_manifest = load_shape_manifest()
    if plan.schema_version == CPU_PREFILL_GENERIC_REFINEMENT_SCHEMA:
        if plan.refinement_shapes != tuple(sorted(plan.refinement_shapes)):
            raise ValueError("CPU prefill generated refinement shapes are not canonical")
        if plan.refinement_routes != tuple(sorted(plan.refinement_routes)):
            raise ValueError("CPU prefill generated refinement routes are not canonical")
    elif plan.refinement_shapes or plan.refinement_routes or plan.probe_plan_digest:
        raise ValueError("historical CPU prefill refinement plan has v3 fields")
    generated_shapes = {shape.name: shape for shape in plan.refinement_shapes}
    if len(generated_shapes) != len(plan.refinement_shapes) or len({
        (shape.n, shape.k) for shape in plan.refinement_shapes
    }) != len(plan.refinement_shapes):
        raise ValueError("CPU prefill generated refinement shapes are duplicated")
    if any(
        not shape.name.startswith("CPUPrefillAutoRefine_")
        or shape.n <= 0
        or shape.k <= 0
        or shape.k % 32 != 0
        or shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
        for shape in plan.refinement_shapes
    ):
        raise ValueError("CPU prefill generated refinement shape is invalid")
    generated_routes = {
        (route.execution_codebook, route.shape_name, route.isa_regime): route
        for route in plan.refinement_routes
    }
    if len(generated_routes) != len(plan.refinement_routes):
        raise ValueError("CPU prefill generated refinement routes are duplicated")

    def resolve_shape(shape_name: str):
        """Resolve a declared shape or a v3 plan-owned generated geometry."""

        if shape_name in generated_shapes:
            return generated_shapes[shape_name]
        return shape_manifest.by_name(shape_name)

    used_generated_names: set[str] = set()
    used_generated_route_keys: set[tuple[int, str, str]] = set()
    sealed_names = split_manifest.shape_names(sealed=True)
    identities = set()
    grouped_m: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for obligation in plan.obligations:
        identity = (
            obligation.runtime_codebook,
            obligation.m,
            obligation.isa_regime,
            obligation.threads,
            obligation.bundle_signature,
        )
        if identity in identities:
            raise ValueError("CPU prefill refinement obligations are duplicated")
        identities.add(identity)
        if (
            obligation.isa_regime not in ISA_REGIMES
            or obligation.threads <= 0
            or obligation.m not in PREFILL_M_BUCKETS
            or obligation.bundle_signature not in CPU_PREFILL_ROUTE_BUNDLES
        ):
            raise ValueError("CPU prefill refinement obligation is invalid")
        selected = obligation.selected_shapes
        minimum = (
            DEFAULT_BOUNDARY_NEIGHBORS
            if obligation.cv_max_regret is not None
            else MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
        )
        if len(selected) < minimum or len(set(selected)) != len(selected):
            raise ValueError("CPU prefill refinement shape inventory is invalid")
        cv_fields = (
            obligation.cv_max_regret,
            obligation.worst_n,
            obligation.worst_k,
        )
        if any(value is None for value in cv_fields) != all(
            value is None for value in cv_fields
        ):
            raise ValueError("CPU prefill refinement CV provenance is incomplete")
        source_format = runtime_aliases(
            "cpu", obligation.runtime_codebook
        )[0]
        for shape_name in selected:
            shape = resolve_shape(shape_name)
            maximum_work_items = cpu_prefill_maximum_weight_elements_for_m(
                obligation.m
            )
            generated = shape_name in generated_shapes
            declared_ineligible = (
                not generated
                and (
                    shape.role != ShapeRole.CERTIFICATION
                    or shape.exact_overlay
                    or shape.prefill_partition == ShapePartition.SEALED
                )
            )
            if (
                shape_name in sealed_names
                or declared_ineligible
                or shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
                or shape.work_items > maximum_work_items
            ):
                raise ValueError(
                    f"{shape_name}: CPU prefill refinement shape is not eligible"
                )
            route_key = (
                obligation.runtime_codebook,
                shape_name,
                obligation.isa_regime,
            )
            if generated:
                try:
                    route = generated_routes[route_key]
                except KeyError as error:
                    raise ValueError(
                        f"{shape_name}: CPU prefill generated route is missing"
                    ) from error
                used_generated_names.add(shape_name)
                used_generated_route_keys.add(route_key)
            else:
                route = route_manifest.route_for(*route_key)
            if (
                (route.n, route.k) != (shape.n, shape.k)
                or route.threads != obligation.threads
                or route.bundle_signature != obligation.bundle_signature
                or route.payload_bytes <= 0
                or route.k_tiles < 0
                or route.bundle_signature not in CPU_PREFILL_ROUTE_BUNDLES
            ):
                raise ValueError(
                    f"{shape_name}: CPU prefill refinement route changed"
                )
            grouped_m[(source_format, shape_name, obligation.isa_regime)].add(
                obligation.m
            )

    if used_generated_names != set(generated_shapes):
        raise ValueError("CPU prefill generated refinement shape is not selected")
    if used_generated_route_keys != set(generated_routes):
        raise ValueError("CPU prefill generated refinement route is not selected")

    expected_records = []
    for (source_format, shape_name, regime), m_values in grouped_m.items():
        shape = resolve_shape(shape_name)
        expected_records.append(CPUPrefillSourceTrainingRecord(
            source_format=source_format,
            shape_name=shape_name,
            n=shape.n,
            k=shape.k,
            isa_regime=regime,
            runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
            m_values=tuple(sorted(m_values)),
        ))
    expected = tuple(sorted(expected_records, key=_record_sort_key))
    if plan.records != expected:
        raise ValueError(
            "CPU prefill refinement records do not match their obligations"
        )


def _print_records(records: tuple[CPUPrefillSourceTrainingRecord, ...]) -> None:
    """Print shell-safe tab-separated process records for the refresh driver."""

    for record in records:
        print("\t".join((
            record.source_format,
            record.shape_name,
            str(record.n),
            str(record.k),
            record.isa_regime,
            record.runtime_isa,
            ",".join(str(value) for value in record.m_values),
        )))


def main() -> int:
    """Build or read one refinement plan and expose its launch inventory."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--development-fit", type=Path)
    source.add_argument("--probe-development-fit", type=Path)
    source.add_argument("--plan", type=Path)
    source.add_argument("--status", type=Path)
    parser.add_argument(
        "--source-fit",
        type=Path,
        help="authenticate an existing plan against its development fit",
    )
    parser.add_argument(
        "--source-observations",
        type=Path,
        help="authenticate an existing plan against its adapted observations",
    )
    parser.add_argument("--route-manifest", action="append", type=Path, default=[])
    parser.add_argument("--refinement-probe-plan", type=Path)
    parser.add_argument(
        "--refinement-probe-route-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--records", action="store_true")
    action.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    if args.status is not None:
        if (
            args.route_manifest
            or args.output
            or args.records
            or args.summary
            or args.source_fit
            or args.source_observations
            or args.refinement_probe_plan
            or args.refinement_probe_route_manifest
        ):
            parser.error("--status accepts no route, output, or display options")
        print(development_fit_unpromoted_domain_count(args.status))
        return 0
    if args.probe_development_fit is not None:
        if (
            args.route_manifest
            or args.split_manifest
            or args.refinement_probe_plan
            or args.refinement_probe_route_manifest
            or args.source_fit
            or args.source_observations
            or args.records
        ):
            parser.error(
                "--probe-development-fit accepts only --output and --summary"
            )
        if args.output is None:
            parser.error("--probe-development-fit requires --output")
        probe = build_cpu_prefill_generic_refinement_probe_plan(
            args.probe_development_fit
        )
        write_cpu_prefill_generic_refinement_probe_plan(args.output, probe)
        if args.summary:
            print(f"schema={probe.schema_version}")
            print(f"source_policy={probe.source_policy_digest}")
            print(f"probe_shapes={len(probe.shapes)}")
        return 0
    if not args.route_manifest:
        parser.error("plan construction and validation require --route-manifest")

    routes = read_cpu_prefill_route_manifests(args.route_manifest)
    split = (
        load_cpu_prefill_split_manifest(args.split_manifest)
        if args.split_manifest is not None
        else load_cpu_prefill_split_manifest()
    )
    if args.development_fit is not None:
        if args.source_fit is not None or args.source_observations is not None:
            parser.error("source authentication is valid only with --plan")
        if args.output is None:
            parser.error("--development-fit requires --output")
        if (args.refinement_probe_plan is None) != (
            not args.refinement_probe_route_manifest
        ):
            parser.error(
                "--refinement-probe-plan and probe route manifests are required together"
            )
        probe_plan = (
            read_cpu_prefill_generic_refinement_probe_plan(
                args.refinement_probe_plan,
                args.development_fit,
            )
            if args.refinement_probe_plan is not None
            else None
        )
        probe_routes = (
            read_cpu_prefill_route_manifests(
                args.refinement_probe_route_manifest
            )
            if args.refinement_probe_route_manifest
            else None
        )
        plan = build_cpu_prefill_generic_refinement_plan(
            args.development_fit,
            routes,
            split,
            probe_plan=probe_plan,
            probe_route_manifest=probe_routes,
        )
        write_cpu_prefill_generic_refinement_plan(args.output, plan)
    else:
        if args.refinement_probe_plan or args.refinement_probe_route_manifest:
            parser.error("an existing refinement plan embeds its probe evidence")
        if args.output is not None:
            parser.error("--output is valid only while building a plan")
        plan = read_cpu_prefill_generic_refinement_plan(args.plan, routes, split)
        if (args.source_fit is None) != (args.source_observations is None):
            parser.error(
                "--source-fit and --source-observations must be supplied together"
            )
        if args.source_fit is not None:
            validate_cpu_prefill_generic_refinement_plan_source(
                plan,
                args.source_fit,
                args.source_observations,
            )

    if args.records:
        _print_records(plan.records)
    elif args.summary:
        print(f"schema={plan.schema_version}")
        print(f"source_state={plan.source_fit_state}")
        print(f"source_policy={plan.source_policy_digest}")
        print(f"obligations={len(plan.obligations)}")
        print(f"process_jobs={len(plan.records)}")
        print(f"source_cells={sum(len(item.m_values) for item in plan.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
