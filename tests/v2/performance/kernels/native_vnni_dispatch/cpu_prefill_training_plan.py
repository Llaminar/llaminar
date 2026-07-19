"""Two-hour covering plan for CPU NativeVNNI ordinary-prefill timing.

The target runtime surface remains the complete model-tiered matrix declared in
``prefill_matrix``.  Timing every source format at every target point and in
every ISA regime, however, repeats the same feature interactions thousands of
times and cannot satisfy the backend-wide two-hour collection budget.

This module builds a deterministic covering array over the dimensions that can
change dispatch policy:

* every CPU runtime codebook is paired with every production geometry;
* every runtime codebook sees every legal M bucket in every aspect class;
* every production geometry sees every M bucket allowed by its tier;
* every runtime codebook/M/aspect dispatch domain is measured in every ISA
  regime, while every production geometry is also paired with every ISA; and
* source-format aliases of one CPU runtime codebook are co-measured at exactly
  the same selected cells.

The plan therefore retains all pairwise policy features and the critical
codebook/M/aspect interaction without taking their full Cartesian product.
Exhaustive native-byte correctness remains the responsibility of the dedicated
all-format grouped sweep; this plan gathers canonical performance evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .format_registry import FORMAT_SPECS, runtime_aliases
from .cpu_prefill_route_manifest import (
    CPU_PREFILL_KPART_BUNDLE,
    CPU_PREFILL_ROUTE_BUNDLES,
    CPUPrefillSerialRoute,
    CPUPrefillSerialRouteManifest,
    read_cpu_prefill_route_manifests,
)
from .cpu_prefill_split_manifest import CPUPrefillSplitManifest
from .prefill_matrix import (
    PREFILL_M_BUCKETS,
    CPUPrefillMeasurement,
    cpu_prefill_measurements,
)
from .profiles import MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
from .shape_manifest import load_shape_manifest
from .segmented_policy import GenericDispatchRule


CPU_PREFILL_TRAINING_PLAN_VERSION = "cpu-prefill-covering-plan-v11"
CPU_PREFILL_SEALED_WITNESS_PLAN_SCHEMA = (
    "cpu-prefill-sealed-witness-plan-v2"
)
CPU_PREFILL_OPENED_WITNESS_V5_PLAN_SHA256 = (
    "sha256:12bb09457267d7e44b3eb0bff2347577fd46bd55ea574ae87689fa2f1f7b7f28"
)
CPU_PREFILL_OPENED_WITNESS_V5_GENERIC_DIGEST = (
    "sha256:aeb6dcc2eefcb3f739b5c0cb1d620469a9bc99b7caf3eda18077c758a74ecdb0"
)
CPU_PREFILL_OPENED_WITNESS_V5_SPLIT_DIGEST = (
    "sha256:bacab14b8f0a6580e8766dcf5ab58375650dd5e14e663593f5e80560bc15a3a0"
)
CPU_PREFILL_OPENED_WITNESS_V5_ROUTE_DIGEST = (
    "sha256:c348275338490a5bd1ca9abda5f8c562a7b1bdbd485fbba6b555af5b72e15324"
)
CPU_PREFILL_OPENED_WITNESS_V5_STRONG_CSV_SHA256 = (
    "sha256:293c45f4b4da163724995b80a564d7cbdfe09ee392918957fb7e8c1fec195fc5"
)
CPU_PREFILL_OPENED_WITNESS_V5_TIMING_CSV_SHA256 = (
    "sha256:a865e4b90e72f69f29010713f56bcc5678596aa4ce2ecf775610b395b60fcb45"
)
CPU_PREFILL_OPENED_WITNESS_V6_PLAN_SHA256 = (
    "sha256:2414a25cdd458dc757f0e3ed1bcd1409169c147b75be8851db38c49092f45e2c"
)
CPU_PREFILL_OPENED_WITNESS_V6_GENERIC_DIGEST = (
    "sha256:251bdbce0e8c265683aa2d3ca3319c0cc41209a7237ccd7ef355fa59c3ea20fb"
)
CPU_PREFILL_OPENED_WITNESS_V6_SPLIT_DIGEST = (
    "sha256:98cdd447c761a68bda2a396c1698a3fddba11522ad00aefc2dd409ae3587fb05"
)
CPU_PREFILL_OPENED_WITNESS_V6_ROUTE_DIGEST = (
    "sha256:546f35754f2dc9b1b3ce02d4ed52df258792290b1bb152edfd7d0d3f0ad7e502"
)
CPU_PREFILL_OPENED_WITNESS_V6_STRONG_CSV_SHA256 = (
    "sha256:0a76a5f1fb18ce1347bcac1bff2f83a5b40250f62e165d006ef039fd8cba9e5c"
)
CPU_PREFILL_OPENED_WITNESS_V6_TIMING_CSV_SHA256 = (
    "sha256:5f95e5d7a5a02e92e822e49ea63a8a2a9b3184505452ebf1e9abd2c392b5e59f"
)

ISA_REGIMES = (
    "avx2-build.avx2-runtime",
    "avx512-build.avx2-runtime",
    "avx512-build.avx512-runtime",
)

RUNTIME_ISA_BY_REGIME = {
    "avx2-build.avx2-runtime": "avx2",
    "avx512-build.avx2-runtime": "avx2",
    "avx512-build.avx512-runtime": "avx512",
}


@dataclass(frozen=True, order=True)
class CPUPrefillRuntimeTrainingCell:
    """One selected runtime-codebook/shape/M/ISA measurement cell."""

    runtime_codebook: int
    shape_name: str
    m: int
    isa_regime: str


@dataclass(frozen=True, order=True)
class CPUPrefillSourceTrainingRecord:
    """One process job after runtime-codebook aliases and M values are grouped."""

    source_format: str
    shape_name: str
    n: int
    k: int
    isa_regime: str
    runtime_isa: str
    m_values: tuple[int, ...]


@dataclass(frozen=True, order=True)
class CPUPrefillCoalescedSourceTrainingJob:
    """One process job that shares activation fixtures across source formats.

    The C++ trainer accepts a comma-separated format inventory and visits every
    format independently.  Formats may share one process only when geometry,
    ISA, and the ordered M inventory are identical, so coalescing changes
    neither the measured cells nor the cross-rank timing phases.
    """

    source_formats: tuple[str, ...]
    format_token: str
    shape_name: str
    n: int
    k: int
    isa_regime: str
    runtime_isa: str
    m_values: tuple[int, ...]

    @property
    def format_phase_count(self) -> int:
        """Return the number of synchronized format phases in this process."""

        return len(self.source_formats)


_ISA_REGIME_BY_RAW_PAIR = {
    ("AVX2", "AVX2"): "avx2-build.avx2-runtime",
    ("AVX512", "AVX2"): "avx512-build.avx2-runtime",
    ("AVX512", "AVX512"): "avx512-build.avx512-runtime",
}


def _route_mapping(route: CPUPrefillSerialRoute) -> dict[str, object]:
    """Return the complete stable representation of one C++ route witness."""

    return {
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


def _route_from_mapping(raw: dict[str, object]) -> CPUPrefillSerialRoute:
    """Reconstruct and type-check an embedded sealed route witness."""

    expected = {
        "execution_codebook",
        "shape_name",
        "n",
        "k",
        "isa_regime",
        "payload_bytes",
        "threads",
        "k_tiles",
        "bundle_signature",
    }
    if set(raw) != expected:
        raise ValueError("CPU prefill sealed route fields are invalid")
    route = CPUPrefillSerialRoute(
        execution_codebook=int(raw["execution_codebook"]),
        shape_name=str(raw["shape_name"]),
        n=int(raw["n"]),
        k=int(raw["k"]),
        isa_regime=str(raw["isa_regime"]),
        payload_bytes=int(raw["payload_bytes"]),
        threads=int(raw["threads"]),
        k_tiles=int(raw["k_tiles"]),
        bundle_signature=str(raw["bundle_signature"]),
    )
    if (
        not route.shape_name
        or route.n <= 0
        or route.k <= 0
        or route.k % 32 != 0
        or not 0 <= route.execution_codebook <= 255
        or route.payload_bytes <= 0
        or route.threads <= 0
        or route.k_tiles < 0
        or route.bundle_signature not in CPU_PREFILL_ROUTE_BUNDLES
        or route.serial_kpart != (route.k_tiles > 1)
    ):
        raise ValueError("CPU prefill sealed route values are invalid")
    # Reuse the typed manifest for its remaining cross-record invariants.
    CPUPrefillSerialRouteManifest((route,))
    return route


@dataclass(frozen=True, order=True)
class CPUPrefillSealedRuleWitness:
    """One frozen generic leaf and its timing-free sealed geometry witness."""

    rule_index: int
    route: CPUPrefillSerialRoute


@dataclass(frozen=True)
class CPUPrefillSealedWitnessPlan:
    """Immutable post-freeze launch plan over fresh route-probed geometry."""

    schema_version: str
    frozen_generic_policy_digest: str
    split_manifest_digest: str
    route_manifest_digest: str
    candidate_route_manifest_digest: str | None
    rule_witnesses: tuple[CPUPrefillSealedRuleWitness, ...]
    records: tuple[CPUPrefillSourceTrainingRecord, ...]

    def canonical_mapping(self) -> dict[str, object]:
        """Return the complete stable JSON representation of this plan."""

        return {
            "schema_version": self.schema_version,
            "frozen_generic_policy_digest": self.frozen_generic_policy_digest,
            "split_manifest_digest": self.split_manifest_digest,
            "route_manifest_digest": self.route_manifest_digest,
            "candidate_route_manifest_digest": (
                self.candidate_route_manifest_digest
            ),
            "rule_witnesses": [
                {
                    "rule_index": witness.rule_index,
                    "route": _route_mapping(witness.route),
                }
                for witness in self.rule_witnesses
            ],
            "records": [
                {
                    "source_format": record.source_format,
                    "shape_name": record.shape_name,
                    "n": record.n,
                    "k": record.k,
                    "isa_regime": record.isa_regime,
                    "runtime_isa": record.runtime_isa,
                    "m_values": list(record.m_values),
                }
                for record in self.records
            ],
        }


CoverageToken = tuple[object, ...]


@lru_cache(maxsize=1)
def _codebook_aliases() -> dict[int, tuple[str, ...]]:
    """Return the complete ordered CPU source-alias set per runtime codebook."""

    codebooks = sorted({spec.cpu_execution_codebook_id for spec in FORMAT_SPECS})
    return {
        codebook: runtime_aliases("cpu", codebook)
        for codebook in codebooks
    }


def _aspect(measurement: CPUPrefillMeasurement) -> str:
    return measurement.shape.aspect_bucket.value


def _cell_cost(
    runtime_codebook: int,
    measurement: CPUPrefillMeasurement,
    m: int,
) -> float:
    """Estimate source-expanded CPU work for one runtime training cell.

    A logarithmic cost still allowed the solver to repeat 7B/M16384 cells in
    all three ISA regimes.  Canonical timing is approximately proportional to
    M*N*K in that tail, and every source alias is a separate correctness/timing
    launch, so both terms belong in the selection objective.
    """

    baseline_work = 896 * 896 * 64
    work = m * measurement.shape.n * measurement.shape.k
    aliases = len(_codebook_aliases()[runtime_codebook])
    return aliases * (1.0 + float(work) / float(baseline_work))


def _base_tokens(
    runtime_codebook: int,
    measurement: CPUPrefillMeasurement,
    m: int,
) -> frozenset[CoverageToken]:
    """Return non-ISA interactions covered by one candidate cell."""

    return frozenset({
        ("codebook_shape", runtime_codebook, measurement.shape.name),
        ("codebook_m_aspect", runtime_codebook, m, _aspect(measurement)),
        ("shape_m", measurement.shape.name, m),
    })


def _required_base_tokens(
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> set[CoverageToken]:
    required: set[CoverageToken] = set()
    for codebook in codebooks:
        for measurement in measurements:
            required.add(("codebook_shape", codebook, measurement.shape.name))
        for aspect in sorted({_aspect(item) for item in measurements}):
            m_values = {
                m
                for item in measurements
                if _aspect(item) == aspect
                for m in item.m_values
            }
            for m in m_values:
                required.add(("codebook_m_aspect", codebook, m, aspect))
    for measurement in measurements:
        for m in measurement.m_values:
            required.add(("shape_m", measurement.shape.name, m))
    return required


def _select_base_cells(
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> list[tuple[int, CPUPrefillMeasurement, int, float]]:
    """Greedily cover required interactions with deterministic cost tie-breaking."""

    candidates = [
        (
            codebook,
            measurement,
            m,
            _cell_cost(codebook, measurement, m),
            _base_tokens(codebook, measurement, m),
        )
        for codebook in codebooks
        for measurement in measurements
        for m in measurement.m_values
    ]
    uncovered = _required_base_tokens(measurements, codebooks)
    selected: list[tuple[int, CPUPrefillMeasurement, int, float]] = []

    class _DescendingKey:
        """Adapt the original ``max`` ordering to Python's min-heap."""

        __slots__ = ("value",)

        def __init__(
            self,
            value: tuple[float, int, float, int, str, int],
        ) -> None:
            self.value = value

        def __lt__(self, other: "_DescendingKey") -> bool:
            return self.value > other.value

    def selection_key(
        candidate_index: int,
        covered: int,
    ) -> tuple[float, int, float, int, str, int]:
        """Return the unchanged deterministic greedy ordering for one cell."""

        codebook, measurement, m, cost, _ = candidates[candidate_index]
        return (
            covered / cost,
            covered,
            -cost,
            -codebook,
            measurement.shape.name,
            -m,
        )

    # A selected cell removes at most three coverage tokens. Only candidates
    # sharing one of those tokens can change rank, so index those dependencies
    # and lazily invalidate their old heap entries. This preserves the exact
    # greedy result without rescanning thousands of unaffected candidates for
    # every selected cell.
    candidates_by_token: dict[CoverageToken, list[int]] = defaultdict(list)
    versions = [0] * len(candidates)
    heap: list[tuple[_DescendingKey, int, int]] = []
    for candidate_index, candidate in enumerate(candidates):
        for token in candidate[4]:
            candidates_by_token[token].append(candidate_index)
        heapq.heappush(
            heap,
            (
                _DescendingKey(selection_key(candidate_index, len(candidate[4]))),
                candidate_index,
                0,
            ),
        )

    while uncovered:
        while heap:
            _, best_index, version = heapq.heappop(heap)
            if version != versions[best_index]:
                continue
            removed_tokens = tuple(
                token
                for token in candidates[best_index][4]
                if token in uncovered
            )
            if removed_tokens:
                break
        else:
            raise ValueError(f"CPU prefill plan cannot cover {sorted(uncovered)[:1]}")

        best = candidates[best_index]
        selected.append(best[:4])
        uncovered.difference_update(removed_tokens)

        touched = {
            candidate_index
            for token in removed_tokens
            for candidate_index in candidates_by_token[token]
        }
        for candidate_index in touched:
            versions[candidate_index] += 1
            covered = sum(
                token in uncovered for token in candidates[candidate_index][4]
            )
            if covered == 0:
                continue
            heapq.heappush(
                heap,
                (
                    _DescendingKey(selection_key(candidate_index, covered)),
                    candidate_index,
                    versions[candidate_index],
                ),
            )
    return selected


def _regime_tokens(
    runtime_codebook: int,
    measurement: CPUPrefillMeasurement,
    m: int,
    regime: str,
) -> frozenset[CoverageToken]:
    return frozenset({
        (
            "codebook_m_aspect_regime",
            runtime_codebook,
            m,
            _aspect(measurement),
            regime,
        ),
        ("shape_regime", measurement.shape.name, regime),
    })


def _required_regime_tokens(
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> set[CoverageToken]:
    required: set[CoverageToken] = set()
    for regime in ISA_REGIMES:
        for measurement in measurements:
            required.add(("shape_regime", measurement.shape.name, regime))
        for codebook in codebooks:
            for aspect in sorted({_aspect(item) for item in measurements}):
                for m in {
                    value
                    for item in measurements
                    if _aspect(item) == aspect
                    for value in item.m_values
                }:
                    required.add((
                        "codebook_m_aspect_regime",
                        codebook,
                        m,
                        aspect,
                        regime,
                    ))
    return required


def _required_minimum_geometry_anchor_cells(
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> set[CPUPrefillRuntimeTrainingCell]:
    """Anchor every generic policy domain at its smallest production geometry.

    A codebook/M/aspect/ISA witness is not by itself enough to train a generic
    policy.  The cheapest covering-array choice can come from a single large
    model family, in which case grouped cross-validation correctly declines to
    certify extrapolation toward a smaller shape. Requiring the cheapest
    production geometry in every domain supplies a real lower edge. Additional
    cheap shapes are selected separately only where the existing covering plan
    lacks enough shape groups for cross-validation.

    These anchors are deliberately lower-edge focused. Decision-tree leaves are
    exhaustive above their final threshold, while extrapolation below the first
    observed feature value lacks any training evidence.  The production total-
    policy validator remains the final proof over every declared shape.
    """

    required: set[CPUPrefillRuntimeTrainingCell] = set()
    aspects = sorted({item.shape.aspect_bucket for item in measurements})
    for codebook in codebooks:
        for regime in ISA_REGIMES:
            for aspect in aspects:
                m_values = sorted({
                    value
                    for item in measurements
                    if item.shape.aspect_bucket == aspect
                    for value in item.m_values
                })
                for m in m_values:
                    eligible = [
                        item
                        for item in measurements
                        if item.shape.aspect_bucket == aspect
                        and m in item.m_values
                    ]
                    anchor = min(
                        eligible,
                        key=lambda item: (
                            item.shape.n * item.shape.k,
                            item.shape.n,
                            item.shape.k,
                            item.shape.name,
                        ),
                    )
                    required.add(CPUPrefillRuntimeTrainingCell(
                        runtime_codebook=codebook,
                        shape_name=anchor.shape.name,
                        m=m,
                        isa_regime=regime,
                    ))
    return required


def _generic_domain_shape_group_counts(
    cells: Iterable[CPUPrefillRuntimeTrainingCell],
    measurements: tuple[CPUPrefillMeasurement, ...],
) -> Counter[tuple[int, int, str]]:
    """Count shapes in each cross-aspect CPU prefill policy domain."""

    _ = measurements
    groups: dict[tuple[int, int, str], set[str]] = defaultdict(set)
    for cell in cells:
        groups[(
            cell.runtime_codebook,
            cell.m,
            cell.isa_regime,
        )].add(cell.shape_name)
    return Counter({domain: len(shapes) for domain, shapes in groups.items()})


def _validate_serial_route_manifest(
    manifest: CPUPrefillSerialRouteManifest,
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> None:
    """Require C++ route evidence for the complete production target matrix."""

    threads_by_regime: dict[str, int] = {}
    payload_by_codebook: dict[int, int] = {}
    for codebook in codebooks:
        for measurement in measurements:
            for regime in ISA_REGIMES:
                route = manifest.route_for(
                    codebook,
                    measurement.shape.name,
                    regime,
                )
                if (route.n, route.k) != (
                    measurement.shape.n,
                    measurement.shape.k,
                ):
                    raise ValueError(
                        "CPU prefill route dimensions disagree with the "
                        f"canonical shape {measurement.shape.name}"
                    )
                previous_threads = threads_by_regime.setdefault(
                    regime,
                    route.threads,
                )
                if previous_threads != route.threads:
                    raise ValueError(
                        f"CPU prefill route thread width varies in {regime}"
                    )
                previous_payload = payload_by_codebook.setdefault(
                    codebook,
                    route.payload_bytes,
                )
                if previous_payload != route.payload_bytes:
                    raise ValueError(
                        "CPU prefill route payload varies for execution "
                        f"codebook {codebook}"
                    )
    if len(set(threads_by_regime.values())) != 1:
        raise ValueError(
            "CPU prefill route ISA probes used different OpenMP widths"
        )


def validate_cpu_prefill_serial_route_manifest(
    manifest: CPUPrefillSerialRouteManifest,
) -> None:
    """Prove route evidence covers the complete current production matrix.

    Replay recipes call this timing-free validator before loading a checkpoint
    or fitting a policy. Keeping the matrix definition here ensures preflight,
    collection planning, and final totality validation cannot drift into three
    subtly different interpretations of required shapes and codebooks.
    """

    _validate_serial_route_manifest(
        manifest,
        cpu_prefill_measurements(),
        tuple(_codebook_aliases()),
    )


def _required_arithmetic_route_cells(
    initial_cells: Iterable[CPUPrefillRuntimeTrainingCell],
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
    manifest: CPUPrefillSerialRouteManifest,
) -> set[CPUPrefillRuntimeTrainingCell]:
    """Close production-shape coverage in each arithmetic family.

    Full-K and serial-K-part rows do not share candidate families. A generic
    learner therefore needs independent shape groups *inside* one arithmetic
    bundle. This baseline selector measures every production geometry when a
    bundle has fewer than the grouped-CV minimum. Additional non-overlay route
    anchors are selected by
    :func:`cpu_prefill_arithmetic_refinement_source_training_records`; exact
    overlays never satisfy this obligation.
    """

    materialized = set(initial_cells)
    by_name = {item.shape.name: item for item in measurements}
    targets: dict[tuple[int, int, str, str], set[str]] = defaultdict(set)
    selected: dict[tuple[int, int, str, str], set[str]] = defaultdict(set)
    for codebook in codebooks:
        for measurement in measurements:
            for m in measurement.m_values:
                for regime in ISA_REGIMES:
                    bundle = manifest.route_for(
                        codebook,
                        measurement.shape.name,
                        regime,
                    ).bundle_signature
                    targets[(codebook, m, regime, bundle)].add(
                        measurement.shape.name
                    )
    for cell in materialized:
        bundle = manifest.route_for(
            cell.runtime_codebook,
            cell.shape_name,
            cell.isa_regime,
        ).bundle_signature
        selected[(
            cell.runtime_codebook,
            cell.m,
            cell.isa_regime,
            bundle,
        )].add(cell.shape_name)

    required: set[CPUPrefillRuntimeTrainingCell] = set()
    for domain, target_shapes in sorted(targets.items()):
        codebook, m, regime, bundle = domain
        needed = min(
            MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS,
            len(target_shapes),
        )
        missing_count = needed - len(selected[domain])
        if missing_count <= 0:
            continue
        candidates = sorted(
            target_shapes - selected[domain],
            key=lambda shape_name: (
                _cell_cost(codebook, by_name[shape_name], m),
                by_name[shape_name].shape.n * by_name[shape_name].shape.k,
                shape_name,
            ),
        )
        if len(candidates) < missing_count:
            raise ValueError(f"CPU prefill route domain cannot close {domain}")
        for shape_name in candidates[:missing_count]:
            cell = CPUPrefillRuntimeTrainingCell(
                runtime_codebook=codebook,
                shape_name=shape_name,
                m=m,
                isa_regime=regime,
            )
            required.add(cell)
            selected[domain].add(shape_name)
    return required


def cpu_prefill_arithmetic_refinement_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Select real non-overlay shapes that make every route bundle trainable.

    Production geometry can expose an arithmetic bundle at fewer than the
    three independent shape groups required by grouped cross-validation. The
    remedy is measured development evidence, not an exact-only exception. This
    selector consumes the C++-emitted serial route manifest, finds the cheapest
    checked-in prefill-development geometries in the same bundle, and measures
    one canonical source alias for each execution codebook/ISA/M domain.

    The route manifest is authoritative because K-part admission depends on
    the production tile planner, payload width, thread count, and host cache
    regime. No Python copy of that policy is permitted here.
    """

    measurements = cpu_prefill_measurements()
    production_names = {item.shape.name for item in measurements}
    manifest = load_shape_manifest()
    development_candidates = tuple(
        shape
        for shape in manifest.shapes
        if shape.prefill_partition is not None
        and shape.prefill_partition.value == "development"
        and shape.name not in production_names
        and not shape.exact_overlay
    )
    codebooks = tuple(sorted(_codebook_aliases()))
    grouped_m: dict[tuple[int, str, str], set[int]] = defaultdict(set)

    for codebook in codebooks:
        for regime in ISA_REGIMES:
            production_by_bundle: dict[str, set[str]] = defaultdict(set)
            m_by_bundle: dict[str, set[int]] = defaultdict(set)
            for measurement in measurements:
                route = route_manifest.route_for(
                    codebook,
                    measurement.shape.name,
                    regime,
                )
                production_by_bundle[route.bundle_signature].add(
                    measurement.shape.name
                )
                m_by_bundle[route.bundle_signature].update(measurement.m_values)

            for bundle, production_shapes in sorted(
                production_by_bundle.items()
            ):
                missing_count = (
                    MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
                    - len(production_shapes)
                )
                if missing_count <= 0:
                    continue
                eligible = []
                for shape in development_candidates:
                    route = route_manifest.route_for(
                        codebook,
                        shape.name,
                        regime,
                    )
                    if route.bundle_signature == bundle:
                        eligible.append(shape)
                eligible.sort(key=lambda shape: (
                    shape.work_items,
                    shape.n,
                    shape.k,
                    shape.name,
                ))
                if len(eligible) < missing_count:
                    raise ValueError(
                        "CPU prefill arithmetic bundle lacks measured generic "
                        "refinement geometries: "
                        f"codebook={codebook} regime={regime} bundle={bundle} "
                        f"production_groups={len(production_shapes)} "
                        f"refinement_groups={len(eligible)}"
                    )
                for shape in eligible[:missing_count]:
                    grouped_m[(codebook, shape.name, regime)].update(
                        m_by_bundle[bundle]
                    )

    records = []
    for (codebook, shape_name, regime), m_values in grouped_m.items():
        shape = manifest.by_name(shape_name)
        records.append(CPUPrefillSourceTrainingRecord(
            source_format=_codebook_aliases()[codebook][0],
            shape_name=shape_name,
            n=shape.n,
            k=shape.k,
            isa_regime=regime,
            runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
            m_values=tuple(sorted(m_values)),
        ))
    return _order_source_records(records)


def _select_regime_cells(
    base_cells: list[tuple[int, CPUPrefillMeasurement, int, float]],
    measurements: tuple[CPUPrefillMeasurement, ...],
    codebooks: tuple[int, ...],
) -> tuple[CPUPrefillRuntimeTrainingCell, ...]:
    """Add the cheapest cells needed to populate every ISA policy domain.

    One base cell may deliberately be measured in multiple ISA regimes.  That
    duplication is necessary: ISA is part of the production policy domain, so
    separate codebook/M/aspect evidence must exist before the generator can
    emit a total selector for AVX2 and AVX512 execution.
    """

    uncovered = _required_regime_tokens(measurements, codebooks)
    regime_loads: Counter[str] = Counter()
    assigned: list[CPUPrefillRuntimeTrainingCell] = []

    candidates = [
        (
            codebook,
            measurement,
            m,
            cost,
            regime,
            _regime_tokens(codebook, measurement, m, regime),
        )
        for codebook, measurement, m, cost in base_cells
        for regime in ISA_REGIMES
    ]

    # Preserve every cell selected for the non-ISA covering obligations.  The
    # first pass gives each one the regime where it covers the most outstanding
    # policy-domain evidence; the second pass below may repeat a cell in other
    # regimes to make the runtime selector total.
    for codebook, measurement, m, _ in sorted(
        base_cells,
        key=lambda item: (-item[3], item[0], item[1].shape.name, item[2]),
    ):
        regime = max(
            ISA_REGIMES,
            key=lambda candidate: (
                len(_regime_tokens(
                    codebook, measurement, m, candidate
                ) & uncovered),
                -regime_loads[candidate],
                -ISA_REGIMES.index(candidate),
            ),
        )
        assigned.append(CPUPrefillRuntimeTrainingCell(
            runtime_codebook=codebook,
            shape_name=measurement.shape.name,
            m=m,
            isa_regime=regime,
        ))
        regime_loads[regime] += 1
        uncovered.difference_update(
            _regime_tokens(codebook, measurement, m, regime)
        )

    while uncovered:
        eligible = [item for item in candidates if item[5] & uncovered]
        if not eligible:
            raise ValueError(
                "CPU prefill plan cannot populate ISA domains: "
                f"{sorted(uncovered)[:1]}"
            )
        best = max(
            eligible,
            key=lambda item: (
                len(item[5] & uncovered) / item[3],
                len(item[5] & uncovered),
                -regime_loads[item[4]],
                -item[3],
                -item[0],
                item[1].shape.name,
                -item[2],
                -ISA_REGIMES.index(item[4]),
            ),
        )
        codebook, measurement, m, _, regime, tokens = best
        assigned.append(CPUPrefillRuntimeTrainingCell(
            runtime_codebook=codebook,
            shape_name=measurement.shape.name,
            m=m,
            isa_regime=regime,
        ))
        regime_loads[regime] += 1
        uncovered.difference_update(tokens)

    return tuple(sorted(assigned))


@lru_cache(maxsize=None)
def cpu_prefill_runtime_training_cells(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillRuntimeTrainingCell, ...]:
    """Return and validate the canonical runtime-codebook covering array."""

    measurements = cpu_prefill_measurements()
    codebooks = tuple(_codebook_aliases())
    base_cells = _select_base_cells(measurements, codebooks)
    initial_cells = {
        *_select_regime_cells(base_cells, measurements, codebooks),
        *_required_minimum_geometry_anchor_cells(measurements, codebooks),
    }
    if route_manifest is not None:
        validate_cpu_prefill_serial_route_manifest(route_manifest)
        initial_cells.update(_required_arithmetic_route_cells(
            initial_cells,
            measurements,
            codebooks,
            route_manifest,
        ))
    cells = tuple(sorted(initial_cells))
    validate_cpu_prefill_runtime_training_cells(cells, route_manifest)
    return cells


def validate_cpu_prefill_runtime_training_cells(
    cells: Iterable[CPUPrefillRuntimeTrainingCell],
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> None:
    """Prove that a proposed plan satisfies every declared coverage obligation."""

    materialized = tuple(cells)
    measurements = cpu_prefill_measurements()
    by_name = {item.shape.name: item for item in measurements}
    codebooks = tuple(_codebook_aliases())
    actual_base: set[CoverageToken] = set()
    actual_regime: set[CoverageToken] = set()
    for cell in materialized:
        if cell.runtime_codebook not in codebooks:
            raise ValueError(f"unknown CPU runtime codebook {cell.runtime_codebook}")
        if cell.shape_name not in by_name:
            raise ValueError(f"unknown CPU prefill shape {cell.shape_name}")
        if cell.isa_regime not in ISA_REGIMES:
            raise ValueError(f"unknown CPU ISA regime {cell.isa_regime}")
        measurement = by_name[cell.shape_name]
        if cell.m not in measurement.m_values:
            raise ValueError(f"{cell.shape_name}: illegal training M={cell.m}")
        actual_base.update(_base_tokens(cell.runtime_codebook, measurement, cell.m))
        actual_regime.update(_regime_tokens(
            cell.runtime_codebook, measurement, cell.m, cell.isa_regime
        ))
    missing_base = _required_base_tokens(measurements, codebooks) - actual_base
    missing_regime = _required_regime_tokens(measurements, codebooks) - actual_regime
    missing_lower_boundaries = (
        _required_minimum_geometry_anchor_cells(measurements, codebooks)
        - set(materialized)
    )
    domain_counts = _generic_domain_shape_group_counts(materialized, measurements)
    required_cross_aspect_domains = {
        (token[1], token[2], token[4])
        for token in _required_regime_tokens(measurements, codebooks)
        if token[0] == "codebook_m_aspect_regime"
    }
    undersubscribed_domains = sorted(
        (domain, domain_counts[domain])
        for domain in required_cross_aspect_domains
        if domain_counts[domain]
        < MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS
    )
    missing_route_coverage: list[tuple[object, ...]] = []
    if route_manifest is not None:
        _validate_serial_route_manifest(route_manifest, measurements, codebooks)
        route_groups: dict[tuple[int, int, str, str], set[str]] = defaultdict(set)
        route_targets: dict[tuple[int, int, str, str], set[str]] = defaultdict(set)
        for codebook in codebooks:
            for measurement in measurements:
                for m in measurement.m_values:
                    for regime in ISA_REGIMES:
                        bundle = route_manifest.route_for(
                            codebook,
                            measurement.shape.name,
                            regime,
                        ).bundle_signature
                        route_targets[(codebook, m, regime, bundle)].add(
                            measurement.shape.name
                        )
        for cell in materialized:
            bundle = route_manifest.route_for(
                cell.runtime_codebook,
                cell.shape_name,
                cell.isa_regime,
            ).bundle_signature
            route_groups[(
                cell.runtime_codebook,
                cell.m,
                cell.isa_regime,
                bundle,
            )].add(cell.shape_name)
        for domain, target_shapes in sorted(route_targets.items()):
            required_count = min(
                MIN_GENERIC_CROSS_VALIDATION_SHAPE_GROUPS,
                len(target_shapes),
            )
            if len(route_groups[domain]) < required_count:
                missing_route_coverage.append((
                    *domain,
                    len(route_groups[domain]),
                    required_count,
                ))
    if (
        missing_base
        or missing_regime
        or missing_lower_boundaries
        or undersubscribed_domains
        or missing_route_coverage
    ):
        raise ValueError(
            "CPU prefill covering plan is incomplete: "
            f"base={sorted(missing_base)[:3]} "
            f"regime={sorted(missing_regime)[:3]} "
            f"lower_boundaries={sorted(missing_lower_boundaries)[:3]} "
            f"undersubscribed={undersubscribed_domains[:3]} "
            f"arithmetic_routes={missing_route_coverage[:3]}"
        )


@lru_cache(maxsize=None)
def cpu_prefill_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Expand runtime cells and build phase-compatible two-socket job groups.

    Production timing coordinates complete candidate rounds through MPI.  Every
    rank in one MPMD launch must therefore visit the same ordered M inventory.
    Records are first grouped by that inventory, expensive groups are scheduled
    first, and similarly expensive records are adjacent within each group.  The
    refresh driver treats each contiguous group as an independent pairing
    domain, so an odd final record runs in a one-rank MPI world rather than being
    paired with an incompatible next group.
    """

    measurements = {item.shape.name: item for item in cpu_prefill_measurements()}
    grouped: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    runtime_cells = cpu_prefill_runtime_training_cells(route_manifest)
    for cell in runtime_cells:
        for source_format in _codebook_aliases()[cell.runtime_codebook]:
            grouped[(source_format, cell.shape_name, cell.isa_regime)].add(cell.m)

    materialized = tuple(
        CPUPrefillSourceTrainingRecord(
            source_format=source_format,
            shape_name=shape_name,
            n=measurements[shape_name].shape.n,
            k=measurements[shape_name].shape.k,
            isa_regime=regime,
            runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
            m_values=tuple(sorted(m_values)),
        )
        for (source_format, shape_name, regime), m_values in grouped.items()
    )
    records = _order_source_records(materialized)
    validate_source_alias_co_measurement(records, runtime_cells)
    return records


def _materialize_opened_witness_records(
    raw_records: tuple[tuple[str, str, str, str, tuple[int, ...]], ...],
    *,
    version: str,
    expected_record_count: int,
    expected_cell_count: int,
    route_manifest: CPUPrefillSerialRouteManifest | None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Validate one immutable failed-certificate launch inventory.

    Every record is checked against current shape ownership, source-codebook
    normalization, runtime ISA semantics, and the production C++ route.  This
    shared validator lets later attempts add evidence without weakening the
    exact sparse inventory represented by an earlier failed certificate.
    """

    shape_manifest = load_shape_manifest()
    codebook_by_source = {
        spec.label: spec.cpu_execution_codebook_id for spec in FORMAT_SPECS
    }
    records = []
    for source_format, shape_name, regime, runtime_isa, m_values in raw_records:
        shape = shape_manifest.by_name(shape_name)
        if (
            shape.prefill_partition is None
            or shape.prefill_partition.value != "development"
        ):
            raise ValueError(
                f"{shape_name}: opened {version} witness is not development evidence"
            )
        if source_format not in codebook_by_source:
            raise ValueError(
                f"{version}: unknown opened source format {source_format}"
            )
        if (
            regime not in ISA_REGIMES
            or RUNTIME_ISA_BY_REGIME[regime] != runtime_isa
        ):
            raise ValueError(
                f"{shape_name}: opened {version} ISA surface is invalid"
            )
        if (
            not m_values
            or tuple(sorted(set(m_values))) != m_values
            or any(value not in PREFILL_M_BUCKETS for value in m_values)
        ):
            raise ValueError(
                f"{shape_name}: opened {version} M inventory is invalid"
            )
        if route_manifest is not None:
            route = route_manifest.route_for(
                codebook_by_source[source_format],
                shape_name,
                regime,
            )
            if (route.n, route.k) != (shape.n, shape.k):
                raise ValueError(
                    f"{shape_name}/{regime}: opened-{version} route dimensions differ"
                )
        records.append(CPUPrefillSourceTrainingRecord(
            source_format=source_format,
            shape_name=shape_name,
            n=shape.n,
            k=shape.k,
            isa_regime=regime,
            runtime_isa=runtime_isa,
            m_values=m_values,
        ))
    ordered = _order_source_records(records)
    if (
        len(ordered) != expected_record_count
        or sum(len(record.m_values) for record in ordered)
        != expected_cell_count
    ):
        raise ValueError(f"CPU prefill opened-{version} witness inventory changed")
    return ordered


def cpu_prefill_opened_witness_v5_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Return the exact failed-v5 sealed inventory as development evidence.

    The 37 records below are a compact checked-in transcription of the
    authenticated witness plan named by the provenance digests above.  They
    must remain sparse: broadening one record to another M, ISA, codebook, or
    geometry would claim evidence that was never measured.  Source aliases
    remain separate because the v5 certificate scored their worst surface.
    """

    raw_records = (
        ("Q4_0", "V7CPUPrefillSealed_VeryWide_17280x544", "avx512-build.avx2-runtime", "avx2", (16384,)),
        ("Q5_0", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q5_0", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q4_1", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q4_K", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q5_0", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q5_0", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q5_0", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx512-runtime", "avx512", (4096, 8192, 16384)),
        ("Q4_0", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (2048, 4096)),
        ("Q4_0", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (2048, 4096)),
        ("Q4_0", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx512-runtime", "avx512", (256, 1024, 4096)),
        ("Q4_0", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx512-runtime", "avx512", (256, 1024, 4096)),
        ("IQ4_NL", "V7CPUPrefillSealed_Wide_2208x544", "avx512-build.avx512-runtime", "avx512", (8192,)),
        ("IQ4_XS", "V7CPUPrefillSealed_Wide_2208x544", "avx512-build.avx512-runtime", "avx512", (8192,)),
        ("Q5_1", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (1024, 2048)),
        ("Q5_K", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (1024, 2048)),
        ("Q5_1", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (1024, 2048)),
        ("Q5_K", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (1024, 2048)),
        ("IQ4_NL", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("IQ4_XS", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("IQ2_S", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (256, 1024)),
        ("IQ2_S", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (256, 1024)),
        ("Q4_0", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (8192, 16384)),
        ("Q5_0", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx512-runtime", "avx512", (256, 1024, 4096, 8192, 16384)),
        ("IQ4_NL", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("IQ4_XS", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("IQ4_NL", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx512-runtime", "avx512", (2048, 8192)),
        ("IQ4_XS", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx512-runtime", "avx512", (2048, 8192)),
        ("IQ2_XXS", "V7CPUPrefillSealed_VeryWide_5664x352", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_K", "V7CPUPrefillSealed_VeryWide_5664x352", "avx2-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V7CPUPrefillSealed_Wide_2208x544", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V7CPUPrefillSealed_Wide_2208x544", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ2_XXS", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_K", "V7CPUPrefillSealed_Tall_240x416", "avx2-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V7CPUPrefillSealed_Tall_240x416", "avx512-build.avx512-runtime", "avx512", (1024,)),
    )
    return _materialize_opened_witness_records(
        raw_records,
        version="v5",
        expected_record_count=37,
        expected_cell_count=86,
        route_manifest=route_manifest,
    )


def cpu_prefill_opened_witness_v6_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Return the exact failed-v6 sealed inventory as development evidence."""

    raw_records = (
        ("Q4_1", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q4_K", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("IQ4_NL", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("IQ4_XS", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("IQ4_NL", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("IQ4_XS", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx512-runtime", "avx512", (2048,)),
        ("Q5_0", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx2-runtime", "avx2", (1024,)),
        ("Q4_0", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("Q5_0", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (1024,)),
        ("IQ3_XXS", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("Q4_0", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("IQ4_NL", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("IQ4_XS", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("Q4_0", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (8192,)),
        ("IQ2_XXS", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V8CPUPrefillSealed_VeryWide_5792x352", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V8CPUPrefillSealed_Wide_2272x544", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ2_XXS", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q5_0", "V8CPUPrefillSealed_Tall_224x416", "avx512-build.avx512-runtime", "avx512", (256, 1024)),
    )
    return _materialize_opened_witness_records(
        raw_records,
        version="v6",
        expected_record_count=21,
        expected_cell_count=34,
        route_manifest=route_manifest,
    )


def cpu_prefill_opened_witness_v7_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Return exactly the failed-v7 sealed launches as development evidence.

    The v7 witness plan selected only three of its eight committed geometries.
    These 17 sparse process records and 30 source/M cells are transcribed from
    the immutable sealed-v7 CSV; the five geometries that were never launched
    remain eligible for the rotated v8 holdout.
    """

    raw_records = (
        ("IQ2_XXS", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ2_XXS", "V9CPUPrefillSealed_VeryWide_5920x352", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("IQ3_XXS", "V9CPUPrefillSealed_VeryWide_5920x352", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q8_0", "V9CPUPrefillSealed_Wide_2336x544", "avx512-build.avx2-runtime", "avx2", (64,)),
        ("Q5_0", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx512-runtime", "avx512", (256, 1024)),
        ("IQ3_XXS", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("Q4_0", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("Q4_0", "V9CPUPrefillSealed_VeryWide_5920x352", "avx512-build.avx512-runtime", "avx512", (1024,)),
        ("Q5_0", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (1024,)),
        ("Q5_0", "V9CPUPrefillSealed_VeryWide_5920x352", "avx512-build.avx2-runtime", "avx2", (1024,)),
        ("Q4_1", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("Q4_K", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (1024, 2048, 4096, 8192, 16384)),
        ("IQ4_NL", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("IQ4_XS", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (2048, 4096, 8192)),
        ("Q4_0", "V9CPUPrefillSealed_Tall_208x416", "avx512-build.avx2-runtime", "avx2", (8192,)),
    )
    return _materialize_opened_witness_records(
        raw_records,
        version="v7",
        expected_record_count=17,
        expected_cell_count=30,
        route_manifest=route_manifest,
    )


def cpu_prefill_development_refinement_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Return only the post-v3 development enrichment launch inventory.

    The opened v3 holdout remains valuable canonical timing and is crossed with
    every source alias at its already-measured M=64. The focused v4
    neighborhoods are performance-only densification: one canonical source
    alias represents each normalized execution codebook, while the baseline
    covering array continues to prove alias co-measurement. Every canonical M
    bucket is measured at the focused neighbors so leave-one-shape-out CV can
    learn both sides of the dominant launch transitions without rebuilding the
    original two-hour corpus.
    """

    manifest = load_shape_manifest()
    opened_shapes = tuple(
        shape
        for shape in manifest.shapes
        if shape.model_family == "cpu-prefill-sealed-v3"
    )
    focused_shapes = tuple(
        shape
        for shape in manifest.shapes
        if shape.model_family == "cpu-prefill-cv-refinement-v4"
    )
    if len(opened_shapes) != 8 or len(focused_shapes) != 4:
        raise ValueError(
            "CPU prefill v4 development enrichment inventory is incomplete"
        )

    records = []
    for regime in ISA_REGIMES:
        for shape in opened_shapes:
            for spec in FORMAT_SPECS:
                if route_manifest is not None:
                    route = route_manifest.route_for(
                        spec.cpu_execution_codebook_id,
                        shape.name,
                        regime,
                    )
                    if (route.n, route.k) != (shape.n, shape.k):
                        raise ValueError(
                            f"{shape.name}/{regime}: opened route dimensions differ"
                        )
                records.append(CPUPrefillSourceTrainingRecord(
                    source_format=spec.label,
                    shape_name=shape.name,
                    n=shape.n,
                    k=shape.k,
                    isa_regime=regime,
                    runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
                    m_values=(64,),
                ))
        for shape in focused_shapes:
            for codebook, aliases in _codebook_aliases().items():
                if route_manifest is not None:
                    route = route_manifest.route_for(
                        codebook,
                        shape.name,
                        regime,
                    )
                    if (route.n, route.k) != (shape.n, shape.k):
                        raise ValueError(
                            f"{shape.name}/{regime}: refinement route dimensions differ"
                        )
                records.append(CPUPrefillSourceTrainingRecord(
                    source_format=aliases[0],
                    shape_name=shape.name,
                    n=shape.n,
                    k=shape.k,
                    isa_regime=regime,
                    runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
                    m_values=PREFILL_M_BUCKETS,
                ))
    records.extend(
        cpu_prefill_opened_witness_v5_source_training_records(route_manifest)
    )
    records.extend(
        cpu_prefill_opened_witness_v6_source_training_records(route_manifest)
    )
    records.extend(
        cpu_prefill_opened_witness_v7_source_training_records(route_manifest)
    )
    if route_manifest is not None:
        records.extend(
            cpu_prefill_arithmetic_refinement_source_training_records(
                route_manifest
            )
        )
    return _order_source_records(tuple(records))


def _rule_isa_regime(rule: GenericDispatchRule) -> tuple[str, int]:
    """Decode one frozen CPU rule's build/runtime regime and thread width."""

    parts = rule.domain.architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(
            "CPU prefill generic rule has malformed architecture class"
        )
    build = parts[1].removeprefix("build=").lower()
    runtime = parts[2].removeprefix("runtime=").lower()
    threads = int(parts[3].removeprefix("threads="))
    regime = f"{build}-build.{runtime}-runtime"
    if regime not in ISA_REGIMES or threads <= 0:
        raise ValueError(f"CPU prefill rule has unsupported ISA surface {regime}")
    return regime, threads


def build_cpu_prefill_sealed_witness_plan(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    split_manifest: CPUPrefillSplitManifest,
    route_manifest: CPUPrefillSerialRouteManifest,
    *,
    candidate_route_manifest: CPUPrefillSerialRouteManifest | None = None,
    development_dimensions: frozenset[tuple[int, int]] = frozenset(),
) -> CPUPrefillSealedWitnessPlan:
    """Project every frozen leaf onto its cheapest fresh physical witness.

    The original fixed eight-shape pool predated total generic dispatch and
    cannot cover thousands of independently learned leaves.  A route-probe
    transaction therefore supplies fresh, untimed geometries after the generic
    policy is frozen.  Selection remains timing-free: it may inspect only leaf
    predicates, the development geometry inventory, and routes emitted by the
    production C++ tile planner.  The resulting plan embeds the exact selected
    routes, allowing sealed certification to revalidate the choice without
    retaining the much larger candidate pool.
    """

    if not rules:
        raise ValueError("CPU prefill sealed witness plan requires generic rules")
    if not frozen_generic_policy_digest.startswith("sha256:"):
        raise ValueError("CPU prefill witness plan requires a policy digest")

    shape_manifest = load_shape_manifest()
    generated_routes: dict[
        tuple[int, str, str], list[CPUPrefillSerialRoute]
    ] = defaultdict(list)
    if candidate_route_manifest is not None:
        dimensions_by_name: dict[str, tuple[int, int]] = {}
        for route in candidate_route_manifest.routes():
            if not route.shape_name.startswith("CPUPrefillAutoRefine_"):
                continue
            dimensions = (route.n, route.k)
            previous = dimensions_by_name.setdefault(
                route.shape_name,
                dimensions,
            )
            if previous != dimensions:
                raise ValueError(
                    f"{route.shape_name}: sealed candidate dimensions changed"
                )
            if dimensions in development_dimensions:
                continue
            generated_routes[(
                route.execution_codebook,
                route.isa_regime,
                route.bundle_signature,
            )].append(route)

    grouped_m: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    witnesses: list[CPUPrefillSealedRuleWitness] = []
    sealed_m = set(split_manifest.sealed_m_values)
    for rule_index, rule in enumerate(rules):
        regime, threads = _rule_isa_regime(rule)
        if rule.domain.m not in sealed_m:
            raise ValueError(
                f"generic rule M={rule.domain.m} is outside the sealed envelope"
            )
        eligible: list[CPUPrefillSerialRoute] = []
        for shape_name in split_manifest.sealed_shapes:
            shape = shape_manifest.by_name(shape_name)
            route = route_manifest.route_for(
                rule.domain.runtime_codebook_id,
                shape_name,
                regime,
            )
            if (route.n, route.k) != (shape.n, shape.k):
                raise ValueError(
                    f"{shape_name}/{regime}: route dimensions disagree"
                )
            if (
                route.threads == threads
                and route.bundle_signature == rule.domain.bundle_signature
                and rule.matches(route.n, route.k)
            ):
                eligible.append(route)
        for route in generated_routes.get((
            rule.domain.runtime_codebook_id,
            regime,
            rule.domain.bundle_signature,
        ), ()):
            if (
                route.threads == threads
                and rule.matches(route.n, route.k)
            ):
                eligible.append(route)
        if not eligible:
            raise ValueError(
                "CPU prefill sealed geometry cannot exercise frozen rule "
                f"index={rule_index} codebook={rule.domain.runtime_codebook_id} "
                f"M={rule.domain.m}"
            )
        witness = min(
            eligible,
            key=lambda route: (
                route.n * route.k,
                route.n,
                route.k,
                route.shape_name,
            ),
        )
        witnesses.append(CPUPrefillSealedRuleWitness(rule_index, witness))
        grouped_m[(
            witness.shape_name,
            regime,
            rule.domain.runtime_codebook_id,
        )].add(rule.domain.m)

    records = []
    for (shape_name, regime, codebook), m_values in grouped_m.items():
        selected_route = next(
            witness.route
            for witness in witnesses
            if witness.route.shape_name == shape_name
            and witness.route.isa_regime == regime
            and witness.route.execution_codebook == codebook
        )
        for source_format in _codebook_aliases()[codebook]:
            records.append(CPUPrefillSourceTrainingRecord(
                source_format=source_format,
                shape_name=shape_name,
                n=selected_route.n,
                k=selected_route.k,
                isa_regime=regime,
                runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
                m_values=tuple(sorted(m_values)),
            ))

    return CPUPrefillSealedWitnessPlan(
        schema_version=CPU_PREFILL_SEALED_WITNESS_PLAN_SCHEMA,
        frozen_generic_policy_digest=frozen_generic_policy_digest,
        split_manifest_digest=split_manifest.digest(),
        route_manifest_digest=route_manifest.digest(),
        candidate_route_manifest_digest=(
            candidate_route_manifest.digest()
            if candidate_route_manifest is not None
            else None
        ),
        rule_witnesses=tuple(witnesses),
        records=_order_source_records(records),
    )


def write_cpu_prefill_sealed_witness_plan(
    path: Path,
    plan: CPUPrefillSealedWitnessPlan,
) -> None:
    """Atomically publish a frozen-policy-bound sealed launch plan."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.canonical_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_cpu_prefill_sealed_witness_plan(
    path: Path,
) -> CPUPrefillSealedWitnessPlan:
    """Read the strict witness-plan schema used by the refresh driver."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    expected_fields = {
        "schema_version",
        "frozen_generic_policy_digest",
        "split_manifest_digest",
        "route_manifest_digest",
        "candidate_route_manifest_digest",
        "rule_witnesses",
        "records",
    }
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ValueError("CPU prefill sealed witness-plan fields are invalid")
    if raw["schema_version"] != CPU_PREFILL_SEALED_WITNESS_PLAN_SCHEMA:
        raise ValueError("unsupported CPU prefill sealed witness-plan schema")
    witness_fields = {"rule_index", "route"}
    record_fields = {
        "source_format",
        "shape_name",
        "n",
        "k",
        "isa_regime",
        "runtime_isa",
        "m_values",
    }
    if not isinstance(raw["rule_witnesses"], list) or not all(
        isinstance(item, dict)
        and set(item) == witness_fields
        and isinstance(item["route"], dict)
        for item in raw["rule_witnesses"]
    ):
        raise ValueError("CPU prefill sealed witness entries are invalid")
    if not isinstance(raw["records"], list) or not all(
        isinstance(item, dict) and set(item) == record_fields
        for item in raw["records"]
    ):
        raise ValueError("CPU prefill sealed witness records are invalid")
    witnesses = tuple(
        CPUPrefillSealedRuleWitness(
            rule_index=int(item["rule_index"]),
            route=_route_from_mapping(item["route"]),
        )
        for item in raw["rule_witnesses"]
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
    plan = CPUPrefillSealedWitnessPlan(
        schema_version=str(raw["schema_version"]),
        frozen_generic_policy_digest=str(
            raw["frozen_generic_policy_digest"]
        ),
        split_manifest_digest=str(raw["split_manifest_digest"]),
        route_manifest_digest=str(raw["route_manifest_digest"]),
        candidate_route_manifest_digest=(
            str(raw["candidate_route_manifest_digest"])
            if raw["candidate_route_manifest_digest"] is not None
            else None
        ),
        rule_witnesses=witnesses,
        records=records,
    )
    if tuple(witness.rule_index for witness in witnesses) != tuple(
        range(len(witnesses))
    ):
        raise ValueError(
            "CPU prefill sealed witness indices are not canonical and complete"
        )
    if not records:
        raise ValueError("CPU prefill sealed witness records are empty")
    if not all(
        digest.startswith("sha256:")
        for digest in (
            plan.frozen_generic_policy_digest,
            plan.split_manifest_digest,
            plan.route_manifest_digest,
            *(
                (plan.candidate_route_manifest_digest,)
                if plan.candidate_route_manifest_digest is not None
                else ()
            ),
        )
    ):
        raise ValueError("CPU prefill sealed witness digests are invalid")
    shape_manifest = load_shape_manifest()
    declared_names = set(shape_manifest.names())
    witness_dimensions: dict[str, tuple[int, int]] = {}
    for witness in witnesses:
        route = witness.route
        dimensions = (route.n, route.k)
        previous = witness_dimensions.setdefault(route.shape_name, dimensions)
        if previous != dimensions:
            raise ValueError(
                f"{route.shape_name}: sealed witness dimensions changed"
            )
        if route.shape_name in declared_names:
            shape = shape_manifest.by_name(route.shape_name)
            if dimensions != (shape.n, shape.k):
                raise ValueError(
                    f"{route.shape_name}: sealed witness route dimensions disagree"
                )
        elif not route.shape_name.startswith("CPUPrefillAutoRefine_"):
            raise ValueError(
                f"unknown CPU prefill sealed shape {route.shape_name}"
            )
    source_formats = {spec.label for spec in FORMAT_SPECS}
    identities = set()
    for record in records:
        if record.source_format not in source_formats:
            raise ValueError(
                f"unknown CPU prefill sealed source format {record.source_format}"
            )
        if record.shape_name not in witness_dimensions:
            raise ValueError(
                f"{record.shape_name}: sealed record lacks a rule witness"
            )
        if (record.n, record.k) != witness_dimensions[record.shape_name]:
            raise ValueError(
                f"{record.shape_name}: sealed witness dimensions disagree"
            )
        if (
            record.isa_regime not in ISA_REGIMES
            or RUNTIME_ISA_BY_REGIME[record.isa_regime]
            != record.runtime_isa
        ):
            raise ValueError(
                f"{record.shape_name}: sealed witness ISA regime is invalid"
            )
        if (
            not record.m_values
            or tuple(sorted(set(record.m_values))) != record.m_values
            or any(value not in PREFILL_M_BUCKETS for value in record.m_values)
        ):
            raise ValueError(
                f"{record.shape_name}: sealed witness M inventory is invalid"
            )
        identity = (
            record.source_format,
            record.shape_name,
            record.isa_regime,
        )
        if identity in identities:
            raise ValueError("CPU prefill sealed witness records are duplicated")
        identities.add(identity)
    if _order_source_records(records) != records:
        raise ValueError("CPU prefill sealed witness records are not canonical")
    return plan


def validate_cpu_prefill_sealed_witness_plan(
    plan: CPUPrefillSealedWitnessPlan,
    rules: tuple[GenericDispatchRule, ...],
    split_manifest: CPUPrefillSplitManifest,
    route_manifest: CPUPrefillSerialRouteManifest,
    *,
    development_dimensions: frozenset[tuple[int, int]],
) -> None:
    """Prove an embedded witness route exercises each frozen leaf freshly.

    This validator is deliberately independent of the discarded candidate
    pool.  It authenticates the production route for checked-in sealed shapes,
    replays every frozen predicate against generated shapes, rejects any
    geometry visible to development fitting, and reconstructs the exact launch
    records from the embedded C++ route witnesses.
    """

    if plan.frozen_generic_policy_digest == "":
        raise ValueError("CPU prefill sealed plan lacks a frozen policy digest")
    if plan.split_manifest_digest != split_manifest.digest():
        raise ValueError("CPU prefill sealed plan disagrees with split manifest")
    if plan.route_manifest_digest != route_manifest.digest():
        raise ValueError("CPU prefill sealed plan disagrees with production routes")
    if len(plan.rule_witnesses) != len(rules):
        raise ValueError("CPU prefill sealed plan does not cover every frozen rule")

    declared_sealed = set(split_manifest.sealed_shapes)
    generated_seen = False
    grouped_m: dict[tuple[str, str, int], set[int]] = defaultdict(set)
    route_by_record: dict[
        tuple[str, str, int], CPUPrefillSerialRoute
    ] = {}
    for rule_index, (rule, witness) in enumerate(
        zip(rules, plan.rule_witnesses, strict=True)
    ):
        if witness.rule_index != rule_index:
            raise ValueError("CPU prefill sealed rule indices changed")
        route = witness.route
        regime, threads = _rule_isa_regime(rule)
        if (
            route.execution_codebook != rule.domain.runtime_codebook_id
            or route.isa_regime != regime
            or route.threads != threads
            or route.bundle_signature != rule.domain.bundle_signature
            or not rule.matches(route.n, route.k)
        ):
            raise ValueError(
                f"CPU prefill sealed route does not exercise rule {rule_index}"
            )
        if route.shape_name in declared_sealed:
            if route_manifest.route_for(
                route.execution_codebook,
                route.shape_name,
                route.isa_regime,
            ) != route:
                raise ValueError(
                    f"{route.shape_name}: sealed production route changed"
                )
        else:
            generated_seen = True
            if not route.shape_name.startswith("CPUPrefillAutoRefine_"):
                raise ValueError("CPU prefill generated sealed shape is not typed")
            if (route.n, route.k) in development_dimensions:
                raise ValueError(
                    f"{route.shape_name}: sealed geometry was visible to fitting"
                )
        grouped_m[(
            route.shape_name,
            route.isa_regime,
            route.execution_codebook,
        )].add(rule.domain.m)
        route_by_record[(
            route.shape_name,
            route.isa_regime,
            route.execution_codebook,
        )] = route

    if generated_seen and plan.candidate_route_manifest_digest is None:
        raise ValueError("CPU prefill generated seal lacks route-pool provenance")
    expected_records = []
    for key, m_values in grouped_m.items():
        shape_name, regime, codebook = key
        route = route_by_record[key]
        for source_format in _codebook_aliases()[codebook]:
            expected_records.append(CPUPrefillSourceTrainingRecord(
                source_format=source_format,
                shape_name=shape_name,
                n=route.n,
                k=route.k,
                isa_regime=regime,
                runtime_isa=RUNTIME_ISA_BY_REGIME[regime],
                m_values=tuple(sorted(m_values)),
            ))
    if _order_source_records(expected_records) != plan.records:
        raise ValueError("CPU prefill sealed launch records changed")


def validate_cpu_prefill_sealed_witness_plan_file(
    path: Path,
    expected: CPUPrefillSealedWitnessPlan,
) -> None:
    """Require the published witness plan to equal a fresh reconstruction."""

    actual = read_cpu_prefill_sealed_witness_plan(path)
    if actual.canonical_mapping() != expected.canonical_mapping():
        raise ValueError(
            "CPU prefill sealed witness plan changed after generic freeze"
        )


def _source_record_schedule_key(
    record: CPUPrefillSourceTrainingRecord,
) -> tuple[object, ...]:
    """Order adjacent MPMD jobs by calibrated dual-socket running time.

    The refresh driver pairs adjacent records. AVX2 took about 1.6x the native
    AVX512 time on the measured 7B/M16384 tail, so an integer 8:5 weight keeps
    the sort deterministic while pairing jobs with similar expected duration.
    """

    work = _source_record_work(record)
    return (
        -work,
        record.source_format,
        record.shape_name,
        record.isa_regime,
        record.m_values,
    )


def _order_source_records(
    records: Iterable[CPUPrefillSourceTrainingRecord],
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Create phase-compatible, work-balanced MPMD process groups."""

    materialized = tuple(records)
    identities = {
        (record.source_format, record.shape_name, record.isa_regime)
        for record in materialized
    }
    if len(identities) != len(materialized):
        raise ValueError("CPU prefill source jobs contain duplicate identities")
    by_m_inventory: dict[
        tuple[int, ...], list[CPUPrefillSourceTrainingRecord]
    ] = defaultdict(list)
    for record in materialized:
        by_m_inventory[record.m_values].append(record)
    ordered_groups = sorted(
        by_m_inventory.items(),
        key=lambda item: (
            -max(_source_record_work(record) for record in item[1]),
            -sum(_source_record_work(record) for record in item[1]),
            item[0],
        ),
    )
    return tuple(
        record
        for _, group in ordered_groups
        for record in sorted(group, key=_source_record_schedule_key)
    )


def coalesce_cpu_prefill_source_training_records(
    records: Iterable[CPUPrefillSourceTrainingRecord],
) -> tuple[CPUPrefillCoalescedSourceTrainingJob, ...]:
    """Coalesce only records that can share one synchronized trainer process.

    Weight packing and kernel timing remain format-specific.  The optimization
    removes process startup and permits the trainer to cache format-independent
    Q8_1 activation fixtures.  Grouping across geometry, ISA, or M inventory
    would change the measured workload and is therefore deliberately forbidden.

    Jobs are partitioned by both M inventory and format count.  Every rank in a
    coordinated MPMD launch consequently executes the same number of ordered
    ``(format, M)`` phases and cannot strand a peer in an MPI round barrier.
    """

    materialized = tuple(records)
    if not materialized:
        return ()
    format_order = {
        spec.label: index for index, spec in enumerate(FORMAT_SPECS)
    }
    grouped: dict[
        tuple[str, int, int, str, str, tuple[int, ...]], list[str]
    ] = defaultdict(list)
    source_identities: set[tuple[str, str, str]] = set()
    for record in materialized:
        identity = (
            record.source_format,
            record.shape_name,
            record.isa_regime,
        )
        if identity in source_identities:
            raise ValueError(
                "CPU prefill source jobs contain duplicate identities"
            )
        source_identities.add(identity)
        if record.source_format not in format_order:
            raise ValueError(
                f"unknown CPU prefill source format {record.source_format}"
            )
        grouped[(
            record.shape_name,
            record.n,
            record.k,
            record.isa_regime,
            record.runtime_isa,
            record.m_values,
        )].append(record.source_format)

    jobs = []
    for (
        shape_name,
        n,
        k,
        isa_regime,
        runtime_isa,
        m_values,
    ), raw_formats in grouped.items():
        source_formats = tuple(sorted(
            raw_formats,
            key=format_order.__getitem__,
        ))
        token_payload = ",".join(source_formats).encode("ascii")
        format_token = (
            source_formats[0]
            if len(source_formats) == 1
            else "formats-" + hashlib.sha256(token_payload).hexdigest()[:16]
        )
        jobs.append(CPUPrefillCoalescedSourceTrainingJob(
            source_formats=source_formats,
            format_token=format_token,
            shape_name=shape_name,
            n=n,
            k=k,
            isa_regime=isa_regime,
            runtime_isa=runtime_isa,
            m_values=m_values,
        ))

    def job_work(job: CPUPrefillCoalescedSourceTrainingJob) -> int:
        return (
            sum(job.m_values)
            * job.n
            * job.k
            * len(job.source_formats)
        )

    by_phase_inventory: dict[
        tuple[tuple[int, ...], int],
        list[CPUPrefillCoalescedSourceTrainingJob],
    ] = defaultdict(list)
    for job in jobs:
        by_phase_inventory[(
            job.m_values,
            job.format_phase_count,
        )].append(job)
    ordered_groups = sorted(
        by_phase_inventory.items(),
        key=lambda item: (
            -max(job_work(job) for job in item[1]),
            -sum(job_work(job) for job in item[1]),
            item[0],
        ),
    )
    return tuple(
        job
        for _, group in ordered_groups
        for job in sorted(
            group,
            key=lambda job: (
                -job_work(job),
                job.source_formats,
                job.shape_name,
                job.isa_regime,
            ),
        )
    )


def cpu_prefill_missing_base_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest,
    observed_paths: Iterable[Path],
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Return only current base-plan cells absent from historical aggregates.

    The production arithmetic-route manifest can legitimately add a base-plan
    obligation after an earlier corpus was frozen. Recollecting the complete
    two-hour baseline would duplicate valid evidence. This diff reads only cell
    identities, never timing or winners, and splits a process record when just
    some of its M values are missing. The ordinary adapter still validates the
    resulting aggregate's complete candidate matrix before any fit may begin.
    """

    observed: set[tuple[str, str, int, str]] = set()
    required_columns = {
        "source_format",
        "shape",
        "m",
        "build_isa",
        "runtime_isa_effective",
    }
    for path in (Path(item) for item in observed_paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = required_columns.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU prefill base-diff columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                pair = (
                    raw["build_isa"].strip().upper(),
                    raw["runtime_isa_effective"].strip().upper(),
                )
                try:
                    regime = _ISA_REGIME_BY_RAW_PAIR[pair]
                except KeyError as error:
                    raise ValueError(
                        f"{path}:{row_number}: unsupported CPU ISA pair {pair}"
                    ) from error
                observed.add((
                    raw["source_format"].strip().upper(),
                    raw["shape"].strip(),
                    int(raw["m"]),
                    regime,
                ))

    missing_records = []
    for record in cpu_prefill_source_training_records(route_manifest):
        missing_m = tuple(
            m
            for m in record.m_values
            if (
                record.source_format,
                record.shape_name,
                m,
                record.isa_regime,
            ) not in observed
        )
        if missing_m:
            missing_records.append(CPUPrefillSourceTrainingRecord(
                source_format=record.source_format,
                shape_name=record.shape_name,
                n=record.n,
                k=record.k,
                isa_regime=record.isa_regime,
                runtime_isa=record.runtime_isa,
                m_values=missing_m,
            ))
    return _order_source_records(missing_records)


def cpu_prefill_development_update_source_training_records(
    route_manifest: CPUPrefillSerialRouteManifest,
    observed_base_paths: Iterable[Path],
) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Combine normal enrichment jobs with a deduplicated base-plan delta."""

    grouped: dict[
        tuple[str, str, int, int, str, str], set[int]
    ] = defaultdict(set)
    for record in (
        *cpu_prefill_development_refinement_source_training_records(
            route_manifest
        ),
        *cpu_prefill_missing_base_source_training_records(
            route_manifest,
            observed_base_paths,
        ),
    ):
        grouped[(
            record.source_format,
            record.shape_name,
            record.n,
            record.k,
            record.isa_regime,
            record.runtime_isa,
        )].update(record.m_values)
    return _order_source_records(
        CPUPrefillSourceTrainingRecord(
            source_format=identity[0],
            shape_name=identity[1],
            n=identity[2],
            k=identity[3],
            isa_regime=identity[4],
            runtime_isa=identity[5],
            m_values=tuple(sorted(m_values)),
        )
        for identity, m_values in grouped.items()
    )


def _source_record_work(record: CPUPrefillSourceTrainingRecord) -> int:
    """Return the calibrated integer work estimate used for socket pairing."""

    isa_weight = 8 if record.runtime_isa == "avx2" else 5
    return sum(record.m_values) * record.n * record.k * isa_weight


def validate_source_alias_co_measurement(
    records: Iterable[CPUPrefillSourceTrainingRecord],
    runtime_cells: Iterable[CPUPrefillRuntimeTrainingCell],
) -> None:
    """Require every source alias to own exactly its runtime codebook's cells."""

    actual: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    for record in records:
        for m in record.m_values:
            actual[record.source_format].add(
                (record.shape_name, m, record.isa_regime)
            )
    expected: dict[int, set[tuple[str, int, str]]] = defaultdict(set)
    for cell in runtime_cells:
        expected[cell.runtime_codebook].add(
            (cell.shape_name, cell.m, cell.isa_regime)
        )
    for codebook, aliases in _codebook_aliases().items():
        for alias in aliases:
            if actual[alias] != expected[codebook]:
                raise ValueError(
                    f"{alias}: source alias cells disagree with CPU codebook {codebook}"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--development-refinement-records", action="store_true")
    parser.add_argument(
        "--development-update-records-from",
        action="append",
        type=Path,
        default=[],
        help=(
            "Historical base aggregate to diff before printing development "
            "refinement plus missing base-plan records; repeatable"
        ),
    )
    parser.add_argument(
        "--sealed-witness-plan",
        type=Path,
        help="Frozen-policy-bound JSON plan to print as sealed process records",
    )
    parser.add_argument(
        "--coalesce-format-fixtures",
        action="store_true",
        help=(
            "Print geometry/ISA/M-compatible source formats as one process "
            "job with a stable path token and synchronized phase count"
        ),
    )
    parser.add_argument(
        "--source-formats",
        help=(
            "Optional comma-separated source-format filter applied before "
            "process-job coalescing"
        ),
    )
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--route-manifest",
        action="append",
        type=str,
        default=[],
        help="C++ serial-route CSV; repeat once per ISA regime",
    )
    args = parser.parse_args()
    route_manifest = (
        read_cpu_prefill_route_manifests(args.route_manifest)
        if args.route_manifest
        else None
    )
    selected_modes = sum((
        args.records,
        args.development_refinement_records,
        bool(args.development_update_records_from),
        bool(args.sealed_witness_plan),
        args.summary,
    ))
    if selected_modes != 1:
        parser.error("use exactly one record inventory or --summary")
    if args.coalesce_format_fixtures and args.summary:
        parser.error("format fixture coalescing requires a record inventory")
    if args.development_update_records_from:
        if route_manifest is None:
            parser.error("development base-plan diff requires route manifests")
        records = cpu_prefill_development_update_source_training_records(
            route_manifest,
            args.development_update_records_from,
        )
    elif args.development_refinement_records:
        records = cpu_prefill_development_refinement_source_training_records(
            route_manifest
        )
    elif args.sealed_witness_plan:
        records = read_cpu_prefill_sealed_witness_plan(
            args.sealed_witness_plan
        ).records
    else:
        records = cpu_prefill_source_training_records(route_manifest)
    if args.source_formats is not None:
        requested_formats = {
            value.strip().upper()
            for value in args.source_formats.split(",")
            if value.strip()
        }
        known_formats = {spec.label for spec in FORMAT_SPECS}
        unknown_formats = requested_formats.difference(known_formats)
        if not requested_formats or unknown_formats:
            parser.error(
                "source-format filter is empty or unknown: "
                f"{sorted(unknown_formats)}"
            )
        records = tuple(
            record
            for record in records
            if record.source_format in requested_formats
        )
    if (
        args.records
        or args.development_refinement_records
        or args.development_update_records_from
        or args.sealed_witness_plan
    ):
        if args.coalesce_format_fixtures:
            for job in coalesce_cpu_prefill_source_training_records(records):
                print("\t".join((
                    ",".join(job.source_formats),
                    job.format_token,
                    job.shape_name,
                    str(job.n),
                    str(job.k),
                    job.isa_regime,
                    job.runtime_isa,
                    ",".join(str(m) for m in job.m_values),
                    str(job.format_phase_count),
                )))
        else:
            for record in records:
                print("\t".join((
                    record.source_format,
                    record.shape_name,
                    str(record.n),
                    str(record.k),
                    record.isa_regime,
                    record.runtime_isa,
                    ",".join(str(m) for m in record.m_values),
                )))
    elif args.summary:
        runtime = cpu_prefill_runtime_training_cells(route_manifest)
        print(f"version={CPU_PREFILL_TRAINING_PLAN_VERSION}")
        print(f"runtime_cells={len(runtime)}")
        print(f"source_cells={sum(len(item.m_values) for item in records)}")
        print(f"process_jobs={len(records)}")
        for regime in ISA_REGIMES:
            print(
                f"{regime}="
                f"{sum(cell.isa_regime == regime for cell in runtime)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
