"""Reviewed staged split for CPU NativeVNNI prefill certification.

The v1 and v2 transactions were opened while repairing the learner and a
partial-N memory-safety defect. The v3 holdout then exposed both a launch-wave
modeling gap and a protocol gap: it contained only M=64, so rules for every
larger prefill bucket could never receive sealed exercise. Those measurements
are now immutable development evidence. Version 4 adds focused neighborhoods
around the two dominant development-CV cliffs. Its Cartesian sealed collector
was stopped after proving that fixed per-source preconditioning made the full
matrix fundamentally miss the backend-wide two-hour target. Version 5
committed a fresh geometry pool plus a deterministic frozen-leaf witness projection:
only rule predicates and timing-free C++ routes may select cells after freeze,
and every selected codebook is still expanded to all source aliases. Sealed
rows launched only after the generic policy digest and witness-plan bytes
existed. That certificate rejected 21 of 70 leaves, concentrated at the
very-wide full-K schedule crossover. Version 6 promotes the exact v5 witness
inventory to development evidence and reserves a new untouched pool around
that crossover. Version 6 then missed the native-AVX512 IQ4 `nbc1`/`nbc4`
boundary by 3.0684% on its fresh very-wide witness. Version 7 promotes the
exact v6 witness inventory to development, rotates only the three geometries
that were measured, and retains five still-unopened geometries. The timing-free
projection and no-refit rule remain unchanged. Version 7 then opened three
V9 witness geometries while rejecting a policy that was not generically total.
Version 8 promotes exactly those measured cells to development, retains the
five v7 geometries that were never launched, and reserves three fresh V10
geometries for the first total-generic certificate. Version 9 adds the four
Qwen 3.6 35B-A3B MoE production projections to development. Version 10 makes
production ownership declarative: every canonical production geometry is
resolved into development in stable measurement-plan order. Version 11 bounds
new CPU evidence at M=2048, introduces M=512, and caps geometries owned by a
14B-or-larger model at M=512. Version 12 profiles sub-14B overlays only at
M={64,128,256,512} and 14B-or-larger overlays only at M={64,128}. Version 13
keeps the full overlay inventory while profiling below-7B owners at M={32,128},
7B-to-below-14B owners at M={32,64}, and 14B-or-larger owners at M={32}.
Versions 8 through 12 remain readable so
in-flight, digest-bound refinement rounds can finish without rewriting their
historical split identities.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .corpus import ObservationCorpus
from .prefill_matrix import (
    CPU_PREFILL_M_BUCKETS,
    CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS,
    LEGACY_GPU_PREFILL_M_BUCKETS,
    cpu_prefill_measurements,
)
from .shape_manifest import ShapePartition, ShapeRole, load_shape_manifest


MANIFEST_PATH = Path(__file__).with_name("manifests") / (
    "native_vnni_cpu_prefill_split_v13.json"
)
SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v13"
LEGACY_SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v8"
V9_SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v9"
V10_SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v10"
V11_SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v11"
V12_SCHEMA_VERSION = "native-vnni-cpu-prefill-split-v12"
SUPPORTED_SCHEMA_VERSIONS = frozenset((
    LEGACY_SCHEMA_VERSION,
    V9_SCHEMA_VERSION,
    V10_SCHEMA_VERSION,
    V11_SCHEMA_VERSION,
    V12_SCHEMA_VERSION,
    SCHEMA_VERSION,
))
V11_CPU_PREFILL_M_BUCKETS = (64, 256, 512, 1024, 2048)
V12_CPU_PREFILL_M_BUCKETS = (64, 128, 256, 512)
V9_QWEN36_MOE_PRODUCTION_SHAPES = frozenset((
    "35BMoE_Expert_GateUp",
    "35BMoE_Expert_Down",
    "Qwen36MoE_GDN_QKVProjection",
    "Qwen36MoE_GDN_ZProjection",
))

# Version 10 originally appended its declaratively owned production geometries
# in the order returned by the then-current prefill matrix. The later
# all-exact-overlay refactor retained the identical shape set but changed that
# append order, after the original order-sensitive digest had entered signed
# refinement, lineage, and sealed-witness artifacts. Readers accept this one
# published identity only after ``__post_init__`` has validated the complete
# current semantic ownership set.
V10_HISTORICAL_ORDER_DIGESTS = frozenset((
    "sha256:f12fe2a80463f7e5ed756332b2fe72aa3034f7815c2c2d074c662a61791f2e01",
))


@dataclass(frozen=True)
class CPUPrefillSplitManifest:
    """Immutable geometry assignments for profiler-informed certification."""

    schema_version: str
    seed: str
    development_shapes: tuple[str, ...]
    sealed_shapes: tuple[str, ...]
    sealed_m_values: tuple[int, ...]
    include_all_production_shapes: bool = False

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError("unsupported CPU prefill split schema")
        if not self.seed.strip():
            raise ValueError("CPU prefill split seed must not be empty")
        development = set(self.development_shapes)
        sealed = set(self.sealed_shapes)
        if len(development) != len(self.development_shapes):
            raise ValueError("CPU prefill development shapes are not unique")
        if len(sealed) != len(self.sealed_shapes):
            raise ValueError("CPU prefill sealed shapes are not unique")
        if development & sealed:
            raise ValueError("CPU prefill split partitions overlap")
        shape_manifest = load_shape_manifest()
        declared_refinement = {
            shape.name
            for shape in shape_manifest.shapes
            if shape.prefill_partition == ShapePartition.DEVELOPMENT
        }
        if self.schema_version == SCHEMA_VERSION:
            if not self.include_all_production_shapes:
                raise ValueError(
                    "current CPU prefill split must include every production "
                    "shape declaratively"
                )
            production = {
                measurement.shape.name
                for measurement in cpu_prefill_measurements()
            }
            expected_development = production | declared_refinement
            if development != expected_development:
                raise ValueError(
                    "CPU prefill development split must contain every "
                    "production and opened refinement geometry: "
                    f"missing={sorted(expected_development - development)} "
                    f"unexpected={sorted(development - expected_development)}"
                )
        elif self.schema_version in {
            V10_SCHEMA_VERSION,
            V11_SCHEMA_VERSION,
            V12_SCHEMA_VERSION,
        }:
            if not self.include_all_production_shapes:
                raise ValueError(
                    "declarative CPU prefill split must include every "
                    "production shape"
                )
            production = {
                measurement.shape.name
                for measurement in cpu_prefill_measurements()
            }
            unknown = development.difference(production, declared_refinement)
            if not production.issubset(development) or unknown:
                raise ValueError(
                    "historical declarative CPU prefill ownership is invalid: "
                    f"missing_production={sorted(production - development)} "
                    f"unexpected={sorted(unknown)}"
                )
        else:
            if self.include_all_production_shapes:
                raise ValueError(
                    "historical CPU prefill splits cannot change production "
                    "ownership dynamically"
                )
            unknown = {
                name for name in development
                if shape_manifest.by_name(name).role != ShapeRole.PRODUCTION
                and name not in declared_refinement
            }
            if unknown:
                raise ValueError(
                    "historical CPU prefill development split contains an "
                    "invalid ownership assignment: "
                    f"unexpected={sorted(unknown)}"
                )
        if len(sealed) < 8:
            raise ValueError(
                "CPU prefill split requires at least eight fresh sealed geometries"
            )
        if self.schema_version == SCHEMA_VERSION:
            expected_sealed_m_values = CPU_PREFILL_M_BUCKETS
        elif self.schema_version == V11_SCHEMA_VERSION:
            expected_sealed_m_values = V11_CPU_PREFILL_M_BUCKETS
        elif self.schema_version == V12_SCHEMA_VERSION:
            expected_sealed_m_values = V12_CPU_PREFILL_M_BUCKETS
        else:
            expected_sealed_m_values = LEGACY_GPU_PREFILL_M_BUCKETS
        if self.sealed_m_values != expected_sealed_m_values:
            raise ValueError(
                "CPU prefill sealed pool must expose every canonical "
                f"M bucket; expected={expected_sealed_m_values!r}"
            )

        declared_sealed = {
            shape.name
            for shape in shape_manifest.shapes
            if shape.prefill_partition == ShapePartition.SEALED
        }
        if sealed != declared_sealed:
            raise ValueError(
                "CPU prefill split sealed inventory disagrees with canonical "
                "shape ownership: "
                f"missing={sorted(declared_sealed - sealed)} "
                f"unexpected={sorted(sealed - declared_sealed)}"
            )
        for name in self.sealed_shapes:
            shape = shape_manifest.by_name(name)
            if shape.role != ShapeRole.CERTIFICATION or shape.exact_overlay:
                raise ValueError(
                    f"{name}: sealed CPU prefill shape must be certification-only"
                )
            if shape.work_items > CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS:
                raise ValueError(f"{name}: exceeds the CPU measurement envelope")
        for name in declared_refinement:
            shape = shape_manifest.by_name(name)
            if shape.role != ShapeRole.CERTIFICATION or shape.exact_overlay:
                raise ValueError(
                    f"{name}: opened CPU prefill evidence must remain "
                    "non-overlay certification geometry"
                )

    def digest(self) -> str:
        """Hash every reviewed assignment used by the frozen policy."""

        payload = {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "development_shapes": list(self.development_shapes),
            "sealed_shapes": list(self.sealed_shapes),
            "sealed_m_values": list(self.sealed_m_values),
        }
        # The declarative ownership flag was introduced by v10. Omitting it
        # from historical payloads preserves the exact v8/v9 identities that
        # authenticate already-collected evidence and refinement plans.
        if self.schema_version in {
            V10_SCHEMA_VERSION,
            V11_SCHEMA_VERSION,
            V12_SCHEMA_VERSION,
            SCHEMA_VERSION,
        }:
            payload["include_all_production_shapes"] = (
                self.include_all_production_shapes
            )
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def accepted_digests(self) -> frozenset[str]:
        """Return current and published order-only identities for this split."""

        historical = (
            V10_HISTORICAL_ORDER_DIGESTS
            if self.schema_version == V10_SCHEMA_VERSION
            else frozenset()
        )
        return frozenset((self.digest(), *historical))

    def shape_names(self, *, sealed: bool) -> frozenset[str]:
        """Return one partition's complete logical geometry inventory."""

        return frozenset(
            self.sealed_shapes if sealed else self.development_shapes
        )

    def require_partition(
        self,
        corpus: ObservationCorpus,
        *,
        sealed: bool,
        additional_development_shapes: Iterable[str] = (),
        additional_development_geometries: Mapping[
            str, tuple[int, int]
        ] | None = None,
        burned_sealed_development_shapes: Iterable[str] = (),
    ) -> None:
        """Reject mixed or incomplete shape-name partitions.

        A failed development fit may authorize additional non-overlay boundary
        points through a digest-bound refinement plan. The caller must name
        those geometries explicitly here. A separately authenticated failed
        seal may name reserved sealed shapes as burned development evidence;
        callers must identify that subset explicitly. Sealed ownership remains
        immutable and cannot be extended by any refinement transaction.
        """

        expected = set(self.shape_names(sealed=sealed))
        additional = set(additional_development_shapes)
        generated = dict(additional_development_geometries or {})
        burned = set(burned_sealed_development_shapes)
        if set(generated).difference(additional):
            raise ValueError(
                "CPU prefill generated refinement geometry is not authorized"
            )
        if burned.difference(additional):
            raise ValueError(
                "CPU prefill burned-seal development shape is not authorized"
            )
        if sealed and additional:
            raise ValueError("sealed CPU prefill partition cannot be refined")
        if not sealed:
            shape_manifest = load_shape_manifest()
            sealed_names = self.shape_names(sealed=True)
            for name in additional:
                try:
                    shape = shape_manifest.by_name(name)
                except KeyError:
                    dimensions = generated.get(name)
                    if dimensions is None:
                        raise ValueError(
                            f"{name}: CPU prefill refinement shape is unknown"
                        ) from None
                    n, k = dimensions
                    eligible = (
                        name.startswith("CPUPrefillAutoRefine_")
                        and n > 0
                        and k > 0
                        and k % 32 == 0
                        and n * k <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
                    )
                else:
                    if name in generated and generated[name] != (shape.n, shape.k):
                        raise ValueError(
                            f"{name}: CPU prefill refinement geometry changed"
                        )
                    if name in burned:
                        eligible = (
                            name in sealed_names
                            and shape.role == ShapeRole.CERTIFICATION
                            and not shape.exact_overlay
                            and shape.prefill_partition == ShapePartition.SEALED
                            and shape.work_items
                            <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
                        )
                    else:
                        eligible = (
                            name not in sealed_names
                            and shape.role == ShapeRole.CERTIFICATION
                            and not shape.exact_overlay
                            and shape.prefill_partition != ShapePartition.SEALED
                            and shape.work_items
                            <= CPU_PREFILL_MAXIMUM_WEIGHT_ELEMENTS
                        )
                if not eligible:
                    raise ValueError(
                        f"{name}: CPU prefill refinement shape is not eligible"
                    )
            expected.update(additional)
        actual = {row.shape_name for row in corpus}
        if actual != expected:
            label = "sealed" if sealed else "development"
            raise ValueError(
                f"CPU prefill {label} shape inventory is incomplete: "
                f"missing={sorted(expected - actual)} "
                f"unexpected={sorted(actual - expected)}"
            )


def load_cpu_prefill_split_manifest(
    path: Path = MANIFEST_PATH,
) -> CPUPrefillSplitManifest:
    """Read the strict checked-in CPU prefill split manifest."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    schema_version = str(raw.get("schema_version", ""))
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("unsupported CPU prefill split schema")
    expected_fields = {
        "schema_version",
        "seed",
        "development_shapes",
        "sealed_shapes",
        "sealed_m_values",
    }
    if schema_version in {
        V10_SCHEMA_VERSION,
        V11_SCHEMA_VERSION,
        V12_SCHEMA_VERSION,
        SCHEMA_VERSION,
    }:
        expected_fields.add("include_all_production_shapes")
    if set(raw) != expected_fields:
        raise ValueError(
            "CPU prefill split fields do not match schema: "
            f"missing={sorted(expected_fields - set(raw))} "
            f"unexpected={sorted(set(raw) - expected_fields)}"
        )
    include_all_production_shapes = raw.get(
        "include_all_production_shapes", False
    )
    if not isinstance(include_all_production_shapes, bool):
        raise ValueError(
            "CPU prefill include_all_production_shapes must be boolean"
        )
    development_shapes = [
        str(item) for item in raw["development_shapes"]
    ]
    if include_all_production_shapes:
        declared = set(development_shapes)
        for measurement in cpu_prefill_measurements():
            name = measurement.shape.name
            if name not in declared:
                development_shapes.append(name)
                declared.add(name)
    return CPUPrefillSplitManifest(
        schema_version=schema_version,
        seed=str(raw["seed"]),
        development_shapes=tuple(development_shapes),
        sealed_shapes=tuple(str(item) for item in raw["sealed_shapes"]),
        sealed_m_values=tuple(int(item) for item in raw["sealed_m_values"]),
        include_all_production_shapes=include_all_production_shapes,
    )
