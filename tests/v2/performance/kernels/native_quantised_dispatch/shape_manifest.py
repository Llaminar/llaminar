"""Validated cross-backend shape inventory for NativeQuantised policy training.

The JSON manifest owns generic development/sealed partitioning, while the
declarative Qwen release catalog contributes deduplicated production geometry.
Trainer binaries, the refresh transaction, and the common policy compiler all
consume the merged result. Keeping dimensions, exact-overlay ownership, and
sealed assignments together prevents one backend from silently certifying a
smaller shape surface than the others.

Expert operation provenance is separate from deduplicated matrix keys. A
matrix can serve several models or roles; collectors obtain original source
and participant intervals from the typed expert catalog, never from N/K.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Iterable

from .schema import AspectBucket
from .qwen_release_geometry import (
    qwen_mtp_head_geometries,
    qwen_release_geometries,
)


MANIFEST_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "native_quantised_decode_shapes_v5.json"
)


class ShapeRole(str, Enum):
    """Reason one shape belongs to the canonical measurement inventory."""

    PRODUCTION = "production"
    CERTIFICATION = "certification"


class ShapePartition(str, Enum):
    """Frozen learner visibility for one semantic-contract surface."""

    DEVELOPMENT = "development"
    SEALED = "sealed"


class ExpertProjectionRole(str, Enum):
    """Semantic expert operation; gate and up share projection geometry."""

    GATE_UP = "gate_up"
    DOWN = "down"


class ExpertProjectionLayout(str, Enum):
    """Source-owned layouts implemented by MoEExpertProjectionOwnership."""

    COMPLETE = "complete"
    INTERMEDIATE_SHARD = "intermediate_shard"
    DOWN_COLUMNS = "down_columns"


@dataclass(frozen=True)
class ExpertProjectionGeometry:
    """One semantic participant slice resolved to a canonical exact key.

    ``source_n/k`` describe the unsliced expert. ``first_n/k`` and
    ``physical_n/k`` describe its owned rectangle. ``policy_n`` is the full
    projection-local arithmetic width before output-task tiling, not the
    global source N or an output stride. Current expert descriptors publish
    local N without the dense-only OriginalOutputSlice equivalence scope.
    Identical physical keys may retain several legitimate role/layout owners;
    a collector deduplicates execution identities, not this source provenance.
    """

    model_family: str
    role: ExpertProjectionRole
    layout: ExpertProjectionLayout
    tp_degree: int
    participant: int
    intermediate_begin: int
    intermediate_width: int
    shape_name: str
    source_n: int
    source_k: int
    first_n: int
    first_k: int
    physical_n: int
    physical_k: int
    policy_n: int

    def __post_init__(self) -> None:
        """Reject incomplete or out-of-source participant metadata."""

        if (not self.model_family or not self.shape_name
                or not isinstance(self.role, ExpertProjectionRole)
                or not isinstance(self.layout, ExpertProjectionLayout)
                or self.tp_degree <= 0
                or not 0 <= self.participant < self.tp_degree
                or min(self.source_n, self.source_k, self.physical_n,
                       self.physical_k, self.intermediate_width, self.policy_n) <= 0
                or min(self.first_n, self.first_k, self.intermediate_begin) < 0
                or self.first_n + self.physical_n > self.source_n
                or self.first_k + self.physical_k > self.source_k
                or self.intermediate_begin + self.intermediate_width
                    > (self.source_n if self.role == ExpertProjectionRole.GATE_UP
                       else self.source_k)
                or self.policy_n != self.physical_n):
            raise ValueError("invalid source-owned expert projection geometry")


@dataclass(frozen=True)
class _ExpertSourceGeometry:
    """Typed operation owners of already-declared full projection keys."""

    model_family: str
    gate_up_shape: str
    down_shape: str
    source_block: int
    tp_degrees: tuple[int, ...]


# Full shape records own H/F; this typed catalog names their model/role owners.
# The actual QFN native down source
# (Q5_1/Q8_0) uses 32-element blocks; MoEExpertShardCohort::balancedIntervals
# preserves those blocks. This is one central source declaration, not a list
# of backend timing shapes or an inference from a coincidentally equal N/K.
_EXPERT_SOURCE_GEOMETRIES = (
    _ExpertSourceGeometry("qwen4exp", "Qwen4Exp_640x2560", "Qwen4Exp_2560x640",
                          32, (1, 2, 4, 8)),
)


@dataclass(frozen=True, order=True)
class NativeQuantisedShape:
    """One indivisible model or certification shape group."""

    name: str
    n: int
    k: int
    model_family: str
    role: ShapeRole
    exact_overlay: bool
    aspect_bucket: AspectBucket
    fast_partition: ShapePartition | None
    verifier_partition: ShapePartition | None
    prefill_partition: ShapePartition | None

    @property
    def work_items(self) -> int:
        """Return the exact work feature for this shape."""

        return self.n * self.k


@dataclass(frozen=True)
class NativeQuantisedShapeManifest:
    """Versioned, validated collection of canonical decode shapes."""

    schema_version: str
    description: str
    maximum_supported_weight_elements: int
    maximum_cpu_measurement_weight_elements: int
    shapes: tuple[NativeQuantisedShape, ...]

    @cached_property
    def _shapes_by_name(self) -> dict[str, tuple[NativeQuantisedShape, ...]]:
        """Index exact names once while retaining duplicate diagnostics.

        Completeness validation visits every observation in a production
        corpus.  Scanning the complete shape inventory for each row turns that
        linear validation into quadratic work.  A tuple-valued index preserves
        the historical ``by_name`` behavior for manually constructed duplicate
        manifests while making every normal lookup constant-time.
        """

        indexed: dict[str, list[NativeQuantisedShape]] = {}
        for shape in self.shapes:
            indexed.setdefault(shape.name, []).append(shape)
        return {
            name: tuple(matches)
            for name, matches in indexed.items()
        }

    def by_name(self, name: str) -> NativeQuantisedShape:
        """Resolve one exact case-sensitive trainer shape name."""

        matches = self._shapes_by_name.get(name, ())
        if len(matches) != 1:
            raise KeyError(f"unknown NativeQuantised shape {name!r}")
        return matches[0]

    def names(self, *, production_only: bool = False) -> tuple[str, ...]:
        """Return stable manifest-order names for a refresh profile."""

        return tuple(
            shape.name
            for shape in self.shapes
            if not production_only or shape.role == ShapeRole.PRODUCTION
        )

    def exact_overlay_names_for_dimensions(
        self,
        dimensions: Iterable[tuple[int, int]],
    ) -> tuple[str, ...]:
        """Resolve ordered geometry dimensions to canonical overlay names.

        The merged release catalog deliberately deduplicates physical matrix
        dimensions against older reviewed production aliases.  Focused sweep
        profiles therefore select by geometry and resolve names here instead
        of assuming that a release-derived display name survived that merge.
        Missing or ambiguous production overlays are fatal.
        """

        names = []
        for n, k in dimensions:
            matches = tuple(
                shape
                for shape in self.shapes
                if shape.role == ShapeRole.PRODUCTION
                and shape.exact_overlay
                and (shape.n, shape.k) == (n, k)
            )
            if len(matches) != 1:
                raise ValueError(
                    f"geometry {n}x{k} resolves to {len(matches)} exact overlays"
                )
            names.append(matches[0].name)
        return tuple(names)

    def partition_names(
        self,
        *,
        verifier: bool,
        partition: ShapePartition,
    ) -> tuple[str, ...]:
        """Return one semantic surface's frozen partition in manifest order."""

        field = "verifier_partition" if verifier else "fast_partition"
        return tuple(
            shape.name
            for shape in self.shapes
            if getattr(shape, field) == partition
        )

    def cpu_measurement_names(
        self,
        *,
        verifier: bool | None = None,
        partition: ShapePartition | None = None,
    ) -> tuple[str, ...]:
        """Return shapes inside the bounded canonical CPU timing envelope.

        The full manifest remains the runtime support and cross-backend shape
        inventory. CPU policy fitting includes the complete supported release
        envelope so every exact overlay, including the wider-vocabulary Qwen
        3.6 MTP heads, receives a measured CPU winner.
        """

        if partition is not None and verifier is None:
            raise ValueError("a CPU measurement partition requires a surface")
        field = None
        if verifier is not None:
            field = "verifier_partition" if verifier else "fast_partition"
        return tuple(
            shape.name
            for shape in self.shapes
            if shape.work_items <= self.maximum_cpu_measurement_weight_elements
            and (field is None or getattr(shape, field) is not None)
            and (
                partition is None
                or getattr(shape, field) == partition
            )
        )

    def digest(self) -> str:
        """Hash every policy-relevant shape and partition field canonically."""

        payload = {
            "schema_version": self.schema_version,
            "description": self.description,
            "maximum_supported_weight_elements": (
                self.maximum_supported_weight_elements
            ),
            "maximum_cpu_measurement_weight_elements": (
                self.maximum_cpu_measurement_weight_elements
            ),
            "shapes": [
                {
                    "name": shape.name,
                    "n": shape.n,
                    "k": shape.k,
                    "model_family": shape.model_family,
                    "role": shape.role.value,
                    "exact_overlay": shape.exact_overlay,
                    "aspect_bucket": shape.aspect_bucket.value,
                    "fast_partition": (
                        shape.fast_partition.value
                        if shape.fast_partition is not None
                        else None
                    ),
                    "verifier_partition": (
                        shape.verifier_partition.value
                        if shape.verifier_partition is not None
                        else None
                    ),
                    "prefill_partition": (
                        shape.prefill_partition.value
                        if shape.prefill_partition is not None
                        else None
                    ),
                }
                for shape in self.shapes
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _aspect_bucket(n: int, k: int) -> AspectBucket:
    """Classify dimensions with the shared exact aspect boundaries."""

    ratio = n / k
    if ratio >= 16.0:
        return AspectBucket.VERY_WIDE
    if ratio >= 2.0:
        return AspectBucket.WIDE
    if ratio >= 0.75:
        return AspectBucket.BALANCED
    return AspectBucket.TALL


def _parse_partition(
    raw: object,
    *,
    name: str,
    field: str,
) -> ShapePartition | None:
    """Parse one surface assignment, preserving explicit non-applicability.

    JSON ``null`` means that the geometry must never be measured or adapted for
    that semantic contract.  This is intentionally different from assigning a
    development partition: surface-specific refinement can enrich Fast M=1
    evidence without multiplying the grouped-verifier M sweep.
    """

    if raw is None:
        return None
    try:
        return ShapePartition(str(raw))
    except ValueError as error:
        raise ValueError(
            f"{name}: {field} must be development, sealed, or null"
        ) from error


def _parse_shape(raw: object) -> NativeQuantisedShape:
    """Parse and validate one JSON shape record without implicit defaults."""

    if not isinstance(raw, dict):
        raise ValueError("shape manifest entries must be JSON objects")
    required = {
        "name",
        "n",
        "k",
        "model_family",
        "role",
        "exact_overlay",
        "aspect_bucket",
        "fast_partition",
        "verifier_partition",
    }
    allowed = required | {"prefill_partition"}
    missing = sorted(required - set(raw))
    unexpected = sorted(set(raw) - allowed)
    if missing or unexpected:
        raise ValueError(
            f"shape manifest field mismatch: missing={missing} "
            f"unexpected={unexpected}"
        )
    name = str(raw["name"]).strip()
    model_family = str(raw["model_family"]).strip()
    n = int(raw["n"])
    k = int(raw["k"])
    if not name or not model_family:
        raise ValueError("shape name and model_family must be non-empty")
    if n <= 0 or k <= 0 or k % 32 != 0:
        raise ValueError(
            f"{name}: N and K must be positive and K must be divisible by 32"
        )
    bucket = AspectBucket(str(raw["aspect_bucket"]))
    expected_bucket = _aspect_bucket(n, k)
    if bucket != expected_bucket:
        raise ValueError(
            f"{name}: declared aspect bucket {bucket.value} does not match "
            f"{expected_bucket.value}"
        )
    role = ShapeRole(str(raw["role"]))
    exact_overlay = raw["exact_overlay"]
    if not isinstance(exact_overlay, bool):
        raise ValueError(f"{name}: exact_overlay must be a JSON boolean")
    if role == ShapeRole.CERTIFICATION and exact_overlay:
        raise ValueError(
            f"{name}: certification-grid shapes must exercise generic dispatch"
        )
    fast_partition = _parse_partition(
        raw["fast_partition"],
        name=name,
        field="fast_partition",
    )
    verifier_partition = _parse_partition(
        raw["verifier_partition"],
        name=name,
        field="verifier_partition",
    )
    prefill_partition = _parse_partition(
        raw.get("prefill_partition"),
        name=name,
        field="prefill_partition",
    )
    if (
        fast_partition is None
        and verifier_partition is None
        and prefill_partition is None
    ):
        raise ValueError(
            f"{name}: at least one semantic-contract surface must apply"
        )
    return NativeQuantisedShape(
        name=name,
        n=n,
        k=k,
        model_family=model_family,
        role=role,
        exact_overlay=exact_overlay,
        aspect_bucket=bucket,
        fast_partition=fast_partition,
        verifier_partition=verifier_partition,
        prefill_partition=prefill_partition,
    )


def load_shape_manifest(
    path: Path = MANIFEST_PATH,
    *,
    include_release_geometries: bool = True,
) -> NativeQuantisedShapeManifest:
    """Load the checked-in manifest and enforce cross-surface coverage rules.

    ``include_release_geometries`` exists for provenance readers that must
    reconstruct the pre-release-catalog manifest identity. Runtime planning
    and all new evidence transactions use the default merged inventory.
    """

    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "description",
        "maximum_supported_weight_elements",
        "maximum_cpu_measurement_weight_elements",
        "shapes",
    }:
        raise ValueError("NativeQuantised shape manifest has an invalid root schema")
    shapes_raw = raw["shapes"]
    if not isinstance(shapes_raw, list) or not shapes_raw:
        raise ValueError("NativeQuantised shape manifest must contain shapes")
    declared_shapes = tuple(_parse_shape(item) for item in shapes_raw)
    release_geometries = (
        qwen_release_geometries() if include_release_geometries else ()
    )
    production_dimensions = {
        (shape.n, shape.k)
        for shape in declared_shapes
        if shape.role == ShapeRole.PRODUCTION
    }
    release_shapes = tuple(
        NativeQuantisedShape(
            name=geometry.shape_name,
            n=geometry.n,
            k=geometry.k,
            model_family="qwen35-qwen36-release-geometries",
            role=ShapeRole.PRODUCTION,
            exact_overlay=True,
            aspect_bucket=_aspect_bucket(geometry.n, geometry.k),
            fast_partition=ShapePartition.DEVELOPMENT,
            verifier_partition=ShapePartition.DEVELOPMENT,
            prefill_partition=None,
        )
        for geometry in release_geometries
        if (geometry.n, geometry.k) not in production_dimensions
    )
    shapes = (*declared_shapes, *release_shapes)
    maximum_supported_weight_elements = int(
        raw["maximum_supported_weight_elements"]
    )
    if maximum_supported_weight_elements <= 0:
        raise ValueError(
            "maximum_supported_weight_elements must be positive"
        )
    maximum_cpu_measurement_weight_elements = int(
        raw["maximum_cpu_measurement_weight_elements"]
    )
    if not 0 < maximum_cpu_measurement_weight_elements <= maximum_supported_weight_elements:
        raise ValueError(
            "maximum_cpu_measurement_weight_elements must be positive and "
            "must not exceed maximum_supported_weight_elements"
        )
    oversized = tuple(
        shape
        for shape in shapes
        if shape.work_items > maximum_supported_weight_elements
    )
    if oversized:
        detail = ", ".join(
            f"{shape.name}={shape.n}x{shape.k}"
            for shape in oversized
        )
        raise ValueError(
            "shape manifest exceeds its supported projection-work envelope: "
            + detail
        )
    names = [shape.name for shape in shapes]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"duplicate NativeQuantised shape names: {duplicates}")
    # A production overlay may intentionally share dimensions with one generic
    # certification row. They are distinct evidence groups but still compile
    # to one geometry-only exact key. Duplicate dimensions within the same role
    # would merely repeat an overlay or a generic witness and remain invalid.
    for role in ShapeRole:
        dimensions = [
            (shape.n, shape.k) for shape in shapes if shape.role == role
        ]
        if len(dimensions) != len(set(dimensions)):
            duplicates = sorted({
                dimensions_pair
                for dimensions_pair in dimensions
                if dimensions.count(dimensions_pair) > 1
            })
            raise ValueError(
                f"duplicate {role.value} NativeQuantised shape dimensions: "
                f"{duplicates}"
            )

    for bucket in AspectBucket:
        bucket_shapes = [shape for shape in shapes if shape.aspect_bucket == bucket]
        for contract in ("fast", "verifier"):
            partition_field = f"{contract}_partition"
            development = [
                shape
                for shape in bucket_shapes
                if getattr(shape, partition_field) == ShapePartition.DEVELOPMENT
            ]
            sealed = [
                shape
                for shape in bucket_shapes
                if getattr(shape, partition_field) == ShapePartition.SEALED
            ]
            if len(development) < 5 or len(sealed) < 3:
                raise ValueError(
                    f"{bucket.value}/{contract}: requires at least five "
                    f"development and three sealed shapes, got "
                    f"{len(development)}/{len(sealed)}"
                )
        fast_sealed = {
            shape.name
            for shape in bucket_shapes
            if shape.fast_partition == ShapePartition.SEALED
        }
        verifier_sealed = {
            shape.name
            for shape in bucket_shapes
            if shape.verifier_partition == ShapePartition.SEALED
        }
        overlap = sorted(fast_sealed & verifier_sealed)
        if overlap:
            raise ValueError(
                f"{bucket.value}: Fast and verifier sealed groups overlap: {overlap}"
            )

    return NativeQuantisedShapeManifest(
        schema_version=str(raw["schema_version"]),
        description=str(raw["description"]),
        maximum_supported_weight_elements=maximum_supported_weight_elements,
        maximum_cpu_measurement_weight_elements=(
            maximum_cpu_measurement_weight_elements
        ),
        shapes=shapes,
    )


def load_declared_shape_manifest(
    path: Path = MANIFEST_PATH,
) -> NativeQuantisedShapeManifest:
    """Reconstruct the immutable JSON-only inventory for old provenance.

    The Qwen release catalog is additive. Historical evidence bound to the
    manifest before that catalog existed remains trustworthy when every shape
    it references still passes the current record and route validation.
    """

    return load_shape_manifest(path, include_release_geometries=False)


def expert_projection_geometries(
    manifest: NativeQuantisedShapeManifest | None = None,
) -> tuple[ExpertProjectionGeometry, ...]:
    """Resolve declared expert operations without reverse-identifying shapes.

    The central source catalog currently covers the requested QFN complete,
    equal TP2/4/8 intermediate, and owned-down-column layouts. Source-native
    remainder intervals retain their original participant identity. This is
    not an assertion of arbitrary proportional-topology coverage. The shared
    manifest remains the sole physical key authority: missing or ambiguous
    geometry is fatal, and reused release keys keep their canonical names.
    """

    selected = manifest or load_shape_manifest()
    result: list[ExpertProjectionGeometry] = []
    for source in _EXPERT_SOURCE_GEOMETRIES:
        gate_up = selected.by_name(source.gate_up_shape)
        down = selected.by_name(source.down_shape)
        intermediate, hidden = gate_up.n, gate_up.k
        if (not source.model_family or source.source_block <= 0
                or (down.n, down.k) != (hidden, intermediate)
                or any(shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay
                       for shape in (gate_up, down))
                or intermediate % source.source_block != 0
                or not source.tp_degrees
                or tuple(sorted(set(source.tp_degrees))) != source.tp_degrees
                or any(degree <= 0 or hidden % degree != 0
                       or intermediate // source.source_block < degree
                       for degree in source.tp_degrees)):
            raise ValueError("invalid canonical expert source geometry")

        def append(role: ExpertProjectionRole, layout: ExpertProjectionLayout,
                   degree: int, participant: int, begin: int, width: int,
                   first_n: int, first_k: int, n: int, k: int) -> None:
            """Bind one source rectangle to exactly one shared overlay key."""

            source_n, source_k = ((intermediate, hidden)
                                  if role == ExpertProjectionRole.GATE_UP
                                  else (hidden, intermediate))
            name, = selected.exact_overlay_names_for_dimensions(((n, k),))
            result.append(ExpertProjectionGeometry(
                source.model_family, role, layout, degree, participant,
                begin, width, name, source_n, source_k,
                first_n, first_k, n, k, n))

        for degree in source.tp_degrees:
            if degree == 1:
                append(ExpertProjectionRole.GATE_UP, ExpertProjectionLayout.COMPLETE,
                       degree, 0, 0, intermediate, 0, 0,
                       intermediate, hidden)
                append(ExpertProjectionRole.DOWN, ExpertProjectionLayout.COMPLETE,
                       degree, 0, 0, intermediate, 0, 0,
                       hidden, intermediate)
                continue
            blocks, remainder = divmod(intermediate // source.source_block,
                                       degree)
            begin = 0
            for participant in range(degree):
                width = (blocks + (participant < remainder)) * source.source_block
                append(ExpertProjectionRole.GATE_UP, ExpertProjectionLayout.INTERMEDIATE_SHARD,
                       degree, participant, begin, width, begin, 0,
                       width, hidden)
                append(ExpertProjectionRole.DOWN, ExpertProjectionLayout.INTERMEDIATE_SHARD,
                       degree, participant, begin, width, 0, begin,
                       hidden, width)
                begin += width
                # This layout leaves gate/up complete and owns a disjoint N
                # interval of down, with complete K and no reduction reorder.
                output_width = hidden // degree
                append(ExpertProjectionRole.GATE_UP, ExpertProjectionLayout.DOWN_COLUMNS,
                       degree, participant, 0, intermediate, 0, 0,
                       intermediate, hidden)
                append(ExpertProjectionRole.DOWN, ExpertProjectionLayout.DOWN_COLUMNS,
                       degree, participant, 0, intermediate,
                       participant * output_width, 0,
                       output_width, intermediate)
    return tuple(result)


def partition_assignments(
    shape_group_pairs: Iterable[tuple[str, str]],
    *,
    verifier: bool,
    manifest: NativeQuantisedShapeManifest | None = None,
) -> dict[str, ShapePartition]:
    """Map adapted shape-group IDs to their frozen manifest partitions.

    The adapter owns the opaque shape-group identifier while the manifest owns
    the human-readable shape name.  This bridge verifies that every represented
    group resolves to exactly one declared shape and that no shape name is
    assigned inconsistently across aliases, modes, candidates, or M values.
    """

    selected_manifest = manifest or load_shape_manifest()
    result: dict[str, ShapePartition] = {}
    names_by_group: dict[str, str] = {}
    for shape_group_id, shape_name in shape_group_pairs:
        previous = names_by_group.setdefault(shape_group_id, shape_name)
        if previous != shape_name:
            raise ValueError(
                f"shape group {shape_group_id!r} maps to both {previous!r} "
                f"and {shape_name!r}"
            )
        shape = selected_manifest.by_name(shape_name)
        assignment = (
            shape.verifier_partition if verifier else shape.fast_partition
        )
        if assignment is None:
            surface = "verifier" if verifier else "Fast"
            raise ValueError(
                f"shape {shape_name!r} is not applicable to the {surface} "
                "semantic-contract surface"
            )
        result[shape_group_id] = assignment
    return result


def main() -> int:
    """Expose stable manifest queries to the shell refresh transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--names",
        choices=(
            "all",
            "production",
            "mtp-head",
            "fast-development",
            "fast-sealed",
            "verifier-development",
            "verifier-sealed",
        ),
        help="Print one comma-separated shape inventory",
    )
    parser.add_argument(
        "--records",
        action="store_true",
        help="Print manifest-order tab-separated name, N, and K records",
    )
    parser.add_argument(
        "--cpu-measurement",
        action="store_true",
        help="Restrict names/records to the canonical CPU timing envelope",
    )
    parser.add_argument("--lookup", help="Resolve one exact shape name")
    parser.add_argument(
        "--field",
        choices=("n", "k", "fast_partition", "verifier_partition"),
        help="Field printed for --lookup",
    )
    args = parser.parse_args()
    manifest = load_shape_manifest()
    if args.names:
        if args.records or args.lookup or args.field:
            parser.error(
                "--names cannot be combined with --records/--lookup/--field"
            )
        if args.names in {"all", "production"}:
            names = manifest.names(production_only=args.names == "production")
        elif args.names == "mtp-head":
            names = manifest.exact_overlay_names_for_dimensions(
                (geometry.n, geometry.k)
                for geometry in qwen_mtp_head_geometries()
            )
        else:
            surface, partition_name = args.names.split("-", maxsplit=1)
            names = manifest.partition_names(
                verifier=surface == "verifier",
                partition=ShapePartition(partition_name),
            )
        if args.cpu_measurement:
            allowed = set(manifest.cpu_measurement_names())
            names = tuple(name for name in names if name in allowed)
        print(",".join(names))
        return 0
    if args.records:
        if args.lookup or args.field:
            parser.error("--records cannot be combined with --lookup/--field")
        allowed = (
            set(manifest.cpu_measurement_names())
            if args.cpu_measurement
            else None
        )
        for shape in manifest.shapes:
            if allowed is not None and shape.name not in allowed:
                continue
            print(f"{shape.name}\t{shape.n}\t{shape.k}")
        return 0
    if not args.lookup or not args.field:
        parser.error("use --names, --records, or both --lookup and --field")
    value = getattr(manifest.by_name(args.lookup), args.field)
    if value is None:
        print("not-applicable")
    else:
        print(value.value if isinstance(value, Enum) else value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
