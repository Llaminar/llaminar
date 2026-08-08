"""Validated cross-backend shape inventory for NativeVNNI policy training.

The JSON manifest owns generic development/sealed partitioning, while the
declarative Qwen release catalog contributes deduplicated production geometry.
Trainer binaries, the refresh transaction, and the common policy compiler all
consume the merged result. Keeping dimensions, exact-overlay ownership, and
sealed assignments together prevents one backend from silently certifying a
smaller shape surface than the others.
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
    / "native_vnni_decode_shapes_v5.json"
)


class ShapeRole(str, Enum):
    """Reason one shape belongs to the canonical measurement inventory."""

    PRODUCTION = "production"
    CERTIFICATION = "certification"


class ShapePartition(str, Enum):
    """Frozen learner visibility for one semantic-contract surface."""

    DEVELOPMENT = "development"
    SEALED = "sealed"


@dataclass(frozen=True, order=True)
class NativeVNNIShape:
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
class NativeVNNIShapeManifest:
    """Versioned, validated collection of canonical decode shapes."""

    schema_version: str
    description: str
    maximum_supported_weight_elements: int
    maximum_cpu_measurement_weight_elements: int
    shapes: tuple[NativeVNNIShape, ...]

    @cached_property
    def _shapes_by_name(self) -> dict[str, tuple[NativeVNNIShape, ...]]:
        """Index exact names once while retaining duplicate diagnostics.

        Completeness validation visits every observation in a production
        corpus.  Scanning the complete shape inventory for each row turns that
        linear validation into quadratic work.  A tuple-valued index preserves
        the historical ``by_name`` behavior for manually constructed duplicate
        manifests while making every normal lookup constant-time.
        """

        indexed: dict[str, list[NativeVNNIShape]] = {}
        for shape in self.shapes:
            indexed.setdefault(shape.name, []).append(shape)
        return {
            name: tuple(matches)
            for name, matches in indexed.items()
        }

    def by_name(self, name: str) -> NativeVNNIShape:
        """Resolve one exact case-sensitive trainer shape name."""

        matches = self._shapes_by_name.get(name, ())
        if len(matches) != 1:
            raise KeyError(f"unknown NativeVNNI shape {name!r}")
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


def _parse_shape(raw: object) -> NativeVNNIShape:
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
    return NativeVNNIShape(
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
) -> NativeVNNIShapeManifest:
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
        raise ValueError("NativeVNNI shape manifest has an invalid root schema")
    shapes_raw = raw["shapes"]
    if not isinstance(shapes_raw, list) or not shapes_raw:
        raise ValueError("NativeVNNI shape manifest must contain shapes")
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
        NativeVNNIShape(
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
        raise ValueError(f"duplicate NativeVNNI shape names: {duplicates}")
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
                f"duplicate {role.value} NativeVNNI shape dimensions: "
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

    return NativeVNNIShapeManifest(
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
) -> NativeVNNIShapeManifest:
    """Reconstruct the immutable JSON-only inventory for old provenance.

    The Qwen release catalog is additive. Historical evidence bound to the
    manifest before that catalog existed remains trustworthy when every shape
    it references still passes the current record and route validation.
    """

    return load_shape_manifest(path, include_release_geometries=False)


def partition_assignments(
    shape_group_pairs: Iterable[tuple[str, str]],
    *,
    verifier: bool,
    manifest: NativeVNNIShapeManifest | None = None,
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
