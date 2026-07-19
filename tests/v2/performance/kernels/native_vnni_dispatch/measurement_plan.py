"""Versioned, backend-aware NativeVNNI GPU corpus measurement plan.

The shape manifest is the broad supported geometry inventory.  This module is
the narrower timing contract: it names the common feature-covering development
set and preserves backend/format-specific evidence only on the surface that
motivated it.  Keeping that distinction explicit prevents a local refinement
from silently becoming an all-format Cartesian product that takes many hours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import AspectBucket, Backend, ExecutionMode, SemanticContract
from .shape_manifest import (
    MANIFEST_PATH,
    NativeVNNIShapeManifest,
    ShapePartition,
    ShapeRole,
    load_shape_manifest,
)


MEASUREMENT_PLAN_PATH = (
    Path(__file__).resolve().parent
    / "manifests"
    / "native_vnni_gpu_measurement_plan_v1.json"
)


@dataclass(frozen=True)
class FastDevelopmentExtension:
    """One backend/format-specific Fast-M1 development evidence scope."""

    backend: Backend
    source_formats: tuple[str, ...]
    shape_model_families: tuple[str, ...]
    shape_names: tuple[str, ...]

    def applies(self, backend: Backend, source_format: str) -> bool:
        """Return whether this extension owns the requested physical surface."""

        return self.backend == backend and source_format in self.source_formats


@dataclass(frozen=True)
class NativeVNNIGPUMeasurementPlan:
    """Resolved common and scoped GPU timing inventory."""

    schema_version: str
    shape_manifest_schema: str
    include_all_production_shapes: bool
    common_development_shapes: tuple[str, ...]
    fast_development_extensions: tuple[FastDevelopmentExtension, ...]

    def fast_development_shapes(
        self,
        *,
        backend: Backend,
        source_format: str,
    ) -> tuple[str, ...]:
        """Return common plus scoped Fast-M1 development shape names."""

        result = list(self.common_development_shapes)
        for extension in self.fast_development_extensions:
            if extension.applies(backend, source_format):
                result.extend(extension.shape_names)
        return tuple(result)

    def scoped_fast_shapes(
        self,
        *,
        backend: Backend,
        source_format: str,
    ) -> tuple[str, ...]:
        """Return only Fast-M1 shapes added for one backend/format surface."""

        return tuple(
            shape_name
            for extension in self.fast_development_extensions
            if extension.applies(backend, source_format)
            for shape_name in extension.shape_names
        )

    def fast_shape_applies(
        self,
        manifest: NativeVNNIShapeManifest,
        *,
        backend: Backend,
        source_format: str,
        shape_name: str,
    ) -> bool:
        """Return whether one physical surface owns the Fast-M1 contract.

        Common development geometry and the sealed Fast holdout apply to every
        backend/format.  A scoped extension applies only to the exact backend
        and source-format selector recorded in this plan.
        """

        if shape_name in self.common_development_shapes:
            return True
        if shape_name in manifest.partition_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        ):
            return True
        return shape_name in self.scoped_fast_shapes(
            backend=backend,
            source_format=source_format,
        )

    def verifier_shape_applies(
        self,
        manifest: NativeVNNIShapeManifest,
        *,
        shape_name: str,
    ) -> bool:
        """Return whether a shape owns grouped serial-M1 verifier evidence."""

        return (
            shape_name in self.common_development_shapes
            or shape_name in manifest.partition_names(
                verifier=True,
                partition=ShapePartition.SEALED,
            )
        )

    def development_shape_names(self, *, fast: bool) -> tuple[str, ...]:
        """Return the common development inventory for one semantic surface."""

        del fast
        return self.common_development_shapes

    def expected_shape_names(
        self,
        manifest: NativeVNNIShapeManifest,
        *,
        backend: Backend,
    ) -> frozenset[str]:
        """Return every shape that a complete backend transaction must contain."""

        result = set(self.common_development_shapes)
        result.update(manifest.partition_names(
            verifier=False,
            partition=ShapePartition.SEALED,
        ))
        result.update(manifest.partition_names(
            verifier=True,
            partition=ShapePartition.SEALED,
        ))
        for extension in self.fast_development_extensions:
            if extension.backend == backend:
                result.update(extension.shape_names)
        return frozenset(result)

    def expected_decode_surfaces(
        self,
        manifest: NativeVNNIShapeManifest,
        *,
        backend: Backend,
        source_formats: tuple[str, ...],
        execution_modes: tuple[ExecutionMode, ...],
        verifier_m_values: frozenset[int],
    ) -> frozenset[tuple[SemanticContract, int, str, str, ExecutionMode]]:
        """Enumerate every physical surface required for installation.

        Keeping this Cartesian inventory in the plan prevents backend
        analyzers from drifting on whether a shape owns Fast M1, grouped
        verifier depths, or a backend/format-specific development extension.
        Candidate rows are intentionally absent from this identity: each
        analyzer separately proves its complete forceable candidate registry.
        """

        surfaces = set()
        shape_names = self.expected_shape_names(manifest, backend=backend)
        for source_format in source_formats:
            for execution_mode in execution_modes:
                for shape_name in shape_names:
                    if self.fast_shape_applies(
                        manifest,
                        backend=backend,
                        source_format=source_format,
                        shape_name=shape_name,
                    ):
                        surfaces.add((
                            SemanticContract.FAST,
                            1,
                            shape_name,
                            source_format,
                            execution_mode,
                        ))
                    if self.verifier_shape_applies(
                        manifest,
                        shape_name=shape_name,
                    ):
                        surfaces.update(
                            (
                                SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
                                verifier_m,
                                shape_name,
                                source_format,
                                execution_mode,
                            )
                            for verifier_m in verifier_m_values
                        )
        return frozenset(surfaces)

    def digest(self, manifest: NativeVNNIShapeManifest) -> str:
        """Bind the resolved timing plan to the broad shape manifest."""

        payload = {
            "schema_version": self.schema_version,
            "shape_manifest_schema": self.shape_manifest_schema,
            "shape_manifest_digest": manifest.digest(),
            "include_all_production_shapes": self.include_all_production_shapes,
            "common_development_shapes": list(self.common_development_shapes),
            "fast_development_extensions": [
                {
                    "backend": extension.backend.value,
                    "source_formats": list(extension.source_formats),
                    "shape_model_families": list(
                        extension.shape_model_families
                    ),
                    "shape_names": list(extension.shape_names),
                }
                for extension in self.fast_development_extensions
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_gpu_measurement_plan(
    path: Path = MEASUREMENT_PLAN_PATH,
    *,
    manifest: NativeVNNIShapeManifest | None = None,
) -> NativeVNNIGPUMeasurementPlan:
    """Load, resolve, and validate one checked-in GPU measurement plan."""

    selected_manifest = manifest or load_shape_manifest()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "shape_manifest_schema",
        "include_all_production_shapes",
        "common_development_shapes",
        "fast_development_extensions",
    }
    if set(raw) != required:
        raise ValueError(
            "GPU measurement plan field mismatch: "
            f"missing={sorted(required - set(raw))} "
            f"unexpected={sorted(set(raw) - required)}"
        )
    if raw["shape_manifest_schema"] != selected_manifest.schema_version:
        raise ValueError(
            "GPU measurement plan shape schema does not match the selected "
            "manifest"
        )

    include_all_production_shapes = raw["include_all_production_shapes"]
    if not isinstance(include_all_production_shapes, bool):
        raise ValueError("include_all_production_shapes must be a JSON boolean")
    declared_common = tuple(
        str(name) for name in raw["common_development_shapes"]
    )
    if len(declared_common) != len(set(declared_common)) or not declared_common:
        raise ValueError("common GPU development shapes must be unique and non-empty")
    production_names = tuple(
        shape.name
        for shape in selected_manifest.shapes
        if shape.role == ShapeRole.PRODUCTION
    )
    common = declared_common
    if include_all_production_shapes:
        common = (*common, *(
            name for name in production_names if name not in common
        ))
    common_shapes = tuple(selected_manifest.by_name(name) for name in common)
    production = set(production_names)
    if not production.issubset(common):
        raise ValueError(
            "common GPU development plan omits production shapes: "
            f"{sorted(production - set(common))}"
        )
    if len(declared_common) != 64:
        raise ValueError(
            "GPU common development plan must retain its 64 reviewed base shapes"
        )
    if {
        shape.aspect_bucket for shape in common_shapes
    } != set(AspectBucket):
        raise ValueError("common GPU development shapes must cover every aspect")
    for shape in common_shapes:
        if (
            shape.fast_partition != ShapePartition.DEVELOPMENT
            or shape.verifier_partition != ShapePartition.DEVELOPMENT
        ):
            raise ValueError(
                f"common GPU shape {shape.name} is not development evidence "
                "for both Fast and verifier surfaces"
            )

    extensions = []
    seen_surfaces: set[tuple[Backend, str, str]] = set()
    for extension_raw in raw["fast_development_extensions"]:
        extension_required = {
            "backend",
            "source_formats",
            "shape_model_families",
        }
        if not isinstance(extension_raw, dict) or set(extension_raw) != extension_required:
            raise ValueError("Fast development extension field mismatch")
        backend = Backend(str(extension_raw["backend"]))
        source_formats = tuple(
            str(value).strip().upper()
            for value in extension_raw["source_formats"]
        )
        model_families = tuple(
            str(value).strip()
            for value in extension_raw["shape_model_families"]
        )
        if (
            not source_formats
            or not model_families
            or len(source_formats) != len(set(source_formats))
            or len(model_families) != len(set(model_families))
        ):
            raise ValueError("Fast development extension selectors must be unique")
        shape_names = tuple(
            shape.name
            for shape in selected_manifest.shapes
            if shape.model_family in model_families
        )
        if not shape_names:
            raise ValueError("Fast development extension resolves no shapes")
        for shape_name in shape_names:
            shape = selected_manifest.by_name(shape_name)
            if (
                shape.fast_partition != ShapePartition.DEVELOPMENT
                or shape.verifier_partition is not None
                or shape_name in common
            ):
                raise ValueError(
                    f"scoped Fast shape {shape_name} has invalid applicability"
                )
        for source_format in source_formats:
            for shape_name in shape_names:
                identity = (backend, source_format, shape_name)
                if identity in seen_surfaces:
                    raise ValueError(
                        f"duplicate scoped GPU measurement surface {identity}"
                    )
                seen_surfaces.add(identity)
        extensions.append(FastDevelopmentExtension(
            backend=backend,
            source_formats=source_formats,
            shape_model_families=model_families,
            shape_names=shape_names,
        ))

    return NativeVNNIGPUMeasurementPlan(
        schema_version=str(raw["schema_version"]),
        shape_manifest_schema=str(raw["shape_manifest_schema"]),
        include_all_production_shapes=include_all_production_shapes,
        common_development_shapes=common,
        fast_development_extensions=tuple(extensions),
    )


def main() -> int:
    """Expose stable plan queries to the shell refresh transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=MEASUREMENT_PLAN_PATH,
    )
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        default=MANIFEST_PATH,
    )
    parser.add_argument(
        "--names",
        choices=("common-development", "scoped-fast"),
        required=True,
    )
    parser.add_argument("--backend", choices=("cuda", "rocm"))
    parser.add_argument("--source-format")
    args = parser.parse_args()
    manifest = load_shape_manifest(args.shape_manifest)
    plan = load_gpu_measurement_plan(args.plan, manifest=manifest)
    if args.names == "common-development":
        if args.backend or args.source_format:
            parser.error("common development names have no backend/format selector")
        names = plan.common_development_shapes
    else:
        if not args.backend or not args.source_format:
            parser.error("scoped Fast names require --backend and --source-format")
        names = plan.scoped_fast_shapes(
            backend=Backend(args.backend),
            source_format=args.source_format.upper(),
        )
    print(",".join(names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
