"""Authenticated additive migrations between CPU prefill development splits.

Production shape inventories evolve after expensive timing corpora have already
been collected.  Re-running a deterministic covering-array solver against the
larger inventory is not a migration: the solver can select different cells for
unrelated legacy shapes, silently changing the evidence contract and forcing a
full recollection.  This module instead records an explicit additive lineage.

One lineage plan binds the immutable source aggregate and timing sidecar, the
source and target split manifests, both route-manifest generations, every
source refinement-plan digest, and the exact process-cell inventory present in
the source corpus.  Its increment is restricted to production shapes newly
introduced by the target split.  The analyzer can therefore require
byte-authenticated legacy evidence plus exactly the new launch cells, while
rejecting both missing and undeclared rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .cpu_prefill_generic_refinement import (
    CPUPrefillGenericRefinementPlan,
    read_cpu_prefill_generic_refinement_plan,
)
from .cpu_prefill_route_manifest import (
    CPUPrefillSerialRouteManifest,
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
    cpu_prefill_source_training_records,
)
from .format_registry import format_spec
from .shape_manifest import (
    ShapeRole,
    load_declared_shape_manifest,
    load_shape_manifest,
)


CPU_PREFILL_DEVELOPMENT_LINEAGE_SCHEMA = (
    "cpu-prefill-development-lineage-v3"
)

_ISA_REGIME_BY_RAW_PAIR = {
    ("AVX2", "AVX2"): "avx2-build.avx2-runtime",
    ("AVX512", "AVX2"): "avx512-build.avx2-runtime",
    ("AVX512", "AVX512"): "avx512-build.avx512-runtime",
}


def _sha256_file(path: Path) -> str:
    """Return the content identity of one immutable raw evidence file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _record_mapping(record: CPUPrefillSourceTrainingRecord) -> dict[str, object]:
    """Serialize one grouped process record without relying on dataclass order."""

    return {
        "source_format": record.source_format,
        "shape_name": record.shape_name,
        "n": record.n,
        "k": record.k,
        "isa_regime": record.isa_regime,
        "runtime_isa": record.runtime_isa,
        "m_values": list(record.m_values),
    }


def _record_from_mapping(raw: Mapping[str, object]) -> CPUPrefillSourceTrainingRecord:
    """Parse one strict grouped process record from lineage JSON."""

    required = {
        "source_format",
        "shape_name",
        "n",
        "k",
        "isa_regime",
        "runtime_isa",
        "m_values",
    }
    if set(raw) != required:
        raise ValueError("CPU prefill lineage record schema is invalid")
    return CPUPrefillSourceTrainingRecord(
        source_format=str(raw["source_format"]),
        shape_name=str(raw["shape_name"]),
        n=int(raw["n"]),
        k=int(raw["k"]),
        isa_regime=str(raw["isa_regime"]),
        runtime_isa=str(raw["runtime_isa"]),
        m_values=tuple(int(value) for value in raw["m_values"]),
    )


def _record_sort_key(record: CPUPrefillSourceTrainingRecord) -> tuple[object, ...]:
    """Use a stable work-first order compatible with the socket scheduler."""

    return (
        record.m_values,
        -(record.n * record.k * max(record.m_values)),
        record.shape_name,
        record.source_format,
        record.isa_regime,
    )


def _record_cells(
    records: Iterable[CPUPrefillSourceTrainingRecord],
) -> frozenset[tuple[str, str, int, str]]:
    """Expand grouped process records into their exact source launch cells."""

    cells = []
    for record in records:
        cells.extend(
            (
                record.source_format,
                record.shape_name,
                m,
                record.isa_regime,
            )
            for m in record.m_values
        )
    if len(cells) != len(set(cells)):
        raise ValueError("CPU prefill lineage contains duplicate launch cells")
    return frozenset(cells)


def _source_records(path: Path) -> tuple[CPUPrefillSourceTrainingRecord, ...]:
    """Collapse a validated raw aggregate into its exact process inventory."""

    required = {
        "source_format",
        "shape",
        "m",
        "n",
        "k",
        "build_isa",
        "runtime_isa_effective",
        "threads",
    }
    grouped: dict[tuple[str, str, int, int, str, str], set[int]] = defaultdict(set)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{path}: CPU prefill lineage source lacks {sorted(missing)}"
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
            source_format = raw["source_format"].strip().upper()
            shape_name = raw["shape"].strip()
            n = int(raw["n"])
            k = int(raw["k"])
            m = int(raw["m"])
            if not source_format or not shape_name or min(n, k, m) <= 0:
                raise ValueError(
                    f"{path}:{row_number}: invalid CPU prefill launch identity"
                )
            grouped[(
                source_format,
                shape_name,
                n,
                k,
                regime,
                RUNTIME_ISA_BY_REGIME[regime],
            )].add(m)
    records = tuple(sorted((
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
    ), key=_record_sort_key))
    if not records:
        raise ValueError(f"{path}: CPU prefill lineage source is empty")
    _record_cells(records)
    return records


def _source_thread_count(path: Path) -> int:
    """Return the single OpenMP width authenticated by the source aggregate."""

    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "threads" not in (reader.fieldnames or ()):
            raise ValueError(f"{path}: CPU prefill lineage source lacks threads")
        counts = {
            int(raw["threads"])
            for raw in reader
        }
    if len(counts) != 1 or next(iter(counts), 0) <= 0:
        raise ValueError(
            f"{path}: CPU prefill lineage source has inconsistent thread counts"
        )
    return next(iter(counts))


@dataclass(frozen=True)
class CPUPrefillDevelopmentLineagePlan:
    """Immutable proof that one target corpus is an additive split migration."""

    schema_version: str
    source_split_manifest_digest: str
    target_split_manifest_digest: str
    source_route_manifest_digest: str
    target_route_manifest_digest: str
    shape_manifest_digest: str
    source_aggregate_digest: str
    source_timing_digest: str
    source_refinement_plan_digests: tuple[str, ...]
    source_refinement_split_manifest_digests: tuple[str, ...]
    thread_count: int
    added_production_shapes: tuple[str, ...]
    source_records: tuple[CPUPrefillSourceTrainingRecord, ...]
    increment_records: tuple[CPUPrefillSourceTrainingRecord, ...]

    def canonical_mapping(self) -> dict[str, object]:
        """Return the complete stable representation used for publication."""

        return {
            "schema_version": self.schema_version,
            "source_split_manifest_digest": self.source_split_manifest_digest,
            "target_split_manifest_digest": self.target_split_manifest_digest,
            "source_route_manifest_digest": self.source_route_manifest_digest,
            "target_route_manifest_digest": self.target_route_manifest_digest,
            "shape_manifest_digest": self.shape_manifest_digest,
            "source_aggregate_digest": self.source_aggregate_digest,
            "source_timing_digest": self.source_timing_digest,
            "source_refinement_plan_digests": list(
                self.source_refinement_plan_digests
            ),
            "source_refinement_split_manifest_digests": list(
                self.source_refinement_split_manifest_digests
            ),
            "thread_count": self.thread_count,
            "added_production_shapes": list(self.added_production_shapes),
            "source_records": [
                _record_mapping(record) for record in self.source_records
            ],
            "increment_records": [
                _record_mapping(record) for record in self.increment_records
            ],
        }

    def digest(self) -> str:
        """Return the content identity used by resumable collection shards."""

        encoded = json.dumps(
            self.canonical_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _validate_split_extension(
    source: CPUPrefillSplitManifest,
    target: CPUPrefillSplitManifest,
) -> tuple[str, ...]:
    """Require a target split that adds production shapes and changes nothing else."""

    if source.sealed_shapes != target.sealed_shapes:
        raise ValueError("CPU prefill lineage cannot change sealed geometries")
    if source.sealed_m_values != target.sealed_m_values:
        raise ValueError("CPU prefill lineage cannot change sealed M values")
    source_shapes = source.shape_names(sealed=False)
    target_shapes = target.shape_names(sealed=False)
    if not source_shapes < target_shapes:
        raise ValueError(
            "CPU prefill lineage target must strictly extend development shapes"
        )
    added = tuple(sorted(target_shapes - source_shapes))
    manifest = load_shape_manifest()
    for name in added:
        shape = manifest.by_name(name)
        if shape.role != ShapeRole.PRODUCTION or not shape.exact_overlay:
            raise ValueError(
                f"{name}: CPU prefill lineage additions must be production overlays"
            )
    return added


def _validate_records(
    plan: CPUPrefillDevelopmentLineagePlan,
    source_split: CPUPrefillSplitManifest,
    target_split: CPUPrefillSplitManifest,
    source_route_manifest: CPUPrefillSerialRouteManifest,
    target_route_manifest: CPUPrefillSerialRouteManifest,
    source_refinement_plans: Iterable[
        CPUPrefillGenericRefinementPlan
    ],
) -> None:
    """Prove all source and increment cells belong to their declared partitions."""

    route_thread_counts = {
        source_route_manifest.thread_count(),
        target_route_manifest.thread_count(),
    }
    if plan.thread_count <= 0 or route_thread_counts != {plan.thread_count}:
        raise ValueError(
            "CPU prefill lineage thread count disagrees with route manifests"
        )

    if tuple(sorted(plan.source_records, key=_record_sort_key)) != plan.source_records:
        raise ValueError("CPU prefill lineage source records are not canonical")
    if tuple(sorted(plan.increment_records, key=_record_sort_key)) != (
        plan.increment_records
    ):
        raise ValueError("CPU prefill lineage increment records are not canonical")
    source_cells = _record_cells(plan.source_records)
    increment_cells = _record_cells(plan.increment_records)
    if source_cells & increment_cells:
        raise ValueError("CPU prefill lineage source and increment cells overlap")

    plans = tuple(source_refinement_plans)
    refinement_cells = {
        (record.source_format, record.shape_name, m, record.isa_regime)
        for refinement_plan in plans
        for record in refinement_plan.records
        for m in record.m_values
    }
    base_shapes = set(source_split.development_shapes)
    source_shapes = {record.shape_name for record in plan.source_records}
    if not base_shapes <= source_shapes:
        raise ValueError(
            "CPU prefill lineage source shape inventory is incomplete: "
            f"missing={sorted(base_shapes-source_shapes)}"
        )
    source_extra_cells = {
        cell for cell in source_cells if cell[1] not in base_shapes
    }
    unauthorized_extra_cells = source_extra_cells - refinement_cells
    if unauthorized_extra_cells:
        raise ValueError(
            "CPU prefill lineage source contains cells not authorized by its "
            "refinement plans: "
            f"{sorted(unauthorized_extra_cells)[:8]}"
        )
    refinement_shapes = {cell[1] for cell in source_extra_cells}
    increment_shapes = {record.shape_name for record in plan.increment_records}
    if increment_shapes != set(plan.added_production_shapes):
        raise ValueError("CPU prefill lineage increment shape inventory is incomplete")
    expected_target_shapes = set(target_split.development_shapes) | refinement_shapes
    if source_shapes | increment_shapes != expected_target_shapes:
        raise ValueError("CPU prefill lineage does not cover the target split")

    shape_manifest = load_shape_manifest()
    for record in plan.source_records:
        shape = shape_manifest.by_name(record.shape_name)
        if (record.n, record.k) != (shape.n, shape.k):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage dimensions changed"
            )
        if (
            record.isa_regime not in ISA_REGIMES
            or record.runtime_isa != RUNTIME_ISA_BY_REGIME[record.isa_regime]
            or not record.m_values
            or tuple(sorted(set(record.m_values))) != record.m_values
        ):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage record is invalid"
            )
        codebook = format_spec(record.source_format).cpu_execution_codebook_id
        route = source_route_manifest.route_for(
            codebook,
            record.shape_name,
            record.isa_regime,
        )
        if (route.n, route.k) != (record.n, record.k):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage route changed"
            )
    for record in plan.increment_records:
        shape = shape_manifest.by_name(record.shape_name)
        if (record.n, record.k) != (shape.n, shape.k):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage dimensions changed"
            )
        if (
            record.isa_regime not in ISA_REGIMES
            or record.runtime_isa != RUNTIME_ISA_BY_REGIME[record.isa_regime]
            or not record.m_values
            or tuple(sorted(set(record.m_values))) != record.m_values
        ):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage record is invalid"
            )
        codebook = format_spec(record.source_format).cpu_execution_codebook_id
        route = target_route_manifest.route_for(
            codebook,
            record.shape_name,
            record.isa_regime,
        )
        if (route.n, route.k) != (record.n, record.k):
            raise ValueError(
                f"{record.shape_name}: CPU prefill lineage route changed"
            )


def build_cpu_prefill_development_lineage_plan(
    source_aggregate: Path,
    source_timing: Path,
    source_split: CPUPrefillSplitManifest,
    target_split: CPUPrefillSplitManifest,
    source_route_manifest: CPUPrefillSerialRouteManifest,
    target_route_manifest: CPUPrefillSerialRouteManifest,
    source_refinement_plans: Iterable[CPUPrefillGenericRefinementPlan],
) -> CPUPrefillDevelopmentLineagePlan:
    """Build an additive migration plan from already-published source evidence."""

    added = _validate_split_extension(source_split, target_split)
    source_threads = _source_thread_count(source_aggregate)
    route_threads = {
        source_route_manifest.thread_count(),
        target_route_manifest.thread_count(),
    }
    if route_threads != {source_threads}:
        raise ValueError(
            "CPU prefill lineage source thread count disagrees with route manifests"
        )
    increment_records = tuple(sorted((
        record
        for record in cpu_prefill_source_training_records(target_route_manifest)
        if record.shape_name in set(added)
    ), key=_record_sort_key))
    plans = tuple(source_refinement_plans)
    plan = CPUPrefillDevelopmentLineagePlan(
        schema_version=CPU_PREFILL_DEVELOPMENT_LINEAGE_SCHEMA,
        source_split_manifest_digest=source_split.digest(),
        target_split_manifest_digest=target_split.digest(),
        source_route_manifest_digest=source_route_manifest.digest(),
        target_route_manifest_digest=target_route_manifest.digest(),
        shape_manifest_digest=load_shape_manifest().digest(),
        source_aggregate_digest=_sha256_file(source_aggregate),
        source_timing_digest=_sha256_file(source_timing),
        source_refinement_plan_digests=tuple(item.digest() for item in plans),
        source_refinement_split_manifest_digests=tuple(sorted({
            item.split_manifest_digest for item in plans
        })),
        thread_count=source_threads,
        added_production_shapes=added,
        source_records=_source_records(source_aggregate),
        increment_records=increment_records,
    )
    _validate_records(
        plan,
        source_split,
        target_split,
        source_route_manifest,
        target_route_manifest,
        plans,
    )
    return plan


def _read_source_refinement_plans(
    paths: Iterable[Path],
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifests: Iterable[CPUPrefillSplitManifest],
) -> tuple[CPUPrefillGenericRefinementPlan, ...]:
    """Authenticate each historical plan against the split it names.

    A long-lived development corpus can span more than one reviewed split.
    Selecting one split for every plan would either reject valid history or
    tempt callers to rewrite old plan bytes.  Instead, the plan's immutable
    split digest chooses from an explicit caller-supplied set of historical
    manifests.  Unknown digests remain a hard provenance failure.
    """

    splits_by_digest = {
        split.digest(): split for split in split_manifests
    }
    plans = []
    for path in paths:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: CPU prefill refinement plan is invalid")
        split_digest = str(raw.get("split_manifest_digest", ""))
        try:
            split = splits_by_digest[split_digest]
        except KeyError as error:
            raise ValueError(
                f"{path}: no historical split manifest matches {split_digest}"
            ) from error
        plans.append(
            read_cpu_prefill_generic_refinement_plan(
                path,
                route_manifest,
                split,
            )
        )
    return tuple(plans)


def write_cpu_prefill_development_lineage_plan(
    path: Path,
    plan: CPUPrefillDevelopmentLineagePlan,
) -> None:
    """Publish one lineage plan atomically after all invariants pass."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.canonical_mapping(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_cpu_prefill_development_lineage_plan(
    path: Path,
    source_aggregate: Path,
    source_timing: Path,
    source_split: CPUPrefillSplitManifest,
    target_split: CPUPrefillSplitManifest,
    source_route_manifest: CPUPrefillSerialRouteManifest,
    target_route_manifest: CPUPrefillSerialRouteManifest,
    source_refinement_plans: Iterable[CPUPrefillGenericRefinementPlan],
) -> CPUPrefillDevelopmentLineagePlan:
    """Read a lineage plan and reauthenticate every external source artifact."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "source_split_manifest_digest",
        "target_split_manifest_digest",
        "source_route_manifest_digest",
        "target_route_manifest_digest",
        "shape_manifest_digest",
        "source_aggregate_digest",
        "source_timing_digest",
        "source_refinement_plan_digests",
        "source_refinement_split_manifest_digests",
        "thread_count",
        "added_production_shapes",
        "source_records",
        "increment_records",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise ValueError("CPU prefill development lineage root schema is invalid")
    plan = CPUPrefillDevelopmentLineagePlan(
        schema_version=str(raw["schema_version"]),
        source_split_manifest_digest=str(raw["source_split_manifest_digest"]),
        target_split_manifest_digest=str(raw["target_split_manifest_digest"]),
        source_route_manifest_digest=str(raw["source_route_manifest_digest"]),
        target_route_manifest_digest=str(raw["target_route_manifest_digest"]),
        shape_manifest_digest=str(raw["shape_manifest_digest"]),
        source_aggregate_digest=str(raw["source_aggregate_digest"]),
        source_timing_digest=str(raw["source_timing_digest"]),
        source_refinement_plan_digests=tuple(
            str(item) for item in raw["source_refinement_plan_digests"]
        ),
        source_refinement_split_manifest_digests=tuple(
            str(item)
            for item in raw["source_refinement_split_manifest_digests"]
        ),
        thread_count=int(raw["thread_count"]),
        added_production_shapes=tuple(
            str(item) for item in raw["added_production_shapes"]
        ),
        source_records=tuple(
            _record_from_mapping(item) for item in raw["source_records"]
        ),
        increment_records=tuple(
            _record_from_mapping(item) for item in raw["increment_records"]
        ),
    )
    plans = tuple(source_refinement_plans)
    expected = {
        "schema_version": CPU_PREFILL_DEVELOPMENT_LINEAGE_SCHEMA,
        "source_split_manifest_digest": source_split.digest(),
        "target_split_manifest_digest": target_split.digest(),
        "source_route_manifest_digest": source_route_manifest.digest(),
        "target_route_manifest_digest": target_route_manifest.digest(),
        "source_aggregate_digest": _sha256_file(source_aggregate),
        "source_timing_digest": _sha256_file(source_timing),
        "source_refinement_plan_digests": tuple(item.digest() for item in plans),
        "source_refinement_split_manifest_digests": tuple(sorted({
            item.split_manifest_digest for item in plans
        })),
        "thread_count": _source_thread_count(source_aggregate),
        "added_production_shapes": _validate_split_extension(
            source_split,
            target_split,
        ),
    }
    actual = {
        name: getattr(plan, name) for name in expected
    }
    mismatches = [name for name in expected if actual[name] != expected[name]]
    accepted_shape_manifest_digests = {
        load_shape_manifest().digest(),
        load_declared_shape_manifest().digest(),
    }
    if plan.shape_manifest_digest not in accepted_shape_manifest_digests:
        mismatches.append("shape_manifest_digest")
    if mismatches:
        raise ValueError(
            "CPU prefill development lineage provenance changed: "
            + ", ".join(sorted(mismatches))
        )
    _validate_records(
        plan,
        source_split,
        target_split,
        source_route_manifest,
        target_route_manifest,
        plans,
    )
    return plan


def _print_records(records: Iterable[CPUPrefillSourceTrainingRecord]) -> None:
    """Expose stable tab-separated process jobs to the refresh transaction."""

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


def _print_launch_records(plan: CPUPrefillDevelopmentLineagePlan) -> None:
    """Emit one authenticated thread width followed by the launch inventory.

    A lineage plan reauthenticates a large immutable source corpus before any
    field may be consumed. Emitting both values in one process prevents the
    refresh driver from hashing and parsing that corpus independently for a
    thread-count query and then again for its process records.
    """

    print(f"thread_count\t{plan.thread_count}")
    _print_records(plan.increment_records)


def main() -> int:
    """Build, authenticate, or print one additive development lineage plan."""

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--build", action="store_true")
    source.add_argument("--plan", type=Path)
    parser.add_argument("--source-aggregate", required=True, type=Path)
    parser.add_argument("--source-timing", required=True, type=Path)
    parser.add_argument("--source-split-manifest", required=True, type=Path)
    parser.add_argument("--target-split-manifest", required=True, type=Path)
    parser.add_argument(
        "--source-refinement-plan",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--source-refinement-split-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--source-route-manifest",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument("--route-manifest", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--records", action="store_true")
    parser.add_argument("--thread-count", action="store_true")
    parser.add_argument("--launch-records", action="store_true")
    args = parser.parse_args()
    if not args.route_manifest:
        parser.error("CPU prefill lineage requires target route manifests")
    if not args.source_route_manifest:
        parser.error("CPU prefill lineage requires source route manifests")
    if args.build and args.output is None:
        parser.error("--build requires --output")
    if args.plan is not None and args.output is not None:
        parser.error("--output is valid only with --build")

    source_routes = read_cpu_prefill_route_manifests(
        args.source_route_manifest
    )
    target_routes = read_cpu_prefill_route_manifests(args.route_manifest)
    source_split = load_cpu_prefill_split_manifest(args.source_split_manifest)
    target_split = load_cpu_prefill_split_manifest(args.target_split_manifest)
    refinement_splits = (
        source_split,
        *(
            load_cpu_prefill_split_manifest(path)
            for path in args.source_refinement_split_manifest
        ),
    )
    refinement_plans = _read_source_refinement_plans(
        args.source_refinement_plan,
        source_routes,
        refinement_splits,
    )
    if args.build:
        plan = build_cpu_prefill_development_lineage_plan(
            args.source_aggregate,
            args.source_timing,
            source_split,
            target_split,
            source_routes,
            target_routes,
            refinement_plans,
        )
        write_cpu_prefill_development_lineage_plan(args.output, plan)
    else:
        plan = read_cpu_prefill_development_lineage_plan(
            args.plan,
            args.source_aggregate,
            args.source_timing,
            source_split,
            target_split,
            source_routes,
            target_routes,
            refinement_plans,
        )
    output_modes = sum((args.records, args.thread_count, args.launch_records))
    if output_modes > 1:
        parser.error(
            "--records, --thread-count, and --launch-records are mutually "
            "exclusive"
        )
    if args.launch_records:
        _print_launch_records(plan)
    elif args.records:
        _print_records(plan.increment_records)
    elif args.thread_count:
        print(plan.thread_count)
    elif not args.build:
        print(plan.digest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
