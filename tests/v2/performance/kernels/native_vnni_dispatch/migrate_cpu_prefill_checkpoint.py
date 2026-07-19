#!/usr/bin/env python3
"""Rebase validated CPU prefill checkpoints onto the current collection plan.

The v13 timing protocol changes only the stopping rule for a candidate that has
not reached median stability: the hard sample ceiling and elapsed timing budget
must now both be satisfied.  A finalized v12 forceable row that already stopped
because it was stable therefore has identical acquisition evidence under v13.
The one-sample, non-forceable diagnostic rows are likewise unchanged, except
that v13 names their stop reason ``fixed_samples`` instead of the misleading
``hard_max_samples`` label.

The current v13 protocol is also accepted as a source when a covering-plan
contract changes.  In that case an aggregate is reusable only when its filename
and complete M inventory exactly match the current canonical source record.  A
record that gained even one measurement is omitted wholesale so the refresh
driver cannot mistake a partial old shard for a complete new one.

This utility deliberately refuses unstable forceable evidence, timing
mismatches, incomplete sidecars, and unsupported protocols.  It copies timing
sidecars byte-for-byte, validates every translated aggregate and sidecar pair
with the current production adapter, and publishes the destination directory
atomically.  Source ``.inprogress`` files are recorded in the migration
manifest but are never copied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


KERNEL_PERF_ROOT = Path(__file__).resolve().parent.parent
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_prefill import (  # noqa: E402
    CPUPrefillAdapterContext,
    adapt_cpu_prefill_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.profiles import (  # noqa: E402
    LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
    LEGACY_ADAPTIVE_TIMING_PROTOCOL,
    MeasurementProfile,
)
from native_vnni_dispatch.cpu_prefill_training_plan import (  # noqa: E402
    cpu_prefill_source_training_records,
)
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPUPrefillSerialRouteManifest,
    read_cpu_prefill_route_manifests,
)


LEGACY_TIMING_PROTOCOL = "elapsed-stability-interleaved-v12"
LEGACY_DIAGNOSTIC_STOP = "hard_max_samples"
TARGET_DIAGNOSTIC_STOP = "fixed_samples"
TARGET_CONTRACT_FILE = "cpu_prefill_collection_contract.sha256"
MIGRATION_MANIFEST_FILE = "cpu_prefill_checkpoint_migration.json"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class MigratedAggregate:
    """Counts proven while translating one aggregate CSV shard."""

    rows: int
    forceable_rows: int
    diagnostic_rows: int


def _canonical_target_inventory(
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> dict[str, frozenset[int]]:
    """Map every current atomic shard name to its required complete M set."""

    return {
        (
            f"cpu_prefill.{record.source_format}.{record.shape_name}."
            f"{record.isa_regime}.csv"
        ): frozenset(record.m_values)
        for record in cpu_prefill_source_training_records(route_manifest)
    }


def _aggregate_m_inventory(path: Path) -> frozenset[int]:
    """Read the distinct M phases represented by one finalized aggregate."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "m" not in reader.fieldnames:
            raise ValueError(f"{path}: aggregate lacks an M column")
        values = frozenset(int(raw["m"]) for raw in reader)
    if not values:
        raise ValueError(f"{path}: aggregate contains no M phases")
    return values


def _explicit_bool(name: str, value: str, path: Path, row_number: int) -> bool:
    """Parse the trainer's explicit CSV boolean representation."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(
        f"{path}:{row_number}: {name} must be an explicit boolean, got {value!r}"
    )


def _require_sha256(name: str, value: str) -> str:
    """Return one normalized bare SHA-256 digest or fail closed."""

    normalized = value.strip().lower()
    if not SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal digits")
    return normalized


def _corpus_digest(paths: Iterable[Path], root: Path) -> str:
    """Hash relative names and bytes so the manifest authenticates file layout."""

    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.name):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _canonical_fieldnames(fieldnames: list[str], path: Path) -> list[str]:
    """Place v13 ceiling provenance beside the timing protocol field."""

    try:
        protocol_index = fieldnames.index("timing_protocol")
    except ValueError as exc:
        raise ValueError(f"{path}: source lacks timing_protocol") from exc
    migrated = [
        field for field in fieldnames if field != "timing_ceiling_policy"
    ]
    protocol_index = migrated.index("timing_protocol")
    migrated.insert(protocol_index + 1, "timing_ceiling_policy")
    return migrated


def _migrate_row(
    raw: Mapping[str, str],
    *,
    path: Path,
    row_number: int,
) -> tuple[dict[str, str], bool]:
    """Translate one row only when its finalized evidence remains reusable."""

    row = dict(raw)
    source_protocol = row.get("timing_protocol", "").strip()
    if source_protocol == LEGACY_TIMING_PROTOCOL:
        source_diagnostic_stop = LEGACY_DIAGNOSTIC_STOP
        if row.get("timing_ceiling_policy", "").strip():
            raise ValueError(
                f"{path}:{row_number}: legacy row unexpectedly has a ceiling policy"
            )
    elif source_protocol == LEGACY_ADAPTIVE_TIMING_PROTOCOL:
        source_diagnostic_stop = TARGET_DIAGNOSTIC_STOP
        if row.get("timing_ceiling_policy", "").strip() != (
            LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
        ):
            raise ValueError(
                f"{path}:{row_number}: current row has the wrong ceiling policy"
            )
    else:
        raise ValueError(
            f"{path}:{row_number}: unsupported timing protocol {source_protocol!r}"
        )
    for field in ("bit_mismatches", "repeat_byte_mismatches"):
        if int(row.get(field, "-1")) != 0:
            raise ValueError(f"{path}:{row_number}: {field} is not zero")
    if row.get("grouped_output_digest", "") != row.get(
        "serial_output_digest", ""
    ):
        raise ValueError(f"{path}:{row_number}: output digests disagree")

    forceable = _explicit_bool(
        "route_counter_ok", row.get("route_counter_ok", ""), path, row_number
    )
    converged = _explicit_bool(
        "timing_converged", row.get("timing_converged", ""), path, row_number
    )
    stop_reason = row.get("timing_stop_reason", "").strip()
    sample_count = int(row.get("sample_count", "0"))
    if forceable:
        if not converged or stop_reason not in {
            "stable_sample_floor",
            "elapsed_stable",
        }:
            raise ValueError(
                f"{path}:{row_number}: forceable row did not finalize stable"
            )
    elif (
        converged
        or sample_count != 1
        or stop_reason != source_diagnostic_stop
    ):
        raise ValueError(
            f"{path}:{row_number}: diagnostic row is not fixed one-sample evidence"
        )

    row["timing_protocol"] = LEGACY_ADAPTIVE_TIMING_PROTOCOL
    row["timing_ceiling_policy"] = LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
    if not forceable:
        row["timing_stop_reason"] = TARGET_DIAGNOSTIC_STOP
    return row, forceable


def _migrate_aggregate(source: Path, destination: Path) -> MigratedAggregate:
    """Write one translated aggregate into the private staging directory."""

    rows = forceable_rows = diagnostic_rows = 0
    with source.open(newline="", encoding="utf-8") as input_handle:
        reader = csv.DictReader(input_handle)
        if not reader.fieldnames:
            raise ValueError(f"{source}: aggregate has no CSV header")
        fieldnames = _canonical_fieldnames(list(reader.fieldnames), source)
        with destination.open("w", newline="", encoding="utf-8") as output_handle:
            # The production trainer writes canonical LF records.  The refresh
            # combiner authenticates headers byte-for-byte, so Python's default
            # CRLF CSV dialect would make otherwise identical fresh and
            # migrated shards impossible to combine on Linux.
            writer = csv.DictWriter(
                output_handle,
                fieldnames=fieldnames,
                lineterminator="\n",
            )
            writer.writeheader()
            for row_number, raw in enumerate(reader, start=2):
                migrated, forceable = _migrate_row(
                    raw,
                    path=source,
                    row_number=row_number,
                )
                writer.writerow(migrated)
                rows += 1
                forceable_rows += int(forceable)
                diagnostic_rows += int(not forceable)
    if rows == 0:
        raise ValueError(f"{source}: aggregate contains no rows")
    return MigratedAggregate(rows, forceable_rows, diagnostic_rows)


def _production_validation_context(
    aggregate: Path,
    timing_sidecar: Path,
) -> CPUPrefillAdapterContext:
    """Build non-semantic provenance needed to invoke production validation."""

    return CPUPrefillAdapterContext(
        profile=MeasurementProfile.PRODUCTION,
        run_id="cpu-prefill-v12-to-v13-checkpoint-migration",
        corpus_id=raw_corpus_id((aggregate, timing_sidecar)),
        git_revision="checkpoint-source-provenance-retained-in-rows",
        build_id="checkpoint-source-provenance-retained-in-rows",
        compiler_id="checkpoint-source-provenance-retained-in-rows",
        architecture_class="checkpoint-source-provenance-retained-in-rows",
        device_name="checkpoint-source-provenance-retained-in-rows",
        driver_runtime="checkpoint-source-provenance-retained-in-rows",
        serial_m1_policy_hash="sha256:" + "0" * 64,
        raw_timing_sidecar_retained=True,
    )


def _validate_migrated_pair(
    aggregate: Path,
    timing_sidecar: Path,
    expected: MigratedAggregate,
) -> None:
    """Authenticate sidecars and all production promotion invariants."""

    corpus = adapt_cpu_prefill_csv(
        (aggregate,),
        _production_validation_context(aggregate, timing_sidecar),
        timing_sidecars=(timing_sidecar,),
    )
    if len(corpus) != expected.rows:
        raise ValueError(f"{aggregate}: adapter row count changed during migration")
    forceable_rows = sum(int(row.forced_route_ok) for row in corpus)
    if forceable_rows != expected.forceable_rows:
        raise ValueError(
            f"{aggregate}: route identity changed during migration "
            f"({forceable_rows} != {expected.forceable_rows})"
        )


def migrate_cpu_prefill_checkpoint(
    source_dir: Path,
    output_dir: Path,
    target_contract_digest: str,
    *,
    target_inventory: Mapping[str, frozenset[int]] | None = None,
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> dict[str, object]:
    """Validate, translate, and atomically publish one checkpoint directory.

    The output directory may be absent or already created by the refresh
    wrapper, but it must be empty.  No source artifact is modified.
    """

    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    target_contract = _require_sha256(
        "target contract digest", target_contract_digest
    )
    if source == output:
        raise ValueError("source and output checkpoint directories must differ")
    if not source.is_dir():
        raise ValueError(f"source checkpoint directory does not exist: {source}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output checkpoint directory must be empty: {output}")

    source_contract_path = source / TARGET_CONTRACT_FILE
    if not source_contract_path.is_file():
        raise ValueError(f"source checkpoint lacks {TARGET_CONTRACT_FILE}")
    source_contract = _require_sha256(
        "source contract digest",
        source_contract_path.read_text(encoding="utf-8"),
    )
    required_inventory = dict(
        target_inventory
        if target_inventory is not None
        else _canonical_target_inventory(route_manifest)
    )
    if not required_inventory or any(
        not name.startswith("cpu_prefill.")
        or not name.endswith(".csv")
        or not m_values
        or any(m < 2 for m in m_values)
        for name, m_values in required_inventory.items()
    ):
        raise ValueError("target checkpoint inventory is invalid")

    aggregate_paths = sorted(
        path
        for path in source.glob("cpu_prefill.*.csv")
        if not path.name.endswith(".timing.csv")
    )
    timing_paths = sorted(source.glob("cpu_prefill.*.timing.csv"))
    if not aggregate_paths:
        raise ValueError("source checkpoint has no finalized aggregate shards")
    expected_timing = {
        path.with_name(path.name.removesuffix(".csv") + ".timing.csv")
        for path in aggregate_paths
    }
    if set(timing_paths) != expected_timing:
        missing = sorted(path.name for path in expected_timing - set(timing_paths))
        orphaned = sorted(path.name for path in set(timing_paths) - expected_timing)
        raise ValueError(
            "source checkpoint aggregate/timing pairs disagree: "
            f"missing={missing[:3]} orphaned={orphaned[:3]}"
        )

    compatible_aggregates: list[Path] = []
    omitted_incompatible: list[dict[str, object]] = []
    for path in aggregate_paths:
        observed_m = _aggregate_m_inventory(path)
        required_m = required_inventory.get(path.name)
        if required_m is None:
            omitted_incompatible.append({
                "aggregate": path.name,
                "reason": "not-present-in-target-plan",
                "source_m_values": sorted(observed_m),
                "target_m_values": [],
            })
        elif observed_m != required_m:
            omitted_incompatible.append({
                "aggregate": path.name,
                "reason": "complete-m-inventory-changed",
                "source_m_values": sorted(observed_m),
                "target_m_values": sorted(required_m),
            })
        else:
            compatible_aggregates.append(path)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.migrating-",
        dir=output.parent,
    ))
    migrated_counts = MigratedAggregate(0, 0, 0)
    try:
        for source_aggregate in compatible_aggregates:
            source_timing = source_aggregate.with_name(
                source_aggregate.name.removesuffix(".csv") + ".timing.csv"
            )
            staged_aggregate = stage / source_aggregate.name
            staged_timing = stage / source_timing.name
            counts = _migrate_aggregate(source_aggregate, staged_aggregate)
            shutil.copyfile(source_timing, staged_timing)
            if staged_timing.read_bytes() != source_timing.read_bytes():
                raise ValueError(f"{source_timing}: timing sidecar copy changed bytes")
            _validate_migrated_pair(staged_aggregate, staged_timing, counts)
            migrated_counts = MigratedAggregate(
                migrated_counts.rows + counts.rows,
                migrated_counts.forceable_rows + counts.forceable_rows,
                migrated_counts.diagnostic_rows + counts.diagnostic_rows,
            )

        ignored_inprogress = sorted(
            path.name for path in source.glob("cpu_prefill.*.inprogress")
        )
        source_corpus_paths = [*aggregate_paths, *timing_paths]
        target_corpus_paths = [
            *(stage / path.name for path in compatible_aggregates),
            *(
                stage / (path.name.removesuffix(".csv") + ".timing.csv")
                for path in compatible_aggregates
            ),
        ]
        manifest: dict[str, object] = {
            "schema": "cpu-prefill-checkpoint-migration-v2",
            "source_directory": str(source),
            "source_contract_digest": source_contract,
            "target_contract_digest": target_contract,
            "accepted_source_timing_protocols": [
                LEGACY_TIMING_PROTOCOL,
                LEGACY_ADAPTIVE_TIMING_PROTOCOL,
            ],
            "target_timing_protocol": LEGACY_ADAPTIVE_TIMING_PROTOCOL,
            "target_timing_ceiling_policy": (
                LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
            ),
            "source_aggregate_shards": len(aggregate_paths),
            "target_plan_shards": len(required_inventory),
            "aggregate_shards": len(compatible_aggregates),
            "timing_sidecars": len(compatible_aggregates),
            "rows": migrated_counts.rows,
            "forceable_rows": migrated_counts.forceable_rows,
            "diagnostic_rows": migrated_counts.diagnostic_rows,
            "ignored_inprogress_files": ignored_inprogress,
            "omitted_incompatible_finalized_shards": omitted_incompatible,
            "source_corpus_digest": _corpus_digest(source_corpus_paths, source),
            "target_corpus_digest": _corpus_digest(target_corpus_paths, stage),
        }
        (stage / TARGET_CONTRACT_FILE).write_text(
            target_contract + "\n",
            encoding="utf-8",
        )
        (stage / MIGRATION_MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        if output.exists():
            output.rmdir()
        os.replace(stage, output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def main() -> int:
    """Parse the explicit migration transaction and report its evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--target-contract-digest", required=True)
    parser.add_argument(
        "--route-manifest",
        action="append",
        type=Path,
        default=[],
        help="C++ serial-route CSV; repeat once per ISA regime",
    )
    args = parser.parse_args()

    route_manifest = (
        read_cpu_prefill_route_manifests(args.route_manifest)
        if args.route_manifest
        else None
    )

    manifest = migrate_cpu_prefill_checkpoint(
        args.source_dir,
        args.output_dir,
        args.target_contract_digest,
        route_manifest=route_manifest,
    )
    print(
        "migrated CPU prefill checkpoint: "
        f"shards={manifest['aggregate_shards']} rows={manifest['rows']} "
        f"forceable={manifest['forceable_rows']} "
        f"diagnostic={manifest['diagnostic_rows']} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
