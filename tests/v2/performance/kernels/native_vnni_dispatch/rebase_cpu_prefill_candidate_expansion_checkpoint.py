"""Rebase authenticated CPU prefill shards onto a newer timing policy.

Long candidate sweeps are durable evidence collections, not disposable jobs.
Changing an adapter threshold or timing-policy implementation may change the
content-addressed plan token even though the measured kernel binary, candidate
registry, and complete measurement-cell inventory are identical.  This module
provides the only supported bridge across that narrow lifecycle boundary.

The source directory is never modified.  Its plan is authenticated using the
implementation identity recorded when collection began. A separately created
target plan must match every measurement-defining field except the timing
implementation digest. A rebuilt harness binary also requires an explicit
review note; the source and target build digests plus that note are persisted
in the rebase manifest, while any other experiment mismatch remains fatal.
Each complete source aggregate/timing pair is then
checked for its exact candidate/cell inventory and replayed through the current
production adapter.  Accepted bytes are copied to temporary target files,
validated a second time, and atomically published under the target plan token.
Rejected or incomplete pairs remain absent so the normal collector measures
only those cells again.  An atomic manifest records every publication and every
cell left for recollection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Callable

from .cpu_prefill_candidate_expansion import (
    CPUPrefillCandidateExpansionPlan,
    CPUPrefillCandidateExpansionRecord,
    cpu_prefill_candidate_expansion_plan_token,
    read_cpu_prefill_candidate_expansion_plan,
    _sha256_file,
)
from .validate_cpu_prefill_partial import validate_cpu_prefill_partial


CPU_PREFILL_CANDIDATE_EXPANSION_REBASE_SCHEMA = (
    "cpu-prefill-candidate-expansion-checkpoint-rebase-v1"
)
DEFAULT_PLAN_NAME = "cpu_prefill_candidate_expansion.v4.json"
REBASE_WORKERS_ENVIRONMENT = "LLAMINAR_NATIVE_VNNI_REBASE_WORKERS"
DEFAULT_MAXIMUM_REBASE_WORKERS = 32

_MEASUREMENT_COMPATIBILITY_FIELDS = (
    "schema_version",
    "source_aggregate_sha256",
    "source_timing_sha256",
    "candidate_registry_digest",
    "collection_build_digest",
    "source_cell_count",
    "source_cell_digest",
    "source_candidate_ids",
    "anchor_candidate_ids",
    "expansion_candidate_ids",
)

_EXPECTED_ISA = {
    "avx2-build.avx2-runtime": ("AVX2", "AVX2"),
    "avx512-build.avx2-runtime": ("AVX512", "AVX2"),
    "avx512-build.avx512-runtime": ("AVX512", "AVX512"),
}


def _record_key(record: CPUPrefillCandidateExpansionRecord) -> str:
    """Return a stable human-readable identity for one launch record."""

    return "/".join((
        record.source_format,
        record.shape_name,
        record.isa_regime,
    ))


def _partial_paths(
    directory: Path,
    token: str,
    record: CPUPrefillCandidateExpansionRecord,
) -> tuple[Path, Path]:
    """Map one authenticated plan record to its aggregate/timing pair."""

    stem = (
        f"cpu_prefill.candidate-expansion.{token}."
        f"{record.source_format}.{record.shape_name}.{record.isa_regime}"
    )
    aggregate = Path(directory) / f"{stem}.csv"
    return aggregate, Path(directory) / f"{stem}.timing.csv"


def _require_compatible_plans(
    source: CPUPrefillCandidateExpansionPlan,
    target: CPUPrefillCandidateExpansionPlan,
    *,
    harness_only_build_change_audit: str | None = None,
) -> None:
    """Reject any rebase that could change the measured experiment.

    The target may widen a historical selected cohort, but every inherited
    record must be byte-for-byte identical and the source records must be a
    strict subset of the target records.  In particular, widening the static
    split-manifest cohort to the complete immutable development aggregate is
    valid: dynamic generic-refinement geometries already belong to that source
    corpus and must not be discarded merely because their names are absent
    from the static split.  The trainer build digest remains mandatory;
    changing generated code, compiler flags, linked kernels, or the
    measurement binary is a new experiment and cannot inherit an older shard
    unless the caller supplies a recorded audit that the binary delta is
    confined to the timing harness. That escape hatch permits only the build
    digest mismatch; candidate, source, geometry, and route inventories remain
    hard requirements.
    """

    mismatches = [
        field
        for field in _MEASUREMENT_COMPATIBILITY_FIELDS
        if getattr(source, field) != getattr(target, field)
    ]
    if (
        mismatches == ["collection_build_digest"]
        and harness_only_build_change_audit is not None
    ):
        if not harness_only_build_change_audit.strip():
            raise ValueError(
                "CPU prefill harness-only build audit must explain the change"
            )
        mismatches = []
    if mismatches:
        raise ValueError(
            "CPU prefill checkpoint plans describe different measurements: "
            + ", ".join(mismatches)
        )
    source_records = frozenset(source.records)
    target_records = frozenset(target.records)
    if source.selection_policy == target.selection_policy:
        selection_compatible = (
            source.selected_cell_count == target.selected_cell_count
            and source.selected_cell_digest == target.selected_cell_digest
            and source_records == target_records
        )
    else:
        allowed_widenings = {
            (
                "manifest-production-cells-v1",
                "split-development-cells-v1",
            ),
            (
                "manifest-production-cells-v1",
                "all-immutable-source-cells-v1",
            ),
            (
                "split-development-cells-v1",
                "all-immutable-source-cells-v1",
            ),
        }
        selection_compatible = (
            (source.selection_policy, target.selection_policy)
            in allowed_widenings
            and source.selected_cell_count < target.selected_cell_count
            and source_records < target_records
        )
    if not selection_compatible:
        raise ValueError(
            "CPU prefill checkpoint plans describe incompatible selected-cell "
            "inventories"
        )


def _validate_record_inventory(
    aggregate: Path,
    record: CPUPrefillCandidateExpansionRecord,
    plan: CPUPrefillCandidateExpansionPlan,
) -> None:
    """Prove that a shard contains every declared candidate/cell exactly once."""

    try:
        expected_build_isa, expected_runtime_isa = _EXPECTED_ISA[
            record.isa_regime
        ]
    except KeyError as exc:
        raise ValueError(
            f"unsupported CPU prefill ISA regime {record.isa_regime}"
        ) from exc

    expected = {
        (m, candidate_id)
        for m in record.m_values
        for candidate_id in plan.collection_candidate_ids
    }
    observed: set[tuple[int, str]] = set()
    with Path(aggregate).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "backend",
            "phase",
            "source_format",
            "shape",
            "m",
            "n",
            "k",
            "candidate_id",
            "build_isa",
            "runtime_isa_requested",
            "runtime_isa_effective",
            "threads",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"{aggregate}: missing inventory columns {sorted(missing)}"
            )
        for row_number, row in enumerate(reader, start=2):
            actual_surface = (
                row["backend"].strip().lower(),
                row["phase"].strip(),
                row["source_format"].strip().upper(),
                row["shape"].strip(),
                int(row["n"]),
                int(row["k"]),
                row["build_isa"].strip().upper(),
                row["runtime_isa_requested"].strip().upper(),
                row["runtime_isa_effective"].strip().upper(),
                int(row["threads"]),
            )
            expected_surface = (
                "cpu",
                "prefill_gemm",
                record.source_format,
                record.shape_name,
                record.n,
                record.k,
                expected_build_isa,
                expected_runtime_isa,
                expected_runtime_isa,
                record.threads,
            )
            if actual_surface != expected_surface:
                raise ValueError(
                    f"{aggregate}:{row_number}: record surface disagrees with plan"
                )
            identity = (int(row["m"]), row["candidate_id"].strip())
            if identity in observed:
                raise ValueError(
                    f"{aggregate}:{row_number}: duplicate M/candidate row {identity}"
                )
            observed.add(identity)
    if observed != expected:
        missing = sorted(expected.difference(observed))
        unexpected = sorted(observed.difference(expected))
        raise ValueError(
            f"{aggregate}: candidate/cell inventory differs from plan; "
            f"missing={missing[:4]} unexpected={unexpected[:4]}"
        )


def _copy_file_durably(source: Path, target: Path) -> None:
    """Copy one immutable evidence file and flush it before publication."""

    shutil.copyfile(source, target)
    with target.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_json_atomically(path: Path, payload: dict[str, object]) -> None:
    """Publish a durable JSON manifest without exposing partial contents."""

    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def cpu_prefill_candidate_expansion_rebase_worker_count(
    record_count: int,
    environment: dict[str, str] | None = None,
) -> int:
    """Return the bounded process width for independent shard validation.

    Parsing retained timing sidecars is Python-heavy work and therefore does
    not benefit from threads. A process per active shard exposes both CPU
    sockets while the conservative default cap avoids turning a checkpoint
    migration into an unbounded burst of filesystem readers. The environment
    override lets slower storage request a lower queue depth, but it may never
    disable all validation workers.
    """

    if record_count <= 0:
        return 1
    source = os.environ if environment is None else environment
    configured = source.get(REBASE_WORKERS_ENVIRONMENT, "").strip()
    if configured:
        try:
            maximum = int(configured)
        except ValueError as error:
            raise ValueError(
                f"{REBASE_WORKERS_ENVIRONMENT} must be a positive integer"
            ) from error
        if maximum <= 0:
            raise ValueError(
                f"{REBASE_WORKERS_ENVIRONMENT} must be a positive integer"
            )
    else:
        maximum = min(
            DEFAULT_MAXIMUM_REBASE_WORKERS,
            os.cpu_count() or 1,
        )
    return min(record_count, maximum)


def _rebase_record(
    source_directory: Path,
    target_directory: Path,
    source_token: str,
    target_token: str,
    source_plan: CPUPrefillCandidateExpansionPlan,
    target_plan: CPUPrefillCandidateExpansionPlan,
    source_records: frozenset[CPUPrefillCandidateExpansionRecord],
    record: CPUPrefillCandidateExpansionRecord,
    validator: Callable[..., None],
) -> tuple[str, dict[str, object]]:
    """Validate and publish one independent record transaction.

    Every record has unique source and target paths, so this function can run
    in a child process. Publication still uses durable temporary files followed
    by atomic replacement. A rejected or incomplete source pair becomes a
    recollection obligation without invalidating unrelated evidence.
    """

    key = _record_key(record)
    if record not in source_records:
        return "recollect", {
            "record": key,
            "reason": "target_record_not_in_source_plan",
        }
    source_aggregate, source_timing = _partial_paths(
        source_directory, source_token, record
    )
    target_aggregate, target_timing = _partial_paths(
        target_directory, target_token, record
    )
    if not source_aggregate.is_file() or not source_timing.is_file():
        return "recollect", {
            "record": key,
            "reason": "source_pair_incomplete",
        }
    try:
        _validate_record_inventory(source_aggregate, record, source_plan)
        validator(
            source_aggregate,
            source_timing,
            candidate_expansion=True,
        )
    except (OSError, ValueError) as exc:
        return "recollect", {
            "record": key,
            "reason": "current_policy_rejected",
            "detail": str(exc),
        }

    source_aggregate_sha = _sha256_file(source_aggregate)
    source_timing_sha = _sha256_file(source_timing)
    target_state = (target_aggregate.exists(), target_timing.exists())
    if any(target_state):
        if not all(target_state):
            raise ValueError(
                f"target checkpoint has an incomplete pair for {key}"
            )
        if (
            _sha256_file(target_aggregate) != source_aggregate_sha
            or _sha256_file(target_timing) != source_timing_sha
        ):
            raise ValueError(
                f"target checkpoint already contains different bytes for {key}"
            )
        _validate_record_inventory(target_aggregate, record, target_plan)
        validator(
            target_aggregate,
            target_timing,
            candidate_expansion=True,
        )
        status = "already_published"
    else:
        aggregate_temporary = target_aggregate.with_name(
            f"{target_aggregate.name}.{os.getpid()}.rebase.tmp"
        )
        timing_temporary = target_timing.with_name(
            f"{target_timing.name}.{os.getpid()}.rebase.tmp"
        )
        try:
            _copy_file_durably(source_aggregate, aggregate_temporary)
            _copy_file_durably(source_timing, timing_temporary)
            _validate_record_inventory(
                aggregate_temporary, record, target_plan
            )
            validator(
                aggregate_temporary,
                timing_temporary,
                candidate_expansion=True,
            )
            os.replace(aggregate_temporary, target_aggregate)
            os.replace(timing_temporary, target_timing)
        finally:
            aggregate_temporary.unlink(missing_ok=True)
            timing_temporary.unlink(missing_ok=True)
        status = "published"

    return "published", {
        "record": key,
        "status": status,
        "aggregate_sha256": source_aggregate_sha,
        "timing_sha256": source_timing_sha,
        "m_values": list(record.m_values),
    }


def _production_rebase_record(
    task: tuple[
        Path,
        Path,
        str,
        str,
        CPUPrefillCandidateExpansionPlan,
        CPUPrefillCandidateExpansionPlan,
        frozenset[CPUPrefillCandidateExpansionRecord],
        CPUPrefillCandidateExpansionRecord,
    ],
) -> tuple[str, dict[str, object]]:
    """Process-pool entry point using the canonical current validator."""

    return _rebase_record(*task, validator=validate_cpu_prefill_partial)


def rebase_cpu_prefill_candidate_expansion_checkpoint(
    source_directory: Path,
    target_directory: Path,
    target_plan_path: Path,
    *,
    source_plan_path: Path | None = None,
    validator: Callable[..., None] = validate_cpu_prefill_partial,
    harness_only_build_change_audit: str | None = None,
) -> Path:
    """Revalidate and republish every compatible completed checkpoint shard.

    ``validator`` exists to let lifecycle unit tests isolate transaction
    behavior. Production callers use the default current adapter, including
    raw timing-sidecar authentication and installable timing thresholds.
    ``harness_only_build_change_audit`` is deliberately explicit and persisted;
    omitting it preserves the ordinary exact-binary requirement.
    """

    source_directory = Path(source_directory).resolve()
    target_directory = Path(target_directory).resolve()
    target_plan_path = Path(target_plan_path).resolve()
    source_plan_path = (
        Path(source_plan_path).resolve()
        if source_plan_path is not None
        else source_directory / DEFAULT_PLAN_NAME
    )
    if source_directory == target_directory:
        raise ValueError(
            "checkpoint rebase requires distinct source and target directories"
        )

    source_plan = read_cpu_prefill_candidate_expansion_plan(
        source_plan_path,
        authenticate_current_implementation=False,
    )
    target_plan = read_cpu_prefill_candidate_expansion_plan(target_plan_path)
    _require_compatible_plans(
        source_plan,
        target_plan,
        harness_only_build_change_audit=harness_only_build_change_audit,
    )

    source_token = cpu_prefill_candidate_expansion_plan_token(source_plan_path)
    target_token = cpu_prefill_candidate_expansion_plan_token(target_plan_path)
    target_directory.mkdir(parents=True, exist_ok=True)
    source_records = frozenset(source_plan.records)
    tasks = tuple(
        (
            source_directory,
            target_directory,
            source_token,
            target_token,
            source_plan,
            target_plan,
            source_records,
            record,
        )
        for record in target_plan.records
    )
    production_validation = validator is validate_cpu_prefill_partial
    worker_count = (
        cpu_prefill_candidate_expansion_rebase_worker_count(len(tasks))
        if production_validation
        else 1
    )
    if production_validation:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            results = tuple(executor.map(_production_rebase_record, tasks))
    else:
        # Injected validators are test seams and may be closures that cannot be
        # pickled. Keeping this path serial also gives unit tests deterministic
        # callback ordering without weakening production parallelism.
        results = tuple(
            _rebase_record(*task, validator=validator) for task in tasks
        )
    published = [payload for kind, payload in results if kind == "published"]
    recollect = [payload for kind, payload in results if kind == "recollect"]

    manifest_path = target_directory / (
        "cpu_prefill_candidate_expansion_rebase."
        f"{source_token}.to.{target_token}.json"
    )
    _write_json_atomically(manifest_path, {
        "schema_version": CPU_PREFILL_CANDIDATE_EXPANSION_REBASE_SCHEMA,
        "source_directory": str(source_directory),
        "target_directory": str(target_directory),
        "source_plan_path": str(source_plan_path),
        "target_plan_path": str(target_plan_path),
        "source_plan_digest": source_plan.digest(),
        "target_plan_digest": target_plan.digest(),
        "source_plan_token": source_token,
        "target_plan_token": target_token,
        "source_implementation_digest": source_plan.implementation_digest,
        "target_implementation_digest": target_plan.implementation_digest,
        "source_collection_build_digest": source_plan.collection_build_digest,
        "target_collection_build_digest": target_plan.collection_build_digest,
        "build_compatibility": (
            "audited-harness-only-change-v1"
            if source_plan.collection_build_digest
            != target_plan.collection_build_digest
            else "exact"
        ),
        "harness_only_build_change_audit": harness_only_build_change_audit,
        "measurement_compatibility_fields": list(
            _MEASUREMENT_COMPATIBILITY_FIELDS
        ),
        "validation_workers": worker_count,
        "selection_compatibility": (
            "exact"
            if source_plan.selection_policy == target_plan.selection_policy
            else (
                f"{source_plan.selection_policy}-to-"
                f"{target_plan.selection_policy}"
            )
        ),
        "published_count": len(published),
        "recollect_count": len(recollect),
        "published": published,
        "recollect": recollect,
    })
    return manifest_path


def main() -> int:
    """Parse paths and run one fail-closed checkpoint-rebase transaction."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--target-directory", type=Path, required=True)
    parser.add_argument("--target-plan", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path)
    parser.add_argument(
        "--harness-only-build-change-audit",
        help=(
            "Explicit review note permitting only collection_build_digest to "
            "change; the note and both digests are persisted in the manifest"
        ),
    )
    args = parser.parse_args()
    manifest = rebase_cpu_prefill_candidate_expansion_checkpoint(
        args.source_directory,
        args.target_directory,
        args.target_plan,
        source_plan_path=args.source_plan,
        harness_only_build_change_audit=(
            args.harness_only_build_change_audit
        ),
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    print(
        "rebased "
        f"{raw['published_count']} checkpoint shard(s); "
        f"{raw['recollect_count']} shard(s) require collection -> {manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
