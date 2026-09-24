#!/usr/bin/env python3
"""Reparse authenticated CUDA batch reports without launching any device work.

Collector v7 and earlier interpreted Nsight Kbyte/Mbyte resource units as IEC
prefixes. This offline transaction preserves the original evidence, commands,
timings, request identities, raw artifacts and stream attribution. It publishes
a distinct current-generation manifest only after all raw bytes authenticate
and only byte-valued metrics change. It never resumes or truncates a live
journal and never calls the profiler, a trainer, or a GPU scorer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Sequence

from .profiler_collectors import (
    _parse_ncu_csv_with_stream_ids,
    _partition_cuda_batch_dispatches,
    _sha256_files,
)
from .profiler_evidence import (
    PROFILER_COLLECTOR_VERSION,
    ProfilerEvidence,
    ProfilerEvidenceManifest,
    ProfilerEvidenceStatus,
    ProfilerRequest,
    _offline_worker_count,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
    validate_profiler_evidence_coverage,
    write_profiler_evidence_manifest,
)
from .schema import Backend


@dataclass(frozen=True)
class ReparseReport:
    """Bind the old/new interpretation to the same exact physical evidence."""

    source_evidence_digest: str
    output_evidence_digest: str
    evidence_count: int
    batch_count: int
    changed_metric_count: int


class ReparseOutputPolicy(Enum):
    """Separate fresh publication from authenticated crash-resume reuse."""

    REQUIRE_NEW = "require_new"
    REUSE_IDENTICAL = "reuse_identical"


@dataclass(frozen=True)
class _Batch:
    """One independently hashed physical report and its ordered request IDs."""

    directory: Path
    plan_digest: str
    request_ids: tuple[str, ...]


# Read-only, fork-inherited evidence avoids sending a full manifest per batch.
_REQUESTS: dict[str, ProfilerRequest] = {}
_SOURCE: dict[str, ProfilerEvidence] = {}
_RAW_ROOT = Path()


def _member_directory(request_id: str, expected_digest: str) -> Path:
    """Resolve exactly one retained attempt by bytes, never by latest filename."""

    matches = [
        directory for directory in sorted((_RAW_ROOT / request_id).glob("attempt-*"))
        if directory.is_dir() and not directory.is_symlink()
        and _sha256_files(directory) == expected_digest
    ]
    if len(matches) != 1:
        raise ValueError(f"{request_id}: missing or ambiguous authenticated raw attempt")
    return matches[0]


def _unit_only_record(
    source: ProfilerEvidence, dispatches: tuple,
) -> tuple[ProfilerEvidence, int]:
    """Reject arithmetic, attribution, timing, or non-byte metric changes."""

    if len(source.dispatches) != len(dispatches):
        raise ValueError("reparse changed the physical dispatch count")
    changes = 0
    for old, new in zip(source.dispatches, dispatches):
        if replace(new, metrics=old.metrics) != old:
            raise ValueError("reparse changed physical dispatch identity")
        if len(old.metrics) != len(new.metrics):
            raise ValueError("reparse changed the metric inventory")
        for before, after in zip(old.metrics, new.metrics):
            if before == after:
                continue
            byte_metric = (
                before.metric_id.endswith("_bytes")
                or "memory_bytes" in before.metric_id
            )
            if not byte_metric or replace(after, value=before.value) != before:
                raise ValueError(f"reparse changed non-byte evidence: {before.metric_id}")
            changes += 1
    return replace(source, dispatches=dispatches,
                   collector_version=PROFILER_COLLECTOR_VERSION), changes


def _reparse_batch(batch: _Batch) -> tuple[tuple[ProfilerEvidence, ...], int]:
    """Authenticate one batch and rejoin its immutable per-request streams."""

    members = []
    for request_id in batch.request_ids:
        source = _SOURCE[request_id]
        directory = _member_directory(request_id, source.raw_artifact_digest)
        member = json.loads((directory / "batch-member.json").read_text())
        if (member["request_id"] != request_id
                or member["batch_plan_digest"] != batch.plan_digest
                or member["target_launches"] != 1
                or member["physical_dispatch_count"] != len(source.dispatches)):
            raise ValueError(f"{request_id}: raw batch membership does not match evidence")
        members.append(member)
    expected_hashes = {member["batch_raw_artifact_digest"] for member in members}
    if expected_hashes != {_sha256_files(batch.directory)}:
        raise ValueError("retained CUDA batch raw artifact digest mismatch")
    members.sort(key=lambda member: member["request_launch_order"])
    if [member["request_launch_order"] for member in members] != list(range(len(members))):
        raise ValueError("retained CUDA batch launch order is incomplete")
    streams = tuple((member["request_id"], member["stream_id"]) for member in members)
    records = _parse_ncu_csv_with_stream_ids(
        batch.directory / "ncu-raw.csv", _REQUESTS[batch.request_ids[0]],
    )
    partitioned = _partition_cuda_batch_dispatches(records, streams)
    changed, result = 0, []
    for request_id in batch.request_ids:
        updated, count = _unit_only_record(_SOURCE[request_id], partitioned[request_id])
        updated.validate(_REQUESTS[request_id])
        result.append(updated)
        changed += count
    return tuple(result), changed


def reparse_cuda_evidence(
    *, requests_path: Path, source_path: Path, raw_directory: Path,
    output_path: Path, workers: int | None = None,
    output_policy: ReparseOutputPolicy = ReparseOutputPolicy.REQUIRE_NEW,
) -> ReparseReport:
    """Publish a distinct normalized manifest from a completed CUDA transaction.

    Every source record must be terminal and every successful request must be
    covered exactly once by a complete authenticated batch. Failed, missing,
    unbatched, foreign, tampered, or ambiguously duplicated records fail hard.
    The original manifest and journal remain byte-for-byte untouched.
    """

    if output_path.resolve() == source_path.resolve() or (
            output_path.exists() and output_policy == ReparseOutputPolicy.REQUIRE_NEW):
        raise ValueError("reparse requires a new output path; preserve the source evidence")
    requests = read_profiler_request_manifest(requests_path)
    source = read_profiler_evidence_manifest(source_path)
    validate_profiler_evidence_coverage(requests, source, require_complete=True)
    if any(request.backend != Backend.CUDA for request in requests.requests):
        raise ValueError("Nsight byte-unit reparse requires an all-CUDA transaction")
    global _REQUESTS, _SOURCE, _RAW_ROOT
    _REQUESTS = {request.request_id: request for request in requests.requests}
    _SOURCE = {record.request_id: record for record in source.evidence}
    _RAW_ROOT = raw_directory
    successful = {key for key, value in _SOURCE.items()
                  if value.status == ProfilerEvidenceStatus.COMPLETE}
    batches, seen = [], set()
    for plan in sorted((raw_directory / "_cuda_process_batches").glob("*/attempt-*/requests.tsv")):
        with plan.open(newline="") as stream:
            ids = tuple(row["request_id"] for row in csv.DictReader(stream, delimiter="\t"))
        if not successful.intersection(ids):
            continue
        if len(ids) != len(set(ids)) or not set(ids) <= successful:
            raise ValueError("raw CUDA batch is not a complete subset of successful requests")
        digest = hashlib.sha256(plan.read_bytes()).hexdigest()
        # Old failed attempts can retain the same plan. Select the directory
        # authenticated by the source member before admitting a batch job.
        first = ids[0]
        member_dir = _member_directory(first, _SOURCE[first].raw_artifact_digest)
        member = json.loads((member_dir / "batch-member.json").read_text())
        if member["batch_plan_digest"] != digest:
            continue
        if seen.intersection(ids):
            raise ValueError("duplicate authenticated CUDA batch coverage")
        seen.update(ids)
        batches.append(_Batch(plan.parent, digest, ids))
    if seen != successful:
        raise ValueError(f"missing authenticated CUDA batch coverage: {len(successful - seen)}")
    physical_limit = _offline_worker_count(len(batches),
        environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS", records_per_worker=1)
    count = min(workers, physical_limit) if workers is not None else physical_limit
    if not batches:
        count = 1
    if count < 1:
        raise ValueError("reparse worker count must be positive")
    results: dict[str, ProfilerEvidence] = {}
    changed = 0

    def accept(result: tuple[tuple[ProfilerEvidence, ...], int]) -> None:
        """Merge only disjoint request-local records in the owning process."""
        nonlocal changed
        records, delta = result
        for record in records:
            if record.request_id in results:
                raise ValueError("reparse produced a duplicate request")
            results[record.request_id] = record
        changed += delta

    if count == 1 or len(batches) <= 1:
        for batch in batches:
            accept(_reparse_batch(batch))
    else:
        with ProcessPoolExecutor(max_workers=min(count, len(batches)),
                mp_context=multiprocessing.get_context("fork")) as pool:
            for result in pool.map(_reparse_batch, batches):
                accept(result)
    for record in source.evidence:
        if record.status == ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED:
            results[record.request_id] = replace(record, collector_version=PROFILER_COLLECTOR_VERSION)
    normalized = replace(source, collector_version=PROFILER_COLLECTOR_VERSION,
                         evidence=tuple(results[item.request_id] for item in source.evidence))
    validate_profiler_evidence_coverage(requests, normalized, require_complete=True)
    if output_path.exists():
        if read_profiler_evidence_manifest(output_path).digest() != normalized.digest():
            raise ValueError("pending reparse output differs from authenticated raw evidence")
    else:
        write_profiler_evidence_manifest(output_path, normalized)
    return ReparseReport(source.digest(), normalized.digest(), len(results), len(batches), changed)


def upgrade_cuda_evidence(
    *, requests_path: Path, source_path: Path, raw_directory: Path,
    workers: int | None = None,
) -> ReparseReport | None:
    """Promote a completed correction atomically, retaining the original inode.

    A live/incremental journal is deliberately not touched. The source remains
    in place until the complete replacement passes validation; a hard-linked
    versioned backup preserves its original bytes after atomic promotion. A
    current-generation manifest is a no-op, so normal fresh runs pay no replay.
    """
    if Path(str(source_path) + ".inprogress.jsonl").exists():
        raise ValueError("cannot upgrade a live or unfinished profiler journal")
    source = read_profiler_evidence_manifest(source_path)
    if source.collector_version == PROFILER_COLLECTOR_VERSION:
        return None
    def identity() -> tuple[int, int, int, int]:
        """Reads can change atime; only inode/extent/content timestamps matter."""
        value = source_path.stat()
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns
    original_identity = identity()
    output = Path(str(source_path) + ".ncu-si-units-v8.json")
    backup = Path(str(source_path) + ".before-ncu-si-units-v8.json")
    report = reparse_cuda_evidence(requests_path=requests_path, source_path=source_path,
        raw_directory=raw_directory, output_path=output, workers=workers,
        output_policy=ReparseOutputPolicy.REUSE_IDENTICAL)
    if identity() != original_identity:
        raise ValueError("source profiler evidence changed during offline reparse")
    if backup.exists():
        if read_profiler_evidence_manifest(backup).digest() != report.source_evidence_digest:
            raise ValueError("existing reparse backup belongs to another source")
    else:
        os.link(source_path, backup)
    # Record lineage before changing the canonical pointer. Both manifests
    # retain all original physical command/raw-artifact digests.
    receipt = Path(str(source_path) + ".ncu-si-units-v8.lineage.json")
    if receipt.exists():
        if json.loads(receipt.read_text()) != asdict(report):
            raise ValueError("existing reparse lineage belongs to another transaction")
    else:
        with receipt.open("x") as stream:
            json.dump(asdict(report), stream, sort_keys=True)
            stream.write("\n")
    os.replace(output, source_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the offline-only CLI; it has no executable or profiler option."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--source-evidence", required=True)
    parser.add_argument("--raw-directory", required=True)
    output = parser.add_mutually_exclusive_group(required=True)
    output.add_argument("--output-evidence")
    output.add_argument("--upgrade-in-place", action="store_true",
                        help="atomically promote and retain a versioned original backup")
    args = parser.parse_args(argv)
    common = dict(requests_path=Path(args.requests), source_path=Path(args.source_evidence),
                  raw_directory=Path(args.raw_directory))
    report = upgrade_cuda_evidence(**common) if args.upgrade_in_place else reparse_cuda_evidence(
        **common, output_path=Path(args.output_evidence))
    print(json.dumps(asdict(report) if report else {"already_current": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
