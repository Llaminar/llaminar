"""Provenance-bound profiler requests and evidence for NativeVNNI training.

Canonical latency collection and hardware-counter collection are deliberately
different transactions.  The timing trainer first emits immutable common
observations.  This module retains every timing observation, then derives exact
profiler requests for the fastest, middle, and slowest timing 5% in every
format/shape/work-size contest.  The separate profiler process can therefore
teach the learner why candidates win or lose without repeating counters for
the uninformative interior of every timing rank.  Nsight Compute replay,
rocprofiler counter passes, and Linux ``perf`` instrumentation can never
perturb the latency samples used to choose or certify a dispatch policy.

A candidate may execute more than one device kernel (for example a CUDA K-part
producer followed by an ordered reducer).  Profiler evidence therefore owns an
ordered list of dispatches rather than pretending that one policy candidate is
always one physical kernel.  Every dispatch carries the complete backend metric
inventory; unsupported optional metrics are explicit records with reasons and
are never represented by a missing JSON field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import mmap
import multiprocessing
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_observation import (
    read_observation_csv,
    read_observation_rows,
    write_observation_csv,
)
from .candidate_registry import (
    CandidateRegistry,
    candidate_registry_digest,
    cpu_native_vnni_decode_registry,
    cpu_native_vnni_prefill_registry,
    cpu_native_vnni_verifier_registry,
    cuda_native_vnni_gemv_registry,
    rocm_moe_grouped_prefill_registry,
    rocm_native_vnni_decode_formula_registry,
    rocm_native_vnni_decode_registry,
)
from .corpus import ObservationCorpus
from .schema import (
    Backend,
    COMPATIBLE_OBSERVATION_LEARNER_VERSIONS,
    COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS,
    ExecutionMode,
    FEATURE_SCHEMA_VERSION,
    LEARNER_VERSION,
    NativeVNNIObservation,
    OBSERVATION_COLUMNS,
    POLICY_ABI,
    SCHEMA_VERSION,
    SemanticContract,
)


LEGACY_PROFILER_REQUEST_SCHEMA_VERSION = "native-vnni-profiler-request-v3"
EXHAUSTIVE_PROFILER_REQUEST_SCHEMA_VERSION = (
    "native-vnni-profiler-request-v4-exact-point"
)
PROFILER_REQUEST_SCHEMA_VERSION = (
    "native-vnni-profiler-request-v5-stratified-exact-point"
)
SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS = frozenset({
    LEGACY_PROFILER_REQUEST_SCHEMA_VERSION,
    EXHAUSTIVE_PROFILER_REQUEST_SCHEMA_VERSION,
    PROFILER_REQUEST_SCHEMA_VERSION,
})
PROFILER_TIMING_STRATUM_FRACTION = 0.05
PROFILER_EVIDENCE_SCHEMA_VERSION = "native-vnni-profiler-evidence-v1"
PROFILER_METRIC_SET_VERSION = "native-vnni-profiler-metrics-v1"
PROFILER_COLLECTOR_VERSION = (
    "native-vnni-isolated-profiler-v8-ncu-si-byte-units"
)
SUPPORTED_PROFILER_COLLECTOR_VERSIONS = frozenset({
    "native-vnni-isolated-profiler-v1",
    "native-vnni-isolated-profiler-v2",
    "native-vnni-isolated-profiler-v3-direct-per-tid-batched",
    "native-vnni-isolated-profiler-v4-direct-per-tid-gpu-batched",
    "native-vnni-isolated-profiler-v5-direct-per-tid-gpu-stream-batched",
    "native-vnni-isolated-profiler-v6-rocm-instruction-work",
    "native-vnni-isolated-profiler-v7-rocm-exact-process-batches",
    PROFILER_COLLECTOR_VERSION,
})
PROFILE_PROTOCOL = "isolated-production-candidate-launch-v1"


def _sha256_json(value: Any) -> str:
    """Return a stable SHA-256 identity for one JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


_PARALLEL_MANIFEST_RECORDS: tuple[Any, ...] = ()
_PARALLEL_REQUEST_VALIDATION_RECORDS: tuple[Any, ...] = ()
_PARALLEL_COVERAGE_PAIRS: tuple[tuple[Any, Any], ...] = ()
_PARALLEL_PROFILER_FEATURE_ROWS: tuple[Any, ...] = ()
_PARALLEL_PROFILER_FEATURE_FIELDNAMES: tuple[str, ...] = ()
_PARALLEL_PROFILER_FEATURE_METRIC_IDS: tuple[str, ...] = ()
_PARALLEL_EXPORT_REQUESTS: dict[str, Mapping[str, Any]] = {}
_PARALLEL_EXPORT_EVIDENCE: tuple[Mapping[str, Any], ...] = ()
_PARALLEL_EXPORT_OBSERVATIONS: dict[str, NativeVNNIObservation] = {}
_PARALLEL_EXPORT_OBSERVATION_ROWS: tuple[NativeVNNIObservation, ...] = ()
_PARALLEL_REQUEST_DECODE_RECORDS: tuple[Mapping[str, Any], ...] = ()
_PARALLEL_PROFILER_GROUP_CORPUS: ObservationCorpus | None = None
_PARALLEL_PROFILER_RUNTIME_GROUPS: tuple[tuple[Any, ...], ...] = ()
_PARALLEL_EVIDENCE_DECODE_RECORDS: tuple[Mapping[str, Any], ...] = ()
_PARALLEL_REQUEST_BUILD_OBSERVATIONS: tuple[NativeVNNIObservation, ...] = ()


def _parse_manifest_array_range(
    task: tuple[Path, int, int],
) -> tuple[Mapping[str, Any], ...]:
    """Decode one canonical manifest-array byte range in a worker process."""

    path, begin, end = task
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as document:
            encoded = document[begin:end].strip(b",")
    decoded = json.loads(b"[" + encoded + b"]")
    if not isinstance(decoded, list) or not all(
        isinstance(record, Mapping) for record in decoded
    ):
        raise ValueError("profiler manifest records must be JSON objects")
    return tuple(decoded)


def _parse_pretty_manifest_array_range(
    task: tuple[Path, int, int],
) -> tuple[Mapping[str, Any], ...]:
    """Decode one legacy indented manifest range in a worker process.

    Older profiler collectors published semantically canonical manifests with
    ``indent=2``. A multi-gigabyte evidence file must not be copied into one
    Python string and decoded under one GIL merely because its presentation
    contains whitespace. The parent supplies ranges that begin at a top-level
    record and end immediately before another top-level record or the array
    close. Trimming a trailing separator turns each range back into a valid
    standalone JSON array without touching nested dispatch or metric objects.
    """

    path, begin, end = task
    with path.open("rb") as handle:
        handle.seek(begin)
        encoded = handle.read(end - begin).strip()
    if encoded.endswith(b","):
        encoded = encoded[:-1].rstrip()
    decoded = json.loads(b"[" + encoded + b"]")
    if not isinstance(decoded, list) or not all(
        isinstance(record, Mapping) for record in decoded
    ):
        raise ValueError("profiler manifest records must be JSON objects")
    return tuple(decoded)


def _read_pretty_manifest_document(
    path: Path,
    array_name: str,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> dict[str, Any] | None:
    """Parse and authenticate an older indented manifest in parallel.

    The legacy collector used deterministic ``sort_keys=True, indent=2`` JSON.
    Top-level array members therefore start two spaces deeper than the root
    member, while every nested object starts deeper still. Those indentation
    boundaries let physical-core workers decode disjoint record ranges without
    a corpus-sized UTF-8 string or serial ``json.loads`` call.

    Authentication remains semantic and presentation-independent: after
    parallel decode, the existing canonical manifest reducer hashes the compact
    sorted-key representation and compares it with the retained root digest.
    Hand-authored JSON that does not match the deterministic indented layout
    returns ``None`` and remains on the compatibility parser.
    """

    member = json.dumps(array_name, separators=(",", ":")).encode()
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as document:
            member_offset = document.find(member)
            if member_offset < 0:
                return None
            line_begin = document.rfind(b"\n", 0, member_offset) + 1
            member_indent = bytes(document[line_begin:member_offset])
            if not member_indent or member_indent.strip():
                return None
            colon = document.find(b":", member_offset + len(member))
            array_open = document.find(b"[", colon + 1 if colon >= 0 else 0)
            if colon < 0 or array_open < 0:
                return None
            close_marker = b"\n" + member_indent + b"]"
            array_end = document.find(close_marker, array_open + 1)
            if array_end < 0:
                return None

            scalar_document = (
                document[: array_open + 1] + document[array_end:]
            )
            try:
                root = json.loads(scalar_document)
            except json.JSONDecodeError:
                return None
            if not isinstance(root, dict) or root.get(array_name) != []:
                return None
            declared_count_name = (
                "request_count" if array_name == "requests" else "evidence_count"
            )
            try:
                declared_count = int(root[declared_count_name])
                expected_digest = str(root["manifest_digest"])
            except (KeyError, TypeError, ValueError):
                return None

            content_begin = array_open + 1
            while content_begin < array_end and document[content_begin] in b" \t\r\n":
                content_begin += 1
            if content_begin == array_end:
                records: tuple[Mapping[str, Any], ...] = ()
            else:
                record_indent = member_indent + b"  "
                if document[content_begin] != ord("{"):
                    return None
                record_start = b"\n" + record_indent + b"{"
                if workers is None:
                    worker_count = _offline_worker_count(
                        declared_count,
                        environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
                    )
                else:
                    if workers < 1:
                        raise ValueError(
                            "profiler manifest parse workers must be positive"
                        )
                    worker_count = min(
                        workers,
                        _physical_core_count(),
                        max(1, declared_count),
                    )
                if declared_count < parallel_threshold:
                    worker_count = 1

                boundaries = [content_begin]
                for worker_index in range(1, worker_count):
                    target = content_begin + (
                        (array_end - content_begin) * worker_index // worker_count
                    )
                    separator = document.find(record_start, target, array_end)
                    if separator < 0:
                        return None
                    boundaries.append(separator + 1 + len(record_indent))
                boundaries.append(array_end)
                boundaries = sorted(set(boundaries))
                tasks = tuple(
                    (path, begin, end)
                    for begin, end in zip(boundaries, boundaries[1:])
                    if begin < end
                )
                if len(tasks) == 1:
                    partitions = (_parse_pretty_manifest_array_range(tasks[0]),)
                else:
                    with ProcessPoolExecutor(
                        max_workers=len(tasks),
                        mp_context=multiprocessing.get_context("fork"),
                    ) as executor:
                        partitions = tuple(executor.map(
                            _parse_pretty_manifest_array_range,
                            tasks,
                        ))
                records = tuple(
                    record for partition in partitions for record in partition
                )

    if len(records) != declared_count:
        raise ValueError(f"profiler {array_name} count does not match inventory")
    scalar_fields = {
        name: value
        for name, value in root.items()
        if name not in {array_name, "manifest_digest"}
    }
    actual_digest = _sha256_manifest_records(
        scalar_fields,
        array_name,
        records,
    )
    if actual_digest != expected_digest:
        manifest_label = "request" if array_name == "requests" else "evidence"
        raise ValueError(
            f"profiler {manifest_label} manifest digest does not match contents"
        )
    root[array_name] = list(records)
    return root


def _read_canonical_manifest_document(
    path: Path,
    array_name: str,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> dict[str, Any] | None:
    """Parse and authenticate our compact manifest encoding in parallel.

    NativeVNNI manifests are intentionally emitted as compact, key-sorted JSON.
    That gives the reader two useful properties: the record array has an exact
    byte delimiter, and deleting the root ``manifest_digest`` member recreates
    the historical canonical digest payload byte-for-byte. Workers can decode
    disjoint record ranges while the parent hashes the immutable source bytes.

    Hand-authored or legacy whitespace-rich JSON returns ``None`` and follows
    the compatible serial semantic parser. The fast path therefore changes no
    accepted schema or digest semantics.
    """

    array_prefix = f'"{array_name}":['.encode()
    with path.open("rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as document:
            prefix_offset = document.find(array_prefix)
            if prefix_offset < 0:
                return None
            array_begin = prefix_offset + len(array_prefix)
            array_end = document.rfind(b'],"')
            if array_end < array_begin:
                return None

            # Parse only the small scalar root. Copying the prefix and suffix
            # is bounded independently of a multi-gigabyte record inventory.
            scalar_document = (
                document[:array_begin] + b"]" + document[array_end + 1 :]
            )
            try:
                root = json.loads(scalar_document)
            except json.JSONDecodeError:
                return None
            if not isinstance(root, dict) or root.get(array_name) != []:
                return None
            declared_count_name = (
                "request_count" if array_name == "requests" else "evidence_count"
            )
            try:
                declared_count = int(root[declared_count_name])
                expected_digest = str(root["manifest_digest"])
            except (KeyError, TypeError, ValueError):
                return None

            if array_begin == array_end:
                records: tuple[Mapping[str, Any], ...] = ()
            else:
                first_record = document[array_begin : min(array_begin + 256, array_end)]
                if not first_record.startswith(b'{"'):
                    return None
                key_end = first_record.find(b'":')
                if key_end < 2:
                    return None
                record_prefix = b',{"' + first_record[2 : key_end + 2]
                if workers is None:
                    worker_count = _offline_worker_count(
                        declared_count,
                        environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
                    )
                else:
                    if workers < 1:
                        raise ValueError("profiler manifest parse workers must be positive")
                    worker_count = min(
                        workers,
                        _physical_core_count(),
                        max(1, declared_count),
                    )
                if declared_count < parallel_threshold:
                    worker_count = 1

                boundaries = [array_begin]
                for worker_index in range(1, worker_count):
                    target = array_begin + (
                        (array_end - array_begin) * worker_index // worker_count
                    )
                    separator = document.find(record_prefix, target, array_end)
                    if separator < 0:
                        return None
                    boundaries.append(separator + 1)
                boundaries.append(array_end)
                boundaries = sorted(set(boundaries))
                tasks = tuple(
                    (path, begin, end)
                    for begin, end in zip(boundaries, boundaries[1:])
                    if begin < end
                )

                if len(tasks) == 1:
                    partitions = (_parse_manifest_array_range(tasks[0]),)
                else:
                    with ProcessPoolExecutor(
                        max_workers=len(tasks),
                        mp_context=multiprocessing.get_context("fork"),
                    ) as executor:
                        partitions = tuple(executor.map(
                            _parse_manifest_array_range,
                            tasks,
                        ))
                records = tuple(
                    record for partition in partitions for record in partition
                )

            if len(records) != declared_count:
                raise ValueError(
                    f"profiler {array_name} count does not match inventory"
                )

            digest_member = (
                b'"manifest_digest":'
                + json.dumps(expected_digest, separators=(",", ":")).encode()
            )
            digest_offset = document.find(digest_member)
            if digest_offset < 0:
                return None
            remove_begin = digest_offset
            remove_end = digest_offset + len(digest_member)
            if remove_begin > 0 and document[remove_begin - 1] == ord(","):
                remove_begin -= 1
            elif remove_end < len(document) and document[remove_end] == ord(","):
                remove_end += 1
            else:
                return None
            document_end = len(document)
            while document_end and document[document_end - 1] in b"\r\n":
                document_end -= 1
            digest = hashlib.sha256()
            document_view = memoryview(document)
            try:
                digest.update(document_view[:remove_begin])
                digest.update(document_view[remove_end:document_end])
            finally:
                document_view.release()
            actual_digest = "sha256:" + digest.hexdigest()
            if actual_digest != expected_digest:
                manifest_label = (
                    "request" if array_name == "requests" else "evidence"
                )
                raise ValueError(
                    f"profiler {manifest_label} manifest digest does not match contents"
                )

    root[array_name] = list(records)
    return root


def _physical_core_count() -> int:
    """Return the affinity-visible physical-core count for offline hashing."""

    try:
        visible_cpus = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        visible_cpus = tuple(range(os.cpu_count() or 1))
    physical = set()
    for cpu in visible_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return max(1, len(visible_cpus))
        physical.add((package, core))
    return max(1, len(physical))


def _offline_worker_count(
    record_count: int,
    *,
    environment_name: str,
    records_per_worker: int = 4096,
) -> int:
    """Return a physical-core-capped worker count for offline record work."""

    if record_count < 0:
        raise ValueError("offline record count cannot be negative")
    if record_count == 0:
        return 0
    requested = int(os.environ.get(
        environment_name,
        str(_physical_core_count()),
    ))
    if requested < 1:
        raise ValueError(f"{environment_name} must be positive")
    useful_workers = max(1, record_count // records_per_worker)
    return min(requested, _physical_core_count(), useful_workers)


def _equal_ranges(record_count: int, worker_count: int) -> tuple[tuple[int, int], ...]:
    """Partition an ordered inventory into contiguous balanced ranges."""

    if record_count < 0 or worker_count < 1 or worker_count > record_count:
        raise ValueError("invalid ordered range partition")
    base, remainder = divmod(record_count, worker_count)
    ranges = []
    begin = 0
    for worker_index in range(worker_count):
        size = base + (1 if worker_index < remainder else 0)
        ranges.append((begin, begin + size))
        begin += size
    return tuple(ranges)


def _validate_profiler_request_range(bounds: tuple[int, int]) -> None:
    """Validate one inherited request range without serializing its records."""

    begin, end = bounds
    for index in range(begin, end):
        _PARALLEL_REQUEST_VALIDATION_RECORDS[index].validate()


def _build_profiler_request_range(
    bounds: tuple[int, int],
) -> tuple[ProfilerRequest, ...]:
    """Convert one inherited observation range without pickling its input."""

    begin, end = bounds
    return tuple(
        profiler_request_for_observation(
            _PARALLEL_REQUEST_BUILD_OBSERVATIONS[index]
        )
        for index in range(begin, end)
    )


def _validate_coverage_range(
    bounds: tuple[int, int],
) -> tuple[int, int, tuple[str, ...]]:
    """Validate one inherited request/evidence range and reduce status counts."""

    begin, end = bounds
    return _validate_coverage_pairs_inline(
        _PARALLEL_COVERAGE_PAIRS[index]
        for index in range(begin, end)
    )


def _digest_export_observation_range(
    bounds: tuple[int, int],
) -> tuple[str, ...]:
    """Hash one inherited observation range in canonical row order."""

    begin, end = bounds
    return tuple(
        _PARALLEL_EXPORT_OBSERVATION_ROWS[index].digest()
        for index in range(begin, end)
    )


def _parallel_export_observation_digests(
    observations: tuple[NativeVNNIObservation, ...],
    *,
    workers: int,
) -> tuple[str, ...]:
    """Compute canonical observation identities across physical cores.

    Exact-point feature export joins every request to its timing witness by the
    witness's canonical SHA-256 digest. Recomputing more than one hundred
    thousand JSON-backed row identities on the parent held the GIL for most of
    a fit-only replay. Fork workers inherit the validated immutable rows
    copy-on-write and return only compact ordered digest strings. Flattening the
    range results in submission order preserves the historical join and
    duplicate-detection semantics exactly.
    """

    global _PARALLEL_EXPORT_OBSERVATION_ROWS

    if not observations:
        return ()
    worker_count = min(workers, len(observations))
    if worker_count < 1:
        raise ValueError("profiler observation digest worker count must be positive")
    if worker_count == 1 or len(observations) < 4096:
        return tuple(observation.digest() for observation in observations)

    _PARALLEL_EXPORT_OBSERVATION_ROWS = observations
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            shards = tuple(executor.map(
                _digest_export_observation_range,
                _equal_ranges(len(observations), worker_count),
            ))
        return tuple(digest for shard in shards for digest in shard)
    finally:
        _PARALLEL_EXPORT_OBSERVATION_ROWS = ()


def _validate_coverage_pairs_inline(
    pairs: Iterable[tuple[Any, Any]],
) -> tuple[int, int, tuple[str, ...]]:
    """Validate ordered evidence pairs and return deterministic status totals."""

    complete_count = 0
    unsupported_count = 0
    failed = []
    for request, item in pairs:
        item.validate(request)
        expected_status = (
            ProfilerEvidenceStatus.COMPLETE
            if request.profile_required
            else ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED
        )
        if item.status != expected_status:
            failed.append(item.request_id)
        elif item.status == ProfilerEvidenceStatus.COMPLETE:
            complete_count += 1
        else:
            unsupported_count += 1
    return complete_count, unsupported_count, tuple(failed)


def _manifest_record_mapping(record: Any) -> Mapping[str, Any]:
    """Return one canonical object or already decoded immutable JSON record."""

    if isinstance(record, Mapping):
        return record
    return record.canonical_mapping()


def _serialize_manifest_record_range(
    task: tuple[int, int, Path],
) -> Path:
    """Serialize one ordered record range directly to a private shard."""

    begin, end, output = task
    with output.open("w", encoding="utf-8") as handle:
        for index in range(begin, end):
            if index != begin:
                handle.write(",")
            json.dump(
                _manifest_record_mapping(_PARALLEL_MANIFEST_RECORDS[index]),
                handle,
                sort_keys=True,
                separators=(",", ":"),
            )
    return output


def _sha256_manifest_records(
    scalar_fields: Mapping[str, Any],
    array_name: str,
    records: tuple[Any, ...],
) -> str:
    """Hash a large canonical manifest without materializing its giant dict.

    The legacy identity is compact ``json.dumps(..., sort_keys=True)`` over one
    root object. Root keys are emitted in that same lexical order while record
    shards are serialized independently and reduced in original array order.
    This preserves every historical digest byte, caps peak memory, and lets the
    two-socket host use physical cores without oversubscribing hyperthreads.
    """

    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_MANIFEST_DIGEST_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("manifest digest worker count must be positive")
    worker_count = min(requested_workers, max(1, len(records) // 4096))
    def reduce_digest(serialized_shards: Iterable[bytes | Path]) -> str:
        digest = hashlib.sha256()
        digest.update(b"{")
        root_names = sorted((*scalar_fields, array_name))
        for field_index, name in enumerate(root_names):
            if field_index != 0:
                digest.update(b",")
            digest.update(json.dumps(name, separators=(",", ":")).encode())
            digest.update(b":")
            if name == array_name:
                digest.update(b"[")
                for shard_index, shard in enumerate(serialized_shards):
                    if shard_index != 0:
                        digest.update(b",")
                    if isinstance(shard, bytes):
                        digest.update(shard)
                    else:
                        with shard.open("rb") as handle:
                            while block := handle.read(1024 * 1024):
                                digest.update(block)
                digest.update(b"]")
            else:
                digest.update(json.dumps(
                    scalar_fields[name], sort_keys=True, separators=(",", ":")
                ).encode())
        digest.update(b"}")
        return "sha256:" + digest.hexdigest()

    if worker_count == 1:
        return reduce_digest((_serialize_manifest_records_inline(records),))

    global _PARALLEL_MANIFEST_RECORDS
    _PARALLEL_MANIFEST_RECORDS = records
    try:
        with tempfile.TemporaryDirectory(
            prefix="native-vnni-manifest-digest-"
        ) as temporary_directory:
            root = Path(temporary_directory)
            tasks = tuple(
                (begin, end, root / f"part-{index:04d}.json")
                for index, (begin, end) in enumerate(
                    _equal_ranges(len(records), worker_count)
                )
            )
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                serialized_shards = tuple(executor.map(
                    _serialize_manifest_record_range,
                    tasks,
                ))
            return reduce_digest(serialized_shards)
    finally:
        _PARALLEL_MANIFEST_RECORDS = ()


def _serialize_manifest_records_inline(records: tuple[Any, ...]) -> bytes:
    """Serialize small inventories in process to avoid fork overhead."""

    return ",".join(
        json.dumps(
            _manifest_record_mapping(record),
            sort_keys=True,
            separators=(",", ":"),
        )
        for record in records
    ).encode()


def _write_manifest_records_json(
    path: Path,
    scalar_fields: Mapping[str, Any],
    array_name: str,
    records: tuple[Any, ...],
) -> None:
    """Atomically stream one large manifest without building its JSON tree.

    Request inventories can contain hundreds of thousands of records.  Calling
    ``json.dumps`` on a root mapping first duplicates every request as a Python
    dictionary and then allocates the complete encoded document as one string.
    Besides a multi-gigabyte peak, that final copy is constrained by the GIL.

    Large inventories use the same physical-core-bounded record sharding as
    manifest hashing.  Workers serialize disjoint ordered ranges, and the
    parent concatenates those private files into one compact JSON document.
    Small inventories stream one record at a time in process.  Neither path
    changes the manifest digest because the digest authenticates canonical
    field values rather than presentation whitespace.
    """

    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_MANIFEST_WRITE_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("manifest writer worker count must be positive")
    worker_count = min(requested_workers, max(1, len(records) // 4096))
    staged_path = path.with_name(path.name + ".inprogress")
    published = False

    def write_root(serialized_shards: Iterable[bytes | Path]) -> None:
        with staged_path.open("wb") as output:
            output.write(b"{")
            root_names = sorted((*scalar_fields, array_name))
            for field_index, name in enumerate(root_names):
                if field_index != 0:
                    output.write(b",")
                output.write(json.dumps(
                    name, separators=(",", ":")
                ).encode())
                output.write(b":")
                if name == array_name:
                    output.write(b"[")
                    for shard_index, shard in enumerate(serialized_shards):
                        if shard_index != 0:
                            output.write(b",")
                        if isinstance(shard, bytes):
                            output.write(shard)
                        else:
                            with shard.open("rb") as handle:
                                shutil.copyfileobj(
                                    handle, output, length=1024 * 1024
                                )
                    output.write(b"]")
                else:
                    output.write(json.dumps(
                        scalar_fields[name],
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode())
            output.write(b"}\n")

    try:
        if worker_count == 1:
            write_root(
                json.dumps(
                    _manifest_record_mapping(record),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                for record in records
            )
        else:
            global _PARALLEL_MANIFEST_RECORDS
            _PARALLEL_MANIFEST_RECORDS = records
            try:
                with tempfile.TemporaryDirectory(
                    prefix=f".{path.name}.parts-",
                    dir=path.parent,
                ) as temporary_directory:
                    root = Path(temporary_directory)
                    tasks = tuple(
                        (begin, end, root / f"part-{index:04d}.json")
                        for index, (begin, end) in enumerate(
                            _equal_ranges(len(records), worker_count)
                        )
                    )
                    with ProcessPoolExecutor(
                        max_workers=worker_count,
                        mp_context=multiprocessing.get_context("fork"),
                    ) as executor:
                        serialized_shards = tuple(executor.map(
                            _serialize_manifest_record_range,
                            tasks,
                        ))
                    write_root(serialized_shards)
            finally:
                _PARALLEL_MANIFEST_RECORDS = ()
        os.replace(staged_path, path)
        published = True
    finally:
        if not published:
            staged_path.unlink(missing_ok=True)


def _is_sha256(value: str) -> bool:
    """Return whether ``value`` is one complete lowercase SHA-256 identity."""

    if len(value) != 71 or not value.startswith("sha256:"):
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _required_text(name: str, value: Any) -> str:
    """Normalize one required string and reject empty provenance fields."""

    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _require_exact_keys(
    raw: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    """Reject omitted and forward-version fields instead of ignoring either."""

    actual = set(raw)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{label} fields do not match schema: "
            f"missing={sorted(required - actual)} "
            f"unexpected={sorted(actual - required)}"
        )


def _candidate_registries(backend: Backend) -> tuple[CandidateRegistry, ...]:
    """Return every reviewed registry that can own one backend observation."""

    if backend == Backend.CPU:
        return (
            cpu_native_vnni_decode_registry(),
            cpu_native_vnni_verifier_registry(),
            cpu_native_vnni_prefill_registry(),
        )
    if backend == Backend.CUDA:
        return (cuda_native_vnni_gemv_registry(),)
    if backend == Backend.ROCM:
        return (
            rocm_native_vnni_decode_registry(),
            rocm_native_vnni_decode_formula_registry(),
            rocm_moe_grouped_prefill_registry(),
        )
    raise ValueError(f"unsupported profiler backend {backend}")


def _registry_for_observation(
    observation: NativeVNNIObservation,
) -> tuple[CandidateRegistry, Any]:
    """Resolve one observation to exactly one reviewed forceable candidate.

    Shape-resolved CUDA and ROCm formula rows are synthetic learner evidence
    backed by a concrete exact-KB timing row. They are intentionally rejected
    here: the concrete source row is the physical launch that must be profiled,
    while the formula itself does not name another kernel variant.
    """

    matches = []
    for registry in _candidate_registries(observation.backend):
        try:
            matches.append((registry, registry.resolve(observation.candidate_id)))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError(
            f"{observation.candidate_id}: expected exactly one candidate registry, "
            f"found {len(matches)}"
        )
    registry, candidate = matches[0]
    if candidate.config_json.get("family") in {
        "kpar_formula",
        "fused_kpar_formula",
        "clamped_kb_formula",
    }:
        raise ValueError(
            f"{observation.candidate_id}: a shape-resolved policy formula is not "
            "a physical profiler launch; use the unprojected concrete observation"
        )
    if observation.candidate_family != candidate.candidate_family:
        raise ValueError("observation candidate_family disagrees with registry")
    if observation.config_json != candidate.config_json:
        raise ValueError("observation config_json disagrees with candidate registry")
    if observation.arithmetic_fingerprint != candidate.arithmetic_fingerprint:
        raise ValueError(
            "observation arithmetic_fingerprint disagrees with candidate registry"
        )
    if observation.candidate_policy_hash != candidate.candidate_policy_hash():
        raise ValueError("observation candidate_policy_hash is stale")
    return registry, candidate


@dataclass(frozen=True, order=True)
class ProfilerRequest:
    """One isolated physical-candidate profiling obligation.

    The request repeats the launch discriminators needed by a standalone
    backend trainer.  It also binds those discriminators to the complete common
    observation and raw timing sidecar digest, preventing a profiler result
    from being joined to a newer timing corpus merely because shape and
    candidate names happen to match.
    """

    request_id: str
    observation_digest: str
    run_id: str
    corpus_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    backend: Backend
    architecture_class: str
    device_name: str
    driver_runtime: str
    threading_or_stream_mode: str
    semantic_contract: SemanticContract
    operation_kind: str
    bundle_signature: str
    projection_n_vector: tuple[int, ...]
    source_format: str
    source_codebook_id: int
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    shape_group_id: str
    shape_name: str
    execution_mode: ExecutionMode
    m: int
    aggregate_n: int
    k: int
    candidate_registry_surface: str
    candidate_id: str
    effective_candidate_id: str
    candidate_family: str
    config_json: dict[str, Any]
    schedule_signature: str
    workspace_signature: str
    prepared_resources: tuple[str, ...]
    arithmetic_fingerprint: str
    candidate_policy_hash: str
    trial_set_hash: str
    timing_sample_hash: str
    supported: bool
    graph_capture_ok: bool
    forced_route_ok: bool
    observed_candidate_id: str
    profile_protocol: str
    target_launches_per_profiler_pass: int
    active_rows: int | None = None

    @property
    def profile_required(self) -> bool:
        """Return whether a successful isolated profile is mandatory."""

        return (
            self.supported
            and self.forced_route_ok
            and (
                self.execution_mode != ExecutionMode.GRAPH_CAPTURED
                or self.graph_capture_ok
            )
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the deterministic JSON form used by request manifests."""

        # A shallow field projection is sufficient because this is a frozen
        # value object and the structured members are immutable by contract.
        # ``asdict`` recursively deep-copies every nested value and made a
        # 119k-request authentication pass spend minutes copying dictionaries.
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["backend"] = self.backend.value
        result["semantic_contract"] = self.semantic_contract.value
        result["execution_mode"] = self.execution_mode.value
        result["projection_n_vector"] = list(self.projection_n_vector)
        result["prepared_resources"] = list(self.prepared_resources)
        if self.active_rows is None:
            result.pop("active_rows")
        return result

    def validate(self) -> None:
        """Reject requests that cannot identify one production launch."""

        if self.active_rows is not None and (
            type(self.active_rows) is not int or not 1 <= self.active_rows <= self.m
            or self.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ):
            raise ValueError("profiler active_rows must fit the grouped physical M")
        for name in (
            "request_id",
            "run_id",
            "corpus_id",
            "git_revision",
            "build_id",
            "compiler_id",
            "architecture_class",
            "device_name",
            "driver_runtime",
            "threading_or_stream_mode",
            "operation_kind",
            "bundle_signature",
            "source_format",
            "prepared_family_id",
            "packing_abi",
            "shape_group_id",
            "shape_name",
            "candidate_registry_surface",
            "candidate_id",
            "effective_candidate_id",
            "candidate_family",
            "schedule_signature",
            "workspace_signature",
            "arithmetic_fingerprint",
            "candidate_policy_hash",
            "trial_set_hash",
            "timing_sample_hash",
            "observed_candidate_id",
        ):
            _required_text(name, getattr(self, name))
        for name in (
            "observation_digest",
            "candidate_policy_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 identity")
        if self.profile_protocol != PROFILE_PROTOCOL:
            raise ValueError("unsupported profiler request protocol")
        if self.target_launches_per_profiler_pass != 1:
            raise ValueError("each profiler pass must contain exactly one target launch")
        if self.m <= 0 or self.aggregate_n <= 0 or self.k <= 0:
            raise ValueError("profile dimensions must be positive")
        if not self.projection_n_vector or any(
            dimension <= 0 for dimension in self.projection_n_vector
        ):
            raise ValueError("projection_n_vector must contain positive dimensions")
        if sum(self.projection_n_vector) != self.aggregate_n:
            raise ValueError("projection_n_vector does not sum to aggregate_n")
        if not self.prepared_resources or any(
            not resource.strip() for resource in self.prepared_resources
        ):
            raise ValueError("prepared_resources must identify registry resources")
        if self.candidate_id.strip().upper() == "AUTO":
            raise ValueError("AUTO is not a physical profiler candidate")
        if self.config_json.get("family") in {
            "kpar_formula", "fused_kpar_formula", "clamped_kb_formula",
        }:
            raise ValueError("shape-resolved formulas are not physical profiler requests")

    @classmethod
    def validate_mapping_fields(cls, raw: Mapping[str, Any]) -> None:
        """Share the record schema with the allocation-free resume reader.

        Historical full-row receipts omit occupancy, preserving their original
        digests. Both readers must accept that omission but reject unknown fields.
        """
        expected_fields = set(cls.__dataclass_fields__) - {"active_rows"}
        if "active_rows" in raw:
            expected_fields.add("active_rows")
        _require_exact_keys(raw, expected_fields, "profiler request")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerRequest":
        """Parse one exact request record from a JSON manifest."""

        cls.validate_mapping_fields(raw)
        result = cls(
            request_id=str(raw["request_id"]),
            observation_digest=str(raw["observation_digest"]),
            run_id=str(raw["run_id"]),
            corpus_id=str(raw["corpus_id"]),
            git_revision=str(raw["git_revision"]),
            build_id=str(raw["build_id"]),
            compiler_id=str(raw["compiler_id"]),
            backend=Backend(str(raw["backend"])),
            architecture_class=str(raw["architecture_class"]),
            device_name=str(raw["device_name"]),
            driver_runtime=str(raw["driver_runtime"]),
            threading_or_stream_mode=str(raw["threading_or_stream_mode"]),
            semantic_contract=SemanticContract(str(raw["semantic_contract"])),
            operation_kind=str(raw["operation_kind"]),
            bundle_signature=str(raw["bundle_signature"]),
            projection_n_vector=tuple(
                int(value) for value in raw["projection_n_vector"]
            ),
            source_format=str(raw["source_format"]),
            source_codebook_id=int(raw["source_codebook_id"]),
            prepared_family_id=str(raw["prepared_family_id"]),
            packing_abi=str(raw["packing_abi"]),
            runtime_codebook_id=int(raw["runtime_codebook_id"]),
            shape_group_id=str(raw["shape_group_id"]),
            shape_name=str(raw["shape_name"]),
            execution_mode=ExecutionMode(str(raw["execution_mode"])),
            m=int(raw["m"]),
            aggregate_n=int(raw["aggregate_n"]),
            k=int(raw["k"]),
            candidate_registry_surface=str(raw["candidate_registry_surface"]),
            candidate_id=str(raw["candidate_id"]),
            effective_candidate_id=str(raw["effective_candidate_id"]),
            candidate_family=str(raw["candidate_family"]),
            config_json=dict(raw["config_json"]),
            schedule_signature=str(raw["schedule_signature"]),
            workspace_signature=str(raw["workspace_signature"]),
            prepared_resources=tuple(str(value) for value in raw["prepared_resources"]),
            arithmetic_fingerprint=str(raw["arithmetic_fingerprint"]),
            candidate_policy_hash=str(raw["candidate_policy_hash"]),
            trial_set_hash=str(raw["trial_set_hash"]),
            timing_sample_hash=str(raw["timing_sample_hash"]),
            supported=bool(raw["supported"]),
            graph_capture_ok=bool(raw["graph_capture_ok"]),
            forced_route_ok=bool(raw["forced_route_ok"]),
            observed_candidate_id=str(raw["observed_candidate_id"]),
            profile_protocol=str(raw["profile_protocol"]),
            target_launches_per_profiler_pass=int(
                raw["target_launches_per_profiler_pass"]
            ),
            active_rows=raw.get("active_rows"),
        )
        result.validate()
        return result


def _request_identity_fields(
    observation: NativeVNNIObservation,
    registry_surface: str,
) -> dict[str, Any]:
    """Return launch identity fields used to derive a stable request ID."""

    return {
        "observation_digest": observation.digest(),
        "backend": observation.backend.value,
        "architecture_class": observation.architecture_class,
        "registry_surface": registry_surface,
        "source_format": observation.source_format,
        "execution_mode": observation.execution_mode.value,
        "m": observation.m,
        "n": observation.aggregate_n,
        "k": observation.k,
        "candidate_id": observation.candidate_id,
        "effective_candidate_id": observation.effective_candidate_id,
        "profile_protocol": PROFILE_PROTOCOL,
    }


def profiler_request_for_observation(
    observation: NativeVNNIObservation,
) -> ProfilerRequest:
    """Derive one immutable isolated-profile request from a timing row."""

    observation.validate()
    registry, candidate = _registry_for_observation(observation)
    identity = _request_identity_fields(observation, registry.surface)
    request_hash = _sha256_json(identity)[7:]
    result = ProfilerRequest(
        request_id=f"nvprof-{observation.backend.value}-{request_hash[:24]}",
        observation_digest=observation.digest(),
        run_id=observation.run_id,
        corpus_id=observation.corpus_id,
        git_revision=observation.git_revision,
        build_id=observation.build_id,
        compiler_id=observation.compiler_id,
        backend=observation.backend,
        architecture_class=observation.architecture_class,
        device_name=observation.device_name,
        driver_runtime=observation.driver_runtime,
        threading_or_stream_mode=observation.threading_or_stream_mode,
        semantic_contract=observation.semantic_contract,
        operation_kind=observation.operation_kind,
        bundle_signature=observation.bundle_signature,
        projection_n_vector=observation.projection_n_vector,
        source_format=observation.source_format,
        source_codebook_id=observation.source_codebook_id,
        prepared_family_id=observation.prepared_family_id,
        packing_abi=observation.packing_abi,
        runtime_codebook_id=observation.runtime_codebook_id,
        shape_group_id=observation.shape_group_id,
        shape_name=observation.shape_name,
        execution_mode=observation.execution_mode,
        m=observation.m,
        aggregate_n=observation.aggregate_n,
        k=observation.k,
        candidate_registry_surface=registry.surface,
        candidate_id=observation.candidate_id,
        effective_candidate_id=observation.effective_candidate_id,
        candidate_family=observation.candidate_family,
        config_json=observation.config_json,
        schedule_signature=candidate.schedule_signature,
        workspace_signature=candidate.workspace_signature,
        prepared_resources=candidate.prepared_resources,
        arithmetic_fingerprint=observation.arithmetic_fingerprint,
        candidate_policy_hash=observation.candidate_policy_hash,
        trial_set_hash=observation.trial_set_hash,
        timing_sample_hash=observation.timing_sample_hash,
        supported=observation.supported,
        graph_capture_ok=observation.graph_capture_ok,
        forced_route_ok=observation.forced_route_ok,
        observed_candidate_id=observation.observed_candidate_id,
        profile_protocol=PROFILE_PROTOCOL,
        target_launches_per_profiler_pass=1,
        active_rows=observation.active_rows,
    )
    result.validate()
    return result


def _profiler_requests_for_observations(
    observations: Sequence[NativeVNNIObservation],
    *,
    workers: int | None = None,
) -> tuple[ProfilerRequest, ...]:
    """Convert observations in parallel while preserving their exact order.

    Request construction validates each timing row, resolves registry metadata,
    and hashes several provenance identities. Those operations are independent
    and Python-GIL-bound. Fork workers inherit the immutable observation tuple,
    so the parent sends only balanced index ranges and receives one ordered
    request partition per physical core.
    """

    records = tuple(observations)
    if not records:
        return ()
    if workers is None:
        worker_count = _offline_worker_count(
            len(records),
            environment_name="LLAMINAR_NATIVE_VNNI_REQUEST_BUILD_WORKERS",
            records_per_worker=2048,
        )
    else:
        if workers < 1:
            raise ValueError("profiler request build workers must be positive")
        worker_count = min(workers, _physical_core_count(), len(records))
    if worker_count <= 1:
        return tuple(profiler_request_for_observation(row) for row in records)

    global _PARALLEL_REQUEST_BUILD_OBSERVATIONS
    _PARALLEL_REQUEST_BUILD_OBSERVATIONS = records
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            partitions = tuple(executor.map(
                _build_profiler_request_range,
                _equal_ranges(len(records), worker_count),
            ))
    finally:
        _PARALLEL_REQUEST_BUILD_OBSERVATIONS = ()
    return tuple(request for partition in partitions for request in partition)


@dataclass(frozen=True)
class ProfilerRequestManifest:
    """Complete profiler obligation inventory for one timing corpus."""

    corpus_digest: str
    candidate_registry_digest: str
    requests: tuple[ProfilerRequest, ...]
    learner_version: str = LEARNER_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    schema_version: str = PROFILER_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS:
            raise ValueError(
                "unsupported profiler request schema_version="
                f"{self.schema_version!r}; accepted versions are "
                f"{sorted(SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS)!r}"
            )
        if not self.requests:
            raise ValueError("profiler request manifest must not be empty")
        if not _is_sha256(self.corpus_digest):
            raise ValueError("profiler request corpus_digest is invalid")
        if not _is_sha256(self.candidate_registry_digest):
            raise ValueError("profiler request registry digest is invalid")
        if self.learner_version not in COMPATIBLE_OBSERVATION_LEARNER_VERSIONS:
            raise ValueError(
                f"unsupported profiler request learner_version={self.learner_version!r}; "
                "accepted immutable evidence versions are "
                f"{sorted(COMPATIBLE_OBSERVATION_LEARNER_VERSIONS)!r}"
            )
        if (
            self.feature_schema_version
            not in COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "unsupported profiler request feature_schema_version="
                f"{self.feature_schema_version!r}; accepted immutable evidence "
                "versions are "
                f"{sorted(COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS)!r}"
            )
        request_ids = [request.request_id for request in self.requests]
        observation_ids = [request.observation_digest for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("profiler request IDs must be unique")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("one timing observation cannot create two profiler requests")
        worker_count = _offline_worker_count(
            len(self.requests),
            environment_name="LLAMINAR_NATIVE_VNNI_VALIDATION_WORKERS",
        )
        if worker_count <= 1:
            for request in self.requests:
                request.validate()
        else:
            global _PARALLEL_REQUEST_VALIDATION_RECORDS
            _PARALLEL_REQUEST_VALIDATION_RECORDS = self.requests
            try:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    tuple(executor.map(
                        _validate_profiler_request_range,
                        _equal_ranges(len(self.requests), worker_count),
                    ))
            finally:
                _PARALLEL_REQUEST_VALIDATION_RECORDS = ()
        if self.schema_version == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION:
            _validate_matched_profiler_anchors(self.requests)
        else:
            _validate_exact_profiler_launches(self.requests)

    def payload_mapping(self) -> dict[str, Any]:
        """Return the digest-covered root without its self hash."""

        return {
            "schema_version": self.schema_version,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "observation_schema_version": SCHEMA_VERSION,
            "policy_abi": POLICY_ABI,
            "learner_version": self.learner_version,
            "feature_schema_version": self.feature_schema_version,
            "corpus_digest": self.corpus_digest,
            "candidate_registry_digest": self.candidate_registry_digest,
            "request_count": len(self.requests),
            "requests": [request.canonical_mapping() for request in self.requests],
        }

    def digest(self) -> str:
        """Hash every request and all schema/provenance versions."""

        return self._cached_digest

    @cached_property
    def _cached_digest(self) -> str:
        """Serialize this immutable request inventory once per process."""

        return _sha256_manifest_records(
            {
                "schema_version": self.schema_version,
                "metric_set_version": PROFILER_METRIC_SET_VERSION,
                "observation_schema_version": SCHEMA_VERSION,
                "policy_abi": POLICY_ABI,
                "learner_version": self.learner_version,
                "feature_schema_version": self.feature_schema_version,
                "corpus_digest": self.corpus_digest,
                "candidate_registry_digest": self.candidate_registry_digest,
                "request_count": len(self.requests),
            },
            "requests",
            self.requests,
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the complete JSON document including its self hash."""

        return {**self.payload_mapping(), "manifest_digest": self.digest()}


def build_profiler_request_manifest(
    corpus: ObservationCorpus,
) -> ProfilerRequestManifest:
    """Create exact requests for three timing strata in every work contest.

    Dynamic counters are point evidence. IPC, cache behavior, occupancy,
    throughput, and duration observed at one ``(M,N,K)`` launch must never be
    attached to another work size. The current manifest profiles the fastest
    5%, centered median 5%, and slowest 5% of physical candidates in every
    backend/ISA/format/shape/mode/M contest. Canonical timing remains exhaustive
    and is still the only latency label used for dispatch regret.

    Timing rows may still contain genuine aliases: two source formats can
    prepare the identical execution codebook, and two shape names can describe
    the same physical geometry. Those rows share one profiler request only
    when every launch-changing field is identical. The deterministic
    representative remains a cryptographic witness for the shared invocation;
    no unsupported or non-forceable row can stand in for executable evidence.
    """

    selected = {
        row.digest(): row
        for row in _stratified_profiler_observations(corpus)
    }

    requests = tuple(sorted(
        _profiler_requests_for_observations(tuple(selected.values())),
        key=lambda request: (
            request.backend.value,
            request.architecture_class,
            request.semantic_contract.value,
            request.operation_kind,
            request.source_format,
            request.execution_mode.value,
            request.m,
            request.aggregate_n,
            request.k,
            request.effective_candidate_id,
            request.observation_digest,
        ),
    ))
    return ProfilerRequestManifest(
        corpus_digest=corpus.digest(),
        candidate_registry_digest=candidate_registry_digest(),
        requests=requests,
        schema_version=PROFILER_REQUEST_SCHEMA_VERSION,
    )


def _profiler_contest_key(row: NativeVNNIObservation) -> tuple[object, ...]:
    """Identify one candidate timing contest that requires three strata.

    Source format and shape aliases stay explicit here even when they happen to
    prepare the same physical kernel. The sampling guarantee is user-facing:
    every measured format, shape, and M must contribute its own fastest,
    centered, and slowest evidence. Physical-launch deduplication happens only
    after those per-contest obligations have been selected.
    """

    return (
        row.backend,
        row.architecture_class,
        row.build_id,
        row.compiler_id,
        row.device_name,
        row.driver_runtime,
        row.threading_or_stream_mode,
        row.semantic_contract,
        row.operation_kind,
        row.bundle_signature,
        row.source_format,
        row.source_codebook_id,
        row.prepared_family_id,
        row.packing_abi,
        row.runtime_codebook_id,
        row.shape_group_id,
        row.shape_name,
        row.execution_mode,
        row.m,
        row.projection_n_vector,
        row.aggregate_n,
        row.k,
        *((row.active_rows,) if row.active_rows is not None else ()),
    )


def _profiler_representative_key(
    row: NativeVNNIObservation,
) -> tuple[object, ...]:
    """Choose one deterministic timing witness for a physical launch alias."""

    return (
        row.source_format,
        row.source_codebook_id,
        row.shape_name,
        row.shape_group_id,
        row.candidate_id,
        row.digest(),
    )


def _profiler_timing_rank_key(
    item: tuple[tuple[object, ...], NativeVNNIObservation],
) -> tuple[object, ...]:
    """Rank physical candidates from fastest to slowest deterministically."""

    launch_key, row = item
    return (
        row.median_us,
        row.p95_us,
        row.min_us,
        row.mad_us,
        row.cv,
        row.effective_candidate_id,
        repr(launch_key),
        row.digest(),
    )


def _timing_stratum_indices(candidate_count: int) -> tuple[int, ...]:
    """Return the union of fastest, centered, and slowest timing bands.

    ``ceil`` and a minimum of one keep small candidate contests informative.
    Overlapping strata are deduplicated, so a contest with one or two physical
    candidates profiles each candidate exactly once.
    """

    if candidate_count <= 0:
        raise ValueError("profiler timing strata require at least one candidate")
    stratum_size = max(
        1,
        math.ceil(candidate_count * PROFILER_TIMING_STRATUM_FRACTION),
    )
    median_begin = (candidate_count - stratum_size) // 2
    return tuple(sorted({
        *range(stratum_size),
        *range(median_begin, median_begin + stratum_size),
        *range(candidate_count - stratum_size, candidate_count),
    }))


def _stratified_profiler_observations(
    corpus: ObservationCorpus,
) -> tuple[NativeVNNIObservation, ...]:
    """Return only the timing witnesses from stratified launch entries."""

    return tuple(
        row for _, row in _stratified_profiler_observation_entries(corpus)
    )


def _stratified_profiler_observation_entries(
    corpus: ObservationCorpus,
    *,
    workers: int | None = None,
) -> tuple[tuple[tuple[object, ...], NativeVNNIObservation], ...]:
    """Select deterministic exact-point profiler witnesses for one corpus.

    Every launchable physical candidate first participates in each canonical
    format/shape/M contest represented by its timing aliases. The three timing
    strata are selected independently per contest, then their physical launch
    keys are unioned. True aliases consequently share one profiler process
    while no contest can disappear merely because another format prepared the
    same runtime codebook.
    """

    runtime_groups = _profiler_runtime_groups(corpus)
    if workers is None:
        worker_count = _offline_worker_count(
            len(corpus),
            environment_name="LLAMINAR_NATIVE_VNNI_PROFILER_GROUP_WORKERS",
            records_per_worker=32768,
        )
    else:
        if workers < 1:
            raise ValueError("profiler grouping workers must be positive")
        worker_count = min(workers, _physical_core_count(), len(runtime_groups))

    if worker_count <= 1:
        selected_entries, launchable_candidates, never_launchable = (
            _select_profiler_strata(corpus)
        )
        shards = ((selected_entries, launchable_candidates, never_launchable),)
    else:
        assignments = _balanced_profiler_runtime_group_assignments(
            corpus,
            runtime_groups,
            worker_count,
        )
        global _PARALLEL_PROFILER_GROUP_CORPUS
        global _PARALLEL_PROFILER_RUNTIME_GROUPS
        _PARALLEL_PROFILER_GROUP_CORPUS = corpus
        _PARALLEL_PROFILER_RUNTIME_GROUPS = runtime_groups
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                shards = tuple(executor.map(
                    _select_profiler_strata_for_runtime_groups,
                    assignments,
                ))
        finally:
            _PARALLEL_PROFILER_GROUP_CORPUS = None
            _PARALLEL_PROFILER_RUNTIME_GROUPS = ()

    selected_by_launch: dict[
        tuple[object, ...], NativeVNNIObservation
    ] = {}
    all_launchable_candidates: set[tuple[object, ...]] = set()
    unsupported_by_candidate: dict[
        tuple[object, ...], NativeVNNIObservation
    ] = {}
    for selected_entries, launchable_candidates, never_launchable in shards:
        all_launchable_candidates.update(launchable_candidates)
        for launch_key, row in selected_entries:
            previous = selected_by_launch.get(launch_key)
            if (
                previous is None
                or _profiler_representative_key(row)
                < _profiler_representative_key(previous)
            ):
                selected_by_launch[launch_key] = row
        for candidate_key, row in never_launchable:
            previous = unsupported_by_candidate.get(candidate_key)
            if (
                previous is None
                or _unsupported_profiler_representative_key(row)
                < _unsupported_profiler_representative_key(previous)
            ):
                unsupported_by_candidate[candidate_key] = row

    selected: dict[
        str,
        tuple[tuple[object, ...], NativeVNNIObservation],
    ] = {
        row.digest(): (launch_key, row)
        for launch_key, row in selected_by_launch.items()
    }

    # Retain one explicit capability record only for candidates that are never
    # launchable anywhere in the corpus. It creates no profiler process because
    # ``profile_required`` is false, but preserves the reviewed reason that the
    # registry member has no executable point in this generation.
    for candidate_key, representative in unsupported_by_candidate.items():
        if candidate_key in all_launchable_candidates:
            continue
        unsupported_launch_key = (
            *candidate_key,
            representative.execution_mode,
            representative.m,
            representative.projection_n_vector,
            representative.aggregate_n,
            representative.k,
        )
        selected[representative.digest()] = (
            unsupported_launch_key,
            representative,
        )

    return tuple(selected[digest] for digest in sorted(selected))


def _profiler_workload_partition_key(runtime_key: Any) -> tuple[object, ...]:
    """Keep every alias and candidate for one physical workload together.

    ``launch_k_tiles`` is deliberately absent. Shape-resolved formula rows may
    carry a projected tile count, but they describe the same concrete physical
    launch and must remain in the same profiler contest as that launch.
    """

    return (
        runtime_key.backend,
        runtime_key.architecture_class,
        runtime_key.operation_kind,
        runtime_key.bundle_signature,
        runtime_key.projection_n_vector,
        runtime_key.prepared_family_id,
        runtime_key.packing_abi,
        runtime_key.runtime_codebook_id,
        runtime_key.execution_mode,
        runtime_key.m,
        runtime_key.aggregate_n,
        runtime_key.k,
    )


def _profiler_runtime_groups(
    corpus: ObservationCorpus,
) -> tuple[tuple[Any, ...], ...]:
    """Group indexed runtime keys without scanning the observation inventory."""

    grouped: dict[tuple[object, ...], list[Any]] = {}
    for runtime_key in corpus.runtime_keys():
        grouped.setdefault(
            _profiler_workload_partition_key(runtime_key), []
        ).append(runtime_key)
    return tuple(
        tuple(sorted(grouped[key]))
        for key in sorted(grouped)
    )


def _balanced_profiler_runtime_group_assignments(
    corpus: ObservationCorpus,
    runtime_groups: tuple[tuple[Any, ...], ...],
    worker_count: int,
) -> tuple[tuple[int, ...], ...]:
    """Balance whole workload groups by row count across fork workers."""

    sizes = tuple(
        sum(len(corpus.rows_for_runtime_key(key)) for key in group)
        for group in runtime_groups
    )
    assignments: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    for group_index in sorted(
        range(len(runtime_groups)),
        key=lambda index: (-sizes[index], index),
    ):
        worker_index = min(
            range(worker_count),
            key=lambda index: (loads[index], index),
        )
        assignments[worker_index].append(group_index)
        loads[worker_index] += sizes[group_index]
    return tuple(tuple(sorted(assignment)) for assignment in assignments)


def _select_profiler_strata_for_runtime_groups(
    group_indices: tuple[int, ...],
) -> tuple[
    tuple[tuple[tuple[object, ...], NativeVNNIObservation], ...],
    frozenset[tuple[object, ...]],
    tuple[tuple[tuple[object, ...], NativeVNNIObservation], ...],
]:
    """Select strata for inherited whole-workload groups in one worker."""

    if _PARALLEL_PROFILER_GROUP_CORPUS is None:
        raise RuntimeError("parallel profiler grouping corpus is not installed")
    rows = (
        row
        for group_index in group_indices
        for runtime_key in _PARALLEL_PROFILER_RUNTIME_GROUPS[group_index]
        for row in _PARALLEL_PROFILER_GROUP_CORPUS.rows_for_runtime_key(runtime_key)
    )
    return _select_profiler_strata(rows)


def _unsupported_profiler_representative_key(
    row: NativeVNNIObservation,
) -> tuple[object, ...]:
    """Choose one deterministic capability witness across work points."""

    return (
        row.m,
        row.aggregate_n,
        row.k,
        *_profiler_representative_key(row),
    )


def _select_profiler_strata(
    rows: Iterable[NativeVNNIObservation],
) -> tuple[
    tuple[tuple[tuple[object, ...], NativeVNNIObservation], ...],
    frozenset[tuple[object, ...]],
    tuple[tuple[tuple[object, ...], NativeVNNIObservation], ...],
]:
    """Reduce complete workload groups to selected and capability witnesses."""

    launchable, never_launchable = _group_profiler_observation_rows(rows)
    launchable_candidates = frozenset(
        key[:-len(_profiled_workload_key(rows[0]))]
        for key, rows in launchable.items()
    )
    contests: dict[
        tuple[object, ...],
        dict[tuple[object, ...], NativeVNNIObservation],
    ] = {}
    for launch_key, rows in launchable.items():
        for row in rows:
            members = contests.setdefault(_profiler_contest_key(row), {})
            previous = members.get(launch_key)
            if (
                previous is None
                or _profiler_representative_key(row)
                < _profiler_representative_key(previous)
            ):
                members[launch_key] = row

    selected_launch_keys: set[tuple[object, ...]] = set()
    for members in contests.values():
        ranked = sorted(members.items(), key=_profiler_timing_rank_key)
        selected_launch_keys.update(
            ranked[index][0]
            for index in _timing_stratum_indices(len(ranked))
        )

    selected: dict[tuple[object, ...], NativeVNNIObservation] = {}
    for launch_key in selected_launch_keys:
        representative = min(
            launchable[launch_key],
            key=_profiler_representative_key,
        )
        selected[launch_key] = representative

    unsupported = {}
    for candidate_key, rows in never_launchable.items():
        representative = min(
            rows,
            key=_unsupported_profiler_representative_key,
        )
        unsupported[candidate_key] = representative

    return (
        tuple(selected.items()),
        launchable_candidates,
        tuple(unsupported.items()),
    )


def _group_profiler_observations(
    corpus: ObservationCorpus,
) -> tuple[
    dict[tuple[object, ...], list[NativeVNNIObservation]],
    dict[tuple[object, ...], list[NativeVNNIObservation]],
]:
    """Group timing rows by the exact profiler identity they represent.

    The grouping key is intentionally the same field sequence returned by
    :func:`_profiled_exact_launch_key`.  Building it directly from a timing row
    and its reviewed candidate registry entry lets resume checks compare the
    current corpus to authenticated coverage before constructing expensive
    request value objects.  Registry metadata is cached per physical candidate
    surface because schedule, workspace, and prepared resources do not change
    with ``M``, ``N``, or ``K``.
    """

    return _group_profiler_observation_rows(corpus)


def _group_profiler_observation_rows(
    rows: Iterable[NativeVNNIObservation],
) -> tuple[
    dict[tuple[object, ...], list[NativeVNNIObservation]],
    dict[tuple[object, ...], list[NativeVNNIObservation]],
]:
    """Group an arbitrary complete-workload row iterable by launch identity."""

    launchable: dict[
        tuple[object, ...], list[NativeVNNIObservation]
    ] = {}
    never_launchable: dict[
        tuple[object, ...], list[NativeVNNIObservation]
    ] = {}
    physical_keys: dict[tuple[object, ...], tuple[object, ...]] = {}
    for row in rows:
        candidate_identity = (
            row.backend,
            row.architecture_class,
            row.operation_kind,
            row.bundle_signature,
            row.prepared_family_id,
            row.packing_abi,
            row.runtime_codebook_id,
            row.effective_candidate_id,
            row.arithmetic_fingerprint,
            row.candidate_policy_hash,
            row.threading_or_stream_mode,
        )
        physical_key = physical_keys.get(candidate_identity)
        if physical_key is None:
            registry, candidate = _registry_for_observation(row)
            physical_key = (
                row.backend,
                row.architecture_class,
                row.operation_kind,
                row.bundle_signature,
                row.prepared_family_id,
                row.packing_abi,
                row.runtime_codebook_id,
                registry.surface,
                row.effective_candidate_id,
                row.arithmetic_fingerprint,
                row.candidate_policy_hash,
                candidate.schedule_signature,
                candidate.workspace_signature,
                candidate.prepared_resources,
                row.threading_or_stream_mode,
            )
            physical_keys[candidate_identity] = physical_key
        if row.supported and row.forced_route_ok and (
            row.generic_eligible or row.active_rows is not None
        ):
            launch_key = (
                *physical_key,
                *_profiled_workload_key(row),
            )
            launchable.setdefault(launch_key, []).append(row)
        else:
            never_launchable.setdefault(physical_key, []).append(row)
    return launchable, never_launchable


def build_missing_profiler_request_transaction(
    corpus: ObservationCorpus,
    covered_manifests: Iterable[ProfilerRequestManifest],
    *,
    covered_launch_keys: set[tuple[object, ...]] | None = None,
) -> tuple[ObservationCorpus | None, ProfilerRequestManifest | None]:
    """Derive only exact physical launches absent from prior transactions.

    A timing corpus may gain authenticated candidate-expansion or refinement
    rows after an earlier profiler transaction was collected. Reprofiling the
    complete corpus would waste hours and create duplicate authority for every
    unchanged launch. This function builds the current complete obligation
    inventory, compares launch-changing identity rather than representative
    source aliases, and returns a self-contained delta transaction.

    The returned requests retain the exact observation digests chosen from the
    enlarged corpus. Their compact observation corpus therefore provides the
    timing witnesses needed by normal evidence validation and later additive
    composition. An already complete corpus returns ``(None, None)`` so a
    turnkey replay can stop successfully without creating empty manifests.
    """

    covered_manifests = tuple(covered_manifests)
    if covered_launch_keys is not None and covered_manifests:
        raise ValueError(
            "provide covered manifests or preauthenticated launch keys, not both"
        )
    covered_keys = (
        covered_launch_keys
        if covered_launch_keys is not None
        else _covered_profiler_launch_keys(covered_manifests)
    )
    missing_rows = tuple(
        row
        for launch_key, row in _stratified_profiler_observation_entries(corpus)
        if launch_key not in covered_keys
    )

    if not missing_rows:
        return None, None

    missing_requests = tuple(sorted(
        _profiler_requests_for_observations(missing_rows),
        key=lambda request: (
            request.backend.value,
            request.architecture_class,
            request.semantic_contract.value,
            request.operation_kind,
            request.source_format,
            request.execution_mode.value,
            request.m,
            request.aggregate_n,
            request.k,
            request.effective_candidate_id,
            request.observation_digest,
        ),
    ))

    missing_by_digest = {row.digest(): row for row in missing_rows}
    ordered_missing_rows = []
    for request in missing_requests:
        try:
            ordered_missing_rows.append(
                missing_by_digest[request.observation_digest]
            )
        except KeyError as error:
            raise ValueError(
                f"{request.request_id}: missing request lost its timing witness"
            ) from error
    observations = ObservationCorpus._from_validated(ordered_missing_rows)
    requests = ProfilerRequestManifest(
        corpus_digest=observations.digest(),
        candidate_registry_digest=candidate_registry_digest(),
        requests=missing_requests,
        learner_version=LEARNER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        schema_version=PROFILER_REQUEST_SCHEMA_VERSION,
    )
    return observations, requests


def count_uncovered_profiler_requests(
    requests: ProfilerRequestManifest,
    covered_manifests: Iterable[ProfilerRequestManifest],
) -> int:
    """Count exact launches in ``requests`` absent from prior transactions.

    Request IDs deliberately bind a profile record to its complete timing
    witness, including run provenance. Resume decisions instead concern the
    physical launch that already paid for hardware counters. Keeping this
    comparison explicit lets a regenerated timing corpus reuse an authenticated
    profile while still invalidating changed arithmetic, schedules, workspace,
    execution modes, or geometry.
    """

    covered_keys = _covered_profiler_launch_keys(covered_manifests)
    return sum(
        1
        for request in requests.requests
        if _profiled_exact_launch_key(request) not in covered_keys
    )


def _covered_profiler_launch_keys(
    manifests: Iterable[ProfilerRequestManifest],
) -> set[tuple[object, ...]]:
    """Return the authenticated physical coverage represented by manifests."""

    covered_keys: set[tuple[object, ...]] = set()
    for manifest in manifests:
        for request in manifest.requests:
            key = (
                _profiled_physical_candidate_key(request)
                if manifest.schema_version == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION
                else _profiled_exact_launch_key(request)
            )
            covered_keys.add(key)
    return covered_keys


def write_profiler_request_manifest(
    path: Path, manifest: ProfilerRequestManifest
) -> None:
    """Write a deterministic request manifest for backend collectors."""

    _write_manifest_records_json(
        path,
        {
            "schema_version": manifest.schema_version,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "observation_schema_version": SCHEMA_VERSION,
            "policy_abi": POLICY_ABI,
            "learner_version": manifest.learner_version,
            "feature_schema_version": manifest.feature_schema_version,
            "corpus_digest": manifest.corpus_digest,
            "candidate_registry_digest": manifest.candidate_registry_digest,
            "request_count": len(manifest.requests),
            "manifest_digest": manifest.digest(),
        },
        "requests",
        manifest.requests,
    )


def _read_profiler_request_document(
    path: Path,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> dict[str, Any]:
    """Read and authenticate raw request JSON without constructing records."""

    canonical_raw = _read_canonical_manifest_document(
        path,
        "requests",
        workers=workers,
        parallel_threshold=parallel_threshold,
    )
    pretty_raw = (
        None
        if canonical_raw is not None
        else _read_pretty_manifest_document(
            path,
            "requests",
            workers=workers,
            parallel_threshold=parallel_threshold,
        )
    )
    raw = canonical_raw if canonical_raw is not None else pretty_raw
    if raw is None:
        raw = json.loads(path.read_text(encoding="utf-8"))
    root_fields = {
        "schema_version",
        "metric_set_version",
        "observation_schema_version",
        "policy_abi",
        "learner_version",
        "feature_schema_version",
        "corpus_digest",
        "candidate_registry_digest",
        "request_count",
        "requests",
        "manifest_digest",
    }
    _require_exact_keys(raw, root_fields, "profiler request manifest")
    exact_versions = {
        "metric_set_version": PROFILER_METRIC_SET_VERSION,
        "observation_schema_version": SCHEMA_VERSION,
        "policy_abi": POLICY_ABI,
    }
    for name, expected in exact_versions.items():
        if raw[name] != expected:
            raise ValueError(
                f"profiler request {name}={raw[name]!r}; expected {expected!r}"
            )
    if raw["schema_version"] not in SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported profiler request schema_version="
            f"{raw['schema_version']!r}; accepted versions are "
            f"{sorted(SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS)!r}"
        )
    if raw["learner_version"] not in COMPATIBLE_OBSERVATION_LEARNER_VERSIONS:
        raise ValueError(
            "unsupported profiler request learner_version="
            f"{raw['learner_version']!r}"
        )
    if (
        raw["feature_schema_version"]
        not in COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS
    ):
        raise ValueError(
            "unsupported profiler request feature_schema_version="
            f"{raw['feature_schema_version']!r}"
        )
    for name in ("corpus_digest", "candidate_registry_digest"):
        if not _is_sha256(str(raw[name])):
            raise ValueError(f"profiler request {name} is invalid")
    requests_raw = raw["requests"]
    if not isinstance(requests_raw, list):
        raise ValueError("profiler requests must be a JSON array")
    if not requests_raw:
        raise ValueError("profiler request manifest must not be empty")
    if int(raw["request_count"]) != len(requests_raw):
        raise ValueError("profiler request_count does not match request inventory")
    # Canonical files were authenticated directly from their immutable bytes.
    # The compatibility parser must recreate the semantic canonical payload.
    if canonical_raw is None and pretty_raw is None:
        digest = _sha256_manifest_records(
            {
                "schema_version": raw["schema_version"],
                "metric_set_version": raw["metric_set_version"],
                "observation_schema_version": raw["observation_schema_version"],
                "policy_abi": raw["policy_abi"],
                "learner_version": raw["learner_version"],
                "feature_schema_version": raw["feature_schema_version"],
                "corpus_digest": raw["corpus_digest"],
                "candidate_registry_digest": raw["candidate_registry_digest"],
                "request_count": len(requests_raw),
            },
            "requests",
            tuple(requests_raw),
        )
        if str(raw["manifest_digest"]) != digest:
            raise ValueError("profiler request manifest digest does not match contents")
    return raw


def _decode_profiler_request_range(
    bounds: tuple[int, int],
) -> tuple[ProfilerRequest, ...]:
    """Construct one inherited range of authenticated request records."""

    begin, end = bounds
    return tuple(
        ProfilerRequest.from_mapping(_PARALLEL_REQUEST_DECODE_RECORDS[index])
        for index in range(begin, end)
    )


def read_profiler_request_manifest(
    path: Path,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> ProfilerRequestManifest:
    """Read and authenticate one exact profiler request document.

    Large exact-point inventories contain six figures of deeply validated
    records.  Their root digest is authenticated first; fork workers then
    construct disjoint ranges from the inherited immutable JSON array.  Ordered
    range assembly keeps the result byte-for-byte equivalent to serial decode
    while avoiding one GIL-bound CPU core during every cache miss.
    """

    raw = _read_profiler_request_document(
        path,
        workers=workers,
        parallel_threshold=parallel_threshold,
    )
    records = tuple(raw["requests"])
    if workers is None:
        worker_count = _offline_worker_count(
            len(records),
            environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
        )
    else:
        if workers < 1:
            raise ValueError("profiler request decode worker count must be positive")
        worker_count = min(workers, _physical_core_count(), len(records))
    if worker_count <= 1 or len(records) < parallel_threshold:
        requests = tuple(ProfilerRequest.from_mapping(record) for record in records)
    else:
        global _PARALLEL_REQUEST_DECODE_RECORDS
        _PARALLEL_REQUEST_DECODE_RECORDS = records
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                decoded = tuple(executor.map(
                    _decode_profiler_request_range,
                    _equal_ranges(len(records), worker_count),
                ))
        finally:
            _PARALLEL_REQUEST_DECODE_RECORDS = ()
        requests = tuple(item for partition in decoded for item in partition)
    manifest = ProfilerRequestManifest(
        corpus_digest=str(raw["corpus_digest"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        requests=requests,
        learner_version=str(raw["learner_version"]),
        feature_schema_version=str(raw["feature_schema_version"]),
        schema_version=str(raw["schema_version"]),
    )
    return manifest


def read_profiler_request_coverage_keys(
    path: Path,
) -> set[tuple[object, ...]]:
    """Authenticate one request manifest and project only coverage identity.

    Resume planning does not consume provenance text, timing hashes, or the
    collector-facing request value objects. Decoding a six-figure manifest into
    full dataclasses solely to form a set of physical launch keys held one CPU
    core for tens of seconds on every fit-only replay. This reader preserves
    root digest authentication and strict record structure while projecting the
    exact typed key directly from raw JSON. Legacy anchor manifests retain the
    ordinary typed reader because their shape-independent coverage semantics are
    intentionally different and no longer occur in current corpora.
    """

    raw = _read_profiler_request_document(path)
    if raw["schema_version"] == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION:
        return _covered_profiler_launch_keys((
            read_profiler_request_manifest(path),
        ))

    request_ids = set()
    observation_digests = set()
    required_launch_owners: dict[tuple[object, ...], str] = {}
    covered_keys: set[tuple[object, ...]] = set()
    for record in raw["requests"]:
        ProfilerRequest.validate_mapping_fields(record)
        request_id = str(record["request_id"])
        observation_digest = str(record["observation_digest"])
        _required_text("request_id", request_id)
        if not _is_sha256(observation_digest):
            raise ValueError("observation_digest must be a SHA-256 identity")
        if request_id in request_ids:
            raise ValueError("profiler request IDs must be unique")
        if observation_digest in observation_digests:
            raise ValueError(
                "one timing observation cannot create two profiler requests"
            )
        request_ids.add(request_id)
        observation_digests.add(observation_digest)

        key = _typed_raw_exact_profiler_launch_key(record)
        if _raw_request_profile_required(record):
            owner = required_launch_owners.setdefault(key, request_id)
            if owner != request_id:
                raise ValueError(
                    "profiler manifest contains duplicate evidence obligations "
                    f"for exact physical launch {key}: {owner} and {request_id}"
                )
        covered_keys.add(key)
    return covered_keys


class MetricCategory(str, Enum):
    """Whether one metric is static launch state or a dynamic measurement."""

    STATIC_RESOURCE = "static_resource"
    DYNAMIC_COUNTER = "dynamic_counter"


class MetricAvailability(str, Enum):
    """Explicit disposition of one canonical profiler metric."""

    MEASURED = "measured"
    UNSUPPORTED_BY_TOOL = "unsupported_by_tool"
    COLLECTION_FAILED = "collection_failed"


@dataclass(frozen=True, order=True)
class MetricDefinition:
    """Canonical metric expected from one backend collector."""

    metric_id: str
    category: MetricCategory
    unit: str
    required: bool


CPU_METRIC_DEFINITIONS = (
    MetricDefinition("cpu.cycles", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.ref_cycles", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.instructions", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.task_clock_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("cpu.wall_clock_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("cpu.branches", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.branch_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.cache_references", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.cache_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.l1d_loads", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.l1d_load_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.llc_loads", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.llc_load_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
)

CUDA_METRIC_DEFINITIONS = (
    MetricDefinition("gpu.duration_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("gpu.registers_per_thread", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.static_shared_memory_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.dynamic_shared_memory_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.local_memory_bytes_per_thread", MetricCategory.STATIC_RESOURCE, "bytes", False),
    MetricDefinition("gpu.theoretical_occupancy_pct", MetricCategory.STATIC_RESOURCE, "percent", True),
    MetricDefinition("gpu.achieved_occupancy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.compute_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.alu_pipe_utilization_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.fma_pipe_utilization_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.tensor_pipe_utilization_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.dram_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.l1_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.l2_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.executed_ipc_active", MetricCategory.DYNAMIC_COUNTER, "instructions_per_cycle", False),
    MetricDefinition("gpu.warp_cycles_per_issued_instruction", MetricCategory.DYNAMIC_COUNTER, "cycles", False),
    MetricDefinition("gpu.local_memory_spill_requests", MetricCategory.DYNAMIC_COUNTER, "count", False),
)

ROCM_METRIC_DEFINITIONS = (
    MetricDefinition("gpu.duration_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("gpu.vgpr_count", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.sgpr_count", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.lds_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.scratch_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.achieved_occupancy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.gpu_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.valu_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.valu_utilization_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.memory_unit_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.memory_unit_stalled_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.l2_cache_hit_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.fetch_kib", MetricCategory.DYNAMIC_COUNTER, "KiB", False),
    MetricDefinition("gpu.write_kib", MetricCategory.DYNAMIC_COUNTER, "KiB", False),
    MetricDefinition("gpu.wavefront_count", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("gpu.lds_bank_conflict_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition(
        "gpu.valu_instructions_per_workitem",
        MetricCategory.DYNAMIC_COUNTER,
        "instructions_per_workitem",
        False,
    ),
    MetricDefinition(
        "gpu.flat_vmem_instructions_per_workitem",
        MetricCategory.DYNAMIC_COUNTER,
        "instructions_per_workitem",
        False,
    ),
)


def metric_definitions(backend: Backend) -> tuple[MetricDefinition, ...]:
    """Return the exact v1 metric inventory for one backend."""

    if backend == Backend.CPU:
        return CPU_METRIC_DEFINITIONS
    if backend == Backend.CUDA:
        return CUDA_METRIC_DEFINITIONS
    if backend == Backend.ROCM:
        return ROCM_METRIC_DEFINITIONS
    raise ValueError(f"unsupported profiler backend {backend}")


@dataclass(frozen=True, order=True)
class ProfilerMetric:
    """One measured or explicitly unavailable canonical profiler feature."""

    metric_id: str
    category: MetricCategory
    unit: str
    availability: MetricAvailability
    value: float | None
    source_name: str
    reason: str | None

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible metric record."""

        result = asdict(self)
        result["category"] = self.category.value
        result["availability"] = self.availability.value
        return result

    def validate(self, definition: MetricDefinition) -> None:
        """Validate identity, units, finiteness, and explicit unavailability."""

        if self.metric_id != definition.metric_id:
            raise ValueError("profiler metric ID disagrees with metric definition")
        if self.category != definition.category or self.unit != definition.unit:
            raise ValueError(f"{self.metric_id}: metric category/unit mismatch")
        _required_text("source_name", self.source_name)
        if self.availability == MetricAvailability.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(f"{self.metric_id}: measured value must be finite")
            if self.reason not in (None, ""):
                raise ValueError(f"{self.metric_id}: measured value cannot have a failure reason")
        else:
            if self.value is not None:
                raise ValueError(f"{self.metric_id}: unavailable metric cannot carry a value")
            _required_text("reason", self.reason)
            if definition.required:
                raise ValueError(
                    f"{self.metric_id}: required profiler metric was not measured"
                )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerMetric":
        """Parse one exact metric record."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiler metric")
        value = raw["value"]
        return cls(
            metric_id=str(raw["metric_id"]),
            category=MetricCategory(str(raw["category"])),
            unit=str(raw["unit"]),
            availability=MetricAvailability(str(raw["availability"])),
            value=None if value is None else float(value),
            source_name=str(raw["source_name"]),
            reason=None if raw["reason"] is None else str(raw["reason"]),
        )


class ProfiledDispatchKind(str, Enum):
    """Physical work represented by one profiler dispatch record."""

    GPU_KERNEL = "gpu_kernel"
    CPU_PARALLEL_REGION = "cpu_parallel_region"


@dataclass(frozen=True)
class ProfiledDispatch:
    """One ordered kernel in a candidate pipeline or one CPU kernel region."""

    dispatch_index: int
    dispatch_kind: ProfiledDispatchKind
    kernel_name: str
    kernel_fingerprint: str
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    metrics: tuple[ProfilerMetric, ...]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the exact nested JSON representation."""

        return {
            "dispatch_index": self.dispatch_index,
            "dispatch_kind": self.dispatch_kind.value,
            "kernel_name": self.kernel_name,
            "kernel_fingerprint": self.kernel_fingerprint,
            "grid": None if self.grid is None else list(self.grid),
            "block": None if self.block is None else list(self.block),
            "metrics": [metric.canonical_mapping() for metric in self.metrics],
        }

    def validate(self, backend: Backend) -> None:
        """Require complete metric coverage and valid physical geometry."""

        if self.dispatch_index < 0:
            raise ValueError("dispatch_index must be non-negative")
        _required_text("kernel_name", self.kernel_name)
        if not _is_sha256(self.kernel_fingerprint):
            raise ValueError("kernel_fingerprint must be a SHA-256 identity")
        expected_kind = (
            ProfiledDispatchKind.CPU_PARALLEL_REGION
            if backend == Backend.CPU
            else ProfiledDispatchKind.GPU_KERNEL
        )
        if self.dispatch_kind != expected_kind:
            raise ValueError("profiled dispatch kind disagrees with backend")
        if backend == Backend.CPU:
            if self.grid is not None or self.block is not None:
                raise ValueError("CPU profiler regions do not have GPU launch geometry")
        else:
            for name, geometry in (("grid", self.grid), ("block", self.block)):
                if geometry is None or len(geometry) != 3 or any(value <= 0 for value in geometry):
                    raise ValueError(f"GPU {name} must contain three positive dimensions")

        definitions = {item.metric_id: item for item in metric_definitions(backend)}
        observed = {metric.metric_id: metric for metric in self.metrics}
        if len(observed) != len(self.metrics):
            raise ValueError("profiled dispatch contains duplicate metric IDs")
        if set(observed) != set(definitions):
            raise ValueError(
                "profiled dispatch metric inventory mismatch: "
                f"missing={sorted(set(definitions) - set(observed))} "
                f"unexpected={sorted(set(observed) - set(definitions))}"
            )
        for metric_id, definition in definitions.items():
            observed[metric_id].validate(definition)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfiledDispatch":
        """Parse one exact ordered dispatch record."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiled dispatch")

        def geometry(name: str) -> tuple[int, int, int] | None:
            value = raw[name]
            if value is None:
                return None
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{name} must be null or a three-element array")
            return tuple(int(item) for item in value)  # type: ignore[return-value]

        return cls(
            dispatch_index=int(raw["dispatch_index"]),
            dispatch_kind=ProfiledDispatchKind(str(raw["dispatch_kind"])),
            kernel_name=str(raw["kernel_name"]),
            kernel_fingerprint=str(raw["kernel_fingerprint"]),
            grid=geometry("grid"),
            block=geometry("block"),
            metrics=tuple(
                ProfilerMetric.from_mapping(item) for item in raw["metrics"]
            ),
        )


class ProfilerEvidenceStatus(str, Enum):
    """Disposition of one isolated candidate profile transaction."""

    COMPLETE = "complete"
    CANDIDATE_UNSUPPORTED = "candidate_unsupported"
    TOOL_UNAVAILABLE = "tool_unavailable"
    LAUNCH_FAILED = "launch_failed"
    PARSE_FAILED = "parse_failed"


EXPECTED_PROFILER_TOOL = {
    Backend.CPU: "linux-perf",
    Backend.CUDA: "nsight-compute",
    Backend.ROCM: "rocprofiler-sdk",
}


@dataclass(frozen=True)
class ProfilerEvidence:
    """Complete outcome of one request's separate profiler invocation."""

    request_id: str
    observation_digest: str
    backend: Backend
    status: ProfilerEvidenceStatus
    status_reason: str | None
    profiler_tool: str
    profiler_tool_version: str
    metric_set_version: str
    collector_version: str
    command_digest: str
    raw_artifact_digest: str
    profiler_pass_count: int
    target_launches_per_profiler_pass: int
    dispatches: tuple[ProfiledDispatch, ...]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return deterministic JSON for the evidence sidecar."""

        return {
            "request_id": self.request_id,
            "observation_digest": self.observation_digest,
            "backend": self.backend.value,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "profiler_tool": self.profiler_tool,
            "profiler_tool_version": self.profiler_tool_version,
            "metric_set_version": self.metric_set_version,
            "collector_version": self.collector_version,
            "command_digest": self.command_digest,
            "raw_artifact_digest": self.raw_artifact_digest,
            "profiler_pass_count": self.profiler_pass_count,
            "target_launches_per_profiler_pass": self.target_launches_per_profiler_pass,
            "dispatches": [dispatch.canonical_mapping() for dispatch in self.dispatches],
        }

    def validate(self, request: ProfilerRequest) -> None:
        """Validate request binding and successful/failed-state invariants."""

        if self.request_id != request.request_id:
            raise ValueError("profiler evidence request_id mismatch")
        if self.observation_digest != request.observation_digest:
            raise ValueError("profiler evidence observation_digest mismatch")
        if self.backend != request.backend:
            raise ValueError("profiler evidence backend mismatch")
        if self.metric_set_version != PROFILER_METRIC_SET_VERSION:
            raise ValueError("unsupported profiler metric set")
        if self.collector_version not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
            raise ValueError("unsupported profiler collector version")
        if self.target_launches_per_profiler_pass != 1:
            raise ValueError("profiler evidence must isolate one target launch per pass")

        if self.status == ProfilerEvidenceStatus.COMPLETE:
            if not request.profile_required:
                raise ValueError("unreachable candidate cannot claim complete profile evidence")
            if self.status_reason not in (None, ""):
                raise ValueError("complete profiler evidence cannot have a failure reason")
            if self.profiler_tool != EXPECTED_PROFILER_TOOL[request.backend]:
                raise ValueError("profiler evidence used the wrong backend tool")
            _required_text("profiler_tool_version", self.profiler_tool_version)
            if not _is_sha256(self.command_digest):
                raise ValueError("profiler command digest is invalid")
            if not _is_sha256(self.raw_artifact_digest):
                raise ValueError("profiler raw artifact digest is invalid")
            if self.profiler_pass_count <= 0:
                raise ValueError("complete profiler evidence requires at least one pass")
            if not self.dispatches:
                raise ValueError("complete profiler evidence has no physical dispatches")
            if tuple(dispatch.dispatch_index for dispatch in self.dispatches) != tuple(
                range(len(self.dispatches))
            ):
                raise ValueError("profiled dispatches must have contiguous launch order")
            for dispatch in self.dispatches:
                dispatch.validate(request.backend)
            return

        _required_text("status_reason", self.status_reason)
        if self.dispatches:
            raise ValueError("failed profiler evidence cannot publish partial dispatch metrics")
        if self.profiler_pass_count != 0:
            raise ValueError("failed profiler evidence must report zero completed passes")
        if self.status == ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED:
            if request.profile_required:
                raise ValueError("required candidate cannot be marked unsupported")
        elif not request.profile_required:
            raise ValueError(
                "an unreachable candidate must use candidate_unsupported, not a tool failure"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerEvidence":
        """Parse one exact evidence record; request validation happens later."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiler evidence")
        return cls(
            request_id=str(raw["request_id"]),
            observation_digest=str(raw["observation_digest"]),
            backend=Backend(str(raw["backend"])),
            status=ProfilerEvidenceStatus(str(raw["status"])),
            status_reason=(
                None if raw["status_reason"] is None else str(raw["status_reason"])
            ),
            profiler_tool=str(raw["profiler_tool"]),
            profiler_tool_version=str(raw["profiler_tool_version"]),
            metric_set_version=str(raw["metric_set_version"]),
            collector_version=str(raw["collector_version"]),
            command_digest=str(raw["command_digest"]),
            raw_artifact_digest=str(raw["raw_artifact_digest"]),
            profiler_pass_count=int(raw["profiler_pass_count"]),
            target_launches_per_profiler_pass=int(
                raw["target_launches_per_profiler_pass"]
            ),
            dispatches=tuple(
                ProfiledDispatch.from_mapping(item) for item in raw["dispatches"]
            ),
        )


@dataclass(frozen=True)
class ProfilerEvidenceManifest:
    """Incremental or complete evidence bound to one request manifest."""

    request_manifest_digest: str
    corpus_digest: str
    candidate_registry_digest: str
    evidence: tuple[ProfilerEvidence, ...]
    collector_version: str = PROFILER_COLLECTOR_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_manifest_digest",
            "corpus_digest",
            "candidate_registry_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"profiler evidence {name} is invalid")
        request_ids = [item.request_id for item in self.evidence]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("profiler evidence contains duplicate request IDs")
        if self.collector_version not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
            raise ValueError("unsupported profiler evidence collector")
        if any(
            item.collector_version != self.collector_version
            for item in self.evidence
        ):
            raise ValueError("profiler evidence manifest mixes collector generations")

    def payload_mapping(self) -> dict[str, Any]:
        """Return digest-covered evidence root without its self hash."""

        return {
            "schema_version": PROFILER_EVIDENCE_SCHEMA_VERSION,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "collector_version": self.collector_version,
            "request_manifest_digest": self.request_manifest_digest,
            "corpus_digest": self.corpus_digest,
            "candidate_registry_digest": self.candidate_registry_digest,
            "evidence_count": len(self.evidence),
            "evidence": [item.canonical_mapping() for item in self.evidence],
        }

    def digest(self) -> str:
        """Hash all profiler records independently of filesystem location."""

        return self._cached_digest

    @cached_property
    def _cached_digest(self) -> str:
        """Serialize this immutable evidence inventory once per process."""

        return _sha256_manifest_records(
            {
                "schema_version": PROFILER_EVIDENCE_SCHEMA_VERSION,
                "metric_set_version": PROFILER_METRIC_SET_VERSION,
                "collector_version": self.collector_version,
                "request_manifest_digest": self.request_manifest_digest,
                "corpus_digest": self.corpus_digest,
                "candidate_registry_digest": self.candidate_registry_digest,
                "evidence_count": len(self.evidence),
            },
            "evidence",
            self.evidence,
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return complete evidence JSON including its self hash."""

        return {**self.payload_mapping(), "manifest_digest": self.digest()}


def write_profiler_evidence_manifest(
    path: Path, manifest: ProfilerEvidenceManifest
) -> None:
    """Write a deterministic profiler evidence sidecar."""

    _write_manifest_records_json(
        path,
        {
            "schema_version": PROFILER_EVIDENCE_SCHEMA_VERSION,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "collector_version": manifest.collector_version,
            "request_manifest_digest": manifest.request_manifest_digest,
            "corpus_digest": manifest.corpus_digest,
            "candidate_registry_digest": manifest.candidate_registry_digest,
            "evidence_count": len(manifest.evidence),
            "manifest_digest": manifest.digest(),
        },
        "evidence",
        manifest.evidence,
    )


def _read_profiler_evidence_document(
    path: Path,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> dict[str, Any]:
    """Read and authenticate raw evidence JSON without constructing records."""

    canonical_raw = _read_canonical_manifest_document(
        path,
        "evidence",
        workers=workers,
        parallel_threshold=parallel_threshold,
    )
    pretty_raw = (
        None
        if canonical_raw is not None
        else _read_pretty_manifest_document(
            path,
            "evidence",
            workers=workers,
            parallel_threshold=parallel_threshold,
        )
    )
    raw = canonical_raw if canonical_raw is not None else pretty_raw
    if raw is None:
        raw = json.loads(path.read_text(encoding="utf-8"))
    root_fields = {
        "schema_version",
        "metric_set_version",
        "collector_version",
        "request_manifest_digest",
        "corpus_digest",
        "candidate_registry_digest",
        "evidence_count",
        "evidence",
        "manifest_digest",
    }
    _require_exact_keys(raw, root_fields, "profiler evidence manifest")
    if raw["schema_version"] != PROFILER_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported profiler evidence schema")
    if raw["metric_set_version"] != PROFILER_METRIC_SET_VERSION:
        raise ValueError("unsupported profiler evidence metric set")
    if raw["collector_version"] not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
        raise ValueError("unsupported profiler evidence collector")
    for name in (
        "request_manifest_digest",
        "corpus_digest",
        "candidate_registry_digest",
    ):
        if not _is_sha256(str(raw[name])):
            raise ValueError(f"profiler evidence {name} is invalid")
    evidence_raw = raw["evidence"]
    if not isinstance(evidence_raw, list):
        raise ValueError("profiler evidence inventory must be a JSON array")
    if int(raw["evidence_count"]) != len(evidence_raw):
        raise ValueError("profiler evidence_count does not match inventory")
    if canonical_raw is None and pretty_raw is None:
        digest = _sha256_manifest_records(
            {
                "schema_version": raw["schema_version"],
                "metric_set_version": raw["metric_set_version"],
                "collector_version": raw["collector_version"],
                "request_manifest_digest": raw["request_manifest_digest"],
                "corpus_digest": raw["corpus_digest"],
                "candidate_registry_digest": raw["candidate_registry_digest"],
                "evidence_count": len(evidence_raw),
            },
            "evidence",
            tuple(evidence_raw),
        )
        if str(raw["manifest_digest"]) != digest:
            raise ValueError("profiler evidence manifest digest does not match contents")
    return raw


def _decode_profiler_evidence_range(
    bounds: tuple[int, int],
) -> tuple[ProfilerEvidence, ...]:
    """Construct one inherited range of authenticated evidence records."""

    begin, end = bounds
    return tuple(
        ProfilerEvidence.from_mapping(_PARALLEL_EVIDENCE_DECODE_RECORDS[index])
        for index in range(begin, end)
    )


def read_profiler_evidence_manifest(
    path: Path,
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> ProfilerEvidenceManifest:
    """Read and authenticate an incremental or complete evidence sidecar.

    Counter records are considerably deeper than request records because every
    physical dispatch owns a complete metric inventory.  Decode their already
    authenticated JSON ranges on physical cores in parallel, preserving exact
    manifest order for deterministic joins and digests.
    """

    raw = _read_profiler_evidence_document(
        path,
        workers=workers,
        parallel_threshold=parallel_threshold,
    )
    records = tuple(raw["evidence"])
    if workers is None:
        worker_count = _offline_worker_count(
            len(records),
            environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
        )
    else:
        if workers < 1:
            raise ValueError("profiler evidence decode worker count must be positive")
        worker_count = min(workers, _physical_core_count(), len(records))
    if worker_count <= 1 or len(records) < parallel_threshold:
        evidence = tuple(ProfilerEvidence.from_mapping(record) for record in records)
    else:
        global _PARALLEL_EVIDENCE_DECODE_RECORDS
        _PARALLEL_EVIDENCE_DECODE_RECORDS = records
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                decoded = tuple(executor.map(
                    _decode_profiler_evidence_range,
                    _equal_ranges(len(records), worker_count),
                ))
        finally:
            _PARALLEL_EVIDENCE_DECODE_RECORDS = ()
        evidence = tuple(item for partition in decoded for item in partition)
    manifest = ProfilerEvidenceManifest(
        request_manifest_digest=str(raw["request_manifest_digest"]),
        corpus_digest=str(raw["corpus_digest"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        evidence=evidence,
        collector_version=str(raw["collector_version"]),
    )
    return manifest


@dataclass(frozen=True)
class ProfilerCoverageReport:
    """Human-readable disposition of one request/evidence join."""

    request_count: int
    required_count: int
    unsupported_count: int
    complete_count: int
    failure_count: int
    missing_request_ids: tuple[str, ...]
    failed_request_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether every request has its only valid terminal state."""

        return not self.missing_request_ids and not self.failed_request_ids


def validate_profiler_evidence_coverage(
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> ProfilerCoverageReport:
    """Join sidecars exactly and enforce complete supported-candidate coverage.

    ``require_complete=False`` is useful while collectors checkpoint a long
    all-format sweep.  It never relaxes record validation: any evidence record
    that is present must still be internally complete and bound to its request.
    """

    if evidence.request_manifest_digest != requests.digest():
        raise ValueError("profiler evidence belongs to another request manifest")
    if evidence.corpus_digest != requests.corpus_digest:
        raise ValueError("profiler evidence belongs to another timing corpus")
    if evidence.candidate_registry_digest != requests.candidate_registry_digest:
        raise ValueError("profiler evidence uses another candidate registry")

    request_by_id = {request.request_id: request for request in requests.requests}
    evidence_by_id = {item.request_id: item for item in evidence.evidence}
    extras = sorted(set(evidence_by_id) - set(request_by_id))
    if extras:
        raise ValueError(f"profiler evidence contains unknown requests: {extras}")

    missing = tuple(sorted(set(request_by_id) - set(evidence_by_id)))
    failed = []
    complete_count = 0
    unsupported_count = 0
    required_count = sum(request.profile_required for request in requests.requests)
    ordered_pairs = tuple(
        (request_by_id[request_id], item)
        for request_id, item in sorted(evidence_by_id.items())
    )
    worker_count = _offline_worker_count(
        len(ordered_pairs),
        environment_name="LLAMINAR_NATIVE_VNNI_VALIDATION_WORKERS",
    )
    if worker_count <= 1:
        reductions = (_validate_coverage_pairs_inline(ordered_pairs),)
    else:
        global _PARALLEL_COVERAGE_PAIRS
        _PARALLEL_COVERAGE_PAIRS = ordered_pairs
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                reductions = tuple(executor.map(
                    _validate_coverage_range,
                    _equal_ranges(len(ordered_pairs), worker_count),
                ))
        finally:
            _PARALLEL_COVERAGE_PAIRS = ()
    for completed, unsupported, worker_failures in reductions:
        complete_count += completed
        unsupported_count += unsupported
        failed.extend(worker_failures)

    report = ProfilerCoverageReport(
        request_count=len(request_by_id),
        required_count=required_count,
        unsupported_count=unsupported_count,
        complete_count=complete_count,
        failure_count=len(failed),
        missing_request_ids=missing,
        failed_request_ids=tuple(failed),
    )
    if require_complete and not report.complete:
        raise ValueError(
            "profiler coverage is incomplete: "
            f"missing={list(report.missing_request_ids)} "
            f"failed={list(report.failed_request_ids)}"
        )
    return report


@dataclass(frozen=True)
class ComposedProfilerEvidence:
    """One authenticated union of independently collected profiler corpora.

    Candidate-family expansion deliberately profiles only newly introduced
    physical kernels.  A later policy fit therefore owns several immutable
    request/evidence transactions rather than one monolithic profiler replay.
    This value packages their exact timing witnesses and records into the
    canonical three files expected by corpus publication without pretending
    that counters measured at one anchor geometry came from another geometry.
    """

    observations: ObservationCorpus
    requests: ProfilerRequestManifest
    evidence: ProfilerEvidenceManifest


def _profiled_physical_candidate_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return the shape-independent kernel identity used by profiler fitting.

    Geometry is intentionally absent.  One physical kernel variant may have
    been profiled at only one representative anchor, and two additive sources
    must never contribute competing anchors for that same fitted descriptor.
    """

    return (
        request.backend,
        request.architecture_class,
        request.operation_kind,
        request.bundle_signature,
        request.prepared_family_id,
        request.packing_abi,
        request.runtime_codebook_id,
        request.candidate_registry_surface,
        request.effective_candidate_id,
        request.arithmetic_fingerprint,
        request.candidate_policy_hash,
        request.schedule_signature,
        request.workspace_signature,
        request.prepared_resources,
        request.threading_or_stream_mode,
    )


def _profiled_candidate_surface_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return the candidate-independent surface for comparable counters.

    Candidate identity is deliberately absent. Every launchable candidate in
    this surface competes for the same runtime work and therefore must be
    profiled at the same anchor set. Comparing IPC, cache, or throughput from
    candidate families measured at unrelated geometries confounds the family
    label with work size and can silently poison the learned dispatch fit.
    """

    return (
        request.backend,
        request.architecture_class,
        request.operation_kind,
        request.bundle_signature,
        request.prepared_family_id,
        request.packing_abi,
        request.runtime_codebook_id,
        request.candidate_registry_surface,
    )


def _profiled_anchor(request: ProfilerRequest) -> tuple[object, ...]:
    """Return every geometry field that changes one profiler launch."""

    return (
        request.m,
        request.projection_n_vector,
        request.aggregate_n,
        request.k,
        *((request.active_rows,) if request.active_rows is not None else ()),
    )


def _profiled_workload_key(
    value: NativeVNNIObservation | ProfilerRequest,
) -> tuple[object, ...]:
    """Bind physical geometry and device-counted occupancy as one workload.

    Omit the optional suffix for historical full-row evidence so existing
    immutable profiler obligations retain their exact identity. Counted rows
    must never deduplicate against another occupancy or pointer-free launch.
    """

    return (
        value.execution_mode, value.m, value.projection_n_vector,
        value.aggregate_n, value.k,
        *((value.active_rows,) if value.active_rows is not None else ()),
    )


def _profiled_exact_launch_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return every discriminator that identifies one physical invocation.

    Dynamic counters are meaningful only for the exact work submitted to the
    kernel.  In particular, changing execution mode or any of ``M``, ``N``,
    or ``K`` creates another profiling obligation even when the selected
    template instantiation is unchanged.  Source-format and shape-name aliases
    are intentionally absent because prepared-family/codebook and geometry
    already identify the physical work they share.
    """

    return (
        *_profiled_physical_candidate_key(request),
        *_profiled_workload_key(request),
    )


def _validate_exact_profiler_launches(
    requests: Iterable[ProfilerRequest],
) -> None:
    """Require exactly one request for every executable physical invocation.

    The request builder may collapse true observation aliases before this
    validation runs.  Any remaining duplicate means two evidence records could
    claim ownership of the same kernel/ISA/codebook/candidate/M/N/K launch,
    which would make the attached counters ambiguous.
    """

    owner_by_launch: dict[tuple[object, ...], str] = {}
    for request in requests:
        if not request.profile_required:
            continue
        launch_key = _profiled_exact_launch_key(request)
        owner = owner_by_launch.setdefault(launch_key, request.request_id)
        if owner != request.request_id:
            raise ValueError(
                "profiler manifest contains duplicate evidence obligations for "
                f"exact physical launch {launch_key}: {owner} and "
                f"{request.request_id}"
            )


def _validate_matched_profiler_anchors(
    requests: Iterable[ProfilerRequest],
) -> None:
    """Require identical profiler anchor sets across competing candidates.

    A base transaction and a later candidate-family expansion are each valid
    in isolation when their own candidates share an anchor. Their union is not
    valid if the two transactions chose different geometries. Validating the
    complete request inventory catches that additive-evidence failure before
    any normalized counter is admitted to a learner.

    The current feature catalog owns one shape-independent descriptor per
    physical candidate, so each candidate must have exactly one anchor. A
    future multi-anchor model must first make anchor identity an explicit
    descriptor dimension rather than silently collapsing several launches.
    """

    anchors_by_surface: dict[
        tuple[object, ...],
        dict[str, set[tuple[object, ...]]],
    ] = {}
    for request in requests:
        if not request.profile_required:
            continue
        candidate_anchors = anchors_by_surface.setdefault(
            _profiled_candidate_surface_key(request),
            {},
        )
        candidate_anchors.setdefault(
            request.effective_candidate_id,
            set(),
        ).add(_profiled_anchor(request))

    for surface, anchors_by_candidate in anchors_by_surface.items():
        multiple_anchors = {
            candidate: tuple(sorted(anchors))
            for candidate, anchors in anchors_by_candidate.items()
            if len(anchors) != 1
        }
        if multiple_anchors:
            raise ValueError(
                "profiler physical candidates must use exactly one anchor for "
                f"comparable surface {surface}: {multiple_anchors}"
            )
        if len(anchors_by_candidate) < 2:
            continue
        anchor_sets = {
            tuple(sorted(anchors))
            for anchors in anchors_by_candidate.values()
        }
        if len(anchor_sets) != 1:
            detail = {
                candidate: tuple(sorted(anchors))
                for candidate, anchors in sorted(anchors_by_candidate.items())
            }
            raise ValueError(
                "profiler candidates use unmatched anchor sets for comparable "
                f"surface {surface}: {detail}"
            )


def compose_profiler_evidence(
    sources: Iterable[
        tuple[
            ObservationCorpus,
            ProfilerRequestManifest,
            ProfilerEvidenceManifest,
        ]
    ],
) -> ComposedProfilerEvidence:
    """Compose complete additive profiler transactions without new launches.

    Every source is authenticated independently before any record is admitted.
    Exact duplicate sources are harmless and deduplicated, while a repeated
    request with different contents, conflicting evidence, a second anchor for
    one physical candidate, mixed registry generations, or incomplete coverage
    is a hard failure.  The resulting top-level manifests are rebound only to
    the union transaction; request IDs, observation digests, launch geometry,
    raw-artifact identities, and all counters remain byte-for-byte unchanged.

    The composed ``corpus_digest`` identifies the complete ordered set of
    authenticated source request/evidence/witness digests.  Feature export uses
    the returned exact compact witnesses, so it still re-derives every retained
    request from the timing observation that actually produced the launch.
    """

    source_set = tuple(sources)
    if not source_set:
        raise ValueError("profiler evidence composition requires at least one source")

    request_by_id: dict[str, ProfilerRequest] = {}
    evidence_by_id: dict[str, ProfilerEvidence] = {}
    observation_by_digest: dict[str, NativeVNNIObservation] = {}
    physical_candidate_owner: dict[tuple[object, ...], str] = {}
    source_provenance: set[tuple[str, str, str]] = set()
    registry_digests: set[str] = set()
    source_feature_schema_versions: set[str] = set()
    collector_version: str | None = None
    request_schema_version: str | None = None

    for source_index, (observations, requests, evidence) in enumerate(source_set):
        validate_profiler_evidence_coverage(requests, evidence, require_complete=True)
        compact = compact_profiler_observation_witnesses(
            observations,
            requests,
            evidence,
            require_complete=True,
        )
        source_provenance.add((requests.digest(), evidence.digest(), compact.digest()))

        # Feature engineering is intentionally downstream of immutable raw
        # counter collection. A newer compatible exporter may add normalized
        # learner features without changing any profiled physical launch or
        # metric. Retain those older transactions and normalize the composed
        # manifest to FEATURE_SCHEMA_VERSION below. Request/collector schemas
        # still describe collection semantics and therefore remain exact.
        source_feature_schema_versions.add(requests.feature_schema_version)
        expected_metadata = (
            ("collector", collector_version, evidence.collector_version),
            ("request schema", request_schema_version, requests.schema_version),
        )
        for label, previous, current in expected_metadata:
            if previous is not None and previous != current:
                raise ValueError(
                    f"profiler source {source_index} uses another {label} generation"
                )
        registry_digests.add(requests.candidate_registry_digest)
        collector_version = evidence.collector_version
        request_schema_version = requests.schema_version

        source_evidence_by_id = {item.request_id: item for item in evidence.evidence}
        for request in requests.requests:
            previous_request = request_by_id.setdefault(request.request_id, request)
            if (
                previous_request is not request
                and previous_request.canonical_mapping()
                != request.canonical_mapping()
            ):
                raise ValueError(
                    f"{request.request_id}: profiler sources disagree on request contents"
                )

            item = source_evidence_by_id[request.request_id]
            previous_evidence = evidence_by_id.setdefault(request.request_id, item)
            # Nearly every additive request ID is unique. ``setdefault`` then
            # returns ``item`` itself, so recursively converting its dispatches
            # and metrics to dictionaries twice cannot discover a conflict.
            # Reserve the expensive canonical comparison for a genuine
            # duplicate ID supplied by another source transaction.
            if (
                previous_evidence is not item
                and previous_evidence.canonical_mapping()
                != item.canonical_mapping()
            ):
                raise ValueError(
                    f"{request.request_id}: profiler sources publish conflicting evidence"
                )

            physical_key = (
                _profiled_physical_candidate_key(request)
                if requests.schema_version
                == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION
                else _profiled_exact_launch_key(request)
            )
            owner = physical_candidate_owner.setdefault(physical_key, request.request_id)
            if owner != request.request_id:
                raise ValueError(
                    "profiler sources contain duplicate evidence for physical launch "
                    f"{physical_key}: {owner} and {request.request_id}"
                )

        for observation in compact:
            digest = observation.digest()
            previous = observation_by_digest.setdefault(digest, observation)
            if previous.canonical_mapping() != observation.canonical_mapping():
                raise ValueError(
                    f"profiler observation digest collision for {digest}"
                )

    assert registry_digests
    assert source_feature_schema_versions
    assert collector_version is not None
    assert request_schema_version is not None
    if not observation_by_digest:
        raise ValueError("profiler evidence composition has no completed launch witnesses")

    composed_corpus_digest = _sha256_json({
        "schema_version": "native-vnni-composed-profiler-corpus-v1",
        "sources": [
            {
                "request_manifest_digest": request_digest,
                "evidence_manifest_digest": evidence_digest,
                "witness_corpus_digest": witness_digest,
            }
            for request_digest, evidence_digest, witness_digest in sorted(
                source_provenance
            )
        ],
    })
    composed_registry_digest = _sha256_json({
        "schema_version": "native-vnni-composed-candidate-registry-v1",
        "source_registry_digests": sorted(registry_digests),
    })
    composed_requests = ProfilerRequestManifest(
        corpus_digest=composed_corpus_digest,
        candidate_registry_digest=composed_registry_digest,
        requests=tuple(request_by_id[key] for key in sorted(request_by_id)),
        learner_version=LEARNER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        schema_version=request_schema_version,
    )
    composed_evidence = ProfilerEvidenceManifest(
        request_manifest_digest=composed_requests.digest(),
        corpus_digest=composed_requests.corpus_digest,
        candidate_registry_digest=composed_requests.candidate_registry_digest,
        evidence=tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
        collector_version=collector_version,
    )
    witnesses = ObservationCorpus._from_validated(
        observation_by_digest[key] for key in sorted(observation_by_digest)
    )
    validate_profiler_evidence_coverage(
        composed_requests,
        composed_evidence,
        require_complete=True,
    )
    _validate_feature_observation_join(
        witnesses,
        composed_requests,
        frozenset(observation_by_digest),
    )
    return ComposedProfilerEvidence(
        observations=witnesses,
        requests=composed_requests,
        evidence=composed_evidence,
    )


@dataclass(frozen=True)
class ProfilerFeatureRow:
    """One physical dispatch joined to its canonical timing observation.

    Candidate latency belongs to the complete production candidate pipeline,
    while profiler metrics belong to one ordered physical dispatch in that
    pipeline.  Keeping both levels explicit prevents a multi-kernel candidate
    from accidentally treating one dispatch duration as end-to-end latency.
    """

    observation: NativeVNNIObservation
    request_id: str
    observation_digest: str
    candidate_registry_surface: str
    schedule_signature: str
    workspace_signature: str
    dispatch_count: int
    dispatch_index: int
    dispatch_kind: ProfiledDispatchKind
    kernel_name: str
    kernel_fingerprint: str
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    metric_values: dict[str, float | None]
    metric_availability: dict[str, str]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a nested deterministic representation for model tooling."""

        return {
            "observation": self.observation.canonical_mapping(),
            "request_id": self.request_id,
            "observation_digest": self.observation_digest,
            "candidate_registry_surface": self.candidate_registry_surface,
            "schedule_signature": self.schedule_signature,
            "workspace_signature": self.workspace_signature,
            "dispatch_count": self.dispatch_count,
            "dispatch_index": self.dispatch_index,
            "dispatch_kind": self.dispatch_kind.value,
            "kernel_name": self.kernel_name,
            "kernel_fingerprint": self.kernel_fingerprint,
            "grid": None if self.grid is None else list(self.grid),
            "block": None if self.block is None else list(self.block),
            "metric_values": dict(sorted(self.metric_values.items())),
            "metric_availability": dict(
                sorted(self.metric_availability.items())
            ),
        }


def _profiler_feature_csv_mapping(
    row: ProfilerFeatureRow,
    metric_ids: tuple[str, ...],
) -> dict[str, object]:
    """Flatten one profiler row using the canonical feature-table encoding."""

    mapping = row.observation.canonical_mapping()
    for name, value in tuple(mapping.items()):
        if isinstance(value, (dict, list)):
            mapping[name] = json.dumps(
                value, sort_keys=True, separators=(",", ":")
            )
        elif value is None:
            mapping[name] = ""
    mapping.update({
        "profiler.request_id": row.request_id,
        "profiler.observation_digest": row.observation_digest,
        "profiler.candidate_registry_surface": row.candidate_registry_surface,
        "profiler.schedule_signature": row.schedule_signature,
        "profiler.workspace_signature": row.workspace_signature,
        "profiler.dispatch_count": row.dispatch_count,
        "profiler.dispatch_index": row.dispatch_index,
        "profiler.dispatch_kind": row.dispatch_kind.value,
        "profiler.kernel_name": row.kernel_name,
        "profiler.kernel_fingerprint": row.kernel_fingerprint,
        "profiler.grid": json.dumps(row.grid, separators=(",", ":")),
        "profiler.block": json.dumps(row.block, separators=(",", ":")),
    })
    for metric_id in metric_ids:
        value = row.metric_values.get(metric_id)
        mapping[f"{metric_id}.value"] = "" if value is None else value
        mapping[f"{metric_id}.availability"] = (
            row.metric_availability.get(metric_id, "not_applicable")
        )
    return mapping


def _write_profiler_feature_range(task: tuple[int, int, Path]) -> Path:
    """Format one inherited feature-row range into an ordered CSV shard."""

    begin, end, output = task
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PARALLEL_PROFILER_FEATURE_FIELDNAMES,
        )
        for index in range(begin, end):
            writer.writerow(_profiler_feature_csv_mapping(
                _PARALLEL_PROFILER_FEATURE_ROWS[index],
                _PARALLEL_PROFILER_FEATURE_METRIC_IDS,
            ))
    return output


def _raw_request_profile_required(raw: Mapping[str, Any]) -> bool:
    """Evaluate the immutable request capability predicate without decoding."""

    return bool(raw["supported"]) and bool(raw["forced_route_ok"]) and (
        str(raw["execution_mode"]) != ExecutionMode.GRAPH_CAPTURED.value
        or bool(raw["graph_capture_ok"])
    )


def _raw_exact_profiler_launch_key(
    raw: Mapping[str, Any],
) -> tuple[object, ...]:
    """Project raw authenticated JSON onto exact physical launch identity."""

    return (
        str(raw["backend"]),
        str(raw["architecture_class"]),
        str(raw["operation_kind"]),
        str(raw["bundle_signature"]),
        str(raw["prepared_family_id"]),
        str(raw["packing_abi"]),
        int(raw["runtime_codebook_id"]),
        str(raw["candidate_registry_surface"]),
        str(raw["effective_candidate_id"]),
        str(raw["arithmetic_fingerprint"]),
        str(raw["candidate_policy_hash"]),
        str(raw["schedule_signature"]),
        str(raw["workspace_signature"]),
        tuple(str(value) for value in raw["prepared_resources"]),
        str(raw["threading_or_stream_mode"]),
        str(raw["execution_mode"]),
        int(raw["m"]),
        tuple(int(value) for value in raw["projection_n_vector"]),
        int(raw["aggregate_n"]),
        int(raw["k"]),
        *((int(raw["active_rows"]),) if raw.get("active_rows") is not None else ()),
    )


def _typed_raw_exact_profiler_launch_key(
    raw: Mapping[str, Any],
) -> tuple[object, ...]:
    """Project raw JSON onto the typed key used by in-memory observations."""

    key = list(_raw_exact_profiler_launch_key(raw))
    key[0] = Backend(str(key[0]))
    key[15] = ExecutionMode(str(key[15]))
    return tuple(key)


def _write_direct_profiler_export_range(
    task: tuple[int, int, Path],
) -> tuple[Path, int, int, tuple[str, ...]]:
    """Validate and emit one evidence range without returning record objects."""

    begin, end, output = task
    complete_count = 0
    unsupported_count = 0
    failed = []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=_PARALLEL_PROFILER_FEATURE_FIELDNAMES,
        )
        for index in range(begin, end):
            evidence = ProfilerEvidence.from_mapping(
                _PARALLEL_EXPORT_EVIDENCE[index]
            )
            request = ProfilerRequest.from_mapping(
                _PARALLEL_EXPORT_REQUESTS[evidence.request_id]
            )
            evidence.validate(request)
            expected_status = (
                ProfilerEvidenceStatus.COMPLETE
                if request.profile_required
                else ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED
            )
            if evidence.status != expected_status:
                failed.append(request.request_id)
                continue
            if evidence.status == ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED:
                unsupported_count += 1
                continue

            try:
                observation = _PARALLEL_EXPORT_OBSERVATIONS[
                    request.observation_digest
                ]
            except KeyError as error:
                raise ValueError(
                    f"{request.request_id}: profiler feature observations "
                    "omit the exact timing witness"
                ) from error
            if profiler_request_for_observation(observation) != request:
                raise ValueError(
                    f"{request.request_id}: profiler request is not the exact "
                    "derivative of its timing observation"
                )

            for dispatch in sorted(
                evidence.dispatches,
                key=lambda record: record.dispatch_index,
            ):
                writer.writerow(_profiler_feature_csv_mapping(
                    ProfilerFeatureRow(
                        observation=observation,
                        request_id=request.request_id,
                        observation_digest=request.observation_digest,
                        candidate_registry_surface=(
                            request.candidate_registry_surface
                        ),
                        schedule_signature=request.schedule_signature,
                        workspace_signature=request.workspace_signature,
                        dispatch_count=len(evidence.dispatches),
                        dispatch_index=dispatch.dispatch_index,
                        dispatch_kind=dispatch.dispatch_kind,
                        kernel_name=dispatch.kernel_name,
                        kernel_fingerprint=dispatch.kernel_fingerprint,
                        grid=dispatch.grid,
                        block=dispatch.block,
                        metric_values={
                            metric.metric_id: metric.value
                            for metric in dispatch.metrics
                        },
                        metric_availability={
                            metric.metric_id: metric.availability.value
                            for metric in dispatch.metrics
                        },
                    ),
                    _PARALLEL_PROFILER_FEATURE_METRIC_IDS,
                ))
            complete_count += 1
    return output, complete_count, unsupported_count, tuple(failed)


def _profiler_feature_csv_schema() -> tuple[
    tuple[str, ...], tuple[str, ...]
]:
    """Return canonical feature-table fields and the union metric inventory."""

    metric_ids = tuple(sorted({
        definition.metric_id
        for backend in Backend
        for definition in metric_definitions(backend)
    }))
    profiler_columns = (
        "profiler.request_id",
        "profiler.observation_digest",
        "profiler.candidate_registry_surface",
        "profiler.schedule_signature",
        "profiler.workspace_signature",
        "profiler.dispatch_count",
        "profiler.dispatch_index",
        "profiler.dispatch_kind",
        "profiler.kernel_name",
        "profiler.kernel_fingerprint",
        "profiler.grid",
        "profiler.block",
    )
    metric_columns = tuple(
        column
        for metric_id in metric_ids
        for column in (f"{metric_id}.value", f"{metric_id}.availability")
    )
    return (*OBSERVATION_COLUMNS, *profiler_columns, *metric_columns), metric_ids


def _validate_feature_observation_join(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    required_observation_digests: frozenset[str],
) -> dict[str, NativeVNNIObservation]:
    """Authenticate the canonical timing corpus used by a feature export.

    The profiler request manifest binds the complete observation corpus, but
    deliberately does not duplicate timing values.  Export therefore requires
    the original common-observation CSV and proves both corpus identity and the
    exact bounded-subset request derivation before exposing timing and profiler
    fields in one table. Request construction deliberately selects the three
    timing strata rather than profiling every canonical timing row.
    """

    observation_rows = observations.observations
    digest_worker_count = _offline_worker_count(
        len(observation_rows),
        environment_name="LLAMINAR_NATIVE_VNNI_FEATURE_JOIN_WORKERS",
        records_per_worker=4096,
    )
    observation_digests = _parallel_export_observation_digests(
        observation_rows,
        workers=digest_worker_count,
    )
    observations_by_digest = dict(zip(
        observation_digests,
        observation_rows,
        strict=True,
    ))
    if len(observations_by_digest) != len(observations):
        raise ValueError("profiler feature observations contain duplicate rows")
    exact_request_witnesses = (
        set(observations_by_digest) == required_observation_digests
    )
    full_corpus_authenticated = (
        False
        if exact_request_witnesses
        else observations.digest() == requests.corpus_digest
    )
    if not full_corpus_authenticated and not exact_request_witnesses:
        raise ValueError(
            "profiler feature observations belong to another timing corpus; "
            "a compact source must contain exactly the request observation "
            "witnesses"
        )
    if not required_observation_digests.issubset(observations_by_digest):
        raise ValueError(
            "profiler requests are not an authenticated timing-corpus subset"
        )
    for request in requests.requests:
        if request.observation_digest not in required_observation_digests:
            continue
        expected = profiler_request_for_observation(
            observations_by_digest[request.observation_digest]
        )
        if expected != request:
            raise ValueError(
                f"{request.request_id}: profiler request is not the exact "
                "derivative of its timing observation"
            )
    return observations_by_digest


def compact_profiler_observation_witnesses(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> ObservationCorpus:
    """Extract an exact, authenticated timing witness for retained evidence.

    Long-lived corpus bundles may retain a later aggregate that is a strict
    superset of the observations used to create an earlier profiler request
    manifest. Passing that aggregate directly to feature export would weaken
    its whole-corpus identity check. Reprofiling the same kernel candidates is
    unnecessary, however: each request already owns the cryptographic digest
    and complete semantic identity of its canonical timing observation.

    This function first authenticates profiler coverage, selects only timing
    rows referenced by completed evidence records, and then sends the compact
    corpus through the normal exact-witness validator. Missing, duplicated, or
    modified observations therefore remain hard failures. The returned corpus
    is suitable for immutable storage beside the request/evidence manifests and
    can be used by ``export-features`` without any provenance exception.
    """

    validate_profiler_evidence_coverage(
        requests, evidence, require_complete=require_complete
    )
    request_by_id = {request.request_id: request for request in requests.requests}
    required_observation_digests = frozenset(
        request_by_id[item.request_id].observation_digest
        for item in evidence.evidence
        if item.status == ProfilerEvidenceStatus.COMPLETE
    )
    observation_rows = observations.observations
    digest_worker_count = _offline_worker_count(
        len(observation_rows),
        environment_name="LLAMINAR_NATIVE_VNNI_COMPACT_WITNESS_WORKERS",
        records_per_worker=4096,
    )
    observation_digests = _parallel_export_observation_digests(
        observation_rows,
        workers=digest_worker_count,
    )
    observations_by_digest = dict(zip(
        observation_digests,
        observation_rows,
        strict=True,
    ))
    if len(observations_by_digest) != len(observations):
        raise ValueError("profiler witness source contains duplicate rows")
    missing = required_observation_digests.difference(observations_by_digest)
    if missing:
        ordered_missing = sorted(missing)
        preview_limit = 16
        preview = ordered_missing[:preview_limit]
        remainder = len(ordered_missing) - len(preview)
        suffix = f"; {remainder} more omitted" if remainder else ""
        raise ValueError(
            "profiler witness source omits "
            f"{len(ordered_missing)} requested timing observations: "
            f"{preview}{suffix}"
        )

    compact = ObservationCorpus._from_validated(
        observations_by_digest[digest]
        for digest in sorted(required_observation_digests)
    )
    _validate_feature_observation_join(
        compact,
        requests,
        required_observation_digests,
    )
    return compact


def read_profiler_feature_observation_witnesses(
    paths: Iterable[Path],
) -> ObservationCorpus:
    """Recover canonical timing witnesses embedded in exported feature CSVs.

    A profiler feature table repeats every common observation column so it can
    remain useful for offline model experiments without joining a second file.
    Multi-dispatch candidates repeat that observation once per physical kernel.
    This reader strips profiler-only columns, reconstructs the strict common
    observation, verifies the recorded observation digest, and deduplicates
    only byte-identical witnesses. It is therefore a provenance recovery path,
    not a way to trust arbitrary derived features or bypass request validation.
    """

    observations_by_digest: dict[str, NativeVNNIObservation] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = set(OBSERVATION_COLUMNS).difference(fields)
            if missing:
                raise ValueError(
                    f"{path}: profiler feature table omits common observation "
                    f"columns: {sorted(missing)}"
                )
            if "profiler.observation_digest" not in fields:
                raise ValueError(
                    f"{path}: profiler feature table omits observation digest"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    observation = NativeVNNIObservation.from_mapping({
                        name: raw[name] for name in OBSERVATION_COLUMNS
                    })
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{row_number}: {error}") from error
                digest = observation.digest()
                recorded_digest = raw["profiler.observation_digest"]
                if recorded_digest != digest:
                    raise ValueError(
                        f"{path}:{row_number}: profiler feature observation "
                        f"digest {recorded_digest!r} does not match {digest!r}"
                    )
                previous = observations_by_digest.setdefault(digest, observation)
                if previous.canonical_mapping() != observation.canonical_mapping():
                    raise ValueError(
                        f"{path}:{row_number}: duplicate profiler feature "
                        "observation digest has different content"
                    )
    return ObservationCorpus._from_validated(
        observations_by_digest[digest]
        for digest in sorted(observations_by_digest)
    )


def profiler_feature_rows(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> tuple[ProfilerFeatureRow, ...]:
    """Project validated evidence into one row per physical candidate dispatch.

    This is an offline training/diagnostic join. Hardware counters are not
    available to a production resolver and therefore cannot become runtime
    policy predicates. Unsupported candidate requests legitimately produce no
    physical dispatch row; their explicit terminal evidence remains in the
    source manifest and in the coverage report.
    """

    validate_profiler_evidence_coverage(
        requests, evidence, require_complete=require_complete
    )
    request_by_id = {request.request_id: request for request in requests.requests}
    required_observation_digests = frozenset(
        request_by_id[item.request_id].observation_digest
        for item in evidence.evidence
        if item.status == ProfilerEvidenceStatus.COMPLETE
    )
    observations_by_digest = _validate_feature_observation_join(
        observations,
        requests,
        required_observation_digests,
    )
    rows = []
    for item in sorted(evidence.evidence, key=lambda record: record.request_id):
        if item.status != ProfilerEvidenceStatus.COMPLETE:
            continue
        request = request_by_id[item.request_id]
        observation = observations_by_digest[request.observation_digest]
        for dispatch in sorted(
            item.dispatches, key=lambda record: record.dispatch_index
        ):
            values = {metric.metric_id: metric.value for metric in dispatch.metrics}
            availability = {
                metric.metric_id: metric.availability.value
                for metric in dispatch.metrics
            }
            rows.append(ProfilerFeatureRow(
                observation=observation,
                request_id=request.request_id,
                observation_digest=request.observation_digest,
                candidate_registry_surface=request.candidate_registry_surface,
                schedule_signature=request.schedule_signature,
                workspace_signature=request.workspace_signature,
                dispatch_count=len(item.dispatches),
                dispatch_index=dispatch.dispatch_index,
                dispatch_kind=dispatch.dispatch_kind,
                kernel_name=dispatch.kernel_name,
                kernel_fingerprint=dispatch.kernel_fingerprint,
                grid=dispatch.grid,
                block=dispatch.block,
                metric_values=values,
                metric_availability=availability,
            ))
    return tuple(rows)


def write_profiler_feature_csv(
    path: Path,
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> None:
    """Write a deterministic feature table with parallel row formatting.

    Formatting hundreds of megabytes of nested observation and metric fields is
    CPU-bound under the Python GIL. Fork workers inherit the validated immutable
    row inventory, write disjoint contiguous shards, and the parent concatenates
    those shards in source order. The result is byte-identical to serial CSV
    emission and is atomically published only after every shard succeeds.
    """

    rows = profiler_feature_rows(
        observations, requests, evidence, require_complete=require_complete
    )
    fieldnames, metric_ids = _profiler_feature_csv_schema()
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
        return
    if workers is None:
        workers = _offline_worker_count(
            len(rows),
            environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
        )
    if workers < 1:
        raise ValueError("profiler feature CSV worker count must be positive")
    worker_count = min(workers, len(rows))
    if worker_count <= 1 or len(rows) < parallel_threshold:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(_profiler_feature_csv_mapping(row, metric_ids))
        return

    global _PARALLEL_PROFILER_FEATURE_ROWS
    global _PARALLEL_PROFILER_FEATURE_FIELDNAMES
    global _PARALLEL_PROFILER_FEATURE_METRIC_IDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.parts-",
        dir=path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        ranges = _equal_ranges(len(rows), worker_count)
        tasks = tuple(
            (begin, end, temporary_root / f"part-{index:04d}.csv")
            for index, (begin, end) in enumerate(ranges)
        )
        _PARALLEL_PROFILER_FEATURE_ROWS = rows
        _PARALLEL_PROFILER_FEATURE_FIELDNAMES = tuple(fieldnames)
        _PARALLEL_PROFILER_FEATURE_METRIC_IDS = metric_ids
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                shard_paths = tuple(executor.map(
                    _write_profiler_feature_range,
                    tasks,
                ))
        finally:
            _PARALLEL_PROFILER_FEATURE_ROWS = ()
            _PARALLEL_PROFILER_FEATURE_FIELDNAMES = ()
            _PARALLEL_PROFILER_FEATURE_METRIC_IDS = ()

        staged_path = temporary_root / "complete.csv"
        with staged_path.open("wb") as output:
            header_buffer = io.StringIO(newline="")
            csv.DictWriter(
                header_buffer,
                fieldnames=fieldnames,
            ).writeheader()
            output.write(header_buffer.getvalue().encode("utf-8"))
            for shard_path in shard_paths:
                with shard_path.open("rb") as shard:
                    shutil.copyfileobj(shard, output, length=1024 * 1024)
        os.replace(staged_path, path)


def write_profiler_feature_csv_from_files(
    path: Path,
    observation_paths: Iterable[Path],
    request_path: Path,
    evidence_path: Path,
    *,
    workers: int | None = None,
) -> None:
    """Authenticate and export a large exact-point transaction in parallel.

    This is the production CLI path. The parent parses the two JSON documents
    once and authenticates their canonical digests, then fork workers inherit
    those immutable mappings. Each worker constructs, validates, joins, and
    formats only its own request/evidence range directly into an ordered file
    shard. No corpus-sized dataclass inventory is serialized back through a
    multiprocessing pipe, and the parent performs only small reductions plus
    byte concatenation.
    """

    request_document = _read_profiler_request_document(
        request_path,
        workers=workers,
    )
    evidence_document = _read_profiler_evidence_document(
        evidence_path,
        workers=workers,
    )
    if (
        request_document["schema_version"]
        != PROFILER_REQUEST_SCHEMA_VERSION
    ):
        observations = read_observation_csv(observation_paths)
        write_profiler_feature_csv(
            path,
            observations,
            read_profiler_request_manifest(request_path),
            read_profiler_evidence_manifest(evidence_path),
            workers=workers,
        )
        return

    if (
        str(evidence_document["request_manifest_digest"])
        != str(request_document["manifest_digest"])
    ):
        raise ValueError("profiler evidence belongs to another request manifest")
    for name in ("corpus_digest", "candidate_registry_digest"):
        if str(evidence_document[name]) != str(request_document[name]):
            raise ValueError(
                f"profiler evidence uses another request {name}"
            )

    request_records = tuple(request_document["requests"])
    request_by_id: dict[str, Mapping[str, Any]] = {}
    observation_request_ids = set()
    exact_launch_owners: dict[tuple[object, ...], str] = {}
    required_count = 0
    for raw in request_records:
        request_id = str(raw["request_id"])
        if request_id in request_by_id:
            raise ValueError("profiler request IDs must be unique")
        request_by_id[request_id] = raw
        observation_digest = str(raw["observation_digest"])
        if observation_digest in observation_request_ids:
            raise ValueError(
                "one timing observation cannot create two profiler requests"
            )
        observation_request_ids.add(observation_digest)
        if _raw_request_profile_required(raw):
            required_count += 1
            launch_key = _raw_exact_profiler_launch_key(raw)
            owner = exact_launch_owners.setdefault(launch_key, request_id)
            if owner != request_id:
                raise ValueError(
                    "profiler manifest contains duplicate evidence obligations "
                    f"for exact physical launch {launch_key}: {owner} and "
                    f"{request_id}"
                )

    evidence_records = tuple(sorted(
        evidence_document["evidence"],
        key=lambda raw: str(raw["request_id"]),
    ))
    evidence_ids = [str(raw["request_id"]) for raw in evidence_records]
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("profiler evidence contains duplicate request IDs")
    if any(
        str(raw["collector_version"])
        != str(evidence_document["collector_version"])
        for raw in evidence_records
    ):
        raise ValueError("profiler evidence manifest mixes collector generations")
    extras = sorted(set(evidence_ids).difference(request_by_id))
    if extras:
        raise ValueError(f"profiler evidence contains unknown requests: {extras}")
    missing = sorted(set(request_by_id).difference(evidence_ids))
    if missing:
        raise ValueError(
            "profiler coverage is incomplete: "
            f"missing={missing} failed=[]"
        )

    if workers is None:
        worker_count = _offline_worker_count(
            len(evidence_records),
            environment_name="LLAMINAR_NATIVE_VNNI_IO_WORKERS",
        )
    else:
        if workers < 1:
            raise ValueError("profiler feature CSV worker count must be positive")
        worker_count = min(
            workers,
            _physical_core_count(),
            len(evidence_records),
        )

    observations = tuple(read_observation_rows(observation_paths))
    observation_digests = _parallel_export_observation_digests(
        observations,
        workers=worker_count,
    )
    observations_by_digest: dict[str, NativeVNNIObservation] = {}
    for observation, digest in zip(observations, observation_digests):
        if digest in observations_by_digest:
            raise ValueError("profiler feature observations contain duplicate rows")
        observations_by_digest[digest] = observation
    required_observation_digests = frozenset(
        str(request_by_id[request_id]["observation_digest"])
        for request_id, raw in zip(evidence_ids, evidence_records)
        if str(raw["status"]) == ProfilerEvidenceStatus.COMPLETE.value
    )
    exact_request_witnesses = (
        set(observations_by_digest) == required_observation_digests
    )
    if not exact_request_witnesses:
        full_corpus = ObservationCorpus._from_validated(observations)
        if full_corpus.digest() != str(request_document["corpus_digest"]):
            raise ValueError(
                "profiler feature observations belong to another timing corpus; "
                "a compact source must contain exactly the request observation "
                "witnesses"
            )
    if not required_observation_digests.issubset(observations_by_digest):
        raise ValueError(
            "profiler requests are not an authenticated timing-corpus subset"
        )

    fieldnames, metric_ids = _profiler_feature_csv_schema()
    global _PARALLEL_EXPORT_REQUESTS
    global _PARALLEL_EXPORT_EVIDENCE
    global _PARALLEL_EXPORT_OBSERVATIONS
    global _PARALLEL_PROFILER_FEATURE_FIELDNAMES
    global _PARALLEL_PROFILER_FEATURE_METRIC_IDS
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.parts-",
        dir=path.parent,
    ) as temporary_directory:
        root = Path(temporary_directory)
        tasks = tuple(
            (begin, end, root / f"part-{index:04d}.csv")
            for index, (begin, end) in enumerate(
                _equal_ranges(len(evidence_records), worker_count)
            )
        )
        _PARALLEL_EXPORT_REQUESTS = request_by_id
        _PARALLEL_EXPORT_EVIDENCE = evidence_records
        _PARALLEL_EXPORT_OBSERVATIONS = observations_by_digest
        _PARALLEL_PROFILER_FEATURE_FIELDNAMES = fieldnames
        _PARALLEL_PROFILER_FEATURE_METRIC_IDS = metric_ids
        try:
            if worker_count == 1:
                reductions = tuple(
                    _write_direct_profiler_export_range(task) for task in tasks
                )
            else:
                with ProcessPoolExecutor(
                    max_workers=worker_count,
                    mp_context=multiprocessing.get_context("fork"),
                ) as executor:
                    reductions = tuple(executor.map(
                        _write_direct_profiler_export_range,
                        tasks,
                    ))
        finally:
            _PARALLEL_EXPORT_REQUESTS = {}
            _PARALLEL_EXPORT_EVIDENCE = ()
            _PARALLEL_EXPORT_OBSERVATIONS = {}
            _PARALLEL_PROFILER_FEATURE_FIELDNAMES = ()
            _PARALLEL_PROFILER_FEATURE_METRIC_IDS = ()

        complete_count = sum(item[1] for item in reductions)
        unsupported_count = sum(item[2] for item in reductions)
        failed = tuple(
            request_id
            for item in reductions
            for request_id in item[3]
        )
        if failed:
            raise ValueError(
                "profiler coverage is incomplete: missing=[] "
                f"failed={list(failed)}"
            )
        if complete_count != required_count:
            raise ValueError(
                "profiler coverage complete-count mismatch: "
                f"expected={required_count} actual={complete_count}"
            )
        if complete_count + unsupported_count != len(request_records):
            raise ValueError("profiler coverage terminal-state count mismatch")

        staged_path = root / "complete.csv"
        with staged_path.open("wb") as output:
            header_buffer = io.StringIO(newline="")
            csv.DictWriter(
                header_buffer,
                fieldnames=fieldnames,
            ).writeheader()
            output.write(header_buffer.getvalue().encode("utf-8"))
            for shard_path, *_ in reductions:
                with shard_path.open("rb") as shard:
                    shutil.copyfileobj(shard, output, length=1024 * 1024)
        os.replace(staged_path, path)


def _emit_requests(args: argparse.Namespace) -> int:
    """CLI implementation for deriving a request sidecar from timing CSVs."""

    corpus = read_observation_csv(Path(path) for path in args.observation)
    manifest = build_profiler_request_manifest(corpus)
    # The manifest owns independent immutable request records.  Release the
    # much larger observation corpus and all of its runtime/domain indices
    # before the parallel serializer forks or writes its output shards.
    del corpus
    write_profiler_request_manifest(Path(args.output), manifest)
    print(
        json.dumps(
            {
                "manifest_digest": manifest.digest(),
                "request_count": len(manifest.requests),
                "required_count": sum(
                    request.profile_required for request in manifest.requests
                ),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _emit_missing_requests(args: argparse.Namespace) -> int:
    """CLI implementation for one resumable exact-point delta transaction."""

    corpus = read_observation_csv(Path(path) for path in args.observation)
    covered_keys: set[tuple[object, ...]] = set()
    for path in args.covered_requests:
        covered_keys.update(read_profiler_request_coverage_keys(Path(path)))
    observations, requests = build_missing_profiler_request_transaction(
        corpus,
        (),
        covered_launch_keys=covered_keys,
    )
    if observations is None or requests is None:
        print(json.dumps({
            "status": "complete",
            "missing_request_count": 0,
        }, sort_keys=True))
        return 0

    observation_output = Path(args.output_observation)
    request_output = Path(args.output_requests)
    observation_output.parent.mkdir(parents=True, exist_ok=True)
    request_output.parent.mkdir(parents=True, exist_ok=True)
    write_observation_csv(observation_output, observations)
    write_profiler_request_manifest(request_output, requests)
    print(json.dumps({
        "status": "delta-required",
        "missing_request_count": len(requests.requests),
        "profile_required_count": sum(
            request.profile_required for request in requests.requests
        ),
        "observation_output": str(observation_output),
        "request_output": str(request_output),
        "request_manifest_digest": requests.digest(),
    }, sort_keys=True))
    return 0


def _count_uncovered_requests(args: argparse.Namespace) -> int:
    """CLI implementation for checking one durable delta transaction."""

    requests = read_profiler_request_manifest(Path(args.requests))
    covered = tuple(
        read_profiler_request_manifest(Path(path))
        for path in args.covered_requests
    )
    print(count_uncovered_profiler_requests(requests, covered))
    return 0


def _validate_evidence(args: argparse.Namespace) -> int:
    """CLI implementation for authenticating a profiler evidence join."""

    requests = read_profiler_request_manifest(Path(args.requests))
    evidence = read_profiler_evidence_manifest(Path(args.evidence))
    report = validate_profiler_evidence_coverage(
        requests, evidence, require_complete=not args.allow_incomplete
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


def _export_features(args: argparse.Namespace) -> int:
    """CLI implementation for the dispatch-level offline feature table."""

    output = Path(args.output)
    if args.allow_incomplete:
        observations = read_observation_csv(
            Path(path) for path in args.observation
        )
        write_profiler_feature_csv(
            output,
            observations,
            read_profiler_request_manifest(Path(args.requests)),
            read_profiler_evidence_manifest(Path(args.evidence)),
            require_complete=False,
        )
    else:
        write_profiler_feature_csv_from_files(
            output,
            (Path(path) for path in args.observation),
            Path(args.requests),
            Path(args.evidence),
        )
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


def _compact_witnesses(args: argparse.Namespace) -> int:
    """CLI implementation for publishing an exact profiler timing witness."""

    evidence_path = Path(args.evidence)
    with evidence_path.open("rb") as handle:
        legacy_pretty_evidence = handle.read(2) != b'{"'
    observations = (
        read_observation_csv(Path(path) for path in args.observation)
        if args.observation
        else read_profiler_feature_observation_witnesses(
            Path(path) for path in args.feature_table
        )
    )
    requests = read_profiler_request_manifest(Path(args.requests))
    evidence = read_profiler_evidence_manifest(evidence_path)
    compact = compact_profiler_observation_witnesses(
        observations,
        requests,
        evidence,
        require_complete=not args.allow_incomplete,
    )
    output = Path(args.output)
    write_observation_csv(output, compact.observations)
    if legacy_pretty_evidence:
        # Old collectors used deterministic indented JSON. The parallel reader
        # above has now authenticated every record and the exact timing join, so
        # replace only its presentation with the compact canonical encoding.
        # The semantic manifest digest remains identical, while subsequent
        # export/certification phases can use direct mmap byte authentication.
        write_profiler_evidence_manifest(evidence_path, evidence)
    print(json.dumps({
        "corpus_digest": compact.digest(),
        "evidence_encoding_migrated": legacy_pretty_evidence,
        "observation_count": len(compact),
        "output": str(output),
    }, sort_keys=True))
    return 0


def _ordered_compose_source_paths(
    observations: Sequence[str],
    requests: Sequence[str],
    evidence: Sequence[str],
) -> tuple[tuple[Path, Path, Path], ...]:
    """Order source triples to minimize large-parent profiler-reader forks."""

    source_counts = {len(observations), len(requests), len(evidence)}
    if len(source_counts) != 1:
        raise ValueError(
            "compose-evidence requires one observation, request, and evidence "
            "path per source transaction"
        )
    # Manifest readers fork physical-core workers for large record arrays. If
    # the largest base transaction is parsed first, every later worker pool
    # inherits that already-decoded object graph and spends most of its startup
    # copying page tables for data it never reads. Composition is explicitly
    # source-order independent: provenance, requests, evidence, and witnesses
    # all receive canonical sorting below. Parse the smallest request inventory
    # first so the parent reaches its peak size only for the final reader.
    ordered = tuple(sorted(
        (
            (
                Path(observation_path),
                Path(request_path),
                Path(evidence_path),
                source_index,
            )
            for source_index, (
                observation_path,
                request_path,
                evidence_path,
            ) in enumerate(zip(
                observations,
                requests,
                evidence,
                strict=True,
            ))
        ),
        key=lambda item: (item[1].stat().st_size, item[3]),
    ))
    return tuple(
        (observation, request, item_evidence)
        for observation, request, item_evidence, _index in ordered
    )


def _compose_evidence(args: argparse.Namespace) -> int:
    """CLI implementation for unioning authenticated additive transactions."""

    source_paths = _ordered_compose_source_paths(
        args.source_observation,
        args.source_requests,
        args.source_evidence,
    )
    composed = compose_profiler_evidence(
        (
            read_observation_csv((observation_path,)),
            read_profiler_request_manifest(request_path),
            read_profiler_evidence_manifest(evidence_path),
        )
        for observation_path, request_path, evidence_path in source_paths
    )
    output_observations = Path(args.output_observation)
    output_requests = Path(args.output_requests)
    output_evidence = Path(args.output_evidence)
    write_observation_csv(output_observations, composed.observations.observations)
    write_profiler_request_manifest(output_requests, composed.requests)
    write_profiler_evidence_manifest(output_evidence, composed.evidence)
    print(json.dumps({
        "evidence_count": len(composed.evidence.evidence),
        "evidence_manifest_digest": composed.evidence.digest(),
        "observation_count": len(composed.observations),
        "output_evidence": str(output_evidence),
        "output_observation": str(output_observations),
        "output_requests": str(output_requests),
        "request_count": len(composed.requests.requests),
        "request_manifest_digest": composed.requests.digest(),
    }, sort_keys=True))
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the small sidecar generation/validation command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser(
        "emit-requests",
        help="derive immutable isolated-profile requests from common observations",
    )
    emit.add_argument("--observation", action="append", required=True)
    emit.add_argument("--output", required=True)
    emit.set_defaults(handler=_emit_requests)

    missing = subparsers.add_parser(
        "emit-missing-requests",
        help=(
            "derive only exact physical launches absent from prior request "
            "transactions"
        ),
    )
    missing.add_argument("--observation", action="append", required=True)
    missing.add_argument("--covered-requests", action="append", required=True)
    missing.add_argument("--output-observation", required=True)
    missing.add_argument("--output-requests", required=True)
    missing.set_defaults(handler=_emit_missing_requests)

    count_uncovered = subparsers.add_parser(
        "count-uncovered-requests",
        help=(
            "print how many exact physical launches in one request manifest "
            "are absent from prior request transactions"
        ),
    )
    count_uncovered.add_argument("--requests", required=True)
    count_uncovered.add_argument(
        "--covered-requests", action="append", required=True
    )
    count_uncovered.set_defaults(handler=_count_uncovered_requests)

    validate = subparsers.add_parser(
        "validate-evidence",
        help="authenticate profiler evidence and enforce one result per request",
    )
    validate.add_argument("--requests", required=True)
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--allow-incomplete", action="store_true")
    validate.set_defaults(handler=_validate_evidence)

    compact = subparsers.add_parser(
        "compact-witnesses",
        help=(
            "extract exact request timing witnesses from a retained observation "
            "superset"
        ),
    )
    compact_source = compact.add_mutually_exclusive_group(required=True)
    compact_source.add_argument("--observation", action="append")
    compact_source.add_argument(
        "--feature-table",
        action="append",
        help=(
            "recover canonical timing rows embedded in an existing profiler "
            "feature CSV"
        ),
    )
    compact.add_argument("--requests", required=True)
    compact.add_argument("--evidence", required=True)
    compact.add_argument("--output", required=True)
    compact.add_argument("--allow-incomplete", action="store_true")
    compact.set_defaults(handler=_compact_witnesses)

    compose = subparsers.add_parser(
        "compose-evidence",
        help=(
            "compose complete additive profiler transactions without "
            "relaunching their physical candidates"
        ),
    )
    compose.add_argument("--source-observation", action="append", required=True)
    compose.add_argument("--source-requests", action="append", required=True)
    compose.add_argument("--source-evidence", action="append", required=True)
    compose.add_argument("--output-observation", required=True)
    compose.add_argument("--output-requests", required=True)
    compose.add_argument("--output-evidence", required=True)
    compose.set_defaults(handler=_compose_evidence)

    export = subparsers.add_parser(
        "export-features",
        help="write one offline training row per physical candidate dispatch",
    )
    export.add_argument("--observation", action="append", required=True)
    export.add_argument("--requests", required=True)
    export.add_argument("--evidence", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-incomplete", action="store_true")
    export.set_defaults(handler=_export_features)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiler sidecar command line."""

    args = build_argument_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
