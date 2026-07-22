"""CSV/JSON serialization helpers for the common NativeVNNI observation row."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

from .corpus import ObservationCorpus
from .schema import NativeVNNIObservation, OBSERVATION_COLUMNS


OPTIONAL_OBSERVATION_COLUMNS = frozenset({
    "launch_k_tiles",
    "launch_n_block_chunks",
    "adaptive_timing_evidence",
})


# Large policy corpora are already resident in the analyzer.  Fork workers
# inherit this immutable tuple copy-on-write, so each task carries only row
# bounds and a shard pathname instead of repeatedly pickling observation rows.
_PARALLEL_WRITE_OBSERVATIONS: tuple[NativeVNNIObservation, ...] = ()


def _physical_core_count() -> int:
    """Return affinity-visible physical cores for offline CSV processing."""

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


def _observation_csv_row(
    observation: NativeVNNIObservation,
) -> dict[str, object]:
    """Validate and flatten one observation using the canonical CSV encoding."""

    observation.validate()
    row = observation.canonical_mapping()
    row["projection_n_vector"] = json.dumps(
        row["projection_n_vector"], separators=(",", ":")
    )
    row["config_json"] = json.dumps(
        row["config_json"], sort_keys=True, separators=(",", ":")
    )
    if "adaptive_timing_evidence" in row:
        row["adaptive_timing_evidence"] = json.dumps(
            row["adaptive_timing_evidence"],
            sort_keys=True,
            separators=(",", ":"),
        )
    return row


def _write_observation_range(task: tuple[int, int, Path]) -> Path:
    """Serialize one disjoint row range for deterministic parent assembly."""

    start, stop, output = task
    if not _PARALLEL_WRITE_OBSERVATIONS:
        raise RuntimeError("parallel observation writer context is unavailable")
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        for index in range(start, stop):
            writer.writerow(_observation_csv_row(
                _PARALLEL_WRITE_OBSERVATIONS[index]
            ))
    return output


def _write_observations_serial(
    handle,
    observations: Iterable[NativeVNNIObservation],
) -> None:
    """Write canonical rows to an existing text handle."""

    writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
    writer.writeheader()
    for observation in observations:
        writer.writerow(_observation_csv_row(observation))


def _validated_observation_header(path: Path) -> tuple[tuple[str, ...], int]:
    """Return one strict CSV header and the first data-row byte offset."""

    with path.open("rb") as handle:
        encoded_header = handle.readline()
        data_offset = handle.tell()
    try:
        header = next(csv.reader([encoded_header.decode("utf-8")]))
    except (UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"{path}: invalid common observation header") from error
    missing = (
        set(OBSERVATION_COLUMNS).difference(header)
        - OPTIONAL_OBSERVATION_COLUMNS
    )
    unexpected = set(header).difference(OBSERVATION_COLUMNS)
    if missing or unexpected:
        raise ValueError(
            f"{path}: common observation header mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )
    return tuple(header), data_offset


def _observation_file_ranges(
    path: Path,
    data_offset: int,
    partition_count: int,
) -> tuple[tuple[int, int], ...]:
    """Split a canonical one-row-per-line CSV at complete row boundaries."""

    file_size = path.stat().st_size
    data_size = max(0, file_size - data_offset)
    if data_size == 0:
        return ()
    partition_count = min(partition_count, data_size)
    boundaries = [data_offset]
    with path.open("rb") as handle:
        for partition_index in range(1, partition_count):
            target = data_offset + data_size * partition_index // partition_count
            handle.seek(target)
            handle.readline()
            boundary = handle.tell()
            if boundary < file_size:
                boundaries.append(boundary)
    boundaries.append(file_size)
    boundaries = sorted(set(boundaries))
    return tuple(
        (begin, end)
        for begin, end in zip(boundaries, boundaries[1:])
        if begin < end
    )


def _read_observation_range(
    task: tuple[Path, tuple[str, ...], int, int],
) -> tuple[NativeVNNIObservation, ...]:
    """Parse and validate one complete-line CSV byte range in a worker."""

    path, fieldnames, begin, end = task
    with path.open("rb") as handle:
        handle.seek(begin)
        encoded = handle.read(end - begin)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}:byte-{begin}: invalid UTF-8") from error
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        fieldnames=fieldnames,
    )
    rows = []
    for local_row, raw in enumerate(reader):
        try:
            rows.append(NativeVNNIObservation.from_mapping(raw))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}:byte-{begin}:row-{local_row}: {error}"
            ) from error
    return tuple(rows)


def read_observation_rows(
    paths: Iterable[Path],
    *,
    workers: int | None = None,
    parallel_threshold_bytes: int = 16 * 1024 * 1024,
) -> tuple[NativeVNNIObservation, ...]:
    """Parse strict CSV shards in deterministic physical-core partitions.

    Generated observation CSVs guarantee one canonical record per physical
    line. Large inputs are therefore split only after a newline and decoded by
    fork workers. The parent receives validated row tuples in file/range order,
    preserving the serial corpus exactly while removing the former GIL-bound
    parse stage from fit-only replays.
    """

    path_list = tuple(paths)
    headers = []
    total_bytes = 0
    for path in path_list:
        header, data_offset = _validated_observation_header(path)
        headers.append((path, header, data_offset))
        total_bytes += max(0, path.stat().st_size - data_offset)

    if workers is None:
        workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_IO_WORKERS",
            str(_physical_core_count()),
        ))
    if workers < 1:
        raise ValueError("observation CSV worker count must be positive")
    worker_count = min(workers, _physical_core_count())
    if worker_count <= 1 or total_bytes < parallel_threshold_bytes:
        rows = []
        for path, _header, _data_offset in headers:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row_number, raw in enumerate(reader, start=2):
                    try:
                        rows.append(NativeVNNIObservation.from_mapping(raw))
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"{path}:{row_number}: {exc}") from exc
        return tuple(rows)

    target_bytes = max(1, (total_bytes + worker_count - 1) // worker_count)
    tasks = []
    for path, header, data_offset in headers:
        data_size = max(0, path.stat().st_size - data_offset)
        partitions = max(1, (data_size + target_bytes - 1) // target_bytes)
        tasks.extend(
            (path, header, begin, end)
            for begin, end in _observation_file_ranges(
                path, data_offset, partitions
            )
        )
    if not tasks:
        return ()
    with ProcessPoolExecutor(
        max_workers=min(worker_count, len(tasks)),
        mp_context=multiprocessing.get_context("fork"),
    ) as executor:
        partitions = tuple(executor.map(_read_observation_range, tasks))
    return tuple(row for partition in partitions for row in partition)
def read_observation_csv(paths: Iterable[Path]) -> ObservationCorpus:
    """Read strict CSV shards and build a candidate-consistent corpus index."""

    return ObservationCorpus._from_validated(read_observation_rows(paths))


def write_observation_csv(
    path: Path,
    observations: Iterable[NativeVNNIObservation],
    *,
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> None:
    """Write deterministic flat rows with parallel large-corpus formatting.

    CSV formatting is CPU-bound and held by Python's GIL.  Above
    ``parallel_threshold``, forked workers serialize disjoint row ranges to
    private files.  The parent concatenates those files in source order and
    atomically publishes the result, making the parallel output byte-identical
    to the serial encoding while avoiding multi-process writes to one handle.
    """

    global _PARALLEL_WRITE_OBSERVATIONS

    rows = observations if isinstance(observations, tuple) else tuple(observations)
    if workers is None:
        workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_IO_WORKERS",
            str(_physical_core_count()),
        ))
    if workers < 1:
        raise ValueError("observation CSV worker count must be positive")
    worker_count = min(workers, len(rows))
    if worker_count <= 1 or len(rows) < parallel_threshold:
        with path.open("w", newline="", encoding="utf-8") as handle:
            _write_observations_serial(handle, rows)
        return

    rows_per_worker = (len(rows) + worker_count - 1) // worker_count
    with tempfile.TemporaryDirectory(
        prefix=f".{path.name}.parts-",
        dir=path.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        tasks = tuple(
            (
                start,
                min(start + rows_per_worker, len(rows)),
                temporary_root / f"part-{part:04d}.csv",
            )
            for part, start in enumerate(range(0, len(rows), rows_per_worker))
        )
        _PARALLEL_WRITE_OBSERVATIONS = tuple(rows)
        try:
            with ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                shard_paths = tuple(executor.map(
                    _write_observation_range,
                    tasks,
                ))
        finally:
            _PARALLEL_WRITE_OBSERVATIONS = ()

        staged_path = temporary_root / "complete.csv"
        with staged_path.open("wb") as output:
            header_buffer = io.StringIO(newline="")
            csv.DictWriter(
                header_buffer,
                fieldnames=OBSERVATION_COLUMNS,
            ).writeheader()
            output.write(header_buffer.getvalue().encode("utf-8"))
            for shard_path in shard_paths:
                with shard_path.open("rb") as shard:
                    shutil.copyfileobj(shard, output, length=1024 * 1024)
        os.replace(staged_path, path)
