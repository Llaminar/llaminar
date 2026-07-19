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


OPTIONAL_OBSERVATION_COLUMNS = frozenset({"adaptive_timing_evidence"})


# Large policy corpora are already resident in the analyzer.  Fork workers
# inherit this immutable tuple copy-on-write, so each task carries only row
# bounds and a shard pathname instead of repeatedly pickling observation rows.
_PARALLEL_WRITE_OBSERVATIONS: tuple[NativeVNNIObservation, ...] = ()


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


def read_observation_rows(
    paths: Iterable[Path],
) -> tuple[NativeVNNIObservation, ...]:
    """Parse and validate strict common-schema CSV shards exactly once."""

    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = (
                set(OBSERVATION_COLUMNS).difference(reader.fieldnames or ())
                - OPTIONAL_OBSERVATION_COLUMNS
            )
            unexpected = set(reader.fieldnames or ()).difference(OBSERVATION_COLUMNS)
            if missing or unexpected:
                raise ValueError(
                    f"{path}: common observation header mismatch: "
                    f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    rows.append(NativeVNNIObservation.from_mapping(raw))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
    return tuple(rows)


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
            str(min(16, len(os.sched_getaffinity(0)))),
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
