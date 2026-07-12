"""CSV/JSON serialization helpers for the common NativeVNNI observation row."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .corpus import ObservationCorpus
from .schema import NativeVNNIObservation, OBSERVATION_COLUMNS


def read_observation_csv(paths: Iterable[Path]) -> ObservationCorpus:
    """Read one or more strict common-schema CSV shards into a corpus."""

    rows = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = set(OBSERVATION_COLUMNS).difference(reader.fieldnames or ())
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
    return ObservationCorpus(rows)


def write_observation_csv(path: Path, observations: Iterable[NativeVNNIObservation]) -> None:
    """Write deterministic flat rows with JSON-encoded vector/config fields."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        for observation in observations:
            observation.validate()
            row = observation.canonical_mapping()
            row["projection_n_vector"] = json.dumps(
                row["projection_n_vector"], separators=(",", ":")
            )
            row["config_json"] = json.dumps(
                row["config_json"], sort_keys=True, separators=(",", ":")
            )
            writer.writerow(row)
