"""Recover one immutable CUDA development generation from a combined corpus.

CUDA policy certification rewrites the canonical common-observation path with
both development and sealed rows. A later fit-only replay still needs the
original development-only bytes because profiler request identities and schema
migration are derived from that exact generation. This module removes the
explicitly named sealed generation without reserializing retained CSV records,
then emits the uniform development provenance needed by the refresh wrapper.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence


CONTEXT_FIELDS = (
    "run_id",
    "git_revision",
    "build_id",
    "compiler_id",
    "architecture_class",
    "device_name",
    "driver_runtime",
    "serial_m1_policy_hash",
)


def _publish_text(path: Path, text: str) -> None:
    """Atomically publish a small UTF-8 metadata file."""

    temporary = path.with_name(path.name + ".inprogress")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def extract_cuda_development_generation(
    source: Path,
    destination: Path,
    context_output: Path,
    *,
    excluded_run_id: str,
) -> tuple[int, int]:
    """Publish all non-sealed Fast rows while preserving their exact bytes.

    The excluded run ID comes from the separately persisted sealed measurement
    context. No row-count or ordering heuristic is used. Retained rows must
    describe one uniform CUDA Fast generation, and every physical line must be
    one complete CSV record so a byte-preserving copy is unambiguous.

    Returns:
        A ``(retained_rows, excluded_rows)`` pair for diagnostics.
    """

    if not excluded_run_id or "\n" in excluded_run_id or "\r" in excluded_run_id:
        raise ValueError("excluded CUDA run ID must be one non-empty line")

    temporary = destination.with_name(destination.name + ".inprogress")
    values = {field: set() for field in CONTEXT_FIELDS}
    retained_rows = 0
    excluded_rows = 0

    try:
        with source.open("rb") as input_handle, temporary.open(
            "wb"
        ) as output_handle:
            raw_header = input_handle.readline()
            if not raw_header:
                raise ValueError(f"{source}: CUDA observation corpus is empty")
            try:
                header = next(csv.reader([raw_header.decode("utf-8")]))
            except (UnicodeDecodeError, csv.Error) as exc:
                raise ValueError(
                    f"{source}: invalid CUDA observation header"
                ) from exc

            required = {
                "backend",
                "semantic_contract",
                *CONTEXT_FIELDS,
            }
            missing = required - set(header)
            if missing:
                raise ValueError(
                    f"{source}: missing CUDA resume columns {sorted(missing)}"
                )
            indexes = {name: header.index(name) for name in required}
            output_handle.write(raw_header)

            for row_number, raw_record in enumerate(input_handle, start=2):
                try:
                    record = next(csv.reader([raw_record.decode("utf-8")]))
                except (UnicodeDecodeError, csv.Error) as exc:
                    raise ValueError(
                        f"{source}: row {row_number} is not one CSV record"
                    ) from exc
                if len(record) != len(header):
                    raise ValueError(
                        f"{source}: row {row_number} spans physical lines or "
                        "has the wrong column count"
                    )
                run_id = record[indexes["run_id"]]
                if run_id == excluded_run_id:
                    excluded_rows += 1
                    continue
                if record[indexes["backend"]] != "cuda" or (
                    record[indexes["semantic_contract"]] != "Fast"
                ):
                    raise ValueError(
                        f"{source}: row {row_number} is not CUDA Fast evidence"
                    )
                for field in CONTEXT_FIELDS:
                    value = record[indexes[field]]
                    if not value or "\n" in value or "\r" in value:
                        raise ValueError(
                            f"{source}: row {row_number} has invalid {field}"
                        )
                    values[field].add(value)
                    if len(values[field]) > 1:
                        raise ValueError(
                            f"{source}: retained CUDA {field} is not uniform"
                        )
                output_handle.write(raw_record)
                retained_rows += 1

        if retained_rows == 0:
            raise ValueError(f"{source}: no CUDA development rows were retained")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    context = "".join(next(iter(values[field])) + "\n" for field in CONTEXT_FIELDS)
    _publish_text(context_output, context)
    return retained_rows, excluded_rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run byte-preserving CUDA development-generation recovery."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--context-output", required=True, type=Path)
    parser.add_argument("--exclude-run-id", required=True)
    args = parser.parse_args(argv)

    retained, excluded = extract_cuda_development_generation(
        args.source,
        args.output,
        args.context_output,
        excluded_run_id=args.exclude_run_id,
    )
    print(
        "Recovered CUDA development observation generation: "
        f"retained={retained} excluded_sealed={excluded}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
