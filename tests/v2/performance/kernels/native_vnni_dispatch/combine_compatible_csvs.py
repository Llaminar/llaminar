#!/usr/bin/env python3
"""Atomically combine compatible CSV evidence generations by column union.

NativeVNNI measurement schemas may gain provenance-only columns between
refinement rounds.  The older rows remain immutable and valid, but a strict
textual header comparison cannot publish the additive development aggregate.
This utility preserves every input row, orders columns by first appearance,
and leaves newly introduced fields empty for evidence that predates them.

It is intentionally separate from the refresh driver's strict same-generation
combiner.  A caller must opt into compatibility merging only at a reviewed
cross-generation boundary where the downstream adapter validates semantics.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Sequence


def _read_header(path: Path) -> tuple[str, ...]:
    """Read one nonempty, duplicate-free CSV header from ``path``."""

    with path.open(newline="", encoding="utf-8") as source:
        header = next(csv.reader(source), None)
    if not header:
        raise ValueError(f"empty CSV header while combining {path}")
    if len(header) != len(set(header)):
        raise ValueError(f"duplicate CSV header field while combining {path}")
    return tuple(header)


def combine_compatible_csvs(output: Path, inputs: Sequence[Path]) -> None:
    """Publish the ordered column union of ``inputs`` atomically at ``output``.

    Input files are streamed twice: once for their small headers and once for
    their rows.  This bounds memory use for multi-gigabyte timing corpora and
    detects a source file whose header changes during publication.
    """

    if not inputs:
        raise ValueError("at least one compatible CSV input is required")

    headers = tuple(_read_header(path) for path in inputs)
    fieldnames = tuple(dict.fromkeys(field for header in headers for field in header))
    temporary = output.with_name(f"{output.name}.inprogress")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(
                destination,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            for path, expected_header in zip(inputs, headers, strict=True):
                with path.open(newline="", encoding="utf-8") as source:
                    reader = csv.DictReader(source)
                    if tuple(reader.fieldnames or ()) != expected_header:
                        raise ValueError(
                            f"CSV header changed while combining {path}"
                        )
                    writer.writerows(reader)
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    """Parse the command line and publish one compatible aggregate."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    combine_compatible_csvs(args.output, args.input)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
