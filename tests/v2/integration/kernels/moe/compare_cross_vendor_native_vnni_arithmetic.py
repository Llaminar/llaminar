#!/usr/bin/env python3
"""Compare exact CUDA/ROCm grouped-MoE evidence for every source codebook.

The C++ emitters own the canonical format inventory, production preparation,
and arithmetic checkpoints. This driver keeps CUDA and HIP in separate
processes, validates their emitted manifests, and joins every FP32 word and
prepared byte without a tolerance. Per-format CSV reports survive failures so
the first divergent boundary can be diagnosed without rerunning the campaign.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
from pathlib import Path
import struct
import subprocess
import sys
from typing import Iterable


@dataclass(frozen=True)
class CheckpointWord:
    """One exact FP32 checkpoint word emitted by a backend process."""

    value: str
    bits: str


@dataclass(frozen=True)
class ArtifactSpec:
    """Code-owned shape and byte contract for one binary artifact."""

    byte_count: int
    element_type: str
    rows: int
    routes: int
    width: int


def _arguments() -> argparse.Namespace:
    """Parse emitter paths and the persistent diagnostic directory."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-emitter", type=Path, required=True)
    parser.add_argument("--rocm-emitter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--format",
        dest="requested_formats",
        action="append",
        help="Run one canonical source format; repeat for a focused subset.",
    )
    return parser.parse_args()


def _run(
    command: list[str], timeout_seconds: int
) -> tuple[int, str, str]:
    """Run one bounded diagnostic subprocess and preserve its complete text."""

    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, str) else ""
        stderr = error.stderr if isinstance(error.stderr, str) else ""
        return (
            124,
            stdout,
            stderr + f"\nprocess exceeded {timeout_seconds} seconds",
        )


def _list_formats(executable: Path) -> list[str]:
    """Read the canonical registry directly from one compiled emitter."""

    if not executable.is_file():
        raise ValueError(f"missing emitter: {executable}")
    code, stdout, stderr = _run(
        [str(executable), "--list-formats"], timeout_seconds=15
    )
    if code != 0:
        raise ValueError(
            f"format inventory failed for {executable}: {stderr.strip()}"
        )
    formats = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not formats or len(formats) != len(set(formats)):
        raise ValueError(
            f"invalid canonical format inventory from {executable}: {formats}"
        )
    return formats


def _write_process_evidence(
    path: Path,
    rows: Iterable[tuple[str, str, Path, int, str, str]],
) -> None:
    """Persist every subprocess result, including failures and timeouts."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("backend", "format", "executable", "exit_code", "stdout", "stderr")
        )
        writer.writerows(rows)


def _read_manifest_rows(
    path: Path,
    expected_backend: str,
    expected_format: str,
) -> tuple[list[str], list[dict[str, str]]]:
    """Load a backend-owned CSV manifest and validate its identity columns."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or reader.fieldnames[:2] != [
            "backend",
            "format",
        ]:
            raise ValueError(f"{path}: malformed identity columns")
        rows = list(reader)
        fields = reader.fieldnames
    if not rows:
        raise ValueError(f"{path}: empty manifest")
    for row in rows:
        if row["backend"] != expected_backend:
            raise ValueError(
                f"{path}: expected backend {expected_backend}, "
                f"got {row['backend']}"
            )
        if row["format"] != expected_format:
            raise ValueError(
                f"{path}: expected format {expected_format}, "
                f"got {row['format']}"
            )
    return fields, rows


def _semantic_manifest_rows(
    fields: list[str], rows: list[dict[str, str]]
) -> list[tuple[str, ...]]:
    """Remove backend identity while preserving every semantic CSV field."""

    semantic_fields = [field for field in fields if field != "backend"]
    return [tuple(row[field] for field in semantic_fields) for row in rows]


def _compare_descriptor_manifests(
    cuda_csv: Path,
    rocm_csv: Path,
    report_path: Path,
    format_name: str,
) -> tuple[int, str]:
    """Require identical descriptor metadata for every matrix variant."""

    cuda_fields, cuda_rows = _read_manifest_rows(
        Path(f"{cuda_csv}.descriptor_manifest.csv"), "cuda", format_name
    )
    rocm_fields, rocm_rows = _read_manifest_rows(
        Path(f"{rocm_csv}.descriptor_manifest.csv"), "rocm", format_name
    )
    if cuda_fields != rocm_fields:
        raise ValueError(f"{format_name}: descriptor fields differ")
    cuda_semantic = _semantic_manifest_rows(cuda_fields, cuda_rows)
    rocm_semantic = _semantic_manifest_rows(rocm_fields, rocm_rows)
    count = max(len(cuda_semantic), len(rocm_semantic))
    mismatches = 0
    first = ""
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("row", "cuda", "rocm", "exact_match"))
        for index in range(count):
            left = cuda_semantic[index] if index < len(cuda_semantic) else ()
            right = rocm_semantic[index] if index < len(rocm_semantic) else ()
            exact = left == right
            if not exact:
                mismatches += 1
                if not first:
                    first = f"descriptor row {index}: {left} != {right}"
            writer.writerow((index, repr(left), repr(right), str(exact).lower()))
    return mismatches, first


def _read_checkpoint_manifest(
    path: Path,
    backend: str,
    format_name: str,
) -> dict[tuple[str, str], int]:
    """Read declared checkpoint sizes and reject duplicate stage identities."""

    fields, rows = _read_manifest_rows(path, backend, format_name)
    expected_fields = ["backend", "format", "case", "stage", "elements"]
    if fields != expected_fields:
        raise ValueError(f"{path}: expected fields {expected_fields}, got {fields}")
    result: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["case"], row["stage"])
        if key in result:
            raise ValueError(f"{path}: duplicate checkpoint manifest key {key}")
        count = int(row["elements"])
        if count <= 0:
            raise ValueError(f"{path}: non-positive checkpoint size at {key}")
        result[key] = count
    return result


def _read_checkpoints(
    path: Path,
    expected_backend: str,
    expected_format: str,
    manifest: dict[tuple[str, str], int],
) -> dict[tuple[str, str, int], CheckpointWord]:
    """Load and structurally validate every exact FP32 checkpoint word."""

    checkpoints: dict[tuple[str, str, int], CheckpointWord] = {}
    counts: dict[tuple[str, str], int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        expected_fields = [
            "backend",
            "format",
            "case",
            "stage",
            "index",
            "value",
            "bits",
        ]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"{path}: expected fields {expected_fields}, got {reader.fieldnames}"
            )
        for row in reader:
            if row["backend"] != expected_backend:
                raise ValueError(f"{path}: unexpected backend {row['backend']}")
            if row["format"] != expected_format:
                raise ValueError(f"{path}: unexpected format {row['format']}")
            stage_key = (row["case"], row["stage"])
            if stage_key not in manifest:
                raise ValueError(f"{path}: undeclared checkpoint {stage_key}")
            index = int(row["index"])
            if index < 0 or index >= manifest[stage_key]:
                raise ValueError(f"{path}: out-of-range index at {stage_key}")
            key = (*stage_key, index)
            if key in checkpoints:
                raise ValueError(f"{path}: duplicate checkpoint {key}")
            bits = row["bits"].lower()
            if len(bits) != 10 or not bits.startswith("0x"):
                raise ValueError(f"{path}: malformed FP32 word {bits!r} at {key}")
            int(bits[2:], 16)
            checkpoints[key] = CheckpointWord(row["value"], bits)
            counts[stage_key] = counts.get(stage_key, 0) + 1
    if counts != manifest:
        raise ValueError(
            f"{path}: checkpoint counts differ from manifest: "
            f"expected={manifest}, actual={counts}"
        )
    return checkpoints


def _absolute_error(cuda_value: str, rocm_value: str) -> float:
    """Compute a readable error while exact FP32 words remain authoritative."""

    left = float(cuda_value)
    right = float(rocm_value)
    if math.isnan(left) or math.isnan(right):
        return math.nan
    return abs(left - right)


def _compare_checkpoints(
    cuda_csv: Path,
    rocm_csv: Path,
    report_path: Path,
    format_name: str,
) -> tuple[int, str]:
    """Write a complete FP32 join and return mismatch count/first location."""

    cuda_manifest = _read_checkpoint_manifest(
        Path(f"{cuda_csv}.checkpoint_manifest.csv"), "cuda", format_name
    )
    rocm_manifest = _read_checkpoint_manifest(
        Path(f"{rocm_csv}.checkpoint_manifest.csv"), "rocm", format_name
    )
    if cuda_manifest != rocm_manifest:
        raise ValueError(f"{format_name}: checkpoint manifests differ")
    cuda = _read_checkpoints(cuda_csv, "cuda", format_name, cuda_manifest)
    rocm = _read_checkpoints(rocm_csv, "rocm", format_name, rocm_manifest)
    if cuda.keys() != rocm.keys():
        raise ValueError(f"{format_name}: checkpoint key sets differ")

    mismatches = 0
    first = ""
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "format",
                "case",
                "stage",
                "index",
                "cuda_value",
                "rocm_value",
                "cuda_bits",
                "rocm_bits",
                "exact_match",
                "abs_error",
            )
        )
        for case_label, stage, index in sorted(cuda):
            left = cuda[(case_label, stage, index)]
            right = rocm[(case_label, stage, index)]
            exact = left.bits == right.bits
            if not exact:
                mismatches += 1
                if not first:
                    first = (
                        f"{case_label}/{stage}[{index}] "
                        f"{left.bits} != {right.bits}"
                    )
            writer.writerow(
                (
                    format_name,
                    case_label,
                    stage,
                    index,
                    left.value,
                    right.value,
                    left.bits,
                    right.bits,
                    str(exact).lower(),
                    format(_absolute_error(left.value, right.value), ".9g"),
                )
            )
    return mismatches, first


def _read_artifact_manifest(
    path: Path,
    backend: str,
    format_name: str,
) -> dict[str, ArtifactSpec]:
    """Read binary artifact declarations and prove their shaped byte counts."""

    fields, rows = _read_manifest_rows(path, backend, format_name)
    expected_fields = [
        "backend",
        "format",
        "artifact",
        "bytes",
        "element_type",
        "rows",
        "routes",
        "width",
    ]
    if fields != expected_fields:
        raise ValueError(f"{path}: expected fields {expected_fields}, got {fields}")
    element_sizes = {"bytes": 1, "i8": 1, "fp16_bits": 2, "u32": 4, "fp32": 4}
    result: dict[str, ArtifactSpec] = {}
    for row in rows:
        name = row["artifact"]
        if name in result:
            raise ValueError(f"{path}: duplicate artifact {name}")
        spec = ArtifactSpec(
            byte_count=int(row["bytes"]),
            element_type=row["element_type"],
            rows=int(row["rows"]),
            routes=int(row["routes"]),
            width=int(row["width"]),
        )
        if spec.byte_count <= 0 or spec.element_type not in element_sizes:
            raise ValueError(f"{path}: invalid artifact contract for {name}")
        shaped = spec.rows > 0 or spec.routes > 0 or spec.width > 0
        if shaped:
            if min(spec.rows, spec.routes, spec.width) <= 0:
                raise ValueError(f"{path}: partial shape for {name}")
            expected_bytes = (
                spec.rows
                * spec.routes
                * spec.width
                * element_sizes[spec.element_type]
            )
            if expected_bytes != spec.byte_count:
                raise ValueError(
                    f"{path}: {name} declares {spec.byte_count} bytes, "
                    f"shape requires {expected_bytes}"
                )
        result[name] = spec
    return result


def _first_byte_mismatch(left: bytes, right: bytes) -> int:
    """Return the first differing byte offset, or -1 for exact equality."""

    if left == right:
        return -1
    for offset in range(min(len(left), len(right))):
        if left[offset] != right[offset]:
            return offset
    return min(len(left), len(right))


def _fp32_details(
    left: bytes,
    right: bytes,
    offset: int,
    spec: ArtifactSpec,
) -> tuple[str, str, str, str, str]:
    """Decode a first FP32 mismatch into semantic route coordinates."""

    if spec.element_type != "fp32" or offset < 0:
        return "", "", "", "", ""
    element = offset // 4
    row_width = spec.routes * spec.width
    row = element // row_width
    within_row = element % row_width
    route = within_row // spec.width
    column = within_row % spec.width
    aligned = element * 4
    if aligned + 4 > min(len(left), len(right)):
        return str(row), str(route), str(column), "EOF", "EOF"
    return (
        str(row),
        str(route),
        str(column),
        format(struct.unpack_from("<f", left, aligned)[0], ".9g"),
        format(struct.unpack_from("<f", right, aligned)[0], ".9g"),
    )


def _compare_blobs(
    cuda_csv: Path,
    rocm_csv: Path,
    report_path: Path,
    format_name: str,
) -> tuple[int, str]:
    """Compare every prepared and canonical binary boundary byte-for-byte."""

    cuda_manifest = _read_artifact_manifest(
        Path(f"{cuda_csv}.artifact_manifest.csv"), "cuda", format_name
    )
    rocm_manifest = _read_artifact_manifest(
        Path(f"{rocm_csv}.artifact_manifest.csv"), "rocm", format_name
    )
    if cuda_manifest != rocm_manifest:
        raise ValueError(f"{format_name}: artifact manifests differ")

    mismatches = 0
    first = ""
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "format",
                "artifact",
                "expected_bytes",
                "cuda_bytes",
                "rocm_bytes",
                "exact_match",
                "first_mismatch_offset",
                "row",
                "route",
                "column",
                "cuda_value",
                "rocm_value",
                "cuda_byte",
                "rocm_byte",
            )
        )
        for name, spec in sorted(cuda_manifest.items()):
            cuda_path = Path(f"{cuda_csv}.{name}")
            rocm_path = Path(f"{rocm_csv}.{name}")
            left = cuda_path.read_bytes()
            right = rocm_path.read_bytes()
            if len(left) != spec.byte_count or len(right) != spec.byte_count:
                raise ValueError(
                    f"{format_name}/{name}: expected {spec.byte_count} bytes, "
                    f"got CUDA={len(left)} ROCm={len(right)}"
                )
            offset = _first_byte_mismatch(left, right)
            exact = offset < 0
            cuda_byte = "" if exact else f"0x{left[offset]:02x}"
            rocm_byte = "" if exact else f"0x{right[offset]:02x}"
            row, route, column, cuda_value, rocm_value = _fp32_details(
                left, right, offset, spec
            )
            if not exact:
                mismatches += 1
                if not first:
                    first = (
                        f"{name}[{offset}] {cuda_byte} != {rocm_byte}"
                    )
            writer.writerow(
                (
                    format_name,
                    name,
                    spec.byte_count,
                    len(left),
                    len(right),
                    str(exact).lower(),
                    "" if exact else offset,
                    row,
                    route,
                    column,
                    cuda_value,
                    rocm_value,
                    cuda_byte,
                    rocm_byte,
                )
            )
    return mismatches, first


def main() -> int:
    """Run the canonical inventory and enforce all exact cross-vendor gates."""

    args = _arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        cuda_formats = _list_formats(args.cuda_emitter)
        rocm_formats = _list_formats(args.rocm_emitter)
    except ValueError as error:
        print(f"invalid format inventory: {error}", file=sys.stderr)
        return 1
    if cuda_formats != rocm_formats:
        print(
            f"backend format inventories differ: CUDA={cuda_formats} "
            f"ROCm={rocm_formats}",
            file=sys.stderr,
        )
        return 1

    selected_formats = cuda_formats
    if args.requested_formats:
        unknown = sorted(set(args.requested_formats) - set(cuda_formats))
        if unknown:
            print(f"unknown requested formats: {unknown}", file=sys.stderr)
            return 1
        requested = set(args.requested_formats)
        selected_formats = [
            format_name for format_name in cuda_formats
            if format_name in requested
        ]

    process_rows: list[tuple[str, str, Path, int, str, str]] = []
    summary_rows: list[tuple[str, str, int, int, int, str]] = []
    total_failures = 0
    for format_name in selected_formats:
        format_dir = args.output_dir / format_name
        format_dir.mkdir(parents=True, exist_ok=True)
        cuda_csv = format_dir / "cuda.csv"
        rocm_csv = format_dir / "rocm.csv"
        process_codes: dict[str, int] = {}
        for backend, executable, output_csv in (
            ("cuda", args.cuda_emitter, cuda_csv),
            ("rocm", args.rocm_emitter, rocm_csv),
        ):
            code, stdout, stderr = _run(
                [str(executable), format_name, str(output_csv)],
                timeout_seconds=180,
            )
            process_codes[backend] = code
            process_rows.append(
                (backend, format_name, executable, code, stdout, stderr)
            )

        if any(code != 0 for code in process_codes.values()):
            total_failures += 1
            detail = " ".join(
                f"{backend}={code}" for backend, code in process_codes.items()
            )
            summary_rows.append((format_name, "process_failed", 0, 0, 0, detail))
            continue

        try:
            descriptor_mismatches, descriptor_first = (
                _compare_descriptor_manifests(
                    cuda_csv,
                    rocm_csv,
                    format_dir / "descriptor_comparison.csv",
                    format_name,
                )
            )
            checkpoint_mismatches, checkpoint_first = _compare_checkpoints(
                cuda_csv,
                rocm_csv,
                format_dir / "checkpoint_comparison.csv",
                format_name,
            )
            blob_mismatches, blob_first = _compare_blobs(
                cuda_csv,
                rocm_csv,
                format_dir / "blob_comparison.csv",
                format_name,
            )
        except (OSError, ValueError) as error:
            total_failures += 1
            summary_rows.append((format_name, "invalid_evidence", 0, 0, 0, str(error)))
            continue

        first = descriptor_first or checkpoint_first or blob_first
        status = (
            "pass"
            if descriptor_mismatches == 0
            and checkpoint_mismatches == 0
            and blob_mismatches == 0
            else "mismatch"
        )
        if status != "pass":
            total_failures += 1
        summary_rows.append(
            (
                format_name,
                status,
                descriptor_mismatches,
                checkpoint_mismatches,
                blob_mismatches,
                first,
            )
        )

    _write_process_evidence(
        args.output_dir / "process_execution.csv", process_rows
    )
    with (args.output_dir / "format_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "format",
                "status",
                "descriptor_mismatches",
                "checkpoint_word_mismatches",
                "binary_artifact_mismatches",
                "first_failure",
            )
        )
        writer.writerows(summary_rows)

    if total_failures:
        print(
            f"cross-vendor grouped-MoE arithmetic failed for "
            f"{total_failures}/{len(selected_formats)} formats; "
            f"evidence: {args.output_dir}",
            file=sys.stderr,
        )
        return 1
    print(
        f"cross-vendor grouped-MoE arithmetic is byte-exact for all "
        f"{len(selected_formats)} selected canonical source formats; "
        f"evidence: {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
