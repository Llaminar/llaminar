"""Validate one CPU prefill aggregate/timing shard before publication.

The refresh driver writes trainer output to ``.inprogress`` files and only
publishes a shard after this module accepts the pair under the same production
adapter contract used by policy fitting. The same validation is repeated for
an existing shard before resume reuses it. This closes a lifecycle gap where
an old or interrupted nonempty CSV could otherwise be mistaken for completed,
installable evidence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .adapters.cpu_prefill import (
    CPUPrefillAdapterContext,
    adapt_cpu_prefill_csv,
)
from .adapters.evidence import raw_corpus_id
from .profiles import MeasurementProfile


def validate_cpu_prefill_partial(
    aggregate: Path,
    timing_sidecar: Path,
    *,
    candidate_expansion: bool = False,
    expected_m_values: tuple[int, ...] | None = None,
    require_append_compatible: bool = False,
) -> tuple[int, ...]:
    """Validate one raw shard pair using the installable adapter contract.

    Run-level provenance is intentionally synthetic because this transaction
    gate validates trainer-owned row and timing evidence, not the later common
    corpus envelope. The raw shard bytes still receive a real corpus digest,
    and every numerical, route, byte-equality, complete-round, adaptive timing,
    and sidecar-integrity check remains active.
    """

    aggregate = Path(aggregate)
    timing_sidecar = Path(timing_sidecar)
    if require_append_compatible:
        with aggregate.open(newline="") as stream:
            fieldnames = csv.DictReader(stream).fieldnames or ()
        if "complete_round_probe_duration_us" not in fieldnames:
            raise ValueError(
                "CPU prefill resumable shard uses a legacy append schema"
            )
    context = CPUPrefillAdapterContext(
        profile=MeasurementProfile.PARTIAL_PRODUCTION,
        run_id="cpu-prefill-partial-validation",
        corpus_id=raw_corpus_id((aggregate, timing_sidecar)),
        git_revision="partial-validation",
        build_id="partial-validation",
        compiler_id="partial-validation",
        architecture_class="partial-validation",
        device_name="partial-validation",
        driver_runtime="partial-validation",
        serial_m1_policy_hash="sha256:" + "0" * 64,
        raw_timing_sidecar_retained=True,
        anchored_candidate_expansion=candidate_expansion,
    )
    adapt_cpu_prefill_csv(
        (aggregate,),
        context,
        timing_sidecars=(timing_sidecar,),
    )
    aggregate_m_values = _ordered_m_values(aggregate)
    timing_m_values = _ordered_m_values(timing_sidecar)
    if aggregate_m_values != timing_m_values:
        raise ValueError(
            "CPU prefill aggregate/timing M inventories differ: "
            f"aggregate={aggregate_m_values!r} timing={timing_m_values!r}"
        )
    if expected_m_values is not None and aggregate_m_values != expected_m_values:
        raise ValueError(
            "CPU prefill partial does not contain the exact planned M inventory: "
            f"observed={aggregate_m_values!r} expected={expected_m_values!r}"
        )
    return aggregate_m_values


def _ordered_m_values(path: Path) -> tuple[int, ...]:
    """Return the contiguous ordered M phases represented by one CSV.

    The C++ trainer flushes aggregate and timing rows after each complete M
    phase. An interrupted process can therefore leave a valid prefix that is
    safe to append to, but a repeated M after a later phase would indicate a
    malformed or manually concatenated shard and must never be resumed.
    """

    values: list[int] = []
    seen: set[int] = set()
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "m" not in reader.fieldnames:
            raise ValueError(f"CPU prefill CSV lacks an m column: {path}")
        for row in reader:
            value = int(row["m"])
            if not values or values[-1] != value:
                if value in seen:
                    raise ValueError(
                        f"CPU prefill CSV repeats a completed M phase: {path}: M={value}"
                    )
                seen.add(value)
                values.append(value)
    if not values:
        raise ValueError(f"CPU prefill CSV contains no measurement rows: {path}")
    return tuple(values)


def _parse_m_values(raw: str) -> tuple[int, ...]:
    """Parse a positive, unique, ordered comma-separated M inventory."""

    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or any(value <= 1 for value in values):
        raise ValueError("CPU prefill M values must all be greater than one")
    if len(values) != len(set(values)):
        raise ValueError("CPU prefill M inventory contains duplicates")
    return values


def main() -> int:
    """Parse a shard pair and return nonzero when it is not installable."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--timing-sidecar", type=Path, required=True)
    parser.add_argument(
        "--candidate-expansion",
        action="store_true",
        help=(
            "Accept the shorter complete-round timing floors reserved for an "
            "anchor-normalized additive candidate cohort"
        ),
    )
    parser.add_argument(
        "--expected-m-values",
        help="Require this exact ordered comma-separated M inventory",
    )
    parser.add_argument(
        "--planned-m-values",
        help=(
            "Require the observed inventory to be a prefix of this ordered "
            "inventory; used only to resume an interrupted trainer process"
        ),
    )
    parser.add_argument(
        "--print-m-values",
        action="store_true",
        help="Print the authenticated observed M inventory",
    )
    parser.add_argument(
        "--print-missing-m-values",
        action="store_true",
        help="Print the unmeasured suffix of --planned-m-values",
    )
    parser.add_argument(
        "--require-append-compatible",
        action="store_true",
        help="Reject a valid historical shard whose header cannot be appended",
    )
    args = parser.parse_args()
    if args.expected_m_values and args.planned_m_values:
        parser.error("use only one of --expected-m-values and --planned-m-values")
    if args.print_missing_m_values and not args.planned_m_values:
        parser.error("--print-missing-m-values requires --planned-m-values")
    expected = (
        _parse_m_values(args.expected_m_values)
        if args.expected_m_values
        else None
    )
    observed = validate_cpu_prefill_partial(
        args.input,
        args.timing_sidecar,
        candidate_expansion=args.candidate_expansion,
        expected_m_values=expected,
        require_append_compatible=args.require_append_compatible,
    )
    planned = (
        _parse_m_values(args.planned_m_values)
        if args.planned_m_values
        else None
    )
    if planned is not None and observed != planned[: len(observed)]:
        raise ValueError(
            "CPU prefill resumable shard is not an ordered plan prefix: "
            f"observed={observed!r} planned={planned!r}"
        )
    if args.print_m_values:
        print(",".join(str(value) for value in observed))
    if args.print_missing_m_values:
        assert planned is not None
        print(",".join(str(value) for value in planned[len(observed) :]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
