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
) -> None:
    """Validate one raw shard pair using the installable adapter contract.

    Run-level provenance is intentionally synthetic because this transaction
    gate validates trainer-owned row and timing evidence, not the later common
    corpus envelope. The raw shard bytes still receive a real corpus digest,
    and every numerical, route, byte-equality, complete-round, adaptive timing,
    and sidecar-integrity check remains active.
    """

    aggregate = Path(aggregate)
    timing_sidecar = Path(timing_sidecar)
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
    args = parser.parse_args()
    validate_cpu_prefill_partial(
        args.input,
        args.timing_sidecar,
        candidate_expansion=args.candidate_expansion,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
