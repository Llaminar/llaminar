"""Strict interleaved timing evidence and simultaneous regret certification.

Broad NativeVNNI sweeps identify a provisional policy candidate and the
runtime-representable exact reference. They do not, by themselves, establish a
sub-five-percent performance difference: hundreds of candidates, device clock
drift, and winner selection all bias an unpaired minimum. This module consumes
the second-stage timing protocol where selected and exact candidates alternate
inside deterministic pairs, then bootstraps the maximum regret across every
required cell with one simultaneous one-sided confidence bound.

Paired evidence is deliberately not a ``NativeVNNIObservation`` corpus. During
development it may replace biased broad timing ratios and trigger a fresh
cross-validation fit. It may never alter the frozen policy after sealed
evaluation, and sealed observations may never generate paired requests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .schema import P95_REGRET_BUDGET


LEGACY_PAIRED_PROTOCOL_VERSION = "cuda-paired-interleaved-v1"
CUDA_PAIRED_PROTOCOL_VERSION = "cuda-paired-interleaved-v2"
PAIRED_PROTOCOL_VERSION = "native-vnni-paired-interleaved-v3"
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
DEFAULT_FAMILYWISE_ALPHA = 0.05
DEFAULT_MAX_REGRET = P95_REGRET_BUDGET
DEFAULT_MAX_UNCERTAINTY = 0.005
BOOTSTRAP_REPLICATE_BATCH = 4_096

COMMON_REQUIRED_COLUMNS = frozenset({
    "protocol_version",
    "backend",
    "phase",
    "source_format",
    "source_codebook",
    "execution_codebook",
    "shape",
    "execution_mode",
    "m",
    "n",
    "k",
    "pair_index",
    "configured_pair_count",
    "within_pair_order",
    "cell_order_seed",
    "pair_order_seed",
    "candidate_role",
    "candidate_id",
    "timed_replays",
    "latency_us",
    "latency_us_hex",
    "warmup_count",
    "bit_mismatches",
    "first_bit_mismatch",
    "repeat_byte_mismatches",
    "max_abs",
    "relative_l2",
    "cosine",
    "symmetric_kld",
    "grouped_output_digest",
    "serial_output_digest",
    "graph_capture_ok",
    "workspace_ok",
    "explicit_stream_ok",
    "route_counter_ok",
    "observed_candidate_id",
    "observed_path",
    "observed_tile_n",
    "observed_cpt",
    "observed_effective_kb",
    "serial_m1_candidate_id",
    "numerical_correctness",
    "correctness_pass",
})
REQUIRED_COLUMNS_V1 = COMMON_REQUIRED_COLUMNS
REQUIRED_COLUMNS_V2 = COMMON_REQUIRED_COLUMNS | {"request_id"}
REQUIRED_COLUMNS_V3 = REQUIRED_COLUMNS_V2 | {"architecture_class"}
# Latest-schema compatibility name used by strict producer tests.
REQUIRED_COLUMNS = REQUIRED_COLUMNS_V3


def _parse_bool(name: str, value: str) -> bool:
    """Parse an explicit producer boolean without accepting missing values."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _percentile(values: Iterable[float], quantile: float) -> float:
    """Return a conservative nearest-rank upper percentile."""

    ordered = sorted(values)
    if not ordered:
        return math.inf
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[min(len(ordered), rank) - 1]


@dataclass(frozen=True, order=True)
class PairedCellKey:
    """Every retained discriminator for one confirmation timing surface."""

    backend: str
    source_format: str
    source_codebook: int
    execution_codebook: int
    shape: str
    execution_mode: str
    m: int
    n: int
    k: int
    architecture_class: str = ""


@dataclass(frozen=True)
class PairedCellEvidence:
    """Validated selected/reference latency pairs for one runtime cell."""

    key: PairedCellKey
    selected_candidate_id: str
    exact_candidate_id: str
    selected_latency_us: tuple[float, ...]
    exact_latency_us: tuple[float, ...]
    selected_first_count: int
    exact_first_count: int
    request_id: str = ""

    @property
    def pair_count(self) -> int:
        """Return the complete number of retained interleaved pairs."""

        return len(self.selected_latency_us)

    @property
    def log_latency_ratios(self) -> tuple[float, ...]:
        """Return paired selected/reference log ratios in pair order."""

        return tuple(
            math.log(selected / exact)
            for selected, exact in zip(
                self.selected_latency_us,
                self.exact_latency_us,
                strict=True,
            )
        )

    @property
    def observed_regret(self) -> float:
        """Return the robust median paired latency ratio minus one."""

        return math.exp(statistics.median(self.log_latency_ratios)) - 1.0


@dataclass(frozen=True)
class PairedTimingComparison:
    """One development-only effective-candidate latency-ratio correction."""

    key: PairedCellKey
    selected_effective_candidate_id: str
    exact_effective_candidate_id: str
    selected_to_exact_median_ratio: float
    pair_count: int


def paired_timing_comparisons(
    cells: Iterable[PairedCellEvidence],
) -> dict[PairedCellKey, tuple[PairedTimingComparison, ...]]:
    """Index validated paired ratios for development cost construction.

    The ratio is dimensionless. The learner anchors it to the broad reference
    latency from the same source-format/mode surface, avoiding invalid absolute
    comparisons between confirmation sessions collected at different clocks.
    """

    grouped: dict[PairedCellKey, list[PairedTimingComparison]] = defaultdict(list)
    for cell in cells:
        comparison = PairedTimingComparison(
            key=cell.key,
            selected_effective_candidate_id=cell.selected_candidate_id,
            exact_effective_candidate_id=cell.exact_candidate_id,
            selected_to_exact_median_ratio=math.exp(
                statistics.median(cell.log_latency_ratios)
            ),
            pair_count=cell.pair_count,
        )
        if comparison.selected_to_exact_median_ratio <= 0.0 or not math.isfinite(
            comparison.selected_to_exact_median_ratio
        ):
            raise ValueError(f"{cell.key}: paired latency ratio is invalid")
        if comparison not in grouped[cell.key]:
            grouped[cell.key].append(comparison)

    result = {}
    for key, comparisons in grouped.items():
        result[key] = tuple(sorted(
            comparisons,
            key=lambda item: (
                item.selected_effective_candidate_id,
                item.exact_effective_candidate_id,
            ),
        ))
    return result


@dataclass(frozen=True)
class PairedCellResult:
    """One cell's observed regret retained in the final certificate."""

    key: PairedCellKey
    selected_candidate_id: str
    exact_candidate_id: str
    pair_count: int
    selected_first_count: int
    exact_first_count: int
    observed_regret: float
    request_id: str = ""


@dataclass(frozen=True)
class PairedConfirmationReport:
    """Simultaneous one-sided maximum-regret confidence certificate."""

    protocol_version: str
    bootstrap_method: str
    bootstrap_replicates: int
    familywise_alpha: float
    cells: tuple[PairedCellResult, ...]
    observed_max_regret: float
    simultaneous_95pct_upper_regret: float
    upper_bound_uncertainty: float

    def require_promotable(
        self,
        *,
        max_regret: float = DEFAULT_MAX_REGRET,
        max_uncertainty: float = DEFAULT_MAX_UNCERTAINTY,
    ) -> None:
        """Fail closed unless both the regret and precision gates pass."""

        failures = []
        if not self.cells:
            failures.append("no paired confirmation cells")
        if self.simultaneous_95pct_upper_regret >= max_regret:
            failures.append(
                "simultaneous max-regret UCB "
                f"{self.simultaneous_95pct_upper_regret:.4%} >= "
                f"{max_regret:.2%}"
            )
        if self.upper_bound_uncertainty > max_uncertainty:
            failures.append(
                "max-regret UCB uncertainty "
                f"{self.upper_bound_uncertainty:.4%} > "
                f"{max_uncertainty:.2%}"
            )
        if failures:
            raise ValueError(
                "paired confirmation is not promotable: " + "; ".join(failures)
            )


@dataclass(frozen=True)
class _RawPairRow:
    """One parsed candidate half of an interleaved timing pair."""

    key: PairedCellKey
    request_id: str
    pair_index: int
    configured_pair_count: int
    within_pair_order: int
    cell_order_seed: int
    pair_order_seed: int
    candidate_role: str
    candidate_id: str
    latency_us: float


def _parse_row(
    path: Path,
    row_number: int,
    raw: Mapping[str, str],
    legacy_request_id: str,
) -> _RawPairRow:
    """Validate one backend-qualified producer row before aggregation."""

    prefix = f"{path}:{row_number}"
    protocol_version = raw["protocol_version"].strip()
    if protocol_version not in {
        LEGACY_PAIRED_PROTOCOL_VERSION,
        CUDA_PAIRED_PROTOCOL_VERSION,
        PAIRED_PROTOCOL_VERSION,
    }:
        raise ValueError(f"{prefix}: unsupported paired protocol version")
    backend = raw["backend"].strip().lower()
    if backend not in {"cpu", "cuda", "rocm"}:
        raise ValueError(f"{prefix}: unknown paired backend")
    expected_phase = "decode_m1" if backend == "cpu" else "decode"
    if raw["phase"].strip() != expected_phase:
        raise ValueError(f"{prefix}: wrong paired confirmation surface")
    execution_mode = raw["execution_mode"].strip().lower()
    if execution_mode not in {"eager", "graph_captured"}:
        raise ValueError(f"{prefix}: unknown paired execution mode")
    if backend == "cpu" and execution_mode != "eager":
        raise ValueError(f"{prefix}: CPU paired timing must use eager execution")

    role = raw["candidate_role"].strip().lower()
    if role not in {"selected", "exact"}:
        raise ValueError(f"{prefix}: candidate_role must be selected or exact")
    candidate_id = raw["candidate_id"].strip().lower()
    if not candidate_id:
        raise ValueError(f"{prefix}: candidate_id must not be empty")
    if raw["observed_candidate_id"].strip().lower() != candidate_id:
        raise ValueError(f"{prefix}: forced paired route did not execute")
    if protocol_version in {
        CUDA_PAIRED_PROTOCOL_VERSION,
        PAIRED_PROTOCOL_VERSION,
    }:
        request_id = raw["request_id"].strip()
        if not request_id:
            raise ValueError(f"{prefix}: paired row has no request_id")
    else:
        # Legacy files predate multi-edge transactions. Binding their synthetic
        # identity to the input path preserves support for the retained tuning
        # corpus while ensuring two independent files cannot collapse into one
        # ambiguous pair-index namespace.
        request_id = legacy_request_id

    pair_index = int(raw["pair_index"])
    pair_count = int(raw["configured_pair_count"])
    within_pair_order = int(raw["within_pair_order"])
    cell_order_seed = int(raw["cell_order_seed"])
    pair_order_seed = int(raw["pair_order_seed"])
    timed_replays = int(raw["timed_replays"])
    warmups = int(raw["warmup_count"])
    if pair_index < 0 or pair_count < 1 or pair_index >= pair_count:
        raise ValueError(f"{prefix}: invalid pair index/count")
    if within_pair_order not in {0, 1}:
        raise ValueError(f"{prefix}: within_pair_order must be zero or one")
    if cell_order_seed <= 0 or pair_order_seed <= 0:
        raise ValueError(f"{prefix}: paired order seeds must be positive")
    if timed_replays <= 0 or warmups < 5:
        raise ValueError(f"{prefix}: paired timing lacks promotion-strength setup")

    latency = float.fromhex(raw["latency_us_hex"].strip())
    readable_latency = float(raw["latency_us"])
    if latency <= 0.0 or not math.isfinite(latency):
        raise ValueError(f"{prefix}: paired latency must be positive and finite")
    if not math.isclose(
        latency,
        readable_latency,
        rel_tol=0.0,
        abs_tol=5.1e-7,
    ):
        raise ValueError(f"{prefix}: readable and exact paired latency disagree")

    required_true = (
        "workspace_ok",
        "explicit_stream_ok",
        "route_counter_ok",
        "numerical_correctness",
        "correctness_pass",
    )
    if any(not _parse_bool(name, raw[name]) for name in required_true):
        raise ValueError(f"{prefix}: paired correctness/route proof failed")
    if execution_mode == "graph_captured" and not _parse_bool(
        "graph_capture_ok", raw["graph_capture_ok"]
    ):
        raise ValueError(f"{prefix}: graph paired timing did not capture")
    if int(raw["repeat_byte_mismatches"]) != 0:
        raise ValueError(f"{prefix}: paired candidate is not repeatable")
    if not raw["grouped_output_digest"].strip() or not raw[
        "serial_output_digest"
    ].strip():
        raise ValueError(f"{prefix}: paired output digests are missing")

    key = PairedCellKey(
        backend=backend,
        source_format=raw["source_format"].strip().upper(),
        source_codebook=int(raw["source_codebook"]),
        execution_codebook=int(raw["execution_codebook"]),
        shape=raw["shape"].strip(),
        execution_mode=execution_mode,
        m=int(raw["m"]),
        n=int(raw["n"]),
        k=int(raw["k"]),
        architecture_class=(
            raw["architecture_class"].strip()
            if protocol_version == PAIRED_PROTOCOL_VERSION
            else ""
        ),
    )
    if (
        not key.source_format
        or not key.shape
        or key.m < 1
        or key.n <= 0
        or key.k <= 0
        or (
            protocol_version == PAIRED_PROTOCOL_VERSION
            and not key.architecture_class
        )
    ):
        raise ValueError(f"{prefix}: invalid paired runtime identity")
    return _RawPairRow(
        key=key,
        request_id=request_id,
        pair_index=pair_index,
        configured_pair_count=pair_count,
        within_pair_order=within_pair_order,
        cell_order_seed=cell_order_seed,
        pair_order_seed=pair_order_seed,
        candidate_role=role,
        candidate_id=candidate_id,
        latency_us=latency,
    )


def read_paired_confirmation_csv(
    paths: Iterable[Path],
    *,
    minimum_pairs: int = 30,
) -> tuple[PairedCellEvidence, ...]:
    """Read strict paired shards and return complete cells in stable order."""

    if minimum_pairs < 1:
        raise ValueError("minimum_pairs must be positive")
    rows_by_cell: dict[
        tuple[PairedCellKey, str], list[_RawPairRow]
    ] = defaultdict(list)
    for path in (Path(item) for item in paths):
        legacy_request_id = (
            "legacy-v1:" + hashlib.sha256(path.read_bytes()).hexdigest()[:24]
        )
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = frozenset(reader.fieldnames or ())
            if fields == REQUIRED_COLUMNS_V1:
                expected = REQUIRED_COLUMNS_V1
            elif fields == REQUIRED_COLUMNS_V2:
                expected = REQUIRED_COLUMNS_V2
            elif fields == REQUIRED_COLUMNS_V3:
                expected = REQUIRED_COLUMNS_V3
            else:
                expected = REQUIRED_COLUMNS_V3
            missing = expected - fields
            unexpected = fields - expected
            if missing or unexpected:
                raise ValueError(
                    f"{path}: paired header mismatch: missing={sorted(missing)} "
                    f"unexpected={sorted(unexpected)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                parsed = _parse_row(
                    path,
                    row_number,
                    raw,
                    legacy_request_id,
                )
                if fields == REQUIRED_COLUMNS_V1 and raw[
                    "protocol_version"
                ].strip() != LEGACY_PAIRED_PROTOCOL_VERSION:
                    raise ValueError(
                        f"{path}:{row_number}: v2 protocol requires request_id"
                    )
                if fields == REQUIRED_COLUMNS_V2 and raw[
                    "protocol_version"
                ].strip() != CUDA_PAIRED_PROTOCOL_VERSION:
                    raise ValueError(
                        f"{path}:{row_number}: v2 header requires CUDA v2 protocol"
                    )
                if fields == REQUIRED_COLUMNS_V3 and raw[
                    "protocol_version"
                ].strip() != PAIRED_PROTOCOL_VERSION:
                    raise ValueError(
                        f"{path}:{row_number}: v3 header requires generic v3 protocol"
                    )
                rows_by_cell[(parsed.key, parsed.request_id)].append(parsed)

    cells = []
    for (key, request_id), rows in sorted(rows_by_cell.items()):
        configured_counts = {row.configured_pair_count for row in rows}
        cell_seeds = {row.cell_order_seed for row in rows}
        if len(configured_counts) != 1 or len(cell_seeds) != 1:
            raise ValueError(
                f"{key}/{request_id}: paired count or cell seed changed"
            )
        pair_count = next(iter(configured_counts))
        if pair_count < minimum_pairs:
            raise ValueError(
                f"{key}/{request_id}: only {pair_count} pairs; "
                f"require at least {minimum_pairs}"
            )

        by_pair: dict[int, list[_RawPairRow]] = defaultdict(list)
        for row in rows:
            by_pair[row.pair_index].append(row)
        if sorted(by_pair) != list(range(pair_count)):
            raise ValueError(
                f"{key}/{request_id}: paired indices are not contiguous"
            )

        role_candidates: dict[str, set[str]] = defaultdict(set)
        selected = []
        exact = []
        first_counts = {"selected": 0, "exact": 0}
        for pair_index in range(pair_count):
            pair = by_pair[pair_index]
            if len(pair) != 2:
                raise ValueError(f"{key}: pair {pair_index} does not have two rows")
            if {row.within_pair_order for row in pair} != {0, 1}:
                raise ValueError(f"{key}: pair {pair_index} order is incomplete")
            if len({row.pair_order_seed for row in pair}) != 1:
                raise ValueError(f"{key}: pair {pair_index} seed changed")
            by_role = {row.candidate_role: row for row in pair}
            if set(by_role) != {"selected", "exact"}:
                raise ValueError(f"{key}: pair {pair_index} roles are incomplete")
            for role, row in by_role.items():
                role_candidates[role].add(row.candidate_id)
            selected.append(by_role["selected"].latency_us)
            exact.append(by_role["exact"].latency_us)
            first = min(pair, key=lambda row: row.within_pair_order)
            first_counts[first.candidate_role] += 1

        if any(len(candidates) != 1 for candidates in role_candidates.values()):
            raise ValueError(f"{key}: candidate identity changed across pairs")
        selected_id = next(iter(role_candidates["selected"]))
        exact_id = next(iter(role_candidates["exact"]))
        if selected_id == exact_id:
            raise ValueError(f"{key}: selected and exact candidates are identical")
        # A deterministic RNG is allowed to be slightly imbalanced, but a
        # protocol that almost always runs one role first is not interleaved.
        minimum_first = max(1, pair_count // 4)
        if min(first_counts.values()) < minimum_first:
            raise ValueError(f"{key}: paired first-candidate order is imbalanced")

        cells.append(PairedCellEvidence(
            key=key,
            selected_candidate_id=selected_id,
            exact_candidate_id=exact_id,
            selected_latency_us=tuple(selected),
            exact_latency_us=tuple(exact),
            selected_first_count=first_counts["selected"],
            exact_first_count=first_counts["exact"],
            request_id=request_id,
        ))
    if not cells:
        raise ValueError("paired confirmation corpus contains no cells")
    return tuple(cells)


def _bootstrap_seed(cells: Iterable[PairedCellEvidence], seed: str) -> int:
    """Bind deterministic bootstrap resampling to the exact cell inventory."""

    digest = hashlib.sha256(seed.encode())
    for cell in cells:
        digest.update(repr(cell.key).encode())
        digest.update(b"\0")
        digest.update(cell.selected_candidate_id.encode())
        digest.update(b"\0")
        digest.update(cell.exact_candidate_id.encode())
        digest.update(b"\0")
        digest.update(cell.request_id.encode())
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "little")


def _physical_core_count() -> int:
    """Return the physical cores available to this process's affinity mask.

    Linux exposes a stable ``(package_id, core_id)`` identity for each logical
    CPU. Counting those identities avoids scheduling two bootstrap workers on
    sibling hyperthreads. The conservative fallback uses the logical CPU count
    only when sysfs is unavailable, which keeps this offline tool portable.
    """

    logical_cpus = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else list(range(os.cpu_count() or 1))
    )
    physical_cores: set[tuple[str, str]] = set()
    for cpu in logical_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package_id = (topology / "physical_package_id").read_text().strip()
            core_id = (topology / "core_id").read_text().strip()
        except OSError:
            return max(1, len(logical_cpus))
        physical_cores.add((package_id, core_id))
    return max(1, len(physical_cores))


def _cell_bootstrap_seed(cell: PairedCellEvidence, seed: str) -> int:
    """Derive a worker-order-independent random stream for one paired edge."""

    digest = hashlib.sha256(seed.encode())
    digest.update(repr(cell.key).encode())
    digest.update(b"\0")
    digest.update(cell.selected_candidate_id.encode())
    digest.update(b"\0")
    digest.update(cell.exact_candidate_id.encode())
    digest.update(b"\0")
    digest.update(cell.request_id.encode())
    return int.from_bytes(digest.digest()[:16], "little")


def _bootstrap_partition_maximum(
    arguments: tuple[tuple[PairedCellEvidence, ...], int, str],
) -> np.ndarray:
    """Compute per-replicate maxima for one disjoint partition of cells.

    Each cell owns an independent deterministic PCG64 stream. Consequently,
    changing the worker count or partition order cannot change the generated
    resamples. Replicates are generated in bounded batches so every worker uses
    a small persistent high-water allocation even for a large evidence set.
    """

    cells, bootstrap_replicates, seed = arguments
    partition_maxima = np.full(bootstrap_replicates, -np.inf, dtype=np.float64)
    for cell in cells:
        values = np.asarray(cell.log_latency_ratios, dtype=np.float64)
        rng = np.random.Generator(np.random.PCG64(_cell_bootstrap_seed(cell, seed)))
        for begin in range(0, bootstrap_replicates, BOOTSTRAP_REPLICATE_BATCH):
            end = min(begin + BOOTSTRAP_REPLICATE_BATCH, bootstrap_replicates)
            indices = rng.integers(
                0,
                values.size,
                size=(end - begin, values.size),
                dtype=np.int32,
            )
            sampled_medians = np.median(values[indices], axis=1)
            regrets = np.expm1(sampled_medians)
            np.maximum(
                partition_maxima[begin:end],
                regrets,
                out=partition_maxima[begin:end],
            )
    return partition_maxima


def certify_paired_confirmation(
    cells: Iterable[PairedCellEvidence],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    familywise_alpha: float = DEFAULT_FAMILYWISE_ALPHA,
    seed: str = "native-vnni-paired-maximum-bootstrap-v1",
    workers: int | None = None,
) -> PairedConfirmationReport:
    """Bootstrap one simultaneous upper bound for maximum paired regret.

    Every replicate independently resamples complete selected/reference ratios
    within each cell and then takes the maximum cell median. The final quantile
    is therefore a confidence bound on the familywise maximum itself, not a
    collection of invalid independent per-cell intervals. Cell-specific random
    streams make the result invariant to physical-core worker scheduling.
    """

    ordered = tuple(sorted(cells, key=lambda cell: cell.key))
    if not ordered:
        raise ValueError("paired certification requires at least one cell")
    if bootstrap_replicates < 1_000:
        raise ValueError("paired certification requires at least 1000 bootstraps")
    if not 0.0 < familywise_alpha < 1.0:
        raise ValueError("familywise_alpha must be strictly between zero and one")
    if workers is not None and workers < 1:
        raise ValueError("paired certification workers must be positive")

    observed = tuple(cell.observed_regret for cell in ordered)
    observed_max = max(observed)
    # Preserve the inventory-bound seed contract, then give every cell its own
    # deterministic substream. This makes parallel reduction reproducible.
    inventory_seed = f"{seed}:{_bootstrap_seed(ordered, seed):016x}"
    requested_workers = workers if workers is not None else _physical_core_count()
    worker_count = min(requested_workers, len(ordered))
    partitions = tuple(
        tuple(ordered[index::worker_count])
        for index in range(worker_count)
    )
    arguments = tuple(
        (partition, bootstrap_replicates, inventory_seed)
        for partition in partitions
    )
    if worker_count == 1:
        partition_maxima = (_bootstrap_partition_maximum(arguments[0]),)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
        ) as executor:
            partition_maxima = tuple(executor.map(
                _bootstrap_partition_maximum,
                arguments,
            ))
    maxima = np.maximum.reduce(partition_maxima)
    upper = _percentile(maxima, 1.0 - familywise_alpha)
    uncertainty = max(0.0, upper - observed_max)
    results = tuple(
        PairedCellResult(
            key=cell.key,
            selected_candidate_id=cell.selected_candidate_id,
            exact_candidate_id=cell.exact_candidate_id,
            pair_count=cell.pair_count,
            selected_first_count=cell.selected_first_count,
            exact_first_count=cell.exact_first_count,
            observed_regret=cell.observed_regret,
            request_id=cell.request_id,
        )
        for cell in ordered
    )
    return PairedConfirmationReport(
        protocol_version=PAIRED_PROTOCOL_VERSION,
        bootstrap_method="paired-cell-median maximum bootstrap v2 vectorized",
        bootstrap_replicates=bootstrap_replicates,
        familywise_alpha=familywise_alpha,
        cells=results,
        observed_max_regret=observed_max,
        simultaneous_95pct_upper_regret=upper,
        upper_bound_uncertainty=uncertainty,
    )


def report_mapping(report: PairedConfirmationReport) -> dict[str, object]:
    """Return a stable JSON-compatible certificate mapping."""

    payload = asdict(report)
    payload["cells"] = [
        {
            **asdict(cell),
            "key": asdict(cell.key),
        }
        for cell in report.cells
    ]
    return payload


def main() -> int:
    """Validate paired CSVs, compute the certificate, and fail if requested."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=30)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "parallel bootstrap workers; defaults to the physical cores "
            "available in the process affinity mask"
        ),
    )
    parser.add_argument("--require-promotable", action="store_true")
    args = parser.parse_args()

    cells = read_paired_confirmation_csv(
        args.inputs,
        minimum_pairs=args.minimum_pairs,
    )
    report = certify_paired_confirmation(
        cells,
        bootstrap_replicates=args.bootstrap_replicates,
        workers=args.workers,
    )
    if args.require_promotable:
        report.require_promotable()
    payload = report_mapping(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps({
        "cell_count": len(report.cells),
        "observed_max_regret": report.observed_max_regret,
        "simultaneous_95pct_upper_regret": (
            report.simultaneous_95pct_upper_regret
        ),
        "upper_bound_uncertainty": report.upper_bound_uncertainty,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
