"""CPU NativeVNNI ordinary-prefill trainer adapter.

The ordinary M>1 GEMM launcher has a different policy surface from grouped
verifier rows. Full-K candidates vary N work sharing; long-K candidates inherit
the exact production serial-M1 K partition and vary grouped row sharing. This
adapter rejects mismatched arithmetic routes, requires byte equality with
serial M1 rows, and preserves arbitrary positive prefill depths for the shared
learned dispatcher.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import raw_corpus_id, verify_aggregate_timing
from ..candidate_registry import cpu_native_vnni_prefill_registry
from ..corpus import ObservationCorpus
from ..format_registry import format_spec
from ..profiles import (
    ADAPTIVE_TIMING_CEILING_POLICY,
    ADAPTIVE_TIMING_PROTOCOL,
    ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER,
    CPU_PREFILL_DYNAMIC_WATCHDOG_MULTIPLIER,
    CPU_PREFILL_DYNAMIC_WATCHDOG_PADDING_US,
    CANDIDATE_EXPANSION_TIMING_STOP_REASON,
    LEGACY_ADAPTIVE_TIMING_CEILING_POLICY,
    SUPPORTED_ADAPTIVE_TIMING_PROTOCOLS,
    MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US,
    MAX_PROMOTION_MEDIAN_RELATIVE_DRIFT,
    MIN_CANDIDATE_EXPANSION_TIMED_DURATION_US,
    MIN_CANDIDATE_EXPANSION_TRANSITION_WARMUP_DURATION_US,
    MIN_CANDIDATE_EXPANSION_WARMUP_DURATION_US,
    MIN_CANDIDATE_EXPANSION_WARMUPS,
    MIN_CANDIDATE_EXPANSION_ADAPTIVE_SAMPLES,
    MIN_CANDIDATE_EXPANSION_MAXIMUM_SAMPLES,
    MIN_PROMOTION_ADAPTIVE_SAMPLES,
    MIN_PROMOTION_MAXIMUM_SAMPLES,
    MIN_PROMOTION_SAMPLES,
    MIN_PROMOTION_TIMED_DURATION_US,
    MIN_PROMOTION_TRANSITION_WARMUP_BUDGET_CEILING_US,
    MIN_PROMOTION_TRANSITION_WARMUP_DURATION_US,
    MIN_PROMOTION_TRANSITION_WARMUP_LATENCY_MULTIPLIER,
    MIN_PROMOTION_WARMUP_DURATION_US,
    MIN_PROMOTION_WARMUPS,
    MeasurementProfile,
)
from ..cpu_prefill_route_manifest import (
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
)
from ..schema import (
    LEARNER_VERSION,
    POLICY_ABI,
    SCHEMA_VERSION,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    SemanticContract,
    classify_aspect,
)


CPU_PREFILL_TRIAL_SET_VERSION = "cpu-prefill-full-k-q8-rows-v16"
CPU_PREFILL_LEGACY_TRIAL_SET_VERSION = "cpu-prefill-full-k-q8-rows-v13"
CPU_SERIAL_POLICY_ID = "cpu.nvnni.production.serial-m1-v2"
CPU_PREFILL_SERIAL_ORACLE_POLICY = "boundary-quartile-sentinel-v1"
CPU_PREFILL_PRECONDITIONING_POLICY = "source-complete-round-v1"
CPU_PREFILL_COORDINATION_POLICY = "mpi-complete-round-v1"
REQUIRED_RAW_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "build_isa", "runtime_isa_requested",
    "runtime_isa_effective", "threads", "measurement_coordination",
    "mpi_world_size", "mpi_rank", "weight_bytes", "warmup_count",
    "serial_oracle_policy", "serial_oracle_rows",
    "preconditioning_policy", "preconditioning_m",
    "preconditioning_budget_us", "preconditioning_duration_us",
    "warmup_round_count", "warmup_budget_policy", "warmup_probe_latency_us",
    "warmup_budget_floor_us", "warmup_latency_multiplier",
    "warmup_budget_ceiling_us", "warmup_budget_us", "warmup_duration_us",
    "global_warmup_round_count", "warmup_wall_duration_us",
    "warmup_max_round_duration_us", "warmup_round_timeout_us",
    "timing_order_seed",
    "global_timing_round_count",
    "sample_count", "timing_protocol", "timing_ceiling_policy",
    "min_sample_count",
    "stable_sample_count", "max_sample_count",
    "timing_budget_us", "timed_duration_us", "median_stability_limit",
    "median_relative_drift", "timing_converged", "timing_stop_reason",
    "min_us", "median_us", "p95_us", "mad_us", "cv",
    "serial_median_us", "speedup", "bit_mismatches", "first_bit_mismatch",
    "repeat_byte_mismatches", "max_abs", "relative_l2", "cosine",
    "symmetric_kld", "grouped_output_digest", "serial_output_digest",
    "timing_sample_digest", "route_counter_ok", "observed_candidate_id",
    "k_tiles", "k_tile_blocks", "n_block_chunks", "numerical_correctness",
    "correctness_pass", "is_winner",
})
STATIONARY_RAW_COLUMNS = frozenset({
    "stationary_sample_begin",
    "stationary_sample_count",
    "stationary_duration_us",
})

REQUIRED_TIMING_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "build_isa", "runtime_isa_requested",
    "runtime_isa_effective", "sample_index", "latency_us", "latency_us_hex",
})

RawTimingKey = tuple[
    str, int, int, str, str, int, int, int, str, str, str, str
]
MeasurementCellKey = tuple[
    str, int, int, str, str, int, int, int, str, str, str, int
]


def _required(name: str, value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _cpu_isa(name: str, value: str, *, allow_auto: bool = False) -> str:
    normalized = _required(name, value).upper()
    allowed = {"AVX2", "AVX512"}
    if allow_auto:
        allowed.add("AUTO")
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}")
    return normalized


def _sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _timing_key(raw: Mapping[str, str]) -> RawTimingKey:
    return (
        raw["source_format"].strip().upper(),
        int(raw["source_codebook"]),
        int(raw["execution_codebook"]),
        raw["shape"].strip(),
        raw["execution_mode"].strip().lower(),
        int(raw["m"]),
        int(raw["n"]),
        int(raw["k"]),
        raw["candidate_id"].strip().lower(),
        raw["build_isa"].strip().upper(),
        raw["runtime_isa_requested"].strip().upper(),
        raw["runtime_isa_effective"].strip().upper(),
    )


def _measurement_cell_key(raw: Mapping[str, str]) -> MeasurementCellKey:
    """Identify candidates that share one complete-round measurement cell."""

    timing_key = _timing_key(raw)
    return (
        *timing_key[:8],
        *timing_key[9:],
        int(raw["threads"]),
    )


def read_cpu_prefill_timing_sidecars(
    paths: Iterable[Path],
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact acquisition-order steady-clock samples for prefill rows."""

    indexed: dict[RawTimingKey, list[float]] = {}
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TIMING_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU prefill timing columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["backend"].strip().lower() != "cpu" or (
                    raw["phase"].strip() != "prefill_gemm"
                ):
                    raise ValueError(f"{path}:{row_number}: wrong timing surface")
                key = _timing_key(raw)
                samples = indexed.setdefault(key, [])
                sample_index = int(raw["sample_index"])
                if sample_index != len(samples):
                    raise ValueError(
                        f"{path}:{row_number}: non-contiguous timing index"
                    )
                latency_us = float.fromhex(raw["latency_us_hex"].strip())
                readable_us = float(raw["latency_us"])
                if latency_us <= 0.0 or not math.isfinite(latency_us):
                    raise ValueError(f"{path}:{row_number}: invalid latency")
                if not math.isclose(
                    readable_us, latency_us, rel_tol=0.0, abs_tol=5.1e-10
                ):
                    raise ValueError(
                        f"{path}:{row_number}: readable/exact latency mismatch"
                    )
                samples.append(latency_us)
    return {key: tuple(samples) for key, samples in indexed.items()}


def _upper_median(values: Iterable[float]) -> float:
    """Reproduce the C++ upper-median convention for one non-empty window."""

    ordered = sorted(values)
    if not ordered:
        raise ValueError("median window must not be empty")
    return ordered[len(ordered) // 2]


def _median_relative_drift(samples: tuple[float, ...]) -> float:
    """Compare acquisition-order half-window medians for clock/thermal drift."""

    if len(samples) < 2:
        return math.inf
    midpoint = len(samples) // 2
    first = _upper_median(samples[:midpoint])
    second = _upper_median(samples[midpoint:])
    denominator = max(first, second)
    return abs(first - second) / denominator if denominator > 0.0 else math.inf


def _stationary_window(
    samples: tuple[float, ...],
    minimum_samples: int,
    minimum_duration_us: float,
) -> tuple[int, tuple[float, ...], float]:
    """Reproduce the C++ shortest trailing evidence-window selection."""

    begin = len(samples)
    duration_us = 0.0
    while begin > 0 and (
        len(samples) - begin < minimum_samples
        or duration_us < minimum_duration_us
    ):
        begin -= 1
        duration_us += samples[begin]
    return begin, samples[begin:], duration_us


def _serial_oracle_row_count(m: int) -> int:
    """Reproduce the C++ boundary/quartile sentinel row inventory."""

    selected = {row for row in range(min(m, 4))}
    for anchor in (m // 4, m // 2, (3 * m) // 4):
        selected.update(row for row in (anchor - 1, anchor) if 0 <= row < m)
    selected.update(range(max(0, m - 6), m))
    return len(selected)


@dataclass(frozen=True)
class CPUPrefillAdapterContext:
    """Immutable run and CPU architecture provenance."""

    profile: MeasurementProfile
    run_id: str
    corpus_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    architecture_class: str
    device_name: str
    driver_runtime: str
    serial_m1_policy_hash: str
    raw_timing_sidecar_retained: bool
    anchored_candidate_expansion: bool = False

    @classmethod
    def workflow_smoke(cls, *, corpus_id: str) -> "CPUPrefillAdapterContext":
        return cls(
            profile=MeasurementProfile.QUICK,
            run_id="workflow-smoke",
            corpus_id=corpus_id,
            git_revision="workflow-uncommitted",
            build_id="workflow-release-build",
            compiler_id="workflow-cpu-compiler",
            architecture_class="workflow-cpu-architecture",
            device_name="workflow-cpu-device",
            driver_runtime="workflow-cpu-runtime",
            serial_m1_policy_hash="sha256:" + "0" * 64,
            raw_timing_sidecar_retained=False,
            anchored_candidate_expansion=False,
        )

    def validate(self) -> None:
        for name in (
            "run_id", "corpus_id", "git_revision", "build_id", "compiler_id",
            "architecture_class", "device_name", "driver_runtime",
            "serial_m1_policy_hash",
        ):
            _required(name, getattr(self, name))
        if not self.corpus_id.startswith("sha256:"):
            raise ValueError("corpus_id must be a sha256 digest")
        if not self.serial_m1_policy_hash.startswith("sha256:"):
            raise ValueError("serial_m1_policy_hash must be a sha256 digest")
        if self.profile.installable and not self.raw_timing_sidecar_retained:
            raise ValueError("installable CPU prefill evidence needs raw timings")


def _trial_set_hash(
    source_format: str,
    shape: str,
    m: int,
    n: int,
    k: int,
    build_isa: str,
    runtime_isa: str,
    threads: int,
    coordination_policy: str,
    mpi_world_size: int,
    timing_protocol: str,
) -> str:
    return _sha256({
        "version": (
            CPU_PREFILL_TRIAL_SET_VERSION
            if timing_protocol == ADAPTIVE_TIMING_PROTOCOL
            else CPU_PREFILL_LEGACY_TRIAL_SET_VERSION
        ),
        "source_format": source_format,
        "shape": shape,
        "m": m,
        "n": n,
        "k": k,
        "build_isa": build_isa,
        "runtime_isa": runtime_isa,
        "threads": threads,
        "coordination_policy": coordination_policy,
        "mpi_world_size": mpi_world_size,
    })


def adapt_cpu_prefill_row(
    raw: Mapping[str, str],
    context: CPUPrefillAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
) -> NativeVNNIObservation:
    """Convert one strong decode-equivalent prefill candidate into evidence."""

    if raw.get("backend", "").strip().lower() != "cpu" or (
        raw.get("phase", "").strip() != "prefill_gemm"
    ):
        raise ValueError("CPU prefill adapter received the wrong surface")
    if raw["execution_mode"].strip().lower() != "eager":
        raise ValueError("CPU prefill observations must be eager")
    m, n, k = (int(raw[name]) for name in ("m", "n", "k"))
    if m < 2 or n <= 0 or k <= 0:
        raise ValueError("CPU prefill dimensions must be positive with M >= 2")
    serial_oracle_policy = _required(
        "serial_oracle_policy", raw["serial_oracle_policy"]
    )
    serial_oracle_rows = int(raw["serial_oracle_rows"])
    if serial_oracle_policy != CPU_PREFILL_SERIAL_ORACLE_POLICY:
        raise ValueError(
            f"unsupported CPU prefill serial oracle {serial_oracle_policy}"
        )
    if serial_oracle_rows != _serial_oracle_row_count(m):
        raise ValueError("CPU prefill serial oracle row count disagrees")
    preconditioning_policy = _required(
        "preconditioning_policy", raw["preconditioning_policy"]
    )
    preconditioning_m = int(raw["preconditioning_m"])
    preconditioning_budget_us = float(raw["preconditioning_budget_us"])
    preconditioning_duration_us = float(raw["preconditioning_duration_us"])
    if preconditioning_policy not in {
        CPU_PREFILL_PRECONDITIONING_POLICY,
        "disabled",
    }:
        raise ValueError("unsupported CPU prefill preconditioning policy")
    if preconditioning_policy == "disabled" and (
        preconditioning_m != 0
        or preconditioning_budget_us != 0.0
        or preconditioning_duration_us != 0.0
    ):
        raise ValueError("disabled CPU prefill preconditioning has evidence")
    if preconditioning_policy == CPU_PREFILL_PRECONDITIONING_POLICY and (
        preconditioning_m < 2
        or preconditioning_budget_us <= 0.0
        or preconditioning_duration_us < preconditioning_budget_us
    ):
        raise ValueError("CPU prefill source preconditioning is incomplete")

    registry = cpu_native_vnni_prefill_registry()
    candidate = registry.resolve(raw["candidate_id"])
    source_format = raw["source_format"].strip().upper()
    spec = format_spec(source_format)
    runtime_codebook = spec.runtime_codebook("cpu")
    if int(raw["source_codebook"]) != spec.source_codebook_id or (
        int(raw["execution_codebook"]) != runtime_codebook
    ):
        raise ValueError(f"{source_format}: codebook disagrees with registry")

    route_counter_ok = _parse_bool("route_counter_ok", raw["route_counter_ok"])
    observed_raw = _required(
        "observed_candidate_id", raw["observed_candidate_id"]
    )
    try:
        observed_id = registry.resolve(observed_raw).effective_candidate_id
    except ValueError:
        observed_id = f"unresolved:{observed_raw}"
    forced_route_ok = route_counter_ok and (
        observed_id == candidate.effective_candidate_id
    )
    k_tiles = int(raw["k_tiles"])
    k_tile_blocks = int(raw["k_tile_blocks"])
    blocks_per_row = (k + 31) // 32
    k_tile_policy = candidate.config_json["k_tile_policy"]
    if k_tile_policy == "full_k":
        arithmetic_route_ok = (
            k_tiles == 1 and k_tile_blocks == blocks_per_row
        )
    elif k_tile_policy == "inherit_serial_m1":
        arithmetic_route_ok = (
            k_tiles > 1
            and k_tile_blocks == (blocks_per_row + k_tiles - 1) // k_tiles
        )
    else:
        raise ValueError(f"unknown CPU prefill K policy {k_tile_policy}")
    if not arithmetic_route_ok:
        forced_route_ok = False

    mismatch_count = int(raw["bit_mismatches"])
    repeat_mismatches = int(raw["repeat_byte_mismatches"])
    grouped_digest = _required(
        "grouped_output_digest", raw["grouped_output_digest"]
    )
    serial_digest = _required("serial_output_digest", raw["serial_output_digest"])
    if (mismatch_count == 0) != (grouped_digest == serial_digest):
        raise ValueError("CPU prefill mismatch count disagrees with digests")
    diagnostics = tuple(float(raw[name]) for name in (
        "max_abs", "relative_l2", "cosine", "symmetric_kld"
    ))
    if not all(math.isfinite(value) for value in diagnostics):
        raise ValueError("CPU prefill diagnostics must be finite")
    numerical_correctness = _parse_bool(
        "numerical_correctness", raw["numerical_correctness"]
    )
    expected_pass = (
        forced_route_ok and mismatch_count == 0 and repeat_mismatches == 0
        and numerical_correctness
    )
    if _parse_bool("correctness_pass", raw["correctness_pass"]) != expected_pass:
        raise ValueError("CPU prefill correctness_pass disagrees with evidence")

    warmups, samples = (int(raw[name]) for name in (
        "warmup_count", "sample_count"
    ))
    warmup_rounds = int(raw["warmup_round_count"])
    global_warmup_rounds = int(raw["global_warmup_round_count"])
    warmup_budget_policy = _required(
        "warmup_budget_policy", raw["warmup_budget_policy"]
    )
    warmup_probe_latency_us = float(raw["warmup_probe_latency_us"])
    warmup_budget_floor_us = float(raw["warmup_budget_floor_us"])
    warmup_latency_multiplier = float(raw["warmup_latency_multiplier"])
    warmup_budget_ceiling_us = float(raw["warmup_budget_ceiling_us"])
    warmup_budget_us = float(raw["warmup_budget_us"])
    warmup_duration_us = float(raw["warmup_duration_us"])
    warmup_wall_duration_us = float(raw["warmup_wall_duration_us"])
    warmup_max_round_duration_us = float(
        raw["warmup_max_round_duration_us"]
    )
    complete_round_probe_duration_us = float(
        raw.get("complete_round_probe_duration_us", "0") or "0"
    )
    warmup_round_timeout_us = float(raw["warmup_round_timeout_us"])
    timing_order_seed = int(raw["timing_order_seed"])
    global_timing_rounds = int(raw["global_timing_round_count"])
    timing_protocol = _required("timing_protocol", raw["timing_protocol"])
    timing_ceiling_policy = _required(
        "timing_ceiling_policy", raw["timing_ceiling_policy"]
    )
    minimum_samples = int(raw["min_sample_count"])
    stable_samples = int(raw["stable_sample_count"])
    maximum_samples = int(raw["max_sample_count"])
    timing_budget_us = float(raw["timing_budget_us"])
    timed_duration_us = float(raw["timed_duration_us"])
    stability_limit = float(raw["median_stability_limit"])
    reported_drift = float(raw["median_relative_drift"])
    timing_converged = _parse_bool(
        "timing_converged", raw["timing_converged"]
    )
    stop_reason = _required("timing_stop_reason", raw["timing_stop_reason"])
    anchored_complete_rounds = (
        context.anchored_candidate_expansion
        and stop_reason == CANDIDATE_EXPANSION_TIMING_STOP_REASON
    )
    if (
        stop_reason == CANDIDATE_EXPANSION_TIMING_STOP_REASON
        and not context.anchored_candidate_expansion
    ):
        raise ValueError(
            "CPU prefill anchored timing stop requires candidate expansion"
        )
    if timing_protocol not in SUPPORTED_ADAPTIVE_TIMING_PROTOCOLS:
        raise ValueError(f"unsupported CPU prefill timing protocol {timing_protocol}")
    if timing_protocol == ADAPTIVE_TIMING_PROTOCOL:
        missing_stationary = STATIONARY_RAW_COLUMNS.difference(raw)
        if missing_stationary:
            raise ValueError(
                "CPU prefill stationary timing columns are missing: "
                f"{sorted(missing_stationary)}"
            )
        stationary_begin = int(raw["stationary_sample_begin"])
        stationary_count = int(raw["stationary_sample_count"])
        stationary_duration_us = float(raw["stationary_duration_us"])
    else:
        stationary_begin = 0
        stationary_count = samples
        stationary_duration_us = timed_duration_us
    expected_ceiling_policy = (
        ADAPTIVE_TIMING_CEILING_POLICY
        if timing_protocol == ADAPTIVE_TIMING_PROTOCOL
        else LEGACY_ADAPTIVE_TIMING_CEILING_POLICY
    )
    if timing_ceiling_policy != expected_ceiling_policy:
        raise ValueError(
            f"unsupported CPU prefill timing ceiling {timing_ceiling_policy}"
        )
    if not (1 <= minimum_samples <= stable_samples <= maximum_samples):
        raise ValueError("CPU prefill adaptive sample bounds are invalid")
    if samples < 1:
        raise ValueError("CPU prefill sample count must be positive")
    if not all(math.isfinite(value) and value >= 0.0 for value in (
        preconditioning_budget_us, preconditioning_duration_us,
        warmup_probe_latency_us, warmup_budget_floor_us,
        warmup_latency_multiplier, warmup_budget_ceiling_us,
        warmup_budget_us, warmup_duration_us, timing_budget_us,
        warmup_wall_duration_us, warmup_max_round_duration_us,
        complete_round_probe_duration_us, warmup_round_timeout_us,
        timed_duration_us, stationary_duration_us, stability_limit,
    )):
        raise ValueError("CPU prefill adaptive timing thresholds must be finite")
    expected_warmup_policy = "disabled"
    if preconditioning_policy != "disabled":
        if m == preconditioning_m:
            expected_warmup_policy = "source-fixed-v1"
        elif (
            context.anchored_candidate_expansion
            and warmup_budget_policy == "transition-fixed-v1"
        ):
            expected_warmup_policy = "transition-fixed-v1"
        else:
            expected_warmup_policy = "transition-probe-scaled-v1"
    if warmup_budget_policy != expected_warmup_policy:
        raise ValueError(
            "CPU prefill warmup budget policy disagrees with source phase"
        )
    if warmup_budget_policy == "disabled":
        if any(value != 0.0 for value in (
            warmup_budget_floor_us,
            warmup_latency_multiplier,
            warmup_budget_ceiling_us,
            warmup_budget_us,
        )):
            raise ValueError("disabled CPU prefill warmup has a budget")
    elif warmup_budget_policy == "source-fixed-v1":
        if (
            warmup_budget_floor_us != preconditioning_budget_us
            or warmup_latency_multiplier != 0.0
            or warmup_budget_ceiling_us != 0.0
            or warmup_budget_us != warmup_budget_floor_us
        ):
            raise ValueError("CPU prefill source warmup budget is inconsistent")
    elif warmup_budget_policy == "transition-fixed-v1":
        if (
            not context.anchored_candidate_expansion
            or warmup_budget_floor_us <= 0.0
            or warmup_latency_multiplier != 0.0
            or warmup_budget_ceiling_us != 0.0
            or warmup_budget_us != warmup_budget_floor_us
        ):
            raise ValueError(
                "CPU prefill anchored transition warmup is inconsistent"
            )
    else:
        expected_warmup_budget_us = max(
            warmup_budget_floor_us,
            min(
                warmup_budget_ceiling_us,
                warmup_probe_latency_us * warmup_latency_multiplier,
            ),
        )
        if (
            warmup_probe_latency_us <= 0.0
            or warmup_budget_floor_us <= 0.0
            or warmup_latency_multiplier <= 0.0
            or warmup_budget_ceiling_us < warmup_budget_floor_us
            or not math.isclose(
                warmup_budget_us,
                expected_warmup_budget_us,
                rel_tol=0.0,
                abs_tol=1.0e-5,
            )
        ):
            raise ValueError("CPU prefill transition warmup formula is invalid")
    coordination_policy = _required(
        "measurement_coordination", raw["measurement_coordination"]
    )
    mpi_world_size = int(raw["mpi_world_size"])
    mpi_rank = int(raw["mpi_rank"])
    if coordination_policy not in {
        CPU_PREFILL_COORDINATION_POLICY,
        "process-local-complete-round-v1",
    }:
        raise ValueError("unsupported CPU prefill measurement coordination")
    if mpi_world_size < 1 or not (0 <= mpi_rank < mpi_world_size):
        raise ValueError("CPU prefill MPI rank provenance is invalid")
    if (context.profile.installable and
            coordination_policy == "process-local-complete-round-v1" and
            mpi_world_size != 1):
        raise ValueError("multi-rank CPU prefill evidence lacks MPI coordination")
    if (warmup_rounds < 0 or global_warmup_rounds < 0 or
            global_timing_rounds < 0 or timing_order_seed <= 0):
        raise ValueError("CPU prefill interleaved timing provenance is invalid")
    if forced_route_ok and warmups != warmup_rounds + 2:
        raise ValueError("CPU prefill warmup count disagrees with route launches")
    if forced_route_ok and (
        warmup_rounds != global_warmup_rounds or
        samples > global_timing_rounds
    ):
        raise ValueError("CPU prefill global round provenance disagrees")
    if (
        forced_route_ok
        and context.anchored_candidate_expansion
        and timing_protocol == ADAPTIVE_TIMING_PROTOCOL
        and stop_reason != "fixed_samples"
        and samples != global_timing_rounds
    ):
        raise ValueError(
            "stationary CPU prefill candidates did not share the final round"
        )
    if forced_route_ok and (
        warmup_round_timeout_us <= 0.0 or
        warmup_max_round_duration_us >= warmup_round_timeout_us or
        warmup_wall_duration_us < warmup_max_round_duration_us
    ):
        raise ValueError("CPU prefill warmup round watchdog is invalid")
    if complete_round_probe_duration_us > 0.0:
        expected_round_timeout_us = max(
            MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US,
            complete_round_probe_duration_us
            * CPU_PREFILL_DYNAMIC_WATCHDOG_MULTIPLIER
            + CPU_PREFILL_DYNAMIC_WATCHDOG_PADDING_US,
        )
        if not math.isclose(
            warmup_round_timeout_us,
            expected_round_timeout_us,
            rel_tol=0.0,
            abs_tol=5.1e-7,
        ):
            raise ValueError(
                "CPU prefill dynamic warmup watchdog disagrees with probes"
            )
    elif warmup_round_timeout_us > MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US:
        raise ValueError(
            "CPU prefill legacy warmup watchdog exceeds the reviewed bound"
        )
    if stop_reason not in {
        "elapsed_stable", "stable_sample_floor", "hard_samples_and_elapsed",
        "hard_samples_and_four_elapsed",
        "fixed_samples", "stationary_window",
        CANDIDATE_EXPANSION_TIMING_STOP_REASON,
    }:
        raise ValueError("CPU prefill timing stop reason is invalid")
    if anchored_complete_rounds and not (
        samples >= stable_samples
        and timed_duration_us >= timing_budget_us
        and stationary_begin == 0
        and stationary_count == samples
        and math.isclose(
            stationary_duration_us,
            timed_duration_us,
            rel_tol=0.0,
            abs_tol=5.1e-7,
        )
    ):
        raise ValueError(
            "CPU prefill anchored complete-round epoch lacks evidence floors"
        )
    if stop_reason == "elapsed_stable" and not (
        samples >= minimum_samples
        and timed_duration_us >= timing_budget_us
        and timing_converged
    ):
        raise ValueError("CPU prefill elapsed stop lacks elapsed stable evidence")
    if stop_reason == "stable_sample_floor" and not (
        samples >= stable_samples and timing_converged
    ):
        raise ValueError("CPU prefill sample-floor stop lacks stable evidence")
    if stop_reason == "hard_samples_and_elapsed" and not (
        timing_protocol == LEGACY_ADAPTIVE_TIMING_PROTOCOL
        and samples >= maximum_samples
        and timed_duration_us >= timing_budget_us
    ):
        raise ValueError("CPU prefill legacy hard ceiling lacks both guards")
    if stop_reason == "hard_samples_and_four_elapsed" and not (
        timing_protocol == ADAPTIVE_TIMING_PROTOCOL
        and samples >= maximum_samples
        and timed_duration_us >= (
            ADAPTIVE_TIMING_RECOVERY_WINDOW_MULTIPLIER * timing_budget_us
        )
    ):
        raise ValueError("CPU prefill recovery ceiling lacks both guards")
    if timing_protocol == ADAPTIVE_TIMING_PROTOCOL:
        if not (
            0 <= stationary_begin < samples
            and stationary_count >= 1
            and stationary_begin + stationary_count == samples
            and stationary_duration_us <= timed_duration_us + 5.1e-7
        ):
            raise ValueError("CPU prefill stationary timing window is invalid")
        if (
            forced_route_ok
            and stop_reason not in {
                "fixed_samples", CANDIDATE_EXPANSION_TIMING_STOP_REASON,
            }
            and not (
            stationary_count >= stable_samples
            and stationary_duration_us >= timing_budget_us
            )
        ):
            raise ValueError(
                "CPU prefill stationary timing window lacks evidence floors"
            )
        if stop_reason == "stationary_window" and not timing_converged:
            raise ValueError("CPU prefill stationary stop lacks convergence")
    if timing_samples is not None:
        if len(timing_samples) != samples:
            raise ValueError("CPU prefill timing sample count disagrees")
        if timing_protocol == ADAPTIVE_TIMING_PROTOCOL:
            if anchored_complete_rounds:
                expected_stationary_begin = 0
                stationary_samples = timing_samples
                expected_stationary_duration = sum(timing_samples)
            else:
                (
                    expected_stationary_begin,
                    stationary_samples,
                    expected_stationary_duration,
                ) = _stationary_window(
                    timing_samples,
                    stable_samples,
                    timing_budget_us,
                )
            if (
                stationary_begin != expected_stationary_begin
                or stationary_count != len(stationary_samples)
            ):
                raise ValueError(
                    "CPU prefill stationary suffix disagrees with sidecar"
                )
        else:
            stationary_samples = timing_samples
            expected_stationary_duration = sum(stationary_samples)
        verify_aggregate_timing(
            raw, sorted(stationary_samples), median_field="median_us"
        )
        recomputed_duration = sum(timing_samples)
        if not math.isclose(
            timed_duration_us,
            recomputed_duration,
            rel_tol=0.0,
            abs_tol=5.1e-7,
        ):
            raise ValueError("CPU prefill timed duration disagrees with sidecar")
        recomputed_stationary_duration = expected_stationary_duration
        if not math.isclose(
            stationary_duration_us,
            recomputed_stationary_duration,
            rel_tol=0.0,
            abs_tol=5.1e-7,
        ):
            raise ValueError(
                "CPU prefill stationary duration disagrees with sidecar"
            )
        recomputed_drift = _median_relative_drift(stationary_samples)
        if not math.isclose(
            reported_drift,
            recomputed_drift,
            rel_tol=0.0,
            abs_tol=5.1e-7,
        ):
            raise ValueError("CPU prefill median drift disagrees with sidecar")
        if timing_converged != (recomputed_drift <= stability_limit):
            raise ValueError("CPU prefill convergence flag disagrees with sidecar")
    elif context.profile.installable:
        raise ValueError("installable CPU prefill row lacks raw timing")

    if context.profile.installable and forced_route_ok:
        minimum_preconditioning_us = (
            MIN_CANDIDATE_EXPANSION_WARMUP_DURATION_US
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_WARMUP_DURATION_US
        )
        minimum_transition_us = (
            MIN_CANDIDATE_EXPANSION_TRANSITION_WARMUP_DURATION_US
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_TRANSITION_WARMUP_DURATION_US
        )
        minimum_timing_us = (
            MIN_CANDIDATE_EXPANSION_TIMED_DURATION_US
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_TIMED_DURATION_US
        )
        minimum_adaptive_samples = (
            MIN_CANDIDATE_EXPANSION_ADAPTIVE_SAMPLES
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_ADAPTIVE_SAMPLES
        )
        minimum_maximum_samples = (
            MIN_CANDIDATE_EXPANSION_MAXIMUM_SAMPLES
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_MAXIMUM_SAMPLES
        )
        if timing_protocol == ADAPTIVE_TIMING_PROTOCOL:
            timing_stop_complete = (
                (
                    stop_reason == "stationary_window"
                    and stationary_count >= stable_samples
                    and stationary_duration_us >= timing_budget_us
                )
                or (
                    anchored_complete_rounds
                    and stationary_count == samples
                    and stationary_duration_us >= timing_budget_us
                )
            )
            adaptive_sample_complete = (
                stationary_count >= minimum_adaptive_samples
            )
        else:
            timing_stop_complete = (
                (
                    stop_reason == "elapsed_stable"
                    and timed_duration_us >= timing_budget_us
                )
                or (
                    stop_reason == "stable_sample_floor"
                    and samples >= stable_samples
                    and stable_samples >= MIN_PROMOTION_SAMPLES
                )
            )
            adaptive_sample_complete = (
                minimum_samples >= minimum_adaptive_samples
            )
        required_warmups = (
            MIN_CANDIDATE_EXPANSION_WARMUPS
            if context.anchored_candidate_expansion
            else MIN_PROMOTION_WARMUPS
        )
        preconditioning_complete = (
            preconditioning_policy == CPU_PREFILL_PRECONDITIONING_POLICY
            and preconditioning_m >= 2
            and preconditioning_budget_us >= minimum_preconditioning_us
            and preconditioning_duration_us >= preconditioning_budget_us
        )
        if context.anchored_candidate_expansion:
            preconditioning_complete = preconditioning_complete or (
                preconditioning_policy == "disabled"
                and preconditioning_m == 0
                and preconditioning_budget_us == 0.0
                and preconditioning_duration_us == 0.0
            )
        warmup_policy_complete = (
            warmup_budget_policy == "source-fixed-v1"
            or (
                context.anchored_candidate_expansion
                and warmup_budget_policy == "transition-fixed-v1"
                and warmup_latency_multiplier == 0.0
                and warmup_budget_ceiling_us == 0.0
            )
            or (
                warmup_latency_multiplier
                >= MIN_PROMOTION_TRANSITION_WARMUP_LATENCY_MULTIPLIER
                and warmup_budget_ceiling_us
                >= MIN_PROMOTION_TRANSITION_WARMUP_BUDGET_CEILING_US
            )
        )
        if context.anchored_candidate_expansion:
            warmup_policy_complete = warmup_policy_complete or (
                warmup_budget_policy == "disabled"
                and warmup_budget_us == 0.0
                and warmup_duration_us == 0.0
                and warmup_latency_multiplier == 0.0
                and warmup_budget_ceiling_us == 0.0
            )
        promotion_complete = (
            warmups >= required_warmups
            and coordination_policy == CPU_PREFILL_COORDINATION_POLICY
            and 1 <= mpi_world_size <= 2
            and preconditioning_complete
            and warmup_budget_us
            >= minimum_transition_us
            and warmup_policy_complete
            and warmup_duration_us >= warmup_budget_us
            and (
                complete_round_probe_duration_us > 0.0
                or warmup_round_timeout_us
                <= MAX_PROMOTION_WARMUP_ROUND_TIMEOUT_US
            )
            and adaptive_sample_complete
            and maximum_samples >= minimum_maximum_samples
            and samples >= minimum_samples
            and timing_budget_us >= minimum_timing_us
            and stability_limit <= MAX_PROMOTION_MEDIAN_RELATIVE_DRIFT
            and (timing_converged or anchored_complete_rounds)
            and timing_stop_complete
        )
        if not promotion_complete:
            raise ValueError(
                "installable CPU prefill adaptive timing evidence is incomplete"
            )

    threads = int(raw["threads"])
    build_isa = _cpu_isa("build_isa", raw["build_isa"])
    requested_isa = _cpu_isa(
        "runtime_isa_requested", raw["runtime_isa_requested"], allow_auto=True
    )
    runtime_isa = _cpu_isa("runtime_isa_effective", raw["runtime_isa_effective"])
    if requested_isa != "AUTO" and requested_isa != runtime_isa:
        raise ValueError("requested CPU prefill ISA was not effective")
    if build_isa == "AVX2" and runtime_isa != "AVX2":
        raise ValueError("AVX2 build cannot execute AVX512 prefill")
    architecture_class = (
        f"{context.architecture_class}|build={build_isa}|"
        f"runtime={runtime_isa}|threads={threads}"
    )
    weight_bytes = int(raw["weight_bytes"])
    median_us = float(raw["median_us"])
    observation = NativeVNNIObservation(
        schema_version=SCHEMA_VERSION,
        run_id=context.run_id,
        corpus_id=context.corpus_id,
        git_revision=context.git_revision,
        build_id=f"{context.build_id}|cpu_isa={build_isa}",
        compiler_id=context.compiler_id,
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        backend=Backend.CPU,
        architecture_class=architecture_class,
        device_name=context.device_name,
        driver_runtime=context.driver_runtime,
        threading_or_stream_mode=(
            f"openmp:build={build_isa}:requested={requested_isa}:"
            f"effective={runtime_isa}:threads={threads}:"
            f"coordination={coordination_policy}:world={mpi_world_size}"
        ),
        semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        operation_kind="NativeVNNIPrefillProjection",
        bundle_signature=(
            CPU_PREFILL_KPART_BUNDLE
            if k_tiles > 1
            else CPU_PREFILL_FULL_K_BUNDLE
        ),
        projection_n_vector=(n,),
        source_format=source_format,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cpu"),
        packing_abi=spec.packing_abi("cpu"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=f"cpu-prefill:{raw['shape']}:n{n}:k{k}",
        shape_name=raw["shape"].strip(),
        execution_mode=ExecutionMode.EAGER,
        m=m,
        aggregate_n=n,
        k=k,
        aspect_ratio=float(n) / float(k),
        aspect_bucket=classify_aspect(n, k),
        work_items=n * k,
        n_tail_class=f"n_mod_64={n % 64}",
        k_tail_class=f"k_mod_32={k % 32}",
        alignment_class="cpu_native_vnni_64b_aligned",
        candidate_id=candidate.candidate_id,
        effective_candidate_id=candidate.effective_candidate_id,
        candidate_family=candidate.candidate_family,
        config_json=candidate.config_json,
        supported=forced_route_ok,
        graph_capture_ok=False,
        generic_eligible=True,
        arithmetic_fingerprint=candidate.arithmetic_fingerprint,
        serial_m1_policy_id=CPU_SERIAL_POLICY_ID,
        serial_m1_policy_hash=context.serial_m1_policy_hash,
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction,
        uses_atomic_reduction=False,
        trial_set_hash=_trial_set_hash(
            source_format, raw["shape"], m, n, k,
            build_isa, runtime_isa, threads,
            coordination_policy, mpi_world_size,
            timing_protocol,
        ),
        numerical_correctness=numerical_correctness,
        bitwise_equal=mismatch_count == 0,
        repeat_equal=repeat_mismatches == 0,
        mismatch_count=mismatch_count,
        first_mismatch_index=(
            None if mismatch_count == 0 else int(raw["first_bit_mismatch"])
        ),
        grouped_output_digest=grouped_digest,
        serial_output_digest=serial_digest,
        max_abs=diagnostics[0],
        relative_l2=diagnostics[1],
        cosine=diagnostics[2],
        symmetric_kld=diagnostics[3],
        warmup_count=warmups,
        sample_count=samples,
        min_us=float(raw["min_us"]),
        median_us=median_us,
        p95_us=float(raw["p95_us"]),
        mad_us=float(raw["mad_us"]),
        cv=float(raw["cv"]),
        timing_sample_hash=_required(
            "timing_sample_digest", raw["timing_sample_digest"]
        ),
        effective_bandwidth_gbs=(
            float(weight_bytes) / (median_us * 1.0e-6) / 1.0e9
            if median_us > 0.0 else 0.0
        ),
        forced_route_ok=forced_route_ok,
        observed_candidate_id=observed_id,
        route_counter_ok=route_counter_ok,
        workspace_ok=True,
        explicit_stream_ok=True,
        adaptive_timing_evidence={
            "coordination_policy": coordination_policy,
            "mpi_world_size": mpi_world_size,
            "preconditioning_policy": preconditioning_policy,
            "preconditioning_m": preconditioning_m,
            "preconditioning_budget_us": preconditioning_budget_us,
            "preconditioning_duration_us": preconditioning_duration_us,
            "warmup_budget_policy": warmup_budget_policy,
            "warmup_budget_us": warmup_budget_us,
            "warmup_duration_us": warmup_duration_us,
            "warmup_latency_multiplier": warmup_latency_multiplier,
            "warmup_budget_ceiling_us": warmup_budget_ceiling_us,
            "complete_round_probe_duration_us": (
                complete_round_probe_duration_us
            ),
            "warmup_round_timeout_us": warmup_round_timeout_us,
            "timing_protocol": timing_protocol,
            "timing_ceiling_policy": timing_ceiling_policy,
            "minimum_samples": minimum_samples,
            "stable_samples": stable_samples,
            "maximum_samples": maximum_samples,
            "timing_budget_us": timing_budget_us,
            "timed_duration_us": timed_duration_us,
            "median_stability_limit": stability_limit,
            "median_relative_drift": reported_drift,
            "timing_converged": timing_converged,
            "stop_reason": stop_reason,
            **({
                "stationary_sample_begin": stationary_begin,
                "stationary_sample_count": stationary_count,
                "stationary_duration_us": stationary_duration_us,
            } if timing_protocol == ADAPTIVE_TIMING_PROTOCOL else {}),
        },
    )
    observation.validate()
    return observation


def adapt_cpu_prefill_csv(
    paths: Iterable[Path],
    context: CPUPrefillAdapterContext,
    *,
    timing_sidecars: Iterable[Path] = (),
) -> ObservationCorpus:
    """Read strong CPU prefill shards into one validated common corpus."""

    context.validate()
    timing_index = read_cpu_prefill_timing_sidecars(timing_sidecars)
    if context.profile.installable and not timing_index:
        raise ValueError("installable CPU prefill corpus lacks timing sidecars")
    observations = []
    complete_round_provenance: dict[
        MeasurementCellKey, tuple[object, ...]
    ] = {}
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU prefill columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    samples = timing_index.pop(_timing_key(raw), None)
                    observation = adapt_cpu_prefill_row(raw, context, samples)
                    if observation.forced_route_ok:
                        cell_key = _measurement_cell_key(raw)
                        provenance = (
                            int(raw["warmup_count"]),
                            int(raw["warmup_round_count"]),
                            int(raw["global_warmup_round_count"]),
                            float(raw["warmup_wall_duration_us"]),
                            float(raw["warmup_max_round_duration_us"]),
                            float(
                                raw.get(
                                    "complete_round_probe_duration_us", "0"
                                )
                                or "0"
                            ),
                            float(raw["warmup_round_timeout_us"]),
                            int(raw["timing_order_seed"]),
                            int(raw["global_timing_round_count"]),
                            raw["timing_protocol"].strip(),
                            (
                                int(raw["sample_count"])
                                if (
                                    context.anchored_candidate_expansion
                                    and raw["timing_protocol"].strip()
                                    == ADAPTIVE_TIMING_PROTOCOL
                                )
                                else None
                            ),
                            raw["timing_ceiling_policy"].strip(),
                            raw["measurement_coordination"].strip(),
                            int(raw["mpi_world_size"]),
                            int(raw["mpi_rank"]),
                        )
                        previous = complete_round_provenance.setdefault(
                            cell_key, provenance
                        )
                        if previous != provenance:
                            raise ValueError(
                                "complete warmup-round provenance disagrees "
                                f"within measurement cell: {previous} != {provenance}"
                            )
                    observations.append(observation)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not observations:
        raise ValueError("CPU prefill inputs contained no observations")
    if timing_index:
        first = next(iter(timing_index))
        raise ValueError(f"CPU prefill timing has no aggregate row for {first}")
    return ObservationCorpus(observations)


__all__ = [
    "CPUPrefillAdapterContext",
    "adapt_cpu_prefill_csv",
    "adapt_cpu_prefill_row",
    "raw_corpus_id",
    "read_cpu_prefill_timing_sidecars",
]
