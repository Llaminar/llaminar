"""Strict adapter for CPU NativeVNNI production M=1 schedule evidence.

The CPU decode learner selects OpenMP N-chunk ownership only.  The existing
serial-M1 full-K/K-part arithmetic partition is frozen independently, so every
forceable candidate must be byte-identical to the explicit serial oracle.
Nominal schedules that normalize to another physical route remain negative
support evidence and can never create timing or profiler authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import verify_aggregate_timing
from ..candidate_registry import cpu_native_vnni_decode_registry
from ..corpus import ObservationCorpus
from ..format_registry import format_spec
from ..profiles import MIN_PROMOTION_SAMPLES, MIN_PROMOTION_WARMUPS, MeasurementProfile
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


CPU_DECODE_TRIAL_SET_VERSION = "cpu-decode-frozen-serial-m1-v1"
CPU_FROZEN_SERIAL_POLICY_ID = "cpu.nvnni.frozen-serial-m1-arithmetic-v1"

REQUIRED_RAW_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "build_isa", "runtime_isa_requested",
    "runtime_isa_effective", "threads", "weight_bytes", "warmup_count",
    "sample_count", "min_us", "median_us", "p95_us", "mad_us", "cv",
    "bit_mismatches", "first_bit_mismatch", "repeat_byte_mismatches",
    "max_abs", "relative_l2", "cosine", "symmetric_kld", "output_digest",
    "oracle_output_digest", "timing_sample_digest", "route_counter_ok",
    "observed_candidate_id", "k_tiles", "n_block_chunks", "serial_kpart",
    "numerical_correctness", "correctness_pass", "is_winner",
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


def _sha256(value: object) -> str:
    """Return one deterministic content identity."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _required(name: str, value: str) -> str:
    """Return non-empty trainer text or reject the evidence row."""

    text = value.strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _parse_bool(name: str, value: str) -> bool:
    """Parse only explicit trainer booleans."""

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _cpu_isa(name: str, value: str, *, allow_auto: bool = False) -> str:
    """Normalize one declared CPU ISA and reject scalar policy evidence."""

    normalized = _required(name, value).upper()
    allowed = {"AVX2", "AVX512"}
    if allow_auto:
        allowed.add("AUTO")
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return normalized


def _timing_key(raw: Mapping[str, str]) -> RawTimingKey:
    """Project an aggregate or sample row to the exact timing identity."""

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


def read_cpu_decode_timing_sidecars(
    paths: Iterable[Path],
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact sorted steady-clock samples with contiguous indices."""

    indexed: dict[RawTimingKey, list[float]] = {}
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TIMING_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU decode timing columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["backend"].strip().lower() != "cpu" or (
                    raw["phase"].strip() != "decode_m1"
                ):
                    raise ValueError(f"{path}:{row_number}: wrong timing surface")
                if raw["execution_mode"].strip().lower() != "eager":
                    raise ValueError(f"{path}:{row_number}: CPU mode must be eager")
                key = _timing_key(raw)
                sample_index = int(raw["sample_index"])
                samples = indexed.setdefault(key, [])
                if sample_index != len(samples):
                    raise ValueError(
                        f"{path}:{row_number}: timing sample index {sample_index} "
                        f"is not the next contiguous index {len(samples)}"
                    )
                latency_us = float.fromhex(raw["latency_us_hex"].strip())
                readable_us = float(raw["latency_us"])
                if latency_us <= 0.0 or not math.isfinite(latency_us):
                    raise ValueError(f"{path}:{row_number}: invalid raw latency")
                if not math.isclose(
                    readable_us, latency_us, rel_tol=0.0, abs_tol=5.1e-10
                ):
                    raise ValueError(
                        f"{path}:{row_number}: readable and exact latency disagree"
                    )
                samples.append(latency_us)

    result = {}
    for key, samples in indexed.items():
        values = tuple(samples)
        if tuple(sorted(values)) != values:
            raise ValueError(f"timing samples are not trainer-sorted for {key}")
        result[key] = values
    return result


@dataclass(frozen=True)
class CPUDecodeAdapterContext:
    """Immutable collection and CPU architecture provenance."""

    profile: MeasurementProfile
    run_id: str
    corpus_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    architecture_class: str
    device_name: str
    driver_runtime: str
    frozen_serial_policy_hash: str
    raw_timing_sidecar_retained: bool

    def validate(self) -> None:
        """Reject placeholder or incomplete installable provenance."""

        for name in (
            "run_id", "corpus_id", "git_revision", "build_id", "compiler_id",
            "architecture_class", "device_name", "driver_runtime",
            "frozen_serial_policy_hash",
        ):
            _required(name, getattr(self, name))
        if not self.corpus_id.startswith("sha256:"):
            raise ValueError("corpus_id must be a sha256 digest")
        if not self.frozen_serial_policy_hash.startswith("sha256:"):
            raise ValueError("frozen_serial_policy_hash must be a sha256 digest")
        if self.profile.installable:
            markers = (
                self.run_id, self.git_revision, self.build_id, self.compiler_id,
                self.architecture_class, self.device_name, self.driver_runtime,
            )
            if any(
                "workflow" in item.lower() or "unknown" in item.lower()
                for item in markers
            ):
                raise ValueError("installable CPU decode evidence has placeholders")
            if not self.raw_timing_sidecar_retained:
                raise ValueError("installable CPU decode evidence needs raw samples")


@lru_cache(maxsize=None)
def _trial_set_hash(
    source_format: str,
    shape: str,
    n: int,
    k: int,
    build_isa: str,
    runtime_isa: str,
    threads: int,
) -> str:
    """Hash one reused M=1 tensor/ISA/thread trial identity."""

    return _sha256({
        "version": CPU_DECODE_TRIAL_SET_VERSION,
        "source_format": source_format,
        "shape": shape,
        "m": 1,
        "n": n,
        "k": k,
        "build_isa": build_isa,
        "runtime_isa": runtime_isa,
        "threads": threads,
    })


def adapt_cpu_decode_row(
    raw: Mapping[str, str],
    context: CPUDecodeAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
) -> NativeVNNIObservation:
    """Convert one strong CPU M=1 candidate row to common evidence."""

    if raw.get("backend", "").strip().lower() != "cpu" or (
        raw.get("phase", "").strip() != "decode_m1"
    ):
        raise ValueError("CPU decode adapter received the wrong surface")
    if raw["execution_mode"].strip().lower() != "eager":
        raise ValueError("CPU decode observations must use eager execution")
    if int(raw["m"]) != 1:
        raise ValueError("CPU decode schedule evidence requires M=1")

    n = int(raw["n"])
    k = int(raw["k"])
    if n <= 0 or k <= 0 or k % 32 != 0:
        raise ValueError("CPU decode geometry is invalid")
    registry = cpu_native_vnni_decode_registry()
    candidate = registry.resolve(raw["candidate_id"])
    if not candidate.supports_contract(SemanticContract.FAST):
        raise ValueError(f"{candidate.candidate_id} lacks Fast M=1 semantics")

    source_format = raw["source_format"].strip().upper()
    spec = format_spec(source_format)
    if int(raw["source_codebook"]) != spec.source_codebook_id:
        raise ValueError(f"{source_format}: source codebook disagrees with registry")
    runtime_codebook = spec.runtime_codebook("cpu")
    if int(raw["execution_codebook"]) != runtime_codebook:
        raise ValueError(f"{source_format}: execution codebook disagrees with registry")

    route_counter_ok = _parse_bool("route_counter_ok", raw["route_counter_ok"])
    observed_raw = _required("observed_candidate_id", raw["observed_candidate_id"])
    try:
        observed_id = registry.resolve(observed_raw).effective_candidate_id
    except ValueError:
        observed_id = f"unresolved:{observed_raw}"
    forced_route_ok = route_counter_ok and (
        observed_id == candidate.effective_candidate_id
    )

    mismatch_count = int(raw["bit_mismatches"])
    repeat_mismatches = int(raw["repeat_byte_mismatches"])
    output_digest = _required("output_digest", raw["output_digest"])
    oracle_digest = _required("oracle_output_digest", raw["oracle_output_digest"])
    if (mismatch_count == 0) != (output_digest == oracle_digest):
        raise ValueError("CPU decode mismatch count disagrees with output digests")
    diagnostics = tuple(float(raw[name]) for name in (
        "max_abs", "relative_l2", "cosine", "symmetric_kld"
    ))
    if not all(math.isfinite(value) for value in diagnostics):
        raise ValueError("CPU decode diagnostics must be finite")
    numerical_correctness = _parse_bool(
        "numerical_correctness", raw["numerical_correctness"]
    )
    expected_pass = (
        forced_route_ok
        and mismatch_count == 0
        and repeat_mismatches == 0
        and numerical_correctness
    )
    if _parse_bool("correctness_pass", raw["correctness_pass"]) != expected_pass:
        raise ValueError("CPU decode correctness_pass disagrees with evidence")

    warmups = int(raw["warmup_count"])
    samples = int(raw["sample_count"])
    if context.profile.installable and forced_route_ok and (
        warmups < MIN_PROMOTION_WARMUPS or samples < MIN_PROMOTION_SAMPLES
    ):
        raise ValueError(
            f"installable profile requires {MIN_PROMOTION_WARMUPS}/"
            f"{MIN_PROMOTION_SAMPLES} timing, got {warmups}/{samples}"
        )
    if timing_samples is not None:
        if len(timing_samples) != samples:
            raise ValueError(
                f"raw timing sidecar has {len(timing_samples)} samples; "
                f"expected {samples}"
            )
        verify_aggregate_timing(raw, timing_samples, median_field="median_us")
    elif context.profile.installable:
        raise ValueError("installable CPU decode corpus is missing raw timing")

    threads = int(raw["threads"])
    if threads <= 0:
        raise ValueError("threads must be positive")
    build_isa = _cpu_isa("build_isa", raw["build_isa"])
    requested_runtime_isa = _cpu_isa(
        "runtime_isa_requested", raw["runtime_isa_requested"], allow_auto=True
    )
    effective_runtime_isa = _cpu_isa(
        "runtime_isa_effective", raw["runtime_isa_effective"]
    )
    if requested_runtime_isa != "AUTO" and (
        requested_runtime_isa != effective_runtime_isa
    ):
        raise ValueError("requested CPU runtime ISA did not become effective")
    if build_isa == "AVX2" and effective_runtime_isa != "AVX2":
        raise ValueError("an AVX2 build cannot execute AVX512")

    n_block_chunks = int(raw["n_block_chunks"])
    if forced_route_ok and (
        n_block_chunks != int(candidate.config_json["n_block_chunks"])
    ):
        raise ValueError("CPU decode route counter disagrees with candidate NBC")
    k_tiles = int(raw["k_tiles"])
    serial_kpart = _parse_bool("serial_kpart", raw["serial_kpart"])
    if serial_kpart != (k_tiles > 1):
        raise ValueError("serial_kpart disagrees with the frozen K-tile count")

    architecture_class = (
        f"{context.architecture_class}|build={build_isa}|"
        f"runtime={effective_runtime_isa}|threads={threads}"
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
            f"openmp:build={build_isa}:requested={requested_runtime_isa}:"
            f"effective={effective_runtime_isa}:threads={threads}"
        ),
        semantic_contract=SemanticContract.FAST,
        operation_kind="NativeVNNIFastM1Projection",
        bundle_signature=(
            "single-native-vnni-decode:serial-kpart:fp32-output:v1"
            if serial_kpart
            else "single-native-vnni-decode:serial-full-k:fp32-output:v1"
        ),
        projection_n_vector=(n,),
        source_format=source_format,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cpu"),
        packing_abi=spec.packing_abi("cpu"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=f"cpu-decode:{raw['shape']}:n{n}:k{k}",
        shape_name=raw["shape"].strip(),
        execution_mode=ExecutionMode.EAGER,
        m=1,
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
        serial_m1_policy_id=CPU_FROZEN_SERIAL_POLICY_ID,
        serial_m1_policy_hash=context.frozen_serial_policy_hash,
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction,
        uses_atomic_reduction=False,
        trial_set_hash=_trial_set_hash(
            source_format,
            raw["shape"],
            n,
            k,
            build_isa,
            effective_runtime_isa,
            threads,
        ),
        numerical_correctness=numerical_correctness,
        bitwise_equal=mismatch_count == 0,
        repeat_equal=repeat_mismatches == 0,
        mismatch_count=mismatch_count,
        first_mismatch_index=(
            None if mismatch_count == 0 else int(raw["first_bit_mismatch"])
        ),
        grouped_output_digest=output_digest,
        serial_output_digest=oracle_digest,
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
    )
    observation.validate()
    return observation


def adapt_cpu_decode_csv(
    paths: Iterable[Path],
    context: CPUDecodeAdapterContext,
    *,
    timing_sidecars: Iterable[Path] = (),
) -> ObservationCorpus:
    """Read strong CPU M=1 shards into one validated common corpus."""

    context.validate()
    timing_index = read_cpu_decode_timing_sidecars(timing_sidecars)
    if context.profile.installable and not timing_index:
        raise ValueError("installable CPU decode corpus is missing sidecars")
    observations = []
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing strong CPU decode columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    samples = timing_index.pop(_timing_key(raw), None)
                    observations.append(adapt_cpu_decode_row(raw, context, samples))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not observations:
        raise ValueError("CPU decode inputs contained no observations")
    if timing_index:
        first = next(iter(timing_index))
        raise ValueError(f"CPU decode sidecar has no aggregate row for {first}")
    return ObservationCorpus(observations)
