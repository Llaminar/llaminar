"""ROCm NativeVNNI decode trainer adapter.

The production speedometer emits one row for a forceable serial-M1 KB choice
or for the grouped verifier's explicit ``INHERIT_SERIAL_M1`` schedule.  This
adapter validates route identity, serial-policy provenance, exact output
evidence, repeat stability, and raw HIP-event samples before constructing the
backend-neutral observation corpus.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import verify_aggregate_timing
from ..candidate_registry import rocm_native_vnni_decode_registry
from ..corpus import ObservationCorpus
from ..format_registry import format_spec
from ..profiles import (
    MIN_PROMOTION_SAMPLES,
    MIN_PROMOTION_WARMUPS,
    MeasurementProfile,
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


ROCM_DECODE_TRIAL_SET_VERSION = "rocm-decode-deterministic-input-weights-v2"
ROCM_DECODE_SERIAL_POLICY_ID = "rocm.nvnni.production.serial-m1-policy-v2"

REQUIRED_RAW_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "kb", "target_waves", "weight_bytes", "warmup_count",
    "sample_count", "min_us", "median_us", "p95_us", "mad_us", "cv",
    "effective_bandwidth_gbs", "bit_mismatches", "first_bit_mismatch",
    "repeat_byte_mismatches", "max_abs", "relative_l2", "cosine",
    "symmetric_kld", "grouped_output_digest", "serial_output_digest",
    "timing_sample_digest", "route_counter_ok", "observed_candidate_id",
    "observed_path", "serial_m1_kb", "serial_m1_target_waves",
    "serial_route_counter_ok", "numerical_correctness", "correctness_pass",
    "is_winner",
})

REQUIRED_TIMING_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "kb", "target_waves", "sample_index", "timed_replays",
    "latency_us", "latency_us_hex",
})

RawTimingKey = tuple[str, int, int, str, str, int, int, int, str, int, int]


def _sha256(value: object) -> str:
    """Hash one deterministic trial/provenance description."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def _execution_mode(value: str) -> ExecutionMode:
    try:
        return ExecutionMode(value.strip().lower())
    except ValueError as exc:
        raise ValueError(f"unknown ROCm decode execution mode {value!r}") from exc


def _semantic_contract(m: int) -> SemanticContract:
    if m == 1:
        return SemanticContract.FAST
    if m >= 2:
        return SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    raise ValueError(f"ROCm decode trainer requires positive M, got M={m}")


def _timing_key(raw: Mapping[str, str]) -> RawTimingKey:
    """Project aggregate and sidecar rows onto one exact candidate trial."""

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
        int(raw["kb"]),
        int(raw["target_waves"]),
    )


def read_rocm_decode_timing_sidecars(
    paths: Iterable[Path],
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact HIP-event microseconds and validate sidecar continuity."""

    indexed: dict[RawTimingKey, dict[int, float]] = {}
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TIMING_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing ROCm decode timing columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["backend"].strip().lower() != "rocm" or (
                    raw["phase"].strip() != "decode"
                ):
                    raise ValueError(f"{path}:{row_number}: wrong timing sidecar surface")
                _execution_mode(raw["execution_mode"])
                if int(raw["timed_replays"]) != 1:
                    raise ValueError(
                        f"{path}:{row_number}: decode timing rows require one replay"
                    )
                key = _timing_key(raw)
                sample_index = int(raw["sample_index"])
                if sample_index < 0 or sample_index in indexed.setdefault(key, {}):
                    raise ValueError(
                        f"{path}:{row_number}: duplicate/negative sample index {sample_index}"
                    )

                latency_us = float.fromhex(raw["latency_us_hex"].strip())
                readable_us = float(raw["latency_us"])
                if latency_us <= 0.0 or not math.isfinite(latency_us):
                    raise ValueError(f"{path}:{row_number}: invalid raw latency")
                if not math.isclose(
                    readable_us, latency_us, rel_tol=0.0, abs_tol=5.1e-7
                ):
                    raise ValueError(
                        f"{path}:{row_number}: readable and exact latency fields disagree"
                    )
                indexed[key][sample_index] = latency_us

    result = {}
    for key, samples in indexed.items():
        expected = list(range(len(samples)))
        if sorted(samples) != expected:
            raise ValueError(f"timing sidecar has non-contiguous samples for {key}")
        values = tuple(samples[index] for index in expected)
        if tuple(sorted(values)) != values:
            raise ValueError(f"timing sidecar samples are not trainer-sorted for {key}")
        result[key] = values
    return result


@dataclass(frozen=True)
class ROCmDecodeAdapterContext:
    """Immutable build, device, profile, and frozen-M1 provenance."""

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
    threading_or_stream_mode: str = "explicit_non_default_hip_stream"

    @classmethod
    def workflow_smoke(cls, *, corpus_id: str) -> "ROCmDecodeAdapterContext":
        """Create visibly non-installable provenance for local harness checks."""

        return cls(
            profile=MeasurementProfile.QUICK,
            run_id="workflow-smoke",
            corpus_id=corpus_id,
            git_revision="workflow-uncommitted",
            build_id="workflow-release-build",
            compiler_id="workflow-hip-compiler",
            architecture_class="workflow-rocm-architecture",
            device_name="workflow-rocm-device",
            driver_runtime="workflow-rocm-runtime",
            serial_m1_policy_hash="sha256:" + "0" * 64,
            raw_timing_sidecar_retained=False,
        )

    def validate(self) -> None:
        """Reject placeholder provenance and missing sidecars for promotion."""

        for name in (
            "run_id", "corpus_id", "git_revision", "build_id", "compiler_id",
            "architecture_class", "device_name", "driver_runtime",
            "serial_m1_policy_hash", "threading_or_stream_mode",
        ):
            _required(name, getattr(self, name))
        if not self.corpus_id.startswith("sha256:"):
            raise ValueError("corpus_id must be a sha256 digest")
        if not self.serial_m1_policy_hash.startswith("sha256:"):
            raise ValueError("serial_m1_policy_hash must be a sha256 digest")
        if self.profile.installable:
            markers = (
                self.run_id,
                self.git_revision,
                self.build_id,
                self.compiler_id,
                self.architecture_class,
                self.device_name,
                self.driver_runtime,
            )
            if any("workflow" in item.lower() or "unknown" in item.lower()
                   for item in markers):
                raise ValueError("installable ROCm decode evidence has placeholder provenance")
            if not self.raw_timing_sidecar_retained:
                raise ValueError(
                    "installable ROCm decode evidence must retain raw timing sidecars"
                )


def raw_corpus_id(paths: Iterable[Path]) -> str:
    """Hash ordered aggregate/sidecar shard bytes for corpus provenance."""

    digest = hashlib.sha256()
    for path in sorted(Path(item) for item in paths):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _trial_set_hash(raw: Mapping[str, str]) -> str:
    """Identify the deterministic C++ input/weight fixture for one row."""

    return _sha256({
        "version": ROCM_DECODE_TRIAL_SET_VERSION,
        "source_format": raw["source_format"].strip().upper(),
        "shape": raw["shape"].strip(),
        "m": int(raw["m"]),
        "n": int(raw["n"]),
        "k": int(raw["k"]),
    })


def adapt_rocm_decode_row(
    raw: Mapping[str, str],
    context: ROCmDecodeAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
) -> NativeVNNIObservation:
    """Convert one strong ROCm decode row into the common schema."""

    if raw.get("backend", "").strip().lower() != "rocm":
        raise ValueError("ROCm decode adapter received a non-ROCm row")
    if raw.get("phase", "").strip() != "decode":
        raise ValueError("ROCm decode adapter received a non-decode row")

    m = int(raw["m"])
    n = int(raw["n"])
    k = int(raw["k"])
    contract = _semantic_contract(m)
    execution_mode = _execution_mode(raw["execution_mode"])
    registry = rocm_native_vnni_decode_registry()
    candidate = registry.resolve(raw["candidate_id"])
    if not candidate.supports_contract(contract):
        raise ValueError(
            f"{candidate.candidate_id} does not support {contract.value}"
        )

    source_format = raw["source_format"].strip().upper()
    format_entry = format_spec(source_format)
    if int(raw["source_codebook"]) != format_entry.source_codebook_id:
        raise ValueError(f"{source_format}: source codebook disagrees with registry")
    runtime_codebook = format_entry.runtime_codebook("rocm")
    if int(raw["execution_codebook"]) != runtime_codebook:
        raise ValueError(f"{source_format}: execution codebook disagrees with registry")

    observed_kb = int(raw["kb"])
    observed_waves = int(raw["target_waves"])
    serial_kb = int(raw["serial_m1_kb"])
    serial_waves = int(raw["serial_m1_target_waves"])
    if min(observed_kb, observed_waves, serial_kb, serial_waves) <= 0:
        raise ValueError("ROCm decode route configuration must be positive")
    if contract == SemanticContract.FAST:
        config_matches = observed_kb == int(candidate.config_json["kb"])
    else:
        config_matches = observed_kb == serial_kb and observed_waves == serial_waves

    route_counter_ok = _parse_bool("route_counter_ok", raw["route_counter_ok"])
    serial_route_ok = _parse_bool(
        "serial_route_counter_ok", raw["serial_route_counter_ok"]
    )
    observed_raw = _required("observed_candidate_id", raw["observed_candidate_id"])
    try:
        observed_id = registry.resolve(observed_raw).effective_candidate_id
    except ValueError:
        observed_id = f"unresolved:{observed_raw}"
    forced_route_ok = (
        route_counter_ok
        and serial_route_ok
        and config_matches
        and observed_id == candidate.effective_candidate_id
    )

    mismatch_count = int(raw["bit_mismatches"])
    repeat_mismatches = int(raw["repeat_byte_mismatches"])
    grouped_digest = _required("grouped_output_digest", raw["grouped_output_digest"])
    serial_digest = _required("serial_output_digest", raw["serial_output_digest"])
    if (mismatch_count == 0) != (grouped_digest == serial_digest):
        raise ValueError(
            "ROCm decode mismatch count disagrees with grouped/serial digests"
        )
    diagnostics = tuple(float(raw[name]) for name in (
        "max_abs", "relative_l2", "cosine", "symmetric_kld"
    ))
    if not all(math.isfinite(value) for value in diagnostics):
        raise ValueError("ROCm decode diagnostics must be finite")
    numerical_correctness = _parse_bool(
        "numerical_correctness", raw["numerical_correctness"]
    )
    expected_pass = (
        route_counter_ok
        and repeat_mismatches == 0
        and numerical_correctness
        and (contract == SemanticContract.FAST or mismatch_count == 0)
    )
    if _parse_bool("correctness_pass", raw["correctness_pass"]) != expected_pass:
        raise ValueError("ROCm decode correctness_pass disagrees with evidence fields")

    warmups = int(raw["warmup_count"])
    samples = int(raw["sample_count"])
    if context.profile.installable and (
        warmups < MIN_PROMOTION_WARMUPS or samples < MIN_PROMOTION_SAMPLES
    ):
        raise ValueError(
            f"installable profile requires at least "
            f"{MIN_PROMOTION_WARMUPS} warmups/{MIN_PROMOTION_SAMPLES} samples; "
            f"got {warmups}/{samples}"
        )
    if timing_samples is not None:
        if len(timing_samples) != samples:
            raise ValueError(
                f"raw timing sidecar has {len(timing_samples)} samples; expected {samples}"
            )
        verify_aggregate_timing(raw, timing_samples, median_field="median_us")
    elif context.profile.installable:
        raise ValueError("installable profile is missing raw timing samples")

    observed_path = _required("observed_path", raw["observed_path"])
    used_atomics = observed_path == "atomic_reduce"
    projection_vector = (n,)
    observation = NativeVNNIObservation(
        schema_version=SCHEMA_VERSION,
        run_id=context.run_id,
        corpus_id=context.corpus_id,
        git_revision=context.git_revision,
        build_id=context.build_id,
        compiler_id=context.compiler_id,
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        backend=Backend.ROCM,
        architecture_class=context.architecture_class,
        device_name=context.device_name,
        driver_runtime=context.driver_runtime,
        threading_or_stream_mode=context.threading_or_stream_mode,
        semantic_contract=contract,
        operation_kind="NativeVNNIDecodeProjection",
        bundle_signature="single-native-vnni-projection:fp32-output:v2",
        projection_n_vector=projection_vector,
        source_format=source_format,
        source_codebook_id=format_entry.source_codebook_id,
        prepared_family_id=format_entry.prepared_family("rocm"),
        packing_abi=format_entry.packing_abi("rocm"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=f"rocm-decode:{raw['shape']}:n{n}:k{k}",
        shape_name=raw["shape"].strip(),
        execution_mode=execution_mode,
        m=m,
        aggregate_n=n,
        k=k,
        aspect_ratio=float(n) / float(k),
        aspect_bucket=classify_aspect(n, k),
        work_items=n * k,
        n_tail_class=f"n_mod_256={n % 256}",
        k_tail_class=f"k_mod_256={k % 256}",
        alignment_class="gpu_prepared_native_vnni_16b_aligned",
        candidate_id=candidate.candidate_id,
        effective_candidate_id=candidate.effective_candidate_id,
        candidate_family=candidate.candidate_family,
        config_json=candidate.config_json,
        supported=config_matches,
        graph_capture_ok=candidate.graph_capture_supported,
        generic_eligible=True,
        arithmetic_fingerprint=candidate.arithmetic_fingerprint,
        serial_m1_policy_id=ROCM_DECODE_SERIAL_POLICY_ID,
        serial_m1_policy_hash=context.serial_m1_policy_hash,
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction and not used_atomics,
        uses_atomic_reduction=candidate.uses_atomic_reduction or used_atomics,
        trial_set_hash=_trial_set_hash(raw),
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
        median_us=float(raw["median_us"]),
        p95_us=float(raw["p95_us"]),
        mad_us=float(raw["mad_us"]),
        cv=float(raw["cv"]),
        timing_sample_hash=_required(
            "timing_sample_digest", raw["timing_sample_digest"]
        ),
        effective_bandwidth_gbs=float(raw["effective_bandwidth_gbs"]),
        forced_route_ok=forced_route_ok,
        observed_candidate_id=observed_id,
        route_counter_ok=route_counter_ok and serial_route_ok,
        workspace_ok=True,
        explicit_stream_ok=True,
    )
    observation.validate()
    return observation


def adapt_rocm_decode_csv(
    paths: Iterable[Path],
    context: ROCmDecodeAdapterContext,
    *,
    timing_sidecars: Iterable[Path] = (),
) -> ObservationCorpus:
    """Read strong decode shards and return one validated common corpus."""

    context.validate()
    timing_index = read_rocm_decode_timing_sidecars(timing_sidecars)
    if context.profile.installable and not timing_index:
        raise ValueError("installable ROCm decode corpus is missing raw timing sidecars")
    observations = []
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing strong ROCm decode columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    samples = timing_index.pop(_timing_key(raw), None)
                    observations.append(adapt_rocm_decode_row(raw, context, samples))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not observations:
        raise ValueError("ROCm decode trainer inputs contained no observations")
    if timing_index:
        first = next(iter(timing_index))
        raise ValueError(f"raw timing sidecar has no aggregate row for {first}")
    return ObservationCorpus(observations)
