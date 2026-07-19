"""CUDA NativeVNNI public-M1 and grouped-verifier trainer adapter.

The CUDA speedometer emits one row for an explicit, normalized public-M1
candidate or for the grouped verifier's ``INHERIT_SERIAL_M1`` candidate.  This
adapter proves that the requested production route really ran, validates raw
CUDA-event timings, and requires grouped runtime-M output bytes to equal repeated
public M1 decode rows.  Approximate diagnostics remain useful breadcrumbs but
cannot make a verifier candidate eligible.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import raw_corpus_id, verify_aggregate_timing
from ..candidate_registry import cuda_native_vnni_gemv_registry
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


CUDA_DECODE_TRIAL_SET_VERSION = "cuda-decode-sample-interleaved-order-v6"
CUDA_DECODE_SERIAL_POLICY_ID = "cuda.nvnni.production.public-m1-policy-v4"
CUDA_DECODE_MEASUREMENT_PROTOCOL = "sample_interleaved_v1"

REQUIRED_RAW_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "measurement_order", "measurement_order_seed",
    "measurement_protocol", "family",
    "tile_n", "cpt", "target_waves", "mkg",
    "max_kb", "exact_kb", "force_two_phase", "weight_bytes", "warmup_count",
    "sample_count", "min_us", "median_us", "p95_us", "mad_us", "cv",
    "effective_bandwidth_gbs", "bit_mismatches", "first_bit_mismatch",
    "repeat_byte_mismatches", "max_abs", "relative_l2", "cosine",
    "symmetric_kld", "grouped_output_digest", "serial_output_digest",
    "timing_sample_digest", "supported", "graph_capture_ok", "workspace_ok",
    "explicit_stream_ok", "route_counter_ok", "observed_candidate_id",
    "observed_path", "observed_tile_n", "observed_cpt", "observed_effective_kb",
    "serial_m1_candidate_id", "serial_route_counter_ok",
    "numerical_correctness", "correctness_pass", "is_winner",
})

REQUIRED_TIMING_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "measurement_order", "measurement_order_seed",
    "measurement_protocol", "sample_measurement_order", "sample_order_seed",
    "sample_index", "timed_replays", "latency_us",
    "latency_us_hex",
})

RawTimingKey = tuple[str, int, int, str, str, int, int, int, str, int, int]
MeasurementGroupKey = tuple[str, int, int, str, str, int, int, int]


def _sha256(value: object) -> str:
    """Return a stable identity for a deterministic trial description."""

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
        raise ValueError(f"unknown CUDA decode execution mode {value!r}") from exc


def _semantic_contract(m: int) -> SemanticContract:
    if m == 1:
        return SemanticContract.FAST
    if m >= 2:
        return SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    raise ValueError(f"CUDA decode trainer requires M>=1, got M={m}")


def _timing_key(raw: Mapping[str, str]) -> RawTimingKey:
    """Project aggregate and sidecar rows onto one normalized trial."""

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
        int(raw["measurement_order"]),
        int(raw["measurement_order_seed"]),
    )


def _measurement_group_key(raw: Mapping[str, str]) -> MeasurementGroupKey:
    """Identify one candidate permutation shared by aggregate rows."""

    return (
        raw["source_format"].strip().upper(),
        int(raw["source_codebook"]),
        int(raw["execution_codebook"]),
        raw["shape"].strip(),
        raw["execution_mode"].strip().lower(),
        int(raw["m"]),
        int(raw["n"]),
        int(raw["k"]),
    )


def _validate_measurement_orders(
    groups: Mapping[MeasurementGroupKey, list[tuple[int, int, str]]],
) -> None:
    """Require one complete deterministic candidate permutation per cell."""

    for key, entries in groups.items():
        seeds = {seed for _order, seed, _candidate in entries}
        orders = [order for order, _seed, _candidate in entries]
        candidates = [candidate for _order, _seed, candidate in entries]
        if len(seeds) != 1 or next(iter(seeds), 0) <= 0:
            raise ValueError(
                f"CUDA measurement-order seed is inconsistent for {key}"
            )
        if sorted(orders) != list(range(len(entries))):
            raise ValueError(
                "CUDA measurement order is not one contiguous permutation "
                f"for {key}: {sorted(orders)}"
            )
        if len(candidates) != len(set(candidates)):
            raise ValueError(
                f"CUDA measurement order repeats a candidate for {key}"
            )


def read_cuda_decode_timing_sidecars(
    paths: Iterable[Path],
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact samples and prove complete interleaved candidate rounds."""

    indexed: dict[RawTimingKey, list[float]] = {}
    replay_count: dict[RawTimingKey, int] = {}
    sample_rounds: dict[
        tuple[MeasurementGroupKey, int],
        list[tuple[int, int, str]],
    ] = defaultdict(list)
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TIMING_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CUDA decode timing columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["backend"].strip().lower() != "cuda" or (
                    raw["phase"].strip() != "decode"
                ):
                    raise ValueError(f"{path}:{row_number}: wrong timing sidecar surface")
                if raw["measurement_protocol"].strip() != (
                    CUDA_DECODE_MEASUREMENT_PROTOCOL
                ):
                    raise ValueError(
                        f"{path}:{row_number}: CUDA timing is not "
                        "sample-interleaved"
                    )
                _execution_mode(raw["execution_mode"])
                timed_replays = int(raw["timed_replays"])
                if timed_replays <= 0:
                    raise ValueError(
                        f"{path}:{row_number}: CUDA timed_replays must be positive"
                    )
                key = _timing_key(raw)
                previous_replays = replay_count.setdefault(key, timed_replays)
                if previous_replays != timed_replays:
                    raise ValueError(
                        f"{path}:{row_number}: CUDA timed_replays changed from "
                        f"{previous_replays} to {timed_replays} within one trial"
                    )
                sample_index = int(raw["sample_index"])
                samples = indexed.setdefault(key, [])
                if sample_index != len(samples):
                    raise ValueError(
                        f"{path}:{row_number}: timing sample index {sample_index} "
                        f"is not the next contiguous index {len(samples)}"
                    )
                sample_measurement_order = int(raw["sample_measurement_order"])
                sample_order_seed = int(raw["sample_order_seed"])
                if sample_measurement_order < 0 or sample_order_seed <= 0:
                    raise ValueError(
                        f"{path}:{row_number}: invalid interleaved sample order"
                    )
                sample_rounds[
                    (_measurement_group_key(raw), sample_index)
                ].append((
                    sample_measurement_order,
                    sample_order_seed,
                    raw["candidate_id"].strip().lower(),
                ))
                latency_us = float.fromhex(raw["latency_us_hex"].strip())
                readable_us = float(raw["latency_us"])
                if latency_us <= 0.0 or not math.isfinite(latency_us):
                    raise ValueError(f"{path}:{row_number}: invalid raw latency")
                if not math.isclose(
                    readable_us, latency_us, rel_tol=0.0, abs_tol=5.1e-7
                ):
                    raise ValueError(
                        f"{path}:{row_number}: readable and exact latency disagree"
                    )
                samples.append(latency_us)

    rounds_by_group: dict[
        MeasurementGroupKey,
        dict[int, list[tuple[int, int, str]]],
    ] = defaultdict(dict)
    for (group, sample_index), rows in sample_rounds.items():
        rounds_by_group[group][sample_index] = rows
    for group, rounds in rounds_by_group.items():
        if sorted(rounds) != list(range(len(rounds))):
            raise ValueError(
                f"CUDA interleaved sample indices are not contiguous for {group}"
            )
        expected_candidates: set[str] | None = None
        for sample_index, rows in sorted(rounds.items()):
            orders = [order for order, _seed, _candidate in rows]
            seeds = {seed for _order, seed, _candidate in rows}
            candidates = {candidate for _order, _seed, candidate in rows}
            if sorted(orders) != list(range(len(rows))):
                raise ValueError(
                    "CUDA sample measurement order is not one contiguous "
                    f"permutation for {group} sample {sample_index}"
                )
            if len(seeds) != 1 or next(iter(seeds), 0) <= 0:
                raise ValueError(
                    f"CUDA sample-order seed is inconsistent for {group} "
                    f"sample {sample_index}"
                )
            if len(candidates) != len(rows):
                raise ValueError(
                    f"CUDA sample round repeats a candidate for {group} "
                    f"sample {sample_index}"
                )
            if expected_candidates is None:
                expected_candidates = candidates
            elif candidates != expected_candidates:
                raise ValueError(
                    f"CUDA sample round candidate set changed for {group} "
                    f"sample {sample_index}"
                )

    result = {}
    for key, samples in indexed.items():
        # Aggregate statistics and timing digests are intentionally computed
        # from sorted samples. The raw sidecar remains in chronological sample
        # order so the independent per-round permutations can be audited.
        result[key] = tuple(sorted(samples))
    return result


@dataclass(frozen=True)
class CUDADecodeAdapterContext:
    """Immutable build, GPU, profile, and frozen-public-M1 provenance."""

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
    threading_or_stream_mode: str = "explicit_non_default_cuda_stream"

    @classmethod
    def workflow_smoke(cls, *, corpus_id: str) -> "CUDADecodeAdapterContext":
        """Create conspicuously non-installable local workflow provenance."""

        return cls(
            profile=MeasurementProfile.QUICK,
            run_id="workflow-smoke",
            corpus_id=corpus_id,
            git_revision="workflow-uncommitted",
            build_id="workflow-release-build",
            compiler_id="workflow-cuda-compiler",
            architecture_class="workflow-cuda-architecture",
            device_name="workflow-cuda-device",
            driver_runtime="workflow-cuda-runtime",
            serial_m1_policy_hash="sha256:" + "0" * 64,
            raw_timing_sidecar_retained=False,
        )

    def validate(self) -> None:
        """Reject incomplete or placeholder provenance for promotion runs."""

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
                self.run_id, self.git_revision, self.build_id, self.compiler_id,
                self.architecture_class, self.device_name, self.driver_runtime,
            )
            if any("workflow" in item.lower() or "unknown" in item.lower()
                   for item in markers):
                raise ValueError("installable CUDA evidence has placeholder provenance")
            if not self.raw_timing_sidecar_retained:
                raise ValueError("installable CUDA evidence requires timing sidecars")


@lru_cache(maxsize=None)
def _trial_set_hash_values(
    source_format: str,
    shape: str,
    m: int,
    n: int,
    k: int,
) -> str:
    """Hash one reused deterministic trial identity exactly once."""

    return _sha256({
        "version": CUDA_DECODE_TRIAL_SET_VERSION,
        "source_format": source_format,
        "shape": shape,
        "m": m,
        "n": n,
        "k": k,
    })


def _trial_set_hash(raw: Mapping[str, str]) -> str:
    """Identify the deterministic input and packed-weight fixture."""

    return _trial_set_hash_values(
        raw["source_format"].strip().upper(),
        raw["shape"].strip(),
        int(raw["m"]),
        int(raw["n"]),
        int(raw["k"]),
    )


def _raw_candidate_config(raw: Mapping[str, str]) -> dict[str, object]:
    """Read the requested launcher configuration from one aggregate row."""

    family = raw["family"].strip().lower()
    if family == "inherit_serial_m1":
        return {"family": family}
    return {
        "family": family,
        "tile_n": int(raw["tile_n"]),
        "cpt": int(raw["cpt"]),
        "target_waves": int(raw["target_waves"]),
        "min_kgroups_per_cta": int(raw["mkg"]),
        "max_kb": int(raw["max_kb"]),
        "exact_kb": int(raw["exact_kb"]),
        "force_two_phase": int(raw["force_two_phase"]),
    }


def adapt_cuda_decode_row(
    raw: Mapping[str, str],
    context: CUDADecodeAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
) -> NativeVNNIObservation:
    """Convert one strong CUDA route observation into the common schema."""

    if raw.get("backend", "").strip().lower() != "cuda" or (
        raw.get("phase", "").strip() != "decode"
    ):
        raise ValueError("CUDA decode adapter received the wrong surface")

    m = int(raw["m"])
    n = int(raw["n"])
    k = int(raw["k"])
    contract = _semantic_contract(m)
    execution_mode = _execution_mode(raw["execution_mode"])
    measurement_order = int(raw["measurement_order"])
    measurement_order_seed = int(raw["measurement_order_seed"])
    if raw["measurement_protocol"].strip() != CUDA_DECODE_MEASUREMENT_PROTOCOL:
        raise ValueError(
            "CUDA broad evidence must use sample-interleaved measurement"
        )
    if measurement_order < 0 or measurement_order_seed <= 0:
        raise ValueError(
            "CUDA measurement order must be non-negative with a positive seed"
        )
    registry = cuda_native_vnni_gemv_registry()
    candidate = registry.resolve(raw["candidate_id"])
    if not candidate.supports_contract(contract):
        raise ValueError(
            f"{candidate.candidate_id} does not support {contract.value}"
        )
    if _raw_candidate_config(raw) != candidate.config_json:
        raise ValueError("CUDA requested configuration disagrees with registry")

    source_format = raw["source_format"].strip().upper()
    spec = format_spec(source_format)
    if int(raw["source_codebook"]) != spec.source_codebook_id:
        raise ValueError(f"{source_format}: source codebook disagrees with registry")
    runtime_codebook = spec.runtime_codebook("cuda")
    if int(raw["execution_codebook"]) != runtime_codebook:
        raise ValueError(f"{source_format}: execution codebook disagrees with registry")

    supported = _parse_bool("supported", raw["supported"])
    route_counter_ok = _parse_bool("route_counter_ok", raw["route_counter_ok"])
    serial_route_ok = _parse_bool(
        "serial_route_counter_ok", raw["serial_route_counter_ok"]
    )
    observed_raw = _required("observed_candidate_id", raw["observed_candidate_id"])
    try:
        observed_id = registry.resolve(observed_raw).effective_candidate_id
    except ValueError:
        observed_id = f"unresolved:{observed_raw}"
    serial_candidate = registry.resolve(raw["serial_m1_candidate_id"])
    if not serial_candidate.supports_contract(SemanticContract.FAST):
        raise ValueError("serial_m1_candidate_id is not a public-M1 candidate")

    observed_path = _required("observed_path", raw["observed_path"]).lower()
    observed_tile_n = int(raw["observed_tile_n"])
    observed_cpt = int(raw["observed_cpt"])
    observed_kb = int(raw["observed_effective_kb"])
    serial_config = serial_candidate.config_json
    inherited_route_ok = (
        contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        or (
            observed_path == serial_config["family"]
            and observed_tile_n == int(serial_config["tile_n"])
            and observed_cpt == int(serial_config["cpt"])
            and (
                observed_path != "kpar"
                or observed_kb == int(serial_config["exact_kb"])
            )
        )
    )
    requested_config = candidate.config_json
    requested_kb_ok = (
        contract != SemanticContract.FAST
        or observed_path != "kpar"
        or observed_kb == int(requested_config["exact_kb"])
    )
    forced_route_ok = (
        supported
        and route_counter_ok
        and serial_route_ok
        and observed_id == candidate.effective_candidate_id
        and inherited_route_ok
        and requested_kb_ok
    )

    mismatch_count = int(raw["bit_mismatches"])
    repeat_mismatches = int(raw["repeat_byte_mismatches"])
    grouped_digest = _required("grouped_output_digest", raw["grouped_output_digest"])
    serial_digest = _required("serial_output_digest", raw["serial_output_digest"])
    if (mismatch_count == 0) != (grouped_digest == serial_digest):
        raise ValueError("CUDA mismatch count disagrees with output digests")
    diagnostics = tuple(float(raw[name]) for name in (
        "max_abs", "relative_l2", "cosine", "symmetric_kld"
    ))
    if not all(math.isfinite(value) for value in diagnostics):
        raise ValueError("CUDA diagnostics must be finite")
    numerical_correctness = _parse_bool(
        "numerical_correctness", raw["numerical_correctness"]
    )
    expected_pass = (
        forced_route_ok
        and repeat_mismatches == 0
        and numerical_correctness
        and (
            contract == SemanticContract.FAST
            or mismatch_count == 0
        )
    )
    if _parse_bool("correctness_pass", raw["correctness_pass"]) != expected_pass:
        raise ValueError("CUDA correctness_pass disagrees with strong evidence")

    warmups = int(raw["warmup_count"])
    samples = int(raw["sample_count"])
    if context.profile.installable and supported and (
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
        raise ValueError("installable CUDA corpus is missing raw timing samples")

    graph_capture_ok = _parse_bool("graph_capture_ok", raw["graph_capture_ok"])
    workspace_ok = _parse_bool("workspace_ok", raw["workspace_ok"])
    explicit_stream_ok = _parse_bool(
        "explicit_stream_ok", raw["explicit_stream_ok"]
    )
    observation = NativeVNNIObservation(
        schema_version=SCHEMA_VERSION,
        run_id=context.run_id,
        corpus_id=context.corpus_id,
        git_revision=context.git_revision,
        build_id=context.build_id,
        compiler_id=context.compiler_id,
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        backend=Backend.CUDA,
        architecture_class=context.architecture_class,
        device_name=context.device_name,
        driver_runtime=context.driver_runtime,
        threading_or_stream_mode=context.threading_or_stream_mode,
        semantic_contract=contract,
        operation_kind="NativeVNNIDecodeProjection",
        bundle_signature="single-native-vnni-projection:fp32-output:v2",
        projection_n_vector=(n,),
        source_format=source_format,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cuda"),
        packing_abi=spec.packing_abi("cuda"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=f"cuda-decode:{raw['shape']}:n{n}:k{k}",
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
        alignment_class="cuda_gpu_prepared_native_vnni_16b_aligned",
        candidate_id=candidate.candidate_id,
        effective_candidate_id=candidate.effective_candidate_id,
        candidate_family=candidate.candidate_family,
        config_json=candidate.config_json,
        supported=supported,
        graph_capture_ok=graph_capture_ok,
        generic_eligible=True,
        arithmetic_fingerprint=candidate.arithmetic_fingerprint,
        serial_m1_policy_id=CUDA_DECODE_SERIAL_POLICY_ID,
        serial_m1_policy_hash=context.serial_m1_policy_hash,
        candidate_policy_hash=candidate.candidate_policy_hash(),
        ordered_reduction=candidate.ordered_reduction,
        uses_atomic_reduction=candidate.uses_atomic_reduction,
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
        workspace_ok=workspace_ok,
        explicit_stream_ok=explicit_stream_ok,
    )
    observation.validate()
    return observation


def adapt_cuda_decode_csv(
    paths: Iterable[Path],
    context: CUDADecodeAdapterContext,
    *,
    timing_sidecars: Iterable[Path] = (),
) -> ObservationCorpus:
    """Read strong CUDA shards and return one validated common corpus."""

    context.validate()
    timing_index = read_cuda_decode_timing_sidecars(timing_sidecars)
    if context.profile.installable and not timing_index:
        raise ValueError("installable CUDA corpus is missing timing sidecars")
    observations = []
    measurement_groups: dict[
        MeasurementGroupKey, list[tuple[int, int, str]]
    ] = defaultdict(list)
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing strong CUDA decode columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    measurement_groups[_measurement_group_key(raw)].append((
                        int(raw["measurement_order"]),
                        int(raw["measurement_order_seed"]),
                        raw["candidate_id"].strip().lower(),
                    ))
                    samples = timing_index.pop(_timing_key(raw), None)
                    observations.append(adapt_cuda_decode_row(raw, context, samples))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not observations:
        raise ValueError("CUDA decode trainer inputs contained no observations")
    _validate_measurement_orders(measurement_groups)
    if timing_index:
        first = next(iter(timing_index))
        raise ValueError(f"CUDA timing sidecar has no aggregate row for {first}")
    return ObservationCorpus(observations)
