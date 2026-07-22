"""CPU NativeVNNI grouped-verifier trainer adapter.

CPU candidates are explicit production policies, not inferred labels.  The
adapter keeps normalized/unsupported launches in the corpus, verifies every
raw steady-clock sample, and admits a candidate only when its effective route,
repeat bytes, and complete FP32 output match serial M=1 decode.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import raw_corpus_id, verify_aggregate_timing
from ..candidate_registry import cpu_native_vnni_verifier_registry
from ..candidate_observation import (
    _physical_core_count,
    write_observation_csv,
)
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


CPU_VERIFIER_TRIAL_SET_VERSION = "cpu-verifier-deterministic-q8-rows-v3"
CPU_SERIAL_POLICY_ID = "cpu.nvnni.production.serial-m1-v2"

REQUIRED_RAW_COLUMNS = frozenset({
    "backend", "phase", "source_format", "source_codebook",
    "execution_codebook", "shape", "execution_mode", "m", "n", "k",
    "candidate_id", "build_isa", "runtime_isa_requested",
    "runtime_isa_effective", "threads", "weight_bytes", "warmup_count",
    "sample_count", "min_us", "median_us", "p95_us", "mad_us", "cv",
    "serial_median_us", "speedup", "bit_mismatches", "first_bit_mismatch",
    "repeat_byte_mismatches", "max_abs", "relative_l2", "cosine",
    "symmetric_kld", "grouped_output_digest", "serial_output_digest",
    "timing_sample_digest", "route_counter_ok", "observed_candidate_id",
    "k_tiles", "n_block_chunks", "numerical_correctness",
    "correctness_pass", "is_winner",
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


_PARALLEL_CPU_VERIFIER_RECORDS: tuple[
    tuple[Path, int, Mapping[str, str], tuple[float, ...] | None], ...
] = ()
_PARALLEL_CPU_VERIFIER_CONTEXT: CPUVerifierAdapterContext | None = None


def _sha256(value: object) -> str:
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


def _cpu_isa(name: str, value: str, *, allow_auto: bool = False) -> str:
    """Normalize one declared CPU ISA and reject non-policy regimes."""

    normalized = _required(name, value).upper()
    allowed = {"AVX2", "AVX512"}
    if allow_auto:
        allowed.add("AUTO")
    if normalized not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}, got {value!r}")
    return normalized


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


def _read_cpu_verifier_timing_sidecars_serial(
    paths: Iterable[Path],
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact steady-clock samples in the calling process."""

    indexed: dict[RawTimingKey, list[float]] = {}
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_TIMING_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing CPU verifier timing columns {sorted(missing)}"
                )
            for row_number, raw in enumerate(reader, start=2):
                if raw["backend"].strip().lower() != "cpu" or (
                    raw["phase"].strip() != "verifier_rows"
                ):
                    raise ValueError(f"{path}:{row_number}: wrong timing sidecar surface")
                if raw["execution_mode"].strip().lower() != "eager":
                    raise ValueError(f"{path}:{row_number}: CPU execution mode must be eager")
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
                        f"{path}:{row_number}: readable and exact latency fields disagree"
                    )
                samples.append(latency_us)

    result = {}
    for key, samples in indexed.items():
        values = tuple(samples)
        if tuple(sorted(values)) != values:
            raise ValueError(f"timing sidecar samples are not trainer-sorted for {key}")
        result[key] = values
    return result


def _read_cpu_verifier_timing_range(
    task: tuple[Path, str, int, int, int, Path],
) -> Path:
    """Parse one byte-disjoint timing range into a compact pickle shard."""

    path, header, data_start, start, stop, output = task
    indexed: dict[RawTimingKey, tuple[int, list[float]]] = {}
    with path.open("rb") as handle:
        handle.seek(start)
        if start > data_start:
            handle.seek(start - 1)
            if handle.read(1) != b"\n":
                handle.readline()

        def selected_lines():
            """Yield complete CSV records whose first byte belongs here."""

            yield header
            while handle.tell() < stop:
                encoded = handle.readline()
                if not encoded:
                    break
                yield encoded.decode("utf-8")

        reader = csv.DictReader(selected_lines())
        for local_row, raw in enumerate(reader, start=1):
            try:
                if raw["backend"].strip().lower() != "cpu" or (
                    raw["phase"].strip() != "verifier_rows"
                ):
                    raise ValueError("wrong timing sidecar surface")
                if raw["execution_mode"].strip().lower() != "eager":
                    raise ValueError("CPU execution mode must be eager")
                key = _timing_key(raw)
                sample_index = int(raw["sample_index"])
                latency_us = float.fromhex(raw["latency_us_hex"].strip())
                readable_us = float(raw["latency_us"])
                if latency_us <= 0.0 or not math.isfinite(latency_us):
                    raise ValueError("invalid raw latency")
                if not math.isclose(
                    readable_us,
                    latency_us,
                    rel_tol=0.0,
                    abs_tol=5.1e-10,
                ):
                    raise ValueError(
                        "readable and exact latency fields disagree"
                    )
                first_index, samples = indexed.setdefault(
                    key, (sample_index, [])
                )
                if sample_index != first_index + len(samples):
                    raise ValueError(
                        f"timing sample index {sample_index} is not the next "
                        f"contiguous index {first_index + len(samples)}"
                    )
                samples.append(latency_us)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:byte-range={start}:{stop}:"
                    f"row={local_row}: {exc}"
                ) from exc

    compact = {
        key: (first_index, tuple(samples))
        for key, (first_index, samples) in indexed.items()
    }
    with output.open("wb") as handle:
        pickle.dump(compact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output


def read_cpu_verifier_timing_sidecars(
    paths: Iterable[Path],
    *,
    workers: int | None = None,
    parallel_threshold_bytes: int = 64 * 1024 * 1024,
) -> dict[RawTimingKey, tuple[float, ...]]:
    """Read exact timing samples using deterministic physical-core shards.

    Grouped-verifier sidecars are the same flat, independently parseable CSV
    evidence as M=1 decode sidecars. Large files are divided at byte ranges;
    each worker advances to a complete record and writes a private compact
    shard. The parent merges file/range order and revalidates global sample
    continuity, so worker scheduling cannot alter the authenticated corpus.
    """

    paths = tuple(Path(item) for item in paths)
    if not paths:
        return {}
    if workers is None:
        workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_IO_WORKERS",
            str(_physical_core_count()),
        ))
    if workers < 1:
        raise ValueError("CPU verifier timing worker count must be positive")
    workers = min(workers, _physical_core_count())
    total_bytes = sum(path.stat().st_size for path in paths)
    if workers <= 1 or total_bytes < parallel_threshold_bytes:
        return _read_cpu_verifier_timing_sidecars_serial(paths)

    indexed: dict[RawTimingKey, list[float]] = {}
    with tempfile.TemporaryDirectory(
        prefix="native-vnni-cpu-verifier-timing-",
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        tasks = []
        target_chunk_bytes = max(1, parallel_threshold_bytes)
        for path_index, path in enumerate(paths):
            with path.open("rb") as handle:
                encoded_header = handle.readline()
                data_start = handle.tell()
            header = encoded_header.decode("utf-8")
            fieldnames = next(csv.reader((header,)), ())
            missing = REQUIRED_TIMING_COLUMNS.difference(fieldnames)
            if missing:
                raise ValueError(
                    f"{path}: missing CPU verifier timing columns "
                    f"{sorted(missing)}"
                )
            file_size = path.stat().st_size
            data_bytes = max(0, file_size - data_start)
            path_workers = min(
                workers,
                max(
                    1,
                    (data_bytes + target_chunk_bytes - 1)
                    // target_chunk_bytes,
                ),
            )
            for range_index in range(path_workers):
                start = data_start + data_bytes * range_index // path_workers
                stop = (
                    data_start
                    + data_bytes * (range_index + 1) // path_workers
                )
                tasks.append((
                    path,
                    header,
                    data_start,
                    start,
                    stop,
                    temporary_root
                    / f"timing-{path_index:04d}-{range_index:04d}.pickle",
                ))

        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            shards = tuple(executor.map(
                _read_cpu_verifier_timing_range,
                tasks,
            ))

        for shard in shards:
            with shard.open("rb") as handle:
                compact = pickle.load(handle)
            for key, (first_index, values) in compact.items():
                samples = indexed.setdefault(key, [])
                if first_index != len(samples):
                    raise ValueError(
                        f"timing sample index {first_index} is not the next "
                        f"contiguous index {len(samples)} for {key}"
                    )
                samples.extend(values)

    result = {}
    for key, samples in indexed.items():
        values = tuple(samples)
        if tuple(sorted(values)) != values:
            raise ValueError(
                f"timing sidecar samples are not trainer-sorted for {key}"
            )
        result[key] = values
    return result


@dataclass(frozen=True)
class CPUVerifierAdapterContext:
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
    minimum_promotion_warmups: int = MIN_PROMOTION_WARMUPS
    minimum_promotion_samples: int = MIN_PROMOTION_SAMPLES

    @classmethod
    def workflow_smoke(
        cls,
        *,
        corpus_id: str,
        minimum_promotion_warmups: int = MIN_PROMOTION_WARMUPS,
        minimum_promotion_samples: int = MIN_PROMOTION_SAMPLES,
    ) -> "CPUVerifierAdapterContext":
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
            minimum_promotion_warmups=minimum_promotion_warmups,
            minimum_promotion_samples=minimum_promotion_samples,
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
        for name in (
            "minimum_promotion_warmups",
            "minimum_promotion_samples",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.profile.installable:
            markers = (
                self.run_id, self.git_revision, self.build_id, self.compiler_id,
                self.architecture_class, self.device_name, self.driver_runtime,
            )
            if any("workflow" in item.lower() or "unknown" in item.lower()
                   for item in markers):
                raise ValueError("installable CPU verifier evidence has placeholder provenance")
            if not self.raw_timing_sidecar_retained:
                raise ValueError("installable CPU verifier evidence needs raw timing sidecars")

    def validate_promotion_timing(
        self,
        warmups: int,
        samples: int,
        *,
        forced_route_ok: bool,
    ) -> None:
        """Enforce the transaction's explicit installable timing floor."""

        if self.profile.installable and forced_route_ok and (
            warmups < self.minimum_promotion_warmups
            or samples < self.minimum_promotion_samples
        ):
            raise ValueError(
                "installable profile requires "
                f"{self.minimum_promotion_warmups}/"
                f"{self.minimum_promotion_samples} timing, got "
                f"{warmups}/{samples}"
            )


@lru_cache(maxsize=None)
def _trial_set_hash(
    source_format: str,
    shape: str,
    m: int,
    n: int,
    k: int,
    build_isa: str,
    runtime_isa: str,
    threads: int,
) -> str:
    """Hash one reused CPU tensor/ISA/thread trial identity once."""

    return _sha256({
        "version": CPU_VERIFIER_TRIAL_SET_VERSION,
        "source_format": source_format,
        "shape": shape,
        "m": m,
        "n": n,
        "k": k,
        "build_isa": build_isa,
        "runtime_isa": runtime_isa,
        "threads": threads,
    })


def adapt_cpu_verifier_row(
    raw: Mapping[str, str],
    context: CPUVerifierAdapterContext,
    timing_samples: tuple[float, ...] | None = None,
) -> NativeVNNIObservation:
    """Convert one strong CPU candidate row into the common observation."""

    if raw.get("backend", "").strip().lower() != "cpu" or (
        raw.get("phase", "").strip() != "verifier_rows"
    ):
        raise ValueError("CPU verifier adapter received the wrong surface")
    if raw["execution_mode"].strip().lower() != "eager":
        raise ValueError("CPU verifier observations must use eager execution")

    m = int(raw["m"])
    n = int(raw["n"])
    k = int(raw["k"])
    if m < 2:
        raise ValueError(f"CPU grouped verifier M must be at least 2; got {m}")
    registry = cpu_native_vnni_verifier_registry()
    candidate = registry.resolve(raw["candidate_id"])
    if not candidate.supports_contract(
        SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    ):
        raise ValueError(f"{candidate.candidate_id} lacks verifier semantics")

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
    grouped_digest = _required("grouped_output_digest", raw["grouped_output_digest"])
    serial_digest = _required("serial_output_digest", raw["serial_output_digest"])
    if (mismatch_count == 0) != (grouped_digest == serial_digest):
        raise ValueError("CPU mismatch count disagrees with output digests")
    diagnostics = tuple(float(raw[name]) for name in (
        "max_abs", "relative_l2", "cosine", "symmetric_kld"
    ))
    if not all(math.isfinite(value) for value in diagnostics):
        raise ValueError("CPU verifier diagnostics must be finite")
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
        raise ValueError("CPU correctness_pass disagrees with strong evidence")

    warmups = int(raw["warmup_count"])
    samples = int(raw["sample_count"])
    context.validate_promotion_timing(
        warmups,
        samples,
        forced_route_ok=forced_route_ok,
    )
    if timing_samples is not None:
        if len(timing_samples) != samples:
            raise ValueError(
                f"raw timing sidecar has {len(timing_samples)} samples; expected {samples}"
            )
        verify_aggregate_timing(raw, timing_samples, median_field="median_us")
    elif context.profile.installable:
        raise ValueError("installable CPU corpus is missing raw timing samples")

    threads = int(raw["threads"])
    if threads <= 0:
        raise ValueError("threads must be positive")
    build_isa = _cpu_isa("build_isa", raw["build_isa"])
    requested_runtime_isa = _cpu_isa(
        "runtime_isa_requested",
        raw["runtime_isa_requested"],
        allow_auto=True,
    )
    effective_runtime_isa = _cpu_isa(
        "runtime_isa_effective", raw["runtime_isa_effective"]
    )
    if requested_runtime_isa != "AUTO" and (
        requested_runtime_isa != effective_runtime_isa
    ):
        raise ValueError("requested CPU runtime ISA did not become effective")
    if build_isa == "AVX2" and effective_runtime_isa != "AVX2":
        raise ValueError("an AVX2-only build cannot execute an AVX512 runtime path")
    k_tiles = int(raw["k_tiles"])
    n_block_chunks = int(raw["n_block_chunks"])
    if k_tiles < 0 or n_block_chunks <= 0:
        raise ValueError("CPU verifier launch geometry is invalid")
    if forced_route_ok and candidate.config_json["policy"] == "WideRows" and (
        effective_runtime_isa == "AVX2" or m == 2
    ):
        raise ValueError(
            "CPU telemetry claimed an impossible effective WideRows route"
        )
    if (
        forced_route_ok
        and candidate.config_json.get("k_tile_policy") == "full_k"
        and k_tiles > 1
    ):
        raise ValueError(
            "CPU telemetry claimed a full-K verifier route in a K-partition domain"
        )
    expected_n_block_chunks = candidate.config_json.get("n_block_chunks")
    if (
        forced_route_ok
        and expected_n_block_chunks is not None
        and n_block_chunks != int(expected_n_block_chunks)
    ):
        raise ValueError(
            "CPU verifier telemetry disagrees with the candidate N-block width"
        )
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
        semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
        operation_kind="NativeVNNIDecodeProjection",
        bundle_signature="single-native-vnni-projection:fp32-output:v2",
        projection_n_vector=(n,),
        source_format=source_format,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("cpu"),
        packing_abi=spec.packing_abi("cpu"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=f"cpu-verifier:{raw['shape']}:n{n}:k{k}",
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
            source_format,
            raw["shape"],
            m,
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
        launch_k_tiles=k_tiles,
        launch_n_block_chunks=n_block_chunks,
    )
    observation.validate()
    return observation


def _read_cpu_verifier_raw_records(
    paths: Iterable[Path],
) -> tuple[tuple[Path, int, Mapping[str, str]], ...]:
    """Read aggregate rows once while retaining precise diagnostics."""

    records = []
    for path in (Path(item) for item in paths):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_RAW_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"{path}: missing strong CPU verifier columns "
                    f"{sorted(missing)}"
                )
            records.extend(
                (path, row_number, raw)
                for row_number, raw in enumerate(reader, start=2)
            )
    return tuple(records)


def _adapt_cpu_verifier_range(
    task: tuple[int, int, Path],
) -> tuple[Path, Path, int]:
    """Adapt one range into canonical CSV and private object shards."""

    if _PARALLEL_CPU_VERIFIER_CONTEXT is None:
        raise RuntimeError(
            "parallel CPU verifier adapter context is unavailable"
        )
    start, stop, output = task
    observations = []
    for index in range(start, stop):
        path, row_number, raw, samples = _PARALLEL_CPU_VERIFIER_RECORDS[index]
        try:
            observations.append(adapt_cpu_verifier_row(
                raw,
                _PARALLEL_CPU_VERIFIER_CONTEXT,
                samples,
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_number}: {exc}") from exc
    immutable_observations = tuple(observations)
    write_observation_csv(output, immutable_observations, workers=1)
    object_output = output.with_suffix(".pickle")
    with object_output.open("wb") as handle:
        pickle.dump(
            immutable_observations,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return output, object_output, len(immutable_observations)


def _read_cpu_verifier_object_shards(
    results: Iterable[tuple[Path, Path, int]],
) -> Iterable[NativeVNNIObservation]:
    """Yield trusted worker results without reparsing the published CSV.

    Object shards exist only inside the adapter-owned temporary directory and
    are never durable or accepted as evidence. Each row was already fully
    validated by ``adapt_cpu_verifier_row``; the parent checks transfer shape
    and types before rebuilding canonical corpus indices.
    """

    for _csv_path, object_path, expected_count in results:
        with object_path.open("rb") as handle:
            observations = pickle.load(handle)
        if not isinstance(observations, tuple):
            raise RuntimeError("CPU verifier adapter object shard is not a tuple")
        if len(observations) != expected_count:
            raise RuntimeError("CPU verifier adapter object shard count changed")
        if not all(
            isinstance(observation, NativeVNNIObservation)
            for observation in observations
        ):
            raise RuntimeError("CPU verifier adapter object shard has invalid rows")
        yield from observations


def adapt_cpu_verifier_csv_to_common(
    paths: Iterable[Path],
    context: CPUVerifierAdapterContext,
    output: Path,
    *,
    timing_sidecars: Iterable[Path] = (),
    workers: int | None = None,
    parallel_threshold: int = 4096,
) -> ObservationCorpus:
    """Adapt grouped evidence in workers and atomically publish common CSV.

    Fork workers inherit the immutable aggregate rows, exact timing tuples,
    and provenance. Each worker writes a canonical CSV shard for a contiguous
    input range plus a private object-transfer shard and returns only their
    paths and row count. The parent concatenates canonical CSV in source order
    and indexes the validated objects directly, avoiding giant process results,
    duplicate CSV parsing, and any change to durable common evidence bytes.
    """

    global _PARALLEL_CPU_VERIFIER_CONTEXT
    global _PARALLEL_CPU_VERIFIER_RECORDS

    context.validate()
    timing_index = read_cpu_verifier_timing_sidecars(
        timing_sidecars,
        workers=workers,
    )
    if context.profile.installable and not timing_index:
        raise ValueError("installable CPU corpus is missing timing sidecars")
    raw_records = _read_cpu_verifier_raw_records(paths)
    if not raw_records:
        raise ValueError("CPU verifier inputs contained no observations")

    available_timing = dict(timing_index)
    records = []
    for path, row_number, raw in raw_records:
        try:
            samples = available_timing.pop(_timing_key(raw), None)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_number}: {exc}") from exc
        records.append((path, row_number, raw, samples))
    if available_timing:
        first = next(iter(available_timing))
        raise ValueError(
            f"CPU timing sidecar has no aggregate row for {first}"
        )

    if workers is None:
        workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_IO_WORKERS",
            str(_physical_core_count()),
        ))
    if workers < 1:
        raise ValueError("CPU verifier adapter worker count must be positive")
    worker_count = min(
        workers,
        _physical_core_count(),
        len(records),
    )
    if worker_count <= 1 or len(records) < parallel_threshold:
        corpus = ObservationCorpus(
            adapt_cpu_verifier_row(raw, context, samples)
            for _path, _row_number, raw, samples in records
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(output, corpus, workers=1)
        return corpus

    output.parent.mkdir(parents=True, exist_ok=True)
    rows_per_worker = (len(records) + worker_count - 1) // worker_count
    _PARALLEL_CPU_VERIFIER_CONTEXT = context
    _PARALLEL_CPU_VERIFIER_RECORDS = tuple(records)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{output.name}.adapt-",
            dir=output.parent,
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            tasks = tuple(
                (
                    start,
                    min(start + rows_per_worker, len(records)),
                    temporary_root / f"part-{part:04d}.csv",
                )
                for part, start in enumerate(
                    range(0, len(records), rows_per_worker)
                )
            )
            with ProcessPoolExecutor(
                max_workers=len(tasks),
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                results = tuple(executor.map(
                    _adapt_cpu_verifier_range,
                    tasks,
                ))
            if sum(count for _csv, _objects, count in results) != len(records):
                raise RuntimeError("parallel CPU verifier adapter lost rows")

            staged = temporary_root / "complete.csv"
            with staged.open("wb") as destination:
                for part, (shard, _objects, _count) in enumerate(results):
                    with shard.open("rb") as source:
                        if part:
                            source.readline()
                        shutil.copyfileobj(
                            source,
                            destination,
                            length=1024 * 1024,
                        )
            corpus = ObservationCorpus._from_validated(
                _read_cpu_verifier_object_shards(results)
            )
            os.replace(staged, output)
    finally:
        _PARALLEL_CPU_VERIFIER_CONTEXT = None
        _PARALLEL_CPU_VERIFIER_RECORDS = ()

    return corpus


def adapt_cpu_verifier_csv(
    paths: Iterable[Path],
    context: CPUVerifierAdapterContext,
    *,
    timing_sidecars: Iterable[Path] = (),
) -> ObservationCorpus:
    """Read strong CPU shards and return one validated common corpus."""

    context.validate()
    timing_index = read_cpu_verifier_timing_sidecars(timing_sidecars)
    if context.profile.installable and not timing_index:
        raise ValueError("installable CPU corpus is missing timing sidecars")
    observations = []
    for path, row_number, raw in _read_cpu_verifier_raw_records(paths):
        try:
            samples = timing_index.pop(_timing_key(raw), None)
            observations.append(adapt_cpu_verifier_row(raw, context, samples))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{row_number}: {exc}") from exc
    if not observations:
        raise ValueError("CPU verifier inputs contained no observations")
    if timing_index:
        first = next(iter(timing_index))
        raise ValueError(f"CPU timing sidecar has no aggregate row for {first}")
    return ObservationCorpus(observations)
