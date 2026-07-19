"""Profiler-informed candidate regret modeling for NativeVNNI policy fitting.

Hardware counters are development-only explanatory features. They are joined
to immutable canonical timing observations through authenticated profiler
request/evidence manifests, summarized per physical launch candidate, and used
to train a deterministic nonlinear surrogate inside each grouped CV fold.
Measured latency remains authoritative for exact winners, the hard
five-percent topology, held-out CV, and sealed promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Protocol

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction import DictVectorizer

from .corpus import ObservationCorpus, RuntimeKey
from .format_registry import format_spec
from .profiler_evidence import (
    ProfilerFeatureRow,
    profiler_feature_rows,
    read_profiler_evidence_manifest,
    read_profiler_request_manifest,
)
from .schema import Backend, ExecutionMode, NativeVNNIObservation


PROFILER_FEATURE_CATALOG_CACHE_VERSION = "native-vnni-profiler-catalog-cache-v1"


def _file_sha256(path: Path) -> str:
    """Hash one canonical source artifact without loading it into Python memory."""

    with path.open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def _cache_payload_digest(payload: Mapping[str, object]) -> str:
    """Return the canonical identity of one derived catalog-cache payload."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


PROFILER_MODEL_VERSION = (
    "native-vnni-profiler-extra-trees-v13-cpu-decode-launch-geometry"
)
PROFILER_EXTRA_TREES = 64
PROFILER_EXTRA_TREES_MAX_DEPTH = 10
PROFILER_EXTRA_TREES_MIN_SAMPLES_LEAF = 3
PROFILER_EXTRA_TREES_RANDOM_STATE = 19348
PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT = 0.5

_GPU_STATIC_RESOURCE_METRICS = frozenset((
    "gpu.registers_per_thread",
    "gpu.static_shared_memory_bytes",
    "gpu.dynamic_shared_memory_bytes",
    "gpu.local_memory_bytes_per_thread",
    "gpu.vgpr_count",
    "gpu.sgpr_count",
    "gpu.lds_bytes",
    "gpu.scratch_bytes",
))
_GPU_PERCENT_METRICS = frozenset((
    "gpu.theoretical_occupancy_pct",
    "gpu.achieved_occupancy_pct",
    "gpu.compute_throughput_pct_of_peak",
    "gpu.dram_throughput_pct_of_peak",
    "gpu.l1_throughput_pct_of_peak",
    "gpu.l2_throughput_pct_of_peak",
    "gpu.gpu_busy_pct",
    "gpu.valu_busy_pct",
    "gpu.valu_utilization_pct",
    "gpu.memory_unit_busy_pct",
    "gpu.memory_unit_stalled_pct",
    "gpu.l2_cache_hit_pct",
    "gpu.lds_bank_conflict_pct",
))
_GPU_RATE_METRICS = frozenset((
    "gpu.executed_ipc_active",
    "gpu.warp_cycles_per_issued_instruction",
))


class CandidateCostLike(Protocol):
    """Structural interface consumed without importing the policy module."""

    runtime_key: RuntimeKey
    shape_group_id: str
    candidate_id: str
    max_surface_regret: float
    profiler_predicted_regret: float


@dataclass(frozen=True, order=True)
class PhysicalCandidateKey:
    """Complete identity of one profiled physical kernel invocation.

    Scheduling candidate IDs are intentionally not sufficient. NativeVNNI
    candidates with the same row tiling can dispatch LUT-decoded, direct INT8,
    or another prepared codebook microkernel. Reusing counters across those
    paths would inject another format's instruction and cache behavior into the
    fit. Dynamic counters also vary with execution mode and every dimension of
    the submitted work, so ``M``, the projection vector, aggregate ``N``, and
    ``K`` are part of the key rather than metadata on a reusable descriptor.
    """

    backend: Backend
    architecture_class: str
    operation_kind: str
    bundle_signature: str
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    effective_candidate_id: str
    execution_mode: ExecutionMode
    m: int
    projection_n_vector: tuple[int, ...]
    aggregate_n: int
    k: int

    def canonical_tuple(self) -> tuple[str, ...]:
        """Return the stable JSON/sort representation of this identity."""

        return (
            self.backend.value,
            self.architecture_class,
            self.operation_kind,
            self.bundle_signature,
            self.prepared_family_id,
            self.packing_abi,
            str(self.runtime_codebook_id),
            self.effective_candidate_id,
            self.execution_mode.value,
            str(self.m),
            "[" + ",".join(str(value) for value in self.projection_n_vector) + "]",
            str(self.aggregate_n),
            str(self.k),
        )
ProfilerObservationKey = tuple[RuntimeKey, str, str]
ProfilerObservationIndex = Mapping[
    ProfilerObservationKey, NativeVNNIObservation
]


def _physical_key(row: NativeVNNIObservation) -> PhysicalCandidateKey:
    """Return the request layer's exact physical launch identity."""

    return PhysicalCandidateKey(
        backend=row.backend,
        architecture_class=row.architecture_class,
        operation_kind=row.operation_kind,
        bundle_signature=row.bundle_signature,
        prepared_family_id=row.prepared_family_id,
        packing_abi=row.packing_abi,
        runtime_codebook_id=row.runtime_codebook_id,
        effective_candidate_id=row.effective_candidate_id,
        execution_mode=row.execution_mode,
        m=row.m,
        projection_n_vector=row.projection_n_vector,
        aggregate_n=row.aggregate_n,
        k=row.k,
    )


def _signed_log1p(value: float) -> float:
    """Compress counter magnitudes while preserving signed diagnostics."""

    return math.copysign(math.log1p(abs(value)), value)


def _flatten_config(
    prefix: str,
    value: object,
    output: dict[str, float | str],
) -> None:
    """Flatten typed candidate configuration into deterministic model fields."""

    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)):
        output[prefix] = _signed_log1p(float(value))
    elif isinstance(value, str):
        output[prefix] = value
    elif isinstance(value, Mapping):
        for name, item in sorted(value.items()):
            _flatten_config(f"{prefix}.{name}", item, output)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _flatten_config(f"{prefix}.{index}", item, output)
    elif value is not None:
        raise ValueError(f"unsupported profiler-model config value at {prefix}")


def _execution_regime(operation_kind: str, m: int) -> str:
    """Classify one launch by the performance question its counters answer.

    Operation identity takes precedence over row count because grouped MoE
    prefill and grouped MTP decode can both use small matrices. The bounded-M
    fallback covers older evidence whose operation name predates the explicit
    naming convention.
    """

    normalized = operation_kind.strip().lower()
    if "prefill" in normalized or "gemm" in normalized:
        return "prefill"
    if (
        "decode" in normalized
        or "gemv" in normalized
        or "verifier" in normalized
    ):
        return "decode"
    return "decode" if m <= 15 else "prefill"


def _sum_metric(
    metric_values: Mapping[str, list[float]],
    metric_id: str,
) -> float | None:
    """Return one finite dispatch sum or ``None`` when unsupported."""

    values = metric_values.get(metric_id, ())
    return sum(values) if values else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return one non-negative finite ratio only when both inputs are usable."""

    if (
        numerator is None
        or denominator is None
        or denominator <= 0.0
    ):
        return None
    result = numerator / denominator
    return result if math.isfinite(result) and result >= 0.0 else None


def _estimated_work(observation: NativeVNNIObservation) -> tuple[float, float, float]:
    """Return logical MACs, payload bytes, and output bytes for normalization.

    NativeVNNI payload sizes are specified per 32 weights. Activations and
    outputs use a conservative four-byte estimate; this is an explanatory
    geometry feature, not a byte-exact runtime accounting claim.
    """

    spec = format_spec(observation.source_format)
    logical_macs = float(
        observation.m * observation.aggregate_n * observation.k
    )
    weight_bytes = (
        float(observation.aggregate_n * observation.k)
        * float(spec.payload_bytes)
        / 32.0
    )
    activation_bytes = float(observation.m * observation.k * 4)
    output_bytes = float(observation.m * observation.aggregate_n * 4)
    return logical_macs, weight_bytes + activation_bytes + output_bytes, output_bytes


def _normalized_metric_features(
    observation: NativeVNNIObservation,
    metric_values: Mapping[str, list[float]],
) -> dict[str, float]:
    """Derive regime-aware rates without exposing raw absolute counters.

    Decode/GEMV retains both bytes-moved and logical-compute views. M=1 is
    predominantly bandwidth-bound, while grouped verifier rows reuse a weight
    payload and can become compute-bound as M rises. Prefill/GEMM is normalized
    by logical MACs. Static resource usage and already-normalized profiler
    percentages remain valid candidate attributes in either regime.
    """

    features: dict[str, float] = {}
    regime = _execution_regime(observation.operation_kind, observation.m)
    logical_macs, expected_bytes, output_bytes = _estimated_work(observation)

    cycles = _sum_metric(metric_values, "cpu.cycles")
    reference_cycles = _sum_metric(metric_values, "cpu.ref_cycles")
    instructions = _sum_metric(metric_values, "cpu.instructions")
    task_clock_ns = _sum_metric(metric_values, "cpu.task_clock_ns")
    wall_clock_ns = _sum_metric(metric_values, "cpu.wall_clock_ns")
    l1d_loads = _sum_metric(metric_values, "cpu.l1d_loads")
    l1d_load_misses = _sum_metric(metric_values, "cpu.l1d_load_misses")
    llc_loads = _sum_metric(metric_values, "cpu.llc_loads")
    llc_load_misses = _sum_metric(metric_values, "cpu.llc_load_misses")
    active_thread_concurrency = _ratio(task_clock_ns, wall_clock_ns)
    cpu_threads = _cpu_parallelism_width(observation.architecture_class)
    duration_features_reliable = (
        wall_clock_ns is not None
        and wall_clock_ns >= 500_000.0
        and active_thread_concurrency is not None
        and cpu_threads is not None
        and active_thread_concurrency <= float(cpu_threads) * 1.05
    )
    features["profile.cpu.duration_features_reliable"] = float(
        duration_features_reliable
    )
    cpu_ratios = {
        "branch_miss_fraction": _ratio(
            _sum_metric(metric_values, "cpu.branch_misses"),
            _sum_metric(metric_values, "cpu.branches"),
        ),
        "cache_miss_fraction": _ratio(
            _sum_metric(metric_values, "cpu.cache_misses"),
            _sum_metric(metric_values, "cpu.cache_references"),
        ),
        "l1d_load_miss_fraction": _ratio(
            l1d_load_misses,
            l1d_loads,
        ),
        "llc_load_miss_fraction": _ratio(
            llc_load_misses,
            llc_loads,
        ),
    }
    if duration_features_reliable:
        cpu_ratios.update({
            "ipc": _ratio(instructions, cycles),
            "core_to_reference_cycle_ratio": _ratio(
                cycles, reference_cycles
            ),
            "active_thread_concurrency": active_thread_concurrency,
        })
    for name, value in cpu_ratios.items():
        if value is not None:
            features[f"metric.cpu.{name}"] = value

    # Counter rates retain useful traffic information even when the matching
    # denominator event is unavailable in the simultaneous event group. In
    # particular, this keeps the already-paid LLC-miss evidence model-visible
    # on machines whose generic LLC-load event is not schedulable.
    normalized_counts = [
        ("instructions", instructions),
        ("l1d_loads", l1d_loads),
        ("l1d_load_misses", l1d_load_misses),
        ("llc_load_misses", llc_load_misses),
    ]
    if duration_features_reliable:
        normalized_counts.extend((
            ("cycles", cycles),
            ("reference_cycles", reference_cycles),
            ("task_clock_ns", task_clock_ns),
            ("wall_clock_ns", wall_clock_ns),
        ))
    normalization_units = (
        (
            ("per_expected_mib", expected_bytes / float(1 << 20)),
            ("per_million_macs", logical_macs / 1.0e6),
        )
        if regime == "decode"
        else (("per_million_macs", logical_macs / 1.0e6),)
    )
    for name, value in normalized_counts:
        for unit_name, work_units in normalization_units:
            normalized = _ratio(value, work_units)
            if normalized is not None:
                features[
                    f"metric.cpu.{regime}.{name}_{unit_name}_log1p"
                ] = _signed_log1p(normalized)

    for metric_id in sorted(_GPU_STATIC_RESOURCE_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            features[f"metric.{metric_id}.maximum_log1p"] = _signed_log1p(
                max(values)
            )
    for metric_id in sorted(_GPU_PERCENT_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            fractions = [value / 100.0 for value in values]
            features[f"metric.{metric_id}.fraction_mean"] = (
                sum(fractions) / len(fractions)
            )
            features[f"metric.{metric_id}.fraction_maximum"] = max(fractions)
    for metric_id in sorted(_GPU_RATE_METRICS):
        values = metric_values.get(metric_id, ())
        if values:
            transformed = [_signed_log1p(value) for value in values]
            features[f"metric.{metric_id}.mean_log1p"] = (
                sum(transformed) / len(transformed)
            )
            features[f"metric.{metric_id}.maximum_log1p"] = max(transformed)

    duration_ns = _sum_metric(metric_values, "gpu.duration_ns")
    if duration_ns is not None and duration_ns > 0.0:
        if regime == "decode":
            # One byte/ns is one decimal GB/s.
            features[
                "metric.gpu.decode.effective_gbytes_per_second_log1p"
            ] = (
                _signed_log1p(expected_bytes / duration_ns)
            )
            features[
                "metric.gpu.decode.duration_ns_per_expected_mib_log1p"
            ] = (
                _signed_log1p(duration_ns / (expected_bytes / float(1 << 20)))
            )
            # Grouped verifier rows reuse one weight payload while their MAC
            # count grows with M. Retaining the compute view beside bandwidth
            # lets one model describe the observed memory-to-compute transition
            # without misclassifying grouped decode as ordinary prefill.
            features["metric.gpu.decode.effective_gops_log1p"] = _signed_log1p(
                (2.0 * logical_macs) / duration_ns
            )
            features[
                "metric.gpu.decode.duration_ns_per_gmac_log1p"
            ] = _signed_log1p(duration_ns / (logical_macs / 1.0e9))
        else:
            # One operation/ns is one decimal GOP/s; each MAC contributes two
            # arithmetic operations to the conventional GEMM throughput claim.
            features["metric.gpu.prefill.effective_gops_log1p"] = _signed_log1p(
                (2.0 * logical_macs) / duration_ns
            )
            features[
                "metric.gpu.prefill.duration_ns_per_gmac_log1p"
            ] = _signed_log1p(duration_ns / (logical_macs / 1.0e9))

    spill_requests = _sum_metric(
        metric_values, "gpu.local_memory_spill_requests"
    )
    spills_per_million_outputs = _ratio(
        spill_requests,
        output_bytes / 4.0 / 1.0e6,
    )
    if spills_per_million_outputs is not None:
        features[
            "metric.gpu.spills_per_million_outputs_log1p"
        ] = _signed_log1p(spills_per_million_outputs)

    fetch_bytes = _sum_metric(metric_values, "gpu.fetch_kib")
    if fetch_bytes is not None:
        fetch_bytes *= 1024.0
        amplification = _ratio(fetch_bytes, expected_bytes)
        if amplification is not None:
            features["metric.gpu.fetch_to_expected_byte_ratio"] = amplification
        if duration_ns is not None and duration_ns > 0.0:
            features[
                "metric.gpu.observed_fetch_gbytes_per_second_log1p"
            ] = (
                _signed_log1p(fetch_bytes / duration_ns)
            )
    write_bytes = _sum_metric(metric_values, "gpu.write_kib")
    if write_bytes is not None:
        write_amplification = _ratio(write_bytes * 1024.0, output_bytes)
        if write_amplification is not None:
            features["metric.gpu.write_to_output_byte_ratio"] = (
                write_amplification
            )
    wavefronts = _sum_metric(metric_values, "gpu.wavefront_count")
    waves_per_million_outputs = _ratio(
        wavefronts,
        output_bytes / 4.0 / 1.0e6,
    )
    if waves_per_million_outputs is not None:
        features[
            "metric.gpu.wavefronts_per_million_outputs_log1p"
        ] = _signed_log1p(waves_per_million_outputs)
    return features


@dataclass(frozen=True)
class ProfilerCandidateDescriptor:
    """One authenticated exact physical invocation's profiler features.

    Candidate configuration and static resource use describe the physical
    kernel. Dynamic hardware counters describe that kernel *at the request
    geometry that produced them*. Retaining the anchor is therefore part of
    the evidence value: a coherent geometry holdout must not consume IPC,
    cache, throughput, or duration evidence measured at the geometry whose
    timing answer it is trying to predict.
    """

    key: PhysicalCandidateKey
    anchor_m: int
    anchor_n: int
    anchor_k: int
    features: Mapping[str, float | str]

    @property
    def anchor_geometry(self) -> tuple[int, int]:
        """Return the N/K geometry used by the isolated profiler launch."""

        return (self.anchor_n, self.anchor_k)


@dataclass(frozen=True)
class ProfilerFeatureCatalog:
    """Complete exact-launch descriptors bound to one timing corpus."""

    corpus_digest: str
    request_manifest_digest: str
    evidence_manifest_digest: str
    descriptors: Mapping[PhysicalCandidateKey, ProfilerCandidateDescriptor]

    def __post_init__(self) -> None:
        """Reject descriptors whose redundant launch geometry is inconsistent."""

        for key, descriptor in self.descriptors.items():
            descriptor_geometry = (
                descriptor.anchor_m,
                descriptor.anchor_n,
                descriptor.anchor_k,
            )
            key_geometry = (key.m, key.aggregate_n, key.k)
            if descriptor.key != key or descriptor_geometry != key_geometry:
                raise ValueError(
                    "profiler descriptor disagrees with its exact physical "
                    f"launch key {key.canonical_tuple()}"
                )

    @cached_property
    def digest(self) -> str:
        """Hash every model-visible descriptor and its source identities.

        A catalog is an immutable evidence value.  Policy cache migration asks
        for this identity once per generic domain, so recomputing the complete
        canonical JSON document at every access would turn a constant identity
        into a serial O(domain-count * descriptor-count) operation.
        """

        payload = {
            "version": PROFILER_MODEL_VERSION,
            "corpus_digest": self.corpus_digest,
            "request_manifest_digest": self.request_manifest_digest,
            "evidence_manifest_digest": self.evidence_manifest_digest,
            "descriptors": [
                {
                    "key": list(key.canonical_tuple()),
                    "anchor": {
                        "m": descriptor.anchor_m,
                        "n": descriptor.anchor_n,
                        "k": descriptor.anchor_k,
                    },
                    "features": dict(sorted(descriptor.features.items())),
                }
                for key, descriptor in sorted(
                    self.descriptors.items(),
                    key=lambda item: item[0].canonical_tuple(),
                )
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @cached_property
    def model_digest(self) -> str:
        """Hash only profiler data that can change fitted predictions.

        ``digest`` remains the full evidence provenance identity recorded in a
        frozen policy.  It intentionally changes when the authenticated timing
        corpus or profiler transaction changes.  A fit-cache key has a narrower
        purpose: it must change only when an input visible to the profiler
        surrogate changes.  Including the whole timing-corpus digest there made
        an additive refinement in one domain invalidate cross-validation for
        every unrelated ISA and M domain even when all normalized profiler
        descriptors were byte-for-byte identical.

        Candidate-cost and cross-M transfer-pool digests independently bind the
        timing rows used by each fit.  This digest therefore covers the model
        implementation version plus the complete normalized descriptor table,
        while omitting source transaction identities that have already been
        authenticated before the catalog is constructed.
        """

        return self.model_digest_for_descriptors(self.descriptors)

    def model_digest_for(
        self,
        observations: Iterable[NativeVNNIObservation],
    ) -> str:
        """Hash the exact descriptor subset reachable by one transfer pool."""

        descriptors = {}
        for observation in observations:
            descriptor = self.descriptor_for(observation)
            descriptors[descriptor.key] = descriptor
        return self.model_digest_for_descriptors(descriptors)

    @staticmethod
    def model_digest_for_descriptors(
        descriptors: Mapping[
            PhysicalCandidateKey,
            ProfilerCandidateDescriptor,
        ],
    ) -> str:
        """Hash normalized descriptors without evidence provenance fields."""

        payload = {
            "version": PROFILER_MODEL_VERSION,
            "descriptors": [
                {
                    "key": list(key.canonical_tuple()),
                    "anchor": {
                        "m": descriptor.anchor_m,
                        "n": descriptor.anchor_n,
                        "k": descriptor.anchor_k,
                    },
                    "features": dict(sorted(descriptor.features.items())),
                }
                for key, descriptor in sorted(
                    descriptors.items(),
                    key=lambda item: item[0].canonical_tuple(),
                )
            ],
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def descriptor_for(
        self,
        observation: NativeVNNIObservation,
    ) -> ProfilerCandidateDescriptor:
        """Resolve one timing row to its separately profiled exact launch."""

        key = _physical_key(observation)
        descriptor = self.descriptors.get(key)
        if descriptor is None:
            raise ValueError(
                "profiler feature catalog omits physical candidate "
                f"{observation.effective_candidate_id} for "
                f"{observation.architecture_class}/{observation.operation_kind}/"
                f"{observation.prepared_family_id}/cb{observation.runtime_codebook_id}/"
                f"M{observation.m}/N{observation.aggregate_n}/K{observation.k}/"
                f"{observation.execution_mode.value}"
            )
        return descriptor

    def rebind(self, corpus: ObservationCorpus) -> "ProfilerFeatureCatalog":
        """Reuse immutable physical-candidate evidence in a newer fit corpus.

        Adapter/schema evolution may change timing-row digests without changing
        the physical invocation. Rebinding is permitted only when every exact
        eligible target point resolves to existing evidence; it never
        synthesizes counters or weakens source-manifest validation.
        """

        rebound = ProfilerFeatureCatalog(
            corpus_digest=corpus.digest(),
            request_manifest_digest=self.request_manifest_digest,
            evidence_manifest_digest=self.evidence_manifest_digest,
            descriptors=self.descriptors,
        )
        for observation in corpus:
            if observation.supported and observation.generic_eligible:
                rebound.descriptor_for(observation)
        return rebound


def _physical_key_from_cache(values: list[str]) -> PhysicalCandidateKey:
    """Restore one exact physical key from its stable canonical tuple."""

    if len(values) != 13:
        raise ValueError("profiler catalog cache key changed tuple arity")
    projection_text = values[10]
    if not projection_text.startswith("[") or not projection_text.endswith("]"):
        raise ValueError("profiler catalog cache projection vector is malformed")
    projection_body = projection_text[1:-1]
    projection = tuple(
        int(value) for value in projection_body.split(",") if value
    )
    if not projection:
        raise ValueError("profiler catalog cache projection vector is empty")
    return PhysicalCandidateKey(
        backend=Backend(values[0]),
        architecture_class=values[1],
        operation_kind=values[2],
        bundle_signature=values[3],
        prepared_family_id=values[4],
        packing_abi=values[5],
        runtime_codebook_id=int(values[6]),
        effective_candidate_id=values[7],
        execution_mode=ExecutionMode(values[8]),
        m=int(values[9]),
        projection_n_vector=projection,
        aggregate_n=int(values[11]),
        k=int(values[12]),
    )


def _write_profiler_feature_catalog_cache(
    path: Path,
    catalog: ProfilerFeatureCatalog,
    *,
    request_file_digest: str,
    evidence_file_digest: str,
) -> None:
    """Publish an authenticated normalized catalog for repeat fitting."""

    catalog_digest = catalog.digest
    model_digest = catalog.model_digest
    payload = {
        "schema_version": PROFILER_FEATURE_CATALOG_CACHE_VERSION,
        "profiler_model_version": PROFILER_MODEL_VERSION,
        "request_file_digest": request_file_digest,
        "evidence_file_digest": evidence_file_digest,
        "corpus_digest": catalog.corpus_digest,
        "request_manifest_digest": catalog.request_manifest_digest,
        "evidence_manifest_digest": catalog.evidence_manifest_digest,
        "catalog_digest": catalog_digest,
        "model_digest": model_digest,
        "descriptor_count": len(catalog.descriptors),
        "descriptors": [
            {
                "key": list(key.canonical_tuple()),
                "anchor_m": descriptor.anchor_m,
                "anchor_n": descriptor.anchor_n,
                "anchor_k": descriptor.anchor_k,
                "features": dict(sorted(descriptor.features.items())),
            }
            for key, descriptor in sorted(
                catalog.descriptors.items(),
                key=lambda item: item[0].canonical_tuple(),
            )
        ],
    }
    document = {**payload, "cache_digest": _cache_payload_digest(payload)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".inprogress")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_profiler_feature_catalog_cache(
    path: Path,
    *,
    request_file_digest: str,
    evidence_file_digest: str,
) -> ProfilerFeatureCatalog | None:
    """Load a cache only when both canonical source files are unchanged."""

    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "cache_digest" not in raw:
        return None
    cache_digest = str(raw.pop("cache_digest"))
    if _cache_payload_digest(raw) != cache_digest:
        return None
    if (
        raw.get("schema_version") != PROFILER_FEATURE_CATALOG_CACHE_VERSION
        or raw.get("profiler_model_version") != PROFILER_MODEL_VERSION
        or raw.get("request_file_digest") != request_file_digest
        or raw.get("evidence_file_digest") != evidence_file_digest
    ):
        return None
    descriptors = {}
    for item in raw.get("descriptors", ()):
        key = _physical_key_from_cache(list(item["key"]))
        descriptor = ProfilerCandidateDescriptor(
            key=key,
            anchor_m=int(item["anchor_m"]),
            anchor_n=int(item["anchor_n"]),
            anchor_k=int(item["anchor_k"]),
            features=dict(item["features"]),
        )
        if key in descriptors:
            raise ValueError("profiler catalog cache repeats a physical key")
        descriptors[key] = descriptor
    if len(descriptors) != int(raw.get("descriptor_count", -1)):
        return None
    catalog = ProfilerFeatureCatalog(
        corpus_digest=str(raw["corpus_digest"]),
        request_manifest_digest=str(raw["request_manifest_digest"]),
        evidence_manifest_digest=str(raw["evidence_manifest_digest"]),
        descriptors=descriptors,
    )
    # The enclosing cache digest authenticates these precomputed identities.
    # Priming cached_property slots prevents another descriptor-wide JSON pass.
    catalog.__dict__["digest"] = str(raw["catalog_digest"])
    catalog.__dict__["model_digest"] = str(raw["model_digest"])
    return catalog


def build_profiler_feature_catalog(
    corpus: ObservationCorpus,
    feature_rows: Iterable[ProfilerFeatureRow],
    *,
    request_manifest_digest: str,
    evidence_manifest_digest: str,
) -> ProfilerFeatureCatalog:
    """Aggregate dispatch evidence into normalized exact-point descriptors.

    The authenticated evidence retains raw counters. The learned catalog does
    not: absolute duration, cycle, cache, grid, and traffic counts describe the
    representative request's size as much as they describe the kernel. Only
    static launch resources, normalized rates, and regime-aware work costs are
    admitted to model records.
    """

    grouped: dict[str, list[ProfilerFeatureRow]] = defaultdict(list)
    for row in feature_rows:
        grouped[row.request_id].append(row)
    descriptors = {}
    for request_id, rows in sorted(grouped.items()):
        observation = rows[0].observation
        if any(_physical_key(row.observation) != _physical_key(observation) for row in rows):
            raise ValueError(f"{request_id}: profiler dispatches changed candidate")
        features: dict[str, float | str] = {
            "profile.dispatch_count": _signed_log1p(float(len(rows))),
            "profile.execution_regime": _execution_regime(
                observation.operation_kind, observation.m
            ),
            "profile.candidate_family": observation.candidate_family,
        }
        _flatten_config("config", observation.config_json, features)
        metric_values: dict[str, list[float]] = defaultdict(list)
        for dispatch in rows:
            features[
                f"profile.dispatch_kind.{dispatch.dispatch_index}"
            ] = dispatch.dispatch_kind.value
            if dispatch.block is not None:
                features[f"profile.block_threads.{dispatch.dispatch_index}"] = (
                    _signed_log1p(float(math.prod(dispatch.block)))
                )
            for metric_id, value in dispatch.metric_values.items():
                if value is not None and math.isfinite(value):
                    metric_values[metric_id].append(float(value))
        if not metric_values:
            raise ValueError(f"{request_id}: candidate has no available profiler metrics")
        features.update(_normalized_metric_features(observation, metric_values))
        key = _physical_key(observation)
        descriptor = ProfilerCandidateDescriptor(
            key=key,
            anchor_m=observation.m,
            anchor_n=observation.aggregate_n,
            anchor_k=observation.k,
            features=features,
        )
        previous = descriptors.setdefault(key, descriptor)
        if previous != descriptor:
            raise ValueError(
                f"{request_id}: duplicate exact physical launch descriptors differ"
            )

    catalog = ProfilerFeatureCatalog(
        corpus_digest=corpus.digest(),
        request_manifest_digest=request_manifest_digest,
        evidence_manifest_digest=evidence_manifest_digest,
        descriptors=descriptors,
    )
    # Every eligible physical invocation in development must have evidence. A
    # sparse profiler sidecar cannot silently become a partially informed fit.
    for observation in corpus:
        if observation.supported and observation.generic_eligible:
            catalog.descriptor_for(observation)
    return catalog


def load_profiler_feature_catalog(
    corpus: ObservationCorpus,
    request_manifest_path: Path,
    evidence_manifest_path: Path,
    *,
    source_corpus: ObservationCorpus | None = None,
    cache_path: Path | None = None,
) -> ProfilerFeatureCatalog:
    """Authenticate profiler sidecars and build the development feature set.

    ``source_corpus`` is the immutable common CSV from which the profiler
    requests were emitted. Supplying it allows later schema/feature refits to
    reuse the same raw profiler corpus without relaunching hardware counters.
    """

    evidence_corpus = source_corpus or corpus
    request_file_digest = _file_sha256(request_manifest_path)
    evidence_file_digest = _file_sha256(evidence_manifest_path)
    cached = (
        _read_profiler_feature_catalog_cache(
            cache_path,
            request_file_digest=request_file_digest,
            evidence_file_digest=evidence_file_digest,
        )
        if cache_path is not None
        else None
    )
    if cached is not None and cached.corpus_digest == evidence_corpus.digest():
        return (
            cached
            if cached.corpus_digest == corpus.digest()
            else cached.rebind(corpus)
        )

    requests = read_profiler_request_manifest(request_manifest_path)
    evidence = read_profiler_evidence_manifest(evidence_manifest_path)
    rows = profiler_feature_rows(
        evidence_corpus,
        requests,
        evidence,
        require_complete=True,
    )
    catalog = build_profiler_feature_catalog(
        evidence_corpus,
        rows,
        request_manifest_digest=requests.digest(),
        evidence_manifest_digest=evidence.digest(),
    )
    if cache_path is not None:
        _write_profiler_feature_catalog_cache(
            cache_path,
            catalog,
            request_file_digest=request_file_digest,
            evidence_file_digest=evidence_file_digest,
        )
    return catalog if evidence_corpus.digest() == corpus.digest() else catalog.rebind(corpus)


def merge_profiler_feature_catalogs(
    corpus: ObservationCorpus,
    catalogs: Iterable[ProfilerFeatureCatalog],
) -> ProfilerFeatureCatalog:
    """Merge independently authenticated additive profiler transactions.

    A newly implemented physical candidate needs a new isolated counter launch,
    but unchanged candidates do not. Each input catalog has already proved its
    own timing-corpus, request-manifest, and evidence-manifest identities. This
    function unions only their exact physical descriptors, rejects
    conflicting duplicate evidence, content-addresses the complete provenance
    set, and finally re-establishes complete coverage against ``corpus``.

    The coverage check happens after the union. Consequently, neither an old
    catalog that lacks a new candidate nor a new incremental catalog that lacks
    old candidates is usable alone; together they must describe every supported
    generic launch in the expanded fit corpus.
    """

    sources = tuple(catalogs)
    if not sources:
        raise ValueError("profiler catalog merge requires at least one source")

    descriptors: dict[PhysicalCandidateKey, ProfilerCandidateDescriptor] = {}
    for catalog in sources:
        for key, descriptor in catalog.descriptors.items():
            previous = descriptors.setdefault(key, descriptor)
            if previous != descriptor:
                raise ValueError(
                    "additive profiler catalogs disagree for exact physical launch "
                    f"{key.canonical_tuple()}"
                )

    provenance = sorted({
        (
            catalog.corpus_digest,
            catalog.request_manifest_digest,
            catalog.evidence_manifest_digest,
            catalog.digest,
        )
        for catalog in sources
    })
    def additive_digest(kind: str, identities: object) -> str:
        encoded = json.dumps(
            {
                "schema_version": "native-vnni-additive-profiler-catalog-v1",
                "kind": kind,
                "identities": identities,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    merged = ProfilerFeatureCatalog(
        corpus_digest=corpus.digest(),
        request_manifest_digest=additive_digest(
            "request-manifests",
            sorted({
                (catalog.corpus_digest, catalog.request_manifest_digest)
                for catalog in sources
            }),
        ),
        evidence_manifest_digest=additive_digest(
            "evidence-manifests",
            provenance,
        ),
        descriptors=descriptors,
    )
    for observation in corpus:
        if observation.supported and observation.generic_eligible:
            merged.descriptor_for(observation)
    return merged


def _runtime_features(
    key: RuntimeKey,
    payload_bytes_per_32_weights: float,
) -> dict[str, float]:
    """Return runtime-visible geometry in decode/prefill economic units."""

    if payload_bytes_per_32_weights <= 0.0:
        raise ValueError("runtime format payload width must be positive")
    logical_macs = float(key.m * key.aggregate_n * key.k)
    weight_bytes = (
        float(key.aggregate_n * key.k)
        * payload_bytes_per_32_weights
        / 32.0
    )
    expected_bytes = (
        weight_bytes
        + float(key.m * key.k * 4)
        + float(key.m * key.aggregate_n * 4)
    )
    regime = _execution_regime(key.operation_kind, key.m)
    result = {
        "runtime.log2_m": math.log2(key.m),
        "runtime.log2_n": math.log2(key.aggregate_n),
        "runtime.log2_k": math.log2(key.k),
        "runtime.log2_work": math.log2(key.aggregate_n * key.k),
        "runtime.log2_macs": math.log2(logical_macs),
        "runtime.aspect": key.aggregate_n / key.k,
        "runtime.n_tiles_64": math.ceil(key.aggregate_n / 64),
        "runtime.n_tiles_128": math.ceil(key.aggregate_n / 128),
        "runtime.n_tiles_256": math.ceil(key.aggregate_n / 256),
    }
    if regime == "decode":
        result.update({
            "runtime.decode.log2_expected_bytes": math.log2(expected_bytes),
            "runtime.decode.weight_byte_fraction": weight_bytes / expected_bytes,
            "runtime.decode.arithmetic_intensity": (
                2.0 * logical_macs / expected_bytes
            ),
        })
    else:
        result.update({
            "runtime.prefill.log2_expected_bytes": math.log2(expected_bytes),
            "runtime.prefill.arithmetic_intensity": (
                2.0 * logical_macs / expected_bytes
            ),
        })
    return result


def _flattened_positive_integer(
    features: Mapping[str, float | str],
    name: str,
) -> int | None:
    """Recover one positive integer from the catalog's signed-log encoding."""

    encoded = features.get(name)
    if not isinstance(encoded, float):
        return None
    decoded = math.expm1(encoded)
    rounded = round(decoded)
    if rounded <= 0 or not math.isclose(decoded, rounded, abs_tol=1.0e-9):
        raise ValueError(
            f"profiler candidate feature {name!r} is not a positive integer"
        )
    return int(rounded)


def _cpu_parallelism_width(architecture_class: str) -> int | None:
    """Extract the production OpenMP width from one CPU architecture class."""

    for component in architecture_class.split("|"):
        if not component.startswith("threads="):
            continue
        value = component.removeprefix("threads=")
        if not value.isdigit() or int(value) <= 0:
            raise ValueError(
                "CPU profiler architecture class has invalid thread width: "
                f"{architecture_class!r}"
            )
        return int(value)
    return None


def _cpu_decode_schedule_features(
    key: RuntimeKey,
    candidate_features: Mapping[str, float | str],
) -> dict[str, float]:
    """Model the production CPU M=1 N-chunk ownership geometry.

    CPU decode candidates preserve the serial-M1 arithmetic and differ only in
    how many adjacent 64-column chunks one task owns.  The raw
    ``n_block_chunks`` value does not expose the economically important launch
    transitions:

    * full-K decode skips OpenMP when the N-block count is below the physical
      thread count, making the number of serial blocks the dominant overhead;
    * K-part decode always publishes an N-block-by-K-tile producer grid and a
      second N-chunk reduction grid; and
    * both routes cross worker-wave and N-tail boundaries as ``N`` changes.

    The exact K-tile count belongs to the frozen arithmetic policy and is not a
    candidate parameter.  It is intentionally not guessed here.  The emitted
    K-part features therefore describe the exact candidate-dependent N-block
    and reduction axes shared by every K tile, while ``N``/``K`` and the bundle
    identity let the nonlinear model learn the arithmetic policy's remaining
    geometry.  Every returned value is available before dispatch and contains
    no timing, winner, shape-name, or sealed-corpus information.
    """

    if (
        key.backend != Backend.CPU
        or key.operation_kind != "NativeVNNIFastM1Projection"
        or key.m != 1
        or candidate_features.get("config.route") != "n_chunk_grid"
    ):
        return {}
    threads = _cpu_parallelism_width(key.architecture_class)
    if threads is None:
        raise ValueError(
            "CPU decode profiler model requires threads in architecture class"
        )
    n_block_chunks = _flattened_positive_integer(
        candidate_features,
        "config.n_block_chunks",
    )
    if n_block_chunks is None:
        raise ValueError("CPU decode n_chunk_grid omits n_block_chunks")

    n_chunks = (key.aggregate_n + 63) // 64
    n_blocks = (n_chunks + n_block_chunks - 1) // n_block_chunks
    n_capacity = n_blocks * n_block_chunks * 64
    serial_kpart = ":serial-kpart:" in key.bundle_signature
    serial_full_k = ":serial-full-k:" in key.bundle_signature
    if serial_kpart == serial_full_k:
        raise ValueError(
            "CPU decode bundle must name exactly one serial arithmetic regime"
        )

    # This condition is byte-for-byte the production full-K cutoff in
    # gemv_native_vnni_preq(). K-part never takes the serial fast path because
    # its producer and ordered reduction are two OpenMP workshares.
    serial_fast_path = serial_full_k and n_blocks < threads
    n_block_waves = (n_blocks + threads - 1) // threads
    final_n_block_wave = (n_blocks - 1) % threads + 1
    reduction_waves = (n_chunks + threads - 1) // threads
    final_reduction_wave = (n_chunks - 1) % threads + 1
    logical_macs = key.aggregate_n * key.k

    return {
        "schedule.cpu_decode.n_block_chunks": float(n_block_chunks),
        "schedule.cpu_decode.n_chunks": float(n_chunks),
        "schedule.cpu_decode.n_blocks": float(n_blocks),
        "schedule.cpu_decode.log2_n_blocks": math.log2(n_blocks),
        "schedule.cpu_decode.n_block_waves": float(n_block_waves),
        "schedule.cpu_decode.log2_n_block_waves": math.log2(n_block_waves),
        "schedule.cpu_decode.final_n_block_wave_utilization": (
            final_n_block_wave / float(threads)
        ),
        "schedule.cpu_decode.n_block_thread_coverage": min(
            1.0,
            n_blocks / float(threads),
        ),
        "schedule.cpu_decode.n_grid_utilization": (
            key.aggregate_n / float(n_capacity)
        ),
        "schedule.cpu_decode.log2_output_values_per_n_block": math.log2(
            key.aggregate_n / float(n_blocks)
        ),
        "schedule.cpu_decode.log2_macs_per_n_block": math.log2(
            logical_macs / float(n_blocks)
        ),
        "schedule.cpu_decode.serial_full_k_fast_path": float(serial_fast_path),
        "schedule.cpu_decode.openmp_region": float(not serial_fast_path),
        "schedule.cpu_decode.serial_full_k_block_count": (
            float(n_blocks) if serial_fast_path else 0.0
        ),
        "schedule.cpu_decode.kpart_producer_n_blocks": (
            float(n_blocks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_reduction_tasks": (
            float(n_chunks) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_reduction_waves": (
            float(reduction_waves) if serial_kpart else 0.0
        ),
        "schedule.cpu_decode.kpart_final_reduction_wave_utilization": (
            final_reduction_wave / float(threads) if serial_kpart else 0.0
        ),
    }


def _cpu_prefill_schedule_features(
    key: RuntimeKey,
    candidate_features: Mapping[str, float | str],
) -> dict[str, float]:
    """Model the exact CPU ordinary-prefill task geometry for one candidate.

    The three full-K CPU schedules perform identical arithmetic but expose very
    different units of work to OpenMP:

    * row-chunk-grid publishes one `(row, 64-column chunk)` task;
    * two-row N-major publishes one N-block task that visits every row pair;
    * two-row pair-grid publishes one `(row pair, N block)` task.

    Raw `(M,N,K)` and a categorical route name do not tell a learner where
    those schedules cross worker-wave boundaries. These features reproduce the
    production formulas exactly and expose only runtime-visible geometry. They
    contain no timing label, shape identity, exact winner, or sealed evidence.
    Other backends and CPU K-part candidates return an empty map until their
    own reviewed launch formulas are represented symmetrically.
    """

    if key.backend != Backend.CPU or "prefill" not in key.operation_kind.lower():
        return {}
    route = candidate_features.get("config.route")
    if route not in {
        "row_chunk_grid",
        "two_row_full_output_tiles",
        "two_row_pair_grid",
    }:
        return {}
    threads = _cpu_parallelism_width(key.architecture_class)
    if threads is None:
        raise ValueError(
            "CPU prefill profiler model requires threads in architecture class"
        )

    n_chunks = (key.aggregate_n + 63) // 64
    row_pairs = (key.m + 1) // 2
    if route == "row_chunk_grid":
        n_block_chunks = 1
        n_blocks = n_chunks
        parallel_tasks = key.m * n_chunks
        rows_per_task = 1
        m_pair_utilization = 1.0
    else:
        n_block_chunks = _flattened_positive_integer(
            candidate_features,
            "config.n_block_chunks",
        )
        if n_block_chunks is None:
            raise ValueError(
                f"CPU prefill route {route!r} omits n_block_chunks"
            )
        n_blocks = (n_chunks + n_block_chunks - 1) // n_block_chunks
        m_pair_utilization = key.m / float(row_pairs * 2)
        if route == "two_row_pair_grid":
            parallel_tasks = row_pairs * n_blocks
            rows_per_task = min(2, key.m)
        else:
            parallel_tasks = n_blocks
            rows_per_task = key.m

    parallel_waves = (parallel_tasks + threads - 1) // threads
    final_wave_tasks = (parallel_tasks - 1) % threads + 1
    n_capacity = n_blocks * n_block_chunks * 64
    output_values = key.m * key.aggregate_n
    logical_macs = output_values * key.k
    return {
        "schedule.cpu_prefill.parallel_tasks": float(parallel_tasks),
        "schedule.cpu_prefill.log2_parallel_tasks": math.log2(parallel_tasks),
        "schedule.cpu_prefill.parallel_waves": float(parallel_waves),
        "schedule.cpu_prefill.log2_parallel_waves": math.log2(parallel_waves),
        "schedule.cpu_prefill.final_wave_utilization": (
            final_wave_tasks / float(threads)
        ),
        "schedule.cpu_prefill.thread_coverage": min(
            1.0,
            parallel_tasks / float(threads),
        ),
        "schedule.cpu_prefill.n_chunks": float(n_chunks),
        "schedule.cpu_prefill.n_blocks": float(n_blocks),
        "schedule.cpu_prefill.n_block_chunks": float(n_block_chunks),
        "schedule.cpu_prefill.n_grid_utilization": (
            key.aggregate_n / float(n_capacity)
        ),
        "schedule.cpu_prefill.row_pairs": float(row_pairs),
        "schedule.cpu_prefill.rows_per_task": float(rows_per_task),
        "schedule.cpu_prefill.m_pair_utilization": m_pair_utilization,
        "schedule.cpu_prefill.log2_output_values_per_task": math.log2(
            output_values / float(parallel_tasks)
        ),
        "schedule.cpu_prefill.log2_macs_per_task": math.log2(
            logical_macs / float(parallel_tasks)
        ),
    }


def build_profiler_observation_index(
    corpus: ObservationCorpus,
) -> dict[ProfilerObservationKey, NativeVNNIObservation]:
    """Select one deterministic alias row for every candidate point.

    Observation digests serialize the complete correctness and timing proof, so
    computing them is deliberately more expensive than projecting a runtime
    key.  A grouped CV domain reuses the same candidate points in every fold.
    Building this index once per domain therefore avoids repeatedly hashing the
    same immutable rows without changing which alias supplies profiler data.
    """

    grouped: dict[
        tuple[RuntimeKey, str, str], list[NativeVNNIObservation]
    ] = defaultdict(list)
    for row in corpus:
        grouped[(
            corpus.runtime_key_for(row),
            row.shape_group_id,
            row.candidate_id,
        )].append(row)
    return {
        key: min(
            rows,
            key=lambda row: (
                row.source_format,
                row.execution_mode.value,
                row.effective_candidate_id,
                row.digest(),
            ),
        )
        for key, rows in grouped.items()
    }


def _model_record(
    key: RuntimeKey,
    descriptor: ProfilerCandidateDescriptor,
    source_format: str,
) -> dict[str, float | str]:
    """Combine runtime geometry with a compact reviewed interaction set.

    Interacting every counter with every runtime dimension produced hundreds
    of weak columns from one representative profile. Decode uses byte volume,
    shape, and row count; prefill uses MAC volume, arithmetic intensity, shape,
    and row count. Only normalized metrics and launch/config geometry receive
    interactions.
    """

    payload_bytes = float(format_spec(source_format).payload_bytes)
    runtime = _runtime_features(key, payload_bytes)
    regime = _execution_regime(key.operation_kind, key.m)
    interaction_names = (
        (
            "runtime.decode.log2_expected_bytes",
            "runtime.aspect",
            "runtime.log2_m",
            "runtime.n_tiles_128",
        )
        if regime == "decode"
        else (
            "runtime.prefill.log2_expected_bytes",
            "runtime.prefill.arithmetic_intensity",
            "runtime.aspect",
            "runtime.log2_m",
        )
    )
    anchor_log2_m = math.log2(descriptor.anchor_m)
    anchor_log2_n = math.log2(descriptor.anchor_n)
    anchor_log2_k = math.log2(descriptor.anchor_k)
    record: dict[str, float | str] = {
        **runtime,
        **_cpu_decode_schedule_features(key, descriptor.features),
        **_cpu_prefill_schedule_features(key, descriptor.features),
        # These dimensions are constant in the old per-domain Ridge model but
        # become essential when one cross-M surrogate transfers evidence across
        # tensor formats and work sizes. They are runtime-visible identities,
        # never shape-name overlays or sealed labels.
        "runtime.backend": key.backend.value,
        "runtime.architecture_class": key.architecture_class,
        "runtime.semantic_contract": key.semantic_contract.value,
        "runtime.operation_kind": key.operation_kind,
        "runtime.bundle_signature": key.bundle_signature,
        "runtime.prepared_family_id": key.prepared_family_id,
        "runtime.packing_abi": key.packing_abi,
        "runtime.codebook": str(key.runtime_codebook_id),
        "runtime.execution_mode": key.execution_mode.value,
        # Dynamic counters are measurements at this exact profiler request,
        # not timeless properties of the candidate. The anchor and distance
        # features let the development learner reason about transfer while the
        # coherent-CV layer can remove counters measured inside a held region.
        "profile.anchor.m": float(descriptor.anchor_m),
        "profile.anchor.n": float(descriptor.anchor_n),
        "profile.anchor.k": float(descriptor.anchor_k),
        "profile.anchor.log2_m": anchor_log2_m,
        "profile.anchor.log2_n": anchor_log2_n,
        "profile.anchor.log2_k": anchor_log2_k,
        "profile.anchor.abs_delta_log2_m": abs(
            runtime["runtime.log2_m"] - anchor_log2_m
        ),
        "profile.anchor.abs_delta_log2_n": abs(
            runtime["runtime.log2_n"] - anchor_log2_n
        ),
        "profile.anchor.abs_delta_log2_k": abs(
            runtime["runtime.log2_k"] - anchor_log2_k
        ),
        "profile.anchor.dynamic_metrics_available": 1.0,
    }
    for name, value in descriptor.features.items():
        record[name] = value
        if isinstance(value, float) and name.startswith((
            "metric.",
            "config.",
            "profile.block_threads.",
            "profile.dispatch_count",
        )):
            for runtime_name in interaction_names:
                runtime_value = runtime[runtime_name]
                record[f"interaction.{runtime_name}.{name}"] = (
                    runtime_value * value
                )
    return record


def _is_anchor_dynamic_profiler_feature(name: str) -> bool:
    """Return whether a model column contains request-local counter evidence.

    Candidate configuration, dispatch kind, block size, and static GPU
    resource requirements survive geometry holdout. CPU performance counters
    and GPU achieved-throughput/rate measurements do not. Interactions inherit
    the locality of the counter they contain.
    """

    if name.startswith("interaction."):
        metric_offset = name.find("metric.")
        if metric_offset >= 0:
            return _is_anchor_dynamic_profiler_feature(name[metric_offset:])
        return "profile.dispatch_count" in name
    if name.startswith("profile.dispatch_count"):
        return True
    if name == "profile.cpu.duration_features_reliable":
        return True
    if name.startswith("metric.cpu."):
        return True
    if not name.startswith("metric.gpu."):
        return False
    static_metric_ids = (
        *_GPU_STATIC_RESOURCE_METRICS,
        "gpu.theoretical_occupancy_pct",
    )
    return not any(
        name.startswith(f"metric.{metric_id}.")
        for metric_id in static_metric_ids
    )


def _profiler_model_input_record(
    record: Mapping[str, float | str],
) -> dict[str, float | str]:
    """Return runtime/static inputs available for every unseen work point.

    Exact dynamic counters are training evidence, not runtime-visible inputs.
    Feeding them into training while masking them for a held point creates a
    missing-feature distribution shift and lets the forest key directly on a
    paid measurement. The multi-output target below carries their explanatory
    signal without requiring a profiler at dispatch time.
    """

    return {
        name: value
        for name, value in record.items()
        if not name.startswith("profile.anchor.")
        and not _is_anchor_dynamic_profiler_feature(name)
    }


def _profiler_auxiliary_target_record(
    record: Mapping[str, float | str],
) -> dict[str, float]:
    """Return non-redundant exact dynamic metrics as auxiliary fit targets."""

    return {
        name: value
        for name, value in record.items()
        if isinstance(value, float)
        and _is_anchor_dynamic_profiler_feature(name)
        and not name.startswith("interaction.")
        and name != "profile.anchor.dynamic_metrics_available"
        # This bit says whether duration-derived counters passed the control
        # overhead gate; it is not itself an economic property. Likewise, the
        # one-shot wall/task clocks are diagnostics beside the canonical
        # repeated timing corpus, not auxiliary timing labels.
        and name != "profile.cpu.duration_features_reliable"
        and "task_clock_ns_" not in name
        and "wall_clock_ns_" not in name
    }


def _center_complete_candidate_relative_auxiliary(
    auxiliary_records: list[dict[str, float]],
    point_members: Mapping[tuple[RuntimeKey, str], list[int]],
) -> list[dict[str, float]]:
    """Center only metrics available for every candidate in one contest.

    A missing optional counter must mean "no evidence at this point", not the
    numeric value zero. A metric is therefore admitted for a runtime/shape
    contest only when every competing candidate published it. Unavailable
    contests remain all-zero after matrix assembly, which is the neutral value
    for candidate-relative supervision.
    """

    centered = [dict() for _ in auxiliary_records]
    for members in point_members.values():
        names = {
            name
            for index in members
            for name in auxiliary_records[index]
        }
        for name in names:
            if not all(name in auxiliary_records[index] for index in members):
                continue
            mean = sum(
                auxiliary_records[index][name] for index in members
            ) / float(len(members))
            for index in members:
                centered[index][name] = auxiliary_records[index][name] - mean
    return centered


def _auxiliary_metric_reliability_weight(name: str) -> float:
    """Return an audited relative trust weight for one profiler target.

    The CPU control probes showed instruction counts to be the most repeatable
    signal, L1 traffic to be useful but somewhat noisier, and sparse LLC misses
    to be diagnostic rather than authoritative. Duration-derived cycle/rate
    features are admitted only after the reliability gate and remain below the
    deterministic work counters. GPU metrics retain equal provisional weight
    until their backend-specific repeatability probes calibrate this table.
    """

    if name.startswith("profile.dispatch_count"):
        return 1.0
    if "instructions_" in name:
        return 1.0
    if "l1d_load" in name:
        return 0.75
    if "llc_load" in name:
        return 0.25
    if name.startswith("metric.cpu."):
        return 0.5
    return 0.5


def _mask_held_out_profiler_anchor(
    record: dict[str, float | str],
    held_out_geometries: frozenset[tuple[int, int]],
) -> dict[str, float | str]:
    """Remove dynamic counters measured at a coherently held N/K geometry."""

    anchor = (
        int(record["profile.anchor.n"]),
        int(record["profile.anchor.k"]),
    )
    if anchor not in held_out_geometries:
        return record
    masked = {
        name: value
        for name, value in record.items()
        if not _is_anchor_dynamic_profiler_feature(name)
    }
    masked["profile.anchor.dynamic_metrics_available"] = 0.0
    return masked


def apply_profiler_regret_predictions(
    costs: Iterable[CandidateCostLike],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    *,
    observation_index: ProfilerObservationIndex | None = None,
    model_training_costs: Iterable[CandidateCostLike] | None = None,
    model_record_index: Mapping[
        tuple[RuntimeKey, str, str], dict[str, float | str]
    ] | None = None,
    excluded_profiler_geometries: frozenset[tuple[int, int]] = frozenset(),
) -> list[CandidateCostLike]:
    """Train one deterministic ExtraTrees surrogate and attach predictions.

    ``costs`` are the rows whose fitting costs receive predictions.
    ``model_training_costs`` may be a wider, leakage-controlled cross-M pool
    spanning tensor formats and prefill work sizes. The policy learner excludes
    every held-out geometry across that entire pool before supplying it. This
    lets the nonlinear model learn candidate/shape/work-size crossovers without
    seeing a held-out timing label under another codebook or M. Exact dynamic
    counters are candidate-relative auxiliary targets, never model inputs;
    ``excluded_profiler_geometries`` applies the same boundary to those targets.
    Runtime geometry, static candidate configuration, schedule formulas, and
    static resource use remain available inputs.

    Predictions are clamped to non-negative regret. The policy learner owns the
    deliberately separate question of how much a surrogate may influence a
    measured candidate cost; keeping the raw prediction here makes that budget
    explicit and independently testable instead of hiding it in model output.
    """

    costs = list(costs)
    if not costs:
        return costs
    training_costs = list(
        costs if model_training_costs is None else model_training_costs
    )
    if not training_costs:
        raise ValueError("profiler model training pool is empty")
    candidate_ids = {cost.candidate_id for cost in training_costs}
    measured_targets = {
        float(cost.max_surface_regret) for cost in training_costs
    }
    if len(candidate_ids) < 2 or len(measured_targets) < 2:
        # A one-candidate domain and an exact all-candidate tie have no dispatch
        # choice for profiler features to explain. Preserve measured costs; the
        # catalog completeness gate still proves that every launch was profiled.
        return costs
    observations = (
        build_profiler_observation_index(corpus)
        if observation_index is None
        else observation_index
    )

    def record_for(cost: CandidateCostLike) -> dict[str, float | str]:
        """Resolve one candidate point to its authenticated model record."""

        point = (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
        if model_record_index is not None:
            record = model_record_index.get(point)
            if record is None:
                raise ValueError(
                    f"profiler record index cannot resolve candidate point {point}"
                )
            return _mask_held_out_profiler_anchor(
                record,
                excluded_profiler_geometries,
            )
        observation = observations.get(point)
        if observation is None:
            raise ValueError(
                f"profiler model cannot resolve candidate point {point}"
            )
        descriptor = catalog.descriptor_for(observation)
        return _mask_held_out_profiler_anchor(
            _model_record(
                cost.runtime_key,
                descriptor,
                observation.source_format,
            ),
            excluded_profiler_geometries,
        )

    training_records = []
    regret_targets = []
    for cost in training_costs:
        training_records.append(record_for(cost))
        regret_targets.append(math.log1p(max(0.0, cost.max_surface_regret)))

    input_vectorizer = DictVectorizer(sparse=False)
    training_matrix = input_vectorizer.fit_transform([
        _profiler_model_input_record(record) for record in training_records
    ])
    auxiliary_records = [
        _profiler_auxiliary_target_record(record)
        for record in training_records
    ]
    point_members: dict[tuple[RuntimeKey, str], list[int]] = defaultdict(list)
    for index, cost in enumerate(training_costs):
        point_members[(cost.runtime_key, cost.shape_group_id)].append(index)
    auxiliary_records = _center_complete_candidate_relative_auxiliary(
        auxiliary_records,
        point_members,
    )
    auxiliary_names = sorted({
        name for record in auxiliary_records for name in record
    })
    varying_auxiliary: list[tuple[str, float, float]] = []
    for name in auxiliary_names:
        values = [record.get(name, 0.0) for record in auxiliary_records]
        mean = sum(values) / float(len(values))
        variance = sum((value - mean) ** 2 for value in values) / float(
            len(values)
        )
        if variance > 1.0e-18:
            varying_auxiliary.append((name, mean, math.sqrt(variance)))
    if not varying_auxiliary:
        # A fold with no varying exact dynamic evidence is measured-only.
        # Candidate configuration and static resources cannot masquerade as
        # profiler influence by themselves.
        return costs

    regret_mean = sum(regret_targets) / float(len(regret_targets))
    regret_variance = sum(
        (value - regret_mean) ** 2 for value in regret_targets
    ) / float(len(regret_targets))
    if regret_variance <= 1.0e-18:
        return costs
    regret_scale = math.sqrt(regret_variance)
    auxiliary_weight_sum = sum(
        _auxiliary_metric_reliability_weight(name)
        for name, _mean, _scale in varying_auxiliary
    )
    auxiliary_scales = {
        name: math.sqrt(
            PROFILER_AUXILIARY_TARGET_TOTAL_WEIGHT
            * _auxiliary_metric_reliability_weight(name)
            / auxiliary_weight_sum
        )
        for name, _mean, _scale in varying_auxiliary
    }
    targets = []
    for regret, record in zip(
        regret_targets, auxiliary_records, strict=True
    ):
        targets.append([
            (regret - regret_mean) / regret_scale,
            *(
                auxiliary_scales[name]
                * (record.get(name, 0.0) - mean)
                / scale
                for name, mean, scale in varying_auxiliary
            ),
        ])
    model = ExtraTreesRegressor(
        n_estimators=PROFILER_EXTRA_TREES,
        max_depth=PROFILER_EXTRA_TREES_MAX_DEPTH,
        min_samples_leaf=PROFILER_EXTRA_TREES_MIN_SAMPLES_LEAF,
        max_features=1.0,
        n_jobs=1,
        random_state=PROFILER_EXTRA_TREES_RANDOM_STATE,
    )
    model.fit(training_matrix, targets)
    predictions = model.predict(input_vectorizer.transform([
        _profiler_model_input_record(record_for(cost)) for cost in costs
    ]))
    return [
        replace(
            cost,
            profiler_predicted_regret=max(
                0.0,
                math.expm1(
                    float(prediction[0]) * regret_scale + regret_mean
                ),
            ),
        )
        for cost, prediction in zip(costs, predictions, strict=True)
    ]


def build_profiler_model_record_index(
    costs: Iterable[CandidateCostLike],
    corpus: ObservationCorpus,
    catalog: ProfilerFeatureCatalog,
    *,
    observation_index: ProfilerObservationIndex | None = None,
) -> dict[tuple[RuntimeKey, str, str], dict[str, float | str]]:
    """Materialize immutable surrogate features once per candidate point.

    Cross-validation fits many leakage-controlled forests over different row
    subsets of the same cross-M transfer pool. Rebuilding runtime/profiler
    interactions for every fold is pure duplicate work and used to serialize
    policy fitting for several minutes. This index is independent of timing
    targets and held-out labels, so every forest can safely reuse it while
    retaining its own strictly filtered training-row set.
    """

    observations = (
        build_profiler_observation_index(corpus)
        if observation_index is None
        else observation_index
    )
    records = {}
    for cost in costs:
        point = (cost.runtime_key, cost.shape_group_id, cost.candidate_id)
        if point in records:
            continue
        observation = observations.get(point)
        if observation is None:
            raise ValueError(
                f"profiler record index cannot resolve candidate point {point}"
            )
        descriptor = catalog.descriptor_for(observation)
        records[point] = _model_record(
            cost.runtime_key,
            descriptor,
            observation.source_format,
        )
    return records
