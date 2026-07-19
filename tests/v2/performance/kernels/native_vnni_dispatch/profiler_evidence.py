"""Provenance-bound profiler requests and evidence for NativeVNNI training.

Canonical latency collection and hardware-counter collection are deliberately
different transactions.  The timing trainer first emits immutable common
observations.  This module derives one profiler request from each observation,
then validates profiler evidence gathered by a separate process invocation.
Consequently, Nsight Compute replay, rocprofiler counter passes, and Linux
``perf`` instrumentation can never perturb the latency samples used to choose
or certify a dispatch policy.

A candidate may execute more than one device kernel (for example a CUDA K-part
producer followed by an ordered reducer).  Profiler evidence therefore owns an
ordered list of dispatches rather than pretending that one policy candidate is
always one physical kernel.  Every dispatch carries the complete backend metric
inventory; unsupported optional metrics are explicit records with reasons and
are never represented by a missing JSON field.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .candidate_observation import read_observation_csv, write_observation_csv
from .candidate_registry import (
    CandidateRegistry,
    candidate_registry_digest,
    cpu_native_vnni_decode_registry,
    cpu_native_vnni_prefill_registry,
    cpu_native_vnni_verifier_registry,
    cuda_native_vnni_gemv_registry,
    rocm_moe_grouped_prefill_registry,
    rocm_native_vnni_decode_registry,
)
from .corpus import ObservationCorpus
from .schema import (
    Backend,
    COMPATIBLE_OBSERVATION_LEARNER_VERSIONS,
    COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS,
    ExecutionMode,
    FEATURE_SCHEMA_VERSION,
    LEARNER_VERSION,
    NativeVNNIObservation,
    OBSERVATION_COLUMNS,
    POLICY_ABI,
    SCHEMA_VERSION,
    SemanticContract,
)


LEGACY_PROFILER_REQUEST_SCHEMA_VERSION = "native-vnni-profiler-request-v3"
PROFILER_REQUEST_SCHEMA_VERSION = "native-vnni-profiler-request-v4-exact-point"
SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS = frozenset({
    LEGACY_PROFILER_REQUEST_SCHEMA_VERSION,
    PROFILER_REQUEST_SCHEMA_VERSION,
})
PROFILER_EVIDENCE_SCHEMA_VERSION = "native-vnni-profiler-evidence-v1"
PROFILER_METRIC_SET_VERSION = "native-vnni-profiler-metrics-v1"
PROFILER_COLLECTOR_VERSION = (
    "native-vnni-isolated-profiler-v3-direct-per-tid-batched"
)
SUPPORTED_PROFILER_COLLECTOR_VERSIONS = frozenset({
    "native-vnni-isolated-profiler-v1",
    "native-vnni-isolated-profiler-v2",
    PROFILER_COLLECTOR_VERSION,
})
PROFILE_PROTOCOL = "isolated-production-candidate-launch-v1"


def _sha256_json(value: Any) -> str:
    """Return a stable SHA-256 identity for one JSON-compatible value."""

    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


_PARALLEL_MANIFEST_RECORDS: tuple[Any, ...] = ()


def _physical_core_count() -> int:
    """Return the affinity-visible physical-core count for offline hashing."""

    try:
        visible_cpus = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        visible_cpus = tuple(range(os.cpu_count() or 1))
    physical = set()
    for cpu in visible_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return max(1, len(visible_cpus))
        physical.add((package, core))
    return max(1, len(physical))


def _serialize_manifest_record_range(bounds: tuple[int, int]) -> bytes:
    """Serialize one contiguous record range without changing array order."""

    begin, end = bounds
    return ",".join(
        json.dumps(
            _PARALLEL_MANIFEST_RECORDS[index].canonical_mapping(),
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(begin, end)
    ).encode()


def _sha256_manifest_records(
    scalar_fields: Mapping[str, Any],
    array_name: str,
    records: tuple[Any, ...],
) -> str:
    """Hash a large canonical manifest without materializing its giant dict.

    The legacy identity is compact ``json.dumps(..., sort_keys=True)`` over one
    root object. Root keys are emitted in that same lexical order while record
    shards are serialized independently and reduced in original array order.
    This preserves every historical digest byte, caps peak memory, and lets the
    two-socket host use physical cores without oversubscribing hyperthreads.
    """

    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_MANIFEST_DIGEST_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("manifest digest worker count must be positive")
    worker_count = min(requested_workers, max(1, len(records) // 4096))
    if worker_count == 1:
        serialized_shards = (_serialize_manifest_records_inline(records),)
    else:
        base = len(records) // worker_count
        remainder = len(records) % worker_count
        bounds = []
        begin = 0
        for worker_index in range(worker_count):
            size = base + (1 if worker_index < remainder else 0)
            bounds.append((begin, begin + size))
            begin += size
        global _PARALLEL_MANIFEST_RECORDS
        _PARALLEL_MANIFEST_RECORDS = records
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                serialized_shards = tuple(executor.map(
                    _serialize_manifest_record_range,
                    bounds,
                ))
        finally:
            _PARALLEL_MANIFEST_RECORDS = ()

    digest = hashlib.sha256()
    digest.update(b"{")
    root_names = sorted((*scalar_fields, array_name))
    for field_index, name in enumerate(root_names):
        if field_index != 0:
            digest.update(b",")
        digest.update(json.dumps(name, separators=(",", ":")).encode())
        digest.update(b":")
        if name == array_name:
            digest.update(b"[")
            for shard_index, shard in enumerate(serialized_shards):
                if shard_index != 0 and shard:
                    digest.update(b",")
                digest.update(shard)
            digest.update(b"]")
        else:
            digest.update(json.dumps(
                scalar_fields[name], sort_keys=True, separators=(",", ":")
            ).encode())
    digest.update(b"}")
    return "sha256:" + digest.hexdigest()


def _serialize_manifest_records_inline(records: tuple[Any, ...]) -> bytes:
    """Serialize small inventories in process to avoid fork overhead."""

    return ",".join(
        json.dumps(
            record.canonical_mapping(), sort_keys=True, separators=(",", ":")
        )
        for record in records
    ).encode()


def _is_sha256(value: str) -> bool:
    """Return whether ``value`` is one complete lowercase SHA-256 identity."""

    if len(value) != 71 or not value.startswith("sha256:"):
        return False
    return all(character in "0123456789abcdef" for character in value[7:])


def _required_text(name: str, value: Any) -> str:
    """Normalize one required string and reject empty provenance fields."""

    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def _require_exact_keys(
    raw: Mapping[str, Any], expected: Iterable[str], label: str
) -> None:
    """Reject omitted and forward-version fields instead of ignoring either."""

    actual = set(raw)
    required = set(expected)
    if actual != required:
        raise ValueError(
            f"{label} fields do not match schema: "
            f"missing={sorted(required - actual)} "
            f"unexpected={sorted(actual - required)}"
        )


def _candidate_registries(backend: Backend) -> tuple[CandidateRegistry, ...]:
    """Return every reviewed registry that can own one backend observation."""

    if backend == Backend.CPU:
        return (
            cpu_native_vnni_decode_registry(),
            cpu_native_vnni_verifier_registry(),
            cpu_native_vnni_prefill_registry(),
        )
    if backend == Backend.CUDA:
        return (cuda_native_vnni_gemv_registry(),)
    if backend == Backend.ROCM:
        return (
            rocm_native_vnni_decode_registry(),
            rocm_moe_grouped_prefill_registry(),
        )
    raise ValueError(f"unsupported profiler backend {backend}")


def _registry_for_observation(
    observation: NativeVNNIObservation,
) -> tuple[CandidateRegistry, Any]:
    """Resolve one observation to exactly one reviewed forceable candidate.

    Shape-resolved CUDA formula rows are synthetic learner evidence backed by a
    concrete exact-KB timing row.  They are intentionally rejected here: the
    concrete source row is the physical launch that must be profiled, while the
    formula itself does not name another kernel variant.
    """

    matches = []
    for registry in _candidate_registries(observation.backend):
        try:
            matches.append((registry, registry.resolve(observation.candidate_id)))
        except ValueError:
            continue
    if len(matches) != 1:
        raise ValueError(
            f"{observation.candidate_id}: expected exactly one candidate registry, "
            f"found {len(matches)}"
        )
    registry, candidate = matches[0]
    if candidate.config_json.get("family") == "kpar_formula":
        raise ValueError(
            f"{observation.candidate_id}: a shape-resolved policy formula is not "
            "a physical profiler launch; use the unprojected concrete observation"
        )
    if observation.candidate_family != candidate.candidate_family:
        raise ValueError("observation candidate_family disagrees with registry")
    if observation.config_json != candidate.config_json:
        raise ValueError("observation config_json disagrees with candidate registry")
    if observation.arithmetic_fingerprint != candidate.arithmetic_fingerprint:
        raise ValueError(
            "observation arithmetic_fingerprint disagrees with candidate registry"
        )
    if observation.candidate_policy_hash != candidate.candidate_policy_hash():
        raise ValueError("observation candidate_policy_hash is stale")
    return registry, candidate


@dataclass(frozen=True, order=True)
class ProfilerRequest:
    """One isolated physical-candidate profiling obligation.

    The request repeats the launch discriminators needed by a standalone
    backend trainer.  It also binds those discriminators to the complete common
    observation and raw timing sidecar digest, preventing a profiler result
    from being joined to a newer timing corpus merely because shape and
    candidate names happen to match.
    """

    request_id: str
    observation_digest: str
    run_id: str
    corpus_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    backend: Backend
    architecture_class: str
    device_name: str
    driver_runtime: str
    threading_or_stream_mode: str
    semantic_contract: SemanticContract
    operation_kind: str
    bundle_signature: str
    projection_n_vector: tuple[int, ...]
    source_format: str
    source_codebook_id: int
    prepared_family_id: str
    packing_abi: str
    runtime_codebook_id: int
    shape_group_id: str
    shape_name: str
    execution_mode: ExecutionMode
    m: int
    aggregate_n: int
    k: int
    candidate_registry_surface: str
    candidate_id: str
    effective_candidate_id: str
    candidate_family: str
    config_json: dict[str, Any]
    schedule_signature: str
    workspace_signature: str
    prepared_resources: tuple[str, ...]
    arithmetic_fingerprint: str
    candidate_policy_hash: str
    trial_set_hash: str
    timing_sample_hash: str
    supported: bool
    graph_capture_ok: bool
    forced_route_ok: bool
    observed_candidate_id: str
    profile_protocol: str
    target_launches_per_profiler_pass: int

    @property
    def profile_required(self) -> bool:
        """Return whether a successful isolated profile is mandatory."""

        return (
            self.supported
            and self.forced_route_ok
            and (
                self.execution_mode != ExecutionMode.GRAPH_CAPTURED
                or self.graph_capture_ok
            )
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the deterministic JSON form used by request manifests."""

        # A shallow field projection is sufficient because this is a frozen
        # value object and the structured members are immutable by contract.
        # ``asdict`` recursively deep-copies every nested value and made a
        # 119k-request authentication pass spend minutes copying dictionaries.
        result = {field.name: getattr(self, field.name) for field in fields(self)}
        result["backend"] = self.backend.value
        result["semantic_contract"] = self.semantic_contract.value
        result["execution_mode"] = self.execution_mode.value
        result["projection_n_vector"] = list(self.projection_n_vector)
        result["prepared_resources"] = list(self.prepared_resources)
        return result

    def validate(self) -> None:
        """Reject requests that cannot identify one production launch."""

        for name in (
            "request_id",
            "run_id",
            "corpus_id",
            "git_revision",
            "build_id",
            "compiler_id",
            "architecture_class",
            "device_name",
            "driver_runtime",
            "threading_or_stream_mode",
            "operation_kind",
            "bundle_signature",
            "source_format",
            "prepared_family_id",
            "packing_abi",
            "shape_group_id",
            "shape_name",
            "candidate_registry_surface",
            "candidate_id",
            "effective_candidate_id",
            "candidate_family",
            "schedule_signature",
            "workspace_signature",
            "arithmetic_fingerprint",
            "candidate_policy_hash",
            "trial_set_hash",
            "timing_sample_hash",
            "observed_candidate_id",
        ):
            _required_text(name, getattr(self, name))
        for name in (
            "observation_digest",
            "candidate_policy_hash",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256 identity")
        if self.profile_protocol != PROFILE_PROTOCOL:
            raise ValueError("unsupported profiler request protocol")
        if self.target_launches_per_profiler_pass != 1:
            raise ValueError("each profiler pass must contain exactly one target launch")
        if self.m <= 0 or self.aggregate_n <= 0 or self.k <= 0:
            raise ValueError("profile dimensions must be positive")
        if not self.projection_n_vector or any(
            dimension <= 0 for dimension in self.projection_n_vector
        ):
            raise ValueError("projection_n_vector must contain positive dimensions")
        if sum(self.projection_n_vector) != self.aggregate_n:
            raise ValueError("projection_n_vector does not sum to aggregate_n")
        if not self.prepared_resources or any(
            not resource.strip() for resource in self.prepared_resources
        ):
            raise ValueError("prepared_resources must identify registry resources")
        if self.candidate_id.strip().upper() == "AUTO":
            raise ValueError("AUTO is not a physical profiler candidate")
        if self.config_json.get("family") == "kpar_formula":
            raise ValueError("shape-resolved formulas are not physical profiler requests")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerRequest":
        """Parse one exact request record from a JSON manifest."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiler request")
        result = cls(
            request_id=str(raw["request_id"]),
            observation_digest=str(raw["observation_digest"]),
            run_id=str(raw["run_id"]),
            corpus_id=str(raw["corpus_id"]),
            git_revision=str(raw["git_revision"]),
            build_id=str(raw["build_id"]),
            compiler_id=str(raw["compiler_id"]),
            backend=Backend(str(raw["backend"])),
            architecture_class=str(raw["architecture_class"]),
            device_name=str(raw["device_name"]),
            driver_runtime=str(raw["driver_runtime"]),
            threading_or_stream_mode=str(raw["threading_or_stream_mode"]),
            semantic_contract=SemanticContract(str(raw["semantic_contract"])),
            operation_kind=str(raw["operation_kind"]),
            bundle_signature=str(raw["bundle_signature"]),
            projection_n_vector=tuple(
                int(value) for value in raw["projection_n_vector"]
            ),
            source_format=str(raw["source_format"]),
            source_codebook_id=int(raw["source_codebook_id"]),
            prepared_family_id=str(raw["prepared_family_id"]),
            packing_abi=str(raw["packing_abi"]),
            runtime_codebook_id=int(raw["runtime_codebook_id"]),
            shape_group_id=str(raw["shape_group_id"]),
            shape_name=str(raw["shape_name"]),
            execution_mode=ExecutionMode(str(raw["execution_mode"])),
            m=int(raw["m"]),
            aggregate_n=int(raw["aggregate_n"]),
            k=int(raw["k"]),
            candidate_registry_surface=str(raw["candidate_registry_surface"]),
            candidate_id=str(raw["candidate_id"]),
            effective_candidate_id=str(raw["effective_candidate_id"]),
            candidate_family=str(raw["candidate_family"]),
            config_json=dict(raw["config_json"]),
            schedule_signature=str(raw["schedule_signature"]),
            workspace_signature=str(raw["workspace_signature"]),
            prepared_resources=tuple(str(value) for value in raw["prepared_resources"]),
            arithmetic_fingerprint=str(raw["arithmetic_fingerprint"]),
            candidate_policy_hash=str(raw["candidate_policy_hash"]),
            trial_set_hash=str(raw["trial_set_hash"]),
            timing_sample_hash=str(raw["timing_sample_hash"]),
            supported=bool(raw["supported"]),
            graph_capture_ok=bool(raw["graph_capture_ok"]),
            forced_route_ok=bool(raw["forced_route_ok"]),
            observed_candidate_id=str(raw["observed_candidate_id"]),
            profile_protocol=str(raw["profile_protocol"]),
            target_launches_per_profiler_pass=int(
                raw["target_launches_per_profiler_pass"]
            ),
        )
        result.validate()
        return result


def _request_identity_fields(
    observation: NativeVNNIObservation,
    registry_surface: str,
) -> dict[str, Any]:
    """Return launch identity fields used to derive a stable request ID."""

    return {
        "observation_digest": observation.digest(),
        "backend": observation.backend.value,
        "architecture_class": observation.architecture_class,
        "registry_surface": registry_surface,
        "source_format": observation.source_format,
        "execution_mode": observation.execution_mode.value,
        "m": observation.m,
        "n": observation.aggregate_n,
        "k": observation.k,
        "candidate_id": observation.candidate_id,
        "effective_candidate_id": observation.effective_candidate_id,
        "profile_protocol": PROFILE_PROTOCOL,
    }


def profiler_request_for_observation(
    observation: NativeVNNIObservation,
) -> ProfilerRequest:
    """Derive one immutable isolated-profile request from a timing row."""

    observation.validate()
    registry, candidate = _registry_for_observation(observation)
    identity = _request_identity_fields(observation, registry.surface)
    request_hash = _sha256_json(identity)[7:]
    result = ProfilerRequest(
        request_id=f"nvprof-{observation.backend.value}-{request_hash[:24]}",
        observation_digest=observation.digest(),
        run_id=observation.run_id,
        corpus_id=observation.corpus_id,
        git_revision=observation.git_revision,
        build_id=observation.build_id,
        compiler_id=observation.compiler_id,
        backend=observation.backend,
        architecture_class=observation.architecture_class,
        device_name=observation.device_name,
        driver_runtime=observation.driver_runtime,
        threading_or_stream_mode=observation.threading_or_stream_mode,
        semantic_contract=observation.semantic_contract,
        operation_kind=observation.operation_kind,
        bundle_signature=observation.bundle_signature,
        projection_n_vector=observation.projection_n_vector,
        source_format=observation.source_format,
        source_codebook_id=observation.source_codebook_id,
        prepared_family_id=observation.prepared_family_id,
        packing_abi=observation.packing_abi,
        runtime_codebook_id=observation.runtime_codebook_id,
        shape_group_id=observation.shape_group_id,
        shape_name=observation.shape_name,
        execution_mode=observation.execution_mode,
        m=observation.m,
        aggregate_n=observation.aggregate_n,
        k=observation.k,
        candidate_registry_surface=registry.surface,
        candidate_id=observation.candidate_id,
        effective_candidate_id=observation.effective_candidate_id,
        candidate_family=observation.candidate_family,
        config_json=observation.config_json,
        schedule_signature=candidate.schedule_signature,
        workspace_signature=candidate.workspace_signature,
        prepared_resources=candidate.prepared_resources,
        arithmetic_fingerprint=observation.arithmetic_fingerprint,
        candidate_policy_hash=observation.candidate_policy_hash,
        trial_set_hash=observation.trial_set_hash,
        timing_sample_hash=observation.timing_sample_hash,
        supported=observation.supported,
        graph_capture_ok=observation.graph_capture_ok,
        forced_route_ok=observation.forced_route_ok,
        observed_candidate_id=observation.observed_candidate_id,
        profile_protocol=PROFILE_PROTOCOL,
        target_launches_per_profiler_pass=1,
    )
    result.validate()
    return result


@dataclass(frozen=True)
class ProfilerRequestManifest:
    """Complete profiler obligation inventory for one timing corpus."""

    corpus_digest: str
    candidate_registry_digest: str
    requests: tuple[ProfilerRequest, ...]
    learner_version: str = LEARNER_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    schema_version: str = PROFILER_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS:
            raise ValueError(
                "unsupported profiler request schema_version="
                f"{self.schema_version!r}; accepted versions are "
                f"{sorted(SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS)!r}"
            )
        if not self.requests:
            raise ValueError("profiler request manifest must not be empty")
        if not _is_sha256(self.corpus_digest):
            raise ValueError("profiler request corpus_digest is invalid")
        if not _is_sha256(self.candidate_registry_digest):
            raise ValueError("profiler request registry digest is invalid")
        if self.learner_version not in COMPATIBLE_OBSERVATION_LEARNER_VERSIONS:
            raise ValueError(
                f"unsupported profiler request learner_version={self.learner_version!r}; "
                "accepted immutable evidence versions are "
                f"{sorted(COMPATIBLE_OBSERVATION_LEARNER_VERSIONS)!r}"
            )
        if (
            self.feature_schema_version
            not in COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS
        ):
            raise ValueError(
                "unsupported profiler request feature_schema_version="
                f"{self.feature_schema_version!r}; accepted immutable evidence "
                "versions are "
                f"{sorted(COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS)!r}"
            )
        request_ids = [request.request_id for request in self.requests]
        observation_ids = [request.observation_digest for request in self.requests]
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("profiler request IDs must be unique")
        if len(set(observation_ids)) != len(observation_ids):
            raise ValueError("one timing observation cannot create two profiler requests")
        for request in self.requests:
            request.validate()
        if self.schema_version == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION:
            _validate_matched_profiler_anchors(self.requests)
        else:
            _validate_exact_profiler_launches(self.requests)

    def payload_mapping(self) -> dict[str, Any]:
        """Return the digest-covered root without its self hash."""

        return {
            "schema_version": self.schema_version,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "observation_schema_version": SCHEMA_VERSION,
            "policy_abi": POLICY_ABI,
            "learner_version": self.learner_version,
            "feature_schema_version": self.feature_schema_version,
            "corpus_digest": self.corpus_digest,
            "candidate_registry_digest": self.candidate_registry_digest,
            "request_count": len(self.requests),
            "requests": [request.canonical_mapping() for request in self.requests],
        }

    def digest(self) -> str:
        """Hash every request and all schema/provenance versions."""

        return self._cached_digest

    @cached_property
    def _cached_digest(self) -> str:
        """Serialize this immutable request inventory once per process."""

        return _sha256_manifest_records(
            {
                "schema_version": self.schema_version,
                "metric_set_version": PROFILER_METRIC_SET_VERSION,
                "observation_schema_version": SCHEMA_VERSION,
                "policy_abi": POLICY_ABI,
                "learner_version": self.learner_version,
                "feature_schema_version": self.feature_schema_version,
                "corpus_digest": self.corpus_digest,
                "candidate_registry_digest": self.candidate_registry_digest,
                "request_count": len(self.requests),
            },
            "requests",
            self.requests,
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the complete JSON document including its self hash."""

        return {**self.payload_mapping(), "manifest_digest": self.digest()}


def build_profiler_request_manifest(
    corpus: ObservationCorpus,
) -> ProfilerRequestManifest:
    """Create one isolated request for every measured physical invocation.

    Dynamic counters are point evidence. IPC, cache behavior, occupancy,
    throughput, and duration observed at one ``(M,N,K)`` launch must never be
    attached to another work size. The v4 manifest therefore covers every
    launchable generic timing point independently across backend, ISA/runtime,
    prepared codebook, execution mode, geometry, work size, and effective
    candidate.

    Timing rows may still contain genuine aliases: two source formats can
    prepare the identical execution codebook, and two shape names can describe
    the same physical geometry. Those rows share one profiler request only
    when every launch-changing field is identical. The deterministic
    representative remains a cryptographic witness for the shared invocation;
    no unsupported or non-forceable row can stand in for executable evidence.
    """

    launchable: dict[
        tuple[object, ...], list[NativeVNNIObservation]
    ] = {}
    never_launchable: dict[
        tuple[object, ...], list[NativeVNNIObservation]
    ] = {}
    for row in corpus:
        candidate_key = (
            row.backend,
            row.architecture_class,
            row.operation_kind,
            row.bundle_signature,
            row.prepared_family_id,
            row.packing_abi,
            row.runtime_codebook_id,
            row.effective_candidate_id,
        )
        if row.supported and row.forced_route_ok and row.generic_eligible:
            launch_key = (
                *candidate_key,
                row.execution_mode,
                row.m,
                row.projection_n_vector,
                row.aggregate_n,
                row.k,
            )
            launchable.setdefault(launch_key, []).append(row)
        else:
            never_launchable.setdefault(candidate_key, []).append(row)

    selected: dict[str, NativeVNNIObservation] = {}
    launchable_candidates = {
        key[:8] for key in launchable
    }
    for rows in launchable.values():
        representative = min(
            rows,
            key=lambda row: (
                row.source_format,
                row.source_codebook_id,
                row.shape_name,
                row.shape_group_id,
                row.digest(),
            ),
        )
        selected[representative.digest()] = representative

    # Retain one explicit capability record only for candidates that are never
    # launchable anywhere in the corpus. It creates no profiler process because
    # ``profile_required`` is false, but preserves the reviewed reason that the
    # registry member has no executable point in this generation.
    for candidate_key, rows in never_launchable.items():
        if candidate_key in launchable_candidates:
            continue
        representative = min(
            rows,
            key=lambda row: (
                row.m,
                row.aggregate_n,
                row.k,
                row.source_format,
                row.shape_name,
                row.digest(),
            ),
        )
        selected[representative.digest()] = representative

    requests = tuple(sorted(
        (profiler_request_for_observation(row) for row in selected.values()),
        key=lambda request: (
            request.backend.value,
            request.architecture_class,
            request.semantic_contract.value,
            request.operation_kind,
            request.source_format,
            request.execution_mode.value,
            request.m,
            request.aggregate_n,
            request.k,
            request.effective_candidate_id,
            request.observation_digest,
        ),
    ))
    return ProfilerRequestManifest(
        corpus_digest=corpus.digest(),
        candidate_registry_digest=candidate_registry_digest(),
        requests=requests,
        schema_version=PROFILER_REQUEST_SCHEMA_VERSION,
    )


def build_missing_profiler_request_transaction(
    corpus: ObservationCorpus,
    covered_manifests: Iterable[ProfilerRequestManifest],
) -> tuple[ObservationCorpus | None, ProfilerRequestManifest | None]:
    """Derive only exact physical launches absent from prior transactions.

    A timing corpus may gain authenticated candidate-expansion or refinement
    rows after an earlier profiler transaction was collected. Reprofiling the
    complete corpus would waste hours and create duplicate authority for every
    unchanged launch. This function builds the current complete obligation
    inventory, compares launch-changing identity rather than representative
    source aliases, and returns a self-contained delta transaction.

    The returned requests retain the exact observation digests chosen from the
    enlarged corpus. Their compact observation corpus therefore provides the
    timing witnesses needed by normal evidence validation and later additive
    composition. An already complete corpus returns ``(None, None)`` so a
    turnkey replay can stop successfully without creating empty manifests.
    """

    complete = build_profiler_request_manifest(corpus)
    covered_keys = _covered_profiler_launch_keys(covered_manifests)

    missing_requests = tuple(
        request
        for request in complete.requests
        if _profiled_exact_launch_key(request) not in covered_keys
    )
    if not missing_requests:
        return None, None

    rows_by_digest = {row.digest(): row for row in corpus}
    missing_rows = []
    for request in missing_requests:
        try:
            missing_rows.append(rows_by_digest[request.observation_digest])
        except KeyError as error:
            raise ValueError(
                f"{request.request_id}: missing request lost its timing witness"
            ) from error
    observations = ObservationCorpus._from_validated(missing_rows)
    requests = ProfilerRequestManifest(
        corpus_digest=observations.digest(),
        candidate_registry_digest=complete.candidate_registry_digest,
        requests=missing_requests,
        learner_version=complete.learner_version,
        feature_schema_version=complete.feature_schema_version,
        schema_version=complete.schema_version,
    )
    return observations, requests


def count_uncovered_profiler_requests(
    requests: ProfilerRequestManifest,
    covered_manifests: Iterable[ProfilerRequestManifest],
) -> int:
    """Count exact launches in ``requests`` absent from prior transactions.

    Request IDs deliberately bind a profile record to its complete timing
    witness, including run provenance. Resume decisions instead concern the
    physical launch that already paid for hardware counters. Keeping this
    comparison explicit lets a regenerated timing corpus reuse an authenticated
    profile while still invalidating changed arithmetic, schedules, workspace,
    execution modes, or geometry.
    """

    covered_keys = _covered_profiler_launch_keys(covered_manifests)
    return sum(
        1
        for request in requests.requests
        if _profiled_exact_launch_key(request) not in covered_keys
    )


def _covered_profiler_launch_keys(
    manifests: Iterable[ProfilerRequestManifest],
) -> set[tuple[object, ...]]:
    """Return the authenticated physical coverage represented by manifests."""

    covered_keys: set[tuple[object, ...]] = set()
    for manifest in manifests:
        for request in manifest.requests:
            key = (
                _profiled_physical_candidate_key(request)
                if manifest.schema_version == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION
                else _profiled_exact_launch_key(request)
            )
            covered_keys.add(key)
    return covered_keys


def write_profiler_request_manifest(
    path: Path, manifest: ProfilerRequestManifest
) -> None:
    """Write a deterministic request manifest for backend collectors."""

    path.write_text(
        json.dumps(manifest.canonical_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_profiler_request_manifest(path: Path) -> ProfilerRequestManifest:
    """Read and authenticate one exact profiler request document."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    root_fields = {
        "schema_version",
        "metric_set_version",
        "observation_schema_version",
        "policy_abi",
        "learner_version",
        "feature_schema_version",
        "corpus_digest",
        "candidate_registry_digest",
        "request_count",
        "requests",
        "manifest_digest",
    }
    _require_exact_keys(raw, root_fields, "profiler request manifest")
    exact_versions = {
        "metric_set_version": PROFILER_METRIC_SET_VERSION,
        "observation_schema_version": SCHEMA_VERSION,
        "policy_abi": POLICY_ABI,
    }
    for name, expected in exact_versions.items():
        if raw[name] != expected:
            raise ValueError(
                f"profiler request {name}={raw[name]!r}; expected {expected!r}"
            )
    if raw["schema_version"] not in SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported profiler request schema_version="
            f"{raw['schema_version']!r}; accepted versions are "
            f"{sorted(SUPPORTED_PROFILER_REQUEST_SCHEMA_VERSIONS)!r}"
        )
    requests_raw = raw["requests"]
    if not isinstance(requests_raw, list):
        raise ValueError("profiler requests must be a JSON array")
    if int(raw["request_count"]) != len(requests_raw):
        raise ValueError("profiler request_count does not match request inventory")
    manifest = ProfilerRequestManifest(
        corpus_digest=str(raw["corpus_digest"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        requests=tuple(ProfilerRequest.from_mapping(record) for record in requests_raw),
        learner_version=str(raw["learner_version"]),
        feature_schema_version=str(raw["feature_schema_version"]),
        schema_version=str(raw["schema_version"]),
    )
    if str(raw["manifest_digest"]) != manifest.digest():
        raise ValueError("profiler request manifest digest does not match contents")
    return manifest


class MetricCategory(str, Enum):
    """Whether one metric is static launch state or a dynamic measurement."""

    STATIC_RESOURCE = "static_resource"
    DYNAMIC_COUNTER = "dynamic_counter"


class MetricAvailability(str, Enum):
    """Explicit disposition of one canonical profiler metric."""

    MEASURED = "measured"
    UNSUPPORTED_BY_TOOL = "unsupported_by_tool"
    COLLECTION_FAILED = "collection_failed"


@dataclass(frozen=True, order=True)
class MetricDefinition:
    """Canonical metric expected from one backend collector."""

    metric_id: str
    category: MetricCategory
    unit: str
    required: bool


CPU_METRIC_DEFINITIONS = (
    MetricDefinition("cpu.cycles", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.ref_cycles", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.instructions", MetricCategory.DYNAMIC_COUNTER, "count", True),
    MetricDefinition("cpu.task_clock_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("cpu.wall_clock_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("cpu.branches", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.branch_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.cache_references", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.cache_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.l1d_loads", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.l1d_load_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.llc_loads", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("cpu.llc_load_misses", MetricCategory.DYNAMIC_COUNTER, "count", False),
)

CUDA_METRIC_DEFINITIONS = (
    MetricDefinition("gpu.duration_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("gpu.registers_per_thread", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.static_shared_memory_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.dynamic_shared_memory_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.local_memory_bytes_per_thread", MetricCategory.STATIC_RESOURCE, "bytes", False),
    MetricDefinition("gpu.theoretical_occupancy_pct", MetricCategory.STATIC_RESOURCE, "percent", True),
    MetricDefinition("gpu.achieved_occupancy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.compute_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.dram_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.l1_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.l2_throughput_pct_of_peak", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.executed_ipc_active", MetricCategory.DYNAMIC_COUNTER, "instructions_per_cycle", False),
    MetricDefinition("gpu.warp_cycles_per_issued_instruction", MetricCategory.DYNAMIC_COUNTER, "cycles", False),
    MetricDefinition("gpu.local_memory_spill_requests", MetricCategory.DYNAMIC_COUNTER, "count", False),
)

ROCM_METRIC_DEFINITIONS = (
    MetricDefinition("gpu.duration_ns", MetricCategory.DYNAMIC_COUNTER, "ns", True),
    MetricDefinition("gpu.vgpr_count", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.sgpr_count", MetricCategory.STATIC_RESOURCE, "count", True),
    MetricDefinition("gpu.lds_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.scratch_bytes", MetricCategory.STATIC_RESOURCE, "bytes", True),
    MetricDefinition("gpu.achieved_occupancy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.gpu_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.valu_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.valu_utilization_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.memory_unit_busy_pct", MetricCategory.DYNAMIC_COUNTER, "percent", True),
    MetricDefinition("gpu.memory_unit_stalled_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.l2_cache_hit_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
    MetricDefinition("gpu.fetch_kib", MetricCategory.DYNAMIC_COUNTER, "KiB", False),
    MetricDefinition("gpu.write_kib", MetricCategory.DYNAMIC_COUNTER, "KiB", False),
    MetricDefinition("gpu.wavefront_count", MetricCategory.DYNAMIC_COUNTER, "count", False),
    MetricDefinition("gpu.lds_bank_conflict_pct", MetricCategory.DYNAMIC_COUNTER, "percent", False),
)


def metric_definitions(backend: Backend) -> tuple[MetricDefinition, ...]:
    """Return the exact v1 metric inventory for one backend."""

    if backend == Backend.CPU:
        return CPU_METRIC_DEFINITIONS
    if backend == Backend.CUDA:
        return CUDA_METRIC_DEFINITIONS
    if backend == Backend.ROCM:
        return ROCM_METRIC_DEFINITIONS
    raise ValueError(f"unsupported profiler backend {backend}")


@dataclass(frozen=True, order=True)
class ProfilerMetric:
    """One measured or explicitly unavailable canonical profiler feature."""

    metric_id: str
    category: MetricCategory
    unit: str
    availability: MetricAvailability
    value: float | None
    source_name: str
    reason: str | None

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible metric record."""

        result = asdict(self)
        result["category"] = self.category.value
        result["availability"] = self.availability.value
        return result

    def validate(self, definition: MetricDefinition) -> None:
        """Validate identity, units, finiteness, and explicit unavailability."""

        if self.metric_id != definition.metric_id:
            raise ValueError("profiler metric ID disagrees with metric definition")
        if self.category != definition.category or self.unit != definition.unit:
            raise ValueError(f"{self.metric_id}: metric category/unit mismatch")
        _required_text("source_name", self.source_name)
        if self.availability == MetricAvailability.MEASURED:
            if self.value is None or not math.isfinite(self.value):
                raise ValueError(f"{self.metric_id}: measured value must be finite")
            if self.reason not in (None, ""):
                raise ValueError(f"{self.metric_id}: measured value cannot have a failure reason")
        else:
            if self.value is not None:
                raise ValueError(f"{self.metric_id}: unavailable metric cannot carry a value")
            _required_text("reason", self.reason)
            if definition.required:
                raise ValueError(
                    f"{self.metric_id}: required profiler metric was not measured"
                )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerMetric":
        """Parse one exact metric record."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiler metric")
        value = raw["value"]
        return cls(
            metric_id=str(raw["metric_id"]),
            category=MetricCategory(str(raw["category"])),
            unit=str(raw["unit"]),
            availability=MetricAvailability(str(raw["availability"])),
            value=None if value is None else float(value),
            source_name=str(raw["source_name"]),
            reason=None if raw["reason"] is None else str(raw["reason"]),
        )


class ProfiledDispatchKind(str, Enum):
    """Physical work represented by one profiler dispatch record."""

    GPU_KERNEL = "gpu_kernel"
    CPU_PARALLEL_REGION = "cpu_parallel_region"


@dataclass(frozen=True)
class ProfiledDispatch:
    """One ordered kernel in a candidate pipeline or one CPU kernel region."""

    dispatch_index: int
    dispatch_kind: ProfiledDispatchKind
    kernel_name: str
    kernel_fingerprint: str
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    metrics: tuple[ProfilerMetric, ...]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return the exact nested JSON representation."""

        return {
            "dispatch_index": self.dispatch_index,
            "dispatch_kind": self.dispatch_kind.value,
            "kernel_name": self.kernel_name,
            "kernel_fingerprint": self.kernel_fingerprint,
            "grid": None if self.grid is None else list(self.grid),
            "block": None if self.block is None else list(self.block),
            "metrics": [metric.canonical_mapping() for metric in self.metrics],
        }

    def validate(self, backend: Backend) -> None:
        """Require complete metric coverage and valid physical geometry."""

        if self.dispatch_index < 0:
            raise ValueError("dispatch_index must be non-negative")
        _required_text("kernel_name", self.kernel_name)
        if not _is_sha256(self.kernel_fingerprint):
            raise ValueError("kernel_fingerprint must be a SHA-256 identity")
        expected_kind = (
            ProfiledDispatchKind.CPU_PARALLEL_REGION
            if backend == Backend.CPU
            else ProfiledDispatchKind.GPU_KERNEL
        )
        if self.dispatch_kind != expected_kind:
            raise ValueError("profiled dispatch kind disagrees with backend")
        if backend == Backend.CPU:
            if self.grid is not None or self.block is not None:
                raise ValueError("CPU profiler regions do not have GPU launch geometry")
        else:
            for name, geometry in (("grid", self.grid), ("block", self.block)):
                if geometry is None or len(geometry) != 3 or any(value <= 0 for value in geometry):
                    raise ValueError(f"GPU {name} must contain three positive dimensions")

        definitions = {item.metric_id: item for item in metric_definitions(backend)}
        observed = {metric.metric_id: metric for metric in self.metrics}
        if len(observed) != len(self.metrics):
            raise ValueError("profiled dispatch contains duplicate metric IDs")
        if set(observed) != set(definitions):
            raise ValueError(
                "profiled dispatch metric inventory mismatch: "
                f"missing={sorted(set(definitions) - set(observed))} "
                f"unexpected={sorted(set(observed) - set(definitions))}"
            )
        for metric_id, definition in definitions.items():
            observed[metric_id].validate(definition)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfiledDispatch":
        """Parse one exact ordered dispatch record."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiled dispatch")

        def geometry(name: str) -> tuple[int, int, int] | None:
            value = raw[name]
            if value is None:
                return None
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{name} must be null or a three-element array")
            return tuple(int(item) for item in value)  # type: ignore[return-value]

        return cls(
            dispatch_index=int(raw["dispatch_index"]),
            dispatch_kind=ProfiledDispatchKind(str(raw["dispatch_kind"])),
            kernel_name=str(raw["kernel_name"]),
            kernel_fingerprint=str(raw["kernel_fingerprint"]),
            grid=geometry("grid"),
            block=geometry("block"),
            metrics=tuple(
                ProfilerMetric.from_mapping(item) for item in raw["metrics"]
            ),
        )


class ProfilerEvidenceStatus(str, Enum):
    """Disposition of one isolated candidate profile transaction."""

    COMPLETE = "complete"
    CANDIDATE_UNSUPPORTED = "candidate_unsupported"
    TOOL_UNAVAILABLE = "tool_unavailable"
    LAUNCH_FAILED = "launch_failed"
    PARSE_FAILED = "parse_failed"


EXPECTED_PROFILER_TOOL = {
    Backend.CPU: "linux-perf",
    Backend.CUDA: "nsight-compute",
    Backend.ROCM: "rocprofiler-sdk",
}


@dataclass(frozen=True)
class ProfilerEvidence:
    """Complete outcome of one request's separate profiler invocation."""

    request_id: str
    observation_digest: str
    backend: Backend
    status: ProfilerEvidenceStatus
    status_reason: str | None
    profiler_tool: str
    profiler_tool_version: str
    metric_set_version: str
    collector_version: str
    command_digest: str
    raw_artifact_digest: str
    profiler_pass_count: int
    target_launches_per_profiler_pass: int
    dispatches: tuple[ProfiledDispatch, ...]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return deterministic JSON for the evidence sidecar."""

        return {
            "request_id": self.request_id,
            "observation_digest": self.observation_digest,
            "backend": self.backend.value,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "profiler_tool": self.profiler_tool,
            "profiler_tool_version": self.profiler_tool_version,
            "metric_set_version": self.metric_set_version,
            "collector_version": self.collector_version,
            "command_digest": self.command_digest,
            "raw_artifact_digest": self.raw_artifact_digest,
            "profiler_pass_count": self.profiler_pass_count,
            "target_launches_per_profiler_pass": self.target_launches_per_profiler_pass,
            "dispatches": [dispatch.canonical_mapping() for dispatch in self.dispatches],
        }

    def validate(self, request: ProfilerRequest) -> None:
        """Validate request binding and successful/failed-state invariants."""

        if self.request_id != request.request_id:
            raise ValueError("profiler evidence request_id mismatch")
        if self.observation_digest != request.observation_digest:
            raise ValueError("profiler evidence observation_digest mismatch")
        if self.backend != request.backend:
            raise ValueError("profiler evidence backend mismatch")
        if self.metric_set_version != PROFILER_METRIC_SET_VERSION:
            raise ValueError("unsupported profiler metric set")
        if self.collector_version not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
            raise ValueError("unsupported profiler collector version")
        if self.target_launches_per_profiler_pass != 1:
            raise ValueError("profiler evidence must isolate one target launch per pass")

        if self.status == ProfilerEvidenceStatus.COMPLETE:
            if not request.profile_required:
                raise ValueError("unreachable candidate cannot claim complete profile evidence")
            if self.status_reason not in (None, ""):
                raise ValueError("complete profiler evidence cannot have a failure reason")
            if self.profiler_tool != EXPECTED_PROFILER_TOOL[request.backend]:
                raise ValueError("profiler evidence used the wrong backend tool")
            _required_text("profiler_tool_version", self.profiler_tool_version)
            if not _is_sha256(self.command_digest):
                raise ValueError("profiler command digest is invalid")
            if not _is_sha256(self.raw_artifact_digest):
                raise ValueError("profiler raw artifact digest is invalid")
            if self.profiler_pass_count <= 0:
                raise ValueError("complete profiler evidence requires at least one pass")
            if not self.dispatches:
                raise ValueError("complete profiler evidence has no physical dispatches")
            if tuple(dispatch.dispatch_index for dispatch in self.dispatches) != tuple(
                range(len(self.dispatches))
            ):
                raise ValueError("profiled dispatches must have contiguous launch order")
            for dispatch in self.dispatches:
                dispatch.validate(request.backend)
            return

        _required_text("status_reason", self.status_reason)
        if self.dispatches:
            raise ValueError("failed profiler evidence cannot publish partial dispatch metrics")
        if self.profiler_pass_count != 0:
            raise ValueError("failed profiler evidence must report zero completed passes")
        if self.status == ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED:
            if request.profile_required:
                raise ValueError("required candidate cannot be marked unsupported")
        elif not request.profile_required:
            raise ValueError(
                "an unreachable candidate must use candidate_unsupported, not a tool failure"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "ProfilerEvidence":
        """Parse one exact evidence record; request validation happens later."""

        _require_exact_keys(raw, cls.__dataclass_fields__, "profiler evidence")
        return cls(
            request_id=str(raw["request_id"]),
            observation_digest=str(raw["observation_digest"]),
            backend=Backend(str(raw["backend"])),
            status=ProfilerEvidenceStatus(str(raw["status"])),
            status_reason=(
                None if raw["status_reason"] is None else str(raw["status_reason"])
            ),
            profiler_tool=str(raw["profiler_tool"]),
            profiler_tool_version=str(raw["profiler_tool_version"]),
            metric_set_version=str(raw["metric_set_version"]),
            collector_version=str(raw["collector_version"]),
            command_digest=str(raw["command_digest"]),
            raw_artifact_digest=str(raw["raw_artifact_digest"]),
            profiler_pass_count=int(raw["profiler_pass_count"]),
            target_launches_per_profiler_pass=int(
                raw["target_launches_per_profiler_pass"]
            ),
            dispatches=tuple(
                ProfiledDispatch.from_mapping(item) for item in raw["dispatches"]
            ),
        )


@dataclass(frozen=True)
class ProfilerEvidenceManifest:
    """Incremental or complete evidence bound to one request manifest."""

    request_manifest_digest: str
    corpus_digest: str
    candidate_registry_digest: str
    evidence: tuple[ProfilerEvidence, ...]
    collector_version: str = PROFILER_COLLECTOR_VERSION

    def __post_init__(self) -> None:
        for name in (
            "request_manifest_digest",
            "corpus_digest",
            "candidate_registry_digest",
        ):
            if not _is_sha256(getattr(self, name)):
                raise ValueError(f"profiler evidence {name} is invalid")
        request_ids = [item.request_id for item in self.evidence]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("profiler evidence contains duplicate request IDs")
        if self.collector_version not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
            raise ValueError("unsupported profiler evidence collector")
        if any(
            item.collector_version != self.collector_version
            for item in self.evidence
        ):
            raise ValueError("profiler evidence manifest mixes collector generations")

    def payload_mapping(self) -> dict[str, Any]:
        """Return digest-covered evidence root without its self hash."""

        return {
            "schema_version": PROFILER_EVIDENCE_SCHEMA_VERSION,
            "metric_set_version": PROFILER_METRIC_SET_VERSION,
            "collector_version": self.collector_version,
            "request_manifest_digest": self.request_manifest_digest,
            "corpus_digest": self.corpus_digest,
            "candidate_registry_digest": self.candidate_registry_digest,
            "evidence_count": len(self.evidence),
            "evidence": [item.canonical_mapping() for item in self.evidence],
        }

    def digest(self) -> str:
        """Hash all profiler records independently of filesystem location."""

        return self._cached_digest

    @cached_property
    def _cached_digest(self) -> str:
        """Serialize this immutable evidence inventory once per process."""

        return _sha256_manifest_records(
            {
                "schema_version": PROFILER_EVIDENCE_SCHEMA_VERSION,
                "metric_set_version": PROFILER_METRIC_SET_VERSION,
                "collector_version": self.collector_version,
                "request_manifest_digest": self.request_manifest_digest,
                "corpus_digest": self.corpus_digest,
                "candidate_registry_digest": self.candidate_registry_digest,
                "evidence_count": len(self.evidence),
            },
            "evidence",
            self.evidence,
        )

    def canonical_mapping(self) -> dict[str, Any]:
        """Return complete evidence JSON including its self hash."""

        return {**self.payload_mapping(), "manifest_digest": self.digest()}


def write_profiler_evidence_manifest(
    path: Path, manifest: ProfilerEvidenceManifest
) -> None:
    """Write a deterministic profiler evidence sidecar."""

    path.write_text(
        json.dumps(manifest.canonical_mapping(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_profiler_evidence_manifest(path: Path) -> ProfilerEvidenceManifest:
    """Read and authenticate an incremental or complete evidence sidecar."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    root_fields = {
        "schema_version",
        "metric_set_version",
        "collector_version",
        "request_manifest_digest",
        "corpus_digest",
        "candidate_registry_digest",
        "evidence_count",
        "evidence",
        "manifest_digest",
    }
    _require_exact_keys(raw, root_fields, "profiler evidence manifest")
    if raw["schema_version"] != PROFILER_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("unsupported profiler evidence schema")
    if raw["metric_set_version"] != PROFILER_METRIC_SET_VERSION:
        raise ValueError("unsupported profiler evidence metric set")
    if raw["collector_version"] not in SUPPORTED_PROFILER_COLLECTOR_VERSIONS:
        raise ValueError("unsupported profiler evidence collector")
    evidence_raw = raw["evidence"]
    if not isinstance(evidence_raw, list):
        raise ValueError("profiler evidence inventory must be a JSON array")
    if int(raw["evidence_count"]) != len(evidence_raw):
        raise ValueError("profiler evidence_count does not match inventory")
    manifest = ProfilerEvidenceManifest(
        request_manifest_digest=str(raw["request_manifest_digest"]),
        corpus_digest=str(raw["corpus_digest"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        evidence=tuple(ProfilerEvidence.from_mapping(item) for item in evidence_raw),
        collector_version=str(raw["collector_version"]),
    )
    if str(raw["manifest_digest"]) != manifest.digest():
        raise ValueError("profiler evidence manifest digest does not match contents")
    return manifest


@dataclass(frozen=True)
class ProfilerCoverageReport:
    """Human-readable disposition of one request/evidence join."""

    request_count: int
    required_count: int
    unsupported_count: int
    complete_count: int
    failure_count: int
    missing_request_ids: tuple[str, ...]
    failed_request_ids: tuple[str, ...]

    @property
    def complete(self) -> bool:
        """Return whether every request has its only valid terminal state."""

        return not self.missing_request_ids and not self.failed_request_ids


def validate_profiler_evidence_coverage(
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> ProfilerCoverageReport:
    """Join sidecars exactly and enforce complete supported-candidate coverage.

    ``require_complete=False`` is useful while collectors checkpoint a long
    all-format sweep.  It never relaxes record validation: any evidence record
    that is present must still be internally complete and bound to its request.
    """

    if evidence.request_manifest_digest != requests.digest():
        raise ValueError("profiler evidence belongs to another request manifest")
    if evidence.corpus_digest != requests.corpus_digest:
        raise ValueError("profiler evidence belongs to another timing corpus")
    if evidence.candidate_registry_digest != requests.candidate_registry_digest:
        raise ValueError("profiler evidence uses another candidate registry")

    request_by_id = {request.request_id: request for request in requests.requests}
    evidence_by_id = {item.request_id: item for item in evidence.evidence}
    extras = sorted(set(evidence_by_id) - set(request_by_id))
    if extras:
        raise ValueError(f"profiler evidence contains unknown requests: {extras}")

    missing = tuple(sorted(set(request_by_id) - set(evidence_by_id)))
    failed = []
    complete_count = 0
    unsupported_count = 0
    required_count = sum(request.profile_required for request in requests.requests)
    for request_id, item in sorted(evidence_by_id.items()):
        request = request_by_id[request_id]
        item.validate(request)
        expected_status = (
            ProfilerEvidenceStatus.COMPLETE
            if request.profile_required
            else ProfilerEvidenceStatus.CANDIDATE_UNSUPPORTED
        )
        if item.status != expected_status:
            failed.append(request_id)
        elif item.status == ProfilerEvidenceStatus.COMPLETE:
            complete_count += 1
        else:
            unsupported_count += 1

    report = ProfilerCoverageReport(
        request_count=len(request_by_id),
        required_count=required_count,
        unsupported_count=unsupported_count,
        complete_count=complete_count,
        failure_count=len(failed),
        missing_request_ids=missing,
        failed_request_ids=tuple(failed),
    )
    if require_complete and not report.complete:
        raise ValueError(
            "profiler coverage is incomplete: "
            f"missing={list(report.missing_request_ids)} "
            f"failed={list(report.failed_request_ids)}"
        )
    return report


@dataclass(frozen=True)
class ComposedProfilerEvidence:
    """One authenticated union of independently collected profiler corpora.

    Candidate-family expansion deliberately profiles only newly introduced
    physical kernels.  A later policy fit therefore owns several immutable
    request/evidence transactions rather than one monolithic profiler replay.
    This value packages their exact timing witnesses and records into the
    canonical three files expected by corpus publication without pretending
    that counters measured at one anchor geometry came from another geometry.
    """

    observations: ObservationCorpus
    requests: ProfilerRequestManifest
    evidence: ProfilerEvidenceManifest


def _profiled_physical_candidate_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return the shape-independent kernel identity used by profiler fitting.

    Geometry is intentionally absent.  One physical kernel variant may have
    been profiled at only one representative anchor, and two additive sources
    must never contribute competing anchors for that same fitted descriptor.
    """

    return (
        request.backend,
        request.architecture_class,
        request.operation_kind,
        request.bundle_signature,
        request.prepared_family_id,
        request.packing_abi,
        request.runtime_codebook_id,
        request.candidate_registry_surface,
        request.effective_candidate_id,
        request.arithmetic_fingerprint,
        request.candidate_policy_hash,
        request.schedule_signature,
        request.workspace_signature,
        request.prepared_resources,
        request.threading_or_stream_mode,
    )


def _profiled_candidate_surface_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return the candidate-independent surface for comparable counters.

    Candidate identity is deliberately absent. Every launchable candidate in
    this surface competes for the same runtime work and therefore must be
    profiled at the same anchor set. Comparing IPC, cache, or throughput from
    candidate families measured at unrelated geometries confounds the family
    label with work size and can silently poison the learned dispatch fit.
    """

    return (
        request.backend,
        request.architecture_class,
        request.operation_kind,
        request.bundle_signature,
        request.prepared_family_id,
        request.packing_abi,
        request.runtime_codebook_id,
        request.candidate_registry_surface,
    )


def _profiled_anchor(request: ProfilerRequest) -> tuple[object, ...]:
    """Return every geometry field that changes one profiler launch."""

    return (
        request.m,
        request.projection_n_vector,
        request.aggregate_n,
        request.k,
    )


def _profiled_exact_launch_key(
    request: ProfilerRequest,
) -> tuple[object, ...]:
    """Return every discriminator that identifies one physical invocation.

    Dynamic counters are meaningful only for the exact work submitted to the
    kernel.  In particular, changing execution mode or any of ``M``, ``N``,
    or ``K`` creates another profiling obligation even when the selected
    template instantiation is unchanged.  Source-format and shape-name aliases
    are intentionally absent because prepared-family/codebook and geometry
    already identify the physical work they share.
    """

    return (
        *_profiled_physical_candidate_key(request),
        request.execution_mode,
        *_profiled_anchor(request),
    )


def _validate_exact_profiler_launches(
    requests: Iterable[ProfilerRequest],
) -> None:
    """Require exactly one request for every executable physical invocation.

    The request builder may collapse true observation aliases before this
    validation runs.  Any remaining duplicate means two evidence records could
    claim ownership of the same kernel/ISA/codebook/candidate/M/N/K launch,
    which would make the attached counters ambiguous.
    """

    owner_by_launch: dict[tuple[object, ...], str] = {}
    for request in requests:
        if not request.profile_required:
            continue
        launch_key = _profiled_exact_launch_key(request)
        owner = owner_by_launch.setdefault(launch_key, request.request_id)
        if owner != request.request_id:
            raise ValueError(
                "profiler manifest contains duplicate evidence obligations for "
                f"exact physical launch {launch_key}: {owner} and "
                f"{request.request_id}"
            )


def _validate_matched_profiler_anchors(
    requests: Iterable[ProfilerRequest],
) -> None:
    """Require identical profiler anchor sets across competing candidates.

    A base transaction and a later candidate-family expansion are each valid
    in isolation when their own candidates share an anchor. Their union is not
    valid if the two transactions chose different geometries. Validating the
    complete request inventory catches that additive-evidence failure before
    any normalized counter is admitted to a learner.

    The current feature catalog owns one shape-independent descriptor per
    physical candidate, so each candidate must have exactly one anchor. A
    future multi-anchor model must first make anchor identity an explicit
    descriptor dimension rather than silently collapsing several launches.
    """

    anchors_by_surface: dict[
        tuple[object, ...],
        dict[str, set[tuple[object, ...]]],
    ] = {}
    for request in requests:
        if not request.profile_required:
            continue
        candidate_anchors = anchors_by_surface.setdefault(
            _profiled_candidate_surface_key(request),
            {},
        )
        candidate_anchors.setdefault(
            request.effective_candidate_id,
            set(),
        ).add(_profiled_anchor(request))

    for surface, anchors_by_candidate in anchors_by_surface.items():
        multiple_anchors = {
            candidate: tuple(sorted(anchors))
            for candidate, anchors in anchors_by_candidate.items()
            if len(anchors) != 1
        }
        if multiple_anchors:
            raise ValueError(
                "profiler physical candidates must use exactly one anchor for "
                f"comparable surface {surface}: {multiple_anchors}"
            )
        if len(anchors_by_candidate) < 2:
            continue
        anchor_sets = {
            tuple(sorted(anchors))
            for anchors in anchors_by_candidate.values()
        }
        if len(anchor_sets) != 1:
            detail = {
                candidate: tuple(sorted(anchors))
                for candidate, anchors in sorted(anchors_by_candidate.items())
            }
            raise ValueError(
                "profiler candidates use unmatched anchor sets for comparable "
                f"surface {surface}: {detail}"
            )


def compose_profiler_evidence(
    sources: Iterable[
        tuple[
            ObservationCorpus,
            ProfilerRequestManifest,
            ProfilerEvidenceManifest,
        ]
    ],
) -> ComposedProfilerEvidence:
    """Compose complete additive profiler transactions without new launches.

    Every source is authenticated independently before any record is admitted.
    Exact duplicate sources are harmless and deduplicated, while a repeated
    request with different contents, conflicting evidence, a second anchor for
    one physical candidate, mixed registry generations, or incomplete coverage
    is a hard failure.  The resulting top-level manifests are rebound only to
    the union transaction; request IDs, observation digests, launch geometry,
    raw-artifact identities, and all counters remain byte-for-byte unchanged.

    The composed ``corpus_digest`` identifies the complete ordered set of
    authenticated source request/evidence/witness digests.  Feature export uses
    the returned exact compact witnesses, so it still re-derives every retained
    request from the timing observation that actually produced the launch.
    """

    source_set = tuple(sources)
    if not source_set:
        raise ValueError("profiler evidence composition requires at least one source")

    request_by_id: dict[str, ProfilerRequest] = {}
    evidence_by_id: dict[str, ProfilerEvidence] = {}
    observation_by_digest: dict[str, NativeVNNIObservation] = {}
    physical_candidate_owner: dict[tuple[object, ...], str] = {}
    source_provenance: set[tuple[str, str, str]] = set()
    registry_digests: set[str] = set()
    source_feature_schema_versions: set[str] = set()
    collector_version: str | None = None
    request_schema_version: str | None = None

    for source_index, (observations, requests, evidence) in enumerate(source_set):
        validate_profiler_evidence_coverage(requests, evidence, require_complete=True)
        compact = compact_profiler_observation_witnesses(
            observations,
            requests,
            evidence,
            require_complete=True,
        )
        source_provenance.add((requests.digest(), evidence.digest(), compact.digest()))

        # Feature engineering is intentionally downstream of immutable raw
        # counter collection. A newer compatible exporter may add normalized
        # learner features without changing any profiled physical launch or
        # metric. Retain those older transactions and normalize the composed
        # manifest to FEATURE_SCHEMA_VERSION below. Request/collector schemas
        # still describe collection semantics and therefore remain exact.
        source_feature_schema_versions.add(requests.feature_schema_version)
        expected_metadata = (
            ("collector", collector_version, evidence.collector_version),
            ("request schema", request_schema_version, requests.schema_version),
        )
        for label, previous, current in expected_metadata:
            if previous is not None and previous != current:
                raise ValueError(
                    f"profiler source {source_index} uses another {label} generation"
                )
        registry_digests.add(requests.candidate_registry_digest)
        collector_version = evidence.collector_version
        request_schema_version = requests.schema_version

        source_evidence_by_id = {item.request_id: item for item in evidence.evidence}
        for request in requests.requests:
            previous_request = request_by_id.setdefault(request.request_id, request)
            if previous_request.canonical_mapping() != request.canonical_mapping():
                raise ValueError(
                    f"{request.request_id}: profiler sources disagree on request contents"
                )

            item = source_evidence_by_id[request.request_id]
            previous_evidence = evidence_by_id.setdefault(request.request_id, item)
            if previous_evidence.canonical_mapping() != item.canonical_mapping():
                raise ValueError(
                    f"{request.request_id}: profiler sources publish conflicting evidence"
                )

            physical_key = (
                _profiled_physical_candidate_key(request)
                if requests.schema_version
                == LEGACY_PROFILER_REQUEST_SCHEMA_VERSION
                else _profiled_exact_launch_key(request)
            )
            owner = physical_candidate_owner.setdefault(physical_key, request.request_id)
            if owner != request.request_id:
                raise ValueError(
                    "profiler sources contain duplicate evidence for physical launch "
                    f"{physical_key}: {owner} and {request.request_id}"
                )

        for observation in compact:
            digest = observation.digest()
            previous = observation_by_digest.setdefault(digest, observation)
            if previous.canonical_mapping() != observation.canonical_mapping():
                raise ValueError(
                    f"profiler observation digest collision for {digest}"
                )

    assert registry_digests
    assert source_feature_schema_versions
    assert collector_version is not None
    assert request_schema_version is not None
    if not observation_by_digest:
        raise ValueError("profiler evidence composition has no completed launch witnesses")

    composed_corpus_digest = _sha256_json({
        "schema_version": "native-vnni-composed-profiler-corpus-v1",
        "sources": [
            {
                "request_manifest_digest": request_digest,
                "evidence_manifest_digest": evidence_digest,
                "witness_corpus_digest": witness_digest,
            }
            for request_digest, evidence_digest, witness_digest in sorted(
                source_provenance
            )
        ],
    })
    composed_registry_digest = _sha256_json({
        "schema_version": "native-vnni-composed-candidate-registry-v1",
        "source_registry_digests": sorted(registry_digests),
    })
    composed_requests = ProfilerRequestManifest(
        corpus_digest=composed_corpus_digest,
        candidate_registry_digest=composed_registry_digest,
        requests=tuple(request_by_id[key] for key in sorted(request_by_id)),
        learner_version=LEARNER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        schema_version=request_schema_version,
    )
    composed_evidence = ProfilerEvidenceManifest(
        request_manifest_digest=composed_requests.digest(),
        corpus_digest=composed_requests.corpus_digest,
        candidate_registry_digest=composed_requests.candidate_registry_digest,
        evidence=tuple(evidence_by_id[key] for key in sorted(evidence_by_id)),
        collector_version=collector_version,
    )
    witnesses = ObservationCorpus._from_validated(
        observation_by_digest[key] for key in sorted(observation_by_digest)
    )
    validate_profiler_evidence_coverage(
        composed_requests,
        composed_evidence,
        require_complete=True,
    )
    _validate_feature_observation_join(
        witnesses,
        composed_requests,
        frozenset(observation_by_digest),
    )
    return ComposedProfilerEvidence(
        observations=witnesses,
        requests=composed_requests,
        evidence=composed_evidence,
    )


@dataclass(frozen=True)
class ProfilerFeatureRow:
    """One physical dispatch joined to its canonical timing observation.

    Candidate latency belongs to the complete production candidate pipeline,
    while profiler metrics belong to one ordered physical dispatch in that
    pipeline.  Keeping both levels explicit prevents a multi-kernel candidate
    from accidentally treating one dispatch duration as end-to-end latency.
    """

    observation: NativeVNNIObservation
    request_id: str
    observation_digest: str
    candidate_registry_surface: str
    schedule_signature: str
    workspace_signature: str
    dispatch_count: int
    dispatch_index: int
    dispatch_kind: ProfiledDispatchKind
    kernel_name: str
    kernel_fingerprint: str
    grid: tuple[int, int, int] | None
    block: tuple[int, int, int] | None
    metric_values: dict[str, float | None]
    metric_availability: dict[str, str]

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a nested deterministic representation for model tooling."""

        return {
            "observation": self.observation.canonical_mapping(),
            "request_id": self.request_id,
            "observation_digest": self.observation_digest,
            "candidate_registry_surface": self.candidate_registry_surface,
            "schedule_signature": self.schedule_signature,
            "workspace_signature": self.workspace_signature,
            "dispatch_count": self.dispatch_count,
            "dispatch_index": self.dispatch_index,
            "dispatch_kind": self.dispatch_kind.value,
            "kernel_name": self.kernel_name,
            "kernel_fingerprint": self.kernel_fingerprint,
            "grid": None if self.grid is None else list(self.grid),
            "block": None if self.block is None else list(self.block),
            "metric_values": dict(sorted(self.metric_values.items())),
            "metric_availability": dict(
                sorted(self.metric_availability.items())
            ),
        }


def _validate_feature_observation_join(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    required_observation_digests: frozenset[str],
) -> dict[str, NativeVNNIObservation]:
    """Authenticate the canonical timing corpus used by a feature export.

    The profiler request manifest binds the complete observation corpus, but
    deliberately does not duplicate timing values.  Export therefore requires
    the original common-observation CSV and proves both corpus identity and the
    exact bounded-subset request derivation before exposing timing and profiler
    fields in one table. Request construction deliberately selects one
    representative per physical candidate rather than profiling every
    canonical timing row.
    """

    observations_by_digest = {
        observation.digest(): observation for observation in observations
    }
    if len(observations_by_digest) != len(observations):
        raise ValueError("profiler feature observations contain duplicate rows")
    request_digests = {request.observation_digest for request in requests.requests}
    full_corpus_authenticated = observations.digest() == requests.corpus_digest
    exact_request_witnesses = (
        set(observations_by_digest) == required_observation_digests
    )
    if not full_corpus_authenticated and not exact_request_witnesses:
        raise ValueError(
            "profiler feature observations belong to another timing corpus; "
            "a compact source must contain exactly the request observation "
            "witnesses"
        )
    if not required_observation_digests.issubset(observations_by_digest):
        raise ValueError(
            "profiler requests are not an authenticated timing-corpus subset"
        )
    for request in requests.requests:
        if request.observation_digest not in required_observation_digests:
            continue
        expected = profiler_request_for_observation(
            observations_by_digest[request.observation_digest]
        )
        if expected != request:
            raise ValueError(
                f"{request.request_id}: profiler request is not the exact "
                "derivative of its timing observation"
            )
    return observations_by_digest


def compact_profiler_observation_witnesses(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> ObservationCorpus:
    """Extract an exact, authenticated timing witness for retained evidence.

    Long-lived corpus bundles may retain a later aggregate that is a strict
    superset of the observations used to create an earlier profiler request
    manifest. Passing that aggregate directly to feature export would weaken
    its whole-corpus identity check. Reprofiling the same kernel candidates is
    unnecessary, however: each request already owns the cryptographic digest
    and complete semantic identity of its canonical timing observation.

    This function first authenticates profiler coverage, selects only timing
    rows referenced by completed evidence records, and then sends the compact
    corpus through the normal exact-witness validator. Missing, duplicated, or
    modified observations therefore remain hard failures. The returned corpus
    is suitable for immutable storage beside the request/evidence manifests and
    can be used by ``export-features`` without any provenance exception.
    """

    validate_profiler_evidence_coverage(
        requests, evidence, require_complete=require_complete
    )
    request_by_id = {request.request_id: request for request in requests.requests}
    required_observation_digests = frozenset(
        request_by_id[item.request_id].observation_digest
        for item in evidence.evidence
        if item.status == ProfilerEvidenceStatus.COMPLETE
    )
    observations_by_digest = {
        observation.digest(): observation for observation in observations
    }
    if len(observations_by_digest) != len(observations):
        raise ValueError("profiler witness source contains duplicate rows")
    missing = required_observation_digests.difference(observations_by_digest)
    if missing:
        raise ValueError(
            "profiler witness source omits requested timing observations: "
            f"{sorted(missing)}"
        )

    compact = ObservationCorpus._from_validated(
        observations_by_digest[digest]
        for digest in sorted(required_observation_digests)
    )
    _validate_feature_observation_join(
        compact,
        requests,
        required_observation_digests,
    )
    return compact


def read_profiler_feature_observation_witnesses(
    paths: Iterable[Path],
) -> ObservationCorpus:
    """Recover canonical timing witnesses embedded in exported feature CSVs.

    A profiler feature table repeats every common observation column so it can
    remain useful for offline model experiments without joining a second file.
    Multi-dispatch candidates repeat that observation once per physical kernel.
    This reader strips profiler-only columns, reconstructs the strict common
    observation, verifies the recorded observation digest, and deduplicates
    only byte-identical witnesses. It is therefore a provenance recovery path,
    not a way to trust arbitrary derived features or bypass request validation.
    """

    observations_by_digest: dict[str, NativeVNNIObservation] = {}
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or ())
            missing = set(OBSERVATION_COLUMNS).difference(fields)
            if missing:
                raise ValueError(
                    f"{path}: profiler feature table omits common observation "
                    f"columns: {sorted(missing)}"
                )
            if "profiler.observation_digest" not in fields:
                raise ValueError(
                    f"{path}: profiler feature table omits observation digest"
                )
            for row_number, raw in enumerate(reader, start=2):
                try:
                    observation = NativeVNNIObservation.from_mapping({
                        name: raw[name] for name in OBSERVATION_COLUMNS
                    })
                except (TypeError, ValueError) as error:
                    raise ValueError(f"{path}:{row_number}: {error}") from error
                digest = observation.digest()
                recorded_digest = raw["profiler.observation_digest"]
                if recorded_digest != digest:
                    raise ValueError(
                        f"{path}:{row_number}: profiler feature observation "
                        f"digest {recorded_digest!r} does not match {digest!r}"
                    )
                previous = observations_by_digest.setdefault(digest, observation)
                if previous.canonical_mapping() != observation.canonical_mapping():
                    raise ValueError(
                        f"{path}:{row_number}: duplicate profiler feature "
                        "observation digest has different content"
                    )
    return ObservationCorpus._from_validated(
        observations_by_digest[digest]
        for digest in sorted(observations_by_digest)
    )


def profiler_feature_rows(
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> tuple[ProfilerFeatureRow, ...]:
    """Project validated evidence into one row per physical candidate dispatch.

    This is an offline training/diagnostic join. Hardware counters are not
    available to a production resolver and therefore cannot become runtime
    policy predicates. Unsupported candidate requests legitimately produce no
    physical dispatch row; their explicit terminal evidence remains in the
    source manifest and in the coverage report.
    """

    validate_profiler_evidence_coverage(
        requests, evidence, require_complete=require_complete
    )
    request_by_id = {request.request_id: request for request in requests.requests}
    required_observation_digests = frozenset(
        request_by_id[item.request_id].observation_digest
        for item in evidence.evidence
        if item.status == ProfilerEvidenceStatus.COMPLETE
    )
    observations_by_digest = _validate_feature_observation_join(
        observations,
        requests,
        required_observation_digests,
    )
    rows = []
    for item in sorted(evidence.evidence, key=lambda record: record.request_id):
        if item.status != ProfilerEvidenceStatus.COMPLETE:
            continue
        request = request_by_id[item.request_id]
        observation = observations_by_digest[request.observation_digest]
        for dispatch in sorted(
            item.dispatches, key=lambda record: record.dispatch_index
        ):
            values = {metric.metric_id: metric.value for metric in dispatch.metrics}
            availability = {
                metric.metric_id: metric.availability.value
                for metric in dispatch.metrics
            }
            rows.append(ProfilerFeatureRow(
                observation=observation,
                request_id=request.request_id,
                observation_digest=request.observation_digest,
                candidate_registry_surface=request.candidate_registry_surface,
                schedule_signature=request.schedule_signature,
                workspace_signature=request.workspace_signature,
                dispatch_count=len(item.dispatches),
                dispatch_index=dispatch.dispatch_index,
                dispatch_kind=dispatch.dispatch_kind,
                kernel_name=dispatch.kernel_name,
                kernel_fingerprint=dispatch.kernel_fingerprint,
                grid=dispatch.grid,
                block=dispatch.block,
                metric_values=values,
                metric_availability=availability,
            ))
    return tuple(rows)


def write_profiler_feature_csv(
    path: Path,
    observations: ObservationCorpus,
    requests: ProfilerRequestManifest,
    evidence: ProfilerEvidenceManifest,
    *,
    require_complete: bool = True,
) -> None:
    """Write a flat dispatch-level feature table for offline model training."""

    rows = profiler_feature_rows(
        observations, requests, evidence, require_complete=require_complete
    )
    metric_ids = tuple(sorted({
        definition.metric_id
        for backend in Backend
        for definition in metric_definitions(backend)
    }))
    profiler_columns = (
        "profiler.request_id",
        "profiler.observation_digest",
        "profiler.candidate_registry_surface",
        "profiler.schedule_signature",
        "profiler.workspace_signature",
        "profiler.dispatch_count",
        "profiler.dispatch_index",
        "profiler.dispatch_kind",
        "profiler.kernel_name",
        "profiler.kernel_fingerprint",
        "profiler.grid",
        "profiler.block",
    )
    metric_columns = tuple(
        column
        for metric_id in metric_ids
        for column in (f"{metric_id}.value", f"{metric_id}.availability")
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*OBSERVATION_COLUMNS, *profiler_columns, *metric_columns),
        )
        writer.writeheader()
        for row in rows:
            mapping = row.observation.canonical_mapping()
            for name, value in tuple(mapping.items()):
                if isinstance(value, (dict, list)):
                    mapping[name] = json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    )
                elif value is None:
                    mapping[name] = ""
            mapping.update({
                "profiler.request_id": row.request_id,
                "profiler.observation_digest": row.observation_digest,
                "profiler.candidate_registry_surface": (
                    row.candidate_registry_surface
                ),
                "profiler.schedule_signature": row.schedule_signature,
                "profiler.workspace_signature": row.workspace_signature,
                "profiler.dispatch_count": row.dispatch_count,
                "profiler.dispatch_index": row.dispatch_index,
                "profiler.dispatch_kind": row.dispatch_kind.value,
                "profiler.kernel_name": row.kernel_name,
                "profiler.kernel_fingerprint": row.kernel_fingerprint,
                "profiler.grid": json.dumps(row.grid, separators=(",", ":")),
                "profiler.block": json.dumps(row.block, separators=(",", ":")),
            })
            for metric_id in metric_ids:
                value = row.metric_values.get(metric_id)
                mapping[f"{metric_id}.value"] = "" if value is None else value
                mapping[f"{metric_id}.availability"] = (
                    row.metric_availability.get(metric_id, "not_applicable")
                )
            writer.writerow(mapping)


def _emit_requests(args: argparse.Namespace) -> int:
    """CLI implementation for deriving a request sidecar from timing CSVs."""

    corpus = read_observation_csv(Path(path) for path in args.observation)
    manifest = build_profiler_request_manifest(corpus)
    write_profiler_request_manifest(Path(args.output), manifest)
    print(
        json.dumps(
            {
                "manifest_digest": manifest.digest(),
                "request_count": len(manifest.requests),
                "required_count": sum(
                    request.profile_required for request in manifest.requests
                ),
                "output": args.output,
            },
            sort_keys=True,
        )
    )
    return 0


def _emit_missing_requests(args: argparse.Namespace) -> int:
    """CLI implementation for one resumable exact-point delta transaction."""

    corpus = read_observation_csv(Path(path) for path in args.observation)
    covered = tuple(
        read_profiler_request_manifest(Path(path))
        for path in args.covered_requests
    )
    observations, requests = build_missing_profiler_request_transaction(
        corpus,
        covered,
    )
    if observations is None or requests is None:
        print(json.dumps({
            "status": "complete",
            "missing_request_count": 0,
        }, sort_keys=True))
        return 0

    observation_output = Path(args.output_observation)
    request_output = Path(args.output_requests)
    observation_output.parent.mkdir(parents=True, exist_ok=True)
    request_output.parent.mkdir(parents=True, exist_ok=True)
    write_observation_csv(observation_output, observations)
    write_profiler_request_manifest(request_output, requests)
    print(json.dumps({
        "status": "delta-required",
        "missing_request_count": len(requests.requests),
        "profile_required_count": sum(
            request.profile_required for request in requests.requests
        ),
        "observation_output": str(observation_output),
        "request_output": str(request_output),
        "request_manifest_digest": requests.digest(),
    }, sort_keys=True))
    return 0


def _count_uncovered_requests(args: argparse.Namespace) -> int:
    """CLI implementation for checking one durable delta transaction."""

    requests = read_profiler_request_manifest(Path(args.requests))
    covered = tuple(
        read_profiler_request_manifest(Path(path))
        for path in args.covered_requests
    )
    print(count_uncovered_profiler_requests(requests, covered))
    return 0


def _validate_evidence(args: argparse.Namespace) -> int:
    """CLI implementation for authenticating a profiler evidence join."""

    requests = read_profiler_request_manifest(Path(args.requests))
    evidence = read_profiler_evidence_manifest(Path(args.evidence))
    report = validate_profiler_evidence_coverage(
        requests, evidence, require_complete=not args.allow_incomplete
    )
    print(json.dumps(asdict(report), sort_keys=True))
    return 0


def _export_features(args: argparse.Namespace) -> int:
    """CLI implementation for the dispatch-level offline feature table."""

    observations = read_observation_csv(Path(path) for path in args.observation)
    requests = read_profiler_request_manifest(Path(args.requests))
    evidence = read_profiler_evidence_manifest(Path(args.evidence))
    output = Path(args.output)
    write_profiler_feature_csv(
        output,
        observations,
        requests,
        evidence,
        require_complete=not args.allow_incomplete,
    )
    print(json.dumps({"output": str(output)}, sort_keys=True))
    return 0


def _compact_witnesses(args: argparse.Namespace) -> int:
    """CLI implementation for publishing an exact profiler timing witness."""

    observations = (
        read_observation_csv(Path(path) for path in args.observation)
        if args.observation
        else read_profiler_feature_observation_witnesses(
            Path(path) for path in args.feature_table
        )
    )
    requests = read_profiler_request_manifest(Path(args.requests))
    evidence = read_profiler_evidence_manifest(Path(args.evidence))
    compact = compact_profiler_observation_witnesses(
        observations,
        requests,
        evidence,
        require_complete=not args.allow_incomplete,
    )
    output = Path(args.output)
    write_observation_csv(output, compact.observations)
    print(json.dumps({
        "corpus_digest": compact.digest(),
        "observation_count": len(compact),
        "output": str(output),
    }, sort_keys=True))
    return 0


def _compose_evidence(args: argparse.Namespace) -> int:
    """CLI implementation for unioning authenticated additive transactions."""

    source_counts = {
        len(args.source_observation),
        len(args.source_requests),
        len(args.source_evidence),
    }
    if len(source_counts) != 1:
        raise ValueError(
            "compose-evidence requires one observation, request, and evidence "
            "path per source transaction"
        )
    composed = compose_profiler_evidence(
        (
            read_observation_csv((Path(observation_path),)),
            read_profiler_request_manifest(Path(request_path)),
            read_profiler_evidence_manifest(Path(evidence_path)),
        )
        for observation_path, request_path, evidence_path in zip(
            args.source_observation,
            args.source_requests,
            args.source_evidence,
            strict=True,
        )
    )
    output_observations = Path(args.output_observation)
    output_requests = Path(args.output_requests)
    output_evidence = Path(args.output_evidence)
    write_observation_csv(output_observations, composed.observations.observations)
    write_profiler_request_manifest(output_requests, composed.requests)
    write_profiler_evidence_manifest(output_evidence, composed.evidence)
    print(json.dumps({
        "evidence_count": len(composed.evidence.evidence),
        "evidence_manifest_digest": composed.evidence.digest(),
        "observation_count": len(composed.observations),
        "output_evidence": str(output_evidence),
        "output_observation": str(output_observations),
        "output_requests": str(output_requests),
        "request_count": len(composed.requests.requests),
        "request_manifest_digest": composed.requests.digest(),
    }, sort_keys=True))
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the small sidecar generation/validation command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    emit = subparsers.add_parser(
        "emit-requests",
        help="derive immutable isolated-profile requests from common observations",
    )
    emit.add_argument("--observation", action="append", required=True)
    emit.add_argument("--output", required=True)
    emit.set_defaults(handler=_emit_requests)

    missing = subparsers.add_parser(
        "emit-missing-requests",
        help=(
            "derive only exact physical launches absent from prior request "
            "transactions"
        ),
    )
    missing.add_argument("--observation", action="append", required=True)
    missing.add_argument("--covered-requests", action="append", required=True)
    missing.add_argument("--output-observation", required=True)
    missing.add_argument("--output-requests", required=True)
    missing.set_defaults(handler=_emit_missing_requests)

    count_uncovered = subparsers.add_parser(
        "count-uncovered-requests",
        help=(
            "print how many exact physical launches in one request manifest "
            "are absent from prior request transactions"
        ),
    )
    count_uncovered.add_argument("--requests", required=True)
    count_uncovered.add_argument(
        "--covered-requests", action="append", required=True
    )
    count_uncovered.set_defaults(handler=_count_uncovered_requests)

    validate = subparsers.add_parser(
        "validate-evidence",
        help="authenticate profiler evidence and enforce one result per request",
    )
    validate.add_argument("--requests", required=True)
    validate.add_argument("--evidence", required=True)
    validate.add_argument("--allow-incomplete", action="store_true")
    validate.set_defaults(handler=_validate_evidence)

    compact = subparsers.add_parser(
        "compact-witnesses",
        help=(
            "extract exact request timing witnesses from a retained observation "
            "superset"
        ),
    )
    compact_source = compact.add_mutually_exclusive_group(required=True)
    compact_source.add_argument("--observation", action="append")
    compact_source.add_argument(
        "--feature-table",
        action="append",
        help=(
            "recover canonical timing rows embedded in an existing profiler "
            "feature CSV"
        ),
    )
    compact.add_argument("--requests", required=True)
    compact.add_argument("--evidence", required=True)
    compact.add_argument("--output", required=True)
    compact.add_argument("--allow-incomplete", action="store_true")
    compact.set_defaults(handler=_compact_witnesses)

    compose = subparsers.add_parser(
        "compose-evidence",
        help=(
            "compose complete additive profiler transactions without "
            "relaunching their physical candidates"
        ),
    )
    compose.add_argument("--source-observation", action="append", required=True)
    compose.add_argument("--source-requests", action="append", required=True)
    compose.add_argument("--source-evidence", action="append", required=True)
    compose.add_argument("--output-observation", required=True)
    compose.add_argument("--output-requests", required=True)
    compose.add_argument("--output-evidence", required=True)
    compose.set_defaults(handler=_compose_evidence)

    export = subparsers.add_parser(
        "export-features",
        help="write one offline training row per physical candidate dispatch",
    )
    export.add_argument("--observation", action="append", required=True)
    export.add_argument("--requests", required=True)
    export.add_argument("--evidence", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-incomplete", action="store_true")
    export.set_defaults(handler=_export_features)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the profiler sidecar command line."""

    args = build_argument_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
