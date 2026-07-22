"""Strict common observation schema for NativeVNNI dispatch training.

One row is an aggregate measurement for one explicit candidate, source-format
alias, runtime key, execution mode, and deterministic payload trial set. The
schema intentionally requires correctness and route evidence even when a row
will later be rejected; an empty field is never interpreted as a pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field, fields
from decimal import Decimal, InvalidOperation
from enum import Enum
from functools import cached_property
from typing import Any, Mapping

from .format_registry import format_spec


SCHEMA_VERSION = 1
POLICY_ABI = 2
DEFAULT_P95_REGRET_PERCENT = Decimal("5")
DEFAULT_MINIMUM_PASSING_DOMAIN_PERCENT = Decimal("95")


def _promotion_percent(
    environment_name: str,
    default: Decimal,
    *,
    allow_zero: bool,
) -> Decimal:
    """Parse one process-wide promotion percentage without silent clipping.

    The turnkey shell transaction exports these values before launching any
    fitter, seal collector, artifact writer, or artifact validator. Keeping
    the parser here gives every backend exactly one interpretation of a manual
    best-effort threshold while leaving timing, correctness, coverage, and
    bitwise-verifier gates untouched.
    """

    raw = os.environ.get(environment_name, str(default)).strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as error:
        raise ValueError(
            f"{environment_name} must be a finite decimal percentage"
        ) from error
    lower_bound = Decimal("0") if allow_zero else Decimal("0.000001")
    if not value.is_finite() or value < lower_bound or value > Decimal("100"):
        interval = "[0, 100]" if allow_zero else "(0, 100]"
        raise ValueError(f"{environment_name} must be in {interval}")
    return value


P95_REGRET_PERCENT = _promotion_percent(
    "LLAMINAR_NATIVE_VNNI_PROMOTION_P95_REGRET_PERCENT",
    DEFAULT_P95_REGRET_PERCENT,
    allow_zero=False,
)
MINIMUM_PASSING_DOMAIN_PERCENT = _promotion_percent(
    "LLAMINAR_NATIVE_VNNI_PROMOTION_MIN_PASSING_DOMAIN_PERCENT",
    DEFAULT_MINIMUM_PASSING_DOMAIN_PERCENT,
    allow_zero=True,
)
P95_REGRET_BUDGET = float(P95_REGRET_PERCENT / Decimal("100"))
MINIMUM_PASSING_DOMAIN_FRACTION = float(
    MINIMUM_PASSING_DOMAIN_PERCENT / Decimal("100")
)
LEARNER_VERSION = "native-vnni-bounded-tree-beam-regret-v26"
COMPATIBLE_OBSERVATION_LEARNER_VERSIONS = frozenset((
    "native-vnni-bounded-tree-beam-regret-v8",
    "native-vnni-bounded-tree-beam-regret-v9",
    "native-vnni-bounded-tree-beam-regret-v10",
    "native-vnni-bounded-tree-beam-regret-v11",
    "native-vnni-bounded-tree-beam-regret-v12",
    "native-vnni-bounded-tree-beam-regret-v13",
    "native-vnni-bounded-tree-beam-regret-v14",
    "native-vnni-bounded-tree-beam-regret-v15",
    "native-vnni-bounded-tree-beam-regret-v16",
    "native-vnni-bounded-tree-beam-regret-v17",
    "native-vnni-bounded-tree-beam-regret-v18",
    "native-vnni-bounded-tree-beam-regret-v19",
    "native-vnni-bounded-tree-beam-regret-v20",
    "native-vnni-bounded-tree-beam-regret-v21",
    "native-vnni-bounded-tree-beam-regret-v23",
    "native-vnni-bounded-tree-beam-regret-v24",
    "native-vnni-bounded-tree-beam-regret-v25",
    LEARNER_VERSION,
))
FEATURE_SCHEMA_VERSION = "execution-mode-n-k-work-aspect-tile-wave-tree-v11"
COMPATIBLE_PROFILER_FEATURE_SCHEMA_VERSIONS = frozenset((
    "execution-mode-n-k-work-aspect-tile-occupancy-tree-v5",
    "execution-mode-n-k-work-aspect-tile-wave-tree-v6",
    "execution-mode-n-k-work-aspect-tile-wave-tree-v7",
    "execution-mode-n-k-work-aspect-tile-wave-tree-v8",
    "execution-mode-n-k-work-aspect-tile-wave-tree-v9",
    "execution-mode-n-k-work-aspect-tile-wave-tree-v10",
    FEATURE_SCHEMA_VERSION,
))


class Backend(str, Enum):
    """Production backend whose candidate launcher produced an observation."""

    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"


class SemanticContract(str, Enum):
    """Numerical contract under which a candidate may be selected."""

    FAST = "Fast"
    VERIFIER_SERIAL_M1_BITWISE = "VerifierSerialM1Bitwise"


class ExecutionMode(str, Enum):
    """Production execution mode exercised by the observation."""

    EAGER = "eager"
    GRAPH_CAPTURED = "graph_captured"


class AspectBucket(str, Enum):
    """Version-1 backend-neutral aggregate-N to K aspect bucket."""

    VERY_WIDE = "very_wide"
    WIDE = "wide"
    BALANCED = "balanced"
    TALL = "tall"


def classify_aspect(aggregate_n: int, k: int) -> AspectBucket:
    """Classify one shape with the normative v1 16/2/0.75 boundaries."""

    if aggregate_n <= 0 or k <= 0:
        raise ValueError(f"aspect dimensions must be positive, got N={aggregate_n} K={k}")
    if aggregate_n >= 16 * k:
        return AspectBucket.VERY_WIDE
    if aggregate_n >= 2 * k:
        return AspectBucket.WIDE
    # Compare N/K >= 3/4 without introducing floating-point boundary drift.
    if 4 * aggregate_n >= 3 * k:
        return AspectBucket.BALANCED
    return AspectBucket.TALL


def _required_text(name: str, value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must not be empty")
    return text


def _parse_bool(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean, got {value!r}")


def _parse_json(name: str, value: Any, expected_type: type) -> Any:
    if isinstance(value, expected_type):
        return value
    try:
        decoded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON") from exc
    if not isinstance(decoded, expected_type):
        raise ValueError(f"{name} must decode to {expected_type.__name__}")
    return decoded


@dataclass(frozen=True)
class NativeVNNIObservation:
    """One complete, auditable candidate observation on a runtime surface."""

    # Schema and build provenance.
    schema_version: int
    run_id: str
    corpus_id: str
    git_revision: str
    build_id: str
    compiler_id: str
    policy_abi: int
    learner_version: str

    # Hardware identity.
    backend: Backend
    architecture_class: str
    device_name: str
    driver_runtime: str
    threading_or_stream_mode: str

    # Runtime key and source alias.
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

    # Generic learner features.
    aspect_ratio: float
    aspect_bucket: AspectBucket
    work_items: int
    n_tail_class: str
    k_tail_class: str
    alignment_class: str

    # Candidate identity and support.
    candidate_id: str
    effective_candidate_id: str
    candidate_family: str
    config_json: dict[str, Any]
    supported: bool
    graph_capture_ok: bool
    generic_eligible: bool

    # Arithmetic and serial-oracle identity.
    arithmetic_fingerprint: str
    serial_m1_policy_id: str
    serial_m1_policy_hash: str
    candidate_policy_hash: str
    ordered_reduction: bool
    uses_atomic_reduction: bool

    # Correctness evidence.
    trial_set_hash: str
    numerical_correctness: bool
    bitwise_equal: bool
    repeat_equal: bool
    mismatch_count: int
    first_mismatch_index: int | None
    grouped_output_digest: str
    serial_output_digest: str
    max_abs: float
    relative_l2: float
    cosine: float
    symmetric_kld: float

    # Robust aggregate timing and raw-sample provenance.
    warmup_count: int
    sample_count: int
    min_us: float
    median_us: float
    p95_us: float
    mad_us: float
    cv: float
    timing_sample_hash: str
    effective_bandwidth_gbs: float

    # Measured-route proof.
    forced_route_ok: bool
    observed_candidate_id: str
    route_counter_ok: bool
    workspace_ok: bool
    explicit_stream_ok: bool

    # Optional structured proof for reviewed adaptive timing protocols. Empty
    # preserves the original fixed-sample schema and its corpus/request digests.
    launch_k_tiles: int = 0
    launch_n_block_chunks: int = 0
    adaptive_timing_evidence: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject malformed, ambiguous, or runtime-inexpressible observations."""

        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version={self.schema_version}; expected {SCHEMA_VERSION}"
            )
        if self.policy_abi != POLICY_ABI:
            raise ValueError(f"unsupported policy_abi={self.policy_abi}; expected {POLICY_ABI}")
        if self.learner_version not in COMPATIBLE_OBSERVATION_LEARNER_VERSIONS:
            raise ValueError(
                f"unsupported learner_version={self.learner_version!r}; "
                "accepted observation versions are "
                f"{sorted(COMPATIBLE_OBSERVATION_LEARNER_VERSIONS)!r}"
            )

        for name in (
            "run_id", "corpus_id", "git_revision", "build_id", "compiler_id",
            "architecture_class", "device_name", "driver_runtime",
            "threading_or_stream_mode", "operation_kind", "bundle_signature",
            "prepared_family_id", "packing_abi", "shape_group_id", "shape_name",
            "n_tail_class", "k_tail_class", "alignment_class", "candidate_id",
            "effective_candidate_id", "candidate_family", "arithmetic_fingerprint",
            "serial_m1_policy_id", "serial_m1_policy_hash", "candidate_policy_hash",
            "trial_set_hash", "grouped_output_digest", "serial_output_digest",
            "timing_sample_hash", "observed_candidate_id",
        ):
            _required_text(name, getattr(self, name))

        if self.candidate_id.strip().upper() == "AUTO":
            raise ValueError("AUTO is not an explicit forceable candidate_id")
        if self.observed_candidate_id.strip().upper() == "AUTO":
            raise ValueError("observed_candidate_id must name the effective production route")
        if self.m <= 0 or self.aggregate_n <= 0 or self.k <= 0:
            raise ValueError("m, aggregate_n, and k must be positive")
        if self.launch_k_tiles < 0:
            raise ValueError("launch_k_tiles must be non-negative")
        if self.launch_n_block_chunks < 0:
            raise ValueError("launch_n_block_chunks must be non-negative")
        if not self.projection_n_vector or any(value <= 0 for value in self.projection_n_vector):
            raise ValueError("projection_n_vector must contain positive dimensions")
        if sum(self.projection_n_vector) != self.aggregate_n:
            raise ValueError(
                "aggregate_n must equal the sum of the ordered projection_n_vector"
            )
        if self.work_items != self.aggregate_n * self.k:
            raise ValueError("work_items does not match aggregate_n * k")
        expected_aspect = float(self.aggregate_n) / float(self.k)
        if not math.isclose(self.aspect_ratio, expected_aspect, rel_tol=1.0e-12, abs_tol=0.0):
            raise ValueError("aspect_ratio does not match aggregate_n / k")
        if self.aspect_bucket != classify_aspect(self.aggregate_n, self.k):
            raise ValueError("aspect_bucket does not match the v1 feature policy")

        spec = format_spec(self.source_format)
        if self.source_codebook_id != spec.source_codebook_id:
            raise ValueError(
                f"source codebook mismatch for {spec.label}: "
                f"expected {spec.source_codebook_id}, got {self.source_codebook_id}"
            )
        expected_runtime = spec.runtime_codebook(self.backend.value)
        if self.runtime_codebook_id != expected_runtime:
            raise ValueError(
                f"runtime codebook mismatch for {self.backend.value}/{spec.label}: "
                f"expected {expected_runtime}, got {self.runtime_codebook_id}"
            )
        if self.prepared_family_id != spec.prepared_family(self.backend.value):
            raise ValueError("prepared_family_id does not match the canonical registry")
        if self.packing_abi != spec.packing_abi(self.backend.value):
            raise ValueError("packing_abi does not match the canonical registry")

        if self.mismatch_count < 0:
            raise ValueError("mismatch_count must be non-negative")
        if self.bitwise_equal and self.mismatch_count != 0:
            raise ValueError("bitwise_equal cannot accompany a nonzero mismatch_count")
        if not self.bitwise_equal and self.mismatch_count == 0:
            raise ValueError("a failed byte comparison must report at least one mismatch")
        if self.mismatch_count == 0 and self.first_mismatch_index is not None:
            raise ValueError("first_mismatch_index must be null when no mismatch exists")
        if self.mismatch_count > 0 and self.first_mismatch_index is None:
            raise ValueError("first_mismatch_index is required for a byte mismatch")
        if self.warmup_count < 0 or self.sample_count <= 0:
            raise ValueError("timing counts are invalid")
        if not isinstance(self.adaptive_timing_evidence, dict):
            raise ValueError("adaptive_timing_evidence must be a JSON object")
        if any(not str(name).strip() for name in self.adaptive_timing_evidence):
            raise ValueError("adaptive timing evidence keys must be non-empty")
        if self.min_us <= 0.0 or self.median_us <= 0.0 or self.p95_us <= 0.0:
            raise ValueError("timing values must be positive")
        if self.min_us > self.median_us or self.median_us > self.p95_us:
            raise ValueError("expected min_us <= median_us <= p95_us")
        if self.mad_us < 0.0 or self.cv < 0.0:
            raise ValueError("timing dispersion values must be non-negative")

    def canonical_mapping(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible flat representation."""

        # NativeVNNIObservation is frozen and structured members are immutable
        # by contract. Avoid recursive deepcopy while authenticating large
        # profiler corpora; explicit JSON normalization below remains unchanged.
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        result["backend"] = self.backend.value
        result["semantic_contract"] = self.semantic_contract.value
        result["execution_mode"] = self.execution_mode.value
        result["aspect_bucket"] = self.aspect_bucket.value
        result["projection_n_vector"] = list(self.projection_n_vector)
        # Empty evidence is the backward-compatible fixed-sample representation.
        # Omitting it from digests keeps pre-extension fixed timing corpora and
        # their profiler request IDs reusable without a migration.
        if self.launch_k_tiles == 0:
            result.pop("launch_k_tiles")
        if self.launch_n_block_chunks == 0:
            result.pop("launch_n_block_chunks")
        if not self.adaptive_timing_evidence:
            result.pop("adaptive_timing_evidence")
        return result

    def digest(self) -> str:
        """Hash the complete observation for corpus identity and deduplication."""

        return self._cached_digest

    @cached_property
    def _cached_canonical_json(self) -> str:
        """Serialize the frozen observation once for every identity consumer.

        Per-row profiler joins and whole-corpus authentication use the same
        canonical JSON bytes. Keeping that immutable serialization beside the
        value prevents a production fit from encoding roughly eighty fields once
        for the observation digest and again for the corpus digest. The property
        is not a dataclass field, so it cannot enter ``asdict()`` or alter the
        historical hash representation.
        """

        return json.dumps(
            self.canonical_mapping(), sort_keys=True, separators=(",", ":")
        )

    @cached_property
    def _cached_digest(self) -> str:
        """Compute the immutable observation identity once per parsed row.

        Production corpora repeatedly authenticate the same row while they
        compact profiler witnesses, compose additive transactions, and join
        profiler features. ``NativeVNNIObservation`` is a frozen value object,
        so serializing its roughly eighty fields on every one of those passes
        only burns preprocessing time. ``cached_property`` stores no dataclass
        field and therefore cannot alter ``asdict()``, CSV output, or the
        historical digest bytes.

        Structured members such as ``config_json`` are part of the frozen
        value contract: callers must create a replacement observation rather
        than mutate a nested dictionary after construction. All corpus readers
        validate and then transfer exclusive immutable ownership accordingly.
        """

        encoded = self._cached_canonical_json.encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "NativeVNNIObservation":
        """Parse one CSV/JSON mapping and reject every missing required field."""

        optional = {
            "launch_k_tiles",
            "launch_n_block_chunks",
            "adaptive_timing_evidence",
        }
        missing = [
            name
            for name in cls.__dataclass_fields__
            if name not in raw and name not in optional
        ]
        if missing:
            raise ValueError(f"observation is missing required fields: {missing}")

        first_mismatch_raw = raw["first_mismatch_index"]
        first_mismatch = (
            None
            if first_mismatch_raw is None or str(first_mismatch_raw).strip().lower() in {"", "null", "none"}
            else int(first_mismatch_raw)
        )
        projection_vector = _parse_json(
            "projection_n_vector", raw["projection_n_vector"], list
        )
        config = _parse_json("config_json", raw["config_json"], dict)
        adaptive_raw = raw.get("adaptive_timing_evidence", "")
        adaptive_timing_evidence = (
            {}
            if adaptive_raw is None or str(adaptive_raw).strip() == ""
            else _parse_json(
                "adaptive_timing_evidence", adaptive_raw, dict
            )
        )

        observation = cls(
            schema_version=int(raw["schema_version"]),
            run_id=_required_text("run_id", raw["run_id"]),
            corpus_id=_required_text("corpus_id", raw["corpus_id"]),
            git_revision=_required_text("git_revision", raw["git_revision"]),
            build_id=_required_text("build_id", raw["build_id"]),
            compiler_id=_required_text("compiler_id", raw["compiler_id"]),
            policy_abi=int(raw["policy_abi"]),
            learner_version=_required_text("learner_version", raw["learner_version"]),
            backend=Backend(str(raw["backend"]).strip().lower()),
            architecture_class=_required_text("architecture_class", raw["architecture_class"]),
            device_name=_required_text("device_name", raw["device_name"]),
            driver_runtime=_required_text("driver_runtime", raw["driver_runtime"]),
            threading_or_stream_mode=_required_text(
                "threading_or_stream_mode", raw["threading_or_stream_mode"]
            ),
            semantic_contract=SemanticContract(str(raw["semantic_contract"]).strip()),
            operation_kind=_required_text("operation_kind", raw["operation_kind"]),
            bundle_signature=_required_text("bundle_signature", raw["bundle_signature"]),
            projection_n_vector=tuple(int(value) for value in projection_vector),
            source_format=_required_text("source_format", raw["source_format"]).upper(),
            source_codebook_id=int(raw["source_codebook_id"]),
            prepared_family_id=_required_text("prepared_family_id", raw["prepared_family_id"]),
            packing_abi=_required_text("packing_abi", raw["packing_abi"]),
            runtime_codebook_id=int(raw["runtime_codebook_id"]),
            shape_group_id=_required_text("shape_group_id", raw["shape_group_id"]),
            shape_name=_required_text("shape_name", raw["shape_name"]),
            execution_mode=ExecutionMode(str(raw["execution_mode"]).strip()),
            m=int(raw["m"]),
            aggregate_n=int(raw["aggregate_n"]),
            k=int(raw["k"]),
            aspect_ratio=float(raw["aspect_ratio"]),
            aspect_bucket=AspectBucket(str(raw["aspect_bucket"]).strip()),
            work_items=int(raw["work_items"]),
            n_tail_class=_required_text("n_tail_class", raw["n_tail_class"]),
            k_tail_class=_required_text("k_tail_class", raw["k_tail_class"]),
            alignment_class=_required_text("alignment_class", raw["alignment_class"]),
            candidate_id=_required_text("candidate_id", raw["candidate_id"]),
            effective_candidate_id=_required_text(
                "effective_candidate_id", raw["effective_candidate_id"]
            ),
            candidate_family=_required_text("candidate_family", raw["candidate_family"]),
            config_json=config,
            supported=_parse_bool("supported", raw["supported"]),
            graph_capture_ok=_parse_bool("graph_capture_ok", raw["graph_capture_ok"]),
            generic_eligible=_parse_bool("generic_eligible", raw["generic_eligible"]),
            arithmetic_fingerprint=_required_text(
                "arithmetic_fingerprint", raw["arithmetic_fingerprint"]
            ),
            serial_m1_policy_id=_required_text("serial_m1_policy_id", raw["serial_m1_policy_id"]),
            serial_m1_policy_hash=_required_text(
                "serial_m1_policy_hash", raw["serial_m1_policy_hash"]
            ),
            candidate_policy_hash=_required_text(
                "candidate_policy_hash", raw["candidate_policy_hash"]
            ),
            ordered_reduction=_parse_bool("ordered_reduction", raw["ordered_reduction"]),
            uses_atomic_reduction=_parse_bool(
                "uses_atomic_reduction", raw["uses_atomic_reduction"]
            ),
            trial_set_hash=_required_text("trial_set_hash", raw["trial_set_hash"]),
            numerical_correctness=_parse_bool(
                "numerical_correctness", raw["numerical_correctness"]
            ),
            bitwise_equal=_parse_bool("bitwise_equal", raw["bitwise_equal"]),
            repeat_equal=_parse_bool("repeat_equal", raw["repeat_equal"]),
            mismatch_count=int(raw["mismatch_count"]),
            first_mismatch_index=first_mismatch,
            grouped_output_digest=_required_text(
                "grouped_output_digest", raw["grouped_output_digest"]
            ),
            serial_output_digest=_required_text(
                "serial_output_digest", raw["serial_output_digest"]
            ),
            max_abs=float(raw["max_abs"]),
            relative_l2=float(raw["relative_l2"]),
            cosine=float(raw["cosine"]),
            symmetric_kld=float(raw["symmetric_kld"]),
            warmup_count=int(raw["warmup_count"]),
            sample_count=int(raw["sample_count"]),
            min_us=float(raw["min_us"]),
            median_us=float(raw["median_us"]),
            p95_us=float(raw["p95_us"]),
            mad_us=float(raw["mad_us"]),
            cv=float(raw["cv"]),
            timing_sample_hash=_required_text(
                "timing_sample_hash", raw["timing_sample_hash"]
            ),
            effective_bandwidth_gbs=float(raw["effective_bandwidth_gbs"]),
            forced_route_ok=_parse_bool("forced_route_ok", raw["forced_route_ok"]),
            observed_candidate_id=_required_text(
                "observed_candidate_id", raw["observed_candidate_id"]
            ),
            route_counter_ok=_parse_bool("route_counter_ok", raw["route_counter_ok"]),
            workspace_ok=_parse_bool("workspace_ok", raw["workspace_ok"]),
            explicit_stream_ok=_parse_bool("explicit_stream_ok", raw["explicit_stream_ok"]),
            launch_k_tiles=int(raw.get("launch_k_tiles") or 0),
            launch_n_block_chunks=int(
                raw.get("launch_n_block_chunks") or 0
            ),
            adaptive_timing_evidence=adaptive_timing_evidence,
        )
        observation.validate()
        return observation


OBSERVATION_COLUMNS = tuple(NativeVNNIObservation.__dataclass_fields__)
