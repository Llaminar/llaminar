#!/usr/bin/env python3
"""Run the complete real-weight production parity matrix and measure its SLA.

The C++ `ProductionParity` case is the atomic matrix cell: it constructs one
production runner, executes full stage-by-stage prefill parity, validates graph-
stable snapshot publication, resets request data, and executes incremental
decode parity. CTest combines all precision cells for one test type/backend in
one process so immutable model and prepared-weight ownership is amortized while
runner, arena, graph, stream, and KV state remain exact per cell.

The performance target belongs to the whole selected matrix. Before model
staging, the driver builds and runs the complete CMake-owned ``V2_Unit_*``
suite, then runs the ``ProductionParityPreflight`` label: a model-free
integration gate for MPI lifecycle, orchestration, graph, stream/event,
collective, and movement invariants. It then prepares the model
fixture once, capacity-checks and atomically stages the complete selected GGUF
corpus into tmpfs, gives CPU, CUDA, and ROCm exclusive resource identities, and
overlaps only campaigns whose backend sets are disjoint. Every production child
then consumes the exact immutable staged paths declared by the typed campaign.
Staging is private and run-scoped by default. An explicit persistent-cache
directory instead retains atomically published, source-identity-bound read-only
GGUFs for rapid iteration; the driver never removes that cache. Model-byte
hashing is deliberately absent: the copy transaction proves its exact byte
count and stable source/destination identities, while the numerical checkpoints
are the authoritative proof that those weights match the reference.
Hybrid campaigns claim every backend they name. Crossing the target is recorded
as a performance failure after the complete matrix has run; it never stops
campaign admission or truncates correctness evidence. A correctness red stops
its GTest aggregate immediately, publishes one first-failure identity, cancels
already-running disjoint-backend siblings, and prevents further admission while
preserving completed and failing evidence. A separate completion timeout bounds
setup phases that do not publish exact-cell progress. Every exact GTest matrix
cell instead has one independent ten-minute progress deadline. The driver observes the fresh
per-cell log publication already owned by the artifact contract, so this
watchdog preserves one-process model-context amortization and still terminates
a stuck CTest/MPI process group with the exact cell identity. No campaign gets
an independent hour and no concurrent launch may oversubscribe a backend.

CTest/GTest registration remains the matrix source of truth.  The driver never
copies model, topology, backend, or precision tables into another manifest.
Use ``--list`` for a deterministic coverage inventory without running models.
Use ``--stage-models-only`` with an explicit persistent cache to prepare that
same identity-bound corpus for repeated focused production-path debugging.
Individual fixup invocations may reuse a passed preflight report only while the
Ninja build log and CTest registration remain older than that report. This
amortizes the integration gate across unchanged exact cells without creating an
unchecked skip path; any rebuild or reconfiguration invalidates the evidence.
"""

from __future__ import annotations

import argparse
import csv
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator


BACKEND_ORDER = ("CPU", "CUDA", "ROCm")
KV_PRECISIONS = ("FP16", "FP32", "Q8_1", "Q16_1", "TQ")
PRODUCTION_CAMPAIGN_NAME = re.compile(r"(?:^|_)ProductionCampaign(?:_|$)")
GLOBAL_TARGET_SECONDS = 4500.0
COMPLETION_TIMEOUT_SECONDS = 21600.0
EXACT_CELL_TIMEOUT_SECONDS = 600.0
REGISTERED_TIMEOUT_SECONDS = COMPLETION_TIMEOUT_SECONDS
MODEL_FIXTURE_NAME = "V2_Models"
MODEL_FIXTURE_TEST = "V2_FetchModelsFixture"
PRODUCTION_PARITY_PREFLIGHT_LABEL = "ProductionParityPreflight"
PRODUCTION_PARITY_UNIT_PREFIX = "V2_Unit_"
PRODUCTION_PARITY_UNIT_LABEL = "Unit"
PRODUCTION_PARITY_UNIT_BUILD_TARGET = "v2_unit_gate"
MODEL_RAMDISK_ROOT = Path("/dev/shm")
SESSION_IPC_RAMDISK_ROOT = Path("/dev/shm")
PERSISTENT_MODEL_CACHE_SCHEMA_VERSION = 2
CAMPAIGN_REPORT_SCHEMA_VERSION = 13
MODEL_COPY_CHUNK_BYTES = 16 * 1024 * 1024
MODEL_RAMDISK_MINIMUM_RESERVE_BYTES = 1024 * 1024 * 1024
MODEL_RAMDISK_RESERVE_FRACTION = 0.05
MODEL_MANIFEST_ENV = "LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS"
MODEL_RAMDISK_ENV = "LLAMINAR_PRODUCTION_PARITY_MODEL_RAMDISK"
ARTIFACT_ROOT_ENV = "LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT"
PARITY_RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "v2"
    / "integration"
    / "parity"
    / "results"
)
REQUIRED_CSV_HEADERS = {
    "prefill_layers.csv": (
        "backend,layer,avg_cosine,min_cosine,worst_stage,max_cosine_drop,"
        "max_drop_stage,stages_compared,max_kurtosis,max_kurtosis_stage,passed"
    ),
    "prefill_summary.csv": (
        "backend,lm_head_cosine,lm_head_kl,lm_head_top1,lm_head_top5,"
        "lm_head_pytorch_top1_in_topk,early_layers_passed,total_layers_passed,"
        "overall_passed"
    ),
    "prefill_stages.csv": (
        "backend,layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,snr_db,"
        "rmse,error_entropy,is_routing,routing_overlap,routing_top1_match,"
        "routing_weight_l1,llaminar_min,llaminar_max,llaminar_mean,"
        "llaminar_stddev,llaminar_kurtosis,llaminar_skewness,llaminar_p95,"
        "llaminar_p99,llaminar_outlier_frac,llaminar_dynamic_range,"
        "llaminar_sparsity,llaminar_zero_frac,llaminar_nan_count,"
        "llaminar_inf_count,llaminar_elements,pytorch_min,pytorch_max,"
        "pytorch_mean,pytorch_stddev,pytorch_kurtosis,pytorch_skewness,"
        "pytorch_p95,pytorch_p99,pytorch_outlier_frac,pytorch_dynamic_range,"
        "pytorch_sparsity,pytorch_zero_frac,pytorch_nan_count,"
        "pytorch_inf_count,pytorch_elements"
    ),
    "decode_steps.csv": (
        "backend,step,cosine,kl_divergence,top1_overlap,top5_overlap,"
        "llaminar_token,pytorch_token,token_match,top3_match,top5_match,passed"
    ),
    "decode_layers.csv": (
        "backend,step,layer,avg_cosine,min_cosine,worst_stage,max_cosine_drop,"
        "max_drop_stage,stages_compared,passed"
    ),
    "decode_stages.csv": (
        "backend,step,layer,stage,cosine,cosine_drop,rel_l2,max_abs_diff,"
        "snr_db,rmse,error_entropy,is_routing,routing_overlap,"
        "routing_top1_match,routing_weight_l1,llaminar_min,llaminar_max,"
        "llaminar_mean,llaminar_stddev,llaminar_kurtosis,llaminar_skewness,"
        "llaminar_p95,llaminar_p99,llaminar_outlier_frac,"
        "llaminar_dynamic_range,llaminar_sparsity,llaminar_zero_frac,"
        "llaminar_nan_count,llaminar_inf_count,llaminar_elements,pytorch_min,"
        "pytorch_max,pytorch_mean,pytorch_stddev,pytorch_kurtosis,"
        "pytorch_skewness,pytorch_p95,pytorch_p99,pytorch_outlier_frac,"
        "pytorch_dynamic_range,pytorch_sparsity,pytorch_zero_frac,"
        "pytorch_nan_count,pytorch_inf_count,pytorch_elements"
    ),
    "prefix_restore.csv": (
        "backend,phase,cache_config_enabled,cache_ready,cache_bypassed,"
        "cache_bypass_reason,request_enabled,request_bypassed,"
        "request_bypass_reason,hit,partial_hit,requested_tokens,"
        "matched_tokens,matched_blocks,terminal_logits_restored,"
        "terminal_hidden_restored,mtp_state_restored,hybrid_state_restored,"
        "storage_tier,current_position,state_compared,state_equivalent,"
        "state_detail,observed_moe_movement_epoch,"
        "oracle_moe_movement_epoch,main_kv_policy,"
        "main_kv_numerically_compared,"
        "main_kv_exact_prefix_segments,"
        "main_kv_numerical_suffix_payloads,"
        "main_kv_numerical_elements,"
        "main_kv_minimum_cosine,"
        "main_kv_maximum_relative_l2,"
        "main_kv_maximum_abs,"
        "main_kv_numerically_passed,gdn_state_policy,"
        "gdn_numerically_compared,gdn_numerical_payloads,"
        "gdn_numerical_elements,gdn_minimum_cosine,"
        "gdn_maximum_relative_l2,gdn_maximum_abs,"
        "gdn_numerically_passed,terminal_hidden_policy,"
        "terminal_hidden_numerically_compared,"
        "terminal_hidden_numerical_elements,"
        "terminal_hidden_numerical_cosine,"
        "terminal_hidden_numerical_rel_l2,"
        "terminal_hidden_numerical_max_abs,"
        "terminal_hidden_numerical_passed,"
        "terminal_logits_policy,"
        "terminal_logits_numerically_compared,"
        "terminal_logits_numerical_elements,"
        "terminal_logits_numerical_cosine,"
        "terminal_logits_numerical_rel_l2,"
        "terminal_logits_numerical_max_abs,"
        "terminal_logits_numerical_passed,"
        "observed_terminal_hidden_hash_available,"
        "observed_terminal_hidden_bytes,observed_terminal_hidden_hash,"
        "observed_terminal_logits_hash_available,"
        "observed_terminal_logits_bytes,observed_terminal_logits_hash,"
        "oracle_terminal_hidden_hash_available,"
        "oracle_terminal_hidden_bytes,oracle_terminal_hidden_hash,"
        "oracle_terminal_logits_hash_available,"
        "oracle_terminal_logits_bytes,oracle_terminal_logits_hash,"
        "checkpoint_compared,checkpoint_cosine,"
        "checkpoint_passed,passed"
    ),
    "production_path.csv": (
        "backend,device,execution_path,execution_topology,graph_contract,"
        "forward_full_graph_capture,forward_full_graph_replay,"
        "full_graph_capture,full_graph_replay,decode_graph_capture,"
        "decode_graph_replay,device_generation_controller,"
        "generation_execution_policy,native_generation_parent,"
        "hosted_ticket_boundary_certified,generation_loop_certified,"
        "generation_certification_detail,"
        "segmented_plan,"
        "segmented_capture,segmented_replay,model_context_reused,elapsed_seconds,"
        "budget_seconds,within_budget"
    ),
}
MTP_TRANSACTIONS_HEADER = (
    "backend,requested_draft_depth,response_limited_draft_depth,"
    "snapshot_execution_draft_depth,last_transaction_draft_depth,"
    "reference_step,condition_position,emitted_token_count,emitted_tokens,"
    "serial_oracle_tokens,serial_token_exact,attempted_draft_tokens,"
    "verifier_transactions,verifier_identity_transaction_count,"
    "verifier_identity_depth,production_verifier_draft_tokens,"
    "before_position,after_position,"
    "before_draft_steps,after_draft_steps,before_verifier_runs,"
    "after_verifier_runs,acceptance_witness_executed,"
    "acceptance_witness_serial_token_exact,"
    "acceptance_witness_accepted_token_delta,"
    "acceptance_witness_attempted_draft_tokens,"
    "acceptance_witness_verifier_transactions,"
    "acceptance_witness_emitted_tokens,"
    "acceptance_witness_serial_oracle_tokens,"
    "dynamic_policy_witness_executed,"
    "dynamic_policy_witness_serial_token_exact,"
    "dynamic_policy_witness_window_delta,"
    "dynamic_policy_witness_attempted_draft_tokens,"
    "dynamic_policy_witness_verifier_transactions,"
    "dynamic_policy_witness_emitted_tokens,"
    "dynamic_policy_witness_serial_oracle_tokens"
)
CSV_ARTIFACTS_REQUIRING_DATA = frozenset(
    {
        "prefill_layers.csv",
        "prefill_summary.csv",
        "prefill_stages.csv",
        "decode_steps.csv",
        "prefix_restore.csv",
        "production_path.csv",
    }
)
REQUIRED_CAMPAIGN_LABELS = {
    "Campaign",
    "ProductionPath",
    "AllPrecisions",
    "WholeMatrix75MinuteTarget",
}
REQUIRED_CAMPAIGN_ENV = {
    "GTEST_FAIL_FAST=1",
    "LLAMINAR_PRODUCTION_PARITY=1",
    "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN=1",
    "LLAMINAR_PRODUCTION_PARITY_TARGET_SECONDS=4500",
    "LLAMINAR_GPU_GRAPHS=1",
    "LLAMINAR_PERF_STATS_SUMMARY=1",
}
INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION = 1
INDIVIDUAL_PROGRESS_REPORT_SCHEMA_VERSION = 2
INDIVIDUAL_GREEN_PROVENANCE = frozenset(
    {
        "exact_process_exit_zero_and_fresh_artifact_contract",
        "aggregate_exit_zero_and_fresh_artifact_contract",
        "aggregate_gtest_fail_fast_prefix_before_declared_first_red",
    }
)


@dataclass(frozen=True, order=True)
class CampaignGroup:
    """Scheduling and economy identity for a set of matrix cells."""

    backends: str
    kv_precision: str

    @property
    def name(self) -> str:
        safe_backends = self.backends.replace("+", "-").lower()
        return f"{safe_backends}__kv-{self.kv_precision.lower()}"


@dataclass(frozen=True)
class CampaignCell:
    """One discovered CTest production-parity matrix cell."""

    name: str
    group: CampaignGroup
    gtest_cases: tuple[str, ...] = ()
    model_files: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    environment: tuple[str, ...] = ()
    working_directory: str = ""

    @property
    def test_type(self) -> str:
        """Return the CTest prefix identifying model family and topology."""

        marker = "_ProductionCampaign_"
        if marker not in self.name:
            raise ValueError(f"campaign name has no test-type boundary: {self.name}")
        return self.name.split(marker, 1)[0]

    @property
    def precision_types(self) -> tuple[str, ...]:
        """Return exact KV precision tags represented by the GTest cells."""

        present = [
            precision
            for precision in KV_PRECISIONS
            if any(
                re.search(
                    rf"(?:^|_)KV_?{re.escape(precision)}(?:_|$)", case
                )
                for case in self.gtest_cases
            )
        ]
        return tuple(present) if present else ("TOPOLOGY_DEFAULT",)


class PriorEvidenceCoverageKind(str, Enum):
    """Typed prior-artifact coverage state for one campaign aggregate."""

    UNSEEN = "unseen"
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class PriorEvidenceCoverage:
    """Exact-cell coverage supplied only as a campaign scheduling hint."""

    kind: PriorEvidenceCoverageKind
    observed_exact_cells: int
    total_exact_cells: int

    @property
    def unseen_exact_cells(self) -> int:
        """Return exact cells for which no complete prior artifact exists."""

        return self.total_exact_cells - self.observed_exact_cells


@dataclass(frozen=True)
class PriorArtifactEvidenceIndex:
    """Immutable union of complete exact-cell artifacts from explicit roots.

    This index is never correctness evidence for the active run. It can only
    prioritize aggregates so interrupted local campaigns discover new cells
    before rerunning cells already observed on the same binary slice.
    """

    roots: tuple[Path, ...]
    observed_gtest_cases: frozenset[str]

    def coverage_for(self, cell: CampaignCell) -> PriorEvidenceCoverage:
        """Classify one aggregate without changing its selected exact cells."""

        total = len(cell.gtest_cases)
        observed = sum(
            gtest_case in self.observed_gtest_cases
            for gtest_case in cell.gtest_cases
        )
        if observed == 0:
            kind = PriorEvidenceCoverageKind.UNSEEN
        elif observed == total:
            kind = PriorEvidenceCoverageKind.COMPLETE
        else:
            kind = PriorEvidenceCoverageKind.PARTIAL
        return PriorEvidenceCoverage(
            kind=kind,
            observed_exact_cells=observed,
            total_exact_cells=total,
        )


class ProcessTerminationKind(str, Enum):
    """Typed reason that the campaign driver terminated a process group."""

    NONE = "none"
    COMPLETION_TIMEOUT = "completion_timeout"
    EXACT_CELL_TIMEOUT = "exact_cell_timeout"
    CAMPAIGN_CANCELLED = "campaign_cancelled"


@dataclass(frozen=True)
class ExactCellTimeoutWatch:
    """Progress files and deadline used to watch one aggregate GTest process.

    Each tuple maps an exact generated GTest identity to the fresh
    ``test_log.txt`` created when that cell enters its parity fixture. A new
    file transfers watchdog authority to that cell without splitting the
    process or reloading immutable model weights.
    """

    timeout_seconds: float
    progress_files: tuple[tuple[str, Path], ...]
    not_before_wall_time_ns: int


@dataclass
class ProcessTerminationEvidence:
    """Evidence published when ``_run_process`` terminates a process group."""

    kind: ProcessTerminationKind = ProcessTerminationKind.NONE
    exact_gtest_case: str = ""
    cancelling_campaign: str = ""


class CampaignCancellation:
    """First-failure authority shared by all workers in one matrix run.

    A worker owns and terminates only its own subprocess group. This authority
    publishes the first genuine campaign failure to those workers, avoiding
    unsafe cross-thread access to ``Popen`` while still making cancellation
    immediate and process-group complete.
    """

    def __init__(self) -> None:
        self._requested = threading.Event()
        self._lock = threading.Lock()
        self._failing_campaign = ""

    def request_after_failure(self, campaign: str) -> bool:
        """Publish ``campaign`` as the first failure, returning true once."""

        if not campaign:
            raise ValueError("failing campaign identity must not be empty")
        with self._lock:
            if self._requested.is_set():
                return False
            self._failing_campaign = campaign
            self._requested.set()
            return True

    @property
    def requested(self) -> bool:
        """Return whether a worker has published a campaign failure."""

        return self._requested.is_set()

    @property
    def failing_campaign(self) -> str:
        """Return the immutable first-failure identity, or an empty string."""

        with self._lock:
            return self._failing_campaign


def _publish_campaign_failure(
    cancellation: CampaignCancellation | None,
    campaign: str,
) -> None:
    """Publish and log the first failure in a concurrent campaign matrix."""

    if (
        cancellation is not None
        and cancellation.request_after_failure(campaign)
    ):
        print(
            "[production-parity] fail_fast_status=CANCELLING "
            f"first_failed_campaign={campaign}",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True)
class CampaignResult:
    """Outcome and global-timeline evidence for one test-type campaign."""

    campaign: str
    test_type: str
    backends: str
    precision_set: str
    precision_types: tuple[str, ...]
    gtest_cases: tuple[str, ...]
    elapsed_seconds: float
    target_seconds: float
    return_code: int
    target_met: bool
    started_offset_seconds: float = 0.0
    finished_offset_seconds: float = 0.0
    completion_timeout_seconds: float = 0.0
    exact_cell_timeout_seconds: float = 0.0
    timed_out_gtest_case: str = ""
    cancelled_by_campaign: str = ""
    outcome: str = "completed"
    artifact_contract_passed: bool = True
    validated_artifact_file_count: int = 0
    artifact_directories: tuple[str, ...] = ()
    artifact_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagedModelEvidence:
    """Identity-bound evidence for one GGUF available in the selected tmpfs."""

    source_path: str
    filename: str
    size_bytes: int
    elapsed_seconds: float
    cache_status: str = "copied"
    source_identity: tuple[int, ...] = ()


@dataclass(frozen=True)
class CampaignMatrixResult:
    """Machine-readable correctness and economy proof for one matrix run."""

    schema_version: int
    global_target_seconds: float
    global_elapsed_seconds: float
    global_target_met: bool
    correctness_passed: bool
    performance_requirements_met: bool
    completion_timeout_seconds: float
    exact_cell_timeout_seconds: float
    preflight_return_code: int
    preflight_elapsed_seconds: float
    preflight_test_count: int
    preflight_tests: tuple[str, ...]
    fixture_return_code: int
    fixture_elapsed_seconds: float
    model_staging_return_code: int
    model_staging_elapsed_seconds: float
    model_staging_root: str
    model_staging_filesystem: str
    model_staging_mode: str
    model_staging_error: str
    artifact_root: str
    staged_model_count: int
    staged_model_bytes: int
    staged_models: tuple[StagedModelEvidence, ...]
    prior_evidence_roots: tuple[str, ...]
    prior_observed_exact_cell_count: int
    unseen_first_scheduling: bool
    scheduling_policy: str
    campaign_count: int
    exact_matrix_cell_count: int
    artifact_contract_passed: bool
    validated_artifact_file_count: int
    artifact_error_count: int
    campaigns: tuple[CampaignResult, ...]


@dataclass(frozen=True)
class IndividualGreenEvidence:
    """Durable proof that one exact command and fresh CSV contract passed.

    This evidence is intentionally diagnostic progress, not certification.
    Code may change between entries while a failing slice is repaired; only a
    final fresh unfiltered campaign can prove the combined tree remains green.
    """

    gtest_case: str
    campaign: str
    git_revision: str
    passed_wall_time_ns: int
    elapsed_seconds: float
    validated_artifact_file_count: int
    artifact_directory: str
    execution_contract_sha256: str
    provenance: str = "exact_process_exit_zero_and_fresh_artifact_contract"


@dataclass(frozen=True)
class IndividualGreenLedger:
    """Atomically persisted exact-cell progress for an interrupted fixup."""

    schema_version: int
    certification_eligible: bool
    entries: tuple[IndividualGreenEvidence, ...]

    @property
    def green_cases(self) -> frozenset[str]:
        """Return exact identities already proved green individually."""

        return frozenset(entry.gtest_case for entry in self.entries)


@dataclass(frozen=True)
class IndividualProgressReport:
    """Non-certifying report for one sequential unseen-cell invocation."""

    schema_version: int
    mode: str
    certification_eligible: bool
    selected_exact_cell_count: int
    green_before_count: int
    attempted_count: int
    newly_green_count: int
    green_after_count: int
    remaining_unseen_count: int
    stopped_on_failure: bool
    preflight_return_code: int
    preflight_test_count: int
    preflight_tests: tuple[str, ...]
    fixture_return_code: int
    model_staging_return_code: int
    artifact_root: str
    green_ledger: str
    results: tuple[CampaignResult, ...]


def _property_map(test: dict[str, Any]) -> dict[str, Any]:
    properties = test.get("properties", [])
    if isinstance(properties, dict):
        return properties
    return {
        str(entry.get("name", "")): entry.get("value")
        for entry in properties
        if isinstance(entry, dict)
    }


def _has_campaign_label(test: dict[str, Any]) -> bool:
    labels = _property_map(test).get("LABELS", [])
    if isinstance(labels, str):
        labels = labels.split(";")
    return "Campaign" in labels


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return value.split(";")
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def discover_production_parity_unit_tests(
    build_dir: Path,
) -> tuple[str, ...]:
    """Discover and validate the complete device-free unit prerequisite.

    CTest registration is the sole inventory authority.  Looking at the full
    JSON document lets this audit reject either half of a naming/label drift:
    a ``V2_Unit_*`` entry without the Unit label, or a Unit-labelled entry
    outside the canonical namespace.  CMake's ``v2_unit_gate`` owns executable
    construction; Python never copies its target list.
    """

    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CTest unit-gate discovery failed:\n"
            + (completed.stderr or completed.stdout)
        )

    document = json.loads(completed.stdout)
    names: list[str] = []
    for test in document.get("tests", []):
        name = str(test.get("name", ""))
        properties = _property_map(test)
        labels = _as_string_list(properties.get("LABELS", []))
        has_unit_name = name.startswith(PRODUCTION_PARITY_UNIT_PREFIX)
        has_unit_label = PRODUCTION_PARITY_UNIT_LABEL in labels
        if has_unit_name != has_unit_label:
            raise RuntimeError(
                "unit test naming/label contract disagrees: "
                f"{name or '<unnamed>'} labels={labels}"
            )
        if not has_unit_name:
            continue
        if "Campaign" in labels or "FullModel" in labels:
            raise RuntimeError(
                f"unit prerequisite contains a model campaign: {name}"
            )
        if _as_string_list(properties.get("FIXTURES_REQUIRED", [])):
            raise RuntimeError(
                f"unit prerequisite requires a fixture: {name}"
            )
        if _as_string_list(properties.get("REQUIRED_FILES", [])):
            raise RuntimeError(
                f"unit prerequisite requires external files: {name}"
            )
        try:
            timeout = float(properties.get("TIMEOUT", 0.0))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"unit prerequisite timeout is not numeric: {name}"
            ) from error
        if timeout <= 0.0:
            raise RuntimeError(
                f"unit prerequisite has no positive timeout: {name}"
            )
        names.append(name)

    names.sort()
    if not names:
        raise RuntimeError("no V2 unit tests discovered")
    if len(set(names)) != len(names):
        raise RuntimeError("CTest returned duplicate V2 unit tests")
    return tuple(names)


def discover_production_parity_preflight_tests(
    build_dir: Path,
) -> tuple[str, ...]:
    """Discover and validate the model-free integration preflight from CTest.

    CMake labels are the inventory authority. The campaign driver validates
    that every selected test is an Integration test, has no model fixture or
    required file, and has a short per-test timeout before it executes the
    label. This prevents a future registration edit from turning preflight into
    another model campaign or an unbounded wait.
    """

    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
            "-L",
            f"^{PRODUCTION_PARITY_PREFLIGHT_LABEL}$",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CTest production parity preflight discovery failed:\n"
            + (completed.stderr or completed.stdout)
        )

    document = json.loads(completed.stdout)
    names: list[str] = []
    for test in document.get("tests", []):
        name = str(test.get("name", ""))
        properties = _property_map(test)
        labels = _as_string_list(properties.get("LABELS", []))
        if PRODUCTION_PARITY_PREFLIGHT_LABEL not in labels:
            continue
        if not name.startswith("V2_Integration_") or "Integration" not in labels:
            raise RuntimeError(
                f"production parity preflight is not an Integration test: {name}"
            )
        if "Campaign" in labels or "FullModel" in labels:
            raise RuntimeError(
                f"production parity preflight contains a model campaign: {name}"
            )
        if _as_string_list(properties.get("FIXTURES_REQUIRED", [])):
            raise RuntimeError(
                f"production parity preflight requires a fixture: {name}"
            )
        if _as_string_list(properties.get("REQUIRED_FILES", [])):
            raise RuntimeError(
                f"production parity preflight requires model/files: {name}"
            )
        try:
            timeout = float(properties.get("TIMEOUT", 0.0))
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"production parity preflight timeout is not numeric: {name}"
            ) from error
        if timeout <= 0.0 or timeout > 120.0:
            raise RuntimeError(
                f"production parity preflight timeout must be in (0, 120] seconds: "
                f"{name} has {timeout}"
            )
        names.append(name)

    names.sort()
    if not names:
        raise RuntimeError(
            "no model-free ProductionParityPreflight integration tests discovered"
        )
    if len(set(names)) != len(names):
        raise RuntimeError(
            "CTest returned duplicate ProductionParityPreflight tests"
        )
    return tuple(names)


def _exact_gtest_cases(test: dict[str, Any]) -> tuple[str, ...]:
    """Extract and validate an exact, wildcard-free ProductionParity filter."""

    command = _as_string_list(test.get("command", []))
    filters = [
        argument.removeprefix("--gtest_filter=")
        for argument in command
        if argument.startswith("--gtest_filter=")
    ]
    if len(filters) != 1 or not filters[0]:
        raise ValueError("campaign must have exactly one nonempty GTest filter")
    if any(character in filters[0] for character in "*?"):
        raise ValueError("campaign GTest filter must enumerate exact matrix cells")
    cases = tuple(filters[0].split(":"))
    if len(set(cases)) != len(cases):
        raise ValueError("campaign GTest filter contains duplicate matrix cells")
    if any(
        re.search(r"\.ProductionParity(?:/|_|$)", case) is None
        for case in cases
    ):
        raise ValueError("campaign GTest filter contains a non-ProductionParity case")
    return cases


def _validate_campaign_registration(
    test: dict[str, Any],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    str,
]:
    """Fail closed when CTest no longer carries production/economy contracts."""

    properties = _property_map(test)
    labels = set(_as_string_list(properties.get("LABELS", [])))
    missing_labels = REQUIRED_CAMPAIGN_LABELS - labels
    if missing_labels:
        raise ValueError(
            "campaign is missing labels: " + ", ".join(sorted(missing_labels))
        )

    environment = set(_as_string_list(properties.get("ENVIRONMENT", [])))
    missing_environment = REQUIRED_CAMPAIGN_ENV - environment
    if missing_environment:
        raise ValueError(
            "campaign is missing environment: "
            + ", ".join(sorted(missing_environment))
        )
    if not any(
        entry.startswith("LLAMINAR_PERF_STATS_FILTER=")
        and "forward_graph" in entry.split("=", 1)[1].split(",")
        for entry in environment
    ):
        raise ValueError("campaign does not collect forward_graph PerfStats")

    try:
        timeout = float(properties.get("TIMEOUT", 0.0))
    except (TypeError, ValueError) as error:
        raise ValueError("campaign TIMEOUT is not numeric") from error
    if timeout != REGISTERED_TIMEOUT_SECONDS:
        raise ValueError(
            f"campaign TIMEOUT is {timeout}, expected {REGISTERED_TIMEOUT_SECONDS}"
        )
    model_files = tuple(
        sorted(set(_as_string_list(properties.get("REQUIRED_FILES", []))))
    )
    if not model_files:
        raise ValueError("campaign has no REQUIRED_FILES GGUF manifest")
    if any(Path(path).suffix.lower() != ".gguf" for path in model_files):
        raise ValueError("campaign REQUIRED_FILES contains a non-GGUF input")

    manifests = [
        entry.split("=", 1)[1]
        for entry in environment
        if entry.startswith(MODEL_MANIFEST_ENV + "=")
    ]
    if len(manifests) != 1:
        raise ValueError(
            f"campaign must declare exactly one {MODEL_MANIFEST_ENV} value"
        )
    manifest_files = tuple(sorted(filter(None, manifests[0].split("|"))))
    if manifest_files != model_files:
        raise ValueError(
            "campaign runtime model manifest does not match REQUIRED_FILES"
        )

    command = tuple(_as_string_list(test.get("command", [])))
    if not command or any(not argument for argument in command):
        raise ValueError("campaign has an empty registered command")
    working_directory = str(properties.get("WORKING_DIRECTORY", ""))
    if not working_directory or not Path(working_directory).is_absolute():
        raise ValueError(
            "campaign WORKING_DIRECTORY must be an absolute path"
        )
    return (
        _exact_gtest_cases(test),
        model_files,
        command,
        tuple(_as_string_list(properties.get("ENVIRONMENT", []))),
        working_directory,
    )


def classify_campaign(name: str) -> CampaignGroup:
    """Derive backend and precision-set identity from a CTest campaign name."""

    present = [backend for backend in BACKEND_ORDER if backend in name]
    if not present:
        raise ValueError(f"campaign name has no backend identity: {name}")

    if "ALL_PRECISIONS" in name:
        precision = "ALL"
    else:
        precision_match = re.search(
            r"(?:^|_)KV_("
            + "|".join(map(re.escape, KV_PRECISIONS))
            + r")(?:_|$)",
            name,
        )
        precision = (
            precision_match.group(1) if precision_match else "TOPOLOGY_DEFAULT"
        )
    return CampaignGroup(backends="+".join(present), kv_precision=precision)


def discover_campaigns(
    build_dir: Path,
    backend_regex: str = ".*",
    precision_regex: str = ".*",
    campaign_regex: str = ".*",
    exclude_campaign_regex: str | None = None,
) -> list[CampaignCell]:
    """Read and validate the selected production campaigns from CTest.

    Selection uses only identity encoded in the registered CTest name.  It is
    therefore safe to apply before validating the heavier command, environment,
    and GGUF contracts.  This matters during focused development: an unselected
    parity executable may still have stale post-build discovery metadata, but it
    must neither expand the requested matrix nor prevent a selected, current
    campaign from running.  A default/full discovery still validates every
    production campaign and remains fail-closed.
    """

    completed = subprocess.run(
        [
            "ctest",
            "--test-dir",
            str(build_dir),
            "--show-only=json-v1",
            "-L",
            "Campaign",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CTest campaign discovery failed:\n"
            + (completed.stderr or completed.stdout)
        )

    document = json.loads(completed.stdout)
    backend = re.compile(backend_regex)
    precision = re.compile(precision_regex)
    campaign = re.compile(campaign_regex)
    excluded = (
        re.compile(exclude_campaign_regex)
        if exclude_campaign_regex is not None
        else None
    )
    cells: list[CampaignCell] = []
    for test in document.get("tests", []):
        name = str(test.get("name", ""))
        if not PRODUCTION_CAMPAIGN_NAME.search(name) or not _has_campaign_label(test):
            continue
        group = classify_campaign(name)
        if (
            backend.fullmatch(group.backends) is None
            or precision.fullmatch(group.kv_precision) is None
            or campaign.fullmatch(name) is None
            or (excluded is not None and excluded.fullmatch(name) is not None)
        ):
            continue
        try:
            (
                exact_cases,
                model_files,
                command,
                environment,
                working_directory,
            ) = _validate_campaign_registration(test)
        except ValueError as error:
            raise RuntimeError(f"invalid production campaign {name}: {error}") from error
        cells.append(
            CampaignCell(
                name=name,
                group=group,
                gtest_cases=exact_cases,
                model_files=model_files,
                command=command,
                environment=environment,
                working_directory=working_directory,
            )
        )

    cells.sort(key=lambda cell: (cell.group, cell.name))
    if not cells:
        raise RuntimeError(
            f"no ProductionCampaign tests discovered in {build_dir}; "
            "build the parity targets after configuring the Integration tree"
        )
    if len({cell.name for cell in cells}) != len(cells):
        raise RuntimeError("CTest returned duplicate ProductionCampaign names")
    return cells


def filter_campaigns(
    cells: Iterable[CampaignCell],
    backend_regex: str,
    precision_regex: str,
    campaign_regex: str = ".*",
    exclude_campaign_regex: str | None = None,
) -> list[CampaignCell]:
    """Select cells using full regular-expression matches on registered identity.

    Campaign-name selection is intentionally applied to CTest-discovered names;
    it never creates a second topology/model manifest. This permits a focused
    architecture slice (for example, all non-ExpertOverlay campaigns) to retain
    the same whole-slice target and machine-readable coverage proof as the default
    whole-matrix run.
    """

    backend = re.compile(backend_regex)
    precision = re.compile(precision_regex)
    campaign = re.compile(campaign_regex)
    excluded = (
        re.compile(exclude_campaign_regex)
        if exclude_campaign_regex is not None
        else None
    )
    return [
        cell
        for cell in cells
        if backend.fullmatch(cell.group.backends)
        and precision.fullmatch(cell.group.kv_precision)
        and campaign.fullmatch(cell.name)
        and (excluded is None or excluded.fullmatch(cell.name) is None)
    ]


def group_campaigns(
    cells: Iterable[CampaignCell],
) -> dict[CampaignGroup, list[CampaignCell]]:
    """Return a stable group-to-cell mapping."""

    grouped: dict[CampaignGroup, list[CampaignCell]] = {}
    for cell in cells:
        grouped.setdefault(cell.group, []).append(cell)
    for group_cells in grouped.values():
        group_cells.sort(key=lambda cell: cell.name)
    return dict(sorted(grouped.items()))


class ModelStagingError(RuntimeError):
    """Raised when the selected GGUF corpus cannot be staged exactly."""


@dataclass(frozen=True)
class ModelStagingWorkspace:
    """Paths and lifetime mode for one identity-bound model-cache workspace."""

    root: Path
    models: Path
    mode: str
    persistent: bool

    def protect_published_models(self) -> None:
        """Seal persistent model metadata while child campaigns consume it.

        The cache lock remains held by the driver.  Removing owner-write from
        the root and model directory makes an unrelated ``rm -rf /dev/shm/*``
        fail before it can unlink the published manifest or GGUFs.
        """

        if self.persistent:
            _set_persistent_cache_write_access(self.root, writable=False)


def _set_persistent_cache_write_access(
    root: Path,
    *,
    writable: bool,
) -> None:
    """Apply the portable same-owner deletion guard for a persistent cache.

    A tmpfs cannot survive a reboot, remount, or container replacement.  This
    guard instead prevents accidental same-mount cleanup by ordinary processes:
    directories are owner read/execute while idle, then made owner-writable only
    after the campaign has acquired the cache's exclusive lock.  Symlinks and
    non-directory cache components fail closed.

    Args:
        root: Exact persistent cache root.
        writable: Whether staging metadata and model entries may be mutated.
    """

    if not root.exists() or root.is_symlink() or not root.is_dir():
        raise ModelStagingError(
            f"persistent cache root disappeared or changed type: {root}"
        )

    paths = [root, root / "models"]
    existing: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_dir():
            raise ModelStagingError(
                f"persistent cache component is not a real directory: {path}"
            )
        existing.append(path)

    # Make the root traversable/writable before its children on admission.  On
    # sealing, protect children first so there is never a window in which a
    # writable model directory sits below an already advertised protected root.
    ordered = existing if writable else list(reversed(existing))
    for path in ordered:
        path.chmod(0o700 if writable else 0o500)


def _reclaim_interrupted_persistent_transactions(root: Path) -> tuple[int, int]:
    """Remove unpublished cache transactions while holding the cache lock.

    Model copies and manifest updates are published with atomic renames.  A
    SIGKILL, host reboot, or editor-container reload can therefore leave only
    their hidden transaction files behind.  Once the persistent campaign lock
    has been acquired, no healthy producer can still own one of those files,
    so retaining it would merely consume tmpfs capacity and make setup cease to
    be idempotent.

    Only the two cache-owned transaction namespaces are eligible.  Published
    GGUFs, the lock, and operator-owned files are never touched.
    A matching directory is treated as corrupt state rather than recursively
    removed.

    Args:
        root: Exclusively locked persistent cache root.

    Returns:
        The number of transactions and regular-file bytes reclaimed.
    """

    candidates: list[Path] = []
    models = root / "models"
    if models.exists():
        if models.is_symlink() or not models.is_dir():
            raise ModelStagingError(
                f"persistent model directory is not a real directory: {models}"
            )
        candidates.extend(
            path
            for path in models.iterdir()
            if re.fullmatch(r"\..+\.copying-[0-9]+", path.name)
        )

    candidates.extend(
        path
        for path in root.iterdir()
        if re.fullmatch(
            r"\.model-cache-manifest\.json\.publishing-[0-9]+-[0-9]+",
            path.name,
        )
    )

    reclaimed_bytes = 0
    for path in sorted(candidates):
        transaction_stat = path.lstat()
        if stat.S_ISDIR(transaction_stat.st_mode):
            raise ModelStagingError(
                "persistent cache transaction path is unexpectedly a "
                f"directory: {path}"
            )
        if stat.S_ISREG(transaction_stat.st_mode):
            reclaimed_bytes += transaction_stat.st_size
        # unlink() removes a matching symlink itself and never follows it.
        path.unlink()

    return len(candidates), reclaimed_bytes


@contextmanager
def model_staging_workspace(
    ramdisk_root: Path,
    persistent_cache_directory: Path | None,
    completion_deadline: float | None,
) -> Iterator[ModelStagingWorkspace]:
    """Own either a disposable run root or an exclusively locked stable cache.

    The persistent lock remains held until all child campaigns exit. This lets
    later invocations update stale entries atomically without replacing a file
    that an active inference process is reading. Persistent contents are never
    removed here; the operator owns their eventual manual cleanup.
    """

    resolved_ramdisk_root = ramdisk_root.resolve(strict=True)
    if persistent_cache_directory is None:
        with tempfile.TemporaryDirectory(
            prefix="llaminar-production-parity-",
            dir=resolved_ramdisk_root,
        ) as raw_run_root:
            root = Path(raw_run_root)
            yield ModelStagingWorkspace(
                root=root,
                models=root / "models",
                mode="run_scoped",
                persistent=False,
            )
        return

    candidate = (
        persistent_cache_directory
        if persistent_cache_directory.is_absolute()
        else resolved_ramdisk_root / persistent_cache_directory
    )
    if candidate.is_symlink():
        raise ModelStagingError(
            f"persistent model cache directory must not be a symlink: {candidate}"
        )
    candidate.mkdir(mode=0o700, parents=True, exist_ok=True)
    root = candidate.resolve(strict=True)
    try:
        root.relative_to(resolved_ramdisk_root)
    except ValueError as error:
        raise ModelStagingError(
            f"persistent model cache {root} is outside ramdisk root "
            f"{resolved_ramdisk_root}"
        ) from error
    if root == resolved_ramdisk_root:
        raise ModelStagingError(
            "persistent model cache must be a child directory, not the "
            "ramdisk mount root itself"
        )

    # `/dev/shm` is an IPC namespace, not a durable cache authority.  On hosts
    # with systemd-logind RemoveIPC, the complete namespace can be reclaimed
    # when a login session ends; directory mode sealing cannot protect against
    # that privileged lifecycle.  Persistent campaigns therefore require the
    # repository-owned named tmpfs (or another explicit memory mount) outside
    # `/dev/shm`.  Run-scoped staging remains valid there for isolated CI
    # containers whose IPC namespace is itself the intended lifetime.
    session_ipc_root = SESSION_IPC_RAMDISK_ROOT.resolve(strict=True)
    try:
        root.relative_to(session_ipc_root)
    except ValueError:
        pass
    else:
        raise ModelStagingError(
            "persistent model cache requires a dedicated tmpfs mount outside "
            f"{session_ipc_root}; session IPC cleanup may remove /dev/shm "
            "during a campaign. Run scripts/ci/setup_production_parity_tmpfs.sh "
            "and use --model-ramdisk-root /mnt/llaminar-production-parity"
        )
    filesystem_type = _filesystem_type(root)
    if filesystem_type not in {"tmpfs", "ramfs"}:
        raise ModelStagingError(
            f"persistent model cache requires tmpfs/ramfs, got "
            f"{filesystem_type} for {root}"
        )

    lock_path = root / ".campaign.lock"
    open_flags = os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    lock_fd = os.open(lock_path, open_flags, 0o600)
    lock_acquired = False
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
                break
            except BlockingIOError:
                if (
                    completion_deadline is not None
                    and time.monotonic() >= completion_deadline
                ):
                    raise TimeoutError(
                        f"completion timeout expired waiting for persistent "
                        f"model cache lock: {lock_path}"
                    )
                time.sleep(0.05)
        _set_persistent_cache_write_access(root, writable=True)
        reclaimed_count, reclaimed_bytes = (
            _reclaim_interrupted_persistent_transactions(root)
        )
        if reclaimed_count:
            print(
                "[production-parity] "
                f"reclaimed_interrupted_cache_transactions={reclaimed_count} "
                f"reclaimed_bytes={reclaimed_bytes}",
                flush=True,
            )
        print(
            f"[production-parity] persistent_model_cache={root} "
            "lifetime=manual_cleanup",
            flush=True,
        )
        yield ModelStagingWorkspace(
            root=root,
            models=root / "models",
            mode="persistent",
            persistent=True,
        )
    finally:
        try:
            if lock_acquired:
                # Seal every cache-owned directory before releasing the lock.
                # A later campaign can unseal it only after acquiring this same
                # authority; ordinary cleanup cannot unlink published GGUFs.
                _set_persistent_cache_write_access(root, writable=False)
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)


def _filesystem_type(path: Path) -> str:
    """Return the Linux filesystem type for the deepest mount owning ``path``."""

    resolved = path.resolve(strict=True)
    best_mount = Path("/")
    best_type = ""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ModelStagingError(f"cannot read /proc/self/mountinfo: {error}") from error

    def decode_mount_field(value: str) -> str:
        return (
            value.replace(r"\040", " ")
            .replace(r"\011", "\t")
            .replace(r"\012", "\n")
            .replace(r"\134", "\\")
        )

    for line in lines:
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        left_fields = left.split()
        right_fields = right.split()
        if len(left_fields) < 5 or not right_fields:
            continue
        mount_point = Path(decode_mount_field(left_fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        if len(mount_point.parts) >= len(best_mount.parts):
            best_mount = mount_point
            best_type = right_fields[0]

    if not best_type:
        raise ModelStagingError(f"cannot resolve filesystem type for {resolved}")
    return best_type


def _split_gguf_members(path: Path) -> tuple[Path, ...]:
    """Expand a standard ``-00001-of-000NN.gguf`` declaration atomically."""

    match = re.fullmatch(
        r"(?P<prefix>.*)-(?P<part>[0-9]{5})-of-(?P<count>[0-9]{5})(?P<suffix>\.gguf)",
        path.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        return (path,)

    count = int(match.group("count"))
    if count <= 0:
        raise ModelStagingError(f"invalid split GGUF member count in {path}")
    members = tuple(
        path.with_name(
            f"{match.group('prefix')}-{part:05d}-of-{count:05d}"
            f"{match.group('suffix')}"
        )
        for part in range(1, count + 1)
    )
    missing = [str(member) for member in members if not member.is_file()]
    if missing:
        raise ModelStagingError(
            "declared split GGUF is incomplete: " + ", ".join(missing)
        )
    return members


def selected_model_files(cells: Iterable[CampaignCell]) -> tuple[Path, ...]:
    """Resolve, expand, deduplicate, and collision-check selected GGUF inputs."""

    sources: list[Path] = []
    for raw_path in sorted(
        {path for cell in cells for path in cell.model_files}
    ):
        path = Path(raw_path)
        if not path.is_file():
            raise ModelStagingError(f"declared production GGUF does not exist: {path}")
        if path.suffix.lower() != ".gguf":
            raise ModelStagingError(f"declared production model is not GGUF: {path}")
        sources.extend(_split_gguf_members(path))

    canonical: dict[Path, Path] = {}
    by_filename: dict[str, Path] = {}
    for source in sources:
        resolved = source.resolve(strict=True)
        canonical.setdefault(resolved, source)
        previous = by_filename.get(source.name)
        if previous is not None and not os.path.samefile(previous, source):
            raise ModelStagingError(
                f"distinct production models share staging filename {source.name}: "
                f"{previous} and {source}"
            )
        by_filename[source.name] = source

    if not canonical:
        raise ModelStagingError("selected production matrix declares no GGUF files")
    return tuple(sorted(canonical, key=lambda path: (path.name, str(path))))


def _source_identity(stat_result: os.stat_result) -> tuple[int, ...]:
    """Return the mutation-sensitive identity authenticated during a copy."""

    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _persistent_manifest_path(staging_directory: Path) -> Path:
    """Return the cache-owned manifest beside the persistent model directory."""

    return staging_directory.parent / "model-cache-manifest.json"


def _load_persistent_manifest(staging_directory: Path) -> dict[str, Any]:
    """Load and cheaply migrate the persistent cache authority.

    Schema 1 stored full-file SHA-256 values. Schema 2 deliberately relies on
    stable source and destination filesystem identities instead. Existing
    schema-1 entries can be upgraded without rereading model payloads; entries
    that predate destination identity binding simply miss and are recopied.
    """

    path = _persistent_manifest_path(staging_directory)
    if not path.exists():
        return {
            "schema_version": PERSISTENT_MODEL_CACHE_SCHEMA_VERSION,
            "models": {},
        }
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ModelStagingError(
            f"persistent model cache manifest is unreadable: {path}: {error}"
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("models"), dict):
        raise ModelStagingError(
            f"persistent model cache manifest has an unsupported shape: {path}"
        )
    schema_version = document.get("schema_version")
    if schema_version not in {1, PERSISTENT_MODEL_CACHE_SCHEMA_VERSION}:
        raise ModelStagingError(
            f"persistent model cache manifest has an unsupported shape: {path}"
        )
    if schema_version == 1:
        document["schema_version"] = PERSISTENT_MODEL_CACHE_SCHEMA_VERSION
        for raw_entry in document["models"].values():
            if isinstance(raw_entry, dict):
                raw_entry.pop("sha256", None)
        _publish_persistent_manifest(staging_directory, document)
    return document


def _publish_persistent_manifest(
    staging_directory: Path,
    document: dict[str, Any],
) -> None:
    """Atomically publish cache metadata after a model is fully authenticated."""

    path = _persistent_manifest_path(staging_directory)
    temporary = path.with_name(
        f".{path.name}.publishing-{os.getpid()}-{time.time_ns()}"
    )
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o400)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _persistent_manifest_entry(
    source: Path,
    destination: Path,
    evidence: StagedModelEvidence,
) -> dict[str, Any]:
    """Build the mutation-sensitive authority for one persistent cache entry."""

    cached_stat = destination.stat(follow_symlinks=False)
    return {
        "source_path": str(source),
        "source_identity": list(evidence.source_identity),
        "cached_identity": list(_source_identity(cached_stat)),
        "size_bytes": evidence.size_bytes,
    }


def _reuse_persistent_model(
    source: Path,
    destination: Path,
    raw_entry: Any,
) -> StagedModelEvidence | None:
    """Authenticate and reuse one immutable tmpfs entry when its source is unchanged.

    The source's device/inode/size/mtime/ctime tuple makes an ordinary source
    replacement or mutation invalidate the entry without rereading slow model
    storage. The read-only tmpfs identity is bound to that source identity at
    atomic publication, so an unchanged pair can be reused using metadata-only
    checks. An entry without either identity is a miss, never an invitation to
    scan the model payload.
    """

    if not isinstance(raw_entry, dict):
        return None
    started = time.monotonic()
    source_before = source.stat()
    if (
        raw_entry.get("source_path") != str(source)
        or raw_entry.get("source_identity")
        != list(_source_identity(source_before))
        or raw_entry.get("size_bytes") != source_before.st_size
    ):
        return None
    try:
        cached_before = destination.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(cached_before.st_mode)
        or cached_before.st_size != source_before.st_size
        or cached_before.st_mode & 0o222
    ):
        return None

    cached_identity = _source_identity(cached_before)
    if raw_entry.get("cached_identity") != list(cached_identity):
        return None

    cached_after = destination.stat(follow_symlinks=False)
    source_after = source.stat()
    if _source_identity(source_after) != _source_identity(source_before):
        raise ModelStagingError(
            f"GGUF changed while its persistent cache entry was validated: {source}"
        )
    if _source_identity(cached_after) != _source_identity(cached_before):
        raise ModelStagingError(
            f"persistent cached GGUF changed while it was validated: {destination}"
        )
    return StagedModelEvidence(
        source_path=str(source),
        filename=destination.name,
        size_bytes=cached_before.st_size,
        elapsed_seconds=time.monotonic() - started,
        cache_status="reused",
        source_identity=_source_identity(source_before),
    )


def _stage_one_model(
    source: Path,
    destination: Path,
    completion_deadline: float | None,
) -> StagedModelEvidence:
    """Stream one source into an atomic, identity-bound tmpfs transaction.

    Full-write and byte-count checks, source mutation checks, fsync, read-only
    publication, and destination identity binding make the cache transaction
    exact without forcing a CPU-bound content-hash pass over every model byte.
    """

    started = time.monotonic()
    source_before = source.stat()
    temporary = destination.with_name(
        f".{destination.name}.copying-{os.getpid()}"
    )
    copied_bytes = 0
    try:
        with source.open("rb", buffering=0) as input_stream:
            if _source_identity(os.fstat(input_stream.fileno())) != _source_identity(
                source_before
            ):
                raise ModelStagingError(
                    f"GGUF changed between stat and open: {source}"
                )
            with temporary.open("xb", buffering=0) as output_stream:
                while True:
                    if (
                        completion_deadline is not None
                        and time.monotonic() >= completion_deadline
                    ):
                        raise TimeoutError(
                            f"completion timeout expired while staging {source}"
                        )
                    chunk = input_stream.read(MODEL_COPY_CHUNK_BYTES)
                    if not chunk:
                        break
                    written = output_stream.write(chunk)
                    if written != len(chunk):
                        raise ModelStagingError(
                            f"short tmpfs write for {source}: {written}/{len(chunk)}"
                        )
                    copied_bytes += len(chunk)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            opened_after = os.fstat(input_stream.fileno())

        path_after = source.stat()
        expected_identity = _source_identity(source_before)
        if (
            _source_identity(opened_after) != expected_identity
            or _source_identity(path_after) != expected_identity
        ):
            raise ModelStagingError(f"GGUF changed while it was staged: {source}")
        if copied_bytes != source_before.st_size:
            raise ModelStagingError(
                f"staged GGUF byte count mismatch for {source}: "
                f"{copied_bytes} != {source_before.st_size}"
            )

        temporary.chmod(0o400)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    elapsed = time.monotonic() - started
    return StagedModelEvidence(
        source_path=str(source),
        filename=destination.name,
        size_bytes=copied_bytes,
        elapsed_seconds=elapsed,
        source_identity=_source_identity(source_before),
    )


def stage_models_in_ramdisk(
    cells: Iterable[CampaignCell],
    staging_directory: Path,
    completion_deadline: float | None,
    *,
    persistent: bool = False,
) -> tuple[tuple[StagedModelEvidence, ...], str]:
    """Stage or reuse the selected corpus in a capacity-proven memory filesystem.

    Run-scoped staging always creates an empty private directory. Persistent
    staging instead owns a versioned manifest and keeps immutable model files
    across invocations. A persistent hit requires unchanged source and read-only
    cached-file identities; misses are copied and published atomically without
    exposing partial weights. Exact byte counts and mutation checks protect the
    copy, while numerical parity proves the loaded weight contents.
    """

    sources = selected_model_files(cells)
    staging_directory.mkdir(mode=0o700, parents=True, exist_ok=persistent)
    filesystem_type = _filesystem_type(staging_directory)
    if filesystem_type not in {"tmpfs", "ramfs"}:
        raise ModelStagingError(
            f"production model staging requires tmpfs/ramfs, got {filesystem_type} "
            f"for {staging_directory}"
        )

    manifest = (
        _load_persistent_manifest(staging_directory)
        if persistent
        else None
    )
    reusable: dict[Path, StagedModelEvidence] = {}
    if manifest is not None:
        for source in sources:
            reusable_record = _reuse_persistent_model(
                source,
                staging_directory / source.name,
                manifest["models"].get(source.name),
            )
            if reusable_record is not None:
                reusable[source] = reusable_record

    sources_to_copy = tuple(source for source in sources if source not in reusable)
    required_bytes = sum(source.stat().st_size for source in sources_to_copy)
    reserve_bytes = max(
        MODEL_RAMDISK_MINIMUM_RESERVE_BYTES,
        math.ceil(
            sum(source.stat().st_size for source in sources)
            * MODEL_RAMDISK_RESERVE_FRACTION
        ),
    )
    filesystem = os.statvfs(staging_directory)
    available_bytes = filesystem.f_bavail * filesystem.f_frsize
    if required_bytes > 0 and required_bytes + reserve_bytes > available_bytes:
        raise ModelStagingError(
            "insufficient ramdisk capacity for selected GGUF corpus: "
            f"required={required_bytes} reserve={reserve_bytes} "
            f"available={available_bytes}"
        )

    print(
        f"[production-parity] staging_models={len(sources)} "
        f"cache_hits={len(reusable)} copy_bytes={required_bytes} "
        f"mode={'persistent' if persistent else 'run_scoped'} "
        f"filesystem={filesystem_type} "
        f"destination={staging_directory}",
        flush=True,
    )
    evidence: list[StagedModelEvidence] = []
    for index, source in enumerate(sources, start=1):
        if (
            completion_deadline is not None
            and time.monotonic() >= completion_deadline
        ):
            raise TimeoutError(
                "completion timeout expired before GGUF staging completed"
            )
        cached_record = reusable.get(source)
        if cached_record is not None:
            evidence.append(cached_record)
            throughput_mib_s = (
                cached_record.size_bytes
                / (1024.0 * 1024.0)
                / max(cached_record.elapsed_seconds, 1e-9)
            )
            print(
                f"[production-parity] reused_model={cached_record.filename} "
                f"elapsed_seconds={cached_record.elapsed_seconds:.3f} "
                f"metadata_validation_throughput_mib_s={throughput_mib_s:.1f}",
                flush=True,
            )
            continue
        print(
            f"[production-parity] staging_model={index}/{len(sources)} "
            f"source={source} bytes={source.stat().st_size}",
            flush=True,
        )
        record = _stage_one_model(
            source,
            staging_directory / source.name,
            completion_deadline,
        )
        evidence.append(record)
        if manifest is not None:
            destination = staging_directory / source.name
            manifest["models"][source.name] = _persistent_manifest_entry(
                source,
                destination,
                record,
            )
            _publish_persistent_manifest(staging_directory, manifest)
        throughput_mib_s = (
            record.size_bytes / (1024.0 * 1024.0) / max(record.elapsed_seconds, 1e-9)
        )
        print(
            f"[production-parity] staged_model={record.filename} "
            f"elapsed_seconds={record.elapsed_seconds:.3f} "
            f"throughput_mib_s={throughput_mib_s:.1f}",
            flush=True,
        )
    return tuple(evidence), filesystem_type


def _ctest_exact_regex(cells: Iterable[CampaignCell]) -> str:
    names = [re.escape(cell.name) for cell in cells]
    if not names:
        raise ValueError("cannot build a CTest regex for an empty campaign group")
    # CTest uses its own extended-regular-expression implementation, not
    # Python/PCRE syntax.  In particular, a non-capturing group (`(?:...)`) is
    # rejected and can otherwise degrade into a misleading zero-test success.
    return "^(" + "|".join(names) + ")$"


def _exact_registered_command(
    cell: CampaignCell,
    gtest_case: str,
) -> tuple[str, ...]:
    """Narrow one registered aggregate command to one exact matrix cell.

    The CTest-discovered argv remains authoritative for MPI rank count,
    affinity, backend exports, executable, and every other production launch
    detail. Only its already validated exact GTest filter may be narrowed.
    """

    if gtest_case not in cell.gtest_cases:
        raise ValueError(
            f"exact case {gtest_case} is not registered by {cell.name}"
        )
    filter_indexes = [
        index
        for index, argument in enumerate(cell.command)
        if argument.startswith("--gtest_filter=")
    ]
    if len(filter_indexes) != 1:
        raise ValueError(
            f"campaign {cell.name} must have exactly one registered GTest filter"
        )
    filter_index = filter_indexes[0]
    registered_cases = tuple(
        cell.command[filter_index]
        .removeprefix("--gtest_filter=")
        .split(":")
    )
    if registered_cases != cell.gtest_cases:
        raise ValueError(
            f"campaign {cell.name} command/filter identity is inconsistent"
        )
    command = list(cell.command)
    command[filter_index] = f"--gtest_filter={gtest_case}"
    return tuple(command)


def _registered_environment(cell: CampaignCell) -> dict[str, str]:
    """Convert CTest's exact environment contract to a process mapping."""

    environment: dict[str, str] = {}
    for assignment in cell.environment:
        name, separator, value = assignment.partition("=")
        if not separator or not name or "\x00" in assignment:
            raise ValueError(
                f"campaign {cell.name} has invalid environment assignment"
            )
        environment[name] = value
    return environment


def _exact_execution_contract_sha256(
    cell: CampaignCell,
    gtest_case: str,
) -> str:
    """Hash the registered argv, environment, and working directory."""

    command = _exact_registered_command(cell, gtest_case)
    environment = _registered_environment(cell)
    payload = json.dumps(
        {
            "command": command,
            "environment": sorted(environment.items()),
            "working_directory": cell.working_directory,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _single_exact_cell(
    cell: CampaignCell,
    gtest_case: str,
) -> CampaignCell:
    """Return one typed view of an aggregate for exact artifact validation."""

    return CampaignCell(
        name=cell.name,
        group=cell.group,
        gtest_cases=(gtest_case,),
        model_files=cell.model_files,
        command=_exact_registered_command(cell, gtest_case),
        environment=cell.environment,
        working_directory=cell.working_directory,
    )


def _current_git_short_hash() -> str:
    """Return the exact revision namespace used by the C++ artifact writer."""

    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision:
        raise RuntimeError(
            "cannot resolve the production parity artifact revision: "
            + (completed.stderr.strip() or "git returned no revision")
        )
    return revision


def _gtest_artifact_directory_name(gtest_case: str) -> str:
    """Mirror ``ParityCSVArtifactWriter::resultsDir`` without a second ID table."""

    suite, separator, test = gtest_case.partition(".")
    if not separator or not suite or not test:
        raise ValueError(f"invalid exact GTest case identity: {gtest_case}")
    raw_name = f"{suite}/{test}"
    return re.sub(r'[/\\:*?"<>|]', "_", raw_name)


def _exact_cell_timeout_watch(
    cell: CampaignCell,
    artifact_results_root: Path | None,
    timeout_seconds: float,
    not_before_wall_time_ns: int,
) -> ExactCellTimeoutWatch | None:
    """Build the file-backed watchdog contract for one discovered campaign."""

    if not cell.gtest_cases:
        return None
    root = artifact_results_root
    if root is None:
        root = PARITY_RESULTS_DIRECTORY / _current_git_short_hash()
    return ExactCellTimeoutWatch(
        timeout_seconds=timeout_seconds,
        progress_files=tuple(
            (
                gtest_case,
                root
                / _gtest_artifact_directory_name(gtest_case)
                / "test_log.txt",
            )
            for gtest_case in cell.gtest_cases
        ),
        not_before_wall_time_ns=not_before_wall_time_ns,
    )


def _required_csv_headers_for_case(gtest_case: str) -> dict[str, str]:
    """Return the exact artifact contract for one typed matrix cell."""

    headers = dict(REQUIRED_CSV_HEADERS)
    if re.search(
        r"(?:^|_)MTP(?:Depth(?:1|2|3|15)|DynamicDepth)(?:_|$)",
        gtest_case,
    ):
        headers["mtp_transactions.csv"] = MTP_TRANSACTIONS_HEADER
    return headers


def _prior_artifact_file_has_evidence(
    path: Path,
    expected_header: str,
    *,
    requires_data: bool,
) -> bool:
    """Check the minimum immutable artifact shape needed for scheduling.

    Prior files are deliberately inspected only far enough to reject missing,
    empty, schema-mismatched, or header-only required evidence. The active run
    still performs complete freshness and row-shape validation; this
    lightweight predicate can never certify or skip an exact cell.
    """

    try:
        file_stat = path.stat()
        if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_size == 0:
            return False
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            expected_columns = next(csv.reader([expected_header]))
            if header != expected_columns:
                return False
            if requires_data:
                evidence_row = next(reader, None)
                if (
                    evidence_row is None
                    or len(evidence_row) != len(expected_columns)
                ):
                    return False
    except (OSError, UnicodeError, csv.Error):
        return False
    return True


def _prior_root_contains_complete_cell(
    root: Path,
    gtest_case: str,
) -> bool:
    """Return whether one prior root contains the complete cell contract."""

    try:
        directory = root / _gtest_artifact_directory_name(gtest_case)
    except ValueError:
        return False
    return all(
        _prior_artifact_file_has_evidence(
            directory / filename,
            expected_header,
            requires_data=(
                filename in CSV_ARTIFACTS_REQUIRING_DATA
                or filename == "mtp_transactions.csv"
            ),
        )
        for filename, expected_header in _required_csv_headers_for_case(
            gtest_case
        ).items()
    )


def inspect_prior_artifact_evidence(
    cells: Iterable[CampaignCell],
    roots: Iterable[Path],
) -> PriorArtifactEvidenceIndex:
    """Index complete exact cells from explicit prior artifact roots."""

    selected = tuple(cells)
    normalized_roots = tuple(
        dict.fromkeys(Path(root).resolve() for root in roots)
    )
    exact_cases = sorted(
        {
            gtest_case
            for cell in selected
            for gtest_case in cell.gtest_cases
        }
    )
    observed = frozenset(
        gtest_case
        for gtest_case in exact_cases
        if any(
            _prior_root_contains_complete_cell(root, gtest_case)
            for root in normalized_roots
        )
    )
    return PriorArtifactEvidenceIndex(
        roots=normalized_roots,
        observed_gtest_cases=observed,
    )


def prior_evidence_campaign_counts(
    cells: Iterable[CampaignCell],
    evidence: PriorArtifactEvidenceIndex,
) -> dict[PriorEvidenceCoverageKind, int]:
    """Count aggregates in each typed prior-evidence state."""

    counts = {kind: 0 for kind in PriorEvidenceCoverageKind}
    for cell in cells:
        counts[evidence.coverage_for(cell).kind] += 1
    return counts


def validate_campaign_artifacts(
    cell: CampaignCell,
    campaign_started_wall_time_ns: int,
    *,
    revision_results_root: Path | None = None,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    """Validate fresh, schema-exact numerical evidence for every GTest cell.

    Artifact freshness is part of the contract: a previously green directory
    cannot hide a new test that stopped publishing evidence. Header-only files
    are accepted only for the optional decode layer/stage diagnostics; prefill,
    decode-logit, and production-path authorities must each contain data rows.
    """

    if not cell.gtest_cases:
        return 0, (), ()

    root = revision_results_root
    if root is None:
        root = PARITY_RESULTS_DIRECTORY / _current_git_short_hash()

    validated = 0
    directories: list[str] = []
    errors: list[str] = []
    for gtest_case in cell.gtest_cases:
        try:
            directory = root / _gtest_artifact_directory_name(gtest_case)
        except ValueError as error:
            errors.append(str(error))
            continue
        directories.append(str(directory))

        required_headers = _required_csv_headers_for_case(gtest_case)
        for filename, expected_header in required_headers.items():
            path = directory / filename
            try:
                file_stat = path.stat()
            except OSError as error:
                errors.append(f"{gtest_case}: missing {filename}: {error}")
                continue
            if not stat.S_ISREG(file_stat.st_mode):
                errors.append(f"{gtest_case}: {filename} is not a regular file")
                continue
            if (
                revision_results_root is None
                and file_stat.st_mtime_ns < campaign_started_wall_time_ns
            ):
                errors.append(
                    f"{gtest_case}: {filename} is stale "
                    f"(mtime_ns={file_stat.st_mtime_ns}, "
                    f"campaign_start_ns={campaign_started_wall_time_ns})"
                )
                continue

            try:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream))
            except (OSError, UnicodeError, csv.Error) as error:
                errors.append(f"{gtest_case}: cannot parse {filename}: {error}")
                continue
            if not rows:
                errors.append(f"{gtest_case}: {filename} is empty")
                continue

            expected_columns = next(csv.reader([expected_header]))
            if rows[0] != expected_columns:
                errors.append(
                    f"{gtest_case}: {filename} header does not match the "
                    "canonical parity schema"
                )
                continue
            malformed_row = next(
                (
                    row_index
                    for row_index, row in enumerate(rows[1:], start=2)
                    if len(row) != len(expected_columns)
                ),
                None,
            )
            if malformed_row is not None:
                errors.append(
                    f"{gtest_case}: {filename} row {malformed_row} has the "
                    "wrong column count"
                )
                continue
            requires_data = (
                filename in CSV_ARTIFACTS_REQUIRING_DATA
                or filename == "mtp_transactions.csv"
            )
            if requires_data and len(rows) < 2:
                errors.append(f"{gtest_case}: {filename} has no evidence row")
                continue
            validated += 1

    return validated, tuple(directories), tuple(errors)


def create_campaign_artifact_root(report_path: Path) -> Path:
    """Create a unique durable evidence root beside the aggregate report."""

    base = report_path.parent / "production-campaign-artifacts"
    base.mkdir(parents=True, exist_ok=True)
    run_name = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + f"-{os.getpid()}-{time.time_ns()}"
    )
    root = (base / run_name).resolve()
    root.mkdir(mode=0o755)
    return root


def _empty_individual_green_ledger() -> IndividualGreenLedger:
    """Return the only valid initial state for an exact-cell progress ledger."""

    return IndividualGreenLedger(
        schema_version=INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION,
        certification_eligible=False,
        entries=(),
    )


def _decode_individual_green_ledger(document: Any) -> IndividualGreenLedger:
    """Validate a persisted progress ledger without trusting loose JSON."""

    if not isinstance(document, dict):
        raise ValueError("individual green ledger must be a JSON object")
    if document.get("schema_version") != INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION:
        raise ValueError("individual green ledger schema version is unsupported")
    if document.get("certification_eligible") is not False:
        raise ValueError(
            "individual green ledger must be explicitly non-certifying"
        )
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, (list, tuple)):
        raise ValueError("individual green ledger entries must be a sequence")
    entries: list[IndividualGreenEvidence] = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"individual green ledger entry {index} is invalid")
        try:
            entry = IndividualGreenEvidence(**raw_entry)
        except TypeError as error:
            raise ValueError(
                f"individual green ledger entry {index} has invalid fields"
            ) from error
        if (
            not entry.gtest_case
            or not entry.campaign
            or not entry.git_revision
            or entry.passed_wall_time_ns <= 0
            or entry.elapsed_seconds < 0.0
            or entry.validated_artifact_file_count <= 0
            or not entry.artifact_directory
            or re.fullmatch(r"[0-9a-f]{64}", entry.execution_contract_sha256)
            is None
            or entry.provenance not in INDIVIDUAL_GREEN_PROVENANCE
        ):
            raise ValueError(
                f"individual green ledger entry {index} has invalid evidence"
            )
        entries.append(entry)
    cases = [entry.gtest_case for entry in entries]
    if len(set(cases)) != len(cases):
        raise ValueError("individual green ledger contains duplicate exact cells")
    return IndividualGreenLedger(
        schema_version=INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION,
        certification_eligible=False,
        entries=tuple(entries),
    )


def load_individual_green_ledger(path: Path) -> IndividualGreenLedger:
    """Load one durable progress ledger, or return an empty typed ledger."""

    try:
        payload = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _empty_individual_green_ledger()
    try:
        return _decode_individual_green_ledger(json.loads(payload))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError(f"cannot parse individual green ledger {path}") from error


def _write_individual_green_ledger_unlocked(
    path: Path,
    ledger: IndividualGreenLedger,
) -> None:
    """Atomically publish a fully validated non-certifying progress ledger."""

    validated = _decode_individual_green_ledger(asdict(ledger))
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(asdict(validated), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def record_individual_green(
    path: Path,
    cell: CampaignCell,
    result: CampaignResult,
) -> IndividualGreenLedger:
    """Record one green only after exact exit-zero and fresh CSV validation."""

    if len(result.gtest_cases) != 1:
        raise ValueError("individual green result must name exactly one GTest cell")
    gtest_case = result.gtest_cases[0]
    expected_artifact_count = len(_required_csv_headers_for_case(gtest_case))
    if (
        result.campaign != cell.name
        or gtest_case not in cell.gtest_cases
        or result.return_code != 0
        or result.outcome != "completed"
        or not result.artifact_contract_passed
        or result.artifact_errors
        or result.validated_artifact_file_count != expected_artifact_count
        or len(result.artifact_directories) != 1
    ):
        raise ValueError(
            f"refusing to record non-green exact result for {gtest_case}"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        ledger = load_individual_green_ledger(path)
        if gtest_case in ledger.green_cases:
            raise ValueError(f"exact cell is already green: {gtest_case}")
        entry = IndividualGreenEvidence(
            gtest_case=gtest_case,
            campaign=cell.name,
            git_revision=_current_git_short_hash(),
            passed_wall_time_ns=time.time_ns(),
            elapsed_seconds=result.elapsed_seconds,
            validated_artifact_file_count=(
                result.validated_artifact_file_count
            ),
            artifact_directory=result.artifact_directories[0],
            execution_contract_sha256=_exact_execution_contract_sha256(
                cell,
                gtest_case,
            ),
        )
        updated = IndividualGreenLedger(
            schema_version=INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION,
            certification_eligible=False,
            entries=(*ledger.entries, entry),
        )
        _write_individual_green_ledger_unlocked(path, updated)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return updated


def seed_individual_green_ledger_from_campaign_report(
    report_path: Path,
    cells: Iterable[CampaignCell],
    ledger_path: Path,
    *,
    declared_first_reds: Iterable[tuple[str, str]] = (),
) -> IndividualGreenLedger:
    """Import only aggregate greens and audited fail-fast green prefixes.

    A failed aggregate is never inferred green from complete CSVs. Its prefix
    is importable only when the caller supplies the exact first-red identity
    observed in GTest output and the current registration still requires
    ``GTEST_FAIL_FAST=1``. This is an interruption-recovery tool; imported
    entries remain explicitly non-certifying.
    """

    selected = tuple(cells)
    by_campaign = {cell.name: cell for cell in selected}
    if len(by_campaign) != len(selected):
        raise ValueError("selected campaigns contain duplicate identities")
    first_red_items = tuple(declared_first_reds)
    first_red_map = dict(first_red_items)
    if len(first_red_map) != len(first_red_items):
        raise ValueError("declared first-red campaigns must be unique")
    unknown_first_reds = set(first_red_map) - set(by_campaign)
    if unknown_first_reds:
        raise ValueError(
            "declared first red names an unselected campaign: "
            + ", ".join(sorted(unknown_first_reds))
        )

    try:
        document = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read campaign report {report_path}") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") not in {12, CAMPAIGN_REPORT_SCHEMA_VERSION}
    ):
        raise ValueError("campaign report has an unsupported schema")
    raw_campaigns = document.get("campaigns")
    if not isinstance(raw_campaigns, list):
        raise ValueError("campaign report has no campaign result list")
    try:
        artifact_root = Path(str(document["artifact_root"])).resolve(strict=True)
    except (KeyError, OSError) as error:
        raise ValueError("campaign report artifact root is unavailable") from error
    if not artifact_root.is_dir():
        raise ValueError("campaign report artifact root is not a directory")

    report_results: dict[str, dict[str, Any]] = {}
    for raw_result in raw_campaigns:
        if not isinstance(raw_result, dict):
            raise ValueError("campaign report contains an invalid result")
        campaign_name = str(raw_result.get("campaign", ""))
        if campaign_name in report_results:
            raise ValueError("campaign report contains duplicate results")
        report_results[campaign_name] = raw_result

    candidates: list[tuple[CampaignCell, str, str]] = []
    consumed_first_reds: set[str] = set()
    for campaign_name, raw_result in report_results.items():
        cell = by_campaign.get(campaign_name)
        if cell is None:
            continue
        reported_cases = tuple(str(case) for case in raw_result.get("gtest_cases", []))
        if reported_cases != cell.gtest_cases:
            raise ValueError(
                f"campaign report matrix changed for {campaign_name}"
            )
        if (
            raw_result.get("return_code") == 0
            and raw_result.get("outcome") == "completed"
            and raw_result.get("artifact_contract_passed") is True
            and not raw_result.get("artifact_errors")
        ):
            expected_count = sum(
                len(_required_csv_headers_for_case(gtest_case))
                for gtest_case in cell.gtest_cases
            )
            if raw_result.get("validated_artifact_file_count") != expected_count:
                raise ValueError(
                    f"green campaign report has incomplete artifacts: {campaign_name}"
                )
            candidates.extend(
                (
                    cell,
                    gtest_case,
                    "aggregate_exit_zero_and_fresh_artifact_contract",
                )
                for gtest_case in cell.gtest_cases
            )
            continue

        first_red = first_red_map.get(campaign_name)
        if first_red is None:
            continue
        if raw_result.get("return_code") in (None, 0):
            raise ValueError(
                f"declared first-red campaign did not fail: {campaign_name}"
            )
        if "GTEST_FAIL_FAST=1" not in cell.environment:
            raise ValueError(
                f"declared first-red campaign lacks fail-fast: {campaign_name}"
            )
        try:
            first_red_index = cell.gtest_cases.index(first_red)
        except ValueError as error:
            raise ValueError(
                f"declared first red is not registered by {campaign_name}"
            ) from error
        candidates.extend(
            (
                cell,
                gtest_case,
                "aggregate_gtest_fail_fast_prefix_before_declared_first_red",
            )
            for gtest_case in cell.gtest_cases[:first_red_index]
        )
        consumed_first_reds.add(campaign_name)
    if consumed_first_reds != set(first_red_map):
        missing = set(first_red_map) - consumed_first_reds
        raise ValueError(
            "declared first-red campaign is absent from report: "
            + ", ".join(sorted(missing))
        )

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = ledger_path.with_name(ledger_path.name + ".lock")
    with lock_path.open("a+", encoding="utf-8") as lock_stream:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX)
        ledger = load_individual_green_ledger(ledger_path)
        validate_green_ledger_selection(selected, ledger)
        entries = list(ledger.entries)
        observed = set(ledger.green_cases)
        revision = _current_git_short_hash()
        for cell, gtest_case, provenance in candidates:
            if gtest_case in observed:
                continue
            if not _prior_root_contains_complete_cell(
                artifact_root,
                gtest_case,
            ):
                raise ValueError(
                    f"campaign report lacks complete artifacts for {gtest_case}"
                )
            directory = artifact_root / _gtest_artifact_directory_name(
                gtest_case
            )
            passed_wall_time_ns = max(
                (directory / filename).stat().st_mtime_ns
                for filename in _required_csv_headers_for_case(gtest_case)
            )
            entries.append(
                IndividualGreenEvidence(
                    gtest_case=gtest_case,
                    campaign=cell.name,
                    git_revision=revision,
                    passed_wall_time_ns=passed_wall_time_ns,
                    elapsed_seconds=0.0,
                    validated_artifact_file_count=len(
                        _required_csv_headers_for_case(gtest_case)
                    ),
                    artifact_directory=str(directory),
                    execution_contract_sha256=(
                        _exact_execution_contract_sha256(cell, gtest_case)
                    ),
                    provenance=provenance,
                )
            )
            observed.add(gtest_case)
        updated = IndividualGreenLedger(
            schema_version=INDIVIDUAL_GREEN_LEDGER_SCHEMA_VERSION,
            certification_eligible=False,
            entries=tuple(entries),
        )
        _write_individual_green_ledger_unlocked(ledger_path, updated)
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
    return updated


def unseen_individual_cells(
    cells: Iterable[CampaignCell],
    ledger: IndividualGreenLedger,
) -> tuple[tuple[CampaignCell, str], ...]:
    """Return only exact identities absent from proven-green progress."""

    selected = tuple(cells)
    all_cases = [
        gtest_case
        for cell in selected
        for gtest_case in cell.gtest_cases
    ]
    if len(set(all_cases)) != len(all_cases):
        raise ValueError("selected campaigns contain duplicate exact GTest cells")
    return tuple(
        (cell, gtest_case)
        for cell in selected
        for gtest_case in cell.gtest_cases
        if gtest_case not in ledger.green_cases
    )


def validate_green_ledger_selection(
    cells: Iterable[CampaignCell],
    ledger: IndividualGreenLedger,
) -> None:
    """Reject selected ledger entries whose registered launch contract moved."""

    owners = {
        gtest_case: cell
        for cell in cells
        for gtest_case in cell.gtest_cases
    }
    for entry in ledger.entries:
        cell = owners.get(entry.gtest_case)
        if cell is None:
            continue
        if entry.campaign != cell.name:
            raise ValueError(
                f"green ledger campaign changed for {entry.gtest_case}"
            )
        current_contract = _exact_execution_contract_sha256(
            cell,
            entry.gtest_case,
        )
        if entry.execution_contract_sha256 != current_contract:
            raise ValueError(
                f"green ledger execution contract changed for "
                f"{entry.gtest_case}"
            )


def campaign_resources(cell: CampaignCell) -> frozenset[str]:
    """Return the exclusive backend resources claimed by one campaign."""

    resources = frozenset(cell.group.backends.split("+"))
    if not resources or not resources.issubset(BACKEND_ORDER):
        raise ValueError(
            f"campaign {cell.name} has invalid backend resources: "
            f"{sorted(resources)}"
        )
    return resources


def scheduling_order(
    cells: Iterable[CampaignCell],
    prior_artifact_roots: Iterable[Path] = (),
    *,
    prior_evidence: PriorArtifactEvidenceIndex | None = None,
) -> list[CampaignCell]:
    """Prioritize unseen work, then wide claims, deterministically.

    Without explicit prior roots this preserves the canonical wide-first
    ordering exactly. Prior evidence only changes ordering: every selected
    aggregate and every exact GTest case remains in the returned schedule.
    """

    selected = list(cells)
    roots = tuple(prior_artifact_roots)
    if prior_evidence is not None and roots:
        raise ValueError(
            "provide prior artifact roots or a prior evidence index, not both"
        )
    evidence = prior_evidence
    if evidence is None and roots:
        evidence = inspect_prior_artifact_evidence(selected, roots)

    if evidence is None or not evidence.roots:
        return sorted(
            selected,
            key=lambda cell: (
                -len(campaign_resources(cell)),
                cell.group.backends,
                cell.name,
            ),
        )

    coverage_rank = {
        PriorEvidenceCoverageKind.UNSEEN: 0,
        PriorEvidenceCoverageKind.PARTIAL: 1,
        PriorEvidenceCoverageKind.COMPLETE: 2,
    }

    def priority(cell: CampaignCell) -> tuple[int, int, int, str, str]:
        coverage = evidence.coverage_for(cell)
        return (
            coverage_rank[coverage.kind],
            -coverage.unseen_exact_cells,
            -len(campaign_resources(cell)),
            cell.group.backends,
            cell.name,
        )

    return sorted(selected, key=priority)


def select_runnable_campaigns(
    pending: Iterable[CampaignCell],
    occupied_resources: Iterable[str],
) -> list[CampaignCell]:
    """Greedily select a maximal backend-disjoint launch set."""

    claimed = set(occupied_resources)
    selected: list[CampaignCell] = []
    for cell in pending:
        resources = campaign_resources(cell)
        if claimed.isdisjoint(resources):
            selected.append(cell)
            claimed.update(resources)
    return selected


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate one exact process group, escalating after a short grace."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_process(
    command: list[str],
    timeout_seconds: float | None,
    environment_overrides: dict[str, str] | None = None,
    *,
    working_directory: Path | None = None,
    exact_cell_watch: ExactCellTimeoutWatch | None = None,
    cancellation: CampaignCancellation | None = None,
    termination_evidence: ProcessTerminationEvidence | None = None,
) -> int:
    """Run one process group under cancellation and exact typed deadlines.

    The exact-cell watchdog is deliberately file-driven. CTest buffers child
    output, whereas the parity fixture publishes its unique ``test_log.txt``
    synchronously when a generated cell starts. Observing that existing
    artifact boundary keeps the watchdog independent of MPI stdout ordering
    and avoids splitting an aggregate that shares immutable prepared weights.
    When that typed watch exists it is the sole timeout authority: each exact
    cell receives the complete budget, so a healthy many-cell aggregate cannot
    be killed by an unrelated cumulative setup deadline.
    """

    if exact_cell_watch is not None:
        if exact_cell_watch.timeout_seconds <= 0.0:
            raise ValueError("exact-cell timeout must be positive")
        cases = [case for case, _ in exact_cell_watch.progress_files]
        paths = [path for _, path in exact_cell_watch.progress_files]
        if len(set(cases)) != len(cases):
            raise ValueError("exact-cell timeout watch contains duplicate cases")
        if len(set(paths)) != len(paths):
            raise ValueError("exact-cell timeout watch contains duplicate paths")

    if termination_evidence is not None:
        termination_evidence.kind = ProcessTerminationKind.NONE
        termination_evidence.exact_gtest_case = ""
        termination_evidence.cancelling_campaign = ""

    # A worker selected before the first failure can reach this boundary after
    # cancellation publication. Do not create even a short-lived CTest/MPI
    # process in that state.
    if cancellation is not None and cancellation.requested:
        if termination_evidence is not None:
            termination_evidence.kind = (
                ProcessTerminationKind.CAMPAIGN_CANCELLED
            )
            termination_evidence.cancelling_campaign = (
                cancellation.failing_campaign
            )
        return 130

    environment = None
    if environment_overrides:
        environment = os.environ.copy()
        environment.update(environment_overrides)
    process = subprocess.Popen(
        command,
        start_new_session=True,
        env=environment,
        cwd=working_directory,
    )
    started = time.monotonic()
    completion_deadline = (
        None
        if timeout_seconds is None or exact_cell_watch is not None
        else started + max(timeout_seconds, 0.001)
    )

    if exact_cell_watch is None and cancellation is None:
        try:
            if completion_deadline is None:
                return process.wait()
            return process.wait(
                timeout=max(completion_deadline - time.monotonic(), 0.001)
            )
        except subprocess.TimeoutExpired:
            if termination_evidence is not None:
                termination_evidence.kind = (
                    ProcessTerminationKind.COMPLETION_TIMEOUT
                )
            _terminate_process_group(process)
            return 124

    if exact_cell_watch is None:
        while True:
            if cancellation is not None and cancellation.requested:
                if termination_evidence is not None:
                    termination_evidence.kind = (
                        ProcessTerminationKind.CAMPAIGN_CANCELLED
                    )
                    termination_evidence.cancelling_campaign = (
                        cancellation.failing_campaign
                    )
                _terminate_process_group(process)
                return 130

            return_code = process.poll()
            if return_code is not None:
                return return_code

            now = time.monotonic()
            if completion_deadline is not None and now >= completion_deadline:
                if termination_evidence is not None:
                    termination_evidence.kind = (
                        ProcessTerminationKind.COMPLETION_TIMEOUT
                    )
                _terminate_process_group(process)
                return 124

            poll_seconds = 0.25
            if completion_deadline is not None:
                poll_seconds = min(
                    poll_seconds,
                    max(completion_deadline - now, 0.001),
                )
            try:
                return process.wait(timeout=poll_seconds)
            except subprocess.TimeoutExpired:
                pass

    # Startup belongs to the first exact cell. Creating its progress file does
    # not restart the clock; only publication of a different cell transfers
    # watchdog authority and begins a fresh ten-minute budget.
    current_case = (
        exact_cell_watch.progress_files[0][0]
        if exact_cell_watch.progress_files
        else ""
    )
    cell_deadline = started + exact_cell_watch.timeout_seconds
    observed_any_cell = False
    observed_paths: set[Path] = set()

    while True:
        if cancellation is not None and cancellation.requested:
            if termination_evidence is not None:
                termination_evidence.kind = (
                    ProcessTerminationKind.CAMPAIGN_CANCELLED
                )
                termination_evidence.cancelling_campaign = (
                    cancellation.failing_campaign
                )
            _terminate_process_group(process)
            return 130

        now = time.monotonic()
        newly_started: list[tuple[int, str, Path]] = []
        for case, path in exact_cell_watch.progress_files:
            if path in observed_paths:
                continue
            try:
                modified_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            if modified_ns < exact_cell_watch.not_before_wall_time_ns:
                continue
            newly_started.append((modified_ns, case, path))

        for _, case, path in sorted(newly_started):
            observed_paths.add(path)
            if not observed_any_cell:
                current_case = case
                observed_any_cell = True
            elif case != current_case:
                current_case = case
                cell_deadline = now + exact_cell_watch.timeout_seconds

        return_code = process.poll()
        if return_code is not None:
            return return_code

        now = time.monotonic()
        exact_cell_expired = now >= cell_deadline
        completion_expired = (
            completion_deadline is not None and now >= completion_deadline
        )
        if exact_cell_expired and (
            completion_deadline is None or cell_deadline <= completion_deadline
        ):
            if termination_evidence is not None:
                termination_evidence.kind = (
                    ProcessTerminationKind.EXACT_CELL_TIMEOUT
                )
                termination_evidence.exact_gtest_case = current_case
            print(
                "[production-parity] exact_cell_timeout "
                f"seconds={exact_cell_watch.timeout_seconds:.3f} "
                f"gtest_case={current_case or '<before-first-cell>'}",
                file=sys.stderr,
                flush=True,
            )
            _terminate_process_group(process)
            return 124
        if completion_expired:
            if termination_evidence is not None:
                termination_evidence.kind = (
                    ProcessTerminationKind.COMPLETION_TIMEOUT
                )
            _terminate_process_group(process)
            return 124

        deadlines = [cell_deadline]
        if completion_deadline is not None:
            deadlines.append(completion_deadline)
        poll_seconds = min(0.25, max(min(deadlines) - now, 0.001))
        try:
            return process.wait(timeout=poll_seconds)
        except subprocess.TimeoutExpired:
            pass


def run_production_parity_preflight(
    build_dir: Path,
    timeout_seconds: float | None,
) -> tuple[int, float, tuple[str, ...]]:
    """Build and run every model-free prerequisite before model staging.

    The phases deliberately have separate, source-owned inventories: CTest's
    complete Unit namespace and the Integration-only preflight label.  Their
    concatenated identities remain the existing report receipt, so a green
    campaign proves both without adding a parallel manifest.
    """

    unit_tests = discover_production_parity_unit_tests(build_dir)
    integration_tests = discover_production_parity_preflight_tests(build_dir)
    tests = unit_tests + integration_tests
    timeout_text = (
        "disabled" if timeout_seconds is None else f"{timeout_seconds:.3f}"
    )
    started = time.monotonic()
    deadline = (
        None if timeout_seconds is None else started + timeout_seconds
    )

    def remaining_timeout() -> float | None:
        """Return this prerequisite transaction's remaining wall time."""

        if deadline is None:
            return None
        return max(deadline - time.monotonic(), 0.001)

    print(
        "[production-parity] preflight_status=RUNNING "
        f"unit_test_count={len(unit_tests)} "
        f"integration_test_count={len(integration_tests)} "
        f"test_count={len(tests)} "
        f"completion_timeout_remaining_seconds={timeout_text}",
        flush=True,
    )

    build_command = [
        "cmake",
        "--build",
        str(build_dir),
        "--parallel",
        "--target",
        PRODUCTION_PARITY_UNIT_BUILD_TARGET,
    ]
    print(
        "[production-parity] unit_build_status=RUNNING "
        f"target={PRODUCTION_PARITY_UNIT_BUILD_TARGET}",
        flush=True,
    )
    return_code = _run_process(build_command, remaining_timeout())
    print(
        "[production-parity] unit_build_status="
        f"{'PASS' if return_code == 0 else 'FAIL'} "
        f"target={PRODUCTION_PARITY_UNIT_BUILD_TARGET}",
        flush=True,
    )
    if return_code != 0:
        elapsed = time.monotonic() - started
        print(
            "[production-parity] preflight_status=FAIL "
            f"phase=unit_build test_count={len(tests)} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        return return_code, elapsed, tests

    # Ninja can regenerate CMake while building the gate, adding or removing
    # registrations. CTest below consumes that regenerated inventory, so its
    # receipt must name those exact tests rather than the pre-build snapshot.
    unit_tests = discover_production_parity_unit_tests(build_dir)
    integration_tests = discover_production_parity_preflight_tests(build_dir)
    tests = unit_tests + integration_tests

    unit_command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--parallel",
        "--no-tests=error",
        "-R",
        f"^{PRODUCTION_PARITY_UNIT_PREFIX}",
    ]
    print(
        "[production-parity] unit_test_status=RUNNING "
        f"test_count={len(unit_tests)}",
        flush=True,
    )
    return_code = _run_process(unit_command, remaining_timeout())
    print(
        "[production-parity] unit_test_status="
        f"{'PASS' if return_code == 0 else 'FAIL'} "
        f"test_count={len(unit_tests)}",
        flush=True,
    )
    if return_code != 0:
        elapsed = time.monotonic() - started
        print(
            "[production-parity] preflight_status=FAIL "
            f"phase=unit_test test_count={len(tests)} "
            f"elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
        return return_code, elapsed, tests

    integration_command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--parallel",
        "--no-tests=error",
        "-L",
        f"^{PRODUCTION_PARITY_PREFLIGHT_LABEL}$",
    ]
    print(
        "[production-parity] integration_preflight_status=RUNNING "
        f"test_count={len(integration_tests)}",
        flush=True,
    )
    return_code = _run_process(integration_command, remaining_timeout())
    elapsed = time.monotonic() - started
    print(
        "[production-parity] integration_preflight_status="
        f"{'PASS' if return_code == 0 else 'FAIL'} "
        f"test_count={len(integration_tests)}",
        flush=True,
    )
    print(
        "[production-parity] preflight_status="
        f"{'PASS' if return_code == 0 else 'FAIL'} "
        f"test_count={len(tests)} elapsed_seconds={elapsed:.3f}",
        flush=True,
    )
    return return_code, elapsed, tests


def reuse_unchanged_production_parity_preflight(
    build_dir: Path,
    report_path: Path,
) -> tuple[int, float, tuple[str, ...]]:
    """Reuse a passed individual preflight only for an unchanged build tree.

    The prior report is evidence that the canonical runner admitted its cell
    only after a successful preflight. Ninja's append-only command log is the
    conservative rebuild boundary, while every generated CTest registration
    file is the inventory boundary. A newer boundary makes the receipt stale
    and forces the caller to run preflight again; there is deliberately no raw
    ``--skip-preflight`` switch.
    """

    resolved_report = report_path.expanduser().resolve(strict=True)
    try:
        document = json.loads(resolved_report.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError, OSError) as error:
        raise ValueError(
            f"cannot read reusable preflight report {resolved_report}"
        ) from error
    if not isinstance(document, dict):
        raise ValueError("reusable preflight report must be a JSON object")
    if (
        document.get("schema_version") != INDIVIDUAL_PROGRESS_REPORT_SCHEMA_VERSION
        or document.get("mode") != "sequential_unseen_exact_cells"
        or document.get("certification_eligible") is not False
        or document.get("preflight_return_code") != 0
    ):
        raise ValueError(
            "reusable preflight report is not a passed individual-run receipt"
        )

    resolved_build = build_dir.expanduser().resolve(strict=True)
    build_boundaries = [
        resolved_build / ".ninja_log",
        resolved_build / "build.ninja",
        *sorted(resolved_build.rglob("CTestTestfile.cmake")),
    ]
    if len(build_boundaries) < 3:
        raise ValueError(
            f"build tree has no generated CTest registration: {resolved_build}"
        )
    report_mtime_ns = resolved_report.stat().st_mtime_ns
    for boundary in build_boundaries:
        try:
            boundary_mtime_ns = boundary.stat().st_mtime_ns
        except OSError as error:
            raise ValueError(
                f"preflight build-identity boundary is unavailable: {boundary}"
            ) from error
        if boundary_mtime_ns > report_mtime_ns:
            raise ValueError(
                "reusable preflight report is stale because the build or CTest "
                f"registration changed afterward: {boundary}"
            )

    tests = (
        discover_production_parity_unit_tests(resolved_build)
        + discover_production_parity_preflight_tests(resolved_build)
    )
    reported_tests = tuple(document.get("preflight_tests", ()))
    if (
        document.get("preflight_test_count") != len(tests)
        or reported_tests != tests
    ):
        raise ValueError(
            "reusable preflight report does not prove the current complete "
            "unit and integration prerequisite inventory"
        )
    print(
        "[production-parity] preflight_status=REUSED "
        f"test_count={len(tests)} evidence_report={resolved_report} "
        "build_identity=unchanged",
        flush=True,
    )
    return 0, 0.0, tests


def prepare_model_fixture(
    build_dir: Path,
    timeout_seconds: float | None,
) -> tuple[int, float]:
    """Stage model fixtures exactly once before concurrent campaign admission."""

    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--no-tests=error",
        "-R",
        f"^{MODEL_FIXTURE_TEST}$",
    ]
    started = time.monotonic()
    return_code = _run_process(command, timeout_seconds)
    return return_code, time.monotonic() - started


def run_individual_cell(
    cell: CampaignCell,
    gtest_case: str,
    completion_timeout_seconds: float | None,
    *,
    exact_cell_timeout_seconds: float = EXACT_CELL_TIMEOUT_SECONDS,
    global_started_at: float | None = None,
    target_seconds: float = GLOBAL_TARGET_SECONDS,
    environment_overrides: dict[str, str] | None = None,
    artifact_results_root: Path | None = None,
) -> CampaignResult:
    """Run one exact registered production command and validate fresh CSVs.

    This path exists only for sequential failure discovery. It deliberately
    bypasses CTest's aggregate boundary while retaining CTest's exact launch
    argv, environment, working directory, MPI topology, and model declaration.
    The resulting progress is non-certifying until the ordinary unfiltered
    campaign reruns every cell together.
    """

    exact_cell = _single_exact_cell(cell, gtest_case)
    if not exact_cell.working_directory:
        raise ValueError(
            f"campaign {cell.name} has no registered working directory"
        )
    environment = _registered_environment(exact_cell)
    if environment_overrides:
        environment.update(environment_overrides)
    timeout_text = (
        "disabled"
        if completion_timeout_seconds is None
        else f"{completion_timeout_seconds:.3f}"
    )
    print(
        "[production-parity] individual_cell_status=RUNNING "
        f"campaign={cell.name} gtest_case={gtest_case} "
        f"resources={'+'.join(sorted(campaign_resources(cell)))} "
        f"completion_timeout_remaining_seconds={timeout_text} "
        f"exact_cell_timeout_seconds={exact_cell_timeout_seconds:.3f}",
        flush=True,
    )

    started = time.monotonic()
    started_wall_time_ns = time.time_ns()
    termination_evidence = ProcessTerminationEvidence()
    process_return_code = _run_process(
        list(exact_cell.command),
        completion_timeout_seconds,
        environment_overrides=environment,
        working_directory=Path(exact_cell.working_directory),
        exact_cell_watch=_exact_cell_timeout_watch(
            exact_cell,
            artifact_results_root,
            exact_cell_timeout_seconds,
            started_wall_time_ns,
        ),
        termination_evidence=termination_evidence,
    )
    try:
        artifact_count, artifact_directories, artifact_errors = (
            validate_campaign_artifacts(
                exact_cell,
                started_wall_time_ns,
                revision_results_root=artifact_results_root,
            )
        )
    except (OSError, RuntimeError, ValueError) as error:
        artifact_count = 0
        artifact_directories = ()
        artifact_errors = (f"artifact validation failed: {error}",)

    return_code = process_return_code
    if (
        process_return_code == 124
        and termination_evidence.kind
        is ProcessTerminationKind.EXACT_CELL_TIMEOUT
    ):
        outcome = "exact_cell_timeout"
    elif process_return_code == 124:
        outcome = "completion_timeout"
    else:
        outcome = "completed"
    if process_return_code == 0 and artifact_errors:
        return_code = 126
        outcome = "artifact_contract_failed"
    for error in artifact_errors:
        print(
            "[production-parity] individual_cell_artifact_error "
            f"gtest_case={gtest_case} error={error}",
            file=sys.stderr,
            flush=True,
        )

    finished = time.monotonic()
    elapsed = finished - started
    finished_offset = (
        finished - global_started_at
        if global_started_at is not None
        else elapsed
    )
    result = CampaignResult(
        campaign=cell.name,
        test_type=cell.test_type,
        backends=cell.group.backends,
        precision_set=cell.group.kv_precision,
        precision_types=exact_cell.precision_types,
        gtest_cases=(gtest_case,),
        elapsed_seconds=elapsed,
        target_seconds=target_seconds,
        return_code=return_code,
        target_met=finished_offset <= target_seconds,
        started_offset_seconds=(
            started - global_started_at
            if global_started_at is not None
            else 0.0
        ),
        finished_offset_seconds=finished_offset,
        completion_timeout_seconds=(completion_timeout_seconds or 0.0),
        exact_cell_timeout_seconds=exact_cell_timeout_seconds,
        timed_out_gtest_case=termination_evidence.exact_gtest_case,
        outcome=outcome,
        artifact_contract_passed=not artifact_errors,
        validated_artifact_file_count=artifact_count,
        artifact_directories=artifact_directories,
        artifact_errors=artifact_errors,
    )
    print(
        "[production-parity] individual_cell_status="
        f"{'PASS' if result.return_code == 0 else 'FAIL'} "
        f"gtest_case={gtest_case} elapsed_seconds={elapsed:.3f} "
        f"validated_artifacts={artifact_count}",
        flush=True,
    )
    return result


def run_campaign(
    build_dir: Path,
    cell: CampaignCell,
    completion_timeout_seconds: float | None,
    *,
    exact_cell_timeout_seconds: float = EXACT_CELL_TIMEOUT_SECONDS,
    global_started_at: float | None = None,
    target_seconds: float = GLOBAL_TARGET_SECONDS,
    environment_overrides: dict[str, str] | None = None,
    artifact_results_root: Path | None = None,
    cancellation: CampaignCancellation | None = None,
) -> CampaignResult:
    """Run one campaign and authenticate its newly written CSV evidence."""

    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "--output-on-failure",
        "--progress",
        "--no-tests=error",
        "-FA",
        MODEL_FIXTURE_NAME,
        "-R",
        _ctest_exact_regex([cell]),
    ]
    timeout_text = (
        "disabled"
        if completion_timeout_seconds is None
        else f"{completion_timeout_seconds:.3f}"
    )
    print(
        f"[production-parity] campaign={cell.name} "
        f"resources={'+'.join(sorted(campaign_resources(cell)))} "
        f"completion_timeout_remaining_seconds={timeout_text} "
        f"exact_cell_timeout_seconds={exact_cell_timeout_seconds:.3f}",
        flush=True,
    )
    started = time.monotonic()
    started_wall_time_ns = time.time_ns()
    termination_evidence = ProcessTerminationEvidence()
    process_return_code = _run_process(
        command,
        completion_timeout_seconds,
        environment_overrides=environment_overrides,
        exact_cell_watch=_exact_cell_timeout_watch(
            cell,
            artifact_results_root,
            exact_cell_timeout_seconds,
            started_wall_time_ns,
        ),
        cancellation=cancellation,
        termination_evidence=termination_evidence,
    )

    cancelled = (
        termination_evidence.kind
        is ProcessTerminationKind.CAMPAIGN_CANCELLED
    )
    if process_return_code != 0 and not cancelled:
        # Publish a process failure before parsing artifacts so active sibling
        # campaigns stop immediately rather than consuming more model time.
        _publish_campaign_failure(cancellation, cell.name)

    if cancelled:
        artifact_count = 0
        artifact_directories = ()
        artifact_errors = ()
    else:
        try:
            artifact_count, artifact_directories, artifact_errors = (
                validate_campaign_artifacts(
                    cell,
                    started_wall_time_ns,
                    revision_results_root=artifact_results_root,
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            artifact_count = 0
            artifact_directories = ()
            artifact_errors = (f"artifact validation failed: {error}",)

    return_code = process_return_code
    if (
        process_return_code == 124
        and termination_evidence.kind
        is ProcessTerminationKind.EXACT_CELL_TIMEOUT
    ):
        outcome = "exact_cell_timeout"
    elif cancelled:
        outcome = "cancelled_after_campaign_failure"
    elif process_return_code == 124:
        outcome = "completion_timeout"
    else:
        outcome = "completed"
    if process_return_code == 0 and artifact_errors:
        return_code = 126
        outcome = "artifact_contract_failed"
        _publish_campaign_failure(cancellation, cell.name)
        for error in artifact_errors:
            print(
                f"[production-parity] campaign={cell.name} "
                f"artifact_error={error}",
                file=sys.stderr,
                flush=True,
            )
    finished = time.monotonic()
    elapsed = finished - started
    finished_offset = (
        finished - global_started_at
        if global_started_at is not None
        else elapsed
    )
    return CampaignResult(
        campaign=cell.name,
        test_type=cell.test_type,
        backends=cell.group.backends,
        precision_set=cell.group.kv_precision,
        precision_types=cell.precision_types,
        gtest_cases=cell.gtest_cases,
        elapsed_seconds=elapsed,
        target_seconds=target_seconds,
        return_code=return_code,
        target_met=finished_offset <= target_seconds,
        started_offset_seconds=(
            started - global_started_at
            if global_started_at is not None
            else 0.0
        ),
        finished_offset_seconds=finished_offset,
        completion_timeout_seconds=(completion_timeout_seconds or 0.0),
        exact_cell_timeout_seconds=exact_cell_timeout_seconds,
        timed_out_gtest_case=termination_evidence.exact_gtest_case,
        cancelled_by_campaign=termination_evidence.cancelling_campaign,
        outcome=outcome,
        artifact_contract_passed=not cancelled and not artifact_errors,
        validated_artifact_file_count=artifact_count,
        artifact_directories=artifact_directories,
        artifact_errors=artifact_errors,
    )


def _run_campaign_and_publish_failure(
    build_dir: Path,
    cell: CampaignCell,
    completion_timeout_seconds: float | None,
    *,
    exact_cell_timeout_seconds: float,
    global_started_at: float,
    target_seconds: float,
    environment_overrides: dict[str, str] | None,
    artifact_results_root: Path | None,
    cancellation: CampaignCancellation,
) -> CampaignResult:
    """Run one worker and publish even catastrophic worker-boundary failures."""

    try:
        result = run_campaign(
            build_dir,
            cell,
            completion_timeout_seconds,
            exact_cell_timeout_seconds=exact_cell_timeout_seconds,
            global_started_at=global_started_at,
            target_seconds=target_seconds,
            environment_overrides=environment_overrides,
            artifact_results_root=artifact_results_root,
            cancellation=cancellation,
        )
    except BaseException:
        _publish_campaign_failure(cancellation, cell.name)
        raise

    if (
        result.return_code != 0
        and result.outcome != "cancelled_after_campaign_failure"
    ):
        _publish_campaign_failure(cancellation, cell.name)
    return result


def _not_run_result(
    cell: CampaignCell,
    target_seconds: float,
    elapsed_seconds: float,
    *,
    return_code: int = 124,
    outcome: str = "not_started_before_completion_timeout",
    cancelled_by_campaign: str = "",
) -> CampaignResult:
    """Describe a campaign that infrastructure prevented from starting."""

    return CampaignResult(
        campaign=cell.name,
        test_type=cell.test_type,
        backends=cell.group.backends,
        precision_set=cell.group.kv_precision,
        precision_types=cell.precision_types,
        gtest_cases=cell.gtest_cases,
        elapsed_seconds=0.0,
        target_seconds=target_seconds,
        return_code=return_code,
        target_met=elapsed_seconds <= target_seconds,
        started_offset_seconds=elapsed_seconds,
        finished_offset_seconds=elapsed_seconds,
        completion_timeout_seconds=0.0,
        cancelled_by_campaign=cancelled_by_campaign,
        outcome=outcome,
        artifact_contract_passed=False,
        artifact_errors=(
            ()
            if cancelled_by_campaign
            else ("campaign did not execute, so no fresh artifacts exist",)
        ),
    )


def run_campaign_matrix(
    build_dir: Path,
    cells: Iterable[CampaignCell],
    target_seconds: float,
    *,
    exact_cell_timeout_seconds: float = EXACT_CELL_TIMEOUT_SECONDS,
    global_started_at: float | None = None,
    environment_overrides: dict[str, str] | None = None,
    artifact_results_root: Path | None = None,
    prior_evidence: PriorArtifactEvidenceIndex | None = None,
) -> tuple[list[CampaignResult], float]:
    """Run until the first red, cancelling siblings and preserving evidence.

    Exact-cell progress is the only inference timeout authority. The soft
    whole-matrix target remains reporting evidence and never controls campaign
    admission, regardless of how many process-amortized aggregates are
    selected.
    """

    selected = list(cells)
    started = (
        global_started_at if global_started_at is not None else time.monotonic()
    )
    pending = scheduling_order(selected, prior_evidence=prior_evidence)
    original_order = {cell.name: index for index, cell in enumerate(selected)}
    results: list[CampaignResult] = []
    running: dict[
        concurrent.futures.Future[CampaignResult],
        tuple[CampaignCell, frozenset[str]],
    ] = {}
    occupied: set[str] = set()
    cancellation = CampaignCancellation()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(BACKEND_ORDER),
        thread_name_prefix="production-parity",
    ) as executor:
        while pending or running:
            if cancellation.requested and pending:
                elapsed = time.monotonic() - started
                failing_campaign = cancellation.failing_campaign
                for cell in pending:
                    results.append(
                        _not_run_result(
                            cell,
                            target_seconds,
                            elapsed,
                            return_code=130,
                            outcome="not_started_after_campaign_failure",
                            cancelled_by_campaign=failing_campaign,
                        )
                    )
                    print(
                        f"[production-parity] campaign={cell.name} "
                        "correctness_status=NOT_STARTED "
                        f"cancelled_by_campaign={failing_campaign}",
                        flush=True,
                    )
                pending.clear()

            if not cancellation.requested:
                runnable = select_runnable_campaigns(pending, occupied)
                for cell in runnable:
                    # Failure may be published by a worker while this launch
                    # set is being admitted. Recheck before every submission.
                    if cancellation.requested:
                        break
                    resources = campaign_resources(cell)
                    pending.remove(cell)
                    occupied.update(resources)
                    future = executor.submit(
                        _run_campaign_and_publish_failure,
                        build_dir,
                        cell,
                        None,
                        exact_cell_timeout_seconds=exact_cell_timeout_seconds,
                        global_started_at=started,
                        target_seconds=target_seconds,
                        environment_overrides=environment_overrides,
                        artifact_results_root=artifact_results_root,
                        cancellation=cancellation,
                    )
                    running[future] = (cell, resources)
            if not running:
                continue

            done, _ = concurrent.futures.wait(
                running,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                cell, resources = running.pop(future)
                occupied.difference_update(resources)
                try:
                    result = future.result()
                except Exception as error:  # pragma: no cover - catastrophic worker boundary
                    print(
                        f"[production-parity] campaign={cell.name} worker_error={error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    result = CampaignResult(
                        campaign=cell.name,
                        test_type=cell.test_type,
                        backends=cell.group.backends,
                        precision_set=cell.group.kv_precision,
                        precision_types=cell.precision_types,
                        gtest_cases=cell.gtest_cases,
                        elapsed_seconds=0.0,
                        target_seconds=target_seconds,
                        return_code=125,
                        target_met=(time.monotonic() - started) <= target_seconds,
                        started_offset_seconds=time.monotonic() - started,
                        finished_offset_seconds=time.monotonic() - started,
                        outcome="worker_error",
                        artifact_contract_passed=False,
                        artifact_errors=(
                            "campaign worker failed before artifact validation",
                        ),
                    )
                results.append(result)
                if result.outcome == "cancelled_after_campaign_failure":
                    correctness_status = "CANCELLED"
                else:
                    correctness_status = (
                        "PASS" if result.return_code == 0 else "FAIL"
                    )
                target_status = "MET" if result.target_met else "MISSED"
                print(
                    f"[production-parity] campaign={cell.name} "
                    f"correctness_status={correctness_status} "
                    f"artifact_status="
                    f"{'PASS' if result.artifact_contract_passed else 'FAIL'} "
                    f"validated_artifacts="
                    f"{result.validated_artifact_file_count} "
                    f"whole_matrix_target={target_status} "
                    f"elapsed_seconds={result.elapsed_seconds:.3f} "
                    f"global_elapsed_seconds={result.finished_offset_seconds:.3f} "
                    f"cancelled_by_campaign="
                    f"{result.cancelled_by_campaign or '<none>'}",
                    flush=True,
                )

    results.sort(key=lambda result: original_order[result.campaign])
    return results, time.monotonic() - started


def write_report(
    path: Path,
    result: CampaignMatrixResult | IndividualProgressReport,
) -> None:
    """Atomically publish global matrix coverage and timing evidence."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(asdict(result), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _positive_seconds(raw: str) -> float:
    value = float(raw)
    if not math.isfinite(value) or not value > 0.0:
        raise argparse.ArgumentTypeError("duration must be finite and positive")
    return value


def _positive_integer(raw: str) -> int:
    """Parse one strictly positive diagnostic work-admission bound."""

    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "count must be a positive integer"
        ) from error
    if value <= 0:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return value


def _existing_directory(raw: str) -> Path:
    """Resolve one existing directory for explicit scheduling evidence."""

    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise argparse.ArgumentTypeError(
            f"artifact root does not exist: {raw}: {error}"
        ) from error
    if not path.is_dir():
        raise argparse.ArgumentTypeError(
            f"artifact root is not a directory: {path}"
        )
    return path


def _existing_file(raw: str) -> Path:
    """Resolve one existing regular file for explicit recovery evidence."""

    try:
        path = Path(raw).expanduser().resolve(strict=True)
    except OSError as error:
        raise argparse.ArgumentTypeError(
            f"evidence file does not exist: {raw}: {error}"
        ) from error
    if not path.is_file():
        raise argparse.ArgumentTypeError(
            f"evidence path is not a regular file: {path}"
        )
    return path


def _campaign_case_pair(raw: str) -> tuple[str, str]:
    """Parse one explicit ``CAMPAIGN=GTEST_CASE`` fail-fast declaration."""

    campaign, separator, gtest_case = raw.partition("=")
    if not separator or not campaign or not gtest_case:
        raise argparse.ArgumentTypeError(
            "first-red declaration must be CAMPAIGN=GTEST_CASE"
        )
    return campaign, gtest_case


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=Path("build_v2_integration"))
    parser.add_argument(
        "--backend",
        default=".*",
        help="full-match regex for backend signatures such as CPU, CUDA, or CUDA+ROCm",
    )
    parser.add_argument(
        "--precision",
        default=".*",
        help="full-match regex for the precision set (normally ALL)",
    )
    parser.add_argument(
        "--campaign",
        default=".*",
        help="full-match regex for registered CTest campaign names",
    )
    parser.add_argument(
        "--exclude-campaign",
        default=None,
        help="full-match regex for registered campaign names to omit",
    )
    parser.add_argument(
        "--target-seconds",
        type=_positive_seconds,
        default=GLOBAL_TARGET_SECONDS,
        help=(
            "soft wall-clock performance target for the complete selected "
            "matrix; crossing it never truncates correctness execution"
        ),
    )
    parser.add_argument(
        "--completion-timeout-seconds",
        type=_positive_seconds,
        default=COMPLETION_TIMEOUT_SECONDS,
        help=(
            "independent safety timeout for setup phases that do not publish "
            "exact-cell progress; inference uses the fixed per-cell watchdog"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("parity-results/production-campaigns.json"),
    )
    parser.add_argument(
        "--model-ramdisk-root",
        type=Path,
        default=MODEL_RAMDISK_ROOT,
        help=(
            "existing tmpfs/ramfs mount used for identity-bound GGUF staging"
        ),
    )
    parser.add_argument(
        "--persistent-model-cache-dir",
        type=Path,
        default=None,
        help=(
            "stable child directory of --model-ramdisk-root; identity-bound "
            "GGUFs are reused idempotently and never removed automatically, "
            "then owner-write-sealed between campaigns on the same tmpfs mount"
        ),
    )
    parser.add_argument(
        "--prioritize-unseen-from-artifact-root",
        dest="prior_artifact_roots",
        action="append",
        type=_existing_directory,
        default=[],
        help=(
            "repeatable prior campaign artifact root used only to schedule "
            "unseen aggregates before partial and previously complete ones; "
            "all selected exact cells still rerun and publish fresh evidence"
        ),
    )
    parser.add_argument(
        "--green-ledger",
        type=Path,
        default=None,
        help=(
            "durable non-certifying exact-cell progress ledger; required by "
            "--run-unseen-cells-individually"
        ),
    )
    parser.add_argument(
        "--seed-green-from-report",
        type=_existing_file,
        default=None,
        help=(
            "interrupted schema-12 campaign report whose aggregate greens "
            "seed the non-certifying ledger before individual execution"
        ),
    )
    parser.add_argument(
        "--declare-first-red",
        dest="declared_first_reds",
        action="append",
        type=_campaign_case_pair,
        default=[],
        help=(
            "repeatable CAMPAIGN=GTEST_CASE declaration from preserved GTest "
            "output; imports only the fail-fast green prefix before that red"
        ),
    )
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument(
        "--stage-models-only",
        action="store_true",
        help=(
            "prepare and seal the selected campaign corpus in the explicit "
            "persistent model cache without launching inference"
        ),
    )
    operation.add_argument(
        "--list",
        action="store_true",
        help="print coverage without running",
    )
    operation.add_argument(
        "--run-unseen-cells-individually",
        action="store_true",
        help=(
            "diagnostic mode: execute only exact cells absent from the green "
            "ledger, sequentially, stopping at the first red; this mode never "
            "certifies the complete campaign"
        ),
    )
    parser.add_argument(
        "--max-unseen-cells",
        type=_positive_integer,
        default=None,
        help=(
            "maximum unseen exact cells admitted by one individual-mode "
            "invocation; successful bounded work exits zero while remaining "
            "cells stay absent from the non-certifying ledger"
        ),
    )
    parser.add_argument(
        "--reuse-passed-preflight-report",
        type=_existing_file,
        default=None,
        help=(
            "individual-mode report from a passed preflight; reuse is accepted "
            "only when Ninja and CTest build-identity boundaries are unchanged"
        ),
    )
    arguments = parser.parse_args(argv)
    if (
        arguments.stage_models_only
        and arguments.persistent_model_cache_dir is None
    ):
        parser.error(
            "--stage-models-only requires --persistent-model-cache-dir"
        )
    if (
        arguments.run_unseen_cells_individually
        and arguments.green_ledger is None
    ):
        parser.error(
            "--run-unseen-cells-individually requires --green-ledger"
        )
    if (
        not arguments.run_unseen_cells_individually
        and arguments.green_ledger is not None
    ):
        parser.error(
            "--green-ledger is only valid with "
            "--run-unseen-cells-individually"
        )
    if (
        not arguments.run_unseen_cells_individually
        and arguments.max_unseen_cells is not None
    ):
        parser.error(
            "--max-unseen-cells requires "
            "--run-unseen-cells-individually"
        )
    if (
        not arguments.run_unseen_cells_individually
        and arguments.reuse_passed_preflight_report is not None
    ):
        parser.error(
            "--reuse-passed-preflight-report requires "
            "--run-unseen-cells-individually"
        )
    if (
        not arguments.run_unseen_cells_individually
        and (
            arguments.seed_green_from_report is not None
            or arguments.declared_first_reds
        )
    ):
        parser.error(
            "green-ledger recovery options require "
            "--run-unseen-cells-individually"
        )
    if (
        arguments.declared_first_reds
        and arguments.seed_green_from_report is None
    ):
        parser.error("--declare-first-red requires --seed-green-from-report")
    if len(dict(arguments.declared_first_reds)) != len(
        arguments.declared_first_reds
    ):
        parser.error("--declare-first-red campaign identities must be unique")
    arguments.prior_artifact_roots = tuple(
        dict.fromkeys(arguments.prior_artifact_roots)
    )
    arguments.declared_first_reds = tuple(arguments.declared_first_reds)
    return arguments


def stage_selected_models_only(
    args: argparse.Namespace,
    selected: tuple[CampaignCell, ...],
) -> int:
    """Prepare the selected real-weight corpus without starting inference.

    This is the campaign-owned setup lifecycle used for focused iteration. It
    intentionally retains model discovery, fixture preparation, tmpfs capacity
    proof, cache locking, atomic publication, and idle sealing from the full
    runner. The CLI requires a persistent cache because a run-scoped workspace
    would be deleted as this function returns.

    Args:
        args: Validated campaign-runner arguments.
        selected: Exact registered campaigns whose declared models are staged.

    Returns:
        Zero after every model is published and sealed, otherwise the fixture
        or staging failure code.
    """

    started = time.monotonic()
    completion_deadline = started + args.completion_timeout_seconds
    fixture_return_code, fixture_elapsed = prepare_model_fixture(
        args.build_dir,
        max(completion_deadline - time.monotonic(), 0.001),
    )
    if fixture_return_code != 0:
        print(
            "[production-parity] stage_models_only_status=FAIL "
            f"fixture_return_code={fixture_return_code} "
            f"fixture_elapsed_seconds={fixture_elapsed:.3f}",
            file=sys.stderr,
            flush=True,
        )
        return fixture_return_code

    try:
        with model_staging_workspace(
            args.model_ramdisk_root,
            args.persistent_model_cache_dir,
            completion_deadline,
        ) as workspace:
            staged_models, filesystem_type = stage_models_in_ramdisk(
                selected,
                workspace.models,
                completion_deadline,
                persistent=workspace.persistent,
            )
            workspace.protect_published_models()
            print(
                "[production-parity] stage_models_only_status=PASS "
                f"staged_model_count={len(staged_models)} "
                f"staged_model_bytes="
                f"{sum(model.size_bytes for model in staged_models)} "
                f"filesystem={filesystem_type} "
                f"cache_root={workspace.root} "
                f"elapsed_seconds={time.monotonic() - started:.3f}",
                flush=True,
            )
        return 0
    except TimeoutError as error:
        print(
            "[production-parity] stage_models_only_status=FAIL "
            f"error={error}",
            file=sys.stderr,
            flush=True,
        )
        return 124
    except (OSError, ModelStagingError) as error:
        print(
            "[production-parity] stage_models_only_status=FAIL "
            f"error={error}",
            file=sys.stderr,
            flush=True,
        )
        return 2


def run_unseen_cells_individually(
    args: argparse.Namespace,
    selected: tuple[CampaignCell, ...],
) -> int:
    """Run unseen exact cells sequentially and persist only authenticated greens.

    The ledger is a resumable diagnostic queue, never a substitute for the
    final unfiltered matrix. A red remains unseen and is therefore the first
    cell retried after its focused fix. Previously green identities are never
    relaunched by this mode while their registered execution contract is
    unchanged.
    """

    if args.green_ledger is None:
        raise ValueError("individual mode requires an explicit green ledger")
    ledger_path = args.green_ledger.expanduser().resolve()
    if args.seed_green_from_report is not None:
        seed_individual_green_ledger_from_campaign_report(
            args.seed_green_from_report,
            selected,
            ledger_path,
            declared_first_reds=args.declared_first_reds,
        )
    ledger = load_individual_green_ledger(ledger_path)
    validate_green_ledger_selection(selected, ledger)
    all_pending = unseen_individual_cells(selected, ledger)
    pending = (
        all_pending
        if args.max_unseen_cells is None
        else all_pending[: args.max_unseen_cells]
    )
    selected_cases = frozenset(
        gtest_case
        for cell in selected
        for gtest_case in cell.gtest_cases
    )
    green_before = len(selected_cases & ledger.green_cases)

    if not pending:
        report = IndividualProgressReport(
            schema_version=INDIVIDUAL_PROGRESS_REPORT_SCHEMA_VERSION,
            mode="sequential_unseen_exact_cells",
            certification_eligible=False,
            selected_exact_cell_count=len(selected_cases),
            green_before_count=green_before,
            attempted_count=0,
            newly_green_count=0,
            green_after_count=green_before,
            remaining_unseen_count=0,
            stopped_on_failure=False,
            preflight_return_code=125,
            preflight_test_count=0,
            preflight_tests=(),
            fixture_return_code=0,
            model_staging_return_code=0,
            artifact_root="",
            green_ledger=str(ledger_path),
            results=(),
        )
        write_report(args.report, report)
        print(
            "[production-parity] individual_sweep_status=COMPLETE "
            f"green={green_before}/{len(selected_cases)} "
            "certification_eligible=false next=run_unfiltered_campaign",
            flush=True,
        )
        return 0

    artifact_root = create_campaign_artifact_root(args.report)
    started = time.monotonic()
    completion_deadline = started + args.completion_timeout_seconds
    results: list[CampaignResult] = []
    preflight_return_code = 125
    preflight_tests: tuple[str, ...] = ()
    fixture_return_code = 125
    model_staging_return_code = 125
    stopped_on_failure = False
    exit_code = 0

    print(
        "[production-parity] individual_sweep_status=RUNNING "
        f"green_before={green_before} unseen={len(all_pending)} "
        f"admitted_unseen={len(pending)} "
        f"selected={len(selected_cases)} certification_eligible=false",
        flush=True,
    )
    if args.reuse_passed_preflight_report is None:
        (
            preflight_return_code,
            _,
            preflight_tests,
        ) = run_production_parity_preflight(
            args.build_dir,
            max(completion_deadline - time.monotonic(), 0.001),
        )
    else:
        (
            preflight_return_code,
            _,
            preflight_tests,
        ) = reuse_unchanged_production_parity_preflight(
            args.build_dir,
            args.reuse_passed_preflight_report,
        )
    if preflight_return_code != 0:
        exit_code = preflight_return_code
        stopped_on_failure = True
    else:
        fixture_return_code, _ = prepare_model_fixture(
            args.build_dir,
            max(completion_deadline - time.monotonic(), 0.001),
        )
        if fixture_return_code != 0:
            exit_code = fixture_return_code
            stopped_on_failure = True

    if not stopped_on_failure:
        try:
            with model_staging_workspace(
                args.model_ramdisk_root,
                args.persistent_model_cache_dir,
                completion_deadline,
            ) as workspace:
                staged_directory = workspace.models
                stage_models_in_ramdisk(
                    selected,
                    staged_directory,
                    completion_deadline,
                    persistent=workspace.persistent,
                )
                workspace.protect_published_models()
                model_staging_return_code = 0
                environment_overrides = {
                    MODEL_RAMDISK_ENV: str(staged_directory),
                    ARTIFACT_ROOT_ENV: str(artifact_root),
                }
                ordinal = {
                    gtest_case: index
                    for index, gtest_case in enumerate(
                        (
                            gtest_case
                            for cell in selected
                            for gtest_case in cell.gtest_cases
                        ),
                        start=1,
                    )
                }
                for cell, gtest_case in pending:
                    print(
                        "[production-parity] individual_cell_progress="
                        f"{ordinal[gtest_case]}/{len(selected_cases)} "
                        f"gtest_case={gtest_case}",
                        flush=True,
                    )
                    result = run_individual_cell(
                        cell,
                        gtest_case,
                        None,
                        exact_cell_timeout_seconds=EXACT_CELL_TIMEOUT_SECONDS,
                        global_started_at=started,
                        target_seconds=args.target_seconds,
                        environment_overrides=environment_overrides,
                        artifact_results_root=artifact_root,
                    )
                    results.append(result)
                    if result.return_code != 0:
                        exit_code = result.return_code
                        stopped_on_failure = True
                        break
                    ledger = record_individual_green(
                        ledger_path,
                        cell,
                        result,
                    )
                    print(
                        "[production-parity] individual_cell_recorded_green "
                        f"gtest_case={gtest_case} "
                        f"green={len(selected_cases & ledger.green_cases)}/"
                        f"{len(selected_cases)}",
                        flush=True,
                    )
        except TimeoutError as error:
            model_staging_return_code = 124
            exit_code = 124
            stopped_on_failure = True
            print(
                f"[production-parity] individual_sweep_error={error}",
                file=sys.stderr,
                flush=True,
            )
        except (OSError, ModelStagingError) as error:
            model_staging_return_code = 2
            exit_code = 2
            stopped_on_failure = True
            print(
                f"[production-parity] individual_sweep_error={error}",
                file=sys.stderr,
                flush=True,
            )

    ledger = load_individual_green_ledger(ledger_path)
    validate_green_ledger_selection(selected, ledger)
    remaining = unseen_individual_cells(selected, ledger)
    green_after = len(selected_cases & ledger.green_cases)
    report = IndividualProgressReport(
        schema_version=INDIVIDUAL_PROGRESS_REPORT_SCHEMA_VERSION,
        mode="sequential_unseen_exact_cells",
        certification_eligible=False,
        selected_exact_cell_count=len(selected_cases),
        green_before_count=green_before,
        attempted_count=len(results),
        newly_green_count=green_after - green_before,
        green_after_count=green_after,
        remaining_unseen_count=len(remaining),
        stopped_on_failure=stopped_on_failure,
        preflight_return_code=preflight_return_code,
        preflight_test_count=len(preflight_tests),
        preflight_tests=preflight_tests,
        fixture_return_code=fixture_return_code,
        model_staging_return_code=model_staging_return_code,
        artifact_root=str(artifact_root),
        green_ledger=str(ledger_path),
        results=tuple(results),
    )
    write_report(args.report, report)
    if not remaining and not stopped_on_failure:
        print(
            "[production-parity] individual_sweep_status=COMPLETE "
            f"green={green_after}/{len(selected_cases)} "
            "certification_eligible=false next=run_unfiltered_campaign",
            flush=True,
        )
        return 0
    if not stopped_on_failure and args.max_unseen_cells is not None:
        print(
            "[production-parity] individual_sweep_status=PAUSED "
            f"green={green_after}/{len(selected_cases)} "
            f"remaining_unseen={len(remaining)} "
            f"admission_limit={args.max_unseen_cells} "
            "certification_eligible=false",
            flush=True,
        )
        return 0
    print(
        "[production-parity] individual_sweep_status="
        f"{'FAIL' if stopped_on_failure else 'INCOMPLETE'} "
        f"green={green_after}/{len(selected_cases)} "
        f"remaining_unseen={len(remaining)} certification_eligible=false",
        flush=True,
    )
    return exit_code if exit_code != 0 else 3


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        selected = discover_campaigns(
            args.build_dir,
            args.backend,
            args.precision,
            args.campaign,
            args.exclude_campaign,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"production parity campaign error: {error}", file=sys.stderr)
        return 2

    grouped = group_campaigns(selected)
    if not grouped:
        print("production parity campaign selection is empty", file=sys.stderr)
        return 2

    prior_evidence = inspect_prior_artifact_evidence(
        selected,
        args.prior_artifact_roots,
    )
    if prior_evidence.roots:
        prior_counts = prior_evidence_campaign_counts(selected, prior_evidence)
        print(
            "[production-parity] unseen_first_scheduling=true "
            f"prior_evidence_roots={len(prior_evidence.roots)} "
            "prior_observed_exact_cells="
            f"{len(prior_evidence.observed_gtest_cases)} "
            "unseen_campaigns="
            f"{prior_counts[PriorEvidenceCoverageKind.UNSEEN]} "
            "partial_campaigns="
            f"{prior_counts[PriorEvidenceCoverageKind.PARTIAL]} "
            "complete_campaigns="
            f"{prior_counts[PriorEvidenceCoverageKind.COMPLETE]}",
            flush=True,
        )

    if args.list:
        print(
            f"coverage: {len(selected)} campaigns, "
            f"{sum(len(cell.gtest_cases) for cell in selected)} exact matrix cells"
        )
        for group, cells in grouped.items():
            print(f"{group.name}: {len(cells)}")
            for cell in cells:
                precisions = ",".join(cell.precision_types)
                print(
                    f"  {cell.name} [{len(cell.gtest_cases)} matrix cells; "
                    f"kv_precisions={precisions}]"
                )
        return 0

    if args.stage_models_only:
        return stage_selected_models_only(args, tuple(selected))

    if args.run_unseen_cells_individually:
        try:
            return run_unseen_cells_individually(args, tuple(selected))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(
                f"production parity individual sweep error: {error}",
                file=sys.stderr,
            )
            return 2

    try:
        artifact_root = create_campaign_artifact_root(args.report)
    except OSError as error:
        print(
            f"production parity campaign error: cannot create artifact root: {error}",
            file=sys.stderr,
        )
        return 2

    global_started = time.monotonic()
    global_completion_deadline = (
        global_started + args.completion_timeout_seconds
    )
    print(
        f"[production-parity] global_campaigns={len(selected)} "
        f"exact_matrix_cells={sum(len(cell.gtest_cases) for cell in selected)} "
        f"global_target_seconds={args.target_seconds:.3f} "
        f"completion_timeout_seconds={args.completion_timeout_seconds:.3f} "
        f"exact_cell_timeout_seconds={EXACT_CELL_TIMEOUT_SECONDS:.3f}",
        flush=True,
    )
    try:
        (
            preflight_return_code,
            preflight_elapsed,
            preflight_tests,
        ) = run_production_parity_preflight(
            args.build_dir,
            max(global_completion_deadline - time.monotonic(), 0.001),
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(
            f"production parity preflight error: {error}",
            file=sys.stderr,
            flush=True,
        )
        preflight_return_code = 2
        preflight_elapsed = time.monotonic() - global_started
        preflight_tests = ()

    fixture_return_code = 125
    fixture_elapsed = 0.0
    if preflight_return_code == 0:
        fixture_return_code, fixture_elapsed = prepare_model_fixture(
            args.build_dir,
            max(global_completion_deadline - time.monotonic(), 0.001),
        )
    model_staging_return_code = 125
    model_staging_elapsed = 0.0
    model_staging_filesystem = ""
    model_staging_mode = (
        "persistent"
        if args.persistent_model_cache_dir is not None
        else "run_scoped"
    )
    model_staging_location = args.model_ramdisk_root
    model_staging_error = (
        "not attempted because the model fixture failed"
        if preflight_return_code == 0
        else "not attempted because production parity preflight failed"
    )
    staged_models: tuple[StagedModelEvidence, ...] = ()
    if fixture_return_code == 0:
        staging_started = time.monotonic()
        try:
            # Run-scoped workspaces disappear after the children exit. An
            # explicitly selected persistent workspace remains locked for that
            # same lifetime but retains identity-bound weights for the next
            # invocation.
            with model_staging_workspace(
                args.model_ramdisk_root,
                args.persistent_model_cache_dir,
                global_completion_deadline,
            ) as workspace:
                model_staging_mode = workspace.mode
                model_staging_location = workspace.root
                staged_directory = workspace.models
                staged_models, model_staging_filesystem = stage_models_in_ramdisk(
                    selected,
                    staged_directory,
                    global_completion_deadline,
                    persistent=workspace.persistent,
                )
                model_staging_return_code = 0
                model_staging_error = ""
                model_staging_elapsed = time.monotonic() - staging_started

                workspace.protect_published_models()
                results, global_elapsed = run_campaign_matrix(
                    args.build_dir,
                    selected,
                    args.target_seconds,
                    exact_cell_timeout_seconds=EXACT_CELL_TIMEOUT_SECONDS,
                    global_started_at=global_started,
                    environment_overrides={
                        MODEL_RAMDISK_ENV: str(staged_directory),
                        ARTIFACT_ROOT_ENV: str(artifact_root),
                    },
                    artifact_results_root=artifact_root,
                    prior_evidence=prior_evidence,
                )
        except TimeoutError as error:
            model_staging_return_code = 124
            model_staging_error = str(error)
            print(
                f"[production-parity] model_staging_error={model_staging_error}",
                file=sys.stderr,
                flush=True,
            )
            model_staging_elapsed = time.monotonic() - staging_started
            global_elapsed = time.monotonic() - global_started
            results = [
                _not_run_result(
                    cell,
                    args.target_seconds,
                    global_elapsed,
                    outcome="model_staging_completion_timeout",
                )
                for cell in selected
            ]
        except (OSError, ModelStagingError) as error:
            model_staging_return_code = 2
            model_staging_error = str(error)
            print(
                f"[production-parity] model_staging_error={model_staging_error}",
                file=sys.stderr,
                flush=True,
            )
            model_staging_elapsed = time.monotonic() - staging_started
            global_elapsed = time.monotonic() - global_started
            results = [
                _not_run_result(
                    cell,
                    args.target_seconds,
                    global_elapsed,
                    return_code=model_staging_return_code,
                    outcome="model_staging_failed",
                )
                for cell in selected
            ]
    else:
        global_elapsed = time.monotonic() - global_started
        prerequisite_return_code = (
            preflight_return_code
            if preflight_return_code != 0
            else fixture_return_code
        )
        prerequisite_outcome = (
            "production_parity_preflight_failed"
            if preflight_return_code != 0
            else "model_fixture_failed"
        )
        results = [
            _not_run_result(
                cell,
                args.target_seconds,
                global_elapsed,
                return_code=prerequisite_return_code,
                outcome=prerequisite_outcome,
            )
            for cell in selected
        ]

    all_campaigns_passed = all(result.return_code == 0 for result in results)
    artifact_contract_passed = all(
        result.artifact_contract_passed for result in results
    )
    correctness_passed = (
        preflight_return_code == 0
        and fixture_return_code == 0
        and model_staging_return_code == 0
        and all_campaigns_passed
        and artifact_contract_passed
    )
    global_target_met = global_elapsed <= args.target_seconds
    performance_requirements_met = global_target_met
    report = CampaignMatrixResult(
        schema_version=CAMPAIGN_REPORT_SCHEMA_VERSION,
        global_target_seconds=args.target_seconds,
        global_elapsed_seconds=global_elapsed,
        global_target_met=global_target_met,
        correctness_passed=correctness_passed,
        performance_requirements_met=performance_requirements_met,
        completion_timeout_seconds=args.completion_timeout_seconds,
        exact_cell_timeout_seconds=EXACT_CELL_TIMEOUT_SECONDS,
        preflight_return_code=preflight_return_code,
        preflight_elapsed_seconds=preflight_elapsed,
        preflight_test_count=len(preflight_tests),
        preflight_tests=preflight_tests,
        fixture_return_code=fixture_return_code,
        fixture_elapsed_seconds=fixture_elapsed,
        model_staging_return_code=model_staging_return_code,
        model_staging_elapsed_seconds=model_staging_elapsed,
        model_staging_root=str(model_staging_location),
        model_staging_filesystem=model_staging_filesystem,
        model_staging_mode=model_staging_mode,
        model_staging_error=model_staging_error,
        artifact_root=str(artifact_root),
        staged_model_count=len(staged_models),
        staged_model_bytes=sum(model.size_bytes for model in staged_models),
        staged_models=staged_models,
        prior_evidence_roots=tuple(
            str(root) for root in prior_evidence.roots
        ),
        prior_observed_exact_cell_count=len(
            prior_evidence.observed_gtest_cases
        ),
        unseen_first_scheduling=bool(prior_evidence.roots),
        scheduling_policy=(
            "model_free_integration_preflight_"
            "exclusive_backend_sets_maximal_disjoint_"
            + (
                "prior_artifact_unseen_first_"
                if prior_evidence.roots
                else ""
            )
            +
            "identity_bound_tmpfs_model_staging_"
            f"{model_staging_mode}_numerical_reference_proof"
        ),
        campaign_count=len(selected),
        exact_matrix_cell_count=sum(
            len(cell.gtest_cases) for cell in selected
        ),
        artifact_contract_passed=artifact_contract_passed,
        validated_artifact_file_count=sum(
            result.validated_artifact_file_count for result in results
        ),
        artifact_error_count=sum(
            len(result.artifact_errors) for result in results
        ),
        campaigns=tuple(results),
    )
    write_report(args.report, report)
    overall_passed = correctness_passed and performance_requirements_met
    print(
        f"[production-parity] correctness_status="
        f"{'PASS' if correctness_passed else 'FAIL'} "
        f"performance_status="
        f"{'PASS' if performance_requirements_met else 'FAIL'} "
        f"elapsed_seconds={global_elapsed:.3f} "
        f"target_seconds={args.target_seconds:.3f}",
        flush=True,
    )
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
