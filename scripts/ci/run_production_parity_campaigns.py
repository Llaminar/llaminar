#!/usr/bin/env python3
"""Run the complete real-weight production parity matrix and measure its SLA.

The C++ `ProductionParity` case is the atomic matrix cell: it constructs one
production runner, executes full stage-by-stage prefill parity, validates graph-
stable snapshot publication, resets request data, and executes incremental
decode parity. CTest combines all precision cells for one test type/backend in
one process so immutable model and prepared-weight ownership is amortized while
runner, arena, graph, stream, and KV state remain exact per cell.

The performance target belongs to the whole selected matrix. This driver prepares the model
fixture once, capacity-checks and atomically stages the complete selected GGUF
corpus into tmpfs, gives CPU, CUDA, and ROCm exclusive resource identities, and
overlaps only campaigns whose backend sets are disjoint. Every production child
then SHA-authenticates the staged bytes against its reference pack through one
shared identity-keyed digest cache before inference. Staging is private and
run-scoped by default. An explicit persistent-cache directory instead retains
atomically published, source-identity-bound read-only GGUFs and authenticated
digest evidence for rapid iteration; the driver never removes that cache.
Hybrid campaigns claim every backend they name. Crossing the target is recorded
as a performance failure after the complete matrix has run; it never stops
campaign admission or truncates correctness evidence. A separate, deliberately
generous completion timeout exists only to terminate a genuinely stuck run. No
campaign gets an independent hour and no concurrent launch may oversubscribe a
backend.

CTest/GTest registration remains the matrix source of truth.  The driver never
copies model, topology, backend, or precision tables into another manifest.
Use ``--list`` for a deterministic coverage inventory without running models.
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
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator


BACKEND_ORDER = ("CPU", "CUDA", "ROCm")
KV_PRECISIONS = ("FP16", "FP32", "Q8_1", "Q16_1", "TQ")
PRODUCTION_CAMPAIGN_NAME = re.compile(r"(?:^|_)ProductionCampaign(?:_|$)")
GLOBAL_TARGET_SECONDS = 3600.0
COMPLETION_TIMEOUT_SECONDS = 21600.0
REGISTERED_TIMEOUT_SECONDS = COMPLETION_TIMEOUT_SECONDS
MODEL_FIXTURE_NAME = "V2_Models"
MODEL_FIXTURE_TEST = "V2_FetchModelsFixture"
MODEL_RAMDISK_ROOT = Path("/dev/shm")
PERSISTENT_MODEL_CACHE_SCHEMA_VERSION = 1
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
        "state_detail,observed_terminal_hidden_hash_available,"
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
        "backend,device,execution_path,homogeneous_gpu,"
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
    "after_verifier_runs"
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
    "WholeMatrixOneHourTarget",
}
REQUIRED_CAMPAIGN_ENV = {
    "LLAMINAR_PRODUCTION_PARITY=1",
    "LLAMINAR_PRODUCTION_PARITY_PROCESS_CAMPAIGN=1",
    "LLAMINAR_PRODUCTION_PARITY_TARGET_SECONDS=3600",
    "LLAMINAR_GPU_GRAPHS=1",
    "LLAMINAR_PERF_STATS_SUMMARY=1",
}


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
    outcome: str = "completed"
    artifact_contract_passed: bool = True
    validated_artifact_file_count: int = 0
    artifact_directories: tuple[str, ...] = ()
    artifact_errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class StagedModelEvidence:
    """Authenticated evidence for one GGUF available in the selected tmpfs."""

    source_path: str
    filename: str
    size_bytes: int
    sha256: str
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
    scheduling_policy: str
    campaign_count: int
    exact_matrix_cell_count: int
    authenticated_model_digest_count: int
    artifact_contract_passed: bool
    validated_artifact_file_count: int
    artifact_error_count: int
    campaigns: tuple[CampaignResult, ...]


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
) -> tuple[tuple[str, ...], tuple[str, ...]]:
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
    return _exact_gtest_cases(test), model_files


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
            exact_cases, model_files = _validate_campaign_registration(test)
        except ValueError as error:
            raise RuntimeError(f"invalid production campaign {name}: {error}") from error
        cells.append(
            CampaignCell(
                name=name,
                group=group,
                gtest_cases=exact_cases,
                model_files=model_files,
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
    """Paths and lifetime mode for one authenticated model-cache workspace."""

    root: Path
    models: Path
    digests: Path
    mode: str
    persistent: bool


def _reclaim_interrupted_persistent_transactions(root: Path) -> tuple[int, int]:
    """Remove unpublished cache transactions while holding the cache lock.

    Model copies and manifest updates are published with atomic renames.  A
    SIGKILL, host reboot, or editor-container reload can therefore leave only
    their hidden transaction files behind.  Once the persistent campaign lock
    has been acquired, no healthy producer can still own one of those files,
    so retaining it would merely consume tmpfs capacity and make setup cease to
    be idempotent.

    Only the two cache-owned transaction namespaces are eligible.  Published
    GGUFs, digest records, the lock, and operator-owned files are never touched.
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
                digests=root / "digests",
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
    try:
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            digests=root / "digests",
            mode="persistent",
            persistent=True,
        )
    finally:
        try:
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


def _hash_file(path: Path, completion_deadline: float | None) -> str:
    """Hash a file while honoring only the stuck-run safety deadline."""

    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while True:
            if (
                completion_deadline is not None
                and time.monotonic() >= completion_deadline
            ):
                raise TimeoutError(
                    "completion timeout expired during GGUF authentication"
                )
            chunk = stream.read(MODEL_COPY_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _persistent_manifest_path(staging_directory: Path) -> Path:
    """Return the cache-owned manifest beside the persistent model directory."""

    return staging_directory.parent / "model-cache-manifest.json"


def _load_persistent_manifest(staging_directory: Path) -> dict[str, Any]:
    """Load the persistent cache authority, failing closed on malformed state."""

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
    if (
        not isinstance(document, dict)
        or document.get("schema_version")
        != PERSISTENT_MODEL_CACHE_SCHEMA_VERSION
        or not isinstance(document.get("models"), dict)
    ):
        raise ModelStagingError(
            f"persistent model cache manifest has an unsupported shape: {path}"
        )
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
        "sha256": evidence.sha256,
    }


def _reuse_persistent_model(
    source: Path,
    destination: Path,
    raw_entry: Any,
    completion_deadline: float | None,
) -> tuple[StagedModelEvidence, bool] | None:
    """Authenticate and reuse one immutable tmpfs entry when its source is unchanged.

    The source's device/inode/size/mtime/ctime tuple makes an ordinary source
    replacement or mutation invalidate the entry without rereading slow model
    storage. The first successful cache publication records the SHA-256 computed
    while streaming the source plus the read-only tmpfs file identity. Every
    production child independently authenticates that exact staged identity
    against its reference pack before inference. Unchanged source and cached
    identities can therefore be reused without an extra staging-layer reread.
    Legacy entries lacking the cached identity receive one final SHA-256
    validation and request an atomic manifest upgrade.

    @return Evidence plus whether the caller must republish the upgraded
            manifest, or ``None`` when the entry must be recopied.
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
        or not isinstance(raw_entry.get("sha256"), str)
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
    expected_cached_identity = raw_entry.get("cached_identity")
    manifest_upgrade_required = False
    if expected_cached_identity == list(cached_identity):
        cached_digest = raw_entry["sha256"]
    else:
        # Version-1 manifests predate destination identity binding. Perform
        # exactly one full RAM-resident verification before installing it.
        cached_digest = _hash_file(destination, completion_deadline)
        manifest_upgrade_required = True

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
    if cached_digest != raw_entry["sha256"]:
        return None
    if manifest_upgrade_required:
        raw_entry["cached_identity"] = list(_source_identity(cached_after))
    return (
        StagedModelEvidence(
            source_path=str(source),
            filename=destination.name,
            size_bytes=cached_before.st_size,
            sha256=cached_digest,
            elapsed_seconds=time.monotonic() - started,
            cache_status="reused",
            source_identity=_source_identity(source_before),
        ),
        manifest_upgrade_required,
    )


def _stage_one_model(
    source: Path,
    destination: Path,
    completion_deadline: float | None,
) -> StagedModelEvidence:
    """Stream one source into an atomic, identity-bound tmpfs transaction.

    SHA-256 is accumulated while the cold source bytes are already in flight and
    retained as evidence. A second SHA pass over the just-written RAM copy would
    duplicate the mandatory reference-pack authentication performed by every
    production child before inference. Full-write and byte-count checks, source
    mutation checks, fsync, read-only publication, and destination identity
    binding make the cache transaction safe until that canonical check runs.
    """

    started = time.monotonic()
    source_before = source.stat()
    temporary = destination.with_name(
        f".{destination.name}.copying-{os.getpid()}"
    )
    source_digest = hashlib.sha256()
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
                    source_digest.update(chunk)
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
        sha256=source_digest.hexdigest(),
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
    across invocations. A persistent hit requires an unchanged source identity
    and unchanged read-only cached-file identity bound to the source-stream
    digest; misses are copied and published atomically without exposing partial
    weights. The production reference gate authenticates the staged destination
    bytes before inference and coalesces that result in the digest cache.
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
    manifest_upgrade_required = False
    if manifest is not None:
        for source in sources:
            reusable_result = _reuse_persistent_model(
                source,
                staging_directory / source.name,
                manifest["models"].get(source.name),
                completion_deadline,
            )
            if reusable_result is not None:
                record, entry_upgraded = reusable_result
                reusable[source] = record
                manifest_upgrade_required = (
                    manifest_upgrade_required or entry_upgraded
                )
        if manifest_upgrade_required:
            _publish_persistent_manifest(staging_directory, manifest)

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
                f"validation_throughput_mib_s={throughput_mib_s:.1f} "
                f"sha256={cached_record.sha256}",
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
            f"throughput_mib_s={throughput_mib_s:.1f} sha256={record.sha256}",
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


def _required_csv_headers_for_case(gtest_case: str) -> dict[str, str]:
    """Return the exact artifact contract for one typed matrix cell."""

    headers = dict(REQUIRED_CSV_HEADERS)
    if re.search(
        r"(?:^|_)MTP(?:Depth(?:1|2|3|15)|DynamicDepth)(?:_|$)",
        gtest_case,
    ):
        headers["mtp_transactions.csv"] = MTP_TRANSACTIONS_HEADER
    return headers


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


def campaign_resources(cell: CampaignCell) -> frozenset[str]:
    """Return the exclusive backend resources claimed by one campaign."""

    resources = frozenset(cell.group.backends.split("+"))
    if not resources or not resources.issubset(BACKEND_ORDER):
        raise ValueError(
            f"campaign {cell.name} has invalid backend resources: "
            f"{sorted(resources)}"
        )
    return resources


def scheduling_order(cells: Iterable[CampaignCell]) -> list[CampaignCell]:
    """Put wide hybrid claims first while retaining deterministic ordering."""

    return sorted(
        cells,
        key=lambda cell: (
            -len(campaign_resources(cell)),
            cell.group.backends,
            cell.name,
        ),
    )


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


def _run_process(
    command: list[str],
    timeout_seconds: float | None,
    environment_overrides: dict[str, str] | None = None,
) -> int:
    """Run one process group, terminating it only at the safety timeout."""

    environment = None
    if environment_overrides:
        environment = os.environ.copy()
        environment.update(environment_overrides)
    process = subprocess.Popen(
        command,
        start_new_session=True,
        env=environment,
    )
    try:
        if timeout_seconds is None:
            return process.wait()
        return process.wait(timeout=max(timeout_seconds, 0.001))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        return 124


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


def run_campaign(
    build_dir: Path,
    cell: CampaignCell,
    completion_timeout_seconds: float | None,
    *,
    global_started_at: float | None = None,
    target_seconds: float = GLOBAL_TARGET_SECONDS,
    environment_overrides: dict[str, str] | None = None,
    artifact_results_root: Path | None = None,
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
        f"completion_timeout_remaining_seconds={timeout_text}",
        flush=True,
    )
    started = time.monotonic()
    started_wall_time_ns = time.time_ns()
    process_return_code = _run_process(
        command,
        completion_timeout_seconds,
        environment_overrides=environment_overrides,
    )
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
    outcome = "completion_timeout" if process_return_code == 124 else "completed"
    if process_return_code == 0 and artifact_errors:
        return_code = 126
        outcome = "artifact_contract_failed"
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
        outcome=outcome,
        artifact_contract_passed=not artifact_errors,
        validated_artifact_file_count=artifact_count,
        artifact_directories=artifact_directories,
        artifact_errors=artifact_errors,
    )


def _not_run_result(
    cell: CampaignCell,
    target_seconds: float,
    elapsed_seconds: float,
    *,
    return_code: int = 124,
    outcome: str = "not_started_before_completion_timeout",
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
        outcome=outcome,
        artifact_contract_passed=False,
        artifact_errors=("campaign did not execute, so no fresh artifacts exist",),
    )


def run_campaign_matrix(
    build_dir: Path,
    cells: Iterable[CampaignCell],
    target_seconds: float,
    completion_timeout_seconds: float | None,
    *,
    global_started_at: float | None = None,
    global_completion_deadline: float | None = None,
    environment_overrides: dict[str, str] | None = None,
    artifact_results_root: Path | None = None,
) -> tuple[list[CampaignResult], float]:
    """Run every cell, measuring one SLA and enforcing only a safety timeout."""

    selected = list(cells)
    started = (
        global_started_at if global_started_at is not None else time.monotonic()
    )
    completion_deadline = (
        global_completion_deadline
        if global_completion_deadline is not None
        else (
            started + completion_timeout_seconds
            if completion_timeout_seconds is not None
            else None
        )
    )
    pending = scheduling_order(selected)
    original_order = {cell.name: index for index, cell in enumerate(selected)}
    results: list[CampaignResult] = []
    running: dict[
        concurrent.futures.Future[CampaignResult],
        tuple[CampaignCell, frozenset[str]],
    ] = {}
    occupied: set[str] = set()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(BACKEND_ORDER),
        thread_name_prefix="production-parity",
    ) as executor:
        while pending or running:
            now = time.monotonic()
            completion_time_remains = (
                completion_deadline is None or now < completion_deadline
            )
            if completion_time_remains:
                runnable = select_runnable_campaigns(pending, occupied)
                for cell in runnable:
                    resources = campaign_resources(cell)
                    pending.remove(cell)
                    occupied.update(resources)
                    timeout_seconds = (
                        max(completion_deadline - time.monotonic(), 0.001)
                        if completion_deadline is not None
                        else None
                    )
                    future = executor.submit(
                        run_campaign,
                        build_dir,
                        cell,
                        timeout_seconds,
                        global_started_at=started,
                        target_seconds=target_seconds,
                        environment_overrides=environment_overrides,
                        artifact_results_root=artifact_results_root,
                    )
                    running[future] = (cell, resources)
            elif pending:
                elapsed = time.monotonic() - started
                results.extend(
                    _not_run_result(cell, target_seconds, elapsed)
                    for cell in pending
                )
                pending.clear()

            if not running:
                continue

            wait_seconds = (
                max(completion_deadline - time.monotonic(), 0.0)
                if completion_deadline is not None
                else None
            )
            done, _ = concurrent.futures.wait(
                running,
                timeout=(
                    wait_seconds
                    if wait_seconds is None or wait_seconds > 0.0
                    else 0.1
                ),
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
                correctness_status = "PASS" if result.return_code == 0 else "FAIL"
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
                    f"global_elapsed_seconds={result.finished_offset_seconds:.3f}",
                    flush=True,
                )

    results.sort(key=lambda result: original_order[result.campaign])
    return results, time.monotonic() - started


def write_report(path: Path, result: CampaignMatrixResult) -> None:
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
            "independent stuck-run safety timeout for fixture staging and the "
            "complete matrix"
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
            "existing tmpfs/ramfs mount used for authenticated GGUF staging"
        ),
    )
    parser.add_argument(
        "--persistent-model-cache-dir",
        type=Path,
        default=None,
        help=(
            "stable child directory of --model-ramdisk-root; authenticated "
            "GGUFs and digest evidence are reused idempotently and never "
            "removed automatically"
        ),
    )
    parser.add_argument("--list", action="store_true", help="print coverage without running")
    return parser.parse_args(argv)


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
                    f"precisions={precisions}]"
                )
        return 0

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
        f"completion_timeout_seconds={args.completion_timeout_seconds:.3f}",
        flush=True,
    )
    fixture_return_code, fixture_elapsed = prepare_model_fixture(
        args.build_dir,
        max(global_completion_deadline - time.monotonic(), 0.001),
    )
    authenticated_model_digest_count = 0
    model_staging_return_code = 125
    model_staging_elapsed = 0.0
    model_staging_filesystem = ""
    model_staging_mode = (
        "persistent"
        if args.persistent_model_cache_dir is not None
        else "run_scoped"
    )
    model_staging_location = args.model_ramdisk_root
    model_staging_error = "not attempted because the model fixture failed"
    staged_models: tuple[StagedModelEvidence, ...] = ()
    if fixture_return_code == 0:
        staging_started = time.monotonic()
        try:
            # Run-scoped workspaces disappear after the children exit. An
            # explicitly selected persistent workspace remains locked for that
            # same lifetime but retains authenticated weights and digest records
            # for the next invocation.
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

                # Every production cell checks the reference pack's recorded
                # model SHA-256. This private cache coalesces the canonical
                # staged-destination validation across child processes. The
                # staging loop deliberately does not perform the same expensive
                # RAM-resident SHA pass a second time.
                digest_cache = workspace.digests
                digest_cache.mkdir(
                    mode=0o700,
                    parents=True,
                    exist_ok=workspace.persistent,
                )
                results, global_elapsed = run_campaign_matrix(
                    args.build_dir,
                    selected,
                    args.target_seconds,
                    args.completion_timeout_seconds,
                    global_started_at=global_started,
                    global_completion_deadline=global_completion_deadline,
                    environment_overrides={
                        "LLAMINAR_PRODUCTION_PARITY_DIGEST_CACHE": str(
                            digest_cache
                        ),
                        MODEL_RAMDISK_ENV: str(staged_directory),
                        ARTIFACT_ROOT_ENV: str(artifact_root),
                    },
                    artifact_results_root=artifact_root,
                )
                authenticated_model_digest_count = sum(
                    1 for _ in digest_cache.glob("*.sha256")
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
        results = [
            _not_run_result(
                cell,
                args.target_seconds,
                global_elapsed,
                return_code=fixture_return_code,
                outcome="model_fixture_failed",
            )
            for cell in selected
        ]

    all_campaigns_passed = all(result.return_code == 0 for result in results)
    artifact_contract_passed = all(
        result.artifact_contract_passed for result in results
    )
    correctness_passed = (
        fixture_return_code == 0
        and model_staging_return_code == 0
        and all_campaigns_passed
        and artifact_contract_passed
    )
    global_target_met = global_elapsed <= args.target_seconds
    performance_requirements_met = global_target_met
    report = CampaignMatrixResult(
        schema_version=8,
        global_target_seconds=args.target_seconds,
        global_elapsed_seconds=global_elapsed,
        global_target_met=global_target_met,
        correctness_passed=correctness_passed,
        performance_requirements_met=performance_requirements_met,
        completion_timeout_seconds=args.completion_timeout_seconds,
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
        scheduling_policy=(
            "exclusive_backend_sets_maximal_disjoint_"
            "identity_bound_tmpfs_model_staging_"
            f"{model_staging_mode}_reference_authenticated_model_digests"
        ),
        campaign_count=len(selected),
        exact_matrix_cell_count=sum(
            len(cell.gtest_cases) for cell in selected
        ),
        authenticated_model_digest_count=authenticated_model_digest_count,
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
