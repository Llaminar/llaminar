#!/usr/bin/env python3
"""Run a provenance-locked, two-turn AIME 2025 evaluation over HTTP.

This tool exercises the deployed Llaminar Release server rather than calling
model internals.  It owns server startup when ``--binary`` and ``--model`` are
provided, verifies an explicit multi-turn history canary, and then evaluates
all 30 problems from the pinned ``math-ai/aime25`` dataset.  Each problem uses
two OpenAI-compatible requests: an initial solution and a review turn that
receives the complete prior message history.  Only an explicit ``Answer: NNN``
or ``\\boxed{NNN}`` is scored, preventing incidental numbers in a derivation
from becoming false positives.

Results are fsynced one problem at a time.  A manifest binds the corpus to the
exact dataset, model, executable, prompts, sampling policy, and server launch
arguments, so an interrupted run can resume without mixing incompatible
evidence.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


# GPU host-transfer policy is shared with the canonical server E2E gate. Keep
# this runner from inventing a second allowlist as production control-plane
# boundaries evolve.
REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_E2E_POLICY_DIR = REPO_ROOT / "tests/v2/e2e/server"
if str(SERVER_E2E_POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_E2E_POLICY_DIR))
from gpu_host_transfer_perf_policy import (  # noqa: E402
    validate_gpu_host_transfer_policy,
)


DATASET_REVISION = "563bb8404243c5f09de6ec262f2db674fe5bce9b"
DATASET_SHA256 = "b4e273c02d3e7fe1b74b59eae768fc8230bfb0f79539890cb56f4361caac0331"
DATASET_URL = (
    "https://huggingface.co/datasets/math-ai/aime25/resolve/"
    f"{DATASET_REVISION}/test.jsonl?download=true"
)
MANIFEST_SCHEMA = "llaminar-aime25-http-v5"
RESULT_SCHEMA = "llaminar-aime25-http-result-v2"
FIRST_TURN_SCHEMA = "llaminar-aime25-http-first-turn-v1"
CANARY_SCHEMA = "llaminar-aime25-http-history-canary-v1"
PREFIX_CACHE_CANARY_SCHEMA = "llaminar-aime25-prefix-cache-canary-v1"
EXECUTION_CONTRACT_SCHEMA = "llaminar-aime25-execution-contract-v2"
PROMPT_SCHEMA = "aime25-two-turn-review-v2"
CHAT_PATH = "/v1/chat/completions"
RUNTIME_ENVIRONMENT_PREFIXES = (
    "LLAMINAR_",
    "CUDA_",
    "HIP_",
    "ROCR_",
    "NCCL_",
    "RCCL_",
    "OMP_",
    "MKL_",
    "OPENBLAS_",
)
SUPPORTED_DEVICE_BACKENDS = ("cpu", "cuda", "rocm")
MAX_SUPPORTED_MTP_DEPTH = 15
PREFIX_CACHE_CANARY_MAX_TOKENS = 16
MANAGED_MTP_OPTIONS = frozenset({
    "--mtp",
    "--mtp-draft-tokens",
    "--mtp-verify-mode",
    "--mtp-depth-policy",
    "--mtp-min-draft-tokens",
    "--mtp-initial-draft-tokens",
    "--mtp-max-draft-tokens",
})

SYSTEM_PROMPT = (
    "You are solving an American Invitational Mathematics Examination problem. "
    "Reason carefully, check all constraints, and do not guess. The official "
    "answer is an integer from 000 through 999. End your response with exactly "
    "one line in the form `Answer: NNN`, using three decimal digits."
)
REVIEW_PROMPT = (
    "Review the proposed solution independently from the original problem. "
    "Recompute any fragile steps and correct the result if necessary. End with "
    "exactly one line in the form `Answer: NNN`, using three decimal digits."
)
PREFIX_CACHE_CANARY_MESSAGES = (
    {
        "role": "system",
        "content": "Follow the user's formatting instruction exactly.",
    },
    {
        "role": "user",
        "content": (
            "Prefix-cache restore canary 88421. Reply only with `cache-ready`."
        ),
    },
)


@dataclasses.dataclass(frozen=True)
class AimeProblem:
    """One canonical AIME problem and integer answer."""

    problem_id: str
    problem: str
    answer: int


@dataclasses.dataclass(frozen=True)
class ChatReply:
    """Validated fields retained from one chat-completion response."""

    content: str
    reasoning_content: str
    finish_reason: str
    usage: Mapping[str, int]
    elapsed_seconds: float


def sha256_file(path: Path) -> str:
    """Return a prefixed SHA-256 digest without loading the file into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def mapping_digest(payload: Mapping[str, object]) -> str:
    """Digest a JSON mapping through one canonical byte representation."""

    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def parse_device_backend(device: str) -> str:
    """Return a supported backend from one explicit Llaminar device spec."""

    backend_pattern = "|".join(SUPPORTED_DEVICE_BACKENDS)
    match = re.fullmatch(
        rf"({backend_pattern}):([0-9]+)",
        device.strip().lower(),
    )
    if match is None:
        raise ValueError(
            "device must be one of cpu:N, cuda:N, or rocm:N with a "
            "non-negative ordinal"
        )
    return match.group(1)


def managed_server_arguments(
    mtp_depth: int | None,
    prefix_cache: bool,
) -> tuple[str, ...]:
    """Build the benchmark-owned production feature policy."""

    arguments: list[str] = []
    if mtp_depth is not None:
        arguments.extend((
            "--mtp",
            "--mtp-draft-tokens", str(mtp_depth),
            "--mtp-depth-policy", "fixed",
            "--mtp-verify-mode", "speculative-sampling",
        ))
    if prefix_cache:
        arguments.extend((
            "--prefix-cache",
            "--prefix-cache-storage", "ram",
            "--prefix-cache-ram-budget-mb", "1024",
            "--prefix-cache-terminal-state", "auto",
            "--prefix-cache-moe-policy", "placement-fingerprint",
        ))
    return tuple(arguments)


def reject_managed_server_argument_conflicts(
    extra_args: Sequence[str],
    mtp_depth: int | None,
    prefix_cache: bool,
) -> None:
    """Reject raw arguments that can contradict a typed benchmark policy."""

    for token in extra_args:
        option = token.split("=", maxsplit=1)[0]
        if mtp_depth is not None and option in MANAGED_MTP_OPTIONS:
            raise ValueError(
                f"{option} is owned by --mtp-depth; remove it from "
                "--server-extra-args"
            )
        if prefix_cache and option.startswith("--prefix-cache"):
            raise ValueError(
                f"{option} is owned by --prefix-cache; remove it from "
                "--server-extra-args"
            )


def resolve_linked_library(binary: Path, soname: str) -> Path:
    """Resolve one mandatory shared object from the executable's live closure."""

    completed = subprocess.run(
        ("ldd", str(Path(binary).resolve())),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"ldd failed for {binary} with exit code {completed.returncode}"
        )
    match = re.search(
        rf"^\s*{re.escape(soname)}\s+=>\s+(\S+)\s+\(",
        completed.stdout,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"{binary} does not resolve mandatory {soname}")
    resolved = Path(match.group(1)).resolve()
    if not resolved.is_file():
        raise ValueError(f"{binary} resolves missing {soname}: {resolved}")
    return resolved


def require_release_build(binary: Path) -> str:
    """Prove that the executable belongs to a configured Release build tree."""

    resolved = Path(binary).resolve()
    for directory in (resolved.parent, *resolved.parents):
        cache = directory / "CMakeCache.txt"
        if not cache.is_file():
            continue
        with cache.open(encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("CMAKE_BUILD_TYPE:") and "=" in line:
                    build_type = line.rstrip("\n").split("=", maxsplit=1)[1]
                    if build_type != "Release":
                        raise ValueError(
                            "AIME25 production evidence requires a Release "
                            f"server; observed {build_type!r} in {cache}"
                        )
                    return build_type
        raise ValueError(f"{cache}: CMAKE_BUILD_TYPE is absent")
    raise ValueError(f"{binary}: no governing CMakeCache.txt was found")


def inherited_runtime_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Select every inherited knob capable of changing benchmark execution."""

    return {
        name: value
        for name, value in sorted(environment.items())
        if name.startswith(RUNTIME_ENVIRONMENT_PREFIXES)
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Fsync and atomically replace one JSON document."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".inprogress")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(staging, path)


def append_jsonl_durable(path: Path, payload: Mapping[str, object]) -> None:
    """Append one completed problem and make it durable before returning."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_execution_contract(
    perfstats_path: Path,
    requested_device: str,
    *,
    required_mtp_depth: int | None = None,
    require_prefix_cache: bool = False,
) -> dict[str, object]:
    """Prove requested features and GPU graph ownership from PerfStats."""

    device_prefix = parse_device_backend(requested_device)
    display_backend = {
        "cpu": "CPU",
        "cuda": "CUDA",
        "rocm": "ROCm",
    }[device_prefix]
    expected_device = display_backend + requested_device[len(device_prefix):]

    try:
        payload = json.loads(Path(perfstats_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"unable to read completed PerfStats evidence: {perfstats_path}"
        ) from error
    if payload.get("schema") != "llaminar.perf_stats.v1":
        raise ValueError("AIME25 PerfStats has an unsupported schema")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("AIME25 PerfStats contains no records")

    def value(record: Mapping[str, object]) -> float:
        try:
            return float(record.get("value", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def active(record: Mapping[str, object]) -> bool:
        try:
            return value(record) > 0.0 or float(record.get("total_ns", 0.0)) > 0.0
        except (TypeError, ValueError):
            return value(record) > 0.0

    def tags(record: Mapping[str, object]) -> Mapping[str, object]:
        candidate = record.get("tags", {})
        return candidate if isinstance(candidate, dict) else {}

    def tag_int(record: Mapping[str, object], name: str) -> int:
        try:
            return int(str(tags(record).get(name, "0")))
        except ValueError:
            return 0

    process_records = [
        record for record in records
        if isinstance(record, dict)
    ]
    _, separator, ordinal = requested_device.partition(":")
    target_device_aliases = {
        requested_device.lower(),
        expected_device.lower(),
        device_prefix,
    }
    if device_prefix in {"cuda", "rocm"}:
        target_device_aliases.add("gpu")
    if device_prefix in {"cuda", "rocm"} and separator and ordinal:
        target_device_aliases.add(f"gpu:{ordinal}")
    if not any(
        str(record.get("device", "")).strip().lower()
        in target_device_aliases
        for record in process_records
    ):
        raise ValueError(
            f"AIME25 PerfStats contains no evidence for {expected_device}"
        )

    # PerfStats producers historically used backend-qualified, canonical,
    # generic-GPU, and empty process-wide device labels. This benchmark owns a
    # single-device server process, so its execution certificate must inspect
    # every record rather than let a forbidden operation escape validation by
    # choosing a different spelling for the same accelerator.
    relevant = process_records

    mtp_requests = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "mtp" and
        record.get("name") == "device_resident_generation_requests"
    ]
    mtp_verifier_runs = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "mtp" and
        record.get("name") == "verifier_runs" and
        tags(record).get("path") == "device_resident_generation_loop"
    ]
    mtp_depth_selections = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "mtp" and
        record.get("name") == "decode_transaction_depth_selections"
    ]
    completed_mtp_verifier_runs = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "mtp" and
        record.get("name") == "verifier_runs"
    ]
    if required_mtp_depth is not None:
        expected_depth = str(required_mtp_depth)
        if not mtp_depth_selections or any(
            str(tags(record).get("depth_policy", "")) != "fixed" or
            str(tags(record).get("requested_depth", "")) != expected_depth or
            str(tags(record).get("capture_depth", "")) != expected_depth
            for record in mtp_depth_selections
        ):
            raise ValueError(
                "AIME25 MTP did not select the requested fixed depth "
                f"{required_mtp_depth}"
            )
        if sum(value(record) for record in completed_mtp_verifier_runs) <= 0.0:
            raise ValueError(
                "AIME25 PerfStats contains no completed MTP verifier runs"
            )
        if device_prefix in {"cuda", "rocm"} and not mtp_requests:
            raise ValueError(
                "AIME25 PerfStats does not prove device-resident MTP execution"
            )
        invalid_depth_records = [
            record for record in mtp_requests
            if any(
                str(tags(record).get(field, "")) != expected_depth
                for field in ("depth", "capture_depth", "final_depth")
            ) or tag_int(record, "transactions") <= 0 or
            tag_int(record, "state_commits") <= 0
        ]
        if invalid_depth_records:
            raise ValueError(
                "AIME25 device-resident MTP did not remain at requested "
                f"fixed depth {required_mtp_depth}"
            )
        if (
            device_prefix in {"cuda", "rocm"} and
            sum(value(record) for record in mtp_verifier_runs) <= 0.0
        ):
            raise ValueError(
                "AIME25 PerfStats contains no device-resident MTP verifier runs"
            )

    prefix_hits = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "prefix_cache" and
        record.get("name") == "lookup_results" and
        str(tags(record).get("hit_type", "")).lower() in {"partial", "full"} and
        tag_int(record, "cached_tokens") > 0
    ]
    prefix_restores = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "prefix_cache" and
        record.get("name") == "populate_restores"
    ]
    prefix_harvests = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "prefix_cache" and
        record.get("name") == "harvest_inserts"
    ]
    if require_prefix_cache:
        if not prefix_harvests:
            raise ValueError(
                "AIME25 PerfStats does not prove prefix-cache population"
            )
        if not prefix_hits or not prefix_restores:
            raise ValueError(
                "AIME25 PerfStats does not prove prefix-cache hit and restore"
            )
        if required_mtp_depth is not None and not any(
            str(tags(record).get("includes_mtp_state", "false")).lower() == "true"
            for record in prefix_restores
        ):
            raise ValueError(
                "AIME25 prefix-cache restore did not include MTP state"
            )

    report: dict[str, object] = {
        "schema_version": EXECUTION_CONTRACT_SCHEMA,
        "device": expected_device,
        "perfstats_sha256": sha256_file(perfstats_path),
        "required_mtp_depth": required_mtp_depth,
        "mtp_depth_selections": sum(
            value(record) for record in mtp_depth_selections
        ),
        "completed_mtp_verifier_runs": sum(
            value(record) for record in completed_mtp_verifier_runs
        ),
        "device_resident_mtp_requests": sum(
            value(record) for record in mtp_requests
        ),
        "device_resident_mtp_verifier_runs": sum(
            value(record) for record in mtp_verifier_runs
        ),
        "prefix_cache_required": require_prefix_cache,
        "prefix_cache_hit_lookups": sum(value(record) for record in prefix_hits),
        "prefix_cache_restores": sum(value(record) for record in prefix_restores),
        "prefix_cache_harvests": sum(value(record) for record in prefix_harvests),
    }
    if device_prefix == "cpu":
        return report

    host_transfer_validation = validate_gpu_host_transfer_policy(relevant)
    if host_transfer_validation.error:
        raise ValueError("AIME25 " + host_transfer_validation.error)
    report.update({
        "final_response_host_operations": list(
            host_transfer_validation.final_response_operations
        ),
        "rocm_scheduler_dispatch_operations": list(
            host_transfer_validation.scheduler_dispatch_operations
        ),
        "prefix_cache_tier_transfer_operations": list(
            host_transfer_validation.prefix_cache_operations
        ),
    })

    segmented = [
        record for record in relevant
        if active(record) and (
            str(tags(record).get("collective_segmented", "false")).lower() == "true" or
            (
                record.get("domain") == "forward_graph" and
                "segmented" in str(record.get("name", "")).lower()
            )
        )
    ]
    manual_graphs = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "forward_graph" and
        str(record.get("name", "")).startswith("full_graph_plan_") and
        tags(record).get("type") == "manual"
    ]
    rejected_prefill = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "forward_graph" and
        str(tags(record).get("capture_phase", "")).lower() == "rejected"
    ]
    invalid_decode_policy = [
        record for record in relevant
        if active(record) and
        record.get("domain") == "forward_graph" and
        record.get("name") == "decode_capture_policy" and
        (
            str(tags(record).get("allow_graph_replay", "false")).lower() != "true" or
            str(tags(record).get("collective_segmented", "false")).lower() != "false" or
            tags(record).get("replay_plan_policy") != "require_full_graph"
        )
    ]
    if segmented:
        raise ValueError("AIME25 GPU execution used segmented graph replay")
    if manual_graphs:
        raise ValueError("AIME25 GPU execution admitted manual graph stages")
    if rejected_prefill:
        raise ValueError("AIME25 GPU prefill graph capture was rejected")
    if invalid_decode_policy:
        raise ValueError(
            "AIME25 decode admitted a segmented/non-monolithic graph policy"
        )

    def phase_value(name: str, phase: str) -> float:
        return sum(
            value(record) for record in relevant
            if record.get("domain") == "forward_graph" and
            record.get("name") == name and
            str(tags(record).get("phase", "")).lower() == phase
        )

    decode_capture = phase_value("decode_graph_phase", "capture")
    decode_replay = phase_value("decode_graph_phase", "replay")
    prefill_capture = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "prefill_graph_phase" and
        str(tags(record).get("capture_phase", "")).lower() == "capture"
    )
    prefill_replay = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "prefill_graph_phase" and
        str(tags(record).get("capture_phase", "")).lower() == "replay"
    )
    decode_publications = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "durable_output_publications" and
        record.get("phase") == "decode"
    )
    prefill_publications = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "durable_output_publications" and
        record.get("phase") == "prefill"
    )
    decode_graph_owned = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "decode_graph_phase"
    )
    prefill_graph_owned = sum(
        value(record) for record in relevant
        if record.get("domain") == "forward_graph" and
        record.get("name") == "prefill_graph_phase"
    )
    if decode_capture <= 0.0 or decode_replay <= 0.0:
        raise ValueError(
            "AIME25 PerfStats does not prove captured decode plus replay"
        )
    if prefill_capture <= 0.0 or prefill_replay <= 0.0:
        raise ValueError(
            "AIME25 PerfStats does not prove captured prefill plus replay"
        )
    if decode_publications <= 0.0 or decode_graph_owned < decode_publications:
        raise ValueError(
            "AIME25 PerfStats exposes decode work outside graph ownership"
        )
    if prefill_publications <= 0.0 or prefill_graph_owned < prefill_publications:
        raise ValueError(
            "AIME25 PerfStats exposes prefill work outside graph ownership"
        )

    report.update({
        "decode_capture": decode_capture,
        "decode_replay": decode_replay,
        "prefill_capture": prefill_capture,
        "prefill_replay": prefill_replay,
        "decode_publications": decode_publications,
        "decode_graph_owned": decode_graph_owned,
        "prefill_publications": prefill_publications,
        "prefill_graph_owned": prefill_graph_owned,
        "segmented_graph_records": 0,
        "manual_graphs": 0,
        "forbidden_host_transfer_operations": [],
    })
    return report


def download_pinned_dataset(path: Path) -> Path:
    """Download and authenticate the immutable 30-row AIME dataset."""

    path = Path(path)
    if path.is_file() and sha256_file(path) == f"sha256:{DATASET_SHA256}":
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".inprogress")
    staging.unlink(missing_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "llaminar-aime25-benchmark/1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response, staging.open(
        "wb"
    ) as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
        handle.flush()
        os.fsync(handle.fileno())
    observed = sha256_file(staging)
    expected = f"sha256:{DATASET_SHA256}"
    if observed != expected:
        staging.unlink(missing_ok=True)
        raise ValueError(
            f"AIME25 dataset digest mismatch: expected {expected}, got {observed}"
        )
    os.replace(staging, path)
    return path


def load_problems(path: Path) -> tuple[AimeProblem, ...]:
    """Parse and validate the exact 30-problem benchmark inventory."""

    if sha256_file(path) != f"sha256:{DATASET_SHA256}":
        raise ValueError("AIME25 input is not the pinned dataset revision")
    problems = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(row, dict) or set(row) != {
                "problem", "answer", "id"
            }:
                raise ValueError(f"{path}:{line_number}: unexpected schema")
            problem_id = str(row["id"])
            problem = str(row["problem"]).strip()
            answer = int(row["answer"])
            if not problem or not problem_id or answer not in range(1000):
                raise ValueError(f"{path}:{line_number}: invalid problem row")
            problems.append(AimeProblem(problem_id, problem, answer))
    if len(problems) != 30 or len({item.problem_id for item in problems}) != 30:
        raise ValueError("pinned AIME25 dataset must contain 30 unique problems")
    return tuple(sorted(problems, key=lambda item: int(item.problem_id)))


_ANSWER_PATTERNS = (
    re.compile(r"(?i)\banswer\s*:\s*(?:\\boxed\s*\{\s*)?(\d{1,3})"),
    re.compile(r"\\boxed\s*\{\s*(\d{1,3})\s*\}"),
)


def extract_explicit_answer(text: str) -> int | None:
    """Return the final explicitly marked AIME integer, never an incidental one."""

    matches: list[tuple[int, int]] = []
    for pattern in _ANSWER_PATTERNS:
        for match in pattern.finditer(text):
            value = int(match.group(1))
            if value in range(1000):
                matches.append((match.start(), value))
    return max(matches)[1] if matches else None


def post_chat(
    base_url: str,
    messages: Sequence[Mapping[str, str]],
    *,
    max_tokens: int,
    enable_thinking: bool,
    temperature: float,
    seed: int,
    timeout_seconds: float,
) -> ChatReply:
    """Post one non-streaming completion and validate its response contract."""

    payload = {
        "model": "llaminar",
        "messages": list(messages),
        "stream": False,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "temperature": temperature,
        "seed": seed,
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + CHAT_PATH,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:1000]}") from error
    elapsed = time.monotonic() - started
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:1000]}")
    try:
        data = json.loads(body)
        choice = data["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        finish_reason = str(choice["finish_reason"])
        usage_raw = data["usage"]
        usage = {
            name: int(usage_raw[name])
            for name in (
                "prompt_tokens", "completion_tokens", "total_tokens"
            )
        }
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"invalid chat completion response: {body[:1000]}"
        ) from error
    if not content and not reasoning:
        raise RuntimeError("chat completion returned no content or reasoning")
    return ChatReply(content, reasoning, finish_reason, usage, elapsed)


def wait_for_health(
    base_url: str,
    process: subprocess.Popen[bytes] | None,
    timeout_seconds: float,
) -> None:
    """Wait for explicit health publication and detect early server death."""

    deadline = time.monotonic() + timeout_seconds
    health_url = base_url.rstrip("/") + "/health"
    last_error = "server has not published health"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Llaminar server exited early with code {process.returncode}"
            )
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                payload = json.loads(response.read().decode("utf-8"))
                if response.status == 200 and payload.get("status") == "ok":
                    return
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(
        f"server did not become healthy within {timeout_seconds:.1f}s: {last_error}"
    )


def require_free_port(host: str, port: int) -> None:
    """Reject accidental attachment to an unrelated process."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((host, port)) == 0:
            raise ValueError(f"HTTP port {host}:{port} is already in use")


def stop_server(process: subprocess.Popen[bytes], grace_seconds: float = 15.0) -> None:
    """Terminate the complete MPI/bootstrap process group without hanging."""

    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def run_history_canary(
    base_url: str,
    request_timeout: float,
    seed: int,
) -> dict[str, object]:
    """Prove that a second HTTP request consumes explicit prior-turn history."""

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "Follow the user's formatting instruction exactly.",
        },
        {
            "role": "user",
            "content": (
                "Remember benchmark nonce 731. Reply only with `remembered`."
            ),
        },
    ]
    first = post_chat(
        base_url,
        messages,
        max_tokens=32,
        enable_thinking=False,
        temperature=0.0,
        seed=seed,
        timeout_seconds=request_timeout,
    )
    messages.extend((
        {"role": "assistant", "content": first.content},
        {
            "role": "user",
            "content": "What was the nonce? Reply exactly `Answer: NNN`.",
        },
    ))
    second = post_chat(
        base_url,
        messages,
        max_tokens=32,
        enable_thinking=False,
        temperature=0.0,
        seed=seed,
        timeout_seconds=request_timeout,
    )
    observed = extract_explicit_answer(second.content)
    if observed != 731:
        raise RuntimeError(
            "multi-turn history canary failed: expected 731, "
            f"content={second.content!r}"
        )
    return {
        "first_content": first.content,
        "second_content": second.content,
        "observed_answer": observed,
        "first_usage": dict(first.usage),
        "second_usage": dict(second.usage),
    }


def deterministic_reply_signature(reply: ChatReply) -> Mapping[str, object]:
    """Return response fields that must survive an exact prefix restore."""

    return {
        "content": reply.content,
        "reasoning_content": reply.reasoning_content,
        "finish_reason": reply.finish_reason,
        "usage": dict(reply.usage),
    }


def run_prefix_cache_restore_canary(
    base_url: str,
    request_timeout: float,
    seed: int,
) -> dict[str, object]:
    """Populate and replay one byte-identical prompt through the live cache.

    A rendered multi-turn chat is not generally a token prefix of the request
    that generated its assistant message: the chat template replaces the old
    generation boundary with a serialized assistant turn. This canary keeps
    cache correctness independent of that semantic-history check by issuing
    the exact same HTTP payload twice. The completed PerfStats ledger then
    proves that the replay imported a prefix payload, including shifted MTP KV
    state when MTP is enabled.
    """

    messages = tuple(dict(message) for message in PREFIX_CACHE_CANARY_MESSAGES)
    replies = []
    for _ in range(2):
        replies.append(post_chat(
            base_url,
            messages,
            max_tokens=PREFIX_CACHE_CANARY_MAX_TOKENS,
            enable_thinking=False,
            temperature=0.0,
            seed=seed,
            timeout_seconds=request_timeout,
        ))

    first_signature = deterministic_reply_signature(replies[0])
    replay_signature = deterministic_reply_signature(replies[1])
    if first_signature != replay_signature:
        raise RuntimeError(
            "prefix-cache exact replay changed deterministic model output: "
            f"first={first_signature!r} replay={replay_signature!r}"
        )
    return {
        "request_sha256": mapping_digest({
            "messages": list(messages),
            "max_tokens": PREFIX_CACHE_CANARY_MAX_TOKENS,
            "enable_thinking": False,
            "temperature": 0.0,
            "seed": seed,
        }),
        "first": dataclasses.asdict(replies[0]),
        "replay": dataclasses.asdict(replies[1]),
    }


def validate_prefix_cache_canary(
    path: Path,
    manifest_digest: str,
) -> dict[str, object]:
    """Authenticate persisted evidence from one exact-prompt replay pair."""

    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prefix-cache canary {path}") from error
    if row.get("schema_version") != PREFIX_CACHE_CANARY_SCHEMA:
        raise ValueError("unsupported prefix-cache canary schema")
    if row.get("manifest_digest") != manifest_digest:
        raise ValueError("prefix-cache canary belongs to another manifest")
    first = row.get("first")
    replay = row.get("replay")
    if not isinstance(first, dict) or not isinstance(replay, dict):
        raise ValueError("prefix-cache canary is missing replay evidence")
    comparable_fields = (
        "content",
        "reasoning_content",
        "finish_reason",
        "usage",
    )
    if any(first.get(field) != replay.get(field) for field in comparable_fields):
        raise ValueError("prefix-cache canary replay is not deterministic")
    return row


def validate_history_canary(
    path: Path,
    manifest_digest: str,
) -> dict[str, object]:
    """Authenticate persisted evidence that explicit HTTP history worked."""

    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid multi-turn canary {path}") from error
    if row.get("schema_version") != CANARY_SCHEMA:
        raise ValueError("unsupported multi-turn canary schema")
    if row.get("manifest_digest") != manifest_digest:
        raise ValueError("multi-turn canary belongs to another manifest")
    if row.get("observed_answer") != 731:
        raise ValueError("multi-turn canary does not prove the expected nonce")
    return row


def completed_results(
    path: Path,
    manifest_digest: str,
) -> dict[str, dict[str, object]]:
    """Read and authenticate the manifest-bound append-only resume surface."""

    if not Path(path).exists():
        return {}
    result: dict[str, dict[str, object]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if row.get("schema_version") != RESULT_SCHEMA:
                raise ValueError(f"{path}:{line_number}: unsupported result schema")
            if row.get("manifest_digest") != manifest_digest:
                raise ValueError(
                    f"{path}:{line_number}: result belongs to another manifest"
                )
            problem_id = str(row["problem_id"])
            if problem_id in result:
                raise ValueError(f"{path}:{line_number}: duplicate problem result")
            result[problem_id] = row
    return result


def load_first_turn_checkpoint(
    path: Path,
    manifest_digest: str,
    problem_id: str,
) -> ChatReply | None:
    """Load one authenticated first turn for exact review-only resumption."""

    if not Path(path).exists():
        return None
    try:
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if row.get("schema_version") != FIRST_TURN_SCHEMA:
            raise ValueError("unsupported first-turn checkpoint schema")
        if row.get("manifest_digest") != manifest_digest:
            raise ValueError("first-turn checkpoint belongs to another manifest")
        if str(row.get("problem_id")) != problem_id:
            raise ValueError("first-turn checkpoint has the wrong problem ID")
        reply = row["reply"]
        return ChatReply(
            content=str(reply["content"]),
            reasoning_content=str(reply["reasoning_content"]),
            finish_reason=str(reply["finish_reason"]),
            usage={name: int(value) for name, value in reply["usage"].items()},
            elapsed_seconds=float(reply["elapsed_seconds"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid first-turn checkpoint {path}: {error}") from error


def write_first_turn_checkpoint(
    path: Path,
    manifest_digest: str,
    problem_id: str,
    reply: ChatReply,
) -> None:
    """Atomically publish a completed first turn before review begins."""

    write_json_atomic(path, {
        "schema_version": FIRST_TURN_SCHEMA,
        "manifest_digest": manifest_digest,
        "problem_id": problem_id,
        "reply": dataclasses.asdict(reply),
    })


def solve_problem(
    base_url: str,
    problem: AimeProblem,
    *,
    manifest_digest: str,
    first_reply: ChatReply | None = None,
    publish_first_reply: Callable[[ChatReply], None] | None = None,
    first_max_tokens: int,
    review_max_tokens: int,
    enable_thinking: bool,
    temperature: float,
    seed: int,
    request_timeout: float,
) -> dict[str, object]:
    """Run one initial and one full-history review request, then score both."""

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.problem},
    ]
    first = first_reply
    if first is None:
        first = post_chat(
            base_url,
            messages,
            max_tokens=first_max_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
            seed=seed,
            timeout_seconds=request_timeout,
        )
        if publish_first_reply is not None:
            publish_first_reply(first)
    first_answer = extract_explicit_answer(first.content)
    assistant_history = first.content
    if first.reasoning_content:
        assistant_history = (
            "Reasoning:\n" + first.reasoning_content.rstrip()
            + "\n\nFinal response:\n" + first.content.lstrip()
        )
    messages.extend((
        {"role": "assistant", "content": assistant_history},
        {"role": "user", "content": REVIEW_PROMPT},
    ))
    review = post_chat(
        base_url,
        messages,
        max_tokens=review_max_tokens,
        enable_thinking=enable_thinking,
        temperature=temperature,
        seed=seed,
        timeout_seconds=request_timeout,
    )
    review_answer = extract_explicit_answer(review.content)
    return {
        "schema_version": RESULT_SCHEMA,
        "manifest_digest": manifest_digest,
        "problem_id": problem.problem_id,
        "expected_answer": problem.answer,
        "first_answer": first_answer,
        "review_answer": review_answer,
        "first_correct": first_answer == problem.answer,
        "review_correct": review_answer == problem.answer,
        "first": dataclasses.asdict(first),
        "review": dataclasses.asdict(review),
    }


def selected_problem_ids(raw: str | None, problems: Sequence[AimeProblem]) -> tuple[str, ...]:
    """Parse a comma/range selection while defaulting to all 30 problems."""

    available = {item.problem_id for item in problems}
    if not raw:
        return tuple(item.problem_id for item in problems)
    selected: list[str] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = (int(part) for part in token.split("-", maxsplit=1))
            selected.extend(str(value) for value in range(first, last + 1))
        else:
            selected.append(str(int(token)))
    if not selected or len(selected) != len(set(selected)) or not set(selected) <= available:
        raise ValueError("problem selection must contain unique dataset IDs")
    return tuple(sorted(selected, key=int))


def make_summary(
    rows: Iterable[Mapping[str, object]],
    selected_ids: Sequence[str],
) -> dict[str, object]:
    """Aggregate first-turn and reviewed exact accuracy plus token/time evidence."""

    ordered = sorted(rows, key=lambda row: int(str(row["problem_id"])))
    first_correct = sum(bool(row["first_correct"]) for row in ordered)
    review_correct = sum(bool(row["review_correct"]) for row in ordered)
    first_seconds = sum(float(row["first"]["elapsed_seconds"]) for row in ordered)  # type: ignore[index]
    review_seconds = sum(float(row["review"]["elapsed_seconds"]) for row in ordered)  # type: ignore[index]
    completion_tokens = sum(
        int(row[turn]["usage"]["completion_tokens"])  # type: ignore[index]
        for row in ordered
        for turn in ("first", "review")
    )
    return {
        "schema_version": "llaminar-aime25-http-summary-v1",
        "selected_problem_ids": list(selected_ids),
        "completed": len(ordered),
        "first_correct": first_correct,
        "review_correct": review_correct,
        "first_accuracy": first_correct / len(ordered) if ordered else 0.0,
        "review_accuracy": review_correct / len(ordered) if ordered else 0.0,
        "first_seconds": first_seconds,
        "review_seconds": review_seconds,
        "completion_tokens": completion_tokens,
        "completion_tokens_per_second": (
            completion_tokens / (first_seconds + review_seconds)
            if first_seconds + review_seconds > 0.0 else 0.0
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the server-owning and client-only benchmark modes."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--device",
        required=True,
        help="Explicit execution target: cpu:N, cuda:N, or rocm:N.",
    )
    parser.add_argument("--base-url")
    parser.add_argument(
        "--server-identity",
        help="Required provenance label when attaching to an external server.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20980)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument(
        "--server-extra-args",
        default="",
        help="Shell-split extra serve arguments recorded in the manifest.",
    )
    parser.add_argument(
        "--mtp-depth",
        type=int,
        help=(
            "Enable fixed-depth, speculative-sampling MTP and certify the "
            "observed device-resident depth (supported range: 1-15)."
        ),
    )
    parser.add_argument(
        "--prefix-cache",
        action="store_true",
        help=(
            "Enable and certify RAM-backed prefix reuse with terminal state "
            "and MoE placement fingerprints."
        ),
    )
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-ids")
    parser.add_argument("--first-max-tokens", type=int, default=4096)
    parser.add_argument("--review-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=900.0)
    args = parser.parse_args(argv)
    if bool(args.base_url) == bool(args.binary or args.model):
        parser.error("provide either --base-url or both --binary and --model")
    if not args.base_url and (args.binary is None or args.model is None):
        parser.error("server-owning mode requires --binary and --model")
    if args.base_url and not args.server_identity:
        parser.error("client-only mode requires --server-identity")
    if not args.base_url and args.server_identity:
        parser.error("--server-identity is only valid with --base-url")
    try:
        args.backend = parse_device_backend(args.device)
    except ValueError as error:
        parser.error(str(error))
    if any(value <= 0 for value in (
        args.context_length,
        args.first_max_tokens,
        args.review_max_tokens,
        args.startup_timeout,
        args.request_timeout,
    )):
        parser.error("lengths, token budgets, and timeouts must be positive")
    if args.temperature < 0.0:
        parser.error("temperature must be non-negative")
    if args.mtp_depth is not None and not (
        1 <= args.mtp_depth <= MAX_SUPPORTED_MTP_DEPTH
    ):
        parser.error(
            f"--mtp-depth must be between 1 and {MAX_SUPPORTED_MTP_DEPTH}"
        )
    if args.base_url and (args.mtp_depth is not None or args.prefix_cache):
        parser.error(
            "--mtp-depth and --prefix-cache require server-owning mode so "
            "their execution can be certified"
        )
    try:
        args.extra_args = tuple(shlex.split(args.server_extra_args))
        reject_managed_server_argument_conflicts(
            args.extra_args,
            args.mtp_depth,
            args.prefix_cache,
        )
    except ValueError as error:
        parser.error(str(error))
    args.managed_server_args = managed_server_arguments(
        args.mtp_depth,
        args.prefix_cache,
    )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Own the live server, run/resume AIME25, and publish a summary."""

    args = parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = args.dataset or (
        Path.home() / ".cache/llaminar/benchmarks/aime25" /
        f"{DATASET_REVISION}.jsonl"
    )
    if args.dataset is None:
        download_pinned_dataset(dataset)
    problems = load_problems(dataset)
    problem_ids = selected_problem_ids(args.problem_ids, problems)
    selected = tuple(item for item in problems if item.problem_id in problem_ids)

    build_type = require_release_build(args.binary) if args.binary else None
    binary_digest = sha256_file(args.binary) if args.binary else None
    core_library = (
        resolve_linked_library(args.binary, "libllaminar2_core.so")
        if args.binary else None
    )
    model_digest = sha256_file(args.model) if args.model else None
    extra_args = args.extra_args
    managed_args = args.managed_server_args
    base_url = args.base_url or f"http://{args.host}:{args.port}"
    backend = parse_device_backend(args.device)
    execution_contract_required = bool(
        args.binary and (
            backend in {"cuda", "rocm"} or
            args.mtp_depth is not None or
            args.prefix_cache
        )
    )
    inherited_environment = inherited_runtime_environment(os.environ)
    # The owned server always replaces this output-only setting below. Keeping
    # a stale inherited path in the manifest would describe an environment the
    # server never executes.
    inherited_environment.pop("LLAMINAR_PERF_STATS_JSON", None)
    manifest_body: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": {
            "repository": "math-ai/aime25",
            "revision": DATASET_REVISION,
            "sha256": f"sha256:{DATASET_SHA256}",
        },
        "prompt_schema": PROMPT_SCHEMA,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "system_prompt": SYSTEM_PROMPT,
        "review_prompt": REVIEW_PROMPT,
        "problem_ids": list(problem_ids),
        "binary": str(args.binary.resolve()) if args.binary else None,
        "cmake_build_type": build_type,
        "binary_sha256": binary_digest,
        "core_library": str(core_library) if core_library else None,
        "core_library_sha256": (
            sha256_file(core_library) if core_library else None
        ),
        "model": str(args.model.resolve()) if args.model else None,
        "model_sha256": model_digest,
        "backend": backend,
        "device": args.device,
        "base_url": base_url,
        "server_identity": args.server_identity,
        "context_length": args.context_length,
        "server_managed_args": list(managed_args),
        "server_extra_args": list(extra_args),
        "mtp_policy": {
            "enabled": args.mtp_depth is not None,
            "depth": args.mtp_depth,
            "depth_policy": "fixed" if args.mtp_depth is not None else None,
            "verify_mode": (
                "speculative-sampling"
                if args.mtp_depth is not None else None
            ),
        },
        "prefix_cache_policy": {
            "enabled": args.prefix_cache,
            "storage": "ram" if args.prefix_cache else None,
            "ram_budget_mb": 1024 if args.prefix_cache else None,
            "terminal_state": "auto" if args.prefix_cache else None,
            "moe_policy": (
                "placement-fingerprint" if args.prefix_cache else None
            ),
            "restore_canary_schema": (
                PREFIX_CACHE_CANARY_SCHEMA if args.prefix_cache else None
            ),
            "restore_canary_max_tokens": (
                PREFIX_CACHE_CANARY_MAX_TOKENS if args.prefix_cache else None
            ),
        },
        "first_max_tokens": args.first_max_tokens,
        "review_max_tokens": args.review_max_tokens,
        "enable_thinking": args.enable_thinking,
        "temperature": args.temperature,
        "seed": args.seed,
        "inherited_runtime_environment": inherited_environment,
        "perfstats_enabled": bool(args.binary),
        "execution_contract_required": execution_contract_required,
    }
    manifest_body["manifest_digest"] = mapping_digest(manifest_body)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest_body:
            raise ValueError(
                "existing AIME25 result directory belongs to another run"
            )
    else:
        write_json_atomic(manifest_path, manifest_body)

    results_path = output_dir / "results.jsonl"
    manifest_digest = str(manifest_body["manifest_digest"])
    existing_results = completed_results(results_path, manifest_digest)
    if not set(existing_results) <= set(problem_ids):
        raise ValueError("result file contains a problem outside the manifest")

    process: subprocess.Popen[bytes] | None = None
    log_handle = None
    try:
        if args.binary:
            require_free_port(args.host, args.port)
            log_path = output_dir / "server.log"
            log_handle = log_path.open("ab", buffering=0)
            perfstats_path = output_dir / "perfstats.json"
            environment = os.environ.copy()
            environment["LLAMINAR_PERF_STATS_JSON"] = str(perfstats_path)
            command = [
                str(args.binary.resolve()),
                "serve",
                "--host", args.host,
                "--port", str(args.port),
                "--context-length", str(args.context_length),
                "--device", args.device,
                "--model", str(args.model.resolve()),
                *managed_args,
                *extra_args,
            ]
            write_json_atomic(output_dir / "server-command.json", {
                "argv": command,
                "perfstats": str(perfstats_path),
            })
            process = subprocess.Popen(
                command,
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        wait_for_health(base_url, process, args.startup_timeout)
        canary_path = output_dir / "multi-turn-canary.json"
        if canary_path.exists():
            validate_history_canary(canary_path, manifest_digest)
        else:
            canary = {
                "schema_version": CANARY_SCHEMA,
                "manifest_digest": manifest_digest,
                **run_history_canary(
                    base_url, args.request_timeout, args.seed
                ),
            }
            write_json_atomic(canary_path, canary)
            print("PASS multi-turn history canary", flush=True)

        prefix_cache_canary_path: Path | None = None
        if args.prefix_cache:
            # Run this pair for every owned-server process, including resumed
            # AIME runs. A persisted result from an earlier process cannot
            # certify that the current process restored its live cache.
            prefix_cache_canary_path = output_dir / "prefix-cache-canary.json"
            prefix_cache_canary = {
                "schema_version": PREFIX_CACHE_CANARY_SCHEMA,
                "manifest_digest": manifest_digest,
                **run_prefix_cache_restore_canary(
                    base_url, args.request_timeout, args.seed
                ),
            }
            write_json_atomic(prefix_cache_canary_path, prefix_cache_canary)
            validate_prefix_cache_canary(
                prefix_cache_canary_path, manifest_digest
            )
            print("PASS exact-prompt prefix-cache replay canary", flush=True)

        by_id = {item.problem_id: item for item in selected}
        for index, problem_id in enumerate(problem_ids, start=1):
            if problem_id in existing_results:
                print(
                    f"SKIP [{index}/{len(problem_ids)}] AIME25 id={problem_id}",
                    flush=True,
                )
                continue
            print(
                f"RUN  [{index}/{len(problem_ids)}] AIME25 id={problem_id}",
                flush=True,
            )
            first_turn_path = (
                output_dir / "turns" / f"{problem_id}.first.json"
            )
            first_reply = load_first_turn_checkpoint(
                first_turn_path,
                manifest_digest,
                problem_id,
            )

            def publish_first_reply(reply: ChatReply) -> None:
                write_first_turn_checkpoint(
                    first_turn_path,
                    manifest_digest,
                    problem_id,
                    reply,
                )

            result = solve_problem(
                base_url,
                by_id[problem_id],
                manifest_digest=manifest_digest,
                first_reply=first_reply,
                publish_first_reply=publish_first_reply,
                first_max_tokens=args.first_max_tokens,
                review_max_tokens=args.review_max_tokens,
                enable_thinking=args.enable_thinking,
                temperature=args.temperature,
                seed=args.seed,
                request_timeout=args.request_timeout,
            )
            append_jsonl_durable(results_path, result)
            existing_results[problem_id] = result
            print(
                "PASS" if result["review_correct"] else "MISS",
                f"id={problem_id} expected={result['expected_answer']:03d} "
                f"first={result['first_answer']} review={result['review_answer']}",
                flush=True,
            )

        if process is not None:
            stop_server(process)
            process = None
        if log_handle is not None:
            log_handle.close()
            log_handle = None
        if execution_contract_required:
            execution_contract = validate_execution_contract(
                output_dir / "perfstats.json",
                args.device,
                required_mtp_depth=args.mtp_depth,
                require_prefix_cache=args.prefix_cache,
            )
            execution_contract["manifest_digest"] = manifest_digest
            if prefix_cache_canary_path is not None:
                execution_contract["prefix_cache_canary_sha256"] = (
                    sha256_file(prefix_cache_canary_path)
                )
            write_json_atomic(
                output_dir / "execution-contract.json",
                execution_contract,
            )

        summary = make_summary(existing_results.values(), problem_ids)
        write_json_atomic(output_dir / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0 if summary["completed"] == len(problem_ids) else 2
    finally:
        if process is not None:
            stop_server(process)
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
