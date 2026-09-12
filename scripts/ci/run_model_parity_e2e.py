#!/usr/bin/env python3
"""Certify tagged canonical model-parity cells through the Release HTTP harness.

CTest owns the binary/cell inventory. GoogleTest's typed parameter printer owns
configuration and eligibility. This module neither expands axes nor maintains
a model/topology table. The mature HTTP and full long-context checks remain the
behavioral oracle; mathematical parity remains a separate, complementary gate.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import run_production_parity_campaigns as parity
from production_artifacts import digest, image_identity
from model_parity_inventory import InventoryScope, discover as discover_inventory, export_manifest, source_revision

ROOT = Path(__file__).resolve().parents[2]


def discover(args: argparse.Namespace) -> list[tuple[parity.CampaignCell, str, dict]]:
    """Select typed E2E tags from the same inventory as generation regression."""
    return discover_inventory(args, InventoryScope.E2E)


def readiness_timeout_seconds(profile: dict) -> int:
    """Validate exported readiness policy before staging or starting a server."""
    readiness = profile.get("readiness_timeout_seconds")
    if (type(readiness) is not int or readiness <= 0
            or readiness > parity.EXACT_CELL_TIMEOUT_SECONDS):
        raise ValueError("E2E readiness timeout must be a positive integer within the exact-cell watchdog; rebuild/export stale manifests")
    return readiness


def thinking_modes(profile: dict) -> str:
    """Reject stale discovery instead of guessing reasoning support from names."""
    modes = profile.get("thinking_modes")
    if modes not in ("both", "non-thinking"):
        raise ValueError("E2E thinking_modes must be explicitly both or non-thinking; rebuild/export stale manifests")
    return modes


def movement_evidence(profile: dict) -> str:
    """Use the typed cell's obligation; a runtime policy default is not proof."""
    evidence = profile.get("movement_evidence")
    if evidence not in ("not_applicable", "forbidden", "required"):
        raise ValueError("E2E movement_evidence must be explicit; rebuild/export stale manifests")
    return evidence


def certification_environment(profile: dict, artifact_dir: Path) -> dict[str, str]:
    """Pin full checks and geometry to the typed profile, not inherited knobs."""
    readiness = readiness_timeout_seconds(profile)
    return {**os.environ,
            "LLAMINAR_E2E_THINKING_MODES": thinking_modes(profile),
            "LLAMINAR_E2E_MOVEMENT_EVIDENCE": movement_evidence(profile),
            "LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS": str(readiness),
            "LLAMINAR_E2E_LONG_CONTEXT": "1",
            "LLAMINAR_E2E_LONG_CONTEXT_TIER": "full",
            "LLAMINAR_E2E_CONTEXT_LENGTH": str(profile["context_length"]),
            "LLAMINAR_E2E_LONG_MIN_PROMPT_TOKENS": str(profile["minimum_prompt_tokens"]),
            "LLAMINAR_E2E_LONG_MAX_TOKENS": str(profile["generation_tokens"]),
            "LLAMINAR_E2E_LONG_REQUEST_TIMEOUT": str(profile["request_timeout_seconds"]),
            "LLAMINAR_E2E_LOG_DIR": str(artifact_dir),
            "LLAMINAR_E2E_PERF_STATS": "1"}


def validate_long_context_evidence(directory: Path, profile: dict) -> None:
    """Reject a successful shell invocation that omitted full behavioral proof."""
    evidence = json.loads((directory / "long_context_results.json").read_text())
    if (evidence.get("schema") != 1 or evidence.get("tier") != "full"
            or evidence.get("complete") is not True
            or len(evidence.get("results", [])) != 8
            or not all(row.get("passed") is True for row in evidence["results"])):
        raise ValueError("missing or failed full long-context evidence")
    for field in ("context_length", "minimum_prompt_tokens", "generation_tokens"):
        if evidence.get(field) != profile[field]:
            raise ValueError(f"long-context evidence has wrong {field}")


def run_server_harness(command: list[str], environment: dict[str, str], log) -> int:
    """Bound one exact HTTP cell and retire its full server/MPI process group.

    The canonical ten-minute cell watchdog includes server startup and checks.
    A request timeout cannot bound a stuck launcher or shutdown. Reuse the
    parity driver's existing process-group retirement protocol.
    """
    with subprocess.Popen(command, cwd=ROOT, env=environment,
                          stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=True) as process:
        try:
            return process.wait(timeout=parity.EXACT_CELL_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            parity._terminate_process_group(process)
            return 124
        except BaseException:
            parity._terminate_process_group(process)
            raise


def harness_backend(backend_signature: str) -> str:
    """Select the legacy harness's evidence lane, never its server placement.

    CPU NodeTP still needs CPU-only memory/graph assertions. The generic TP
    label denotes a GPU lane in the mature harness. Actual device/rank/domain
    arguments always come from the canonical argv file, independently of this
    diagnostic label.
    """
    return "cpu:0" if backend_signature == "CPU" else "tp"


def main(argv: list[str] | None = None) -> int:
    """Run tagged cases sequentially, retaining independent evidence per cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_v2_integration")
    parser.add_argument("--binary", type=Path, default=ROOT / "build_v2_release/llaminar2")
    parser.add_argument("--container-image")
    parser.add_argument("--manifest", type=Path,
                        help="Consume the CI builder's revision-bound typed discovery artifact")
    parser.add_argument("--export-manifest", type=Path,
                        help="Export tagged configurations without running inference")
    parser.add_argument("--source-revision",
                        help="CI source revision for export when the builder image omits .git")
    parser.add_argument("--backend", default=".*")
    parser.add_argument("--campaign", default=".*")
    parser.add_argument("--cell", default=".*")
    parser.add_argument("--port", type=int, default=19080)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    parser.add_argument("--persistent-model-cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--report", type=Path, default=ROOT / "parity-results/e2e-certification.json")
    args = parser.parse_args(argv)
    selected = discover(args)
    if args.export_manifest:
        args.export_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.export_manifest.write_text(json.dumps(export_manifest(
            selected, source_revision(args), InventoryScope.E2E), indent=2) + "\n")
        return 0
    for _, exact, record in selected:
        readiness = readiness_timeout_seconds(record["e2e"])
        thinking_modes(record["e2e"])
        movement_evidence(record["e2e"])
        print(f"[model-parity-e2e] selected {exact} context={record['e2e']['context_length']} readiness={readiness}s", flush=True)
    if args.list:
        return 0
    if args.container_image:
        # Pin tags once so every server and its report describe identical bytes.
        args.container_image = image_identity(args.container_image)["id"]
    if not args.container_image:
        cache = args.binary.resolve().parent / "CMakeCache.txt"
        if not cache.is_file() or not re.search(r"^CMAKE_BUILD_TYPE:STRING=Release$", cache.read_text(), re.M):
            raise ValueError("E2E certification requires a Release binary and its CMakeCache.txt")
    report = {"schema": 1, "selected": len(selected), "correctness_passed": False, "cells": [],
              "image": args.container_image,
              "source_revision": args.source_revision or subprocess.check_output(
                  ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "manifest_digest": digest(json.loads(args.manifest.read_text())) if args.manifest else None}
    args.report = args.report.resolve()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    run_root = args.report.parent / ("e2e-" + str(time.time_ns()))
    run_root.mkdir()
    started = time.monotonic()
    # Reuse the campaign's only staging/cache authority and keep its lease
    # until every server exits. Only the tagged models enter the RAM corpus.
    # Retain CMake's complete model manifest, including every GGUF shard. The
    # entry file alone is not a complete weight artifact for split models.
    staging_cells = [campaign for campaign, _, _ in selected]
    try:
        with parity.model_staging_workspace(args.model_ramdisk_root,
                                           args.persistent_model_cache_dir, None) as workspace:
            staged, _ = parity.stage_models_in_ramdisk(
                staging_cells, workspace.models, None, persistent=workspace.persistent)
            workspace.protect_published_models()
            staged_paths = {Path(item.source_path).resolve(): str(workspace.models / item.filename) for item in staged}
            for index, (campaign, exact, record) in enumerate(selected, 1):
                artifact_dir = run_root / str(index)
                artifact_dir.mkdir()
                config_file = artifact_dir / "server-args.json"
                config_file.write_text(json.dumps(record["e2e"]["server_args"]))
                model = staged_paths[Path(record["model"]).resolve()]
                if any(ch in str(model) + record["id"] for ch in "|\r\n"):
                    raise ValueError("model/cell identity cannot contain harness delimiters")
                command = ["bash", str(ROOT / "tests/v2/e2e/server/test_server_e2e.sh"),
                           "--binary", str(args.binary.resolve()),
                           "--suite", f"{model}|{harness_backend(campaign.group.backends)}|200||{record['id']}|e2e-certification",
                           "--server-args-file", str(config_file)]
                command += ["--port", str(args.port)]
                if args.container_image:
                    command += ["--container-image", args.container_image]
                print(f"[model-parity-e2e] {index}/{len(selected)} RUN {exact}", flush=True)
                cell_started = time.monotonic()
                with (artifact_dir / "harness.log").open("w") as log:
                    return_code = run_server_harness(command,
                        certification_environment(record["e2e"], artifact_dir), log)
                evidence_error = None
                if return_code == 0:
                    try:
                        validate_long_context_evidence(artifact_dir, record["e2e"])
                    except (ValueError, OSError) as error:
                        return_code = 1
                        evidence_error = str(error)
                report["cells"].append({"case": exact, "configuration": record,
                    "return_code": return_code, "artifacts": str(artifact_dir),
                    "evidence_error": evidence_error,
                    "outcome": "cell_timeout" if return_code == 124 else
                               "passed" if return_code == 0 else "failed",
                    "elapsed_seconds": time.monotonic() - cell_started})
                print(f"[model-parity-e2e] {index}/{len(selected)} "
                      f"{'PASS' if return_code == 0 else 'FAIL'} {exact}", flush=True)
                args.report.write_text(json.dumps(report, indent=2) + "\n")
                if return_code:
                    return return_code
        report["correctness_passed"] = True
        return 0
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"[model-parity-e2e] ERROR {error}", file=sys.stderr)
        sys.exit(1)
