#!/usr/bin/env python3
"""Exercise canonical generation cells with the shared Release server harness.

Routine regression executes only serial controls and dynamic MTP, using the
versioned approved serial corpus. Collection and comparison of unapproved
observations remain explicit diagnostics. No run records MTP-specific answers,
repairs expected tokens, commits or downloads a corpus.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import time

from generation_regression_http import MTPPolicy, admit_control, generation_profile, observation_traces, serial_workload_identity
from generation_tokens import compare_tokens
from generation_corpus import ApprovedGenerationCorpus
from model_parity_inventory import InventoryScope, discover, source_revision
from production_artifacts import digest, image_identity, model_identities, validate_prerequisites, write_json
import run_model_parity_e2e as e2e
import run_production_parity_campaigns as parity

ROOT = Path(__file__).resolve().parents[2]


class RunMode(str, Enum):
    """Regression consumes approval; acquisition can never approve its own output."""
    REGRESSION = "regression"
    COLLECT = "collect-controls"
    COMPARE = "compare-controls"


@dataclass(frozen=True)
class GenerationCell:
    """Bind one selected cell to its actual current canonical serial definition.

    A stable control ID and matching prompts do not authenticate runtime policy:
    topology/economy defaults can change without renaming the generated case.
    Carry the complete Off record discovered from the same inventory so old
    observations cannot silently authorize a different configuration. No CLI
    option parsing or second policy projection is involved.
    """
    campaign: parity.CampaignCell
    exact: str
    configuration: dict
    serial_control: dict

    def admit_control(self, expected: dict) -> dict:
        """Validate persisted evidence against both canonical configurations."""
        if not isinstance(expected, dict) or expected.get("configuration") != self.serial_control:
            raise ValueError("serial observation differs from the current canonical control configuration")
        return admit_control(self.configuration, expected)


def select_cells(args: argparse.Namespace, scope: InventoryScope = InventoryScope.GENERATION) -> list[GenerationCell]:
    """Resolve controls from the full inventory, including for a narrow selector.

    Exact control relationships come from C++; Python neither expands another
    MTP axis nor rewrites argv to synthesize a serial configuration.
    """
    unfiltered = copy.copy(args)
    unfiltered.backend = unfiltered.campaign = unfiltered.cell = ".*"
    inventory = discover(unfiltered, InventoryScope.ALL)
    by_id = {}
    for row in inventory:
        record = row[2]
        generation_profile(record)
        if record["id"] in by_id:
            raise ValueError("ambiguous canonical generation cell identity: " + record["id"])
        by_id[record["id"]] = row
    selected, seen = [], set()
    for campaign, exact, record in inventory:
        if not scope.accepts(record):
            continue
        if (not re.fullmatch(args.backend, campaign.group.backends)
                or not re.fullmatch(args.campaign, campaign.name) or not re.fullmatch(args.cell, exact)):
            continue
        control_id = generation_profile(record)["serial_control_id"]
        if control_id not in by_id:
            raise ValueError("canonical inventory omitted serial control: " + control_id)
        control = by_id[control_id]
        if (MTPPolicy(generation_profile(control[2])["mtp_policy"]) is not MTPPolicy.OFF
                or serial_workload_identity(control[2]) != serial_workload_identity(record)):
            raise ValueError("invalid canonical serial-control relationship: " + exact)
        chosen = control if args.mode is RunMode.COLLECT else (campaign, exact, record)
        if chosen[1] not in seen:
            seen.add(chosen[1])
            selected.append(GenerationCell(*chosen, serial_control=control[2]))
    if not selected:
        raise ValueError("no canonical generation cells selected")
    return selected


def control_path(root: Path, record: dict) -> Path:
    """Use a bounded filename for metadata identity, never hash model payloads."""
    key = digest({"control_id": generation_profile(record)["serial_control_id"]})
    return root / key / "generation" / "observations.json"


def generation_environment(record: dict, directory: Path) -> dict:
    """Select the shared harness workload without inherited E2E substitutions."""
    profile = generation_profile(record)
    return {**os.environ, "LLAMINAR_E2E_LONG_CONTEXT": "0",
            "LLAMINAR_E2E_THINKING_MODES": "non-thinking",
            "LLAMINAR_E2E_STARTUP_TIMEOUT_SECONDS": str(profile["readiness_timeout_seconds"]),
            "LLAMINAR_E2E_MOVEMENT_EVIDENCE": e2e.movement_evidence(record["runtime"]),
            "LLAMINAR_E2E_LOG_DIR": str(directory), "LLAMINAR_E2E_PERF_STATS": "1",
            "LLAMINAR_E2E_PERF_STATS_GPU_STAGE_TIMING": "0", "LLAMINAR_E2E_TRACE_TOKENS": "0"}


def validate_regression_report(report: dict, inventory: dict, prerequisites: dict,
                               image: str, cpu_isa: str, corpus_digest: str) -> None:
    """Authenticate complete routine coverage and immutable baseline/image binding.

    Diagnostic comparison reports cannot certify an image. The pipeline loads
    the expected corpus pin from its admitted source, never from this report.
    Fixed-depth cells cannot replace omitted dynamic or serial controls.
    """
    validate_prerequisites(prerequisites)
    expected = {row["case"]: row["configuration"] for row in inventory["cells"]
                if InventoryScope.GENERATION.accepts(row["configuration"])}
    if (not expected or report.get("mode") != RunMode.REGRESSION.value
            or report.get("complete") is not True or report.get("passed") is not True
            or report.get("certification_eligible") is not True or report.get("image") != image
            or report.get("cpu_isa") != cpu_isa or report.get("corpus_digest") != corpus_digest
            or report.get("source_revision") != inventory["source_revision"]
            or report.get("inventory_digest") != digest(inventory)
            or report.get("prerequisite_report_digest") != digest(prerequisites)
            or type(report.get("selected")) is not int or report["selected"] != len(expected)):
        raise ValueError("generation regression is incomplete or has stale image/corpus/prerequisite identity")
    rows = report.get("cells")
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(row, dict) or row.get("case") not in expected
                   or type(row.get("return_code")) is not int or row["return_code"] != 0
                   or row.get("evidence_error") is not None
                   or row.get("configuration") != expected[row["case"]] for row in rows)
            or len({row["case"] for row in rows}) != len(expected)):
        raise ValueError("generation regression omitted, duplicated or changed canonical routine cells")


def main(argv: list[str] | None = None) -> int:
    """Run prerequisites once, then exact sequential cells with fail-fast evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", type=RunMode, choices=list(RunMode), default=RunMode.REGRESSION)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_v2_integration")
    parser.add_argument("--binary", type=Path, default=ROOT / "build_v2_release/llaminar2")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-revision")
    parser.add_argument("--backend", default=".*")
    parser.add_argument("--campaign", default=".*")
    parser.add_argument("--cell", default=".*")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--container-image")
    parser.add_argument("--controls", type=Path, help="Unapproved serial observation directory, read-only during comparison")
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "corpora")
    parser.add_argument("--cpu-isa", choices=("AVX512", "AVX2"))
    parser.add_argument("--prerequisite-report", type=Path,
                        help="Pipeline-owned installed-builder receipt; never rerun gates for each runtime cell")
    parser.add_argument("--output", type=Path, required=True, help="New evidence directory; existing output is never replaced")
    parser.add_argument("--port", type=int, default=19480)
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    parser.add_argument("--persistent-model-cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--reuse-preflight-report", type=Path,
                        help="Canonical numerical/generation prerequisite receipt for an unchanged build")
    args = parser.parse_args(argv)
    if args.prerequisite_report and (args.reuse_preflight_report or not args.container_image or not args.manifest):
        parser.error("installed prerequisite evidence requires an image/full manifest and excludes local receipt reuse")
    if args.mode is RunMode.REGRESSION and not args.list and (not args.cpu_isa or not args.manifest):
        parser.error("regression requires --cpu-isa and the full --manifest")
    if args.mode is RunMode.REGRESSION and args.controls:
        parser.error("regression consumes the reviewed corpus, not unapproved --controls")
    selected = select_cells(args)
    if args.mode is RunMode.COMPARE and args.controls is None:
        parser.error("comparison requires --controls; expected streams are never generated implicitly")
    if args.mode is RunMode.COLLECT and args.controls is not None:
        parser.error("control collection cannot modify or reuse an existing control corpus")
    for cell in selected:
        record = cell.configuration
        print(f"[model-parity-generation] selected={cell.exact} "
              f"control={generation_profile(record)['serial_control_id']}", flush=True)
        if args.mode is RunMode.COMPARE and not args.list:
            cell.admit_control(json.loads(control_path(args.controls, record).read_text()))
    if args.list:
        return 0
    corpus = None
    inventory = json.loads(args.manifest.read_text()) if args.manifest else None
    if args.mode is RunMode.REGRESSION:
        corpus = ApprovedGenerationCorpus.load_reviewed(
            ROOT, args.corpus_root, args.cpu_isa, inventory, model_identities(inventory))
        # Validate every selected mapping before paying for any gate or model.
        for cell in selected:
            corpus.expected(cell.configuration)
    if args.container_image:
        image = image_identity(args.container_image)
        if corpus and (image["labels"].get("org.llaminar.cpu_isa") != args.cpu_isa
                       or image["labels"].get("org.llaminar.build_type") != "Release"):
            raise ValueError("generation image does not match the admitted Release/ISA corpus")
        args.container_image = image["id"]
    else:
        cache = args.binary.resolve().parent / "CMakeCache.txt"
        if not cache.is_file() or not re.search(r"^CMAKE_BUILD_TYPE:STRING=Release$", cache.read_text(), re.M):
            raise ValueError("generation requires a Release binary and its CMakeCache.txt")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "mode": args.mode.value,
              "certification_eligible": bool(corpus and args.container_image and args.prerequisite_report
                                               and args.backend == args.campaign == args.cell == ".*"),
              "source_revision": source_revision(args), "image": args.container_image,
              "cpu_isa": args.cpu_isa, "inventory_digest": digest(inventory) if inventory else None,
              "corpus_digest": corpus.pin.document_digest if corpus else None,
              "selected": len(selected), "complete": False, "passed": False, "cells": []}
    started = time.monotonic()
    try:
        installed = json.loads(args.prerequisite_report.read_text()) if args.prerequisite_report else None
        if installed is not None:
            validate_prerequisites(installed)
            report["prerequisite_report_digest"] = digest(installed)
        prerequisites = ((installed["preflight_return_code"], installed["preflight_elapsed_seconds"],
                          installed["preflight_tests"]) if installed is not None else
                         parity.reuse_unchanged_production_parity_preflight(args.build_dir, args.reuse_preflight_report)
                         if args.reuse_preflight_report else
                         parity.run_production_parity_preflight(args.build_dir, None,
                                                               artifact_directory=args.output / "preflight"))
        code, elapsed, tests = prerequisites
        report.update(preflight_return_code=code, preflight_elapsed_seconds=elapsed,
                      preflight_tests=list(tests), preflight_test_count=len(tests),
                      preflight_build_directory=str(args.build_dir.resolve()),
                      preflight_completed_ns=time.time_ns())
        if args.reuse_preflight_report:
            previous = json.loads(args.reuse_preflight_report.read_text())
            report["preflight_completed_ns"] = previous.get(
                "preflight_completed_ns", args.reuse_preflight_report.stat().st_mtime_ns)
            report["preflight_reused_from"] = str(args.reuse_preflight_report.resolve())
        elif installed is not None:
            report["preflight_completed_ns"] = installed["preflight_completed_ns"]
            report["preflight_build_directory"] = installed["preflight_build_directory"]
        if code:
            raise RuntimeError("generation prerequisites failed")
        with parity.model_staging_workspace(args.model_ramdisk_root, args.persistent_model_cache_dir, None) as workspace:
            staged, _ = parity.stage_models_in_ramdisk([cell.campaign for cell in selected], workspace.models,
                                                       None, persistent=workspace.persistent)
            workspace.protect_published_models()
            paths = {Path(item.source_path).resolve(): str(workspace.models / item.filename) for item in staged}
            for index, cell in enumerate(selected, 1):
                campaign, exact, record = cell.campaign, cell.exact, cell.configuration
                key = (digest({"control_id": record["id"]}) if args.mode is RunMode.COLLECT
                       else digest({"cell_id": record["id"]}))
                directory = args.output / key
                directory.mkdir()
                write_json(directory / "configuration.json", record)
                write_json(directory / "server-args.json", [*record["runtime"]["server_args"],
                           "--context-length", str(record["runtime"]["context_length"])])
                model = paths[Path(record["model"]).resolve()]
                if any(ch in model + record["id"] for ch in "|\r\n"):
                    raise ValueError("model/cell identity contains a harness delimiter")
                command = ["bash", str(ROOT / "tests/v2/e2e/server/test_server_e2e.sh"),
                           "--binary", str(args.binary.resolve()), "--port", str(args.port),
                           "--suite", f"{model}|{e2e.harness_backend(campaign.group.backends)}|"
                                      f"{generation_profile(record)['max_tokens']}||{record['id']}|"
                                      "generation-regression,require-decode-graph-replay",
                           "--server-args-file", str(directory / "server-args.json"),
                           "--generation-configuration", str(directory / "configuration.json")]
                if args.container_image:
                    command += ["--container-image", args.container_image]
                if args.mode is RunMode.COMPARE:
                    command += ["--generation-control", str(control_path(args.controls.resolve(), record))]
                elif corpus:
                    write_json(directory / "expected-tokens.json", corpus.expectations_document(record))
                    command += ["--generation-expected-tokens", str(directory / "expected-tokens.json")]
                print(f"[model-parity-generation] {index}/{len(selected)} RUN {exact}", flush=True)
                cell_started = time.monotonic()
                with (directory / "harness.log").open("w") as log:
                    code = e2e.run_e2e_process(command, generation_environment(record, directory), log)
                evidence_error = None
                if code == 0:
                    try:
                        evidence = json.loads((directory / "generation/observations.json").read_text())
                        traces = observation_traces(record, evidence)
                        if args.mode is RunMode.COMPARE:
                            controls = cell.admit_control(json.loads(control_path(args.controls, record).read_text()))
                            if any(compare_tokens(controls[name], trace) for name, trace in traces.items()):
                                raise ValueError("persisted generation responses differ from their serial control")
                        elif corpus:
                            controls = corpus.expected(record)
                            if any(compare_tokens(controls[name], trace) for name, trace in traces.items()):
                                raise ValueError("persisted generation responses differ from reviewed serial tokens")
                    except (ValueError, OSError) as error:
                        code, evidence_error = 1, str(error)
                report["cells"].append({"case": exact, "configuration": record, "return_code": code,
                                       "evidence_error": evidence_error,
                                       "elapsed_seconds": time.monotonic() - cell_started,
                                       "artifacts": str(directory)})
                write_json(args.output / "report.json", report)
                print(f"[model-parity-generation] {index}/{len(selected)} "
                      f"{'PASS' if code == 0 else 'FAIL'} {exact}", flush=True)
                if code:
                    raise RuntimeError("generation cell failed: " + exact)
        report.update(complete=True, passed=True)
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
