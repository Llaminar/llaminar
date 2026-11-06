#!/usr/bin/env python3
## @file collect_native_speculation.py
## @brief Freeze and collect native local AR/MTP evidence through canonical runners.
## The independent evaluator owns inputs, process environment and output storage.
## Saved native apply documents own placement; canonical prerequisite validators
## own admission. This adapter never stages models, searches placements, invents
## missing observations, retries policies or claims production qualification.
"""Collect a frozen local native-speculation experiment; never certify an image.

``freeze`` checks existing local artifacts and emits a content-pinned collection
manifest. ``collect`` uses that manifest to run the complete paired schedule via
the canonical native benchmark launcher. See docs/native-speculation-collection.md.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

import evaluate_native_speculation as evaluation
from production_artifacts import digest, model_identities, validate_prerequisites, write_json
import run_model_parity_benchmarks as benchmark
import run_production_prerequisites as prerequisites

SCHEMA = "llaminar.native-speculation-collection.v1"
EXECUTION_SCHEMA = "llaminar.native-speculation-local-execution.v1"
STATUS_SCHEMA = "llaminar.native-speculation-collection-status.v1"


def canonical_path(value: object) -> Path:
    """Require explicit existing absolute paths; never infer a binary or model."""
    evaluation.require(isinstance(value, str) and Path(value).is_absolute(), "expected absolute local path")
    path = Path(value).resolve(strict=True)
    evaluation.require(path == Path(value), f"path must name its canonical target: {value}")
    return path


def file_identity(path: Path) -> list[int]:
    """Observe storage identity only; this is not an immutable-content certificate."""
    evaluation.require(path.is_file(), f"expected existing regular file: {path}")
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def snapshot(paths: list[str]) -> dict:
    """Bind every retained input file to the same native stat identity tuple."""
    return {name: file_identity(Path(name)) for name in paths}


def binary_sha256(path: Path) -> str:
    """Hash a small local executable once while rejecting concurrent replacement."""
    before = file_identity(path)
    with path.open("rb") as handle:
        result = hashlib.file_digest(handle, "sha256").hexdigest()
    evaluation.require(file_identity(path) == before, f"binary changed while hashing: {path}")
    return result


def source_root(build: Path) -> Path:
    """Resolve a configured checkout without guessing from the executable name."""
    homes = [line.split("=", 1)[1] for line in (build / "CMakeCache.txt").read_text().splitlines()
             if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    evaluation.require(len(homes) == 1, "build needs exactly one configured source directory")
    home = canonical_path(homes[0])
    evaluation.require(home.parts[-2:] == ("src", "v2"), "expected a configured V2 checkout")
    return home.parent.parent


def build_contract(build: Path) -> dict:
    """Match test/Release ISA, backend and compiler selection, not optimization flags."""
    entries = {}
    for line in (build / "CMakeCache.txt").read_text().splitlines():
        if "=" in line and not line.startswith(("#", "//")):
            key, value = line.split("=", 1)
            name = key.split(":", 1)[0]
            evaluation.require(name not in entries, f"duplicated CMake cache entry: {name}")
            entries[name] = value
    required = {"LLAMINAR_CPU_ISA", "HAVE_CUDA", "HAVE_ROCM", "CMAKE_CXX_COMPILER"}
    evaluation.require(required <= set(entries) and entries["LLAMINAR_CPU_ISA"].upper() in {"AVX2", "AVX512"}
                       and all(entries[key] in {"ON", "OFF"} for key in ("HAVE_CUDA", "HAVE_ROCM")),
                       "build lacks an explicit ISA/backend/compiler contract")
    for backend, compiler in (("HAVE_CUDA", "CMAKE_CUDA_COMPILER"), ("HAVE_ROCM", "CMAKE_HIP_COMPILER")):
        if entries[backend] == "ON":
            required.add(compiler)
    evaluation.require(required <= set(entries) and all(entries[key] for key in required), "build lacks selected compiler identities")
    return {key: entries[key].upper() if key == "LLAMINAR_CPU_ISA" else entries[key] for key in sorted(required)}


def source_revision(root: Path) -> str:
    """Admit an exact clean source commit; annotated object identities are not commits."""
    revision = subprocess.check_output(["git", "rev-parse", "HEAD^{commit}"], cwd=root, text=True).strip()
    subprocess.run(["git", "diff-index", "--quiet", "HEAD", "--"], cwd=root, check=True)
    return revision


def inspect_inputs(plan_path: Path, plan_sha256: str) -> tuple[evaluation.FrozenPlan, dict, dict]:
    """Reuse canonical admission and bind one local deployment before execution.

    Model metadata must provide the complete existing file list and its canonical
    stat identity. Payload hashes/acquisition provenance remain external. Compiler
    and runtime-library metadata are trusted build declarations, not discovered
    attestations. Full lossless apply documents are passed to the native parser;
    the collector checks only its local/fixed/unprofiled execution constraints.
    """
    plan_path = plan_path.resolve(strict=True)
    plan = evaluation.load_plan(plan_path, plan_sha256)
    document = plan.document
    paths = {str(plan_path)}

    def read(reference: dict, directory: Path) -> tuple[dict, Path]:
        """Read pinned setup JSON and include its actual path in the input guard."""
        value, _ = evaluation.artifact(reference, directory)
        path = (directory / reference["path"]).resolve(strict=True)
        paths.add(str(path))
        return value, path

    model, _ = read(document["deployment"]["model"], plan_path.parent)
    files = model.get("files")
    evaluation.require(isinstance(files, list) and files and len(set(files)) == len(files),
                       "collector model artifact needs a complete unique files list")
    for name in files:
        file_identity(canonical_path(name))
        paths.add(name)
    observed_models = model_identities({"cells": [{"model_files": files}]})
    evaluation.require(digest(observed_models) == digest(model.get("stat_identity")),
                       "model storage identity differs from frozen acquisition metadata")
    hardware, _ = read(document["deployment"]["hardware"], plan_path.parent)
    evaluation.require(digest(hardware) == digest(benchmark.hardware_identity()), "physical hardware identity differs")
    execution, execution_path = read(document["deployment"]["execution"], plan_path.parent)
    evaluation.require(set(execution) == {"schema", "environment", "roles"}
                       and execution["schema"] == EXECUTION_SCHEMA, "unsupported local execution binding")
    environment = execution["environment"]
    evaluation.require(isinstance(environment, dict) and environment.get("PATH")
                       and all(isinstance(key, str) and key and "=" not in key and "\x00" not in key
                               and isinstance(value, str) and "\x00" not in value for key, value in environment.items()),
                       "collector needs an explicit complete child environment")
    # The collector never inherits a profiler, MPI parent, or hidden runtime
    # knobs. An externally attached profiler is still an evaluator trust issue.
    forbidden = ("OMPI_", "PMI_", "PMIX_", "NSYS_", "NCU_", "ROCPROF_", "LD_PRELOAD", "LD_AUDIT",
                 "CUDA_INJECTION", "CUDA_PROFILE", "HSA_TOOLS_LIB", "ROCP_TOOL_LIBRARIES")
    diagnostic_names = {"PROFILE", "PROFILER", "DETERMINISTIC", "BENCHMARK", "TRACE", "DUMP", "SNAPSHOT", "SNAPSHOTS", "TIMING", "DEBUG"}
    evaluation.require(not any(key.startswith(forbidden) or
                               (key.startswith("LLAMINAR_") and diagnostic_names.intersection(key.split("_")))
                               for key in environment),
                       "profiled, deterministic, inherited-MPI or iteration overrides are not collectable")
    roles = execution["roles"]
    evaluation.require(isinstance(roles, dict) and set(roles) == {role.value for role in evaluation.Role},
                       "execution must bind every role")
    runtime, placements = {}, []
    for role in evaluation.Role:
        binding = roles[role.value]
        evaluation.require(isinstance(binding, dict) and set(binding) == {
            "binary", "configuration", "preflight_build_directory", "prerequisites"}, "incomplete local role binding")
        binary = canonical_path(binding["binary"])
        evaluation.require(os.access(binary, os.X_OK), "native binary is not executable")
        paths.update((str(binary), str(binary.parent / "CMakeCache.txt")))
        build, _ = read(document["implementations"][role.value]["build"], plan_path.parent)
        evaluation.require(binary_sha256(binary) == build["binary_sha256"], "native binary digest differs")
        evaluation.require(benchmark.e2e.release_binary_cpu_isa(binary).value == build["cpu_isa"], "native Release ISA differs")
        root = source_root(binary.parent)
        evaluation.require(source_revision(root) == build["source_revision"], "native source revision differs")
        integration = canonical_path(binding["preflight_build_directory"])
        evaluation.require(source_root(integration) == root, "prerequisites belong to another source checkout")
        evaluation.require(digest(build_contract(binary.parent)) == digest(build_contract(integration)),
                           "Release and prerequisite ISA/backend/compiler contracts differ")
        receipt, receipt_path = read(binding["prerequisites"], execution_path.parent)
        validate_prerequisites(receipt)
        prerequisites.validate_source_freshness(integration, receipt_path, source_root=root)
        code, _, _ = prerequisites.reuse_unchanged_production_test_preflight(integration, receipt_path)
        evaluation.require(code == 0, "canonical prerequisite admission failed")
        paths.update(str(path) for path in (integration / "CMakeCache.txt", integration / "build.ninja",
                                           integration / ".ninja_log", *integration.rglob("CTestTestfile.cmake")))
        config, config_path = read(binding["configuration"], execution_path.parent)
        evaluation.require(set(config) == {"kind", "schema_version", "configuration"}
                           and config["kind"] == "llaminar.orchestration-config"
                           and type(config["schema_version"]) is int and config["schema_version"] == 1,
                           "collector requires a lossless native apply document")
        policy = config["configuration"]
        evaluation.require(policy["planning_mode"] == "Apply" and policy["hostfile"] == ""
                           and policy["mpi_no_bootstrap"] is False and policy["deterministic"] is False,
                           "collector requires fixed local apply intent with ordinary bootstrap and optimized dispatch")
        evaluation.require(policy["model_path"] == files[0], "apply document selects another primary model")
        evaluation.require(policy["mtp_activation_policy"] == ("Disabled" if role is evaluation.Role.AR else "Enabled")
                           and policy["mtp"]["enabled"] is (role is not evaluation.Role.AR)
                           and type(policy["mtp"]["max_request_batch"]) is int and policy["mtp"]["max_request_batch"] == 1
                           and type(document["implementations"][role.value]["config"]["mtp_max_request_batch"]) is int
                           and document["implementations"][role.value]["config"]["mtp_max_request_batch"] == 1,
                           "collector requires native single-request AR/MTP role intent")
        placements.append({key: value for key, value in policy.items() if key not in {"mtp", "mtp_activation_policy"}})
        runtime[role.value] = {"binary": str(binary), "configuration": str(config_path), "source_root": str(root)}
    evaluation.require(len({digest(value) for value in placements}) == 1, "saved placement or non-MTP policy differs")
    for request in plan.requests.values():
        work = request["workload"]
        evaluation.require(work["measurement_iterations"] <= 100 and work["warmup_iterations"] <= 100,
                           "native benchmark iteration environment is clamped at 100")
    observed = {"files": snapshot(sorted(paths)), "hardware": hardware,
                "cpu_affinity": sorted(os.sched_getaffinity(0))}
    return plan, {"environment": environment, "roles": runtime}, observed


def schedule(plan: evaluation.FrozenPlan) -> list[dict]:
    """Rotate role order deterministically while keeping held-out requests last."""
    result = []
    roles = [role.value for role in evaluation.Role]
    requests = plan.requests
    for split in evaluation.Split:
        for request_index, (name, request) in enumerate(requests.items()):
            if request["split"] != split.value:
                continue
            for pair_index, pair in enumerate(plan.document["pairs"]):
                offset = (request_index + pair_index) % len(roles)
                result.append({"request": name, "pair": pair, "roles": roles[offset:] + roles[:offset]})
    return result


def freeze(plan_path: Path, plan_sha256: str, output: Path) -> Path:
    """Check local setup, then publish one new externally pinnable collection manifest."""
    plan, runtime, observed = inspect_inputs(plan_path, plan_sha256)
    output = output.resolve()
    for binding in runtime["roles"].values():
        evaluation.require(not output.is_relative_to(binding["source_root"]), "collection metadata must live outside source checkouts")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": SCHEMA, "plan": {"path": str(plan_path.resolve()), "sha256": plan.sha256},
                "identity": plan.identity(), "runtime": runtime, "observed": observed, "schedule": schedule(plan),
                "certification_eligible": False}
    path = output / "collection.json"
    write_json(path, manifest)
    return path


def collect(collection_path: Path, expected_sha256: str, output: Path) -> dict:
    """Collect every frozen block once; failed or changed runs never publish green evidence.

    File identities bracket each launch; canonical source/receipt checks and binary
    hashes are repeated at collection admission and completion. The metadata is
    diagnostic evidence binding, not protection against a malicious evaluator or
    a transient external profiler/hardware change. Local storage stays fixed.
    """
    manifest, _ = evaluation.read_document(collection_path, expected_sha256)
    evaluation.require(set(manifest) == {"schema", "plan", "identity", "runtime", "observed", "schedule", "certification_eligible"}
                       and manifest["schema"] == SCHEMA and manifest["certification_eligible"] is False,
                       "unsupported frozen collection manifest")
    output = output.resolve()
    for binding in manifest["runtime"]["roles"].values():
        evaluation.require(not output.is_relative_to(binding["source_root"]), "collection output must live outside source checkouts")
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema": STATUS_SCHEMA, "status": "collecting", "certification_eligible": False,
              "collection_sha256": expected_sha256, "plan_sha256": manifest["plan"]["sha256"],
              "planned_observations": len(manifest["schedule"]) * len(evaluation.Role), "observations": []}
    write_json(output / "status.json", status)
    started = time.monotonic()
    try:
        plan_path = Path(manifest["plan"]["path"])
        plan, runtime, observed = inspect_inputs(plan_path, manifest["plan"]["sha256"])
        evaluation.require(digest({"identity": plan.identity(), "runtime": runtime, "observed": observed, "schedule": schedule(plan)})
                           == digest({key: manifest[key] for key in ("identity", "runtime", "observed", "schedule")}),
                           "collection inputs differ from the frozen setup")
        requests = plan.requests
        documents = plan.document["implementations"]
        reference_outputs = {}
        paths = list(observed["files"])
        for block in manifest["schedule"]:
            for name in block["roles"]:
                evaluation.require(snapshot(paths) == observed["files"]
                                   and sorted(os.sched_getaffinity(0)) == observed["cpu_affinity"], "inputs changed before launch")
                number = len(status["observations"]) + 1
                directory = output / f"observation-{number:04d}"
                directory.mkdir()
                binding = runtime["roles"][name]
                work = requests[block["request"]]["workload"]
                status["active"] = {"request": block["request"], "pair": block["pair"], "role": name,
                                    "directory": str(directory)}
                write_json(output / "status.json", status)
                arguments = ["benchmark", "--config", binding["configuration"],
                             *benchmark.workload_arguments(str(directory / "benchmark.json"), work)]
                benchmark.run_local_benchmark(Path(binding["binary"]), directory, arguments, work,
                                              environment=runtime["environment"])
                evaluation.require(snapshot(paths) == observed["files"], "inputs changed during launch")
                data, data_sha = evaluation.read_document(directory / "benchmark.json")
                sample = evaluation.native_measurement(data, work, documents[name]["config"], evaluation.Role(name))
                evaluation.timing_observations(data)
                previous = reference_outputs.setdefault(block["request"], sample["output"])
                evaluation.require(previous == sample["output"], "exact token mismatch between collected roles/blocks")
                observation = {"request": block["request"], "pair": block["pair"], "role": name,
                               "deployment_sha256": plan.deployment_sha256, "build_sha256": plan.builds[name],
                               "measurement": "unprofiled",
                               "benchmark": {"path": str((directory / "benchmark.json").relative_to(output)), "sha256": data_sha}}
                status["observations"].append(observation)
                write_json(output / "status.json", status)
        _, final_runtime, final_observed = inspect_inputs(plan_path, plan.sha256)
        evaluation.require(digest(final_runtime) == digest(runtime) and digest(final_observed) == digest(observed),
                           "collection inputs changed before completion")
        evidence = {"schema": evaluation.EVIDENCE_SCHEMA, "plan_sha256": plan.sha256,
                    "observations": status["observations"]}
        pending = output / "evidence.pending.json"
        write_json(pending, evidence)
        report = evaluation.evaluate(plan, pending)
        write_json(output / "comparison.json", report)
        pending.replace(output / "evidence.json")
        status.pop("active", None)
        status.update(status="complete", evidence_sha256=evaluation.read_document(output / "evidence.json")[1])
        return status
    except (Exception, KeyboardInterrupt) as error:
        status.update(status="failed", failure={"type": type(error).__name__, "detail": str(error)})
        raise
    finally:
        status["elapsed_wall_seconds"] = time.monotonic() - started
        write_json(output / "status.json", status)


def main(argv: list[str] | None = None) -> int:
    """Expose explicit freeze/collect phases without a fixture, retry or gate bypass."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("freeze")
    setup.add_argument("--plan", type=Path, required=True)
    setup.add_argument("--plan-sha256", required=True)
    setup.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("collect")
    run.add_argument("--collection", type=Path, required=True)
    run.add_argument("--collection-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            path = freeze(args.plan, args.plan_sha256, args.output)
            print(f"Frozen collection: {path}\nSHA-256: {evaluation.read_document(path)[1]}")
        else:
            collect(args.collection, args.collection_sha256, args.output)
            print(f"Diagnostic evidence written to {args.output}; not production qualification.")
        return 0
    except (ValueError, KeyError, TypeError, AttributeError, OSError, subprocess.SubprocessError, KeyboardInterrupt) as error:
        print(f"[native-speculation-collection] ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
