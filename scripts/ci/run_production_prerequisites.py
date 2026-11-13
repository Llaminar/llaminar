#!/usr/bin/env python3
"""Run the canonical Unit/preflight transaction without admitting any model.

Run this command in the installed test runner before the outer CI driver starts
Release HTTP cells in the sibling runtime image. Local callers use the same
authority with an ordinary build tree. CTest owns both complete inventories;
the default entrypoint has no user selectors, skip switches, receipt synthesis or
model staging. CMake-authored optional wide geometries are selected against the
original native hardware inventory and remain explicitly unqualified when absent;
full registration and qualification evidence travel together. An explicit focused-panel diagnostic names DeviceFree checks only and
emits a separate non-certifying schema; it never denotes the complete gates. Image/source/ISA binding belongs to the outer pipeline that launches
the immutable test-runner ID and retains this command's canonical receipt.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import subprocess

from run_production_parity_campaigns import (
    PreflightResult,
    run_production_test_preflight,
    reuse_unchanged_production_test_preflight,
)


def validate_source_freshness(build: Path, report: Path) -> None:
    """Reject edits to interpreted tests and policies that Ninja cannot rebuild.

    Local receipt reuse belongs to this checkout's configured build. Git owns
    the source inventory; file and containing-directory metadata conservatively
    detects edits, additions and removals after the actual gate. No payload is
    read or hashed, and ignored models, caches and build products are excluded.
    The canonical validator separately authenticates success and test inventory.
    """
    _source_root_for_build(build)
    document = json.loads(report.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    completed = document.get("preflight_completed_ns")
    if type(completed) is not int or completed <= 0:
        raise ValueError("local prerequisite reuse requires an explicit gate completion time")
    _validate_source_freshness_at(build, completed)



def _source_root_for_build(build):
    """Keep the existing checkout authority and failure precedence for both receipts."""
    root = Path(__file__).resolve().parents[2]
    cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
    homes = [line.split("=", 1)[1] for line in cache.splitlines()
             if line.startswith("CMAKE_HOME_DIRECTORY:INTERNAL=")]
    if len(homes) != 1 or Path(homes[0]).resolve() != root / "src/v2":
        raise ValueError("prerequisite reuse requires this checkout's configured build")
    return root

def _validate_source_freshness_at(build: Path, completed: int) -> None:
    """Use the same interpreted-source boundary for complete and explicit focused evidence."""
    root = _source_root_for_build(build)
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
    ).decode("utf-8").split("\0")
    paths: set[Path] = set()
    for name in filter(None, names):
        path = root / name
        paths.add(path)
        # A removed tracked file still has a live containing directory. Its
        # timestamp records deletion; do not pretend that missing bytes passed.
        paths.update(parent for parent in path.parents if parent.is_relative_to(root))
    for path in sorted(paths):
        try:
            value = path.lstat()
        except FileNotFoundError:
            continue  # The containing-directory boundary remains mandatory.
        if max(value.st_mtime_ns, value.st_ctime_ns) > completed:
            raise ValueError(f"prerequisite source changed after the completed gate: {path}")



# The complete prerequisite CLI above remains the default authority.  This
# explicit diagnostic transaction cannot be consumed as complete gate evidence.
@dataclass(frozen=True)
class FocusedPanelPrerequisiteReport:
    """Closed evidence for exact DeviceFree checks, never a complete preflight.

    The unique schema/status and absence of a certifiable ``passed`` field keep
    legacy complete-gate and published-image consumers from promoting a local
    diagnosis by ignoring newly added booleans. Completion time belongs to the
    actual selected CTest run and remains unchanged across panel reuse.
    """
    focused_build_directory: str
    focused_completed_ns: int
    focused_return_code: int
    focused_elapsed_seconds: float
    focused_tests: tuple[str, ...]
    focused_test_count: int
    registrations: tuple[dict, ...]
    artifact_sha256: dict[str, str]
    status: str
    complete: bool
    schema: str = "llaminar.focused-panel-prerequisites.v1"
    mode: str = "paired_panel_focused_diagnostic"
    diagnostic_only: bool = True
    full_preflight: bool = False
    certification_eligible: bool = False
    native_backend_regressions: str = "not_covered_by_DeviceFree_selection"


def _focused_names(names):
    """Require explicit distinct canonical test identities, never regex filters."""
    if not isinstance(names,(tuple,list)):
        raise ValueError("focused panel checks need an explicit test-name sequence")
    selected=tuple(names)
    if (not selected or any(type(name) is not str or not re.fullmatch(r"V2_[A-Za-z0-9_]+",name) for name in selected)
            or len(set(selected))!=len(selected)):
        raise ValueError("focused panel checks need distinct exact canonical test names")
    return tuple(sorted(selected))


def _focused_registrations(build, names):
    """Read only the original CTest commands/properties for the declared checks."""
    from run_production_parity_campaigns import _property_map
    selected=_focused_names(names)
    pattern="^("+"|".join(re.escape(name) for name in selected)+")$"
    result=subprocess.run(["ctest","--test-dir",str(build),"--show-only=json-v1","-R",pattern],
        check=True,capture_output=True,text=True)
    tests=json.loads(result.stdout).get("tests",[])
    if len(tests)!=len(selected) or {test.get("name") for test in tests}!=set(selected):
        raise ValueError("focused panel selection has unknown or duplicate CTest identities")
    records=[]
    for test in tests:
        props=_property_map(test);labels=props.get("LABELS",[])
        if isinstance(labels,str):labels=labels.split(";")
        timeout=props.get("TIMEOUT")
        if ("DeviceFree" not in labels or not ({"Unit","ProductionTestPreflight"}&set(labels))
                or {"Campaign","FullModel","FullDeviceInventory"}&set(labels)
                or any(props.get(key) for key in ("FIXTURES_REQUIRED","FIXTURES_SETUP","FIXTURES_CLEANUP","REQUIRED_FILES"))
                or type(timeout) not in (int,float) or not 0<timeout<=120):
            raise ValueError("focused panel selection is not an original bounded DeviceFree check: "+test["name"])
        command=test.get("command")
        if not isinstance(command,list) or not command or any(type(arg) is not str or not arg or "\0" in arg for arg in command):
            raise ValueError("focused panel check is not built/runnable: "+test["name"])
        records.append(dict(name=test["name"],command=command,properties=props))
    return tuple(sorted(records,key=lambda item:item["name"]))


def _focused_artifact_paths(build, registrations):
    """Name ordinary build boundaries and actual selected executable/script inputs."""
    paths={build/".ninja_log",build/"build.ninja",build/"CMakeCache.txt",*build.rglob("CTestTestfile.cmake")}
    if not (build/".ninja_log").is_file() or not list(build.rglob("CTestTestfile.cmake")):
        raise ValueError("focused panel checks require an ordinary source-built Ninja/CTest tree")
    for record in registrations:
        executable=Path(record["command"][0])
        if executable.is_absolute() and executable.is_file():paths.add(executable)
        # Remaining arguments may name GGUF/reference payloads. Only bounded
        # scripts/small metadata are receipt inputs; payload bytes are never hashed.
        for argument in record["command"][1:]:
            path=Path(argument)
            if path.is_absolute() and path.is_file() and (path.suffix in {".py",".sh",".cmake",".json"} or os.access(path,os.X_OK)):
                if not os.access(path,os.X_OK) and path.stat().st_size>16*1024*1024:raise ValueError("focused test metadata exceeds its bounded extent")
                paths.add(path)
    return paths


def _focused_sha(path):
    """Authenticate finite test/report/build inputs, never model/reference payloads."""
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def _focused_junit(path, names):
    """Reject missing, extra, skipped, zero-body or failed selected CTest evidence."""
    tree=ET.parse(path);cases=tree.findall(".//testcase")
    if (len(cases)!=len(names) or {case.get("name") for case in cases}!=set(names)
            or tree.findall(".//failure") or tree.findall(".//error") or tree.findall(".//skipped")):
        raise ValueError("focused panel CTest evidence is incomplete, failed or skipped")
    for case in cases:
        if case.get("status", "run") not in ("run","passed"):
            raise ValueError("focused panel CTest did not execute its original case")
        output="\n".join(node.text or "" for node in case.findall(".//system-out"))
        if re.search(r"\[\s*SKIPPED\s*\]|skipped=[1-9][0-9]*|0 tests? from 0 test suites?|Ran 0 tests",output):
            raise ValueError("focused panel check contains an embedded skip or zero-test body")


def run_focused_panel_prerequisites(build, names, output):
    """Build ordinary owning targets and run only explicitly named host checks.

    This opt-in producer never calls either complete gate target, selects a
    native/backend regression, loads a model or changes any accuracy policy.
    CTest retains its registered timeouts, CPU weights and failure contracts.
    """
    from run_production_parity_campaigns import _run_process,write_report
    build=build.expanduser().resolve(strict=True);selected=_focused_names(names)
    output=output.expanduser().resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();registrations=_focused_registrations(build,selected)
    targets={"llaminar_model_kld_canary","v2_integration_parity_qwen4exp_production_matrix"}
    for record in registrations:
        # The canonical guardian is COMMAND[0]; an owned test executable may
        # occur after its process boundary or a script driver's arguments.
        for argument in record["command"]:
            executable=Path(argument)
            if (executable.is_absolute() and executable.is_relative_to(build)
                    and executable.is_file() and os.access(executable,os.X_OK)):
                targets.add(str(executable.relative_to(build)))
    code=_run_process(["cmake","--build",str(build),"--parallel","--target",*sorted(targets)],None)
    if code:
        report=FocusedPanelPrerequisiteReport(str(build),time.time_ns(),code,time.monotonic()-started,
            selected,len(selected),registrations,{},"DIAGNOSTIC_FOCUSED_BUILD_FAILED",False)
        write_report(output/"focused-prerequisites.json",report);return code
    registrations=_focused_registrations(build,selected)
    pattern="^("+"|".join(re.escape(name) for name in selected)+")$"
    code=_run_process(["ctest","--test-dir",str(build),"--parallel",str(len(os.sched_getaffinity(0))),
        "--output-on-failure","--no-tests=error","--test-output-size-passed","10485760",
        "--test-output-size-failed","10485760","-R",pattern,"--output-log",str(output/"focused-tests.log"),
        "--output-junit",str(output/"focused-tests.xml")],None,
        {"LLAMINAR_FORCE_CPU_ONLY_STARTUP":"1","LLAMINAR_SKIP_CUDA_STARTUP":"1","LLAMINAR_SKIP_ROCM_STARTUP":"1"})
    if code==0:
        try:_focused_junit(output/"focused-tests.xml",selected)
        except (ValueError,OSError,ET.ParseError):code=1
    paths=_focused_artifact_paths(build,registrations)
    paths.update(path for path in [output/"focused-tests.log",output/"focused-tests.xml"] if path.is_file())
    report=FocusedPanelPrerequisiteReport(str(build),time.time_ns(),code,time.monotonic()-started,
        selected,len(selected),registrations,{str(path):_focused_sha(path) for path in sorted(paths)},
        "DIAGNOSTIC_FOCUSED_CHECKS_PASSED" if code==0 else "DIAGNOSTIC_FOCUSED_CHECKS_FAILED",code==0)
    write_report(output/"focused-prerequisites.json",report);return code


def validate_focused_panel_prerequisites(build, report):
    """Reuse one authentic focused receipt without rerunning its checks.

    All selected definitions/artifacts, source freshness and build boundaries
    remain exact across every prompt/cell. This never delegates to, widens or
    produces a complete preflight receipt; the strict consumer rejects it.
    """
    build=build.expanduser().resolve(strict=True);report=report.expanduser().resolve(strict=True)
    value=json.loads(report.read_text())
    if (type(value) is not dict or set(value)!={field.name for field in fields(FocusedPanelPrerequisiteReport)} or value.get("schema")!="llaminar.focused-panel-prerequisites.v1"
            or value.get("mode")!="paired_panel_focused_diagnostic"
            or value.get("status")!="DIAGNOSTIC_FOCUSED_CHECKS_PASSED" or value.get("complete") is not True
            or value.get("diagnostic_only") is not True or value.get("full_preflight") is not False
            or value.get("certification_eligible") is not False or "passed" in value
            or type(value.get("focused_return_code")) is not int or value["focused_return_code"]!=0
            or value.get("native_backend_regressions")!="not_covered_by_DeviceFree_selection"
            or value.get("focused_build_directory")!=str(build)):
        raise ValueError("not a passed explicit focused-panel diagnostic receipt")
    completed=value.get("focused_completed_ns")
    if type(completed) is not int or not 0<completed<=report.stat().st_mtime_ns:
        raise ValueError("focused panel receipt has no original completion boundary")
    elapsed=value.get("focused_elapsed_seconds")
    if type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed<0:
        raise ValueError("focused panel elapsed observation is invalid")
    names=_focused_names(value.get("focused_tests",()))
    if type(value.get("focused_test_count")) is not int or value["focused_test_count"]!=len(names):
        raise ValueError("focused panel selected count differs")
    registrations=_focused_registrations(build,names)
    if json.loads(json.dumps(registrations))!=value.get("registrations"):
        raise ValueError("focused panel CTest definitions changed")
    _validate_source_freshness_at(build,completed)
    paths=_focused_artifact_paths(build,registrations)|{report.parent/"focused-tests.log",report.parent/"focused-tests.xml"}
    pins=value.get("artifact_sha256")
    if type(pins) is not dict or set(pins)!={str(path) for path in paths}:
        raise ValueError("focused panel artifact identity is incomplete")
    for path in paths:
        if not path.is_file() or _focused_sha(path)!=pins[str(path)]:raise ValueError("focused panel artifact changed: "+str(path))
        if path.is_relative_to(build) and path.stat().st_mtime_ns>completed:
            raise ValueError("focused panel build changed after selected checks")
    _focused_junit(report.parent/"focused-tests.xml",names)
    return value


def main(argv: list[str] | None = None) -> int:
    """Execute complete gates or authenticate an explicitly selected prior receipt.

    An installed receipt authenticates the prebuilt test inventory instead of
    attempting to rebuild stripped objects. It never replaces test execution.
    An existing result directory is rejected before any build or device work,
    so a failed rerun cannot overwrite earlier prerequisite evidence.
    Local reuse rebuilds first, then rejects changed build or test inventory.
    Reuse never writes a replacement receipt or silently selects a new run.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--installed-build-receipt", type=Path,
                        help="Authenticate an immutable test runner's installed test files; still run both gates")
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--output", type=Path,
                        help="New directory for the canonical receipt, CTest logs and JUnit evidence")
    evidence.add_argument("--reuse-report", type=Path,
                         help="Rebuild prerequisites, then authenticate an unchanged complete prior receipt")
    parser.add_argument("--diagnostic-focused-panel", action="store_true",
                        help="Explicit local DeviceFree subset only; never complete prerequisites/certification")
    parser.add_argument("--focused-test", action="append", default=[],
                        help="Exact canonical DeviceFree CTest identity; only with --diagnostic-focused-panel")
    args = parser.parse_args(argv)
    if args.focused_test and not args.diagnostic_focused_panel:
        parser.error("--focused-test requires explicit --diagnostic-focused-panel")
    if args.diagnostic_focused_panel and (not args.focused_test or args.reuse_report is not None or args.installed_build_receipt is not None):
        parser.error("focused diagnostics require exact selected tests and fresh local --output")
    build = args.build_dir.expanduser().resolve(strict=True)
    installed = (args.installed_build_receipt.expanduser().resolve(strict=True)
                 if args.installed_build_receipt is not None else None)
    if args.reuse_report is not None:
        if installed is not None:
            parser.error("Receipt reuse requires a local build, not an installed runner receipt")
        # Source edits must cross Ninja's rebuild boundary before evidence can
        # be reused. A rebuild or inventory change then makes the old receipt
        # fail closed; this mode never silently runs a different gate.
        code = subprocess.run(["cmake", "--build", str(build), "--parallel", "--target",
                               "v2_unit_gate", "v2_production_test_preflight_gate"]).returncode
        if code:
            return code
        try:
            validate_source_freshness(build, args.reuse_report)
            return reuse_unchanged_production_test_preflight(build, args.reuse_report).return_code
        except ValueError as error:
            parser.error(str(error))
    if args.diagnostic_focused_panel:
        return run_focused_panel_prerequisites(build, args.focused_test, args.output)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    preflight_result = run_production_test_preflight(
        build, None, installed_build_receipt=installed, artifact_directory=output)
    code = preflight_result.return_code
    return code


if __name__ == "__main__":
    raise SystemExit(main())
