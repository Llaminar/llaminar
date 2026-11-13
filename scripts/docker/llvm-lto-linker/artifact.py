#!/usr/bin/env python3
"""Configure and certify the exact RCCL final-link artifact using compiler-owned records.

RCCL is an external dependency: complete allocation evidence reports its physical
spills without claiming zero-spill, native-runtime or performance qualification.
The selected output path identifies the collective DSO. Host dependencies retain
ordinary compiler behavior. Only the RCCL DSO must authenticate its linker,
architecture set and per-partition object-record paths. Final allocation proofs
use the existing canonical YAML/metadata parser and remain build/test artifacts;
they do not introduce model hashing, execution admission or runtime device probes.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
from dataclasses import asdict
from enum import Enum
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PARTITIONS = 8

class ArtifactRole(Enum):
    """Output authority, independent of a generic shared-library compiler flag."""
    COLLECTIVE = "collective_dso"
    HOST = "host_dependency_or_utility"

def digest(path):
    """Bind only compiler/dependency artifacts, never runtime model/request state."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    """Write one complete build/test certificate with explicit schema."""
    Path(path).write_text(json.dumps(value, indent=2) + "\n")

def module(name, path):
    """Load a code-owned proof/parser module without duplicating its arithmetic policy."""
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value

def pins(path):
    """Read the canonical simple assignment document without executing shell input."""
    values = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise RuntimeError("Malformed canonical ROCm pin assignment")
        value = shlex.split(raw, comments=True)
        if len(value) != 1 or name in values:
            raise RuntimeError("Repeated or non-scalar canonical ROCm pin")
        values[name] = value[0]
    return values

def load_guard(path):
    """Keep LLVM YAML parsing and physical-spill classification single-source."""
    return module("llaminar_rccl_final_guard", path)

def compiler_artifact(prefix, pin_file, guard_path):
    """Authenticate the builder's exact fixed executable and complete original-code proof."""
    prefix = Path(prefix)
    receipt_path = prefix / "share/llaminar-llvm-lto/receipt.json"
    proof_path = prefix / "share/llaminar-llvm-lto/proof.json"
    receipt, proof = json.loads(receipt_path.read_text()), json.loads(proof_path.read_text())
    canonical = pins(pin_file)
    patch = HERE.parent / "patches/llvm-parallel-lto-remarks.patch"
    if receipt.get("schema") != 1 or receipt.get("proof_sha256") != digest(proof_path):
        raise RuntimeError("Fixed-linker proof receipt is missing or stale")
    gvn_patch = HERE.parent / "patches/llvm-gvn-filtered-remarks.patch"
    if digest(gvn_patch) != canonical["LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256"]:
        raise RuntimeError("GVN diagnostic repair differs from its canonical patch pin")
    expected = {"llvm_revision": canonical["LLAMINAR_ROCM_LLVM_REVISION"],
                "llvm_patch_sha256": digest(patch),
                "backend_source_sha256": canonical["LLAMINAR_ROCM_LLVM_LTO_BACKEND_SHA256"],
                "driver_source_sha256": canonical["LLAMINAR_ROCM_LLVM_LLD_DRIVER_SHA256"],
                "gvn_source_sha256": canonical["LLAMINAR_ROCM_LLVM_GVN_SOURCE_SHA256"],
                "gvn_patch_sha256": canonical["LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256"]}
    if any(proof.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Fixed-linker artifact differs from the canonical source/patch pin")
    expected_sources = {path.name: digest(path) for path in
                        (HERE / "build.py", HERE / "main.cpp", HERE / "PressureProbe.hip", HERE / "GVNRemarkProbe.cpp", Path(guard_path))}
    if proof.get("source_inputs") != expected_sources:
        raise RuntimeError("Compiler-owned proof inputs changed; rebuild its artifact")
    builder = module("llaminar_pinned_lto_builder", HERE / "build.py")
    targets = receipt.get("architectures")
    if not isinstance(targets, list) or len(set(targets)) != len(targets) or \
       not set(canonical["LLAMINAR_ROCM_GPU_TARGETS"].split(";")).issubset(targets):
        raise RuntimeError("Compiler artifact does not prove the complete shipping target set")
    builder.verify(proof, targets, prefix / "bin/ld.lld", load_guard(guard_path))
    if receipt.get("linker_sha256") != digest(prefix / "bin/ld.lld"):
        raise RuntimeError("Fixed-linker executable changed after its builder receipt")
    return receipt_path, receipt

def classify_output(argv, expected, cwd):
    """Recognize the exact selected RCCL output; similarly named foreign outputs reject."""
    outputs = [argv[index + 1] for index, value in enumerate(argv[:-1]) if value == "-o"]
    if len(outputs) != 1:
        raise RuntimeError("Link invocation requires exactly one original output path")
    actual = Path(outputs[0])
    actual = (actual if actual.is_absolute() else Path(cwd) / actual).resolve()
    expected = Path(expected).resolve()
    if actual == expected:
        if "-shared" not in argv:
            raise RuntimeError("Selected RCCL output lost its shared-library role")
        return ArtifactRole.COLLECTIVE
    if actual.name.startswith("librccl.so"):
        raise RuntimeError("Link selected a foreign RCCL output path")
    return ArtifactRole.HOST

def authenticate_driver(spec, argv, cwd):
    """Authenticate actual native driver actions before any collective final-codegen work."""
    if digest(spec["linker"]) != spec["linker_sha256"] or digest(spec["linker_receipt"]) != spec["linker_receipt_sha256"]:
        raise RuntimeError("Configured fixed-linker artifact changed")
    role = classify_output(argv, spec["library"], cwd)
    if role is ArtifactRole.HOST:
        return {"role": role.value, "native_commands": []}
    builder = module("llaminar_rccl_native_driver", HERE / "build.py")
    observation = builder.authenticate_wrapper(argv, cwd, spec["wrapper"], spec["linker"],
        spec["architectures"], PARTITIONS, spec["library"])
    return {"role": role.value, **observation}

def configure(args):
    """Write nested-RCCL-only flags/spec; parent inference compiler settings remain untouched."""
    canonical = pins(args.pins)
    prefix = args.linker_prefix or Path(canonical["LLAMINAR_ROCM_LTO_LINKER_PREFIX"])
    receipt_path, receipt = compiler_artifact(prefix, args.pins, args.spill_guard)
    targets = args.architectures.split(";")
    if len(set(targets)) != len(targets) or not targets or not set(targets).issubset(receipt["architectures"]):
        raise RuntimeError("RCCL requested an unproved or repeated compiler target")
    library = args.library.resolve()
    build = library.parent
    records = build / "llaminar-final-allocation"
    records.mkdir(parents=True, exist_ok=True)
    spec = {"schema": 1, "library": str(library), "architectures": targets,
            "record_root": str(records), "linker": str(prefix / "bin/ld.lld"),
            "wrapper": str(args.sdk / "lib/llvm/bin/clang-linker-wrapper"),
            "linker_sha256": receipt["linker_sha256"], "linker_receipt": str(receipt_path),
            "linker_receipt_sha256": digest(receipt_path), "spill_guard": str(args.spill_guard.resolve())}
    spec_path = records / "configuration.json"
    dump(spec_path, spec)
    compile_flags = ["--offload-new-driver", "--hip-path=" + str(args.sdk), "--rocm-path=" + str(args.sdk),
                     "-Xarch_device", "-fsave-optimization-record=yaml", "-Xarch_device",
                     "-foptimization-record-passes=stack-frame-layout", "-Xarch_device",
                     "-Rpass-analysis=stack-frame-layout"]
    # The new wrapper forwards only explicitly named offload-compiler options.
    # Its own dumpdir includes each target; a fixed shared remark filename would
    # race/overwrite targets, and -Xarch_<gfx> is not forwarded at this layer.
    link_flags = ["--offload-new-driver", "-Xoffload-compiler", "--ld-path=" + str(prefix / "bin/ld.lld"),
                  "-Rpass-analysis=stack-frame-layout|kernel-resource-usage",
                  "SHELL:-Xoffload-linker --lto-partitions=" + str(PARTITIONS)]
    # RCCL's own -Xoffload-linker operand controls kernarg preloading. Keep this
    # second option/operand pair grouped so CMake's option deduplication cannot
    # orphan the partition count or send it to the host linker.
    launcher = ";".join((sys.executable, str(Path(__file__).resolve()), "gate", "--spec", str(spec_path), "--"))
    target_hook = records / "target-link-dependencies.cmake"
    # The upstream rccl target appears after project(). Bind its native flags,
    # exact output gate and linker dependencies at the end of that directory.
    # fmt and host utilities receive none of these native link settings.
    literal = lambda value: "[====[" + value + "]====]"
    native_options = " ".join(literal(value) for value in link_flags)
    target_hook.write_text(
        '# Generated dependency-owned native final-link lifetime.\n'
        'if(PROJECT_NAME STREQUAL "rccl")\n'
        '  cmake_language(DEFER CALL set_property TARGET rccl APPEND PROPERTY LINK_DEPENDS '
        + literal(spec["linker"]) + ' ' + literal(str(receipt_path)) + ' ' + literal(str(Path(__file__).resolve())) + ')\n'
        '  cmake_language(DEFER CALL set_property TARGET rccl APPEND PROPERTY LINK_OPTIONS '
        + native_options + ')\n'
        '  cmake_language(DEFER CALL set_property TARGET rccl PROPERTY CXX_LINKER_LAUNCHER '
        + literal(launcher) + ')\n'
        'endif()\n')
    cache = {"CMAKE_CXX_FLAGS": shlex.join(compile_flags), "CMAKE_SHARED_LINKER_FLAGS": "",
             "CMAKE_CXX_COMPILER_LAUNCHER": "ccache", "CMAKE_CXX_LINKER_LAUNCHER": "",
             "CMAKE_PROJECT_INCLUDE": str(target_hook)}
    text = "# Generated nested RCCL final-link contract; never included in the parent project.\n"
    for name, value in cache.items():
        if "]====]" in value:
            raise RuntimeError("Unrepresentable CMake contract value")
        text += f"set({name} [====[{value}]====] CACHE STRING \"RCCL final allocation contract\" FORCE)\n"
    text += 'set(REPORT_KERNEL_RESOURCE_USE OFF CACHE BOOL "Combined final diagnostic policy is explicit" FORCE)\n'
    args.write_cache.write_text(text)


def normalize_functions(functions):
    """Preserve every final physical slot and observation; no deduplication or estimates."""
    return {name: {"instances": entry.instances, "slots": [asdict(slot) for slot in entry.slots]}
            for name, entry in functions.items()}


def resource_usage(functions, kernels):
    """Report external RCCL physical spills while requiring complete kernel allocation coverage.

    This dependency evidence has no authority to relax the independent absolute
    zero-spill guard on project kernel targets or certify native/performance gates.
    Slot extents are aggregate compiler observations, not simultaneous live bytes.
    """
    for name, required in Counter(kernel.name for kernel in kernels).items():
        if name not in functions or functions[name].instances < required:
            raise RuntimeError("Missing final RCCL per-function allocation evidence: " + name)
    spilled = []
    for name, entry in sorted(functions.items()):
        slots = [slot for slot in entry.slots if slot.kind == "Spill" and slot.size]
        if slots:
            spilled.append({"function": name, "slots": len(slots), "bytes": sum(slot.size for slot in slots)})
    return {"physical_spills": {"functions": len(spilled), "slots": sum(row["slots"] for row in spilled),
                "bytes": sum(row["bytes"] for row in spilled), "by_function": spilled},
            "register_moves": [{"function": kernel.name, "sgpr": kernel.sgpr, "vgpr": kernel.vgpr}
                               for kernel in kernels if kernel.sgpr or kernel.vgpr]}


@contextmanager
def record_transaction(spec):
    """Exclude overlapping links/proof reads for this target-owned record directory."""
    root = Path(spec["record_root"])
    with (root / "link-records.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Concurrent RCCL allocation-record transaction") from error
        yield lock.fileno()


def object_records(spec, arch):
    """Require every current object partition; absent output must never borrow old records."""
    builder = module("llaminar_rccl_record_paths", HERE / "build.py")
    base = builder.final_record_base(spec["library"], arch)
    files = [Path(str(base) + f".part.{index}.obj.yaml") for index in range(PARTITIONS)]
    if not all(path.is_file() for path in files):
        raise RuntimeError("RCCL final object partition allocation record is missing")
    return files


def withdraw_records(spec):
    """Withdraw only this authenticated target's prior object/assembly allocation files."""
    builder = module("llaminar_rccl_record_paths", HERE / "build.py")
    for arch in spec["architectures"]:
        base = builder.final_record_base(spec["library"], arch)
        base.unlink(missing_ok=True)
        for index in range(PARTITIONS):
            for kind in ("obj", "asm"):
                Path(str(base) + f".part.{index}.{kind}.yaml").unlink(missing_ok=True)


def compile_proof(args):
    """Keep final code/record extraction exclusive with respect to subsequent relinking."""
    spec = json.loads(args.spec.read_text())
    with record_transaction(spec):
        compile_proof_locked(args)


def compile_proof_locked(args):
    """Audit all declared final code targets using only their actual object-backend YAML files."""
    spec = json.loads(args.spec.read_text())
    guard = load_guard(spec["spill_guard"])
    library = Path(spec["library"])
    llvm = args.sdk / "lib/llvm/bin"
    work = args.report.parent / (args.report.stem + "-objects")
    work.mkdir(parents=True, exist_ok=True)
    def tool(argv):
        return subprocess.check_output([str(llvm / argv[0]), *argv[1:]], text=True)
    tool(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + str(work / "bundle"), str(library), str(work / "host.so")])
    targets = [value for value in tool(["clang-offload-bundler", "--list", "--type=o",
                "--input=" + str(work / "bundle")]).splitlines() if value.startswith(("hip-", "hipv4-"))]
    observations = []
    for arch in spec["architectures"]:
        matching = [value for value in targets if value.endswith(arch)]
        if len(matching) != 1:
            raise RuntimeError("RCCL final code lost or repeated its declared native target")
        code = work / (arch + ".co")
        tool(["clang-offload-bundler", "--unbundle", "--type=o", "--targets=" + matching[0],
              "--input=" + str(work / "bundle"), "--output=" + str(code)])
        notes = json.loads(tool(["llvm-readobj", "--notes", "--elf-output-style=JSON", str(code)]))
        metadata = [note["AMDGPU Metadata"] for obj in notes for section in obj.get("NoteSections", [])
                    for note in section["NoteSection"]["Notes"] if "AMDGPU Metadata" in note]
        kernels = [kernel for block in metadata for kernel in guard.parse_metadata(block)]
        if len(metadata) != PARTITIONS or len({kernel.name for kernel in kernels}) != len(kernels) or not kernels:
            raise RuntimeError("RCCL final partition metadata is incomplete or repeats a kernel")
        files = object_records(spec, arch)
        for path in files:
            partition_functions, _ = guard.allocation_evidence(path.read_text())
            if not partition_functions:
                raise RuntimeError("Empty final RCCL object-partition allocation evidence")
        functions, _ = guard.allocation_evidence("\n".join(path.read_text() for path in files))
        usage = resource_usage(functions, kernels)
        observations.append({"architecture": arch, "partitions": PARTITIONS, "code_sha256": digest(code),
            "kernels": [asdict(kernel) for kernel in kernels], "allocations": normalize_functions(functions),
            "record_files": [{"sha256": digest(path), "bytes": path.stat().st_size} for path in files],
            "resource_usage": usage})
    receipt_path = Path(spec["linker_receipt"])
    if digest(receipt_path) != spec["linker_receipt_sha256"]:
        raise RuntimeError("Compiler artifact changed during final RCCL code generation")
    compiler_receipt = json.loads(receipt_path.read_text())
    compiler_proof_path = receipt_path.with_name("proof.json")
    if digest(compiler_proof_path) != compiler_receipt["proof_sha256"]:
        raise RuntimeError("Final compiler proof changed during RCCL linking")
    report = {"schema": 2, "scope": "external-rccl-resource-evidence",
              "resource_evidence_complete": True, "zero_spill_qualified": False,
              "performance_qualified": False, "producer_sha256": digest(Path(__file__)), "library_sha256": digest(library),
              "receipt_sha256": digest(args.receipt), "linker_sha256": spec["linker_sha256"],
              "linker_receipt_sha256": spec["linker_receipt_sha256"], "architectures": spec["architectures"],
              "observations": observations,
              "compiler_artifact": {"receipt_json": receipt_path.read_text(),
                                    "proof_json": compiler_proof_path.read_text()},
              "native_runtime_qualified": False}
    dump(args.report, report)


def verify_proof(args):
    """Revalidate a compact typed dependency proof without SDK tools or accelerator access."""
    report = json.loads(args.report.read_text())
    if report.get("schema") != 2 or report.get("scope") != "external-rccl-resource-evidence" or \
       report.get("resource_evidence_complete") is not True:
        raise RuntimeError("Missing complete final RCCL resource evidence")
    if any(report.get(key) is not False for key in
           ("zero_spill_qualified", "native_runtime_qualified", "performance_qualified")):
        raise RuntimeError("RCCL resource evidence cannot claim separate qualification")
    if report.get("library_sha256") != digest(args.library) or report.get("receipt_sha256") != digest(args.receipt):
        raise RuntimeError("RCCL allocation proof does not name the exact selected artifact/receipt")
    if report.get("producer_sha256") != digest(Path(__file__)):
        raise RuntimeError("RCCL allocation proof producer changed; current final-link proof required")
    expected = args.architectures.split(";")
    rows = report.get("observations", [])
    if report.get("architectures") != expected or [row.get("architecture") for row in rows] != expected:
        raise RuntimeError("Final RCCL allocation proof target coverage changed")
    compiler = report.get("compiler_artifact")
    if not isinstance(compiler, dict) or set(compiler) != {"receipt_json", "proof_json"}:
        raise RuntimeError("Final RCCL proof lost its compiler artifact observations")
    compiler_receipt = json.loads(compiler["receipt_json"])
    compiler_proof = json.loads(compiler["proof_json"])
    if hashlib.sha256(compiler["receipt_json"].encode()).hexdigest() != report["linker_receipt_sha256"]:
        raise RuntimeError("Final RCCL proof replaced its original compiler receipt")
    if hashlib.sha256(compiler["proof_json"].encode()).hexdigest() != compiler_receipt["proof_sha256"]:
        raise RuntimeError("Final RCCL proof replaced its original compiler observations")
    canonical = pins(HERE.parent / "rocm-release.env")
    if compiler_receipt.get("schema") != 1 or compiler_receipt["linker_sha256"] != report["linker_sha256"]:
        raise RuntimeError("Final RCCL proof names a foreign native linker")
    if compiler_proof.get("toolchain", {}).get("linker_sha256") != compiler_receipt["linker_sha256"]:
        raise RuntimeError("Final compiler observations name a different linker executable")
    expected_sources = {path.name: digest(path) for path in
                        (HERE / "build.py", HERE / "main.cpp", HERE / "PressureProbe.hip", HERE / "GVNRemarkProbe.cpp", Path(args.spill_guard))}
    if compiler_proof.get("source_inputs") != expected_sources:
        raise RuntimeError("Final compiler contract proof is stale for its code-owned inputs")
    if compiler_proof.get("llvm_revision") != canonical["LLAMINAR_ROCM_LLVM_REVISION"] or \
       compiler_proof.get("llvm_patch_sha256") != digest(HERE.parent / "patches/llvm-parallel-lto-remarks.patch"):
        raise RuntimeError("Final compiler proof has stale source/patch pins")
    if compiler_proof.get("backend_source_sha256") != canonical["LLAMINAR_ROCM_LLVM_LTO_BACKEND_SHA256"] or \
       compiler_proof.get("driver_source_sha256") != canonical["LLAMINAR_ROCM_LLVM_LLD_DRIVER_SHA256"]:
        raise RuntimeError("Final compiler proof changed pinned upstream translation units")
    if (compiler_proof.get("gvn_source_sha256") != canonical["LLAMINAR_ROCM_LLVM_GVN_SOURCE_SHA256"]
            or compiler_proof.get("gvn_patch_sha256") != canonical["LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256"]
            or digest(HERE.parent / "patches/llvm-gvn-filtered-remarks.patch")
               != canonical["LLAMINAR_ROCM_LLVM_GVN_PATCH_SHA256"]):
        raise RuntimeError("Final compiler proof changed pinned GVN diagnostic repair")
    builder = module("llaminar_final_compiler_proof", HERE / "build.py")
    guard = load_guard(args.spill_guard)
    compiler_targets = compiler_receipt.get("architectures")
    if not isinstance(compiler_targets, list) or len(set(compiler_targets)) != len(compiler_targets) or \
       not set(canonical["LLAMINAR_ROCM_GPU_TARGETS"].split(";")).issubset(compiler_targets):
        raise RuntimeError("Compiler proof lost a shipping target observation")
    if not set(expected).issubset(compiler_targets):
        raise RuntimeError("Selected collective target has no compiler proof")
    builder.verify(compiler_proof, compiler_targets, guard=guard)
    for row in rows:
        if row.get("partitions") != PARTITIONS or len(row.get("record_files", [])) != PARTITIONS:
            raise RuntimeError("Final RCCL allocation proof lost a backend partition")
        kernels = []
        for value in row["kernels"]:
            if set(value) != {"name", "sgpr", "vgpr", "private_bytes", "dynamic_stack"}:
                raise RuntimeError("Incomplete final kernel metadata in RCCL proof")
            if not isinstance(value["name"], str) or not value["name"]:
                raise RuntimeError("Invalid final kernel name in RCCL proof")
            if any(type(value[key]) is not int or value[key] < 0 for key in ("sgpr", "vgpr", "private_bytes")):
                raise RuntimeError("Invalid final kernel resource count in RCCL proof")
            if type(value["dynamic_stack"]) is not bool:
                raise RuntimeError("Invalid final kernel stack policy in RCCL proof")
            kernels.append(guard.KernelSpills(**value))
        for record in row["record_files"]:
            if set(record) != {"sha256", "bytes"} or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
                raise RuntimeError("Missing final object-record identity in RCCL proof")
            if type(record["bytes"]) is not int or record["bytes"] < 0:
                raise RuntimeError("Invalid final object-record extent in RCCL proof")
        if not kernels or len({kernel.name for kernel in kernels}) != len(kernels):
            raise RuntimeError("Final RCCL allocation proof repeats or omits kernel metadata")
        functions = {}
        for name, value in row["allocations"].items():
            if set(value) != {"slots", "instances"}:
                raise RuntimeError("Incomplete physical allocation observation in RCCL proof")
            if type(value["instances"]) is not int or value["instances"] < 1:
                raise RuntimeError("Invalid final allocation observation count")
            slots = []
            for slot in value["slots"]:
                if set(slot) != {"kind", "size"}:
                    raise RuntimeError("Incomplete physical stack slot in RCCL proof")
                if slot["kind"] not in {"Spill", "Fixed", "VariableSized", "Protector", "Variable"}:
                    raise RuntimeError("Unknown physical stack slot type in RCCL proof")
                if type(slot["size"]) is not int or slot["size"] < 0:
                    raise RuntimeError("Invalid physical stack slot extent in RCCL proof")
                slots.append(guard.StackSlot(**slot))
            functions[name] = guard.FunctionAllocation(slots, value["instances"])
        if json.dumps(row.get("resource_usage"), sort_keys=True) != \
           json.dumps(resource_usage(functions, kernels), sort_keys=True):
            raise RuntimeError("Final RCCL resource usage is incomplete or inconsistent")


def main():
    """Execute one explicit builder/test action; no installation or inference action is exposed."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    config = commands.add_parser("configure")
    for name in ("sdk", "pins", "spill-guard", "library", "write-cache"):
        config.add_argument("--" + name, type=Path, required=True)
    config.add_argument("--linker-prefix", type=Path)
    config.add_argument("--architectures", required=True)
    gate = commands.add_parser("gate")
    gate.add_argument("--spec", type=Path, required=True)
    gate.add_argument("argv", nargs=argparse.REMAINDER)
    compile_action = commands.add_parser("compile")
    for name in ("sdk", "spec", "report", "receipt"):
        compile_action.add_argument("--" + name, type=Path, required=True)
    verify_action = commands.add_parser("verify")
    for name in ("report", "library", "receipt", "spill-guard"):
        verify_action.add_argument("--" + name, type=Path, required=True)
    verify_action.add_argument("--architectures", required=True)
    args = parser.parse_args()
    if args.action == "configure":
        configure(args)
    elif args.action == "gate":
        argv = args.argv[1:] if args.argv and args.argv[0] == "--" else args.argv
        spec = json.loads(args.spec.read_text())
        if classify_output(argv, spec["library"], Path.cwd()) is not ArtifactRole.COLLECTIVE:
            raise RuntimeError("Configured collective target selected a non-collective output")
        with record_transaction(spec) as lock_fd:
            # Wrapper authentication retains extracted inputs alongside this DSO;
            # keep it inside the original exclusive final-link lifetime.
            observation = authenticate_driver(spec, argv, Path.cwd())
            withdraw_records(spec)
            dump(Path(spec["record_root"]) / "link-admission.json", observation)
            # Retain the lock until the actual driver and all backend children terminate.
            raise SystemExit(subprocess.run(argv, pass_fds=(lock_fd,)).returncode)
    elif args.action == "compile":
        compile_proof(args)
    else:
        verify_proof(args)

if __name__ == "__main__":
    main()
