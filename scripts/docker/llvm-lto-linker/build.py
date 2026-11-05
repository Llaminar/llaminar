#!/usr/bin/env python3
"""Build/prove the pinned parallel-LTO remark repair from the selected SDK archives.

Only LTOBackend.cpp and the exact tiny linker driver are compiled. The SDK stays
immutable; every executable is linked against its existing ABI-matched archives.
Original-code controls and byte-identical device objects authenticate the repair
before a builder-owned fixed linker is published. This is compile-time evidence,
never inference admission or a native GPU correctness/performance certificate.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PARTITIONS = (2, 8)
KERNELS = frozenset(f"partition_probe_{i}" for i in range(16))

def digest(path):
    """Return exact build-artifact identity without probing any accelerator."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    """Publish one complete JSON build observation only after its operation ends."""
    Path(path).write_text(json.dumps(value, indent=2) + "\n")

def run(argv, directory, name):
    """Retain exact argv, diagnostics and actual compiler status; never retry another path."""
    dump(directory / (name + "-command.json"), argv)
    started = time.monotonic()
    with (directory / (name + ".log")).open("w") as log:
        process = subprocess.run(argv, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
    dump(directory / (name + "-result.json"), {
        "returncode": process.returncode, "seconds": time.monotonic() - started})
    if process.returncode:
        raise RuntimeError(f"{directory}/{name}.log: command exited {process.returncode}")

def load_guard(path):
    """Use the canonical spill parser for both compiler controls and final records."""
    spec = importlib.util.spec_from_file_location("llaminar_lto_spill_guard", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

def obtain_source(args, relative, expected, destination):
    """Read exact pinned upstream input from an explicit local tree or immutable URL."""
    if args.source_dir:
        raw = (args.source_dir / relative).read_bytes()
    else:
        url = f"https://raw.githubusercontent.com/ROCm/llvm-project/{args.revision}/{relative}"
        with urllib.request.urlopen(url, timeout=30) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError(f"Pinned LLVM input differs: {relative}")
    destination.write_bytes(raw)

def build_linkers(args, work):
    """Compile the repaired member and relink original/repaired controls from one SDK closure."""
    llvm = args.sdk / "lib/llvm"
    compiler, config = llvm / "bin/clang++", llvm / "bin/llvm-config"
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    if args.revision not in version:
        raise RuntimeError("Selected SDK compiler revision does not match the LLVM source pin")
    source = work / "source"
    source.mkdir()
    obtain_source(args, "llvm/lib/LTO/LTOBackend.cpp", args.backend_sha256, source / "LTOBackend.cpp")
    obtain_source(args, "lld/tools/lld/lld.cpp", args.driver_sha256, source / "lld.cpp")
    shutil.copyfile(HERE / "main.cpp", source / "main.cpp")
    patched = work / "patched-source/llvm/lib/LTO"
    patched.mkdir(parents=True)
    shutil.copyfile(source / "LTOBackend.cpp", patched / "LTOBackend.cpp")
    run(["git", "apply", "--check", str(args.patch)], work / "patched-source", "check-patch")
    run(["git", "apply", str(args.patch)], work / "patched-source", "apply-patch")
    flags = shlex.split(subprocess.check_output([str(config), "--cxxflags"], text=True))
    common = [str(compiler), *flags, "-O3", "-DNDEBUG", "-fPIC"]
    for variant in ("original", "patched"):
        (work / variant).mkdir()
        shutil.copyfile(llvm / "lib/libLLVMLTO.a", work / variant / "libLLVMLTO.a")
    jobs = [(source / "LTOBackend.cpp", work / "original/LTOBackend.cpp.o", "backend-original"),
            (patched / "LTOBackend.cpp", work / "patched/LTOBackend.cpp.o", "backend-patched"),
            (source / "lld.cpp", work / "lld.o", "driver"),
            (source / "main.cpp", work / "main.o", "entry")]
    def compile_one(job):
        src, obj, name = job
        run([*common, "-c", str(src), "-o", str(obj)], work, name)
    with ThreadPoolExecutor() as pool:
        list(pool.map(compile_one, jobs))
    libraries = shlex.split(subprocess.check_output([str(config), "--link-static", "--libs", "all"], text=True))
    systems = shlex.split(subprocess.check_output([str(config), "--link-static", "--system-libs"], text=True))
    # The SDK describes its versioned sysdep SONAMEs using -lfoo.so.N. Bind the
    # exact declared filename; ordinary -l would append a second .so suffix.
    systems = ["-l:lib" + arg[2:] if arg.startswith("-lrocm_sysdeps_") else arg for arg in systems]
    closure = {}
    for variant in ("original", "patched"):
        directory = work / variant
        run([str(llvm / "bin/llvm-ar"), "rcs", str(directory / "libLLVMLTO.a"),
             str(directory / "LTOBackend.cpp.o")], work, "archive-" + variant)
        selected = [str(directory / "libLLVMLTO.a") if lib == "-lLLVMLTO" else lib for lib in libraries]
        llvm_archives = [llvm / "lib" / ("lib" + lib[2:] + ".a") for lib in libraries if lib != "-lLLVMLTO"]
        lld_archives = [llvm / "lib" / ("lib" + name + ".a") for name in
                        ("lldCOFF", "lldELF", "lldMinGW", "lldMachO", "lldWasm", "lldCommon")]
        for archive in [*llvm_archives, *lld_archives, llvm / "lib/libLLVMLTO.a"]:
            closure[str(archive)] = digest(archive)
        run([str(compiler), "-fuse-ld=lld", "-O3", str(work / "main.o"), str(work / "lld.o"),
             "-L" + str(llvm / "lib"), "-L" + str(args.sdk / "lib/rocm_sysdeps/lib"),
             "-Wl,-rpath," + str(args.sdk / "lib/rocm_sysdeps/lib"), "-Wl,--start-group",
             *map(str, lld_archives), *selected, "-Wl,--end-group", *systems,
             "-o", str(directory / "lld")], work, "link-" + variant)
    return {"compiler_sha256": digest(compiler), "compiler_version": version,
            "sdk_static_archives": closure,
            "original_linker_sha256": digest(work / "original/lld"),
            "linker_sha256": digest(work / "patched/lld")}

def prove(args, work, guard):
    """Exercise actual two/eight-partition codegen on every declared target without launching kernels."""
    llvm = args.sdk / "lib/llvm/bin"
    base = [str(args.sdk / "bin/hipcc"), "--hip-path=" + str(args.sdk), "--rocm-path=" + str(args.sdk),
            "-O3", "-fPIC", "-fgpu-rdc", "-save-temps=obj", "-Xarch_device", "-fsave-optimization-record=yaml",
            "-Xarch_device", "-foptimization-record-passes=stack-frame-layout", "-Xarch_device",
            "-foptimization-record-file=-", "-Rpass-analysis=stack-frame-layout|kernel-resource-usage"]
    def compile_one(arch):
        directory = work / ("probe-" + arch)
        directory.mkdir()
        run([*base, "--offload-arch=" + arch, "-c", str(HERE / "PressureProbe.hip"),
             "-o", str(directory / "probe.o")], directory, "compile")
    with ThreadPoolExecutor() as pool:
        list(pool.map(compile_one, args.architectures))
    observations = []
    for arch in args.architectures:
        for partitions in PARTITIONS:
            pair = []
            for variant in ("original", "patched"):
                directory = work / f"{variant}-{arch}-p{partitions}"
                directory.mkdir()
                argv = [*base, "--offload-arch=" + arch, "-B" + str(work / variant), "-shared",
                        str(work / ("probe-" + arch) / "probe.o"), "-Xoffload-linker",
                        "--lto-partitions=" + str(partitions), "-o", str(directory / "probe.so")]
                driver = subprocess.run([*argv, "-###"], cwd=directory, capture_output=True, text=True, check=True)
                (directory / "driver-actions.log").write_text(driver.stdout + driver.stderr)
                commands = [shlex.split(line) for line in (driver.stdout + driver.stderr).splitlines()
                            if "elf64_amdgpu" in line and "-flavor" in line]
                if len(commands) != 1 or commands[0][0] != str(work / variant / "lld"):
                    raise RuntimeError("Compiler selected a different final native linker")
                if "--lto-partitions=" + str(partitions) not in commands[0]:
                    raise RuntimeError("Compiler changed declared parallel partition count")
                run(argv, directory, "link")
                def tool(argv):
                    return subprocess.check_output([str(llvm / argv[0]), *argv[1:]], text=True)
                tool(["llvm-objcopy", "--dump-section", ".hip_fatbin=" + str(directory / "bundle"),
                      str(directory / "probe.so"), str(directory / "host.so")])
                target = next(value for value in tool(["clang-offload-bundler", "--list", "--type=o",
                              "--input=" + str(directory / "bundle")]).splitlines()
                              if value.startswith(("hip-", "hipv4-")))
                code = directory / "device.co"
                tool(["clang-offload-bundler", "--unbundle", "--type=o", "--targets=" + target,
                      "--input=" + str(directory / "bundle"), "--output=" + str(code)])
                notes = json.loads(tool(["llvm-readobj", "--notes", "--elf-output-style=JSON", str(code)]))
                metadata = [note["AMDGPU Metadata"] for obj in notes for section in obj.get("NoteSections", [])
                            for note in section["NoteSection"]["Notes"] if "AMDGPU Metadata" in note]
                kernels = [kernel for block in metadata for kernel in guard.parse_metadata(block)]
                if len(metadata) != partitions or {kernel.name for kernel in kernels} != KERNELS or len(kernels) != 16:
                    raise RuntimeError("Final partition/kernel coverage differs from the pressure witness")
                files = sorted(directory.glob("*.part.*.obj.yaml"))
                asm_files = sorted(directory.glob("*.part.*.asm.yaml"))
                functions = {}
                for path in files:
                    parsed, _ = guard.allocation_evidence(path.read_text())
                    if not parsed:
                        raise RuntimeError("A witness partition lost its final allocation records")
                    for name, allocation in parsed.items():
                        if name in functions:
                            raise RuntimeError("Duplicate witness function across partition records")
                        functions[name] = allocation
                row = {"architecture": arch, "partitions": partitions, "variant": variant,
                       "metadata_partitions": len(metadata), "kernels": len(kernels),
                       "allocation_functions": len(functions), "record_files": len(files),
                       "assembly_record_files": len(asm_files), "code_sha256": digest(code),
                       "kernels_metadata": [asdict(kernel) for kernel in kernels],
                       "allocations": {name: {"instances": value.instances,
                           "slots": [asdict(slot) for slot in value.slots]} for name, value in functions.items()}}
                if variant == "original":
                    if files or functions:
                        raise RuntimeError("Original-code negative control unexpectedly captured partition allocations")
                else:
                    if len(files) != partitions or len(asm_files) != partitions or set(functions) != KERNELS:
                        raise RuntimeError("Repaired linker has incomplete partition allocations")
                    if any(value.instances != 1 for value in functions.values()):
                        raise RuntimeError("Witness records were duplicated or overwritten")
                    spill_slots = sum(slot.kind == "Spill" and slot.size > 0
                                      for value in functions.values() for slot in value.slots)
                    try:
                        guard.audit_device_allocations(functions, kernels)
                    except guard.SpillGuardError as error:
                        if not spill_slots or "missing per-function" in str(error):
                            raise
                        row["guard_rejected_actual_spills"] = True
                    else:
                        if spill_slots:
                            raise RuntimeError("Final physical spill slots escaped the canonical guard")
                        row["guard_accepted_spill_free_frames"] = True
                pair.append(row)
                observations.append(row)
            if pair[0]["code_sha256"] != pair[1]["code_sha256"]:
                raise RuntimeError("Remark repair changed final device code bytes")
    return observations

def verify(report, expected_architectures, binary=None, guard=None):
    """Authenticate complete typed compile proof without a compiler, SDK, model or GPU."""
    if report.get("schema") != 1 or report.get("certificate") is not True:
        raise RuntimeError("Missing repaired-linker build certificate")
    if report.get("architectures") != expected_architectures:
        raise RuntimeError("Repaired-linker proof target set changed")
    if binary and digest(binary) != report["toolchain"]["linker_sha256"]:
        raise RuntimeError("Repaired-linker executable differs from its compiled proof")
    rows = report.get("observations", [])
    expected = {(arch, parts, variant) for arch in expected_architectures
                for parts in PARTITIONS for variant in ("original", "patched")}
    actual = [(row["architecture"], row["partitions"], row["variant"]) for row in rows]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError("Missing or repeated repaired-linker proof observation")
    for arch in expected_architectures:
        for parts in PARTITIONS:
            original, repaired = [next(row for row in rows if row["architecture"] == arch
                                 and row["partitions"] == parts and row["variant"] == variant)
                                 for variant in ("original", "patched")]
            if original["allocation_functions"] or original["record_files"]:
                raise RuntimeError("Original-code negative control no longer rejects lost records")
            if any(repaired[key] != value for key, value in
                   (("metadata_partitions", parts), ("kernels", 16), ("allocation_functions", 16),
                    ("record_files", parts), ("assembly_record_files", parts))):
                raise RuntimeError("Incomplete repaired final partition evidence")
            if original["code_sha256"] != repaired["code_sha256"]:
                raise RuntimeError("Repaired final code bytes differ from original control")
            if original.get("allocations") or len(original.get("kernels_metadata", [])) != 16:
                raise RuntimeError("Original negative-control observations are incomplete")
            if original["kernels_metadata"] != repaired.get("kernels_metadata"):
                raise RuntimeError("Remark repair changed final kernel resource metadata")
            if set(repaired.get("allocations", {})) != KERNELS:
                raise RuntimeError("Final allocation records lost original kernels")
            if guard:
                kernels = []
                for value in repaired["kernels_metadata"]:
                    if set(value) != {"name", "sgpr", "vgpr", "private_bytes", "dynamic_stack"}:
                        raise RuntimeError("Incomplete normalized final kernel metadata")
                    if any(type(value[key]) is not int or value[key] < 0 for key in ("sgpr", "vgpr", "private_bytes")):
                        raise RuntimeError("Invalid normalized resource count")
                    if type(value["dynamic_stack"]) is not bool:
                        raise RuntimeError("Invalid normalized stack policy")
                    kernels.append(guard.KernelSpills(**value))
                if {kernel.name for kernel in kernels} != KERNELS:
                    raise RuntimeError("Repeated or missing normalized kernel identity")
                functions = {}
                for name, entry in repaired["allocations"].items():
                    if set(entry) != {"instances", "slots"} or entry["instances"] != 1:
                        raise RuntimeError("Duplicated normalized final allocation observation")
                    slots = []
                    for slot in entry["slots"]:
                        if set(slot) != {"kind", "size"} or slot["kind"] not in {
                                "Spill", "Fixed", "VariableSized", "Protector", "Variable"}:
                            raise RuntimeError("Unknown normalized final allocation slot")
                        if type(slot["size"]) is not int or slot["size"] < 0:
                            raise RuntimeError("Invalid normalized physical stack extent")
                        slots.append(guard.StackSlot(**slot))
                    functions[name] = guard.FunctionAllocation(slots, instances=1)
                spilling = any(slot.kind == "Spill" and slot.size for entry in functions.values() for slot in entry.slots)
                try:
                    guard.audit_device_allocations(functions, kernels)
                except guard.SpillGuardError:
                    if not spilling or repaired.get("guard_rejected_actual_spills") is not True:
                        raise
                else:
                    if spilling or repaired.get("guard_accepted_spill_free_frames") is not True:
                        raise RuntimeError("Final compiler witness observations do not prove its allocation outcome")

def main():
    """Build a private artifact or verify an existing compiler-owned proof; no runtime mode exists."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    build = commands.add_parser("build")
    for name in ("sdk", "prefix", "patch", "spill-guard"):
        build.add_argument("--" + name, type=Path, required=True)
    for name in ("revision", "backend-sha256", "driver-sha256", "architectures"):
        build.add_argument("--" + name, required=True)
    build.add_argument("--work-dir", type=Path)
    build.add_argument("--source-dir", type=Path)
    check = commands.add_parser("verify")
    check.add_argument("--report", type=Path, required=True)
    check.add_argument("--architectures", required=True)
    check.add_argument("--binary", type=Path)
    check.add_argument("--spill-guard", type=Path, required=True)
    args = parser.parse_args()
    args.architectures = args.architectures.split(";")
    if not args.architectures or len(set(args.architectures)) != len(args.architectures):
        raise RuntimeError("Compiler proof requires distinct declared architecture targets")
    if args.action == "verify":
        verify(json.loads(args.report.read_text()), args.architectures, args.binary, load_guard(args.spill_guard))
        return
    temporary = None
    if args.work_dir:
        work = args.work_dir.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="llaminar-llvm-lto-")
        work = Path(temporary.name)
    toolchain = build_linkers(args, work)
    report = {"schema": 1, "certificate": True, "architectures": args.architectures,
              "llvm_revision": args.revision, "llvm_patch_sha256": digest(args.patch),
              "backend_source_sha256": args.backend_sha256, "driver_source_sha256": args.driver_sha256,
              "toolchain": toolchain, "observations": prove(args, work, load_guard(args.spill_guard)),
              "source_inputs": {path.name: digest(path) for path in
                  (Path(__file__), HERE / "main.cpp", HERE / "PressureProbe.hip", args.spill_guard)},
              "native_runtime_qualified": False}
    verify(report, args.architectures, work / "patched/lld", load_guard(args.spill_guard))
    binary = args.prefix / "bin/lld"
    share = args.prefix / "share/llaminar-llvm-lto"
    binary.parent.mkdir(parents=True, exist_ok=True)
    share.mkdir(parents=True, exist_ok=True)
    if not binary.exists() or digest(binary) != digest(work / "patched/lld"):
        shutil.copyfile(work / "patched/lld", binary.with_suffix(".new"))
        binary.with_suffix(".new").chmod(0o755)
        binary.with_suffix(".new").replace(binary)
    # Preserve mtimes only for exact, freshly reproven bytes. LINK_DEPENDS then
    # invalidates the collective on a real toolchain/proof change while a repeated
    # identical installation does not trigger another full device LTO.
    for path, value in [(share / "proof.json", report), (share / "receipt.json", {
            "schema": 1, "llvm_revision": args.revision, "patch_sha256": digest(args.patch),
            "linker_sha256": digest(binary),
            "proof_sha256": hashlib.sha256((json.dumps(report, indent=2) + "\n").encode()).hexdigest(),
            "architectures": args.architectures})]:
        raw = (json.dumps(value, indent=2) + "\n").encode()
        if not path.exists() or path.read_bytes() != raw:
            temporary_path = path.with_suffix(".new")
            temporary_path.write_bytes(raw)
            temporary_path.replace(path)
    if temporary:
        temporary.cleanup()

if __name__ == "__main__":
    main()
