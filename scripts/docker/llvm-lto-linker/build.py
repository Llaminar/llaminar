#!/usr/bin/env python3
"""Build/prove the pinned parallel-LTO remark repair from the selected SDK archives.

Only LTOBackend.cpp/GVN.cpp, the exact driver and its host diagnostic witness are compiled. The SDK stays
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
import stat
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

def verify_gvn_filter_proof(proof):
    """Verify actual original/repaired guard observations and unchanged optimization.

    :param proof: Built host witness observations bound to exact GVN/archive/executable bytes.
    :return: None after all selected/excluded diagnostic paths and IR bytes agree.
    :raises RuntimeError: Missing/typed-invalid evidence, altered decisions or lost requested remarks.
    """
    expected = [("filtered", True, "stack-frame-layout", False),
                ("gvn_selected", True, "gvn", False), ("unfiltered", True, "", False),
                ("terminal_override", True, "stack-frame-layout", True),
                ("terminal_only", False, "", True), ("disabled", False, "", False)]
    if not isinstance(proof, dict) or proof.get("schema") != 1:
        raise RuntimeError("Missing actual GVN diagnostic guard proof")
    variants = proof.get("variants")
    if (not isinstance(variants, list) or len(variants) != 2
            or not all(isinstance(row, dict) for row in variants)
            or [row.get("variant") for row in variants] != ["original", "patched"]):
        raise RuntimeError("Missing/repeated GVN diagnostic control variant")
    observations = []
    for variant in variants:
        for field in ("gvn_object_sha256", "scalar_archive_sha256", "executable_sha256"):
            value = variant.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise RuntimeError("Unbound actual GVN diagnostic witness")
        data = variant.get("observation", {})
        if not isinstance(data, dict):
            raise RuntimeError("Malformed actual GVN selector observation")
        cases = data.get("cases")
        if data.get("schema") != 1 or not isinstance(cases, list) or len(cases) != len(expected):
            raise RuntimeError("Incomplete actual GVN selector matrix")
        for row, (name, stream, selection, terminal) in zip(cases, expected):
            if (not isinstance(row, dict) or row.get("name") != name or type(row.get("stream_enabled")) is not bool
                    or row["stream_enabled"] != stream or row.get("filter") != selection
                    or type(row.get("terminal_gvn")) is not bool or row["terminal_gvn"] != terminal
                    or type(row.get("constructed")) is not int or row["constructed"] < 0
                    or not isinstance(row.get("messages"), list)
                    or not all(isinstance(message, str) and message for message in row["messages"])
                    or row["constructed"] != len(row["messages"])
                    or not isinstance(row.get("serialized"), str)
                    or not isinstance(row.get("optimized_ir"), str) or not row["optimized_ir"]):
                raise RuntimeError("Malformed actual GVN guard observation")
        observations.append(cases)
    if variants[0]["gvn_object_sha256"] == variants[1]["gvn_object_sha256"]:
        raise RuntimeError("GVN original/repaired observations borrowed the same object")
    baseline, repaired = observations
    canonical_ir = baseline[0]["optimized_ir"]
    if any(row["optimized_ir"] != canonical_ir for cases in observations for row in cases):
        raise RuntimeError("Diagnostic-only GVN repair changed optimized IR bytes")
    if baseline[0]["constructed"] <= 0 or repaired[0]["constructed"] != 0:
        raise RuntimeError("GVN filtered diagnostic original-positive/repaired-negative control failed")
    for cases in observations:
        for index in (0, 3, 4, 5):
            if "LoadClobbered" in cases[index]["serialized"]:
                raise RuntimeError("Excluded GVN serialization bypassed the selected filter")
        if cases[5]["constructed"] != 0:
            raise RuntimeError("Disabled GVN diagnostics performed extra report work")
    for index in (1, 2, 3, 4):
        left, right = baseline[index], repaired[index]
        if (left["constructed"] <= 0 or left["messages"] != right["messages"]
                or left["serialized"] != right["serialized"]):
            raise RuntimeError("GVN repair lost/changed explicitly requested diagnostic output")
        if index in (1, 2) and "LoadClobbered" not in right["serialized"]:
            raise RuntimeError("Selected GVN serialized remark is missing")

def build_linkers(args, work):
    """Replace only exact pinned archive members and prove their actual diagnostics/code.

    :param args: Canonical SDK/source/patch/output build inputs from the existing CLI owner.
    :param work: Exclusive fresh original builder work directory.
    :return: Actual compiler/archive/linker identities plus executable GVN guard observations.
    """
    llvm = args.sdk / "lib/llvm"
    compiler, config = llvm / "bin/clang++", llvm / "bin/llvm-config"
    version = subprocess.check_output([str(compiler), "--version"], text=True)
    if args.revision not in version:
        raise RuntimeError("Selected SDK compiler revision does not match the LLVM source pin")
    if digest(args.gvn_patch) != args.gvn_patch_sha256:
        raise RuntimeError("Pinned GVN diagnostic patch differs")
    source = work / "source"
    source.mkdir()
    obtain_source(args, "llvm/lib/LTO/LTOBackend.cpp", args.backend_sha256, source / "LTOBackend.cpp")
    obtain_source(args, "lld/tools/lld/lld.cpp", args.driver_sha256, source / "lld.cpp")
    obtain_source(args, "llvm/lib/Transforms/Scalar/GVN.cpp", args.gvn_sha256, source / "GVN.cpp")
    for name in ("main.cpp", "GVNRemarkProbe.cpp"):
        shutil.copyfile(HERE / name, source / name)
    patched = work / "patched-source"
    for relative, name in (("llvm/lib/LTO/LTOBackend.cpp", "LTOBackend.cpp"),
                           ("llvm/lib/Transforms/Scalar/GVN.cpp", "GVN.cpp")):
        destination = patched / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)
    for name, patch in (("lto", args.patch), ("gvn", args.gvn_patch)):
        run(["git", "apply", "--check", str(patch)], patched, "check-" + name + "-patch")
        run(["git", "apply", str(patch)], patched, "apply-" + name + "-patch")
    flags = shlex.split(subprocess.check_output([str(config), "--cxxflags"], text=True))
    common = [str(compiler), *flags, "-O3", "-DNDEBUG", "-fPIC"]
    for variant in ("original", "patched"):
        (work / variant).mkdir()
        for name in ("LLVMLTO", "LLVMScalarOpts"):
            shutil.copyfile(llvm / ("lib/lib" + name + ".a"), work / variant / ("lib" + name + ".a"))
    jobs = [(source / "LTOBackend.cpp", work / "original/LTOBackend.cpp.o", "backend-original"),
            (patched / "llvm/lib/LTO/LTOBackend.cpp", work / "patched/LTOBackend.cpp.o", "backend-patched"),
            (source / "GVN.cpp", work / "original/GVN.cpp.o", "gvn-original"),
            (patched / "llvm/lib/Transforms/Scalar/GVN.cpp", work / "patched/GVN.cpp.o", "gvn-patched"),
            (source / "lld.cpp", work / "lld.o", "driver"),
            (source / "main.cpp", work / "main.o", "entry"),
            (source / "GVNRemarkProbe.cpp", work / "GVNRemarkProbe.o", "gvn-witness")]
    def compile_one(job):
        src, obj, name = job
        run([*common, "-c", str(src), "-o", str(obj)], work, name)
    with ThreadPoolExecutor() as pool:
        list(pool.map(compile_one, jobs))
    libraries = shlex.split(subprocess.check_output([str(config), "--link-static", "--libs", "all"], text=True))
    if "-lLLVMScalarOpts" not in libraries or "-lLLVMLTO" not in libraries:
        raise RuntimeError("SDK lost the original LLVM archive authorities")
    systems = shlex.split(subprocess.check_output([str(config), "--link-static", "--system-libs"], text=True))
    # SDK sysdep -lfoo.so.N entries name exact SONAMEs; ordinary -l would
    # append a second suffix and select a different/nonexistent library.
    systems = ["-l:lib" + arg[2:] if arg.startswith("-lrocm_sysdeps_") else arg for arg in systems]
    closure = {}
    gvn_proof = {"schema": 1, "variants": []}
    for variant in ("original", "patched"):
        directory = work / variant
        for archive, member in (("libLLVMLTO.a", "LTOBackend.cpp.o"), ("libLLVMScalarOpts.a", "GVN.cpp.o")):
            members = subprocess.check_output([str(llvm / "bin/llvm-ar"), "t", str(directory / archive)], text=True).splitlines()
            if members.count(member) != 1:
                raise RuntimeError("SDK archive lost its exact replacement member: " + member)
            run([str(llvm / "bin/llvm-ar"), "rcs", str(directory / archive), str(directory / member)],
                work, "archive-" + variant + "-" + member)
        replacements = {"-lLLVMLTO": directory / "libLLVMLTO.a", "-lLLVMScalarOpts": directory / "libLLVMScalarOpts.a"}
        selected = [str(replacements[lib]) if lib in replacements else lib for lib in libraries]
        llvm_archives = [llvm / "lib" / ("lib" + lib[2:] + ".a") for lib in libraries]
        lld_archives = [llvm / "lib" / ("lib" + name + ".a") for name in
                        ("lldCOFF", "lldELF", "lldMinGW", "lldMachO", "lldWasm", "lldCommon")]
        for archive in [*llvm_archives, *lld_archives]:
            closure[str(archive)] = digest(archive)
        link = [str(compiler), "-fuse-ld=lld", "-O3", "-L" + str(llvm / "lib"),
                "-L" + str(args.sdk / "lib/rocm_sysdeps/lib"),
                "-Wl,-rpath," + str(args.sdk / "lib/rocm_sysdeps/lib")]
        run([*link, str(work / "main.o"), str(work / "lld.o"), "-Wl,--start-group",
             *map(str, lld_archives), *selected, "-Wl,--end-group", *systems,
             "-o", str(directory / "ld.lld")], work, "link-" + variant)
        witness = directory / "gvn-remark-probe"
        run([*link, str(work / "GVNRemarkProbe.o"), "-Wl,--start-group", *selected,
             "-Wl,--end-group", *systems, "-o", str(witness)], work, "link-gvn-witness-" + variant)
        run([str(witness)], work, "run-gvn-witness-" + variant)
        gvn_proof["variants"].append({"variant": variant, "gvn_object_sha256": digest(directory / "GVN.cpp.o"),
            "scalar_archive_sha256": digest(directory / "libLLVMScalarOpts.a"), "executable_sha256": digest(witness),
            "observation": json.loads((work / ("run-gvn-witness-" + variant + ".log")).read_text())})
    verify_gvn_filter_proof(gvn_proof)
    return {"compiler_sha256": digest(compiler), "compiler_version": version,
            "sdk_static_archives": closure, "gvn_filter_proof": gvn_proof,
            "original_linker_sha256": digest(work / "original/ld.lld"),
            "linker_sha256": digest(work / "patched/ld.lld")}

def final_record_base(output, arch):
    """Return the wrapper's target-unique default final remark filename.

    :param output: Exact absolute final host output path authenticated by its owner.
    :param arch: One declared AMDGPU architecture; never a glob or discovered target.
    :return: Exact final-LTO YAML base, before the backend partition/kind suffix.
    """
    return Path(str(output) + ".amdgcn." + arch + ".img.opt.ld.yaml")

def command_lines(diagnostics):
    """Parse printed tool argv without executing shell text or guessing command order.

    :param diagnostics: Captured native driver or linker-wrapper diagnostic text.
    :return: Parsed lines; callers authenticate exact executable and target identities.
    """
    commands = []
    for line in diagnostics.splitlines():
        try:
            command = shlex.split(line)
        except ValueError:
            continue
        if command:
            commands.append(command)
    return commands

def same_executable(observed, expected, cwd, basename):
    """Authenticate aliases of one existing physical executable, preserving argv spelling.

    The basename remains part of driver semantics: generic lld must never stand
    in for ld.lld even if a symlink reaches the same physical file. Files are
    statted, never loaded or inspected as ELF; missing/dangling/foreign paths fail.

    :param observed: Exact executable or joined path-option value reported by the driver.
    :param expected: Original configured executable authority, possibly through an SDK alias.
    :param cwd: Original link working directory for relative observed/configured paths.
    :param basename: Required invocation basename for this exact driver role.
    :return: True only for existing regular executable files with identical device/inode.
    """
    if Path(observed).name != basename or Path(expected).name != basename:
        return False
    identities = []
    try:
        for spelling in (observed, expected):
            path = Path(spelling)
            if not path.is_absolute():
                path = Path(cwd) / path
            info = path.resolve(strict=True).stat()
            if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
                return False
            identities.append((info.st_dev, info.st_ino))
    except (OSError, RuntimeError, ValueError):
        return False
    return identities[0] == identities[1]

def authenticate_wrapper(argv, cwd, wrapper, linker, architectures, partitions, output):
    """Authenticate both real driver levels before the final device link executes.

    Wrapper dry-run writes extracted inputs but executes no child tools; save-temps
    retains those exact inputs for each inner clang -###. Only preview command
    emission is serialized; actual link argv and LTO partitions remain unchanged.
    The existing collective record transaction must enclose this function and the
    subsequent actual link.

    :param argv: Original complete host HIP link argv, including its compiler.
    :param cwd: Original link working directory, retained for relative input paths.
    :param wrapper: Exact selected SDK clang-linker-wrapper path.
    :param linker: Exact builder-proven repaired/original-control ld.lld executable.
    :param architectures: Complete ordered, distinct declared target set.
    :param partitions: Required positive parallel LTO partition count.
    :param output: Exact absolute host output used to derive owned remark paths.
    :return: Actual native linker argv and all original diagnostic levels.
    :raises RuntimeError: Missing/foreign/repeated wrapper, target, linker or record policy.
    """
    wrapper, linker = str(wrapper), str(linker)
    if Path(linker).name != "ld.lld":
        raise RuntimeError("Final native linker requires the canonical ld.lld basename")
    if not architectures or len(set(architectures)) != len(architectures) or partitions <= 0:
        raise RuntimeError("Invalid declared native architecture/partition contract")
    outer = subprocess.run([*argv, "-###"], cwd=cwd, capture_output=True, text=True, check=True)
    outer_text = outer.stdout + outer.stderr
    wrappers = [command for command in command_lines(outer_text)
                if Path(command[0]).name == "clang-linker-wrapper"]
    if (len(wrappers) != 1
            or not same_executable(wrappers[0][0], wrapper, cwd, "clang-linker-wrapper")):
        raise RuntimeError("HIP driver lost or replaced its exact native architecture wrapper")
    # Preview prints device commands from worker threads into one diagnostic stream.
    # The pinned wrapper selects its last --wrapper-jobs value. Insert this
    # preview-only override after its original options but before an optional
    # end-of-options marker; literal inputs after that marker remain inputs.
    # Actual argv and the subsequent real parallel link are unchanged.
    # Do not parse/reassemble torn commands or replace the original input list.
    preview_argv = list(wrappers[0])
    options_end = preview_argv.index("--") if "--" in preview_argv else len(preview_argv)
    preview_argv.insert(options_end, "--wrapper-jobs=1")
    preview = subprocess.run([preview_argv[0], "--dry-run", "--save-temps", *preview_argv[1:]],
                             cwd=cwd, capture_output=True, text=True, check=True)
    wrapper_text = preview.stdout + preview.stderr
    inner = [command for command in command_lines(wrapper_text)
             if "--target=amdgcn-amd-amdhsa" in command]
    if (len(inner) != len(architectures)
            or any(not same_executable(command[0], Path(wrapper).parent / "clang", cwd, "clang")
                   for command in inner)):
        raise RuntimeError("HIP wrapper lost or repeated a native architecture compiler")
    native_commands, inner_diagnostics = [], []
    for arch in architectures:
        matching = [command for command in inner if "-mcpu=" + arch in command]
        if len(matching) != 1:
            raise RuntimeError("HIP wrapper changed a native target or exact fixed linker")
        linker_paths = [arg[len("--ld-path="):] for arg in matching[0] if arg.startswith("--ld-path=")]
        if (len(linker_paths) != 1
                or not same_executable(linker_paths[0], linker, cwd, "ld.lld")):
            raise RuntimeError("HIP wrapper changed a native target or exact fixed linker")
        probe = subprocess.run([*matching[0], "-###"], cwd=cwd,
                               capture_output=True, text=True, check=True)
        text = probe.stdout + probe.stderr
        inner_diagnostics.append({"architecture": arch, "diagnostics": text})
        commands = [command for command in command_lines(text)
                    if any(arg.startswith("-plugin-opt=mcpu=") for arg in command)]
        if (len(commands) != 1 or not same_executable(commands[0][0], linker, cwd, "ld.lld")
                or [arg for arg in commands[0] if arg.startswith("-plugin-opt=mcpu=")]
                   != ["-plugin-opt=mcpu=" + arch]):
            raise RuntimeError("Inner HIP compiler replaced the exact final native linker")
        command = commands[0]
        # The wrapper retains a relative host -o spelling in its dumpdir.
        # Resolve only the observed record identity against the original link
        # cwd; never rewrite the original driver or emitted native command.
        prefix = "-plugin-opt=opt-remarks-filename="
        filenames = [arg[len(prefix):] for arg in command if arg.startswith(prefix)]
        if len(filenames) != 1 or not filenames[0]:
            raise RuntimeError("Inner HIP compiler lost its unique final allocation filename")
        filename = Path(filenames[0])
        if not filename.is_absolute():
            filename = Path(cwd) / filename
        if ([arg for arg in command if arg.startswith("--lto-partitions=")]
                != ["--lto-partitions=" + str(partitions)]
                or filename.resolve() != final_record_base(output, arch).resolve()
                or "-plugin" in command
                or not any("pass-remarks-analysis=" in arg and "stack-frame-layout" in arg
                           for arg in command)):
            raise RuntimeError("Inner HIP compiler changed final partition/allocation policy")
        native_commands.append(command)
    return {"native_commands": native_commands, "driver_diagnostics": outer_text,
            "wrapper_diagnostics": wrapper_text, "inner_driver_diagnostics": inner_diagnostics}

def parse_emitted_functions(objects):
    """Read every defined STT_FUNC identity from one actual final ELF symbol table.

    Binding/visibility never filter code coverage: weak/local/global helpers are
    all physical functions. Undefined references and non-function descriptors
    have no independently emitted function body and do not enter this roster.

    :param objects: Actual llvm-readobj --symbols --elf-output-style=JSON output.
    :return: Sorted unique nonempty emitted function names, without an allowlist.
    :raises RuntimeError: Incomplete/ambiguous symbol-table schema or duplicate identities.
    """
    if (not isinstance(objects, list) or len(objects) != 1
            or not isinstance(objects[0], dict)
            or not isinstance(objects[0].get("Symbols"), list)
            or not objects[0]["Symbols"]):
        raise RuntimeError("Missing unique final ELF symbol table")
    names = []
    for entry in objects[0]["Symbols"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("Symbol"), dict):
            raise RuntimeError("Malformed final ELF symbol record")
        symbol = entry["Symbol"]
        kind, section, name = (symbol.get(key) for key in ("Type", "Section", "Name"))
        if (not all(isinstance(value, dict) for value in (kind, section, name))
                or type(kind.get("Value")) is not int or kind["Value"] < 0
                or type(section.get("Value")) is not int or section["Value"] < 0
                or not isinstance(name.get("Name"), str)):
            raise RuntimeError("Incomplete typed final ELF symbol identity")
        # ELF STT_FUNC=2 and SHN_UNDEF=0 are format constants, not helper policy.
        if kind["Value"] == 2 and section["Value"] != 0:
            if not name["Name"] or "\x00" in name["Name"]:
                raise RuntimeError("Empty/invalid emitted function identity")
            names.append(name["Name"])
    if not names or len(set(names)) != len(names):
        raise RuntimeError("Missing or repeated emitted function identity")
    return sorted(names)

def prove(args, work, guard):
    """Exercise actual two/eight-partition codegen on every declared target without launching kernels."""
    llvm = args.sdk / "lib/llvm/bin"
    base = [str(args.sdk / "bin/hipcc"), "--hip-path=" + str(args.sdk), "--rocm-path=" + str(args.sdk),
            "--offload-new-driver", "-O3", "-fPIC", "-fgpu-rdc", "-save-temps=obj", "-Xarch_device", "-fsave-optimization-record=yaml",
            "-Xarch_device", "-foptimization-record-passes=stack-frame-layout",
            "-Rpass-analysis=stack-frame-layout|kernel-resource-usage"]
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
                argv = [*base, "--offload-arch=" + arch, "-Xoffload-compiler", "--ld-path=" + str(work / variant / "ld.lld"),
                        "-shared",
                        str(work / ("probe-" + arch) / "probe.o"), "-Xoffload-linker",
                        "--lto-partitions=" + str(partitions), "-o", str(directory / "probe.so")]
                admission = authenticate_wrapper(argv, directory, llvm / "clang-linker-wrapper",
                    work / variant / "ld.lld", [arch], partitions, directory / "probe.so")
                dump(directory / "driver-admission.json", admission)
                (directory / "driver-actions.log").write_text(admission["driver_diagnostics"])
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
                symbols = json.loads(tool(["llvm-readobj", "--symbols", "--elf-output-style=JSON", str(code)]))
                dump(directory / "emitted-symbols.json", symbols)
                emitted_functions = parse_emitted_functions(symbols)
                if not KERNELS.issubset(emitted_functions):
                    raise RuntimeError("Final emitted function roster lost a pressure kernel")
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
                       "emitted_functions": emitted_functions,
                       "allocation_functions": len(functions), "record_files": len(files),
                       "assembly_record_files": len(asm_files), "code_sha256": digest(code),
                       "kernels_metadata": [asdict(kernel) for kernel in kernels],
                       "allocations": {name: {"instances": value.instances,
                           "slots": [asdict(slot) for slot in value.slots]} for name, value in functions.items()}}
                if variant == "original":
                    if files or functions:
                        raise RuntimeError("Original-code negative control unexpectedly captured partition allocations")
                else:
                    if (len(files) != partitions or len(asm_files) != partitions
                            or set(functions) != set(emitted_functions)):
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
            if pair[0]["emitted_functions"] != pair[1]["emitted_functions"]:
                raise RuntimeError("Remark repair changed the final emitted function roster")
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
    verify_gvn_filter_proof(report.get("toolchain", {}).get("gvn_filter_proof"))
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
            for row in (original, repaired):
                emitted = row.get("emitted_functions")
                if (not isinstance(emitted, list)
                        or not all(isinstance(name, str) and name and "\x00" not in name for name in emitted)
                        or emitted != sorted(set(emitted)) or not KERNELS.issubset(emitted)):
                    raise RuntimeError("Incomplete typed emitted function roster")
                if row["metadata_partitions"] != parts or row["kernels"] != 16:
                    raise RuntimeError("Final partition/kernel coverage differs from the pressure witness")
            if original["emitted_functions"] != repaired["emitted_functions"]:
                raise RuntimeError("Original/repaired emitted function rosters differ")
            if original["allocation_functions"] or original["record_files"]:
                raise RuntimeError("Original-code negative control no longer rejects lost records")
            if any(repaired[key] != value for key, value in
                   (("metadata_partitions", parts), ("kernels", 16),
                    ("allocation_functions", len(repaired["emitted_functions"])),
                    ("record_files", parts), ("assembly_record_files", parts))):
                raise RuntimeError("Incomplete repaired final partition evidence")
            if original["code_sha256"] != repaired["code_sha256"]:
                raise RuntimeError("Repaired final code bytes differ from original control")
            if original.get("allocations") or len(original.get("kernels_metadata", [])) != 16:
                raise RuntimeError("Original negative-control observations are incomplete")
            if original["kernels_metadata"] != repaired.get("kernels_metadata"):
                raise RuntimeError("Remark repair changed final kernel resource metadata")
            if set(repaired.get("allocations", {})) != set(repaired["emitted_functions"]):
                raise RuntimeError("Final allocation records differ from the complete emitted function roster")
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
    for name in ("sdk", "prefix", "patch", "gvn-patch", "spill-guard"):
        build.add_argument("--" + name, type=Path, required=True)
    for name in ("revision", "backend-sha256", "driver-sha256", "gvn-sha256", "gvn-patch-sha256", "architectures"):
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
              "gvn_source_sha256": args.gvn_sha256, "gvn_patch_sha256": args.gvn_patch_sha256,
              "toolchain": toolchain, "observations": prove(args, work, load_guard(args.spill_guard)),
              "source_inputs": {path.name: digest(path) for path in
                  (Path(__file__), HERE / "main.cpp", HERE / "PressureProbe.hip", HERE / "GVNRemarkProbe.cpp", args.spill_guard)},
              "native_runtime_qualified": False}
    verify(report, args.architectures, work / "patched/ld.lld", load_guard(args.spill_guard))
    binary = args.prefix / "bin/ld.lld"
    share = args.prefix / "share/llaminar-llvm-lto"
    binary.parent.mkdir(parents=True, exist_ok=True)
    share.mkdir(parents=True, exist_ok=True)
    if not binary.exists() or digest(binary) != digest(work / "patched/ld.lld"):
        shutil.copyfile(work / "patched/ld.lld", binary.with_suffix(".new"))
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
