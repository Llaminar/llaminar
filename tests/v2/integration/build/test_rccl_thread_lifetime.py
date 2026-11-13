#!/usr/bin/env python3
"""Build actual RCCL thread/caller replays; verify installed source/DSO evidence.

Compilation and literal reverse-patched negatives belong to the SDK builder.
Installed verification reads only this source family, a report and DSO bytes;
no vendor source, compiler, dynamic library load or accelerator is required.
Source-only diagnostics deliberately cannot produce an admission certificate.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
PATCH = "scripts/docker/patches/rccl-created-thread-lifetime.patch"
PROTOCOL_PATCH = "scripts/docker/patches/rccl-communicator-protocol-defaults.patch"
SMALLAG_PATCH = "scripts/docker/patches/rccl-small-allgather-ll-policy.patch"
BUILD = "tests/v2/integration/build/"
PROBES = {
    "group": BUILD + "rccl_thread_lifetime_probe.cpp",
    "original_group": BUILD + "rccl_thread_lifetime_original_probe.cpp",
    "caller": BUILD + "rccl_thread_lifetime_caller_probe.cpp",
}
INPUTS = (PATCH, PROTOCOL_PATCH, SMALLAG_PATCH,
          "scripts/docker/apply-rccl-capture-patch.sh", "scripts/docker/apply-rccl-patch-chain.py",
          *PROBES.values(),
          BUILD + "rccl_thread_lifetime_source.py", BUILD + "test_rccl_thread_lifetime.py",
          BUILD + "test_rccl_selector_taxonomy.py", BUILD + "test_rccl_small_allgather_policy.py",
          "tests/v2/unit/scripts/check_cpu_prepared_allocation_policy.py",
          "tests/v2/unit/scripts/check_gpu_blocking_sync_policy.py")
SHA = re.compile(r"[0-9a-f]{64}")
REPETITIONS = 20
EXPECTED = {
    ("native", "group"): dict(schema=1, suite="created-thread", cases=120,
                               real_thread_cases=40, join_failure_cases=40, violations=0),
    ("original", "group"): dict(schema=1, suite="created-thread", cases=40,
                                 real_thread_cases=0, join_failure_cases=0, violations=40),
    ("native", "caller"): dict(schema=1, suite="ignored-join-caller", cases=40,
                                ordinary_error_cases=20, join_failure_cases=20, violations=0),
    ("original", "caller"): dict(schema=1, suite="ignored-join-caller", cases=40,
                                  ordinary_error_cases=20, join_failure_cases=20, violations=20),
}


def require(condition, message):
    """Reject absent, malformed or stale evidence without an inferred success."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """Hash exact code/compiler/runtime input bytes without loading a library."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    """Return the relocatable complete regression and extraction identity."""
    return {name: digest(ROOT / name) for name in INPUTS}


def module(name):
    """Load an existing source-owned builder primitive without executing its CLI."""
    path = ROOT / (BUILD + name + ".py")
    spec = importlib.util.spec_from_file_location("thread_proof_" + name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def receipt_suffix():
    """Keep the new repair before the existing strict protocol/smallAG suffix."""
    return ("-thread-" + digest(ROOT / PATCH) + "-protocol-" + digest(ROOT / PROTOCOL_PATCH)
            + "-smallag-" + digest(ROOT / SMALLAG_PATCH))


def verify_result(row, variant, suite):
    """Require complete exact JSON output and the intended process polarity."""
    require(type(row) is dict and set(row) == {"returncode", "stdout", "stderr", "measured"},
            "Malformed native/original execution record")
    require(type(row["returncode"]) is int and row["returncode"] == int(variant == "original")
            and row["stderr"] == "" and type(row["stdout"]) is str,
            "Native/original execution crashed, passed vacuously or emitted diagnostics")
    expected = EXPECTED[(variant, suite)]
    require(type(row["measured"]) is dict and row["measured"] == expected
            and all(type(row["measured"].get(key)) is type(value)
                    for key, value in expected.items()), "Incomplete native/original case matrix")
    try:
        parsed = json.loads(row["stdout"])
    except (ValueError, TypeError) as error:
        raise RuntimeError("Invalid native/original stdout") from error
    require(parsed == expected and all(type(parsed.get(key)) is type(value)
                                      for key, value in expected.items()),
            "Native stdout differs from recorded measurement")


def verify_report(report, library):
    """Verify installed identity and every original rejection without SDK access."""
    require(type(report) is dict and report.get("schema") == 1
            and type(report.get("schema")) is int and report.get("certificate") is True
            and report.get("sources") == source_identity()
            and report.get("library_sha256") == digest(library)
            and report.get("native_init_runtime_qualified") is False,
            "Thread proof source/runtime identity differs or is diagnostic-only")
    receipt = report.get("build_receipt")
    require(type(receipt) is str and receipt.endswith(receipt_suffix())
            and len(receipt) > len(receipt_suffix()), "Missing exact canonical thread repair chain")
    for name in ("native_source_sha256", "native_header_sha256", "original_source_sha256",
                 "original_header_sha256", "literal_caller_sha256", "compile_database_sha256"):
        require(type(report.get(name)) is str and SHA.fullmatch(report[name]),
                "Missing native/original/caller/compile identity")
    require(report["native_source_sha256"] != report["original_source_sha256"]
            and report["native_header_sha256"] != report["original_header_sha256"],
            "Original control did not reverse actual source and private ABI")
    compiler = report.get("compiler", {})
    require(type(compiler) is dict and type(compiler.get("path")) is str
            and Path(compiler["path"]).is_absolute()
            and type(compiler.get("stat")) is list and len(compiler["stat"]) == 5
            and all(type(value) is int for value in compiler["stat"])
            and type(report.get("compiler_version")) is str and bool(report["compiler_version"])
            and type(report.get("sdk")) is str and Path(report["sdk"]).is_absolute()
            and type(report.get("architecture")) is str
            and re.fullmatch(r"gfx[0-9a-z]+", report["architecture"]),
            "Missing compiler/SDK/architecture closure")
    require(Path(compiler["path"]).is_relative_to(Path(report["sdk"]) / "lib/llvm/bin"),
            "Recorded compiler does not belong to selected SDK")
    closure = report.get("dependency_closure", {})
    require(type(closure) is dict and len(closure) >= 3
            and all(type(name) is str and Path(name).is_absolute()
                    and type(value) is str and SHA.fullmatch(value)
                    for name, value in closure.items()), "Missing actual compiler dependency closure")
    require(report.get("dependency_closure_sha256") == closure_digest(closure),
            "Compiler dependency closure identity differs")
    native_files = report.get("native_files", {})
    require(type(native_files) is dict and set(native_files) == {"group", "header", "caller"}
            and all(type(path) is str and Path(path).is_absolute() for path in native_files.values())
            and closure.get(native_files["group"]) == report["native_source_sha256"]
            and closure.get(native_files["header"]) == report["native_header_sha256"]
            and closure.get(native_files["caller"]) == report["literal_caller_sha256"]
            and compiler["path"] in closure,
            "Compiler closure omits actual native source/header/caller/compiler")
    inputs = report.get("native_compile_inputs")
    require(type(inputs) is list and len(inputs) == 2 and all(type(row) is dict for row in inputs)
            and [row.get("source") for row in inputs] == [native_files["group"], native_files["caller"]]
            and all(type(row.get("flags")) is list and row["flags"]
                    and all(type(flag) is str for flag in row["flags"]) for row in inputs),
            "Missing actual per-translation-unit native compiler inputs")
    for row in inputs:
        module("test_rccl_small_allgather_policy").native_relocation_mode(row["flags"])
    observations = report.get("observations")
    require(type(observations) is list and len(observations) == 2
            and all(type(row) is dict for row in observations)
            and [row.get("variant") for row in observations] == ["native", "original"],
            "Missing or duplicate native/original observations")
    native, original = observations
    require(type(native.get("compile_exit")) is int and native["compile_exit"] == 0
            and type(native.get("full_group_compile_exit")) is int
            and native["full_group_compile_exit"] == 0
            and type(original.get("compile_exit")) is int and original["compile_exit"] == 0,
            "Actual native/original compilation did not complete")
    verify_result(native.get("group"), "native", "group")
    verify_result(native.get("caller"), "native", "caller")
    rejected = original.get("rejections")
    require(type(rejected) is list and len(rejected) == REPETITIONS,
            "Missing repeated actual original rejections")
    for index, row in enumerate(rejected):
        require(type(row) is dict and type(row.get("iteration")) is int
                and row["iteration"] == index and set(row) == {"iteration", "group", "caller"},
                "Duplicate or incomplete original replay")
        verify_result(row["group"], "original", "group")
        verify_result(row["caller"], "original", "caller")


def closure_digest(closure):
    """Bind the canonical sorted compiler dependency map in the outcome record."""
    return hashlib.sha256(json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_inputs(build):
    """Reuse the native definition/include extractor and exact RDC contract."""
    source, flags = module("test_rccl_selector_taxonomy").inputs(build, "src/group.cc")
    init, _ = module("test_rccl_selector_taxonomy").inputs(build, "src/init.cc")
    rows = json.loads((build / "compile_commands.json").read_text())
    result = {}
    for name, native_source in (("group", source), ("caller", init)):
        _, native_flags = module("test_rccl_selector_taxonomy").inputs(
            build, "src/group.cc" if name == "group" else "src/init.cc")
        matches = [row for row in rows
                   if (Path(row["directory"]) / row["file"]).resolve() == native_source]
        require(len(matches) == 1, "Expected one actual native compiler row")
        words = matches[0].get("arguments") or shlex.split(matches[0]["command"])
        rdc = module("test_rccl_small_allgather_policy").native_relocation_mode(words)
        result[name] = [*native_flags, *rdc]
    return source, init, result


def dependency_paths(path, directory):
    """Parse compiler-produced Make dependencies, preserving escaped path characters."""
    raw = path.read_text().replace("\\\n", "")
    require(":" in raw, "Compiler did not produce a dependency rule")
    dependencies = raw.split(":", 1)[1].replace("$$", "$")
    return {((directory / value) if not Path(value).is_absolute() else Path(value)).resolve(strict=True)
            for value in shlex.split(dependencies)}


def escape_make(path):
    """Escape a path for Ninja's Make depfile reader; never invoke a shell."""
    return str(path).replace("\\", "\\\\").replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ").replace(":", "\\:")


def write_depfile(path, target, dependencies):
    """Publish only the exact builder-owned depfile after all input checks succeed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x") as stream:
        stream.write(escape_make(target) + ": " + " ".join(escape_make(p) for p in sorted(dependencies)) + "\n")
    pending.replace(path)


def compiler_stat(path):
    """Observe the exact resolved compiler without altering its driver invocation."""
    value = path.stat()
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def run_probe(binary):
    """Run only the CPU oracle with native runtime/preload environment removed."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("NCCL_", "RCCL_", "AMD_LOG_"))
           and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
    result = subprocess.run([str(binary)], text=True, capture_output=True, env=env, timeout=30)
    try:
        measured = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("Native replay did not emit complete structured output") from error
    return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr, measured=measured)


def run_build(args, temporary, group, header, init, flags):
    """Compile actual bodies/ABI and literal reverse-patched controls with compiler deps."""
    extract = module("rccl_thread_lifetime_source")
    initial = {path.resolve(): digest(path) for path in (group, header, init)}
    old = temporary / "original"
    prefixes = {}
    for source, relative in ((group, "src/group.cc"), (header, "src/include/group.h")):
        data = source.read_bytes()
        prefix = b'#include "hip/hip_runtime.h"\n'
        prefixes[relative] = prefix if data.startswith(prefix) else b""
        target = old / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data[len(prefixes[relative]):])
    subprocess.run(["git", "apply", "--reverse", str(ROOT / PATCH)], cwd=old, check=True,
                   capture_output=True, text=True)
    for relative, prefix in prefixes.items():
        target = old / relative
        target.write_bytes(prefix + target.read_bytes())
    old_group, old_header = old / "src/group.cc", old / "src/include/group.h"
    require(digest(group) != digest(old_group) and digest(header) != digest(old_header),
            "Original control did not reverse source and header")
    old_hashes = {path.resolve(): digest(path) for path in (old_group, old_header)}
    dependencies = set()
    snapshot = dict(initial)
    binaries = {}
    jobs = []
    common = [str(Path(args.compiler).absolute()), "-x", "hip", "--offload-host-only",
              "--offload-arch=" + args.architecture, "--hip-path=" + str(args.sdk),
              "--rocm-path=" + str(args.sdk), "-std=c++17", "-O3", "-pthread",
              "-ffunction-sections", "-fdata-sections", "-fvisibility=hidden"]
    for variant, implementation, private_header in (("native", group, header),
                                                    ("original", old_group, old_header)):
        generated = temporary / (variant + "-inputs")
        extract.write_inputs(implementation, init, generated, native=variant == "native")
        extra = ["-I" + str(private_header.parent), "-I" + str(generated)]
        for suite in ("group", "caller"):
            probe = ROOT / PROBES["original_group" if variant == "original" and suite == "group" else suite]
            name = variant + "-" + suite
            defines = ["-DLLAMINAR_HAS_THREAD_RECEIPT=" + str(int(variant == "native"))]
            jobs.append((name, probe, [*extra, *flags[suite], *defines], True))
            binaries[(variant, suite)] = temporary / name
    jobs.append(("native-full-group", group, ["-I" + str(header.parent), *flags["group"]], False))

    def compile_one(job):
        """Snapshot discovered inputs, compile, then authenticate actual emitted deps."""
        name, source, extra, link = job
        before = temporary / (name + "-before.d")
        emitted = temporary / (name + ".d")
        obj = temporary / (name + ".o")
        subprocess.run([*common, *extra, "-M", "-MF", str(before), "-MT", name, str(source)],
                       cwd=args.rccl_build, check=True, capture_output=True, text=True)
        discovered = dependency_paths(before, args.rccl_build)
        original = {path: digest(path) for path in discovered}
        subprocess.run([*common, *extra, "-MD", "-MF", str(emitted), "-MT", name,
                        "-c", str(source), "-o", str(obj)], cwd=args.rccl_build,
                       check=True, capture_output=True, text=True)
        require(dependency_paths(emitted, args.rccl_build) == discovered
                and all(digest(path) == value for path, value in original.items()),
                "Source/header closure changed during native compilation")
        if link:
            command = [args.host_compiler, str(obj), "-pthread", "-Wl,--gc-sections", "-o", str(temporary / name)]
            if name != "original-group":
                command += ["-Wl,--wrap=abort"]
            subprocess.run(command, cwd=args.rccl_build, check=True, capture_output=True, text=True)
        return original

    with ThreadPoolExecutor() as executor:
        compiled = list(executor.map(compile_one, jobs))
    for closure in compiled:
        for path, value in closure.items():
            require(path not in snapshot or snapshot[path] == value, "Native compiles observed different header contents")
            snapshot[path] = value
            if not path.is_relative_to(temporary):
                dependencies.add(path)
    native = dict(variant="native", compile_exit=0, full_group_compile_exit=0,
                  group=run_probe(binaries[("native", "group")]),
                  caller=run_probe(binaries[("native", "caller")]))
    original = dict(variant="original", compile_exit=0, rejections=[])
    for index in range(REPETITIONS):
        original["rejections"].append(dict(iteration=index,
            group=run_probe(binaries[("original", "group")]),
            caller=run_probe(binaries[("original", "caller")])))
    require(all(digest(path) == value for path, value in {**snapshot, **old_hashes}.items()),
            "Native/source header changed during execution")
    # Temporary reverse-patched files and exact-body includes derive solely from
    # these persistent native sources plus the bound repair/extraction inputs.
    dependencies.update((group, header, init))
    pins = dict(native_source_sha256=initial[group.resolve()], native_header_sha256=initial[header.resolve()],
                original_source_sha256=old_hashes[old_group.resolve()], original_header_sha256=old_hashes[old_header.resolve()],
                literal_caller_sha256=initial[init.resolve()])
    persistent = {str(path): value for path, value in snapshot.items()
                  if not path.is_relative_to(temporary)}
    return [native, original], dependencies, pins, persistent


def compile_report(args):
    """Retire stale evidence, authenticate one completed build and atomically publish."""
    require(args.report.resolve() != args.library.resolve()
            and args.depfile.resolve() != args.library.resolve()
            and args.depfile.resolve() != args.report.resolve(), "Proof outputs alias a native input")
    atomic = module("test_rccl_small_allgather_policy")
    atomic.retire_proof(args.report)
    atomic.retire_proof(args.depfile)
    require(not args.candidate_source, "Diagnostic source override cannot certify a DSO")
    require(digest(args.rccl_build / "librccl.so") == digest(args.library),
            "Selected DSO is not the supplied completed native build")
    receipt = (args.rccl_build / ".llaminar-rccl-commit").read_text().strip()
    require(receipt.endswith(receipt_suffix()) and len(receipt) > len(receipt_suffix()),
            "Native build lacks exact canonical thread repair chain")
    group, init, flags = native_inputs(args.rccl_build)
    header = group.parent / "include/group.h"
    driver = Path(args.compiler).absolute()
    compiler = driver.resolve(strict=True)
    require(compiler == (args.sdk / "lib/llvm/bin/clang++").resolve(strict=True),
            "Certificate requires the selected SDK's actual Clang driver")
    original_compiler = compiler_stat(compiler)
    database = args.rccl_build / "compile_commands.json"
    database_sha = digest(database)
    original_library = digest(args.library)
    sources = source_identity()
    with tempfile.TemporaryDirectory(prefix="llaminar-rccl-thread-") as directory:
        observations, dependencies, pins, compiled_closure = run_build(args, Path(directory), group, header, init, flags)
    dependencies.update((ROOT / name).resolve() for name in INPUTS)
    dependencies.update((database.resolve(), (args.rccl_build / ".llaminar-rccl-commit").resolve(),
                         args.library.resolve(), compiler))
    closure = {str(path): compiled_closure.get(str(path), digest(path)) for path in sorted(dependencies)}
    report = dict(schema=1, certificate=True, native_init_runtime_qualified=False,
                  sources=sources, library_sha256=original_library,
                  build_receipt=receipt, compiler=dict(path=str(compiler), stat=original_compiler),
                  compiler_version=subprocess.check_output([str(driver), "--version"], text=True),
                  sdk=str(args.sdk.resolve(strict=True)), architecture=args.architecture,
                  compile_database_sha256=database_sha,
                  native_files=dict(group=str(group.resolve()), header=str(header.resolve()), caller=str(init.resolve())),
                  native_compile_inputs=[dict(source=str(group.resolve()), flags=flags["group"]),
                                         dict(source=str(init.resolve()), flags=flags["caller"])],
                  dependency_closure=closure,
                  dependency_closure_sha256=closure_digest(closure), observations=observations, **pins)
    require(source_identity() == sources and compiler_stat(compiler) == original_compiler
            and digest(database) == database_sha and digest(args.library) == original_library
            and (args.rccl_build / ".llaminar-rccl-commit").read_text().strip() == receipt,
            "Native source/compiler/runtime inputs changed during proof")
    require(all(digest(path) == value for path, value in closure.items()),
            "Compiler dependency content changed before proof publication")
    verify_report(report, args.library)
    write_depfile(args.depfile, args.report, dependencies)
    atomic.publish_verified_report(report, args.report)


def source_only(args):
    """Run private source replays without constructing or retaining a certificate."""
    require(args.report is None and args.depfile is None and args.library is None,
            "Source-only diagnostics cannot emit installed proof or depfile")
    group, init, flags = native_inputs(args.rccl_build)
    if args.candidate_source:
        group = args.candidate_source / "group.cc"
    header = group.parent / "include/group.h"
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rccl-thread-source-", dir=args.evidence_dir) as directory:
        observations, _, pins, compiled_closure = run_build(args, Path(directory), group, header, init, flags)
    for variant in observations:
        if variant["variant"] == "native":
            for suite in ("group", "caller"):
                verify_result(variant[suite], "native", suite)
        else:
            for row in variant["rejections"]:
                for suite in ("group", "caller"):
                    verify_result(row[suite], "original", suite)
    (args.evidence_dir / "source-only.json").write_text(json.dumps(
        dict(certificate=False, native_init_runtime_qualified=False, sources=source_identity(),
             dependency_closure=compiled_closure, observations=observations, **pins), indent=2) + "\n")
    print("RCCL thread source-only replays passed; certificate=false")


def main():
    """Keep the installed verify interface independent of builder/compiler state."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "verify", "source"))
    for name in ("report", "library", "rccl-build", "sdk", "depfile", "candidate-source", "evidence-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--compiler")
    parser.add_argument("--host-compiler", default="c++")
    parser.add_argument("--architecture")
    args = parser.parse_args()
    if args.mode == "verify":
        require(args.report is not None and args.library is not None, "Verify requires report and library")
        verify_report(json.loads(args.report.read_text()), args.library)
    else:
        require(all((args.rccl_build, args.compiler, args.architecture, args.sdk)),
                "Builder requires completed native build/compiler/architecture/SDK")
        if args.mode == "compile":
            require(args.report is not None and args.library is not None and args.depfile is not None,
                    "Compile requires report, library and depfile")
            compile_report(args)
        else:
            require(args.evidence_dir is not None, "Source-only diagnostics require evidence-dir")
            source_only(args)


if __name__ == "__main__":
    main()
