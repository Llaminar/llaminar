#!/usr/bin/env python3
"""Compile actual NCCL/RCCL partial-file lifecycles; passively verify installed evidence.

Only the builder compiles complete native translation units and literal
reverse-patched originals. Installed verification reads exact policy/report/DSO
bytes without native loading, compiler execution, vendor source or GPU access.
Source-only mode cannot publish a certificate for an existing unpatched DSO.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile

DOCKER = Path(__file__).resolve().parents[1]
ROOT = DOCKER.parents[1]
HERE = Path(__file__).resolve().parent
SHA = re.compile(r"[0-9a-f]{64}")
UNIT = "src/misc/shmutils.cc"
SCOPE = "host-named-file-partial-create-only"


def require(condition, message):
    """Reject absent, stale or incomplete evidence without substituting another path."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """Hash exact small source/compiler/DSO bytes without loading native code."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def is_sha(value):
    """Reject numeric/bool substitutions for exact hexadecimal identity strings."""
    return type(value) is str and SHA.fullmatch(value) is not None


def load_module(path):
    """Reuse existing identity and dependency/publication helpers only when needed."""
    spec = importlib.util.spec_from_file_location("shared_file_" + Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rccl_helper():
    """Keep compiler dependency parsing/atomic publication in the existing RCCL helper."""
    return load_module(ROOT / "tests/v2/integration/build/test_rccl_thread_lifetime.py")


def patch(backend):
    """Return the sole new source delta for literal original reconstruction."""
    return DOCKER / "patches" / (backend + "-partial-file-cleanup.patch")


def source_identity(backend):
    """Bind the shared replay and existing ordered backend policy, using relocatable names."""
    if backend == "nccl":
        return load_module(DOCKER / "nccl-gin-selection/proof.py").source_identity()
    names = ("shared-file-lifecycle/proof.py", "shared-file-lifecycle/probe.cpp",
             "patches/rccl-partial-file-cleanup.patch", "apply-rccl-capture-patch.sh",
             "apply-rccl-patch-chain.py")
    result = {name: digest(DOCKER / name) for name in names}
    for name in ("test_rccl_thread_lifetime.py", "test_rccl_selector_taxonomy.py",
                 "test_rccl_small_allgather_policy.py"):
        relative = "tests/v2/integration/build/" + name
        result[relative] = digest(ROOT / relative)
    # Authenticate every actual ordered patch, not an independent copied chain.
    chain = (DOCKER / "apply-rccl-capture-patch.sh").read_text()
    for name in re.findall(r"rccl-[a-z0-9-]+\.patch", chain):
        result["patches/" + name] = digest(DOCKER / "patches" / name)
    return result


def valid_receipt(backend, receipt):
    """Require the new patch in the existing complete native build identity."""
    if type(receipt) is not str:
        return False
    if backend == "nccl":
        prefix = load_module(DOCKER / "nccl-gin-selection/proof.py").receipt_prefix()
        return receipt.startswith(prefix) and len(receipt) > len(prefix)
    marker = "-shm-" + digest(patch(backend)) + "-selector-"
    return receipt.count(marker) == 1 and receipt.endswith(rccl_helper().receipt_suffix())


def expected(variant):
    """Name the complete observed case counts; original controls must witness both defects."""
    original = variant == "original"
    return dict(schema=1, replays=20, fd0_cases=60, symlink_cases=20,
                multi_attach_cases=20, case_executions=260, failed_open_path_allocations=20 if original else 0,
                failed_creator_names_retained=80 if original else 0,
                expected_retained_names=80 if original else 0,
                live_neighbor_and_failed_attach_preserved=True,
                preexisting_files_clobbered=40 if original else 0,
                all_exact_diagnostic_files_removed=True)


def verify_outcomes(observations):
    """Accept only complete native/original successful checks of the intended observations."""
    require(type(observations) is list and len(observations) == 2,
            "Missing complete native/original controls")
    for row, variant in zip(observations, ("candidate", "original")):
        require(type(row) is dict and row.get("variant") == variant
                and type(row.get("compile_exit")) is int and row["compile_exit"] == 0
                and type(row.get("returncode")) is int and row["returncode"] == 0
                and row.get("stderr") == "" and type(row.get("stdout")) is str,
                "Replay compilation/crash/unrelated failure cannot certify cleanup")
        try:
            measured = json.loads(row["stdout"])
        except (TypeError, ValueError) as error:
            raise RuntimeError("Replay did not publish complete structured observations") from error
        wanted = expected(variant)
        require(type(measured) is dict and measured == wanted
                and all(type(measured.get(key)) is type(value) for key, value in wanted.items()),
                "Incomplete replay counts or missing original defect witnesses")


def closure_digest(closure):
    """Bind the exact compiler-produced dependency inventory in canonical order."""
    return hashlib.sha256(json.dumps(closure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_report(report, library, backend, receipt=None):
    """Validate installed proof without consulting compiler, SDK or vendor source paths."""
    require(type(report) is dict and type(report.get("schema")) is int and report["schema"] == 1
            and report.get("certificate") is True and report.get("backend") == backend
            and report.get("scope") == SCOPE and report.get("native_runtime_qualified") is False
            and report.get("sources") == source_identity(backend)
            and report.get("library_sha256") == digest(library), "Diagnostic/stale source or DSO proof")
    require(valid_receipt(backend, report.get("build_receipt"))
            and (receipt is None or report["build_receipt"] == receipt), "Missing exact patched build receipt")
    compiler = report.get("compiler", {})
    require(type(compiler) is dict and type(compiler.get("path")) is str
            and Path(compiler["path"]).is_absolute() and type(compiler.get("version")) is str
            and bool(compiler["version"]) and is_sha(compiler.get("sha256")),
            "Missing immutable native compiler identity")
    require(type(report.get("sdk")) is str and Path(report["sdk"]).is_absolute()
            and type(report.get("architecture")) is str and bool(report["architecture"]),
            "Missing selected SDK/architecture")
    if backend == "nccl":
        prefix = load_module(DOCKER / "nccl-gin-selection/proof.py").receipt_prefix()
        require(report["architecture"] == report["build_receipt"][len(prefix):],
                "Recorded architecture differs from native build receipt")
    if backend == "rccl":
        require(re.fullmatch(r"gfx[0-9a-z]+", report["architecture"]) is not None
                and is_sha(report.get("compile_database_sha256")),
                "Missing selected native RCCL compilation identity")
    closure = report.get("dependency_closure", {})
    source = report.get("native_source")
    require(type(closure) is dict and len(closure) >= 3
            and all(type(name) is str and Path(name).is_absolute() and type(value) is str
                    and SHA.fullmatch(value) for name, value in closure.items())
            and type(source) is str and closure.get(source) == report.get("native_source_sha256")
            and closure.get(compiler["path"]) == compiler["sha256"]
            and report.get("dependency_closure_sha256") == closure_digest(closure),
            "Incomplete actual compiler/source/header closure")
    require(is_sha(report.get("original_source_sha256"))
            and report["original_source_sha256"] != report["native_source_sha256"],
            "Literal original is absent or identical")
    tools = compiler.get("tools", {})
    require(type(tools) is dict and compiler["path"] in tools
            and all(type(name) is str and Path(name).is_absolute() and type(value) is str
                    and SHA.fullmatch(value) and closure.get(name) == value for name, value in tools.items()),
            "Compiler frontend/linker tools are absent from the actual closure")
    commands = report.get("native_compile_commands")
    require(type(commands) is list and len(commands) == 2
            and all(type(command) is list and command and all(type(word) is str for word in command)
                    and "-O3" in command and "-c" in command for command in commands),
            "Missing actual complete-TU compile commands")
    probes = report.get("probe_compile_commands")
    require(type(probes) is list and len(probes) == 2
            and all(type(command) is list and command and all(type(word) is str for word in command)
                    and "-MD" in command and "-Wl,--wrap=malloc" in command for command in probes),
            "Missing actual replay compiler dependency commands")
    verify_outcomes(report.get("observations"))


def execute(command, *, cwd=None, env=None):
    """Run bounded CPU compiler/replay commands and preserve a precise failure diagnostic."""
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
    require(result.returncode == 0, "CPU command failed: " + repr(command) + "\n" + result.stderr[-6000:])
    return result


def native_inputs(args):
    """Select the actual native host compilation closure; no runtime discovery occurs."""
    require(args.sdk is not None and args.sdk.is_dir(), "Selected SDK headers are required in the builder")
    require(args.backend != "rccl" or args.rccl_build is not None,
            "RCCL builder proof requires its actual compile database")
    driver = shutil.which(str(args.compiler))
    require(driver is not None, "Selected native compiler is unavailable")
    driver = Path(driver).absolute()
    if args.backend == "nccl":
        require(args.source is not None, "NCCL proof requires the selected source build")
        source = args.source.resolve(strict=True) / UNIT
        roots = (args.source / "build/include", args.source / "src/include",
                 args.source / "src/include/plugin", args.sdk / "include", args.sdk / "include/cccl")
        flags = ["-std=c++17", "-O3", "-fPIC", *("-I" + str(p.resolve()) for p in roots)]
        return source, driver, flags, None
    helper = rccl_helper()
    source, _ = helper.module("test_rccl_selector_taxonomy").inputs(args.rccl_build, UNIT)
    rows = json.loads((args.rccl_build / "compile_commands.json").read_text())
    matches = [row for row in rows if (Path(row["directory"]) / row["file"]).resolve() == source]
    require(len(matches) == 1, "Expected exactly one actual shmutils native compiler row")
    row = matches[0]
    words = row.get("arguments") or shlex.split(row["command"])
    helper.module("test_rccl_small_allgather_policy").native_relocation_mode(words)
    require("-O3" in words and "--offload-arch=" + args.architecture in words,
            "Selected native row lacks canonical O3/architecture")
    require(driver.resolve() == (args.sdk / "lib/llvm/bin/clang++").resolve(strict=True),
            "RCCL proof requires the selected SDK compiler")
    # Preserve every native argument except the driver/source/object/dependency
    # outputs. The new source remains in the same include closure and host mode.
    flags = []
    index = 1
    while index < len(words):
        word = words[index]
        if word in ("-o", "-MF", "-MT", "-MQ"):
            index += 2
            continue
        if word not in ("-c", "-MD", "-MMD", "-MP") and word != row["file"] and word != str(source):
            flags.append(word)
        index += 1
    flags += ["--cuda-host-only", "--hip-path=" + str(args.sdk), "--rocm-path=" + str(args.sdk)]
    return source, driver, flags, row["directory"]


def run_oracles(args, directory):
    """Compile whole actual units and the reverse-only-new-patch original; run real file lifecycles."""
    source, driver, flags, cwd = native_inputs(args)
    if args.candidate_source is not None:
        source = args.candidate_source.resolve(strict=True)
    compiler = driver.resolve(strict=True)
    original = directory / "original" / UNIT
    original.parent.mkdir(parents=True)
    original.write_bytes(source.read_bytes())
    execute(["git", "apply", "--reverse", "--check", str(patch(args.backend))], cwd=directory / "original")
    execute(["git", "apply", "--reverse", str(patch(args.backend))], cwd=directory / "original")
    native_sha = digest(source)
    report = dict(schema=1, certificate=False, backend=args.backend, scope=SCOPE,
                  native_runtime_qualified=False, sources=source_identity(args.backend),
                  compiler=dict(path=str(compiler), sha256=digest(compiler),
                                version=execute([str(driver), "--version"]).stdout),
                  sdk=str(args.sdk.resolve(strict=True)), architecture=args.architecture,
                  native_source=str(source), native_source_sha256=native_sha,
                  original_source_sha256=digest(original), native_compile_commands=[], probe_compile_commands=[], observations=[])
    dependencies = {source, compiler}
    # A ccache masquerading driver is not the compiler frontend. Bind GCC's
    # actual frontend and link collector as well; Clang's frontend is integrated.
    tools = {str(compiler): report["compiler"]["sha256"]}
    if "clang" not in report["compiler"]["version"].lower():
        for name in ("cc1plus", "collect2"):
            tool = Path(execute([str(driver), "-print-prog-name=" + name]).stdout.strip())
            require(tool.is_absolute() and tool.is_file(), "Native GCC tool identity unavailable: " + name)
            tool = tool.resolve(strict=True)
            tools[str(tool)] = digest(tool)
            dependencies.add(tool)
    report["compiler"]["tools"] = tools
    if args.backend == "rccl":
        database = args.rccl_build / "compile_commands.json"
        dependencies.add(database.resolve())
        report["compile_database_sha256"] = digest(database)
    observed = {source: native_sha, compiler: report["compiler"]["sha256"]}
    for variant, unit in (("candidate", source), ("original", original)):
        obj = directory / (variant + ".o")
        dep = directory / (variant + ".d")
        command = [str(driver), *flags, "-ffunction-sections", "-fdata-sections", "-MD", "-MF", str(dep),
                   "-c", str(unit), "-o", str(obj)]
        def parse_dependencies(dependency_file=dep):
            """Read actual compiler output before/after object creation, including system headers."""
            if args.backend == "rccl":
                return rccl_helper().dependency_paths(dependency_file, Path(cwd))
            text = dependency_file.read_text().replace("\\\n", "").split(":", 1)[1].replace("$$", "$")
            return {Path(name).resolve(strict=True) for name in shlex.split(text)}

        execute([str(driver), *flags, "-M", "-MF", str(dep), "-MT", str(obj), str(unit)], cwd=cwd)
        before = {path: digest(path) for path in parse_dependencies()}
        execute(command, cwd=cwd)
        require(parse_dependencies() == set(before) and all(digest(path) == value
                for path, value in before.items()), "Actual compiler dependency changed during compilation")
        report["native_compile_commands"].append(command)
        for path, value in before.items():
            if not path.is_relative_to(directory):
                require(observed.get(path, value) == value, "Header identity changed between replay variants")
                observed[path] = value
                dependencies.add(path)
        binary = directory / (variant + "-probe")
        probe_dep = directory / (variant + "-probe.d")
        execute([str(driver), "-std=c++17", "-O2", "-M", "-MF", str(probe_dep), str(HERE / "probe.cpp")])
        probe_before = {path: digest(path) for path in parse_dependencies(probe_dep)}
        probe_command = [str(driver), "-std=c++17", "-O2", "-ffunction-sections", "-fdata-sections",
                         "-MD", "-MF", str(probe_dep), str(HERE / "probe.cpp"), str(obj),
                         "-Wl,--gc-sections", "-Wl,--wrap=fallocate", "-Wl,--wrap=mmap",
                         "-Wl,--wrap=malloc", "-ldl", "-o", str(binary)]
        execute(probe_command)
        require(parse_dependencies(probe_dep) == set(probe_before) and all(digest(path) == value
                for path, value in probe_before.items()), "Probe compiler dependency changed during linking")
        for path, value in probe_before.items():
            require(observed.get(path, value) == value, "Probe header differs from native compilation")
            observed[path] = value
            dependencies.add(path)
        report["probe_compile_commands"].append(probe_command)
        scratch = directory / (variant + "-scratch")
        scratch.mkdir()
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("NCCL_", "RCCL_", "AMD_LOG_"))
               and key not in ("LD_PRELOAD", "LD_LIBRARY_PATH")}
        result = execute([str(binary), variant, str(scratch)], env=env)
        require(not list(scratch.iterdir()), "Replay retained private scratch files")
        report["observations"].append(dict(variant=variant, compile_exit=0,
            returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    dependencies.update((HERE / name).resolve() for name in ("proof.py", "probe.cpp"))
    dependencies.add(patch(args.backend).resolve())
    for name in report["sources"]:
        dependencies.add((ROOT / name if name.startswith("tests/") else DOCKER / name).resolve())
    closure = {str(path): observed.get(path, digest(path)) for path in sorted(dependencies)}
    require(digest(source) == native_sha and digest(compiler) == report["compiler"]["sha256"]
            and source_identity(args.backend) == report["sources"]
            and all(digest(path) == value for path, value in closure.items()), "Inputs changed during native replay")
    report.update(dependency_closure=closure, dependency_closure_sha256=closure_digest(closure))
    verify_outcomes(report["observations"])
    return report, dependencies


def retire(path):
    """Remove only exact final/interrupted builder outputs before a failing rebuild."""
    path.unlink(missing_ok=True)
    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)


def publish(report, path):
    """Atomically publish complete verified JSON beside its final destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    pending.replace(path)


def compile_report(args):
    """Bind a complete native build and atomically publish only verified stable evidence."""
    require(args.report is not None and args.library is not None and args.candidate_source is None,
            "Certificate requires report/library and forbids diagnostic source overrides")
    protected = {args.library.resolve(), patch(args.backend).resolve(), (HERE / "proof.py").resolve(),
                 (HERE / "probe.cpp").resolve()}
    require(args.report.resolve() not in protected and
            (args.depfile is None or args.depfile.resolve() not in protected | {args.report.resolve()}),
            "Proof output aliases a native/policy input")
    for item in (args.receipt, args.compiler):
        if item is not None:
            resolved = shutil.which(str(item)) if item == args.compiler else str(item)
            if resolved:
                protected.add(Path(resolved).resolve())
    if args.source:
        protected.add((args.source / UNIT).resolve())
    if args.rccl_build:
        protected.update((args.rccl_build / name).resolve() for name in
                         ("compile_commands.json", ".llaminar-rccl-commit", "hipify/" + UNIT))
    require(args.report.suffix == ".json" and args.report.resolve() not in protected
            and (args.depfile is None or (args.depfile.suffix == ".d" and args.depfile.resolve() not in protected)),
            "Proof output aliases a native/receipt input")
    retire(args.report)
    if args.depfile:
        retire(args.depfile)
    if args.backend == "nccl":
        require(args.receipt is not None and args.source is not None, "Missing source-owned NCCL receipt")
        require(args.library.resolve() in {p.resolve() for p in (args.source / "build/lib").glob("*.so.*")},
                "Selected DSO is not the supplied completed source build")
        receipt_path = args.receipt
    else:
        require(args.rccl_build is not None and args.depfile is not None, "Missing native RCCL build/depfile")
        require(digest(args.rccl_build / "librccl.so") == digest(args.library), "Selected DSO differs from native build")
        receipt_path = args.rccl_build / ".llaminar-rccl-commit"
    receipt = receipt_path.read_text().strip()
    require(valid_receipt(args.backend, receipt), "Native build lacks new source patch identity")
    library_sha = digest(args.library)
    with tempfile.TemporaryDirectory(prefix="llaminar-shared-file-") as temporary:
        report, dependencies = run_oracles(args, Path(temporary))
    report.update(certificate=True, build_receipt=receipt, library_sha256=library_sha)
    require(receipt_path.read_text().strip() == receipt and digest(args.library) == library_sha
            and source_identity(args.backend) == report["sources"]
            and all(digest(path) == value for path, value in report["dependency_closure"].items()),
            "Native closure changed before publication")
    verify_report(report, args.library, args.backend, receipt)
    if args.depfile:
        dependencies.update((args.library.resolve(), receipt_path.resolve()))
        rccl_helper().write_depfile(args.depfile, args.report, dependencies)
    publish(report, args.report)


def main():
    """Separate builder certification, source diagnostics and passive installed verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "source-only", "verify"))
    parser.add_argument("--backend", choices=("nccl", "rccl"), required=True)
    for name in ("report", "library", "receipt", "source", "sdk", "rccl-build", "depfile", "candidate-source", "evidence-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--architecture", default="toolkit-default")
    args = parser.parse_args()
    try:
        if args.mode == "verify":
            require(args.report is not None and args.library is not None, "Missing report/library")
            receipt = args.receipt.read_text().strip() if args.receipt else None
            verify_report(json.loads(args.report.read_text()), args.library, args.backend, receipt)
        elif args.mode == "compile":
            compile_report(args)
        else:
            require(args.report is None and args.library is None and args.depfile is None
                    and args.evidence_dir is not None, "Diagnostic mode cannot emit an installed certificate")
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            retire(args.evidence_dir / "source-only.json")
            with tempfile.TemporaryDirectory(prefix="host-closure-", dir=args.evidence_dir) as temporary:
                report, _ = run_oracles(args, Path(temporary))
            publish(report, args.evidence_dir / "source-only.json")
        print("Shared-file lifecycle: " + args.backend + " " + args.mode + " PASS")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        parser.exit(1, "Shared-file proof rejected: " + str(error) + "\n")


if __name__ == "__main__":
    main()
