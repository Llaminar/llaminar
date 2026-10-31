#!/usr/bin/env python3
"""Authenticate the actual native group thread-creation/join lifetime repair.

The builder compiles complete selected group.cc and its literal reverse-patched
original, plus the exact ignored-result native abort caller. Installed preflight
checks the source/library/receipt-bound observations without SDK or GPU access.
Source-only oracle execution deliberately cannot mint an installed certificate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

DOCKER = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("nccl_gin_receipt_policy",
                                             DOCKER / "nccl-gin-selection/proof.py")
GIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GIN)
PATCH = "patches/nccl-created-thread-ownership.patch"
INPUTS = GIN.INPUTS
NATIVE_OUTPUT = "".join(f"PREFATAL verified actual join ownership kind {kind}\n"
                        for _ in range(20) for kind in (1, 2, 3)) + (
    "PASS actual native thread ownership: 140 successful-retirement + 60 fatal-join "
    "lifecycles, 2360 assertions; delayed entry, missing-member abort, returned-before-create "
    "receipt, all joined before cleanup, direct no-thread path, native job roots retired\n")
ORIGINAL_ERROR = "REJECT cleanup before all created workers joined\n"


def digest(path):
    """Use the same source/DSO byte authentication as the canonical GIN proof."""
    return GIN.digest(path)


def source_identity():
    """One canonical source/patch/probe identity authenticates both native proofs."""
    return GIN.source_identity()


def receipt_prefix():
    """Reuse the existing installer receipt; no second provenance ledger is created."""
    return GIN.receipt_prefix()


def verify_outcomes(observations):
    """Require all actual-body cases and every exact literal original rejection."""
    if (not isinstance(observations, list) or not all(isinstance(row, dict) for row in observations)
            or [row.get("variant") for row in observations] != [
            "native", "original"]):
        raise RuntimeError("NCCL thread proof requires complete unique native/original controls")
    native, original = observations
    if (type(native.get("compile_exit")) is not int or native["compile_exit"] != 0
            or type(native.get("returncode")) is not int or native["returncode"] != 0
            or native.get("stdout") != NATIVE_OUTPUT or native.get("stderr") != ""):
        raise RuntimeError("NCCL thread native replay is incomplete or has an unrelated failure")
    runs = original.get("runs")
    if (type(original.get("compile_exit")) is not int or original["compile_exit"] != 0
            or not isinstance(runs, list) or len(runs) != 20
            or any(not isinstance(row, dict) or type(row.get("returncode")) is not int
                   or row != dict(returncode=87, stdout="", stderr=ORIGINAL_ERROR) for row in runs)):
        raise RuntimeError("Original NCCL thread proof lacks all 20 exact destructor-order rejections")


def verify_report(report, library, receipt):
    """Reject source-only output, stale dependency bytes or disconnected proof outcomes."""
    if (type(report.get("schema")) is not int or report["schema"] != 1
            or report.get("certificate") is not True
            or report.get("sources") != source_identity()
            or report.get("library_sha256") != digest(library)
            or report.get("build_receipt") != receipt or not receipt.startswith(receipt_prefix())
            or not receipt[len(receipt_prefix()):] or not report.get("compiler_version")):
        raise RuntimeError("NCCL thread proof does not match this source/runtime/build receipt")
    compiler = report.get("compiler_identity")
    if (not isinstance(compiler, dict) or compiler.get("version") != report["compiler_version"]
            or not isinstance(compiler.get("executable"), str) or not compiler["executable"]
            or not isinstance(compiler.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", compiler["sha256"])):
        raise RuntimeError("NCCL thread proof lacks immutable builder compiler identity")
    keys = ("native_source_sha256", "original_source_sha256", "native_header_sha256",
            "original_header_sha256", "native_init_sha256", "native_caller_sha256")
    if any(not isinstance(report.get(key), str)
           or not re.fullmatch(r"[0-9a-f]{64}", report[key]) for key in keys):
        raise RuntimeError("NCCL thread proof lacks actual source/header/caller identities")
    if (report["native_source_sha256"] == report["original_source_sha256"]
            or report["native_header_sha256"] == report["original_header_sha256"]):
        raise RuntimeError("NCCL thread proof requires distinct actual and original implementations")
    verify_outcomes(report.get("observations"))


def mask_literals(text):
    """Strict bounded C++ lexical mask; retain offsets and reject malformed comments/literals."""
    chars = list(text)
    index = 0
    raw = re.compile(r'(?:u8|u|U|L)?R"([^ ()\\\t\r\n]{0,16})\(')
    while index < len(text):
        start = index
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            while end >= 0 and text[:end].rstrip("\r").endswith("\\"):
                end = text.find("\n", end + 1)
            index = len(text) if end < 0 else end
        elif text.startswith("/*", index):
            end = text.find("*/", index + 2)
            if end < 0:
                raise RuntimeError("Unterminated native caller source comment")
            index = end + 2
        elif match := raw.match(text, index):
            terminator = ")" + match.group(1) + '"'
            end = text.find(terminator, match.end())
            if end < 0:
                raise RuntimeError("Unterminated native caller raw literal")
            index = end + len(terminator)
        elif text[index].isdigit():
            # C++ pp-number digit separators are not character-literal openers.
            index += 1
            while index < len(text) and (text[index].isalnum() or text[index] in "._'"):
                index += 1
            continue
        elif text[index] in "\"'":
            quote = text[index]
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == quote:
                    index += 1
                    break
                elif text[index] in "\r\n":
                    raise RuntimeError("Unterminated native caller quoted literal")
                else:
                    index += 1
            else:
                raise RuntimeError("Unterminated native caller quoted literal")
        else:
            index += 1
            continue
        for offset in range(start, min(index, len(text))):
            if chars[offset] != "\n":
                chars[offset] = " "
    return "".join(chars)


def extract_caller(text):
    """Extract one exact bounded global caller, excluding literal/comment braces and fake names."""
    masked = mask_literals(text)
    signature = re.compile(r"\bncclResult_t\s+ncclCommEnsureReady\s*\(\s*ncclComm_t\s+comm\s*\)\s*\{")
    matches = list(signature.finditer(masked))
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one actual native EnsureReady definition")
    match = matches[0]
    start, end = match.start(), match.end()
    depth = 1
    while depth and end < len(masked) and end - start <= 8192:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    if depth or masked[end:].lstrip().startswith("}"):
        raise RuntimeError("Unbalanced or oversized native EnsureReady definition")
    return text[start:end] + "\n"


def caller_body(source):
    """Compile the unchanged selected native caller that intentionally ignores abort results."""
    return extract_caller((source / "src/init.cc").read_text())


def compiler_identity(compiler):
    """Snapshot the actual executable bytes and version without needing the compiler at verify."""
    # Preserve argv[0]: resolving a g++ -> ccache symlink before invocation would
    # change its command interpretation. Authenticate both invocation and target.
    executable = Path(shutil.which(compiler) or compiler).absolute()
    resolved = executable.resolve(strict=True)
    frontend = subprocess.check_output([str(executable), "-print-prog-name=cc1plus"], text=True).strip()
    frontend_path = Path(shutil.which(frontend) or frontend)
    frontend_digest = digest(frontend_path.resolve()) if frontend_path.is_file() else None
    return dict(executable=str(executable), resolved_executable=str(resolved), sha256=digest(resolved),
                version=subprocess.check_output([str(executable), "--version"], text=True),
                frontend=frontend, frontend_sha256=frontend_digest)


def authenticate_inputs(source, compiler, report):
    """Reject source, probe or compiler replacement while the actual oracles were running."""
    expected = {"native_source_sha256": "src/group.cc", "native_header_sha256": "src/include/group.h",
                "native_init_sha256": "src/init.cc"}
    if (any(digest(source / name) != report[key] for key, name in expected.items())
            or source_identity() != report["sources"]
            or compiler_identity(compiler) != report["compiler_identity"]
            or hashlib.sha256(caller_body(source).encode()).hexdigest() != report["native_caller_sha256"]):
        raise RuntimeError("Native source/probe/compiler changed during thread oracle execution")


def run_oracles(source, sdk, compiler):
    """Compile actual native code and real pthread controls without loading a GPU library."""
    implementation = source / "src/group.cc"
    header = source / "src/include/group.h"
    report = dict(certificate=False, sources=source_identity(),
                  compiler_identity=compiler_identity(compiler),
                  native_source_sha256=digest(implementation), native_header_sha256=digest(header),
                  native_init_sha256=digest(source / "src/init.cc"), observations=[])
    report["compiler_version"] = report["compiler_identity"]["version"]
    with tempfile.TemporaryDirectory(prefix="llaminar-nccl-thread-") as directory:
        temporary = Path(directory)
        original = temporary / "original"
        for name in ("src/group.cc", "src/include/group.h"):
            target = original / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / name).read_bytes())
        subprocess.run(["git", "apply", "--reverse", str(DOCKER / PATCH)],
                       cwd=original, check=True, capture_output=True, text=True)
        report["original_source_sha256"] = digest(original / "src/group.cc")
        report["original_header_sha256"] = digest(original / "src/include/group.h")
        caller = temporary / "actual_ensure_ready.inc"
        caller.write_text(caller_body(source))
        report["native_caller_sha256"] = digest(caller)
        for variant, native_source, first_include in (
                ("native", implementation, source / "src/include"),
                ("original", original / "src/group.cc", original / "src/include")):
            binary = temporary / ("probe-" + variant)
            includes = (first_include, temporary, source / "src/include",
                        source / "src/include/plugin", source / "src/graph",
                        source / "build/include", sdk / "include", sdk / "include/cccl")
            command = [report["compiler_identity"]["executable"], "-std=c++17", "-O2", "-pthread", "-ffunction-sections",
                       "-fdata-sections", "-Wl,--gc-sections",
                       *("-I" + str(path) for path in includes),
                       '-DNATIVE_GROUP_SOURCE="' + str(native_source) + '"',
                       *( ["-Wl,--wrap=abort"] if variant == "native" else ["-DQFN_ORIGINAL_SOURCE"]),
                       str(DOCKER / "nccl-thread-lifetime/probe.cpp"), "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            if compiled.returncode:
                raise RuntimeError(f"NCCL thread {variant} compilation failed:\n{compiled.stderr}")
            observation = dict(variant=variant, compile_exit=compiled.returncode)
            runs = []
            for _ in range(1 if variant == "native" else 20):
                result = subprocess.run([str(binary)], capture_output=True, text=True,
                                        timeout=30, preexec_fn=GIN.disable_core)
                runs.append(dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
            if variant == "native":
                observation.update(runs[0])
            else:
                observation["runs"] = runs
            report["observations"].append(observation)
    verify_outcomes(report["observations"])
    authenticate_inputs(source, compiler, report)
    return report


def compile_report(args):
    """Publish a certificate only for the selected source build's actual completed DSO."""
    args.report.unlink(missing_ok=True)
    pending = args.report.with_suffix(".tmp")
    pending.unlink(missing_ok=True)
    source = args.source.resolve(strict=True)
    library = args.library.resolve(strict=True)
    if not any(path.resolve() == library for path in (source / "build/lib").glob("*.so.*")):
        raise RuntimeError("Thread compilation requires the selected source build's actual DSO")
    receipt = args.receipt.read_text().strip()
    library_digest = digest(library)
    report = run_oracles(source, args.sdk, args.compiler)
    report.update(schema=1, certificate=True, library_sha256=library_digest, build_receipt=receipt)
    verify_report(report, library, receipt)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        pending.write_text(json.dumps(report, indent=2) + "\n")
        authenticate_inputs(source, args.compiler, report)
        if args.receipt.read_text().strip() != receipt:
            raise RuntimeError("Native build receipt changed during thread oracle execution")
        verify_report(report, library, receipt)
        pending.replace(args.report)
    except BaseException:
        pending.unlink(missing_ok=True)
        args.report.unlink(missing_ok=True)
        raise
    print("NCCL threads: 200 actual-body lifecycles pass; literal original rejects 20/20")


def main():
    """Keep compilation in the builder and passive byte/receipt checks in installed preflight."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("identity", "probe", "compile", "verify"))
    for option in ("report", "library", "receipt", "source", "sdk"):
        parser.add_argument("--" + option, type=Path)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if args.mode == "identity":
        print(GIN.proof_identity())
        return
    if args.mode == "probe":
        if not all((args.source, args.sdk, args.report)):
            parser.error("probe requires --source, --sdk and --report; it emits no certificate")
        args.report.unlink(missing_ok=True)
        pending = args.report.with_suffix(".tmp")
        pending.unlink(missing_ok=True)
        report = run_oracles(args.source.resolve(strict=True), args.sdk, args.compiler)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        try:
            pending.write_text(json.dumps(report, indent=2) + "\n")
            authenticate_inputs(args.source, args.compiler, report)
            pending.replace(args.report)
        except BaseException:
            pending.unlink(missing_ok=True)
            args.report.unlink(missing_ok=True)
            raise
        print("NCCL source-only oracle passes; no DSO certificate was emitted")
    elif args.mode == "compile":
        if not all((args.source, args.sdk, args.report, args.library, args.receipt)):
            parser.error("compile requires source, SDK, report, actual built library and receipt")
        compile_report(args)
    else:
        if not all((args.report, args.library, args.receipt)):
            parser.error("verify requires report, selected library and build receipt")
        verify_report(json.loads(args.report.read_text()), args.library, args.receipt.read_text().strip())
        print("NCCL threads: source/runtime-bound lifecycle and original-code negative verified")


if __name__ == "__main__":
    main()
