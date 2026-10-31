#!/usr/bin/env python3
"""Build and authenticate the actual NCCL GIN auto-selection regression.

The dependency builder compiles its real plugin translation unit and an exact
reverse-patched negative control. Only provider callbacks are mocked: the native
plugin structure, selector and state transitions remain authoritative. Installed
Unit/preflight verifies the source/library-bound proof without CUDA, an SDK or
vendor source. This certificate covers selection, not network transport.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import resource
import signal
import subprocess
import tempfile


DOCKER = Path(__file__).resolve().parent.parent
REVISION = "dbc86fd06e8b0c4517b95d8958a09ccacf9520c9"
PATCHES = tuple("patches/" + name for name in (
    "nccl-capture-reentry.patch", "nccl-device-live-rows.patch",
    "nccl-gin-auto-selection.patch"))
INPUTS = (*PATCHES, "nccl-gin-selection/probe.cpp", "nccl-gin-selection/proof.py",
          "patches/nccl-created-thread-ownership.patch", "nccl-thread-lifetime/probe.cpp",
          "nccl-thread-lifetime/proof.py", "patches/nccl-partial-file-cleanup.patch",
          "shared-file-lifecycle/probe.cpp", "shared-file-lifecycle/proof.py")
CASES = ("auto-available", "auto-zero-devices", "auto-negative-inventory",
         "auto-query-error", "auto-probe-error", "explicit-GDAKI",
         "explicit-GDAKI-unavailable", "explicit-GDAKI-error", "explicit-proxy",
         "external-auto", "no-GIN")
NATIVE_OUTPUT = "PASS: 220 actual-source selection replays, 2160 assertions, zero GPU/IB calls\n"


def digest(path):
    """Authenticate a small source file or native DSO, never cache/model data."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    """Use relocatable names shared by the toolchain installer and test runner."""
    return {name: digest(DOCKER / name) for name in INPUTS}


def proof_identity():
    """Invalidate installer reuse when the executable regression changes."""
    return hashlib.sha256(json.dumps(source_identity(), sort_keys=True).encode()).hexdigest()


def receipt_prefix():
    """Bind the canonical source/patch sequence and regression before architecture."""
    return ":".join((REVISION, *(digest(DOCKER / name) for name in PATCHES),
                     proof_identity())) + ":"


def verify_report(report, library, receipt):
    """Fail on stale runtime, incomplete execution or a disconnected negative."""
    if (report.get("schema") != 1 or report.get("sources") != source_identity()
            or report.get("library_sha256") != digest(library)
            or report.get("build_receipt") != receipt
            or not receipt.startswith(receipt_prefix())
            or not receipt[len(receipt_prefix()):]
            or not report.get("compiler_version")):
        raise RuntimeError("NCCL GIN proof does not match this source/runtime/build receipt")
    source_digests = [report.get(key) for key in ("native_source_sha256", "original_source_sha256")]
    if (any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in source_digests) or source_digests[0] == source_digests[1]):
        raise RuntimeError("NCCL GIN proof lacks distinct actual and original implementations")
    observations = report.get("observations", [])
    if [row.get("variant") for row in observations] != ["native", "original"]:
        raise RuntimeError("NCCL GIN proof has incomplete or duplicate controls")
    native, original = observations
    if (native.get("compile_exit") != 0 or native.get("returncode") != 0
            or native.get("stdout") != NATIVE_OUTPUT
            or native.get("stderr") != "".join(f"replay {case}\n" for case in CASES) * 20):
        raise RuntimeError("NCCL GIN replay did not complete all 220 real-source cases")
    if (original.get("compile_exit") != 0 or original.get("returncode") != -signal.SIGSEGV
            or original.get("stdout") != ""
            or original.get("stderr") != "replay auto-available\nreplay auto-zero-devices\n"):
        raise RuntimeError("Original NCCL GIN selector did not reproduce the zero-device fault")


def disable_core():
    """Retain the negative's exit/diagnostic without generating a core artifact."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def compile_report(args):
    """Compile both complete native plugin bodies against the built dependency."""
    args.report.unlink(missing_ok=True)
    source = args.source.resolve(strict=True)
    implementation = source / "src/plugin/net.cc"
    library = args.library.resolve(strict=True)
    # A copied runtime DSO cannot masquerade as the artifact of these sources.
    build_libraries = list((source / "build/lib").glob("*.so.*"))
    if not any(path.resolve() == library for path in build_libraries):
        raise RuntimeError("GIN compilation requires the selected source build's actual DSO")
    receipt = args.receipt.read_text().strip()
    report = dict(schema=1, sources=source_identity(), library_sha256=digest(library),
                  build_receipt=receipt, native_source_sha256=digest(implementation),
                  compiler_version=subprocess.check_output([args.compiler, "--version"], text=True),
                  observations=[])
    with tempfile.TemporaryDirectory(prefix="llaminar-nccl-gin-") as directory:
        temporary = Path(directory)
        original = temporary / "src/plugin/net.cc"
        original.parent.mkdir(parents=True)
        original.write_bytes(implementation.read_bytes())
        subprocess.run(["git", "apply", "--reverse", str(DOCKER / PATCHES[2])],
                       cwd=temporary, check=True)
        report["original_source_sha256"] = digest(original)
        for variant, native_source in (("native", implementation), ("original", original)):
            binary = temporary / variant
            includes = (source / "src", source / "src/include", source / "src/include/plugin",
                        source / "build/include", args.sdk / "include", args.sdk / "include/cccl")
            command = [args.compiler, "-std=c++17", "-O2", "-ffunction-sections",
                       "-fdata-sections", "-Wl,--gc-sections", "-pthread",
                       *("-I" + str(path) for path in includes),
                       '-DNATIVE_NET_SOURCE="' + str(native_source) + '"',
                       str(DOCKER / "nccl-gin-selection/probe.cpp"), "-o", str(binary)]
            compiled = subprocess.run(command, capture_output=True, text=True)
            if compiled.returncode:
                raise RuntimeError(f"NCCL GIN {variant} compilation failed:\n{compiled.stderr}")
            result = subprocess.run([str(binary)], capture_output=True, text=True,
                                    timeout=30, preexec_fn=disable_core)
            report["observations"].append(dict(variant=variant, compile_exit=compiled.returncode,
                returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    verify_report(report, library, receipt)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pending = args.report.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(args.report)
    print("NCCL GIN: 220 actual-source replays pass; original zero-device fault reproduced")


def main():
    """Keep native compilation in the builder and passive verification in CTest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("identity", "compile", "verify"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--sdk", type=Path)
    parser.add_argument("--compiler", default="g++")
    args = parser.parse_args()
    if args.mode == "identity":
        print(proof_identity())
        return
    if not all((args.report, args.library, args.receipt)):
        parser.error("compile/verify require --report, --library and --receipt")
    if args.mode == "compile":
        if not all((args.source, args.sdk)):
            parser.error("compile also requires --source and --sdk")
        compile_report(args)
    else:
        verify_report(json.loads(args.report.read_text()), args.library,
                      args.receipt.read_text().strip())
        print("NCCL GIN: source/runtime-bound selection and original-code negative verified")


if __name__ == "__main__":
    main()
