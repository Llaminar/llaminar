#!/usr/bin/env python3
"""Build and verify device-free regressions for native RCCL protocol defaults.

The builder compiles the real dependency consumers with their own native
headers and definitions. A private reverse-patched negative control must fail
the same geometry oracle. Installed Unit/preflight checks authenticate these
outcomes against the selected RCCL DSO and current regression source without
requiring the vendor source tree, compiler SDK or an accelerator.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
PATCH = "scripts/docker/patches/rccl-communicator-protocol-defaults.patch"
INPUTS = (PATCH, "tests/v2/integration/build/rccl_protocol_geometry_probe.cpp",
          "tests/v2/integration/build/test_rccl_protocol_geometry.py")
MODES = ("forward", "reverse", "concurrent")
CHECKS = {"forward": 1080, "reverse": 1080, "concurrent": 1728000}


def digest(path):
    """Bind source or native executable bytes; no model/cache data is read."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_identity():
    """Return the complete small regression/policy identity present in runners."""
    return {name: digest(ROOT / name) for name in INPUTS}


def native_compile_inputs(build):
    """Read the exact dependency translation unit from its CMake compile database."""
    entries = json.loads((build / "compile_commands.json").read_text())
    matches = [row for row in entries if row["file"].endswith("/hipify/src/rccl_wrap.cc")]
    if len(matches) != 1:
        raise RuntimeError("RCCL compile database must identify exactly one native protocol consumer")
    row = matches[0]
    words = row.get("arguments") or shlex.split(row["command"])
    definitions = []
    index = 0
    while index < len(words):
        argument = words[index]
        if argument in ("-D", "-I", "-isystem", "-iquote", "-include"):
            if index + 1 == len(words):
                raise RuntimeError("Incomplete native compiler input")
            definitions.extend(words[index:index + 2])
            index += 2
        else:
            if argument.startswith(("-D", "-I")):
                definitions.append(argument)
            index += 1
    if not definitions:
        raise RuntimeError("RCCL native compile inputs are empty")
    source = Path(row["file"])
    if not source.is_absolute():
        source = Path(row["directory"]) / source
    return source.resolve(strict=True), definitions


def verify_report(report, library):
    """Reject stale sources/runtime, missing cases, wrong or vacuous outcomes."""
    if (report.get("schema") != 1 or report.get("sources") != source_identity()
            or report.get("library_sha256") != digest(library)
            or not report.get("compiler_version")
            or not report.get("native_source_sha256")
            or not report.get("native_arch_source_sha256")):
        raise RuntimeError("RCCL protocol proof identity does not match this source/runtime")
    observations = report.get("observations", [])
    if [(row.get("variant"), row.get("mode")) for row in observations] != [
            (variant, mode) for variant in ("native", "original") for mode in MODES]:
        raise RuntimeError("RCCL protocol proof has incomplete or duplicate cases")
    for row in observations:
        measured = row.get("measured", {})
        if measured.get("mode") != row["mode"] or measured.get("checks") != CHECKS[row["mode"]]:
            raise RuntimeError("RCCL protocol oracle did not complete its full geometry matrix")
        failures = measured.get("failures")
        expected = 0 if row["variant"] == "native" else 1
        if (row.get("returncode") != expected or type(failures) is not int
                or failures < 0 or failures > measured["checks"]
                or (failures != 0 if expected == 0 else failures == 0)):
            raise RuntimeError("RCCL protocol consumer/negative control has an unexpected outcome")


def compile_report(args):
    """Run actual native consumers and the preserved faulty implementation on CPU."""
    args.report.unlink(missing_ok=True)
    source, native_inputs = native_compile_inputs(args.rccl_build)
    arch_source = source.parent / "misc/archinfo.cc"
    native_library = args.rccl_build / "librccl.so"
    if digest(native_library) != digest(args.library):
        raise RuntimeError("Selected RCCL DSO is not the native build supplying the tested consumers")
    receipt = (args.rccl_build / ".llaminar-rccl-commit").read_text().strip()
    if not receipt.endswith("-protocol-" + digest(ROOT / PATCH)):
        raise RuntimeError("RCCL build receipt does not contain the current protocol repair")
    report = dict(schema=1, sources=source_identity(), library_sha256=digest(args.library),
                  compiler_version=subprocess.check_output([args.compiler, "--version"], text=True),
                  native_source_sha256=digest(source), native_arch_source_sha256=digest(arch_source),
                  rccl_receipt=receipt, native_compile_inputs=native_inputs, observations=[])
    with tempfile.TemporaryDirectory(prefix="llaminar-rccl-protocol-") as directory:
        temporary = Path(directory)
        original = temporary / "src/rccl_wrap.cc"
        original.parent.mkdir()
        generated = source.read_bytes()
        preamble = b'#include "hip/hip_runtime.h"\n'
        if not generated.startswith(preamble):
            raise RuntimeError("RCCL generated source lacks the canonical HIP include preamble")
        # hipify adds this include ahead of the source's first line. Temporarily
        # detach it so Git can reverse the file-header hunk at its real boundary,
        # then restore the exact generated include before compiling old code.
        original.write_bytes(generated[len(preamble):])
        # Reverse the exact production repair in a disposable copy. The old
        # consumers must fail, so a vacuous or disconnected test cannot pass.
        subprocess.run(["git", "apply", "--reverse", str(ROOT / PATCH)],
                       cwd=temporary, check=True)
        original.write_bytes(preamble + original.read_bytes())
        symbols = temporary / "probe.map"
        symbols.write_text("{ local: *; };\n")
        for variant, implementation in (("native", source), ("original", original)):
            binary = temporary / variant
            command = [args.compiler, "-x", "hip", "--offload-host-only",
                       "--offload-arch=" + args.architecture,
                       "--hip-path=" + str(args.sdk), "--rocm-path=" + str(args.sdk),
                       "-std=c++17", "-O3", "-ffunction-sections", "-fdata-sections",
                       "-fvisibility=hidden", "-Wl,--gc-sections", "-pthread",
                       "-Wl,--version-script=" + str(symbols), *native_inputs,
                       '-DLLAMINAR_RCCL_WRAP_SOURCE="' + str(implementation) + '"',
                       '-DLLAMINAR_RCCL_ARCH_SOURCE="' + str(arch_source) + '"',
                       str(ROOT / INPUTS[1]), "-o", str(binary), "-x", "none", str(args.library),
                       "-Wl,-rpath," + str(args.library.resolve().parent),
                       "-Wl,-rpath," + str(args.sdk / "lib")]
            # RCCL exports its trace object's destructor even when no trace was
            # constructed. Link that dependency normally; localize the probe's
            # unrelated API definitions so they cannot interpose on the DSO.
            subprocess.run(command, check=True, cwd=args.rccl_build)
            for mode in MODES:
                result = subprocess.run([str(binary), mode], text=True, capture_output=True)
                if result.stderr:
                    raise RuntimeError("Unexpected native protocol diagnostic: " + result.stderr)
                observation = dict(variant=variant, mode=mode, returncode=result.returncode,
                                   measured=json.loads(result.stdout))
                report["observations"].append(observation)
                print(json.dumps(observation), flush=True)
    verify_report(report, args.library)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pending = args.report.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(args.report)


def main():
    """Keep compilation in the builder and source/runtime verification in CTest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "verify"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--rccl-build", type=Path)
    parser.add_argument("--compiler")
    parser.add_argument("--architecture")
    parser.add_argument("--sdk", type=Path)
    args = parser.parse_args()
    if args.mode == "compile":
        if not all((args.rccl_build, args.compiler, args.architecture, args.sdk)):
            parser.error("compile requires --rccl-build, --compiler, --architecture and --sdk")
        compile_report(args)
    else:
        verify_report(json.loads(args.report.read_text()), args.library)
        print("RCCL: native and negative-control protocol geometry outcomes verified")


if __name__ == "__main__":
    main()
