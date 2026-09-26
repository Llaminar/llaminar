#!/usr/bin/env python3
"""Build and verify source-bound GPU spill-compiler proofs without accelerators.

Compilation belongs to the build dependency graph, where the actual SDKs live.
CTest verifies the complete recorded compiler outcomes, including rejected
spills and rebuilds, without importing compiler SDKs into the installed runner.
These are explicit modes: missing evidence is fatal, never a reason to skip
the test, synthesize success, or try another compiler.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
PROOF_INPUTS = (
    "tests/v2/integration/build/test_gpu_spill_compilation.py",
    "tests/v2/integration/build/CMakeLists.txt",
    "tests/v2/integration/build/gpu_spill_probe.cu",
    "tests/v2/integration/build/gpu_spill_probe.hip",
    "cmake/GpuSpillGuard.cmake",
    "scripts/build/gpu_spill_guard.py",
)


def source_identity():
    """Bind the tiny compiler probes and enforcement policy to installed source."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in PROOF_INPUTS}


def cases():
    """One inventory owns configuration, pressure witness and rebuild coverage."""
    for config in ("Release", "Integration", "Debug"):
        variants = ("clean", "private", "spill", "helper", "clean_helper")
        if config != "Debug":
            # Scalar-register constraints require optimized uniformity analysis.
            variants += ("scalar", "mixed")
        for variant in variants:
            for attempt in (("initial",) if config == "Debug" else ("initial", "rebuild")):
                yield config, variant, attempt


def verify_observation(language, observation):
    """Require the actual diagnostic, exit status and object-retirement result."""
    config, variant = observation["config"], observation["variant"]
    diagnostic = observation["diagnostic"]
    rejected = config != "Debug" and variant in ("spill", "helper")
    if rejected:
        expected = ("register spills are forbidden" if language == "HIP"
                    else "Registers are spilled")
        if observation["returncode"] == 0 or expected not in diagnostic:
            raise RuntimeError(f"{config}: {variant} was not rejected for register spills:\n{diagnostic}")
        if observation["objects_present"]:
            raise RuntimeError("failed spill compilation retained a linkable object")
    elif observation["returncode"] != 0 or not observation["objects_present"]:
        raise RuntimeError(f"{config}: {variant} did not produce a valid object:\n{diagnostic}")
    if language == "HIP" and variant in ("scalar", "mixed"):
        if "scalar-to-vector register moves; no memory spill" not in diagnostic:
            raise RuntimeError(f"{config}: scalar witness did not exercise register moves:\n{diagnostic}")


def verify_report(report, language, compiler, architectures):
    """Reject stale, incomplete or foreign proofs without executing a compiler."""
    if (report.get("schema") != 1 or report.get("language") != language
            or report.get("compiler") != compiler
            or report.get("architectures") != architectures
            or not report.get("compiler_version")
            or report.get("sources") != source_identity()):
        raise RuntimeError("GPU spill compiler proof identity does not match this build/source")
    observations = report.get("observations", [])
    if [(item.get("config"), item.get("variant"), item.get("attempt"))
            for item in observations] != list(cases()):
        raise RuntimeError("GPU spill compiler proof has incomplete or duplicate cases")
    for observation in observations:
        verify_observation(language, observation)


def compile_report(args):
    """Run the real production guard, retaining both first-build and rebuild evidence."""
    # A failed rebuild cannot leave an older successful proof for CTest to use.
    args.report.unlink(missing_ok=True)
    report = {"schema": 1, "language": args.language, "compiler": args.compiler,
              "architectures": args.architectures, "sources": source_identity(),
              "compiler_version": subprocess.check_output([args.compiler, "--version"], text=True),
              "observations": []}
    with tempfile.TemporaryDirectory(prefix="llaminar-spill-probe-") as directory:
        for config, variant, attempt in cases():
            build = Path(directory) / config
            if not build.exists():
                subprocess.run([args.cmake, "-S", str(Path(__file__).parent), "-B", str(build),
                                "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={args.ninja}",
                                f"-DPROBE_LANGUAGE={args.language}", f"-DCMAKE_BUILD_TYPE={config}",
                                f"-DCMAKE_{args.language}_COMPILER={args.compiler}",
                                f"-DCMAKE_{args.language}_ARCHITECTURES={args.architectures}"], check=True)
            if attempt == "rebuild":
                # Rejected objects must already be absent. Retire only this
                # private fixture's successful object to exercise ccache hits.
                for output in (build / f"CMakeFiles/probe_{variant}.dir").glob("*.o"):
                    output.unlink()
            result = subprocess.run([args.cmake, "--build", str(build), "--parallel",
                                     "--target", f"probe_{variant}"], text=True, capture_output=True)
            observation = {"config": config, "variant": variant, "attempt": attempt,
                           "returncode": result.returncode, "diagnostic": result.stdout + result.stderr,
                           "objects_present": bool(list((build / f"CMakeFiles/probe_{variant}.dir").glob("*.o")))}
            verify_observation(args.language, observation)
            report["observations"].append(observation)
            print(f"{args.language} {config} {variant} {attempt}: expected outcome verified", flush=True)
    verify_report(report, args.language, args.compiler, args.architectures)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pending = args.report.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(args.report)


def main():
    """Compilation is build-owned; installed preflight only validates its proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "verify"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--language", choices=("CUDA", "HIP"), required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--cmake")
    parser.add_argument("--ninja")
    args = parser.parse_args()
    if args.mode == "compile":
        if not args.cmake or not args.ninja:
            parser.error("compile requires --cmake and --ninja")
        compile_report(args)
    else:
        verify_report(json.loads(args.report.read_text()), args.language, args.compiler, args.architectures)
        print(f"{args.language}: all {len(list(cases()))} source-bound compiler outcomes verified")


if __name__ == "__main__":
    main()
