#!/usr/bin/env python3
"""Compile and authenticate HIP cache/header replay without an accelerator.

The build owns actual compiler/cache/Ninja execution. Installed preflight only
verifies its complete source/compiler/SDK/architecture-bound receipt, so runtime
test images need neither compilers nor profiler tools. The replay changes only a
device-conditional header while the root source and compile command stay fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[4]
FIXTURE = Path(__file__).parent / "hip_device_input_identity"
INPUTS = (
    "tests/v2/integration/build/test_hip_device_input_identity.py",
    "tests/v2/integration/build/hip_device_input_identity/CMakeLists.txt",
    "tests/v2/integration/build/hip_device_input_identity/probe.hip",
    "cmake/GpuSpillGuard.cmake", "scripts/build/gpu_spill_guard.py",
)
STAGES = (("cold512", 512), ("warm512", 512), ("changed256", 256),
          ("warm256", 256), ("restored512", 512), ("invalid", None), ("reset512", 512))


def source_identity():
    """Bind both the replay and real compiler launcher to the recorded source."""
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in INPUTS}


def verify_report(report, compiler, architectures, rocm_path):
    """Reject missing, stale or incomplete observations without invoking native tools."""
    if (report.get("schema") != 1 or report.get("compiler") != compiler
            or report.get("architectures") != architectures or report.get("rocm_path") != rocm_path
            or report.get("sources") != source_identity() or not report.get("compiler_version")
            or not report.get("cache_version")):
        raise RuntimeError("HIP device-input proof identity does not match this source/build")
    observations = report.get("observations", [])
    if [(item.get("stage"), item.get("words")) for item in observations] != list(STAGES):
        raise RuntimeError("HIP device-input proof has incomplete or repeated replay stages")
    for item in observations:
        if item["words"] is None:
            if (item.get("returncode") == 0 or item.get("object_present")
                    or "device input preprocessing failed" not in item.get("diagnostic", "")):
                raise RuntimeError("invalid device header did not reject and retire the old object")
            continue
        expected = {target: item["words"] * 4 for target in architectures.split(";")}
        if (item.get("returncode") != 0 or not item.get("object_present")
                or item.get("lds_bytes") != expected or not item.get("device_header_dependency")):
            raise RuntimeError(f"{item['stage']}: native device input/dependency mismatch")
        results = item.get("cache_results", [])
        if item["stage"] in ("warm512", "warm256", "restored512", "reset512"):
            if "direct_cache_hit" not in results:
                raise RuntimeError(f"{item['stage']}: unchanged device inputs did not retain cache reuse")
        elif "cache_miss" not in results:
            raise RuntimeError(f"{item['stage']}: new device inputs reused an old cache entry")


def native_lds(output, llvm_bin, root):
    """Inspect each final architecture's allocation instead of trusting source text."""
    bundle = root / "probe.fatbin"
    subprocess.run([str(llvm_bin / "llvm-objcopy"), "--dump-section", f".hip_fatbin={bundle}",
                    str(output), str(root / "host-copy.o")], check=True, capture_output=True)
    bundler = str(llvm_bin / "clang-offload-bundler")
    targets = subprocess.check_output([bundler, "--list", "--type=o", f"--input={bundle}"], text=True)
    result = {}
    for index, target in enumerate(targets.splitlines()):
        if not target.startswith(("hip-", "hipv4-")):
            continue
        code = root / f"device-{index}.co"
        subprocess.run([bundler, "--unbundle", "--type=o", f"--targets={target}",
                        f"--input={bundle}", f"--output={code}"], check=True, capture_output=True)
        notes = subprocess.check_output([str(llvm_bin / "llvm-readelf"), "--notes", str(code)], text=True)
        sizes = re.findall(r"\.group_segment_fixed_size:\s*(\d+)", notes)
        architecture = target.rpartition("--")[2]
        if len(sizes) != 1 or architecture in result:
            raise RuntimeError("native cache probe needs one unique kernel per architecture")
        result[architecture] = int(sizes[0])
    return result


def compile_report(args):
    """Replay cold/warm builds, hidden-header edits, rejection and reset through Ninja."""
    args.report.unlink(missing_ok=True)
    report = {"schema": 1, "compiler": args.compiler, "architectures": args.architectures,
              "rocm_path": args.rocm_path, "sources": source_identity(),
              "compiler_version": subprocess.check_output([args.compiler, "--version"], text=True),
              "cache_version": subprocess.check_output([args.ccache, "--version"], text=True),
              "observations": []}
    with tempfile.TemporaryDirectory(prefix="llaminar-hip-cache-proof-") as directory:
        root = Path(directory)
        source, build = root / "source", root / "build"
        source.mkdir()
        shutil.copyfile(FIXTURE / "probe.hip", source / "probe.hip")
        header = source / "device_dimensions.inc"
        header.write_text("constexpr unsigned kDeviceWords = 512;\n")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("CCACHE_")}
        config = root / "ccache.conf"
        extra = root / "user-extra-input"
        extra.write_text("Retain the caller's configured cache inputs.\n")
        config.write_text("direct_mode = true\ndepend_mode = false\n"
                          f"extra_files_to_hash = {extra}\n")
        environment.update(CCACHE_DIR=str(root / "cache"), CCACHE_CONFIGPATH=str(config))
        subprocess.run([args.cmake, "-S", str(FIXTURE), "-B", str(build), "-G", "Ninja",
                        f"-DCMAKE_MAKE_PROGRAM={args.ninja}", "-DCMAKE_BUILD_TYPE=Release",
                        f"-DCMAKE_HIP_COMPILER={args.compiler}", f"-DPROBE_CCACHE={args.ccache}",
                        f"-DCMAKE_HIP_ARCHITECTURES={args.architectures}", f"-DROCM_PATH={args.rocm_path}",
                        f"-DLLAMINAR_SOURCE_ROOT={ROOT}", f"-DPROBE_SOURCE_ROOT={source}"],
                       env=environment, check=True, capture_output=True, text=True)
        output = None
        for stage, words in STAGES:
            if stage.startswith("warm"):
                # A forced rebuild must still obtain a real cache hit. Header
                # edits below leave the prior object in place to test Ninja.
                output.unlink()
            else:
                header.write_text("#error device-only invalid header\n" if words is None
                                  else f"constexpr unsigned kDeviceWords = {words};\n")
            log = root / f"{stage}.ccache.log"
            environment["CCACHE_LOGFILE"] = str(log)
            result = subprocess.run([args.cmake, "--build", str(build), "--parallel", "--target", "probe"],
                                    env=environment, capture_output=True, text=True)
            if output is None:
                objects = list(build.rglob("probe.hip.o"))
                if len(objects) != 1:
                    raise RuntimeError(f"cache probe did not create one object:\n{result.stdout}{result.stderr}")
                output = objects[0]
            item = {"stage": stage, "words": words, "returncode": result.returncode,
                    "object_present": output.exists(), "diagnostic": result.stdout + result.stderr,
                    "cache_results": re.findall(r"Result: ([^\n]+)", log.read_text()) if log.exists() else []}
            if result.returncode == 0:
                item["lds_bytes"] = native_lds(output, Path(args.compiler).parent, root)
                dependencies = subprocess.check_output([args.ninja, "-C", str(build), "-t", "deps",
                    str(output.relative_to(build))], text=True)
                item["device_header_dependency"] = str(header) in dependencies
            report["observations"].append(item)
            print(f"HIP device input replay: {stage}, exit={result.returncode}", flush=True)
        verify_report(report, args.compiler, args.architectures, args.rocm_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    pending = args.report.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(args.report)


def main():
    """Separate compiler-owned execution from compiler-free preflight verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("compile", "verify"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--rocm-path", required=True)
    parser.add_argument("--cmake")
    parser.add_argument("--ninja")
    parser.add_argument("--ccache")
    args = parser.parse_args()
    if args.mode == "compile":
        if not all((args.cmake, args.ninja, args.ccache)):
            parser.error("compile requires --cmake, --ninja and --ccache")
        compile_report(args)
    else:
        verify_report(json.loads(args.report.read_text()), args.compiler, args.architectures, args.rocm_path)
        print("HIP: all device-header/cache/dependency replay stages verified")


if __name__ == "__main__":
    main()
