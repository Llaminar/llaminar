#!/usr/bin/env python3
"""Exercise production CMake spill enforcement with real compilers, without GPUs."""

import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    """Both optimized build types reject spills and accept legitimate local arrays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("CUDA", "HIP"), required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architectures", required=True)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ninja", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="llaminar-spill-probe-") as directory:
        for config in ("Release", "Integration", "Debug"):
            build = Path(directory) / config
            subprocess.run([args.cmake, "-S", str(Path(__file__).parent), "-B", str(build),
                            "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={args.ninja}",
                            f"-DPROBE_LANGUAGE={args.language}", f"-DCMAKE_BUILD_TYPE={config}",
                            f"-DCMAKE_{args.language}_COMPILER={args.compiler}",
                            f"-DCMAKE_{args.language}_ARCHITECTURES={args.architectures}"], check=True)
            variants = ("clean", "private", "spill", "helper", "clean_helper")
            if config != "Debug":
                # Inline scalar-register constraints need optimized HIP
                # uniformity analysis. Debug exemption is proved by the actual
                # spilling kernel and helper above, not this pressure witness.
                variants += ("scalar", "mixed")
            for variant in variants:
                result = subprocess.run([args.cmake, "--build", str(build), "--parallel",
                                         "--target", f"probe_{variant}"], text=True, capture_output=True)
                diagnostic = result.stdout + result.stderr
                if config != "Debug" and variant in ("spill", "helper"):
                    if result.returncode == 0 or "spill" not in diagnostic.lower():
                        raise RuntimeError(f"{config}: spilling kernel was not rejected:\n{diagnostic}")
                    if args.language == "HIP" and "register spills are forbidden" not in diagnostic:
                        raise RuntimeError(f"{config}: HIP failed for the wrong reason:\n{diagnostic}")
                    if args.language == "CUDA" and "Registers are spilled" not in diagnostic:
                        raise RuntimeError(f"{config}: CUDA failed for the wrong reason:\n{diagnostic}")
                    if list((build / f"CMakeFiles/probe_{variant}.dir").glob("*.o")):
                        raise RuntimeError("failed spill compilation retained a linkable object")
                elif result.returncode:
                    raise RuntimeError(f"{config}: {variant} unexpectedly failed:\n{diagnostic}")
                if args.language == "HIP" and variant in ("scalar", "mixed"):
                    if "scalar-to-vector register moves; no memory spill" not in diagnostic:
                        raise RuntimeError(f"{config}: scalar witness did not exercise register moves:\n{diagnostic}")
                print(f"{args.language} {config} {variant}: expected outcome verified", flush=True)
                if config != "Debug":
                    # A second failed build reads the same compiler result from
                    # ccache; a successful object's removal also exercises a
                    # hit. Only this private temporary project's output is retired.
                    for output in (build / f"CMakeFiles/probe_{variant}.dir").glob("*.o"):
                        output.unlink()
                    cached = subprocess.run([args.cmake, "--build", str(build), "--parallel",
                                             "--target", f"probe_{variant}"],
                                            text=True, capture_output=True)
                    if (cached.returncode == 0) != (result.returncode == 0):
                        raise RuntimeError(f"{config}: cached {variant} changed the guard outcome:\n"
                                           + cached.stdout + cached.stderr)


if __name__ == "__main__":
    main()
