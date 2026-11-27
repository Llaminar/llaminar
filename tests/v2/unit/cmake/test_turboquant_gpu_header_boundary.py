#!/usr/bin/env python3
"""@file test_turboquant_gpu_header_boundary.py
@brief Compile the real shared TurboQuant header without host SIMD leaves.

CUDA/HIP translation units inherit the project's CPU ISA definitions in both
passes. Their device pass (__CUDA_ARCH__ / __HIP_DEVICE_COMPILE__) sees no CPU
SIMD flags, so CPUFeatures.h skips its host coherence checks; their host pass
receives the canonical CPU envelope flags (llaminar_cpu_host_compiler_options)
and must satisfy those checks. These device-free C++ compiler controls
reproduce both preprocessing contracts; they do not qualify a GPU compiler or
execution. The removed-guard control reproduces the original unresolved AVX
dispatch, and an incoherent host pass must still fail the coherence guard.
"""
import argparse
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[4]
HEADER = ROOT / "src/v2/kernels/cpu/turboquant/TurboQuantRotation.h"
ISA_CMAKE = ROOT / "cmake/CPUInstructionSet.cmake"
BOUNDARY = "#if !defined(__CUDACC__) && !defined(__HIPCC__)"
CXX = "c++"
# The macro each frontend defines in its device pass only.
DEVICE_PASS = {"__CUDACC__": "-D__CUDA_ARCH__=800", "__HIPCC__": "-D__HIP_DEVICE_COMPILE__=1"}
SOURCE = '#include "TurboQuantContext.h"\nstatic_assert(sizeof(llaminar2::TurboQuantContext) > 0);\n'
COHERENCE_ERROR = "Declared AVX2 envelope requires AVX2, FMA and F16C"


def x86_envelope_flags(avx512):
    """The canonical x86 envelope flags, read from the one resolver that defines them."""
    text = ISA_CMAKE.read_text(encoding="utf-8")
    base = re.findall(r'set\(flags "(-msse4\.1[^"]*)"\)', text)
    wide = re.findall(r'string\(APPEND flags " (-mavx512f[^"]*)"\)', text)
    if len(base) != 1 or len(wide) != 1:
        raise AssertionError(f"{ISA_CMAKE} no longer spells exactly one x86 AVX2 and AVX512 flag set")
    return base[0].split() + (wide[0].split() if avx512 else [])


def compile_source(source, frontend, avx512, device_pass=True, host_flags=()):
    """Parse production declarations under one GPU frontend pass and the selected x86 envelope.

    The device pass carries the envelope definitions but no CPU SIMD flags; a
    host pass carries exactly @p host_flags, which a coherent host pass must
    take from the envelope's canonical flags.
    """
    command = [CXX, "-std=c++20", "-fsyntax-only", "-x", "c++", "-",
               "-I", str(HEADER.parent), f"-D{frontend}=1",
               "-DLLAMINAR_COMPILED_WITH_AVX2=1",
               f"-DLLAMINAR_COMPILED_WITH_AVX512={int(avx512)}",
               "-DLLAMINAR_COMPILED_WITH_NEON=0",
               "-DLLAMINAR_COMPILED_WITH_NEON_DOTPROD=0",
               "-DLLAMINAR_COMPILED_WITH_NEON_I8MM=0",
               "-DLLAMINAR_COMPILED_WITH_FP16FML=0"]
    command += [DEVICE_PASS[frontend]] if device_pass else list(host_flags)
    return subprocess.run(command, input=source, text=True, capture_output=True,
                          check=False, timeout=30)


def compiler_targets_x86_64():
    """Whether the C++ compiler under test can run an x86 host pass at all."""
    result = subprocess.run([CXX, "-dM", "-E", "-x", "c++", "-"], input="", text=True,
                            capture_output=True, check=True, timeout=30)
    return "#define __x86_64__ 1" in result.stdout


class TurboQuantGPUHeaderBoundaryTests(unittest.TestCase):
    """Compile actual declarations; reject both the original bug and CPU calls."""

    def test_shared_context_parses_for_both_frontends_and_x86_envelopes(self):
        """Cold GPU context builders must not name uncompiled CPU ISA leaves (device pass)."""
        for frontend in ("__CUDACC__", "__HIPCC__"):
            for avx512 in (False, True):
                with self.subTest(frontend=frontend, avx512=avx512):
                    result = compile_source(SOURCE, frontend, avx512)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_context_parses_in_coherent_x86_host_passes(self):
        """A GPU TU's host pass with the envelope's canonical flags parses the shared context."""
        if not compiler_targets_x86_64():
            self.skipTest(f"{CXX} does not target x86-64; an x86 host pass cannot be compiled")
        for frontend in ("__CUDACC__", "__HIPCC__"):
            for avx512 in (False, True):
                with self.subTest(frontend=frontend, avx512=avx512):
                    result = compile_source(SOURCE, frontend, avx512, device_pass=False,
                                            host_flags=x86_envelope_flags(avx512))
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_incoherent_host_pass_still_fails_the_envelope_guard(self):
        """A host pass that declares AVX2 without its SIMD flags is rejected, not admitted as a device pass."""
        for frontend in ("__CUDACC__", "__HIPCC__"):
            with self.subTest(frontend=frontend):
                result = compile_source(SOURCE, frontend, False, device_pass=False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(COHERENCE_ERROR, result.stderr)

    def test_gpu_frontend_cannot_call_cpu_execution_entrypoints(self):
        """An accidental CPU dispatch from a GPU TU is rejected, not demoted."""
        for name in ("apply_rotation", "apply_rotation_transpose"):
            with self.subTest(entrypoint=name):
                source = ('#include "TurboQuantRotation.h"\n'
                          f'auto forbidden_cpu_entrypoint = &llaminar2::{name};\n')
                result = compile_source(source, "__CUDACC__", False)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(name, result.stderr)

    def test_original_unconditional_dispatch_is_rejected(self):
        """Remove only the new boundary; the production body must fail as in CI."""
        source = HEADER.read_text(encoding="utf-8")
        self.assertEqual(source.count(BOUNDARY), 1)
        start = source.index(BOUNDARY)
        end = source.index("\n#endif", start)
        source = source[:end] + source[end:].replace("\n#endif", "", 1)
        source = source.replace(BOUNDARY, "", 1)
        result = compile_source(source, "__CUDACC__", False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("apply_avx2", result.stderr)
        self.assertIn("transpose_avx2", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cxx", default=CXX)
    args, rest = parser.parse_known_args()
    CXX = args.cxx
    unittest.main(argv=[__file__, *rest])
