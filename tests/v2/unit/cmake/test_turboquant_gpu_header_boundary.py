#!/usr/bin/env python3
"""@file test_turboquant_gpu_header_boundary.py
@brief Compile the real shared TurboQuant header without host SIMD leaves.

CUDA/HIP translation units inherit the project's CPU ISA definitions, not the
CPU compiler's SIMD flags. These device-free C++ compiler controls reproduce
that preprocessing contract; they do not qualify a GPU compiler or execution.
The removed-guard control reproduces the original unresolved AVX dispatch.
"""
import argparse
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[4]
HEADER = ROOT / "src/v2/kernels/cpu/turboquant/TurboQuantRotation.h"
BOUNDARY = "#if !defined(__CUDACC__) && !defined(__HIPCC__)"
CXX = "c++"


def compile_source(source, frontend, avx512):
    """Parse production declarations with the selected envelope but no SIMD flags."""
    command = [CXX, "-std=c++20", "-fsyntax-only", "-x", "c++", "-",
               "-I", str(HEADER.parent), f"-D{frontend}=1",
               "-DLLAMINAR_COMPILED_WITH_AVX2=1",
               f"-DLLAMINAR_COMPILED_WITH_AVX512={int(avx512)}",
               "-DLLAMINAR_COMPILED_WITH_NEON=0",
               "-DLLAMINAR_COMPILED_WITH_NEON_DOTPROD=0",
               "-DLLAMINAR_COMPILED_WITH_NEON_I8MM=0"]
    return subprocess.run(command, input=source, text=True, capture_output=True,
                          check=False, timeout=30)


class TurboQuantGPUHeaderBoundaryTests(unittest.TestCase):
    """Compile actual declarations; reject both the original bug and CPU calls."""

    def test_shared_context_parses_for_both_frontends_and_x86_envelopes(self):
        """Cold GPU context builders must not name uncompiled CPU ISA leaves."""
        source = '#include "TurboQuantContext.h"\nstatic_assert(sizeof(llaminar2::TurboQuantContext) > 0);\n'
        for frontend in ("__CUDACC__", "__HIPCC__"):
            for avx512 in (False, True):
                with self.subTest(frontend=frontend, avx512=avx512):
                    result = compile_source(source, frontend, avx512)
                    self.assertEqual(result.returncode, 0, result.stderr)

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
