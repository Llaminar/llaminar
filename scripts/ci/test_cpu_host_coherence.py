# @file test_cpu_host_coherence.py
# @brief Compile the actual host feature contract using real native compiler features.
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class HostCoherence(unittest.TestCase):
    def test_native_envelope_features(self):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if not compiler:
            self.fail('configured host compiler unavailable')
        arm = platform.machine().lower() in ('arm64', 'aarch64')
        x86 = platform.machine().lower() in ('x86_64', 'amd64')
        if not (arm or x86):
            self.skipTest('native architecture has no supported envelope')
        base = ['-std=c++23', '-fsyntax-only', '-I' + str(ROOT / 'src/v2')]
        cases = [('NEON', '-mcpu=generic -march=armv8-a+simd', dict(NEON=1, NEON_DOTPROD=0, NEON_I8MM=0, FP16FML=0, AVX2=0, AVX512=0)),
                 ('DotProd', '-mcpu=generic -march=armv8-a+simd+dotprod', dict(NEON=1, NEON_DOTPROD=1, NEON_I8MM=0, FP16FML=0, AVX2=0, AVX512=0)),
                 ('I8MM', '-mcpu=generic -march=armv8-a+simd+dotprod+i8mm+fp16fml', dict(NEON=1, NEON_DOTPROD=1, NEON_I8MM=1, FP16FML=1, AVX2=0, AVX512=0))] if arm else [
                 ('AVX2', '-mavx2 -mfma -mf16c', dict(NEON=0, NEON_DOTPROD=0, NEON_I8MM=0, FP16FML=0, AVX2=1, AVX512=0)),
                 ('AVX512', '-mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni', dict(NEON=0, NEON_DOTPROD=0, NEON_I8MM=0, FP16FML=0, AVX2=1, AVX512=1))]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'contract.cpp'
            source.write_text('#include "utils/CPUFeatures.h"\nstatic_assert(LLAMINAR_CPU_HOST_CODEGEN_PASS == 1);\n')
            def compile(flags, declarations):
                return subprocess.run([compiler] + base + flags.split() + ['-DLLAMINAR_COMPILED_WITH_' + key + '=' + str(value) for key, value in declarations.items()] + [str(source)], capture_output=True, text=True)
            for name, flags, declarations in cases:
                with self.subTest(envelope=name):
                    positive = compile(flags, declarations)
                    self.assertEqual(positive.returncode, 0, positive.stderr)
                    names = ('NEON', 'NEON_DOTPROD', 'NEON_I8MM', 'FP16FML') if arm else ('AVX512',)
                    for feature in names:
                        mutated = dict(declarations)
                        mutated[feature] = 1 - mutated[feature]
                        negative = compile(flags, mutated)
                        self.assertNotEqual(negative.returncode, 0, f'{name}: accepted mismatched {feature}')
            if x86:
                # AVX2's declared contract additionally requires real FMA and F16C.
                for flag in ('-mno-avx2', '-mno-fma', '-mno-f16c'):
                    result = compile(cases[0][1] + ' ' + flag, cases[0][2])
                    self.assertNotEqual(result.returncode, 0)

    def test_other_architecture_is_unqualified(self):
        if platform.machine().lower() in ('arm64', 'aarch64'):
            self.skipTest('x86 compiler qualification unavailable on native ARM runner')
        self.skipTest('ARM compiler qualification unavailable on native x86 runner')

    def test_coherence_has_one_pass_authority(self):
        header = (ROOT / 'src/v2/utils/CPUFeatures.h').read_text()
        block = header.split('#if LLAMINAR_CPU_HOST_CODEGEN_PASS', 1)[1].split('#endif // host compiler coherence', 1)[0]
        self.assertIn('LLAMINAR_COMPILED_WITH_NEON_I8MM', block)
        for exception in ('__CUDA_ARCH__', '__HIP_DEVICE_COMPILE__'):
            self.assertNotIn(exception, block)
        self.assertEqual(header.count('#if LLAMINAR_CPU_HOST_CODEGEN_PASS'), 1)
