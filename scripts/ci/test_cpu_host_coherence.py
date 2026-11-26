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
        compiler = os.environ['LLAMINAR_TEST_CXX']
        standard = os.environ['LLAMINAR_TEST_CXX_STANDARD']
        if not standard:
            self.fail('configured CXX standard required')
        if not compiler:
            self.fail('configured host compiler unavailable')
        arm = platform.machine().lower() in ('arm64', 'aarch64')
        x86 = platform.machine().lower() in ('x86_64', 'amd64')
        if not (arm or x86):
            self.fail('native architecture has no supported envelope')
        base = ['-std=c++' + standard, '-fsyntax-only', '-I' + str(ROOT / 'src/v2')]
        cases = []
        with tempfile.TemporaryDirectory() as resolver_directory:
            script = Path(resolver_directory) / 'resolve.cmake'
            output = Path(resolver_directory) / 'resolved.txt'
            script.write_text('include("' + str(ROOT / 'cmake/CPUInstructionSet.cmake') + '")\n' +
                'foreach(isa ' + ('NEON NEON_DOTPROD NEON_I8MM' if arm else 'AVX2 AVX512') + ')\n' +
                'llaminar_cpu_instruction_set("' + ('aarch64' if arm else 'x86_64') + '" "${isa}" flags definitions)\n' +
                'file(APPEND "' + str(output) + '" "${isa}|${flags}|${definitions}\\n")\nendforeach()\n')
            subprocess.run([os.environ['LLAMINAR_TEST_CMAKE'], '-P', str(script)], check=True)
            for row in output.read_text().splitlines():
                name, flags, definitions = row.split('|')
                declarations = {token.split('=')[0].removeprefix('LLAMINAR_COMPILED_WITH_'): int(token.split('=')[1]) for token in definitions.split(';')}
                if not flags or not declarations:
                    self.fail('empty canonical envelope')
                cases.append((name, flags, declarations))
        import hashlib
        print('QUALIFICATION host-only ' + platform.machine() + ' compiler=' + compiler + ' sha256=' + hashlib.sha256(Path(compiler).read_bytes()).hexdigest() + ' version=' + subprocess.check_output([compiler, '--version'], text=True).splitlines()[0])
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

    def test_coherence_has_one_pass_authority(self):
        header = (ROOT / 'src/v2/utils/CPUFeatures.h').read_text()
        block = header.split('#if LLAMINAR_CPU_HOST_CODEGEN_PASS', 1)[1].split('#endif // host compiler coherence', 1)[0]
        self.assertIn('LLAMINAR_COMPILED_WITH_NEON_I8MM', block)
        for exception in ('__CUDA_ARCH__', '__HIP_DEVICE_COMPILE__'):
            self.assertNotIn(exception, block)
        self.assertEqual(header.count('#if LLAMINAR_CPU_HOST_CODEGEN_PASS'), 1)
