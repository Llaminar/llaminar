# @file test_cpu_host_generated_commands.py
# @brief Configure real CMake language rules with simulated compilers; never qualify native codegen.
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from cpu_host_compile_contract import verify

ROOT = Path(__file__).resolve().parents[2]
# GPU compiler metadata is test-only simulation. CMake still owns every
# language compile rule. Forced C/CXX stubs do not qualify compiler behavior
# or meaningful try_compile execution; this fixture proves generation only.
PROJECT = r'''
cmake_minimum_required(VERSION 4.0)
project(HostEnvelope LANGUAGES C CXX)
foreach(lang CUDA HIP)
 if(lang STREQUAL CUDA)
  set(id NVIDIA)
  set(arch 80)
 else()
  set(id Clang)
  set(arch gfx906)
 endif()
 file(WRITE "${CMAKE_PLATFORM_INFO_DIR}/CMake${lang}Compiler.cmake" "set(CMAKE_${lang}_COMPILER ${SIMULATED_${lang}})\nset(CMAKE_${lang}_COMPILER_ID ${id})\nset(CMAKE_${lang}_COMPILER_VERSION 13.0)\nset(CMAKE_${lang}_COMPILER_LOADED TRUE)\nset(CMAKE_${lang}_COMPILER_WORKS TRUE)\nset(CMAKE_${lang}_COMPILER_FORCED TRUE)\nset(CMAKE_${lang}_STANDARD_COMPUTED_DEFAULT 17)\nset(CMAKE_${lang}_EXTENSIONS_COMPUTED_DEFAULT ON)\nset(CMAKE_${lang}_ARCHITECTURES ${arch})\n")
endforeach()
set(_CMAKE_HIP_DEVICE_RUNTIME_TARGET "")
enable_language(CUDA HIP)
set(CMAKE_SYSTEM_PROCESSOR "${TEST_PROCESSOR}")
include("${AUTHORITY}")
llaminar_configure_cpu_instruction_set()
if(TEST_COMMA)
 llaminar_cpu_host_compiler_options("-bad,token" cuda hip)
endif()
file(WRITE "${CMAKE_BINARY_DIR}/expected.json" "{\"flags\":\"${LLAMINAR_CPU_COMPILE_FLAGS}\",\"definitions\":\"${LLAMINAR_CPU_COMPILE_DEFINITIONS}\"}")
set_source_files_properties(fixture.cu PROPERTIES LANGUAGE CUDA)
set_source_files_properties(fixture.hip PROPERTIES LANGUAGE HIP)
add_library(envelope OBJECT fixture.c fixture.cpp fixture.cu fixture.hip)
'''


class GeneratedCommands(unittest.TestCase):
    def test_generated_language_matrix_and_original_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for language in ('C', 'CXX', 'CUDA', 'HIP'):
                compiler = root / ('compiler-' + language)
                compiler.write_text('#!/bin/sh\nexit 0\n')
                compiler.chmod(0o755)
            (root / 'CMakeLists.txt').write_text(PROJECT)
            for suffix in ('c', 'cpp', 'cu', 'hip'):
                (root / ('fixture.' + suffix)).write_text('int fixture;\n')
            authority = ROOT / 'cmake/CPUInstructionSet.cmake'
            original = root / 'original.cmake'
            # Exact pre-Q4 authority, not a synthetic missing-option command.
            original.write_bytes((ROOT / 'tests/v2/fixtures/cpu_host_compiler/CPUInstructionSet.pre-q4.cmake').read_bytes())
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), '3545ed22b920549840e01f85179ccec9a9ec3785694200a89a93778ef2de4543')
            compilers = {lang: str(root / ('compiler-' + lang)) for lang in ('C', 'CXX', 'CUDA', 'HIP')}
            for isa in ('NEON', 'NEON_DOTPROD', 'NEON_I8MM', 'AVX2', 'AVX512'):
                for config in ('Debug', 'Release', 'Integration'):
                    for negative in (False, True):
                        with self.subTest(isa=isa, config=config, original=negative):
                            build = root / f'{isa}-{config}-{negative}'
                            command = ['cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja', '-DCMAKE_MAKE_PROGRAM=' + shutil.which('ninja'), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', '-DCMAKE_BUILD_TYPE=' + config, '-DLLAMINAR_CPU_ISA=' + isa, '-DTEST_PROCESSOR=' + ('aarch64' if isa.startswith('NEON') else 'x86_64'), '-DAUTHORITY=' + str(original if negative else authority)]
                            for lang, compiler in compilers.items():
                                command += [f'-DSIMULATED_{lang}={compiler}']
                            for lang in ('C', 'CXX'):
                                command += [f'-DCMAKE_{lang}_COMPILER={compilers[lang]}', f'-DCMAKE_{lang}_COMPILER_ID=GNU', f'-DCMAKE_{lang}_COMPILER_FORCED=TRUE', f'-DCMAKE_{lang}_COMPILER_WORKS=TRUE']
                            result = subprocess.run(command, capture_output=True, text=True)
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            expected = json.loads((build / 'expected.json').read_text())
                            rows = json.loads((build / 'compile_commands.json').read_text())
                            self.assertEqual(len(rows), 4)
                            if negative:
                                with self.assertRaises(ValueError):
                                    verify(build / 'compile_commands.json', expected['flags'], expected['definitions'].split(';'), compilers)
                            else:
                                verify(build / 'compile_commands.json', expected['flags'], expected['definitions'].split(';'), compilers)

            comma_command = command.copy()
            comma_command[comma_command.index('-B') + 1] = str(root / 'comma-negative')
            comma_command = [('-DAUTHORITY=' + str(authority)) if token.startswith('-DAUTHORITY=') else token for token in comma_command]
            result = subprocess.run(comma_command + ['-DTEST_COMMA=ON'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nvcc separator', result.stderr)


# Real CMake generation through V2CPUSecondaryEnvelope.cmake: a dedicated
# subdirectory drops the canonical envelope and declares its own per target.
SECONDARY_PROJECT = r'''
cmake_minimum_required(VERSION 4.0)
project(SecondaryEnvelope LANGUAGES C CXX)
set(CMAKE_SYSTEM_PROCESSOR "${TEST_PROCESSOR}")
include("${MODULE}")
llaminar_configure_cpu_instruction_set()
add_compile_definitions(UNRELATED_DEFINITION=1)
add_executable(canonical fixture.cpp)
if(TEST_LATE)
 v2_cpu_secondary_envelope_contract_arguments(early)
endif()
if(TEST_ROOT_DECLARATION)
 v2_declare_cpu_secondary_envelope_target(canonical SCALAR)
endif()
add_subdirectory(secondary)
v2_cpu_secondary_envelope_contract_arguments(arguments)
string(JOIN "\n" content ${arguments})
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/contract-arguments.txt" CONTENT "${content}\n")
file(WRITE "${CMAKE_BINARY_DIR}/expected.json" "{\"flags\":\"${LLAMINAR_CPU_COMPILE_FLAGS}\",\"definitions\":\"${LLAMINAR_CPU_COMPILE_DEFINITIONS}\"}")
'''
SECONDARY_SUBDIRECTORY = r'''
include("${MODULE}")
v2_enter_cpu_secondary_envelope_scope()
foreach(isa IN LISTS SECONDARY_ISAS)
 add_executable(secondary_${isa} ../fixture.cpp)
 v2_declare_cpu_secondary_envelope_target(secondary_${isa} ${isa})
endforeach()
'''


class GeneratedSecondaryEnvelopes(unittest.TestCase):
    def test_secondary_scope_generates_one_coherent_envelope_per_target(self):
        from cpu_host_compile_contract import secondary_envelopes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compilers = {}
            for language in ('C', 'CXX'):
                compiler = root / ('compiler-' + language)
                compiler.write_text('#!/bin/sh\nexit 0\n')
                compiler.chmod(0o755)
                compilers[language] = str(compiler)
            (root / 'CMakeLists.txt').write_text(SECONDARY_PROJECT)
            (root / 'secondary').mkdir()
            (root / 'secondary' / 'CMakeLists.txt').write_text(SECONDARY_SUBDIRECTORY)
            (root / 'fixture.cpp').write_text('int main() { return 0; }\n')

            def configure(name, processor, isa, secondary_isas, *extra):
                command = ['cmake', '-S', str(root), '-B', str(root / name), '-G', 'Ninja',
                           '-DCMAKE_MAKE_PROGRAM=' + shutil.which('ninja'), '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                           '-DCMAKE_BUILD_TYPE=Release', '-DLLAMINAR_CPU_ISA=' + isa, '-DTEST_PROCESSOR=' + processor,
                           '-DMODULE=' + str(ROOT / 'tests/v2/cmake/V2CPUSecondaryEnvelope.cmake'),
                           '-DSECONDARY_ISAS=' + ';'.join(secondary_isas), *extra]
                for language, compiler in compilers.items():
                    command += [f'-DCMAKE_{language}_COMPILER={compiler}', f'-DCMAKE_{language}_COMPILER_ID=GNU',
                                f'-DCMAKE_{language}_COMPILER_FORCED=TRUE', f'-DCMAKE_{language}_COMPILER_WORKS=TRUE']
                return subprocess.run(command, capture_output=True, text=True)

            for isa, secondary_isas in (('AVX512', ['SCALAR', 'AVX2']), ('AVX2', ['SCALAR'])):
                with self.subTest(isa=isa):
                    result = configure(isa, 'x86_64', isa, secondary_isas)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    build = root / isa
                    expected = json.loads((build / 'expected.json').read_text())
                    groups = dict(targets=[], flags=[], definitions=[], objects=[])
                    for line in (build / 'contract-arguments.txt').read_text().splitlines():
                        option, value = line.split('=', 1)
                        groups[dict(zip(('--secondary-envelope', '--secondary-flags', '--secondary-definition', '--secondary-objects'),
                                        groups))[option]].append(value)
                    secondary = secondary_envelopes(*groups.values())
                    self.assertEqual(sorted(secondary), sorted('secondary_' + item for item in secondary_isas))
                    self.assertEqual(secondary['secondary_SCALAR'].flags, '-march=x86-64')
                    rows = json.loads((build / 'compile_commands.json').read_text())
                    self.assertEqual(len(rows), 1 + len(secondary_isas))
                    for row in rows:
                        # Unrelated directory definitions survive; no -mno-* lowering exists.
                        self.assertIn('-DUNRELATED_DEFINITION=1', row['command'])
                        self.assertNotIn('-mno-', row['command'])
                    database = build / 'compile_commands.json'
                    verify(database, expected['flags'], expected['definitions'].split(';'), compilers, secondary)
                    # Without its binding, every reduced row fails against the canonical envelope.
                    with self.assertRaisesRegex(ValueError, 'mismatched CPU envelope'):
                        verify(database, expected['flags'], expected['definitions'].split(';'), compilers)

            for name, processor, isa, secondary_isas, extra, message in (
                    ('late', 'x86_64', 'AVX512', ['SCALAR'], ['-DTEST_LATE=ON'], 'after the host compile contract'),
                    ('root', 'x86_64', 'AVX512', [], ['-DTEST_ROOT_DECLARATION=ON'], 'v2_enter_cpu_secondary_envelope_scope'),
                    ('arm-scalar', 'aarch64', 'NEON', ['SCALAR'], [], 'x86-64 only')):
                with self.subTest(negative=name):
                    result = configure(name, processor, isa, secondary_isas, *extra)
                    self.assertNotEqual(result.returncode, 0)
                    # CMake wraps diagnostics; compare whitespace-normalized text.
                    self.assertIn(message, ' '.join(result.stderr.split()))
