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
# language compile rule and runs its ordinary C/CXX try_compile machinery.
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
