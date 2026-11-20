#!/usr/bin/env python3
"""@file test_portability_build_types.py
@brief Exercise genuine CMake per-configuration cache initialization without a runtime compiler SDK.

This fixture invokes the real CMake initialization function used by compiler
language setup; it does not synthesize compiler execution evidence. The separate
CPUCompilerConfiguration executable compiles and executes real C/C++ flag
witnesses during ordinary builds/tests, including installed test runners. No
runtime compiler, executable temporary mount or writable source tree is needed.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE = ROOT / 'tests/v2/cmake/V2PortabilityBuildTypes.cmake'
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--cmake', required=True)
parser.add_argument('--ninja', required=True)
OPTIONS, REST = parser.parse_known_args()


class PortabilityBuildTypesTests(unittest.TestCase):
    """Actual CMake cache behavior, complementary to the separately compiled C/C++ witness."""

    def setUp(self):
        """Own every generated source/build artifact in one lexical temporary root."""
        self.temporary = tempfile.TemporaryDirectory(prefix='llaminar-build-types-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        (self.source / 'CMakeLists.txt').write_text(f'''# @file CMakeLists.txt
# @brief Exercise the real per-config cache initializer, without loading compiler SDKs.
cmake_minimum_required(VERSION 3.20)
include("{MODULE.as_posix()}")
project(IntegrationFlagCache LANGUAGES NONE)
include(CMakeInitializeConfigs)
cmake_initialize_per_config_variable(CMAKE_C_FLAGS "C compiler flags")
cmake_initialize_per_config_variable(CMAKE_CXX_FLAGS "CXX compiler flags")
v2_validate_portability_integration_flags()
file(WRITE "${{CMAKE_BINARY_DIR}}/observed-flags.txt"
    "C=${{CMAKE_C_FLAGS_INTEGRATION}}\\nCXX=${{CMAKE_CXX_FLAGS_INTEGRATION}}\\n")
''', encoding='utf-8')

    def configure(self, build, *extra):
        """Run the exact selected CMake/Ninja with caller-owned cache overrides."""
        return subprocess.run([
            OPTIONS.cmake, '-S', str(self.source), '-B', str(build), '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Integration',
            f'-DCMAKE_MAKE_PROGRAM:FILEPATH={OPTIONS.ninja}', *extra,
        ], text=True, capture_output=True, check=False)

    def observed_flags(self, build):
        """Read values actually published by CMake, not a source-text approximation."""
        return dict(line.split('=', 1) for line in
                    (build / 'observed-flags.txt').read_text(encoding='utf-8').splitlines())

    def test_c_and_cxx_defaults_precede_real_cache_initialization(self):
        """The old after-project CACHE assignment cannot initialize an existing empty entry."""
        build = self.root / 'default'
        configured = self.configure(build)
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        self.assertEqual(self.observed_flags(build), {'C': '-O3 -g -DNDEBUG', 'CXX': '-O3 -g -DNDEBUG'})

    def test_explicit_compiler_envelope_is_not_forced_or_rewritten(self):
        """Nonempty caller policy retains both its order and exact authored values."""
        build = self.root / 'authored'
        c_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_AUTHORED_C=1'
        cpp_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_AUTHORED_CXX=1'
        configured = self.configure(build, f'-DCMAKE_C_FLAGS_INTEGRATION:STRING={c_flags}',
                                    f'-DCMAKE_CXX_FLAGS_INTEGRATION:STRING={cpp_flags}')
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        self.assertEqual(self.observed_flags(build), {'C': c_flags, 'CXX': cpp_flags})

    def test_authored_initializers_are_not_shadowed(self):
        """A cache-authored _INIT is caller policy, not a slot for unconditional defaults."""
        build = self.root / 'authored-init'
        c_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_AUTHORED_INIT_C=1'
        cpp_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_AUTHORED_INIT_CXX=1'
        configured = self.configure(build, f'-DCMAKE_C_FLAGS_INTEGRATION_INIT:STRING={c_flags}',
                                    f'-DCMAKE_CXX_FLAGS_INTEGRATION_INIT:STRING={cpp_flags}')
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        self.assertEqual(self.observed_flags(build), {'C': c_flags, 'CXX': cpp_flags})

    def test_normal_initializers_are_not_shadowed(self):
        """A pre-include normal initializer also survives real per-configuration initialization."""
        build = self.root / 'normal-init'
        c_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_NORMAL_INIT_C=1'
        cpp_flags = '-O3 -g0 -DNDEBUG -DLLAMINAR_NORMAL_INIT_CXX=1'
        path = self.source / 'CMakeLists.txt'
        include = f'include("{MODULE.as_posix()}")'
        text = path.read_text(encoding='utf-8')
        path.write_text(text.replace(include,
            f'set(CMAKE_C_FLAGS_INTEGRATION_INIT "{c_flags}")\n'
            f'set(CMAKE_CXX_FLAGS_INTEGRATION_INIT "{cpp_flags}")\n' + include, 1), encoding='utf-8')
        configured = self.configure(build)
        self.assertEqual(configured.returncode, 0, configured.stdout + configured.stderr)
        self.assertEqual(self.observed_flags(build), {'C': c_flags, 'CXX': cpp_flags})

    def test_empty_inherited_cache_is_rejected_not_silently_qualified(self):
        """Both languages reject the actual old empty-cache state without compiling anything."""
        for language in ('C', 'CXX'):
            with self.subTest(language=language):
                build = self.root / ('empty-' + language)
                configured = self.configure(build, f'-DCMAKE_{language}_FLAGS_INTEGRATION:STRING=')
                self.assertNotEqual(configured.returncode, 0)
                self.assertIn(f'empty {language} compiler-flag cache', configured.stdout + configured.stderr)
                self.assertFalse((build / 'observed-flags.txt').exists())


if __name__ == '__main__':
    unittest.main(argv=[__file__, *REST])
