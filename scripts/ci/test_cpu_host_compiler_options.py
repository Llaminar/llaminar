# @file test_cpu_host_compiler_options.py
# @brief Exercise canonical ISA host forwarding without requiring a GPU compiler.
import pathlib
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class CPUHostOptions(unittest.TestCase):
    def test_all_envelopes_and_separator_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            script = pathlib.Path(directory) / 'check.cmake'
            script.write_text(f'include("{ROOT}/cmake/CPUInstructionSet.cmake")\n' + '''
foreach(isa NEON NEON_DOTPROD NEON_I8MM AVX2 AVX512)
 if(isa MATCHES "^NEON")
  set(processor aarch64)
 else()
  set(processor x86_64)
 endif()
 llaminar_cpu_instruction_set("${processor}" "${isa}" flags definitions)
 llaminar_cpu_host_compiler_options("${flags}" cuda hip)
 separate_arguments(tokens NATIVE_COMMAND "${flags}")
 foreach(token IN LISTS tokens)
  string(FIND "${cuda}" "-Xcompiler=${token}" found)
  if(found LESS 0)
   message(FATAL_ERROR "missing CUDA host option")
  endif()
  string(FIND "${hip}" "-Xarch_host ${token}" found)
  if(found LESS 0)
   message(FATAL_ERROR "missing HIP host option")
  endif()
 endforeach()
endforeach()
''')
            subprocess.run([shutil.which('cmake'), '-P', str(script)], check=True)
            script.write_text(script.read_text() + '\nllaminar_cpu_host_compiler_options("-bad,token" cuda hip)\n')
            result = subprocess.run([shutil.which('cmake'), '-P', str(script)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('nvcc separator', result.stderr)

    def test_compile_database_original_negative(self):
        import json
        from cpu_host_compile_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            database = pathlib.Path(directory) / 'compile_commands.json'
            base = ['nvcc', '-DLLAMINAR_COMPILED_WITH_NEON=1', '-c', 'test.cu']
            database.write_text(json.dumps([dict(file='test.cu', host_language='CUDA', arguments=base)]))
            with self.assertRaisesRegex(ValueError, 'missing host option'):
                verify(database, '-mcpu=generic -march=armv8-a+simd', ['LLAMINAR_COMPILED_WITH_NEON=1'])
            database.write_text(json.dumps([dict(file='test.cu', host_language='CUDA', arguments=base + ['-Xcompiler=-mcpu=generic', '-Xcompiler=-march=armv8-a+simd'])]))
            verify(database, '-mcpu=generic -march=armv8-a+simd', ['LLAMINAR_COMPILED_WITH_NEON=1'])

    def test_host_language_and_declaration_failures(self):
        import json
        from cpu_host_compile_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            database = pathlib.Path(directory) / 'commands.json'
            definition = 'LLAMINAR_COMPILED_WITH_NEON=1'
            for language, options in [('C', ['-mcpu=generic']), ('CXX', ['-mcpu=generic']), ('CUDA', ['-Xcompiler=-mcpu=generic']), ('HIP', ['-Xarch_host', '-mcpu=generic'])]:
                command = ['custom-wrapper', '-D' + definition] + options
                row = dict(file='fixture', host_language=language, arguments=command)
                database.write_text(json.dumps([row]))
                verify(database, '-mcpu=generic', [definition])
                for bad in (command[:-1], command + ['-DLLAMINAR_COMPILED_WITH_NEON=0']):
                    database.write_text(json.dumps([dict(row, arguments=bad)]))
                    with self.assertRaises(ValueError):
                        verify(database, '-mcpu=generic', [definition])
            with self.assertRaisesRegex(ValueError, 'empty expected'):
                verify(database, '')

    def test_host_codegen_predicate_scope(self):
        header = (ROOT / 'src/v2/utils/CPUFeatures.h').read_text()
        self.assertEqual(header.count('#define LLAMINAR_CPU_HOST_CODEGEN_PASS 1'), 1)
        self.assertEqual(header.count('#define LLAMINAR_CPU_HOST_CODEGEN_PASS 0'), 1)
        start = header.index('#if LLAMINAR_CPU_HOST_CODEGEN_PASS')
        end = header.index('#endif // host compiler coherence', start)
        block = header[start:end]
        self.assertIn('LLAMINAR_COMPILED_WITH_NEON_I8MM', block)
        self.assertNotIn('__CUDA_ARCH__', block)
        self.assertNotIn('__HIP_DEVICE_COMPILE__', block)

    def test_forwarded_non_isa_options_and_contrary_isa(self):
        import json
        from cpu_host_compile_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            database = pathlib.Path(directory) / 'commands.json'
            definition = 'LLAMINAR_COMPILED_WITH_AVX2=1'
            base = ['nvcc', '-D' + definition, '-Xcompiler=-mavx2']
            def check(extra):
                database.write_text(json.dumps([dict(file='fixture.cu', host_language='CUDA', arguments=base + extra)]))
                verify(database, '-mavx2', [definition])
            for extra in (['-Xcompiler=-fopenmp'], ['-Xcompiler=-O3,-Wno-unused,-fvisibility=hidden'], ['-Xcompiler', '-fopenmp'], ['-Xcompiler=-m64']):
                check(extra)
            for extra in (['-Xcompiler=-march=native'], ['-Xcompiler=-O3,-mavx512f'], ['-Xcompiler', '-mavx512f'], ['--compiler-options=-mgfni'], ['--compiler-options', '-mno-lzcnt'], ['-Xcompiler=-mgeneral-regs-only'], ['-Xarch_device', '-mavx2']):
                with self.assertRaisesRegex(ValueError, 'contrary|leaks to device'):
                    check(extra)
            with self.assertRaisesRegex(ValueError, 'empty expected CPU definitions'):
                verify(database, '-mavx2', [])

    def test_shared_clang_identity_and_hip_contrary_option(self):
        import json
        from cpu_host_compile_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            database = pathlib.Path(directory) / 'commands.json'
            compiler = '/usr/bin/clang'
            definitions = ['LLAMINAR_COMPILED_WITH_AVX2=1']
            command = [compiler, '-x', 'hip', '-D' + definitions[0], '-Xarch_host', '-mavx2']
            row = dict(file='LANGUAGEHIP.cpp', arguments=command)
            compilers = dict(C=compiler, CXX=compiler, HIP=compiler)
            database.write_text(json.dumps([row]))
            verify(database, '-mavx2', definitions, compilers)
            row['arguments'] += ['-Xarch_host', '-mno-avx512f']
            database.write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, 'contrary|leaks to device'):
                verify(database, '-mavx2', definitions, compilers)

    def test_exact_hip_device_options(self):
        import json
        from cpu_host_compile_contract import verify
        with tempfile.TemporaryDirectory() as directory:
            database = pathlib.Path(directory) / 'commands.json'
            definition = 'LLAMINAR_COMPILED_WITH_AVX2=1'
            base = ['clang++', '-D' + definition, '-Xarch_host', '-mavx2']
            def check(language, extra):
                database.write_text(json.dumps([dict(file='fixture.cpp', host_language=language, arguments=base + extra)]))
                verify(database, '-mavx2', [definition])
            check('HIP', ['-mwavefrontsize64', '-Xarch_device', '-mllvm=-amdgpu-spill-cfi-saved-regs=false'])
            for forwarded in (['-Xarch_host', '-mwavefrontsize64'], ['-Xcompiler=-mwavefrontsize64'], ['--compiler-options', '-mwavefrontsize64']):
                with self.assertRaisesRegex(ValueError, 'contrary'):
                    check('HIP', ['-mwavefrontsize64'] + forwarded)
            with self.assertRaisesRegex(ValueError, 'contrary'):
                check('HIP', ['-Xarch_device', '-mllvm=-amdgpu-spill-cfi-saved-regs=false', '-Xarch_host', '-mllvm=-amdgpu-spill-cfi-saved-regs=false'])
            for language, extra in [('CXX', ['-mwavefrontsize64']), ('HIP', ['-Xarch_device', '-mavx2']), ('HIP', ['-m32']), ('HIP', ['-Xarch_host', '-mwavefrontsize64']), ('HIP', ['-Xarch_device', '-mllvm=unknown'])]:
                with self.assertRaises(ValueError):
                    check(language, extra)


class CPUHostSecondaryEnvelopes(unittest.TestCase):
    """Bind reduced-ISA test rows to their own declared envelope, and only those rows."""

    CANONICAL_FLAGS = '-msse4.1 -mavx -mavx2 -mfma -mf16c -mbmi -mbmi2 -mpopcnt -mavx512f -mavx512bw -mavx512dq -mavx512vl -mavx512vnni'
    CANONICAL = ['LLAMINAR_COMPILED_WITH_AVX2=1', 'LLAMINAR_COMPILED_WITH_AVX512=1', 'LLAMINAR_COMPILED_WITH_NEON=0',
                 'LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0', 'LLAMINAR_COMPILED_WITH_NEON_I8MM=0', 'LLAMINAR_COMPILED_WITH_FP16FML=0']
    SCALAR = ['LLAMINAR_COMPILED_WITH_AVX2=0', 'LLAMINAR_COMPILED_WITH_AVX512=0', 'LLAMINAR_COMPILED_WITH_NEON=0',
              'LLAMINAR_COMPILED_WITH_NEON_DOTPROD=0', 'LLAMINAR_COMPILED_WITH_NEON_I8MM=0', 'LLAMINAR_COMPILED_WITH_FP16FML=0']

    def check(self, directory, rows, secondary):
        import json
        from cpu_host_compile_contract import verify
        database = pathlib.Path(directory) / 'compile_commands.json'
        database.write_text(json.dumps(rows))
        verify(database, self.CANONICAL_FLAGS, self.CANONICAL, secondary=secondary)

    def rows(self, directory, scalar_options, scalar_definitions, scalar_output='secondary/scalar.dir/t.cpp.o'):
        canonical = ['c++'] + ['-D' + item for item in self.CANONICAL] + self.CANONICAL_FLAGS.split() + ['-c', 't.cpp']
        scalar = ['c++'] + ['-D' + item for item in scalar_definitions] + scalar_options + ['-c', 't.cpp']
        return [dict(directory=directory, file='t.cpp', output='canonical.dir/t.cpp.o', host_language='CXX', arguments=canonical),
                dict(directory=directory, file='t.cpp', output=scalar_output, host_language='CXX', arguments=scalar)]

    def scalar(self, directory, flags='-march=x86-64', definitions=None):
        from cpu_host_compile_contract import SecondaryEnvelope
        return {'v2_test_scalar': SecondaryEnvelope(flags, tuple(definitions or self.SCALAR),
                                                    (directory + '/secondary/./scalar.dir/t.cpp.o',))}

    def test_declared_secondary_row_matches_its_own_envelope(self):
        with tempfile.TemporaryDirectory() as directory:
            self.check(directory, self.rows(directory, ['-march=x86-64'], self.SCALAR), self.scalar(directory))

    def test_secondary_row_with_wrong_flags_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            # The former reduced-ISA rule: canonical flags lowered with -mno-*.
            lowered = self.CANONICAL_FLAGS.split() + ['-mno-avx', '-mno-avx2', '-mno-avx512f', '-mno-fma']
            for options in (lowered, ['-march=x86-64', '-mavx2'], []):
                with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'contrary|missing host option'):
                    self.check(directory, self.rows(directory, options, self.SCALAR), self.scalar(directory))

    def test_undeclared_reduced_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = self.rows(directory, ['-march=x86-64'], self.SCALAR, scalar_output='other/scalar.dir/t.cpp.o')
            with self.assertRaisesRegex(ValueError, 'mismatched CPU envelope'):
                self.check(directory, rows, None)
            with self.assertRaisesRegex(ValueError, 'mismatched CPU envelope'):
                self.check(directory, rows, self.scalar(directory))

    def test_contradictory_definitions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            # The former reduced-ISA rule inherited AVX2=1 alongside its own values.
            contradictory = self.SCALAR + ['LLAMINAR_COMPILED_WITH_AVX2=1']
            with self.assertRaisesRegex(ValueError, 'contradictory CPU definition'):
                self.check(directory, self.rows(directory, ['-march=x86-64'], contradictory), self.scalar(directory))
            with self.assertRaisesRegex(ValueError, 'mismatched CPU envelope'):
                self.check(directory, self.rows(directory, ['-march=x86-64'], self.CANONICAL), self.scalar(directory))

    def test_incomplete_or_unbound_declarations_are_rejected(self):
        from cpu_host_compile_contract import secondary_envelopes
        with tempfile.TemporaryDirectory() as directory:
            rows = self.rows(directory, ['-march=x86-64'], self.SCALAR)
            with self.assertRaisesRegex(ValueError, 'every CPU definition'):
                self.check(directory, rows, self.scalar(directory, definitions=self.SCALAR[:2]))
            missing = self.scalar(directory)
            missing['v2_test_absent'] = missing['v2_test_scalar'].__class__('-march=x86-64', tuple(self.SCALAR), (directory + '/absent.o',))
            with self.assertRaisesRegex(ValueError, 'compiled 0 times'):
                self.check(directory, rows, missing)
            with self.assertRaisesRegex(ValueError, 'compiled 2 times'):
                self.check(directory, rows + rows[1:], self.scalar(directory))
        definitions = ['t=' + item for item in self.SCALAR]
        parsed = secondary_envelopes(['t'], ['t=-march=x86-64'], definitions, ['t=/b/a.o|/b/c.o'])
        self.assertEqual(parsed['t'].flags, '-march=x86-64')
        self.assertEqual(parsed['t'].objects, ('/b/a.o', '/b/c.o'))
        for arguments in ((['t'], [], definitions, ['t=/b/a.o']),
                          (['t'], ['t=-march=x86-64'], [], ['t=/b/a.o']),
                          (['t'], ['t=-march=x86-64'], definitions, []),
                          (['t'], ['u=-march=x86-64'], definitions, ['t=/b/a.o']),
                          (['t', 't'], ['t=-march=x86-64'], definitions, ['t=/b/a.o'])):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                secondary_envelopes(*arguments)
