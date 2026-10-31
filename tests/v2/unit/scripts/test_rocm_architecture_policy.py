#!/usr/bin/env python3
"""Replay actual architecture selection, CTest environment and BLAS closure admission.

No compiler or accelerator is invoked. CMake configures language-free fixtures;
the production add_v2_test body is loaded verbatim so inherited ISA spoofing is
checked in the launched child, rather than inferred from registration text.
"""
import hashlib
import json
import importlib.util
import sys
from unittest import mock
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]


def run(args, **kwargs):
    """Execute a bounded device-free command and retain its diagnostic output."""
    return subprocess.run(args, text=True, capture_output=True, timeout=20, **kwargs)


class ROCmArchitecturePolicyTests(unittest.TestCase):
    """Prove native identity and complete selected dependency target coverage."""

    def test_default_explicit_and_environment_targets(self):
        """CMake cache and HIPARCHS win over the common shipping default."""
        for explicit, env_value, expected in [(None, None, 'gfx906;gfx1100'),
                ('gfx1100', None, 'gfx1100'), ('gfx1100;gfx906', None, 'gfx1100;gfx906'),
                (None, 'gfx1100', 'gfx1100'), ('gfx906', 'gfx1100', 'gfx906'),
                # Selection is extensible; this language-free replay does not
                # claim compiler, native instruction or runtime qualification.
                ('gfx90a;gfx942', None, 'gfx90a;gfx942'),
                (None, 'gfx1030;gfx1201', 'gfx1030;gfx1201')]:
            with self.subTest(explicit=explicit, env=env_value), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp)/'src';source.mkdir()
                (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.27)\nproject(Policy NONE)\n'
                    f'include("{ROOT}/cmake/ROCmArchitectures.cmake")\n'
                    'file(WRITE "${CMAKE_BINARY_DIR}/targets.txt" "${CMAKE_HIP_ARCHITECTURES}\\n${RCCL_GPU_TARGETS}")\n')
                env = dict(os.environ);env.pop('HIPARCHS', None)
                if env_value: env['HIPARCHS']=env_value
                args=['cmake','-S',str(source),'-B',str(Path(tmp)/'build'),'-G','Ninja']
                if explicit: args.append('-DCMAKE_HIP_ARCHITECTURES='+explicit)
                result=run(args,env=env);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual((Path(tmp)/'build/targets.txt').read_text(),expected+'\n'+expected)

    def test_actual_rocm_configuration_binds_before_language_discovery(self):
        """Execute the production ROCm block, intercepting only SDK discovery."""
        text=(ROOT/'src/v2/CMakeLists.txt').read_text()
        start=text.index('if(HAVE_ROCM)', text.index('# ROCm Backend Configuration'))
        end=text.index('# Backend Library Compilation',start)
        block=text[start:end].rsplit('# =============================================================================',1)[0]
        for explicit in (None,'gfx1100'):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as tmp:
                base=Path(tmp); source=base/'src/v2'; source.mkdir(parents=True)
                (base/'cmake').symlink_to(ROOT/'cmake',target_is_directory=True)
                (base/'scripts').symlink_to(ROOT/'scripts',target_is_directory=True)
                (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.27)\nproject(ActualBlock NONE)\n'
                    'set(HAVE_ROCM ON)\nmacro(enable_language)\nset(AT_DISCOVERY "${CMAKE_HIP_ARCHITECTURES}")\nendmacro()\n'
                    'macro(find_package)\nendmacro()\n'+block+'\n'
                    'get_directory_property(HIP_OPTIONS COMPILE_OPTIONS)\n'
                    'file(WRITE "${CMAKE_BINARY_DIR}/options.txt" "${HIP_OPTIONS}")\n'
                    'file(WRITE "${CMAKE_BINARY_DIR}/targets.txt" "${AT_DISCOVERY}\\n${CMAKE_HIP_ARCHITECTURES}")\n')
                env=dict(os.environ);env.pop('HIPARCHS',None)
                args=['cmake','-S',str(source),'-B',str(base/'build'),'-G','Ninja']
                if explicit:args.append('-DCMAKE_HIP_ARCHITECTURES='+explicit)
                result=run(args,env=env)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                expected=explicit or 'gfx906;gfx1100'
                self.assertEqual((base/'build/targets.txt').read_text(),expected+'\n'+expected)
                self.assertIn('$<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>',
                    (base/'build/options.txt').read_text())

    def test_actual_nested_rccl_targets_and_reuse_identity(self):
        """Production argv retains both targets and invalidates an old-target DSO."""
        text=(ROOT/'src/v2/CMakeLists.txt').read_text()
        start=text.index('        message(STATUS "V2: NOTE: This may take')
        end=text.index('        # Configure RCCL',start)
        configure=text[start:end]
        start=text.index('    set(RCCL_BUILD_ID "')
        end=text.index('    # Check if RCCL build is up-to-date',start)
        identity=text[start:end]
        ids=[]
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            probe=base/'argv.py';probe.write_text('import sys,json;print(json.dumps(sys.argv[1:]))')
            for targets in ('gfx906','gfx906;gfx1100'):
                script=base/'probe.cmake'
                script.write_text(f'set(RCCL_GPU_TARGETS "{targets}")\n'+identity+configure+
                    f'execute_process(COMMAND "{sys.executable}" "{probe}" ${{RCCL_CONFIG_ARGS}} OUTPUT_VARIABLE ARGV)\n'
                    f'file(WRITE "{base}/argv.json" "${{ARGV}}")\n'
                    f'file(WRITE "{base}/identity.txt" "${{RCCL_BUILD_ID}}")\n')
                result=run(['cmake','-P',str(script)])
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                argv=json.loads((base/'argv.json').read_text())
                self.assertIn('-DGPU_TARGETS='+targets,argv)
                self.assertIn('-DGPU_BUILD_TARGETS='+targets,argv)
                self.assertNotIn('gfx1100',argv)
                ids.append((base/'identity.txt').read_text())
        self.assertNotEqual(*ids)

    def test_wave32_is_rejected_before_hip_discovery(self):
        """A caller cannot silently change fixed-order wave64 production arithmetic."""
        for variable in ('CMAKE_HIP_FLAGS','CMAKE_HIP_FLAGS_INIT'):
            with self.subTest(variable=variable), tempfile.TemporaryDirectory() as tmp:
                source=Path(tmp)/'src';source.mkdir()
                (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.27)\nproject(WavePolicy NONE)\n'
                    f'include("{ROOT}/cmake/ROCmArchitectures.cmake")\n')
                result=run(['cmake','-S',str(source),'-B',str(Path(tmp)/'build'),'-G','Ninja',
                    '-D'+variable+'=-mwavefrontsize32'])
                self.assertNotEqual(result.returncode,0)
                self.assertIn('HIP kernels require wave64',result.stdout+result.stderr)

    def test_actual_registration_clears_inherited_architecture_spoof(self):
        """Replay the actual registration function through CTest and its child."""
        text=(ROOT/'tests/v2/CMakeLists.txt').read_text()
        start=text.index('function(add_v2_test TEST_NAME)')
        end=text.index('\nendfunction()',start)+len('\nendfunction()')
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'src';source.mkdir(); build=Path(tmp)/'build'
            helper=source/'owner.sh'
            helper.write_text('#!/bin/sh\nshift 3\nexec "$@"\n');helper.chmod(0o755)
            probe=source/'probe.py'
            probe.write_text('import os\nassert "HSA_OVERRIDE_GFX_VERSION" not in os.environ, dict(os.environ)\n')
            (source/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.27)\nproject(Env NONE)\nenable_testing()\n'
                'add_executable(llaminar_mpi_session IMPORTED GLOBAL)\n'
                f'set_property(TARGET llaminar_mpi_session PROPERTY IMPORTED_LOCATION "{helper}")\n'
                'set(V2_TEST_SOCKETS 1)\nset(V2_TEST_CORES_PER_SOCKET 1)\n'+text[start:end]+'\n'
                f'add_v2_test(NativeIdentity COMMAND "{shutil.which("python3")}" "{probe}" LABELS Integration ROCm DeviceFree NO_MPI NO_MODELS MPI_PROCS 1)\n')
            result=run(['cmake','-S',str(source),'-B',str(build),'-G','Ninja'])
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            result=run(['ctest','--test-dir',str(build),'--output-on-failure'],env={**os.environ,'HSA_OVERRIDE_GFX_VERSION':'9.0.6'})
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_actual_campaign_child_uses_native_identity(self):
        """The focused runner bypasses CTest but must preserve the same unset edge."""
        path=ROOT/'scripts/ci/run_production_parity_campaigns.py'
        spec=importlib.util.spec_from_file_location('rocm_identity_campaign_replay',path)
        module=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=module
        with mock.patch.object(sys,'path',[str(path.parent),*sys.path]):
            spec.loader.exec_module(module)
        with mock.patch.dict(os.environ,{'HSA_OVERRIDE_GFX_VERSION':'9.0.6'}):
            self.assertEqual(module._run_process([sys.executable,'-c',
                'import os; assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ'],10),0)
            self.assertEqual(module._run_process([sys.executable,'-c',
                'import os; assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ'],10,
                {'HSA_OVERRIDE_GFX_VERSION':'9.0.6'}),0)

    def test_actual_blas_receipt_requires_both_targets(self):
        """A valid receipt cannot hide a missing target binary or dispatch table."""
        pins={}
        for line in (ROOT/'scripts/docker/rocm-release.env').read_text().splitlines():
            if line and not line.startswith('#'):
                k,v=line.split('=',1);pins[k]=v.strip("'")
        with tempfile.TemporaryDirectory() as tmp:
            sdk=Path(tmp)/'sdk'; output=Path(tmp)/'output'
            (sdk/'.info').mkdir(parents=True);(sdk/'.info/version').write_text(pins['LLAMINAR_ROCM_VERSION']+'\n')
            payloads={'lib/librocblas.so':b'device-free DSO identity'}
            for arch in ('gfx906','gfx1100'):
                for suffix in ('co','dat'):
                    payloads[f'lib/rocblas/library/{arch}/Tensile_{arch}.{suffix}']=f'{arch}-{suffix}'.encode()
            for name,data in payloads.items():
                p=output/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
            marker=output/'share/llaminar/rocblas-source.txt';marker.parent.mkdir(parents=True)
            def receipt():
                entries=''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n' for name,data in sorted(payloads.items()))
                identity=':'.join(pins[k] for k in ('LLAMINAR_ROCM_VERSION','LLAMINAR_ROCM_DEB_VERSION','LLAMINAR_ROCM_LIBRARIES_REVISION'))
                marker.write_text(identity+':gfx906:xnack-;gfx1100:'+hashlib.sha256(entries.encode()).hexdigest()+'\n')
            receipt()
            env={**os.environ,'ROCM_PATH':str(sdk),'ROCBLAS_INSTALL_PREFIX':str(output)}
            env.pop('ROCBLAS_GPU_TARGETS',None)
            result=run(['bash',str(ROOT/'scripts/docker/install-rocblas.sh')],env=env)
            self.assertEqual(result.returncode,0,result.stderr)
            for name,data in list(payloads.items()):
                if 'gfx1100' not in name: continue
                with self.subTest(missing=name):
                    (output/name).unlink();payloads.pop(name);receipt()
                    result=run(['bash',str(ROOT/'scripts/docker/install-rocblas.sh')],env=env)
                    self.assertNotEqual(result.returncode,0)
                    self.assertNotIn('already installed',result.stdout)
                    self.assertIn('matching amdrocm-llvm-dev',result.stderr)
                    payloads[name]=data;(output/name).write_bytes(data);receipt()


if __name__=='__main__':
    unittest.main()
