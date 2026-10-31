#!/usr/bin/env python3
## @file test_qfn_tp4_source_closure.py
#  @brief Validate actual QFN source closure and owned registration metadata without model execution.
"""Exercise QFN's real CMake source declaration, discovery and strict inventory.

Tiny fake GGUF bytes and a metadata-only GoogleTest lister isolate registration:
no real model, compiler, accelerator, inference or native test executable runs.
The actual target's CMake call supplies scopes, the production discovery script
publishes REQUIRED_FILES/environment, and the ordinary inventory consumer joins
original roles. A missing inactive predictor remains an error even with MTP off.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from test_source_admission_discovery import discovery_session_arguments

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import model_parity_inventory as inventory

TARGET = 'v2_integration_parity_qwen4exp_production_matrix'
PREDICTOR = 'mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf'


def source_call(text, prefix):
    """Select one original multiline CMake declaration without interpreting argv.

    The closed target/set calls used here contain no nested closing command.
    CMake itself subsequently expands variables and parses their argument lists.
    """
    start = text.index(prefix)
    end = text.index('\n    )', start) + len('\n    )') if prefix.startswith('    discover') else text.index(')', start) + 1
    return text[start:end]


def cmake(command):
    """Run only a model-free CMake script or CTest metadata query, never a build."""
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise AssertionError(result.stderr or result.stdout)
    return result


class QFNTP4SourceClosureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def declared_sources(self):
        """Evaluate the installed target call with a metadata-only declaration sink.

        This sink isolates the actual callsite from target/compiler setup. It
        performs no source-role selection: the original discovery script below
        owns backend union, required files and runtime manifest publication.
        """
        source = (ROOT / 'tests/v2/CMakeLists.txt').read_text()
        model_set = source_call(source, '    set(V2_PARITY_MODEL_QWEN_FLASH_NEXT\n')
        registration = source_call(source, '    discover_v2_parity_tests(' + TARGET + '\n')
        sink = self.root / 'declaration.cmake'
        sink.write_text('''function(discover_v2_parity_tests target)
  cmake_parse_arguments(ARG "" "MPI_PROCS;TEST_PREFIX;TIMEOUT;PRODUCTION_PERF_STATS_FILTER"
    "LABELS;MODEL_FILES;CPU_MODEL_FILES;CUDA_MODEL_FILES;ROCM_MODEL_FILES;PRODUCTION_ISOLATED_TESTS" ${ARGN})
  foreach(scope IN ITEMS MODEL_FILES CPU_MODEL_FILES CUDA_MODEL_FILES ROCM_MODEL_FILES)
    file(WRITE "${CMAKE_CURRENT_LIST_DIR}/${scope}.txt" "${ARG_${scope}}")
  endforeach()
endfunction()
''' + model_set + '\n' + registration + '\n')
        cmake(['cmake', '-P', str(sink)])
        return {scope: [v for v in (self.root / (scope + '.txt')).read_text().split(';') if v]
                for scope in ('MODEL_FILES', 'CPU_MODEL_FILES', 'CUDA_MODEL_FILES', 'ROCM_MODEL_FILES')}

    def discover(self, omit_predictor=False):
        """Feed actual scoped declarations to production CMake and live inventory."""
        declared = self.declared_sources()
        # Preserve original names/roles while replacing only fixture storage roots.
        sources = {scope: [str(self.root / Path(v).name) for v in values]
                   for scope, values in declared.items()}
        for path in {p for values in sources.values() for p in values}:
            Path(path).write_bytes(b'GGUF metadata fixture only')
        main = sources['MODEL_FILES'][0]
        predictor = str(self.root / PREDICTOR)
        cases = ['CPU0', 'CUDA0', 'ROCm0123_TP4_AllGPUExperts', 'CUDA0_ROCm0_CPU0']
        records = []
        for case in cases:
            record = dict(model_parity_schema=1, id=case, model=main, e2e=None,
                          cross_host_e2e=[], runtime={'generation': {'mtp_policy': 'off'}})
            if case == 'ROCm0123_TP4_AllGPUExperts':
                record['shared_predictor'] = predictor
            records.append(dict(name='ProductionParity/' + case, value_param=json.dumps(record)))
        binary = self.root / TARGET
        # Only --gtest_list_tests and --gtest_output=json are handled. No test body exists.
        binary.write_text('#!/usr/bin/env python3\nimport json,sys\n'
            + 'rows=' + repr(records) + '\n'
            + "if not any(a.startswith('--gtest_output=json:') for a in sys.argv):\n print('QFNFixtureSuite.')\n"
            + " for row in rows: print('  '+row['name'])\n"
            + "else:\n path=next(a.split(':',1)[1] for a in sys.argv if a.startswith('--gtest_output=json:'))\n"
            + " with open(path,'w') as out: json.dump({'testsuites':[{'name':'QFNFixtureSuite','testsuite':rows}]},out)\n")
        binary.chmod(0o755)
        output = self.root / 'generated.cmake'
        command = ['cmake', '-DTEST_EXECUTABLE=' + str(binary), '-DCTEST_FILE=' + str(output),
                   '-DTEST_PREFIX=V2_Integration_Parity_Qwen4Exp_Fixture',
                   '-DLABELS=V2;Integration;Parity;CPU;CUDA;ROCm', '-DMPI_PROCS=1',
                   '-DCORES_PER_SOCKET=1', '-DNUM_SOCKETS=1', '-DWORKING_DIR=' + str(self.root)]
        command += discovery_session_arguments(self.root)
        for scope, values in sources.items():
            if omit_predictor and scope == 'ROCM_MODEL_FILES':
                values = [v for v in values if Path(v).name != PREDICTOR]
            command.append('-D' + scope + '_SERIALIZED=' + '|'.join(values))
        command += ['-P', str(ROOT / 'tests/v2/cmake/V2ParityTestDiscovery.cmake')]
        cmake(command)
        (self.root / 'CTestTestfile.cmake').write_text('include("' + str(output) + '")\n')
        campaigns = inventory.parity.discover_campaigns(self.root)
        return sources, campaigns

    def test_actual_rocm_registration_and_inventory_retain_inactive_predictor(self):
        sources, campaigns = self.discover()
        self.assertEqual(len(sources['MODEL_FILES']), 4)
        self.assertEqual(len(campaigns), 4)
        for campaign in campaigns:
            with self.subTest(backend=campaign.group.backends):
                expected = sources['MODEL_FILES'] + ([str(self.root / PREDICTOR)]
                    if 'ROCm' in campaign.group.backends else [])
                self.assertEqual(list(campaign.model_files), expected)
                env = dict(entry.split('=', 1) for entry in campaign.environment)
                self.assertEqual(env['LLAMINAR_PRODUCTION_PARITY_DECLARED_MODELS'], '|'.join(expected))
        self.assertEqual([Path(v).name for v in sources['ROCM_MODEL_FILES']], [PREDICTOR])
        self.assertEqual(sources['CPU_MODEL_FILES'], [])
        self.assertEqual(sources['CUDA_MODEL_FILES'], [])
        args = argparse.Namespace(manifest=None, build_dir=self.root, source_revision='fixture',
                                  backend='.*', campaign='.*', cell='.*')
        selected = inventory.discover(args, inventory.InventoryScope.ALL)
        self.assertEqual(len(selected), 4)
        tp4 = next(item for item in selected if 'TP4_AllGPUExperts' in item[1])
        self.assertEqual(tp4[2]['shared_predictor'], str(self.root / PREDICTOR))
        self.assertEqual(tp4[2]['runtime']['generation']['mtp_policy'], 'off')
        self.assertEqual(len(tp4[0].model_files), 5)

    def test_missing_predictor_in_actual_discovery_is_rejected_by_live_inventory(self):
        self.discover(omit_predictor=True)
        args = argparse.Namespace(manifest=None, build_dir=self.root, source_revision='fixture',
                                  backend='.*', campaign='.*', cell='.*')
        with self.assertRaisesRegex(ValueError, 'shared_predictor is not in the canonical GGUF manifest'):
            inventory.discover(args, inventory.InventoryScope.ALL)


if __name__ == '__main__':
    unittest.main()
