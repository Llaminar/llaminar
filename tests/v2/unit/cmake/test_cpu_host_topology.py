#!/usr/bin/env python3
"""@file test_cpu_host_topology.py
@brief Device-free topology evidence, hybrid identity and actual CMake admission regressions.

Temporary sysfs-shaped files are explicit metadata fixtures, not an SDK, host
capability advertisement or substituted discovery command. The recorded Spark
input retains package36 and twenty singleton sibling groups; repeated cluster/
lscpu presentation identities never enter physical-core counting. Real CMake
fixtures exercise the production decoder/finalizer without a compiler or model.
"""
from __future__ import annotations
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
MODULE_DIR = ROOT / 'tests/v2/cmake'
SPEC = importlib.util.spec_from_file_location('llaminar_test_host_topology', MODULE_DIR / 'CPUHostTopology.py')
assert SPEC and SPEC.loader
TOPOLOGY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOPOLOGY
SPEC.loader.exec_module(TOPOLOGY)


def cpu(number, package, core, siblings):
    """Construct one explicit immutable metadata fixture, not a native observation."""
    return TOPOLOGY.LogicalCPU(number, package, core, TOPOLOGY.CPUIdSet.parse(siblings))


def recorded_spark_document():
    """Replay only original recorded sysfs/affinity data from the small source-pinned fixture."""
    record = json.loads((ROOT / 'tests/v2/fixtures/cpu/recorded_gb10_hybrid_topology.json').read_text(encoding='utf-8'))
    rows = tuple(cpu(row['cpu_id'], row['package_id'], row['core_id'], row['thread_siblings'])
                 for row in record['sysfs_cpus'])
    if [row.cpu_id for row in rows] != record['affinity_cpus']:
        raise AssertionError('recorded original sysfs and affinity identities disagree')
    return record, TOPOLOGY.TestHostTopology(rows).document()


class CPUHostTopologyTests(unittest.TestCase):
    """Reject guessed geometry and exercise genuine production admission interfaces."""

    def test_hybrid_single_package_has_twenty_distinct_sibling_groups(self):
        """Recorded Spark input: two clusters/twenty sibling groups are one package, not ten cores."""
        record, document = recorded_spark_document()
        self.assertEqual(record['provenance']['evidence_archive_sha256'],
                         '15497193eeb13213c06c0d202ac031f59e2a24776b38a9bc25c72a0a57362c67')
        self.assertEqual(record['legacy_summary']['threads_per_core'], '1\n1')
        self.assertEqual(len({(row[1], row[2]) for row in record['legacy_lscpu_cpu_socket_core_node']}), 10)
        self.assertEqual({row['cluster_id'] for row in record['sysfs_cpus']}, {56, 1144})
        self.assertEqual(document['package_count'], 1)
        self.assertEqual(document['physical_core_count'], 20)
        self.assertEqual(document['logical_cpu_count'], 20)
        self.assertEqual(document['core_thread_widths'], [1] * 20)
        self.assertEqual(document['packages'], [{'package_id': 36, 'physical_cores': 20, 'logical_cpus': 20}])
        self.assertEqual(TOPOLOGY.TestHostTopology.from_document(document).document(), document)

    def test_tooling_queries_use_sibling_groups_on_recorded_gb10(self):
        """Shell tooling receives twenty GB10 physical cores in CPU order, never lscpu's ten."""
        _, document = recorded_spark_document()
        topology = TOPOLOGY.TestHostTopology.from_document(document)
        self.assertEqual(TOPOLOGY.query(topology, 'packages', []), '1')
        self.assertEqual(TOPOLOGY.query(topology, 'max-cores-per-package', []), '20')
        self.assertEqual(TOPOLOGY.query(topology, 'profiler-cpus', ['20', '1']),
                         ','.join(str(cpu) for cpu in range(20)))
        self.assertEqual(TOPOLOGY.query(topology, 'profiler-cpus', ['4', '1']), '0,1,2,3')
        with self.assertRaises(TOPOLOGY.TopologyError):
            TOPOLOGY.query(topology, 'profiler-cpus', ['21', '1'])
        with self.assertRaises(TOPOLOGY.TopologyError):
            TOPOLOGY.query(topology, 'profiler-cpus', ['1', '2'])
        with self.assertRaises(TOPOLOGY.TopologyError):
            TOPOLOGY.query(topology, 'cores', [])

    def test_tooling_queries_keep_one_cpu_per_smt_core_and_package_lanes(self):
        """SMT siblings collapse to their first CPU; lanes follow package order."""
        fixture = ROOT / 'tests/v2/fixtures/cpu/dual_socket_28core_smt2_topology.json'
        document = json.loads(fixture.read_text(encoding='utf-8'))
        topology = TOPOLOGY.TestHostTopology.from_document(document)
        self.assertEqual(TOPOLOGY.query(topology, 'packages', []), '2')
        self.assertEqual(TOPOLOGY.query(topology, 'max-cores-per-package', []), '28')
        self.assertEqual(TOPOLOGY.query(topology, 'profiler-cpus', ['2', '2']), '0,1;28,29')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'topology.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(TOPOLOGY.main(['--document', str(path), 'query', 'packages']), 0)
            self.assertEqual(output.getvalue(), '2\n')
            document['packages'][0]['physical_cores'] = 27
            path.write_text(json.dumps(document), encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(TOPOLOGY.main(['--document', str(path), 'query', 'packages']), 1)

    def test_core_labels_are_not_authority_when_sibling_groups_are_distinct(self):
        """A repeated core label cannot merge disjoint genuine physical sibling sets."""
        rows = tuple(cpu(index, 36, index % 10, str(index)) for index in range(20))
        self.assertEqual(TOPOLOGY.TestHostTopology(rows).document()['physical_core_count'], 20)

    def test_asymmetric_packages_and_partial_smt_retain_original_vectors(self):
        """The observation keeps unequal geometry, without inventing a shared scalar policy."""
        rows = (cpu(0, 4, 0, '0,8'), cpu(8, 4, 0, '0,8'),
                cpu(1, 4, 1, '1,9'), cpu(2, 4, 2, '2,10'), cpu(21, 36, 0, '21'))
        document = TOPOLOGY.TestHostTopology(rows).document()
        self.assertEqual(document['physical_core_count'], 4)
        self.assertEqual(document['logical_cpu_count'], 5)
        self.assertEqual(document['core_thread_widths'], [2, 1, 1, 1])
        self.assertNotIn('cores_per_socket', document)
        self.assertNotIn('threads_per_core', document)
        self.assertEqual([row['physical_cores'] for row in document['packages']], [3, 1])

    def test_missing_conflicting_and_unknown_identity_fail_closed(self):
        """No missing field, overlapping sibling group or invalid identity becomes one invented core."""
        examples = [(), (cpu(0, -1, 0, '0'),), (cpu(0, 36, -1, '0'),),
                    (cpu(0, 36, 0, '1'),), (cpu(0, 36, 0, '0'), cpu(0, 36, 0, '0')),
                    (cpu(0, 36, 0, '0-1'), cpu(1, 36, 1, '1')),
                    (cpu(0, 36, 0, '0,8'), cpu(1, 36, 1, '1,8'))]
        for rows in examples:
            with self.subTest(rows=rows), self.assertRaises(TOPOLOGY.TopologyError):
                TOPOLOGY.TestHostTopology(rows)
        for text in ['', '-1', '0,0', '2-1', '0-2,1-3', '0 1', '1\n1']:
            with self.subTest(text=text), self.assertRaises(TOPOLOGY.TopologyError):
                TOPOLOGY.CPUIdSet.parse(text)
        sparse = TOPOLOGY.CPUIdSet.parse('0-1000000000000')
        self.assertEqual(len(sparse.intervals), 1)
        self.assertTrue(sparse.contains(19))

    def test_derived_document_corruption_cannot_supply_another_budget(self):
        """Immutable original records, not copied counters or bool/float coercion, define policy."""
        document = TOPOLOGY.TestHostTopology((cpu(0, 36, 0, '0'),)).document()
        for key, value in [('package_count', 2), ('physical_core_count', 20),
                           ('omp_workers_per_rank', 4), ('schema_version', True),
                           ('logical_slots_per_physical_budget', '1\n1'),
                           ('logical_cpu_count', 1.0)]:
            altered = {**document, key: value}
            with self.subTest(key=key), self.assertRaises(TOPOLOGY.TopologyError):
                TOPOLOGY.TestHostTopology.from_document(altered)

    def test_capture_requires_complete_stable_original_affinity(self):
        """Real file reads and a changed affinity witness cannot publish incomplete topology."""
        with tempfile.TemporaryDirectory(prefix='llaminar-topology-metadata-') as directory:
            root = Path(directory)
            for index in range(20):
                topology = root / f'cpu{index}/topology'
                topology.mkdir(parents=True)
                for name, text in [('physical_package_id', '36'), ('core_id', str(index)),
                                   ('thread_siblings_list', str(index))]:
                    (topology / name).write_text(text + '\n', encoding='ascii')
            observed = TOPOLOGY.observe_linux(root, lambda: range(20))
            self.assertEqual(observed.document()['physical_core_count'], 20)
            changed = iter((range(20), range(19)))
            with self.assertRaisesRegex(TOPOLOGY.TopologyError, 'affinity changed'):
                TOPOLOGY.observe_linux(root, lambda: next(changed))
            (root / 'cpu19/topology/core_id').unlink()
            with self.assertRaisesRegex(TOPOLOGY.TopologyError, 'incomplete kernel topology'):
                TOPOLOGY.observe_linux(root, lambda: range(20))

    def test_actual_cmake_gate_ignores_multiline_presentation_geometry(self):
        """The production finalizer uses authenticated slot policy, not the old multiline1/1 value."""
        with tempfile.TemporaryDirectory(prefix='llaminar-hybrid-cmake-') as directory:
            root = Path(directory)
            source, build = root / 'source', root / 'build'
            source.mkdir()
            _, document = recorded_spark_document()
            (source / 'topology.json').write_text(json.dumps(document), encoding='utf-8')
            (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.24)
project(HybridTopologyMetadata NONE)
enable_testing()
include([=[''' + str(MODULE_DIR / 'V2TestHostTopology.cmake') + ''']=])
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/topology.json" evidence)
# Prior presentation values cannot remain a live authority after checked binding.
set(V2_TEST_THREADS_PER_CORE "1\\n1")
set(V2_TEST_SOCKETS "1\\n1")
set(V2_TEST_CORES_PER_SOCKET "10\\n10")
v2_bind_test_host_topology("${evidence}")
if(NOT V2_TEST_SOCKETS EQUAL 1 OR NOT V2_TEST_CORES_PER_SOCKET EQUAL 20 OR NOT V2_TEST_THREADS_PER_CORE EQUAL 1)
    message(FATAL_ERROR "Checked original topology failed to replace stale presentation values")
endif()
set(V2_PERF_TESTS_ONLY OFF)
function(register_owned_marker name labels budget)
    add_custom_target(${name}_target COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/${name}.built")
    add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E true)
    set_tests_properties(${name} PROPERTIES LABELS "${labels}" PROCESSORS ${budget})
    if("Unit" IN_LIST labels)
        set_property(GLOBAL APPEND PROPERTY V2_UNIT_GATE_TARGETS ${name}_target)
    endif()
    set_property(GLOBAL PROPERTY V2_TEST_BUILD_TARGETS_${name} ${name}_target)
endfunction()
register_owned_marker(ActualUnit "V2;Unit" 4)
register_owned_marker(ActualPreflight "V2;Integration;ProductionTestPreflight" 2)
include([=[''' + str(MODULE_DIR / 'V2TestGateFinalization.cmake') + ''']=])
register_owned_marker(LateUnit "V2;Unit" 3)
''', encoding='utf-8')
            ninja = shutil.which('ninja')
            self.assertIsNotNone(ninja)
            result = subprocess.run(['cmake', '-S', str(source), '-B', str(build), '-G', 'Ninja',
                                     '-DCMAKE_MAKE_PROGRAM:FILEPATH=' + str(ninja)],
                                    capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            built = subprocess.run(['cmake', '--build', str(build), '--parallel', '--target',
                                    'v2_unit_gate', 'v2_production_test_preflight_gate'],
                                   capture_output=True, text=True, timeout=20)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            self.assertEqual({item.stem for item in build.glob('*.built')},
                             {'ActualUnit', 'ActualPreflight', 'LateUnit'})
            listing = subprocess.check_output(['ctest', '--test-dir', str(build), '--show-only=json-v1'],
                                              text=True, timeout=20)
            properties = {test['name']: {row['name']: row['value'] for row in test['properties']}
                          for test in json.loads(listing)['tests']}
            self.assertEqual(properties['ActualUnit']['PROCESSORS'], 4)
            self.assertEqual(properties['LateUnit']['PROCESSORS'], 3)
            self.assertEqual(properties['ActualPreflight']['PROCESSORS'], 2)


    def test_actual_owner_retires_receipt_before_failed_readmission(self):
        """Success cannot survive rejected projection or failed native observer admission."""
        _, good = recorded_spark_document()
        rejected = TOPOLOGY.TestHostTopology((cpu(0, 4, 0, '0'), cpu(1, 4, 1, '1'), cpu(2, 36, 0, '2'))).document()
        for failure in ('projection', 'observer'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(prefix='llaminar-topology-retirement-') as directory:
                root = Path(directory)
                (root / 'good.json').write_text(json.dumps(good), encoding='utf-8')
                (root / 'rejected.json').write_text(json.dumps(rejected), encoding='utf-8')
                preamble = ('cmake_minimum_required(VERSION 3.24)\ninclude([=[' +
                            str(MODULE_DIR / 'V2TestHostTopology.cmake') + ']=])\n')
                publish = root / 'publish.cmake'
                publish.write_text(preamble + 'file(READ good.json evidence)\n'
                                   'v2_admit_test_host_topology("${evidence}")\n', encoding='utf-8')
                fail = root / 'fail.cmake'
                fail.write_text(preamble + ('file(READ rejected.json evidence)\n'
                    'v2_admit_test_host_topology("${evidence}")\n' if failure == 'projection' else
                    'set(Python3_EXECUTABLE /missing-native-python-interpreter)\n'
                    'v2_observe_test_host_topology()\n'), encoding='utf-8')
                success = subprocess.run(['cmake', '-P', str(publish)], cwd=root,
                                         capture_output=True, text=True, timeout=20)
                self.assertEqual(success.returncode, 0, success.stdout + success.stderr)
                receipt = root / 'v2-test-host-topology.json'
                self.assertEqual(json.loads(receipt.read_text()), good)
                for _ in range(2):
                    receipt.with_suffix('.json.tmp').write_text('{"interrupted":', encoding='utf-8')
                    result = subprocess.run(['cmake', '-P', str(fail)], cwd=root,
                                            capture_output=True, text=True, timeout=20)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(receipt.exists(), 'rejected attempt retained a success-looking receipt')
                    self.assertFalse(receipt.with_suffix('.json.tmp').exists())

    def test_cmake_rejects_asymmetric_projection_and_noninteger_counts(self):
        """Current consumers reject unrepresentable geometry instead of min/max/one substitutions."""
        documents = [
            TOPOLOGY.TestHostTopology((cpu(0, 4, 0, '0'), cpu(1, 4, 1, '1'), cpu(2, 36, 0, '2'))).document(),
            TOPOLOGY.TestHostTopology((cpu(0, 4, 0, '0,8'), cpu(8, 4, 0, '0,8'), cpu(1, 4, 1, '1'))).document(),
        ]
        normal = TOPOLOGY.TestHostTopology((cpu(0, 36, 0, '0'),)).document()
        documents.extend([{**normal, 'package_count': '1\n1'}, {**normal, 'logical_cpu_count': True}])
        for document in documents:
            with self.subTest(document=document), tempfile.TemporaryDirectory(prefix='llaminar-topology-reject-') as directory:
                root = Path(directory)
                (root / 'evidence.json').write_text(json.dumps(document), encoding='utf-8')
                script = root / 'bind.cmake'
                script.write_text('cmake_minimum_required(VERSION 3.24)\ninclude([=[' +
                    str(MODULE_DIR / 'V2TestHostTopology.cmake') + ']=])\nfile(READ [=[' +
                    str(root / 'evidence.json') + ']=] evidence)\nv2_bind_test_host_topology("${evidence}")\n', encoding='utf-8')
                result = subprocess.run(['cmake', '-P', str(script)], capture_output=True, text=True, timeout=20)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stdout + result.stderr, 'cannot be represented|missing/non-numeric')


if __name__ == '__main__':
    unittest.main()
