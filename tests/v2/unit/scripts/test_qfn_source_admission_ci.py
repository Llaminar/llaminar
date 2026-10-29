#!/usr/bin/env python3
"""Fast controls for typed hardware selection and fail-closed source CI receipts.

These synthetic inputs test policy only. Runtime qualification still requires
real executable metadata, native startup inventory and per-rank retirement.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import nullcontext
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
from run_qfn_source_admission import (SourceAdmissionContract, source_contract, SourceCaseFailure, authenticate_membership, gather_selected, run_case, registrations, select, source_parameters,
                                      validate_outcome, validate_retirement, cleanup_prefixes)
from qfn_source_admission_image import validate_report, validate_binding
from production_artifacts import digest
import run_develop_image_gate as gate
import qfn_source_admission_image as image_gate


def parameter(backend='cpu', ordinal=0, numa_node=0):
    """Return a record whose opaque name cannot encode its physical topology."""
    return {'qfn_source_admission_schema': 2, 'id': 'Opaque', 'mpi_ranks': 2,
            'mtp_enabled': False, 'retained_mtp_draft_capacity': 0,
            'participants': [{'rank': rank, 'backend': backend, 'ordinal': ordinal, 'numa_node': numa_node}
                             for rank in range(2)]}


def hardware():
    """Return sparse actual identities, including CUDA3 but no CUDA0 or CPU NUMA1."""
    return {'qfn_source_hardware_schema': 2, 'scope': 'local-host-all-ranks', 'devices': [
        {'backend': 'cpu', 'ordinal': 0, 'numa_node': 0},
        {'backend': 'cuda', 'ordinal': 3, 'numa_node': 0}]}


class QFNSourceAdmissionCI(unittest.TestCase):
    """Missing, foreign and stale evidence must fail rather than reduce coverage."""

    def test_explicit_large_card_class_only_filters_its_authored_positive(self):
        minimum = 32 * 1024**3
        large = {**parameter('cuda', 0), 'mtp_enabled': True,
                 'retained_mtp_draft_capacity': 15, 'minimum_cuda_total_memory_bytes': minimum}
        mandatory = {'off': parameter('cuda', 0),
                     'bounded': {**parameter('cuda', 0), 'mtp_enabled': True, 'retained_mtp_draft_capacity': 3},
                     'negative': {**parameter('cuda', 0), 'mtp_enabled': True,
                                  'retained_mtp_draft_capacity': 15, 'gpu_memory_cap_mb': 13312}}
        observed = {'qfn_source_hardware_schema': 2, 'scope': 'local-host-all-ranks', 'devices': [
            {'backend': 'cuda', 'ordinal': 0, 'numa_node': 0, 'total_memory_bytes': 24 * 1024**3},
            {'backend': 'cuda', 'ordinal': 3, 'numa_node': 0, 'total_memory_bytes': 64 * 1024**3}]}
        selected, unavailable = select({**mandatory, 'large': large}, observed)
        self.assertEqual(set(selected), set(mandatory))
        self.assertEqual(unavailable['large']['qualification'], 'hardware-capacity-ineligible-unqualified')
        self.assertEqual(unavailable['large']['insufficient'][0]['required_total_memory_bytes'], minimum)
        self.assertEqual(unavailable['large']['insufficient'][0]['observed_total_memory_bytes'], 24 * 1024**3)
        observed['devices'][0]['total_memory_bytes'] = minimum
        selected, unavailable = select({**mandatory, 'large': large}, observed)
        self.assertEqual(set(selected), {*mandatory, 'large'})
        self.assertFalse(unavailable)

    def test_explicit_physical_class_authenticates_each_rank_and_required_device(self):
        minimum = 32 * 1024**3
        large = {**parameter('cuda', 0), 'mtp_enabled': True,
                 'retained_mtp_draft_capacity': 15, 'minimum_cuda_total_memory_bytes': minimum}
        large['participants'][1]['ordinal'] = 1
        observed = {'qfn_source_hardware_schema': 2, 'scope': 'local-host-all-ranks', 'devices': [
            {'backend': 'cpu', 'ordinal': 0, 'numa_node': 0},
            {'backend': 'cuda', 'ordinal': 0, 'numa_node': 0, 'total_memory_bytes': minimum},
            {'backend': 'cuda', 'ordinal': 1, 'numa_node': 0, 'total_memory_bytes': minimum - 1},
            {'backend': 'cuda', 'ordinal': 3, 'numa_node': 0, 'total_memory_bytes': minimum * 2}]}
        selected, unavailable = select({'mandatory': parameter(), 'large': large}, observed)
        self.assertEqual(selected, ['mandatory'])
        evidence = unavailable['large']['insufficient']
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['participant']['rank'], 1)
        self.assertEqual(evidence[0]['participant']['ordinal'], 1)
        for stale in (None, 0, -1, True, '34359738368', 34359738368.0):
            bad = copy.deepcopy(observed)
            bad['devices'][2]['total_memory_bytes'] = stale
            with self.subTest(stale=stale), self.assertRaises(ValueError):
                select({'mandatory': parameter(), 'large': large}, bad)
        del observed['devices'][2]['total_memory_bytes']
        with self.assertRaises(ValueError):
            select({'mandatory': parameter(), 'large': large}, observed)
        for invalid in (None, 0, -1, True, 'large'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                select({'mandatory': parameter(), 'large': {**large, 'minimum_cuda_total_memory_bytes': invalid}}, hardware())

    def test_qualifying_positive_failure_stays_red_and_capacity_unqualified_stays_visible(self):
        from run_qfn_source_admission import valid_unqualified
        small = {'qualification': 'hardware-capacity-ineligible-unqualified', 'insufficient': [{
            'participant': {'rank': 0, 'backend': 'cuda', 'ordinal': 0, 'numa_node': 0},
            'required_total_memory_bytes': 32 * 1024**3, 'observed_total_memory_bytes': 24 * 1024**3}]}
        self.assertTrue(valid_unqualified(small))
        self.assertFalse(valid_unqualified({**small, 'insufficient': []}))
        report = {'schema': 1, 'complete': True, 'passed': False, 'all_resources_retired': True,
                  'selected_cases': ['Source.FullSourceAdmission/Large'], 'selected': 1, 'discovered': 2,
                  'hardware_ineligible': 1, 'attempted': 1, 'passed_cells': 0, 'failed_cells': 1,
                  'unexecuted_selected': 0, 'inventory_digest': 'inventory', 'hardware_digest': 'hardware',
                  'results': {'Source.FullSourceAdmission/Large': {'passed': False, 'retirement_proved': True,
                      'qualification': 'failed-source-contract', 'contract': 'positive-source-admission', 'error': 'PMA regression'}},
                  'unqualified': {'Small': small}}
        validate_report(report, require_pass=False)
        with self.assertRaises(ValueError):
            validate_report(report)

    def test_exact_sparse_identities_and_cpu_numa(self):
        records = {'one': parameter(), 'CUDA8': parameter('cuda', 3),
                   'CPU0': parameter('cpu', 0, 1), 'CUDA0': parameter('cuda', 0)}
        selected, unavailable = select(records, hardware())
        self.assertEqual(selected, ['one', 'CUDA8'])
        self.assertEqual(set(unavailable), {'CPU0', 'CUDA0'})
        self.assertTrue(all(row['qualification'] == 'hardware-ineligible-unqualified'
                            for row in unavailable.values()))

    def test_source_serialization_does_not_consume_numerical_schema(self):
        name = 'CandidateSourceAdmission/Opaque'
        document = {'testsuites': [{'name': 'Source', 'testsuite': [
            {'name': name, 'value_param': json.dumps(parameter())},
            {'name': 'ProductionParity/Numeric', 'value_param': '{"model_parity_schema":1}'}]}]}
        self.assertEqual(source_parameters(document), {'Source.' + name: parameter()})
        document['testsuites'][0]['testsuite'].append(document['testsuites'][0]['testsuite'][0])
        with self.assertRaises(ValueError):
            source_parameters(document)

    def test_missing_foreign_stale_and_empty_metadata(self):
        for change in ({'qfn_source_admission_schema': 1}, {'participants': []}, {'mpi_ranks': 0},
                       {'participants': [dict(parameter()['participants'][0], numa_node=-1)]},
                       {'participants': [parameter()['participants'][0]]},
                       {'id': '../foreign'}, {'participants': [dict(parameter()['participants'][0], rank=2)]},
                       {'participants': [dict(parameter()['participants'][0], ordinal=1)]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                select({'exact': {**parameter(), **change}}, hardware())
        for change in ({'qfn_source_hardware_schema': 1}, {'scope': 'remote'}, {'devices': []}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                select({'exact': parameter()}, {**hardware(), **change})
        with self.assertRaises(ValueError):
            select({'exact': parameter('rocm', 8)}, hardware())

    def test_candidate_failure_and_image_mismatch_are_not_passes(self):
        report = {'schema': 1, 'complete': True, 'passed': True, 'all_resources_retired': True,
                  'selected_cases': ['Source.CandidateSourceAdmission/Opaque'], 'selected': 1, 'discovered': 2, 'hardware_ineligible': 1,
                  'attempted': 1, 'passed_cells': 1, 'failed_cells': 0, 'unexecuted_selected': 0,
                  'inventory_digest': 'inventory', 'hardware_digest': 'hardware',
                  'results': {'Source.CandidateSourceAdmission/Opaque': {'passed': True, 'retirement_proved': True, 'qualification': 'native-source-admission', 'contract': 'positive-source-admission'}},
                  'unqualified': {'Absent': {'qualification': 'hardware-ineligible-unqualified', 'missing': [{}]}}}
        validate_report(report)
        for change in ({'results': {}}, {'selected': 2}, {'discovered': 1}, {'passed': False},
                       {'all_resources_retired': False},
                       {'results': {'Source.CandidateSourceAdmission/Opaque': {'passed': False, 'qualification': 'expected-failure'}}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_report({**report, **change})
        source = {'revision': 'source', 'tree': 'tree'}
        binding = {'schema': 1, 'source': source, 'cpu_isa': 'AVX512', 'runtime_image': 'runtime',
                   'test_runner_image': 'companion', 'report_digest': digest(report), 'container_retired': True,
                   'prefix_cleanup_complete': True, 'driver_report_digest': 'driver-evidence'}
        binding['test_runner_identity'] = {'id': 'companion', 'labels': {
            'org.opencontainers.image.revision': source['revision'], 'org.llaminar.source_tree': source['tree'],
            'org.llaminar.build_type': 'Release', 'org.llaminar.cpu_isa': 'AVX512',
            'org.llaminar.cuda': 'ON', 'org.llaminar.rocm': 'ON', 'org.llaminar.image_role': 'test-runner',
            'org.llaminar.integration_skipped': '0', 'org.llaminar.test_runner_inventory': 'full-matrix'}}
        validate_binding(binding, report, source=source, isa='AVX512', runtime='runtime')
        for key, value in (('source', {}), ('cpu_isa', 'AVX2'), ('runtime_image', 'foreign'),
                           ('report_digest', 'stale'), ('container_retired', False), ('prefix_cleanup_complete', False), ('test_runner_identity', None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_binding({**binding, key: value}, report, source=source, isa='AVX512', runtime='runtime')

    def test_zero_exit_skip_missing_rank_and_private_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            xml = root / 'result.xml'
            xml.write_text('<testsuites><testsuite><testcase name="exact"/></testsuite></testsuites>')
            validate_outcome(xml, '[  PASSED  ] 1 test.')
            with self.assertRaises(ValueError):
                validate_outcome(xml, '[  SKIPPED ] exact')
            xml.write_text('<testsuites><testsuite><testcase><skipped/></testcase></testsuite></testsuites>')
            with self.assertRaises(ValueError):
                validate_outcome(xml, '')
            with self.assertRaises(FileNotFoundError):
                validate_retirement(root, parameter(), contract=SourceAdmissionContract.POSITIVE)
            cache = root / 'source-admission/owned/rank-0/prefix-cache'
            cache.mkdir(parents=True)
            (cache / 'disposable').write_text('payload')
            retained = cache.parent / 'admission.csv'
            retained.write_text('evidence')
            cleanup_prefixes(root)
            self.assertFalse(cache.exists())
            self.assertEqual(retained.read_text(), 'evidence')

    def test_exact_ctest_world_and_foreign_membership_are_rejected(self):
        authenticate_membership({'exact': {'mpi_ranks': 2}}, {'exact': parameter()})
        with self.assertRaisesRegex(ValueError, 'MPI launch'):
            authenticate_membership({'exact': {'mpi_ranks': 1}}, {'exact': parameter()})
        with self.assertRaisesRegex(ValueError, 'foreign'):
            authenticate_membership({'foreign': {'mpi_ranks': 2}}, {'exact': parameter()})

    def test_gathers_ordinary_failures_and_cleans_each_case_before_continuing(self):
        selected = [f'Source.FullSourceAdmission/{name}' for name in ('first', 'second', 'third')]
        records = {name: parameter() for name in selected}
        registered = {name: {} for name in selected}
        report = {'results': {}, 'selected': 3}
        visited = []
        def run(entry, record, output, build, environment, artifact, index, exact):
            visited.append(exact)
            cache = artifact / 'source-admission/Opaque/rank-0/prefix-cache'
            cache.mkdir(parents=True)
            if exact != selected[-1]:
                raise SourceCaseFailure('ordinary native defect', retirement_proved=True)
            return True
        with tempfile.TemporaryDirectory() as temporary, patch('run_qfn_source_admission.run_case', side_effect=run):
            output = Path(temporary)
            gather_selected(selected, records, registered, output, output, {}, report)
            self.assertEqual(visited, selected)
            self.assertEqual(report['attempted'], 3)
            self.assertTrue(report['all_resources_retired'])
            self.assertEqual((report['passed_cells'], report['failed_cells'], report['unexecuted_selected']), (1, 2, 0))
            self.assertEqual([row['passed'] for row in report['results'].values()], [False, False, True])
            self.assertFalse(list(output.glob('evidence-*/source-admission/*/rank-*/prefix-cache')))
            self.assertEqual(len(json.loads((output / 'source-admission.json').read_text())['results']), 3)

    def test_strict_positive_and_negative_per_rank_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(2):
                directory = root / f'source-admission/Opaque/rank-{rank}'
                directory.mkdir(parents=True)
                (directory / 'retirement.csv').write_text('cell,runner_retired,model_retired,prefix_removed\nOpaque,1,1,1\n')
                (directory / 'admission.csv').write_text('cell,layers,initialized,snapshots_enabled,numerical_inference_executed\nOpaque,48,1,0,0\n')
                (directory / 'admission_memory.txt').write_text('actual PMA attestation')
                (directory / 'mtp-setup.csv').write_text('cell,active,retained_depth,controller_execution_proved\nOpaque,0,0,0\n')
            validate_retirement(root, parameter(), contract=SourceAdmissionContract.POSITIVE)
            directory = root / 'source-admission/Opaque/rank-1'
            for filename, changed in (
                ('admission.csv', 'cell,layers,initialized,snapshots_enabled,numerical_inference_executed\nOpaque,47,1,0,0\n'),
                ('admission.csv', 'cell,layers,initialized,snapshots_enabled,numerical_inference_executed\nOpaque,48,1,1,0\n'),
                ('admission_memory.txt', ''),
                ('mtp-setup.csv', 'cell,active,retained_depth,controller_execution_proved\nOpaque,0,0,1\n')):
                path = directory / filename
                original = path.read_text()
                path.write_text(changed)
                with self.subTest(filename=filename, changed=changed), self.assertRaises(ValueError):
                    validate_retirement(root, parameter(), contract=SourceAdmissionContract.POSITIVE)
                path.write_text(original)
            (directory / 'mtp-setup.csv').unlink()
            with self.assertRaises(FileNotFoundError):
                validate_retirement(root, parameter(), contract=SourceAdmissionContract.POSITIVE)
            for rank in range(2):
                directory = root / f'source-admission/Opaque/rank-{rank}'
                (directory / 'admission.csv').unlink()
                (directory / 'rejection.csv').write_text('cell,initialized,expected_rejection,numerical_inference_executed\nOpaque,0,1,0\n')
            validate_retirement(root, parameter(), contract=SourceAdmissionContract.UNSUPPORTED_REJECTION)
            (directory / 'admission.csv').write_text('foreign positive receipt')
            with self.assertRaises(ValueError):
                validate_retirement(root, parameter(), contract=SourceAdmissionContract.UNSUPPORTED_REJECTION)
            (directory / 'admission.csv').unlink()
            for changed in ('Opaque,1,1,0', 'Opaque,0,1,1'):
                (directory / 'rejection.csv').write_text('cell,initialized,expected_rejection,numerical_inference_executed\n' + changed + '\n')
                with self.assertRaises(ValueError):
                    validate_retirement(root, parameter(), contract=SourceAdmissionContract.UNSUPPORTED_REJECTION)

    def test_master_benchmark_cannot_consume_old_e2e_without_source_receipts(self):
        from types import SimpleNamespace
        from test_published_image_suite import write_e2e_bundle, image_pair
        import run_published_image_suite as suite
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = image_pair()
            write_e2e_bundle(root, pair)
            receipt = json.loads((root / 'e2e-receipt.json').read_text())
            receipt.pop('qfn_source_admission_required')
            (root / 'e2e-receipt.json').write_text(json.dumps(receipt))
            args = SimpleNamespace(e2e_bundle=root, qfn_source_admission=True)
            with patch.object(suite, 'build_driver') as native:
                with self.assertRaisesRegex(ValueError, 'requires QFN source admission'):
                    suite.run_benchmarks(args, pair, root)
            native.assert_not_called()

    def test_driver_window_finishes_after_child_retirement_on_failure(self):
        from contextlib import nullcontext
        import run_published_image_suite as suite
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = root / 'models'
            models.mkdir()
            with patch.object(image_gate.pipeline, 'require_image'), \
                 patch.object(image_gate.docker_paths, 'device_args', return_value=[]), \
                 patch.object(image_gate.docker_paths, 'mounts', return_value=[]), \
                 patch.object(suite, 'build_driver_for_runtime', return_value='tools') as build_tools, \
                 patch.object(image_gate, 'passive_kernel_reader', return_value=nullcontext()), \
                 patch.object(image_gate.driver, 'begin', side_effect=lambda *_: events.append('begin')), \
                 patch.object(image_gate.driver, 'finish', side_effect=lambda *_: events.append('finish') or True), \
                 patch.object(image_gate.driver, 'validate_evidence'), \
                 patch.object(image_gate.subprocess, 'check_output', side_effect=lambda *_args, **_kwargs: events.append('retirement') or ''), \
                 patch.object(image_gate.pipeline, 'run', side_effect=ValueError('native red')):
                with self.assertRaisesRegex(ValueError, 'native red'):
                    image_gate.run_admission({'test-runner': {'id': 'image'}}, {}, 'AVX512', root / 'output', models, {'id': 'runtime'})
            self.assertEqual(build_tools.call_args.args[0], {'id': 'runtime'})
            self.assertEqual(events[0], 'begin')
            self.assertEqual(events[-1], 'finish')
            self.assertIn('retirement', events)

    def test_receipt_reuse_requires_the_original_driver_report_and_boot_journal(self):
        from test_published_image_suite import write_e2e_bundle, image_pair
        import run_published_image_suite as suite
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = image_pair()
            write_e2e_bundle(root, pair)
            suite.admit_e2e(root, pair, require_qfn_source_admission=True)
            lane = root / 'avx2/source-admission'
            report = lane / 'source.driver-diagnostics.json'
            data = json.loads(report.read_text())
            journal = lane / data['journal']
            original = journal.read_text()
            journal.write_text(original + '{}\n')
            with self.assertRaises(ValueError):
                suite.admit_e2e(root, pair, require_qfn_source_admission=True)
            journal.write_text(original)
            data['boot_id'] = 'different-boot'
            report.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                suite.admit_e2e(root, pair, require_qfn_source_admission=True)

    def test_master_gathers_complete_red_on_both_isas_and_never_starts_http(self):
        from types import SimpleNamespace
        import run_published_image_suite as suite
        pair = {'source': {}, 'images': {isa: {'id': isa} for isa in suite.ISAS}}
        visited = []
        def source(companion, model_source, isa, directory, models, runtime):
            visited.append(isa)
            if isa == suite.ISAS[0]:
                raise image_gate.SourceAdmissionRed({'isa': isa}, {'results': {'ordinary-red': {'passed': False}}})
            return {'isa': isa}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(suite, 'inventory_companion', return_value={}), \
                 patch.object(suite, 'run_admission', side_effect=source), \
                 patch.object(suite, 'build_driver') as http:
                with self.assertRaisesRegex(ValueError, 'source-admission cell failures'):
                    suite.run_e2e(SimpleNamespace(qfn_source_admission=True, models=root), pair, {'cells': []}, root)
            self.assertEqual(visited, list(suite.ISAS))
            http.assert_not_called()
            receipt = json.loads((root / 'e2e-receipt.json').read_text())
            self.assertEqual(set(receipt['qfn_source_admission_reports']), set(suite.ISAS))
            self.assertEqual(receipt['failed_qfn_source_cells'], {suite.ISAS[0]: ['ordinary-red']})
            self.assertFalse(receipt['complete'])

    def test_master_driver_or_infrastructure_failure_stops_before_second_isa(self):
        from types import SimpleNamespace
        import run_published_image_suite as suite
        pair = {'source': {}, 'images': {isa: {'id': isa} for isa in suite.ISAS}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(suite, 'inventory_companion', return_value={}), \
                 patch.object(suite, 'run_admission', side_effect=ValueError('incomplete driver window')) as source, \
                 patch.object(suite, 'build_driver') as http:
                with self.assertRaisesRegex(ValueError, 'incomplete driver window'):
                    suite.run_e2e(SimpleNamespace(qfn_source_admission=True, models=root), pair, {'cells': []}, root)
            self.assertEqual(source.call_count, 1)
            http.assert_not_called()

    def test_retirement_proof_is_independent_of_complete_red_contracts(self):
        report = {'schema': 1, 'complete': True, 'passed': False, 'all_resources_retired': True,
                  'selected_cases': ['Source.CandidateSourceAdmission/Opaque'], 'selected': 1, 'discovered': 1, 'hardware_ineligible': 0,
                  'attempted': 1, 'passed_cells': 0, 'failed_cells': 1, 'unexecuted_selected': 0,
                  'inventory_digest': 'inventory', 'hardware_digest': 'hardware', 'unqualified': {},
                  'results': {'Source.CandidateSourceAdmission/Opaque': {'passed': False, 'retirement_proved': True,
                      'qualification': 'failed-source-contract', 'contract': 'positive-source-admission', 'error': 'defect'}}}
        validate_report(report, require_pass=False)
        with self.assertRaises(ValueError):
            validate_report(report)
        report['results']['Source.CandidateSourceAdmission/Opaque']['retirement_proved'] = False
        with self.assertRaises(ValueError):
            validate_report(report, require_pass=False)
        report['all_resources_retired'] = False
        validate_report(report, require_pass=False)

    def test_passive_tools_reader_owns_only_the_existing_kernel_scope(self):
        calls = []
        name = None
        def run(command, **kwargs):
            nonlocal name
            calls.append(command)
            if command[:2] == ['docker', 'run']:
                name = command[command.index('--name') + 1]
        def inspect(command, **kwargs):
            if command[:2] == ['docker', 'inspect']:
                return json.dumps([{'Image': 'tools-image', 'Config': {'Labels': {'org.llaminar.kernel_reader': name}}}])
            return 'owned-container'
        with patch.dict(os.environ), patch.object(image_gate.subprocess, 'run', side_effect=run), \
             patch.object(image_gate.subprocess, 'check_output', side_effect=inspect):
            previous = os.environ.get(image_gate.driver.READER_CONTAINER_ENV)
            with self.assertRaisesRegex(ValueError, 'ordinary failure'):
                with image_gate.passive_kernel_reader('tools-image'):
                    self.assertEqual(os.environ[image_gate.driver.READER_CONTAINER_ENV], name)
                    raise ValueError('ordinary failure')
            self.assertEqual(os.environ.get(image_gate.driver.READER_CONTAINER_ENV), previous)
        launch = calls[0]
        self.assertEqual(launch[launch.index('--cap-add') + 1], 'SYSLOG')
        self.assertEqual(launch[launch.index('--device') + 1], '/dev/kmsg:/dev/kmsg:r')
        self.assertFalse(set(launch) & {'--privileged', '--mount', '-v'})
        self.assertEqual(calls[-1], ['docker', 'rm', '--force', name])

    def test_ctest_timeout_is_terminal_infrastructure(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'cell-0000.xml').write_text('<testsuites><testsuite><testcase><failure message="Timeout"/></testcase></testsuite></testsuites>')
            with patch('run_qfn_source_admission.subprocess.run', return_value=SimpleNamespace(returncode=8)):
                with self.assertRaisesRegex(RuntimeError, 'terminal infrastructure') as failure:
                    run_case({'test': 'exact'}, parameter(), output, output, {}, output, 0, 'exact')
            self.assertNotIsInstance(failure.exception, SourceCaseFailure)

    def test_capacity_metadata_is_explicit_and_cannot_leak_into_positive_cases(self):
        def document(method, record):
            return {'testsuites': [{'name': 'Source', 'testsuite': [{
                'name': method + '/Opaque', 'value_param': json.dumps(record)}]}]}
        record = {**parameter(), 'gpu_memory_cap_mb': 13312}
        exact = 'Source.RejectCapacitySourceAdmission/Opaque'
        self.assertEqual(source_contract(exact), SourceAdmissionContract.CAPACITY_REJECTION)
        self.assertEqual(source_parameters(document('RejectCapacitySourceAdmission', record)), {exact: record})
        for changed in (parameter(), {**record, 'gpu_memory_cap_mb': 0},
                        {**record, 'gpu_memory_cap_mb': -1}, {**record, 'gpu_memory_cap_mb': True},
                        {**record, 'gpu_memory_cap_mb': '13312'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                source_parameters(document('RejectCapacitySourceAdmission', changed))
        with self.assertRaises(ValueError):
            source_parameters(document('FullSourceAdmission', record))
        for changed in ({'mtp_enabled': None}, {'retained_mtp_draft_capacity': None},
                        {'mtp_enabled': True, 'retained_mtp_draft_capacity': 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                source_parameters(document('FullSourceAdmission', {**parameter(), **changed}))

    def test_capacity_receipt_cannot_masquerade_as_structural_rejection_or_positive(self):
        record = {**parameter(), 'gpu_memory_cap_mb': 13312}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(record['mpi_ranks']):
                directory = root / f'source-admission/Opaque/rank-{rank}'
                directory.mkdir(parents=True)
                (directory / 'retirement.csv').write_text('cell,runner_retired,model_retired,prefix_removed\nOpaque,1,1,1\n')
                (directory / 'capacity-diagnostic.txt').write_text('canonical native capacity rejection')
                (directory / 'capacity-rejection.csv').write_text('cell,initialized,capacity_rejected,gpu_memory_cap_mb,numerical_inference_executed\nOpaque,0,1,13312,0\n')
            validate_retirement(root, record, contract=SourceAdmissionContract.CAPACITY_REJECTION)
            for wrong in (SourceAdmissionContract.POSITIVE, SourceAdmissionContract.UNSUPPORTED_REJECTION):
                with self.subTest(wrong=wrong), self.assertRaises(FileNotFoundError):
                    validate_retirement(root, record, contract=wrong)
            for name, contents in (('admission.csv', 'foreign positive'), ('rejection.csv', 'foreign unsupported')):
                (directory / name).write_text(contents)
                with self.assertRaises(ValueError):
                    validate_retirement(root, record, contract=SourceAdmissionContract.CAPACITY_REJECTION)
                (directory / name).unlink()
            path = directory / 'capacity-rejection.csv'
            header = 'cell,initialized,capacity_rejected,gpu_memory_cap_mb,numerical_inference_executed\n'
            for row in ('Opaque,1,1,13312,0', 'Opaque,0,0,13312,0', 'Opaque,0,1,13311,0', 'Opaque,0,1,13312,1'):
                path.write_text(header + row + '\n')
                with self.subTest(row=row), self.assertRaises(ValueError):
                    validate_retirement(root, record, contract=SourceAdmissionContract.CAPACITY_REJECTION)
            path.write_text(header + 'Opaque,0,1,13312,0\n')
            (directory / 'capacity-diagnostic.txt').write_text('')
            with self.assertRaises(ValueError):
                validate_retirement(root, record, contract=SourceAdmissionContract.CAPACITY_REJECTION)

    def test_explicit_retained_three_mtp_receipt_is_not_inferred_from_its_name(self):
        record = {**parameter(), 'mtp_enabled': True, 'retained_mtp_draft_capacity': 3}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(record['mpi_ranks']):
                directory = root / f'source-admission/Opaque/rank-{rank}'
                directory.mkdir(parents=True)
                (directory / 'retirement.csv').write_text('cell,runner_retired,model_retired,prefix_removed\nOpaque,1,1,1\n')
                (directory / 'admission.csv').write_text('cell,layers,initialized,snapshots_enabled,numerical_inference_executed\nOpaque,48,1,0,0\n')
                (directory / 'admission_memory.txt').write_text('native PMA attestation')
                (directory / 'mtp-setup.csv').write_text('cell,active,retained_depth,controller_execution_proved\nOpaque,1,3,0\n')
            validate_retirement(root, record, contract=SourceAdmissionContract.POSITIVE)
            path = directory / 'mtp-setup.csv'
            for row in ('Opaque,1,15,0', 'Opaque,0,3,0', 'Opaque,1,3,1'):
                path.write_text('cell,active,retained_depth,controller_execution_proved\n' + row + '\n')
                with self.subTest(row=row), self.assertRaises(ValueError):
                    validate_retirement(root, record, contract=SourceAdmissionContract.POSITIVE)

    def test_capacity_contract_cannot_be_relabelled_unsupported_or_candidate(self):
        exact = 'Source.RejectCapacitySourceAdmission/Opaque'
        props = {'LABELS': ['Admission', 'RealModel', 'FullModel', 'NegativeAdmission', 'CapacityAdmission'],
                 'FIXTURES_REQUIRED': ['V2_Models'], 'RESOURCE_LOCK': ['Integration_Serial'],
                 'RUN_SERIAL': True, 'TIMEOUT': 900, 'REQUIRED_FILES': ['/canonical/model.gguf']}
        entry = {'name': 'V2_Integration_Parity_Qwen4Exp_Production_capacity',
                 'command': ['mpirun', '-np', '2', '/binary/v2_integration_parity_qwen4exp_production_matrix',
                             '--gtest_filter=' + exact],
                 'properties': [{'name': key, 'value': value} for key, value in props.items()]}
        self.assertIn(exact, registrations({'tests': [entry]}))
        for labels in (['Admission', 'RealModel', 'FullModel', 'NegativeAdmission'],
                       props['LABELS'] + ['UnqualifiedCandidate']):
            changed = copy.deepcopy(entry)
            changed['properties'] = [{'name': key, 'value': value} for key, value in {**props, 'LABELS': labels}.items()]
            with self.assertRaisesRegex(ValueError, 'CONTRACT_LABELS'):
                registrations({'tests': [changed]})
        report = {'schema': 1, 'complete': True, 'passed': True, 'all_resources_retired': True,
                  'selected_cases': [exact], 'selected': 1, 'discovered': 1, 'hardware_ineligible': 0,
                  'attempted': 1, 'passed_cells': 1, 'failed_cells': 0, 'unexecuted_selected': 0,
                  'inventory_digest': 'inventory', 'hardware_digest': 'hardware', 'unqualified': {},
                  'results': {exact: {'passed': True, 'retirement_proved': True,
                      'qualification': 'native-source-admission', 'contract': 'expected-capacity-rejection'}}}
        validate_report(report)
        for wrong in ('expected-native-rejection', 'positive-source-admission'):
            report['results'][exact]['contract'] = wrong
            with self.assertRaises(ValueError):
                validate_report(report)

    def test_feature_source_failure_prevents_successful_parent_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            source = {'revision': 'a' * 40, 'tree': 'tree'}
            images = {'test-runner': {'id': 'companion'}, 'runtime': {'id': 'runtime'}}
            with patch.object(gate, 'source_identity', return_value=source), \
                 patch.object(gate, 'device_lease', return_value=nullcontext()), \
                 patch.object(gate, 'build', return_value=images) as build, \
                 patch.object(gate, 'require_image_pair'), \
                 patch.object(gate, 'run_prerequisites', return_value={'preflight_test_count': 1}), \
                 patch.object(gate, 'run_admission', side_effect=ValueError('native candidate unsupported')):
                with self.assertRaisesRegex(ValueError, 'native candidate unsupported'):
                    gate.main(['--cpu-isa', 'AVX512', '--source-admission', '--output', str(output)])
            self.assertEqual(build.call_args.kwargs['test_inventory'], gate.TestRunnerInventory.FULL_MATRIX)
            self.assertFalse(json.loads((output / 'develop-image-gate.json').read_text())['complete'])

    def test_source_registration_rejects_reference_xfail_and_missing_sources(self):
        props = {'LABELS': ['Admission', 'RealModel', 'FullModel', 'UnqualifiedCandidate'],
                 'FIXTURES_REQUIRED': ['V2_Models'], 'RESOURCE_LOCK': ['Integration_Serial'],
                 'RUN_SERIAL': True, 'TIMEOUT': 900, 'REQUIRED_FILES': ['/canonical/model.gguf']}
        test = {'name': 'V2_Integration_Parity_Qwen4Exp_Production_opaque',
                'command': ['mpirun', '-np', '2', '/binary/v2_integration_parity_qwen4exp_production_matrix',
                            '--gtest_filter=Suite.CandidateSourceAdmission/Opaque'],
                'properties': [{'name': key, 'value': value} for key, value in props.items()]}
        self.assertEqual(len(registrations({'tests': [test]})), 1)
        missing_timeout = copy.deepcopy(test)
        missing_timeout['properties'] = [prop for prop in test['properties'] if prop['name'] != 'TIMEOUT']
        with self.assertRaises(ValueError) as missing:
            registrations({'tests': [missing_timeout]})
        self.assertIn(test['name'], str(missing.exception))
        self.assertIn('TIMEOUT', str(missing.exception))

        for change in ({'LABELS': props['LABELS'] + ['GoldenReference']}, {'WILL_FAIL': True},
                       {'DISABLED': True}, {'REQUIRED_FILES': []}, {'TIMEOUT': 0}, {'TIMEOUT': -1},
                       {'TIMEOUT': '900'}, {'TIMEOUT': float('inf')}, {'TIMEOUT': float('nan')}):
            changed = copy.deepcopy(test)
            changed['properties'] = [{'name': key, 'value': value} for key, value in {**props, **change}.items()]
            with self.subTest(change=change), self.assertRaises(ValueError):
                registrations({'tests': [changed]})


if __name__ == '__main__':
    unittest.main()
