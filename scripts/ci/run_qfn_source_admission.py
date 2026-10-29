#!/usr/bin/env python3
"""Gate hardware-eligible QFN source cells from installed CTest/GTest metadata.

This is setup admission, separate from numerical parity and model-free preflight.
No topology/depth list or vendor-count heuristic lives here. The exact image's
native startup inventory selects typed participants; unavailable cells remain
unqualified. Every selected cell, including candidates, must pass without skips
and publish per-rank retirement. Canonical GGUF paths are read-only inputs, never
rewritten into another source tier. Broad optional downloads are suppressed only
after every selected REQUIRED_FILES input has been validated.
"""
from __future__ import annotations
import argparse
import csv
from enum import Enum
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from production_artifacts import digest, write_json

PREFIX = 'V2_Integration_Parity_Qwen4Exp_Production_'
class SourceAdmissionContract(str, Enum):
    """Disjoint native proof obligations; capacity is not structural support."""
    POSITIVE = 'positive-source-admission'
    UNSUPPORTED_REJECTION = 'expected-native-rejection'
    CAPACITY_REJECTION = 'expected-capacity-rejection'


SOURCE_METHOD_CONTRACTS = {
    'FullSourceAdmission': SourceAdmissionContract.POSITIVE,
    'CandidateSourceAdmission': SourceAdmissionContract.POSITIVE,
    'RejectUnsupportedSourceAdmission': SourceAdmissionContract.UNSUPPORTED_REJECTION,
    'RejectCapacitySourceAdmission': SourceAdmissionContract.CAPACITY_REJECTION,
}
METHODS = tuple(SOURCE_METHOD_CONTRACTS)


def source_contract(exact: str) -> SourceAdmissionContract:
    """Read the explicit GTest method contract, never a topology or capacity guess."""
    matches = [contract for method, contract in SOURCE_METHOD_CONTRACTS.items()
               if f'.{method}/' in exact]
    if len(matches) != 1:
        raise ValueError('unknown or ambiguous source admission contract')
    return matches[0]


def identity(device: dict) -> tuple[str, int, int]:
    """Validate an exact physical address; CPU NUMA is not CPU ordinal."""
    backend, ordinal, numa = (device.get(key) for key in ('backend', 'ordinal', 'numa_node'))
    if (backend not in ('cpu', 'cuda', 'rocm') or type(ordinal) is not int or ordinal < 0
            or type(numa) is not int or numa < -1 or (backend == 'cpu' and (ordinal != 0 or numa < 0))):
        raise ValueError('invalid exact source-admission device identity')
    # GPU locality is observed but does not rename the backend address.
    return backend, ordinal, numa if backend == 'cpu' else -1


def provisioning_requirement(record: dict) -> int | None:
    """Read an authored physical test class; never derive a memory estimate."""
    minimum = record.get('minimum_cuda_total_memory_bytes')
    participants = record.get('participants')
    if 'minimum_cuda_total_memory_bytes' in record and (not isinstance(participants, list)
            or not all(isinstance(p, dict) for p in participants)):
        raise ValueError('physical provisioning requires typed participant identities')
    if 'minimum_cuda_total_memory_bytes' in record and (type(minimum) is not int or minimum <= 0
            or record.get('mtp_enabled') is not True
            or record.get('retained_mtp_draft_capacity') != 15
            or 'gpu_memory_cap_mb' in record
            or not any(p.get('backend') == 'cuda' for p in record.get('participants', []))):
        raise ValueError('invalid authored CUDA physical provisioning prerequisite')
    return minimum


def valid_unqualified(row: dict) -> bool:
    """Authenticate disjoint missing-device and explicit physical-class evidence."""
    if row.get('qualification') == 'hardware-ineligible-unqualified':
        return bool(row.get('missing'))
    if row.get('qualification') != 'hardware-capacity-ineligible-unqualified':
        return False
    evidence = row.get('insufficient')
    if not isinstance(evidence, list) or not evidence:
        return False
    for entry in evidence:
        required, observed = entry.get('required_total_memory_bytes'), entry.get('observed_total_memory_bytes')
        if (type(required) is not int or type(observed) is not int or not 0 < observed < required
                or not isinstance(entry.get('participant'), dict)
                or type(entry['participant'].get('rank')) is not int or entry['participant']['rank'] < 0):
            return False
        try:
            if identity(entry['participant'])[0] != 'cuda':
                return False
        except ValueError:
            return False
    return True


def select(records: dict, hardware: dict) -> tuple[list[str], dict]:
    """Select exact typed participants against one native local-host observation."""
    if (type(hardware.get('qfn_source_hardware_schema')) is not int
            or hardware.get('qfn_source_hardware_schema') != 2
            or hardware.get('scope') != 'local-host-all-ranks'
            or not isinstance(hardware.get('devices'), list) or not hardware['devices']):
        raise ValueError('missing or foreign native source hardware metadata')
    available = [identity(device) for device in hardware['devices']]
    if len(set(available)) != len(available):
        raise ValueError('duplicate native hardware identity')
    observed_devices = {identity(device): device for device in hardware['devices']}
    selected, unavailable = [], {}
    for exact, record in records.items():
        ranks = record.get('mpi_ranks')
        participants = record.get('participants')
        if (type(record.get('qfn_source_admission_schema')) is not int
                or record.get('qfn_source_admission_schema') != 2 or type(ranks) is not int or ranks <= 0
                or not isinstance(record.get('id'), str) or not record['id']
                or Path(record['id']).name != record['id'] or record['id'] in ('.', '..')
                or len(record['id'].encode()) >= 256
                or not isinstance(participants, list) or not participants):
            raise ValueError('missing or stale typed source-admission metadata')
        minimum = provisioning_requirement(record)
        declared = set()
        missing = []
        insufficient = []
        for participant in participants:
            rank = participant.get('rank')
            address = identity(participant)
            if type(rank) is not int or not 0 <= rank < ranks or (rank, address) in declared:
                raise ValueError('invalid or duplicate typed source participant')
            declared.add((rank, address))
            if address not in available:
                missing.append(participant)
            elif minimum is not None and address[0] == 'cuda':
                total = observed_devices[address].get('total_memory_bytes')
                if type(total) is not int or total <= 0:
                    raise ValueError('required CUDA physical total is missing or stale')
                if total < minimum:
                    insufficient.append({'participant': participant,
                        'required_total_memory_bytes': minimum, 'observed_total_memory_bytes': total})
        if {rank for rank, _ in declared} != set(range(ranks)):
            raise ValueError('typed source participants do not cover the declared world')
        if missing:
            unavailable[exact] = {'qualification': 'hardware-ineligible-unqualified', 'missing': missing}
        elif insufficient:
            unavailable[exact] = {'qualification': 'hardware-capacity-ineligible-unqualified',
                                  'insufficient': insufficient}
        else:
            selected.append(exact)
    if not selected:
        raise ValueError('source admission has no hardware-eligible cells')
    return selected, unavailable


def source_parameters(document: dict) -> dict:
    """Read complete source-only GTest JSON; ordinary numerical JSON is untouched."""
    records = {}
    for suite in document['testsuites']:
        for test in suite['testsuite']:
            if test['name'].split('/', 1)[0] not in METHODS:
                continue
            exact = suite['name'] + '.' + test['name']
            if exact in records:
                raise ValueError('duplicate source GTest identity')
            record = json.loads(test['value_param'])
            capacity = SOURCE_METHOD_CONTRACTS[test['name'].split('/', 1)[0]] == SourceAdmissionContract.CAPACITY_REJECTION
            if type(record.get('qfn_source_admission_schema')) is not int or record['qfn_source_admission_schema'] != 2:
                raise ValueError('stale source provisioning metadata schema; rebuild discovery')
            minimum = provisioning_requirement(record)
            if minimum is not None and SOURCE_METHOD_CONTRACTS[test['name'].split('/', 1)[0]] != SourceAdmissionContract.POSITIVE:
                raise ValueError('physical provisioning belongs only to authored positive source coverage')
            retained = record.get('retained_mtp_draft_capacity')
            active = record.get('mtp_enabled')
            if (type(retained) is not int or retained < 0 or type(active) is not bool
                    or (active and retained == 0)):
                raise ValueError('source metadata lacks its typed MTP activation/retained bound')
            cap = record.get('gpu_memory_cap_mb')
            if capacity and (type(cap) is not int or cap <= 0):
                raise ValueError('capacity source metadata requires its positive authored GPU cap')
            if not capacity and 'gpu_memory_cap_mb' in record:
                raise ValueError('GPU cap metadata belongs only to the explicit capacity contract')
            if record.get('id') != test['name'].split('/', 1)[1]:
                raise ValueError('stale source parameter name/identity')
            records[exact] = record
    if not records:
        raise ValueError('missing source-only GTest inventory')
    return records


def registrations(document: dict) -> dict:
    """Require exact, reference-free CTest source contracts without reconstructing axes."""
    result = {}
    for test in document['tests']:
        if not test['name'].startswith(PREFIX):
            continue
        command = test['command']
        filters = [arg.removeprefix('--gtest_filter=') for arg in command if arg.startswith('--gtest_filter=')]
        if len(filters) != 1 or not any('.' + method + '/' in filters[0] for method in METHODS):
            continue
        exact = filters[0]
        props = {prop['name']: prop['value'] for prop in test['properties']}
        labels = set(props.get('LABELS', []))
        contract = source_contract(exact)
        requirements = {
            'GTEST_FILTER': exact not in result and not any(char in exact for char in '*?:\n\r'),
            'LABELS': {'Admission', 'RealModel', 'FullModel'} <= labels and not labels & {
                'PyTorch', 'GoldenReference', 'Parity', 'ProductionTestPreflight'},
            'CONTRACT_LABELS': ('NegativeAdmission' in labels) == (contract != SourceAdmissionContract.POSITIVE)
                and ('CapacityAdmission' in labels) == (contract == SourceAdmissionContract.CAPACITY_REJECTION)
                and ('UnqualifiedCandidate' in labels) == ('.CandidateSourceAdmission/' in exact),
            'DISABLED': not props.get('DISABLED'),
            'WILL_FAIL': not props.get('WILL_FAIL'),
            'FIXTURES_REQUIRED': props.get('FIXTURES_REQUIRED') == ['V2_Models'],
            'RESOURCE_LOCK': props.get('RESOURCE_LOCK') == ['Integration_Serial'],
            'RUN_SERIAL': props.get('RUN_SERIAL') is True,
            'REQUIRED_FILES': bool(props.get('REQUIRED_FILES')),
            'TIMEOUT': type(props.get('TIMEOUT')) in (int, float)
                and math.isfinite(props['TIMEOUT']) and props['TIMEOUT'] > 0,
        }
        failed = [field for field, valid in requirements.items() if not valid]
        if failed:
            raise ValueError(f"invalid source-only CTest admission contract: {test['name']}: {', '.join(failed)}")
        binaries = [arg for arg in command if Path(arg).name == 'v2_integration_parity_qwen4exp_production_matrix']
        if len(binaries) != 1:
            raise ValueError('ambiguous source executable')
        world_options = [i for i, arg in enumerate(command) if arg in ('-np', '-n', '--np')]
        if len(world_options) != 1 or world_options[0] + 1 >= len(command):
            raise ValueError('source CTest has no exact MPI launch world')
        world_text = command[world_options[0] + 1]
        if not world_text.isdecimal() or int(world_text) <= 0:
            raise ValueError('invalid source CTest MPI world')
        result[exact] = {'test': test['name'], 'properties': props, 'binary': binaries[0],
                         'mpi_ranks': int(world_text)}
    if not result:
        raise ValueError('empty QFN source CTest inventory')
    return result


def authenticate_membership(registered: dict, records: dict) -> None:
    """Reject stale membership and a typed world different from its exact MPI launch."""
    if set(records) != set(registered):
        raise ValueError('stale or foreign CTest/GTest source membership')
    if any(record['mpi_ranks'] != registered[exact]['mpi_ranks'] for exact, record in records.items()):
        raise ValueError('typed source world disagrees with its CTest MPI launch')


def validate_outcome(xml: Path, log: str) -> None:
    """Reject CTest/GTest skips and incomplete execution even when process exit is zero."""
    cases = ET.parse(xml).getroot().findall('.//testcase')
    if (len(cases) != 1 or any(cases[0].find(key) is not None for key in ('failure', 'error', 'skipped'))
            or re.search(r'\[\s*SKIPPED\s*\]', log)):
        raise ValueError('source admission did not execute exactly one passing, non-skipped CTest')


def validate_native_retirement(root: Path, record: dict) -> None:
    """Require every rank's fresh native retirement independently of contract success."""
    for rank in range(record['mpi_ranks']):
        directory = root / 'source-admission' / record['id'] / f'rank-{rank}'
        with (directory / 'retirement.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if (len(rows) != 1 or rows[0].get('cell') != record['id']
                or any(rows[0].get(key) != '1' for key in ('runner_retired', 'model_retired', 'prefix_removed'))
                or (directory / 'prefix-cache').exists()):
            raise ValueError('missing or incomplete source final retirement')


def validate_retirement(root: Path, record: dict, *, contract: SourceAdmissionContract) -> None:
    """Require actual native retirement plus the exact source contract evidence."""
    if not isinstance(contract, SourceAdmissionContract):
        raise ValueError('source retirement requires an explicit typed contract')
    validate_native_retirement(root, record)
    for rank in range(record['mpi_ranks']):
        directory = root / 'source-admission' / record['id'] / f'rank-{rank}'
        receipt = directory / {
            SourceAdmissionContract.POSITIVE: 'admission.csv',
            SourceAdmissionContract.UNSUPPORTED_REJECTION: 'rejection.csv',
            SourceAdmissionContract.CAPACITY_REJECTION: 'capacity-rejection.csv',
        }[contract]
        with receipt.open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 1 or rows[0].get('cell') != record['id']:
            raise ValueError('missing source admission evidence')
        if contract == SourceAdmissionContract.UNSUPPORTED_REJECTION:
            if (rows[0].get('expected_rejection') != '1' or rows[0].get('initialized') != '0'
                    or rows[0].get('numerical_inference_executed') != '0'
                    or (directory / 'admission.csv').exists() or (directory / 'capacity-rejection.csv').exists()):
                raise ValueError('missing explicit native rejection evidence')
        elif contract == SourceAdmissionContract.CAPACITY_REJECTION:
            cap = record.get('gpu_memory_cap_mb')
            if (type(cap) is not int or cap <= 0 or rows[0].get('gpu_memory_cap_mb') != str(cap)
                    or rows[0].get('initialized') != '0' or rows[0].get('capacity_rejected') != '1'
                    or rows[0].get('numerical_inference_executed') != '0'
                    or (directory / 'admission.csv').exists() or (directory / 'rejection.csv').exists()
                    or not (directory / 'capacity-diagnostic.txt').is_file()
                    or not (directory / 'capacity-diagnostic.txt').read_text().strip()):
                raise ValueError('missing explicit authored-capacity rejection evidence')
        else:
            if (rows[0].get('initialized') != '1' or rows[0].get('layers') != '48'
                    or rows[0].get('snapshots_enabled') != '0'
                    or rows[0].get('numerical_inference_executed') != '0'
                    or (directory / 'rejection.csv').exists() or (directory / 'capacity-rejection.csv').exists()
                    or not (directory / 'admission_memory.txt').is_file()
                    or not (directory / 'admission_memory.txt').read_text().strip()):
                raise ValueError('source admission receipt changed its setup-only contract')
            with (directory / 'mtp-setup.csv').open() as stream:
                mtp_rows = list(csv.DictReader(stream))
            retained = record.get('retained_mtp_draft_capacity')
            active = record.get('mtp_enabled')
            if (len(mtp_rows) != 1 or mtp_rows[0].get('cell') != record['id']
                    or type(retained) is not int or retained < 0 or type(active) is not bool
                    or mtp_rows[0].get('retained_depth') != str(retained)
                    or mtp_rows[0].get('active') != str(int(active))
                    or mtp_rows[0].get('controller_execution_proved') != '0'):
                raise ValueError('missing setup-only MTP observation')


def cleanup_prefixes(root: Path) -> None:
    """Retire only disposable caches within this invocation's private evidence root."""
    for path in root.glob('source-admission/*/rank-*/prefix-cache'):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('foreign symlink in source prefix cleanup')
        shutil.rmtree(path)


class SourceCaseFailure(RuntimeError):
    """One executed selected contract failed; gather subsequent ordinary cases."""
    def __init__(self, message: str, *, retirement_proved: bool = False):
        super().__init__(message)
        self.retirement_proved = retirement_proved



def run_case(entry: dict, record: dict, output: Path, build: Path, environment: dict,
             artifact: Path, index: int, exact: str) -> bool:
    """Execute one registered contract; launch errors remain terminal infrastructure failures."""
    xml = output / f'cell-{index:04d}.xml'
    log_path = output / f'cell-{index:04d}.log'
    with log_path.open('w') as log:
        result = subprocess.run(['ctest', '--test-dir', str(build), '--output-on-failure',
            '--no-tests=error', '--fixture-exclude-any', '.*', '--output-junit', str(xml),
            '-R', '^' + re.escape(entry['test']) + '$'],
            env={**environment, 'LLAMINAR_SKIP_MODEL_DOWNLOAD': '1',
                 'LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT': str(artifact)},
            stdout=log, stderr=subprocess.STDOUT)
    try:
        cases = ET.parse(xml).getroot().findall('.//testcase')
    except (FileNotFoundError, ET.ParseError) as error:
        raise RuntimeError('incomplete CTest execution envelope') from error
    if len(cases) != 1:
        raise RuntimeError('incomplete CTest execution envelope')
    if any(re.search(r'timeout|timed out', failure.get('message', ''), re.I)
           for failure in cases[0].findall('failure')):
        raise RuntimeError('source CTest timeout is a terminal infrastructure failure')
    retired = False
    try:
        validate_native_retirement(artifact, record)
        retired = True
    except (ValueError, FileNotFoundError):
        pass  # Unknown native retirement is recorded, never promoted to proof.
    try:
        if result.returncode:
            raise ValueError(f'source CTest failed ({result.returncode}): {exact}')
        validate_outcome(xml, log_path.read_text())
        validate_retirement(artifact, record, contract=source_contract(exact))
    except (ValueError, FileNotFoundError, ET.ParseError) as error:
        raise SourceCaseFailure(str(error), retirement_proved=retired) from error
    return retired


def refresh_counts(report: dict) -> None:
    """Keep attempted/red/unexecuted counts truthful on partial and complete reports."""
    report['passed_cells'] = sum(row['passed'] for row in report['results'].values())
    report['failed_cells'] = sum(not row['passed'] for row in report['results'].values())
    report['unexecuted_selected'] = report.get('selected', 0) - report.get('attempted', 0)
    report['all_resources_retired'] = bool(report['results']) and all(
        row.get('retirement_proved') is True for row in report['results'].values())


def gather_selected(selected: list, records: dict, registered: dict, output: Path,
                    build: Path, environment: dict, report: dict) -> None:
    """Record all ordinary selected failures, preserving exact ownership and cleanup."""
    for index, exact in enumerate(selected):
        for filename, previous in report.get('source_files', {}).items():
            stat = Path(filename).stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != tuple(previous):
                raise ValueError('canonical source file changed before next cell')
        report['attempted'] = index + 1
        print(f'[qfn-source-admission] {index + 1}/{len(selected)} {exact}', flush=True)
        artifact = output / f'evidence-{index:04d}'
        artifact.mkdir()
        outcome = {'passed': False, 'retirement_proved': False, 'qualification': 'failed-source-contract', 'evidence': artifact.name,
                   'contract': source_contract(exact).value}
        try:
            retired = run_case(registered[exact], records[exact], output, build, environment, artifact, index, exact)
            outcome.update(passed=True, retirement_proved=retired, qualification='native-source-admission')
        except SourceCaseFailure as error:
            outcome.update(error=str(error), retirement_proved=error.retirement_proved)
        finally:
            report['results'][exact] = outcome
            # Cleanup failures are infrastructure failures: retain the report,
            # but never start another cell with unretired disposable storage.
            try:
                cleanup_prefixes(artifact)
            except Exception as error:
                outcome.update(passed=False, qualification='infrastructure-failure',
                               error='prefix cleanup: ' + str(error))
                raise
            finally:
                refresh_counts(report)
                write_json(output / 'source-admission.json', report)


def main(argv=None) -> int:
    """Discover, select and execute the immutable installed source-only transaction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 1, 'complete': False, 'passed': False, 'results': {}, 'all_resources_retired': False}
    report_path = args.output / 'source-admission.json'
    write_json(report_path, report)
    try:
        listing = json.loads(subprocess.check_output(['ctest', '--test-dir', str(args.build_dir),
                             '--show-only=json-v1'], text=True))
        registered = registrations(listing)
        binaries = {row['binary'] for row in registered.values()}
        if len(binaries) != 1:
            raise ValueError('source inventory belongs to more than one executable')
        binary = binaries.pop()
        environment = {**os.environ, 'HWLOC_COMPONENTS': '-gl,-opencl',
                       'OMPI_MCA_btl_vader_single_copy_mechanism': 'none'}
        with tempfile.TemporaryDirectory(prefix='qfn-source-metadata-') as temporary:
            parameters = Path(temporary) / 'parameters.json'
            subprocess.run([binary, '--gtest_list_tests', f'--gtest_output=json:{parameters}'],
                           env={**environment, 'LLAMINAR_FORCE_CPU_ONLY_STARTUP': '1'},
                           check=True, capture_output=True, timeout=30)
            records = source_parameters(json.loads(parameters.read_text()))
        authenticate_membership(registered, records)
        hardware = args.output / 'hardware.json'
        subprocess.run([binary, f'--qfn-source-hardware-inventory={hardware}'],
                       env={**environment, 'LLAMINAR_FORCE_CPU_ONLY_STARTUP': '0',
                            'LLAMINAR_SKIP_CUDA_STARTUP': '0', 'LLAMINAR_SKIP_ROCM_STARTUP': '0'},
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        selected, unavailable = select(records, json.loads(hardware.read_text()))
        print(f'[qfn-source-admission] discovered={len(records)} selected={len(selected)} '
              f'hardware_ineligible_unqualified={len(unavailable)}', flush=True)
        report.update(discovered=len(records), selected=len(selected), hardware_ineligible=len(unavailable),
                      unqualified=unavailable, inventory_digest=digest({'ctest': registered, 'typed': records}),
                      hardware_digest=digest(json.loads(hardware.read_text())), selected_cases=selected)
        write_json(args.output / 'inventory.json', {'registered': registered, 'typed': records, 'hardware_ineligible': unavailable})
        # No optional fetch can turn an absent required source into a skipped pass.
        source_stats = {}
        for exact in selected:
            for filename in registered[exact]['properties']['REQUIRED_FILES']:
                path = Path(filename)
                if not path.is_absolute() or path.suffix.lower() != '.gguf' or not path.is_file():
                    raise ValueError(f'missing canonical source GGUF: {filename}')
                stat = path.stat()
                source_stats[filename] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        report['source_files'] = source_stats
        write_json(report_path, report)
        gather_selected(selected, records, registered, args.output, args.build_dir, environment, report)
        for filename, previous in source_stats.items():
            stat = Path(filename).stat()
            if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) != previous:
                raise ValueError('canonical source file changed during admission')
        report.update(complete=True, passed=all(row['passed'] for row in report['results'].values()))
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        refresh_counts(report)
        write_json(report_path, report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
