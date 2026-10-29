#!/usr/bin/env python3
"""Bind discovered QFN source admission to immutable source/ISA image siblings.

Only canonical model inputs and private evidence/scratch are mounted. Reference
corpora and model download credentials never enter this setup-only transaction.
The owning parent already holds the shared device lease. Native context/MPI
retirement is validated inside the child; Docker retirement is checked here.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import docker_paths
import run_production_pipeline as pipeline
from production_artifacts import digest, write_json
from run_qfn_source_admission import cleanup_prefixes, source_contract, valid_unqualified
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
import gpu_driver_diagnostics as driver


def validate_driver_window(directory: Path, binding: dict) -> None:
    """Replay the established boot/cursor/collector journal before admitting a binding."""
    driver.validate_evidence(directory)
    reports = list(directory.glob('*.driver-diagnostics.json'))
    if binding.get('driver_report_digest') != digest(json.loads(reports[0].read_text())):
        raise ValueError('source-admission driver evidence changed')


@contextmanager
def passive_kernel_reader(image: str):
    """Own the existing tools-image SYSLOG/read-only-kmsg scope, never inference privileges."""
    name = 'llaminar-suite-driver-' + uuid.uuid4().hex
    previous = os.environ.get(driver.READER_CONTAINER_ENV)
    try:
        subprocess.run(['docker', 'run', '--detach', '--rm', '--init', '--name', name,
                        '--label', 'org.llaminar.kernel_reader=' + name,
                        '--cap-add', 'SYSLOG', '--device', '/dev/kmsg:/dev/kmsg:r',
                        image, 'python3', '-c', 'import signal; signal.pause()'],
                       check=True, stdout=subprocess.DEVNULL)
        os.environ[driver.READER_CONTAINER_ENV] = name
        yield
    finally:
        if previous is None:
            os.environ.pop(driver.READER_CONTAINER_ENV, None)
        else:
            os.environ[driver.READER_CONTAINER_ENV] = previous
        remaining = subprocess.check_output(['docker', 'ps', '--all', '--quiet', '--filter',
                                              'name=^/' + name + '$'], text=True).strip()
        if remaining:
            owned = json.loads(subprocess.check_output(['docker', 'inspect', name], text=True))[0]
            if (owned['Image'] != image or owned['Config'].get('Labels', {}).get('org.llaminar.kernel_reader') != name):
                raise ValueError('foreign passive kernel-reader ownership')
            subprocess.run(['docker', 'rm', '--force', name], check=True, timeout=30,
                           stdout=subprocess.DEVNULL)



class SourceAdmissionRed(RuntimeError):
    """Complete selected-cell results are red, with clean driver and retired child scope."""
    def __init__(self, binding: dict, report: dict):
        super().__init__('Complete QFN source-admission report contains failed contracts')
        self.binding, self.report = binding, report


def validate_report(report: dict, *, require_pass: bool = True) -> None:
    """Authenticate complete coverage; red is permitted only for failure collection."""
    selected = report.get('selected_cases')
    results = report.get('results', {})
    unavailable = report.get('unqualified', {})
    passed = sum(row.get('passed') is True for row in results.values())
    retired = bool(results) and all(row.get('retirement_proved') is True for row in results.values())
    if (report.get('schema') != 1 or report.get('complete') is not True
            or not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or set(results) != set(selected) or set(results) & set(unavailable)
            or report.get('selected') != len(selected) or report.get('attempted') != len(selected)
            or report.get('passed_cells') != passed or report.get('failed_cells') != len(selected) - passed
            or report.get('unexecuted_selected') != 0 or report.get('passed') is not (passed == len(selected))
            or report.get('all_resources_retired') is not retired
            or report.get('hardware_ineligible') != len(unavailable)
            or report.get('discovered') != len(selected) + len(unavailable)
            or any(type(row.get('retirement_proved')) is not bool or type(row.get('passed')) is not bool
                   or (row['passed'] and (row.get('qualification') != 'native-source-admission' or not row['retirement_proved']))
                   or (not row['passed'] and (row.get('qualification') != 'failed-source-contract' or not row.get('error')))
                   for row in results.values())
            or any(row.get('contract') != source_contract(exact).value
                   for exact, row in results.items())
            or any(not valid_unqualified(row)
                   for row in unavailable.values())
            or not report.get('inventory_digest') or not report.get('hardware_digest')
            or (require_pass and (not report.get('passed') or not retired))):
        raise ValueError('incomplete or unqualified QFN source admission receipt')


def validate_binding(binding: dict, report: dict, *, source: dict, isa: str, runtime: str,
                     require_pass: bool = True) -> None:
    """Authenticate the complete report against its source/ISA/runtime sibling."""
    validate_report(report, require_pass=require_pass)
    companion = binding.get('test_runner_identity')
    pipeline.require_image(companion, source, isa, pipeline.ImageRole.TEST_RUNNER,
                           test_inventory=pipeline.TestRunnerInventory.FULL_MATRIX)
    if not companion.get('id') or companion['id'] != binding.get('test_runner_image'):
        raise ValueError('source-admission companion identity changed')
    if (binding.get('schema') != 1 or binding.get('source') != source or binding.get('cpu_isa') != isa
            or binding.get('runtime_image') != runtime or not binding.get('test_runner_image')
            or binding.get('report_digest') != digest(report) or binding.get('container_retired') is not True
            or binding.get('prefix_cleanup_complete') is not True or not binding.get('driver_report_digest')):
        raise ValueError('stale or foreign source-admission image binding')


def run_admission(images: dict, source: dict, isa: str, directory: Path,
                  models: Path, runtime: dict) -> dict:
    """Run the canonical selected inventory in one explicitly retired companion."""
    image = images[pipeline.ImageRole.TEST_RUNNER.value]
    pipeline.require_image(image, source, isa, pipeline.ImageRole.TEST_RUNNER,
                           test_inventory=pipeline.TestRunnerInventory.FULL_MATRIX)
    pipeline.require_image(runtime, source, isa, pipeline.ImageRole.RUNTIME)
    directory.mkdir(parents=True, exist_ok=False)
    scratch = directory / 'ctest-temporary'
    scratch.mkdir()
    models = models.resolve(strict=True)
    name = 'llaminar-qfn-source-admission-' + uuid.uuid4().hex
    command = ['python3', 'scripts/ci/run_qfn_source_admission.py', '--build-dir',
               'build_v2_integration', '--output', '/ci-results/admission']
    launch = ['docker', 'run', '--rm', '--name', name,
              *docker_paths.device_args(image['id'], 'CPU+CUDA+ROCm', user=f'{os.getuid()}:{os.getgid()}'),
              *(part for group in os.getgroups() for part in ('--group-add', str(group))),
              *docker_paths.mounts([(models, '/opt/llaminar-models', True), (models, '/src/models', True),
                                    (directory, '/ci-results', False),
                                    (scratch, '/src/build_v2_integration/Testing/Temporary', False)]),
              '-w', '/src', image['id'],
              *pipeline.logged_container_command(command, '/ci-results/admission.process.log')]
    # ARC's runner pod has neither SYSLOG nor kmsg. Reuse the already
    # established suite-driver tools image and exact passive reader scope.
    from run_published_image_suite import build_driver_for_runtime
    tools = build_driver_for_runtime(runtime, directory)
    driver_state = directory / 'driver-state.json'
    driver_report = directory / 'source.driver-diagnostics.json'
    child_error = None
    with passive_kernel_reader(tools):
        driver.begin(driver_state)
        try:
            try:
                try:
                    pipeline.run(launch, directory / 'admission.log')
                except subprocess.CalledProcessError as error:
                    child_error = error
            finally:
                # --rm owns normal retirement; an exact-name lookup covers cancellation
                # and startup failures without removing another job's containers.
                remaining = subprocess.check_output(['docker', 'ps', '--all', '--quiet', '--filter',
                                                      'name=^/' + name + '$'], text=True).strip()
                if remaining:
                    subprocess.run(['docker', 'rm', '--force', name], check=True, timeout=30,
                                   stdout=subprocess.DEVNULL)
                if subprocess.check_output(['docker', 'ps', '--all', '--quiet', '--filter',
                                             'name=^/' + name + '$'], text=True).strip():
                    raise ValueError('source-admission container did not retire')
                for artifact in (directory / 'admission').glob('evidence-*'):
                    if artifact.is_symlink():
                        raise ValueError('foreign source artifact symlink during prefix cleanup')
                    cleanup_prefixes(artifact)
        finally:
            if not driver.finish(driver_state, driver_report):
                raise ValueError('QFN source-admission driver/boot window failed')
            driver.validate_evidence(directory)
    report = json.loads((directory / 'admission/source-admission.json').read_text())
    validate_report(report, require_pass=False)
    if (child_error is None) != report['passed']:
        raise ValueError('source child exit disagrees with its complete result')
    binding = {'schema': 1, 'source': source, 'cpu_isa': isa,
               'test_runner_image': image['id'], 'test_runner_identity': image, 'runtime_image': runtime['id'],
               'report_digest': digest(report), 'container_retired': True, 'prefix_cleanup_complete': True,
               'driver_report_digest': digest(json.loads(driver_report.read_text()))}
    validate_binding(binding, report, source=source, isa=isa, runtime=runtime['id'], require_pass=False)
    validate_driver_window(directory, binding)
    write_json(directory / 'image-binding.json', binding)
    if not report['passed']:
        raise SourceAdmissionRed(binding, report)
    return binding
