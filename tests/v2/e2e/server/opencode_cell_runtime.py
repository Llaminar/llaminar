"""Own an exact-image OpenCode app lifetime for a canonical model/topology cell.

The typed inventory supplies production placement, precision and MTP intent.
PhysicalMemoryAuthority admits context; this driver adds only app/cache budgets.
Continuous observers bracket inference and native retirement. Request duration
is unbounded, while administrative operations retain finite control deadlines.
No model or cache payload is inspected to manufacture qualification evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.request import Request, urlopen
import uuid

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import docker_paths
from production_artifacts import write_json
from run_model_parity_benchmarks import benchmark_profile
from run_model_parity_e2e import readiness_timeout_seconds
from flash_attention_perf_policy import validate_flash_attention_plan_policy
from gpu_host_transfer_perf_policy import validate_gpu_host_transfer_policy
from graph_capture_perf_policy import DecodeGraphRequirement, validate_graph_capture_policy
from llep_verifier_perf_policy import validate_llep_verifier_policy
from mtp_device_generation_perf_policy import (validate_cuda_dynamic_mtp_device_generation_policy,
    validate_host_scheduled_mtp_device_generation_policy)
from opencode_certification_evidence import validate_app_report, validate_native_coverage
from opencode_context_admission import AdmissionOutcome, ContextPolicy, select_context
from opencode_stress_policy import app_archive_compaction_assessment
from opencode_tool_stress import wait_for_client
from prefix_reuse_measurement import integer
from ranked_perf_artifacts import collect_and_publish_ranked_perf_stats, validate_memory_authority
from resource_growth_evidence import WorkloadWindow, assess_observations
from runtime_feature_perf_policy import MovementEvidence, validate_runtime_feature_policy
from runtime_stats_observer import RuntimeStatsObserver
from server_execution_contract import (automatic_selection_policy, validate_automatic_selection,
                                       validate_server_execution_contract)

OUTPUT_TOKENS = 32768
RAM_MIB = 16384
DISK_MIB = 32768


@dataclass(frozen=True)
class CodingCell:
    """One inventory-selected cell and its lifecycle-owned external resources."""

    image: str
    configuration: dict
    backends: str
    model: Path
    output: Path
    port: int = 19080
    opencode: str = '/usr/local/bin/opencode'

    def __post_init__(self) -> None:
        """Reject stale discovery and invalid geometry before touching Docker."""
        if (not re.fullmatch(r'sha256:[0-9a-f]{64}', self.image)
                or type(self.port) is not int or not 1 <= self.port <= 65535
                or not self.model.is_absolute() or not self.output.is_absolute()):
            raise ValueError('Coding cell requires an immutable image and absolute owned paths')
        benchmark_profile(self.configuration)
        selection = automatic_selection_policy(self.configuration['e2e'])
        kinds = self.backends.split('+')
        if (not kinds or len(set(kinds)) != len(kinds)
                or not set(kinds) <= {'CPU', 'CUDA', 'ROCm'}
                or {kind.lower() for kind in kinds} != set(selection['device_counts'])):
            raise ValueError('Coding backend inventory disagrees with the typed placement intent')
        readiness_timeout_seconds(self.configuration['e2e'])
        generation_policy(self.configuration)

    def server_arguments(self, context: int, *, dry_run: bool = False) -> list[str]:
        """Retain production intent verbatim and vary only PMA context demand."""
        if type(context) is not int or context <= OUTPUT_TOKENS:
            raise ValueError('Coding context must retain its full reasoning budget and prompt space')
        return ['serve', *benchmark_profile(self.configuration)['args'],
                '-m', '/model/' + self.model.name, '--context-length', str(context),
                '--host', '127.0.0.1', '--port', str(self.port),
                '--prefix-cache-ram-budget-mb', str(RAM_MIB),
                '--prefix-cache-disk-budget-mb', str(DISK_MIB),
                '--prefix-cache-disk-dir', '/prefix-cache', *(['--dry-run'] if dry_run else [])]


def command(arguments: list[str], *, timeout: float | None = 30) -> str:
    """Bound administration, retaining stdout/stderr on a precise native failure."""
    return subprocess.check_output(arguments, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def metadata(url: str, route: str = '/stats') -> dict:
    """Read one bounded metadata response without placing a deadline on inference."""
    with urlopen(url + route, timeout=5) as response:
        payload = response.read(1024 * 1024 + 1)
        if response.status != 200 or len(payload) > 1024 * 1024:
            raise ValueError('Invalid or unbounded server metadata response')
        value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError('Server metadata is not an object')
    if route == '/stats':
        if value.get('object') != 'llaminar.stats' or value.get('schema_version') != 1:
            raise ValueError('Unsupported server statistics schema')
        integer(value['epoch'], 'epoch')
        requests = value['requests']
        for key in ('active', 'active_in_current_epoch', 'started', 'completed', 'failed', 'disconnected'):
            integer(requests[key], key)
        if requests['started'] != sum(requests[key] for key in
                ('active_in_current_epoch', 'completed', 'failed', 'disconnected')):
            raise ValueError('Server statistics omitted request accounting')
    return value


@dataclass
class ContainerOwners:
    """Own creation intent before a daemon call can succeed and lose its reply.

    A unique name is retained even when the client fails during creation.
    Retirement inspects that name; only a successful exact daemon inventory
    proves that an unacknowledged create left no native process behind.
    """

    identities: dict[str, str | None] = field(default_factory=dict)
    commands: dict[str, list[str]] = field(default_factory=dict)
    retired: dict[str, dict] = field(default_factory=dict)

    def create(self, name: str, arguments: list[str]) -> str:
        """Claim one unique name before contacting Docker, then bind its full ID."""
        if name in self.identities or name in self.commands:
            raise ValueError('Container name already owned: ' + name)
        self.identities[name] = None
        # A native init forwards cancellation to ordinary helper processes;
        # Linux PID 1 otherwise ignores default-action termination signals.
        self.commands[name] = ['docker', 'create', '--name', name, '--init',
                               *docker_paths.coding_owner_arguments(), *arguments]
        identity = command(self.commands[name]).strip()
        if not re.fullmatch(r'[0-9a-f]{64}', identity):
            raise ValueError('Docker create omitted its full container identity')
        self.identities[name] = identity
        return identity

    def retire(self, name: str) -> dict:
        """Retire exactly one claimed name, preserving uncertain daemon failures."""
        if name not in self.identities:
            raise ValueError('Cannot retire an unowned container: ' + name)
        if self.identities[name] is None:
            value = docker_paths.inspect_owned_container(name)
            if value is None:
                return {'Running': False, 'Pid': 0, 'Status': 'absent'}
            self.identities[name] = value['Id']
        return retire_owned(name)

    def remove(self, name: str) -> None:
        """Forget an admission container only after its native exit and daemon removal."""
        state = self.retire(name)
        if state['Status'] != 'absent':
            command(['docker', 'rm', name])
        self.retired[name] = state
        del self.identities[name]

    def close(self) -> dict:
        """Try every owner even if one cannot retire; an unknown owner blocks successors."""
        retired, errors = dict(self.retired), {}
        for name in self.identities:
            try:
                retired[name] = self.retire(name)
            except Exception as error:
                errors[name] = repr(error)
        return {'passed': not errors, 'retired': retired, 'errors': errors}


def container_state(name: str) -> dict:
    """Ask the daemon, rather than an earlier receipt, whether native owners remain."""
    return json.loads(command(['docker', 'inspect', name]))[0]['State']


def require_retired(name: str) -> dict:
    """Authenticate retirement before observers close or another cell is admitted."""
    state = container_state(name)
    if state['Running'] or state['Pid'] != 0 or state['Status'] not in {'exited', 'created'}:
        raise ValueError('Owned native container has not retired: ' + name)
    return state


def retire_owned(name: str) -> dict:
    """Exceptionally retire this exact container; callers preserve the original failure."""
    if container_state(name)['Running']:
        command(['docker', 'stop', '--time', '30', name], timeout=40)
    return require_retired(name)


def admission_outcome(returncode: int, output: str) -> AdmissionOutcome:
    """Only a controlled native capacity rejection may probe a smaller context."""
    if returncode == 0 and '--dry-run complete: preflight validation passed' in output:
        return AdmissionOutcome.ACCEPTED
    capacity = ('Memory plan validation failed' in output or (
        'ExpertOverlay has no resident captured-prefill shape with complete expert coverage under the physical BOM'
        in output and 'requires_additional_bytes=' in output))
    if returncode == 1 and capacity:
        return AdmissionOutcome.CAPACITY_REJECTED
    raise ValueError('Unexpected native admission outcome; context was not retried')


def run_retired(owners: ContainerOwners, name: str, arguments: list[str], *, log: Path,
                timeout: float | None = None, input_text: str | None = None
                ) -> tuple[subprocess.CompletedProcess, dict]:
    """Join a helper's native exit and retained logs before releasing ownership.

    Attached output can lose a fast child's final bytes. Docker's completed
    log is the single output authority; the attach client only supplies stdin
    and waits. Output streams remain separate for JSON decoders and diagnostics.
    """
    if name in owners.identities or name in owners.commands:
        raise ValueError('Helper container name already belongs to another lifetime')
    try:
        owners.create(name, arguments)
        result = subprocess.run(['docker', 'start', '--attach',
            *(['--interactive'] if input_text is not None else []), name], text=True,
            input=input_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        state = require_retired(name)
        if (state['Status'] != 'exited' or state['ExitCode'] != result.returncode
                or state['OOMKilled'] or state['Error']):
            raise ValueError('Helper command disagrees with its native retirement')
        transcript = subprocess.run(['docker', 'logs', name], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        log.write_text(transcript.stdout + transcript.stderr)
        result = subprocess.CompletedProcess(result.args, result.returncode,
                                              transcript.stdout, transcript.stderr)
    except BaseException as error:
        if isinstance(error, subprocess.TimeoutExpired):
            output = error.stdout or b''
            try:
                log.write_bytes(output if isinstance(output, bytes) else output.encode('utf-8'))
            except OSError as logging_error:
                error.add_note('Helper log publication failed: ' + repr(logging_error))
        try:
            owners.remove(name)
        except Exception as cleanup_error:
            # The owner remains registered for the enclosing lifetime/outer
            # daemon audit. Preserve cancellation or the original native fault.
            error.add_note('Helper retirement remains unproven: ' + repr(cleanup_error))
        raise
    else:
        owners.remove(name)
        return result, state


def admit_context(cell: CodingCell, name: str, devices: list[str], owners: ContainerOwners) -> dict:
    """Describe metadata in the same image, then prove maximum or adjacent PMA bounds."""
    describe = ['--network', 'none', '--cap-add', 'SYS_NICE',
                '--security-opt', 'seccomp=unconfined',
                *docker_paths.mounts([(cell.model.parent, '/model', True)]),
                '--entrypoint', '/usr/local/bin/llaminar_native_tool_evidence_decoder',
                cell.image, '--describe-model', '/model/' + cell.model.name]
    described, _ = run_retired(owners, name + '-metadata', describe,
                               log=cell.output / 'model-metadata.log', timeout=30)
    if described.returncode:
        raise ValueError('Exact-image model metadata helper failed: ' + described.stdout)
    model = json.loads(described.stdout)
    write_json(cell.output / 'model-metadata.json', model)

    def probe(context: int) -> AdmissionOutcome:
        """Retain each independent native result and prove its exact container exited."""
        probe_name = name + '-admit-' + str(context)
        launch = [*devices,
                  *docker_paths.mounts([(cell.model.parent, '/model', True)]),
                  cell.image, *cell.server_arguments(context, dry_run=True)]
        # Model/plan construction is not governed by a coding-turn deadline.
        result, state = run_retired(owners, probe_name, launch, log=cell.output / f'admit-{context}.log')
        outcome = admission_outcome(result.returncode, result.stdout + result.stderr)
        write_json(cell.output / f'admit-{context}.json', {
            'context_tokens': context, 'outcome': outcome.value,
            'command': owners.commands[probe_name],
            'retired_container': state})
        return outcome

    result = select_context(ContextPolicy(model['context_tokens'], model['context_alignment_tokens']), probe)
    write_json(cell.output / 'context-admission.json', result)
    return result


def generation_policy(configuration: dict) -> tuple[str, str]:
    """Reject future or incomplete generation choices before admitting a workload."""
    generation = configuration['runtime']['generation']
    mtp, verifier = generation.get('mtp_policy'), generation.get('mtp_verify_mode')
    if (not isinstance(mtp, str) or not re.fullmatch(r'off|dynamic|depth_[1-9][0-9]*', mtp)
            or verifier not in {'greedy', 'speculative-sampling'}):
        raise ValueError('Invalid canonical coding generation policy')
    return mtp, verifier


def runtime_evidence(perf: dict, configuration: dict) -> dict:
    """Use the canonical physical graph, topology, MTP, transfer and feature validators."""
    validate_memory_authority(perf)
    contract = validate_server_execution_contract(perf)
    selection = validate_automatic_selection(perf, automatic_selection_policy(configuration['e2e']))
    features, records = contract.features, perf['records']
    expected_mtp, expected_verifier = generation_policy(configuration)
    if not features.prefix_cache or features.mtp != (expected_mtp != 'off'):
        raise ValueError('App runtime changed selected prefix-cache or MTP policy')
    if expected_mtp == 'dynamic' and features.mtp_depth_policy != 'dynamic':
        raise ValueError('App runtime disabled the selected dynamic controller')
    if features.mtp and features.mtp_verify_mode != expected_verifier:
        raise ValueError('App runtime changed the selected MTP verification policy')
    if expected_mtp.startswith('depth_') and (
            features.mtp_depth_policy != 'fixed'
            or features.mtp_max_depth != int(expected_mtp.removeprefix('depth_'))):
        raise ValueError('App runtime changed the selected fixed MTP depth')
    checks = {}
    if contract.uses_gpu:
        checks['graph'] = validate_graph_capture_policy(records, contract.device_kinds,
            require_prefill_lifecycle=True, decode_requirement=DecodeGraphRequirement.REPLAY)
        checks['transfers'] = validate_gpu_host_transfer_policy(records, device_kinds=contract.device_kinds)
        checks['attention'] = validate_flash_attention_plan_policy(records, contract.attention_device_kinds)
        if features.mtp and features.mtp_depth_policy == 'dynamic':
            if contract.device_kinds == frozenset({'cuda'}):
                checks['mtp'] = validate_cuda_dynamic_mtp_device_generation_policy(records,
                    expected_minimum_depth=features.mtp_min_depth, expected_maximum_depth=features.mtp_max_depth)
            else:
                checks['mtp'] = validate_host_scheduled_mtp_device_generation_policy(records,
                    device_kinds=contract.device_kinds, expected_minimum_depth=features.mtp_min_depth,
                    expected_maximum_depth=features.mtp_max_depth)
        if features.mtp and features.current_batch_llep:
            checks['llep'] = validate_llep_verifier_policy(records)
    error = validate_runtime_feature_policy(records, features,
        MovementEvidence.FORBIDDEN if features.residency_maintenance == 'off' else MovementEvidence.NOT_APPLICABLE,
        terminal_movement=perf.get('terminal_movement', ()))
    archive = app_archive_compaction_assessment(records)
    return {'passed': error is None and archive['passed'] and all(row.error is None for row in checks.values()),
            'features': asdict(features), 'selection': selection, 'feature_error': error,
            'archive_compaction': archive,
            'checks': {name: {key: sorted(value) if isinstance(value, (set, frozenset)) else value
                              for key, value in asdict(check).items()} for name, check in checks.items()}}


def fresh_observation(output: Path, identity: str, *, after: float = 0) -> dict:
    """Wait for a metadata observer boundary, independently of response duration."""
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        path = output / 'resources/latest.json'
        if path.exists():
            value = json.loads(path.read_text())
            cell = value.get('cells', {}).get('app', {})
            if value.get('errors') or cell.get('errors'):
                raise ValueError('Resource observer failed: ' + repr(value.get('errors', []) + cell.get('errors', [])))
            if (value.get('observed', 0) >= after and cell.get('container_id') == identity
                    and cell.get('inference_pids') and cell.get('stats_observation')):
                return value
        time.sleep(.2)
    raise ValueError('Resource observer failed to publish a fresh native boundary')


def run_app(cell: CodingCell) -> dict:
    """Retain every independent audit and retire all owned native resources on failure.

    The resource observer has its own retirement receipt. The cell's final
    completion flag is published only after observer closure and every audit,
    so a queued successor cannot mistake intermediate retirement for success.
    """
    cell.output.mkdir(parents=True, exist_ok=False)
    archive, perf_dir = cell.output / 'archive', cell.output / 'perf'
    archive.mkdir()
    perf_dir.mkdir()
    name = 'llaminar-opencode-' + uuid.uuid4().hex
    observer_name = name + '-resources'
    url = 'http://127.0.0.1:' + str(cell.port)
    state = {'schema': 1, 'complete': False, 'passed': False, 'phase': 'admission',
             'started': time.time(), 'image': cell.image, 'configuration': cell.configuration,
             'model_file': str(cell.model), 'container': name, 'errors': [],
             'native_retired': False, 'observer_retired': False}
    observer_state = {'complete': False, 'phase': 'admission', 'container_id': None,
                      'model': cell.model.name, 'devices': cell.backends,
                      'archive_host_directory': str(archive)}
    driver = [sys.executable, str(Path(__file__).with_name('gpu_driver_diagnostics.py'))]
    server_created = observer_created = driver_started = False
    owners = ContainerOwners()
    logger = logfile = None
    baseline = window = stats_final = None
    interrupted = None

    def publish() -> None:
        """Keep lifecycle and the observer's independent stop boundary atomic."""
        write_json(cell.output / 'controller.json', state)
        write_json(cell.output / 'resource-controller.json', observer_state)

    def audit(label: str, action):
        """Preserve a failed audit while allowing independent evidence to finish."""
        try:
            value = action()
            if isinstance(value, dict) and value.get('passed') is False:
                raise ValueError(label + ' reported failure')
            return value
        except Exception as error:
            state['errors'].append(label + ': ' + repr(error))
            return None

    publish()
    try:
        command([*driver, 'begin', '--state', str(cell.output / 'driver-state.json')])
        driver_started = True
        devices = docker_paths.device_args(cell.image, cell.backends)
        admission = admit_context(cell, name, devices, owners)
        state['context_admission'] = admission
        launch = [*devices,
            *docker_paths.mounts([(cell.model.parent, '/model', True), (archive, '/prefix-cache', False),
                                 (perf_dir, '/perf', False)]),
            '--env', 'LLAMINAR_LOG_LEVEL=INFO', '--env', 'LLAMINAR_TRACE_GENERATED_TOKENS=1',
            '--env', 'LLAMINAR_PERF_STATS_JSON=/perf/perf.rank-{rank}.json',
            '--env', 'LLAMINAR_ENABLE_SERVER_SHUTDOWN_ENDPOINT=1',
            cell.image, *cell.server_arguments(admission['context_tokens'])]
        state.update(container_id=owners.create(name, launch),
                     command=owners.commands[name])
        server_created = True
        observer_state['container_id'] = state['container_id']
        manifest = {'backends': [kind.lower() for kind in cell.backends.split('+')],
                    'cells': {'app': {'controller': '/evidence/resource-controller.json'}},
                    'filesystems': {'evidence': '/evidence', 'cache': '/models'}}
        write_json(cell.output / 'resource-manifest.json', manifest)
        publish()
        owners.create(observer_name, [*devices, '--pid', 'host', '--cgroupns', 'host',
            *docker_paths.mounts([(Path(__file__).parent, '/scripts', True),
                                 (cell.output, '/evidence', False), (archive.parent, '/models', True)]),
            '--entrypoint', 'python3', cell.image, '/scripts/resource_usage_observer.py',
            '--manifest', '/evidence/resource-manifest.json', '--output', '/evidence/resources', '--interval', '5'])
        observer_created = True
        command(['docker', 'start', observer_name])
        state['server_started'] = time.time()
        command(['docker', 'start', name])
        logfile = (cell.output / 'server.log').open('w')
        logger = subprocess.Popen(['docker', 'logs', '--follow', name], stdout=logfile, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + readiness_timeout_seconds(cell.configuration['e2e'])
        while True:
            if not container_state(name)['Running']:
                raise ValueError('Server exited before readiness')
            try:
                if metadata(url, '/health').get('status') == 'ok':
                    break
            except OSError:
                pass
            if time.monotonic() >= deadline:
                raise ValueError('Server did not publish readiness within its declared startup policy')
            time.sleep(.5)
        with RuntimeStatsObserver(lambda: metadata(url), cell.output / 'stats-webapp-observations.jsonl'):
            baseline = fresh_observation(cell.output, state['container_id'])
            write_json(cell.output / 'sealed-baseline.json', baseline)
            stats = baseline['cells']['app']['stats_observation']['stats']
            if (stats['configuration']['context_tokens'] != admission['context_tokens']
                    or stats['configuration']['prefix_cache']['ram_budget_bytes_per_participant'] != RAM_MIB * 1024**2
                    or stats['configuration']['prefix_cache']['disk_budget_bytes'] != DISK_MIB * 1024**2):
                raise ValueError('Running server differs from admitted workload/cache budgets')
            state.update(phase='app', model=stats['model'], workload_started=time.time())
            observer_state['phase'] = 'app'
            publish()
            client = [sys.executable, str(Path(__file__).with_name('opencode_tool_stress.py')),
                      '--base-url', url, '--model', state['model'], '--opencode', cell.opencode,
                      '--context-length', str(admission['context_tokens']), '--max-tokens', str(OUTPUT_TOKENS),
                      '--gate', 'stress', '--case', 'webapp', '--iterations', '1', '--concurrency', '1',
                      '--measure-prefix-reuse', '--output', str(cell.output / 'app')]
            write_json(cell.output / 'app-command.json', {'command': client})
            with (cell.output / 'app.log').open('w') as log:
                process = subprocess.Popen(client, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                returncode, interrupted_by_harness = wait_for_client(process)
            state['workload_returncode'] = returncode
            if returncode or interrupted_by_harness:
                state['errors'].append('OpenCode stress gate exited ' + str(returncode))
            state['workload_finished'] = time.time()
            fresh_observation(cell.output, state['container_id'], after=state['workload_finished'])
            window = WorkloadWindow(baseline=baseline['observed'], started=state['workload_started'],
                finished=state['workload_finished'], maximum_sample_gap_seconds=20,
                maximum_stats_age_seconds=10, page_size_bytes=os.sysconf('SC_PAGE_SIZE'))
            write_json(cell.output / 'workload-window.json', asdict(window))
            stats_final = metadata(url)
            write_json(cell.output / 'stats-final.json', stats_final)
    except BaseException as error:
        state['errors'].append('workload: ' + repr(error))
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            interrupted = error
    finally:
        state['phase'] = 'retirement'
        if server_created:
            def shutdown():
                """Require HTTP-authorized shutdown and the native exit-zero frontier."""
                if not container_state(name)['Running']:
                    raise ValueError('Native server exited before administrative shutdown')
                with urlopen(Request(url + '/admin/shutdown', data=b'', method='POST'), timeout=10) as response:
                    if response.status != 202 or json.load(response).get('status') != 'shutting_down':
                        raise ValueError('Administrative shutdown was not accepted')
                code = int(command(['docker', 'wait', name]))
                retired = require_retired(name)
                if code or retired['ExitCode'] != 0 or retired['OOMKilled'] or retired['Error']:
                    raise ValueError('Native retirement failed')
                state['normal_shutdown'] = True
                return retired
            audit('normal shutdown', shutdown)
            retired = audit('native retirement', lambda: owners.retire(name))
            if retired is not None:
                state.update(native_retired=True, retired_container=retired)
                observer_state.update(complete=True, phase='retired')
        else:
            state['native_retired'] = True  # No serving start was issued.
            observer_state.update(complete=True, phase='not_started')
        audit('retirement publication', publish)
        if logger is not None:
            def close_log():
                """Join the already retired native stream before reading token evidence."""
                try:
                    if logger.wait(timeout=30) != 0:
                        raise ValueError('Native log follower failed')
                finally:
                    if logger.poll() is None:
                        logger.terminate()
                        logger.wait(timeout=10)
                    logfile.close()
            audit('log retirement', close_log)
        if observer_created:
            def close_observer():
                """Require natural observer closure after native retirement, preserving exceptional cleanup."""
                try:
                    if int(command(['docker', 'wait', observer_name])):
                        raise ValueError('Resource observer exited nonzero')
                finally:
                    retired = owners.retire(observer_name)
                    state['observer_retired'] = not retired['Running']
            audit('resource observer retirement', close_observer)
        else:
            state['observer_retired'] = True
        state['owned_containers'] = owners.close()
        if not state['owned_containers']['passed']:
            state['errors'].append('Native container retirement remains unproven')
        audit('retirement evidence', lambda: write_json(
            cell.output / 'native-retirement.json', state['owned_containers']))
        if state.get('normal_shutdown') and state['owned_containers']['passed']:
            perf = audit('ranked PerfStats', lambda: collect_and_publish_ranked_perf_stats(perf_dir / 'perf.json'))
            if perf is not None:
                def runtime_audit():
                    """Keep complete runtime policy results alongside original rank artifacts."""
                    value = runtime_evidence(perf, cell.configuration)
                    write_json(cell.output / 'runtime-policies.json', value)
                    return value
                audit('runtime policies', runtime_audit)
                if window is not None:
                    def resources():
                        """Reconcile every host sample with sealed baseline and native PMA ownership."""
                        capacity = {tier: stats_final['prefix_cache']['storage']['tiers'][tier]['capacity_bytes']
                                    for tier in ('ram', 'disk')}
                        with (cell.output / 'resources/observations.jsonl').open() as stream:
                            value = assess_observations(observations=(json.loads(line) for line in stream),
                                perf=perf, cell='app', container_id=state['container_id'], boot_id=baseline['boot_id'],
                                window=window, cache_capacity=capacity)
                        write_json(cell.output / 'resource-growth.json', value)
                        return value
                    audit('resource growth', resources)
            def archive_audit():
                """Read only journal metadata and inode extents after all cache owners have retired."""
                result, _ = run_retired(owners, name + '-archive-audit', [
                    '--network', 'none', '--user', '0',
                    *docker_paths.mounts([(archive, '/prefix-cache', True), (Path(__file__).parent, '/scripts', True)]),
                    '--entrypoint', 'python3', cell.image, '/scripts/prefix_archive_metadata_audit.py',
                    '--directory', '/prefix-cache', '--budget-bytes', str(DISK_MIB * 1024**2)],
                    log=cell.output / 'archive-audit.log')
                if result.returncode:
                    raise ValueError('Retired prefix archive metadata audit failed')
                value = json.loads(result.stdout)
                write_json(cell.output / 'archive-audit.json', value)
                return value
            audit('archive ownership', archive_audit)
            def native_audit():
                """Use the exact serving image's decoder for every recorded response and call."""
                with (cell.output / 'native-audit.log').open('w') as log:
                    subprocess.run([sys.executable, str(Path(__file__).with_name('opencode_native_audit.py')),
                        '--exchanges', str(cell.output / 'app'), '--server-log', str(cell.output / 'server.log'),
                        '--server-start-unix', str(state['server_started']), '--decoder-image', cell.image,
                        '--model', str(cell.model), '--output', str(cell.output / 'native-audit')],
                        stdout=log, stderr=subprocess.STDOUT, check=True)
                app = validate_app_report(json.loads((cell.output / 'app/stress.json').read_text()),
                    model=state['model'], context_tokens=state['context_admission']['context_tokens'])
                native = json.loads((cell.output / 'native-audit/result.json').read_text())
                validate_native_coverage(native, app)
                state['app'] = app
                return native
            audit('complete native/app evidence', native_audit)
        if state['owned_containers']['passed'] and server_created:
            def reclaim_cache():
                """Keep matrix disk use bounded while retaining every small audit receipt."""
                result, _ = run_retired(owners, name + '-reclaim', [
                    '--network', 'none', '--user', '0',
                    *docker_paths.mounts([(archive, '/prefix-cache', False),
                        (cell.output / 'native-retirement.json', '/retirement.json', True),
                        (Path(__file__).parent, '/scripts', True)]),
                    '--entrypoint', 'python3', cell.image, '/scripts/retired_prefix_archive.py',
                    '--directory', '/prefix-cache', '--retirement', '/retirement.json'],
                    log=cell.output / 'archive-reclamation.log')
                if result.returncode:
                    raise ValueError('Retired prefix archive reclamation failed')
                value = json.loads(result.stdout)
                write_json(cell.output / 'archive-reclamation.json', value)
                return value
            audit('cache reclamation', reclaim_cache)
            state['owned_containers'] = owners.close()
            if not state['owned_containers']['passed']:
                state['errors'].append('Archive helper retirement remains unproven')
        if state['owned_containers']['passed']:
            for owned_name in tuple(owners.identities):
                audit('container removal ' + owned_name, lambda: owners.remove(owned_name))
        if driver_started:
            audit('driver closure', lambda: command([*driver, 'finish', '--state', str(cell.output / 'driver-state.json'),
                '--report', str(cell.output / 'driver-diagnostics.json')], timeout=None))
        state.update(complete=True, finished=time.time(), phase='complete')
        state['passed'] = (not state['errors'] and state.get('workload_returncode') == 0
                           and state.get('normal_shutdown') is True and 'app' in state
                           and window is not None and state['native_retired'] and state['observer_retired'])
        publish()
    if interrupted is not None:
        raise interrupted
    if (not state['native_retired'] or not state['observer_retired']
            or not state['owned_containers']['passed']):
        raise RuntimeError('Cell cleanup could not prove native/observer retirement; no successor may run')
    return state
