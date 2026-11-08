#!/usr/bin/env python3
"""Project one explicit frozen history into the existing exact-cell campaign.

This module does not discover cells, generate references, run a model or judge
numerics. The canonical driver retains staging, prerequisites, topology, path
checks, all-stage execution and six CSVs. A logits-only reference is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

HANDOFF_FIELDS = ('LLAMINAR_PARITY_HISTORY_REPLAY_FILE',
                  'LLAMINAR_PARITY_HISTORY_REPLAY_SHA256',
                  'LLAMINAR_PARITY_HISTORY_REPLAY_CELL')
MAXIMUM_DESCRIPTOR_BYTES = 16 * 1024 * 1024


def _require(condition, message):
    """Reject malformed diagnostic metadata before any canonical execution."""
    if not condition:
        raise ValueError(message)


def _digest(value):
    """Use the existing shared corpus metadata serialization convention."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _sha(path, maximum):
    """Hash only bounded small metadata; never model or checkpoint payloads."""
    path = Path(path)
    size = path.stat().st_size
    _require(0 < size <= maximum, 'Frozen history metadata extent is invalid')
    with path.open('rb') as input:
        value = input.read(maximum + 1)
    _require(len(value) == size and len(value) <= maximum, 'Frozen history metadata changed while reading')
    return hashlib.sha256(value).hexdigest(), value


def _source(descriptor):
    """Bind original reference stat fields, including same-size replacements."""
    path = Path(descriptor['path']).resolve(strict=True)
    value = path.stat()
    actual = dict(path=str(path), device=value.st_dev, inode=value.st_ino,
                  bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns)
    original = json.loads(descriptor['object_identity'])
    _require(type(original) is dict and set(original) == set(actual) and
             type(original['path']) is str and
             all(type(original[k]) is int for k in actual if k != 'path') and
             original == actual and path.is_file() and value.st_size > 0 and
             descriptor['bytes'] == value.st_size and descriptor['name'] == path.name,
             'Frozen original source stat identity changed')


def _tokens(metadata, key):
    """Read the original canonical metadata token field, without snapshots."""
    entries = [line.split(':', 1)[1].strip() for line in metadata.splitlines()
               if line.startswith(key + ':')]
    _require(len(entries) == 1, 'Missing or duplicate frozen reference token metadata')
    return [int(value.strip()) for value in entries[0].split(',') if value.strip()]


@dataclass(frozen=True)
class HistoryReplaySelection:
    """Passive original metadata pin, never a topology or numerical certificate."""
    path: Path
    sha256: str
    descriptor: dict
    original_bytes: bytes

    def launch(self, command, environment):
        """Project explicit workers without changing the original process population.

        Single-rank histories retain their existing whole-affinity projection.
        Distributed histories retain original socket binding and per-rank workers;
        the typed native fixture authenticates the original cohort and inventory.
        No command is executed here, and ordinary campaigns never call this.
        """
        from model_kld_corpus import physical_workers
        for key in ('GOMP_CPU_AFFINITY', 'KMP_HW_SUBSET', 'KMP_PLACE_THREADS', 'OMP_THREAD_LIMIT'):
            _require(not os.environ.get(key, '').strip() and not environment.get(key, '').strip(),
                     'Frozen replay cannot inherit additional CPU resource control: ' + key)
        configuration = self.descriptor['expected_config'].get('configuration', {})
        workers = configuration.get('n_threads')
        participants = configuration.get('mpi_procs')
        _require(type(workers) is int and workers > 0 and
                 type(participants) is int and 1 <= participants <= 8 and
                 workers * participants <= physical_workers(),
                 'Frozen replay worker request exceeds original rank-population physical CPU resources')
        command = list(command)
        # CTest's original guardian remains the process/cleanup authority. Split
        # only its closed public envelope, project the existing MPI body, then
        # reattach the identical prefix; never execute or resolve another owner.
        guardian = []
        if command and Path(command[0]).name == 'llaminar_mpi_session':
            _require(len(command) >= 5 and Path(command[0]).is_absolute() and
                     Path(command[0]).is_file() and command[1] == '--session-base' and
                     Path(command[2]).is_absolute() and Path(command[2]).is_dir() and
                     command[3] == '--',
                     'Frozen replay has an incomplete original process-owner envelope')
            guardian, command = command[:4], command[4:]
        _require(len(command) >= 7 and Path(command[0]).name == 'mpirun' and
                 command[1:7] == ['-np', str(participants), '--bind-to', 'socket', '--map-by', 'socket'],
                 'Frozen replay needs the original canonical rank-population socket launch')
        # Lifecycle controls are required, while MPI owns native transport
        # selection. Preserve one explicitly registered mechanism verbatim;
        # MPI validates its installed values. Never add a default or admit an
        # unrelated resource directive that could become an affinity owner.
        cursor = 7
        required_mca = {'mpi_leave_pinned': '1', 'orte_allowed_exit_without_sync': '1'}
        optional_mechanism = 'btl_vader_single_copy_mechanism'
        seen = set()
        exports = {}
        while cursor < len(command) and command[cursor] in ('--mca', '-x'):
            if command[cursor] == '--mca':
                _require(cursor + 2 < len(command),
                         'Frozen replay has incomplete MPI resource or transport override')
                name, value = command[cursor + 1:cursor + 3]
                _require(name not in seen and
                         ((name in required_mca and required_mca[name] == value) or
                          (name == optional_mechanism and type(value) is str and bool(value))),
                         'Frozen replay has unsupported MPI resource or transport overrides')
                seen.add(name)
                cursor += 3
            else:
                _require(cursor + 1 < len(command), 'Frozen replay has incomplete MPI environment export')
                key, separator, value = command[cursor + 1].partition('=')
                _require(separator and key.isascii() and key.isidentifier() and key not in exports and
                         environment.get(key) == value,
                         'Frozen replay MPI export differs from the original registered environment')
                exports[key] = cursor + 1
                cursor += 2
        # The owned canonical launcher publishes an explicit application
        # boundary. Legacy direct MPI declarations omit it; preserve either
        # original form and keep resource exports before that boundary.
        application = cursor + int(cursor < len(command) and command[cursor] == '--')
        _require(set(required_mca).issubset(seen) and len(command[application:]) == 2 and
                 Path(command[application]).is_absolute() and
                 command[application + 1] == '--gtest_filter=' + self.descriptor['canonical_cell'],
                 'Frozen replay cannot represent additional MPI or native launch controls')
        if participants == 1:
            command[4], command[6] = 'none', 'slot'
        # Multi-rank histories retain the original one-rank-per-socket binding.
        # Actual typed fixture/inventory validates CPU cohort and bound NUMA;
        # this projection neither creates a topology nor substitutes affinity.
        values = dict(environment)
        filters = values.get('LLAMINAR_PERF_STATS_FILTER', '').split(',')
        _require('forward_graph' in filters, 'Frozen replay lost its canonical runtime evidence filter')
        family = 'kernel.cpu_moe_router_q8_worker_rows'
        if family not in filters:
            filters.append(family)
        values.update(OMP_NUM_THREADS=str(workers), MKL_NUM_THREADS=str(workers),
                      OPENBLAS_NUM_THREADS='1', OMP_PLACES='cores', OMP_PROC_BIND='false',
                      OMP_DYNAMIC='false', OMP_NESTED='false', KMP_AFFINITY='disabled',
                      LLAMINAR_PERF_STATS_FILTER=','.join(filters))
        # CTest's campaign command embeds KEY=value exports. Rewrite those from
        # the same projected environment so an old -x value cannot win over the
        # newly admitted worker/filter setting. Preserve every unrelated export.
        for key, index in exports.items():
            command[index] = key + '=' + values[key]
        resource_keys = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                         'OMP_PLACES', 'OMP_PROC_BIND', 'OMP_DYNAMIC', 'OMP_NESTED',
                         'KMP_AFFINITY', 'LLAMINAR_PERF_STATS_FILTER')
        additions = [item for key in resource_keys if key not in exports
                     for item in ('-x', key + '=' + values[key])]
        command[cursor:cursor] = additions
        return tuple(guardian + command), values

    def verify_unchanged(self):
        """Recheck metadata/source owners around the existing canonical execution."""
        digest, _ = _sha(self.path, MAXIMUM_DESCRIPTOR_BYTES)
        _require(digest == self.sha256, 'Explicit frozen history file changed')
        source = self.descriptor['source']
        for descriptor in source['shards'] + ([source['shared_predictor']] if 'shared_predictor' in source else []):
            _source(descriptor)
        metadata_path = Path(self.descriptor['reference_directory']) / 'metadata.txt'
        metadata_digest, metadata = _sha(metadata_path, 1024 * 1024)
        _require(metadata_digest == self.descriptor['full_stage_metadata_digest'],
                 'Frozen full-stage metadata changed; explicit regeneration is required')
        text = metadata.decode('utf-8')
        expected = self.descriptor['continuation_ids']
        _require(_tokens(text, 'token_ids') == self.descriptor['prompt_ids'] and
                 _tokens(text, 'decode_tokens')[:len(expected)] == expected,
                 'Frozen full-stage reference prompt or D+1 teacher IDs differ')

    def require_binding(self, artifact_directories):
        """Require the actual fixture's projection receipt in its original output."""
        expected = dict(schema='llaminar.model-kld-history-binding.v1', descriptor_sha256=self.sha256,
            **{key:self.descriptor[key] for key in ('canonical_cell', 'history_id', 'source_digest',
                'full_stage_metadata_digest', 'prompt', 'prompt_ids', 'continuation_ids',
                'decode_rows', 'reference_directory')})
        _require(len(artifact_directories) == 1, 'Frozen history needs one original exact-cell artifact directory')
        path = Path(artifact_directories[0]) / 'history_replay_binding.json'
        _, raw = _sha(path, MAXIMUM_DESCRIPTOR_BYTES)
        _require(json.loads(raw) == expected, 'Actual canonical fixture did not bind the original frozen history')
        geometry_path = Path(artifact_directories[0]) / 'history_replay_prefill_geometry.json'
        _, raw = _sha(geometry_path, MAXIMUM_DESCRIPTOR_BYTES)
        geometry = json.loads(raw)
        fixed = dict(schema='llaminar.model-kld-history-prefill-geometry.v1',
                     descriptor_sha256=self.sha256, canonical_cell=self.descriptor['canonical_cell'],
                     history_id=self.descriptor['history_id'], capture_mode='production_serving_ladder',
                     logical_rows=len(self.descriptor['prompt_ids']))
        _require(type(geometry) is dict and all(geometry.get(k) == v for k, v in fixed.items()),
                 'Actual fixture omitted its frozen production-serving geometry identity')
        buckets = geometry.get('bucket_rows')
        floor = geometry.get('minimum_rows')
        physical = geometry.get('physical_rows')
        native = geometry.get('native_replay_required')
        observed = geometry.get('native_replay_observed')
        _require(type(buckets) is list and buckets and all(type(n) is int and n > 0 for n in buckets) and
                 buckets == sorted(set(buckets)) and type(floor) is int and floor > 0 and
                 type(physical) is int and type(native) is bool and type(observed) is bool and native == observed,
                 'Actual fixture has incomplete production-serving replay geometry')
        selected = next((n for n in buckets if n >= len(self.descriptor['prompt_ids'])), None)
        _require(selected is not None and (physical == selected if native else physical == len(self.descriptor['prompt_ids'])),
                 'Actual fixture replayed a different physical bucket from its declared serving ladder')

        from model_kld_corpus import physical_workers
        cpu = geometry.get('cpu_execution')
        workers = self.descriptor['expected_config'].get('configuration', {}).get('n_threads')
        participants = self.descriptor['expected_config']['configuration']['mpi_procs']
        _require(type(cpu) is dict and type(workers) is int and workers > 0 and
                 cpu.get('requested_workers') == workers and cpu.get('max_workers') == workers and
                 cpu.get('dynamic') is False and cpu.get('unbound') is True and cpu.get('places') == 'cores' and
                 (cpu.get('allowed_cpus') == sorted(os.sched_getaffinity(0)) if participants == 1 else
                  type(cpu.get('allowed_cpus')) is list and bool(cpu['allowed_cpus']) and
                  cpu['allowed_cpus'] == sorted(set(cpu['allowed_cpus'])) and
                  set(cpu['allowed_cpus']).issubset(os.sched_getaffinity(0))) and
                 type(cpu.get('physical_cores')) is int and (cpu['physical_cores'] == physical_workers() if participants == 1 else cpu['physical_cores'] <= physical_workers()) and cpu['physical_cores'] >= workers and
                 type(cpu.get('cpu_compute_required')) is bool and
                 type(cpu.get('parallel_work_observed')) is bool and
                 (not cpu['cpu_compute_required'] or cpu['parallel_work_observed']),
                 'Frozen replay did not prove the declared workers and original unbound CPU resources')

        if participants > 1:
            masks = []; sockets = set(); nodes = set()
            for rank in range(participants):
                _, raw = _sha(Path(artifact_directories[0]) / f'history_replay_cpu_rank_{rank}.json', MAXIMUM_DESCRIPTOR_BYTES)
                record = json.loads(raw)
                _require(type(record) is dict and record.get('schema') == 'llaminar.model-kld-history-cpu-rank.v1' and
                         record.get('descriptor_sha256') == self.sha256 and
                         type(record.get('world_rank')) is int and record['world_rank'] == rank and
                         type(record.get('world_size')) is int and record['world_size'] == participants and
                         record.get('runner_retired') is True,
                         'Frozen history lacks complete original rank retirement/resources')
                node = record.get('original_node_id'); socket = record.get('socket_id'); numa = record.get('numa_node')
                _require(all(type(value) is int and value >= 0 for value in (node, socket, numa)) and
                         (node, socket) not in sockets and (node, numa) not in nodes,
                         'Frozen history has aliased original physical socket/NUMA resources')
                sockets.add((node, socket)); nodes.add((node, numa))
                observed_cpu = record.get('cpu_execution')
                _require(type(observed_cpu) is dict and observed_cpu.get('requested_workers') == workers and
                         observed_cpu.get('max_workers') == workers and observed_cpu.get('dynamic') is False and
                         observed_cpu.get('unbound') is True and observed_cpu.get('places') == 'cores' and
                         type(observed_cpu.get('physical_cores')) is int and observed_cpu['physical_cores'] >= workers and
                         observed_cpu.get('cpu_compute_required') is True and observed_cpu.get('parallel_work_observed') is True,
                         'Frozen history rank lacks actual requested CPU compute workers')
                mask = observed_cpu.get('allowed_cpus')
                _require(type(mask) is list and mask and all(type(value) is int and value >= 0 for value in mask) and
                         mask == sorted(set(mask)) and set(mask).issubset(os.sched_getaffinity(0)) and
                         all(not set(mask).intersection(previous) for previous in masks),
                         'Frozen history rank CPU affinities overlap or differ from original available resources')
                masks.append(set(mask))
                if rank == 0:
                    _require(observed_cpu == cpu, 'Frozen history root resource record differs from measured prefill')

    def bind(self, artifact_root):
        """Publish an exact immutable handoff beside the original campaign artifacts."""
        self.verify_unchanged()
        path = Path(artifact_root) / 'history-replay.json'
        with path.open('xb') as output:
            output.write(self.original_bytes)
        return {HANDOFF_FIELDS[0]: str(path.resolve()), HANDOFF_FIELDS[1]: self.sha256,
                HANDOFF_FIELDS[2]: self.descriptor['canonical_cell']}


def prepare_history_replay(path, exact_cell, model_files):
    """Validate one registered cell's explicit history before preflight/staging.

    The source digest and stat descriptors come from the frozen corpus. The full
    stage metadata digest is added only after the existing reference generator
    has regenerated this exact prompt/teacher trajectory with all stages enabled.
    """
    path = Path(path).resolve(strict=True)
    digest, raw = _sha(path, MAXIMUM_DESCRIPTOR_BYTES)
    value = json.loads(raw)
    _require(value.get('schema') == 'llaminar.model-kld-history-replay.v1' and
             value.get('canonical_cell') == exact_cell and '.ProductionParity/' in exact_cell,
             'Frozen history belongs to a different original exact cell')
    _require(value.get('reference_scope') == 'full_stages',
             'Frozen history requires full-stage regeneration; LM-only pack is insufficient')
    for key in ('corpus_manifest_digest', 'source_digest', 'reference_report_digest', 'full_stage_metadata_digest'):
        text = value.get(key)
        _require(type(text) is str and len(text) == 64 and all(c in '0123456789abcdef' for c in text),
                 'Missing original frozen corpus/source/reference identity')
    source = value['source']
    _require(value['source_digest'] == _digest(source) and source.get('architecture') == 'qwen4exp' and
             source.get('main_layers') == 48 and source.get('expert_count') == 512 and
             source.get('vocabulary') == 248320 and source.get('reference_arithmetic') == 'cpu_fp32' and
             source.get('reference_execution') == 'incremental_cached' and len(source.get('shards', [])) == 4,
             'Frozen history has foreign original QFN geometry/source')
    main_sources = {Path(part['path']).resolve(strict=True) for part in source['shards']}
    declared_sources = {Path(model).resolve(strict=True) for model in model_files}
    permitted_sources = main_sources | ({Path(source['shared_predictor']['path']).resolve(strict=True)}
                                        if 'shared_predictor' in source else set())
    _require(len(main_sources) == 4 and main_sources <= declared_sources <= permitted_sources,
             'Frozen history model differs from complete CTest original split declaration')
    _require(type(value.get('expected_config')) is dict and value['expected_config'] and
             type(value.get('arithmetic')) is dict and
             set(value['arithmetic']) == {'gdn_projection', 'moe_router', 'hc_up', 'gdn_output', 'fa_output', 'gdn_qkv'},
             'Frozen history lacks original canonical placement/arithmetic provenance')
    for key in ('history_id', 'context_group'):
        name = value.get(key)
        _require(type(name) is str and 0 < len(name) <= 128 and name not in ('.', '..') and
                 all(c.isascii() and (c.isalnum() or c in '_.-') for c in name), 'Invalid frozen history identity')
    depth = value.get('decode_rows')
    _require(type(depth) is int and 0 < depth <= 256 and
             type(value.get('prompt')) is str and value['prompt'] and
             type(value.get('reference_directory')) is str and value['reference_directory'],
             'Invalid frozen history work/reference selection')
    for key in ('prompt_ids', 'continuation_ids'):
        ids = value.get(key)
        _require(type(ids) is list and ids and all(type(i) is int and 0 <= i < 248320 for i in ids),
                 'Invalid frozen original token history')
    _require(len(value['prompt_ids']) <= 4096 and len(value['continuation_ids']) == depth + 1,
             'Frozen history needs exact D+1 Pending teacher IDs')
    selection = HistoryReplaySelection(path, digest, value, raw)
    selection.verify_unchanged()
    return selection
