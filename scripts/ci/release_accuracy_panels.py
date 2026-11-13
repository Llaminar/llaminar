#!/usr/bin/env python3
## @file release_accuracy_panels.py
# @brief Frozen-history release panels in the existing production parity framework.
"""Frozen-history release panels in the existing production parity framework.

The blessed inventory owns topology and serial-control selection. Each model
uses its registered CPU/FP32 PyTorch implementation and the original GGUF;
all stages, native capture, reset/prefix and movement obligations remain active.
Only final vocabulary rows enter the shared FP64 scorer. Compact CSVs and
original fixture receipts are retained independently of large reference packs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from full_vocabulary_metrics import full_vocabulary_metrics
from production_artifacts import digest, validate_manifest, write_json

SCHEMA = 'llaminar.release-accuracy-panel.v1'
SCOPE = 'blessed_serving_serial_controls'
# Required model identities, not a second topology or precision matrix.
MODELS = {
    'Qwen3.8-27B-IQ4_XS.gguf': ('qwen38-dense', 'qwen35'),
    'Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf': ('qfn', 'qwen4exp'),
}
CSV_FIELDS = ['model', 'case', 'history', 'context_group', 'phase', 'decode_offset', 'prompt_tokens', 'logical_position',
              'kl_nats', 'top1_same', 'top5_overlap', 'total_variation', 'maximum_kl']


def sha(path: Path) -> str:
    """Digest compact evidence files, never GGUFs or numerical tensor payloads."""
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def panel() -> dict:
    """Reuse the authored multi-prompt panel and its declared continuation depth."""
    definition = json.loads((ROOT / 'scripts/ci/release_accuracy_prompts.json').read_text())
    if definition['schema'] != 'llaminar.release-accuracy-prompts.v1' or definition['base_panel'] != 'qfn_accuracy_prompts.json':
        raise ValueError('unrecognized authored release panel')
    path = ROOT / 'scripts/ci' / definition['base_panel']
    authored = json.loads(path.read_text())
    if authored['schema'] != 'llaminar.qfn-accuracy-prompts.v1':
        raise ValueError('unrecognized base prompt panel')
    result = authored
    for case in definition['context_cases']:
        result['cases'].append(dict(case, prompt=definition['context_text'] * definition['context_repetitions']))
    result['schema'] = definition['schema']
    names, prompts = set(), set()
    for case in result['cases']:
        name = case['name']
        if (not isinstance(name, str) or not name or any(not (c.isascii() and (c.isalnum() or c in '-_.')) for c in name)
                or name in ('.', '..') or name in names or not isinstance(case['prompt'], str) or not case['prompt']
                or not case['group'] or type(case.get('decode_rows', result['decode_steps'])) is not int
                or not 1 <= case.get('decode_rows', result['decode_steps']) <= 256):
            raise ValueError('malformed or duplicate authored release history')
        if 'context_tokens' in case and (type(case['context_tokens']) is not int or not 1 <= case['context_tokens'] <= 4096):
            raise ValueError('invalid authored context boundary')
        # Boundary histories deliberately share their source text; their exact
        # predeclared prefix lengths become distinct original token histories.
        identity = (case['prompt'], case.get('context_tokens'))
        if identity in prompts:
            raise ValueError('duplicate authored prompt history')
        names.add(name)
        prompts.add(identity)
    return result


def project(inventory: dict, blessed: dict, revision: str) -> list[dict]:
    """Select every required family's exact source-owned blessed serial controls.

    Serving, stress and benchmark coverage share this same typed inventory.
    Numerical-diagnostic candidates never acquire release eligibility here.
    """
    if inventory.get('scope') != 'all' or inventory.get('source_revision') != revision:
        raise ValueError('accuracy panels require the exact full source inventory')
    controls = {}
    for row in inventory['cells']:
        if row['configuration'].get('inventory_role', 'canonical') != 'canonical':
            continue
        identity = row['configuration']['id']
        if identity in controls:
            raise ValueError('duplicate canonical numerical control identity')
        controls[identity] = row
    selected = {}
    for row in validate_manifest(blessed, revision):
        config = row['configuration']
        model = MODELS.get(Path(config['model']).name)
        if model is None:
            raise ValueError("release accuracy manifest contains a model outside the blessed scope")
        control = controls.get(config['runtime']['generation']['serial_control_id'])
        if control is None or control['configuration']['runtime']['generation']['mtp_policy'] != 'off':
            raise ValueError('blessed accuracy topology omitted its canonical MTP-off control')
        if Path(control['configuration']['model']).name != Path(config['model']).name:
            raise ValueError('accuracy control changed the blessed model')
        selected[control['case']] = dict(control, model_key=model[0], reference_family=model[1])
    counts = {key: sum(row['model_key'] == key for row in selected.values()) for key, _ in MODELS.values()}
    if set(counts.values()) != {3}:
        raise ValueError('release accuracy inventory must include exactly three blessed controls for dense Qwen3.8 and QFN')
    return [selected[key] for key in sorted(selected)]


def source_pins(row: dict) -> dict:
    """Retain every original GGUF split's metadata through reference retirement."""
    from model_kld_corpus import identity
    from run_production_parity_campaigns import _split_gguf_members
    return {str(path.resolve(strict=True)): identity(path)
            for part in row['model_files'] for path in _split_gguf_members(Path(part))}


def reference_implementation(root: Path) -> dict[str, str]:
    """Hash installed source inputs identically with or without Git metadata.

    Only authored Python and dependency/build inputs enter this closure; Python
    bytecode, compiler outputs and model payloads never affect reuse. Installed
    source companions carry these same inputs without a Git index.
    """
    paths = {root / name for name in ('Dockerfile', 'requirements.txt',
        'scripts/ci/release_accuracy_panels.py', 'scripts/ci/qfn_accuracy_prompts.json',
        'scripts/ci/release_accuracy_prompts.json')}
    for name in ('python', 'scripts/docker', 'scripts/build'):
        directory = root / name
        if not directory.is_dir():
            raise ValueError('reference implementation source directory is missing: ' + name)
        for path in directory.rglob('*'):
            relative = path.relative_to(directory)
            if (not path.is_file() or any(part in ('__pycache__', 'build', 'dist') for part in relative.parts)
                    or path.suffix in ('.pyc', '.pyo', '.o', '.so', '.a')):
                continue
            if name != 'python' or path.suffix == '.py':
                paths.add(path)
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def reference_key(row: dict, authored: dict) -> str:
    """Bind reusable reference packs to real source metadata and reference code."""
    import torch
    sources = source_pins(row)
    from importlib.metadata import distributions
    implementation = reference_implementation(ROOT)
    packages = sorted((item.metadata['Name'], item.version) for item in distributions())
    return digest(dict(sources=sources, implementation=implementation, panel=authored,
                       torch=str(torch.__version__), packages=packages,
                       threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                       cpu_capability=torch.backends.cpu.get_cpu_capability(), torch_config=torch.__config__.show(),
                       environment={k: v for k, v in os.environ.items()
                                    if k.startswith(('ATEN_', 'TORCH_', 'MKL_', 'OMP_', 'DNNL_', 'ONEDNN_', 'OPENBLAS_'))},
                       family=row['reference_family']))


class ReferenceRetirementError(RuntimeError):
    """The reference owner failed to retire; no further model work is admissible."""


def references(row: dict, authored: dict, root: Path) -> tuple[Path, dict]:
    """Collect/resume immutable histories under the canonical reference lease.

    Checkpoints retain completed histories but remain unqualified until the
    whole authored panel and actual model retirement succeed. An interrupted
    unfinished history is retained separately before that same request restarts;
    completed teacher inputs and numerical observations are never regenerated.
    """
    from python.reference import create_reference_model
    from python.reference.generate_qwen35_pipeline_snapshots import run_prefill_and_decode, write_metadata
    from model_kld_corpus import unchanged
    import uuid
    if not callable(create_reference_model):
        raise RuntimeError('registered CPU/FP32 PyTorch reference implementations are unavailable')
    key = reference_key(row, authored)
    destination = root / key
    root.mkdir(parents=True, exist_ok=True)
    lease_path = Path(tempfile.gettempdir()) / 'llaminar-production-parity-reference-generation.lock'
    descriptor = os.open(lease_path, os.O_CREAT | os.O_CLOEXEC | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, 'r+') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX)
        receipt = destination / 'references.json'
        if destination.exists():
            value = json.loads(receipt.read_text())
            if (value.get('schema') != 'llaminar.release-panel-references.v1' or value.get('key') != key
                    or value.get('authored_panel_digest') != digest(authored)):
                raise ValueError('incompatible frozen reference pack: ' + str(destination))
            if value.get('complete') is True:
                if value.get('reference_owner_retired') is not True:
                    raise ValueError('complete reference pack has no retired owner')
                return destination, value
        else:
            destination.mkdir()
            value = dict(schema='llaminar.release-panel-references.v1', key=key, complete=False,
                         reference_owner_retired=False, authored_panel_digest=digest(authored), histories=[])
        histories = value['histories']
        if [h['id'] for h in histories] != [c['name'] for c in authored['cases'][:len(histories)]]:
            raise ValueError('reference checkpoint reordered or duplicated authored histories')
        value['reference_owner_retired'] = False
        write_json(receipt, value)
        pins, model = source_pins(row), None
        try:
            kwargs = {'cache_bytes': 0} if row['reference_family'] == 'qwen4exp' else {}
            model = create_reference_model(row['reference_family'], row['configuration']['model'], **kwargs)
            vocabulary = int(model.hf_model.config.vocab_size)
            seen = set()
            for index, case in enumerate(authored['cases']):
                directory = destination / case['name']
                depth = case.get('decode_rows', authored['decode_steps'])
                if index < len(histories):
                    original = histories[index]
                    if (original['authored_prompt_digest'] != digest(case) or original['vocabulary'] != vocabulary
                            or original['decode_rows'] != depth or len(original['continuation_ids']) != depth + 1
                            or sha(directory / 'metadata.txt') != original['metadata_sha256']
                            or tuple(original['prompt_ids']) in seen):
                        raise ValueError('completed frozen reference history changed')
                    seen.add(tuple(original['prompt_ids']))
                    continue
                if directory.exists():
                    incomplete = destination / 'incomplete'
                    incomplete.mkdir(exist_ok=True)
                    directory.rename(incomplete / (case['name'] + '-' + uuid.uuid4().hex))
                prompt, expected = case['prompt'], None
                if 'context_tokens' in case:
                    expected = model.tokenizer(prompt)['input_ids'][:case['context_tokens']]
                    if len(expected) != case['context_tokens']:
                        raise ValueError('authored context text cannot supply the declared boundary')
                    prompt = model.tokenizer.decode(expected, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                elif row['reference_family'] == 'qwen4exp':
                    expected = case.get('expected_prompt_ids')
                _, tokens, teachers = run_prefill_and_decode(model, prompt, depth, directory, expected_token_ids=expected)
                if tuple(tokens) in seen or len(teachers) != depth + 1:
                    raise ValueError('duplicate prompt or incomplete frozen teacher history')
                seen.add(tuple(tokens))
                extra = []
                if row['reference_family'] == 'qwen4exp':
                    from python.reference.generate_qwen4exp_pipeline_snapshots import reference_fields
                    extra = reference_fields()
                write_metadata(directory, row['configuration']['model'], model, prompt, tokens,
                               depth, teachers, extra_metadata_lines=extra)
                unchanged(pins)
                histories.append(dict(id=case['name'], context_group=case['group'], prompt=prompt,
                                      prompt_ids=tokens, continuation_ids=teachers,
                                      decode_rows=depth, vocabulary=vocabulary, authored_prompt_digest=digest(case),
                                      metadata_sha256=sha(directory / 'metadata.txt')))
                write_json(receipt, value)
            unchanged(pins)
        finally:
            if model is not None:
                try:
                    model.close()
                except Exception as error:
                    raise ReferenceRetirementError('original PyTorch reference owner did not retire') from error
                value['reference_owner_retired'] = True
                write_json(receipt, value)
        if reference_key(row, authored) != key:
            raise ValueError('original source or reference implementation changed during collection')
        value['complete'] = True
        write_json(receipt, value)
        return destination, value


def score_history(directory: Path, reference: Path, history: dict, row: dict, binding: dict) -> list[dict]:
    """Score actual full-vocabulary rows and require identical repeated replays."""
    import numpy as np
    depth, vocabulary = history['decode_rows'], history['vocabulary']
    observations = defaultdict(list)
    for path in sorted(directory.rglob('llaminar_snapshots/**/manifest.csv')):
        with path.open() as source:
            records = list(csv.DictReader(source))
        for record in records:
            if record['key'] != 'LM_HEAD':
                continue
            phase, offset = record['phase'].lower(), int(record['step'])
            if phase not in ('prefill', 'decode') or (phase == 'prefill' and offset != -1) or (phase == 'decode' and offset not in range(depth)):
                raise ValueError('unexpected native panel row coordinate')
            elements = int(record['elements'])
            payload = Path(record['file'])
            if (record['scalar_type'] != 'FP32' or int(record['exported_elements']) != elements or
                    elements < vocabulary or elements % vocabulary or payload.stat().st_size != elements * 4):
                raise ValueError('native panel row is partial, trimmed or has the wrong vocabulary')
            values = np.memmap(payload, dtype='<f4', mode='r')[-vocabulary:]
            if not np.isfinite(values).all():
                raise ValueError('nonfinite native final logits')
            observations[(phase, offset)].append(np.array(values, copy=True))
    expected = [('prefill', -1)] + [('decode', i) for i in range(depth)]
    if set(observations) != set(expected):
        raise ValueError('missing or extra native frozen-history observations')
    result = []
    for phase, offset in expected:
        copies = observations[(phase, offset)]
        if any(not np.array_equal(copies[0].view('u4'), other.view('u4')) for other in copies[1:]):
            raise ValueError('reset or participant replay changed the exact final-logit row')
        name = 'LM_HEAD.npy' if phase == 'prefill' else f'decode_step{offset}_LM_HEAD.npy'
        gold = np.load(reference / name, allow_pickle=False)
        if gold.dtype != np.float32 or gold.shape[-1] != vocabulary:
            raise ValueError('reference panel omitted the original complete FP32 vocabulary')
        metric = full_vocabulary_metrics(gold.reshape(-1, vocabulary)[-1], copies[0], vocabulary=vocabulary)
        maximum = binding['native_limits']['prefill_maximum' if phase == 'prefill' else 'decode_maximum']
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError('canonical model omitted its numerical acceptance budget')
        result.append(dict(model=row['model_key'], case=row['case'], history=history['id'],
                           context_group=history['context_group'], phase=phase, decode_offset=offset,
                           prompt_tokens=len(history['prompt_ids']),
                           logical_position=len(history['prompt_ids']) - 1 if phase == 'prefill' else len(history['prompt_ids']) + offset,
                           kl_nats=metric['kl_nats'], top1_same=int(metric['top1_same']),
                           top5_overlap=metric['top5_overlap'], total_variation=metric['total_variation'],
                           maximum_kl=maximum))
    return result


def summarize(rows: list[dict]) -> list[dict]:
    """Give each prompt equal weight; retain phase, group and offset boundaries."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for offset in ({row['decode_offset'], 'all'} if row['phase'] == 'decode' else {-1}):
            key = (row['model'], row['case'], row['context_group'], row['phase'], str(offset))
            groups[key][row['history']].append(row)
    result = []
    for key, prompts in sorted(groups.items()):
        item = dict(zip(('model', 'case', 'context_group', 'phase', 'decode_offset'), key))
        item['prompts'] = len(prompts)
        for metric in ('kl_nats', 'top1_same', 'top5_overlap'):
            item['mean_' + metric] = math.fsum(math.fsum(r[metric] for r in values) / len(values)
                                                  for values in prompts.values()) / len(prompts)
        item['passed'] = all(r['kl_nats'] < r['maximum_kl'] for values in prompts.values() for r in values)
        result.append(item)
    return result


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Retain exact row identities and round-trippable numeric observations."""
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    """Run every selected cell/history through the canonical registered lifecycle."""
    from model_parity_inventory import discover, export_manifest, InventoryScope
    from types import SimpleNamespace
    import run_production_parity_campaigns as parity
    import torch
    from model_kld_corpus import physical_workers
    torch.set_num_threads(physical_workers())
    torch.set_num_interop_threads(1)
    selected = discover(SimpleNamespace(build_dir=args.build_dir, manifest=None, backend='.*',
                         campaign='.*', cell='.*'), InventoryScope.ALL)
    inventory = export_manifest(selected, args.source_revision, InventoryScope.ALL)
    blessed = json.loads(args.manifest.read_text())
    plan, authored = project(inventory, blessed, args.source_revision), panel()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema=SCHEMA, complete=False, passed=False, source_revision=args.source_revision,
                  cpu_isa=args.cpu_isa, runtime_image=args.runtime_image, manifest_digest=digest(blessed),
                  plan_digest=digest(plan), panel_digest=digest(authored), cells=[], failures={},
                  prerequisites_digest=digest(json.loads(args.prerequisites_report.read_text())))
    result.update(acceptance_policy='initial_absolute_canonical_model_limits',
                  acceptance_scope=SCOPE, calibrated=False)
    rows = []
    write_json(args.output / 'panel.json', result)
    if parity.prepare_model_fixture(args.build_dir, None)[0] != 0:
        raise ValueError('canonical real-model fixture preparation failed')
    write_json(args.output / 'native-inventory.json', inventory)
    for row in plan:
        cell = dict(case=row['case'], model=row['model_key'], complete=False, passed=False, histories=[])
        result['cells'].append(cell)
        try:
            reference, pack = references(row, authored, args.reference_cache)
            cell['reference_key'] = pack['key']
            for history in pack['histories']:
                lane = args.output / 'native' / digest(row['case']) / history['id']
                lane.mkdir(parents=True)
                descriptor = dict(schema='llaminar.accuracy-panel-history.v1', canonical_cell=row['case'],
                                  model=row['configuration']['model'], reference_scope='full_stages',
                                  reference_arithmetic='cpu_fp32', reference_execution='incremental_cached',
                                  reference_directory=str(reference / history['id']),
                                  **{key: value for key, value in history.items() if key not in ('id', 'context_group')})
                write_json(lane / 'history.json', descriptor)
                environment = dict(os.environ, LLAMINAR_ACCURACY_PANEL_HISTORY=str(lane / 'history.json'),
                                   LLAMINAR_ACCURACY_PANEL_SHA256=sha(lane / 'history.json'))
                command = [sys.executable, str(ROOT / 'scripts/ci/run_production_parity_campaigns.py'),
                           '--build-dir', str(args.build_dir), '--installed-build-receipt', '/src/installed-tests.json',
                           '--reuse-passed-preflight-report', str(args.prerequisites_report),
                           '--exact-cell', row['case'], '--model-ramdisk-root', str(args.model_ramdisk_root),
                           '--persistent-model-cache-dir', 'cache', '--report', str(lane / 'run.json')]
                with (lane / 'run.log').open('w') as log:
                    executed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                report = json.loads((lane / 'run.json').read_text())
                native = report.get('result') or {}
                directories = native.get('artifact_directories', [])
                if (executed.returncode or native.get('return_code') != 0 or native.get('artifact_contract_passed') is not True
                        or report.get('preflight_return_code') != 0 or len(directories) != 1):
                    raise ValueError('canonical full-stage parity failed: ' + str(lane / 'run.log'))
                directory = Path(directories[0])
                binding = json.loads((directory / 'accuracy_panel_binding.json').read_text())
                if {key: value for key, value in binding.items() if key not in ('native_limits', 'descriptor_sha256')} != descriptor or binding['descriptor_sha256'] != environment['LLAMINAR_ACCURACY_PANEL_SHA256']:
                    raise ValueError('native fixture changed the original frozen-history handoff')
                if sha(reference / history['id'] / 'metadata.txt') != history['metadata_sha256']:
                    raise ValueError('reference changed during native parity execution')
                observations = score_history(directory, reference / history['id'], history, row, binding)
                rows.extend(observations)
                proof = lane / 'evidence'
                proof.mkdir()
                for source in [*directory.glob('*.csv'), directory / 'accuracy_panel_binding.json']:
                    shutil.copyfile(source, proof / source.name)
                shutil.copyfile(reference / history['id'] / 'metadata.txt', proof / 'reference-metadata.txt')
                # The source-bound prerequisite report is retained once per
                # ISA. Do not replicate its complete inventory in every row.
                native_proof = {key: report[key] for key in ('gtest_case', 'preflight_return_code',
                                                             'fixture_return_code', 'model_staging_return_code')}
                native_proof['result'] = {key: native[key] for key in ('return_code', 'artifact_contract_passed',
                    'outcome', 'gtest_cases', 'validated_artifact_file_count')}
                cell['histories'].append(dict(id=history['id'], passed=True, binding=binding,
                                             rows=len(observations), native_report=native_proof,
                                             evidence_files={p.name: sha(p) for p in sorted(proof.iterdir())}))
            cell['complete'] = cell['passed'] = True
        except ReferenceRetirementError as error:
            result['failures'][row['case']] = str(error)
            write_json(args.output / 'panel.json', result)
            raise
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            result['failures'][row['case']] = str(error)
            # Retain partial rows, then retire this whole installed companion.
            # A failed native receipt cannot authorize more model/device work.
            break
        write_json(args.output / 'panel.json', result)
    summary = summarize(rows)
    write_csv(args.output / 'accuracy-rows.csv', CSV_FIELDS, rows)
    write_csv(args.output / 'accuracy-summary.csv', list(summary[0]) if summary else ['model'], summary)
    result.update(complete=all(cell['complete'] for cell in result['cells']),
                  passed=bool(summary) and all(cell['passed'] for cell in result['cells']) and all(r['passed'] for r in summary),
                  summary=summary, rows=rows,
                  files={name: sha(args.output / name) for name in ('accuracy-rows.csv', 'accuracy-summary.csv')})
    write_json(args.output / 'panel.json', result)
    return result


def companion(args) -> int:
    """Use the existing cache owner and installed-test lifecycle inside the driver.

    ARC mounts the host cache read-only in the runner pod. Only this existing
    Docker control boundary may acquire its writer lease and create the sibling
    reference cache. Release that lease before any native campaign acquires it.
    """
    from types import SimpleNamespace
    import run_production_parity_campaigns as parity
    import run_production_pipeline as pipeline
    images = json.loads(args.images.read_text())
    with parity.model_staging_workspace(args.model_ramdisk_root, Path('cache'), None) as workspace:
        references = workspace.root / 'release-accuracy-references'
        references.mkdir(exist_ok=True)
    build = SimpleNamespace(models=args.models, model_ramdisk_root=args.model_ramdisk_root,
                            reference_cache_root=references)
    pair = json.loads((args.work / 'images.json').read_text())
    pipeline.require_image(images[pipeline.ImageRole.TEST_RUNNER.value], pair['source'], args.cpu_isa,
                           pipeline.ImageRole.TEST_RUNNER)
    pipeline.run_test_runner(images, ['python3', '-B', 'scripts/ci/release_accuracy_panels.py', 'run',
        '--build-dir', 'build_v2_integration', '--manifest', '/ci-results/manifest.json',
        '--prerequisites-report', '/ci-results/prerequisites/prerequisites.json',
        '--source-revision', pair['source']['revision'], '--cpu-isa', args.cpu_isa,
        '--runtime-image', pair['images'][args.cpu_isa]['id'],
        '--model-ramdisk-root', str(args.model_ramdisk_root), '--output', '/ci-results/panel'],
        build, args.work, 'accuracy-native.log')
    return 0


def main(argv=None) -> int:
    """Execute only inside the exact-source full-matrix installed test companion."""
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest='operation', required=True)
    observer = operations.add_parser('run', help='observe in the exact installed full-matrix test companion')
    observer.add_argument('--build-dir', type=Path, required=True)
    observer.add_argument('--manifest', type=Path, required=True)
    observer.add_argument('--prerequisites-report', type=Path, required=True)
    observer.add_argument('--source-revision', required=True)
    observer.add_argument('--runtime-image', required=True)
    observer.add_argument('--reference-cache', type=Path, default=Path('/reference-cache/release-panels'))
    observer.add_argument('--output', type=Path, required=True)
    controller = operations.add_parser('companion', help='delegate to the installed companion as the existing cache owner')
    controller.add_argument('--images', type=Path, required=True)
    controller.add_argument('--work', type=Path, required=True)
    controller.add_argument('--models', type=Path, required=True)
    for operation in (observer, controller):
        operation.add_argument('--cpu-isa', choices=('AVX512', 'AVX2'), required=True)
        operation.add_argument('--model-ramdisk-root', type=Path, required=True)
    args = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    return companion(args) if args.operation == 'companion' else (0 if run(args)['passed'] else 1)


if __name__ == '__main__':
    raise SystemExit(main())
