## @file release_accuracy_evidence.py
# @brief Authenticate both ISA accuracy panels before stress, benchmarks or promotion.
"""Authenticate both ISA accuracy panels before stress, benchmarks or promotion.

The canonical inventory and original fixture bindings own coverage and limits.
This reader recomputes row coverage, prompt-balanced summaries and CSV identity;
it never executes a model or substitutes historical canaries for panel evidence.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from production_artifacts import digest
from release_accuracy_panels import SCHEMA, SCOPE, panel, project, sha, summarize


def validate_panel(directory: Path, pair: dict, manifest: dict, inventory: dict, isa: str,
                   models_root: str) -> dict:
    """Replay complete native histories and exact final-logit row coverage."""
    import run_production_pipeline as pipeline
    report = json.loads((directory / 'panel.json').read_text())
    native = json.loads((directory / 'native-inventory.json').read_text())
    if pipeline.remap_manifest(native, Path(models_root)) != inventory:
        raise ValueError('panel native inventory differs from source-authenticated E2E inventory')
    authored = panel()
    planned = project(native, manifest, pair['source']['revision'])
    if (report.get('schema') != SCHEMA or report.get('complete') is not True or report.get('passed') is not True
            or report.get('failures') != {} or report.get('source_revision') != pair['source']['revision']
            or report.get('runtime_image') != pair['images'][isa]['id'] or report.get('cpu_isa') != isa
            or report.get('manifest_digest') != digest(manifest) or report.get('plan_digest') != digest(planned)
            or report.get('panel_digest') != digest(authored)
            or report.get('acceptance_policy') != 'initial_absolute_canonical_model_limits'
            or report.get('acceptance_scope') != SCOPE
            or report.get('calibrated') is not False):
        raise ValueError('accuracy panel is failed, partial or belongs to another source/image/ISA')
    cells = report['cells']
    if [cell['case'] for cell in cells] != [row['case'] for row in planned]:
        raise ValueError('accuracy panel changed the complete blessed topology selection')
    expected = set()
    expected_order = []
    limits = {}
    logical_positions = {}
    for cell, row in zip(cells, planned):
        if (cell.get('complete') is not True or cell.get('passed') is not True
                or cell.get('model') != row['model_key']
                or [h['id'] for h in cell['histories']] != [h['name'] for h in authored['cases']]):
            raise ValueError('panel omitted, reordered or duplicated a frozen history')
        for history, selected in zip(cell['histories'], authored['cases']):
            binding, run = history['binding'], history['native_report']
            native_result = run.get('result') or {}
            depth = selected.get('decode_rows', authored['decode_steps'])
            if (history.get('passed') is not True or history['rows'] != depth + 1
                    or run.get('gtest_case') != row['case'] or run.get('preflight_return_code') != 0
                    or run.get('fixture_return_code') != 0 or run.get('model_staging_return_code') != 0
                    or native_result.get('return_code') != 0 or native_result.get('artifact_contract_passed') is not True
                    or native_result.get('outcome') != 'completed' or native_result.get('gtest_cases') != [row['case']]
                    or binding.get('canonical_cell') != row['case'] or binding.get('model') != row['configuration']['model']
                    or binding.get('reference_scope') != 'full_stages'
                    or binding.get('authored_prompt_digest') != digest(selected)
                    or ('context_tokens' not in selected and binding.get('prompt') != selected['prompt'])
                    or ('context_tokens' in selected and len(binding.get('prompt_ids', [])) != selected['context_tokens'])
                    or binding.get('decode_rows') != depth
                    or len(binding.get('continuation_ids', [])) != depth + 1):
                raise ValueError('accuracy evidence lost the original full-stage native fixture proof')
            descriptor = {k: v for k, v in binding.items() if k not in ('descriptor_sha256', 'native_limits')}
            lane = directory / 'native' / digest(row['case']) / history['id']
            if (json.loads((lane / 'history.json').read_text()) != descriptor
                    or sha(lane / 'history.json') != binding['descriptor_sha256']
                    or json.loads((lane / 'evidence/accuracy_panel_binding.json').read_text()) != binding):
                raise ValueError('accuracy panel binding bytes changed')
            from run_production_parity_campaigns import REQUIRED_CSV_HEADERS
            for name in REQUIRED_CSV_HEADERS:
                if not (lane / 'evidence' / name).is_file():
                    raise ValueError('accuracy panel omitted canonical CSV evidence: ' + name)
            evidence = lane / 'evidence'
            if history.get('evidence_files') != {p.name: sha(p) for p in sorted(evidence.iterdir()) if p.is_file()}:
                raise ValueError('canonical native accuracy evidence changed after execution')
            from model_kld_history_replay import _tokens
            metadata = (evidence / 'reference-metadata.txt').read_text()
            if (sha(evidence / 'reference-metadata.txt') != binding['metadata_sha256']
                    or _tokens(metadata, 'token_ids') != binding['prompt_ids']
                    or _tokens(metadata, 'decode_tokens') != binding['continuation_ids']):
                raise ValueError('panel reference metadata differs from frozen native inputs')
            for phase, offset in [('prefill', -1)] + [('decode', i) for i in range(depth)]:
                key = (row['model_key'], row['case'], history['id'], selected['group'], phase, offset)
                expected.add(key)
                expected_order.append(key)
                limits[key] = binding['native_limits']['prefill_maximum' if phase == 'prefill' else 'decode_maximum']
                count = len(binding['prompt_ids'])
                logical_positions[key] = (count, count - 1 if phase == 'prefill' else count + offset)
    seen = set()
    for row in report['rows']:
        key = tuple(row[k] for k in ('model', 'case', 'history', 'context_group', 'phase', 'decode_offset'))
        values = [row[k] for k in ('kl_nats', 'top5_overlap', 'total_variation', 'maximum_kl')]
        if (key not in expected or key in seen or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
                or row['maximum_kl'] != limits[key]
                or (row.get('prompt_tokens'), row.get('logical_position')) != logical_positions[key]
                or not 0 <= row['kl_nats'] < row['maximum_kl'] or type(row['top1_same']) is not int or row['top1_same'] not in (0, 1)
                or not 0 <= row['top5_overlap'] <= 1 or not 0 <= row['total_variation'] <= 1):
            raise ValueError('accuracy panel row is duplicated, nonfinite, misaligned or outside model limits')
        seen.add(key)
    if seen != expected or report.get('summary') != summarize(report['rows']):
        raise ValueError('accuracy panel omitted rows or changed prompt-balanced summaries')
    if [tuple(r[k] for k in ('model', 'case', 'history', 'context_group', 'phase', 'decode_offset'))
            for r in report['rows']] != expected_order:
        raise ValueError('accuracy observations reordered their frozen histories')
    for name in ('accuracy-rows.csv', 'accuracy-summary.csv'):
        if report['files'].get(name) != sha(directory / name):
            raise ValueError('accuracy CSV changed after scoring')
    with (directory / 'accuracy-rows.csv').open() as source:
        observed = list(csv.DictReader(source))
    if observed != [{k: str(v) for k, v in row.items()} for row in report['rows']]:
        raise ValueError('accuracy CSV differs from its actual scored rows')
    with (directory / 'accuracy-summary.csv').open() as source:
        if list(csv.DictReader(source)) != [{k: str(v) for k, v in row.items()} for row in report['summary']]:
            raise ValueError('accuracy summary CSV differs from prompt-balanced observations')
    return report


def validate_matched_histories(reports: dict) -> None:
    """Require both ISA candidates to consume the same original reference rows."""
    from run_published_image_suite import ISAS
    fields = ('prompt', 'prompt_ids', 'continuation_ids', 'decode_rows', 'vocabulary',
              'metadata_sha256', 'authored_prompt_digest', 'reference_arithmetic', 'reference_execution')
    observations = []
    for isa in ISAS:
        observations.append({cell['case']: {
            'reference_key': cell['reference_key'],
            'histories': [{field: history['binding'][field] for field in fields}
                          for history in cell['histories']]}
            for cell in reports[isa]['cells']})
    if any(value != observations[0] for value in observations[1:]):
        raise ValueError('ISA panels did not consume the same reference implementation and frozen teacher histories')


def admit_accuracy(bundle: Path, preflight_bundle: Path, pair: dict, manifest: dict, correctness: dict) -> dict:
    """Join source/ISA-bound companions, preflight and both complete panels."""
    import run_published_image_suite as suite
    receipt = json.loads((bundle / 'accuracy-receipt.json').read_text())
    inventory = json.loads((preflight_bundle / 'all-cells.json').read_text())
    preflight = json.loads((preflight_bundle / 'preflight-receipt.json').read_text())['prerequisites']
    expected = dict(schema=1, complete=True, images_digest=digest(pair), manifest_digest=digest(manifest),
                    prerequisites_reports={isa: digest(correctness[isa]) for isa in suite.ISAS},
                    models_root=receipt['models_root'], reports={}, test_companions={}, retirement_reports={})
    reports = {}
    for isa in suite.ISAS:
        reports[isa] = validate_panel(bundle / isa.lower() / 'panel', pair, manifest, inventory, isa, receipt['models_root'])
        if reports[isa].get('prerequisites_digest') != digest(preflight[isa]['report']):
            raise ValueError('accuracy panel did not reuse the exact complete source-bound preflight')
        expected['reports'][isa] = digest(reports[isa])
        expected['test_companions'][isa] = preflight[isa]['test_runner_image']
        from driver_evidence import validate_retirement
        cleanup = validate_retirement(json.loads((bundle / isa.lower() / 'outer-cleanup.json').read_text()))
        expected['retirement_reports'][isa] = digest(cleanup)
    if receipt != expected:
        raise ValueError('accuracy receipt differs from the authenticated preflight/image pair')
    validate_matched_histories(reports)
    return reports
