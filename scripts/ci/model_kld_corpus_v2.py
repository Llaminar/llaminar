"""QFN fixed-history extension of the canonical full-vocabulary KL corpus.

This module owns no runner, native state, model loader or numeric defaults.
Original v1 approvals and scoring stay in model_kld_corpus. Every v2 verdict
requires exact history/source/topology bindings and an explicit numerical
policy: approved-baseline calibration or an initial absolute pilot. A pilot
accepts only its required groups, never its diagnostic groups or qualification.
Complete collection without either policy is never GREEN.
"""
from collections import defaultdict
from pathlib import Path
import math

from model_kld_corpus import identity, metadata_digest, relative_file, require, unchanged
from full_vocabulary_metrics import full_vocabulary_metrics

SCHEMA = 'llaminar.model-kld-corpus.v2'
METRIC = 'full_vocabulary_fp32_kl_reference_candidate_temperature1_v1'
GROUP_FIELDS = ('backend', 'topology_id', 'context_group', 'phase')
MAXIMUM_LIMITS = ('mean_kl_nats', 'step_mean_kl_nats', 'row_kl_nats',
                  'paired_mean_kl_delta_nats', 'row_kl_delta_nats',
                  'mean_total_variation', 'tail_fraction')
MINIMUM_LIMITS = ('top1_agreement', 'mean_top5_overlap')
ABSOLUTE_MAXIMUM_LIMITS = tuple(name for name in MAXIMUM_LIMITS if 'delta' not in name)
ARITHMETIC_FIELDS = ('gdn_projection', 'moe_router', 'hc_up', 'gdn_output', 'fa_output', 'gdn_qkv')
PROOFS = ('source_authenticated', 'production_path_authenticated',
          'publication_authenticated', 'native_retired', 'reset_byte_equal')



def _finite_json(value):
    """Reject nonfinite metadata and non-JSON objects before identity/approval comparisons."""
    if type(value) is float:
        require(math.isfinite(value), 'Nonfinite metadata')
    elif type(value) is dict:
        require(all(type(k) is str for k in value), 'Invalid metadata key')
        for item in value.values(): _finite_json(item)
    elif type(value) is list:
        for item in value: _finite_json(item)
    else:
        require(value is None or type(value) in (str, int, bool), 'Invalid JSON metadata')


def _number(value, name, *, negative=False):
    """Require finite scalar metadata without accepting bools or numeric strings."""
    require(type(value) in (int, float) and math.isfinite(value) and
            (negative or value >= 0), 'Invalid finite '+name)
    return float(value)


def _identifier(value):
    """Keep row components unambiguous and artifact-safe, without inferring policy."""
    require(type(value) is str and value and value not in ('.', '..') and
            len(value) <= 128 and all(c.isascii() and (c.isalnum() or c in '-_.') for c in value),
            'Invalid corpus identifier')
    return value


def expected_rows(manifest):
    """Project exact phase/position rows from the immutable Pending teacher history.

    D+1 explicit continuation tokens publish prefill plus D decode rows. The
    last pending token is an owned terminal condition, never another forward.
    """
    rows = []
    for history in manifest['histories']:
        first = len(history['prompt_ids'])
        common = dict(history_id=history['id'], context_group=history['context_group'])
        rows.append(dict(common, row_id=history['id']+':prefill', phase='prefill',
                         decode_offset=None, consumed_tokens=first, input_token=None))
        for offset in range(history['decode_rows']):
            rows.append(dict(common, row_id=history['id']+':decode:'+str(offset), phase='decode',
                             decode_offset=offset, consumed_tokens=first+offset+1,
                             input_token=history['continuation_ids'][offset]))
    return rows


def validate_manifest(manifest):
    """Validate fixed original source, complete histories and explicit topology inventory.

    Source descriptors are authenticated metadata from the original GGUF owner;
    this pure validator never reads a weight payload or grants runtime capacity.
    Synthetic device-free fixtures must label their source provenance honestly.
    """
    _finite_json(manifest)
    require(type(manifest) is dict and manifest.get('schema') == SCHEMA, 'Wrong v2 corpus schema')
    require(manifest.get('metric') == METRIC and type(manifest.get('temperature')) in (int, float) and manifest['temperature'] == 1., 'Wrong corpus metric')
    vocabulary = manifest.get('vocabulary')
    require(type(vocabulary) is int and vocabulary >= 5, 'QFN requires full vocabulary and literal top5')
    source = manifest.get('source', {})
    require(type(source) is dict, 'Invalid source descriptor')
    require(source.get('architecture') == 'qwen4exp' and source.get('vocabulary') == vocabulary and
            source.get('reference_arithmetic') == 'cpu_fp32' and
            source.get('reference_execution') == 'incremental_cached' and
            source.get('main_layers') == 48 and source.get('expert_count') == 512 and
            type(source.get('shards')) is list and len(source['shards']) == 4,
            'Invalid original QFN source/reference authority')
    for shard in source['shards']:
        require(type(shard) is dict and type(shard.get('name')) is str and shard['name'] and
                type(shard.get('path')) is str and shard['path'] and Path(shard['path']).name == shard['name'] and
                type(shard.get('bytes')) is int and shard['bytes'] > 0 and
                type(shard.get('object_identity')) is str and shard['object_identity'], 'Missing original shard identity')
    require(len({s['name'] for s in source['shards']}) == 4, 'Duplicate original shard')
    require(type(source.get('tokenizer_revision')) is str and source['tokenizer_revision'], 'Missing tokenizer identity')
    histories = manifest.get('histories')
    require(type(histories) is list and 0 < len(histories) <= 128, 'Invalid bounded fixed histories')
    seen = set(); prompt_histories = set()
    for history in histories:
        require(type(history) is dict, 'Invalid history')
        name = _identifier(history.get('id')); _identifier(history.get('context_group'))
        require(name not in seen, 'Duplicate history'); seen.add(name)
        for field in ('prompt_ids', 'continuation_ids'):
            ids = history.get(field)
            require(type(ids) is list and ids and all(type(v) is int and 0 <= v < vocabulary for v in ids),
                    'Invalid fixed '+field)
        original_prompt = tuple(history['prompt_ids'])
        require(original_prompt not in prompt_histories, 'Duplicate original prompt history')
        prompt_histories.add(original_prompt)
        count = history.get('decode_rows')
        require(type(count) is int and 0 < count <= 256 and len(history['continuation_ids']) == count+1,
                'Pending teacher history needs exactly D+1 continuation tokens')
        require(type(history.get('prompt')) is str and history['prompt'] and
                type(history.get('reference_directory')) is str and history['reference_directory'] and
                history.get('reference_scope') in ('full_stages', 'final_logits_only'),
                'Missing original prompt/diagnostic reference binding')
    reset = manifest.get('reset_control', {})
    require(type(reset) is dict, 'Invalid reset control')
    require(reset.get('history_id') in seen and
            _identifier(reset.get('replay_id')) not in seen, 'Missing distinct terminal reset control')
    require(sum(1+h['decode_rows'] for h in histories) <= 4096, 'Oversized observed row inventory')
    paired = source.get('shared_predictor')
    if 'shared_predictor' in source:
        require(type(paired) is dict and type(paired.get('path')) is str and paired['path'] and
                type(paired.get('name')) is str and Path(paired['path']).name == paired['name'] and
                type(paired.get('bytes')) is int and paired['bytes'] > 0 and
                type(paired.get('object_identity')) is str and paired['object_identity'],
                'Invalid original inactive shared-predictor descriptor')
    topologies = manifest.get('topologies')
    require(type(topologies) is list and 0 < len(topologies) <= 128, 'Invalid bounded canonical topologies')
    seen = set()
    for topology in topologies:
        require(type(topology) is dict, 'Invalid topology declaration')
        name = _identifier(topology.get('id'))
        require(name not in seen, 'Duplicate topology'); seen.add(name)
        require(type(topology.get('backend')) is str and topology['backend'] and
                type(topology.get('canonical_cell')) is str and topology['canonical_cell'] and
                type(topology.get('expected_config')) is dict and topology['expected_config'] and
                topology.get('runtime_config_digest') == metadata_digest(topology['expected_config']) and
                type(topology.get('arithmetic')) is dict and set(topology['arithmetic']) == set(ARITHMETIC_FIELDS) and
                all(type(v) is str and v for v in topology['arithmetic'].values()),
                'Missing original topology/arithmetic binding')
        configuration = topology['expected_config'].get('configuration', {})
        require(type(configuration) is dict, 'Invalid source-owned configuration document')
        require(configuration.get('shared_predictor') == (paired['path'] if paired else None),
                'Original inactive predictor descriptor/configuration disagree')
    reference = manifest.get('reference', {})
    require(type(reference) is dict and reference.get('format') == 'float32_le_row_major' and
            type(reference.get('report')) is str and reference['report'] and
            type(reference.get('report_digest')) is str and len(reference['report_digest']) == 64 and
            all(c in '0123456789abcdef' for c in reference['report_digest']), 'Missing approved reference identity')
    require(len({r['row_id'] for r in expected_rows(manifest)}) == len(expected_rows(manifest)), 'Duplicate row identity')
    return manifest


def _topology(manifest, topology_id):
    """Find one declared topology without borrowing another backend's policy."""
    matches = [t for t in manifest['topologies'] if t['id'] == topology_id]
    require(len(matches) == 1, 'Unknown topology')
    return matches[0]


def _validate_rows(manifest, report):
    """Reject absent, extra, duplicate, reordered or physically incompatible row metadata."""
    wanted = expected_rows(manifest); rows = report.get('rows')
    require(type(rows) is list and len(rows) == len(wanted), 'Incomplete exact row inventory')
    require(len({r.get('row_id') for r in rows if type(r) is dict}) == len(wanted), 'Duplicate/malformed row')
    for actual, expected in zip(rows, wanted):
        require(type(actual) is dict and all(k in actual and actual[k] == v and type(actual[k]) is type(v)
                for k, v in expected.items()), 'Changed/unmatched teacher position')
        require(actual.get('vocabulary') == manifest['vocabulary'] and type(actual.get('vocabulary')) is int,
                'Wrong full-vocabulary extent')
        metrics = actual.get('metrics', {})
        require(type(metrics) is dict, 'Missing row metrics')
        _number(metrics.get('kl_nats'), 'row KL')
        require(0 <= _number(metrics.get('total_variation'), 'TV') <= 1, 'Invalid TV')
        require(type(metrics.get('top1_same')) is bool, 'Invalid top1 equality')
        overlap = _number(metrics.get('top5_overlap'), 'top5 overlap')
        require(metrics.get('top5_width') == 5 and type(metrics.get('top5_width')) is int and
                0 <= overlap <= 1 and abs(overlap*5-round(overlap*5)) < 1e-12,
                'Top5 must be literal intersection/5')
        for name in ('reference_top5_ids', 'candidate_top5_ids'):
            ids = metrics.get(name)
            require(type(ids) is list and len(ids) == 5 and len(set(ids)) == 5 and
                    all(type(v) is int and 0 <= v < manifest['vocabulary'] for v in ids),
                    'Missing/malformed exact top5 IDs')
        ref_ids, actual_ids = metrics['reference_top5_ids'], metrics['candidate_top5_ids']
        require(overlap == len(set(ref_ids) & set(actual_ids))/5 and
                metrics['top1_same'] == (ref_ids[0] == actual_ids[0]), 'Inconsistent agreement metadata')
        for value in metrics.values():
            if type(value) is float: require(math.isfinite(value), 'Nonfinite metric evidence')
    return rows


def validate_observation(manifest, report):
    """Bind an already scored report to original producer/source/config evidence."""
    validate_manifest(manifest); _finite_json(report)
    require(type(report) is dict and report.get('schema') == 'llaminar.model-kld-metrics.v2' and
            report.get('complete') is True and report.get('metric') == METRIC and
            report.get('corpus_manifest_digest') == metadata_digest(manifest) and
            report.get('source_digest') == metadata_digest(manifest['source']) and
            report.get('reference_report_digest') == manifest.get('reference', {}).get('report_digest'),
            'Mismatched metric/source/reference identity')
    topology = _topology(manifest, report.get('topology_id'))
    require(report.get('backend') == topology['backend'] and
            report.get('runtime_config_digest') == topology['runtime_config_digest'] and
            report.get('arithmetic') == topology['arithmetic'], 'Changed backend/topology/arithmetic')
    require(type(report.get('native_identity')) is dict and report['native_identity'] and
            all(type(k) is str and type(v) is str and v for k, v in report['native_identity'].items()),
            'Missing native source/binary identity')
    require(all(report.get('proofs', {}).get(k) is True for k in PROOFS), 'Missing source/path/publication/reset/retirement proof')
    _validate_rows(manifest, report)
    return report


def _mean(values):
    """Use a stable finite mean, preserving signed paired differences."""
    require(bool(values), 'Empty aggregation group')
    try:
        result = math.fsum(values)/len(values)
    except OverflowError as error:
        raise ValueError('Nonfinite aggregate') from error
    require(math.isfinite(result), 'Nonfinite aggregate')
    return result


def _prompt_mean(rows, value):
    """Give every distinct fixed prompt one vote, irrespective of its continuation length."""
    histories = defaultdict(list)
    for row in rows: histories[row['history_id']].append(value(row))
    return _mean([_mean(values) for values in histories.values()])


def _quantile(values, fraction):
    """Diagnostic linear-interpolated quantile over retained raw rows, never a gate default."""
    values = sorted(values); location = (len(values)-1)*fraction
    low = int(location); high = min(low+1, len(values)-1)
    return values[low]+(values[high]-values[low])*(location-low)


def _limits(calibration, manifest, baseline, approval, group_keys):
    """Admit complete explicit calibration; no implicit numeric ceiling or borrowed baseline."""
    if calibration is None: return {}
    _finite_json(calibration)
    require(type(calibration) is dict and calibration.get('schema') == 'llaminar.model-kld-calibration.v1' and
            calibration.get('corpus_manifest_digest') == metadata_digest(manifest) and
            calibration.get('approved_baseline_digest') == metadata_digest(baseline) and
            calibration.get('baseline_approval_digest') == metadata_digest(approval), 'Unbound calibration')
    result = {}
    require(type(calibration.get('groups')) is list, 'Invalid calibration group inventory')
    for item in calibration['groups']:
        require(type(item) is dict, 'Invalid calibration group')
        key = tuple(item.get(k) for k in GROUP_FIELDS)
        require(key in group_keys and key not in result, 'Unknown/duplicate calibration group')
        maximum = item.get('maximum', {}); minimum = item.get('minimum', {})
        require(set(maximum) == set(MAXIMUM_LIMITS) and set(minimum) == set(MINIMUM_LIMITS),
                'Incomplete explicit calibration limits')
        for name in MAXIMUM_LIMITS: _number(maximum[name], name, negative='delta' in name)
        for name in MINIMUM_LIMITS:
            require(0 <= _number(minimum[name], name) <= 1, 'Invalid agreement floor')
        require(0 <= maximum['tail_fraction'] <= 1 and 0 <= maximum['mean_total_variation'] <= 1,
                'Invalid calibrated probability limit')
        tail = _number(item.get('tail_threshold_kl_nats'), 'tail threshold')
        require(tail <= maximum['row_kl_nats'], 'Tail threshold exceeds catastrophe ceiling')
        result[key] = item
    require(set(result) == set(group_keys), 'Missing calibration group')
    return result


def _absolute_pilot_limits(policy, manifest, group_keys):
    """Check a closed, corpus-bound initial policy with no baseline or numeric defaults.

    Each actually observed backend/topology/context/phase has exactly one role.
    Diagnostic entries cannot carry thresholds or contribute to acceptance. At
    least one required entry prevents an empty set of tests from becoming GREEN.
    """
    _finite_json(policy)
    require(type(policy) is dict and set(policy) == {'schema', 'corpus_manifest_digest', 'groups'} and
            policy.get('schema') == 'llaminar.model-kld-absolute-pilot.v1' and
            policy.get('corpus_manifest_digest') == metadata_digest(manifest), 'Unbound absolute pilot')
    require(type(policy['groups']) is list, 'Invalid absolute pilot group inventory')
    roles = {}; limits = {}
    for item in policy['groups']:
        require(type(item) is dict, 'Invalid absolute pilot group')
        key = tuple(item.get(k) for k in GROUP_FIELDS)
        require(all(type(v) is str for v in key) and key in group_keys and key not in roles,
                'Unknown/duplicate absolute pilot group')
        role = item.get('role')
        fields = set(GROUP_FIELDS) | {'role'}
        require(role in ('required', 'diagnostic'), 'Invalid absolute pilot group role')
        if role == 'required':
            fields |= {'maximum', 'minimum', 'tail_threshold_kl_nats'}
        require(set(item) == fields, 'Unexpected/missing absolute pilot group fields')
        roles[key] = role
        if role == 'diagnostic': continue
        maximum, minimum = item['maximum'], item['minimum']
        require(type(maximum) is dict and type(minimum) is dict and
                set(maximum) == set(ABSOLUTE_MAXIMUM_LIMITS) and set(minimum) == set(MINIMUM_LIMITS),
                'Incomplete absolute pilot limits')
        for name in ABSOLUTE_MAXIMUM_LIMITS: _number(maximum[name], name)
        for name in MINIMUM_LIMITS:
            require(0 <= _number(minimum[name], name) <= 1, 'Invalid absolute pilot agreement floor')
        require(0 <= maximum['tail_fraction'] <= 1 and 0 <= maximum['mean_total_variation'] <= 1,
                'Invalid absolute pilot probability limit')
        tail = _number(item['tail_threshold_kl_nats'], 'tail threshold')
        require(tail <= maximum['row_kl_nats'], 'Absolute pilot tail exceeds catastrophe ceiling')
        limits[key] = item
    require(set(roles) == set(group_keys), 'Missing absolute pilot group')
    require(bool(limits), 'Absolute pilot needs at least one required group')
    return limits, roles


def summarize(manifest, candidate, baseline=None, baseline_approval=None, *, calibration=None, absolute_pilot=None):
    """Compare exactly paired fixed histories without pooling topologies/phases/contexts.

    Returns raw rows plus prompt-balanced summaries and per-offset means. A
    calibrated single-row catastrophe fails even when its group mean is small.
    Without an actual approved baseline, the same descriptive aggregation runs
    with unavailable (null) paired fields. No self-baseline or approval is made.
    An explicit absolute pilot instead tests only its required groups, without
    a baseline, paired deltas or calibration. Diagnostic groups stay unavailable.
    Collection without either policy is unavailable for numerical acceptance;
    a tail count needs an explicit threshold, so stays unavailable.
    """
    validate_observation(manifest, candidate)
    require(absolute_pilot is None or (baseline is None and baseline_approval is None and calibration is None),
            'Absolute pilot cannot use a baseline, approval or calibration')
    paired_available = baseline is not None
    require(paired_available == (baseline_approval is not None), 'Baseline and approval must be supplied together')
    require(paired_available or calibration is None, 'Calibration requires an actual approved baseline')
    if paired_available:
        validate_observation(manifest, baseline); _finite_json(baseline_approval)
        require(candidate['topology_id'] == baseline['topology_id'] and candidate['backend'] == baseline['backend'],
                'Unmatched baseline topology/backend')
        require(type(baseline_approval) is dict and baseline_approval.get('schema') == 'llaminar.model-kld-baseline-approval.v1' and
                baseline_approval.get('corpus_manifest_digest') == metadata_digest(manifest) and
                baseline_approval.get('baseline_report_digest') == metadata_digest(baseline), 'Baseline is not explicitly approved')
    paired = []
    for index, current in enumerate(candidate['rows']):
        previous = baseline['rows'][index] if paired_available else None
        require(previous is None or current['row_id'] == previous['row_id'], 'Unmatched approved baseline row')
        row = dict(current)
        row['baseline_kl_nats'] = previous['metrics']['kl_nats'] if previous else None
        row['paired_kl_delta_nats'] = current['metrics']['kl_nats']-previous['metrics']['kl_nats'] if previous else None
        paired.append(row)
    grouped = defaultdict(list)
    for row in paired:
        key = (candidate['backend'], candidate['topology_id'], row['context_group'], row['phase'])
        grouped[key].append(row)
    limits = _limits(calibration, manifest, baseline, baseline_approval, grouped)
    roles = {}
    if absolute_pilot is not None:
        limits, roles = _absolute_pilot_limits(absolute_pilot, manifest, grouped)
    summaries = []; failures = []; replay = defaultdict(list)
    for key, rows in grouped.items():
        metric = lambda name: _prompt_mean(rows, lambda r: float(r['metrics'][name]))
        values = [r['metrics']['kl_nats'] for r in rows]
        steps = defaultdict(list)
        for row in rows: steps[row['decode_offset']].append(row)
        step_means = [dict(decode_offset=offset, prompt_count=len({r['history_id'] for r in subset}),
                           mean_kl_nats=_prompt_mean(subset, lambda r: r['metrics']['kl_nats']),
                           top1_agreement=_prompt_mean(subset, lambda r: float(r['metrics']['top1_same'])),
                           mean_top5_overlap=_prompt_mean(subset, lambda r: r['metrics']['top5_overlap']))
                      for offset, subset in steps.items()]
        summary = dict(zip(GROUP_FIELDS, key), row_count=len(rows), prompt_count=len({r['history_id'] for r in rows}),
            mean_kl_nats=metric('kl_nats'), mean_total_variation=metric('total_variation'),
            top1_agreement=metric('top1_same'), mean_top5_overlap=metric('top5_overlap'),
            paired_mean_kl_delta_nats=_prompt_mean(rows, lambda r: r['paired_kl_delta_nats']) if paired_available else None,
            row_kl_nats=max(values), row_kl_delta_nats=max(r['paired_kl_delta_nats'] for r in rows) if paired_available else None,
            median_row_kl_nats=_quantile(values,.5), p95_row_kl_nats=_quantile(values,.95),
            per_step=step_means, calibrated=bool(limits) and absolute_pilot is None, tail_count=None, tail_fraction=None,
            mean_tail_excess_kl_nats=None, passed=None)
        summary['step_mean_kl_nats'] = max(s['mean_kl_nats'] for s in step_means)
        if key in limits:
            policy = limits[key]; threshold = policy['tail_threshold_kl_nats']
            summary['tail_count'] = sum(v > threshold for v in values)
            summary['tail_fraction'] = _prompt_mean(rows, lambda r: float(r['metrics']['kl_nats'] > threshold))
            summary['mean_tail_excess_kl_nats'] = _prompt_mean(rows, lambda r: max(0., r['metrics']['kl_nats']-threshold))
            bad = [name for name, bound in policy['maximum'].items() if summary[name] > bound]
            bad += [name for name, bound in policy['minimum'].items() if summary[name] < bound]
            summary['passed'] = not bad
            for name in bad: failures.append(dict(zip(GROUP_FIELDS, key), criterion=name))
            for row in rows:
                if row['metrics']['kl_nats'] > threshold or bad:
                    replay[row['history_id']].append(row['row_id'])
        if absolute_pilot is not None:
            summary['acceptance_role'] = roles[key]
            if roles[key] == 'diagnostic':
                for row in rows: replay[row['history_id']].append(row['row_id'])
        summaries.append(summary)
    for history in manifest['histories']:
        if history['id'] not in replay and not limits:
            # Preserve an honest diagnostic link for each uncalibrated history.
            replay[history['id']] = [r['row_id'] for r in paired if r['history_id'] == history['id']]
    replay_requests = []
    for history in manifest['histories']:
        if history['id'] in replay:
            replay_requests.append(dict(history_id=history['id'], canonical_cell=_topology(manifest, candidate['topology_id'])['canonical_cell'],
                topology_id=candidate['topology_id'], backend=candidate['backend'],
                expected_config=_topology(manifest, candidate['topology_id'])['expected_config'],
                reference_directory=history['reference_directory'], reference_scope=history['reference_scope'],
                prompt=history['prompt'],
                source_digest=metadata_digest(manifest['source']), corpus_manifest_digest=metadata_digest(manifest),
                prompt_ids=history['prompt_ids'], continuation_ids=history['continuation_ids'],
                selected_row_ids=replay[history['id']], require_all_layers=True,
                requires_full_stage_reference=history['reference_scope'] != 'full_stages',
                status='canonical_history_projection_required_no_alternate_runner'))
    result = dict(schema='llaminar.model-kld-summary.v2', complete=True, scope='selected_topology_only',
                topology_id=candidate['topology_id'], backend=candidate['backend'],
                source_digest=candidate['source_digest'], reference_report_digest=candidate['reference_report_digest'],
                runtime_config_digest=candidate['runtime_config_digest'], arithmetic=candidate['arithmetic'],
                native_identity=candidate['native_identity'], proofs=candidate['proofs'],
                calibration_digest=metadata_digest(calibration) if calibration is not None else None,
                baseline_approval_digest=metadata_digest(baseline_approval) if paired_available else None, calibrated=bool(limits) and absolute_pilot is None,
                passed=not failures if limits else None, verdict=('RED' if failures else 'GREEN') if limits else 'UNCALIBRATED',
                metric=METRIC, corpus_manifest_digest=metadata_digest(manifest),
                approved_baseline_digest=metadata_digest(baseline) if paired_available else None, paired_baseline_available=paired_available, groups=summaries, rows=paired,
                failures=failures, diagnostic_replays=replay_requests,
                weighting='equal_prompt_means_within_each_backend_topology_context_phase',
                acceptance_phase_or_backend_pooling=False, numerical_verdict_only=True)
    if absolute_pilot is not None:
        result.update(scope='selected_topology_required_groups_only',
            absolute_pilot_digest=metadata_digest(absolute_pilot),
            acceptance_scope=dict(policy='initial_absolute_pilot',
                required_groups=[dict(zip(GROUP_FIELDS, key)) for key in grouped if roles[key] == 'required'],
                diagnostic_groups=[dict(zip(GROUP_FIELDS, key)) for key in grouped if roles[key] == 'diagnostic'],
                diagnostic_verdict='UNAVAILABLE', whole_corpus_accepted=False),
            certification_eligible=False)
    return result


def _workload_digest(manifest):
    """Bind reference rows to exact histories without a circular manifest/report digest."""
    return metadata_digest(dict(histories=manifest['histories'], vocabulary=manifest['vocabulary']))


def _case_frames(manifest, document):
    """Require every original history and every selected row, before accessing payloads."""
    cases = document.get('cases'); wanted = manifest['histories']
    require(type(cases) is list and len(cases) == len(wanted), 'Incomplete history frames')
    require(len({c.get('history_id') for c in cases if type(c) is dict}) == len(wanted), 'Duplicate history frame')
    inventory = expected_rows(manifest)
    for case, history in zip(cases, wanted):
        require(case.get('history_id') == history['id'], 'Changed history frame order')
        expected = [r for r in inventory if r['history_id'] == history['id']]
        rows = case.get('rows')
        require(type(rows) is list and len(rows) == len(expected), 'Incomplete physical row frame')
        for row, original in zip(rows, expected):
            require(type(row) is dict and all(k in row and row[k] == v and type(row[k]) is type(v)
                    for k, v in original.items()) and type(row.get('vocabulary')) is int and
                    row['vocabulary'] == manifest['vocabulary'], 'Wrong physical teacher row/extent')
    return cases


def load_corpus(root, *, catalog):
    """Load an explicitly approved v2 generation through canonical path/stat authority.

    Only metadata and file extents are read here. Full vocabulary payloads are
    consumed once per actual score. v1 catalog/default handling is unchanged.
    """
    import json
    root = Path(root).resolve(strict=True); catalog = Path(catalog).resolve(strict=True)
    approval = json.loads(catalog.read_text())
    require(type(approval) is dict and set(approval) == {'schema', 'manifest_digest'} and approval['schema'] == 2,
            'Explicit v2 corpus approval required')
    manifest_path = root/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    require(metadata_digest(manifest) == approval['manifest_digest'], 'Unapproved corpus generation')
    validate_manifest(manifest)
    reference = manifest.get('reference', {})
    require(reference.get('format') == 'float32_le_row_major', 'Reference must preserve FP32 full rows')
    report_path = relative_file(root, reference.get('report'))
    report = json.loads(report_path.read_text())
    require(metadata_digest(report) == reference.get('report_digest') and
            report.get('schema') == 'llaminar.model-kld-reference.v2' and report.get('complete') is True and
            report.get('source_digest') == metadata_digest(manifest['source']) and
            report.get('workload_digest') == _workload_digest(manifest), 'Wrong original reference metadata')
    cases = _case_frames(manifest, report)
    inputs = {str(p): identity(p) for p in (catalog, manifest_path, report_path)}
    payloads = {}
    for case in cases:
        payload = relative_file(report_path.parent, case.get('logits_file'))
        require(payload.stat().st_size == len(case['rows'])*manifest['vocabulary']*4,
                'Truncated/extra full-vocabulary reference')
        inputs[str(payload)] = identity(payload); payloads[case['history_id']] = payload
    return dict(manifest=manifest, reference=report, reference_payloads=payloads, inputs=inputs)


def score_observation(corpus, report_path):
    """Score one actual native observation with shared full-vocabulary arithmetic.

    The native owner must authenticate source/config/publication and retirement.
    This consumer does not invent those proofs. It never publishes GREEN:
    approval/calibration and summarize() are a separate explicit review step.
    """
    import json
    manifest = corpus['manifest']; validate_manifest(manifest)
    report_path = Path(report_path).resolve(strict=True)
    document = json.loads(report_path.read_text())
    require(document.get('schema') == 'llaminar.model-kld-observation.v2' and document.get('complete') is True,
            'Native observation incomplete')
    topology = _topology(manifest, document.get('topology_id'))
    require(document.get('corpus_manifest_digest') == metadata_digest(manifest) and
            document.get('source_digest') == metadata_digest(manifest['source']) and
            document.get('reference_report_digest') == manifest['reference']['report_digest'] and
            document.get('backend') == topology['backend'] and
            document.get('runtime_config_digest') == topology['runtime_config_digest'] and
            document.get('arithmetic') == topology['arithmetic'], 'Native original identity changed')
    require(all(document.get('proofs', {}).get(k) is True for k in PROOFS), 'Missing native ownership proof')
    native_identity = document.get('native_identity')
    require(type(native_identity) is dict and native_identity and
            all(type(k) is str and type(v) is str and v for k, v in native_identity.items()),
            'Missing native source/binary identity')
    cases = _case_frames(manifest, document)
    reset_history = dict(next(h for h in manifest['histories'] if h['id'] == manifest['reset_control']['history_id']))
    reset_history['id'] = manifest['reset_control']['replay_id']
    reset_case = document.get('reset_case')
    _case_frames(dict(manifest, histories=[reset_history]), dict(cases=[reset_case]))
    pins = {str(report_path): identity(report_path)}; outputs = []
    for case in cases:
        payload = Path(case.get('logits_file', '')).resolve(strict=True)
        require(payload.is_relative_to(report_path.parent), 'Native payload escapes artifact owner')
        require(payload.is_file() and payload.stat().st_size == len(case['rows'])*manifest['vocabulary']*4,
                'Truncated/extra full-vocabulary native rows')
        pins[str(payload)] = identity(payload); outputs.append(payload)
    reset_payload = Path(reset_case.get('logits_file', '')).resolve(strict=True)
    require(reset_payload.is_relative_to(report_path.parent) and reset_payload.is_file() and
            reset_payload.stat().st_size == len(reset_case['rows'])*manifest['vocabulary']*4,
            'Invalid full-row reset artifact')
    pins[str(reset_payload)] = identity(reset_payload)
    owned = {(identity(p)[1], identity(p)[2]) for p in outputs+[reset_payload]}
    require(len(owned) == len(outputs)+1, 'Aliased history/reset payload owner')
    original_payloads = {(identity(p)[1], identity(p)[2]) for p in corpus['reference_payloads'].values()}
    require(not (owned & original_payloads), 'Native output aliases original reference payload')
    unchanged(corpus['inputs']); unchanged(pins)
    import numpy as np
    initial_index = next(i for i,h in enumerate(manifest['histories']) if h['id'] == manifest['reset_control']['history_id'])
    reset_shape = (len(reset_case['rows']), manifest['vocabulary'])
    initial = np.memmap(outputs[initial_index], dtype='<u4', mode='r', shape=reset_shape)
    repeated = np.memmap(reset_payload, dtype='<u4', mode='r', shape=reset_shape)
    require(np.array_equal(initial, repeated), 'Terminal reset changed full-logit bytes')
    del initial, repeated
    rows = []
    for case, path in zip(cases, outputs):
        shape = (len(case['rows']), manifest['vocabulary'])
        reference = np.memmap(corpus['reference_payloads'][case['history_id']], dtype='<f4', mode='r', shape=shape)
        candidate = np.memmap(path, dtype='<f4', mode='r', shape=shape)
        try:
            for original, actual, expected in zip(case['rows'], candidate, reference):
                metrics = full_vocabulary_metrics(expected, actual, vocabulary=manifest['vocabulary'])
                rows.append(dict(original, metrics=metrics))
        finally:
            # Each exact row is consumed by the shared helper; no retained mmap
            # view is exported as a new owner or a second cache.
            del candidate, reference
    unchanged(corpus['inputs']); unchanged(pins)
    result = {k: document[k] for k in ('corpus_manifest_digest', 'source_digest', 'reference_report_digest',
              'backend', 'topology_id', 'runtime_config_digest', 'arithmetic', 'native_identity', 'proofs')}
    result.update(schema='llaminar.model-kld-metrics.v2', complete=True, metric=METRIC,
                  rows=rows, acceptance='not_evaluated_requires_approved_baseline_and_calibration')
    return validate_observation(manifest, result)
