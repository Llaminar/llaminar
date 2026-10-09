"""Authenticate prefix-reuse measurements for an exclusive sequential HTTP workload.

Only small /stats metadata is observed. Each before/after pair must contain
exactly one completed generation and agree with its wire usage. Background
archive traffic is a window measurement, not a claim of per-request causality.
No cache payloads, model retries, counter resets or inference deadlines occur.
"""
from __future__ import annotations

import math
import statistics


def integer(value, name: str) -> int:
    """Require an exact nonnegative counter; booleans and floats are not counters."""
    if type(value) is not int or value < 0:
        raise ValueError(f"Invalid prefix measurement counter: {name}")
    return value


def idle_snapshot(value: dict) -> None:
    """Reject an occupied, partially accounted or unsupported observation boundary."""
    if not isinstance(value, dict) or value.get('object') != 'llaminar.stats' or value.get('schema_version') != 1:
        raise ValueError('Prefix measurement requires supported Llaminar statistics')
    integer(value['epoch'], 'epoch')
    requests = value['requests']
    if integer(requests['active'], 'active') != 0:
        raise ValueError('Prefix measurement requires exclusive idle request boundaries')
    retired = sum(integer(requests[key], key) for key in ('completed', 'failed', 'disconnected'))
    if integer(requests['started'], 'started') != retired:
        raise ValueError('Prefix measurement found incomplete request accounting')


def measure_request(before: dict, after: dict, usage: dict) -> dict:
    """Join one successful wire exchange to its exact completed server observation."""
    idle_snapshot(before)
    idle_snapshot(after)
    if before['epoch'] != after['epoch']:
        raise ValueError('Statistics reset during prefix measurement')
    for key, delta in (('started', 1), ('completed', 1), ('failed', 0), ('disconnected', 0)):
        if after['requests'][key] - before['requests'][key] != delta:
            raise ValueError('Prefix measurement contains missing, failed or competing requests')
    last = after['last_request']
    if last['outcome'] != 'completed' or last['handler_status'] != 200:
        raise ValueError('Prefix measurement lacks a successful terminal request')
    sequence = integer(last['sequence'], 'sequence')
    prior = before['last_request']
    if sequence == 0 or (prior is not None and sequence != integer(prior['sequence'], 'prior sequence') + 1):
        raise ValueError('Prefix measurement request sequence is discontinuous')
    for key, counter in (('prompt_tokens', 'prompt'), ('completion_tokens', 'completion')):
        expected = integer(usage[key], key)
        delta = integer(after['tokens'][counter], counter) - integer(before['tokens'][counter], counter)
        if integer(last[key], key) != expected or delta != expected:
            raise ValueError('Prefix measurement does not match wire token usage')
    prompt = usage['prompt_tokens']
    prefix = last['prefix_cache']
    matched = integer(prefix['matched_tokens'], 'matched_tokens')
    if prompt <= 0 or matched > prompt or integer(prefix['requested_tokens'], 'requested_tokens') != prompt:
        raise ValueError('Prefix measurement has invalid reuse geometry')
    if integer(last['throughput']['prefill_uncached_tokens'], 'prefill_uncached_tokens') != prompt - matched:
        raise ValueError('Prefix measurement prefill work disagrees with restored tokens')
    for name in ('ttft_seconds', 'prefill_seconds'):
        seconds = last['timings'][name]
        if type(seconds) not in (float, int) or not math.isfinite(seconds) or seconds < 0:
            raise ValueError('Prefix measurement lacks a finite completed timing')
    old_storage = before['prefix_cache']['storage']
    new_storage = after['prefix_cache']['storage']
    traffic = {}
    if old_storage['churn'].keys() != new_storage['churn'].keys():
        raise ValueError('Prefix measurement storage event inventory changed')
    for event, counters in new_storage['churn'].items():
        if event == 'scope':
            continue
        traffic[event] = {}
        for name in ('operations', 'bytes'):
            delta = integer(counters[name], name) - integer(old_storage['churn'][event][name], name)
            if delta < 0:
                raise ValueError('Prefix measurement storage counter regressed')
            traffic[event][name] = delta
    tiers = {}
    for name in ('ram', 'disk'):
        old, new = old_storage['tiers'][name], new_storage['tiers'][name]
        if old['enabled'] != new['enabled'] or old['capacity_bytes'] != new['capacity_bytes']:
            raise ValueError('Prefix measurement storage admission changed')
        capacity = integer(new['capacity_bytes'], 'capacity_bytes')
        for observed in (old, new):
            if type(observed['enabled']) is not bool:
                raise ValueError('Prefix measurement tier enablement is not boolean')
            used = integer(observed['used_bytes'], 'used_bytes')
            if used > capacity or (not observed['enabled'] and used != 0):
                raise ValueError('Prefix measurement storage exceeds its admitted bound')
        tiers[name] = {'enabled': new['enabled'], 'capacity_bytes': capacity,
                      'before_bytes': old['used_bytes'], 'after_bytes': new['used_bytes']}
    return {'schema': 1, 'epoch': after['epoch'], 'sequence': sequence,
            'request': last, 'tiers': tiers, 'traffic': traffic,
            'traffic_before': old_storage['churn'], 'traffic_after': new_storage['churn'],
            'traffic_scope': 'completed_operations_between_request_boundary_probes'}


def summarize(measurements: list[dict]) -> dict:
    """Compare ordinary turns, compaction and immediate resumption without hiding gaps."""
    if not measurements:
        raise ValueError('Prefix measurement has no completed requests')
    prior = None
    for row in measurements:
        if row['kind'] not in ('ordinary', 'compaction', 'post_compaction'):
            raise ValueError('Unknown prefix measurement request kind')
        if prior is not None and (row['epoch'] != prior['epoch'] or row['sequence'] != prior['sequence'] + 1):
            raise ValueError('Prefix measurement session has missing request observations')
        if prior is not None:
            for name in ('ram', 'disk'):
                for field in ('enabled', 'capacity_bytes'):
                    if row['tiers'][name][field] != prior['tiers'][name][field]:
                        raise ValueError('Prefix measurement session changed tier admission')
            if row['traffic_before'].keys() != prior['traffic_after'].keys():
                raise ValueError('Prefix measurement session changed event inventory')
            for event in row['traffic']:
                for field in ('operations', 'bytes'):
                    if row['traffic_before'][event][field] < prior['traffic_after'][event][field]:
                        raise ValueError('Prefix measurement background counters regressed')
        prior = row
    groups = {}
    for kind in ('all', 'ordinary', 'compaction', 'post_compaction'):
        rows = measurements if kind == 'all' else [row for row in measurements if row['kind'] == kind]
        prompt = sum(row['request']['prompt_tokens'] for row in rows)
        matched = sum(row['request']['prefix_cache']['matched_tokens'] for row in rows)
        ttft = sorted(row['request']['timings']['ttft_seconds'] for row in rows)
        groups[kind] = {'requests': len(rows), 'prompt_tokens': prompt, 'restored_tokens': matched,
            'token_reuse_rate': matched / prompt if prompt else None,
            'ttft_mean_seconds': statistics.mean(ttft) if ttft else None,
            'ttft_median_seconds': statistics.median(ttft) if ttft else None,
            'ttft_p95_seconds': ttft[math.ceil(.95 * len(ttft)) - 1] if ttft else None,
            'prefill_seconds': sum(row['request']['timings']['prefill_seconds'] for row in rows)}
    return {'schema': 1, 'complete': True, 'requests': len(measurements), 'groups': groups,
            'tiers': {name: {'enabled': measurements[0]['tiers'][name]['enabled'],
                'capacity_bytes': measurements[0]['tiers'][name]['capacity_bytes'],
                'boundary_peak_bytes': max(row['tiers'][name][field]
                    for row in measurements for field in ('before_bytes', 'after_bytes'))}
                for name in ('ram', 'disk')},
            'traffic': {event: {field: measurements[-1]['traffic_after'][event][field] -
                measurements[0]['traffic_before'][event][field]
                for field in ('operations', 'bytes')} for event in measurements[0]['traffic']},
            'scope': 'exclusive_sequential_workload_request_boundary_observations'}
