#!/usr/bin/env python3
"""Assert bounded serving resources without inventing an allocation authority.

PhysicalMemoryAuthority attestations supply any remaining materialization
allowance. OS baselines account for runtime/library overhead, while cache gauges
are checked against each configured instance. Limits never ratchet with observed
growth, a stats reset, or an observer error. Driver mappings are independent
observations, never summed as unique host RAM.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


def natural(value, name: str) -> int:
    """Require exact byte/count integers rather than booleans or rounded floats."""
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


@dataclass(frozen=True)
class GrowthLimit:
    """One immutable observation limit with an explicit allowance provenance."""

    baseline_bytes: int
    remaining_admitted_bytes: int
    resolution_bytes: int
    evidence: str

    def __post_init__(self) -> None:
        """Reject unknown units and allowances that lack an accountable origin."""
        natural(self.baseline_bytes, 'baseline_bytes')
        natural(self.remaining_admitted_bytes, 'remaining_admitted_bytes')
        if natural(self.resolution_bytes, 'resolution_bytes') == 0 or not self.evidence:
            raise ValueError('Resource limit needs positive resolution and named evidence')

    @property
    def maximum_bytes(self) -> int:
        """Permit one measurement quantum, without a percentage or arbitrary reserve."""
        return self.baseline_bytes + self.remaining_admitted_bytes + self.resolution_bytes - 1


def remaining_materialization(attestations: list[dict]) -> dict[str, int]:
    """Read unused admitted capacity from the authority's own complete owner rows.

    Callers authenticate rank membership and completeness with the canonical
    ranked PerfStats validator first. This function checks all byte relationships
    again before deriving diagnostic limits; it never reserves or allocates.
    """
    result = {}
    owners = set()
    if not attestations:
        raise ValueError('Missing PhysicalMemoryAuthority attestation')
    for row in attestations:
        key = (row['allocator'], row['owner'])
        if key in owners or not all(isinstance(value, str) and value for value in key):
            raise ValueError('Duplicate or missing physical memory owner')
        owners.add(key)
        values = {name: natural(row[name], name) for name in (
            'planned_new_bytes', 'materialized_new_bytes', 'committed_new_bytes',
            'planned_resident_bytes', 'adopted_resident_bytes')}
        if not (values['materialized_new_bytes'] <= values['committed_new_bytes'] <= values['planned_new_bytes']
                and values['adopted_resident_bytes'] <= values['planned_resident_bytes']):
            raise ValueError('Physical memory owner exceeded its admitted envelope')
        result.setdefault(key[0], 0)
        result[key[0]] += (values['planned_new_bytes'] - values['materialized_new_bytes']
                           + values['planned_resident_bytes'] - values['adopted_resident_bytes'])
    return result


def cache_limits(stats: dict, expected: Mapping[str, int]) -> dict[str, int]:
    """Assert per-instance and aggregate occupancy, including disabled/shared tiers."""
    usage = {}
    for name in ('ram', 'disk'):
        tier = stats['prefix_cache']['storage']['tiers'][name]
        instances = tier['instances']
        if not isinstance(instances, list):
            raise ValueError('Cache instances must be a list')
        seen = set()
        used = capacity = 0
        for row in instances:
            participants = [row.get('participant')] if name == 'ram' else row['participants']
            if (not isinstance(participants, list) or not participants
                    or any(not isinstance(value, str) or not value for value in participants)
                    or len(set(participants)) != len(participants)):
                raise ValueError('Invalid cache participant identity')
            identity = tuple(sorted(participants))
            if not identity or identity in seen:
                raise ValueError('Missing or duplicated cache instance')
            seen.add(identity)
            if type(row['enabled']) is not bool or type(row['initialized']) is not bool:
                raise ValueError('Invalid cache lifecycle')
            bound = natural(row['capacity_bytes'], 'cache capacity')
            if row['enabled']:
                if not row['initialized'] or bound == 0:
                    raise ValueError('Serving cache is uninitialized')
            elif row['initialized'] or bound:
                raise ValueError('Disabled cache owns capacity')
            occupied = natural(row['used_bytes'], 'cache occupancy')
            if occupied > bound:
                raise ValueError(f'{name} cache instance exceeded capacity: {identity}')
            used += occupied
            capacity += bound
        if (capacity != natural(expected[name], 'expected capacity') or
                natural(tier['capacity_bytes'], 'aggregate cache capacity') != capacity or
                natural(tier['used_bytes'], 'aggregate cache occupancy') != used or
                type(tier['enabled']) is not bool or tier['enabled'] != any(r['enabled'] for r in instances)):
            raise ValueError(f'{name} cache aggregate/configuration mismatch')
        usage[name] = used
    return usage


def cache_identity(stats: dict) -> tuple:
    """Seal participant ownership and budgets independently of changing occupancy.

    Call cache_limits first to authenticate types and uniqueness. Reordering an
    instance list is harmless; replacing a participant or moving capacity from
    one participant to another is a different serving configuration.
    """
    return tuple((name, tuple(sorted(
        (tuple(sorted([row['participant']] if name == 'ram' else row['participants'])),
         row['enabled'], row['initialized'], row['capacity_bytes'])
        for row in stats['prefix_cache']['storage']['tiers'][name]['instances'])))
        for name in ('ram', 'disk'))


class ResourceGrowthAssertions:
    """Retain first failure while observing subsequent progress against fixed limits."""

    def __init__(self, *, identity: dict, limits: Mapping[str, GrowthLimit],
                 cache_capacity: Mapping[str, int], maximum_sample_age_seconds: float):
        """Seal a ready-for-serving baseline; callers may not replace it on reset."""
        if not identity or not limits or not math.isfinite(maximum_sample_age_seconds) or maximum_sample_age_seconds <= 0:
            raise ValueError('Resource gate requires identity, limits, and finite freshness policy')
        self.identity = dict(identity)
        self.limits = dict(limits)
        self.cache_capacity = dict(cache_capacity)
        self.maximum_age = maximum_sample_age_seconds
        self.failures: list[str] = []
        self.samples = 0
        self.completed_progress = False
        self.previous = None
        self.peaks = {}
        self.cache_identity = None

    def observe(self, *, observed: float, identity: dict, metrics: Mapping[str, int],
                stats: dict, errors: list[str]) -> None:
        """Record boundedness/freshness failures without cancelling an inference request."""
        try:
            if errors:
                raise ValueError('Observation failed: ' + repr(errors))
            if identity != self.identity:
                raise ValueError('Serving process/boot/admission identity changed')
            if not math.isfinite(observed):
                raise ValueError('Invalid observation timestamp')
            age = observed - stats['snapshot_unix_seconds']
            if not math.isfinite(age) or not 0 <= age <= self.maximum_age:
                raise ValueError('Stale or future server statistics')
            if self.previous and observed <= self.previous[0]:
                raise ValueError('Observation time did not advance')
            cache_limits(stats, self.cache_capacity)
            current_cache_identity = cache_identity(stats)
            if self.cache_identity is not None and current_cache_identity != self.cache_identity:
                raise ValueError('Cache ownership or per-instance budget changed')
            self.cache_identity = current_cache_identity
            for name, limit in self.limits.items():
                value = natural(metrics[name], name)
                self.peaks[name] = max(self.peaks.get(name, 0), value)
                if value > limit.maximum_bytes:
                    raise ValueError(f'{name} exceeded fixed resource envelope: {value} > {limit.maximum_bytes}; {limit.evidence}')
            sequence = (natural(stats['epoch'], 'epoch'), natural(stats['requests']['completed'], 'completed'))
            if self.previous:
                old = self.previous[1]
                if sequence[0] < old[0] or (sequence[0] == old[0] and sequence[1] < old[1]):
                    raise ValueError('Request counters regressed without a stats reset')
                self.completed_progress |= sequence[0] == old[0] and sequence[1] > old[1]
            self.previous = (observed, sequence)
            self.samples += 1
        except (ValueError, KeyError, TypeError) as error:
            self.failures.append(str(error))

    def result(self) -> dict:
        """Require successful observations spanning actual completed inference work."""
        return {'passed': not self.failures and self.samples >= 2 and self.completed_progress,
                'samples': self.samples, 'completed_progress': self.completed_progress,
                'failures': list(self.failures), 'peak_bytes': dict(self.peaks),
                'maximum_bytes': {name: limit.maximum_bytes for name, limit in self.limits.items()}}
