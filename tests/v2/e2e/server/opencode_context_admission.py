#!/usr/bin/env python3
"""Select the largest model-bounded context through production PMA dry runs.

The callback owns exact-image launch, driver observation and real retirement.
This module only orders candidate contexts; it never estimates capacity, changes
topology, reduces an output budget or retries an unexpected failure. Every probe
must return an explicit admission outcome and preserve its own native evidence.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class AdmissionOutcome(str, Enum):
    """Only a proven capacity rejection permits a smaller context probe."""

    ACCEPTED = 'accepted'
    CAPACITY_REJECTED = 'capacity_rejected'


@dataclass(frozen=True)
class ContextPolicy:
    """Model metadata and public workload requirements, fixed before probing."""

    model_maximum: int
    alignment: int
    output_tokens: int = 32768

    def __post_init__(self) -> None:
        """Reject absent/rounded geometry and a context with no prompt space."""
        if any(type(value) is not int or not 0 < value <= 2**31 - 1
               for value in (self.model_maximum, self.alignment, self.output_tokens)):
            raise ValueError('Context admission requires positive exact model geometry')
        if self.model_maximum <= self.output_tokens:
            raise ValueError('Model context cannot contain the requested output budget and a prompt')


def select_context(policy: ContextPolicy, probe: Callable[[int], AdmissionOutcome]) -> dict:
    """Prove the maximum itself or adjacent admitted/rejected context buckets.

    All candidates retain the same model, topology, MTP, precision and cache
    budgets. Context is the sole varying demand on PhysicalMemoryAuthority.
    A raised exception ends the search; successful smaller candidates cannot
    hide a crash, incomplete native retirement or an unknown planning error.
    """
    attempts = []

    def observe(context: int) -> bool:
        outcome = probe(context)
        if not isinstance(outcome, AdmissionOutcome):
            raise ValueError('PMA probe omitted its typed admission outcome')
        attempts.append({'context_tokens': context, 'outcome': outcome.value})
        return outcome == AdmissionOutcome.ACCEPTED

    if observe(policy.model_maximum):
        return {'schema': 1, 'model_maximum': policy.model_maximum,
                'context_tokens': policy.model_maximum, 'alignment_tokens': policy.alignment,
                'output_tokens': policy.output_tokens, 'model_maximum_admitted': True,
                'first_rejected_context': None, 'attempts': attempts}
    low = (policy.output_tokens + 1 + policy.alignment - 1) // policy.alignment
    high = (policy.model_maximum + policy.alignment - 1) // policy.alignment
    if low * policy.alignment >= policy.model_maximum or not observe(low * policy.alignment):
        raise ValueError('No admitted context can retain the requested output budget')
    while high - low > 1:
        middle = (low + high) // 2
        if observe(middle * policy.alignment):
            low = middle
        else:
            high = middle
    selected = low * policy.alignment
    rejected = min(row['context_tokens'] for row in attempts
                   if row['outcome'] == AdmissionOutcome.CAPACITY_REJECTED.value)
    if not selected < rejected <= selected + policy.alignment:
        raise ValueError('Context search did not prove adjacent admission bounds')
    return {'schema': 1, 'model_maximum': policy.model_maximum, 'context_tokens': selected,
            'alignment_tokens': policy.alignment, 'output_tokens': policy.output_tokens,
            'model_maximum_admitted': False, 'first_rejected_context': rejected, 'attempts': attempts}
