"""Backend-neutral exact/generic NativeVNNI policy IR and deterministic digest."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

from .corpus import GenericDomain, RuntimeKey
from .exact_oracle import ExactWinner
from .schema import FEATURE_SCHEMA_VERSION, LEARNER_VERSION, POLICY_ABI
from .segmented_policy import (
    DomainCrossValidation,
    GenericDispatchRule,
    GenericPolicy,
)


class DispatchMatchKind(str, Enum):
    """Resolution tier selected by the generated runtime policy."""

    EXACT = "Exact"
    GENERIC = "Generic"
    CERTIFIED_FLOOR = "CertifiedFloor"


_PARALLEL_POLICY_VALUES: tuple[Any, ...] = ()


def _physical_core_count() -> int:
    """Return the affinity-visible physical-core count for offline encoding.

    Policy publication is an offline operation, but using every SMT sibling for
    Python object traversal only increases process state and scheduler pressure.
    Package/core identities keep worker defaults aligned with the corpus fitter
    and the rest of the NativeVNNI publication pipeline.
    """

    try:
        visible_cpus = tuple(sorted(os.sched_getaffinity(0)))
    except AttributeError:
        visible_cpus = tuple(range(os.cpu_count() or 1))
    physical_cores = set()
    for cpu in visible_cpus:
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = (topology / "physical_package_id").read_text().strip()
            core = (topology / "core_id").read_text().strip()
        except OSError:
            return max(1, len(visible_cpus))
        physical_cores.add((package, core))
    return max(1, len(physical_cores))


def _normalize_policy_value(value: Any) -> Any:
    """Convert one immutable policy value to its historical JSON structure.

    ``dataclasses.asdict`` recursively deep-copies every descendant before the
    old normalizer traverses that copy a second time. A production policy can
    contain hundreds of thousands of exact entries and CV cells, making that
    convenience operation a multi-minute serial publication stage. Reading
    frozen dataclass fields directly performs one traversal while retaining the
    same field names, enum values, sorted mapping keys, and list representation.
    """

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_policy_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize_policy_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _normalize_policy_value(getattr(value, item.name))
            for item in sorted(fields(value), key=lambda item: item.name)
        }
    return value


def _equal_ranges(
    item_count: int, worker_count: int
) -> tuple[tuple[int, int], ...]:
    """Split an ordered immutable inventory into balanced contiguous ranges."""

    base, remainder = divmod(item_count, worker_count)
    begin = 0
    result = []
    for worker_index in range(worker_count):
        size = base + (1 if worker_index < remainder else 0)
        result.append((begin, begin + size))
        begin += size
    return tuple(result)


def _normalize_policy_range(bounds: tuple[int, int]) -> list[Any]:
    """Normalize one inherited range without pickling source dataclasses."""

    begin, end = bounds
    return [
        _normalize_policy_value(_PARALLEL_POLICY_VALUES[index])
        for index in range(begin, end)
    ]


def _normalize_policy_inventory(
    values: tuple[Any, ...],
    *,
    items_per_worker: int,
) -> list[Any]:
    """Normalize a large ordered policy inventory on physical CPU cores.

    Workers inherit the immutable source tuple through ``fork`` and return only
    normalized JSON-compatible ranges. The parent concatenates ranges in source
    order, so parallel execution cannot perturb artifact bytes or policy hashes.
    Small inventories remain inline to avoid process startup overhead.
    """

    if not values:
        return []
    requested_workers = int(os.environ.get(
        "LLAMINAR_NATIVE_VNNI_POLICY_SERIALIZATION_WORKERS",
        str(_physical_core_count()),
    ))
    if requested_workers < 1:
        raise ValueError("policy serialization worker count must be positive")
    useful_workers = max(1, len(values) // items_per_worker)
    worker_count = min(requested_workers, _physical_core_count(), useful_workers)
    if worker_count <= 1:
        return [_normalize_policy_value(item) for item in values]

    global _PARALLEL_POLICY_VALUES
    _PARALLEL_POLICY_VALUES = values
    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            partitions = tuple(executor.map(
                _normalize_policy_range,
                _equal_ranges(len(values), worker_count),
            ))
    finally:
        _PARALLEL_POLICY_VALUES = ()
    return [item for partition in partitions for item in partition]


@dataclass(frozen=True)
class ExactDispatchEntry:
    """One full-shape overlay emitted above generic aspect/work rules."""

    key: RuntimeKey
    candidate_id: str
    arithmetic_fingerprint: str
    config_json: dict[str, Any]


@dataclass(frozen=True)
class PolicyIR:
    """Complete backend-neutral decisions ready for a backend emitter."""

    policy_abi: int
    learner_version: str
    feature_schema_version: str
    exact_entries: tuple[ExactDispatchEntry, ...]
    generic_rules: tuple[GenericDispatchRule, ...]
    unpromoted_domains: tuple[GenericDomain, ...]
    cross_validation: tuple[DomainCrossValidation, ...]
    metadata: Mapping[str, Any]

    def canonical_mapping(self, *, generic_only: bool = False) -> dict[str, Any]:
        """Return the cached stable JSON representation used by publication.

        ``PolicyIR`` and all of its descendants are immutable by contract. The
        cached mapping must therefore be treated as read-only by callers; this
        avoids normalizing the same very large exact/CV inventories once for a
        diagnostic, again for its digest, and again for the installed artifact.
        """

        return (
            self._canonical_generic_mapping
            if generic_only
            else self._canonical_complete_mapping
        )

    @cached_property
    def _canonical_generic_mapping(self) -> dict[str, Any]:
        """Normalize the generic policy section once per immutable IR."""

        return {
            "policy_abi": self.policy_abi,
            "learner_version": self.learner_version,
            "feature_schema_version": self.feature_schema_version,
            "generic_rules": _normalize_policy_inventory(
                self.generic_rules, items_per_worker=64
            ),
            "unpromoted_domains": _normalize_policy_value(
                self.unpromoted_domains
            ),
            "cross_validation": _normalize_policy_inventory(
                self.cross_validation, items_per_worker=2
            ),
        }

    @cached_property
    def _canonical_complete_mapping(self) -> dict[str, Any]:
        """Extend the cached generic section with overlays and provenance."""

        result = dict(self._canonical_generic_mapping)
        result["exact_entries"] = _normalize_policy_inventory(
            self.exact_entries, items_per_worker=1024
        )
        result["metadata"] = _normalize_policy_value(dict(self.metadata))
        return result

    def digest(self, *, generic_only: bool = False) -> str:
        """Hash either the frozen generic section or the complete final IR."""

        encoded = json.dumps(
            self.canonical_mapping(generic_only=generic_only),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def resolve_exact(self, key: RuntimeKey) -> ExactDispatchEntry | None:
        """Resolve an exact overlay without consulting generic policy."""

        return next((entry for entry in self.exact_entries if entry.key == key), None)

    def resolve_generic(
        self,
        domain: GenericDomain,
        aggregate_n: int,
        k: int,
        launch_k_tiles: int = 0,
    ) -> GenericDispatchRule | None:
        """Resolve only the frozen generic tree for sealed certification."""

        matches = [
            rule
            for rule in self.generic_rules
            if rule.domain == domain
            and rule.matches(aggregate_n, k, launch_k_tiles)
        ]
        if len(matches) > 1:
            raise ValueError(f"generic policy leaves overlap for {domain}")
        return matches[0] if matches else None


def make_policy_ir(
    exact_winners: Mapping[RuntimeKey, ExactWinner],
    generic_policy: GenericPolicy,
    *,
    metadata: Mapping[str, Any],
) -> PolicyIR:
    """Convert selected decisions into deterministic common policy IR."""

    exact_entries = tuple(
        ExactDispatchEntry(
            key=key,
            candidate_id=winner.candidate_id,
            arithmetic_fingerprint=winner.arithmetic_fingerprint,
            config_json=winner.config_json,
        )
        for key, winner in sorted(exact_winners.items())
    )
    return PolicyIR(
        policy_abi=POLICY_ABI,
        learner_version=LEARNER_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        exact_entries=exact_entries,
        generic_rules=generic_policy.rules,
        unpromoted_domains=generic_policy.unpromoted_domains,
        cross_validation=generic_policy.cross_validation,
        metadata=dict(metadata),
    )
