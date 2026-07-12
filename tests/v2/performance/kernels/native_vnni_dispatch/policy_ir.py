"""Backend-neutral exact/generic NativeVNNI policy IR and deterministic digest."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
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
        """Return a stable JSON representation used by manifests and emitters."""

        def normalize(value):
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in sorted(value.items())}
            if isinstance(value, (tuple, list)):
                return [normalize(item) for item in value]
            if hasattr(value, "__dataclass_fields__"):
                return normalize(asdict(value))
            return value

        result = {
            "policy_abi": self.policy_abi,
            "learner_version": self.learner_version,
            "feature_schema_version": self.feature_schema_version,
            "generic_rules": normalize(self.generic_rules),
            "unpromoted_domains": normalize(self.unpromoted_domains),
            "cross_validation": normalize(self.cross_validation),
        }
        if not generic_only:
            result["exact_entries"] = normalize(self.exact_entries)
            result["metadata"] = normalize(dict(self.metadata))
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

    def resolve_generic(self, domain: GenericDomain, work_items: int) -> GenericDispatchRule | None:
        """Resolve only the frozen generic policy for sealed certification."""

        return next((
            rule
            for rule in self.generic_rules
            if rule.domain == domain
            and rule.min_work_items <= work_items <= rule.max_work_items
        ), None)


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
