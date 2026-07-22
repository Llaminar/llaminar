"""Fresh frozen-leaf sealing for CPU NativeVNNI grouped verification.

The grouped policy is certified only after its generic tree is immutable. One
new geometry is generated inside every leaf, and every source-format alias is
measured as an interleaved selected-versus-forceable-candidate contest at the
leaf's exact runtime M. The C++ transaction invokes the production grouped
entry point, proves the requested route, and compares every output byte with
independent production M=1 decode rows before timing can enter this plan.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .candidate_registry import cpu_native_vnni_verifier_registry
from .certification import CertificationReport
from .corpus import GenericDomain, ObservationCorpus
from .cpu_sealed_paired import (
    DEFAULT_BOOTSTRAP_REPLICATES,
    certify_cpu_sealed_pairs,
    cpu_sealed_development_costs,
    fresh_geometry_candidates_for_rule,
    sha256_json,
    write_cpu_sealed_request_shards,
)
from .exact_oracle import candidate_is_eligible
from .format_registry import format_spec, registry_digest, runtime_aliases
from .paired_requests import PairedTimingRequest, write_json
from .schema import SemanticContract
from .segmented_policy import CandidatePointCost, GenericDispatchRule
from .shape_manifest import NativeVNNIShapeManifest


CPU_GROUPED_SEALED_PLAN_SCHEMA = (
    "cpu-grouped-decode-frozen-leaf-paired-seal-v3"
)
CPU_GROUPED_SEALED_RESERVE_SCHEMA = "cpu-grouped-decode-sealed-reserve-v3"
CPU_GROUPED_SEALED_GEOMETRY_PREFIX = "CPUGroupedDecodeAutoSeal_"
CPU_GROUPED_SEALED_OBSERVED_PATH = "grouped-verifier-rows"


def cpu_grouped_sealed_reserve_commitment(
    manifest: NativeVNNIShapeManifest,
) -> str:
    """Commit fitting to the grouped fresh-geometry reserve algorithm."""

    return sha256_json({
        "schema_version": CPU_GROUPED_SEALED_RESERVE_SCHEMA,
        "shape_manifest_digest": manifest.digest(),
        "candidate_registry_digest": (
            cpu_native_vnni_verifier_registry().digest()
        ),
        "format_registry_digest": registry_digest(),
        "geometry_prefix": CPU_GROUPED_SEALED_GEOMETRY_PREFIX,
        "candidate_count_per_rule": 1,
        "search_pattern": "complete-chebyshev-ring-perimeter-v1",
        "delta_quantum": 32,
        "maximum_delta_steps": 128,
    })


def _fresh_shape_name(n: int, k: int) -> str:
    """Name a generated grouped geometry independently of timing outcome."""

    return f"{CPU_GROUPED_SEALED_GEOMETRY_PREFIX}N{n}_K{k}"


def _request_id(fields: Mapping[str, object], plan_seed: str) -> str:
    """Bind one grouped challenger edge to its complete sealed plan."""

    return "nvnni-grouped-seal-" + sha256_json({
        "fields": dict(fields),
        "plan_seed": plan_seed,
    }).removeprefix("sha256:")[:24]


def _registered_candidates() -> tuple[str, ...]:
    """Return the complete economical grouped production candidate set."""

    candidates = tuple(
        candidate.candidate_id
        for candidate in cpu_native_vnni_verifier_registry().entries
    )
    if not candidates:
        raise ValueError("CPU grouped seal requires registered candidates")
    return candidates


def _forceable_candidates_for_rule(
    rule: GenericDispatchRule,
    development: ObservationCorpus,
) -> tuple[str, ...]:
    """Return candidates proved forceable throughout one runtime domain.

    Candidate registration means that a schedule exists in production; it
    does not mean every physical schedule is distinct in every ISA/M domain.
    For example, four-row ``WideRows`` intentionally normalizes to the two-row
    grouped kernel for AVX2 and for M=2. Full-K reuse families similarly
    normalize to Pairwise when serial M1 owns several K tiles. The
    authenticated development corpus records those launches as unsupported
    negative evidence. A sealed tournament must preserve that distinction or
    it will attempt to time several labels for the same physical route.

    Requiring every source-format alias makes the result conservative: one
    candidate is forceable for the rule only when the real production route
    and byte-equivalence gates passed for every alias owned by its runtime
    codebook.
    """

    rows = development.rows_for_generic_domain(rule.domain)
    if not rows:
        raise ValueError("CPU grouped rule has no development observations")
    expected_aliases = frozenset(runtime_aliases(
        "cpu", rule.domain.runtime_codebook_id
    ))
    aliases_by_candidate: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if candidate_is_eligible(row, row.serial_m1_policy_hash):
            aliases_by_candidate[row.candidate_id].add(row.source_format)
    forceable = tuple(
        candidate_id
        for candidate_id in _registered_candidates()
        if frozenset(aliases_by_candidate[candidate_id]) == expected_aliases
    )
    if rule.candidate_id not in forceable:
        raise ValueError(
            "CPU grouped leaf selected a candidate that is not forceable "
            f"throughout its domain: {rule.candidate_id}"
        )
    return forceable


@dataclass(frozen=True, order=True)
class CPUGroupedSealedRuleWitness:
    """One frozen grouped leaf and its unseen geometry/M witness."""

    rule_index: int
    shape: str
    m: int
    n: int
    k: int
    architecture_class: str
    runtime_codebook: int
    bundle_signature: str
    selected_candidate_id: str
    forceable_candidate_ids: tuple[str, ...]


@dataclass(frozen=True, order=True)
class CPUGroupedSealedRequest:
    """Associate one paired candidate edge with a grouped generic leaf."""

    rule_index: int
    request: PairedTimingRequest


@dataclass(frozen=True)
class CPUGroupedSealedPlan:
    """Complete grouped fresh-leaf witness and challenger transaction."""

    schema_version: str
    frozen_generic_policy_digest: str
    development_corpus_digest: str
    sealed_build_id: str
    reserve_commitment: str
    candidate_registry_digest: str
    format_registry_digest: str
    plan_digest: str
    rule_witnesses: tuple[CPUGroupedSealedRuleWitness, ...]
    requests: tuple[CPUGroupedSealedRequest, ...]

    def canonical_mapping(
        self, *, include_plan_digest: bool = True
    ) -> dict[str, object]:
        """Return stable JSON for auditing and request regeneration."""

        result = {
            "schema_version": self.schema_version,
            "frozen_generic_policy_digest": self.frozen_generic_policy_digest,
            "development_corpus_digest": self.development_corpus_digest,
            "sealed_build_id": self.sealed_build_id,
            "reserve_commitment": self.reserve_commitment,
            "candidate_registry_digest": self.candidate_registry_digest,
            "format_registry_digest": self.format_registry_digest,
            "rule_witnesses": [asdict(item) for item in self.rule_witnesses],
            "requests": [
                {"rule_index": item.rule_index, **asdict(item.request)}
                for item in self.requests
            ],
        }
        if include_plan_digest:
            result["plan_digest"] = self.plan_digest
        return result


def build_cpu_grouped_sealed_plan(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    sealed_build_id: str,
    supplemental_dimensions_by_group: Mapping[str, tuple[int, int]] | None = None,
) -> CPUGroupedSealedPlan:
    """Generate every grouped leaf witness after the policy is frozen."""

    if not rules or not frozen_generic_policy_digest.startswith("sha256:"):
        raise ValueError("CPU grouped sealing requires a frozen generic policy")
    if not sealed_build_id.startswith("sha256:"):
        raise ValueError("CPU grouped sealing requires a sealed build digest")
    dimensions_by_group: dict[str, tuple[int, int]] = {}
    for row in development:
        dimensions = (row.aggregate_n, row.k)
        previous = dimensions_by_group.setdefault(row.shape_group_id, dimensions)
        if previous != dimensions:
            raise ValueError("CPU grouped shape group changed dimensions")
    for shape_group_id, dimensions in (
        supplemental_dimensions_by_group or {}
    ).items():
        previous = dimensions_by_group.setdefault(shape_group_id, dimensions)
        if previous != dimensions:
            raise ValueError(
                "CPU grouped supplemental shape group changed dimensions"
            )
    forbidden = frozenset({
        *dimensions_by_group.values(),
        *((shape.n, shape.k) for shape in manifest.shapes),
    })
    witnesses = []
    for rule_index, rule in enumerate(rules):
        if (
            rule.domain.semantic_contract
            != SemanticContract.VERIFIER_SERIAL_M1_BITWISE
            or rule.domain.m <= 1
        ):
            raise ValueError("CPU grouped seal received a non-verifier rule")
        if rule.candidate_id not in _registered_candidates():
            raise ValueError(
                f"grouped leaf selects unknown candidate {rule.candidate_id}"
            )
        forceable = _forceable_candidates_for_rule(rule, development)
        n, k = fresh_geometry_candidates_for_rule(
            rule,
            dimensions_by_group,
            forbidden,
            manifest.maximum_cpu_measurement_weight_elements,
            1,
        )[0]
        witnesses.append(CPUGroupedSealedRuleWitness(
            rule_index=rule_index,
            shape=_fresh_shape_name(n, k),
            m=rule.domain.m,
            n=n,
            k=k,
            architecture_class=rule.domain.architecture_class,
            runtime_codebook=rule.domain.runtime_codebook_id,
            bundle_signature=rule.domain.bundle_signature,
            selected_candidate_id=rule.candidate_id,
            forceable_candidate_ids=forceable,
        ))

    seed_payload = {
        "schema_version": CPU_GROUPED_SEALED_PLAN_SCHEMA,
        "frozen_generic_policy_digest": frozen_generic_policy_digest,
        "development_corpus_digest": development.digest(),
        "sealed_build_id": sealed_build_id,
        "reserve_commitment": cpu_grouped_sealed_reserve_commitment(manifest),
        "candidate_registry_digest": (
            cpu_native_vnni_verifier_registry().digest()
        ),
        "format_registry_digest": registry_digest(),
        "rule_witnesses": [asdict(item) for item in witnesses],
    }
    plan_digest = sha256_json(seed_payload)
    requests = []
    for witness in witnesses:
        rule = rules[witness.rule_index]
        for source_format in runtime_aliases("cpu", witness.runtime_codebook):
            spec = format_spec(source_format)
            challengers = tuple(
                candidate_id
                for candidate_id in witness.forceable_candidate_ids
                if candidate_id != witness.selected_candidate_id
            )
            reason = "grouped_frozen_leaf_exhaustive_challenger"
            if not challengers:
                # A one-candidate domain has zero dispatch regret by
                # construction, but its generic rule still needs untouched
                # route and byte-equivalence evidence. Replaying the selected
                # grouped schedule in both interleaved positions supplies that
                # proof without inventing a dispatchable serial fallback.
                challengers = (witness.selected_candidate_id,)
                reason = "grouped_frozen_leaf_single_candidate_witness"
            for challenger in challengers:
                fields = {
                    "rule_index": witness.rule_index,
                    "source_format": source_format,
                    "architecture_class": witness.architecture_class,
                    "shape": witness.shape,
                    "m": witness.m,
                    "n": witness.n,
                    "k": witness.k,
                    "selected_candidate_id": witness.selected_candidate_id,
                    "exact_candidate_id": challenger,
                }
                request = PairedTimingRequest(
                    request_id=_request_id(fields, plan_digest),
                    source_format=source_format,
                    source_codebook=spec.source_codebook_id,
                    execution_codebook=spec.cpu_execution_codebook_id,
                    architecture_class=witness.architecture_class,
                    shape=witness.shape,
                    shape_group_id=(
                        f"cpu-grouped-sealed:{witness.shape}:m{witness.m}:"
                        f"n{witness.n}:k{witness.k}"
                    ),
                    execution_mode=rule.domain.execution_mode.value,
                    m=witness.m,
                    n=witness.n,
                    k=witness.k,
                    selected_candidate_id=witness.selected_candidate_id,
                    exact_candidate_id=challenger,
                    observed_cv_regret=0.0,
                    reason=reason,
                )
                requests.append(CPUGroupedSealedRequest(
                    witness.rule_index, request
                ))
    if not requests:
        raise ValueError("CPU grouped sealed plan contains no challenger edges")
    return CPUGroupedSealedPlan(
        schema_version=CPU_GROUPED_SEALED_PLAN_SCHEMA,
        frozen_generic_policy_digest=frozen_generic_policy_digest,
        development_corpus_digest=development.digest(),
        sealed_build_id=sealed_build_id,
        reserve_commitment=cpu_grouped_sealed_reserve_commitment(manifest),
        candidate_registry_digest=(
            cpu_native_vnni_verifier_registry().digest()
        ),
        format_registry_digest=registry_digest(),
        plan_digest=plan_digest,
        rule_witnesses=tuple(witnesses),
        requests=tuple(sorted(requests)),
    )


def write_cpu_grouped_sealed_plan(
    path: Path,
    plan: CPUGroupedSealedPlan,
    request_shard_directory: Path,
    *,
    max_requests_per_shard: int = 16,
) -> tuple[Path, ...]:
    """Publish the grouped audit plan and resumable request shards."""

    write_json(path, plan.canonical_mapping())
    return write_cpu_sealed_request_shards(
        request_shard_directory,
        plan.requests,
        plan.development_corpus_digest,
        plan.plan_digest,
        "cpu-grouped-decode-sealed-request-shards-v1",
        max_requests_per_shard=max_requests_per_shard,
    )


def read_cpu_grouped_sealed_plan(path: Path) -> CPUGroupedSealedPlan:
    """Read a grouped plan and reject stale or self-inconsistent inputs."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "frozen_generic_policy_digest",
        "development_corpus_digest",
        "sealed_build_id",
        "reserve_commitment",
        "candidate_registry_digest",
        "format_registry_digest",
        "plan_digest",
        "rule_witnesses",
        "requests",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("CPU grouped sealed plan fields are invalid")
    witnesses = tuple(
        CPUGroupedSealedRuleWitness(
            rule_index=int(item["rule_index"]),
            shape=str(item["shape"]),
            m=int(item["m"]),
            n=int(item["n"]),
            k=int(item["k"]),
            architecture_class=str(item["architecture_class"]),
            runtime_codebook=int(item["runtime_codebook"]),
            bundle_signature=str(item["bundle_signature"]),
            selected_candidate_id=str(item["selected_candidate_id"]),
            forceable_candidate_ids=tuple(item["forceable_candidate_ids"]),
        )
        for item in raw["rule_witnesses"]
    )
    request_fields = set(PairedTimingRequest.__dataclass_fields__)
    requests = []
    for item in raw["requests"]:
        if set(item) != request_fields | {"rule_index"}:
            raise ValueError("CPU grouped sealed request fields are invalid")
        requests.append(CPUGroupedSealedRequest(
            int(item["rule_index"]),
            PairedTimingRequest(**{
                key: item[key] for key in request_fields
            }),
        ))
    plan = CPUGroupedSealedPlan(
        schema_version=str(raw["schema_version"]),
        frozen_generic_policy_digest=str(
            raw["frozen_generic_policy_digest"]
        ),
        development_corpus_digest=str(raw["development_corpus_digest"]),
        sealed_build_id=str(raw["sealed_build_id"]),
        reserve_commitment=str(raw["reserve_commitment"]),
        candidate_registry_digest=str(raw["candidate_registry_digest"]),
        format_registry_digest=str(raw["format_registry_digest"]),
        plan_digest=str(raw["plan_digest"]),
        rule_witnesses=witnesses,
        requests=tuple(requests),
    )
    seed = plan.canonical_mapping(include_plan_digest=False)
    seed.pop("requests")
    if sha256_json(seed) != plan.plan_digest:
        raise ValueError("CPU grouped sealed plan digest is invalid")
    if plan.schema_version != CPU_GROUPED_SEALED_PLAN_SCHEMA:
        raise ValueError("unsupported CPU grouped sealed plan schema")
    if (
        plan.candidate_registry_digest
        != cpu_native_vnni_verifier_registry().digest()
    ):
        raise ValueError("CPU grouped plan uses a stale candidate registry")
    if plan.format_registry_digest != registry_digest():
        raise ValueError("CPU grouped plan uses a stale format registry")
    if tuple(item.rule_index for item in witnesses) != tuple(
        range(len(witnesses))
    ):
        raise ValueError("CPU grouped sealed rule indices are incomplete")
    if len({item.request.request_id for item in requests}) != len(requests):
        raise ValueError("CPU grouped sealed plan repeats request IDs")
    return plan


def validate_cpu_grouped_sealed_plan(
    plan: CPUGroupedSealedPlan,
    rules: tuple[GenericDispatchRule, ...],
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    sealed_build_id: str,
    supplemental_dimensions_by_group: Mapping[str, tuple[int, int]] | None = None,
) -> None:
    """Replay freshness, leaf ownership, M, alias, and challenger totality."""

    if len(plan.rule_witnesses) != len(rules):
        raise ValueError("CPU grouped plan does not cover every frozen rule")
    if plan.development_corpus_digest != development.digest():
        raise ValueError("CPU grouped plan development digest changed")
    if plan.sealed_build_id != sealed_build_id:
        raise ValueError("CPU grouped plan sealed build digest changed")
    if plan.reserve_commitment != cpu_grouped_sealed_reserve_commitment(manifest):
        raise ValueError("CPU grouped reserve commitment changed")
    development_dimensions = {
        *((row.aggregate_n, row.k) for row in development),
        *(supplemental_dimensions_by_group or {}).values(),
    }
    static_dimensions = {(shape.n, shape.k) for shape in manifest.shapes}
    requests_by_rule: dict[int, list[PairedTimingRequest]] = defaultdict(list)
    for item in plan.requests:
        requests_by_rule[item.rule_index].append(item.request)
    for index, (rule, witness) in enumerate(zip(
        rules, plan.rule_witnesses, strict=True
    )):
        forceable = _forceable_candidates_for_rule(rule, development)
        if (
            witness.rule_index != index
            or witness.m != rule.domain.m
            or not rule.matches(witness.n, witness.k)
        ):
            raise ValueError(
                f"CPU grouped witness does not exercise rule {index}"
            )
        if (
            witness.architecture_class != rule.domain.architecture_class
            or witness.runtime_codebook != rule.domain.runtime_codebook_id
            or witness.bundle_signature != rule.domain.bundle_signature
            or witness.selected_candidate_id != rule.candidate_id
            or witness.forceable_candidate_ids != forceable
        ):
            raise ValueError(f"CPU grouped witness changed rule {index}")
        if (witness.n, witness.k) in development_dimensions | static_dimensions:
            raise ValueError("CPU grouped witness geometry was visible to fitting")
        challengers = tuple(
            candidate_id
            for candidate_id in forceable
            if candidate_id != witness.selected_candidate_id
        ) or (witness.selected_candidate_id,)
        expected = {
            (source_format, challenger)
            for source_format in runtime_aliases(
                "cpu", witness.runtime_codebook
            )
            for challenger in challengers
        }
        observed = {
            (request.source_format, request.exact_candidate_id)
            for request in requests_by_rule[index]
        }
        if observed != expected:
            raise ValueError(
                f"CPU grouped challenger matrix incomplete for rule {index}"
            )


def certify_cpu_grouped_sealed_pairs(
    rules: tuple[GenericDispatchRule, ...],
    frozen_generic_policy_digest: str,
    plan: CPUGroupedSealedPlan,
    evidence_paths: Iterable[Path],
    *,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    workers: int | None = None,
) -> CertificationReport:
    """Build the grouped certificate from byte-proven paired evidence."""

    return certify_cpu_sealed_pairs(
        rules,
        frozen_generic_policy_digest,
        plan.frozen_generic_policy_digest,
        plan.plan_digest,
        plan.rule_witnesses,
        plan.requests,
        evidence_paths,
        lambda _rule: CPU_GROUPED_SEALED_OBSERVED_PATH,
        bootstrap_replicates=bootstrap_replicates,
        workers=workers,
    )


def cpu_grouped_burned_seal_costs(
    development: ObservationCorpus,
    plan: CPUGroupedSealedPlan,
    evidence_paths: Iterable[Path],
) -> dict[GenericDomain, tuple[CandidatePointCost, ...]]:
    """Promote one inspected grouped seal to generic-only development costs.

    A burned plan intentionally predates the corpus generation it informs.
    Authenticate its immutable registry identities, then map each witness onto
    the enlarged corpus by the complete generic-domain owner below. Requiring
    the old and new corpus digests to match would reject every legitimate
    additive geometry or later burned-seal generation.
    """

    if (
        plan.candidate_registry_digest
        != cpu_native_vnni_verifier_registry().digest()
    ):
        raise ValueError("burned CPU grouped seal uses a stale candidate registry")
    if plan.format_registry_digest != registry_digest():
        raise ValueError("burned CPU grouped seal uses a stale format registry")
    domains = development.generic_domains()

    def domain_for_witness(
        witness: CPUGroupedSealedRuleWitness,
    ) -> GenericDomain:
        matches = tuple(
            domain
            for domain in domains
            if domain.architecture_class == witness.architecture_class
            and domain.runtime_codebook_id == witness.runtime_codebook
            and domain.bundle_signature == witness.bundle_signature
            and domain.m == witness.m
        )
        if len(matches) != 1:
            raise ValueError(
                "burned CPU grouped witness does not identify one generic domain"
            )
        return matches[0]

    return cpu_sealed_development_costs(
        plan.plan_digest,
        plan.rule_witnesses,
        plan.requests,
        evidence_paths,
        domain_for_witness,
        lambda _domain: CPU_GROUPED_SEALED_OBSERVED_PATH,
    )
