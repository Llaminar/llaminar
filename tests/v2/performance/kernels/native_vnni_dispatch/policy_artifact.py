"""Atomic frozen and certified NativeVNNI policy artifact handling.

Backend analyzers own only CSV adaptation and C++ emission. This module owns
the backend-neutral transaction boundary so CUDA, ROCm, and CPU cannot attach
different meanings to "installable". A production include is publishable only
when it is bound to a frozen generic policy and a complete sealed certificate.
The default criteria are strict five-percent p95 observed/UCB in at least 95%
of domains; an explicit best-effort transaction may record different criteria
without weakening correctness or structural gates. Worst-cell and global-p95
values remain diagnostic metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Mapping

from .compiler import CompiledPolicy, FrozenPolicy
from .corpus import RuntimeKey
from .segmented_policy import domain_promotion_quota_is_satisfied


def _mapping_digest(payload: Mapping[str, object]) -> str:
    """Return the canonical SHA-256 identity used by :class:`PolicyIR`."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _generic_policy_mapping(policy: Mapping[str, object]) -> dict[str, object]:
    """Extract exactly the fields covered by ``PolicyIR.digest(generic_only)``."""

    required = {
        "policy_abi",
        "learner_version",
        "feature_schema_version",
        "generic_rules",
        "unpromoted_domains",
        "cross_validation",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise ValueError(f"policy artifact omits generic IR fields: {missing}")
    return {key: policy[key] for key in sorted(required)}


def runtime_key_mapping(key: RuntimeKey) -> dict[str, object]:
    """Serialize one common runtime key for an auditable certificate cell."""

    return {
        "backend": key.backend.value,
        "architecture_class": key.architecture_class,
        "semantic_contract": key.semantic_contract.value,
        "operation_kind": key.operation_kind,
        "bundle_signature": key.bundle_signature,
        "projection_n_vector": list(key.projection_n_vector),
        "prepared_family_id": key.prepared_family_id,
        "packing_abi": key.packing_abi,
        "runtime_codebook_id": key.runtime_codebook_id,
        "execution_mode": key.execution_mode.value,
        "m": key.m,
        "aggregate_n": key.aggregate_n,
        "k": key.k,
        "launch_k_tiles": key.launch_k_tiles,
    }


def _normalize(value: object) -> object:
    """Convert nested policy dataclasses and enums to canonical JSON values."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _normalize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _normalize(asdict(value))
    return value


def certification_mapping(compiled: CompiledPolicy) -> dict[str, object]:
    """Serialize every promotion gate, cell, and frozen-rule coverage result."""

    report = compiled.certification
    return {
        "sealed_cell_count": report.sealed_cell_count,
        "out_of_scope_cell_count": report.out_of_scope_cell_count,
        "required_cell_count": report.required_cell_count,
        "covered_cell_count": report.covered_cell_count,
        "coverage": report.coverage,
        "verifier_bitwise_failures": report.verifier_bitwise_failures,
        "unexercised_rule_count": report.unexercised_rule_count,
        "unpromoted_domain_count": report.unpromoted_domain_count,
        "required_domain_count": report.required_domain_count,
        "p95_regret_budget": report.p95_regret_budget,
        "minimum_passing_domain_fraction": (
            report.minimum_passing_domain_fraction
        ),
        "passing_domain_count": report.passing_domain_count(),
        "passing_domain_fraction": report.passing_domain_fraction,
        "domain_promotion_quota_satisfied": (
            domain_promotion_quota_is_satisfied(
                report.passing_domain_count(),
                report.required_domain_count,
                minimum_passing_fraction=(
                    report.minimum_passing_domain_fraction
                ),
            )
        ),
        "max_observed_regret": report.max_observed_regret,
        "p95_observed_regret": report.p95_observed_regret,
        "p95_simultaneous_95pct_upper_regret": (
            report.p95_simultaneous_95pct_upper_regret
        ),
        "max_simultaneous_95pct_upper_regret": (
            report.max_simultaneous_95pct_upper_regret
        ),
        "rule_coverage": [
            {
                "sealed_hit_count": item.sealed_hit_count,
                "domain": _normalize(item.rule.domain),
                "predicates": _normalize(item.rule.predicates),
                "candidate_id": item.rule.candidate_id,
            }
            for item in report.rule_coverage
        ],
        "domain_results": [
            {
                "domain": _normalize(item.domain),
                "sealed_cell_count": item.sealed_cell_count,
                "p95_observed_regret": item.p95_observed_regret,
                "p95_simultaneous_95pct_upper_regret": (
                    item.p95_simultaneous_95pct_upper_regret
                ),
                "passes_p95_budget": item.passes(report.p95_regret_budget),
            }
            for item in report.domain_results
        ],
        "cells": [
            {
                "runtime_key": runtime_key_mapping(cell.runtime_key),
                "shape_group_id": cell.shape_group_id,
                "selected_candidate_id": cell.selected_candidate_id,
                "exact_candidate_id": cell.exact_candidate_id,
                "observed_worst_surface_regret": (
                    cell.observed_worst_surface_regret
                ),
                "simultaneous_95pct_upper_regret": (
                    cell.simultaneous_95pct_upper_regret
                ),
                "alias_count": cell.alias_count,
                "execution_mode_count": cell.execution_mode_count,
            }
            for cell in report.cells
        ],
    }


def write_frozen_policy(path: Path, frozen: FrozenPolicy) -> None:
    """Atomically publish development IR before sealed evidence is opened."""

    payload = {
        "state": "frozen_development",
        "policy": frozen.policy_ir.canonical_mapping(),
        "policy_digest": frozen.policy_ir.digest(),
        "frozen_generic_policy_digest": frozen.generic_digest,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_development_fit_diagnostic(path: Path, frozen: FrozenPolicy) -> None:
    """Publish reusable failed-fit IR that can never pass installation gates.

    This artifact exists so boundary-refinement planning can inspect every
    unpromoted domain and CV failure without opening sealed evidence or
    pretending that a partial generic policy was frozen for publication.
    Re-fitting the same immutable timing/profiler corpus may replace it freely.
    """

    promotion_diagnostics = _normalize(frozen.promotion_diagnostics)
    payload = {
        "state": "development_fit_diagnostic_noninstallable",
        "policy": frozen.policy_ir.canonical_mapping(),
        "policy_digest": frozen.policy_ir.digest(),
        "frozen_generic_policy_digest": frozen.generic_digest,
        "promotion_diagnostics": promotion_diagnostics,
        "promotion_diagnostics_digest": _mapping_digest({
            "promotion_diagnostics": promotion_diagnostics,
        }),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_frozen_policy_file(path: Path, frozen: FrozenPolicy) -> None:
    """Require a prior process to have published these exact development bytes."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    expected = {
        "state": "frozen_development",
        "policy": frozen.policy_ir.canonical_mapping(),
        "policy_digest": frozen.policy_ir.digest(),
        "frozen_generic_policy_digest": frozen.generic_digest,
    }
    if payload != expected:
        raise ValueError(
            "development policy does not match the pre-sealed frozen artifact"
        )


def compiled_policy_payload(compiled: CompiledPolicy) -> dict[str, object]:
    """Build the canonical production artifact and complete cell certificate."""

    return {
        "state": "sealed_certified",
        "policy": compiled.policy_ir.canonical_mapping(),
        "policy_digest": compiled.policy_ir.digest(),
        "frozen_generic_policy_digest": (
            compiled.certification.frozen_generic_policy_digest
        ),
        "certification": certification_mapping(compiled),
    }


def write_compiled_policy(path: Path, compiled: CompiledPolicy) -> None:
    """Atomically write common IR and its generic-only sealed certificate."""

    compiled.certification.require_promotable()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(compiled_policy_payload(compiled), sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_certification_diagnostic(path: Path, compiled: CompiledPolicy) -> None:
    """Write a sealed, explicitly non-installable report even when gates fail.

    The diagnostic retains the immutable frozen generic digest and complete
    selected-versus-exact cell inventory. It deliberately uses a state that the
    installation validator rejects, so debugging a failed fit cannot publish it
    accidentally as a production policy.
    """

    promotion_error = ""
    try:
        compiled.certification.require_promotable()
    except ValueError as error:
        promotion_error = str(error)
    payload = {
        "state": "sealed_certification_diagnostic_noninstallable",
        "promotable": not promotion_error,
        "promotion_error": promotion_error,
        "policy": compiled.policy_ir.canonical_mapping(),
        "policy_digest": compiled.policy_ir.digest(),
        "frozen_generic_policy_digest": (
            compiled.certification.frozen_generic_policy_digest
        ),
        "certification": certification_mapping(compiled),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_installable_policy_artifact(
    path: Path,
    *,
    include_path: Path | None = None,
) -> dict[str, object]:
    """Fail closed unless a staged policy is cryptographically certifiable.

    The optional include check binds backend-emitted C++ to both the complete IR
    and frozen generic digest.  This function is deliberately suitable for a
    final shell transaction immediately before an atomic install.
    """

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("state") != "sealed_certified":
        raise ValueError("policy artifact is not in sealed_certified state")
    policy = payload.get("policy")
    certificate = payload.get("certification")
    if not isinstance(policy, dict) or not isinstance(certificate, dict):
        raise ValueError("certified policy artifact has an invalid root schema")

    policy_digest = _mapping_digest(policy)
    if payload.get("policy_digest") != policy_digest:
        raise ValueError("certified policy digest does not match its common IR")
    generic_digest = _mapping_digest(_generic_policy_mapping(policy))
    if payload.get("frozen_generic_policy_digest") != generic_digest:
        raise ValueError("certified generic digest does not match its frozen IR")

    p95_regret = float(certificate.get("p95_regret_budget", float("nan")))
    minimum_passing_fraction = float(certificate.get(
        "minimum_passing_domain_fraction", float("nan")
    ))
    if not 0.0 < p95_regret <= 1.0:
        raise ValueError("sealed certificate has an invalid p95 regret budget")
    if not 0.0 <= minimum_passing_fraction <= 1.0:
        raise ValueError(
            "sealed certificate has an invalid passing-domain fraction"
        )
    metadata = policy.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("certified policy omits promotion metadata")
    if (
        metadata.get("promotion_p95_regret_budget") != p95_regret
        or metadata.get("promotion_minimum_passing_domain_fraction")
        != minimum_passing_fraction
    ):
        raise ValueError(
            "sealed certificate promotion criteria differ from the frozen policy"
        )

    sealed = int(certificate.get("sealed_cell_count", -1))
    out_of_scope = int(certificate.get("out_of_scope_cell_count", -1))
    required = int(certificate.get("required_cell_count", -1))
    covered = int(certificate.get("covered_cell_count", -1))
    if sealed < 0 or out_of_scope < 0 or sealed != required + out_of_scope:
        raise ValueError(
            "sealed generic scope accounting is inconsistent: "
            f"sealed={sealed} required={required} out_of_scope={out_of_scope}"
        )
    if out_of_scope != 0 or sealed != required:
        raise ValueError(
            "sealed certificate excludes cells from generic scope: "
            f"sealed={sealed} required={required} out_of_scope={out_of_scope}"
        )
    if required <= 0 or covered != required:
        raise ValueError(f"sealed coverage is incomplete: {covered}/{required}")
    if float(certificate.get("coverage", -1.0)) != 1.0:
        raise ValueError("sealed coverage is not exactly one")
    if int(certificate.get("verifier_bitwise_failures", -1)) != 0:
        raise ValueError("sealed certificate reports verifier byte failures")
    if int(certificate.get("unexercised_rule_count", -1)) != 0:
        raise ValueError("sealed certificate contains unexercised generic rules")
    if int(certificate.get("unpromoted_domain_count", -1)) != 0:
        raise ValueError("sealed certificate contains unpromoted generic domains")
    unpromoted = policy.get("unpromoted_domains")
    if not isinstance(unpromoted, list) or unpromoted:
        raise ValueError("certified policy is missing mandatory generic dispatch")
    domain_results = certificate.get("domain_results")
    if not isinstance(domain_results, list) or not domain_results:
        raise ValueError("sealed certificate omits per-domain p95 evidence")
    required_domains = int(certificate.get("required_domain_count", -1))
    if required_domains != len(domain_results):
        raise ValueError(
            "sealed domain accounting is inconsistent: "
            f"required={required_domains} results={len(domain_results)}"
        )
    seen_domains = set()
    passing_domains = 0
    domain_cell_count = 0
    for result in domain_results:
        if not isinstance(result, dict) or not isinstance(
            result.get("domain"), dict
        ):
            raise ValueError("sealed certificate has an invalid domain result")
        domain_identity = json.dumps(
            result["domain"], sort_keys=True, separators=(",", ":")
        )
        if domain_identity in seen_domains:
            raise ValueError("sealed certificate repeats a generic domain")
        seen_domains.add(domain_identity)
        domain_cells = int(result.get("sealed_cell_count", -1))
        if domain_cells <= 0:
            raise ValueError("sealed domain result has no covered cells")
        domain_cell_count += domain_cells
        observed = float(result.get("p95_observed_regret", float("inf")))
        upper = float(result.get(
            "p95_simultaneous_95pct_upper_regret", float("inf")
        ))
        passes = observed < p95_regret and upper < p95_regret
        if result.get("passes_p95_budget") is not passes:
            raise ValueError("sealed domain p95 decision is inconsistent")
        passing_domains += int(passes)
    if domain_cell_count != covered:
        raise ValueError(
            "sealed domain cells do not cover the certificate: "
            f"domains={domain_cell_count} covered={covered}"
        )
    reported_passing = int(certificate.get("passing_domain_count", -1))
    if reported_passing != passing_domains:
        raise ValueError(
            "sealed passing-domain accounting is inconsistent: "
            f"reported={reported_passing} actual={passing_domains}"
        )
    passing_fraction = float(certificate.get("passing_domain_fraction", -1.0))
    expected_fraction = float(passing_domains) / float(required_domains)
    if passing_fraction != expected_fraction:
        raise ValueError("sealed passing-domain fraction is inconsistent")
    quota_satisfied = domain_promotion_quota_is_satisfied(
        passing_domains,
        required_domains,
        minimum_passing_fraction=minimum_passing_fraction,
    )
    if certificate.get("domain_promotion_quota_satisfied") is not quota_satisfied:
        raise ValueError("sealed domain promotion decision is inconsistent")
    if not quota_satisfied:
        raise ValueError(
            "sealed domain promotion quota "
            f"{passing_domains}/{required_domains} does not reach "
            f"{100.0 * minimum_passing_fraction:g}%"
        )

    cells = certificate.get("cells")
    if not isinstance(cells, list) or len(cells) != required:
        raise ValueError("sealed certificate cell inventory is incomplete")
    if include_path is not None:
        source = include_path.read_text(encoding="utf-8")
        if f"// Common policy digest: {policy_digest}" not in source:
            raise ValueError("generated include is not bound to the common policy IR")
        if f"// Frozen generic policy digest: {generic_digest}" not in source:
            raise ValueError("generated include is not bound to the frozen generic IR")
    return payload


def main() -> int:
    """Validate one staged policy/include pair at the installation boundary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-json", required=True, type=Path)
    parser.add_argument("--include", required=True, type=Path, dest="include_path")
    args = parser.parse_args()
    validate_installable_policy_artifact(
        args.policy_json,
        include_path=args.include_path,
    )
    print(f"certified installable policy: {args.policy_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
