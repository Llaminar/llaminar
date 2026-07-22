#!/usr/bin/env python3
"""Compile strong CPU NativeVNNI prefill evidence into a staged selector.

The selector is deliberately independent from the grouped-verifier table.
Runtime M is bucketed to the nearest measured depth and clamped by the exact
projection geometry's economical tier. Every emitted candidate retains one
full-K accumulation tile and has passed serial-M1 byte equality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_prefill import (  # noqa: E402
    CPU_PREFILL_FULL_K_BUNDLE,
    CPU_PREFILL_KPART_BUNDLE,
    CPUPrefillAdapterContext,
    adapt_cpu_prefill_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    OPTIONAL_OBSERVATION_COLUMNS,
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_prefill_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    certify_frozen_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import ObservationCorpus, RuntimeKey  # noqa: E402
from native_vnni_dispatch.cpu_prefill_route_manifest import (  # noqa: E402
    CPU_PREFILL_ROUTE_BUNDLES,
    CPUPrefillSerialRouteManifest,
    read_cpu_prefill_route_manifests,
)
from native_vnni_dispatch.cpu_prefill_split_manifest import (  # noqa: E402
    MANIFEST_PATH,
    CPUPrefillSplitManifest,
    load_cpu_prefill_split_manifest,
)
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
)
from native_vnni_dispatch.cpu_prefill_training_plan import (  # noqa: E402
    CPUPrefillSourceTrainingRecord,
    CPUPrefillSealedWitnessPlan,
    ISA_REGIMES,
    build_cpu_prefill_sealed_witness_plan,
    cpu_prefill_development_refinement_source_training_records,
    cpu_prefill_runtime_training_cells,
    cpu_prefill_source_training_records,
    read_cpu_prefill_sealed_witness_plan,
    validate_cpu_prefill_sealed_witness_plan,
    write_cpu_prefill_sealed_witness_plan,
)
from native_vnni_dispatch.cpu_prefill_generic_refinement import (  # noqa: E402
    CPUPrefillGenericRefinementPlan,
    read_cpu_prefill_generic_refinement_plan,
)
from native_vnni_dispatch.cpu_prefill_development_lineage import (  # noqa: E402
    CPUPrefillDevelopmentLineagePlan,
    read_cpu_prefill_development_lineage_plan,
)
from native_vnni_dispatch.cpu_prefill_candidate_expansion import (  # noqa: E402
    CPUPrefillCandidateExpansionPlan,
    normalize_cpu_prefill_candidate_expansion,
    read_cpu_prefill_candidate_expansion_plan,
    validate_normalized_cpu_prefill_candidate_expansion,
)
from native_vnni_dispatch.cpu_prefill_burned_seal import (  # noqa: E402
    CPUPrefillBurnedSealTransaction,
    read_cpu_prefill_burned_seal_transaction,
)
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.prefill_matrix import (  # noqa: E402
    CPU_PREFILL_M_BUCKETS,
    cpu_prefill_measurements,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerFeatureCatalog,
    load_profiler_feature_catalog,
    merge_profiler_feature_catalogs,
)
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_frozen_policy_file,
    write_certification_diagnostic,
    write_compiled_policy,
    write_development_fit_diagnostic,
    write_frozen_policy,
)
from native_vnni_dispatch.policy_ir import PolicyIR  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    NativeVNNIObservation,
    OBSERVATION_COLUMNS,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    DEFAULT_TREE_LEAVES,
    MAX_TREE_LEAVES,
    GenericDispatchRule,
    PolicyFitCache,
    fit_generic_policy,
    validate_generic_rule_partition,
)
from native_vnni_dispatch.validation import (  # noqa: E402
    require_candidate_matrix_complete,
    require_canonical_alias_coverage,
    require_registry_candidate_coverage,
)


def _read_lineage_source_refinement_plans(
    paths: list[Path],
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifests: tuple[CPUPrefillSplitManifest, ...],
) -> tuple[CPUPrefillGenericRefinementPlan, ...]:
    """Authenticate historical refinement plans against their own splits.

    A development lineage can span more than one reviewed split. For example,
    the current v9-to-v10 migration includes early refinement rounds created
    under v8 and later rounds created under v9. Treating every plan as though it
    belonged to the newest source split either rejects valid evidence or tempts
    callers to omit the older rounds entirely.

    Each refinement JSON already carries its immutable split digest. This
    helper reads only that routing field first, selects the matching reviewed
    split, then delegates the complete schema and inventory authentication to
    ``read_cpu_prefill_generic_refinement_plan``. No unvalidated plan object is
    ever returned to fitting.
    """

    manifests_by_digest: dict[str, CPUPrefillSplitManifest] = {}
    for manifest in split_manifests:
        digest = manifest.digest()
        if digest in manifests_by_digest:
            raise ValueError(
                "CPU prefill lineage source split manifests are duplicated: "
                f"{digest}"
            )
        manifests_by_digest[digest] = manifest

    plans: list[CPUPrefillGenericRefinementPlan] = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(
                f"{path}: CPU prefill refinement plan root is not an object"
            )
        split_digest = raw.get("split_manifest_digest")
        if not isinstance(split_digest, str) or not split_digest:
            raise ValueError(
                f"{path}: CPU prefill refinement plan has no split digest"
            )
        try:
            split_manifest = manifests_by_digest[split_digest]
        except KeyError as error:
            raise ValueError(
                f"{path}: no lineage source split manifest matches "
                f"{split_digest}"
            ) from error
        plans.append(
            read_cpu_prefill_generic_refinement_plan(
                path,
                route_manifest,
                split_manifest,
            )
        )
    return tuple(plans)


@dataclass(frozen=True, order=True)
class PrefillPolicyEntry:
    """One exact common-oracle decision for the CPU selector ABI."""

    build_isa: str
    runtime_isa: str
    threads: int
    codebook: int
    m: int
    n: int
    k: int
    bundle_signature: str
    policy: str
    candidate_id: str
    shape_name: str
    max_surface_regret: float
    max_cv: float
    shape_aliases: tuple[str, ...] = ()

    @property
    def shape_names(self) -> tuple[str, ...]:
        """Return every measured name sharing this exact runtime geometry."""

        return (self.shape_name, *self.shape_aliases)


@dataclass(frozen=True)
class CPUPrefillGenericRule:
    """One generic rule and its certified CPU runtime surface."""

    build_isa: str
    runtime_isa: str
    threads: int
    rule: GenericDispatchRule


def _cpu_runtime_surface(architecture_class: str) -> tuple[str, str, int]:
    parts = architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(f"malformed CPU architecture class {architecture_class!r}")
    build = parts[1].removeprefix("build=")
    runtime = parts[2].removeprefix("runtime=")
    threads = int(parts[3].removeprefix("threads="))
    if build not in {"AVX2", "AVX512"} or runtime not in {"AVX2", "AVX512"}:
        raise ValueError("CPU prefill policy requires AVX2/AVX512")
    if build == "AVX2" and runtime != "AVX2":
        raise ValueError("AVX2 build cannot own AVX512 policy")
    if threads <= 0:
        raise ValueError("CPU prefill thread count must be positive")
    return build, runtime, threads


def _candidate_policy(candidate_id: str) -> str:
    candidate = cpu_native_vnni_prefill_registry().resolve(candidate_id)
    config = candidate.config_json
    if config["route"] == "row_chunk_grid":
        return "RowChunkGrid"
    if config["route"] == "decode_equivalent_kpart_rows":
        return (
            "KPartWideRows"
            if config["policy"] == "WideRows"
            else "KPartPairwise"
        )
    nbc = int(config["n_block_chunks"])
    if nbc not in {1, 2, 4, 8, 16}:
        raise ValueError(f"unsupported generated CPU prefill nbc={nbc}")
    if config["route"] == "two_row_pair_grid":
        return f"TwoRowPairGridNbc{nbc}"
    if config["route"] == "two_row_full_output_tiles":
        return f"TwoRowNbc{nbc}"
    raise ValueError(
        f"unsupported generated CPU prefill route {config['route']!r}"
    )


def _bundle_uses_serial_kpart(bundle_signature: str) -> bool:
    """Decode the authenticated production arithmetic regime."""

    if bundle_signature == CPU_PREFILL_KPART_BUNDLE:
        return True
    if bundle_signature == CPU_PREFILL_FULL_K_BUNDLE:
        return False
    raise ValueError(f"unsupported CPU prefill bundle {bundle_signature!r}")


def _serial_hashes(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> dict[RuntimeKey, str]:
    return {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }


def _production_exact_overlay_shapes() -> dict[
    tuple[int, int, int], tuple[str, ...]
]:
    """Return declared production shape names for each exact runtime cell.

    Adaptive refinement geometries intentionally look like ordinary timing
    rows because they train the same generic policy. They are not released
    model shapes and must never become exact dispatch overlays. The checked-in
    production prefill matrix is the sole authority for overlay eligibility.
    """

    names: dict[tuple[int, int, int], set[str]] = {}
    for measurement in cpu_prefill_measurements():
        for m in measurement.m_values:
            names.setdefault(
                (measurement.shape.n, measurement.shape.k, m),
                set(),
            ).add(measurement.shape.name)
    return {
        geometry: tuple(sorted(shape_names))
        for geometry, shape_names in names.items()
    }


def _retain_production_exact_overlays(policy_ir: PolicyIR) -> PolicyIR:
    """Remove synthetic learning geometries from exact dispatch publication."""

    production_cells = _production_exact_overlay_shapes()
    return replace(
        policy_ir,
        exact_entries=tuple(
            entry for entry in policy_ir.exact_entries
            if (
                entry.key.aggregate_n,
                entry.key.k,
                entry.key.m,
            ) in production_cells
        ),
    )


def select_entries(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[PrefillPolicyEntry]:
    """Choose byte-exact candidates with the common robust oracle."""

    winners = build_exact_winners(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    production_cells = _production_exact_overlay_shapes()
    entries = []
    for key, winner in sorted(winners.items()):
        if key.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            raise ValueError(f"unexpected CPU prefill contract {key}")
        shape_names = production_cells.get((key.aggregate_n, key.k, key.m))
        if shape_names is None:
            continue
        if len(key.projection_n_vector) != 1:
            raise ValueError(f"ambiguous CPU prefill runtime key {key}")
        build_isa, runtime_isa, threads = _cpu_runtime_surface(
            key.architecture_class
        )
        entries.append(PrefillPolicyEntry(
            build_isa=build_isa,
            runtime_isa=runtime_isa,
            threads=threads,
            codebook=key.runtime_codebook_id,
            m=key.m,
            n=key.aggregate_n,
            k=key.k,
            bundle_signature=key.bundle_signature,
            policy=_candidate_policy(winner.candidate_id),
            candidate_id=winner.candidate_id,
            shape_name=shape_names[0],
            max_surface_regret=winner.max_surface_regret,
            max_cv=winner.max_cv,
            shape_aliases=shape_names[1:],
        ))
    return entries


def select_generic_rules(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[CPUPrefillGenericRule]:
    """Fit cross-aspect shape rules for non-exact production geometries.

    Aspect ratio already participates in every reviewed feature policy. Keeping
    a hard aspect bucket in the outer domain as well would require three
    independently measured shapes in every bucket before grouped CV can run.
    CPU prefill instead retains each row's true ratio and exact runtime key while
    collapsing only the generic-domain index. The fitted tree must learn any
    economical aspect boundary from measured evidence.
    """

    generic_corpus = corpus.with_collapsed_aspect_domains()
    generic = fit_generic_policy(
        generic_corpus,
        serial_m1_hashes=_serial_hashes(generic_corpus, serial_m1_policy_hash),
    )
    return [
        CPUPrefillGenericRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in generic.rules
    ]


def _emit_generic_rules(
    rules: tuple[GenericDispatchRule, ...],
) -> list[CPUPrefillGenericRule]:
    """Attach CPU ISA/thread surfaces to frozen common policy rules."""

    return [
        CPUPrefillGenericRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in rules
    ]


def _sealed_commitment(manifest: CPUPrefillSplitManifest) -> str:
    """Commit to the untouched CPU prefill sealed inventory before fitting."""

    payload = {
        "protocol": "cpu-native-vnni-prefill-sealed-witness-v8",
        "split_manifest_digest": manifest.digest(),
        "sealed_shapes": list(manifest.sealed_shapes),
        "sealed_m_values": list(manifest.sealed_m_values),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_partition(
    corpus: ObservationCorpus,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    *,
    sealed: bool,
    sealed_witness_plan: CPUPrefillSealedWitnessPlan | None = None,
    generic_refinement_plans: tuple[
        CPUPrefillGenericRefinementPlan, ...
    ] = (),
    development_lineage: CPUPrefillDevelopmentLineagePlan | None = None,
    candidate_expansion_plan: CPUPrefillCandidateExpansionPlan | None = None,
    burned_sealed_witness_plans: tuple[
        CPUPrefillSealedWitnessPlan, ...
    ] = (),
) -> ObservationCorpus:
    """Validate one complete, physically separate CPU prefill partition."""

    if sealed:
        if development_lineage is not None:
            raise ValueError("sealed CPU prefill evidence cannot use development lineage")
        if sealed_witness_plan is None:
            raise ValueError("sealed CPU prefill validation requires a witness plan")
        validate_sealed_complete(
            corpus,
            route_manifest,
            split_manifest,
            sealed_witness_plan,
        )
    else:
        if len(set(burned_sealed_witness_plans)) != len(
            burned_sealed_witness_plans
        ):
            raise ValueError("CPU prefill burned-seal plans are duplicated")
        if len({plan.digest() for plan in generic_refinement_plans}) != len(
            generic_refinement_plans
        ):
            raise ValueError("CPU prefill refinement plan lineage is duplicated")
        additional_records = tuple(
            record
            for plan in (
                *generic_refinement_plans,
                *burned_sealed_witness_plans,
            )
            for record in plan.records
        )
        additional_shapes = {
            record.shape_name for record in additional_records
        }
        additional_geometries: dict[str, tuple[int, int]] = {}
        for plan in generic_refinement_plans:
            for shape in plan.refinement_shapes:
                dimensions = (shape.n, shape.k)
                previous = additional_geometries.setdefault(
                    shape.name,
                    dimensions,
                )
                if previous != dimensions:
                    raise ValueError(
                        "CPU prefill refinement shape changed across plan lineage: "
                        f"{shape.name}={previous} != {dimensions}"
                    )
        for plan in burned_sealed_witness_plans:
            for record in plan.records:
                dimensions = (record.n, record.k)
                previous = additional_geometries.setdefault(
                    record.shape_name,
                    dimensions,
                )
                if previous != dimensions:
                    raise ValueError(
                        "CPU prefill burned-seal shape changed across "
                        f"transactions: {record.shape_name}="
                        f"{previous} != {dimensions}"
                    )
        if development_lineage is not None:
            additional_shapes.update(
                record.shape_name
                for record in development_lineage.source_records
                if record.shape_name
                not in split_manifest.shape_names(sealed=False)
            )
        if candidate_expansion_plan is not None:
            # The candidate-expansion plan authenticates exactly the immutable
            # source cells on which the new physical family was measured. A
            # later failed fit may add fresh boundary geometries, but those
            # rows must not be mistaken for part of the normalized expansion
            # cohort: they are an ordinary complete-registry transaction in
            # their own right.
            expansion_cells = _candidate_expansion_expected_cells(
                candidate_expansion_plan
            )
            expansion = ObservationCorpus(tuple(
                row
                for row in corpus
                if _candidate_expansion_cell(row) in expansion_cells
            ))
            validate_candidate_expansion_complete(
                expansion,
                candidate_expansion_plan,
            )
            additional_shapes.update(
                record.shape_name
                for record in candidate_expansion_plan.records
                if record.shape_name
                not in split_manifest.shape_names(sealed=False)
            )
        split_manifest.require_partition(
            corpus,
            sealed=False,
            additional_development_shapes=additional_shapes,
            additional_development_geometries=additional_geometries,
            burned_sealed_development_shapes={
                record.shape_name
                for plan in burned_sealed_witness_plans
                for record in plan.records
            },
        )
        if candidate_expansion_plan is not None:
            # Validate the union, not merely the original expansion cohort.
            # This requires every newly refined cell to contain every current
            # registry candidate and proves its exact declared launch
            # inventory. Historical refinement records already represented in
            # the expansion plan are harmless set duplicates here.
            lineage_records = (
                (
                    *development_lineage.source_records,
                    *development_lineage.increment_records,
                )
                if development_lineage is not None
                else ()
            )
            validate_complete(
                corpus,
                route_manifest,
                base_records=tuple(candidate_expansion_plan.records),
                additional_records=(*additional_records, *lineage_records),
            )
        elif development_lineage is None:
            validate_complete(
                corpus,
                route_manifest,
                expected_shapes=frozenset({
                    *split_manifest.shape_names(sealed=False),
                    *additional_shapes,
                }),
                additional_records=additional_records,
            )
        else:
            validate_complete(
                corpus,
                route_manifest,
                base_records=(
                    *development_lineage.source_records,
                    *development_lineage.increment_records,
                ),
                additional_records=additional_records,
            )
    return corpus


def freeze_cpu_prefill_policy(
    development: ObservationCorpus,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    serial_m1_policy_hash: str,
    profiler_feature_catalog: ProfilerFeatureCatalog | None,
    fit_cache_directory: Path | None = None,
    generic_refinement_plans: tuple[
        CPUPrefillGenericRefinementPlan, ...
    ] = (),
    development_lineage: CPUPrefillDevelopmentLineagePlan | None = None,
    *,
    candidate_expansion_plan: CPUPrefillCandidateExpansionPlan | None = None,
    burned_sealed_witness_plans: tuple[
        CPUPrefillSealedWitnessPlan, ...
    ] = (),
    max_leaves: int = DEFAULT_TREE_LEAVES,
    profiler_ablation_catalog_digest: str | None = None,
) -> FrozenPolicy:
    """Fit prefill rules from development timing and reviewed profiler input.

    ``profiler_ablation_catalog_digest`` is accepted only for a controlled
    development-fit experiment. It proves that complete profiler evidence was
    loaded and validated even though the learner deliberately did not receive
    its feature catalog. Frozen/certified production calls leave it unset.
    """

    if (
        profiler_ablation_catalog_digest is not None
        and profiler_feature_catalog is not None
    ):
        raise ValueError(
            "profiler ablation cannot also provide learner profiler features"
        )

    development = _require_partition(
        development,
        route_manifest,
        split_manifest,
        sealed=False,
        generic_refinement_plans=generic_refinement_plans,
        development_lineage=development_lineage,
        candidate_expansion_plan=candidate_expansion_plan,
        burned_sealed_witness_plans=burned_sealed_witness_plans,
    )
    generic_development = development.with_collapsed_aspect_domains()
    fit_cache = (
        PolicyFitCache(directory=fit_cache_directory)
        if fit_cache_directory is not None
        else None
    )
    frozen = freeze_policy(
        generic_development,
        sealed_commitment=_sealed_commitment(split_manifest),
        split_manifest_digest=split_manifest.digest(),
        serial_m1_hashes=_serial_hashes(
            generic_development,
            serial_m1_policy_hash,
        ),
        profiler_feature_catalog=profiler_feature_catalog,
        fit_cache=fit_cache,
        max_leaves=max_leaves,
        metadata={
            "backend": "cpu",
            "operation_kind": "NativeVNNIPrefillProjection",
            "semantic_contract": SemanticContract.VERIFIER_SERIAL_M1_BITWISE.value,
            "cpu_prefill_split_schema": split_manifest.schema_version,
            "cpu_prefill_split_digest": split_manifest.digest(),
            "generic_aspect_domains": "collapsed",
            "generic_max_leaves": max_leaves,
            **(
                {
                    "profiler_features_ablated": True,
                    "profiler_ablation_catalog_digest": (
                        profiler_ablation_catalog_digest
                    ),
                }
                if profiler_ablation_catalog_digest is not None
                else {}
            ),
        },
    )
    return FrozenPolicy(
        policy_ir=_retain_production_exact_overlays(frozen.policy_ir),
        promotion_diagnostics=frozen.promotion_diagnostics,
    )


def certify_cpu_prefill_policy(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    sealed: ObservationCorpus,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    sealed_witness_plan: CPUPrefillSealedWitnessPlan,
    serial_m1_policy_hash: str,
    *,
    require_promotable: bool = True,
    generic_refinement_plans: tuple[
        CPUPrefillGenericRefinementPlan, ...
    ] = (),
    development_lineage: CPUPrefillDevelopmentLineagePlan | None = None,
    candidate_expansion_plan: CPUPrefillCandidateExpansionPlan | None = None,
    burned_sealed_witness_plans: tuple[
        CPUPrefillSealedWitnessPlan, ...
    ] = (),
) -> CompiledPolicy:
    """Certify the immutable CPU prefill policy on sealed geometries."""

    if (
        sealed_witness_plan.frozen_generic_policy_digest
        != frozen.generic_digest
    ):
        raise ValueError(
            "sealed witness plan is bound to a different generic policy"
        )
    development = _require_partition(
        development,
        route_manifest,
        split_manifest,
        sealed=False,
        generic_refinement_plans=generic_refinement_plans,
        development_lineage=development_lineage,
        candidate_expansion_plan=candidate_expansion_plan,
        burned_sealed_witness_plans=burned_sealed_witness_plans,
    ).with_collapsed_aspect_domains()
    sealed = _require_partition(
        sealed,
        route_manifest,
        split_manifest,
        sealed=True,
        sealed_witness_plan=sealed_witness_plan,
    ).with_collapsed_aspect_domains()
    combined_hashes = {
        **_serial_hashes(development, serial_m1_policy_hash),
        **_serial_hashes(sealed, serial_m1_policy_hash),
    }
    compiled = certify_frozen_policy(
        frozen,
        development,
        sealed,
        serial_m1_hashes=combined_hashes,
        require_promotable=require_promotable,
    )
    return CompiledPolicy(
        policy_ir=_retain_production_exact_overlays(compiled.policy_ir),
        certification=compiled.certification,
    )


def validate_emitter_inputs(
    policy_ir: PolicyIR,
    entries: list[PrefillPolicyEntry],
    generic_rules: list[CPUPrefillGenericRule],
) -> None:
    """Prove CPU prefill emission consumes the exact common policy IR."""

    ir_exact = {
        (
            *_cpu_runtime_surface(entry.key.architecture_class),
            entry.key.runtime_codebook_id,
            entry.key.m,
            entry.key.aggregate_n,
            entry.key.k,
            entry.key.bundle_signature,
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    emitted_exact = {
        (
            entry.build_isa,
            entry.runtime_isa,
            entry.threads,
            entry.codebook,
            entry.m,
            entry.n,
            entry.k,
            entry.bundle_signature,
        ): entry.candidate_id
        for entry in entries
    }
    if emitted_exact != ir_exact:
        raise ValueError("CPU prefill exact emitter inputs disagree with common IR")
    if tuple(item.rule for item in generic_rules) != policy_ir.generic_rules:
        raise ValueError("CPU prefill generic emitter inputs disagree with common IR")


def validate_isa_regimes(corpus: ObservationCorpus) -> None:
    observed = {
        _cpu_runtime_surface(key.architecture_class)[:2]
        for key in corpus.runtime_keys()
    }
    required = {
        ("AVX2", "AVX2"),
        ("AVX512", "AVX2"),
        ("AVX512", "AVX512"),
    }
    if observed != required:
        raise ValueError(
            f"CPU prefill ISA matrix incomplete: missing={sorted(required-observed)}"
        )


def validate_complete(
    corpus: ObservationCorpus,
    route_manifest: CPUPrefillSerialRouteManifest,
    *,
    expected_shapes: frozenset[str] | None = None,
    base_records: tuple[CPUPrefillSourceTrainingRecord, ...] | None = None,
    additional_records: tuple[CPUPrefillSourceTrainingRecord, ...] = (),
) -> None:
    """Require the exact declared development plan and complete candidates.

    The base covering array and opened v3 evidence co-measure every source
    alias of each CPU runtime codebook.  Focused v4 CV densification instead
    measures one canonical alias for every runtime codebook because aliases
    share the same prepared family, packing ABI, and physical launch route.
    The exact ``expected`` inventory below enforces that distinction.  Reusing
    the sealed-holdout per-key alias validator here would incorrectly demand
    repeated measurements of physically identical focused launches.

    The separate all-format grouped sweep remains the native-byte correctness
    authority.  Fresh sealed economy evidence follows the frozen-leaf witness
    artifact and is validated by :func:`validate_sealed_complete` below.
    """

    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_prefill_registry(),
    )
    declared_base_records = (
        (
            *cpu_prefill_source_training_records(route_manifest),
            *cpu_prefill_development_refinement_source_training_records(
                route_manifest
            ),
        )
        if base_records is None
        else base_records
    )
    expected = {
        (
            record.source_format,
            record.shape_name,
            m,
            record.isa_regime,
        )
        for record in (
            *declared_base_records,
            *additional_records,
        )
        if expected_shapes is None or record.shape_name in expected_shapes
        for m in record.m_values
    }
    observed = {
        (
            row.source_format,
            row.shape_name,
            row.m,
            _isa_regime(row.architecture_class),
        )
        for row in corpus
    }
    if observed != expected:
        raise ValueError(
            "CPU prefill covering plan incomplete: "
            f"missing={sorted(expected-observed)[:3]} "
            f"unexpected={sorted(observed-expected)[:3]}"
        )
    validate_isa_regimes(corpus)


def validate_sealed_complete(
    corpus: ObservationCorpus,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    sealed_witness_plan: CPUPrefillSealedWitnessPlan,
) -> None:
    """Require the complete frozen-leaf witness plan and no undeclared cell.

    The witness artifact binds every frozen generic leaf to a fresh route-probed
    geometry without reading timing, expands every selected codebook to all
    aliases, and records every selected M/ISA cell. Exact equality with that
    artifact prevents a partial run from masquerading as a certificate while
    avoiding a Cartesian sweep of unused candidate geometries.
    """

    if sealed_witness_plan.split_manifest_digest != split_manifest.digest():
        raise ValueError("sealed witness plan disagrees with split manifest")
    if sealed_witness_plan.route_manifest_digest != route_manifest.digest():
        raise ValueError("sealed witness plan disagrees with C++ routes")
    require_canonical_alias_coverage(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_prefill_registry(),
    )
    records = sealed_witness_plan.records
    expected = {
        (record.source_format, record.shape_name, m, record.isa_regime)
        for record in records
        for m in record.m_values
    }
    observed = {
        (
            row.source_format,
            row.shape_name,
            row.m,
            _isa_regime(row.architecture_class),
        )
        for row in corpus
    }
    if observed != expected:
        raise ValueError(
            "CPU prefill sealed witness matrix incomplete: "
            f"missing={sorted(expected-observed)[:3]} "
            f"unexpected={sorted(observed-expected)[:3]}"
        )
    expected_shapes = {record.shape_name for record in records}
    witness_shapes = {
        witness.route.shape_name
        for witness in sealed_witness_plan.rule_witnesses
    }
    if expected_shapes != witness_shapes:
        raise ValueError("sealed witness launch and route shapes disagree")
    if {row.shape_name for row in corpus} != expected_shapes:
        raise ValueError("CPU prefill sealed witness shape inventory is incomplete")
    if {row.source_format for row in corpus} != {
        record.source_format for record in records
    }:
        raise ValueError("CPU prefill sealed witness alias inventory is incomplete")


def _isa_regime(architecture_class: str) -> str:
    """Return the canonical collection-plan name for one corpus ISA surface."""

    build, runtime, _ = _cpu_runtime_surface(architecture_class)
    regime = f"{build.lower()}-build.{runtime.lower()}-runtime"
    if regime not in ISA_REGIMES:
        raise ValueError(f"unknown CPU prefill ISA regime {regime}")
    return regime


def _candidate_expansion_expected_cells(
    plan: CPUPrefillCandidateExpansionPlan,
) -> frozenset[tuple[str, str, int, int, int, str, int]]:
    """Expand the authenticated record inventory into exact runtime cells."""

    for record in plan.records:
        expected_runtime = record.isa_regime.split(".", maxsplit=1)[1]
        expected_runtime = expected_runtime.removesuffix("-runtime")
        if record.runtime_isa != expected_runtime:
            raise ValueError(
                "CPU prefill candidate-expansion runtime ISA disagrees with "
                f"its regime: {record}"
            )
    expected = frozenset(
        (
            record.source_format,
            record.shape_name,
            record.n,
            record.k,
            m,
            record.isa_regime,
            record.threads,
        )
        for record in plan.records
        for m in record.m_values
    )
    if len(expected) != plan.selected_cell_count:
        raise ValueError(
            "CPU prefill candidate-expansion plan contains duplicate cells"
        )
    return expected


def _candidate_expansion_cell(
    row: NativeVNNIObservation,
) -> tuple[str, str, int, int, int, str, int]:
    """Return the plan-level identity for one common observation."""

    _, _, threads = _cpu_runtime_surface(row.architecture_class)
    return (
        row.source_format,
        row.shape_name,
        row.aggregate_n,
        row.k,
        row.m,
        _isa_regime(row.architecture_class),
        threads,
    )


def _select_candidate_expansion_source(
    source: ObservationCorpus,
    plan: CPUPrefillCandidateExpansionPlan,
) -> ObservationCorpus:
    """Authenticate and select the reviewed complete-matrix fit cohort.

    The immutable predecessor may contain exploratory and superseded refinement
    geometries. They remain durable evidence on disk, but they cannot enter a
    fit after a new candidate family is measured only on the reviewed
    development split: doing so would give CV different candidate universes at
    different shapes. The expansion plan therefore commits to every production
    overlay and systematic generic witness used by this fit, and this function
    returns exactly that authenticated source submatrix.
    """

    expected_cells = _candidate_expansion_expected_cells(plan)
    selected = ObservationCorpus(tuple(
        row
        for row in source
        if _candidate_expansion_cell(row) in expected_cells
    ))
    observed_cells = {
        _candidate_expansion_cell(row) for row in selected
    }
    if observed_cells != expected_cells:
        raise ValueError(
            "CPU prefill candidate-expansion source selection is incomplete: "
            f"missing={sorted(expected_cells-observed_cells)[:3]} "
            f"unexpected={sorted(observed_cells-expected_cells)[:3]}"
        )

    expected_candidates = frozenset(plan.source_candidate_ids)
    candidates_by_cell: dict[
        tuple[str, str, int, int, int, str, int], set[str]
    ] = {}
    for row in selected:
        candidates_by_cell.setdefault(
            _candidate_expansion_cell(row), set()
        ).add(row.candidate_id)
    malformed = [
        (cell, sorted(expected_candidates-candidates),
         sorted(candidates-expected_candidates))
        for cell, candidates in candidates_by_cell.items()
        if candidates != expected_candidates
    ]
    if malformed:
        raise ValueError(
            "CPU prefill candidate-expansion source candidate matrix is "
            f"incomplete: {malformed[:3]}"
        )
    if len(selected) != len(expected_cells) * len(expected_candidates):
        raise ValueError(
            "CPU prefill candidate-expansion source contains duplicate rows"
        )
    return selected


def validate_candidate_expansion_complete(
    corpus: ObservationCorpus,
    plan: CPUPrefillCandidateExpansionPlan,
) -> None:
    """Require one complete registry inventory at every reviewed fit cell.

    Exact overlays and generic CV must be derived from the same launchable
    candidate universe. Historical source cells outside the reviewed split are
    intentionally absent from this derived corpus rather than retained with a
    smaller candidate set or populated with synthetic measurements.
    """

    if plan.selection_policy not in {
        "all-immutable-source-cells-v1",
        "split-development-cells-v1",
    }:
        raise ValueError(
            "installable CPU prefill expansion requires an immutable "
            "development-cell selection"
        )
    expected_cells = _candidate_expansion_expected_cells(plan)
    candidates_by_cell: dict[
        tuple[str, str, int, int, int, str, int], set[str]
    ] = {}
    for row in corpus:
        candidates_by_cell.setdefault(
            _candidate_expansion_cell(row), set()
        ).add(row.candidate_id)
    observed_cells = set(candidates_by_cell)
    if len(observed_cells) != plan.selected_cell_count:
        raise ValueError(
            "CPU prefill candidate-expansion selected cell count disagrees with "
            f"its plan: expected={plan.selected_cell_count} "
            f"observed={len(observed_cells)}"
        )
    missing_selected = expected_cells.difference(observed_cells)
    if missing_selected:
        raise ValueError(
            "CPU prefill candidate-expansion selected cells are missing: "
            f"{sorted(missing_selected)[:3]}"
        )

    registry_candidates = frozenset({
        *plan.source_candidate_ids,
        *plan.expansion_candidate_ids,
    })
    current_registry_candidates = frozenset(
        entry.candidate_id
        for entry in cpu_native_vnni_prefill_registry().entries
    )
    if registry_candidates != current_registry_candidates:
        raise ValueError(
            "CPU prefill candidate-expansion plan does not cover the registry"
        )
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_prefill_registry(),
    )
    malformed = []
    for cell, observed_candidates in candidates_by_cell.items():
        if observed_candidates != registry_candidates:
            malformed.append((
                cell,
                sorted(registry_candidates.difference(observed_candidates)),
                sorted(observed_candidates.difference(registry_candidates)),
            ))
    if malformed:
        raise ValueError(
            "CPU prefill candidate-expansion candidate coverage is malformed: "
            f"{malformed[:3]}"
        )

    expected_row_count = plan.selected_cell_count * len(registry_candidates)
    if len(corpus) != expected_row_count:
        raise ValueError(
            "CPU prefill candidate-expansion corpus contains duplicate rows"
        )
    validate_isa_regimes(corpus)


def validate_total_policy(
    entries: list[PrefillPolicyEntry],
    generic_rules: list[CPUPrefillGenericRule],
    route_manifest: CPUPrefillSerialRouteManifest | None = None,
) -> None:
    """Prove every supported production cell has one generic dispatch rule.

    Exact overlays are additive specializations for measured model shapes. They
    must never make an otherwise incomplete generic policy appear total: an
    unseen shape with the same format, M, and arithmetic bundle still needs a
    learned dispatch decision. Consequently, ``entries`` provide only route
    consistency evidence when no typed C++ route manifest is available.
    """

    groups = {
        (item.build_isa, item.runtime_isa, item.threads)
        for item in generic_rules
    }
    observed_regimes = {
        f"{build.lower()}-build.{runtime.lower()}-runtime"
        for build, runtime, _ in groups
    }
    if observed_regimes != set(ISA_REGIMES):
        raise ValueError(
            "CPU prefill generated generic policy has incomplete ISA groups: "
            f"missing={sorted(set(ISA_REGIMES)-observed_regimes)}"
        )

    codebooks = sorted({
        cell.runtime_codebook
        for cell in cpu_prefill_runtime_training_cells()
    })
    bundle_by_geometry: dict[tuple[int, int, int], str] = {}
    for entry in entries:
        geometry = (entry.codebook, entry.n, entry.k)
        previous = bundle_by_geometry.setdefault(
            geometry,
            entry.bundle_signature,
        )
        if previous != entry.bundle_signature:
            raise ValueError(
                "CPU prefill serial K regime disagrees across ISA/M evidence: "
                f"{geometry}: {previous} != {entry.bundle_signature}"
            )
        if route_manifest is not None:
            regime = (
                f"{entry.build_isa.lower()}-build."
                f"{entry.runtime_isa.lower()}-runtime"
            )
            for shape_name in entry.shape_names:
                route = route_manifest.route_for(
                    entry.codebook,
                    shape_name,
                    regime,
                )
                if route.threads != entry.threads:
                    raise ValueError(
                        "CPU prefill measured policy thread width disagrees "
                        f"with route manifest for {shape_name}"
                    )
                if route.bundle_signature != entry.bundle_signature:
                    raise ValueError(
                        "CPU prefill measured policy bundle disagrees with "
                        f"the production route manifest for {shape_name}"
                    )
    missing: list[tuple[object, ...]] = []
    malformed: list[tuple[object, ...]] = []
    for build, runtime, threads in sorted(groups):
        for codebook in codebooks:
            for m in CPU_PREFILL_M_BUCKETS:
                for bundle_signature in sorted(CPU_PREFILL_ROUTE_BUNDLES):
                    key = (
                        build,
                        runtime,
                        threads,
                        codebook,
                        m,
                        bundle_signature,
                    )
                    domain_rules = [
                        item.rule
                        for item in generic_rules
                        if (
                            item.build_isa == build
                            and item.runtime_isa == runtime
                            and item.threads == threads
                            and item.rule.domain.runtime_codebook_id == codebook
                            and item.rule.domain.m == m
                            and item.rule.domain.bundle_signature
                            == bundle_signature
                        )
                    ]
                    if not domain_rules:
                        missing.append(key)
                        continue
                    domains = {rule.domain for rule in domain_rules}
                    if (
                        len(domains) != 1
                        or not next(iter(domains)).all_aspects
                    ):
                        malformed.append((*key, "aspect-domain"))
                        continue
                    try:
                        validate_generic_rule_partition(domain_rules)
                    except ValueError as error:
                        malformed.append((*key, str(error)))
    if missing or malformed:
        raise ValueError(
            "CPU prefill generated generic policy is not total: "
            f"missing={missing[:3]} malformed={malformed[:3]}"
        )


def _pack_key(codebook: int, m: int, n: int, k: int) -> int:
    """Pack 8-bit codebook, 16-bit M, and 20-bit N/K dimensions."""

    if m > 0xFFFF or n > 0xFFFFF or k > 0xFFFFF:
        raise ValueError(f"CPU prefill key exceeds ABI: M={m} N={n} K={k}")
    return (
        ((codebook & 0xFF) << 56)
        | ((m & 0xFFFF) << 40)
        | ((k & 0xFFFFF) << 20)
        | (n & 0xFFFFF)
    )


def generate_include(
    entries: list[PrefillPolicyEntry],
    generic_rules: list[CPUPrefillGenericRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> str:
    """Render exact and generic decode-equivalent decisions into a selector."""

    lines = [
        "// Auto-generated by analyze_cpu_native_vnni_prefill_trainer.py. DO NOT EDIT.",
        "// Every winner inherits the serial-M1 reduction tree byte exactly.",
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "#define LLAMINAR_CPU_NVNNI_PREFILL_POLICY_ABI 2",
        (
            "#define LLAMINAR_CPU_NVNNI_PREFILL_POLICY_CERTIFIED "
            f"{1 if certification is not None else 0}"
        ),
        "",
        "#include <cstdint>",
        "",
        "namespace llaminar2::cpu::native_vnni::generated",
        "{",
        "enum class CPUNativeVNNIPrefillPolicy : uint8_t",
        "{",
        "    RowChunkGrid = 0,",
        "    TwoRowNbc1 = 1,",
        "    TwoRowNbc2 = 2,",
        "    TwoRowNbc4 = 3,",
        "    TwoRowNbc8 = 4,",
        "    TwoRowNbc16 = 5,",
        "    KPartPairwise = 6,",
        "    KPartWideRows = 7,",
        "    TwoRowPairGridNbc1 = 8,",
        "    TwoRowPairGridNbc2 = 9,",
        "    TwoRowPairGridNbc4 = 10,",
        "    TwoRowPairGridNbc8 = 11,",
        "    TwoRowPairGridNbc16 = 12,",
        "};",
        "",
        "enum class CPUNativeVNNIPrefillBuildISA : uint8_t { AVX2 = 0, AVX512 = 1 };",
        "enum class CPUNativeVNNIPrefillRuntimeISA : uint8_t { AVX2 = 0, AVX512 = 1 };",
        "",
        "inline constexpr int bucketCPUNativeVNNIPrefillM(int m, int n, int k)",
        "{",
        "    (void)n;",
        "    (void)k;",
    ]
    for bucket in CPU_PREFILL_M_BUCKETS:
        lines.extend([
            f"    if (m <= {bucket})",
            f"        return {bucket};",
        ])
    lines.extend([
        f"    return {CPU_PREFILL_M_BUCKETS[-1]};",
        "}",
        "",
        "inline constexpr uint64_t packCPUNativeVNNIPrefillPolicyKey(",
        "    uint8_t codebook, int policy_m, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(codebook) << 56) |",
        "           (static_cast<uint64_t>(policy_m & 0xFFFF) << 40) |",
        "           (static_cast<uint64_t>(k & 0xFFFFF) << 20) |",
        "           static_cast<uint64_t>(n & 0xFFFFF);",
        "}",
        "",
        "inline bool selectCPUNativeVNNIPrefillGeneratedPolicy(",
        "    CPUNativeVNNIPrefillBuildISA build_isa,",
        "    CPUNativeVNNIPrefillRuntimeISA runtime_isa, int threads,",
        "    uint8_t codebook, int m, int n, int k, bool serial_kpart,",
        "    CPUNativeVNNIPrefillPolicy &policy)",
        "{",
        "    const int policy_m = bucketCPUNativeVNNIPrefillM(m, n, k);",
        "    const uint64_t key =",
        "        packCPUNativeVNNIPrefillPolicyKey(codebook, policy_m, n, k);",
    ])
    if certification is not None:
        lines[5:5] = [
            f"// Common policy digest: {policy_digest}",
            (
                "// Frozen generic policy digest: "
                f"{certification.frozen_generic_policy_digest}"
            ),
            (
                "// Sealed generic certificate: coverage="
                f"{certification.covered_cell_count}/"
                f"{certification.required_cell_count} max-regret="
                f"{certification.max_observed_regret:.6%} "
                "max-simultaneous-ucb="
                f"{certification.max_simultaneous_95pct_upper_regret:.6%}"
            ),
        ]

    group_keys = sorted({
        (entry.build_isa, entry.runtime_isa, entry.threads)
        for entry in entries
    } | {
        (item.build_isa, item.runtime_isa, item.threads)
        for item in generic_rules
    })
    for build_isa, runtime_isa, threads in group_keys:
        group_entries = [
            entry for entry in entries
            if (entry.build_isa, entry.runtime_isa, entry.threads)
            == (build_isa, runtime_isa, threads)
        ]
        group_rules = [
            item for item in generic_rules
            if (item.build_isa, item.runtime_isa, item.threads)
            == (build_isa, runtime_isa, threads)
        ]
        lines.extend([
            f"    if (build_isa == CPUNativeVNNIPrefillBuildISA::{build_isa} &&",
            f"        runtime_isa == CPUNativeVNNIPrefillRuntimeISA::{runtime_isa} &&",
            f"        threads == {threads})",
            "    {",
            "        if (serial_kpart && m < 3)",
            "        {",
            "            policy = CPUNativeVNNIPrefillPolicy::KPartPairwise;",
            "            return true;",
            "        }",
            "        switch (key)",
            "        {",
        ])
        for entry in sorted(group_entries):
            shape_names = ", ".join(entry.shape_names)
            expected_serial_kpart = (
                "true"
                if _bundle_uses_serial_kpart(entry.bundle_signature)
                else "false"
            )
            lines.extend([
                f"        case 0x{_pack_key(entry.codebook, entry.m, entry.n, entry.k):016x}ULL:",
                f"            // {shape_names} {entry.candidate_id}",
                f"            if (serial_kpart != {expected_serial_kpart})",
                "                break;",
                f"            policy = CPUNativeVNNIPrefillPolicy::{entry.policy};",
                "            return true;",
            ])
        lines.extend([
            "        default:",
            "            break;",
            "        }",
        ])
        if group_rules:
            lines.extend([
                "        const long long work_items =",
                "            static_cast<long long>(n) * static_cast<long long>(k);",
            ])
        for item in sorted(
            group_rules,
            key=lambda value: (
                value.rule.domain.runtime_codebook_id,
                value.rule.domain.m,
                generic_rule_sort_key(value.rule),
            ),
        ):
            rule = item.rule
            conditions = [
                f"codebook == {rule.domain.runtime_codebook_id}",
                f"policy_m == {rule.domain.m}",
                (
                    "serial_kpart"
                    if _bundle_uses_serial_kpart(rule.domain.bundle_signature)
                    else "!serial_kpart"
                ),
                *(
                    ()
                    if rule.domain.all_aspects
                    else (aspect_condition(rule.domain.aspect_bucket),)
                ),
                *(predicate_condition(item) for item in rule.predicates),
            ]
            lines.extend(render_if_header(conditions, indent="        "))
            lines.extend([
                "        {",
                "            policy = CPUNativeVNNIPrefillPolicy::"
                f"{_candidate_policy(rule.candidate_id)};",
                "            return true;",
                "        }",
            ])
        lines.append("    }")
    lines.extend([
        "    return false;",
        "}",
        "",
        "} // namespace llaminar2::cpu::native_vnni::generated",
        "",
    ])
    return "\n".join(lines)


def write_summary(path: Path, entries: list[PrefillPolicyEntry]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "build_isa", "runtime_isa", "threads", "execution_codebook",
            "m", "n", "k", "policy", "candidate_id", "shape",
            "bundle_signature", "max_surface_regret", "max_cv",
        ])
        for entry in sorted(entries):
            writer.writerow([
                entry.build_isa, entry.runtime_isa, entry.threads,
                entry.codebook, entry.m, entry.n, entry.k, entry.policy,
                entry.candidate_id, ";".join(entry.shape_names),
                entry.bundle_signature,
                f"{entry.max_surface_regret:.9f}", f"{entry.max_cv:.9f}",
            ])


def _context_from_args(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    *,
    run_id: str | None = None,
    authenticated_corpus_id: str | None = None,
    provenance_prefix: str = "",
) -> CPUPrefillAdapterContext:
    """Build one adapter context from an explicitly typed provenance role.

    Development and sealed evidence can be measured by different binaries.
    Certification therefore supplies the historical development provenance on
    the ordinary arguments and the fresh holdout provenance on ``sealed_*``
    arguments.  Selecting the role here keeps the two identities separate all
    the way into CSV authentication; there is intentionally no fallback from
    missing sealed provenance to development provenance.
    """

    argument_prefix = f"{provenance_prefix}_" if provenance_prefix else ""

    def provenance(name: str) -> str:
        return getattr(args, f"{argument_prefix}{name}")

    corpus_id = authenticated_corpus_id or raw_corpus_id((
        *inputs,
        *timing_sidecars,
    ))
    if (
        not corpus_id.startswith("sha256:")
        or len(corpus_id) != 71
        or any(
            character not in "0123456789abcdef"
            for character in corpus_id.removeprefix("sha256:")
        )
    ):
        raise ValueError("authenticated raw corpus identity is not SHA-256")
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CPUPrefillAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return CPUPrefillAdapterContext(
        profile=profile,
        run_id=run_id or args.run_id,
        corpus_id=corpus_id,
        git_revision=provenance("git_revision"),
        build_id=provenance("build_id"),
        compiler_id=provenance("compiler_id"),
        architecture_class=provenance("architecture_class"),
        device_name=provenance("device_name"),
        driver_runtime=provenance("driver_runtime"),
        serial_m1_policy_hash=provenance("serial_m1_policy_hash"),
        raw_timing_sidecar_retained=bool(timing_sidecars),
    )


_PARALLEL_ADAPTATION_MIN_TIMING_BYTES = 64 * 1024 * 1024
_PARALLEL_CHECKPOINT_MIN_BYTES = 64 * 1024 * 1024
_DEFAULT_CHECKPOINT_WORKERS = 8


def _read_observation_checkpoint_range(
    task: tuple[Path, int, int, tuple[str, ...]],
) -> tuple[NativeVNNIObservation, ...]:
    """Parse one newline-aligned byte range of a generated checkpoint CSV.

    Common observation checkpoints are emitted by ``write_observation_csv``.
    Every record is one physical UTF-8 line because JSON-valued columns are
    compact and observation strings reject line breaks. A worker still checks
    the preceding byte before discarding a partial first line, so adjacent
    ranges own every record exactly once.
    """

    path, start, end, fieldnames = task
    observations = []
    with path.open("rb") as handle:
        if start > 0:
            handle.seek(start - 1)
            begins_at_record = handle.read(1) == b"\n"
        else:
            begins_at_record = True
        handle.seek(start)
        if not begins_at_record:
            handle.readline()
        while handle.tell() < end:
            byte_offset = handle.tell()
            encoded = handle.readline()
            if not encoded:
                break
            try:
                line = encoded.decode("utf-8")
                raw = next(csv.DictReader(
                    (line,),
                    fieldnames=fieldnames,
                ))
                observations.append(NativeVNNIObservation.from_mapping(raw))
            except (
                StopIteration,
                UnicodeDecodeError,
                TypeError,
                ValueError,
                csv.Error,
            ) as exc:
                raise ValueError(
                    f"{path}:byte {byte_offset}: {exc}"
                ) from exc
    return tuple(observations)


def _read_observation_checkpoint(
    path: Path,
    *,
    requested_workers: int | None = None,
    minimum_parallel_bytes: int = _PARALLEL_CHECKPOINT_MIN_BYTES,
) -> ObservationCorpus:
    """Read one checkpoint in parallel while preserving exact source order."""

    path = Path(path)
    if not path.exists():
        # Preserve the serial reader's error and test-interception behavior for
        # synthetic paths. Real checkpoints take the size-gated parallel path.
        return read_observation_csv((path,))
    if requested_workers is None:
        requested_workers = int(os.environ.get(
            "LLAMINAR_NATIVE_VNNI_ADAPTATION_WORKERS",
            str(min(_DEFAULT_CHECKPOINT_WORKERS, os.cpu_count() or 1)),
        ))
    if requested_workers < 1:
        raise ValueError("checkpoint adaptation worker count must be positive")
    if (
        requested_workers == 1
        or path.stat().st_size < minimum_parallel_bytes
        or os.environ.get(
            "LLAMINAR_NATIVE_VNNI_DISABLE_PARALLEL_ADAPTATION"
        ) == "1"
    ):
        return read_observation_csv((path,))

    with path.open("rb") as handle:
        header = handle.readline()
        data_start = handle.tell()
    try:
        fieldnames = tuple(next(csv.reader((header.decode("utf-8"),))))
    except (StopIteration, UnicodeDecodeError, csv.Error) as exc:
        raise ValueError(f"{path}: invalid common observation header") from exc
    missing = (
        set(OBSERVATION_COLUMNS).difference(fieldnames)
        - OPTIONAL_OBSERVATION_COLUMNS
    )
    unexpected = set(fieldnames).difference(OBSERVATION_COLUMNS)
    if missing or unexpected:
        raise ValueError(
            f"{path}: common observation header mismatch: "
            f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
        )

    file_size = path.stat().st_size
    data_bytes = max(0, file_size - data_start)
    if data_bytes == 0:
        return read_observation_csv((path,))
    worker_count = min(requested_workers, max(1, data_bytes))
    chunk_bytes = max(1, (data_bytes + worker_count - 1) // worker_count)
    tasks = tuple(
        (
            path,
            data_start + index * chunk_bytes,
            min(file_size, data_start + (index + 1) * chunk_bytes),
            fieldnames,
        )
        for index in range(worker_count)
        if data_start + index * chunk_bytes < file_size
    )
    with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
        partitions = tuple(executor.map(
            _read_observation_checkpoint_range,
            tasks,
        ))
    observations = tuple(
        observation
        for partition in partitions
        for observation in partition
    )
    return ObservationCorpus._from_validated(observations)


def _adapt_cpu_prefill_worker(
    task: tuple[
        tuple[Path, ...],
        CPUPrefillAdapterContext,
        tuple[Path, ...],
    ],
) -> ObservationCorpus:
    """Adapt one independent raw timing transaction in a worker process.

    Production timing sidecars contain millions of samples.  Parsing and
    validating them is Python CPU work, so threads remain serialized by the
    interpreter lock.  A process worker gives each independent transaction a
    real core and, importantly, releases its large temporary timing index when
    the worker exits; only the compact canonical observation corpus crosses
    the process boundary.
    """

    inputs, context, timing_sidecars = task
    return adapt_cpu_prefill_csv(
        inputs,
        context,
        timing_sidecars=timing_sidecars,
    )


def _candidate_expansion_adaptation_is_large(
    source_timing: tuple[Path, ...],
    expansion_timing: Path,
) -> bool:
    """Return whether two-process adaptation amortizes worker startup cost."""

    if os.environ.get("LLAMINAR_NATIVE_VNNI_DISABLE_PARALLEL_ADAPTATION") == "1":
        return False
    timing_bytes = sum(path.stat().st_size for path in source_timing)
    timing_bytes += expansion_timing.stat().st_size
    return timing_bytes >= _PARALLEL_ADAPTATION_MIN_TIMING_BYTES


def _validate_adapted_development_checkpoint(
    corpus: ObservationCorpus,
    context: CPUPrefillAdapterContext,
    *,
    allow_corpus_rebase: bool = False,
) -> None:
    """Bind a compact development checkpoint to its raw timing transaction.

    A common-observation CSV saves repeated multi-gigabyte sidecar parsing, but
    it is authoritative only for the exact raw inputs named by ``context``.
    Every adapted row therefore has to retain the run identity and corpus hash
    that the production adapter would derive from those raw aggregate/timing
    paths. CPU ISA suffixes are adapter-owned refinements of the common build
    and architecture identities, so they are checked as strict prefixes.
    """

    context.validate()
    if not corpus.observations:
        raise ValueError("adapted CPU prefill development checkpoint is empty")
    checkpoint_corpus_id = corpus.observations[0].corpus_id
    if allow_corpus_rebase and not checkpoint_corpus_id.startswith("sha256:"):
        raise ValueError(
            "adapted CPU prefill development checkpoint has no corpus identity"
        )
    exact_fields = {
        "run_id": context.run_id,
        "corpus_id": (
            checkpoint_corpus_id if allow_corpus_rebase else context.corpus_id
        ),
        "git_revision": context.git_revision,
        "compiler_id": context.compiler_id,
        "device_name": context.device_name,
        "driver_runtime": context.driver_runtime,
        "serial_m1_policy_hash": context.serial_m1_policy_hash,
    }
    for row in corpus:
        for field, expected in exact_fields.items():
            if getattr(row, field) != expected:
                raise ValueError(
                    "adapted CPU prefill development checkpoint belongs to "
                    f"another transaction: {field}"
                )
        if not row.build_id.startswith(f"{context.build_id}|cpu_isa="):
            raise ValueError(
                "adapted CPU prefill development checkpoint belongs to "
                "another build"
            )
        if not row.architecture_class.startswith(
            f"{context.architecture_class}|build="
        ):
            raise ValueError(
                "adapted CPU prefill development checkpoint belongs to "
                "another architecture"
            )


def _common_checkpoint_requires_write(
    common_observations: Path | None,
    adapted_full_development_input: Path | None,
) -> bool:
    """Return whether development fit must publish a checkpoint CSV.

    A full-development checkpoint is validated against the complete raw corpus
    before fitting. Rewriting that same path after a read-only fit adds no
    evidence and can serialize hundreds of megabytes. A different output path
    still receives the normal deterministic checkpoint publication.
    """

    return (
        common_observations is not None
        and (
            adapted_full_development_input is None
            or common_observations.resolve()
            != adapted_full_development_input.resolve()
        )
    )


def _rebind_adapted_development_checkpoint(
    corpus: ObservationCorpus,
    context: CPUPrefillAdapterContext,
) -> ObservationCorpus:
    """Move an authenticated prefix checkpoint into a larger transaction.

    The caller must validate ``corpus`` against the exact historical raw
    prefix before invoking this helper.  Only run-level provenance changes
    when a newly adapted suffix is appended; candidate identity, correctness,
    timing, ISA suffixes, and every measured value remain byte-for-byte the
    historical evidence.  Rebinding lets an hour-scale corpus grow
    transactionally without reparsing its already authenticated timing
    sidecars.
    """

    return ObservationCorpus(tuple(
        replace(
            row,
            run_id=context.run_id,
            corpus_id=context.corpus_id,
            git_revision=context.git_revision,
            compiler_id=context.compiler_id,
            device_name=context.device_name,
            driver_runtime=context.driver_runtime,
            serial_m1_policy_hash=context.serial_m1_policy_hash,
        )
        for row in corpus
    ))


def _validate_checkpoint_candidate_expansion(
    checkpoint: ObservationCorpus,
    plan: CPUPrefillCandidateExpansionPlan | None,
) -> None:
    """Reauthenticate the normalized candidate-expansion cohort, if any."""

    if plan is None:
        return
    expansion_cells = _candidate_expansion_expected_cells(plan)
    normalized_expansion = ObservationCorpus(tuple(
        row
        for row in checkpoint
        if _candidate_expansion_cell(row) in expansion_cells
    ))
    validate_normalized_cpu_prefill_candidate_expansion(
        normalized_expansion,
        plan,
    )


def _adapt_development_with_candidate_expansion(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    context: CPUPrefillAdapterContext,
    generic_refinement_plans: tuple[
        CPUPrefillGenericRefinementPlan, ...
    ] = (),
) -> tuple[ObservationCorpus, CPUPrefillCandidateExpansionPlan | None]:
    """Adapt immutable source evidence plus one anchored candidate cohort.

    Candidate expansion deliberately remains a separate raw aggregate.  Its
    complete-round cadence has a different candidate inventory from the source
    run, so concatenating the CSV bytes would falsely claim that all candidates
    were interleaved in one timing cohort.  Each side is first validated under
    the production adapter, then the expansion module scales only the new
    candidates through its contemporaneous anchor.
    """

    full_checkpoint = getattr(
        args,
        "adapted_full_development_input",
        None,
    )
    rebase_checkpoint = getattr(
        args,
        "adapted_full_development_rebase_input",
        None,
    )
    prefix_checkpoint = getattr(
        args,
        "adapted_development_prefix_input",
        None,
    )
    prefix_count = getattr(
        args,
        "adapted_development_prefix_count",
        None,
    )
    plan = None
    if args.candidate_expansion_plan_json is None:
        if full_checkpoint is not None:
            checkpoint = _read_observation_checkpoint(full_checkpoint)
            _validate_adapted_development_checkpoint(checkpoint, context)
            return checkpoint, None
    elif not inputs or len(inputs) != len(timing_sidecars):
        raise ValueError(
            "CPU prefill candidate expansion requires one timing sidecar per "
            "development aggregate"
        )
    else:
        # Authenticate all immutable plan and file identities before spending
        # CPU time adapting any transaction. A compact checkpoint is immutable
        # fitted evidence, so current implementation drift is a collection
        # concern rather than a reason to expire historical measurements.
        plan = read_cpu_prefill_candidate_expansion_plan(
            args.candidate_expansion_plan_json,
            source_aggregate=inputs[0],
            source_timing=timing_sidecars[0],
            collection_build_digest=None,
            authenticate_current_implementation=(
                args.adapted_development_input is None
                and full_checkpoint is None
                and rebase_checkpoint is None
                and prefix_checkpoint is None
            ),
        )
    if rebase_checkpoint is not None:
        checkpoint = _read_observation_checkpoint(rebase_checkpoint)
        _validate_adapted_development_checkpoint(
            checkpoint,
            context,
            allow_corpus_rebase=True,
        )
        _validate_checkpoint_candidate_expansion(checkpoint, plan)
        return _rebind_adapted_development_checkpoint(checkpoint, context), plan
    if full_checkpoint is not None:
        checkpoint = _read_observation_checkpoint(full_checkpoint)
        _validate_adapted_development_checkpoint(checkpoint, context)
        _validate_checkpoint_candidate_expansion(checkpoint, plan)
        return checkpoint, plan

    if prefix_checkpoint is not None:
        checkpoint = _read_observation_checkpoint(prefix_checkpoint)
        checkpoint_corpus_ids = {row.corpus_id for row in checkpoint}
        if checkpoint_corpus_ids == {context.corpus_id}:
            # Idempotent resume: the previous invocation published the full
            # checkpoint before a later stage was interrupted.
            _validate_adapted_development_checkpoint(checkpoint, context)
            _validate_checkpoint_candidate_expansion(checkpoint, plan)
            return checkpoint, plan
        if prefix_count is None:
            # Refinement plans are derived from the learner's generic corpus,
            # where shape aliases sharing one runtime geometry have already
            # been collapsed. Match that exact authored view rather than the
            # full exact-overlay checkpoint digest.
            checkpoint_digest = (
                checkpoint.with_collapsed_aspect_domains().digest()
            )
            matching_plan_indexes = tuple(
                index
                for index, refinement_plan in enumerate(
                    generic_refinement_plans
                )
                if refinement_plan.source_development_corpus_digest
                == checkpoint_digest
            )
            if len(matching_plan_indexes) != 1:
                raise ValueError(
                    "adapted CPU prefill prefix checkpoint is not the "
                    "authenticated source of exactly one declared refinement "
                    "plan"
                )
            remaining_refinement_count = (
                len(generic_refinement_plans) - matching_plan_indexes[0]
            )
            prefix_count = len(inputs) - remaining_refinement_count
        if not 0 < prefix_count < len(inputs):
            raise ValueError(
                "adapted CPU prefill prefix checkpoint requires a prefix "
                "count smaller than the complete development input count"
            )
        prefix_inputs = inputs[:prefix_count]
        prefix_timing = timing_sidecars[:prefix_count]
        prefix_context_inputs, prefix_context_timing = _development_context_paths(
            args,
            prefix_inputs,
            prefix_timing,
        )
        prefix_context = _context_from_args(
            args,
            prefix_context_inputs,
            prefix_context_timing,
        )
        _validate_adapted_development_checkpoint(checkpoint, prefix_context)
        _validate_checkpoint_candidate_expansion(checkpoint, plan)
        suffix = adapt_cpu_prefill_csv(
            inputs[prefix_count:],
            context,
            timing_sidecars=timing_sidecars[prefix_count:],
        )
        rebound = _rebind_adapted_development_checkpoint(checkpoint, context)
        return ObservationCorpus((
            *rebound.observations,
            *suffix.observations,
        )), plan

    if plan is None:
        return (
            adapt_cpu_prefill_csv(
                inputs,
                context,
                timing_sidecars=timing_sidecars,
            ),
            None,
        )

    additive_inputs = inputs[1:]
    additive_timing = timing_sidecars[1:]
    if args.adapted_development_input is not None:
        normalized = read_observation_csv((args.adapted_development_input,))
        validate_normalized_cpu_prefill_candidate_expansion(normalized, plan)
    else:
        expansion_context = replace(context, anchored_candidate_expansion=True)
        source_task = ((inputs[0],), context, (timing_sidecars[0],))
        expansion_task = (
            (args.candidate_expansion_input,),
            expansion_context,
            (args.candidate_expansion_timing_sidecar,),
        )
        if _candidate_expansion_adaptation_is_large(
            (timing_sidecars[0],),
            args.candidate_expansion_timing_sidecar,
        ):
            # The source and expansion are independent raw transactions. Two
            # workers expose their available adaptation parallelism without
            # duplicating either corpus or oversubscribing later tree search.
            with ProcessPoolExecutor(max_workers=2) as executor:
                source_future = executor.submit(
                    _adapt_cpu_prefill_worker,
                    source_task,
                )
                expansion_future = executor.submit(
                    _adapt_cpu_prefill_worker,
                    expansion_task,
                )
                source = source_future.result()
                expansion = expansion_future.result()
        else:
            source = _adapt_cpu_prefill_worker(source_task)
            expansion = _adapt_cpu_prefill_worker(expansion_task)
        selected_source = _select_candidate_expansion_source(source, plan)
        normalized = normalize_cpu_prefill_candidate_expansion(
            selected_source,
            expansion,
            plan,
        )

    if not additive_inputs:
        return normalized, plan

    # Every post-expansion refinement launch measures the complete current
    # registry in one ordinary timing cohort. Keep it separate from anchor
    # normalization, then append its canonical observations to the immutable
    # normalized base. The plan/partition validator proves that no undeclared
    # geometry or partial candidate matrix entered the fit.
    additive = adapt_cpu_prefill_csv(
        additive_inputs,
        context,
        timing_sidecars=additive_timing,
    )
    return ObservationCorpus((
        *normalized.observations,
        *additive.observations,
    )), plan


def _development_context_paths(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Include additive raw evidence in the run-level corpus identity."""

    if args.candidate_expansion_plan_json is None:
        return inputs, timing_sidecars
    return (
        (*inputs, args.candidate_expansion_input),
        (*timing_sidecars, args.candidate_expansion_timing_sidecar),
    )


def _burned_seal_context(
    transaction: CPUPrefillBurnedSealTransaction,
    profile: MeasurementProfile,
) -> CPUPrefillAdapterContext:
    """Reconstruct the independent adapter identity of one failed seal."""

    provenance = transaction.provenance
    aggregate = transaction.aggregate.resolve(transaction.path)
    timing = transaction.timing.resolve(transaction.path)
    return CPUPrefillAdapterContext(
        profile=profile,
        run_id=provenance.run_id,
        corpus_id=raw_corpus_id((aggregate, timing)),
        git_revision=provenance.git_revision,
        build_id=provenance.build_id,
        compiler_id=provenance.compiler_id,
        architecture_class=provenance.architecture_class,
        device_name=provenance.device_name,
        driver_runtime=provenance.driver_runtime,
        serial_m1_policy_hash=provenance.serial_m1_policy_hash,
        raw_timing_sidecar_retained=True,
    )


def _checkpoint_candidate_expansion_plan(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
) -> CPUPrefillCandidateExpansionPlan | None:
    """Authenticate the expansion plan represented by a compact checkpoint."""

    if args.candidate_expansion_plan_json is None:
        return None
    if not inputs or len(inputs) != len(timing_sidecars):
        raise ValueError(
            "CPU prefill checkpoint requires one timing sidecar per "
            "development aggregate"
        )
    return read_cpu_prefill_candidate_expansion_plan(
        args.candidate_expansion_plan_json,
        source_aggregate=inputs[0],
        source_timing=timing_sidecars[0],
        collection_build_digest=None,
        authenticate_current_implementation=False,
    )


def _validate_burned_seal_checkpoint_partition(
    checkpoint: ObservationCorpus,
    transaction: CPUPrefillBurnedSealTransaction,
    context: CPUPrefillAdapterContext,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
) -> tuple[ObservationCorpus, CPUPrefillSealedWitnessPlan]:
    """Authenticate one original-provenance partition of a mixed checkpoint."""

    partition = ObservationCorpus(tuple(
        row for row in checkpoint if row.corpus_id == context.corpus_id
    ))
    _validate_adapted_development_checkpoint(partition, context)
    plan = read_cpu_prefill_sealed_witness_plan(
        transaction.witness_plan.resolve(transaction.path)
    )
    _require_partition(
        partition,
        route_manifest,
        split_manifest,
        sealed=True,
        sealed_witness_plan=plan,
    )
    return partition, plan


def _adapt_development_with_burned_seals(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    context: CPUPrefillAdapterContext,
    route_manifest: CPUPrefillSerialRouteManifest,
    split_manifest: CPUPrefillSplitManifest,
    generic_refinement_plans: tuple[
        CPUPrefillGenericRefinementPlan, ...
    ] = (),
) -> tuple[
    ObservationCorpus,
    CPUPrefillCandidateExpansionPlan | None,
    tuple[CPUPrefillSealedWitnessPlan, ...],
]:
    """Compose base development and failed seals without provenance relabeling.

    A complete mixed checkpoint retains each source transaction's own corpus
    ID and build fields.  It is validated by partitioning on those authenticated
    IDs, never by pretending the composition was measured in one process.  On
    the first replay, an existing base-only checkpoint can be supplied through
    ``--adapted-development-before-burned-seals-input``; only the failed seals
    are then adapted from their retained raw timing sidecars.
    """

    transactions: tuple[CPUPrefillBurnedSealTransaction, ...] = getattr(
        args,
        "burned_sealed_development_transactions",
        (),
    )
    if not transactions:
        corpus, candidate_plan = _adapt_development_with_candidate_expansion(
            args,
            inputs,
            timing_sidecars,
            context,
            generic_refinement_plans,
        )
        return corpus, candidate_plan, ()

    profile = MeasurementProfile(args.profile)
    contexts = tuple(
        _burned_seal_context(transaction, profile)
        for transaction in transactions
    )
    if len({item.corpus_id for item in contexts}) != len(contexts):
        raise ValueError("CPU prefill burned seals repeat one raw transaction")
    if any(
        item.serial_m1_policy_hash != context.serial_m1_policy_hash
        for item in contexts
    ):
        raise ValueError(
            "CPU prefill burned seal changed the production serial M=1 policy"
        )

    checkpoint = None
    complete_checkpoint_path = args.adapted_full_development_input
    before_burned = args.adapted_development_before_burned_seals_input
    if complete_checkpoint_path is None and before_burned is not None:
        candidate_checkpoint = _read_observation_checkpoint(before_burned)
        candidate_corpus_ids = {
            row.corpus_id for row in candidate_checkpoint
        }
        if any(item.corpus_id in candidate_corpus_ids for item in contexts):
            # Idempotent resume after the first composition published the
            # mixed checkpoint. Treat it as complete and validate every role;
            # never append the burned rows a second time.
            complete_checkpoint_path = before_burned
            checkpoint = candidate_checkpoint

    if complete_checkpoint_path is not None:
        if checkpoint is None:
            checkpoint = _read_observation_checkpoint(
                complete_checkpoint_path
            )
        base = ObservationCorpus(tuple(
            row for row in checkpoint if row.corpus_id == context.corpus_id
        ))
        _validate_adapted_development_checkpoint(base, context)
        candidate_plan = _checkpoint_candidate_expansion_plan(
            args,
            inputs,
            timing_sidecars,
        )
        _validate_checkpoint_candidate_expansion(base, candidate_plan)
        plans = []
        represented_corpus_ids = {context.corpus_id}
        for transaction, burned_context in zip(
            transactions,
            contexts,
            strict=True,
        ):
            _partition, plan = _validate_burned_seal_checkpoint_partition(
                checkpoint,
                transaction,
                burned_context,
                route_manifest,
                split_manifest,
            )
            plans.append(plan)
            represented_corpus_ids.add(burned_context.corpus_id)
        unexpected = {
            row.corpus_id for row in checkpoint
        }.difference(represented_corpus_ids)
        if unexpected:
            raise ValueError(
                "mixed CPU prefill checkpoint contains undeclared transaction "
                f"identities: {sorted(unexpected)}"
            )
        return checkpoint, candidate_plan, tuple(plans)

    base_args = args
    if before_burned is not None:
        base_args = argparse.Namespace(**vars(args))
        base_args.adapted_full_development_input = before_burned
        base_args.adapted_development_before_burned_seals_input = None
    base, candidate_plan = _adapt_development_with_candidate_expansion(
        base_args,
        inputs,
        timing_sidecars,
        context,
        generic_refinement_plans,
    )

    burned_corpora = []
    plans = []
    for transaction, burned_context in zip(
        transactions,
        contexts,
        strict=True,
    ):
        aggregate = transaction.aggregate.resolve(transaction.path)
        timing = transaction.timing.resolve(transaction.path)
        burned = adapt_cpu_prefill_csv(
            (aggregate,),
            burned_context,
            timing_sidecars=(timing,),
        )
        plan = read_cpu_prefill_sealed_witness_plan(
            transaction.witness_plan.resolve(transaction.path)
        )
        _require_partition(
            burned,
            route_manifest,
            split_manifest,
            sealed=True,
            sealed_witness_plan=plan,
        )
        burned_corpora.append(burned)
        plans.append(plan)

    return ObservationCorpus((
        *base.observations,
        *(
            row
            for burned in burned_corpora
            for row in burned.observations
        ),
    )), candidate_plan, tuple(plans)


def _load_development_profiler_catalogs(
    args: argparse.Namespace,
    corpus: ObservationCorpus,
) -> ProfilerFeatureCatalog:
    """Load old and additive profiler transactions into one complete catalog.

    Reusable profiler observations remain bound to the exact common CSV that
    emitted their requests. A later candidate-expansion transaction supplies a
    second common CSV containing only the newly implemented physical family.
    Each triplet is authenticated independently before descriptor merging; the
    merged catalog then proves complete candidate coverage against ``corpus``.
    """

    requests = tuple(args.development_profiler_requests)
    evidence = tuple(args.development_profiler_evidence)
    observations = tuple(args.development_profiler_observations)
    catalogs = []
    for index, (request_path, evidence_path) in enumerate(
        zip(requests, evidence, strict=True)
    ):
        source = (
            read_observation_csv((observations[index],))
            if observations
            else corpus
        )
        catalogs.append(load_profiler_feature_catalog(
            source,
            request_path,
            evidence_path,
        ))
    return merge_profiler_feature_catalogs(corpus, catalogs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input", nargs="+", type=Path, dest="input_options")
    parser.add_argument("--timing-sidecar", action="append", type=Path, default=[])
    parser.add_argument(
        "--authenticated-raw-corpus-id",
        help=(
            "Raw input identity already proven by an authenticated replay "
            "recipe; avoids re-hashing multi-gigabyte immutable sidecars"
        ),
    )
    parser.add_argument("--development-input", action="append", type=Path, default=[])
    parser.add_argument(
        "--development-timing-sidecar", action="append", type=Path, default=[]
    )
    parser.add_argument("--sealed-input", action="append", type=Path, default=[])
    parser.add_argument(
        "--sealed-timing-sidecar", action="append", type=Path, default=[]
    )
    parser.add_argument("--freeze-generic", action="store_true")
    parser.add_argument("--certify-generic", action="store_true")
    parser.add_argument(
        "--development-fit-diagnostic",
        type=Path,
        help=(
            "Write reusable, explicitly non-installable development CV IR "
            "for generic boundary-refinement planning"
        ),
    )
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--frozen-policy-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument(
        "--sealed-witness-plan-json",
        type=Path,
        help="Frozen-policy-bound timing-free sealed launch plan",
    )
    parser.add_argument(
        "--sealed-candidate-route-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Fresh zero-kernel C++ route candidates for frozen-leaf sealed "
            "witness selection; repeat for disjoint route files"
        ),
    )
    parser.add_argument(
        "--certification-diagnostic",
        type=Path,
        help=(
            "Write a sealed non-installable report before enforcing promotion "
            "gates"
        ),
    )
    parser.add_argument(
        "--fit-cache-dir",
        type=Path,
        help="Persistent content-addressed candidate-cost and CV cache",
    )
    parser.add_argument(
        "--generic-max-leaves",
        type=int,
        default=DEFAULT_TREE_LEAVES,
        help=(
            "Maximum reviewed generic-tree leaves retained in the grouped-CV "
            f"tournament (1..{MAX_TREE_LEAVES})"
        ),
    )
    parser.add_argument(
        "--generic-refinement-plan-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Authenticated additional development launch inventory; repeat "
            "for every failed-fit refinement round"
        ),
    )
    parser.add_argument(
        "--development-lineage-plan-json",
        type=Path,
        help=(
            "Authenticated additive migration from an older CPU prefill "
            "development split"
        ),
    )
    parser.add_argument("--development-lineage-source-input", type=Path)
    parser.add_argument(
        "--development-lineage-source-timing-sidecar",
        type=Path,
    )
    parser.add_argument(
        "--development-lineage-source-split-manifest",
        type=Path,
    )
    parser.add_argument(
        "--development-lineage-source-refinement-plan-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Source-split refinement plan authenticated by the additive "
            "lineage; repeat in historical order"
        ),
    )
    parser.add_argument(
        "--development-lineage-source-refinement-split-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Additional reviewed split used by historical source refinement "
            "plans; repeat once per distinct older split"
        ),
    )
    parser.add_argument(
        "--development-lineage-source-route-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Historical source serial-route CSV; repeat once per ISA regime"
        ),
    )
    parser.add_argument(
        "--candidate-expansion-plan-json",
        type=Path,
        help=(
            "Authenticated additive candidate inventory derived from the "
            "single immutable development input"
        ),
    )
    parser.add_argument(
        "--candidate-expansion-input",
        type=Path,
        help="Raw anchor-plus-new-candidate aggregate named by the expansion plan",
    )
    parser.add_argument(
        "--candidate-expansion-timing-sidecar",
        type=Path,
        help="Raw timing samples for the candidate-expansion aggregate",
    )
    parser.add_argument(
        "--adapted-development-input",
        type=Path,
        help=(
            "Canonical candidate-expansion observation checkpoint; development "
            "fit mode authenticates its complete plan matrix and skips raw "
            "sidecar adaptation"
        ),
    )
    parser.add_argument(
        "--adapted-full-development-input",
        type=Path,
        help=(
            "Canonical complete development checkpoint; fit, freeze, and "
            "certification reauthenticate it against the declared raw corpus "
            "and skip timing-sidecar adaptation"
        ),
    )
    parser.add_argument(
        "--adapted-full-development-rebase-input",
        type=Path,
        help=(
            "Canonical complete checkpoint from an authenticated equivalent "
            "recipe path. Adaptation validates its full plan partition and "
            "changes transaction provenance only; raw timing is not re-read"
        ),
    )
    parser.add_argument(
        "--burned-sealed-development-manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Content-addressed failed-seal transaction admitted only as a "
            "new development generation; repeat in inspection order"
        ),
    )
    parser.add_argument(
        "--adapted-development-before-burned-seals-input",
        type=Path,
        help=(
            "Authenticated complete base-development checkpoint that omits "
            "the declared burned seals; only those seals are adapted and "
            "appended"
        ),
    )
    parser.add_argument(
        "--adapted-development-prefix-input",
        type=Path,
        help=(
            "Canonical checkpoint for the first N development inputs; "
            "adaptation authenticates that historical prefix, adapts only "
            "the remaining suffix, and publishes a full checkpoint"
        ),
    )
    parser.add_argument(
        "--adapted-development-prefix-count",
        type=int,
        help=(
            "Number of leading --input/--timing-sidecar pairs represented by "
            "--adapted-development-prefix-input"
        ),
    )
    parser.add_argument(
        "--development-profiler-requests",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--development-profiler-evidence",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--development-profiler-observations",
        action="append",
        type=Path,
        default=[],
        help=(
            "Original common CSV bound to one reusable profiler sidecar; "
            "repeat in request/evidence order for additive catalogs"
        ),
    )
    parser.add_argument(
        "--ablate-profiler-features",
        action="store_true",
        help=(
            "Validate the complete development profiler catalog but withhold "
            "its features from a development-only fit"
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument(
        "--candidate-expansion-profiler-observations",
        type=Path,
        help=(
            "Write only newly added candidates for an incremental profiler "
            "request/evidence transaction"
        ),
    )
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-isa-matrix", action="store_true")
    parser.add_argument(
        "--profile",
        choices=[profile.value for profile in MeasurementProfile],
        default=MeasurementProfile.QUICK.value,
    )
    parser.add_argument(
        "--route-manifest",
        action="append",
        type=Path,
        default=[],
        help="C++ serial-route CSV; repeat once per ISA regime",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reviewed CPU prefill development/sealed shape split",
    )
    provenance_names = (
        "run-id", "git-revision", "build-id", "compiler-id",
        "architecture-class", "device-name", "driver-runtime",
        "serial-m1-policy-hash",
    )
    for name in provenance_names:
        parser.add_argument(f"--{name}", default="")
    for name in provenance_names[1:]:
        parser.add_argument(
            f"--sealed-{name}",
            default="",
            help=(
                "Fresh sealed-measurement provenance used only by generic "
                "certification; all sealed provenance arguments are required "
                "together"
            ),
        )
    args = parser.parse_args()
    if not 1 <= args.generic_max_leaves <= MAX_TREE_LEAVES:
        parser.error(
            "--generic-max-leaves must be between 1 and "
            f"{MAX_TREE_LEAVES}"
        )
    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    expansion_options = (
        args.candidate_expansion_plan_json,
        args.candidate_expansion_input,
        args.candidate_expansion_timing_sidecar,
    )
    if any(expansion_options) != all(expansion_options):
        parser.error(
            "CPU prefill candidate expansion requires its plan, aggregate, "
            "and timing sidecar together"
        )
    if args.candidate_expansion_profiler_observations and not (
        args.adapt_only and all(expansion_options)
    ):
        parser.error(
            "--candidate-expansion-profiler-observations requires "
            "--adapt-only and a complete candidate-expansion transaction"
        )
    development_fit = args.development_fit_diagnostic is not None
    try:
        args.burned_sealed_development_transactions = tuple(
            read_cpu_prefill_burned_seal_transaction(path)
            for path in args.burned_sealed_development_manifest
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(f"invalid CPU prefill burned-seal transaction: {error}")
    if len({
        item.transaction_digest
        for item in args.burned_sealed_development_transactions
    }) != len(args.burned_sealed_development_transactions):
        parser.error("CPU prefill burned-seal transactions are duplicated")
    checkpoint_replay_modes = sum(
        value is not None
        for value in (
            args.adapted_development_input,
            args.adapted_full_development_input,
            args.adapted_full_development_rebase_input,
            args.adapted_development_prefix_input,
            args.adapted_development_before_burned_seals_input,
        )
    )
    if checkpoint_replay_modes > 1:
        parser.error(
            "CPU prefill checkpoint replay modes are mutually exclusive"
        )
    if (
        args.adapted_development_before_burned_seals_input is not None
        and not args.burned_sealed_development_transactions
    ):
        parser.error(
            "--adapted-development-before-burned-seals-input requires at "
            "least one --burned-sealed-development-manifest"
        )
    if args.burned_sealed_development_transactions and not (
        args.adapt_only
        or development_fit
        or args.freeze_generic
        or args.certify_generic
    ):
        parser.error(
            "burned CPU prefill seals are restricted to adaptation, "
            "development fit, freeze, or certification"
        )
    if (
        args.adapted_development_prefix_count is not None
        and args.adapted_development_prefix_input is None
    ):
        parser.error(
            "CPU prefill checkpoint prefix count requires its checkpoint input"
        )
    if (
        args.adapted_development_prefix_count is not None
        and args.adapted_development_prefix_count <= 0
    ):
        parser.error("CPU prefill checkpoint prefix count must be positive")
    if args.adapted_development_prefix_input is not None and not args.adapt_only:
        parser.error(
            "--adapted-development-prefix-input is restricted to adaptation mode"
        )
    if (
        args.adapted_full_development_rebase_input is not None
        and not args.adapt_only
    ):
        parser.error(
            "--adapted-full-development-rebase-input is restricted to "
            "adaptation mode"
        )
    if args.adapted_development_input is not None and not development_fit:
        parser.error(
            "--adapted-development-input is restricted to development fit mode"
        )
    if args.adapted_full_development_input is not None and not (
        development_fit or args.freeze_generic or args.certify_generic
    ):
        parser.error(
            "--adapted-full-development-input requires development fit, "
            "freeze, or certification mode"
        )
    if sum((args.freeze_generic, args.certify_generic, development_fit)) > 1:
        parser.error(
            "development fit, freeze, and certification modes are mutually exclusive"
        )
    if args.adapt_only and (
        args.freeze_generic or args.certify_generic or development_fit
    ):
        parser.error("--adapt-only cannot freeze or certify a policy")
    if args.ablate_profiler_features and not development_fit:
        parser.error(
            "--ablate-profiler-features is restricted to development fit mode"
        )
    if len(args.development_profiler_requests) != len(
        args.development_profiler_evidence
    ):
        parser.error(
            "development profiler request/evidence counts must match"
        )
    if args.development_profiler_observations and len(
        args.development_profiler_observations
    ) != len(args.development_profiler_requests):
        parser.error(
            "development profiler observations must accompany every catalog"
        )
    if (args.freeze_generic or args.certify_generic or development_fit) and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "production CPU prefill fit/freeze/certification requires complete "
            "development profiler evidence"
        )
    separate_certification = bool(
        args.development_input
        or args.development_timing_sidecar
        or args.sealed_input
        or args.sealed_timing_sidecar
        or args.frozen_policy_json
    )
    if args.freeze_generic:
        if not inputs or separate_certification:
            parser.error("--freeze-generic accepts development --input only")
        if not args.policy_json:
            parser.error("--freeze-generic requires --policy-json")
        if not args.sealed_witness_plan_json:
            parser.error(
                "--freeze-generic requires --sealed-witness-plan-json"
            )
    elif development_fit:
        if not inputs or separate_certification:
            parser.error(
                "--development-fit-diagnostic accepts development --input only"
            )
    elif args.certify_generic:
        if inputs or args.timing_sidecar:
            parser.error("--certify-generic accepts separate partition inputs only")
        if not args.development_input or not args.sealed_input:
            parser.error("certification requires development and sealed inputs")
        if not args.frozen_policy_json or not args.policy_json:
            parser.error("certification requires frozen and compiled policy JSON")
        if not args.sealed_witness_plan_json:
            parser.error(
                "--certify-generic requires --sealed-witness-plan-json"
            )
    elif separate_certification:
        parser.error("separate partition inputs require --certify-generic")
    elif not inputs:
        parser.error("at least one strong CPU prefill CSV is required")
    if args.certification_diagnostic and not args.certify_generic:
        parser.error("--certification-diagnostic requires --certify-generic")
    sealed_provenance = tuple(
        getattr(args, f"sealed_{name.replace('-', '_')}")
        for name in provenance_names[1:]
    )
    if args.certify_generic and not all(sealed_provenance):
        parser.error(
            "--certify-generic requires complete explicit --sealed-* provenance"
        )
    if not args.certify_generic and any(sealed_provenance):
        parser.error("--sealed-* provenance requires --certify-generic")
    if args.sealed_candidate_route_manifest and not args.freeze_generic:
        parser.error(
            "--sealed-candidate-route-manifest is restricted to generic freeze"
        )
    if (
        MeasurementProfile(args.profile) == MeasurementProfile.PRODUCTION
        and not args.freeze_generic
        and not args.certify_generic
        and not development_fit
        and not args.adapt_only
    ):
        parser.error(
            "production CPU prefill policy emission requires development freeze "
            "and separate sealed certification"
        )

    route_manifest = (
        read_cpu_prefill_route_manifests(args.route_manifest)
        if args.route_manifest
        else None
    )
    sealed_candidate_route_manifest = (
        read_cpu_prefill_route_manifests(
            args.sealed_candidate_route_manifest
        )
        if args.sealed_candidate_route_manifest
        else None
    )
    if (
        args.freeze_generic or args.certify_generic or development_fit
    ) and route_manifest is None:
        parser.error(
            "CPU prefill fit/freeze/certification requires production route manifests"
        )
    split_manifest = load_cpu_prefill_split_manifest(args.split_manifest)
    generic_refinement_plans = (
        tuple(
            read_cpu_prefill_generic_refinement_plan(
                path,
                route_manifest,
                split_manifest,
            )
            for path in args.generic_refinement_plan_json
        )
        if route_manifest is not None
        else ()
    )
    lineage_options = (
        args.development_lineage_plan_json,
        args.development_lineage_source_input,
        args.development_lineage_source_timing_sidecar,
        args.development_lineage_source_split_manifest,
    )
    if any(lineage_options) != all(lineage_options):
        parser.error(
            "development lineage requires its plan, source aggregate, source "
            "timing sidecar, and source split manifest"
        )
    if args.development_lineage_source_refinement_plan_json and not all(
        lineage_options
    ):
        parser.error(
            "source refinement plans require a complete development lineage"
        )
    if (
        args.development_lineage_source_refinement_split_manifest
        or args.development_lineage_source_route_manifest
    ) and not all(lineage_options):
        parser.error(
            "source route and refinement split manifests require a complete "
            "development lineage"
        )
    development_lineage = None
    if all(lineage_options):
        if route_manifest is None:
            parser.error("CPU prefill development lineage requires route manifests")
        if not args.development_lineage_source_route_manifest:
            parser.error(
                "CPU prefill development lineage requires source route manifests"
            )
        source_split_manifest = load_cpu_prefill_split_manifest(
            args.development_lineage_source_split_manifest
        )
        source_refinement_split_manifests = (
            source_split_manifest,
            *(
                load_cpu_prefill_split_manifest(path)
                for path in (
                    args.development_lineage_source_refinement_split_manifest
                )
            ),
        )
        source_route_manifest = read_cpu_prefill_route_manifests(
            args.development_lineage_source_route_manifest
        )
        source_refinement_plans = _read_lineage_source_refinement_plans(
            args.development_lineage_source_refinement_plan_json,
            source_route_manifest,
            source_refinement_split_manifests,
        )
        development_lineage = read_cpu_prefill_development_lineage_plan(
            args.development_lineage_plan_json,
            args.development_lineage_source_input,
            args.development_lineage_source_timing_sidecar,
            source_split_manifest,
            split_manifest,
            source_route_manifest,
            route_manifest,
            source_refinement_plans,
        )
    if args.generic_refinement_plan_json and not (
        args.freeze_generic
        or args.certify_generic
        or development_fit
        or args.adapt_only
    ):
        parser.error(
            "--generic-refinement-plan-json requires adaptation, fit, freeze, "
            "or certification"
        )

    if args.adapt_only:
        if not inputs or separate_certification:
            parser.error("--adapt-only accepts development --input only")
        if route_manifest is None:
            parser.error("CPU prefill adaptation requires production route manifests")
        timing = tuple(args.timing_sidecar)
        context_inputs, context_timing = _development_context_paths(
            args, inputs, timing
        )
        context = _context_from_args(
            args,
            context_inputs,
            context_timing,
            authenticated_corpus_id=args.authenticated_raw_corpus_id,
        )
        (
            corpus,
            candidate_expansion_plan,
            burned_sealed_witness_plans,
        ) = _adapt_development_with_burned_seals(
            args,
            inputs,
            timing,
            context,
            route_manifest,
            split_manifest,
            generic_refinement_plans,
        )
        _require_partition(
            corpus,
            route_manifest,
            split_manifest,
            sealed=False,
            generic_refinement_plans=generic_refinement_plans,
            development_lineage=development_lineage,
            candidate_expansion_plan=candidate_expansion_plan,
            burned_sealed_witness_plans=burned_sealed_witness_plans,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            [],
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
        ), encoding="utf-8")
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        if args.candidate_expansion_profiler_observations:
            expansion_candidates = frozenset(
                candidate_expansion_plan.expansion_candidate_ids
            )
            profiler_corpus = ObservationCorpus(tuple(
                row for row in corpus
                if row.candidate_id in expansion_candidates
            ))
            args.candidate_expansion_profiler_observations.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            write_observation_csv(
                args.candidate_expansion_profiler_observations,
                profiler_corpus,
            )
        print(
            f"adapted {len(corpus)} CPU prefill development observations "
            f"without fitting -> {args.output}"
        )
        return 0

    if development_fit:
        development_fit_started = time.perf_counter()
        timing = tuple(args.timing_sidecar)
        context_inputs, context_timing = _development_context_paths(
            args, inputs, timing
        )
        context = _context_from_args(
            args,
            context_inputs,
            context_timing,
            authenticated_corpus_id=args.authenticated_raw_corpus_id,
        )
        context_complete = time.perf_counter()
        (
            corpus,
            candidate_expansion_plan,
            burned_sealed_witness_plans,
        ) = _adapt_development_with_burned_seals(
            args,
            inputs,
            timing,
            context,
            route_manifest,
            split_manifest,
            generic_refinement_plans,
        )
        checkpoint_complete = time.perf_counter()
        profiler_catalog = _load_development_profiler_catalogs(args, corpus)
        profiler_catalog_complete = time.perf_counter()
        frozen = freeze_cpu_prefill_policy(
            corpus,
            route_manifest,
            split_manifest,
            context.serial_m1_policy_hash,
            None if args.ablate_profiler_features else profiler_catalog,
            args.fit_cache_dir,
            generic_refinement_plans,
            development_lineage,
            candidate_expansion_plan=candidate_expansion_plan,
            burned_sealed_witness_plans=burned_sealed_witness_plans,
            max_leaves=args.generic_max_leaves,
            profiler_ablation_catalog_digest=(
                profiler_catalog.digest
                if args.ablate_profiler_features
                else None
            ),
        )
        policy_fit_complete = time.perf_counter()
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, generic_rules)
        emission_inputs_complete = time.perf_counter()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            policy_digest=frozen.policy_ir.digest(),
        ), encoding="utf-8")
        write_development_fit_diagnostic(
            args.development_fit_diagnostic,
            frozen,
        )
        if _common_checkpoint_requires_write(
            args.common_observations,
            args.adapted_full_development_input,
        ):
            args.common_observations.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            write_observation_csv(args.common_observations, corpus)
        artifacts_complete = time.perf_counter()
        if os.environ.get("LLAMINAR_NATIVE_VNNI_POLICY_TIMING", "0") == "1":
            print(
                "NativeVNNI CPU prefill analyzer timing "
                f"context_identity={context_complete - development_fit_started:.3f}s "
                f"checkpoint={checkpoint_complete - context_complete:.3f}s "
                "profiler_catalog="
                f"{profiler_catalog_complete - checkpoint_complete:.3f}s "
                f"policy_fit={policy_fit_complete - profiler_catalog_complete:.3f}s "
                "emission_inputs="
                f"{emission_inputs_complete - policy_fit_complete:.3f}s "
                f"artifact_write={artifacts_complete - emission_inputs_complete:.3f}s "
                f"total={artifacts_complete - development_fit_started:.3f}s",
                file=sys.stderr,
                flush=True,
            )
        required_domain_count = len(frozen.policy_ir.cross_validation)
        rejected_domain_count = len(frozen.promotion_diagnostics)
        over_budget_domain_count = sum(
            diagnostic.rejection_stage not in {
                "cross_validation_missing",
                "cross_validation_coverage",
                "final_fit_empty",
            }
            for diagnostic in frozen.promotion_diagnostics
        )
        passing_domain_fraction = (
            float(required_domain_count - rejected_domain_count)
            / float(required_domain_count)
            if required_domain_count
            else 0.0
        )
        print(
            f"fitted {len(corpus)} CPU prefill development observations; "
            f"generic_rules={len(frozen.policy_ir.generic_rules)} "
            f"unpromoted_domains={len(frozen.policy_ir.unpromoted_domains)} "
            f"over_budget_domains={over_budget_domain_count} "
            f"passing_domain_fraction={passing_domain_fraction:.6f} "
            f"-> {args.development_fit_diagnostic}"
        )
        return 0

    if args.freeze_generic:
        timing = tuple(args.timing_sidecar)
        context_inputs, context_timing = _development_context_paths(
            args, inputs, timing
        )
        context = _context_from_args(
            args,
            context_inputs,
            context_timing,
            authenticated_corpus_id=args.authenticated_raw_corpus_id,
        )
        (
            corpus,
            candidate_expansion_plan,
            burned_sealed_witness_plans,
        ) = _adapt_development_with_burned_seals(
            args,
            inputs,
            timing,
            context,
            route_manifest,
            split_manifest,
            generic_refinement_plans,
        )
        profiler_catalog = _load_development_profiler_catalogs(args, corpus)
        frozen = freeze_cpu_prefill_policy(
            corpus,
            route_manifest,
            split_manifest,
            context.serial_m1_policy_hash,
            profiler_catalog,
            args.fit_cache_dir,
            generic_refinement_plans,
            development_lineage,
            candidate_expansion_plan=candidate_expansion_plan,
            burned_sealed_witness_plans=burned_sealed_witness_plans,
            max_leaves=args.generic_max_leaves,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, generic_rules)
        # Generic dispatch is the architecture, while exact entries are only
        # additive overlays. Prove totality before committing a frozen policy
        # or deriving any post-freeze sealed witness launches.
        validate_total_policy(entries, generic_rules, route_manifest)
        sealed_witness_plan = build_cpu_prefill_sealed_witness_plan(
            frozen.policy_ir.generic_rules,
            frozen.generic_digest,
            split_manifest,
            route_manifest,
            candidate_route_manifest=sealed_candidate_route_manifest,
            development_dimensions=frozenset(
                (row.aggregate_n, row.k) for row in corpus
            ),
        )
        validate_cpu_prefill_sealed_witness_plan(
            sealed_witness_plan,
            frozen.policy_ir.generic_rules,
            split_manifest,
            route_manifest,
            development_dimensions=frozenset(
                (row.aggregate_n, row.k) for row in corpus
            ),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
            policy_digest=frozen.policy_ir.digest(),
        ), encoding="utf-8")
        write_frozen_policy(args.policy_json, frozen)
        write_cpu_prefill_sealed_witness_plan(
            args.sealed_witness_plan_json,
            sealed_witness_plan,
        )
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"froze {len(corpus)} CPU prefill development observations as "
            f"{frozen.generic_digest} -> {args.output}"
        )
        return 0

    if args.certify_generic:
        development_inputs = tuple(args.development_input)
        development_timing = tuple(args.development_timing_sidecar)
        context_inputs, context_timing = _development_context_paths(
            args, development_inputs, development_timing
        )
        development_context = _context_from_args(
            args,
            context_inputs,
            context_timing,
            authenticated_corpus_id=args.authenticated_raw_corpus_id,
        )
        (
            development,
            candidate_expansion_plan,
            burned_sealed_witness_plans,
        ) = _adapt_development_with_burned_seals(
            args,
            development_inputs,
            development_timing,
            development_context,
            route_manifest,
            split_manifest,
            generic_refinement_plans,
        )
        profiler_catalog = _load_development_profiler_catalogs(
            args,
            development,
        )
        frozen = freeze_cpu_prefill_policy(
            development,
            route_manifest,
            split_manifest,
            development_context.serial_m1_policy_hash,
            profiler_catalog,
            args.fit_cache_dir,
            generic_refinement_plans,
            development_lineage,
            candidate_expansion_plan=candidate_expansion_plan,
            burned_sealed_witness_plans=burned_sealed_witness_plans,
            max_leaves=args.generic_max_leaves,
        )
        # This byte comparison completes before the first sealed file is read.
        validate_frozen_policy_file(args.frozen_policy_json, frozen)
        sealed_witness_plan = read_cpu_prefill_sealed_witness_plan(
            args.sealed_witness_plan_json
        )
        validate_cpu_prefill_sealed_witness_plan(
            sealed_witness_plan,
            frozen.policy_ir.generic_rules,
            split_manifest,
            route_manifest,
            development_dimensions=frozenset(
                (row.aggregate_n, row.k) for row in development
            ),
        )

        sealed_inputs = tuple(args.sealed_input)
        sealed_timing = tuple(args.sealed_timing_sidecar)
        sealed_context = _context_from_args(
            args,
            sealed_inputs,
            sealed_timing,
            run_id=f"{args.run_id}-sealed",
            provenance_prefix="sealed",
        )
        sealed = adapt_cpu_prefill_csv(
            sealed_inputs,
            sealed_context,
            timing_sidecars=sealed_timing,
        )
        compiled = certify_cpu_prefill_policy(
            frozen,
            development,
            sealed,
            route_manifest,
            split_manifest,
            sealed_witness_plan,
            development_context.serial_m1_policy_hash,
            require_promotable=not bool(args.certification_diagnostic),
            generic_refinement_plans=generic_refinement_plans,
            development_lineage=development_lineage,
            candidate_expansion_plan=candidate_expansion_plan,
            burned_sealed_witness_plans=burned_sealed_witness_plans,
        )
        if args.certification_diagnostic:
            write_certification_diagnostic(
                args.certification_diagnostic,
                compiled,
            )
            # The diagnostic is useful precisely when this raises. It is not an
            # alternate publication path and cannot relax any promotion gate.
            compiled.certification.require_promotable()
        corpus = ObservationCorpus((
            *development.observations,
            *sealed.observations,
        ))
        entries = select_entries(
            corpus,
            development_context.serial_m1_policy_hash,
        )
        generic_rules = _emit_generic_rules(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled.policy_ir, entries, generic_rules)
        validate_total_policy(entries, generic_rules, route_manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=development_context.profile,
            policy_digest=compiled.policy_ir.digest(),
            certification=compiled.certification,
        ), encoding="utf-8")
        write_compiled_policy(args.policy_json, compiled)
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"certified frozen CPU prefill policy {frozen.generic_digest} "
            f"against {len(sealed)} sealed observations -> {args.output}"
        )
        return 0

    timing = tuple(args.timing_sidecar)
    context_inputs, context_timing = _development_context_paths(
        args, inputs, timing
    )
    context = _context_from_args(
        args,
        context_inputs,
        context_timing,
        authenticated_corpus_id=args.authenticated_raw_corpus_id,
    )
    corpus, candidate_expansion_plan = _adapt_development_with_candidate_expansion(
        args, inputs, timing, context
    )
    entries = select_entries(corpus, context.serial_m1_policy_hash)
    generic_rules = select_generic_rules(corpus, context.serial_m1_policy_hash)
    if args.require_complete:
        if route_manifest is None:
            raise ValueError(
                "complete CPU prefill policy generation requires production "
                "C++ serial-route manifests"
            )
        if candidate_expansion_plan is None:
            validate_complete(corpus, route_manifest)
        else:
            validate_candidate_expansion_complete(
                corpus,
                candidate_expansion_plan,
            )
        validate_total_policy(entries, generic_rules, route_manifest)
    elif args.require_isa_matrix:
        validate_isa_regimes(corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        generate_include(
            entries,
            generic_rules,
            corpus_digest=corpus.digest(),
            registry_digest=candidate_registry_digest(),
            profile=context.profile,
        ),
        encoding="utf-8",
    )
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_summary(args.summary, entries)
    if args.common_observations:
        args.common_observations.parent.mkdir(parents=True, exist_ok=True)
        write_observation_csv(args.common_observations, corpus)
    print(
        f"adapted {len(corpus)} strong CPU prefill observations; generated "
        f"{len(entries)} exact policies -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
