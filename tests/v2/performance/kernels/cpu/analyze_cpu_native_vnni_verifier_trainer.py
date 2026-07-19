#!/usr/bin/env python3
"""Compile strong CPU grouped-verifier evidence through the common policy core.

Byte equality, effective route identity, repeat stability, alias coverage, and
candidate selection are shared with CUDA/ROCm.  The backend emitter only maps
the common Pairwise/WideRows decisions into the existing CPU selector ABI; it
does not apply independent cosine, L2, KL, or speedup filters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path


KERNEL_PERF_ROOT = Path(__file__).resolve().parents[1]
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.adapters.cpu_verifier import (  # noqa: E402
    CPUVerifierAdapterContext,
    adapt_cpu_verifier_csv,
    raw_corpus_id,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    read_observation_csv,
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_verifier_registry,
)
from native_vnni_dispatch.certification import CertificationReport  # noqa: E402
from native_vnni_dispatch.compiler import (  # noqa: E402
    CompiledPolicy,
    FrozenPolicy,
    certify_frozen_policy,
    freeze_policy,
)
from native_vnni_dispatch.corpus import ObservationCorpus, RuntimeKey  # noqa: E402
from native_vnni_dispatch.cpp_predicates import (  # noqa: E402
    aspect_condition,
    generic_rule_sort_key,
    predicate_condition,
    render_if_header,
)
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.policy_artifact import (  # noqa: E402
    validate_frozen_policy_file,
    write_compiled_policy,
    write_frozen_policy,
)
from native_vnni_dispatch.policy_ir import PolicyIR  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.profiler_model import (  # noqa: E402
    ProfilerFeatureCatalog,
    load_profiler_feature_catalog,
)
from native_vnni_dispatch.schema import SemanticContract  # noqa: E402
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
    fit_generic_policy,
)
from native_vnni_dispatch.shape_manifest import (  # noqa: E402
    MANIFEST_PATH,
    NativeVNNIShapeManifest,
    ShapePartition,
    load_shape_manifest,
    partition_assignments,
)
from native_vnni_dispatch.validation import (  # noqa: E402
    require_candidate_matrix_complete,
    require_canonical_alias_coverage,
    require_registry_candidate_coverage,
    require_verifier_m_matrix,
)


@dataclass(frozen=True, order=True)
class PolicyEntry:
    """One common exact decision encoded for the CPU generated selector."""

    build_isa: str
    runtime_isa: str
    threads: int
    codebook: int
    m: int
    n: int
    k: int
    policy: str
    candidate_id: str
    shape_name: str
    max_surface_regret: float
    max_cv: float


@dataclass(frozen=True)
class CPUGenericDispatchRule:
    """One generic rule plus the CPU runtime surface that certified it."""

    build_isa: str
    runtime_isa: str
    threads: int
    rule: GenericDispatchRule


def _cpu_runtime_surface(architecture_class: str) -> tuple[str, str, int]:
    """Decode the adapter-owned ISA/thread suffix from an architecture key."""

    parts = architecture_class.rsplit("|", 3)
    if len(parts) != 4:
        raise ValueError(
            "CPU architecture class lacks build/runtime/thread identity: "
            f"{architecture_class!r}"
        )
    build_part, runtime_part, threads_part = parts[1:]
    if not build_part.startswith("build=") or not runtime_part.startswith("runtime="):
        raise ValueError(f"malformed CPU architecture class {architecture_class!r}")
    if not threads_part.startswith("threads="):
        raise ValueError(f"malformed CPU thread class {architecture_class!r}")
    build_isa = build_part.removeprefix("build=")
    runtime_isa = runtime_part.removeprefix("runtime=")
    threads = int(threads_part.removeprefix("threads="))
    if build_isa not in {"AVX2", "AVX512"}:
        raise ValueError(f"unsupported CPU build ISA {build_isa!r}")
    if runtime_isa not in {"AVX2", "AVX512"}:
        raise ValueError(f"unsupported CPU runtime ISA {runtime_isa!r}")
    if build_isa == "AVX2" and runtime_isa != "AVX2":
        raise ValueError("AVX2 build cannot own an AVX512 runtime policy")
    if threads <= 0:
        raise ValueError("CPU policy thread count must be positive")
    return build_isa, runtime_isa, threads


def _pack_key(codebook: int, m: int, n: int, k: int) -> int:
    return (
        ((codebook & 0xFF) << 56)
        | ((m & 0xFF) << 48)
        | ((k & 0xFFFFFF) << 24)
        | (n & 0xFFFFFF)
    )


def _candidate_policy(candidate_id: str) -> str:
    candidate = cpu_native_vnni_verifier_registry().resolve(candidate_id)
    policy = str(candidate.config_json["policy"])
    if policy not in {"Pairwise", "WideRows"}:
        raise ValueError(f"unknown CPU verifier policy {policy!r}")
    return policy


def _serial_hashes(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> dict[RuntimeKey, str]:
    return {
        key: serial_m1_policy_hash
        for key in corpus.runtime_keys()
        if key.semantic_contract == SemanticContract.VERIFIER_SERIAL_M1_BITWISE
    }


def select_entries(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[PolicyEntry]:
    """Select exact candidates with the common alias-robust oracle."""

    winners = build_exact_winners(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    entries = []
    for key, winner in sorted(winners.items()):
        if key.semantic_contract != SemanticContract.VERIFIER_SERIAL_M1_BITWISE:
            raise ValueError(f"unexpected CPU semantic contract in {key}")
        if len(key.projection_n_vector) != 1:
            raise ValueError("CPU verifier policy only supports one projection")
        rows = corpus.rows_for_runtime_key(key)
        shape_names = sorted({row.shape_name for row in rows})
        if len(shape_names) != 1:
            raise ValueError(f"multiple shape names collapse onto CPU key {key}")
        build_isa, runtime_isa, threads = _cpu_runtime_surface(
            key.architecture_class
        )
        entries.append(PolicyEntry(
            build_isa=build_isa,
            runtime_isa=runtime_isa,
            threads=threads,
            codebook=key.runtime_codebook_id,
            m=key.m,
            n=key.aggregate_n,
            k=key.k,
            policy=_candidate_policy(winner.candidate_id),
            candidate_id=winner.candidate_id,
            shape_name=shape_names[0],
            max_surface_regret=winner.max_surface_regret,
            max_cv=winner.max_cv,
        ))
    return entries


def select_generic_rules(
    corpus: ObservationCorpus,
    serial_m1_policy_hash: str,
) -> list[CPUGenericDispatchRule]:
    """Fit shared aspect/work rules without a backend-local label learner."""

    generic = fit_generic_policy(
        corpus,
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
    )
    return [
        CPUGenericDispatchRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in generic.rules
    ]


def _emit_generic_rules(
    rules: tuple[GenericDispatchRule, ...],
) -> list[CPUGenericDispatchRule]:
    """Attach CPU ISA/thread surfaces to already-frozen common IR rules."""

    return [
        CPUGenericDispatchRule(
            *_cpu_runtime_surface(rule.domain.architecture_class),
            rule=rule,
        )
        for rule in rules
    ]


def _sealed_commitment(manifest: NativeVNNIShapeManifest) -> str:
    """Commit to the bounded CPU verifier sealed inventory before fitting."""

    payload = {
        "schema": "cpu-native-vnni-verifier-split-v1",
        "shape_manifest_digest": manifest.digest(),
        "sealed_shapes": manifest.cpu_measurement_names(
            verifier=True,
            partition=ShapePartition.SEALED,
        ),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _require_partition(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    partition: ShapePartition,
) -> ObservationCorpus:
    """Reject cross-partition or incomplete CPU verifier evidence."""

    assignments = partition_assignments(
        ((row.shape_group_id, row.shape_name) for row in corpus),
        verifier=True,
        manifest=manifest,
    )
    if set(assignments.values()) != {partition}:
        raise ValueError(
            f"CPU verifier {partition.value} input crosses a manifest partition"
        )
    validate_complete(
        corpus,
        manifest,
        require_full_inventory=True,
        expected_partition=partition,
    )
    return corpus


def freeze_cpu_verifier_policy(
    development: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    serial_m1_policy_hash: str,
    profiler_feature_catalog: ProfilerFeatureCatalog | None = None,
) -> FrozenPolicy:
    """Fit grouped-verifier rules without receiving any sealed observations."""

    corpus = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    )
    return freeze_policy(
        corpus,
        sealed_commitment=_sealed_commitment(manifest),
        split_manifest_digest=manifest.digest(),
        serial_m1_hashes=_serial_hashes(corpus, serial_m1_policy_hash),
        profiler_feature_catalog=profiler_feature_catalog,
        metadata={
            "backend": "cpu",
            "semantic_contract": SemanticContract.VERIFIER_SERIAL_M1_BITWISE.value,
            "shape_manifest_schema": manifest.schema_version,
            "shape_manifest_digest": manifest.digest(),
            "split_surface": "cpu_verifier",
        },
    )


def certify_cpu_verifier_policy(
    frozen: FrozenPolicy,
    development: ObservationCorpus,
    sealed: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    serial_m1_policy_hash: str,
) -> CompiledPolicy:
    """Certify the immutable CPU verifier policy on physically sealed rows."""

    development = _require_partition(
        development,
        manifest,
        ShapePartition.DEVELOPMENT,
    )
    sealed = _require_partition(
        sealed,
        manifest,
        ShapePartition.SEALED,
    )
    combined_hashes = {
        **_serial_hashes(development, serial_m1_policy_hash),
        **_serial_hashes(sealed, serial_m1_policy_hash),
    }
    return certify_frozen_policy(
        frozen,
        development,
        sealed,
        serial_m1_hashes=combined_hashes,
    )


def validate_emitter_inputs(
    policy_ir: PolicyIR,
    entries: list[PolicyEntry],
    generic_rules: list[CPUGenericDispatchRule],
) -> None:
    """Prove that backend emission consumes the exact certified common IR."""

    # Compare architecture surfaces in their decoded representation so the C++
    # emitter cannot accidentally collapse AVX2-build and AVX512-dispatch rows.
    decoded_ir_exact = {
        (
            *_cpu_runtime_surface(entry.key.architecture_class),
            entry.key.runtime_codebook_id,
            entry.key.m,
            entry.key.aggregate_n,
            entry.key.k,
        ): entry.candidate_id
        for entry in policy_ir.exact_entries
    }
    decoded_emitted_exact = {
        (
            entry.build_isa,
            entry.runtime_isa,
            entry.threads,
            entry.codebook,
            entry.m,
            entry.n,
            entry.k,
        ): entry.candidate_id
        for entry in entries
    }
    if decoded_ir_exact != decoded_emitted_exact:
        raise ValueError("CPU verifier emitter exact entries diverge from common IR")
    if tuple(item.rule for item in generic_rules) != policy_ir.generic_rules:
        raise ValueError("CPU verifier emitter generic rules diverge from common IR")


def validate_isa_regime_matrix(corpus: ObservationCorpus) -> None:
    """Require every supported CPU build/runtime dispatch combination."""

    observed_regimes = {
        _cpu_runtime_surface(key.architecture_class)[:2]
        for key in corpus.runtime_keys()
    }
    required_regimes = {
        ("AVX2", "AVX2"),
        ("AVX512", "AVX2"),
        ("AVX512", "AVX512"),
    }
    missing_regimes = sorted(required_regimes.difference(observed_regimes))
    if missing_regimes:
        rendered = ", ".join(
            f"{build}/{runtime}" for build, runtime in missing_regimes
        )
        raise ValueError(f"CPU ISA regime matrix incomplete: {rendered}")


def validate_complete(
    corpus: ObservationCorpus,
    manifest: NativeVNNIShapeManifest,
    *,
    require_full_inventory: bool,
    expected_partition: ShapePartition | None = None,
) -> None:
    """Validate the grouped verifier matrix and reviewed shape applicability."""

    require_canonical_alias_coverage(corpus)
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_verifier_registry(),
    )
    observed_names = {row.shape_name for row in corpus}
    disallowed = sorted(
        name
        for name in observed_names
        if manifest.by_name(name).verifier_partition is None
    )
    if disallowed:
        raise ValueError(
            "CPU verifier corpus contains Fast-only shapes: "
            f"{disallowed[:3]}"
        )
    if require_full_inventory:
        expected_names = set(manifest.cpu_measurement_names(
            verifier=True,
            partition=expected_partition,
        ))
        if observed_names != expected_names:
            raise ValueError(
                "CPU production verifier shape inventory is incomplete: "
                f"missing={sorted(expected_names - observed_names)[:3]} "
                f"unexpected={sorted(observed_names - expected_names)[:3]}"
            )
    validate_isa_regime_matrix(corpus)


def _parse_required_key(raw: str) -> tuple[int, int, int, int, str]:
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid --require-key {raw!r}; expected FORMAT:M:N:K")
    from native_vnni_dispatch.format_registry import format_spec

    spec = format_spec(parts[0])
    m, n, k = (int(value, 0) for value in parts[1:])
    if m < 2 or n <= 0 or k <= 0:
        raise ValueError(f"invalid --require-key dimensions in {raw!r}")
    return spec.runtime_codebook("cpu"), m, n, k, spec.label


def validate_required_keys(entries: list[PolicyEntry], required: list[str]) -> None:
    present = {
        (row.build_isa, row.runtime_isa, row.codebook, row.m, row.n, row.k)
        for row in entries
    }
    regimes = {
        ("AVX2", "AVX2"),
        ("AVX512", "AVX2"),
        ("AVX512", "AVX512"),
    }
    missing = []
    for raw in required:
        codebook, m, n, k, label = _parse_required_key(raw)
        for build_isa, runtime_isa in sorted(regimes):
            if (build_isa, runtime_isa, codebook, m, n, k) not in present:
                missing.append(
                    f"{build_isa}/{runtime_isa}:{label}:M{m}:N{n}:K{k}"
                )
    if missing:
        raise ValueError("missing required CPU verifier policy row(s): " + ", ".join(missing))


def generate_include(
    entries: list[PolicyEntry],
    generic_rules: list[CPUGenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
    policy_digest: str = "",
    certification: CertificationReport | None = None,
) -> str:
    """Render exact/generic decisions into an ISA-aware CPU selector."""

    if certification is None:
        decision_comment = (
            "// Decisions: common NativeVNNI alias-robust exact oracle plus "
            "development-only generic policy; this artifact is not installable "
            "without sealed certification."
        )
    else:
        decision_comment = (
            "// Decisions: common NativeVNNI alias-robust exact oracle plus "
            "frozen development policy with generic-only sealed certification."
        )
    lines = [
        "// Auto-generated by analyze_cpu_native_vnni_verifier_trainer.py. DO NOT EDIT.",
        decision_comment,
        f"// Measurement profile: {profile.value}",
        f"// Common corpus digest: {corpus_digest}",
        f"// Candidate registry digest: {registry_digest}",
        "#pragma once",
        "#define LLAMINAR_CPU_NVNNI_VERIFIER_POLICY_ABI 2",
        "",
        "#include <cstdint>",
        "",
        "namespace llaminar2::cpu::native_vnni::generated",
        "{",
        "enum class CPUNativeVNNIVerifierRowsPolicy : uint8_t",
        "{",
        "    Pairwise = 0,",
        "    WideRows = 1,",
        "};",
        "",
        "enum class CPUNativeVNNIBuildISA : uint8_t",
        "{",
        "    AVX2 = 0,",
        "    AVX512 = 1,",
        "};",
        "",
        "enum class CPUNativeVNNIRuntimeISA : uint8_t",
        "{",
        "    AVX2 = 0,",
        "    AVX512 = 1,",
        "};",
        "",
        "inline constexpr uint64_t packCPUNativeVNNIVerifierRowsPolicyKey(",
        "    uint8_t codebook, int m, int n, int k)",
        "{",
        "    return (static_cast<uint64_t>(codebook) << 56) |",
        "           (static_cast<uint64_t>(m & 0xFF) << 48) |",
        "           (static_cast<uint64_t>(k & 0xFFFFFF) << 24) |",
        "           static_cast<uint64_t>(n & 0xFFFFFF);",
        "}",
        "",
        "inline bool selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
        "    CPUNativeVNNIBuildISA build_isa,",
        "    CPUNativeVNNIRuntimeISA runtime_isa, int threads,",
        "    uint8_t codebook, int m, int n, int k,",
        "    CPUNativeVNNIVerifierRowsPolicy &policy, float *measured_speedup = nullptr)",
        "{",
        "    const uint64_t key =",
        "        packCPUNativeVNNIVerifierRowsPolicyKey(codebook, m, n, k);",
    ]
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
                f"{certification.max_observed_regret:.6%} max-simultaneous-ucb="
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
            entry
            for entry in entries
            if (entry.build_isa, entry.runtime_isa, entry.threads)
            == (build_isa, runtime_isa, threads)
        ]
        group_rules = [
            item
            for item in generic_rules
            if (item.build_isa, item.runtime_isa, item.threads)
            == (build_isa, runtime_isa, threads)
        ]
        lines.extend([
            f"    if (build_isa == CPUNativeVNNIBuildISA::{build_isa} &&",
            f"        runtime_isa == CPUNativeVNNIRuntimeISA::{runtime_isa} &&",
            f"        threads == {threads})",
            "    {",
        ])
        if group_entries:
            lines.extend([
                "        switch (key)",
                "        {",
            ])
            for entry in sorted(group_entries):
                lines.extend([
                    "        case "
                    f"0x{_pack_key(entry.codebook, entry.m, entry.n, entry.k):016x}ULL:",
                    f"            // CB={entry.codebook} {entry.shape_name} "
                    f"{entry.candidate_id} "
                    f"max-regret={entry.max_surface_regret:.4%} "
                    f"max-cv={entry.max_cv:.4%}",
                    "            policy = "
                    f"CPUNativeVNNIVerifierRowsPolicy::{entry.policy};",
                    "            if (measured_speedup)",
                    "                *measured_speedup = 0.0f;",
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
                "            static_cast<long long>(n) * "
                "static_cast<long long>(k);",
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
                    f"m == {rule.domain.m}",
                    aspect_condition(rule.domain.aspect_bucket),
                    *(
                        predicate_condition(predicate)
                        for predicate in rule.predicates
                    ),
                ]
                lines.extend(render_if_header(conditions, indent="        "))
                lines.extend([
                    "        {",
                    "            policy = "
                    "CPUNativeVNNIVerifierRowsPolicy::"
                    f"{_candidate_policy(rule.candidate_id)};",
                    "            if (measured_speedup)",
                    "                *measured_speedup = 0.0f;",
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


def write_summary(path: Path, entries: list[PolicyEntry]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "build_isa", "runtime_isa", "threads", "execution_codebook",
            "m", "n", "k", "policy", "candidate_id", "shape",
            "max_surface_regret", "max_cv",
        ])
        for entry in sorted(entries):
            writer.writerow([
                entry.build_isa, entry.runtime_isa, entry.threads,
                entry.codebook, entry.m, entry.n, entry.k, entry.policy,
                entry.candidate_id, entry.shape_name,
                f"{entry.max_surface_regret:.9f}", f"{entry.max_cv:.9f}",
            ])


def _context_from_args(
    args: argparse.Namespace,
    inputs: tuple[Path, ...],
    timing_sidecars: tuple[Path, ...],
    *,
    run_id: str | None = None,
) -> CPUVerifierAdapterContext:
    corpus_id = raw_corpus_id((*inputs, *timing_sidecars))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CPUVerifierAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return CPUVerifierAdapterContext(
        profile=profile,
        run_id=run_id or args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        serial_m1_policy_hash=args.serial_m1_policy_hash,
        raw_timing_sidecar_retained=bool(timing_sidecars),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input", nargs="+", type=Path, dest="input_options")
    parser.add_argument("--timing-sidecar", action="append", type=Path, default=[])
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
    parser.add_argument("--adapt-only", action="store_true")
    parser.add_argument("--frozen-policy-json", type=Path)
    parser.add_argument("--policy-json", type=Path)
    parser.add_argument("--development-profiler-requests", type=Path)
    parser.add_argument("--development-profiler-evidence", type=Path)
    parser.add_argument(
        "--development-profiler-observations",
        type=Path,
        help="Original common CSV bound to reusable profiler sidecars",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", "--summary-csv", dest="summary", type=Path)
    parser.add_argument("--common-observations", type=Path)
    parser.add_argument("--require-key", action="append", default=[])
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--require-isa-matrix", action="store_true")
    parser.add_argument(
        "--profile", choices=[profile.value for profile in MeasurementProfile],
        default=MeasurementProfile.QUICK.value,
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--git-revision", default="")
    parser.add_argument("--build-id", default="")
    parser.add_argument("--compiler-id", default="")
    parser.add_argument("--architecture-class", default="")
    parser.add_argument("--device-name", default="")
    parser.add_argument("--driver-runtime", default="")
    parser.add_argument("--serial-m1-policy-hash", default="")
    parser.add_argument(
        "--shape-manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Reviewed grouped-verifier shape applicability",
    )
    args = parser.parse_args()

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if args.freeze_generic and args.certify_generic:
        parser.error("--freeze-generic and --certify-generic are mutually exclusive")
    if args.adapt_only and (args.freeze_generic or args.certify_generic):
        parser.error("--adapt-only cannot freeze or certify a policy")
    if bool(args.development_profiler_requests) != bool(
        args.development_profiler_evidence
    ):
        parser.error(
            "development profiler requests and evidence are required together"
        )
    if args.development_profiler_observations and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "development profiler observations require requests and evidence"
        )
    if (args.freeze_generic or args.certify_generic) and not (
        args.development_profiler_requests
        and args.development_profiler_evidence
    ):
        parser.error(
            "production CPU freeze/certification requires complete "
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
        if not inputs or args.development_input or args.sealed_input:
            parser.error("--freeze-generic accepts development --input only")
        if not args.policy_json:
            parser.error("--freeze-generic requires --policy-json")
    elif args.certify_generic:
        if inputs or args.timing_sidecar:
            parser.error("--certify-generic accepts separate partition inputs only")
        if not args.development_input or not args.sealed_input:
            parser.error("certification requires development and sealed inputs")
        if not args.frozen_policy_json or not args.policy_json:
            parser.error("certification requires frozen and compiled policy JSON")
    elif separate_certification:
        parser.error("separate partition inputs require --certify-generic")
    elif not inputs:
        parser.error("at least one strong CPU trainer CSV is required")
    if (
        MeasurementProfile(args.profile) == MeasurementProfile.PRODUCTION
        and not args.freeze_generic
        and not args.certify_generic
        and not args.adapt_only
    ):
        parser.error(
            "production CPU policy emission requires development freeze and "
            "separate sealed certification"
        )

    manifest = load_shape_manifest(args.shape_manifest)
    if args.adapt_only:
        if not inputs or separate_certification:
            parser.error("--adapt-only accepts development --input only")
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = adapt_cpu_verifier_csv(
            inputs,
            context,
            timing_sidecars=timing_sidecars,
        )
        validate_complete(
            corpus,
            manifest,
            require_full_inventory=(
                context.profile == MeasurementProfile.PRODUCTION
            ),
            expected_partition=ShapePartition.DEVELOPMENT,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        validate_required_keys(entries, args.require_key)
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
        print(
            f"adapted {len(corpus)} CPU verifier development observations "
            f"without fitting -> {args.output}"
        )
        return 0

    if args.freeze_generic:
        timing_sidecars = tuple(args.timing_sidecar)
        context = _context_from_args(args, inputs, timing_sidecars)
        corpus = adapt_cpu_verifier_csv(
            inputs,
            context,
            timing_sidecars=timing_sidecars,
        )
        profiler_catalog = load_profiler_feature_catalog(
            corpus,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
                else None
            ),
        )
        frozen = freeze_cpu_verifier_policy(
            corpus,
            manifest,
            context.serial_m1_policy_hash,
            profiler_catalog,
        )
        entries = select_entries(corpus, context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(frozen.policy_ir.generic_rules)
        validate_emitter_inputs(frozen.policy_ir, entries, generic_rules)
        validate_required_keys(entries, args.require_key)
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
        if args.summary:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            write_summary(args.summary, entries)
        if args.common_observations:
            args.common_observations.parent.mkdir(parents=True, exist_ok=True)
            write_observation_csv(args.common_observations, corpus)
        print(
            f"froze {len(corpus)} CPU verifier development observations as "
            f"{frozen.generic_digest} -> {args.output}"
        )
        return 0

    if args.certify_generic:
        development_inputs = tuple(args.development_input)
        development_timing = tuple(args.development_timing_sidecar)
        development_context = _context_from_args(
            args,
            development_inputs,
            development_timing,
        )
        development = adapt_cpu_verifier_csv(
            development_inputs,
            development_context,
            timing_sidecars=development_timing,
        )
        profiler_catalog = load_profiler_feature_catalog(
            development,
            args.development_profiler_requests,
            args.development_profiler_evidence,
            source_corpus=(
                read_observation_csv((args.development_profiler_observations,))
                if args.development_profiler_observations
                else None
            ),
        )
        frozen = freeze_cpu_verifier_policy(
            development,
            manifest,
            development_context.serial_m1_policy_hash,
            profiler_catalog,
        )
        # This byte comparison is deliberately complete before the first
        # sealed CSV or timing sidecar is opened.
        validate_frozen_policy_file(args.frozen_policy_json, frozen)

        sealed_inputs = tuple(args.sealed_input)
        sealed_timing = tuple(args.sealed_timing_sidecar)
        sealed_context = _context_from_args(
            args,
            sealed_inputs,
            sealed_timing,
            run_id=f"{args.run_id}-sealed",
        )
        sealed = adapt_cpu_verifier_csv(
            sealed_inputs,
            sealed_context,
            timing_sidecars=sealed_timing,
        )
        compiled = certify_cpu_verifier_policy(
            frozen,
            development,
            sealed,
            manifest,
            development_context.serial_m1_policy_hash,
        )
        corpus = ObservationCorpus((
            *development.observations,
            *sealed.observations,
        ))
        entries = select_entries(corpus, development_context.serial_m1_policy_hash)
        generic_rules = _emit_generic_rules(compiled.policy_ir.generic_rules)
        validate_emitter_inputs(compiled.policy_ir, entries, generic_rules)
        validate_required_keys(entries, args.require_key)
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
            f"certified frozen CPU verifier policy {frozen.generic_digest} "
            f"against {len(sealed)} sealed observations -> {args.output}"
        )
        return 0

    timing_sidecars = tuple(args.timing_sidecar)
    context = _context_from_args(args, inputs, timing_sidecars)
    corpus = adapt_cpu_verifier_csv(
        inputs,
        context,
        timing_sidecars=timing_sidecars,
    )
    entries = select_entries(corpus, context.serial_m1_policy_hash)
    generic_rules = select_generic_rules(corpus, context.serial_m1_policy_hash)
    if args.require_complete:
        validate_complete(
            corpus,
            manifest,
            require_full_inventory=(
                args.profile == MeasurementProfile.PRODUCTION.value
            ),
        )
    elif args.require_isa_matrix:
        validate_isa_regime_matrix(corpus)
    validate_required_keys(entries, args.require_key)

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
        f"adapted {len(corpus)} strong CPU observations; generated "
        f"{len(entries)} exact policies -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
