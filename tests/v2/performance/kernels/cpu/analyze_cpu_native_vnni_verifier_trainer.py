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
    write_observation_csv,
)
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_verifier_registry,
)
from native_vnni_dispatch.corpus import ObservationCorpus, RuntimeKey  # noqa: E402
from native_vnni_dispatch.exact_oracle import build_exact_winners  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import AspectBucket, SemanticContract  # noqa: E402
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericDispatchRule,
    fit_generic_policy,
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


def validate_complete(corpus: ObservationCorpus) -> None:
    require_canonical_alias_coverage(corpus)
    require_verifier_m_matrix(corpus)
    require_candidate_matrix_complete(corpus)
    require_registry_candidate_coverage(
        corpus,
        cpu_native_vnni_verifier_registry(),
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


def _aspect_condition(bucket: AspectBucket) -> str:
    return {
        AspectBucket.VERY_WIDE: "aspect_ratio >= 16.0f",
        AspectBucket.WIDE: "aspect_ratio >= 2.0f",
        AspectBucket.BALANCED: "aspect_ratio >= 0.75f",
        AspectBucket.TALL: "true",
    }[bucket]


def generate_include(
    entries: list[PolicyEntry],
    generic_rules: list[CPUGenericDispatchRule],
    *,
    corpus_digest: str,
    registry_digest: str,
    profile: MeasurementProfile,
) -> str:
    """Render exact/generic decisions into an ISA-aware CPU selector."""

    lines = [
        "// Auto-generated by analyze_cpu_native_vnni_verifier_trainer.py. DO NOT EDIT.",
        "// Decisions: common NativeVNNI alias-robust exact oracle + segmented-regret learner.",
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
                "        const float aspect_ratio =",
                "            k > 0 ? static_cast<float>(n) / "
                "static_cast<float>(k) : 0.0f;",
                "        const long long work_items =",
                "            static_cast<long long>(n) * "
                "static_cast<long long>(k);",
            ])
            for item in sorted(
                group_rules,
                key=lambda value: (
                    value.rule.domain.runtime_codebook_id,
                    value.rule.domain.m,
                    list(AspectBucket).index(
                        value.rule.domain.aspect_bucket
                    ),
                    value.rule.min_work_items,
                ),
            ):
                rule = item.rule
                lines.extend([
                    "        if (codebook == "
                    f"{rule.domain.runtime_codebook_id} && "
                    f"m == {rule.domain.m} &&",
                    f"            {_aspect_condition(rule.domain.aspect_bucket)} &&",
                    f"            work_items >= {rule.min_work_items}LL &&",
                    f"            work_items <= {rule.max_work_items}LL)",
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
) -> CPUVerifierAdapterContext:
    corpus_id = raw_corpus_id((*inputs, *args.timing_sidecar))
    profile = MeasurementProfile(args.profile)
    if not profile.installable:
        return CPUVerifierAdapterContext.workflow_smoke(corpus_id=corpus_id)
    return CPUVerifierAdapterContext(
        profile=profile,
        run_id=args.run_id,
        corpus_id=corpus_id,
        git_revision=args.git_revision,
        build_id=args.build_id,
        compiler_id=args.compiler_id,
        architecture_class=args.architecture_class,
        device_name=args.device_name,
        driver_runtime=args.driver_runtime,
        serial_m1_policy_hash=args.serial_m1_policy_hash,
        raw_timing_sidecar_retained=bool(args.timing_sidecar),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input", nargs="+", type=Path, dest="input_options")
    parser.add_argument("--timing-sidecar", action="append", type=Path, default=[])
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
    args = parser.parse_args()

    positional = tuple(args.inputs)
    optional = tuple(args.input_options or ())
    if positional and optional:
        parser.error("use positional inputs or --input, not both")
    inputs = positional or optional
    if not inputs:
        parser.error("at least one strong CPU trainer CSV is required")

    context = _context_from_args(args, inputs)
    corpus = adapt_cpu_verifier_csv(
        inputs,
        context,
        timing_sidecars=args.timing_sidecar,
    )
    entries = select_entries(corpus, context.serial_m1_policy_hash)
    generic_rules = select_generic_rules(corpus, context.serial_m1_policy_hash)
    if args.require_complete:
        validate_complete(corpus)
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
