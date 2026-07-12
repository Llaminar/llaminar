#!/usr/bin/env python3
"""Regression tests for the common NativeVNNI dispatch policy compiler."""

from __future__ import annotations

import dataclasses
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.certification import certify_generic_policy  # noqa: E402
from native_vnni_dispatch.candidate_registry import (  # noqa: E402
    candidate_registry_digest,
    cpu_native_vnni_verifier_registry,
    cuda_native_vnni_gemv_registry,
    rocm_moe_grouped_prefill_registry,
    rocm_native_vnni_decode_registry,
)
from native_vnni_dispatch.compiler import compile_policy  # noqa: E402
from native_vnni_dispatch.corpus import (  # noqa: E402
    ObservationCorpus,
    SurfaceKey,
    generic_domain,
    runtime_key,
)
from native_vnni_dispatch.exact_oracle import (  # noqa: E402
    build_exact_winner,
    build_exact_winners,
    candidate_is_eligible,
)
from native_vnni_dispatch.format_registry import (  # noqa: E402
    FORMAT_SPECS,
    format_spec,
    registry_digest,
    runtime_aliases,
)
from native_vnni_dispatch.policy_ir import make_policy_ir  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    LEARNER_VERSION,
    NativeVNNIObservation,
    SemanticContract,
    classify_aspect,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    GenericPolicy,
    fit_generic_policy,
)
from native_vnni_dispatch.splits import (  # noqa: E402
    Partition,
    SplitManifest,
)
from native_vnni_dispatch.validation import (  # noqa: E402
    require_candidate_matrix_complete,
    require_verifier_m_matrix,
)
from native_vnni_dispatch.adapters.rocm_moe import (  # noqa: E402
    ROCmMoEAdapterContext,
    adapt_rocm_moe_csv,
    adapt_rocm_moe_row,
    raw_corpus_id,
    timing_sample_digest,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402


SERIAL_HASH = "sha256:serial-m1-policy-v1"


def observation(
    *,
    source_format: str = "Q4_0",
    candidate: str = "candidate.a",
    family: str = "family-a",
    shape_group: str = "shape-0",
    shape_name: str | None = None,
    n: int = 512,
    k: int = 2048,
    m: int = 2,
    latency_us: float = 10.0,
    p95_scale: float = 1.0,
    cv: float = 0.0,
    mode: ExecutionMode = ExecutionMode.GRAPH_CAPTURED,
    contract: SemanticContract = SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
    bitwise_equal: bool = True,
    repeat_equal: bool = True,
    uses_atomic: bool = False,
    route_ok: bool = True,
    generic_eligible: bool = True,
    serial_hash: str = SERIAL_HASH,
) -> NativeVNNIObservation:
    """Build one complete strict observation for compact synthetic corpora."""

    spec = format_spec(source_format)
    runtime_codebook = spec.runtime_codebook("rocm")
    mismatch_count = 0 if bitwise_equal else 1
    result = NativeVNNIObservation(
        schema_version=1,
        run_id="unit-run",
        corpus_id="sha256:unit-corpus",
        git_revision="0123456789abcdef",
        build_id="unit-release-build",
        compiler_id="unit-compiler",
        policy_abi=1,
        learner_version=LEARNER_VERSION,
        backend=Backend.ROCM,
        architecture_class="gfx906-native-vnni-v1",
        device_name="unit-gfx906",
        driver_runtime="unit-rocm",
        threading_or_stream_mode="explicit_non_default_stream",
        semantic_contract=contract,
        operation_kind="SingleProjection",
        bundle_signature=f"single:cb{runtime_codebook}:fp32:v1",
        projection_n_vector=(n,),
        source_format=spec.label,
        source_codebook_id=spec.source_codebook_id,
        prepared_family_id=spec.prepared_family("rocm"),
        packing_abi=spec.packing_abi("rocm"),
        runtime_codebook_id=runtime_codebook,
        shape_group_id=shape_group,
        shape_name=shape_name or shape_group,
        execution_mode=mode,
        m=m,
        aggregate_n=n,
        k=k,
        aspect_ratio=float(n) / float(k),
        aspect_bucket=classify_aspect(n, k),
        work_items=n * k,
        n_tail_class=f"n_mod_256={n % 256}",
        k_tail_class=f"k_mod_256={k % 256}",
        alignment_class="prepared_16b_aligned",
        candidate_id=candidate,
        effective_candidate_id=candidate,
        candidate_family=family,
        config_json={"candidate": candidate},
        supported=True,
        graph_capture_ok=True,
        generic_eligible=generic_eligible,
        arithmetic_fingerprint=f"sha256:arithmetic:{candidate}",
        serial_m1_policy_id="serial.m1.production",
        serial_m1_policy_hash=serial_hash,
        candidate_policy_hash=f"sha256:policy:{candidate}",
        ordered_reduction=not uses_atomic,
        uses_atomic_reduction=uses_atomic,
        trial_set_hash="sha256:unit-trials",
        numerical_correctness=True,
        bitwise_equal=bitwise_equal,
        repeat_equal=repeat_equal,
        mismatch_count=mismatch_count,
        first_mismatch_index=None if bitwise_equal else 7,
        grouped_output_digest=(
            "sha256:equal-output" if bitwise_equal else "sha256:one-ulp-output"
        ),
        serial_output_digest="sha256:equal-output",
        max_abs=0.0 if bitwise_equal else 1.0e-7,
        relative_l2=0.0 if bitwise_equal else 1.0e-9,
        cosine=1.0,
        symmetric_kld=0.0,
        warmup_count=5,
        sample_count=30,
        min_us=latency_us,
        median_us=latency_us,
        p95_us=latency_us * p95_scale,
        mad_us=latency_us * cv * 0.67,
        cv=cv,
        timing_sample_hash=f"sha256:timings:{candidate}:{shape_group}:{source_format}:{mode.value}",
        effective_bandwidth_gbs=100.0,
        forced_route_ok=route_ok,
        observed_candidate_id=candidate,
        route_counter_ok=route_ok,
        workspace_ok=True,
        explicit_stream_ok=True,
    )
    result.validate()
    return result


def candidate_rows_for_aliases(
    candidate: str,
    latencies: dict[str, float],
    **kwargs,
) -> list[NativeVNNIObservation]:
    """Create one graph-captured row for each named source-format alias."""

    return [
        observation(
            source_format=source_format,
            candidate=candidate,
            family=candidate,
            latency_us=latency,
            **kwargs,
        )
        for source_format, latency in latencies.items()
    ]


class NativeVNNICommonDispatchPolicyTest(unittest.TestCase):
    """Prove numerical filtering, robust regret, and sealed policy behavior."""

    def test_registry_matches_full_quantized_inventory_and_q8_alias_semantics(self) -> None:
        self.assertEqual(len(FORMAT_SPECS), 21)
        self.assertEqual(format_spec("Q8_K").source_codebook_id, 21)
        self.assertEqual(format_spec("Q8_K").runtime_codebook("rocm"), 19)
        self.assertEqual(format_spec("Q8_1").runtime_codebook("cuda"), 19)
        self.assertEqual(format_spec("Q8_1").runtime_codebook("cpu"), 20)
        self.assertEqual(runtime_aliases("rocm", 19), ("Q8_0", "Q8_1", "Q8_K"))
        self.assertEqual(registry_digest(), registry_digest())

    def test_candidate_registries_are_explicit_complete_and_digestible(self) -> None:
        rocm_moe = rocm_moe_grouped_prefill_registry()
        rocm_decode = rocm_native_vnni_decode_registry()
        cpu = cpu_native_vnni_verifier_registry()
        cuda = cuda_native_vnni_gemv_registry()

        self.assertEqual(len(rocm_moe.entries), 12)
        self.assertEqual(len(rocm_decode.entries), 65)
        self.assertEqual(len(cpu.entries), 2)
        self.assertEqual(len(cuda.entries), 1801)
        self.assertTrue(candidate_registry_digest().startswith("sha256:"))
        with self.assertRaisesRegex(ValueError, "not an explicit candidate"):
            rocm_decode.resolve("AUTO")

        rocm_fast = rocm_decode.resolve("KB32")
        self.assertEqual(rocm_fast, rocm_decode.resolve("KB32/TW24"))
        self.assertTrue(rocm_fast.supports_contract(SemanticContract.FAST))
        self.assertFalse(rocm_fast.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        rocm_verifier = rocm_decode.resolve("INHERIT_SERIAL_M1")
        self.assertFalse(rocm_verifier.supports_contract(SemanticContract.FAST))
        self.assertTrue(rocm_verifier.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))

        public_m1 = cuda.resolve(
            "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        )
        self.assertTrue(public_m1.supports_contract(SemanticContract.FAST))
        self.assertFalse(public_m1.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        self.assertEqual(public_m1.config_json["target_waves"], 0)
        self.assertEqual(public_m1.config_json["exact_kb"], 8)
        self.assertEqual(public_m1.config_json["force_two_phase"], 1)
        self.assertTrue(public_m1.ordered_reduction)
        self.assertFalse(public_m1.uses_atomic_reduction)

        cuda_verifier = cuda.resolve("INHERIT_SERIAL_M1")
        self.assertFalse(cuda_verifier.supports_contract(SemanticContract.FAST))
        self.assertTrue(cuda_verifier.supports_contract(
            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
        ))
        self.assertTrue(cuda_verifier.ordered_reduction)
        self.assertFalse(cuda_verifier.uses_atomic_reduction)
        with self.assertRaisesRegex(ValueError, "unknown forceable candidate"):
            cuda.resolve("cuda.nvnni.kpar.tn256.cpt4.tw8.mkg4.kb4.phase2")

    @staticmethod
    def rocm_moe_raw_row(**overrides) -> dict[str, str]:
        """Build one strong-evidence row from the production ROCm speedometer."""

        row = {
            "backend": "rocm",
            "phase": "grouped_prefill",
            "source_format": "Q8_K",
            "source_codebook": "21",
            "execution_codebook": "19",
            "shape": "gate_ratio_1_4",
            "role": "gateup",
            "candidate_id": "tm12_tn256",
            "m": "32",
            "n": "512",
            "k": "2048",
            "tile_m": "12",
            "tile_n": "256",
            "warmup_count": "2",
            "sample_count": "3",
            "timed_replays": "12",
            "min_us": "330.0",
            "graph_us": "331.0",
            "p95_us": "333.0",
            "mad_us": "1.0",
            "cv": "0.004",
            "pipeline_gops": "4863.0",
            "bit_mismatches": "0",
            "first_bit_mismatch": "0",
            "repeat_bit_mismatches": "0",
            "max_abs": "0",
            "relative_l2": "0",
            "cosine": "1",
            "symmetric_kld": "0",
            "grouped_output_digest": "fnv1a64:equal",
            "serial_output_digest": "fnv1a64:equal",
            "timing_sample_digest": "fnv1a64:timings",
            "route_counter_ok": "1",
            "observed_candidate_id": "tm12_tn256",
            "is_winner": "1",
        }
        row.update({name: str(value) for name, value in overrides.items()})
        return row

    def test_rocm_moe_adapter_preserves_alias_bundle_and_route_proof(self) -> None:
        context = ROCmMoEAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "1" * 64
        )
        result = adapt_rocm_moe_row(self.rocm_moe_raw_row(), context)

        self.assertEqual(result.source_codebook_id, 21)
        self.assertEqual(result.runtime_codebook_id, 19)
        self.assertEqual(result.projection_n_vector, (512, 512))
        self.assertEqual(result.aggregate_n, 1024)
        self.assertEqual(
            result.candidate_id,
            "rocm.moe.grouped_prefill.tm12.tn256",
        )
        self.assertEqual(result.observed_candidate_id, result.effective_candidate_id)
        self.assertTrue(result.forced_route_ok)
        self.assertTrue(result.bitwise_equal)
        self.assertTrue(result.repeat_equal)

    def test_rocm_moe_adapter_retains_failed_route_as_ineligible_evidence(self) -> None:
        context = ROCmMoEAdapterContext.workflow_smoke(
            corpus_id="sha256:" + "2" * 64
        )
        result = adapt_rocm_moe_row(
            self.rocm_moe_raw_row(route_counter_ok=0, observed_candidate_id="missing"),
            context,
        )
        self.assertFalse(result.forced_route_ok)
        self.assertFalse(candidate_is_eligible(result, context.serial_m1_policy_hash))

    def test_installable_rocm_moe_profile_rejects_smoke_timing(self) -> None:
        context = ROCmMoEAdapterContext(
            profile=MeasurementProfile.PRODUCTION,
            run_id="run-20260711",
            corpus_id="sha256:" + "3" * 64,
            git_revision="0123456789abcdef",
            build_id="release-20260711",
            compiler_id="hip-clang-7.1",
            architecture_class="gfx906-native-vnni-v1",
            device_name="AMD-Instinct-MI50",
            driver_runtime="ROCm-7.1",
            serial_m1_policy_hash=SERIAL_HASH,
            raw_timing_sidecar_retained=True,
        )
        context.validate()
        with self.assertRaisesRegex(ValueError, "at least 5 warmups/30 samples"):
            adapt_rocm_moe_row(self.rocm_moe_raw_row(), context)

    def test_rocm_moe_raw_timing_sidecar_is_digest_and_statistic_checked(self) -> None:
        samples_ms = (0.330, 0.331, 0.332)
        mean_ms = sum(samples_ms) / len(samples_ms)
        cv = math.sqrt(
            sum((value - mean_ms) ** 2 for value in samples_ms) / len(samples_ms)
        ) / mean_ms
        aggregate = self.rocm_moe_raw_row(
            min_us=330.0,
            graph_us=331.0,
            p95_us=332.0,
            cv=f"{cv:.6f}",
            timing_sample_digest=timing_sample_digest(samples_ms),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            aggregate_path = root / "aggregate.csv"
            timing_path = root / "timing.csv"
            with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(aggregate))
                writer.writeheader()
                writer.writerow(aggregate)

            timing_fields = (
                "backend", "phase", "source_format", "source_codebook",
                "execution_codebook", "shape", "role", "candidate_id", "m",
                "n", "k", "tile_m", "tile_n", "sample_index", "timed_replays",
                "latency_us", "latency_ms_hex",
            )
            with timing_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=timing_fields)
                writer.writeheader()
                for index, value in enumerate(samples_ms):
                    writer.writerow({
                        **{name: aggregate[name] for name in timing_fields
                           if name in aggregate},
                        "sample_index": index,
                        "timed_replays": 4,
                        "latency_us": f"{value * 1000.0:.9f}",
                        "latency_ms_hex": value.hex(),
                    })

            context = ROCmMoEAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate_path, timing_path))
            )
            corpus = adapt_rocm_moe_csv(
                (aggregate_path,), context, timing_sidecars=(timing_path,)
            )
            self.assertEqual(len(corpus), 1)
            self.assertEqual(next(iter(corpus)).timing_sample_hash,
                             timing_sample_digest(samples_ms))

            broken = dict(aggregate)
            broken["timing_sample_digest"] = "fnv1a64:0000000000000000"
            with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(broken))
                writer.writeheader()
                writer.writerow(broken)
            with self.assertRaisesRegex(ValueError, "digest does not match"):
                adapt_rocm_moe_csv(
                    (aggregate_path,), context, timing_sidecars=(timing_path,)
                )

    def test_schema_rejects_missing_field_and_auto_candidate(self) -> None:
        raw = observation().canonical_mapping()
        raw.pop("route_counter_ok")
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            NativeVNNIObservation.from_mapping(raw)

        with self.assertRaisesRegex(ValueError, "AUTO"):
            dataclasses.replace(observation(), candidate_id="AUTO").validate()

    def test_one_ulp_candidate_is_ineligible_despite_perfect_diagnostics(self) -> None:
        one_ulp = observation(
            candidate="candidate.one_ulp",
            bitwise_equal=False,
            latency_us=1.0,
        )
        safe = observation(candidate="candidate.safe", latency_us=2.0)

        self.assertFalse(candidate_is_eligible(one_ulp, SERIAL_HASH))
        winner = build_exact_winner([one_ulp, safe], current_serial_m1_hash=SERIAL_HASH)
        self.assertEqual(winner.candidate_id, "candidate.safe")

    def test_atomic_and_stale_oracle_candidates_are_filtered(self) -> None:
        atomic = observation(candidate="candidate.atomic", uses_atomic=True)
        stale = observation(candidate="candidate.stale", serial_hash="sha256:old")
        self.assertFalse(candidate_is_eligible(atomic, SERIAL_HASH))
        self.assertFalse(candidate_is_eligible(stale, SERIAL_HASH))

    def test_opposing_alias_winners_choose_worst_surface_robust_candidate(self) -> None:
        rows = []
        rows += candidate_rows_for_aliases(
            "candidate.a", {"Q4_1": 1.0, "Q4_K": 2.0}
        )
        rows += candidate_rows_for_aliases(
            "candidate.b", {"Q4_1": 2.0, "Q4_K": 1.0}
        )
        rows += candidate_rows_for_aliases(
            "candidate.compromise", {"Q4_1": 1.2, "Q4_K": 1.2}
        )

        winner = build_exact_winner(rows, current_serial_m1_hash=SERIAL_HASH)
        self.assertEqual(winner.candidate_id, "candidate.compromise")
        self.assertAlmostEqual(winner.max_surface_regret, 0.2)

    def test_missing_alias_surface_cannot_win_or_quietly_fallback(self) -> None:
        rows = candidate_rows_for_aliases(
            "candidate.partial", {"Q4_1": 1.0}
        )
        required = frozenset({
            SurfaceKey("Q4_1", ExecutionMode.GRAPH_CAPTURED),
            SurfaceKey("Q4_K", ExecutionMode.GRAPH_CAPTURED),
        })
        with self.assertRaisesRegex(ValueError, "no eligible candidate"):
            build_exact_winner(
                rows,
                required_surfaces=required,
                current_serial_m1_hash=SERIAL_HASH,
            )

    def test_regret_learner_rejects_high_modal_accuracy_with_one_large_miss(self) -> None:
        rows = []
        for index in range(100):
            group = f"shape-{index:03d}"
            a_latency = 1.0 if index < 97 else 1.40
            rows.append(observation(
                candidate="candidate.modal",
                family="modal",
                shape_group=group,
                latency_us=a_latency,
            ))
            rows.append(observation(
                candidate="candidate.robust",
                family="robust",
                shape_group=group,
                latency_us=1.02,
            ))
        policy = fit_generic_policy(ObservationCorpus(rows), max_segments=1)
        self.assertEqual(len(policy.rules), 1)
        self.assertEqual(policy.rules[0].candidate_id, "candidate.robust")
        self.assertLessEqual(policy.rules[0].development_max_regret, 0.02 + 1.0e-12)

    def test_low_winner_label_accuracy_can_pass_when_all_regret_is_below_three_pct(self) -> None:
        rows = []
        for index in range(8):
            group = f"alternating-{index}"
            rows.extend([
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=group,
                    latency_us=1.0 if index % 2 == 0 else 1.02,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=group,
                    latency_us=1.02 if index % 2 == 0 else 1.0,
                ),
                observation(
                    candidate="candidate.never_exact",
                    family="stable",
                    shape_group=group,
                    latency_us=1.01,
                ),
            ])
        policy = fit_generic_policy(ObservationCorpus(rows), max_segments=1)
        self.assertEqual(policy.rules[0].candidate_id, "candidate.never_exact")
        self.assertLess(policy.rules[0].development_max_regret, 0.03)

    def test_single_shape_domain_remains_exact_only(self) -> None:
        corpus = ObservationCorpus([
            observation(candidate="candidate.a"),
            observation(candidate="candidate.b", latency_us=11.0),
        ])
        policy = fit_generic_policy(corpus)
        self.assertFalse(policy.rules)
        self.assertEqual(len(policy.unpromoted_domains), 1)

    def test_grouped_cv_selects_two_segments_and_uses_midpoint_boundary(self) -> None:
        rows = []
        lower_ns = (128, 160, 192)
        upper_ns = (256, 320, 384)
        for index, n in enumerate((*lower_ns, *upper_ns)):
            group = f"cv-shape-{index}"
            lower = n in lower_ns
            rows.extend([
                observation(
                    candidate="candidate.low",
                    family="low",
                    shape_group=group,
                    n=n,
                    latency_us=10.0 if lower else 14.0,
                ),
                observation(
                    candidate="candidate.high",
                    family="high",
                    shape_group=group,
                    n=n,
                    latency_us=14.0 if lower else 10.0,
                ),
            ])

        policy = fit_generic_policy(ObservationCorpus(rows), max_segments=2)
        self.assertEqual(len(policy.rules), 2)
        self.assertEqual(len(policy.cross_validation), 1)
        cv = policy.cross_validation[0]
        self.assertEqual(cv.selected_max_segments, 2)
        self.assertEqual(cv.fold_count, 6)
        self.assertEqual(cv.covered_point_count, cv.required_point_count)
        expected_boundary = ((192 * 2048) + (256 * 2048)) // 2
        self.assertEqual(policy.rules[0].max_work_items, expected_boundary)
        self.assertEqual(policy.rules[1].min_work_items, expected_boundary + 1)

    def test_shape_group_split_never_leaks_candidate_alias_or_m_rows(self) -> None:
        rows = []
        for group in ("development-a", "development-b", "sealed-c"):
            for m in (2, 3, 4):
                for source_format in ("Q4_1", "Q4_K"):
                    rows.append(observation(
                        source_format=source_format,
                        candidate="candidate.a",
                        shape_group=group,
                        m=m,
                    ))
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "development-a": Partition.DEVELOPMENT,
                "development-b": Partition.DEVELOPMENT,
                "sealed-c": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        self.assertEqual(set(development.shape_groups()), {"development-a", "development-b"})
        self.assertFalse(sealed.opened)
        opened = sealed.open("sha256:frozen-generic")
        self.assertEqual(opened.shape_groups(), ("sealed-c",))

    def test_compile_freezes_generic_before_sealed_and_does_not_refit(self) -> None:
        rows = []
        for group, n in (
            ("dev-a", 256),
            ("dev-b", 384),
            ("dev-c", 448),
            ("sealed", 512),
        ):
            rows.extend([
                observation(
                    candidate="candidate.a",
                    family="a",
                    shape_group=group,
                    n=n,
                    latency_us=10.0,
                ),
                observation(
                    candidate="candidate.b",
                    family="b",
                    shape_group=group,
                    n=n,
                    latency_us=11.0,
                ),
            ])
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "dev-a": Partition.DEVELOPMENT,
                "dev-b": Partition.DEVELOPMENT,
                "dev-c": Partition.DEVELOPMENT,
                "sealed": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        compiled = compile_policy(development, sealed)
        frozen = compiled.certification.frozen_generic_policy_digest
        self.assertEqual(compiled.policy_ir.digest(generic_only=True), frozen)
        self.assertEqual(compiled.certification.coverage, 1.0)
        self.assertEqual(compiled.certification.max_observed_regret, 0.0)

    def test_installable_compiler_rejects_short_timing_corpus(self) -> None:
        rows = []
        for group, n in (("dev-a", 256), ("dev-b", 384), ("sealed", 512)):
            rows.extend([
                dataclasses.replace(
                    observation(
                        candidate="candidate.a",
                        shape_group=group,
                        n=n,
                        latency_us=10.0,
                    ),
                    warmup_count=1,
                    sample_count=2,
                ),
                dataclasses.replace(
                    observation(
                        candidate="candidate.b",
                        shape_group=group,
                        n=n,
                        latency_us=11.0,
                    ),
                    warmup_count=1,
                    sample_count=2,
                ),
            ])
        corpus = ObservationCorpus(rows)
        manifest = SplitManifest(
            version="unit-split-v1",
            seed="unit",
            assignments={
                "dev-a": Partition.DEVELOPMENT,
                "dev-b": Partition.DEVELOPMENT,
                "sealed": Partition.SEALED,
            },
        )
        development, sealed = manifest.partition(corpus)
        with self.assertRaisesRegex(ValueError, "non-promotable timing"):
            compile_policy(development, sealed)

    def test_exact_overlay_cannot_mask_broken_generic_policy(self) -> None:
        development = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="dev-a", latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-a", latency_us=14.0),
            observation(candidate="candidate.fast", shape_group="dev-b", n=384, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-b", n=384, latency_us=14.0),
            observation(candidate="candidate.fast", shape_group="dev-c", n=448, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="dev-c", n=448, latency_us=14.0),
        ])
        exact = build_exact_winners(development)
        generic = fit_generic_policy(development, max_segments=1)
        poisoned_rule = dataclasses.replace(
            generic.rules[0], candidate_id="candidate.slow"
        )
        poisoned = make_policy_ir(
            exact,
            GenericPolicy(rules=(poisoned_rule,), unpromoted_domains=()),
            metadata={"test": "exact overlay must be bypassed"},
        )
        sealed = ObservationCorpus([
            observation(candidate="candidate.fast", shape_group="sealed", n=512, latency_us=10.0),
            observation(candidate="candidate.slow", shape_group="sealed", n=512, latency_us=14.0),
        ])
        report = certify_generic_policy(poisoned, sealed)
        self.assertAlmostEqual(report.max_observed_regret, 0.4)
        with self.assertRaisesRegex(ValueError, "not promotable"):
            report.require_promotable()

    def test_inventory_validators_reject_missing_m_and_candidate_surface(self) -> None:
        corpus = ObservationCorpus([
            observation(source_format="Q4_1", candidate="candidate.a", m=2),
            observation(source_format="Q4_K", candidate="candidate.b", m=2),
        ])
        with self.assertRaisesRegex(ValueError, "M matrix incomplete"):
            require_verifier_m_matrix(corpus)
        with self.assertRaisesRegex(ValueError, "candidate matrix"):
            require_candidate_matrix_complete(corpus)


if __name__ == "__main__":
    unittest.main()
