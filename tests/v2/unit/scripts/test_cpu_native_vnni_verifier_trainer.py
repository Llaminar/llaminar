#!/usr/bin/env python3
"""Regression tests for the common-backed CPU verifier policy analyzer."""

from __future__ import annotations

import csv
import dataclasses
import importlib.util
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.adapters.cpu_verifier import (  # noqa: E402
    CPUVerifierAdapterContext,
    adapt_cpu_verifier_csv,
    adapt_cpu_verifier_csv_to_common,
    adapt_cpu_verifier_row,
    read_cpu_verifier_timing_sidecars,
    raw_corpus_id,
)
from native_vnni_dispatch.adapters.evidence import (  # noqa: E402
    native_double_digest,
)
from native_vnni_dispatch.candidate_observation import (  # noqa: E402
    write_observation_csv,
)
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
    NativeVNNIObservation,
    SemanticContract,
)
from native_vnni_dispatch.segmented_policy import (  # noqa: E402
    FeatureAxis,
    FeaturePredicate,
    FeatureThreshold,
    GenericDispatchRule,
)

ANALYZER = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "cpu"
    / "analyze_cpu_native_vnni_verifier_trainer.py"
)


class CPUNativeVNNIVerifierTrainerTest(unittest.TestCase):
    """Exercise byte eligibility and common exact winner emission."""

    @staticmethod
    def row(
        candidate: str,
        median_us: float,
        *,
        mismatches: int = 0,
        build_isa: str = "AVX512",
        runtime_isa: str = "AVX512",
    ) -> dict[str, object]:
        digest = (
            "fnv1a64:different000000"
            if mismatches else "fnv1a64:equal00000000000"
        )
        return {
            "backend": "cpu",
            "phase": "verifier_rows",
            "source_format": "Q4_0",
            "source_codebook": 0,
            "execution_codebook": 0,
            "shape": "Qwen36_FFN_GateUp",
            "execution_mode": "eager",
            "m": 3,
            "n": 17408,
            "k": 5120,
            "candidate_id": candidate,
            "build_isa": build_isa,
            "runtime_isa_requested": runtime_isa,
            "runtime_isa_effective": runtime_isa,
            "threads": 28,
            "weight_bytes": 1000000,
            "warmup_count": 2,
            "sample_count": 3,
            "min_us": median_us,
            "median_us": median_us,
            "p95_us": median_us,
            "mad_us": 0,
            "cv": 0.01,
            "serial_median_us": 20,
            "speedup": 20 / median_us,
            "bit_mismatches": mismatches,
            "first_bit_mismatch": 7 if mismatches else 0,
            "repeat_byte_mismatches": 0,
            "max_abs": 1.0e-6 if mismatches else 0,
            "relative_l2": 1.0e-7 if mismatches else 0,
            "cosine": 1,
            "symmetric_kld": 0,
            "grouped_output_digest": digest,
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": f"fnv1a64:{candidate.lower():0<16}"[:24],
            "route_counter_ok": 1,
            "observed_candidate_id": candidate,
            "k_tiles": 1,
            "n_block_chunks": 2,
            "numerical_correctness": 1,
            "correctness_pass": 0 if mismatches else 1,
            "is_winner": 0,
        }

    def run_analyzer(
        self,
        rows: list[dict[str, object]],
        *extra: str,
    ) -> tuple[subprocess.CompletedProcess[str], str, list[dict[str, str]]]:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "trainer.csv"
            output = root / "generated.inc"
            summary = root / "summary.csv"
            common = root / "common.csv"
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    str(input_csv),
                    "--output",
                    str(output),
                    "--summary",
                    str(summary),
                    "--common-observations",
                    str(common),
                    *extra,
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            generated = output.read_text(encoding="utf-8") if output.exists() else ""
            summary_rows = []
            if summary.exists():
                with summary.open(newline="", encoding="utf-8") as handle:
                    summary_rows = list(csv.DictReader(handle))
            return result, generated, summary_rows

    def test_fastest_byte_exact_policy_is_emitted_through_common_oracle(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 6.0),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("common NativeVNNI alias-robust exact oracle", generated)
        self.assertIn(
            "kCPUNativeVNNIVerifierExactAVX512AVX512T28Policies[]",
            generated,
        )
        self.assertRegex(generated, r"Policies\[\]\s*=\s*\{\s*1,\s*\}")
        self.assertEqual(summary[0]["policy"], "WideRows")

    def test_inexact_faster_candidate_is_filtered_before_selection(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 5.0, mismatches=1),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(summary[0]["policy"], "Pairwise")
        self.assertRegex(generated, r"Policies\[\]\s*=\s*\{\s*0,\s*\}")

    def test_build_and_runtime_isa_regimes_train_independent_rules(self) -> None:
        result, generated, summary = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 6.0),
            self.row(
                "Pairwise",
                8.0,
                build_isa="AVX512",
                runtime_isa="AVX2",
            ),
            self.row(
                "Pairwise",
                9.0,
                build_isa="AVX2",
                runtime_isa="AVX2",
            ),
        ])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CPUNativeVNNIBuildISA::AVX512", generated)
        self.assertIn("CPUNativeVNNIRuntimeISA::AVX2", generated)
        self.assertIn("CPUNativeVNNIBuildISA::AVX2", generated)
        policies = {
            (row["build_isa"], row["runtime_isa"]): row["policy"]
            for row in summary
        }
        self.assertEqual(policies[("AVX512", "AVX512")], "WideRows")
        self.assertEqual(policies[("AVX512", "AVX2")], "Pairwise")
        self.assertEqual(policies[("AVX2", "AVX2")], "Pairwise")

    def test_geometry_aliases_share_one_exact_overlay(self) -> None:
        """Reviewed labels with identical runtime dimensions are aliases."""

        first = self.row("Pairwise", 10.0)
        first["shape"] = "ReleaseShape"
        second = self.row("Pairwise", 10.0)
        second["shape"] = "SyntheticWitnessSameGeometry"

        result, generated, summary = self.run_analyzer([first, second])

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(summary), 1)
        self.assertEqual(
            summary[0]["shape_names"],
            "ReleaseShape;SyntheticWitnessSameGeometry",
        )
        packed = (
            (0 << 56)
            | (3 << 48)
            | (5120 << 24)
            | 17408
        )
        self.assertEqual(generated.count(f"0x{packed:016x}ULL"), 1)
        self.assertNotIn("ReleaseShape", generated)
        self.assertIn("selectCPUNativeVNNIVerifierRowsExactPolicy", generated)

    def test_compact_exact_overlay_table_compiles_and_selects_winner(self) -> None:
        """The compact key/policy-byte representation preserves precedence."""

        result, generated, _ = self.run_analyzer([
            self.row("Pairwise", 10.0),
            self.row("WideRows", 6.0),
        ])
        self.assertEqual(result.returncode, 0, result.stderr)

        source = "\n".join((
            generated,
            "int main() {",
            "  using namespace llaminar2::cpu::native_vnni::generated;",
            "  CPUNativeVNNIVerifierRowsPolicy policy{};",
            "  const bool selected =",
            "      selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            "          CPUNativeVNNIBuildISA::AVX512,",
            "          CPUNativeVNNIRuntimeISA::AVX512,",
            "          28, 0, 3, 17408, 5120, 0, policy);",
            "  return selected &&",
            "         policy == CPUNativeVNNIVerifierRowsPolicy::WideRows ? 0 : 1;",
            "}",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "compact-exact-overlay"
            compiled = subprocess.run(
                ["g++", "-std=c++20", "-x", "c++", "-o", str(binary), "-"],
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr.decode())

    def test_require_complete_rejects_missing_depths_and_candidates(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("Pairwise", 10.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("verifier M matrix incomplete", result.stderr)

    def test_production_one_pass_emission_is_forbidden(self) -> None:
        """Coverage-only fitting must never produce an installable CPU table."""

        result, generated, _ = self.run_analyzer(
            [self.row("Pairwise", 10.0)],
            "--profile",
            "production",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(generated)
        self.assertIn(
            "requires development freeze and separate sealed certification",
            result.stderr,
        )

    def test_required_isa_matrix_rejects_single_build_corpus(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("Pairwise", 10.0)],
            "--require-isa-matrix",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CPU ISA regime matrix incomplete", result.stderr)

    def test_relaxed_legacy_csv_is_rejected(self) -> None:
        row = self.row("Pairwise", 10.0)
        del row["bit_mismatches"]

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong CPU verifier columns", result.stderr)

    def test_installable_timing_floor_is_explicit_and_overrideable(self) -> None:
        """A reviewed best-effort floor applies to the exact raw samples."""

        samples = (10.0, 10.0, 10.0)
        raw = {
            key: str(value)
            for key, value in self.row("Pairwise", 10.0).items()
        }
        raw["cv"] = "0.0"
        raw["timing_sample_digest"] = native_double_digest(samples)
        production = CPUVerifierAdapterContext(
            profile=MeasurementProfile.PRODUCTION,
            run_id="unit-production",
            corpus_id="sha256:" + "1" * 64,
            git_revision="deadbeef",
            build_id="sha256:" + "2" * 64,
            compiler_id="gcc-13",
            architecture_class="x86_64-unit",
            device_name="unit-cpu",
            driver_runtime="linux-unit",
            serial_m1_policy_hash="sha256:" + "3" * 64,
            raw_timing_sidecar_retained=True,
        )

        with self.assertRaisesRegex(ValueError, "requires 5/30 timing"):
            adapt_cpu_verifier_row(raw, production, samples)

        best_effort = dataclasses.replace(
            production,
            minimum_promotion_warmups=2,
            minimum_promotion_samples=3,
        )
        best_effort.validate()
        observation = adapt_cpu_verifier_row(raw, best_effort, samples)
        self.assertEqual(observation.warmup_count, 2)
        self.assertEqual(observation.sample_count, 3)
        self.assertEqual(observation.launch_k_tiles, 1)
        self.assertEqual(observation.launch_n_block_chunks, 2)

    def test_reused_grouped_common_requires_exact_raw_provenance(self) -> None:
        """A checkpoint may skip adaptation but may not change generations."""

        spec = importlib.util.spec_from_file_location(
            "cpu_native_vnni_grouped_reuse_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "trainer.csv"
            common_csv = root / "common.csv"
            rows = [self.row("Pairwise", 10.0), self.row("WideRows", 6.0)]
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            context = CPUVerifierAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((input_csv,)),
            )
            corpus = adapt_cpu_verifier_csv((input_csv,), context)
            write_observation_csv(common_csv, corpus)

            replay = module._load_reused_development_common(
                common_csv,
                context,
            )
            self.assertEqual(replay.digest(), corpus.digest())
            with self.assertRaisesRegex(
                ValueError,
                "changed raw/build provenance fields.*run_id",
            ):
                module._load_reused_development_common(
                    common_csv,
                    dataclasses.replace(context, run_id="another-generation"),
                )

            strict_production = dataclasses.replace(
                context,
                profile=MeasurementProfile.PRODUCTION,
                run_id="unit-production",
                git_revision="deadbeef",
                build_id="sha256:" + "2" * 64,
                compiler_id="gcc-13",
                architecture_class="x86_64-unit",
                device_name="unit-cpu",
                driver_runtime="linux-unit",
                raw_timing_sidecar_retained=True,
            )
            production_rows = [
                dataclasses.replace(
                    row,
                    run_id=strict_production.run_id,
                    git_revision=strict_production.git_revision,
                    build_id=(
                        f"{strict_production.build_id}|cpu_isa=AVX512"
                    ),
                    compiler_id=strict_production.compiler_id,
                    architecture_class=(
                        f"{strict_production.architecture_class}|build=AVX512|"
                        "runtime=AVX512|threads=28"
                    ),
                    device_name=strict_production.device_name,
                    driver_runtime=strict_production.driver_runtime,
                )
                for row in corpus
            ]
            write_observation_csv(
                common_csv,
                type(corpus)(tuple(production_rows)),
            )
            with self.assertRaisesRegex(ValueError, "requires 5/30 timing"):
                module._load_reused_development_common(
                    common_csv,
                    strict_production,
                )

    def test_parallel_timing_and_adaptation_match_serial_bytes(self) -> None:
        """Grouped worker paths preserve observations and canonical bytes."""

        samples = (10.0, 10.0, 10.0)
        rows = []
        timing_rows = []
        for index in range(8):
            raw = self.row("Pairwise", 10.0)
            raw["shape"] = f"UnitGrouped{index}"
            raw["n"] = 17408 + 32 * index
            raw["sample_count"] = len(samples)
            raw["cv"] = 0.0
            raw["timing_sample_digest"] = native_double_digest(samples)
            rows.append(raw)
            for sample_index, latency in enumerate(samples):
                timing_rows.append({
                    "backend": raw["backend"],
                    "phase": raw["phase"],
                    "source_format": raw["source_format"],
                    "source_codebook": raw["source_codebook"],
                    "execution_codebook": raw["execution_codebook"],
                    "shape": raw["shape"],
                    "execution_mode": raw["execution_mode"],
                    "m": raw["m"],
                    "n": raw["n"],
                    "k": raw["k"],
                    "candidate_id": raw["candidate_id"],
                    "build_isa": raw["build_isa"],
                    "runtime_isa_requested": raw["runtime_isa_requested"],
                    "runtime_isa_effective": raw["runtime_isa_effective"],
                    "sample_index": sample_index,
                    "latency_us": f"{latency:.9f}",
                    "latency_us_hex": latency.hex(),
                })

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            aggregate = root / "aggregate.csv"
            timing = root / "timing.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with timing.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=tuple(timing_rows[0]),
                )
                writer.writeheader()
                writer.writerows(timing_rows)

            serial_timing = read_cpu_verifier_timing_sidecars(
                (timing,), workers=1
            )
            parallel_timing = read_cpu_verifier_timing_sidecars(
                (timing,), workers=2, parallel_threshold_bytes=0
            )
            self.assertEqual(parallel_timing, serial_timing)

            context = CPUVerifierAdapterContext.workflow_smoke(
                corpus_id=raw_corpus_id((aggregate,)),
            )
            serial = adapt_cpu_verifier_csv(
                (aggregate,),
                context,
                timing_sidecars=(timing,),
            )
            expected = root / "expected.csv"
            write_observation_csv(expected, serial, workers=1)
            actual = root / "actual.csv"
            with mock.patch.object(
                NativeVNNIObservation,
                "from_mapping",
                side_effect=AssertionError(
                    "parallel grouped adapter reparsed its canonical CSV"
                ),
            ):
                parallel = adapt_cpu_verifier_csv_to_common(
                    (aggregate,),
                    context,
                    actual,
                    timing_sidecars=(timing,),
                    workers=2,
                    parallel_threshold=0,
                )

            self.assertEqual(parallel.observations, serial.observations)
            self.assertEqual(actual.read_bytes(), expected.read_bytes())

    def test_grouped_profiler_catalog_uses_the_persistent_fit_cache(self) -> None:
        """Freeze and certification must not rebuild paid profiler features."""

        spec = importlib.util.spec_from_file_location(
            "cpu_native_vnni_grouped_profiler_cache_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        corpus = object()
        sentinel = object()
        arguments = types.SimpleNamespace(
            development_profiler_requests=Path("requests.json"),
            development_profiler_evidence=Path("evidence.json"),
            development_profiler_observations=Path("observations.csv"),
            fit_cache_dir=Path("fit-cache"),
        )

        with mock.patch.object(
            module,
            "load_profiler_feature_catalog",
            return_value=sentinel,
        ) as loader:
            result = module._load_profiler_catalog(corpus, arguments)

        self.assertIs(result, sentinel)
        loader.assert_called_once_with(
            corpus,
            Path("requests.json"),
            Path("evidence.json"),
            source_corpus_path=Path("observations.csv"),
            cache_path=Path("fit-cache/profiler_feature_catalog_v1.json"),
        )

    def test_grouped_freeze_consumes_paired_development_comparisons(self) -> None:
        """Grouped refinement evidence must affect the frozen generic fit."""

        spec = importlib.util.spec_from_file_location(
            "cpu_native_vnni_grouped_paired_fit_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        development = object()
        manifest = mock.Mock()
        manifest.digest.return_value = "sha256:manifest"
        paired = {"paired-cell": ("comparison",)}
        frozen = object()

        with (
            mock.patch.object(
                module, "_require_partition", return_value=development
            ),
            mock.patch.object(
                module,
                "cpu_grouped_sealed_reserve_commitment",
                return_value="sha256:reserve",
            ),
            mock.patch.object(module, "_serial_hashes", return_value={}),
            mock.patch.object(
                module,
                "paired_comparison_digest",
                return_value="sha256:paired",
            ),
            mock.patch.object(module, "freeze_policy", return_value=frozen) as fit,
        ):
            result = module.freeze_cpu_verifier_policy(
                development,
                manifest,
                "sha256:serial",
                paired_development_comparisons=paired,
                max_leaves=1,
                minimum_promotion_warmups=1,
                minimum_promotion_samples=3,
            )

        self.assertIs(result, frozen)
        self.assertIs(
            fit.call_args.kwargs["paired_development_comparisons"], paired
        )
        self.assertEqual(fit.call_args.kwargs["max_leaves"], 1)
        self.assertEqual(
            fit.call_args.kwargs["minimum_promotion_warmups"], 1
        )
        self.assertEqual(
            fit.call_args.kwargs["minimum_promotion_samples"], 3
        )
        self.assertEqual(
            fit.call_args.kwargs["metadata"]["generic_max_leaves"], 1
        )
        self.assertEqual(
            fit.call_args.kwargs["metadata"][
                "paired_development_evidence_digest"
            ],
            "sha256:paired",
        )

    def test_generic_tree_emitter_compiles_every_launch_geometry_axis(self) -> None:
        """CPU emission must consume common tree predicates for each ISA lane."""

        spec = importlib.util.spec_from_file_location(
            "cpu_native_vnni_analyzer_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class=(
                "x86_64-test|build=AVX512|runtime=AVX512|threads=28"
            ),
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIVerifierRows",
            bundle_signature="single",
            prepared_family_id="NativeVNNI_cpu_CB0",
            packing_abi="native-vnni-cpu-cb0-v1",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=15,
            aspect_bucket=AspectBucket.BALANCED,
        )
        rule = GenericDispatchRule(
            domain=domain,
            predicates=(
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.N_TILES_512, 35, 1),
                    True,
                ),
                FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.K_GROUPS_PER_N_TILE_512,
                        5,
                        3,
                    ),
                    False,
                ),
            ),
            candidate_id="cpu.nvnni.verifier.wide_rows",
            arithmetic_fingerprint="sha256:test-cpu-rule",
            development_shape_groups=("shape-a", "shape-b"),
            development_max_regret=0.01,
            development_p95_regret=0.01,
            development_mean_regret=0.01,
        )
        wrapped = module.CPUGenericDispatchRule(
            build_isa="AVX512",
            runtime_isa="AVX512",
            threads=28,
            rule=rule,
        )
        generated = module.generate_include(
            [],
            [wrapped],
            corpus_digest="sha256:test-corpus",
            registry_digest="sha256:test-registry",
            profile=MeasurementProfile.QUICK,
        )
        source = "\n".join((
            generated,
            "int main() {",
            "  using namespace llaminar2::cpu::native_vnni::generated;",
            "  CPUNativeVNNIVerifierRowsPolicy policy{};",
            "  return selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            "      CPUNativeVNNIBuildISA::AVX512,",
            "      CPUNativeVNNIRuntimeISA::AVX512,",
            "      28, 0, 15, 17408, 5120, 0, policy) ? 0 : 1;",
            "}",
        ))
        compiled = subprocess.run(
            ["g++", "-std=c++20", "-x", "c++", "-fsyntax-only", "-"],
            input=source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        self.assertNotIn("aspect_ratio", generated)
        self.assertNotIn("min_work_items", generated)

    def test_generated_selector_is_total_over_m_and_uses_runtime_k_tiles(self) -> None:
        """Unseen depths bucket to M31 without erasing K-part geometry."""

        spec = importlib.util.spec_from_file_location(
            "cpu_native_vnni_totality_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        domain = GenericDomain(
            backend=Backend.CPU,
            architecture_class=(
                "x86_64-test|build=AVX512|runtime=AVX512|threads=28"
            ),
            semantic_contract=SemanticContract.VERIFIER_SERIAL_M1_BITWISE,
            operation_kind="NativeVNNIVerifierRows",
            bundle_signature="single",
            prepared_family_id="NativeVNNI_cpu_CB0",
            packing_abi="native-vnni-cpu-cb0-v1",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.EAGER,
            m=31,
            aspect_bucket=AspectBucket.WIDE,
        )
        threshold = FeatureThreshold(FeatureAxis.KPART_K_TILE_COUNT, 1, 1)

        def rule(less_equal: bool, candidate_id: str) -> GenericDispatchRule:
            return GenericDispatchRule(
                domain=domain,
                predicates=(FeaturePredicate(threshold, less_equal),),
                candidate_id=candidate_id,
                arithmetic_fingerprint="sha256:test-cpu-m-totality",
                development_shape_groups=("shape-a", "shape-b"),
                development_max_regret=0.01,
                development_p95_regret=0.01,
                development_mean_regret=0.01,
            )

        wrapped = [
            module.CPUGenericDispatchRule(
                "AVX512",
                "AVX512",
                28,
                rule(
                    True,
                    "cpu.nvnni.verifier.full_k.two_row_pair_grid.nbc1",
                ),
            ),
            module.CPUGenericDispatchRule(
                "AVX512",
                "AVX512",
                28,
                rule(False, "cpu.nvnni.verifier.pairwise"),
            ),
        ]
        exact_m2 = module.PolicyEntry(
            build_isa="AVX512",
            runtime_isa="AVX512",
            threads=28,
            codebook=0,
            m=2,
            n=17408,
            k=5120,
            policy="Pairwise",
            candidate_id="cpu.nvnni.verifier.pairwise",
            shape_names=("exact-m2",),
            max_surface_regret=0.0,
            max_cv=0.0,
        )
        generated = module.generate_include(
            [exact_m2],
            wrapped,
            corpus_digest="sha256:test-corpus",
            registry_digest="sha256:test-registry",
            profile=MeasurementProfile.QUICK,
        )
        source = "\n".join((
            generated,
            "#include <climits>",
            "int main() {",
            "  using namespace llaminar2::cpu::native_vnni::generated;",
            "  CPUNativeVNNIVerifierRowsPolicy policy{};",
            "  const int depths[] = {17, 30, 31, 32, 255, 256, 258, INT_MAX};",
            "  for (const int m : depths) {",
            "    if (!selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            "            CPUNativeVNNIBuildISA::AVX512,",
            "            CPUNativeVNNIRuntimeISA::AVX512,",
            "            28, 0, m, 17408, 5120, 0, policy) ||",
            "        policy != CPUNativeVNNIVerifierRowsPolicy::",
            "            FullKTwoRowPairGridNbc1)",
            "      return 1;",
            "  }",
            "  if (!selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            "          CPUNativeVNNIBuildISA::AVX512,",
            "          CPUNativeVNNIRuntimeISA::AVX512,",
            "          28, 0, 32, 17408, 5120, 2, policy) ||",
            "      policy != CPUNativeVNNIVerifierRowsPolicy::Pairwise)",
            "    return 2;",
            "  if (selectCPUNativeVNNIVerifierRowsGeneratedPolicy(",
            "          CPUNativeVNNIBuildISA::AVX512,",
            "          CPUNativeVNNIRuntimeISA::AVX512,",
            "          28, 0, 1, 17408, 5120, 0, policy))",
            "    return 3;",
            "  return 0;",
            "}",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "cpu-verifier-totality"
            compiled = subprocess.run(
                ["g++", "-std=c++20", "-x", "c++", "-o", str(binary), "-"],
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr.decode())


if __name__ == "__main__":
    unittest.main()
