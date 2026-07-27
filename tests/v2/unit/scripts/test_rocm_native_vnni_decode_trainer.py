#!/usr/bin/env python3
"""Regression tests for the common-backed ROCm decode policy analyzer."""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
KERNEL_PERF_ROOT = REPO_ROOT / "tests" / "v2" / "performance" / "kernels"
if str(KERNEL_PERF_ROOT) not in sys.path:
    sys.path.insert(0, str(KERNEL_PERF_ROOT))

from native_vnni_dispatch.corpus import GenericDomain  # noqa: E402
from native_vnni_dispatch.profiles import MeasurementProfile  # noqa: E402
from native_vnni_dispatch.schema import (  # noqa: E402
    AspectBucket,
    Backend,
    ExecutionMode,
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
    / "rocm"
    / "analyze_rocm_native_vnni_decode_trainer.py"
)
TRAINER_SOURCE = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "rocm"
    / "Perf__NativeVNNI_Throughput.cpp"
)
RUNTIME_SOURCE = (
    REPO_ROOT
    / "src"
    / "v2"
    / "kernels"
    / "rocm"
    / "gemm"
    / "ROCmGemvKernel_native_VNNI.hip"
)
DEBUG_ENV_SOURCE = REPO_ROOT / "src" / "v2" / "utils" / "DebugEnv.h"


class ROCmNativeVNNIDecodeTrainerTest(unittest.TestCase):
    """Exercise strong adaptation and alias/mode-robust current-ABI emission."""

    @staticmethod
    def row(
        candidate: str,
        execution_mode: str,
        median_us: float,
        *,
        m: int = 1,
    ) -> dict[str, object]:
        """Construct one complete Q8_0 candidate observation."""

        verifier = m > 1
        kb = 32 if verifier else int(candidate.removeprefix("KB"))
        canonical_candidate = "INHERIT_SERIAL_M1" if verifier else candidate
        return {
            "backend": "rocm",
            "phase": "decode",
            "source_format": "Q8_0",
            "source_codebook": 19,
            "execution_codebook": 19,
            "shape": "35BMoE_Expert_GateUp",
            "execution_mode": execution_mode,
            "m": m,
            "n": 512,
            "k": 2048,
            "candidate_id": canonical_candidate,
            "kb": kb,
            "target_waves": 4,
            "weight_bytes": 1114112,
            "warmup_count": 2,
            "sample_count": 3,
            "min_us": median_us,
            "median_us": median_us,
            "p95_us": median_us,
            "mad_us": 0,
            "cv": 0.01,
            "effective_bandwidth_gbs": 30,
            "bit_mismatches": 0,
            "first_bit_mismatch": 0,
            "repeat_byte_mismatches": 0,
            "max_abs": 0,
            "relative_l2": 0,
            "cosine": 1,
            "symmetric_kld": 0,
            "grouped_output_digest": "fnv1a64:equal00000000000",
            "serial_output_digest": "fnv1a64:equal00000000000",
            "timing_sample_digest": f"fnv1a64:{candidate.lower():0<16}"[:24],
            "route_counter_ok": 1,
            "observed_candidate_id": canonical_candidate,
            "observed_path": "split_reduce",
            "serial_m1_kb": 32,
            "serial_m1_target_waves": 4,
            "serial_route_counter_ok": 1,
            "numerical_correctness": 1,
            "correctness_pass": 1,
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
            if result.returncode == 0:
                self.assertTrue(common.exists())
                with common.open(newline="", encoding="utf-8") as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), len(rows))
            return result, generated, summary_rows

    def test_opposing_execution_modes_choose_lowest_worst_surface_regret(self) -> None:
        rows = [
            self.row("KB8", "eager", 10.0),
            self.row("KB8", "graph_captured", 100.0),
            self.row("KB32", "eager", 20.0),
            self.row("KB32", "graph_captured", 20.0),
            self.row("INHERIT_SERIAL_M1", "eager", 30.0, m=2),
            self.row("INHERIT_SERIAL_M1", "graph_captured", 31.0, m=2),
        ]

        result, generated, summary = self.run_analyzer(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("common alias/mode-robust exact oracle", generated)
        self.assertIn("{32, 4}", generated)
        self.assertNotIn("M=2 512x2048", generated)
        self.assertEqual(summary[0]["candidate_id"], "rocm.nvnni.decode.fast.kb32")
        self.assertEqual(summary[0]["certified_verifier_key_count"], "1")

    def test_explicit_fast_kb_is_rejected_for_verifier_depth(self) -> None:
        row = self.row("INHERIT_SERIAL_M1", "eager", 30.0, m=2)
        row["candidate_id"] = "KB32"
        row["observed_candidate_id"] = "KB32"

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not support Verifier", result.stderr)

    def test_require_complete_rejects_partial_alias_candidate_and_depth_matrix(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical alias/mode coverage is incomplete", result.stderr)

    def test_production_rejects_obsolete_one_pass_policy_emission(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--profile",
            "production",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires development freeze and separate sealed certification",
            result.stderr,
        )

    def test_production_freeze_requires_profiler_evidence(self) -> None:
        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--profile",
            "production",
            "--freeze-generic",
            "--policy-json",
            "/tmp/unused-rocm-policy.json",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "requires complete development profiler evidence",
            result.stderr,
        )

    def test_generic_leaf_budget_must_be_positive(self) -> None:
        """ROCm tree segmentation must remain an explicit bounded choice."""

        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--generic-max-leaves",
            "0",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "--generic-max-leaves must be in [1, 32]",
            result.stderr,
        )

    def test_sealed_provenance_overrides_are_atomic(self) -> None:
        """A fresh seal cannot inherit only part of a retained build identity."""

        result, _, _ = self.run_analyzer(
            [self.row("KB32", "eager", 20.0)],
            "--profile",
            "production",
            "--certify-generic",
            "--sealed-run-id",
            "fresh-seal",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "sealed provenance overrides must be supplied together",
            result.stderr,
        )

    def test_reused_development_audit_is_part_of_frozen_metadata(self) -> None:
        """Harness-only rebuild authorization survives freeze reconstruction."""

        source = ANALYZER.read_text(encoding="utf-8")
        self.assertIn('metadata["development_build_change_audit"]', source)
        self.assertGreaterEqual(
            source.count("args.development_build_change_audit"),
            4,
        )
        self.assertIn("--fit-cache-dir", source)
        self.assertIn("PolicyFitCache(directory=fit_cache_directory)", source)
        self.assertIn("sealed=True", source)

    def test_profiler_only_launch_skips_paid_timing_and_d2h_work(self) -> None:
        """Each rocprof pass pays only fixed preconditioning plus its target."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        profiler_branch = source.index('if (!profiler_request_id.empty())')
        correctness_download = source.index(
            "std::vector<float> first_output;",
            profiler_branch,
        )
        timed_samples = source.index(
            "result.timing_samples_us.reserve",
            correctness_download,
        )

        self.assertLess(profiler_branch, correctness_download)
        self.assertLess(correctness_download, timed_samples)
        isolated = source[profiler_branch:correctness_download]
        self.assertIn("kProfilerPreconditioningLaunches = 2", isolated)
        self.assertIn("result.isolated_profile_launches = 1", isolated)
        self.assertIn("result.valid = true;", isolated)
        self.assertIn("return result;", isolated)
        self.assertNotIn("copyTrainerOutputToHost", isolated)
        self.assertNotIn("hipEventCreate", isolated)

    def test_profiler_batch_lifetime_matches_validated_rocm_boundary(self) -> None:
        """The trainer must reject a plan before rocprofiler corrupts HSA state."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "static constexpr size_t kMaximumRequestsPerProcess = 256;",
            source,
        )
        self.assertIn(
            "static constexpr size_t kMaximumGraphRequestsPerProcess = 256;",
            source,
        )
        self.assertNotIn(
            "static constexpr size_t kMaximumRequestsPerProcess = 512;",
            source,
        )

    def test_profiler_only_weight_fixture_avoids_full_matrix_rng(self) -> None:
        """Profiler replay preserves source bytes without regenerating N*K values."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        helper_start = source.index(
            "static std::unique_ptr<TensorBase> makeProfilerWeightFixture("
        )
        helper_end = source.index("using GEMVShape", helper_start)
        helper = source[helper_start:helper_end]
        trainer_start = source.index(
            "TEST_F(NativeVNNIPerfTest, TrainerCsv_CodebookTagged)"
        )
        trainer = source[trainer_start:]

        self.assertIn("auto seed_row = format.create(1, K);", helper)
        self.assertIn("std::memcpy(", helper)
        self.assertIn("initialized_rows", helper)
        self.assertIn(
            "!profiler_request_id.empty() || profiler_batch.enabled()",
            trainer,
        )
        self.assertIn("? makeProfilerWeightFixture(", trainer)
        self.assertIn(": fmt.create(", trainer)

    def test_legacy_weak_csv_is_rejected(self) -> None:
        row = self.row("KB32", "eager", 20.0)
        del row["repeat_byte_mismatches"]

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong ROCm decode columns", result.stderr)

    def test_cpp_trainer_does_not_cap_grouped_verifier_runtime_m(self) -> None:
        """Verifier inheritance and no-atomic gates apply beyond legacy M=4."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("M >= 2 && M <= 4", source)
        self.assertIn(
            "M >= 2 &&\n"
            "                    kind == "
            "DecodeCandidateKind::VerifierInheritSerialM1",
            source,
        )

    def test_generated_query_never_manufactures_a_q8_policy_on_miss(self) -> None:
        """Codebook 19 must earn coverage from the same generated table as all formats."""

        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("resolveNativeVNNIQ80DirectRuntimeConfig", source)
        self.assertNotIn(
            "LLAMINAR_ROCM_NVNNI_Q8_DIRECT",
            DEBUG_ENV_SOURCE.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "cache.insert(cache_key, false, cached);\n"
            "        return cached;",
            source,
        )
        self.assertIn(
            "return resolveNativeVNNIDecodeGeneratedRuntimeConfig(\n"
            "        codebook_id, 1, N, K);",
            source,
        )
        self.assertIn("generated dispatch miss", source)

    def test_serial_query_publishes_shape_clamped_effective_kb(self) -> None:
        """A nominal generic formula must report the physical production KB."""

        source = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("const int k_groups = k / 32;", source)
        self.assertIn(
            "cached.kb = (requested_kb < k_groups) ? requested_kb : k_groups;",
            source,
        )
        self.assertNotIn(
            "cached.kb = static_cast<int>(generated_cfg.kb);",
            source,
        )

    def test_generic_tree_emitter_compiles_every_launch_geometry_axis(self) -> None:
        """ROCm must consume the common predicate IR rather than legacy ranges."""

        spec = importlib.util.spec_from_file_location(
            "rocm_native_vnni_analyzer_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        domain = GenericDomain(
            backend=Backend.ROCM,
            architecture_class="gfx906-test",
            semantic_contract=SemanticContract.FAST,
            operation_kind="NativeVNNIDecodeProjection",
            bundle_signature="single",
            prepared_family_id="NativeVNNI_rocm_CB19",
            packing_abi="native-vnni-rocm-cb19-v1",
            runtime_codebook_id=19,
            execution_mode=ExecutionMode.EAGER,
            m=1,
            aspect_bucket=AspectBucket.BALANCED,
        )
        rule = GenericDispatchRule(
            domain=domain,
            predicates=(
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.N_TILES_256, 4, 1),
                    True,
                ),
                FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.K_GROUPS_PER_N_TILE_256,
                        9,
                        2,
                    ),
                    False,
                ),
            ),
            candidate_id="rocm.nvnni.decode.fast.kb32",
            arithmetic_fingerprint="sha256:test-rocm-rule",
            development_shape_groups=("shape-a", "shape-b"),
            development_max_regret=0.01,
            development_p95_regret=0.01,
            development_mean_regret=0.01,
        )
        generated = module.generate_include(
            [],
            [rule],
            corpus_digest="sha256:test-corpus",
            registry_digest="sha256:test-registry",
            profile=MeasurementProfile.QUICK,
        )
        source = "\n".join((
            generated,
            "int main() {",
            "  llaminar2::rocm::generated::ROCmNativeVNNIDecodeDispatchConfig out{};",
            "  return llaminar2::rocm::generated::selectROCmNativeVNNIDecodeGenerated(",
            "      19, 1, 900, 1024, out) ? 0 : 1;",
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

    def test_additive_overlay_reuses_modern_total_generic_base(self) -> None:
        """Exact additions preserve modern generic coverage without a fallback."""

        spec = importlib.util.spec_from_file_location(
            "rocm_native_vnni_overlay_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        entries = [
            module.FastEntry(
                codebook=19,
                n=n,
                k=k,
                kb=kb,
                candidate_id=f"rocm.nvnni.decode.fast.kb{kb}",
                shape_name=name,
                max_surface_regret=0.01,
                max_cv=0.01,
            )
            for name, n, k, kb in (
                ("Tall", 128, 512, 16),
                ("Balanced", 512, 512, 8),
                ("Wide", 2048, 512, 4),
                ("VeryWide", 16384, 512, 1),
            )
        ]
        base = (
            REPO_ROOT
            / "src"
            / "v2"
            / "kernels"
            / "rocm"
            / "gemm"
            / "ROCmNativeVNNIDecodeDispatchGenerated.inc"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated_path = root / "generated.inc"
            source_path = root / "totality.cpp"
            binary_path = root / "totality"
            module.emit_overlay(entries, generated_path, base)
            generated = generated_path.read_text(encoding="utf-8")
            self.assertIn(module.OVERLAY_BEGIN, generated)
            self.assertNotIn(module.GENERIC_OVERLAY_BEGIN, generated)
            self.assertNotIn(
                "selectROCmNativeVNNIDecodeAspectFallback", generated
            )
            self.assertIn("codebook_id == 19", generated)
            self.assertIn("kCommonExactOverlay", generated)
            self.assertNotIn("codebook_id == 19 &&", generated)

            source_path.write_text(
                "\n".join((
                    f'#include "{generated_path}"',
                    "int main() {",
                    "  using namespace llaminar2::rocm::generated;",
                    "  ROCmNativeVNNIDecodeDispatchConfig out{};",
                    "  const int ns[] = {64, 1024, 4096, 32768};",
                    "  const int ks[] = {1024, 1024, 1024, 1024};",
                    "  for (int i = 0; i < 4; ++i)",
                    "    if (!selectROCmNativeVNNIDecodeGenerated(",
                    "            19, 1, ns[i], ks[i], out)) return 1;",
                    "  if (selectROCmNativeVNNIDecodeGenerated(",
                    "          19, 2, 1024, 1024, out)) return 2;",
                    "  return 0;",
                    "}",
                )),
                encoding="utf-8",
            )
            compiled = subprocess.run(
                [
                    "g++", "-std=c++20", str(source_path),
                    "-o", str(binary_path),
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr)
            executed = subprocess.run(
                [str(binary_path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_generic_supplement_deduplicates_equal_work_boundaries(self) -> None:
        """N*K collisions keep the lowest-regret representable launch."""

        spec = importlib.util.spec_from_file_location(
            "rocm_native_vnni_boundary_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        entries = [
            module.FastEntry(
                codebook=19,
                n=n,
                k=k,
                kb=kb,
                candidate_id=f"rocm.nvnni.decode.fast.kb{kb}",
                shape_name=f"Shape{kb}",
                max_surface_regret=regret,
                max_cv=0.01,
            )
            for n, k, kb, regret in (
                (1024, 2048, 8, 0.04),
                (2048, 1024, 4, 0.01),
            )
        ]

        self.assertEqual(
            module._compress_totalizing_rules(entries),
            [(1024 * 2048, 4)],
        )


if __name__ == "__main__":
    unittest.main()
