#!/usr/bin/env python3
"""Regression tests for the common-backed CUDA decode policy analyzer."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
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
from native_vnni_dispatch.format_registry import FORMAT_SPECS  # noqa: E402
from native_vnni_dispatch.measurement_plan import (  # noqa: E402
    load_gpu_measurement_plan,
)
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
from native_vnni_dispatch.shape_manifest import load_shape_manifest  # noqa: E402
ANALYZER = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "cuda"
    / "gemm"
    / "analyze_cuda_native_vnni_decode_trainer.py"
)
TRAINER_SOURCE = (
    REPO_ROOT
    / "tests"
    / "v2"
    / "performance"
    / "kernels"
    / "cuda"
    / "gemm"
    / "Perf__CUDANativeVNNIDecodeTrainer.cpp"
)
RUNTIME_SOURCE = (
    REPO_ROOT
    / "src"
    / "v2"
    / "kernels"
    / "cuda"
    / "gemm"
    / "CUDANativeVNNIGemvShardImpl.cu.inc"
)
DEBUG_ENV_SOURCE = REPO_ROOT / "src" / "v2" / "utils" / "DebugEnv.h"


class CUDANativeVNNIDecodeTrainerTest(unittest.TestCase):
    """Exercise exact-KB emission and typed grouped-verifier dispatch."""

    @staticmethod
    def load_analyzer_module():
        """Load the analyzer as a module for focused pure-policy regressions."""

        module_name = "cuda_native_vnni_analyzer_reachability_test_module"
        spec = importlib.util.spec_from_file_location(module_name, ANALYZER)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import CUDA analyzer from {ANALYZER}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_freeze_cli_reports_missing_input_without_obsolete_state(self) -> None:
        """The current freeze parser must not inspect retired M1 input flags."""

        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    "--freeze-generic",
                    "--profile",
                    "production",
                    "--development-profiler-requests",
                    "/dev/null",
                    "--development-profiler-evidence",
                    "/dev/null",
                    "--output",
                    str(Path(directory) / "dispatch.inc"),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "--freeze-generic requires development --input shards",
            result.stderr,
        )
        self.assertNotIn("AttributeError", result.stderr)

    def test_economical_kblock_geometry_is_cached_by_k(self) -> None:
        """Coverage validation must not recompute one K geometry per row."""

        spec = importlib.util.spec_from_file_location(
            "cuda_native_vnni_analyzer_kblock_cache_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        module._economical_exact_kblocks.cache_clear()
        first = module._economical_exact_kblocks(512)
        second = module._economical_exact_kblocks(512)

        self.assertIs(first, second)
        self.assertEqual(module._economical_exact_kblocks.cache_info().misses, 1)
        self.assertEqual(module._economical_exact_kblocks.cache_info().hits, 1)

    def test_grouped_row_reuse_candidates_are_complete_and_pruned(self) -> None:
        """Grouped CUDA tuning must expose every useful trained row tile."""

        trainer = TRAINER_SOURCE.read_text(encoding="utf-8")
        runtime = RUNTIME_SOURCE.read_text(encoding="utf-8")
        for rows in (2, 4, 8, 16, 32, 64):
            self.assertIn(f"case {rows}:", runtime)
            self.assertIn(str(rows), trainer)
        self.assertIn("std::bit_ceil", trainer)
        self.assertIn("grouped_rows > maximum_useful_rows", trainer)
        self.assertIn("eight packed activation words for every row", runtime)

        registry = __import__(
            "native_vnni_dispatch.candidate_registry",
            fromlist=["cuda_native_vnni_gemv_registry"],
        ).cuda_native_vnni_gemv_registry()
        for rows in (2, 4, 8, 16, 32, 64):
            candidate = registry.resolve(
                "cuda.nvnni.decode.verifier.inherit_serial_m1."
                f"r{rows}"
            )
            self.assertEqual(candidate.config_json["grouped_rows"], rows)
        tensor_core = registry.resolve(
            "cuda.nvnni.decode.verifier.tensor_core_mma16"
        )
        self.assertEqual(
            tensor_core.config_json["family"], "tensor_core_mma16"
        )
        with self.assertRaisesRegex(ValueError, "unknown forceable candidate"):
            registry.resolve(
                "cuda.nvnni.decode.verifier.inherit_serial_m1"
            )
        self.assertIn("mma.sync.aligned.m16n8k32", runtime)

    def test_grouped_completeness_prunes_tiles_dominated_at_runtime_m(self) -> None:
        """The analyzer must expect exactly the trainer's launchable row tiles."""

        module = self.load_analyzer_module()
        registry = __import__(
            "native_vnni_dispatch.candidate_registry",
            fromlist=["cuda_native_vnni_gemv_registry"],
        ).cuda_native_vnni_gemv_registry()
        grouped = {
            int(candidate.config_json["grouped_rows"]): candidate.candidate_id
            for candidate in registry.entries
            if candidate.config_json.get("family") == "inherit_serial_m1"
        }
        expected_maximum = {
            2: 2,
            3: 4,
            4: 4,
            5: 8,
            8: 8,
            9: 16,
            16: 16,
            31: 32,
        }
        for m, maximum in expected_maximum.items():
            with self.subTest(m=m):
                reachable = {
                    candidate_id
                    for rows, candidate_id in grouped.items()
                    if module._grouped_row_tile_is_reachable(rows, m)
                }
                self.assertEqual(
                    reachable,
                    {
                        candidate_id
                        for rows, candidate_id in grouped.items()
                        if rows <= maximum
                    },
                )

    def test_paired_refinement_requires_serial_byte_equality(self) -> None:
        """Paired confirmation must never certify a merely close candidate."""

        trainer = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "evidence.comparison.mismatch_count == 0",
            trainer,
        )
        self.assertIn(
            '<< " serial_byte_mismatches="',
            trainer,
        )

    @staticmethod
    def row(
        candidate: str,
        execution_mode: str,
        median_us: float,
        *,
        m: int = 1,
        serial_candidate: str | None = None,
    ) -> dict[str, object]:
        """Construct one complete Q8_K production-route observation."""

        verifier = m > 1
        serial = serial_candidate or candidate
        serial_kb = int(serial.rsplit("kb", 1)[1])
        serial_tile = int(serial.split(".tn", 1)[1].split(".", 1)[0])
        serial_cpt = int(serial.split(".cpt", 1)[1].split(".", 1)[0])
        if verifier:
            requested = "cuda.nvnni.decode.verifier.inherit_serial_m1.r2"
            family = "inherit_serial_m1"
            tile_n = 0
            cpt = 0
            exact_kb = 0
            force_two_phase = 0
            grouped_rows = 2
            observed = requested
        else:
            requested = candidate
            family = "kpar"
            tile_n = int(candidate.split(".tn", 1)[1].split(".", 1)[0])
            cpt = int(candidate.split(".cpt", 1)[1].split(".", 1)[0])
            exact_kb = int(candidate.rsplit("kb", 1)[1])
            force_two_phase = 1
            grouped_rows = 0
            observed = candidate
        return {
            "backend": "cuda",
            "phase": "decode",
            "source_format": "Q8_K",
            "source_codebook": 21,
            "execution_codebook": 19,
            "shape": "35BMoE_Expert_GateUp",
            "execution_mode": execution_mode,
            "m": m,
            "n": 512,
            "k": 2048,
            "candidate_id": requested,
            "measurement_protocol": "sample_interleaved_v1",
            "family": family,
            "tile_n": tile_n,
            "cpt": cpt,
            "target_waves": 0,
            "mkg": 0,
            "max_kb": 0,
            "exact_kb": exact_kb,
            "force_two_phase": force_two_phase,
            "grouped_rows": grouped_rows,
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
            "grouped_output_digest": "sha256:equal",
            "serial_output_digest": "sha256:equal",
            "timing_sample_digest": "sha256:timing",
            "supported": 1,
            "graph_capture_ok": 1,
            "workspace_ok": 1,
            "explicit_stream_ok": 1,
            "route_counter_ok": 1,
            "observed_candidate_id": observed,
            "observed_path": "kpar",
            "observed_tile_n": serial_tile if verifier else tile_n,
            "observed_cpt": serial_cpt if verifier else cpt,
            "observed_effective_kb": serial_kb if verifier else exact_kb,
            "serial_m1_candidate_id": serial,
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
        """Run the analyzer against a temporary strong-evidence corpus."""

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "trainer.csv"
            output = root / "generated.inc"
            summary = root / "summary.csv"
            common = root / "common.csv"
            prepared_rows = [dict(row) for row in rows]
            grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
            for row in prepared_rows:
                key = tuple(row[field] for field in (
                    "source_format",
                    "source_codebook",
                    "execution_codebook",
                    "shape",
                    "execution_mode",
                    "m",
                    "n",
                    "k",
                ))
                grouped.setdefault(key, []).append(row)
            for group_index, group_rows in enumerate(grouped.values(), start=1):
                for measurement_order, row in enumerate(group_rows):
                    row.setdefault("measurement_order", measurement_order)
                    row.setdefault("measurement_order_seed", 100000 + group_index)
            with input_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=tuple(prepared_rows[0])
                )
                writer.writeheader()
                writer.writerows(prepared_rows)
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

    def test_opposing_modes_emit_independent_exact_kpart_winners(self) -> None:
        """The generated ABI must preserve each mode's economical exact KB."""

        kb8 = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        kb32 = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        rows = [
            self.row(kb8, "eager", 10.0),
            self.row(kb8, "graph_captured", 100.0),
            self.row(kb32, "eager", 20.0),
            self.row(kb32, "graph_captured", 20.0),
            self.row(kb32, "eager", 30.0, m=2, serial_candidate=kb32),
            self.row(
                kb32,
                "graph_captured",
                31.0,
                m=2,
                serial_candidate=kb32,
            ),
        ]

        result, generated, summary = self.run_analyzer(rows)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("common alias-robust, mode-specific exact oracle", generated)
        self.assertIn("{256, 4, 0, 0, 0, 1, 8}", generated)
        self.assertIn("{256, 4, 0, 0, 0, 1, 32}", generated)
        self.assertIn("M=2 512x2048", generated)
        self.assertIn("selectGeneratedGroupedTuning", generated)
        self.assertIn("LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2", generated)
        self.assertEqual(len(summary), 2)
        by_mode = {row["execution_mode"]: row for row in summary}
        self.assertEqual(by_mode["eager"]["candidate_id"], kb8)
        self.assertEqual(by_mode["eager"]["exact_kb"], "8")
        self.assertEqual(by_mode["graph_captured"]["candidate_id"], kb32)
        self.assertEqual(by_mode["graph_captured"]["exact_kb"], "32")
        self.assertEqual(
            by_mode["eager"]["certified_verifier_key_count"], "2"
        )

    def test_explicit_fast_candidate_is_rejected_for_verifier_depth(self) -> None:
        """Verifier rows may certify inheritance but cannot publish a Fast route."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        row = self.row(candidate, "eager", 30.0, m=2)
        row.update({
            "candidate_id": candidate,
            "family": "kpar",
            "tile_n": 256,
            "cpt": 4,
            "exact_kb": 32,
            "force_two_phase": 1,
            "observed_candidate_id": candidate,
        })

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not support Verifier", result.stderr)

    def test_require_complete_rejects_partial_matrix(self) -> None:
        """Promotion cannot silently omit aliases, modes, candidates, or depths."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        result, _, _ = self.run_analyzer(
            [self.row(candidate, "eager", 20.0)],
            "--require-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical alias/mode coverage is incomplete", result.stderr)

    def test_exact_m1_stage_has_no_uncertified_or_default_route(self) -> None:
        """Staging emits exact evidence only and leaves misses as hard misses."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        rows = [
            self.row(candidate, "eager", 20.0),
            self.row(candidate, "graph_captured", 21.0),
        ]
        result, generated, _ = self.run_analyzer(rows, "--exact-only")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("generic policy intentionally absent", generated)
        self.assertIn("Uncovered runtime keys return false", generated)
        self.assertNotIn("const float aspect_ratio", generated)
        self.assertNotIn("classifyShapeGenerated", generated)
        self.assertNotIn("selectGeneratedTuning", generated)

    def test_fast_m1_completeness_gate_rejects_partial_surface(self) -> None:
        """The stage gate is strict without requiring causally later M>1 rows."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        result, _, _ = self.run_analyzer(
            [self.row(candidate, "eager", 20.0)],
            "--exact-only",
            "--require-fast-m1-complete",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("canonical alias/mode coverage is incomplete", result.stderr)

    def test_legacy_weak_csv_is_rejected(self) -> None:
        """A missing exact-KB or repeat-stability proof invalidates the corpus."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        row = self.row(candidate, "eager", 20.0)
        del row["repeat_byte_mismatches"]

        result, _, _ = self.run_analyzer([row])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing strong CUDA decode columns", result.stderr)

    def test_duplicate_measurement_order_is_rejected(self) -> None:
        """Every cell must expose one auditable randomized permutation."""

        kb8 = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb8"
        kb32 = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        first = self.row(kb8, "eager", 10.0)
        second = self.row(kb32, "eager", 11.0)
        for row in (first, second):
            row["measurement_order"] = 0
            row["measurement_order_seed"] = 123456789

        result, _, _ = self.run_analyzer([first, second])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not one contiguous permutation", result.stderr)

    def test_cpp_trainer_randomizes_and_publishes_candidate_order(self) -> None:
        """Every broad timing round must independently permute candidates."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("measurementOrderSeed", source)
        self.assertIn("std::mt19937_64 order_engine(order_seed)", source)
        self.assertIn("std::shuffle(", source)
        self.assertIn("measurement_order_seed", source)
        self.assertIn("runCandidatesInterleaved", source)
        self.assertIn("sample_interleaved_v1", source)
        self.assertIn("sample_measurement_orders", source)
        self.assertIn("sample_order_seeds", source)
        self.assertIn("interleaved_timed_launch", source)

    def test_cpp_trainer_has_a_distinct_interleaved_confirmation_mode(self) -> None:
        """Frozen timing must retain identity and balance the first-launch role."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_CSV", source)
        self.assertIn(
            "LLAMINAR_CUDA_NVNNI_DECODE_PAIRED_REQUEST_MANIFEST",
            source,
        )
        self.assertIn("runPairedConfirmation", source)
        self.assertIn("pairedOrderSeed", source)
        self.assertIn("pairedRequestSeed", source)
        self.assertIn("first_candidate_offset", source)
        self.assertIn(
            "(static_cast<size_t>(sample) + first_candidate_offset) % 2",
            source,
        )
        self.assertIn("within_pair_order", source)
        self.assertIn("cuda-paired-interleaved-v2", source)
        self.assertIn("request_id", source)
        self.assertIn("static_cast<size_t>(config.samples)", source)
        self.assertNotIn("CandidateEvidence runCandidate(", source)
        self.assertNotIn("PAIRED_SELECTED", source)
        self.assertNotIn("PAIRED_EXACT", source)

    def test_analyzer_threads_paired_development_evidence_into_common_compiler(self) -> None:
        """Final CUDA emission must use the same tournament as development CV."""

        source = ANALYZER.read_text(encoding="utf-8")
        self.assertIn("--paired-development-csv", source)
        self.assertIn("read_paired_confirmation_csv", source)
        self.assertIn("paired_timing_comparisons", source)
        self.assertIn(
            "paired_development_comparisons=paired_development_comparisons",
            source,
        )

    def test_cpp_oracle_compares_every_grouped_verifier_row(self) -> None:
        """Prevent regression to certifying only the first row of runtime M."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "static_cast<size_t>(m) * static_cast<size_t>(n)",
            source,
        )
        self.assertNotIn(
            "repeated_output, serial.output, static_cast<size_t>(n)",
            source,
        )

    def test_explicit_exact_kb_can_exercise_trailing_empty_partitions(self) -> None:
        """Pinned diagnostics retain dominated KB schedules omitted by training."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        self.assertIn("exactKBlocksForSweep(config, k_groups)", source)
        self.assertIn("if (config.candidate_ids.empty())", source)
        self.assertIn("return economicalExactKBlocks(k_groups);", source)
        self.assertIn("for (int kb = 1; kb <= maximum; ++kb)", source)

    def test_unseen_m1_shapes_use_only_an_explicit_trainer_oracle(self) -> None:
        """Policy expansion must not require or weaken production miss behavior."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        runtime = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("diagnosticM1OracleCandidate", source)
        self.assertIn(
            "cuda.nvnni.decode.fast_m1.kpar.tn128.cpt1.kb1",
            source,
        )
        self.assertIn(
            "m == 1 ? &diagnosticM1OracleCandidate() : nullptr",
            source,
        )
        self.assertIn("queryGraphCapturedExecution", runtime)
        self.assertIn(
            "kCanonicalDecodePolicyGraphCaptured",
            runtime,
        )
        self.assertIn(
            "eager warmup deliberately selects that same",
            runtime,
        )
        self.assertNotIn("diagnosticM1OracleCandidate", runtime)

    def test_captured_verifier_uses_a_captured_serial_m1_oracle(self) -> None:
        """Captured grouped evidence may not borrow an eager serial-M1 route."""

        source = TRAINER_SOURCE.read_text(encoding="utf-8")
        runtime = RUNTIME_SOURCE.read_text(encoding="utf-8")
        self.assertIn("ExecutionMode mode", source)
        self.assertIn("serial_graph_primer", source)
        self.assertIn("serial_graph_launch", source)
        self.assertIn(
            'tag("execution_mode") != executionModeName(mode)',
            source,
        )
        self.assertIn('{"execution_mode", graph_captured', runtime)

    def test_production_runtime_has_no_default_or_rowpar_override(self) -> None:
        """The learned boolean resolver is authoritative for M1 and verifier."""

        runtime = RUNTIME_SOURCE.read_text(encoding="utf-8")
        debug_env = DEBUG_ENV_SOURCE.read_text(encoding="utf-8")
        self.assertEqual(
            runtime.count(
                "selectGeneratedDispatch<trainedPolicyCodebook<CB>()>("
            ),
            1,
        )
        self.assertGreaterEqual(
            runtime.count("selectCachedGeneratedDispatch<CB>("),
            1,
        )
        self.assertGreaterEqual(
            runtime.count("selectCachedGeneratedDispatchForPolicy<CB>("),
            2,
        )
        self.assertEqual(
            runtime.count(
                "selectGeneratedGroupedTuning<trainedPolicyCodebook<CB>()>("
            ),
            1,
        )
        self.assertGreaterEqual(
            runtime.count(
                "selectCachedGeneratedGroupedTuningForPolicy<CB>("
            ),
            1,
        )
        self.assertNotIn("classifyShapeGenerated<CB>", runtime)
        self.assertNotIn("selectGeneratedTuning<CB>", runtime)
        self.assertNotIn("isRowParEnabled", runtime)
        self.assertNotIn("LLAMINAR_CUDA_GEMV_ROWPAR", debug_env)
        self.assertNotIn("cuda_gemv_rowpar", debug_env)

    def test_final_compile_uses_certified_m1_and_grouped_evidence(self) -> None:
        """The causal transaction never reopens raw M1 timing after certification."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        verifier_rows = [
            self.row(candidate, "eager", 30.0, m=2),
            self.row(candidate, "graph_captured", 31.0, m=2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            verifier_csv = root / "verifier.csv"
            certified_include = root / "certified-m1.inc"
            certified_policy = root / "certified-m1.json"
            output = root / "generated.inc"
            for seed_offset, row in enumerate(verifier_rows, start=1):
                row["measurement_order"] = 0
                row["measurement_order_seed"] = 200000 + seed_offset
            with verifier_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=tuple(verifier_rows[0]),
                )
                writer.writeheader()
                writer.writerows(verifier_rows)

            def digest(mapping: dict[str, object]) -> str:
                encoded = json.dumps(
                    mapping,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                return "sha256:" + hashlib.sha256(encoded).hexdigest()

            manifest = load_shape_manifest()
            manifest_digest = manifest.digest()
            measurement_plan_digest = load_gpu_measurement_plan(
                manifest=manifest,
            ).digest(manifest)
            generic_mapping = {
                "policy_abi": 3,
                "learner_version": "test-learner",
                "feature_schema_version": "test-features",
                "generic_rules": [],
                "unpromoted_domains": [],
                "cross_validation": [],
            }
            generic_digest = digest(generic_mapping)
            measurement_context = {
                "run_id": "test-run",
                "git_revision": "deadbeef",
                "build_id": "sha256:test-build",
                "compiler_id": "test-nvcc",
                "architecture_class": "workflow-cuda",
                "device_name": "workflow-cuda-device",
                "driver_runtime": "workflow-cuda-runtime",
                "serial_m1_policy_hash": "sha256:test-m1",
            }
            policy = {
                **generic_mapping,
                "exact_entries": [],
                "metadata": {
                    "shape_manifest_digest": manifest_digest,
                    "measurement_plan_digest": measurement_plan_digest,
                    "frozen_generic_policy_digest": generic_digest,
                    "development_corpus_digest": "sha256:development",
                    "sealed_corpus_digest": "sha256:sealed",
                    "development_measurement_context": measurement_context,
                    "sealed_measurement_context": {
                        **measurement_context,
                        "run_id": "test-run-sealed",
                    },
                    "promotion_p95_regret_budget": 0.05,
                    "promotion_minimum_passing_domain_fraction": 0.95,
                },
            }
            certified_policy.write_text(
                json.dumps({
                    "state": "sealed_certified",
                    "policy": policy,
                    "policy_digest": digest(policy),
                    "frozen_generic_policy_digest": generic_digest,
                    "certification": {
                        "sealed_cell_count": 1,
                        "out_of_scope_cell_count": 0,
                        "required_cell_count": 1,
                        "covered_cell_count": 1,
                        "coverage": 1.0,
                        "verifier_bitwise_failures": 0,
                        "unexercised_rule_count": 0,
                        "unpromoted_domain_count": 0,
                        "required_domain_count": 1,
                        "passing_domain_count": 1,
                        "passing_domain_fraction": 1.0,
                        "domain_promotion_quota_satisfied": True,
                        "p95_regret_budget": 0.05,
                        "minimum_passing_domain_fraction": 0.95,
                        "max_observed_regret": 0.01,
                        "p95_observed_regret": 0.01,
                        "p95_simultaneous_95pct_upper_regret": 0.02,
                        "domain_results": [{
                            "domain": {"test_domain": "cuda-fast-m1"},
                            "sealed_cell_count": 1,
                            "p95_observed_regret": 0.01,
                            "p95_simultaneous_95pct_upper_regret": 0.02,
                            "passes_p95_budget": True,
                        }],
                        "cells": [{}],
                    },
                }, sort_keys=True),
                encoding="utf-8",
            )
            certified_include.write_text(
                "\n".join((
                    f"// Common policy digest: {digest(policy)}",
                    f"// Shape manifest digest: {manifest_digest}",
                    f"// Measurement plan digest: {measurement_plan_digest}",
                    f"// Frozen generic policy digest: {generic_digest}",
                    "// Sealed generic certificate: coverage=1/1",
                    "// immutable staged M1 test include",
                    "",
                )),
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    str(ANALYZER),
                    "--verifier-input",
                    str(verifier_csv),
                    "--build-id",
                    "sha256:verifier-build",
                    "--serial-m1-policy-hash",
                    "sha256:staged-m1",
                    "--output",
                    str(output),
                    "--certified-m1-include",
                    str(certified_include),
                    "--certified-m1-policy-json",
                    str(certified_policy),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            generated = output.read_text(encoding="utf-8")
            self.assertTrue(generated.startswith(
                certified_include.read_text(encoding="utf-8").rstrip()
            ))
            self.assertIn(
                "LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2",
                generated,
            )
            self.assertIn("selectGeneratedGroupedTuning", generated)

    def test_sealed_generic_promotion_is_explicit_and_production_only(self) -> None:
        """Development-only rules cannot masquerade as an installable table."""

        candidate = "cuda.nvnni.decode.fast_m1.kpar.tn256.cpt4.kb32"
        result, _, _ = self.run_analyzer(
            [self.row(candidate, "eager", 20.0)],
            "--certify-generic",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --profile production", result.stderr)

        source = ANALYZER.read_text(encoding="utf-8")
        self.assertIn("def freeze_fast_policy(", source)
        self.assertIn("def certify_fast_policy(", source)
        freeze_source = source[
            source.index("def freeze_fast_policy("):
            source.index("def certify_fast_policy(")
        ]
        self.assertIn("projected_domain_cache_keys", freeze_source)
        self.assertIn("domain_corpus_provider=", freeze_source)
        self.assertIn("domain_corpus_digest_provider=", freeze_source)
        self.assertNotIn(
            "development = project_cuda_shape_resolved_candidates",
            freeze_source,
        )
        self.assertIn("validate_frozen_policy_file", source)
        self.assertIn("validate_certified_m1_artifacts", source)
        self.assertNotIn("def compile_fast_policy(", source)
        self.assertNotIn("compile_policy(", source)
        self.assertNotIn("split.partition(fast)", source)
        self.assertIn("shape.exact_overlay", source)
        self.assertIn("filtering exact overlays changed", source)
        self.assertIn('conditions.append(f"k / 32 >=', source)
        self.assertIn("resolveGeneratedTargetBlocksKBlocks", source)
        self.assertIn("resolveGeneratedCanonicalTargetBlocksKBlocks", source)
        self.assertIn("resolveGeneratedBlocksPerPartitionKBlocks", source)
        self.assertIn("project_cuda_shape_resolved_candidates", source)
        self.assertIn("--freeze-generic", source)
        self.assertIn("--frozen-policy-json", source)
        self.assertIn("--certified-m1-policy-json", source)
        self.assertIn("--policy-json", source)
        self.assertIn("--fit-cache-dir", source)
        self.assertIn("PolicyFitCache(directory=fit_cache_directory)", source)

    def test_formula_tree_emitter_produces_compilable_mode_aware_cpp(self) -> None:
        """A rational tree leaf must resolve a concrete exact KB in C++."""

        spec = importlib.util.spec_from_file_location(
            "cuda_native_vnni_analyzer_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        domain = GenericDomain(
            backend=Backend.CUDA,
            architecture_class="sm86-test",
            semantic_contract=SemanticContract.FAST,
            operation_kind="NativeVNNIDecodeProjection",
            bundle_signature="single",
            prepared_family_id="NativeVNNI_cuda_CB0",
            packing_abi="native-vnni-cuda-cb0-v1",
            runtime_codebook_id=0,
            execution_mode=ExecutionMode.GRAPH_CAPTURED,
            m=1,
            aspect_bucket=AspectBucket.BALANCED,
        )
        rule = GenericDispatchRule(
            domain=domain,
            predicates=(
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.ASPECT_RATIO, 5, 4),
                    True,
                ),
                FeaturePredicate(
                    FeatureThreshold(FeatureAxis.N_TILES_64, 15, 1),
                    True,
                ),
                FeaturePredicate(
                    FeatureThreshold(
                        FeatureAxis.K_GROUPS_PER_N_TILE_64,
                        3,
                        2,
                    ),
                    False,
                ),
            ),
            candidate_id=(
                "cuda.nvnni.decode.fast_m1.kpar_formula."
                "tn128.cpt1.tb328.mkg1"
            ),
            arithmetic_fingerprint="sha256:test-formula",
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
            "enum class NativeGemvShape { WIDE, KPAR, DIRECT, ROWPAR };",
            generated,
            "int main() {",
            "  NativeGemvShape shape{};",
            "  GeneratedDispatchTuning tuning{};",
            "  return selectGeneratedDispatch<0>(true, 1, 900, 1024, shape, tuning) ? 0 : 1;",
            "}",
        ))
        result = subprocess.run(
            ["g++", "-std=c++20", "-x", "c++", "-fsyntax-only", "-"],
            input=source,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_grouped_emitter_prefers_exact_and_covers_unseen_m(self) -> None:
        """Grouped exact overlays precede generic rules with an open M tail."""

        spec = importlib.util.spec_from_file_location(
            "cuda_native_vnni_grouped_analyzer_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def grouped_rule(m: int, candidate_id: str) -> GenericDispatchRule:
            domain = GenericDomain(
                backend=Backend.CUDA,
                architecture_class="sm86-test",
                semantic_contract=(
                    SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                ),
                operation_kind="NativeVNNIDecodeProjection",
                bundle_signature="single",
                prepared_family_id="NativeVNNI_cuda_CB19",
                packing_abi="native-vnni-cuda-cb19-v1",
                runtime_codebook_id=19,
                execution_mode=ExecutionMode.EAGER,
                m=m,
                aspect_bucket=AspectBucket.BALANCED,
            )
            return GenericDispatchRule(
                domain=domain,
                predicates=(),
                candidate_id=candidate_id,
                arithmetic_fingerprint=f"sha256:{candidate_id}",
                development_shape_groups=("shape-a", "shape-b"),
                development_max_regret=0.01,
                development_p95_regret=0.01,
                development_mean_regret=0.01,
            )

        exact = module.GroupedEntry(
            codebook=19,
            execution_mode=ExecutionMode.EAGER,
            m=4,
            n=6144,
            k=5120,
            kernel="tensor_core_mma16",
            grouped_rows=16,
            candidate_id="cuda.nvnni.decode.verifier.tensor_core_mma16",
            shape_name="Qwen36MoE_GDN_ZProjection",
            max_surface_regret=0.01,
            max_cv=0.01,
        )
        generated = module.generate_include(
            [],
            [
                grouped_rule(
                    4,
                    "cuda.nvnni.decode.verifier.inherit_serial_m1.r4",
                ),
                grouped_rule(
                    8,
                    "cuda.nvnni.decode.verifier.tensor_core_mma16",
                ),
            ],
            grouped_entries=[exact],
            corpus_digest="sha256:test-corpus",
            registry_digest="sha256:test-registry",
            profile=MeasurementProfile.QUICK,
        )
        source = "\n".join((
            "enum class NativeGemvShape { WIDE, KPAR, DIRECT, ROWPAR };",
            generated,
            "int main() {",
            "  GeneratedGroupedTuning tuning{};",
            "  if (!selectGeneratedGroupedTuning<19>(false, 4, 6144, 5120, tuning)) return 1;",
            "  if (tuning.kernel != GeneratedGroupedKernel::TensorCoreMma16) return 2;",
            "  if (!selectGeneratedGroupedTuning<19>(false, 5, 6144, 5120, tuning)) return 3;",
            "  if (tuning.kernel != GeneratedGroupedKernel::Dp4aRows || tuning.grouped_rows != 4) return 4;",
            "  if (!selectGeneratedGroupedTuning<19>(false, 1000000, 6144, 5120, tuning)) return 5;",
            "  if (tuning.kernel != GeneratedGroupedKernel::TensorCoreMma16) return 6;",
            "  return 0;",
            "}",
        ))
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "grouped-dispatch"
            result = subprocess.run(
                ["g++", "-std=c++20", "-x", "c++", "-o", str(binary), "-"],
                input=source,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            executed = subprocess.run(
                [str(binary)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(executed.returncode, 0, executed.stderr)

    def test_grouped_generic_installation_requires_full_domain_totality(self) -> None:
        """Every GPU codebook, mode, canonical M, and aspect needs a rule."""

        spec = importlib.util.spec_from_file_location(
            "cuda_native_vnni_grouped_totality_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        def rule(
            codebook: int,
            mode: ExecutionMode,
            m: int,
            aspect: AspectBucket,
        ) -> GenericDispatchRule:
            return GenericDispatchRule(
                domain=GenericDomain(
                    backend=Backend.CUDA,
                    architecture_class="sm86-test",
                    semantic_contract=(
                        SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                    ),
                    operation_kind="NativeVNNIDecodeProjection",
                    bundle_signature="single",
                    prepared_family_id=f"NativeVNNI_cuda_CB{codebook}",
                    packing_abi=f"native-vnni-cuda-cb{codebook}-v1",
                    runtime_codebook_id=codebook,
                    execution_mode=mode,
                    m=m,
                    aspect_bucket=aspect,
                ),
                predicates=(),
                candidate_id=(
                    "cuda.nvnni.decode.verifier.inherit_serial_m1.r2"
                ),
                arithmetic_fingerprint="sha256:grouped-r2",
                development_shape_groups=("shape-a", "shape-b"),
                development_max_regret=0.01,
                development_p95_regret=0.01,
                development_mean_regret=0.01,
            )

        rules = [
            rule(codebook, mode, m, aspect)
            for codebook in {
                format_spec.gpu_execution_codebook_id
                for format_spec in FORMAT_SPECS
            }
            for mode in (ExecutionMode.EAGER, ExecutionMode.GRAPH_CAPTURED)
            for m in (*range(2, 17), 31)
            for aspect in AspectBucket
        ]
        module.validate_grouped_generic_totality(rules)
        with self.assertRaisesRegex(ValueError, "not codebook/mode/M/aspect total"):
            module.validate_grouped_generic_totality(rules[:-1])

    def test_grouped_composition_preserves_certified_m1_bytes(self) -> None:
        """Grouped publication appends to, rather than regenerates, certified M1."""

        spec = importlib.util.spec_from_file_location(
            "cuda_native_vnni_grouped_composition_test_module",
            ANALYZER,
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        certified = "// certified M1 bytes\nconstexpr int sentinel = 7;\n"
        grouped = module.append_grouped_dispatch(
            certified,
            [],
            [
                GenericDispatchRule(
                    domain=GenericDomain(
                        backend=Backend.CUDA,
                        architecture_class="sm86-test",
                        semantic_contract=(
                            SemanticContract.VERIFIER_SERIAL_M1_BITWISE
                        ),
                        operation_kind="NativeVNNIDecodeProjection",
                        bundle_signature="single",
                        prepared_family_id="NativeVNNI_cuda_CB19",
                        packing_abi="native-vnni-cuda-cb19-v1",
                        runtime_codebook_id=19,
                        execution_mode=ExecutionMode.EAGER,
                        m=2,
                        aspect_bucket=AspectBucket.BALANCED,
                    ),
                    predicates=(),
                    candidate_id=(
                        "cuda.nvnni.decode.verifier.inherit_serial_m1.r2"
                    ),
                    arithmetic_fingerprint="sha256:grouped-r2",
                    development_shape_groups=("shape-a", "shape-b"),
                    development_max_regret=0.01,
                    development_p95_regret=0.01,
                    development_mean_regret=0.01,
                )
            ],
            corpus_digest="sha256:test-corpus",
            registry_digest="sha256:test-registry",
            profile=MeasurementProfile.QUICK,
        )

        expected_prefix = certified.rstrip() + "\n\n"
        self.assertTrue(grouped.startswith(expected_prefix))
        self.assertEqual(grouped[: len(expected_prefix)], expected_prefix)
        self.assertIn("LLAMINAR_CUDA_GROUPED_DISPATCH_POLICY_V2", grouped)


if __name__ == "__main__":
    unittest.main()
