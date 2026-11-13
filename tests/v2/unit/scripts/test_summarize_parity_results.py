#!/usr/bin/env python3
"""Unit tests for production-path parity result summaries."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "summarize_parity_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_parity_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


HEADER = (
    "backend,device,execution_path,execution_topology,graph_contract,"
    "forward_full_graph_capture,forward_full_graph_replay,"
    "full_graph_capture,full_graph_replay,decode_graph_capture,"
    "decode_graph_replay,device_generation_controller,"
    "generation_execution_policy,native_generation_parent,"
    "hosted_ticket_boundary_certified,generation_loop_certified,"
    "generation_certification_detail,"
    "segmented_plan,"
    "segmented_capture,segmented_replay,model_context_reused,elapsed_seconds,"
    "budget_seconds,within_budget\n"
)


class ParityResultSummaryTest(unittest.TestCase):
    def _summarize(self, row: str) -> tuple[str, int, int]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name) / "results"
        case = root / "deadbeef" / "case"
        case.mkdir(parents=True)
        (case / "production_path.csv").write_text(
            HEADER + row + "\n", encoding="utf-8"
        )
        return summary.summarize(root, "parity-results")

    def test_homogeneous_full_graph_evidence_passes(self) -> None:
        markdown, total, failed = self._summarize(
            "CUDA,cuda:0,graph,homogeneous_gpu,homogeneous_gpu_captured,"
            "true,false,true,false,false,true,"
            "false,not_observed,false,false,false,not_observed,false,false,false,true,"
            "12.5,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("Production path and economy", markdown)
        self.assertIn("12.50 / 3600", markdown)
        self.assertIn("| ✅ |", markdown)

    def test_homogeneous_segmented_replay_fails(self) -> None:
        markdown, total, failed = self._summarize(
            "ROCm,rocm:0,graph,homogeneous_gpu,homogeneous_gpu_captured,"
            "true,false,true,false,true,false,"
            "false,not_observed,false,false,false,not_observed,false,false,true,false,"
            "8.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 1))
        self.assertIn("| yes | no |", markdown)
        self.assertIn("| ❌ |", markdown)

    def test_uncertified_host_scheduled_mtp_policy_fails(
        self,
    ) -> None:
        markdown, total, failed = self._summarize(
            "ROCm,rocm:0,graph,homogeneous_gpu,homogeneous_gpu_captured,"
            "true,false,false,false,true,true,"
            "true,host_scheduled_captured_transactions,false,"
            "false,false,missing_ticket,false,false,false,false,9.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 1))
        self.assertIn("host_scheduled_captured_transactions", markdown)
        self.assertIn("| ❌ |", markdown)

    def test_certified_rocm_ticket_selected_captured_transactions_pass(
        self,
    ) -> None:
        markdown, total, failed = self._summarize(
            "ROCm,rocm:0,graph,homogeneous_gpu,homogeneous_gpu_captured,"
            "true,true,false,false,true,true,"
            "true,host_scheduled_captured_transactions,false,"
            "true,true,certified_rocm_ticket_boundary,false,false,false,true,"
            "9.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("host_scheduled_captured_transactions", markdown)
        self.assertIn("| ✅ |", markdown)

    def test_native_cuda_generation_parent_passes(self) -> None:
        markdown, total, failed = self._summarize(
            "CUDA,cuda:0,graph,homogeneous_gpu,homogeneous_gpu_captured,"
            "true,true,true,true,true,true,"
            "true,native_conditional_parent,true,false,true,"
            "native_conditional_parent,false,false,false,true,7.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("native_conditional_parent", markdown)
        self.assertIn("| ✅ |", markdown)

    def test_uncaptured_heterogeneous_accelerator_fails(self) -> None:
        markdown, total, failed = self._summarize(
            "ExpertOverlay,cpu:0,graph,heterogeneous_accelerator,"
            "heterogeneous_coordinator_segmented,"
            "true,true,true,true,true,true,false,not_observed,false,false,false,"
            "not_observed,false,false,false,false,7.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 1))
        self.assertIn("| ❌ |", markdown)

    def test_captured_heterogeneous_segments_pass(self) -> None:
        markdown, total, failed = self._summarize(
            "ExpertOverlay,cuda:0,graph,heterogeneous_accelerator,"
            "heterogeneous_coordinator_segmented,"
            "false,false,false,false,true,true,false,not_observed,false,false,false,"
            "not_observed,true,true,true,false,7.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("| ✅ |", markdown)


class FinalLogitStepReportTest(unittest.TestCase):
    """Synthetic recorded metrics exercise the real reporter, not a numerical oracle."""

    def test_per_step_means_and_original_reds_are_not_averaged_away(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            case = root / "source" / "same-model"
            case.mkdir(parents=True)
            (case / "decode_steps.csv").write_text(
                "backend,step,cosine,kl_divergence,top1_overlap,top5_overlap,passed\n"
                "CPU,0,1,.01,1,.8,true\nCPU,1,1,.15,.5,.9,false\n", encoding="utf-8")
            markdown, total, failed = summary.summarize(root, "original")
        self.assertEqual((total, failed), (1, 1))
        self.assertIn("Final logits at each decode step", markdown)
        self.assertIn("| 0 | 0.010000 | 1.0000 | 0.8000 | ✅ |", markdown)
        self.assertIn("| 1 | 0.150000 | 0.5000 | 0.9000 | ❌ |", markdown)
        self.assertIn("set intersection/5", markdown)
        self.assertIn("distinct from reference-top1-in-topK", markdown)

    def test_historical_extra_steps_remain_visible_and_legacy_aggregate_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            case = root / "source" / "historical"
            case.mkdir(parents=True)
            (case / "decode_steps.csv").write_text(
                "backend,step,cosine,kl_divergence,top1_overlap,top5_overlap,passed\n"
                "Other,0,1,.01,1,1,true\nOther,1,1,.02,1,1,true\nOther,2,1,.7,0,.6,false\n", encoding="utf-8")
            markdown, total, failed = summary.summarize(root, "original")
        self.assertEqual((total, failed), (1, 1))
        self.assertIn("Decode (averaged across steps)", markdown)
        self.assertIn("| 2 | 0.700000 | 0.0000 | 0.6000 | ❌ |", markdown)


if __name__ == "__main__":
    unittest.main()
