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
    "backend,device,execution_path,homogeneous_gpu,"
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
            "CUDA,cuda:0,graph,true,true,false,true,false,false,true,"
            "false,not_observed,false,false,false,not_observed,false,false,false,true,"
            "12.5,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("Production path and economy", markdown)
        self.assertIn("12.50 / 3600", markdown)
        self.assertIn("| ✅ |", markdown)

    def test_homogeneous_segmented_replay_fails(self) -> None:
        markdown, total, failed = self._summarize(
            "ROCm,rocm:0,graph,true,true,false,true,false,true,false,"
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
            "ROCm,rocm:0,graph,true,true,false,false,false,true,true,"
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
            "ROCm,rocm:0,graph,true,true,true,false,false,true,true,"
            "true,host_scheduled_captured_transactions,false,"
            "true,true,certified_rocm_ticket_boundary,false,false,false,true,"
            "9.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("host_scheduled_captured_transactions", markdown)
        self.assertIn("| ✅ |", markdown)

    def test_native_cuda_generation_parent_passes(self) -> None:
        markdown, total, failed = self._summarize(
            "CUDA,cuda:0,graph,true,true,true,true,true,true,true,"
            "true,native_conditional_parent,true,false,true,"
            "native_conditional_parent,false,false,false,true,7.0,3600,true"
        )

        self.assertEqual((total, failed), (1, 0))
        self.assertIn("native_conditional_parent", markdown)
        self.assertIn("| ✅ |", markdown)


if __name__ == "__main__":
    unittest.main()
