"""Guard the benchmark frontend's use of production prefill graph policy.

The benchmark must time the same chunked captured prefill path as HTTP serving.
In particular, a 512-token prompt can exceed a PP participant's 256-row
activation arena without becoming one oversized graph submission. This source
policy test protects the frontend boundary without loading a model or device;
the captured PP integration and real-model E2E tests prove the runtime side.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[4]
BENCHMARK_MODE = ROOT / "src/v2/app/modes/BenchmarkMode.cpp"
BENCHMARK_RUNNER = ROOT / "src/v2/utils/BenchmarkRunner.cpp"
INFERENCE_ADAPTER = ROOT / "src/v2/app/InferenceRunnerAdapter.cpp"
ORCHESTRATION_RUNNER = ROOT / "src/v2/execution/runner/OrchestrationRunner.cpp"


class BenchmarkPrefillProductionPolicyTest(unittest.TestCase):
    def test_benchmark_does_not_override_graph_or_bucket_policy(self) -> None:
        mode = BENCHMARK_MODE.read_text(encoding="utf-8")
        self.assertNotRegex(mode, re.compile(r"\b(?:setenv|putenv|unsetenv|mutableDebugEnv)\s*\("))
        self.assertNotIn("BenchmarkPrefillBucketPolicy", mode)
        self.assertIn("logBenchmarkPrefillPolicy();", mode)
        self.assertIn("runner->prepareForInference()", mode)

    def test_benchmark_prefill_reaches_production_chunk_scheduler(self) -> None:
        benchmark = BENCHMARK_RUNNER.read_text(encoding="utf-8")
        adapter = INFERENCE_ADAPTER.read_text(encoding="utf-8")
        production = ORCHESTRATION_RUNNER.read_text(encoding="utf-8")
        self.assertIn("success = runner_->forward(tokens.data(), tokens.size());", benchmark)
        self.assertIn("orch_runner_->prefill(token_vec)", adapter)
        self.assertIn("runner_->forwardPrefillChunkSchedule(", production)
        self.assertIn("const bool long_bucketed_prefill =", production)


if __name__ == "__main__":
    unittest.main()
