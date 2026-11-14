#!/usr/bin/env python3
## @file test_native_quantised_gpu_prefill_trainer_policy.py
#  @brief Guard retained tournament timing separately from isolated profiler collection.
"""Source-policy regressions for CUDA/ROCm NativeQuantised prefill trainers."""

from __future__ import annotations

import unittest
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINER_SOURCES = {
    "cuda": REPO_ROOT
    / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativeQuantisedGemm.cpp",
    "rocm": REPO_ROOT
    / "tests/v2/performance/kernels/rocm/Perf__NativeQuantised_Sweep.cpp",
}


def owned_scope(source: str, marker: str) -> str:
    """Extract one complete named C++ scope, ignoring comments and quoted braces.

    Scope admission is exact: missing, duplicated or unclosed owners fail rather
    than allowing an unrelated diagnostic to enlarge tournament allowances.
    """
    if source.count(marker) != 1:
        raise ValueError(f"expected one source owner: {marker}")
    start = source.index(marker)
    depth = 0
    tokens = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]', re.S)
    for token in tokens.finditer(source, start):
        if token.group() == '{':
            depth += 1
        elif token.group() == '}':
            depth -= 1
            if depth == 0:
                return source[start:token.end()]
            if depth < 0:
                break
    raise ValueError(f"unclosed source owner: {marker}")


def prepared_scope(source: str, backend: str) -> str:
    """Keep all prepared execution methods within the tournament policy boundary."""
    owner = 'PreparedROCmSweepExecution' if backend == 'rocm' else 'PreparedSweepExecution'
    return owned_scope(source, f'class {owner}\n')


def arithmetic_diagnostic(source: str) -> str:
    """The standalone whole-row regression owns its explicit full-output oracle."""
    return owned_scope(source, 'TEST_F(NativeQuantisedSweepTest, WmmaWholeRowScaleAndPartialKReplay)')


class NativeQuantisedGPUPrefillTrainerPolicyTest(unittest.TestCase):
    """Keep candidate timing persistent, stream-local, and device-certified."""

    def test_trainers_forbid_dynamic_gpu_memory_and_full_output_downloads(self) -> None:
        """Candidate tournaments may materialize only terminal counters."""

        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                source = path.read_text(encoding="utf-8")
                for forbidden in (
                    "cudaMalloc(",
                    "cudaFree(",
                    "hipMalloc(",
                    "hipFree(",
                    "cudaDeviceSynchronize(",
                    "hipDeviceSynchronize(",
                    "ensureOnHost(",
                ):
                    self.assertNotIn(forbidden, source)

                helper = (
                    "enqueueCudaFP32ByteComparison"
                    if backend == "cuda"
                    else "enqueueROCmFP32ByteComparison"
                )
                direction = (
                    "cudaMemcpyDeviceToHost"
                    if backend == "cuda"
                    else "hipMemcpyDeviceToHost"
                )
                terminal_size = (
                    "sizeof(result)"
                    if backend == "cuda"
                    else "sizeof(certificate)"
                )
                prepared = prepared_scope(source, backend)
                diagnostic = arithmetic_diagnostic(source) if backend == 'rocm' else ''
                self.assertIn(helper, prepared)
                self.assertEqual(prepared.count(direction), 1)
                self.assertEqual(source.count(direction), prepared.count(direction) + diagnostic.count(direction))
                self.assertIn(terminal_size, prepared)

    def test_trainers_batch_timing_events_before_one_wait(self) -> None:
        """One candidate must not bounce through the host after every sample."""

        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                complete_source = path.read_text(encoding="utf-8")
                source = prepared_scope(complete_source, backend)
                event_wait = (
                    "cudaEventSynchronize("
                    if backend == "cuda"
                    else "hipEventSynchronize("
                )
                stream_wait = (
                    "cudaStreamSynchronize("
                    if backend == "cuda"
                    else "hipStreamSynchronize("
                )
                self.assertEqual(source.count(event_wait), 1)
                profile_waits = 0
                if backend == "rocm":
                    # The opt-in profiler owns one terminal join inside its
                    # ROCTx range. It is outside the candidate sample loop.
                    start = source.index("void profileExactCandidate(")
                    end = source.index("\n    private:", start)
                    profile = source[start:end]
                    self.assertEqual(profile.count(stream_wait), 1)
                    resume = profile.index("roctxProfilerResume(0)")
                    launch = profile.index("const bool launched = graph.launch();")
                    completion = profile.index(stream_wait)
                    pause = profile.index("roctxProfilerPause(0)")
                    self.assertLess(resume, launch)
                    self.assertLess(launch, completion)
                    self.assertLess(completion, pause)
                    profile_waits = 1
                self.assertEqual(source.count(stream_wait) - profile_waits, 2)
                diagnostic = arithmetic_diagnostic(complete_source) if backend == 'rocm' else ''
                self.assertEqual(complete_source.count(stream_wait),
                                 source.count(stream_wait) + diagnostic.count(stream_wait))
                self.assertEqual(complete_source.count(event_wait), source.count(event_wait))
                self.assertIn("bench_runs - 1", source)
                self.assertIn("start_events_.reserve", source)
                self.assertIn("stop_events_.reserve", source)
                self.assertIn("times_us_.resize", source)

    def test_whole_row_diagnostic_has_an_explicit_retained_byte_oracle(self) -> None:
        """Only the named arithmetic regression may download full output planes."""
        scope = arithmetic_diagnostic(TRAINER_SOURCES['rocm'].read_text())
        self.assertEqual(scope.count('hipMemcpyDeviceToHost'), 2)
        self.assertEqual(scope.count('hipStreamSynchronize('), 2)
        self.assertNotIn('hipEventSynchronize(', scope)
        for required in ('ScopedBackendGraphCapture', 'graph.instantiate()', 'graph.launch()',
                         'hipMemsetAsync(', 'std::bit_cast<uint32_t>', 'EXPECT_EQ(observed, reference)'):
            self.assertIn(required, scope)

    def test_named_diagnostic_cannot_hide_tournament_transfers(self) -> None:
        """An extra download/wait in the prepared owner cannot borrow diagnostic counts."""
        source = TRAINER_SOURCES['rocm'].read_text()
        prepared = prepared_scope(source, 'rocm')
        for forbidden, expected in [('hipMemcpyDeviceToHost', 1), ('hipStreamSynchronize(', 3)]:
            changed = source.replace(prepared, prepared.replace('{', '{\n' + forbidden + ';', 1), 1)
            self.assertEqual(arithmetic_diagnostic(changed), arithmetic_diagnostic(source))
            with self.assertRaises(AssertionError):
                self.assertEqual(prepared_scope(changed, 'rocm').count(forbidden), expected)
        for malformed in ('', 'class Missing\n{}; class Missing\n{};', 'class Missing\n{'):
            with self.assertRaises(ValueError):
                owned_scope(malformed, 'class Missing\n')

    def test_trainers_require_explicit_non_default_streams(self) -> None:
        """Every setup, launch, comparison, and terminal copy shares one stream."""

        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                source = path.read_text(encoding="utf-8")
                self.assertIn("ScopedGPUStream", source)
                self.assertIn("stream_owner_->get()", source)
                self.assertIn("kernel_->setGPUStream", source)
                self.assertNotIn("ensureOnDevice(DeviceId::cuda(0))", source)
                self.assertNotIn("ensureOnDevice(DeviceId::rocm(0))", source)

    def test_trainers_publish_every_native_event_sample_to_a_sidecar(self) -> None:
        """Aggregate statistics must retain their raw measurement authority."""

        expected_settings = {
            "cuda": "LLAMINAR_TILE_SWEEP_TIMING_CSV",
            "rocm": "LLAMINAR_ROCM_NATIVE_QUANTISED_SWEEP_TIMING_CSV",
        }
        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                source = path.read_text(encoding="utf-8")
                self.assertIn(expected_settings[backend], source)
                self.assertIn("latency_us_hex", source)
                self.assertIn("timingSamples()", source)
                self.assertIn("sample_index < samples.size()", source)
                self.assertIn("timed_replays", source)


if __name__ == "__main__":
    unittest.main()
