#!/usr/bin/env python3
## @file test_native_quantised_gpu_prefill_trainer_policy.py
#  @brief Guard retained tournament timing separately from isolated profiler collection.
"""Source-policy regressions for CUDA/ROCm NativeQuantised prefill trainers."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINER_SOURCES = {
    "cuda": REPO_ROOT
    / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativeQuantisedGemm.cpp",
    "rocm": REPO_ROOT
    / "tests/v2/performance/kernels/rocm/Perf__NativeQuantised_Sweep.cpp",
}


EVENT_WAIT = {
    "cuda": "cudaEventSynchronize(",
    "rocm": "hipEventSynchronize(",
}
STREAM_WAIT = {
    "cuda": "cudaStreamSynchronize(",
    "rocm": "hipStreamSynchronize(",
}
TIMED_REGION_START = "for (int iteration = 0; iteration < bench_runs; ++iteration)"
TIMED_REGION_END = "result.min_us = "


def assert_batched_timing_contract(backend: str, source: str) -> None:
    """Reject any per-sample host bounce in the candidate measurement.

    The timed region is bounded by the bench event-record loop and the
    post-processing aggregate: inside it, every sample records its start and
    stop events on the stream and exactly one batched event wait drains the
    whole run; a stream synchronization there would bounce the host after
    every sample. Checked stream waits outside the region (warmup checker,
    isolated replay completion) are separate call scopes, and exactly one
    void-cast destructor drain is allowed per trainer.
    """
    event_wait = EVENT_WAIT[backend]
    stream_wait = STREAM_WAIT[backend]
    start = source.find(TIMED_REGION_START)
    end = source.find(TIMED_REGION_END, start)
    if start < 0 or end < 0:
        raise AssertionError("trainer lacks the recorded bench timing region")
    region = source[start:end]
    if stream_wait in region:
        raise AssertionError(
            f"{backend} trainer synchronizes the stream inside the timed "
            "candidate region; samples must batch on events only")
    if region.count(event_wait) != 1:
        raise AssertionError(
            f"{backend} trainer must wait once on the batched stop events, "
            f"found {region.count(event_wait)}")
    void_drain = "(void)" + stream_wait
    if source.count(void_drain) != 1:
        raise AssertionError(
            f"{backend} trainer must keep exactly one void-cast teardown "
            f"drain, found {source.count(void_drain)}")
    if source.count(event_wait) != 1:
        raise AssertionError(
            f"{backend} trainer must keep exactly one event wait in total, "
            f"found {source.count(event_wait)}")


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
                self.assertIn(helper, source)
                self.assertEqual(source.count(direction), 1)
                self.assertIn(terminal_size, source)

    def test_trainers_batch_timing_events_before_one_wait(self) -> None:
        """One candidate must not bounce through the host after every sample."""

        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                assert_batched_timing_contract(
                    backend, path.read_text(encoding="utf-8"))
                source = path.read_text(encoding="utf-8")
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
                self.assertIn("bench_runs - 1", source)
                self.assertIn("start_events_.reserve", source)
                self.assertIn("stop_events_.reserve", source)
                self.assertIn("times_us_.resize", source)

    def test_injected_hot_path_stream_wait_is_rejected(self) -> None:
        """The same validator must reject an extra hot-path wait, not count text."""

        for backend, path in TRAINER_SOURCES.items():
            with self.subTest(backend=backend):
                source = path.read_text(encoding="utf-8")
                stream_wait = STREAM_WAIT[backend]
                start = source.index(TIMED_REGION_START)
                end = source.index(TIMED_REGION_END, start)
                injected = (
                    source[:end]
                    + f"    requireHip({stream_wait}stream_), \"injected host bounce\");\n"
                    + source[end:]
                )
                with self.assertRaisesRegex(
                    AssertionError, "synchronizes the stream inside the timed"
                ):
                    assert_batched_timing_contract(backend, injected)

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
