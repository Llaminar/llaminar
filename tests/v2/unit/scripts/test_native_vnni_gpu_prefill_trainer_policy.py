#!/usr/bin/env python3
"""Source-policy regressions for CUDA/ROCm NativeVNNI prefill trainers."""

from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
TRAINER_SOURCES = {
    "cuda": REPO_ROOT
    / "tests/v2/performance/kernels/cuda/gemm/Perf__CUDANativeVNNIGemm.cpp",
    "rocm": REPO_ROOT
    / "tests/v2/performance/kernels/rocm/Perf__NativeVNNI_Sweep.cpp",
}


class NativeVNNIGPUPrefillTrainerPolicyTest(unittest.TestCase):
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
                self.assertEqual(source.count(stream_wait), 2)
                self.assertIn("bench_runs - 1", source)
                self.assertIn("start_events_.reserve", source)
                self.assertIn("stop_events_.reserve", source)
                self.assertIn("times_us_.resize", source)

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
            "rocm": "LLAMINAR_ROCM_NVNNI_SWEEP_TIMING_CSV",
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
