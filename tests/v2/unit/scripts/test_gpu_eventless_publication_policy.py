#!/usr/bin/env python3
"""Regression tests for the GPU eventless-publication source policy."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_gpu_eventless_publication_policy import (  # noqa: E402
    scan_gpu_kernel_publications,
    scan_file,
    validate,
)


class TestGPUEventlessPublicationPolicy(unittest.TestCase):
    """Prove caller attribution and publication-order detection are stable."""

    def make_repo(
        self,
        source: str,
        relative_path: str = "src/v2/example.cpp",
    ) -> tuple[tempfile.TemporaryDirectory, pathlib.Path, pathlib.Path]:
        """Create a minimal policy-covered repository for one source fixture."""

        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        source_path = root / relative_path
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return temporary, root, source_path

    def test_comments_and_literals_do_not_create_false_positives(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            void harmless() {
                // cudaStreamSynchronize(stream);
                const char *example =
                    "tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE)";
            }
            """
        )
        with temporary:
            self.assertEqual(scan_file(root, source_path), [])
            self.assertEqual(validate(root), [])

    def test_raw_stream_sync_then_eventless_publication_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publishAfterHostFence(Tensor *tensor, void *stream) {
                cudaStreamSynchronize(stream);
                doUnrelatedWork();
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("raw_stream", failures[0])
            self.assertIn("publishAfterHostFence", failures[0])

    def test_blocking_backend_copy_then_eventless_publication_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            bool publishAfterCopy(
                IBackend *backend,
                Tensor *tensor,
                void *host,
                void *device) {
                if (!backend->hostToDevice(device, host, 64, 0))
                    return false;
                tensor->transitionTo(
                    llaminar2::TensorCoherenceState::DEVICE_AUTHORITATIVE);
                return true;
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("backend_sync_copy", failures[0])
            self.assertIn("publishAfterCopy", failures[0])

    def test_blocking_event_wait_then_eventless_publication_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            bool publishAfterEvent(
                IBackend *backend,
                Tensor *tensor,
                void *event) {
                if (!backend->waitForEvent(event, 0))
                    return false;
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
                return true;
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("backend_event", failures[0])

    def test_graph_owned_transfer_publication_after_sync_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publishGraphOutputAfterFence(
                ITensor *tensor,
                DeviceId device,
                void *stream) {
                hipStreamSynchronize(stream);
                TransferEngine::publishGraphOwnedDeviceWrite(tensor, device);
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("raw_stream", failures[0])
            self.assertIn("publishGraphOutputAfterFence", failures[0])

    def test_arena_flags_only_publication_after_sync_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publishArenaOutputAfterFence(
                BufferArena *arena,
                BufferId id,
                DeviceId device,
                void *stream) {
                cudaStreamSynchronize(stream);
                arena->markWrittenFlagsOnly(id, device);
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("raw_stream", failures[0])
            self.assertIn("publishArenaOutputAfterFence", failures[0])

    def test_stage_flags_only_publication_after_sync_fails(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publishStageOutputsAfterFence(
                const std::vector<CoherenceBuffer> &outputs,
                void *stream) {
                hipStreamSynchronize(stream);
                markOutputsDirtyFlagsOnly(outputs);
            }
            """
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("raw_stream", failures[0])
            self.assertIn("publishStageOutputsAfterFence", failures[0])

    def test_event_publication_after_sync_is_not_eventless(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publishRecordedEvent(
                Tensor *tensor,
                DeviceId device,
                void *stream) {
                cudaStreamSynchronize(stream);
                tensor->transitionToWithEvent(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE,
                    device,
                    stream);
            }
            """
        )
        with temporary:
            self.assertEqual(validate(root), [])

    def test_sync_in_different_caller_does_not_taint_publication(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void explicitHostBoundary(void *stream) {
                hipStreamSynchronize(stream);
            }

            void legacyPublication(Tensor *tensor) {
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
            """
        )
        with temporary:
            self.assertEqual(validate(root), [])

    def test_same_named_test_bodies_do_not_share_sync_state(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            TEST_F(Fixture, HostBoundary) {
                hipStreamSynchronize(stream);
            }

            TEST_F(Fixture, LegacyPublication) {
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
            """
        )
        with temporary:
            self.assertEqual(validate(root), [])

    def test_publication_before_later_sync_does_not_match_sequence(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void publicationThenHostBoundary(Tensor *tensor, void *stream) {
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
                hipStreamSynchronize(stream);
            }
            """
        )
        with temporary:
            self.assertEqual(validate(root), [])

    def test_integration_sources_are_covered(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void integrationAntiPattern(Tensor *tensor, void *stream) {
                cudaStreamSynchronize(stream);
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
            """,
            "tests/v2/integration/example.cpp",
        )
        with temporary:
            failures = validate(root)
            self.assertEqual(len(failures), 1)
            self.assertIn("tests/v2/integration/example.cpp", failures[0])

    def test_unit_coherence_fixtures_are_outside_policy_roots(self) -> None:
        temporary, root, _ = self.make_repo(
            """
            void stateMachineFixture(Tensor *tensor, void *stream) {
                cudaStreamSynchronize(stream);
                tensor->transitionTo(
                    TensorCoherenceState::DEVICE_AUTHORITATIVE);
            }
            """,
            "tests/v2/unit/coherence_fixture.cpp",
        )
        with temporary:
            self.assertEqual(validate(root), [])

    def test_cuda_kernel_publication_requires_explicit_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor, int ordinal) {
                kernel<<<1, 32, 0, stream>>>();
                TransferEngine::publishDeviceWrite(
                    tensor,
                    DeviceId::cuda(ordinal));
                return true;
            }
            """,
            "src/v2/kernels/cuda/ops/Example.cu",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
            self.assertIn("omits", violations[0].reason)
            self.assertEqual(len(validate(root)), 1)

    def test_rocm_kernel_publication_rejects_explicit_null_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor, int ordinal) {
                hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, stream);
                llaminar2::TransferEngine::publishDeviceWrite(
                    tensor,
                    DeviceId::rocm(ordinal),
                    nullptr);
                return true;
            }
            """,
            "src/v2/kernels/rocm/ops/Example.hip",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
            self.assertIn("null/default", violations[0].reason)

    def test_current_device_publication_requires_explicit_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor) {
                kernel<<<1, 32, 0, stream>>>();
                TransferEngine::publishCurrentDeviceWrite(tensor);
                return true;
            }
            """,
            "tests/v2/integration/kernels/cuda/Example.cu",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
            self.assertIn("publishCurrentDeviceWrite", violations[0].reason)
            self.assertIn("omits", violations[0].reason)

    def test_current_device_publication_rejects_null_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor) {
                hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, stream);
                TransferEngine::publishCurrentDeviceWrite(tensor, nullptr);
                return true;
            }
            """,
            "tests/v2/performance/kernels/rocm/Example.hip",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
            self.assertIn("publishCurrentDeviceWrite", violations[0].reason)
            self.assertIn("null/default", violations[0].reason)

    def test_current_device_publication_accepts_exact_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor, void *stream) {
                kernel<<<1, 32, 0, stream>>>();
                TransferEngine::publishCurrentDeviceWrite(tensor, stream);
                return true;
            }
            """,
            "tests/v2/integration/kernels/cuda/Example.cu",
        )
        with temporary:
            self.assertEqual(
                scan_gpu_kernel_publications(root, source_path),
                [],
            )

    def test_compute_stage_publication_requires_explicit_stream(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool run(ITensor *tensor, DeviceId device) {
                TransferEngine::publishDeviceWrite(tensor, device);
                return true;
            }
            """,
            "src/v2/execution/compute_stages/stages/Example.cpp",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
            self.assertIn("omits", violations[0].reason)

    def test_completed_publication_is_rejected_from_kernel_code(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor, DeviceId device) {
                launchKernel(tensor);
                TransferEngine::publishCompletedDeviceWrite(tensor, device);
                return true;
            }
            """,
            "src/v2/kernels/cuda/ops/Example.cu",
        )
        with temporary:
            violations = scan_gpu_kernel_publications(root, source_path)
            self.assertEqual(len(violations), 1)
        self.assertIn("synchronous transfer or collective owner", violations[0].reason)

    def test_completed_publication_is_allowed_in_transfer_engine(self) -> None:
        temporary, root, source_path = self.make_repo(
            """
            void TransferEngine::copyCompleted(Tensor *tensor, DeviceId device) {
                TransferEngine::publishCompletedDeviceWrite(tensor, device);
            }
            """,
            "src/v2/transfer/TransferEngine.cpp",
        )
        with temporary:
            self.assertEqual(
                scan_gpu_kernel_publications(root, source_path),
                [],
            )

    def test_gpu_kernel_publication_accepts_nested_explicit_stream_expression(
        self,
    ) -> None:
        temporary, root, source_path = self.make_repo(
            """
            bool launch(Tensor *tensor, int ordinal) {
                TransferEngine::publishDeviceWrite(
                    tensor,
                    DeviceId::cuda(selectOrdinal(ordinal, fallback())),
                    context.streamFor(DeviceId::cuda(ordinal)));
                return true;
            }
            """,
            "src/v2/kernels/cuda/ops/Example.cpp",
        )
        with temporary:
            self.assertEqual(
                scan_gpu_kernel_publications(root, source_path),
                [],
            )
            self.assertEqual(validate(root), [])


if __name__ == "__main__":
    unittest.main()
