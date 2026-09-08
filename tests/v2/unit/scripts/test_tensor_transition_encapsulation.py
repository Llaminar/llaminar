#!/usr/bin/env python3
"""Focused regressions for the tensor transition encapsulation sanitizer."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_tensor_transition_encapsulation import validate  # noqa: E402


class TestTensorTransitionEncapsulation(unittest.TestCase):
    """Prove raw calls are rejected without matching comments or literals."""

    def validate_sources(self, sources: dict[str, str]) -> list[str]:
        """Create a minimal repository and run the policy against it."""

        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            for relative_path, contents in sources.items():
                path = root / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            return validate(root)

    def test_rejects_direct_transition(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/stages/BadStage.cpp": """
                    void publish(TensorBase *tensor) {
                        tensor->transitionTo(
                            TensorCoherenceState::DEVICE_AUTHORITATIVE);
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("BadStage.cpp", failures[0])
        self.assertIn("transitionTo()", failures[0])

    def test_rejects_direct_event_transition_in_tests(self) -> None:
        failures = self.validate_sources(
            {
                "tests/v2/integration/BadKernel.cpp": """
                    void publish(TensorBase *tensor, void *stream) {
                        tensor->transitionToWithEvent(
                            TensorCoherenceState::DEVICE_AUTHORITATIVE,
                            std::nullopt,
                            stream);
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("transitionToWithEvent()", failures[0])

    def test_rejects_direct_host_publication(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/stages/BadHostWriter.cpp": """
                    void publish(ITensor *tensor) {
                        tensor->mark_host_dirty();
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("BadHostWriter.cpp", failures[0])
        self.assertIn("mark_host_dirty()", failures[0])

    def test_rejects_unqualified_inherited_transition(self) -> None:
        failures = self.validate_sources(
            {
                "tests/v2/unit/BadTensor.cpp": """
                    bool ensureOnDevice(DeviceId device) {
                        transitionTo(TensorCoherenceState::SYNCED);
                        return true;
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("BadTensor.cpp", failures[0])
        self.assertIn("transitionTo()", failures[0])

    def test_rejects_private_publication_hook(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/stages/BadGraphStage.cpp": """
                    void publish(TensorBase *tensor, DeviceId device) {
                        tensor->publishGraphOwnedDeviceWriteState(device);
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("BadGraphStage.cpp", failures[0])
        self.assertIn("publishGraphOwnedDeviceWriteState()", failures[0])

    def test_rejects_raw_completion_event_clear(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/stages/BadEventOwner.cpp": """
                    void discard(TensorBase *tensor) {
                        tensor->clearCompletionEvent();
                    }
                """,
            }
        )
        self.assertEqual(len(failures), 1)
        self.assertIn("BadEventOwner.cpp", failures[0])
        self.assertIn("clearCompletionEvent()", failures[0])

    def test_ignores_comments_and_string_literals(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/stages/GoodStage.cpp": r'''
                    // tensor->transitionTo(DEVICE_AUTHORITATIVE);
                    constexpr auto diagnostic =
                        "tensor->transitionToWithEvent(DEVICE_AUTHORITATIVE)";
                    void publish(TensorBase *tensor, DeviceId device) {
                        TransferEngine::publishDeviceWrite(tensor, device);
                    }
                ''',
            }
        )
        self.assertEqual(failures, [])

    def test_allows_transfer_engine_and_slice_delegate(self) -> None:
        failures = self.validate_sources(
            {
                "src/v2/transfer/TransferEngine.cpp": """
                    void publish(TensorBase *tensor) {
                        tensor->publishDeviceWriteStateWithEvent(
                            DeviceId::cuda(0));
                    }
                """,
                "src/v2/tensors/TensorSlice.h": """
                    void delegate(TensorBase *inner) {
                        inner->publishGraphOwnedDeviceWriteState(
                            DeviceId::cuda(0));
                    }
                """,
            }
        )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
