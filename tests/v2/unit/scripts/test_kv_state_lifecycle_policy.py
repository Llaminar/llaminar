#!/usr/bin/env python3
"""Regression tests for the KV-state lifecycle source sanitizer."""

from __future__ import annotations

import pathlib
import sys
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_kv_state_lifecycle_policy import (  # noqa: E402
    extract_function,
    validate_gpu_reset_source,
    validate_interface,
    validate_orchestrator,
    validate_terminal_hidden_mailbox_lifetime,
)


class TestKVStateLifecyclePolicy(unittest.TestCase):
    """Prove the sanitizer rejects the retired and unsafe lifecycle forms."""

    def test_interface_requires_all_typed_reset_methods(self) -> None:
        source = """
        class IKVCache {
          enum class StateRole {};
          struct StateOwnership {};
          enum class StateResetBoundary {};
          struct StateResetContext {
            void *execution_stream;
            const char *reason;
          };
          void bindStateOwnership();
          virtual bool resetRequestState(const StateResetContext &) = 0;
          virtual bool resetSequenceState(int, const StateResetContext &) = 0;
          virtual bool resetLayerSequenceState(
              int, int, const StateResetContext &) = 0;
          virtual bool resetLayerState(int, const StateResetContext &) = 0;
        };
        """
        self.assertEqual(validate_interface(source), [])
        self.assertTrue(validate_interface(source.replace(
            "virtual bool resetLayerState", "bool resetLayerState"
        )))

    def test_interface_rejects_old_clear_vocabulary(self) -> None:
        source = """
        class IKVCache {
          enum class StateRole {};
          struct StateOwnership {};
          enum class StateResetBoundary {};
          struct StateResetContext {
            void *execution_stream;
            const char *reason;
          };
          void bindStateOwnership();
          virtual bool resetRequestState(const StateResetContext &) = 0;
          virtual bool resetSequenceState(int, const StateResetContext &) = 0;
          virtual bool resetLayerSequenceState(
              int, int, const StateResetContext &) = 0;
          virtual bool resetLayerState(int, const StateResetContext &) = 0;
          virtual void clear() = 0;
        };
        """
        self.assertTrue(any("obsolete" in failure for failure in validate_interface(source)))

    def test_gpu_reset_rejects_sync_and_implicit_stream(self) -> None:
        source = """
        bool Cache::resetRequestState(const StateResetContext &context) {
          if (!context.execution_stream || !context.hasReason()) return false;
          cudaStreamSynchronize(getDefaultStream());
          return true;
        }
        """
        body = extract_function(source, "Cache::resetRequestState")
        self.assertIn("cudaStreamSynchronize", body)
        failures = validate_gpu_reset_source("CUDA", "Cache", source)
        self.assertTrue(any("forbidden" in failure for failure in failures))

    def test_orchestrator_requires_ordered_single_stream_transaction(self) -> None:
        header = """
        void resetInferenceState(const InferenceStateResetRequest &request) {
          RequestStateResetTransaction reset_transaction;
          joinPriorDeviceWorkForRequestStateReset(
              reset_transaction.executionStream());
          state_.resetCommittedKVAndRecurrentState(
              reset_transaction.cacheContext());
          state_.resetMTPShiftedSidecarState(
              reset_transaction.cacheContext());
          publishRequestStateResetReady(reset_transaction.executionStream());
          reset_transaction.markPublished();
        }
        """
        self.assertEqual(validate_orchestrator(header, ""), [])
        reordered = header.replace(
            "state_.resetCommittedKVAndRecurrentState(\n"
            "              reset_transaction.cacheContext());",
            "",
        ).replace(
            "publishRequestStateResetReady(reset_transaction.executionStream());",
            "publishRequestStateResetReady(reset_transaction.executionStream());\n"
            "          state_.resetCommittedKVAndRecurrentState(\n"
            "              reset_transaction.cacheContext());",
        )
        self.assertTrue(validate_orchestrator(reordered, ""))

    def test_terminal_hidden_mailbox_rejects_runtime_rebinding(self) -> None:
        source = """
        bool DeviceGraphOrchestrator::initializeBuffers(int seq_len) {
          const auto rows = resolveMTPTerminalHiddenRowCapacity(1, config.mtp);
          arena_->registerBuffer(
              BufferId::PREFIX_TERMINAL_HIDDEN, rows, d_model, "FP32", device);
          state_.prefix_terminal_hidden =
              arena_->getSharedTensor(BufferId::PREFIX_TERMINAL_HIDDEN);
          return true;
        }
        bool DeviceGraphOrchestrator::ensureMTPTerminalHiddenBuffer(int rows) {
          const auto arena_owner = arena_->getSharedTensor(
              BufferId::PREFIX_TERMINAL_HIDDEN);
          return state_.prefix_terminal_hidden == arena_owner;
        }
        """
        self.assertEqual(validate_terminal_hidden_mailbox_lifetime(source), [])

        rebound = source.replace(
            "return state_.prefix_terminal_hidden == arena_owner;",
            "state_.prefix_terminal_hidden = factory->createFP32({rows, d_model});\n"
            "          arena_->bindExternalBuffer(\n"
            "              BufferId::PREFIX_TERMINAL_HIDDEN,\n"
            "              state_.prefix_terminal_hidden.get());\n"
            "          return true;",
        )
        failures = validate_terminal_hidden_mailbox_lifetime(rebound)
        self.assertTrue(any("forbidden" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
