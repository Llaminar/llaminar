#!/usr/bin/env python3
"""Regression tests for the GPU blocking-synchronization source policy."""

from __future__ import annotations

import pathlib
import dataclasses
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_gpu_blocking_sync_policy import (  # noqa: E402
    ALLOWANCES,
    Allowance,
    scan_file,
    function_intervals,
    strip_comments_and_literals,
    validate,
)


class TestGPUBlockingSyncPolicy(unittest.TestCase):
    """Prove that the allowlist is exact and the scanner is lexical."""

    def make_repo(self, source: str) -> tuple[tempfile.TemporaryDirectory, pathlib.Path]:
        """Create the minimal source tree consumed by the policy scanner."""

        temporary = tempfile.TemporaryDirectory()
        root = pathlib.Path(temporary.name)
        source_path = root / "src" / "v2" / "example.cpp"
        source_path.parent.mkdir(parents=True)
        source_path.write_text(source, encoding="utf-8")
        return temporary, root

    @staticmethod
    def allowance(count: int = 1) -> Allowance:
        """Return the reviewed budget for the fixture's explicit boundary."""

        return Allowance(
            path="src/v2/example.cpp",
            caller="hostResultBoundary",
            kind="raw_stream",
            count=count,
            category="host_result",
            reason="fixture host-result boundary",
        )

    def test_comments_and_literals_do_not_create_false_positives(self) -> None:
        temporary, root = self.make_repo(
            """
            void harmless() {
                // cudaStreamSynchronize(stream);
                const char *name = "hipStreamSynchronize(stream)";
            }
            """
        )
        with temporary:
            source_path = root / "src" / "v2" / "example.cpp"
            self.assertEqual(scan_file(root, source_path), [])
            self.assertEqual(validate(root, ()), [])

    def test_unreviewed_caller_fails(self) -> None:
        temporary, root = self.make_repo(
            """
            void hostResultBoundary(void *stream) {
                cudaStreamSynchronize(stream);
            }
            """
        )
        with temporary:
            failures = validate(root, ())
            self.assertEqual(len(failures), 1)
            self.assertIn("unapproved raw_stream", failures[0])
            self.assertIn("hostResultBoundary", failures[0])

    def test_private_blas_lifecycle_cannot_hide_device_synchronization(self) -> None:
        """All library/backend variants reject projection-owned handle lifetime."""
        for library in ("cublas", "cublasLt", "hipblas", "hipblasLt"):
            for operation in ("Create", "Destroy"):
                with self.subTest(library=library, operation=operation):
                    temporary, root = self.make_repo(
                        f"void retireProjection() {{ {library}{operation}(handle); }}"
                    )
                    with temporary:
                        failures = validate(root, ())
                        self.assertEqual(len(failures), 1)
                        self.assertIn("unapproved blas_lifetime", failures[0])
                        self.assertIn("retireProjection", failures[0])

    def test_exact_reviewed_budget_passes(self) -> None:
        temporary, root = self.make_repo(
            """
            void hostResultBoundary(void *stream) {
                cudaStreamSynchronize(stream);
            }
            """
        )
        with temporary:
            self.assertEqual(validate(root, (self.allowance(),)), [])

    def test_inline_lambda_is_attributed_to_enclosing_caller(self) -> None:
        temporary, root = self.make_repo(
            """
            void hostResultBoundary(void *stream) {
                worker.submitAsync([stream]() {
                    cudaStreamSynchronize(stream);
                });
            }
            """
        )
        with temporary:
            source_path = root / "src" / "v2" / "example.cpp"
            callsites = scan_file(root, source_path)
            self.assertEqual(len(callsites), 1)
            self.assertEqual(callsites[0].key.caller, "hostResultBoundary")
            self.assertEqual(validate(root, (self.allowance(),)), [])

    def test_additional_wait_changes_budget_and_fails(self) -> None:
        temporary, root = self.make_repo(
            """
            void hostResultBoundary(void *stream) {
                cudaStreamSynchronize(stream);
                cudaStreamSynchronize(stream);
            }
            """
        )
        with temporary:
            failures = validate(root, (self.allowance(),))
            self.assertEqual(len(failures), 1)
            self.assertIn("expected 1, found 2", failures[0])

    def test_backend_host_event_wait_cannot_bypass_fast_prefilter(self) -> None:
        temporary, root = self.make_repo(
            """
            bool hostResultBoundary(IBackend *backend, void *event) {
                return backend->waitForEvent(event, 0);
            }
            """
        )
        with temporary:
            failures = validate(root, ())
            self.assertEqual(len(failures), 1)
            self.assertIn("unapproved backend_event", failures[0])
            self.assertIn("hostResultBoundary", failures[0])

    def test_pipeline_terminal_allowance_cannot_escape_into_dispatch(self) -> None:
        """The production pipeline budget names one terminal site, not its class."""
        allowance, = (
            entry for entry in ALLOWANCES
            if entry.path.endswith("/PipelineDeviceGeneration.cpp")
            and entry.caller == "PipelineDeviceGeneration::finish"
        )
        self.assertEqual(allowance.caller, "PipelineDeviceGeneration::finish")
        self.assertEqual(allowance.kind, "backend_event")
        self.assertEqual(allowance.category, "host_result")
        self.assertEqual(allowance.count, 1)
        fixture_allowance = dataclasses.replace(allowance, path="src/v2/example.cpp")
        for caller, count, accepted in (
            ("finish", 1, True), ("finish", 2, False),
            ("launch", 1, False), ("prefill", 1, False),
        ):
            with self.subTest(caller=caller, count=count):
                temporary, root = self.make_repo(
                    f"void PipelineDeviceGeneration::{caller}() {{"
                    + "backend->waitForEvent(event, 0);" * count + "}"
                )
                with temporary:
                    self.assertEqual(not validate(root, (fixture_allowance,)), accepted)

    def test_pipeline_forward_only_terminal_and_teardown_budgets_are_exact(self) -> None:
        """Only the two named completion owners gain one exact event-wait site."""
        path = "src/v2/execution/local_execution/orchestrators/PipelineDeviceGeneration.cpp"
        allowances = {entry.caller: entry for entry in ALLOWANCES if entry.path == path}
        self.assertEqual(set(allowances), {
            "PipelineDeviceGeneration::finish", "PipelineDeviceGeneration::finishForwardOnly",
            "PipelineDeviceGeneration::retireForwardOnlyBeforeStageRelease",
        })
        for name, category in (("finishForwardOnly", "host_result"),
                               ("retireForwardOnlyBeforeStageRelease", "lifecycle")):
            allowance = allowances["PipelineDeviceGeneration::" + name]
            self.assertEqual((allowance.kind, allowance.count, allowance.category),
                             ("backend_event", 1, category))
            fixture = dataclasses.replace(allowance, path="src/v2/example.cpp")
            for caller, count, accepted in (
                (name, 1, True), (name, 2, False),
                ("advanceOrdinaryCondition", 1, False), ("launch", 1, False),
                ("prepareForwardOnlyFamily", 1, False), ("prefill", 1, False),
                ("~PipelineDeviceGeneration", 1, False), ("~RankOrchestrator", 1, False),
            ):
                with self.subTest(boundary=name, caller=caller, count=count):
                    temporary, root = self.make_repo(
                        f"void PipelineDeviceGeneration::{caller}() {{"
                        + "backend->waitForEvent(terminal, ordinal);" * count + "}"
                    )
                    with temporary:
                        self.assertEqual(not validate(root, (fixture,)), accepted)

    def test_pipeline_forward_only_event_budget_never_authorizes_device_or_stream_drain(self) -> None:
        """A terminal event allowance cannot disguise a stronger blocking primitive."""
        for caller in ("finishForwardOnly", "retireForwardOnlyBeforeStageRelease"):
            allowance, = (entry for entry in ALLOWANCES
                          if entry.caller == "PipelineDeviceGeneration::" + caller)
            fixture = dataclasses.replace(allowance, path="src/v2/example.cpp")
            for primitive in ("hipStreamSynchronize(stream);", "cudaDeviceSynchronize();",
                              "backend->synchronize(ordinal);"):
                with self.subTest(caller=caller, primitive=primitive):
                    temporary, root = self.make_repo(
                        f"void PipelineDeviceGeneration::{caller}() {{"
                        + "backend->waitForEvent(terminal, ordinal);" + primitive + "}"
                    )
                    with temporary:
                        self.assertTrue(validate(root, (fixture,)))

    def test_pipeline_forward_only_raii_lambda_belongs_to_terminal_owner(self) -> None:
        """A lexical error-path fence stays attributed to finishForwardOnly only."""
        allowance, = (entry for entry in ALLOWANCES
                      if entry.caller == "PipelineDeviceGeneration::finishForwardOnly")
        fixture = dataclasses.replace(allowance, path="src/v2/example.cpp")
        for caller, accepted in (("finishForwardOnly", True), ("advanceOrdinaryCondition", False)):
            with self.subTest(caller=caller):
                temporary, root = self.make_repo(
                    f"bool PipelineDeviceGeneration::{caller}() {{"
                    "const auto retire = [&]() noexcept { backend->waitForEvent(terminal, ordinal); };"
                    "retire(); return true; }"
                )
                with temporary:
                    self.assertEqual(not validate(root, (fixture,)), accepted)

    def test_synchronous_backend_wrapper_is_charged_to_its_caller(self) -> None:
        temporary, root = self.make_repo(
            """
            bool hiddenRoundTrip(IBackend *backend, void *host, void *device) {
                return backend->deviceToHost(host, device, 64, 0);
            }
            """
        )
        with temporary:
            failures = validate(root, ())
            self.assertEqual(len(failures), 1)
            self.assertIn("unapproved backend_sync_copy", failures[0])
            self.assertIn("hiddenRoundTrip", failures[0])

    def test_hosted_reset_terminal_wait_cannot_escape_into_inference(self) -> None:
        """The reset allowance covers one terminal event, never a token or stage wait."""
        allowance, = (
            entry for entry in ALLOWANCES
            if entry.caller == "DeviceGraphOrchestrator::retireRetainedStorageForRequestReset"
        )
        self.assertEqual(allowance.category, "lifecycle")
        self.assertEqual(allowance.kind, "backend_event")
        self.assertEqual(allowance.count, 1)
        fixture_allowance = dataclasses.replace(allowance, path="src/v2/example.cpp")
        for caller, count, accepted in (
            ("retireRetainedStorageForRequestReset", 1, True),
            ("retireRetainedStorageForRequestReset", 2, False),
            ("beginRetainedStorageAdmission", 1, False),
            ("launchOrdinaryGenerationForward", 1, False),
            ("recordTerminalClose", 1, False),
        ):
            with self.subTest(caller=caller, count=count):
                temporary, root = self.make_repo(
                    f"void DeviceGraphOrchestrator::{caller}() {{"
                    + "backend->waitForEvent(event, ordinal);" * count + "}"
                )
                with temporary:
                    self.assertEqual(not validate(root, (fixture_allowance,)), accepted)

    def test_immutable_submission_owner_wait_cannot_escape_into_serving(self) -> None:
        """One failed-submission/last-owner event never permits normal serving waits."""
        allowance, = (
            entry for entry in ALLOWANCES
            if entry.caller == "retireImmutableLookupSubmissionOwner"
        )
        self.assertEqual(allowance.category, "lifecycle")
        self.assertEqual(allowance.kind, "backend_event")
        self.assertEqual(allowance.count, 1)
        fixture_allowance = dataclasses.replace(allowance, path="src/v2/example.cpp")
        for caller, count, accepted in (
            ("retireImmutableLookupSubmissionOwner", 1, True),
            ("retireImmutableLookupSubmissionOwner", 2, False),
            ("beginAdmission", 1, False),
            ("gather", 1, False),
            ("recordTerminalClose", 1, False),
            ("prepareGraphLaunch", 1, False),
        ):
            with self.subTest(caller=caller, count=count):
                temporary, root = self.make_repo(
                    f"void {caller}() {{"
                    + "backend.waitForEvent(event, ordinal);" * count + "}"
                )
                with temporary:
                    self.assertEqual(not validate(root, (fixture_allowance,)), accepted)

    def test_selected_snapshot_wait_is_only_a_diagnostic_boundary(self) -> None:
        """One diagnostic wait site cannot authorize waits in execution/capture."""
        allowance, = (
            entry for entry in ALLOWANCES
            if entry.caller == "DeviceGraphExecutor::acquireGraphSnapshotArenaOnHost"
        )
        self.assertEqual(allowance.category, "diagnostic")
        self.assertEqual(allowance.kind, "backend_event")
        self.assertEqual(allowance.count, 1)
        fixture_allowance = dataclasses.replace(allowance, path="src/v2/example.cpp")
        for caller, count, accepted in (
            ("acquireGraphSnapshotArenaOnHost", 1, True),
            ("acquireGraphSnapshotArenaOnHost", 2, False),
            ("runStages", 1, False),
            ("captureGraphSnapshotOutputs", 1, False),
        ):
            with self.subTest(caller=caller, count=count):
                temporary, root = self.make_repo(
                    f"void DeviceGraphExecutor::{caller}() {{"
                    + "backend->waitForEvent(event, ordinal);" * count + "}"
                )
                with temporary:
                    self.assertEqual(not validate(root, (fixture_allowance,)), accepted)

    def test_asynchronous_backend_wrapper_is_not_rejected(self) -> None:
        temporary, root = self.make_repo(
            """
            bool deviceOwnedCopy(
                IBackend *backend,
                void *host,
                void *device,
                void *stream) {
                return backend->deviceToHostOnStream(
                    host, device, 64, 0, stream);
            }
            """
        )
        with temporary:
            source_path = root / "src" / "v2" / "example.cpp"
            self.assertEqual(scan_file(root, source_path), [])
            self.assertEqual(validate(root, ()), [])

    def test_captured_compute_isolation_requires_async_exact_stream_copies(self) -> None:
        """Restore/cold snapshots retain bytes; synchronous D2D cannot hide in capture."""
        root = pathlib.Path(__file__).resolve().parents[4]
        relative = "src/v2/execution/diagnostics/PreparedGemmIsolationAdapter.h"
        source_path = root / relative
        source = strip_comments_and_literals(source_path.read_text(encoding="utf-8"))
        intervals = function_intervals(source)
        self.assertFalse(any(entry.path == relative for entry in ALLOWANCES))
        self.assertEqual(scan_file(root, source_path), [])
        for caller in ("recordRestore", "freeze"):
            bodies = [source[first + 1:last] for first, last, name in intervals if name == caller]
            self.assertEqual(len(bodies), 1, caller)
            compact = "".join(bodies[0].split())
            self.assertEqual(compact.count("backend.deviceCopyAsync("), 1, caller)
            self.assertIn("copy.bytes,device_.ordinal,stream)", compact)
            self.assertLess(compact.index("requireReady(stream,"), compact.index("backend.deviceCopyAsync("))
            temporary, fixture_root = self.make_repo(
                f"void {caller}(IBackend &backend, void *stream) {{"
                "backend.deviceToDevice(dst, src, bytes, ordinal, stream); }"
            )
            with temporary:
                failures = validate(fixture_root, ())
                self.assertEqual(len(failures), 1)
                self.assertIn("unapproved backend_sync_copy", failures[0])
                self.assertIn(caller, failures[0])
        ready, = [source[first + 1:last] for first, last, name in intervals if name == "requireReady"]
        self.assertIn("word<=2", "".join(ready.split()))

    def test_removed_wait_leaves_stale_allowance_and_fails(self) -> None:
        temporary, root = self.make_repo(
            """
            void hostResultBoundary(void *stream) {
                (void)stream;
            }
            """
        )
        with temporary:
            failures = validate(root, (self.allowance(),))
            self.assertEqual(len(failures), 1)
            self.assertIn("stale allowance", failures[0])


if __name__ == "__main__":
    unittest.main()
