#!/usr/bin/env python3
"""Regression tests for the GPU blocking-synchronization source policy."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from check_gpu_blocking_sync_policy import (  # noqa: E402
    Allowance,
    scan_file,
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
