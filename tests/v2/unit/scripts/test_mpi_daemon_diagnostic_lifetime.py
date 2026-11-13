## @file test_mpi_daemon_diagnostic_lifetime.py
#  @brief Prove diagnostic directory lifetime through the real daemon-test entrypoint.
"""No MPI or device runtime is started; the real entrypoint owns scratch policy."""
from __future__ import annotations

import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SOURCE = Path(__file__).resolve().parents[2] / "integration/scripts/test_mpi_native_daemon.py"


class DaemonDiagnosticLifetime(unittest.TestCase):
    """Keep successful ordinary runs bounded while preserving authored evidence."""

    def exercise(self, *, explicit: bool, failed: bool) -> None:
        """Execute argument parsing and final cleanup with a controlled native outcome."""
        spec = importlib.util.spec_from_file_location("daemon_diagnostic_subject", SOURCE)
        subject = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(subject)
        with tempfile.TemporaryDirectory(prefix="mpi-daemon-lifetime-test-") as temporary:
            workspace = Path(temporary)
            helper = workspace / "bin/llaminar_mpi_session"
            helper.parent.mkdir()
            helper.write_text("fixture helper\n")
            component = helper.parent / "llaminar_mpi_components/mca_schizo_llaminar.so"
            component.parent.mkdir()
            component.write_text("fixture component\n")
            neighbor = workspace / "live-neighbor"
            neighbor.mkdir()
            (neighbor / "keep").write_text("unrelated diagnostic owner\n")
            roots = []

            class NativeOutcome:
                """Supply only the result of native execution, leaving cleanup real."""

                def __init__(self, args, root):
                    self.root = root
                    self.results = []
                    roots.append(root)

                def run(self, mode):
                    (self.root / "output.log").write_text("native outcome fixture\n")
                    if failed:
                        raise RuntimeError("recorded native failure")
                    self.results.append({"case": mode, "passed": True, "final_returncode": 0})

            original_mkdtemp = tempfile.mkdtemp

            def bounded_scratch(*args, **kwargs):
                """Place the entrypoint's real temporary directory under this test owner."""
                return original_mkdtemp(*args, **{**kwargs, "dir": workspace})

            arguments = [str(SOURCE), "--helper", str(helper), "--probe", str(helper),
                         "--module", str(component), "--mpirun", str(helper), "--case", "success"]
            if explicit:
                arguments += ["--evidence", str(workspace / "authored-evidence")]
            output = io.StringIO()
            with mock.patch.object(subject, "DaemonReplay", NativeOutcome), \
                    mock.patch.object(subject.tempfile, "mkdtemp", bounded_scratch), \
                    mock.patch.object(subject.sys, "argv", arguments), \
                    contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                if failed:
                    with self.assertRaisesRegex(RuntimeError, "recorded native failure"):
                        subject.main()
                else:
                    subject.main()
            self.assertEqual(len(roots), 1)
            self.assertEqual(roots[0].exists(), explicit or failed)
            self.assertEqual((neighbor / "keep").read_text(), "unrelated diagnostic owner\n")
            if explicit or failed:
                self.assertIn(str(roots[0]), output.getvalue())

    def test_successful_default_retires_diagnostics(self):
        """A successful ordinary CTest run must not leave a scratch tree."""
        self.exercise(explicit=False, failed=False)

    def test_successful_explicit_evidence_is_retained(self):
        """An explicitly requested evidence artifact survives successful execution."""
        self.exercise(explicit=True, failed=False)

    def test_default_failure_preserves_unqualified_retirement(self):
        """A failing native transaction cannot authorize recursive backing deletion."""
        self.exercise(explicit=False, failed=True)

    def test_explicit_failure_preserves_authored_evidence(self):
        """Explicit failure evidence belongs to its caller until reviewed."""
        self.exercise(explicit=True, failed=True)


if __name__ == "__main__":
    unittest.main()
