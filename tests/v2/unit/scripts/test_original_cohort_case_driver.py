#!/usr/bin/env python3
"""Exercise the original cohort launcher with small, device-free child processes."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


DRIVER = Path(__file__).resolve().parents[2] / "scripts/run_cohort_original_cases.py"


class OriginalCohortCaseDriverTests(unittest.TestCase):
    """Observe the actual process boundary and arguments on each launched child."""

    def setUp(self):
        """Create one stand-in executable and a fresh child invocation log."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.log = Path(temporary.name) / "invocations.jsonl"
        self.child = Path(temporary.name) / "cohort-fixture"
        self.child.write_text(f"#!{sys.executable}\n" + """\
import json
import os
import sys

arguments = sys.argv[1:]
record = dict(pid=os.getpid(), parent=os.getppid(), arguments=arguments,
              original=os.environ.get("COHORT_FIXTURE_ORIGINAL"),
              order=os.environ.get("COHORT_FIXTURE_ORDER"),
              fault=os.environ.get("COHORT_FIXTURE_FAULT"))
with open(os.environ["COHORT_FIXTURE_LOG"], "a") as log:
    log.write(json.dumps(record) + "\\n")
if arguments == ["--gtest_list_tests"]:
    print("Backends/Replay.\\n  First/CUDA  # GetParam() = CUDA\\n  Second/CUDA"
          "\\n  Excluded/CUDA\\n  First/ROCm\\nUnselected.\\n  Irrelevant"
          "\\nTyped/Replay/0.  # TypeParam = int\\n  Value")
else:
    sys.exit(int(os.environ.get("COHORT_FIXTURE_EXIT", "0")))
""")
        self.child.chmod(0o700)
        self.environment = {
            **os.environ,
            "COHORT_FIXTURE_LOG": str(self.log),
            "COHORT_FIXTURE_ORIGINAL": "original value",
            "COHORT_FIXTURE_ORDER": "inherited order",
            "COHORT_FIXTURE_FAULT": "inherited fault",
            "COHORT_FIXTURE_EXIT": "0",
        }

    def launch(self, *arguments):
        """Execute the real driver with the fixture as its ordinary binary argument."""
        return subprocess.run(
            [sys.executable, str(DRIVER), str(self.child), *arguments],
            capture_output=True, text=True, env=self.environment, check=False,
        )

    def invocations(self):
        """Read the facts recorded inside the actual child processes."""
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def test_selection_repetition_and_cartesian_grid_use_fresh_processes(self):
        """List once, then run each selected case and variant in a separate child."""
        passthrough = ["--pipeline-native-family=cuda", "--pipeline-native-degree=2",
                       "--fixture-option=literal value = retained"]
        result = self.launch(
            "--gtest_filter=Backends/Replay.*/*:Unselected.Absent-*.Excluded/*:*/ROCm",
            "--gtest_repeat=2", *passthrough,
            "--original-case-env=COHORT_FIXTURE_ORDER=forward,reverse",
            "--original-case-env=COHORT_FIXTURE_FAULT=first,tail",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        listing, *replays = self.invocations()
        self.assertEqual(listing["arguments"], ["--gtest_list_tests"])
        self.assertEqual((listing["order"], listing["fault"]), ("inherited order", "inherited fault"))

        cases = ["Backends/Replay.First/CUDA", "Backends/Replay.Second/CUDA"]
        expected = [
            (case, order, fault)
            for _, case, order, fault in itertools.product(
                range(2), cases, ("forward", "reverse"), ("first", "tail")
            )
        ]
        self.assertEqual(len(replays), len(expected))
        processes = [listing, *replays]
        self.assertEqual(len({row["pid"] for row in processes}), len(processes))
        self.assertEqual({row["parent"] for row in processes}, {listing["parent"]})
        self.assertEqual({row["original"] for row in processes}, {"original value"})
        for row, (case, order, fault) in zip(replays, expected, strict=True):
            self.assertEqual(row["arguments"], [
                *passthrough, "--gtest_death_test_style=threadsafe",
                "--gtest_repeat=1", f"--gtest_filter={case}",
            ])
            self.assertEqual((row["order"], row["fault"]), (order, fault))

    def test_without_grid_retains_inherited_environment(self):
        """An existing unqualified invocation still gets its one original child."""
        result = self.launch("--gtest_filter=Typed/Replay/0.Value")
        self.assertEqual(result.returncode, 0, result.stderr)
        _, replay = self.invocations()
        self.assertIn("--gtest_filter=Typed/Replay/0.Value", replay["arguments"])
        self.assertEqual((replay["original"], replay["order"], replay["fault"]),
                         ("original value", "inherited order", "inherited fault"))

    def test_invalid_grid_rejects_before_discovery_or_execution(self):
        """Reject duplicate names and missing or empty assignments before any child."""
        invalid = [
            ("COHORT_FIXTURE_ORDER=forward", "COHORT_FIXTURE_ORDER=reverse"),
            ("=forward",),
            ("COHORT_FIXTURE_ORDER",),
            ("COHORT_FIXTURE_ORDER=",),
            ("COHORT_FIXTURE_ORDER=forward,,reverse",),
        ]
        for declarations in invalid:
            with self.subTest(declarations=declarations):
                result = self.launch(*(f"--original-case-env={entry}" for entry in declarations))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("unique environment name and nonempty values", result.stderr)
                self.assertFalse(self.log.exists())

    def test_failed_variant_stops_the_grid_and_reports_child_status(self):
        """A failed child fails the launcher without submitting later variants."""
        result = self.launch(
            "--gtest_filter=Backends/Replay.First/CUDA:Backends/Replay.Second/CUDA",
            "--gtest_repeat=2", "--original-case-env=COHORT_FIXTURE_EXIT=37,0",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("returned non-zero exit status 37", result.stderr)
        listing, replay = self.invocations()
        self.assertEqual(listing["arguments"], ["--gtest_list_tests"])
        self.assertIn("--gtest_filter=Backends/Replay.First/CUDA", replay["arguments"])


if __name__ == "__main__":
    unittest.main()
