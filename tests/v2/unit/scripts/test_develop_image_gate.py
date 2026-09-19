#!/usr/bin/env python3
"""Device-free contract tests for the narrow develop image build/publish gate."""
from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))

import run_develop_image_gate as gate


def images(isa: str) -> dict:
    """Return just the immutable IDs needed by the outer develop transaction."""
    return {"test-runner": {"id": f"test-runner-{isa.lower()}"},
            "runtime": {"id": f"runtime-{isa.lower()}"}}


def prerequisite_report() -> dict:
    """Build one complete synthetic canonical prerequisite receipt."""
    return {"preflight_return_code": 0,
            "preflight_tests": ["V2_Unit_Contract", "V2_Integration_Preflight"],
            "preflight_test_count": 2}


class DevelopImageGateTests(unittest.TestCase):
    """Keep develop publishing smaller than, and incapable of replacing, certification."""

    def test_publish_requires_both_shipping_isas(self):
        """A partial local diagnosis can never overwrite a public develop tag."""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(SystemExit) as error:
                gate.parse_arguments(["--output", str(Path(temporary) / "out"), "--publish",
                                      "--image", "ghcr.io/llaminar/llaminar:develop",
                                      "--cpu-isa", "AVX2"])
        self.assertEqual(error.exception.code, 2)

    def test_builds_and_tests_both_images_before_any_publication(self):
        """The two required gates finish before either mutable registry tag moves."""
        source = {"revision": "revision", "tree": "tree", "dirty": False}
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"

            def build(args, admitted, directory, *, test_inventory):
                self.assertEqual(admitted, source)
                self.assertEqual(test_inventory, gate.TestRunnerInventory.MODEL_FREE)
                events.append(("build", args.cpu_isa))
                return images(args.cpu_isa)

            def prerequisites(built, directory):
                events.append(("prerequisites", built["runtime"]["id"]))
                return prerequisite_report()

            def publish(built, isa, tag, directory):
                if isa == "AVX512":
                    self.assertEqual(events, [
                        ("build", "AVX512"), ("prerequisites", "runtime-avx512"),
                        ("build", "AVX2"), ("prerequisites", "runtime-avx2"),
                    ])
                events.append(("publish", isa))
                return {"tag": gate.runtime_tag(tag, isa), "image": built["runtime"]["id"]}

            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "device_lease", side_effect=lambda: nullcontext()), \
                 patch.object(gate, "build", side_effect=build), \
                 patch.object(gate, "require_image_pair"), \
                 patch.object(gate, "run_prerequisites", side_effect=prerequisites), \
                 patch.object(gate, "publish_runtime", side_effect=publish):
                self.assertEqual(gate.main(["--output", str(output), "--publish",
                                            "--image", "ghcr.io/llaminar/llaminar:develop"]), 0)

            self.assertEqual(events, [
                ("build", "AVX512"), ("prerequisites", "runtime-avx512"),
                ("build", "AVX2"), ("prerequisites", "runtime-avx2"),
                ("publish", "AVX512"), ("publish", "AVX2"),
            ])
            receipt = json.loads((output / "develop-image-gate.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["published"])
            self.assertEqual(receipt["variants"]["AVX512"]["test_runner_inventory"],
                             gate.TestRunnerInventory.MODEL_FREE.value)
            self.assertEqual(receipt["variants"]["AVX2"]["publication"]["tag"],
                             "ghcr.io/llaminar/llaminar:develop-avx2")

    def test_a_failed_preflight_prevents_every_registry_mutation(self):
        """No public tag moves until every image has passed its own installed gate."""
        source = {"revision": "revision", "tree": "tree", "dirty": False}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            with patch.object(gate, "source_identity", return_value=source), \
                 patch.object(gate, "device_lease", side_effect=lambda: nullcontext()), \
                 patch.object(gate, "build", side_effect=lambda args, *_, **__: images(args.cpu_isa)), \
                 patch.object(gate, "require_image_pair"), \
                 patch.object(gate, "run_prerequisites", side_effect=RuntimeError("preflight red")), \
                 patch.object(gate, "publish_runtime") as publish:
                with self.assertRaisesRegex(RuntimeError, "preflight red"):
                    gate.main(["--output", str(output), "--publish",
                               "--image", "ghcr.io/llaminar/llaminar:develop"])
            publish.assert_not_called()


if __name__ == "__main__":
    unittest.main()
