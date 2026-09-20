#!/usr/bin/env python3
"""Device-free regression for the production HTTP harness's ROCm VRAM reader.

The harness runs inside a tools container where ``amd-smi`` may be absent. Its
``rocm-smi`` fallback must parse real multi-colon output before GPU memory can
be used as E2E evidence.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
HARNESS = ROOT / "tests/v2/e2e/server/test_server_e2e.sh"


def memory_functions() -> str:
    """Execute the actual shell functions without starting the full harness."""
    source = HARNESS.read_text(encoding="utf-8")
    functions = []
    for name in ("read_amd_total_gpu_mb", "get_amd_total_gpu_mb",
                 "amd_memory_telemetry_available"):
        match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", source)
        if match is None:
            raise AssertionError(f"missing production telemetry function {name}")
        functions.append(match.group())
    return "\n".join(functions)


class ServerGpuMemoryTelemetryTests(unittest.TestCase):
    """Both vendor readers use the same validated measurement and availability."""

    def probe(self, tools: dict[str, str]) -> list[str]:
        """Run with isolated fake telemetry CLIs and real awk/bash parsing."""
        with tempfile.TemporaryDirectory() as temporary:
            bin_dir = Path(temporary)
            for name, body in tools.items():
                executable = bin_dir / name
                executable.write_text("#!/bin/sh\n" + body + "\n", encoding="utf-8")
                executable.chmod(0o755)
            environment = {**os.environ, "PATH": f"{bin_dir}:/usr/bin:/bin"}
            script = ("set -euo pipefail\n" + memory_functions() +
                      "\nif amd_memory_telemetry_available; then echo available; "
                      "else echo unavailable; fi\nget_amd_total_gpu_mb\n")
            output = subprocess.check_output(["/bin/bash", "-c", script],
                                             env=environment, text=True)
            return output.splitlines()

    def test_rocm_smi_uses_final_byte_field_after_two_colons(self):
        output = self.probe({
            "amd-smi": "exit 1",
            "rocm-smi": "printf 'GPU[0] : VRAM Total Used Memory (B): 104857600\\n"
                        "GPU[1] : VRAM Total Used Memory (B): 209715200\\n'",
        })
        self.assertEqual(output, ["available", "300"])

    def test_failed_amd_smi_falls_back_to_rocm_smi(self):
        output = self.probe({
            "amd-smi": "exit 1",
            "rocm-smi": "printf 'GPU[0] : VRAM Total Used Memory (B): 52428800\\n'",
        })
        self.assertEqual(output, ["available", "50"])

    def test_amd_smi_csv_and_unavailable_readers(self):
        output = self.probe({
            "amd-smi": "printf 'gpu,total_vram,used_vram,free_vram\\n"
                       "0,32752,8192,24560\\n1,32752,1024,31728\\n'",
        })
        self.assertEqual(output, ["available", "9216"])
        self.assertEqual(self.probe({"amd-smi": "exit 1", "rocm-smi": "exit 1"}),
                         ["unavailable", "0"])


if __name__ == "__main__":
    unittest.main()
