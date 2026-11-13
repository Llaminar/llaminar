#!/usr/bin/env python3
## @file test_docker_toolchain_closure.py
# @brief Replay Docker toolchain helper materialization, retirement and RCCL target binding.
"""Exercise real staged helper files without compiling a second GPU SDK.

COPY inputs and explicit helper-tree retirement are replayed in a private
filesystem. At the RCCL consumer boundary the actual Python entrypoint must
start. The original early retirement reproduces the missing-file failure.
Architecture defaults execute the real shell prefix against the canonical pins.
"""
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
DOCKER = ROOT / "Dockerfile"


def instructions(text: str):
    """Join Docker continuations so COPY inputs and RUN lifetimes stay ordered."""
    pending = ""
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        continuation = raw.rstrip().endswith("\\")
        pending += raw.rstrip()[:-1] + " " if continuation else raw
        if not continuation:
            yield pending.strip()
            pending = ""
    if pending:
        raise ValueError("Incomplete Docker instruction")


class DockerToolchainClosureTests(unittest.TestCase):
    """The final-link producer owns its complete staged source until retirement."""

    def replay(self, *, original: bool = False):
        """Materialize actual COPY inputs and run the consumer at its real boundary."""
        text = DOCKER.read_text()
        if original:
            text = text.replace("WORKDIR /src", "RUN rm -rf /tmp/install-scripts\nWORKDIR /src", 1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for instruction in instructions(text):
                if instruction.startswith("COPY "):
                    words = shlex.split(instruction)
                    if any(word.startswith("--from") for word in words):
                        continue
                    destination = root / words[-1].lstrip("/")
                    for source in words[1:-1]:
                        source = ROOT / source
                        if source.is_dir():
                            shutil.copytree(source, destination, dirs_exist_ok=True)
                        else:
                            target = destination / source.name if words[-1].endswith("/") else destination
                            target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, target)
                elif instruction == "RUN rm -rf /tmp/install-scripts":
                    shutil.rmtree(root / "tmp/install-scripts", ignore_errors=True)
                elif instruction.startswith("RUN ") and "llvm-lto-linker/artifact.py configure" in instruction:
                    helper = root / "tmp/install-scripts/llvm-lto-linker/artifact.py"
                    return subprocess.run([sys.executable, str(helper), "--help"],
                                          capture_output=True, text=True)
            raise AssertionError("RCCL final-link consumer is missing from the Docker toolchain")

    def test_linker_consumer_retains_its_staged_files(self):
        """Run the real helper entrypoint after preceding Docker helper lifetimes."""
        result = self.replay()
        self.assertEqual(result.returncode, 0, result.stderr)
        original = self.replay(original=True)
        self.assertNotEqual(original.returncode, 0)
        self.assertIn("No such file or directory", original.stderr)

    def test_rccl_default_and_explicit_architectures_execute_with_pinned_scope(self):
        """Shell-local assignments must bind actual arguments, not log substitutions."""
        command = next(row for row in instructions(DOCKER.read_text())
                       if row.startswith("RUN ") and "llvm-lto-linker/artifact.py configure" in row)
        prefix = command[command.index("set -e;"):command.index("rccl_started_epoch=")]
        prefix = prefix.replace("/src/scripts/docker/rocm-release.env", str(ROOT / "scripts/docker/rocm-release.env"))
        for override in (None, "gfx1100"):
            environment = dict(os.environ)
            environment.pop("RCCL_GPU_TARGETS", None)
            if override is not None:
                environment["RCCL_GPU_TARGETS"] = override
            result = subprocess.run(["/bin/sh", "-ec", prefix +
                                     "printf '%s\\n%s\\n' \"$RCCL_GPU_TARGETS\" \"$LLAMINAR_ROCM_GPU_TARGETS\""],
                                    env=environment, capture_output=True, text=True, check=True)
            selected, canonical = result.stdout.splitlines()
            self.assertTrue(canonical)
            self.assertEqual(selected, override if override is not None else canonical)


if __name__ == "__main__":
    unittest.main()
