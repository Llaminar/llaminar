#!/usr/bin/env python3
## @file test_docker_toolchain_closure.py
# @brief Replay Docker toolchain lifetimes and RCCL architecture/header binding.
"""Exercise real staged helper files without compiling a second GPU SDK.

COPY inputs and explicit helper-tree retirement are replayed in a private
filesystem. At the RCCL consumer boundary the actual Python entrypoint must
start. The original early retirement reproduces the missing-file failure.
Architecture defaults execute the real shell prefix against the canonical pins.
Both application builds resolve the generated RCCL header through their actual
arguments (Release in the Dockerfile, Integration in the canonical operation
script run by the builder) and the production CMake admission check.
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
INTEGRATION_OPERATION = ROOT / "scripts/ci/arm_native_integration_operation.sh"
INTEGRATION_OPERATION_RUN = (
    "RUN --mount=type=cache,id=llaminar-ccache,target=/root/.ccache,sharing=locked "
    "python3 scripts/ci/arm_native_build_evidence.py capture -- "
    "sh scripts/ci/arm_native_integration_operation.sh")


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
        prefix = prefix.replace("/tmp/install-scripts/rocm-image-targets.sh",
                                str(ROOT / "scripts/docker/rocm-image-targets.sh"))
        for override in (None, "gfx1100", "gfx906,gfx1100"):
            environment = dict(os.environ)
            environment.pop("ROCM_RUNTIME_GPU_TARGETS", None)
            if override is not None:
                environment["ROCM_RUNTIME_GPU_TARGETS"] = override
            result = subprocess.run(["/bin/sh", "-ec", prefix +
                                     "printf '%s\\n%s\\n' \"$RCCL_GPU_TARGETS\" \"$LLAMINAR_ROCM_GPU_TARGETS\""],
                                    env=environment, capture_output=True, text=True, check=True)
            selected, canonical = result.stdout.splitlines()
            self.assertTrue(canonical)
            self.assertEqual(selected, override.replace(",", ";") if override is not None else canonical)

    def test_rccl_architecture_list_rejects_non_targets_and_duplicates(self):
        helper = ROOT / "scripts/docker/rocm-image-targets.sh"
        for requested in ("gfx906;native", "gfx906,gfx906", "gfx906\ngfx1100"):
            with self.subTest(requested=requested):
                result = subprocess.run(["bash", str(helper), requested], capture_output=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")

    def test_linker_proves_shipping_targets_and_extra_requested_architectures(self):
        """Exercise the actual installer arguments before any expensive compiler work."""
        directory = ROOT / "scripts/docker"
        shipping = subprocess.check_output(["bash", str(directory / "rocm-image-targets.sh")], text=True).strip()
        subset = shipping.split(";")[0]
        extra = next(arch for arch in ("gfx942", "gfx950", "gfx1201") if arch not in shipping.split(";"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("install-rocm-lto-linker.sh", "rocm-release.env", "rocm-image-targets.sh"):
                shutil.copy2(directory / name, root / name)
            (root / "llvm-lto-linker").mkdir()
            (root / "llvm-lto-linker/build.py").write_text(
                'import sys\nprint(sys.argv[sys.argv.index("--architectures") + 1])\n')
            for chosen, expected in ((subset, shipping), (f"{extra},{subset}", f"{shipping};{extra}")):
                with self.subTest(architectures=chosen):
                    actual = subprocess.check_output(["bash", str(root / "install-rocm-lto-linker.sh")],
                        env={**os.environ, "ROCM_IMAGE_GPU_TARGETS": chosen}, text=True)
                    self.assertEqual(actual.strip(), expected)

    def test_application_builds_admit_the_generated_rccl_header(self):
        """Replay both Docker bindings; source templates must fail ABI admission."""
        cmake = (ROOT / "src/v2/CMakeLists.txt").read_text()
        start = cmake.index("        find_file(_rccl_abi_header")
        end = cmake.index("        target_sources(rocm_backend", start)
        admission = cmake[start:end]
        docker = list(instructions(DOCKER.read_text()))
        # Release is configured directly in the Dockerfile.  Integration is the
        # canonical operation script, which binds only through the builder RUN.
        self.assertEqual(sum(" ".join(row.split()) == INTEGRATION_OPERATION_RUN for row in docker), 1,
                         "The builder must run the canonical Integration operation exactly once")
        operation = [row for row in instructions(INTEGRATION_OPERATION.read_text())
                     if "RCCL_CMAKE_ARGS=" in row]
        release = [row for row in docker if row.startswith("RUN ") and "RCCL_CMAKE_ARGS=" in row]
        self.assertEqual(["build_v2_integration" in row for row in operation], [True])
        self.assertEqual(["build_v2_release" in row and "build_v2_integration" not in row
                          for row in release], [True])
        builds = operation + release
        self.assertEqual(len(builds), 2, "Integration and Release must both be covered")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            header = root / "src/external/rccl/build/include/rccl/rccl.h"
            header.parent.mkdir(parents=True)
            header.write_text("/* generated public RCCL API */\n")
            source = root / "src/external/rccl/src/include"
            source.mkdir(parents=True)
            (source / "rccl.h.in").write_text("/* unconfigured source template */\n")
            probe = root / "admit.cmake"
            probe.write_text(admission + '\nmessage(STATUS "admitted=${_rccl_abi_header}")\n')
            for build in builds:
                start = build.index('RCCL_CMAKE_ARGS="')
                end = build.index("fi;", start) + len("fi;")
                binding = build[start:end]
                for original in (False, True):
                    with self.subTest(build="Integration" if "build_v2_integration" in build else "Release",
                                      original=original):
                        command = binding
                        if original:
                            command = command.replace("/rccl/build/include", "/rccl/src/include")
                        command = command.replace("/src/", str(root / "src") + "/")
                        environment = dict(os.environ, LLAMINAR_ENABLE_ROCM="ON")
                        arguments = subprocess.run(
                            ["/bin/sh", "-ec", command + '\nprintf \'%s\' "$RCCL_CMAKE_ARGS"'],
                            env=environment, capture_output=True, text=True, check=True)
                        result = subprocess.run(["cmake", *shlex.split(arguments.stdout), "-P", str(probe)],
                                                capture_output=True, text=True)
                        if original:
                            self.assertNotEqual(result.returncode, 0)
                            self.assertIn("no selected rccl.h", result.stderr)
                        else:
                            self.assertEqual(result.returncode, 0, result.stderr)
                            self.assertIn(f"admitted={header}", result.stdout)


if __name__ == "__main__":
    unittest.main()
