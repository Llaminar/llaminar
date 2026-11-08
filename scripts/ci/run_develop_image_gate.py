#!/usr/bin/env python3
"""Build and optionally validate/publish both develop runtime images.

This intentionally small gate is distinct from the full production
certification pipeline. The develop push selects build-only publication of
AVX512 and AVX2 full-backend runtimes without compiling or running test images.
The separate PR gate runs the complete CMake-owned Unit inventory and the
canonical host preflight lane inside its installed test runner, without GPUs.
Neither policy discovers models, stages a
corpus, launches a server, runs mathematical parity, E2E, remote MPI, or a
benchmark, and it never calls an image a certified release artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import docker_paths
from production_artifacts import CPUISA, digest, image_identity, write_json
from run_production_cross_host_e2e import CPU_ISA_FEATURES
from run_production_pipeline import (
    TestRunnerInventory, ImageRole, SHIPPING_ISAS, build, logged_container_command,
    require_image, run, source_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def require_host_cpu_isa(isa: str) -> None:
    """Reject incompatible PR test execution before spending time compiling.

    Reuse the shipped CPU feature contract instead of inferring support from
    the runner size or CPU model. Every visible CPU must support the requested
    image; a partial feature set cannot authorize unrestricted test workers.
    Build-only publication deliberately does not require native ISA support.
    """
    required = CPU_ISA_FEATURES[CPUISA(isa)]
    rows = re.findall(r"^flags[ \t]*:[ \t]*(.*)$", Path("/proc/cpuinfo").read_text(), re.M)
    if not rows:
        raise ValueError(f"{isa} PR tests require explicit x86 CPU flags from the runner")
    common = set.intersection(*(set(row.split()) for row in rows))
    missing = sorted(required - common)
    if missing:
        raise ValueError(f"{isa} PR tests cannot run on this runner; missing CPU flags: {', '.join(missing)}")


def runtime_tag(base: str, isa: str) -> str:
    """Return the public tag for one explicitly named shipping ISA."""
    if isa not in SHIPPING_ISAS:
        raise ValueError("develop publication requires an explicit shipping ISA")
    if not isinstance(base, str) or not base or any(character.isspace() for character in base):
        raise ValueError("develop image tag must be a nonempty Docker reference without whitespace")
    return base + ("-avx2" if isa == "AVX2" else "")


def revision_tag(base: str, isa: str, revision: str) -> str:
    """Name one source-pinned sibling without changing the public ISA grammar."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("develop publication requires a full Git revision")
    return runtime_tag(f"{base}-{revision}", isa)


def remote_image_id(tag: str) -> str | None:
    """Read a registry tag's config digest before writing a source-pinned alias.

    A missing manifest is the only admissible cache miss; auth or transport
    errors must never masquerade as permission to overwrite a ref tag.
    """
    probe = subprocess.run(["docker", "buildx", "imagetools", "inspect", "--raw", tag],
                           capture_output=True, text=True, check=False)
    if probe.returncode != 0:
        if any(marker in probe.stderr.lower() for marker in ("not found", "manifest unknown")):
            return None
        raise subprocess.CalledProcessError(probe.returncode, probe.args,
                                            output=probe.stdout, stderr=probe.stderr)
    manifest = json.loads(probe.stdout)
    identity = manifest.get("config", {}).get("digest")
    if not isinstance(identity, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
        raise ValueError("source-pinned registry tag is not a single-platform image")
    return identity


def run_pr_tests(images: dict, directory: Path) -> dict:
    """Run complete Unit and host-preflight inventories without exposing accelerators.

    The test runner already contains its immutable model-free executable inventory
    and installed receipt.  Bind only the evidence directory; attaching models
    or a corpus would make this deliberately model-free develop gate depend on
    a second, undocumented input surface.
    """
    image = images[ImageRole.TEST_RUNNER.value]["id"]
    name = "llaminar-develop-pr-tests-" + uuid.uuid4().hex
    command = ["python3", "scripts/ci/run_develop_pr_tests.py",
               "--build-dir", "build_v2_integration",
               "--installed-build-receipt", "/src/installed-tests.json",
               "--output", "/ci-results/pr-tests"]
    launch = ["docker", "run", "--rm", "--name", name,
              "--user", f"{os.getuid()}:{os.getgid()}",
              *(part for group in os.getgroups() for part in ("--group-add", str(group))),
              *docker_paths.mounts([(directory, "/ci-results", False)]),
              "-w", "/src", image,
              *logged_container_command(command, "/ci-results/pr-tests.process.log")]
    try:
        run(launch, directory / "pr-tests.log")
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report_path = directory / "pr-tests" / "pr-tests.json"
    report = json.loads(report_path.read_text())
    if (report.get("scope") != "unit-and-host" or report.get("complete") is not True
            or report.get("return_code") != 0
            or set(report.get("lanes", {})) != {"unit", "host"}):
        raise ValueError("PR Unit/host receipt is incomplete or unsuccessful")
    for lane in report["lanes"].values():
        tests = lane.get("tests", [])
        if (lane.get("complete") is not True or lane.get("return_code") != 0
                or not tests or lane.get("test_count") != len(tests)
                or len(set(tests)) != len(tests)):
            raise ValueError("PR Unit/host lane is incomplete or unsuccessful")
    return report


def publish_runtime(images: dict, isa: str, base: str, revision: str,
                    directory: Path) -> dict:
    """Publish an authenticated source-ref alias before moving its branch alias."""
    runtime = images[ImageRole.RUNTIME.value]
    tag = runtime_tag(base, isa)
    pinned = revision_tag(base, isa, revision)
    existing = remote_image_id(pinned)
    if existing is not None and existing != runtime["id"]:
        raise ValueError("source-pinned develop tag already names different image bytes")
    for alias in (pinned, tag):
        run(["docker", "tag", runtime["id"], alias], directory / "publish.log")
        if image_identity(alias)["id"] != runtime["id"]:
            raise ValueError("develop publication tag no longer names its admitted runtime image")
        if alias != pinned or existing is None:
            run(["docker", "push", alias], directory / "publish.log")
    return {"tag": tag, "ref_tag": pinned, "image": runtime["id"]}


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse only the small explicit surface supported by the develop gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new ignored directory for build or Unit/host evidence")
    parser.add_argument("--image", help="AVX512 develop runtime tag; AVX2 appends -avx2")
    parser.add_argument("--build-only", action="store_true",
                        help="build runtime images without compiling or executing Unit/preflight tests")
    parser.add_argument("--publish", action="store_true",
                        help="push both runtime images after both selected ISA transactions succeed")
    parser.add_argument("--cpu-isa", choices=SHIPPING_ISAS, action="append", dest="cpu_isas",
                        help="explicit local diagnostic ISA subset; publication requires both")
    args = parser.parse_args(argv)
    requested = args.cpu_isas or list(SHIPPING_ISAS)
    if len(set(requested)) != len(requested):
        parser.error("duplicate CPU ISA selection")
    args.cpu_isas = [isa for isa in SHIPPING_ISAS if isa in requested]
    if args.publish and (not args.image or set(args.cpu_isas) != set(SHIPPING_ISAS)):
        parser.error("--publish requires --image and both AVX512 and AVX2 gates")
    if args.publish and not args.build_only:
        parser.error("--publish requires --build-only; PR Unit/host validation never publishes")
    args.output = args.output.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    """Finish both admitted ISA transactions before publishing either runtime."""
    args = parse_arguments(argv)
    if not args.build_only:
        for isa in args.cpu_isas:
            require_host_cpu_isa(isa)
    if args.output.exists():
        raise ValueError("develop image gate output already exists; preserve prior evidence and choose a new directory")
    args.output.mkdir(parents=True)
    source = source_identity()
    receipt = {"schema": 1, "source": source, "complete": False, "published": False,
               "variants": {}, "image": args.image,
               "validation": "build-only" if args.build_only else "unit-and-host"}
    write_json(args.output / "develop-image-gate.json", receipt)
    for isa in args.cpu_isas:
        if source_identity() != source:
            raise ValueError("source changed after develop image gate admission")
        directory = args.output / isa.lower()
        directory.mkdir()
        build_args = argparse.Namespace(cpu_isa=isa)
        if args.build_only:
            images = build(build_args, source, directory,
                           roles=(ImageRole.RUNTIME,),
                           test_inventory=TestRunnerInventory.UNIT_ONLY)
            require_image(images[ImageRole.RUNTIME.value], source, isa, ImageRole.RUNTIME)
            variant = {"runtime_image": images[ImageRole.RUNTIME.value]["id"]}
        else:
            images = build(build_args, source, directory,
                           roles=(ImageRole.TEST_RUNNER,),
                           test_inventory=TestRunnerInventory.MODEL_FREE)
            require_image(images[ImageRole.TEST_RUNNER.value], source, isa, ImageRole.TEST_RUNNER,
                          test_inventory=TestRunnerInventory.MODEL_FREE)
            pr_tests = run_pr_tests(images, directory)
            variant = {
                "test_runner_image": images[ImageRole.TEST_RUNNER.value]["id"],
                "test_runner_inventory": TestRunnerInventory.MODEL_FREE.value,
                "pr_tests_digest": digest(pr_tests),
                "unit_test_count": pr_tests["lanes"]["unit"]["test_count"],
                "host_test_count": pr_tests["lanes"]["host"]["test_count"],
            }
        receipt["variants"][isa] = variant
        write_json(args.output / "develop-image-gate.json", receipt)
    if source_identity() != source:
        raise ValueError("source changed before develop image gate completion")
    receipt["complete"] = True
    if args.publish:
        # Both builds and any explicitly selected tests must finish before
        # moving a public tag. A network failure can still interrupt pushes.
        for isa in args.cpu_isas:
            images = {ImageRole.RUNTIME.value: {"id": receipt["variants"][isa]["runtime_image"]}}
            receipt["variants"][isa]["publication"] = publish_runtime(
                images, isa, args.image, source["revision"], args.output / isa.lower())
            write_json(args.output / "develop-image-gate.json", receipt)
        receipt["published"] = True
    write_json(args.output / "develop-image-gate.json", receipt)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"[develop-image-gate] ERROR {error}", file=sys.stderr)
        raise SystemExit(1)
