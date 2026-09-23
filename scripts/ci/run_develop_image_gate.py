#!/usr/bin/env python3
"""Build, test and optionally publish the two minimal develop runtime images.

This intentionally small gate is distinct from the full production
certification pipeline.  It owns exactly the develop-branch promise: build the
AVX512 and AVX2 full-backend image pairs, run the complete CMake-owned Unit and
ProductionTestPreflight gates inside each installed test runner, then publish the
two already-tested runtime siblings.  It never discovers models, stages a
corpus, launches a server, runs mathematical parity, E2E, remote MPI, or a
benchmark, and it never calls an image a certified release artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import docker_paths
from production_artifacts import digest, image_identity, validate_prerequisites, write_json
from run_production_pipeline import (
    TestRunnerInventory, ImageRole, SHIPPING_ISAS, build, device_lease, logged_container_command,
    require_image_pair, run, source_identity,
)

ROOT = Path(__file__).resolve().parents[2]


def runtime_tag(base: str, isa: str) -> str:
    """Return the public tag for one explicitly named shipping ISA."""
    if isa not in SHIPPING_ISAS:
        raise ValueError("develop publication requires an explicit shipping ISA")
    if not isinstance(base, str) or not base or any(character.isspace() for character in base):
        raise ValueError("develop image tag must be a nonempty Docker reference without whitespace")
    return base + ("-avx2" if isa == "AVX2" else "")


def run_prerequisites(images: dict, directory: Path) -> dict:
    """Run the single canonical model-free transaction inside one test runner.

    The test runner already contains its immutable Integration executable inventory
    and installed receipt.  Bind only the evidence directory; attaching models
    or a corpus would make this deliberately model-free develop gate depend on
    a second, undocumented input surface.
    """
    image = images[ImageRole.TEST_RUNNER.value]["id"]
    name = "llaminar-develop-prerequisites-" + uuid.uuid4().hex
    command = ["python3", "scripts/ci/run_production_prerequisites.py",
               "--build-dir", "build_v2_integration",
               "--installed-build-receipt", "/src/installed-tests.json",
               "--output", "/ci-results/prerequisites"]
    launch = ["docker", "run", "--rm", "--name", name,
              *docker_paths.device_args(image, "CPU+CUDA+ROCm", user=f"{os.getuid()}:{os.getgid()}"),
              *(part for group in os.getgroups() for part in ("--group-add", str(group))),
              *docker_paths.mounts([(directory, "/ci-results", False)]),
              "-w", "/src", image,
              *logged_container_command(command, "/ci-results/prerequisites.process.log")]
    try:
        run(launch, directory / "prerequisites.log")
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report_path = directory / "prerequisites" / "prerequisites.json"
    report = json.loads(report_path.read_text())
    validate_prerequisites(report)
    return report


def publish_runtime(images: dict, isa: str, base: str, directory: Path) -> dict:
    """Publish one tested runtime by immutable local ID, never a mutable build tag."""
    runtime = images[ImageRole.RUNTIME.value]
    tag = runtime_tag(base, isa)
    run(["docker", "tag", runtime["id"], tag], directory / "publish.log")
    if image_identity(tag)["id"] != runtime["id"]:
        raise ValueError("develop publication tag no longer names its tested runtime image")
    run(["docker", "push", tag], directory / "publish.log")
    return {"tag": tag, "image": runtime["id"]}


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse only the small explicit surface supported by the develop gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new ignored directory for build and prerequisite evidence")
    parser.add_argument("--image", help="AVX512 develop runtime tag; AVX2 appends -avx2")
    parser.add_argument("--publish", action="store_true",
                        help="push both tested runtime images after both ISA gates pass")
    parser.add_argument("--cpu-isa", choices=SHIPPING_ISAS, action="append", dest="cpu_isas",
                        help="explicit local diagnostic ISA subset; publication requires both")
    args = parser.parse_args(argv)
    requested = args.cpu_isas or list(SHIPPING_ISAS)
    if len(set(requested)) != len(requested):
        parser.error("duplicate CPU ISA selection")
    args.cpu_isas = [isa for isa in SHIPPING_ISAS if isa in requested]
    if args.publish and (not args.image or set(args.cpu_isas) != set(SHIPPING_ISAS)):
        parser.error("--publish requires --image and both AVX512 and AVX2 gates")
    args.output = args.output.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    """Build both images, test each image, then make the optional push atomic by gate."""
    args = parse_arguments(argv)
    if args.output.exists():
        raise ValueError("develop image gate output already exists; preserve prior evidence and choose a new directory")
    args.output.mkdir(parents=True)
    source = source_identity()
    receipt = {"schema": 1, "source": source, "complete": False, "published": False,
               "variants": {}, "image": args.image}
    write_json(args.output / "develop-image-gate.json", receipt)
    with device_lease():
        for isa in args.cpu_isas:
            if source_identity() != source:
                raise ValueError("source changed after develop image gate admission")
            directory = args.output / isa.lower()
            directory.mkdir()
            build_args = argparse.Namespace(cpu_isa=isa)
            images = build(build_args, source, directory,
                           test_inventory=TestRunnerInventory.MODEL_FREE)
            require_image_pair(images, source, isa,
                               test_inventory=TestRunnerInventory.MODEL_FREE)
            prerequisites = run_prerequisites(images, directory)
            receipt["variants"][isa] = {
                "test_runner_image": images[ImageRole.TEST_RUNNER.value]["id"],
                "runtime_image": images[ImageRole.RUNTIME.value]["id"],
                "test_runner_inventory": TestRunnerInventory.MODEL_FREE.value,
                "prerequisites_digest": digest(prerequisites),
                "preflight_test_count": prerequisites["preflight_test_count"],
            }
            write_json(args.output / "develop-image-gate.json", receipt)
        if source_identity() != source:
            raise ValueError("source changed before develop image gate completion")
        receipt["complete"] = True
        if args.publish:
            # Do not publish one ISA before the other build/test result exists.
            # A network failure can leave a published tag, but never a tag for
            # an image that skipped its own local Unit/preflight gate.
            for isa in args.cpu_isas:
                images = {ImageRole.TEST_RUNNER.value: {"id": receipt["variants"][isa]["test_runner_image"]},
                          ImageRole.RUNTIME.value: {"id": receipt["variants"][isa]["runtime_image"]}}
                receipt["variants"][isa]["publication"] = publish_runtime(images, isa, args.image,
                                                                             args.output / isa.lower())
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
