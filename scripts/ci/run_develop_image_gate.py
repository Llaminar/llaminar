#!/usr/bin/env python3
## @file run_develop_image_gate.py
# @brief Build the develop images and exercise the PR test inventories in scoped containers.
"""Build and optionally validate/publish both develop runtime images.

This intentionally small gate is distinct from the full production
certification pipeline. The develop push selects build-only publication of
AVX512 and AVX2 full-backend runtimes without compiling or running test images.
Hosted jobs stage one verified image each. A final job publishes both by digest.
The separate PR gate runs the complete CMake-owned Unit inventory and the
canonical host preflight lane inside a device-free container. AVX2 PRs also
require every hardware-qualified production preflight test in a separate GPU container.
Neither policy discovers models, stages a
corpus, launches a server, runs mathematical parity, E2E, remote MPI, or a
benchmark, and it never calls an image a certified release artifact.
"""
from __future__ import annotations

import argparse
from enum import Enum
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

import docker_paths
from run_develop_pr_tests import PRTestScope
from production_artifacts import CPUISA, digest, image_identity, write_json, validate_preflight_qualification
from run_production_cross_host_e2e import CPU_ISA_FEATURES
from run_production_pipeline import (
    TestRunnerInventory, ImageRole, SHIPPING_ISAS, build, logged_container_command,
    require_image, run, source_identity,
)

ROOT = Path(__file__).resolve().parents[2]


class PRGPUCoverage(str, Enum):
    """Explicit accelerator requirement; missing GPUs never change the policy."""

    NONE = "none"
    CUDA_ROCM = "CUDA+ROCm"

    @property
    def validation(self) -> str:
        """Name the complete requested PR evidence without implying release approval."""
        return "unit-and-host" if self is PRGPUCoverage.NONE else "unit-and-hardware-preflight"


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


def _run_pr_scope(images: dict, directory: Path, scope: PRTestScope) -> dict:
    """Run one container with exactly its declared CPU or GPU ownership.

    The test runner already contains its immutable model-free executable inventory
    and installed receipt.  Bind only the evidence directory; attaching models
    or a corpus would make this deliberately model-free develop gate depend on
    a second, undocumented input surface.
    """
    image = images[ImageRole.TEST_RUNNER.value]["id"]
    image_revision = (images[ImageRole.TEST_RUNNER.value].get("labels") or {}).get(
        "org.opencontainers.image.revision", "")
    if not (len(image_revision) == 40 and
            all(c in "0123456789abcdef" for c in image_revision)):
        # Reviewer-93 item 3: source binding needs the exact 40-lower-hex label;
        # an unlabelled runner image cannot bind, so the revision is omitted
        # and the diagnostic records the run as UNBOUND.
        image_revision = ""
    prefix = "pr-tests" if scope is PRTestScope.HOST else "pr-tests/gpu"
    name = "llaminar-develop-" + prefix.replace("/", "-") + "-" + uuid.uuid4().hex
    command = ["python3", "scripts/ci/run_develop_pr_tests.py",
               "--build-dir", "build_v2_integration",
               "--installed-build-receipt", "/src/installed-tests.json",
               "--output", f"/ci-results/{prefix}"]
    if scope is PRTestScope.GPU:
        command += ["--scope", scope.value]
    # Unit and host never see device nodes. The later GPU scope requires both
    # vendors through the same Docker device/group authority as production.
    # Death tests intentionally trap in subprocesses. A core dump for each trap
    # can exhaust the replay deadline on hosts with a crash-reporting service.
    launch = ["docker", "run", "--rm", "--name", name, "--ulimit", "core=0:0",
              *docker_paths.device_args(image, scope.backends, user=f"{os.getuid()}:{os.getgid()}"),
              *(part for group in os.getgroups() for part in ("--group-add", str(group))),
              *docker_paths.mounts([(directory, "/ci-results", False)]),
              *(["-e", f"LLAMINAR_IMAGE_REVISION={image_revision}"]
                if image_revision else []),
              "-w", "/src", image,
              *logged_container_command(command, f"/ci-results/{prefix}.process.log")]
    try:
        run(launch, directory / f"{prefix}.log")
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    report_path = directory / prefix / "pr-tests.json"
    report = json.loads(report_path.read_text())
    if (report.get("scope") != scope.value or report.get("complete") is not True
            or report.get("return_code") != 0
            or set(report.get("lanes", {})) != set(scope.lanes)):
        raise ValueError(f"PR {scope.value} receipt is incomplete or unsuccessful")
    for lane in report["lanes"].values():
        tests = lane.get("tests", [])
        if (lane.get("complete") is not True or lane.get("return_code") != 0
                or (not tests and scope is PRTestScope.HOST) or lane.get("test_count") != len(tests)
                or len(set(tests)) != len(tests)):
            raise ValueError(f"PR {scope.value} lane is incomplete or unsuccessful")
    return report


def run_pr_tests(images: dict, directory: Path,
                 gpu_coverage: PRGPUCoverage = PRGPUCoverage.NONE) -> dict:
    """Finish CPU gates before opening either GPU, retaining separate receipts."""
    try:
        host = _run_pr_scope(images, directory, PRTestScope.HOST)
    except subprocess.CalledProcessError as error:
        # When CTest finished with failures the CPU container has retired. The
        # independent GPU scope can still produce its hardware evidence.
        if error.returncode == 8 and gpu_coverage is PRGPUCoverage.CUDA_ROCM:
            _run_pr_scope(images, directory, PRTestScope.GPU)
        raise
    if gpu_coverage is PRGPUCoverage.NONE:
        return host
    gpu = _run_pr_scope(images, directory, PRTestScope.GPU)
    qualification = gpu.get("qualification")
    if not isinstance(qualification, dict):
        raise ValueError("PR hardware preflight is missing its complete qualification")
    validate_preflight_qualification(qualification.get("registered"), qualification)
    executed = [*host["lanes"]["host"]["tests"],
                *(name for lane in gpu["lanes"].values() for name in lane["tests"])]
    if sorted(executed) != sorted(qualification["selected"]):
        raise ValueError("PR receipts do not cover the exact hardware-qualified production preflight")
    report = {"schema": 1, "scope": gpu_coverage.validation, "complete": True,
              "return_code": 0, "lanes": {**host["lanes"], **gpu["lanes"]},
              "qualification": qualification,
              "scope_digests": {host["scope"]: digest(host), gpu["scope"]: digest(gpu)}}
    write_json(directory / "pr-tests/pr-validation.json", report)
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


def stage_runtime(runtime: str, isa: str, base: str, directory: Path) -> str:
    """Upload one admitted image to a rolling scratch tag and capture its manifest digest."""
    tag = runtime_tag(f"{base}-staging", isa)
    run(["docker", "tag", runtime, tag], directory / "stage.log")
    run(["docker", "push", tag], directory / "stage.log")
    manifest = json.loads(subprocess.check_output(
        ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Manifest.Digest}}", tag],
        text=True))
    if not isinstance(manifest, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", manifest):
        raise ValueError("develop staging tag has no immutable manifest digest")
    if remote_image_id(f"{base.rsplit(':', 1)[0]}@{manifest}") != runtime:
        raise ValueError("staged manifest does not contain the admitted runtime")
    return manifest


def publish_staged_pair(staging: Path, source: dict, base: str, directory: Path) -> dict:
    """Admit both immutable registry images before writing any public develop alias."""
    images = {}
    repository = base.rsplit(":", 1)[0]
    log = directory / "publish.log"
    for isa in SHIPPING_ISAS:
        lane = json.loads((staging / f"develop-stage-{isa}" / "develop-image-gate.json").read_text())
        variants = lane.get("variants", {})
        if (lane.get("schema") != 1 or lane.get("source") != source or lane.get("image") != base
                or lane.get("validation") != "build-only" or lane.get("complete") is not True
                or lane.get("published") is not False or set(variants) != {isa}):
            raise ValueError(f"invalid develop staging receipt for {isa}")
        variant = variants[isa]
        runtime, manifest = (variant.get(key) for key in ("runtime_image", "staged_manifest"))
        if not all(isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value)
                   for value in (runtime, manifest)):
            raise ValueError(f"invalid develop staging digest for {isa}")
        ref = f"{repository}@{manifest}"
        if remote_image_id(ref) != runtime:
            raise ValueError(f"staged manifest does not contain the admitted {isa} runtime")
        image = json.loads(subprocess.check_output(
            ["docker", "buildx", "imagetools", "inspect", "--format", "{{json .Image}}", ref],
            text=True))
        require_image({"labels": image.get("config", {}).get("Labels")}, source, isa, ImageRole.RUNTIME)
        pinned = revision_tag(base, isa, source["revision"])
        present = remote_image_id(pinned)
        if present is not None and present != runtime:
            raise ValueError("source-pinned develop tag already names different image bytes")
        images[isa] = (ref, runtime, pinned, present)
    if source_identity() != source:
        raise ValueError("source changed before develop image publication")

    # Keep the original single-platform manifest when copying a registry tag.
    def copy(ref: str, alias: str, runtime: str) -> None:
        run(["docker", "buildx", "imagetools", "create", "--prefer-index=false", "-t", alias, ref], log)
        if remote_image_id(alias) != runtime:
            raise ValueError(f"develop publication tag {alias} names unexpected image bytes")

    for ref, runtime, pinned, present in images.values():
        if present is None:
            copy(ref, pinned, runtime)
    result = {}
    for isa, (ref, runtime, pinned, _) in images.items():
        tag = runtime_tag(base, isa)
        copy(ref, tag, runtime)
        result[isa] = {"runtime_image": runtime,
                       "publication": {"tag": tag, "ref_tag": pinned, "image": runtime}}
    return result


def parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    """Parse only the small explicit surface supported by the develop gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new ignored directory for build or required PR scope evidence")
    parser.add_argument("--image", help="AVX512 develop runtime tag; AVX2 appends -avx2")
    parser.add_argument("--build-only", action="store_true",
                        help="build runtime images without compiling or executing Unit/preflight tests")
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument("--publish", action="store_true",
                             help="push both runtime images after both local builds succeed")
    publication.add_argument("--stage", action="store_true",
                             help="push one runtime image to its staging tag for CI publication")
    publication.add_argument("--publish-staged", type=Path,
                             help="publish a pair from the two CI staging receipt directories")
    parser.add_argument("--cpu-isa", choices=SHIPPING_ISAS, action="append", dest="cpu_isas",
                        help="explicit local diagnostic ISA subset; publication requires both")
    parser.add_argument("--pr-gpus", type=PRGPUCoverage,
                        choices=[coverage.value for coverage in PRGPUCoverage],
                        default=None,
                        help="PR accelerator coverage (AVX2-only test jobs default to CUDA+ROCm)")
    args = parser.parse_args(argv)
    requested = args.cpu_isas or list(SHIPPING_ISAS)
    if len(set(requested)) != len(requested):
        parser.error("duplicate CPU ISA selection")
    args.cpu_isas = [isa for isa in SHIPPING_ISAS if isa in requested]
    if args.pr_gpus is None:
        # The protected reusable workflow selects one ISA per runner.
        args.pr_gpus = (PRGPUCoverage.CUDA_ROCM if args.cpu_isas == ["AVX2"] and not args.build_only
                        else PRGPUCoverage.NONE)
    if args.publish and (not args.image or set(args.cpu_isas) != set(SHIPPING_ISAS)):
        parser.error("--publish requires --image and both AVX512 and AVX2 gates")
    if args.publish and not args.build_only:
        parser.error("--publish requires --build-only; PR validation never publishes")
    if args.stage and (not args.build_only or not args.image or len(args.cpu_isas) != 1):
        parser.error("--stage requires --build-only, --image and one explicit --cpu-isa")
    if args.publish_staged and (args.build_only or not args.image or set(args.cpu_isas) != set(SHIPPING_ISAS)):
        parser.error("--publish-staged requires --image and both ISA receipts without a build")
    if (args.build_only or args.publish_staged) and args.pr_gpus is not PRGPUCoverage.NONE:
        parser.error("--pr-gpus requires a PR validation run")
    args.output = args.output.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    """Finish both admitted ISA transactions before publishing either runtime."""
    args = parse_arguments(argv)
    if not args.build_only and not args.publish_staged:
        for isa in args.cpu_isas:
            require_host_cpu_isa(isa)
    if args.output.exists():
        raise ValueError("develop image gate output already exists; preserve prior evidence and choose a new directory")
    args.output.mkdir(parents=True)
    source = source_identity()
    receipt = {"schema": 1, "source": source, "complete": False, "published": False,
               "variants": {}, "image": args.image,
               "validation": "build-only" if args.build_only or args.publish_staged else args.pr_gpus.validation}
    write_json(args.output / "develop-image-gate.json", receipt)
    if args.publish_staged:
        receipt["variants"] = publish_staged_pair(args.publish_staged, source, args.image, args.output)
        receipt["complete"] = receipt["published"] = True
        write_json(args.output / "develop-image-gate.json", receipt)
        return 0
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
            if args.stage:
                variant["staged_manifest"] = stage_runtime(variant["runtime_image"], isa, args.image, directory)
        else:
            images = build(build_args, source, directory,
                           roles=(ImageRole.TEST_RUNNER,),
                           test_inventory=TestRunnerInventory.MODEL_FREE)
            require_image(images[ImageRole.TEST_RUNNER.value], source, isa, ImageRole.TEST_RUNNER,
                          test_inventory=TestRunnerInventory.MODEL_FREE)
            pr_tests = run_pr_tests(images, directory, args.pr_gpus)
            variant = {
                "test_runner_image": images[ImageRole.TEST_RUNNER.value]["id"],
                "test_runner_inventory": TestRunnerInventory.MODEL_FREE.value,
                "pr_tests_digest": digest(pr_tests),
                "unit_test_count": pr_tests["lanes"]["unit"]["test_count"],
                "host_test_count": pr_tests["lanes"]["host"]["test_count"],
            }
            if args.pr_gpus is PRGPUCoverage.CUDA_ROCM:
                variant["gpu_test_counts"] = {
                    lane: pr_tests["lanes"][lane]["test_count"] for lane in PRTestScope.GPU.lanes}
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
