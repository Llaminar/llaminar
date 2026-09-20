#!/usr/bin/env python3
"""Run manual HTTP E2E or benchmarks against a branch's published image pair.

This is a consumer of the production pipeline, not another test matrix or an
image builder/publisher. A same-source installed test companion exports the
canonical inventory; inference always uses immutable images pulled from GHCR.
Both full HTTP suites must pass before either ISA may be benchmarked. Compact
benchmark evidence and a deterministic SVG may be published to the branch;
these phase-specific results are not full production-image certificates.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from types import SimpleNamespace
import uuid

import docker_paths
from production_artifacts import (digest, image_identity, positive, validate_image_e2e,
                                  validate_image_e2e_coverage,
                                  validate_manifest, write_json)
import run_production_pipeline as pipeline
from run_develop_image_gate import runtime_tag

ROOT = Path(__file__).resolve().parents[2]
ISAS = pipeline.SHIPPING_ISAS
E2E_WORKFLOW = ".github/workflows/production-e2e.yml"
REPORT_DIRECTORY = Path("benchmarks/production/published")
README_BEGIN = "<!-- published-benchmarks:begin -->"
README_END = "<!-- published-benchmarks:end -->"


def git(*arguments: str) -> str:
    """Read repository identity without touching the user's index or branch."""
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def branch_image(repository: str, branch: str) -> str:
    """Use the develop gate's branch-tag convention and explicit ISA suffix."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository must be OWNER/REPO")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,119}", branch):
        raise ValueError("branch has no unambiguous Docker tag; supply --image explicitly")
    return f"ghcr.io/{repository.lower()}:{branch}"


def validate_pair(pair: dict) -> None:
    """Reject mixed commits, swapped ISAs, non-Release or non-GHCR artifacts."""
    source = pair["source"]
    if (pair.get("schema") != 1 or not re.fullmatch(r"[0-9a-f]{40}", source["revision"])
            or not re.fullmatch(r"[0-9a-f]{40}", source["tree"])
            or set(pair["images"]) != set(ISAS)):
        raise ValueError("published image pair is incomplete")
    for isa in ISAS:
        image = pair["images"][isa]
        pipeline.require_image(image, source, isa, pipeline.ImageRole.RUNTIME)
        if (not re.fullmatch(r"ghcr\.io/[^@\s]+@sha256:[0-9a-f]{64}", image["registry_ref"])
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", image["id"])):
            raise ValueError("published images require immutable GHCR and Docker identities")
    if pair["images"][ISAS[0]]["id"] == pair["images"][ISAS[1]]["id"]:
        raise ValueError("the shipping ISAs must be distinct images")


def pull_pair(args, directory: Path) -> dict:
    """Pull the latest tags once, then authenticate their committed source tree."""
    base = args.image or branch_image(args.repository, args.branch)
    if not base.startswith("ghcr.io/") or "@" in base:
        raise ValueError("--image must name the branch's mutable AVX512 GHCR tag")
    images = {}
    for isa in ISAS:
        tag = runtime_tag(base, isa)
        pipeline.run(["docker", "pull", tag], directory / "pull.log")
        identity = image_identity(tag)
        inspection = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", identity["id"]], text=True))[0]
        repository = tag.rsplit(":", 1)[0]
        references = [value for value in inspection.get("RepoDigests", [])
                      if value.startswith(repository + "@sha256:")]
        if len(references) != 1:
            raise ValueError(f"cannot pin one registry digest for {tag}")
        images[isa] = {**identity, "tag": tag, "registry_ref": references[0]}
    labels = images[ISAS[0]]["labels"]
    source = {"revision": labels["org.opencontainers.image.revision"],
              "tree": labels["org.llaminar.source_tree"], "dirty": False}
    pair = {"schema": 1, "branch": args.branch, "repository": args.repository,
            "source": source, "images": images,
            "workflow_revision": git("rev-parse", "HEAD")}
    validate_pair(pair)
    # A tag can legitimately lag this workflow's commit, but cannot name an
    # unrelated branch or a dirty local snapshot masquerading as that commit.
    if git("rev-parse", source["revision"] + "^{tree}") != source["tree"]:
        raise ValueError("published image source tree differs from its Git commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", source["revision"], "HEAD"],
                   cwd=ROOT, check=True)
    write_json(directory / "images.json", pair)
    return pair


def inventory_companion(build_args, source: dict, work: Path) -> dict:
    """Reuse an exact local test companion or build it on a cold runner.

    Inventory discovery is metadata-only, but exporting its multi-gigabyte
    image and BuildKit cache on every manual retry costs minutes. A local tag
    is only a lookup key: authenticate the immutable image ID and all source,
    ISA, backend, role, and inventory labels before admitting reuse. An absent
    image builds normally; a conflicting tag or Docker failure is fatal.
    """
    role = pipeline.ImageRole.TEST_RUNNER
    tag = pipeline.source_image_tag(source, build_args.cpu_isa, role,
                                    test_inventory=pipeline.TestRunnerInventory.FULL_MATRIX)
    listed = subprocess.check_output(
        ["docker", "image", "ls", "--quiet", "--no-trunc", tag], text=True,
    ).strip().splitlines()
    if len(listed) > 1:
        raise ValueError(f"local inventory tag resolved to multiple images: {tag}")
    if listed:
        image = {"tag": tag, **image_identity(tag)}
        if image["id"] != listed[0]:
            raise ValueError(f"local inventory tag changed during inspection: {tag}")
        pipeline.require_image(image, source, build_args.cpu_isa, role)
        print(f"[published-images] REUSE inventory image={image['id']}", flush=True)
        return {role.value: image}
    return pipeline.build(build_args, source, work, roles=(role,))


def discover_inventory(args, pair: dict, directory: Path) -> dict:
    """Export metadata from the exact image source; never infer cases in Python.

    The matrix is ISA-independent and declares all backends. One full-backend
    AVX2 test companion suffices for discovery; it never runs model inference.
    No source/binary overlays are mounted into either tested runtime image.
    """
    work = directory / "inventory"
    work.mkdir()
    build_args = SimpleNamespace(cpu_isa="AVX2", models=args.models,
                                 model_ramdisk_root=args.model_ramdisk_root,
                                 reference_cache_root=work / "reference-cache")
    build_args.reference_cache_root.mkdir()
    images = inventory_companion(build_args, pair["source"], work)
    pipeline.run_test_runner(images, ["python3", "scripts/ci/model_parity_inventory.py",
        "--build-dir", "build_v2_integration", "--scope", "all",
        "--export-manifest", "/ci-results/container-all-cells.json",
        "--source-revision", pair["source"]["revision"]], build_args, work, "discovery.log")
    inventory = pipeline.remap_manifest(
        json.loads((work / "container-all-cells.json").read_text()), args.models)
    manifest = pipeline.e2e_projection(inventory, pair["source"]["revision"])
    write_json(directory / "all-cells.json", inventory)
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "inventory-image.json", images)
    return manifest


def build_driver(pair: dict, directory: Path) -> str:
    """Install only orchestration tools; inference images remain byte-for-byte fixed."""
    tag = "llaminar-ci:published-suite-driver-" + pair["images"]["AVX2"]["id"][7:23]
    pipeline.run(["docker", "buildx", "build", "--load", "--network=host", "--progress=plain",
        "--file", str(ROOT / "scripts/ci/Dockerfile.published-suite-driver"),
        "--build-arg", "RUNTIME_IMAGE=" + pair["images"]["AVX2"]["registry_ref"],
        "--tag", tag, str(ROOT / "scripts/ci")], directory / "build-driver.log")
    image = image_identity(tag)
    if image["labels"].get("org.llaminar.image_role") != "suite-driver":
        raise ValueError("orchestration tools require a distinct non-runtime image role")
    write_json(directory / "driver-image.json", image)
    return image["id"]


def restore_lane_ownership(driver: str, lane: Path, owner_uid: int, owner_gid: int) -> None:
    """Return nested-driver evidence to the invoking runner after completion.

    The suite driver may need the persistent model-cache owner's UID to retain
    cache locks and tmpfs ownership.  That identity must never escape into the
    checkout or runner temporary directory: a later Actions checkout cannot
    remove a root-owned evidence subtree.  This narrowly scoped control
    container changes only the completed lane back to the UID/GID that created
    it; it neither touches model/cache data nor changes the runtime image.
    """
    if owner_uid < 0 or owner_gid < 0:
        raise ValueError("evidence owner UID/GID must be non-negative")
    resolved = lane.resolve(strict=True)
    command = [
        "docker", "run", "--rm", "--user", "0:0",
        *docker_paths.mounts([(resolved, str(resolved), False)]),
        "--entrypoint", "/bin/chown", driver, "--recursive",
        f"{owner_uid}:{owner_gid}", str(resolved),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT, timeout=30)


def driver_shared_root_environment(pairs: list[tuple[Path, str, bool]]) -> str | None:
    """Project ARC's shared-root topology into the mounted driver namespace.

    The runner's declaration can include cache roots that this short-lived
    suite driver intentionally does not receive.  Passing that superset makes
    the nested resolver reject a valid model/log path because it correctly
    refuses a missing declared root.  Keep only roots containing at least one
    source mounted into this driver.  The resulting environment is still an
    exact subset of the infrastructure declaration, never a caller-invented
    bind permission.
    """
    if os.environ.get(docker_paths.SHARED_DAEMON_ROOTS_ENV) is None:
        return None
    roots = docker_paths.shared_daemon_roots()
    sources = tuple(source.resolve(strict=True) for source, _, _ in pairs)
    selected = tuple(root for root in roots
                     if any(source.is_relative_to(root) for source in sources))
    return os.pathsep.join(map(str, selected)) or None


def run_in_driver(args, driver: str, command: list[str], lane: Path, log: str) -> None:
    """Run canonical harnesses as the existing tmpfs owner, including under ARC.

    ARC's pod deliberately mounts models/cache read-only. A sibling Docker
    control container receives the same physical cache read-write and retains
    its existing owner UID, lock and sealing protocol. No cache is copied,
    chowned, cleared or made world-writable. The new evidence directory alone
    grants its group write access; it remains owned by the invoking user.
    Source is read-only and only the control scripts run there. Actual model
    containers receive the published IDs through the canonical runner argv.
    """
    cache_uid = (args.model_ramdisk_root / "cache").stat().st_uid
    lane_stat = lane.stat()
    lane_owner_uid, lane_owner_gid = lane_stat.st_uid, lane_stat.st_gid
    lane.chmod(stat.S_IMODE(lane_stat.st_mode) | stat.S_ISGID | stat.S_IWGRP)
    endpoint = os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock")
    docker_paths.local_docker_endpoint()
    socket = Path(endpoint.removeprefix("unix://")).resolve(strict=True)
    pairs = [(ROOT, str(ROOT), True), (args.output, str(args.output), True),
             (args.models, str(args.models), True),
             (args.model_ramdisk_root, str(args.model_ramdisk_root), False),
             (lane, str(lane), False), (socket, "/var/run/docker.sock", False)]
    if args.e2e_bundle:
        bundle = args.e2e_bundle.resolve(strict=True)
        pairs.append((bundle, str(bundle), True))
    # The nested canonical HTTP harness launches the immutable runtime image
    # through the same host socket.  ARC's source/model/evidence roots are
    # deliberately mounted at identical paths in all three namespaces, but an
    # ordinary Docker environment does not inherit its caller's variables.
    # Propagate this infrastructure-owned declaration so docker_paths.py can
    # choose the direct, audited mapping rather than inspect the ARC pod name.
    shared_roots = driver_shared_root_environment(pairs)
    name = "llaminar-suite-driver-" + uuid.uuid4().hex
    launch = ["docker", "run", "--rm", "--init", "--name", name,
        *docker_paths.device_args(driver, "CPU+CUDA+ROCm", user=f"{cache_uid}:{lane_stat.st_gid}"),
        "--group-add", str(socket.stat().st_gid), *docker_paths.mounts(pairs),
        "--workdir", str(ROOT), "--env", "DOCKER_HOST=unix:///var/run/docker.sock",
        driver, "python3", *command[1:]]
    if shared_roots is not None:
        # Insert before the image; everything after the image is the immutable
        # control command, whose argv must remain untouched.
        image_index = launch.index(driver)
        launch[image_index:image_index] = ["--env", f"{docker_paths.SHARED_DAEMON_ROOTS_ENV}={shared_roots}"]
    try:
        pipeline.run(launch, lane / log)
    finally:
        # Signal the Python driver first so its process-group/finally cleanup
        # can retire the exact server/MPI container it owns before this parent.
        subprocess.run(["docker", "kill", "--signal", "SIGINT", name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        try:
            subprocess.run(["docker", "wait", name], check=False, timeout=30,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            pass
        subprocess.run(["docker", "rm", "--force", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # The driver shares a cache owner with retained tmpfs pages, while the
        # outer runner owns evidence.  Repair that boundary only after every
        # process that could still write the lane is reaped.
        restore_lane_ownership(driver, lane, lane_owner_uid, lane_owner_gid)


def run_e2e(args, pair: dict, manifest: dict, directory: Path) -> None:
    """Collect every independent HTTP cell on both ISAs before judging the pair.

    An incomplete report is an infrastructure failure and stops the run. A
    complete report with red cells is retained, then the other ISA is tested;
    only two all-green reports may publish a reusable benchmark receipt.
    """
    receipt = {"schema": 1, "complete": False, "images_digest": digest(pair),
               "manifest_digest": digest(manifest), "reports": {}, "failed_cells": {}}
    write_json(directory / "e2e-receipt.json", receipt)
    driver = build_driver(pair, directory)
    for isa in ISAS:
        lane = directory / isa.lower()
        lane.mkdir()
        report = lane / "e2e.json"
        print(f"[published-images] E2E {isa}: {len(manifest['cells'])} canonical cells", flush=True)
        cell_failure = None
        try:
            run_in_driver(args, driver, [sys.executable, str(ROOT / "scripts/ci/run_model_parity_e2e.py"),
                "--manifest", str(directory / "manifest.json"),
                "--source-revision", pair["source"]["revision"],
                "--container-image", pair["images"][isa]["id"],
                "--model-ramdisk-root", str(args.model_ramdisk_root),
                "--report", str(report)], lane, "e2e.log")
        except subprocess.CalledProcessError as error:
            cell_failure = error
        # A failed process without a complete canonical report is not an
        # isolated red cell. Stop rather than guessing whether the next lane
        # could run safely after a driver or cleanup failure.
        evidence = json.loads(report.read_text())
        validate_image_e2e_coverage(evidence, manifest, pair["images"][isa]["id"])
        if (cell_failure is None) != evidence["correctness_passed"]:
            raise ValueError(f"E2E {isa} exit status disagrees with complete cell evidence")
        receipt["reports"][isa] = digest(evidence)
        receipt["failed_cells"][isa] = [row["case"] for row in evidence["cells"]
                                        if row["outcome"] != "passed"]
        write_json(directory / "e2e-receipt.json", receipt)
        print(f"[published-images] E2E {isa}: passed="
              f"{len(evidence['cells']) - len(receipt['failed_cells'][isa])} "
              f"failed={len(receipt['failed_cells'][isa])}", flush=True)
    failed = {isa: cells for isa, cells in receipt["failed_cells"].items() if cells}
    if failed:
        print(f"[published-images] E2E complete failure map: {failed}", flush=True)
        raise ValueError("published image E2E cell failures remain on one or both ISAs")
    for isa in ISAS:
        evidence = json.loads((directory / isa.lower() / "e2e.json").read_text())
        validate_image_e2e(evidence, manifest, pair["images"][isa]["id"])
    receipt["complete"] = True
    write_json(directory / "e2e-receipt.json", receipt)


def admit_e2e(bundle: Path, pair: dict) -> tuple[dict, dict]:
    """Authenticate both image-bound suites before any benchmark is started."""
    previous = json.loads((bundle / "images.json").read_text())
    validate_pair(previous)
    if (previous["branch"] != pair["branch"] or previous["repository"] != pair["repository"]
            or previous["images"] != pair["images"] or previous["source"] != pair["source"]):
        raise ValueError("latest image pair differs from E2E evidence; run the E2E workflow again")
    manifest = json.loads((bundle / "manifest.json").read_text())
    inventory = json.loads((bundle / "all-cells.json").read_text())
    if manifest != pipeline.e2e_projection(inventory, pair["source"]["revision"]):
        raise ValueError("E2E selection is not the complete canonical inventory projection")
    receipt = json.loads((bundle / "e2e-receipt.json").read_text())
    if (receipt.get("schema") != 1 or receipt.get("complete") is not True
            or receipt.get("images_digest") != digest(previous)
            or receipt.get("manifest_digest") != digest(manifest)
            or set(receipt.get("reports", {})) != set(ISAS)):
        raise ValueError("E2E pair receipt is incomplete or stale")
    reports = {}
    for isa in ISAS:
        report = json.loads((bundle / isa.lower() / "e2e.json").read_text())
        validate_image_e2e(report, manifest, pair["images"][isa]["id"])
        if digest(report) != receipt["reports"][isa]:
            raise ValueError("E2E report differs from its completed receipt")
        reports[isa] = report
    return manifest, reports


def download_e2e(args, destination: Path) -> tuple[Path, str]:
    """Download only successful manual E2E evidence from the selected branch."""
    run_id = args.e2e_run
    if not run_id:
        data = json.loads(subprocess.check_output(["gh", "api", "--method", "GET",
            f"repos/{args.repository}/actions/workflows/production-e2e.yml/runs",
            "-f", f"branch={args.branch}", "-f", "event=workflow_dispatch",
            "-f", "status=success", "-f", "per_page=1"], text=True))
        if not data["workflow_runs"]:
            raise ValueError("no successful manual E2E run exists for this branch")
        run_id = str(data["workflow_runs"][0]["id"])
    if not run_id.isdecimal():
        raise ValueError("E2E run ID must be a positive integer")
    run = json.loads(subprocess.check_output(["gh", "api",
        f"repos/{args.repository}/actions/runs/{run_id}"], text=True))
    if (run["path"] != E2E_WORKFLOW or run["event"] != "workflow_dispatch"
            or run["head_branch"] != args.branch or run["status"] != "completed"
            or run["conclusion"] != "success"):
        raise ValueError("selected run is not a successful manual E2E workflow on this branch")
    subprocess.run(["gh", "run", "download", run_id, "--repo", args.repository,
                    "--name", f"published-e2e-{run_id}", "--dir", str(destination)], check=True)
    return destination, run["html_url"]


def validate_benchmark(report: dict, pair: dict, manifest: dict, e2e: dict, isa: str) -> None:
    """Reject partial, duplicate, diagnostic, wrong-image or regressed results."""
    expected = {row["case"]: row for row in validate_manifest(manifest, pair["source"]["revision"])}
    rows = report.get("cells", [])
    if (report.get("complete") is not True or report.get("passed") is not True
            or report.get("diagnostic") is not False
            or report.get("source_revision") != pair["source"]["revision"]
            or report.get("image") != pair["images"][isa]["id"]
            or report.get("manifest_digest") != digest(manifest)
            or report.get("e2e_report_digest") != digest(e2e)
            or len(rows) != len(expected) or {row["case"] for row in rows} != set(expected)):
        raise ValueError(f"{isa} benchmark evidence is incomplete, stale or regressed")
    for row in rows:
        config = expected[row["case"]]["configuration"]
        identity = row["identity"]
        if (identity.get("cpu_isa") != isa or identity.get("configuration") !=
                {**config, "model": Path(config["model"]).name}):
            raise ValueError("benchmark policy differs from its canonical cell")
        for phase in ("prefill", "decode"):
            positive(row["tokens_per_second"][phase])


def run_benchmarks(args, pair: dict, directory: Path) -> dict:
    """Consume the full E2E pair, then use the unchanged production benchmark runner."""
    if args.e2e_bundle:
        bundle, evidence_url = args.e2e_bundle.resolve(strict=True), "local"
    else:
        bundle, evidence_url = download_e2e(args, directory / "e2e-evidence")
    manifest, e2e = admit_e2e(bundle, pair)
    write_json(directory / "manifest.json", manifest)
    reports = {}
    driver = build_driver(pair, directory)
    for isa in ISAS:
        lane = directory / isa.lower()
        lane.mkdir()
        report = lane / "benchmarks.json"
        print(f"[published-images] BENCHMARK {isa}: {len(manifest['cells'])} canonical cells", flush=True)
        run_in_driver(args, driver, [sys.executable, str(ROOT / "scripts/ci/run_model_parity_benchmarks.py"),
            "--manifest", str(bundle / "manifest.json"),
            "--source-revision", pair["source"]["revision"],
            "--image", pair["images"][isa]["id"],
            "--e2e-report", str(bundle / isa.lower() / "e2e.json"),
            "--model-ramdisk-root", str(args.model_ramdisk_root),
            "--report", str(report)], lane, "benchmarks.log")
        reports[isa] = json.loads(report.read_text())
        validate_benchmark(reports[isa], pair, manifest, e2e[isa], isa)
    # Publish compact phase evidence, not raw logs or a full-image certificate.
    result = {"schema": 1, "source": pair["source"], "branch": pair["branch"],
              "repository": pair["repository"],
              "images": pair["images"], "e2e_run": evidence_url,
              "workflow_run": args.run_url,
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "full-http-e2e-and-benchmarks", "full_image_certification": False,
              "variants": {isa: {"hardware": report["hardware"], "workload": report["workload"],
                  "report_digest": digest(report), "e2e_report_digest": digest(e2e[isa]),
                  "cells": [{key: value for key, value in row.items() if key != "artifacts"}
                            for row in report["cells"]]} for isa, report in reports.items()}}
    write_json(directory / "results.json", result)
    from published_benchmark_chart import render_chart
    (directory / "benchmarks.svg").write_text(render_chart(result))
    return result


def readme_with_results(text: str, result: dict) -> str:
    """Replace exactly one owned block; preserve every other README section."""
    if text.count(README_BEGIN) != 1 or text.count(README_END) != 1:
        raise ValueError("README must contain exactly one published benchmark block")
    start, stop = text.index(README_BEGIN), text.index(README_END)
    if stop < start:
        raise ValueError("README benchmark markers are reversed")
    revision = result["source"]["revision"]
    body = (f"{README_BEGIN}\n\n"
            f"![Published-image prefill and decode benchmarks]({REPORT_DIRECTORY}/benchmarks.svg)\n\n"
            f"Tested image source: [`{revision[:12]}`](https://github.com/{result['repository']}/commit/{revision}). "
            f"Both AVX512 and AVX2 passed the full HTTP E2E suite before measurement.\n"
            f"[Exact configurations, image digests and samples]({REPORT_DIRECTORY}/results.json). "
            "This is E2E/benchmark evidence, not full production-image certification.\n\n")
    return text[:start] + body + text[stop:]


def publish_results(args, directory: Path, result: dict) -> None:
    """Commit only chart/JSON/README via a private index and fast-forward push.

    The parent is the current remote branch, not the tested image revision.
    Concurrent source commits survive. A concurrent push after this snapshot
    rejects our ordinary fast-forward push; no force push or branch reset occurs.
    The local checkout and its staging index are never changed.
    """
    import tempfile
    subprocess.run(["git", "fetch", "origin", f"refs/heads/{args.branch}"], cwd=ROOT, check=True)
    parent = git("rev-parse", "FETCH_HEAD")
    old_readme = git("show", f"{parent}:README.md") + "\n"
    readme = readme_with_results(old_readme, result)
    payloads = {"README.md": readme.encode(),
                str(REPORT_DIRECTORY / "results.json"): (directory / "results.json").read_bytes(),
                str(REPORT_DIRECTORY / "benchmarks.svg"): (directory / "benchmarks.svg").read_bytes()}
    with tempfile.TemporaryDirectory(prefix="llaminar-benchmark-index-") as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index"),
               "GIT_AUTHOR_NAME": "Llaminar benchmarks", "GIT_COMMITTER_NAME": "Llaminar benchmarks",
               "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
               "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com"}
        subprocess.run(["git", "read-tree", parent], cwd=ROOT, env=env, check=True)
        for path, content in payloads.items():
            blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"],
                                           input=content, cwd=ROOT).decode().strip()
            subprocess.run(["git", "update-index", "--add", "--cacheinfo", "100644", blob, path],
                           cwd=ROOT, env=env, check=True)
        tree = subprocess.check_output(["git", "write-tree"], cwd=ROOT, env=env, text=True).strip()
        commit = subprocess.check_output(["git", "commit-tree", tree, "-p", parent,
            "-m", f"benchmarks: published images {result['source']['revision'][:12]} [skip ci]"],
            cwd=ROOT, env=env, text=True).strip()
    subprocess.run(["git", "push", "origin", f"{commit}:refs/heads/{args.branch}"], cwd=ROOT, check=True)
    write_json(directory / "publication.json", {"commit": commit, "parent": parent,
        "branch": args.branch, "tested_source": result["source"]["revision"]})


def parse_arguments(argv=None):
    """Keep publication explicit and offer the same entry point locally and in CI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", choices=("e2e", "benchmarks"))
    parser.add_argument("--repository", default=os.environ.get("GITHUB_REPOSITORY", "Llaminar/llaminar"))
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME") or git("branch", "--show-current"))
    parser.add_argument("--image", help="AVX512 branch tag; AVX2 uses the canonical -avx2 suffix")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path("/opt/llaminar-models"))
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--e2e-run", help="successful manual E2E Actions run ID; defaults to the latest")
    evidence.add_argument("--e2e-bundle", type=Path, help="local completed E2E output directory")
    parser.add_argument("--run-url", default="local")
    parser.add_argument("--publish", action="store_true", help="commit only successful complete benchmark results")
    args = parser.parse_args(argv)
    if args.suite != "benchmarks" and (args.publish or args.e2e_run or args.e2e_bundle):
        parser.error("E2E evidence inputs and publication apply only to benchmarks")
    subprocess.run(["git", "check-ref-format", f"refs/heads/{args.branch}"], check=True)
    return args


def main(argv=None) -> int:
    """Own a single device lease and leave durable partial evidence on failure."""
    args = parse_arguments(argv)
    args.output = args.output.resolve()
    args.models = args.models.resolve(strict=True)
    args.model_ramdisk_root = args.model_ramdisk_root.resolve(strict=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with pipeline.device_lease():
        pair = pull_pair(args, args.output)
        docker_paths.publish_model_cache(args.model_ramdisk_root)
        if args.suite == "e2e":
            manifest = discover_inventory(args, pair, args.output)
            run_e2e(args, pair, manifest, args.output)
        else:
            result = run_benchmarks(args, pair, args.output)
            if args.publish:
                publish_results(args, args.output, result)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        print(f"[published-images] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
