#!/usr/bin/env python3
"""Build, certify and optionally publish AVX512 and AVX2 full runtime images.

Local and GitHub runs execute the same ordered pipeline. The numerical driver
owns Unit -> ProductionParityPreflight -> full model parity; this driver then
runs all tagged E2E cells and exactly those benchmarks on a pinned Release
image. Only the final transition embeds certificates. Official publication
commits compact JSON/high-water proposals, never corpora or diagnostic dumps.

--through supports piecewise exercise. --resume reuses only identity-bound,
digest-checked completed phases; modifying source invalidates the entire run.
No selector or skip-gate option exists on this certifying entry point.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from enum import Enum
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

import docker_paths
from production_artifacts import digest, image_identity, ratchet, validate_image_e2e, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[2]
SHIPPING_ISAS = ("AVX512", "AVX2")


class Phase(str, Enum):
    """Only successful ordered transitions can reach certification."""
    BUILD = "build"
    PARITY = "parity"
    E2E = "e2e"
    BENCHMARKS = "benchmarks"
    CERTIFY = "certify"


def run(command: list[str], log: Path, **kwargs) -> None:
    """Retain full logs and surface cell progress without compiler-output noise."""
    print(f"[production-ci] log={log}", flush=True)
    with log.open("a") as stream:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, start_new_session=True, **kwargs) as process:
            try:
                for line in process.stdout:
                    stream.write(line)
                    stream.flush()
                    if line.startswith(("[production-", "[model-parity-")):
                        print(line, end="", flush=True)
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            finally:
                if process.poll() is None:
                    # Retire the whole launcher/MPI family on interruption.
                    # The numerical driver owns this process-group protocol.
                    from run_production_parity_campaigns import _terminate_process_group
                    _terminate_process_group(process)


def source_identity() -> dict:
    """Snapshot source with a private index; never stage or commit user edits.

    Git's tree identity covers tracked code and nonignored new source. Generated
    reports and the optional corpus payload are excluded by repository policy.
    The snapshot is used as Docker's context, so edits after admission cannot
    accidentally produce a differently labeled runtime.
    """
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    with tempfile.TemporaryDirectory(prefix="llaminar-ci-index-") as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}
        subprocess.run(["git", "read-tree", "HEAD"], cwd=ROOT, env=env, check=True)
        subprocess.run(["git", "add", "-A"], cwd=ROOT, env=env, check=True)
        tree = subprocess.check_output(["git", "write-tree"], cwd=ROOT, env=env, text=True).strip()
    committed = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    return {"revision": revision, "tree": tree, "dirty": tree != committed}


def snapshot(tree: str, destination: Path) -> None:
    """Materialize only Git source, without checkout history or LFS downloads."""
    destination.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "archive", tree], cwd=ROOT, stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as archive:
            archive.extractall(destination, filter="data")


@contextmanager
def device_lease():
    """Serialize the whole local pipeline across CI runs sharing accelerators."""
    lock = Path("/tmp/llaminar-production-ci.lock")
    with lock.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another production pipeline owns this node's device lease") from error
        yield


def require_full_runtime(image: dict, source: dict, isa: str) -> None:
    """Never certify an Integration binary, backend subset or mislabeled source."""
    expected = {"org.opencontainers.image.revision": source["revision"],
                "org.llaminar.source_tree": source["tree"], "org.llaminar.build_type": "Release",
                "org.llaminar.cpu_isa": isa, "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON"}
    if any(image["labels"].get(key) != value for key, value in expected.items()):
        raise ValueError("candidate is not the exact full-backend Release build admitted by this run")


def build(args, source: dict, directory: Path) -> dict:
    """Build test/runtime siblings from the same frozen full-fat Docker context."""
    context = directory / "source"
    if not context.exists():
        snapshot(source["tree"], context)
    prefix = f"llaminar-ci:{source['tree'][:16]}-{args.cpu_isa.lower()}"
    images = {}
    for target in ("builder", "runtime"):
        tag = prefix + "-" + target
        command = ["docker", "buildx", "build", "--load", "--network=host", "--progress=plain",
                   "--target", target, "-t", tag, "--build-arg", f"VCS_REF={source['revision']}",
                   "--build-arg", f"LLAMINAR_SOURCE_TREE={source['tree']}",
                   "--build-arg", f"LLAMINAR_TEST_UID={os.getuid()}",
                   "--build-arg", f"LLAMINAR_TEST_GID={os.getgid()}",
                   "--build-arg", f"LLAMINAR_CPU_ISA={args.cpu_isa}", str(context)]
        run(command, directory / f"build-{target}.log")
        images[target] = {"tag": tag, **image_identity(tag)}
    require_full_runtime(images["runtime"], source, args.cpu_isa)
    return images


def logged_container_command(command: list[str], log: str) -> list[str]:
    """Persist process output inside the container before Docker transports it.

    Docker console delivery is not the evidence authority. tee retains early
    Python/admission failures as well as CTest output in the mounted results
    directory; pipefail preserves both the child and log-writer exit status.
    Positional arguments keep arbitrary command arguments out of shell source.
    """
    return ["bash", "-o", "pipefail", "-c", '"${@:2}" 2>&1 | tee "$1"',
            "llaminar-ci", log, *command]


def run_builder(images: dict, command: list[str], args, directory: Path, log: str) -> None:
    """Run installed tests; bind data/results only, never source or binaries."""
    image = images["builder"]["id"]
    name = "llaminar-ci-tests-" + uuid.uuid4().hex
    launch = ["docker", "run", "--rm", "--name", name,
              *docker_paths.device_args(image, "CPU+CUDA+ROCm", user=f"{os.getuid()}:{os.getgid()}"),
              *(part for group in os.getgroups() for part in ("--group-add", str(group))),
              *docker_paths.mounts([(args.models, "/src/models", False),
                                    (args.models, "/opt/llaminar-models", False),
                                    (args.model_ramdisk_root, str(args.model_ramdisk_root), False),
                                    (args.reference_cache_root, "/reference-cache", False),
                                    (directory, "/ci-results", False)]),
              "-e", "LLAMINAR_PARITY_REFERENCE_CACHE_ROOT=/reference-cache",
              "-w", "/src", image,
              *logged_container_command(command, f"/ci-results/{Path(log).stem}.process.log")]
    try:
        run(launch, directory / log)
    finally:
        subprocess.run(["docker", "rm", "-f", name], check=False, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def remap_manifest(document: dict, models: Path) -> dict:
    """Translate paths at the container boundary without inventing configurations."""
    result = json.loads(json.dumps(document))
    def translate(path: str) -> str:
        return str(models / Path(path).relative_to("/src/models"))
    for row in result["cells"]:
        row["configuration"]["model"] = translate(row["configuration"]["model"])
        row["model_files"] = [translate(path) for path in row["model_files"]]
    return result


def validate_parity(report: dict) -> None:
    """Require both numerical evidence and the campaign's whole-run economy gate."""
    for field in ("correctness_passed", "performance_requirements_met", "artifact_contract_passed"):
        if report.get(field) is not True:
            raise ValueError(f"production parity gate is red: {field}")
    prerequisites = report.get("preflight_tests", [])
    if (report.get("preflight_return_code") != 0 or report.get("preflight_test_count") != len(prerequisites)
            or not any(name.startswith("V2_Unit_") for name in prerequisites)
            or not any(name.startswith("V2_Integration_") for name in prerequisites)
            or not report.get("exact_matrix_cell_count")):
        raise ValueError("production parity omitted Unit/preflight or model cells")


def model_identities(manifest: dict) -> dict:
    """Pin every source shard by stat identity without reading/hashing weights."""
    result = {}
    for filename in sorted({path for row in manifest["cells"] for path in row["model_files"]}):
        stat = Path(filename).stat()
        result[filename] = [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    return result


def validate_phase_prefix(phases: dict) -> None:
    """A resumable receipt may contain only a contiguous successful prefix."""
    names = [phase.value for phase in Phase]
    if set(phases) != set(names[:len(phases)]):
        raise ValueError("pipeline receipt has an impossible phase transition")


def certificates(source: dict, images: dict, directory: Path) -> dict:
    """Join complete evidence before making the irreversible certified transition."""
    manifest = json.loads((directory / "manifest.json").read_text())
    cells = validate_manifest(manifest, source["revision"])
    parity = json.loads((directory / "parity.json").read_text())
    validate_parity(parity)
    e2e = json.loads((directory / "e2e.json").read_text())
    validate_image_e2e(e2e, manifest, images["runtime"]["id"])
    benchmarks = json.loads((directory / "benchmarks.json").read_text())
    expected = {row["case"] for row in cells}
    if (benchmarks.get("passed") is not True or benchmarks.get("complete") is not True
            or benchmarks.get("diagnostic") is not False
            or benchmarks.get("e2e_report_digest") != digest(e2e)
            or benchmarks.get("image") != images["runtime"]["id"]
            or benchmarks.get("manifest_digest") != digest(manifest)
            or len(benchmarks.get("cells", [])) != len(expected)
            or {row["case"] for row in benchmarks["cells"]} != expected):
        raise ValueError("benchmark certificate is incomplete, stale or regressed")
    return {"schema": 1, "source": source, "tested_image": images["runtime"]["id"],
            "cpu_isa": images["runtime"]["labels"]["org.llaminar.cpu_isa"],
            "test_image": images["builder"]["id"], "manifest_digest": digest(manifest),
            "parity": {"report_digest": digest(parity), "exact_cells": parity["exact_matrix_cell_count"],
                       "prerequisite_tests": parity["preflight_test_count"]},
            "e2e": {"report_digest": digest(e2e), "cells": sorted(expected)},
            "benchmarks": {"report_digest": digest(benchmarks), "cells": sorted(expected)},
            "certified": True}


def validate_certificate_changes(changes: str) -> None:
    """Allow certificate files/directories only, never overwritten runtime bits."""
    parent = "/usr/local/share/llaminar/certificates"
    allowed = {"/usr", "/usr/local", "/usr/local/share", "/usr/local/share/llaminar", parent,
               *(parent + "/" + name + ".json" for name in ("production", "e2e", "benchmarks"))}
    for line in changes.splitlines():
        action, path = line.split(" ", 1)
        if action not in ("A", "C") or path not in allowed:
            raise ValueError(f"certificate packaging changed runtime data: {line}")


def certify(args, source: dict, images: dict, directory: Path) -> dict:
    """Add evidence to an unstarted candidate container and inspect its delta.

    Parent-layer equality alone is insufficient: a new layer can overwrite a
    tested library. Docker's actual container diff must contain only the three
    certificate files and parent directories. No shell or runtime executes in
    this container, and the exact tested candidate ID is its immutable parent.
    """
    certificate = certificates(source, images, directory)
    candidate = images["runtime"]
    tag = args.image or f"llaminar-certified:{source['tree'][:16]}-{args.cpu_isa.lower()}"
    with tempfile.TemporaryDirectory(prefix="certificate-layer-", dir=directory) as temporary:
        payload = Path(temporary) / "llaminar" / "certificates"
        write_json(payload / "production.json", certificate)
        for name in ("e2e", "benchmarks"):
            shutil.copyfile(directory / f"{name}.json", payload / f"{name}.json")
        container = subprocess.check_output(["docker", "create", candidate["id"]], text=True).strip()
        try:
            run(["docker", "cp", str(payload.parent), container + ":/usr/local/share/"], directory / "certify.log")
            validate_certificate_changes(subprocess.check_output(["docker", "diff", container], text=True))
            run(["docker", "commit", "--change", f"LABEL org.llaminar.production_certificate={digest(certificate)}",
                 container, tag], directory / "certify.log")
        finally:
            subprocess.run(["docker", "rm", "-v", container], check=True, timeout=30,
                           stdout=subprocess.DEVNULL)
    final = image_identity(tag)
    if final["layers"][:len(candidate["layers"])] != candidate["layers"]:
        raise ValueError("certified image did not retain the tested runtime layers")
    installed = json.loads(subprocess.check_output(["docker", "run", "--rm", "--entrypoint", "/bin/cat",
                          final["id"], "/usr/local/share/llaminar/certificates/production.json"], text=True))
    if installed != certificate:
        raise ValueError("certified image does not contain its exact certificate")
    return {"tag": tag, **final, "certificate": certificate}


def publication_payloads(source: dict, directory: Path, finals: dict, baseline: dict) -> dict:
    """Join both certified ISA results against one unchanged baseline authority.

    Each runner starts from the same checked-in baseline. Merging its entire
    proposed file would overwrite the other ISA's improvements, so recompute
    the upward-only proposal once from the disjoint measured identities.
    """
    if set(finals) != set(SHIPPING_ISAS):
        raise ValueError("publication requires both AVX512 and AVX2 certificates")
    payloads = {}
    measurements = []
    for isa in SHIPPING_ISAS:
        path = directory / isa.lower()
        final = finals[isa]
        state = json.loads((path / "pipeline.json").read_text())
        verified = certificates(source, state["images"], path)
        if (final["certificate"] != verified or verified["cpu_isa"] != isa
                or state.get("certified") is not True or state.get("final") != final):
            raise ValueError(f"stale or wrong-ISA certificate: {isa}")
        benchmark = json.loads((path / "benchmarks.json").read_text())
        if benchmark["baseline_digest"] != digest(baseline):
            raise ValueError("high-water marks changed during certification")
        if any(row["identity"]["cpu_isa"] != isa for row in benchmark["cells"]):
            raise ValueError(f"benchmark ISA identity does not match its certificate: {isa}")
        measurements.extend(benchmark["cells"])
        payloads[f"benchmarks/production/results/{source['revision']}-{isa.lower()}.json"] = {
            "image": final["tag"], "image_id": final["id"],
            "certificate": verified, "benchmark": benchmark}
    proposed, comparisons = ratchet(baseline, measurements)
    if not all(row["passed"] for row in comparisons):
        raise ValueError("an ISA benchmark regressed; neither image may be published")
    payloads["benchmarks/production/high_water.json"] = proposed
    return payloads


def publish(source: dict, directory: Path, finals: dict) -> None:
    """Official CI only: commit scoped JSON with an ordinary fast-forward push.

    A result commit records the tested source revision; it is not claimed as a
    newly tested binary revision. No force push, baseline decrease, automatic
    merge or retry against changed source is allowed. Local runs never enter
    this function unless explicitly asked, and still require official context.
    """
    branch = os.environ.get("GITHUB_REF", "")
    if (source["dirty"] or os.environ.get("GITHUB_ACTIONS") != "true"
            or os.environ.get("GITHUB_EVENT_NAME") not in ("push", "workflow_dispatch")
            or branch not in ("refs/heads/develop", "refs/heads/master")
            or os.environ.get("GITHUB_SHA") != source["revision"]):
        raise ValueError("result commit/publication requires an official clean protected-branch CI run")
    remote_head = subprocess.check_output(["git", "ls-remote", "origin", branch], cwd=ROOT, text=True).split()[0]
    if remote_head != source["revision"]:
        raise ValueError("branch advanced during certification; refusing stale result commit")
    baseline_path = ROOT / "benchmarks/production/high_water.json"
    payloads = publication_payloads(source, directory, finals, json.loads(baseline_path.read_text()))
    # Use a private index and commit-tree, leaving the caller's branch/index
    # untouched. Both ISA result records and the combined ratchet land together.
    with tempfile.TemporaryDirectory(prefix="llaminar-ci-publish-") as temporary:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(temporary) / "index"),
               "GIT_AUTHOR_NAME": "github-actions[bot]", "GIT_COMMITTER_NAME": "github-actions[bot]",
               "GIT_AUTHOR_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com",
               "GIT_COMMITTER_EMAIL": "41898282+github-actions[bot]@users.noreply.github.com"}
        subprocess.run(["git", "read-tree", source["revision"]], cwd=ROOT, env=env, check=True)
        for path, value in payloads.items():
            blob = subprocess.check_output(["git", "hash-object", "-w", "--stdin"], cwd=ROOT,
                                           input=json.dumps(value, indent=2, allow_nan=False) + "\n", text=True).strip()
            subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}"],
                           cwd=ROOT, env=env, check=True)
        tree = subprocess.check_output(["git", "write-tree"], cwd=ROOT, env=env, text=True).strip()
        commit = subprocess.check_output(["git", "commit-tree", tree, "-p", source["revision"],
                                          "-m", f"ci: certify {source['revision'][:12]} [skip ci]"],
                                         cwd=ROOT, env=env, text=True).strip()
        # Git and a registry cannot share an atomic transaction. Publish the
        # immutable artifact first: a registry failure must not advance the
        # source branch or ratchet. A later branch race can leave an unreferenced
        # certified image, but never a checked-in claim for a missing image.
        for final in finals.values():
            if image_identity(final["tag"])["id"] != final["id"]:
                raise ValueError("certified image tag changed before publication")
        for isa in SHIPPING_ISAS:
            run(["docker", "push", finals[isa]["tag"]], directory / "publish.log")
        run(["git", "push", "origin", f"{commit}:{branch}"], directory / "publish.log", cwd=ROOT)
    write_json(directory / "publication.json", {"result_commit": commit,
        "images": {isa: {"image": final["tag"], "image_id": final["id"]} for isa, final in finals.items()}})


def run_variant(args, source: dict) -> dict:
    """Advance one ISA prefix; the outer driver owns the shared device lease."""
    args.models = args.models.resolve(strict=True)
    args.model_ramdisk_root = args.model_ramdisk_root.resolve(strict=True)
    args.reference_cache_root.mkdir(parents=True, exist_ok=True)
    args.reference_cache_root = args.reference_cache_root.resolve(strict=True)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "pipeline.json"
    identity = {"source": source, "cpu_isa": args.cpu_isa, "models": str(args.models),
                "test_user": [os.getuid(), os.getgid()],
                "ramdisk": str(args.model_ramdisk_root), "image_tag": args.image,
                "reference_cache_root": str(args.reference_cache_root)}
    state = {"schema": 1, "identity": identity, "phases": {}, "certified": False}
    if args.resume and receipt_path.exists():
        state = json.loads(receipt_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("source/build inputs changed; start a new output directory")
        validate_phase_prefix(state["phases"])
    elif receipt_path.exists():
        raise ValueError("output already has a run; use --resume or a new directory")
    for phase in Phase:
        if source_identity() != source:
            raise ValueError("source changed after admission; no further certificate transitions allowed")
        if "model_sources" in state:
            current = model_identities(json.loads((directory / "manifest.json").read_text()))
            if state["model_sources"] != current:
                raise ValueError("model files changed after numerical certification")
        if phase.value in state["phases"]:
            evidence = state["phases"][phase.value]
            for name, checksum in evidence["files"].items():
                if digest(json.loads((directory / name).read_text())) != checksum:
                    raise ValueError(f"changed/missing {phase.value} evidence: {name}")
            if phase == Phase.BUILD:
                for item in state["images"].values():
                    if image_identity(item["id"])["id"] != item["id"]:
                        raise ValueError("installed build image changed")
            print(f"[production-ci] REUSE {phase.value} (identity/evidence verified)", flush=True)
        else:
            started = time.monotonic()
            print(f"[production-ci] RUN {phase.value}", flush=True)
            write_json(receipt_path, state | {"running": phase.value})
            files = []
            if phase == Phase.BUILD:
                state["images"] = build(args, source, directory)
                # Admit the canonical model files before any model test,
                # not after numerical parity has already consumed them.
                # This also makes a build-only run sufficient to discover
                # exact candidates for standalone diagnostic exercises.
                run_builder(state["images"], ["python3", "scripts/ci/run_model_parity_e2e.py",
                    "--build-dir", "build_v2_integration", "--export-manifest", "/ci-results/container-manifest.json",
                    "--source-revision", source["revision"]], args, directory, "discovery.log")
                write_json(directory / "manifest.json", remap_manifest(
                    json.loads((directory / "container-manifest.json").read_text()), args.models))
                state["model_sources"] = model_identities(json.loads((directory / "manifest.json").read_text()))
                files = ["manifest.json"]
            elif phase == Phase.PARITY:
                run_builder(state["images"], ["python3", "scripts/ci/run_production_parity_campaigns.py",
                    "--build-dir", "build_v2_integration", "--installed-build-receipt", "/src/installed-tests.json",
                    "--model-ramdisk-root", str(args.model_ramdisk_root), "--persistent-model-cache-dir", "cache",
                    "--report", "/ci-results/parity.json"], args, directory, "parity.log")
                validate_parity(json.loads((directory / "parity.json").read_text()))
                files = ["parity.json"]
            elif phase == Phase.E2E:
                run([sys.executable, str(directory / "source/scripts/ci/run_model_parity_e2e.py"),
                     "--manifest", str(directory / "manifest.json"), "--source-revision", source["revision"],
                     "--container-image", state["images"]["runtime"]["id"],
                     "--model-ramdisk-root", str(args.model_ramdisk_root),
                     "--report", str(directory / "e2e.json")], directory / "e2e.log", cwd=ROOT)
                validate_image_e2e(json.loads((directory / "e2e.json").read_text()),
                    json.loads((directory / "manifest.json").read_text()), state["images"]["runtime"]["id"])
                files = ["e2e.json"]
            elif phase == Phase.BENCHMARKS:
                run([sys.executable, str(directory / "source/scripts/ci/run_model_parity_benchmarks.py"),
                     "--manifest", str(directory / "manifest.json"), "--source-revision", source["revision"],
                     "--image", state["images"]["runtime"]["id"],
                     "--e2e-report", str(directory / "e2e.json"),
                     "--model-ramdisk-root", str(args.model_ramdisk_root),
                     "--report", str(directory / "benchmarks.json")], directory / "benchmarks.log", cwd=ROOT)
                files = ["benchmarks.json"]
            elif phase == Phase.CERTIFY:
                state["final"] = certify(args, source, state["images"], directory)
                state["certified"] = True
            state["phases"][phase.value] = {"elapsed_seconds": time.monotonic() - started,
                "files": {name: digest(json.loads((directory / name).read_text())) for name in files}}
            write_json(receipt_path, state)
            print(f"[production-ci] PASS {phase.value}", flush=True)
        if phase.value == args.through:
            break
    if state["certified"]:
        if args.through == "certify":
            if image_identity(state["final"]["id"])["id"] != state["final"]["id"]:
                raise ValueError("certified image is no longer installed")
            print(f"[production-ci] CERTIFIED {state['final']['tag']} ({state['final']['id']})", flush=True)
        else:
            print("[production-ci] requested prefix verified; no new certificate issued", flush=True)
    else:
        print("[production-ci] partial run complete; no image certificate issued", flush=True)
    return state


def variant_arguments(args, isa: str, through: Phase) -> argparse.Namespace:
    """Keep source/workload common while isolating each ISA's image and evidence."""
    return argparse.Namespace(**{**vars(args), "cpu_isa": isa,
        "output": args.output / isa.lower(), "through": through.value,
        "resume": args.resume or through != Phase.BUILD,
        "image": (args.image + ("-avx2" if isa == "AVX2" else "")) if args.image else None})


def drive_variants(args, source: dict) -> dict:
    """Finish a gate for every ISA before entering the next gate for any ISA.

    In particular, both full E2E suites precede the first benchmark. Runs are
    sequential on this node so neither inference nor timing competes for its
    devices. Only immutable dependencies/reference packs are shared, never
    correctness results or image-bound certificates.
    """
    states = {}
    for phase in Phase:
        for isa in args.cpu_isas:
            print(f"[production-ci] ISA={isa} gate={phase.value}", flush=True)
            states[isa] = run_variant(variant_arguments(args, isa, phase), source)
        if phase.value == args.through:
            break
    return states


def main(argv: list[str] | None = None) -> int:
    """Certify both shipping ISAs by default and publish their results together."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("/opt/llaminar-models"))
    parser.add_argument("--model-ramdisk-root", type=Path, default=Path("/mnt/llaminar-production-parity"))
    parser.add_argument("--reference-cache-root", type=Path, default=ROOT,
                        help="persistent HF pack parent; defaults to the existing workspace packs")
    parser.add_argument("--cpu-isa", choices=SHIPPING_ISAS, action="append", dest="cpu_isas",
                        help="local piecewise ISA selection; default and official publication require both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", choices=[phase.value for phase in Phase], default="certify")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image", help="AVX512 certified image tag; AVX2 appends -avx2")
    parser.add_argument("--publish", action="store_true", help="official CI only: publish both certified images and one result commit")
    args = parser.parse_args(argv)
    requested = args.cpu_isas or list(SHIPPING_ISAS)
    if len(set(requested)) != len(requested):
        parser.error("duplicate CPU ISA selection")
    args.cpu_isas = [isa for isa in SHIPPING_ISAS if isa in requested]
    if args.publish and (args.through != "certify" or not args.image
                         or set(args.cpu_isas) != set(SHIPPING_ISAS)):
        parser.error("--publish requires both ISA pipelines, --through certify and an explicit image tag")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    source = source_identity()
    # Bind the collection itself as well as its per-ISA receipts. This prevents
    # resuming a different shipping set in a partially certified directory.
    identity = {"source": source, "cpu_isas": args.cpu_isas, "image": args.image,
                "models": str(args.models.resolve()), "ramdisk": str(args.model_ramdisk_root.resolve()),
                "reference_cache_root": str(args.reference_cache_root.resolve()),
                "test_user": [os.getuid(), os.getgid()]}
    receipt = args.output / "pipeline.json"
    with device_lease():
        if args.resume:
            prior = json.loads(receipt.read_text())
            if prior.get("schema") != 2 or prior.get("identity") != identity:
                raise ValueError("source/ISA collection changed; start a new output directory")
        elif receipt.exists():
            raise ValueError("output already has a run; use --resume or a new directory")
        state = {"schema": 2, "identity": identity, "certified": False,
                 "variants": {isa: f"{isa.lower()}/pipeline.json" for isa in args.cpu_isas}}
        write_json(receipt, state)
        docker_paths.publish_model_cache(args.model_ramdisk_root)
        variants = drive_variants(args, source)
        state["certified"] = (args.through == "certify" and all(v["certified"] for v in variants.values()))
        state["shipping_set_complete"] = state["certified"] and set(variants) == set(SHIPPING_ISAS)
        write_json(receipt, state)
        if args.publish:
            if not state["shipping_set_complete"]:
                raise ValueError("both shipping images must certify before any publication")
            publish(source, args.output, {isa: v["final"] for isa, v in variants.items()})
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"[production-ci] ERROR: {error}", file=sys.stderr)
        sys.exit(1)
