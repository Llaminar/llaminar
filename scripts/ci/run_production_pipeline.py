#!/usr/bin/env python3
"""Build, certify and optionally publish AVX512 and AVX2 full runtime images.

Local and GitHub runs execute the same ordered pipeline. Unit and production
preflight precede approved serial/dynamic-MTP HTTP token regression; this driver then
runs all tagged HTTP E2E cells, the separately declared cross-host MPI E2E
projection (with owned Azure lease retirement), and exactly those benchmarks on
a pinned Release image. Only the final transition embeds certificates. Official
publication commits compact JSON/high-water proposals, never corpora or
diagnostic dumps.
The installed matrix exports one full inventory before model admission. Its
complete shard set owns model identity pins; E2E and benchmark eligibility are
an exact projection, never a separately discovered or configurable matrix.
Container model paths are translated only inside their declared mount; aliases
and parent traversal are rejected before any source shard is admitted.

--through supports piecewise exercise. --resume reuses only identity-bound,
digest-checked completed phases; modifying source invalidates the entire run.
No selector or skip-gate option exists on this certifying entry point.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
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
from azure_cross_host_resources import DEFAULT_TUNNEL_SUBNET
from model_parity_inventory import InventoryScope, cross_host_scenarios
from generation_corpus import ApprovedGenerationCorpus
from run_model_parity_generation import validate_regression_report
from production_artifacts import digest, image_identity, model_identities, ratchet, validate_image_e2e, validate_manifest, validate_prerequisites, write_json

ROOT = Path(__file__).resolve().parents[2]
SHIPPING_ISAS = ("AVX512", "AVX2")
# A self-hosted ARC runner mounts this directory from the physical host.  It
# is deliberately optional so normal developer builds continue to use the
# host Docker daemon's local BuildKit cache without inventing a second cache
# topology.  When declared, the path is a persistent, bounded external cache
# for immutable Docker layers; compiler objects themselves remain in the
# persistent named BuildKit worker's cache mount (cache mounts cannot be
# exported).
DOCKER_BUILD_CACHE_ROOT_ENV = "LLAMINAR_DOCKER_BUILD_CACHE_ROOT"
# Docker build input must not include diagnostic parity CSVs.  Docker itself
# already excludes this generated directory through .dockerignore; excluding it
# during the preceding Git archive avoids materializing almost 0.9 GB per ISA
# only for BuildKit to discard it.  Keep this narrow pathspec synchronized with
# the corresponding .dockerignore entry and protect that coupling by Unit test.
DOCKER_CONTEXT_ARCHIVE_PATHS = (".", ":(exclude)tests/v2/integration/parity/results/**")


class Phase(str, Enum):
    """Only successful ordered transitions can reach certification."""
    BUILD = "build"
    PREREQUISITES = "prerequisites"
    GENERATION = "generation"
    PARITY = "parity"
    E2E = "e2e"
    CROSS_HOST_E2E = "cross-host-e2e"
    BENCHMARKS = "benchmarks"
    CERTIFY = "certify"


def pipeline_phases(mathematical_parity: bool = False) -> tuple[Phase, ...]:
    """Mathematical HF diagnostics are never an implicit routine dependency."""
    return tuple(phase for phase in Phase if phase is not Phase.PARITY or mathematical_parity)


class ImageRole(str, Enum):
    """Installed test runner and shipped runtime have distinct admission roles."""
    TEST_RUNNER = "test-runner"
    RUNTIME = "runtime"


class TestRunnerInventory(str, Enum):
    """Explicit compiled-test inventories for otherwise identical test runners.

    The full production pipeline needs the typed matrix executables for
    generation and HTTP E2E discovery. The develop image gate instead builds
    and runs its complete model-free Unit/preflight contract without compiling
    diagnostic mathematical-parity matrices that it cannot run.
    """
    FULL_MATRIX = "full-matrix"
    MODEL_FREE = "model-free"

    @property
    def docker_matrix_argument(self) -> str:
        """Translate the typed inventory to the Dockerfile's checked argument."""
        return "ON" if self is TestRunnerInventory.FULL_MATRIX else "OFF"


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


def append_build_timeline_event(path: Path, event: str, target: ImageRole, log: Path,
                                started: float, **details: object) -> None:
    """Append one durable, timestamped transition for a Docker image target.

    Buildx deliberately keeps its full plain-progress output in ``log``.  This
    compact companion journal answers the separate operational question of
    which top-level target consumed wall time, including local image import.
    It is appended before starting a target and immediately after its terminal
    transition, so an interrupted job still leaves useful evidence rather than
    an unexplained truncated build log.

    Args:
        path: Per-ISA JSON-lines journal owned by the image build transaction.
        event: Typed lifecycle transition: ``started``, ``completed`` or
            ``failed``.
        target: The Docker target whose lifecycle is being recorded.
        log: Full Buildx output associated with this target.
        started: Monotonic timestamp captured immediately before Buildx launch.
        **details: Terminal identity or failure metadata for the event.
    """
    if event not in {"started", "completed", "failed"}:
        raise ValueError(f"unsupported build timeline event: {event}")
    if not isinstance(target, ImageRole):
        raise TypeError("build timeline target must be an ImageRole")
    record = {
        "schema": 1,
        "event": event,
        "target": target.value,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "log": log.name,
        **details,
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        # There are only two transitions per target, so force the compact
        # lifecycle journal through the host cache before continuing.  Full
        # compiler output remains streaming/flush-only in its much larger log.
        os.fsync(stream.fileno())


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
    """Materialize the exact non-diagnostic Docker source without Git history.

    The source identity remains the complete Git tree, including all tracked
    evidence.  The Docker context is intentionally narrower: diagnostic
    parity-result CSVs are not build inputs under ``.dockerignore`` and are
    excluded before extracting the archive.  This preserves build identity
    while avoiding per-ISA archive I/O that cannot affect a compiled image.
    """
    destination.mkdir()
    with tempfile.TemporaryFile() as stream:
        subprocess.run(["git", "archive", tree, "--", *DOCKER_CONTEXT_ARCHIVE_PATHS],
                       cwd=ROOT, stdout=stream, check=True)
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


def require_image(image: dict, source: dict, isa: str, role: ImageRole, *,
                  test_inventory: TestRunnerInventory = TestRunnerInventory.FULL_MATRIX) -> None:
    """Authenticate either image against caller-owned source, ISA and role.

    Test-runner labels describe their embedded Release build and the unskipped
    Integration installation. The installed-test receipt independently checks
    real test files; labels neither replace that receipt nor waive execution.
    Never infer the expected ISA from the image being checked.
    """
    if not isinstance(role, ImageRole):
        raise TypeError("image admission requires a typed role")
    if not isinstance(test_inventory, TestRunnerInventory):
        raise TypeError("test-runner image admission requires a TestRunnerInventory")
    if isa not in SHIPPING_ISAS:
        raise ValueError("image admission requires an explicit shipping CPU ISA")
    expected = {"org.opencontainers.image.revision": source["revision"],
                "org.llaminar.source_tree": source["tree"], "org.llaminar.build_type": "Release",
                "org.llaminar.cpu_isa": isa, "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON",
                "org.llaminar.image_role": role.value}
    if role is ImageRole.TEST_RUNNER:
        expected["org.llaminar.integration_skipped"] = "0"
        expected["org.llaminar.test_runner_inventory"] = test_inventory.value
    labels = image.get("labels") if isinstance(image, dict) else None
    if not isinstance(labels, dict) or any(labels.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{role.value} image does not match the admitted source/ISA/full-backend build")


def require_image_pair(images: dict, source: dict, isa: str, *,
                       test_inventory: TestRunnerInventory = TestRunnerInventory.FULL_MATRIX) -> None:
    """Reject swapped, incomplete or cross-ISA test/runtime siblings."""
    if not isinstance(images, dict) or set(images) != {role.value for role in ImageRole}:
        raise ValueError("certification requires exactly one test-runner/runtime image pair")
    for role in ImageRole:
        require_image(images[role.value], source, isa, role, test_inventory=test_inventory)
    identities = [images[role.value].get("id") for role in ImageRole]
    if any(not isinstance(identity, str) or not identity for identity in identities) or identities[0] == identities[1]:
        raise ValueError("test-runner/runtime roles require distinct immutable image identities")


def persistent_build_cache_arguments(cpu_isa: str) -> list[str]:
    """Return one isolated durable Buildx cache contract for a shipping lane.

    The runner process is ephemeral, whereas its configured hostPath survives
    pod replacement. A cache is therefore scoped by ISA; BuildKit's own graph
    keys retain the target/stage distinction within that slot. The cache
    directory is an OCI layout owned by the runner, not a second Docker graph
    store. ``reset=true`` bounds each slot to the most recent complete cache
    manifest instead of silently accumulating superseded blobs forever.

    BuildKit intentionally does not export ``RUN --mount=type=cache`` data.
    The Dockerfile names that ccache mount separately and the persistent
    BuildKit worker hosted by the one host Docker daemon retains it. This
    external cache complements it by retaining layer hits across runner-pod
    lifecycle transitions.
    """
    if cpu_isa not in SHIPPING_ISAS:
        raise ValueError("persistent Buildx cache requires an explicit shipping CPU ISA")
    raw_root = os.environ.get(DOCKER_BUILD_CACHE_ROOT_ENV)
    if raw_root is None:
        return []
    if not raw_root:
        raise ValueError(f"{DOCKER_BUILD_CACHE_ROOT_ENV} must not be empty when declared")
    root = Path(raw_root)
    if not root.is_absolute():
        raise ValueError(f"{DOCKER_BUILD_CACHE_ROOT_ENV} must be an absolute directory")
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(
            f"{DOCKER_BUILD_CACHE_ROOT_ENV} must name the pre-mounted persistent host directory"
        ) from error
    if not root.is_dir():
        raise ValueError(f"{DOCKER_BUILD_CACHE_ROOT_ENV} is not a directory: {root}")
    if not os.access(root, os.W_OK | os.X_OK):
        raise ValueError(f"{DOCKER_BUILD_CACHE_ROOT_ENV} is not writable by the CI runner: {root}")
    if os.environ.get(docker_paths.SHARED_DAEMON_ROOTS_ENV) is not None and \
            docker_paths.shared_daemon_path(root) is None:
        raise ValueError(
            f"{DOCKER_BUILD_CACHE_ROOT_ENV} is not declared in "
            f"{docker_paths.SHARED_DAEMON_ROOTS_ENV}"
        )
    slot = root / cpu_isa.lower()
    slot.mkdir(parents=True, exist_ok=True)
    cache_to = f"type=local,dest={slot},mode=max,reset=true"
    arguments = ["--cache-to", cache_to]
    # An empty newly-created directory has no OCI index.  Supplying it as an
    # import source turns an ordinary cold start into a Buildx warning/error;
    # export first, then admit it as an input on following runs.
    if (slot / "index.json").is_file():
        arguments[0:0] = ["--cache-from", f"type=local,src={slot}"]
    return arguments


def build(args, source: dict, directory: Path, *,
          test_inventory: TestRunnerInventory = TestRunnerInventory.FULL_MATRIX,
          roles: tuple[ImageRole, ...] = tuple(ImageRole)) -> dict:
    """Build test/runtime siblings and journal each target's wall-clock cost.

    A published-image workflow can request only the metadata/test companion;
    it must not rebuild or substitute the runtime that is being tested.
    Full certification still requires both siblings through require_image_pair.
    The full Buildx log remains one file per target.  The accompanying
    ``build-timeline.jsonl`` makes local layer import/export visible as part of
    the target duration, instead of attributing that time to compilation by
    omission.  It is CI evidence only and never a cache or input to admission.
    """
    if not isinstance(test_inventory, TestRunnerInventory):
        raise TypeError("image build requires a TestRunnerInventory")
    if (not roles or len(set(roles)) != len(roles)
            or any(not isinstance(role, ImageRole) for role in roles)):
        raise ValueError("image build requires distinct typed image roles")
    context = directory / "source"
    if not context.exists():
        snapshot(source["tree"], context)
    prefix = f"llaminar-ci:{source['tree'][:16]}-{args.cpu_isa.lower()}"
    timeline = directory / "build-timeline.jsonl"
    images = {}
    for role in roles:
        target = role.value
        tag = prefix + "-" + target
        log = directory / f"build-{target}.log"
        command = ["docker", "buildx", "build", "--load", "--network=host", "--progress=plain",
                   "--target", target, "-t", tag, "--build-arg", f"VCS_REF={source['revision']}",
                   "--build-arg", f"LLAMINAR_SOURCE_TREE={source['tree']}",
                   "--build-arg", "LLAMINAR_BUILD_MODEL_PARITY_MATRICES="
                   f"{test_inventory.docker_matrix_argument}",
                   "--build-arg", "LLAMINAR_TEST_RUNNER_INVENTORY="
                   f"{test_inventory.value}",
                   "--build-arg", f"LLAMINAR_TEST_UID={os.getuid()}",
                   "--build-arg", f"LLAMINAR_TEST_GID={os.getgid()}",
                   "--build-arg", f"LLAMINAR_CPU_ISA={args.cpu_isa}",
                   *persistent_build_cache_arguments(args.cpu_isa), str(context)]
        started = time.monotonic()
        append_build_timeline_event(timeline, "started", role, log, started)
        print(f"[production-ci] BUILD target={target} log={log} timeline={timeline}", flush=True)
        try:
            run(command, log)
        except BaseException as error:
            append_build_timeline_event(timeline, "failed", role, log, started,
                                        exception=type(error).__name__, message=str(error))
            raise
        images[target] = {"tag": tag, **image_identity(tag)}
        append_build_timeline_event(timeline, "completed", role, log, started,
                                    image=images[target]["id"])
        print(f"[production-ci] BUILD target={target} status=PASS "
              f"elapsed_seconds={time.monotonic() - started:.3f}", flush=True)
        # Reject a foreign test bundle immediately, before paying for the next
        # image or admitting any discovery/test process from the wrong build.
        require_image(images[target], source, args.cpu_isa, role,
                      test_inventory=test_inventory)
    if set(roles) == set(ImageRole):
        require_image_pair(images, source, args.cpu_isa, test_inventory=test_inventory)
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


def run_test_runner(images: dict, command: list[str], args, directory: Path, log: str) -> None:
    """Run installed tests; bind data/results only, never source or binaries."""
    image = images[ImageRole.TEST_RUNNER.value]["id"]
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
    """Translate only canonical paths inside the installed model mount.

    The caller supplies its already resolved, absolute destination mount.
    This operation is lexical: resolving container paths on the host would
    consult the wrong filesystem. Shard stat pins and reviewed numerical
    provenance independently authenticate the actual weights. Reject aliases
    rather than normalizing malformed source declarations into another model.
    Neither policy fields nor the input discovery document are mutated.
    """
    if not models.is_absolute() or ".." in models.parts or any(ch in str(models) for ch in "\x00\r\n"):
        raise ValueError("model translation requires an absolute resolved destination mount")
    result = json.loads(json.dumps(document))
    def translate(path: str) -> str:
        """Keep every relative shard component inside the admitted namespace."""
        if not isinstance(path, str) or any(ch in path for ch in "\x00\r\n"):
            raise ValueError("container model declaration requires a canonical path")
        source = Path(path)
        if source.as_posix() != path or ".." in source.parts:
            raise ValueError("container model declaration contains a noncanonical path")
        relative = source.relative_to("/src/models")
        if not relative.parts:
            raise ValueError("container model declaration names the mount, not a shard")
        # relative_to alone accepts '..': the canonical check above is what
        # prevents this join from escaping the caller's admitted model mount.
        return str(models / relative)
    for row in result["cells"]:
        row["configuration"]["model"] = translate(row["configuration"]["model"])
        row["model_files"] = [translate(path) for path in row["model_files"]]
    return result


def e2e_projection(inventory: dict, revision: str) -> dict:
    """Validate the full discovered inventory and copy its exact tagged rows.

    The image runs the canonical metadata exporter with ALL scope and no
    selectors. Its retained document, not an E2E count or a second model list,
    owns membership. Check untagged rows before projection so malformed model
    declarations cannot disappear from admission or subsequent resume checks.
    The returned JSON is independent storage; translating or annotating a
    consumer document cannot mutate its full-inventory parent.
    """
    if (not isinstance(inventory, dict) or type(inventory.get("schema")) is not int
            or inventory["schema"] != 1 or inventory.get("source_revision") != revision
            or inventory.get("scope") != InventoryScope.ALL.value):
        raise ValueError("production certification requires the revision-bound all-cell inventory")
    rows = inventory.get("cells")
    if not isinstance(rows, list) or not rows:
        raise ValueError("the full canonical inventory is empty or malformed")
    cases, identities = set(), set()
    for row in rows:
        if not isinstance(row, dict) or any(not isinstance(row.get(key), str) or not row[key]
                                           for key in ("case", "campaign", "backends")):
            raise ValueError("full inventory omitted a canonical cell identity")
        config, files = row.get("configuration"), row.get("model_files")
        if (not isinstance(config, dict) or type(config.get("model_parity_schema")) is not int
                or config["model_parity_schema"] != 1 or not isinstance(config.get("id"), str)
                or not config["id"] or not isinstance(files, list) or not files
                or any(not isinstance(path, str) or not path for path in files)
                or len(set(files)) != len(files) or config.get("model") not in files):
            raise ValueError("full inventory omitted a valid configuration or model-file declaration")
        InventoryScope.ALL.accepts(config)
        if row["case"] in cases or config["id"] in identities:
            raise ValueError("the full canonical inventory contains duplicate cell identities")
        cases.add(row["case"])
        identities.add(config["id"])
    projected = json.loads(json.dumps({**inventory, "scope": InventoryScope.E2E.value,
        "cells": [row for row in rows if InventoryScope.E2E.accepts(row["configuration"])]}))
    validate_manifest(projected, revision)
    return projected


def cross_host_e2e_projection(inventory: dict, revision: str) -> dict:
    """Freeze remote eligibility from the same installed full-cell inventory.

    The build phase may discover an empty projection without provisioning any
    resources. Actual remote-phase admission must require its selected cells;
    an empty document is not a passed remote certificate. Validate even untagged
    records so stale exporters cannot silently erase requested coverage.
    """
    e2e_projection(inventory, revision)  # Reuse complete inventory admission.
    selected, identities = [], set()
    for row in inventory["cells"]:
        scenarios = cross_host_scenarios(row["configuration"])
        for scenario in scenarios:
            if scenario["id"] in identities:
                raise ValueError("duplicate remote scenario identity across canonical source cells")
            identities.add(scenario["id"])
        if scenarios:
            selected.append(row)
    return json.loads(json.dumps({**inventory, "scope": InventoryScope.CROSS_HOST_E2E.value,
                                 "cells": selected}))


def validate_parity(report: dict, cells: list[dict]) -> None:
    """Require complete exact membership, numerical evidence and whole-run economy.

    A positive count alone cannot prove coverage. Join each completed aggregate
    and exact GoogleTest identity to the installed matrix before certification;
    duplicates, omitted cells and a same-sized unrelated selection all fail.
    """
    for field in ("correctness_passed", "performance_requirements_met", "artifact_contract_passed"):
        if report.get(field) is not True:
            raise ValueError(f"production parity gate is red: {field}")
    prerequisites = report.get("preflight_tests", [])
    if (report.get("preflight_return_code") != 0 or report.get("preflight_test_count") != len(prerequisites)
            or not any(name.startswith("V2_Unit_") for name in prerequisites)
            or not any(name.startswith("V2_Integration_") for name in prerequisites)
            or not report.get("exact_matrix_cell_count")):
        raise ValueError("production parity omitted Unit/preflight or model cells")
    expected = {(row["campaign"], row["case"]) for row in cells}
    observed = []
    campaigns = report.get("campaigns")
    if (type(report.get("exact_matrix_cell_count")) is not int
            or report["exact_matrix_cell_count"] != len(expected)
            or not isinstance(campaigns, list) or not campaigns):
        raise ValueError("production parity does not cover the full discovered inventory")
    for campaign in campaigns:
        if (not isinstance(campaign, dict) or type(campaign.get("return_code")) is not int
                or campaign["return_code"] != 0 or campaign.get("outcome") != "completed"
                or campaign.get("artifact_contract_passed") is not True
                or not isinstance(campaign.get("gtest_cases"), list) or not campaign["gtest_cases"]
                or not isinstance(campaign.get("campaign"), str)
                or any(not isinstance(case, str) for case in campaign["gtest_cases"])):
            raise ValueError("production parity contains incomplete aggregate evidence")
        observed.extend((campaign["campaign"], case) for case in campaign["gtest_cases"])
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("production parity exact membership differs from the full discovered inventory")


def validate_cross_host_report(report: dict, manifest: dict, image: str,
                               revision: str, remote_cpu_image: str | None = None) -> None:
    """Authenticate the remote MPI E2E result before benchmarks are admitted.

    The remote runner owns cloud allocation, transport and HTTP execution.  It
    publishes only compact per-scenario evidence here; this validator joins it
    to the already validated manifest and immutable controller/CPU-peer images. In
    particular, a green HTTP report, a VM that merely booted, or an unretired
    Azure lease cannot satisfy a declared remote scenario.
    """
    if not isinstance(manifest, dict) or manifest.get("scope") != InventoryScope.CROSS_HOST_E2E.value:
        raise ValueError("cross-host certification requires a cross-host manifest")
    expected = {
        scenario["id"]: scenario
        for row in manifest.get("cells", [])
        for scenario in cross_host_scenarios(row["configuration"])
    }
    if not expected:
        if (not isinstance(report, dict) or report.get("schema") != 1
                or report.get("eligible") is not False
                or report.get("complete") is not True
                or report.get("source_revision") != revision
                or report.get("image") != image
                or report.get("manifest_digest") != digest(manifest)
                or report.get("scenarios") != []
                or report.get("all_resources_retired") is not True):
            raise ValueError("cross-host E2E is not applicable but its report is malformed")
        return
    remote_image_mismatch = (remote_cpu_image is not None
                             and (not isinstance(remote_cpu_image, str) or not remote_cpu_image
                                  or not isinstance(report, dict)
                                  or report.get("remote_cpu_image") != remote_cpu_image))
    if (not isinstance(report, dict) or report.get("schema") != 1
            or report.get("eligible") is not True or report.get("complete") is not True
            or report.get("source_revision") != revision or report.get("image") != image
            or remote_image_mismatch
            or report.get("manifest_digest") != digest(manifest)
            or report.get("all_resources_retired") is not True
            or not isinstance(report.get("scenarios"), list)):
        raise ValueError("cross-host E2E report is incomplete or not bound to the tested image")
    observed = report["scenarios"]
    if len(observed) != len(expected) or {row.get("id") for row in observed} != set(expected):
        raise ValueError("cross-host E2E scenario membership differs from the canonical projection")
    seen = set()
    for row in observed:
        identity = row.get("id")
        if identity in seen:
            raise ValueError("cross-host E2E report repeats a scenario")
        seen.add(identity)
        source = expected[identity]
        if (row.get("return_code") != 0 or row.get("outcome") != "passed"
                or row.get("frontend") != source["frontend"]
                or row.get("topology") != source["topology"]
                or row.get("resource_retired") is not True
                or row.get("transport_proof") is not True
                or row.get("http_proof") is not True):
            raise ValueError(f"cross-host E2E scenario is incomplete: {identity}")


def validate_phase_prefix(phases: dict, mathematical_parity: bool = False) -> None:
    """A resumable receipt may contain only a contiguous successful prefix."""
    names = [phase.value for phase in pipeline_phases(mathematical_parity)]
    if set(phases) != set(names[:len(phases)]):
        raise ValueError("pipeline receipt has an impossible phase transition")


def phase_failure(phase: Phase, error: Exception) -> dict:
    """Describe one failed, otherwise resumable phase without claiming progress.

    The completed ``phases`` prefix remains the only reusable evidence.  This
    separate typed record names the first uncompleted phase and its ordinary
    Python exception, allowing a later ``--resume`` to retry exactly that
    phase.  It intentionally records no command line or environment, because
    those may contain credential paths owned by the cross-host runner.
    """
    message = str(error).strip() or type(error).__name__
    return {"schema": 1, "phase": phase.value,
            "exception": type(error).__name__, "message": message}


def validate_phase_failure(state: dict, mathematical_parity: bool = False) -> None:
    """Reject a failure record that could conceal a skipped certification gate."""
    failure = state.get("failure")
    if failure is None:
        return
    names = [phase.value for phase in pipeline_phases(mathematical_parity)]
    phases = state.get("phases")
    if (not isinstance(failure, dict) or set(failure) != {"schema", "phase", "exception", "message"}
            or failure.get("schema") != 1 or not isinstance(failure.get("phase"), str)
            or failure["phase"] not in names or not isinstance(failure.get("exception"), str)
            or not failure["exception"] or not isinstance(failure.get("message"), str)
            or not failure["message"] or not isinstance(phases, dict)
            or len(phases) >= len(names) or failure["phase"] != names[len(phases)]):
        raise ValueError("pipeline receipt has an impossible failed phase transition")


def recover_interrupted_phase(state: dict, mathematical_parity: bool = False) -> bool:
    """Convert a legacy in-progress marker into an explicit retryable failure.

    A power loss or SIGKILL cannot run the driver's exception handler.  Older
    receipts therefore contain only ``running`` and are ambiguous on resume.
    Treat that marker as an interrupted first-uncompleted phase, never as a
    pass or a reason to replay an earlier gate.  The caller persists the
    returned state before doing more work.
    """
    running = state.pop("running", None)
    if running is None:
        return False
    if "failure" in state or not isinstance(running, str):
        raise ValueError("pipeline receipt has contradictory interrupted phase state")
    names = [phase.value for phase in pipeline_phases(mathematical_parity)]
    phases = state.get("phases")
    if (running not in names or not isinstance(phases, dict) or len(phases) >= len(names)
            or running != names[len(phases)]):
        raise ValueError("pipeline receipt has an impossible interrupted phase transition")
    state["failure"] = {"schema": 1, "phase": running,
                        "exception": "InterruptedRun",
                        "message": "previous process ended before this phase published a terminal result"}
    return True


def persist_phase_failure(output: Path, phase: Phase, error: Exception,
                          mathematical_parity: bool = False) -> None:
    """Atomically retire one started phase when the outer driver catches it.

    The per-ISA receipt is written before executing a phase.  Keeping failure
    retirement in the shared ISA driver covers every phase implementation and
    avoids duplicated exception paths around Docker, HTTP, Azure and local
    process launches.  A malformed/unrelated receipt is left alone so the
    original operational exception remains the primary diagnostic.
    """
    receipt_path = output / "pipeline.json"
    try:
        state = json.loads(receipt_path.read_text())
        if not isinstance(state, dict) or state.get("running") != phase.value:
            return
        state.pop("running")
        state["failure"] = phase_failure(phase, error)
        validate_phase_prefix(state.get("phases"), mathematical_parity)
        validate_phase_failure(state, mathematical_parity)
        write_json(receipt_path, state)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        # Preserve the original gate failure.  A later resume will still reject
        # a malformed receipt instead of treating work as successfully reused.
        return


def generation_evidence(directory: Path, inventory: dict, image: str, cpu_isa: str,
                        corpus_root: Path) -> tuple[dict, dict]:
    """Reauthenticate source-pinned answers and complete live regression evidence."""
    corpus = ApprovedGenerationCorpus.load_reviewed(
        directory / "source", corpus_root, cpu_isa, inventory, model_identities(inventory))
    prerequisites = json.loads((directory / "prerequisites/prerequisites.json").read_text())
    report = json.loads((directory / "generation/report.json").read_text())
    validate_regression_report(report, inventory, prerequisites, image, cpu_isa, corpus.pin.document_digest)
    return prerequisites, report


def certificates(source: dict, images: dict, directory: Path, *, cpu_isa: str,
                 corpus_root: Path = ROOT / "corpora", mathematical_parity: bool = False) -> dict:
    """Join complete evidence before making the irreversible certified transition."""
    require_image_pair(images, source, cpu_isa)
    inventory = json.loads((directory / "all-cells.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest != e2e_projection(inventory, source["revision"]):
        raise ValueError("E2E/benchmark manifest differs from the full inventory's tagged projection")
    cells = validate_manifest(manifest, source["revision"])
    # Remote MPI is a separate E2E family.  Keep its projection and evidence
    # attached to the same full inventory, image and source identity; it must
    # never be inferred from the ordinary HTTP report.  A model family without
    # a declared cross-host topology is explicitly not applicable, whereas a
    # declared scenario must have a completed cloud/SSH/MPI proof.
    remote_manifest_path = directory / "cross-host-manifest.json"
    remote_report_path = directory / "cross-host-e2e.json"
    # Missing files are incomplete production evidence even when the remote
    # projection is empty. Test fixtures obey this same contract: the absence
    # of a manifest must never become a way to erase declared cloud coverage.
    if not remote_manifest_path.is_file():
        raise ValueError("cross-host phase has no canonical manifest")
    remote_manifest = json.loads(remote_manifest_path.read_text())
    if remote_manifest != cross_host_e2e_projection(inventory, source["revision"]):
        raise ValueError("cross-host manifest differs from the canonical full-inventory projection")
    if not remote_report_path.is_file():
        raise ValueError("cross-host phase has no evidence report")
    remote_report = json.loads(remote_report_path.read_text())
    validate_cross_host_report(remote_report, remote_manifest, images["runtime"]["id"],
                               source["revision"])
    prerequisites, generation = generation_evidence(directory, inventory, images["runtime"]["id"],
                                                    cpu_isa, corpus_root)
    diagnostic = None
    if mathematical_parity:
        diagnostic = json.loads((directory / "parity.json").read_text())
        validate_parity(diagnostic, inventory["cells"])
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
            "cpu_isa": cpu_isa,
            "test_image": images[ImageRole.TEST_RUNNER.value]["id"], "manifest_digest": digest(manifest),
            "inventory_digest": digest(inventory),
            "prerequisites": {"report_digest": digest(prerequisites),
                              "tests": prerequisites["preflight_test_count"]},
            "generation": {"report_digest": digest(generation), "exact_cells": generation["selected"],
                           "corpus_digest": generation["corpus_digest"]},
            "diagnostic_mathematical_parity": {"report_digest": digest(diagnostic)} if diagnostic else None,
            "e2e": {"report_digest": digest(e2e), "cells": sorted(expected)},
            "cross_host_e2e": {"report_digest": digest(remote_report),
                               "manifest_digest": digest(remote_manifest),
                               "cells": sorted(scenario["id"] for row in remote_manifest["cells"]
                                                for scenario in cross_host_scenarios(row["configuration"]))},
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
    certificate = certificates(source, images, directory, cpu_isa=args.cpu_isa,
        corpus_root=getattr(args, "corpus_root", ROOT / "corpora"),
        mathematical_parity=getattr(args, "diagnostic_mathematical_parity", False))
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
        verified = certificates(source, state["images"], path, cpu_isa=isa,
            corpus_root=Path(state["identity"]["corpus_root"]),
            mathematical_parity=state["identity"]["diagnostic_mathematical_parity"])
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


def cross_host_identity(args) -> dict:
    """Return non-secret remote infrastructure intent for resume authentication.

    Cloud credentials and private key material are deliberately excluded.  A
    changed location/SKU/network/SSH-public-key or disposal policy must start a
    new receipt, while the Azure helper records the exact owned lease separately.
    """
    names = ("azure_subscription", "azure_location", "azure_vm_size", "azure_pricing", "azure_image",
             "azure_os_disk_gib", "azure_ssh_source", "azure_ssh_public_key", "remote_cpu_image",
             "azure_private_subnet", "azure_tunnel_subnet", "azure_disposal", "ssh_private_key")
    return {name: str(getattr(args, name, "") or "") for name in names}


def cross_host_cli_arguments(args) -> list[str]:
    """Forward explicit cloud intent without putting credential contents in argv.

    The private key is a path held by the local/CI credential boundary.  Azure
    authentication itself is intentionally delegated to the already logged-in
    `az` process (local login or CI federated login); no token is read or
    serialized by this driver.
    """
    mapping = (("--remote-cpu-image", "resolved_remote_cpu_image"),
               ("--azure-subscription", "azure_subscription"),
               ("--azure-location", "azure_location"),
               ("--azure-vm-size", "azure_vm_size"),
               ("--azure-pricing", "azure_pricing"),
               ("--azure-image", "azure_image"),
               ("--azure-os-disk-gib", "azure_os_disk_gib"),
               ("--azure-ssh-source", "azure_ssh_source"),
               ("--azure-ssh-public-key", "azure_ssh_public_key"),
               ("--azure-private-subnet", "azure_private_subnet"),
               ("--azure-tunnel-subnet", "azure_tunnel_subnet"),
               ("--azure-disposal", "azure_disposal"),
               ("--ssh-private-key", "ssh_private_key"))
    result = []
    for flag, name in mapping:
        value = getattr(args, name, None)
        if flag == "--remote-cpu-image" and value in (None, ""):
            value = getattr(args, "remote_cpu_image", None)
        if value not in (None, ""):
            result.extend((flag, str(value)))
    return result


def run_variant(args, source: dict) -> dict:
    """Advance one ISA prefix; the outer driver owns the shared device lease."""
    args.models = args.models.resolve(strict=True)
    args.model_ramdisk_root = args.model_ramdisk_root.resolve(strict=True)
    args.reference_cache_root.mkdir(parents=True, exist_ok=True)
    args.reference_cache_root = args.reference_cache_root.resolve(strict=True)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    receipt_path = directory / "pipeline.json"
    mathematical_parity = getattr(args, "diagnostic_mathematical_parity", False)
    corpus_root = getattr(args, "corpus_root", ROOT / "corpora").resolve()
    if args.through == Phase.PARITY.value and not mathematical_parity:
        raise ValueError("--through parity requires --diagnostic-mathematical-parity")
    identity = {"source": source, "cpu_isa": args.cpu_isa, "models": str(args.models),
                "test_user": [os.getuid(), os.getgid()],
                "ramdisk": str(args.model_ramdisk_root), "image_tag": args.image,
                "reference_cache_root": str(args.reference_cache_root),
                "corpus_root": str(corpus_root), "diagnostic_mathematical_parity": mathematical_parity,
                "cross_host": cross_host_identity(args)}
    state = {"schema": 1, "identity": identity, "phases": {}, "certified": False}
    if args.resume and receipt_path.exists():
        state = json.loads(receipt_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("source/build inputs changed; start a new output directory")
        validate_phase_prefix(state["phases"], mathematical_parity)
        recovered = recover_interrupted_phase(state, mathematical_parity)
        validate_phase_failure(state, mathematical_parity)
        if recovered:
            write_json(receipt_path, state)
    elif receipt_path.exists():
        raise ValueError("output already has a run; use --resume or a new directory")
    for phase in pipeline_phases(mathematical_parity):
        if source_identity() != source:
            raise ValueError("source changed after admission; no further certificate transitions allowed")
        if "model_sources" in state:
            current = model_identities(json.loads((directory / "all-cells.json").read_text()))
            if state["model_sources"] != current:
                raise ValueError("model files changed after full-inventory admission")
        if phase.value in state["phases"]:
            evidence = state["phases"][phase.value]
            for name, checksum in evidence["files"].items():
                if digest(json.loads((directory / name).read_text())) != checksum:
                    raise ValueError(f"changed/missing {phase.value} evidence: {name}")
            if phase == Phase.BUILD:
                for item in state["images"].values():
                    current_image = image_identity(item["id"])
                    if any(current_image.get(key) != item.get(key) for key in ("id", "labels", "layers")):
                        raise ValueError("installed build image identity/labels/layers changed")
                require_image_pair(state["images"], source, args.cpu_isa)
            elif phase == Phase.GENERATION:
                generation_evidence(directory, json.loads((directory / "all-cells.json").read_text()),
                                    state["images"]["runtime"]["id"], args.cpu_isa, corpus_root)
            print(f"[production-ci] REUSE {phase.value} (identity/evidence verified)", flush=True)
        else:
            started = time.monotonic()
            print(f"[production-ci] RUN {phase.value}", flush=True)
            if "failure" in state:
                # Validation above proves this is exactly the phase now being
                # retried; discard the old terminal record only at retry.
                del state["failure"]
            write_json(receipt_path, state | {"running": phase.value})
            files = []
            if phase == Phase.BUILD:
                state["images"] = build(args, source, directory)
                require_image_pair(state["images"], source, args.cpu_isa)
                # Admit the canonical model files before any model test,
                # not after numerical parity has already consumed them.
                # This also makes a build-only run sufficient to discover
                # exact candidates for standalone diagnostic exercises.
                run_test_runner(state["images"], ["python3", "scripts/ci/model_parity_inventory.py",
                    "--build-dir", "build_v2_integration", "--scope", InventoryScope.ALL.value,
                    "--export-manifest", "/ci-results/container-all-cells.json",
                    "--source-revision", source["revision"]], args, directory, "discovery.log")
                inventory = remap_manifest(
                    json.loads((directory / "container-all-cells.json").read_text()), args.models)
                projected = e2e_projection(inventory, source["revision"])
                remote = cross_host_e2e_projection(inventory, source["revision"])
                state["model_sources"] = model_identities(inventory)
                write_json(directory / "all-cells.json", inventory)
                write_json(directory / "manifest.json", projected)
                write_json(directory / "cross-host-manifest.json", remote)
                files = ["container-all-cells.json", "all-cells.json", "manifest.json", "cross-host-manifest.json"]
            elif phase == Phase.PREREQUISITES:
                run_test_runner(state["images"], ["python3", "scripts/ci/run_production_prerequisites.py",
                    "--build-dir", "build_v2_integration", "--installed-build-receipt", "/src/installed-tests.json",
                    "--output", "/ci-results/prerequisites"], args, directory, "prerequisites.log")
                validate_prerequisites(json.loads((directory / "prerequisites/prerequisites.json").read_text()))
                files = ["prerequisites/prerequisites.json"]
            elif phase == Phase.GENERATION:
                # Reject missing/unapproved answers before launching a server.
                inventory = json.loads((directory / "all-cells.json").read_text())
                ApprovedGenerationCorpus.load_reviewed(directory / "source", corpus_root, args.cpu_isa,
                                                       inventory, state["model_sources"])
                run([sys.executable, str(directory / "source/scripts/ci/run_model_parity_generation.py"),
                     "--mode", "regression", "--manifest", str(directory / "all-cells.json"),
                     "--source-revision", source["revision"], "--container-image", state["images"]["runtime"]["id"],
                     "--cpu-isa", args.cpu_isa, "--corpus-root", str(corpus_root),
                     "--prerequisite-report", str(directory / "prerequisites/prerequisites.json"),
                     "--model-ramdisk-root", str(args.model_ramdisk_root),
                     "--output", str(directory / "generation")], directory / "generation.log", cwd=ROOT)
                generation_evidence(directory, inventory, state["images"]["runtime"]["id"], args.cpu_isa, corpus_root)
                files = ["generation/report.json"]
            elif phase == Phase.PARITY:
                run_test_runner(state["images"], ["python3", "scripts/ci/run_production_parity_campaigns.py",
                    "--build-dir", "build_v2_integration", "--installed-build-receipt", "/src/installed-tests.json",
                    "--reuse-passed-preflight-report", "/ci-results/prerequisites/prerequisites.json",
                    "--model-ramdisk-root", str(args.model_ramdisk_root), "--persistent-model-cache-dir", "cache",
                    "--report", "/ci-results/parity.json"], args, directory, "parity.log")
                validate_parity(json.loads((directory / "parity.json").read_text()),
                                json.loads((directory / "all-cells.json").read_text())["cells"])
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
            elif phase == Phase.CROSS_HOST_E2E:
                remote_manifest = json.loads((directory / "cross-host-manifest.json").read_text())
                if remote_manifest["cells"]:
                    remote_cpu_image = getattr(args, "resolved_remote_cpu_image", None)
                    if not isinstance(remote_cpu_image, str) or not remote_cpu_image:
                        raise ValueError("cross-host E2E requires an admitted AVX2 remote CPU runtime image")
                    command = [sys.executable,
                               str(directory / "source/scripts/ci/run_production_cross_host_e2e.py"),
                               "--manifest", str(directory / "cross-host-manifest.json"),
                               "--source-revision", source["revision"],
                               "--container-image", state["images"]["runtime"]["id"],
                               "--remote-cpu-image", remote_cpu_image,
                               "--models", str(args.models),
                               "--model-ramdisk-root", str(args.model_ramdisk_root),
                               "--report", str(directory / "cross-host-e2e.json")]
                    command.extend(cross_host_cli_arguments(args))
                    run(command, directory / "cross-host-e2e.log", cwd=ROOT)
                else:
                    # Preserve the ordered lifecycle even when this source
                    # inventory has no remote-tagged model.  This is an honest
                    # non-applicable result, never an empty successful proof.
                    write_json(directory / "cross-host-e2e.json", {
                        "schema": 1, "eligible": False, "complete": True,
                        "source_revision": source["revision"],
                        "image": state["images"]["runtime"]["id"],
                        "manifest_digest": digest(remote_manifest),
                        "scenarios": [], "all_resources_retired": True})
                validate_cross_host_report(
                    json.loads((directory / "cross-host-e2e.json").read_text()),
                    remote_manifest, state["images"]["runtime"]["id"], source["revision"],
                    getattr(args, "resolved_remote_cpu_image", None))
                files = ["cross-host-e2e.json"]
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


def remote_cpu_runtime_image(args, states: dict) -> str | None:
    """Choose the explicit peer runtime for a mixed-ISA remote CPU cohort.

    The official shipping set builds AVX2 before any cross-host phase, so an
    AVX512 controller can retain its local performance while remote Azure CPU
    ranks receive the compatible AVX2 sibling.  A caller may supply an
    immutable image override. A partial AVX512-only run leaves this unset; the
    actual cross-host phase then fails only when its manifest declares a
    remote CPU peer, rather than making a non-applicable phase depend on it.
    """
    explicit = getattr(args, "remote_cpu_image", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    avx2 = states.get("AVX2", {}).get("images", {}).get("runtime", {})
    identity = avx2.get("id") if isinstance(avx2, dict) else None
    return identity if isinstance(identity, str) and identity else None


def drive_variants(args, source: dict) -> dict:
    """Finish a gate for every ISA before entering the next gate for any ISA.

    In particular, both HTTP E2E suites and both remote MPI E2E suites precede
    the first benchmark. Runs are
    sequential on this node so neither inference nor timing competes for its
    devices. Only immutable dependencies/reference packs are shared, never
    correctness results or image-bound certificates.
    """
    states = {}
    for phase in pipeline_phases(getattr(args, "diagnostic_mathematical_parity", False)):
        for isa in args.cpu_isas:
            print(f"[production-ci] ISA={isa} gate={phase.value}", flush=True)
            variant = variant_arguments(args, isa, phase)
            if phase is Phase.CROSS_HOST_E2E:
                variant.resolved_remote_cpu_image = remote_cpu_runtime_image(args, states)
            try:
                states[isa] = run_variant(variant, source)
            except Exception as error:
                persist_phase_failure(variant.output, phase, error,
                                      getattr(args, "diagnostic_mathematical_parity", False))
                raise
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
    parser.add_argument("--corpus-root", type=Path, default=ROOT / "corpora",
                        help="materialized versioned token corpus; approval comes from the source snapshot")
    parser.add_argument("--diagnostic-mathematical-parity", action="store_true",
                        help="explicitly add the full HF mathematical matrix; not a routine regression gate")
    parser.add_argument("--cpu-isa", choices=SHIPPING_ISAS, action="append", dest="cpu_isas",
                        help="local piecewise ISA selection; default and official publication require both")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", choices=[phase.value for phase in Phase], default="certify")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--image", help="AVX512 certified image tag; AVX2 appends -avx2")
    parser.add_argument("--azure-subscription", default=os.environ.get("AZURE_SUBSCRIPTION_ID", ""),
                        help="Azure subscription UUID for declared cross-host E2E (or AZURE_SUBSCRIPTION_ID)")
    parser.add_argument("--azure-location", default=os.environ.get("LLAMINAR_AZURE_LOCATION", "uksouth"))
    parser.add_argument("--azure-vm-size", default=os.environ.get("LLAMINAR_AZURE_VM_SIZE", "Standard_E16ads_v6"))
    parser.add_argument("--azure-pricing", choices=("spot", "on-demand"),
                        default=os.environ.get("LLAMINAR_AZURE_PRICING", "spot"),
                        help="Azure VM billing contract for cross-host E2E; Spot never falls back")
    parser.add_argument("--azure-image", default=os.environ.get(
        "LLAMINAR_AZURE_IMAGE", "Canonical:ubuntu-24_04-lts:server:24.04.202608270"))
    parser.add_argument("--azure-os-disk-gib", type=int,
                        default=int(os.environ.get("LLAMINAR_AZURE_OS_DISK_GIB", "256")))
    parser.add_argument("--azure-ssh-source", default=os.environ.get("LLAMINAR_AZURE_SSH_SOURCE", ""),
                        help="One public controller IPv4 /32; required when remote cells are declared")
    parser.add_argument("--azure-ssh-public-key", default=os.environ.get("LLAMINAR_AZURE_SSH_PUBLIC_KEY", ""),
                        help="Path to the Ed25519 public key used by the Azure lease")
    parser.add_argument("--ssh-private-key", default=os.environ.get("LLAMINAR_AZURE_SSH_PRIVATE_KEY", ""),
                        help="Private key path used only by the remote SSH transport")
    parser.add_argument("--remote-cpu-image",
                        help="explicit AVX2 Release runtime for remote CPU MPI peers; official dual-ISA runs select the built sibling")
    parser.add_argument("--azure-private-subnet", default=os.environ.get("LLAMINAR_AZURE_PRIVATE_SUBNET", "10.221.0.0/24"))
    parser.add_argument("--azure-tunnel-subnet", default=os.environ.get("LLAMINAR_AZURE_TUNNEL_SUBNET", DEFAULT_TUNNEL_SUBNET))
    parser.add_argument("--azure-disposal", choices=("delete-owned-group", "deallocate-retain-disks"),
                        default=os.environ.get("LLAMINAR_AZURE_DISPOSAL", "delete-owned-group"))
    parser.add_argument("--publish", action="store_true", help="official CI only: publish both certified images and one result commit")
    args = parser.parse_args(argv)
    if args.through == Phase.PARITY.value and not args.diagnostic_mathematical_parity:
        parser.error("--through parity requires --diagnostic-mathematical-parity")
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
                "corpus_root": str(args.corpus_root.resolve()),
                "diagnostic_mathematical_parity": args.diagnostic_mathematical_parity,
                "test_user": [os.getuid(), os.getgid()],
                "cross_host": cross_host_identity(args)}
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
