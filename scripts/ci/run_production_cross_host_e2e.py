#!/usr/bin/env python3
"""Certify canonical cross-host MPI E2E cells on owned Azure CPU peers.

The C++ parity inventory is the only source of model/topology membership.  An
eligible cell is executed through the existing HTTP harness with one local GPU
rank and one CPU rank per distinct remote VM.  This command owns a durable
Azure lease and stages the exact Release image and complete model shard set;
it never falls back to a local or node-local run.  Its explicit planning-only
diagnostic instead leaves GGUF data on discovery root and proves typed metadata
publication to followers, so an accidental remote model read cannot be masked
by a WAN copy. Authentication is supplied by an existing ``az login`` (or
CI's federated login), and secrets are never written to reports.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, suppress
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

from azure_cross_host_resources import (
    AzureCLI, AzureCPUCapacity, AzureComputePricing, DEFAULT_TUNNEL_SUBNET, Disposal,
    campaign_resources,
)
from cross_host_network import private_mpi_connection, validate_local_network
from cross_host_artifacts import distribute
from docker_paths import validate_attached_execution
from model_parity_inventory import InventoryScope, cross_host_scenarios
from production_artifacts import CPUISA, digest, image_cpu_isa, image_identity, runtime_image_content, write_json
from run_model_parity_e2e import (
    E2ECellBudget, cell_timeout_seconds, certification_environment, run_e2e_process, validate_long_context_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
HARNESS = ROOT / "tests/v2/e2e/server/test_server_e2e.sh"
# The controller container must see its launch files and peer-collected output
# through the Docker daemon's mount namespace.  A caller may put durable
# reports in /tmp, which is intentionally private to this devcontainer, so
# never make the report directory double as a bind-mounted fleet workspace.
CROSS_HOST_CONTAINER_SCRATCH = ROOT / "parity-results" / ".cross-host-container-scratch"
# Ubuntu clears /tmp across VM restarts. Keep the incremental transfer basis
# on the explicitly retained user disk, separate from mounted model inputs.
REMOTE_IMAGE_ARCHIVE = "/home/llaminar/.cache/llaminar-cross-host/runtime-image.tar"


@dataclass(frozen=True)
class RuntimeImage:
    """One authenticated Release runtime and its immutable CPU ISA contract.

    ``identity`` is Docker's local immutable image identity.  Images may use
    different local Docker IDs after import, so only the controller-side ID is
    retained here; :func:`stage_image` authenticates the portable content on
    every remote daemon separately.
    """

    identity: dict
    cpu_isa: CPUISA


CPU_ISA_FEATURES: dict[CPUISA, frozenset[str]] = {
    # Keep this list identical to the global flags in src/v2/CMakeLists.txt.
    # Linux exposes BMI1 as ``bmi1`` even though GCC's option is ``-mbmi``.
    CPUISA.AVX2: frozenset({"sse4_1", "avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "popcnt"}),
    CPUISA.AVX512: frozenset({"sse4_1", "avx", "avx2", "fma", "f16c", "bmi1", "bmi2", "popcnt",
                              "avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vnni"}),
}


def safe_name(value: str) -> str:
    """Map an opaque canonical ID to a bounded local filename component."""
    result = "".join(character if character.isalnum() or character in "._-" else "_"
                    for character in value)
    if not result or result in {".", ".."}:
        raise ValueError("cross-host scenario has no safe artifact name")
    return result[:180]


def load_manifest(path: Path, revision: str) -> dict:
    """Admit a revision-bound cross-host projection and validate every row."""
    document = json.loads(path.read_text())
    if (document.get("schema") != 1 or document.get("scope") != InventoryScope.CROSS_HOST_E2E.value
            or document.get("source_revision") != revision or not isinstance(document.get("cells"), list)):
        raise ValueError("cross-host E2E requires the revision-bound canonical projection")
    for row in document["cells"]:
        configuration = row.get("configuration") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or not isinstance(configuration, dict)
                or not isinstance(configuration.get("model"), str)
                or not configuration["model"] or not isinstance(row.get("model_files"), list)
                or not row["model_files"]):
            raise ValueError("cross-host projection omitted its complete model identity")
        cross_host_scenarios(configuration)
    return document


def selected_rows(manifest: dict, first_cases: list[str] | None = None) -> list[tuple[dict, dict]]:
    """Order the complete canonical projection without excluding any scenarios.

    Explicit priorities shorten feedback for a newly failing or unseen case.
    They change scheduling only: every original row remains mandatory for the
    certificate, and unlisted rows retain their canonical relative order.
    """
    rows = [(parent, scenario) for parent in manifest["cells"]
            for scenario in cross_host_scenarios(parent["configuration"])]
    if not rows:
        raise ValueError("remote phase was invoked without an eligible canonical scenario")
    identities = [scenario["id"] for _, scenario in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("cross-host projection repeats a scenario identity")
    first_cases = first_cases or []
    if len(set(first_cases)) != len(first_cases) or any(ident not in identities for ident in first_cases):
        raise ValueError("prioritized cross-host cases must be distinct canonical scenario IDs")
    priority = {ident: index for index, ident in enumerate(first_cases)}
    rows.sort(key=lambda row: priority.get(row[1]["id"], len(priority)))
    return rows


def checked_file(value: str, label: str) -> Path:
    """Resolve one readable regular file before cloud allocation."""
    path = Path(value).resolve(strict=True)
    if not path.is_file() or not os.access(path, os.R_OK):
        raise ValueError(f"{label} must be a readable regular file")
    return path


def validate_private_key(path: Path) -> None:
    """Require an SSH key that is not readable by group or other users."""
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("SSH private key must not be group/world readable")


def admit_runtime_image(reference: str, source_revision: str) -> RuntimeImage:
    """Bind one requested runtime to the canonical source and ISA contracts.

    A mixed-ISA MPI cohort is valid only when every participant runs the same
    source tree and Release ABI.  It is *not* valid to silently substitute a
    generic CPU image after a peer rejects the controller image.  Callers name
    the optional CPU-peer runtime explicitly; this admission merely rejects a
    stale, non-runtime, incomplete-backend, or cross-source image before a
    large transfer or daemon launch can obscure the defect.
    """
    identity = image_identity(reference)
    labels = identity.get("labels")
    if not isinstance(labels, dict):
        raise ValueError("runtime image omitted OCI labels")
    cpu_isa = image_cpu_isa(identity)
    required = {
        "org.opencontainers.image.revision": source_revision,
        "org.llaminar.image_role": "runtime",
        "org.llaminar.build_type": "Release",
        "org.llaminar.cuda": "ON",
        "org.llaminar.rocm": "ON",
    }
    if any(labels.get(key) != value for key, value in required.items()):
        raise ValueError("runtime image does not match the requested full-backend Release source")
    source_tree = labels.get("org.llaminar.source_tree")
    if not isinstance(source_tree, str) or not source_tree:
        raise ValueError("runtime image omitted its immutable source tree identity")
    return RuntimeImage(identity=identity, cpu_isa=cpu_isa)


def remote_cpu_features(host: str, key: Path) -> frozenset[str]:
    """Read one remote CPU's kernel-authoritative x86 feature set.

    This is physical inventory evidence, not a performance estimate and not a
    replacement for Llaminar's MPI inventory.  Container selection happens
    before the remote rank can join MPI, so the transport owner must first
    prove that its own executable will decode on that host.  ``/proc/cpuinfo``
    is deliberately read through the host SSH boundary rather than from a
    possibly incompatible container image.
    """
    output = ssh(host, key, ["bash", "-lc", "LC_ALL=C grep -m1 '^flags[[:space:]]*:' /proc/cpuinfo"], timeout=30)
    prefix, separator, flags = output.strip().partition(":")
    if prefix.strip() != "flags" or not separator:
        raise ValueError(f"remote host {host} did not publish an x86 CPU flags row")
    parsed = frozenset(flag.lower() for flag in flags.split())
    if not parsed:
        raise ValueError(f"remote host {host} published an empty x86 CPU flags row")
    return parsed


def compatible_remote_cpu_isas(features: frozenset[str]) -> tuple[CPUISA, ...]:
    """Return every shipped ISA executable that this peer can execute.

    Preserve the explicit maximum-ISA ordering so diagnostics explain why a
    peer accepts AVX2 but rejects AVX-512.  No scalar or generic fallback is
    advertised because the release matrix deliberately ships only these two
    concrete compiled contracts.
    """
    compatible = tuple(isa for isa in (CPUISA.AVX512, CPUISA.AVX2)
                       if CPU_ISA_FEATURES[isa].issubset(features))
    if not compatible:
        raise ValueError("remote CPU does not support Llaminar's AVX2 release baseline")
    return compatible


def admit_remote_cpu_runtime(controller: RuntimeImage, peer: RuntimeImage,
                             hosts: list[dict], key: Path, directory: Path) -> None:
    """Prove a peer image is source-coherent and executable on every CPU host.

    The controller may legitimately use AVX-512 while Azure CPU ranks use the
    AVX2 image.  Their common source-tree label is the ABI/protocol contract;
    their distinct local ISA labels are the physical-execution contract.  The
    resulting immutable observation is durable campaign evidence, so no later
    planner result has to reconstruct topology facts from a SIGILL symptom.
    """
    controller_labels = controller.identity["labels"]
    peer_labels = peer.identity["labels"]
    if peer_labels["org.llaminar.source_tree"] != controller_labels["org.llaminar.source_tree"]:
        raise ValueError("remote CPU runtime has a different immutable source tree than the controller")
    observations = []
    for host in hosts:
        address = host["public_ip"]
        compatible = compatible_remote_cpu_isas(remote_cpu_features(address, key))
        if peer.cpu_isa not in compatible:
            offered = ",".join(item.value for item in compatible)
            raise ValueError(f"remote host {address} supports {offered}, but the supplied "
                             f"CPU runtime requires {peer.cpu_isa.value}")
        observations.append({"host": address, "supported_cpu_isas": [item.value for item in compatible],
                             "selected_cpu_isa": peer.cpu_isa.value})
    write_json(directory / "remote-cpu-runtime.json", {
        "controller_image": controller.identity["id"],
        "controller_cpu_isa": controller.cpu_isa.value,
        "remote_cpu_image": peer.identity["id"],
        "remote_cpu_isa": peer.cpu_isa.value,
        "source_revision": controller_labels["org.opencontainers.image.revision"],
        "source_tree": controller_labels["org.llaminar.source_tree"],
        "hosts": observations,
    })


def infrastructure_failure(error: Exception, private_key: Path) -> dict[str, str]:
    """Publish a bounded, key-redacted setup failure in durable campaign evidence.

    A cloud lease can retire successfully even though a local container fleet
    failed before it created a scenario result.  Keeping that cause beside the
    retirement receipt makes the failure actionable without recording the
    private-key path that happened to be present in a subprocess command.
    """
    message = str(error).replace(str(private_key), "<ssh-private-key>")
    return {"type": type(error).__name__, "message": message[:2048] or "no diagnostic text"}


def run(command: list[str], *, cwd: Path = ROOT, env: dict | None = None,
        timeout: int = 1200, log: Path | None = None) -> str:
    """Execute an argv-bounded command with cancellation-owned process retirement.

    Artifact staging can run long enough for CI or an operator to cancel the
    campaign.  A child in the controller's process group would survive the
    signal handler while :class:`subprocess.Popen` waits for it, preventing the
    Azure lease from reaching its cleanup scope.  Each command is therefore a
    separate process group and is retired before the original interruption or
    timeout leaves this function.
    """
    stream = log.open("w") if log else subprocess.PIPE
    process = None
    try:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT if log else subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    finally:
        if log:
            stream.close()
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command, output=stdout, stderr=stderr)
    return stdout or ""


def ssh_argv(host: str, key: Path, command: list[str]) -> list[str]:
    """Build the single quoted SSH boundary shared by sequential and owned jobs."""
    if any("\x00" in part or "\n" in part for part in command):
        raise ValueError("remote command contains a control character")
    payload = " ".join(shlex.quote(part) for part in command)
    return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=20", "-i", str(key), f"llaminar@{host}", payload]


def ssh(host: str, key: Path, command: list[str], timeout: int = 1200) -> str:
    """Run a quoted fixed command on an owned VM over non-interactive SSH."""
    return run(ssh_argv(host, key, command), timeout=timeout)


def import_runtime_archives(hosts: list[dict], key: Path, directory: Path,
                            timeout: float = 600) -> None:
    """Load independent Docker stores concurrently, joining every owned client.

    Start all SSH clients before polling any of them. A file owns each output
    stream, so one noisy importer cannot block another on a full pipe. Failure,
    deadline and controller cancellation all retire the exact process groups
    before returning to the image/VM owners. The Azure lease still owns remote
    daemon retirement: dropping SSH is not proof that Docker stopped importing.
    No image is admitted here; stage_image authenticates every resulting store.
    """
    peers = [peer["public_ip"] for peer in hosts]
    if (not peers or len(set(peers)) != len(peers)
            or len({safe_name(peer) for peer in peers}) != len(peers)
            or not 0 < timeout <= 600):
        raise ValueError("image imports require distinct peers and a bounded positive deadline")
    started = time.monotonic()
    jobs = []
    with ExitStack() as logs:
        try:
            for host in peers:
                log = directory / f"runtime-import-{safe_name(host)}.log"
                output = logs.enter_context(log.open("w"))
                process = subprocess.Popen(
                    ssh_argv(host, key, ["docker", "load", "--input", REMOTE_IMAGE_ARCHIVE]),
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=output,
                    stderr=subprocess.STDOUT, start_new_session=True)
                jobs.append((host, process, log))
                print(f"[production-cross-host] importing full runtime on {host}", flush=True)
            pending = list(jobs)
            while pending:
                for job in list(pending):
                    host, process, log = job
                    status = process.poll()
                    if status is None:
                        continue
                    if status:
                        raise RuntimeError(f"runtime import on {host} failed (exit {status}); see {log}")
                    pending.remove(job)
                    print(f"[production-cross-host] runtime import complete on {host} in "
                          f"{time.monotonic() - started:.1f}s", flush=True)
                if pending:
                    if time.monotonic() - started >= timeout:
                        raise TimeoutError("concurrent runtime imports exceeded their shared deadline")
                    time.sleep(0.1)
        finally:
            # Signal every live client first; do not serialize cancellation by
            # waiting on one peer while another keeps importing in the background.
            retirement_errors = []
            for host, process, _ in jobs:
                try:
                    if process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                except OSError as error:
                    retirement_errors.append(f"{host}: {type(error).__name__}")
            for host, process, _ in jobs:
                try:
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as error:
                    retirement_errors.append(f"{host}: {type(error).__name__}")
            if retirement_errors:
                raise RuntimeError("runtime import client retirement failed: " + "; ".join(retirement_errors))


def wait_remote_runtime(host: str, key: Path, timeout: int = 900) -> None:
    """Wait until cloud-init has prepared Docker and the artifact transport.

    SSH may become reachable before package installation finishes.  Probe on a
    fresh connection so the ``docker`` group membership is also observed; a
    fixed sleep would either race the bootstrap or waste the common fast path.
    """
    deadline = time.monotonic() + timeout
    next_progress = 0.0
    while True:
        try:
            ssh(host, key, ["bash", "-lc",
                            "test -f /var/lib/cloud/instance/boot-finished && "
                            "docker info >/dev/null 2>&1 && command -v rsync >/dev/null"],
                timeout=30)
            return
        except subprocess.CalledProcessError:
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"remote CPU runtime on {host} did not become ready")
            if now >= next_progress:
                print(f"[production-cross-host] waiting for Docker/artifact runtime on {host}", flush=True)
                next_progress = now + 30
            time.sleep(min(5, deadline - now))


def upload_artifact(host: str, key: Path, source: Path, destination: str) -> None:
    """Incrementally publish one immutable file over SSH with its source timestamp.

    Rsync reuses existing archive/model blocks rather than uploading gigabytes
    again. No in-place mutation or independent whole-GGUF hash gate is used;
    rsync owns a temporary file and atomically renames it on successful transfer.
    Docker separately verifies the imported image's exact runtime contents.
    """
    transport = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                            "-o", "ConnectTimeout=20", "-i", str(key)])
    output = run(["rsync", "--times", "--protect-args", "--modify-window=-1", "--stats",
                  "-e", transport, "--", str(source), f"llaminar@{host}:{destination}"], timeout=1800)
    print(output, flush=True)


def cached_runtime_image(host: str, key: Path, expected: dict) -> str | None:
    """Find authenticated complete content among this helper's retained imports.

    Transport tags only bound the search; they are not an identity or approval.
    Daemons may use different immutable lookup IDs for the same image. Compare
    complete config and ordered layer digests, then return that daemon's ID.
    No hit means an explicit artifact transfer, never a different runtime.
    """
    identities = sorted(set(ssh(host, key, ["docker", "image", "ls", "--quiet", "--no-trunc",
                                           "--filter", "reference=llaminar-cross-host-transfer:*"]).split()))
    if not identities:
        return None
    if any(not identity.startswith("sha256:") for identity in identities):
        raise ValueError("remote daemon returned an invalid retained image identity")
    observations = json.loads(ssh(host, key, ["docker", "image", "inspect", *identities]))
    if (not isinstance(observations, list) or len(observations) != len(identities)
            or {item.get("Id") for item in observations} != set(identities)):
        raise ValueError("remote image inspection changed its requested identities")
    for observation in observations:
        if runtime_image_content(observation) == expected:
            return observation["Id"]
    return None


def stage_image(image: str, hosts: list[dict], key: Path, directory: Path) -> dict[str, str]:
    """Authenticate runnable content, retaining each daemon's immutable lookup ID.

    A named archive survives both Docker image stores. The transport tag is
    only a lookup handle: every imported config and ordered filesystem digest
    must match before any container can be created from its immutable ID.
    """
    if not hosts:
        raise ValueError("runtime image staging requires an owned CPU peer")
    archive = directory / "runtime-image.tar"
    admitted = json.loads(run(["docker", "image", "inspect", image]))[0]
    expected = runtime_image_content(admitted)
    imported = {}
    missing = []
    for peer in hosts:
        cached = cached_runtime_image(peer["public_ip"], key, expected)
        if cached is None:
            missing.append(peer)
        else:
            imported[peer["public_ip"]] = cached
    cached_hosts = sorted(imported)
    if not missing:
        print(f"[production-cross-host] verified full runtime cache on {len(hosts)} peers; no image transfer", flush=True)
        write_json(directory / "runtime-import.json", {"source_image": admitted["Id"],
            "runtime_content_digest": digest(expected), "imported": imported, "cached_hosts": cached_hosts})
        return imported
    tag = "llaminar-cross-host-transfer:" + uuid.uuid4().hex
    run(["docker", "tag", admitted["Id"], tag])
    try:
        run(["docker", "save", "-o", str(archive), tag], timeout=1800)
        for peer in missing:
            host = peer["public_ip"]
            ssh(host, key, ["mkdir", "-p", str(Path(REMOTE_IMAGE_ARCHIVE).parent)])
        distribute(archive, REMOTE_IMAGE_ARCHIVE, missing, key, upload_artifact)
        import_runtime_archives(missing, key, directory)
        for peer in missing:
            host = peer["public_ip"]
            observed = json.loads(ssh(host, key, ["docker", "image", "inspect", tag]))[0]
            if runtime_image_content(observed) != expected:
                raise ValueError("remote host loaded different runtime filesystem/configuration")
            if not isinstance(observed.get("Id"), str) or not observed["Id"].startswith("sha256:"):
                raise ValueError("remote daemon omitted its immutable image ID")
            imported[host] = observed["Id"]
    finally:
        if image_identity(tag)["id"] != admitted["Id"]:
            raise ValueError("owned image transfer tag changed; refusing to remove it")
        run(["docker", "image", "rm", tag])
    write_json(directory / "runtime-import.json", {"source_image": admitted["Id"],
        "runtime_content_digest": digest(expected), "imported": imported, "cached_hosts": cached_hosts})
    return imported


def stage_models(parents: list[dict], hosts: list[dict], key: Path, directory: Path,
                 model_dir: Path | None = None) -> Path:
    """Stage all declared GGUF shards and reject ambiguous duplicate names.

    The local copy is made once per lease directory.  A later call may reuse
    that already-admitted directory solely to transfer the same sources to
    newly provisioned peers; it never rebuilds or replaces rank-zero inputs.
    """
    local_stage = model_dir is None
    model_dir = model_dir or (directory / "models")
    if not local_stage and not model_dir.is_dir():
        raise ValueError("reused model stage is not an existing directory")
    model_dir.mkdir(exist_ok=not local_stage)
    sources: dict[str, Path] = {}
    for parent in parents:
        for name in parent["model_files"]:
            source = checked_file(name, "model shard")
            previous = sources.get(source.name)
            if previous is not None and previous != source:
                raise ValueError(f"model shards collide on basename {source.name!r}")
            sources[source.name] = source
    if local_stage:
        for basename, source in sources.items():
            destination = model_dir / basename
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"staged model path already belongs to another shard: {destination}")
            # The staged directory is bind-mounted into rank-zero's container.
            # Symlinking back to the source path would resolve inside that mount
            # and either loop or escape the admitted model namespace.  A real
            # copy also makes the temporary ramdisk an explicit, immutable input
            # for the duration of every cross-host case.
            shutil.copy2(source, destination)
    for peer in hosts:
        ssh(peer["public_ip"], key, ["mkdir", "-p", "/opt/llaminar-models"])
    if hosts:
        for basename in sources:
            # The immutable RAM-staged file is the input for every host, not
            # another read of the SSD source for each upload. Only one copy
            # crosses the WAN; subsequent copies stay inside the leased VNet.
            distribute(model_dir / basename, f"/opt/llaminar-models/{basename}", hosts, key, upload_artifact)
    return model_dir


def run_case(parent: dict, scenario: dict, image: RuntimeImage, model_dir: Path, hosts: list[dict],
             args: argparse.Namespace, directory: Path, *, planning_only: bool = False) -> dict:
    """Run one public frontend, or its explicit non-certifying plan probe.

    A plan probe retains the normal hostfile, container fleet, MPI daemons,
    model metadata and automatic planner.  It stops after the public ``plan``
    command writes an apply document, so it is useful for control-plane
    diagnostics without implying that HTTP inference was certified.
    """
    import cross_host_containers
    from cross_host_containers import CASE_ROOT, container_fleet

    ident = scenario["id"]
    configuration = parent["configuration"]
    model = Path(configuration["model"]).resolve(strict=True)
    staged = model_dir / model.name
    if not staged.is_file():
        raise ValueError("parent model is absent from the complete staged shard set")
    if len(hosts) + 1 != scenario["topology"]["execution_ranks"]:
        raise ValueError("owned VM count differs from the canonical MPI topology")
    if planning_only and scenario["frontend"] != "plan-apply":
        raise ValueError("planning-only probes require the canonical plan-apply frontend")
    artifact = directory / safe_name(ident)
    artifact.mkdir()
    # Plan and direct serve receive the same complete policy. Applying the
    # saved document must not reapply automatic filters or silently override
    # memory-affecting MTP/KV settings after the plan has been admitted.
    policy = [*scenario["server_policy_args"], "--mpi-hostfile", str(CASE_ROOT / "hosts")]
    plan_policy = list(policy)
    if scenario["frontend"] == "plan-apply":
        policy = ["--config", str(CASE_ROOT / "plan.json")]
    profile = {**configuration["e2e"], "movement_evidence": scenario["movement_evidence"]}
    env = certification_environment(profile, artifact)
    plan_args = ["plan", "-m", f"/opt/llaminar-models/{model.name}",
                 *plan_policy, "--format", "json",
                 "--output", str(CASE_ROOT / "plan.json"),
                 "--context-length", str(profile["context_length"])]
    args_file = artifact / "server-args.json"
    write_json(args_file, policy)
    started = time.monotonic()
    planning_proof = False
    evidence_error = None
    CROSS_HOST_CONTAINER_SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cross-host-case-", dir=CROSS_HOST_CONTAINER_SCRATCH) as scratch:
        mounted_root = Path(scratch)
        workspace = mounted_root / "launch"
        # Preserve the opaque canonical name inside the daemon-visible tree so
        # remote evidence remains naturally associated with the durable report.
        mounted_artifact = mounted_root / artifact.name
        mounted_artifact.mkdir()
        # The HTTP harness consumes the durable copy, while the controller
        # container sees this explicitly mirrored immutable launch contract.
        # Do not let a private bind mount inherit incidental parent files.
        write_json(mounted_artifact / "server-args.json", policy)
        with container_fleet(image=image.identity["id"], hosts=hosts, key=args.ssh_private_key,
                workspace=workspace, artifact=mounted_artifact, model_dir=model_dir, model_name=model.name,
                frontend_mode=scenario["frontend"], plan_args=plan_args,
                backend=scenario["topology"]["continuation_backend"],
                continuation_devices=scenario["topology"]["continuation_devices"],
                controller_address=args.controller_address) as launcher:
            # The two public commands are separate lifecycle phases, not a plan
            # job hidden inside server startup. Both still spend one cell deadline.
            budget = E2ECellBudget(cell_timeout_seconds(profile, image.cpu_isa))
            return_code = 0
            if scenario["frontend"] == "plan-apply":
                with (artifact / "plan.log").open("w") as log:
                    return_code = run_e2e_process(
                        [sys.executable, str(Path(cross_host_containers.__file__).resolve()),
                         "prepare", str(workspace / "fleet.json")], env, log, budget=budget)
                if return_code == 0:
                    try:
                        plan = json.loads((workspace / "plan.json").read_text())
                        planning_proof = (plan.get("kind") == "llaminar.orchestration-config"
                                          and isinstance(plan.get("configuration"), dict))
                        if not planning_proof:
                            raise ValueError("public plan omitted its orchestration configuration")
                    except (OSError, ValueError, json.JSONDecodeError) as error:
                        return_code = 1
                        evidence_error = str(error)
            if not planning_only:
                command = ["bash", str(HARNESS), "--binary", str(launcher),
                           "--suite", f"{staged}|tp|200||{ident}|e2e-certification",
                           "--server-args-file", str(args_file), "--cross-host-configuration",
                           str(args.manifest), "--cross-host-case", ident, "--port", str(args.port)]
                if return_code == 0:
                    with (artifact / "harness.log").open("w") as log:
                        return_code = run_e2e_process(command, env, log, budget=budget)
        # Container and remote-peer output was written under the daemon's
        # namespace. Copy it only after every container is retired, then make
        # the caller-selected report directory the sole durable evidence owner.
        if mounted_artifact.exists():
            shutil.copytree(mounted_artifact, artifact, dirs_exist_ok=True)
    if planning_only:
        return {"id": ident, "frontend": scenario["frontend"], "topology": scenario["topology"],
                "return_code": return_code,
                "outcome": "passed" if planning_proof else "failed",
                "planning_proof": planning_proof, "transport_proof": False, "http_proof": False,
                "resource_retired": False, "elapsed_seconds": time.monotonic() - started,
                "artifacts": str(artifact), "evidence_error": evidence_error}
    proof_ok = False
    if return_code == 0:
        try:
            validate_case_evidence(artifact, scenario, profile)
            proof_ok = True
        except (ValueError, OSError) as error:
            return_code = 1
            evidence_error = str(error)
    return {"id": ident, "frontend": scenario["frontend"], "topology": scenario["topology"],
            "return_code": return_code,
            "outcome": "passed" if proof_ok else "cell_timeout" if return_code == 124 else "failed",
            "planning_proof": planning_proof, "transport_proof": proof_ok, "http_proof": proof_ok,
            "resource_retired": False, "elapsed_seconds": time.monotonic() - started,
            "artifacts": str(artifact), "evidence_error": evidence_error}
def validate_case_evidence(directory: Path, scenario: dict, profile: dict) -> None:
    """Require the exact observer result and all eight full HTTP checks.

    Existence of a long-context JSON file is insufficient: a failed or partial
    suite also writes that artifact. Use the ordinary HTTP validator so remote
    certification cannot silently acquire weaker behavioral gates.
    """
    validate_long_context_evidence(directory, profile)
    proof_files = list(directory.glob("*.cross-host.json"))
    if len(proof_files) != 1:
        raise ValueError("cross-host cell requires exactly one transport observer result")
    proof = json.loads(proof_files[0].read_text())
    if (proof.get("case") != scenario["id"] or proof.get("scenario") != scenario
            or not isinstance(proof.get("execution"), dict) or not proof["execution"]):
        raise ValueError("cross-host observer result has a different case/policy or no execution evidence")


def interrupt_campaign(signum: int, _frame) -> None:
    """Unwind process and Azure owners when the CI controller sends SIGTERM.

    The default SIGTERM action exits without executing Python finally blocks.
    Converting it into an exception lets the existing nested owners retire
    workers and the cloud lease; power loss still uses the durable receipt.
    """
    raise InterruptedError(f"cross-host campaign interrupted by signal {signum}")


def main(argv: list[str] | None = None) -> int:
    """Provision, execute, retire, and emit one strict remote evidence report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--remote-cpu-image",
                        help="explicit Release image for remote CPU MPI ranks; defaults to --container-image only when compatible")
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--model-ramdisk-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--first-case", action="append", default=[],
                        help="Run this exact canonical scenario first (repeatable); all scenarios remain required")
    parser.add_argument("--planning-only", action="store_true",
                        help="Explicit diagnostic: run only canonical plan-apply routes; never E2E-certifies")
    parser.add_argument("--azure-subscription", default=os.environ.get("AZURE_SUBSCRIPTION_ID", ""))
    parser.add_argument("--azure-location", default=os.environ.get("LLAMINAR_AZURE_LOCATION", "uksouth"))
    parser.add_argument("--azure-vm-size", default=os.environ.get("LLAMINAR_AZURE_VM_SIZE", "Standard_E16ads_v6"))
    parser.add_argument("--azure-pricing", choices=[item.value for item in AzureComputePricing],
                        default=os.environ.get("LLAMINAR_AZURE_PRICING", AzureComputePricing.SPOT.value),
                        help="Explicit Azure billing contract; Spot is the default and never falls back")
    parser.add_argument("--azure-image", default=os.environ.get("LLAMINAR_AZURE_IMAGE", "Canonical:ubuntu-24_04-lts:server:24.04.202608270"))
    parser.add_argument("--azure-os-disk-gib", type=int, default=int(os.environ.get("LLAMINAR_AZURE_OS_DISK_GIB", "256")))
    parser.add_argument("--azure-ssh-source", default=os.environ.get("LLAMINAR_AZURE_SSH_SOURCE", ""))
    parser.add_argument("--azure-ssh-public-key", type=Path, default=Path(os.environ.get("LLAMINAR_AZURE_SSH_PUBLIC_KEY", "")))
    parser.add_argument("--ssh-private-key", type=Path, default=Path(os.environ.get("LLAMINAR_AZURE_SSH_PRIVATE_KEY", "")))
    parser.add_argument("--azure-private-subnet", default=os.environ.get("LLAMINAR_AZURE_PRIVATE_SUBNET", "10.221.0.0/24"))
    parser.add_argument("--azure-tunnel-subnet", default=os.environ.get("LLAMINAR_AZURE_TUNNEL_SUBNET", DEFAULT_TUNNEL_SUBNET))
    parser.add_argument("--azure-disposal", choices=[item.value for item in Disposal], default=Disposal.DELETE.value)
    parser.add_argument("--reuse-azure-lease", type=Path,
                        help="Reuse this exact retired retain-disks lease; never reuse test results")
    parser.add_argument("--port", type=int, default=19080)
    args = parser.parse_args(argv)
    args.manifest = args.manifest.resolve(strict=True)
    args.models = args.models.resolve(strict=True)
    if not args.models.is_dir():
        raise ValueError("models root must be an existing directory")
    args.ssh_private_key = args.ssh_private_key.resolve(strict=True)
    args.azure_ssh_public_key = args.azure_ssh_public_key.resolve(strict=True)
    validate_private_key(args.ssh_private_key)
    if not args.azure_subscription or not args.azure_ssh_source:
        raise ValueError("Azure subscription and SSH controller /32 are required")
    public_key = args.azure_ssh_public_key.read_text().strip()
    manifest = load_manifest(args.manifest, args.source_revision)
    selected = selected_rows(manifest, args.first_case)
    if args.planning_only:
        selected = [(parent, scenario) for parent, scenario in selected
                    if scenario["frontend"] == "plan-apply"]
        if not selected:
            raise ValueError("canonical cross-host projection has no plan-apply scenario to probe")
    capacity = AzureCPUCapacity(args.azure_location, args.azure_vm_size, args.azure_image,
        args.azure_os_disk_gib, args.azure_ssh_source, args.azure_private_subnet,
        pricing=AzureComputePricing(args.azure_pricing), tunnel_subnet=args.azure_tunnel_subnet)
    # The lease creates its own private connectivity. Check local prerequisites
    # before cloud mutation; a pre-existing route to an uncreated VNet cannot
    # be a meaningful readiness requirement.
    validate_local_network(capacity.private_subnet, capacity.tunnel_subnet)
    controller_runtime = admit_runtime_image(args.container_image, args.source_revision)
    peer_runtime = admit_runtime_image(args.remote_cpu_image or args.container_image, args.source_revision)
    image = controller_runtime.identity["id"]
    validate_attached_execution(image)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.exists():
        raise ValueError("cross-host report already exists; preserve evidence and choose a new report path")
    artifact_root = Path(tempfile.mkdtemp(prefix="cross-host-", dir=args.report.parent))
    receipt = (args.reuse_azure_lease.resolve(strict=True) if args.reuse_azure_lease
               else artifact_root / "azure-lease.json")
    report = {"schema": 1, "eligible": True, "complete": False,
              "diagnostic_planning_only": args.planning_only,
              "source_revision": args.source_revision, "image": image,
              "remote_cpu_image": peer_runtime.identity["id"],
              "manifest_digest": digest(manifest), "scenarios": [], "all_resources_retired": False,
              "scenario_order": [scenario["id"] for _, scenario in selected],
              "azure_lease_receipt": str(receipt), "infrastructure_failure": None}
    try:
        args.model_ramdisk_root.mkdir(parents=True, exist_ok=True)
        # Capacity is shared across the image's cases; execution membership is
        # not. Each frontend gets only its declared peers in a fresh fleet and
        # hostfile. Idle pool members never enter MPI or satisfy a proof.
        count = max(scenario["topology"]["remote_cpu_hosts"] for _, scenario in selected)
        cli = AzureCLI(args.azure_subscription)
        directory = artifact_root
        with tempfile.TemporaryDirectory(prefix="llaminar-cross-host-", dir=args.model_ramdisk_root) as temp:
            staging_directory = Path(temp)
            parents = [parent for parent, _ in selected]
            # Only weights and the image archive are disposable. HTTP,
            # transport, plan, and cleanup evidence outlive the RAM lease.
            model_dir = stage_models(parents, [], args.ssh_private_key, staging_directory)
            with campaign_resources(cli, receipt, count, Disposal(args.azure_disposal),
                                    capacity, public_key, reuse=args.reuse_azure_lease is not None) as hosts:
                hosts = sorted(hosts, key=lambda host: int(host["name"].removeprefix("cpu-")))
                public = [str(item["public_ip"]) for item in hosts]
                for host in public:
                    wait_remote_runtime(host, args.ssh_private_key)
                # A remote CPU rank can use an ISA-specific sibling of the
                # controller image, but only after both source identity and
                # physical host features have been admitted. This happens
                # before image/model transfer and before an MPI daemon can
                # turn a predictable SIGILL into a generic rank failure.
                admit_remote_cpu_runtime(controller_runtime, peer_runtime, hosts,
                                         args.ssh_private_key, directory)
                with private_mpi_connection(hosts, args.ssh_private_key,
                        capacity.private_subnet, capacity.tunnel_subnet, directory) as connection:
                    args.controller_address = connection.controller_address
                    if not args.planning_only:
                        stage_models(parents, hosts, args.ssh_private_key, staging_directory,
                                     model_dir=model_dir)
                    # AutomaticPlanningStartup opens a GGUF only on discovery
                    # root, then publishes the typed PlanningModelMetadata to
                    # every follower.  A planning-only probe must preserve
                    # that production ownership boundary: transferring a
                    # complete model to remote ranks would both waste a WAN
                    # transfer and mask an accidental follower GGUF read.
                    try:
                        imported = stage_image(peer_runtime.identity["id"], hosts,
                                               args.ssh_private_key, staging_directory)
                    finally:
                        # Import diagnostics must survive a failed admission or
                        # cancellation just as the cloud lease receipt does.
                        for log in staging_directory.glob("runtime-import-*.log"):
                            shutil.copyfile(log, directory / log.name)
                    shutil.copyfile(staging_directory / "runtime-import.json", directory / "runtime-import.json")
                    hosts = [{**host, "runtime_image": imported[host["public_ip"]]} for host in hosts]
                    for case_index, (parent, scenario) in enumerate(selected, start=1):
                        print(f"[production-cross-host] START {case_index}/{len(selected)} {scenario['id']}", flush=True)
                        peers = hosts[:scenario["topology"]["remote_cpu_hosts"]]
                        result = run_case(parent, scenario, controller_runtime, model_dir, peers, args, directory,
                                          planning_only=args.planning_only)
                        report["scenarios"].append(result)
                        write_json(args.report, report)
                        print(f"[production-cross-host] {result['outcome'].upper()} {case_index}/{len(selected)} "
                              f"{scenario['id']} in {result['elapsed_seconds']:.1f}s", flush=True)
                        if result["return_code"]:
                            raise RuntimeError(f"cross-host scenario failed: {scenario['id']}")
            # The single lease has now observed exact Azure retirement.
            for result in report["scenarios"]:
                result["resource_retired"] = True
        report["all_resources_retired"] = True
    except Exception as error:
        # A pre-cell infrastructure failure is not a passing empty campaign.
        # Once a case has emitted its own result, that result is its authority;
        # the enclosing rejection only says that the campaign stopped as asked.
        if not report["scenarios"]:
            report["infrastructure_failure"] = infrastructure_failure(error, args.ssh_private_key)
        raise
    finally:
        # A failed cell and successful cloud retirement are independent facts.
        # Preserve both. This is a snapshot of the canonical owner, not another
        # mutable lease that could restart or stop the same VMs independently.
        if receipt.is_file():
            observed_lease = json.loads(receipt.read_text())
            write_json(artifact_root / "azure-retirement.json", observed_lease)
            report["all_resources_retired"] = observed_lease.get("state") == "retired"
            for result in report["scenarios"]:
                result["resource_retired"] = report["all_resources_retired"]
        report["complete"] = bool(report["all_resources_retired"] and len(report["scenarios"]) == len(selected)
                                   and all(item["return_code"] == 0 for item in report["scenarios"]))
        write_json(args.report, report)
    if not report["complete"]:
        raise RuntimeError("remote E2E did not produce complete retired-resource evidence")
    return 0


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupt_campaign)
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError, TimeoutError) as error:
        print(f"[production-cross-host] ERROR {error}", file=sys.stderr)
        sys.exit(1)
