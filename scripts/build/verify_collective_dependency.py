"""Admit the same RCCL or NCCL implementation for the server, tests and benchmarks.

Source installations carry the vendor-specific proof already produced by the
repository's dependency builder. A local build against a previously published
runtime can compare its selected ELF directly with that immutable image. This
check runs during configuration and never opens a GPU or executes the DSO.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "scripts/docker"
IMAGE_PATHS = {"rccl": "/usr/local/lib/librccl.so.1", "nccl": "/usr/local/lib/libllaminar_nccl.so.2"}


def dependency_pins(backend: str) -> dict[str, str]:
    """Read the dependency owner's simple upstream version declarations."""
    name = "rocm" if backend == "rccl" else "nccl"
    contents = (DOCKER / f"{name}-release.env").read_text()
    return dict(re.findall(r"^LLAMINAR_(?:RCCL|NCCL)_GIT_(REF|COMMIT)=([\w.\-]+)$", contents, re.M))


def check_image_source(backend: str, revision: str) -> None:
    """Reject even an official older image when its upstream revision or patches differ."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("published image has no immutable source revision")
    def source(path: str) -> str:
        return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=ROOT, text=True)
    owner = f"scripts/docker/{'rocm' if backend == 'rccl' else 'nccl'}-release.env"
    pins = dependency_pins(backend)
    # Earlier published images kept these same upstream pins in their installer
    # and Dockerfile. Their immutable source tree still identifies the version.
    if subprocess.run(["git", "cat-file", "-e", f"{revision}:{owner}"], cwd=ROOT,
                      stderr=subprocess.DEVNULL).returncode == 0:
        declared = re.search(rf"^LLAMINAR_{backend.upper()}_GIT_COMMIT=([0-9a-f]{{40}})$",
                             source(owner), re.M)
    else:
        declared = None
    if declared:
        matched = declared[1] == pins["COMMIT"]
    elif backend == "rccl":
        matched = f'ARG RCCL_GIT_REF={pins["REF"]}' in source("Dockerfile").splitlines()
    else:
        matched = f'nccl_revision={pins["COMMIT"]}' in source("scripts/docker/install-nccl.sh").splitlines()
    if not matched:
        raise ValueError(f"published image source {revision} uses another {backend.upper()} revision")
    paths = [f"scripts/docker/patches/{backend}-*.patch"]
    if backend == "rccl":
        paths.append("scripts/docker/rccl-functions.txt")
    # Compare upstream patch inputs even when the local checkout is uncommitted.
    tracked = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision,
                                       "scripts/docker/patches"], cwd=ROOT, text=True).splitlines()
    current = sorted(path.name for path in (DOCKER / "patches").glob(f"{backend}-*.patch"))
    archived = sorted(Path(path).name for path in tracked if Path(path).name.startswith(backend + "-")
                      and path.endswith(".patch"))
    if current != archived or subprocess.run(["git", "diff", "--quiet", "--no-ext-diff",
                                              revision, "--", *paths], cwd=ROOT).returncode:
        raise ValueError(f"published image source {revision} has a different {backend.upper()} build policy")


def confirm_published(backend: str, library: Path, image: str) -> None:
    """Compare the selected file with the corresponding ELF in a local immutable GHCR image."""
    if not re.fullmatch(r"ghcr\.io/llaminar/llaminar@sha256:[0-9a-f]{64}", image):
        raise ValueError("select the published Llaminar runtime by its GHCR manifest digest")
    revision = subprocess.check_output(["docker", "image", "inspect", "--format",
        '{{index .Config.Labels "org.opencontainers.image.revision"}}', image], text=True).strip()
    check_image_source(backend, revision)
    container = subprocess.check_output(["docker", "create", "--pull=never", "--network=none",
                                         "--entrypoint=/bin/true", image], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{64}", container):
        raise ValueError("docker did not return the identity of the created image filesystem")
    try:
        # Read through a stopped container: logging drivers, image entrypoints and
        # the host GPU runtime cannot affect the bytes returned by the archive API.
        with subprocess.Popen(["docker", "cp", "-L", f"{container}:{IMAGE_PATHS[backend]}", "-"],
                              stdout=subprocess.PIPE) as copy:
            with tarfile.open(fileobj=copy.stdout, mode="r|*") as archive:
                member = archive.next()
                if member is None or not member.isfile():
                    raise ValueError(f"{IMAGE_PATHS[backend]} is missing from {image}")
                with archive.extractfile(member) as library_bytes:
                    expected = hashlib.file_digest(library_bytes, "sha256").hexdigest()
                if archive.next() is not None:
                    raise ValueError("expected a single regular collective library in the image")
            if copy.wait() != 0:
                raise ValueError("docker could not export the published collective library")
    finally:
        subprocess.run(["docker", "rm", container], check=True, stdout=subprocess.DEVNULL)
    with library.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"{library} differs from {IMAGE_PATHS[backend]} in {image}")


def confirm_source(backend: str, library: Path, report: Path, receipt: Path, architectures: str) -> None:
    """Reuse the source/patch receipt and existing byte-bound build proof from the producing toolchain."""
    if not report.is_file() or not receipt.is_file():
        raise ValueError(f"{library} needs its canonical {backend.upper()} build evidence ({report}, {receipt})")
    proof = DOCKER / "native-completed-broadcast/proof.py"
    spec = importlib.util.spec_from_file_location("llaminar_collective_receipt", proof)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_identity = receipt.read_text().strip()
    pins = dependency_pins(backend)
    if backend == "rccl":
        # Architecture alone does not identify the generated RCCL kernels.
        # Both producers bind the same options into their ELF-specific report.
        functions = (DOCKER / "rccl-functions.txt").read_text().strip()
        generated = hashlib.sha256(f"{architectures}\nOFF\n{functions}".encode()).hexdigest()
        tokens = set(re.findall(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])", build_identity))
        patches = {hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in (DOCKER / "patches").glob("rccl-*.patch")}
        if (f'-{pins["COMMIT"]}-smi-' not in build_identity or not patches.issubset(tokens)
                or f"-targets-{generated}-llvm-lto-" not in build_identity):
            raise ValueError(f"{receipt} has a different RCCL revision, patch set or kernel configuration")
    if not module.valid_receipt(backend, build_identity):
        raise ValueError(f"{receipt} does not identify the repository's pinned {backend.upper()} patches")
    command = [sys.executable]
    if backend == "rccl":
        command += [str(DOCKER / "llvm-lto-linker/artifact.py"), "verify", "--architectures",
                    architectures, "--spill-guard", str(ROOT / "scripts/build/gpu_spill_guard.py")]
    else:
        command += [str(DOCKER / "nccl-gin-selection/proof.py"), "verify"]
    subprocess.run([*command, "--library", str(library), "--report", str(report),
                    "--receipt", str(receipt)], check=True)


def main() -> None:
    """Verify a completed dependency before its absolute path is embedded in a loader."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=IMAGE_PATHS)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--published-image")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--architectures", default="")
    args = parser.parse_args()
    if not args.library.is_absolute() or not args.library.is_file():
        raise ValueError(f"collective library must be an existing absolute file: {args.library}")
    if args.published_image:
        confirm_published(args.backend, args.library, args.published_image)
    elif args.report and args.receipt:
        confirm_source(args.backend, args.library, args.report, args.receipt, args.architectures)
    else:
        parser.error("source installations require both --report and --receipt")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"Collective dependency admission failed: {error}")
