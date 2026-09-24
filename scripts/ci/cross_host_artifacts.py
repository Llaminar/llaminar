#!/usr/bin/env python3
"""Distribute immutable CI inputs with one WAN upload and private peer copies.

SSH authenticates all control commands; no SSH key leaves the controller. An
attached, stdin-owned source serves exactly one file on its Azure private IP,
only to the declared peer IPs carrying a one-use lease token. It publishes no
directory listing, opens no public firewall rule and dies when its owner closes
stdin. Receivers publish a complete file atomically, never a partial download.
Docker import still authenticates the actual image configuration and layers.
GGUF staging checks complete byte counts and preserved source timestamps. There
is no separate whole-model hashing pass. Unchanged owned replicas are reusable;
changed files publish atomically after a complete authenticated transfer.

The same stdlib-only file executes on peers before the image is installed.
Controller-only SSH helpers are consequently imported at their call boundary.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import os
from pathlib import Path
import select
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import uuid


@dataclass(frozen=True)
class ArtifactEndpoint:
    """Private, lease-scoped read capability and expected complete payload size."""
    url: str
    token: str
    size: int


@dataclass(frozen=True)
class FileStamp:
    """Cheap identity of an owned immutable input, not a cryptographic certificate."""
    size: int
    mtime_ns: int


def file_stamp(path: Path) -> FileStamp | None:
    """Inspect a regular file without following a replaced path or reading bytes."""
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("artifact path must be a regular file, never a symlink or directory")
    return FileStamp(observed.st_size, observed.st_mtime_ns)


def serve(source: Path, address: str, allowed: list[str], token: str) -> None:
    """Serve one stable file until the attached owner's stdin closes.

    Client address and bearer capability are both required. The file handle is
    retained across requests so replacing a path cannot substitute another
    input. The controller waits for all clients before retiring this owner.
    """
    with source.open("rb") as payload:
        size = os.fstat(payload.fileno()).st_size
        if size <= 0 or not token or not allowed:
            raise ValueError("artifact source requires nonempty payload and explicit peers")

        class Handler(BaseHTTPRequestHandler):
            """No path traversal, listing, proxy, redirects or unauthenticated reads."""
            def do_GET(self) -> None:
                """Stream the admitted bytes with a private per-client read offset."""
                if (self.client_address[0] not in allowed or self.path != "/artifact"
                        or not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)):
                    self.send_error(403)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                # pread avoids sharing one seek offset between concurrent
                # peers. Bounded chunks neither materialize nor hash the GGUF.
                offset = 0
                while offset < size:
                    chunk = os.pread(payload.fileno(), min(1024 * 1024, size - offset), offset)
                    if not chunk:
                        raise OSError("artifact source was truncated during distribution")
                    self.wfile.write(chunk)
                    offset += len(chunk)

            def log_message(self, _format: str, *args) -> None:
                """Keep source paths and temporary capabilities out of diagnostics."""

        with ThreadingHTTPServer((address, 0), Handler) as server:
            def retire_on_owner_close() -> None:
                """An SSH disconnect or explicit stop retires the only listener."""
                sys.stdin.buffer.read(1)
                server.shutdown()

            threading.Thread(target=retire_on_owner_close, daemon=True).start()
            print(json.dumps({"port": server.server_port, "size": size}), flush=True)
            server.serve_forever(poll_interval=0.05)


def fetch(endpoint: ArtifactEndpoint, destination: Path, *, replacing: FileStamp | None = None,
          mtime_ns: int | None = None) -> None:
    """Publish a complete private copy, replacing only an explicitly observed file.

    The enclosing Azure lease excludes other writers. A caller without that
    ownership must omit ``replacing``; the default never overwrites any data.
    A timestamp is installed only after complete transfer, before publication.
    """
    if endpoint.size <= 0 or not destination.is_absolute():
        raise ValueError("artifact fetch requires a positive size and absolute destination")
    if file_stamp(destination) != replacing:
        raise FileExistsError(f"artifact destination already exists: {destination}")
    temporary = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
    try:
        class DirectArtifactOnly(HTTPRedirectHandler):
            """Never forward a private capability to a redirected endpoint."""
            def redirect_request(self, request, fp, code, message, headers, newurl):
                """Reject redirects before making another network request."""
                return None

        # This is an exact private network edge. A VM's ambient proxy settings
        # must not redirect model bytes or the lease capability outside it.
        opener = build_opener(ProxyHandler({}), DirectArtifactOnly())
        request = Request(endpoint.url, headers={"Authorization": "Bearer " + endpoint.token})
        with opener.open(request, timeout=30) as response:
            if response.status != 200 or response.geturl() != endpoint.url:
                raise ValueError("artifact source redirected or did not return the admitted payload")
            if response.headers.get("Content-Length") != str(endpoint.size):
                raise ValueError("artifact source byte count differs from the admitted input")
            with temporary.open("xb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if temporary.stat().st_size != endpoint.size:
                raise ValueError("artifact transfer ended before the complete input arrived")
        # Linking within the destination directory is atomic and rejects a
        # competing publication; replace()/rename() would overwrite its bytes.
        if mtime_ns is not None:
            os.utime(temporary, ns=(mtime_ns, mtime_ns))
        if replacing is None:
            os.link(temporary, destination)
        else:
            if file_stamp(destination) != replacing:
                raise ValueError("owned artifact changed during replacement")
            os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def remote_command(peer: dict, key: Path, arguments: list[str]) -> list[str]:
    """Ship this small helper over authenticated SSH, without requiring an image."""
    from cross_host_containers import ssh_argv
    return ssh_argv(peer, str(key), ["python3", "-u", "-c", Path(__file__).read_text(), *arguments])


def remote_stamp(peer: dict, key: Path, path: str) -> FileStamp | None:
    """Read one cheap file observation over SSH before deciding to transfer it."""
    result = subprocess.run(remote_command(peer, key, ["probe", path]),
                            capture_output=True, text=True, timeout=30, check=False)
    if result.returncode:
        raise RuntimeError("remote artifact observation failed: " + result.stderr)
    value = json.loads(result.stdout)
    return FileStamp(**value) if value is not None else None


@contextmanager
def private_source(peer: dict, key: Path, path: str, receivers: list[dict], size: int):
    """Own the source process and fail if readiness, transfer or retirement fails."""
    token = uuid.uuid4().hex
    command = remote_command(peer, key, ["serve", path, peer["private_ip"], token,
                                         *(item["private_ip"] for item in receivers)])
    with tempfile.TemporaryFile(mode="w+t") as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=errors, text=True)
        try:
            if not select.select([process.stdout], [], [], 30)[0]:
                raise TimeoutError("private artifact source did not publish readiness")
            observation = json.loads(process.stdout.readline())
            port = observation.get("port")
            if type(port) is not int or not 0 < port < 65536 or observation.get("size") != size:
                raise ValueError("private artifact source published invalid geometry")
            yield ArtifactEndpoint(f"http://{peer['private_ip']}:{port}/artifact", token, size)
        finally:
            # Closing the retained SSH stdin is the source's termination
            # protocol, including on a failed replica. Never leave a daemon.
            process.stdin.close()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise TimeoutError("private artifact source failed to retire")
            finally:
                process.stdout.close()
            if code:
                errors.seek(0)
                raise RuntimeError("private artifact source failed: " + errors.read())


def distribute(source: Path, destination: str, peers: list[dict], key: Path, upload) -> None:
    """Upload once, then copy concurrently inside the owned Azure private network.

    Each caller has already acquired the destination directories. Every peer
    must finish before inference admission, and a failed copy is fatal. There
    is no WAN retry path. The upload callback is an authenticated incremental
    transport; runtime-image authentication remains with Docker import.
    """
    if not peers:
        raise ValueError("artifact distribution requires at least one owned peer")
    private = [ipaddress.ip_address(peer["private_ip"]) for peer in peers]
    if (any(address.version != 4 or not address.is_private or address.is_loopback or address.is_unspecified
            for address in private) or len(set(private)) != len(private)
            or len({peer["public_ip"] for peer in peers}) != len(peers)):
        raise ValueError("artifact distribution requires distinct private peer identities")
    stamp = file_stamp(source)
    if stamp is None or stamp.size <= 0:
        raise ValueError("artifact distribution requires a nonempty source")
    size = stamp.size
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(peers)) as pool:
        observed = list(pool.map(lambda peer: remote_stamp(peer, key, destination), peers))
    # Any already-matching replica can seed missing peers without WAN traffic.
    # Otherwise one incremental upload repairs the first peer's owned cache.
    seed_index = next((i for i, previous in enumerate(observed) if previous == stamp), 0)
    seed = peers[seed_index]
    if observed[seed_index] != stamp:
        print(f"[production-cross-host] stage {source.name}: {size} bytes, one incremental WAN copy", flush=True)
        upload(seed["public_ip"], key, source, destination)
        if remote_stamp(seed, key, destination) != stamp:
            raise ValueError("WAN artifact publication differs from source metadata")
        print(f"[production-cross-host] WAN copy complete: {source.name} in "
              f"{time.monotonic() - started:.1f}s", flush=True)
    receivers = [peer for i, peer in enumerate(peers) if i != seed_index and observed[i] != stamp]
    previous_by_ip = {peer["private_ip"]: previous for peer, previous in zip(peers, observed)}
    if receivers:
        with private_source(seed, key, destination, receivers, size) as endpoint:
            def copy_to_peer(peer: dict) -> None:
                """Await one remote atomic publication before admitting the input."""
                copy_started = time.monotonic()
                command = remote_command(peer, key, ["fetch", destination, endpoint.url,
                    endpoint.token, str(endpoint.size), "--mtime-ns", str(stamp.mtime_ns),
                    "--replacing", json.dumps(asdict(previous_by_ip[peer["private_ip"]])
                                               if previous_by_ip[peer["private_ip"]] else None)])
                # The command embeds a temporary capability; failure reports
                # must not print the argv. Retain the remote diagnostic only.
                result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, timeout=1800, check=False)
                if result.returncode:
                    raise RuntimeError("private artifact copy failed: " + result.stdout)
                print(f"[production-cross-host] private copy complete: {source.name} -> {peer['name']} "
                      f"in {time.monotonic() - copy_started:.1f}s", flush=True)

            with ThreadPoolExecutor(max_workers=len(receivers)) as pool:
                list(pool.map(copy_to_peer, receivers))
    if file_stamp(source) != stamp:
        raise ValueError("artifact source changed during staging")
    print(f"[production-cross-host] staged {source.name} on {len(peers)} peers in "
          f"{time.monotonic() - started:.1f}s", flush=True)


def main() -> None:
    """Execute only the source or receiver part inside a leased CPU VM."""
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    sender = modes.add_parser("serve")
    sender.add_argument("source", type=Path)
    sender.add_argument("address")
    sender.add_argument("token")
    sender.add_argument("allowed", nargs="+")
    receiver = modes.add_parser("fetch")
    receiver.add_argument("destination", type=Path)
    receiver.add_argument("url")
    receiver.add_argument("token")
    receiver.add_argument("size", type=int)
    receiver.add_argument("--mtime-ns", type=int)
    receiver.add_argument("--replacing", default="null")
    probe = modes.add_parser("probe")
    probe.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "serve":
        serve(args.source, args.address, args.allowed, args.token)
    elif args.mode == "fetch":
        previous = json.loads(args.replacing)
        fetch(ArtifactEndpoint(args.url, args.token, args.size), args.destination,
              replacing=FileStamp(**previous) if previous is not None else None, mtime_ns=args.mtime_ns)
    else:
        observed = file_stamp(args.path)
        print(json.dumps(asdict(observed) if observed is not None else None))


if __name__ == "__main__":
    main()
