#!/usr/bin/env python3
"""Lease an authenticated SSH tunnel into the campaign's private Azure network.

One newly owned CPU peer also routes between the SSH tunnel and its private
subnet. Azure owns the return route and NIC forwarding; the guest enables IP
forwarding. The controller adds only an owned TUN device and its specific route,
and removes both on every exit. No public MPI listener rule or pre-existing VPN
is required, and no host-wide forwarding setting is changed on the controller.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import uuid

from cross_host_containers import execute, ssh_argv


@dataclass(frozen=True)
class PrivateMPIConnection:
    """Observed controller identity on the authenticated private MPI network."""
    controller_address: str


def allow_gateway_forwarding(gateway: dict, key: Path, private_subnet: str,
                             tunnel_subnet: str) -> None:
    """Admit only the owned tunnel/VNet edges through Docker's forwarding policy.

    Docker installs a DROP policy on FORWARD even when guest IP forwarding is
    enabled. Two subnet- and interface-qualified rules admit this campaign's
    bidirectional routed traffic; neither the policy nor unrelated edges change.
    The enclosing disposable VM owns these rules until resource retirement.
    """
    for direction, source, destination in (("-i", tunnel_subnet, private_subnet),
                                            ("-o", private_subnet, tunnel_subnet)):
        execute(ssh_argv(gateway, str(key), ["sudo", "-n", "iptables", "-w", "5",
            "-I", "FORWARD", "1", direction, "tun9", "-s", source, "-d", destination, "-j", "ACCEPT"]))


def validate_local_network(private_subnet: str, tunnel_subnet: str) -> None:
    """Reject overlapping installed routes and unavailable TUN before provisioning.

    A default route is expected and is not a conflict. Explicit routes overlapping
    either fresh network would alias existing infrastructure, so they are fatal.
    The deployment's typed capacity validates private ranges and geometry.
    """
    if not Path("/dev/net/tun").is_char_device():
        raise RuntimeError("cross-host SSH networking requires /dev/net/tun on the controller")
    routes = json.loads(execute(["ip", "-j", "route", "show"]))
    requested = [ipaddress.ip_network(private_subnet), ipaddress.ip_network(tunnel_subnet)]
    for route in routes:
        destination = route.get("dst", "default")
        if destination == "default":
            continue
        existing = ipaddress.ip_network(destination, strict=False)
        if any(existing.overlaps(network) for network in requested):
            raise ValueError(f"cross-host network overlaps an installed controller route: {destination}")
    execute(["sudo", "-n", "ip", "tuntap", "show"])


@contextmanager
def remote_tunnel_endpoint(gateway: dict, key: Path):
    """Create the guest endpoint per network lease, not once during first boot.

    TUN devices do not survive a stopped VM restarting. Acquisition refuses an
    existing interface, and only a successful acquisition permits its deletion.
    The attached SSH tunnel must stop before this scope retires its endpoint.
    """
    execute(ssh_argv(gateway, str(key), ["sudo", "-n", "ip", "tuntap", "add", "dev", "tun9",
                                       "mode", "tun", "user", "llaminar"]))
    try:
        yield
    finally:
        execute(ssh_argv(gateway, str(key), ["sudo", "-n", "ip", "link", "delete", "dev", "tun9"]))


@contextmanager
def private_mpi_connection(hosts: list[dict], key: Path, private_subnet: str,
                           tunnel_subnet: str, directory: Path):
    """Keep the exact SSH process and TUN alive until all MPI containers retire.

    The subnet route is attached to the owned interface and disappears with it.
    We never replace a route or adopt an existing TUN. Cloud cleanup owns the
    newly configured guest; receipt recovery owns interrupted cloud resources.
    """
    if not hosts:
        raise ValueError("private MPI transport requires an owned gateway peer")
    gateway = next((host for host in hosts if host["name"] == "cpu-0"), None)
    if gateway is None:
        raise ValueError("owned Azure deployment omitted its gateway CPU peer")
    network = ipaddress.ip_network(tunnel_subnet)
    controller, peer = str(network.network_address + 1), str(network.network_address + 2)
    # Linux TUN names include the numeric ID consumed by ssh -w. A fresh name
    # is acquired with 'add', whose EEXIST is fatal; no existing link is reset.
    number = 10000 + int(uuid.uuid4().hex[:6], 16)
    device = f"tun{number}"
    execute(["sudo", "-n", "ip", "tuntap", "add", "dev", device,
             "mode", "tun", "user", str(os.getuid())])
    process = None
    primary = None
    endpoints = ExitStack()
    try:
        endpoints.enter_context(remote_tunnel_endpoint(gateway, key))
        execute(["sudo", "-n", "ip", "address", "add", f"{controller}/{network.prefixlen}", "dev", device])
        execute(["sudo", "-n", "ip", "link", "set", "dev", device, "up"])
        execute(["sudo", "-n", "ip", "route", "add", private_subnet,
                 "via", peer, "dev", device, "src", controller])
        for command in (["sudo", "-n", "ip", "address", "add", f"{peer}/{network.prefixlen}", "dev", "tun9"],
                        ["sudo", "-n", "ip", "link", "set", "dev", "tun9", "up"],
                        ["sudo", "-n", "sysctl", "-w", "net.ipv4.ip_forward=1"]):
            execute(ssh_argv(gateway, str(key), command))
        allow_gateway_forwarding(gateway, key, private_subnet, tunnel_subnet)
        argv = ssh_argv(gateway, str(key), [])
        # Replace the empty remote command with a retained tunnel-only session.
        argv = argv[:-2] + ["-N", "-T", "-o", "ExitOnForwardFailure=yes",
                           "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=3",
                           "-w", f"{number}:9", argv[-2]]
        with (directory / "ssh-tunnel.log").open("w") as log:
            process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError("authenticated MPI tunnel exited; inspect ssh-tunnel.log")
                try:
                    # This connection must traverse the newly installed private
                    # route. SSH on the VM is a bounded bidirectional witness.
                    for host in hosts:
                        with socket.create_connection((host["private_ip"], 22), timeout=1) as probe:
                            if not probe.recv(128).startswith(b"SSH-"):
                                raise RuntimeError("private CPU peer did not return an SSH banner")
                    break
                except (TimeoutError, ConnectionError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("private MPI tunnel failed its 30-second connectivity probe")
                    time.sleep(0.1)
            yield PrivateMPIConnection(controller)
            if process.poll() is not None:
                raise RuntimeError("authenticated MPI tunnel failed during certification")
    except BaseException as error:
        primary = error
        raise
    finally:
        errors = []
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            except BaseException as error:
                errors.append(error)
        try:
            endpoints.close()
        except BaseException as error:
            errors.append(error)
        try:
            execute(["sudo", "-n", "ip", "link", "delete", "dev", device])
        except BaseException as error:
            errors.append(error)
        if errors:
            raise BaseExceptionGroup("private MPI tunnel cleanup failed", ([primary] if primary else []) + errors)
