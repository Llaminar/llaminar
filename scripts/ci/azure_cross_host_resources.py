#!/usr/bin/env python3
"""Owned Azure resources for cross-host production E2E, not a test matrix.

The caller supplies infrastructure capacity after selecting canonical cases.
This module owns a fresh resource group, its bounded deployment, and verified
retirement. It never logs in, chooses models/topologies, starts inference, or
issues certificates. Unowned Azure resources cannot be adopted or deleted.
An explicitly selected, retired retain-disks lease can restart its own peers;
the original receipt remains the single owner across diagnostic iterations.
Local Azure CLI login and a CI federated login use the same credential boundary;
credentials are neither read into Python nor written to the cleanup receipt.

The receipt is written before the first mutation, so an interrupted controller
can retire the exact owned group. Provider-side daily shutdown limits the cost
of a lost controller after deployment; it is not an exact TTL or a substitute
for the caller's cancellation handler and CI always-run cleanup step.
"""
from __future__ import annotations

import base64
import binascii
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Iterator
import uuid

from production_artifacts import write_json


PURPOSE = "llaminar-cross-host-e2e"
GROUP_PREFIX = "rg-llaminar-e2e-"
DEFAULT_TUNNEL_SUBNET = "10.207.14.0/30"


class Disposal(str, Enum):
    """Retention intent is sealed before allocation, never inferred at cleanup."""
    DELETE = "delete-owned-group"
    DEALLOCATE = "deallocate-retain-disks"


class LeaseState(str, Enum):
    """A receipt distinguishes interrupted acquisition from completed retirement."""
    ALLOCATING = "allocating"
    DEPLOYING = "deploying"
    STARTING = "starting"
    READY = "ready"
    RETIRING = "retiring"
    RETIRED = "retired"


@dataclass(frozen=True)
class AzureCPUCapacity:
    """Cloud infrastructure only; observed CPU ISA/RAM still need runtime admission.

    The image must be a pinned marketplace URN, not ``latest``. SSH is admitted
    only from one explicit public controller address; MPI ports are never
    opened to the Internet. CPU hosts communicate over the private subnet or
    the caller's authenticated tunnel. A SKU is not proof of its actual ISA.
    """
    location: str
    vm_size: str
    image: str
    os_disk_gib: int
    ssh_source: str
    private_subnet: str
    shutdown_after_minutes: int = 120
    tunnel_subnet: str = DEFAULT_TUNNEL_SUBNET

    def __post_init__(self) -> None:
        """Reject ambiguous/unsafe deployment inputs before any Azure mutation."""
        if not re.fullmatch(r"[a-z][a-z0-9]+", self.location):
            raise ValueError("Azure location must be an explicit region name")
        if not re.fullmatch(r"Standard_[A-Za-z0-9_]+", self.vm_size):
            raise ValueError("Azure VM size must be explicit")
        if not re.fullmatch(r"[A-Za-z0-9._-]+:[A-Za-z0-9._-]+:[A-Za-z0-9._-]+:[0-9][A-Za-z0-9._-]*", self.image):
            raise ValueError("Azure image must be a pinned publisher:offer:sku:version URN")
        if type(self.os_disk_gib) is not int or self.os_disk_gib <= 0:
            raise ValueError("Azure OS disk capacity must be a positive integer")
        source = ipaddress.ip_network(self.ssh_source, strict=True)
        if (source.version != 4 or source.prefixlen != 32 or not source.network_address.is_global
                or str(source) != self.ssh_source):
            raise ValueError("SSH requires one canonical public controller IPv4 /32, never a network range")
        subnet = ipaddress.ip_network(self.private_subnet, strict=True)
        private_ranges = (ipaddress.ip_network("10.0.0.0/8"), ipaddress.ip_network("172.16.0.0/12"),
                          ipaddress.ip_network("192.168.0.0/16"))
        if (subnet.version != 4 or str(subnet) != self.private_subnet
                or not any(subnet.subnet_of(parent) for parent in private_ranges)):
            raise ValueError("CPU peer subnet must be an explicit canonical RFC1918 IPv4 network")
        tunnel = ipaddress.ip_network(self.tunnel_subnet, strict=True)
        if (tunnel.version != 4 or tunnel.prefixlen != 30 or str(tunnel) != self.tunnel_subnet
                or not any(tunnel.subnet_of(parent) for parent in private_ranges)
                or tunnel.overlaps(subnet)):
            raise ValueError("SSH tunnel requires a distinct canonical private IPv4 /30")
        if (type(self.shutdown_after_minutes) is not int
                or not 10 <= self.shutdown_after_minutes <= 720):
            raise ValueError("Azure shutdown backstop must be between 10 minutes and 12 hours")


class AzureCLI:
    """Small subprocess boundary; every command selects its admitted subscription."""

    def __init__(self, subscription: str):
        """Bind a canonical subscription without mutating Azure CLI defaults."""
        if str(uuid.UUID(subscription)) != subscription:
            raise ValueError("Azure subscription must be a canonical UUID")
        self.subscription = subscription

    def call(self, *arguments: str):
        """Execute one bounded CLI request without shell expansion or token output.

        Long provider operations must use --no-wait and be polled separately.
        Do not embed command arguments in errors: future deployment parameters
        may contain data inappropriate for a shared console or CI artifact.
        """
        result = subprocess.run(
            ["az", *arguments, "--subscription", self.subscription,
             "--only-show-errors", "--output", "json"],
            capture_output=True, text=True, timeout=60, check=False)
        if result.returncode:
            # Azure may reject a deployment locally, before an activity-log
            # entry exists. Preserve its structured error codes, not arbitrary
            # provider messages or command payloads that could contain secrets.
            codes = sorted(set(re.findall(r'"code"\s*:\s*"([A-Za-z][A-Za-z0-9]{0,79})"',
                                          result.stderr or "")))
            raise RuntimeError(f"Azure CLI {arguments[0]} {arguments[1]} failed "
                               f"(exit {result.returncode}, codes={','.join(codes) or 'unavailable'}); "
                               "inspect the local Azure CLI command log or Azure activity logs")
        return json.loads(result.stdout) if result.stdout.strip() else None

    def require_login(self) -> None:
        """Fail before provisioning if local or CI login cannot access this subscription."""
        account = self.call("account", "show")
        if not isinstance(account, dict) or account.get("id") != self.subscription or account.get("state") != "Enabled":
            raise ValueError("Azure login does not admit the requested subscription")


def remote_bootstrap_cloud_init() -> str:
    """Return the minimal secret-free bootstrap for a fresh CPU peer.

    The deployment deliberately uses a pinned Ubuntu image rather than a
    mutable project image.  Cloud-init installs only the runtimes required by
    the production container launcher; model/image bytes and credentials are
    transferred later over the authenticated SSH boundary.
    """
    payload = """#cloud-config
package_update: true
packages:
  - docker.io
  - openssh-client
  - iproute2
  - python3
  - rsync
write_files:
  - path: /etc/ssh/sshd_config.d/90-llaminar-mpi-tunnel.conf
    permissions: '0600'
    content: |
      PermitTunnel point-to-point
runcmd:
  - [systemctl, enable, --now, docker]
  - [usermod, -aG, docker, llaminar]
  - [install, -d, -m, '0755', -o, llaminar, -g, llaminar, /opt/llaminar-models]
  - [systemctl, reload, ssh]
"""
    return base64.b64encode(payload.encode()).decode()


def deployment_template(capacity: AzureCPUCapacity, count: int, public_key: str,
                        tags: dict[str, str], shutdown_at: datetime) -> dict:
    """Build one atomic ARM deployment description with per-VM shutdown schedules.

    The resource group is owned separately so partial deployment can be retired
    even when Azure never returns VM output. There is no independently maintained
    VM inventory: both deployment outputs and cleanup enumerate this same count.
    All networking is inside the owned group. The NSG admits SSH from the exact
    controller address; Azure's private VirtualNetwork rule carries private peer
    traffic. It has no public MPI, HTTP, RDP or wildcard-SSH rule.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("remote CPU host count must be a positive integer")
    if (not isinstance(public_key, str) or "\n" in public_key or "\r" in public_key
            or not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/]+={0,2}(?: [^\r\n]+)?", public_key)):
        raise ValueError("deployment requires a single public Ed25519 key, never a private key")
    try:
        wire_key = base64.b64decode(public_key.split()[1], validate=True)
    except binascii.Error as error:
        raise ValueError("malformed public SSH key") from error
    if len(wire_key) != 51 or wire_key[:19] != b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20":
        raise ValueError("malformed public Ed25519 key payload")
    # Azure reserves five addresses in each subnet. This is network capacity,
    # not a second CPU/GPU memory accounting system or inferred test topology.
    if count > ipaddress.ip_network(capacity.private_subnet).num_addresses - 5:
        raise ValueError("CPU peer subnet cannot contain the selected remote hosts")
    if shutdown_at.tzinfo is None:
        raise ValueError("provider shutdown time must have an explicit timezone")
    network_api, compute_api = "2023-09-01", "2024-07-01"
    gateway_ip = str(ipaddress.ip_network(capacity.private_subnet).network_address + 4)
    resources = [{
        "type": "Microsoft.Network/networkSecurityGroups", "apiVersion": network_api,
        "name": "cpu-peers", "location": capacity.location, "tags": tags,
        "properties": {"securityRules": [{"name": "controller-ssh", "properties": {
            "priority": 100, "direction": "Inbound", "access": "Allow", "protocol": "Tcp",
            "sourceAddressPrefix": capacity.ssh_source, "sourcePortRange": "*",
            "destinationAddressPrefix": "*", "destinationPortRange": "22"}}, {
            "name": "authenticated-controller", "properties": {
                "priority": 110, "direction": "Inbound", "access": "Allow", "protocol": "*",
                "sourceAddressPrefix": capacity.tunnel_subnet, "sourcePortRange": "*",
                "destinationAddressPrefix": capacity.private_subnet, "destinationPortRange": "*"}}]}}, {
        "type": "Microsoft.Network/routeTables", "apiVersion": network_api,
        "name": "controller-return", "location": capacity.location, "tags": tags,
        "properties": {"routes": [{"name": "ssh-controller", "properties": {
            "addressPrefix": capacity.tunnel_subnet, "nextHopType": "VirtualAppliance",
            "nextHopIpAddress": gateway_ip}}]}}, {
        "type": "Microsoft.Network/virtualNetworks", "apiVersion": network_api,
        "name": "cpu-peers", "location": capacity.location, "tags": tags,
        "dependsOn": ["[resourceId('Microsoft.Network/networkSecurityGroups', 'cpu-peers')]",
                      "[resourceId('Microsoft.Network/routeTables', 'controller-return')]"],
        "properties": {"addressSpace": {"addressPrefixes": [capacity.private_subnet]}, "subnets": [{
            "name": "peers", "properties": {"addressPrefix": capacity.private_subnet,
                "routeTable": {"id": "[resourceId('Microsoft.Network/routeTables', 'controller-return')]"},
                "networkSecurityGroup": {"id": "[resourceId('Microsoft.Network/networkSecurityGroups', 'cpu-peers')]"}}}]}}]
    publisher, offer, sku, version = capacity.image.split(":")
    bootstrap = remote_bootstrap_cloud_init()
    outputs = []
    for index in range(count):
        name = f"cpu-{index}"
        vm_id = f"[resourceId('Microsoft.Compute/virtualMachines', '{name}')]"
        nic_id = f"[resourceId('Microsoft.Network/networkInterfaces', '{name}')]"
        public_id = f"[resourceId('Microsoft.Network/publicIPAddresses', '{name}')]"
        resources.extend([{
            "type": "Microsoft.Network/publicIPAddresses", "apiVersion": network_api,
            "name": name, "location": capacity.location, "tags": tags,
            "sku": {"name": "Standard"}, "properties": {"publicIPAllocationMethod": "Static"}}, {
            "type": "Microsoft.Network/networkInterfaces", "apiVersion": network_api,
            "name": name, "location": capacity.location, "tags": tags,
            "dependsOn": [public_id, "[resourceId('Microsoft.Network/virtualNetworks', 'cpu-peers')]"],
            "properties": {"enableIPForwarding": index == 0,
                "ipConfigurations": [{"name": "primary", "properties": {
                "privateIPAllocationMethod": "Static",
                "privateIPAddress": str(ipaddress.ip_network(capacity.private_subnet).network_address + 4 + index),
                "publicIPAddress": {"id": public_id},
                "subnet": {"id": "[resourceId('Microsoft.Network/virtualNetworks/subnets', 'cpu-peers', 'peers')]"}}}]}}, {
            "type": "Microsoft.Compute/virtualMachines", "apiVersion": compute_api,
            "name": name, "location": capacity.location, "tags": tags, "dependsOn": [nic_id],
            "properties": {"hardwareProfile": {"vmSize": capacity.vm_size},
                "storageProfile": {"imageReference": {"publisher": publisher, "offer": offer,
                    "sku": sku, "version": version}, "osDisk": {"createOption": "FromImage",
                    "diskSizeGB": capacity.os_disk_gib, "managedDisk": {"storageAccountType": "StandardSSD_LRS"}}},
                "osProfile": {"computerName": name, "adminUsername": "llaminar",
                    "customData": bootstrap,
                    "linuxConfiguration": {"disablePasswordAuthentication": True,
                        "ssh": {"publicKeys": [{"path": "/home/llaminar/.ssh/authorized_keys", "keyData": public_key}]}}},
                "networkProfile": {"networkInterfaces": [{"id": nic_id}]}}}, {
            "type": "Microsoft.DevTestLab/schedules", "apiVersion": "2018-09-15",
            "name": f"shutdown-computevm-{name}", "location": capacity.location,
            "tags": tags, "dependsOn": [vm_id], "properties": {
                "status": "Enabled", "taskType": "ComputeVmShutdownTask",
                "dailyRecurrence": {"time": shutdown_at.astimezone(timezone.utc).strftime("%H%M")},
                "timeZoneId": "UTC", "targetResourceId": vm_id,
                "notificationSettings": {"status": "Disabled"}}}])
        outputs.append({"name": name, "id": vm_id,
            "public_ip": f"[reference(resourceId('Microsoft.Network/publicIPAddresses', '{name}'), '{network_api}').ipAddress]",
            "private_ip": f"[reference(resourceId('Microsoft.Network/networkInterfaces', '{name}'), '{network_api}').ipConfigurations[0].properties.privateIPAddress]"})
    return {"$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0", "resources": resources,
            "outputs": {"cpu_hosts": {"type": "array", "value": outputs}}}


class AzureCampaignLease:
    """One durable owner for a fresh group, including partially created resources.

    This is intentionally not a generic cloud-resource adopter. Receipt replay
    may only act on a group whose generated name, subscription, purpose and
    lease tag all match. A different or missing owner is a hard failure, never
    permission to run a broad delete or best-effort cleanup against user data.
    """

    def __init__(self, cli: AzureCLI, receipt: Path, document: dict):
        """Validate the cleanup capability before retaining any resource identity."""
        self.cli, self.receipt = cli, receipt
        lease_id = document.get("lease_id")
        if not isinstance(lease_id, str) or not re.fullmatch(r"[0-9a-f]{32}", lease_id):
            raise ValueError("invalid Azure campaign lease identity")
        group = GROUP_PREFIX + lease_id
        group_id = f"/subscriptions/{cli.subscription}/resourceGroups/{group}"
        if (document.get("schema") != 1 or document.get("subscription") != cli.subscription
                or document.get("group") != group or document.get("group_id") != group_id
                or type(document.get("host_count")) is not int or document["host_count"] <= 0):
            raise ValueError("Azure cleanup receipt has an invalid resource scope")
        Disposal(document["disposal"])
        LeaseState(document["state"])
        self.document = dict(document)

    @classmethod
    def prepare(cls, cli: AzureCLI, receipt: Path, count: int, disposal: Disposal) -> AzureCampaignLease:
        """Persist exact cleanup scope before any allocation; refuse receipt overwrite."""
        if not isinstance(disposal, Disposal) or type(count) is not int or count <= 0:
            raise ValueError("Azure lease requires typed disposal and positive host count")
        if receipt.exists():
            raise ValueError("Azure lease receipt already exists; retire it rather than overwrite it")
        cli.require_login()
        lease_id = uuid.uuid4().hex
        group = GROUP_PREFIX + lease_id
        if cli.call("group", "exists", "--name", group) is not False:
            raise ValueError("refusing to adopt an existing Azure resource group")
        document = {"schema": 1, "lease_id": lease_id, "subscription": cli.subscription,
            "group": group, "group_id": f"/subscriptions/{cli.subscription}/resourceGroups/{group}",
            "host_count": count, "disposal": disposal.value, "state": LeaseState.ALLOCATING.value}
        # A separate per-campaign output directory is owned by the caller. The
        # creation below is exclusive: two workers cannot clobber a cleanup ID.
        receipt.parent.mkdir(parents=True, exist_ok=True)
        with receipt.open("x") as stream:
            json.dump(document, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A power loss must not leave paid resources whose cleanup identity
        # existed only in the page cache. Persist both file data and its new
        # directory entry before the first cloud allocation is permitted.
        parent = os.open(receipt.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        return cls(cli, receipt, document)

    def _publish(self, state: LeaseState, **evidence) -> None:
        """Atomically record lifecycle progress before returning it to a caller."""
        self.document = {**self.document, **evidence, "state": state.value}
        write_json(self.receipt, self.document)

    def _owned_group(self) -> bool:
        """Authenticate the exact target afresh before any cleanup mutation."""
        group = self.document["group"]
        exists = self.cli.call("group", "exists", "--name", group)
        if exists is False:
            return False
        if exists is not True:
            raise ValueError("Azure did not return definite resource existence")
        observed = self.cli.call("group", "show", "--name", group)
        expected_tags = {"purpose": PURPOSE, "llaminar-lease": self.document["lease_id"],
                         "llaminar-disposal": self.document["disposal"]}
        if (observed.get("id", "").lower() != self.document["group_id"].lower()
                or any(observed.get("tags", {}).get(key) != value for key, value in expected_tags.items())):
            raise ValueError("Azure group ownership changed; refusing cleanup mutation")
        return True

    def _poll(self, observe, complete, description: str, timeout: int) -> object:
        """Poll bounded provider operations with periodic human-visible progress."""
        deadline, next_progress = time.monotonic() + timeout, 0.0
        while True:
            value = observe()
            if complete(value):
                return value
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(f"Azure {description} exceeded its infrastructure deadline; receipt={self.receipt}")
            if now >= next_progress:
                print(f"[production-cross-host] waiting for {description}; receipt={self.receipt}", flush=True)
                next_progress = now + 30
            time.sleep(min(5, deadline - now))

    def provision(self, capacity: AzureCPUCapacity, public_key: str) -> list[dict]:
        """Deploy CPU peers and shutdown schedules; failures remain owned by the lease."""
        if self.document["state"] != LeaseState.ALLOCATING.value:
            raise ValueError("Azure lease can only provision once, from allocating state")
        expires = datetime.now(timezone.utc) + timedelta(minutes=capacity.shutdown_after_minutes)
        tags = {"purpose": PURPOSE, "llaminar-lease": self.document["lease_id"],
                "llaminar-disposal": self.document["disposal"], "expires-at": expires.isoformat()}
        template = deployment_template(capacity, self.document["host_count"], public_key, tags, expires)
        # Validate the complete template before creating the resource group.
        # No private key, cloud token, model or executable enters this document.
        path = self.receipt.with_name(f"azure-deployment-{self.document['lease_id']}.json")
        write_json(path, template)
        group = self.document["group"]
        # Provisioning is not resumable by re-running an upsert. In particular,
        # an old receipt must not overwrite somebody else's newly adopted group.
        if self.cli.call("group", "exists", "--name", group) is not False:
            raise ValueError("Azure target appeared before acquisition; refusing group upsert")
        self._publish(LeaseState.DEPLOYING, expires_at=expires.isoformat())
        self.cli.call("group", "create", "--name", group, "--location", capacity.location,
                      "--tags", *(f"{key}={value}" for key, value in tags.items()))
        if not self._owned_group():
            raise ValueError("Azure group vanished during acquisition")
        self.cli.call("deployment", "group", "create", "--resource-group", group,
                      "--name", "cpu-peers", "--template-file", str(path), "--no-wait")

        def completed(value):
            """Treat terminal provider errors as failure, never an empty host pool."""
            state = value["properties"]["provisioningState"]
            if state in ("Failed", "Canceled"):
                raise RuntimeError(f"Azure CPU deployment {state}; receipt={self.receipt}")
            return state == "Succeeded"

        result = self._poll(lambda: self.cli.call("deployment", "group", "show",
            "--resource-group", group, "--name", "cpu-peers"), completed, "CPU deployment", 1200)
        hosts = result["properties"]["outputs"]["cpu_hosts"]["value"]
        expected = {f"cpu-{index}" for index in range(self.document["host_count"])}
        if not isinstance(hosts, list) or len(hosts) != len(expected) or {host["name"] for host in hosts} != expected:
            raise ValueError("Azure deployment omitted or duplicated a CPU host")
        for host in hosts:
            expected_id = self.document["group_id"] + "/providers/Microsoft.Compute/virtualMachines/" + host["name"]
            if host["id"].lower() != expected_id.lower():
                raise ValueError("Azure deployment returned a foreign VM identity")
            for key in ("public_ip", "private_ip"):
                ipaddress.ip_address(host[key])
            if (not ipaddress.ip_address(host["public_ip"]).is_global
                    or ipaddress.ip_address(host["private_ip"]) not in ipaddress.ip_network(capacity.private_subnet)):
                raise ValueError("Azure CPU addresses do not belong to the admitted public/private networks")
        for key in ("public_ip", "private_ip"):
            if len({host[key] for host in hosts}) != len(hosts):
                raise ValueError("Azure deployment returned duplicate CPU host addresses")
        self._publish(LeaseState.READY, hosts=hosts, expires_at=expires.isoformat())
        return hosts

    def restart(self, capacity: AzureCPUCapacity, public_key: str) -> list[dict]:
        """Restart only a complete stopped pool, renewing shutdown before any start.

        Retained disks are an infrastructure cache, not reusable test evidence.
        Revalidate the original deployment intent and every live VM's identity,
        hardware, addresses and provider power state. A mismatch fails before
        mutation. Once STARTING is durable, the enclosing owner must retire even
        a partially started pool. All start submissions precede completion waits.
        """
        if (self.document["state"] != LeaseState.RETIRED.value
                or self.document["disposal"] != Disposal.DEALLOCATE.value
                or self.document.get("retirement") != "vms-deallocated-disks-retained"):
            raise ValueError("reuse requires a retired retain-disks Azure lease")
        self.cli.require_login()
        if not self._owned_group():
            raise ValueError("retained Azure group no longer exists")
        previous = json.loads(self.receipt.with_name(
            f"azure-deployment-{self.document['lease_id']}.json").read_text())
        expires = datetime.now(timezone.utc) + timedelta(minutes=capacity.shutdown_after_minutes)
        requested = deployment_template(capacity, self.document["host_count"], public_key, {}, expires)

        def deployment_intent(template: dict) -> list[dict]:
            """Exclude bootstrap implementation and expired schedule, not user intent."""
            rows = json.loads(json.dumps(template["resources"]))
            result = []
            for row in rows:
                row.pop("tags", None)
                if row["type"] == "Microsoft.DevTestLab/schedules":
                    continue
                if row["type"] == "Microsoft.Compute/virtualMachines":
                    row["properties"]["osProfile"].pop("customData", None)
                result.append(row)
            return result

        if deployment_intent(previous) != deployment_intent(requested):
            raise ValueError("retained Azure deployment differs from requested capacity/network/SSH identity")
        hosts = self.document.get("hosts", [])
        expected_ids = {self.document["group_id"] + "/providers/Microsoft.Compute/virtualMachines/cpu-" + str(index)
                        for index in range(self.document["host_count"])}
        if (len(hosts) != len(expected_ids) or {host["id"] for host in hosts} != expected_ids
                or set(self.document.get("retired_vm_ids", [])) != expected_ids):
            raise ValueError("retained Azure receipt has incomplete VM membership")
        observed = self.cli.call("vm", "list", "--resource-group", self.document["group"], "--show-details")
        if len(observed) != len(hosts) or {host["id"].lower() for host in observed} != {ident.lower() for ident in expected_ids}:
            raise ValueError("retained Azure pool changed membership")
        by_id = {host["id"].lower(): host for host in hosts}
        publisher, offer, sku, version = capacity.image.split(":")
        for vm in observed:
            original = by_id[vm["id"].lower()]
            tags = vm.get("tags", {})
            storage = vm.get("storageProfile", {})
            image = storage.get("imageReference", {})
            disk_id = storage.get("osDisk", {}).get("managedDisk", {}).get("id", "")
            if not disk_id.lower().startswith((self.document["group_id"] + "/providers/Microsoft.Compute/disks/").lower()):
                raise ValueError("retained Azure VM no longer uses its owned managed OS disk")
            # VM list omits managed disk size; the disk resource owns it.
            disk = self.cli.call("disk", "show", "--ids", disk_id)
            if (any(tags.get(key) != value for key, value in {
                    "purpose": PURPOSE, "llaminar-lease": self.document["lease_id"],
                    "llaminar-disposal": Disposal.DEALLOCATE.value}.items())
                    or vm.get("name") != original["name"] or vm.get("powerState") != "VM deallocated"
                    or vm.get("location") != capacity.location
                    or vm.get("hardwareProfile", {}).get("vmSize") != capacity.vm_size
                    or disk.get("diskSizeGB") != capacity.os_disk_gib
                    or any(image.get(key) != value for key, value in {
                        "publisher": publisher, "offer": offer, "sku": sku, "version": version}.items())
                    or vm.get("publicIps") != original["public_ip"]
                    or vm.get("privateIps") != original["private_ip"]):
                raise ValueError("retained Azure VM changed ownership, capacity, addresses or stopped state")
        # No mutations precede this publication. Rejected reuse must not stop
        # another active controller's pool, whereas a partial restart is ours.
        self._publish(LeaseState.STARTING, expires_at=expires.isoformat(),
                      retirement=None, retired_vm_ids=[], reuse_count=self.document.get("reuse_count", 0) + 1)
        for host in hosts:
            self.cli.call("vm", "auto-shutdown", "--ids", host["id"],
                          "--time", expires.strftime("%H%M"))
        for host in hosts:
            self.cli.call("vm", "start", "--ids", host["id"], "--no-wait")
        for host in hosts:
            self._poll(lambda: self.cli.call("vm", "get-instance-view", "--ids", host["id"]),
                lambda value: any(row.get("code") == "PowerState/running"
                                 for row in value.get("instanceView", {}).get("statuses", [])),
                "retained CPU startup", 600)
        self._publish(LeaseState.READY)
        return hosts

    def retire(self) -> None:
        """Verify deletion or deallocation; a pending Azure request is not retirement.

        Group deletion covers partially created NICs/disks/deployments as well
        as VMs. Retain mode first cancels unfinished deployment, submits every
        deallocation without serial waits, and only then observes completion.
        A cleanup error deliberately leaves a non-retired durable receipt.
        """
        if not self._owned_group():
            self._publish(LeaseState.RETIRED, retirement="group-absent")
            return
        self._publish(LeaseState.RETIRING)
        group = self.document["group"]
        if Disposal(self.document["disposal"]) is Disposal.DELETE:
            self.cli.call("group", "delete", "--name", group, "--yes", "--no-wait")
            self._poll(lambda: self.cli.call("group", "exists", "--name", group),
                       lambda value: value is False, "owned resource deletion", 600)
            self._publish(LeaseState.RETIRED, retirement="group-deleted")
            return
        # A late deployment must not start a VM after we observed it stopped.
        # Cancel before enumerating real resources; do not assume failed output
        # contains every partially provisioned VM.
        deployments = self.cli.call("deployment", "group", "list", "--resource-group", group)
        for deployment in deployments:
            if deployment["properties"]["provisioningState"] not in ("Succeeded", "Failed", "Canceled"):
                self.cli.call("deployment", "group", "cancel", "--resource-group", group,
                              "--name", deployment["name"])
                self._poll(lambda: self.cli.call("deployment", "group", "show", "--resource-group", group,
                    "--name", deployment["name"]),
                    lambda value: value["properties"]["provisioningState"] in ("Succeeded", "Failed", "Canceled"),
                    "deployment cancellation", 600)
        hosts = self.cli.call("vm", "list", "--resource-group", group)
        ids = []
        for host in hosts:
            name = host["name"]
            expected = self.document["group_id"] + "/providers/Microsoft.Compute/virtualMachines/" + name
            if (name not in {f"cpu-{index}" for index in range(self.document["host_count"])}
                    or host["id"].lower() != expected.lower()
                    or host.get("tags", {}).get("llaminar-lease") != self.document["lease_id"]):
                raise ValueError("Azure lease contains a foreign VM; refusing deallocation")
            ids.append(host["id"])
        errors = []
        submitted = []
        for vm_id in ids:
            try:
                self.cli.call("vm", "deallocate", "--ids", vm_id, "--no-wait")
                submitted.append(vm_id)
            except Exception as error:
                # Still retire the other peers when a single API submission
                # fails. Do not let one failure strand every remaining VM.
                errors.append(error)
        for vm_id in submitted:
            try:
                self._poll(lambda: self.cli.call("vm", "get-instance-view", "--ids", vm_id),
                    lambda value: any(row.get("code") == "PowerState/deallocated"
                                     for row in value.get("instanceView", {}).get("statuses", [])),
                    "CPU deallocation", 600)
            except Exception as error:
                errors.append(error)
        if deployments and len(ids) != self.document["host_count"]:
            # ARM cancellation does not roll back in-flight resource creation.
            # We have stopped every observed peer, but cannot attest that a
            # missing submitted VM will not appear later. Preserve the receipt
            # and fail cleanup instead of treating an empty list as a proof.
            errors.append(RuntimeError(
                "Azure retained deployment has incomplete VM membership; "
                "re-run receipt cleanup after provider operations settle"))
        if errors:
            raise ExceptionGroup("Azure CPU retirement is incomplete", errors)
        self._publish(LeaseState.RETIRED, retirement="vms-deallocated-disks-retained", retired_vm_ids=ids)


@contextmanager
def receipt_owner(receipt: Path):
    """Exclude concurrent runners/cleanup without locking an atomically replaced inode.

    The stable sibling lock survives receipt publication and process crashes.
    Never delete it: unlinking a lock could give a later process a second inode
    while the original owner is still running. Kernel close releases ownership.
    """
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.with_suffix(receipt.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Azure lease is already owned by another local runner") from None
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def campaign_resources(cli: AzureCLI, receipt: Path, count: int, disposal: Disposal,
                       capacity: AzureCPUCapacity, public_key: str, *, reuse: bool = False) -> Iterator[list[dict]]:
    """Retire partial acquisition as well as failed tests, preserving both errors.

    The caller owns worker/tunnel scopes *inside* this scope so those retire
    before infrastructure. SIGTERM must be translated into ordinary unwinding
    by the runner; SIGKILL/power loss requires receipt cleanup and the provider
    shutdown backstop. A normal return is never produced after cleanup failure.
    """
    with receipt_owner(receipt):
        if reuse:
            lease = AzureCampaignLease(cli, receipt, json.loads(receipt.read_text()))
            if count != lease.document["host_count"] or disposal.value != lease.document["disposal"]:
                raise ValueError("retained lease capacity/disposal differs from selected campaign")
            if (lease.document["state"] != LeaseState.RETIRED.value
                    or lease.document.get("retirement") != "vms-deallocated-disks-retained"):
                raise ValueError("reuse requires a retired retain-disks Azure lease")
        else:
            lease = AzureCampaignLease.prepare(cli, receipt, count, disposal)
        try:
            yield lease.restart(capacity, public_key) if reuse else lease.provision(capacity, public_key)
        except BaseException as primary:
            # Admission rejection before STARTING did not acquire a running
            # pool. Do not mutate a still-retired receipt or active foreign VM.
            if reuse and lease.document["state"] == LeaseState.RETIRED.value:
                raise
            try:
                lease.retire()
            except BaseException as cleanup:
                raise BaseExceptionGroup("cross-host E2E and Azure cleanup both failed", [primary, cleanup]) from None
            raise
        else:
            lease.retire()


def main(argv: list[str] | None = None) -> int:
    """Replay exact receipt cleanup after cancellation without accepting new targets.

    Provisioning deliberately has no independent CLI entry point: the E2E
    driver must derive host count from canonical cases before creating a lease.
    Cleanup cannot change subscription, group, host count or retention intent.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleanup-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    with receipt_owner(args.cleanup_receipt):
        document = json.loads(args.cleanup_receipt.read_text())
        cli = AzureCLI(document["subscription"])
        lease = AzureCampaignLease(cli, args.cleanup_receipt, document)
        cli.require_login()
        lease.retire()
    print(f"[production-cross-host] retired {lease.document['group_id']}: "
          f"{lease.document['retirement']}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.SubprocessError, ExceptionGroup) as error:
        print(f"[production-cross-host] cleanup failed: {error}", file=sys.stderr)
        sys.exit(1)
