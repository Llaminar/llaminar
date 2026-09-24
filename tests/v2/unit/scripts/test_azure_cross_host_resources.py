#!/usr/bin/env python3
"""Device-free fault injection for the owned Azure cross-host E2E resource lease.

No test contacts Azure, provisions a VM or certifies model execution. The fake
provider implements asynchronous terminal states so a zero CLI exit cannot
masquerade as completed deployment or resource retirement. Ownership mutation,
partial allocation and cleanup errors are deliberately adversarial.
"""
from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
import json
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import azure_cross_host_resources as azure
import cross_host_network as networking


SUBSCRIPTION = "11111111-2222-3333-4444-555555555555"
PUBLIC_KEY = "ssh-ed25519 " + base64.b64encode(
    b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00\x20" + bytes(32)).decode() + " unit-only"
CAPACITY = azure.AzureCPUCapacity(
    location="ukwest", vm_size="Standard_D4as_v4",
    image="Canonical:ubuntu-24_04-lts:server:24.04.202409120",
    os_disk_gib=128, ssh_source="8.8.8.8/32", private_subnet="10.221.0.0/24")


class FakeAzure:
    """An exact group/VM provider; unknown commands fail instead of quietly passing."""

    subscription = SUBSCRIPTION

    def __init__(self):
        self.calls = []
        self.group = None
        self.hosts = []
        self.deployment = None
        self.deployment_states = ["Running", "Succeeded"]
        self.deallocation_states = ["PowerState/deallocating", "PowerState/deallocated"]
        self.deallocation_observations = {}
        self.fail = None

    def require_login(self):
        """Record admission separately from the fake resource mutation API."""
        self.calls.append(("require_login",))

    def call(self, *args):
        """Simulate provider operations and allow a failure at any command boundary."""
        self.calls.append(args)
        if self.fail is not None:
            self.fail(args)
        def arg(name):
            return args[args.index(name) + 1]
        prefix = args[:2]
        if prefix == ("group", "exists"):
            return self.group is not None
        if prefix == ("group", "create"):
            self.group = {"id": f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{arg('--name')}",
                          "tags": dict(value.split("=", 1) for value in args[args.index("--tags") + 1:])}
            return self.group
        if prefix == ("group", "show"):
            return self.group
        if prefix == ("group", "delete"):
            self.group = None
            return None
        if args[:3] == ("deployment", "group", "create"):
            template = json.loads(Path(arg("--template-file")).read_text())
            for resource in template["resources"]:
                if resource["type"] != "Microsoft.Compute/virtualMachines":
                    continue
                index = len(self.hosts)
                self.hosts.append({"name": resource["name"],
                    "id": self.group["id"] + "/providers/Microsoft.Compute/virtualMachines/" + resource["name"],
                    "tags": resource["tags"], "public_ip": f"20.0.0.{index + 1}",
                    "private_ip": f"10.221.0.{index + 4}",
                    "publicIps": f"20.0.0.{index + 1}", "privateIps": f"10.221.0.{index + 4}",
                    "powerState": "VM running", "location": resource["location"],
                    "priority": resource["properties"].get("priority"),
                    "evictionPolicy": resource["properties"].get("evictionPolicy"),
                    "billingProfile": resource["properties"].get("billingProfile"),
                    "hardwareProfile": resource["properties"]["hardwareProfile"],
                    "storageProfile": resource["properties"]["storageProfile"]})
                disk = self.hosts[-1]["storageProfile"]["osDisk"]
                disk["managedDisk"] = {"id": self.group["id"] + "/providers/Microsoft.Compute/disks/" + resource["name"]}
            self.deployment = {"name": "cpu-peers", "properties": {"provisioningState": "Running"}}
            return None
        if args[:3] == ("deployment", "group", "show"):
            state = self.deployment_states[0]
            if len(self.deployment_states) > 1:
                self.deployment_states.pop(0)
            return {"properties": {"provisioningState": state,
                "outputs": {"cpu_hosts": {"value": self.hosts}}}}
        if args[:3] == ("deployment", "group", "list"):
            return [self.deployment] if self.deployment else []
        if args[:3] == ("deployment", "group", "cancel"):
            self.deployment_states = ["Canceled"]
            return None
        if prefix == ("vm", "list"):
            return self.hosts
        if prefix == ("disk", "show"):
            return next(host["storageProfile"]["osDisk"] for host in self.hosts
                        if host["storageProfile"]["osDisk"]["managedDisk"]["id"] == arg("--ids"))
        if prefix == ("vm", "deallocate"):
            self.deallocation_observations[arg("--ids")] = list(self.deallocation_states)
            next(host for host in self.hosts if host["id"] == arg("--ids"))["powerState"] = "VM deallocated"
            return None
        if prefix == ("vm", "auto-shutdown"):
            return None
        if prefix == ("vm", "start"):
            self.deallocation_observations[arg("--ids")] = ["PowerState/starting", "PowerState/running"]
            next(host for host in self.hosts if host["id"] == arg("--ids"))["powerState"] = "VM running"
            return None
        if prefix == ("vm", "get-instance-view"):
            values = self.deallocation_observations[arg("--ids")]
            state = values[0]
            if len(values) > 1:
                values.pop(0)
            return {"instanceView": {"statuses": [{"code": state}]}}
        raise AssertionError(f"unexpected fake Azure command: {args}")


class AzureCapacityTests(unittest.TestCase):
    """Network exposure and boot policy cannot depend on Azure CLI defaults."""

    def test_capacity_rejects_unpinned_or_public_network_inputs(self):
        for change in ({"image": "Ubuntu2404"}, {"image": "Canonical:ubuntu:server:latest"},
                       {"ssh_source": "0.0.0.0/0"}, {"ssh_source": "8.8.8.0/24"},
                       {"ssh_source": "127.0.0.1/32"}, {"ssh_source": "8.8.8.8"},
                       {"ssh_source": "2001:4860:4860::8888/128"},
                       {"private_subnet": "8.8.8.0/24"}, {"private_subnet": "::/0"},
                       {"private_subnet": "10.0.0.1/24"}, {"os_disk_gib": True},
                       {"tunnel_subnet": "8.8.8.0/30"}, {"tunnel_subnet": "10.221.0.0/30"},
                       {"tunnel_subnet": "10.207.14.0/24"},
                       {"pricing": "spot"},
                       {"os_disk_gib": 0}, {"location": ""}, {"vm_size": ""},
                       {"shutdown_after_minutes": True}, {"shutdown_after_minutes": 1440}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(CAPACITY, **change)

    def test_template_covers_one_and_two_hosts_with_one_schedule_each(self):
        for count in (1, 2):
            template = azure.deployment_template(CAPACITY, count, PUBLIC_KEY, {"owner": "test"},
                datetime(2026, 9, 14, 23, 57, tzinfo=timezone.utc))
            resources = template["resources"]
            vms = [row for row in resources if row["type"] == "Microsoft.Compute/virtualMachines"]
            schedules = [row for row in resources if row["type"] == "Microsoft.DevTestLab/schedules"]
            self.assertEqual(len(vms), count)
            self.assertEqual(len(schedules), count)
            self.assertEqual(len(template["outputs"]["cpu_hosts"]["value"]), count)
            for schedule in schedules:
                self.assertEqual(schedule["properties"]["dailyRecurrence"]["time"], "2357")
                self.assertEqual(schedule["properties"]["targetResourceId"], schedule["dependsOn"][0])
                self.assertEqual(schedule["properties"]["status"], "Enabled")
            for vm in vms:
                self.assertEqual(vm["properties"]["priority"], "Spot")
                self.assertEqual(vm["properties"]["evictionPolicy"], "Deallocate")
                self.assertEqual(vm["properties"]["billingProfile"], {"maxPrice": -1})
                linux = vm["properties"]["osProfile"]["linuxConfiguration"]
                self.assertTrue(linux["disablePasswordAuthentication"])
                self.assertEqual(linux["ssh"]["publicKeys"][0]["keyData"], PUBLIC_KEY)
                bootstrap = vm["properties"]["osProfile"].get("customData", "")
                decoded = base64.b64decode(bootstrap, validate=True).decode()
                self.assertIn("docker.io", decoded)
                self.assertIn("PermitTunnel point-to-point", decoded)
                self.assertIn("usermod, -aG, docker, llaminar", decoded)
            rules = resources[0]["properties"]["securityRules"]
            self.assertEqual(len(rules), 2)
            self.assertEqual(rules[0]["properties"]["sourceAddressPrefix"], CAPACITY.ssh_source)
            self.assertEqual(rules[0]["properties"]["destinationPortRange"], "22")
            self.assertEqual(rules[1]["properties"]["sourceAddressPrefix"], CAPACITY.tunnel_subnet)
            self.assertEqual(rules[1]["properties"]["destinationAddressPrefix"], CAPACITY.private_subnet)
            route = next(row for row in resources if row["type"] == "Microsoft.Network/routeTables")
            self.assertEqual(route["properties"]["routes"][0]["properties"], {
                "addressPrefix": CAPACITY.tunnel_subnet, "nextHopType": "VirtualAppliance",
                "nextHopIpAddress": "10.221.0.4"})
            nics = [row for row in resources if row["type"] == "Microsoft.Network/networkInterfaces"]
            self.assertEqual([nic["properties"]["enableIPForwarding"] for nic in nics],
                             [True] + [False] * (count - 1))
            self.assertEqual([nic["properties"]["ipConfigurations"][0]["properties"]["privateIPAddress"]
                              for nic in nics], [f"10.221.0.{4 + index}" for index in range(count)])

    def test_on_demand_is_explicit_and_does_not_retain_spot_properties(self):
        capacity = replace(CAPACITY, pricing=azure.AzureComputePricing.ON_DEMAND)
        template = azure.deployment_template(capacity, 1, PUBLIC_KEY, {"owner": "test"},
            datetime(2026, 9, 14, 23, 57, tzinfo=timezone.utc))
        vm = next(row for row in template["resources"]
                  if row["type"] == "Microsoft.Compute/virtualMachines")
        self.assertNotIn("priority", vm["properties"])
        self.assertNotIn("evictionPolicy", vm["properties"])
        self.assertNotIn("billingProfile", vm["properties"])

    def test_invalid_template_arguments_fail_before_deployment(self):
        for count, key, when in ((0, PUBLIC_KEY, datetime.now(timezone.utc)),
                                 (True, PUBLIC_KEY, datetime.now(timezone.utc)),
                                 (252, PUBLIC_KEY, datetime.now(timezone.utc)),
                                 (1, "PRIVATE KEY", datetime.now(timezone.utc)),
                                 (1, "ssh-ed25519 AAA", datetime.now(timezone.utc)),
                                 (1, PUBLIC_KEY + "\n", datetime.now(timezone.utc)),
                                 (1, PUBLIC_KEY, datetime.now())):
            with self.subTest(count=count, key=key), self.assertRaises(ValueError):
                azure.deployment_template(CAPACITY, count, key, {}, when)

    def test_cli_uses_existing_login_and_explicit_subscription_without_secrets(self):
        cli = azure.AzureCLI(SUBSCRIPTION)
        with patch.object(azure.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, json.dumps({"id": SUBSCRIPTION, "state": "Enabled"}))) as command:
            cli.require_login()
        argv = command.call_args.args[0]
        self.assertEqual(argv[:3], ["az", "account", "show"])
        self.assertEqual(argv[argv.index("--subscription") + 1], SUBSCRIPTION)
        self.assertNotIn("login", argv)
        self.assertNotIn("--debug", argv)
        self.assertEqual(command.call_args.kwargs["timeout"], 60)
        self.assertFalse(command.call_args.kwargs.get("shell", False))

    def test_cli_errors_do_not_leak_payload_or_credential_text(self):
        with patch.object(azure.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, "private output", "secret error text")):
            with self.assertRaises(RuntimeError) as caught:
                azure.AzureCLI(SUBSCRIPTION).call("deployment", "group", "private argument")
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_provider_preflight_error_retains_code_without_private_message(self):
        with patch.object(azure.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, "", json.dumps({"code": "QuotaExceeded", "message": "private credential"}))):
            with self.assertRaisesRegex(RuntimeError, "QuotaExceeded") as caught:
                azure.AzureCLI(SUBSCRIPTION).call("deployment", "group")
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("credential", str(caught.exception))

    def test_quota_error_exposes_only_bounded_admission_cardinalities(self):
        error = json.dumps({"code": "QuotaExceeded", "message": (
            "Operation denied. Location: uksouth, Current Limit: 10, Current Usage: 8, "
            "Additional Required: 4, private credential=https://example.invalid/token")})
        with patch.object(azure.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 1, "", error)):
            with self.assertRaisesRegex(RuntimeError,
                    r"QuotaExceeded quota\[location=uksouth,limit=10,usage=8,required=4\]") as caught:
                azure.AzureCLI(SUBSCRIPTION).call("deployment", "group")
        self.assertNotIn("private", str(caught.exception))
        self.assertNotIn("credential", str(caught.exception))
        self.assertNotIn("example", str(caught.exception))


class PrivateMPINetworkTests(unittest.TestCase):
    """The SSH network owns one local interface and preserves all other routes."""

    def test_gateway_opens_only_owned_subnets_not_dockers_forward_policy(self):
        with patch.object(networking, "execute") as run:
            networking.allow_gateway_forwarding({"public_ip": "8.8.8.8"}, Path("/key"),
                                                CAPACITY.private_subnet, CAPACITY.tunnel_subnet)
        commands = [shlex.split(call.args[0][-1]) for call in run.call_args_list]
        self.assertEqual(commands, [
            ["sudo", "-n", "iptables", "-w", "5", "-I", "FORWARD", "1", "-i", "tun9",
             "-s", CAPACITY.tunnel_subnet, "-d", CAPACITY.private_subnet, "-j", "ACCEPT"],
            ["sudo", "-n", "iptables", "-w", "5", "-I", "FORWARD", "1", "-o", "tun9",
             "-s", CAPACITY.private_subnet, "-d", CAPACITY.tunnel_subnet, "-j", "ACCEPT"]])

    def test_preflight_allows_default_route_and_rejects_existing_network_aliases(self):
        with patch.object(Path, "is_char_device", return_value=True), \
             patch.object(networking, "execute", side_effect=['[{"dst":"default","dev":"eth0"}]', ""]):
            networking.validate_local_network(CAPACITY.private_subnet, CAPACITY.tunnel_subnet)
        for destination in (CAPACITY.private_subnet, "10.207.0.0/16", "10.0.0.0/8"):
            with self.subTest(destination=destination), patch.object(Path, "is_char_device", return_value=True), \
                 patch.object(networking, "execute", return_value=json.dumps([{"dst": destination}])):
                with self.assertRaisesRegex(ValueError, "overlaps an installed"):
                    networking.validate_local_network(CAPACITY.private_subnet, CAPACITY.tunnel_subnet)

    def test_network_retires_exact_owned_tun_after_normal_or_failed_use(self):
        for fail_in_body in (False, True):
            calls = []
            process = MagicMock()
            process.poll.return_value = None
            process.wait.return_value = 0
            peer = {"name": "cpu-0", "public_ip": "20.0.0.1", "private_ip": "10.221.0.4"}
            second = {"name": "cpu-1", "public_ip": "20.0.0.2", "private_ip": "10.221.0.5"}
            probe = MagicMock()
            probe.__enter__.return_value.recv.return_value = b"SSH-2.0-fixture\r\n"
            with self.subTest(fail_in_body=fail_in_body), tempfile.TemporaryDirectory() as directory, \
                 patch.object(networking, "execute", side_effect=lambda argv: calls.append(argv) or ""), \
                 patch.object(networking.subprocess, "Popen", return_value=process) as launch, \
                 patch.object(networking.socket, "create_connection", return_value=probe) as connect:
                def drive():
                    with networking.private_mpi_connection([peer, second], Path("/key"), CAPACITY.private_subnet,
                            CAPACITY.tunnel_subnet, Path(directory)) as connection:
                        self.assertEqual(connection.controller_address, "10.207.14.1")
                        if fail_in_body:
                            raise RuntimeError("fixture failure")
                if fail_in_body:
                    with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                        drive()
                else:
                    drive()
                self.assertEqual([call.args[0] for call in connect.call_args_list],
                                 [(peer["private_ip"], 22), (second["private_ip"], 22)])
            added = calls[0][calls[0].index("dev") + 1]
            self.assertEqual(calls[-1], ["sudo", "-n", "ip", "link", "delete", "dev", added])
            self.assertFalse(any("sysctl" in call and call[0] != "ssh" for call in calls))
            remote_commands = [shlex.split(call[-1]) for call in calls if call[0] == "ssh"]
            self.assertIn(["sudo", "-n", "sysctl", "-w", "net.ipv4.ip_forward=1"], remote_commands)
            process.terminate.assert_called_once()
            argv = launch.call_args.args[0]
            self.assertEqual(argv[argv.index("-w") + 1], added.removeprefix("tun") + ":9")
            self.assertIn("ExitOnForwardFailure=yes", argv)

    def test_failed_acquisition_never_deletes_a_preexisting_tun(self):
        with patch.object(networking, "execute", side_effect=RuntimeError("EEXIST")) as command:
            with self.assertRaisesRegex(RuntimeError, "EEXIST"):
                with networking.private_mpi_connection([{"name": "cpu-0"}], Path("/key"),
                        CAPACITY.private_subnet, CAPACITY.tunnel_subnet, Path("/unused")):
                    self.fail("failed acquisition cannot yield")
        self.assertEqual(command.call_count, 1)


class AzureLeaseTests(unittest.TestCase):
    """Acquisition and all exits share one exact durable cleanup capability."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.receipt = Path(self.temporary.name) / "lease.json"
        self.cli = FakeAzure()
        self.sleep = patch.object(azure.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def prepare(self, disposal=azure.Disposal.DELETE, count=2):
        return azure.AzureCampaignLease.prepare(self.cli, self.receipt, count, disposal)

    def test_prepare_is_read_only_and_records_exact_scope_before_mutation(self):
        with patch.object(azure.os, "fsync", wraps=azure.os.fsync) as persist:
            lease = self.prepare()
        self.assertEqual(persist.call_count, 2)  # File data plus directory entry.
        record = json.loads(self.receipt.read_text())
        self.assertEqual(record, lease.document)
        self.assertEqual(record["state"], "allocating")
        self.assertEqual(len(self.cli.calls), 2)
        self.assertIsNone(self.cli.group)
        with self.assertRaises(ValueError):
            self.prepare()

    def test_existing_group_is_never_adopted(self):
        self.cli.group = {"id": "unrelated"}
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.receipt.exists())
        self.assertFalse(any(call[:2] == ("group", "create") for call in self.cli.calls))

    def test_acquisition_is_exact_and_cannot_be_repeated(self):
        lease = self.prepare()
        hosts = lease.provision(CAPACITY, PUBLIC_KEY)
        self.assertEqual([row["name"] for row in hosts], ["cpu-0", "cpu-1"])
        document = json.loads(self.receipt.read_text())
        self.assertEqual(document["state"], "ready")
        self.assertEqual(document["pricing"], azure.AzureComputePricing.SPOT.value)
        self.assertEqual(self.cli.group["tags"]["llaminar-pricing"], "spot")
        with self.assertRaises(ValueError):
            lease.provision(CAPACITY, PUBLIC_KEY)
        lease.retire()
        self.assertEqual(json.loads(self.receipt.read_text())["retirement"], "group-deleted")

    def test_ambiguous_create_response_still_cleans_owned_group(self):
        original = self.cli.call
        def call(*args):
            result = original(*args)
            if args[:2] == ("group", "create"):
                raise TimeoutError("response lost after group creation")
            return result
        with patch.object(self.cli, "call", side_effect=call):
            with self.assertRaises(TimeoutError):
                with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                    self.fail("acquisition should not return")
        self.assertIsNone(self.cli.group)
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retired")

    def test_partial_deployment_failure_and_keyboard_interrupt_retire_resources(self):
        for exception in (RuntimeError("failed upload"), KeyboardInterrupt()):
            with self.subTest(exception=type(exception).__name__):
                receipt = self.receipt.with_name(type(exception).__name__ + ".json")
                cli = FakeAzure()
                with self.assertRaises(type(exception)):
                    with azure.campaign_resources(cli, receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                        raise exception
                self.assertIsNone(cli.group)
                self.assertEqual(json.loads(receipt.read_text())["state"], "retired")

    def test_terminal_provider_failure_never_reaches_inference(self):
        self.cli.deployment_states = ["Failed"]
        with self.assertRaisesRegex(RuntimeError, "deployment Failed"):
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                self.fail("failed provider must not produce hosts")
        self.assertIsNone(self.cli.group)

    def test_bad_host_identity_or_address_cannot_become_ready(self):
        for key, value in (("id", "foreign"), ("name", "foreign"),
                           ("public_ip", "20.0.0.2"), ("public_ip", "127.0.0.1"),
                           ("private_ip", "10.221.0.5"), ("private_ip", "10.222.0.4")):
            with self.subTest(key=key, value=value):
                cli = FakeAzure()
                original = cli.call
                def call(*args):
                    result = original(*args)
                    if args[:3] == ("deployment", "group", "show"):
                        result["properties"]["outputs"]["cpu_hosts"]["value"][0][key] = value
                    return result
                receipt = self.receipt.with_name(key + value.replace(".", "-") + ".json")
                with patch.object(cli, "call", side_effect=call), self.assertRaises(ValueError):
                    with azure.campaign_resources(cli, receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                        self.fail("malformed host output must not reach inference")
                self.assertIsNone(cli.group)

    def test_retirement_can_resume_from_durable_receipt_and_is_idempotent(self):
        lease = self.prepare()
        lease.provision(CAPACITY, PUBLIC_KEY)
        restored = azure.AzureCampaignLease(self.cli, self.receipt, json.loads(self.receipt.read_text()))
        restored.retire()
        count = len(self.cli.calls)
        restored.retire()
        self.assertEqual(self.cli.calls[count:], [("group", "exists", "--name", lease.document["group"])])

    def test_ownership_mutation_blocks_delete_and_preserves_recovery_receipt(self):
        lease = self.prepare()
        lease.provision(CAPACITY, PUBLIC_KEY)
        for key in ("purpose", "llaminar-lease", "llaminar-disposal"):
            original = self.cli.group["tags"][key]
            self.cli.group["tags"][key] = "foreign"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "ownership changed"):
                lease.retire()
            self.cli.group["tags"][key] = original
        self.assertFalse(any(call[:2] == ("group", "delete") for call in self.cli.calls))
        self.assertNotEqual(json.loads(self.receipt.read_text())["state"], "retired")

    def test_cleanup_receipt_cannot_target_existing_development_groups_or_other_subscriptions(self):
        lease = self.prepare()
        for change in ({"group": "rg-llaminar-mpi-crosshost-20260914"},
                       {"group_id": "/subscriptions/other/resourceGroups/user"},
                       {"subscription": "foreign"}, {"lease_id": "*"}, {"host_count": True},
                       {"state": "succeeded"}, {"disposal": "guess"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                azure.AzureCampaignLease(self.cli, self.receipt, lease.document | change)

    def test_both_primary_failure_and_cleanup_failure_remain_visible(self):
        def fail(args):
            if args[:2] == ("group", "delete"):
                raise RuntimeError("cleanup failed")
        self.cli.fail = fail
        with self.assertRaises(BaseExceptionGroup) as caught:
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                raise ValueError("inference failed")
        self.assertEqual([str(error) for error in caught.exception.exceptions], ["inference failed", "cleanup failed"])
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retiring")

    def test_successful_body_cannot_hide_cleanup_failure(self):
        def fail(args):
            if args[:2] == ("group", "delete"):
                raise RuntimeError("cleanup failed")
        self.cli.fail = fail
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DELETE, CAPACITY, PUBLIC_KEY):
                pass
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retiring")

    def test_retained_disks_submit_all_stops_before_waiting_and_cancel_pending_deployment(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        lease.retire()
        calls = self.cli.calls
        cancel = next(i for i, call in enumerate(calls) if call[:3] == ("deployment", "group", "cancel"))
        stops = [i for i, call in enumerate(calls) if call[:2] == ("vm", "deallocate")]
        views = [i for i, call in enumerate(calls) if call[:2] == ("vm", "get-instance-view")]
        self.assertEqual(len(stops), 2)
        self.assertLess(cancel, min(stops))
        self.assertLess(max(stops), min(views))
        self.assertEqual(len(views), 4)  # Deallocating is not deallocated.
        self.assertFalse(any(call[:2] == ("group", "delete") for call in calls))
        self.assertEqual(json.loads(self.receipt.read_text())["retirement"], "vms-deallocated-disks-retained")

    def test_failed_stop_does_not_prevent_other_host_retirement(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        def fail(args):
            if args[:2] == ("vm", "deallocate") and args[args.index("--ids") + 1].endswith("cpu-0"):
                raise RuntimeError("cpu-0 deallocation submission failed")
        self.cli.fail = fail
        with self.assertRaises(ExceptionGroup):
            lease.retire()
        observations = [call for call in self.cli.calls if call[:2] == ("vm", "get-instance-view")]
        self.assertEqual(len(observations), 2)
        self.assertTrue(all(call[call.index("--ids") + 1].endswith("cpu-1") for call in observations))
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retiring")

    def test_stopped_guest_is_not_provider_deallocation_and_watchdog_is_bounded(self):
        lease = self.prepare()
        values = iter([0, 1, 4])
        with patch.object(azure.time, "monotonic", side_effect=lambda: next(values)):
            with self.assertRaises(TimeoutError):
                lease._poll(lambda: "PowerState/stopped", lambda value: value == "PowerState/deallocated", "test", 3)

    def test_retained_partial_deployment_cannot_claim_missing_vms_will_never_appear(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        self.cli.hosts.pop()
        with self.assertRaises(ExceptionGroup) as caught:
            lease.retire()
        self.assertIn("incomplete VM membership", str(caught.exception.exceptions[-1]))
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retiring")
        self.assertTrue(any(call[:2] == ("vm", "deallocate") for call in self.cli.calls))

    def test_cleanup_cli_replays_recorded_scope_and_never_allocates(self):
        lease = self.prepare()
        lease.provision(CAPACITY, PUBLIC_KEY)
        before = len(self.cli.calls)
        with patch.object(azure, "AzureCLI", return_value=self.cli) as factory:
            self.assertEqual(azure.main(["--cleanup-receipt", str(self.receipt)]), 0)
        factory.assert_called_once_with(SUBSCRIPTION)
        self.assertFalse(any(call[:2] == ("group", "create") for call in self.cli.calls[before:]))
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retired")

    def test_reuse_renews_every_shutdown_before_parallel_starts_then_retires(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        lease.retire()
        before = len(self.cli.calls)
        with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DEALLOCATE,
                                      CAPACITY, PUBLIC_KEY, reuse=True) as hosts:
            self.assertEqual([host["name"] for host in hosts], ["cpu-0", "cpu-1"])
            self.assertEqual(json.loads(self.receipt.read_text())["state"], "ready")
        calls = self.cli.calls[before:]
        schedules = [i for i, call in enumerate(calls) if call[:2] == ("vm", "auto-shutdown")]
        starts = [i for i, call in enumerate(calls) if call[:2] == ("vm", "start")]
        views = [i for i, call in enumerate(calls) if call[:2] == ("vm", "get-instance-view")]
        self.assertEqual(len(schedules), 2)
        self.assertEqual(len(starts), 2)
        self.assertLess(max(schedules), min(starts))
        self.assertLess(max(starts), min(views))
        self.assertEqual(json.loads(self.receipt.read_text())["reuse_count"], 1)
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retired")
        self.assertFalse(any(call[:2] == ("group", "create") for call in calls))

    def test_reuse_rejects_live_or_changed_peers_without_mutation(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        lease.retire()
        for key, value in (("powerState", "VM running"), ("publicIps", "20.0.0.200"),
                           ("tags", {}), ("hardwareProfile", {"vmSize": "Standard_B1s"}),
                           ("priority", None), ("evictionPolicy", None),
                           ("billingProfile", None)):
            original = self.cli.hosts[0][key]
            self.cli.hosts[0][key] = value
            before = len(self.cli.calls)
            with self.subTest(key=key), self.assertRaises(ValueError):
                with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DEALLOCATE,
                                              CAPACITY, PUBLIC_KEY, reuse=True):
                    self.fail("changed pool must not be acquired")
            self.assertFalse(any(call[:2] in (("vm", "start"), ("vm", "deallocate"),
                                             ("vm", "auto-shutdown")) for call in self.cli.calls[before:]))
            self.cli.hosts[0][key] = original

    def test_reuse_cannot_change_capacity_or_stop_an_active_receipt(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        before = len(self.cli.calls)
        with self.assertRaisesRegex(ValueError, "retired"):
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DEALLOCATE,
                                          CAPACITY, PUBLIC_KEY, reuse=True):
                pass
        self.assertEqual(len(self.cli.calls), before)
        lease.retire()
        before = len(self.cli.calls)
        with self.assertRaisesRegex(ValueError, "differs"):
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DEALLOCATE,
                    replace(CAPACITY, vm_size="Standard_B1s"), PUBLIC_KEY, reuse=True):
                pass
        self.assertFalse(any(call[:2] == ("vm", "start") for call in self.cli.calls[before:]))

    def test_partial_restart_failure_still_deallocates_entire_owned_pool(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        lease.provision(CAPACITY, PUBLIC_KEY)
        lease.retire()
        def fail(args):
            if args[:2] == ("vm", "start") and args[args.index("--ids") + 1].endswith("cpu-1"):
                raise RuntimeError("restart rejected")
        self.cli.fail = fail
        with self.assertRaisesRegex(RuntimeError, "restart rejected"):
            with azure.campaign_resources(self.cli, self.receipt, 2, azure.Disposal.DEALLOCATE,
                                          CAPACITY, PUBLIC_KEY, reuse=True):
                self.fail("partial startup must not reach inference")
        self.assertEqual(json.loads(self.receipt.read_text())["state"], "retired")
        self.assertTrue(all(host["powerState"] == "VM deallocated" for host in self.cli.hosts))

    def test_exclusive_receipt_owner_blocks_runner_and_cleanup(self):
        lease = self.prepare(azure.Disposal.DEALLOCATE)
        with azure.receipt_owner(self.receipt):
            with self.assertRaisesRegex(ValueError, "already owned"):
                with azure.receipt_owner(self.receipt):
                    pass
            with self.assertRaisesRegex(ValueError, "already owned"):
                azure.main(["--cleanup-receipt", str(self.receipt)])
        with azure.receipt_owner(self.receipt):
            self.assertEqual(lease.document["state"], "allocating")


if __name__ == "__main__":
    unittest.main()
