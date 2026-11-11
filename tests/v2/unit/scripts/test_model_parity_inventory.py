#!/usr/bin/env python3
"""Device-free regressions for the one canonical model-cell discovery boundary.

Synthetic GoogleTest metadata exercises policy and malformed inventories only;
it does not certify inference. Real binaries are listed separately without
loading models, and every runtime gate consumes their exact typed parameters.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import model_parity_inventory as inventory
import run_model_parity_e2e as e2e


def row(name, tagged):
    """Supply opaque topology metadata; Python must not interpret its spelling."""
    return {"case": name, "campaign": "campaign", "backends": "CPU+ROCm",
            "model_files": ["/models/model.gguf"],
            "configuration": {"model_parity_schema": 1, "id": name,
                "cross_host_e2e": [],
                "model": "/models/model.gguf", "e2e": {"server_args": ["opaque"]} if tagged else None}}


def remote_cases():
    """Synthetic wire examples exercise schema validation, not model selection."""
    return [{"schema": 1, "id": f"opaque-{count}-{route}", "frontend": route,
             "topology": {"kind": "cross-host-expert-overlay", "continuation_backend": "rocm",
                          "continuation_devices": 1, "remote_cpu_hosts": count,
                          "cpu_ranks_per_host": 1, "execution_ranks": count + 1,
                          "continuation_priority": -4, "remote_priority": 37},
             "movement_evidence": "required", "owner_order": "ordinal",
             "planning": {"mode": "auto", "strategy": "expert-overlay",
                          "device_counts": {"rocm": 1, "cpu": count},
                          "mpi_ranks": count + 1},
             "server_policy_args": ["--only-strategies", "expert-overlay"]}
            for count in (1, 2) for route in ("plan-apply", "auto-serve")]


class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "manifest.json"
        self.document = {"schema": 1, "source_revision": "revision", "scope": "all",
                         "cells": [row("untagged", False), row("tagged", True)]}
        self.args = argparse.Namespace(manifest=self.path, source_revision="revision",
                                      backend=".*", campaign=".*", cell=".*")

    def discover(self, scope):
        """Publish synthetic metadata, then use the production reader unchanged."""
        self.path.write_text(json.dumps(self.document))
        return inventory.discover(self.args, scope)

    def test_all_cells_and_e2e_are_projections_of_one_inventory(self):
        full = self.discover(inventory.InventoryScope.ALL)
        tagged = e2e.discover(self.args)
        self.assertEqual([exact for _, exact, _ in full], ["untagged", "tagged"])
        self.assertEqual(tagged, full[1:])
        exported = inventory.export_manifest(full, "revision", inventory.InventoryScope.ALL)
        self.assertEqual(exported, self.document)

    def test_local_diagnostic_scope_never_promotes_a_candidate_to_runtime_inventory(self):
        """The explicit projection is inspectable but cannot masquerade as ALL/E2E."""
        candidate=row("candidate",False)
        candidate["configuration"]["inventory_role"]="numerical_diagnostic"
        self.document["scope"]="local-diagnostic";self.document["cells"]=[candidate]
        selected=self.discover(inventory.InventoryScope.LOCAL_DIAGNOSTIC)
        self.assertEqual([exact for _,exact,_ in selected],["candidate"])
        for scope in inventory.InventoryScope:
            if scope==inventory.InventoryScope.LOCAL_DIAGNOSTIC:continue
            with self.subTest(scope=scope),self.assertRaisesRegex(ValueError,"local diagnostic manifest"):
                self.discover(scope)
            self.assertFalse(scope.accepts(candidate["configuration"]))
        self.assertTrue(inventory.InventoryScope.LOCAL_DIAGNOSTIC.accepts(candidate["configuration"]))
        candidate["configuration"]["e2e"]={}
        with self.assertRaisesRegex(ValueError,"certification eligibility"):
            inventory.InventoryScope.LOCAL_DIAGNOSTIC.accepts(candidate["configuration"])

    def test_explicit_predictor_is_required_in_manifest_even_for_mtp_off(self):
        """The optional second source is not discarded by inactive MTP policy."""
        source = self.document["cells"][0]
        source["configuration"]["shared_predictor"] = "/models/predictor.gguf"
        source["configuration"]["runtime"] = {"generation": {"mtp_policy": "off"}}
        with self.assertRaisesRegex(ValueError, "declared GGUF source"):
            self.discover(inventory.InventoryScope.ALL)
        source["model_files"].append("/models/predictor.gguf")
        selected = self.discover(inventory.InventoryScope.ALL)
        self.assertEqual(selected[0][0].model_files,
                         ("/models/model.gguf", "/models/predictor.gguf"))
        self.assertEqual(inventory.export_manifest(selected, "revision", inventory.InventoryScope.ALL),
                         self.document)

    def test_malformed_explicit_predictor_cannot_enter_inventory(self):
        """Absence is legacy behavior; a present source must be a nonempty path."""
        for invalid in (None, "", 7, [], {}):
            self.document["cells"][0]["configuration"]["shared_predictor"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "explicit GGUF source"):
                self.discover(inventory.InventoryScope.ALL)

    def test_live_discovery_retains_predictor_and_both_split_sources(self):
        """Join explicit source roles to the existing exact GGUF staging authority."""
        root = Path(self.temporary.name)
        main = root / "main-00001-of-00002.gguf"
        main_second = root / "main-00002-of-00002.gguf"
        draft = root / "draft.gguf"
        unrelated = root / "unrelated.gguf"
        for path in (main, main_second, draft, unrelated):
            path.write_bytes(b"source fixture")
        main_alias, draft_alias = root / "main-alias.gguf", root / "draft-alias.gguf"
        main_alias.symlink_to(main)
        draft_alias.symlink_to(draft)
        self.args.manifest = None
        self.args.build_dir = Path("/build")
        binary = "/build/v2_integration_parity_model_matrix"
        record = row("opaque", False)["configuration"] | {
            "model": str(main_alias), "shared_predictor": str(draft_alias)}
        campaign = inventory.parity.CampaignCell("campaign", inventory.parity.CampaignGroup("CPU", "ALL"),
            command=(binary,), gtest_cases=("Suite.ProductionParity/opaque",),
            model_files=(str(main), str(draft), str(unrelated)), working_directory=str(root))
        def list_binary(command, **kwargs):
            Path(command[2].removeprefix("--gtest_output=json:")).write_text(json.dumps(
                {"testsuites": [{"name": "Suite", "testsuite": [{"name": "ProductionParity/opaque",
                    "value_param": json.dumps(record)}]}]}))
        with patch.object(inventory.parity, "discover_campaigns", return_value=[campaign]), \
             patch.object(inventory.subprocess, "run", side_effect=list_binary):
            selected = inventory.discover(self.args, inventory.InventoryScope.ALL)
        self.assertEqual(set(selected[0][0].model_files), {str(main), str(main_second), str(draft)})
        self.assertEqual(selected[0][2]["model"], str(main))
        self.assertEqual(selected[0][2]["shared_predictor"], str(draft))
        campaign = inventory.dataclasses.replace(campaign, model_files=(str(main), str(unrelated)))
        with patch.object(inventory.parity, "discover_campaigns", return_value=[campaign]), \
             patch.object(inventory.subprocess, "run", side_effect=list_binary):
            with self.assertRaisesRegex(ValueError, "shared_predictor is not in the canonical GGUF manifest"):
                inventory.discover(self.args, inventory.InventoryScope.ALL)

    def test_predictor_mount_mapping_updates_only_explicit_source_and_argv(self):
        """Staging rewrites the declared artifact without changing inference policy."""
        record = row("mapped", False)["configuration"]
        record["shared_predictor"] = "/models/predictor.gguf"
        record["runtime"] = {"server_args": ["--no-mtp", "--mtp-model", "/models/predictor.gguf"]}
        record["runtime"]["generation"] = {"body": {"args": ["--mtp-model", "user data"]}}
        record["benchmark"] = {"args": ["--mtp-model", "/models/predictor.gguf", "--no-mtp"]}
        original = copy.deepcopy(record)
        mapped = inventory.map_predictor_source(record, lambda path: "/staged/" + Path(path).name)
        self.assertEqual(record, original)
        self.assertEqual(mapped["model"], record["model"])
        self.assertEqual(mapped["shared_predictor"], "/staged/predictor.gguf")
        self.assertEqual(mapped["runtime"]["server_args"], ["--no-mtp", "--mtp-model", "/staged/predictor.gguf"])
        self.assertEqual(mapped["benchmark"]["args"], ["--mtp-model", "/staged/predictor.gguf", "--no-mtp"])
        self.assertEqual(inventory.map_predictor_source(row("old", False)["configuration"],
                         lambda path: self.fail("absent source must not invoke remapping")),
                         row("old", False)["configuration"])

    def test_foreign_or_duplicate_predictor_arguments_are_rejected(self):
        """A CLI argument cannot override the immutable source record."""
        record = row("source", False)["configuration"] | {"shared_predictor": "/models/predictor.gguf"}
        for args in (["--mtp-model", "/models/foreign.gguf"], ["--mtp-model"], [],
                     ["--mtp-model", "/models/predictor.gguf", "--mtp-model", "/models/predictor.gguf"]):
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "foreign shared predictor"):
                inventory.map_predictor_source(record | {"runtime": {"server_args": args}}, str)
        del record["shared_predictor"]
        with self.assertRaisesRegex(ValueError, "undeclared shared predictor"):
            inventory.map_predictor_source(record | {"runtime": {"server_args": ["--mtp-model", "/models/foreign.gguf"]}}, str)

    def test_diagnostic_benchmark_predictor_source_remains_owned_and_uncertified(self):
        """The separate diagnostic argv obeys the same explicit source boundary."""
        record = row("diagnostic", False)["configuration"] | {
            "shared_predictor": "/models/predictor.gguf",
            "diagnostic_benchmark": {"schema": 1, "policy": "diagnostic_declared",
                "certification_eligible": False,
                "args": ["--mtp-model", "/models/predictor.gguf", "--no-mtp"]}}
        original = copy.deepcopy(record)
        mapped = inventory.map_predictor_source(record, lambda value: "/staged/" + Path(value).name)
        self.assertEqual(record, original)
        self.assertIsNone(mapped["e2e"])
        self.assertFalse(mapped["diagnostic_benchmark"]["certification_eligible"])
        self.assertEqual(mapped["diagnostic_benchmark"]["args"],
                         ["--mtp-model", "/staged/predictor.gguf", "--no-mtp"])
        for args in (["--mtp-model", "/models/foreign.gguf"], [],
                     ["--mtp-model", "/models/predictor.gguf", "--mtp-model", "/models/predictor.gguf"]):
            malformed = record | {"diagnostic_benchmark": {"args": args}}
            with self.subTest(args=args), self.assertRaisesRegex(ValueError, "foreign shared predictor"):
                inventory.map_predictor_source(malformed, str)
        del record["shared_predictor"]
        with self.assertRaisesRegex(ValueError, "undeclared shared predictor"):
            inventory.map_predictor_source(record, str)

    def test_tagged_manifest_cannot_claim_full_generation_coverage(self):
        self.document["scope"] = "e2e"
        self.document["cells"] = [row("tagged", True)]
        for legacy in (False, True):
            if legacy:
                self.document.pop("scope")
            with self.subTest(legacy=legacy):
                with self.assertRaisesRegex(ValueError, "all-cell manifest"):
                    self.discover(inventory.InventoryScope.ALL)
                self.assertEqual(len(self.discover(inventory.InventoryScope.E2E)), 1)

    def test_remote_scope_projects_existing_tags_without_running_cloud_commands(self):
        self.document["cells"][1]["configuration"]["cross_host_e2e"] = remote_cases()
        with patch.object(inventory.subprocess, "run", side_effect=AssertionError("no execution during discovery")):
            selected = self.discover(inventory.InventoryScope.CROSS_HOST_E2E)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][2]["cross_host_e2e"], remote_cases())
        exported = inventory.export_manifest(selected, "revision", inventory.InventoryScope.CROSS_HOST_E2E)
        self.assertEqual(exported["scope"], "cross-host-e2e")
        self.document = exported
        self.assertEqual(self.discover(inventory.InventoryScope.CROSS_HOST_E2E), selected)
        with self.assertRaisesRegex(ValueError, "all-cell manifest"):
            self.discover(inventory.InventoryScope.ALL)

    def test_remote_scope_rejects_stale_empty_or_untagged_metadata(self):
        with self.assertRaisesRegex(ValueError, "no cross-host-e2e canonical cells"):
            self.discover(inventory.InventoryScope.CROSS_HOST_E2E)
        self.document["cells"][0]["configuration"].pop("cross_host_e2e")
        with self.assertRaisesRegex(ValueError, "stale metadata"):
            self.discover(inventory.InventoryScope.CROSS_HOST_E2E)
        self.document["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        with self.assertRaisesRegex(ValueError, "HTTP-tagged"):
            self.discover(inventory.InventoryScope.CROSS_HOST_E2E)

    def test_remote_wire_rejects_missing_duplicate_and_inconsistent_frontend_routes(self):
        for mutate in (
            lambda rows: rows.pop(),
            lambda rows: rows.append(copy.deepcopy(rows[0])),
            lambda rows: rows[0].update(frontend="private-runner"),
            lambda rows: rows[0].update(schema=True),
            lambda rows: rows[0].update(id=""),
            lambda rows: rows[0].update(model="/stale/path.gguf"),
            lambda rows: rows[0].update(movement_evidence="not_applicable"),
            lambda rows: rows[0].update(server_policy_args="--mtp"),
            lambda rows: rows[0].update(server_policy_args=["--different-policy"]),
            lambda rows: rows[0].pop("planning"),
            lambda rows: rows[0]["planning"]["device_counts"].update(cpu=7),
            lambda rows: rows[0]["planning"].pop("mpi_ranks"),
            lambda rows: rows[0]["planning"].update(mpi_ranks=7),
            lambda rows: rows[0]["topology"].update(remote_cpu_hosts=0),
            lambda rows: rows[0]["topology"].update(remote_cpu_hosts=True),
            lambda rows: rows[0]["topology"].update(remote_cpu_hosts=2**31 - 1),
            lambda rows: rows[0]["topology"].update(execution_ranks=7),
            lambda rows: rows[0]["topology"].update(continuation_backend="cpu"),
            lambda rows: rows[0]["topology"].update(remote_priority=-4),
        ):
            record = row("source", True)["configuration"]
            record["cross_host_e2e"] = remote_cases()
            mutate(record["cross_host_e2e"])
            with self.subTest(record=record), self.assertRaises(ValueError):
                inventory.cross_host_scenarios(record)

    def test_remote_cli_lists_expanded_frontend_identities_from_the_exporter(self):
        self.document["cells"][1]["configuration"]["cross_host_e2e"] = remote_cases()
        self.path.write_text(json.dumps(self.document))
        with patch("builtins.print") as output:
            self.assertEqual(inventory.main([
                "--manifest", str(self.path), "--source-revision", "revision",
                "--scope", "cross-host-e2e"]), 0)
        self.assertEqual([call.args[0] for call in output.call_args_list[:-1]],
                         [scenario["id"] for scenario in remote_cases()])
        self.assertEqual(output.call_args_list[-1].args[0],
                         "[model-parity-inventory] scope=cross-host-e2e cells=4 source_cells=1")

    def test_remote_harness_selects_only_an_exact_existing_scenario(self):
        record = row("source", True)["configuration"]
        record["cross_host_e2e"] = remote_cases()
        original = copy.deepcopy(record)
        for scenario in record["cross_host_e2e"]:
            self.assertIs(inventory.select_cross_host_scenario(record, scenario["id"]), scenario)
        self.assertEqual(record, original)
        for name in ("", "opaque-.*", "absent", None):
            with self.subTest(name=name), self.assertRaises(ValueError):
                inventory.select_cross_host_scenario(record, name)
        # Narrowing may not hide a missing partner or invalid source identity.
        for mutate in (lambda r: r["cross_host_e2e"].pop(),
                       lambda r: r.pop("model"), lambda r: r.update(model_parity_schema=True),
                       lambda r: r.pop("id")):
            broken = copy.deepcopy(record)
            mutate(broken)
            with self.assertRaises(ValueError):
                inventory.select_cross_host_scenario(broken, remote_cases()[0]["id"])

    def test_remote_harness_selects_exact_scenario_from_manifest_projection(self):
        self.document["cells"][1]["configuration"]["cross_host_e2e"] = remote_cases()
        projected = {"schema": 1, "scope": "cross-host-e2e",
                     "source_revision": "revision",
                     "cells": [self.document["cells"][1]]}
        selected = inventory.select_cross_host_scenario_from_manifest(
            projected, remote_cases()[0]["id"])
        self.assertEqual(selected, remote_cases()[0])

    def test_remote_manifest_rejects_duplicate_scenario_ownership(self):
        self.document["cells"][0] = copy.deepcopy(self.document["cells"][1])
        self.document["cells"][0]["case"] = "duplicate-source"
        self.document["cells"][0]["configuration"]["id"] = "duplicate-source-id"
        self.document["cells"][0]["configuration"]["cross_host_e2e"] = remote_cases()
        self.document["cells"][1]["configuration"]["cross_host_e2e"] = remote_cases()
        with self.assertRaisesRegex(ValueError, "absent or duplicated"):
            inventory.select_cross_host_scenario_from_manifest(
                {"schema": 1, "scope": "cross-host-e2e", "source_revision": "revision",
                 "cells": self.document["cells"]},
                remote_cases()[0]["id"])

    def test_filters_do_not_hide_duplicate_or_invalid_manifest_entries(self):
        self.args.cell = "tagged"
        for mutate in (lambda d: d["cells"].append(copy.deepcopy(d["cells"][0])),
                       lambda d: d["cells"][0]["configuration"].pop("e2e"),
                       lambda d: d["cells"][0].update(model_files=[]),
                       lambda d: d["cells"][0]["configuration"].update(model_parity_schema=0)):
            original = copy.deepcopy(self.document)
            mutate(self.document)
            with self.subTest(document=self.document), self.assertRaises(ValueError):
                self.discover(inventory.InventoryScope.ALL)
            self.document = original

    def test_wrong_scope_revision_and_empty_selection_are_rejected(self):
        for field, value in (("scope", "unknown"), ("source_revision", "stale"), ("schema", 0)):
            original = copy.deepcopy(self.document)
            self.document[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.discover(inventory.InventoryScope.ALL)
            self.document = original
        self.args.cell = "missing"
        with self.assertRaisesRegex(ValueError, "no all canonical cells"):
            self.discover(inventory.InventoryScope.ALL)
        with self.assertRaises(TypeError):
            self.discover("all")

    def test_tagged_scope_rejects_untagged_entries_instead_of_dropping_them(self):
        self.document["scope"] = "e2e"
        with self.assertRaises(ValueError):
            self.discover(inventory.InventoryScope.E2E)

    def test_missing_or_malformed_eligibility_never_becomes_an_implicit_tag(self):
        for value in (False, True, "e2e", []):
            self.document["cells"][0]["configuration"]["e2e"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.discover(inventory.InventoryScope.ALL)

    def test_one_binary_is_listed_once_across_aggregate_slices(self):
        model = Path(self.temporary.name) / "model.gguf"
        model.touch()
        alias = model.with_name("alias.gguf")
        alias.symlink_to(model)
        self.args.manifest = None
        self.args.build_dir = Path("/build")
        binary = "/build/v2_integration_parity_model_matrix"
        records = []
        campaigns = []
        for name, tagged in (("untagged", False), ("tagged", True)):
            record = row(name, tagged)["configuration"] | {"model": str(alias)}
            records.append({"name": "ProductionParity/" + name, "value_param": json.dumps(record)})
            campaigns.append(inventory.parity.CampaignCell(
                name, inventory.parity.CampaignGroup("CPU", "ALL"),
                command=(binary,), gtest_cases=("Suite.ProductionParity/" + name,),
                model_files=(str(alias),), working_directory=self.temporary.name))
        def list_binary(command, **kwargs):
            self.assertEqual(command[:2], [binary, "--gtest_list_tests"])
            self.assertEqual(kwargs["env"]["LLAMINAR_FORCE_CPU_ONLY_STARTUP"], "1")
            Path(command[2].removeprefix("--gtest_output=json:")).write_text(json.dumps(
                {"testsuites": [{"name": "Suite", "testsuite": records}]}))
        with patch.object(inventory.parity, "discover_campaigns", return_value=campaigns), \
             patch.object(inventory.parity, "selected_model_files", return_value=[model]), \
             patch.object(inventory.subprocess, "run", side_effect=list_binary) as launch:
            selected = inventory.discover(self.args, inventory.InventoryScope.ALL)
        self.assertEqual(launch.call_count, 1)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(record["model"] == str(model) for _, _, record in selected))
        # A narrow diagnostic still audits the whole chosen binary. It must
        # catch registration loss even if the lost cell was not selected.
        self.args.cell = "Suite.ProductionParity/tagged"
        with patch.object(inventory.parity, "discover_campaigns", return_value=campaigns[1:]), \
             patch.object(inventory.subprocess, "run", side_effect=list_binary):
            with self.assertRaisesRegex(ValueError, "unregistered=.*untagged"):
                inventory.discover(self.args, inventory.InventoryScope.ALL)


class ExactLiveSelectionTests(unittest.TestCase):
    """Pure authored listings prove selection economy and complete chosen ownership."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / "model.gguf"
        self.model.touch()
        self.args = argparse.Namespace(manifest=None, build_dir=Path("/build"),
            source_revision="revision", backend=".*", campaign=".*", cell="Suite.ProductionParity/wanted")
        self.binary = "/build/v2_integration_parity_selected_matrix"
        self.other = "/build/v2_integration_parity_unrelated_matrix"
        self.cases = ("Suite.ProductionParity/wanted", "Suite.ProductionParity/sibling")
        self.campaigns = [self.campaign("selected", self.binary, self.cases),
                          self.campaign("unrelated", self.other, ("Suite.ProductionParity/other",))]
        self.listings = {self.binary: self.cases, self.other: ("Suite.ProductionParity/other",)}
        self.launched = []
        self.child_environments = []

    def campaign(self, name, binary, cases):
        """Retain real campaign value objects, never executable fixture processes."""
        return inventory.parity.CampaignCell(name, inventory.parity.CampaignGroup("CPU", "ALL"),
            command=(binary,), gtest_cases=cases, model_files=(str(self.model),),
            working_directory=self.temporary.name)

    def discover(self):
        """Stub process creation; production discovery performs every join and rejection."""
        def listing(command, **kwargs):
            self.launched.append(command[0])
            self.child_environments.append(dict(kwargs["env"]))
            records = [{"name": exact.removeprefix("Suite."),
                "value_param": json.dumps(row(exact, False)["configuration"] | {"model": str(self.model)})}
                for exact in self.listings[command[0]]]
            Path(command[2].removeprefix("--gtest_output=json:")).write_text(json.dumps(
                {"testsuites": [{"name": "Suite", "testsuite": records}]}))
        with patch.object(inventory.parity, "discover_campaigns", return_value=self.campaigns), \
             patch.object(inventory.parity, "selected_model_files", return_value=[self.model]), \
             patch.object(inventory.subprocess, "run", side_effect=listing):
            return inventory.discover(self.args, inventory.InventoryScope.ALL)

    def test_native_mpi_selection_is_inherited_without_forced_default(self):
        """Actual inventory launches preserve absent and explicit operator policy."""
        key = "OMPI_MCA_btl_vader_single_copy_mechanism"
        for selected in (None, "cma", "none"):
            with self.subTest(selected=selected), patch.dict(
                    inventory.os.environ, {} if selected is None else {key: selected}, clear=True):
                self.discover()
                observed = self.child_environments[-1]
                self.assertEqual(observed.get(key), selected)
                self.assertEqual(observed["LLAMINAR_FORCE_CPU_ONLY_STARTUP"], "1")
                self.assertEqual(inventory.os.environ.get(key), selected)

    def test_exact_cell_never_lists_unrelated_binary(self):
        self.assertEqual([exact for _, exact, _ in self.discover()], [self.cases[0]])
        self.assertEqual(self.launched, [self.binary])

    def test_chosen_binary_missing_unselected_sibling_still_rejects(self):
        self.listings[self.binary] = self.cases[:1]
        with self.assertRaisesRegex(ValueError, "missing_from_binary=.*sibling"):
            self.discover()
        self.assertEqual(self.launched, [self.binary])

    def test_chosen_binary_unregistered_unselected_sibling_still_rejects(self):
        self.campaigns[0] = self.campaign("selected", self.binary, self.cases[:1])
        with self.assertRaisesRegex(ValueError, "unregistered=.*sibling"):
            self.discover()
        self.assertEqual(self.launched, [self.binary])

    def test_unknown_exact_cell_launches_nothing_and_fails(self):
        self.args.cell = "Suite.ProductionParity/unknown"
        with self.assertRaisesRegex(ValueError, "no all canonical cells"):
            self.discover()
        self.assertEqual(self.launched, [])

    def test_broad_cell_selector_lists_every_eligible_binary(self):
        self.args.cell = ".*"
        self.assertEqual(len(self.discover()), 3)
        self.assertEqual(self.launched, [self.binary, self.other])

    def test_narrow_selector_does_not_hide_unrelated_registered_duplicate(self):
        self.campaigns.append(self.campaign("duplicate", self.other, ("Suite.ProductionParity/other",)))
        with self.assertRaisesRegex(ValueError, "duplicate canonical CTest cell"):
            self.discover()
        self.assertEqual(self.launched, [])

    def test_narrow_selector_rejects_cross_binary_duplicate_before_launch(self):
        self.campaigns.append(self.campaign("duplicate", self.other, self.cases[:1]))
        with self.assertRaisesRegex(ValueError, "duplicate canonical CTest cell"):
            self.discover()
        self.assertEqual(self.launched, [])

    def test_duplicate_within_single_campaign_rejects_before_launch(self):
        self.campaigns[0] = self.campaign("selected", self.binary, self.cases + self.cases[:1])
        with self.assertRaisesRegex(ValueError, "duplicate canonical CTest cell"):
            self.discover()
        self.assertEqual(self.launched, [])



class LosslessRuntimeDocumentTests(unittest.TestCase):
    """Explicit source-owned document projection; no runtime or native admission is fabricated."""
    def record(self):
        """Declare only metadata wire input; this creates no native source authority."""
        overrides = [{"participant": {"world_rank": 0, "backend": "ROCm", "ordinal": 1},
                      "maximum_cache_rows": 64, "cache_physical_byte_limit": 4096}]
        return {"model": "/original/main.gguf", "shared_predictor": "/original/mtp.gguf",
                "runtime": {"configuration_policy": "lossless_declared_document_v1",
                    "configuration_document": {"kind": "llaminar.orchestration-config", "schema_version": 13,
                        "configuration": {"model_path": "/original/main.gguf", "shared_predictor": "/original/mtp.gguf",
                            "ngram_table_policy": {"gpu_adaptive": {"local_hot_cache": {
                                "maximum_cache_rows": 0, "cache_physical_byte_limit": 0,
                                "consumer_overrides": overrides}}}}}}}

    def test_unequal_document_maps_both_original_sources_and_materializes_public_config(self):
        original = self.record()
        mapped = inventory.map_predictor_source(original, lambda p: "/staged/" + Path(p).name)
        with tempfile.TemporaryDirectory() as directory:
            argv = inventory.runtime_server_arguments(mapped, Path(directory), "/staged/main.gguf")
            self.assertEqual(argv[0], "--config")
            actual = json.loads(Path(argv[1]).read_text())["configuration"]
            self.assertEqual(actual["model_path"], "/staged/main.gguf")
            self.assertEqual(actual["shared_predictor"], "/staged/mtp.gguf")
            self.assertEqual(actual["ngram_table_policy"], original["runtime"]["configuration_document"]["configuration"]["ngram_table_policy"])
            with self.assertRaises(FileExistsError):
                inventory.runtime_server_arguments(mapped, Path(directory), "/staged/main.gguf")
        self.assertEqual(original["runtime"]["configuration_document"]["configuration"]["model_path"], "/original/main.gguf")

    def test_malformed_foreign_and_conflicting_arms_reject_before_file_creation(self):
        mutations = [lambda r: r["runtime"].update(server_args=[]),
                     lambda r: r["runtime"].update(configuration_policy="unknown"),
                     lambda r: r["runtime"]["configuration_document"].update(kind="foreign"),
                     lambda r: r["runtime"]["configuration_document"]["configuration"].update(model_path="/foreign/main.gguf"),
                     lambda r: r["runtime"]["configuration_document"]["configuration"].update(shared_predictor="/foreign/mtp.gguf")]
        for mutate in mutations:
            record = self.record(); mutate(record)
            with self.assertRaises(ValueError):
                inventory.map_predictor_source(record, str)
        mapped = inventory.map_predictor_source(self.record(), str)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                inventory.runtime_server_arguments(mapped, Path(directory), "/foreign/main.gguf")
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(inventory.runtime_server_arguments({"runtime": {"server_args": ["--no-mtp"]}}, Path("unused"), "unused"), ["--no-mtp"])

    def test_lossless_benchmark_remaps_same_original_sources_and_rejects_ambiguous_arm(self):
        original=self.record()
        original['diagnostic_benchmark']={**copy.deepcopy(original['runtime']),
            'schema':1,'policy':'diagnostic_declared','certification_eligible':False,'context_length':512}
        mapped=inventory.map_predictor_source(original,lambda p:'/staged/'+Path(p).name)
        runtime=mapped['runtime']['configuration_document']['configuration']
        benchmark=mapped['diagnostic_benchmark']['configuration_document']['configuration']
        self.assertEqual(runtime,benchmark)
        self.assertEqual(benchmark['model_path'],'/staged/main.gguf')
        self.assertEqual(benchmark['shared_predictor'],'/staged/mtp.gguf')
        self.assertEqual(benchmark['ngram_table_policy'],original['runtime']['configuration_document']['configuration']['ngram_table_policy'])
        for mutate in (lambda r:r['diagnostic_benchmark'].update(args=[]),
                lambda r:r['diagnostic_benchmark']['configuration_document']['configuration'].update(shared_predictor='/foreign/mtp.gguf'),
                lambda r:r['diagnostic_benchmark']['configuration_document'].update(kind='foreign')):
            bad=copy.deepcopy(original);mutate(bad)
            with self.assertRaises(ValueError):inventory.map_predictor_source(bad,str)


class ObservedFixedPipelineInventoryTests(unittest.TestCase):
    """Replay declaration binding, original rank retirement and native measured evidence."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.exact = "ObservedFixture.ProductionParity/Exact"
        self.participants = [
            dict(stage=stage, backend="rocm", ordinal=ordinal, uuid=f"gpu-{ordinal}",
                 native_isa="gfx906", numa_node=1, pcie_upstream_bdfs=[f"0000:{stage+8:02x}:00.0"])
            for stage, ordinal in ((0, 2), (0, 4), (1, 6))]
        self.declaration = dict(model_parity_schema=1, id="Exact", model="/model.gguf",
            inventory_role="numerical_diagnostic", e2e=None, cross_host_e2e=[], benchmark=None,
            runtime=dict(generation=dict(mtp_policy="off"), movement_evidence="forbidden",
                         configuration_policy="observed_pipeline_binding_required_v1",
                         observed_pipeline_request=dict(backend="rocm", native_isa="gfx906", mpi_ranks=2,
                                                        stage_sizes=[2, 1])))
        domains = [dict(name=f"stage-{stage}", owner_rank=0, scope="RANK_LOCAL", devices=[
            dict(hostname="observed", numa_node=1, device_type="ROCm", device_ordinal=p["ordinal"])
            for p in self.participants if p["stage"] == stage]) for stage in range(2)]
        self.config = dict(kind="llaminar.orchestration-config", schema_version=1, configuration=dict(
            planning_mode="Apply", mpi_procs=2, execution_rank_selection=[1], domain_definitions=domains,
            pp_stage_definitions=[dict(stage_id=stage, domain_name=f"stage-{stage}") for stage in range(2)]))
        self.exported = dict(schema=1, exact_cell=self.exact, declaration=self.declaration,
            configuration={**self.declaration, "runtime": dict(
                generation=dict(mtp_policy="off"), movement_evidence="forbidden",
                configuration_policy="lossless_declared_document_v1", configuration_document=self.config),
                "diagnostic_benchmark": dict(configuration_policy="lossless_declared_document_v1",
                                             configuration_document=self.config)})
        self.receipts = []
        for rank in range(2):
            path = self.directory / "observed-pipeline" / f"rank-{rank}"
            path.mkdir(parents=True)
            self.receipts.append(path / "observed-pipeline-binding.json")
            self.receipts[-1].write_text(json.dumps(dict(schema=1, cell="Exact", discovery_rank=rank,
                discovery_size=2, active=rank == 1, execution_rank=0 if rank == 1 else None,
                selected_discovery_ranks=[1], original_owner=1, original_hostname="observed",
                original_numa_node=1, requested_native_isa="gfx906", stage_sizes=[2,1],
                participants=self.participants, terminal_join_completed=True, parity_work_succeeded=True)))
        self.config_path = self.receipts[1].parent / "observed-pipeline-config.json"
        self.config_path.write_text(json.dumps(self.exported))

    def bound(self):
        return inventory.bind_observed_pipeline_artifacts(self.exact, self.declaration, self.directory)

    def test_original_binding_rejects_changed_placement_and_inactive_work(self):
        record = self.bound()
        self.assertEqual(record["diagnostic_benchmark"]["configuration_document"], self.config)
        self.assertTrue(inventory.InventoryScope.LOCAL_DIAGNOSTIC.accepts(record))
        self.assertFalse(inventory.InventoryScope.ALL.accepts(record))
        false_selection = copy.deepcopy(self.exported)
        false_selection["configuration"]["runtime"]["configuration_document"]["configuration"]["execution_rank_selection"] = [0]
        self.config_path.write_text(json.dumps(false_selection))
        with self.assertRaisesRegex(ValueError, "rank selection"):
            self.bound()
        self.config_path.write_text(json.dumps(self.exported))
        (self.receipts[0].parent / "observed-pipeline-config.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "inactive original"):
            self.bound()

    def test_runtime_and_terminal_keep_actual_native_owner_roster(self):
        record = self.bound()
        runtime = {key: record["observed_pipeline_binding"][key] for key in (
            "discovery_size", "selected_discovery_ranks", "original_owner", "original_hostname",
            "original_numa_node", "execution_rank", "stage_sizes", "participants")}
        runtime = copy.deepcopy(runtime)
        runtime.update(ple_stage=0, ple_consumers=["ROCm:2", "ROCm:4"],
            fixed_owners=[dict(device=f"ROCm:{ordinal}", stage=stage, participant=member,
                               consumer=consumer, first_head=index*2, heads=2, native_bytes=180)
                for index,(ordinal,stage,member,consumer) in enumerate(((2,0,0,0),(6,1,0,0),(4,0,1,1)))])
        inventory.require_observed_pipeline_runtime(record,runtime)
        report = dict(schema="llaminar.fixed-native-pipeline-terminal.v1", owners={
            owner["device"]: {name: 0 if name.endswith("failures") else 1
                for name in inventory.FIXED_NATIVE_PIPELINE_METRICS} for owner in runtime["fixed_owners"]})
        records = [dict(domain="immutable_lookup", device=device, name=name, phase="request_terminal",
                        kind="counter", count=1, sequence_word_count=1, value=value,
                        tags={"authority":"pipeline_generation_terminal","request":"0"})
                   for device, fields in report["owners"].items() for name,value in fields.items()]
        document = dict(observed_pipeline_binding=runtime, fixed_native_pipeline_terminal=report)
        self.assertEqual(inventory.require_observed_pipeline_terminal(document,records),report)
        with self.assertRaisesRegex(ValueError, "repeats"):
            inventory.require_observed_pipeline_terminal(document,[*records,records[0]])
        wrong_count = copy.deepcopy(records)
        wrong_count[-1]["sequence_word_count"] = 0
        with self.assertRaisesRegex(ValueError, "foreign device, phase or count"):
            inventory.require_observed_pipeline_terminal(document,wrong_count)
        for name in ("adaptive_lookup_calls", "fixed_head_lookup_calls"):
            legacy = dict(records[0], name=name, phase="generation_terminal",
                          tags={"authority":"terminal_device_generation_control","request":"0"})
            with self.assertRaisesRegex(ValueError, "legacy immutable lookup"):
                inventory.require_observed_pipeline_terminal(document,[*records,legacy])
        with self.assertRaisesRegex(ValueError, "totals differ"):
            inventory.require_observed_pipeline_terminal(document,records[:-1])
        runtime["participants"][0]["uuid"] = "other"
        with self.assertRaisesRegex(ValueError, "substituted original"):
            inventory.require_observed_pipeline_runtime(record,runtime)

    def test_release_guardian_owns_one_original_rankfile_session(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from run_model_kld_canaries import original_local_launch
        from run_model_parity_benchmarks import run_declared_guardian
        binary = self.directory / "llaminar"
        launcher = self.directory / "llaminar_mpi_session"
        mpi = self.directory / "mpirun"
        for path in (binary, launcher, mpi):
            path.touch()
            path.chmod(0o700)
        rankfile = self.directory / "original-ranks"
        rankfile.write_text("original observer placement")
        args = SimpleNamespace(binary=binary,mpi_rankfile=rankfile)
        adapter = SimpleNamespace(topology=dict(expected_config=self.config))
        numerical, release = self.directory / "numerical", self.directory / "release"
        numerical.mkdir()
        release.mkdir()
        env = dict(PATH=str(self.directory))
        _, pins, proof = original_local_launch(args,adapter,None,numerical,[str(binary)],env)
        command, release_pins, release_proof = original_local_launch(
            args,adapter,None,release,[str(binary),"benchmark"],env,prepare_session=False)
        self.assertTrue((numerical / "mpi-session").is_dir())
        self.assertFalse((release / "mpi-session").exists())
        self.assertEqual(pins,release_pins)
        self.assertEqual(proof,release_proof)
        self.assertIn("--rankfile",command)
        with patch("run_model_parity_benchmarks.subprocess.Popen") as process, \
             patch("mpi_session_retirement.require_retired") as retire:
            process.return_value.wait.return_value = 0
            process.return_value.poll.return_value = 0
            run_declared_guardian(command,env,release)
        process.assert_called_once()
        retire.assert_called_once()
        self.assertTrue((release / "mpi-session").is_dir())

    def test_original_inactive_observer_rank_is_terminal_only(self):
        from model_kld_corpus_v2 import metadata_digest, validate_distributed_rank_reports
        manifest = dict(schema=2)
        topology = dict(expected_config=self.config, runtime_config_digest="selected")
        document = dict(observed_pipeline_binding=dict(selected_discovery_ranks=[1], discovery_size=2,
                                                       original_owner=1, execution_rank=0))
        files = [self.directory / f"rank-{rank}.json" for rank in range(2)]
        for rank, path in enumerate(files):
            path.write_text(json.dumps(dict(schema="llaminar.qfn-observed-pipeline-rank.v1", discovery_rank=rank,
                discovery_size=2, execution_rank=0 if rank == 1 else None, active=rank == 1,
                native_retired=True, runtime_config_digest="selected", corpus_manifest_digest=metadata_digest(manifest))))
        self.assertEqual(len(validate_distributed_rank_reports(manifest,topology,document,self.directory/"logits.json")),2)
        wrong = json.loads(files[0].read_text())
        wrong["active"] = True
        files[0].write_text(json.dumps(wrong))
        with self.assertRaisesRegex(ValueError,"terminal role"):
            validate_distributed_rank_reports(manifest,topology,document,self.directory/"logits.json")


if __name__ == "__main__":
    unittest.main()
