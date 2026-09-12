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
                "model": "/models/model.gguf", "e2e": {"server_args": ["opaque"]} if tagged else None}}


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


if __name__ == "__main__":
    unittest.main()
