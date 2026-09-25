#!/usr/bin/env python3
"""Device-free contracts for immutable-image E2E, PR, and release publication.

Fixtures model metadata only. Real HTTP/math evidence remains the canonical
runners' responsibility; these tests exercise pair admission, full-suite joins,
publication boundaries and the exact-data renderer without GPUs or weights.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import run_published_image_suite as suite
import wait_for_develop_image_gate as image_wait
import publish_master_release as master_release
import apply_master_ruleset as master_ruleset
import post_pr_benchmark_rag as benchmark_rag
import publish_pr_high_water as high_water
import run_model_parity_e2e as e2e
import published_benchmark_chart as chart
import build_public_docs as public_docs
from production_artifacts import digest, ratchet, write_json
from test_gpu_driver_diagnostics import clean_evidence as clean_driver_evidence


def image_pair():
    """Build a full-backend pair with intentionally different immutable IDs."""
    source = {"revision": "a" * 40, "tree": "b" * 40, "dirty": False}
    images = {}
    for index, isa in enumerate(suite.ISAS, 1):
        images[isa] = {"id": "sha256:" + str(index) * 64, "layers": [f"layer-{isa}"],
            "tag": suite.runtime_tag("ghcr.io/llaminar/llaminar:develop", isa),
            "registry_ref": "ghcr.io/llaminar/llaminar@sha256:" + str(index) * 64,
            "labels": {"org.opencontainers.image.revision": source["revision"],
                "org.llaminar.source_tree": source["tree"], "org.llaminar.build_type": "Release",
                "org.llaminar.cpu_isa": isa, "org.llaminar.image_role": "runtime",
                "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON"}}
    return {"schema": 1, "source": source, "images": images,
            "repository": "Llaminar/llaminar", "branch": "develop", "workflow_revision": "c" * 40}


def inventory_image(source):
    """Describe an exact full-matrix test companion already in local Docker."""
    role = suite.pipeline.ImageRole.TEST_RUNNER
    tag = suite.pipeline.source_image_tag(source, "AVX2", role)
    identity = "sha256:" + "3" * 64
    return {"tag": tag, "id": identity, "layers": ["test-layer"], "labels": {
        "org.opencontainers.image.revision": source["revision"],
        "org.llaminar.source_tree": source["tree"],
        "org.llaminar.build_type": "Release", "org.llaminar.cpu_isa": "AVX2",
        "org.llaminar.cuda": "ON", "org.llaminar.rocm": "ON",
        "org.llaminar.image_role": role.value,
        "org.llaminar.integration_skipped": "0",
        "org.llaminar.test_runner_inventory": suite.pipeline.TestRunnerInventory.FULL_MATRIX.value,
    }}


def full_inventory():
    """One tagged and one untagged case prove the projection, not a copied list."""
    config = {"model_parity_schema": 1, "id": "Tiny_ROCm", "model": "/models/model.gguf",
              "e2e": {"server_args": ["--device", "rocm:0"], "context_length": 8192}}
    row = {"case": "Suite/Tiny.ProductionParity/ROCm", "campaign": "ProductionCampaign",
           "backends": "ROCm", "configuration": config, "model_files": [config["model"]]}
    untagged = deepcopy(row)
    untagged["case"] = "Suite/Tiny.ProductionParity/Diagnostic"
    untagged["configuration"]["id"] = "Tiny_Diagnostic"
    untagged["configuration"]["e2e"] = None
    return {"schema": 1, "source_revision": "a" * 40, "scope": "all", "cells": [row, untagged]}


def write_e2e_bundle(root, pair=None):
    """Write compact successful evidence using the canonical validation shape."""
    pair = pair or image_pair()
    inventory = full_inventory()
    manifest = suite.pipeline.e2e_projection(inventory, pair["source"]["revision"])
    write_json(root / "images.json", pair)
    write_json(root / "all-cells.json", inventory)
    write_json(root / "manifest.json", manifest)
    receipt = {"schema": 1, "complete": True, "images_digest": digest(pair),
               "manifest_digest": digest(manifest), "reports": {}}
    for isa in suite.ISAS:
        report = {"schema": 1, "image": pair["images"][isa]["id"],
            "source_revision": pair["source"]["revision"], "manifest_digest": digest(manifest),
            "selected": len(manifest["cells"]), "correctness_passed": True,
            "cells": [{**row, "outcome": "passed", "return_code": 0} for row in manifest["cells"]]}
        write_json(root / isa.lower() / "e2e.json", report)
        receipt["reports"][isa] = digest(report)
    write_json(root / "e2e-receipt.json", receipt)
    return manifest


def benchmark_result():
    """Create renderer data with deliberately different ISA timing values."""
    config = full_inventory()["cells"][0]["configuration"]
    config = {**config, "model": "Qwen-Example-Q8.gguf"}
    variants = {isa: {"workload": {"decode_tokens": 256}, "cells": [{
        "case": "Suite/Tiny.ProductionParity/ROCm", "prefill_tokens": 512,
        "identity": {"configuration": config, "cpu_isa": isa},
        "tokens_per_second": {"prefill": 1500.25 + index, "decode": 60.125 + index}}]}
        for index, isa in enumerate(suite.ISAS)}
    return {"source": image_pair()["source"], "repository": "Llaminar/llaminar",
            "recorded_at_utc": "2026-09-20T00:00:00+00:00", "variants": variants}


def benchmark_report(pair, manifest, e2e, isa):
    """Use complete runner metadata to test the second evidence join."""
    rows = []
    for cell in manifest["cells"]:
        config = cell["configuration"]
        rows.append({"case": cell["case"], "identity": {"cpu_isa": isa,
            "configuration": {**config, "model": Path(config["model"]).name}},
            "tokens_per_second": {"prefill": 1000.0, "decode": 60.0}})
    baseline = json.loads((ROOT / "benchmarks/production/high_water.json").read_text())
    proposed, comparisons = ratchet(baseline, rows)
    return {"complete": True, "passed": True, "diagnostic": False,
            "source_revision": pair["source"]["revision"], "image": pair["images"][isa]["id"],
            "manifest_digest": digest(manifest), "e2e_report_digest": digest(e2e), "cells": rows,
            "baseline_digest": digest(baseline), "proposed_high_water": proposed,
            "comparisons": comparisons}


def public_release_bundle(root, tag="2026-09-24.1"):
    """Use the release publisher's actual asset map for a tiny archived proof."""
    pair = image_pair()
    pair["workflow_revision"] = pair["source"]["revision"]
    e2e_root, benchmark_root = root / "e2e", root / "benchmarks"
    e2e_root.mkdir(parents=True)
    benchmark_root.mkdir()
    manifest = write_e2e_bundle(e2e_root, pair)
    result = {"source": pair["source"], "images": pair["images"],
              "repository": pair["repository"], "branch": "develop", "passed": True,
              "scope": "full-http-e2e-and-benchmarks", "full_image_certification": False,
              "recorded_at_utc": "2026-09-24T00:00:00+00:00", "variants": {}}
    for isa in suite.ISAS:
        e2e_report = json.loads((e2e_root / isa.lower() / "e2e.json").read_text())
        report = benchmark_report(pair, manifest, e2e_report, isa)
        for row in report["cells"]:
            row["prefill_tokens"] = 512
        write_json(benchmark_root / isa.lower() / "benchmarks.json", report)
        result["variants"][isa] = {
            "report_digest": digest(report), "e2e_report_digest": digest(e2e_report),
            "comparisons": report["comparisons"], "cells": report["cells"],
            "workload": {"decode_tokens": 256}}
    write_json(benchmark_root / "results.json", result)
    (benchmark_root / "benchmarks.svg").write_text(chart.render_chart(result))
    paths = master_release.release_assets(e2e_root, benchmark_root, root)
    metadata = {"id": int(tag.split(".")[1]), "tag_name": tag,
                "draft": False, "prerelease": False,
                "published_at": "2026-09-24T00:00:00Z",
                "body": "# Release notes\n\nImproved inference. <script>bad()</script>",
                "assets": [{"name": path.name} for path in paths]}
    return metadata, root / "assets"


class PublicDocumentationTests(unittest.TestCase):
    """Published pages retain complete certificates and cannot invent a pass."""

    def test_release_reports_render_both_isas_and_keep_original_downloads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, assets = public_release_bundle(root / "bundle")
            evidence = public_docs.load_evidence("Llaminar/llaminar", metadata, assets)
            destination = root / "page"
            public_docs.render_release(evidence, destination)
            self.assertIn("All **1 configurations**", (destination / "index.md").read_text())
            self.assertIn("Passed | Passed", (destination / "e2e.md").read_text())
            self.assertIn("1,000.0", (destination / "benchmarks.md").read_text())
            self.assertIn("&lt;script&gt;", (destination / "release-notes.md").read_text())
            self.assertNotIn("<script>", (destination / "release-notes.md").read_text())
            for asset in assets.iterdir():
                self.assertEqual(asset.read_bytes(), (destination / "assets" / asset.name).read_bytes())

    def test_incomplete_or_mixed_certificates_do_not_become_public_green_pages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, assets = public_release_bundle(root / "bundle")
            report_path = assets / "benchmark-avx2.json"
            original = json.loads(report_path.read_text())
            for field, value in (("complete", False), ("image", "wrong"),
                                 ("cells", []), ("diagnostic", True)):
                with self.subTest(field=field):
                    write_json(report_path, {**original, field: value})
                    with self.assertRaisesRegex(ValueError, "benchmark certificate"):
                        public_docs.load_evidence("Llaminar/llaminar", metadata, assets)
            write_json(report_path, original)
            e2e_path = assets / "e2e-avx2.json"
            changed = json.loads(e2e_path.read_text())
            changed["unrecorded_extra_field"] = True
            write_json(e2e_path, changed)
            with self.assertRaisesRegex(ValueError, "differs from its receipt"):
                public_docs.load_evidence("Llaminar/llaminar", metadata, assets)

    def test_documentation_does_not_rejudge_historical_high_water(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata, assets = public_release_bundle(root / "bundle")
            # The archived proof must not read a future checkout's baseline.
            with patch.object(suite, "ROOT", root / "no-current-baseline"):
                public_docs.load_evidence("Llaminar/llaminar", metadata, assets)
            altered = json.loads((assets / "benchmark-results.json").read_text())
            altered["variants"]["AVX2"]["cells"][0]["tokens_per_second"]["decode"] += 1
            write_json(assets / "benchmark-results.json", altered)
            with self.assertRaisesRegex(ValueError, "compact benchmark differs"):
                public_docs.load_evidence("Llaminar/llaminar", metadata, assets)

    def test_archive_is_complete_and_latest_follows_github_not_a_benchmark_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "index.md").write_text("# Authored documentation\n")
            records, bundles = [], {}
            for tag in ("2026-09-24.1", "2026-09-24.2"):
                record, assets = public_release_bundle(root / tag, tag)
                records.append(record)
                bundles[tag] = assets
            def download(repository, record, destination):
                """Supply the unchanged offline release attachments."""
                import shutil
                shutil.copytree(bundles[record["tag_name"]], destination)
            output = root / "generated"
            with patch.object(public_docs, "published_releases", return_value=(records, records[0]["tag_name"])), \
                 patch.object(public_docs, "download_assets", side_effect=download):
                public_docs.prepare_docs("Llaminar/llaminar", output, source)
            self.assertIn("2026-09-24.1", (output / "docs/releases/latest/index.md").read_text())
            self.assertTrue((output / "docs/releases/2026-09-24.2/index.md").exists())
            self.assertEqual((source / "index.md").read_text(), "# Authored documentation\n")
            self.assertFalse((source / "releases").exists())

    def test_archive_paginates_and_excludes_drafts_and_prereleases(self):
        rows = [{"tag_name": f"2026-09-24.{n}", "draft": False, "prerelease": False}
                for n in range(1, 101)]
        tail = [{"tag_name": "2026-09-25.1", "draft": True, "prerelease": False},
                {"tag_name": "2026-09-26.1", "draft": False, "prerelease": True}]
        with patch.object(public_docs, "github_json", side_effect=[rows, tail, {"tag_name": "2026-09-24.10"}]):
            archive, latest = public_docs.published_releases("Llaminar/llaminar")
        self.assertEqual(len(archive), 100)
        self.assertEqual(archive[0]["tag_name"], "2026-09-24.100")
        self.assertEqual(latest, "2026-09-24.10")

    def test_empty_archive_still_publishes_authored_documentation(self):
        """General docs can launch before the first certified release exists."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "index.md").write_text("# Documentation\n")
            with patch.object(public_docs, "github_json", return_value=[]) as github, \
                 patch.object(public_docs, "download_assets") as download:
                nav = public_docs.prepare_docs("Llaminar/llaminar", root / "generated", source)
            github.assert_called_once()
            download.assert_not_called()
            self.assertEqual(nav, [{"Archive": "releases/index.md"}])
            self.assertIn("No dated releases", (root / "generated/docs/releases/index.md").read_text())
            self.assertFalse((root / "generated/docs/releases/latest").exists())

    def test_docs_workflow_uses_pages_and_never_invokes_inference(self):
        import yaml
        workflow = yaml.load((ROOT / ".github/workflows/docs.yml").read_text(), Loader=yaml.BaseLoader)
        self.assertIn("workflow_call", workflow["on"])
        self.assertEqual(workflow["permissions"]["pages"], "write")
        self.assertEqual(workflow["jobs"]["publish"]["runs-on"], "ubuntu-24.04")
        steps = workflow["jobs"]["publish"]["steps"]
        self.assertTrue(any(step.get("uses", "").startswith("actions/deploy-pages@") for step in steps))
        commands = "\n".join(step.get("run", "") for step in steps)
        self.assertIn("scripts/ci/build_public_docs.py", commands)
        self.assertNotIn("run_production_pipeline.py", commands)
        self.assertNotIn("docker", commands)


class PublishedImageSuiteTests(unittest.TestCase):
    """Wrong provenance or incomplete evidence cannot publish a convincing chart."""

    def test_e2e_runner_collects_failed_timeout_and_passing_cells(self):
        """One red cell must not hide later independent HTTP configurations."""
        from test_server_tool_calling import row_for, tools
        from test_server_execution_contract import automatic_evidence

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_model = root / "model.gguf"
            profile = {"server_args": ["--device", "cpu:0"], "context_length": 8192,
                       "minimum_prompt_tokens": 4096, "generation_tokens": 2048,
                       "request_timeout_seconds": 600, "readiness_timeout_seconds": 180,
                       "cell_timeout_seconds": {"AVX512": 900, "AVX2": 1200},
                       "thinking_modes": "both", "movement_evidence": "not_applicable",
                       "tool_calling": "required",
                       "planning": {"mode": "auto", "strategy": "single", "mpi_ranks": 1,
                                    "device_counts": {"cpu": 1}}}
            campaign = SimpleNamespace(group=SimpleNamespace(backends="CPU"))
            selected = [(campaign, f"Suite.Cell/{index}",
                         {"id": f"cell-{index}", "model": str(source_model),
                          "e2e": profile}) for index in range(3)]
            workspace = SimpleNamespace(models=root / "staged", persistent=True,
                                        protect_published_models=MagicMock())
            staged = [SimpleNamespace(source_path=str(source_model), filename="model.gguf")]
            report_path = root / "e2e.json"
            image = "sha256:" + "a" * 64
            outcomes = iter((1, 124, 0))

            def run_cell(_command, environment, _log, *, budget):
                """Successful shell fixtures retain tool and driver-health evidence."""
                code = next(outcomes)
                if code == 0:
                    write_json(Path(environment["LLAMINAR_E2E_LOG_DIR"]) / "cell.driver-diagnostics.json",
                               clean_driver_evidence())
                    write_json(Path(environment["LLAMINAR_E2E_LOG_DIR"]) / "tool_calling_results.json",
                               {"schema": 1, "complete": True,
                                "results": [row_for(probe) for probe in tools.PROBES]})
                    write_json(Path(environment["LLAMINAR_E2E_LOG_DIR"]) / "automatic_selection_results.json",
                               automatic_evidence(["CPU"], strategy="single"))
                return code

            with patch.object(e2e, "discover", return_value=selected), \
                 patch.object(e2e, "image_identity", return_value={"id": image,
                              "labels": {"org.llaminar.cpu_isa": "AVX2"}}), \
                 patch.object(e2e, "validate_attached_execution"), \
                 patch.object(e2e.parity, "model_staging_workspace",
                              return_value=nullcontext(workspace)), \
                 patch.object(e2e.parity, "stage_models_in_ramdisk",
                              return_value=(staged, None)) as stage, \
                 patch.object(e2e, "run_e2e_process", side_effect=run_cell) as run, \
                 patch.object(e2e, "validate_long_context_evidence") as checks:
                result = e2e.main(["--container-image", image,
                                   "--source-revision", "a" * 40,
                                   "--model-ramdisk-root", str(root),
                                   "--report", str(report_path)])
            self.assertEqual(result, 1)
            self.assertEqual(stage.call_count, 1)
            self.assertEqual(run.call_count, 3)
            self.assertEqual([call.kwargs["budget"].timeout_seconds for call in run.call_args_list],
                             [1200, 1200, 1200])
            checks.assert_called_once()
            report = json.loads(report_path.read_text())
            self.assertEqual(report["selected"], 3)
            self.assertEqual(report["cpu_isa"], "AVX2")
            self.assertEqual([row["timeout_seconds"] for row in report["cells"]], [1200, 1200, 1200])
            self.assertEqual([row["outcome"] for row in report["cells"]],
                             ["failed", "cell_timeout", "passed"])
            self.assertEqual(report["failed_cells"], ["Suite.Cell/0", "Suite.Cell/1"])
            self.assertFalse(report["correctness_passed"])

    def test_published_e2e_runs_both_isas_before_reporting_complete_red_cells(self):
        """A complete red AVX512 report retains AVX2 evidence but blocks admission."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = image_pair()
            inventory = full_inventory()
            manifest = suite.pipeline.e2e_projection(inventory, pair["source"]["revision"])
            write_json(root / "images.json", pair)
            write_json(root / "all-cells.json", inventory)
            write_json(root / "manifest.json", manifest)

            def drive(_args, _driver, _command, lane, _log):
                isa = lane.name.upper()
                failed = isa == "AVX512"
                row = manifest["cells"][0]
                report = {"schema": 1, "image": pair["images"][isa]["id"],
                          "source_revision": pair["source"]["revision"],
                          "manifest_digest": digest(manifest), "selected": 1,
                          "correctness_passed": not failed,
                          "cells": [{"case": row["case"],
                                     "configuration": row["configuration"],
                                     "return_code": 1 if failed else 0,
                                     "outcome": "failed" if failed else "passed"}]}
                write_json(lane / "e2e.json", report)
                if failed:
                    raise subprocess.CalledProcessError(1, "HTTP E2E")

            with patch.object(suite, "build_driver", return_value="tools") as build, \
                 patch.object(suite, "run_in_driver", side_effect=drive) as run:
                with self.assertRaisesRegex(ValueError, "cell failures remain"):
                    suite.run_e2e(SimpleNamespace(model_ramdisk_root=root), pair,
                                  manifest, root)
            build.assert_called_once()
            self.assertEqual(run.call_count, 2)
            receipt = json.loads((root / "e2e-receipt.json").read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(set(receipt["reports"]), set(suite.ISAS))
            self.assertEqual(receipt["failed_cells"], {"AVX512": [manifest["cells"][0]["case"]],
                                                       "AVX2": []})
            with self.assertRaisesRegex(ValueError, "incomplete or stale"):
                suite.admit_e2e(root, pair)

    def test_incomplete_e2e_report_stops_before_other_isa(self):
        """A driver crash cannot masquerade as one ordinary failed cell."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = image_pair()
            manifest = suite.pipeline.e2e_projection(full_inventory(),
                                                     pair["source"]["revision"])

            def interrupted(_args, _driver, _command, lane, _log):
                write_json(lane / "e2e.json", {"schema": 1, "selected": 1,
                                               "correctness_passed": False, "cells": []})
                raise subprocess.CalledProcessError(1, "driver")

            with patch.object(suite, "build_driver", return_value="tools"), \
                 patch.object(suite, "run_in_driver", side_effect=interrupted) as run:
                with self.assertRaisesRegex(ValueError, "complete selected inventory"):
                    suite.run_e2e(SimpleNamespace(model_ramdisk_root=root), pair,
                                  manifest, root)
            self.assertEqual(run.call_count, 1)
            receipt = json.loads((root / "e2e-receipt.json").read_text())
            self.assertFalse(receipt["complete"])
            self.assertEqual(receipt["reports"], {})

    def test_model_free_gate_cannot_shadow_full_matrix_inventory_tag(self):
        """Same source and ISA still have disjoint typed test-runner identities."""
        source = image_pair()["source"]
        role = suite.pipeline.ImageRole.TEST_RUNNER
        full = suite.pipeline.source_image_tag(source, "AVX2", role,
            test_inventory=suite.pipeline.TestRunnerInventory.FULL_MATRIX)
        model_free = suite.pipeline.source_image_tag(source, "AVX2", role,
            test_inventory=suite.pipeline.TestRunnerInventory.MODEL_FREE)
        self.assertNotEqual(full, model_free)
        self.assertIn("full-matrix-test-runner", full)
        self.assertIn("model-free-test-runner", model_free)
        runtime = suite.pipeline.ImageRole.RUNTIME
        self.assertEqual(
            suite.pipeline.source_image_tag(source, "AVX2", runtime,
                test_inventory=suite.pipeline.TestRunnerInventory.FULL_MATRIX),
            suite.pipeline.source_image_tag(source, "AVX2", runtime,
                test_inventory=suite.pipeline.TestRunnerInventory.MODEL_FREE))

    def test_exact_local_inventory_companion_is_reused_without_build(self):
        """A warm manual rerun does not export the same multi-gigabyte image."""
        source = image_pair()["source"]
        image = inventory_image(source)
        args = SimpleNamespace(cpu_isa="AVX2")
        with patch.object(suite.subprocess, "check_output", return_value=image["id"] + "\n"), \
             patch.object(suite, "image_identity", return_value={
                 key: value for key, value in image.items() if key != "tag"}), \
             patch.object(suite.pipeline, "build") as build:
            result = suite.inventory_companion(args, source, Path("/unused"))
        self.assertEqual(result, {"test-runner": image})
        build.assert_not_called()

    def test_missing_inventory_companion_builds_once(self):
        """A cold runner still obtains the same source-authenticated companion."""
        source = image_pair()["source"]
        args = SimpleNamespace(cpu_isa="AVX2")
        expected = {"test-runner": inventory_image(source)}
        with patch.object(suite.subprocess, "check_output", return_value=""), \
             patch.object(suite.pipeline, "build", return_value=expected) as build:
            self.assertEqual(suite.inventory_companion(args, source, Path("/unused")), expected)
        build.assert_called_once_with(args, source, Path("/unused"),
                                      roles=(suite.pipeline.ImageRole.TEST_RUNNER,))

    def test_conflicting_inventory_tag_fails_instead_of_rebuilding(self):
        """A local alias never substitutes a different source or test inventory."""
        source = image_pair()["source"]
        image = inventory_image(source)
        image["labels"]["org.llaminar.test_runner_inventory"] = "incomplete"
        args = SimpleNamespace(cpu_isa="AVX2")
        with patch.object(suite.subprocess, "check_output", return_value=image["id"] + "\n"), \
             patch.object(suite, "image_identity", return_value={
                 key: value for key, value in image.items() if key != "tag"}), \
             patch.object(suite.pipeline, "build") as build:
            with self.assertRaisesRegex(ValueError, "does not match"):
                suite.inventory_companion(args, source, Path("/unused"))
        build.assert_not_called()

    def test_docker_listing_failure_does_not_masquerade_as_cold_cache(self):
        """A daemon error must not trigger an expensive or misleading rebuild."""
        source = image_pair()["source"]
        args = SimpleNamespace(cpu_isa="AVX2")
        with patch.object(suite.subprocess, "check_output", side_effect=
                          suite.subprocess.CalledProcessError(1, "docker image ls")), \
             patch.object(suite.pipeline, "build") as build:
            with self.assertRaises(suite.subprocess.CalledProcessError):
                suite.inventory_companion(args, source, Path("/unused"))
        build.assert_not_called()

    def test_canonical_tags(self):
        base = suite.branch_image("Llaminar/llaminar", "develop")
        self.assertEqual(base, "ghcr.io/llaminar/llaminar:develop")
        self.assertEqual(suite.runtime_tag(base, "AVX2"), base + "-avx2")
        with self.assertRaises(ValueError):
            suite.branch_image("Llaminar/llaminar", "feature/ambiguous")

    def test_pr_pair_rejects_a_different_published_source_revision(self):
        pair = image_pair()
        base = "ghcr.io/llaminar/llaminar:develop-" + pair["source"]["revision"]
        identities = {suite.runtime_tag(base, isa): pair["images"][isa]
                      for isa in suite.ISAS}
        args = SimpleNamespace(image=base, branch="develop",
                               repository="Llaminar/llaminar",
                               expected_source_revision="f" * 40)
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(suite.pipeline, "run"), \
                 patch.object(suite, "image_identity", side_effect=lambda tag: identities[tag]), \
                 patch.object(suite.subprocess, "check_output", side_effect=lambda command, **_: json.dumps([
                     {"RepoDigests": [identities[next(tag for tag in identities if
                         identities[tag]["id"] == command[-1])]["registry_ref"]]}])), \
                 patch.object(suite, "git", return_value="c" * 40):
                with self.assertRaisesRegex(ValueError, "PR head revision"):
                    suite.pull_pair(args, Path(temporary))

    def test_pair_admission_checks_every_isa_and_source_label(self):
        suite.validate_pair(image_pair())
        for label in ("org.llaminar.cpu_isa", "org.opencontainers.image.revision",
                      "org.llaminar.source_tree", "org.llaminar.build_type",
                      "org.llaminar.cuda", "org.llaminar.rocm", "org.llaminar.image_role"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                pair = image_pair()
                pair["images"]["AVX2"]["labels"][label] = "wrong"
                suite.validate_pair(pair)

    def test_pair_requires_both_images_and_registry_provenance(self):
        for mutate in (lambda pair: pair["images"].pop("AVX2"),
                       lambda pair: pair["images"]["AVX2"].update(registry_ref="local:tag"),
                       lambda pair: pair["images"]["AVX2"].update(id=pair["images"]["AVX512"]["id"])):
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                pair = image_pair()
                mutate(pair)
                suite.validate_pair(pair)

    def test_both_full_e2e_suites_are_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_e2e_bundle(root)
            admitted, reports = suite.admit_e2e(root, image_pair())
            self.assertEqual(admitted, manifest)
            self.assertEqual(set(reports), set(suite.ISAS))

    def test_workflow_only_commit_preserves_unchanged_image_evidence(self):
        """Harness repairs do not invalidate E2E for the same exact runtime pair."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_e2e_bundle(root)
            pair = image_pair()
            pair["workflow_revision"] = "d" * 40
            admitted, reports = suite.admit_e2e(root, pair)
            self.assertEqual(admitted, manifest)
            self.assertEqual(set(reports), set(suite.ISAS))

    def test_changed_latest_tag_rejects_earlier_e2e(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_e2e_bundle(root)
            pair = image_pair()
            pair["images"]["AVX512"]["id"] = "sha256:" + "f" * 64
            with self.assertRaisesRegex(ValueError, "differs from E2E"):
                suite.admit_e2e(root, pair)

    def test_one_isa_pass_is_not_pair_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_e2e_bundle(root)
            receipt = json.loads((root / "e2e-receipt.json").read_text())
            receipt["reports"].pop("AVX2")
            write_json(root / "e2e-receipt.json", receipt)
            with self.assertRaisesRegex(ValueError, "incomplete or stale"):
                suite.admit_e2e(root, image_pair())

    def test_forged_projection_cannot_drop_a_tagged_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_e2e_bundle(root)
            inventory = full_inventory()
            inventory["cells"][1]["configuration"]["e2e"] = deepcopy(inventory["cells"][0]["configuration"]["e2e"])
            write_json(root / "all-cells.json", inventory)
            with self.assertRaisesRegex(ValueError, "complete canonical"):
                suite.admit_e2e(root, image_pair())

    def test_edited_e2e_result_fails_receipt_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_e2e_bundle(root)
            path = root / "avx2/e2e.json"
            report = json.loads(path.read_text())
            report["extra_after_publication"] = True
            write_json(path, report)
            with self.assertRaisesRegex(ValueError, "completed receipt"):
                suite.admit_e2e(root, image_pair())

    def test_benchmarks_require_exact_complete_successful_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_e2e_bundle(root)
            pair = image_pair()
            _, evidence = suite.admit_e2e(root, pair)
            report = benchmark_report(pair, manifest, evidence["AVX2"], "AVX2")
            suite.validate_benchmark(report, pair, manifest, evidence["AVX2"], "AVX2")
            for mutation in (lambda row: row.update(complete=False),
                             lambda row: row.update(passed=False),
                             lambda row: row.update(diagnostic=True),
                             lambda row: row.update(image=pair["images"]["AVX512"]["id"]),
                             lambda row: row.update(e2e_report_digest="stale"),
                             lambda row: row["cells"].append(deepcopy(row["cells"][0])),
                             lambda row: row["cells"][0]["identity"].update(cpu_isa="AVX512"),
                             lambda row: row["cells"][0]["identity"]["configuration"].update(model="different.gguf")):
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    changed = deepcopy(report)
                    mutation(changed)
                    suite.validate_benchmark(changed, pair, manifest, evidence["AVX2"], "AVX2")

    def test_complete_red_benchmark_can_be_reported_but_never_certified(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = write_e2e_bundle(root)
            pair = image_pair()
            _, evidence = suite.admit_e2e(root, pair)
            report = benchmark_report(pair, manifest, evidence["AVX2"], "AVX2")
            row = report["cells"][0]
            baseline = {"schema": 1, "regression_threshold_pct": 10, "entries": {
                digest(row["identity"]): {"identity": row["identity"],
                                          "tokens_per_second": {"prefill": 2000.0,
                                                                "decode": 120.0}}}}
            (root / "benchmarks/production").mkdir(parents=True)
            write_json(root / "benchmarks/production/high_water.json", baseline)
            proposed, comparisons = ratchet(baseline, report["cells"])
            report.update(baseline_digest=digest(baseline), proposed_high_water=proposed,
                          comparisons=comparisons, passed=False)
            with patch.object(suite, "ROOT", root):
                suite.validate_benchmark(report, pair, manifest, evidence["AVX2"], "AVX2",
                                         allow_regressions=True)
                with self.assertRaisesRegex(ValueError, "regressed"):
                    suite.validate_benchmark(report, pair, manifest, evidence["AVX2"], "AVX2")
                report["comparisons"][0]["passed"] = True
                with self.assertRaisesRegex(ValueError, "canonical high water"):
                    suite.validate_benchmark(report, pair, manifest, evidence["AVX2"], "AVX2",
                                             allow_regressions=True)

    def test_pr_rag_exposes_every_phase_and_blocks_red(self):
        revision = "a" * 40
        rows = [
            {"case": "Model/CellA", "phase": "prefill", "current": 120.0,
             "high_water": 100.0, "passed": True},
            {"case": "Model/CellA", "phase": "decode", "current": 95.0,
             "high_water": 100.0, "passed": True},
            {"case": "Model/CellB", "phase": "prefill", "current": 70.0,
             "high_water": 100.0, "passed": False},
            {"case": "Model/CellB", "phase": "decode", "current": 60.0,
             "high_water": None, "passed": True},
        ]
        result = {"source": {"revision": revision},
                  "scope": "full-http-e2e-and-benchmarks", "passed": False,
                  "regression_threshold_pct": 10,
                  "variants": {"AVX512": {"comparisons": rows},
                               "AVX2": {"comparisons": rows}}}
        body = benchmark_rag.render(result, revision, "https://example.test/run")
        self.assertIn("🔴 BLOCKED", body)
        self.assertIn("4 green / 2 amber / 2 red", body)
        self.assertIn("AVX512: 2 cells", body)
        self.assertIn("AVX2: 2 cells", body)
        self.assertIn("-30.0%", body)
        with self.assertRaisesRegex(ValueError, "exact PR image pair"):
            benchmark_rag.render(result, "b" * 40, "https://example.test/run")
        result["passed"] = True
        with self.assertRaisesRegex(ValueError, "canonical benchmark evidence"):
            benchmark_rag.render(result, revision, "https://example.test/run")

    def test_pr_rag_updates_owned_comment_instead_of_spamming(self):
        marker = benchmark_rag.MARKER
        comments = [{"id": 9, "body": marker, "user": {"login": "github-actions[bot]"}}]
        with patch.object(benchmark_rag, "github_json", side_effect=[comments, {}]) as api:
            benchmark_rag.publish("Llaminar/llaminar", 42, marker + " green")
        self.assertEqual(api.call_args_list[1].args[0],
                         "repos/Llaminar/llaminar/issues/comments/9")
        self.assertEqual(api.call_args_list[1].kwargs["method"], "PATCH")
        later = [{"id": 91, "body": marker, "user": {"login": "github-actions[bot]"}}]
        with patch.object(benchmark_rag, "github_json",
                          side_effect=[[{"id": i} for i in range(100)], later, {}]) as api:
            benchmark_rag.publish("Llaminar/llaminar", 42, marker + " updated")
        self.assertIn("page=2", api.call_args_list[1].args[0])
        self.assertEqual(api.call_args_list[2].args[0],
                         "repos/Llaminar/llaminar/issues/comments/91")

    def test_red_first_isa_does_not_hide_second_isa_benchmark_numbers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "e2e"
            bundle.mkdir()
            manifest = write_e2e_bundle(bundle)
            pair = image_pair()
            _, e2e_reports = suite.admit_e2e(bundle, pair)
            reports = {isa: benchmark_report(pair, manifest, e2e_reports[isa], isa)
                       for isa in suite.ISAS}
            row = reports["AVX512"]["cells"][0]
            baseline = {"schema": 1, "regression_threshold_pct": 10, "entries": {
                digest(row["identity"]): {"identity": row["identity"],
                                          "tokens_per_second": {"prefill": 2000.0,
                                                                "decode": 120.0}}}}
            (root / "benchmarks/production").mkdir(parents=True)
            write_json(root / "benchmarks/production/high_water.json", baseline)
            for isa, report in reports.items():
                proposed, comparisons = ratchet(baseline, report["cells"])
                report.update(baseline_digest=digest(baseline), proposed_high_water=proposed,
                              comparisons=comparisons,
                              passed=all(item["passed"] for item in comparisons),
                              hardware={}, workload={})
            calls = []

            def run_lane(_args, _driver, command, lane, _log):
                isa = lane.name.upper()
                calls.append(isa)
                write_json(Path(command[command.index("--report") + 1]), reports[isa])
                if isa == "AVX512":
                    raise subprocess.CalledProcessError(1, command)

            output = root / "output"
            output.mkdir()
            args = SimpleNamespace(e2e_bundle=bundle, model_ramdisk_root=root,
                                   run_url="https://example.test/run")
            with patch.object(suite, "ROOT", root), \
                 patch.object(suite, "build_driver", return_value="driver"), \
                 patch.object(suite, "run_in_driver", side_effect=run_lane), \
                 patch.object(chart, "render_chart", return_value="<svg/>"):
                result = suite.run_benchmarks(args, pair, output)
            self.assertEqual(calls, ["AVX512", "AVX2"])
            self.assertFalse(result["passed"])
            self.assertTrue((output / "results.json").exists())
            self.assertEqual(result["variants"]["AVX512"]["comparisons"],
                             reports["AVX512"]["comparisons"])

    def test_failed_second_isa_stops_all_benchmarks_before_first_launch(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_e2e_bundle(root)
            receipt = json.loads((root / "e2e-receipt.json").read_text())
            receipt["complete"] = False
            write_json(root / "e2e-receipt.json", receipt)
            with patch.object(suite, "run_in_driver") as run, \
                 patch.object(suite, "build_driver") as build:
                with self.assertRaises(ValueError):
                    suite.run_benchmarks(SimpleNamespace(e2e_bundle=root), image_pair(), root)
            run.assert_not_called()
            build.assert_not_called()

    def test_control_container_keeps_cache_owner_and_read_only_source(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            lane = root / "evidence"
            lane.mkdir()
            socket = root / "docker.sock"
            socket.touch()
            args = SimpleNamespace(output=root, models=root, model_ramdisk_root=root, e2e_bundle=None)
            mounts = []
            def bind(pairs):
                mounts.extend(pairs)
                return []
            with patch.dict(suite.os.environ, {"DOCKER_HOST": "unix://" + str(socket)}), \
                 patch.object(suite.docker_paths, "device_args", return_value=[]) as devices, \
                 patch.object(suite.docker_paths, "mounts", side_effect=bind), \
                 patch.object(suite.pipeline, "run") as run, \
                 patch.object(suite.subprocess, "run") as cleanup:
                suite.run_in_driver(args, "tools-image", ["python", "runner.py"], lane, "run.log")
            devices.assert_called_once_with("tools-image", "CPU+CUDA+ROCm",
                user=f"{cache.stat().st_uid}:{lane.stat().st_gid}")
            self.assertIn((suite.ROOT, str(suite.ROOT), True), mounts)
            self.assertIn((root, str(root), False), mounts)
            self.assertIn((lane, str(lane), False), mounts)
            command = run.call_args.args[0]
            self.assertEqual(command[-3:], ["tools-image", "python3", "runner.py"])
            self.assertIn("SYSLOG", command)
            self.assertIn("/dev/kmsg:/dev/kmsg:r", command)
            driver_name = command[command.index("--name") + 1]
            self.assertIn(f"LLAMINAR_E2E_KERNEL_READER_CONTAINER={driver_name}", command)
            self.assertEqual(lane.stat().st_mode & 0o007, 0o005)
            repair = next(call.args[0] for call in cleanup.call_args_list
                          if call.args[0][:2] == ["docker", "run"])
            self.assertEqual(repair[-3:], ["--recursive", f"{lane.stat().st_uid}:{lane.stat().st_gid}",
                                            str(lane.resolve())])

    def test_control_container_propagates_the_arc_shared_root_contract(self):
        """Nested HTTP launches retain the exact runner/daemon mount topology."""
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cache = root / "cache"
            cache.mkdir()
            unmounted = root / "not-mounted-into-driver"
            unmounted.mkdir()
            lane = root / "evidence"
            lane.mkdir()
            socket = root / "docker.sock"
            socket.touch()
            args = SimpleNamespace(output=root, models=root, model_ramdisk_root=root, e2e_bundle=None)
            with patch.dict(suite.os.environ, {
                "DOCKER_HOST": "unix://" + str(socket),
                suite.docker_paths.SHARED_DAEMON_ROOTS_ENV: os.pathsep.join((str(root), str(unmounted))),
            }, clear=False), \
                 patch.object(suite.docker_paths, "device_args", return_value=[]), \
                 patch.object(suite.docker_paths, "mounts", return_value=[]), \
                 patch.object(suite.pipeline, "run") as run, \
                 patch.object(suite.subprocess, "run"):
                suite.run_in_driver(args, "tools-image", ["python", "runner.py"], lane, "run.log")
            command = run.call_args.args[0]
            self.assertIn(
                f"{suite.docker_paths.SHARED_DAEMON_ROOTS_ENV}={root}",
                command,
            )
            self.assertNotIn(str(unmounted), command)
            image_index = command.index("tools-image")
            self.assertEqual(command[image_index - 2:image_index], [
                "--env", f"{suite.docker_paths.SHARED_DAEMON_ROOTS_ENV}={root}",
            ])

    def test_render_contains_both_isas_exact_identity_geometry_and_rates(self):
        image = chart.render_chart(benchmark_result())
        ET.fromstring(image)
        for expected in ("AVX512", "AVX2", "a" * 40, "8,192", "512", "256",
                         "1,500.2", "60.1", "1,501.2", "61.1", "1×ROCm"):
            self.assertIn(expected, image)

    def test_svg_metadata_is_escaped(self):
        result = benchmark_result()
        for variant in result["variants"].values():
            variant["cells"][0]["identity"]["configuration"]["model"] = 'unsafe<&>.gguf'
        image = chart.render_chart(result)
        ET.fromstring(image)
        self.assertIn('unsafe&lt;&amp;&gt;', image)

    def test_chart_scale_is_local_to_each_cell_and_phase(self):
        """A much faster unrelated cell must not shrink either ISA's bars."""
        result = benchmark_result()
        for isa, prefill, decode in (("AVX512", 100, 40), ("AVX2", 80, 20)):
            result["variants"][isa]["cells"][0]["tokens_per_second"] = {
                "prefill": prefill, "decode": decode}

        def widths(data):
            """Inspect the semantic SVG row, independently of pixel positions."""
            root = ET.fromstring(chart.render_chart(data))
            cell = root.find(".//{*}g[@class='benchmark-cell']")
            return {phase.attrib["data-phase"]: {
                bar.attrib["data-isa"]: float(bar.attrib["width"])
                for bar in phase.findall("{*}rect[@data-isa]")}
                for phase in cell.findall("{*}g[@data-phase]")}

        expected = {"prefill": {"AVX512": 205, "AVX2": 164},
                    "decode": {"AVX512": 205, "AVX2": 102.5}}
        self.assertEqual(widths(result), expected)
        for variant in result["variants"].values():
            unrelated = deepcopy(variant["cells"][0])
            unrelated["case"] = "ZZZ/Unrelated"
            unrelated["tokens_per_second"] = {"prefill": 1e6, "decode": 1e5}
            variant["cells"].append(unrelated)
        original = deepcopy(result)
        self.assertEqual(widths(result), expected)
        self.assertEqual(result, original, "rendering may not rewrite measured evidence")
        image = chart.render_chart(result)
        self.assertIn("80.0% of AVX512", image)
        self.assertIn("50.0% of AVX512", image)

    def test_chart_keeps_faster_avx2_visible_above_its_avx512_reference(self):
        """A genuine AVX2 win extends the local axis instead of being clipped."""
        result = benchmark_result()
        for isa, prefill in (("AVX512", 100), ("AVX2", 125)):
            result["variants"][isa]["cells"][0]["tokens_per_second"]["prefill"] = prefill
        root = ET.fromstring(chart.render_chart(result))
        phase = root.find(".//{*}g[@data-phase='prefill']")
        bars = {bar.attrib["data-isa"]: bar for bar in phase.findall("{*}rect[@data-isa]")}
        self.assertEqual(float(bars["AVX512"].attrib["width"]), 164)
        self.assertEqual(float(bars["AVX2"].attrib["width"]), 205)
        self.assertEqual(phase.attrib["data-reference"], "AVX512")
        reference = phase.find("{*}line[@class='reference']")
        self.assertEqual(float(reference.attrib["x1"]), float(bars["AVX512"].attrib["x"]) + 164)
        self.assertIn("125.0% of AVX512", bars["AVX2"].find("{*}title").text)

    def test_chart_groups_model_sizes_numerically_without_splitting_finetunes(self):
        """27B, both 35B families and 122B are distinct, ordered sections."""
        result = benchmark_result()
        models = ("Qwen3.5-122B-A10B-Q8-00001-of-00004.gguf",
                  "Ornith-1.5-35B-Q4_K_M.gguf", "Qwen3.8-27B-IQ4_XS.gguf",
                  "Qwen3.6-35B-A3B-IQ3_S.gguf", "unknown.gguf")
        for variant in result["variants"].values():
            template = variant["cells"][0]
            variant["cells"] = []
            for model in models:
                row = deepcopy(template)
                row["case"] = f"Suite/{model}"
                row["identity"]["configuration"]["model"] = model
                variant["cells"].append(row)
        root = ET.fromstring(chart.render_chart(result))
        sections = root.findall("{*}g[@class='model-section']")
        self.assertEqual([section.find("{*}title").text for section in sections],
                         ["27B models", "35B models", "122B models", "Other models"])
        self.assertEqual([section.findall("{*}text")[-1].text for section in sections],
                         ["1 cell", "2 cells", "1 cell", "1 cell"])
        rows = root.findall("{*}g[@class='benchmark-cell']")
        self.assertEqual([row.attrib["data-case"] for row in rows],
                         [f"Suite/{models[index]}" for index in (2, 1, 3, 0, 4)])
        self.assertEqual(chart.model_section({"model": "Qwen-0.5B-Q8.gguf"}),
                         (0.5, "0.5B models"))
        self.assertEqual(chart.model_section({"model": "Qwen-A3B-Q8.gguf"})[1], "Other models")

    def test_chart_rejects_partial_duplicate_nonfinite_or_mismatched_rows(self):
        for mutation in (lambda data: data["variants"].pop("AVX2"),
                         lambda data: data["variants"]["AVX2"]["cells"].clear(),
                         lambda data: data["variants"]["AVX2"]["cells"].append(deepcopy(data["variants"]["AVX2"]["cells"][0])),
                         lambda data: data["variants"]["AVX2"]["cells"][0]["tokens_per_second"].update(decode=float('nan')),
                         lambda data: data["variants"]["AVX2"]["cells"][0].update(prefill_tokens=1)):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                result = deepcopy(benchmark_result())
                mutation(result)
                chart.render_chart(result)

    def test_tier_labels_preserve_priority_and_rank_local_cpu_identity(self):
        config = {"e2e": {"server_args": [
            "--moe-routed-expert-domain", "capacity=localhost:0:cpu:0,localhost:1:cpu:0;scope=node_local",
            "--moe-routed-expert-domain", "continuation=rocm:0,rocm:1;scope=rank_local",
            "--moe-routed-expert-tier", "slow@capacity;priority=20",
            "--moe-routed-expert-tier", "fast@continuation;priority=0"]}}
        self.assertEqual(chart.topology_label(config), "2×ROCm → 2×CPU · ExpertOverlay")

    def test_automatic_labels_do_not_invent_stage_or_tier_order(self):
        config = {"e2e": {"planning": {"mode": "auto", "strategy": "pp",
                    "device_counts": {"rocm": 2, "cuda": 2}}, "server_args": ["--auto"]}}
        self.assertEqual(chart.topology_label(config), "2×CUDA + 2×ROCm · auto PP")
        config["e2e"]["planning"]["strategy"] = "expert-overlay"
        self.assertEqual(chart.topology_label(config), "2×CUDA + 2×ROCm · auto ExpertOverlay")
        config["e2e"]["planning"]["device_counts"]["rocm"] = 0
        with self.assertRaises(ValueError): chart.topology_label(config)

    def test_readme_publication_changes_only_its_owned_block(self):
        original = f"prefix\n{suite.README_BEGIN}\nold\n{suite.README_END}\nsuffix\n"
        updated = suite.readme_with_results(original, benchmark_result())
        self.assertTrue(updated.startswith("prefix\n"))
        self.assertTrue(updated.endswith(f"{suite.README_END}\nsuffix\n"))
        self.assertIn("benchmarks/production/published/benchmarks.svg", updated)
        self.assertIn("a" * 40, updated)
        self.assertEqual(suite.readme_with_results(updated, benchmark_result()), updated)

    def test_missing_or_duplicate_markers_fail_instead_of_overwriting_readme(self):
        for text in ("no markers", suite.README_BEGIN * 2 + suite.README_END,
                     suite.README_END + suite.README_BEGIN):
            with self.subTest(text=text), self.assertRaises(ValueError):
                suite.readme_with_results(text, benchmark_result())

    def test_workflows_are_manual_only_and_share_device_concurrency(self):
        # BaseLoader preserves the YAML `on` key instead of treating it as a
        # YAML-1.1 boolean. These are workflow policy tests, not source scans
        # standing in for runtime provenance/phase tests above.
        import yaml
        for name in ("production-e2e.yml", "production-benchmarks.yml"):
            workflow = yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
            self.assertEqual(set(workflow["on"]), {"workflow_dispatch"})
            self.assertEqual(workflow["concurrency"]["group"], "llaminar-develop-image-gate")
            self.assertEqual(workflow["concurrency"]["cancel-in-progress"], "false")

    def test_master_pr_uses_exact_develop_ref_and_requires_e2e_before_benchmarks(self):
        import yaml
        workflow = yaml.load((ROOT / ".github/workflows/master-pr-certification.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"pull_request"})
        self.assertEqual(workflow["on"]["pull_request"]["branches"], ["master"])
        jobs = workflow["jobs"]
        self.assertEqual(jobs["e2e"]["needs"], "source")
        self.assertEqual(jobs["benchmarks"]["needs"], "e2e")
        self.assertEqual(jobs["e2e"]["concurrency"]["group"], "llaminar-develop-image-gate")
        self.assertEqual(jobs["benchmarks"]["concurrency"]["group"], "llaminar-develop-image-gate")
        self.assertEqual(jobs["benchmarks"]["permissions"]["pull-requests"], "write")
        text = (ROOT / ".github/workflows/master-pr-certification.yml").read_text()
        self.assertIn("--expected-source-revision \"$HEAD_SHA\"", text)
        self.assertIn(":develop-$HEAD_SHA", text)
        self.assertIn("--e2e-bundle", text)
        self.assertIn("scripts/ci/post_pr_benchmark_rag.py", text)
        self.assertNotIn("--publish", text)

    def test_pr_wait_accepts_only_exact_successful_develop_push(self):
        revision = "a" * 40
        branch = {"commit": {"sha": revision}}
        response = {"workflow_runs": [
            {"id": 7, "head_sha": "b" * 40, "head_branch": "develop",
             "event": "push", "status": "completed", "conclusion": "success"},
            {"id": 8, "head_sha": revision, "head_branch": "develop",
             "event": "push", "status": "completed", "conclusion": "success"},
        ]}
        with patch.object(image_wait, "github_json", side_effect=[branch, response]) as github:
            self.assertEqual(image_wait.wait_for_image_gate("Llaminar/llaminar", revision, 1), 8)
        self.assertEqual(github.call_count, 2)
        with patch.object(image_wait, "github_json", side_effect=[branch, {
                "workflow_runs": [{**response["workflow_runs"][1], "conclusion": "failure"}]}]):
            with self.assertRaisesRegex(ValueError, "without publishing both ISAs"):
                image_wait.wait_for_image_gate("Llaminar/llaminar", revision, 1)
        with patch.object(image_wait, "github_json", return_value={"commit": {"sha": "b" * 40}}):
            with self.assertRaisesRegex(ValueError, "no longer the current develop"):
                image_wait.wait_for_image_gate("Llaminar/llaminar", revision, 1)

    def test_master_release_dates_and_ref_tags_are_unambiguous(self):
        day = "2026-09-23"
        master = "d" * 40
        self.assertEqual(master_release.release_tag(day, [], master), (day + ".1", None))
        earlier = {"tag_name": day + ".1", "target_commitish": "e" * 40,
                   "draft": False, "published_at": "2026-09-23T12:00:00Z"}
        self.assertEqual(master_release.release_tag(day, [earlier], master),
                         (day + ".2", day + ".1"))
        draft = {"tag_name": day + ".2", "target_commitish": master,
                 "draft": True, "published_at": None}
        self.assertEqual(master_release.release_tag(day, [earlier, draft], master),
                         (day + ".2", day + ".1"))
        with self.assertRaisesRegex(ValueError, "already has a published release"):
            master_release.release_tag(day, [earlier, {**draft, "draft": False}], master)
        self.assertEqual(master_release.master_image_tags("Llaminar/llaminar", master,
                         day + ".2", "AVX512"), (
                         "ghcr.io/llaminar/llaminar:2026-09-23.2",
                         "ghcr.io/llaminar/llaminar:master-" + master,
                         "ghcr.io/llaminar/llaminar:master"))
        self.assertEqual(master_release.master_image_tags("Llaminar/llaminar", master,
                         day + ".2", "AVX2"), (
                         "ghcr.io/llaminar/llaminar:2026-09-23.2-avx2",
                         "ghcr.io/llaminar/llaminar:master-avx2-" + master,
                         "ghcr.io/llaminar/llaminar:master-avx2"))

    def test_master_release_finds_squash_pr_when_commit_index_is_empty(self):
        """A squash merge may be visible on the PR before GitHub's commit index."""
        master = "d" * 40
        pr = {"number": 9, "merged_at": "2026-09-24T08:03:08Z",
              "merge_commit_sha": master,
              "head": {"ref": "develop", "repo": {"full_name": "Llaminar/llaminar"}},
              "base": {"ref": "master"}}
        unrelated = {**pr, "merge_commit_sha": "e" * 40}
        with patch.object(master_release, "github_json",
                          side_effect=[[], [unrelated, pr]]) as github:
            self.assertEqual(master_release.merged_develop_pr("Llaminar/llaminar", master), pr)
        self.assertEqual(github.call_count, 2)
        self.assertEqual(github.call_args_list[1].args[0],
                         "repos/Llaminar/llaminar/pulls")
        with patch.object(master_release, "github_json",
                          side_effect=[[], [unrelated]]):
            with self.assertRaisesRegex(ValueError, "not one merged"):
                master_release.merged_develop_pr("Llaminar/llaminar", master)

    def test_master_release_never_rewrites_conflicting_immutable_image(self):
        image = {"id": "sha256:" + "a" * 64,
                 "registry_ref": "ghcr.io/llaminar/llaminar@sha256:" + "b" * 64}
        with patch.object(master_release, "remote_image_id", return_value="sha256:" + "c" * 64), \
             patch.object(master_release.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "immutable release image"):
                master_release.promote_tag("ghcr.io/llaminar/llaminar:2026-09-23.1",
                                           image, immutable=True)
            run.assert_not_called()
        with patch.object(master_release, "remote_image_id",
                          side_effect=[None, image["id"]]), \
             patch.object(master_release.subprocess, "run") as run:
            master_release.promote_tag("ghcr.io/llaminar/llaminar:2026-09-23.1",
                                       image, immutable=True)
            self.assertEqual(run.call_args.args[0][:4],
                             ["docker", "buildx", "imagetools", "create"])

    def test_master_aliases_move_only_after_proof_assets_are_uploaded(self):
        pair = image_pair()
        events = []

        def record_run(command, **_kwargs):
            events.append(("gh", command[:3]))

        def record_tag(alias, _image, *, immutable):
            events.append(("tag", immutable, alias))

        with tempfile.TemporaryDirectory() as temporary, \
             patch.object(master_release.subprocess, "run", side_effect=record_run), \
             patch.object(master_release, "promote_tag", side_effect=record_tag):
            master_release.publish("Llaminar/llaminar", "d" * 40, "2026-09-23.1",
                                   Path(temporary) / "notes.md", [Path(temporary) / "e2e.json"],
                                   {"pair": pair}, draft_exists=False)
        upload = next(index for index, event in enumerate(events)
                      if event[0] == "gh" and event[1][:3] == ["gh", "release", "upload"])
        self.assertTrue(all(event[1] for event in events[:upload] if event[0] == "tag"))
        self.assertTrue(all(not event[1] for event in events[upload + 1:] if event[0] == "tag"))
        self.assertEqual(events[-1], ("gh", ["gh", "release", "edit"]))

    def test_release_revalidates_every_attached_e2e_and_benchmark_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            e2e_root, benchmark_root = root / "e2e", root / "benchmarks"
            e2e_root.mkdir()
            benchmark_root.mkdir()
            pair = image_pair()
            pair["workflow_revision"] = pair["source"]["revision"]
            manifest = write_e2e_bundle(e2e_root, pair)
            write_json(benchmark_root / "images.json", pair)
            write_json(benchmark_root / "manifest.json", manifest)
            result = {"source": pair["source"], "images": pair["images"],
                      "repository": pair["repository"], "branch": "develop",
                      "passed": True,
                      "regression_threshold_pct": 10,
                      "scope": "full-http-e2e-and-benchmarks",
                      "full_image_certification": False, "variants": {}}
            for isa in suite.ISAS:
                e2e = json.loads((e2e_root / isa.lower() / "e2e.json").read_text())
                report = benchmark_report(pair, manifest, e2e, isa)
                write_json(benchmark_root / isa.lower() / "benchmarks.json", report)
                result["variants"][isa] = {"report_digest": digest(report),
                    "e2e_report_digest": digest(e2e),
                    "comparisons": report["comparisons"], "cells": report["cells"]}
            write_json(benchmark_root / "results.json", result)
            admitted = master_release.validate_proof(
                e2e_root, benchmark_root, pair["repository"],
                pair["source"]["revision"], pair["source"]["tree"])
            self.assertEqual(admitted["pair"], pair)
            initial = master_release.release_notes("2026-09-23.1", None, [], admitted,
                pair["repository"], "https://github.com/example/actions/runs/1", "d" * 40)
            self.assertIn("Initial dated release", initial)
            self.assertIn("e2e-avx512.json", initial)
            self.assertIn("benchmark-results.json", initial)
            self.assertNotIn("- 123abc", initial)
            later = master_release.release_notes("2026-09-23.2", "2026-09-23.1",
                ["123abc fix: restore prefix"], admitted, pair["repository"],
                "https://github.com/example/actions/runs/2", "d" * 40)
            self.assertIn("123abc fix: restore prefix", later)
            with self.assertRaisesRegex(ValueError, "master tree"):
                master_release.validate_proof(e2e_root, benchmark_root, pair["repository"],
                                              pair["source"]["revision"], "f" * 40)
            result["variants"]["AVX2"]["report_digest"] = "wrong"
            write_json(benchmark_root / "results.json", result)
            with self.assertRaisesRegex(ValueError, "compact benchmark numbers"):
                master_release.validate_proof(e2e_root, benchmark_root, pair["repository"],
                                              pair["source"]["revision"], pair["source"]["tree"])

    def test_master_ruleset_requires_source_and_both_phase_checks(self):
        current = {"name": "master", "target": "branch", "enforcement": "disabled",
                   "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
                   "bypass_actors": [], "rules": [{"type": "pull_request", "parameters": {
                       "allowed_merge_methods": ["squash"]}},
                       {"type": "required_status_checks", "parameters": {
                           "required_status_checks": [{"context": "CI complete"}]}}]}
        proposed = master_ruleset.proposed_ruleset(current)
        self.assertEqual(proposed["enforcement"], "active")
        self.assertEqual(proposed["rules"][0], current["rules"][0])
        checks = proposed["rules"][1]["parameters"]["required_status_checks"]
        self.assertEqual([check["context"] for check in checks], list(master_ruleset.REQUIRED_CHECKS))
        self.assertTrue(all(check["integration_id"] == 15368 for check in checks))

    def test_develop_guard_requires_pr_and_only_dedicated_release_key_bypasses(self):
        current = {"name": "develop", "target": "branch", "enforcement": "disabled",
                   "conditions": {"ref_name": {"include": ["refs/heads/develop"], "exclude": []}},
                   "bypass_actors": [], "rules": [{"type": "deletion"},
                                                  {"type": "non_fast_forward"},
                                                  {"type": "required_linear_history"}]}
        proposed = master_ruleset.proposed_develop_ruleset(current)
        self.assertEqual(proposed["enforcement"], "active")
        self.assertEqual({rule["type"] for rule in proposed["rules"]},
                         {"deletion", "non_fast_forward", "pull_request",
                          "required_status_checks"})
        self.assertEqual(proposed["bypass_actors"], [
            {"actor_id": None, "actor_type": "DeployKey", "bypass_mode": "always"}])
        checks = next(rule for rule in proposed["rules"]
                      if rule["type"] == "required_status_checks")["parameters"]
        self.assertTrue(checks["strict_required_status_checks_policy"])
        self.assertEqual(checks["required_status_checks"], [
            {"context": "Unit + ProductionTestPreflight (AVX512)",
             "integration_id": 15368}])
        self.assertEqual(master_ruleset.proposed_develop_ruleset(proposed), proposed)
        current["rules"].append({"type": "creation"})
        with self.assertRaisesRegex(ValueError, "unknown or missing guard"):
            master_ruleset.proposed_develop_ruleset(current)
        proposed["bypass_actors"] = [
            {"actor_id": 15368, "actor_type": "Integration", "bypass_mode": "always"}]
        with self.assertRaisesRegex(ValueError, "unrelated bypass actor"):
            master_ruleset.proposed_develop_ruleset(proposed)

    def test_release_bypass_rejects_any_other_writable_deploy_key(self):
        with patch.object(master_ruleset, "github_json", return_value=[
                {"title": master_ruleset.RELEASE_DEPLOY_KEY_TITLE,
                 "read_only": False, "enabled": True}]):
            master_ruleset.require_dedicated_release_key("Llaminar/llaminar")
        with patch.object(master_ruleset, "github_json", return_value=[
                {"title": master_ruleset.RELEASE_DEPLOY_KEY_TITLE,
                 "read_only": False, "enabled": True},
                {"title": "other-writer", "read_only": False, "enabled": True}]):
            with self.assertRaisesRegex(ValueError, "exactly one writable deploy key"):
                master_ruleset.require_dedicated_release_key("Llaminar/llaminar")

    def test_high_water_cli_requires_exact_release_ssh_push_target(self):
        with self.assertRaises(SystemExit) as error:
            high_water.main(["--repository", "Llaminar/llaminar",
                             "--master-sha", "a" * 40, "--output", "/tmp/unused",
                             "--push-url", "origin"])
        self.assertEqual(error.exception.code, 2)

    def test_obsolete_policy_installers_cannot_restore_the_old_master_check(self):
        for name in ("apply-rulesets.sh", "apply-branch-protection.sh"):
            self.assertFalse((ROOT / ".github/scripts" / name).exists())

    def test_master_release_workflow_promotes_only_after_pr_artifact_validation(self):
        import yaml
        workflow = yaml.load((ROOT / ".github/workflows/release.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(set(workflow["on"]), {"pull_request_target", "workflow_dispatch"})
        self.assertEqual(workflow["on"]["pull_request_target"]["branches"], ["master"])
        self.assertEqual(workflow["on"]["pull_request_target"]["types"], ["closed"])
        self.assertEqual(set(workflow["jobs"]), {"promote", "high_water", "documentation"})
        self.assertEqual(workflow["jobs"]["documentation"]["needs"], "promote")
        self.assertEqual(workflow["jobs"]["documentation"]["uses"], "./.github/workflows/docs.yml")
        self.assertEqual(workflow["jobs"]["promote"]["concurrency"]["group"],
                         "llaminar-develop-image-gate")
        self.assertEqual(workflow["jobs"]["high_water"]["needs"], "promote")
        text = (ROOT / ".github/workflows/release.yml").read_text()
        self.assertIn("github.event.pull_request.merged == true", text)
        self.assertIn("github.event.pull_request.merge_commit_sha", text)
        self.assertIn("github.ref == 'refs/heads/master'", text)
        self.assertIn("scripts/ci/publish_master_release.py", text)
        self.assertIn("scripts/ci/publish_pr_high_water.py", text)
        self.assertIn("secrets.RELEASE_DEVELOP_DEPLOY_KEY", text)
        self.assertIn("gh api meta --jq '.ssh_keys[]'", text)
        self.assertIn('--push-url "git@github.com:${REPOSITORY}.git"', text)
        self.assertNotIn("release-please", text)

    def test_pr_workflow_run_title_pins_number_and_exact_source(self):
        import yaml
        workflow = yaml.load((ROOT / ".github/workflows/master-pr-certification.yml").read_text(),
                             Loader=yaml.BaseLoader)
        self.assertEqual(workflow["run-name"],
                         "master PR #${{ github.event.pull_request.number }} — "
                         "${{ github.event.pull_request.head.sha }}")

    def test_post_merge_high_water_combines_both_isas_and_uses_skip_ci(self):
        pair = image_pair()
        baseline = {"schema": 1, "regression_threshold_pct": 10, "entries": {}}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            chart = root / "benchmarks.svg"
            chart.write_text("<svg/>")
            result = benchmark_result()
            rows = {isa: [{"case": "Suite/Tiny.ProductionParity/ROCm",
                           "identity": {"case": "cell", "cpu_isa": isa},
                           "tokens_per_second": {"prefill": 100.0, "decode": 50.0}}]
                    for isa in suite.ISAS}
            evidence = {"pair": pair, "result": result, "benchmark_directory": root,
                        "benchmarks": {isa: {"baseline_digest": digest(baseline),
                                            "cells": rows[isa]} for isa in suite.ISAS}}
            with patch.object(high_water, "git_file", return_value=(ROOT / "README.md").read_bytes()):
                payloads = high_water.proposed_payloads(evidence, baseline)
            self.assertEqual(len(json.loads(payloads[high_water.HIGH_WATER])["entries"]), 2)
            self.assertEqual(payloads[str(suite.REPORT_DIRECTORY / "benchmarks.svg")], b"<svg/>")
            evidence["benchmarks"]["AVX2"]["baseline_digest"] = "stale"
            with self.assertRaisesRegex(ValueError, "different source high-water"):
                high_water.proposed_payloads(evidence, baseline)

    def test_high_water_commit_joins_identical_master_tree_without_other_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            (root / "high_water.json").write_text("old\n")
            subprocess.run(["git", "add", "high_water.json"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                            "commit", "-qm", "base"], cwd=root, check=True)
            parent = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                             text=True).strip()
            subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                            "commit", "-qm", "squashed master", "--allow-empty"], cwd=root,
                           check=True)
            master = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                             text=True).strip()
            with patch.object(high_water, "ROOT", root):
                committed = high_water.commit_payloads(parent, master,
                                                       {"high_water.json": b"new\n"}, 7)
                self.assertEqual(high_water.git("rev-parse", f"{committed}^1"), parent)
                self.assertEqual(high_water.git("rev-parse", f"{committed}^2"), master)
                self.assertIn("[skip ci]", high_water.git("log", "-1", "--format=%s", committed))
                self.assertEqual(high_water.git_file(committed, "high_water.json"), b"new\n")
                unchanged = high_water.commit_payloads(parent, master,
                                                       {"high_water.json": b"old\n"}, 7)
                self.assertNotEqual(unchanged, parent)
                self.assertEqual(high_water.git("rev-parse", f"{unchanged}^2"), master)
                (root / "high_water.json").write_text("unrelated master tree\n")
                subprocess.run(["git", "add", "high_water.json"], cwd=root, check=True)
                subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                                "commit", "-qm", "different master tree"], cwd=root,
                               check=True)
                different = high_water.git("rev-parse", "HEAD")
                with self.assertRaisesRegex(ValueError, "trees differ"):
                    high_water.commit_payloads(parent, different,
                                               {"high_water.json": b"new\n"}, 7)

    def test_post_merge_publication_fast_forwards_develop_and_retries_idempotently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote, checkout = root / "remote.git", root / "checkout"
            subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
            subprocess.run(["git", "clone", "-q", str(remote), str(checkout)], check=True)

            def command(*args):
                return subprocess.check_output(["git", *args], cwd=checkout,
                                               text=True).strip()

            command("switch", "-q", "-c", "develop")
            (checkout / "benchmarks/production").mkdir(parents=True)
            baseline = {"schema": 1, "regression_threshold_pct": 10, "entries": {}}
            write_json(checkout / high_water.HIGH_WATER, baseline)
            (checkout / "README.md").write_text(
                "intro\n" + suite.README_BEGIN + "\nold\n" + suite.README_END + "\n")
            command("add", ".")
            command("-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "develop source")
            source = command("rev-parse", "HEAD")
            tree = command("rev-parse", "HEAD^{tree}")
            command("push", "-q", "origin", "develop")
            command("switch", "-q", "-c", "master")
            command("-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "master squash", "--allow-empty")
            master = command("rev-parse", "HEAD")
            self.assertEqual(command("rev-parse", "HEAD^{tree}"), tree)
            source_identity = {"revision": source, "tree": tree, "dirty": False}
            result = benchmark_result()
            result["source"] = source_identity
            rows = {isa: [{"case": "Suite/Tiny.ProductionParity/ROCm",
                           "identity": {"case": "cell", "cpu_isa": isa},
                           "tokens_per_second": {"prefill": 100.0, "decode": 50.0}}]
                    for isa in suite.ISAS}
            evidence = {"pair": {"source": source_identity}, "result": result,
                        "benchmarks": {isa: {"baseline_digest": digest(baseline),
                                            "cells": rows[isa]} for isa in suite.ISAS}}
            benchmark_directory = root / "benchmarks"
            benchmark_directory.mkdir()
            (benchmark_directory / "benchmarks.svg").write_text("<svg/>")
            with patch.object(high_water, "ROOT", checkout), \
                 patch.object(master_release, "merged_develop_pr",
                              return_value={"head": {"sha": source}, "number": 7}), \
                 patch.object(master_release, "certified_pr_run", return_value={"id": 99}), \
                 patch.object(master_release, "download_proof",
                              return_value=(root / "e2e", benchmark_directory)), \
                 patch.object(master_release, "validate_proof", return_value=evidence):
                first = high_water.publish("Llaminar/llaminar", master, root / "proof",
                                           str(remote))
                second = high_water.publish("Llaminar/llaminar", master, root / "proof-retry",
                                            str(remote))
            self.assertFalse(first["reused"])
            self.assertTrue(second["reused"])
            self.assertEqual(first["develop_sha"], second["develop_sha"])
            self.assertEqual(command("rev-parse", "FETCH_HEAD"), first["develop_sha"])
            self.assertIn("[skip ci]", command("log", "-1", "--format=%s", first["develop_sha"]))
            self.assertEqual(command("rev-parse", f"{first['develop_sha']}^1"), source)
            self.assertEqual(command("rev-parse", f"{first['develop_sha']}^2"), master)
            self.assertEqual(command("merge-base", master, first["develop_sha"]), master)
            marks = json.loads(command("show", f"{first['develop_sha']}:{high_water.HIGH_WATER}"))
            self.assertEqual(len(marks["entries"]), 2)

    def test_workflows_keep_generated_evidence_out_of_the_checkout(self):
        """A cache-owner driver may not poison the persistent ARC worktree."""
        for name, prefix in (("production-e2e.yml", "llaminar-published-e2e"),
                             ("production-benchmarks.yml", "llaminar-published-benchmarks")):
            with self.subTest(workflow=name):
                text = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertIn("Retire legacy checkout evidence", text)
                self.assertIn(f'$RUNNER_TEMP/{prefix}-$GITHUB_RUN_ID', text)
                self.assertNotIn(f"--output parity-results/{prefix.removeprefix('llaminar-')}", text)
                self.assertLess(text.index("docker/login-action@v4"),
                                text.index("Retire legacy checkout evidence"))
                self.assertLess(text.index("Retire legacy checkout evidence"),
                                text.index("actions/checkout@v6"))
                self.assertIn("exec python3 scripts/ci/run_published_image_suite.py", text)
        pr = (ROOT / ".github/workflows/master-pr-certification.yml").read_text()
        self.assertEqual(pr.count("exec python3 scripts/ci/run_published_image_suite.py"), 2)

    def test_metadata_companion_does_not_build_a_replacement_runtime(self):
        from types import SimpleNamespace
        calls = []
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            pair = image_pair()
            fake = {"id": "test-image"}
            with patch.object(suite.pipeline, "snapshot"), \
                 patch.object(suite.pipeline, "persistent_build_cache_arguments", return_value=[]), \
                 patch.object(suite.pipeline, "run", side_effect=lambda cmd, _: calls.append(cmd)), \
                 patch.object(suite.pipeline, "image_identity", return_value=fake), \
                 patch.object(suite.pipeline, "require_image") as admitted, \
                 patch.object(suite.pipeline, "require_image_pair") as pair_admission:
                result = suite.pipeline.build(SimpleNamespace(cpu_isa="AVX2"), pair["source"], directory,
                                              roles=(suite.pipeline.ImageRole.TEST_RUNNER,))
            self.assertEqual(set(result), {"test-runner"})
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][calls[0].index("--target") + 1], "test-runner")
            admitted.assert_called_once()
            pair_admission.assert_not_called()


class PublishedImageDependencyTests(unittest.TestCase):
    """Stock ARC runners must acquire tools before downloading E2E evidence."""

    def test_benchmark_workflow_installs_and_checks_github_cli(self):
        """Execute the real setup block with inert tools, including failures."""
        import yaml
        workflow = yaml.load(
            (ROOT / ".github/workflows/production-benchmarks.yml").read_text(),
            Loader=yaml.BaseLoader)
        steps = workflow["jobs"]["benchmarks"]["steps"]
        setup = next(step for step in steps if step.get("name") == "Install GitHub CLI")
        driver = next(step for step in steps
                      if "run_published_image_suite.py" in step.get("run", ""))
        self.assertLess(steps.index(setup), steps.index(driver))
        self.assertEqual(setup["shell"], "bash")
        expected = ["sudo apt-get update", "sudo apt-get install --yes --no-install-recommends gh"]
        for install_exit, version_exit in ((0, 0), (23, 0), (0, 24)):
            with self.subTest(install_exit=install_exit, version_exit=version_exit), \
                 tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                log = root / "calls"
                # The setup shell is real; only network/package operations are
                # replaced. No test may mutate the machine's package database.
                sudo = root / "sudo"
                sudo.write_text('#!/bin/sh\nprintf "sudo %s\\n" "$*" >> "$INSTALL_LOG"\n'
                                'if [ "$2" = "install" ]; then exit "$INSTALL_EXIT"; fi\n')
                gh = root / "gh"
                gh.write_text('#!/bin/sh\nprintf "gh %s\\n" "$*" >> "$INSTALL_LOG"\n'
                              'exit "$VERSION_EXIT"\n')
                sudo.chmod(0o755)
                gh.chmod(0o755)
                env = {**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"],
                       "INSTALL_LOG": str(log), "INSTALL_EXIT": str(install_exit),
                       "VERSION_EXIT": str(version_exit)}
                result = subprocess.run(["bash", "-c", setup["run"]], env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, install_exit or version_exit, result.stderr)
                self.assertEqual(log.read_text().splitlines(),
                                 expected + ([] if install_exit else ["gh --version"]))

    def test_missing_github_cli_stops_remote_benchmarks_before_image_pull(self):
        """Missing tooling must fail before Docker, cache ownership or GPU work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(suite="benchmarks", e2e_bundle=None, publish=False,
                                   output=root / "run", models=root, model_ramdisk_root=root)
            with patch.object(suite, "parse_arguments", return_value=args), \
                 patch("shutil.which", return_value=None), \
                 patch.object(suite.pipeline, "device_lease", return_value=nullcontext()) as lease, \
                 patch.object(suite, "pull_pair") as pull, \
                 patch.object(suite.docker_paths, "publish_model_cache") as cache, \
                 patch.object(suite, "run_benchmarks") as run:
                with self.assertRaisesRegex(ValueError, r"GitHub CLI \(gh\).*required"):
                    suite.main([])
            lease.assert_not_called()
            pull.assert_not_called()
            cache.assert_not_called()
            run.assert_not_called()
            self.assertFalse(args.output.exists())

    def test_explicit_local_e2e_bundle_does_not_require_github_cli(self):
        """Offline same-image admission remains a first-class local workflow."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = SimpleNamespace(suite="benchmarks", e2e_bundle=root, publish=False,
                                   output=root / "run", models=root, model_ramdisk_root=root)
            with patch.object(suite, "parse_arguments", return_value=args), \
                 patch("shutil.which", return_value=None), \
                 patch.object(suite.pipeline, "device_lease", return_value=nullcontext()), \
                 patch.object(suite, "pull_pair", return_value=image_pair()), \
                 patch.object(suite.docker_paths, "publish_model_cache"), \
                 patch.object(suite, "run_benchmarks") as run:
                self.assertEqual(suite.main([]), 0)
            run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
