#!/usr/bin/env python3
"""Device-free contracts for manual, immutable-image E2E/benchmark publication.

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
import run_model_parity_e2e as e2e
import published_benchmark_chart as chart
from production_artifacts import digest, write_json


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
    return {"complete": True, "passed": True, "diagnostic": False,
            "source_revision": pair["source"]["revision"], "image": pair["images"][isa]["id"],
            "manifest_digest": digest(manifest), "e2e_report_digest": digest(e2e), "cells": rows}


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
                """Successful shell fixtures must retain the new tool evidence too."""
                code = next(outcomes)
                if code == 0:
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
