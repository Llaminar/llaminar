#!/usr/bin/env python3
"""Device-free contracts for manual, immutable-image E2E/benchmark publication.

Fixtures model metadata only. Real HTTP/math evidence remains the canonical
runners' responsibility; these tests exercise pair admission, full-suite joins,
publication boundaries and the exact-data renderer without GPUs or weights.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "scripts/ci"))
import run_published_image_suite as suite
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


if __name__ == "__main__":
    unittest.main()
