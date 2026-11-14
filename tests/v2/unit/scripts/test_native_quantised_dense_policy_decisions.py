#!/usr/bin/env python3
"""Replay selective WMMA installation through real authenticated corpus owners.

Synthetic original trainer observations cover alias regressions, ties, literal
retention and profiler projection without opening a device. Negative controls
alter complete inputs, partial decisions or obligations at their public boundary;
all-retain results still consume every original correctness and timing record.
"""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import test_native_quantised_production_dense_prefill_sweep as fixtures
import test_native_quantised_dense_policy_binding as binding_fixtures
from native_quantised_dispatch import dense_prefill_policy_binding as binding
from native_quantised_dispatch import dense_prefill_policy_decisions as owner
from native_quantised_dispatch import dense_production_overlay as overlay
from native_quantised_dispatch import native_prefill_profiler as profiler
from native_quantised_dispatch import production_dense_prefill_sweep as sweep

WMMA = sweep.ROCM_WMMA_CANDIDATE_IDS


def rewrite_rows(path: Path, transform) -> None:
    """Preserve the native CSV schema while modifying a deliberate replay input."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle); columns = reader.fieldnames; rows = list(reader)
    transform(rows)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns); writer.writeheader(); writer.writerows(rows)


def recorded_cell(root: Path, label: str, medians: dict[str, float], *, auto_launch: str | None = None):
    """Use independent clocks and optional actual-Auto physical identity."""
    cell = fixtures.ProductionDensePrefillSweepTest()._scoped_wmma_cell(label)
    _, paths = fixtures._write_cell(root, "rocm", cell=cell)
    def clocks(rows):
        for row in rows:
            value = float(medians.get(row["variant"], 40.)) + (int(row["sample_index"])-1)*.125
            row.update(latency_us=f"{value:.9f}", latency_us_hex=value.hex())
    rewrite_rows(paths.timing, clocks)
    def aggregate(rows):
        winner = min(rows, key=lambda row: medians.get(row["variant"], 40.))["variant"]
        for row in rows:
            value = medians.get(row["variant"], 40.)
            row.update(min_us=f"{value-.125:.3f}", mean_us=f"{value:.3f}",
                       stddev_us=f"{math.sqrt(2*.125**2/3):.3f}", is_best=str(int(row["variant"] == winner)))
        if auto_launch:
            selected = next(row for row in rows if row["variant"] == auto_launch)
            auto = next(row for row in rows if row["variant"] == "Auto")
            auto.update({name: value for name, value in selected.items() if name.startswith("observed_")})
    rewrite_rows(paths.aggregate, aggregate)
    return cell


def commit(root: Path, cells, *, base: str | None = None):
    """Admit a recorded cold query and promote every cell through real validation."""
    base_path = root / "base.inc"; base_path.write_text(base or binding_fixtures.base_fixture())
    response = subprocess.CompletedProcess([], 0, binding_fixtures.recorded_log(base_path.read_text()))
    with mock.patch.object(binding, "trainer_provenance", return_value=fixtures._test_provenance("rocm")), \
            mock.patch.object(binding.subprocess, "run", return_value=response):
        frozen = binding.freeze_installed_policy(base_path, root / "trainer")
    return fixtures._commit_test_cells(root, "rocm", cells, installed_policy=frozen)


def collect(root: Path):
    """Read the original fixture's exact timing cardinality."""
    return owner.collect_policy_decisions(root, warmups=2, samples=3)


class DensePolicyDecisionsReplay(unittest.TestCase):
    """Prove complete decisions, retained lookup semantics and profiler obligations."""

    def test_all_retain_preserves_exact_base_without_profiler_or_empty_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = recorded_cell(root, "Q4_0", {"Auto": 10., WMMA[0]: 10.})
            plan = commit(root, (cell,))
            decisions = collect(root)
            self.assertFalse(decisions.installations)
            with self.assertRaisesRegex(ValueError, "all-retain"):
                profiler.build_manifest(root, warmups=2, samples=3)
            with mock.patch.object(profiler, "require_profile", side_effect=AssertionError("retention cannot profile")):
                owner.write_policy(root, root/"out.inc", base_include=root/"base.inc", receipt_path=root/"receipt.json",
                                   warmups=2, samples=3)
            self.assertEqual((root/"out.inc").read_bytes(), (root/"base.inc").read_bytes())
            receipt = json.loads((root/"receipt.json").read_text())
            self.assertEqual((receipt["outcome"], receipt["installed_keys"], receipt["retained_planned_keys"]),
                             ("no-change", 0, 1))
            self.assertEqual(receipt["decision_digest"], decisions.digest)
            self.assertEqual(receipt["decision_inventory"]["plan_digest"], plan["plan_digest"])

    def test_identical_launch_timing_noise_is_not_an_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cell = recorded_cell(root, "Q4_0", {"Auto": 10., WMMA[0]: 1.}, auto_launch=WMMA[0])
            commit(root, (cell,))
            self.assertFalse(collect(root).installations)

    def test_each_changed_alias_must_improve_before_ranking_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Candidate0 has better pooled clocks, but regresses Q4_K. Candidate1
            # improves both aliases and therefore owns the actual installation.
            cells = (recorded_cell(root, "Q4_1", {"Auto": 10., WMMA[0]: 1., WMMA[1]: 9.}),
                     recorded_cell(root, "Q4_K", {"Auto": 10., WMMA[0]: 11., WMMA[1]: 9.}))
            commit(root, cells)
            decision = collect(root).installations[0]
            self.assertEqual(decision.candidate_id, WMMA[1])
            self.assertEqual(decision.source_formats, ("Q4_1", "Q4_K"))

    def test_one_unchanged_alias_and_one_improvement_still_profiles_every_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = (recorded_cell(root, "Q4_1", {"Auto": 10., WMMA[0]: 11.}, auto_launch=WMMA[0]),
                     recorded_cell(root, "Q4_K", {"Auto": 10., WMMA[0]: 9.}))
            commit(root, cells)
            decisions = collect(root)
            self.assertEqual(len(decisions.installations), 1)
            manifest = profiler.build_manifest(root, warmups=2, samples=3)
            self.assertEqual({(r.source_format, r.config_json["purpose"]) for r in manifest.requests},
                             {(alias, purpose) for alias in ("Q4_1", "Q4_K") for purpose in ("winner", "baseline")})
            for request in manifest.requests:
                self.assertEqual(request.config_json["decision_digest"], decisions.digest)
                if request.config_json["purpose"] == "baseline":
                    self.assertEqual(request.config_json["requested_candidate"], "Auto")

    def test_one_changed_alias_tie_or_regression_retains(self):
        for slow in (10., 11.):
            with self.subTest(slow=slow), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                cells = (recorded_cell(root, "Q4_1", {"Auto": 10., WMMA[0]: 1.}),
                         recorded_cell(root, "Q4_K", {"Auto": 10., WMMA[0]: slow}))
                commit(root, cells)
                self.assertFalse(collect(root).installations)

    def test_conflicting_auto_launches_can_be_retained_without_concretizing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = (recorded_cell(root, "Q4_1", {"Auto": 1.}, auto_launch=WMMA[0]),
                     recorded_cell(root, "Q4_K", {"Auto": 1.}, auto_launch=WMMA[1]))
            commit(root, cells)
            self.assertFalse(collect(root).installations)
            with self.assertRaisesRegex(ValueError, "conflicting physical"):
                overlay.collect_dense_overlay_entries(root, "rocm", warmup_runs=2, bench_runs=3)

    def test_mixed_composition_keeps_exact_rows_absence_and_legacy_envelope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = (recorded_cell(root, "Q4_0", {"Auto": 10., WMMA[0]: 1.}),
                     recorded_cell(root, "IQ4_NL", {"Auto": 1.}),
                     recorded_cell(root, "IQ4_XS", {"Auto": 1.}))
            # Retained IQ4_NL has an exact row; IQ4_XS intentionally has none.
            c = cells[1]
            retained = overlay.DenseOverlayEntry("rocm", overlay.DenseOverlayKey(
                c.source_format.runtime_codebook("rocm"), c.m, c.shape.n, c.shape.k,
                architecture=overlay.GPUArchitecture.parse("rocm", "gfx1100")),
                (c.source_format.label,), (c.shape.name,), "Auto", (64,32,1,2,0,0), 0.,0.,1.)
            base = overlay._strip_generated_doxygen(overlay.render_dense_overlay((retained,)))
            commit(root, cells, base=base)
            decisions = collect(root)
            encoded, receipt = owner.compose_policy(base, decisions)
            before_prefix, before, before_suffix = overlay._split_generated_overlay(base, "rocm", preserve_envelope=True)
            prefix, after, suffix = overlay._split_generated_overlay(encoded, "rocm", preserve_envelope=True)
            self.assertEqual((prefix, suffix), (before_prefix, before_suffix))
            self.assertEqual(after[retained.key], before[retained.key])
            self.assertEqual(len(after), 2)
            self.assertFalse(any(key.execution_codebook == 24 for key in after))
            self.assertEqual((receipt["installed_keys"], receipt["retained_planned_keys"]), (1,2))
            self.assertEqual(len(profiler.build_manifest(root, warmups=2, samples=3).requests), 2)

    def test_totality_rejects_missing_duplicate_and_incomplete_alias_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cells = (recorded_cell(root, "Q4_1", {"Auto": 1.}), recorded_cell(root, "Q4_K", {"Auto": 1.}))
            commit(root, cells); decisions = collect(root)
            for incomplete in ((), decisions.decisions*2):
                with self.assertRaisesRegex(ValueError, "every planned key"):
                    owner.validate_decision_totality(incomplete, cells)
            with self.assertRaisesRegex(ValueError, "alias"):
                owner.validate_decision_totality((replace(decisions.decisions[0], source_formats=("Q4_1",)),), cells)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_1",{"Auto":1.}); commit(root,(cell,))
            with self.assertRaisesRegex(ValueError,"complete original source aliases"):
                collect(root)

    def test_all_retain_still_rejects_corrupt_losing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":1.}); commit(root,(cell,))
            path=sweep.production_dense_prefill_cell_paths(root,cell).aggregate
            rewrite_rows(path, lambda rows: rows[-1].update(correctness_pass="0",byte_mismatches_vs_auto="1"))
            with self.assertRaises(ValueError): collect(root)

    def test_retained_observations_change_the_decision_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":1.}); commit(root,(cell,))
            first=collect(root)
            recorded_cell(root,"Q4_0",{"Auto":2.}); commit(root,(cell,))
            second=collect(root)
            self.assertEqual(first.decisions,second.decisions)
            self.assertNotEqual(first.observations_digest,second.observations_digest)
            self.assertNotEqual(first.digest,second.digest)

    def test_profiles_reject_missing_alias_baseline_or_foreign_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            cells=tuple(recorded_cell(root,label,{"Auto":10.,WMMA[0]:1.}) for label in ("Q4_1","Q4_K"))
            commit(root,cells); manifest=profiler.build_manifest(root,warmups=2,samples=3)
            altered=replace(manifest.requests[0],config_json={**manifest.requests[0].config_json,"decision_digest":"sha256:foreign"})
            variants=(manifest.requests[:-1],tuple(r for r in manifest.requests if r.config_json["purpose"]!="baseline"),
                      (altered,*manifest.requests[1:]))
            for requests in variants:
                with mock.patch.object(profiler,"read_profiler_request_manifest",return_value=replace(manifest,requests=requests)):
                    with self.assertRaisesRegex(ValueError,"current sealed timing corpus"):
                        profiler.require_profile(root,root/"requests",root/"evidence",warmups=2,samples=3)

    def test_publication_requires_original_base_and_profiles_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":10.,WMMA[0]:1.}); commit(root,(cell,))
            options=dict(base_include=root/"base.inc",receipt_path=root/"receipt",warmups=2,samples=3)
            with self.assertRaisesRegex(ValueError,"require winner and actual-Auto"):
                owner.write_policy(root,root/"out",**options)
            self.assertFalse((root/"out").exists())
            (root/"base.inc").write_text((root/"base.inc").read_text()+"\n")
            with self.assertRaisesRegex(ValueError,"original compiled policy"):
                owner.write_policy(root,root/"out",**options)

    def test_bound_cli_admits_all_destinations_before_any_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":1.}); commit(root,(cell,))
            paths=sweep.production_dense_prefill_cell_paths(root,cell)
            profile=root/"profile.json"; profile.write_text("original profile")
            output=root/"output.inc"; receipt=root/"receipt.json"
            output.write_text("prior output"); receipt.write_text("prior receipt")
            common=["overlay","--backend","rocm","--work-dir",str(root),"--output",str(output),
                    "--retain-base-include",str(root/"base.inc"),"--retention-receipt",str(receipt),
                    "--warmup-runs","2","--bench-runs","3"]
            for destination in (root/sweep.SWEEP_PLAN_FILENAME,paths.timing,paths.manifest,paths.log,
                                profile,output,receipt,root/"base.inc",paths.timing_staging,
                                root/"output.inc.inprogress", root/(sweep.SWEEP_PLAN_FILENAME+".inprogress"),
                                root/"profile.json.inprogress"):
                before={path:path.read_bytes() for path in root.rglob("*") if path.is_file()}
                args=[*common,"--summary-csv",str(destination)]
                if destination in (profile,root/"profile.json.inprogress"):
                    args += ["--profiler-requests",str(profile)]
                with self.subTest(path=destination), mock.patch.object(os.sys,"argv",args):
                    with self.assertRaisesRegex(ValueError,"overwrite|distinct|replace"):
                        overlay.main()
                self.assertEqual({path:path.read_bytes() for path in root.rglob("*") if path.is_file()},before)
            with mock.patch.object(os.sys,"argv",[*common,"--summary-csv",str(root/"summary.csv")]):
                self.assertEqual(overlay.main(),0)
            self.assertEqual(output.read_bytes(),(root/"base.inc").read_bytes())
            self.assertEqual(json.loads(receipt.read_text())["outcome"],"no-change")

    def test_changed_base_during_profile_validation_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":10.,WMMA[0]:1.}); commit(root,(cell,))
            def mutate(*args,**kwargs):
                (root/"base.inc").write_text((root/"base.inc").read_text()+"\n")
            with mock.patch.object(profiler,"require_profile",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"base changed"):
                    owner.write_policy(root,root/"out",base_include=root/"base.inc",receipt_path=root/"receipt",
                                       warmups=2,samples=3,profiler_requests=root/"requests",profiler_evidence=root/"evidence")
            self.assertFalse((root/"out").exists())

    def test_profiler_cli_rejects_evidence_outputs_and_staging_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":10.,WMMA[0]:1.}); commit(root,(cell,))
            paths=sweep.production_dense_prefill_cell_paths(root,cell)
            output=root/"requests.json"
            staging=root/"requests.json.inprogress"; staging.symlink_to(paths.timing)
            for destination in (root/sweep.SWEEP_PLAN_FILENAME,paths.aggregate,paths.timing,paths.manifest,output):
                before={path:path.read_bytes() for path in root.rglob("*") if path.is_file()}
                args=["profile","--work-dir",str(root),"--output",str(destination),"--warmup-runs","2","--bench-runs","3"]
                with mock.patch.object(os.sys,"argv",args):
                    with self.assertRaisesRegex(ValueError,"overwrite"):
                        profiler.main()
                self.assertEqual({path:path.read_bytes() for path in root.rglob("*") if path.is_file()},before)
            staging.unlink()
            unrelated=root/"unrelated"; unrelated.write_text("preserve")
            staging.symlink_to(unrelated)
            args=["profile","--work-dir",str(root),"--output",str(output),"--warmup-runs","2","--bench-runs","3"]
            with mock.patch.object(os.sys,"argv",args):
                self.assertEqual(profiler.main(),0)
            self.assertEqual(unrelated.read_text(),"preserve")
            self.assertEqual(len(profiler.read_profiler_request_manifest(output).requests),2)

    def test_unique_staging_preserves_unrelated_existing_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell=recorded_cell(root,"Q4_0",{"Auto":1.}); commit(root,(cell,))
            destinations=tuple(root/name for name in ("out.inc","receipt.json","summary.csv"))
            for index,path in enumerate(destinations):
                other=root/f"unrelated-{index}"; other.write_text("preserve")
                path.with_name(path.name+".inprogress").symlink_to(other)
            owner.write_policy(root,destinations[0],base_include=root/"base.inc",receipt_path=destinations[1],
                               summary_output=destinations[2],warmups=2,samples=3)
            self.assertEqual(destinations[0].read_bytes(),(root/"base.inc").read_bytes())
            self.assertTrue(all(path.read_text()=="preserve" for path in root.glob("unrelated-*")))


if __name__ == "__main__":
    unittest.main()
