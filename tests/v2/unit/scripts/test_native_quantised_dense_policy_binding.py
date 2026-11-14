#!/usr/bin/env python3
"""Replay compiled-table admission, immutable base plans and profiler provenance.

These device-free tests replay native query records against independently
specified integer rows. Counterfeit base bytes, changed selector closures and
old-plan relabeling fail before any accelerator process or publication.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import test_native_quantised_production_dense_prefill_sweep as fixtures
from test_native_quantised_production_dense_prefill_sweep import (
    _commit_test_cells, _test_provenance, _write_cell)
from native_quantised_dispatch import dense_prefill_policy_binding as binding
from native_quantised_dispatch import production_dense_prefill_sweep as sweep
from native_quantised_dispatch import native_prefill_profiler as profiler
from native_quantised_dispatch.corpus_provenance import mapping_digest
from native_quantised_dispatch.dense_production_overlay import (
    DenseOverlayEntry, DenseOverlayKey, DensePreparedWeightFamily, render_dense_overlay)
from native_quantised_dispatch.gpu_architecture import GPUArchitecture


def base_fixture() -> str:
    """Keep legacy and two architecture scopes in one independently specified base."""
    entry = DenseOverlayEntry("rocm", DenseOverlayKey(0,64,128,256), ("Q4_0",), ("test",),
        "N64/MT32/MB1/U2", (64,32,1,2,0,0), 0.,0.,1.)
    return render_dense_overlay((entry,
        replace(entry,key=replace(entry.key,architecture=GPUArchitecture.parse("rocm","gfx906"))),
        replace(entry,key=replace(entry.key,architecture=GPUArchitecture.parse("rocm","gfx1100")),
                candidate_id="WMMA/N64/MT32",launch=(64,32,1,0,0,4))))


def recorded_log(base: str) -> str:
    """Replay exactly one successful native cold query, never a timing observation."""
    count, digest = binding.installed_inventory_identity(base)
    return f"[ RUN      ] {binding.QUERY_TEST}\nROCM_DENSE_PREFILL_POLICY_V1,{count},{digest}\n[  PASSED  ] 1 test.\n"


def frozen_fixture(root: Path) -> dict[str, object]:
    """Exercise the real admission function with one recorded producer and response."""
    base = root / "base.inc"; base.write_text(base_fixture())
    result = subprocess.CompletedProcess([],0,recorded_log(base.read_text()))
    with mock.patch.object(binding,"trainer_provenance",return_value=_test_provenance("rocm")), \
            mock.patch.object(binding.subprocess,"run",return_value=result):
        return binding.freeze_installed_policy(base,root/"trainer")


class DensePolicyBindingReplay(unittest.TestCase):
    """Retain original Auto policy only through its actual compiled producer."""

    def test_compiled_integer_rows_cover_every_field_and_architecture(self):
        canonical = ("rocm-dense-prefill-installed-rows-v1\n"
            "0,64,128,256,0,64,32,1,2,0,0,2,0,0,0\n"
            "0,64,128,256,0,64,32,1,2,0,0,1,9,0,6\n"
            "0,64,128,256,0,64,32,1,0,0,4,1,11,0,0\n")
        self.assertEqual(binding.installed_inventory_identity(base_fixture()),
                         (3,"sha256:"+hashlib.sha256(canonical.encode()).hexdigest()))
        with self.assertRaisesRegex(ValueError,"selector ABI"):
            binding.installed_inventory_identity(base_fixture()+"// changed selector\n")

    def test_freeze_reads_compiled_owner_without_opening_a_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/"base.inc"; base.write_text(base_fixture())
            with mock.patch.object(binding,"trainer_provenance",return_value=_test_provenance("rocm")), \
                    mock.patch.object(binding.subprocess,"run",return_value=subprocess.CompletedProcess([],0,recorded_log(base.read_text()))) as run:
                frozen=binding.freeze_installed_policy(base,root/"trainer")
            binding.validate_installed_policy(frozen,_test_provenance("rocm"))
            self.assertEqual(frozen["base_text"],base.read_text())
            self.assertEqual(run.call_args.args[0][-1],"--gtest_filter="+binding.QUERY_TEST)
            for name in ("LLAMINAR_FORCE_CPU_ONLY_STARTUP","LLAMINAR_SKIP_CUDA_STARTUP","LLAMINAR_SKIP_ROCM_STARTUP"):
                self.assertEqual(run.call_args.kwargs["env"][name],"1")

    def test_query_rejects_missing_duplicate_skipped_or_changed_compiled_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/"base.inc"; base.write_text(base_fixture())
            good=recorded_log(base.read_text())
            for log in ("[  PASSED  ] 1 test.\n",good+good,good+"[  SKIPPED ] query\n",
                        good.replace("sha256:","sha256:0"),good.replace("POLICY_V1,3,","POLICY_V1,4,")):
                with self.subTest(log=log), mock.patch.object(binding,"trainer_provenance",return_value=_test_provenance("rocm")), \
                        mock.patch.object(binding.subprocess,"run",return_value=subprocess.CompletedProcess([],0,log)):
                    with self.assertRaises(ValueError): binding.freeze_installed_policy(base,root/"trainer")

    def test_changed_selector_closure_or_base_during_admission_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/"base.inc"; base.write_text(base_fixture())
            result=subprocess.CompletedProcess([],0,recorded_log(base.read_text()))
            with mock.patch.object(binding,"trainer_provenance",side_effect=[_test_provenance("rocm"),_test_provenance("rocm",core_digest="sha256:changed-selector")]), \
                    mock.patch.object(binding.subprocess,"run",return_value=result):
                with self.assertRaisesRegex(ValueError,"executable/core changed"):
                    binding.freeze_installed_policy(base,root/"trainer")
            def mutate(*args,**kwargs):
                base.write_text(base.read_text()+"\n"); return result
            with mock.patch.object(binding,"trainer_provenance",return_value=_test_provenance("rocm")), \
                    mock.patch.object(binding.subprocess,"run",side_effect=mutate):
                with self.assertRaisesRegex(ValueError,"include changed"):
                    binding.freeze_installed_policy(base,root/"trainer")

    def test_rehashed_base_cannot_replace_original_compiled_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            frozen=frozen_fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError,"different executable/core"):
                binding.validate_installed_policy(frozen,_test_provenance("rocm",core_digest="sha256:new-selector"))
            frozen["base_text"]=frozen["base_text"].replace("{64, 32, 1, 0, false, VNNIPrefillProducer::NativeWmma}",
                                                          "{128, 64, 1, 0, false, VNNIPrefillProducer::NativeWmma}")
            frozen["base_sha256"]=binding._digest(frozen["base_text"])
            frozen["compiled_entry_count"],frozen["compiled_inventory_sha256"]=binding.installed_inventory_identity(frozen["base_text"])
            frozen.pop("binding_digest"); frozen["binding_digest"]=mapping_digest(frozen)
            with self.assertRaisesRegex(ValueError,"actual compiled selector"):
                binding.validate_installed_policy(frozen,_test_provenance("rocm"))

    def test_bound_plan_resume_and_profiles_keep_original_policy(self):
        cell=fixtures.ProductionDensePrefillSweepTest()._scoped_wmma_cell()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen=frozen_fixture(root)
            _write_cell(root,"rocm",cell=cell,winning_candidate="WMMA/N64/MT32")
            plan=_commit_test_cells(root,"rocm",(cell,),installed_policy=frozen)
            self.assertEqual(plan["schema_version"],sweep.BOUND_SWEEP_PLAN_SCHEMA)
            self.assertEqual(sweep.load_planned_dense_prefill_cells(root/sweep.SWEEP_PLAN_FILENAME,"rocm"),((cell,),plan))
            binary=root/"trainer"; binary.write_text("unused"); binary.chmod(0o755)
            with mock.patch.object(sweep.subprocess,"run",side_effect=AssertionError("completed resume must not run a trainer")):
                self.assertEqual(sweep.run_missing_accelerator_cells(binary,root,(cell,),(5,),warmup_runs=2,bench_runs=3,
                    producer=_test_provenance("rocm")),())
            manifest=profiler.build_manifest(root,warmups=2,samples=3)
            self.assertEqual(len(manifest.requests),2)
            for request in manifest.requests:
                self.assertEqual(request.config_json["installed_policy_digest"],frozen["binding_digest"])
                self.assertEqual(request.config_json["sweep_plan_digest"],plan["plan_digest"])
            altered=replace(manifest.requests[0],config_json={**manifest.requests[0].config_json,"installed_policy_digest":"sha256:foreign"})
            with mock.patch.object(profiler,"read_profiler_request_manifest",return_value=replace(manifest,requests=(altered,*manifest.requests[1:]))):
                with self.assertRaisesRegex(ValueError,"current sealed timing corpus"):
                    profiler.require_profile(root,root/"requests",root/"evidence",warmups=2,samples=3)

    def test_plan_schema_and_binding_cannot_be_relabelled(self):
        cell=fixtures.ProductionDensePrefillSweepTest()._scoped_wmma_cell()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen=frozen_fixture(root); path=root/sweep.SWEEP_PLAN_FILENAME
            plan=sweep.write_sweep_plan(path,"rocm",(cell,),_test_provenance("rocm"),installed_policy=frozen)
            for schema in (sweep.SWEEP_PLAN_SCHEMA,sweep.SCOPED_SWEEP_PLAN_SCHEMA):
                changed={**plan,"schema_version":schema}; changed.pop("plan_digest"); changed["plan_digest"]=mapping_digest(changed)
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError): sweep.load_planned_dense_prefill_cells(path,"rocm")
            changed=dict(plan);changed.pop("installed_policy");changed.pop("plan_digest");changed["plan_digest"]=mapping_digest(changed)
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,"binding disagree"):
                sweep.load_planned_dense_prefill_cells(path,"rocm")
            with self.assertRaisesRegex(ValueError,"explicit native-WMMA"):
                sweep.write_sweep_plan(root/"exhaustive.json","rocm",(replace(cell,candidate_scope=sweep.DenseCandidateScope.EXHAUSTIVE),),
                                      _test_provenance("rocm"),installed_policy=frozen)

    def test_cli_cannot_attach_binding_after_collection_was_planned(self):
        cell=fixtures.ProductionDensePrefillSweepTest()._scoped_wmma_cell()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); base=root/"base.inc"; base.write_text(base_fixture())
            sweep.write_sweep_plan(root/sweep.SWEEP_PLAN_FILENAME,"rocm",(cell,),_test_provenance("rocm"))
            args=["sweep","plan","--backend","rocm","--prepared-family","native-quantised","--gpu-architecture","gfx1100",
                  "--candidate-scope","installed-vs-native-wmma","--output-dir",str(root),"--binary",str(root/"trainer"),
                  "--installed-base-include",str(base),"--format",cell.source_format.label,"--shape",cell.shape.name,"--m",str(cell.m)]
            with mock.patch.object(os.sys,"argv",args), mock.patch.object(binding,"freeze_installed_policy",side_effect=AssertionError("late query forbidden")):
                with self.assertRaisesRegex(ValueError,"unbound plan cannot acquire"):
                    sweep.main()


if __name__ == "__main__":
    unittest.main()
