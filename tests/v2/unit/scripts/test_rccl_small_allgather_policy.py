#!/usr/bin/env python3
"""Device-free malformed/stale policy-proof controls; no real native evidence is fabricated."""
import copy
import importlib.util
import json
from types import SimpleNamespace
from unittest import mock
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[4]
SPEC=importlib.util.spec_from_file_location('native_small_gather_proof',
    ROOT/'tests/v2/integration/build/test_rccl_small_allgather_policy.py')
P=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(P)


class NativeSmallAllGatherProofTests(unittest.TestCase):
    """Invoke the actual verifier with synthetic metadata only, never a native run."""

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.library=Path(self.tmp.name)/'fixture.so';self.library.write_bytes(b'metadata fixture not a DSO')
        self.report=dict(schema=1,sources=P.identities(),library_sha256=P.digest(self.library),
            negative_scope=P.VARIANTS[1],native_init_runtime_qualified=False,compile_database_sha256='b'*64,
            rccl_receipt='fixture-smallag-'+P.identities()[P.PATCH],
            compiler=dict(path='/fixture/clang',stat=[1,2,3,4,5]),compiler_version='fixture only',
            architecture='gfx906',sdk='/fixture/sdk',
            native_sources={'/fixture/'+name:'a'*64 for name in (*P.NATIVE,*P.HEADERS)},
            native_compile_inputs=[dict(source='/fixture/'+name,flags=['-DFIXTURE_ONLY','-fgpu-rdc']) for name in P.NATIVE],
            observations=[])
        for variant in P.VARIANTS:
            for mode,names in [('scope',P.SCOPE),('selection',P.SELECTION)]:
                negative=variant==P.VARIANTS[1] and mode=='selection'
                cases=[dict(name=name,checks=5,failures=int(negative and name in P.OLD_FAILURES),observed=[0]*6) for name in names]
                self.report['observations'].append(dict(variant=variant,mode=mode,returncode=int(negative),stderr='',
                                                       measured=dict(mode=mode,cases=cases)))

    def test_changed_dependency_rebuild_retires_prior_and_interrupted_proof(self):
        """An early new-build failure must not leave a prior success accepted."""
        destination=Path(self.tmp.name)/'proof.json'
        pending=destination.with_suffix('.json.tmp')
        destination.write_text('prior successful proof')
        pending.write_text('interrupted newer proof')
        args=SimpleNamespace(report=destination,rccl_build=Path(self.tmp.name)/'missing-build')
        with mock.patch.object(P,'native_inputs',side_effect=RuntimeError('changed dependency unavailable')):
            with self.assertRaisesRegex(RuntimeError,'changed dependency unavailable'):
                P.compile_report(args)
        self.assertFalse(destination.exists())
        self.assertFalse(pending.exists())

    def test_interrupted_publication_is_not_final_and_next_rebuild_can_publish(self):
        """A partially written proof never substitutes for an accepted final file."""
        destination=Path(self.tmp.name)/'new-directory'/'proof.json'
        pending=destination.with_suffix('.json.tmp')
        with self.assertRaises(TypeError):
            P.publish_verified_report({'not_serializable':object()},destination)
        self.assertFalse(destination.exists())
        self.assertTrue(pending.exists())
        P.retire_proof(destination)
        P.retire_proof(destination)  # Fresh rebuild is also a valid lifecycle.
        P.publish_verified_report(self.report,destination)
        self.assertEqual(json.loads(destination.read_text()),self.report)
        self.assertFalse(pending.exists())
        P.verify(json.loads(destination.read_text()),self.library)

    def test_native_relocation_mode_is_preserved_and_conflicts_reject(self):
        """Original RDC mode must survive host-proof lowering and packaging."""
        self.assertEqual(P.native_relocation_mode(['hipcc','-O3','-fgpu-rdc','-c','input.cc']),
                         ['-fgpu-rdc'])
        for tokens in ([],['-fno-gpu-rdc'],['-fgpu-rdc','-fno-gpu-rdc'],
                       ['-fno-gpu-rdc','-fgpu-rdc'],['-fgpu-rdc','-fgpu-rdc'],['-fgpu-rdc=false']):
            with self.subTest(tokens=tokens),self.assertRaises(RuntimeError):
                P.native_relocation_mode(tokens)
        for tokens in (['-DFIXTURE_ONLY'],['-DFIXTURE_ONLY','-fno-gpu-rdc'],
                       ['-fgpu-rdc','-fno-gpu-rdc'],['-fgpu-rdc','-fgpu-rdc']):
            bad=copy.deepcopy(self.report)
            bad['native_compile_inputs'][0]['flags']=tokens
            with self.subTest(installed=tokens),self.assertRaises(RuntimeError):
                P.verify(bad,self.library)

    def test_complete_metadata_contract(self):
        P.verify(self.report,self.library)

    def test_every_source_and_native_runtime_is_bound(self):
        for name in P.INPUTS:
            bad=copy.deepcopy(self.report);bad['sources'][name]='bad'
            with self.subTest(source=name),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        self.library.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):P.verify(self.report,self.library)

    def test_every_abi_publication_and_compile_input_is_required(self):
        for name in self.report['native_sources']:
            bad=copy.deepcopy(self.report);del bad['native_sources'][name]
            with self.subTest(native=name),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        for key in ('compiler','compiler_version','architecture','sdk','rccl_receipt','native_compile_inputs','compile_database_sha256'):
            bad=copy.deepcopy(self.report);del bad[key]
            with self.subTest(closure=key),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        bad=copy.deepcopy(self.report);bad['native_compile_inputs'][0]['flags']=[]
        with self.assertRaises(RuntimeError):P.verify(bad,self.library)

    def test_missing_duplicate_crashed_or_vacuous_outcomes_reject(self):
        for index in range(4):
            for mutate in (lambda row:row.update(returncode=-11),
                           lambda row:row['measured']['cases'].pop(),
                           lambda row:row['measured']['cases'][0].update(checks=0),
                           lambda row:row['measured']['cases'][0].update(failures=True),
                           lambda row:row.update(stderr='unexpected diagnostic')):
                bad=copy.deepcopy(self.report);mutate(bad['observations'][index])
                with self.subTest(index=index),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        bad=copy.deepcopy(self.report);bad['observations'].pop()
        with self.assertRaises(RuntimeError):P.verify(bad,self.library)
        bad=copy.deepcopy(self.report);bad['observations'].append(bad['observations'][0])
        with self.assertRaises(RuntimeError):P.verify(bad,self.library)

    def test_original_must_detect_every_target_but_preserve_excluded_postpasses(self):
        for name in P.OLD_FAILURES:
            bad=copy.deepcopy(self.report)
            next(case for case in bad['observations'][3]['measured']['cases'] if case['name']==name)['failures']=0
            with self.subTest(name=name),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        for name in ('large_16385','other_collective','explicit_selector_excluded','native_gfx950_postpass','native_gfx942_pipeline'):
            bad=copy.deepcopy(self.report)
            next(case for case in bad['observations'][3]['measured']['cases'] if case['name']==name)['observed'][2]=99
            with self.subTest(name=name),self.assertRaises(RuntimeError):P.verify(bad,self.library)
        bad=copy.deepcopy(self.report);bad['negative_scope']='entire_original_runtime'
        with self.assertRaises(RuntimeError):P.verify(bad,self.library)
        bad=copy.deepcopy(self.report);bad['native_init_runtime_qualified']=True
        with self.assertRaises(RuntimeError):P.verify(bad,self.library)


if __name__=='__main__':unittest.main()
