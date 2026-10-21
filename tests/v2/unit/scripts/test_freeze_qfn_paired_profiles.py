"""Toy-file configure boundary for opt-in paired metadata; no model/device work.

All source files contain one byte, descriptors are actual temporary-file stats,
and every numerical policy is the existing pure fixture. No real model, result,
reference payload, subprocess, backend or qualification receipt is consumed.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/ci'))
import freeze_qfn_paired_profiles as freezer
from model_kld_corpus import identity, metadata_digest
from test_model_kld_corpus_v2 import manifest, absolute_pilot


def write_json(path,value):
    """Write only synthetic small metadata and return its exact raw digest."""
    path.write_text(json.dumps(value,sort_keys=True))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def fixture(root):
    """Build one real toy filesystem descriptor per source role, no fake stat pins."""
    m=manifest()
    predictor=dict(name='inactive.gguf',path=str(root/'inactive.gguf'),bytes=1,object_identity='pending')
    m['source']['shared_predictor']=predictor
    for index,descriptor in enumerate(m['source']['shards']+[predictor]):
        path=root/descriptor['name'];path.write_bytes(bytes([index]));descriptor['path']=str(path)
        pin=identity(path)
        descriptor['object_identity']=json.dumps(dict(path=pin[0],device=pin[1],inode=pin[2],bytes=pin[3],mtime_ns=pin[4],ctime_ns=pin[5]))
    t=m['topologies'][0]
    t['canonical_cell']='Fixture/Case.ProductionParity/Original'
    t['expected_config']={'configuration':{'shared_predictor':predictor['path']}}
    t['runtime_config_digest']=metadata_digest(t['expected_config'])
    t['arithmetic']['hc_up']='two_term_q8'
    refs={
        'workload':write_json(root/'manifest.json',m),
        'catalog':write_json(root/'catalog.json',dict(schema=2,manifest_digest=metadata_digest(m))),
        'pilot':write_json(root/'pilot.json',absolute_pilot(m)),
        'benchmark':write_json(root/'benchmark.json',dict(schema=1,prompt_unit=' toy',prompt_repetitions=2,
            decode_tokens=2,warmup_iterations=1,measurement_iterations=3,seed=42,temperature=0,top_k=1,top_p=1))}
    doc=dict(schema='llaminar.qfn-local-paired-profiles.v1',source=m['source'],bindings=[dict(
        exact_cell=t['canonical_cell'],role='required',arithmetic=t['arithmetic'],**refs)])
    binding=root/'bindings.json';write_json(binding,doc)
    return binding,doc,m


class FreezeQFNPairedProfilesTests(unittest.TestCase):
    """All checks run before generated output; source IDs are later typed guards."""

    def test_complete_frozen_identity_and_idempotent_configure_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binding,doc,m=fixture(root)
            header,metadata=freezer.freeze(binding,root)
            self.assertIn(str(binding),header)
            self.assertIn(hashlib.sha256(binding.read_bytes()).hexdigest(),header)
            self.assertIn(metadata_digest(m['source']),header)
            self.assertIn(m['source']['shared_predictor']['object_identity'].replace('"','\\"'),header)
            self.assertEqual(metadata['source'],m['source'])
            self.assertEqual(metadata['exact_cells'],[doc['bindings'][0]['exact_cell']])
            self.assertEqual(len(metadata['dependencies']),10)
            self.assertEqual(metadata['scope'],'local_build_metadata_not_runtime_or_numeric_evidence')
            output=root/'generated.h';freezer.write_if_changed(output,header)
            before=identity(output);freezer.write_if_changed(output,header)
            self.assertEqual(identity(output),before)

    def test_manifest_policy_selector_and_benchmark_reject_before_output(self):
        mutations=(
            lambda d:d.update(schema='foreign'),
            lambda d:d.update(extra=True),
            lambda d:d['bindings'].append(copy.deepcopy(d['bindings'][0])),
            lambda d:d['bindings'][0].update(role='foreign'),
            lambda d:d['bindings'][0]['arithmetic'].update(hc_up='single_q8'),
            lambda d:d['bindings'][0].update(exact_cell='Unknown.ProductionParity/Unknown'),
            lambda d:d['bindings'][0]['workload'].update(sha256='f'*64),
            lambda d:d['source'].pop('shared_predictor'))
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);binding,d,m=fixture(root);mutate(d);write_json(binding,d)
                output=root/'generated.h';metadata=root/'generated.json'
                with self.assertRaises((ValueError,KeyError)):
                    freezer.main(['--bindings',str(binding),'--source-root',str(root),
                        '--header',str(output),'--metadata',str(metadata)])
                self.assertFalse(output.exists());self.assertFalse(metadata.exists())

    def test_original_same_size_replacement_and_missing_predictor_dependency_reject(self):
        for predictor in (False,True):
            with self.subTest(predictor=predictor),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);binding,d,m=fixture(root)
                original=Path(m['source']['shared_predictor']['path'] if predictor else m['source']['shards'][0]['path'])
                replacement=root/'replacement';replacement.write_bytes(b'x');replacement.replace(original)
                with self.assertRaisesRegex(ValueError,'reference-issued metadata'):
                    freezer.freeze(binding,root)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binding,d,m=fixture(root);Path(m['source']['shared_predictor']['path']).unlink()
            with self.assertRaises(FileNotFoundError):freezer.freeze(binding,root)

    def test_policy_group_inventory_cannot_omit_duplicate_or_add_foreign_context(self):
        for action in ('omit','duplicate','foreign'):
            with self.subTest(action=action),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);binding,d,m=fixture(root);pilot=absolute_pilot(m)
                if action=='omit':pilot['groups'].pop()
                if action=='duplicate':pilot['groups'].append(copy.deepcopy(pilot['groups'][0]))
                if action=='foreign':pilot['groups'][0]['context_group']='foreign'
                d['bindings'][0]['pilot']=write_json(root/'pilot.json',pilot);write_json(binding,d)
                with self.assertRaises(ValueError):freezer.freeze(binding,root)

    def test_no_missing_or_shortened_benchmark_obligation(self):
        for field,value in (('measurement_iterations',2),('decode_tokens',0),('prompt_repetitions',False)):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);binding,d,m=fixture(root)
                path=root/'benchmark.json';benchmark=json.loads(path.read_text());benchmark[field]=value
                d['bindings'][0]['benchmark']=write_json(path,benchmark);write_json(binding,d)
                with self.assertRaises(ValueError):freezer.freeze(binding,root)

    def test_changed_source_metadata_is_not_ignored_between_validation_and_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);binding,d,m=fixture(root)
            with patch.object(freezer,'unchanged',side_effect=ValueError('changed before publication')):
                with self.assertRaisesRegex(ValueError,'changed before publication'):freezer.freeze(binding,root)


if __name__=='__main__':unittest.main()
