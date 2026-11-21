#!/usr/bin/env python3
## @file test_release_accuracy_panels.py
# @brief Device-free panel projection, numerical scoring and evidence rejection tests.
"""Device-free panel projection, numerical scoring and evidence rejection tests.

Tiny logits and metadata are explicitly synthetic. They exercise the shared
full-vocabulary scorer and release admission boundaries without executing or
claiming qualification of a real model, topology or container image.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace, ModuleType
from contextlib import contextmanager

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
sys.path.insert(0, str(Path(__file__).parent))
import release_accuracy_panels as panels
import release_accuracy_evidence as evidence
import run_production_pipeline as pipeline
from production_artifacts import digest, write_json
from test_published_image_suite import image_pair, full_inventory
from run_production_parity_campaigns import REQUIRED_CSV_HEADERS


def inventory():
    """Two blessed model identities with three metadata-only topology fixtures each."""
    result = full_inventory()
    template = result['cells'][0]
    result['cells'] = []
    for index, filename in enumerate(panels.MODELS):
        for topology in range(3):
            row = deepcopy(template)
            identity=f'fixture{index}_{topology}'
            row['case'] = f'Suite/Fixture.ProductionParity/Model{index}Topology{topology}'
            row['configuration'].update(id=identity, model='/src/models/' + filename,
                runtime={'generation': {'serial_control_id': identity, 'mtp_policy': 'off'}})
            row['model_files'] = [row['configuration']['model']]
            result['cells'].append(row)
    return result


def authored():
    """Two independent prompts and two decode rows; no real reference approval."""
    return dict(schema='synthetic-unit-panel', decode_steps=2,
                cases=[dict(name='first', group='short', prompt='first prompt'),
                       dict(name='second', group='short', prompt='second prompt')])


def write_panel(root: Path, isa='AVX2'):
    """Build complete metadata boundary fixtures, never numerical certificates."""
    pair, original, panel = image_pair(), inventory(), authored()
    manifest = pipeline.e2e_projection(original, pair['source']['revision'])
    plan = panels.project(original, manifest, pair['source']['revision'])
    report = dict(schema=panels.SCHEMA, complete=True, passed=True, failures={},
                  source_revision=pair['source']['revision'], runtime_image=pair['images'][isa]['id'], cpu_isa=isa,
                  manifest_digest=digest(manifest), plan_digest=digest(plan), panel_digest=digest(panel), cells=[], rows=[],
                  acceptance_policy='initial_absolute_canonical_model_limits',
                  acceptance_scope=panels.SCOPE,calibrated=False)
    write_json(root / 'native-inventory.json', original)
    for row in plan:
        cell = dict(case=row['case'], model=row['model_key'], complete=True, passed=True, histories=[], reference_key=digest(row['model_key']))
        report['cells'].append(cell)
        for selected in panel['cases']:
            lane = root / 'native' / digest(row['case']) / selected['name']
            proof = lane / 'evidence'
            proof.mkdir(parents=True)
            metadata = proof / 'reference-metadata.txt'
            metadata.write_text('token_ids: 1,2\ndecode_tokens: 2,3,4\n')
            descriptor = dict(schema='llaminar.accuracy-panel-history.v1', canonical_cell=row['case'],
                              model=row['configuration']['model'], reference_scope='full_stages',
                              reference_arithmetic='cpu_fp32', reference_execution='incremental_cached',
                              reference_directory='/unused/synthetic-reference', prompt=selected['prompt'],
                              prompt_ids=[1,2], continuation_ids=[2,3,4], decode_rows=2, vocabulary=5,
                              metadata_sha256=panels.sha(metadata), authored_prompt_digest=digest(selected))
            write_json(lane / 'history.json', descriptor)
            binding = dict(descriptor, native_limits={'prefill_maximum': .1, 'decode_maximum': .2},
                           descriptor_sha256=panels.sha(lane / 'history.json'))
            write_json(proof / 'accuracy_panel_binding.json', binding)
            for name, header in REQUIRED_CSV_HEADERS.items():
                (proof / name).write_text(header + '\nsynthetic metadata fixture\n')
            native_report = dict(gtest_case=row['case'], preflight_return_code=0, fixture_return_code=0,
                                 model_staging_return_code=0, result={'return_code':0, 'artifact_contract_passed':True,
                                     'outcome':'completed', 'gtest_cases':[row['case']], 'validated_artifact_file_count':len(REQUIRED_CSV_HEADERS)})
            cell['histories'].append(dict(id=selected['name'], passed=True, binding=binding, rows=3,
                                         native_report=native_report,
                                         evidence_files={p.name: panels.sha(p) for p in sorted(proof.iterdir())}))
            for phase, offset in [('prefill',-1),('decode',0),('decode',1)]:
                report['rows'].append(dict(model=row['model_key'], case=row['case'], history=selected['name'],
                    context_group='short', phase=phase, decode_offset=offset, prompt_tokens=2, logical_position=1 if phase=='prefill' else 2+offset, kl_nats=.001,
                    top1_same=1, top5_overlap=1., total_variation=.001,
                    maximum_kl=.1 if phase=='prefill' else .2))
    report['summary'] = panels.summarize(report['rows'])
    panels.write_csv(root / 'accuracy-rows.csv', panels.CSV_FIELDS, report['rows'])
    panels.write_csv(root / 'accuracy-summary.csv', list(report['summary'][0]), report['summary'])
    report['files'] = {name:panels.sha(root / name) for name in ('accuracy-rows.csv','accuracy-summary.csv')}
    write_json(root / 'panel.json', report)
    return pair, manifest, original, report


class ReleaseAccuracyTests(unittest.TestCase):
    """Coverage cannot be inferred from one prompt, one model or one green summary."""

    def test_complete_red_numerics_still_collect_both_isas(self):
        """A rejected panel retains both ISA observations and retired owner receipts."""
        self._check_complete_red_numerics(safe_retirement=True)

    def test_red_numerics_cannot_admit_another_isa_with_live_children(self):
        """Numerical failure may continue only after its actual owner has retired."""
        self._check_complete_red_numerics(safe_retirement=False)

    def _check_complete_red_numerics(self, *, safe_retirement):
        """Replay identical failed panels with valid and invalid cleanup evidence."""
        import subprocess
        import run_published_image_suite as suite
        from test_published_image_suite import write_preflight_bundle, prerequisite_evidence
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pair = image_pair()
            bundle, output = root / 'preflight', root / 'accuracy'
            write_preflight_bundle(bundle, pair)
            output.mkdir()
            companions = prerequisite_evidence(pair)
            visited = []

            def execute(args, driver, command, lane, log):
                """Publish synthetic panel metadata through the current child boundary."""
                isa = lane.name.upper()
                visited.append(isa)
                self.assertEqual(command[command.index('--cpu-isa') + 1], isa)
                _, _, _, report = write_panel(lane / 'panel', isa)
                cleanup = dict(complete=True, passed=True, errors=[], unexpected_live=[],
                               owner='llaminar-suite-driver-' + '0' * 32, retired={},
                               driver_retirement=dict(Running=False, Pid=0, Status='absent'))
                if isa == 'AVX512' and not safe_retirement:
                    cleanup['driver_retirement'] = dict(Running=True, Pid=123, Status='running')
                write_json(lane / 'outer-cleanup.json', cleanup)
                if isa == 'AVX512':
                    report.update(passed=False, failures={'synthetic_history': 'numerical rejection'})
                    write_json(lane / 'panel/panel.json', report)
                    raise subprocess.CalledProcessError(1, command)

            def companion(args, source, lane):
                """Preserve the prerequisite test image identity for each ISA."""
                return {pipeline.ImageRole.TEST_RUNNER.value: companions[args.cpu_isa]['test_runner_image']}

            with patch.object(suite, 'build_driver', return_value='tools'), \
                 patch.object(suite, 'inventory_companion', side_effect=companion), \
                 patch.object(suite, 'run_in_driver', side_effect=execute):
                expected = 'release accuracy panel failures' if safe_retirement else 'complete native retirement'
                with self.assertRaisesRegex(ValueError, expected):
                    suite.run_accuracy(SimpleNamespace(preflight_bundle=bundle, models=root,
                        model_ramdisk_root=root, suite='accuracy'), pair, output)
            self.assertEqual(visited, list(suite.ISAS) if safe_retirement else ['AVX512'])
            receipt = json.loads((output / 'accuracy-receipt.json').read_text())
            self.assertFalse(receipt['complete'])
            self.assertEqual(set(receipt['reports']), set(suite.ISAS) if safe_retirement else set())
            self.assertEqual(set(receipt['retirement_reports']), set(suite.ISAS) if safe_retirement else set())
            if safe_retirement:
                self.assertEqual(set(receipt['failures']), {'AVX512'})

    def test_outer_driver_labels_and_retires_numerical_children_on_cancel(self):
        """Accuracy uses the shared exact-owner cleanup contract on interruption."""
        import run_published_image_suite as suite
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'cache').mkdir()
            lane = root / 'accuracy'
            lane.mkdir()
            socket = root / 'docker.sock'
            socket.touch()
            args = SimpleNamespace(output=root, models=root, model_ramdisk_root=root,
                                   e2e_bundle=None, suite='accuracy')
            with patch.dict(suite.os.environ, {'DOCKER_HOST': 'unix://' + str(socket)}), \
                 patch.object(suite.docker_paths, 'device_args', return_value=[]), \
                 patch.object(suite.docker_paths, 'mounts', return_value=[]), \
                 patch.object(suite.pipeline, 'run', side_effect=KeyboardInterrupt) as run, \
                 patch.object(suite, 'retire_coding_children') as retire, \
                 patch.object(suite, 'restore_lane_ownership'):
                with self.assertRaises(KeyboardInterrupt):
                    suite.run_in_driver(args, 'tools', ['python', 'release_accuracy_panels.py'], lane, 'accuracy.log')
            command = run.call_args.args[0]
            owner = command[command.index('--name') + 1]
            self.assertIn(suite.docker_paths.CODING_DRIVER_ENV + '=' + owner, command)
            self.assertIn(suite.docker_paths.CODING_OWNER_LABEL + '=' + owner, command)
            retire.assert_called_once_with(owner, lane / 'outer-cleanup.json')
            self.assertTrue(lane.stat().st_mode & 0o020)

    def test_projection_uses_blessed_serial_controls_and_requires_both_blessed_models(self):
        original = inventory()
        blessed = pipeline.e2e_projection(original, 'a'*40)
        self.assertEqual(len(panels.project(original, blessed, 'a'*40)), 6)
        qfn_unblessed = deepcopy(original)
        qfn_unblessed['cells'][-1]['configuration']['e2e'] = None
        serving = pipeline.e2e_projection(qfn_unblessed, 'a'*40)
        self.assertEqual(len(serving['cells']), 5)
        with self.assertRaisesRegex(ValueError, 'must include'):
            panels.project(qfn_unblessed, serving, 'a'*40)
        qfn_unblessed['cells'][-1]['configuration']['inventory_role'] = 'numerical_diagnostic'
        with self.assertRaises(ValueError):
            panels.project(qfn_unblessed, serving, 'a'*40)
        missing = deepcopy(blessed); missing['cells'].pop(0)
        with self.assertRaisesRegex(ValueError, 'must include'):
            panels.project(original, missing, 'a'*40)
        missing = deepcopy(original); missing['cells'][0]['configuration']['runtime']['generation']['mtp_policy']='dynamic'
        with self.assertRaisesRegex(ValueError, 'MTP-off'):
            panels.project(missing, blessed, 'a'*40)

    def test_installed_reference_fingerprint_ignores_generated_files_and_binds_collector(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('Dockerfile', 'requirements.txt', 'python/reference/model.py',
                         'scripts/docker/sdk.env', 'scripts/build/build.sh',
                         'scripts/ci/release_accuracy_panels.py', 'scripts/ci/qfn_accuracy_prompts.json',
                         'scripts/ci/release_accuracy_prompts.json'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('authored source')
            original = panels.reference_implementation(root)
            for name in ('.git/index', 'python/reference/__pycache__/model.pyc', 'python/reference/model.npy'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'generated data')
            self.assertEqual(panels.reference_implementation(root), original)
            (root / 'scripts/ci/release_accuracy_panels.py').write_text('changed history collector')
            self.assertNotEqual(panels.reference_implementation(root), original)

    def test_prompt_balanced_means_keep_prefill_decode_and_offsets_separate(self):
        rows = [dict(model='m',case='c',history=h,context_group='short',phase=phase,decode_offset=offset,
                     kl_nats=value,top1_same=1,top5_overlap=1.,maximum_kl=1.)
                for h,phase,offset,value in [('a','prefill',-1,.01),('b','prefill',-1,.03),
                    ('a','decode',0,.1),('a','decode',1,.3),('b','decode',0,.6)]]
        result = panels.summarize(rows)
        means = {(r['phase'],r['decode_offset']):r['mean_kl_nats'] for r in result}
        self.assertAlmostEqual(means[('prefill','-1')], .02)
        self.assertAlmostEqual(means[('decode','all')], .4)
        self.assertAlmostEqual(means[('decode','0')], .35)

    def test_real_fp64_scorer_rejects_trimmed_nonfinite_and_nonidentical_reset_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); reference=root/'reference'; reference.mkdir()
            directory=root/'native'; directory.mkdir()
            snapshots=directory/'llaminar_snapshots'/'rank-0'; snapshots.mkdir(parents=True)
            gold=np.array([1.,2.,3.,4.,5.],dtype=np.float32)
            records=[]
            for phase,offset in [('prefill',-1),('decode',0),('decode',1),('prefill',-1)]:
                path=snapshots/f'{len(records)}.f32';gold.tofile(path)
                name='LM_HEAD.npy' if phase=='prefill' else f'decode_step{offset}_LM_HEAD.npy'
                np.save(reference/name,gold.reshape(1,1,5))
                records.append(dict(key='LM_HEAD',phase=phase,step=offset,scalar_type='FP32',
                                    elements=5,exported_elements=5,file=str(path)))
            panels.write_csv(snapshots/'manifest.csv',list(records[0]),records)
            history=dict(id='frozen',context_group='short',decode_rows=2,vocabulary=5,prompt_ids=[0,1])
            binding={'native_limits':{'prefill_maximum':.1,'decode_maximum':.2}}
            row=dict(case='actual fixture',model_key='synthetic')
            scored=panels.score_history(directory,reference,history,row,binding)
            self.assertEqual([r['kl_nats'] for r in scored],[0.,0.,0.])
            changed=gold.copy();changed[-1]+=1.;changed.tofile(records[-1]['file'])
            with self.assertRaisesRegex(ValueError,'replay changed'):
                panels.score_history(directory,reference,history,row,binding)
            gold.tofile(records[-1]['file']);np.array([float('nan')]*5,dtype=np.float32).tofile(records[1]['file'])
            with self.assertRaisesRegex(ValueError,'nonfinite'):
                panels.score_history(directory,reference,history,row,binding)
            gold[:-1].tofile(records[1]['file'])
            with self.assertRaisesRegex(ValueError,'partial'):
                panels.score_history(directory,reference,history,row,binding)

    def test_reference_collection_resumes_completed_histories_and_retires_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);model_file=root/'model.gguf';model_file.write_text('synthetic metadata source')
            row=dict(reference_family='qwen35',model_files=[str(model_file)],configuration={'model':str(model_file)})
            model=SimpleNamespace(hf_model=SimpleNamespace(config=SimpleNamespace(vocab_size=5)),close=Mock())
            registry=ModuleType('python.reference');registry.create_reference_model=Mock(return_value=model)
            generator=ModuleType('python.reference.generate_qwen35_pipeline_snapshots')
            calls=[];fail=[True]
            def forward(model,prompt,depth,directory,**kwargs):
                calls.append(prompt);directory.mkdir()
                if prompt=='second prompt' and fail[0]:raise RuntimeError('interrupted reference collection')
                return 1,([0,1] if prompt=='first prompt' else [1,2]),[2,3,4]
            def metadata(directory,*args,**kwargs):
                (directory/'metadata.txt').write_text('original metadata '+str(args))
            generator.run_prefill_and_decode=forward;generator.write_metadata=metadata
            with patch.dict(sys.modules,{'python.reference':registry,
                    'python.reference.generate_qwen35_pipeline_snapshots':generator}), \
                 patch.object(panels,'reference_key',return_value='source-and-reference-key'):
                with self.assertRaisesRegex(RuntimeError,'interrupted'):
                    panels.references(row,authored(),root/'references')
                checkpoint=json.loads((root/'references/source-and-reference-key/references.json').read_text())
                self.assertFalse(checkpoint['complete']);self.assertTrue(checkpoint['reference_owner_retired'])
                self.assertEqual([h['id'] for h in checkpoint['histories']],['first'])
                fail[0]=False
                path,completed=panels.references(row,authored(),root/'references')
                self.assertTrue(completed['complete']);self.assertTrue(completed['reference_owner_retired'])
                self.assertEqual(calls,['first prompt','second prompt','second prompt'])
                self.assertEqual(len(list((path/'incomplete').iterdir())),1)
                self.assertEqual(panels.references(row,authored(),root/'references')[1],completed)
                self.assertEqual(registry.create_reference_model.call_count,2)
                self.assertEqual(model.close.call_count,2)

    def test_reference_close_failure_cannot_publish_a_complete_pack(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'model.gguf';source.write_text('synthetic')
            model=SimpleNamespace(hf_model=SimpleNamespace(config=SimpleNamespace(vocab_size=5)),
                                  close=Mock(side_effect=RuntimeError('unretired owner')))
            registry=ModuleType('python.reference');registry.create_reference_model=Mock(return_value=model)
            helper=ModuleType('python.reference.generate_qwen35_pipeline_snapshots')
            def forward(model,prompt,depth,directory,**kwargs):
                directory.mkdir();return 1,([0,1] if prompt=='first prompt' else [1,2]),[2,3,4]
            def metadata(directory,*args,**kwargs):(directory/'metadata.txt').write_text('metadata')
            helper.run_prefill_and_decode=forward;helper.write_metadata=metadata
            row=dict(reference_family='qwen35',model_files=[str(source)],configuration={'model':str(source)})
            with patch.dict(sys.modules,{'python.reference':registry,
                    'python.reference.generate_qwen35_pipeline_snapshots':helper}), \
                 patch.object(panels,'reference_key',return_value='key'), \
                 self.assertRaises(panels.ReferenceRetirementError):
                panels.references(row,authored(),root/'reference')
            state=json.loads((root/'reference/key/references.json').read_text())
            self.assertFalse(state['complete']);self.assertFalse(state['reference_owner_retired'])

    def test_companion_releases_cache_writer_lease_before_native_launch(self):
        import run_production_parity_campaigns as parity
        from test_published_image_suite import inventory_image
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);cache=root/'cache';cache.mkdir();events=[]
            pair=image_pair();write_json(root/'images.json',pair)
            write_json(root/'companion.json',{'test-runner':inventory_image(pair['source'])})
            @contextmanager
            def lease(*args):
                events.append('acquire');yield SimpleNamespace(root=cache);events.append('release')
            def launch(images,command,build,work,log):
                self.assertEqual(events,['acquire','release'])
                self.assertTrue(build.reference_cache_root.is_dir())
                self.assertEqual(command[:2],['python3','-B'])
                self.assertIn('run',command)
                events.append('native')
            args=SimpleNamespace(images=root/'companion.json',work=root,models=root/'models',
                                 model_ramdisk_root=root,cpu_isa='AVX2')
            with patch.object(parity,'model_staging_workspace',side_effect=lease), \
                 patch.object(pipeline,'run_test_runner',side_effect=launch):
                self.assertEqual(panels.companion(args),0)
            self.assertEqual(events,['acquire','release','native'])

    def test_complete_panel_revalidates_both_isa_contracts(self):
        for isa in ('AVX512','AVX2'):
            with self.subTest(isa=isa), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);pair,manifest,original,report=write_panel(root,isa)
                with patch.object(evidence,'panel',side_effect=authored):
                    self.assertEqual(evidence.validate_panel(root,pair,manifest,original,isa,'/src/models'),report)

    def test_both_isa_lanes_must_share_reference_and_teacher_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            reports={isa:write_panel(root/isa,isa)[-1] for isa in ('AVX512','AVX2')}
            evidence.validate_matched_histories(reports)
            original=deepcopy(reports)
            for field,value in [('prompt_ids',[2,1]),('continuation_ids',[4,3,2]),('metadata_sha256','changed')]:
                changed=deepcopy(original)
                changed['AVX2']['cells'][0]['histories'][0]['binding'][field]=value
                with self.assertRaisesRegex(ValueError,'same reference'):
                    evidence.validate_matched_histories(changed)
            changed=deepcopy(original);changed['AVX2']['cells'][0]['reference_key']='other implementation'
            with self.assertRaisesRegex(ValueError,'same reference'):
                evidence.validate_matched_histories(changed)

    def test_rows_limits_coverage_and_csv_tampering_fail_independently(self):
        faults=[lambda r:r['rows'].append(deepcopy(r['rows'][0])),
                lambda r:r['rows'].pop(), lambda r:r['rows'].reverse(),
                lambda r:r['rows'][0].update(kl_nats=float('nan')),
                lambda r:r['rows'][0].update(maximum_kl=100.),
                lambda r:r['cells'][0]['histories'].pop(),
                lambda r:r['cells'][0]['histories'][0]['native_report']['result'].update(return_code=1),
                lambda r:r.update(cpu_isa='AVX512'), lambda r:r.update(runtime_image='other')]
        for index,fault in enumerate(faults):
            with self.subTest(fault=index), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);pair,manifest,original,report=write_panel(root)
                fault(report);(root/'panel.json').write_text(json.dumps(report))
                with patch.object(evidence,'panel',side_effect=authored), self.assertRaises(ValueError):
                    evidence.validate_panel(root,pair,manifest,original,'AVX2','/src/models')
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);pair,manifest,original,report=write_panel(root)
            (root/'accuracy-rows.csv').write_text('changed')
            with patch.object(evidence,'panel',side_effect=authored), self.assertRaises(ValueError):
                evidence.validate_panel(root,pair,manifest,original,'AVX2','/src/models')


if __name__=='__main__':
    unittest.main()
