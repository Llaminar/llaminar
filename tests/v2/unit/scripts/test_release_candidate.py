#!/usr/bin/env python3
## @file test_release_candidate.py
# @brief Device-free release routing, frozen lineage and phase-order regressions.
"""Device-free release routing, frozen lineage and phase-order regressions.

Temporary Git histories authenticate real ancestry and merge-tree behavior.
Metadata fixtures never claim numerical, hardware or image qualification.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import release_candidate as rc
import publish_master_release as release
import apply_master_ruleset as rules
import wait_for_develop_image_gate as wait
import yaml


class ReleaseCandidateTests(unittest.TestCase):
    """An RC name cannot replace source ancestry, owner review or complete gates."""

    def test_routes_reject_develop_to_master_features_to_rc_and_forks(self):
        for head, base in [('rc/2026-11-13.1', 'master'), ('develop', 'rc/2026-11-13.1')]:
            rc.check_route('Llaminar/llaminar', 'Llaminar/llaminar', head, base)
        for head, base in [('develop', 'master'), ('feat/x', 'master'), ('feat/x', 'rc/v1'),
                           ('rc/', 'master'), ('rc/a/b', 'master')]:
            with self.subTest(head=head, base=base), self.assertRaises(ValueError):
                rc.check_route('Llaminar/llaminar', 'Llaminar/llaminar', head, base)
        with self.assertRaises(ValueError):
            rc.check_route('Llaminar/llaminar', 'other/fork', 'rc/v1', 'master')

    def test_real_frozen_snapshot_survives_new_develop_commits_and_rejects_rc_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
            git('init', '-q'); git('config', 'user.name', 'Test'); git('config', 'user.email', 'test@example.com')
            (root / 'source').write_text('a')
            git('add', 'source'); git('commit', '-qm', 'develop source')
            frozen = git('rev-parse', 'HEAD')
            git('update-ref', 'refs/remotes/origin/develop', frozen)
            with patch.object(rc, 'ROOT', root):
                receipt = rc.admit('Llaminar/llaminar', 'rc/v1', frozen, 'origin/develop')
                (root / 'source').write_text('b'); git('commit', '-qam', 'later development')
                tip = git('rev-parse', 'HEAD'); git('update-ref', 'refs/remotes/origin/develop', tip)
                self.assertEqual(rc.validate(receipt, 'Llaminar/llaminar', 'rc/v1', frozen,
                                             git('rev-parse', frozen + '^{tree}')), frozen)
                (root / 'source').write_text('rc-only edit'); git('commit', '-qam', 'foreign RC edit')
                with patch.object(rc, 'github_json', return_value=[]), self.assertRaisesRegex(ValueError, 'neither'):
                    rc.admit('Llaminar/llaminar', 'rc/v1', git('rev-parse', 'HEAD'), 'origin/develop')
                fake = deepcopy(receipt); fake['develop_revision'] = git('rev-parse', 'HEAD')
                with self.assertRaisesRegex(ValueError, 'outside'):
                    rc.validate(fake, 'Llaminar/llaminar', 'rc/v1', frozen, receipt['source_tree'])

    def test_merged_develop_to_rc_requires_exact_source_tree_and_repository(self):
        source, head = 'a' * 40, 'b' * 40
        pr = dict(merged_at='today', merge_commit_sha=head, number=7, base={'ref': 'rc/v1'},
                  head={'ref': 'develop', 'sha': source, 'repo': {'full_name': 'Llaminar/llaminar'}})
        def ancestor(before, after): return before != head
        with patch.object(rc, 'ancestor', side_effect=ancestor), patch.object(rc, 'git', return_value='c' * 40), \
             patch.object(rc, 'github_json', return_value=[pr]):
            self.assertEqual(rc.admit('Llaminar/llaminar', 'rc/v1', head, 'origin/develop')['source_revision'], source)
        with patch.object(rc, 'ancestor', side_effect=ancestor), patch.object(rc, 'git', side_effect=['x', 'y']), \
             patch.object(rc, 'github_json', return_value=[pr]), self.assertRaisesRegex(ValueError, 'outside'):
            rc.admit('Llaminar/llaminar', 'rc/v1', head, 'origin/develop')
        pr['head']['repo']['full_name'] = 'other/fork'
        with patch.object(rc, 'ancestor', side_effect=ancestor), patch.object(rc, 'github_json', return_value=[pr]), \
             self.assertRaises(ValueError):
            rc.admit('Llaminar/llaminar', 'rc/v1', head, 'origin/develop')

    def test_deleted_squashed_rc_head_is_fetched_from_its_exact_pull_ref(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);remote=root/'remote.git';work=root/'work';checkout=root/'release'
            subprocess.run(['git','init','-q','--bare',str(remote)],check=True)
            subprocess.run(['git','clone','-q',str(remote),str(work)],check=True)
            def git(*args):return subprocess.check_output(['git',*args],cwd=work,text=True).strip()
            git('config','user.name','Test');git('config','user.email','test@example.com')
            git('switch','-q','-c','develop');(work/'source').write_text('a');git('add','source');git('commit','-qm','first')
            seed=git('rev-parse','HEAD');(work/'source').write_text('b');git('commit','-qam','develop source')
            source=git('rev-parse','HEAD');git('push','-q','origin','develop')
            git('switch','-q','-c','rc/v1',seed);git('merge','--squash','develop');git('commit','-qm','RC merge')
            head=git('rev-parse','HEAD')
            # Transfer the object through the pull namespace, without retaining
            # an RC branch in the remote heads fetched by a release checkout.
            git('push','-q','origin',head+':refs/pull/7/head')
            subprocess.run(['git','clone','-q','--single-branch','--branch','develop',str(remote),str(checkout)],check=True)
            with patch.object(release,'ROOT',checkout):
                release.fetch_rc_head({'number':7,'head':{'sha':head}})
                self.assertEqual(release.git('rev-parse','FETCH_HEAD'),head)
                with self.assertRaisesRegex(ValueError,'differs'):
                    release.fetch_rc_head({'number':7,'head':{'sha':source}})

    def test_rc_source_accepts_the_exact_merged_pr_when_commit_index_lags(self):
        head,source='b'*40,'a'*40
        pr={'merged_at':'today','merge_commit_sha':head,'number':7,'base':{'ref':'rc/v1'},
            'head':{'ref':'develop','sha':source,'repo':{'full_name':'Llaminar/llaminar'}}}
        with patch.object(rc,'ancestor',side_effect=lambda before,after:before!=head), \
             patch.object(rc,'git',return_value='c'*40), \
             patch.object(rc,'github_json',side_effect=[[],[pr]]):
            self.assertEqual(rc.admit('Llaminar/llaminar','rc/v1',head,'origin/develop')['develop_pr'],7)

    def test_historical_rc_publication_requires_protected_develop_ancestry(self):
        revision = 'a' * 40
        branch = {'commit': {'sha': 'b' * 40}}
        run = {'id': 42, 'head_sha': revision, 'head_branch': 'develop', 'event': 'push',
               'status': 'completed', 'conclusion': 'success'}
        with patch.object(wait, 'github_json', side_effect=[branch, {'status': 'ahead'}, {'workflow_runs': [run]}]):
            self.assertEqual(wait.wait_for_image_gate('Llaminar/llaminar', revision, 1, frozen_rc=True), 42)
        with patch.object(wait, 'github_json', side_effect=[branch, {'status': 'diverged'}]), self.assertRaises(ValueError):
            wait.wait_for_image_gate('Llaminar/llaminar', revision, 1, frozen_rc=True)

    def test_required_gate_graph_and_source_bound_artifacts(self):
        workflow = yaml.load((ROOT / '.github/workflows/master-pr-certification.yml').read_text(), Loader=yaml.BaseLoader)
        jobs = workflow['jobs']
        self.assertEqual(jobs['accuracy']['needs'], ['source', 'preflight'])
        self.assertEqual(jobs['coding_matrix']['needs'], ['source', 'accuracy'])
        self.assertEqual(jobs['benchmarks']['needs'], ['source', 'coding'])
        self.assertEqual(jobs['eligible']['needs'], ['source', 'preflight', 'accuracy', 'coding', 'benchmarks'])
        for job in jobs.values():
            self.assertEqual(job['runs-on'], ['llaminar-xeon-host'])
        names = {job['name'] for job in jobs.values()} | {'Release branch lineage'}
        self.assertLessEqual(set(rules.REQUIRED_CHECKS), names)
        commands = '\n'.join(s.get('run', '') for j in jobs.values() for s in j['steps'])
        self.assertIn('--frozen-rc', commands)
        self.assertIn('--accuracy-bundle', commands)
        self.assertIn('--opencode-bundle', commands)
        self.assertNotIn('--model-kld-canaries', commands)
        self.assertNotIn('run_published_image_suite.py e2e', commands)
        self.assertNotIn('run_model_parity_e2e.py', commands)
        self.assertNotIn('lfs pull', commands)
        for key, job in jobs.items():
            self.assertIn('!github.event.pull_request.draft', job['if'])
        release = (ROOT / '.github/workflows/release.yml').read_text()
        self.assertIn("startsWith(github.event.pull_request.head.ref, 'rc/')", release)
        self.assertNotIn("github.event.pull_request.head.ref == 'develop'", release)

    def test_rc_guards_keep_owner_approval_and_disallow_direct_updates(self):
        planned = rules.proposed_rc_ruleset()
        self.assertEqual(planned['conditions']['ref_name']['include'], ['refs/heads/rc/**'])
        self.assertEqual(planned['bypass_actors'], [])
        self.assertLessEqual({'non_fast_forward', 'pull_request', 'required_status_checks'},
                             {r['type'] for r in planned['rules']})
        original = dict(name='Owner approval', target='branch', enforcement='active', bypass_actors=[],
                        conditions={'ref_name': {'include': ['refs/heads/master', 'refs/heads/develop'], 'exclude': []}},
                        rules=[{'type': 'pull_request', 'parameters': {'required_approving_review_count': 1}}])
        self.assertEqual(rules.proposed_owner_ruleset(original)['rules'], original['rules'])
        self.assertNotIn('refs/heads/rc/**', original['conditions']['ref_name']['include'])


if __name__ == '__main__':
    unittest.main()
