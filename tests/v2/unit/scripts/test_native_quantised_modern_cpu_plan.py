#!/usr/bin/env python3
"""Device-free canonical CPU model-scope projection; runtime inventory is retained."""
from pathlib import Path
import sys
import unittest
from unittest import mock
import json
import tempfile
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tests/v2/performance/kernels'))
from native_quantised_dispatch import cpu_exact_sweep as sweep

class ModernCPUPlanTest(unittest.TestCase):
    def test_modern_scope_removes_only_old_owned_jobs_and_preserves_original_default(self):
        old=sweep.build_plan(28);modern=sweep.build_plan(28,model_scope='modern')
        self.assertEqual(old['total_jobs'],1269);self.assertEqual(modern['job_counts']['legacy_decode'],315)
        self.assertEqual(modern['job_counts']['legacy_grouped'],315)
        self.assertEqual(modern['job_counts']['expert_rows'],old['job_counts']['expert_rows'])
        old_ids={sweep.job_id(job) for job in sweep.iter_jobs(old)}
        self.assertTrue({sweep.job_id(job) for job in sweep.iter_jobs(modern)}.issubset(old_ids))
        self.assertIn('3B_AttnOut',{shape['name'] for shape in modern['legacy_shapes']})
        self.assertNotIn('model_scope',old)


class ModernCPUCLIPlanTest(unittest.TestCase):
    """New direct workflows select modern; saved v1 resumes keep paid identity."""
    def test_new_direct_cli_defaults_modern_without_native_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=dict(schema='synthetic',plan_digest='sha256:synthetic',job_counts={},total_jobs=0)
            with mock.patch.object(sweep,'build_plan',return_value=plan) as build, mock.patch.object(sweep,'_producer_generation',side_effect=AssertionError('native discovery')):
                self.assertEqual(sweep.main(['--output-dir',directory,'--threads','28','--plan-only']),0)
                self.assertEqual(build.call_args.kwargs['model_scope'],'modern')

    def test_saved_historical_v1_resume_retains_original_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=dict(schema='cpu-exact-sweep-v1',plan_digest='sha256:original',job_counts={},total_jobs=0)
            path=Path(directory)/sweep.PLAN_NAME;path.write_text(json.dumps(plan))
            before=path.read_bytes()
            with mock.patch.object(sweep,'build_plan',return_value=plan) as build, mock.patch.object(sweep,'_producer_generation',side_effect=AssertionError('native discovery')):
                self.assertEqual(sweep.main(['--output-dir',directory,'--threads','28','--resume','--plan-only']),0)
                self.assertEqual(build.call_args.kwargs['model_scope'],'all-retained')
                self.assertEqual(path.read_bytes(),before)

    def test_saved_explicit_scope_and_operator_override_remain_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            plan=dict(schema='synthetic',model_scope='modern',plan_digest='sha256:synthetic',job_counts={},total_jobs=0)
            (Path(directory)/sweep.PLAN_NAME).write_text(json.dumps(plan))
            with mock.patch.object(sweep,'build_plan',return_value=plan) as build:
                sweep.main(['--output-dir',directory,'--threads','28','--resume','--plan-only'])
                self.assertEqual(build.call_args.kwargs['model_scope'],'modern')
            foreign={**plan,'model_scope':'all-retained'}
            with mock.patch.object(sweep,'build_plan',return_value=foreign),self.assertRaises(SystemExit):
                sweep.main(['--output-dir',directory,'--threads','28','--resume','--model-scope','all-retained','--plan-only'])

if __name__=='__main__':unittest.main()
