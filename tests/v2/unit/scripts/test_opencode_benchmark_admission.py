#!/usr/bin/env python3
"""Device-free provenance tests for the post-benchmark OpenCode gate.

Use the existing published-image metadata fixtures and real ratchet validator.
The next workload must consume the exact, complete, passing ISA pair; editing
a summary flag cannot repair a missing, red, stale or contradictory report.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
sys.path.insert(0, str(ROOT / 'tests/v2/unit/scripts'))
import run_published_image_suite as suite
from production_artifacts import digest, write_json
from test_published_image_suite import benchmark_report, image_pair, write_e2e_bundle


class OpenCodeBenchmarkAdmissionTests(unittest.TestCase):
    """A dependent coding workload may only consume fully authenticated gates."""

    def setUp(self):
        """Create both tiny ISA lanes without Docker, models or accelerators."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.pair = image_pair()
        self.manifest = write_e2e_bundle(self.root / 'e2e', self.pair)
        _, self.e2e = suite.admit_e2e(self.root / 'e2e', self.pair)
        self.bundle = self.root / 'benchmarks'
        write_json(self.bundle / 'images.json', self.pair)
        write_json(self.bundle / 'manifest.json', self.manifest)
        baseline = json.loads((suite.ROOT / 'benchmarks/production/high_water.json').read_text())
        self.reports = {isa: benchmark_report(self.pair, self.manifest, self.e2e[isa], isa)
                        for isa in suite.ISAS}
        self.result = {'source': self.pair['source'], 'images': self.pair['images'],
                       'repository': self.pair['repository'], 'branch': 'develop',
                       'scope': 'full-http-e2e-and-benchmarks', 'passed': True,
                       'full_image_certification': False,
                       'regression_threshold_pct': baseline['regression_threshold_pct'],
                       'variants': {}}
        for isa, report in self.reports.items():
            self.result['variants'][isa] = {
                'report_digest': digest(report), 'e2e_report_digest': digest(self.e2e[isa]),
                'comparisons': report['comparisons'], 'cells': deepcopy(report['cells'])}
            write_json(self.bundle / isa.lower() / 'benchmarks.json', report)
        write_json(self.bundle / 'results.json', self.result)

    def admit(self):
        """Use the shared production boundary, not a fixture implementation."""
        return suite.admit_benchmarks(self.bundle, self.pair, self.manifest, self.e2e)

    def test_both_complete_passing_isas_are_admitted(self):
        self.assertEqual(self.admit(), (self.result, self.reports))

    def test_older_pair_and_subset_matrix_cannot_admit_work(self):
        for name, original, changed in (
                ('images.json', self.pair, {**self.pair, 'workflow_revision': 'd' * 40}),
                ('manifest.json', self.manifest, {**self.manifest, 'cells': []})):
            with self.subTest(name=name):
                write_json(self.bundle / name, changed)
                with self.assertRaises(ValueError):
                    self.admit()
                write_json(self.bundle / name, original)

    def test_every_isa_is_mandatory(self):
        for isa in suite.ISAS:
            path = self.bundle / isa.lower() / 'benchmarks.json'
            path.unlink()
            with self.subTest(isa=isa), self.assertRaises(FileNotFoundError):
                self.admit()
            write_json(path, self.reports[isa])

    def test_green_summary_cannot_hide_failed_or_diagnostic_report(self):
        for isa in suite.ISAS:
            for field, value in (('complete', False), ('passed', False), ('diagnostic', True),
                                 ('image', self.pair['images'][next(other for other in suite.ISAS
                                                                  if other != isa)]['id']),
                                 ('manifest_digest', '0' * 64), ('e2e_report_digest', '0' * 64)):
                changed = {**self.reports[isa], field: value}
                path = self.bundle / isa.lower() / 'benchmarks.json'
                write_json(path, changed)
                # Even a forged matching report digest cannot skip validation.
                altered = deepcopy(self.result)
                altered['variants'][isa]['report_digest'] = digest(changed)
                write_json(self.bundle / 'results.json', altered)
                with self.subTest(isa=isa, field=field), self.assertRaises(ValueError):
                    self.admit()
                write_json(path, self.reports[isa])
                write_json(self.bundle / 'results.json', self.result)

    def test_summary_and_per_cell_numbers_must_match(self):
        mutations = [lambda r: r.update(passed=False),
                     lambda r: r['variants'].pop(suite.ISAS[0]),
                     lambda r: r['variants'][suite.ISAS[0]]['cells'][0]
                     ['tokens_per_second'].update(decode=1234.0),
                     lambda r: r.update(regression_threshold_pct=99)]
        for mutate in mutations:
            changed = deepcopy(self.result)
            mutate(changed)
            write_json(self.bundle / 'results.json', changed)
            with self.assertRaises(ValueError):
                self.admit()
        write_json(self.bundle / 'results.json', self.result)


if __name__ == '__main__':
    unittest.main()
