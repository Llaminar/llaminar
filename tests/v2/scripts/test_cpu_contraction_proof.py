#!/usr/bin/env python3
"""Device-free negative controls for installed contraction-proof admission."""
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from cpu_contraction_proof import source_manifest


class ContractionProofAdmission(unittest.TestCase):
    """Incomplete or stale evidence must never admit a builder proof."""

    def test_rejects_missing_stale_and_mutated_counts(self):
        """Replay complete evidence, then mutate each required admission field."""
        script = pathlib.Path(__file__).with_name('cpu_contraction_proof.py')
        root = script.resolve().parents[3]
        lines = [f'{isa} {p}{s}: fused={n}, expected={n}'
                 for isa, s in [('avx2', 'AVX2'), ('avx512f', 'AVX2'), ('avx512f', 'AVX512')]
                 for p, n in [('expert', 0), ('native', 1), ('negative', 1)]]
        valid = dict(version=1, sources=source_manifest(root), compiler='fixture',
                     evidence='\n'.join(lines))
        with tempfile.TemporaryDirectory() as directory:
            report = pathlib.Path(directory) / 'proof.json'
            command = [sys.executable, str(script), 'verify', '--compiler', 'fixture',
                       '--report', str(report)]
            for value, accepted in [(valid, True), ({**valid, 'sources': {}}, False),
                                    ({**valid, 'evidence': '\n'.join(lines[1:])}, False),
                                    ({**valid, 'compiler': 'other'}, False),
                                    ({**valid, 'evidence': valid['evidence'].replace(
                                        'expertAVX2: fused=0', 'expertAVX2: fused=1')}, False)]:
                report.write_text(json.dumps(value))
                result = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, accepted, result.stderr)


if __name__ == '__main__':
    unittest.main()
