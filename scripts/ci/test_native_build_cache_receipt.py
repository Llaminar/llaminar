# @file test_native_build_cache_receipt.py
# @brief Validate streamed BuildKit operation provenance and reject incomplete observations.
import json
from pathlib import Path
import tempfile
import unittest
from native_build_cache_receipt import receipt


class CacheReceiptTests(unittest.TestCase):
    def test_vertex_updates_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / 'progress'
            status = Path(directory) / 'status'
            status.write_text(json.dumps(dict(nonce='origin')))
            initial = dict(id='one', name='RUN python3 arm_native_build_evidence.py capture')
            for cached in (False, True):
                progress.write_text('\n'.join(map(json.dumps, [initial, dict(id='one', completed='now', cached=cached)])))
                self.assertEqual(receipt(progress, status), dict(cached=cached, origin='origin'))
            for events in ([], [initial], [initial, dict(id='one', completed='now', error='failed')], [dict(initial, completed='now'), dict(initial, id='two', completed='now')]):
                progress.write_text('\n'.join(map(json.dumps, events)))
                with self.assertRaises(ValueError):
                    receipt(progress, status)

    def test_real_buildkit_terminal_footer_and_malformed_json(self):
        footer = 'ERROR: failed to build: failed to solve: process "/bin/sh -c python3 scripts/ci/arm_native_build_evidence.py assert" did not complete successfully: exit code: 1'
        with tempfile.TemporaryDirectory() as directory:
            progress = Path(directory) / 'progress'
            status = Path(directory) / 'status'
            status.write_text(json.dumps(dict(nonce='origin')))
            vertex = dict(vertexes=[dict(digest='operation', name='RUN python3 arm_native_build_evidence.py capture', completed='now', cached=True)])
            progress.write_text(json.dumps(vertex) + '\n' + footer + '\n')
            self.assertEqual(receipt(progress, status)['terminal_diagnostics'], [footer])
            for invalid in ('{broken json', 'arbitrary warning', 'ERROR: something else'):
                progress.write_text(json.dumps(vertex) + '\n' + invalid)
                with self.assertRaises((ValueError, json.JSONDecodeError)):
                    receipt(progress, status)
