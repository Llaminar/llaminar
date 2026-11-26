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
