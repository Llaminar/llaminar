#!/usr/bin/env python3
"""Prove the retired archive audit checks physical ownership without payload I/O.

Independent format-3 fixtures represent committed puts, touches, replacements
and retirements. Synthetic payloads are sparse files whose metadata is known;
opening any of them during the audit is forbidden by the test observer.
These small device-free cases require neither a model nor a native server.
"""
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
from prefix_archive_metadata_audit import inventory


class PrefixArchiveMetadataAuditTests(unittest.TestCase):
    """Authenticate live inode extents independently of the production writer."""

    def setUp(self):
        """Give each case one journal and an empty, separately owned payload tier."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.journal = self.root / ('a' * 64 + '.kvcache')
        self.journal.write_bytes(b'LLKVC001' + struct.pack('<II', 3, 80) + b'a' * 64)
        self.payloads = self.journal.with_name(self.journal.name + '.blocks')
        self.payloads.mkdir()
        self.sequence = 0

    def record(self, kind, key, identity=None, size=0):
        """Append independently framed metadata without invoking the native serializer."""
        self.sequence += 1
        value = {'type': {1: 'put', 2: 'delete', 3: 'touch'}[kind], 'key': key, 'sections': []}
        if kind == 1:
            value.update(storage_identity=list(identity), total_bytes=size,
                         sections=[{'bytes': size}, *[{'bytes': 0} for _ in range(5)]])
        encoded = json.dumps(value).encode()
        # This physical-ownership audit does not certify metadata checksums;
        # the native reader owns those. It checks complete record framing.
        with self.journal.open('ab') as stream:
            stream.write(b'LLKVR001' + struct.pack('<IIQQQQQ', 3, kind,
                len(encoded) + 72, len(encoded), 0, self.sequence, 0))
            stream.write(encoded)
            stream.write(b'LLKVDONE' + struct.pack('<Q', 0))

    def payload(self, identity, size):
        """Create a sparse fixture whose logical extent is sufficient for metadata proof."""
        path = self.payloads / f'{identity[0]:016x}{identity[1]:016x}.kvblock'
        with path.open('wb') as stream:
            stream.truncate(size)
        return path

    def audit(self, budget=4096):
        """Fail if the implementation opens a payload instead of observing its inode."""
        original = Path.open

        def guarded(path, *args, **kwargs):
            if path.parent == self.payloads:
                raise AssertionError('Payload content was opened by a metadata audit')
            return original(path, *args, **kwargs)

        with patch.object(Path, 'open', guarded):
            result = inventory(self.root, budget)
        self.assertTrue(result['passed'])
        self.assertEqual(result['payload_read_bytes'], 0)
        self.assertEqual(result['journal_bytes'], self.journal.stat().st_size)
        return result

    def test_empty_archive_has_only_metadata(self):
        result = self.audit()
        self.assertEqual(result['entries'], 0)
        self.assertEqual(result['metadata_read_bytes'], 80)

    def test_put_touch_and_replacement_preserve_one_current_inode(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        self.record(3, {'prefix': 1})
        self.record(1, {'prefix': 1}, (3, 4), 768)
        self.payload((3, 4), 768)
        result = self.audit()
        self.assertEqual(result['entries'], 1)
        self.assertEqual(result['active_payload_bytes'], 768)
        self.assertEqual(result['records'], 3)

    def test_retirement_leaves_no_physical_payload(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        self.record(2, {'prefix': 1})
        self.assertEqual(self.audit()['entries'], 0)
        self.payload((1, 2), 512)
        with self.assertRaisesRegex(ValueError, 'Physical payload files'):
            self.audit()

    def test_missing_or_wrong_extent_is_not_accepted_as_live_backing(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        with self.assertRaisesRegex(ValueError, 'Physical payload files'):
            self.audit()
        self.payload((1, 2), 511)
        with self.assertRaisesRegex(ValueError, 'Physical payload files'):
            self.audit()

    def test_budget_and_alias_ownership_are_independent_invariants(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        self.payload((1, 2), 512)
        self.audit(512)
        with self.assertRaisesRegex(ValueError, 'budget exceeded'):
            self.audit(511)
        self.record(1, {'prefix': 2}, (1, 2), 512)
        with self.assertRaisesRegex(ValueError, 'Duplicate storage ownership'):
            self.audit()

    def test_partial_journal_and_unknown_touch_fail_closed(self):
        self.record(3, {'prefix': 1})
        with self.assertRaisesRegex(ValueError, 'Touch references absent'):
            self.audit()
        data = self.journal.read_bytes()
        self.journal.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, 'Uncommitted journal'):
            self.audit()
        self.journal.write_bytes(data[:81])
        with self.assertRaisesRegex(ValueError, 'Incomplete journal'):
            self.audit()

    def test_symlink_payload_cannot_authenticate_backing(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        external = self.root / 'external'
        external.write_bytes(b'x' * 512)
        (self.payloads / f'{1:016x}{2:016x}.kvblock').symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'Non-regular payload'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
