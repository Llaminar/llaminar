#!/usr/bin/env python3
"""Prove the retired archive audit checks physical ownership without payload I/O.

Independent format-5 fixtures represent committed puts, touches, replacements
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
        self.journal.write_bytes(b'LLKVC001' + struct.pack('<II', 5, 80) + b'a' * 64)
        self.payloads = self.journal.with_name(self.journal.name + '.blocks')
        self.payloads.mkdir()
        self.sequence = 0

    def record(self, kind, key, identity=None, size=0):
        """Append independently framed metadata without invoking the native serializer."""
        self.sequence += 1
        value = {'type': {1: 'put', 2: 'delete', 3: 'touch'}[kind], 'key': key, 'sections': []}
        if kind == 1:
            value.update(storage_identity=list(identity), total_bytes=size,
                         sections=[{'name': name, 'bytes': size if index == 0 else 0}
                                   for index, name in enumerate(('kv', 'hybrid', 'mtp',
                                       'terminal_hidden', 'terminal_logits', 'model_auxiliary',
                                       'model_runtime_state'))])
        encoded = json.dumps(value).encode()
        # This physical-ownership audit does not certify metadata checksums;
        # the native reader owns those. It checks complete record framing.
        with self.journal.open('ab') as stream:
            stream.write(b'LLKVR001' + struct.pack('<IIQQQQQ', 5, kind,
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

    def test_current_header_version_and_exact_model_identity_are_required(self):
        original = self.journal.read_bytes()
        for version in (3, 4, 6):
            with self.subTest(version=version):
                self.journal.write_bytes(original[:8] + struct.pack('<I', version) + original[12:])
                with self.assertRaisesRegex(ValueError, 'Unqualified journal'):
                    self.audit()
        for identity in (b'b' * 64, b'A' * 64, b'g' * 64, bytes([255]) * 64):
            with self.subTest(identity=identity[:1]):
                self.journal.write_bytes(original[:16] + identity)
                with self.assertRaisesRegex(ValueError, 'Unqualified journal'):
                    self.audit()
        self.journal.write_bytes(original)
        self.assertEqual(self.audit()['format'], 5)

    def test_current_record_version_and_seven_ordered_sections_are_required(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        self.payload((1, 2), 512)
        original = self.journal.read_bytes()
        self.journal.write_bytes(original[:88] + struct.pack('<I', 3) + original[92:])
        with self.assertRaisesRegex(ValueError, 'invalid metadata extent'):
            self.audit()
        for change in ('six_sections', 'foreign_name', 'wrong_order'):
            with self.subTest(change=change):
                value = json.loads(original[136:-16])
                if change == 'six_sections':
                    value['sections'].pop()
                elif change == 'foreign_name':
                    value['sections'][-1]['name'] = 'foreign_state'
                else:
                    value['sections'][-2], value['sections'][-1] = value['sections'][-1], value['sections'][-2]
                encoded = json.dumps(value).encode()
                preamble = b'LLKVR001' + struct.pack('<IIQQQQQ', 5, 1, len(encoded) + 72,
                    len(encoded), 0, 1, 0)
                self.journal.write_bytes(original[:80] + preamble + encoded + original[-16:])
                with self.assertRaisesRegex(ValueError, 'Invalid payload metadata'):
                    self.audit()
        self.journal.write_bytes(original)
        self.audit()

    def test_seventh_runtime_state_section_contributes_to_exact_payload_extent(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        original = self.journal.read_bytes()
        value = json.loads(original[136:-16])
        value['sections'][0]['bytes'] = 480
        value['sections'][-1]['bytes'] = 32
        encoded = json.dumps(value).encode()
        self.journal.write_bytes(original[:80] + b'LLKVR001' + struct.pack('<IIQQQQQ',
            5, 1, len(encoded) + 72, len(encoded), 0, 1, 0) + encoded + original[-16:])
        self.payload((1, 2), 512)
        self.assertEqual(self.audit()['active_payload_bytes'], 512)

    def test_symlink_payload_cannot_authenticate_backing(self):
        self.record(1, {'prefix': 1}, (1, 2), 512)
        external = self.root / 'external'
        external.write_bytes(b'x' * 512)
        (self.payloads / f'{1:016x}{2:016x}.kvblock').symlink_to(external)
        with self.assertRaisesRegex(ValueError, 'Non-regular payload'):
            self.audit()


if __name__ == '__main__':
    unittest.main()
