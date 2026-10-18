#!/usr/bin/env python3
"""Exercise the standalone production decoder with a tiny metadata-only GGUF.

The actual serving-image utility is a CTest build dependency, so public header
requirements must propagate without the test directory's private include paths.
Byte vocabulary fixtures prove model admission and Unicode/XML decoding without
model weights, GPU initialization, downloaded assets, or a second native parser.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


def string(value):
    """Encode one GGUF UTF-8 string with its explicit byte length."""
    encoded = value.encode('utf-8')
    return struct.pack('<Q', len(encoded)) + encoded


def metadata_fixture(path):
    """Write all byte tokens and Qwen policy metadata, with no tensor directory."""
    visible = set(range(33, 127)) | set(range(161, 173)) | set(range(174, 256))
    missing = [byte for byte in range(256) if byte not in visible]
    vocabulary = [chr(byte if byte in visible else 256 + missing.index(byte))
                  for byte in range(256)] + ['<|endoftext|>']
    entries = []
    for name, value in (
            ('general.architecture', 'qwen35moe'),
            ('general.name', 'Qwen3.6-35B-A3B'),
            ('tokenizer.ggml.model', 'gpt2'), ('tokenizer.ggml.pre', 'qwen2')):
        entries.append(string(name) + struct.pack('<I', 8) + string(value))
    for name, value in (('qwen35moe.context_length', 262144),
                        ('tokenizer.ggml.bos_token_id', 256),
                        ('tokenizer.ggml.eos_token_id', 256)):
        entries.append(string(name) + struct.pack('<II', 4, value))
    entries.append(string('tokenizer.ggml.tokens') + struct.pack('<IIQ', 9, 8, len(vocabulary))
                   + b''.join(string(token) for token in vocabulary))
    header = b'GGUF' + struct.pack('<IQQ', 3, 0, len(entries)) + b''.join(entries)
    path.write_bytes(header + b'\0' * (-len(header) % 32))


class NativeToolDecoderTests(unittest.TestCase):
    """Use the standalone executable through its installed command contract."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='llaminar-decoder-')
        self.addCleanup(self.temporary.cleanup)
        self.model = Path(self.temporary.name) / 'metadata.gguf'
        metadata_fixture(self.model)

    def invoke(self, arguments, value=None):
        """Run without accelerator visibility and retain precise failure output."""
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', HIP_VISIBLE_DEVICES='',
                           ROCR_VISIBLE_DEVICES='', OMP_NUM_THREADS='1')
        return subprocess.run([str(DECODER), *map(str, arguments)],
            input=None if value is None else json.dumps(value), text=True,
            capture_output=True, timeout=10, env=environment)

    def test_metadata_only_admission_uses_the_model_context(self):
        result = self.invoke(['--describe-model', self.model])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), {
            'schema': 1, 'architecture': 'qwen35moe', 'context_tokens': 262144,
            'context_alignment_tokens': 64, 'default_thinking': True,
            'tool_format': 'qwen_3_xml'})

    def test_byte_split_emoji_and_xml_arguments_survive_native_decode(self):
        filename = '🙂-👩🏽‍💻-🇬🇧-❤️-1️⃣-中文.py'
        raw = ('<tool_call><function=read><parameter=filePath>' + filename
               + '</parameter></function></tool_call>')
        tools = [{'type': 'function', 'function': {'name': 'read', 'parameters': {
            'type': 'object', 'required': ['filePath'],
            'properties': {'filePath': {'type': 'string'}}}}}]
        result = self.invoke([self.model], [{'id': 'byte-vocabulary',
            'tokens': list(raw.encode('utf-8')), 'enable_thinking': False, 'tools': tools}])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [{
            'id': 'byte-vocabulary', 'tool_format': 'qwen_3_xml', 'raw': raw,
            'reasoning': '', 'native_content': raw, 'content': '',
            'calls': [{'name': 'read', 'arguments': {'filePath': filename}}]}])

    def test_invalid_command_and_empty_evidence_are_errors(self):
        for arguments, value, diagnostic in (
                ([], None, 'Expected the served GGUF metadata path'),
                ([self.model], [], 'Expected a nonempty native response array')):
            with self.subTest(arguments=arguments):
                result = self.invoke(arguments, value)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(diagnostic, result.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decoder', type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    DECODER = arguments.decoder.resolve(strict=True)
    unittest.main(argv=[__file__, *remaining])
