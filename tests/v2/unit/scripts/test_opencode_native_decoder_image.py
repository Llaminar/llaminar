#!/usr/bin/env python3
"""Authenticate native audit invocation without running Docker or model code.

The immutable runtime supplies both the decoder and its loader closure. These
tests prove stdin, complete shard-directory mounts and image identity survive
argument construction, and that local diagnostics remain an explicit mode.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
sys.path.insert(0, str(ROOT / 'tests/v2/e2e/server'))
import docker_paths
from opencode_native_audit import decoder_command, decode_records
import opencode_cell_runtime as runtime


class NativeDecoderImageTests(unittest.TestCase):
    """Use exact argument vectors so shell quoting cannot alter paths or input."""

    def test_image_mode_pins_runtime_and_keeps_stdin_and_complete_model_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / 'model with spaces-00001-of-00002.gguf'
            model.touch()
            (model.parent / 'model with spaces-00002-of-00002.gguf').touch()
            immutable = 'sha256:' + 'a' * 64
            with patch.object(docker_paths, 'mounts', return_value=['--mount', 'fixture-readonly']) as mount:
                command = decoder_command(model, image=immutable)
            mount.assert_called_once_with([(model.parent, '/model', True)])
            self.assertIn('--interactive', command)
            self.assertNotIn('--rm', command)
            self.assertNotIn('docker', command)
            self.assertEqual(command[-2:], [immutable, '/model/' + model.name])
            self.assertEqual(command[command.index('--entrypoint') + 1],
                             '/usr/local/bin/llaminar_native_tool_evidence_decoder')
            self.assertEqual(command[command.index('--network') + 1], 'none')
            self.assertNotIn('--gpus', command)
            self.assertNotIn('--device', command)

    def test_local_decoder_is_explicit_and_keeps_exact_paths(self):
        self.assertEqual(decoder_command(Path('/model/$a.gguf'), decoder=Path('/build/helper')),
                         ['/build/helper', '/model/$a.gguf'])

    def test_ambiguous_missing_and_mutable_decoder_identity_fail_before_mounts(self):
        for options in ({}, {'decoder': Path('/a'), 'image': 'sha256:' + 'a' * 64},
                        {'image': 'ghcr.io/llaminar/llaminar:develop'},
                        {'image': 'sha256:' + 'a' * 63}, {'image': False}):
            with self.subTest(options=options), patch.object(docker_paths, 'mounts') as mount:
                with self.assertRaises(ValueError):
                    decoder_command(Path('/must-not-open.gguf'), **options)
                mount.assert_not_called()

    def test_image_mode_rejects_missing_model_or_directory_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            for model in (Path(temporary), Path(temporary) / 'missing.gguf'):
                with self.subTest(model=model), patch.object(docker_paths, 'mounts') as mount:
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        decoder_command(model, image='sha256:' + 'a' * 64)
                    mount.assert_not_called()

    def test_image_decoder_retains_completed_native_logs_and_retirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            model = output / 'model.gguf'
            model.touch()
            records = [{'id': 'response', 'tokens': [1, 2, 3]}]
            native = [{'id': 'response', 'native_content': 'hello'}]
            with patch.object(docker_paths, 'mounts', return_value=[]), \
                 patch.object(runtime, 'run_retired', return_value=(
                     subprocess.CompletedProcess([], 0, json.dumps(native), 'metadata only\n'), {})) as run:
                decode_records(model, records, output, image='sha256:' + 'a' * 64)
            self.assertEqual(json.loads((output / 'decoded.json').read_text()), native)
            self.assertEqual((output / 'decoder.stderr').read_text(), 'metadata only\n')
            self.assertTrue(json.loads((output / 'decoder-retirement.json').read_text())['passed'])
            self.assertEqual(json.loads(run.call_args.kwargs['input_text']), records)


if __name__ == '__main__':
    unittest.main()
