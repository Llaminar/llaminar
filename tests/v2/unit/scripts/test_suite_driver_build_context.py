#!/usr/bin/env python3
"""Authenticate local Docker COPY inputs against the actual suite-driver context.

The published-suite launcher supplies the command under test. Its exact Dockerfile
must be able to copy the pinned OpenCode installer from that context, independent
of the caller's working directory. No Docker daemon, image download or GPU is used.
"""
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import run_published_image_suite as suite


class SuiteDriverBuildContextTests(unittest.TestCase):
    """Prove the real launch context contains every declared local COPY input."""

    def test_exact_build_command_can_copy_its_pinned_installer(self):
        """A repository-relative COPY must fail when the launcher selects scripts/ci."""
        pair = {'images': {'AVX2': {'id': 'sha256:' + 'a' * 64,
            'registry_ref': 'ghcr.io/llaminar/llaminar@sha256:' + 'b' * 64}}}
        checked = []

        def build(command, log):
            """Resolve actual Docker build inputs without running a container builder."""
            context = Path(command[-1]).resolve(strict=True)
            dockerfile = Path(command[command.index('--file') + 1])
            self.assertTrue(context.is_dir())
            for line in dockerfile.read_text().splitlines():
                if not line.lstrip().startswith('COPY '):
                    continue
                fields = shlex.split(line, comments=True)
                self.assertEqual(len(fields), 3, 'Extend the build-input parser for a changed COPY contract')
                source = context / fields[1]
                self.assertTrue(source.is_file(), 'Docker COPY input is absent from its build context: ' + str(source))
                self.assertTrue(source.resolve().is_relative_to(context))
                checked.append(source.resolve())

        image = {'id': 'sha256:' + 'c' * 64, 'labels': {'org.llaminar.image_role': 'suite-driver'}}
        with tempfile.TemporaryDirectory() as temporary, patch.object(suite.pipeline, 'run', side_effect=build), \
                patch.object(suite, 'image_identity', return_value=image):
            self.assertEqual(suite.build_driver(pair, Path(temporary)), image['id'])
        self.assertEqual(checked, [(suite.ROOT / 'scripts/ci/install_opencode_client.py').resolve()])


if __name__ == '__main__':
    unittest.main()
