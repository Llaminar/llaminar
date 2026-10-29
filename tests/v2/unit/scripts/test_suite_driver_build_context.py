#!/usr/bin/env python3
"""Authenticate exact tools build context and runtime inheritance on both callers.

Replay the actual launcher and Dockerfile without a Docker daemon or hardware.
Published registry digests and unpublished feature runtime tags must resolve to
one admitted config ID, source/ISA labels and base layer prefix. The pinned
installer stays within the repository-root context; inference bytes stay fixed.
"""
from copy import deepcopy
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'scripts/ci'))
import run_published_image_suite as suite


def runtime_identity(*, local: bool) -> dict:
    """Describe full identities with deliberately different config/manifest digests."""
    isa = 'AVX512' if local else 'AVX2'
    runtime = {'id': 'sha256:' + 'a' * 64, 'layers': ['runtime-base', 'runtime-serving'],
        'tag': 'llaminar-ci:source-' + isa.lower() + '-runtime',
        'labels': {'org.llaminar.image_role': 'runtime', 'org.llaminar.cpu_isa': isa,
                   'org.llaminar.source_tree': 'b' * 40, 'org.opencontainers.image.revision': 'c' * 40,
                   'org.llaminar.build_type': 'Release', 'org.llaminar.cuda': 'ON', 'org.llaminar.rocm': 'ON'}}
    if not local:
        runtime['registry_ref'] = 'ghcr.io/llaminar/llaminar@sha256:' + 'd' * 64
    return runtime


def tools_identity(runtime: dict) -> dict:
    """Retain exactly the inherited runtime bytes/labels, plus control-only tools."""
    return {'id': 'sha256:' + 'e' * 64, 'layers': runtime['layers'] + ['tools-install'],
            'labels': {**runtime['labels'], 'org.llaminar.image_role': 'suite-driver'}}


class SuiteDriverBuildContextTests(unittest.TestCase):
    """Exercise real build argv and inherited identities, never a fake build inventory."""

    def test_exact_build_command_can_copy_its_pinned_installer(self):
        """Both paired published and standalone feature callers use real root-context COPY."""
        for local in (False, True):
            with self.subTest(local=local):
                runtime = runtime_identity(local=local)
                reference = runtime.get('registry_ref', runtime['tag'])
                tools = tools_identity(runtime)
                checked = []
                references = []
                def identity(name):
                    if name == reference:
                        references.append(name)
                        return runtime
                    self.assertTrue(name.startswith('llaminar-ci:published-suite-driver-'))
                    return tools
                def build(command, log):
                    context = Path(command[-1]).resolve(strict=True)
                    self.assertEqual(context, suite.ROOT.resolve())
                    self.assertEqual(command[command.index('--builder') + 1], 'default')
                    args = [command[i + 1] for i, arg in enumerate(command) if arg == '--build-arg']
                    self.assertIn('RUNTIME_IMAGE=' + reference, args)
                    self.assertNotIn('RUNTIME_IMAGE=' + runtime['id'], args)
                    dockerfile = Path(command[command.index('--file') + 1])
                    self.assertIn('FROM ${RUNTIME_IMAGE}', dockerfile.read_text())
                    for line in dockerfile.read_text().splitlines():
                        if not line.lstrip().startswith('COPY '):
                            continue
                        fields = shlex.split(line, comments=True)
                        self.assertEqual(len(fields), 3, 'Changed COPY contract requires parser review')
                        source = context / fields[1]
                        self.assertTrue(source.is_file(), 'Missing pinned installer: ' + str(source))
                        self.assertTrue(source.resolve().is_relative_to(context))
                        checked.append(source.resolve())
                with tempfile.TemporaryDirectory() as temporary, \
                     patch.object(suite.pipeline, 'run', side_effect=build), \
                     patch.object(suite, 'image_identity', side_effect=identity):
                    if local:
                        result = suite.build_driver_for_runtime(runtime, Path(temporary))
                    else:
                        result = suite.build_driver({'images': {'AVX2': runtime}}, Path(temporary))
                    self.assertEqual(result, tools['id'])
                self.assertEqual(references, [reference, reference])
                self.assertEqual(checked, [(suite.ROOT / 'scripts/ci/install_opencode_client.py').resolve()])

    def test_foreign_or_changed_from_reference_fails_before_certifying_tools(self):
        runtime = runtime_identity(local=True)
        tools = tools_identity(runtime)
        foreign = {**runtime, 'id': 'sha256:' + 'f' * 64}
        for changed_after in (False, True):
            with self.subTest(changed_after=changed_after), tempfile.TemporaryDirectory() as temporary:
                reference_calls = 0
                def identity(name):
                    nonlocal reference_calls
                    if name == runtime['tag']:
                        reference_calls += 1
                        return foreign if not changed_after or reference_calls == 2 else runtime
                    return tools
                with patch.object(suite, 'image_identity', side_effect=identity), patch.object(suite.pipeline, 'run') as build:
                    with self.assertRaisesRegex(ValueError, 'FROM reference changed'):
                        suite.build_driver_for_runtime(runtime, Path(temporary))
                self.assertEqual(build.call_count, int(changed_after))

    def test_raw_ids_mutable_registry_refs_and_incomplete_runtime_fail_closed(self):
        runtime = runtime_identity(local=True)
        bad = [{**runtime, 'tag': runtime['id']},
               {**runtime, 'registry_ref': 'ghcr.io/llaminar/llaminar:develop'},
               {**runtime, 'labels': {}}, {**runtime, 'layers': []}]
        for value in bad:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary, \
                 patch.object(suite.pipeline, 'run') as build:
                with self.assertRaises(ValueError):
                    suite.build_driver_for_runtime(value, Path(temporary))
                build.assert_not_called()

    def test_tools_must_preserve_source_isa_role_and_runtime_layer_prefix(self):
        runtime = runtime_identity(local=False)
        tools = tools_identity(runtime)
        for mutate in (
                lambda value: value['labels'].update({'org.llaminar.image_role': 'runtime'}),
                lambda value: value['labels'].update({'org.llaminar.cpu_isa': 'AVX512'}),
                lambda value: value['labels'].update({'org.llaminar.source_tree': 'foreign'}),
                lambda value: value.update(layers=['different-runtime', 'tools-install'])):
            changed = deepcopy(tools)
            mutate(changed)
            with self.subTest(changed=changed), tempfile.TemporaryDirectory() as temporary, \
                 patch.object(suite.pipeline, 'run'), \
                 patch.object(suite, 'image_identity', side_effect=[runtime, runtime, changed]):
                with self.assertRaisesRegex(ValueError, 'source/ISA/base binding'):
                    suite.build_driver_for_runtime(runtime, Path(temporary))


if __name__ == '__main__':
    unittest.main()
