#!/usr/bin/env python3
"""Device-free deferred-SDMA packaging, proof admission and receipt controls.

Actual original/candidate native callback bodies run only in the pinned runtime
builder. Installed preflight authenticates their source/DSO-bound observation;
it never compiles, loads ROCr, or touches an accelerator. Builder packaging
replays the actual early COPY closure against the installer's literal inputs.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import re
import shlex
from pathlib import Path
import sys
import unittest
from test_rocr_helper_numa import verify_binary

ROOT = Path(__file__).resolve().parents[4]


def proof_owner(root: Path):
    """Read the canonical producer without entering its native dependency."""
    path = root / 'scripts/docker/rocr-deferred-sdma/proof.py'
    spec = importlib.util.spec_from_file_location('rocr_deferred_sdma_contract', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def missing_builder_inputs(dockerfile: str) -> set[str]:
    """Project literal installer inputs through the early builder COPY closure.

    Only shell-form COPY instructions feeding /tmp/install-scripts are relevant.
    Stop at the first RUN: these inputs must exist before dependency setup starts.
    Resolve directory copies to real repository files, preserving destination names.
    """
    installed = set()
    logical = dockerfile.replace('\\\n', ' ')
    for line in logical.splitlines():
        if line.startswith('RUN '):
            break
        if not line.startswith('COPY '):
            continue
        *sources, destination = shlex.split(line)[1:]
        if not destination.startswith('/tmp/install-scripts/'):
            continue
        for source in sources:
            path = ROOT / source
            if path.is_dir():
                for child in path.rglob('*'):
                    if child.is_file():
                        installed.add(str(Path(destination) / child.relative_to(path)))
            elif path.is_file():
                installed.add(str(Path(destination) / path.name)
                              if destination.endswith('/') else destination)
    installer = (ROOT / 'scripts/docker/install-rocr-helper-numa.sh').read_text()
    required = set(re.findall(r'\$\{rocr_script_dir\}/([^"\s]+)', installer))
    return {name for name in required if '/tmp/install-scripts/' + name not in installed}


class ROCrBuilderCopyClosure(unittest.TestCase):
    """Replay the missing-asset packaging defect without running a container."""

    def test_actual_installer_literal_inputs_are_copied_before_setup(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        self.assertEqual(missing_builder_inputs(dockerfile), set())

    def test_original_missing_deferred_sdma_copies_fail_closed(self):
        dockerfile = (ROOT / 'Dockerfile').read_text()
        logical = dockerfile.replace('\\\n', ' ')
        original = '\n'.join(line for line in logical.splitlines()
                             if not (line.startswith('COPY ') and
                                     ('rocm-rocr-deferred-sdma-profiling.patch' in line or
                                      'scripts/docker/rocr-deferred-sdma/' in line)))
        self.assertEqual(missing_builder_inputs(original), {
            'patches/rocm-rocr-deferred-sdma-profiling.patch',
            'rocr-deferred-sdma/proof.py',
            'rocr-deferred-sdma/contract.cpp.in',
        })


class ROCrDeferredSDMAContract(unittest.TestCase):
    """Reject missing/asymmetric transitions and lost source/signal lifetimes."""
    def document(self):
        owner = proof_owner(ROOT)
        document = dict(schema='rocr_deferred_sdma_profiling_v1', source_identity='authored-source',
                        library_sha256='c' * 64, contract_identity=owner.contract_identity())
        for variant in ('original', 'candidate'):
            times = (40, 0, 40, 0, 40, 0, 0) if variant == 'candidate' else (0, 40, 40, 0, 40, 0, 0)
            cases = [dict(name=name, passed=variant == 'candidate' or index >= 2,
                          timestamp_packets=times[index], callbacks=(20, 20, 40, 40, 0, 0, 0)[index],
                          live_bytes=160 if index < 6 else 0, packet_unchanged=True, global_unchanged=True,
                          live_callback_allocations=0) for index, name in enumerate(owner.NAMES)]
            document[variant] = dict(source_sha256=('a' if variant == 'original' else 'b') * 64,
                                     header_sha256=('d' if variant == 'original' else 'e') * 64,
                                     compile_exit=0, run_exit=1 if variant == 'original' else 0, cases=cases)
        return owner, document

    def test_complete_original_candidate_packet_and_lifetime_evidence(self):
        owner, document = self.document()
        owner.verify(document, 'authored-source', 'c' * 64)

    def test_rejects_on_off_and_off_on_rereads_independently(self):
        owner, original = self.document()
        for index, count in ((0, 0), (1, 40)):
            with self.subTest(transition=index):
                document = copy.deepcopy(original)
                document['candidate']['cases'][index]['timestamp_packets'] = count
                with self.assertRaises(ValueError):
                    owner.verify(document, 'authored-source', 'c' * 64)

    def test_rejects_weakened_or_incomplete_native_lifetime_and_controls(self):
        owner, original = self.document()
        for mutate in (
            lambda d: d['candidate']['cases'].pop(),
            lambda d: d['original']['cases'][0].update(passed=True),
            lambda d: d['original'].update(run_exit=0),
            lambda d: d['candidate']['cases'][0].update(packet_unchanged=False),
            lambda d: d['candidate']['cases'][1].update(global_unchanged=False),
            lambda d: d['candidate']['cases'][2].update(callbacks=20),
            lambda d: d['candidate']['cases'][3].update(live_callback_allocations=1),
            lambda d: d['candidate']['cases'][4].update(live_bytes=320),
            lambda d: d['candidate'].update(source_sha256=d['original']['source_sha256']),
            lambda d: d.update(contract_identity='foreign'),
            lambda d: d.update(library_sha256='foreign'),
        ):
            document = copy.deepcopy(original)
            mutate(document)
            with self.assertRaises(ValueError):
                owner.verify(document, 'authored-source', 'c' * 64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--binary', type=Path, action='append')
    args, remaining = parser.parse_known_args()
    if args.binary:
        for binary in args.binary:
            verify_binary(binary, args.source_root)
        print('PASS installed ROCr deferred SDMA source/proof/loader closure')
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
