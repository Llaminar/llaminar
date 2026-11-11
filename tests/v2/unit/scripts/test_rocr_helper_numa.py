#!/usr/bin/env python3
"""Device-free ROCr helper proof/loader admission and authored negative controls.

Installed admission traces the existing binary before main in both inherited
and clean environments; no ROCr API, accelerator or compiler is invoked.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_openmp_runtime_binding import parse_soname, trace_loader
from test_hip_runtime_binding import HIPRuntimeAuthority, parse_hip_runtime, require_installed_receipt

ROOT = Path(__file__).resolve().parents[4]


def proof_owner(root: Path):
    """Read the canonical dependency producer without importing a native runtime."""
    path = root / 'scripts/docker/rocr-helper-numa/proof.py'
    spec = importlib.util.spec_from_file_location('rocr_helper_proof', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def authority(root: Path) -> tuple[object, str, str]:
    """Use the same explicit pins and ordered proof inputs as the installer."""
    owner = proof_owner(root)
    docker = root / 'scripts/docker'
    pins = dict(line.split('=', 1) for line in (docker / 'rocm-release.env').read_text().splitlines()
                if line and not line.startswith('#'))
    identity = ':'.join(pins[key] for key in ('LLAMINAR_ROCM_VERSION', 'LLAMINAR_ROCM_DEB_VERSION',
                       'LLAMINAR_ROCM_SYSTEMS_REVISION', 'LLAMINAR_ROCR_LIBRARY_VERSION'))
    for relative in ('patches/rocm-rocr-helper-numa.patch', 'rocr-helper-numa/proof.py',
                     'rocr-helper-numa/contract.cpp.in', 'patches/rocm-rocr-deferred-sdma-profiling.patch',
                     'rocr-deferred-sdma/proof.py', 'rocr-deferred-sdma/contract.cpp.in'):
        identity += ':' + owner.digest(docker / relative)
    return owner, identity, pins['LLAMINAR_ROCR_LIBRARY_VERSION']


def require_installed(runtime: Path, root: Path) -> None:
    """Require one canonical SDK-relative artifact and its source/DSO-bound proof."""
    owner, identity, version = authority(root)
    physical = runtime.resolve(strict=True)
    prefix = physical.parent.parent
    library = prefix / 'lib' / ('libhsa-runtime64.so.' + version)
    if physical != library or library.is_symlink():
        raise ValueError('ROCr loader selected a foreign artifact')
    for name in ('libhsa-runtime64.so', 'libhsa-runtime64.so.1'):
        if (prefix / 'lib' / name).resolve(strict=True) != library:
            raise ValueError('ROCr SDK aliases disagree')
    marker = (prefix / 'share/llaminar/rocr-helper-numa.txt').read_text().strip()
    if marker != identity + ':' + owner.digest(library):
        raise ValueError('ROCr SDK receipt has stale dependency bytes')
    owner.load_verified(prefix / 'share/llaminar/rocr-helper-numa-proof.json', identity, library)
    sdma_path = root / 'scripts/docker/rocr-deferred-sdma/proof.py'
    spec = importlib.util.spec_from_file_location('rocr_deferred_sdma_proof', sdma_path)
    sdma_owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdma_owner)
    sdma_owner.load_verified(prefix / 'share/llaminar/rocr-deferred-sdma-proof.json', identity, library)
    if (prefix / 'share/licenses/llaminar-rocr/rocm-rocr-deferred-sdma-profiling.patch').read_bytes() != (
            root / 'scripts/docker/patches/rocm-rocr-deferred-sdma-profiling.patch').read_bytes():
        raise ValueError('ROCr installed deferred SDMA patch differs from source authority')
    installed_patch = prefix / 'share/licenses/llaminar-rocr/rocm-rocr-helper-numa.patch'
    if installed_patch.read_bytes() != (root / 'scripts/docker/patches/rocm-rocr-helper-numa.patch').read_bytes():
        raise ValueError('ROCr installed patch differs from source authority')


def verify_binary(binary: Path, root: Path) -> None:
    """Authenticate both loader environments before main, with no runtime calls."""
    readelf = shutil.which('readelf')
    if not readelf:
        raise ValueError('ROCr dependency admission requires readelf')
    for clean in (False, True):
        environment = dict(os.environ)
        if clean:
            for name in ('LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT'):
                environment.pop(name, None)
        trace = trace_loader(binary, environment)
        if re.search(r'=>\s+not found(?:\s|$)', trace):
            raise ValueError('ROCr loader has unresolved dependencies')
        rows = [line.split() for line in trace.splitlines()
                if line.split() and line.split()[0].startswith('libhsa-runtime64.so')]
        if len(rows) != 1 or rows[0][0] != 'libhsa-runtime64.so.1' or len(rows[0]) < 3 or rows[0][1] != '=>':
            raise ValueError('ROCr loader requires one canonical runtime ABI')
        runtime = Path(rows[0][2])
        if not runtime.is_absolute():
            raise ValueError('ROCr loader omitted an absolute artifact')
        result = subprocess.run([readelf, '-d', str(runtime)], check=True, capture_output=True, text=True, timeout=30)
        if parse_soname(result.stdout) != 'libhsa-runtime64.so.1':
            raise ValueError('ROCr selected artifact has a foreign SONAME')
        require_installed(runtime, root)
        # Authenticate both members of this SAME loader closure before requiring
        # their physical SDK root to agree. Independent valid copies elsewhere
        # must not turn a mixed-SDK process into an admitted installation.
        hip_authority = HIPRuntimeAuthority.from_source(root)
        hip_runtime = parse_hip_runtime(trace, hip_authority.soname)
        require_installed_receipt(hip_runtime, hip_authority)
        if hip_runtime.resolve(strict=True).parent != runtime.resolve(strict=True).parent:
            raise ValueError('HIP and ROCr loader artifacts belong to different SDK roots')


class ROCrHelperNUMAContract(unittest.TestCase):
    """Authored immutable-proof negative controls; these are not native replay."""
    def document(self):
        owner = proof_owner(ROOT)
        document = dict(schema='rocr_helper_numa_v1', source_identity='authored-source',
                        library_sha256='c' * 64, contract_identity=owner.contract_identity())
        for variant in ('original', 'candidate'):
            branches = []
            for attr in (False, True):
                cases = []
                masks = (3, 243, 2, 255, 255, 255, 240) if variant == 'candidate' else (255, 255, 2, 255, 255, 255, 255)
                for index, name in enumerate(owner.NAMES):
                    passed = variant == 'candidate' or index in (2, 3, 4, 5)
                    cases.append(dict(name=name, passed=passed, helper_mask=masks[index],
                                      creator_mask=(2, 66, 2, 255, 2, 2, 64)[index], creator_unchanged=True))
                branches.append(dict(attr_affinity=attr, compile_exit=0, run_exit=0 if variant == 'candidate' else 1, cases=cases))
            document[variant] = dict(source_sha256=('a' if variant == 'original' else 'b') * 64, branches=branches)
        return owner, document

    def test_complete_authored_proof(self):
        owner, document = self.document()
        owner.verify(document, 'authored-source', 'c' * 64)

    def test_loader_requires_hip_and_rocr_from_same_authenticated_sdk(self):
        """Valid individual dependency receipts cannot certify mixed SDK roots."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first, second = directory / 'sdk-a/lib', directory / 'sdk-b/lib'
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            hip = first / 'libamdhip64.so.7.16.26385'
            hip.write_bytes(b'authored HIP artifact metadata fixture')
            for same in (True, False):
                with self.subTest(same_sdk=same):
                    hsa = (first if same else second) / 'libhsa-runtime64.so.1.21.0'
                    hsa.write_bytes(b'authored ROCr artifact metadata fixture')
                    trace = ('libamdhip64.so.7 => ' + str(hip) + ' (0x1)\n'
                             'libhsa-runtime64.so.1 => ' + str(hsa) + ' (0x2)\n')
                    with patch(__name__ + '.trace_loader', return_value=trace), \
                            patch(__name__ + '.shutil.which', return_value='/authored/readelf'), \
                            patch(__name__ + '.subprocess.run', return_value=subprocess.CompletedProcess([], 0, '', '')), \
                            patch(__name__ + '.parse_soname', return_value='libhsa-runtime64.so.1'), \
                            patch(__name__ + '.require_installed') as hsa_admission, \
                            patch(__name__ + '.require_installed_receipt') as hip_admission, \
                            patch.object(HIPRuntimeAuthority, 'from_source',
                                         return_value=SimpleNamespace(soname='libamdhip64.so.7')):
                        if same:
                            verify_binary(Path('/authored/not-executed'), ROOT)
                            self.assertEqual(hsa_admission.call_count, 2)
                            self.assertEqual(hip_admission.call_count, 2)
                        else:
                            with self.assertRaisesRegex(ValueError, 'different SDK roots'):
                                verify_binary(Path('/authored/not-executed'), ROOT)
                            hsa_admission.assert_called_once_with(hsa, ROOT)
                            self.assertEqual(hip_admission.call_count, 1)

    def test_rejects_missing_branch_creator_change_and_relabelled_control(self):
        owner, original = self.document()
        for mutate in (lambda d: d['candidate']['branches'].pop(),
                       lambda d: d['candidate']['branches'][0]['cases'][0].update(creator_unchanged=False),
                       lambda d: d['original']['branches'][0]['cases'][0].update(passed=True),
                       lambda d: d['candidate']['branches'][1]['cases'][0].update(helper_mask=255),
                       lambda d: d.update(contract_identity='foreign'),
                       lambda d: d.update(library_sha256='foreign')):
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
        print('PASS installed ROCr helper-NUMA source/proof/loader closure')
    else:
        unittest.main(argv=[sys.argv[0], *remaining])
