#!/usr/bin/env python3
# @file arm_native_build_evidence.py
# @brief Preserve the original Integration attempt before enforcing its exit status.
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('native-build-evidence')
KEYS = ('LLAMINAR_SOURCE_TREE', 'LLAMINAR_TOOLCHAIN_CONTEXT', 'LLAMINAR_CPU_ISA',
        'LLAMINAR_ENABLE_CUDA', 'LLAMINAR_ENABLE_ROCM', 'LLAMINAR_SKIP_INTEGRATION',
        'LLAMINAR_TEST_RUNNER_INVENTORY')


def binding():
    """Return the exact declared build policy consumed by this operation."""
    return {key: os.environ.get(key, '') for key in KEYS}


def capture(command):
    """Capture one original operation, retaining failure rather than rerunning it."""
    shutil.rmtree(ROOT, ignore_errors=True)
    ROOT.mkdir()
    with (ROOT / 'operation.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    print((ROOT / 'operation.log').read_text(), end='')
    proof = Path('build_v2_integration/cpu_native_tile_registers')
    if proof.exists():
        for source in proof.iterdir():
            if source.suffix in ('.json', '.log', '.csv'):
                shutil.copy2(source, ROOT / source.name)
    for name in ('CMakeCache.txt', 'compile_commands.json'):
        source = Path('build_v2_integration') / name
        if source.exists():
            shutil.copy2(source, ROOT / name)
    compiler = shutil.which(os.environ.get('CXX', 'c++'))
    if compiler:
        (ROOT / 'compiler.json').write_text(json.dumps(dict(path=compiler, realpath=str(Path(compiler).resolve()), sha256=hashlib.sha256(Path(compiler).read_bytes()).hexdigest(), version=subprocess.check_output([compiler, '--version'], text=True))))
    required = {'driver.json', 'stack-gate.json', 'fused-stage.json'}
    missing = sorted(required - {p.name for p in ROOT.iterdir()})
    status = dict(schema=1, missing_stage_evidence=missing, command=command, binding=binding(), returncode=result.returncode,
                  skipped=os.environ.get('LLAMINAR_SKIP_INTEGRATION') == '1',
                  files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir()})
    (ROOT / 'status.json').write_text(json.dumps(status, indent=2))


def enforce():
    """Fail closed before any usable builder or image can descend from this stage."""
    status = json.loads((ROOT / 'status.json').read_text())
    assert status['schema'] == 1 and status['binding'] == binding(), 'stale build status'
    for name, digest in status['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, 'changed evidence'
    assert 'operation.log' in status['files'], 'incomplete build status'
    print((ROOT / 'operation.log').read_text(), end='')
    assert not (status['skipped'] and os.environ.get('LLAMINAR_TEST_RUNNER_INVENTORY') == 'arm64-pr'), 'required Integration skipped'
    if status['returncode']:
        raise SystemExit(status['returncode'])
    if os.environ.get('LLAMINAR_TEST_RUNNER_INVENTORY') == 'arm64-pr':
        assert not status['missing_stage_evidence'], 'incomplete native proof evidence'


if __name__ == '__main__':
    if sys.argv[1] == 'capture':
        capture(sys.argv[3:])
    else:
        enforce()
