#!/usr/bin/env python3
# @file arm_envelope_checks.py
# @brief Run compile-only ARM envelopes; preserve every stage even after failure.
"""Probe evidence is not production-object qualification. Never run ISA binaries."""
import json
import hashlib
import time
import os
import re
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys


def source_identity(root):
    """Authenticate archived bytes against the externally supplied manifest digest."""
    root = Path(root)
    payload = (root / 'arm-envelope-manifest.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != os.environ['ARM_ENVELOPE_MANIFEST_SHA256']:
        raise ValueError('manifest digest mismatch')
    manifest = json.loads(payload)
    for field in ('commit', 'tree'):
        if not re.fullmatch('[0-9a-f]{40}', manifest[field]):
            raise ValueError('malformed ' + field)
    inventory = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if inventory != set(manifest['files']) | {'arm-envelope-manifest.json', 'arm-envelope.Dockerfile'}:
        raise ValueError('source inventory mismatch')
    for name, digest in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('source path escapes archive')
        actual = {'symlink': os.readlink(path)} if path.is_symlink() else {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if actual != digest:
            raise ValueError('source mismatch: ' + name)
    return manifest


def identity_bound(compiler_bound, source_bound):
    """Admit an envelope only when both independent identities are authenticated."""
    return compiler_bound and source_bound


def run(root, output, execute=subprocess.run, *, test_identity=None):
    """Run all owned CMake stages and retain commands, status and complete output."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='attempt-', dir=output))
    tools = {name: shutil.which(value) or value for name, value in
             dict(ninja='ninja', cc=os.environ.get('CC', 'cc'),
                  cxx=os.environ.get('CXX', 'c++')).items()}
    scratch = tempfile.TemporaryDirectory(prefix='arm-envelopes-')
    identity = {'tools': tools, 'source': str(Path(root).resolve()), 'versions': {}}
    for name, command in [(n, [p, '--version']) for n, p in tools.items()]:
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            identity['versions'][name] = dict(command=command, returncode=result.returncode, output=result.stdout + result.stderr)
        except OSError as error:
            identity['versions'][name] = dict(command=command, returncode=127, error=str(error))
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    identity['sha256'] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in tools.items() if Path(path).is_file()}
    compiler_bound = all(Path(tools[name]).is_file() and Path('/usr/bin/' + expected).is_file()
                         and Path(tools[name]).resolve() == Path('/usr/bin/' + expected).resolve()
                         for name, expected in [('cc', 'gcc-14'), ('cxx', 'g++-14')])
    try:
        identity['manifest'] = source_identity(root)
        source_bound = True
    except (OSError, ValueError, KeyError) as error:
        identity['source_error'] = str(error)
        source_bound = False
    discovery = Path(scratch.name) / 'discovery'
    discovery.mkdir()
    (discovery / 'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 4.0)\nproject(Identity LANGUAGES C CXX)\n')
    if test_identity is None:
        with (output / 'compiler-discovery.log').open('w') as log:
            try:
                result = execute(['cmake', '-S', str(discovery), '-B', str(discovery / 'build'), '-G', 'Ninja', f'-DCMAKE_MAKE_PROGRAM:FILEPATH={tools["ninja"]}'], stdout=log, stderr=subprocess.STDOUT)
                cache = (discovery / 'build/CMakeCache.txt').read_text()
                for name, variable in [('cc', 'C'), ('cxx', 'CXX')]:
                    observed = next(line.split('=', 1)[1] for line in cache.splitlines() if line.startswith(f'CMAKE_{variable}_COMPILER:FILEPATH='))
                    compiler_bound = compiler_bound and Path(observed).resolve() == Path(tools[name]).resolve()
                    tools[name] = observed
                compiler_bound = compiler_bound and result.returncode == 0
                identity['compiler_cache'] = cache
            except (OSError, StopIteration) as error:
                compiler_bound = False
                identity['compiler_error'] = str(error)
    identity['source_bound'] = source_bound
    identity['tools'] = tools.copy()
    identity['compiler_bound'] = compiler_bound
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    records = []
    for isa in ('NEON_DOTPROD', 'NEON_I8MM'):
        for config in ('Release', 'Integration'):
            build = Path(scratch.name) / f'{isa}-{config}'
            failed = not identity_bound(compiler_bound, source_bound) if test_identity is None else not test_identity
            stages = [
                ('configure', ['cmake', '-S', str(Path(root) / 'tests/v2/portability'), '-B', str(build), '-G', 'Ninja', f'-DLLAMINAR_CPU_ISA={isa}', f'-DCMAKE_BUILD_TYPE={config}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', f'-DCMAKE_MAKE_PROGRAM:FILEPATH={tools["ninja"]}', f'-DCMAKE_C_COMPILER={tools["cc"]}', f'-DCMAKE_CXX_COMPILER={tools["cxx"]}']),
                ('build', ['cmake', '--build', str(build), '--parallel', '--target', 'v2_cpu_native_tile_register_proof']),
                ('verify', ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error', '-R', '^V2_Integration_CPUNativeNEONTileRegisters$'])]
            for stage, command in stages:
                started = time.monotonic()
                skipped = failed
                with (output / f'{isa}-{config}-{stage}.log').open('w') as log:
                    try:
                        if skipped:
                            log.write('skipped-after-failure\n')
                            code = 1
                        else:
                            result = execute(command, stdout=log, stderr=subprocess.STDOUT)
                            code = result.returncode
                    except OSError as error:
                        log.write(f'{type(error).__name__}: {error}\n')
                        code = 127
                if build.exists():
                    snapshot = output / f'{isa}-{config}-{stage}-evidence'
                    snapshot.mkdir()
                    for source in build.rglob('*'):
                        if source.is_file() and source.suffix in ('.json', '.csv', '.log', '.txt'):
                            destination = snapshot / source.relative_to(build)
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source, destination)
                failed = failed or bool(code)
                records.append(dict(isa=isa, configuration=config, stage=stage, commit=identity.get('manifest', {}).get('commit'), tree=identity.get('manifest', {}).get('tree'), command=command, returncode=code, classification='probe-envelope(non-canonical-for-productionGate7)', status='skipped-after-failure' if skipped else 'executed', seconds=time.monotonic()-started))
                (output / 'stages.json').write_text(json.dumps(records, indent=2))
    scratch.cleanup()
    return int(any(r['returncode'] for r in records))


if __name__ == '__main__':
    sys.exit(run(sys.argv[1], sys.argv[2]))
