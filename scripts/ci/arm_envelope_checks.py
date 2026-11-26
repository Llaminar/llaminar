#!/usr/bin/env python3
# @file arm_envelope_checks.py
# @brief Run compile-only ARM envelopes; preserve every stage even after failure.
"""Probe evidence is not production-object qualification. Never run ISA binaries."""
import json
import os
import shutil
import tempfile
from pathlib import Path
import subprocess
import sys


def run(root, output, execute=subprocess.run):
    """Run all owned CMake stages and retain commands, status and complete output."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='attempt-', dir=output))
    tools = {name: shutil.which(value) or value for name, value in
             dict(ninja='ninja', cc=os.environ.get('CC', 'gcc-14'),
                  cxx=os.environ.get('CXX', 'g++-14')).items()}
    identity = {'tools': tools, 'source': str(Path(root).resolve()), 'versions': {}}
    for name, command in [('source', ['git', '-C', str(root), 'rev-parse', 'HEAD', 'HEAD^{tree}'])] + [(n, [p, '--version']) for n, p in tools.items()]:
        try:
            result = subprocess.run(command, capture_output=True, text=True)
            identity['versions'][name] = dict(command=command, returncode=result.returncode, output=result.stdout + result.stderr)
        except OSError as error:
            identity['versions'][name] = dict(command=command, returncode=127, error=str(error))
    (output / 'identity.json').write_text(json.dumps(identity, indent=2))
    records = []
    for isa in ('NEON_DOTPROD', 'NEON_I8MM'):
        for config in ('Release', 'Integration'):
            build = output / f'{isa}-{config}'
            stages = [
                ('configure', ['cmake', '-S', str(Path(root) / 'tests/v2/portability'), '-B', str(build), '-G', 'Ninja', f'-DLLAMINAR_CPU_ISA={isa}', f'-DCMAKE_BUILD_TYPE={config}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', f'-DCMAKE_MAKE_PROGRAM:FILEPATH={tools["ninja"]}', f'-DCMAKE_C_COMPILER={tools["cc"]}', f'-DCMAKE_CXX_COMPILER={tools["cxx"]}']),
                ('build', ['cmake', '--build', str(build), '--parallel', '--target', 'v2_cpu_native_tile_register_proof']),
                ('verify', ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error', '-R', '^V2_Integration_CPUNativeNEONTileRegisters$'])]
            for stage, command in stages:
                with (output / f'{isa}-{config}-{stage}.log').open('w') as log:
                    try:
                        result = execute(command, stdout=log, stderr=subprocess.STDOUT)
                        code = result.returncode
                    except OSError as error:
                        log.write(f'{type(error).__name__}: {error}\n')
                        code = 127
                if stage == 'build' and build.exists():
                    snapshot = output / f'{isa}-{config}-build-evidence'
                    snapshot.mkdir()
                    for name in ('cpu_native_tile_registers', 'CMakeCache.txt', 'compile_commands.json'):
                        source = build / name
                        if source.is_dir():
                            shutil.copytree(source, snapshot / name)
                        elif source.exists():
                            shutil.copy2(source, snapshot / name)
                records.append(dict(isa=isa, configuration=config, stage=stage, command=command, returncode=code))
                (output / 'stages.json').write_text(json.dumps(records, indent=2))
    return int(any(r['returncode'] for r in records))


if __name__ == '__main__':
    sys.exit(run(sys.argv[1], sys.argv[2]))
