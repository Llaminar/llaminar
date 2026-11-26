#!/usr/bin/env python3
# @file arm_envelope_checks.py
# @brief Run compile-only ARM envelopes; preserve every stage even after failure.
"""Probe evidence is not production-object qualification. Never run ISA binaries."""
import json
from pathlib import Path
import subprocess
import sys


def run(root, output, execute=subprocess.run):
    """Run all owned CMake stages and retain commands, status and complete output."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for isa in ('NEON_DOTPROD', 'NEON_I8MM'):
        for config in ('Release', 'Integration'):
            build = output / f'{isa}-{config}'
            stages = [
                ('configure', ['cmake', '-S', str(Path(root) / 'tests/v2/portability'), '-B', str(build), '-G', 'Ninja', f'-DLLAMINAR_CPU_ISA={isa}', f'-DCMAKE_BUILD_TYPE={config}', '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON']),
                ('build', ['cmake', '--build', str(build), '--parallel', '--target', 'v2_cpu_native_tile_register_proof']),
                ('verify', ['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error', '-R', '^V2_Integration_CPUNativeNEONTileRegisters$'])]
            for stage, command in stages:
                with (output / f'{isa}-{config}-{stage}.log').open('w') as log:
                    result = execute(command, stdout=log, stderr=subprocess.STDOUT)
                records.append(dict(isa=isa, configuration=config, stage=stage, command=command, returncode=result.returncode))
                (output / 'stages.json').write_text(json.dumps(records, indent=2))
    return int(any(r['returncode'] for r in records))


if __name__ == '__main__':
    sys.exit(run(sys.argv[1], sys.argv[2]))
