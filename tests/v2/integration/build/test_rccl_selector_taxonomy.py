#!/usr/bin/env python3
"""Builder-only real RCCL parser execution; installed source/DSO-bound verification.

Native init/tuning/collectives translation units compile with their own build
flags. A disposable exact reverse patch must fail the same enum-keyed oracle.
No GPU initialization, substituted parser or source-substring proof is used.
"""
import argparse, hashlib, json, shlex, subprocess, tempfile, re, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
PATCH = 'scripts/docker/patches/rccl-collective-selector-taxonomy.patch'
INPUTS = (PATCH,
    'tests/v2/integration/build/rccl_selector_probe.cpp',
    'tests/v2/integration/build/test_rccl_selector_taxonomy.py')
CHECKS = 341
NATIVE = ('src/init.cc', 'src/graph/tuning.cc', 'src/collectives.cc')

def digest(path):
    """Hash only source/compiler/native library identity, never model payloads."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identities():
    """Authenticate every installed proof input including dependency repair."""
    return {p: digest(ROOT / p) for p in INPUTS}

def verify(report, library):
    """Reject missing, stale, crashed or vacuous native/old-code outcomes."""
    if (
        report.get('schema') != 1
        or report.get('sources') != identities()
        or report.get('library_sha256') != digest(library)
    ):
        raise RuntimeError('Selector proof source/runtime identity differs')
    pins = report.get('native_sources', {})
    suffixes = ('src/init.cc',
        'src/graph/tuning.cc',
        'src/collectives.cc',
        'src/include/device.h',
        'src/include/nccl_common.h')
    if (
        not isinstance(pins, dict)
        or len(pins) != 5
        or any((sum((k.endswith('/' + suffix) for k in pins)) != 1 for suffix in suffixes))
        or any((not isinstance(v, str) or not re.fullmatch('[0-9a-f]{64}', v) for v in pins.values()))
    ):
        raise RuntimeError('Missing native source/enum pins')
    receipt = report.get('rccl_receipt')
    compiler = report.get('compiler', {})
    native_inputs = report.get('native_compile_inputs', [])
    if (
        not isinstance(receipt, str)
        or '-selector-' + report['sources'][PATCH] not in receipt
        or not isinstance(report.get('compiler_version'), str)
        or not report['compiler_version']
        or not isinstance(compiler, dict)
        or not Path(compiler.get('path', '')).is_absolute()
        or not isinstance(compiler.get('stat'), list)
        or len(compiler['stat']) != 5
        or any((type(v) is not int for v in compiler['stat']))
        or not isinstance(report.get('architecture'), str)
        or not report['architecture'].startswith('gfx')
        or not Path(report.get('sdk', '')).is_absolute()
    ):
        raise RuntimeError('Missing compiler/architecture/SDK/receipt closure')
    if (
        not isinstance(native_inputs, list)
        or len(native_inputs) != 3
        or any(
            (
                not isinstance(r, dict)
                or not isinstance(r.get('flags'), list)
                or not r['flags']
                or not all(isinstance(x, str) for x in r['flags'])
                or not isinstance(r.get('source'), str)
            )
            for r in native_inputs
        )
        or {r['source'] for r in native_inputs} != {k for k in pins if k.endswith(tuple(('/' + x for x in NATIVE)))}
    ):
        raise RuntimeError('Missing native compiler inputs')
    rows = report.get('observations', [])
    if [x.get('variant') for x in rows] != ['native', 'original']:
        raise RuntimeError('Incomplete selector outcomes')
    for row in rows:
        measured = row.get('measured', {})
        failures = measured.get('failures')
        negative = row['variant'] == 'original'
        if (
            type(measured.get('logging_warning_calls')) is not int
            or measured['logging_warning_calls'] != (4 if negative else 0)
            or type(measured.get('logging_other_calls')) is not int
            or measured['logging_other_calls'] != 0
        ):
            raise RuntimeError('Native parser logging did not use the bound host-only seam')
        if (
            type(measured.get('checks')) is not int
            or measured['checks'] != CHECKS
            or type(failures) is not int
            or not 0 <= failures <= CHECKS
            or type(row.get('returncode')) is not int
            or row['returncode'] != (1 if negative else 0)
            or (failures == 0) == negative
            or (
                not negative
                and row.get('stderr') != ''
            )
        ):
            raise RuntimeError('Vacuous/failed selector oracle')

def inputs(build, suffix):
    """Borrow the actual source and native header/define inputs from compile database."""
    rows = json.loads((build / 'compile_commands.json').read_text())
    matches = [r for r in rows if r['file'].endswith('/hipify/' + suffix)]
    if len(matches) != 1:
        raise RuntimeError('Native translation unit not unique: ' + suffix)
    row = matches[0]
    words = row.get('arguments') or shlex.split(row['command'])
    flags = []
    i = 0
    while i < len(words):
        w = words[i]
        if w in ('-D', '-I', '-isystem', '-iquote', '-include'):
            flags.extend(words[i:i + 2])
            i += 2
        else:
            if w.startswith(('-D', '-I')):
                flags.append(w)
            i += 1
    source = Path(row['file'])
    source = source if source.is_absolute() else Path(row['directory']) / source
    source = source.resolve(strict=True)
    # A relocated negative-control unit retains the original compiler's implicit
    # quoted-header directory, including graph/tuning.cc's sibling topo.h. The
    # private original headers still take precedence in their own source tree.
    flags.extend(['-iquote', str(source.parent)])
    return (source, flags)

def compile_report(a):
    """Compile actual native units and negative variant, execute only CPU parser logic."""
    a.report.unlink(missing_ok=True)
    # Compile each real native translation unit with its own include/define inputs.
    source_flags = [inputs(a.rccl_build, n) for n in NATIVE]
    receipt = (a.rccl_build / '.llaminar-rccl-commit').read_text().strip()
    if '-selector-' + digest(ROOT / PATCH) not in receipt:
        raise RuntimeError('Dependency receipt lacks selector repair digest')
    if digest(a.rccl_build / 'librccl.so') != digest(a.library):
        raise RuntimeError('Selected library is not proof native build')
    report = dict(schema=1,
        sources=identities(),
        library_sha256=digest(a.library),
        rccl_receipt=receipt,
        compiler_version=subprocess.check_output([a.compiler, '--version'], text=True),
        native_sources={str(p): digest(p) for p, f in source_flags},
        observations=[])
    header = source_flags[0][0].parent / 'include/device.h'
    report['native_sources'][str(header)] = digest(header)
    enum_header = header.parent / 'nccl_common.h'
    report['native_sources'][str(enum_header)] = digest(enum_header)
    compiler = Path(a.compiler).resolve(strict=True)
    st = compiler.stat()
    report.update(compiler=dict(path=str(compiler), stat=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]),
        architecture=a.architecture,
        sdk=str(a.sdk.resolve(strict=True)),
        native_compile_inputs=[dict(source=str(source), flags=flags) for source, flags in source_flags])
    with tempfile.TemporaryDirectory(prefix='llaminar-rccl-selector-') as directory:
        tmp = Path(directory)
        old = tmp / 'original'
        old.mkdir()
        preambles = {}
        for rel, (source, _) in zip(NATIVE, source_flags):
            dest = old / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = source.read_bytes()
            preamble = b'#include "hip/hip_runtime.h"\n'
            preambles[rel] = preamble if data.startswith(preamble) else b''
            if preambles[rel]:
                data = data[len(preamble):]
            dest.write_bytes(data)
        (old / 'src/include').mkdir()
        (old / 'src/include/device.h').write_bytes(header.read_bytes())
        subprocess.run(['git', 'apply', '--reverse', str(ROOT / PATCH)], cwd=old, check=True)
        for rel, preamble in preambles.items():
            dest = old / rel
            dest.write_bytes(preamble + dest.read_bytes())
            if (
                rel != 'src/init.cc'
                and dest.read_bytes() != source_flags[NATIVE.index(rel)][0].read_bytes()
            ):
                raise RuntimeError('Unrelated native negative unit changed')
        (old / 'src/include/nccl_common.h').write_bytes(enum_header.read_bytes())
        symbols = tmp / 'probe.map'
        symbols.write_text('{ local: *; };\n')
        for variant in ('native', 'original'):
            common = [a.compiler,
                '-x',
                'hip',
                '--offload-host-only',
                '--offload-arch=' + a.architecture,
                '--hip-path=' + str(a.sdk),
                '--rocm-path=' + str(a.sdk),
                '-std=c++17',
                '-O3',
                '-ffunction-sections',
                '-fdata-sections',
                '-fvisibility=hidden']
            objects = []
            for index, (source, flags) in enumerate(source_flags):
                obj = tmp / (variant + str(index) + '.o')
                actual = source if variant == 'native' else old / NATIVE[index]
                # The original header precedes generated repaired include roots.
                override = ['-I' + str(old / 'src/include')] if variant == 'original' else []
                subprocess.run([*common, *override, *flags, '-c', str(actual), '-o', str(obj)],
                    cwd=a.rccl_build,
                    check=True)
                objects.append(str(obj))
            binary = tmp / (variant + '-probe')
            # The original header precedes generated repaired include roots.
            override = ['-I' + str(old / 'src/include')] if variant == 'original' else []
            subprocess.run(
                [
                    *common,
                    *override,
                    *source_flags[0][1],
                    str(ROOT / INPUTS[1]),
                    '-x',
                    'none',
                    *objects,
                    str(a.library),
                    '-Wl,--gc-sections',
                    '-Wl,--version-script=' + str(symbols),
                    '-Wl,-rpath,' + str(a.library.resolve().parent),
                    '-Wl,-rpath,' + str(a.sdk / 'lib'),
                    '-pthread',
                    '-o',
                    str(binary),
                ],
                cwd=a.rccl_build,
                check=True)
            # A few hundred host parser checks must complete within the ordinary
            # focused-test bound; a hung negative control cannot publish evidence.
            run = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            row = dict(variant=variant,
                returncode=run.returncode,
                stdout=run.stdout,
                stderr=run.stderr,
                measured=json.loads(run.stdout))
            report['observations'].append(row)
            print(json.dumps(row), flush=True)
    for path, expected in report['native_sources'].items():
        if digest(path) != expected:
            raise RuntimeError('Native source changed during proof: ' + path)
    st = compiler.stat()
    if (
        [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] != report['compiler']['stat']
    ):
        raise RuntimeError('Compiler changed during proof')
    verify(report, a.library)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    pending = a.report.with_suffix(a.report.suffix + '.tmp')
    pending.write_text(json.dumps(report, indent=2) + '\n')
    os.replace(pending, a.report)

def main():
    """Separate builder-owned compilation from installed device-free verification."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['compile', 'verify'])
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--library', type=Path, required=True)
    p.add_argument('--rccl-build', type=Path)
    p.add_argument('--compiler')
    p.add_argument('--architecture')
    p.add_argument('--sdk', type=Path)
    a = p.parse_args()
    if a.mode == 'compile':
        if not all((a.rccl_build, a.compiler, a.architecture, a.sdk)):
            p.error('compile needs actual dependency build/compiler/architecture/SDK')
        compile_report(a)
    else:
        verify(json.loads(a.report.read_text()), a.library)
if __name__ == '__main__':
    main()
