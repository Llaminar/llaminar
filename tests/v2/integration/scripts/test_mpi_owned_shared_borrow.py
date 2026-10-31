## @file test_mpi_owned_shared_borrow.py
#  @brief Replay original-directory replacement and ordinary native descriptor reuse.
"""Use actual owner/native rank publication; observers never delete runtime roots."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
from mpi_session_retirement import require_retired
from test_mpi_session_lifecycle import _mpi_receipt


def run(args, name, native, command, preload=False):
    """Retain one exact native status receipt and authenticate final owner retirement."""
    case = args.evidence / name
    case.mkdir()
    base = case / 'sessions'
    base.mkdir()
    args.session_bases.append(base)
    environment = dict(os.environ, LLAMINAR_FORCE_CPU_ONLY_STARTUP='1',
        LLAMINAR_SKIP_CUDA_STARTUP='1', LLAMINAR_SKIP_ROCM_STARTUP='1', HWLOC_COMPONENTS='-gl')
    if preload:
        environment['LD_PRELOAD'] = str(args.race_module)
    observation = case / 'observation.json'
    command = [str(x).replace('@OBSERVATION@', str(observation)) for x in command]
    receipt = case / 'native-status.json'
    if native:
        command = [str(args.mpirun), '--oversubscribe', '-np', '1', '--mca',
            'orte_allowed_exit_without_sync', '1', '--', *command]
        command = [sys.executable, __file__, '--native-receipt', str(receipt), '--', *command]
    command = [str(args.helper), '--session-base', str(base), '--', *command]
    (case / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    with (case / 'output.log').open('w') as output:
        process = subprocess.Popen(command, env=environment, stdout=output, stderr=subprocess.STDOUT)
        try:
            status = process.wait(timeout=40)
        except BaseException:
            process.terminate()
            process.wait(timeout=40)
            raise
    require_retired([base])
    if status != 0:
        raise AssertionError(f'{name}: actual getter rejected proof: status {status}')
    result = {'status': status, 'retired': True}
    if native:
        result['native_status'] = json.loads(receipt.read_text())
        if result['native_status']['returncode'] != status:
            raise AssertionError('Owner did not preserve original native status')
    if preload:
        result['observation'] = json.loads(observation.read_text())
        if not all(result['observation'][key] for key in ('rejected', 'injected', 'restored')):
            raise AssertionError('Replacement was not injected, rejected and exactly restored')
    (case / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    """Run all distinct native/managed borrow boundaries in one bounded transaction."""
    parser = argparse.ArgumentParser()
    for name in ('helper', 'probe', 'repeat-probe', 'race-module', 'mpirun'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--evidence', type=Path)
    args = parser.parse_args()
    args.session_bases = []
    if args.evidence is None:
        # Automatic temporary cleanup is disabled until the canonical passive
        # observer proves every actual launched base retired. Unknown backing
        # must survive this Python stack even when a test assertion fails.
        diagnostic = tempfile.TemporaryDirectory(prefix="llaminar-owned-backing-", delete=False)
        args.evidence = Path(diagnostic.name) / "transaction"
        try:
            result = transaction(args)
        except BaseException:
            for log in sorted(args.evidence.rglob("output.log"))[-3:]:
                print(str(log) + ":\n" + log.read_text(errors="replace")[-4000:], file=sys.stderr)
            try:
                require_retired(args.session_bases, timeout_seconds=0.0)
            except BaseException as failure:
                print(f"Unknown retirement; runtime backing and diagnostic scope retained at {diagnostic.name}: {failure}", file=sys.stderr)
            else:
                diagnostic.cleanup()
            raise
        try:
            require_retired(args.session_bases, timeout_seconds=0.0)
        except BaseException:
            print(f"Unknown retirement; runtime backing and diagnostic scope retained at {diagnostic.name}", file=sys.stderr)
            raise
        diagnostic.cleanup()
        return result
    return transaction(args)


def transaction(args):
    """Run against an explicit evidence scope; its caller owns diagnostic lifetime."""
    args.evidence.mkdir(parents=True, exist_ok=False)
    assets = [args.helper, args.probe, args.repeat_probe, args.race_module,
        args.helper.parent / 'llaminar_mpi_components/mca_schizo_llaminar.so']
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets}
    results = [run(args, 'replacement-' + mode, mode == 'native',
        [args.probe, mode, '--observation', '@OBSERVATION@'], True) for mode in ('managed', 'native')]
    results += [run(args, 'ordinary-fd-' + str(slot), True, [args.repeat_probe, str(slot)]) for slot in (6, 7)]
    if before != {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in assets}:
        raise AssertionError('Native borrow assets changed during transaction')
    (args.evidence / 'result.json').write_text(json.dumps({'assets': before, 'cases': results}, indent=2) + '\n')
    print('4/4 original-directory borrow regressions passed')


if __name__ == '__main__':
    if len(sys.argv) > 5 and sys.argv[1] == '--native-receipt' and sys.argv[3] == '--':
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]), sys.argv[4:]))
    main()
