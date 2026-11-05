## @file test_prepared_service_abort_diagnostic.py
#  @brief Real prepared-source CPU domain failures retain original reason before MPI abort.
"""Observe actual production guards in owned native children; never fake owners."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts/ci'))
from mpi_session_retirement import require_retired
from test_mpi_session_lifecycle import _mpi_receipt


def prove(args, root: Path, case: str, diagnostic: str) -> None:
    """Require original native nonzero status, reason-before-abort and process retirement."""
    folder=root/case
    folder.mkdir()
    base=folder/'sessions'
    base.mkdir()
    receipt=folder/'native-status.json'
    native=[str(args.mpirun), '-np', '2', '--bind-to', 'socket', '--map-by', 'socket',
            '--mca', 'mpi_leave_pinned', '1', '--mca', 'btl_vader_single_copy_mechanism', 'none',
            '--mca', 'orte_allowed_exit_without_sync', '1', '--', str(args.probe),
            '--gtest_also_run_disabled_tests', '--gtest_filter=CPUExpertDomainNative.DISABLED_'+case]
    command=[str(args.helper), '--session-base', str(base), '--', sys.executable, __file__,
             '--native-receipt', str(receipt), '--', *native]
    (folder/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    # The CMake registration also preserves the original CPU-only scope.
    environment=dict(os.environ, LLAMINAR_FORCE_CPU_ONLY_STARTUP='1',
                     LLAMINAR_SKIP_CUDA_STARTUP='1', LLAMINAR_SKIP_ROCM_STARTUP='1',
                     OMP_NUM_THREADS='1', OMP_DYNAMIC='false')
    with (folder/'output.log').open('w') as output:
        process=subprocess.Popen(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
        try:
            # Match the standard native collective progress boundary; two
            # separate children share the enclosing CTest retirement budget.
            status=process.wait(timeout=30)
        except BaseException:
            # Existing guardian remains responsible for every descendant/root.
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
            finally:
                require_retired([base])
            raise
    require_retired([base])
    result=json.loads(receipt.read_text())
    expected=result['returncode'] if result['returncode']>=0 else 128-result['returncode']
    if status==0 or status!=expected:
        raise AssertionError(f'{case}: failed native status was not preserved: {status}/{result}')
    text=(folder/'output.log').read_text()
    reason=('CPU joint prepared-compute requires original admitted initial source'
            if case=='PreparedCPUJointOriginalReasonBeforeAbort'
            else 'Prepared target jobs require original initial banks and PMA')
    matching=next((line for line in text.splitlines() if diagnostic in line and reason in line),None)
    original=text.find(matching) if matching else -1
    abort=text.find('GlobalTPContext: Abort requested for domain')
    if original<0 or abort<0 or original>=abort:
        raise AssertionError(f'{case}: original reason missing or followed abort\n{text}')
    (folder/'result.json').write_text(json.dumps({'native_status':result,'status':status,
        'original_reason_before_abort':True,'process_files_retired':True},indent=2)+'\n')
    print(f'{case}: original reason visible before native abort; owned processes/files retired')


def main() -> None:
    """Run two intentionally rejected real jobs under the established session helper."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('helper','probe','mpirun'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    def run(root):
        prove(args,root,'PreparedCPUJointOriginalReasonBeforeAbort',
              '[MoEOverlayCPUServiceMeasurement] Original CPU joint job failed participant=')
        prove(args,root,'PreparedTargetsOriginalReasonBeforeAbort',
              '[MoEOverlayCPUServiceMeasurement] Original prepared target sequence failed rank=-1: '
              'Prepared target jobs require original initial banks and PMA')
    if args.evidence:
        args.evidence.mkdir(parents=True,exist_ok=True)
        run(args.evidence)
    else:
        with tempfile.TemporaryDirectory(prefix='prepared-service-original-abort-') as directory:
            run(Path(directory))


if __name__=='__main__':
    if len(sys.argv)>5 and sys.argv[1]=='--native-receipt' and sys.argv[3]=='--':
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]),sys.argv[4:]))
    main()
