## @file test_cpu_pipeline_cold_abort.py
# @brief Authenticate the actual original CPU pipeline cold-failure boundary and owned native exit.
"""Run one real two-socket failure child; preserve its reason, MPI status and retirement."""
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


def prove(args, root: Path) -> None:
    """Use the canonical session guardian around original MPI2/physical-socket ownership."""
    sessions=root/'sessions'
    sessions.mkdir()
    receipt=root/'native-status.json'
    native=[str(args.mpirun), '-np', '2', '--bind-to', 'socket', '--map-by', 'socket',
            '--mca', 'mpi_leave_pinned', '1', '--mca', 'btl_vader_single_copy_mechanism', 'none',
            '--mca', 'orte_allowed_exit_without_sync', '1', '--', str(args.probe),
            '--gtest_also_run_disabled_tests',
            '--gtest_filter=CPUStagePipelineConstructionFailure.DISABLED_OriginalParentReasonBeforeAbort']
    command=[str(args.helper), '--session-base', str(sessions), '--', sys.executable, __file__,
             '--native-receipt', str(receipt), '--', *native]
    (root/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    environment=dict(os.environ, LLAMINAR_FORCE_CPU_ONLY_STARTUP='1',
                     LLAMINAR_SKIP_CUDA_STARTUP='1', LLAMINAR_SKIP_ROCM_STARTUP='1',
                     OMP_NUM_THREADS='1', OMP_DYNAMIC='false')
    with (root/'output.log').open('w') as output:
        process=subprocess.Popen(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
        try:
            status=process.wait(timeout=30)
        except BaseException:
            try:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)
            finally:
                require_retired([sessions])
            raise
    require_retired([sessions])
    native_status=json.loads(receipt.read_text())
    expected=native_status['returncode'] if native_status['returncode']>=0 else 128-native_status['returncode']
    if status==0 or status!=expected:
        raise AssertionError(f'Original native failure status lost: {status}/{native_status}')
    text=(root/'output.log').read_text()
    marker='[OriginalCPUPipelineConstruction] rank=0/2 reason=CPU_PIPELINE_COLD_FAILURE_ORIGINAL_REASON; '
    reason=text.find(marker)
    abort=text.find('MPI_ABORT')
    if reason<0 or abort<reason or 'aborting original parent communicator before native CPU owner unwind' not in text:
        raise AssertionError(f'Original reason absent or not before original MPI abort\n{text}')
    if 'terminate called' in text[:abort]:
        raise AssertionError(f'Native owner unwind masked the original construction failure\n{text}')
    (root/'result.json').write_text(json.dumps({'native_status':native_status,'status':status,
        'original_reason_before_parent_abort':True,'process_files_retired':True},indent=2)+'\n')
    print('Original CPU pipeline cause precedes real parent MPI abort; owned processes/files retired')


def main() -> None:
    """Select one exact original native failure, never a simulated communicator."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('helper','probe','mpirun'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    if args.evidence:
        args.evidence.mkdir(parents=True,exist_ok=True)
        prove(args,args.evidence)
    else:
        with tempfile.TemporaryDirectory(prefix='cpu-pipeline-original-cold-abort-') as directory:
            prove(args,Path(directory))


if __name__=='__main__':
    if len(sys.argv)>5 and sys.argv[1]=='--native-receipt' and sys.argv[3]=='--':
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]),sys.argv[4:]))
    main()
