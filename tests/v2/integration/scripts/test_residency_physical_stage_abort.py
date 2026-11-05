## @file test_residency_physical_stage_abort.py
# @brief Original MPI header rejection terminates before an unmatched payload vote.
"""Own the real two-rank negative child through the canonical MPI guardian."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[4]/'scripts/ci'))
from mpi_session_retirement import require_retired
from test_mpi_session_lifecycle import _mpi_receipt


def prove(args, root: Path) -> None:
    """Require exact original reason and prompt fatal process/file retirement."""
    base=root/'sessions'
    base.mkdir()
    receipt=root/'native-status.json'
    native=[str(args.mpirun),'-np','2','--bind-to','none',
            '--mca','orte_allowed_exit_without_sync','1','--',str(args.probe),
            '--gtest_also_run_disabled_tests',
            '--gtest_filter=Test__MoEOverlayMPIResidencyConsensus.DISABLED_PhysicalHeaderRejectionDuringPayloadIsImmediatelyFatal']
    command=[str(args.helper),'--session-base',str(base),'--',sys.executable,__file__,
             '--native-receipt',str(receipt),'--',*native]
    (root/'command.json').write_text(json.dumps(command,indent=2)+'\n')
    environment=dict(os.environ,LLAMINAR_FORCE_CPU_ONLY_STARTUP='1',
        LLAMINAR_SKIP_CUDA_STARTUP='1',LLAMINAR_SKIP_ROCM_STARTUP='1',
        OMP_NUM_THREADS='1',OMP_DYNAMIC='false')
    with (root/'output.log').open('w') as output:
        process=subprocess.Popen(command,env=environment,stdout=output,stderr=subprocess.STDOUT)
        try:
            # This is a negative-test watchdog, not an increased production
            # timeout. The old path remains stuck until its 30-second timeout.
            status=process.wait(timeout=10)
        except BaseException:
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
        raise AssertionError(f'Original failed status not preserved: {status}/{result}')
    text=(root/'output.log').read_text()
    reason='Remote destination received an invalid chunk header'
    marker='[ExpertOverlay][FATAL][MPI] rank=0 lane=residency_physical_stage detail='+reason
    if ('physical-stage-child reject-original-header' not in text or marker not in text or
        text.index('physical-stage-child reject-original-header')>=text.index(marker) or
        'canonical 30-second' in text or 'unexpectedly-returned' in text):
        raise AssertionError('Original rejection did not immediately abort its original MPI domain\n'+text)
    (root/'result.json').write_text(json.dumps(dict(native_status=result,status=status,
        original_reason_before_abort=True,process_files_retired=True,
        payload_post_observed='physical-stage-child payload-posted' in text),indent=2)+'\n')
    print('Original physical header rejection: prompt fatal domain/process retirement')


def main() -> None:
    """Run only the real negative child; no model or accelerator admission."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('helper','probe','mpirun'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--evidence',type=Path)
    args=parser.parse_args()
    if args.evidence:
        args.evidence.mkdir(parents=True,exist_ok=True)
        prove(args,args.evidence)
    else:
        with tempfile.TemporaryDirectory(prefix='residency-physical-stage-abort-') as folder:
            prove(args,Path(folder))


if __name__=='__main__':
    if len(sys.argv)>5 and sys.argv[1]=='--native-receipt' and sys.argv[3]=='--':
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]),sys.argv[4:]))
    main()
