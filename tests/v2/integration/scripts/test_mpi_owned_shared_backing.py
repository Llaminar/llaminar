## @file test_mpi_owned_shared_backing.py
#  @brief Exercise actual ShmemSpin setup and original-owner abrupt retirement.
"""Observe exact recorded files/processes; only the production guardian cleans backing."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[4]/"scripts/ci"))
from mpi_session_retirement import require_retired
from test_mpi_session_lifecycle import _mpi_receipt


def wait(predicate,description,seconds=20):
    """Bound test readiness, without weakening the native lifetime timeout."""
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.02)
    raise AssertionError("Timed out waiting for "+description)


def identity(path):
    """Read one original file identity, never a global pattern inventory."""
    s=path.lstat();return (s.st_dev,s.st_ino,s.st_uid,s.st_size)


def original_facts(record):
    return tuple(record[k] for k in ("dev","ino","uid","bytes"))


class Case:
    """One bounded actual job; Python observes retirement and never removes backing."""
    def __init__(self,args,name,ranks,pause):
        self.directory=args.evidence/name;self.directory.mkdir();self.base=self.directory/"sessions";self.base.mkdir();args.session_bases.append(self.base)
        self.record=self.directory/"creator.json";self.receipt=self.directory/"native-status.json";self.handles=[]
        native=[str(args.mpirun),"--oversubscribe","-np",str(ranks),"--",str(args.runner),
            "--gtest_filter=ShmemSpinMPITest.BasicAllreduce2048:ShmemSpinMPITest.SharedMemoryNameUnlinkedAfterInitialize:Test__ShmemSpinBackend.ReinitializeUsesFreshSharedMemoryName:Test__ShmemSpinBackend.RootContainedArenaPreservesLiveNeighbor"]
        command=[str(args.helper),"--session-base",str(self.base),"--",sys.executable,__file__,"--native-receipt",str(self.receipt),"--",*native]
        environment=dict(os.environ,LLAMINAR_FORCE_CPU_ONLY_STARTUP="1",LLAMINAR_SKIP_CUDA_STARTUP="1",LLAMINAR_SKIP_ROCM_STARTUP="1",
            OMP_NUM_THREADS="1",OMP_DYNAMIC="false",HWLOC_COMPONENTS="-gl")
        if pause:environment.update(LD_PRELOAD=str(args.pause_module),LLAMINAR_SHARED_BACKING_PAUSE_REPORT=str(self.record))
        (self.directory/"command.json").write_text(json.dumps(command,indent=2)+"\n")
        self.log=(self.directory/"output.log").open("w")
        self.process=subprocess.Popen(command,env=environment,stdout=self.log,stderr=subprocess.STDOUT)
        self.created=None
        try:
          if pause:
            self.created=wait(lambda:json.loads(self.record.read_text()) if self.record.exists() else None,"actual sized shared backing")
            if identity(Path(self.created["path"]))!=original_facts(self.created):raise AssertionError("Creator did not identify its actual inode")
            self.handles.append(os.pidfd_open(self.created["pid"]))

        except BaseException:
            if self.process.poll() is None:self.process.terminate()
            self.process.wait(timeout=40);self.log.close();require_retired([self.base]);raise

    def finish(self,success):
        """Require actual native exit equality and canonical original-owner cleanup."""
        status=self.process.wait(timeout=40);self.log.close()
        native=json.loads(self.receipt.read_text());expected=native["returncode"]
        if expected<0:expected=128-expected
        if status!=expected or (success and status!=0) or (not success and status==0):
            raise AssertionError(f"Owner/native status mismatch: {status}/{native}")
        require_retired([self.base])
        if self.created:
            for key in ("path","temporary","shared"):
                if Path(self.created[key]).exists():raise AssertionError("Original backing/root survived actual retirement: "+self.created[key])
        for descriptor in self.handles:
            observer=select.poll();observer.register(descriptor,select.POLLIN)
            if not observer.poll(0):raise AssertionError("Original creator remained live after cleanup")
            os.close(descriptor)
        if "[  SKIPPED ]" in (self.directory/"output.log").read_text():raise AssertionError("Native backing replay skipped")
        result={"status":status,"native_status":native,"retired":True,"created":self.created}
        (self.directory/"result.json").write_text(json.dumps(result,indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--helper",type=Path,required=True);parser.add_argument("--runner",type=Path,required=True)
    parser.add_argument("--mpirun",type=Path,required=True);parser.add_argument("--pause-module",type=Path,required=True);parser.add_argument("--evidence",type=Path)
    args=parser.parse_args()
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
    args.evidence.mkdir(parents=True,exist_ok=False)
    assets=[args.helper,args.runner,args.pause_module,args.helper.parent/"llaminar_mpi_components/mca_schizo_llaminar.so"]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets};results=[];jobs=[]
    try:
        neighbor=Case(args,"live-neighbor",2,True);jobs.append(neighbor)
        for ranks in (2,4,8):
            ordinary=Case(args,f"success-{ranks}",ranks,False);jobs.append(ordinary);results.append(ordinary.finish(True))
            interrupted=Case(args,f"interrupted-{ranks}",ranks,True);jobs.append(interrupted)
            signal.pidfd_send_signal(interrupted.handles[0],signal.SIGKILL);results.append(interrupted.finish(False))
            if neighbor.process.poll() is not None or identity(Path(neighbor.created["path"]))!=original_facts(neighbor.created):
                raise AssertionError("Another job changed the original live neighbor")
        neighbor.process.terminate();results.append(neighbor.finish(False))
    finally:
        for job in jobs:
            if job.process.poll() is None:
                job.process.terminate();job.process.wait(timeout=40)
            if not job.log.closed:job.log.close()
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    if before!=after:raise AssertionError("Native backing assets changed during transaction")
    (args.evidence/"result.json").write_text(json.dumps({"cases":results,"assets":before,"assets_unchanged":True},indent=2)+"\n")
    print(f"{len(results)}/{len(results)} actual CPU MPI shared-backing cases passed")


if __name__=="__main__":
    if len(sys.argv)>5 and sys.argv[1]=="--native-receipt" and sys.argv[3]=="--":
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]),sys.argv[4:]))
    main()
