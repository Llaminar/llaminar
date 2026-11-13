## @file test_moe_owned_shared_startup.py
#  @brief Native MPI/file regressions for actual MoE startup publication and teardown.
"""Observe original process/file witnesses; the production owner alone cleans roots."""
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
    """Bound diagnostic readiness without extending the native30s request contract."""
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        time.sleep(.02)
    raise AssertionError("Timed out waiting for "+description)


def run(args,mode,ranks):
    """Run one actual helper/real MPI context and preserve its terminal evidence."""
    case=args.evidence/f"{mode}-{ranks}";case.mkdir();base=case/"sessions";base.mkdir();args.session_bases.append(base);reports=case/"reports";reports.mkdir()
    receipt=case/"native-status.json"
    native=[str(args.mpirun),"--oversubscribe","-np",str(ranks),"--",str(args.probe),mode,str(reports)]
    command=[str(args.helper),"--session-base",str(base),"--",sys.executable,__file__,"--native-receipt",str(receipt),"--",*native]
    environment=dict(os.environ,LLAMINAR_FORCE_CPU_ONLY_STARTUP="1",LLAMINAR_SKIP_CUDA_STARTUP="1",LLAMINAR_SKIP_ROCM_STARTUP="1",
        OMP_NUM_THREADS="1",OMP_DYNAMIC="false",HWLOC_COMPONENTS="-gl")
    (case/"command.json").write_text(json.dumps(command,indent=2)+"\n")
    handles=[];started=time.monotonic()
    with (case/"output.log").open("w") as log:
        process=subprocess.Popen(command,env=environment,stdout=log,stderr=subprocess.STDOUT)
        try:
            if mode=="peer_exit":
                peer=wait(lambda:json.loads((reports/"peer-acquired.json").read_text()) if (reports/"peer-acquired.json").exists() else None,"actual acquired declaration")
                creator=json.loads((reports/"creator.json").read_text());actual=Path(creator["path"]).lstat()
                if (actual.st_dev,actual.st_ino,actual.st_uid,actual.st_size)!=tuple(creator[k] for k in ("dev","ino","uid","bytes")):
                    raise AssertionError("Acquired declaration does not name the actual live original file")
                handles=[os.pidfd_open(peer["pid"]),os.pidfd_open(creator["pid"])]
                signal.pidfd_send_signal(handles[0],signal.SIGKILL)
            status=process.wait(timeout=40)
        finally:
            if process.poll() is None:process.terminate();process.wait(timeout=40)
    require_retired([base])
    native_status=json.loads(receipt.read_text());expected=native_status["returncode"]
    if expected<0:expected=128-expected
    fatal=mode in {"late_ack","malformed","partial","peer_exit"}
    if status!=expected or (fatal and status==0) or (not fatal and status!=0):
        raise AssertionError(f"{mode}: wrong native/owner status {status}/{native_status}")
    observations=[json.loads(p.read_text()) for p in reports.glob("rank-*.json")]
    if fatal:
        if observations:raise AssertionError("Successful terminal observation escaped a failed startup")
        diagnostic=(case/"output.log").read_text()
        expected_text={"late_ack":"startup reply has stale/foreign", "malformed":"startup declaration has malformed/stale",
                       "partial":"startup declaration has partial/wrong"}
        if mode in expected_text and expected_text[mode] not in diagnostic:raise AssertionError("Unrelated failure did not exercise expected native decoder")
    else:
        if len(observations)!=ranks:raise AssertionError("Missing actual participant terminal observations")
        participants=2 if mode=="subset" else ranks
        sent=sum(row["sent_bytes"] for row in observations);received=sum(row["received_bytes"] for row in observations)
        expected_bytes=(152+len("moe-startup-probe")+2*88)*(participants-1)
        if sent!=expected_bytes or received!=expected_bytes:raise AssertionError(f"Wrong actual live startup byte extent {sent}/{received}/{expected_bytes}")
        if mode in {"creator_failure","wrong_size","wrong_inode"}:
            if any(row["returned"] or row["mapped"] or not row["failed"] for row in observations):raise AssertionError("Rejected startup exposed a committed file/mapping")
    for descriptor in handles:
        observer=select.poll();observer.register(descriptor,select.POLLIN)
        if not observer.poll(0):raise AssertionError("Original participant remained live after owner cleanup")
        os.close(descriptor)
    creator_report=reports/"creator.json"
    if creator_report.exists() and Path(json.loads(creator_report.read_text())["path"]).exists():raise AssertionError("Original runtime backing survived actual owner retirement")
    result={"mode":mode,"ranks":ranks,"status":status,"native_status":native_status,"retired":True,
            "observations":observations,"seconds":time.monotonic()-started}
    (case/"result.json").write_text(json.dumps(result,indent=2)+"\n");return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--helper",type=Path,required=True);parser.add_argument("--probe",type=Path,required=True)
    parser.add_argument("--mpirun",type=Path,required=True);parser.add_argument("--evidence",type=Path)
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
    assets=[args.helper,args.probe,args.helper.parent/"llaminar_mpi_components/mca_schizo_llaminar.so"]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}
    cases=[("success",2),("success",4),("subset",4),("creator_failure",2),("wrong_size",2),("wrong_inode",2),
           ("late_ack",2),("malformed",2),("partial",2),("peer_exit",2)]
    results=[run(args,mode,ranks) for mode,ranks in cases]
    if before!={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in assets}:raise AssertionError("Native assets changed during transaction")
    (args.evidence/"result.json").write_text(json.dumps({"assets":before,"cases":results,"assets_unchanged":True},indent=2)+"\n")
    print(f"{len(results)}/{len(results)} actual native MoE startup regressions passed")


if __name__=="__main__":
    if len(sys.argv)>5 and sys.argv[1]=="--native-receipt" and sys.argv[3]=="--":
        raise SystemExit(_mpi_receipt(Path(sys.argv[2]),sys.argv[4:]))
    main()
