#!/usr/bin/env python3
## @file proof.py
# @brief Bind actual completed API, host-barrier and protocol replays to one selected dependency DSO.
"""Build and passively verify the native completed Broadcast dependency contract.

The builder replays literal native API, extent, Ring, abort and completion
bodies plus a reverse-patched original. This is a CPU protocol/ABI certificate,
not native transport, model, resource-occupancy or performance qualification.
Installed verification never loads a runtime, compiler or vendor source.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
DOCKER = HERE.parent
SCOPE = "native-completed-broadcast-api-and-device-predicate"
FILES = ("proof.py", "progress_replay.py", "replay.py", "slice_replay.py", "completion_replay.py", "cancellation_replay.py",
         "addressed_replay.py", "rccl_p2p_launch_fixture.cpp", "rccl_p2p_resources.py", "rccl_p2p_role_replay.py", "p2p_launch_replay.py", "p2p_launch_fixture.cpp", "p2p_dispatch_replay.py", "signal_replay.py", "retirement_replay.py", "original_replay.py", "proxy_replay.py", "work_replay.py", "plan_replay.py", "planner_replay.py", "plan_fixture.h", "barrier_replay.py", "protocol_replay.py")
EXPORTS = ("llaminarNcclSendCompletedBytes", "llaminarNcclRecvCompletedBytes", "llaminarNcclSendCompletedSliceOrigin", "llaminarNcclCompletedProtocolVersion", "llaminarNcclBroadcastCompletedBytes", "llaminarNcclBroadcastCompletedSliceOrigin",
           "llaminarNcclCompletionAuthority", "llaminarNcclSignalCompletionAbort",
           "llaminarNcclP2pProgressGeometry", "llaminarNcclObserveP2pProgress")
SOURCE_FILES = ("src/proxy.cc", "src/collectives.cc", "src/enqueue.cc", "src/init.cc", "src/group.cc",
                "src/include/device.h", "src/include/info.h", "src/include/comm.h", "src/include/llaminar_device_extent.h",
                "src/device/common.cu", "src/device/common.h", "src/device/broadcast.h", "src/device/llaminar_device_payload.h",
                "src/device/llaminar_device_completion.h", "src/device/llaminar_device_p2p.h", "src/device/sendrecv.h", "src/transport/shm.cc")
NCCL_SOURCE_FILES=("src/include/llaminar_p2p_launch.h","src/device/primitives.h","src/device/prims_simple.h","src/device/generate.py")
RCCL_SOURCE_FILES=NCCL_SOURCE_FILES+("src/include/llaminar_p2p_entry.h",)
ADDRESSED_OK="PASS actual addressed API/P2P bodies:20epochs,8rank edges,zero/partial/full,source slices,exact bytes,minimal live metadata steps,original completion and abort\n"
P2P_LAUNCH_OK="PASS 20 reset/replays, 43 source/geometry/connection cases per replay plus pending multi-plan retirement, 3202 assertions\n"

def source_files(backend):
    """Keep the exact original native work/launch source closure backend-specific."""
    return SOURCE_FILES+(NCCL_SOURCE_FILES if backend=="nccl" else RCCL_SOURCE_FILES)
PREFIX_OK = "actual API/enqueue/resolver/Ring replay PASS (20 epochs, 0/partial/full, four ranks, in-place/distinct, row control)\n"
SLICE_OK = "actual source-slice API/Ring PASS: root2, 20 epochs, exact offset/live bytes, all roles, zero/end/tail, no payload writes on invalid extents\n"
CANCELLATION_OK = "PASS native cancellation:20 stale counts,20 aborted zero receipts,120 nonabort or immutable traps,unchanged rows and source bounds\n"
COMPLETION_OK = "PASS native completion:100waves,700falseproofs,300abortsticky,100latechannel\n"
RETIREMENT_OK = "PASS actual native proxy retirement:20 abort holds,20 original early frees,20 owner stops,normal stop and peer drain unchanged\n"
SIGNAL_OK = "actual original release-store signal PASS:20absorbing repeats,parent+child words,no destroy/reset/allocation\n"


def require(condition, message):
    """Fail closed on incomplete or disconnected evidence."""
    if not condition:
        raise RuntimeError(message)


def digest(path):
    """Authenticate a build input or DSO, never runtime eligibility or model data."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def patch(backend):
    """Return the one appended canonical ABI patch for this backend."""
    return DOCKER / "patches" / (backend + "-device-completed-broadcast.patch")


def source_identity(backend):
    """Bind the relocatable proof and canonical patch bytes."""
    result = {"native-completed-broadcast/" + name: digest(HERE / name) for name in FILES}
    result["patches/" + patch(backend).name] = digest(patch(backend))
    if backend=="rccl":
        result["llvm-lto-linker/artifact.py"]=digest(DOCKER/"llvm-lto-linker/artifact.py")
        result["../build/gpu_spill_guard.py"]=digest(DOCKER.parent/"build/gpu_spill_guard.py")
    return result


def load(path):
    """Read JSON data without executing a runtime or a builder."""
    return json.loads(Path(path).read_text())


def valid_receipt(backend, receipt):
    """Reuse the original installer/native-build receipt authority."""
    if type(receipt) is not str or not receipt:
        return False
    if backend == "nccl":
        spec = importlib.util.spec_from_file_location("completed_gin_receipt", DOCKER / "nccl-gin-selection/proof.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        prefix = module.receipt_prefix()
        return receipt.startswith(prefix) and len(receipt) > len(prefix)
    return receipt.startswith("targets-") and receipt.count("-completed-" + digest(patch(backend)) + "-capture-") == 1


def verify_observations(observations, backend):
    """Require every exact success and all twenty behavioral original controls."""
    require(type(observations) is dict and set(observations) == ({
        "prefix", "slice", "completion", "cancellation", "signal", "retirement", "proxy", "work", "plan", "original", "barrier", "protocol", "addressed", "p2p_launch", "progress"} | ({"planner","p2p_dispatch"} if backend == "nccl" else {"p2p_role"})), "Missing native replay family")
    for key, expected in (("prefix", PREFIX_OK), ("slice", SLICE_OK),
                          ("completion", COMPLETION_OK), ("cancellation", CANCELLATION_OK), ("signal", SIGNAL_OK), ("retirement", RETIREMENT_OK),
                          ("proxy", "actual SHM send/receive progress PASS: 20 epochs, 0/partial/full, exact memcpy bytes and all capacity steps\n"),
                          ("work", "actual parameter/FIFO native loader PASS, including independently wrapping geometry/tag words, abort-stale count zero, nonabort malformed count traps\n")):
        rows = observations[key]
        require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict,
                "Missing unique backend replay: " + key)
        row = rows[0]
        require(row.get("backend") == backend and type(row.get("returncode")) is int
                and row["returncode"] == 0 and type(row.get("output")) is str,
                "Native replay failed: " + key)
        output = row["output"]
        if key == "prefix":
            lines = output.splitlines(keepends=True)
            require(len(lines) == 2 and re.fullmatch(r"work_size=[1-9][0-9]* payload_offset=[1-9][0-9]*\n", lines[0]),
                    "Missing actual native work layout")
            output = lines[1]
        require(output == expected, "Incomplete native replay: " + key)
    protocol = observations["protocol"]
    require(type(protocol) is dict and protocol.get("backend") == backend and
            protocol.get("compile_exit") == 0 and protocol.get("returncode") == 0 and
            type(protocol.get("protocol_version")) is int and protocol.get("protocol_version") == 3 and protocol.get("runs") == 20 and
            type(protocol.get("source_sha256")) is str and
            re.fullmatch(r"[0-9a-f]{64}", protocol["source_sha256"]),
            "Missing actual host-only completed protocol3 marker")
    barrier = observations["barrier"]
    require(type(barrier) is list and len(barrier) == 2,
            "Missing native group barrier success/original pair")
    candidate, original_barrier = barrier
    require(candidate.get("backend") == backend and candidate.get("label") == "candidate"
            and candidate.get("returncode") == 0 and candidate.get("output") ==
            "PASS actual native barrier:20 delayed peers,20 missing-peer aborts,unchanged gate/arrival,deferred plans,checked launch failure\n",
            "Incomplete actual native group cancellation replay")
    require(original_barrier.get("backend") == backend and original_barrier.get("label") == "original"
            and original_barrier.get("expected_original_blocked") is True
            and original_barrier.get("watchdog_seconds") == 1
            and original_barrier.get("output") == "ORIGINAL_BARRIER_ENTERED\n",
            "Missing literal original uninterruptible barrier control")
    plan = observations["plan"]
    require(type(plan) is dict and plan.get("backend") == backend and plan.get("returncode") == 0
            and plan.get("output") == "PASS native plan:60admissions,40unsupported,20legacy\n"
            and plan.get("original_compile_exit") == 0 and type(plan.get("original")) is list
            and len(plan["original"]) == 20 and all(row == dict(returncode=88, stdout="",
                stderr="ORIGINAL_GENERIC_FALLTHROUGH\n") for row in plan["original"]),
            "Native final-plan admission/original control is incomplete")
    if backend == "nccl":
        planner = observations["planner"]
        require(type(planner) is dict and planner.get("passed") is True
                and planner.get("epochs") == 20 and planner.get("modes") == 3
                and planner.get("protocols") == 3 and planner.get("widths") == [32,64,128,256,512,640]
                and planner.get("capacity_bytes") == [0,8,48,1440,737280,16777216],
                "Bounded native planner geometry replay is incomplete")
        original_planner = planner.get("original", {})
        require(original_planner.get("returncode") == 89 and original_planner.get("expected_failure") is True
                and original_planner.get("stderr") == "ORIGINAL_UNBOUNDED_PLANNER\n",
                "Missing actual original planner failure")
    addressed=observations["addressed"]
    require(type(addressed) is list and len(addressed)==4,"Missing original addressed API/device replay pair")
    candidate,capacity,boundary,control=addressed
    require(candidate.get("backend")==backend and candidate.get("original") is False and candidate.get("exit")==0 and
            candidate.get("stdout")==ADDRESSED_OK and candidate.get("stderr")=="", "Incomplete actual addressed native replay")
    require(control.get("backend")==backend and control.get("original") is True and control.get("exit")==87 and
            control.get("stdout")=="" and control.get("stderr")=="ORIGINAL_UNADDRESSED_PAYLOAD_OR_MISSING_COMPLETION\n",
            "Missing original capacity-payload/completion failure")
    require(capacity.get("backend")==backend and capacity.get("variant")=="capacity" and capacity.get("exit")==90 and
            capacity.get("stdout")=="" and capacity.get("stderr")=="UNNECESSARY_CAPACITY_METADATA_STEPS\n",
            "Missing actual original capacity-schedule failure")
    require(boundary.get("backend")==backend and boundary.get("variant")=="boundary" and boundary.get("exit")==91 and
            boundary.get("stdout")=="" and boundary.get("stderr")=="ZERO_SLICE_POINTER_LEAVES_SOURCE_BANK\n",
            "Missing actual original empty-part pointer failure")
    progress=observations["progress"]
    require(progress.get("backend")==backend and progress.get("compile_exit")==0 and progress.get("returncode")==0 and
            progress.get("output")=="PASS original native connector observer:20 reset/replays,all actual channels/connections,immutable steps,140 fatal provenance cases\n" and
            re.fullmatch(r"[0-9a-f]{64}",progress.get("source_sha256","")), "Missing original native progress observer replay")
    launch=observations["p2p_launch"]
    expected_launch=P2P_LAUNCH_OK if backend=="nccl" else (
        "PASS 20 reset/replays, actual wave32/64, unroll/trace, high-channel and source/geometry/connection cases per replay plus pending multi-plan retirement, 5083 assertions\n")
    require(launch==dict(compile_exit=0,returncode=0,output=expected_launch),"Missing actual native P2P source/connector/launch seal replay")
    if backend=="nccl":
        dispatch=observations["p2p_dispatch"]
        require(type(dispatch) is list and len(dispatch)==2 and dispatch[0].get("control") is False and
            dispatch[0].get("exit_code")==0 and dispatch[1].get("control") is True and dispatch[1].get("exit_code")==31,
            "Missing original unsafe ordinary native dispatch control")
    if backend=="rccl":
        role=observations["p2p_role"]
        require(type(role) is dict and role.get("exit_code")==0 and
                role.get("stdout")=="PASS actual role predicate: 198 Native/Sealed valid/fatal cases\n" and
                re.fullmatch(r"[0-9a-f]{64}",role.get("source_sha256","")) and
                re.fullmatch(r"[0-9a-f]{64}",role.get("snippet_sha256","")),
                "Missing actual sealed/native primitive role assertion replay")
    original = observations["original"]
    wanted = dict(returncode=87, stdout="", stderr="ORIGINAL_MISSING_COMPLETION\n")
    require(type(original) is dict and type(original.get("compile_exit")) is int
            and original["compile_exit"] == 0 and type(original.get("runs")) is list
            and len(original["runs"]) == 20 and all(row == wanted for row in original["runs"]),
            "Missing complete original-body behavioral negative")


def verify_cuda_resources(resources):
    """Require each generated entry exactly once on each distinct actual DSO target."""
    require(type(resources) is dict and resources.get("scope") == "generated-native-byte-entries"
            and resources.get("native_executed") is False, "Missing owned native entry compiler proof")
    targets = resources.get("targets")
    rows = resources.get("entries")
    require(type(targets) is list and bool(targets)
            and all(type(target) is str and re.fullmatch(r"sm_[0-9]+[a-z]?", target) for target in targets)
            and len(set(targets)) == len(targets) and type(rows) is list,
            "Duplicate/malformed native compiler target inventory")
    entries = {"ncclDevKernel_Bytes_Broadcast_RING_" + protocol + "_" + str(mode)
               for protocol in ("LL", "LL128", "SIMPLE") for mode in range(3)}
    wanted = {(target, entry) for target in targets for entry in entries}
    observed = []
    for row in rows:
        require(type(row) is dict and type(row.get("function")) is str
                and type(row.get("architecture")) is str, "Malformed native resource row")
        match = re.fullmatch(r"_Z[0-9]+(ncclDevKernel_Bytes_Broadcast_RING_(?:LL128|LL|SIMPLE)_[012])24ncclDevKernelArgsStorageILm4096EE",
                             row["function"])
        require(match is not None and type(row.get("spill_stores")) is int
                and type(row.get("spill_loads")) is int and row["spill_stores"] == 0
                and row["spill_loads"] == 0 and type(row.get("stack_bytes")) is int
                and row["stack_bytes"] >= 0, "Unqualified native entry allocation")
        observed.append((row["architecture"], match[1]))
    require(len(observed) == len(wanted) and len(set(observed)) == len(observed)
            and set(observed) == wanted, "Duplicate/missing native architecture-entry evidence")
    verify_cuda_p2p_resources(resources.get("p2p"),targets)
    progress=resources.get("progress")
    require(type(progress) is list and len(progress)==len(targets) and
            {row.get("architecture") for row in progress}==set(targets), "Missing original connector observer target inventory")
    for row in progress:
        require("ncclDevKernel_P2pProgressObservation" in row.get("function", "") and
                all(type(row.get(field)) is int and row[field]==0 for field in
                    ("stack_bytes","spill_stores","spill_loads")), "Observer spills or has incomplete allocation evidence")


def verify_cuda_p2p_resources(resources,targets):
    """Require every original native target and the actual compiler negative, never a permissive empty inventory."""
    require(type(resources) is dict and resources.get("scope")=="sealed-addressed-native-p2p" and
        resources.get("native_executed") is False and resources.get("targets")==targets,"Missing owned addressed P2P resources")
    rows=resources.get("entries");controls=resources.get("spill_controls")
    require(type(rows) is list and len(rows)==len(targets) and
        {row.get("architecture") for row in rows}==set(targets),"Incomplete addressed P2P native target inventory")
    for row in rows:
        require(type(row.get("function")) is str and "ncclDevKernel_LocalP2p" in row["function"] and
            type(row.get("stack_bytes")) is int and row["stack_bytes"]==0 and
            type(row.get("spill_stores")) is int and row["spill_stores"]==0 and
            type(row.get("spill_loads")) is int and row["spill_loads"]==0 and
            type(row.get("registers")) is int and row["registers"]>0,"Addressed P2P native entry spills or lacks allocation evidence")
    require(controls==[dict(architecture=target,actual_spill_rejected=True) for target in targets],
        "Missing actual canonical addressed P2P spill-injection control")


def verify_report(report, backend, library, receipt=None):
    """Reject stale, source-only, malformed and unrelated-failure evidence."""
    require(type(report) is dict and type(report.get("schema")) is int and report["schema"] == 1
            and report.get("certificate") is True and report.get("backend") == backend
            and report.get("scope") == SCOPE and type(report.get("protocol_version")) is int and report.get("protocol_version") == 3 and report.get("native_runtime_qualified") is False
            and report.get("sources") == source_identity(backend)
            and report.get("library_sha256") == digest(library), "Stale/source-only native ABI proof")
    require(valid_receipt(backend, report.get("build_receipt")) and
            (receipt is None or report["build_receipt"] == receipt), "Native build receipt mismatch")
    require(report.get("exports") == list(EXPORTS), "Incomplete selected native ABI")
    require(type(report.get("architecture")) is str and bool(report["architecture"]),
            "Missing original native target policy")
    compiler = report.get("compiler", {})
    require(type(compiler) is dict and type(compiler.get("path")) is str
            and Path(compiler["path"]).is_absolute() and bool(compiler.get("version"))
            and re.fullmatch(r"[0-9a-f]{64}", compiler.get("sha256", "")), "Missing builder compiler identity")
    inputs = report.get("native_sources")
    require(type(inputs) is dict and set(inputs) == set(source_files(backend))
            and all(re.fullmatch(r"[0-9a-f]{64}", value) for value in inputs.values()),
            "Missing exact native source closure")
    require(re.fullmatch(r"[0-9a-f]{64}", report.get("original_ring_sha256", "")),
            "Missing literal original source")
    require(report["original_ring_sha256"] != inputs["src/device/broadcast.h"],
            "Original source is identical to candidate")
    if backend == "nccl":
        verify_cuda_resources(report.get("cuda_owned_resources"))
    else:
        verify_rccl_resources(report.get("rccl_owned_resources"),report["architecture"],library)
    verify_observations(report.get("observations"), backend)
    require(report["observations"]["progress"]["source_sha256"]==inputs["src/device/common.cu"],
            "Observer replay is disconnected from original native source")
    if backend=="rccl":
        require(report["observations"]["p2p_role"]["source_sha256"]==inputs["src/device/prims_simple.h"],
                "Primitive role replay is disconnected from original native source")
    require(report["observations"]["protocol"]["source_sha256"] == report["native_sources"]["src/collectives.cc"],
            "Completed protocol marker is disconnected from original native source")


def original_source(source, destination, backend):
    """Reverse the exact new patch privately; preserve real source and other repairs."""
    text = patch(backend).read_text()
    names = {line[6:] for line in text.splitlines() if line.startswith("+++ b/")}
    for name in names:
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe native patch path")
        existing = source / relative
        if existing.exists():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, target)
    result = subprocess.run(["git", "apply", "--reverse", "--check", "--unsafe-paths",
                             "--directory=" + str(destination), str(patch(backend))],
                            cwd="/tmp", capture_output=True, text=True)
    require(result.returncode == 0, "Actual native source does not carry the canonical ABI patch: " + result.stderr)
    subprocess.run(["git", "apply", "--reverse", "--unsafe-paths", "--directory=" + str(destination),
                    str(patch(backend))], cwd="/tmp", check=True)


def owned_cuda_resources(args, work):
    """Compile the actual generated byte entries with fatal spill diagnostics on every DSO SM.

    Untouched upstream kernels remain outside this small project-owned entry
    compilation. Final complete-library build and native execution are separate
    obligations, never inferred from this compiler-only check.
    """
    sdk = args.sdk.resolve(strict=True)
    listing = subprocess.check_output([str(sdk / "bin/cuobjdump"), "--dump-resource-usage", str(args.library)], text=True)
    targets = set()
    architecture = None
    for line in listing.splitlines():
        if line.startswith("arch = sm_"):
            architecture = line.removeprefix("arch = sm_").strip()
        elif "Function " in line and "ncclDevKernel_Bytes_Broadcast_RING" in line:
            require(architecture is not None, "Native byte entry has no target identity")
            targets.add(architecture)
    targets = sorted(targets)
    require(bool(targets), "Selected native DSO has no CUDA architecture inventory")
    generated = (args.build_dir / "obj/device/gensrc/broadcast.cu").read_text().splitlines()
    definitions = [line for line in generated if line.startswith("DEFINE_ncclDevKernelByteMode(")]
    require(len(definitions) == 9, "Native byte mode inventory is incomplete")
    translation_unit = work / "owned-native.cu"
    translation_unit.write_text('#include "common.h"\n#include "broadcast.h"\n' + "\n".join(definitions) + "\n")
    command = [str(sdk / "bin/nvcc"), "-ccbin", args.compiler, "-std=c++17",
               "--expt-extended-lambda", "-O3", "-Xfatbin=-compress-all",
               "-Xptxas=-maxrregcount=96,-v,--warn-on-spills,--warning-as-error",
               "-DDOCA_VERBS_USE_CUDA_WRAPPER", "-DDOCA_VERBS_USE_NET_WRAPPER",
               "-I.", "-I..", "-I" + str(args.build_dir / "include"),
               "-I../include", "-I../include/plugin", "--compiler-options=-fPIC,-fvisibility=hidden"]
    command += ["-gencode=arch=compute_" + arch + ",code=sm_" + arch for arch in targets]
    command += ["-dc", str(translation_unit), "-o", str(work / "owned-native.o")]
    compiled = subprocess.run(command, cwd=args.source / "src/device", capture_output=True, text=True)
    require(compiled.returncode == 0, "Project native byte entries failed spill compilation:\n" + compiled.stdout + compiled.stderr)
    resources = []
    current = None
    for line in compiled.stderr.splitlines():
        match = re.search(r"Compiling entry function '([^']+)' for '(sm_[0-9]+[a-z]?)'", line)
        if match:
            current = dict(function=match[1], architecture=match[2])
        match = re.search(r"([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads", line)
        if match and current is not None:
            current.update(stack_bytes=int(match[1]), spill_stores=int(match[2]), spill_loads=int(match[3]))
            resources.append(current)
            current = None
    require(len(resources) == 9 * len(targets) and all(not row["spill_stores"] and not row["spill_loads"] for row in resources),
            "Incomplete actual native byte entry allocation evidence")
    return dict(scope="generated-native-byte-entries", targets=["sm_" + arch for arch in targets],
                source_sha256=digest(translation_unit), compiler_sha256=digest(sdk / "bin/nvcc"),
                command=command, entries=resources, native_executed=False,
                p2p=owned_cuda_p2p_resources(args,work,["sm_"+arch for arch in targets]),
                progress=owned_cuda_progress_resources(args,work,["sm_"+arch for arch in targets]))


def owned_cuda_progress_resources(args,work,targets):
    """Compile the exact original observer body under the canonical fatal spill policy."""
    source=(args.source/"src/device/common.cu").read_text()
    start=source.index("__global__ __launch_bounds__(64,1) void ncclDevKernel_P2pProgressObservation(")
    end=source.index("{",start)+1; depth=1
    while depth:
        depth+=(source[end]=="{")-(source[end]=="}");end+=1
    unit=work/"owned-progress.cu"
    unit.write_text('#include "device.h"\n'+source[start:end]+"\n")
    command=[str(args.sdk/"bin/nvcc"),"-ccbin",args.compiler,"-std=c++17","-O3",
        "-Xptxas=-maxrregcount=96,-v,--warn-on-spills,--warning-as-error",
        "-I"+str(args.build_dir/"include"),"-I../include","-I../include/plugin"]
    command += ["-gencode=arch=compute_"+target.removeprefix("sm_")+",code="+target for target in targets]
    command += ["-dc",str(unit),"-o",str(work/"owned-progress.o")]
    result=subprocess.run(command,cwd=args.source/"src/device",capture_output=True,text=True)
    (work/"owned-progress-compile.log").write_text(result.stdout+result.stderr)
    require(result.returncode==0,"Original progress observer failed fatal spill compilation: "+result.stderr)
    rows=[];current=None
    for line in result.stderr.splitlines():
        match=re.search(r"Compiling entry function '([^']*ncclDevKernel_P2pProgressObservation[^']*)' for '(sm_[0-9]+[a-z]?)'",line)
        if match:current=dict(function=match[1],architecture=match[2]);rows.append(current)
        match=re.search(r"([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads",line)
        if match and current is not None:current.update(stack_bytes=int(match[1]),spill_stores=int(match[2]),spill_loads=int(match[3]))
        match=re.search(r"Used ([0-9]+) registers",line)
        if match and current is not None:current["registers"]=int(match[1])
    require(len(rows)==len(targets),"Incomplete observer compiler output")
    return rows


def owned_cuda_p2p_resources(args,work,targets):
    """Compile the actual sealed owned entry, and prove its strict rule rejects real injected spills."""
    sdk=args.sdk.resolve(strict=True)
    generated=args.build_dir/"obj/device/gensrc/sendrecv_local_p2p.cu"
    require(generated.exists(),"Native build omitted the actual sealed addressed P2P entry")
    unit=work/"owned-p2p.cu";unit.write_text(generated.read_text())
    base=[str(sdk/"bin/nvcc"),"-ccbin",args.compiler,"-std=c++17","--expt-extended-lambda","-O3",
          "-Xptxas=-maxrregcount=96,-v,--warn-on-spills,--warning-as-error",
          "-I.","-I..","-I"+str(args.build_dir/"include"),"-I../include","-I../include/plugin",
          "--compiler-options=-fPIC,-fvisibility=hidden"]
    architecture_flags=["-gencode=arch=compute_"+target.removeprefix("sm_")+",code="+target for target in targets]
    command=base+architecture_flags+["-dc",str(unit),"-o",str(work/"owned-p2p.o")]
    result=subprocess.run(command,cwd=args.source/"src/device",capture_output=True,text=True)
    require(result.returncode==0,"Sealed addressed P2P entry failed strict compilation:\n"+result.stdout+result.stderr)
    entries=[];current=None
    for line in result.stderr.splitlines():
        match=re.search(r"Compiling entry function '([^']*ncclDevKernel_LocalP2p[^']*)' for '(sm_[0-9]+[a-z]?)'",line)
        if match:
            current=dict(function=match[1],architecture=match[2]);entries.append(current)
        match=re.search(r"([0-9]+) bytes stack frame, ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads",line)
        if match and current is not None:current.update(stack_bytes=int(match[1]),spill_stores=int(match[2]),spill_loads=int(match[3]))
        match=re.search(r"Used ([0-9]+) registers",line)
        if match and current is not None:current["registers"]=int(match[1])
    negative=work/"p2p-pressure";negative.mkdir()
    original=(args.source/"src/device/common.h").read_text()
    changed=original.replace("__launch_bounds__(NCCL_LLAMINAR_LOCAL_P2P_NTHREADS, 1)","__launch_bounds__(640, 1)")
    require(changed!=original,"Actual owned P2P launch annotation is absent")
    (negative/"common.h").write_text(changed)
    badunit=negative/"owned-p2p-pressure.cu";badunit.write_text(generated.read_text())
    controls=[]
    for target in targets:
        arch=target.removeprefix("sm_")
        command=base+["-gencode=arch=compute_"+arch+",code="+target,"-dc",str(badunit),"-o",str(negative/(target+".o"))]
        failure=subprocess.run(command,cwd=args.source/"src/device",capture_output=True,text=True)
        rejected=failure.returncode!=0 and "Registers are spilled" in failure.stderr and "ncclDevKernel_LocalP2p" in failure.stderr
        require(rejected,"Actual owned P2P spill-injection control did not fail at its compiler resource guard")
        controls.append(dict(architecture=target,actual_spill_rejected=True))
    return dict(scope="sealed-addressed-native-p2p",native_executed=False,targets=targets,entries=entries,spill_controls=controls)



def verify_rccl_resources(resources, architecture, library):
    """Require every owned target/entry, complete final records and its whole call-free closure."""
    require(type(resources) is dict and resources.get("scope")=="sealed-addressed-native-p2p" and
            resources.get("native_executed") is False and resources.get("whole_library_zero_spill_qualified") is False and
            resources.get("allocation_evidence_complete") is True and resources.get("library_sha256")==digest(library),
            "Missing complete original RCCL owned-entry resource proof")
    targets=architecture.replace(",",";").split(";")
    require(bool(targets) and len(set(targets))==len(targets) and
            all(re.fullmatch(r"gfx[0-9a-f]+",target) for target in targets) and resources.get("targets")==targets,
            "Changed RCCL owned-entry architecture inventory")
    require(resources.get("producer_sha256")==digest(DOCKER/"llvm-lto-linker/artifact.py") and
            resources.get("spill_guard_sha256")==digest(DOCKER.parent/"build/gpu_spill_guard.py") and
            re.fullmatch(r"[0-9a-f]{64}",resources.get("final_resource_evidence_sha256","")),
            "Disconnected original final-allocation authority")
    roles={"ncclDevKernel_LocalP2p_"+str(unroll)+"_"+trace for unroll in (1,2,4) for trace in ("Plain","Trace")} | {"ncclDevKernel_P2pProgressObservation"}
    wanted={(target,role) for target in targets for role in roles}
    rows=resources.get("entries");require(type(rows) is list,"Missing final owned-entry rows")
    observed=[]
    for row in rows:
        require(type(row) is dict and type(row.get("function")) is str,"Malformed owned-entry resource row")
        match=re.fullmatch(r"_Z[0-9]+(ncclDevKernel_(?:LocalP2p_[124]_(?:Plain|Trace)|P2pProgressObservation)).*",row["function"])
        require(match is not None,"Foreign RCCL resource entry")
        for field in ("private_bytes","vgpr_spills","frame_slots","out_of_line_calls"):
            require(type(row.get(field)) is int and row[field]==0,"RCCL owned entry has memory, spill or unaudited callee: "+field)
        require(row.get("dynamic_stack") is False and type(row.get("allocation_instances")) is int and
                row["allocation_instances"]>0 and type(row.get("sgpr_lane_spills")) is int and row["sgpr_lane_spills"]>=0 and
                re.fullmatch(r"[0-9a-f]{64}",row.get("code_sha256","")),
                "Incomplete native kernel/allocation/ISA closure")
        observed.append((row.get("architecture"),match[1]))
    require(len(observed)==len(wanted) and len(set(observed))==len(observed) and set(observed)==wanted,
            "Duplicate/missing RCCL owned architecture-entry evidence")


def owned_rccl_resources(args, work):
    """Extend the original final-allocation producer with exact owned-entry and ISA observations."""
    root=args.build_dir/"llaminar-final-allocation"
    specification=root/"configuration.json"
    require(specification.is_file(),"Selected RCCL build lost its original final-allocation configuration")
    config=load(specification)
    targets=args.architecture.replace(",",";").split(";")
    guard=DOCKER.parent/"build/gpu_spill_guard.py"
    require(config.get("architectures")==targets and digest(Path(config["spill_guard"]))==digest(guard),
            "RCCL final allocation changed its original targets or guard")
    artifact=DOCKER/"llvm-lto-linker/artifact.py"
    final=work/"rccl-final-allocation.json"
    # Reuse the complete original output collector and its passive validator
    # before narrowing to this separately owned entry. The producer explicitly
    # does not certify unrelated upstream kernels as spill-free.
    commands=[
        [os.sys.executable,"-B",str(artifact),"compile","--sdk",str(args.sdk),"--spec",str(specification),
         "--report",str(final),"--receipt",str(args.receipt)],
        [os.sys.executable,"-B",str(artifact),"verify","--report",str(final),"--library",str(args.library),
         "--receipt",str(args.receipt),"--spill-guard",str(guard),"--architectures",";".join(targets)],
        [os.sys.executable,"-B",str(HERE/"rccl_p2p_resources.py"),"--library",str(args.library),
         "--allocation-root",str(root),"--spill-guard",str(guard),"--llvm-bin",str(args.sdk/"lib/llvm/bin"),
         "--output",str(work/"rccl-owned"),"--architectures",";".join(targets),"--final-report",str(final)]]
    for index,command in enumerate(commands):
        result=subprocess.run(command,capture_output=True,text=True)
        (work/("rccl-resource-"+str(index)+".log")).write_text(result.stdout+result.stderr)
        require(result.returncode==0,"RCCL original resource authority or owned closure rejected:\n"+result.stdout+result.stderr)
    complete=load(final);owned=load(work/"rccl-owned/audit.json")
    require(owned["dso_sha256"]==digest(args.library) and owned["architectures"]==targets,
            "RCCL owned observation lost its exact DSO/target identity")
    rows=[]
    for observation in complete["observations"]:
        target=observation["architecture"]
        allocation=next(row for row in owned["final_allocation"] if row["arch"]==target)["entries"]
        isa=next(row for row in owned["isa"] if row["target"].endswith(target))
        for kernel in observation["kernels"]:
            if not any(role in kernel["name"] for role in ("ncclDevKernel_LocalP2p_","ncclDevKernel_P2pProgressObservation")):continue
            function=kernel["name"];match=re.search(r"ncclDevKernel_(?:LocalP2p_[124]_(?:Plain|Trace)|P2pProgressObservation)",function)
            require(match is not None and function in allocation,"Missing original final owned allocation record")
            body=[row for row in isa["entries"] if match[0]+"(" in row["symbol"]]
            require(len(body)==1,"Missing/ambiguous owned kernel ISA closure")
            rows.append(dict(architecture=target,function=function,private_bytes=kernel["private_bytes"],
                dynamic_stack=kernel["dynamic_stack"],vgpr_spills=kernel["vgpr"],sgpr_lane_spills=kernel["sgpr"],
                frame_slots=len(allocation[function]["slots"]),allocation_instances=allocation[function]["instances"],
                out_of_line_calls=body[0]["out_of_line_calls"],code_sha256=isa["code_sha256"]))
    report=dict(scope="sealed-addressed-native-p2p",native_executed=False,whole_library_zero_spill_qualified=False,
        allocation_evidence_complete=True,library_sha256=digest(args.library),targets=targets,entries=rows,
        producer_sha256=digest(artifact),spill_guard_sha256=digest(guard),final_resource_evidence_sha256=digest(final))
    verify_rccl_resources(report,args.architecture,args.library)
    return report


def compile_report(args):
    """Compile actual bodies and exact negatives, then bind the original DSO/receipt."""
    source = args.source.resolve(strict=True)
    library = args.library.resolve(strict=True)
    receipt = args.receipt.read_text().strip()
    require(valid_receipt(args.backend, receipt), "Builder did not select the completed native patch")
    compiler = Path(shutil.which(args.compiler) or args.compiler).resolve(strict=True)
    symbols = subprocess.check_output(["nm", "-D", str(library)], text=True)
    require(all(any(line.endswith(" " + name) for line in symbols.splitlines()) for name in EXPORTS),
            "Selected DSO lacks the complete native ABI")
    source_before = {name: digest(source / name) for name in source_files(args.backend)}
    library_before = digest(library)
    with tempfile.TemporaryDirectory(prefix="llaminar-native-completed-") as directory:
        work = Path(directory)
        (work / "tree").mkdir()
        (work / "tree" / args.backend).symlink_to(source, target_is_directory=True)
        original = work / "original"
        original.mkdir()
        original_source(source, original, args.backend)
        environment = os.environ.copy()
        environment.update(LLAMINAR_NATIVE_REPLAY_ROOT=str(work),
                           LLAMINAR_NATIVE_REPLAY_BACKEND=args.backend,
                           LLAMINAR_NATIVE_REPLAY_COMPILER=str(compiler),
                           LLAMINAR_NATIVE_REPLAY_ORIGINAL=str(original),
                           LLAMINAR_NATIVE_REPLAY_BUILD=str(args.build_dir.resolve(strict=True)),
                           LLAMINAR_NATIVE_REPLAY_SDK=str(args.sdk.resolve(strict=True)))
        for script in ("replay.py", "slice_replay.py", "completion_replay.py", "cancellation_replay.py", "signal_replay.py", "retirement_replay.py", "proxy_replay.py", "work_replay.py", "plan_replay.py", "original_replay.py", "barrier_replay.py", "protocol_replay.py", "addressed_replay.py", "p2p_launch_replay.py", "progress_replay.py"):
            run = subprocess.run([os.sys.executable, "-B", str(HERE / script)], env=environment,
                                 capture_output=True, text=True)
            require(run.returncode == 0, script + " actual-body replay failed:\n" + run.stdout + run.stderr)
        if args.backend == "nccl":
            for script in ("p2p_dispatch_replay.py",):
                run=subprocess.run([os.sys.executable,"-B",str(HERE/script)],env=environment,capture_output=True,text=True)
                require(run.returncode==0,script+" failed:\n"+run.stdout+run.stderr)
            run = subprocess.run([os.sys.executable, "-B", str(HERE / "planner_replay.py")],
                                 env=environment, capture_output=True, text=True)
            require(run.returncode == 0, "Actual native planner replay failed: " + run.stdout + run.stderr)
        if args.backend=="rccl":
            run=subprocess.run([os.sys.executable,"-B",str(HERE/"rccl_p2p_role_replay.py")],
                               env=environment,capture_output=True,text=True)
            require(run.returncode==0,"Actual native role check replay failed: "+run.stdout+run.stderr)
        files = dict(prefix="result.json", slice="slice-result.json", completion="completion-result.json", cancellation="cancellation-result.json",
                     signal="signal-result.json", retirement="retirement-result.json", proxy="proxy-result.json", work="loader-result.json", plan="plan-result.json", original="original-result.json", barrier="barrier-result.json", protocol="protocol-result.json")
        observations = {key: load(work / "replay" / name) for key, name in files.items()}
        observations["addressed"]=load(work/"replay/addressed/results.json")
        observations["progress"]=load(work/"replay/progress/result.json")
        observations["p2p_launch"]=load(work/"replay/p2p-launch-result.json")
        if args.backend=="rccl":
            observations["p2p_role"]=load(work/"replay/rccl-role/role-check-result.json")
        if args.backend == "nccl":
            observations["p2p_dispatch"]=load(work/"replay/p2p-dispatch-result.json")
            observations["planner"] = load(work / "replay/planner-result.json")
            observations["planner"]["original"] = load(work / "replay/planner-original-result.json")
        verify_observations(observations, args.backend)
        resources = owned_cuda_resources(args, work) if args.backend == "nccl" else None
        rccl_resources = owned_rccl_resources(args,work) if args.backend=="rccl" else None
        report = dict(schema=1, certificate=True, backend=args.backend, scope=SCOPE, protocol_version=3,
                      native_runtime_qualified=False, sources=source_identity(args.backend),
                      library_sha256=library_before, build_receipt=receipt, architecture=args.architecture,
                      compiler=dict(path=str(compiler), sha256=digest(compiler),
                                    version=subprocess.check_output([str(compiler), "--version"], text=True).splitlines()[0]),
                      exports=list(EXPORTS), native_sources=source_before, cuda_owned_resources=resources,rccl_owned_resources=rccl_resources,
                      original_ring_sha256=digest(original / "src/device/broadcast.h"), observations=observations)
        require(source_before == {name: digest(source / name) for name in source_files(args.backend)}
                and library_before == digest(library) and receipt == args.receipt.read_text().strip(),
                "Native source/library/receipt changed during proof")
        verify_report(report, args.backend, library, receipt)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")


def main():
    """Expose builder compilation and installed passive verification separately."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("compile", "verify"):
        command = sub.add_parser(name)
        command.add_argument("--backend", choices=("nccl", "rccl"), required=True)
        command.add_argument("--report", type=Path, required=True)
        command.add_argument("--library", type=Path, required=True)
        command.add_argument("--receipt", type=Path, required=True)
        if name == "compile":
            command.add_argument("--source", type=Path, required=True)
            command.add_argument("--build-dir", type=Path, required=True)
            command.add_argument("--compiler", required=True)
            command.add_argument("--architecture", required=True)
            command.add_argument("--sdk", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compile":
        compile_report(args)
    else:
        verify_report(load(args.report), args.backend, args.library, args.receipt.read_text().strip())
    print("PASS native completed Broadcast " + args.command + " (CPU contract only)")


if __name__ == "__main__":
    main()
