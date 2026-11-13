#!/usr/bin/env python3
## @file p2p_launch_replay.py
# @brief Replay actual native plan/source/connector sealing through its production typed header.
"""Use the original native work, batch, connector and pending-retirement bodies without runtime work."""
from pathlib import Path
import json,os,subprocess,shlex
root=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
backend=os.environ["LLAMINAR_NATIVE_REPLAY_BACKEND"]
source=root/"tree"/backend;output=root/"replay";output.mkdir(exist_ok=True)
sdk=Path(os.environ["LLAMINAR_NATIVE_REPLAY_SDK"])
build=Path(os.environ["LLAMINAR_NATIVE_REPLAY_BUILD"])
compiler=os.environ["LLAMINAR_NATIVE_REPLAY_COMPILER"]
if backend=="nccl":
    command=[compiler,"-std=c++17","-O3","-pthread"]
    command += ["-I"+str(source/path) for path in ("src/include","src/include/plugin","src/include/plugin/tuner")]
    command += ["-I"+str(build/"include"),"-I"+str(sdk/"include"),"-I"+str(sdk/"include/cccl"),"-I"+str(sdk/"targets/x86_64-linux/include/cccl")]
    command += [str(Path(__file__).with_name("p2p_launch_fixture.cpp")),"-o",str(output/"p2p-launch")]
else:
    # Native RCCL headers require the selected HIP compiler's host language
    # model. Compile host-only and link plainly, so this fixture cannot load a
    # HIP/RCCL runtime or create a native device authority.
    table=json.loads((build/"compile_commands.json").read_text())
    owning=next(row for row in table if row["file"].endswith("/enqueue.cc"))
    flags=shlex.split(owning["command"])
    roots=[Path(flag[2:]) for flag in flags if flag.startswith("-I")]
    fmt=[path for path in roots if (path/"fmt/format.h").is_file()]
    if len(fmt)!=1:raise RuntimeError("Original RCCL compiler does not identify one fmt header owner")
    command=[str(sdk/"bin/hipcc"),"-std=c++17","-O3","-pthread","-D__HIP_PLATFORM_AMD__=1",
             "-DENABLE_COLLTRACE","-DFMT_HEADER_ONLY=1","--offload-host-only",
             "--hip-path="+str(sdk),"--rocm-path="+str(sdk)]
    command += ["-I"+str(build/path) for path in ("include","hipify/src/include","hipify/src/include/plugin","hipify/src")]
    command += ["-I"+str(fmt[0]),"-I"+str(sdk/"include"),"-I"+str(sdk/"hip/include")]
    command += ["-x","hip","-c",str(Path(__file__).with_name("rccl_p2p_launch_fixture.cpp")),
                "-o",str(output/"p2p-launch.o")]
compiled=subprocess.run(command,capture_output=True,text=True)
(output/"p2p-launch-compile.log").write_text(compiled.stdout+compiled.stderr)
if compiled.returncode:raise RuntimeError(compiled.stdout+compiled.stderr)
if backend=="rccl":
    link=[compiler,str(output/"p2p-launch.o"),"-pthread","-o",str(output/"p2p-launch")]
    linked=subprocess.run(link,capture_output=True,text=True)
    (output/"p2p-launch-link.log").write_text(linked.stdout+linked.stderr)
    if linked.returncode:raise RuntimeError(linked.stdout+linked.stderr)
    dependencies=subprocess.check_output(["readelf","-d",str(output/"p2p-launch")],text=True)
    if any(name in dependencies for name in ("libamdhip","librccl","libnccl","libcuda","libhsa")):
        raise RuntimeError("Device-free native-type replay acquired a vendor runtime")
run=subprocess.run([str(output/"p2p-launch")],capture_output=True,text=True,timeout=30)
if run.returncode:raise RuntimeError(run.stdout+run.stderr)
(output/"p2p-launch-result.json").write_text(json.dumps(dict(compile_exit=0,returncode=run.returncode,output=run.stdout+run.stderr),indent=2))
print(run.stdout,end="")
