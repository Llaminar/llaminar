"""Replay actual final plan admission and the original generic fallthrough.

Queue/allocation doubles only supply recorded host work. Actual native selector
and finishPlan bodies decide admission; no GPU or alternate planner executes.
"""
from pathlib import Path
import json
import os
import re
import subprocess


def function(text, key):
    """Return one complete literal native definition."""
    start = text.index(key)
    cursor = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[start:cursor]


work = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
backend = os.environ["LLAMINAR_NATIVE_REPLAY_BACKEND"]
source = work / "tree" / backend
build = Path(os.environ["LLAMINAR_NATIVE_REPLAY_BUILD"])
original = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ORIGINAL"])
output = work / "replay"
text = (source / "src/enqueue.cc").read_text()
selector = function(text, "static void* uniformKernelForPlan(")
finish = function(text, "static ncclResult_t finishPlan(")
prelude = (Path(__file__).parent / "plan_fixture.h").read_text()
# This fixture contains collective work only. The original native P2P
# seal/binder is replayed separately with actual source/work/connector types.
prelude += """
enum class ncclP2pLaunchStatus {NotApplicable,Sealed};
struct PlanP2pSeal{ncclP2pLaunchStatus status;};
PlanP2pSeal ncclLlaminarSealP2pLaunch(ncclComm*,ncclKernelPlan*,ncclDevWorkBatch*,int){return {ncclP2pLaunchStatus::NotApplicable};}
void* ncclLlaminarLocalP2pKernelFor(ncclComm*){return nullptr;}
ncclResult_t ncclLlaminarBindP2pLaunch(ncclKernelPlan*,PlanP2pSeal,void* =nullptr){return ncclSuccess;}
"""
if backend == "nccl":
    generated = (build / "obj/device/gensrc/host_table.cc").read_text()
    tables = ""
    for key in ("extern int const ncclDevByteKernelIndexForFunc[][3]",
                "extern int const ncclDevUniformKernelIndexForFunc[]"):
        begin = generated.index(key)
        tables += generated[begin:generated.index(";", begin)+1] + "\n"
    count = re.search(r"extern int const ncclDevFuncIdCount = ([0-9]+);", generated)
    if not count:
        raise RuntimeError("Native function population is absent")
    tables += count[0] + "\n"
    indices = re.findall(r"\{(-?\d+),(-?\d+),(-?\d+)\}", tables)
    function_id = next(i for i, row in enumerate(indices) if int(row[0]) >= 0)
    setup = ""
    assertion = "assert(p.kernelFn==ncclDevKernelList[ncclDevByteKernelIndexForFunc[functionId][mode]]);"
    unsupported = "p.threadPerBlock=256;"
else:
    generated = (build / "hipify/gensrc/host_table.cpp").read_text()
    begin = generated.index("const ncclUniformKernelEntry ncclUniformKernelList[]")
    tables = generated[begin:generated.index(";", begin)+1]
    names = re.findall(r"\(void\*\)(\w+)", tables)
    if not names:
        raise RuntimeError("Native uniform function population is absent")
    tables = ("struct ncclUniformKernelEntry{int funcId;int unrollIndex;bool collTrace;void*kernelFn;};\n"
              + "".join("void " + name + "(){}\n" for name in names) + tables
              + "\nconst int ncclUniformKernelCount=sizeof(ncclUniformKernelList)/sizeof(ncclUniformKernelList[0])-1;\n")
    prelude = prelude.replace("#define MAXCHANNELS 64", "#define MAXCHANNELS 64\n#define __HIP_PLATFORM_AMD__ 1\n#define ENABLE_COLLTRACE")
    prelude = prelude.replace("struct ncclComm{ncclKernelPlanner planner;", "struct ncclComm{int unroll=0;bool collTraceEnabled=false;ncclKernelPlanner planner;")
    prelude = prelude.replace("struct ncclDevKernelArgs{", "struct channelMasks{uint64_t masks[2];channelMasks(uint64_t v=1):masks{v,0}{}};\nstruct ncclDevKernelArgs{").replace("uint64_t channelMask", "channelMasks channelMask")
    entries = re.findall(r"\{[^{}]*\(void\*\)(\w+)[^{}]*\}", tables)
    broadcast_index = next(index for index, name in enumerate(entries) if "Broadcast" in name)
    function_id = None
    setup = "c.unroll=ncclUniformKernelList[0].unrollIndex;c.collTraceEnabled=ncclUniformKernelList[0].collTrace;"
    assertion = "assert(p.kernelFn==ncclUniformKernelList[" + str(broadcast_index) + "].kernelFn);"
    setup = setup.replace("[0]", "[" + str(broadcast_index) + "]")
    unsupported = "b.batch.funcId=999999;"
main = '''
int main(){
 for(int i=0;i<4096;++i)ncclDevKernelList[i]=reinterpret_cast<void*>(size_t(i+20));
 const unsigned functionId=''' + (str(function_id) if function_id is not None else "ncclUniformKernelList[" + str(broadcast_index) + "].funcId") + ''';
 for(int epoch=0;epoch<20;++epoch)for(int mode=0;mode<3;++mode){
  Work w{{nullptr,ncclDevWorkTypeColl,sizeof(ncclDevWorkColl)},{{ncclLlaminarPayloadKind(mode)}}};
  auto run=[&](bool unsupported){
   ncclComm c{};c.workArgsBytes=4096;c.nNodes=1;c.devComm=reinterpret_cast<void*>(99);
   ''' + setup + '''
   ncclWorkBatchList b{nullptr,{functionId,-1}};
   c.planner.wipPlan.channels[0].workBatchQueue.head=&b;
   ncclKernelPlan p{};p.workQueue.head=&w.node;
   if(unsupported){''' + unsupported + '''}
   auto result=finishPlan(&c,&p);
   if(!unsupported){assert(result==ncclSuccess);''' + assertion + '''}
   else if(mode){
    if(result!=ncclInvalidUsage){std::fprintf(stderr,"ORIGINAL_GENERIC_FALLTHROUGH\\n");return 88;}
    assert(p.kernelFn==reinterpret_cast<void*>(7));
   }else assert(result==ncclSuccess);
   return 0;
  };
  if(int code=run(false))return code;
  if(int code=run(true))return code;
 }
 std::printf("PASS native plan:60admissions,40unsupported,20legacy\\n");
}
'''
program = prelude + tables + "\n" + selector + "\n" + finish + main
path = output / (backend + "-plan.cpp")
path.write_text(program)
compiler = os.environ["LLAMINAR_NATIVE_REPLAY_COMPILER"]
command = [compiler, "-std=c++17", "-O2", str(path), "-o", str(path.with_suffix(""))]
subprocess.run(command, check=True)
run = subprocess.run([str(path.with_suffix(""))], capture_output=True, text=True)
if run.returncode:
    raise RuntimeError(run.stdout + run.stderr)
old = function((original / "src/enqueue.cc").read_text(), "static void finishPlan(")
# Add only the new unused selector call-shape argument. Original admission and
# return/error behavior stays literal; this isolates missing failure propagation.
if backend == "nccl":
    old = old.replace("comm->nNodes == 1))", "comm->nNodes == 1, plan))")
old = old.replace("finishPlan(", "finishPlanOriginal(", 1)
adapter = "\nstatic ncclResult_t finishPlan(ncclComm*c,ncclKernelPlan*p){finishPlanOriginal(c,p);return ncclSuccess;}\n"
negative = output / (backend + "-plan-original.cpp")
negative.write_text(program.replace(finish, old + adapter))
negative_command = [compiler, "-std=c++17", "-O2", str(negative), "-o", str(negative.with_suffix(""))]
subprocess.run(negative_command, check=True)
runs = []
for repetition in range(20):
    failed = subprocess.run([str(negative.with_suffix(""))], capture_output=True, text=True)
    row = dict(returncode=failed.returncode, stdout=failed.stdout, stderr=failed.stderr)
    if row != dict(returncode=88, stdout="", stderr="ORIGINAL_GENERIC_FALLTHROUGH\n"):
        raise RuntimeError("Original plan control failed for an unrelated reason: " + repr(row))
    runs.append(row)
(output / "plan-result.json").write_text(json.dumps(dict(backend=backend, returncode=0,
    output=run.stdout, original_compile_exit=0, original=runs), indent=2) + "\n")
print(run.stdout, end="")
