#!/usr/bin/env python3
## @file p2p_dispatch_replay.py
# @brief Prove actual sealed/ordinary native entry policy and original unsafe-control dispatch.
"""Compile the actual native entry macros with device-free typed dispatch stubs."""
from pathlib import Path
import subprocess,json,os
p=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT'])/'replay';s=(Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT'])/'tree/nccl/src/device/common.h').read_text()
def macro(name):
 a=s.index('#define '+name);end=a
 while True:
  end=s.index('\n',end)+1
  if not s[a:end].rstrip().endswith('\\'):return s[a:end]
pre='''#include <cstdint>
#define __global__
#define __launch_bounds__(...)
#define NCCL_GRID_CONSTANT
#define NCCL_MAX_NTHREADS 640
#define NCCL_LLAMINAR_LOCAL_P2P_NTHREADS 256
constexpr int NCCL_ALGO_RING=0,NCCL_PROTO_SIMPLE=0;
enum ncclFunc_t {ncclFuncSendRecv,ncclFuncBroadcast};
enum class ncclWorkNetworkPolicy {Native,NodeLocal};
enum class ncclKernelBatchPolicy {UniformFunction,LocalP2p256};
struct ncclDevKernelArgs4K {int args;};
template<class T>struct FuncCopy{};
template<ncclFunc_t Fn,class T,class R,int A,int P,ncclWorkNetworkPolicy N=ncclWorkNetworkPolicy::Native>
struct RunWorkBatch {static constexpr auto network=N;};
ncclWorkNetworkPolicy observed;ncclKernelBatchPolicy policy;
template<int Id,class Work,ncclKernelBatchPolicy Policy>void ncclKernelMain(const int*){observed=Work::network;policy=Policy;}
'''
macros=macro('DEFINE_ncclDevKernelUniform(')+macro('DEFINE_ncclDevKernelLocalP2p(')
post='''DEFINE_ncclDevKernelUniform(Probe,ncclFuncSendRecv,FuncCopy,int8_t,0,0,0)
DEFINE_ncclDevKernelUniform(Broadcast,ncclFuncBroadcast,FuncCopy,int8_t,0,0,0)
DEFINE_ncclDevKernelLocalP2p(0)
int main(){ncclDevKernelArgs4K a{};ncclDevKernel_Uniform_Probe(a);
 if(observed!=ncclWorkNetworkPolicy::Native||policy!=ncclKernelBatchPolicy::UniformFunction)return 31;
 ncclDevKernel_Uniform_Broadcast(a);if(observed!=ncclWorkNetworkPolicy::NodeLocal)return 32;
 ncclDevKernel_LocalP2p(a);if(observed!=ncclWorkNetworkPolicy::NodeLocal||policy!=ncclKernelBatchPolicy::LocalP2p256)return 33;
}
'''
rows=[]
for negative in [False,True]:
 text=macros.replace('coll == ncclFuncSendRecv ? ncclWorkNetworkPolicy::Native : ncclWorkNetworkPolicy::NodeLocal','ncclWorkNetworkPolicy::NodeLocal')if negative else macros
 stem='dispatch-policy'+('-unsafe-control'if negative else '');f=p/(stem+'.cpp');f.write_text(pre+text+post);a=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(f),'-o',str(p/stem)];v=subprocess.run(a,capture_output=True,text=True);(p/(stem+'-compile.log')).write_text(v.stdout+v.stderr);assert v.returncode==0,v.stderr
 v=subprocess.run([str(p/stem)],capture_output=True,text=True);assert v.returncode==(31 if negative else 0);rows.append(dict(control=negative,exit_code=v.returncode,argv=a))
(p/'p2p-dispatch-result.json').write_text(json.dumps(rows,indent=2));print(rows)
