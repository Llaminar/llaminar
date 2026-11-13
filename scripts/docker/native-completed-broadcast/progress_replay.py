#!/usr/bin/env python3
"""@file progress_replay.py
@brief Replay the literal terminal observer using original native communicator/channel/connector types.

This CPU test cannot load a GPU runtime, create a completion authority or select
inference work. Actual compiled device entry resources remain a separate proof.
Deliberately fatal children alone are non-dumpable so external crash reporting
cannot dominate this bounded replay. They still execute the literal native trap
and must terminate by the host ISA's trap signal (SIGILL from ud2 on x86-64,
SIGTRAP from brk on AArch64); no signal is intercepted or replaced.
"""
from pathlib import Path
import os,json,shlex,subprocess,resource,hashlib
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT']);backend=os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']
source=root/'tree'/backend;build=Path(os.environ['LLAMINAR_NATIVE_REPLAY_BUILD']);sdk=Path(os.environ['LLAMINAR_NATIVE_REPLAY_SDK'])
compiler=os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'];out=root/'replay/progress';out.mkdir(parents=True,exist_ok=True)
s=(source/'src/device/common.cu').read_text();a=s.index('void ncclDevKernel_P2pProgressObservation(');b=s.index('{',a)+1;depth=1
while depth:depth+=(s[b]=='{')-(s[b]=='}');b+=1
body=s[a:b]
prefix=r'''
#include "device.h"
#include <memory>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <sys/wait.h>
#include <sys/prctl.h>
#include <csignal>
#include <unistd.h>
// __builtin_trap() raises the host ISA's trap: ud2/SIGILL on x86-64, brk/SIGTRAP on AArch64.
#if defined(__aarch64__)
constexpr int kNativeTrapSignal=SIGTRAP;
#else
constexpr int kNativeTrapSignal=SIGILL;
#endif
struct ObservationThread {unsigned x=0;};
ObservationThread observationThread,observationBlock;
#define threadIdx observationThread
#define blockIdx observationBlock
#ifndef __HIP_DEVICE_COMPILE__
#define __HIP_DEVICE_COMPILE__ 1
#endif
'''
fixture=r'''
int checks=0;
void check(bool ok){if(!ok)std::abort();++checks;}
int main(){
 uint32_t abort=0,foreign=0;
 NativeComm comm{};comm.rank=0;comm.nRanks=8;comm.nNodes=1;comm.abortFlag=&abort;
 std::vector<ncclDevChannel> channels(MAXCHANNELS);
 std::vector<ncclDevChannelPeer> peers(MAXCHANNELS);
 std::vector<std::vector<ncclDevChannelPeer*>> rows(MAXCHANNELS,std::vector<ncclDevChannelPeer*>(8));
 comm.channels=channels.data();
 std::vector<uint64_t> output(2+2*MAXCHANNELS);
 for(int epoch=0;epoch<20;++epoch){
  uint64_t sent=0,received=0;
  for(int c=0;c<MAXCHANNELS;++c){
   rows[c][7]=&peers[c];channels[c].peers=rows[c].data();
   uint64_t localSent=0,localReceived=0;
   for(int k=0;k<NCCL_MAX_CONNS;++k){
    peers[c].send[k].step=epoch+c+k;peers[c].recv[k].step=2*epoch+c+k;
    localSent+=peers[c].send[k].step;localReceived+=peers[c].recv[k].step;
   }
   // Unused original channels have no peer, rather than fake padding progress.
   if(c%3==0){channels[c].peers=nullptr;localSent=localReceived=0;}
   sent+=localSent;received+=localReceived;
  }
  ncclDevKernel_P2pProgressObservation(&comm,&abort,7,reinterpret_cast<ncclLlaminarP2pProgress*>(output.data()));
  check(output[0]==sent);check(output[1]==received);
  for(int c=0;c<MAXCHANNELS;++c){
   uint64_t localSent=0,localReceived=0;
   for(int k=0;k<NCCL_MAX_CONNS;++k){
    check(peers[c].send[k].step==uint64_t(epoch+c+k));
    check(peers[c].recv[k].step==uint64_t(2*epoch+c+k));
    if(c%3){localSent+=peers[c].send[k].step;localReceived+=peers[c].recv[k].step;}
   }
   check(output[2+2*c]==localSent);check(output[3+2*c]==localReceived);
  }
  for(int bad=0;bad<7;++bad){
   pid_t child=fork();check(child>=0);
   if(!child){
    // These deliberately fatal provenance checks need the original trap, not
    // a userspace crash reporter. RLIMIT_CORE=0 alone does not suppress a
    // piped core handler. Disable dumping only in this diagnostic child;
    // failure exits normally so it cannot masquerade as the expected trap signal.
    if(prctl(PR_SET_DUMPABLE,0L,0L,0L,0L)!=0 || prctl(PR_GET_DUMPABLE,0L,0L,0L,0L)!=0){
     std::fprintf(stderr,"Cannot disable native progress replay child dumps\n");_exit(86);
    }
    NativeComm invalid=comm;const NativeComm* input=&invalid;auto* flag=&abort;int peer=7;
    switch(bad){case 0:input=nullptr;break;case 1:invalid.channels=nullptr;break;
      case 2:flag=&foreign;break;case 3:invalid.nNodes=2;break;
      case 4:peer=-1;break;case 5:peer=8;break;case 6:peer=0;break;}
    ncclDevKernel_P2pProgressObservation(input,flag,peer,reinterpret_cast<ncclLlaminarP2pProgress*>(output.data()));_exit(0);
   }
   int status=0;check(waitpid(child,&status,0)==child);check(WIFSIGNALED(status) && WTERMSIG(status)==kNativeTrapSignal);
  }
 }
 printf("PASS original native connector observer:20 reset/replays,all actual channels/connections,immutable steps,140 fatal provenance cases\n");
}
'''
unit=out/'progress.cpp';unit.write_text(prefix+'\nusing NativeComm='+('ncclKernelComm;' if backend=='nccl' else 'ncclDevComm;')+'\n'+body+'\n'+fixture)
if backend=='nccl':
 command=[compiler,'-std=c++17','-O3','-I'+str(build/'include'),'-I'+str(source/'src/include'),'-I'+str(source/'src/include/plugin'),'-I'+str(sdk/'include'),'-I'+str(sdk/'include/cccl'),str(unit),'-o',str(out/'progress')]
else:
 table=json.loads((build/'compile_commands.json').read_text());own=next(row for row in table if row['file'].endswith('/enqueue.cc'))
 roots=[Path(flag[2:]) for flag in shlex.split(own['command']) if flag.startswith('-I')]
 fmt=[path for path in roots if(path/'fmt/format.h').is_file()]
 if len(fmt)!=1:raise RuntimeError('Original RCCL fmt owner is incomplete')
 command=[str(sdk/'bin/hipcc'),'-std=c++17','-O3','-D__HIP_PLATFORM_AMD__=1','-DENABLE_COLLTRACE','-DFMT_HEADER_ONLY=1','--offload-host-only','--hip-path='+str(sdk),'--rocm-path='+str(sdk)]
 command+=['-I'+str(build/path) for path in ('include','hipify/src/include','hipify/src/include/plugin','hipify/src')]
 command+=['-I'+str(fmt[0]),'-x','hip','-c',str(unit),'-o',str(out/'progress.o')]
r=subprocess.run(command,capture_output=True,text=True);(out/'compile.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
if backend=='rccl':
 r=subprocess.run([compiler,str(out/'progress.o'),'-o',str(out/'progress')],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
dependencies=subprocess.check_output(['readelf','-d',str(out/'progress')],text=True)
if any(name in dependencies for name in ('libamdhip','librccl','libnccl','libcuda','libhsa')):raise RuntimeError('CPU observer replay acquired a runtime')
r=subprocess.run([str(out/'progress')],capture_output=True,text=True,timeout=30)
if r.returncode:raise RuntimeError(r.stdout+r.stderr)
result=dict(backend=backend,compile_exit=0,returncode=0,output=r.stdout,source_sha256=hashlib.sha256(s.encode()).hexdigest())
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(r.stdout,end='')
