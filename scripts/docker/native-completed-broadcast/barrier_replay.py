#!/usr/bin/env python3
## @file barrier_replay.py
# @brief Replay original native GROUP launch and abortable barrier bodies with CPU launch doubles.
"""Replay literal native barrier/launch bodies with original flags and CPU launch doubles."""
from pathlib import Path
import subprocess,json,os
r=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT']);o=r/'replay';o.mkdir(exist_ok=True)
def body(s,key):
 a=s.index(key);i=s.index('{',a);d=1;j=i+1
 while d:d+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
pre=r'''
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <future>
#include <iostream>
#include <thread>
#include <sched.h>
using ncclResult_t=int;
constexpr int ncclSuccess=0,ncclInternalError=3,ncclInvalidUsage=2,ncclInvalidArgument=1;
constexpr int ncclGroupTaskTypeCollective=0,ncclLaunchModeGroup=1;
int ncclParamLaunchMode=ncclLaunchModeGroup;
#define WARN(...)
#define NCCLCHECKGOTO(e,r,label) do{auto value=(e);if(value!=ncclSuccess){r=value;goto label;}}while(0)
#define CUDACHECKGOTO NCCLCHECKGOTO
struct ncclKernelPlan{ncclKernelPlan*next=nullptr;bool isCeColl=false;};
struct ncclComm{
 ncclComm*intraComm0=nullptr;int intraRanks=2,intraBarrierPhase=0,cudaDev=0;
 uint64_t intraBarrierCounter=0,intraBarrierGate=0;
 uint32_t*abortFlag=nullptr,*abortFlagDev=nullptr,*childAbortFlag=nullptr,*childAbortFlagDev=nullptr;
 ncclComm*groupNext[1]{};
 struct{bool capturingGraph=true;ncclKernelPlan*unlaunchedPlansHead=nullptr;}planner;
 int prepared=0,launched=0,finished=0;
};
using ncclComm_t=ncclComm*;
uint64_t clockNano(){return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();}
bool ncclCudaGraphValid(bool graph){return graph;}
int cudaSetDevice(int){return 0;}
int ncclLaunchPrepare(ncclComm*c){++c->prepared;return 0;}
int ncclLaunchKernelBefore_NoUncapturedCuda(ncclComm*,ncclKernelPlan*){return 0;}
int ncclLaunchKernel(ncclComm*c,ncclKernelPlan*){++c->launched;return 0;}
int ncclLaunchCeColl(ncclComm*c,ncclKernelPlan*){++c->launched;return 0;}
int ncclLaunchKernelAfter_NoCuda(ncclComm*,ncclKernelPlan*){return 0;}
int ncclLaunchFinish(ncclComm*c){++c->finished;return 0;}
'''
main=r'''
int main(int argc,char**argv){
 const bool original=argc>1;
 for(int repeat=0;repeat<20;++repeat){
  uint32_t flags[2]{};ncclComm comm[2];ncclKernelPlan plans[2];
  for(int i=0;i<2;++i){comm[i].intraComm0=&comm[0];comm[i].abortFlag=comm[i].abortFlagDev=&flags[i];comm[i].planner.unlaunchedPlansHead=&plans[i];}
  // A real delayed peer performs the original two launch rounds and reduction.
  auto a=std::async(std::launch::async,[&]{return doLaunches(&comm[0]);});
  std::this_thread::sleep_for(std::chrono::milliseconds(1));
  auto b=std::async(std::launch::async,[&]{return doLaunches(&comm[1]);});
  assert(a.get()==ncclSuccess&&b.get()==ncclSuccess);
  for(auto&c:comm)assert(c.prepared==1&&c.launched==1&&c.finished==1);
  assert(comm[0].intraBarrierCounter==0);
 }
 if(original){
  // The original out half never reads this release-store flag: the caller
  // must remain blocked when its real peer never arrives. External watchdog
  // recognizes this exact behavior; no compiler failure counts as a negative.
  uint32_t flag=0;ncclComm comm;ncclKernelPlan plan;
  comm.intraComm0=&comm;comm.abortFlag=comm.abortFlagDev=&flag;comm.planner.unlaunchedPlansHead=&plan;
  std::thread signal([&]{std::this_thread::sleep_for(std::chrono::milliseconds(5));assert(llaminarNcclSignalCompletionAbort(&comm)==0);});
  std::cout<<"ORIGINAL_BARRIER_ENTERED"<<std::endl;
  doLaunches(&comm);signal.join();return 91;
 }
 for(int repeat=0;repeat<20;++repeat){
  uint32_t flag=0;ncclComm comm;ncclKernelPlan plan;
  comm.intraComm0=&comm;comm.abortFlag=comm.abortFlagDev=&flag;comm.planner.unlaunchedPlansHead=&plan;
  auto waiter=std::async(std::launch::async,[&]{return doLaunches(&comm);});
  // Observe the existing actual arrival, not a new protocol state or fake gate.
  while(__atomic_load_n(&comm.intraBarrierCounter,__ATOMIC_ACQUIRE)==0)std::this_thread::yield();
  assert(llaminarNcclSignalCompletionAbort(&comm)==0);
  assert(waiter.wait_for(std::chrono::milliseconds(100))==std::future_status::ready);
  assert(waiter.get()==ncclInternalError);
  assert(comm.prepared==1&&comm.launched==0&&comm.finished==0);
  assert(comm.planner.unlaunchedPlansHead==&plan);
  assert(comm.intraBarrierCounter==((uint64_t(1)<<32)+1)&&comm.intraBarrierGate==0);
  // Repeated abort is absorbing and neither fabricates arrivals nor flips phase.
  const int phase=comm.intraBarrierPhase;
  assert(llaminarNcclSignalCompletionAbort(&comm)==0);
  uint32_t sum=73;assert(ncclLlaminarIntraBarrierOutAbortable(&comm,&sum)==ncclInternalError);
  assert(sum==73&&comm.intraBarrierPhase==phase&&comm.intraBarrierGate==0);
 }
 std::cout<<"PASS actual native barrier:20 delayed peers,20 missing-peer aborts,unchanged gate/arrival,deferred plans,checked launch failure"<<std::endl;
}
'''
results=[]
for backend in [os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']]:
 sdk=r/'tree'/backend
 comm=(sdk/'src/include/comm.h').read_text();init=(sdk/'src/init.cc').read_text()
 incoming=body(comm,'inline void ncclCommIntraBarrierIn(');outgoing=body(comm,'inline uint32_t ncclCommIntraBarrierOut(')
 signal=body(init,'static ncclResult_t setCommAbortFlags(')+'\nncclResult_t '+body(init,'llaminarNcclSignalCompletionAbort(')
 candidate=(r/'tree'/backend/'src/group.cc').read_text();helper=body(candidate,'static ncclResult_t ncclLlaminarIntraBarrierOutAbortable(')
 for label,launch in [('candidate',body(candidate,'static ncclResult_t doLaunches(')),('original',body((Path(os.environ['LLAMINAR_NATIVE_REPLAY_ORIGINAL'])/'src/group.cc').read_text(),'static ncclResult_t doLaunches('))]:
  p=o/f'{backend}-{label}.cpp';p.write_text(pre+'\n'+incoming+'\n'+outgoing+'\n'+signal+'\n'+helper+'\n'+launch+'\n'+main)
  argv=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2','-pthread',str(p),'-o',str(p.with_suffix(''))]
  x=subprocess.run(argv,capture_output=True,text=True);p.with_suffix('.compile.log').write_text(x.stdout+x.stderr)
  if x.returncode:raise SystemExit(x.stderr)
  cmd=[str(p.with_suffix(''))]+(['original']if label=='original'else[])
  try:
   x=subprocess.run(cmd,capture_output=True,text=True,timeout=1);p.with_suffix('.run.log').write_text(x.stdout+x.stderr)
   if label=='original':raise SystemExit('Original native barrier unexpectedly returned')
   if x.returncode:raise SystemExit(x.stdout+x.stderr)
   results.append(dict(backend=backend,label=label,argv=cmd,returncode=x.returncode,output=x.stdout))
  except subprocess.TimeoutExpired as e:
   text=(e.stdout or b'').decode()if isinstance(e.stdout,bytes)else(e.stdout or '')
   p.with_suffix('.run.log').write_text(text)
   if label!='original' or 'ORIGINAL_BARRIER_ENTERED' not in text:raise
   results.append(dict(backend=backend,label=label,argv=cmd,expected_original_blocked=True,watchdog_seconds=1,output=text))
(o/'barrier-result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
