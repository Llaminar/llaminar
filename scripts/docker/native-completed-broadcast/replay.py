#!/usr/bin/env python3
"""Compile isolated actual dependency bodies with CPU SDK/primitive doubles."""
from pathlib import Path
import subprocess, json, re, hashlib, os
ROOT=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
OUT=ROOT/'replay';OUT.mkdir(exist_ok=True)
def body(text, signature):
 """Extract the complete pinned native definition without replacing its implementation."""
 start=text.index(signature); brace=text.index('{',start); depth=1; end=brace+1
 while depth:
  depth+=(text[end]=='{')-(text[end]=='}');end+=1
 return text[start:end]
def decl(text, signature):
 """Retain the actual source-owned native field layout in the CPU fixture."""
 return body(text,signature)+';'
results=[]
for backend in (os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND'],):
 tree=ROOT/'tree'/backend/'src'
 coll=(tree/'collectives.cc').read_text(); enq=(tree/'enqueue.cc').read_text()
 extent=(tree/'include/llaminar_device_extent.h').read_text()
 info=decl((tree/'include/info.h').read_text(),'struct ncclInfo {')
 work=decl((tree/'include/device.h').read_text(),'struct alignas(16) ncclDevWorkColl {')
 payload=(tree/'device/llaminar_device_payload.h').read_text()
 completion_device=(tree/'device/llaminar_device_completion.h').read_text()
 broadcast=body(coll,'ncclResult_t ncclBroadcast'+('_impl' if backend=='rccl' else '')+'(')
 if backend=='rccl':broadcast=broadcast.replace('ncclBroadcast_impl(', 'ncclBroadcast(')
 api='ncclResult_t '+body(coll,'llaminarNcclBroadcastCompletedBytes(')
 authority='ncclResult_t '+body(coll,'llaminarNcclCompletionAuthority(')
 rows='ncclResult_t '+body(coll,'llaminarNcclAllGatherDeviceRows(')
 enqueue=body(enq,'ncclResult_t ncclEnqueueCheck(')
 ringtext=(tree/'device/broadcast.h').read_text()
 if backend == 'nccl':
  ring=body(ringtext,'__device__ __forceinline__ void runRingMode(')
  fixed=body(ringtext,'__device__ __forceinline__ void runRingFixed(')
  ring_source='template<typename T,typename RedOp,typename Proto,bool NodeLocal,int Mode>\n'+ring+'\ntemplate<typename T,typename RedOp,typename Proto>\n'+fixed+"""
// Test dispatch selects the recorded immutable mode; production uses actual
// native generated entries, whose whole-plan selector is replayed separately.
template<typename T,typename RedOp,typename Proto>
void runRing(int tid,int threads,ncclDevWorkColl*work){
 if(work->llaminarPayload.kind==ncclLlaminarPayloadKind::CompletedBytes)
  return runRingMode<T,RedOp,Proto,true,1>(tid,threads,work);
 if(work->llaminarPayload.kind==ncclLlaminarPayloadKind::CompletedSourceSlice)
  return runRingMode<T,RedOp,Proto,true,2>(tid,threads,work);
 return runRingFixed<T,RedOp,Proto>(tid,threads,work);
}
"""
 else:
  ring=body(ringtext,'__device__ __forceinline__ void runRing(')
  ring_source='template<typename T,typename RedOp,typename Proto>\n'+ring

 pre=r'''
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <vector>
#include <unistd.h>
#include <sys/wait.h>
using std::min;
inline size_t min(ssize_t a,size_t b){return std::min(size_t(a),b);}
#define __device__
#define __forceinline__ inline
#define __HIP_DEVICE_COMPILE__ 1
#define WARN(...)
#define INFO(...)
#define TRACE_CALL(...)
#define NVTX3_FUNC_WITH_PARAMS(...)
#define NVTX3_PAYLOAD(...)
#define NCCLCHECK(expr) do { auto r=(expr); if(r!=ncclSuccess)return r; }while(0)
#define NCCLCHECKGOTO(expr,r,label) do { r=(expr);if(r!=ncclSuccess)goto label; }while(0)
#define CUDACHECK NCCLCHECK
#define CUDACHECKGOTO NCCLCHECKGOTO
using ncclResult_t=int;constexpr int ncclSuccess=0,ncclInvalidArgument=1,ncclInvalidUsage=2;
using ncclFunc_t=int;constexpr int ncclFuncBroadcast=0,ncclFuncAllGather=1;
using ncclDataType_t=int;constexpr int ncclInt8=0;
using ncclRedOp_t=int;constexpr int ncclSum=0;
using cudaStream_t=void*;using hipStream_t=void*;
constexpr int ALLGATHER_SLICESTEPS_SINGLE_NODE=1;
constexpr int BROADCAST_CHUNKSTEPS=1,BROADCAST_SLICESTEPS=1,ALLGATHER_CHUNKSTEPS=1,ALLGATHER_SLICESTEPS=1;
struct Comm {uint32_t*abortFlagDev=nullptr;const void*devComm=reinterpret_cast<void*>(0x1234);int nRanks=2,nNodes=1,rank=0,revokedFlag=0,checkPointers=0,cudaDev=0,rcclUseOneSlice=0;struct{bool blocking=true;}config;};
using ncclComm_t=Comm*;
int ncclGroupDepth=0;
struct{int profilerGroupDepth=0;}ncclProfilerApiState;
int ncclCommCount(Comm*c,int*n){*n=c->nRanks;return 0;}
int CommCheck(Comm*c,const char*,const char*){return c?0:1;}
int ncclGroupErrCheck(int r){return r;}
int ncclCommEnsureReady(Comm*){return 0;}
int cudaGetDevice(int*d){*d=0;return 0;}int cudaSetDevice(int){return 0;}
int ncclCommGetAsyncError(Comm*,int*r){*r=0;return 0;}int ncclCommSetAsyncError(Comm*,int){return 0;}
int ncclGroupStartInternal(){++ncclGroupDepth;return 0;}
struct Recorder {static Recorder&instance(){static Recorder r;return r;}template<class T>int record(int,const T&){return 0;}};
constexpr int rrBroadcast=0,mscclFuncBroadcast=0;
bool mscclIsCaller(){return false;}bool mscclAvailable(Comm*){return false;}
template<class...T>int mscclEnqueueCheck(T...){return 2;}
'''
 native=r'''
std::vector<ncclInfo> tasks,plans;
uint64_t sourceExtent=0;size_t headerBytes=0;
int ArgsCheck(ncclInfo*i){return i->count?0:1;}
int taskAppend(Comm*,ncclInfo*i){tasks.push_back(*i);return 0;}
int ncclGroupEndInternal(){
 assert(ncclGroupDepth>0);
 if(--ncclGroupDepth==0){
  for(auto&i:tasks){
   plans.push_back(i);
   if(!i.llaminarPayload.active()&&i.coll==ncclFuncBroadcast&&i.count==8){
    if(i.comm->rank!=i.root)std::memcpy(i.recvbuff,&sourceExtent,8);
    headerBytes+=8;
   }
  }
  tasks.clear();
 }
 return 0;
}
int ncclBroadcast(const void*,void*,size_t,int,int,Comm*,void*);
int ncclEnqueueCheck(ncclInfo*);
unsigned long long atomicAdd(unsigned long long*p,unsigned long long n){auto old=*p;*p+=n;return old;}
int atomicAdd(int*p,int n){auto old=*p;*p+=n;return old;}
int atomicCAS(int*p,int expected,int desired){int old=*p;if(old==expected)*p=desired;return old;}
unsigned atomicExch(unsigned*p,unsigned n){unsigned old=*p;*p=n;return old;}
void __threadfence(){}
'''
 ringstub=r'''
constexpr int WARP_SIZE=64,COLL_UNROLL=1;
struct ncclRing {int userRanks[2]{0,1};int prev=1,next=1;};
uint32_t nativeAbort=0;
struct{struct{ncclRing ring;}channel;int channelId=0;int aborted=0;struct{uint32_t*abortFlag=&nativeAbort;}comm;struct{const void*comm=reinterpret_cast<void*>(0x1234);}args;}ncclShmem;
bool abortInDestructor=false;
size_t admittedCapacity=0,chunk=7;
std::vector<size_t> copiedCounts;
std::vector<unsigned char> sentBytes;
const unsigned char*wireSource=nullptr;
void ncclCollCbdPart(ncclDevWorkColl*,int,int,int,ssize_t*t,ssize_t*o,ssize_t*n,ssize_t*c){if(t)*t=admittedCapacity;*o=0;*n=admittedCapacity;*c=chunk;}
template<int N>struct FanSymmetric{};
template<class T,class R,class F,int X,class P,int Y>struct Primitives {
 T*in;T*out;
 Primitives(int,int,int*,int*,T*i,T*o,uint64_t,int,int,int,ncclDevWorkColl*):in(i),out(o){}
 ~Primitives(){if(abortInDestructor){nativeAbort=1;ncclShmem.aborted=1;}}
 void directSend(size_t a,size_t,int n){copiedCounts.push_back(n);if(n)sentBytes.insert(sentBytes.end(),in+a,in+a+n);}
 void directCopySend(size_t a,size_t b,int n){copiedCounts.push_back(n);if(n)sentBytes.insert(sentBytes.end(),in+a,in+a+n);std::memcpy(out+b,in+a,n*sizeof(T));}
 void directRecv(size_t b,int n){copiedCounts.push_back(n);std::memcpy(out+b,wireSource+b,n*sizeof(T));}
 void directRecvCopyDirectSend(size_t a,size_t b,int n){directRecv(b,n);}
};
template<int,class,class,int,int,int,int,int,int,int,class...T>void reduceCopy(T...){assert(false);}
template<int,int,class,class,int,int,int,int,int,int,int,class...T>void reduceCopy(T...){assert(false);}
void barrier_sync(int,int){}
struct Proto{static constexpr int Id=0;};
'''
 tests=r'''
int main(){
 static_assert(std::is_trivial<ncclLlaminarDevicePayload>::value);
 static_assert(std::is_standard_layout<ncclLlaminarDevicePayload>::value);
 std::cout<<"work_size="<<sizeof(ncclDevWorkColl)<<" payload_offset="<<offsetof(ncclDevWorkColl,llaminarPayload)<<"\n";
 constexpr size_t cap=43;
 unsigned char input[cap],output[cap];for(size_t i=0;i<cap;++i)input[i]=i+1;
 Comm comm;void*stream=reinterpret_cast<void*>(16);
 uint64_t extent=0;unsigned long long receipt=0;
 uint64_t table=17,admission=23,sequence=1;int32_t firstError=0;
 ncclLlaminarByteCompletion completed{};
 ncclLlaminarCompletionOperand completion{&table,&admission,&sequence,&firstError,&completed,&receipt,11,37,0};
 comm.abortFlagDev=&nativeAbort;
 ncclLlaminarCompletionAuthority authority{};
 assert(llaminarNcclCompletionAuthority(&comm,&authority)==0&&authority.abortFlag==&nativeAbort&&authority.deviceCommunicator==comm.devComm);
 assert(llaminarNcclCompletionAuthority(nullptr,&authority)==ncclInvalidArgument);
 for(int epoch=0;epoch<20;++epoch)for(size_t live:{cap,size_t(0),size_t(17),size_t(1),cap,size_t(3)}){
  for(int rank=0;rank<4;++rank)for(bool inPlace:{false,true}){
   sequence=epoch+1;nativeAbort=0;ncclShmem.aborted=0;firstError=0;
   comm.rank=rank;comm.nRanks=4;sourceExtent=live;extent=rank==0?live:cap;
   headerBytes=0;plans.clear();
   std::memset(output,0xa5,cap);
   auto*src=inPlace&&rank==0?output:input;
   if(inPlace&&rank==0)std::memcpy(output,input,cap);
   assert(llaminarNcclBroadcastCompletedBytes(src,output,&extent,cap,0,&completion,&comm,stream)==0);
   assert(ncclGroupDepth==0&&plans.size()==2&&headerBytes==8&&extent==live);
   assert(!plans[0].llaminarPayload.active()&&plans[1].llaminarPayload.kind==ncclLlaminarPayloadKind::CompletedBytes);
   ncclDevWorkColl work{};work.root=0;work.oneNode=1;work.sendbuff=src;work.recvbuff=output;
   work.llaminarPayload=plans.back().llaminarPayload;
   const auto resolved=ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(work.llaminarPayload.binding.bytes.bytes),cap,1,&nativeAbort);
   work.llaminarPayload.binding.resolved.liveBytes=resolved;
   ncclShmem.channel.ring.userRanks[0]=rank;ncclShmem.channel.ring.userRanks[1]=(rank+1)%4;
   admittedCapacity=cap;copiedCounts.clear();wireSource=input;receipt=0;
   runRing<unsigned char,int,Proto>(0,1,&work);
   size_t useful=0;for(auto n:copiedCounts)useful+=n;
   assert(useful==live&&copiedCounts.size()==(cap+chunk-1)/chunk);
   assert(receipt==(rank==3?0:live));
   assert(ncclLlaminarCompletionMatches(completed,table,admission,sequence,live,11,{&nativeAbort,ncclShmem.args.comm},nativeAbort,firstError));
   assert(!ncclLlaminarCompletionMatches(completed,table,admission,sequence+1,live,11,{&nativeAbort,ncclShmem.args.comm},0,0));
   for(size_t i=0;i<cap;++i)assert(output[i]==(inPlace&&rank==0?input[i]:i<live?input[i]:0xa5));
  }
 }
 // Genuine UINT64 arithmetic, without allocating/copying a >4GiB bank.
 extent=(uint64_t(1)<<32)+19;
 assert(ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&extent),extent+7,1,&nativeAbort)==extent);
 int32_t rows=3;auto geometry=(uint64_t(8)<<32)|5;
 assert(ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&rows),geometry,0,&nativeAbort)==15);
 rows=0;assert(ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&rows),geometry,0,&nativeAbort)==0);
 for(uint64_t tag:{uint64_t(3),uint64_t(1)<<32}){
  auto pid=fork();assert(pid>=0);if(!pid){ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&extent),extent+1,tag,&nativeAbort);_exit(0);}
  int status;waitpid(pid,&status,0);assert(WIFSIGNALED(status));
 }
 plans.clear();headerBytes=0;comm.rank=0;comm.nRanks=2;extent=1;
 ncclGroupDepth=1;
 assert(llaminarNcclBroadcastCompletedBytes(input,output,&extent,cap,0,&completion,&comm,stream)==ncclInvalidUsage);
 assert(plans.empty()&&tasks.empty()&&headerBytes==0);
 // Existing row API remains legal in an outer group and retains its geometry.
 rows=1;
 assert(llaminarNcclAllGatherDeviceRows(input,output,&rows,8,5,&receipt,&comm,stream)==0);
 assert(plans.empty()&&tasks.size()==1&&tasks[0].llaminarPayload.kind==ncclLlaminarPayloadKind::Rows);
 assert(ncclGroupEndInternal()==0&&plans.size()==1);
 for(auto invalidStream:{reinterpret_cast<void*>(0),reinterpret_cast<void*>(1),reinterpret_cast<void*>(2)}){
  plans.clear();headerBytes=0;
  assert(llaminarNcclBroadcastCompletedBytes(input,output,&extent,cap,0,&completion,&comm,invalidStream)==1);
  assert(plans.empty()&&headerBytes==0);
 }
 std::cout<<"actual API/enqueue/resolver/Ring replay PASS (20 epochs, 0/partial/full, four ranks, in-place/distinct, row control)\n";
}
'''
 source=('#define __HIP_PLATFORM_AMD__ 1\n' if backend=='rccl' else '')+pre+extent+'\n'+info+'\n'+work+'\n'+native+'\n'+enqueue+'\n'+broadcast+'\n'+api+'\n'+authority+'\n'+rows+'\n'+ringstub+'\n'+payload+'\n'+completion_device+'\n'+ring_source+'\n'+tests
 path=OUT/(backend+'.cpp');path.write_text(source)
 argv=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2','-Wall','-Wextra','-Wno-unused-parameter',str(path),'-o',str(OUT/backend)]
 c=subprocess.run(argv,text=True,capture_output=True);(OUT/(backend+'-compile.log')).write_text(c.stdout+c.stderr)
 if c.returncode:raise SystemExit(f'{backend} compile failed: '+c.stderr[-7000:])
 r=subprocess.run([str(OUT/backend)],text=True,capture_output=True);(OUT/(backend+'-run.log')).write_text(r.stdout+r.stderr)
 results.append(dict(backend=backend,argv=argv,source_sha256=hashlib.sha256(source.encode()).hexdigest(),returncode=r.returncode,output=r.stdout+r.stderr))
 if r.returncode:raise SystemExit(f'{backend} replay failed: '+r.stdout+r.stderr)
(OUT/'result.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
