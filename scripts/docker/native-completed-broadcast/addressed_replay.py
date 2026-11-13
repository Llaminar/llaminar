#!/usr/bin/env python3
## @file addressed_replay.py
# @brief Replay actual addressed API/native P2P bodies against original controls without a runtime.
from pathlib import Path
import re,subprocess,json,os,resource
# Intentional trap controls own no core-file payload.
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
p=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT']);out=p/'replay/addressed';out.mkdir(parents=True,exist_ok=True)
original_root=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ORIGINAL'])
compiler=os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER']
def body(s,key):
 a=s.index(key);i=s.index('{',a);j=i+1;d=1
 while d:d+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
pre=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <deque>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <csignal>
// __builtin_trap() raises the host ISA's trap: ud2/SIGILL on x86-64, brk/SIGTRAP on AArch64.
#if defined(__aarch64__)
constexpr int kNativeTrapSignal=SIGTRAP;
#else
constexpr int kNativeTrapSignal=SIGILL;
#endif
#define __device__
#define __forceinline__ inline
#define __host__
#define __HIP_DEVICE_COMPILE__ 1
#define WARN(...)
#define NCCLCHECK(x) do{int r=(x);if(r)return r;}while(0)
using std::min;
using ncclResult_t=int;constexpr int ncclSuccess=0,ncclInvalidArgument=1,ncclInvalidUsage=2;
using ncclFunc_t=int;constexpr int ncclFuncSend=1,ncclFuncRecv=2;
using ncclDataType_t=int;constexpr int ncclInt8=0;
using ncclRedOp_t=int;constexpr int ncclSum=0;
using cudaStream_t=void*;
struct Comm{int nRanks=8,nNodes=1,rank=3;};using ncclComm_t=Comm*;
int ncclGroupDepth=0;
int ncclCommCount(Comm*c,int*n){*n=c->nRanks;return 0;}
struct Header{bool sending;int peer;size_t bytes;void* data;};std::vector<Header> headers;
int ncclSend(const void*p,size_t n,int,int peer,Comm*,void*){headers.push_back({true,peer,n,const_cast<void*>(p)});return 0;}
int ncclRecv(void*p,size_t n,int,int peer,Comm*,void*){headers.push_back({false,peer,n,p});return 0;}
struct{struct{uint32_t*abortFlag;int rank,p2pnChannelsPerPeer,nNodes,p2pChunkSize,isAllNvlink;}comm;
 struct{void*comm;}args;int channelId,aborted;}ncclShmem;
uint32_t nativeAbort=0;const int nativeComm=17;
unsigned int atomicCAS(int*p,int a,int b){int x=*p;if(x==a)*p=b;return x;}
unsigned int atomicExch(unsigned int*p,unsigned int v){auto x=*p;*p=v;return x;}
int atomicAdd(int*p,int v){auto x=*p;*p+=v;return x;}
unsigned long long atomicAdd(unsigned long long*p,unsigned long long v){auto x=*p;*p+=v;return x;}
void __threadfence(){}
int countOneBits(int x){return __builtin_popcount(unsigned(x));}
int reverseBits(int x,int n){int y=0;for(int i=0;i<n;++i)y=(y<<1)|((x>>i)&1);return y;}
template<class T,class U>T divUp(T a,U b){return (a+b-1)/b;}
template<class T,class U>T alignUp(T a,U b){return divUp(a,b)*b;}
'''
mocks=r'''
std::vector<ncclInfo> plans;
int ncclEnqueueCheck(ncclInfo*i){plans.push_back(*i);return 0;}
constexpr int NCCL_MAX_NET_SIZE=1048576;
int u32fp8Decode(unsigned char){return 512;}
template<int A,int B>struct FanAsymmetric{};
struct Proto{};
struct Packet{std::vector<unsigned char> bytes;};std::deque<Packet> fifo;
size_t sentBytes=0,sendSteps=0,recvSteps=0;uintptr_t sourceLow=0,sourceHigh=0;bool sourceInBounds=true;
template<class T,class R,class F,int D,class P,int E>struct Primitives{
 const unsigned char*source;unsigned char*destination;
 template<class...X>Primitives(int,int,const int*,const int*,const void*s,void*d,X...):
  source(static_cast<const unsigned char*>(s)),destination(static_cast<unsigned char*>(d)){if(s && (uintptr_t(s)<sourceLow || uintptr_t(s)>sourceHigh))sourceInBounds=false;}
 void directSend(size_t at,size_t,int n){assert(n>=0);Packet packet;packet.bytes.resize(n);if(n)memcpy(packet.bytes.data(),source+at,n);fifo.push_back(std::move(packet));sentBytes+=n;++sendSteps;}
 void directRecv(size_t at,int n){assert(!fifo.empty());auto packet=std::move(fifo.front());fifo.pop_front();assert(packet.bytes.size()==size_t(n));if(n)memcpy(destination+at,packet.bytes.data(),n);++recvSteps;}
};
'''
main=r'''
int main(){
 ncclShmem.comm.abortFlag=&nativeAbort;ncclShmem.args.comm=const_cast<int*>(&nativeComm);
 ncclShmem.comm.p2pnChannelsPerPeer=4;ncclShmem.comm.nNodes=1;ncclShmem.comm.p2pChunkSize=512;
 constexpr size_t capacity=16384,bankBytes=capacity+4096;
 alignas(64) unsigned char source[bankBytes],destination[capacity];
 sourceLow=uintptr_t(source);sourceHigh=uintptr_t(source)+bankBytes;
 for(size_t i=0;i<bankBytes;++i)source[i]=(i*11+5)%251;
 uint64_t table=17,admission=23,sequence=1,sourceExtent=0,receivedExtent=0;
 int32_t sourceError=0,receiveError=0;unsigned long long observed=0;
 ncclLlaminarByteCompletion sourceReceipt{},receiveReceipt{};
 ncclLlaminarCompletionOperand sendOperand{&table,&admission,&sequence,&sourceError,&sourceReceipt,&observed,71,9,0};
 ncclLlaminarCompletionOperand recvOperand{&table,&admission,&sequence,&receiveError,&receiveReceipt,nullptr,71,9,0};
 Comm sender,receiver;receiver.rank=6;auto stream=reinterpret_cast<void*>(0x100);

 // Ordinary foreign native work retains its original fixed capacity schedule,
 // including required zero-part progress. It has no counted receipt authority.
 for(int parts:{1,2,4,8}){
  sentBytes=sendSteps=recvSteps=observed=0;sourceInBounds=true;fifo.clear();size_t originalSteps=0;
  memset(destination,0xa5,capacity);
  for(int part=0;part<parts;++part){
   size_t first,last;ncclP2pPartBounds(parts,part,capacity,&first,&last);
   originalSteps+=std::max(size_t(1),divUp(last-first,size_t(512)));
   ncclDevWorkP2p send{},receive{};send.nP2pChannels=receive.nP2pChannels=parts;
   send.nSendChannels=receive.nRecvChannels=parts;send.sendRank=6;receive.recvRank=3;
   send.sendAddr=source+first;receive.recvAddr=destination+first;send.sendBytes=receive.recvBytes=last-first;
   ncclShmem.channelId=part;ncclShmem.comm.rank=3;runSend<unsigned char,int,Proto,ReplayPolicyNative>(0,256,0,&send);
   ncclShmem.comm.rank=6;runRecv<unsigned char,int,Proto,ReplayPolicyNative>(0,256,0,&receive);
  }
  assert(sourceInBounds&&fifo.empty()&&sentBytes==capacity&&sendSteps==originalSteps&&recvSteps==originalSteps&&observed==0);
  assert(memcmp(source,destination,capacity)==0);
 }
 for(int epoch=1;epoch<=20;++epoch)for(int parts:{1,2,4,8})for(uint64_t live:{16384ull,1ull,4095ull,4096ull,4097ull,0ull,512ull})for(bool sliced:{false,true}){
  sequence=epoch;sourceExtent=live;receivedExtent=live;sourceError=receiveError=0;nativeAbort=0;ncclShmem.aborted=0;
  sourceReceipt={};receiveReceipt={};headers.clear();plans.clear();sentBytes=sendSteps=recvSteps=observed=0;
  memset(destination,0xa5,capacity);sourceInBounds=true;size_t expectedSteps=0;ncclLlaminarSourceSlice slice{100000+(live==0?bankBytes:2048),live};
  int rc=sliced?llaminarNcclSendCompletedSliceOrigin(source,&slice,bankBytes,capacity,6,100000,&sendOperand,&sender,stream):
   llaminarNcclSendCompletedBytes(source,&sourceExtent,capacity,6,&sendOperand,&sender,stream);
  assert(rc==0 && headers.size()==1 && headers[0].sending && headers[0].peer==6 && headers[0].bytes==8);
  assert(plans.size()==1 && plans[0].coll==ncclFuncSend && plans[0].root==6 && plans[0].count==capacity);
  const auto sendInfo=plans[0];headers.clear();plans.clear();
  assert(llaminarNcclRecvCompletedBytes(destination,&receivedExtent,capacity,3,&recvOperand,&receiver,stream)==0);
  assert(headers.size()==1 && !headers[0].sending && headers[0].peer==3 && headers[0].bytes==8);
  assert(plans.size()==1 && plans[0].coll==ncclFuncRecv && plans[0].root==3);
  const auto recvInfo=plans[0];
  for(int part=0;part<parts;++part){
   size_t first,last;ncclP2pPartBounds(parts,part,capacity,&first,&last);
   const size_t local=first>=live?0:std::min(last-first,size_t(live)-first);expectedSteps+=std::max(size_t(1),divUp(local,size_t(512)));
   ncclDevWorkP2p send{},receive{};send.nP2pChannels=receive.nP2pChannels=parts;
   send.nSendChannels=receive.nRecvChannels=parts;send.sendRank=6;receive.recvRank=3;
   send.sendAddr=source+first;receive.recvAddr=destination+first;
   send.sendBytes=receive.recvBytes=last-first;
   send.sendTransfer={sendInfo.llaminarPayload,capacity,sendInfo.llaminarSourceOffsetOrigin};
   receive.recvTransfer={recvInfo.llaminarPayload,capacity,0};
   ncclShmem.channelId=part;ncclShmem.comm.rank=3;runSend<unsigned char,int,Proto,ReplayPolicySealed>(0,256,0,&send);
   ncclShmem.comm.rank=6;runRecv<unsigned char,int,Proto,ReplayPolicySealed>(0,256,0,&receive);
   assert(fifo.empty());
  }
  if(sentBytes!=live || observed!=live ||
    !ncclLlaminarCompletionMatches(receiveReceipt,table,admission,sequence,live,71,{&nativeAbort,&nativeComm},0,receiveError)){
    fprintf(stderr,"ORIGINAL_UNADDRESSED_PAYLOAD_OR_MISSING_COMPLETION\n");return 87;
  }
  if(!sourceInBounds){fprintf(stderr,"ZERO_SLICE_POINTER_LEAVES_SOURCE_BANK\n");return 91;}
  if(sendSteps!=expectedSteps || recvSteps!=expectedSteps){fprintf(stderr,"UNNECESSARY_CAPACITY_METADATA_STEPS\n");return 90;}
  for(size_t i=0;i<capacity;++i)assert(destination[i]==(i<live?source[i+(sliced?2048:0)]:0xa5));
 }
 headers.clear();plans.clear();
 for(int peer:{-1,3,8})assert(llaminarNcclSendCompletedBytes(source,&sourceExtent,capacity,peer,&sendOperand,&sender,stream)==2);
 ncclGroupDepth=1;assert(llaminarNcclSendCompletedBytes(source,&sourceExtent,capacity,6,&sendOperand,&sender,stream)==2);ncclGroupDepth=0;
 assert(headers.empty()&&plans.empty());
 for(uintptr_t bad:{0u,1u,2u})assert(llaminarNcclSendCompletedBytes(source,&sourceExtent,capacity,6,&sendOperand,&sender,reinterpret_cast<void*>(bad))==1);
 for(int epoch=0;epoch<20;++epoch){
  receivedExtent=UINT64_MAX;nativeAbort=1;receiveError=0;receiveReceipt={};
  ncclDevWorkP2p work{};work.nP2pChannels=work.nRecvChannels=1;work.recvBytes=capacity;work.recvAddr=destination;
  work.recvTransfer.payload.kind=ncclLlaminarPayloadKind::CompletedBytes;
  work.recvTransfer.payload.binding.bytes={&receivedExtent,capacity,&recvOperand};work.recvTransfer.capacityBytes=capacity;
  ncclShmem.channelId=0;
  const auto extent=ncclLlaminarResolveP2p(&work,false);assert(extent.liveBytes==0&&extent.localBytes==0);
  ncclLlaminarBeginCompletionOperand(&recvOperand,0,true);ncclLlaminarFinishCompletionOperand(&recvOperand,0,0,true);
  assert(receiveError==9&&receiveReceipt.phase==ncclLlaminarCompletionPhase::Aborted);
 }
 // Incompatible recorded work/count/source metadata must trap before a
 // native payload primitive can obtain a useful count.
 nativeAbort=0;ncclShmem.aborted=0;
 for(int replay=0;replay<20;++replay)for(int defect=0;defect<6;++defect){
  pid_t child=fork();assert(child>=0);
  if(child==0){
   ncclDevWorkP2p work{};work.nP2pChannels=work.nSendChannels=1;work.sendAddr=source;work.sendBytes=capacity;
   uint64_t count=capacity+1;ncclLlaminarSourceSlice slice{100000-1,1};
   work.sendTransfer.capacityBytes=capacity;work.sendTransfer.payload.kind=ncclLlaminarPayloadKind::CompletedBytes;
   work.sendTransfer.payload.binding.bytes={&count,capacity,&sendOperand};
   if(defect==1)work.sendTransfer.payload.binding.bytes.bytes=nullptr;
   if(defect==2){work.sendTransfer.payload.kind=ncclLlaminarPayloadKind::CompletedSourceSlice;
      work.sendTransfer.payload.binding.slice={&slice,bankBytes,&sendOperand};work.sendTransfer.sourceOffsetOrigin=100000;}
   if(defect==3){work.sendTransfer.payload.kind=ncclLlaminarPayloadKind::CompletedSourceSlice;
      slice={100000+bankBytes,1};work.sendTransfer.payload.binding.slice={&slice,bankBytes,&sendOperand};work.sendTransfer.sourceOffsetOrigin=100000;}
   if(defect==4)work.sendTransfer.payload.kind=static_cast<ncclLlaminarPayloadKind>(99);
   if(defect==5)work.sendTransfer.capacityBytes=0;
   ncclShmem.channelId=0;(void)ncclLlaminarResolveP2p(&work,true);_exit(99);
  }
  int status=0;assert(waitpid(child,&status,0)==child);assert(WIFSIGNALED(status)&&WTERMSIG(status)==kNativeTrapSignal);
 }
 puts("PASS actual addressed API/P2P bodies:20epochs,8rank edges,zero/partial/full,source slices,exact bytes,minimal live metadata steps,original completion and abort");
}
'''
results=[]
for backend in [os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']]:
 root=p/'tree'/backend/'src';coll=(root/'collectives.cc').read_text();device=(root/'include/device.h').read_text()
 extent=(root/'include/llaminar_device_extent.h').read_text();info=body((root/'include/info.h').read_text(),'struct ncclInfo {')+';';work=body(device,'struct alignas(16) ncclDevWorkP2p {')+';'
 bounds=body(device,'inline __host__ __device__ void ncclP2pPartBounds(')
 channel=body(device,'inline __device__ int ncclP2pChannelToPart(')
 payload=(root/'device/llaminar_device_payload.h').read_text()
 # Keep actual resolver helpers and discard collective-only selectors/clippers.
 selected='\n'.join(body(payload,key)for key in ['__device__ __forceinline__ uint32_t ncclLlaminarAcquireAbort(', '__device__ __forceinline__ uint64_t ncclLlaminarResolveRowWord(', '__device__ __forceinline__ uint64_t ncclLlaminarResolvePayloadWord(', '__device__ __forceinline__ ncclLlaminarSourceSlice ncclLlaminarResolveSourceSlice('])
 completion=(root/'device/llaminar_device_completion.h').read_text();completion='\n'.join(body(completion,key)for key in ['__device__ __forceinline__ void ncclLlaminarBeginCompletionOperand(', '__device__ __forceinline__ void ncclLlaminarFinishCompletionOperand('])
 helper=re.sub(r'^#include .*\n','',(root/'device/llaminar_device_p2p.h').read_text(),flags=re.M)
 api='\n'.join(body(coll,key)for key in ['static ncclResult_t llaminarNcclAddressedCompleted(', 'ncclResult_t\nllaminarNcclSendCompletedBytes(', 'ncclResult_t\nllaminarNcclRecvCompletedBytes(', 'ncclResult_t\nllaminarNcclSendCompletedSliceOrigin('])
 for labelkind in ["candidate","capacity","boundary","original"]:
  original=labelkind=="original"
  methodroot=root
  src=((original_root/'src/device/sendrecv.h')if original else(methodroot/'device/sendrecv.h')).read_text()
  selected_helper=re.sub(r'^#include .*\n','',(methodroot/'device/llaminar_device_p2p.h').read_text(),flags=re.M) if not original else helper
  if labelkind=="capacity":
   # Reproduce the original completed capacity schedule while preserving exact
   # current native types, connector ownership, receipt and clipping bodies.
   old="if(transfer.payload.active())bytes=extent.localBytes;"
   assert src.count(old)==2,"Missing unique original capacity control anchors"
   src=src.replace(old,"/* Original capacity-driven schedule. */")
  if labelkind=="boundary":
   old="local?offset:0";assert selected_helper.count(old)==1
   selected_helper=selected_helper.replace(old,"offset")
  policy=(root/'include/llaminar_p2p_entry.h').read_text() if backend=='rccl' else body((root/'device/common.h').read_text(),'enum class ncclWorkNetworkPolicy')+';'
  policy+='\nusing ReplayPolicy='+('ncclP2pEntryPolicy;constexpr auto ReplayPolicySealed=ReplayPolicy::SealedLocal256;constexpr auto ReplayPolicyNative=ReplayPolicy::Native;' if backend=='rccl' else 'ncclWorkNetworkPolicy;constexpr auto ReplayPolicySealed=ReplayPolicy::NodeLocal;constexpr auto ReplayPolicyNative=ReplayPolicy::Native;')
  methods=[]
  for function in ['runSend','runRecv']:
   signature=re.search(r'__device__\s+(?:__forceinline__\s+)?void\s+'+function+r'\(',src)
   if signature is None:raise RuntimeError('Missing actual original P2P method '+function)
   methods.append('template<class T,class RedOp,class Proto, ReplayPolicy '+('EntryPolicy' if backend=='rccl' else 'Network')+'>\n'+body(src,signature.group(0)))
  methods='\n'.join(methods)
  full=pre+'\n'+extent+'\n'+info+'\n'+work+'\n'+bounds+'\n'+channel+'\n'+mocks+'\n'+selected+'\n'+completion+'\n'+selected_helper+'\n'+api+'\n'+policy+'\n'+methods+'\n'+main
  label=backend+'-'+labelkind;f=out/(label+'.cpp');f.write_text(full);cmd=[compiler,'-std=c++17','-O2',str(f),'-o',str(out/label)];r=subprocess.run(cmd,capture_output=True,text=True);(out/(label+'-compile.log')).write_text(r.stdout+r.stderr)
  if r.returncode:print(label,'compile failed');print(r.stderr[:4000]);raise SystemExit(1)
  r=subprocess.run([str(out/label)],capture_output=True,text=True);(out/(label+'.log')).write_text(r.stdout+r.stderr);results.append({'backend':backend,'original':original,'variant':labelkind,'exit':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'compile_argv':cmd});print(label,r.returncode)
  expected=87 if original else 90 if labelkind=='capacity' else 91 if labelkind=='boundary' else 0
  assert r.returncode==expected,r.stderr
(out/'results.json').write_text(json.dumps(results,indent=2))
