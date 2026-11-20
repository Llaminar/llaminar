#!/usr/bin/env python3
"""Exercise actual source-slice ABI and Ring roles with root coordinate two."""
from pathlib import Path
import subprocess,json,os
root=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"]);out=root/'replay'
def body(s,key):
 """Extract the complete pinned native definition without replacing its implementation."""
 a=s.index(key);i=s.index('{',a);n=1;j=i+1
 while n:n+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
main=r'''
#include <csignal>
// __builtin_trap() raises the host ISA's trap: ud2/SIGILL on x86-64, brk/SIGTRAP on AArch64.
#if defined(__aarch64__)
constexpr int kNativeTrapSignal=SIGTRAP;
#else
constexpr int kNativeTrapSignal=SIGILL;
#endif
unsigned char* failedOutput=nullptr;size_t failedOutputSize=0;
void expectedTrap(int){for(size_t i=0;i<failedOutputSize;++i)if(failedOutput[i]!=0xa5)_exit(67);_exit(66);}
int main(){
 constexpr size_t cap=43,bank=128;unsigned char source[bank],dest[cap];
 for(size_t i=0;i<bank;++i)source[i]=(i*7+3)%251;
 Comm comm;comm.nRanks=4;constexpr int rootRank=2;void*stream=reinterpret_cast<void*>(16);
 unsigned long long receipt=0;uint64_t receiverExtent=cap;
 uint64_t table=17,admission=23,sequence=1;int32_t firstError=0;
 ncclLlaminarByteCompletion completed{};
 ncclLlaminarCompletionOperand completion{&table,&admission,&sequence,&firstError,&completed,&receipt,11,37,0};
 comm.abortFlagDev=&nativeAbort;
 for(int epoch=0;epoch<20;++epoch)
 for(ncclLlaminarSourceSlice slice: {ncclLlaminarSourceSlice{0,cap},{70,17},{bank,0},{bank-3,3},{1,0},{0,1},{32,cap}})
 for(int rank=0;rank<4;++rank){
  sequence=epoch+1;nativeAbort=0;ncclShmem.aborted=0;firstError=0;abortInDestructor=false;
  const bool isRoot=rank==rootRank;comm.rank=rank;receiverExtent=cap;sourceExtent=slice.liveBytes;
  plans.clear();headerBytes=0;std::memset(dest,0xa5,cap);receipt=0;
  auto*extent=isRoot?&slice.liveBytes:&receiverExtent;
  assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,isRoot?&slice:nullptr,extent,isRoot?bank:0,cap,rootRank,0,&completion,&comm,stream)==0);
  assert(plans.size()==2&&headerBytes==8&&ncclGroupDepth==0&&*extent==slice.liveBytes);
  ncclDevWorkColl work{};work.root=rootRank;work.oneNode=1;work.sendbuff=source;work.recvbuff=dest;
  work.llaminarPayload=plans.back().llaminarPayload;
  assert(work.llaminarPayload.kind==(isRoot?ncclLlaminarPayloadKind::CompletedSourceSlice:ncclLlaminarPayloadKind::CompletedBytes));
  if(isRoot){auto ptr=work.llaminarPayload.binding.slice.slice;assert(ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(ptr),bank,2,&nativeAbort)==reinterpret_cast<uint64_t>(ptr));}
  else work.llaminarPayload.binding.resolved.liveBytes=ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(extent),cap,1,&nativeAbort);
  ncclShmem.channel.ring.userRanks[0]=rank;ncclShmem.channel.ring.userRanks[1]=(rank+1)%4;
  admittedCapacity=cap;wireSource=source+slice.byteOffset;copiedCounts.clear();sentBytes.clear();
  runRing<unsigned char,int,Proto>(0,1,&work);
  size_t useful=0;for(auto n:copiedCounts)useful+=n;
  assert(useful==slice.liveBytes&&copiedCounts.size()==(cap+chunk-1)/chunk);
  assert(receipt==((rank+1)%4==rootRank?0:slice.liveBytes));
  assert(ncclLlaminarCompletionMatches(completed,table,admission,sequence,slice.liveBytes,11,{&nativeAbort,ncclShmem.args.comm},nativeAbort,firstError));
  for(size_t i=0;i<cap;++i)assert(dest[i]==(i<slice.liveBytes?source[slice.byteOffset+i]:0xa5));
 }
 // Root in-place is exact for nonempty offset zero; empty slices can name bank end.
 for(auto slice:{ncclLlaminarSourceSlice{0,cap},{70,17},{bank-3,3},{bank,0}}){
  unsigned char original[bank];std::memcpy(original,source,bank);comm.rank=rootRank;plans.clear();headerBytes=0;
  assert(llaminarNcclBroadcastCompletedSliceOrigin(source,source,&slice,&slice.liveBytes,bank,cap,rootRank,0,&completion,&comm,stream)==0);
  ncclDevWorkColl work{};work.root=rootRank;work.oneNode=1;work.sendbuff=source;work.recvbuff=source;work.llaminarPayload=plans.back().llaminarPayload;
  ncclShmem.channel.ring.userRanks[0]=rootRank;ncclShmem.channel.ring.userRanks[1]=3;admittedCapacity=cap;sentBytes.clear();
  runRing<unsigned char,int,Proto>(0,1,&work);assert(std::memcmp(original,source,bank)==0);
  assert(sentBytes.size()==slice.liveBytes);for(size_t i=0;i<sentBytes.size();++i)assert(sentBytes[i]==original[slice.byteOffset+i]);
 }
 // Device malformed/overflow extents must trap before the first payload write.
 for(auto slice:{ncclLlaminarSourceSlice{bank+1,0},{bank,1},{UINT64_MAX,1},{64,65},{0,cap+1},{UINT64_MAX-1,5}}){
  const auto pid=fork();assert(pid>=0);
  if(!pid){
   std::memset(dest,0xa5,cap);failedOutput=dest;failedOutputSize=cap;std::signal(kNativeTrapSignal,expectedTrap);
   ncclDevWorkColl work{};work.root=rootRank;work.oneNode=1;work.sendbuff=source;work.recvbuff=dest;
   work.llaminarPayload.binding.slice={&slice,bank,&completion};work.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedSourceSlice;
   ncclShmem.channel.ring.userRanks[0]=rootRank;ncclShmem.channel.ring.userRanks[1]=3;admittedCapacity=cap;
   runRing<unsigned char,int,Proto>(0,1,&work);_exit(0);
  }
  int status;waitpid(pid,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==66);
 }

 // The exact source pointer is already the admitted subspan base. Its descriptor
 // remains absolute; normalized offset is private native arithmetic only.
 for(uint64_t origin : {uint64_t(19),UINT64_MAX-uint64_t(bank)})
 for(auto relative : {ncclLlaminarSourceSlice{0,cap},{70,17},{bank,0},{bank-3,3}}){
  auto absolute=relative;absolute.byteOffset+=origin;
  comm.rank=rootRank;plans.clear();headerBytes=0;firstError=0;nativeAbort=0;ncclShmem.aborted=0;
  std::memset(dest,0xa5,cap);sourceExtent=absolute.liveBytes;
  assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,&absolute,&absolute.liveBytes,bank,cap,rootRank,origin,&completion,&comm,stream)==0);
  assert(plans.size()==2&&plans.back().llaminarSourceOffsetOrigin==origin);
  ncclDevWorkColl work{};work.root=rootRank;work.oneNode=1;work.sendbuff=source;work.recvbuff=dest;
  work.llaminarPayload=plans.back().llaminarPayload;work.llaminarSourceOffsetOrigin=plans.back().llaminarSourceOffsetOrigin;
  ncclShmem.channel.ring.userRanks[0]=rootRank;ncclShmem.channel.ring.userRanks[1]=3;
  admittedCapacity=cap;wireSource=source+relative.byteOffset;
  runRing<unsigned char,int,Proto>(0,1,&work);
  for(size_t i=0;i<cap;++i)assert(dest[i]==(i<relative.liveBytes?source[relative.byteOffset+i]:0xa5));
  assert(absolute.byteOffset==origin+relative.byteOffset);
 }
 // Lower-bound violations, including zero bytes, fail before the first write.
 for(auto absolute : {ncclLlaminarSourceSlice{18,0},{0,1},{UINT64_MAX,1}}){
  const auto pid=fork();assert(pid>=0);
  if(!pid){
   std::memset(dest,0xa5,cap);failedOutput=dest;failedOutputSize=cap;std::signal(kNativeTrapSignal,expectedTrap);
   ncclDevWorkColl work{};work.root=rootRank;work.oneNode=1;work.sendbuff=source;work.recvbuff=dest;
   work.llaminarPayload.binding.slice={&absolute,bank,&completion};work.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedSourceSlice;
   work.llaminarSourceOffsetOrigin=19;
   ncclShmem.channel.ring.userRanks[0]=rootRank;ncclShmem.channel.ring.userRanks[1]=3;admittedCapacity=cap;
   runRing<unsigned char,int,Proto>(0,1,&work);_exit(0);
  }
  int status;waitpid(pid,&status,0);assert(WIFEXITED(status)&&WEXITSTATUS(status)==66);
 }
 // Host boundary rejects a foreign root count binding and receiver-only root metadata.
 ncclLlaminarSourceSlice slice{3,17};comm.rank=rootRank;plans.clear();headerBytes=0;
 assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,&slice,&receiverExtent,bank,cap,rootRank,0,&completion,&comm,stream)==1);
 assert(plans.empty()&&headerBytes==0);
 comm.rank=0;assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,&slice,&slice.liveBytes,bank,cap,rootRank,0,&completion,&comm,stream)==1);
 assert(plans.empty()&&headerBytes==0);
 // Immutable subspan overflow and a receiver-authored origin reject before metadata side effects.
 comm.rank=rootRank;
 assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,&slice,&slice.liveBytes,bank,cap,rootRank,UINT64_MAX,&completion,&comm,stream)==1);
 assert(plans.empty()&&headerBytes==0);
 comm.rank=0;
 assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,nullptr,&receiverExtent,0,cap,rootRank,1,&completion,&comm,stream)==1);
 assert(plans.empty()&&headerBytes==0);
 comm.rank=rootRank;ncclGroupDepth=1;
 assert(llaminarNcclBroadcastCompletedSliceOrigin(source,dest,&slice,&slice.liveBytes,bank,cap,rootRank,0,&completion,&comm,stream)==2);
 assert(plans.empty()&&headerBytes==0);ncclGroupDepth=0;
 std::cout<<"actual source-slice API/Ring PASS: root2, 20 epochs, exact offset/live bytes, all roles, zero/end/tail, no payload writes on invalid extents\n";
}
'''
results=[]
for backend in [os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']]:
 s=(out/(backend+'.cpp')).read_text().replace('int main(){','int legacyMain(){')
 api='ncclResult_t '+body((root/'tree'/backend/'src/collectives.cc').read_text(),'llaminarNcclBroadcastCompletedSliceOrigin(')
 source=s+'\n'+api+'\n'+main;p=out/(backend+'-slice.cpp');p.write_text(source)
 argv=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(p),'-o',str(p.with_suffix(''))]
 c=subprocess.run(argv,capture_output=True,text=True);(out/(backend+'-slice-compile.log')).write_text(c.stdout+c.stderr)
 if c.returncode:raise SystemExit(c.stderr[-3000:])
 r=subprocess.run([str(p.with_suffix(''))],capture_output=True,text=True);(out/(backend+'-slice-run.log')).write_text(r.stdout+r.stderr)
 if r.returncode:raise SystemExit(r.stdout+r.stderr)
 results.append(dict(backend=backend,argv=argv,returncode=r.returncode,output=r.stdout+r.stderr))
(out/'slice-result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
