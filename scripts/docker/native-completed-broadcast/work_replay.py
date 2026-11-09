#!/usr/bin/env python3
"""Replay exact parameter/FIFO native work pack resolution with actual ABI types."""
from pathlib import Path
import subprocess,json,os
root=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"]);out=root/'replay'
def declaration(s,key):
 """Retain the actual source-owned native field layout in the CPU fixture."""
 a=s.index(key);i=s.index('{',a);n=1;j=i+1
 while n:n+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]+';'
results=[]
for backend in [os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']]:
 src=root/'tree'/backend/'src'
 common=(src/'device/common.h').read_text()
 key='if (ncclShmem.args.workStorageType == ncclDevWorkStorageTypeArgs) {'
 a=common.index(key);z=common.index('char* dst = ncclShmem.workStorage;',a)
 block=common[a:z]
 pre=r'''
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
// __builtin_trap() raises the host ISA's trap: ud2/SIGILL on x86-64, brk/SIGTRAP on AArch64.
#if defined(__aarch64__)
constexpr int kNativeTrapSignal=SIGTRAP;
#else
constexpr int kNativeTrapSignal=SIGILL;
#endif
using std::min;
#define __device__
#define __forceinline__ inline
#define __HIP_DEVICE_COMPILE__ 1
unsigned long long atomicAdd(unsigned long long*p,unsigned long long n){auto x=*p;*p+=n;return x;}
'''
 pre+=(src/'include/llaminar_device_extent.h').read_text()+'\n'+declaration((src/'include/device.h').read_text(),'struct alignas(16) ncclDevWorkColl {')+'\n'+(src/'device/llaminar_device_payload.h').read_text()
 source=pre+r'''
constexpr int ncclDevWorkStorageTypeArgs=0,ncclDevWorkStorageTypeFifo=1,ncclDevWorkTypeP2p=0,ncclDevWorkTypeColl=1;
struct alignas(16) ulong2{uint64_t x,y;};
uint32_t nativeAbort=0; struct NativeComm{uint32_t*abortFlag=&nativeAbort;} originalComm;
struct {struct{int workStorageType;void*workBuf;uint32_t workMask;NativeComm*comm=&originalComm;}args;}ncclShmem;
struct Batch{uint32_t offsetBase;int workType;};
void loadPack(void*args,Batch batch,uint32_t srcWork,uint32_t workSize,uint32_t packInWork,void*destination){
 ulong2 tmp;
'''+block+r'''
 std::memcpy(static_cast<char*>(destination)+packInWork*16,&tmp,16);
}
int main(){
 alignas(16) unsigned char fifo[256],linear[512];
 uint64_t extent=0;unsigned long long receipt=0; ncclLlaminarCompletionOperand completion{};completion.outgoingPayloadBytes=&receipt;
 ncclDevWorkColl original{},observed{};
 original.llaminarSourceOffsetOrigin=0xfedcba9876543210ULL;
 original.llaminarPayload.binding.bytes={&extent,73,&completion};original.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedBytes;
 for(auto storage:{ncclDevWorkStorageTypeArgs,ncclDevWorkStorageTypeFifo})
 for(uint32_t offset=0;offset<256;offset+=16)
 for(auto live:{uint64_t(73),uint64_t(0),uint64_t(19),uint64_t(3)}){
  extent=live;std::memset(fifo,0xcd,sizeof(fifo));std::memset(linear,0xcd,sizeof(linear));
  std::memcpy(linear+offset,&original,sizeof(original));
  for(uint32_t i=0;i<sizeof(original);i+=16)std::memcpy(fifo+((offset+i)&255),reinterpret_cast<char*>(&original)+i,16);
  ncclShmem.args={storage,fifo,255};
  for(uint32_t p=0;p<sizeof(original)/16;++p)loadPack(linear,{offset,ncclDevWorkTypeColl},0,sizeof(original),p,&observed);
  assert(observed.llaminarSourceOffsetOrigin==original.llaminarSourceOffsetOrigin);
  assert(observed.llaminarPayload.binding.resolved.liveBytes==live);
  assert(observed.llaminarPayload.binding.completed.completion==&completion);
  assert(observed.llaminarPayload.kind==ncclLlaminarPayloadKind::CompletedBytes&&observed.llaminarPayload.reserved==0);
 }
 // Replay stale native count writes from a terminated count Broadcast through
 // BOTH actual loader branches and independently wrapping work/tag words.
 for(auto storage:{ncclDevWorkStorageTypeArgs,ncclDevWorkStorageTypeFifo})
 for(uint32_t offset=0;offset<256;offset+=16){
  extent=~uint64_t(0);nativeAbort=1;
  std::memcpy(linear+offset,&original,sizeof(original));
  for(uint32_t i=0;i<sizeof(original);i+=16)std::memcpy(fifo+((offset+i)&255),reinterpret_cast<char*>(&original)+i,16);
  ncclShmem.args={storage,fifo,255};
  for(uint32_t p=0;p<sizeof(original)/16;++p)loadPack(linear,{offset,ncclDevWorkTypeColl},0,sizeof(original),p,&observed);
  assert(observed.llaminarPayload.binding.resolved.liveBytes==0);
  nativeAbort=0;
  auto pid=fork();assert(pid>=0);if(!pid){
   for(uint32_t p=0;p<sizeof(original)/16;++p)loadPack(linear,{offset,ncclDevWorkTypeColl},0,sizeof(original),p,&observed);
   _exit(0);
  }
  int status=0;assert(waitpid(pid,&status,0)==pid && WIFSIGNALED(status) && WTERMSIG(status)==kNativeTrapSignal);
 }
 std::cout<<"actual parameter/FIFO native loader PASS, including independently wrapping geometry/tag words, abort-stale count zero, nonabort malformed count traps\n";
}
'''
 p=out/(backend+'-loader.cpp');p.write_text(source);argv=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(p),'-o',str(p.with_suffix(''))]
 c=subprocess.run(argv,text=True,capture_output=True);(out/(backend+'-loader-compile.log')).write_text(c.stdout+c.stderr)
 if c.returncode:raise SystemExit(c.stderr[-3000:])
 r=subprocess.run([str(p.with_suffix(''))],text=True,capture_output=True);(out/(backend+'-loader-run.log')).write_text(r.stdout+r.stderr)
 if r.returncode:raise SystemExit(r.stdout+r.stderr)
 results.append(dict(backend=backend,argv=argv,returncode=r.returncode,output=r.stdout+r.stderr))
(out/'loader-result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
