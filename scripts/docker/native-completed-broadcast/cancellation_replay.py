#!/usr/bin/env python3
"""@file cancellation_replay.py
@brief Replay native payload resolution after an aborted count receive.

Only primitive IO and SDK functions are doubled. Exact native resolver/Ring/
completion bodies reject malformed live counts without abort and ignore the
received count after absorbing original communicator cancellation. Locally
owned source-slice descriptors retain their strict bounds checks throughout.
"""
from pathlib import Path
import json
import os
import subprocess

root = Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT'])
backend = os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']
out = root / 'replay'
main = r'''
#include <sys/resource.h>
#include <csignal>
// __builtin_trap() raises the host ISA's trap: ud2/SIGILL on x86-64, brk/SIGTRAP on AArch64.
#if defined(__aarch64__)
constexpr int kNativeTrapSignal=SIGTRAP;
#else
constexpr int kNativeTrapSignal=SIGILL;
#endif
template<class F> void mustTrap(F action){
 auto pid=fork();assert(pid>=0);if(!pid){action();_exit(0);}
 int status=0;assert(waitpid(pid,&status,0)==pid && WIFSIGNALED(status) && WTERMSIG(status)==kNativeTrapSignal);
}
int main(){
 rlimit noCore{0,0};assert(setrlimit(RLIMIT_CORE,&noCore)==0);
 constexpr size_t capacity=512;unsigned char source[capacity],destination[capacity];
 uint64_t table=0,admission=0,sequence=0,extent=~uint64_t(0);
 int32_t error=18;unsigned long long observer=0;
 ncclLlaminarByteCompletion receipt{};
 ncclLlaminarCompletionOperand operand{&table,&admission,&sequence,&error,&receipt,&observer,11,37,0};
 ncclDevWorkColl work{};work.root=0;work.oneNode=1;work.channelLo=0;
 work.sendbuff=source;work.recvbuff=destination;
 ncclShmem.channel.ring.userRanks[0]=1;ncclShmem.channel.ring.userRanks[1]=0;
 admittedCapacity=capacity;wireSource=source;ncclShmem.channelId=0;abortInDestructor=false;
 for(int repeat=0;repeat<20;++repeat){
  // The count receive has already terminated on the stream. Its stale UINT64
  // must not be interpreted as a count after original abort was published.
  extent=~uint64_t(0);nativeAbort=1;ncclShmem.aborted=0;error=18;
  work.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedBytes;
  work.llaminarPayload.binding.bytes={&extent,capacity,&operand};
  work.llaminarPayload.binding.resolved.liveBytes=ncclLlaminarResolvePayloadWord(
      reinterpret_cast<uint64_t>(&extent),capacity,1,&nativeAbort);
  assert(work.llaminarPayload.binding.resolved.liveBytes==0);
  runRing<unsigned char,int,Proto>(0,1,&work);
  assert(error==18 && receipt.phase==ncclLlaminarCompletionPhase::Aborted && receipt.bytes==0);
  nativeAbort=0;
  mustTrap([&]{ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&extent),capacity,1,&nativeAbort);});
  // Abort is no permission to forgive immutable mode/reserved/pointer/geometry.
  nativeAbort=1;
  mustTrap([&]{ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&extent),capacity,1ull<<32,&nativeAbort);});
  mustTrap([&]{ncclLlaminarResolvePayloadWord(0,capacity,1,&nativeAbort);});
  mustTrap([&]{ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&extent),0,1,&nativeAbort);});
  // Ordinary rows retain their exact original contract even after native abort.
  int32_t rows=3;const uint64_t geometry=(uint64_t(9)<<32)|5;
  assert(ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&rows),geometry,0,&nativeAbort)==15);
  rows=-1;mustTrap([&]{ncclLlaminarResolvePayloadWord(reinterpret_cast<uint64_t>(&rows),geometry,0,&nativeAbort);});
  // Local source-slice descriptors were not received over the canceled wire:
  // their original bounds stay strict even with native cancellation set.
  ncclLlaminarSourceSlice slice{~uint64_t(0),~uint64_t(0)};
  ncclLlaminarDeviceSlice binding{&slice,capacity,&operand};
  mustTrap([&]{ncclLlaminarResolveSourceSlice(binding,capacity,17);});
  slice={17,0};const auto empty=ncclLlaminarResolveSourceSlice(binding,capacity,17);
  assert(empty.byteOffset==0 && empty.liveBytes==0 && nativeAbort==1);
  ncclShmem.channel.ring.userRanks[0]=1;ncclShmem.channel.ring.userRanks[1]=0;
 }
 std::cout<<"PASS native cancellation:20 stale counts,20 aborted zero receipts,120 nonabort or immutable traps,unchanged rows and source bounds\n";
}
'''
source = (out / (backend + '.cpp')).read_text().replace('int main(){', 'int unusedPrefixMain(){')
path = out / (backend + '-cancellation.cpp')
path.write_text(source + '\n' + main)
command = [os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'], '-std=c++17', '-O2', str(path), '-o', str(path.with_suffix(''))]
subprocess.run(command, check=True)
run = subprocess.run([str(path.with_suffix(''))], capture_output=True, text=True, check=True)
rows = [dict(backend=backend, returncode=run.returncode, output=run.stdout)]
(out / 'cancellation-result.json').write_text(json.dumps(rows, indent=2) + '\n')
print(run.stdout, end='')
