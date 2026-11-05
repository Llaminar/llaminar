"""Replay native completion/error bodies without loading a runtime or a model.

Primitive callbacks are CPU doubles; the actual patched Ring body, immutable
metadata resolver and completion predicate remain authoritative. Native GPU
capture/ACK tests separately qualify barriers, transport and retirement.
"""
from pathlib import Path
import json
import os
import subprocess

root = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
backend = os.environ["LLAMINAR_NATIVE_REPLAY_BACKEND"]
output = root / "replay"
main = r'''
int main() {
  constexpr size_t capacity = 512;
  unsigned char source[capacity], destination[capacity];
  for (size_t i = 0; i < capacity; ++i) source[i] = (7*i+3)%251;
  uint64_t table=17, admission=23, sequence=1, extent=0;
  uint32_t foreignAbort=0;
  int32_t error=0;
  unsigned long long byteCounter=0;
  ncclLlaminarByteCompletion receipt{};
  ncclLlaminarCompletionOperand operand{&table,&admission,&sequence,&error,
      &receipt,&byteCounter,11,37,0};
  ncclDevWorkColl work{};
  work.root=0; work.oneNode=1; work.channelLo=0;
  work.sendbuff=source; work.recvbuff=destination;
  work.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedBytes;
  work.llaminarPayload.binding.bytes={&extent,capacity,&operand};
  ncclShmem.channel.ring.userRanks[0]=1;
  ncclShmem.channel.ring.userRanks[1]=0;
  admittedCapacity=capacity; wireSource=source; ncclShmem.channelId=0;
  for (int epoch=1; epoch<=20; ++epoch) {
    for (uint64_t live : {512ull,0ull,256ull,0ull,512ull}) {
      sequence=epoch; extent=live; nativeAbort=0; ncclShmem.aborted=0;
      abortInDestructor=false; error=0;
      std::memset(destination,0xa5,capacity);
      work.llaminarPayload.binding.completed.liveBytes=extent;
      runRing<unsigned char,int,Proto>(0,1,&work);
      assert(ncclLlaminarCompletionMatches(receipt,table,admission,sequence,
          extent,11,{&nativeAbort,ncclShmem.args.comm},0,error));
      for (size_t i=0; i<capacity; ++i)
        assert(destination[i] == (i<extent ? source[i] : 0xa5));
      ncclLlaminarBeginCompletion(&work,0);
      assert(!ncclLlaminarCompletionMatches(receipt,table,admission,sequence,
          extent,11,{&nativeAbort,ncclShmem.args.comm},0,0));
      runRing<unsigned char,int,Proto>(0,1,&work);
      for (int fault=0; fault<7; ++fault) {
        auto observed=receipt;
        uint32_t aborted=0;
        int32_t first=0;
        const volatile uint32_t* expected=&nativeAbort;
        if (fault==0) observed.phase=ncclLlaminarCompletionPhase::Pending;
        if (fault==1) ++observed.admission;
        if (fault==2) ++observed.sequence;
        if (fault==3) ++observed.bytes;
        if (fault==4) expected=&foreignAbort;
        if (fault==5) aborted=1;
        if (fault==6) first=91;
        assert(!ncclLlaminarCompletionMatches(observed,table,admission,
            sequence,extent,11,{expected,ncclShmem.args.comm},aborted,first));
      }
      for (int failure=0; failure<3; ++failure) {
        error=failure==2 ? 91 : 0;
        nativeAbort=failure==0 ? 1 : 0;
        ncclShmem.aborted=0; abortInDestructor=failure!=0;
        runRing<unsigned char,int,Proto>(0,1,&work);
        assert(error == (failure==2 ? 91 : 37));
        assert(!ncclLlaminarCompletionMatches(receipt,table,admission,
            sequence,extent,11,{&nativeAbort,ncclShmem.args.comm},nativeAbort,error));
      }
      nativeAbort=0; ncclShmem.aborted=0; abortInDestructor=false;
      error=0; ncclShmem.channelId=0;
      runRing<unsigned char,int,Proto>(0,1,&work);
      assert(receipt.phase==ncclLlaminarCompletionPhase::Completed);
      ncclShmem.channelId=1; abortInDestructor=true;
      runRing<unsigned char,int,Proto>(0,1,&work);
      assert(receipt.phase==ncclLlaminarCompletionPhase::Completed && error==37);
      assert(!ncclLlaminarCompletionMatches(receipt,table,admission,
          sequence,extent,11,{&nativeAbort,ncclShmem.args.comm},nativeAbort,error));
      ncclShmem.channelId=0;
    }
  }
  std::cout << "PASS native completion:100waves,700falseproofs,300abortsticky,100latechannel\n";
}
'''
source = (output / (backend + ".cpp")).read_text().replace("int main(){", "int unusedPrefixMain(){")
path = output / (backend + "-completion.cpp")
path.write_text(source + "\n" + main)
command = [os.environ["LLAMINAR_NATIVE_REPLAY_COMPILER"], "-std=c++17", "-O2",
           str(path), "-o", str(path.with_suffix(""))]
compiled = subprocess.run(command, capture_output=True, text=True)
(output / "completion-compile.log").write_text(compiled.stdout + compiled.stderr)
if compiled.returncode:
    raise RuntimeError(compiled.stderr)
run = subprocess.run([str(path.with_suffix(""))], capture_output=True, text=True)
if run.returncode:
    raise RuntimeError(run.stdout + run.stderr)
(output / "completion-result.json").write_text(json.dumps([
    dict(backend=backend, argv=command, returncode=run.returncode,
         output=run.stdout)], indent=2) + "\n")
print(run.stdout, end="")
