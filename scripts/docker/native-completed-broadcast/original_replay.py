"""Compile the literal reverse-patched fixed Ring as the completion negative.

Only its declaration name/conditional declaration wrapper is adapted to the
shared CPU fixture. The original function body is unchanged. Missing symbols,
compiler errors and unrelated crashes cannot satisfy this control.
"""
from pathlib import Path
import json
import os
import subprocess


def function(text, signature):
    """Extract the complete pinned function; preserve its original body bytes."""
    start = text.index(signature)
    brace = text.index("{", start)
    cursor = brace + 1
    depth = 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[start:cursor]


root = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"])
backend = os.environ["LLAMINAR_NATIVE_REPLAY_BACKEND"]
original = Path(os.environ["LLAMINAR_NATIVE_REPLAY_ORIGINAL"])
output = root / "replay"
text = (original / "src/device/broadcast.h").read_text()
if backend == "nccl":
    native = function(text, "__device__ __forceinline__ void runRing(")
else:
    native = function(text, "__device__ __attribute__((noinline)) void runRing(")
    # The original RCCL source has two alternate declarations around one body.
    # Extraction starts at the selected declaration, after the opening #else.
    prefix, separator, remainder = native.partition("{\n#endif\n")
    if not separator:
        raise RuntimeError("Original RCCL conditional declaration changed")
    native = prefix + "{\n" + remainder
native = native.replace("void runRing(", "void runRingOriginal(", 1)
source = (output / (backend + ".cpp")).read_text().replace("int main(){", "int unusedNativeMain(){")
main = r'''
int main() {
  constexpr size_t capacity=512;
  unsigned char source[capacity], destination[capacity];
  for (size_t i=0; i<capacity; ++i) source[i]=(7*i+3)%251;
  std::memset(destination,0xa5,capacity);
  uint64_t table=17, admission=23, sequence=1;
  int32_t error=0;
  unsigned long long bytes=0;
  ncclLlaminarByteCompletion receipt{};
  ncclLlaminarCompletionOperand operand{&table,&admission,&sequence,&error,
      &receipt,&bytes,11,37,0};
  ncclDevWorkColl work{};
  work.root=0; work.oneNode=1; work.channelLo=0;
  work.sendbuff=source; work.recvbuff=destination;
  work.llaminarPayload.kind=ncclLlaminarPayloadKind::CompletedBytes;
  work.llaminarPayload.binding.completed.liveBytes=0;
  work.llaminarPayload.binding.completed.completion=&operand;
  nativeAbort=0; ncclShmem.aborted=0; abortInDestructor=false;
  admittedCapacity=capacity; wireSource=source;
  ncclShmem.channel.ring.userRanks[0]=1;
  ncclShmem.channel.ring.userRanks[1]=0;
  runRingOriginal<unsigned char,int,Proto>(0,1,&work);
  if (!ncclLlaminarCompletionMatches(receipt,table,admission,sequence,0,
      11,{&nativeAbort,ncclShmem.args.comm},0,error)) {
    std::cerr << "ORIGINAL_MISSING_COMPLETION\n";
    return 87;
  }
  return 0;
}
'''
path = output / (backend + "-original.cpp")
path.write_text(source + "\ntemplate<typename T,typename RedOp,typename Proto>\n" + native + "\n" + main)
command = [os.environ["LLAMINAR_NATIVE_REPLAY_COMPILER"], "-std=c++17", "-O2",
           str(path), "-o", str(path.with_suffix(""))]
compiled = subprocess.run(command, capture_output=True, text=True)
(output / "original-compile.log").write_text(compiled.stdout + compiled.stderr)
if compiled.returncode:
    raise RuntimeError(compiled.stderr)
runs = []
for repetition in range(20):
    run = subprocess.run([str(path.with_suffix(""))], capture_output=True, text=True)
    row = dict(returncode=run.returncode, stdout=run.stdout, stderr=run.stderr)
    if row != dict(returncode=87, stdout="", stderr="ORIGINAL_MISSING_COMPLETION\n"):
        raise RuntimeError("Original negative failed for an unrelated reason: " + repr(row))
    runs.append(row)
(output / "original-result.json").write_text(json.dumps(dict(
    argv=command, compile_exit=0, runs=runs), indent=2) + "\n")
print("PASS original fixed Ring:20 missing-completion witnesses")
