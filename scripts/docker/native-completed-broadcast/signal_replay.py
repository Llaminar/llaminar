"""Replay actual native signal-only wrapper and unchanged original abort-store body."""
from pathlib import Path
import subprocess,json,os
r=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"]);o=r/'replay';o.mkdir(exist_ok=True)
def body(s,key):
 """Extract the complete pinned native definition without replacing its implementation."""
 a=s.index(key);i=s.index('{',a);n=1;j=i+1
 while n:n+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
results=[]
for backend in [os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND']]:
 source=(r/'tree'/backend/'src/init.cc').read_text();helper=body(source,'static ncclResult_t setCommAbortFlags(');actual='ncclResult_t '+body(source,'llaminarNcclSignalCompletionAbort(')
 program='''#include <cstdint>
#include <cassert>
#include <iostream>
using ncclResult_t=int;constexpr int ncclSuccess=0,ncclInvalidArgument=1;
struct Comm {uint32_t*abortFlag;uint32_t*abortFlagDev;uint32_t*childAbortFlag;uint32_t*childAbortFlagDev;int destroyFlag=73;};
using ncclComm_t=Comm*;
'''+helper+'\n'+actual+'''
int main(){uint32_t host=0,device=0,childHost=0,childDevice=0;
Comm original{&host,&device,&childHost,&childDevice};
for(int repeat=0;repeat<20;++repeat){assert(llaminarNcclSignalCompletionAbort(&original)==0);
assert(host==1&&device==1&&childHost==1&&childDevice==1&&original.destroyFlag==73);}
assert(llaminarNcclSignalCompletionAbort(nullptr)==ncclInvalidArgument);
original.abortFlagDev=nullptr;host=0;assert(llaminarNcclSignalCompletionAbort(&original)==ncclInvalidArgument&&host==0);
std::cout<<"actual original release-store signal PASS:20absorbing repeats,parent+child words,no destroy/reset/allocation"<<std::endl;}
''';p=o/(backend+'-signal.cpp');p.write_text(program);cmd=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(p),'-o',str(p.with_suffix(''))];subprocess.run(cmd,check=True);q=subprocess.run([str(p.with_suffix(''))],capture_output=True,text=True,check=True);results.append(dict(backend=backend,argv=cmd,returncode=q.returncode,output=q.stdout))
(o/'signal-result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
