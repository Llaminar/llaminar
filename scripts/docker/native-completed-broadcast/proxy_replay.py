#!/usr/bin/env python3
"""Exercise original SHM progress bodies with exact SDK memcpy/event doubles."""
from pathlib import Path
import subprocess,json,hashlib,os
root=Path(os.environ["LLAMINAR_NATIVE_REPLAY_ROOT"]);out=root/'replay'
def body(s,signature):
 """Extract the complete pinned native definition without replacing its implementation."""
 a=s.index(signature);i=s.index('{',a);n=1;j=i+1
 while n:n+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
prefix=r'''
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>
#include <iostream>
constexpr int ncclSuccess=0,NCCL_STEPS=8,NCCL_PROTO_SIMPLE=0,ncclProxyOpReady=1,ncclProxyOpProgress=2,ncclProxyOpNone=3;
using ncclResult_t=int;using cudaError_t=int;
constexpr int cudaSuccess=0,cudaErrorNotReady=1,cudaMemcpyDeviceToHost=0,cudaMemcpyHostToDevice=1;
#define ROUNDUP(x,y) (((x)+(y)-1)/(y)*(y))
#define CUDACHECK(e) do{int r=(e);if(r)return r;}while(0)
struct ncclConnFifo{int size;};
struct Mem{ncclConnFifo connFifo[NCCL_STEPS]{};uint64_t tail=0;};
struct shmProxyInfo{uint64_t step=0;Mem*ceRecvMem;Mem*recvMem;char*shmFifo;char*devFifo;void*stream;int events[NCCL_STEPS]{};};
struct Connection{void*transportResources;};
struct ncclProxySubArgs{Connection*connection;uint64_t base=0,posted=0,transmitted=0,done=0,nsteps=0;};
struct ncclProxyArgs{int state=ncclProxyOpReady,idle=0,protocol=0,nsubs=1,done=0,sliceSteps=1,chunkSteps=1;ncclProxySubArgs*subs;};
struct ncclProxyState{int buffSizes[1]{128};};
std::vector<size_t>copySizes;
int cudaMemcpyAsync(void*d,const void*s,size_t n,int,void*stream){assert(stream);copySizes.push_back(n);std::memcpy(d,s,n);return 0;}
int cudaEventRecord(int,void*s){assert(s);return 0;}
int cudaEventQuery(int){return 0;}
'''
tests=r'''
int main(){
 for(bool send:{true,false})for(int epoch=0;epoch<20;++epoch)for(int live:{48,0,17,1,48,3}){
  char dev[128],host[128];std::memset(dev,0x31,sizeof(dev));std::memset(host,0xa5,sizeof(host));
  if(!send){std::memset(dev,0xa5,sizeof(dev));std::memset(host,0x31,sizeof(host));}
  Mem ce{},rx{};shmProxyInfo r{0,&ce,&rx,host,dev,reinterpret_cast<void*>(16),{}};
  Connection c{&r};ncclProxySubArgs sub{};sub.connection=&c;sub.nsteps=3;
  ncclProxyArgs args{};args.subs=&sub;ncclProxyState state{};
  auto*producer=send?&ce:&rx;producer->tail=3;
  size_t useful=0;
  for(int i=0;i<3;++i){int n=live-i*16;n=n<0?0:n>16?16:n;producer->connFifo[i].size=n;useful+=n;}
  copySizes.clear();
  for(int i=0;i<4&&args.state!=ncclProxyOpNone;++i){auto e=send?shmSendProxyProgress(&state,&args):shmRecvProxyProgress(&state,&args);assert(e==0);}
  assert(args.state==ncclProxyOpNone&&sub.done==3&&r.step==3&&args.done==1);
  size_t copied=0;for(auto n:copySizes)copied+=n;
  assert(copied==useful&&useful==size_t(live)&&copySizes.size()==3);
  auto*dest=send?host:dev;for(int i=0;i<128;++i)assert(static_cast<unsigned char>(dest[i])==(i<live?0x31:0xa5));
  assert((send?rx.tail:ce.tail)==3);
 }
 std::cout<<"actual SHM send/receive progress PASS: 20 epochs, 0/partial/full, exact memcpy bytes and all capacity steps\n";
}
'''
results=[]
for backend in (os.environ['LLAMINAR_NATIVE_REPLAY_BACKEND'],):
 s=(root/'tree'/backend/'src/transport/shm.cc').read_text()
 source=prefix+'\n'+body(s,'static ncclResult_t shmSendProxyProgress(')+'\n'+body(s,'static ncclResult_t shmRecvProxyProgress(')+'\n'+tests
 p=out/(backend+'-proxy.cpp');p.write_text(source);argv=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(p),'-o',str(p.with_suffix(''))]
 c=subprocess.run(argv,capture_output=True,text=True);(out/(backend+'-proxy-compile.log')).write_text(c.stdout+c.stderr)
 if c.returncode:raise SystemExit(c.stderr[-5000:])
 r=subprocess.run([str(p.with_suffix(''))],capture_output=True,text=True);(out/(backend+'-proxy-run.log')).write_text(r.stdout+r.stderr)
 results.append(dict(backend=backend,argv=argv,source_sha256=hashlib.sha256(source.encode()).hexdigest(),returncode=r.returncode,output=r.stdout+r.stderr))
 if r.returncode:raise SystemExit(r.stdout+r.stderr)
(out/'proxy-result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
