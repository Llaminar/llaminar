"""Replay actual native tuning with immutable mode and actual worker-width closure."""
from pathlib import Path
import subprocess,json,os
work=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ROOT']);r=work/'replay';source=work/'tree'/'nccl';original=Path(os.environ['LLAMINAR_NATIVE_REPLAY_ORIGINAL'])
def body(s,key):
 """Extract the complete pinned native definition without replacing its implementation."""
 a=s.index(key);b=s.index('{',a);i=b+1;depth=1
 while depth:depth+=(s[i]=='{')-(s[i]=='}');i+=1
 return s[a:i]
code=body((source/'src/enqueue.cc').read_text(),'static ncclResult_t topoGetAlgoInfo(')
pre='''#include <cassert>
#include <cstdio>
#include <cstddef>
#include <algorithm>
#define WARP_SIZE 32
#define NCCL_MAX_NTHREADS 640
#define NCCL_NUM_PROTOCOLS 3
#define NCCL_NUM_ALGORITHMS 7
#define NCCL_ALGO_PROTO_IGNORE -1
#define NCCL_ALGO_UNDEF -1
#define NCCL_PROTO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_NVLS 3
#define NCCL_ALGO_NVLS_TREE 4
#define NCCL_ALGO_PAT 5
#define NCCL_PROTO_SIMPLE 2
#define NCCL_COLL 0
#define WARN(...) ((void)0)
#define TRACE(...) ((void)0)
enum ncclResult_t{ncclSuccess,ncclInvalidUsage,ncclInternalError};
enum ncclFunc_t{ncclFuncBroadcast,ncclFuncAllReduce};
enum class ncclLlaminarPayloadKind{Rows,CompletedBytes,CompletedSourceSlice};
struct ncclTaskColl{int algorithm,protocol;ncclFunc_t func=ncclFuncBroadcast;int datatype=0;struct{ncclLlaminarPayloadKind kind;}llaminarPayload;int nMaxChannels,nWarps;};
struct ncclSimInfo_t{float estimatedTime;};
struct ncclComm{int nChannels=8;int maxThreads[7][3];int threadThresholds[7][3];struct{struct{int nHeads=1;}collnetDirect;}channels[1];int nNodes=1;int nvlsChannels=8;};
const char*ncclGetEnv(const char*){return nullptr;}
const char*ncclFuncToString(ncclFunc_t){return "Broadcast";}
const char*ncclDatatypeToString(int){return "int8";}
'''
main='''
int main(){for(int epoch=0;epoch<20;++epoch)for(int mode=0;mode<3;++mode)for(int proto=0;proto<3;++proto)for(int width:{32,64,128,256,512,640})for(size_t bytes:{size_t(0),size_t(8),size_t(48),size_t(1440),size_t(737280),size_t(1<<24)}){
ncclComm comm{};float costs[7][3];for(int a=0;a<7;++a)for(int p=0;p<3;++p){costs[a][p]=-1;comm.maxThreads[a][p]=width;comm.threadThresholds[a][p]=64;}costs[NCCL_ALGO_RING][proto]=1;
ncclTaskColl info{};info.llaminarPayload.kind=ncclLlaminarPayloadKind(mode);ncclSimInfo_t sim{};
assert(topoGetAlgoInfo(&comm,&info,bytes,reinterpret_cast<float**>(costs),&sim)==ncclSuccess);
assert(info.algorithm==NCCL_ALGO_RING && info.protocol==proto && info.nMaxChannels>=1 && info.nMaxChannels<=8 && sim.estimatedTime==1);
if(mode && !(info.nWarps*32<=128 && info.nWarps>=3)){std::fprintf(stderr,"ORIGINAL_UNBOUNDED_PLANNER\\n");return 89;}
if(!mode && bytes==(1<<24) && width==512)assert(info.nWarps*32==(proto==NCCL_PROTO_SIMPLE?544:512));
if(mode){costs[NCCL_ALGO_RING][proto]=-1;costs[NCCL_ALGO_TREE][proto]=1;
if(topoGetAlgoInfo(&comm,&info,bytes,reinterpret_cast<float**>(costs),nullptr)!=ncclInvalidUsage){std::fprintf(stderr,"ORIGINAL_UNBOUNDED_PLANNER\\n");return 89;}
costs[NCCL_ALGO_TREE][proto]=-1;costs[NCCL_ALGO_RING][proto]=1;info.func=ncclFuncAllReduce;
if(topoGetAlgoInfo(&comm,&info,bytes,reinterpret_cast<float**>(costs),nullptr)!=ncclInvalidUsage){std::fprintf(stderr,"ORIGINAL_UNBOUNDED_PLANNER\\n");return 89;}}
}
}
'''
p=r/'planner-replay.cpp';p.write_text(pre+code+main);cmd=[os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(p),'-o',str(r/'planner-replay')];subprocess.run(cmd,check=True);subprocess.run([str(r/'planner-replay')],check=True);(r/'planner-result.json').write_text(json.dumps({'argv':cmd,'passed':True,'epochs':20,'modes':3,'protocols':3,'widths':[32,64,128,256,512,640],'capacity_bytes':[0,8,48,1440,737280,1<<24],'actual_native_tuning_body':True},indent=2)+'\n');print('actual planner6480 geometries +unsupported/legacy controls GREEN')
original=body((original/'src/enqueue.cc').read_text(),'static ncclResult_t topoGetAlgoInfo(');np=r/'planner-original.cpp';np.write_text(pre+original+main);subprocess.run([os.environ['LLAMINAR_NATIVE_REPLAY_COMPILER'],'-std=c++17','-O2',str(np),'-o',str(r/'planner-original')],check=True);p=subprocess.run([str(r/'planner-original')],capture_output=True);assert p.returncode==89 and p.stderr.decode()=="ORIGINAL_UNBOUNDED_PLANNER\n";(r/'planner-original-result.json').write_text(json.dumps({'returncode':p.returncode,'expected_failure':True,'actual_original_body':True,'stderr':p.stderr.decode()},indent=2)+'\n');print('original planner RED on missing family bound')
