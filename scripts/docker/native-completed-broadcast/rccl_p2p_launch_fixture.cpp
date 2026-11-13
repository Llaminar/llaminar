/** @file rccl_p2p_launch_fixture.cpp
 * @brief Device-free replay of actual native launch sealing and publication.
 * Native plan/work/batch/connector types and the production inline bodies are
 * used directly. Each active half validates its serialized connector index;
 * inactive halves must not borrow or constrain another connector. No CUDA
 * runtime, NCCL DSO or copied validator is involved.
 */
#include "llaminar_p2p_launch.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <cstddef>
#include <string_view>
ncclTransport p2pTransport{},shmTransport{},netTransport{};
std::unordered_map<uint64_t,int> ncclDevFuncNameToId{{uint64_t(ncclFuncSendRecv)<<RCCL_COLL_SHIFT,669}};
extern int const ncclLocalP2pKernelCount=6;
extern ncclLocalP2pKernelEntry const ncclLocalP2pKernelList[]={
 {0,false,reinterpret_cast<void*>(0x1110)}, {1,false,reinterpret_cast<void*>(0x1120)},
 {2,false,reinterpret_cast<void*>(0x1130)}, {0,true,reinterpret_cast<void*>(0x1210)},
 {1,true,reinterpret_cast<void*>(0x1220)}, {2,true,reinterpret_cast<void*>(0x1230)}};
void ncclDebugLog(ncclDebugLogLevel,unsigned long,const char*,int,const char*,...){}

// No trace, runtime, stream or allocation is created by this CPU fixture.
namespace latency_profiler { CollTrace::~CollTrace() {} }
// Actual native type destructors see only their value-initialized empty events.
// Reject any attempt to claim that this device-free replay owns a live handle.
extern "C" hipError_t hipEventDestroy(hipEvent_t event) {
 if(event)std::abort();return hipSuccess;
}
int checks=0;
void verify(bool condition){if(!condition)std::abort();++checks;}
ncclResult_t proxy(ncclProxyState*,ncclProxyArgs*) {return ncclSuccess;}
struct Work {ncclWorkList node{};ncclDevWorkP2p descriptor{};};
static_assert(offsetof(Work,descriptor)==sizeof(ncclWorkList));
struct Fixture {
 std::unique_ptr<ncclComm> comm=std::make_unique<ncclComm>();
 ncclKernelPlan plan{};
 std::array<Work,5> works{};
 std::array<ncclDevWorkBatch,3> batches{};
 std::array<std::array<ncclChannelPeer,6>,2> peers{};
 std::array<std::array<ncclChannelPeer*,6>,2> links{};
 ncclTaskP2p task{};
 explicit Fixture(bool counted=true,int nworks=1) {
  comm->rank=0;comm->nRanks=6;comm->nNodes=1;comm->p2pnChannels=2;comm->p2pnChannelsPerPeer=2;comm->WarpSize=64;
  for(int c=0;c<2;++c){for(int r=0;r<6;++r){links[c][r]=&peers[c][r];auto& peer=peers[c][r];
   peer.send[1].connected=peer.recv[1].connected=1;
   peer.send[1].transportComm=&p2pTransport.send;peer.recv[1].transportComm=&p2pTransport.recv;}
   comm->channels[c].peers=links[c].data();
   batches[c].workType=ncclDevWorkTypeP2p;batches[c].funcId=ncclDevFuncId_P2p();
   batches[c].offsetBitset=(uint64_t(1)<<nworks)-1;}
  for(int i=0;i<nworks;++i){auto& node=works[i].node;node.workType=ncclDevWorkTypeP2p;node.size=sizeof(ncclDevWorkP2p);
   auto& work=works[i].descriptor;work.nP2pChannels=work.nSendChannels=work.nRecvChannels=2;
   work.sendConnIndex=work.recvConnIndex=1; // Actual original planner default.
   work.sendRank=work.recvRank=i+1;work.sendBytes=work.recvBytes=256;
   work.sendProtoLL=work.recvProtoLL=1;
   ncclIntruQueueEnqueue(&plan.workQueue,&node);plan.workBytes+=sizeof(ncclDevWorkP2p);}
  plan.nWorkBatches=2;plan.channelMask.masks[0]=3;plan.threadPerBlock=640;
  plan.kernelFn=reinterpret_cast<void*>(0x2220);
  if(counted){task.llaminarTransfer.payload.kind=ncclLlaminarPayloadKind::Rows;task.llaminarTransfer.payload.binding.rows={reinterpret_cast<int const*>(0x1000),16,1,0};
   ncclIntruQueueEnqueue(&plan.p2pTaskQueue,&task);}
 }
 ncclP2pLaunchSeal seal(){return ncclLlaminarSealP2pLaunch(comm.get(),&plan,batches.data(),plan.nWorkBatches);}
 void good(){auto s=seal();verify(s.status==ncclP2pLaunchStatus::Sealed);verify(ncclLlaminarBindP2pLaunch(&plan,s,ncclLlaminarLocalP2pKernelFor(comm.get()))==ncclSuccess);
  verify(plan.threadPerBlock==256 && plan.kernelFn==ncclLlaminarLocalP2pKernelFor(comm.get()) && plan.kernelSpecialized);}
 void bad(ncclP2pLaunchStatus status){auto s=seal();verify(s.status==status);verify(s.requirement==ncclP2pLaunchRequirement::DeviceCounted);
  verify(ncclLlaminarBindP2pLaunch(&plan,s,ncclLlaminarLocalP2pKernelFor(comm.get()))==ncclInvalidUsage);verify(plan.threadPerBlock==640 && plan.kernelFn==reinterpret_cast<void*>(0x2220));}
};
/** @brief Original selected-connector failure states; none changes native transport selection. */
enum class SelectedConnectionFailure { Disconnected, Network, Proxy };

/**
 * @brief Replay each active work half against its actual selected native connector.
 * Alternate local connectors must ignore poisoned unselected connector1.
 * Selected disconnection/NET/proxy and out-of-range indices reject before
 * publication; inactive halves retain their original unconstrained behavior.
 */
void replaySelectedWorkConnectors() {
 for(int replay=0;replay<20;++replay)for(bool send:{false,true}) {
  {Fixture f;
   auto& work=f.works[0].descriptor;
   if(send)work.sendConnIndex=0;else work.recvConnIndex=0;
   for(auto& channel:f.peers)for(auto& peer:channel) {
    auto& selected=send?peer.send[0]:peer.recv[0];
    auto& unselected=send?peer.send[1]:peer.recv[1];
    selected.connected=1;
    selected.transportComm=send?&p2pTransport.send:&shmTransport.recv;
    unselected.transportComm=send?&netTransport.send:&netTransport.recv;
    unselected.proxyConn.proxyProgress=&proxy;
   }
   f.good();}
  for(const auto failure:{SelectedConnectionFailure::Disconnected,
      SelectedConnectionFailure::Network,SelectedConnectionFailure::Proxy}) {
   Fixture f;
   auto& work=f.works[0].descriptor;
   if(send)work.sendConnIndex=0;else work.recvConnIndex=0;
   for(auto& channel:f.peers)for(auto& peer:channel) {
    auto& selected=send?peer.send[0]:peer.recv[0];
    selected.connected=failure!=SelectedConnectionFailure::Disconnected;
    selected.transportComm=failure==SelectedConnectionFailure::Network
        ?(send?&netTransport.send:&netTransport.recv)
        :(send?&p2pTransport.send:&shmTransport.recv);
    if(failure==SelectedConnectionFailure::Proxy)selected.proxyConn.proxyProgress=&proxy;
   }
   f.bad(ncclP2pLaunchStatus::Connection);
  }
  {Fixture f;
   auto& work=f.works[0].descriptor;
   if(send)work.sendConnIndex=NCCL_MAX_CONNS;else work.recvConnIndex=NCCL_MAX_CONNS;
   f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;
   auto& work=f.works[0].descriptor;
   if(send)work.sendConnIndex=NCCL_MAX_CONNS;else work.recvConnIndex=NCCL_MAX_CONNS;
   (send?work.nSendChannels:work.nRecvChannels)=0;
   f.good();}
 }
}

int main(int argc,char** argv){
 if(argc>1) {
  if(argc!=2 || std::string_view(argv[1])!="--selected-connectors-only")return 64;
  replaySelectedWorkConnectors();
  std::cout<<"PASS actual selected RCCL connectors: 240 cases, "<<checks<<" assertions\n";
  return 0;
 }
 replaySelectedWorkConnectors();
 for(int replay=0;replay<20;++replay){
  for(int width:{32,64})for(int works:{1,2,3,4}){Fixture f(true,works);f.comm->WarpSize=width;
   if(works<=256/(2*width))f.good();else f.bad(ncclP2pLaunchStatus::Geometry);}
  for(int unroll:{0,1,2}){Fixture f;f.comm->unroll=unroll;f.good();}
#ifdef ENABLE_COLLTRACE
  for(int unroll:{0,1,2}){Fixture f;f.comm->unroll=unroll;f.comm->collTraceEnabled=true;f.good();}
#endif
  {Fixture f;f.comm->p2pnChannels=4;f.plan.channelMask.masks[0]=12;
   f.comm->channels[2].peers=f.links[0].data();f.comm->channels[3].peers=f.links[1].data();
   f.works[0].descriptor.nP2pChannels=4;f.works[0].descriptor.channelBase=1;f.good();}
  {Fixture f;f.comm->WarpSize=16;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.comm->p2pnChannelsPerPeer=0;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f(false);f.works[0].descriptor.sendTransfer.payload.kind=ncclLlaminarPayloadKind::CompletedBytes;
   f.works[0].descriptor.sendTransfer.payload.binding.bytes.bytes=nullptr;
   f.peers[0][1].recv[1].transportComm=&netTransport.recv;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.plan.channelMask.masks[0]=0;f.plan.channelMask.masks[1]=uint64_t(1)<<1;f.plan.nWorkBatches=1;
   f.comm->p2pnChannels=128;f.comm->p2pnChannelsPerPeer=1;f.comm->channels[65].peers=f.links[0].data();
   auto&w=f.works[0].descriptor;w.nP2pChannels=128;w.channelBase=65;w.nSendChannels=w.nRecvChannels=1;f.good();}
  {Fixture f(false);f.works[0].descriptor.sendBytes=f.works[0].descriptor.recvBytes=4;f.good();} // Original fixed metadata has no completed receipt.
  for(const auto kind:{ncclLlaminarPayloadKind::CompletedBytes,ncclLlaminarPayloadKind::CompletedSourceSlice}) {
   {Fixture f;f.task.llaminarTransfer.payload.kind=kind;f.good();}
   {Fixture f;f.task.llaminarTransfer.payload.kind=kind;f.task.llaminarTransfer.payload.binding.bytes.bytes=nullptr;
    f.peers[0][1].recv[1].transportComm=&netTransport.recv;f.bad(ncclP2pLaunchStatus::Connection);}
  }
  {Fixture f;f.works[0].descriptor.sendBytes=f.works[0].descriptor.recvBytes=0;f.good();}
  {Fixture f;f.works[0].descriptor.recvProtoLL=f.works[0].descriptor.sendProtoLL=0;f.good();}
  {Fixture f;for(auto& ch:f.peers)for(auto& peer:ch){peer.send[1].transportComm=&shmTransport.send;peer.recv[1].transportComm=&shmTransport.recv;}f.good();}
  {Fixture f;f.works[0].descriptor.sendRank=f.works[0].descriptor.recvRank=0;f.good();}
  {Fixture f;f.works[0].descriptor.nRecvChannels=0;f.peers[0][1].recv[1].transportComm=&netTransport.recv;f.good();}
  {Fixture f;f.plan.channelMask.masks[0]=2;f.plan.nWorkBatches=1;f.batches[0].offsetBitset=1;
   f.comm->p2pnChannelsPerPeer=1;f.works[0].descriptor.channelBase=1;f.works[0].descriptor.nSendChannels=1;f.works[0].descriptor.nRecvChannels=0;f.good();}
  {Fixture f;f.plan.nWorkBatches=3;f.batches[2]=f.batches[0];f.batches[0].nextJump=2;f.good();}
  {Fixture f;f.batches[1].funcId=1;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f;f.batches[1].workType=ncclDevWorkTypeColl;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f;f.works[0].node.workType=ncclDevWorkTypeColl;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f(true,5);f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.batches[0].nextExtends=1;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.batches[0].offsetBitset=0;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.plan.workBytes=sizeof(ncclDevWorkP2p)-1;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f;f.batches[0].offsetBase=16;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f;f.batches[0].offsetBase=f.plan.workBytes;f.bad(ncclP2pLaunchStatus::ForeignWork);}
  {Fixture f;f.comm->nNodes=2;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.comm->p2pnChannels=3;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.works[0].descriptor.nP2pChannels=1;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.works[0].descriptor.channelBase=2;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.works[0].descriptor.nSendChannels=3;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.works[0].descriptor.nSendChannels=f.works[0].descriptor.nRecvChannels=0;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.batches[0].nextJump=1;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.plan.nWorkBatches=3;f.bad(ncclP2pLaunchStatus::Geometry);}
  {Fixture f;f.works[0].descriptor.sendRank=6;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[1][1].send[1].connected=0;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[1][1].send[1].proxyConn.proxyProgress=&proxy;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[0][1].recv[1].transportComm=&netTransport.recv;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.works[0].descriptor.sendNetReg=1;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[0][1].recv[1].conn.flags=NCCL_DIRECT_NIC;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[0][1].recv[1].conn.netDeviceHandle.netDeviceType=NCCL_NET_DEVICE_UNPACK;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.peers[1][1].send[1].conn.netDeviceHandle.handle=reinterpret_cast<void*>(1);f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f;f.links[1][1]=nullptr;f.bad(ncclP2pLaunchStatus::Connection);}
  {Fixture f(false);f.batches[0].funcId=1;auto s=f.seal();verify(s.status==ncclP2pLaunchStatus::ForeignWork);
   verify(ncclLlaminarBindP2pLaunch(&f.plan,s,ncclLlaminarLocalP2pKernelFor(f.comm.get()))==ncclSuccess);verify(f.plan.threadPerBlock==640);}
 }
 for(int replay=0;replay<20;++replay) {
  ncclKernelPlanner planner{};ncclKernelPlan first{},second{},rejected{};
  first.persistent=second.persistent=rejected.persistent=true;
  ncclIntruQueueEnqueue(&planner.planQueue,&first);ncclIntruQueueEnqueue(&planner.planQueue,&second);
  planner.unlaunchedPlansHead=&first;int retired=0;std::array<ncclKernelPlan*,3> seen{};
  const auto result=ncclLlaminarRetirePendingP2pPlanSet(&planner,&rejected,[&](ncclKernelPlan* plan){
   seen[retired++]=plan;return plan==&rejected?ncclInvalidUsage:ncclSuccess;});
  verify(result==ncclInvalidUsage && retired==3);verify(seen[0]==&rejected && seen[1]==&first && seen[2]==&second);
  verify(!planner.planQueue.head && !planner.planQueue.tail && !planner.unlaunchedPlansHead);
 }
 {Fixture f;auto s=f.seal();verify(ncclLlaminarBindP2pLaunch(&f.plan,s,nullptr)==ncclInvalidUsage);verify(f.plan.threadPerBlock==640);}
 {Fixture f;f.comm->unroll=3;verify(ncclLlaminarLocalP2pKernelFor(f.comm.get())==nullptr);}
 std::cout<<"PASS 20 reset/replays, actual wave32/64, unroll/trace, high-channel and source/geometry/connection cases per replay plus pending multi-plan retirement, "<<checks<<" assertions\n";
}
