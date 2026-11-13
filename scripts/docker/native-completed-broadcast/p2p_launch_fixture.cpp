/** @file probe.cpp
 * @brief Device-free replay of actual native launch sealing and publication.
 * Native plan/work/batch/connector types and the production inline bodies are
 * used directly. No CUDA runtime, NCCL DSO or copied validator is involved.
 */
#include "llaminar_p2p_launch.h"
#include <array>
#include <cassert>
#include <iostream>
#include <memory>
#include <cstddef>
ncclTransport p2pTransport{},shmTransport{},netTransport{};
extern int const ncclDevFuncRowToId[]={669};
extern int const ncclDevKernelCount=1;
extern int const ncclDevLocalP2pKernelIndex=0;
void* ncclDevKernelList[]={reinterpret_cast<void*>(0x1110)};
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
  comm->rank=0;comm->nRanks=6;comm->nNodes=1;comm->p2pnChannels=2;
  for(int c=0;c<2;++c){for(int r=0;r<6;++r){links[c][r]=&peers[c][r];auto& peer=peers[c][r];
   peer.send[1].connected=peer.recv[1].connected=1;
   peer.send[1].transportComm=&p2pTransport.send;peer.recv[1].transportComm=&p2pTransport.recv;}
   comm->channels[c].peers=links[c].data();
   batches[c].workType=ncclDevWorkTypeP2p;batches[c].funcId=ncclDevFuncId_P2p();
   batches[c].offsetBitset=(uint64_t(1)<<nworks)-1;}
  for(int i=0;i<nworks;++i){auto& node=works[i].node;node.workType=ncclDevWorkTypeP2p;node.size=sizeof(ncclDevWorkP2p);
   auto& work=works[i].descriptor;work.nP2pChannels=work.nSendChannels=work.nRecvChannels=2;
   work.sendRank=work.recvRank=i+1;work.sendBytes=work.recvBytes=256;
   work.sendProtoLL=work.recvProtoLL=1;
   ncclIntruQueueEnqueue(&plan.workQueue,&node);plan.workBytes+=sizeof(ncclDevWorkP2p);}
  plan.nWorkBatches=2;plan.channelMask=3;plan.threadPerBlock=640;
  plan.kernelFn=reinterpret_cast<void*>(0x2220);
  if(counted){task.llaminarTransfer.payload.kind=ncclLlaminarPayloadKind::Rows;task.llaminarTransfer.payload.binding.rows={reinterpret_cast<int const*>(0x1000),16,1,0};
   ncclIntruQueueEnqueue(&plan.p2pTaskQueue,&task);}
 }
 ncclP2pLaunchSeal seal(){return ncclLlaminarSealP2pLaunch(comm.get(),&plan,batches.data(),plan.nWorkBatches);}
 void good(){auto s=seal();verify(s.status==ncclP2pLaunchStatus::Sealed);verify(ncclLlaminarBindP2pLaunch(&plan,s)==ncclSuccess);
  verify(plan.threadPerBlock==256 && plan.kernelFn==ncclDevKernelList[0] && plan.kernelSpecialized);}
 void bad(ncclP2pLaunchStatus status){auto s=seal();verify(s.status==status);verify(s.requirement==ncclP2pLaunchRequirement::DeviceCounted);
  verify(ncclLlaminarBindP2pLaunch(&plan,s)==ncclInvalidUsage);verify(plan.threadPerBlock==640 && plan.kernelFn==reinterpret_cast<void*>(0x2220));}
};
int main(){
 for(int replay=0;replay<20;++replay){
  for(int works:{1,2,3,4}){Fixture f(true,works);f.good();}
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
  {Fixture f;f.plan.channelMask=2;f.plan.nWorkBatches=1;f.batches[0].offsetBitset=1;
   f.works[0].descriptor.channelBase=1;f.works[0].descriptor.nSendChannels=1;f.works[0].descriptor.nRecvChannels=0;f.good();}
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
   verify(ncclLlaminarBindP2pLaunch(&f.plan,s)==ncclSuccess);verify(f.plan.threadPerBlock==640);}
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
 {Fixture f;auto s=f.seal();ncclDevKernelList[0]=nullptr;verify(ncclLlaminarBindP2pLaunch(&f.plan,s)==ncclInvalidUsage);verify(f.plan.threadPerBlock==640);}
 std::cout<<"PASS 20 reset/replays, 43 source/geometry/connection cases per replay plus pending multi-plan retirement, "<<checks<<" assertions\n";
}
