/**
 * @file rccl_small_allgather_probe.cpp
 * @brief Execute actual native small-message admission and enqueue selection on host metadata.
 *
 * The builder includes the original enqueue translation unit and links original
 * native tuning/protocol helpers with their real compile inputs. No GPU API,
 * communicator initialization, transport callback or model executes. Synthetic
 * descriptors are only functional inputs, never native ownership/capacity proof.
 * The negative variant changes only the exact enqueue implementation, retaining
 * the current private ABI and selected dependency DSO for every host helper.
 */
#include LLAMINAR_RCCL_ENQUEUE_SOURCE
#include "llaminar_small_allgather_scope.h"
#include "SmallAllGatherPolicy.h"
#include <array>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// A typed transport identity fixture only. No callback is installed or invoked.
// Probe symbols are localized by a version script, never interposing on the DSO.
ncclTransport shmTransport{};

/** @brief Host-only logging boundary for the actual compiled native consumers. */
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}

namespace {
/** @brief Retain every assertion and actual selection; failures never truncate the matrix. */
struct Case {
  const char* name;
  int checks = 0, failures = 0;
  std::array<int,6> observed{};
  /** @brief Count one independent expected outcome, even in the old-code negative. */
  void check(bool value) { ++checks; failures += !value; }
};

/** @brief Actual original descriptor types, independently owned per communicator rank. */
struct Clique {
  std::array<std::unique_ptr<ncclComm>,4> comms;
  std::array<std::unique_ptr<ncclTopoSystem>,4> topologies;
  std::array<std::array<ncclPeerInfo,4>,4> info{};
  std::array<std::array<std::array<ncclChannelPeer,4>,4>,4> endpoints{};
  std::array<std::array<std::array<ncclChannelPeer*,4>,4>,4> peers{};
  std::array<ncclComm_t,4> owners{};

  /** @brief Construct complete permuted topology and original connected SHM endpoint facts. */
  Clique() {
    constexpr int order[]{3,0,2,1};
    for (int rank=0; rank<4; ++rank) {
      comms[rank]=std::make_unique<ncclComm>();
      topologies[rank]=std::make_unique<ncclTopoSystem>();
      owners[rank]=comms[rank].get();
      auto& comm=*comms[rank];
      comm.rank=rank;comm.nRanks=comm.localRanks=comm.intraRanks=comm.maxLocalRanks=4;
      comm.nNodes=1;comm.nChannels=4;comm.WarpSize=64;comm.commHash=42;
      comm.config.blocking=1;comm.config.CTAPolicy=NCCL_CTA_POLICY_DEFAULT;comm.initState=ncclSuccess;
      comm.topo=topologies[rank].get();comm.peerInfo=info[rank].data();
      comm.topo->nodes[GPU].count=4;
      for (int slot=0; slot<4; ++slot) {
        auto& gpu=comm.topo->nodes[GPU].nodes[slot].gpu;
        gpu.rank=order[slot];std::strcpy(gpu.gcn,"gfx906");
      }
      for (int channel=0; channel<4; ++channel) {
        for (int peer=0; peer<4; ++peer) peers[rank][channel][peer]=&endpoints[rank][channel][peer];
        comm.channels[channel].peers=peers[rank][channel].data();
        comm.channels[channel].ring.prev=(rank+3)%4;comm.channels[channel].ring.next=(rank+1)%4;
        auto& send=endpoints[rank][channel][(rank+1)%4].send[0];
        auto& recv=endpoints[rank][channel][(rank+3)%4].recv[0];
        send.connected=recv.connected=1;
        send.transportComm=&shmTransport.send;recv.transportComm=&shmTransport.recv;
      }
      for (int algorithm=0; algorithm<NCCL_NUM_ALGORITHMS; ++algorithm)
        for (int protocol=0; protocol<NCCL_NUM_PROTOCOLS; ++protocol) {
          comm.maxThreads[algorithm][protocol]=256;
          comm.threadThresholds[algorithm][protocol]=1;
        }
      // Authored native tuning inputs make original SIMPLE strictly cheaper.
      // The actual ncclTopoGetAlgoTime computes costs; no table implementation
      // or native algorithm/protocol/postpass is replaced in this probe.
      for (int function=0; function<NCCL_NUM_FUNCTIONS; ++function) {
        comm.bandwidths[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=1.0f;
        comm.bandwidths[function][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]=2.0f;
        comm.latencies[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=40.0f;
        comm.latencies[function][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]=10.0f;
      }
    }
    for (int owner=0; owner<4; ++owner)
      for (int rank=0; rank<4; ++rank) {
        auto& peer=info[owner][rank];peer.rank=rank;peer.hostHash=11;peer.pidHash=22;peer.comm=owners[rank];
      }
  }

  /** @return Exact current owner flags, retaining an intentionally nonuniform no-op witness. */
  std::array<bool,4> flags() const {
    std::array<bool,4> value{};
    for (int rank=0; rank<4; ++rank) value[rank]=comms[rank]->llaminarSmallAllGatherLL;
    return value;
  }
};

/** @brief Verify actual helper publication is unanimous and no partial write can occur. */
std::vector<Case> scopeCases() {
  std::vector<Case> rows;
  const auto rejected=[&](const char* name, auto mutate, bool complete=true, bool absent=true, int count=4) {
    auto clique=std::make_unique<Clique>();
    clique->owners[1]->llaminarSmallAllGatherLL=true;
    const auto before=clique->flags();mutate(*clique);
    Case row{name};
    row.check(!llaminarPublishSmallAllGatherPolicy(clique->owners.data(),count,complete,absent));
    row.check(clique->flags()==before);rows.push_back(row);
  };
  {
    auto clique=std::make_unique<Clique>();Case row{"scope_complete_clique"};
    row.check(llaminarSmallAllGatherCliqueScope(clique->owners.data(),4,true));
    row.check(llaminarPublishSmallAllGatherPolicy(clique->owners.data(),4,true,true));
    row.check(clique->flags()==std::array<bool,4>{true,true,true,true});rows.push_back(row);
  }
  {
    Case row{"scope_null_output_unowned"};
    row.check(!llaminarPublishSmallAllGatherPolicy(nullptr,4,false,true));rows.push_back(row);
  }
  {
    Case row{"scope_null_output_completed"};
    row.check(!llaminarPublishSmallAllGatherPolicy(nullptr,4,true,true));rows.push_back(row);
  }
  rejected("scope_nested_or_incomplete",[](Clique&){},false);
  rejected("scope_explicit_selector",[](Clique&){},true,false);
  rejected("scope_partial_count",[](Clique&){},true,true,3);
  rejected("scope_missing_output_owner",[](Clique& c){c.owners[2]=nullptr;});
  rejected("scope_duplicate_output_owner",[](Clique& c){c.owners[2]=c.owners[1];});
  rejected("scope_nonblocking",[](Clique& c){c.owners[2]->config.blocking=0;});
  rejected("scope_failed_initialization",[](Clique& c){c.owners[2]->initState=ncclInternalError;});
  rejected("scope_missing_original_owner",[](Clique& c){c.info[0][2].comm=nullptr;});
  rejected("scope_foreign_original_owner",[](Clique& c){c.info[0][2].comm=c.owners[1];});
  rejected("scope_foreign_process",[](Clique& c){c.info[0][2].pidHash=99;});
  rejected("scope_foreign_host",[](Clique& c){c.info[0][2].hostHash=99;});
  rejected("scope_foreign_communicator",[](Clique& c){c.owners[2]->commHash=99;});
  rejected("scope_missing_reciprocal_owner",[](Clique& c){c.info[2][0].comm=nullptr;});
  rejected("scope_mixed_architecture",[](Clique& c){std::strcpy(c.owners[2]->topo->nodes[GPU].nodes[1].gpu.gcn,"gfx950");});
  rejected("scope_duplicate_topology_rank",[](Clique& c){c.owners[2]->topo->nodes[GPU].nodes[1].gpu.rank=3;});
  rejected("scope_unconnected_endpoint",[](Clique& c){c.endpoints[2][1][3].send[0].connected=0;});
  rejected("scope_foreign_transport",[](Clique& c){c.endpoints[2][1][3].send[0].transportComm=nullptr;});
  rejected("scope_copy_fifo",[](Clique& c){c.endpoints[2][1][3].send[0].conn.connFifo=reinterpret_cast<ncclConnFifo*>(1);});
  rejected("scope_proxy_progress",[](Clique& c){
    using Progress=decltype(c.endpoints[2][1][3].send[0].proxyConn.proxyProgress);
    c.endpoints[2][1][3].send[0].proxyConn.proxyProgress=reinterpret_cast<Progress>(1);
  });
  rejected("scope_external_tuner",[](Clique& c){c.owners[2]->tuner=reinterpret_cast<ncclTuner_t*>(1);});
  rejected("scope_multinode",[](Clique& c){c.owners[2]->nNodes=2;});
  rejected("scope_rank2",[](Clique& c){c.owners[2]->nRanks=2;});
  rejected("scope_rank3",[](Clique& c){c.owners[2]->nRanks=3;});
  rejected("scope_rank8",[](Clique& c){c.owners[2]->nRanks=8;});
  rejected("scope_runtime_connection",[](Clique& c){c.owners[2]->runtimeConn=true;});
  rejected("scope_unmeasured_channels",[](Clique& c){c.owners[2]->nChannels=8;});
  return rows;
}

/** @brief Authored cost-construction inputs; the actual native tuner computes every cost. */
enum class CostInput { Valid, MissingLL, NaNLL, InfiniteLL, NegativeLL };

/** @brief Execute the original full selector; caller supplies only authored native metadata. */
Case selection(const char* name,size_t count,ncclDataType_t datatype,int wantedProtocol,
               bool admitted=true,ncclFunc_t function=ncclFuncAllGather,const char* architecture="gfx906",
               CostInput cost=CostInput::Valid,ncclResult_t wantedStatus=ncclSuccess) {
  auto clique=std::make_unique<Clique>();
  const bool published=llaminarPublishSmallAllGatherPolicy(clique->owners.data(),4,true,admitted);
  auto* comm=clique->owners[0];
  for (int slot=0;slot<4;++slot) std::strcpy(comm->topo->nodes[GPU].nodes[slot].gpu.gcn,architecture);
  // Non-admitted alternate architecture cases cannot carry the gfx906 policy.
  if (std::strcmp(architecture,"gfx906")!=0) comm->llaminarSmallAllGatherLL=false;
  if (cost==CostInput::MissingLL) comm->bandwidths[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=0.0f;
  if (cost==CostInput::NaNLL) comm->latencies[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=std::numeric_limits<float>::quiet_NaN();
  if (cost==CostInput::InfiniteLL) comm->latencies[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=std::numeric_limits<float>::infinity();
  if (cost==CostInput::NegativeLL) comm->bandwidths[function][NCCL_ALGO_RING][NCCL_PROTO_LL]=-1.0f;
  ncclTaskColl info{};info.func=function;info.datatype=datatype;info.count=count;
  const auto beforeCount=info.count;const auto beforeFlags=clique->flags();
  const auto status=getAlgoInfo(comm,&info,0,0,1,nullptr);
  Case row{name};row.check(published==admitted);row.check(status==wantedStatus);
  if (status==ncclSuccess && wantedStatus==ncclSuccess) row.check(info.protocol==wantedProtocol);
  if (std::strcmp(name,"native_gfx950_postpass")==0) row.check(info.algorithm==NCCL_ALGO_TREE);
  if (std::strcmp(name,"native_gfx942_pipeline")==0) row.check(info.pipeline==1);
  row.check(info.count==beforeCount);row.check(clique->flags()==beforeFlags);
  row.observed={int(status),info.algorithm,info.protocol,info.nMaxChannels,info.nWarps,info.pipeline};
  return row;
}

/** @brief Selection boundaries and unchanged original no-plugin postpasses. */
std::vector<Case> selectionCases() {
  std::vector<Case> rows;
  rows.push_back(selection("small_1",1,ncclInt8,NCCL_PROTO_LL));
  rows.push_back(selection("small_10240_fp32",2560,ncclFloat32,NCCL_PROTO_LL));
  rows.push_back(selection("boundary_16383",16383,ncclInt8,NCCL_PROTO_LL));
  rows.push_back(selection("boundary_16384",4096,ncclFloat32,NCCL_PROTO_LL));
  rows.push_back(selection("large_16385",16385,ncclInt8,NCCL_PROTO_SIMPLE));
  rows.push_back(selection("large_1048576",1048576,ncclInt8,NCCL_PROTO_SIMPLE));
  rows.push_back(selection("other_collective",2560,ncclFloat32,NCCL_PROTO_SIMPLE,true,ncclFuncAllReduce));
  rows.push_back(selection("explicit_selector_excluded",2560,ncclFloat32,NCCL_PROTO_SIMPLE,false));
  rows.push_back(selection("overflow",std::numeric_limits<size_t>::max(),ncclFloat64,NCCL_PROTO_SIMPLE,
                           true,ncclFuncAllGather,"gfx906",CostInput::Valid,ncclInvalidArgument));
  rows.push_back(selection("unavailable_ll",2560,ncclFloat32,NCCL_PROTO_SIMPLE,
                           true,ncclFuncAllGather,"gfx906",CostInput::MissingLL,ncclInvalidUsage));
  rows.push_back(selection("nan_ll",2560,ncclFloat32,NCCL_PROTO_SIMPLE,
                           true,ncclFuncAllGather,"gfx906",CostInput::NaNLL,ncclInvalidArgument));
  rows.push_back(selection("infinite_ll",2560,ncclFloat32,NCCL_PROTO_SIMPLE,
                           true,ncclFuncAllGather,"gfx906",CostInput::InfiniteLL,ncclInvalidArgument));
  rows.push_back(selection("negative_ll",2560,ncclFloat32,NCCL_PROTO_SIMPLE,
                           true,ncclFuncAllGather,"gfx906",CostInput::NegativeLL,ncclInvalidArgument));
  for (const auto item : {std::pair<const char*,size_t>{"zero_policy",0},
                          std::pair<const char*,size_t>{"nondivisible_policy",40961}}) {
    std::array<float,NCCL_NUM_ALGORITHMS*NCCL_NUM_PROTOCOLS> table{};
    table.fill(NCCL_ALGO_PROTO_IGNORE);
    table[NCCL_ALGO_RING*NCCL_NUM_PROTOCOLS+NCCL_PROTO_LL]=40.0f;
    table[NCCL_ALGO_RING*NCCL_NUM_PROTOCOLS+NCCL_PROTO_SIMPLE]=10.0f;
    const auto before=table;int channels=77;Case row{item.first};
    const auto status=llaminar::collective::preferSmallAllGatherLL(4,1,ncclFuncAllGather,item.second,
      reinterpret_cast<float**>(table.data()),NCCL_NUM_ALGORITHMS,NCCL_NUM_PROTOCOLS,&channels);
    row.check(status==(item.second==0?ncclSuccess:ncclInvalidArgument));
    row.check(std::memcmp(before.data(),table.data(),sizeof(table))==0);row.check(channels==77);
    row.observed[0]=int(status);rows.push_back(row);
  }
  rows.push_back(selection("native_gfx950_postpass",4096,ncclFloat32,NCCL_PROTO_LL,
                           false,ncclFuncAllReduce,"gfx950"));
  rows.push_back(selection("native_gfx942_pipeline",4096,ncclBfloat16,NCCL_PROTO_SIMPLE,
                           false,ncclFuncAllReduce,"gfx942"));
  return rows;
}
} // namespace

/** @brief Run a bounded host matrix and emit measured assertion/selection metadata. */
int main(int argc,char** argv) {
  if(argc!=2 || (std::string(argv[1])!="scope" && std::string(argv[1])!="selection")) return 2;
  const auto rows=std::string(argv[1])=="scope"?scopeCases():selectionCases();
  unsigned failures=0;
  std::printf("{\"mode\":\"%s\",\"cases\":[",argv[1]);
  for(size_t index=0;index<rows.size();++index) {
    const auto& row=rows[index];failures+=row.failures;
    std::printf("%s{\"name\":\"%s\",\"checks\":%d,\"failures\":%d,\"observed\":[%d,%d,%d,%d,%d,%d]}",
      index?",":"",row.name,row.checks,row.failures,row.observed[0],row.observed[1],row.observed[2],
      row.observed[3],row.observed[4],row.observed[5]);
  }
  std::printf("]}\n");return failures?1:0;
}
