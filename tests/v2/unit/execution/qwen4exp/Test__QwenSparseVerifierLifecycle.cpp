/**
 * @file Test__QwenSparseVerifierLifecycle.cpp
 * @brief Actual cache auxiliary prior producer, retained stage and CPU publisher lifecycle.
 *
 * Dedicated admitted prior bytes survive repeated accepted-prefix publication;
 * pool bytes and inactive tail capacity remain outside traffic. Setup capability
 * identity rejects replaced/retired producers before any new graph edge is added.
 */
#include "execution/qwen4exp/Qwen4ExpAuxiliaryState.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>
using namespace llaminar2;
namespace {
using Stage=QwenSparseAttentionStage;using Float=Qwen4ExpFloatBinding;using Int=Qwen4ExpIntBinding;
/** @brief A setup-only dependency endpoint; no model or native cache bytes are mutated. */
class Endpoint final:public IComputeStage {
public:
    explicit Endpoint():IComputeStage(DeviceId::cpu()){}
    bool execute(IDeviceContext*)override{return true;}
    ComputeStageType type()const override{return ComputeStageType::COPY;}
    bool supportsBackend(ComputeBackendType b)const override{return b==ComputeBackendType::CPU;}
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
};
/** @brief Real sparse auxiliary state with explicit ephemeral prior/candidate declarations. */
struct Fixture {
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    std::shared_ptr<IKVCache> cache=createCPURingKVCache(ActivationPrecision::FP32,mpi,1,1,16,1,128,DeviceId::cpu());
    FP32Tensor pools{{4,128}},partial{{4,128}},raw{{5,128}},retained{{5,128}};
    INT32Tensor replay{{3}},prior{{388}},traffic{{2}},copied{{2}};
    Qwen4ExpAuxiliaryState*owner=nullptr;
    Fixture(){
        std::memset(prior.raw_mutable_data(),0x5a,prior.size_bytes());
        std::memset(traffic.raw_mutable_data(),0x6b,traffic.size_bytes());
        std::fill_n(pools.mutable_data(),pools.numel(),-91.0f);
        Qwen4ExpAuxiliaryState::Params p{.cache=*cache,.device=DeviceId::cpu(),.requestCapacity=1,.eosToken=42};
        p.sparseLayers.push_back({3,0,0,16,Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),
            Qwen4ExpAuxiliaryState::SparseVerifierStorage{5,Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS),
                Int(copied,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES),{Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR)},
                Int(traffic,BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC)}});
        auto state=std::make_unique<Qwen4ExpAuxiliaryState>(std::move(p));owner=state.get();cache->attachAuxiliaryState(std::move(state),nullptr);
        for(std::size_t i=0;i<partial.numel();++i)partial.mutable_data()[i]=float(i)+0.25f;
        for(std::size_t i=0;i<raw.numel();++i)raw.mutable_data()[i]=float(i)+1000.25f;
        auto*r=static_cast<int*>(replay.raw_mutable_data());r[0]=3;r[1]=5;r[2]=17;
    }
    /** @return Same-owner snapshot capability ordered after canonical replay publication. */
    ImmutableVerifierPrior snapshot(ComputeGraph&graph){
        graph.addNode("replay",std::make_unique<Endpoint>(),DeviceId::cpu());
        return owner->appendSparseVerifierPrior(graph,"layer3.",3,0,5,Int(replay,BufferId::QFN_QSA_REPLAY),{"replay"});
    }
    /** @return Actual retained native stage, with packed immutable original source. */
    std::unique_ptr<Stage> retention(){
        const QwenSparseTransactionPriorBinding original{Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR)};
        return std::make_unique<Stage>(Stage::Params{.device_id=DeviceId::cpu(),.rowCapacity=5,.tokenCapacity=16,
            .replay=original.storage,.operation=Stage::RetainCandidates{Float(raw,BufferId::QFN_QSA_RAW_KEY),
                Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS),Int(copied,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES),
                Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),0,1,RequestRowOrigins::fixedStride(0,5),original}});
    }
};
}
/** @brief A live original3-row tail must be restored after accepted rows complete a pool block. */
TEST(QwenSparseVerifierLifecycle, DedicatedOriginalPriorSurvivesRepeatedFullPartialAndZeroPublication){
    llaminar::v2::ThreadCountGuard threads(1);Fixture f;
    EXPECT_EQ(static_cast<const unsigned char*>(f.prior.raw_data())[0],0x5a);
    EXPECT_EQ(static_cast<const unsigned char*>(f.traffic.raw_data())[0],0x6b);
    ComputeGraph graph;auto capability=f.snapshot(graph);
    graph.addNode("append",std::make_unique<Endpoint>(),DeviceId::cpu());capability.orderBefore(graph,"append");
    graph.addNode("retain",f.retention(),DeviceId::cpu());graph.addDependency("retain","append");
    const auto producer=capability.snapshotProducer();std::move(capability).retainForPublication(graph,"retain");
    const auto&appendDependencies=graph.getNode("append")->dependencies;
    EXPECT_NE(std::find(appendDependencies.begin(),appendDependencies.end(),producer),appendDependencies.end());
    CPUDeviceContext context(DeviceId::cpu(),1);
    for(auto*stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&context));
    const auto original=std::vector<unsigned char>(static_cast<const unsigned char*>(f.prior.raw_data()),static_cast<const unsigned char*>(f.prior.raw_data())+f.prior.size_bytes());
    const auto*traffic=static_cast<const qwen4exp::QwenSparsePriorTraffic*>(f.traffic.raw_data());
    EXPECT_EQ(traffic->metadata_bytes,16u);EXPECT_EQ(traffic->payload_bytes,3u*128*sizeof(float));
    EXPECT_EQ(*static_cast<const std::uint64_t*>(f.copied.raw_data()),5u*128*sizeof(float));
    std::vector<float> sequence(f.partial.data(),f.partial.data()+3*128);sequence.insert(sequence.end(),f.raw.data(),f.raw.data()+5*128);
    auto expected=std::vector<float>(f.partial.data(),f.partial.data()+f.partial.numel());
    for(int accepted:{5,2,0,1,0,5,0}){
        MTPSpecStepPlan plan;plan.request_index=0;plan.request_id=39;plan.draft_count=4;plan.target_rows=5;
        plan.accepted_count=accepted?accepted-1:0;const int row=accepted-1;
        const auto result=publishAcceptedMTPSpecStateFromVerifierRow(plan,row,{graph.getNode("retain")->stage.get()},DeviceId::cpu(),nullptr,true);
        ASSERT_TRUE(result.ok)<<result.error;EXPECT_EQ(result.restored_stage_count,1);
        const int tail=(3+accepted)%4;std::copy_n(sequence.data()+((3+accepted)/4*4)*128,tail*128,expected.data());
        EXPECT_EQ(std::vector<float>(f.partial.data(),f.partial.data()+f.partial.numel()),expected);
        EXPECT_EQ(std::memcmp(f.prior.raw_data(),original.data(),original.size()),0);
        for(std::size_t i=0;i<f.pools.numel();++i)EXPECT_EQ(f.pools.data()[i],-91.0f);
    }
}
/** @brief Replacing the snapshot by name or clearing/rebuilding cannot authenticate a stale capability. */
TEST(QwenSparseVerifierLifecycle, ReplacedAndClearedSnapshotCapabilitiesFailBeforeEdgeMutation){
    for(bool clear:{false,true}){Fixture f;ComputeGraph graph;auto capability=f.snapshot(graph);
        const auto name=capability.snapshotProducer();
        if(clear)graph.clear();
        graph.addNode(name,std::make_unique<Endpoint>(),DeviceId::cpu());
        graph.addNode("consumer",std::make_unique<Endpoint>(),DeviceId::cpu());const auto size=graph.size();
        EXPECT_THROW(capability.orderBefore(graph,"consumer"),std::invalid_argument);
        EXPECT_EQ(graph.size(),size);EXPECT_TRUE(graph.getNode("consumer")->dependencies.empty());
    }
}
