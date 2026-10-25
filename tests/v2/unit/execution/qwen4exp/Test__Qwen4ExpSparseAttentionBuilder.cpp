/**
 * @file Test__Qwen4ExpSparseAttentionBuilder.cpp
 * @brief Device-free prepared-GEMM/native-cache proof of sparse graph composition.
 *
 * Exact split GGUF-shaped projections feed real sparse stages and a native
 * FP16 cache. An independent zero-query causal mean verifies output wiring;
 * construction rejects missing producers, aliases and duplicate node identity
 * without publishing a partial graph. No accelerator or checkpoint is loaded.
 */
#include <gtest/gtest.h>
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpSparseAttentionBuilder.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "kernels/qwen4exp/QwenSparseMath.h"
#include "loaders/PreparedWeightStore.h"
#include "models/GraphTypes.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/OpenMPUtils.h"
#include <algorithm>
#include <array>
#include <limits>
#include "../../../utils/PreparedWeightTestHarness.h"

using namespace llaminar2;
namespace
{
using Builder=Qwen4ExpSparseAttentionBuilder;
using Float=Qwen4ExpFloatBinding;
using Int=Qwen4ExpIntBinding;
using Weight=Qwen4ExpImmutableWeightBinding;
/** @brief Complete admitted tiny-H fixture retaining the real fixed QSA head geometry. */
struct Fixture
{
    static constexpr int rows=3,hidden=8,tokens=8,blocks=2,positions=16;
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    CPURingKVCacheFP16 cache{mpi,1,1,tokens,2,256};
    PreparedWeightStore store{ModelContextId{99179}};
    FP32Tensor embedding{{rows,hidden}},input{{rows,hidden}},coreQuery{{rows,6144}},output{{rows,6144}},
        queryWeight{{512,hidden}},keyWeight{{128,hidden}},gain{{128}},cosine{{positions,32}},sine{{positions,32}},
        query{{rows,512}},raw{{rows,128}},prior{{4,128}},pools{{blocks,128}},prepared{{rows,512}},scores{{rows,blocks}};
    INT32Tensor replay{{3}},selected{{rows,2051}},count{{rows}};
    LayerWeights layer;
    WeightBinding queryBinding,keyBinding;
    /** @brief Prepare frozen split indexer matrices and seed the persistent native cache. */
    Fixture()
    {
        for(auto *tensor : {&embedding,&queryWeight,&keyWeight})
            for(std::size_t i=0;i<tensor->numel();++i)tensor->mutable_data()[i]=float(int(i%19)-9)/37;
        std::fill_n(coreQuery.mutable_data(),coreQuery.numel(),0.0f);
        std::fill_n(prior.mutable_data(),prior.numel(),0.25f);
        std::fill_n(gain.mutable_data(),gain.numel(),1.0f);
        std::fill_n(cosine.mutable_data(),cosine.numel(),1.0f);
        std::fill_n(sine.mutable_data(),sine.numel(),0.0f);
        auto *state=static_cast<int *>(replay.raw_mutable_data());state[0]=2;state[1]=rows;state[2]=0;
        auto native=cache.acquireSparseAttentionNativeView(0,0,nullptr,{},nullptr);
        auto *key=static_cast<std::uint16_t *>(const_cast<void *>(native.keys.data));
        auto *value=static_cast<std::uint16_t *>(const_cast<void *>(native.values.data));
        for(int token=0;token<tokens;++token)for(int head=0;head<2;++head)for(int d=0;d<256;++d) {
            key[(token*2+head)*256+d]=0;
            value[(token*2+head)*256+d]=qwen4exp::detail::encodeHalf(float(token+head)/8);
        }
        queryBinding=prepare(queryWeight,"indexer.q_proj.weight");keyBinding=prepare(keyWeight,"indexer.k_proj.weight");
    }
    /** @brief Resolve immutable prepared authority through the production store. */
    WeightBinding prepare(TensorBase &tensor,const char *name)
    {
        auto binding=test::makePreparedWeightTestBinding(&tensor,DeviceId::cpu(),name,store.modelId());
        binding.prepared=store.prepareGemm(binding);return binding;
    }
    /** @brief Exact ordinary attention handoff; no gathered K/V views are present. */
    AttentionConsumerBinding consumer()
    {
        return {FullAttentionLayerIdentity::main(
            LayerGraphSourceIdentity::main(0,48,0,GraphCacheLayerIndex::ModelGlobal),
            GraphAttentionConsumer::Selected),layer,"qsa.",DeviceId::cpu(),{input,BufferId::NORMALIZED},{coreQuery,BufferId::Q_PROJ},
            {output,BufferId::ATTN_OUTPUT},&cache,0,0,rows,1,24,2,0,0,{},"ready","ready","ready"};
    }
    /** @brief Fixed geometry with explicit ordinary prepared projection arithmetic. */
    Qwen4ExpSparseAttentionPolicy policy() { return {hidden,tokens,blocks,positions}; }
    /** @brief Prepared split projections and already-converted native norm gains. */
    Qwen4ExpSparseAttentionWeights weights()
    {
        return {Qwen4ExpPreparedProjection(queryBinding,store,DeviceId::cpu()),Qwen4ExpPreparedProjection(keyBinding,store,DeviceId::cpu()),
            Weight(gain),Weight(gain),Weight(cosine),Weight(sine)};
    }
    /** @brief Independent arena identities and immutable setup-owned CPU control record. */
    Qwen4ExpSparseAttentionBuffers buffers()
    {
        return {Int(replay,BufferId::QFN_QSA_REPLAY),Float(query,BufferId::QFN_QSA_QUERY),Float(raw,BufferId::QFN_QSA_RAW_KEY),
            Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS),Float(pools,BufferId::QFN_QSA_POOLED_KEYS),Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY),
            Float(scores,BufferId::QFN_QSA_SCORES),Int(selected,BufferId::QFN_QSA_SELECTED),Int(count,BufferId::QFN_QSA_SELECTED_COUNT),
            {},{},{},"ready",""};
    }
    /** @brief Publish the normalized projection input through an ordinary real CPU stage. */
    void seed(ComputeGraph &graph)
    {
        graph.addNode("ready",std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
            .rows={rows,{}},.operation=Qwen4ExpPrimitiveStage::ResidualDown{hidden,
                Float(embedding,BufferId::FFN_NORMALIZED),Float(input,BufferId::NORMALIZED)}}),DeviceId::cpu());
    }
};
}
/** @test Four flat sparse nodes use exact prepared identities, native cache and causal outputs. */
TEST(Qwen4ExpSparseAttentionBuilder, RealPreparedCPUProjectionsReachNativeSparseConsumer)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;ComputeGraph graph;f.seed(graph);
    auto policy=f.policy();policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    policy.projectionRows=DeviceRowRange::fullyActive(Fixture::rows);
    EXPECT_EQ(Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),"qsa.qsa_commit_partial");
    EXPECT_EQ(graph.size(),6u);
    for(const auto *name : {"qsa.qsa_query_proj","qsa.qsa_key_proj"}) {
        auto *stage=dynamic_cast<GEMMStage *>(graph.getNode(name)->stage.get());ASSERT_TRUE(stage);
        EXPECT_TRUE(stage->getParams().force_decode_equivalent_verifier_prefill);
        EXPECT_EQ(stage->getParams().k,Fixture::hidden);ASSERT_TRUE(stage->getParams().prepared_ref);
    }
    auto *attention=dynamic_cast<QwenSparseAttentionStage *>(graph.getNode("qsa.attention")->stage.get());ASSERT_TRUE(attention);
    EXPECT_FALSE(std::get<QwenSparseAttentionStage::SelectedAttention>(attention->getParams().operation).gate);
    CPUDeviceContext context(DeviceId::cpu(),1);
    for(auto *stage : graph.getExecutionStages())ASSERT_TRUE(stage->execute(&context));
    for(int row=0;row<Fixture::rows;++row)for(int head=0;head<24;++head)for(int d=0;d<256;++d)
        EXPECT_NEAR(f.output.data()[row*6144+head*256+d],float((2+row)/2.0+head/12)/8,1e-7);
}
/** @test Bad edges/aliases/shards/capture identities leave the graph unchanged. */
TEST(Qwen4ExpSparseAttentionBuilder, RejectsPartialGraphPublicationAndAmbiguousOwnership)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;ComputeGraph graph;f.seed(graph);
    auto buffers=f.buffers();buffers.replayReady="missing";
    EXPECT_THROW(Builder::append(graph,f.consumer(),f.policy(),f.weights(),buffers),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    buffers=f.buffers();buffers.rawKey=buffers.query;
    EXPECT_THROW(Builder::append(graph,f.consumer(),f.policy(),f.weights(),buffers),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    auto consumer=f.consumer();consumer.cache=nullptr;
    EXPECT_THROW(Builder::append(graph,consumer,f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    auto badShard=f.consumer();badShard.kvHeadStart=1;
    EXPECT_THROW(Builder::append(graph,badShard,f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    (void)Builder::append(graph,f.consumer(),f.policy(),f.weights(),f.buffers());
    EXPECT_THROW(Builder::append(graph,f.consumer(),f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),6u);
}

/** @test Both public composition paths reject foreign cache/source and dense predictor identities before mutation. */
TEST(Qwen4ExpSparseAttentionBuilder, RejectsDensePredictorAndMismatchedLayerIdentity)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f; ComputeGraph graph; f.seed(graph);
    auto cacheConsumer=f.consumer();
    cacheConsumer.cacheLayer=1;
    EXPECT_THROW(Builder::append(graph,cacheConsumer,f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    auto sourceConsumer=f.consumer();
    sourceConsumer.modelLayer=1;
    EXPECT_THROW(Builder::append(graph,sourceConsumer,f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    auto consumer=f.consumer();
    const Qwen4ExpPredictorMetadata metadata{48,Qwen4ExpPredictorAttention::Dense,
        Qwen4ExpPredictorHiddenNorm::PerBranch,Qwen4ExpPredictorSource::SharedGGUF};
    consumer.layerIdentity=FullAttentionLayerIdentity::qwen4ExpSharedPredictor(
        LayerGraphSourceIdentity::qwen4ExpSharedPredictor(48,metadata,48,0,0));
    consumer.modelLayer=48;
    EXPECT_THROW(Builder::append(graph,consumer,f.policy(),f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    EXPECT_THROW(Builder::appendRequests(graph,consumer,f.policy(),f.weights(),
        DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
}

/** @test Flat prepared projections preserve independent request DAGs and defer only exact tail candidates. */
TEST(Qwen4ExpSparseAttentionBuilder, ThreeRequestsUseIndependentNativeBanksAndRetainedPublicationHooks)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;constexpr int requests=3,rows=Fixture::rows,total=requests*rows,width=128;
    CPURingKVCacheFP16 cache(f.mpi,1,requests,Fixture::tokens,2,256);
    FP32Tensor embedding({total,Fixture::hidden}),input({total,Fixture::hidden}),query({total,512}),raw({total,width}),
        retained({total,width}),prior({requests,4*width}),pools({requests,Fixture::blocks*width}),prepared({total,512}),
        scores({total,Fixture::blocks}),coreQuery({total,6144}),output({total,6144});
    INT32Tensor replay({requests*3}),selected({total,2051}),count({total}),observed({requests*2});
    INT32Tensor transactionPrior({requests*388}),priorTraffic({requests*2});
    std::array<int,requests> live{3,0,2};const std::array<int,requests> prefix{1,2,3};
    for(std::size_t i=0;i<embedding.numel();++i)embedding.mutable_data()[i]=float(i%31+1)/37;
    std::fill_n(prior.mutable_data(),prior.numel(),0.25f);
    const std::vector<float> originalTail(prior.data(),prior.data()+prior.numel());
    std::fill_n(coreQuery.mutable_data(),coreQuery.numel(),0.0f);
    std::fill_n(query.mutable_data(),query.numel(),-101.25f);
    std::fill_n(raw.mutable_data(),raw.numel(),-102.5f);
    for(int request=0;request<requests;++request)for(int row=live[request];row<rows;++row)
        std::fill_n(embedding.mutable_data()+(request*rows+row)*Fixture::hidden,Fixture::hidden,
            std::numeric_limits<float>::quiet_NaN());
    std::fill_n(output.mutable_data(),output.numel(),-55.0f);
    std::fill_n(retained.mutable_data(),retained.numel(),-71.0f);
    auto *state=static_cast<int *>(replay.raw_mutable_data());
    for(int r=0;r<requests;++r) {
        state[r*3]=prefix[r];state[r*3+1]=live[r];state[r*3+2]=0;
        auto native=cache.acquireSparseAttentionNativeView(0,r,nullptr,{},nullptr);
        auto *keys=static_cast<std::uint16_t *>(const_cast<void *>(native.keys.data));
        auto *values=static_cast<std::uint16_t *>(const_cast<void *>(native.values.data));
        for(int token=0;token<Fixture::tokens;++token)for(int head=0;head<2;++head)for(int d=0;d<256;++d) {
            keys[(token*2+head)*256+d]=0;
            values[(token*2+head)*256+d]=qwen4exp::detail::encodeHalf(float(r*2)+float(token+head)/8);
        }
    }
    ComputeGraph graph;
    graph.addNode("ready",std::make_unique<Qwen4ExpPrimitiveStage>(Qwen4ExpPrimitiveStage::Params{
        .rows={total,{}},.operation=Qwen4ExpPrimitiveStage::ResidualDown{Fixture::hidden,
            Float(embedding,BufferId::FFN_NORMALIZED),Float(input,BufferId::NORMALIZED)}}),DeviceId::cpu());
    AttentionConsumerBinding consumer{FullAttentionLayerIdentity::main(
            LayerGraphSourceIdentity::main(0,48,0,GraphCacheLayerIndex::ModelGlobal),
            GraphAttentionConsumer::Selected),f.layer,"qsa.",DeviceId::cpu(),{input,BufferId::NORMALIZED},{coreQuery,BufferId::Q_PROJ},
        {output,BufferId::ATTN_OUTPUT},&cache,0,0,total,requests,24,2,0,0,{},"ready","ready","ready"};
    std::vector<Qwen4ExpSparseAttentionRequest> requestViews;
    for(int r=0;r<requests;++r) {
        const auto first=std::size_t(r)*rows;
        const auto snapshotName="snapshot"+std::to_string(r);
        const QwenSparseTransactionPriorBinding original{Int(transactionPrior,BufferId::QFN_QSA_TRANSACTION_PRIOR,
            std::size_t(r)*sizeof(qwen4exp::QwenSparseTransactionPrior))};
        graph.addNode(snapshotName,std::make_unique<QwenSparseAttentionStage>(QwenSparseAttentionStage::Params{
            .device_id=DeviceId::cpu(),.rowCapacity=rows,.tokenCapacity=Fixture::tokens,
            .replay=Int(replay,BufferId::QFN_QSA_REPLAY,r*3*sizeof(int)),
            .operation=QwenSparseAttentionStage::SnapshotPrior{
                Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS,r*4*width*sizeof(float)),original,
                Int(priorTraffic,BufferId::QFN_QSA_TRANSACTION_PRIOR_TRAFFIC,r*sizeof(qwen4exp::QwenSparsePriorTraffic))}}),DeviceId::cpu());
        graph.addDependency(snapshotName,"ready");
        requestViews.push_back({r,{original.storage,
            Float(query,BufferId::QFN_QSA_QUERY,first*512*sizeof(float)),Float(raw,BufferId::QFN_QSA_RAW_KEY,first*width*sizeof(float)),
            Float(prior,BufferId::QFN_QSA_PARTIAL_KEYS,r*4*width*sizeof(float)),
            Float(pools,BufferId::QFN_QSA_POOLED_KEYS,r*Fixture::blocks*width*sizeof(float)),
            Float(prepared,BufferId::QFN_QSA_PREPARED_QUERY,first*512*sizeof(float)),
            Float(scores,BufferId::QFN_QSA_SCORES,first*Fixture::blocks*sizeof(float)),
            Int(selected,BufferId::QFN_QSA_SELECTED,first*2051*sizeof(int)),Int(count,BufferId::QFN_QSA_SELECTED_COUNT,first*sizeof(int)),
            {},{},{},snapshotName,"",Qwen4ExpSparseAttentionRetention{
                Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,first*width*sizeof(float)),
                Int(observed,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,r*sizeof(std::uint64_t)),RequestRowOrigins::fixedStride(0,rows),original}}});
    }
    auto policy=f.policy();policy.publication=Qwen4ExpSparsePublication::DeferredVerifierAcceptance;
    policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    // Construction alone must reject metadata/workspace ownership mistakes;
    // these checks deliberately run before any stage or graph mutation.
    const auto requestRows=DeviceRequestRowRanges::deviceCounted(requests,rows,live.data());
    INT32Tensor traffic({total,24*sizeof(qwen4exp::QwenSparseAttentionObservation)/sizeof(int)});
    INT32Tensor gpuPartial({total,24*qwen4exp::QwenSparsePartitions*sizeof(qwen4exp::QwenSparseAttentionPartial)/sizeof(int)});
    INT32Tensor rowParameters({requests*3+total*sizeof(attention::AttentionDeviceParams)/sizeof(int)});
    const auto admittedPriorNodes=graph.size();
    const auto rejectsBeforeMutation=[&](const auto &badViews) {
        try {
            (void)Builder::appendRequests(graph,consumer,policy,f.weights(),requestRows,badViews);
            FAIL()<<"Aliased request-local workspace unexpectedly admitted";
        } catch(const std::invalid_argument &error) {
            EXPECT_NE(std::string(error.what()).find("leases"),std::string::npos)<<error.what();
        }
        EXPECT_EQ(graph.size(),admittedPriorNodes);
    };
    for(int kind=0;kind<3;++kind) {
        auto bad=requestViews;
        for(int request=0;request<requests;++request) {
            if(kind==0)bad[request].buffers.observations=Int(traffic,BufferId::QFN_QSA_OBSERVATIONS,
                request*rows*24*sizeof(qwen4exp::QwenSparseAttentionObservation));
            if(kind==1)bad[request].buffers.gpuPartials=Int(gpuPartial,BufferId::QFN_QSA_PARTIALS,
                request*rows*24*qwen4exp::QwenSparsePartitions*sizeof(qwen4exp::QwenSparseAttentionPartial));
            if(kind==2) {
                bad[request].buffers.rowParameters=Int(rowParameters,BufferId::QFN_QSA_REPLAY,
                    requests*3*sizeof(int)+request*rows*sizeof(attention::AttentionDeviceParams));
            }
        }
        if(kind==0)bad[2].buffers.observations=bad[0].buffers.observations;
        if(kind==1)bad[2].buffers.gpuPartials=bad[0].buffers.gpuPartials;
        if(kind==2)bad[2].buffers.rowParameters=bad[0].buffers.rowParameters;
        rejectsBeforeMutation(bad);
    }
    auto counterAlias=requestViews;
    counterAlias[0].buffers.observations=Int(observed,BufferId::QFN_QSA_OBSERVATIONS);
    // The tiny counter tensor cannot represent an attention observation bank,
    // and must never be silently reused as that larger operation's storage.
    EXPECT_THROW(Builder::appendRequests(graph,consumer,policy,f.weights(),requestRows,counterAlias),std::invalid_argument);
    EXPECT_EQ(graph.size(),admittedPriorNodes);
    auto ordinary=policy;ordinary.projectionArithmetic=Qwen4ExpProjectionArithmetic::Ordinary;
    EXPECT_THROW(Builder::appendRequests(graph,consumer,ordinary,f.weights(),requestRows,requestViews),std::invalid_argument);
    EXPECT_EQ(graph.size(),admittedPriorNodes);
    const auto terminals=Builder::appendRequests(graph,consumer,policy,f.weights(),requestRows,requestViews);
    ASSERT_EQ(terminals.size(),requests);EXPECT_EQ(graph.size(),admittedPriorNodes+11u);
    for(int r=0;r<requests;++r) {
        EXPECT_EQ(terminals[r],"qsa.request"+std::to_string(r)+"_attention");
        const auto *node=graph.getNode(terminals[r]);ASSERT_NE(node,nullptr);
        for(const auto &dependency:node->dependencies)for(int other=0;other<requests;++other)if(other!=r)
            EXPECT_EQ(dependency.find("request"+std::to_string(other)+"_"),std::string::npos);
    }
    CPUDeviceContext context(DeviceId::cpu(),1);
    for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&context));
    EXPECT_EQ(std::memcmp(prior.data(),originalTail.data(),prior.size_bytes()),0);
    for(int request=0;request<requests;++request)for(int row=live[request];row<rows;++row) {
        for(int col=0;col<512;++col)EXPECT_EQ(query.data()[(request*rows+row)*512+col],-101.25f);
        for(int col=0;col<width;++col)EXPECT_EQ(raw.data()[(request*rows+row)*width+col],-102.5f);
    }
    for(const auto *name:{"qsa.qsa_query_proj","qsa.qsa_key_proj"}) {
        auto *projection=dynamic_cast<GEMMStage *>(graph.getNode(name)->stage.get());ASSERT_NE(projection,nullptr);
        ASSERT_TRUE(projection->getParams().request_rows);
        EXPECT_TRUE(projection->getParams().force_decode_equivalent_verifier_prefill);
        EXPECT_EQ(projection->getParams().request_rows->lengthOwner(),live.data());
        EXPECT_NE(projection->graphCaptureVariantSignature(),0u);
    }
    const auto *copied=static_cast<const std::uint64_t *>(observed.raw_data());
    for(int r=0;r<requests;++r) {
        EXPECT_EQ(copied[r],std::uint64_t(live[r])*width*sizeof(float));
        for(int row=0;row<rows;++row)for(int head=0;head<24;++head)for(int d=0;d<256;++d)
            if(row<live[r])EXPECT_NEAR(output.data()[(r*rows+row)*6144+head*256+d],
                float(r*2)+float((prefix[r]+row)/2.0+head/12)/8,1e-7);
            else EXPECT_EQ(output.data()[(r*rows+row)*6144+head*256+d],-55.0f);
        for(int row=0;row<rows;++row)for(int d=0;d<width;++d)
            EXPECT_EQ(retained.data()[(r*rows+row)*width+d],row<live[r]?raw.data()[(r*rows+row)*width+d]:-71.0f);
    }
    MTPSpecStepPlanBatch plans;plans.ok=true;plans.request_count=requests;
    for(int r=0;r<requests;++r) {
        MTPSpecStepPlan step;step.request_index=r;step.draft_count=2;step.target_rows=rows;
        step.accepted_count=r==1?0:r==0?1:2;plans.steps.push_back(step);
    }
    const std::array<int,requests> acceptedRows{0,-1,7};
    const auto published=publishAcceptedMTPSpecStateFromVerifierRows(plans,acceptedRows.data(),graph,DeviceId::cpu(),nullptr,true);
    ASSERT_TRUE(published.ok)<<published.error;EXPECT_EQ(published.restored_stage_count,requests);
    for(int d=0;d<width;++d) {
        EXPECT_EQ(prior.data()[d],originalTail[d]);EXPECT_EQ(prior.data()[width+d],raw.data()[d]);
        EXPECT_EQ(prior.data()[8*width+d],raw.data()[(2*rows+1)*width+d]);
    }
    EXPECT_EQ(std::memcmp(prior.data()+4*width,originalTail.data()+4*width,4*width*sizeof(float)),0);
}

/** @brief M1 is still counted work; graph construction reads no native GPU control contents. */
TEST(Qwen4ExpSparseAttentionBuilder, VerifierProjectionRowsPreserveM1NativeAuthorityAndRejectForeignOwners)
{
    std::int32_t count=0;
    for(auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
        for(int live:{0,1,0}) {
            count=live;
            const auto rows=Builder::verifierProjectionRows(device,{1,Qwen4ExpNativeIntInput{device,&count,1}});
            ASSERT_TRUE(rows);EXPECT_EQ(rows->countOwner(),&count);EXPECT_EQ(rows->capacity(),1);
            EXPECT_EQ(rows->activeRowsFor(count),live);
        }
        EXPECT_FALSE(Builder::verifierProjectionRows(device,{1,std::nullopt}));
        EXPECT_THROW(Builder::verifierProjectionRows(device,{1,Qwen4ExpNativeIntInput{DeviceId::cpu(),&count,1}}),std::invalid_argument);
    }
    count=1;EXPECT_FALSE(Builder::verifierProjectionRows(DeviceId::cpu(),{1,Qwen4ExpNativeIntInput{DeviceId::cpu(),&count,1}}));
    count=0;EXPECT_THROW(Builder::verifierProjectionRows(DeviceId::cpu(),{1,Qwen4ExpNativeIntInput{DeviceId::cpu(),&count,1}}),std::invalid_argument);
}


/** @test Both declarative entrypoints preserve each selector and reject corruption before graph mutation. */
TEST(Qwen4ExpSparseAttentionBuilder, SelectorPolicyIsExplicitAndCaptureSealed)
{
    using Selection=qwen4exp::QwenSparsePoolSelection;
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    EXPECT_EQ(f.policy().poolSelection,Selection::Installed);
    constexpr std::array selections{Selection::Installed,Selection::Bounded256_512,
        Selection::Bounded512_512,Selection::Bounded1024_1024};
    std::vector<std::uint64_t> identities;
    for(const auto selection:selections) {
        SCOPED_TRACE(static_cast<int>(selection));
        std::uint64_t singleIdentity=0;
        for(const bool requestsEntry:{false,true}) {
            SCOPED_TRACE(requestsEntry);
            ComputeGraph graph;f.seed(graph);
            auto policy=f.policy();policy.poolSelection=selection;
            if(requestsEntry) {
                const auto terminals=Builder::appendRequests(graph,f.consumer(),policy,f.weights(),
                    DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}});
                ASSERT_EQ(terminals.size(),1u);EXPECT_EQ(terminals.front(),"qsa.qsa_commit_partial");
            } else {
                EXPECT_EQ(Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),"qsa.qsa_commit_partial");
            }
            ASSERT_EQ(graph.size(),6u);
            const auto* node=graph.getNode("qsa.qsa_indexer");ASSERT_NE(node,nullptr);
            const auto* stage=dynamic_cast<const QwenSparseAttentionStage*>(node->stage.get());ASSERT_NE(stage,nullptr);
            EXPECT_EQ(std::get<QwenSparseAttentionStage::Indexer>(stage->getParams().operation).poolSelection,selection);
            const auto identity=stage->graphCaptureVariantSignature();ASSERT_NE(identity,0u);
            if(requestsEntry)EXPECT_EQ(identity,singleIdentity);
            else singleIdentity=identity;
        }
        EXPECT_EQ(std::find(identities.begin(),identities.end(),singleIdentity),identities.end());
        identities.push_back(singleIdentity);
    }
    ComputeGraph unchanged;f.seed(unchanged);
    auto invalid=f.policy();invalid.poolSelection=static_cast<Selection>(255);
    EXPECT_THROW(Builder::append(unchanged,f.consumer(),invalid,f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(unchanged.size(),1u);
    EXPECT_THROW(Builder::appendRequests(unchanged,f.consumer(),invalid,f.weights(),
        DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
    EXPECT_EQ(unchanged.size(),1u);
}

/**
 * @test Production selector declaration uses typed phase and backend at every decode capacity.
 * @details Calling the pure helper does not create a native context or query hardware.
 * In particular M1 Prefill and CPU/CUDA grouped Decode must remain Installed.
 */
TEST(Qwen4ExpSparseAttentionBuilder, ProductionPoolSelectionUsesTypedPhaseAndBackend)
{
    using Builder=Qwen4ExpSparseAttentionBuilder;
    using Phase=ForwardExecutionPhase;
    using Selection=qwen4exp::QwenSparsePoolSelection;
    for(const auto type:std::array{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm,
                                  DeviceType::Vulkan,DeviceType::Metal})
        for(int ordinal:{0,3})
        {
            SCOPED_TRACE(int(type));
            SCOPED_TRACE(ordinal);
            const DeviceId device(type,ordinal);
            for(int capacity=1;capacity<=16;++capacity)
            {
                SCOPED_TRACE(capacity);
                EXPECT_EQ(Builder::poolSelectionForInvocation(Phase::Prefill,device,capacity),Selection::Installed);
                if(type==DeviceType::ROCm)
                    EXPECT_EQ(Builder::poolSelectionForInvocation(Phase::Decode,device,capacity),Selection::Bounded1024_1024);
                else
                    EXPECT_EQ(Builder::poolSelectionForInvocation(Phase::Decode,device,capacity),Selection::Installed);
            }
            for(int capacity:{17,33,512,1024,2048,std::numeric_limits<int>::max()})
            {
                SCOPED_TRACE(capacity);
                EXPECT_EQ(Builder::poolSelectionForInvocation(Phase::Prefill,device,capacity),Selection::Installed);
                EXPECT_EQ(Builder::poolSelectionForInvocation(Phase::Decode,device,capacity),Selection::Installed);
            }
        }
}

/** @test Invalid enum/participant/capacity inputs fail before the selected policy can be published. */
TEST(Qwen4ExpSparseAttentionBuilder, ProductionPoolSelectionRejectsInvalidInvocation)
{
    using Builder=Qwen4ExpSparseAttentionBuilder;
    using Phase=ForwardExecutionPhase;
    for(const auto phase:{Phase::Prefill,Phase::Decode})
    {
        for(const auto device:std::array{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
            for(int capacity:{0,-1,std::numeric_limits<int>::min()})
                EXPECT_THROW(Builder::poolSelectionForInvocation(phase,device,capacity),std::invalid_argument);
        for(int capacity:{1,16,17})
        {
            EXPECT_THROW(Builder::poolSelectionForInvocation(phase,DeviceId::invalid(),capacity),std::invalid_argument);
            EXPECT_THROW(Builder::poolSelectionForInvocation(phase,DeviceId(DeviceType::ROCm,-1),capacity),std::invalid_argument);
            EXPECT_THROW(Builder::poolSelectionForInvocation(phase,DeviceId(static_cast<DeviceType>(255),0),capacity),std::invalid_argument);
        }
    }
    for(const auto device:std::array{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(int capacity:{1,16,17})
            EXPECT_THROW(Builder::poolSelectionForInvocation(static_cast<Phase>(255),device,capacity),std::invalid_argument);
}

/** @test Both builder entrypoints forward the default and reject unsupported attention arithmetic before publishing nodes. */
TEST(Qwen4ExpSparseAttentionBuilder, AttentionArithmeticDefaultForwardingAndColdRejection)
{
    using Arithmetic=qwen4exp::QwenSparseAttentionArithmetic;
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    EXPECT_EQ(f.policy().attentionArithmetic,qwen4exp::kDefaultQwenSparseAttentionArithmetic);
    EXPECT_EQ(f.policy().attentionArithmetic,Arithmetic::FixedFP32);
    for(const bool requestsEntry:{false,true}) {
        SCOPED_TRACE(requestsEntry);
        ComputeGraph graph;f.seed(graph);
        if(requestsEntry)
            (void)Builder::appendRequests(graph,f.consumer(),f.policy(),f.weights(),
                DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}});
        else (void)Builder::append(graph,f.consumer(),f.policy(),f.weights(),f.buffers());
        ASSERT_EQ(graph.size(),6u);
        const auto *node=graph.getNode("qsa.attention");ASSERT_NE(node,nullptr);
        const auto *stage=dynamic_cast<const QwenSparseAttentionStage *>(node->stage.get());ASSERT_NE(stage,nullptr);
        EXPECT_EQ(std::get<QwenSparseAttentionStage::SelectedAttention>(stage->getParams().operation).arithmetic,Arithmetic::FixedFP32);
        EXPECT_NE(stage->graphCaptureVariantSignature(),0u);
        for(const auto arithmetic:{Arithmetic::TileFP16WithFP32Totals,static_cast<Arithmetic>(255)}) {
            auto policy=f.policy();policy.attentionArithmetic=arithmetic;
            ComputeGraph unchanged;f.seed(unchanged);
            if(requestsEntry)
                EXPECT_THROW(Builder::appendRequests(unchanged,f.consumer(),policy,f.weights(),
                    DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
            else EXPECT_THROW(Builder::append(unchanged,f.consumer(),policy,f.weights(),f.buffers()),std::invalid_argument);
            EXPECT_EQ(unchanged.size(),1u);
        }
    }
}

/** @test The original typed phase selects8/33; M1 prompt and grouped Decode never alias phase semantics. */
TEST(Qwen4ExpSparseAttentionBuilder, Phase33ResolvesExactPhaseAndCanonicalRetainedFamily)
{
    using A=qwen4exp::QwenSparseAttentionArithmetic;using S=qwen4exp::QwenSparseAttentionPhaseSchedule;
    using P=ForwardExecutionPhase;
    for(const auto phase:{P::Prefill,P::Decode})for(int rows:{1,2,16,17,512,1024,2048})for(int retained:{0,1,16,2048}) {
        EXPECT_EQ(Builder::attentionArithmeticForInvocation(phase,DeviceId::rocm(0),rows,1,retained,A::TileFP16WithFP32Totals,S::Uniform),A::TileFP16WithFP32Totals);
        if(phase==P::Decode && rows>std::max(1,retained)) {
            EXPECT_THROW(Builder::attentionArithmeticForInvocation(phase,DeviceId::rocm(0),rows,1,retained,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
        } else EXPECT_EQ(Builder::attentionArithmeticForInvocation(phase,DeviceId::rocm(0),rows,1,retained,A::TileFP16WithFP32Totals,S::Prefill8Decode33),
            phase==P::Prefill?A::TileFP16WithFP32Totals:A::TileFP16WithFP32Totals33);
    }
    for(int ordinal:{0,1,2})EXPECT_EQ(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId(DeviceType::CPU,ordinal),1,1,0,A::FixedFP32,S::Uniform),A::FixedFP32);
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId(DeviceType::ROCm,-1),DeviceId::invalid()})
        EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,device,1,1,0,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
    for(const auto base:{A::FixedFP32,A::TileFP16WithFP32Totals33,static_cast<A>(255)})
        EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId::rocm(0),1,1,0,base,S::Prefill8Decode33),std::invalid_argument);
    EXPECT_THROW(Builder::attentionArithmeticForInvocation(static_cast<P>(255),DeviceId::rocm(0),1,1,0,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
    for(int bad:{0,-1})EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId::rocm(0),bad,1,0,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
    EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId::rocm(0),1,2,16,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
    EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId::rocm(0),1,1,-1,A::TileFP16WithFP32Totals,S::Prefill8Decode33),std::invalid_argument);
    EXPECT_THROW(Builder::attentionArithmeticForInvocation(P::Decode,DeviceId::rocm(0),1,1,0,A::TileFP16WithFP32Totals,static_cast<S>(255)),std::invalid_argument);
}
/** @test Invalid/foreign33 construction cannot publish even the first projection node. */
TEST(Qwen4ExpSparseAttentionBuilder, Phase33ColdRejectionLeavesGraphUnchanged)
{
    Fixture f;using A=qwen4exp::QwenSparseAttentionArithmetic;
    auto p=f.policy();p.attentionArithmetic=A::TileFP16WithFP32Totals33;
    p.executionPhase=ForwardExecutionPhase::Decode;p.retainedMtpCandidateRows=Fixture::rows;
    for(bool requestsEntry:{false,true}) {
        ComputeGraph g;f.seed(g);
        if(requestsEntry)EXPECT_THROW(Builder::appendRequests(g,f.consumer(),p,f.weights(),
            DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
        else EXPECT_THROW(Builder::append(g,f.consumer(),p,f.weights(),f.buffers()),std::invalid_argument);
        EXPECT_EQ(g.size(),1u);
    }
}


/**
 * @test Only explicit Decode elides queries; Prefill and absent phase retain AllLive.
 * @details Both builder entrypoints use the same fixed geometry and original
 * grouped arithmetic. Exact key GEMM/capture identity remains unchanged, so a
 * query gate cannot silently suppress key history needed by later ranking.
 */
TEST(Qwen4ExpSparseAttentionBuilder, QueryDemandUsesOnlyExplicitDecodeAndLeavesKeysUnchanged)
{
    using Phase=ForwardExecutionPhase;
    using Demand=qwen4exp::QwenSparseQueryDemand;
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    const std::array<std::optional<Phase>,3> phases{std::nullopt,Phase::Prefill,Phase::Decode};
    std::array<std::uint64_t,3> queryIdentities{},indexerIdentities{};
    std::uint64_t keyIdentity=0;
    for(std::size_t phase=0;phase<phases.size();++phase)for(const bool requestsEntry:{false,true})
    {
        SCOPED_TRACE(::testing::Message()<<"phase="<<phase<<" requestsEntry="<<requestsEntry);
        ComputeGraph graph;f.seed(graph);
        auto policy=f.policy();policy.executionPhase=phases[phase];
        // Hold arithmetic fixed to isolate the declared query-work policy.
        policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
        if(requestsEntry)
            (void)Builder::appendRequests(graph,f.consumer(),policy,f.weights(),
                DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}});
        else
            (void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers());
        ASSERT_EQ(graph.size(),6u);
        const auto *query=dynamic_cast<const GEMMStage *>(graph.getNode("qsa.qsa_query_proj")->stage.get());
        const auto *key=dynamic_cast<const GEMMStage *>(graph.getNode("qsa.qsa_key_proj")->stage.get());
        const auto *indexer=dynamic_cast<const QwenSparseAttentionStage *>(graph.getNode("qsa.qsa_indexer")->stage.get());
        ASSERT_NE(query,nullptr);
        ASSERT_NE(key,nullptr);
        ASSERT_NE(indexer,nullptr);
        const auto &q=query->getParams();const auto &k=key->getParams();
        const auto &i=std::get<QwenSparseAttentionStage::Indexer>(indexer->getParams().operation);
        const bool ranking=phases[phase]==Phase::Decode;
        EXPECT_EQ(i.queryDemand,ranking?Demand::RankingOnly:Demand::AllLive);
        EXPECT_EQ(q.projection_row_work.has_value(),ranking);
        if(ranking) {
            const auto &work=*q.projection_row_work;
            EXPECT_EQ(work.rowsPerRequest,Fixture::rows);
            EXPECT_EQ(work.sourceCapacity,Fixture::tokens);
            EXPECT_EQ(work.firstRequiredLogicalRow,2051);
            ASSERT_EQ(work.sources.size(),1u);
            EXPECT_EQ(work.sources[0].tensor,&f.replay);
            EXPECT_EQ(work.sources[0].frontierByteOffset,0u);
            EXPECT_EQ(work.sources[0].liveByteOffset,sizeof(std::int32_t));
        }
        EXPECT_FALSE(k.projection_row_work);
        EXPECT_EQ(k.A,&f.input);EXPECT_EQ(k.C,&f.raw);
        EXPECT_EQ(k.m,Fixture::rows);EXPECT_EQ(k.n,128);EXPECT_EQ(k.k,Fixture::hidden);
        EXPECT_TRUE(k.force_decode_equivalent_verifier_prefill);
        EXPECT_EQ(i.rawKey.tensor,&f.raw);EXPECT_EQ(i.pooledKeys.tensor,&f.pools);
        EXPECT_EQ(i.selected.tensor,&f.selected);EXPECT_EQ(i.selectedCount.tensor,&f.count);
        EXPECT_FALSE(i.transactionPrior);
        const auto keyNow=key->graphCaptureVariantSignature();
        if(!keyIdentity)keyIdentity=keyNow;
        EXPECT_EQ(keyNow,keyIdentity);
        if(requestsEntry) {
            EXPECT_EQ(query->graphCaptureVariantSignature(),queryIdentities[phase]);
            EXPECT_EQ(indexer->graphCaptureVariantSignature(),indexerIdentities[phase]);
        } else {
            queryIdentities[phase]=query->graphCaptureVariantSignature();
            indexerIdentities[phase]=indexer->graphCaptureVariantSignature();
        }
    }
    EXPECT_EQ(queryIdentities[0],queryIdentities[1]);
    EXPECT_NE(queryIdentities[1],queryIdentities[2]);
    EXPECT_EQ(indexerIdentities[0],indexerIdentities[1]);
    EXPECT_NE(indexerIdentities[1],indexerIdentities[2]);
}

/** @test Invalid typed phase never chooses AllLive or mutates a partially published graph. */
TEST(Qwen4ExpSparseAttentionBuilder, QueryDemandRejectsUnknownPhaseBeforeGraphMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;ComputeGraph graph;f.seed(graph);
    auto policy=f.policy();policy.executionPhase=static_cast<ForwardExecutionPhase>(255);
    EXPECT_THROW((void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    EXPECT_THROW((void)Builder::appendRequests(graph,f.consumer(),policy,f.weights(),
        DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
}

/** @test An admission-selected short program omits the node and all of its edges. */
TEST(Qwen4ExpSparseAttentionBuilder, ProvenNoRankingOmitsProjectionOnlyForExplicitDecode)
{
    using Phase=ForwardExecutionPhase;
    using Program=SparseQueryExecutionProgram;
    using Demand=qwen4exp::QwenSparseQueryDemand;
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    for (const auto phase : {Phase::Prefill,Phase::Decode})
        for (const auto program : {Program::RankingPossible,Program::ProvenNoRanking})
            for (const bool requestsEntry : {false,true})
            {
                SCOPED_TRACE(::testing::Message()<<"phase="<<int(phase)<<" program="<<int(program)<<" requests="<<requestsEntry);
                ComputeGraph graph;f.seed(graph);
                auto policy=f.policy();policy.executionPhase=phase;policy.queryExecutionProgram=program;
                policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
                if(phase==Phase::Prefill && program==Program::ProvenNoRanking) {
                    EXPECT_THROW((void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),std::invalid_argument);
                    EXPECT_EQ(graph.size(),1u);
                    continue;
                }
                if(requestsEntry)
                    (void)Builder::appendRequests(graph,f.consumer(),policy,f.weights(),
                        DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}});
                else (void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers());
                const bool omit=phase==Phase::Decode && program==Program::ProvenNoRanking;
                EXPECT_EQ(graph.size(),omit?5u:6u);
                EXPECT_EQ(graph.getNode("qsa.qsa_query_proj")==nullptr,omit);
                const auto *node=graph.getNode("qsa.qsa_indexer");ASSERT_NE(node,nullptr);
                const auto *stage=dynamic_cast<const QwenSparseAttentionStage *>(node->stage.get());ASSERT_NE(stage,nullptr);
                EXPECT_EQ(std::get<QwenSparseAttentionStage::Indexer>(stage->getParams().operation).queryDemand,
                    omit?Demand::ProvenNoRanking:phase==Phase::Decode?Demand::RankingOnly:Demand::AllLive);
                EXPECT_EQ(std::find(node->dependencies.begin(),node->dependencies.end(),"qsa.qsa_query_proj")
                    ==node->dependencies.end(),omit);
                for(const auto *name:{"qsa.qsa_key_proj","ready"})
                    EXPECT_NE(std::find(node->dependencies.begin(),node->dependencies.end(),name),node->dependencies.end());
                const auto *key=dynamic_cast<const GEMMStage *>(graph.getNode("qsa.qsa_key_proj")->stage.get());ASSERT_NE(key,nullptr);
                EXPECT_FALSE(key->getParams().projection_row_work);
                EXPECT_EQ(key->getParams().C,&f.raw);
                ASSERT_NE(graph.getNode("qsa.attention"),nullptr);
                ASSERT_NE(graph.getNode("qsa.qsa_commit_partial"),nullptr);
            }
}

/** @test Physical omission leaves required output/state bytes and unauthored query banks unchanged. */
TEST(Qwen4ExpSparseAttentionBuilder, ProvenNoRankingPreservesCompleteCPUOperationBytes)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture full,shortWindow;
    for(auto *f:{&full,&shortWindow}) {
        for(auto *tensor:{&f->query,&f->prepared,&f->scores,&f->raw,&f->pools,&f->output})
            std::fill_n(tensor->mutable_data(),tensor->numel(),-37.25f);
        std::fill_n(static_cast<int *>(f->selected.raw_mutable_data()),f->selected.numel(),-97);
        std::fill_n(static_cast<int *>(f->count.raw_mutable_data()),f->count.numel(),-98);
    }
    for(auto *f:{&full,&shortWindow}) {
        ComputeGraph graph;f->seed(graph);
        auto policy=f->policy();policy.executionPhase=ForwardExecutionPhase::Decode;
        policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
        policy.queryExecutionProgram=f==&full ? SparseQueryExecutionProgram::RankingPossible : SparseQueryExecutionProgram::ProvenNoRanking;
        (void)Builder::append(graph,f->consumer(),policy,f->weights(),f->buffers());
        CPUDeviceContext context(DeviceId::cpu(),1);
        // The complete sibling owns a causal query-work descriptor. Prepare
        // its immutable source addresses at the same cold launch boundary as
        // the graph executor before comparing either program's numeric work.
        for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->prepareGraphLaunch(&context,nullptr));
        for(auto *stage:graph.getExecutionStages())ASSERT_TRUE(stage->execute(&context));
    }
    for(const auto &[a,b]:{std::pair{&full.raw,&shortWindow.raw},std::pair{&full.prior,&shortWindow.prior},
            std::pair{&full.pools,&shortWindow.pools},std::pair{&full.output,&shortWindow.output}})
        EXPECT_EQ(std::memcmp(a->data(),b->data(),a->size_bytes()),0);
    EXPECT_EQ(std::memcmp(full.selected.raw_data(),shortWindow.selected.raw_data(),full.selected.size_bytes()),0);
    EXPECT_EQ(std::memcmp(full.count.raw_data(),shortWindow.count.raw_data(),full.count.size_bytes()),0);
    for(const auto *tensor:{&shortWindow.query,&shortWindow.prepared,&shortWindow.scores})
        for(std::size_t i=0;i<tensor->numel();++i)EXPECT_EQ(tensor->data()[i],-37.25f);
}

/** @test Invalid programs and unphased omission cannot mutate a graph or manufacture a proof. */
TEST(Qwen4ExpSparseAttentionBuilder, ProvenNoRankingRejectsInvalidProgramBeforeGraphMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f;
    for(const auto program:{SparseQueryExecutionProgram::ProvenNoRanking,static_cast<SparseQueryExecutionProgram>(-1),static_cast<SparseQueryExecutionProgram>(2)})
        for(const auto phase:std::array<std::optional<ForwardExecutionPhase>,3>{std::nullopt,ForwardExecutionPhase::Prefill,ForwardExecutionPhase::Decode}) {
            if(program==SparseQueryExecutionProgram::ProvenNoRanking && phase==ForwardExecutionPhase::Decode)continue;
            ComputeGraph graph;f.seed(graph);
            auto policy=f.policy();policy.queryExecutionProgram=program;policy.executionPhase=phase;
            EXPECT_THROW((void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),std::invalid_argument);
            EXPECT_EQ(graph.size(),1u);
            EXPECT_THROW((void)Builder::appendRequests(graph,f.consumer(),policy,f.weights(),
                DeviceRequestRowRanges::fullyActive(Fixture::rows),{{0,f.buffers()}}),std::invalid_argument);
            EXPECT_EQ(graph.size(),1u);
        }
}

/** @test Omitting indexer queries cannot weaken the retained raw-key arithmetic contract. */
TEST(Qwen4ExpSparseAttentionBuilder, ProvenNoRankingStillRequiresExplicitGroupedProjectionArithmetic)
{
    llaminar::v2::ThreadCountGuard threads(1);
    Fixture f; ComputeGraph graph; f.seed(graph);
    auto policy=f.policy(); policy.executionPhase=ForwardExecutionPhase::Decode;
    policy.queryExecutionProgram=SparseQueryExecutionProgram::ProvenNoRanking;
    ASSERT_EQ(policy.projectionArithmetic,Qwen4ExpProjectionArithmetic::Ordinary);
    EXPECT_THROW((void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()),std::invalid_argument);
    EXPECT_EQ(graph.size(),1u);
    policy.projectionArithmetic=Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier;
    EXPECT_NO_THROW((void)Builder::append(graph,f.consumer(),policy,f.weights(),f.buffers()));
    EXPECT_EQ(graph.size(),5u);
}
