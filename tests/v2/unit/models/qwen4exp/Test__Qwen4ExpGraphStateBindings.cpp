/**
 * @file Test__Qwen4ExpGraphStateBindings.cpp
 * @brief Prove early cache setup consumes the same QFN arena roles as forward graphs.
 *
 * No model weights are loaded. Exact metadata, small native cache frontiers and
 * arena-backed state prove initialization before the first ModelBuffers forward
 * publication, including physical rank-three-to-matrix flattening and PP offsets.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "execution/compute_stages/stages/qwen4exp/Qwen4ExpPLEBuilder.h"
#include "../../../utils/Qwen4ExpPlacementFixture.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/qwen4exp/Qwen4ExpBufferLayout.h"
#include "models/qwen4exp/Qwen4ExpTrunkBinding.h"
#include "memory/BufferArena.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "kernels/cpu/CPURingKVCache.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include <gtest/gtest.h>
#include <algorithm>

namespace llaminar2
{
/** @brief Observe the actual QFN PLE context factory inside its real invocation scope.
 * @details No cache, weight, source or executable is substituted. The factory under
 * test is the same one addModelLayerInput passes directly to the reusable builder.
 */
struct Qwen4ExpPLEProjectionTestPeer
{
    /** @brief Bind the exact invocation and delegate to the production context factory. */
    static Qwen4ExpPLEContext context(Qwen4ExpGraph &graph,const ForwardInput &input,DeviceId device)
    {
        QwenGraphBase::ForwardInputConstructionScope invocation(graph,input);
        return graph.pleContext(input,device,"actual_ple_");
    }
    /** @brief Keep the original scope while testing a different claimed complete invocation. */
    static Qwen4ExpPLEContext mismatchedInvocation(Qwen4ExpGraph &graph,const ForwardInput &original,const ForwardInput &claimed)
    {
        QwenGraphBase::ForwardInputConstructionScope invocation(graph,original);
        return graph.pleContext(claimed,claimed.device,"actual_ple_");
    }
};
}
namespace llaminar2::test
{
/** @test Core residual ownership survives graph-fragment remapping without an extension alias. */
TEST(Qwen4ExpGraphStateBindings, TrunkBindingUsesCanonicalCoreViewAndFragmentIdentity)
{
    ActivationBuffers buffers;
    EXPECT_THROW(qwen4ExpTrunkBinding(buffers), std::invalid_argument);
    FP32Tensor trunk{{2, 16}};
    buffers.current_hidden = &trunk;
    ASSERT_TRUE(buffers.extensions.empty());
    const auto main = qwen4ExpTrunkBinding(buffers);
    EXPECT_EQ(main.tensor, &trunk);
    EXPECT_EQ(main.buffer, BufferId::HIDDEN_STATE);
    buffers.binding_ids[BufferId::HIDDEN_STATE] = BufferId::MTP_PROJECTED;
    const auto predictor = qwen4ExpTrunkBinding(buffers);
    EXPECT_EQ(predictor.tensor, &trunk);
    EXPECT_EQ(predictor.buffer, BufferId::MTP_PROJECTED);
}

/** @test Real schema resolution preserves native state types through physical allocation. */
TEST(Qwen4ExpGraphStateBindings, DeclaredArenaRetainsIntegerHistoriesAndAllSparseBanks)
{
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    TensorFactory factory(mpi);
    GraphConfig config;
    config.default_device=DeviceId::cpu();config.max_seq_len=512;
    config.max_request_count=1;config.max_activation_rows=128;
    config.local_n_heads=24;config.local_n_kv_heads=2;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
    admitQwen4ExpTestTable(config);
        Qwen4ExpGraph graph(config);
    const auto schema=graph.getSchema();
    const auto resolver=graph.getResolverConfig(128);
    const auto requirements=BufferAllocator::resolveLayerBuffers(schema,resolver);
    BufferArena arena(ArenaConfig{.factory=&factory});
    // Allocate only persistent model state while preserving the actual schema,
    // resolver and typed arena path. No expert weights or GPU work are needed.
    for(const auto role:{BufferId::QFN_PLE_TOKEN_HISTORY,BufferId::QFN_PLE_CONV_HISTORY,
                         BufferId::QFN_QSA_POOLED_KEYS,BufferId::QFN_QSA_PARTIAL_KEYS})
    {
        const auto declaration=std::find_if(qwen4ExpBufferDeclarations().begin(),qwen4ExpBufferDeclarations().end(),
            [role](const auto &entry){return entry.id==role;});
        ASSERT_NE(declaration,qwen4ExpBufferDeclarations().end());
        const auto descriptor=std::find_if(requirements.buffers.begin(),requirements.buffers.end(),
            [&](const auto &entry){return entry.name==declaration->spec.name;});
        ASSERT_NE(descriptor,requirements.buffers.end());
        ASSERT_TRUE(arena.registerBuffer(role,*descriptor));
    }
    ASSERT_TRUE(arena.allocate());
    ASSERT_EQ(dynamic_cast<TensorBase *>(arena.getTensor(BufferId::QFN_PLE_TOKEN_HISTORY))->native_type(),TensorType::INT32);
    graph.setArena(&arena);
    auto cache=createCPURingKVCache(ActivationPrecision::FP32,mpi,48,1,512,1,128,DeviceId::cpu());
    ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr));
    const auto *owner=dynamic_cast<const Qwen4ExpAuxiliaryState *>(cache->auxiliaryState());
    ASSERT_NE(owner,nullptr);
    EXPECT_EQ(owner->bindings().sparseLayers.size(),12u);
    EXPECT_EQ(owner->bindings().pleHistories.size(),1u);
}

/** @test Arena binding precedes cache attachment and uses physical matrix geometry. */
TEST(Qwen4ExpGraphStateBindings, CacheSetupBindsDeclaredRolesBeforeFirstForward)
{
    MPIContext mpi{0,1,MPI_COMM_WORLD};
    for(const int first:{0,3,4})
    {
        SCOPED_TRACE(first);
        GraphConfig config;
        config.default_device=DeviceId::cpu();config.pp_layer_offset=first;
        config.n_layers=first==3?1:4;config.max_seq_len=8;config.max_request_count=1;
        config.max_activation_rows=1;config.local_n_heads=24;config.local_n_kv_heads=2;
        Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
        admitQwen4ExpTestTable(config);
        Qwen4ExpGraph graph(config);
        FP32Tensor pools{{1,2*128}},partial{{1,4*128}},history{{1,18*10240}};
        INT32Tensor tokens{{1,4}};
        std::fill_n(pools.mutable_data(),pools.numel(),-71.0f);
        std::fill_n(partial.mutable_data(),partial.numel(),-72.0f);
        BufferArena arena;
        ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_QSA_POOLED_KEYS,&pools));
        ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_QSA_PARTIAL_KEYS,&partial));
        if(first==0)
        {
            ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_PLE_TOKEN_HISTORY,&tokens));
            ASSERT_TRUE(arena.registerExternalBuffer(BufferId::QFN_PLE_CONV_HISTORY,&history));
        }
        graph.setArena(&arena);
        // This cache fixture retains global-indexed slots without full GDN
        // allocation. The model-state adapter must retain that public identity.
        auto cache=createCPURingKVCache(ActivationPrecision::FP32,mpi,8,1,8,1,128,DeviceId::cpu());
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr));
        auto *owner=dynamic_cast<Qwen4ExpAuxiliaryState *>(cache->auxiliaryState());
        ASSERT_NE(owner,nullptr);
        ASSERT_EQ(owner->bindings().sparseLayers.size(),1u);
        const auto &sparse=owner->bindings().sparseLayers.front();
        EXPECT_EQ(sparse.layer,first==4?7:3);
        EXPECT_EQ(sparse.cacheLayer,sparse.layer);
        EXPECT_EQ(sparse.pooledKeys.tensor,&pools);
        EXPECT_EQ(sparse.pooledKeys.byteOffset,0u);
        EXPECT_EQ(owner->bindings().pleHistories.size(),first==0?1u:0u);
        for(std::size_t i=0;i<pools.numel();++i)EXPECT_EQ(pools.data()[i],-71.0f);
        for(std::size_t i=0;i<partial.numel();++i)EXPECT_EQ(partial.data()[i],0.0f);
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr));
        EXPECT_EQ(cache->auxiliaryState(),owner); // Setup replay preserves the sole owner.
    }
}
/** @test Actual QFN graph policy binds the exact original CPU/CUDA/ROCm verifier query, including M1 live0. */
TEST(Qwen4ExpGraphStateBindings, PLEVerifierContextRetainsOriginalQueryOnEveryBackend)
{
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const int capacity:{1,15})
        {
            GraphConfig config;config.default_device=device;config.max_request_count=1;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),config);
            config.grouped_mtp_verifier=true;config.compute_all_position_logits=true;
            config.mtp.enabled=true;config.mtp.draft_tokens=1;config.mtp.graph_capacity_draft_tokens=1;
            Qwen4ExpGraph graph(config);
            std::vector<int> lengths{capacity};std::int32_t live=0;
            ForwardInput input;input.device=device;input.batch_size=1;input.seq_len=capacity;
            input.execution_phase=ForwardExecutionPhase::Decode;input.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
            if(device.is_cpu())input.sequence_lengths=&lengths;else input.sequence_lengths_device=&live;
            const auto context=Qwen4ExpPLEProjectionTestPeer::context(graph,input,device);
            EXPECT_EQ(context.arithmetic,Qwen4ExpProjectionArithmetic::DecodeEquivalentVerifier);
            ASSERT_TRUE(context.projectionRows);ASSERT_TRUE(context.rows.liveRows);
            const auto *native=std::get_if<Qwen4ExpNativeIntInput>(&context.rows.liveRows->value);ASSERT_NE(native,nullptr);
            const auto *original=device.is_cpu()?lengths.data():input.sequence_lengths_device;
            EXPECT_EQ(native->owner,device);EXPECT_EQ(native->data,original);EXPECT_EQ(native->capacity,1u);
            EXPECT_EQ(context.projectionRows->captureIdentity(),DeviceRowRange::deviceCounted(capacity,original).captureIdentity());
            if(device.is_gpu())
            {
                // These are construction-only opaque GPU-address witnesses. No
                // backend is initialized and production never host-reads live.
                for(const int value:{capacity,1,0})
                {
                    live=value;const auto replay=Qwen4ExpPLEProjectionTestPeer::context(graph,input,device);
                    EXPECT_EQ(replay.projectionRows->captureIdentity(),context.projectionRows->captureIdentity());
                }
            }
            auto copied=input;
            EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::mismatchedInvocation(graph,input,copied),std::invalid_argument);
            auto bad=input;bad.batch_size=2;
            EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::context(graph,bad,device),std::invalid_argument);
            if(device.is_gpu())
            {
                bad=input;bad.execution_role=ForwardExecutionRole::MainInference;
                EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::context(graph,bad,device),std::invalid_argument);
                const auto foreign=device.is_cuda()?DeviceId::cuda(1):DeviceId::rocm(1);
                EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::context(graph,input,foreign),std::invalid_argument);
                bad=input;bad.sequence_lengths_device=nullptr;
                EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::context(graph,bad,device),std::invalid_argument);
            }
            else
            {
                lengths[0]=capacity-1;
                EXPECT_THROW(Qwen4ExpPLEProjectionTestPeer::context(graph,input,device),std::invalid_argument);
            }
        }
}

/** @test Ordinary PLE retains its established arithmetic and does not silently opt into verifier kernels. */
TEST(Qwen4ExpGraphStateBindings, PLEOrdinaryContextKeepsItsOriginalArithmeticPolicy)
{
    GraphConfig config;config.default_device=DeviceId::cpu();config.max_request_count=1;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
    Qwen4ExpGraph graph(config);std::vector<int> lengths{1};ForwardInput input;
    input.device=DeviceId::cpu();input.seq_len=1;input.batch_size=1;input.sequence_lengths=&lengths;
    input.execution_phase=ForwardExecutionPhase::Decode;input.execution_role=ForwardExecutionRole::MainInference;
    const auto context=Qwen4ExpPLEProjectionTestPeer::context(graph,input,input.device);
    EXPECT_EQ(context.arithmetic,Qwen4ExpProjectionArithmetic::Ordinary);EXPECT_FALSE(context.projectionRows);
    ASSERT_TRUE(context.rows.liveRows);
    EXPECT_EQ(std::get<Qwen4ExpNativeIntInput>(context.rows.liveRows->value).data,lengths.data());
}
/** @test PLE's grouped-only context rejects the broader common helper's CPU roles and every foreign phase. */
TEST(Qwen4ExpGraphStateBindings, PLEVerifierContextRejectsForeignRolesAndPhasesBeforeGraphWork)
{
    for(const auto device:{DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
        for(const int capacity:{1,15})
        {
            GraphConfig config;config.default_device=device;config.max_request_count=1;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(),config);
            config.grouped_mtp_verifier=true;config.compute_all_position_logits=true;
            config.mtp.enabled=true;config.mtp.draft_tokens=1;config.mtp.graph_capacity_draft_tokens=1;
            Qwen4ExpGraph graph(config);std::vector<int> lengths{capacity};std::int32_t nativeCount=0;
            ForwardInput input;input.device=device;input.seq_len=capacity;input.batch_size=1;
            if(device.is_cpu())input.sequence_lengths=&lengths;else input.sequence_lengths_device=&nativeCount;
            for(const auto role:{ForwardExecutionRole::MainInference,ForwardExecutionRole::MTPCondition,
                ForwardExecutionRole::GroupedMTPVerifier})
                for(const auto phase:{ForwardExecutionPhase::Prefill,ForwardExecutionPhase::Decode})
                {
                    if(role==ForwardExecutionRole::GroupedMTPVerifier && phase==ForwardExecutionPhase::Decode)continue;
                    input.execution_role=role;input.execution_phase=phase;
                    try
                    {
                        (void)Qwen4ExpPLEProjectionTestPeer::context(graph,input,device);
                        FAIL()<<"Expected PLE's exact grouped-verifier invocation guard";
                    }
                    catch(const std::invalid_argument &error)
                    { EXPECT_NE(std::string(error.what()).find("PLE grouped projections require GroupedMTPVerifier Decode invocation"),
                        std::string::npos)<<error.what(); }
                    EXPECT_EQ(input.execution_role,role);EXPECT_EQ(input.execution_phase,phase);
                    EXPECT_TRUE(graph.config().grouped_mtp_verifier);
                }
            // Rejection cannot poison later valid composition or replace the
            // exact original primitive/query authority. No backend is launched.
            input.execution_role=ForwardExecutionRole::GroupedMTPVerifier;input.execution_phase=ForwardExecutionPhase::Decode;
            const auto valid=Qwen4ExpPLEProjectionTestPeer::context(graph,input,device);
            ASSERT_TRUE(valid.projectionRows);
            const auto *count=device.is_cpu()?lengths.data():input.sequence_lengths_device;
            EXPECT_EQ(valid.projectionRows->captureIdentity(),DeviceRowRange::deviceCounted(capacity,count).captureIdentity());
        }
}
} // namespace llaminar2::test
