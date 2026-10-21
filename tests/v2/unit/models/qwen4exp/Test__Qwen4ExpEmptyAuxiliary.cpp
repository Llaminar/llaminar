/**
 * @file Test__Qwen4ExpEmptyAuxiliary.cpp
 * @brief Prove truthful QSA/PLE absence while canonical recurrent state stays cache-owned.
 *
 * Actual graph metadata and production hybrid caches exercise the failing setup
 * boundary. No source weights, GPU backend, empty extension, or dummy arena is used.
 */
#include "Qwen4ExpEmptyAuxiliaryFixture.h"
#include "../../../mocks/MockMPIContext.h"
#include "backends/BackendManager.h"
#include "kernels/IHybridKVCache.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>

namespace llaminar2::test
{
namespace
{
/** @brief Ensure the CPU factory has its actual host backend without initializing GPUs. */
void requireCPU() {if(!hasCPUBackend())initCPUBackend(-1);}
}
/** @test Actual source placement distinguishes absent, PLE-only, sparse-only and combined intervals. */
TEST(Qwen4ExpEmptyAuxiliary, ProjectsCanonicalSourceOwnedIntervals)
{
    using Content=Qwen4ExpGraph::AuxiliaryContent;
    const Qwen4ExpConfig source(qwen4MetadataFixture(false));
    const int ple=source.ngrams().decoder_layer;
    const auto sparse=std::find(source.mixers().begin(),source.mixers().end(),Qwen4ExpMixer::SparseAttention);
    ASSERT_NE(sparse,source.mixers().end());
    const int sparseLayer=static_cast<int>(sparse-source.mixers().begin());
    for(int layer:emptyAuxiliaryLayers())
        EXPECT_EQ(Qwen4ExpGraph(emptyAuxiliaryConfig(DeviceId::cpu(),layer)).auxiliaryContent(),Content::Absent)<<layer;
    EXPECT_EQ(Qwen4ExpGraph(emptyAuxiliaryConfig(DeviceId::cpu(),ple)).auxiliaryContent(),Content::PLEOnly);
    EXPECT_EQ(Qwen4ExpGraph(emptyAuxiliaryConfig(DeviceId::cpu(),sparseLayer)).auxiliaryContent(),Content::SparseOnly);
    EXPECT_EQ(Qwen4ExpGraph(emptyAuxiliaryConfig(DeviceId::cpu(),0,sparseLayer+1)).auxiliaryContent(),Content::PLEAndSparse);
    auto invalid=emptyAuxiliaryConfig(DeviceId::cpu(),0);
    invalid.pp_layer_offset=-1;
    EXPECT_THROW((void)Qwen4ExpGraph(invalid).auxiliaryContent(),std::invalid_argument);
    invalid.pp_layer_offset=source.mainLayers();
    EXPECT_THROW((void)Qwen4ExpGraph(invalid).auxiliaryContent(),std::invalid_argument);
    invalid.pp_layer_offset=0;invalid.n_layers=0;
    EXPECT_THROW((void)Qwen4ExpGraph(invalid).auxiliaryContent(),std::invalid_argument);
}
/** @test First and shifted GDN-only PP slices initialize repeatedly without adopting or resetting recurrence. */
TEST(Qwen4ExpEmptyAuxiliary, ActualCPUCachePreservesStateWithoutAnAuxiliaryOwner)
{
    requireCPU();MockMPIContext mpi;
    const auto layers=emptyAuxiliaryLayers();
    ASSERT_GT(layers.size(),1u);
    for(int layer:{layers.front(),layers.back()}) {
        SCOPED_TRACE(layer);
        auto config=emptyAuxiliaryConfig(DeviceId::cpu(),layer);
        Qwen4ExpGraph graph(config);
        auto cache=emptyAuxiliaryCache(config,mpi);
        auto *hybrid=dynamic_cast<IHybridKVCache *>(cache.get());ASSERT_NE(hybrid,nullptr);
        ASSERT_EQ(hybrid->kvLayerCount(),0);ASSERT_EQ(hybrid->gdnLayerCount(),1);
        auto *state=hybrid->getGDNState(layer);ASSERT_NE(state,nullptr);
        ASSERT_FALSE(state->recurrence_state.empty());ASSERT_FALSE(state->conv_state.empty());
        std::fill(state->recurrence_state.begin(),state->recurrence_state.end(),17.25f);
        std::fill(state->conv_state.begin(),state->conv_state.end(),11.5f);
        auto *recurrence=state->recurrence_state.data();auto *conv=state->conv_state.data();
        ASSERT_EQ(cache->auxiliaryState(),nullptr);
        // There is intentionally no arena: this participant owns no auxiliary bank.
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr));
        ASSERT_NO_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr));
        EXPECT_EQ(cache->auxiliaryState(),nullptr);
        EXPECT_EQ(state->recurrence_state.data(),recurrence);EXPECT_EQ(state->conv_state.data(),conv);
        EXPECT_TRUE(std::all_of(state->recurrence_state.begin(),state->recurrence_state.end(),[](float x){return x==17.25f;}));
        EXPECT_TRUE(std::all_of(state->conv_state.begin(),state->conv_state.end(),[](float x){return x==11.5f;}));
        ASSERT_TRUE(cache->resetRequestState(IKVCache::StateResetContext::testReinitialization(nullptr)));
        EXPECT_TRUE(std::all_of(state->recurrence_state.begin(),state->recurrence_state.end(),[](float x){return x==0.0f;}));
        EXPECT_TRUE(std::all_of(state->conv_state.begin(),state->conv_state.end(),[](float x){return x==0.0f;}));
        EXPECT_EQ(cache->auxiliaryState(),nullptr);
    }
}
/** @test Absence authenticates cache origin/context/requests/domain rather than bypassing setup validation. */
TEST(Qwen4ExpEmptyAuxiliary, RejectsForeignCacheGeometryAndStreamsBeforeMutation)
{
    requireCPU();MockMPIContext mpi;
    const auto layers=emptyAuxiliaryLayers();
    auto config=emptyAuxiliaryConfig(DeviceId::cpu(),layers.front());
    Qwen4ExpGraph graph(config);
    const auto reject=[&](GraphConfig cacheConfig) {
        auto cache=emptyAuxiliaryCache(cacheConfig,mpi);
        auto *state=dynamic_cast<IHybridKVCache *>(cache.get())->getGDNState(cacheConfig.pp_layer_offset);
        std::fill(state->recurrence_state.begin(),state->recurrence_state.end(),19.0f);
        EXPECT_THROW(graph.initializeCacheAuxiliaryState(*cache,nullptr),std::invalid_argument);
        EXPECT_EQ(cache->auxiliaryState(),nullptr);
        EXPECT_TRUE(std::all_of(state->recurrence_state.begin(),state->recurrence_state.end(),[](float x){return x==19.0f;}));
    };
    reject(emptyAuxiliaryConfig(DeviceId::cpu(),layers.back()));
    auto wrongContext=config;wrongContext.max_seq_len+=1;reject(wrongContext);
    auto wrongLayers=config;wrongLayers.n_layers=2;reject(wrongLayers);
    auto cache=emptyAuxiliaryCache(config,mpi);
    auto larger=config;larger.max_request_count+=1;
    EXPECT_THROW(Qwen4ExpGraph(larger).initializeCacheAuxiliaryState(*cache,nullptr),std::invalid_argument);
    EXPECT_THROW(graph.initializeCacheAuxiliaryState(*cache,reinterpret_cast<void *>(std::uintptr_t{1})),std::invalid_argument);
    auto gpu=config;gpu.default_device=DeviceId::cuda(0);
    EXPECT_THROW(Qwen4ExpGraph(gpu).initializeCacheAuxiliaryState(*cache,reinterpret_cast<void *>(std::uintptr_t{1})),std::invalid_argument);
    // Nonempty source ownership still requires actual declared banks and cannot
    // enter the no-state path merely because no arena has been provided.
    auto ple=emptyAuxiliaryConfig(DeviceId::cpu(),config.flash_next->ngrams().decoder_layer);
    auto pleCache=emptyAuxiliaryCache(ple,mpi);
    EXPECT_THROW(Qwen4ExpGraph(ple).initializeCacheAuxiliaryState(*pleCache,nullptr),std::invalid_argument);
    EXPECT_EQ(cache->auxiliaryState(),nullptr);EXPECT_EQ(pleCache->auxiliaryState(),nullptr);
}
} // namespace llaminar2::test
