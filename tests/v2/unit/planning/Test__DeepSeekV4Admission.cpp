/**
 * @file Test__DeepSeekV4Admission.cpp
 * @brief Device-free admission regressions for DeepSeek-V4 model-owned attention state.
 *
 * DeepSeek-V4 keeps its raw window ring, compressor state and packed
 * compressed/indexer pools in model arena banks; the canonical cache is a
 * frontier-only hybrid of model_state layers. These tests pin the three
 * admission invariants that follow:
 *  - the GPU graph arena is priced from exactly the resolved schema roots the
 *    orchestrator registers (formerly the GPU estimate returned before the
 *    DeepSeek-V4 extension, so no bank was charged), and the banks scale with
 *    context only through the packed pools and bounded indexer scratch;
 *  - the canonical cache charges no KV payload (layers are ModelOwned);
 *  - no full-context K/V conversion scratch is admitted for a cache without
 *    attention layers (formerly ~4 GiB at 1M context);
 *  - the admitted roots equal, root by root, the actual DeepSeekV4Graph
 *    schema/resolver inventory the orchestrator registers (CPU and GPU
 *    allocators, several row envelopes).
 */
#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "execution/debug/BufferRole.h"
#include "memory/GraphArenaStorageLayout.h"
#include "models/deepseek4/DeepSeekV4BufferLayout.h"
#include "planning/ActivationMemoryEstimator.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "planning/WorkspaceMemoryEstimator.h"
#include "models/deepseek4/DeepSeekV4Graph.h"
#include "models/deepseek4/DeepSeekV4GraphConfigBuilder.h"
#include "../models/deepseek4/DeepSeekV4MetadataFixture.h"

#include <map>

#include <cstdint>
#include <string>

using namespace llaminar2;

namespace
{
/** @brief Release DeepSeek-V4-Flash geometry (43 layers: 21 CSA, 20 HCA). */
DeepSeekV4MemoryGeometry releaseGeometry()
{
    return {43, 4096, 4, 64, 512, 1024, 8, 1024, 128, 64, 128, 512, 256, 6, 21, 20, 64};
}

ModelMemoryProfile deepSeekV4Profile()
{
    ModelMemoryProfile profile;
    profile.architecture = "deepseek4";
    profile.n_layers = 43;
    profile.d_model = 4096;
    profile.d_ff = 2048;
    profile.n_heads = 64;
    profile.n_kv_heads = 1;
    profile.head_dim = 512;
    profile.vocab_size = 129280;
    profile.max_seq_len = 1 << 20;
    profile.expert_count = 256;
    profile.expert_used_count = 6;
    profile.expert_feed_forward_length = 2048;
    profile.expert_shared_feed_forward_length = 2048;
    profile.deepseek_v4 = releaseGeometry();
    return profile;
}

ActivationGraphMemoryGeometry graphGeometry(int rows, int context)
{
    ActivationGraphMemoryGeometry geometry;
    geometry.batch_size = 1;
    geometry.resident_graph_rows = rows;
    geometry.context_tokens = context;
    geometry.local_d_ff = 2048;
    geometry.local_n_heads = 64;
    geometry.local_n_kv_heads = 1;
    geometry.local_vocab = 129280;
    geometry.last_layer = 42;
    geometry.total_shards = 1;
    geometry.mtp_target_query_rows = 2;
    return geometry;
}

/** @brief Independent linear sum of resolved roots (GPU arenas are unpadded). */
std::size_t linearRootBytes(const GraphArenaStorageLayout &layout, bool deepSeekOnly)
{
    std::size_t bytes = 0;
    for (const auto &entry : layout.entries())
    {
        bool declared = false;
        for (const auto &declaration : deepSeekV4BufferDeclarations()) declared |= declaration.id == entry.id;
        if (deepSeekOnly && !declared) continue;
        std::size_t elements = 1;
        for (const auto dimension : entry.descriptor.shape) elements *= dimension;
        const bool half = entry.descriptor.tensor_type == BufferTensorType::BF16 ||
                          entry.descriptor.tensor_type == BufferTensorType::FP16;
        bytes += elements * (half ? 2 : 4);
    }
    return bytes;
}
} // namespace

TEST(DeepSeekV4Admission, GPUArenaPricesEveryRegisteredRootOnce)
{
    const auto profile = deepSeekV4Profile();
    const auto device = DeviceId::cuda(0);
    for (const int rows : {64, 512})
    {
        const auto small = graphGeometry(rows, 32768), large = graphGeometry(rows, 1 << 20);
        const std::size_t admitted = ActivationMemoryEstimator::estimate(profile, small, device);
        const auto layout = ActivationMemoryEstimator::declareSchemaStorage(profile, small, device);
        const std::size_t seeds = admitted - linearRootBytes(layout, false);
        EXPECT_EQ(seeds, sizeof(std::uint64_t)) << "only the request seed bank lies outside the schema";
        // The declared DeepSeek-V4 roots are exactly the extension inventory.
        const std::size_t trunk = 2ull * rows * 3 * 4096 * sizeof(float);
        EXPECT_EQ(linearRootBytes(layout, true) + trunk, deepSeekV4ArenaExtensionBytes(releaseGeometry(), rows, 32768));
        // Context scales only the packed pools and the bounded score scratch.
        const auto scratch = [&](int context) {
            return static_cast<std::size_t>(deepSeekV4IndexerScoreRows(rows, context)) *
                   ((static_cast<std::size_t>(context) + 3) / 4) * sizeof(float);
        };
        EXPECT_EQ(ActivationMemoryEstimator::estimate(profile, large, device) - admitted,
                  deepSeekV4PersistentPoolBytes(releaseGeometry(), 1 << 20) -
                      deepSeekV4PersistentPoolBytes(releaseGeometry(), 32768) + scratch(1 << 20) - scratch(32768));
        EXPECT_LE(scratch(1 << 20), kDeepSeekV4IndexerScoreBudgetBytes);
    }
    // 1M: 3.43 GiB of pools (21 * (584/4 + 68/4) + 20 * 584/128 bytes per token).
    EXPECT_EQ(deepSeekV4PersistentPoolBytes(releaseGeometry(), 1 << 20), 3684958208ull);
    EXPECT_EQ(deepSeekV4PersistentPoolBytes(releaseGeometry(), 32768), 115154944ull);
}

TEST(DeepSeekV4Admission, CanonicalCacheChargesNoKVPayload)
{
    const auto profile = deepSeekV4Profile();
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0)})
    {
        const auto estimate = PersistentStateMemoryEstimator::estimate(profile, device, 1, 1 << 20, 1, 1, 0, 64, 0, -1,
                                                                       "fp16", MTPStateRole::Disabled);
        EXPECT_EQ(estimate.main_full_attention_layers, 0);
        EXPECT_EQ(estimate.main_gdn_layers, 0);
        EXPECT_EQ(estimate.main_model_owned_layers, 43);
        EXPECT_EQ(estimate.main_kv_family, KVCacheFamily::Hybrid);
        EXPECT_EQ(estimate.main_kv_cache_bytes,
                  KVCacheMemoryEstimator::estimate(KVCacheFamily::Hybrid, 0, 1, 1 << 20, 1, 512, "fp16", device));
        EXPECT_LT(estimate.kv_cache_bytes, std::size_t{4096}) << "only frontier metadata remains";
        EXPECT_EQ(estimate.live_recurrent_state_bytes, 0u);
        EXPECT_FALSE(PersistentStateMemoryEstimator::isFullAttentionLayer(profile, 7));
        EXPECT_EQ(PersistentStateMemoryEstimator::layerStateKind(profile, 7), LayerStateKind::ModelOwned);
    }
}

TEST(DeepSeekV4Admission, NoKVConversionScratchWithoutAttentionLayers)
{
    // An ordinary dense model reserves full-context conversion scratch; the
    // same profile with model-owned attention state reserves none, so its
    // workspace is independent of the context horizon.
    auto profile = deepSeekV4Profile();
    WorkspaceMemoryGeometry geometry{
        .device = DeviceId::cuda(0),
        .device_compute_units = 82,
        .batch_size = 1,
        .resident_graph_rows = 64,
        .max_context_rows = 32768,
        .local_d_ff = 2048,
        .local_query_heads = 64,
        .local_kv_heads = 1,
        .first_layer = 0,
        .last_layer = 42,
        .total_shards = 1,
    };
    auto longContext = geometry;
    longContext.max_context_rows = 1 << 20;
    EXPECT_EQ(WorkspaceMemoryEstimator::estimate(profile, geometry),
              WorkspaceMemoryEstimator::estimate(profile, longContext));
    profile.deepseek_v4.reset();
    profile.architecture = "llama";
    profile.n_heads = 32;
    profile.n_kv_heads = 8;
    profile.head_dim = 128;
    geometry.local_query_heads = longContext.local_query_heads = 32;
    geometry.local_kv_heads = longContext.local_kv_heads = 8;
    EXPECT_GT(WorkspaceMemoryEstimator::estimate(profile, longContext),
              WorkspaceMemoryEstimator::estimate(profile, geometry) + (std::size_t{1} << 30))
        << "control: a full-attention cache converts the complete context";
}

TEST(DeepSeekV4Admission, AdmittedRootsEqualTheActualGraphInventory)
{
    const auto directory = test::deepSeekV4MetadataFixture();
    const auto profile = ModelMemoryProfile::fromGGUF(directory);
    ASSERT_TRUE(profile.deepseek_v4.has_value());
    for (const auto device : {DeviceId{DeviceType::CPU, 1}, DeviceId::cuda(0)})
        for (const int rows : {1, 3, 64})
        {
            SCOPED_TRACE(::testing::Message() << device.toString() << " rows=" << rows);
            GraphConfig config;
            config.default_device = device;
            DeepSeekV4GraphConfigBuilder::populateFromDirectory(directory, config);
            config.max_seq_len = 512;
            config.max_activation_rows = rows;
            config.max_request_count = 1;
            DeepSeekV4Graph graph(config, nullptr);
            // Exactly the orchestrator's registration (no retained MTP roots).
            const auto actual = GraphArenaStorageLayout::resolve(graph.getSchema(), graph.getResolverConfig(rows), {});

            ActivationGraphMemoryGeometry geometry;
            geometry.batch_size = 1;
            geometry.resident_graph_rows = rows;
            geometry.context_tokens = config.max_seq_len;
            geometry.local_d_ff = config.d_ff;
            geometry.local_n_heads = config.n_heads;
            geometry.local_n_kv_heads = config.n_kv_heads;
            geometry.local_vocab = config.vocab_size;
            geometry.last_layer = config.n_layers - 1;
            geometry.total_shards = 1;
            geometry.routed_compute_policy = config.moe.routed_compute_policy;
            geometry.mtp_target_query_rows = 1;
            geometry.mtp_terminal_logits_layout = config.mtpTerminalLogitsLayout();
            const auto admitted = ActivationMemoryEstimator::declareSchemaStorage(profile, geometry, device);

            std::map<BufferId, const BufferDescriptor *> want, got;
            for (const auto &entry : actual.entries()) ASSERT_TRUE(want.emplace(entry.id, &entry.descriptor).second);
            for (const auto &entry : admitted.entries()) ASSERT_TRUE(got.emplace(entry.id, &entry.descriptor).second);
            EXPECT_EQ(got.size(), want.size());
            for (const auto &[id, descriptor] : want)
            {
                SCOPED_TRACE(std::string(bufferIdName(id)) + " root " + descriptor->name);
                const auto found = got.find(id);
                ASSERT_NE(found, got.end());
                EXPECT_EQ(found->second->tensor_type, descriptor->tensor_type);
                EXPECT_EQ(found->second->shape, descriptor->shape);
            }
            EXPECT_EQ(linearRootBytes(admitted, false), linearRootBytes(actual, false));
            if (device.is_gpu())
                EXPECT_EQ(ActivationMemoryEstimator::estimate(profile, geometry, device),
                          linearRootBytes(actual, false) + sizeof(std::uint64_t));
        }
}
