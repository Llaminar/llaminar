/**
 * @file Test__PipelinePrefixState.cpp
 * @brief Model-free CPU/CUDA/ROCm proofs of stage-owned prefix archives.
 *
 * Real cache factories, admission, archive and restore exercise attention and
 * recurrent-only pipeline shards. Keep this fixture separate from the larger
 * generation/composition suite so prefix lifecycle iteration stays inexpensive.
 */
#include <gtest/gtest.h>
#include <algorithm>
#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/prefix_cache/PrefixCacheStateProbe.h"
#include "execution/prefix_cache/PrefixPayloadLayout.h"
#include "kernels/IHybridKVCache.h"
#include "models/qwen35/Qwen35Graph.h"
#include "planning/MemoryPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "tensors/TensorKernels.h"
#include "../../../mocks/MockModelContext.h"

namespace llaminar2::test
{
/** @brief Explicit backend selection keeps CPU-only preflight device-free. */
class PipelinePrefixState : public ::testing::TestWithParam<std::string>
{
protected:
    /** @brief Small real Qwen schema; model weights are unnecessary for state admission. */
    static GraphConfig stateConfig(DeviceId device)
    {
        GraphConfig config;
        config.d_model = 32;
        config.d_ff = config.d_ff_local = 64;
        config.n_heads = config.local_n_heads = 1;
        config.n_kv_heads = config.local_n_kv_heads = 1;
        config.head_dim = 32;
        config.n_layers = 3;
        config.total_n_layers = 4;
        config.vocab_size = config.vocab_local = 32;
        config.max_seq_len = 64;
        config.default_device = device;
        config.layer_types.assign(4, "full_attention");
        config.gdn.conv_kernel_size = 4;
        config.gdn.state_size = config.head_dim;
        config.gdn.inner_size = config.d_model;
        config.gdn.group_count = config.n_kv_heads;
        config.gdn.time_step_rank = config.n_heads;
        // Match the production Qwen configuration builder.  A retained GPU
        // MTP family owns the shifted-KV append inside the captured prefill
        // transaction; the generic GraphConfig default is intentionally not
        // a valid substitute for graph-native MTP execution.
        config.mtp_request_terminal_hidden_publication =
            MTPRequestTerminalHiddenPublicationPolicy::GraphCapturedDeviceGeometry;
        config.mtp_shifted_prefill_hidden_publication =
            MTPShiftedPrefillHiddenPublicationPolicy::GraphIntegratedKVTransaction;
        return config;
    }
};

/**
 * @brief Prefix preparation follows the stage's state, not whole-model geometry.
 *
 * An all-recurrent head has no attention payload and a middle stage has no
 * learned predictor. Both still participate in prefix restore. Exercise the
 * public lookup admission on real caches before any model arithmetic, across
 * the common public cache formats and active/capacity-only speculative policies.
 */
TEST_P(PipelinePrefixState, PrefixPreparationMatchesStageOwnership)
{
    initCPUBackend(-1);
    const auto device = GetParam() == "CPU" ? DeviceId::cpu()
                      : GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto prove = [&]
    {
        for (const auto precision : {KVCachePrecision::FP32, KVCachePrecision::FP16,
                                     KVCachePrecision::Q8_1})
            for (const bool execution_enabled : {false, true})
                for (int stage = 0; stage < 3; ++stage)
                {
                    SCOPED_TRACE(GetParam() + ":stage=" + std::to_string(stage) +
                        ":mtp=" + std::to_string(execution_enabled) +
                        ":format=" + std::to_string(static_cast<int>(precision)));
                    auto config = stateConfig(device);
                    config.d_model = config.head_dim = 64;
                    config.gdn.state_size = config.gdn.inner_size = 64;
                    config.layer_types = {"gdn", "full_attention", "gdn", "full_attention"};
                    config.kv_cache_precision = precision;
                    config.mtp.enabled = execution_enabled;
                    config.mtp.graph_capacity_draft_tokens = 15;
                    config.prefix_cache.enabled = true;
                    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                    config.prefix_cache.ram_budget_bytes = 1024 * 1024;
                    config.prefix_cache.block_size = 4;
                    ModelMemoryProfile profile;
                    profile.architecture = "qwen35";
                    profile.n_layers = 4;
                    profile.mtp_layer_count = 1;
                    profile.d_model = profile.head_dim = profile.d_ff = 64;
                    profile.vocab_size = 32;
                    profile.n_heads = profile.n_kv_heads = 1;
                    profile.max_seq_len = 64;
                    profile.full_attention_interval = 2;
                    profile.gdn_conv_kernel_size = 4;
                    profile.gdn_state_size = profile.gdn_inner_size = 64;
                    profile.gdn_group_count = profile.gdn_time_step_rank = 1;
                    DevicePlanConfig input;
                    input.world_rank = 0;
                    input.device = device;
                    input.device_total_bytes = device.is_gpu()
                        ? getBackendFor(device)->deviceMemoryTotal(0) : 8ull << 30;
                    input.device_free_bytes = device.is_gpu()
                        ? getBackendFor(device)->deviceMemoryFree(0) : 8ull << 30;
                    input.device_compute_units = 128;
                    input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                        .device = DeviceId::cpu(), .total_bytes = 1ull << 30,
                        .admission_available_bytes = 1ull << 30};
                    input.first_layer = input.last_layer = stage;
                    input.owns_embedding = stage == 0;
                    input.max_seq_len = input.activation_seq_len = 64;
                    input.kv_precision = kvCachePrecisionToString(precision);
                    input.mtp_enabled = true; // Price retained capacity, not request enablement.
                    input.mtp_target_query_rows = 16;
                    input.prefix_cache = config.prefix_cache;
                    auto memory = std::make_shared<PhysicalMemoryAuthority>(
                        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                            MemoryPlanner::plan(profile, {input}).physicalPlan()), 0);
                    auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
                    auto owner = DeviceGraphOrchestrator::createForTest({
                        .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph,
                        .physical_memory_authority = memory});
                    auto &runner = *owner;
                    runner.setPPStageConfig({.first_layer = stage, .last_layer = stage + 1,
                        .has_embedding = stage == 0, .has_lm_head = stage == 2});
                    ASSERT_TRUE(runner.initializeInferenceStateFromArena(1, 64, device));
                    const auto lookup = runner.lookupPrefix({});
                    ASSERT_TRUE(lookup.supported) << lookup.bypass_reason;
                    EXPECT_TRUE(lookup.cache_enabled);
                    EXPECT_EQ(lookup.requires_terminal_hidden, stage == 2 && execution_enabled);
                    EXPECT_EQ(lookup.requires_terminal_logits, stage == 2);
                    EXPECT_EQ(runner.inferenceState().mtp_kv_caches.size(), stage == 2 ? 1u : 0u);
                    if (stage != 0)
                        continue;

                    auto *hybrid = dynamic_cast<IHybridKVCache *>(runner.inferenceState().kv_cache.get());
                    ASSERT_NE(hybrid, nullptr);
                    void *stream = device.is_gpu()
                        ? GPUDeviceContextPool::instance().getContext(device).defaultStream() : nullptr;
                    std::vector<float> state_values(64 * 64, 0.0F);
                    const auto seed = [&](float value)
                    {
                        std::fill(state_values.begin(), state_values.end(), value);
                        if (device.is_gpu())
                            ASSERT_TRUE(hybrid->getRecurrenceKernel(0)->importState(
                                state_values.data(), nullptr, stream));
                        else
                            std::copy(state_values.begin(), state_values.end(), hybrid->getRecurrenceState(0));
                    };
                    const auto expect_state = [&](float expected)
                    {
                        if (device.is_gpu())
                        {
                            ASSERT_TRUE(hybrid->getRecurrenceKernel(0)->exportState(
                                state_values.data(), nullptr, stream));
                            ASSERT_TRUE(getBackendFor(device)->synchronizeStream(stream, 0));
                        }
                        else
                            std::copy_n(hybrid->getRecurrenceState(0), state_values.size(), state_values.begin());
                        EXPECT_TRUE(std::all_of(state_values.begin(), state_values.end(),
                            [expected](float value) { return value == expected; }));
                    };
                    const std::vector<int32_t> short_prompt{1, 2, 3, 4, 5, 6, 7, 8, 9};
                    auto long_prompt = short_prompt;
                    long_prompt.insert(long_prompt.end(), {10, 11, 12, 13});
                    const auto first_admission = runner.lookupPrefix(short_prompt);
                    seed(0.25F);
                    ASSERT_FALSE(HasFatalFailure());
                    ASSERT_TRUE(runner.harvestPrefix(first_admission, short_prompt, 9));
                    const auto short_hit = runner.lookupPrefix(short_prompt);
                    ASSERT_EQ(short_hit.cached_tokens, 9);
                    ASSERT_EQ(short_hit.blocks.size(), 1u)
                        << "Recurrent-only participants must not allocate empty ancestor blocks";
                    EXPECT_EQ(short_hit.blocks.back().layout.faKVBytes(), 0u);
                    EXPECT_EQ(short_hit.blocks.back().key.block_index, 2);

                    const auto second_admission = runner.lookupPrefix(long_prompt);
                    ASSERT_EQ(second_admission.cached_tokens, 9);
                    seed(0.5F);
                    ASSERT_FALSE(HasFatalFailure());
                    ASSERT_TRUE(runner.harvestPrefix(second_admission, long_prompt, 13));
                    const auto long_hit = runner.lookupPrefix(long_prompt);
                    ASSERT_EQ(long_hit.cached_tokens, 13);
                    ASSERT_EQ(long_hit.blocks.size(), 2u);
                    const auto clamped = long_hit.clampedTo(9);
                    ASSERT_EQ(clamped.cached_tokens, 9);
                    runner.resetInferenceState(InferenceStateResetRequest::requestBoundary("prefix-stage-proof"));
                    ASSERT_TRUE(runner.populatePrefix(clamped));
                    expect_state(0.25F);
                    ASSERT_TRUE(runner.populatePrefix(long_hit));
                    expect_state(0.5F);

                    auto changed_ancestor = long_prompt;
                    changed_ancestor.front() = 20;
                    EXPECT_EQ(runner.lookupPrefix(changed_ancestor).cached_tokens, 0)
                        << "A matching terminal chunk cannot hide a different earlier prompt";
                }
    };
    if (device.is_cpu())
        prove();
    else
    {
        ASSERT_NE(device.is_cuda() ? getCUDABackend() : getROCmBackend(), nullptr);
        GPUDeviceContextPool::instance().getContext(device).submitAndWait(prove);
    }
}

/**
 * @brief Nonzero PP offsets must not alias local ordinals with global layers.
 *
 * A one-layer slice cannot expose this defect: the namespaces overlap only
 * when the slice is wider than its first layer index. Real auto plans often
 * put layer zero on one device and layers 1..63 on another. Sweep offsets on
 * both sides of full-attention boundaries using the real cache/admission path.
 */
TEST_P(PipelinePrefixState, OffsetHybridSlicesUseGlobalLayerIdentity)
{
    initCPUBackend(-1);
    const auto device = GetParam() == "CPU" ? DeviceId::cpu()
                      : GetParam() == "CUDA" ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto prove = [&]
    {
        for (const auto precision : {KVCachePrecision::FP32, KVCachePrecision::FP16,
                                     KVCachePrecision::Q8_1})
            for (const int first_layer : {1, 2, 3, 5, 7})
            {
                SCOPED_TRACE(GetParam() + ":first=" + std::to_string(first_layer) +
                    ":format=" + std::to_string(static_cast<int>(precision)));
                auto config = stateConfig(device);
                config.d_model = config.head_dim = 64;
                config.gdn.state_size = config.gdn.inner_size = 64;
                config.n_layers = config.total_n_layers = 8;
                config.layer_types.assign(8, "gdn");
                config.layer_types[3] = config.layer_types[7] = "full_attention";
                config.kv_cache_precision = precision;
                config.prefix_cache.enabled = true;
                config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
                config.prefix_cache.ram_budget_bytes = 1024 * 1024;
                config.prefix_cache.block_size = 4;
                ModelMemoryProfile profile;
                profile.architecture = "qwen35";
                profile.n_layers = 8;
                profile.d_model = profile.head_dim = profile.d_ff = 64;
                profile.vocab_size = 32;
                profile.n_heads = profile.n_kv_heads = 1;
                profile.max_seq_len = 64;
                profile.full_attention_interval = 4;
                profile.gdn_conv_kernel_size = 4;
                profile.gdn_state_size = profile.gdn_inner_size = 64;
                profile.gdn_group_count = profile.gdn_time_step_rank = 1;
                DevicePlanConfig input;
                input.world_rank = 0;
                input.device = device;
                input.device_total_bytes = device.is_gpu()
                    ? getBackendFor(device)->deviceMemoryTotal(0) : 8ull << 30;
                input.device_free_bytes = device.is_gpu()
                    ? getBackendFor(device)->deviceMemoryFree(0) : 8ull << 30;
                input.device_compute_units = 128;
                input.associated_host_memory = PhysicalMemoryResource{.world_rank = 0,
                    .device = DeviceId::cpu(), .total_bytes = 1ull << 30,
                    .admission_available_bytes = 1ull << 30};
                input.first_layer = first_layer;
                input.last_layer = 7;
                input.owns_embedding = false;
                input.max_seq_len = input.activation_seq_len = 64;
                input.kv_precision = kvCachePrecisionToString(precision);
                input.prefix_cache = config.prefix_cache;
                auto memory = std::make_shared<PhysicalMemoryAuthority>(
                    std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(
                        MemoryPlanner::plan(profile, {input}).physicalPlan()), 0);
                auto graph = std::make_shared<Qwen35Graph>(config, nullptr);
                auto owner = DeviceGraphOrchestrator::createForTest({
                    .model_ctx = MockModelContext::createMinimal(), .graph_builder = graph,
                    .physical_memory_authority = memory});
                owner->setPPStageConfig({.first_layer = first_layer, .last_layer = 8,
                    .has_embedding = false, .has_lm_head = true});
                ASSERT_TRUE(owner->initializeInferenceStateFromArena(1, 64, device));
                const auto &cache = *owner->inferenceState().kv_cache;
                EXPECT_EQ(firstRestorablePrefixLayer(cache), first_layer <= 3 ? 3 : 7);
                const auto layout = buildDensePrefixPayloadLayout(cache, device, 4);
                std::vector<int> attention_layers;
                for (int ordinal = 0; ordinal < layout.fa_layers; ++ordinal)
                    attention_layers.push_back(prefixFALayerForIndex(cache, ordinal));
                EXPECT_EQ(attention_layers, first_layer <= 3 ? std::vector<int>({3, 7})
                                                            : std::vector<int>({7}));
                EXPECT_EQ(prefixFALayerForIndex(cache, -1), -1);
                EXPECT_EQ(prefixFALayerForIndex(cache, layout.fa_layers), -1);
                EXPECT_TRUE(layout.hasRestorableMainState());
                EXPECT_GT(layout.bytes_per_fa_layer_k, 0u);
                EXPECT_GT(layout.bytes_per_fa_layer_v, 0u);

                void *stream = device.is_gpu()
                    ? GPUDeviceContextPool::instance().getContext(device).defaultStream() : nullptr;
                std::vector<int> observed_layers;
                for (const auto &probe : inspectHybridGDNForPrefixProbe(cache, stream))
                    observed_layers.push_back(probe.global_layer);
                std::vector<int> expected_layers;
                for (int layer = first_layer; layer < 8; ++layer)
                    if (layer != 3 && layer != 7)
                        expected_layers.push_back(layer);
                EXPECT_EQ(observed_layers, expected_layers);
                const auto lookup = owner->lookupPrefix({});
                EXPECT_TRUE(lookup.supported) << lookup.bypass_reason;
            }
    };
    if (device.is_cpu())
        prove();
    else
    {
        ASSERT_NE(device.is_cuda() ? getCUDABackend() : getROCmBackend(), nullptr);
        GPUDeviceContextPool::instance().getContext(device).submitAndWait(prove);
    }
}

INSTANTIATE_TEST_SUITE_P(Backends, PipelinePrefixState,
    ::testing::Values("CPU"
#ifdef HAVE_CUDA
        , "CUDA"
#endif
#ifdef HAVE_ROCM
        , "ROCm"
#endif
    ), [](const auto &info) { return info.param; });
}
