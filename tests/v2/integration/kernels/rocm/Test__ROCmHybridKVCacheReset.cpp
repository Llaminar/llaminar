/**
 * @file Test__ROCmHybridKVCacheReset.cpp
 * @brief Regression tests for ROCm hybrid KV/GDN cache reset semantics.
 *
 * Exercises the standard clear()/clear_layer() APIs on ROCm hybrid caches that
 * compress full-attention layers and keep GDN recurrence/short-conv GPU state
 * in cache-owned kernels. The tests compare a reset cache against a freshly
 * constructed cache while asserting that reset does not recreate kernel objects.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

#include "backends/DeviceId.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/kvcache/ROCmRingKVCache.h"
#include "tensors/TensorKernels.h"
#endif

namespace
{
#ifdef HAVE_ROCM
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    /// @brief Returns true when at least one ROCm device is visible to HIP.
    bool hasROCm()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    /// @brief RAII wrapper for a device FP32 buffer used by direct ROCm kernel calls.
    struct HipFloatBuffer
    {
        float *ptr = nullptr;
        size_t count = 0;

        explicit HipFloatBuffer(size_t n) : count(n)
        {
            if (count > 0)
                EXPECT_EQ(hipMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(float)), hipSuccess);
        }

        explicit HipFloatBuffer(const std::vector<float> &host) : HipFloatBuffer(host.size())
        {
            if (count > 0)
            {
                EXPECT_EQ(hipMemcpy(ptr, host.data(), count * sizeof(float), hipMemcpyHostToDevice), hipSuccess);
            }
        }

        ~HipFloatBuffer()
        {
            if (ptr)
                (void)hipFree(ptr);
        }

        HipFloatBuffer(const HipFloatBuffer &) = delete;
        HipFloatBuffer &operator=(const HipFloatBuffer &) = delete;

        std::vector<float> toHost() const
        {
            std::vector<float> host(count);
            if (count > 0)
            {
                EXPECT_EQ(hipMemcpy(host.data(), ptr, count * sizeof(float), hipMemcpyDeviceToHost), hipSuccess);
            }
            return host;
        }
    };

    struct HybridCacheHandle
    {
        std::unique_ptr<llaminar2::IKVCache> owner;
        llaminar2::IHybridKVCache *hybrid = nullptr;
    };

    /// @brief Builds a tiny Qwen3.5-style layer map with GDN/FA/GDN layers.
    llaminar2::HybridKVCacheConfig makeHybridConfig()
    {
        llaminar2::HybridKVCacheConfig hybrid;
        hybrid.layer_types = {"gdn", "full_attention", "gdn"};
        hybrid.gdn_conv_kernel_size = 3;
        hybrid.gdn_state_size = 2;
        hybrid.gdn_inner_size = 2;
        hybrid.gdn_group_count = 1;
        hybrid.gdn_time_step_rank = 1;
        hybrid.n_heads = 1;
        hybrid.local_n_heads = 0;
        return hybrid;
    }

    /// @brief Creates a ROCm hybrid cache through KernelFactory so GDN kernels are initialized.
    HybridCacheHandle createHybridCache()
    {
        auto hybrid_config = makeHybridConfig();
        llaminar::v2::kernels::KVCacheConfig config;
        config.precision = llaminar2::ActivationPrecision::FP32;
        config.device = llaminar2::DeviceId::rocm(0);
        config.num_layers = static_cast<int>(hybrid_config.layer_types.size());
        config.batch_size = 1;
        config.max_seq_len = 8;
        config.n_kv_heads = 1;
        config.head_dim = 2;
        config.hybrid_config = &hybrid_config;

        HybridCacheHandle handle;
        handle.owner = KernelFactory::createKVCache(config);
        handle.hybrid = dynamic_cast<llaminar2::IHybridKVCache *>(handle.owner.get());
        if (!handle.hybrid)
            throw std::runtime_error("KernelFactory did not create an IHybridKVCache");
        return handle;
    }

    /// @brief Compares two vectors elementwise with a label that names the failing phase.
    void expectNearVector(const std::vector<float> &actual,
                          const std::vector<float> &expected,
                          float tol,
                          const char *label)
    {
        ASSERT_EQ(actual.size(), expected.size()) << label;
        for (size_t i = 0; i < actual.size(); ++i)
        {
            ASSERT_NEAR(actual[i], expected[i], tol) << label << " at index " << i;
        }
    }

    /// @brief Generates small deterministic FP32 inputs that keep recurrence math finite.
    std::vector<float> pattern(size_t count, float seed)
    {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
        {
            const float a = static_cast<float>((i % 7) + 1) * 0.0071f;
            const float b = static_cast<float>((i % 5) - 2) * 0.0013f;
            values[i] = 0.01f * seed + a + b;
        }
        return values;
    }

    /// @brief Runs one ROCm short-conv decode step and returns the host output.
    std::vector<float> runConvDecode(llaminar2::IHybridKVCache *cache, int layer, float seed)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->conv_kernel || state->conv_kernel_size <= 1 || state->conv_state.empty())
            throw std::runtime_error("invalid GDN convolution state in ROCm hybrid KV cache test");

        const int channels = static_cast<int>(state->conv_state.size() /
                                              static_cast<size_t>(state->conv_kernel_size - 1));
        const int kernel_size = state->conv_kernel_size;
        auto input = pattern(static_cast<size_t>(channels), seed);
        std::vector<float> weight(static_cast<size_t>(channels) * static_cast<size_t>(kernel_size), 0.0f);
        for (int c = 0; c < channels; ++c)
        {
            // Make stale history observable: current token and both history slots contribute.
            weight[static_cast<size_t>(c) * kernel_size + 0] = 0.25f;
            weight[static_cast<size_t>(c) * kernel_size + 1] = -0.5f;
            weight[static_cast<size_t>(c) * kernel_size + 2] = 1.0f;
        }

        HipFloatBuffer d_input(input);
        HipFloatBuffer d_weight(weight);
        HipFloatBuffer d_output(static_cast<size_t>(channels));

        state->conv_kernel->setGPUStream(nullptr);
        EXPECT_TRUE(state->conv_kernel->forward(
            d_input.ptr, d_weight.ptr, nullptr,
            d_output.ptr, state->conv_state.data(),
            /*seq_len=*/1, channels, kernel_size,
            /*apply_silu=*/false));
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return d_output.toHost();
    }

    /// @brief Runs one in-place short-conv decode step and returns output plus exported GPU state.
    std::pair<std::vector<float>, std::vector<float>> runInPlaceConvDecodeAndExportState(
        llaminar2::IHybridKVCache *cache, int layer, const std::vector<float> &input)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->conv_kernel || state->conv_kernel_size <= 1 || state->conv_state.empty())
            throw std::runtime_error("invalid GDN convolution state in ROCm hybrid KV cache test");

        const int kernel_size = state->conv_kernel_size;
        const int state_len = kernel_size - 1;
        const int channels = static_cast<int>(input.size());
        if (state->conv_state.size() != static_cast<size_t>(channels * state_len))
            throw std::runtime_error("in-place short-conv test input does not match cache state shape");

        std::vector<float> weight(static_cast<size_t>(channels) * static_cast<size_t>(kernel_size), 0.0f);
        for (int c = 0; c < channels; ++c)
            weight[static_cast<size_t>(c) * kernel_size + state_len] = 2.0f;

        HipFloatBuffer d_input_output(input);
        HipFloatBuffer d_weight(weight);

        state->conv_kernel->setGPUStream(nullptr);
        if (!state->conv_kernel->allocateGPUScratch(channels))
            throw std::runtime_error("failed to allocate in-place short-conv scratch");
        if (!state->conv_kernel->forward(
                d_input_output.ptr, d_weight.ptr, nullptr,
                d_input_output.ptr, state->conv_state.data(),
                /*seq_len=*/1, channels, kernel_size,
                /*apply_silu=*/false))
            throw std::runtime_error("in-place short-conv decode failed");
        if (hipDeviceSynchronize() != hipSuccess)
            throw std::runtime_error("in-place short-conv decode did not synchronize successfully");

        std::vector<float> exported_state(state->conv_state.size());
        if (!state->conv_kernel->exportState(exported_state.data(), nullptr, nullptr))
            throw std::runtime_error("failed to export in-place short-conv GPU state");
        return {d_input_output.toHost(), exported_state};
    }

    /// @brief Runs one ROCm GDN recurrence decode step and returns the host output.
    std::vector<float> runRecurrenceDecode(llaminar2::IHybridKVCache *cache, int layer, float seed)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->rec_kernel || state->n_v_heads <= 0 || state->d_k <= 0 || state->d_v <= 0)
            throw std::runtime_error("invalid GDN recurrence state in ROCm hybrid KV cache test");

        const int n_heads = state->n_v_heads;
        const int d_k = state->d_k;
        const int d_v = state->d_v;
        const size_t qk_count = static_cast<size_t>(n_heads) * static_cast<size_t>(d_k);
        const size_t v_count = static_cast<size_t>(n_heads) * static_cast<size_t>(d_v);

        auto q = pattern(qk_count, seed + 0.10f);
        auto k = pattern(qk_count, seed + 0.20f);
        auto v = pattern(v_count, seed + 0.30f);
        std::vector<float> alpha(static_cast<size_t>(n_heads), 0.0f);
        std::vector<float> beta(static_cast<size_t>(n_heads), 0.0f);   // sigmoid(beta) = 0.5
        std::vector<float> a_log(static_cast<size_t>(n_heads), -2.0f); // mild decay
        std::vector<float> dt_bias(static_cast<size_t>(n_heads), 0.0f);

        HipFloatBuffer d_q(q), d_k_buf(k), d_v_buf(v), d_alpha(alpha), d_beta(beta), d_a_log(a_log), d_dt_bias(dt_bias);
        HipFloatBuffer d_output(v_count);

        state->rec_kernel->setGPUStream(nullptr);
        EXPECT_TRUE(state->rec_kernel->recurrent_step(
            d_q.ptr, d_k_buf.ptr, d_v_buf.ptr,
            d_alpha.ptr, d_beta.ptr,
            d_a_log.ptr, d_dt_bias.ptr,
            d_output.ptr, state->recurrence_state.data(),
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
        EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);
        return d_output.toHost();
    }

    /// @brief Mutates both GDN GPU states so reset-vs-fresh comparisons are meaningful.
    void mutateGDNState(llaminar2::IHybridKVCache *cache, int layer)
    {
        (void)runConvDecode(cache, layer, 1.0f);
        (void)runConvDecode(cache, layer, 2.0f);
        (void)runRecurrenceDecode(cache, layer, 1.5f);
        (void)runRecurrenceDecode(cache, layer, 2.5f);
    }

    /// @brief Appends two tokens to the compressed full-attention entry.
    void appendFullAttentionToken(llaminar2::IKVCache *cache, int layer)
    {
        constexpr int tokens = 2;
        constexpr int kv_dim = 2;
        std::vector<float> k = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v = {-1.0f, -2.0f, -3.0f, -4.0f};
        HipFloatBuffer d_k(k), d_v(v);
        auto *rocm_cache = dynamic_cast<llaminar2::IROCmRingKVCache *>(cache);
        ASSERT_NE(rocm_cache, nullptr);
        ASSERT_TRUE(rocm_cache->append(layer, 0, d_k.ptr, d_v.ptr, tokens, 0));
    }
#endif
} // namespace

#ifdef HAVE_ROCM

TEST(Test__ROCmHybridKVCacheReset, ClearPreservesCacheAndGDNKernelObjectsButMatchesFreshState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto fresh = createHybridCache();

    ASSERT_EQ(cache.owner->n_layers(), 3);
    EXPECT_EQ(cache.hybrid->kvLayerCount(), 1);
    EXPECT_EQ(cache.hybrid->gdnLayerCount(), 2);

    auto *gdn0 = cache.hybrid->getGDNState(0);
    ASSERT_NE(gdn0, nullptr);
    auto *conv_ptr = gdn0->conv_kernel.get();
    auto *rec_ptr = gdn0->rec_kernel.get();
    ASSERT_NE(conv_ptr, nullptr);
    ASSERT_NE(rec_ptr, nullptr);

    appendFullAttentionToken(cache.owner.get(), /*layer=*/1);
    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 2);
    mutateGDNState(cache.hybrid, /*layer=*/0);

    cache.owner->clear();

    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 0);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr)
        << "clear() must not recreate cache-owned GDN conv kernels";
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr)
        << "clear() must not recreate cache-owned GDN recurrence kernels";

    auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 3.0f);
    auto fresh_conv = runConvDecode(fresh.hybrid, /*layer=*/0, 3.0f);
    expectNearVector(actual_conv, fresh_conv, 1e-5f, "conv output after clear vs fresh");

    auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 3.5f);
    auto fresh_rec = runRecurrenceDecode(fresh.hybrid, /*layer=*/0, 3.5f);
    expectNearVector(actual_rec, fresh_rec, 1e-4f, "recurrence output after clear vs fresh");
}

TEST(Test__ROCmHybridKVCacheReset, ClearLayerResetsGDNGPUKernelState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto fresh = createHybridCache();

    mutateGDNState(cache.hybrid, /*layer=*/0);
    cache.owner->clear_layer(/*layer=*/0);

    auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 4.0f);
    auto fresh_conv = runConvDecode(fresh.hybrid, /*layer=*/0, 4.0f);
    expectNearVector(actual_conv, fresh_conv, 1e-5f, "conv output after clear_layer vs fresh");

    auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 4.5f);
    auto fresh_rec = runRecurrenceDecode(fresh.hybrid, /*layer=*/0, 4.5f);
    expectNearVector(actual_rec, fresh_rec, 1e-4f, "recurrence output after clear_layer vs fresh");
}

TEST(Test__ROCmHybridKVCacheReset, InPlaceShortConvDecodeStoresRawProjectionInGPUState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state = cache.hybrid->getGDNState(0);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->conv_kernel_size, 3);

    const int state_len = state->conv_kernel_size - 1;
    ASSERT_GT(state_len, 0);
    ASSERT_EQ(state->conv_state.size() % static_cast<size_t>(state_len), 0u);
    const int channels = static_cast<int>(state->conv_state.size() / static_cast<size_t>(state_len));
    ASSERT_GT(channels, 0);

    std::vector<float> input(static_cast<size_t>(channels));
    for (int c = 0; c < channels; ++c)
        input[static_cast<size_t>(c)] = 0.25f * static_cast<float>(c + 1);

    const auto [output, exported_state] =
        runInPlaceConvDecodeAndExportState(cache.hybrid, /*layer=*/0, input);

    ASSERT_EQ(output.size(), input.size());
    ASSERT_EQ(exported_state.size(), input.size() * static_cast<size_t>(state_len));
    for (size_t c = 0; c < input.size(); ++c)
    {
        EXPECT_FLOAT_EQ(output[c], 2.0f * input[c])
            << "in-place decode output for channel " << c;
        EXPECT_FLOAT_EQ(exported_state[c * static_cast<size_t>(state_len)], 0.0f)
            << "history slot 0 for channel " << c;
        EXPECT_FLOAT_EQ(exported_state[c * static_cast<size_t>(state_len) + static_cast<size_t>(state_len - 1)], input[c])
            << "decode must store the raw QKV projection, not the in-place output, for channel " << c;
    }
}

TEST(Test__ROCmHybridKVCacheReset, DevicePointerStateExportRoundTripRestoresGPUKernelState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state = cache.hybrid->getGDNState(0);
    ASSERT_NE(state, nullptr);
    ASSERT_NE(state->conv_kernel, nullptr);
    ASSERT_NE(state->rec_kernel, nullptr);

    mutateGDNState(cache.hybrid, /*layer=*/0);

    const size_t conv_bytes = state->conv_kernel->stateBytes();
    const size_t rec_bytes = state->rec_kernel->stateBytes();
    ASSERT_GT(conv_bytes, 0u);
    ASSERT_GT(rec_bytes, 0u);
    ASSERT_EQ(conv_bytes % sizeof(float), 0u);
    ASSERT_EQ(rec_bytes % sizeof(float), 0u);

    std::vector<float> expected_conv(conv_bytes / sizeof(float));
    std::vector<float> expected_rec(rec_bytes / sizeof(float));
    ASSERT_TRUE(state->conv_kernel->exportState(expected_conv.data(), nullptr, nullptr));
    ASSERT_TRUE(state->rec_kernel->exportState(expected_rec.data(), nullptr, nullptr));

    HipFloatBuffer d_conv_snapshot(expected_conv.size());
    HipFloatBuffer d_rec_snapshot(expected_rec.size());
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    ASSERT_TRUE(state->conv_kernel->exportState(nullptr, d_conv_snapshot.ptr, stream));
    ASSERT_TRUE(state->rec_kernel->exportState(nullptr, d_rec_snapshot.ptr, stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    cache.owner->clear_layer(/*layer=*/0);
    ASSERT_TRUE(state->conv_kernel->importState(nullptr, d_conv_snapshot.ptr, stream));
    ASSERT_TRUE(state->rec_kernel->importState(nullptr, d_rec_snapshot.ptr, stream));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    std::vector<float> actual_conv(expected_conv.size());
    std::vector<float> actual_rec(expected_rec.size());
    ASSERT_TRUE(state->conv_kernel->exportState(actual_conv.data(), nullptr, nullptr));
    ASSERT_TRUE(state->rec_kernel->exportState(actual_rec.data(), nullptr, nullptr));

    expectNearVector(actual_conv, expected_conv, 1e-6f, "device-pointer short-conv state restore");
    expectNearVector(actual_rec, expected_rec, 1e-6f, "device-pointer recurrence state restore");
}

TEST(Test__ROCmHybridKVCacheReset, HybridPrefixStateRoundTripRestoresHostAndGPUStateAndPreservesKernels)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state0 = cache.hybrid->getGDNState(0);
    auto *state2 = cache.hybrid->getGDNState(2);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state2, nullptr);
    ASSERT_FALSE(state0->recurrence_state.empty());
    ASSERT_FALSE(state0->conv_state.empty());
    ASSERT_FALSE(state2->recurrence_state.empty());
    ASSERT_FALSE(state2->conv_state.empty());

    state0->recurrence_state[0] = 10.0f;
    state0->conv_state[0] = 11.0f;
    state2->recurrence_state[0] = 20.0f;
    state2->conv_state[0] = 21.0f;

    auto *conv_ptr = state0->conv_kernel.get();
    auto *rec_ptr = state0->rec_kernel.get();
    ASSERT_NE(conv_ptr, nullptr);
    ASSERT_NE(rec_ptr, nullptr);
    mutateGDNState(cache.hybrid, /*layer=*/0);

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.total_layers, 3);
    EXPECT_EQ(metadata.gdn_layers, 2);
    ASSERT_GT(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);
    EXPECT_TRUE(metadata.has_device_kernel_state);

    std::vector<uint8_t> payload(metadata.host_bytes + metadata.device_bytes);
    auto *host_payload = payload.data();
    auto *device_payload = payload.data() + metadata.host_bytes;
    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(desc, host_payload, device_payload));

    const auto expected_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto expected_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);

    cache.owner->clear();
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    EXPECT_FLOAT_EQ(state0->recurrence_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state0->conv_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state2->recurrence_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state2->conv_state[0], 0.0f);

    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(desc, host_payload, device_payload));
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    EXPECT_FLOAT_EQ(state0->recurrence_state[0], 10.0f);
    EXPECT_FLOAT_EQ(state0->conv_state[0], 11.0f);
    EXPECT_FLOAT_EQ(state2->recurrence_state[0], 20.0f);
    EXPECT_FLOAT_EQ(state2->conv_state[0], 21.0f);

    const auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);
    expectNearVector(actual_conv, expected_conv, 1e-5f, "conv output after hybrid prefix import");
    expectNearVector(actual_rec, expected_rec, 1e-4f, "recurrence output after hybrid prefix import");
}

TEST(Test__ROCmHybridKVCacheReset, ClearLayerResetsCompressedFullAttentionEntry)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();

    appendFullAttentionToken(cache.owner.get(), /*layer=*/1);
    ASSERT_EQ(cache.owner->get_cached_tokens(1, 0), 2);

    cache.owner->clear_layer(/*layer=*/1);

    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 0)
        << "clear_layer(global FA layer) must reset the compressed parent-cache entry";
}

#else

TEST(Test__ROCmHybridKVCacheReset, SkipsWithoutROCm)
{
    GTEST_SKIP() << "ROCm support not compiled";
}

#endif
