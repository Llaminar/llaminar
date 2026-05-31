/**
 * @file Test__CUDAHybridKVCacheReset.cpp
 * @brief Regression tests for CUDA hybrid KV/GDN cache reset semantics.
 *
 * Exercises clear() and clear_layer() on CUDA hybrid caches that compress
 * full-attention layers and keep GDN recurrence/short-conv GPU state in
 * cache-owned kernels. The tests compare reset cache outputs against a fresh
 * cache while asserting that reset preserves the kernel object identities.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#include "backends/DeviceId.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "tensors/TensorKernels.h"
#endif

namespace
{
#ifdef HAVE_CUDA
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    /// @brief Throws when a CUDA runtime call fails, preserving the failing operation name.
    void checkCuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(status));
        }
    }

    /// @brief Returns true when at least one CUDA device is visible to the runtime.
    bool hasCUDA()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    /// @brief RAII wrapper for a device FP32 buffer used by direct CUDA kernel calls.
    struct CudaFloatBuffer
    {
        float *ptr = nullptr;
        size_t count = 0;

        explicit CudaFloatBuffer(size_t n) : count(n)
        {
            if (count > 0)
                checkCuda(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(float)), "cudaMalloc");
        }

        explicit CudaFloatBuffer(const std::vector<float> &host) : CudaFloatBuffer(host.size())
        {
            if (count > 0)
            {
                checkCuda(cudaMemcpy(ptr, host.data(), count * sizeof(float), cudaMemcpyHostToDevice),
                          "cudaMemcpy host-to-device");
            }
        }

        ~CudaFloatBuffer()
        {
            if (ptr)
                (void)cudaFree(ptr);
        }

        CudaFloatBuffer(const CudaFloatBuffer &) = delete;
        CudaFloatBuffer &operator=(const CudaFloatBuffer &) = delete;

        std::vector<float> toHost() const
        {
            std::vector<float> host(count);
            if (count > 0)
            {
                checkCuda(cudaMemcpy(host.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost),
                          "cudaMemcpy device-to-host");
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

    /// @brief Creates a CUDA hybrid cache through KernelFactory so GDN kernels are initialized.
    HybridCacheHandle createHybridCache()
    {
        auto hybrid_config = makeHybridConfig();
        llaminar::v2::kernels::KVCacheConfig config;
        config.precision = llaminar2::ActivationPrecision::FP32;
        config.device = llaminar2::DeviceId::cuda(0);
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

    /// @brief Runs one CUDA short-conv decode step and returns the host output.
    std::vector<float> runConvDecode(llaminar2::IHybridKVCache *cache, int layer, float seed)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->conv_kernel || state->conv_kernel_size <= 1 || state->conv_state.empty())
            throw std::runtime_error("invalid GDN convolution state in CUDA hybrid KV cache test");

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

        CudaFloatBuffer d_input(input);
        CudaFloatBuffer d_weight(weight);
        CudaFloatBuffer d_output(static_cast<size_t>(channels));

        state->conv_kernel->setGPUStream(nullptr);
        if (!state->conv_kernel->forward(
                d_input.ptr, d_weight.ptr, nullptr,
                d_output.ptr, state->conv_state.data(),
                /*seq_len=*/1, channels, kernel_size,
                /*apply_silu=*/false))
        {
            throw std::runtime_error("CUDA short-conv recurrent decode failed");
        }
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after short-conv");
        return d_output.toHost();
    }

    /// @brief Runs one CUDA GDN recurrence decode step and returns the host output.
    std::vector<float> runRecurrenceDecode(llaminar2::IHybridKVCache *cache, int layer, float seed)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->rec_kernel || state->n_v_heads <= 0 || state->d_k <= 0 || state->d_v <= 0)
            throw std::runtime_error("invalid GDN recurrence state in CUDA hybrid KV cache test");

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

        CudaFloatBuffer d_q(q), d_k_buf(k), d_v_buf(v), d_alpha(alpha), d_beta(beta), d_a_log(a_log), d_dt_bias(dt_bias);
        CudaFloatBuffer d_output(v_count);

        state->rec_kernel->setGPUStream(nullptr);
        if (!state->rec_kernel->recurrent_step(
                d_q.ptr, d_k_buf.ptr, d_v_buf.ptr,
                d_alpha.ptr, d_beta.ptr,
                d_a_log.ptr, d_dt_bias.ptr,
                d_output.ptr, state->recurrence_state.data(),
                n_heads, d_k, d_v,
                /*use_qk_l2norm=*/true))
        {
            throw std::runtime_error("CUDA GDN recurrent decode failed");
        }
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after GDN recurrence");
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

    /// @brief Appends two tokens to the compressed full-attention entry through CUDA device pointers.
    void appendFullAttentionToken(llaminar2::IKVCache *cache, int layer)
    {
        constexpr int tokens = 2;
        std::vector<float> k = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> v = {-1.0f, -2.0f, -3.0f, -4.0f};
        CudaFloatBuffer d_k(k), d_v(v);
        auto *cuda_cache = dynamic_cast<llaminar2::ICUDARingKVCache *>(cache);
        if (!cuda_cache)
            throw std::runtime_error("hybrid cache does not expose ICUDARingKVCache");
        if (!cuda_cache->append(layer, 0, d_k.ptr, d_v.ptr, tokens, 0))
            throw std::runtime_error("ICUDARingKVCache append failed");
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize after KV append");
    }
#endif
} // namespace

#ifdef HAVE_CUDA

TEST(Test__CUDAHybridKVCacheReset, ClearPreservesCacheAndGDNKernelObjectsButMatchesFreshState)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

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

TEST(Test__CUDAHybridKVCacheReset, ClearLayerResetsGDNGPUKernelState)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

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

TEST(Test__CUDAHybridKVCacheReset, HybridPrefixStateRoundTripRestoresHostStateAndPreservesKernels)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

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

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.total_layers, 3);
    EXPECT_EQ(metadata.gdn_layers, 2);
    ASSERT_GT(metadata.host_bytes, 0u);
    EXPECT_EQ(metadata.device_bytes, 0u);
    EXPECT_FALSE(metadata.has_device_kernel_state);

    std::vector<uint8_t> payload(metadata.host_bytes);
    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(desc, payload.data(), nullptr));

    cache.owner->clear();
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    EXPECT_FLOAT_EQ(state0->recurrence_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state0->conv_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state2->recurrence_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state2->conv_state[0], 0.0f);

    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(desc, payload.data(), nullptr));
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    EXPECT_FLOAT_EQ(state0->recurrence_state[0], 10.0f);
    EXPECT_FLOAT_EQ(state0->conv_state[0], 11.0f);
    EXPECT_FLOAT_EQ(state2->recurrence_state[0], 20.0f);
    EXPECT_FLOAT_EQ(state2->conv_state[0], 21.0f);
}

TEST(Test__CUDAHybridKVCacheReset, ClearLayerResetsCompressedFullAttentionEntry)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    auto cache = createHybridCache();

    appendFullAttentionToken(cache.owner.get(), /*layer=*/1);
    ASSERT_EQ(cache.owner->get_cached_tokens(1, 0), 2);

    cache.owner->clear_layer(/*layer=*/1);

    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 0)
        << "clear_layer(global FA layer) must reset the compressed parent-cache entry";
}

#else

TEST(Test__CUDAHybridKVCacheReset, SkipsWithoutCUDA)
{
    GTEST_SKIP() << "CUDA support not compiled";
}

#endif
