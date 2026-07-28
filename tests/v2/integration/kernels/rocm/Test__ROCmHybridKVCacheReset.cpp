/**
 * @file Test__ROCmHybridKVCacheReset.cpp
 * @brief Regression tests for ROCm hybrid KV/GDN cache reset semantics.
 *
 * Exercises explicit request/layer reset transactions on ROCm hybrid caches that
 * compress full-attention layers and keep GDN recurrence/short-conv GPU state
 * in cache-owned kernels. The tests compare a reset cache against a freshly
 * constructed cache while asserting that reset does not recreate kernel objects.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>

#include "backends/DeviceId.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/HybridKVCacheConfig.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/IKVCache.h"
#include "kernels/KernelFactory.h"
#include "kernels/rocm/kvcache/ROCmRingKVCache.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/Tensors.h"
#include "tensors/TensorKernels.h"
#include "transfer/TransferEngine.h"
#endif

namespace
{
#ifdef HAVE_ROCM
    using KernelFactory = llaminar::v2::kernels::KernelFactory;

    /// @brief Throws when a HIP runtime call fails, preserving the failing operation name.
    void checkHip(hipError_t status, const char *operation)
    {
        if (status != hipSuccess)
        {
            throw std::runtime_error(std::string(operation) + " failed: " + hipGetErrorString(status));
        }
    }

    /// @brief Returns true when at least one ROCm device is visible to HIP.
    bool hasROCm()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    /**
     * @brief Owns one HIP stream for tests that exercise GPU-resident cache state.
     *
     * Hybrid prefix import/export requires an explicit stream so that production
     * code never falls back to implicit default-stream ordering. The tests use
     * this small RAII wrapper to make that contract visible and leak-free.
     */
    struct HipStream
    {
        hipStream_t stream = nullptr;

        HipStream()
        {
            checkHip(hipStreamCreate(&stream), "hipStreamCreate");
        }

        ~HipStream()
        {
            if (stream)
                (void)hipStreamDestroy(stream);
        }

        HipStream(const HipStream &) = delete;
        HipStream &operator=(const HipStream &) = delete;

        /// @brief Returns the stream as the opaque pointer expected by kernel interfaces.
        void *opaque() const { return stream; }

        /// @brief Blocks until all work enqueued on the stream has completed.
        void synchronize(const char *operation) const
        {
            checkHip(hipStreamSynchronize(stream), operation);
        }
    };

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

    /// @brief Builds a compact but production-valid Qwen3.5-style GDN/FA/GDN layer map.
    llaminar2::HybridKVCacheConfig makeHybridConfig()
    {
        llaminar2::HybridKVCacheConfig hybrid;
        hybrid.layer_types = {"gdn", "full_attention", "gdn"};
        hybrid.gdn_conv_kernel_size = 3;
        hybrid.gdn_state_size = 128;
        hybrid.gdn_inner_size = 128;
        hybrid.gdn_group_count = 1;
        hybrid.gdn_time_step_rank = 1;
        hybrid.n_heads = 1;
        hybrid.local_n_heads = 0;
        return hybrid;
    }

    /**
     * @brief Builds a TP-sharded one-layer GDN map with distinct local/full state sizes.
     *
     * The production partial-prefix path restores two GDN banks: a local bank for
     * suffix prefill and a full bank for decode. This config makes those banks
     * different sizes so a regression cannot accidentally pass by aliasing them.
     */
    llaminar2::HybridKVCacheConfig makeShardedHybridConfig()
    {
        llaminar2::HybridKVCacheConfig hybrid;
        hybrid.layer_types = {"gdn"};
        hybrid.gdn_conv_kernel_size = 3;
        hybrid.gdn_state_size = 2;
        hybrid.gdn_inner_size = 8;
        hybrid.gdn_group_count = 2;
        hybrid.gdn_time_step_rank = 4;
        hybrid.n_heads = 4;
        hybrid.local_n_heads = 2;
        return hybrid;
    }

    /// @brief Builds an offset FA map where global FA layer 3 compresses to parent KV slot 0.
    llaminar2::HybridKVCacheConfig makeOffsetFullAttentionHybridConfig()
    {
        llaminar2::HybridKVCacheConfig hybrid;
        hybrid.layer_types = {
            "gdn", "gdn", "gdn",
            "full_attention", "full_attention", "full_attention", "full_attention",
            "gdn"};
        hybrid.gdn_conv_kernel_size = 3;
        hybrid.gdn_state_size = 128;
        hybrid.gdn_inner_size = 128;
        hybrid.gdn_group_count = 1;
        hybrid.gdn_time_step_rank = 1;
        hybrid.n_heads = 1;
        hybrid.local_n_heads = 0;
        return hybrid;
    }

    /// @brief Creates a ROCm hybrid cache through KernelFactory so GDN kernels are initialized.
    HybridCacheHandle createHybridCache(const llaminar2::HybridKVCacheConfig &hybrid_config)
    {
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

    /**
     * @brief Creates a ROCm hybrid cache with explicit KV storage precision and shape.
     *
     * Prefix-cache payload tests need to exercise the same FP16 KV storage and
     * compressed full-attention map used by production Qwen3.6 partial-prefix
     * restore.  The smaller reset tests keep using createHybridCache(), while
     * this helper lets a regression choose a later global FA layer and a wider
     * row shape without changing those existing fixtures.
     */
    HybridCacheHandle createHybridCacheWithShape(
        const llaminar2::HybridKVCacheConfig &hybrid_config,
        llaminar2::ActivationPrecision precision,
        int max_seq_len,
        int n_kv_heads,
        int head_dim)
    {
        llaminar::v2::kernels::KVCacheConfig config;
        config.precision = precision;
        config.device = llaminar2::DeviceId::rocm(0);
        config.num_layers = static_cast<int>(hybrid_config.layer_types.size());
        config.batch_size = 1;
        config.max_seq_len = max_seq_len;
        config.n_kv_heads = n_kv_heads;
        config.head_dim = head_dim;
        config.hybrid_config = &hybrid_config;

        HybridCacheHandle handle;
        handle.owner = KernelFactory::createKVCache(config);
        handle.hybrid = dynamic_cast<llaminar2::IHybridKVCache *>(handle.owner.get());
        if (!handle.hybrid)
            throw std::runtime_error("KernelFactory did not create an IHybridKVCache");
        return handle;
    }

    /**
     * @brief Builds a production-shaped hybrid layer map with late FA layers.
     *
     * Qwen3.6 MoE stores only full-attention layers in the KV cache.  Layer 31
     * is late enough in that compressed slab to catch bugs that earlier
     * first-FA-layer smoke tests miss.
     */
    llaminar2::HybridKVCacheConfig makeLateFullAttentionHybridConfig()
    {
        llaminar2::HybridKVCacheConfig hybrid;
        hybrid.layer_types.assign(40, "gdn");
        const int fa_layers[] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
        for (int layer : fa_layers)
            hybrid.layer_types[static_cast<size_t>(layer)] = "full_attention";
        hybrid.gdn_conv_kernel_size = 3;
        hybrid.gdn_state_size = 128;
        hybrid.gdn_inner_size = 128;
        hybrid.gdn_group_count = 1;
        hybrid.gdn_time_step_rank = 1;
        hybrid.n_heads = 4;
        hybrid.local_n_heads = 0;
        return hybrid;
    }

    /// @brief Fills a byte buffer with a deterministic non-zero pattern.
    std::vector<uint8_t> bytePattern(size_t bytes, uint32_t seed)
    {
        std::vector<uint8_t> out(bytes);
        uint32_t state = seed == 0 ? 0x9e3779b9u : seed;
        for (uint8_t &value : out)
        {
            state = state * 1664525u + 1013904223u;
            value = static_cast<uint8_t>((state >> 24) ^ (state >> 11));
        }
        return out;
    }

    /// @brief Exports a logical KV block into byte vectors for exact comparison.
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> exportLogicalBytes(
        llaminar2::IKVCache *cache,
        int layer,
        int token_count,
        const HipStream &stream)
    {
        const auto layout = cache->logicalBlockLayout(layer, token_count);
        std::vector<uint8_t> k(layout.k_bytes);
        std::vector<uint8_t> v(layout.v_bytes);
        llaminar2::IKVCache::KVCacheLogicalBlockDescriptor desc;
        desc.layer = layer;
        desc.seq_idx = 0;
        desc.logical_token_start = 0;
        desc.token_count = token_count;
        desc.stream = stream.opaque();
        if (!cache->exportLogicalBlock(desc, k.data(), v.data()))
            throw std::runtime_error("failed to export logical KV block");
        return {std::move(k), std::move(v)};
    }

    /**
     * @brief Creates a host/device FP16 tensor with finite deterministic bit patterns.
     *
     * Captured-append tests compare raw cache bytes rather than decoded
     * floating-point values.  Keeping the payload finite makes the fixture
     * reusable for later tests that may run numeric kernels before append.
     */
    std::unique_ptr<llaminar2::FP16Tensor> createDeviceFP16PatternTensor(
        int rows,
        int cols,
        uint32_t seed,
        const HipStream &stream)
    {
        auto tensor = std::make_unique<llaminar2::FP16Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)});
        auto *data = static_cast<uint16_t *>(tensor->raw_mutable_data());
        if (!data)
            throw std::runtime_error("FP16 tensor host storage is unavailable");

        uint32_t state = seed == 0 ? 0x6d2b79f5u : seed;
        const size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        for (size_t i = 0; i < count; ++i)
        {
            state = state * 1664525u + 1013904223u;
            data[i] = static_cast<uint16_t>(0x3000u | ((state >> 16) & 0x0fffu));
        }
        llaminar2::TransferEngine::publishHostWrite(tensor);
        if (!tensor->ensureOnDevice(llaminar2::DeviceId::rocm(0), stream.opaque()))
            throw std::runtime_error("failed to upload FP16 test tensor");
        return tensor;
    }

    /// @brief Returns a byte-exact copy of the tensor's host-owned payload.
    std::vector<uint8_t> tensorHostBytes(const llaminar2::ITensor &tensor)
    {
        const auto *bytes = static_cast<const uint8_t *>(tensor.raw_data());
        if (!bytes)
            throw std::runtime_error("tensor host bytes are unavailable");
        return std::vector<uint8_t>(bytes, bytes + tensor.size_bytes());
    }

    /// @brief Creates the default tiny ROCm hybrid cache used by reset tests.
    HybridCacheHandle createHybridCache()
    {
        return createHybridCache(makeHybridConfig());
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
            ASSERT_TRUE(std::isfinite(actual[i])) << label << " actual value is not finite at index " << i;
            ASSERT_TRUE(std::isfinite(expected[i])) << label << " expected value is not finite at index " << i;
            ASSERT_NEAR(actual[i], expected[i], tol) << label << " at index " << i;
        }
    }

    /// @brief Generates deterministic FP32 state snapshots with clearly separated ranges.
    std::vector<float> statePattern(size_t count, float base)
    {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i)
            values[i] = base + static_cast<float>(i) * 0.125f;
        return values;
    }

    /**
     * @brief Encode deterministic FP32 rows in one native ROCm cache format.
     *
     * The grouped converted-read regression must cover the actual storage
     * representation, including complete Q8_1 blocks. Native tensors also keep
     * append independent of format-specific conversion scratch so the test
     * isolates hybrid layer remapping and resident gather behavior.
     */
    std::unique_ptr<llaminar2::ITensor> makeNativeCacheTensor(
        llaminar2::ActivationPrecision precision,
        const std::vector<float> &values,
        size_t rows,
        size_t cols)
    {
        const std::vector<size_t> shape = {rows, cols};
        switch (precision)
        {
        case llaminar2::ActivationPrecision::FP32:
        {
            auto tensor = std::make_unique<llaminar2::FP32Tensor>(
                shape,
                llaminar2::DeviceId::cpu());
            std::copy(values.begin(), values.end(), tensor->mutable_data());
            return tensor;
        }
        case llaminar2::ActivationPrecision::FP16:
        {
            std::vector<uint16_t> encoded(values.size());
            std::transform(
                values.begin(), values.end(), encoded.begin(),
                [](float value)
                {
                    return llaminar2::simd::fp32_to_fp16(value);
                });
            return std::make_unique<llaminar2::FP16Tensor>(shape, encoded);
        }
        case llaminar2::ActivationPrecision::BF16:
        {
            std::vector<uint16_t> encoded(values.size());
            std::transform(
                values.begin(), values.end(), encoded.begin(),
                [](float value)
                {
                    return llaminar2::simd::fp32_to_bf16(value);
                });
            return std::make_unique<llaminar2::BF16Tensor>(shape, encoded);
        }
        case llaminar2::ActivationPrecision::Q8_1:
        {
            auto quantized = llaminar2::Q8_1Tensor::quantize_from_fp32(
                values.data(),
                shape);
            if (!quantized)
                throw std::runtime_error("Q8_1 test tensor quantization failed");
            return std::make_unique<llaminar2::Q8_1Tensor>(*quantized);
        }
        default:
            throw std::runtime_error("unsupported native ROCm hybrid cache test format");
        }
    }

    /// @brief Host snapshot of the local live GDN bank exported through GPU kernels.
    struct GDNStateSnapshot
    {
        std::vector<float> recurrence;
        std::vector<float> conv;
    };

    /**
     * @brief Exports the local live GDN bank for a layer into host vectors.
     *
     * The host-staged prefix payload stores live GPU state in host memory. This
     * helper gives tests a precise expected value for that explicit archive
     * boundary without introducing a second live-state representation.
     */
    GDNStateSnapshot exportLocalGDNState(llaminar2::IHybridKVCache *cache, int layer, const HipStream &stream)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->rec_kernel || !state->conv_kernel)
            throw std::runtime_error("invalid GDN state while exporting local ROCm test snapshot");

        GDNStateSnapshot snapshot;
        snapshot.recurrence.resize(
            static_cast<size_t>(state->local_recurrence_state_size));
        snapshot.conv.resize(static_cast<size_t>(state->local_conv_state_size));
        if (!state->rec_kernel->exportStateForSize(
                static_cast<int>(snapshot.recurrence.size()),
                snapshot.recurrence.data(),
                nullptr,
                stream.opaque()))
        {
            throw std::runtime_error("failed to export local ROCm recurrence state snapshot");
        }
        if (!state->conv_kernel->exportStateForSize(
                static_cast<int>(snapshot.conv.size()),
                snapshot.conv.data(),
                nullptr,
                stream.opaque()))
        {
            throw std::runtime_error("failed to export local ROCm conv state snapshot");
        }
        stream.synchronize("hipStreamSynchronize after local GDN state snapshot export");
        return snapshot;
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
        if (!state || !state->conv_kernel || state->conv_kernel_size <= 1 ||
            state->local_conv_state_size <= 0)
            throw std::runtime_error("invalid GDN convolution state in ROCm hybrid KV cache test");

        const int channels = state->local_conv_state_size /
                             (state->conv_kernel_size - 1);
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

        HipStream stream;
        state->conv_kernel->setGPUStream(stream.opaque());
        EXPECT_TRUE(state->conv_kernel->forward(
            d_input.ptr, d_weight.ptr, nullptr,
            d_output.ptr, nullptr,
            /*seq_len=*/1, channels, kernel_size,
            /*apply_silu=*/false));
        stream.synchronize("hipStreamSynchronize after short-conv");
        return d_output.toHost();
    }

    /// @brief Runs one in-place short-conv decode step and returns output plus exported GPU state.
    std::pair<std::vector<float>, std::vector<float>> runInPlaceConvDecodeAndExportState(
        llaminar2::IHybridKVCache *cache, int layer, const std::vector<float> &input)
    {
        auto *state = cache->getGDNState(layer);
        if (!state || !state->conv_kernel || state->conv_kernel_size <= 1 ||
            state->local_conv_state_size <= 0)
            throw std::runtime_error("invalid GDN convolution state in ROCm hybrid KV cache test");

        const int kernel_size = state->conv_kernel_size;
        const int state_len = kernel_size - 1;
        const int channels = static_cast<int>(input.size());
        if (state->local_conv_state_size != channels * state_len)
            throw std::runtime_error("in-place short-conv test input does not match cache state shape");

        std::vector<float> weight(static_cast<size_t>(channels) * static_cast<size_t>(kernel_size), 0.0f);
        for (int c = 0; c < channels; ++c)
            weight[static_cast<size_t>(c) * kernel_size + state_len] = 2.0f;

        HipFloatBuffer d_input_output(input);
        HipFloatBuffer d_weight(weight);
        HipFloatBuffer d_scratch(static_cast<size_t>(channels));

        HipStream stream;
        state->conv_kernel->setGPUStream(stream.opaque());
        state->conv_kernel->bindScratchWorkspace(d_scratch.ptr, channels);
        if (!state->conv_kernel->forward(
                d_input_output.ptr, d_weight.ptr, nullptr,
                d_input_output.ptr, nullptr,
                /*seq_len=*/1, channels, kernel_size,
                /*apply_silu=*/false))
            throw std::runtime_error("in-place short-conv decode failed");
        stream.synchronize("hipStreamSynchronize after in-place short-conv");

        std::vector<float> exported_state(
            static_cast<size_t>(state->local_conv_state_size));
        if (!state->conv_kernel->exportState(exported_state.data(), nullptr, stream.opaque()))
            throw std::runtime_error("failed to export in-place short-conv GPU state");
        stream.synchronize("hipStreamSynchronize after in-place short-conv state export");
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
        std::vector<float> alpha(static_cast<size_t>(n_heads), 0.1f);
        std::vector<float> beta(static_cast<size_t>(n_heads), -1.0f);
        std::vector<float> a_log(static_cast<size_t>(n_heads), -0.25f);
        std::vector<float> dt_bias(static_cast<size_t>(n_heads), -0.5f);

        HipFloatBuffer d_q(q), d_k_buf(k), d_v_buf(v), d_alpha(alpha), d_beta(beta), d_a_log(a_log), d_dt_bias(dt_bias);
        HipFloatBuffer d_output(v_count);

        HipStream stream;
        state->rec_kernel->setGPUStream(stream.opaque());
        EXPECT_TRUE(state->rec_kernel->recurrent_step(
            d_q.ptr, d_k_buf.ptr, d_v_buf.ptr,
            d_alpha.ptr, d_beta.ptr,
            d_a_log.ptr, d_dt_bias.ptr,
            d_output.ptr, nullptr,
            n_heads, d_k, d_v,
            /*use_qk_l2norm=*/true));
        stream.synchronize("hipStreamSynchronize after GDN recurrence");
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
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
    }

    /// @brief Copies one device-owned sequence metadata integer back for assertions.
    int copyDeviceInt(const int *device_value, const char *label)
    {
        if (!device_value)
            throw std::runtime_error(std::string(label) + " returned a null device pointer");

        int host_value = -1;
        if (hipMemcpy(&host_value, device_value, sizeof(int), hipMemcpyDeviceToHost) != hipSuccess)
            throw std::runtime_error(std::string(label) + " failed");
        return host_value;
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

    HipStream reset_stream;
    ASSERT_TRUE(cache.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));

    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 0);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr)
        << "clear() must not recreate cache-owned GDN conv kernels";
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr)
        << "clear() must not recreate cache-owned GDN recurrence kernels";

    auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 3.0f);
    auto fresh_conv = runConvDecode(fresh.hybrid, /*layer=*/0, 3.0f);
    expectNearVector(actual_conv, fresh_conv, 1e-5f, "conv output after request reset vs fresh");

    auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 3.5f);
    auto fresh_rec = runRecurrenceDecode(fresh.hybrid, /*layer=*/0, 3.5f);
    expectNearVector(actual_rec, fresh_rec, 1e-4f, "recurrence output after request reset vs fresh");
}

TEST(Test__ROCmHybridKVCacheReset, ClearLayerResetsGDNGPUKernelState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto fresh = createHybridCache();

    mutateGDNState(cache.hybrid, /*layer=*/0);
    HipStream reset_stream;
    ASSERT_TRUE(cache.owner->resetLayerState(
        /*layer=*/0,
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));

    auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 4.0f);
    auto fresh_conv = runConvDecode(fresh.hybrid, /*layer=*/0, 4.0f);
    expectNearVector(actual_conv, fresh_conv, 1e-5f, "conv output after layer reset vs fresh");

    auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 4.5f);
    auto fresh_rec = runRecurrenceDecode(fresh.hybrid, /*layer=*/0, 4.5f);
    expectNearVector(actual_rec, fresh_rec, 1e-4f, "recurrence output after layer reset vs fresh");
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
    ASSERT_EQ(state->local_conv_state_size % state_len, 0);
    const int channels = state->local_conv_state_size / state_len;
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

    ASSERT_TRUE(cache.owner->resetLayerState(
        /*layer=*/0,
        llaminar2::IKVCache::StateResetContext::testReinitialization(stream)));
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

TEST(Test__ROCmHybridKVCacheReset, HybridPrefixStateRoundTripRestoresDeviceStateAndPreservesKernels)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state0 = cache.hybrid->getGDNState(0);
    auto *state2 = cache.hybrid->getGDNState(2);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state2, nullptr);
    EXPECT_TRUE(state0->recurrence_state.empty());
    EXPECT_TRUE(state0->conv_state.empty());
    EXPECT_TRUE(state2->recurrence_state.empty());
    EXPECT_TRUE(state2->conv_state.empty());
    EXPECT_EQ(cache.hybrid->getRecurrenceState(0), nullptr);
    EXPECT_EQ(cache.hybrid->getConvState(0), nullptr);

    auto *conv_ptr = state0->conv_kernel.get();
    auto *rec_ptr = state0->rec_kernel.get();
    ASSERT_NE(conv_ptr, nullptr);
    ASSERT_NE(rec_ptr, nullptr);
    mutateGDNState(cache.hybrid, /*layer=*/0);
    mutateGDNState(cache.hybrid, /*layer=*/2);

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.total_layers, 3);
    EXPECT_EQ(metadata.gdn_layers, 2);
    EXPECT_EQ(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);
    EXPECT_TRUE(metadata.has_device_kernel_state);

    HipStream stream;
    std::vector<uint8_t> payload(metadata.device_bytes);
    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    desc.stream = stream.opaque();
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(desc, payload.data(), nullptr));
    const auto expected_state0 = exportLocalGDNState(cache.hybrid, /*layer=*/0, stream);
    const auto expected_state2 = exportLocalGDNState(cache.hybrid, /*layer=*/2, stream);

    const auto expected_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto expected_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);

    ASSERT_TRUE(cache.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            stream.opaque())));
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(desc, payload.data(), nullptr));
    EXPECT_EQ(cache.hybrid->getGDNState(0)->conv_kernel.get(), conv_ptr);
    EXPECT_EQ(cache.hybrid->getGDNState(0)->rec_kernel.get(), rec_ptr);
    const auto actual_state0 = exportLocalGDNState(cache.hybrid, /*layer=*/0, stream);
    const auto actual_state2 = exportLocalGDNState(cache.hybrid, /*layer=*/2, stream);
    expectNearVector(actual_state0.recurrence, expected_state0.recurrence, 0.0f,
                     "device recurrence bank restore for layer 0");
    expectNearVector(actual_state0.conv, expected_state0.conv, 0.0f,
                     "device conv bank restore for layer 0");
    expectNearVector(actual_state2.recurrence, expected_state2.recurrence, 0.0f,
                     "device recurrence bank restore for layer 2");
    expectNearVector(actual_state2.conv, expected_state2.conv, 0.0f,
                     "device conv bank restore for layer 2");

    const auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);
    expectNearVector(actual_conv, expected_conv, 1e-5f, "conv output after hybrid prefix import");
    expectNearVector(actual_rec, expected_rec, 1e-4f, "recurrence output after hybrid prefix import");
}

TEST(Test__ROCmHybridKVCacheReset, HostStagedHybridPrefixStateSerializesOnlyDeviceBanksContiguously)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    mutateGDNState(cache.hybrid, /*layer=*/0);
    mutateGDNState(cache.hybrid, /*layer=*/2);

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);

    HipStream stream;
    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    desc.stream = stream.opaque();
    constexpr size_t kGuardBytes = 256;
    std::vector<uint8_t> staged_payload(metadata.device_bytes + kGuardBytes, 0xCD);
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(
        desc,
        staged_payload.data(),
        nullptr));
    EXPECT_FALSE(std::all_of(
        staged_payload.begin(),
        staged_payload.begin() + static_cast<std::ptrdiff_t>(metadata.device_bytes),
        [](uint8_t value) { return value == 0xCD; }));
    EXPECT_TRUE(std::all_of(
        staged_payload.begin() + static_cast<std::ptrdiff_t>(metadata.device_bytes),
        staged_payload.end(),
        [](uint8_t value) { return value == 0xCD; }));

    ASSERT_TRUE(cache.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            stream.opaque())));
    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(
        desc,
        staged_payload.data(),
        nullptr));

    std::vector<uint8_t> roundtrip_payload(metadata.device_bytes, 0);
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(
        desc,
        roundtrip_payload.data(),
        nullptr));
    EXPECT_TRUE(std::equal(
        roundtrip_payload.begin(),
        roundtrip_payload.end(),
        staged_payload.begin()));
}

TEST(Test__ROCmHybridKVCacheReset, AsyncDeviceOnlyHybridPrefixStateRoundTripRestoresAfterExplicitStreamSync)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state0 = cache.hybrid->getGDNState(0);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state0->conv_kernel, nullptr);
    ASSERT_NE(state0->rec_kernel, nullptr);

    mutateGDNState(cache.hybrid, /*layer=*/0);

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);
    ASSERT_EQ(metadata.device_bytes % sizeof(float), 0u);

    HipFloatBuffer device_payload(metadata.device_bytes / sizeof(float));

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    desc.stream = stream;
    desc.synchronize = false;
    desc.include_host_state = false;
    desc.include_device_state = true;
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(
        desc,
        nullptr,
        device_payload.ptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

    const auto expected_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto expected_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);

    ASSERT_TRUE(cache.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(stream)));
    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(
        desc,
        nullptr,
        device_payload.ptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);

    const auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);
    expectNearVector(actual_conv, expected_conv, 1e-5f, "async conv output after hybrid prefix import");
    expectNearVector(actual_rec, expected_rec, 1e-4f, "async recurrence output after hybrid prefix import");
}

TEST(Test__ROCmHybridKVCacheReset, HostStagedDeviceOnlyHybridPrefixStateRoundTripStartsAtOffsetZero)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    auto *state0 = cache.hybrid->getGDNState(0);
    ASSERT_NE(state0, nullptr);
    ASSERT_NE(state0->conv_kernel, nullptr);
    ASSERT_NE(state0->rec_kernel, nullptr);

    mutateGDNState(cache.hybrid, /*layer=*/0);

    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);

    llaminar2::HybridPrefixStateDescriptor desc;
    desc.seq_idx = 0;
    desc.logical_token_count = 4;
    HipStream stream;
    desc.stream = stream.opaque();
    desc.include_host_state = false;
    desc.include_device_state = true;

    constexpr size_t kGuardBytes = 256;
    std::vector<uint8_t> payload(metadata.device_bytes + kGuardBytes, 0xCD);
    ASSERT_TRUE(cache.hybrid->exportHybridPrefixState(
        desc,
        payload.data(),
        nullptr));
    EXPECT_FALSE(std::all_of(
        payload.begin(),
        payload.begin() + static_cast<std::ptrdiff_t>(metadata.device_bytes),
        [](uint8_t value) { return value == 0xCD; }));
    EXPECT_TRUE(std::all_of(
        payload.begin() + static_cast<std::ptrdiff_t>(metadata.device_bytes),
        payload.end(),
        [](uint8_t value) { return value == 0xCD; }));

    const auto expected_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto expected_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);

    ASSERT_TRUE(cache.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            stream.opaque())));
    ASSERT_TRUE(cache.hybrid->importHybridPrefixState(
        desc,
        payload.data(),
        nullptr));

    const auto actual_conv = runConvDecode(cache.hybrid, /*layer=*/0, 6.0f);
    const auto actual_rec = runRecurrenceDecode(cache.hybrid, /*layer=*/0, 6.5f);
    expectNearVector(actual_conv, expected_conv, 1e-5f, "host-staged device-only conv import");
    expectNearVector(actual_rec, expected_rec, 1e-4f, "host-staged device-only recurrence import");
}

TEST(Test__ROCmHybridKVCacheReset, HostStagedSuffixPrefillImportKeepsLocalAndFullGDNBanksDistinct)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto source = createHybridCache(makeShardedHybridConfig());
    auto *source_state = source.hybrid->getGDNState(0);
    ASSERT_NE(source_state, nullptr);
    ASSERT_NE(source_state->conv_kernel, nullptr);
    ASSERT_NE(source_state->rec_kernel, nullptr);

    const int local_rec_floats = source_state->local_recurrence_state_size;
    const int local_conv_floats = source_state->local_conv_state_size;
    const int full_rec_floats = source_state->full_recurrence_state_size;
    const int full_conv_floats = source_state->full_conv_state_size;
    ASSERT_GT(local_rec_floats, 0);
    ASSERT_GT(local_conv_floats, 0);
    ASSERT_GT(full_rec_floats, local_rec_floats);
    ASSERT_GT(full_conv_floats, local_conv_floats);

    const auto expected_local_rec = statePattern(static_cast<size_t>(local_rec_floats), 100.0f);
    const auto expected_local_conv = statePattern(static_cast<size_t>(local_conv_floats), 200.0f);
    const auto expected_full_rec = statePattern(static_cast<size_t>(full_rec_floats), 300.0f);
    const auto expected_full_conv = statePattern(static_cast<size_t>(full_conv_floats), 400.0f);

    HipStream stream;
    ASSERT_TRUE(source_state->rec_kernel->importStateForSize(
        local_rec_floats, expected_local_rec.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(source_state->conv_kernel->importStateForSize(
        local_conv_floats, expected_local_conv.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(source_state->rec_kernel->importStateForSize(
        full_rec_floats, expected_full_rec.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(source_state->conv_kernel->importStateForSize(
        full_conv_floats, expected_full_conv.data(), nullptr, stream.opaque()));
    stream.synchronize("hipStreamSynchronize after source state seeding");

    const auto metadata = source.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    EXPECT_EQ(metadata.device_bytes,
              static_cast<size_t>(local_conv_floats + full_conv_floats +
                                  local_rec_floats + full_rec_floats) *
                  sizeof(float));
    ASSERT_GT(metadata.device_bytes, 0u);

    llaminar2::HybridPrefixStateDescriptor export_desc;
    export_desc.seq_idx = 0;
    export_desc.logical_token_count = 4;
    export_desc.stream = stream.opaque();
    export_desc.include_host_state = false;
    export_desc.include_device_state = true;

    std::vector<uint8_t> staged_payload(metadata.device_bytes, 0xCD);
    ASSERT_TRUE(source.hybrid->exportHybridPrefixState(
        export_desc,
        staged_payload.data(),
        nullptr));

    auto target = createHybridCache(makeShardedHybridConfig());
    auto *target_state = target.hybrid->getGDNState(0);
    ASSERT_NE(target_state, nullptr);
    ASSERT_NE(target_state->conv_kernel, nullptr);
    ASSERT_NE(target_state->rec_kernel, nullptr);

    llaminar2::HybridPrefixStateDescriptor import_desc;
    import_desc.seq_idx = 0;
    import_desc.logical_token_count = 4;
    import_desc.stream = stream.opaque();
    import_desc.include_host_state = false;
    import_desc.include_device_state = true;
    ASSERT_TRUE(target.hybrid->importHybridPrefixState(
        import_desc,
        staged_payload.data(),
        nullptr));

    EXPECT_GT(target_state->rec_kernel->stateBytes(), 0u);
    EXPECT_GT(target_state->conv_kernel->stateBytes(), 0u);
    EXPECT_TRUE(target_state->rec_kernel->isGPUStateReady(local_rec_floats));
    EXPECT_TRUE(target_state->rec_kernel->isGPUStateReady(full_rec_floats));
    EXPECT_TRUE(target_state->recurrence_state.empty());
    EXPECT_TRUE(target_state->conv_state.empty());

    std::vector<float> actual_local_rec(static_cast<size_t>(local_rec_floats));
    std::vector<float> actual_local_conv(static_cast<size_t>(local_conv_floats));
    std::vector<float> actual_full_rec(static_cast<size_t>(full_rec_floats));
    std::vector<float> actual_full_conv(static_cast<size_t>(full_conv_floats));
    ASSERT_TRUE(target_state->rec_kernel->exportStateForSize(
        local_rec_floats, actual_local_rec.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(target_state->conv_kernel->exportStateForSize(
        local_conv_floats, actual_local_conv.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(target_state->rec_kernel->exportStateForSize(
        full_rec_floats, actual_full_rec.data(), nullptr, stream.opaque()));
    ASSERT_TRUE(target_state->conv_kernel->exportStateForSize(
        full_conv_floats, actual_full_conv.data(), nullptr, stream.opaque()));
    stream.synchronize("hipStreamSynchronize after target state export");

    expectNearVector(actual_local_rec, expected_local_rec, 0.0f, "local recurrence bank after suffix import");
    expectNearVector(actual_local_conv, expected_local_conv, 0.0f, "local conv bank after suffix import");
    expectNearVector(actual_full_rec, expected_full_rec, 0.0f, "full recurrence bank after suffix import");
    expectNearVector(actual_full_conv, expected_full_conv, 0.0f, "full conv bank after suffix import");
}

TEST(Test__ROCmHybridKVCacheReset, GPUHybridCacheExposesNoHostLiveState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();
    const auto metadata = cache.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    EXPECT_GT(metadata.device_bytes, 0u);

    for (int layer : {0, 2})
    {
        const auto *state = cache.hybrid->getGDNState(layer);
        ASSERT_NE(state, nullptr);
        EXPECT_TRUE(state->recurrence_state.empty());
        EXPECT_TRUE(state->conv_state.empty());
        EXPECT_GT(state->local_recurrence_state_size, 0);
        EXPECT_GT(state->local_conv_state_size, 0);
        EXPECT_EQ(cache.hybrid->getRecurrenceState(layer), nullptr);
        EXPECT_EQ(cache.hybrid->getConvState(layer), nullptr);
    }
}

TEST(Test__ROCmHybridKVCacheReset, ClearLayerResetsCompressedFullAttentionEntry)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache();

    appendFullAttentionToken(cache.owner.get(), /*layer=*/1);
    ASSERT_EQ(cache.owner->get_cached_tokens(1, 0), 2);

    HipStream reset_stream;
    ASSERT_TRUE(cache.owner->resetLayerState(
        /*layer=*/1,
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            reset_stream.opaque())));

    EXPECT_EQ(cache.owner->get_cached_tokens(1, 0), 0)
        << "resetLayerState(global FA layer) must reset the compressed parent-cache entry";
}

TEST(Test__ROCmHybridKVCacheReset, DeviceSequenceMetadataPointersUseCompressedFullAttentionSlot)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache(makeOffsetFullAttentionHybridConfig());

    ASSERT_EQ(cache.owner->n_layers(), 8);
    ASSERT_EQ(cache.hybrid->kvLayerCount(), 4);
    ASSERT_EQ(cache.owner->deviceCachedTokenCountPtr(/*layer=*/0, /*seq_idx=*/0), nullptr)
        << "GDN layers do not own KV sequence metadata";

    appendFullAttentionToken(cache.owner.get(), /*layer=*/3);

    EXPECT_EQ(cache.owner->get_cached_tokens(/*layer=*/3, /*seq_idx=*/0), 2);
    EXPECT_EQ(copyDeviceInt(cache.owner->deviceCachedTokenCountPtr(/*layer=*/3, /*seq_idx=*/0),
                            "hipMemcpy cached-token count"),
              2)
        << "Global FA layer 3 must read compressed parent KV slot 0, not parent slot 3.";
    EXPECT_EQ(copyDeviceInt(cache.owner->deviceRingHeadPtr(/*layer=*/3, /*seq_idx=*/0),
                            "hipMemcpy ring head"),
              2)
        << "Device-owned ring-head metadata must be remapped with the KV payload.";

    EXPECT_EQ(cache.owner->get_cached_tokens(/*layer=*/6, /*seq_idx=*/0), 0);
    EXPECT_EQ(copyDeviceInt(cache.owner->deviceCachedTokenCountPtr(/*layer=*/6, /*seq_idx=*/0),
                            "hipMemcpy unrelated cached-token count"),
              0)
        << "The old un-remapped path would read this parent slot for global layer 3.";
}

TEST(Test__ROCmHybridKVCacheReset, GetKVUsesCompressedFullAttentionSlot)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    auto cache = createHybridCache(makeOffsetFullAttentionHybridConfig());
    appendFullAttentionToken(cache.owner.get(), /*layer=*/3);

    llaminar2::ITensor *k = nullptr;
    llaminar2::ITensor *v = nullptr;
    int kv_len = -1;
    ASSERT_TRUE(cache.owner->get_kv(/*layer=*/3, /*seq_idx=*/0, &k, &v, &kv_len));
    EXPECT_NE(k, nullptr);
    EXPECT_NE(v, nullptr);
    EXPECT_EQ(kv_len, 2)
        << "get_kv(global FA layer) must remap through the hybrid layer map.";

    const llaminar2::ITensor *ck = nullptr;
    const llaminar2::ITensor *cv = nullptr;
    int const_kv_len = -1;
    const llaminar2::IKVCache *const_cache = cache.owner.get();
    ASSERT_TRUE(const_cache->get_kv(/*layer=*/3, /*seq_idx=*/0, &ck, &cv, &const_kv_len));
    EXPECT_NE(ck, nullptr);
    EXPECT_NE(cv, nullptr);
    EXPECT_EQ(const_kv_len, 2);
}

/**
 * @brief Every converted resident format honors the hybrid FA layer map.
 *
 * Global layer 3 compresses to parent KV slot 0, while global layer 6
 * compresses to slot 3. The inherited converted-read implementation formerly
 * accepted global id 3 as parent slot 3 and silently returned layer 6. A plain
 * one-layer ring supplies the same grouped resident kernel as an unambiguous
 * slot-zero oracle for every supported ROCm cache format.
 */
TEST(Test__ROCmHybridKVCacheReset, ConvertedResidentReadAllFormatsUsesCompressedFullAttentionSlot)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int global_first_fa_layer = 3;
    constexpr int global_fourth_fa_layer = 6;
    constexpr int token_count = 2;
    constexpr int max_seq_len = 8;
    constexpr int n_kv_heads = 1;
    constexpr int head_dim = 32;
    constexpr size_t live_elements =
        static_cast<size_t>(token_count) * n_kv_heads * head_dim;

    const std::array<llaminar2::ActivationPrecision, 4> formats = {
        llaminar2::ActivationPrecision::FP32,
        llaminar2::ActivationPrecision::FP16,
        llaminar2::ActivationPrecision::BF16,
        llaminar2::ActivationPrecision::Q8_1,
    };

    for (const auto precision : formats)
    {
        SCOPED_TRACE(llaminar2::activationPrecisionToString(precision));
        const auto hybrid_config = makeOffsetFullAttentionHybridConfig();
        auto cache = createHybridCacheWithShape(
            hybrid_config,
            precision,
            max_seq_len,
            n_kv_heads,
            head_dim);

        llaminar::v2::kernels::KVCacheConfig oracle_config;
        oracle_config.precision = precision;
        oracle_config.device = llaminar2::DeviceId::rocm(0);
        oracle_config.num_layers = 1;
        oracle_config.batch_size = 1;
        oracle_config.max_seq_len = max_seq_len;
        oracle_config.n_kv_heads = n_kv_heads;
        oracle_config.head_dim = head_dim;
        auto oracle_cache = KernelFactory::createKVCache(oracle_config);
        ASSERT_NE(oracle_cache, nullptr);

        auto *workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(cache.owner.get());
        auto *oracle_workspace_consumer =
            dynamic_cast<llaminar2::IWorkspaceConsumer *>(oracle_cache.get());
        ASSERT_NE(workspace_consumer, nullptr);
        ASSERT_NE(oracle_workspace_consumer, nullptr);

        const auto requirements = workspace_consumer->getWorkspaceRequirements(
            token_count,
            /*n=*/1,
            head_dim);
        llaminar2::DeviceWorkspaceManager workspace(
            llaminar2::DeviceId::rocm(0),
            requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(workspace.allocate(requirements));
        workspace_consumer->bindWorkspace(&workspace);

        const auto oracle_requirements =
            oracle_workspace_consumer->getWorkspaceRequirements(
                token_count,
                /*n=*/1,
                head_dim);
        llaminar2::DeviceWorkspaceManager oracle_workspace(
            llaminar2::DeviceId::rocm(0),
            oracle_requirements.total_bytes_with_alignment() + 4096);
        ASSERT_TRUE(oracle_workspace.allocate(oracle_requirements));
        oracle_workspace_consumer->bindWorkspace(&oracle_workspace);

        const std::vector<float> first_k = statePattern(live_elements, 0.25f);
        const std::vector<float> first_v = statePattern(live_elements, -0.75f);
        const std::vector<float> fourth_k = statePattern(live_elements, 8.25f);
        const std::vector<float> fourth_v = statePattern(live_elements, -9.75f);
        auto first_k_tensor = makeNativeCacheTensor(
            precision, first_k, token_count, n_kv_heads * head_dim);
        auto first_v_tensor = makeNativeCacheTensor(
            precision, first_v, token_count, n_kv_heads * head_dim);
        auto fourth_k_tensor = makeNativeCacheTensor(
            precision, fourth_k, token_count, n_kv_heads * head_dim);
        auto fourth_v_tensor = makeNativeCacheTensor(
            precision, fourth_v, token_count, n_kv_heads * head_dim);
        HipStream stream;

        ASSERT_TRUE(cache.owner->appendWithStream(
            global_first_fa_layer,
            /*seq_idx=*/0,
            first_k_tensor.get(),
            first_v_tensor.get(),
            token_count,
            stream.opaque()));
        ASSERT_TRUE(cache.owner->appendWithStream(
            global_fourth_fa_layer,
            /*seq_idx=*/0,
            fourth_k_tensor.get(),
            fourth_v_tensor.get(),
            token_count,
            stream.opaque()));
        ASSERT_TRUE(oracle_cache->appendWithStream(
            /*layer=*/0,
            /*seq_idx=*/0,
            first_k_tensor.get(),
            first_v_tensor.get(),
            token_count,
            stream.opaque()));
        stream.synchronize("hipStreamSynchronize after hybrid append");

        llaminar2::IKVCache::KVReadParams read;
        read.n_kv_heads = n_kv_heads;
        read.head_dim = head_dim;
        read.rope_dim = head_dim;
        read.gpu_stream = stream.opaque();

        llaminar2::ITensor *resident_k = nullptr;
        llaminar2::ITensor *resident_v = nullptr;
        ASSERT_TRUE(cache.owner->get_kv_batched_converted_device_view(
            global_first_fa_layer,
            /*first_seq_idx=*/0,
            /*request_count=*/1,
            llaminar2::ActivationPrecision::FP16,
            &resident_k,
            &resident_v,
            read));
        ASSERT_NE(resident_k, nullptr);
        ASSERT_NE(resident_v, nullptr);

        llaminar2::ITensor *oracle_k = nullptr;
        llaminar2::ITensor *oracle_v = nullptr;
        ASSERT_TRUE(oracle_cache->get_kv_batched_converted_device_view(
            /*layer=*/0,
            /*first_seq_idx=*/0,
            /*request_count=*/1,
            llaminar2::ActivationPrecision::FP16,
            &oracle_k,
            &oracle_v,
            read));
        ASSERT_NE(oracle_k, nullptr);
        ASSERT_NE(oracle_v, nullptr);

        std::vector<uint16_t> actual_k(live_elements);
        std::vector<uint16_t> actual_v(live_elements);
        std::vector<uint16_t> expected_k(live_elements);
        std::vector<uint16_t> expected_v(live_elements);
        checkHip(
            hipMemcpyAsync(
                actual_k.data(), resident_k->gpu_data_ptr(),
                actual_k.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost, stream.stream),
            "hipMemcpyAsync resident K");
        checkHip(
            hipMemcpyAsync(
                actual_v.data(), resident_v->gpu_data_ptr(),
                actual_v.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost, stream.stream),
            "hipMemcpyAsync resident V");
        checkHip(
            hipMemcpyAsync(
                expected_k.data(), oracle_k->gpu_data_ptr(),
                expected_k.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost, stream.stream),
            "hipMemcpyAsync oracle K");
        checkHip(
            hipMemcpyAsync(
                expected_v.data(), oracle_v->gpu_data_ptr(),
                expected_v.size() * sizeof(uint16_t),
                hipMemcpyDeviceToHost, stream.stream),
            "hipMemcpyAsync oracle V");
        stream.synchronize("hipStreamSynchronize after resident read");

        EXPECT_EQ(actual_k, expected_k)
            << "resident K read selected the wrong compressed FA slot";
        EXPECT_EQ(actual_v, expected_v)
            << "resident V read selected the wrong compressed FA slot";
        workspace_consumer->unbindWorkspace();
        oracle_workspace_consumer->unbindWorkspace();
    }
}

TEST(Test__ROCmHybridKVCacheReset, PrefixPayloadSlabRoundTripPreservesLateFullAttentionKRows)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const auto hybrid_config = makeLateFullAttentionHybridConfig();
    auto source = createHybridCacheWithShape(
        hybrid_config,
        llaminar2::ActivationPrecision::FP16,
        /*max_seq_len=*/16,
        /*n_kv_heads=*/4,
        /*head_dim=*/8);
    auto target = createHybridCacheWithShape(
        hybrid_config,
        llaminar2::ActivationPrecision::FP16,
        /*max_seq_len=*/16,
        /*n_kv_heads=*/4,
        /*head_dim=*/8);

    constexpr int kTokenCount = 6;
    const std::vector<int> fa_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
    const auto first_layout = source.owner->logicalBlockLayout(fa_layers.front(), kTokenCount);
    ASSERT_GT(first_layout.k_bytes, 0u);
    ASSERT_EQ(first_layout.k_bytes, first_layout.v_bytes);

    HipStream stream;
    std::vector<uint8_t> prefix_k_slab(
        static_cast<size_t>(fa_layers.size()) * first_layout.k_bytes);
    std::vector<uint8_t> prefix_v_slab(
        static_cast<size_t>(fa_layers.size()) * first_layout.v_bytes);

    for (size_t i = 0; i < fa_layers.size(); ++i)
    {
        const int layer = fa_layers[i];
        const auto layout = source.owner->logicalBlockLayout(layer, kTokenCount);
        ASSERT_EQ(layout.k_bytes, first_layout.k_bytes);
        ASSERT_EQ(layout.v_bytes, first_layout.v_bytes);

        const auto expected_k = bytePattern(layout.k_bytes, 0x31000000u + static_cast<uint32_t>(layer));
        const auto expected_v = bytePattern(layout.v_bytes, 0x71000000u + static_cast<uint32_t>(layer));

        llaminar2::IKVCache::KVCacheLogicalBlockDescriptor desc;
        desc.layer = layer;
        desc.seq_idx = 0;
        desc.logical_token_start = 0;
        desc.token_count = kTokenCount;
        desc.stream = stream.opaque();
        ASSERT_TRUE(source.owner->importLogicalBlock(
            desc,
            expected_k.data(),
            expected_v.data()))
            << "source import failed for global layer " << layer;

        auto [source_k, source_v] = exportLogicalBytes(
            source.owner.get(),
            layer,
            kTokenCount,
            stream);
        ASSERT_EQ(source_k, expected_k)
            << "source export changed K payload for global layer " << layer;
        ASSERT_EQ(source_v, expected_v)
            << "source export changed V payload for global layer " << layer;

        std::memcpy(
            prefix_k_slab.data() + i * first_layout.k_bytes,
            source_k.data(),
            source_k.size());
        std::memcpy(
            prefix_v_slab.data() + i * first_layout.v_bytes,
            source_v.data(),
            source_v.size());
    }

    ASSERT_TRUE(target.owner->resetRequestState(
        llaminar2::IKVCache::StateResetContext::testReinitialization(
            stream.opaque())));
    for (size_t i = 0; i < fa_layers.size(); ++i)
    {
        const int layer = fa_layers[i];
        llaminar2::IKVCache::KVCacheLogicalBlockDescriptor desc;
        desc.layer = layer;
        desc.seq_idx = 0;
        desc.logical_token_start = 0;
        desc.token_count = kTokenCount;
        desc.stream = stream.opaque();

        ASSERT_TRUE(target.owner->importLogicalBlock(
            desc,
            prefix_k_slab.data() + i * first_layout.k_bytes,
            prefix_v_slab.data() + i * first_layout.v_bytes))
            << "target import failed for global layer " << layer;

        auto [actual_k, actual_v] = exportLogicalBytes(
            target.owner.get(),
            layer,
            kTokenCount,
            stream);
        const auto expected_k = bytePattern(first_layout.k_bytes, 0x31000000u + static_cast<uint32_t>(layer));
        const auto expected_v = bytePattern(first_layout.v_bytes, 0x71000000u + static_cast<uint32_t>(layer));
        EXPECT_EQ(actual_k, expected_k)
            << "prefix slab roundtrip changed K payload for global layer " << layer;
        EXPECT_EQ(actual_v, expected_v)
            << "prefix slab roundtrip changed V payload for global layer " << layer;
    }
}

TEST(Test__ROCmHybridKVCacheReset, CapturedPrefillAppendPreservesLateFullAttentionPrefixPayload)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const auto hybrid_config = makeLateFullAttentionHybridConfig();
    constexpr int kMaxSeqLen = 512;
    constexpr int kTokenCount = 256;
    constexpr int kKvHeads = 2;
    constexpr int kHeadDim = 32;
    const int kv_cols = kKvHeads * kHeadDim;
    const std::vector<int> fa_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};

    auto cache = createHybridCacheWithShape(
        hybrid_config,
        llaminar2::ActivationPrecision::FP16,
        kMaxSeqLen,
        kKvHeads,
        kHeadDim);
    HipStream stream;

    struct CapturedLayerAppend
    {
        int layer = 0;
        std::unique_ptr<llaminar2::FP16Tensor> k;
        std::unique_ptr<llaminar2::FP16Tensor> v;
        std::unique_ptr<llaminar2::KVCacheAppendStage> stage;
        std::vector<uint8_t> expected_k;
        std::vector<uint8_t> expected_v;
    };

    std::vector<CapturedLayerAppend> appends;
    appends.reserve(fa_layers.size());
    for (int layer : fa_layers)
    {
        CapturedLayerAppend append;
        append.layer = layer;
        append.k = createDeviceFP16PatternTensor(
            kTokenCount,
            kv_cols,
            0x1000u + static_cast<uint32_t>(layer),
            stream);
        append.v = createDeviceFP16PatternTensor(
            kTokenCount,
            kv_cols,
            0x8000u + static_cast<uint32_t>(layer),
            stream);
        append.expected_k = tensorHostBytes(*append.k);
        append.expected_v = tensorHostBytes(*append.v);

        llaminar2::KVCacheAppendStage::Params params;
        params.device_id = llaminar2::DeviceId::rocm(0);
        params.K = append.k.get();
        params.V = append.v.get();
        params.kv_cache = cache.owner.get();
        params.layer_idx = layer;
        params.seq_idx = 0;
        params.num_tokens = kTokenCount;
        params.batch_size = 1;
        params.seq_len = kTokenCount;
        append.stage = std::make_unique<llaminar2::KVCacheAppendStage>(params);
        append.stage->setGPUStream(stream.opaque());
        append.stage->updatePrefillReplayParams({kTokenCount, kTokenCount, 0});
        append.stage->updateDynamicParams(/*pos_offset=*/0, kTokenCount);
        appends.push_back(std::move(append));
    }
    stream.synchronize("hipStreamSynchronize after captured append metadata upload");

    hipGraph_t graph = nullptr;
    hipGraphExec_t exec = nullptr;
    checkHip(hipStreamBeginCapture(stream.stream, hipStreamCaptureModeGlobal),
             "hipStreamBeginCapture captured prefill append");
    {
        llaminar2::GraphCaptureGuard guard;
        for (auto &append : appends)
        {
            ASSERT_TRUE(append.stage->execute(nullptr))
                << "captured append failed for layer " << append.layer;
        }
    }
    checkHip(hipStreamEndCapture(stream.stream, &graph),
             "hipStreamEndCapture captured prefill append");
    ASSERT_NE(graph, nullptr);
    checkHip(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
             "hipGraphInstantiate captured prefill append");
    checkHip(hipGraphLaunch(exec, stream.stream),
             "hipGraphLaunch captured prefill append");
    stream.synchronize("hipStreamSynchronize after captured prefill append launch");

    for (const auto &append : appends)
    {
        EXPECT_EQ(cache.owner->get_cached_tokens(append.layer, 0), kTokenCount)
            << "host KV count must advance exactly once for layer " << append.layer;
    }

    const auto expected31_it = std::find_if(
        appends.begin(),
        appends.end(),
        [](const CapturedLayerAppend &append)
        {
            return append.layer == 31;
        });
    ASSERT_NE(expected31_it, appends.end());
    const auto layer31 = exportLogicalBytes(
        cache.owner.get(),
        /*layer=*/31,
        kTokenCount,
        stream);
    EXPECT_EQ(layer31.first, expected31_it->expected_k)
        << "captured prefill append must preserve late FA K payload bytes";
    EXPECT_EQ(layer31.second, expected31_it->expected_v)
        << "captured prefill append must preserve late FA V payload bytes";

    if (exec)
        (void)hipGraphExecDestroy(exec);
    if (graph)
        (void)hipGraphDestroy(graph);
}

TEST(Test__ROCmHybridKVCacheReset, PartialHitHybridStateImportDoesNotOverwriteLateFullAttentionKV)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const auto hybrid_config = makeLateFullAttentionHybridConfig();
    constexpr int kTokenCount = 6;
    auto source = createHybridCacheWithShape(
        hybrid_config,
        llaminar2::ActivationPrecision::FP16,
        /*max_seq_len=*/16,
        /*n_kv_heads=*/4,
        /*head_dim=*/8);
    auto target = createHybridCacheWithShape(
        hybrid_config,
        llaminar2::ActivationPrecision::FP16,
        /*max_seq_len=*/16,
        /*n_kv_heads=*/4,
        /*head_dim=*/8);
    HipStream stream;

    const auto metadata = source.hybrid->hybridPrefixStateMetadata();
    EXPECT_EQ(metadata.host_bytes, 0u);
    ASSERT_GT(metadata.device_bytes, 0u);
    std::vector<uint8_t> hybrid_payload(metadata.device_bytes, 0xCD);

    llaminar2::HybridPrefixStateDescriptor export_desc;
    export_desc.seq_idx = 0;
    export_desc.logical_token_count = kTokenCount;
    export_desc.stream = stream.opaque();
    export_desc.include_host_state = false;
    export_desc.include_device_state = true;
    ASSERT_TRUE(source.hybrid->exportHybridPrefixState(
        export_desc,
        hybrid_payload.data(),
        nullptr));

    const auto layout = target.owner->logicalBlockLayout(/*layer=*/31, kTokenCount);
    ASSERT_GT(layout.k_bytes, 0u);
    ASSERT_GT(layout.v_bytes, 0u);
    const auto expected_k = bytePattern(layout.k_bytes, 0x3100001fu);
    const auto expected_v = bytePattern(layout.v_bytes, 0x7100001fu);
    llaminar2::IKVCache::KVCacheLogicalBlockDescriptor kv_desc;
    kv_desc.layer = 31;
    kv_desc.seq_idx = 0;
    kv_desc.logical_token_start = 0;
    kv_desc.token_count = kTokenCount;
    kv_desc.stream = stream.opaque();
    ASSERT_TRUE(target.owner->importLogicalBlock(
        kv_desc,
        expected_k.data(),
        expected_v.data()));

    llaminar2::HybridPrefixStateDescriptor import_desc;
    import_desc.seq_idx = 0;
    import_desc.logical_token_count = kTokenCount;
    import_desc.stream = stream.opaque();
    import_desc.include_host_state = false;
    import_desc.include_device_state = true;
    ASSERT_TRUE(target.hybrid->importHybridPrefixState(
        import_desc,
        hybrid_payload.data(),
        nullptr));

    const auto actual = exportLogicalBytes(
        target.owner.get(),
        /*layer=*/31,
        kTokenCount,
        stream);
    EXPECT_EQ(actual.first, expected_k)
        << "partial-hit hybrid restore must not modify late FA K payload";
    EXPECT_EQ(actual.second, expected_v)
        << "partial-hit hybrid restore must not modify late FA V payload";
}

#else

TEST(Test__ROCmHybridKVCacheReset, SkipsWithoutROCm)
{
    GTEST_SKIP() << "ROCm support not compiled";
}

#endif
