/**
 * @file Test__ROCmRingKVCacheTQ.cpp
 * @brief Comprehensive unit tests for ROCm TurboQuant KV Cache
 *
 * Ports the CUDA TQ KV cache test suite (Test__CUDARingKVCacheTQ.cpp)
 * to ROCm/HIP. Tests the ROCmRingKVCacheTQ class which stores K in TQ8
 * (8-bit, 256 centroids) and V in TQ4 (4-bit, 16 centroids).
 *
 * Tests:
 * 1.  Basic append + retrieve roundtrip (TQ8-K/TQ4-V)
 * 2.  Ring buffer wrap-around preserves newest tokens
 * 3.  Incremental decode-like append pattern
 * 4.  Multi-layer independent data
 * 5.  Clear/reset semantics
 * 6.  Quantization error bounds (cosine similarity)
 * 7.  TQ8 K quality strictly better than TQ4 V
 * 8.  get_kv_converted with RoPE-on-read
 * 9.  get_kv_converted without RoPE (dequant only)
 * 10. Eviction correctness
 * 11. Shadow buffer invalidation on append
 * 12. Head dim 128 support
 * 13. RoPE position correctness
 * 14. Host-created tensors explicitly prepared before device-only append
 * 15. Metadata accessor verification
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <numeric>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <string>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/kvcache/ROCmRingKVCacheTQ.h"
#include "kernels/rocm/kvcache/ROCmRingKVCacheTQFactory.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/IKVCache.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "tensors/GpuTensorView.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    bool hasROCm()
    {
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
    }

    /**
     * @brief Owns a non-default HIP stream for tests that must use explicit-stream cache append.
     *
     * The ROCm TQ cache now rejects no-stream append paths. A scoped stream keeps the tests aligned
     * with stage/graph execution, where all GPU KV writes and reads use the caller's stream.
     */
    class ScopedHipStream
    {
    public:
        ScopedHipStream()
        {
            EXPECT_EQ(hipStreamCreate(&stream_), hipSuccess);
        }

        ~ScopedHipStream()
        {
            if (stream_)
                (void)hipStreamDestroy(stream_);
        }

        void *opaque() const { return static_cast<void *>(stream_); }
        hipStream_t stream() const { return stream_; }

        void synchronize() const
        {
            ASSERT_NE(stream_, nullptr);
            ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
        }

    private:
        hipStream_t stream_ = nullptr;
    };

    /// @brief Append through the required explicit-stream API.
    bool appendWithTestStream(ROCmRingKVCacheTQ &cache, int layer, int seq_idx,
                              const ITensor *K, const ITensor *V, int num_tokens,
                              const ScopedHipStream &stream)
    {
        auto ensure = [&](const ITensor *tensor)
        {
            if (tensor && tensor->gpu_data_ptr())
                return true;
            auto *base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(tensor));
            return base && base->ensureOnDevice(DeviceId::rocm(0), stream.opaque());
        };
        if (!ensure(K) || !ensure(V))
            return false;
        return cache.appendWithStream(layer, seq_idx, K, V, num_tokens, stream.opaque());
    }

    std::vector<float> generateRandomFP32(size_t count, unsigned seed = 42)
    {
        std::vector<float> data(count);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto &val : data)
            val = dist(rng);
        return data;
    }

    float computeCosineSimilarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        if (norm_a < 1e-30 || norm_b < 1e-30)
            return 0.0f;
        return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
    }

    float computeMSE(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = static_cast<double>(a[i]) - b[i];
            sum += diff * diff;
        }
        return static_cast<float>(sum / n);
    }

    // Upload FP32 host vector to GPU, returning device pointer
    float *uploadToGPU(const std::vector<float> &host_data)
    {
        float *d_ptr = nullptr;
        size_t bytes = host_data.size() * sizeof(float);
        (void)hipMalloc(&d_ptr, bytes);
        (void)hipMemcpy(d_ptr, host_data.data(), bytes, hipMemcpyHostToDevice);
        return d_ptr;
    }

    // Download FP16 GPU buffer to FP32 host vector
    std::vector<float> downloadFP16ToFP32(const void *d_ptr, size_t count)
    {
        std::vector<uint16_t> h_fp16(count);
        (void)hipMemcpy(h_fp16.data(), d_ptr, count * sizeof(uint16_t), hipMemcpyDeviceToHost);

        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i)
            result[i] = fp16_to_fp32(h_fp16[i]);
        return result;
    }

    // Helper: create FP32Tensor from host data
    std::unique_ptr<FP32Tensor> createFP32Tensor(const std::vector<float> &data,
                                                 size_t rows, size_t cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        std::memcpy(tensor->mutable_data(), data.data(), data.size() * sizeof(float));
        return tensor;
    }

    /**
     * @brief Verify an empty converted read reports no live K/V tensors.
     *
     * This guards the request-reset path against stale converted scratch pointers after a clear.
     */
    void expectConvertedEmpty(ROCmRingKVCacheTQ &cache, int layer, int seq_idx)
    {
        ITensor *out_k = reinterpret_cast<ITensor *>(static_cast<uintptr_t>(0x1));
        ITensor *out_v = reinterpret_cast<ITensor *>(static_cast<uintptr_t>(0x1));
        int kv_len = -1;

        ASSERT_TRUE(cache.get_kv_converted(layer, seq_idx, ActivationPrecision::FP16,
                                           &out_k, &out_v, &kv_len, nullptr));
        EXPECT_EQ(kv_len, 0);
        EXPECT_EQ(out_k, nullptr);
        EXPECT_EQ(out_v, nullptr);

        const ITensor *raw_k = cache.get_k(layer, seq_idx);
        EXPECT_EQ(raw_k, nullptr)
            << "An empty device-owned ring has no scalar tensor view";
    }

} // namespace

// =============================================================================
// 1. Basic Append + Retrieve Roundtrip (Split TQ8-K / TQ4-V)
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, BasicAppendRetrieve_SplitTQ)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    ASSERT_NE(cache_ptr, nullptr);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    EXPECT_EQ(cache->n_layers(), n_layers);
    EXPECT_EQ(cache->max_seq_len(), max_seq_len);
    EXPECT_EQ(cache->n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache->head_dim(), head_dim);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    // Append 10 tokens
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);

    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Retrieve via get_k/get_v (returns FP16 shadow buffers)
    const ITensor *out_k = cache->get_k(0, 0);
    const ITensor *out_v = cache->get_v(0, 0);
    ASSERT_NE(out_k, nullptr);
    ASSERT_NE(out_v, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(num_tokens));
    EXPECT_EQ(out_v->shape()[0], static_cast<size_t>(num_tokens));

    // Download FP16 and verify cosine similarity
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    float min_cos_k = 1.0f, avg_cos_k = 0.0f;
    float min_cos_v = 1.0f, avg_cos_v = 0.0f;
    for (int t = 0; t < num_tokens; ++t)
    {
        float cos_k = computeCosineSimilarity(
            h_K.data() + t * kv_dim, result_K.data() + t * kv_dim, kv_dim);
        float cos_v = computeCosineSimilarity(
            h_V.data() + t * kv_dim, result_V.data() + t * kv_dim, kv_dim);
        min_cos_k = std::min(min_cos_k, cos_k);
        min_cos_v = std::min(min_cos_v, cos_v);
        avg_cos_k += cos_k;
        avg_cos_v += cos_v;
    }
    avg_cos_k /= num_tokens;
    avg_cos_v /= num_tokens;

    EXPECT_GT(avg_cos_k, 0.97f) << "TQ8 K average cosine too low";
    EXPECT_GT(min_cos_k, 0.94f) << "TQ8 K minimum cosine too low";
    EXPECT_GT(avg_cos_v, 0.88f) << "TQ4 V average cosine too low";
    EXPECT_GT(min_cos_v, 0.78f) << "TQ4 V minimum cosine too low";

    LOG_INFO("[Test] ROCm Split TQ roundtrip: K cos=" << avg_cos_k << "/" << min_cos_k
                                                      << ", V cos=" << avg_cos_v << "/" << min_cos_v);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 2. Ring Buffer Wrap-Around
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, WrapAround_PreservesNewest)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    // Append 12 tokens (overwrites first 4)
    const int num_tokens = 12;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 100);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 200);

    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    // Append in two batches
    auto k1 = std::make_unique<GpuTensorView>(d_K, 8, kv_dim, TensorType::FP32, 0);
    auto v1 = std::make_unique<GpuTensorView>(d_V, 8, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k1.get(), v1.get(), 8, stream));

    auto k2 = std::make_unique<GpuTensorView>(
        d_K + 8 * kv_dim, 4, kv_dim, TensorType::FP32, 0);
    auto v2 = std::make_unique<GpuTensorView>(
        d_V + 8 * kv_dim, 4, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k2.get(), v2.get(), 4, stream));
    stream.synchronize();

    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq_len);

    const ITensor *out_k = cache->get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(max_seq_len));

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 3. Incremental Decode-Like Append
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, IncrementalAppend_DecodeLike)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    std::vector<std::vector<float>> all_K;
    const int decode_steps = 20;

    for (int step = 0; step < decode_steps; ++step)
    {
        auto h_K = generateRandomFP32(kv_dim, 1000 + step);
        auto h_V = generateRandomFP32(kv_dim, 2000 + step);
        all_K.push_back(h_K);

        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);

        auto kv = std::make_unique<GpuTensorView>(d_K, 1, kv_dim, TensorType::FP32, 0);
        auto vv = std::make_unique<GpuTensorView>(d_V, 1, kv_dim, TensorType::FP32, 0);

        ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), 1, stream));
        stream.synchronize();
        EXPECT_EQ(cache->get_cached_tokens(0, 0), step + 1);

        (void)hipFree(d_K);
        (void)hipFree(d_V);
    }

    const ITensor *out_k = cache->get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(decode_steps));

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), decode_steps * kv_dim);

    float min_cos = 1.0f;
    for (int t = 0; t < decode_steps; ++t)
    {
        float cos = computeCosineSimilarity(
            all_K[t].data(), result_K.data() + t * kv_dim, kv_dim);
        min_cos = std::min(min_cos, cos);
    }
    EXPECT_GT(min_cos, 0.94f) << "TQ8 K decode cosine too low: " << min_cos;
}

// =============================================================================
// 4. Multi-Layer Independent Data
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, MultiLayer_IndependentData)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_layers = 4;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 5;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        n_layers, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    for (int layer = 0; layer < n_layers; ++layer)
    {
        auto h_K = generateRandomFP32(num_tokens * kv_dim, 100 * layer + 1);
        auto h_V = generateRandomFP32(num_tokens * kv_dim, 100 * layer + 2);
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);

        auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
        auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);

        ASSERT_TRUE(appendWithTestStream(*cache, layer, 0, kv.get(), vv.get(), num_tokens, stream));
        stream.synchronize();
        EXPECT_EQ(cache->get_cached_tokens(layer, 0), num_tokens);

        (void)hipFree(d_K);
        (void)hipFree(d_V);
    }

    // Verify layers are different
    std::vector<std::vector<float>> layer_results;
    for (int layer = 0; layer < n_layers; ++layer)
    {
        const ITensor *out_k = cache->get_k(layer, 0);
        ASSERT_NE(out_k, nullptr);
        layer_results.push_back(
            downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim));
    }

    for (int i = 0; i < n_layers; ++i)
    {
        for (int j = i + 1; j < n_layers; ++j)
        {
            float cos = computeCosineSimilarity(
                layer_results[i].data(), layer_results[j].data(), num_tokens * kv_dim);
            EXPECT_LT(cos, 0.5f) << "Layers " << i << " and " << j
                                 << " too similar (cos=" << cos << ")";
        }
    }
}

// =============================================================================
// 5. Clear/Reset Semantics
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, Clear_ResetsAllLayers)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_layers = 2;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        n_layers, 1, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(5 * kv_dim, 100);
    auto h_V = generateRandomFP32(5 * kv_dim, 200);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, 5, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, 5, kv_dim, TensorType::FP32, 0);

    for (int l = 0; l < n_layers; ++l)
        ASSERT_TRUE(appendWithTestStream(*cache, l, 0, kv.get(), vv.get(), 5, stream));
    stream.synchronize();

    // Clear single sequence
    ASSERT_TRUE(cache->resetLayerSequenceState(
        0,
        0,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(1, 0), 5);

    // Clear all
    ASSERT_TRUE(cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    for (int l = 0; l < n_layers; ++l)
        EXPECT_EQ(cache->get_cached_tokens(l, 0), 0);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

TEST(Test__ROCmRingKVCacheTQ, AppendRequiresExplicitNonNullStream)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 3;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(1, 1, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 700);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 701);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);
    auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);

    EXPECT_FALSE(cache->append(0, 0, k_view.get(), v_view.get(), num_tokens));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
    EXPECT_FALSE(cache->appendWithStream(0, 0, k_view.get(), v_view.get(), num_tokens, nullptr));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    ScopedHipStream stream;
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

TEST(Test__ROCmRingKVCacheTQ, ClearSequenceLayerAndAllInvalidateConvertedScratch)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_layers = 2;
    const int batch_size = 2;
    const int num_tokens = 4;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(n_layers, batch_size, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto append_seeded = [&](int layer, int seq_idx, unsigned seed)
    {
        auto h_K = generateRandomFP32(num_tokens * kv_dim, seed);
        auto h_V = generateRandomFP32(num_tokens * kv_dim, seed + 1000);
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);
        auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
        auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
        ASSERT_TRUE(appendWithTestStream(*cache, layer, seq_idx, k_view.get(), v_view.get(), num_tokens, stream));
        stream.synchronize();
        (void)hipFree(d_K);
        (void)hipFree(d_V);
    };

    append_seeded(0, 0, 10);
    append_seeded(0, 1, 20);
    append_seeded(1, 0, 30);
    append_seeded(1, 1, 40);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 1, ActivationPrecision::FP16,
                                        &out_k, &out_v, &kv_len, nullptr));
    ASSERT_EQ(kv_len, num_tokens);
    ASSERT_NE(out_k, nullptr);

    ASSERT_TRUE(cache->resetLayerSequenceState(
        0,
        1,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(*cache, 0, 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
    EXPECT_EQ(cache->get_cached_tokens(1, 0), num_tokens);
    EXPECT_EQ(cache->get_cached_tokens(1, 1), num_tokens);

    ASSERT_TRUE(cache->resetLayerState(
        1,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(*cache, 1, 0);
    expectConvertedEmpty(*cache, 1, 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    ASSERT_TRUE(cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(*cache, 0, 0);
    expectConvertedEmpty(*cache, 0, 1);
}

TEST(Test__ROCmRingKVCacheTQ, ClearThenReappendConvertedScratchUsesNewRows)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 6;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(1, 1, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto append_host = [&](const std::vector<float> &h_K, const std::vector<float> &h_V)
    {
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);
        auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
        auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
        ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
        stream.synchronize();
        (void)hipFree(d_K);
        (void)hipFree(d_V);
    };

    auto h_K_a = generateRandomFP32(num_tokens * kv_dim, 900);
    auto h_V_a = generateRandomFP32(num_tokens * kv_dim, 901);
    auto h_K_b = generateRandomFP32(num_tokens * kv_dim, 1900);
    auto h_V_b = generateRandomFP32(num_tokens * kv_dim, 1901);

    append_host(h_K_a, h_V_a);

    ITensor *out_k_a = nullptr;
    ITensor *out_v_a = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_a, &out_v_a, &kv_len, nullptr));
    ASSERT_EQ(kv_len, num_tokens);
    auto k_a = downloadFP16ToFP32(out_k_a->gpu_data_ptr(), num_tokens * kv_dim);

    ASSERT_TRUE(cache->resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(*cache, 0, 0);

    append_host(h_K_b, h_V_b);
    ITensor *out_k_b = nullptr;
    ITensor *out_v_b = nullptr;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_b, &out_v_b, &kv_len, nullptr));
    ASSERT_EQ(kv_len, num_tokens);
    auto k_b = downloadFP16ToFP32(out_k_b->gpu_data_ptr(), num_tokens * kv_dim);
    auto v_b = downloadFP16ToFP32(out_v_b->gpu_data_ptr(), num_tokens * kv_dim);

    EXPECT_GT(computeCosineSimilarity(h_K_b.data(), k_b.data(), h_K_b.size()), 0.94f);
    EXPECT_GT(computeCosineSimilarity(h_V_b.data(), v_b.data(), h_V_b.size()), 0.78f);
    EXPECT_LT(computeCosineSimilarity(k_a.data(), k_b.data(), k_b.size()), 0.5f)
        << "Converted scratch after clear/reappend still resembles the old request";
}

// =============================================================================
// 6. Quantization Error Bounds
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, QuantizationError_WithinBounds)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 32;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 64, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 314);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 271);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    const ITensor *out_k = cache->get_k(0, 0);
    const ITensor *out_v = cache->get_v(0, 0);
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    float mse_k = computeMSE(h_K.data(), result_K.data(), num_tokens * kv_dim);
    float mse_v = computeMSE(h_V.data(), result_V.data(), num_tokens * kv_dim);

    EXPECT_LT(mse_k, 0.02f) << "TQ8 K MSE too high";
    EXPECT_LT(mse_v, 0.08f) << "TQ4 V MSE too high";

    LOG_INFO("[Test] ROCm Quantization MSE: K=" << mse_k << " V=" << mse_v);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 7. TQ8 K Quality Strictly Better Than TQ4 V
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, KQuality_StrictlyBetterThan_V)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 20;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 64, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_data = generateRandomFP32(num_tokens * kv_dim, 999);
    float *d_data = uploadToGPU(h_data);

    auto view = std::make_unique<GpuTensorView>(d_data, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, view.get(), view.get(), num_tokens, stream)); // Same data for K and V
    stream.synchronize();

    const ITensor *out_k = cache->get_k(0, 0);
    const ITensor *out_v = cache->get_v(0, 0);
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    float cos_k = computeCosineSimilarity(h_data.data(), result_K.data(), num_tokens * kv_dim);
    float cos_v = computeCosineSimilarity(h_data.data(), result_V.data(), num_tokens * kv_dim);

    EXPECT_GT(cos_k, cos_v) << "TQ8 K should be higher quality than TQ4 V";
    LOG_INFO("[Test] ROCm Same-data quality: K cos=" << cos_k << " V cos=" << cos_v);

    (void)hipFree(d_data);
}

// =============================================================================
// 8. get_kv_converted with RoPE-on-read
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, GetKVConverted_WithRoPE)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 8;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    // Get without RoPE
    ITensor *out_k_noRoPE = nullptr;
    ITensor *out_v_noRoPE = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_noRoPE, &out_v_noRoPE, &kv_len, nullptr));
    EXPECT_EQ(kv_len, num_tokens);

    auto result_noRoPE = downloadFP16ToFP32(out_k_noRoPE->gpu_data_ptr(), num_tokens * kv_dim);

    // Get with RoPE
    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = 10000.0f;
    rope_params.position_start = 0;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    ITensor *out_k_rope = nullptr;
    ITensor *out_v_rope = nullptr;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_rope, &out_v_rope, &kv_len, &rope_params));
    EXPECT_EQ(kv_len, num_tokens);

    auto result_withRoPE = downloadFP16ToFP32(out_k_rope->gpu_data_ptr(), num_tokens * kv_dim);

    // RoPE should change K values
    float diff = 0.0f;
    for (size_t i = 0; i < result_noRoPE.size(); ++i)
        diff += std::abs(result_noRoPE[i] - result_withRoPE[i]);
    EXPECT_GT(diff, 0.01f) << "RoPE should modify K values";

    // V should be unchanged by RoPE
    auto v_noRoPE = downloadFP16ToFP32(out_v_noRoPE->gpu_data_ptr(), num_tokens * kv_dim);
    auto v_withRoPE = downloadFP16ToFP32(out_v_rope->gpu_data_ptr(), num_tokens * kv_dim);
    float v_diff = 0.0f;
    for (size_t i = 0; i < v_noRoPE.size(); ++i)
        v_diff += std::abs(v_noRoPE[i] - v_withRoPE[i]);
    EXPECT_LT(v_diff, 0.01f) << "RoPE should NOT modify V values";

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 9. get_kv_converted Without RoPE (Dequant Only)
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, GetKVConverted_DequantOnly)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 10;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 555);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 666);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k, &out_v, &kv_len, nullptr));
    EXPECT_EQ(kv_len, num_tokens);

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);

    // Should match get_k() output
    const ITensor *direct_k = cache->get_k(0, 0);
    auto direct_K = downloadFP16ToFP32(direct_k->gpu_data_ptr(), num_tokens * kv_dim);

    for (size_t i = 0; i < result_K.size(); ++i)
        EXPECT_NEAR(result_K[i], direct_K[i], 1e-6f) << "Mismatch at index " << i;

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 10. Eviction Correctness
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, Eviction_ReducesCount)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 20;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    // Evict 5 tokens
    cache->evict_oldest(0, 0, 5);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens - 5);

    // Evict all remaining
    cache->evict_oldest(0, 0, num_tokens);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 11. Shadow Buffer Invalidation on Append
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, ShadowInvalidation_AfterAppend)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    // Append batch 1
    auto h_K1 = generateRandomFP32(5 * kv_dim, 100);
    auto h_V1 = generateRandomFP32(5 * kv_dim, 200);
    float *d_K1 = uploadToGPU(h_K1);
    float *d_V1 = uploadToGPU(h_V1);

    auto kv1 = std::make_unique<GpuTensorView>(d_K1, 5, kv_dim, TensorType::FP32, 0);
    auto vv1 = std::make_unique<GpuTensorView>(d_V1, 5, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv1.get(), vv1.get(), 5, stream));
    stream.synchronize();

    // Force shadow creation
    const ITensor *k1 = cache->get_k(0, 0);
    ASSERT_NE(k1, nullptr);
    EXPECT_EQ(k1->shape()[0], 5u);

    // Append batch 2
    auto h_K2 = generateRandomFP32(3 * kv_dim, 300);
    auto h_V2 = generateRandomFP32(3 * kv_dim, 400);
    float *d_K2 = uploadToGPU(h_K2);
    float *d_V2 = uploadToGPU(h_V2);

    auto kv2 = std::make_unique<GpuTensorView>(d_K2, 3, kv_dim, TensorType::FP32, 0);
    auto vv2 = std::make_unique<GpuTensorView>(d_V2, 3, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv2.get(), vv2.get(), 3, stream));
    stream.synchronize();

    // Shadow should be regenerated with new count
    const ITensor *k2 = cache->get_k(0, 0);
    ASSERT_NE(k2, nullptr);
    EXPECT_EQ(k2->shape()[0], 8u); // 5 + 3

    (void)hipFree(d_K1);
    (void)hipFree(d_V1);
    (void)hipFree(d_K2);
    (void)hipFree(d_V2);
}

// =============================================================================
// 12. Head Dim 128 Support
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, HeadDim128_BasicRoundtrip)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_kv_heads = 4;
    const int head_dim = 128;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 777);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 888);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    const ITensor *out_k = cache->get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);

    float avg_cos = 0.0f;
    for (int t = 0; t < num_tokens; ++t)
    {
        float cos = computeCosineSimilarity(
            h_K.data() + t * kv_dim, result_K.data() + t * kv_dim, kv_dim);
        avg_cos += cos;
    }
    avg_cos /= num_tokens;
    EXPECT_GT(avg_cos, 0.96f) << "HeadDim128 TQ8 K cosine too low";

    LOG_INFO("[Test] ROCm HeadDim128 avg K cosine: " << avg_cos);

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 13. RoPE Position Correctness
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, RoPE_PositionCorrectness)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 4;
    const int n_kv_heads = 1;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const float rope_theta = 10000.0f;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    std::vector<float> h_K(num_tokens * kv_dim, 1.0f);
    std::vector<float> h_V(num_tokens * kv_dim, 1.0f);

    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, 0);
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, 0);
    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    // Get with two different position starts
    IKVCache::KVReadParams rope_params;
    rope_params.rope_theta = rope_theta;
    rope_params.n_kv_heads = n_kv_heads;
    rope_params.head_dim = head_dim;

    rope_params.position_start = 0;
    ITensor *out_k_pos0 = nullptr;
    ITensor *out_v_pos0 = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_pos0, &out_v_pos0, &kv_len, &rope_params));

    auto result_pos0 = downloadFP16ToFP32(out_k_pos0->gpu_data_ptr(), num_tokens * kv_dim);

    rope_params.position_start = 10;
    ITensor *out_k_pos10 = nullptr;
    ITensor *out_v_pos10 = nullptr;
    ASSERT_TRUE(cache->get_kv_converted(0, 0, ActivationPrecision::FP16,
                                        &out_k_pos10, &out_v_pos10, &kv_len, &rope_params));

    auto result_pos10 = downloadFP16ToFP32(out_k_pos10->gpu_data_ptr(), num_tokens * kv_dim);

    // Different position offsets should produce different results
    float diff = 0.0f;
    for (size_t i = 0; i < result_pos0.size(); ++i)
        diff += std::abs(result_pos0[i] - result_pos10[i]);
    EXPECT_GT(diff, 0.01f) << "Different position offsets should produce different K values";

    (void)hipFree(d_K);
    (void)hipFree(d_V);
}

// =============================================================================
// 14. Host-created tensors are explicitly prepared before device-only append
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, HostCreatedTensorIsPreparedOnDeviceBeforeAppend)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int num_tokens = 5;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        1, 1, 32, n_kv_heads, head_dim, &tq_ctx, 0);
    auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_ptr.get());
    ASSERT_NE(cache, nullptr);
    ScopedHipStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);

    auto k_tensor = createFP32Tensor(h_K, num_tokens, kv_dim);
    auto v_tensor = createFP32Tensor(h_V, num_tokens, kv_dim);

    ASSERT_TRUE(appendWithTestStream(*cache, 0, 0, k_tensor.get(), v_tensor.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    const ITensor *out_k = cache->get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(num_tokens));

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    float cos_k = computeCosineSimilarity(h_K.data(), result_K.data(), num_tokens * kv_dim);
    EXPECT_GT(cos_k, 0.94f) << "Prepared device append quality too low";
}

// =============================================================================
// 15. Metadata Accessor Verification
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, MetadataAccessors)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    const int n_layers = 3;
    const int batch_size = 2;
    const int max_seq_len = 64;
    const int n_kv_heads = 4;
    const int head_dim = 128;

    TurboQuantContext tq_ctx(head_dim, 42);
    auto cache_ptr = createROCmRingKVCacheTQ(
        n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    ASSERT_NE(cache_ptr, nullptr);

    EXPECT_EQ(cache_ptr->n_layers(), n_layers);
    EXPECT_EQ(cache_ptr->max_seq_len(), max_seq_len);
    EXPECT_EQ(cache_ptr->k_precision(), ActivationPrecision::TQ8);
    EXPECT_EQ(cache_ptr->v_precision(), ActivationPrecision::TQ4);
    EXPECT_FALSE(cache_ptr->is_sharded());
}

/**
 * @brief Prove grouped resident TQ reads equal scalar ROCm dequant bytes.
 *
 * The test covers both supported TQ block dimensions, a wrapped request-zero
 * ring, and a shorter request-one ring. The grouped device-state path is
 * captured and replayed as production attention uses it. Every live FP16 K/V
 * word must equal the existing scalar dequant route. Inactive capacity remains
 * unspecified because attention consumes only the canonical device count;
 * clearing the maximum context on every short decode would waste bandwidth.
 */
TEST(Test__ROCmRingKVCacheTQ, CapturedResidentRequestBatchMatchesScalarDequantBytes)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int batch_size = 2;
    constexpr int max_seq_len = 6;
    constexpr int n_kv_heads = 2;
    constexpr std::array<int, batch_size> expected_counts{6, 4};

    for (const int head_dim : {64, 128})
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));
        const int kv_dim = n_kv_heads * head_dim;
        TurboQuantContext tq_ctx(head_dim, 42);
        auto cache_owner = createROCmRingKVCacheTQ(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0);
        auto *cache = dynamic_cast<ROCmRingKVCacheTQ *>(cache_owner.get());
        ASSERT_NE(cache, nullptr);
        ScopedHipStream stream;
        std::vector<float *> allocations;

        auto appendChunk = [&](int request, int rows, unsigned k_seed, unsigned v_seed)
        {
            const auto host_k = generateRandomFP32(
                static_cast<size_t>(rows) * kv_dim, k_seed);
            const auto host_v = generateRandomFP32(
                static_cast<size_t>(rows) * kv_dim, v_seed);
            float *device_k = uploadToGPU(host_k);
            float *device_v = uploadToGPU(host_v);
            allocations.push_back(device_k);
            allocations.push_back(device_v);
            GpuTensorView k_view(
                device_k, rows, kv_dim, TensorType::FP32, /*device_id=*/0);
            GpuTensorView v_view(
                device_v, rows, kv_dim, TensorType::FP32, /*device_id=*/0);
            return appendWithTestStream(
                *cache, /*layer=*/0, request, &k_view, &v_view, rows, stream);
        };

        ASSERT_TRUE(appendChunk(0, 5, 101, 201));
        ASSERT_TRUE(appendChunk(0, 4, 102, 202));
        ASSERT_TRUE(appendChunk(1, 4, 103, 203));
        stream.synchronize();
        for (float *allocation : allocations)
            ASSERT_EQ(hipFree(allocation), hipSuccess);
        ASSERT_EQ(cache->get_cached_tokens(0, 0), expected_counts[0]);
        ASSERT_EQ(cache->get_cached_tokens(0, 1), expected_counts[1]);

        std::array<std::vector<uint16_t>, batch_size> scalar_k;
        std::array<std::vector<uint16_t>, batch_size> scalar_v;
        IKVCache::KVReadParams read_params;
        read_params.gpu_stream = stream.opaque();
        for (int request = 0; request < batch_size; ++request)
        {
            ITensor *serial_k = nullptr;
            ITensor *serial_v = nullptr;
            int serial_count = 0;
            ASSERT_TRUE(cache->get_kv_converted(
                0, request, ActivationPrecision::FP16,
                &serial_k, &serial_v, &serial_count, &read_params));
            ASSERT_EQ(serial_count, expected_counts[request]);
            ASSERT_NE(serial_k, nullptr);
            ASSERT_NE(serial_v, nullptr);
            stream.synchronize();

            const size_t elements =
                static_cast<size_t>(serial_count) * kv_dim;
            scalar_k[request].resize(elements);
            scalar_v[request].resize(elements);
            ASSERT_EQ(
                hipMemcpy(scalar_k[request].data(), serial_k->gpu_data_ptr(),
                          elements * sizeof(uint16_t), hipMemcpyDeviceToHost),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(scalar_v[request].data(), serial_v->gpu_data_ptr(),
                          elements * sizeof(uint16_t), hipMemcpyDeviceToHost),
                hipSuccess);
        }

        ITensor *grouped_k = nullptr;
        ITensor *grouped_v = nullptr;
        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal),
            hipSuccess);
        bool gather_ok = false;
        {
            GraphCaptureGuard guard;
            gather_ok = cache->get_kv_batched_device_view(
                /*layer=*/0,
                /*first_seq_idx=*/0,
                batch_size,
                &grouped_k,
                &grouped_v,
                stream.opaque());
        }
        ASSERT_EQ(hipStreamEndCapture(stream.stream(), &graph), hipSuccess);
        ASSERT_TRUE(gather_ok);
        ASSERT_NE(grouped_k, nullptr);
        ASSERT_NE(grouped_v, nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream.stream()), hipSuccess);
        stream.synchronize();

        const size_t grouped_elements =
            static_cast<size_t>(batch_size) * max_seq_len * kv_dim;
        std::vector<uint16_t> actual_k(grouped_elements);
        std::vector<uint16_t> actual_v(grouped_elements);
        ASSERT_EQ(
            hipMemcpy(actual_k.data(), grouped_k->gpu_data_ptr(),
                      grouped_elements * sizeof(uint16_t), hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(actual_v.data(), grouped_v->gpu_data_ptr(),
                      grouped_elements * sizeof(uint16_t), hipMemcpyDeviceToHost),
            hipSuccess);

        for (int request = 0; request < batch_size; ++request)
        {
            const size_t request_offset =
                static_cast<size_t>(request) * max_seq_len * kv_dim;
            const size_t live_elements =
                static_cast<size_t>(expected_counts[request]) * kv_dim;
            auto expectWordsEqual = [&](const uint16_t *actual,
                                        const std::vector<uint16_t> &expected,
                                        const char *label)
            {
                size_t first_difference = live_elements;
                for (size_t index = 0; index < live_elements; ++index)
                {
                    if (actual[index] != expected[index])
                    {
                        first_difference = index;
                        break;
                    }
                }
                EXPECT_EQ(first_difference, live_elements)
                    << label << " grouped bytes differ for request " << request
                    << " first_word=" << first_difference
                    << " row=" << first_difference / static_cast<size_t>(kv_dim)
                    << " column=" << first_difference % static_cast<size_t>(kv_dim)
                    << " actual_bits=0x" << std::hex
                    << (first_difference < live_elements
                            ? actual[first_difference]
                            : uint16_t{0})
                    << " expected_bits=0x"
                    << (first_difference < live_elements
                            ? expected[first_difference]
                            : uint16_t{0})
                    << std::dec;
            };
            expectWordsEqual(
                actual_k.data() + request_offset, scalar_k[request], "TQ8 K");
            expectWordsEqual(
                actual_v.data() + request_offset, scalar_v[request], "TQ4 V");
        }

        ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    }
}

/**
 * @brief Proves unequal captured TQ appends preserve exact continuation bytes.
 *
 * The production stage records an eight-row FP32-to-TQ append for two requests,
 * while resident device metadata publishes logical lengths `{8, 5}`. Replaying
 * that graph with one real row against a ten-row ring forces every illegal
 * padded store to wrap through live history. The resulting TQ8-K/TQ4-V cache is
 * dequantized and compared byte-for-byte with an exact-row reference cache.
 * Both supported block dimensions execute their native HIP specializations.
 */
TEST(Test__ROCmRingKVCacheTQ, CapturedUnequalRequestLengthsPreserveContinuationBytes)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int batch_size = 2;
    constexpr int captured_rows = 8;
    constexpr int max_seq_len = 10;
    constexpr int n_kv_heads = 2;
    constexpr std::array<int, batch_size> initial_counts{captured_rows, 5};
    constexpr std::array<int, batch_size> final_counts{captured_rows + 1, 6};

    for (const int head_dim : {64, 128})
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));
        const int kv_dim = n_kv_heads * head_dim;
        const size_t source_elements =
            static_cast<size_t>(batch_size) * captured_rows * kv_dim;
        auto k_values = generateRandomFP32(source_elements, 1701 + head_dim);
        auto v_values = generateRandomFP32(source_elements, 1907 + head_dim);
        for (int request = 0; request < batch_size; ++request)
        {
            const size_t request_begin =
                static_cast<size_t>(request) * captured_rows * kv_dim;
            for (size_t index = 0;
                 index < static_cast<size_t>(captured_rows) * kv_dim;
                 ++index)
            {
                k_values[request_begin + index] += 0.5f * request;
                v_values[request_begin + index] -= 0.375f * request;
            }
        }

        auto k_tensor = createFP32Tensor(
            k_values, batch_size * captured_rows, kv_dim);
        auto v_tensor = createFP32Tensor(
            v_values, batch_size * captured_rows, kv_dim);
        ASSERT_NE(k_tensor, nullptr);
        ASSERT_NE(v_tensor, nullptr);

        TurboQuantContext tq_ctx(head_dim, 42);
        ScopedHipStream stream;
        ROCmRingKVCacheTQ actual(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0);
        ROCmRingKVCacheTQ reference(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0);
        ASSERT_TRUE(k_tensor->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));
        ASSERT_TRUE(v_tensor->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));

        int32_t *device_lengths = nullptr;
        ASSERT_EQ(
            hipMalloc(&device_lengths, batch_size * sizeof(int32_t)),
            hipSuccess);
        const std::array<int32_t, batch_size> first_device_lengths{
            initial_counts[0], initial_counts[1]};
        ASSERT_EQ(
            hipMemcpyAsync(
                device_lengths,
                first_device_lengths.data(),
                batch_size * sizeof(int32_t),
                hipMemcpyHostToDevice,
                stream.stream()),
            hipSuccess);

        KVCacheAppendStage append_stage({
            .device_id = DeviceId::rocm(0),
            .K = k_tensor.get(),
            .V = v_tensor.get(),
            .kv_cache = &actual,
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = batch_size * captured_rows,
            .batch_size = batch_size,
            .seq_len = captured_rows,
            .request_sequence_lengths_device = device_lengths,
            .head_dim = head_dim,
            .turboquant_ctx = &tq_ctx,
        });
        append_stage.setGPUStream(stream.opaque());
        append_stage.updateDynamicParams(/*pos_offset=*/0, captured_rows);
        stream.synchronize();

        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal),
            hipSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = append_stage.execute(nullptr);
        }
        ASSERT_EQ(hipStreamEndCapture(stream.stream(), &graph), hipSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream.stream()), hipSuccess);
        stream.synchronize();

        constexpr std::array<int32_t, batch_size> continuation_lengths{1, 1};
        ASSERT_EQ(
            hipMemcpyAsync(
                device_lengths,
                continuation_lengths.data(),
                batch_size * sizeof(int32_t),
                hipMemcpyHostToDevice,
                stream.stream()),
            hipSuccess);
        append_stage.updateDynamicParams(
            /*pos_offset=*/captured_rows,
            /*seq_len=*/captured_rows);
        ASSERT_EQ(hipGraphLaunch(graph_exec, stream.stream()), hipSuccess);
        stream.synchronize();
        auto *k_base = static_cast<float *>(k_tensor->gpu_data_ptr());
        auto *v_base = static_cast<float *>(v_tensor->gpu_data_ptr());
        ASSERT_NE(k_base, nullptr);
        ASSERT_NE(v_base, nullptr);
        for (int request = 0; request < batch_size; ++request)
        {
            const size_t request_offset =
                static_cast<size_t>(request) * captured_rows * kv_dim;
            GpuTensorView initial_k(
                k_base + request_offset,
                initial_counts[request], kv_dim,
                TensorType::FP32, /*device_id=*/0);
            GpuTensorView initial_v(
                v_base + request_offset,
                initial_counts[request], kv_dim,
                TensorType::FP32, /*device_id=*/0);
            ASSERT_TRUE(appendWithTestStream(
                reference, 0, request,
                &initial_k, &initial_v,
                initial_counts[request], stream));

            GpuTensorView continuation_k(
                k_base + request_offset,
                /*rows=*/1, kv_dim,
                TensorType::FP32, /*device_id=*/0);
            GpuTensorView continuation_v(
                v_base + request_offset,
                /*rows=*/1, kv_dim,
                TensorType::FP32, /*device_id=*/0);
            ASSERT_TRUE(appendWithTestStream(
                reference, 0, request,
                &continuation_k, &continuation_v,
                /*num_tokens=*/1, stream));
        }
        stream.synchronize();

        IKVCache::KVReadParams read_params;
        read_params.gpu_stream = stream.opaque();
        for (int request = 0; request < batch_size; ++request)
        {
            read_params.requested_token_count = final_counts[request];
            SCOPED_TRACE("request=" + std::to_string(request));
            EXPECT_EQ(actual.get_cached_tokens(0, request), final_counts[request]);
            EXPECT_EQ(actual.ring_head(0, request), final_counts[request]);
            EXPECT_EQ(reference.get_cached_tokens(0, request), final_counts[request]);

            int device_count = -1;
            int device_head = -1;
            ASSERT_EQ(
                hipMemcpyAsync(
                    &device_count,
                    actual.deviceCachedTokenCountPtr(0, request),
                    sizeof(int), hipMemcpyDeviceToHost, stream.stream()),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpyAsync(
                    &device_head,
                    actual.deviceRingHeadPtr(0, request),
                    sizeof(int), hipMemcpyDeviceToHost, stream.stream()),
                hipSuccess);
            stream.synchronize();
            EXPECT_EQ(device_count, final_counts[request]);
            EXPECT_EQ(device_head, final_counts[request]);

            ITensor *actual_k = nullptr;
            ITensor *actual_v = nullptr;
            int actual_rows = 0;
            ASSERT_TRUE(actual.get_kv_converted(
                0, request, ActivationPrecision::FP16,
                &actual_k, &actual_v, &actual_rows, &read_params));
            ASSERT_EQ(actual_rows, final_counts[request]);
            stream.synchronize();
            const size_t bytes =
                static_cast<size_t>(actual_rows) * kv_dim * sizeof(uint16_t);
            std::vector<uint8_t> actual_k_bytes(bytes);
            std::vector<uint8_t> actual_v_bytes(bytes);
            ASSERT_EQ(
                hipMemcpy(
                    actual_k_bytes.data(), actual_k->gpu_data_ptr(),
                    bytes, hipMemcpyDeviceToHost),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(
                    actual_v_bytes.data(), actual_v->gpu_data_ptr(),
                    bytes, hipMemcpyDeviceToHost),
                hipSuccess);

            ITensor *reference_k = nullptr;
            ITensor *reference_v = nullptr;
            int reference_rows = 0;
            ASSERT_TRUE(reference.get_kv_converted(
                0, request, ActivationPrecision::FP16,
                &reference_k, &reference_v, &reference_rows, &read_params));
            ASSERT_EQ(reference_rows, final_counts[request]);
            stream.synchronize();
            std::vector<uint8_t> reference_k_bytes(bytes);
            std::vector<uint8_t> reference_v_bytes(bytes);
            ASSERT_EQ(
                hipMemcpy(
                    reference_k_bytes.data(), reference_k->gpu_data_ptr(),
                    bytes, hipMemcpyDeviceToHost),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(
                    reference_v_bytes.data(), reference_v->gpu_data_ptr(),
                    bytes, hipMemcpyDeviceToHost),
                hipSuccess);

            EXPECT_EQ(
                std::memcmp(
                    actual_k_bytes.data(), reference_k_bytes.data(), bytes),
                0)
                << "captured TQ8 K continuation changed live bytes";
            EXPECT_EQ(
                std::memcmp(
                    actual_v_bytes.data(), reference_v_bytes.data(), bytes),
                0)
                << "captured TQ4 V continuation changed live bytes";
        }

        ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
        ASSERT_EQ(hipFree(device_lengths), hipSuccess);
    }
}

/**
 * @brief Proves captured grouped dequant follows post-append device state.
 *
 * Four rows initialize the compressed cache. A graph captures one append plus
 * the production request-batched TQ materialization and is launched twice.
 * The read kernel must consume the HIP-resident head/count advanced by the
 * preceding append node on each replay. The six-row result is compared
 * byte-for-byte with two serial appends for both supported TQ dimensions.
 */
TEST(Test__ROCmRingKVCacheTQ, CapturedGroupedDequantReadsPostAppendDeviceState)
{
    if (!hasROCm())
        GTEST_SKIP() << "ROCm not available";

    constexpr int history_rows = 4;
    constexpr int final_rows = history_rows + 2;
    constexpr int max_seq_len = 8;
    constexpr int n_kv_heads = 2;
    constexpr float rope_theta = 10000.0f;
    constexpr int position_start = 3;

    for (const int head_dim : {64, 128})
    {
        SCOPED_TRACE("head_dim=" + std::to_string(head_dim));
        const int kv_dim = n_kv_heads * head_dim;
        auto history_k_values = generateRandomFP32(
            static_cast<size_t>(history_rows) * kv_dim, 2701 + head_dim);
        auto history_v_values = generateRandomFP32(
            static_cast<size_t>(history_rows) * kv_dim, 2907 + head_dim);
        auto continuation_k_values = generateRandomFP32(kv_dim, 3109 + head_dim);
        auto continuation_v_values = generateRandomFP32(kv_dim, 3301 + head_dim);
        auto history_k = createFP32Tensor(history_k_values, history_rows, kv_dim);
        auto history_v = createFP32Tensor(history_v_values, history_rows, kv_dim);
        auto continuation_k = createFP32Tensor(continuation_k_values, 1, kv_dim);
        auto continuation_v = createFP32Tensor(continuation_v_values, 1, kv_dim);
        ASSERT_NE(history_k, nullptr);
        ASSERT_NE(history_v, nullptr);
        ASSERT_NE(continuation_k, nullptr);
        ASSERT_NE(continuation_v, nullptr);

        TurboQuantContext tq_ctx(head_dim, 42);
        ScopedHipStream stream;
        ROCmRingKVCacheTQ actual(
            /*n_layers=*/1, /*batch_size=*/1, max_seq_len,
            n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0);
        ROCmRingKVCacheTQ reference(
            /*n_layers=*/1, /*batch_size=*/1, max_seq_len,
            n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0);
        ASSERT_TRUE(history_k->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));
        ASSERT_TRUE(history_v->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));
        ASSERT_TRUE(continuation_k->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));
        ASSERT_TRUE(continuation_v->ensureOnDevice(DeviceId::rocm(0), stream.opaque()));
        ASSERT_TRUE(appendWithTestStream(
            actual, 0, 0, history_k.get(), history_v.get(), history_rows, stream));
        ASSERT_TRUE(appendWithTestStream(
            reference, 0, 0, history_k.get(), history_v.get(), history_rows, stream));
        stream.synchronize();

        IKVCache::KVReadParams read_params;
        read_params.rope_theta = rope_theta;
        read_params.position_start = position_start;
        read_params.n_kv_heads = n_kv_heads;
        read_params.head_dim = head_dim;
        read_params.rope_dim = head_dim;
        read_params.turboquant_ctx = &tq_ctx;
        read_params.gpu_stream = stream.opaque();

        KVCacheAppendStage append_stage({
            .device_id = DeviceId::rocm(0),
            .K = continuation_k.get(),
            .V = continuation_v.get(),
            .kv_cache = &actual,
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = 1,
            .batch_size = 1,
            .seq_len = 1,
            .head_dim = head_dim,
            .turboquant_ctx = &tq_ctx,
        });
        append_stage.setGPUStream(stream.opaque());
        append_stage.updateDynamicDevicePositionIds(
            actual.deviceCachedTokenCountPtr(0, 0), /*seq_len=*/1);
        append_stage.updateDynamicParams(/*pos_offset=*/history_rows, /*seq_len=*/1);
        stream.synchronize();

        ITensor *captured_k = nullptr;
        ITensor *captured_v = nullptr;
        hipGraph_t graph = nullptr;
        hipGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            hipStreamBeginCapture(stream.stream(), hipStreamCaptureModeGlobal),
            hipSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = append_stage.execute(nullptr) &&
                         actual.get_kv_batched_converted_device_view(
                             0, 0, /*request_count=*/1,
                             ActivationPrecision::FP16,
                             &captured_k, &captured_v, read_params);
        }
        ASSERT_EQ(hipStreamEndCapture(stream.stream(), &graph), hipSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(captured_k, nullptr);
        ASSERT_NE(captured_v, nullptr);
        ASSERT_EQ(captured_k->rows(), static_cast<size_t>(max_seq_len));
        ASSERT_EQ(captured_v->rows(), static_cast<size_t>(max_seq_len));
        ASSERT_EQ(
            hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            hipSuccess);

        ASSERT_EQ(hipGraphLaunch(graph_exec, stream.stream()), hipSuccess);
        stream.synchronize();
        EXPECT_EQ(actual.get_cached_tokens(0, 0), history_rows + 1);

        ASSERT_EQ(hipGraphLaunch(graph_exec, stream.stream()), hipSuccess);
        stream.synchronize();
        EXPECT_EQ(actual.get_cached_tokens(0, 0), final_rows);
        int device_count = -1;
        int device_head = -1;
        ASSERT_EQ(
            hipMemcpy(
                &device_count, actual.deviceCachedTokenCountPtr(0, 0),
                sizeof(int), hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                &device_head, actual.deviceRingHeadPtr(0, 0),
                sizeof(int), hipMemcpyDeviceToHost),
            hipSuccess);
        EXPECT_EQ(device_count, final_rows);
        EXPECT_EQ(device_head, final_rows);

        ASSERT_TRUE(appendWithTestStream(
            reference, 0, 0,
            continuation_k.get(), continuation_v.get(), 1, stream));
        ITensor *reference_k = nullptr;
        ITensor *reference_v = nullptr;
        int reference_rows = 0;
        ASSERT_TRUE(reference.get_kv_converted(
            0, 0, ActivationPrecision::FP16,
            &reference_k, &reference_v, &reference_rows, &read_params));
        ASSERT_EQ(reference_rows, history_rows + 1);
        stream.synchronize();
        ASSERT_TRUE(appendWithTestStream(
            reference, 0, 0,
            continuation_k.get(), continuation_v.get(), 1, stream));
        ASSERT_TRUE(reference.get_kv_converted(
            0, 0, ActivationPrecision::FP16,
            &reference_k, &reference_v, &reference_rows, &read_params));
        ASSERT_EQ(reference_rows, final_rows);
        stream.synchronize();

        const size_t bytes =
            static_cast<size_t>(final_rows) * kv_dim * sizeof(uint16_t);
        std::vector<uint8_t> actual_k_bytes(bytes);
        std::vector<uint8_t> actual_v_bytes(bytes);
        std::vector<uint8_t> reference_k_bytes(bytes);
        std::vector<uint8_t> reference_v_bytes(bytes);
        ASSERT_EQ(
            hipMemcpy(
                actual_k_bytes.data(), captured_k->gpu_data_ptr(),
                bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                actual_v_bytes.data(), captured_v->gpu_data_ptr(),
                bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                reference_k_bytes.data(), reference_k->gpu_data_ptr(),
                bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                reference_v_bytes.data(), reference_v->gpu_data_ptr(),
                bytes, hipMemcpyDeviceToHost),
            hipSuccess);
        EXPECT_EQ(
            std::memcmp(actual_k_bytes.data(), reference_k_bytes.data(), bytes),
            0)
            << "captured device-owned TQ8 K dequant differs from exact continuation";
        EXPECT_EQ(
            std::memcmp(actual_v_bytes.data(), reference_v_bytes.data(), bytes),
            0)
            << "captured device-owned TQ4 V dequant differs from exact continuation";

        EXPECT_EQ(actual.get_cached_tokens(0, 0), final_rows);
        EXPECT_EQ(actual.ring_head(0, 0), final_rows);
        ASSERT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
        ASSERT_EQ(hipGraphDestroy(graph), hipSuccess);
    }
}

#endif // HAVE_ROCM

// =============================================================================
// No ROCm Fallback
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, NoROCm_Skipped)
{
#ifndef HAVE_ROCM
    SUCCEED() << "HAVE_ROCM not defined, compile-time guard working";
#else
    if (!hasROCm())
        GTEST_SKIP() << "ROCm compiled but no device";
#endif
}
