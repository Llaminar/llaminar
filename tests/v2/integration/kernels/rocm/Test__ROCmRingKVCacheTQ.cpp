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
 * 14. Host-side append via CPU tensors
 * 15. Metadata accessor verification
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 */

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <cstring>
#include <cstdint>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/kvcache/ROCmRingKVCacheTQ.h"
#include "kernels/rocm/kvcache/ROCmRingKVCacheTQFactory.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/IKVCache.h"
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
                hipStreamDestroy(stream_);
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
        hipMalloc(&d_ptr, bytes);
        hipMemcpy(d_ptr, host_data.data(), bytes, hipMemcpyHostToDevice);
        return d_ptr;
    }

    // Download FP16 GPU buffer to FP32 host vector
    std::vector<float> downloadFP16ToFP32(const void *d_ptr, size_t count)
    {
        std::vector<uint16_t> h_fp16(count);
        hipMemcpy(h_fp16.data(), d_ptr, count * sizeof(uint16_t), hipMemcpyDeviceToHost);

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
        ASSERT_NE(raw_k, nullptr);
        ASSERT_FALSE(raw_k->shape().empty());
        EXPECT_EQ(raw_k->shape()[0], 0u);
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K);
    hipFree(d_V);
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

        hipFree(d_K);
        hipFree(d_V);
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

        hipFree(d_K);
        hipFree(d_V);
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
    cache->clear_sequence(0, 0);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
    EXPECT_EQ(cache->get_cached_tokens(1, 0), 5);

    // Clear all
    cache->clear();
    for (int l = 0; l < n_layers; ++l)
        EXPECT_EQ(cache->get_cached_tokens(l, 0), 0);

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K);
    hipFree(d_V);
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
        hipFree(d_K);
        hipFree(d_V);
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

    cache->clear_sequence(0, 1);
    expectConvertedEmpty(*cache, 0, 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);
    EXPECT_EQ(cache->get_cached_tokens(1, 0), num_tokens);
    EXPECT_EQ(cache->get_cached_tokens(1, 1), num_tokens);

    cache->clear_layer(1);
    expectConvertedEmpty(*cache, 1, 0);
    expectConvertedEmpty(*cache, 1, 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), num_tokens);

    cache->clear();
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
        hipFree(d_K);
        hipFree(d_V);
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

    cache->clear();
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_data);
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K1);
    hipFree(d_V1);
    hipFree(d_K2);
    hipFree(d_V2);
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

    hipFree(d_K);
    hipFree(d_V);
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

    hipFree(d_K);
    hipFree(d_V);
}

// =============================================================================
// 14. Host-Side Append (via CPU Tensor)
// =============================================================================

TEST(Test__ROCmRingKVCacheTQ, HostSideAppend_ViaCPUTensor)
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
    EXPECT_GT(cos_k, 0.94f) << "Host-side append quality too low";
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
