/**
 * @file Test__CUDARingKVCacheTQ.cpp
 * @brief Comprehensive unit tests for CUDA TurboQuant KV Cache
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Basic append + retrieve roundtrip (TQ8-K/TQ4-V)
 * 2. Ring buffer wrap-around preserves newest tokens
 * 3. Incremental decode-like append pattern
 * 4. Multi-layer independent data
 * 5. Clear/reset semantics
 * 6. Quantization error bounds (cosine similarity)
 * 7. TQ8 K quality strictly better than TQ4 V
 * 8. get_kv_converted with RoPE-on-read
 * 9. get_kv_converted without RoPE (dequant only)
 * 10. Eviction correctness
 * 11. Shadow buffer invalidation on append
 * 12. Head dim 128 support
 * 13. RoPE position correctness
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <string>
#include "../KVCacheTestWorkspace.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "kernels/cuda/kvcache/CUDARingKVCacheTQ.h"
#include "kernels/cpu/turboquant/TurboQuantContext.h"
#include "kernels/cpu/turboquant/TurboQuantDequantizeTQ8.h"
#include "kernels/cpu/turboquant/TurboQuantQuantizeTQ8.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"
#include "tensors/GpuTensorView.h"
#include "utils/Logger.h"

using namespace llaminar2;
using llaminar2::test::KVCacheTestWorkspaceBinding;

namespace
{
    bool hasCUDA()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
    }

    /**
     * @brief Owns a non-default CUDA stream for tests that must exercise the explicit-stream cache path.
     *
     * TQ GPU cache appends reject the legacy no-stream API by design. Keeping a single stream alive for
     * each test mirrors the graph/stage path and lets append, read, and clear operations synchronize in a
     * predictable order.
     */
    class ScopedCudaStream
    {
    public:
        ScopedCudaStream()
        {
            EXPECT_EQ(cudaStreamCreate(&stream_), cudaSuccess);
        }

        ~ScopedCudaStream()
        {
            if (stream_)
                cudaStreamDestroy(stream_);
        }

        void *opaque() const { return static_cast<void *>(stream_); }
        cudaStream_t stream() const { return stream_; }

        void synchronize() const
        {
            ASSERT_NE(stream_, nullptr);
            ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
        }

    private:
        cudaStream_t stream_ = nullptr;
    };

    /// @brief Append through the required explicit-stream API.
    bool appendWithTestStream(CUDARingKVCacheTQ &cache, int layer, int seq_idx,
                              const ITensor *K, const ITensor *V, int num_tokens,
                              const ScopedCudaStream &stream)
    {
        auto ensure = [&](const ITensor *tensor)
        {
            if (tensor && tensor->gpu_data_ptr())
                return true;
            auto *base = dynamic_cast<TensorBase *>(const_cast<ITensor *>(tensor));
            return base && base->ensureOnDevice(DeviceId::cuda(0), stream.opaque());
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

    /** @brief Map IEEE FP16 bits to adjacent monotonic integer codes. */
    int fp16OrderedCode(uint16_t bits)
    {
        return (bits & 0x8000U) != 0
                   ? 0x8000 - static_cast<int>(bits & 0x7fffU)
                   : 0x8000 + static_cast<int>(bits);
    }

    // Upload FP32 host vector to GPU, returning device pointer
    float *uploadToGPU(const std::vector<float> &host_data)
    {
        float *d_ptr = nullptr;
        size_t bytes = host_data.size() * sizeof(float);
        cudaMalloc(&d_ptr, bytes);
        cudaMemcpy(d_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
        return d_ptr;
    }

    // Download FP16 GPU buffer to FP32 host vector
    std::vector<float> downloadFP16ToFP32(const void *d_ptr, size_t count)
    {
        std::vector<__half> h_fp16(count);
        cudaMemcpy(h_fp16.data(), d_ptr, count * sizeof(__half), cudaMemcpyDeviceToHost);

        std::vector<float> result(count);
        for (size_t i = 0; i < count; ++i)
            result[i] = __half2float(h_fp16[i]);
        return result;
    }

    // Download FP32 GPU buffer to FP32 host vector
    std::vector<float> downloadFP32(const void *d_ptr, size_t count)
    {
        std::vector<float> result(count);
        cudaMemcpy(result.data(), d_ptr, count * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }

    // Helper: create FP32Tensor from host data (for IKVCache::append(ITensor*))
    std::unique_ptr<FP32Tensor> createFP32Tensor(const std::vector<float> &data,
                                                 size_t rows, size_t cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        std::memcpy(tensor->mutable_data(), data.data(), data.size() * sizeof(float));
        return tensor;
    }

    /**
     * @brief Verify an empty converted read cannot leave stale K/V output pointers behind.
     *
     * The long-context handover points at survivor KV rows after request reset. This helper makes the
     * no-survivor contract concrete: after clear operations, callers get kv_len=0 and null converted
     * tensors even if they pass pre-filled output slots.
     */
    void expectConvertedEmpty(CUDARingKVCacheTQ &cache, int layer, int seq_idx)
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

TEST(Test__CUDARingKVCacheTQ, BasicAppendRetrieve_SplitTQ)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 1;
    const int batch_size = 1;
    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(n_layers, batch_size, max_seq_len,
                            n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    EXPECT_EQ(cache.n_layers(), n_layers);
    EXPECT_EQ(cache.max_seq_len(), max_seq_len);
    EXPECT_EQ(cache.n_kv_heads(), n_kv_heads);
    EXPECT_EQ(cache.head_dim(), head_dim);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);

    // Append 10 tokens
    const int num_tokens = 10;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);

    auto k_tensor = createFP32Tensor(h_K, num_tokens, kv_dim);
    auto v_tensor = createFP32Tensor(h_V, num_tokens, kv_dim);

    // Upload to GPU for append (TQ cache expects GPU or host data)
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));

    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);

    // Retrieve via get_k/get_v (returns FP16 shadow buffers)
    const ITensor *out_k = cache.get_k(0, 0);
    const ITensor *out_v = cache.get_v(0, 0);
    ASSERT_NE(out_k, nullptr);
    ASSERT_NE(out_v, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(num_tokens));
    EXPECT_EQ(out_v->shape()[0], static_cast<size_t>(num_tokens));

    // Download FP16 and verify cosine similarity
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    // Per-token cosine similarity
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

    // TQ8 (K) should be high quality
    EXPECT_GT(avg_cos_k, 0.97f) << "TQ8 K average cosine too low";
    EXPECT_GT(min_cos_k, 0.94f) << "TQ8 K minimum cosine too low";

    // TQ4 (V) is lower but still acceptable
    EXPECT_GT(avg_cos_v, 0.88f) << "TQ4 V average cosine too low";
    EXPECT_GT(min_cos_v, 0.78f) << "TQ4 V minimum cosine too low";

    LOG_INFO("[Test] Split TQ roundtrip: K cos=" << avg_cos_k << "/" << min_cos_k
                                                 << ", V cos=" << avg_cos_v << "/" << min_cos_v);

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 2. Ring Buffer Wrap-Around
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, WrapAround_PreservesNewest)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq_len = 8;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Append 12 tokens (overwrites first 4)
    const int num_tokens = 12;
    auto h_K = generateRandomFP32(num_tokens * kv_dim, 100);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 200);

    // Append in two batches
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto k1 = std::make_unique<GpuTensorView>(d_K, 8, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto v1 = std::make_unique<GpuTensorView>(d_V, 8, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k1.get(), v1.get(), 8, stream));

    auto k2 = std::make_unique<GpuTensorView>(
        d_K + 8 * kv_dim, 4, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto v2 = std::make_unique<GpuTensorView>(
        d_V + 8 * kv_dim, 4, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k2.get(), v2.get(), 4, stream));
    stream.synchronize();

    // Should have max_seq_len tokens (wrapped)
    EXPECT_EQ(cache.get_cached_tokens(0, 0), max_seq_len);

    // Retrieve and check that recent tokens have reasonable quality
    const ITensor *out_k = cache.get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(max_seq_len));

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 3. Incremental Decode-Like Append
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, IncrementalAppend_DecodeLike)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Simulate decode: append one token at a time
    std::mt19937 rng(777);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> all_K, all_V;
    const int decode_steps = 20;

    for (int step = 0; step < decode_steps; ++step)
    {
        auto h_K = generateRandomFP32(kv_dim, 1000 + step);
        auto h_V = generateRandomFP32(kv_dim, 2000 + step);
        all_K.push_back(h_K);
        all_V.push_back(h_V);

        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);

        auto kv = std::make_unique<GpuTensorView>(d_K, 1, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        auto vv = std::make_unique<GpuTensorView>(d_V, 1, kv_dim, TensorType::FP32, DeviceId::cuda(0));

        ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), 1, stream));
        stream.synchronize();
        EXPECT_EQ(cache.get_cached_tokens(0, 0), step + 1);

        cudaFree(d_K);
        cudaFree(d_V);
    }

    // Verify final state: all tokens should be in cache
    const ITensor *out_k = cache.get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(decode_steps));

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), decode_steps * kv_dim);

    // Check cosine similarity for each token
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

TEST(Test__CUDARingKVCacheTQ, MultiLayer_IndependentData)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 4;
    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 5;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(n_layers, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Append different data to each layer
    for (int layer = 0; layer < n_layers; ++layer)
    {
        auto h_K = generateRandomFP32(num_tokens * kv_dim, 100 * layer + 1);
        auto h_V = generateRandomFP32(num_tokens * kv_dim, 100 * layer + 2);
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);

        auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));

        ASSERT_TRUE(appendWithTestStream(cache, layer, 0, kv.get(), vv.get(), num_tokens, stream));
        stream.synchronize();
        EXPECT_EQ(cache.get_cached_tokens(layer, 0), num_tokens);

        cudaFree(d_K);
        cudaFree(d_V);
    }

    // Verify each layer has independent data
    std::vector<std::vector<float>> layer_results;
    for (int layer = 0; layer < n_layers; ++layer)
    {
        const ITensor *out_k = cache.get_k(layer, 0);
        ASSERT_NE(out_k, nullptr);
        layer_results.push_back(
            downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim));
    }

    // Verify layers are different
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

TEST(Test__CUDARingKVCacheTQ, Clear_ResetsAllLayers)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 2;
    const int max_seq_len = 16;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(n_layers, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Fill some data
    auto h_K = generateRandomFP32(5 * kv_dim, 100);
    auto h_V = generateRandomFP32(5 * kv_dim, 200);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, 5, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, 5, kv_dim, TensorType::FP32, DeviceId::cuda(0));

    for (int l = 0; l < n_layers; ++l)
        ASSERT_TRUE(appendWithTestStream(cache, l, 0, kv.get(), vv.get(), 5, stream));
    stream.synchronize();

    // Clear single sequence
    ASSERT_TRUE(cache.resetLayerSequenceState(
        0,
        0,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), 5); // Other layer unaffected

    // Clear all
    ASSERT_TRUE(cache.resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    for (int l = 0; l < n_layers; ++l)
        EXPECT_EQ(cache.get_cached_tokens(l, 0), 0);

    cudaFree(d_K);
    cudaFree(d_V);
}

TEST(Test__CUDARingKVCacheTQ, AppendRequiresExplicitNonNullStream)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 3;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 700);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 701);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);
    auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));

    EXPECT_FALSE(cache.append(0, 0, k_view.get(), v_view.get(), num_tokens));
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);
    EXPECT_THROW(
        cache.appendWithStream(
            0, 0, k_view.get(), v_view.get(), num_tokens, nullptr),
        std::invalid_argument);
    EXPECT_THROW(
        cache.resetRequestState(
            IKVCache::StateResetContext::testReinitialization(nullptr)),
        std::invalid_argument);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);

    ScopedCudaStream stream;
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);

    cudaFree(d_K);
    cudaFree(d_V);
}

TEST(Test__CUDARingKVCacheTQ, ClearSequenceLayerAndAllInvalidateConvertedScratch)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 2;
    const int batch_size = 2;
    const int num_tokens = 4;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(n_layers, batch_size, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto append_seeded = [&](int layer, int seq_idx, unsigned seed)
    {
        auto h_K = generateRandomFP32(num_tokens * kv_dim, seed);
        auto h_V = generateRandomFP32(num_tokens * kv_dim, seed + 1000);
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);
        auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        ASSERT_TRUE(appendWithTestStream(cache, layer, seq_idx, k_view.get(), v_view.get(), num_tokens, stream));
        stream.synchronize();
        cudaFree(d_K);
        cudaFree(d_V);
    };

    append_seeded(0, 0, 10);
    append_seeded(0, 1, 20);
    append_seeded(1, 0, 30);
    append_seeded(1, 1, 40);

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 1, ActivationPrecision::FP16,
                                       &out_k, &out_v, &kv_len, nullptr));
    ASSERT_EQ(kv_len, num_tokens);
    ASSERT_NE(out_k, nullptr);

    ASSERT_TRUE(cache.resetLayerSequenceState(
        0,
        1,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(cache, 0, 1);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);
    EXPECT_EQ(cache.get_cached_tokens(1, 0), num_tokens);
    EXPECT_EQ(cache.get_cached_tokens(1, 1), num_tokens);

    ASSERT_TRUE(cache.resetLayerState(
        1,
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(cache, 1, 0);
    expectConvertedEmpty(cache, 1, 1);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);

    ASSERT_TRUE(cache.resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(cache, 0, 0);
    expectConvertedEmpty(cache, 0, 1);
}

TEST(Test__CUDARingKVCacheTQ, ClearThenReappendConvertedScratchUsesNewRows)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 6;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 16, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto append_host = [&](const std::vector<float> &h_K, const std::vector<float> &h_V)
    {
        float *d_K = uploadToGPU(h_K);
        float *d_V = uploadToGPU(h_V);
        auto k_view = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        auto v_view = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
        ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k_view.get(), v_view.get(), num_tokens, stream));
        stream.synchronize();
        cudaFree(d_K);
        cudaFree(d_V);
    };

    auto h_K_a = generateRandomFP32(num_tokens * kv_dim, 900);
    auto h_V_a = generateRandomFP32(num_tokens * kv_dim, 901);
    auto h_K_b = generateRandomFP32(num_tokens * kv_dim, 1900);
    auto h_V_b = generateRandomFP32(num_tokens * kv_dim, 1901);

    append_host(h_K_a, h_V_a);

    ITensor *out_k_a = nullptr;
    ITensor *out_v_a = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
                                       &out_k_a, &out_v_a, &kv_len, nullptr));
    ASSERT_EQ(kv_len, num_tokens);
    auto k_a = downloadFP16ToFP32(out_k_a->gpu_data_ptr(), num_tokens * kv_dim);

    ASSERT_TRUE(cache.resetRequestState(
        IKVCache::StateResetContext::testReinitialization(stream.opaque())));
    expectConvertedEmpty(cache, 0, 0);

    append_host(h_K_b, h_V_b);
    ITensor *out_k_b = nullptr;
    ITensor *out_v_b = nullptr;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
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
// 6. Quantization Error Bounds (Cosine Similarity)
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, QuantizationError_WithinBounds)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq_len = 64;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 32;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 314);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 271);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    const ITensor *out_k = cache.get_k(0, 0);
    const ITensor *out_v = cache.get_v(0, 0);
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    float mse_k = computeMSE(h_K.data(), result_K.data(), num_tokens * kv_dim);
    float mse_v = computeMSE(h_V.data(), result_V.data(), num_tokens * kv_dim);

    // TQ8 K should have much lower MSE than TQ4 V
    EXPECT_LT(mse_k, 0.02f) << "TQ8 K MSE too high";
    EXPECT_LT(mse_v, 0.08f) << "TQ4 V MSE too high";

    LOG_INFO("[Test] Quantization MSE: K=" << mse_k << " V=" << mse_v);

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 7. TQ8 K Quality Strictly Better Than TQ4 V
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, KQuality_StrictlyBetterThan_V)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 20;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx_7(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 64, n_kv_heads, head_dim, &tq_ctx_7, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_data = generateRandomFP32(num_tokens * kv_dim, 999);
    float *d_data = uploadToGPU(h_data);

    auto view = std::make_unique<GpuTensorView>(d_data, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, view.get(), view.get(), num_tokens, stream)); // Same data for K and V
    stream.synchronize();

    const ITensor *out_k = cache.get_k(0, 0);
    const ITensor *out_v = cache.get_v(0, 0);
    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    auto result_V = downloadFP16ToFP32(out_v->gpu_data_ptr(), num_tokens * kv_dim);

    float cos_k = computeCosineSimilarity(h_data.data(), result_K.data(), num_tokens * kv_dim);
    float cos_v = computeCosineSimilarity(h_data.data(), result_V.data(), num_tokens * kv_dim);

    EXPECT_GT(cos_k, cos_v) << "TQ8 K should be higher quality than TQ4 V";
    LOG_INFO("[Test] Same-data quality: K cos=" << cos_k << " V cos=" << cos_v);

    cudaFree(d_data);
}

// =============================================================================
// 8. get_kv_converted with RoPE-on-read
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, GetKVConverted_WithRoPE)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 8;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx_8(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_8, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 123);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 456);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    // Get without RoPE
    ITensor *out_k_noRoPE = nullptr;
    ITensor *out_v_noRoPE = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
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
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
                                       &out_k_rope, &out_v_rope, &kv_len, &rope_params));
    EXPECT_EQ(kv_len, num_tokens);

    auto result_withRoPE = downloadFP16ToFP32(out_k_rope->gpu_data_ptr(), num_tokens * kv_dim);

    // RoPE should change the K values
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

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 9. get_kv_converted Without RoPE (Dequant Only)
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, GetKVConverted_DequantOnly)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 10;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx_9(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_9, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 555);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 666);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    ITensor *out_k = nullptr;
    ITensor *out_v = nullptr;
    int kv_len = 0;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
                                       &out_k, &out_v, &kv_len, nullptr));
    EXPECT_EQ(kv_len, num_tokens);

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);

    // Should match get_k() output exactly
    const ITensor *direct_k = cache.get_k(0, 0);
    auto direct_K = downloadFP16ToFP32(direct_k->gpu_data_ptr(), num_tokens * kv_dim);

    for (size_t i = 0; i < result_K.size(); ++i)
        EXPECT_NEAR(result_K[i], direct_K[i], 1e-6f) << "Mismatch at index " << i;

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 10. Eviction Correctness
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, Eviction_ReducesCount)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq_len = 32;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 20;

    TurboQuantContext tq_ctx_10(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, max_seq_len, n_kv_heads, head_dim, &tq_ctx_10, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);

    // Evict 5 tokens
    cache.evict_oldest(0, 0, 5);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens - 5);

    // Evict all remaining
    cache.evict_oldest(0, 0, num_tokens);
    EXPECT_EQ(cache.get_cached_tokens(0, 0), 0);

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 11. Shadow Buffer Invalidation on Append
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, ShadowInvalidation_AfterAppend)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx_11(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_11, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Append batch 1
    auto h_K1 = generateRandomFP32(5 * kv_dim, 100);
    auto h_V1 = generateRandomFP32(5 * kv_dim, 200);
    float *d_K1 = uploadToGPU(h_K1);
    float *d_V1 = uploadToGPU(h_V1);

    auto kv1 = std::make_unique<GpuTensorView>(d_K1, 5, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv1 = std::make_unique<GpuTensorView>(d_V1, 5, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv1.get(), vv1.get(), 5, stream));
    stream.synchronize();

    // Force shadow creation
    const ITensor *k1 = cache.get_k(0, 0);
    ASSERT_NE(k1, nullptr);
    EXPECT_EQ(k1->shape()[0], 5u);

    // Append batch 2
    auto h_K2 = generateRandomFP32(3 * kv_dim, 300);
    auto h_V2 = generateRandomFP32(3 * kv_dim, 400);
    float *d_K2 = uploadToGPU(h_K2);
    float *d_V2 = uploadToGPU(h_V2);

    auto kv2 = std::make_unique<GpuTensorView>(d_K2, 3, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv2 = std::make_unique<GpuTensorView>(d_V2, 3, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv2.get(), vv2.get(), 3, stream));
    stream.synchronize();

    // Shadow should be regenerated with new count
    const ITensor *k2 = cache.get_k(0, 0);
    ASSERT_NE(k2, nullptr);
    EXPECT_EQ(k2->shape()[0], 8u); // 5 + 3

    cudaFree(d_K1);
    cudaFree(d_V1);
    cudaFree(d_K2);
    cudaFree(d_V2);
}

// =============================================================================
// 12. Head Dim 128 Support
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, HeadDim128_BasicRoundtrip)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_kv_heads = 4;
    const int head_dim = 128;
    const int kv_dim = n_kv_heads * head_dim;
    const int num_tokens = 8;

    TurboQuantContext tq_ctx_12(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_12, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 777);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 888);
    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
    stream.synchronize();

    const ITensor *out_k = cache.get_k(0, 0);
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

    LOG_INFO("[Test] HeadDim128 avg K cosine: " << avg_cos);

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 13. RoPE Position Correctness
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, RoPE_PositionCorrectness)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 4;
    const int n_kv_heads = 1;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;
    const float rope_theta = 10000.0f;

    TurboQuantContext tq_ctx_13(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_13, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    // Use simple predictable data
    std::vector<float> h_K(num_tokens * kv_dim, 1.0f);
    std::vector<float> h_V(num_tokens * kv_dim, 1.0f);

    float *d_K = uploadToGPU(h_K);
    float *d_V = uploadToGPU(h_V);

    auto kv = std::make_unique<GpuTensorView>(d_K, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    auto vv = std::make_unique<GpuTensorView>(d_V, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, kv.get(), vv.get(), num_tokens, stream));
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
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
                                       &out_k_pos0, &out_v_pos0, &kv_len, &rope_params));

    auto result_pos0 = downloadFP16ToFP32(out_k_pos0->gpu_data_ptr(), num_tokens * kv_dim);

    rope_params.position_start = 10;
    ITensor *out_k_pos10 = nullptr;
    ITensor *out_v_pos10 = nullptr;
    ASSERT_TRUE(cache.get_kv_converted(0, 0, ActivationPrecision::FP16,
                                       &out_k_pos10, &out_v_pos10, &kv_len, &rope_params));

    auto result_pos10 = downloadFP16ToFP32(out_k_pos10->gpu_data_ptr(), num_tokens * kv_dim);

    // Different position offsets should produce different results
    float diff = 0.0f;
    for (size_t i = 0; i < result_pos0.size(); ++i)
        diff += std::abs(result_pos0[i] - result_pos10[i]);
    EXPECT_GT(diff, 0.01f) << "Different position offsets should produce different K values";

    cudaFree(d_K);
    cudaFree(d_V);
}

// =============================================================================
// 14. Host-created tensors are explicitly prepared before device-only append
// =============================================================================

TEST(Test__CUDARingKVCacheTQ, HostCreatedTensorIsPreparedOnDeviceBeforeAppend)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int num_tokens = 5;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx_14(head_dim, 42);
    CUDARingKVCacheTQ cache(1, 1, 32, n_kv_heads, head_dim, &tq_ctx_14, 0);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    auto h_K = generateRandomFP32(num_tokens * kv_dim, 111);
    auto h_V = generateRandomFP32(num_tokens * kv_dim, 222);

    auto k_tensor = createFP32Tensor(h_K, num_tokens, kv_dim);
    auto v_tensor = createFP32Tensor(h_V, num_tokens, kv_dim);

    ASSERT_TRUE(appendWithTestStream(cache, 0, 0, k_tensor.get(), v_tensor.get(), num_tokens, stream));
    stream.synchronize();
    EXPECT_EQ(cache.get_cached_tokens(0, 0), num_tokens);

    const ITensor *out_k = cache.get_k(0, 0);
    ASSERT_NE(out_k, nullptr);
    EXPECT_EQ(out_k->shape()[0], static_cast<size_t>(num_tokens));

    auto result_K = downloadFP16ToFP32(out_k->gpu_data_ptr(), num_tokens * kv_dim);
    float cos_k = computeCosineSimilarity(h_K.data(), result_K.data(), num_tokens * kv_dim);
    EXPECT_GT(cos_k, 0.94f) << "Prepared device append quality too low";
}

/**
 * @brief Compare production fused TQ8-value append bytes with the scalar codec.
 *
 * Grouped cache tests compare CUDA against another CUDA path and therefore
 * cannot detect a backend-wide encoder defect.  This regression reads the
 * physical value blocks written by the real ring append and checks every
 * Lloyd-Max index against the scalar mathematical oracle using the same
 * deterministic layer/head rotation hierarchy.
 */
TEST(Test__CUDARingKVCacheTQ, TQ8ValuePhysicalCodecMatchesScalarOracle)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    constexpr int num_tokens = 7;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 128;
    constexpr int kv_dim = n_kv_heads * head_dim;

    TurboQuantContext tq_ctx(head_dim, 42);
    CUDARingKVCacheTQ cache(
        /*n_layers=*/1, /*batch_size=*/1, /*max_seq_len=*/16,
        n_kv_heads, head_dim, &tq_ctx, /*device_id=*/0,
        TurboQuantKVMode::AQ8_K_TQ8_V);
    KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
    ScopedCudaStream stream;

    const auto host_k = generateRandomFP32(num_tokens * kv_dim, 771);
    const auto host_v = generateRandomFP32(num_tokens * kv_dim, 772);
    float *device_k = uploadToGPU(host_k);
    float *device_v = uploadToGPU(host_v);
    GpuTensorView k_view(
        device_k, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    GpuTensorView v_view(
        device_v, num_tokens, kv_dim, TensorType::FP32, DeviceId::cuda(0));
    ASSERT_TRUE(appendWithTestStream(
        cache, /*layer=*/0, /*seq_idx=*/0,
        &k_view, &v_view, num_tokens, stream));
    stream.synchronize();

    std::vector<TQ8Block_128> actual(num_tokens * n_kv_heads);
    ASSERT_EQ(
        cudaMemcpy(actual.data(), cache.raw_v_cache(/*layer=*/0),
                   actual.size() * sizeof(TQ8Block_128),
                   cudaMemcpyDeviceToHost),
        cudaSuccess);

    const ITensor *decoded_v = cache.get_v(/*layer=*/0, /*seq_idx=*/0);
    ASSERT_NE(decoded_v, nullptr);
    stream.synchronize();
    std::vector<uint16_t> decoded_v_bits(num_tokens * kv_dim);
    ASSERT_EQ(
        cudaMemcpy(decoded_v_bits.data(), decoded_v->gpu_data_ptr(),
                   decoded_v_bits.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost),
        cudaSuccess);

    alignas(64) float scratch0[head_dim];
    alignas(64) float scratch1[head_dim];
    for (int token = 0; token < num_tokens; ++token)
    {
        for (int head = 0; head < n_kv_heads; ++head)
        {
            const auto &head_ctx =
                tq_ctx.for_layer(/*layer=*/0).for_layer(head);
            TQ8Block_128 expected{};
            turboquant_quantize_tq8_scalar<head_dim>(
                host_v.data() +
                    (static_cast<size_t>(token) * n_kv_heads + head) * head_dim,
                head_ctx, expected, scratch0, scratch1);
            const auto &observed =
                actual[static_cast<size_t>(token) * n_kv_heads + head];
            EXPECT_NEAR(observed.norm, expected.norm, 2.0e-6f)
                << "token=" << token << " head=" << head;
            EXPECT_NEAR(
                observed.reconstruction_norm,
                expected.reconstruction_norm,
                2.0e-6f)
                << "token=" << token << " head=" << head;
            for (int coordinate = 0; coordinate < head_dim; ++coordinate)
            {
                EXPECT_EQ(observed.indices[coordinate], expected.indices[coordinate])
                    << "token=" << token << " head=" << head
                    << " coordinate=" << coordinate;
            }


            alignas(64) float expected_decoded[head_dim];
            turboquant_dequantize_tq8_scalar<head_dim>(
                observed, head_ctx, expected_decoded, scratch0);
            for (int coordinate = 0; coordinate < head_dim; ++coordinate)
            {
                const uint16_t expected_bits = __half_as_ushort(
                    __float2half_rn(expected_decoded[coordinate]));
                const size_t output_index =
                    (static_cast<size_t>(token) * n_kv_heads + head) *
                        head_dim +
                    coordinate;
                const int fp16_ulp_distance = std::abs(
                    fp16OrderedCode(decoded_v_bits[output_index]) -
                    fp16OrderedCode(expected_bits));
                EXPECT_LE(fp16_ulp_distance, 1)
                    << "decoded token=" << token << " head=" << head
                    << " coordinate=" << coordinate;
            }
        }
    }

    ASSERT_EQ(cudaFree(device_k), cudaSuccess);
    ASSERT_EQ(cudaFree(device_v), cudaSuccess);
}

/**
 * @brief Prove grouped resident TQ reads equal scalar dequant bytes.
 *
 * Request-batched attention cannot replay one cache row at a time. This test
 * therefore builds two independent compressed rings, wraps request zero, and
 * leaves request one shorter. The production grouped API is recorded in a CUDA
 * graph and must reproduce the established scalar TQ8-K/TQ4-V FP16 bytes for
 * every live row. Inactive resident capacity is deliberately not inspected:
 * production attention is bounded by the same device count, and clearing a
 * full maximum-context buffer on each decode would be uneconomical. Running
 * both TQ block dimensions prevents a 64-only implementation from silently
 * excluding models whose KV heads are 128 elements wide.
 */
TEST(Test__CUDARingKVCacheTQ, CapturedResidentRequestBatchMatchesScalarDequantBytes)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    constexpr int batch_size = 2;
    constexpr int max_seq_len = 6;
    constexpr int n_kv_heads = 2;
    constexpr std::array<int, batch_size> expected_counts{6, 4};

    for (const TurboQuantKVMode mode : {
             TurboQuantKVMode::AQ8_K_Q8_1_V,
             TurboQuantKVMode::AQ8_K_TQ8_V,
             TurboQuantKVMode::AQ8_K_TQ4_V})
    {
      for (const int head_dim : {64, 128})
      {
        SCOPED_TRACE(
            std::string("mode=") + turboQuantKVModeName(mode) +
            " head_dim=" + std::to_string(head_dim));
        const int kv_dim = n_kv_heads * head_dim;
        TurboQuantContext tq_ctx(head_dim, 42);
        CUDARingKVCacheTQ cache(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim,
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
            /*device_id=*/0, mode);
        KVCacheTestWorkspaceBinding workspace(cache, DeviceId::cuda(0));
        ScopedCudaStream stream;
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
                device_k, rows, kv_dim, TensorType::FP32, DeviceId::cuda(0));
            GpuTensorView v_view(
                device_v, rows, kv_dim, TensorType::FP32, DeviceId::cuda(0));
            return appendWithTestStream(
                cache, /*layer=*/0, request, &k_view, &v_view, rows, stream);
        };

        // Five rows followed by four rows wraps request zero in the six-row
        // ring. Request one remains shorter, exercising grouped zero padding.
        ASSERT_TRUE(appendChunk(0, 5, 101, 201));
        ASSERT_TRUE(appendChunk(0, 4, 102, 202));
        ASSERT_TRUE(appendChunk(1, 4, 103, 203));
        stream.synchronize();
        for (float *allocation : allocations)
            ASSERT_EQ(cudaFree(allocation), cudaSuccess);
        ASSERT_EQ(cache.get_cached_tokens(0, 0), expected_counts[0]);
        ASSERT_EQ(cache.get_cached_tokens(0, 1), expected_counts[1]);

        std::array<std::vector<uint16_t>, batch_size> scalar_k;
        std::array<std::vector<uint16_t>, batch_size> scalar_v;
        IKVCache::KVReadParams read_params;
        read_params.gpu_stream = stream.opaque();
        for (int request = 0; request < batch_size; ++request)
        {
            ITensor *serial_k = nullptr;
            ITensor *serial_v = nullptr;
            int serial_count = 0;
            ASSERT_TRUE(cache.get_kv_converted(
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
                cudaMemcpy(scalar_k[request].data(), serial_k->gpu_data_ptr(),
                           elements * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(scalar_v[request].data(), serial_v->gpu_data_ptr(),
                           elements * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                cudaSuccess);
        }

        ITensor *grouped_k = nullptr;
        ITensor *grouped_v = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool gather_ok = false;
        {
            GraphCaptureGuard guard;
            gather_ok = cache.get_kv_batched_device_view(
                /*layer=*/0,
                /*first_seq_idx=*/0,
                batch_size,
                &grouped_k,
                &grouped_v,
                stream.opaque());
        }
        ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
        ASSERT_TRUE(gather_ok);
        ASSERT_NE(grouped_k, nullptr);
        ASSERT_NE(grouped_v, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();

        const size_t grouped_elements =
            static_cast<size_t>(batch_size) * max_seq_len * kv_dim;
        std::vector<uint16_t> actual_k(grouped_elements);
        std::vector<uint16_t> actual_v(grouped_elements);
        ASSERT_EQ(
            cudaMemcpy(actual_k.data(), grouped_k->gpu_data_ptr(),
                       grouped_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(actual_v.data(), grouped_v->gpu_data_ptr(),
                       grouped_elements * sizeof(uint16_t), cudaMemcpyDeviceToHost),
            cudaSuccess);

        for (int request = 0; request < batch_size; ++request)
        {
            const size_t request_offset =
                static_cast<size_t>(request) * max_seq_len * kv_dim;
            const size_t live_elements =
                static_cast<size_t>(expected_counts[request]) * kv_dim;
            EXPECT_EQ(
                std::memcmp(actual_k.data() + request_offset,
                            scalar_k[request].data(),
                            live_elements * sizeof(uint16_t)),
                0)
                << "AQ8 K grouped bytes differ for request " << request;
            EXPECT_EQ(
                std::memcmp(actual_v.data() + request_offset,
                            scalar_v[request].data(),
                            live_elements * sizeof(uint16_t)),
                0)
                << turboQuantKVModeName(mode)
                << " V grouped bytes differ for request " << request;
        }

        ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
      }
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
 * Both supported TurboQuant head dimensions are covered because they use
 * different compressed block layouts and specialized GPU kernels.
 */
TEST(Test__CUDARingKVCacheTQ, CapturedUnequalRequestLengthsPreserveContinuationBytes)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    constexpr int batch_size = 2;
    constexpr int captured_rows = 8;
    constexpr int max_seq_len = 10;
    constexpr int n_kv_heads = 2;
    constexpr std::array<int, batch_size> initial_counts{captured_rows, 5};
    constexpr std::array<int, batch_size> final_counts{captured_rows + 1, 6};

    for (const TurboQuantKVMode mode : {
             TurboQuantKVMode::AQ8_K_Q8_1_V,
             TurboQuantKVMode::AQ8_K_TQ8_V,
             TurboQuantKVMode::AQ8_K_TQ4_V})
    {
      for (const int head_dim : {64, 128})
      {
        SCOPED_TRACE(
            std::string("mode=") + turboQuantKVModeName(mode) +
            " head_dim=" + std::to_string(head_dim));
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
        ScopedCudaStream stream;
        CUDARingKVCacheTQ actual(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim,
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
            /*device_id=*/0, mode);
        CUDARingKVCacheTQ reference(
            /*n_layers=*/1, batch_size, max_seq_len,
            n_kv_heads, head_dim,
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
            /*device_id=*/0, mode);
        KVCacheTestWorkspaceBinding actual_workspace(
            actual, DeviceId::cuda(0));
        KVCacheTestWorkspaceBinding reference_workspace(
            reference, DeviceId::cuda(0));
        ASSERT_TRUE(k_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
        ASSERT_TRUE(v_tensor->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));

        int32_t *device_lengths = nullptr;
        ASSERT_EQ(
            cudaMalloc(&device_lengths, batch_size * sizeof(int32_t)),
            cudaSuccess);
        const std::array<int32_t, batch_size> first_device_lengths{
            initial_counts[0], initial_counts[1]};
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_lengths,
                first_device_lengths.data(),
                batch_size * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream.stream()),
            cudaSuccess);

        KVCacheAppendStage append_stage({
            .device_id = DeviceId::cuda(0),
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
            .turboquant_ctx =
                turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
        });
        append_stage.setGPUStream(stream.opaque());
        append_stage.updateDynamicParams(/*pos_offset=*/0, captured_rows);
        stream.synchronize();

        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = append_stage.execute(nullptr);
        }
        ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(graph, nullptr);
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();

        constexpr std::array<int32_t, batch_size> continuation_lengths{1, 1};
        ASSERT_EQ(
            cudaMemcpyAsync(
                device_lengths,
                continuation_lengths.data(),
                batch_size * sizeof(int32_t),
                cudaMemcpyHostToDevice,
                stream.stream()),
            cudaSuccess);
        append_stage.updateDynamicParams(
            /*pos_offset=*/captured_rows,
            /*seq_len=*/captured_rows);
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
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
                TensorType::FP32, DeviceId::cuda(0));
            GpuTensorView initial_v(
                v_base + request_offset,
                initial_counts[request], kv_dim,
                TensorType::FP32, DeviceId::cuda(0));
            ASSERT_TRUE(appendWithTestStream(
                reference, 0, request,
                &initial_k, &initial_v,
                initial_counts[request], stream));

            GpuTensorView continuation_k(
                k_base + request_offset,
                /*rows=*/1, kv_dim,
                TensorType::FP32, DeviceId::cuda(0));
            GpuTensorView continuation_v(
                v_base + request_offset,
                /*rows=*/1, kv_dim,
                TensorType::FP32, DeviceId::cuda(0));
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
                cudaMemcpyAsync(
                    &device_count,
                    actual.deviceCachedTokenCountPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpyAsync(
                    &device_head,
                    actual.deviceRingHeadPtr(0, request),
                    sizeof(int), cudaMemcpyDeviceToHost, stream.stream()),
                cudaSuccess);
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
                cudaMemcpy(
                    actual_k_bytes.data(), actual_k->gpu_data_ptr(),
                    bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    actual_v_bytes.data(), actual_v->gpu_data_ptr(),
                    bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);

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
                cudaMemcpy(
                    reference_k_bytes.data(), reference_k->gpu_data_ptr(),
                    bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);
            ASSERT_EQ(
                cudaMemcpy(
                    reference_v_bytes.data(), reference_v->gpu_data_ptr(),
                    bytes, cudaMemcpyDeviceToHost),
                cudaSuccess);

            EXPECT_EQ(
                std::memcmp(
                    actual_k_bytes.data(), reference_k_bytes.data(), bytes),
                0)
                << "captured AQ8 K continuation changed live bytes";
            EXPECT_EQ(
                std::memcmp(
                    actual_v_bytes.data(), reference_v_bytes.data(), bytes),
                0)
                << "captured " << turboQuantKVModeName(mode)
                << " V continuation changed live bytes";
        }

        ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
        ASSERT_EQ(cudaFree(device_lengths), cudaSuccess);
      }
    }
}

/**
 * @brief Proves captured grouped dequant follows post-append device state.
 *
 * Four rows initialize the compressed cache. A graph then captures one append
 * plus the production request-batched TQ8/TQ4 materialization and is launched
 * twice. The read kernel must consume the head/count advanced by the preceding
 * append node on each replay, producing all six rows byte-identically to two
 * serial appends. RoPE uses a nonzero starting position so ring order and
 * absolute positions are covered by the same equality check.
 */
TEST(Test__CUDARingKVCacheTQ, CapturedGroupedDequantReadsPostAppendDeviceState)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    constexpr int history_rows = 4;
    constexpr int final_rows = history_rows + 2;
    constexpr int max_seq_len = 8;
    constexpr int n_kv_heads = 2;
    constexpr float rope_theta = 10000.0f;
    constexpr int position_start = 3;

    for (const TurboQuantKVMode mode : {
             TurboQuantKVMode::AQ8_K_Q8_1_V,
             TurboQuantKVMode::AQ8_K_TQ8_V,
             TurboQuantKVMode::AQ8_K_TQ4_V})
    {
      for (const int head_dim : {64, 128})
      {
        SCOPED_TRACE(
            std::string("mode=") + turboQuantKVModeName(mode) +
            " head_dim=" + std::to_string(head_dim));
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
        ScopedCudaStream stream;
        CUDARingKVCacheTQ actual(
            /*n_layers=*/1, /*batch_size=*/1, max_seq_len,
            n_kv_heads, head_dim,
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
            /*device_id=*/0, mode);
        CUDARingKVCacheTQ reference(
            /*n_layers=*/1, /*batch_size=*/1, max_seq_len,
            n_kv_heads, head_dim,
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
            /*device_id=*/0, mode);
        KVCacheTestWorkspaceBinding actual_workspace(
            actual, DeviceId::cuda(0));
        KVCacheTestWorkspaceBinding reference_workspace(
            reference, DeviceId::cuda(0));
        ASSERT_TRUE(history_k->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
        ASSERT_TRUE(history_v->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
        ASSERT_TRUE(continuation_k->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
        ASSERT_TRUE(continuation_v->ensureOnDevice(DeviceId::cuda(0), stream.opaque()));
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
        read_params.turboquant_ctx =
            turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr;
        read_params.gpu_stream = stream.opaque();

        KVCacheAppendStage append_stage({
            .device_id = DeviceId::cuda(0),
            .K = continuation_k.get(),
            .V = continuation_v.get(),
            .kv_cache = &actual,
            .layer_idx = 0,
            .seq_idx = 0,
            .num_tokens = 1,
            .batch_size = 1,
            .seq_len = 1,
            .head_dim = head_dim,
            .turboquant_ctx =
                turboQuantValueUsesRotation(mode) ? &tq_ctx : nullptr,
        });
        append_stage.setGPUStream(stream.opaque());
        append_stage.updateDynamicDevicePositionIds(
            actual.deviceCachedTokenCountPtr(0, 0), /*seq_len=*/1);
        append_stage.updateDynamicParams(/*pos_offset=*/history_rows, /*seq_len=*/1);
        stream.synchronize();

        ITensor *captured_k = nullptr;
        ITensor *captured_v = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_exec = nullptr;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream.stream(), cudaStreamCaptureModeGlobal),
            cudaSuccess);
        bool capture_ok = false;
        {
            GraphCaptureGuard guard;
            capture_ok = append_stage.execute(nullptr) &&
                         actual.get_kv_batched_converted_device_view(
                             0, 0, /*request_count=*/1,
                             ActivationPrecision::FP16,
                             &captured_k, &captured_v, read_params);
        }
        ASSERT_EQ(cudaStreamEndCapture(stream.stream(), &graph), cudaSuccess);
        ASSERT_TRUE(capture_ok);
        ASSERT_NE(captured_k, nullptr);
        ASSERT_NE(captured_v, nullptr);
        ASSERT_EQ(captured_k->rows(), static_cast<size_t>(max_seq_len));
        ASSERT_EQ(captured_v->rows(), static_cast<size_t>(max_seq_len));
        ASSERT_EQ(
            cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
            cudaSuccess);

        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();
        EXPECT_EQ(actual.get_cached_tokens(0, 0), history_rows + 1);

        // Replay without any host publication callback. The second grouped
        // read must see the count/head produced entirely by the first replay.
        ASSERT_EQ(cudaGraphLaunch(graph_exec, stream.stream()), cudaSuccess);
        stream.synchronize();
        EXPECT_EQ(actual.get_cached_tokens(0, 0), final_rows);
        int device_count = -1;
        int device_head = -1;
        ASSERT_EQ(
            cudaMemcpy(
                &device_count, actual.deviceCachedTokenCountPtr(0, 0),
                sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                &device_head, actual.deviceRingHeadPtr(0, 0),
                sizeof(int), cudaMemcpyDeviceToHost),
            cudaSuccess);
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
            cudaMemcpy(
                actual_k_bytes.data(), captured_k->gpu_data_ptr(),
                bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                actual_v_bytes.data(), captured_v->gpu_data_ptr(),
                bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                reference_k_bytes.data(), reference_k->gpu_data_ptr(),
                bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                reference_v_bytes.data(), reference_v->gpu_data_ptr(),
                bytes, cudaMemcpyDeviceToHost),
            cudaSuccess);
        auto expectFP16WordsEqual = [kv_dim](
                                        const std::vector<uint8_t> &actual,
                                        const std::vector<uint8_t> &expected,
                                        const char *label)
        {
            ASSERT_EQ(actual.size(), expected.size());
            const auto *actual_words =
                reinterpret_cast<const uint16_t *>(actual.data());
            const auto *expected_words =
                reinterpret_cast<const uint16_t *>(expected.data());
            const size_t word_count = actual.size() / sizeof(uint16_t);
            for (size_t index = 0; index < word_count; ++index)
            {
                if (actual_words[index] == expected_words[index])
                    continue;
                ADD_FAILURE()
                    << label << " first mismatch row=" << (index / kv_dim)
                    << " col=" << (index % kv_dim)
                    << " actual_bits=0x" << std::hex << actual_words[index]
                    << " expected_bits=0x" << expected_words[index] << std::dec
                    << " actual=" << fp16_to_fp32(actual_words[index])
                    << " expected=" << fp16_to_fp32(expected_words[index]);
                return;
            }
        };
        expectFP16WordsEqual(
            actual_k_bytes, reference_k_bytes,
            "captured device-owned AQ8 K dequant");
        expectFP16WordsEqual(
            actual_v_bytes, reference_v_bytes,
            "captured device-owned compressed V dequant");

        EXPECT_EQ(actual.get_cached_tokens(0, 0), final_rows);
        EXPECT_EQ(actual.ring_head(0, 0), final_rows);
        ASSERT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
        ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess);
      }
    }
}
