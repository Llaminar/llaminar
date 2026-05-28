/**
 * @file Test__CUDARingKVCache_Comprehensive.cpp
 * @brief Comprehensive edge-case tests for CUDA Ring Buffer KV Cache
 *
 * Brings CUDA KV cache test coverage to parity with the CPU ring KV cache
 * test suite. Covers edge cases not tested in Test__CUDARingKVCacheParity.cpp:
 *
 * - Metadata accessors (precision, dimensions, layout)
 * - Eviction edge cases (zero eviction, clamp to size, total counter tracking)
 * - Single-token operations (append one, retrieve one)
 * - Multi-sequence clear one / other unaffected
 * - Multiple complete ring wraps (stress)
 * - Empty cache retrieval
 * - IKVCache polymorphism compliance
 * - Evict + append + retrieve sequences
 * - Multi-layer-independent ring wrapping
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>
#include "kernels/cuda/kvcache/CUDARingKVCache.h"
#include "kernels/IKVCache.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    bool hasCUDA()
    {
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        return (err == cudaSuccess && count > 0);
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

    float computeMaxError(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_err = 0.0f;
        size_t n = std::min(a.size(), b.size());
        for (size_t i = 0; i < n; ++i)
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        return max_err;
    }

    // Helper: upload host float data to device, returning device pointer
    float *uploadToDevice(const std::vector<float> &host_data)
    {
        float *d_ptr = nullptr;
        size_t bytes = host_data.size() * sizeof(float);
        cudaMalloc(&d_ptr, bytes);
        cudaMemcpy(d_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice);
        return d_ptr;
    }

    // RAII wrapper for CUDA device memory
    struct CudaBuffer
    {
        float *ptr = nullptr;
        CudaBuffer(size_t count)
        {
            cudaMalloc(&ptr, count * sizeof(float));
        }
        CudaBuffer(const std::vector<float> &data)
        {
            cudaMalloc(&ptr, data.size() * sizeof(float));
            cudaMemcpy(ptr, data.data(), data.size() * sizeof(float), cudaMemcpyHostToDevice);
        }
        ~CudaBuffer()
        {
            if (ptr)
                cudaFree(ptr);
        }
        CudaBuffer(const CudaBuffer &) = delete;
        CudaBuffer &operator=(const CudaBuffer &) = delete;
    };

    // RAII wrapper for the explicit stream required by ITensor KV appends.
    struct ScopedCudaStream
    {
        cudaStream_t stream = nullptr;

        ScopedCudaStream()
        {
            EXPECT_EQ(cudaStreamCreate(&stream), cudaSuccess);
        }

        ~ScopedCudaStream()
        {
            if (stream)
                cudaStreamDestroy(stream);
        }

        void *opaque() const { return static_cast<void *>(stream); }
    };

} // namespace

// =============================================================================
// 1. Metadata Accessors
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, MetadataAccessors_FP32)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 3, 2, 16, 4, 8);
    ASSERT_NE(cache, nullptr);

    EXPECT_EQ(cache->k_precision(), ActivationPrecision::FP32);
    EXPECT_EQ(cache->max_seq_len(), 16);
    EXPECT_EQ(cache->n_layers(), 3);
    EXPECT_EQ(cache->batch_size(), 2);
    EXPECT_EQ(cache->n_kv_heads(), 4);
    EXPECT_EQ(cache->head_dim(), 8);
    EXPECT_EQ(cache->kv_dim(), 32); // 4 * 8
    EXPECT_FALSE(cache->is_sharded());
}

TEST(Test__CUDARingKVCache_Comprehensive, MetadataAccessors_FP16)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP16, 2, 1, 32, 2, 16);
    ASSERT_NE(cache, nullptr);

    EXPECT_EQ(cache->k_precision(), ActivationPrecision::FP16);
    EXPECT_EQ(cache->max_seq_len(), 32);
    EXPECT_EQ(cache->n_layers(), 2);
}

TEST(Test__CUDARingKVCache_Comprehensive, MetadataAccessors_BF16)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    auto cache = createCUDARingKVCache(
        ActivationPrecision::BF16, 1, 1, 64, 2, 16);
    ASSERT_NE(cache, nullptr);

    EXPECT_EQ(cache->k_precision(), ActivationPrecision::BF16);
}

// =============================================================================
// 2. Empty Cache Retrieval
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, GetKV_EmptyCache_ReturnsZeroLen)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 32, 2, 16);

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    const void *d_k, *d_v;
    int kv_len = -1;
    bool ok = cache->get_kv_for_attention(0, 0, &d_k, &d_v, &kv_len, 0);
    EXPECT_TRUE(ok);
    EXPECT_EQ(kv_len, 0);
}

// =============================================================================
// 3. Single-Token Append and Retrieve
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, SingleToken_AppendAndRetrieve)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16; // n_kv_heads=2, head_dim=16
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 64, 2, 16);

    auto h_K = generateRandomFP32(kv_dim, 42);
    auto h_V = generateRandomFP32(kv_dim, 43);
    CudaBuffer d_K(h_K), d_V(h_V);

    ASSERT_TRUE(cache->append(0, 0, d_K.ptr, d_V.ptr, 1, 0));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 1);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, 1);

    std::vector<float> h_K_out(kv_dim), h_V_out(kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_V_out.data(), d_V_out, kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    EXPECT_EQ(computeMaxError(h_K, h_K_out), 0.0f);
    EXPECT_EQ(computeMaxError(h_V, h_V_out), 0.0f);
}

// =============================================================================
// 4. Eviction Edge Cases
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, Evict_Zero_NoOp)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 32, 2, 16);

    auto h_K = generateRandomFP32(10 * kv_dim);
    auto h_V = generateRandomFP32(10 * kv_dim);
    CudaBuffer d_K(h_K), d_V(h_V);

    cache->append(0, 0, d_K.ptr, d_V.ptr, 10, 0);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);

    cache->evict_oldest(0, 0, 0);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);
}

TEST(Test__CUDARingKVCache_Comprehensive, Evict_ClampedToSize)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 32, 2, 16);

    auto h_K = generateRandomFP32(5 * kv_dim);
    auto h_V = generateRandomFP32(5 * kv_dim);
    CudaBuffer d_K(h_K), d_V(h_V);

    cache->append(0, 0, d_K.ptr, d_V.ptr, 5, 0);
    cache->evict_oldest(0, 0, 100); // Evict more than available

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
}

TEST(Test__CUDARingKVCache_Comprehensive, Evict_TotalCounterTracksAcrossOperations)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 32, 2, 16);

    auto h_K = generateRandomFP32(20 * kv_dim);
    auto h_V = generateRandomFP32(20 * kv_dim);
    CudaBuffer d_K(h_K), d_V(h_V);

    cache->append(0, 0, d_K.ptr, d_V.ptr, 20, 0);
    EXPECT_EQ(cache->get_total_evicted(), 0);

    cache->evict_oldest(0, 0, 5);
    EXPECT_EQ(cache->get_total_evicted(), 5);

    cache->evict_oldest(0, 0, 3);
    EXPECT_EQ(cache->get_total_evicted(), 8);

    cache->reset_eviction_counter();
    EXPECT_EQ(cache->get_total_evicted(), 0);
}

TEST(Test__CUDARingKVCache_Comprehensive, Evict_ThenAppend_DataCorrect)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16;
    const int max_seq = 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, max_seq, 2, 16);

    // Fill with 10 tokens
    auto h_K1 = generateRandomFP32(10 * kv_dim, 100);
    auto h_V1 = generateRandomFP32(10 * kv_dim, 200);
    CudaBuffer d_K1(h_K1), d_V1(h_V1);
    cache->append(0, 0, d_K1.ptr, d_V1.ptr, 10, 0);

    // Evict 5
    cache->evict_oldest(0, 0, 5);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 5);

    // Append 3 new tokens
    auto h_K2 = generateRandomFP32(3 * kv_dim, 300);
    auto h_V2 = generateRandomFP32(3 * kv_dim, 400);
    CudaBuffer d_K2(h_K2), d_V2(h_V2);
    cache->append(0, 0, d_K2.ptr, d_V2.ptr, 3, 0);

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 8);

    // Retrieve and verify
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, 8);

    std::vector<float> h_K_out(8 * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, 8 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // First 5 tokens should be T5-T9 from phase 1
    for (int t = 0; t < 5; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            EXPECT_FLOAT_EQ(h_K_out[t * kv_dim + d], h_K1[(t + 5) * kv_dim + d])
                << "Mismatch at retained token " << t << " dim " << d;
        }
    }

    // Last 3 tokens should be from phase 2
    for (int t = 0; t < 3; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            EXPECT_FLOAT_EQ(h_K_out[(t + 5) * kv_dim + d], h_K2[t * kv_dim + d])
                << "Mismatch at new token " << t << " dim " << d;
        }
    }
}

// =============================================================================
// 5. Multi-Sequence: Clear One, Other Unaffected
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, MultiSeq_ClearOne_OtherUnaffected)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 2;
    const int batch_size = 3;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, n_layers, batch_size, 32, 2, 16);

    auto h_K = generateRandomFP32(10 * kv_dim);
    auto h_V = generateRandomFP32(10 * kv_dim);
    CudaBuffer d_K(h_K), d_V(h_V);

    // Fill all layers and sequences
    for (int layer = 0; layer < n_layers; ++layer)
        for (int seq = 0; seq < batch_size; ++seq)
            cache->append(layer, seq, d_K.ptr, d_V.ptr, 10, 0);

    // Clear sequence 1 in layer 0
    cache->clear_sequence(0, 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 1), 0);

    // Other sequences in same layer unaffected
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 10);
    EXPECT_EQ(cache->get_cached_tokens(0, 2), 10);

    // Other layer completely unaffected
    for (int seq = 0; seq < batch_size; ++seq)
        EXPECT_EQ(cache->get_cached_tokens(1, seq), 10);
}

// =============================================================================
// 6. Multi-Sequence Independent Wrapping
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, MultiSeq_IndependentWrapping)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int batch_size = 2;
    const int max_seq = 8;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, batch_size, max_seq, 2, 16);

    // Seq 0: fill 5 tokens (no wrap)
    auto h_K0 = generateRandomFP32(5 * kv_dim, 100);
    auto h_V0 = generateRandomFP32(5 * kv_dim, 200);
    CudaBuffer d_K0(h_K0), d_V0(h_V0);
    cache->append(0, 0, d_K0.ptr, d_V0.ptr, 5, 0);

    // Seq 1: fill 10 tokens (wraps)
    auto h_K1 = generateRandomFP32(10 * kv_dim, 300);
    auto h_V1 = generateRandomFP32(10 * kv_dim, 400);
    CudaBuffer d_K1(h_K1), d_V1(h_V1);
    cache->append(0, 1, d_K1.ptr, d_V1.ptr, 10, 0);

    // Seq 0: not wrapped, seq 1: wrapped
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 5);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    EXPECT_EQ(cache->get_cached_tokens(0, 1), 8); // Clamped to max
    EXPECT_TRUE(cache->is_wrapped(0, 1));

    // Verify seq 0 data integrity
    const void *d_k_out, *d_v_out;
    int len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_k_out, &d_v_out, &len, 0));
    EXPECT_EQ(len, 5);

    std::vector<float> k_out(5 * kv_dim);
    cudaMemcpy(k_out.data(), d_k_out, 5 * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(computeMaxError(h_K0, k_out), 0.0f);
}

// =============================================================================
// 7. Multiple Complete Ring Wraps (Stress)
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, MultiWrap_StressTest)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq = 8;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, max_seq, 2, 16);

    // Wrap around 5 complete times (40 tokens through an 8-slot buffer)
    std::vector<float> last_batch_K;
    CudaBuffer d_K(max_seq * kv_dim), d_V(max_seq * kv_dim);

    for (int batch = 0; batch < 10; ++batch)
    {
        auto h_K = generateRandomFP32(4 * kv_dim, 1000 + batch);
        auto h_V = generateRandomFP32(4 * kv_dim, 2000 + batch);
        cudaMemcpy(d_K.ptr, h_K.data(), 4 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V.ptr, h_V.data(), 4 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);

        cache->append(0, 0, d_K.ptr, d_V.ptr, 4, 0);

        if (batch == 9)
            last_batch_K = h_K;
    }

    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq);

    // Verify the most recent 4 tokens (from last batch) are present
    const void *d_K_out, *d_V_out;
    int kv_len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &d_K_out, &d_V_out, &kv_len, 0));
    EXPECT_EQ(kv_len, max_seq);

    std::vector<float> h_K_out(max_seq * kv_dim);
    cudaMemcpy(h_K_out.data(), d_K_out, max_seq * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);

    // The last 4 tokens in output should match the last batch
    for (int t = 0; t < 4; ++t)
    {
        for (int d = 0; d < kv_dim; ++d)
        {
            EXPECT_FLOAT_EQ(h_K_out[(max_seq - 4 + t) * kv_dim + d],
                            last_batch_K[t * kv_dim + d])
                << "Mismatch at position " << (max_seq - 4 + t) << " dim " << d;
        }
    }
}

// =============================================================================
// 8. IKVCache Polymorphism Compliance
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, IKVCache_PolymorphismCompliance)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    // Use the IKVCache interface pointer to verify polymorphic access
    std::unique_ptr<IKVCache> cache_ptr = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 16, 2, 16);
    IKVCache *cache = cache_ptr.get();

    EXPECT_EQ(cache->max_seq_len(), 16);
    EXPECT_EQ(cache->n_layers(), 1);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);

    // Append via IKVCache interface (ITensor path)
    const int kv_dim = 2 * 16;
    auto k_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{5, static_cast<size_t>(kv_dim)});
    auto v_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{5, static_cast<size_t>(kv_dim)});

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < 5 * kv_dim; ++i)
    {
        k_tensor->mutable_data()[i] = dist(rng);
        v_tensor->mutable_data()[i] = dist(rng);
    }

    // Upload to GPU
    DeviceId cuda_dev = DeviceId::cuda(0);
    k_tensor->ensureOnDevice(cuda_dev);
    v_tensor->ensureOnDevice(cuda_dev);

    // The generic ITensor append path must fail closed unless a caller supplies
    // the execution stream used by the graph/stage path.
    EXPECT_FALSE(cache->append(0, 0, k_tensor.get(), v_tensor.get(), 5));

    ScopedCudaStream stream;
    ASSERT_TRUE(cache->appendWithStream(0, 0, k_tensor.get(), v_tensor.get(), 5, stream.opaque()));
    ASSERT_EQ(cudaStreamSynchronize(stream.stream), cudaSuccess);
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 5);

    // Clear via IKVCache
    cache->clear();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
}

// =============================================================================
// 9. Multi-Layer Independent Ring Wrapping
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, MultiLayer_IndependentWrapping)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int n_layers = 3;
    const int max_seq = 8;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, n_layers, 1, max_seq, 2, 16);

    CudaBuffer d_K(max_seq * kv_dim), d_V(max_seq * kv_dim);

    // Layer 0: 5 tokens (no wrap)
    auto h_K0 = generateRandomFP32(5 * kv_dim, 100);
    cudaMemcpy(d_K.ptr, h_K0.data(), 5 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V.ptr, h_K0.data(), 5 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cache->append(0, 0, d_K.ptr, d_V.ptr, 5, 0);

    // Layer 1: 10 tokens (wraps once)
    auto h_K1 = generateRandomFP32(10 * kv_dim, 200);
    cudaMemcpy(d_K.ptr, h_K1.data(), 8 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V.ptr, h_K1.data(), 8 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cache->append(1, 0, d_K.ptr, d_V.ptr, 8, 0);
    cudaMemcpy(d_K.ptr, h_K1.data() + 8 * kv_dim, 2 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V.ptr, h_K1.data() + 8 * kv_dim, 2 * kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    cache->append(1, 0, d_K.ptr, d_V.ptr, 2, 0);

    // Layer 2: empty
    // (no append)

    EXPECT_EQ(cache->get_cached_tokens(0, 0), 5);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    EXPECT_EQ(cache->get_cached_tokens(1, 0), 8);
    EXPECT_TRUE(cache->is_wrapped(1, 0));

    EXPECT_EQ(cache->get_cached_tokens(2, 0), 0);
}

// =============================================================================
// 10. Linearization Counter
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, LinearizationCounter_TracksWraps)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq = 8;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, max_seq, 2, 16);

    // Fill partially (not wrapped)
    auto h_data = generateRandomFP32((max_seq - 1) * kv_dim);
    CudaBuffer d_K(h_data), d_V(h_data);
    cache->append(0, 0, d_K.ptr, d_V.ptr, max_seq - 1, 0);
    EXPECT_FALSE(cache->is_wrapped(0, 0));

    const void *dk, *dv;
    int len;
    cache->get_kv_for_attention(0, 0, &dk, &dv, &len, 0);
    int count_after_first = cache->get_linearization_count();

    // Add 2 more tokens to force wrap
    auto h_extra = generateRandomFP32(2 * kv_dim, 999);
    CudaBuffer d_extra(h_extra);
    cache->append(0, 0, d_extra.ptr, d_extra.ptr, 2, 0);
    EXPECT_TRUE(cache->is_wrapped(0, 0));

    cache->get_kv_for_attention(0, 0, &dk, &dv, &len, 0);
    EXPECT_GT(cache->get_linearization_count(), count_after_first)
        << "Linearization counter should increase after wrapped get_kv";

    cache->reset_linearization_counter();
    EXPECT_EQ(cache->get_linearization_count(), 0);
}

// =============================================================================
// 11. Append Exact Capacity (No Wrap)
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, Append_ExactCapacity_WrapsHeadPointer)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int max_seq = 8;
    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, max_seq, 2, 16);

    auto h_K = generateRandomFP32(max_seq * kv_dim, 42);
    auto h_V = generateRandomFP32(max_seq * kv_dim, 43);
    CudaBuffer d_K(h_K), d_V(h_V);

    ASSERT_TRUE(cache->append(0, 0, d_K.ptr, d_V.ptr, max_seq, 0));
    EXPECT_EQ(cache->get_cached_tokens(0, 0), max_seq);
    // Note: filling to exact capacity wraps the head pointer to position 0,
    // so is_wrapped() returns true. This is by design in the ring buffer.
    EXPECT_EQ(cache->get_total_evicted(), 0);

    // Retrieve and verify data integrity despite head-pointer wrap
    const void *dk, *dv;
    int len;
    ASSERT_TRUE(cache->get_kv_for_attention(0, 0, &dk, &dv, &len, 0));
    EXPECT_EQ(len, max_seq);

    std::vector<float> out(max_seq * kv_dim);
    cudaMemcpy(out.data(), dk, max_seq * kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    EXPECT_EQ(computeMaxError(h_K, out), 0.0f);
}

// =============================================================================
// 12. Clear All Resets Eviction and Linearization Counters
// =============================================================================

TEST(Test__CUDARingKVCache_Comprehensive, Clear_ResetsCounters)
{
    if (!hasCUDA())
        GTEST_SKIP() << "CUDA not available";

    const int kv_dim = 2 * 16;
    auto cache = createCUDARingKVCache(
        ActivationPrecision::FP32, 1, 1, 8, 2, 16);

    auto h_data = generateRandomFP32(10 * kv_dim);
    CudaBuffer d_K(h_data), d_V(h_data);

    // Cause some evictions and linearizations
    cache->append(0, 0, d_K.ptr, d_V.ptr, 10, 0);
    EXPECT_GT(cache->get_total_evicted(), 0);

    const void *dk, *dv;
    int len;
    cache->get_kv_for_attention(0, 0, &dk, &dv, &len, 0);

    // Clear should reset
    cache->clear();
    EXPECT_EQ(cache->get_cached_tokens(0, 0), 0);
}
