/**
 * @file Test__Q8_1_KVCache_Integration.cpp
 * @brief Integration tests for Q8_1 attention with typed KV cache
 *
 * This test validates the complete data flow:
 * 1. FusedGEMM produces Q8_1 Q/K/V tensors
 * 2. K/V are stored in typed UnifiedKVCache<Q8_1>
 * 3. K/V are retrieved from cache
 * 4. Attention kernel computes with Q and cached K/V
 *
 * Compares Q8_1 path vs FP32 reference to ensure correctness.
 *
 * Key insight: Standalone Q8_1 attention test achieves 0.999969 cosine,
 * but pipeline achieves only 0.855. This test isolates where the issue is.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/UnifiedKVCache.h"
#include "tensors/TensorFactory.h"
#include "kernels/cpu/attention/CPUAttentionKernelTyped.h"
#include "kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "pipelines/attention/GQAAttention.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelLoader.h"
#include "kernels/KernelFactory.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

/**
 * @brief Test fixture for Q8_1 KV Cache integration
 */
class Test__Q8_1_KVCache_Integration : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;
    std::shared_ptr<MPIContext> mpi_ctx_;

    // Qwen 2.5-0.5B attention config
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr int D_MODEL = N_HEADS * HEAD_DIM;   // 896
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128
    static constexpr int N_LAYERS = 24;

    // Thresholds
    static constexpr double MIN_COSINE = 0.999;          // Expect high similarity for Q8_1 roundtrip
    static constexpr double MIN_COSINE_ATTENTION = 0.99; // Attention accumulates error

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
    }

    /**
     * @brief Compute cosine similarity between two vectors
     */
    double compute_cosine(const float *a, const float *b, size_t count) const
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += a[i] * b[i];
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-12);
    }

    /**
     * @brief Compute FP32 reference attention output
     *
     * Q: [seq_len, n_heads * head_dim]
     * K: [kv_len, n_kv_heads * head_dim]
     * V: [kv_len, n_kv_heads * head_dim]
     * Output: [seq_len, n_heads * head_dim]
     */
    void compute_fp32_reference(
        const float *Q, const float *K, const float *V,
        float *output,
        int seq_len, int kv_len,
        int n_heads, int n_kv_heads, int head_dim)
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int heads_per_kv = n_heads / n_kv_heads;
        const int q_stride = n_heads * head_dim;
        const int kv_stride = n_kv_heads * head_dim;

        // Process each head
        for (int h = 0; h < n_heads; ++h)
        {
            int kv_h = h / heads_per_kv;

            for (int q_pos = 0; q_pos < seq_len; ++q_pos)
            {
                // Step 1: Compute Q @ K^T scores
                std::vector<float> scores(kv_len);
                for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; ++d)
                    {
                        float q_val = Q[q_pos * q_stride + h * head_dim + d];
                        float k_val = K[k_pos * kv_stride + kv_h * head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[k_pos] = dot * scale;

                    // Causal mask
                    if (k_pos > q_pos)
                    {
                        scores[k_pos] = -std::numeric_limits<float>::infinity();
                    }
                }

                // Step 2: Softmax
                float max_score = *std::max_element(scores.begin(), scores.end());
                float sum_exp = 0.0f;
                for (int k = 0; k < kv_len; ++k)
                {
                    scores[k] = std::exp(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                for (int k = 0; k < kv_len; ++k)
                {
                    scores[k] /= sum_exp;
                }

                // Step 3: Compute output = scores @ V
                for (int d = 0; d < head_dim; ++d)
                {
                    float acc = 0.0f;
                    for (int k_pos = 0; k_pos < kv_len; ++k_pos)
                    {
                        float v_val = V[k_pos * kv_stride + kv_h * head_dim + d];
                        acc += scores[k_pos] * v_val;
                    }
                    output[q_pos * q_stride + h * head_dim + d] = acc;
                }
            }
        }
    }

    /**
     * @brief Generate random FP32 data
     */
    void generate_random_data(float *data, size_t count, unsigned int seed)
    {
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(gen);
        }
    }

    /**
     * @brief Print first few values of a tensor for debugging
     */
    void print_tensor_sample(const float *data, int rows, int cols, const char *name)
    {
        if (rank_ != 0)
            return;

        std::cout << name << " sample (first row, first 8 values):" << std::endl;
        for (int i = 0; i < std::min(8, cols); ++i)
        {
            std::cout << std::fixed << std::setprecision(4) << data[i] << " ";
        }
        std::cout << std::endl;
    }
};

/**
 * @test Test Q8_1 quantize -> store in cache -> retrieve -> dequantize roundtrip
 *
 * Verifies that the KV cache stores and retrieves Q8_1 data correctly.
 */
TEST_F(Test__Q8_1_KVCache_Integration, KVCache_Q8_1_RoundTrip)
{
    const int SEQ_LEN = 9; // Same as test prompt "The quick brown fox jumps over the lazy dog"
    const int LAYER = 0;

    // Create typed KV cache
    auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::Q8_1>>(
        /*mpi_ctx=*/*mpi_ctx_,
        /*n_layers=*/1,
        /*batch_size=*/1,
        /*max_seq_len=*/512,
        /*n_kv_heads=*/N_KV_HEADS,
        /*head_dim=*/HEAD_DIM,
        /*device_idx=*/-1);

    // Create original FP32 K/V data
    std::vector<float> K_fp32_orig(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32_orig(SEQ_LEN * KV_DIM);
    generate_random_data(K_fp32_orig.data(), K_fp32_orig.size(), 42);
    generate_random_data(V_fp32_orig.data(), V_fp32_orig.size(), 43);

    // Quantize to Q8_1
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});

    simd::quantize_fp32_to_q8_1_blocks(K_fp32_orig.data(), K_q8_1.mutable_q8_1_blocks(), K_fp32_orig.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32_orig.data(), V_q8_1.mutable_q8_1_blocks(), V_fp32_orig.size());

    // Store in KV cache
    ASSERT_TRUE(kv_cache->append_kv(LAYER, &K_q8_1, &V_q8_1, SEQ_LEN));
    EXPECT_EQ(kv_cache->get_cached_tokens(LAYER), SEQ_LEN);

    // Retrieve from cache
    auto cached_K = kv_cache->get_k(LAYER);
    auto cached_V = kv_cache->get_v(LAYER);
    ASSERT_NE(cached_K, nullptr);
    ASSERT_NE(cached_V, nullptr);

    // Cast to Q8_1Tensor
    auto *cached_K_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_K.get());
    auto *cached_V_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_V.get());
    ASSERT_NE(cached_K_q8_1, nullptr);
    ASSERT_NE(cached_V_q8_1, nullptr);

    // Dequantize cached K/V back to FP32
    std::vector<float> K_fp32_roundtrip(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32_roundtrip(SEQ_LEN * KV_DIM);

    simd::dequantize_q8_1_to_fp32(cached_K_q8_1->q8_1_blocks(), K_fp32_roundtrip.data(), K_fp32_roundtrip.size());
    simd::dequantize_q8_1_to_fp32(cached_V_q8_1->q8_1_blocks(), V_fp32_roundtrip.data(), V_fp32_roundtrip.size());

    // Compute cosine similarity between original and roundtrip
    double K_cosine = compute_cosine(K_fp32_orig.data(), K_fp32_roundtrip.data(), K_fp32_orig.size());
    double V_cosine = compute_cosine(V_fp32_orig.data(), V_fp32_roundtrip.data(), V_fp32_orig.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "KV Cache Q8_1 roundtrip test:" << std::endl
                  << "  K cosine similarity: " << K_cosine << std::endl
                  << "  V cosine similarity: " << V_cosine << std::endl;
    }

    EXPECT_GE(K_cosine, MIN_COSINE) << "K roundtrip cosine " << K_cosine << " below threshold " << MIN_COSINE;
    EXPECT_GE(V_cosine, MIN_COSINE) << "V roundtrip cosine " << V_cosine << " below threshold " << MIN_COSINE;
}

/**
 * @test Test Q8_1 attention kernel with data from KV cache
 *
 * Creates Q8_1 K/V, stores in cache, retrieves, runs attention, compares to FP32 reference.
 */
TEST_F(Test__Q8_1_KVCache_Integration, Attention_Q8_1_WithKVCache)
{
    const int SEQ_LEN = 9;
    const int LAYER = 0;

    // Create typed KV cache
    auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::Q8_1>>(
        /*mpi_ctx=*/*mpi_ctx_,
        /*n_layers=*/1,
        /*batch_size=*/1,
        /*max_seq_len=*/512,
        /*n_kv_heads=*/N_KV_HEADS,
        /*head_dim=*/HEAD_DIM,
        /*device_idx=*/-1);

    // Generate FP32 Q/K/V data
    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32(SEQ_LEN * KV_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 100);
    generate_random_data(K_fp32.data(), K_fp32.size(), 101);
    generate_random_data(V_fp32.data(), V_fp32.size(), 102);

    // Compute FP32 reference output
    std::vector<float> ref_output(SEQ_LEN * D_MODEL);
    compute_fp32_reference(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), ref_output.data(),
        SEQ_LEN, SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM);

    // Create Q8_1 tensors
    Q8_1Tensor Q_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor output_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});

    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_q8_1.mutable_q8_1_blocks(), Q_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(K_fp32.data(), K_q8_1.mutable_q8_1_blocks(), K_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32.data(), V_q8_1.mutable_q8_1_blocks(), V_fp32.size());

    // Store K/V in cache
    ASSERT_TRUE(kv_cache->append_kv(LAYER, &K_q8_1, &V_q8_1, SEQ_LEN));

    // Retrieve K/V from cache
    auto cached_K = kv_cache->get_k(LAYER);
    auto cached_V = kv_cache->get_v(LAYER);
    auto *cached_K_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_K.get());
    auto *cached_V_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_V.get());
    ASSERT_NE(cached_K_q8_1, nullptr);
    ASSERT_NE(cached_V_q8_1, nullptr);

    // Create attention kernel
    CPUAttentionKernelTyped<ActivationPrecision::Q8_1> attention_kernel;

    // Build causal mask
    std::vector<float> mask(SEQ_LEN * SEQ_LEN);
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        for (int j = 0; j < SEQ_LEN; ++j)
        {
            mask[i * SEQ_LEN + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    FP32Tensor mask_tensor({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(SEQ_LEN)});
    std::memcpy(mask_tensor.mutable_data(), mask.data(), mask.size() * sizeof(float));

    // Allocate workspaces (required but not used for Q8_1 fused kernel)
    FP32Tensor scores_workspace({static_cast<size_t>(N_HEADS * SEQ_LEN * SEQ_LEN)});
    FP32Tensor context_workspace({static_cast<size_t>(SEQ_LEN * D_MODEL)});

    // Run Q8_1 attention kernel with cached K/V
    // IMPORTANT: We pass Q8_1 blocks reinterpreted as float*
    bool success = attention_kernel.compute(
        reinterpret_cast<const float *>(Q_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(cached_K_q8_1->q8_1_blocks()),
        reinterpret_cast<const float *>(cached_V_q8_1->q8_1_blocks()),
        reinterpret_cast<float *>(output_q8_1.mutable_q8_1_blocks()),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        /*causal=*/true,
        /*window_size=*/-1,
        &scores_workspace,
        nullptr, // buffer
        &context_workspace,
        &mask_tensor,
        /*use_bf16=*/false,
        mpi_ctx_.get(),
        /*device_idx=*/-1);

    ASSERT_TRUE(success) << "Attention kernel failed";

    // Dequantize output for comparison
    std::vector<float> q8_1_output(SEQ_LEN * D_MODEL);
    simd::dequantize_q8_1_to_fp32(output_q8_1.q8_1_blocks(), q8_1_output.data(), q8_1_output.size());

    // Compute metrics
    double cosine = compute_cosine(q8_1_output.data(), ref_output.data(), q8_1_output.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "Attention with KV cache test:" << std::endl
                  << "  Cosine similarity: " << cosine << std::endl;

        // Print samples
        print_tensor_sample(ref_output.data(), SEQ_LEN, D_MODEL, "FP32 reference output");
        print_tensor_sample(q8_1_output.data(), SEQ_LEN, D_MODEL, "Q8_1 output");
    }

    EXPECT_GE(cosine, MIN_COSINE_ATTENTION)
        << "Attention cosine " << cosine << " below threshold " << MIN_COSINE_ATTENTION;
}

/**
 * @test Test attention comparing direct Q8_1 vs cache-retrieved Q8_1
 *
 * This test isolates whether the KV cache is introducing the divergence.
 * Runs attention with:
 *   1. Direct Q8_1 K/V tensors (not from cache)
 *   2. K/V retrieved from cache
 * Compares outputs - they should be identical!
 */
TEST_F(Test__Q8_1_KVCache_Integration, Attention_Direct_vs_CacheRetrieved)
{
    const int SEQ_LEN = 9;
    const int LAYER = 0;

    // Create typed KV cache
    auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::Q8_1>>(
        /*mpi_ctx=*/*mpi_ctx_,
        /*n_layers=*/1,
        /*batch_size=*/1,
        /*max_seq_len=*/512,
        /*n_kv_heads=*/N_KV_HEADS,
        /*head_dim=*/HEAD_DIM,
        /*device_idx=*/-1);

    // Generate FP32 data and quantize
    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32(SEQ_LEN * KV_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 200);
    generate_random_data(K_fp32.data(), K_fp32.size(), 201);
    generate_random_data(V_fp32.data(), V_fp32.size(), 202);

    // Create Q8_1 tensors
    Q8_1Tensor Q_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor output_direct({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    Q8_1Tensor output_cached({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});

    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_q8_1.mutable_q8_1_blocks(), Q_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(K_fp32.data(), K_q8_1.mutable_q8_1_blocks(), K_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32.data(), V_q8_1.mutable_q8_1_blocks(), V_fp32.size());

    // Store K/V in cache
    ASSERT_TRUE(kv_cache->append_kv(LAYER, &K_q8_1, &V_q8_1, SEQ_LEN));

    // Retrieve K/V from cache
    auto cached_K = kv_cache->get_k(LAYER);
    auto cached_V = kv_cache->get_v(LAYER);
    auto *cached_K_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_K.get());
    auto *cached_V_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_V.get());
    ASSERT_NE(cached_K_q8_1, nullptr);
    ASSERT_NE(cached_V_q8_1, nullptr);

    // Build causal mask
    std::vector<float> mask(SEQ_LEN * SEQ_LEN);
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        for (int j = 0; j < SEQ_LEN; ++j)
        {
            mask[i * SEQ_LEN + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    FP32Tensor mask_tensor({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(SEQ_LEN)});
    std::memcpy(mask_tensor.mutable_data(), mask.data(), mask.size() * sizeof(float));

    // Workspaces
    FP32Tensor scores_workspace({static_cast<size_t>(N_HEADS * SEQ_LEN * SEQ_LEN)});
    FP32Tensor context_workspace({static_cast<size_t>(SEQ_LEN * D_MODEL)});

    CPUAttentionKernelTyped<ActivationPrecision::Q8_1> attention_kernel;

    // Run attention with DIRECT K/V tensors
    ASSERT_TRUE(attention_kernel.compute(
        reinterpret_cast<const float *>(Q_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(K_q8_1.q8_1_blocks()), // Direct K
        reinterpret_cast<const float *>(V_q8_1.q8_1_blocks()), // Direct V
        reinterpret_cast<float *>(output_direct.mutable_q8_1_blocks()),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        true, -1, &scores_workspace, nullptr, &context_workspace, &mask_tensor,
        false, mpi_ctx_.get(), -1));

    // Run attention with CACHE-RETRIEVED K/V tensors
    ASSERT_TRUE(attention_kernel.compute(
        reinterpret_cast<const float *>(Q_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(cached_K_q8_1->q8_1_blocks()), // From cache
        reinterpret_cast<const float *>(cached_V_q8_1->q8_1_blocks()), // From cache
        reinterpret_cast<float *>(output_cached.mutable_q8_1_blocks()),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        true, -1, &scores_workspace, nullptr, &context_workspace, &mask_tensor,
        false, mpi_ctx_.get(), -1));

    // Dequantize outputs
    std::vector<float> output_direct_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> output_cached_fp32(SEQ_LEN * D_MODEL);
    simd::dequantize_q8_1_to_fp32(output_direct.q8_1_blocks(), output_direct_fp32.data(), output_direct_fp32.size());
    simd::dequantize_q8_1_to_fp32(output_cached.q8_1_blocks(), output_cached_fp32.data(), output_cached_fp32.size());

    // Compare direct vs cached
    double cosine_direct_vs_cached = compute_cosine(
        output_direct_fp32.data(), output_cached_fp32.data(), output_direct_fp32.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "Direct vs Cache-retrieved attention:" << std::endl
                  << "  Cosine similarity: " << cosine_direct_vs_cached << std::endl;
    }

    // These should be identical (or very close - both use Q8_1 data)
    EXPECT_GE(cosine_direct_vs_cached, 0.9999)
        << "Direct vs cached cosine " << cosine_direct_vs_cached << " indicates KV cache is corrupting data!";
}

/**
 * @test Test GQAAttention static compute method with Q8_1 tensors
 *
 * This tests the actual code path used by the pipeline - GQAAttention::compute
 * which dynamically detects Q8_1 tensors and routes to the Q8_1 kernel.
 */
TEST_F(Test__Q8_1_KVCache_Integration, GQAAttention_Q8_1_Path)
{
    const int SEQ_LEN = 9;

    // Generate FP32 data
    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32(SEQ_LEN * KV_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 300);
    generate_random_data(K_fp32.data(), K_fp32.size(), 301);
    generate_random_data(V_fp32.data(), V_fp32.size(), 302);

    // Compute FP32 reference
    std::vector<float> ref_output(SEQ_LEN * D_MODEL);
    compute_fp32_reference(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), ref_output.data(),
        SEQ_LEN, SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM);

    // Create Q8_1 tensors
    auto Q_q8_1 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    auto K_q8_1 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    auto V_q8_1 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    auto output_q8_1 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});

    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_q8_1->mutable_q8_1_blocks(), Q_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(K_fp32.data(), K_q8_1->mutable_q8_1_blocks(), K_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32.data(), V_q8_1->mutable_q8_1_blocks(), V_fp32.size());

    // Create workspace tensors
    auto workspace_scores = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(N_HEADS * SEQ_LEN * SEQ_LEN)});
    auto workspace_mask = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(SEQ_LEN), static_cast<size_t>(SEQ_LEN)});

    // Configure GQAAttention
    GQAAttentionConfig config;
    config.n_heads = N_HEADS;
    config.n_kv_heads = N_KV_HEADS;
    config.head_dim = HEAD_DIM;
    config.causal = true;
    config.window_size = -1;
    config.precision = ActivationPrecision::Q8_1;
    config.mpi_ctx = mpi_ctx_;
    config.mpi_strategy = MPIStrategy::None;
    config.workspace_scores = workspace_scores;
    config.workspace_mask = workspace_mask;

    // Call GQAAttention::compute (this is what the pipeline uses!)
    bool success = GQAAttention::compute(
        Q_q8_1.get(),
        K_q8_1.get(),
        V_q8_1.get(),
        output_q8_1.get(),
        config,
        /*batch_size=*/1,
        /*sequence_lengths=*/nullptr);

    ASSERT_TRUE(success) << "GQAAttention::compute failed";

    // Dequantize output
    std::vector<float> gqa_output(SEQ_LEN * D_MODEL);
    simd::dequantize_q8_1_to_fp32(output_q8_1->q8_1_blocks(), gqa_output.data(), gqa_output.size());

    // Compute cosine
    double cosine = compute_cosine(gqa_output.data(), ref_output.data(), gqa_output.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "GQAAttention Q8_1 path test:" << std::endl
                  << "  Cosine similarity vs FP32 ref: " << cosine << std::endl;
    }

    EXPECT_GE(cosine, MIN_COSINE_ATTENTION)
        << "GQAAttention cosine " << cosine << " below threshold " << MIN_COSINE_ATTENTION;
}

/**
 * @test Test strided Q8_1 attention - simulating multi-head layout
 *
 * In the pipeline, Q is [seq_len, n_heads * head_dim] with interleaved heads.
 * K/V are [seq_len, n_kv_heads * head_dim] with interleaved kv_heads.
 *
 * The Q8_1 block layout must correctly handle these strides.
 */
TEST_F(Test__Q8_1_KVCache_Integration, Strided_MultiHead_Layout)
{
    const int SEQ_LEN = 9;

    // Generate data with realistic magnitudes
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32(SEQ_LEN * KV_DIM);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Compute reference
    std::vector<float> ref_output(SEQ_LEN * D_MODEL);
    compute_fp32_reference(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), ref_output.data(),
        SEQ_LEN, SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM);

    // Quantize
    Q8_1Tensor Q_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor output_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});

    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_q8_1.mutable_q8_1_blocks(), Q_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(K_fp32.data(), K_q8_1.mutable_q8_1_blocks(), K_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32.data(), V_q8_1.mutable_q8_1_blocks(), V_fp32.size());

    // Verify Q8_1 block layout for Q
    // Q: [9, 896] = [9, 14 heads * 64 head_dim]
    // Q8_1 blocks: head_dim/32 = 2 blocks per head
    // Total blocks per row: 14 * 2 = 28 blocks
    const int q_blocks_per_row = N_HEADS * (HEAD_DIM / 32);
    EXPECT_EQ(q_blocks_per_row, 28);

    // Verify Q8_1 block layout for K
    // K: [9, 128] = [9, 2 kv_heads * 64 head_dim]
    // Blocks per row: 2 * 2 = 4
    const int k_blocks_per_row = N_KV_HEADS * (HEAD_DIM / 32);
    EXPECT_EQ(k_blocks_per_row, 4);

    // Build mask
    std::vector<float> mask(SEQ_LEN * SEQ_LEN);
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        for (int j = 0; j < SEQ_LEN; ++j)
        {
            mask[i * SEQ_LEN + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    FP32Tensor mask_tensor({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(SEQ_LEN)});
    std::memcpy(mask_tensor.mutable_data(), mask.data(), mask.size() * sizeof(float));

    // Workspaces
    FP32Tensor scores_workspace({static_cast<size_t>(N_HEADS * SEQ_LEN * SEQ_LEN)});
    FP32Tensor context_workspace({static_cast<size_t>(SEQ_LEN * D_MODEL)});

    // Run attention
    CPUAttentionKernelTyped<ActivationPrecision::Q8_1> kernel;
    ASSERT_TRUE(kernel.compute(
        reinterpret_cast<const float *>(Q_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(K_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(V_q8_1.q8_1_blocks()),
        reinterpret_cast<float *>(output_q8_1.mutable_q8_1_blocks()),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        true, -1, &scores_workspace, nullptr, &context_workspace, &mask_tensor,
        false, mpi_ctx_.get(), -1));

    // Dequantize
    std::vector<float> q8_1_output(SEQ_LEN * D_MODEL);
    simd::dequantize_q8_1_to_fp32(output_q8_1.q8_1_blocks(), q8_1_output.data(), q8_1_output.size());

    double cosine = compute_cosine(q8_1_output.data(), ref_output.data(), q8_1_output.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "Strided multi-head layout test:" << std::endl
                  << "  Q blocks per row: " << q_blocks_per_row << std::endl
                  << "  K blocks per row: " << k_blocks_per_row << std::endl
                  << "  Cosine similarity: " << cosine << std::endl;

        // Debug: print per-head comparison for first position
        std::cout << "  Per-head comparison (pos 0, dim 0-3):" << std::endl;
        for (int h = 0; h < std::min(4, N_HEADS); ++h)
        {
            std::cout << "    Head " << h << ": ref=[";
            for (int d = 0; d < 4; ++d)
            {
                std::cout << std::fixed << std::setprecision(4) << ref_output[h * HEAD_DIM + d] << " ";
            }
            std::cout << "] q8_1=[";
            for (int d = 0; d < 4; ++d)
            {
                std::cout << std::fixed << std::setprecision(4) << q8_1_output[h * HEAD_DIM + d] << " ";
            }
            std::cout << "]" << std::endl;
        }
    }

    EXPECT_GE(cosine, MIN_COSINE_ATTENTION)
        << "Strided attention cosine " << cosine << " below threshold " << MIN_COSINE_ATTENTION;
}

/**
 * @test Validate Q8_1 block copy semantics in KV cache
 *
 * Ensures the KV cache copy_append_data properly handles Q8_1 block boundaries.
 */
TEST_F(Test__Q8_1_KVCache_Integration, KVCache_BlockCopy_Semantics)
{
    const int SEQ_LEN = 9;
    const int LAYER = 0;

    auto kv_cache = std::make_unique<UnifiedKVCache<ActivationPrecision::Q8_1>>(
        /*mpi_ctx=*/*mpi_ctx_,
        /*n_layers=*/1,
        /*batch_size=*/1,
        /*max_seq_len=*/512,
        /*n_kv_heads=*/N_KV_HEADS,
        /*head_dim=*/HEAD_DIM,
        /*device_idx=*/-1);

    // Create K tensor with known pattern
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});

    // Fill with identifiable pattern
    const int blocks_per_row = KV_DIM / 32; // 128 / 32 = 4
    Q8_1Block *K_blocks = K_q8_1.mutable_q8_1_blocks();
    Q8_1Block *V_blocks = V_q8_1.mutable_q8_1_blocks();

    for (int row = 0; row < SEQ_LEN; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            int idx = row * blocks_per_row + b;
            K_blocks[idx].d = simd::fp32_to_fp16(static_cast<float>(row + 1) * 0.1f);
            V_blocks[idx].d = simd::fp32_to_fp16(static_cast<float>(row + 1) * 0.2f);
            K_blocks[idx].sum_qs = static_cast<int16_t>(row * 100 + b);
            V_blocks[idx].sum_qs = static_cast<int16_t>(row * 100 + b + 50);
            for (int i = 0; i < 32; ++i)
            {
                K_blocks[idx].qs[i] = static_cast<int8_t>((row * 4 + b + i) % 128);
                V_blocks[idx].qs[i] = static_cast<int8_t>((row * 4 + b + i + 64) % 128);
            }
        }
    }

    // Append to cache
    ASSERT_TRUE(kv_cache->append_kv(LAYER, &K_q8_1, &V_q8_1, SEQ_LEN));

    // Retrieve
    auto cached_K = kv_cache->get_k(LAYER);
    auto *cached_K_q8_1 = dynamic_cast<Q8_1Tensor *>(cached_K.get());
    ASSERT_NE(cached_K_q8_1, nullptr);

    // Verify block-by-block
    const Q8_1Block *cached_K_blocks = cached_K_q8_1->q8_1_blocks();
    bool all_match = true;
    int mismatch_count = 0;

    for (int row = 0; row < SEQ_LEN; ++row)
    {
        for (int b = 0; b < blocks_per_row; ++b)
        {
            int idx = row * blocks_per_row + b;
            const Q8_1Block &orig = K_blocks[idx];
            const Q8_1Block &cached = cached_K_blocks[idx];

            if (orig.d != cached.d || orig.sum_qs != cached.sum_qs)
            {
                all_match = false;
                mismatch_count++;
            }

            for (int i = 0; i < 32; ++i)
            {
                if (orig.qs[i] != cached.qs[i])
                {
                    all_match = false;
                    mismatch_count++;
                }
            }
        }
    }

    if (rank_ == 0)
    {
        std::cout << "KV Cache block copy semantics test:" << std::endl
                  << "  Blocks per row: " << blocks_per_row << std::endl
                  << "  Total blocks: " << SEQ_LEN * blocks_per_row << std::endl
                  << "  All blocks match: " << (all_match ? "YES" : "NO") << std::endl
                  << "  Mismatches: " << mismatch_count << std::endl;
    }

    EXPECT_TRUE(all_match) << "KV cache block copy has " << mismatch_count << " mismatches!";
}

/**
 * @brief Main function for MPI-aware test execution
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

/**
 * @test Compare FusedGEMM Q8_1 output layout vs per-head quantization
 *
 * KEY HYPOTHESIS: FusedGEMM produces Q8_1 with row-wise quantization,
 * but attention kernel expects per-head quantization.
 *
 * This test:
 * 1. Computes FP32 Q projection
 * 2. Quantizes using FusedGEMM method (row-wise)
 * 3. Quantizes using per-head method (like the working standalone test)
 * 4. Compares the two Q8_1 outputs
 */
TEST_F(Test__Q8_1_KVCache_Integration, FusedGEMM_vs_PerHead_Quantization)
{
    const int SEQ_LEN = 9;

    // Generate FP32 Q projection (same as pipeline produces before quantization)
    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    generate_random_data(Q_fp32.data(), Q_fp32.size(), 500);

    // Method 1: Row-wise quantization (like FusedGEMM does)
    // Quantize entire row [d_model] together - 896 values = 28 blocks
    Q8_1Tensor Q_rowwise({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_rowwise.mutable_q8_1_blocks(), Q_fp32.size());

    // Method 2: Per-head quantization (like the working standalone test)
    // Quantize each head [head_dim] separately - 64 values = 2 blocks per head
    Q8_1Tensor Q_perhead({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    const int head_dim_blocks = HEAD_DIM / 32;
    const int blocks_per_row = N_HEADS * head_dim_blocks;

    for (int s = 0; s < SEQ_LEN; ++s)
    {
        for (int h = 0; h < N_HEADS; ++h)
        {
            const float *fp32_head = Q_fp32.data() + s * D_MODEL + h * HEAD_DIM;
            Q8_1Block *q8_head = Q_perhead.mutable_q8_1_blocks() + s * blocks_per_row + h * head_dim_blocks;

            // Quantize this head (64 floats = 2 Q8_1 blocks)
            for (int b = 0; b < head_dim_blocks; ++b)
            {
                const float *src = fp32_head + b * 32;
                Q8_1Block &block = q8_head[b];

                // Find max abs value in this 32-element block
                float max_abs = 0.0f;
                for (int i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(src[i]));
                }

                // Compute scale
                float d = max_abs / 127.0f;
                float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;
                block.d = simd::fp32_to_fp16(d);

                // Quantize and compute sum
                int16_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    int8_t q = static_cast<int8_t>(std::round(std::clamp(src[i] * inv_d, -127.0f, 127.0f)));
                    block.qs[i] = q;
                    sum_qs += q;
                }
                block.sum_qs = sum_qs;
            }
        }
    }

    // Dequantize both methods back to FP32
    std::vector<float> Q_rowwise_dequant(SEQ_LEN * D_MODEL);
    std::vector<float> Q_perhead_dequant(SEQ_LEN * D_MODEL);

    simd::dequantize_q8_1_to_fp32(Q_rowwise.q8_1_blocks(), Q_rowwise_dequant.data(), Q_rowwise_dequant.size());
    simd::dequantize_q8_1_to_fp32(Q_perhead.q8_1_blocks(), Q_perhead_dequant.data(), Q_perhead_dequant.size());

    // Compare original FP32 vs each dequantized version
    double cosine_rowwise = compute_cosine(Q_fp32.data(), Q_rowwise_dequant.data(), Q_fp32.size());
    double cosine_perhead = compute_cosine(Q_fp32.data(), Q_perhead_dequant.data(), Q_fp32.size());

    // Compare the two Q8_1 methods against each other
    double cosine_methods = compute_cosine(Q_rowwise_dequant.data(), Q_perhead_dequant.data(), Q_fp32.size());

    // Also compare Q8_1 blocks directly
    const Q8_1Block *rowwise_blocks = Q_rowwise.q8_1_blocks();
    const Q8_1Block *perhead_blocks = Q_perhead.q8_1_blocks();

    int scale_mismatches = 0;
    int value_mismatches = 0;
    double max_scale_diff = 0.0;

    for (int i = 0; i < SEQ_LEN * blocks_per_row; ++i)
    {
        float d_rowwise = simd::fp16_to_fp32(rowwise_blocks[i].d);
        float d_perhead = simd::fp16_to_fp32(perhead_blocks[i].d);

        if (std::abs(d_rowwise - d_perhead) > 1e-4f)
        {
            scale_mismatches++;
            max_scale_diff = std::max(max_scale_diff, static_cast<double>(std::abs(d_rowwise - d_perhead)));
        }

        for (int j = 0; j < 32; ++j)
        {
            if (rowwise_blocks[i].qs[j] != perhead_blocks[i].qs[j])
            {
                value_mismatches++;
            }
        }
    }

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "FusedGEMM vs Per-Head Quantization comparison:" << std::endl
                  << "  FP32 vs Row-wise cosine: " << cosine_rowwise << std::endl
                  << "  FP32 vs Per-head cosine: " << cosine_perhead << std::endl
                  << "  Row-wise vs Per-head cosine: " << cosine_methods << std::endl
                  << "  Scale mismatches: " << scale_mismatches << " / " << (SEQ_LEN * blocks_per_row) << std::endl
                  << "  Value mismatches: " << value_mismatches << " / " << (SEQ_LEN * blocks_per_row * 32) << std::endl
                  << "  Max scale diff: " << max_scale_diff << std::endl;

        // If methods differ, print sample blocks
        if (scale_mismatches > 0)
        {
            std::cout << "  Sample block comparison (row 0, block 0):" << std::endl;
            std::cout << "    Row-wise scale: " << simd::fp16_to_fp32(rowwise_blocks[0].d) << std::endl;
            std::cout << "    Per-head scale: " << simd::fp16_to_fp32(perhead_blocks[0].d) << std::endl;
            std::cout << "    Row-wise values[0:4]: ";
            for (int i = 0; i < 4; ++i)
                std::cout << static_cast<int>(rowwise_blocks[0].qs[i]) << " ";
            std::cout << std::endl;
            std::cout << "    Per-head values[0:4]: ";
            for (int i = 0; i < 4; ++i)
                std::cout << static_cast<int>(perhead_blocks[0].qs[i]) << " ";
            std::cout << std::endl;
        }
    }

    // If both methods achieve similar accuracy, this isn't the issue
    // But if they differ significantly, this could explain the pipeline divergence!

    // Note: The test passes as informational - we want to see the comparison
    EXPECT_GE(cosine_rowwise, 0.999) << "Row-wise quantization should be accurate";
    EXPECT_GE(cosine_perhead, 0.999) << "Per-head quantization should be accurate";
}

/**
 * @test Test Q8_1 attention after RoPE - compares full Q8_1 chain vs FP32
 *
 * This test validates the complete attention chain with RoPE:
 * 1. Generate FP32 Q/K/V
 * 2. Apply FP32 RoPE -> compute FP32 attention (reference)
 * 3. Quantize Q/K/V to Q8_1 -> apply Q8_1 RoPE -> Q8_1 attention
 * 4. Compare outputs
 *
 * This is the key test - if the issue is in RoPE, this will reveal it.
 */
TEST_F(Test__Q8_1_KVCache_Integration, Q8_1_RoPE_Then_Attention)
{
    const int SEQ_LEN = 9;
    const float ROPE_THETA = 1000000.0f; // Qwen default

    // Generate FP32 Q/K/V
    std::vector<float> Q_fp32(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32(SEQ_LEN * KV_DIM);
    std::vector<float> V_fp32(SEQ_LEN * KV_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 600);
    generate_random_data(K_fp32.data(), K_fp32.size(), 601);
    generate_random_data(V_fp32.data(), V_fp32.size(), 602);

    // Position IDs (0 to SEQ_LEN-1)
    std::vector<int> position_ids(SEQ_LEN);
    for (int i = 0; i < SEQ_LEN; ++i)
        position_ids[i] = i;

    // === FP32 Reference Path ===
    std::vector<float> Q_fp32_rope(SEQ_LEN * D_MODEL);
    std::vector<float> K_fp32_rope(SEQ_LEN * KV_DIM);
    std::vector<float> ref_output(SEQ_LEN * D_MODEL);

    // Copy and apply FP32 RoPE
    std::memcpy(Q_fp32_rope.data(), Q_fp32.data(), Q_fp32.size() * sizeof(float));
    std::memcpy(K_fp32_rope.data(), K_fp32.data(), K_fp32.size() * sizeof(float));

    // Simple FP32 RoPE implementation
    auto apply_fp32_rope = [&](std::vector<float> &tensor, int n_heads, int stride)
    {
        for (int seq = 0; seq < SEQ_LEN; ++seq)
        {
            int pos = position_ids[seq];
            for (int h = 0; h < n_heads; ++h)
            {
                float *head = tensor.data() + seq * stride + h * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM / 2; ++d)
                {
                    float freq = 1.0f / std::pow(ROPE_THETA, static_cast<float>(2 * d) / HEAD_DIM);
                    float angle = static_cast<float>(pos) * freq;
                    float cos_val = std::cos(angle);
                    float sin_val = std::sin(angle);

                    float x = head[d];
                    float y = head[d + HEAD_DIM / 2];
                    head[d] = x * cos_val - y * sin_val;
                    head[d + HEAD_DIM / 2] = x * sin_val + y * cos_val;
                }
            }
        }
    };

    apply_fp32_rope(Q_fp32_rope, N_HEADS, D_MODEL);
    apply_fp32_rope(K_fp32_rope, N_KV_HEADS, KV_DIM);

    // Compute FP32 reference attention
    compute_fp32_reference(
        Q_fp32_rope.data(), K_fp32_rope.data(), V_fp32.data(), ref_output.data(),
        SEQ_LEN, SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM);

    // === Q8_1 Path ===
    // Quantize Q/K/V
    Q8_1Tensor Q_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});
    Q8_1Tensor K_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor V_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(KV_DIM)});
    Q8_1Tensor output_q8_1({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(D_MODEL)});

    simd::quantize_fp32_to_q8_1_blocks(Q_fp32.data(), Q_q8_1.mutable_q8_1_blocks(), Q_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(K_fp32.data(), K_q8_1.mutable_q8_1_blocks(), K_fp32.size());
    simd::quantize_fp32_to_q8_1_blocks(V_fp32.data(), V_q8_1.mutable_q8_1_blocks(), V_fp32.size());

    // Apply Q8_1 RoPE using the typed RoPE kernel
    CPURoPEKernelT<ActivationPrecision::Q8_1> rope_kernel;
    ASSERT_TRUE(rope_kernel.apply_typed(
        Q_q8_1.mutable_q8_1_blocks(),
        K_q8_1.mutable_q8_1_blocks(),
        position_ids.data(),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        ROPE_THETA, -1));

    // Dequantize Q/K after RoPE for intermediate comparison
    std::vector<float> Q_q8_1_rope_dequant(SEQ_LEN * D_MODEL);
    std::vector<float> K_q8_1_rope_dequant(SEQ_LEN * KV_DIM);
    simd::dequantize_q8_1_to_fp32(Q_q8_1.q8_1_blocks(), Q_q8_1_rope_dequant.data(), Q_q8_1_rope_dequant.size());
    simd::dequantize_q8_1_to_fp32(K_q8_1.q8_1_blocks(), K_q8_1_rope_dequant.data(), K_q8_1_rope_dequant.size());

    // Compare Q after RoPE
    double Q_rope_cosine = compute_cosine(Q_fp32_rope.data(), Q_q8_1_rope_dequant.data(), Q_fp32_rope.size());
    double K_rope_cosine = compute_cosine(K_fp32_rope.data(), K_q8_1_rope_dequant.data(), K_fp32_rope.size());

    // Build causal mask
    std::vector<float> mask(SEQ_LEN * SEQ_LEN);
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        for (int j = 0; j < SEQ_LEN; ++j)
        {
            mask[i * SEQ_LEN + j] = (j <= i) ? 0.0f : -std::numeric_limits<float>::infinity();
        }
    }
    FP32Tensor mask_tensor({static_cast<size_t>(SEQ_LEN), static_cast<size_t>(SEQ_LEN)});
    std::memcpy(mask_tensor.mutable_data(), mask.data(), mask.size() * sizeof(float));

    // Run Q8_1 attention
    FP32Tensor scores_workspace({static_cast<size_t>(N_HEADS * SEQ_LEN * SEQ_LEN)});
    FP32Tensor context_workspace({static_cast<size_t>(SEQ_LEN * D_MODEL)});
    CPUAttentionKernelTyped<ActivationPrecision::Q8_1> attention_kernel;

    ASSERT_TRUE(attention_kernel.compute(
        reinterpret_cast<const float *>(Q_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(K_q8_1.q8_1_blocks()),
        reinterpret_cast<const float *>(V_q8_1.q8_1_blocks()),
        reinterpret_cast<float *>(output_q8_1.mutable_q8_1_blocks()),
        SEQ_LEN, N_HEADS, N_KV_HEADS, HEAD_DIM,
        true, -1, &scores_workspace, nullptr, &context_workspace, &mask_tensor,
        false, mpi_ctx_.get(), -1));

    // Dequantize output
    std::vector<float> q8_1_output(SEQ_LEN * D_MODEL);
    simd::dequantize_q8_1_to_fp32(output_q8_1.q8_1_blocks(), q8_1_output.data(), q8_1_output.size());

    // Final comparison
    double final_cosine = compute_cosine(ref_output.data(), q8_1_output.data(), ref_output.size());

    if (rank_ == 0)
    {
        std::cout << std::fixed << std::setprecision(6)
                  << "Q8_1 RoPE + Attention test:" << std::endl
                  << "  Q after RoPE cosine (Q8_1 vs FP32): " << Q_rope_cosine << std::endl
                  << "  K after RoPE cosine (Q8_1 vs FP32): " << K_rope_cosine << std::endl
                  << "  Final attention output cosine: " << final_cosine << std::endl;

        // Print sample values
        print_tensor_sample(Q_fp32_rope.data(), SEQ_LEN, D_MODEL, "FP32 Q after RoPE");
        print_tensor_sample(Q_q8_1_rope_dequant.data(), SEQ_LEN, D_MODEL, "Q8_1 Q after RoPE (dequant)");
        print_tensor_sample(ref_output.data(), SEQ_LEN, D_MODEL, "FP32 reference output");
        print_tensor_sample(q8_1_output.data(), SEQ_LEN, D_MODEL, "Q8_1 output (dequant)");
    }

    // RoPE should preserve good accuracy
    EXPECT_GE(Q_rope_cosine, 0.99) << "Q after RoPE cosine " << Q_rope_cosine << " below threshold";
    EXPECT_GE(K_rope_cosine, 0.99) << "K after RoPE cosine " << K_rope_cosine << " below threshold";
    EXPECT_GE(final_cosine, 0.98) << "Final cosine " << final_cosine << " below threshold";
}
