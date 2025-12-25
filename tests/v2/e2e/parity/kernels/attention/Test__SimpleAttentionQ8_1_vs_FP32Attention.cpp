/**
 * @file Test__SimpleAttentionQ8_1_vs_FP32Attention.cpp
 * @brief Kernel Parity: QuantisedAttentionJit_Q8_1_Fused (Simple Q8_1) vs FP32 Attention
 *
 * @category e2e/parity/kernels/attention
 * @tested   QuantisedAttentionJit_Q8_1_Fused (1082-line simple JIT attention)
 * @reference FP32 decomposed attention (Q@K^T -> softmax -> @V)
 *
 * This test isolates the attention mechanism to compare:
 *   - FP32 path: Q @ K^T → softmax → scores @ V (decomposed)
 *   - Q8_1 path: Fused online softmax attention (QuantisedAttentionJit_Q8_1_Fused)
 *
 * The test uses identical input data (FP32 quantized to Q8_1) and compares
 * the final attention context output after dequantization.
 *
 * This helps isolate whether divergence comes from:
 *   1. Q8_1 quantization precision loss
 *   2. Online vs standard softmax algorithm differences
 *   3. Accumulated rounding errors in fused computation
 *
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <cstring>

#include "kernels/KernelFactory.h"
#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#include "kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "utils/Logger.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{
    /**
     * @brief Compute cosine similarity between two float arrays
     */
    double cosine_similarity(const float *a, const float *b, size_t n)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }
        if (norm_a < 1e-12 || norm_b < 1e-12)
            return 0.0;
        return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
    }

    /**
     * @brief Compute max absolute difference
     */
    double max_abs_diff(const float *a, const float *b, size_t n)
    {
        double max_diff = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            max_diff = std::max(max_diff, diff);
        }
        return max_diff;
    }

    /**
     * @brief Compute mean absolute difference
     */
    double mean_abs_diff(const float *a, const float *b, size_t n)
    {
        double sum = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            sum += std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        }
        return sum / static_cast<double>(n);
    }

    /**
     * @brief Generate random FP32 data with controlled magnitude
     */
    void generate_random_data(float *data, size_t n, float scale = 1.0f, unsigned seed = 42)
    {
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, scale);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(gen);
        }
    }

    /**
     * @brief Quantize FP32 to Q8_1 blocks
     */
    void quantize_fp32_to_q8_1(const float *fp32, int num_rows, int row_dim, Q8_1Block *blocks)
    {
        int num_blocks = (row_dim + 31) / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                Q8_1Block &block = blocks[row * num_blocks + b];
                const float *src = fp32 + row * row_dim + b * 32;

                // Find max abs value
                float max_abs = 0.0f;
                for (int i = 0; i < 32 && (b * 32 + i) < row_dim; ++i)
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
                    float val = (b * 32 + i < row_dim) ? src[i] : 0.0f;
                    int8_t q = static_cast<int8_t>(std::round(std::clamp(val * inv_d, -127.0f, 127.0f)));
                    block.qs[i] = q;
                    sum_qs += q;
                }
                block.sum_qs = sum_qs;
            }
        }
    }

    /**
     * @brief Dequantize Q8_1 blocks to FP32
     */
    void dequant_q8_1_to_fp32(const Q8_1Block *blocks, int num_rows, int row_dim, float *out_fp32)
    {
        int num_blocks = (row_dim + 31) / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                const Q8_1Block &block = blocks[row * num_blocks + b];
                float d = simd::fp16_to_fp32(block.d);
                for (int i = 0; i < 32 && (b * 32 + i) < row_dim; ++i)
                {
                    out_fp32[row * row_dim + b * 32 + i] = static_cast<float>(block.qs[i]) * d;
                }
            }
        }
    }

    /**
     * @brief Compute FP32 reference attention (decomposed)
     *
     * Implements: output = softmax(Q @ K^T * scale + mask) @ V
     */
    void compute_fp32_reference(
        int M, int N, int D,
        const float *Q_fp32, const float *K_fp32, const float *V_fp32,
        float scale,
        float *output_fp32,
        const float *mask = nullptr,
        int mask_stride = 0)
    {
        // Step 1: Compute Q @ K^T -> scores [M, N]
        std::vector<float> scores(M * N);
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float dot = 0.0f;
                for (int d = 0; d < D; ++d)
                {
                    dot += Q_fp32[i * D + d] * K_fp32[j * D + d];
                }
                float score = dot * scale;
                // Apply mask if provided
                if (mask)
                {
                    score += mask[i * mask_stride + j];
                }
                scores[i * N + j] = score;
            }
        }

        // Step 2: Row-wise softmax
        for (int i = 0; i < M; ++i)
        {
            // Find max for numerical stability
            float max_val = scores[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, scores[i * N + j]);
            }

            // Compute exp and sum
            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] = std::exp(scores[i * N + j] - max_val);
                sum_exp += scores[i * N + j];
            }

            // Normalize
            float inv_sum = 1.0f / sum_exp;
            for (int j = 0; j < N; ++j)
            {
                scores[i * N + j] *= inv_sum;
            }
        }

        // Step 3: scores @ V -> output [M, D]
        for (int i = 0; i < M; ++i)
        {
            for (int d = 0; d < D; ++d)
            {
                float acc = 0.0f;
                for (int j = 0; j < N; ++j)
                {
                    acc += scores[i * N + j] * V_fp32[j * D + d];
                }
                output_fp32[i * D + d] = acc;
            }
        }
    }

    /**
     * @brief Create causal attention mask (lower triangular)
     */
    void create_causal_mask(int M, int N, float *mask)
    {
        const float NEG_INF = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                mask[i * N + j] = (j <= i) ? 0.0f : NEG_INF;
            }
        }
    }

} // anonymous namespace

/**
 * @brief Test fixture for Q8_1 vs FP32 attention parity
 */
class Test__Q8_1_vs_FP32_AttentionParity : public ::testing::Test
{
protected:
    // Qwen2 0.5B dimensions
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr int D_MODEL = N_HEADS * HEAD_DIM;           // 896
    static constexpr int HEAD_DIM_BLOCKS = (HEAD_DIM + 31) / 32; // 2 blocks per head

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    void TearDown() override {}

    /**
     * @brief Run single-head FP32 reference attention
     */
    void run_fp32_single_head(
        int seq_len, int kv_len, int head_dim,
        const float *Q_fp32, const float *K_fp32, const float *V_fp32,
        float *output_fp32,
        bool causal)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        std::vector<float> mask;
        const float *mask_ptr = nullptr;
        int mask_stride = 0;

        if (causal)
        {
            mask.resize(seq_len * kv_len);
            create_causal_mask(seq_len, kv_len, mask.data());
            mask_ptr = mask.data();
            mask_stride = kv_len;
        }

        compute_fp32_reference(seq_len, kv_len, head_dim,
                               Q_fp32, K_fp32, V_fp32,
                               scale, output_fp32,
                               mask_ptr, mask_stride);
    }

    /**
     * @brief Run single-head Q8_1 JIT fused attention
     */
    void run_q8_1_single_head(
        int seq_len, int kv_len, int head_dim,
        const Q8_1Block *Q_blocks, const Q8_1Block *K_blocks, const Q8_1Block *V_blocks,
        Q8_1Block *output_blocks,
        bool causal)
    {
        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        int head_dim_blocks = (head_dim + 31) / 32;
        int stride_bytes = head_dim_blocks * static_cast<int>(sizeof(Q8_1Block));

        // Create JIT kernel
        QuantisedAttentionJit_Q8_1_Fused jit_kernel(head_dim, false);
        auto kernel_func = jit_kernel.get_kernel();

        // Create mask if causal
        std::vector<float> mask;
        const float *mask_ptr = nullptr;
        int mask_stride = 0;

        if (causal)
        {
            mask.resize(seq_len * kv_len);
            create_causal_mask(seq_len, kv_len, mask.data());
            mask_ptr = mask.data();
            mask_stride = kv_len;
        }

        // Build params
        FusedQ8_1AttentionParams params;
        params.Q = Q_blocks;
        params.K = K_blocks;
        params.V = V_blocks;
        params.output = output_blocks;
        params.M = seq_len;
        params.N = kv_len;
        params.head_dim = head_dim;
        params.Q_stride_bytes = stride_bytes;
        params.K_stride_bytes = stride_bytes;
        params.V_stride_bytes = stride_bytes;
        params.output_stride_bytes = stride_bytes;
        params.scale = scale;
        params.mask = mask_ptr;
        params.mask_stride = mask_stride;

        // Execute JIT kernel
        kernel_func(&params);
    }

    /**
     * @brief Compare outputs and print detailed metrics
     */
    void compare_outputs(
        const std::string &test_name,
        const float *fp32_output,
        const float *q8_1_output,
        size_t n,
        double &out_cosine,
        double &out_max_diff,
        double &out_mean_diff)
    {
        out_cosine = cosine_similarity(fp32_output, q8_1_output, n);
        out_max_diff = max_abs_diff(fp32_output, q8_1_output, n);
        out_mean_diff = mean_abs_diff(fp32_output, q8_1_output, n);

        LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
        LOG_INFO("║  " << std::left << std::setw(60) << test_name << "  ║");
        LOG_INFO("╠════════════════════════════════════════════════════════════════╣");
        LOG_INFO("║  Cosine Similarity: " << std::fixed << std::setprecision(6)
                                          << std::setw(40) << out_cosine << "  ║");
        LOG_INFO("║  Max Abs Diff:      " << std::scientific << std::setprecision(4)
                                          << std::setw(40) << out_max_diff << "  ║");
        LOG_INFO("║  Mean Abs Diff:     " << std::scientific << std::setprecision(4)
                                          << std::setw(40) << out_mean_diff << "  ║");
        LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

        // Print first few values for debugging
        LOG_DEBUG("  FP32 output[0:8]: "
                  << fp32_output[0] << ", " << fp32_output[1] << ", "
                  << fp32_output[2] << ", " << fp32_output[3] << ", "
                  << fp32_output[4] << ", " << fp32_output[5] << ", "
                  << fp32_output[6] << ", " << fp32_output[7]);
        LOG_DEBUG("  Q8_1 output[0:8]: "
                  << q8_1_output[0] << ", " << q8_1_output[1] << ", "
                  << q8_1_output[2] << ", " << q8_1_output[3] << ", "
                  << q8_1_output[4] << ", " << q8_1_output[5] << ", "
                  << q8_1_output[6] << ", " << q8_1_output[7]);
    }

    int rank_ = 0;
    int world_size_ = 1;
};

/**
 * @brief Test: Single token attention (decode-like, single head)
 *
 * seq_len=1, kv_len=1, single head - simplest case
 * This isolates the attention kernel without multi-head complexity
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, SingleHead_SingleToken)
{
    const int seq_len = 1;
    const int kv_len = 1;
    const size_t fp32_size = seq_len * HEAD_DIM;
    const size_t block_count = seq_len * HEAD_DIM_BLOCKS;

    // Create FP32 input for single head
    std::vector<float> Q_fp32(seq_len * HEAD_DIM);
    std::vector<float> K_fp32(kv_len * HEAD_DIM);
    std::vector<float> V_fp32(kv_len * HEAD_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 1.0f, 42);
    generate_random_data(K_fp32.data(), K_fp32.size(), 1.0f, 43);
    generate_random_data(V_fp32.data(), V_fp32.size(), 0.5f, 44);

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, HEAD_DIM, V_q8.data());

    // Create output buffers
    std::vector<float> out_fp32(seq_len * HEAD_DIM);
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);

    // Run FP32 reference
    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                         out_fp32.data(), false);

    // Run Q8_1 JIT
    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), false);

    // Dequantize Q8_1 output for comparison
    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    // Compare
    double cosine, max_diff, mean_diff;
    compare_outputs("SingleHead_SingleToken (seq=1, kv=1)",
                    out_fp32.data(), out_q8_dequant.data(),
                    fp32_size, cosine, max_diff, mean_diff);

    // For single token, we expect high similarity
    EXPECT_GE(cosine, 0.95) << "Single token attention cosine too low";
}

/**
 * @brief Test: Short sequence with causal mask (single head)
 *
 * seq_len=9, kv_len=9, single head with causal masking
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, SingleHead_ShortSequenceCausal)
{
    const int seq_len = 9;
    const int kv_len = 9;
    const size_t fp32_size = seq_len * HEAD_DIM;

    std::vector<float> Q_fp32(seq_len * HEAD_DIM);
    std::vector<float> K_fp32(kv_len * HEAD_DIM);
    std::vector<float> V_fp32(kv_len * HEAD_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 1.0f, 100);
    generate_random_data(K_fp32.data(), K_fp32.size(), 1.0f, 101);
    generate_random_data(V_fp32.data(), V_fp32.size(), 0.5f, 102);

    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, HEAD_DIM, V_q8.data());

    std::vector<float> out_fp32(seq_len * HEAD_DIM);
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);

    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                         out_fp32.data(), true);

    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), true);

    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    double cosine, max_diff, mean_diff;
    compare_outputs("SingleHead_ShortSequenceCausal (seq=9, kv=9)",
                    out_fp32.data(), out_q8_dequant.data(),
                    fp32_size, cosine, max_diff, mean_diff);

    EXPECT_GE(cosine, 0.90) << "Short sequence causal attention cosine too low";
}

/**
 * @brief Test: Decode step (single head)
 *
 * seq_len=1, kv_len=32 - typical decode scenario
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, SingleHead_DecodeStep)
{
    const int seq_len = 1;
    const int kv_len = 32;
    const size_t fp32_size = seq_len * HEAD_DIM;

    std::vector<float> Q_fp32(seq_len * HEAD_DIM);
    std::vector<float> K_fp32(kv_len * HEAD_DIM);
    std::vector<float> V_fp32(kv_len * HEAD_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 1.0f, 200);
    generate_random_data(K_fp32.data(), K_fp32.size(), 1.0f, 201);
    generate_random_data(V_fp32.data(), V_fp32.size(), 0.5f, 202);

    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, HEAD_DIM, V_q8.data());

    std::vector<float> out_fp32(seq_len * HEAD_DIM);
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);

    // Decode doesn't use causal mask
    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                         out_fp32.data(), false);

    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), false);

    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    double cosine, max_diff, mean_diff;
    compare_outputs("SingleHead_DecodeStep (seq=1, kv=32)",
                    out_fp32.data(), out_q8_dequant.data(),
                    fp32_size, cosine, max_diff, mean_diff);

    EXPECT_GE(cosine, 0.90) << "Decode step attention cosine too low";
}

/**
 * @brief Test: Quantization baseline error (no attention)
 *
 * This test quantizes FP32 → Q8_1 → dequantize and compares to original
 * to establish the baseline quantization error (independent of attention)
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, QuantizationBaselineError)
{
    const int seq_len = 9;
    const size_t size = seq_len * HEAD_DIM;

    std::vector<float> original(size);
    generate_random_data(original.data(), size, 1.0f, 500);

    std::vector<Q8_1Block> quantized(seq_len * HEAD_DIM_BLOCKS);
    quantize_fp32_to_q8_1(original.data(), seq_len, HEAD_DIM, quantized.data());

    std::vector<float> dequantized(size);
    dequant_q8_1_to_fp32(quantized.data(), seq_len, HEAD_DIM, dequantized.data());

    double cosine = cosine_similarity(original.data(), dequantized.data(), size);
    double max_diff = max_abs_diff(original.data(), dequantized.data(), size);
    double mean_diff = mean_abs_diff(original.data(), dequantized.data(), size);

    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║      Q8_1 Quantization Baseline Error (no attention)           ║");
    LOG_INFO("╠════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  Cosine Similarity: " << std::fixed << std::setprecision(6)
                                      << std::setw(40) << cosine << "  ║");
    LOG_INFO("║  Max Abs Diff:      " << std::scientific << std::setprecision(4)
                                      << std::setw(40) << max_diff << "  ║");
    LOG_INFO("║  Mean Abs Diff:     " << std::scientific << std::setprecision(4)
                                      << std::setw(40) << mean_diff << "  ║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    EXPECT_GE(cosine, 0.999) << "Q8_1 quantization baseline cosine too low";
}

/**
 * @brief Test: Isolate algorithm difference (FP32 on dequantized inputs)
 *
 * This test runs FP32 attention on dequantized Q8_1 inputs, then compares
 * to Q8_1 fused attention. This isolates algorithm differences (online vs
 * standard softmax) from input quantization error.
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, IsolateAlgorithmDifference)
{
    const int seq_len = 9;
    const int kv_len = 9;
    const size_t fp32_size = seq_len * HEAD_DIM;

    // Create original FP32 data
    std::vector<float> Q_orig(seq_len * HEAD_DIM);
    std::vector<float> K_orig(kv_len * HEAD_DIM);
    std::vector<float> V_orig(kv_len * HEAD_DIM);

    generate_random_data(Q_orig.data(), Q_orig.size(), 1.0f, 700);
    generate_random_data(K_orig.data(), K_orig.size(), 1.0f, 701);
    generate_random_data(V_orig.data(), V_orig.size(), 0.5f, 702);

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_orig.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_orig.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_orig.data(), kv_len, HEAD_DIM, V_q8.data());

    // Dequantize to get "Q8_1-quantized" FP32 values
    std::vector<float> Q_dequant(seq_len * HEAD_DIM);
    std::vector<float> K_dequant(kv_len * HEAD_DIM);
    std::vector<float> V_dequant(kv_len * HEAD_DIM);

    dequant_q8_1_to_fp32(Q_q8.data(), seq_len, HEAD_DIM, Q_dequant.data());
    dequant_q8_1_to_fp32(K_q8.data(), kv_len, HEAD_DIM, K_dequant.data());
    dequant_q8_1_to_fp32(V_q8.data(), kv_len, HEAD_DIM, V_dequant.data());

    // Run FP32 attention on dequantized inputs
    std::vector<float> out_fp32_dequant(seq_len * HEAD_DIM);
    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_dequant.data(), K_dequant.data(), V_dequant.data(),
                         out_fp32_dequant.data(), true);

    // Run Q8_1 fused attention
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);
    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), true);

    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    LOG_INFO("");
    LOG_INFO("=== Isolating Algorithm vs Quantization Error ===");
    LOG_INFO("");

    // Compare: FP32 kernel on dequantized inputs vs Q8_1 kernel
    // Both use SAME input values, so difference is ONLY from:
    // 1. Online vs standard softmax algorithm
    // 2. Intermediate precision (FP32 vs Q8_1 accumulators)
    double cos_algo, max_algo, mean_algo;
    compare_outputs("Algorithm Diff: FP32(dequant) vs Q8_1(fused)",
                    out_fp32_dequant.data(), out_q8_dequant.data(),
                    fp32_size, cos_algo, max_algo, mean_algo);

    // Also compare original FP32 vs FP32 on dequantized inputs
    // This shows ONLY input quantization error propagation
    std::vector<float> out_fp32_orig(seq_len * HEAD_DIM);
    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_orig.data(), K_orig.data(), V_orig.data(),
                         out_fp32_orig.data(), true);

    double cos_quant, max_quant, mean_quant;
    compare_outputs("Quant Error: FP32(orig) vs FP32(dequant)",
                    out_fp32_orig.data(), out_fp32_dequant.data(),
                    fp32_size, cos_quant, max_quant, mean_quant);

    LOG_INFO("");
    LOG_INFO("Summary:");
    LOG_INFO("  - Algorithm difference (online vs standard softmax): cos=" << std::fixed << std::setprecision(6) << cos_algo);
    LOG_INFO("  - Input quantization error propagation: cos=" << std::fixed << std::setprecision(6) << cos_quant);
    LOG_INFO("");

    // Key insight: cos_algo tells us how much error comes from the Q8_1 fused kernel
    // algorithm vs FP32 decomposed, given identical input values
    EXPECT_GE(cos_algo, 0.95) << "Algorithm difference too large (online vs standard softmax)";
}

/**
 * @brief Test: Realistic data scale with larger Q/K magnitudes
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, RealisticDataScale)
{
    const int seq_len = 9;
    const int kv_len = 9;
    const size_t fp32_size = seq_len * HEAD_DIM;

    // Realistic scales observed in actual model:
    // - Q after RoPE: typically ±5-10
    // - K after RoPE: typically ±5-10
    // - V: typically ±0.5-2
    std::vector<float> Q_fp32(seq_len * HEAD_DIM);
    std::vector<float> K_fp32(kv_len * HEAD_DIM);
    std::vector<float> V_fp32(kv_len * HEAD_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 5.0f, 600);
    generate_random_data(K_fp32.data(), K_fp32.size(), 5.0f, 601);
    generate_random_data(V_fp32.data(), V_fp32.size(), 1.0f, 602);

    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, HEAD_DIM, V_q8.data());

    std::vector<float> out_fp32(seq_len * HEAD_DIM);
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);

    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                         out_fp32.data(), true);

    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), true);

    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    double cosine, max_diff, mean_diff;
    compare_outputs("RealisticDataScale (Q/K scale=5, V scale=1)",
                    out_fp32.data(), out_q8_dequant.data(),
                    fp32_size, cosine, max_diff, mean_diff);

    EXPECT_GE(cosine, 0.85) << "Realistic scale attention cosine too low";
}

/**
 * @brief Test: Per-token analysis for sequence
 *
 * Analyze cosine similarity per output token to identify where divergence occurs
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, PerTokenAnalysis)
{
    const int seq_len = 9;
    const int kv_len = 9;

    std::vector<float> Q_fp32(seq_len * HEAD_DIM);
    std::vector<float> K_fp32(kv_len * HEAD_DIM);
    std::vector<float> V_fp32(kv_len * HEAD_DIM);

    generate_random_data(Q_fp32.data(), Q_fp32.size(), 1.0f, 300);
    generate_random_data(K_fp32.data(), K_fp32.size(), 1.0f, 301);
    generate_random_data(V_fp32.data(), V_fp32.size(), 0.5f, 302);

    std::vector<Q8_1Block> Q_q8(seq_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> K_q8(kv_len * HEAD_DIM_BLOCKS);
    std::vector<Q8_1Block> V_q8(kv_len * HEAD_DIM_BLOCKS);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, HEAD_DIM, V_q8.data());

    std::vector<float> out_fp32(seq_len * HEAD_DIM);
    std::vector<Q8_1Block> out_q8(seq_len * HEAD_DIM_BLOCKS);

    run_fp32_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_fp32.data(), K_fp32.data(), V_fp32.data(),
                         out_fp32.data(), true);

    run_q8_1_single_head(seq_len, kv_len, HEAD_DIM,
                         Q_q8.data(), K_q8.data(), V_q8.data(),
                         out_q8.data(), true);

    std::vector<float> out_q8_dequant(seq_len * HEAD_DIM);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, HEAD_DIM, out_q8_dequant.data());

    LOG_INFO("╔════════════════════════════════════════════════════════════════╗");
    LOG_INFO("║         Per-Token Attention Cosine Similarity                  ║");
    LOG_INFO("╠════════════════════════════════════════════════════════════════╣");

    double min_cosine = 1.0;
    double max_cosine = 0.0;
    double sum_cosine = 0.0;
    int worst_token = -1;

    for (int t = 0; t < seq_len; ++t)
    {
        const float *fp32_tok = out_fp32.data() + t * HEAD_DIM;
        const float *q8_tok = out_q8_dequant.data() + t * HEAD_DIM;

        double tok_cos = cosine_similarity(fp32_tok, q8_tok, HEAD_DIM);
        sum_cosine += tok_cos;

        if (tok_cos < min_cosine)
        {
            min_cosine = tok_cos;
            worst_token = t;
        }
        max_cosine = std::max(max_cosine, tok_cos);

        // With causal mask, token t attends to positions 0..t
        int attend_count = t + 1;
        std::string status = tok_cos >= 0.95 ? "✓" : (tok_cos >= 0.90 ? "~" : "✗");
        LOG_INFO("║  Token " << t << " (attends to " << attend_count << " pos): cos="
                             << std::fixed << std::setprecision(6) << tok_cos
                             << "  " << status << std::string(20, ' ') << "║");
    }

    double avg_cosine = sum_cosine / seq_len;
    LOG_INFO("╠════════════════════════════════════════════════════════════════╣");
    LOG_INFO("║  Min cosine:  " << std::fixed << std::setprecision(6) << min_cosine
                                << " (token " << worst_token << ")" << std::string(26, ' ') << "║");
    LOG_INFO("║  Max cosine:  " << std::fixed << std::setprecision(6) << max_cosine
                                << std::string(36, ' ') << "║");
    LOG_INFO("║  Avg cosine:  " << std::fixed << std::setprecision(6) << avg_cosine
                                << std::string(36, ' ') << "║");
    LOG_INFO("╚════════════════════════════════════════════════════════════════╝");

    EXPECT_GE(min_cosine, 0.85) << "Worst token (token " << worst_token << ") cosine too low";
    EXPECT_GE(avg_cosine, 0.90) << "Average token cosine too low";
}

/**
 * @brief Test: Multi-head attention with GQA (Qwen2 0.5B layout)
 *
 * This tests the full multi-head attention including GQA (14 Q heads, 2 KV heads)
 * using the CPUAttentionKernelT directly rather than the single-head JIT kernel.
 *
 * This is closer to what the E2E pipeline does.
 */
TEST_F(Test__Q8_1_vs_FP32_AttentionParity, MultiHead_GQA_FullPipeline)
{
    const int seq_len = 9;
    const int kv_len = 9;
    const int n_heads = N_HEADS;       // 14
    const int n_kv_heads = N_KV_HEADS; // 2

    // Q layout: [seq_len, n_heads * head_dim] = [9, 896]
    // K/V layout: [kv_len, n_kv_heads * head_dim] = [9, 128]
    const size_t q_size = seq_len * n_heads * HEAD_DIM;
    const size_t kv_size = kv_len * n_kv_heads * HEAD_DIM;
    const size_t out_size = seq_len * n_heads * HEAD_DIM;

    const int q_blocks_per_row = n_heads * HEAD_DIM_BLOCKS;     // 14 * 2 = 28
    const int kv_blocks_per_row = n_kv_heads * HEAD_DIM_BLOCKS; // 2 * 2 = 4

    // Create FP32 input for all heads
    std::vector<float> Q_fp32(q_size);
    std::vector<float> K_fp32(kv_size);
    std::vector<float> V_fp32(kv_size);

    generate_random_data(Q_fp32.data(), q_size, 1.0f, 1000);
    generate_random_data(K_fp32.data(), kv_size, 1.0f, 1001);
    generate_random_data(V_fp32.data(), kv_size, 0.5f, 1002);

    // Quantize to Q8_1
    std::vector<Q8_1Block> Q_q8(seq_len * q_blocks_per_row);
    std::vector<Q8_1Block> K_q8(kv_len * kv_blocks_per_row);
    std::vector<Q8_1Block> V_q8(kv_len * kv_blocks_per_row);

    quantize_fp32_to_q8_1(Q_fp32.data(), seq_len, n_heads * HEAD_DIM, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), kv_len, n_kv_heads * HEAD_DIM, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), kv_len, n_kv_heads * HEAD_DIM, V_q8.data());

    // === Run FP32 attention using CPUAttentionKernelT<FP32> ===
    std::vector<float> out_fp32(out_size);

    // Create attention kernel
    CPUAttentionKernelT<ActivationPrecision::FP32> fp32_kernel;

    // Need to provide proper shape Q tensor for the FP32 kernel
    // The kernel expects: Q[seq_len, n_heads * head_dim], K[kv_len, n_kv_heads * head_dim], etc.
    std::vector<float> scores_fp32(n_heads * seq_len * kv_len);

    bool fp32_success = fp32_kernel.compute(
        Q_fp32.data(), K_fp32.data(), V_fp32.data(), out_fp32.data(),
        seq_len, n_heads, n_kv_heads, HEAD_DIM,
        true,                               // causal
        -1,                                 // window_size (disabled)
        nullptr, nullptr, nullptr, nullptr, // workspaces (will be allocated internally)
        false,                              // use_bf16
        nullptr,                            // mpi_ctx
        -1                                  // device_idx
    );

    ASSERT_TRUE(fp32_success) << "FP32 multi-head attention failed";

    // === Run Q8_1 attention using CPUAttentionKernelT<Q8_1> ===
    std::vector<Q8_1Block> out_q8(seq_len * q_blocks_per_row);
    std::memset(out_q8.data(), 0, out_q8.size() * sizeof(Q8_1Block));

    CPUAttentionKernelT<ActivationPrecision::Q8_1> q8_1_kernel;

    // Build causal mask
    std::vector<float> mask(seq_len * kv_len);
    create_causal_mask(seq_len, kv_len, mask.data());

    // Create a dummy FP32Tensor for the mask
    FP32Tensor mask_tensor({static_cast<size_t>(seq_len * kv_len)});
    std::memcpy(mask_tensor.mutable_data(), mask.data(), mask.size() * sizeof(float));

    bool q8_1_success = q8_1_kernel.compute_q8_1(
        Q_q8.data(), K_q8.data(), V_q8.data(), out_q8.data(),
        seq_len, n_heads, n_kv_heads, HEAD_DIM,
        true,    // causal
        -1,      // window_size
        nullptr, // scores (not used by fused kernel)
        &mask_tensor,
        nullptr, // mpi_ctx
        -1       // device_idx
    );

    ASSERT_TRUE(q8_1_success) << "Q8_1 multi-head attention failed";

    // Dequantize Q8_1 output for comparison
    std::vector<float> out_q8_dequant(out_size);
    dequant_q8_1_to_fp32(out_q8.data(), seq_len, n_heads * HEAD_DIM, out_q8_dequant.data());

    // === Compare ===
    double cosine, max_diff, mean_diff;
    compare_outputs("MultiHead_GQA_FullPipeline (14Q/2KV, seq=9)",
                    out_fp32.data(), out_q8_dequant.data(),
                    out_size, cosine, max_diff, mean_diff);

    // Also do per-head analysis
    LOG_INFO("");
    LOG_INFO("=== Per-Head Analysis (GQA: 7 Q heads per KV head) ===");
    LOG_INFO("");

    double min_head_cos = 1.0;
    double max_head_cos = 0.0;
    double sum_head_cos = 0.0;
    int worst_head = -1;

    for (int h = 0; h < n_heads; ++h)
    {
        // Extract head h's output for all tokens
        std::vector<float> fp32_head(seq_len * HEAD_DIM);
        std::vector<float> q8_1_head(seq_len * HEAD_DIM);

        for (int t = 0; t < seq_len; ++t)
        {
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                int src_idx = t * n_heads * HEAD_DIM + h * HEAD_DIM + d;
                int dst_idx = t * HEAD_DIM + d;
                fp32_head[dst_idx] = out_fp32[src_idx];
                q8_1_head[dst_idx] = out_q8_dequant[src_idx];
            }
        }

        double head_cos = cosine_similarity(fp32_head.data(), q8_1_head.data(), fp32_head.size());
        sum_head_cos += head_cos;

        if (head_cos < min_head_cos)
        {
            min_head_cos = head_cos;
            worst_head = h;
        }
        max_head_cos = std::max(max_head_cos, head_cos);

        int kv_h = h / (n_heads / n_kv_heads); // GQA: 7 Q heads share 1 KV head
        std::string status = head_cos >= 0.95 ? "✓" : (head_cos >= 0.90 ? "~" : "✗");
        LOG_INFO("  Head " << std::setw(2) << h << " (KV " << kv_h << "): cos="
                           << std::fixed << std::setprecision(6) << head_cos << "  " << status);
    }

    double avg_head_cos = sum_head_cos / n_heads;
    LOG_INFO("");
    LOG_INFO("  Summary: min=" << std::fixed << std::setprecision(6) << min_head_cos
                               << " (head " << worst_head << "), max=" << max_head_cos
                               << ", avg=" << avg_head_cos);

    // Expect high similarity for multi-head attention
    EXPECT_GE(cosine, 0.90) << "Multi-head GQA attention cosine too low";
    EXPECT_GE(min_head_cos, 0.85) << "Worst head cosine too low";
}

// Main function for GoogleTest with MPI
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}