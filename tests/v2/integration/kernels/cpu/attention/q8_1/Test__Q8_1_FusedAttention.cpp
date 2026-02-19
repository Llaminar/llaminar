/**
 * @file Test__Q8_1_FusedAttention.cpp
 * @brief Integration tests for fully fused Q8_1 attention kernel correctness
 *
 * Validates the QuantisedAttentionJit_Q8_1_Fused kernel against a pure FP32
 * reference implementation. The fused kernel performs:
 *   1. Q8_1 × Q8_1 dot product (Q @ K^T)
 *   2. Online softmax (no intermediate score storage)
 *   3. Online weighted V accumulation
 *   4. Online requantization to Q8_1 output
 *
 * The FP32 reference uses:
 *   1. FP32 GEMM for Q @ K^T
 *   2. FP32 Softmax (row-wise)
 *   3. FP32 GEMM for scores @ V
 *   4. Final FP32 output (then quantized for comparison)
 *
 * Metrics computed:
 *   - Relative L2 error: ||fused - ref||_2 / ||ref||_2
 *   - Cosine similarity: (fused · ref) / (||fused|| × ||ref||)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"
#include "kernels/cpu/gemm_v4/QuantisedAttentionJit_Q8_1_Fused.h"
#include "kernels/cpu/ops/CPUSoftmaxKernelT.h"
#include "utils/CPUFeatures.h"
#include "loaders/ModelContext.h"
#include "loaders/ModelLoader.h"
#include "tensors/TensorFactory.h"
#include "utils/MPIContext.h"
#include "kernels/KernelFactory.h"
#include "models/qwen/Qwen2Schema.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{
    ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device_id = DeviceId::cpu())
    {
        auto *prepared = llaminar::v2::kernels::KernelFactory::getOrCreatePreparedGemmWeights(tensor, device_id);
        if (!prepared)
        {
            return nullptr;
        }
        return llaminar::v2::kernels::KernelFactory::getOrCreateGemmEngine(prepared);
    }
}

/**
 * @brief Model attention configuration for test cases
 */
struct TestModelConfig
{
    std::string name;
    int head_dim; ///< Dimension per head (must be multiple of 32)
};

// Model configurations
static const TestModelConfig QWEN_0_5B = {"Qwen-0.5B", 64};
static const TestModelConfig QWEN_7B = {"Qwen-7B", 128};

/**
 * @brief Test fixture for Q8_1 Fused Attention kernel correctness
 */
class Test__Q8_1_FusedAttention : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    // Correctness thresholds for Q8_1 quantization
    // Q8_1 has ~0.4% quantization error per operation, compounded through attention
    static constexpr double MIN_COSINE_SIM = 0.98;   // Allow for quantization noise
    static constexpr double MAX_REL_L2_ERROR = 0.15; // 15% relative L2 (quantization accumulates)

    // Stricter thresholds for small sequences (less error accumulation)
    static constexpr double MIN_COSINE_SIM_SMALL = 0.99;
    static constexpr double MAX_REL_L2_ERROR_SMALL = 0.10;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    /**
     * @brief Compute pure FP32 reference attention
     *
     * Implements: output = softmax(Q @ K^T * scale + mask) @ V
     *
     * @param M Sequence length (Q rows)
     * @param N KV length (K/V rows)
     * @param D Head dimension
     * @param Q_fp32 Query [M, D]
     * @param K_fp32 Key [N, D]
     * @param V_fp32 Value [N, D]
     * @param scale Attention scale (1/sqrt(D))
     * @param output_fp32 Output [M, D]
     * @param mask Optional attention mask [M, N] (nullptr if none)
     * @param mask_stride Stride between mask rows (in floats)
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
     *
     * mask[i][j] = 0.0 if j <= i, -inf otherwise
     *
     * @param M Sequence length
     * @param N KV length
     * @param mask Output mask [M, N]
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

    /**
     * @brief Dequantize Q8_1 blocks to FP32 for comparison
     */
    void dequant_q8_1_to_fp32(const Q8_1Block *blocks, int num_rows, int head_dim, float *out_fp32)
    {
        int num_blocks = head_dim / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                const Q8_1Block &block = blocks[row * num_blocks + b];
                float d = simd::fp16_to_fp32(block.d);
                for (int i = 0; i < 32; ++i)
                {
                    out_fp32[row * head_dim + b * 32 + i] = static_cast<float>(block.qs[i]) * d;
                }
            }
        }
    }

    /**
     * @brief Quantize FP32 to Q8_1 blocks
     */
    void quantize_fp32_to_q8_1(const float *fp32, int num_rows, int head_dim, Q8_1Block *blocks)
    {
        int num_blocks = head_dim / 32;
        for (int row = 0; row < num_rows; ++row)
        {
            for (int b = 0; b < num_blocks; ++b)
            {
                Q8_1Block &block = blocks[row * num_blocks + b];
                const float *src = fp32 + row * head_dim + b * 32;

                // Find max abs value
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

    /**
     * @brief Compute relative L2 error and cosine similarity
     * @return pair of (rel_l2_error, cosine_similarity)
     */
    std::pair<double, double> compute_metrics(
        const float *actual, const float *reference, int count)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (int i = 0; i < count; ++i)
        {
            double a = actual[i];
            double r = reference[i];
            double diff = a - r;

            sum_sq_diff += diff * diff;
            sum_sq_ref += r * r;

            dot_prod += a * r;
            norm_act += a * a;
            norm_ref += r * r;
        }

        double rel_l2 = (sum_sq_ref > 1e-12) ? std::sqrt(sum_sq_diff / sum_sq_ref) : 0.0;
        double cosine = (norm_act > 1e-12 && norm_ref > 1e-12)
                            ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref))
                            : 0.0;

        return {rel_l2, cosine};
    }

    /**
     * @brief Run fused attention test
     */
    void run_fused_attention_test(const TestModelConfig &model, int seq_len, int kv_len)
    {
        const int M = seq_len;
        const int N = kv_len;
        const int D = model.head_dim;
        const int num_blocks = D / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Generate random FP32 data
        std::mt19937 gen(42 + seq_len * 1000 + kv_len * 10 + D);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> Q_fp32(M * D);
        std::vector<float> K_fp32(N * D);
        std::vector<float> V_fp32(N * D);

        for (auto &x : Q_fp32)
            x = dist(gen);
        for (auto &x : K_fp32)
            x = dist(gen);
        for (auto &x : V_fp32)
            x = dist(gen);

        // Compute FP32 reference
        std::vector<float> ref_output(M * D);
        compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

        // Quantize inputs to Q8_1
        std::vector<Q8_1Block> Q_q8(M * num_blocks);
        std::vector<Q8_1Block> K_q8(N * num_blocks);
        std::vector<Q8_1Block> V_q8(N * num_blocks);

        quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

        // Output buffer for fused kernel (Q8_1)
        std::vector<Q8_1Block> output_q8(M * num_blocks);
        std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

        // Strides in bytes
        const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

        // Run fused kernel using reference implementation
        // (JIT kernel requires specific head_dim, use reference for general testing)
        fused_q8_1_attention_reference(
            Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
            M, N, D,
            num_blocks, num_blocks, num_blocks, num_blocks,
            scale, nullptr, 0);

        // Dequantize output for comparison
        std::vector<float> fused_output(M * D);
        dequant_q8_1_to_fp32(output_q8.data(), M, D, fused_output.data());

        // Compute metrics
        auto [rel_l2, cosine] = compute_metrics(fused_output.data(), ref_output.data(), M * D);

        // Choose thresholds based on sequence length
        double min_cosine = (M * N < 256) ? MIN_COSINE_SIM_SMALL : MIN_COSINE_SIM;
        double max_l2 = (M * N < 256) ? MAX_REL_L2_ERROR_SMALL : MAX_REL_L2_ERROR;

        // Print results
        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "  " << model.name << " M=" << M << " N=" << N << " D=" << D
                      << " | Cosine: " << cosine
                      << " | Rel L2: " << rel_l2 << std::endl;
        }

        // Assert correctness
        EXPECT_GE(cosine, min_cosine)
            << model.name << " M=" << M << " N=" << N << " D=" << D
            << ": Cosine similarity " << cosine << " below threshold " << min_cosine;

        EXPECT_LE(rel_l2, max_l2)
            << model.name << " M=" << M << " N=" << N << " D=" << D
            << ": Relative L2 error " << rel_l2 << " above threshold " << max_l2;
    }

    /**
     * @brief Test the JIT kernel specifically (only for supported head_dim)
     */
    void run_jit_kernel_test(int head_dim, int seq_len, int kv_len)
    {
        const int M = seq_len;
        const int N = kv_len;
        const int D = head_dim;
        const int num_blocks = D / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Generate random FP32 data
        std::mt19937 gen(12345 + seq_len * 100 + kv_len);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        std::vector<float> Q_fp32(M * D);
        std::vector<float> K_fp32(N * D);
        std::vector<float> V_fp32(N * D);

        for (auto &x : Q_fp32)
            x = dist(gen);
        for (auto &x : K_fp32)
            x = dist(gen);
        for (auto &x : V_fp32)
            x = dist(gen);

        // Compute FP32 reference
        std::vector<float> ref_output(M * D);
        compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, ref_output.data());

        // Quantize inputs to Q8_1
        std::vector<Q8_1Block> Q_q8(M * num_blocks);
        std::vector<Q8_1Block> K_q8(N * num_blocks);
        std::vector<Q8_1Block> V_q8(N * num_blocks);

        quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

        // Create JIT kernel
        QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
        auto kernel_func = jit_kernel.get_kernel();

        // Output buffer
        std::vector<Q8_1Block> output_q8(M * num_blocks);
        std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

        // Strides in bytes
        const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

        // Build params
        FusedQ8_1AttentionParams params;
        params.Q = Q_q8.data();
        params.K = K_q8.data();
        params.V = V_q8.data();
        params.output = output_q8.data();
        params.M = M;
        params.N = N;
        params.head_dim = D;
        params.Q_stride_bytes = stride_bytes;
        params.K_stride_bytes = stride_bytes;
        params.V_stride_bytes = stride_bytes;
        params.output_stride_bytes = stride_bytes;
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride = 0;

        // Execute JIT kernel
        kernel_func(&params);

        // Dequantize output for comparison
        std::vector<float> jit_output(M * D);
        dequant_q8_1_to_fp32(output_q8.data(), M, D, jit_output.data());

        // Compute metrics
        auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);

        if (rank_ == 0)
        {
            std::cout << std::fixed << std::setprecision(6)
                      << "  JIT D=" << D << " M=" << M << " N=" << N
                      << " | Cosine: " << cosine
                      << " | Rel L2: " << rel_l2 << std::endl;
        }

        // JIT kernel should match reference implementation closely
        EXPECT_GE(cosine, 0.95)
            << "JIT D=" << D << " M=" << M << " N=" << N
            << ": Cosine similarity " << cosine << " below threshold 0.95";

        EXPECT_LE(rel_l2, 0.20)
            << "JIT D=" << D << " M=" << M << " N=" << N
            << ": Relative L2 error " << rel_l2 << " above threshold 0.20";
    }
};

// =============================================================================
// Reference Implementation Tests (all head_dim supported)
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_SingleToken)
{
    // Single token decode scenario
    run_fused_attention_test(QWEN_0_5B, 1, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_SmallPrefill)
{
    // Small prefill
    run_fused_attention_test(QWEN_0_5B, 8, 8);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_MediumPrefill)
{
    // Medium prefill
    run_fused_attention_test(QWEN_0_5B, 32, 32);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_LargePrefill)
{
    // Larger prefill
    run_fused_attention_test(QWEN_0_5B, 64, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen05B_Decode_LongContext)
{
    // Decode with long KV cache
    run_fused_attention_test(QWEN_0_5B, 1, 256);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_SingleToken)
{
    // Single token with larger head_dim
    run_fused_attention_test(QWEN_7B, 1, 64);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_SmallPrefill)
{
    run_fused_attention_test(QWEN_7B, 8, 8);
}

TEST_F(Test__Q8_1_FusedAttention, Reference_Qwen7B_MediumPrefill)
{
    run_fused_attention_test(QWEN_7B, 32, 32);
}

// =============================================================================
// JIT vs C++ Reference Debug Test
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_Debug)
{
    // Compare JIT kernel against C++ reference with random data
    const int M = 1;
    const int N = 4;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use random data like the failing tests
    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize inputs to Q8_1
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Debug: Print Q8_1 block info
    std::cerr << "\n=== Q8_1 Block Debug ===" << std::endl;
    for (int b = 0; b < num_blocks; ++b)
    {
        std::cerr << "Q Block " << b << ": d=" << simd::fp16_to_fp32(Q_q8[b].d)
                  << " sum_qs=" << Q_q8[b].sum_qs;
        std::cerr << " qs[0..3]=" << (int)Q_q8[b].qs[0] << "," << (int)Q_q8[b].qs[1]
                  << "," << (int)Q_q8[b].qs[2] << "," << (int)Q_q8[b].qs[3] << std::endl;
    }
    for (int n = 0; n < N; ++n)
    {
        for (int b = 0; b < num_blocks; ++b)
        {
            std::cerr << "K[" << n << "] Block " << b << ": d="
                      << simd::fp16_to_fp32(K_q8[n * num_blocks + b].d)
                      << " sum_qs=" << K_q8[n * num_blocks + b].sum_qs << std::endl;
        }
    }

    // Manually compute expected dot product for Q[0] · K[0]
    std::cerr << "\n=== Manual dot computation ===" << std::endl;
    float ref_dot = 0.0f;
    float jit_dot_no_scale = 0.0f;
    for (int b = 0; b < num_blocks; ++b)
    {
        float d_q = simd::fp16_to_fp32(Q_q8[b].d);
        float d_k = simd::fp16_to_fp32(K_q8[b].d);

        // Reference: signed × signed
        int32_t ref_acc = 0;
        for (int i = 0; i < 32; ++i)
        {
            ref_acc += static_cast<int32_t>(Q_q8[b].qs[i]) * static_cast<int32_t>(K_q8[b].qs[i]);
        }
        float ref_block = static_cast<float>(ref_acc) * d_q * d_k;
        ref_dot += ref_block;

        // JIT simulation: unsigned × signed with correction
        int32_t jit_acc = 0;
        for (int i = 0; i < 32; ++i)
        {
            uint8_t q_unsigned = static_cast<uint8_t>(static_cast<int>(Q_q8[b].qs[i]) + 128);
            int8_t k_signed = K_q8[b].qs[i];
            jit_acc += static_cast<int32_t>(q_unsigned) * static_cast<int32_t>(k_signed);
        }
        float correction = 128.0f * static_cast<float>(K_q8[b].sum_qs);
        float jit_block = (static_cast<float>(jit_acc) - correction) * d_q * d_k;
        jit_dot_no_scale += jit_block;

        std::cerr << "Block " << b << ": ref_acc=" << ref_acc << " jit_acc=" << jit_acc
                  << " correction=" << correction << " d_q*d_k=" << (d_q * d_k)
                  << " ref_block=" << ref_block << " jit_block=" << jit_block << std::endl;
    }
    std::cerr << "Total: ref_dot=" << ref_dot << " jit_dot=" << jit_dot_no_scale << std::endl;
    std::cerr << "After scale: ref=" << (ref_dot * scale) << " jit=" << (jit_dot_no_scale * scale) << std::endl;

    // Run C++ reference
    std::vector<Q8_1Block> ref_output_q8(M * num_blocks);
    std::memset(ref_output_q8.data(), 0, ref_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), ref_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> ref_output(M * D);
    dequant_q8_1_to_fp32(ref_output_q8.data(), M, D, ref_output.data());

    // Run JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    std::cerr << "\n=== Output Comparison ===" << std::endl;
    std::cerr << "Ref output[0..7]: ";
    for (int i = 0; i < 8; ++i)
        std::cerr << ref_output[i] << " ";
    std::cerr << std::endl;
    std::cerr << "JIT output[0..7]: ";
    for (int i = 0; i < 8; ++i)
        std::cerr << jit_output[i] << " ";
    std::cerr << std::endl;

    auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);
    std::cerr << "\nCosine: " << cosine << " Rel L2: " << rel_l2 << std::endl;

    EXPECT_GE(cosine, 0.99) << "JIT vs C++ reference: Cosine too low";
    EXPECT_LE(rel_l2, 0.05) << "JIT vs C++ reference: L2 error too high";
}

// =============================================================================
// JIT Kernel Tests (specific head_dim)
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_SingleToken)
{
    run_jit_kernel_test(64, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_SmallBatch)
{
    run_jit_kernel_test(64, 4, 16);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D64_MediumBatch)
{
    run_jit_kernel_test(64, 8, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D128_SingleToken)
{
    run_jit_kernel_test(128, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D128_SmallBatch)
{
    run_jit_kernel_test(128, 4, 16);
}

// Test D=256 (8 blocks) - validates dynamic stack spilling for arbitrary head_dim
TEST_F(Test__Q8_1_FusedAttention, JIT_D256_SingleToken)
{
    run_jit_kernel_test(256, 1, 32);
}

TEST_F(Test__Q8_1_FusedAttention, JIT_D256_SmallBatch)
{
    run_jit_kernel_test(256, 4, 16);
}

// Test D=192 (6 blocks) - odd number of spilled blocks
TEST_F(Test__Q8_1_FusedAttention, JIT_D192_SingleToken)
{
    run_jit_kernel_test(192, 1, 32);
}

// Test D=512 (16 blocks) - stress test for large head dimensions
TEST_F(Test__Q8_1_FusedAttention, JIT_D512_SingleToken)
{
    run_jit_kernel_test(512, 1, 16); // Smaller N for faster test
}

// Diagnostic test: Compare JIT vs C++ Q8_1 reference (not FP32)
// This isolates JIT correctness from quantization error
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_D128_Batch)
{
    const int M = 4;
    const int N = 16;
    const int D = 128;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Generate random FP32 data
    std::mt19937 gen(98765);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize inputs to Q8_1
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // Run C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // Run JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Compare JIT vs C++ reference (both use Q8_1, should match very closely)
    auto [rel_l2_jit_cpp, cosine_jit_cpp] = compute_metrics(jit_output.data(), cpp_output.data(), M * D);

    // Also compute vs FP32 for context
    std::vector<float> fp32_output(M * D);
    compute_fp32_reference(M, N, D, Q_fp32.data(), K_fp32.data(), V_fp32.data(), scale, fp32_output.data());
    auto [rel_l2_jit_fp32, cosine_jit_fp32] = compute_metrics(jit_output.data(), fp32_output.data(), M * D);
    auto [rel_l2_cpp_fp32, cosine_cpp_fp32] = compute_metrics(cpp_output.data(), fp32_output.data(), M * D);

    if (rank_ == 0)
    {
        std::cerr << "\n=== JIT vs C++ vs FP32 Comparison (D=" << D << " M=" << M << " N=" << N << ") ===" << std::endl;
        std::cerr << "  JIT vs C++ Ref:  Cosine=" << std::fixed << std::setprecision(6) << cosine_jit_cpp
                  << "  RelL2=" << rel_l2_jit_cpp << std::endl;
        std::cerr << "  JIT vs FP32:     Cosine=" << cosine_jit_fp32 << "  RelL2=" << rel_l2_jit_fp32 << std::endl;
        std::cerr << "  C++ vs FP32:     Cosine=" << cosine_cpp_fp32 << "  RelL2=" << rel_l2_cpp_fp32 << std::endl;

        // Per-row analysis
        std::cerr << "\n  Per-row analysis:" << std::endl;
        for (int m = 0; m < M; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "    Row " << m << ": Cosine=" << row_cos << " RelL2=" << row_l2 << std::endl;
        }
    }

    // JIT should match C++ reference very closely (both Q8_1)
    // Note: RelL2 tolerance relaxed from 0.001 to 0.005 after fixing the zmm22/zmm23
    // clobber bug. The JIT uses faster exp approximation which introduces small errors.
    EXPECT_GE(cosine_jit_cpp, 0.9999) << "JIT vs C++ reference should be near-identical";
    EXPECT_LE(rel_l2_jit_cpp, 0.005) << "JIT vs C++ reference L2 should be small";
}

// Debug test for D=128 to identify which blocks are incorrect
TEST_F(Test__Q8_1_FusedAttention, JIT_D128_BlockAnalysis)
{
    const int M = 1;
    const int N = 4; // Small N for simpler analysis
    const int D = 128;
    const int num_blocks = D / 32; // 4 blocks
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use uniform value for simpler analysis
    std::vector<float> Q_fp32(M * D, 0.1f);
    std::vector<float> K_fp32(N * D, 0.1f);
    std::vector<float> V_fp32(N * D, 0.1f); // All same value

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run reference
    std::vector<Q8_1Block> ref_output_q8(M * num_blocks);
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), ref_output_q8.data(),
        M, N, D, num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> ref_output(M * D);
    dequant_q8_1_to_fp32(ref_output_q8.data(), M, D, ref_output.data());

    // Run JIT
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.K_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.V_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.output_stride_bytes = num_blocks * sizeof(Q8_1Block);
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Analyze per-block
    std::cerr << "\n=== D=128 Block-by-Block Analysis ===" << std::endl;
    for (int b = 0; b < num_blocks; ++b)
    {
        std::cerr << "\nBlock " << b << " (elements " << (b * 32) << "-" << (b * 32 + 31) << "):" << std::endl;
        std::cerr << "  Ref[" << b * 32 << "..+3]: "
                  << ref_output[b * 32] << " " << ref_output[b * 32 + 1] << " "
                  << ref_output[b * 32 + 2] << " " << ref_output[b * 32 + 3] << std::endl;
        std::cerr << "  JIT[" << b * 32 << "..+3]: "
                  << jit_output[b * 32] << " " << jit_output[b * 32 + 1] << " "
                  << jit_output[b * 32 + 2] << " " << jit_output[b * 32 + 3] << std::endl;

        // Compute per-block metrics
        float block_sum_sq_diff = 0, block_sum_sq_ref = 0;
        for (int i = 0; i < 32; ++i)
        {
            float diff = jit_output[b * 32 + i] - ref_output[b * 32 + i];
            block_sum_sq_diff += diff * diff;
            block_sum_sq_ref += ref_output[b * 32 + i] * ref_output[b * 32 + i];
        }
        float block_rel_l2 = std::sqrt(block_sum_sq_diff / (block_sum_sq_ref + 1e-10f));
        std::cerr << "  Block Rel L2: " << block_rel_l2 << std::endl;
    }

    // Overall metrics
    auto [rel_l2, cosine] = compute_metrics(jit_output.data(), ref_output.data(), M * D);
    std::cerr << "\nOverall Cosine: " << cosine << " Rel L2: " << rel_l2 << std::endl;

    EXPECT_GE(cosine, 0.95) << "D=128 JIT: Cosine too low";
}

// =============================================================================
// Numerical Properties Tests
// =============================================================================

TEST_F(Test__Q8_1_FusedAttention, SoftmaxRowSumProperty)
{
    // Verify that attention weights sum to 1.0 (approximately)
    // by checking that output is bounded

    const int M = 4;
    const int N = 16;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 gen(99);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> Q_fp32(M * D), K_fp32(N * D), V_fp32(N * D);
    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Quantize
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    std::vector<Q8_1Block> output_q8(M * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run fused kernel
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    // Dequantize and check bounds
    std::vector<float> output(M * D);
    dequant_q8_1_to_fp32(output_q8.data(), M, D, output.data());

    // Find V value range
    float v_min = *std::min_element(V_fp32.begin(), V_fp32.end());
    float v_max = *std::max_element(V_fp32.begin(), V_fp32.end());

    // Output should be within V range (weighted average property)
    for (int i = 0; i < M * D; ++i)
    {
        // Allow some slack for quantization noise
        EXPECT_GE(output[i], v_min - 0.5f)
            << "Output[" << i << "] = " << output[i] << " below V_min - 0.5 = " << (v_min - 0.5f);
        EXPECT_LE(output[i], v_max + 0.5f)
            << "Output[" << i << "] = " << output[i] << " above V_max + 0.5 = " << (v_max + 0.5f);
    }
}

TEST_F(Test__Q8_1_FusedAttention, IdentityKVTest)
{
    // When K = Q and V is identity-like, output should roughly equal input
    // This tests the softmax concentration property

    const int M = 2;
    const int N = 2;
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Create Q with distinct patterns
    std::vector<float> Q_fp32(M * D, 0.0f);
    for (int i = 0; i < D; ++i)
    {
        Q_fp32[i] = 1.0f;      // First row: all 1s
        Q_fp32[D + i] = -1.0f; // Second row: all -1s
    }

    // K = Q (same as queries)
    std::vector<float> K_fp32 = Q_fp32;

    // V = identity pattern
    std::vector<float> V_fp32(N * D, 0.0f);
    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 1.0f;      // First V row
        V_fp32[D + i] = -1.0f; // Second V row
    }

    // Quantize
    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);
    std::vector<Q8_1Block> output_q8(M * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    // Run fused kernel
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    // Dequantize
    std::vector<float> output(M * D);
    dequant_q8_1_to_fp32(output_q8.data(), M, D, output.data());

    // When Q[0] · K[0] >> Q[0] · K[1], softmax should concentrate on first V row
    // So output[0] should be close to V[0] = all 1s
    float avg_row0 = 0.0f;
    for (int i = 0; i < D; ++i)
    {
        avg_row0 += output[i];
    }
    avg_row0 /= D;

    // Should be closer to 1.0 than -1.0
    EXPECT_GT(avg_row0, 0.0f)
        << "First output row average should be positive (close to V[0]=1), got " << avg_row0;

    float avg_row1 = 0.0f;
    for (int i = 0; i < D; ++i)
    {
        avg_row1 += output[D + i];
    }
    avg_row1 /= D;

    // Should be closer to -1.0 than 1.0
    EXPECT_LT(avg_row1, 0.0f)
        << "Second output row average should be negative (close to V[1]=-1), got " << avg_row1;
}

// Diagnostic: M=2 N=1 test to isolate row-by-row correctness
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M2_N1)
{
    const int M = 2;
    const int N = 1; // Single K/V position - no softmax complexity
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    std::mt19937 gen(55555);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    std::vector<float> Q_fp32(M * D);
    std::vector<float> K_fp32(N * D);
    std::vector<float> V_fp32(N * D);

    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    if (rank_ == 0)
    {
        std::cerr << "\n=== M=2 N=1 per-row analysis ===" << std::endl;
        for (int m = 0; m < M; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "  Row " << m << ": Cosine=" << std::fixed << std::setprecision(6)
                      << row_cos << " RelL2=" << row_l2 << std::endl;

            // Print first few values
            std::cerr << "    CPP[" << m << "][0..7]: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << cpp_output[m * D + i] << " ";
            std::cerr << std::endl;
            std::cerr << "    JIT[" << m << "][0..7]: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << jit_output[m * D + i] << " ";
            std::cerr << std::endl;
        }
    }

    // Both rows should match very closely
    for (int m = 0; m < M; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

// Diagnostic: M=2 N=2 test - minimum softmax case
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M2_N2)
{
    const int M = 2;
    const int N = 2; // Two K/V positions - softmax is meaningful
    const int D = 64;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Use deterministic values for easier debugging
    // Q: row 0 = [1,0,0,...], row 1 = [0,1,0,...]
    // K: row 0 = [1,0,0,...], row 1 = [0,1,0,...]
    // V: row 0 = [1,1,1,...], row 1 = [2,2,2,...]
    std::vector<float> Q_fp32(M * D, 0.0f);
    std::vector<float> K_fp32(N * D, 0.0f);
    std::vector<float> V_fp32(N * D, 0.0f);

    // Q[0] and Q[1] both have 1.0 at position 0 (IDENTICAL rows)
    Q_fp32[0] = 1.0f;
    Q_fp32[D + 0] = 1.0f; // Same as Q[0]!

    // K[0] has 1.0 at position 0, K[1] has 1.0 at position 1
    K_fp32[0] = 1.0f;
    K_fp32[D + 1] = 1.0f;

    // V[0] = all 0s, V[1] = all 1s (distinct values to check V selection)
    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 0.0f;     // V[0] = 0
        V_fp32[D + i] = 1.0f; // V[1] = 1
    }

    std::vector<Q8_1Block> Q_q8(M * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), M, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // C++ Q8_1 reference
    std::vector<Q8_1Block> cpp_output_q8(M * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        M, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);

    std::vector<float> cpp_output(M * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), M, D, cpp_output.data());

    // JIT kernel
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    std::vector<Q8_1Block> jit_output_q8(M * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    FusedQ8_1AttentionParams params;
    params.Q = Q_q8.data();
    params.K = K_q8.data();
    params.V = V_q8.data();
    params.output = jit_output_q8.data();
    params.M = M;
    params.N = N;
    params.head_dim = D;
    params.Q_stride_bytes = stride_bytes;
    params.K_stride_bytes = stride_bytes;
    params.V_stride_bytes = stride_bytes;
    params.output_stride_bytes = stride_bytes;
    params.scale = scale;
    params.mask = nullptr;
    params.mask_stride = 0;

    kernel_func(&params);

    std::vector<float> jit_output(M * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), M, D, jit_output.data());

    // Both rows should match very closely
    for (int m = 0; m < M; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

// Diagnostic: Test M=2 but call kernel TWICE with M=1 each
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_CppReference_M1_TwoCalls)
{
    const int D = 64;
    const int N = 2;
    const int num_blocks = D / 32;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Same setup as other tests
    std::vector<float> Q_fp32(2 * D, 0.0f);
    std::vector<float> K_fp32(N * D, 0.0f);
    std::vector<float> V_fp32(N * D, 0.0f);

    Q_fp32[0] = 1.0f;     // Q[0]: nonzero at position 0
    Q_fp32[D + 1] = 1.0f; // Q[1]: nonzero at position 1
    K_fp32[0] = 1.0f;     // K[0]: nonzero at position 0
    K_fp32[D + 1] = 1.0f; // K[1]: nonzero at position 1

    for (int i = 0; i < D; ++i)
    {
        V_fp32[i] = 0.0f;
        V_fp32[D + i] = 1.0f;
    }

    std::vector<Q8_1Block> Q_q8(2 * num_blocks);
    std::vector<Q8_1Block> K_q8(N * num_blocks);
    std::vector<Q8_1Block> V_q8(N * num_blocks);

    quantize_fp32_to_q8_1(Q_fp32.data(), 2, D, Q_q8.data());
    quantize_fp32_to_q8_1(K_fp32.data(), N, D, K_q8.data());
    quantize_fp32_to_q8_1(V_fp32.data(), N, D, V_q8.data());

    const int stride_bytes = num_blocks * static_cast<int>(sizeof(Q8_1Block));

    // Create SINGLE JIT kernel instance
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(D, false);
    auto kernel_func = jit_kernel.get_kernel();

    // C++ reference for both rows
    std::vector<Q8_1Block> cpp_output_q8(2 * num_blocks);
    std::memset(cpp_output_q8.data(), 0, cpp_output_q8.size() * sizeof(Q8_1Block));
    fused_q8_1_attention_reference(
        Q_q8.data(), K_q8.data(), V_q8.data(), cpp_output_q8.data(),
        2, N, D,
        num_blocks, num_blocks, num_blocks, num_blocks,
        scale, nullptr, 0);
    std::vector<float> cpp_output(2 * D);
    dequant_q8_1_to_fp32(cpp_output_q8.data(), 2, D, cpp_output.data());

    // Call JIT kernel TWICE with M=1 each time
    std::vector<Q8_1Block> jit_output_q8(2 * num_blocks);
    std::memset(jit_output_q8.data(), 0, jit_output_q8.size() * sizeof(Q8_1Block));

    // First call: Q[0]
    FusedQ8_1AttentionParams params0;
    params0.Q = Q_q8.data(); // Start at Q[0]
    params0.K = K_q8.data();
    params0.V = V_q8.data();
    params0.output = jit_output_q8.data(); // Output to row 0
    params0.M = 1;
    params0.N = N;
    params0.head_dim = D;
    params0.Q_stride_bytes = stride_bytes;
    params0.K_stride_bytes = stride_bytes;
    params0.V_stride_bytes = stride_bytes;
    params0.output_stride_bytes = stride_bytes;
    params0.scale = scale;
    params0.mask = nullptr;
    params0.mask_stride = 0;

    kernel_func(&params0);

    // Second call: Q[1] (using same kernel instance!)
    FusedQ8_1AttentionParams params1;
    params1.Q = Q_q8.data() + num_blocks; // Start at Q[1]
    params1.K = K_q8.data();
    params1.V = V_q8.data();
    params1.output = jit_output_q8.data() + num_blocks; // Output to row 1
    params1.M = 1;
    params1.N = N;
    params1.head_dim = D;
    params1.Q_stride_bytes = stride_bytes;
    params1.K_stride_bytes = stride_bytes;
    params1.V_stride_bytes = stride_bytes;
    params1.output_stride_bytes = stride_bytes;
    params1.scale = scale;
    params1.mask = nullptr;
    params1.mask_stride = 0;

    kernel_func(&params1);

    std::vector<float> jit_output(2 * D);
    dequant_q8_1_to_fp32(jit_output_q8.data(), 2, D, jit_output.data());

    if (rank_ == 0)
    {
        std::cerr << "\n=== Two separate M=1 calls ===" << std::endl;
        for (int m = 0; m < 2; ++m)
        {
            auto [row_l2, row_cos] = compute_metrics(
                jit_output.data() + m * D, cpp_output.data() + m * D, D);
            std::cerr << "  Row " << m << ": Cosine=" << std::fixed << std::setprecision(6)
                      << row_cos << " RelL2=" << row_l2 << std::endl;
            std::cerr << "    CPP: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << cpp_output[m * D + i] << " ";
            std::cerr << std::endl;
            std::cerr << "    JIT: ";
            for (int i = 0; i < 8; ++i)
                std::cerr << jit_output[m * D + i] << " ";
            std::cerr << std::endl;
        }
    }

    for (int m = 0; m < 2; ++m)
    {
        auto [row_l2, row_cos] = compute_metrics(
            jit_output.data() + m * D, cpp_output.data() + m * D, D);
        EXPECT_GE(row_cos, 0.9999) << "Row " << m << " should match";
    }
}

/**
 * @test JIT_vs_FP32_Strided_Qwen05B_Layout
 * @brief Tests with the exact stride layout used by Qwen 2.5 0.5B pipeline
 *
 * This test recreates the exact memory layout used in the real pipeline:
 * - n_heads = 14, n_kv_heads = 2, head_dim = 64
 * - Q stride = 14 * 2 blocks = 28 blocks per row
 * - K/V stride = 2 * 2 blocks = 4 blocks per row
 * - Output stride = 14 * 2 blocks = 28 blocks per row
 *
 * The JIT kernel receives pointers to head-specific data within the larger buffer.
 * Tests verify correctness when iterating through heads with these strides.
 */
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_FP32_Strided_Qwen05B_Layout)
{
    // Qwen 2.5 0.5B parameters
    const int n_heads = 14;
    const int n_kv_heads = 2;
    const int head_dim = 64;
    const int seq_len = 9; // Test prompt length: "The quick brown fox jumps over the lazy dog"
    const int kv_len = 9;

    const int heads_per_kv = n_heads / n_kv_heads;
    const int head_dim_blocks = head_dim / 32;

    // Multi-head tensor strides (as used in pipeline)
    const int q_blocks_per_row = n_heads * head_dim_blocks;    // 14 * 2 = 28
    const int k_blocks_per_row = n_kv_heads * head_dim_blocks; // 2 * 2 = 4
    const int out_blocks_per_row = n_heads * head_dim_blocks;  // 14 * 2 = 28

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Allocate multi-head buffers
    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(kv_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_len * n_kv_heads * head_dim);
    std::vector<float> ref_output(seq_len * n_heads * head_dim, 0.0f);

    // Random init
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (auto &x : Q_fp32)
        x = dist(gen);
    for (auto &x : K_fp32)
        x = dist(gen);
    for (auto &x : V_fp32)
        x = dist(gen);

    // Compute FP32 reference for ALL heads
    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        // Extract head-specific data (strided access)
        std::vector<float> Q_h(seq_len * head_dim);
        std::vector<float> K_h(kv_len * head_dim);
        std::vector<float> V_h(kv_len * head_dim);

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                Q_h[s * head_dim + d] = Q_fp32[s * n_heads * head_dim + h * head_dim + d];
            }
        }
        for (int k = 0; k < kv_len; ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                K_h[k * head_dim + d] = K_fp32[k * n_kv_heads * head_dim + kv_h * head_dim + d];
                V_h[k * head_dim + d] = V_fp32[k * n_kv_heads * head_dim + kv_h * head_dim + d];
            }
        }

        // Compute reference attention for this head
        std::vector<float> head_output(seq_len * head_dim);
        compute_fp32_reference(seq_len, kv_len, head_dim, Q_h.data(), K_h.data(), V_h.data(), scale, head_output.data());

        // Copy back to output (strided)
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                ref_output[s * n_heads * head_dim + h * head_dim + d] = head_output[s * head_dim + d];
            }
        }
    }

    // Quantize inputs to Q8_1 with multi-head layout
    std::vector<Q8_1Block> Q_q8(seq_len * q_blocks_per_row);
    std::vector<Q8_1Block> K_q8(kv_len * k_blocks_per_row);
    std::vector<Q8_1Block> V_q8(kv_len * k_blocks_per_row);
    std::vector<Q8_1Block> output_q8(seq_len * out_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Quantize Q (multi-head layout)
    for (int s = 0; s < seq_len; ++s)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            const float *fp32_head = Q_fp32.data() + s * n_heads * head_dim + h * head_dim;
            Q8_1Block *q8_head = Q_q8.data() + s * q_blocks_per_row + h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Quantize K (multi-head layout)
    for (int k = 0; k < kv_len; ++k)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            const float *fp32_head = K_fp32.data() + k * n_kv_heads * head_dim + kv_h * head_dim;
            Q8_1Block *q8_head = K_q8.data() + k * k_blocks_per_row + kv_h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Quantize V (multi-head layout)
    for (int k = 0; k < kv_len; ++k)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            const float *fp32_head = V_fp32.data() + k * n_kv_heads * head_dim + kv_h * head_dim;
            Q8_1Block *q8_head = V_q8.data() + k * k_blocks_per_row + kv_h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Run JIT kernel for each head with proper striding
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(head_dim, false);
    auto kernel_func = jit_kernel.get_kernel();

    const int q_stride_bytes = q_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
    const int k_stride_bytes = k_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
    const int out_stride_bytes = out_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        // Head-specific pointers (same as pipeline)
        const Q8_1Block *Q_h = Q_q8.data() + h * head_dim_blocks;
        const Q8_1Block *K_h = K_q8.data() + kv_h * head_dim_blocks;
        const Q8_1Block *V_h = V_q8.data() + kv_h * head_dim_blocks;
        Q8_1Block *out_h = output_q8.data() + h * head_dim_blocks;

        FusedQ8_1AttentionParams params;
        params.Q = Q_h;
        params.K = K_h;
        params.V = V_h;
        params.output = out_h;
        params.M = seq_len;
        params.N = kv_len;
        params.head_dim = head_dim;
        params.Q_stride_bytes = q_stride_bytes;
        params.K_stride_bytes = k_stride_bytes;
        params.V_stride_bytes = k_stride_bytes;
        params.output_stride_bytes = out_stride_bytes;
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride = 0;

        kernel_func(&params);
    }

    // Dequantize output
    std::vector<float> jit_output(seq_len * n_heads * head_dim);
    for (int s = 0; s < seq_len; ++s)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            const Q8_1Block *q8_head = output_q8.data() + s * out_blocks_per_row + h * head_dim_blocks;
            float *fp32_head = jit_output.data() + s * n_heads * head_dim + h * head_dim;
            dequant_q8_1_to_fp32(q8_head, 1, head_dim, fp32_head);
        }
    }

    // Compare per-head
    if (rank_ == 0)
    {
        std::cerr << "\n=== Strided Layout Test (Qwen 0.5B parameters) ===" << std::endl;
        std::cerr << "  n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                  << " head_dim=" << head_dim << " seq_len=" << seq_len << std::endl;
        std::cerr << "  Q stride=" << q_blocks_per_row << " K stride=" << k_blocks_per_row << " blocks/row" << std::endl;
    }

    double min_cosine = 1.0;
    double max_l2 = 0.0;
    int worst_head = -1;

    for (int h = 0; h < n_heads; ++h)
    {
        // Extract head output for comparison
        std::vector<float> head_ref(seq_len * head_dim);
        std::vector<float> head_jit(seq_len * head_dim);

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                head_ref[s * head_dim + d] = ref_output[s * n_heads * head_dim + h * head_dim + d];
                head_jit[s * head_dim + d] = jit_output[s * n_heads * head_dim + h * head_dim + d];
            }
        }

        auto [rel_l2, cosine] = compute_metrics(head_jit.data(), head_ref.data(), seq_len * head_dim);

        if (cosine < min_cosine)
        {
            min_cosine = cosine;
            worst_head = h;
        }
        max_l2 = std::max(max_l2, rel_l2);

        if (rank_ == 0 && (cosine < 0.98 || h < 3))
        {
            std::cerr << "  Head " << h << " (kv_h=" << (h / heads_per_kv) << "): "
                      << "Cosine=" << std::fixed << std::setprecision(6) << cosine
                      << " RelL2=" << rel_l2 << std::endl;
        }
    }

    if (rank_ == 0)
    {
        std::cerr << "  Worst head: " << worst_head << " (cosine=" << min_cosine << ")" << std::endl;
        std::cerr << "  Max L2: " << max_l2 << std::endl;
    }

    // Q8_1 attention should achieve >=0.98 cosine sim vs FP32
    EXPECT_GE(min_cosine, 0.98)
        << "Strided attention: worst head " << worst_head << " cosine " << min_cosine << " below 0.98";
    EXPECT_LE(max_l2, 0.15)
        << "Strided attention: max L2 error " << max_l2 << " above 0.15";
}

/**
 * @test JIT_vs_FP32_Strided_Qwen05B_CausalMask_RealWeights
 * @brief Tests with causal masking using REAL model weights from GGUF
 *
 * This test loads the actual Qwen 2.5 0.5B model, computes QKV projections
 * from real embeddings, and tests the fused attention kernel with realistic
 * data. This replicates the exact data distribution seen in actual inference.
 *
 * Uses the same model as the E2E parity tests: models/qwen2.5-0.5b-instruct-q4_0.gguf
 */
TEST_F(Test__Q8_1_FusedAttention, JIT_vs_FP32_Strided_Qwen05B_CausalMask)
{
    // ========================================================================
    // Load real model weights
    // ========================================================================
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    auto mpi_ctx = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
    auto model_ctx = ModelContext::create(model_path, mpi_ctx);
    if (!model_ctx)
    {
        GTEST_SKIP() << "Model not found: " << model_path;
    }

    // Configure weight sharding from Qwen2 schema (required before getWeightForDevice())
    Qwen2SchemaFactory schema_factory;
    model_ctx->weightManager()->setWeightShardingConfig(schema_factory.getWeightShardingConfig());

    const auto &metadata = model_ctx->model();

    // Qwen 2.5 0.5B parameters (from model metadata)
    const int n_heads = static_cast<int>(metadata.head_count);       // 14
    const int n_kv_heads = static_cast<int>(metadata.head_count_kv); // 2
    const int d_model = static_cast<int>(metadata.embedding_length); // 896
    const int head_dim = d_model / n_heads;                          // 896/14 = 64

    // Test tokens: "Hello, world" (same as E2E test)
    const std::vector<int> test_tokens = {9906, 11, 1879};
    const int seq_len = static_cast<int>(test_tokens.size());
    const int kv_len = seq_len;

    const int heads_per_kv = n_heads / n_kv_heads;
    const int head_dim_blocks = head_dim / 32;

    // Multi-head tensor strides (as used in pipeline)
    const int q_blocks_per_row = n_heads * head_dim_blocks;    // 14 * 2 = 28
    const int k_blocks_per_row = n_kv_heads * head_dim_blocks; // 2 * 2 = 4
    const int out_blocks_per_row = n_heads * head_dim_blocks;  // 14 * 2 = 28

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Create causal mask
    std::vector<float> causal_mask(seq_len * kv_len);
    create_causal_mask(seq_len, kv_len, causal_mask.data());

    // ========================================================================
    // Load embedding and QKV projection weights from layer 0
    // ========================================================================
    auto embd_weight = model_ctx->getWeightForDevice("token_embd.weight");
    auto wq = model_ctx->getWeightForDevice("blk.0.attn_q.weight");
    auto wk = model_ctx->getWeightForDevice("blk.0.attn_k.weight");
    auto wv = model_ctx->getWeightForDevice("blk.0.attn_v.weight");

    if (!embd_weight || !wq || !wk || !wv)
    {
        GTEST_SKIP() << "Failed to load model weights";
    }

    if (rank_ == 0)
    {
        std::cerr << "\n=== Loading real model weights ===" << std::endl;
        std::cerr << "  Model: " << model_path << std::endl;
        std::cerr << "  d_model=" << d_model << " n_heads=" << n_heads
                  << " n_kv_heads=" << n_kv_heads << " head_dim=" << head_dim << std::endl;
        std::cerr << "  Embedding shape: [" << embd_weight->shape()[0] << ", " << embd_weight->shape()[1] << "]" << std::endl;
        std::cerr << "  Embedding type: " << static_cast<int>(embd_weight->native_type()) << std::endl;
        std::cerr << "  Wq shape: [" << wq->shape()[0] << ", " << wq->shape()[1] << "]" << std::endl;
        std::cerr << "  Wq type: " << static_cast<int>(wq->native_type()) << std::endl;
        std::cerr << "  Wk shape: [" << wk->shape()[0] << ", " << wk->shape()[1] << "]" << std::endl;
        std::cerr << "  Wk type: " << static_cast<int>(wk->native_type()) << std::endl;
        std::cerr << "  Wv shape: [" << wv->shape()[0] << ", " << wv->shape()[1] << "]" << std::endl;
        std::cerr << "  Wv type: " << static_cast<int>(wv->native_type()) << std::endl;
    }

    // ========================================================================
    // Compute embeddings for test tokens (row-wise dequant to avoid 544MB alloc)
    // ========================================================================
    std::vector<float> embeddings(seq_len * d_model);
    for (int t = 0; t < seq_len; ++t)
    {
        int token_id = test_tokens[t];
        // Use row-wise dequantization - only dequant the rows we need
        embd_weight->to_fp32_row(static_cast<size_t>(token_id), &embeddings[t * d_model]);
    }

    // ========================================================================
    // Compute QKV projections using GEMM kernels (weights are already packed)
    // ========================================================================
    std::vector<float> Q_fp32(seq_len * n_heads * head_dim);
    std::vector<float> K_fp32(kv_len * n_kv_heads * head_dim);
    std::vector<float> V_fp32(kv_len * n_kv_heads * head_dim);
    std::vector<float> ref_output(seq_len * n_heads * head_dim, 0.0f);

    // Get cached GEMM engines (weights already packed during getWeightForDevice())
    auto *gemm_q = getPreparedKernel(wq.get(), DeviceId::cpu());
    auto *gemm_k = getPreparedKernel(wk.get(), DeviceId::cpu());
    auto *gemm_v = getPreparedKernel(wv.get(), DeviceId::cpu());

    if (!gemm_q || !gemm_k || !gemm_v)
    {
        GTEST_SKIP() << "Failed to create GEMM kernels for weights";
    }

    if (rank_ == 0)
        std::cerr << "  Computing Q projection via GEMM kernel..." << std::endl;

    // Q = embeddings @ Wq^T  ->  [seq_len, d_model] @ [n_heads*head_dim, d_model]^T = [seq_len, n_heads*head_dim]
    // m=seq_len, n=n_heads*head_dim, k=d_model
    bool ok_q = gemm_q->multiply(embeddings.data(), Q_fp32.data(),
                                 seq_len, n_heads * head_dim, d_model,
                                 /*transpose_B=*/true);
    ASSERT_TRUE(ok_q) << "Q projection GEMM failed";

    if (rank_ == 0)
        std::cerr << "  Computing K projection via GEMM kernel..." << std::endl;

    // K = embeddings @ Wk^T  ->  [seq_len, d_model] @ [n_kv_heads*head_dim, d_model]^T = [seq_len, n_kv_heads*head_dim]
    bool ok_k = gemm_k->multiply(embeddings.data(), K_fp32.data(),
                                 seq_len, n_kv_heads * head_dim, d_model,
                                 /*transpose_B=*/true);
    ASSERT_TRUE(ok_k) << "K projection GEMM failed";

    if (rank_ == 0)
        std::cerr << "  Computing V projection via GEMM kernel..." << std::endl;

    // V = embeddings @ Wv^T  ->  [seq_len, d_model] @ [n_kv_heads*head_dim, d_model]^T = [seq_len, n_kv_heads*head_dim]
    bool ok_v = gemm_v->multiply(embeddings.data(), V_fp32.data(),
                                 seq_len, n_kv_heads * head_dim, d_model,
                                 /*transpose_B=*/true);
    ASSERT_TRUE(ok_v) << "V projection GEMM failed";

    if (rank_ == 0)
    {
        // Print Q/K/V stats
        auto print_stats = [](const std::vector<float> &v, const char *name)
        {
            float min_v = *std::min_element(v.begin(), v.end());
            float max_v = *std::max_element(v.begin(), v.end());
            double sum = std::accumulate(v.begin(), v.end(), 0.0);
            std::cerr << "  " << name << ": min=" << min_v << " max=" << max_v
                      << " mean=" << (sum / v.size()) << std::endl;
        };
        print_stats(Q_fp32, "Q");
        print_stats(K_fp32, "K");
        print_stats(V_fp32, "V");
    }

    // Compute FP32 reference for ALL heads WITH causal mask
    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        // Extract head-specific data (strided access)
        std::vector<float> Q_h(seq_len * head_dim);
        std::vector<float> K_h(kv_len * head_dim);
        std::vector<float> V_h(kv_len * head_dim);

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                Q_h[s * head_dim + d] = Q_fp32[s * n_heads * head_dim + h * head_dim + d];
            }
        }
        for (int k = 0; k < kv_len; ++k)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                K_h[k * head_dim + d] = K_fp32[k * n_kv_heads * head_dim + kv_h * head_dim + d];
                V_h[k * head_dim + d] = V_fp32[k * n_kv_heads * head_dim + kv_h * head_dim + d];
            }
        }

        // Compute reference attention for this head WITH causal mask
        std::vector<float> head_output(seq_len * head_dim);
        compute_fp32_reference(seq_len, kv_len, head_dim,
                               Q_h.data(), K_h.data(), V_h.data(), scale, head_output.data(),
                               causal_mask.data(), kv_len);

        // Copy back to output (strided)
        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                ref_output[s * n_heads * head_dim + h * head_dim + d] = head_output[s * head_dim + d];
            }
        }
    }

    // Quantize inputs to Q8_1 with multi-head layout
    std::vector<Q8_1Block> Q_q8(seq_len * q_blocks_per_row);
    std::vector<Q8_1Block> K_q8(kv_len * k_blocks_per_row);
    std::vector<Q8_1Block> V_q8(kv_len * k_blocks_per_row);
    std::vector<Q8_1Block> output_q8(seq_len * out_blocks_per_row);
    std::memset(output_q8.data(), 0, output_q8.size() * sizeof(Q8_1Block));

    // Quantize Q (multi-head layout)
    for (int s = 0; s < seq_len; ++s)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            const float *fp32_head = Q_fp32.data() + s * n_heads * head_dim + h * head_dim;
            Q8_1Block *q8_head = Q_q8.data() + s * q_blocks_per_row + h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Quantize K (multi-head layout)
    for (int k = 0; k < kv_len; ++k)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            const float *fp32_head = K_fp32.data() + k * n_kv_heads * head_dim + kv_h * head_dim;
            Q8_1Block *q8_head = K_q8.data() + k * k_blocks_per_row + kv_h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Quantize V (multi-head layout)
    for (int k = 0; k < kv_len; ++k)
    {
        for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
        {
            const float *fp32_head = V_fp32.data() + k * n_kv_heads * head_dim + kv_h * head_dim;
            Q8_1Block *q8_head = V_q8.data() + k * k_blocks_per_row + kv_h * head_dim_blocks;
            quantize_fp32_to_q8_1(fp32_head, 1, head_dim, q8_head);
        }
    }

    // Run JIT kernel for each head with proper striding AND causal mask
    QuantisedAttentionJit_Q8_1_Fused jit_kernel(head_dim, false);
    auto kernel_func = jit_kernel.get_kernel();

    const int q_stride_bytes = q_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
    const int k_stride_bytes = k_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));
    const int out_stride_bytes = out_blocks_per_row * static_cast<int>(sizeof(Q8_1Block));

    for (int h = 0; h < n_heads; ++h)
    {
        int kv_h = h / heads_per_kv;

        // Head-specific pointers (same as pipeline)
        const Q8_1Block *Q_h = Q_q8.data() + h * head_dim_blocks;
        const Q8_1Block *K_h = K_q8.data() + kv_h * head_dim_blocks;
        const Q8_1Block *V_h = V_q8.data() + kv_h * head_dim_blocks;
        Q8_1Block *out_h = output_q8.data() + h * head_dim_blocks;

        FusedQ8_1AttentionParams params;
        params.Q = Q_h;
        params.K = K_h;
        params.V = V_h;
        params.output = out_h;
        params.M = seq_len;
        params.N = kv_len;
        params.head_dim = head_dim;
        params.Q_stride_bytes = q_stride_bytes;
        params.K_stride_bytes = k_stride_bytes;
        params.V_stride_bytes = k_stride_bytes;
        params.output_stride_bytes = out_stride_bytes;
        params.scale = scale;
        params.mask = causal_mask.data(); // Causal mask!
        params.mask_stride = kv_len;

        kernel_func(&params);
    }

    // Dequantize output
    std::vector<float> jit_output(seq_len * n_heads * head_dim);
    for (int s = 0; s < seq_len; ++s)
    {
        for (int h = 0; h < n_heads; ++h)
        {
            const Q8_1Block *q8_head = output_q8.data() + s * out_blocks_per_row + h * head_dim_blocks;
            float *fp32_head = jit_output.data() + s * n_heads * head_dim + h * head_dim;
            dequant_q8_1_to_fp32(q8_head, 1, head_dim, fp32_head);
        }
    }

    // Compare per-head
    if (rank_ == 0)
    {
        std::cerr << "\n=== Q8_1 Fused Attention with REAL MODEL WEIGHTS ===" << std::endl;
        std::cerr << "  Model: " << model_path << std::endl;
        std::cerr << "  n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                  << " head_dim=" << head_dim << " seq_len=" << seq_len << std::endl;
        std::cerr << "  Q stride=" << q_blocks_per_row << " K stride=" << k_blocks_per_row << " blocks/row" << std::endl;
    }

    double min_cosine = 1.0;
    double max_l2 = 0.0;
    int worst_head = -1;

    for (int h = 0; h < n_heads; ++h)
    {
        // Extract head output for comparison
        std::vector<float> head_ref(seq_len * head_dim);
        std::vector<float> head_jit(seq_len * head_dim);

        for (int s = 0; s < seq_len; ++s)
        {
            for (int d = 0; d < head_dim; ++d)
            {
                head_ref[s * head_dim + d] = ref_output[s * n_heads * head_dim + h * head_dim + d];
                head_jit[s * head_dim + d] = jit_output[s * n_heads * head_dim + h * head_dim + d];
            }
        }

        auto [rel_l2, cosine] = compute_metrics(head_jit.data(), head_ref.data(), seq_len * head_dim);

        if (cosine < min_cosine)
        {
            min_cosine = cosine;
            worst_head = h;
        }
        max_l2 = std::max(max_l2, rel_l2);

        if (rank_ == 0 && (cosine < 0.98 || h < 3))
        {
            std::cerr << "  Head " << h << " (kv_h=" << (h / heads_per_kv) << "): "
                      << "Cosine=" << std::fixed << std::setprecision(6) << cosine
                      << " RelL2=" << rel_l2 << std::endl;
        }
    }

    if (rank_ == 0)
    {
        std::cerr << "  Worst head: " << worst_head << " (cosine=" << min_cosine << ")" << std::endl;
        std::cerr << "  Max L2: " << max_l2 << std::endl;
    }

    // Q8_1 attention with causal mask should achieve >=0.98 cosine sim vs FP32
    EXPECT_GE(min_cosine, 0.98)
        << "Causal mask attention: worst head " << worst_head << " cosine " << min_cosine << " below 0.98";
    EXPECT_LE(max_l2, 0.15)
        << "Causal mask attention: max L2 error " << max_l2 << " above 0.15";
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
