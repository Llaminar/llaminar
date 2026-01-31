/**
 * @file Test__Q8_1_OnlineSoftmax.cpp
 * @brief Integration tests for Q8_1 OnlineSoftmax kernel correctness
 *
 * Validates the Q8_1 x Q8_1 attention kernel (Q @ K^T with fused softmax)
 * against FP32 reference implementation across various model configurations
 * and sequence lengths.
 *
 * Test configurations match real transformer architectures:
 * - Qwen 0.5B (14 heads, D=64)
 * - Qwen 7B (28 heads, D=128)
 * - Qwen 32B (40 heads, D=128)
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_Q8_1_OnlineSoftmax.h"

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

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
static const TestModelConfig QWEN_32B = {"Qwen-32B", 128};

/**
 * @brief Test fixture for Q8_1 OnlineSoftmax kernel correctness
 */
class Test__Q8_1_OnlineSoftmax : public ::testing::Test
{
protected:
    int rank_ = 0;
    int world_size_ = 1;

    // Correctness thresholds
    static constexpr double MIN_COSINE_SIM = 0.9999; // Must be very close
    static constexpr double MAX_L2_ERROR = 0.02;     // Allow 2% relative error

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    }

    /**
     * @brief Compute FP32 reference: softmax(Q * K^T * scale)
     */
    void compute_reference(
        int M, int N, int K,
        const float *Q, const float *K_data,
        float scale,
        float *C_ref)
    {
        // Compute Q * K^T
        for (int i = 0; i < M; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                float sum = 0.0f;
                for (int kk = 0; kk < K; ++kk)
                {
                    sum += Q[i * K + kk] * K_data[j * K + kk];
                }
                C_ref[i * N + j] = sum * scale;
            }
        }

        // Softmax per row
        for (int i = 0; i < M; ++i)
        {
            float max_val = C_ref[i * N];
            for (int j = 1; j < N; ++j)
            {
                max_val = std::max(max_val, C_ref[i * N + j]);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] = std::exp(C_ref[i * N + j] - max_val);
                sum_exp += C_ref[i * N + j];
            }

            for (int j = 0; j < N; ++j)
            {
                C_ref[i * N + j] /= sum_exp;
            }
        }
    }

    /**
     * @brief Verify kernel output against reference
     * @return pair of (l2_error, cosine_similarity)
     */
    std::pair<double, double> verify_correctness(
        int M, int N,
        const float *C_act, const float *C_ref)
    {
        double sum_sq_diff = 0.0;
        double sum_sq_ref = 0.0;
        double dot_prod = 0.0;
        double norm_act = 0.0;
        double norm_ref = 0.0;

        for (int i = 0; i < M * N; ++i)
        {
            double diff = C_act[i] - C_ref[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += C_ref[i] * C_ref[i];

            dot_prod += C_act[i] * C_ref[i];
            norm_act += C_act[i] * C_act[i];
            norm_ref += C_ref[i] * C_ref[i];
        }

        double l2_error = (sum_sq_ref > 0.0) ? std::sqrt(sum_sq_diff) / std::sqrt(sum_sq_ref) : 0.0;
        double cosine_sim = (norm_act > 0.0 && norm_ref > 0.0)
                                ? dot_prod / (std::sqrt(norm_act) * std::sqrt(norm_ref))
                                : 0.0;

        return {l2_error, cosine_sim};
    }

    /**
     * @brief Run correctness test for given configuration
     */
    void run_correctness_test(const TestModelConfig &model, int seq_len, int kv_len)
    {
        int S = seq_len;
        int KV = kv_len;
        int D = model.head_dim;

        float scale = 1.0f / std::sqrt(static_cast<float>(D));

        int k_blocks = (D + 31) / 32;
        int Q_stride_bytes = k_blocks * sizeof(Q8_1Block);
        int K_stride_bytes = k_blocks * sizeof(Q8_1Block);

        // Generate random FP32 data
        std::mt19937 gen(42 + seq_len * 1000 + kv_len); // Deterministic but varied
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        std::vector<float> Q_fp32(S * D);
        std::vector<float> K_fp32(KV * D);

        for (auto &x : Q_fp32)
            x = dist(gen);
        for (auto &x : K_fp32)
            x = dist(gen);

        // Quantize to Q8_1
        auto Q_tensor = Q8_1Tensor::quantize_from_fp32(Q_fp32.data(), {static_cast<size_t>(S), static_cast<size_t>(D)});
        auto K_tensor = Q8_1Tensor::quantize_from_fp32(K_fp32.data(), {static_cast<size_t>(KV), static_cast<size_t>(D)});

        // Copy to flat layout
        std::vector<Q8_1Block> Q_blocks(static_cast<size_t>(S) * k_blocks);
        std::vector<Q8_1Block> K_blocks_data(static_cast<size_t>(KV) * k_blocks);

        const Q8_1Block *Q_src = Q_tensor->decode_to_q8_1(0, 0);
        const Q8_1Block *K_src = K_tensor->decode_to_q8_1(0, 0);

        for (int s = 0; s < S; ++s)
        {
            for (int kb = 0; kb < k_blocks; ++kb)
            {
                Q_blocks[s * k_blocks + kb] = Q_src[s * k_blocks + kb];
            }
        }

        for (int kv = 0; kv < KV; ++kv)
        {
            for (int kb = 0; kb < k_blocks; ++kb)
            {
                K_blocks_data[kv * k_blocks + kb] = K_src[kv * k_blocks + kb];
            }
        }

        // Output buffer
        std::vector<float> C(S * KV);

        // Create JIT kernels (static for reuse)
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m4(4);
        static QuantisedGemmJit_Q8_1_OnlineSoftmax kernel_m1(1);

        // Build params
        OnlineSoftmaxParams params;
        params.Q = Q_blocks.data();
        params.K = K_blocks_data.data();
        params.C = C.data();
        params.M = S;
        params.N = KV;
        params.K_blocks = k_blocks;
        params.Q_stride_bytes = Q_stride_bytes;
        params.K_stride_bytes = K_stride_bytes;
        params.C_stride_bytes = KV * sizeof(float);
        params.scale = scale;
        params.mask = nullptr;
        params.mask_stride_bytes = 0;

        // Run kernel
        int m_blocking = 4;
        for (int i = 0; i < S; i += m_blocking)
        {
            int current_m = std::min(m_blocking, S - i);

            OnlineSoftmaxParams p = params;
            p.Q = reinterpret_cast<const char *>(params.Q) + i * Q_stride_bytes;
            p.C = params.C + i * KV;
            p.M = current_m;

            if (current_m == 4)
            {
                kernel_m4.get_kernel()(&p);
            }
            else
            {
                auto func = kernel_m1.get_kernel();
                for (int j = 0; j < current_m; ++j)
                {
                    OnlineSoftmaxParams p1 = p;
                    p1.Q = reinterpret_cast<const char *>(p.Q) + j * Q_stride_bytes;
                    p1.C = p.C + j * KV;
                    p1.M = 1;
                    func(&p1);
                }
            }
        }

        // Dequantize for reference computation
        std::vector<float> Q_deq(S * D), K_deq(KV * D);
        Q_tensor->to_fp32(Q_deq.data());
        K_tensor->to_fp32(K_deq.data());

        // Compute reference
        std::vector<float> C_ref(S * KV);
        compute_reference(S, KV, D, Q_deq.data(), K_deq.data(), scale, C_ref.data());

        // Verify
        auto [l2_error, cosine_sim] = verify_correctness(S, KV, C.data(), C_ref.data());

        // Assert correctness
        EXPECT_GE(cosine_sim, MIN_COSINE_SIM)
            << model.name << " S=" << S << " KV=" << KV << " D=" << D
            << ": Cosine similarity " << cosine_sim << " below threshold " << MIN_COSINE_SIM;

        EXPECT_LE(l2_error, MAX_L2_ERROR)
            << model.name << " S=" << S << " KV=" << KV << " D=" << D
            << ": L2 error " << l2_error << " above threshold " << MAX_L2_ERROR;

        // Verify softmax properties (sum to 1.0 per row)
        // Note: Online softmax with Q8_1 quantization has ~1-2% variance in row sums
        for (int i = 0; i < S; ++i)
        {
            float row_sum = 0.0f;
            for (int j = 0; j < KV; ++j)
            {
                row_sum += C[i * KV + j];
                EXPECT_GE(C[i * KV + j], 0.0f)
                    << "Negative softmax value at row " << i << ", col " << j;
            }
            EXPECT_NEAR(row_sum, 1.0f, 0.02f) // 2% tolerance for quantized softmax
                << "Row " << i << " does not sum to 1.0 (got " << row_sum << ")";
        }
    }
};

// =============================================================================
// Qwen 0.5B Tests (D=64)
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen05B_Decode_KV128)
{
    run_correctness_test(QWEN_0_5B, 1, 128);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen05B_Decode_KV1024)
{
    run_correctness_test(QWEN_0_5B, 1, 1024);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen05B_Prefill_256)
{
    run_correctness_test(QWEN_0_5B, 256, 256);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen05B_Prefill_512)
{
    run_correctness_test(QWEN_0_5B, 512, 512);
}

// =============================================================================
// Qwen 7B Tests (D=128)
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen7B_Decode_KV128)
{
    run_correctness_test(QWEN_7B, 1, 128);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen7B_Decode_KV1024)
{
    run_correctness_test(QWEN_7B, 1, 1024);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen7B_Decode_KV4096)
{
    run_correctness_test(QWEN_7B, 1, 4096);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen7B_Prefill_256)
{
    run_correctness_test(QWEN_7B, 256, 256);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen7B_Prefill_512)
{
    run_correctness_test(QWEN_7B, 512, 512);
}

// =============================================================================
// Qwen 32B Tests (D=128)
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen32B_Decode_KV128)
{
    run_correctness_test(QWEN_32B, 1, 128);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen32B_Decode_KV1024)
{
    run_correctness_test(QWEN_32B, 1, 1024);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen32B_Decode_KV4096)
{
    run_correctness_test(QWEN_32B, 1, 4096);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen32B_Prefill_256)
{
    run_correctness_test(QWEN_32B, 256, 256);
}

TEST_F(Test__Q8_1_OnlineSoftmax, Qwen32B_Prefill_512)
{
    run_correctness_test(QWEN_32B, 512, 512);
}

// =============================================================================
// Long Context Tests
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax, LongContext_Decode_KV8192)
{
    run_correctness_test(QWEN_7B, 1, 8192);
}

TEST_F(Test__Q8_1_OnlineSoftmax, LongContext_Prefill_1024)
{
    run_correctness_test(QWEN_7B, 1024, 1024);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__Q8_1_OnlineSoftmax, EdgeCase_SmallKV)
{
    // Minimum KV length (tests tail handling)
    run_correctness_test(QWEN_7B, 1, 32);
}

TEST_F(Test__Q8_1_OnlineSoftmax, EdgeCase_NonMultipleOf8_KV)
{
    // KV length not multiple of 8 (tests scalar tail)
    run_correctness_test(QWEN_7B, 4, 37);
}

TEST_F(Test__Q8_1_OnlineSoftmax, EdgeCase_NonMultipleOf4_M)
{
    // M not multiple of 4 (tests M1 kernel fallback)
    run_correctness_test(QWEN_7B, 7, 128);
}

TEST_F(Test__Q8_1_OnlineSoftmax, EdgeCase_LargeM)
{
    // Large M to test parallelization
    run_correctness_test(QWEN_7B, 2048, 128);
}

/**
 * @brief Main function with MPI initialization
 */
int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
