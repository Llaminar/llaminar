#include <gtest/gtest.h>

#include "tensors/TensorKernels.h" // IActivationTensor, ActivationPack
#include "tensors/Tensors.h"       // FP32Tensor definition
#include "kernels/cpu/gemm_v4/OneDNNGemmAdapter.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h" // Inline definitions for run_onednn_*

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

namespace
{

    void fill_simple_matrices(int M, int N, int K,
                              FP32Tensor &A,
                              FP32Tensor &B)
    {
        // A: [M, K]
        for (int m = 0; m < M; ++m)
        {
            for (int k = 0; k < K; ++k)
            {
                float *a_ptr = A.mutable_data();
                a_ptr[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                    0.1f * static_cast<float>(m) + 0.01f * static_cast<float>(k);
            }
        }

        // B: [N, K]
        for (int n = 0; n < N; ++n)
        {
            for (int k = 0; k < K; ++k)
            {
                float *b_ptr = B.mutable_data();
                b_ptr[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)] =
                    0.2f * static_cast<float>(n) - 0.015f * static_cast<float>(k);
            }
        }
    }

    std::vector<float> matmul_reference(int M, int N, int K,
                                        const FP32Tensor &A,
                                        const FP32Tensor &B,
                                        const std::vector<float> *bias = nullptr)
    {
        std::vector<float> out(static_cast<size_t>(M) * static_cast<size_t>(N), 0.0f);
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    float a = A.data()[static_cast<size_t>(m) * static_cast<size_t>(K) + static_cast<size_t>(k)];
                    float b = B.data()[static_cast<size_t>(n) * static_cast<size_t>(K) + static_cast<size_t>(k)];
                    acc += a * b;
                }
                if (bias)
                {
                    acc += (*bias)[static_cast<size_t>(n)];
                }
                out[static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n)] = acc;
            }
        }
        return out;
    }

} // namespace

TEST(Test__OneDNNGemmAdapter, BasicMatmulMatchesReference)
{
    constexpr int M = 4;
    constexpr int N = 3;
    constexpr int K = 5;

    FP32Tensor A({static_cast<size_t>(M), static_cast<size_t>(K)});
    FP32Tensor B({static_cast<size_t>(N), static_cast<size_t>(K)});
    fill_simple_matrices(M, N, K, A, B);

    FP32Tensor output({static_cast<size_t>(M), static_cast<size_t>(N)});

    bool ok = onednn_gemm_adapter(M, N, K, A, B, output, nullptr);
    ASSERT_TRUE(ok);

    auto reference = matmul_reference(M, N, K, A, B, nullptr);

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            size_t idx = static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n);
            EXPECT_NEAR(output.data()[idx], reference[idx], 2e-1f) << "Mismatch at (" << m << "," << n << ")";
        }
    }
}

TEST(Test__OneDNNGemmAdapter, BiasMatmulMatchesReference)
{
    constexpr int M = 3;
    constexpr int N = 4;
    constexpr int K = 6;

    FP32Tensor A({static_cast<size_t>(M), static_cast<size_t>(K)});
    FP32Tensor B({static_cast<size_t>(N), static_cast<size_t>(K)});
    fill_simple_matrices(M, N, K, A, B);

    std::vector<float> bias(static_cast<size_t>(N));
    for (int n = 0; n < N; ++n)
    {
        bias[static_cast<size_t>(n)] = 0.05f * static_cast<float>(n - 1);
    }

    FP32Tensor output({static_cast<size_t>(M), static_cast<size_t>(N)});

    bool ok = onednn_gemm_adapter(M, N, K, A, B, output, bias.data());
    ASSERT_TRUE(ok);

    auto reference = matmul_reference(M, N, K, A, B, &bias);

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            size_t idx = static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n);
            EXPECT_NEAR(output.data()[idx], reference[idx], 2e-1f) << "Mismatch at (" << m << "," << n << ")";
        }
    }
}

// This test emphasizes per-output-channel weight scaling:
// - We take the baseline pattern from fill_simple_matrices()
// - We then amplify a single output channel (n = 1) in B
//   so that its dynamic range differs sharply from the others.
// The adapter should still track the FP32 reference because
// WeightPack::col_scales holds a distinct scale per output channel
// that is used during from_int32_with_scales() dequantization.
TEST(Test__OneDNNGemmAdapter, PerChannelScaleRespondsToOutlierChannel)
{
    constexpr int M = 4;
    constexpr int N = 3;
    constexpr int K = 5;

    FP32Tensor A({static_cast<size_t>(M), static_cast<size_t>(K)});
    FP32Tensor B({static_cast<size_t>(N), static_cast<size_t>(K)});

    // Start from the baseline pattern.
    fill_simple_matrices(M, N, K, A, B);

    // Make one output channel (n = 1) significantly larger to exercise
    // per-output-channel weight scales.
    float *b_ptr = B.mutable_data();
    for (int k = 0; k < K; ++k)
    {
        size_t idx = static_cast<size_t>(1) * static_cast<size_t>(K) + static_cast<size_t>(k);
        b_ptr[idx] *= 5.0f;
    }

    FP32Tensor output({static_cast<size_t>(M), static_cast<size_t>(N)});

    bool ok = onednn_gemm_adapter(M, N, K, A, B, output, nullptr);
    ASSERT_TRUE(ok);

    auto reference = matmul_reference(M, N, K, A, B, nullptr);

    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            size_t idx = static_cast<size_t>(m) * static_cast<size_t>(N) + static_cast<size_t>(n);
            EXPECT_NEAR(output.data()[idx], reference[idx], 2e-1f)
                << "Mismatch at (" << m << "," << n << ")";
        }
    }
}
