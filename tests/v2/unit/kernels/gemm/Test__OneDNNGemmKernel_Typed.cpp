#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>
#include <iostream>
#include "tensors/FP16Utils.h"

#ifdef HAVE_ONEDNN
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"

using llaminar2::gemm_v4::OneDNNGemmKernel;

namespace
{

    // Helper to convert float to BF16 (truncate lower 16 bits)
    uint16_t float_to_bf16(float f)
    {
        uint32_t x;
        std::memcpy(&x, &f, sizeof(float));
        return static_cast<uint16_t>(x >> 16);
    }

    // Helper to convert BF16 to float (pad lower 16 bits with 0)
    float bf16_to_float(uint16_t b)
    {
        uint32_t x = static_cast<uint32_t>(b) << 16;
        float f;
        std::memcpy(&f, &x, sizeof(float));
        return f;
    }

    // Reference Softmax implementation
    void reference_softmax(float *C, int m, int n)
    {
        for (int i = 0; i < m; ++i)
        {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int j = 0; j < n; ++j)
            {
                max_val = std::max(max_val, C[i * n + j]);
            }

            float sum = 0.0f;
            for (int j = 0; j < n; ++j)
            {
                C[i * n + j] = std::exp(C[i * n + j] - max_val);
                sum += C[i * n + j];
            }

            for (int j = 0; j < n; ++j)
            {
                C[i * n + j] /= sum;
            }
        }
    }

    // Reference GEMM implementation for validation
    void reference_gemm(const float *A, const float *B, float *C,
                        int m, int n, int k, bool transpose_B,
                        float alpha, float beta)
    {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l)
                {
                    float a_val = A[i * k + l];
                    float b_val = transpose_B ? B[j * k + l] : B[l * n + j];
                    sum += a_val * b_val;
                }
                if (beta == 0.0f)
                {
                    C[i * n + j] = alpha * sum;
                }
                else
                {
                    C[i * n + j] = alpha * sum + beta * C[i * n + j];
                }
            }
        }
    }

} // namespace

class Test__OneDNNGemmKernel_Typed : public ::testing::Test
{
protected:
    OneDNNGemmKernel kernel;

    Test__OneDNNGemmKernel_Typed() : kernel(nullptr) {}

    void SetUp() override
    {
        // Setup code if needed
    }
};

TEST_F(Test__OneDNNGemmKernel_Typed, FloatFloat_Typed_MatchesReference)
{
    int m = 16;
    int n = 16;
    int k = 32;

    std::vector<float> A(m * k);
    std::vector<float> B(k * n); // Transposed B (n x k) stored as k x n if transpose_B=true?
                                 // Wait, if transpose_B=true, B is n x k physically?
                                 // multiply_activations signature says: B is (n, k) if transpose_B is true.
                                 // Let's check OneDNNGemmKernel implementation.
                                 // It says: "C = alpha * A @ B^T + beta * C"
                                 // If transpose_B is true, B is expected to be (n, k) in memory?
                                 // Or B is (k, n) and we transpose it?
                                 // Usually "B^T" implies B is (k, n) and we want (n, k) effectively?
                                 // Let's check prepare_rhs_for_matmul in OneDNNGemmKernel.h

    // Re-reading OneDNNGemmKernel.h:
    // const float *rhs_ptr = prepare_rhs_for_matmul(B, n, k, transpose_B);
    // If transpose_B is true, it assumes B is already in a format suitable for B^T operation?
    // Actually, standard GEMM A(m,k) * B(k,n) = C(m,n).
    // If transpose_B is true, it means the input B pointer points to a (n, k) matrix, and we want to treat it as (k, n) by transposing?
    // Or does it mean we want A * B^T where B is (n, k)?
    // The comment says: C = alpha * A @ B^T + beta * C
    // This usually means A is (m, k) and B is (n, k).

    // Let's assume B is (n, k) when transpose_B=true.

    std::vector<float> B_transposed(n * k);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    // Initialize with random data
    for (auto &val : A)
        val = static_cast<float>(rand()) / RAND_MAX;
    for (auto &val : B_transposed)
        val = static_cast<float>(rand()) / RAND_MAX;

    // Run Kernel
    bool success = kernel.multiply_activations_typed<float, float>(
        A.data(), B_transposed.data(), C.data(),
        m, n, k, true /* transpose_B */, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    // Run Reference
    // A is (m, k), B_transposed is (n, k). We want A * B_transposed^T.
    // So effectively A * B where B is (k, n).
    // In reference_gemm, if transpose_B is true, it accesses B as B[j*k + l].
    // This matches B being (n, k).
    reference_gemm(A.data(), B_transposed.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);

    // Compare
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }
}

TEST_F(Test__OneDNNGemmKernel_Typed, BF16BF16_Typed_MatchesReference)
{
    int m = 16;
    int n = 16;
    int k = 32;

    std::vector<uint16_t> A_bf16(m * k);
    std::vector<uint16_t> B_bf16(n * k); // B is (n, k)
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    std::vector<float> A_float(m * k);
    std::vector<float> B_float(n * k);

    // Initialize
    for (int i = 0; i < m * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        A_bf16[i] = float_to_bf16(val);
        A_float[i] = bf16_to_float(A_bf16[i]);
    }
    for (int i = 0; i < n * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        B_bf16[i] = float_to_bf16(val);
        B_float[i] = bf16_to_float(B_bf16[i]);
    }

    // Run Kernel
    bool success = kernel.multiply_activations_typed<uint16_t, uint16_t>(
        A_bf16.data(), B_bf16.data(), C.data(),
        m, n, k, true /* transpose_B */, 1.0f, 0.0f,
        nullptr, -1, llaminar2::ActivationFormat::BF16);
    ASSERT_TRUE(success);

    // Run Reference using converted floats
    reference_gemm(A_float.data(), B_float.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);

    // Compare
    // BF16 has less precision, so tolerance should be higher
    // But since we compute reference using the exact same BF16 values (converted back to float),
    // the only difference is accumulation order and intermediate precision (float vs float).
    // OneDNN might use AVX512 BF16 instructions which accumulate in float.
    // So results should be very close.
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2f);
    }
}

TEST_F(Test__OneDNNGemmKernel_Typed, Strided_FloatFloat_MatchesContiguous)
{
    int m = 8;
    int n = 8;
    int k = 16;

    // Create larger buffers to simulate striding
    int lda = k + 5;
    int ldb = k + 3; // B is (n, k)
    int ldc = n + 2;

    std::vector<float> A_strided(m * lda);
    std::vector<float> B_strided(n * ldb);
    std::vector<float> C_strided(m * ldc, 0.0f);

    std::vector<float> A_dense(m * k);
    std::vector<float> B_dense(n * k);
    std::vector<float> C_dense(m * n, 0.0f);

    // Initialize strided and dense buffers with same data
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float val = static_cast<float>(rand()) / RAND_MAX;
            A_strided[i * lda + j] = val;
            A_dense[i * k + j] = val;
        }
    }
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < k; ++j)
        {
            float val = static_cast<float>(rand()) / RAND_MAX;
            B_strided[i * ldb + j] = val;
            B_dense[i * k + j] = val;
        }
    }

    // Run Strided Kernel
    bool success = kernel.multiply_activations_strided_typed<float, float>(
        A_strided.data(), B_strided.data(), C_strided.data(),
        m, n, k, lda, ldb, ldc, true /* transpose_B */, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    // Run Dense Kernel (as reference)
    success = kernel.multiply_activations_typed<float, float>(
        A_dense.data(), B_dense.data(), C_dense.data(),
        m, n, k, true, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    // Compare
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            EXPECT_NEAR(C_strided[i * ldc + j], C_dense[i * n + j], 1e-5f);
        }
    }
}

TEST_F(Test__OneDNNGemmKernel_Typed, Unsupported_Int8Int8_ReturnsFalse)
{
    int m = 4;
    int n = 4;
    int k = 4;
    std::vector<int8_t> A(m * k);
    std::vector<int8_t> B(n * k);
    std::vector<float> C(m * n);

    // Should return false and log error
    bool success = kernel.multiply_activations_typed<int8_t, int8_t>(
        A.data(), B.data(), C.data(),
        m, n, k, true, 1.0f, 0.0f,
        nullptr, -1, llaminar2::ActivationFormat::INT8);
    EXPECT_FALSE(success);
}

TEST_F(Test__OneDNNGemmKernel_Typed, BF16BF16_FusedSoftmax_MatchesReference)
{
    int m = 16;
    int n = 16;
    int k = 32;

    std::vector<uint16_t> A_bf16(m * k);
    std::vector<uint16_t> B_bf16(n * k);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    std::vector<float> A_float(m * k);
    std::vector<float> B_float(n * k);

    // Initialize
    for (int i = 0; i < m * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        A_bf16[i] = float_to_bf16(val);
        A_float[i] = bf16_to_float(A_bf16[i]);
    }
    for (int i = 0; i < n * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        B_bf16[i] = float_to_bf16(val);
        B_float[i] = bf16_to_float(B_bf16[i]);
    }

    // Run Kernel
    bool success = kernel.multiply_with_softmax_typed<uint16_t, uint16_t>(
        A_bf16.data(), B_bf16.data(), C.data(),
        m, n, k, 1.0f /* scale */, true /* transpose_B */, 1 /* axis */,
        nullptr /* mask */, false /* is_causal */, nullptr, -1, llaminar2::ActivationFormat::BF16);
    ASSERT_TRUE(success);

    // Run Reference
    reference_gemm(A_float.data(), B_float.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);
    reference_softmax(C_ref.data(), m, n);

    // Compare
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2f);
    }
}

TEST_F(Test__OneDNNGemmKernel_Typed, FP16FP16_FusedSoftmax_MatchesReference)
{
    int m = 16;
    int n = 16;
    int k = 32;

    std::vector<uint16_t> A_fp16(m * k);
    std::vector<uint16_t> B_fp16(n * k);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);

    std::vector<float> A_float(m * k);
    std::vector<float> B_float(n * k);

    // Initialize
    for (int i = 0; i < m * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        A_fp16[i] = llaminar2::fp32_to_fp16(val);
        A_float[i] = llaminar2::fp16_to_fp32(A_fp16[i]);
    }
    for (int i = 0; i < n * k; ++i)
    {
        float val = static_cast<float>(rand()) / RAND_MAX;
        B_fp16[i] = llaminar2::fp32_to_fp16(val);
        B_float[i] = llaminar2::fp16_to_fp32(B_fp16[i]);
    }

    // Run Kernel
    bool success = kernel.multiply_with_softmax_typed<uint16_t, uint16_t>(
        A_fp16.data(), B_fp16.data(), C.data(),
        m, n, k, 1.0f /* scale */, true /* transpose_B */, 1 /* axis */,
        nullptr /* mask */, false /* is_causal */, nullptr, -1, llaminar2::ActivationFormat::FP16);
    ASSERT_TRUE(success);

    // Run Reference
    reference_gemm(A_float.data(), B_float.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);
    reference_softmax(C_ref.data(), m, n);

    // Compare
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-2f);
    }
}

TEST_F(Test__OneDNNGemmKernel_Typed, FloatFloat_FusedSoftmax_WithMask_MatchesReference)
{
    int m = 16;
    int n = 16;
    int k = 32;

    std::vector<float> A(m * k);
    std::vector<float> B(n * k);
    std::vector<float> C(m * n, 0.0f);
    std::vector<float> C_ref(m * n, 0.0f);
    std::vector<float> mask(m * n);

    // Initialize
    for (auto &val : A)
        val = static_cast<float>(rand()) / RAND_MAX;
    for (auto &val : B)
        val = static_cast<float>(rand()) / RAND_MAX;
    for (auto &val : mask)
        val = (static_cast<float>(rand()) / RAND_MAX) - 0.5f; // Some negative values

    // Run Kernel
    bool success = kernel.multiply_with_softmax_typed<float, float>(
        A.data(), B.data(), C.data(),
        m, n, k, 1.0f /* scale */, true /* transpose_B */, 1 /* axis */,
        mask.data(), false /* is_causal */, nullptr, -1, llaminar2::ActivationFormat::FP32);
    ASSERT_TRUE(success);

    // Run Reference
    reference_gemm(A.data(), B.data(), C_ref.data(), m, n, k, true, 1.0f, 0.0f);

    // Apply mask
    for (int i = 0; i < m * n; ++i)
    {
        C_ref[i] += mask[i];
    }

    reference_softmax(C_ref.data(), m, n);

    // Compare
    for (size_t i = 0; i < C.size(); ++i)
    {
        EXPECT_NEAR(C[i], C_ref[i], 1e-4f);
    }
}

#endif // HAVE_ONEDNN
