// Minimal test to isolate OpenBLAS cblas_sbgemm behavior
#include <gtest/gtest.h>
#include <cblas.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cstring>

// Simple BF16 conversion (truncate FP32 to BF16)
static uint16_t float_to_bf16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    // Round to nearest even
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

static float bf16_to_float(uint16_t bf16)
{
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

TEST(MinimalBF16Test, DirectCblasCall)
{
    // Minimal 2x2 matrix multiplication to isolate issue
    // A = [1.0, 2.0]    B = [1.0, 0.0]    Expected C = [5.0, 2.0]
    //     [3.0, 4.0]        [2.0, 1.0]                  [11.0, 4.0]

    int m = 2, n = 2, k = 2;

    // Create FP32 matrices
    std::vector<float> A_fp32 = {1.0f, 2.0f, 3.0f, 4.0f}; // Row-major 2x2
    std::vector<float> B_fp32 = {1.0f, 0.0f, 2.0f, 1.0f}; // Row-major 2x2

    // Convert to BF16 (raw uint16_t)
    std::vector<uint16_t> A_bf16(4), B_bf16(4);
    for (int i = 0; i < 4; ++i)
    {
        A_bf16[i] = float_to_bf16(A_fp32[i]);
        B_bf16[i] = float_to_bf16(B_fp32[i]);
    }

    // Output in FP32
    std::vector<float> C(4, 0.0f);

    std::cout << "Input A (BF16): ";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << bf16_to_float(A_bf16[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "Input B (BF16): ";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << bf16_to_float(B_bf16[i]) << " ";
    }
    std::cout << std::endl;

    // Call cblas_sbgemm
    cblas_sbgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                 m, n, k,
                 1.0f,
                 reinterpret_cast<const bfloat16 *>(A_bf16.data()), k, // lda = k = 2
                 reinterpret_cast<const bfloat16 *>(B_bf16.data()), n, // ldb = n = 2
                 0.0f, C.data(), n);                                   // ldc = n = 2

    std::cout << "Output C (FP32): ";
    for (int i = 0; i < 4; ++i)
    {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    // Check for NaN
    bool has_nan = false;
    for (int i = 0; i < 4; ++i)
    {
        if (std::isnan(C[i]))
        {
            has_nan = true;
            std::cout << "NaN detected at index " << i << std::endl;
        }
    }

    EXPECT_FALSE(has_nan) << "cblas_sbgemm produced NaN!";

    // Expected: C = [[5, 2], [11, 4]] (approximately, due to BF16 quantization)
    if (!has_nan)
    {
        EXPECT_NEAR(C[0], 5.0f, 0.1f);
        EXPECT_NEAR(C[1], 2.0f, 0.1f);
        EXPECT_NEAR(C[2], 11.0f, 0.1f);
        EXPECT_NEAR(C[3], 4.0f, 0.1f);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
