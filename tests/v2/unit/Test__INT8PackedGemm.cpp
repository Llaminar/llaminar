/**
 * @file Test__INT8PackedGemm.cpp
 * @brief Unit tests for INT8×INT8→INT32 GEMM with AVX512 VNNI
 *
 * Tests:
 * - Infrastructure: Check if INT8 GEMM is supported
 * - Factory: Verify kernel creation
 * - Basic correctness: Small matrix multiplication (TODO when INT8Tensor ready)
 *
 * @author David Sanftenberg
 */

#include "kernels/cpu/gemm/int8/PackedInt8Gemm.h"
#include "kernels/cpu/gemm/GemmAutoTuner.h"
#include "kernels/cpu/SimdTraits.h"
#include "kernels/cpu/GemmMicroKernelTemplateINT8.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"
#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include <cmath>

using namespace llaminar2::kernels::gemm;

/**
 * @brief Test fixture for INT8 GEMM tests
 */
class INT8PackedGemmTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Log test environment
        LOG_INFO("INT8 GEMM Support: " << (isINT8GemmSupported() ? "YES" : "NO"));

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
        LOG_INFO("Compiler flags: AVX512F + AVX512VNNI enabled");
#else
        LOG_INFO("Compiler flags: AVX512 VNNI not enabled");
#endif
    }
};

/**
 * @brief Test 1: Verify INT8 GEMM support detection
 */
TEST_F(INT8PackedGemmTest, SupportsINT8)
{
    bool supported = isINT8GemmSupported();

#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    // Should be supported if compiled with AVX512 VNNI
    EXPECT_TRUE(supported) << "INT8 GEMM should be supported with AVX512 VNNI flags";
#else
    // Should NOT be supported without AVX512 VNNI
    EXPECT_FALSE(supported) << "INT8 GEMM should NOT be supported without AVX512 VNNI";
#endif

    if (supported)
    {
        LOG_INFO("✓ INT8 GEMM is supported on this CPU");
    }
    else
    {
        LOG_WARN("✗ INT8 GEMM is NOT supported (missing AVX512 VNNI)");
    }
}

/**
 * @brief Test 2: Verify factory function returns kernel (or nullptr if unsupported)
 */
TEST_F(INT8PackedGemmTest, FactoryCreatesKernel)
{
    auto kernel = createINT8PackedGemm();

    if (isINT8GemmSupported())
    {
        ASSERT_NE(kernel, nullptr) << "Factory should return non-null kernel when supported";
        LOG_INFO("✓ Factory created INT8 GEMM kernel");
    }
    else
    {
        EXPECT_EQ(kernel, nullptr) << "Factory should return nullptr when unsupported";
        LOG_INFO("✓ Factory correctly returns nullptr (unsupported platform)");
    }
}

/**
 * @brief Test 3: Verify AVX512VNNI SimdTraits compile
 *
 * This test verifies the SimdTraits<AVX512VNNITag> template specialization
 * compiles correctly, even if we can't execute VNNI instructions at runtime.
 */
TEST_F(INT8PackedGemmTest, SimdTraitsCompile)
{
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
    // Check trait constants
    using Traits = llaminar2::kernels::simd::SimdTraits<llaminar2::kernels::simd::AVX512VNNITag>;

    EXPECT_EQ(Traits::vector_width, 64) << "Vector width should be 64 int8s";
    EXPECT_EQ(Traits::accum_width, 16) << "Accumulator width should be 16 int32s";
    EXPECT_EQ(Traits::dot_group_size, 4) << "VNNI performs 4-way dot products";
    EXPECT_STREQ(Traits::isa_name, "AVX512VNNI") << "ISA name should be AVX512VNNI";

    LOG_INFO("✓ AVX512VNNI SimdTraits constants verified");
    LOG_INFO("  - Vector width: " << Traits::vector_width << " int8s");
    LOG_INFO("  - Accumulator width: " << Traits::accum_width << " int32s");
    LOG_INFO("  - Dot group size: " << Traits::dot_group_size);
#else
    GTEST_SKIP() << "AVX512 VNNI not enabled, skipping trait verification";
#endif
}

/**
 * @brief Test 4: Basic INT8 correctness with known 4×4 matrices
 *
 * Tests INT8×INT8→INT32 GEMM with simple known values.
 * Verifies both INT32 accumulation and dequantization accuracy.
 */
TEST_F(INT8PackedGemmTest, BasicCorrectness_4x4)
{
    if (!isINT8GemmSupported())
    {
        GTEST_SKIP() << "INT8 GEMM not supported on this platform";
    }

    auto gemm = createINT8PackedGemm();
    ASSERT_NE(gemm, nullptr);

    // Test case: Simple 4×4 matrix multiplication
    // A = [1, 2, 3, 4]    B = [1, 0, 0, 0]
    //     [5, 6, 7, 8]        [0, 1, 0, 0]
    //     [9, 10, 11, 12]     [0, 0, 1, 0]
    //     [13, 14, 15, 16]    [0, 0, 0, 1]
    //
    // Expected: C = A × B = A (identity multiply)

    std::vector<float> A_fp32 = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f};

    std::vector<float> B_fp32 = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f};

    // Create INT8 tensors (quantization happens automatically)
    auto B_tensor = std::make_shared<llaminar2::INT8Tensor>(
        std::vector<size_t>{4, 4}, B_fp32);

    float scale_B = B_tensor->scale();
    LOG_INFO("B scale: " << scale_B);

    // Allocate output buffer (interpreted as int32)
    std::vector<float> C_buffer(16, 0.0f);
    int32_t *C_int32 = reinterpret_cast<int32_t *>(C_buffer.data());

    // Prepare A as int8 (reinterpreted from float* for API compatibility)
    std::vector<int8_t> A_int8(16);
    float scale_A = 1.0f;
    for (size_t i = 0; i < 16; ++i)
    {
        A_int8[i] = static_cast<int8_t>(std::round(A_fp32[i] / scale_A));
    }
    const float *A_as_float = reinterpret_cast<const float *>(A_int8.data());

    // Call INT8 GEMM (returns int32)
    bool success = gemm->multiply(
        A_as_float,
        C_buffer.data(),
        4, 4, 4, // m=4, n=4, k=4
        B_tensor.get(),
        false, // transpose_B=false
        1, 0); // alpha=1, beta=0

    ASSERT_TRUE(success) << "INT8 GEMM should succeed";

    // Verify INT32 results (before dequantization)
    // Expected: C[i] ≈ A[i] (since B is identity)
    // INT32 accumulation: C_int32[i] ≈ A_int8[i] (no scaling yet)
    LOG_INFO("INT32 accumulation results:");
    for (int i = 0; i < 16; ++i)
    {
        LOG_INFO("  C_int32[" << i << "] = " << C_int32[i]
                              << " (expected ≈ " << A_int8[i] << ")");
    }

    // Verify INT32 values match expected accumulation
    for (int i = 0; i < 16; ++i)
    {
        int32_t expected_int32 = static_cast<int32_t>(A_int8[i]);
        EXPECT_NEAR(C_int32[i], expected_int32, 2)
            << "INT32 mismatch at index " << i;
    }

    // Dequantize and verify FP32 accuracy
    std::vector<float> C_float(16);
    float scale_combined = scale_A * scale_B;
    for (size_t i = 0; i < 16; ++i)
    {
        C_float[i] = static_cast<float>(C_int32[i]) * scale_combined;
    }

    LOG_INFO("Dequantized FP32 results:");
    for (int i = 0; i < 16; ++i)
    {
        LOG_INFO("  C_float[" << i << "] = " << C_float[i]
                              << " (expected " << A_fp32[i] << ")");
    }

    // Verify dequantized values match original A (identity multiply)
    for (int i = 0; i < 16; ++i)
    {
        EXPECT_NEAR(C_float[i], A_fp32[i], 0.5f)
            << "Dequantized mismatch at index " << i;
    }

    LOG_INFO("✓ 4×4 identity multiply correctness verified");
}

/**
 * @brief Test 5: INT8 correctness with non-trivial 3×3 multiplication
 *
 * Tests with actual non-identity matrix to verify computation correctness.
 */
TEST_F(INT8PackedGemmTest, BasicCorrectness_3x3_NonIdentity)
{
    if (!isINT8GemmSupported())
    {
        GTEST_SKIP() << "INT8 GEMM not supported on this platform";
    }

    auto gemm = createINT8PackedGemm();
    ASSERT_NE(gemm, nullptr);

    // Test case: 3×3 matrix multiplication
    // A = [1, 2, 3]       B = [7, 8, 9]
    //     [4, 5, 6]           [10, 11, 12]
    //                         [13, 14, 15]
    //
    // Expected: C = A × B = [66, 72, 78]
    //                        [156, 171, 186]

    std::vector<float> A_fp32 = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f};

    std::vector<float> B_fp32 = {
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f};

    // Expected result (computed by hand)
    std::vector<float> expected = {
        66.0f, 72.0f, 78.0f,
        156.0f, 171.0f, 186.0f};

    // Create INT8 tensors
    auto B_tensor = std::make_shared<llaminar2::INT8Tensor>(
        std::vector<size_t>{3, 3}, B_fp32);

    float scale_B = B_tensor->scale();
    LOG_INFO("B scale: " << scale_B);

    // Allocate output buffer
    std::vector<float> C_buffer(6, 0.0f);
    int32_t *C_int32 = reinterpret_cast<int32_t *>(C_buffer.data());

    // Prepare A as int8
    std::vector<int8_t> A_int8(6);
    float scale_A = 1.0f;
    for (size_t i = 0; i < 6; ++i)
    {
        A_int8[i] = static_cast<int8_t>(std::round(A_fp32[i] / scale_A));
    }
    const float *A_as_float = reinterpret_cast<const float *>(A_int8.data());

    // Call INT8 GEMM
    bool success = gemm->multiply(
        A_as_float,
        C_buffer.data(),
        2, 3, 3, // m=2, n=3, k=3
        B_tensor.get(),
        false,
        1, 0);

    ASSERT_TRUE(success);

    // Dequantize
    std::vector<float> C_float(6);
    float scale_combined = scale_A * scale_B;
    for (size_t i = 0; i < 6; ++i)
    {
        C_float[i] = static_cast<float>(C_int32[i]) * scale_combined;
    }

    LOG_INFO("3×3 Non-identity multiply results:");
    for (int i = 0; i < 6; ++i)
    {
        LOG_INFO("  C[" << i << "] = " << C_float[i]
                        << " (expected " << expected[i] << ")");
    }

    // Verify results with tolerance for quantization error
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_NEAR(C_float[i], expected[i], expected[i] * 0.05f)
            << "Result mismatch at index " << i;
    }

    LOG_INFO("✓ 3×3 non-identity multiply correctness verified");
}

/**
 * @brief Test 6: INT8 correctness with alpha/beta scaling
 *
 * Tests C = alpha * A*B + beta * C_old pattern.
 */
TEST_F(INT8PackedGemmTest, BasicCorrectness_AlphaBeta)
{
    if (!isINT8GemmSupported())
    {
        GTEST_SKIP() << "INT8 GEMM not supported on this platform";
    }

    auto gemm = createINT8PackedGemm();
    ASSERT_NE(gemm, nullptr);

    // Test case: 2×2 with alpha=2, beta=3
    // A = [1, 2]    B = [5, 6]    C_old = [10, 20]
    //     [3, 4]        [7, 8]            [30, 40]
    //
    // A × B = [19, 22]
    //         [43, 50]
    //
    // Expected: C_new = 2 * [19, 22] + 3 * [10, 20] = [38 + 30, 44 + 60] = [68, 104]
    //                         [43, 50]      [30, 40]   [86 + 90, 100 + 120]  [176, 220]

    std::vector<float> A_fp32 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> B_fp32 = {5.0f, 6.0f, 7.0f, 8.0f};

    auto B_tensor = std::make_shared<llaminar2::INT8Tensor>(
        std::vector<size_t>{2, 2}, B_fp32);

    float scale_B = B_tensor->scale();

    // Allocate output buffer with initial values (C_old)
    std::vector<int32_t> C_old_int32 = {10, 20, 30, 40};
    std::vector<float> C_buffer(4);
    std::memcpy(C_buffer.data(), C_old_int32.data(), 4 * sizeof(int32_t));
    int32_t *C_int32 = reinterpret_cast<int32_t *>(C_buffer.data());

    // Prepare A as int8
    std::vector<int8_t> A_int8(4);
    float scale_A = 1.0f;
    for (size_t i = 0; i < 4; ++i)
    {
        A_int8[i] = static_cast<int8_t>(std::round(A_fp32[i] / scale_A));
    }
    const float *A_as_float = reinterpret_cast<const float *>(A_int8.data());

    // Call INT8 GEMM with alpha=2, beta=3
    bool success = gemm->multiply(
        A_as_float,
        C_buffer.data(),
        2, 2, 2, // m=2, n=2, k=2
        false,
        2.0f, 3.0f,  // alpha=2, beta=3
        nullptr, 0); // mpi_ctx, device_idx

    ASSERT_TRUE(success);

    // Verify INT32 results
    // Expected INT32: 2 * (A_int8 × B_int8) + 3 * C_old_int32
    // A × B (int32) ≈ [19, 22, 43, 50] (before dequantization)
    LOG_INFO("Alpha/Beta INT32 results:");
    for (int i = 0; i < 4; ++i)
    {
        LOG_INFO("  C_int32[" << i << "] = " << C_int32[i]);
    }

    // For INT32 accumulation with alpha/beta:
    // C_new = alpha * (A×B) + beta * C_old
    // Since we're in INT32 domain, scales don't apply yet
    // Just verify the integer arithmetic is correct

    // Expected INT32 values (approximate, depends on quantization):
    // C[0] = 2 * 19 + 3 * 10 = 38 + 30 = 68
    // C[1] = 2 * 22 + 3 * 20 = 44 + 60 = 104
    // C[2] = 2 * 43 + 3 * 30 = 86 + 90 = 176
    // C[3] = 2 * 50 + 3 * 40 = 100 + 120 = 220
    std::vector<int32_t> expected_int32 = {68, 104, 176, 220};

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_NEAR(C_int32[i], expected_int32[i], 10)
            << "INT32 alpha/beta mismatch at index " << i;
    }

    LOG_INFO("✓ Alpha/Beta scaling correctness verified");
}

/**
 * @brief Test 7: INT8 correctness with larger 8×8 matrices
 *
 * Tests with larger matrices to stress cache blocking and packing.
 */
TEST_F(INT8PackedGemmTest, BasicCorrectness_8x8_Larger)
{
    if (!isINT8GemmSupported())
    {
        GTEST_SKIP() << "INT8 GEMM not supported on this platform";
    }

    auto gemm = createINT8PackedGemm();
    ASSERT_NE(gemm, nullptr);

    // Test case: 8×8 matrices with sequential values
    // A[i,j] = i * 8 + j + 1
    // B[i,j] = (i + j) % 16 + 1

    std::vector<float> A_fp32(64);
    std::vector<float> B_fp32(64);

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            A_fp32[i * 8 + j] = static_cast<float>(i * 8 + j + 1);
            B_fp32[i * 8 + j] = static_cast<float>((i + j) % 16 + 1);
        }
    }

    // Create INT8 tensors
    auto B_tensor = std::make_shared<llaminar2::INT8Tensor>(
        std::vector<size_t>{8, 8}, B_fp32);

    float scale_B = B_tensor->scale();

    // Allocate output buffer
    std::vector<float> C_buffer(64, 0.0f);
    int32_t *C_int32 = reinterpret_cast<int32_t *>(C_buffer.data());

    // Prepare A as int8
    std::vector<int8_t> A_int8(64);
    float scale_A = 1.0f;
    for (size_t i = 0; i < 64; ++i)
    {
        A_int8[i] = static_cast<int8_t>(std::round(A_fp32[i] / scale_A));
    }
    const float *A_as_float = reinterpret_cast<const float *>(A_int8.data());

    // Call INT8 GEMM
    bool success = gemm->multiply(
        A_as_float,
        C_buffer.data(),
        8, 8, 8, // m=8, n=8, k=8
        B_tensor.get(),
        false,
        1, 0);

    ASSERT_TRUE(success);

    // Compute reference FP32 result
    std::vector<float> C_ref(64, 0.0f);
    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < 8; ++k)
            {
                sum += A_fp32[i * 8 + k] * B_fp32[k * 8 + j];
            }
            C_ref[i * 8 + j] = sum;
        }
    }

    // Dequantize INT8 result
    std::vector<float> C_float(64);
    float scale_combined = scale_A * scale_B;
    for (size_t i = 0; i < 64; ++i)
    {
        C_float[i] = static_cast<float>(C_int32[i]) * scale_combined;
    }

    // Verify results with tolerance
    LOG_INFO("8×8 larger matrix multiply results (first 16 elements):");
    for (int i = 0; i < 16; ++i)
    {
        LOG_INFO("  C[" << i << "] = " << C_float[i]
                        << " (ref " << C_ref[i] << ")");
    }

    int mismatch_count = 0;
    for (int i = 0; i < 64; ++i)
    {
        float abs_error = std::abs(C_float[i] - C_ref[i]);
        float rel_error = abs_error / (std::abs(C_ref[i]) + 1e-6f);

        if (rel_error > 0.1f)
        { // 10% tolerance for quantization
            mismatch_count++;
            LOG_WARN("Mismatch at [" << i << "]: got " << C_float[i]
                                     << ", expected " << C_ref[i]
                                     << " (rel_error " << rel_error << ")");
        }
    }

    // Allow up to 10% of elements to have larger errors due to quantization
    EXPECT_LT(mismatch_count, 7) << "Too many mismatches in 8×8 multiply";

    LOG_INFO("✓ 8×8 larger matrix correctness verified (mismatches: "
             << mismatch_count << "/64)");
}

/**
 * @brief Test 5: Verify INT8 micro-kernel template compiles
 *
 * This test is disabled because it requires AVX512VNNI to be enabled at compile time.
 * The actual micro-kernel template compilation is verified by the INT8PackedGemm.cpp
 * build itself.
 */
TEST_F(INT8PackedGemmTest, DISABLED_MicroKernelTemplateCompiles)
{
    GTEST_SKIP() << "Micro-kernel instantiation test disabled (requires AVX512VNNI compile flags)";
}

/**
 * @brief Test 8: Verify all registered INT8 variants are created
 *
 * Tests that the variant registration system creates the expected number of
 * INT8 GEMM kernel variants. This validates the code generation integration.
 */
TEST_F(INT8PackedGemmTest, VariantRegistration)
{
    if (!isINT8GemmSupported())
    {
        GTEST_SKIP() << "INT8 GEMM not supported on this platform";
    }

    // Register all variants
    auto variants = llaminar2::kernels::gemm::registerINT8MicroKernelVariants();

    // Expected variant count calculation:
    // mr_values: {1, 2, 4, 8, 16, 32} = 6
    // nr_values: {1, 2, 4, 6, 8, 16, 32} = 7
    // Constraint: mr * nr <= 48
    //
    // Valid (mr, nr) pairs:
    // mr=1: 1,2,4,6,8,16,32 (7 pairs)
    // mr=2: 1,2,4,6,8,16,32 (7 pairs, 2*32=64 > 48, skip 32) = 6 pairs
    // mr=4: 1,2,4,6,8,16 (6 pairs, 4*32=128 > 48, 4*16=64 > 48, skip 16,32) = 4 pairs
    // mr=8: 1,2,4,6 (4 pairs, 8*8=64 > 48, skip 8+)
    // mr=16: 1,2 (2 pairs, 16*4=64 > 48, skip 4+)
    // mr=32: 1 (1 pair, 32*2=64 > 48, skip 2+)
    //
    // Total (mr,nr) pairs: 7 + 6 + 4 + 4 + 2 + 1 = 24
    //
    // unroll_k: {1, 2, 4, 8} = 4
    // prefetch: {0, 1, 2, 3} = 4
    //
    // Total variants: 24 * 4 * 4 = 384

    int expected_count = 384;

    LOG_INFO("Registered " << variants.size() << " INT8 VNNI micro-kernel variants");
    LOG_INFO("Expected: " << expected_count << " variants");

    EXPECT_EQ(variants.size(), expected_count)
        << "Variant count mismatch - check mr/nr constraints";

    // Verify some specific variants exist
    std::map<std::string, bool> variant_names;
    for (const auto &variant : variants)
    {
        std::string name = variant->name();
        variant_names[name] = true;

        // Verify name format: INT8_AVX512VNNI_MxN_uK_pP
        EXPECT_TRUE(name.find("INT8_AVX512VNNI_") == 0)
            << "Invalid variant name: " << name;
    }

    // Check for some expected specific variants
    EXPECT_TRUE(variant_names.count("INT8_AVX512VNNI_4x4_u4_p0") > 0)
        << "Missing expected variant 4x4_u4_p0";
    EXPECT_TRUE(variant_names.count("INT8_AVX512VNNI_8x8_u2_p1") > 0)
        << "Missing expected variant 8x8_u2_p1";
    EXPECT_TRUE(variant_names.count("INT8_AVX512VNNI_1x1_u1_p0") > 0)
        << "Missing expected variant 1x1_u1_p0";

    LOG_INFO("✓ INT8 variant registration verified (" << variants.size() << " variants)");
}

/**
 * @brief Main function
 */
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
