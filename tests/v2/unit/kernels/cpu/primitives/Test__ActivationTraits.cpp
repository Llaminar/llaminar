/**
 * @file Test__ActivationTraits.cpp
 * @brief Unit tests for ActivationTraits template specializations
 * @author David Sanftenberg
 *
 * Tests:
 * 1. Softmax trait methods work correctly for each precision
 * 2. GEMM kernel creation works
 * 3. Workspace allocation works
 * 4. INT32 conversion strategy (INT32→FP32→softmax→INT32)
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <memory>

#include "v2/kernels/cpu/primitives/ActivationTraits.h"
#include "v2/tensors/Tensors.h"

using namespace llaminar2;
using namespace llaminar2::primitives;

namespace
{
    constexpr float FP32_TOLERANCE = 1e-5f;
    constexpr float BF16_TOLERANCE = 5e-3f;
    constexpr float FP16_TOLERANCE = 5e-4f;

    // Helper: Check that array sums to 1.0 (softmax property)
    void expect_sums_to_one(const float *data, int count, float tolerance = 1e-5f)
    {
        float sum = 0.0f;
        for (int i = 0; i < count; ++i)
        {
            sum += data[i];
        }
        EXPECT_NEAR(sum, 1.0f, tolerance) << "Softmax output should sum to 1.0";
    }

    // Helper: BF16→FP32 conversion
    std::vector<float> bf16_to_fp32(const uint16_t *bf16, int count)
    {
        std::vector<float> fp32(count);
        for (int i = 0; i < count; ++i)
        {
            uint32_t fp32_bits = static_cast<uint32_t>(bf16[i]) << 16;
            std::memcpy(&fp32[i], &fp32_bits, sizeof(float));
        }
        return fp32;
    }

    // Helper: FP16→FP32 conversion
    std::vector<float> fp16_to_fp32(const uint16_t *fp16, int count)
    {
        std::vector<float> fp32(count);
        for (int i = 0; i < count; ++i)
        {
#if defined(__F16C__)
            __m128i vec = _mm_cvtsi32_si128(fp16[i]);
            __m128 fp32_vec = _mm_cvtph_ps(vec);
            fp32[i] = _mm_cvtss_f32(fp32_vec);
#else
            // Manual conversion
            uint16_t h = fp16[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7C00) >> 10;
            uint32_t mant = (h & 0x03FF);

            uint32_t fp32_bits;
            if (exp == 0)
            {
                if (mant == 0)
                {
                    fp32_bits = sign;
                }
                else
                {
                    exp = 1;
                    while ((mant & 0x0400) == 0)
                    {
                        mant <<= 1;
                        exp--;
                    }
                    mant &= 0x03FF;
                    fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
                }
            }
            else if (exp == 0x1F)
            {
                fp32_bits = sign | 0x7F800000 | (mant << 13);
            }
            else
            {
                fp32_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }

            std::memcpy(&fp32[i], &fp32_bits, sizeof(float));
#endif
        }
        return fp32;
    }

} // anonymous namespace

// ============================================================================
// FP32Tensor Traits Tests
// ============================================================================

TEST(ActivationTraits_FP32, SoftmaxBasicCorrectness)
{
    constexpr int rows = 4;
    constexpr int cols = 8;
    std::vector<float> scores(rows * cols);

    // Initialize with sequential values
    for (int i = 0; i < rows * cols; ++i)
    {
        scores[i] = static_cast<float>(i) / 10.0f;
    }

    // Apply softmax via traits
    ActivationTraits<FP32Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);

    // Each row should sum to 1.0
    for (int r = 0; r < rows; ++r)
    {
        expect_sums_to_one(scores.data() + r * cols, cols, FP32_TOLERANCE);
    }
}

TEST(ActivationTraits_FP32, SoftmaxCausalMasking)
{
    constexpr int rows = 4;
    constexpr int cols = 8;
    std::vector<float> scores(rows * cols, 1.0f);

    ActivationTraits<FP32Tensor>::apply_softmax(scores.data(), rows, cols, true, 1.0f);

    // Row r should have zeros after column r
    for (int r = 0; r < rows; ++r)
    {
        for (int c = r + 1; c < cols; ++c)
        {
            EXPECT_EQ(scores[r * cols + c], 0.0f)
                << "Causal mask failed at row " << r << ", col " << c;
        }
    }
}

TEST(ActivationTraits_FP32, GemmKernelCreation)
{
    auto gemm = ActivationTraits<FP32Tensor>::create_activation_gemm();
    ASSERT_NE(gemm, nullptr) << "GEMM kernel creation failed";
}

TEST(ActivationTraits_FP32, WorkspaceAllocation)
{
    auto workspace = ActivationTraits<FP32Tensor>::allocate_workspace({32, 64});
    ASSERT_NE(workspace, nullptr) << "Workspace allocation failed";

    auto shape = workspace->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 64);
}

// ============================================================================
// BF16Tensor Traits Tests
// ============================================================================

TEST(ActivationTraits_BF16, SoftmaxBasicCorrectness)
{
    constexpr int rows = 4;
    constexpr int cols = 8;
    std::vector<uint16_t> scores(rows * cols);

    // Initialize with sequential values (as BF16)
    for (int i = 0; i < rows * cols; ++i)
    {
        float fp32_val = static_cast<float>(i) / 10.0f;
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_val, sizeof(float));
        uint32_t rounding_bias = 0x7FFF + ((fp32_bits >> 16) & 1);
        uint32_t rounded = fp32_bits + rounding_bias;
        scores[i] = static_cast<uint16_t>(rounded >> 16);
    }

    // Apply softmax via traits
    ActivationTraits<BF16Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);

    // Each row should sum to ~1.0 (within BF16 tolerance)
    for (int r = 0; r < rows; ++r)
    {
        auto row_fp32 = bf16_to_fp32(scores.data() + r * cols, cols);
        expect_sums_to_one(row_fp32.data(), cols, BF16_TOLERANCE);
    }
}

TEST(ActivationTraits_BF16, GemmKernelCreation)
{
    auto gemm = ActivationTraits<BF16Tensor>::create_activation_gemm();
    ASSERT_NE(gemm, nullptr) << "BF16 GEMM kernel creation failed";
}

TEST(ActivationTraits_BF16, WorkspaceAllocation)
{
    auto workspace = ActivationTraits<BF16Tensor>::allocate_workspace({32, 64});
    ASSERT_NE(workspace, nullptr) << "BF16 workspace allocation failed";

    auto shape = workspace->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 64);
}

// ============================================================================
// FP16Tensor Traits Tests
// ============================================================================

TEST(ActivationTraits_FP16, SoftmaxBasicCorrectness)
{
    constexpr int rows = 4;
    constexpr int cols = 8;
    std::vector<uint16_t> scores(rows * cols);

    // Initialize with sequential values (as FP16)
    for (int i = 0; i < rows * cols; ++i)
    {
        float fp32_val = static_cast<float>(i) / 10.0f;

#if defined(__F16C__)
        __m128 fp32_vec = _mm_set_ss(fp32_val);
        __m128i fp16_vec = _mm_cvtps_ph(fp32_vec, _MM_FROUND_TO_NEAREST_INT);
        scores[i] = static_cast<uint16_t>(_mm_cvtsi128_si32(fp16_vec));
#else
        // Manual conversion
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &fp32_val, sizeof(float));

        uint32_t sign = (fp32_bits & 0x80000000) >> 16;
        int32_t exp = ((fp32_bits & 0x7F800000) >> 23) - 127 + 15;
        uint32_t mant = (fp32_bits & 0x007FFFFF);

        uint16_t fp16;
        if (exp <= 0)
        {
            if (exp < -10)
            {
                fp16 = static_cast<uint16_t>(sign);
            }
            else
            {
                mant |= 0x00800000;
                mant >>= (1 - exp);
                fp16 = static_cast<uint16_t>(sign | (mant >> 13));
            }
        }
        else if (exp >= 0x1F)
        {
            fp16 = static_cast<uint16_t>(sign | 0x7C00);
        }
        else
        {
            fp16 = static_cast<uint16_t>(sign | (exp << 10) | (mant >> 13));
        }

        scores[i] = fp16;
#endif
    }

    // Apply softmax via traits
    ActivationTraits<FP16Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);

    // Each row should sum to ~1.0 (within FP16 tolerance)
    for (int r = 0; r < rows; ++r)
    {
        auto row_fp32 = fp16_to_fp32(scores.data() + r * cols, cols);
        expect_sums_to_one(row_fp32.data(), cols, FP16_TOLERANCE);
    }
}

TEST(ActivationTraits_FP16, GemmKernelCreation)
{
    auto gemm = ActivationTraits<FP16Tensor>::create_activation_gemm();
    ASSERT_NE(gemm, nullptr) << "FP16 GEMM kernel creation failed";
}

TEST(ActivationTraits_FP16, WorkspaceAllocation)
{
    auto workspace = ActivationTraits<FP16Tensor>::allocate_workspace({32, 64});
    ASSERT_NE(workspace, nullptr) << "FP16 workspace allocation failed";

    auto shape = workspace->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 64);
}

// ============================================================================
// INT32Tensor Traits Tests
// ============================================================================

TEST(ActivationTraits_INT32, SoftmaxConversionStrategy)
{
    constexpr int rows = 2;
    constexpr int cols = 4;
    std::vector<int32_t> scores = {
        100, 200, 300, 400, // Row 0
        50, 150, 250, 350   // Row 1
    };

    // Apply softmax via traits (INT32→FP32→softmax→INT32)
    ActivationTraits<INT32Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);

    // After conversion, values will be in [0,1] range
    // But converted to INT32, so they'll be 0 or 1 (precision loss expected)
    // This test mainly validates that the conversion doesn't crash
    // and produces reasonable output

    // All values should be non-negative
    for (int i = 0; i < rows * cols; ++i)
    {
        EXPECT_GE(scores[i], 0) << "Softmax output should be non-negative";
    }

    // NOTE: This conversion strategy (FP32 probabilities [0,1] → INT32 [0,1])
    // loses significant precision. In production, we'd scale probabilities
    // to a larger range (e.g., [0, 2^16]) before converting to INT32.
    // For now, we just validate the trait compiles and runs without crashing.
}

TEST(ActivationTraits_INT32, GemmKernelCreation)
{
    // INT32 GEMM returns nullptr (INT32 is output-only from INT8 GEMM)
    auto gemm = ActivationTraits<INT32Tensor>::create_activation_gemm();
    EXPECT_EQ(gemm, nullptr) << "INT32 GEMM should return nullptr (output-only tensor type)";
}

TEST(ActivationTraits_INT32, WorkspaceAllocation)
{
    auto workspace = ActivationTraits<INT32Tensor>::allocate_workspace({32, 64});
    ASSERT_NE(workspace, nullptr) << "INT32 workspace allocation failed";

    auto shape = workspace->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 32);
    EXPECT_EQ(shape[1], 64);
}

// ============================================================================
// Cross-Trait Compatibility Tests
// ============================================================================

TEST(ActivationTraits_CrossTrait, AllTraitsSoftmaxCallable)
{
    constexpr int rows = 2;
    constexpr int cols = 4;

    // FP32
    {
        std::vector<float> scores(rows * cols, 1.0f);
        ActivationTraits<FP32Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);
        // Should not crash
    }

    // BF16
    {
        std::vector<uint16_t> scores(rows * cols, 0x3F80); // BF16 representation of 1.0
        ActivationTraits<BF16Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);
        // Should not crash
    }

    // FP16
    {
        std::vector<uint16_t> scores(rows * cols, 0x3C00); // FP16 representation of 1.0
        ActivationTraits<FP16Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);
        // Should not crash
    }

    // INT32
    {
        std::vector<int32_t> scores(rows * cols, 100);
        ActivationTraits<INT32Tensor>::apply_softmax(scores.data(), rows, cols, false, 1.0f);
        // Should not crash
    }

    SUCCEED() << "All trait softmax methods callable without crashing";
}

TEST(ActivationTraits_CrossTrait, AllTraitsGemmCreatable)
{
    auto fp32_gemm = ActivationTraits<FP32Tensor>::create_activation_gemm();
    auto bf16_gemm = ActivationTraits<BF16Tensor>::create_activation_gemm();
    auto fp16_gemm = ActivationTraits<FP16Tensor>::create_activation_gemm();
    auto int32_gemm = ActivationTraits<INT32Tensor>::create_activation_gemm();

    EXPECT_NE(fp32_gemm, nullptr);
    EXPECT_NE(bf16_gemm, nullptr);
    EXPECT_NE(fp16_gemm, nullptr);
    EXPECT_EQ(int32_gemm, nullptr) << "INT32 GEMM should return nullptr (output-only)";
}

TEST(ActivationTraits_CrossTrait, AllTraitsWorkspaceAllocatable)
{
    auto fp32_ws = ActivationTraits<FP32Tensor>::allocate_workspace({16, 32});
    auto bf16_ws = ActivationTraits<BF16Tensor>::allocate_workspace({16, 32});
    auto fp16_ws = ActivationTraits<FP16Tensor>::allocate_workspace({16, 32});
    auto int32_ws = ActivationTraits<INT32Tensor>::allocate_workspace({16, 32});

    EXPECT_NE(fp32_ws, nullptr);
    EXPECT_NE(bf16_ws, nullptr);
    EXPECT_NE(fp16_ws, nullptr);
    EXPECT_NE(int32_ws, nullptr);
}
