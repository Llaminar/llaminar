/**
 * @file Test__RoPEPrecisionCorrectness.cpp
 * @brief Mathematical correctness tests for RoPE native precision primitives
 * @author David Sanftenberg
 *
 * Tests verify:
 * 1. Scalar/AVX2/AVX512 implementation parity (bit-exact for same precision)
 * 2. Cross-precision accuracy (BF16/FP16 vs FP32 within tolerance)
 * 3. Edge cases (various head_dim, seq_len, position values)
 */

#include <gtest/gtest.h>
#include "kernels/cpu/primitives/RoPEPrimitives.h"
#include "tensors/SIMDHelpers.h"
#include "tensors/FP16Utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

namespace llaminar2::test
{

    /**
     * @brief Test fixture for RoPE precision correctness
     */
    class RoPEPrecisionCorrectnessTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Use deterministic random seed for reproducible tests
            rng_.seed(42);
        }

        /**
         * @brief Generate random FP32 data in range [-1, 1]
         */
        std::vector<float> generateRandomFP32(size_t count)
        {
            std::vector<float> data(count);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(rng_);
            }
            return data;
        }

        /**
         * @brief Convert FP32 to BF16 and back
         */
        std::vector<uint16_t> fp32ToBF16(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> bf16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                bf16[i] = simd::fp32_to_bf16(fp32[i]);
            }
            return bf16;
        }

        /**
         * @brief Convert BF16 to FP32
         */
        std::vector<float> bf16ToFP32(const std::vector<uint16_t> &bf16)
        {
            std::vector<float> fp32(bf16.size());
            for (size_t i = 0; i < bf16.size(); ++i)
            {
                fp32[i] = simd::bf16_to_fp32(bf16[i]);
            }
            return fp32;
        }

        /**
         * @brief Convert FP32 to FP16 and back
         */
        std::vector<uint16_t> fp32ToFP16(const std::vector<float> &fp32)
        {
            std::vector<uint16_t> fp16(fp32.size());
            for (size_t i = 0; i < fp32.size(); ++i)
            {
                fp16[i] = fp32_to_fp16(fp32[i]);
            }
            return fp16;
        }

        /**
         * @brief Convert FP16 to FP32
         */
        std::vector<float> fp16ToFP32(const std::vector<uint16_t> &fp16)
        {
            std::vector<float> fp32(fp16.size());
            for (size_t i = 0; i < fp16.size(); ++i)
            {
                fp32[i] = fp16_to_fp32(fp16[i]);
            }
            return fp32;
        }

        /**
         * @brief Compare two float vectors with tolerance
         */
        void expectNear(const std::vector<float> &a, const std::vector<float> &b,
                        float abs_tol, const std::string &msg)
        {
            ASSERT_EQ(a.size(), b.size()) << msg << ": Size mismatch";

            float max_diff = 0.0f;
            size_t max_diff_idx = 0;

            for (size_t i = 0; i < a.size(); ++i)
            {
                float diff = std::abs(a[i] - b[i]);
                if (diff > max_diff)
                {
                    max_diff = diff;
                    max_diff_idx = i;
                }
                EXPECT_NEAR(a[i], b[i], abs_tol)
                    << msg << ": Mismatch at index " << i
                    << " (a=" << a[i] << ", b=" << b[i] << ")";
            }

            if (max_diff > abs_tol)
            {
                std::cout << msg << ": Max diff = " << max_diff
                          << " at index " << max_diff_idx
                          << " (a=" << a[max_diff_idx] << ", b=" << b[max_diff_idx] << ")"
                          << std::endl;
            }
        }

        /**
         * @brief Compare two uint16_t vectors (bit-exact)
         */
        void expectBitExact(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b,
                            const std::string &msg)
        {
            ASSERT_EQ(a.size(), b.size()) << msg << ": Size mismatch";

            for (size_t i = 0; i < a.size(); ++i)
            {
                EXPECT_EQ(a[i], b[i])
                    << msg << ": Mismatch at index " << i
                    << " (a=" << a[i] << ", b=" << b[i] << ")";
            }
        }

        /**
         * @brief Compare two uint16_t vectors with ULP tolerance
         *
         * SIMD implementations may have different rounding behavior than scalar
         * (e.g., truncation vs round-to-nearest-even in FP32→BF16 conversion).
         * A 1 ULP difference is acceptable for BF16 operations.
         *
         * @param ulp_tolerance Maximum ULP difference allowed (typically 1)
         */
        void expectNearULP(const std::vector<uint16_t> &a, const std::vector<uint16_t> &b,
                           uint16_t ulp_tolerance, const std::string &msg)
        {
            ASSERT_EQ(a.size(), b.size()) << msg << ": Size mismatch";

            size_t mismatch_count = 0;
            for (size_t i = 0; i < a.size(); ++i)
            {
                uint16_t diff = (a[i] > b[i]) ? (a[i] - b[i]) : (b[i] - a[i]);
                if (diff > ulp_tolerance)
                {
                    ++mismatch_count;
                    // Only report first few mismatches to avoid log spam
                    if (mismatch_count <= 5)
                    {
                        ADD_FAILURE() << msg << ": ULP diff too large at index " << i
                                      << " (a=" << a[i] << ", b=" << b[i]
                                      << ", diff=" << diff << ", max=" << ulp_tolerance << ")";
                    }
                }
            }
            if (mismatch_count > 5)
            {
                ADD_FAILURE() << msg << ": ... and " << (mismatch_count - 5) << " more ULP mismatches";
            }
        }

        /**
         * @brief Compute cosine similarity between two vectors
         */
        float cosineSimilarity(const std::vector<float> &a, const std::vector<float> &b)
        {
            if (a.size() != b.size() || a.empty())
                return 0.0f;

            double dot = 0.0;
            double norm_a = 0.0;
            double norm_b = 0.0;

            for (size_t i = 0; i < a.size(); ++i)
            {
                dot += a[i] * b[i];
                norm_a += a[i] * a[i];
                norm_b += b[i] * b[i];
            }

            if (norm_a == 0.0 || norm_b == 0.0)
                return 0.0f;
            return static_cast<float>(dot / (std::sqrt(norm_a) * std::sqrt(norm_b)));
        }

        /**
         * @brief Quantize FP32 to Q8_1 blocks (Scalar reference implementation)
         */
        std::vector<Q8_1Block> quantizeQ8_1(const std::vector<float> &fp32)
        {
            size_t num_blocks = fp32.size() / 32;
            std::vector<Q8_1Block> blocks(num_blocks);

            for (size_t b = 0; b < num_blocks; ++b)
            {
                float max_abs = 0.0f;
                for (size_t i = 0; i < 32; ++i)
                {
                    max_abs = std::max(max_abs, std::abs(fp32[b * 32 + i]));
                }

                float d = max_abs / 127.0f;
                if (d < 1e-10f)
                    d = 1e-10f; // Avoid div by zero

                blocks[b].d = fp32_to_fp16(d);
                blocks[b].sum_qs = 0;

                for (size_t i = 0; i < 32; ++i)
                {
                    float val = fp32[b * 32 + i];
                    int8_t q = static_cast<int8_t>(std::round(val / d));
                    blocks[b].qs[i] = q;
                    blocks[b].sum_qs += q;
                }
            }
            return blocks;
        }

        /**
         * @brief Dequantize Q8_1 blocks to FP32 (Scalar reference implementation)
         */
        std::vector<float> dequantizeQ8_1(const std::vector<Q8_1Block> &blocks)
        {
            std::vector<float> fp32(blocks.size() * 32);
            for (size_t b = 0; b < blocks.size(); ++b)
            {
                float d = fp16_to_fp32(blocks[b].d);
                for (size_t i = 0; i < 32; ++i)
                {
                    fp32[b * 32 + i] = blocks[b].qs[i] * d;
                }
            }
            return fp32;
        }

        std::mt19937 rng_;
    };

    // ============================================================================
    // FP32 Implementation Parity Tests (Scalar vs AVX2 vs AVX512)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, FP32_ScalarVsAVX2_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        // Generate random head data
        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx2 = head_fp32;

        // Get inverse frequencies
        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        // Apply scalar variant
        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

// Apply AVX2 variant + scalar tail
#if defined(__AVX2__)
        int processed = primitives::apply_rope_to_head_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
#else
        head_avx2 = head_scalar; // Fallback if no AVX2
#endif

        // Should be bit-exact (same precision, same operations)
        expectNear(head_scalar, head_avx2, 1e-6f, "FP32 Scalar vs AVX2");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FP32_ScalarVsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        expectNear(head_scalar, head_avx512, 1e-6f, "FP32 Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FP32_AVX2VsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_avx2 = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

#if defined(__AVX2__)
        int processed_avx2 = primitives::apply_rope_to_head_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx2.data(), position, inv_freq, head_dim, processed_avx2);
#endif

#if defined(__AVX512F__)
        int processed_avx512 = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed_avx512);
#else
        head_avx512 = head_avx2;
#endif

        expectNear(head_avx2, head_avx512, 1e-6f, "FP32 AVX2 vs AVX512");
    }

    // ============================================================================
    // BF16 Implementation Parity Tests (Scalar vs AVX2 vs AVX512)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, BF16_ScalarVsAVX2_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        // Generate random data and convert to BF16
        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_bf16 = fp32ToBF16(head_fp32);
        auto head_scalar = head_bf16;
        auto head_avx2 = head_bf16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        // Apply scalar variant
        primitives::apply_rope_to_head_bf16_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

// Apply AVX2 variant + scalar tail
#if defined(__AVX2__)
        int processed = primitives::apply_rope_to_head_bf16_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_bf16_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
#else
        head_avx2 = head_scalar;
#endif

        // Allow 1 ULP difference due to SIMD truncation vs scalar rounding in FP32→BF16
        // Scalar uses round-to-nearest-even, SIMD uses shift-right (truncation)
        expectNearULP(head_scalar, head_avx2, 1, "BF16 Scalar vs AVX2");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, BF16_ScalarVsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_bf16 = fp32ToBF16(head_fp32);
        auto head_scalar = head_bf16;
        auto head_avx512 = head_bf16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_bf16_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_bf16_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_bf16_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        // Allow 1 ULP difference due to SIMD truncation vs scalar rounding in FP32→BF16
        expectNearULP(head_scalar, head_avx512, 1, "BF16 Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, BF16_AVX2VsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_bf16 = fp32ToBF16(head_fp32);
        auto head_avx2 = head_bf16;
        auto head_avx512 = head_bf16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

#if defined(__AVX2__)
        int processed_avx2 = primitives::apply_rope_to_head_bf16_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_bf16_scalar(head_avx2.data(), position, inv_freq, head_dim, processed_avx2);
#endif

#if defined(__AVX512F__)
        int processed_avx512 = primitives::apply_rope_to_head_bf16_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_bf16_scalar(head_avx512.data(), position, inv_freq, head_dim, processed_avx512);
#else
        head_avx512 = head_avx2;
#endif

        expectBitExact(head_avx2, head_avx512, "BF16 AVX2 vs AVX512");
    }

    // ============================================================================
    // FP16 Implementation Parity Tests (Scalar vs AVX2 vs AVX512)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, FP16_ScalarVsAVX2_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_fp16 = fp32ToFP16(head_fp32);
        auto head_scalar = head_fp16;
        auto head_avx2 = head_fp16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_fp16_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        int processed = primitives::apply_rope_to_head_fp16_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_fp16_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
#else
        head_avx2 = head_scalar;
#endif

        expectBitExact(head_scalar, head_avx2, "FP16 Scalar vs AVX2");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FP16_ScalarVsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_fp16 = fp32ToFP16(head_fp32);
        auto head_scalar = head_fp16;
        auto head_avx512 = head_fp16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_fp16_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_fp16_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_fp16_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        expectBitExact(head_scalar, head_avx512, "FP16 Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FP16_AVX2VsAVX512_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_fp16 = fp32ToFP16(head_fp32);
        auto head_avx2 = head_fp16;
        auto head_avx512 = head_fp16;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

#if defined(__AVX2__)
        int processed_avx2 = primitives::apply_rope_to_head_fp16_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_fp16_scalar(head_avx2.data(), position, inv_freq, head_dim, processed_avx2);
#endif

#if defined(__AVX512F__)
        int processed_avx512 = primitives::apply_rope_to_head_fp16_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_fp16_scalar(head_avx512.data(), position, inv_freq, head_dim, processed_avx512);
#else
        head_avx512 = head_avx2;
#endif

        expectBitExact(head_avx2, head_avx512, "FP16 AVX2 vs AVX512");
    }

    // ============================================================================
    // Cross-Precision Accuracy Tests (BF16/FP16 vs FP32 Reference)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, BF16_VsFP32_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        // Generate random FP32 data
        auto head_fp32_orig = generateRandomFP32(head_dim);

        // Create copies for each precision
        auto head_fp32 = head_fp32_orig;
        auto head_bf16 = fp32ToBF16(head_fp32_orig);

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        // Apply RoPE in FP32
        primitives::apply_rope_to_head_scalar(head_fp32.data(), position, inv_freq, head_dim, 0);

        // Apply RoPE in BF16
        primitives::apply_rope_to_head_bf16_scalar(head_bf16.data(), position, inv_freq, head_dim, 0);

        // Convert BF16 result to FP32 for comparison
        auto head_bf16_as_fp32 = bf16ToFP32(head_bf16);

        // BF16 has ~3 decimal digits of precision, so expect reasonable tolerance
        // Rotation involves sin/cos which can amplify errors, so be more permissive
        expectNear(head_fp32, head_bf16_as_fp32, 1e-2f, "BF16 vs FP32");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FP16_VsFP32_SingleHead)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f;

        auto head_fp32_orig = generateRandomFP32(head_dim);
        auto head_fp32 = head_fp32_orig;
        auto head_fp16 = fp32ToFP16(head_fp32_orig);

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_fp32.data(), position, inv_freq, head_dim, 0);
        primitives::apply_rope_to_head_fp16_scalar(head_fp16.data(), position, inv_freq, head_dim, 0);

        auto head_fp16_as_fp32 = fp16ToFP32(head_fp16);

        // FP16 has better precision than BF16 (~3-4 decimal digits)
        expectNear(head_fp32, head_fp16_as_fp32, 1e-3f, "FP16 vs FP32");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, INT32_NotSupported)
    {
        const int seq_len = 4;
        const int head_dim = 64;
        const int n_heads = 4;
        const int n_kv_heads = 2;
        const int n_past = 0;
        const float freq_base = 10000.0f;

        std::vector<int32_t> q_int32(seq_len * n_heads * head_dim, 0);
        std::vector<int32_t> k_int32(seq_len * n_kv_heads * head_dim, 0);

        // Should return false (not supported)
        bool result = primitives::apply_rope_int32(
            q_int32.data(), k_int32.data(),
            seq_len, head_dim,
            n_heads, n_kv_heads,
            n_past, freq_base);

        EXPECT_FALSE(result) << "INT32 RoPE should not be supported";
    }

    // ============================================================================
    // Edge Case Tests (Various head_dim, positions, etc.)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, EdgeCase_SmallHeadDim64)
    {
        const int head_dim = 64;
        const int position = 3;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx2 = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        int processed = primitives::apply_rope_to_head_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
#else
        head_avx2 = head_scalar;
#endif

#if defined(__AVX512F__)
        int processed512 = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed512);
#else
        head_avx512 = head_scalar;
#endif

        expectNear(head_scalar, head_avx2, 1e-6f, "head_dim=64 Scalar vs AVX2");
        expectNear(head_scalar, head_avx512, 1e-6f, "head_dim=64 Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, EdgeCase_LargeHeadDim256)
    {
        const int head_dim = 256;
        const int position = 15;
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        expectNear(head_scalar, head_avx512, 1e-6f, "head_dim=256 Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, EdgeCase_LargePosition1024)
    {
        const int head_dim = 128;
        const int position = 1024; // Large position (long context)
        const float freq_base = 10000.0f;

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx2 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX2__)
        int processed = primitives::apply_rope_to_head_avx2(head_avx2.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx2.data(), position, inv_freq, head_dim, processed);
#else
        head_avx2 = head_scalar;
#endif

        expectNear(head_scalar, head_avx2, 1e-6f, "position=1024 Scalar vs AVX2");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, EdgeCase_LLaMAFreqBase)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 10000.0f; // LLaMA uses 10000.0

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        expectNear(head_scalar, head_avx512, 1e-6f, "LLaMA freq_base Scalar vs AVX512");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, EdgeCase_QwenFreqBase)
    {
        const int head_dim = 128;
        const int position = 7;
        const float freq_base = 1000000.0f; // Qwen 2.5 uses 1000000.0

        auto head_fp32 = generateRandomFP32(head_dim);
        auto head_scalar = head_fp32;
        auto head_avx512 = head_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        primitives::apply_rope_to_head_scalar(head_scalar.data(), position, inv_freq, head_dim, 0);

#if defined(__AVX512F__)
        int processed = primitives::apply_rope_to_head_avx512(head_avx512.data(), position, inv_freq, head_dim);
        primitives::apply_rope_to_head_scalar(head_avx512.data(), position, inv_freq, head_dim, processed);
#else
        head_avx512 = head_scalar;
#endif

        expectNear(head_scalar, head_avx512, 1e-6f, "Qwen freq_base Scalar vs AVX512");
    }

    // ============================================================================
    // Full Tensor Tests (Multiple heads, sequences)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, FullTensor_BF16_MultipleHeads)
    {
        const int seq_len = 4;
        const int head_dim = 64;
        const int q_heads = 4;
        const int k_heads = 2; // GQA
        const int n_past = 0;
        const float freq_base = 10000.0f;

        // Generate random Q and K tensors
        auto q_fp32 = generateRandomFP32(seq_len * q_heads * head_dim);
        auto k_fp32 = generateRandomFP32(seq_len * k_heads * head_dim);

        // Convert to BF16
        auto q_bf16_ref = fp32ToBF16(q_fp32);
        auto k_bf16_ref = fp32ToBF16(k_fp32);
        auto q_bf16_test = q_bf16_ref;
        auto k_bf16_test = k_bf16_ref;

        // Apply RoPE using vectorized (production) path
        primitives::apply_rope_bf16(
            q_bf16_test.data(), k_bf16_test.data(),
            seq_len, head_dim,
            q_heads, k_heads,
            n_past, freq_base,
            nullptr);

        // Apply RoPE using scalar-only path (reference)
        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);
        for (int pos = 0; pos < seq_len; ++pos)
        {
            for (int h = 0; h < q_heads; ++h)
            {
                uint16_t *head_ptr = q_bf16_ref.data() + (pos * q_heads + h) * head_dim;
                primitives::apply_rope_to_head_bf16_scalar(head_ptr, pos + n_past, inv_freq, head_dim, 0);
            }
            for (int h = 0; h < k_heads; ++h)
            {
                uint16_t *head_ptr = k_bf16_ref.data() + (pos * k_heads + h) * head_dim;
                primitives::apply_rope_to_head_bf16_scalar(head_ptr, pos + n_past, inv_freq, head_dim, 0);
            }
        }

        // Compare (Recurrence vs Direct computation may differ by LSBs)
        auto q_ref_f = bf16ToFP32(q_bf16_ref);
        auto q_test_f = bf16ToFP32(q_bf16_test);
        expectNear(q_ref_f, q_test_f, 2e-2f, "BF16 Full Tensor Q");

        auto k_ref_f = bf16ToFP32(k_bf16_ref);
        auto k_test_f = bf16ToFP32(k_bf16_test);
        expectNear(k_ref_f, k_test_f, 2e-2f, "BF16 Full Tensor K");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, FullTensor_FP16_MultipleHeads)
    {
        const int seq_len = 4;
        const int head_dim = 64;
        const int q_heads = 4;
        const int k_heads = 2;
        const int n_past = 0;
        const float freq_base = 10000.0f;

        auto q_fp32 = generateRandomFP32(seq_len * q_heads * head_dim);
        auto k_fp32 = generateRandomFP32(seq_len * k_heads * head_dim);

        auto q_fp16_ref = fp32ToFP16(q_fp32);
        auto k_fp16_ref = fp32ToFP16(k_fp32);
        auto q_fp16_test = q_fp16_ref;
        auto k_fp16_test = k_fp16_ref;

        // Vectorized path
        primitives::apply_rope_fp16(
            q_fp16_test.data(), k_fp16_test.data(),
            seq_len, head_dim,
            q_heads, k_heads,
            n_past, freq_base,
            nullptr);

        // Scalar reference
        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);
        for (int pos = 0; pos < seq_len; ++pos)
        {
            for (int h = 0; h < q_heads; ++h)
            {
                uint16_t *head_ptr = q_fp16_ref.data() + (pos * q_heads + h) * head_dim;
                primitives::apply_rope_to_head_fp16_scalar(head_ptr, pos + n_past, inv_freq, head_dim, 0);
            }
            for (int h = 0; h < k_heads; ++h)
            {
                uint16_t *head_ptr = k_fp16_ref.data() + (pos * k_heads + h) * head_dim;
                primitives::apply_rope_to_head_fp16_scalar(head_ptr, pos + n_past, inv_freq, head_dim, 0);
            }
        }

        // expectBitExact(q_fp16_ref, q_fp16_test, "FP16 Full Tensor Q");
        // expectBitExact(k_fp16_ref, k_fp16_test, "FP16 Full Tensor K");

        // Recurrence vs Direct computation may differ by LSBs
        auto q_ref_f = fp16ToFP32(q_fp16_ref);
        auto q_test_f = fp16ToFP32(q_fp16_test);
        expectNear(q_ref_f, q_test_f, 1e-3f, "FP16 Full Tensor Q");

        auto k_ref_f = fp16ToFP32(k_fp16_ref);
        auto k_test_f = fp16ToFP32(k_fp16_test);
        expectNear(k_ref_f, k_test_f, 1e-3f, "FP16 Full Tensor K");
    }

    // ============================================================================
    // Non-Contiguous Position IDs (Batched Inference)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, NonContiguousPositionIDs_TwoSequences)
    {
        // CRITICAL TEST: Validates RoPE works with non-contiguous position IDs
        // Scenario: Batched inference with 2 independent sequences
        //   Sequence 0: tokens at positions [0, 1]
        //   Sequence 1: tokens at positions [0, 1]
        //   Flattened layout: [seq0_tok0, seq0_tok1, seq1_tok0, seq1_tok1]
        //   Position IDs: [0, 1, 0, 1] (non-contiguous!)
        //
        // Expected: Each sequence should get RoPE applied independently with its own positions

        const int batch_size = 2;
        const int seq_len_per_batch = 2;
        const int total_tokens = batch_size * seq_len_per_batch;
        const int head_dim = 64;
        const int q_heads = 2;
        const int k_heads = 2;
        const float freq_base = 10000.0f;

        // Generate random data for both sequences
        auto q_fp32 = generateRandomFP32(total_tokens * q_heads * head_dim);
        auto k_fp32 = generateRandomFP32(total_tokens * k_heads * head_dim);

        // Non-contiguous position IDs: [0, 1, 0, 1] for 2 sequences of length 2
        std::vector<int> position_ids = {0, 1, 0, 1};

        // Reference: Process each sequence separately with its own positions
        auto q_ref = q_fp32;
        auto k_ref = k_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        // Sequence 0: tokens 0-1, positions 0-1
        for (int tok = 0; tok < seq_len_per_batch; ++tok)
        {
            int position = position_ids[tok];
            for (int h = 0; h < q_heads; ++h)
            {
                float *head_ptr = q_ref.data() + (tok * q_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
            for (int h = 0; h < k_heads; ++h)
            {
                float *head_ptr = k_ref.data() + (tok * k_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
        }

        // Sequence 1: tokens 2-3, positions 0-1 (independent!)
        for (int tok = 0; tok < seq_len_per_batch; ++tok)
        {
            int global_tok = seq_len_per_batch + tok;
            int position = position_ids[global_tok];
            for (int h = 0; h < q_heads; ++h)
            {
                float *head_ptr = q_ref.data() + (global_tok * q_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
            for (int h = 0; h < k_heads; ++h)
            {
                float *head_ptr = k_ref.data() + (global_tok * k_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
        }

        // Test: Apply RoPE with position_ids array (this is what we're implementing!)
        auto q_test = q_fp32;
        auto k_test = k_fp32;

        // TODO: After refactoring, this will be a single call:
        //   primitives::apply_rope_vectorized(q_test.data(), k_test.data(),
        //       total_tokens, head_dim, q_heads, k_heads, position_ids.data(), freq_base, nullptr);
        //
        // For now, manually apply per-token to test expected behavior
        for (int tok = 0; tok < total_tokens; ++tok)
        {
            int position = position_ids[tok];

            // Apply to Q heads for this token
            float *q_token = q_test.data() + tok * q_heads * head_dim;
            primitives::apply_rope_vectorized(
                q_token, nullptr,        // Only Q for this token
                1, head_dim, q_heads, 0, // seq_len=1, k_heads=0 (no K)
                position, freq_base, nullptr);

            // Apply to K heads for this token
            float *k_token = k_test.data() + tok * k_heads * head_dim;
            primitives::apply_rope_vectorized(
                nullptr, k_token,        // Only K for this token
                1, head_dim, 0, k_heads, // seq_len=1, q_heads=0 (no Q)
                position, freq_base, nullptr);
        }

        // Compare: Should match reference when per-token positions are correctly applied
        expectNear(q_ref, q_test, 1e-5f, "Non-contiguous Q");
        expectNear(k_ref, k_test, 1e-5f, "Non-contiguous K");
    }

    TEST_F(RoPEPrecisionCorrectnessTest, NonContiguousPositionIDs_VariableSequenceLengths)
    {
        // Advanced test: Batched inference with DIFFERENT sequence lengths
        // Sequence 0: 1 token at position 5 (continuing from previous context)
        // Sequence 1: 3 tokens at positions [0, 1, 2] (new sequence)
        // Flattened: [seq0_tok0, seq1_tok0, seq1_tok1, seq1_tok2]
        // Position IDs: [5, 0, 1, 2] (highly non-contiguous!)

        const int total_tokens = 4;
        const int head_dim = 64;
        const int q_heads = 2;
        const int k_heads = 2;
        const float freq_base = 10000.0f;

        auto q_fp32 = generateRandomFP32(total_tokens * q_heads * head_dim);
        auto k_fp32 = generateRandomFP32(total_tokens * k_heads * head_dim);

        // Non-contiguous position IDs: [5, 0, 1, 2]
        std::vector<int> position_ids = {5, 0, 1, 2};

        // Reference: Apply per-token positions
        auto q_ref = q_fp32;
        auto k_ref = k_fp32;

        const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

        for (int tok = 0; tok < total_tokens; ++tok)
        {
            int position = position_ids[tok];
            for (int h = 0; h < q_heads; ++h)
            {
                float *head_ptr = q_ref.data() + (tok * q_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
            for (int h = 0; h < k_heads; ++h)
            {
                float *head_ptr = k_ref.data() + (tok * k_heads + h) * head_dim;
                primitives::apply_rope_to_head_scalar(head_ptr, position, inv_freq, head_dim, 0);
            }
        }

        // Test: Apply with position_ids array
        auto q_test = q_fp32;
        auto k_test = k_fp32;

        // TODO: After refactoring, this will be a single call with position_ids array
        // For now, manually apply per-token to test expected behavior
        for (int tok = 0; tok < total_tokens; ++tok)
        {
            int position = position_ids[tok];

            // Apply to Q heads for this token
            float *q_token = q_test.data() + tok * q_heads * head_dim;
            primitives::apply_rope_vectorized(
                q_token, nullptr,
                1, head_dim, q_heads, 0,
                position, freq_base, nullptr);

            // Apply to K heads for this token
            float *k_token = k_test.data() + tok * k_heads * head_dim;
            primitives::apply_rope_vectorized(
                nullptr, k_token,
                1, head_dim, 0, k_heads,
                position, freq_base, nullptr);
        }

        expectNear(q_ref, q_test, 1e-5f, "Variable length Q");
        expectNear(k_ref, k_test, 1e-5f, "Variable length K");
    }

    // ============================================================================
    // Q8_1 Accuracy Tests (Cosine Similarity vs FP32 Golden)
    // ============================================================================

    TEST_F(RoPEPrecisionCorrectnessTest, Q8_1_Accuracy_CosineSimilarity)
    {
        const std::vector<int> head_dims = {64, 128};   // Common head dims
        const std::vector<int> seq_lens = {1, 32, 128}; // Various sequence lengths
        const float freq_base = 10000.0f;
        const float rope_theta = 10000.0f;

        for (int head_dim : head_dims)
        {
            for (int seq_len : seq_lens)
            {
                // 1. Generate random FP32 data (Q tensor)
                // Q8_1 requires head_dim to be multiple of 32
                ASSERT_EQ(head_dim % 32, 0);

                size_t total_elements = seq_len * head_dim;
                auto q_fp32 = generateRandomFP32(total_elements);

                // 2. Create Golden Reference (FP32)
                auto q_golden = q_fp32;
                const auto &inv_freq = primitives::get_inv_freq_cached(head_dim, freq_base);

                // Apply scalar RoPE to golden reference (token by token)
                for (int i = 0; i < seq_len; ++i)
                {
                    primitives::apply_rope_to_head_scalar(
                        q_golden.data() + i * head_dim,
                        i, // position = index for simplicity
                        inv_freq,
                        head_dim,
                        0);
                }

                // 3. Quantize to Q8_1
                auto q_q8_1 = quantizeQ8_1(q_fp32);

                // 4. Apply Q8_1 RoPE (Optimized Kernel)
                // Create position IDs [0, 1, 2, ...]
                std::vector<int> pos_ids(seq_len);
                std::iota(pos_ids.begin(), pos_ids.end(), 0);

                // We treat this as 1 head for simplicity of the test
                primitives::apply_rope_q8_1_integer(
                    q_q8_1.data(),
                    nullptr, // No K tensor
                    pos_ids.data(),
                    seq_len,
                    1, // n_heads
                    0, // n_kv_heads
                    head_dim,
                    rope_theta);

                // 5. Dequantize result
                auto q_result_fp32 = dequantizeQ8_1(q_q8_1);

                // 6. Compute Cosine Similarity
                // We compute similarity per token to ensure every token is correct
                for (int i = 0; i < seq_len; ++i)
                {
                    std::vector<float> golden_token(q_golden.begin() + i * head_dim, q_golden.begin() + (i + 1) * head_dim);
                    std::vector<float> result_token(q_result_fp32.begin() + i * head_dim, q_result_fp32.begin() + (i + 1) * head_dim);

                    float similarity = cosineSimilarity(golden_token, result_token);

                    // Q8_1 is lossy, but RoPE is just rotation, so it should preserve relative structure well.
                    // However, quantization noise exists. 0.999 is a reasonable target for 8-bit.
                    EXPECT_GT(similarity, 0.999f)
                        << "Q8_1 RoPE Cosine Similarity too low at token " << i
                        << " (Head Dim: " << head_dim << ")";
                }
            }
        }
    }

} // namespace llaminar2::test
