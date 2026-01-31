/**
 * @file Test__OnlineSoftmaxV2_Unit.cpp
 * @brief Unit tests for V2 deferred normalization online softmax
 *
 * These tests prove the mathematical correctness of the V2 implementation
 * BEFORE integrating it into the attention pipeline.
 *
 * Key properties verified:
 * 1. Deferred normalization produces same result as running average
 * 2. __int128 rescale matches FP64 reference
 * 3. Chunked accumulation stays within INT32 bounds
 * 4. sum_w_scaled tracking is exact
 */

#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>
#include <array>
#include <iostream>
#include <iomanip>

namespace
{

    // ============================================================================
    // V2 Helper Functions (to be moved to OnlineSoftmax.h after validation)
    // ============================================================================

    /**
     * @brief 128-bit rescale for overflow-safe int64_t × int32_t
     */
    inline int64_t rescale_int64(int64_t value, int32_t scale_num, int scale_shift)
    {
#if defined(__GNUC__) || defined(__clang__)
        __int128 product = static_cast<__int128>(value) * static_cast<__int128>(scale_num);
        return static_cast<int64_t>(product >> scale_shift);
#else
        // Fallback for testing (may lose precision for very large values)
        return static_cast<int64_t>(
            static_cast<double>(value) * scale_num / (1ULL << scale_shift));
#endif
    }

    /**
     * @brief V2 online softmax state (deferred normalization)
     */
    struct OnlineSoftmaxStateV2
    {
        int32_t m = std::numeric_limits<int32_t>::min(); // Running max
        int64_t sum_w_unscaled = 0;                      // Weight sum before shift (for verification)
        int64_t sum_w_scaled = 0;                        // Weight sum after shift (for final divide)
        int count = 0;

        // Configuration
        int weight_shift = 20; // 30-bit LUT → 10-bit VNNI-safe
        int chunk_size = 60;

        void reset()
        {
            m = std::numeric_limits<int32_t>::min();
            sum_w_unscaled = 0;
            sum_w_scaled = 0;
            count = 0;
        }
    };

} // namespace

// ============================================================================
// Test: __int128 Rescale Correctness
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, Rescale128_MatchesFP64_SmallValues)
{
    // Small values that fit in 64-bit multiply
    int64_t value = 12345678;
    int32_t scale_num = 256;
    int scale_shift = 8;

    // FP64 reference
    double fp_result = static_cast<double>(value) * scale_num / (1ULL << scale_shift);

    // __int128 implementation
    int64_t int_result = rescale_int64(value, scale_num, scale_shift);

    EXPECT_EQ(int_result, static_cast<int64_t>(fp_result))
        << "Small value rescale mismatch";
}

TEST(Test__OnlineSoftmaxV2_Unit, Rescale128_MatchesFP64_LargeValues)
{
    // Large values that would overflow 64-bit multiply
    int64_t value = (1LL << 50);   // ~1 quadrillion
    int32_t scale_num = (1 << 28); // ~268 million
    int scale_shift = 30;

    // FP64 reference (may lose some precision)
    double fp_result = static_cast<double>(value) * scale_num / (1ULL << scale_shift);

    // __int128 implementation (exact)
    int64_t int_result = rescale_int64(value, scale_num, scale_shift);

    // Allow small relative error due to FP64 precision limits
    double rel_error = std::abs(static_cast<double>(int_result) - fp_result) / std::abs(fp_result);
    EXPECT_LT(rel_error, 1e-10)
        << "Large value rescale error: int=" << int_result
        << " fp=" << fp_result << " rel_err=" << rel_error;
}

TEST(Test__OnlineSoftmaxV2_Unit, Rescale128_EdgeCases)
{
    // Zero value
    EXPECT_EQ(rescale_int64(0, 12345, 10), 0);

    // Scale factor = 1 (scale_num = 2^scale_shift)
    int64_t value = 9876543210LL;
    EXPECT_EQ(rescale_int64(value, 1 << 20, 20), value);

    // Negative value
    int64_t neg_value = -123456789LL;
    int64_t result = rescale_int64(neg_value, 100, 5);
    EXPECT_LT(result, 0) << "Sign should be preserved";
}

// ============================================================================
// Test: Chunk Accumulation Overflow Safety
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, ChunkAccumulation_NoOverflow_10BitWeights)
{
    constexpr int CHUNK_SIZE = 60;
    constexpr int HEAD_DIM = 128;
    constexpr int32_t MAX_WEIGHT = 1023; // 10-bit after >> 20
    constexpr int16_t MAX_V = 32767;

    std::vector<int32_t> chunk_accum(HEAD_DIM, 0);

    // Worst case: max weight × max V × chunk_size
    for (int k = 0; k < CHUNK_SIZE; ++k)
    {
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            int32_t product = MAX_WEIGHT * MAX_V;

            // Verify no overflow before accumulation
            int64_t new_val = static_cast<int64_t>(chunk_accum[d]) + product;
            ASSERT_LT(std::abs(new_val), static_cast<int64_t>(INT32_MAX))
                << "Overflow at k=" << k << " d=" << d;

            chunk_accum[d] += product;
        }
    }

    // Verify max accumulator value
    int64_t expected_max = static_cast<int64_t>(CHUNK_SIZE) * MAX_WEIGHT * MAX_V;
    std::cout << "Max chunk accumulator: " << expected_max
              << " (INT32_MAX=" << INT32_MAX << ")" << std::endl;

    EXPECT_LT(expected_max, static_cast<int64_t>(INT32_MAX))
        << "Chunk size too large for 10-bit weights!";
}

TEST(Test__OnlineSoftmaxV2_Unit, ChunkAccumulation_NoOverflow_8BitWeights)
{
    constexpr int CHUNK_SIZE = 128;
    constexpr int HEAD_DIM = 128;
    constexpr int32_t MAX_WEIGHT = 255; // 8-bit after >> 22
    constexpr int16_t MAX_V = 32767;

    // Worst case calculation
    int64_t expected_max = static_cast<int64_t>(CHUNK_SIZE) * MAX_WEIGHT * MAX_V;

    std::cout << "8-bit weights, chunk=" << CHUNK_SIZE
              << ": max_accum=" << expected_max
              << " (INT32_MAX=" << INT32_MAX << ")" << std::endl;

    EXPECT_LT(expected_max, static_cast<int64_t>(INT32_MAX))
        << "Chunk size too large for 8-bit weights!";
}

// ============================================================================
// Test: Deferred Normalization Equivalence
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, DeferredNormalization_EquivalentToRunningAverage)
{
    constexpr int HEAD_DIM = 64;
    constexpr int KV_LEN = 128;

    // Generate synthetic scores and V values
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> score_dist(-10000, 10000);
    std::uniform_int_distribution<int16_t> v_dist(-1000, 1000);

    std::vector<int32_t> scores(KV_LEN);
    std::vector<std::array<int16_t, HEAD_DIM>> V(KV_LEN);

    for (int k = 0; k < KV_LEN; ++k)
    {
        scores[k] = score_dist(rng);
        for (int d = 0; d < HEAD_DIM; ++d)
        {
            V[k][d] = v_dist(rng);
        }
    }

    // Find max score
    int32_t max_score = *std::max_element(scores.begin(), scores.end());

    // ===== V1: Running average approach (current FP implementation) =====
    std::vector<int32_t> context_v1(HEAD_DIM, 0);
    double l_processed = 0.0;

    for (int k = 0; k < KV_LEN; ++k)
    {
        // Simulate exp2 weight (using FP for simplicity)
        double weight = std::exp2(static_cast<double>(scores[k] - max_score) / 1000.0);
        double l_new = l_processed + weight;

        if (l_new > 0.0)
        {
            for (int d = 0; d < HEAD_DIM; ++d)
            {
                double numerator = static_cast<double>(context_v1[d]) * l_processed + weight * V[k][d];
                context_v1[d] = static_cast<int32_t>(std::round(numerator / l_new));
            }
        }
        l_processed = l_new;
    }

    // ===== V2: Deferred normalization approach =====
    std::vector<int64_t> context_v2(HEAD_DIM, 0);
    int64_t sum_w = 0;

    for (int k = 0; k < KV_LEN; ++k)
    {
        // Same weight calculation, but scale to integer
        int64_t weight = static_cast<int64_t>(
            std::exp2(static_cast<double>(scores[k] - max_score) / 1000.0) * 1000000);
        sum_w += weight;

        for (int d = 0; d < HEAD_DIM; ++d)
        {
            context_v2[d] += weight * V[k][d];
        }
    }

    // Finalize: single division
    std::vector<int32_t> context_v2_final(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        context_v2_final[d] = static_cast<int32_t>(context_v2[d] / sum_w);
    }

    // ===== Compare =====
    int mismatches = 0;
    int max_diff = 0;
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        int diff = std::abs(context_v1[d] - context_v2_final[d]);
        max_diff = std::max(max_diff, diff);
        if (diff > 2)
        { // Allow small tolerance for FP precision
            mismatches++;
            if (mismatches <= 5)
            {
                std::cout << "Mismatch at d=" << d
                          << ": V1=" << context_v1[d]
                          << " V2=" << context_v2_final[d]
                          << " diff=" << diff << std::endl;
            }
        }
    }

    std::cout << "Total mismatches (|diff| > 2): " << mismatches
              << "/" << HEAD_DIM << ", max_diff=" << max_diff << std::endl;

    EXPECT_LE(mismatches, HEAD_DIM / 10)
        << "Too many mismatches between V1 and V2";
    EXPECT_LE(max_diff, 5)
        << "Max difference too large";
}

// ============================================================================
// Test: Weight Sum Tracking
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, WeightSumTracking_ExactMatch)
{
    constexpr int N = 100;
    constexpr int WEIGHT_SHIFT = 20;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int32_t> weight_dist(0, (1 << 30) - 1); // 30-bit weights

    std::vector<int32_t> weights_raw(N);
    for (int i = 0; i < N; ++i)
    {
        weights_raw[i] = weight_dist(rng);
    }

    // Method 1: Accumulate raw, then shift sum
    int64_t sum_raw = 0;
    for (int i = 0; i < N; ++i)
    {
        sum_raw += weights_raw[i];
    }
    int64_t sum_shifted_after = sum_raw >> WEIGHT_SHIFT;

    // Method 2: Shift then accumulate (V2 approach)
    int64_t sum_shifted_during = 0;
    for (int i = 0; i < N; ++i)
    {
        sum_shifted_during += (weights_raw[i] >> WEIGHT_SHIFT);
    }

    // These are NOT equal due to truncation!
    std::cout << "Sum(raw) >> shift = " << sum_shifted_after << std::endl;
    std::cout << "Sum(raw >> shift) = " << sum_shifted_during << std::endl;
    std::cout << "Difference = " << (sum_shifted_after - sum_shifted_during) << std::endl;

    // V2 uses Method 2 for BOTH numerator and denominator,
    // so the division is consistent:
    // Σ((w>>s) × V) / Σ(w>>s) uses the same shifted weights throughout

    // The key insight: as long as we use the SAME shifted weight in both
    // the P×V accumulation AND the sum, the division is exact.

    // Simulate with synthetic V values
    constexpr int HEAD_DIM = 64;
    std::vector<int16_t> V_vals(N);
    std::uniform_int_distribution<int16_t> v_dist(-1000, 1000);
    for (int i = 0; i < N; ++i)
    {
        V_vals[i] = v_dist(rng);
    }

    // Compute context using V2 approach (consistent scaled weights)
    int64_t context_accum = 0;
    int64_t sum_w_scaled = 0;
    for (int i = 0; i < N; ++i)
    {
        int32_t w_scaled = weights_raw[i] >> WEIGHT_SHIFT;
        context_accum += static_cast<int64_t>(w_scaled) * V_vals[i];
        sum_w_scaled += w_scaled;
    }

    int32_t context_v2 = (sum_w_scaled > 0)
                             ? static_cast<int32_t>(context_accum / sum_w_scaled)
                             : 0;

    // Compute reference using FP64
    double fp_numerator = 0.0;
    double fp_denominator = 0.0;
    for (int i = 0; i < N; ++i)
    {
        double w = static_cast<double>(weights_raw[i] >> WEIGHT_SHIFT);
        fp_numerator += w * V_vals[i];
        fp_denominator += w;
    }
    double context_fp = (fp_denominator > 0) ? (fp_numerator / fp_denominator) : 0.0;

    std::cout << "Context V2 (int64): " << context_v2 << std::endl;
    std::cout << "Context FP64: " << context_fp << std::endl;

    EXPECT_NEAR(context_v2, context_fp, 1.0)
        << "V2 integer division should match FP64 reference";
}

// ============================================================================
// Test: INT64 Total Accumulation Safety
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, TotalAccumulation_NoOverflow_32K_Sequence)
{
    constexpr int KV_LEN = 32768; // 32K sequence
    constexpr int HEAD_DIM = 128;
    constexpr int32_t MAX_WEIGHT_SCALED = 1023; // 10-bit
    constexpr int16_t MAX_V = 32767;

    // Worst case: all positions have max weight and max V
    int64_t max_per_position = static_cast<int64_t>(MAX_WEIGHT_SCALED) * MAX_V;
    int64_t max_total = static_cast<int64_t>(KV_LEN) * max_per_position;

    std::cout << "32K sequence worst case:" << std::endl;
    std::cout << "  Per position max: " << max_per_position << " ("
              << std::log2(max_per_position) << " bits)" << std::endl;
    std::cout << "  Total max: " << max_total << " ("
              << std::log2(max_total) << " bits)" << std::endl;
    std::cout << "  INT64_MAX: " << std::numeric_limits<int64_t>::max()
              << " (63 bits)" << std::endl;

    EXPECT_LT(std::log2(max_total), 63.0)
        << "INT64 overflow for 32K sequence!";
}

TEST(Test__OnlineSoftmaxV2_Unit, TotalAccumulation_NoOverflow_128K_Sequence)
{
    constexpr int KV_LEN = 131072;              // 128K sequence
    constexpr int32_t MAX_WEIGHT_SCALED = 1023; // 10-bit
    constexpr int16_t MAX_V = 32767;

    int64_t max_per_position = static_cast<int64_t>(MAX_WEIGHT_SCALED) * MAX_V;
    int64_t max_total = static_cast<int64_t>(KV_LEN) * max_per_position;

    std::cout << "128K sequence worst case: " << max_total
              << " (" << std::log2(max_total) << " bits)" << std::endl;

    EXPECT_LT(std::log2(max_total), 63.0)
        << "INT64 overflow for 128K sequence!";
}

// ============================================================================
// Test: Rescale After Max Change
// ============================================================================

TEST(Test__OnlineSoftmaxV2_Unit, RescaleAfterMaxChange_PreservesRatio)
{
    constexpr int HEAD_DIM = 64;

    // Simulate: we've accumulated context for positions 0..99
    // Then position 100 has a higher score, requiring rescale

    std::mt19937 rng(999);
    std::vector<int64_t> context_before(HEAD_DIM);
    std::uniform_int_distribution<int64_t> ctx_dist(1000000, 100000000);
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        context_before[d] = ctx_dist(rng);
    }
    int64_t sum_w_before = 50000000;

    // Compute ratio before rescale
    std::vector<double> ratio_before(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        ratio_before[d] = static_cast<double>(context_before[d]) / sum_w_before;
    }

    // Apply rescale (simulate exp2(-5) ≈ 0.03125)
    int32_t scale_num = 1 << 25; // ≈ 0.03125 × 2^30
    int scale_shift = 30;

    std::vector<int64_t> context_after(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        context_after[d] = rescale_int64(context_before[d], scale_num, scale_shift);
    }
    int64_t sum_w_after = rescale_int64(sum_w_before, scale_num, scale_shift);

    // Compute ratio after rescale
    std::vector<double> ratio_after(HEAD_DIM);
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        ratio_after[d] = static_cast<double>(context_after[d]) / sum_w_after;
    }

    // Ratios should be preserved!
    double max_ratio_diff = 0.0;
    for (int d = 0; d < HEAD_DIM; ++d)
    {
        double diff = std::abs(ratio_before[d] - ratio_after[d]);
        max_ratio_diff = std::max(max_ratio_diff, diff);
    }

    std::cout << "Max ratio difference after rescale: " << max_ratio_diff << std::endl;

    EXPECT_LT(max_ratio_diff, 1e-6)
        << "Rescale should preserve context/sum_w ratio";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
