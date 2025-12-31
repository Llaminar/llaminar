/**
 * @file Test__VNNIOverflowAnalysis.cpp
 * @brief Comprehensive VNNI INT32 overflow analysis for Q16 attention
 *
 * This test analyzes overflow risks in both Q×K^T and P×V accumulation
 * paths, and determines safe operating parameters.
 *
 * KEY INSIGHT: VPDPWSSD accumulates 2 products per instruction per INT32 lane.
 * The overflow risk depends on HOW MANY products accumulate in each lane.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <limits>

namespace
{

    // ============================================================================
    // VNNI Safety Constants
    // ============================================================================

    /**
     * These constants define the VNNI overflow prevention contract.
     * They should be moved to a shared header once the design is finalized.
     */
    namespace vnni_safety
    {

        constexpr int64_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max(); // 2,147,483,647
        constexpr int16_t INT16_MAX_VAL = 32767;

        /**
         * @brief Calculate max safe INT16 value for a given head dimension
         *
         * VPDPWSSD accumulates (head_dim / 16) products per INT32 lane.
         * To prevent overflow: N × val² ≤ INT32_MAX
         * Therefore: val ≤ sqrt(INT32_MAX / N)
         */
        constexpr int16_t max_safe_int16(int head_dim)
        {
            // Products per INT32 lane
            int products_per_lane = head_dim / 16;
            // Max val² before overflow
            int64_t max_val_squared = INT32_MAX_VAL / products_per_lane;
            // Max safe value (conservative floor)
            return static_cast<int16_t>(std::sqrt(static_cast<double>(max_val_squared)));
        }

        // Pre-computed safe limits for common head dimensions
        constexpr int16_t MAX_SAFE_INT16_64 = 23170;  // sqrt(2^31 / 4)
        constexpr int16_t MAX_SAFE_INT16_96 = 18918;  // sqrt(2^31 / 6)
        constexpr int16_t MAX_SAFE_INT16_128 = 16383; // sqrt(2^31 / 8)
        constexpr int16_t MAX_SAFE_INT16_192 = 13377; // sqrt(2^31 / 12)
        constexpr int16_t MAX_SAFE_INT16_256 = 11585; // sqrt(2^31 / 16)

        // Conservative default: safe for head_dim ≤ 192
        constexpr int16_t MAX_SAFE_INT16_DEFAULT = MAX_SAFE_INT16_192;

        /**
         * @brief Calculate equivalent FP32 range for a given INT16 limit and scale
         */
        constexpr float safe_fp32_range(int16_t max_int16, float kv_cache_scale)
        {
            return static_cast<float>(max_int16) * kv_cache_scale / 32767.0f;
        }

    } // namespace vnni_safety

    /**
     * @brief Calculate max products per INT32 lane before overflow
     *
     * Given INT16 values of magnitude |val|, how many products can accumulate?
     * N × val² ≤ INT32_MAX → N ≤ INT32_MAX / val²
     */
    int max_safe_products(int16_t max_int16_val)
    {
        int64_t val_squared = static_cast<int64_t>(max_int16_val) * max_int16_val;
        return static_cast<int>(vnni_safety::INT32_MAX_VAL / val_squared);
    }

    /**
     * @brief Calculate INT16 value from FP32 range and kv_cache_scale (unclamped)
     * Note: This can overflow INT16 for extreme values - use fp32_to_int16_safe for production
     */
    int16_t fp32_to_int16(float fp32_val, float kv_cache_scale)
    {
        return static_cast<int16_t>(std::round(fp32_val * 32767.0f / kv_cache_scale));
    }

    /**
     * @brief Convert FP32 to INT32 first, then show what needs clipping
     * This is the proper implementation for production code
     */
    int32_t fp32_to_int32(float fp32_val, float kv_cache_scale)
    {
        return static_cast<int32_t>(std::round(fp32_val * 32767.0f / kv_cache_scale));
    }

    /**
     * @brief Clip INT16 to safe range for given head_dim
     */
    int16_t clip_to_safe(int16_t val, int head_dim)
    {
        int16_t limit;
        switch (head_dim)
        {
        case 64:
            limit = vnni_safety::MAX_SAFE_INT16_64;
            break;
        case 96:
            limit = vnni_safety::MAX_SAFE_INT16_96;
            break;
        case 128:
            limit = vnni_safety::MAX_SAFE_INT16_128;
            break;
        case 192:
            limit = vnni_safety::MAX_SAFE_INT16_192;
            break;
        case 256:
            limit = vnni_safety::MAX_SAFE_INT16_256;
            break;
        default:
            limit = vnni_safety::MAX_SAFE_INT16_DEFAULT;
            break;
        }
        return std::max(static_cast<int16_t>(-limit), std::min(val, limit));
    }

    /**
     * @brief Clip INT32 to safe INT16 range for given head_dim (CORRECT implementation)
     */
    int16_t clip_int32_to_safe_int16(int32_t val, int head_dim)
    {
        int16_t limit;
        switch (head_dim)
        {
        case 64:
            limit = vnni_safety::MAX_SAFE_INT16_64;
            break;
        case 96:
            limit = vnni_safety::MAX_SAFE_INT16_96;
            break;
        case 128:
            limit = vnni_safety::MAX_SAFE_INT16_128;
            break;
        case 192:
            limit = vnni_safety::MAX_SAFE_INT16_192;
            break;
        case 256:
            limit = vnni_safety::MAX_SAFE_INT16_256;
            break;
        default:
            limit = vnni_safety::MAX_SAFE_INT16_DEFAULT;
            break;
        }
        int32_t clamped = std::max(static_cast<int32_t>(-limit), std::min(val, static_cast<int32_t>(limit)));
        return static_cast<int16_t>(clamped);
    }

    // ============================================================================
    // Q×K^T Analysis
    // ============================================================================

    class VNNIOverflowAnalysis : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            std::cout << std::fixed << std::setprecision(2);
        }
    };

    /**
     * @brief Analyze Q×K^T dot product overflow risk
     *
     * Q×K^T computes: score = Σ(Q[i] × K[i]) for i in [0, head_dim)
     *
     * With VPDPWSSD:
     * - Each ZMM processes 32 INT16 → 16 partial sums
     * - head_dim/16 products accumulate per INT32 lane
     * - Final horizontal sum combines 16 lanes
     */
    TEST_F(VNNIOverflowAnalysis, QKDotProduct_HeadDimOverflow)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           Q×K^T DOT PRODUCT OVERFLOW ANALYSIS                  ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;
        const float actual_range = 3.0f; // Typical post-norm activations

        int16_t max_int16 = fp32_to_int16(actual_range, kv_cache_scale);
        int max_safe_N = max_safe_products(max_int16);

        std::cout << "  Configuration:\n";
        std::cout << "    kv_cache_scale: " << kv_cache_scale << "\n";
        std::cout << "    actual_range:   ±" << actual_range << "\n";
        std::cout << "    INT16 value:    ±" << max_int16 << "\n";
        std::cout << "    Max safe products per INT32 lane: " << max_safe_N << "\n\n";

        std::cout << "  VPDPWSSD Accumulation Pattern:\n";
        std::cout << "    - 16 INT32 lanes process 32 INT16 pairs per instruction\n";
        std::cout << "    - Products per lane = head_dim / 16\n\n";

        std::cout << "  head_dim | Products/Lane | Max Sum (worst case) | Status\n";
        std::cout << "  ---------+---------------+----------------------+--------\n";

        for (int head_dim : {64, 96, 128, 192, 256})
        {
            int products_per_lane = head_dim / 16; // After all VPDPWSSD for this head_dim
            int64_t max_sum = static_cast<int64_t>(products_per_lane) * max_int16 * max_int16;
            bool safe = max_sum <= vnni_safety::INT32_MAX_VAL;

            std::cout << "  " << std::setw(8) << head_dim
                      << " | " << std::setw(13) << products_per_lane
                      << " | " << std::setw(20) << max_sum
                      << " | " << (safe ? "✓ SAFE" : "✗ OVERFLOW") << "\n";
        }

        // Calculate max safe head_dim
        int max_safe_head_dim = max_safe_N * 16;
        std::cout << "\n  Maximum safe head_dim (without chunking): " << max_safe_head_dim << "\n";

        // With these parameters (actual=3.0, scale=8.0), head_dim up to ~224 is safe
        // This is because products_per_lane = head_dim/16, and we need products_per_lane ≤ 14
        EXPECT_GE(max_safe_head_dim, 192) << "MLA head_dim=192 should be safe for Q×K^T";
    }

    /**
     * @brief Analyze P×V weighted accumulation overflow risk
     *
     * P×V computes: context[d] = Σ(P[k] × V[k][d]) for k in [0, kv_len)
     *
     * CRITICAL: This accumulates across kv_len, not head_dim!
     * For long sequences, this can overflow even with moderate weights.
     */
    TEST_F(VNNIOverflowAnalysis, PVAccumulation_SeqLenOverflow)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           P×V ACCUMULATION OVERFLOW ANALYSIS                   ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;
        const float actual_range = 3.0f;

        int16_t max_v_int16 = fp32_to_int16(actual_range, kv_cache_scale);

        // P weights come from softmax, range [0, 32767] where sum ≈ 32767
        // But for worst-case analysis, assume max weight at each position
        int16_t max_p_int16 = vnni_safety::INT16_MAX_VAL; // Softmax can produce near-1.0 weights

        std::cout << "  Configuration:\n";
        std::cout << "    V values: INT16 ±" << max_v_int16 << "\n";
        std::cout << "    P weights: INT16 [0, " << max_p_int16 << "] (softmax output)\n\n";

        std::cout << "  CRITICAL DIFFERENCE from Q×K^T:\n";
        std::cout << "    - P×V accumulates across kv_len (sequence length)\n";
        std::cout << "    - Each context[d] receives kv_len products\n";
        std::cout << "    - Long sequences = high overflow risk!\n\n";

        // Calculate max safe kv_len for different scenarios
        std::cout << "  Scenario Analysis:\n\n";

        // Scenario 1: Worst case (all weights max, all V max same sign)
        std::cout << "  1. WORST CASE (all weights=32767, all V=" << max_v_int16 << "):\n";
        int64_t product_worst = static_cast<int64_t>(max_p_int16) * max_v_int16;
        int max_safe_kv_worst = static_cast<int>(vnni_safety::INT32_MAX_VAL / product_worst);
        std::cout << "     Product per position: " << product_worst << "\n";
        std::cout << "     Max safe kv_len: " << max_safe_kv_worst << "\n";
        std::cout << "     → EXTREMELY LIMITED!\n\n";

        // Scenario 2: Realistic softmax (weights sum to 32767, distributed)
        // If attention is uniform: each weight ≈ 32767/kv_len
        std::cout << "  2. UNIFORM ATTENTION (weights distributed evenly):\n";
        std::cout << "     For kv_len=N: each weight ≈ 32767/N\n";
        std::cout << "     Max sum ≈ N × (32767/N) × " << max_v_int16 << " = 32767 × " << max_v_int16 << "\n";
        int64_t uniform_max = static_cast<int64_t>(vnni_safety::INT16_MAX_VAL) * max_v_int16;
        std::cout << "     = " << uniform_max << " → ";
        std::cout << (uniform_max <= vnni_safety::INT32_MAX_VAL ? "✓ SAFE for any kv_len!" : "✗ OVERFLOW") << "\n\n";

        // Scenario 3: Spiky attention (one dominant weight)
        std::cout << "  3. SPIKY ATTENTION (one dominant position):\n";
        std::cout << "     One weight ≈ 32767, others ≈ 0\n";
        std::cout << "     Max contribution: 32767 × " << max_v_int16 << " = " << uniform_max << "\n";
        std::cout << "     → " << (uniform_max <= vnni_safety::INT32_MAX_VAL ? "✓ SAFE" : "✗ OVERFLOW") << "\n\n";

        // Scenario 4: Multiple hot positions
        std::cout << "  4. MULTIPLE HOT POSITIONS (k positions with weight 32767/k each):\n";
        std::cout << "     k | Weight Each | Max Sum | Status\n";
        std::cout << "     --+-------------+---------+--------\n";
        for (int k : {1, 2, 4, 8, 16, 32, 64})
        {
            // If k positions have equal high weight
            int16_t weight_each = vnni_safety::INT16_MAX_VAL / k;
            // Worst case: all V values are max and same sign
            int64_t max_sum = static_cast<int64_t>(k) * weight_each * max_v_int16;
            bool safe = max_sum <= vnni_safety::INT32_MAX_VAL;
            std::cout << "     " << std::setw(2) << k
                      << " | " << std::setw(11) << weight_each
                      << " | " << std::setw(7) << max_sum
                      << " | " << (safe ? "✓ SAFE" : "✗ OVER") << "\n";
        }

        std::cout << "\n  CONCLUSION:\n";
        std::cout << "    - P×V is SAFE when softmax weights are naturally distributed\n";
        std::cout << "    - Danger only with artificial/pathological weight patterns\n";
        std::cout << "    - Real attention has sum(P)=1.0 (32767 in INT16) → bounded\n";
    }

    /**
     * @brief The real constraint: softmax weight normalization
     */
    TEST_F(VNNIOverflowAnalysis, SoftmaxConstraint)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           SOFTMAX WEIGHT NORMALIZATION CONSTRAINT              ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;
        const float actual_range = 3.0f;
        int16_t max_v_int16 = fp32_to_int16(actual_range, kv_cache_scale);

        std::cout << "  The CRITICAL insight:\n\n";
        std::cout << "    Softmax produces P[k] such that: Σ P[k] = 1.0 (= 32767 in INT16)\n\n";
        std::cout << "    For P×V:\n";
        std::cout << "      context[d] = Σ P[k] × V[k][d]\n\n";
        std::cout << "    Absolute worst case (all V[k][d] have same sign and max value):\n";
        std::cout << "      |context[d]| ≤ Σ|P[k]| × max|V| = 32767 × " << max_v_int16 << "\n";
        std::cout << "                  = " << (static_cast<int64_t>(vnni_safety::INT16_MAX_VAL) * max_v_int16) << "\n\n";

        int64_t theoretical_max = static_cast<int64_t>(vnni_safety::INT16_MAX_VAL) * max_v_int16;
        bool safe = theoretical_max <= vnni_safety::INT32_MAX_VAL;

        std::cout << "    INT32_MAX = " << vnni_safety::INT32_MAX_VAL << "\n";
        std::cout << "    Theoretical max = " << theoretical_max << "\n";
        std::cout << "    Headroom = " << (vnni_safety::INT32_MAX_VAL - theoretical_max) << "\n";
        std::cout << "    Status: " << (safe ? "✓ SAFE!" : "✗ OVERFLOW!") << "\n\n";

        if (safe)
        {
            std::cout << "  ═══════════════════════════════════════════════════════════════\n";
            std::cout << "  ║ P×V is PROVABLY SAFE when softmax is properly normalized!   ║\n";
            std::cout << "  ║ No chunked accumulation needed for P×V!                     ║\n";
            std::cout << "  ═══════════════════════════════════════════════════════════════\n";
        }

        EXPECT_TRUE(safe) << "P×V should be safe with softmax normalization";
    }

    /**
     * @brief Comprehensive safety summary
     */
    TEST_F(VNNIOverflowAnalysis, SafetySummary)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           VNNI OVERFLOW SAFETY SUMMARY                         ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;
        const float actual_range = 3.0f;
        int16_t max_int16 = fp32_to_int16(actual_range, kv_cache_scale);
        int max_products = max_safe_products(max_int16);
        int max_head_dim = max_products * 16;

        std::cout << "  With kv_cache_scale=" << kv_cache_scale
                  << " and typical activations |x| < " << actual_range << ":\n\n";

        std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │ Q×K^T DOT PRODUCT                                           │\n";
        std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
        std::cout << "  │ Products per INT32 lane: head_dim / 16                      │\n";
        std::cout << "  │ Max safe products: " << std::setw(3) << max_products << "                                      │\n";
        std::cout << "  │ Max safe head_dim: " << std::setw(3) << max_head_dim << " (without chunking)                │\n";
        std::cout << "  │                                                             │\n";
        std::cout << "  │ Status by head_dim:                                         │\n";
        std::cout << "  │   64:  " << std::setw(2) << (64 / 16) << " products/lane → ✓ SAFE                       │\n";
        std::cout << "  │   96:  " << std::setw(2) << (96 / 16) << " products/lane → ✓ SAFE                       │\n";
        std::cout << "  │   128: " << std::setw(2) << (128 / 16) << " products/lane → ✓ SAFE                       │\n";
        std::cout << "  │   192: " << std::setw(2) << (192 / 16) << " products/lane → ✓ SAFE                       │\n";
        std::cout << "  │   256: " << std::setw(2) << (256 / 16) << " products/lane → " << (max_head_dim >= 256 ? "✓ SAFE" : "✗ NEED CHUNKING") << "               │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";

        int64_t pv_max = static_cast<int64_t>(vnni_safety::INT16_MAX_VAL) * max_int16;
        bool pv_safe = pv_max <= vnni_safety::INT32_MAX_VAL;

        std::cout << "  ┌─────────────────────────────────────────────────────────────┐\n";
        std::cout << "  │ P×V WEIGHTED ACCUMULATION                                   │\n";
        std::cout << "  ├─────────────────────────────────────────────────────────────┤\n";
        std::cout << "  │ Constraint: Σ P[k] = 32767 (softmax normalization)          │\n";
        std::cout << "  │ Max |context[d]| = 32767 × " << std::setw(5) << max_int16 << " = " << std::setw(12) << pv_max << "       │\n";
        std::cout << "  │ INT32_MAX = " << std::setw(12) << vnni_safety::INT32_MAX_VAL << "                              │\n";
        std::cout << "  │ Status: " << (pv_safe ? "✓ SAFE for ANY sequence length!" : "✗ OVERFLOW!") << "                     │\n";
        std::cout << "  └─────────────────────────────────────────────────────────────┘\n\n";

        std::cout << "  ═══════════════════════════════════════════════════════════════\n";
        std::cout << "  ║ CONCLUSION: NO CHUNKED ACCUMULATION NEEDED!                 ║\n";
        std::cout << "  ║                                                             ║\n";
        std::cout << "  ║ With kv_cache_scale=8.0 and actual activations |x| < 3.0:   ║\n";
        std::cout << "  ║   - Q×K^T: Safe for head_dim ≤ " << std::setw(3) << max_head_dim << "                        ║\n";
        std::cout << "  ║   - P×V:   Safe for ANY sequence length (softmax bounded)   ║\n";
        std::cout << "  ║                                                             ║\n";
        std::cout << "  ║ The key is FIXED-SCALE quantization that maps:              ║\n";
        std::cout << "  ║   |x| = 3.0 → INT16 = 12288 (not 32767!)                    ║\n";
        std::cout << "  ═══════════════════════════════════════════════════════════════\n";

        EXPECT_GE(max_head_dim, 192) << "MLA support requires head_dim=192 safe";
        EXPECT_TRUE(pv_safe) << "P×V must be safe with softmax normalization";
    }

    /**
     * @brief What if actual activations exceed expected range?
     */
    TEST_F(VNNIOverflowAnalysis, SafetyMargins)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           SAFETY MARGINS: ACTIVATION RANGE SWEEP              ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;

        std::cout << "  With kv_cache_scale=" << kv_cache_scale << ":\n\n";
        std::cout << "  actual |x| | INT16 | Safe Q×K (head_dim) | Safe P×V?\n";
        std::cout << "  -----------+-------+---------------------+-----------\n";

        for (float actual : {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f})
        {
            int16_t max_int16 = fp32_to_int16(actual, kv_cache_scale);
            int max_products = max_safe_products(max_int16);
            int max_head_dim = max_products * 16;
            int64_t pv_max = static_cast<int64_t>(vnni_safety::INT16_MAX_VAL) * max_int16;
            bool pv_safe = pv_max <= vnni_safety::INT32_MAX_VAL;

            std::cout << "  " << std::setw(10) << actual
                      << " | " << std::setw(5) << max_int16
                      << " | " << std::setw(19) << max_head_dim
                      << " | " << (pv_safe ? "✓ YES" : "✗ NO") << "\n";
        }

        std::cout << "\n  NOTE: At |x| = 8.0 (full kv_cache_scale), INT16 = 32767\n";
        std::cout << "        and max safe head_dim = 32 (very limited!)\n";
        std::cout << "        But typical activations are |x| < 4.0, so we have margin.\n";
    }

    /**
     * @brief Test clipping behavior and verify it prevents overflow
     */
    TEST_F(VNNIOverflowAnalysis, ClippingPreventsOverflow)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           INT16 CLIPPING FOR GUARANTEED SAFETY                 ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        const float kv_cache_scale = 8.0f;

        std::cout << "  MAX_SAFE_INT16 limits by head_dim:\n\n";
        std::cout << "    head_dim │ MAX_SAFE_INT16 │ Equiv FP32 Range │ Clipping at\n";
        std::cout << "    ─────────┼────────────────┼──────────────────┼─────────────\n";

        for (int head_dim : {64, 96, 128, 192, 256})
        {
            int16_t limit;
            switch (head_dim)
            {
            case 64:
                limit = vnni_safety::MAX_SAFE_INT16_64;
                break;
            case 96:
                limit = vnni_safety::MAX_SAFE_INT16_96;
                break;
            case 128:
                limit = vnni_safety::MAX_SAFE_INT16_128;
                break;
            case 192:
                limit = vnni_safety::MAX_SAFE_INT16_192;
                break;
            case 256:
                limit = vnni_safety::MAX_SAFE_INT16_256;
                break;
            default:
                limit = vnni_safety::MAX_SAFE_INT16_DEFAULT;
                break;
            }
            float fp32_range = vnni_safety::safe_fp32_range(limit, kv_cache_scale);

            std::cout << "    " << std::setw(8) << head_dim
                      << " │ " << std::setw(14) << limit
                      << " │ " << std::setw(16) << std::setprecision(2) << fp32_range
                      << " │ |x| > " << fp32_range << "\n";
        }

        std::cout << "\n  Testing clipping with outliers:\n\n";

        // Test clipping behavior - use proper INT32 intermediate to show actual values
        for (int head_dim : {128, 192})
        {
            int16_t limit = (head_dim == 128) ? vnni_safety::MAX_SAFE_INT16_128
                                              : vnni_safety::MAX_SAFE_INT16_192;

            // Test extreme outlier
            float extreme_fp32 = 10.0f; // Way beyond normal range
            int32_t unclipped_i32 = fp32_to_int32(extreme_fp32, kv_cache_scale);
            int16_t clipped = clip_int32_to_safe_int16(unclipped_i32, head_dim);

            std::cout << "    head_dim=" << head_dim << ": FP32=" << extreme_fp32
                      << " → INT32=" << unclipped_i32 << " → clipped INT16=" << clipped << "\n";

            EXPECT_LE(std::abs(clipped), limit)
                << "Clipped value should be within safe limit";

            // Verify no overflow with clipped values
            int products_per_lane = head_dim / 16;
            int64_t max_sum = static_cast<int64_t>(products_per_lane) * static_cast<int64_t>(limit) * static_cast<int64_t>(limit);
            EXPECT_LE(max_sum, vnni_safety::INT32_MAX_VAL)
                << "Clipped values should never overflow INT32";
        }

        std::cout << "\n  ═══════════════════════════════════════════════════════════════\n";
        std::cout << "  ║ CONCLUSION: Clipping to MAX_SAFE_INT16 guarantees safety!    ║\n";
        std::cout << "  ║ Even extreme outliers (|x| > 8.0) are handled safely.        ║\n";
        std::cout << "  ═══════════════════════════════════════════════════════════════\n";
    }

    /**
     * @brief Verify pre-computed constants are correct
     */
    TEST_F(VNNIOverflowAnalysis, ConstantsAreCorrect)
    {
        std::cout << "\n╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║           VERIFYING PRE-COMPUTED SAFETY CONSTANTS              ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";

        // Verify each constant is correct
        auto verify_constant = [](int head_dim, int16_t expected, const char *name)
        {
            int products_per_lane = head_dim / 16;
            int64_t max_val_squared = vnni_safety::INT32_MAX_VAL / products_per_lane;
            int16_t computed = static_cast<int16_t>(std::sqrt(static_cast<double>(max_val_squared)));

            std::cout << "    " << name << ": expected=" << expected
                      << ", computed=" << computed;

            // Allow small rounding differences
            bool ok = std::abs(expected - computed) <= 1;
            std::cout << (ok ? " ✓\n" : " ✗\n");

            return ok;
        };

        EXPECT_TRUE(verify_constant(64, vnni_safety::MAX_SAFE_INT16_64, "MAX_SAFE_INT16_64"));
        EXPECT_TRUE(verify_constant(96, vnni_safety::MAX_SAFE_INT16_96, "MAX_SAFE_INT16_96"));
        EXPECT_TRUE(verify_constant(128, vnni_safety::MAX_SAFE_INT16_128, "MAX_SAFE_INT16_128"));
        EXPECT_TRUE(verify_constant(192, vnni_safety::MAX_SAFE_INT16_192, "MAX_SAFE_INT16_192"));
        EXPECT_TRUE(verify_constant(256, vnni_safety::MAX_SAFE_INT16_256, "MAX_SAFE_INT16_256"));
    }

} // anonymous namespace
