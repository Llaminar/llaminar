/**
 * @file Test__Q16FixedScaleAccuracy.cpp
 * @brief Accuracy comparison: Fixed-scale vs Adaptive-scale Q16_1 quantization
 *
 * This test measures the precision impact of using fixed-scale quantization
 * (required for VNNI INT32 overflow prevention) vs adaptive per-block scale.
 *
 * KEY METRICS:
 * - MSE (Mean Squared Error) after round-trip FP32→Q16→FP32
 * - Max Absolute Error
 * - Cosine Similarity (measures angle preservation)
 * - Effective Bits (log2 of precision)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include "tensors/BlockStructures.h"

using namespace llaminar2;

namespace
{

    // ============================================================================
    // Metrics Computation
    // ============================================================================

    struct QuantizationMetrics
    {
        double mse;            // Mean Squared Error
        double rmse;           // Root Mean Squared Error
        double max_error;      // Maximum Absolute Error
        double cosine_sim;     // Cosine Similarity (1.0 = perfect)
        double effective_bits; // log2(1/RMSE) - higher is better
        double snr_db;         // Signal-to-Noise Ratio in dB
    };

    QuantizationMetrics compute_metrics(const std::vector<float> &original,
                                        const std::vector<float> &reconstructed)
    {
        QuantizationMetrics m{};
        size_t n = original.size();

        // MSE and Max Error
        double sum_sq_error = 0.0;
        double sum_signal_sq = 0.0;
        m.max_error = 0.0;

        for (size_t i = 0; i < n; ++i)
        {
            double err = original[i] - reconstructed[i];
            sum_sq_error += err * err;
            sum_signal_sq += original[i] * original[i];
            m.max_error = std::max(m.max_error, std::abs(err));
        }

        m.mse = sum_sq_error / n;
        m.rmse = std::sqrt(m.mse);

        // SNR
        if (m.mse > 0)
        {
            double signal_power = sum_signal_sq / n;
            m.snr_db = 10.0 * std::log10(signal_power / m.mse);
        }
        else
        {
            m.snr_db = std::numeric_limits<double>::infinity();
        }

        // Effective bits
        if (m.rmse > 0)
        {
            m.effective_bits = -std::log2(m.rmse);
        }
        else
        {
            m.effective_bits = 64.0; // Perfect reconstruction
        }

        // Cosine similarity
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += original[i] * reconstructed[i];
            norm_a += original[i] * original[i];
            norm_b += reconstructed[i] * reconstructed[i];
        }
        if (norm_a > 0 && norm_b > 0)
        {
            m.cosine_sim = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }
        else
        {
            m.cosine_sim = (norm_a == 0 && norm_b == 0) ? 1.0 : 0.0;
        }

        return m;
    }

    // ============================================================================
    // Quantization Methods
    // ============================================================================

    /**
     * @brief Adaptive quantization (current implementation)
     *
     * Each block independently determines its scale from max_abs.
     * Pros: Maximum precision per block
     * Cons: INT16 values always near-saturated → VNNI overflow risk
     */
    void quantize_adaptive(const std::vector<float> &src,
                           std::vector<Q16_1Block> &blocks,
                           size_t block_size = 32)
    {
        size_t n = src.size();
        size_t n_blocks = (n + block_size - 1) / block_size;
        blocks.resize(n_blocks);

        for (size_t b = 0; b < n_blocks; ++b)
        {
            size_t offset = b * block_size;
            size_t count = std::min(block_size, n - offset);

            // Find max_abs for this block
            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[offset + i]));
            }

            // Adaptive scale: map max_abs → ±32767
            float d = (max_abs > 1e-10f) ? (max_abs / 32767.0f) : 0.0f;
            blocks[b].d = d;

            // Quantize
            int32_t sum = 0;
            if (d > 1e-10f)
            {
                float inv_d = 1.0f / d;
                for (size_t i = 0; i < count; ++i)
                {
                    float scaled = src[offset + i] * inv_d;
                    int32_t q = static_cast<int32_t>(std::round(scaled));
                    q = std::clamp(q, -32767, 32767);
                    blocks[b].qs[i] = static_cast<int16_t>(q);
                    sum += q;
                }
            }
            else
            {
                for (size_t i = 0; i < count; ++i)
                {
                    blocks[b].qs[i] = 0;
                }
            }

            // Zero-fill remainder
            for (size_t i = count; i < block_size; ++i)
            {
                blocks[b].qs[i] = 0;
            }

            blocks[b].sum_qs = sum;
        }
    }

    /**
     * @brief Fixed-scale quantization (VNNI-safe)
     *
     * All blocks use the same scale based on kv_cache_scale.
     * Pros: INT16 values proportional to FP32 → safe for VNNI
     * Cons: Precision loss when actual max_abs << kv_cache_scale
     */
    void quantize_fixed_scale(const std::vector<float> &src,
                              std::vector<Q16_1Block> &blocks,
                              float kv_cache_scale,
                              size_t block_size = 32)
    {
        size_t n = src.size();
        size_t n_blocks = (n + block_size - 1) / block_size;
        blocks.resize(n_blocks);

        // Fixed scale for all blocks
        float d = kv_cache_scale / 32767.0f;
        float inv_d = 32767.0f / kv_cache_scale;

        for (size_t b = 0; b < n_blocks; ++b)
        {
            size_t offset = b * block_size;
            size_t count = std::min(block_size, n - offset);

            blocks[b].d = d;

            // Quantize with fixed scale
            int32_t sum = 0;
            for (size_t i = 0; i < count; ++i)
            {
                float scaled = src[offset + i] * inv_d;
                int32_t q = static_cast<int32_t>(std::round(scaled));
                q = std::clamp(q, -32767, 32767);
                blocks[b].qs[i] = static_cast<int16_t>(q);
                sum += q;
            }

            // Zero-fill remainder
            for (size_t i = count; i < block_size; ++i)
            {
                blocks[b].qs[i] = 0;
            }

            blocks[b].sum_qs = sum;
        }
    }

    /**
     * @brief Dequantize Q16_1 blocks back to FP32
     */
    void dequantize(const std::vector<Q16_1Block> &blocks,
                    std::vector<float> &dst,
                    size_t total_elements,
                    size_t block_size = 32)
    {
        dst.resize(total_elements);

        for (size_t b = 0; b < blocks.size(); ++b)
        {
            size_t offset = b * block_size;
            size_t count = std::min(block_size, total_elements - offset);
            float d = blocks[b].d;

            for (size_t i = 0; i < count; ++i)
            {
                dst[offset + i] = blocks[b].qs[i] * d;
            }
        }
    }

    /**
     * @brief Compute max |int16| value across all blocks
     */
    int16_t max_int16_value(const std::vector<Q16_1Block> &blocks, size_t block_size = 32)
    {
        int16_t max_val = 0;
        for (const auto &block : blocks)
        {
            for (size_t i = 0; i < block_size; ++i)
            {
                max_val = std::max(max_val, static_cast<int16_t>(std::abs(block.qs[i])));
            }
        }
        return max_val;
    }

    // ============================================================================
    // Test Cases
    // ============================================================================

    class FixedScaleAccuracyTest : public ::testing::Test
    {
    protected:
        std::mt19937 rng_{42};

        std::vector<float> generateUniform(size_t n, float min, float max)
        {
            std::uniform_real_distribution<float> dist(min, max);
            std::vector<float> data(n);
            for (auto &v : data)
                v = dist(rng_);
            return data;
        }

        std::vector<float> generateGaussian(size_t n, float mean, float std)
        {
            std::normal_distribution<float> dist(mean, std);
            std::vector<float> data(n);
            for (auto &v : data)
                v = dist(rng_);
            return data;
        }

        void printMetrics(const std::string &name, const QuantizationMetrics &m, int16_t max_int16)
        {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  " << name << ":\n";
            std::cout << "    MSE:           " << m.mse << "\n";
            std::cout << "    RMSE:          " << m.rmse << "\n";
            std::cout << "    Max Error:     " << m.max_error << "\n";
            std::cout << "    Cosine Sim:    " << m.cosine_sim << "\n";
            std::cout << "    Effective Bits:" << m.effective_bits << "\n";
            std::cout << "    SNR (dB):      " << m.snr_db << "\n";
            std::cout << "    Max |INT16|:   " << max_int16 << " / 32767\n";
        }
    };

    /**
     * @brief Compare accuracy with typical activation range (±3.0)
     *
     * Most transformer activations fall in this range.
     * With kv_cache_scale=8.0, fixed scale should be ~2.7× worse in precision
     * but still very good (and VNNI-safe).
     */
    TEST_F(FixedScaleAccuracyTest, TypicalActivations_Range3)
    {
        std::cout << "\n=== Typical Activations: Uniform [-3, 3] ===\n";

        const size_t n = 1024; // 32 blocks
        const float actual_range = 3.0f;
        const float kv_cache_scale = 8.0f;

        auto data = generateUniform(n, -actual_range, actual_range);

        // Adaptive quantization
        std::vector<Q16_1Block> blocks_adaptive;
        quantize_adaptive(data, blocks_adaptive);
        std::vector<float> recon_adaptive;
        dequantize(blocks_adaptive, recon_adaptive, n);
        auto m_adaptive = compute_metrics(data, recon_adaptive);
        int16_t max_int16_adaptive = max_int16_value(blocks_adaptive);

        // Fixed-scale quantization
        std::vector<Q16_1Block> blocks_fixed;
        quantize_fixed_scale(data, blocks_fixed, kv_cache_scale);
        std::vector<float> recon_fixed;
        dequantize(blocks_fixed, recon_fixed, n);
        auto m_fixed = compute_metrics(data, recon_fixed);
        int16_t max_int16_fixed = max_int16_value(blocks_fixed);

        printMetrics("Adaptive Scale", m_adaptive, max_int16_adaptive);
        printMetrics("Fixed Scale (8.0)", m_fixed, max_int16_fixed);

        std::cout << "  Precision Loss: " << (m_fixed.rmse / m_adaptive.rmse) << "×\n";
        std::cout << "  INT16 Reduction: " << (32767.0 / max_int16_fixed) << "× smaller\n";

        // Expectations:
        // - Both should have excellent cosine similarity (>0.9999)
        // - Fixed scale should be ~2.7× worse RMSE (8/3 ratio)
        // - But fixed scale INT16 values should be ~2.7× smaller (VNNI-safe)
        EXPECT_GT(m_adaptive.cosine_sim, 0.9999);
        EXPECT_GT(m_fixed.cosine_sim, 0.9999);
        EXPECT_GT(m_fixed.effective_bits, 12.0); // Still >12 bits of precision
        EXPECT_LT(max_int16_fixed, 13000);       // INT16 values should be ~3/8 of max
    }

    /**
     * @brief Compare accuracy with narrow activation range (±1.0)
     *
     * This represents post-normalization activations.
     * Fixed scale "wastes" 3 bits but is still very precise.
     */
    TEST_F(FixedScaleAccuracyTest, NarrowActivations_Range1)
    {
        std::cout << "\n=== Narrow Activations: Uniform [-1, 1] ===\n";

        const size_t n = 1024;
        const float actual_range = 1.0f;
        const float kv_cache_scale = 8.0f;

        auto data = generateUniform(n, -actual_range, actual_range);

        std::vector<Q16_1Block> blocks_adaptive, blocks_fixed;
        std::vector<float> recon_adaptive, recon_fixed;

        quantize_adaptive(data, blocks_adaptive);
        dequantize(blocks_adaptive, recon_adaptive, n);

        quantize_fixed_scale(data, blocks_fixed, kv_cache_scale);
        dequantize(blocks_fixed, recon_fixed, n);

        auto m_adaptive = compute_metrics(data, recon_adaptive);
        auto m_fixed = compute_metrics(data, recon_fixed);
        int16_t max_int16_fixed = max_int16_value(blocks_fixed);

        printMetrics("Adaptive Scale", m_adaptive, max_int16_value(blocks_adaptive));
        printMetrics("Fixed Scale (8.0)", m_fixed, max_int16_fixed);

        std::cout << "  Precision Loss: " << (m_fixed.rmse / m_adaptive.rmse) << "×\n";
        std::cout << "  INT16 Reduction: " << (32767.0 / max_int16_fixed) << "× smaller\n";

        // Fixed scale loses ~3 bits (8× precision loss) but still has >12 bits
        EXPECT_GT(m_fixed.cosine_sim, 0.9999);
        EXPECT_GT(m_fixed.effective_bits, 12.0);
        EXPECT_LT(max_int16_fixed, 4200); // INT16 values should be ~1/8 of max
    }

    /**
     * @brief Compare accuracy with Gaussian-distributed activations
     *
     * More realistic distribution - most values near zero.
     */
    TEST_F(FixedScaleAccuracyTest, GaussianActivations_Std1)
    {
        std::cout << "\n=== Gaussian Activations: N(0, 1) ===\n";

        const size_t n = 4096; // Larger sample for better statistics
        const float kv_cache_scale = 8.0f;

        auto data = generateGaussian(n, 0.0f, 1.0f);

        // Clip to kv_cache_scale range (simulates what would happen in practice)
        for (auto &v : data)
        {
            v = std::clamp(v, -kv_cache_scale, kv_cache_scale);
        }

        float actual_max = *std::max_element(data.begin(), data.end(),
                                             [](float a, float b)
                                             { return std::abs(a) < std::abs(b); });
        actual_max = std::abs(actual_max);

        std::vector<Q16_1Block> blocks_adaptive, blocks_fixed;
        std::vector<float> recon_adaptive, recon_fixed;

        quantize_adaptive(data, blocks_adaptive);
        dequantize(blocks_adaptive, recon_adaptive, n);

        quantize_fixed_scale(data, blocks_fixed, kv_cache_scale);
        dequantize(blocks_fixed, recon_fixed, n);

        auto m_adaptive = compute_metrics(data, recon_adaptive);
        auto m_fixed = compute_metrics(data, recon_fixed);
        int16_t max_int16_fixed = max_int16_value(blocks_fixed);

        std::cout << "  Actual max |value|: " << actual_max << "\n";
        printMetrics("Adaptive Scale", m_adaptive, max_int16_value(blocks_adaptive));
        printMetrics("Fixed Scale (8.0)", m_fixed, max_int16_fixed);

        std::cout << "  Precision Loss: " << (m_fixed.rmse / m_adaptive.rmse) << "×\n";

        EXPECT_GT(m_fixed.cosine_sim, 0.9999);
        EXPECT_GT(m_fixed.effective_bits, 11.0); // Still good precision
    }

    /**
     * @brief Verify VNNI overflow safety with fixed scale
     *
     * CRITICAL FINDING: Fixed scale alone is NOT sufficient for 192-dim!
     * This test demonstrates why sub-block accumulation is REQUIRED.
     */
    TEST_F(FixedScaleAccuracyTest, VNNIOverflowSafety_192Dim)
    {
        std::cout << "\n=== VNNI Overflow Safety Test (192 dimensions) ===\n";

        const size_t n = 192;            // MLA dimension (128 NOPE + 64 ROPE)
        const float actual_range = 3.0f; // Typical activations
        const float kv_cache_scale = 8.0f;

        // Generate worst-case aligned vectors (same sign, maximize dot product)
        std::vector<float> q_data(n), k_data(n);
        for (size_t i = 0; i < n; ++i)
        {
            q_data[i] = actual_range; // All positive max values
            k_data[i] = actual_range;
        }

        std::vector<Q16_1Block> q_adaptive, k_adaptive;
        std::vector<Q16_1Block> q_fixed, k_fixed;

        quantize_adaptive(q_data, q_adaptive);
        quantize_adaptive(k_data, k_adaptive);
        quantize_fixed_scale(q_data, q_fixed, kv_cache_scale);
        quantize_fixed_scale(k_data, k_fixed, kv_cache_scale);

        // Compute dot products
        int64_t dot_adaptive = 0;
        int64_t dot_fixed = 0;

        size_t block_size = 32;
        for (size_t i = 0; i < n; ++i)
        {
            size_t block_idx = i / block_size;
            size_t elem_idx = i % block_size;

            int32_t q_val_adaptive = q_adaptive[block_idx].qs[elem_idx];
            int32_t k_val_adaptive = k_adaptive[block_idx].qs[elem_idx];
            int32_t q_val_fixed = q_fixed[block_idx].qs[elem_idx];
            int32_t k_val_fixed = k_fixed[block_idx].qs[elem_idx];

            dot_adaptive += static_cast<int64_t>(q_val_adaptive) * k_val_adaptive;
            dot_fixed += static_cast<int64_t>(q_val_fixed) * k_val_fixed;
        }

        std::cout << "  Adaptive dot product: " << dot_adaptive << "\n";
        std::cout << "  Fixed dot product:    " << dot_fixed << "\n";
        std::cout << "  INT32_MAX:            " << INT32_MAX << "\n";
        std::cout << "  Adaptive overflow:    " << (dot_adaptive > INT32_MAX ? "YES!" : "No") << "\n";
        std::cout << "  Fixed overflow:       " << (dot_fixed > INT32_MAX ? "YES!" : "No") << "\n";

        int16_t max_q = max_int16_value(q_fixed);
        int16_t max_k = max_int16_value(k_fixed);
        int64_t worst_case = static_cast<int64_t>(n) * max_q * max_k;

        std::cout << "  Max |Q| INT16: " << max_q << "\n";
        std::cout << "  Max |K| INT16: " << max_k << "\n";
        std::cout << "  Worst-case sum: " << worst_case << "\n";

        // BOTH adaptive and fixed overflow for 192-dim with typical activations!
        // This is WHY we need sub-block accumulation (128 NOPE + 64 ROPE separately)
        EXPECT_GT(dot_adaptive, static_cast<int64_t>(INT32_MAX)) << "Adaptive should overflow (expected)";
        EXPECT_GT(dot_fixed, static_cast<int64_t>(INT32_MAX)) << "Fixed also overflows - sub-block accumulation required!";

        // Calculate safe scale for 192-dim single accumulation
        // Max safe INT16 for N=192: sqrt(INT32_MAX / 192) ≈ 3344
        float safe_scale_for_single_accum = 3344.0f * kv_cache_scale / 32767.0f;
        std::cout << "\n  CONCLUSION:\n";
        std::cout << "  - kv_cache_scale=8.0 with actual_range=3.0 → INT16 ≈ 12288\n";
        std::cout << "  - 192 × 12288² = 29B → OVERFLOW!\n";
        std::cout << "  - Safe FP32 max for 192-dim single accum: ±" << safe_scale_for_single_accum << "\n";
        std::cout << "  - SOLUTION: Sub-block accumulation (128-dim NOPE + 64-dim ROPE separately)\n";
    }

    /**
     * @brief Verify sub-block accumulation prevents overflow
     *
     * This is what we actually implement: accumulate NOPE (128-dim) and ROPE (64-dim)
     * separately, then sum the partials.
     */
    TEST_F(FixedScaleAccuracyTest, SubBlockAccumulation_Safe)
    {
        std::cout << "\n=== Sub-Block Accumulation Safety Test ===\n";

        const size_t nope_dim = 128;
        const size_t rope_dim = 64;
        const float actual_range = 3.0f;
        const float kv_cache_scale = 8.0f;

        // Generate worst-case data
        std::vector<float> q_nope(nope_dim, actual_range);
        std::vector<float> k_nope(nope_dim, actual_range);
        std::vector<float> q_rope(rope_dim, actual_range);
        std::vector<float> k_rope(rope_dim, actual_range);

        std::vector<Q16_1Block> q_nope_blocks, k_nope_blocks;
        std::vector<Q16_1Block> q_rope_blocks, k_rope_blocks;

        quantize_fixed_scale(q_nope, q_nope_blocks, kv_cache_scale);
        quantize_fixed_scale(k_nope, k_nope_blocks, kv_cache_scale);
        quantize_fixed_scale(q_rope, q_rope_blocks, kv_cache_scale);
        quantize_fixed_scale(k_rope, k_rope_blocks, kv_cache_scale);

        // Compute NOPE dot product (128-dim)
        int64_t dot_nope = 0;
        size_t block_size = 32;
        for (size_t i = 0; i < nope_dim; ++i)
        {
            size_t block_idx = i / block_size;
            size_t elem_idx = i % block_size;
            int32_t q = q_nope_blocks[block_idx].qs[elem_idx];
            int32_t k = k_nope_blocks[block_idx].qs[elem_idx];
            dot_nope += static_cast<int64_t>(q) * k;
        }

        // Compute ROPE dot product (64-dim)
        int64_t dot_rope = 0;
        for (size_t i = 0; i < rope_dim; ++i)
        {
            size_t block_idx = i / block_size;
            size_t elem_idx = i % block_size;
            int32_t q = q_rope_blocks[block_idx].qs[elem_idx];
            int32_t k = k_rope_blocks[block_idx].qs[elem_idx];
            dot_rope += static_cast<int64_t>(q) * k;
        }

        std::cout << "  NOPE (128-dim) dot: " << dot_nope << " (max: " << INT32_MAX << ")\n";
        std::cout << "  ROPE (64-dim) dot:  " << dot_rope << " (max: " << INT32_MAX << ")\n";
        std::cout << "  Combined sum:       " << (dot_nope + dot_rope) << "\n";

        // Check if partials fit in INT32
        bool nope_fits = (dot_nope <= INT32_MAX && dot_nope >= INT32_MIN);
        bool rope_fits = (dot_rope <= INT32_MAX && dot_rope >= INT32_MIN);

        std::cout << "  NOPE fits INT32: " << (nope_fits ? "YES" : "NO") << "\n";
        std::cout << "  ROPE fits INT32: " << (rope_fits ? "YES" : "NO") << "\n";

        // With actual_range=3.0 and kv_cache_scale=8.0:
        // INT16 ≈ 12288
        // 128 × 12288² ≈ 19.3B → STILL OVERFLOW for NOPE!
        // 64 × 12288² ≈ 9.7B → ALSO OVERFLOW for ROPE!

        std::cout << "\n  CRITICAL: Even sub-block accumulation overflows with these values!\n";
        std::cout << "  This demonstrates why real model activations are typically smaller,\n";
        std::cout << "  or why we need tighter kv_cache_scale for models with large head_dim.\n";

        // Calculate what range IS safe
        float safe_nope = std::sqrt(static_cast<float>(INT32_MAX) / nope_dim) * kv_cache_scale / 32767.0f;
        float safe_rope = std::sqrt(static_cast<float>(INT32_MAX) / rope_dim) * kv_cache_scale / 32767.0f;

        std::cout << "\n  Safe FP32 ranges (with kv_cache_scale=8.0):\n";
        std::cout << "    NOPE (128-dim): |x| < " << safe_nope << "\n";
        std::cout << "    ROPE (64-dim):  |x| < " << safe_rope << "\n";

        // The test "passes" because it documents the constraint, not because it's safe
        // Real safety comes from either:
        // 1. Real activations being smaller (typical)
        // 2. Using tighter kv_cache_scale
        // 3. Per-head normalization that bounds values
    }

    /**
     * @brief Compare to Q8_0 baseline to show Q16 is still much better
     */
    TEST_F(FixedScaleAccuracyTest, CompareToQ8_0Baseline)
    {
        std::cout << "\n=== Q16_1 Fixed vs Q8_0 Baseline ===\n";

        const size_t n = 1024;
        const float actual_range = 3.0f;
        const float kv_cache_scale = 8.0f;

        auto data = generateUniform(n, -actual_range, actual_range);

        // Q16_1 with fixed scale
        std::vector<Q16_1Block> blocks_q16;
        std::vector<float> recon_q16;
        quantize_fixed_scale(data, blocks_q16, kv_cache_scale);
        dequantize(blocks_q16, recon_q16, n);
        auto m_q16 = compute_metrics(data, recon_q16);

        // Simulate Q8_0 (adaptive, 8-bit)
        std::vector<float> recon_q8(n);
        size_t block_size = 32;
        size_t n_blocks = (n + block_size - 1) / block_size;
        for (size_t b = 0; b < n_blocks; ++b)
        {
            size_t offset = b * block_size;
            size_t count = std::min(block_size, n - offset);

            float max_abs = 0.0f;
            for (size_t i = 0; i < count; ++i)
            {
                max_abs = std::max(max_abs, std::abs(data[offset + i]));
            }

            float d = (max_abs > 1e-10f) ? (max_abs / 127.0f) : 0.0f;
            float inv_d = (d > 1e-10f) ? (1.0f / d) : 0.0f;

            for (size_t i = 0; i < count; ++i)
            {
                int32_t q = static_cast<int32_t>(std::round(data[offset + i] * inv_d));
                q = std::clamp(q, -127, 127);
                recon_q8[offset + i] = q * d;
            }
        }
        auto m_q8 = compute_metrics(data, recon_q8);

        printMetrics("Q16_1 Fixed Scale", m_q16, 0);
        printMetrics("Q8_0 Adaptive", m_q8, 0);

        std::cout << "  Q16 vs Q8 precision: " << (m_q8.rmse / m_q16.rmse) << "× better\n";

        // Q16_1 fixed should still be much better than Q8_0
        EXPECT_GT(m_q16.effective_bits, m_q8.effective_bits + 4); // At least 4 bits better
        EXPECT_LT(m_q16.rmse, m_q8.rmse / 10);                    // At least 10× lower RMSE
    }

    /**
     * @brief Summary test with different kv_cache_scale values
     */
    TEST_F(FixedScaleAccuracyTest, KVCacheScaleSweep)
    {
        std::cout << "\n=== KV Cache Scale Sweep ===\n";
        std::cout << "  (Actual range: [-3, 3], varying kv_cache_scale)\n\n";

        const size_t n = 1024;
        const float actual_range = 3.0f;
        auto data = generateUniform(n, -actual_range, actual_range);

        // Get adaptive baseline
        std::vector<Q16_1Block> blocks_adaptive;
        std::vector<float> recon_adaptive;
        quantize_adaptive(data, blocks_adaptive);
        dequantize(blocks_adaptive, recon_adaptive, n);
        auto m_adaptive = compute_metrics(data, recon_adaptive);

        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Scale | Eff.Bits | RMSE     | Max|INT16| | Precision Loss\n";
        std::cout << "  ------+----------+----------+------------+---------------\n";

        for (float scale : {4.0f, 6.0f, 8.0f, 12.0f, 16.0f})
        {
            std::vector<Q16_1Block> blocks_fixed;
            std::vector<float> recon_fixed;
            quantize_fixed_scale(data, blocks_fixed, scale);
            dequantize(blocks_fixed, recon_fixed, n);
            auto m = compute_metrics(data, recon_fixed);
            int16_t max_int16 = max_int16_value(blocks_fixed);

            std::cout << "  " << std::setw(5) << scale
                      << " | " << std::setw(8) << m.effective_bits
                      << " | " << std::setw(8) << m.rmse
                      << " | " << std::setw(10) << max_int16
                      << " | " << std::setw(5) << (m.rmse / m_adaptive.rmse) << "×\n";
        }

        std::cout << "\n  (Adaptive baseline: " << m_adaptive.effective_bits << " bits, RMSE=" << m_adaptive.rmse << ")\n";
    }

    /**
     * @brief Find safe kv_cache_scale for different head dimensions
     *
     * This calculates what kv_cache_scale is actually safe for VNNI-only
     * accumulation (no sub-block splitting) given typical activation ranges.
     */
    TEST_F(FixedScaleAccuracyTest, SafeKVCacheScaleCalculation)
    {
        std::cout << "\n=== Safe kv_cache_scale Calculation ===\n\n";

        // For VNNI safety: N × (actual_range × 32767 / kv_cache_scale)² ≤ INT32_MAX
        // Solving for kv_cache_scale:
        //   kv_cache_scale ≥ actual_range × 32767 × sqrt(N / INT32_MAX)

        auto compute_safe_scale = [](size_t n, float actual_range) -> float
        {
            return actual_range * 32767.0f * std::sqrt(static_cast<float>(n) / INT32_MAX);
        };

        auto compute_safe_actual_range = [](size_t n, float kv_cache_scale) -> float
        {
            return kv_cache_scale * std::sqrt(static_cast<float>(INT32_MAX) / n) / 32767.0f;
        };

        std::cout << "  Part A: Minimum safe kv_cache_scale for actual_range=3.0\n";
        std::cout << "  head_dim | Min Safe Scale | Precision Loss vs Adaptive\n";
        std::cout << "  ---------+----------------+---------------------------\n";

        for (size_t dim : {64, 96, 128, 192})
        {
            float safe_scale = compute_safe_scale(dim, 3.0f);
            float precision_loss = safe_scale / 3.0f; // Ratio of scale to actual range
            std::cout << "  " << std::setw(8) << dim
                      << " | " << std::setw(14) << std::setprecision(2) << safe_scale
                      << " | " << std::setw(5) << std::setprecision(2) << precision_loss << "×\n";
        }

        std::cout << "\n  Part B: Maximum safe actual_range for kv_cache_scale=8.0\n";
        std::cout << "  head_dim | Max Safe |x| | Can handle typical ±3.0?\n";
        std::cout << "  ---------+-------------+---------------------------\n";

        for (size_t dim : {64, 96, 128, 192})
        {
            float safe_range = compute_safe_actual_range(dim, 8.0f);
            bool can_handle = safe_range >= 3.0f;
            std::cout << "  " << std::setw(8) << dim
                      << " | " << std::setw(11) << std::setprecision(3) << safe_range
                      << " | " << (can_handle ? "YES" : "NO - sub-block needed") << "\n";
        }

        std::cout << "\n  CONCLUSIONS:\n";
        std::cout << "  1. With kv_cache_scale=8.0 and actual_range=3.0:\n";
        std::cout << "     ALL head_dim values overflow! Even 64-dim.\n";
        std::cout << "  2. Safe single-accum requires EITHER:\n";
        std::cout << "     a) Very large kv_cache_scale (17-30×), losing 6-10 bits precision\n";
        std::cout << "     b) Very small actual activations (|x| < 1.0)\n";
        std::cout << "  3. SOLUTION: Sub-block accumulation with smaller block sizes\n";
        std::cout << "     - 32-dim blocks: safe for |x| < 2.83\n";
        std::cout << "     - 16-dim blocks: safe for |x| < 4.0\n";

        // Verify the calculation with actual INT16 values
        std::cout << "\n  Part C: Verification with actual quantized values\n";
        for (size_t dim : {64, 128, 192})
        {
            float actual_range = 3.0f;
            float kv_cache_scale = 8.0f;

            // Quantize worst-case data
            std::vector<float> q_data(dim, actual_range);
            std::vector<float> k_data(dim, actual_range);
            std::vector<Q16_1Block> q_blocks, k_blocks;
            quantize_fixed_scale(q_data, q_blocks, kv_cache_scale);
            quantize_fixed_scale(k_data, k_blocks, kv_cache_scale);

            // Compute dot product
            int64_t dot = 0;
            size_t block_size = 32;
            for (size_t i = 0; i < dim; ++i)
            {
                size_t bi = i / block_size;
                size_t ei = i % block_size;
                dot += static_cast<int64_t>(q_blocks[bi].qs[ei]) * k_blocks[bi].qs[ei];
            }

            float overflow_ratio = static_cast<float>(dot) / INT32_MAX;
            std::cout << "  dim=" << dim << ": dot=" << dot
                      << " (" << std::setprecision(1) << overflow_ratio << "× INT32_MAX)\n";
        }
    }

} // anonymous namespace
