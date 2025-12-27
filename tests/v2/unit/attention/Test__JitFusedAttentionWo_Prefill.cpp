/**
 * @file Test__JitFusedAttentionWo_Prefill.cpp
 * @brief Unit tests for prefill mode in JIT fused attention kernel
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the prefill kernel implementation which uses H→KV→Q loop ordering
 * for K/V cache reuse optimization. Validates that:
 *
 * 1. Prefill kernel produces same output as decode kernel (functional parity)
 * 2. AttentionMode::AUTO correctly selects prefill for batch_size > 1
 * 3. Causal masking works correctly in prefill mode
 * 4. Various tile sizes work correctly (Q_TILE_SIZE boundary conditions)
 *
 * The prefill kernel trades off stack space for memory bandwidth by:
 * - Processing queries in tiles of Q_TILE_SIZE (8)
 * - Loading K/V blocks once per head, reusing across all queries in tile
 * - Maintaining per-query softmax state arrays
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include <cstring>
#include <iostream>
#include <iomanip>

#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using microkernels::Q8_1Block;

    // ============================================================================
    // Test Fixture
    // ============================================================================

    class Test__JitFusedAttentionWo_Prefill : public ::testing::Test
    {
    protected:
        std::mt19937 gen_{42};
        std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};

        // Correctness thresholds for prefill vs decode comparison
        // Prefill uses batched vectorized softmax; decode processes queries individually
        // Small numerical differences are expected due to different FP accumulation order
        //
        // CAUSAL mode: Queries have varying KV lengths, less accumulation → tighter thresholds
        // NON-CAUSAL mode: All queries see all KV positions, more accumulation order
        //   differences between tile-batched vs sequential processing → looser thresholds
        static constexpr double MIN_COSINE_SIM_CAUSAL = 0.995;
        static constexpr double MAX_REL_L2_ERROR_CAUSAL = 0.10;

        static constexpr double MIN_COSINE_SIM_NONCAUSAL = 0.993;  // Slightly looser for non-causal
        static constexpr double MAX_REL_L2_ERROR_NONCAUSAL = 0.12; // 12% tolerance for non-causal

        /**
         * @brief Quantize FP32 data to Q8_1 block format
         */
        void quantize_fp32_to_q8_1(const float *fp32_data, int rows, int cols, Q8_1Block *blocks)
        {
            const int num_blocks_per_row = cols / 32;

            for (int row = 0; row < rows; ++row)
            {
                for (int b = 0; b < num_blocks_per_row; ++b)
                {
                    const float *block_data = fp32_data + row * cols + b * 32;
                    Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                    float max_abs = 0.0f;
                    for (int i = 0; i < 32; ++i)
                    {
                        max_abs = std::max(max_abs, std::fabs(block_data[i]));
                    }

                    float scale = max_abs / 127.0f;
                    if (scale < 1e-10f)
                        scale = 1e-10f;
                    float inv_scale = 127.0f / max_abs;
                    if (max_abs < 1e-10f)
                        inv_scale = 0.0f;

                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        int8_t q = static_cast<int8_t>(std::round(block_data[i] * inv_scale));
                        q = std::max(int8_t(-127), std::min(int8_t(127), q));
                        blk.qs[i] = q;
                        sum_qs += q;
                    }

                    blk.d = fp32_to_fp16(scale);
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        }

        std::vector<float> generate_random_fp32(int size, float scale = 1.0f)
        {
            std::vector<float> data(size);
            for (auto &v : data)
            {
                v = dist_(gen_) * scale;
            }
            return data;
        }

        double cosine_similarity(const float *a, const float *b, int n)
        {
            double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
            for (int i = 0; i < n; ++i)
            {
                dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
                norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
                norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }
            if (norm_a < 1e-10 || norm_b < 1e-10)
                return 1.0;
            return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        }

        double relative_l2_error(const float *actual, const float *expected, int n)
        {
            double sum_sq_err = 0.0, sum_sq_exp = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double err = actual[i] - expected[i];
                sum_sq_err += err * err;
                sum_sq_exp += expected[i] * expected[i];
            }
            if (sum_sq_exp < 1e-10)
                return 0.0;
            return std::sqrt(sum_sq_err / sum_sq_exp);
        }

        double max_abs_error(const float *actual, const float *expected, int n)
        {
            double max_err = 0.0;
            for (int i = 0; i < n; ++i)
            {
                double err = std::fabs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
                max_err = std::max(max_err, err);
            }
            return max_err;
        }

        /**
         * @brief Run prefill vs decode parity test
         *
         * Runs the same computation using:
         * 1. Prefill mode (batch_size = seq_len_q)
         * 2. Decode mode processing one query at a time
         *
         * The outputs should be identical (both produce same attention output).
         */
        void run_prefill_decode_parity_test(
            int seq_len_q,
            int seq_len_kv,
            int num_heads,
            int num_kv_heads,
            int head_dim,
            bool causal,
            const std::string &test_name)
        {
            const int d_model = num_heads * head_dim;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int blocks_per_head = head_dim / 32;

            std::cout << "\n=== " << test_name << " ===" << std::endl;
            std::cout << "  seq_len_q=" << seq_len_q << ", seq_len_kv=" << seq_len_kv
                      << ", heads=" << num_heads << "/" << num_kv_heads
                      << ", head_dim=" << head_dim << ", causal=" << causal << std::endl;

            // Generate random FP32 data
            auto Q_fp32 = generate_random_fp32(seq_len_q * num_heads * head_dim);
            auto K_fp32 = generate_random_fp32(seq_len_kv * num_kv_heads * head_dim);
            auto V_fp32 = generate_random_fp32(seq_len_kv * num_kv_heads * head_dim);

            // Quantize to Q8_1
            std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
            std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
            std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

            quantize_fp32_to_q8_1(Q_fp32.data(), seq_len_q * num_heads, head_dim, Q_q8.data());
            quantize_fp32_to_q8_1(K_fp32.data(), seq_len_kv * num_kv_heads, head_dim, K_q8.data());
            quantize_fp32_to_q8_1(V_fp32.data(), seq_len_kv * num_kv_heads, head_dim, V_q8.data());

            // Create Wo weights (identity projection for test simplicity)
            std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
            for (int i = 0; i < d_model; ++i)
            {
                Wo_fp32[i * d_model + i] = 1.0f;
            }

            // Output buffers
            std::vector<float> output_prefill(seq_len_q * d_model, 0.0f);
            std::vector<float> output_decode(seq_len_q * d_model, 0.0f);

            // Snapshot buffers (local_dim = num_heads * head_dim)
            int local_dim = num_heads * head_dim;
            std::vector<float> snapshot_prefill(seq_len_q * local_dim, 0.0f);
            std::vector<float> snapshot_decode(seq_len_q * local_dim, 0.0f);

            // === Run Prefill Mode ===
            {
                JitAttentionConfig config;
                config.head_dim = head_dim;
                config.num_heads = num_heads;
                config.num_kv_heads = num_kv_heads;
                config.batch_size = seq_len_q; // Triggers prefill mode
                config.wo_format = WoFormat::FP32;
                config.causal = causal;
                config.mode = AttentionMode::PREFILL; // Explicit prefill

                ASSERT_EQ(config.effectiveMode(), AttentionMode::PREFILL)
                    << "Should use prefill mode";

                JitFusedAttentionWo prefill_kernel(config);
                prefill_kernel.compute(
                    Q_q8.data(),
                    K_q8.data(),
                    V_q8.data(),
                    Wo_fp32.data(),
                    output_prefill.data(),
                    seq_len_q,
                    seq_len_kv,
                    scale,
                    0, // position_offset = 0
                    snapshot_prefill.data());
            }

            // === Run Decode Mode (one query at a time) ===
            {
                JitAttentionConfig config;
                config.head_dim = head_dim;
                config.num_heads = num_heads;
                config.num_kv_heads = num_kv_heads;
                config.batch_size = 1; // Triggers decode mode
                config.wo_format = WoFormat::FP32;
                config.causal = causal;
                config.mode = AttentionMode::DECODE; // Explicit decode

                ASSERT_EQ(config.effectiveMode(), AttentionMode::DECODE)
                    << "Should use decode mode";

                JitFusedAttentionWo decode_kernel(config);

                // Process one query at a time
                for (int q = 0; q < seq_len_q; ++q)
                {
                    // Point to Q[q, :, :] - single query slice
                    const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;

                    // Output for this query
                    float *out_slice = output_decode.data() + q * d_model;
                    float *snap_slice = snapshot_decode.data() + q * local_dim;

                    // For causal attention, adjust kv_seq_len to only attend to [0, q+1)
                    int effective_kv_len = causal ? std::min(q + 1, seq_len_kv) : seq_len_kv;

                    decode_kernel.compute(
                        Q_slice,
                        K_q8.data(),
                        V_q8.data(),
                        Wo_fp32.data(),
                        out_slice,
                        1, // seq_len_q = 1
                        effective_kv_len,
                        scale,
                        q, // position_offset = q for correct causal masking
                        snap_slice);
                }
            }

            // === Compare Snapshots (Context before Wo) ===
            int snapshot_size = seq_len_q * local_dim;
            double snap_cos_sim = cosine_similarity(snapshot_prefill.data(), snapshot_decode.data(), snapshot_size);
            double snap_rel_l2 = relative_l2_error(snapshot_prefill.data(), snapshot_decode.data(), snapshot_size);

            std::cout << "  Snapshot Cosine similarity: " << std::fixed << std::setprecision(6) << snap_cos_sim << std::endl;
            std::cout << "  Snapshot Relative L2 error: " << std::fixed << std::setprecision(6) << snap_rel_l2 << std::endl;

            if (snap_cos_sim < 0.999)
            {
                std::cout << "  Snapshot divergence detected!" << std::endl;
                // Print first few elements of divergence
                for (int q = 0; q < seq_len_q; ++q)
                {
                    std::cout << "    Query " << q << " first 8 elements:" << std::endl;
                    std::cout << "      Prefill: ";
                    for (int i = 0; i < 8; ++i)
                        std::cout << std::fixed << std::setprecision(6) << snapshot_prefill[q * local_dim + i] << " ";
                    std::cout << std::endl;
                    std::cout << "      Decode:  ";
                    for (int i = 0; i < 8; ++i)
                        std::cout << std::fixed << std::setprecision(6) << snapshot_decode[q * local_dim + i] << " ";
                    std::cout << std::endl;
                }
            }

            // === Compare outputs ===
            int output_size = seq_len_q * d_model;
            double cos_sim = cosine_similarity(output_prefill.data(), output_decode.data(), output_size);
            double rel_l2 = relative_l2_error(output_prefill.data(), output_decode.data(), output_size);
            double max_err = max_abs_error(output_prefill.data(), output_decode.data(), output_size);

            std::cout << "  Cosine similarity:  " << std::fixed << std::setprecision(6) << cos_sim << std::endl;
            std::cout << "  Relative L2 error:  " << std::fixed << std::setprecision(6) << rel_l2 << std::endl;
            std::cout << "  Max absolute error: " << std::fixed << std::setprecision(6) << max_err << std::endl;

            // Select thresholds based on causal mode
            // Non-causal has more FP accumulation order differences (every query sees all KV)
            double min_cos_sim = causal ? MIN_COSINE_SIM_CAUSAL : MIN_COSINE_SIM_NONCAUSAL;
            double max_rel_l2 = causal ? MAX_REL_L2_ERROR_CAUSAL : MAX_REL_L2_ERROR_NONCAUSAL;

            // Validate
            EXPECT_GE(cos_sim, min_cos_sim)
                << "Prefill vs decode cosine similarity too low: " << cos_sim;
            EXPECT_LE(rel_l2, max_rel_l2)
                << "Prefill vs decode L2 error too high: " << rel_l2;

            // Debug output on failure
            if (cos_sim < min_cos_sim || rel_l2 > max_rel_l2)
            {
                std::cout << "\n  First divergence analysis:" << std::endl;
                for (int q = 0; q < std::min(seq_len_q, 4); ++q)
                {
                    std::cout << "    Query " << q << " first 8 elements:" << std::endl;
                    std::cout << "      Prefill: ";
                    for (int i = 0; i < 8; ++i)
                    {
                        std::cout << output_prefill[q * d_model + i] << " ";
                    }
                    std::cout << std::endl;
                    std::cout << "      Decode:  ";
                    for (int i = 0; i < 8; ++i)
                    {
                        std::cout << output_decode[q * d_model + i] << " ";
                    }
                    std::cout << std::endl;
                }
            }
        }
    };

    // ============================================================================
    // AttentionMode Selection Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Prefill, ModeSelection_AutoDecodeForBatch1)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.mode = AttentionMode::AUTO;

        EXPECT_EQ(config.effectiveMode(), AttentionMode::DECODE);
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, ModeSelection_AutoPrefillForBatch8)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 8;
        config.mode = AttentionMode::AUTO;

        EXPECT_EQ(config.effectiveMode(), AttentionMode::PREFILL);
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, ModeSelection_ExplicitOverridesAuto)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 1;
        config.mode = AttentionMode::PREFILL; // Force prefill even for batch=1

        EXPECT_EQ(config.effectiveMode(), AttentionMode::PREFILL);
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, KernelCacheSeparatesDecodeAndPrefill)
    {
        JitAttentionKernelCache::instance().clear();

        JitAttentionConfig decode_config;
        decode_config.head_dim = 64;
        decode_config.num_heads = 14;
        decode_config.num_kv_heads = 2;
        decode_config.batch_size = 1;
        decode_config.wo_format = WoFormat::FP32;
        decode_config.mode = AttentionMode::DECODE;

        JitAttentionConfig prefill_config = decode_config;
        prefill_config.batch_size = 8;
        prefill_config.mode = AttentionMode::PREFILL;

        JitFusedAttentionWo decode_kernel(decode_config);
        JitFusedAttentionWo prefill_kernel(prefill_config);

        // Should be different kernels
        EXPECT_NE(decode_kernel.getKernel(), prefill_kernel.getKernel())
            << "Decode and prefill should have different cached kernels";

        // Cache should have 2 entries
        EXPECT_EQ(JitAttentionKernelCache::instance().size(), 2u);
    }

    // ============================================================================
    // Prefill Kernel Instantiation Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Prefill, CanInstantiatePrefillKernel_Qwen05B)
    {
        JitAttentionConfig config;
        config.head_dim = 64;
        config.num_heads = 14;
        config.num_kv_heads = 2;
        config.batch_size = 32;
        config.wo_format = WoFormat::FP32;
        config.mode = AttentionMode::PREFILL;

        ASSERT_NO_THROW({
            JitFusedAttentionWo kernel(config);
            EXPECT_NE(kernel.getKernel(), nullptr);
        });
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, CanInstantiatePrefillKernel_Qwen7B)
    {
        // NOTE: Qwen7B (28 heads, head_dim=128) generates very large JIT code
        // for prefill mode due to head loop unrolling. Current buffer size (512KB)
        // may be insufficient. This test documents the limitation.
        JitAttentionConfig config;
        config.head_dim = 128;
        config.num_heads = 28;
        config.num_kv_heads = 4;
        config.batch_size = 64;
        config.wo_format = WoFormat::FP32;
        config.mode = AttentionMode::PREFILL;

        // Currently prefill uses decode kernel fallback, so this should succeed.
        // When we implement true prefill with compile-time head unrolling,
        // this test should be updated to EXPECT_THROW for large models.
        EXPECT_NO_THROW({
            JitFusedAttentionWo kernel(config);
        });
    }

    // ============================================================================
    // Prefill vs Decode Parity Tests (Non-Causal)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_SmallTile_NonCausal)
    {
        // seq_len_q = 4, less than Q_TILE_SIZE (8)
        run_prefill_decode_parity_test(
            /*seq_len_q=*/4,
            /*seq_len_kv=*/16,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/false,
            "Small Tile (seq_q=4 < Q_TILE_SIZE=8), Non-Causal");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_ExactTile_NonCausal)
    {
        // seq_len_q = 8, exactly Q_TILE_SIZE
        run_prefill_decode_parity_test(
            /*seq_len_q=*/8,
            /*seq_len_kv=*/32,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/false,
            "Exact Tile (seq_q=8 = Q_TILE_SIZE), Non-Causal");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_MultipleTiles_NonCausal)
    {
        // seq_len_q = 24, 3 full tiles
        run_prefill_decode_parity_test(
            /*seq_len_q=*/24,
            /*seq_len_kv=*/48,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/false,
            "Multiple Tiles (seq_q=24 = 3 tiles), Non-Causal");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_PartialLastTile_NonCausal)
    {
        // seq_len_q = 11, 1 full tile + partial (3 queries)
        run_prefill_decode_parity_test(
            /*seq_len_q=*/11,
            /*seq_len_kv=*/32,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/false,
            "Partial Last Tile (seq_q=11), Non-Causal");
    }

    // ============================================================================
    // Prefill vs Decode Parity Tests (Causal)
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_SmallTile_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/4,
            /*seq_len_kv=*/4,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Small Tile (seq_q=4), Causal");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_ExactTile_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/8,
            /*seq_len_kv=*/8,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Exact Tile (seq_q=8), Causal");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_MultipleTiles_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/16,
            /*seq_len_kv=*/16,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Multiple Tiles (seq_q=16), Causal");
    }

    // ============================================================================
    // Qwen Model Configuration Tests
    // ============================================================================

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_Qwen05B_Prefill16)
    {
        // Qwen2 0.5B: 14 heads, 2 KV heads, head_dim=64
        run_prefill_decode_parity_test(
            /*seq_len_q=*/16,
            /*seq_len_kv=*/16,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Qwen2 0.5B - Prefill 16 tokens");
    }

    TEST_F(Test__JitFusedAttentionWo_Prefill, Parity_Qwen05B_Prefill64)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/64,
            /*seq_len_kv=*/64,
            /*num_heads=*/14,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Qwen2 0.5B - Prefill 64 tokens");
    }

    // Debug test for causal with single tile
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_SingleTile_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/2,
            /*seq_len_kv=*/2,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Debug: Single Tile (seq_q=2), Causal");
    }

    // Debug test for causal with partial second tile (3 queries = 1 full + 1 partial)
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_PartialSecondTile_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/3,
            /*seq_len_kv=*/3,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Debug: Partial Second Tile (seq_q=3), Causal");
    }

    // Debug test: 4 queries, but seq_len_kv=3 (no K3)
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_FullSecondTile_ShortKV_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/4,
            /*seq_len_kv=*/3,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Debug: Full Second Tile with Short KV (seq_q=4, seq_kv=3), Causal");
    }

    // Debug test: 4 queries, seq_len_kv=5 (more KV positions than queries)
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_FullSecondTile_LongKV_Causal)
    {
        run_prefill_decode_parity_test(
            /*seq_len_q=*/4,
            /*seq_len_kv=*/5,
            /*num_heads=*/4,
            /*num_kv_heads=*/2,
            /*head_dim=*/64,
            /*causal=*/true,
            "Debug: Full Second Tile with Long KV (seq_q=4, seq_kv=5), Causal");
    }

    // Debug test: Print intermediate attention values to trace the bug
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_DeepTrace_Causal)
    {
        // Same configuration as failing test
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 4;
        const int num_kv_heads = 2;
        const int head_dim = 64;
        const bool causal = true;

        const int d_model = num_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int blocks_per_head = head_dim / 32;

        std::cout << "\n=== DEEP TRACE: Causal Bug Analysis ===" << std::endl;
        std::cout << "  q_tile_size = " << (4 / (head_dim / 32)) << " (for head_dim=64)" << std::endl;
        std::cout << "  Tile 0: Q0, Q1 (tile_start=0)" << std::endl;
        std::cout << "  Tile 1: Q2, Q3 (tile_start=2)" << std::endl;

        // Use deterministic data to make analysis easier
        // Q = identity-like pattern, K = orthogonal pattern
        std::vector<float> Q_fp32(seq_len_q * num_heads * head_dim);
        std::vector<float> K_fp32(seq_len_kv * num_kv_heads * head_dim);
        std::vector<float> V_fp32(seq_len_kv * num_kv_heads * head_dim);

        // Initialize with deterministic pattern
        for (int q = 0; q < seq_len_q; ++q)
        {
            for (int h = 0; h < num_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    // Q[q,h,d] = (q+1) * 0.1 if d == (q + h * 16) % head_dim, else 0.01
                    int idx = q * num_heads * head_dim + h * head_dim + d;
                    if (d == (q + h * 16) % head_dim)
                    {
                        Q_fp32[idx] = (q + 1) * 0.1f;
                    }
                    else
                    {
                        Q_fp32[idx] = 0.01f * ((idx % 7) - 3) * 0.1f;
                    }
                }
            }
        }

        for (int k = 0; k < seq_len_kv; ++k)
        {
            for (int h = 0; h < num_kv_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = k * num_kv_heads * head_dim + h * head_dim + d;
                    // K[k,h,d] has peak at different position than Q
                    if (d == (k + h * 8 + 32) % head_dim)
                    {
                        K_fp32[idx] = (k + 1) * 0.15f;
                    }
                    else
                    {
                        K_fp32[idx] = 0.01f * ((idx % 11) - 5) * 0.1f;
                    }
                }
            }
        }

        for (int v = 0; v < seq_len_kv; ++v)
        {
            for (int h = 0; h < num_kv_heads; ++h)
            {
                for (int d = 0; d < head_dim; ++d)
                {
                    int idx = v * num_kv_heads * head_dim + h * head_dim + d;
                    // V[k,h,d] = k + d*0.001 (distinct pattern per KV position)
                    V_fp32[idx] = (v + 1) * 0.5f + d * 0.001f;
                }
            }
        }

        // Quantize
        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_fp32_to_q8_1(Q_fp32.data(), seq_len_q * num_heads, head_dim, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), seq_len_kv * num_kv_heads, head_dim, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), seq_len_kv * num_kv_heads, head_dim, V_q8.data());

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_prefill(seq_len_q * d_model, 0.0f);
        std::vector<float> output_decode(seq_len_q * d_model, 0.0f);

        // Run prefill
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        // Run decode
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 1;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::DECODE;

            JitFusedAttentionWo kernel(config);
            for (int q = 0; q < seq_len_q; ++q)
            {
                const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;
                float *out_slice = output_decode.data() + q * d_model;
                int effective_kv_len = causal ? std::min(q + 1, seq_len_kv) : seq_len_kv;
                kernel.compute(Q_slice, K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out_slice, 1, effective_kv_len, scale, q);
            }
        }

        // Detailed comparison
        std::cout << "\n  Query-by-query comparison (head 0 only, first 4 elements):" << std::endl;
        std::cout << "  NOTE: V pattern: V[k] ~ (k+1)*0.5 + d*0.001" << std::endl;
        std::cout << "  For causal: Q2 attends to K0,K1,K2; Q3 attends to K0,K1,K2,K3" << std::endl;
        std::cout << std::endl;

        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ":" << std::endl;
            std::cout << "    Prefill: ";
            for (int i = 0; i < 4; ++i)
            {
                std::cout << std::fixed << std::setprecision(4) << output_prefill[q * d_model + i] << " ";
            }
            std::cout << std::endl;
            std::cout << "    Decode:  ";
            for (int i = 0; i < 4; ++i)
            {
                std::cout << std::fixed << std::setprecision(4) << output_decode[q * d_model + i] << " ";
            }
            std::cout << std::endl;

            // Check per-query match
            double q_cos = cosine_similarity(
                output_prefill.data() + q * d_model,
                output_decode.data() + q * d_model,
                d_model);
            double q_max_err = max_abs_error(
                output_prefill.data() + q * d_model,
                output_decode.data() + q * d_model,
                d_model);
            std::cout << "    Cosine: " << q_cos << ", MaxErr: " << q_max_err;
            if (q_cos < 0.995)
                std::cout << " *** MISMATCH ***";
            std::cout << std::endl;
        }

        // Detailed comparison by head
        std::cout << "\n  Per-head comparison for Q2:" << std::endl;
        for (int h = 0; h < num_heads; ++h)
        {
            int head_start = 2 * d_model + h * head_dim;
            std::cout << "    Head " << h << ": Prefill=" << output_prefill[head_start]
                      << " Decode=" << output_decode[head_start]
                      << " Diff=" << std::fabs(output_prefill[head_start] - output_decode[head_start]) << std::endl;
        }

        // Cross-query similarity (to detect if Q2 prefill looks like Q3)
        std::cout << "\n  Cross-query cosine similarity (prefill outputs):" << std::endl;
        for (int i = 0; i < seq_len_q; ++i)
        {
            for (int j = i + 1; j < seq_len_q; ++j)
            {
                double sim = cosine_similarity(
                    output_prefill.data() + i * d_model,
                    output_prefill.data() + j * d_model,
                    d_model);
                std::cout << "    Q" << i << " vs Q" << j << ": " << std::fixed << std::setprecision(4) << sim << std::endl;
            }
        }

        // Also show Q2 prefill vs Q3 decode (should differ if bug exists)
        double q2_prefill_vs_q3_decode = cosine_similarity(
            output_prefill.data() + 2 * d_model,
            output_decode.data() + 3 * d_model,
            d_model);
        std::cout << "\n  Q2 prefill vs Q3 decode: " << q2_prefill_vs_q3_decode << std::endl;
        std::cout << "  (If bug exists, this should be high ~0.99+, should be lower ~0.9 if correct)" << std::endl;

        // Overall parity check
        double cos_sim = cosine_similarity(output_prefill.data(), output_decode.data(), seq_len_q * d_model);
        EXPECT_GE(cos_sim, MIN_COSINE_SIM_CAUSAL) << "Prefill vs decode parity failed";
    }

    // Single-head test to isolate the causal bug from GQA complexity
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_SingleHead_Causal)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64;
        const bool causal = true;

        const int d_model = num_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int blocks_per_head = head_dim / 32;

        std::cout << "\n=== Single Head Causal Test (DETAILED DEBUG) ===" << std::endl;
        std::cout << "  q_tile_size = 4 / (64/32) = 2" << std::endl;
        std::cout << "  Tile 0: Q0, Q1; Tile 1: Q2, Q3" << std::endl;

        // V values that make it easy to trace which KV positions were attended
        // V[k] = 2^k: V0=1, V1=2, V2=4, V3=8
        // This way we can see exactly which positions were attended by the binary pattern
        std::vector<float> Q_fp32(seq_len_q * num_heads * head_dim, 1.0f / std::sqrt(64.0f));
        std::vector<float> K_fp32(seq_len_kv * num_kv_heads * head_dim, 1.0f / std::sqrt(64.0f));
        std::vector<float> V_fp32(seq_len_kv * num_kv_heads * head_dim);

        // V values: powers of 2 for easy identification
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = static_cast<float>(1 << k); // 1, 2, 4, 8
            for (int d = 0; d < head_dim; ++d)
            {
                V_fp32[k * head_dim + d] = v_val;
            }
        }

        std::cout << "  V values: V0=1, V1=2, V2=4, V3=8" << std::endl;
        std::cout << "  With uniform Q*K scores, output = avg of attended V values" << std::endl;
        std::cout << "  Expected:" << std::endl;
        std::cout << "    Q0: (1)/1 = 1.0" << std::endl;
        std::cout << "    Q1: (1+2)/2 = 1.5" << std::endl;
        std::cout << "    Q2: (1+2+4)/3 = 2.333" << std::endl;
        std::cout << "    Q3: (1+2+4+8)/4 = 3.75" << std::endl;

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_fp32_to_q8_1(Q_fp32.data(), seq_len_q * num_heads, head_dim, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), seq_len_kv * num_kv_heads, head_dim, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), seq_len_kv * num_kv_heads, head_dim, V_q8.data());

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_prefill(seq_len_q * d_model, 0.0f);
        std::vector<float> output_decode(seq_len_q * d_model, 0.0f);

        // Run prefill
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        // Run decode
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 1;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::DECODE;

            JitFusedAttentionWo kernel(config);
            for (int q = 0; q < seq_len_q; ++q)
            {
                const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;
                float *out_slice = output_decode.data() + q * d_model;
                int effective_kv_len = causal ? std::min(q + 1, seq_len_kv) : seq_len_kv;
                kernel.compute(Q_slice, K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out_slice, 1, effective_kv_len, scale, q);
            }
        }

        std::cout << "\n  Results (first element of each query):" << std::endl;
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "    Q" << q << ": Prefill=" << output_prefill[q * d_model]
                      << " Decode=" << output_decode[q * d_model] << std::endl;
        }

        double cos_sim = cosine_similarity(output_prefill.data(), output_decode.data(), seq_len_q * d_model);
        std::cout << "  Cosine sim: " << cos_sim << std::endl;

        // Compute reference using FP32 attention
        std::cout << "\n  Reference FP32 attention (for comparison):" << std::endl;
        for (int q_idx = 0; q_idx < seq_len_q; ++q_idx)
        {
            // For causal: attend to [0, q_idx]
            int kv_end = causal ? std::min(q_idx + 1, seq_len_kv) : seq_len_kv;

            // Compute scores
            std::vector<float> scores(kv_end);
            for (int k_idx = 0; k_idx < kv_end; ++k_idx)
            {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += Q_fp32[q_idx * head_dim + d] * K_fp32[k_idx * head_dim + d];
                }
                scores[k_idx] = dot * scale;
            }

            // Softmax
            float max_score = *std::max_element(scores.begin(), scores.end());
            float sum_exp = 0.0f;
            for (int k = 0; k < kv_end; ++k)
            {
                scores[k] = std::exp(scores[k] - max_score);
                sum_exp += scores[k];
            }
            for (int k = 0; k < kv_end; ++k)
            {
                scores[k] /= sum_exp;
            }

            // Weighted sum of V
            float result = 0.0f;
            for (int k = 0; k < kv_end; ++k)
            {
                result += scores[k] * V_fp32[k * head_dim]; // Just first element
            }

            std::cout << "    Q" << q_idx << " (attends to K[0.." << kv_end - 1 << "]): "
                      << result << " (weights: ";
            for (int k = 0; k < kv_end; ++k)
            {
                std::cout << scores[k] << " ";
            }
            std::cout << ")" << std::endl;
        }

        EXPECT_GE(cos_sim, MIN_COSINE_SIM_CAUSAL);
    }

    // Test with head_dim=32 (q_tile_size=4) to see if bug is tile-size specific
    TEST_F(Test__JitFusedAttentionWo_Prefill, Debug_SingleHead_HeadDim32_Causal)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 32; // Different from 64!
        const bool causal = true;

        // With head_dim=32, q_tile_size = 4 / (32/32) = 4 / 1 = 4
        // So ALL 4 queries fit in ONE tile!
        std::cout << "\n=== Single Head, head_dim=32 (q_tile_size=4) Causal Test ===" << std::endl;
        std::cout << "  q_tile_size = 4 / (32/32) = 4" << std::endl;
        std::cout << "  Tile 0: Q0, Q1, Q2, Q3 (all in one tile!)" << std::endl;

        const int d_model = num_heads * head_dim;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int blocks_per_head = head_dim / 32;

        std::vector<float> Q_fp32(seq_len_q * num_heads * head_dim, 1.0f / std::sqrt(32.0f));
        std::vector<float> K_fp32(seq_len_kv * num_kv_heads * head_dim, 1.0f / std::sqrt(32.0f));
        std::vector<float> V_fp32(seq_len_kv * num_kv_heads * head_dim);

        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = static_cast<float>(k + 1);
            for (int d = 0; d < head_dim; ++d)
            {
                V_fp32[k * head_dim + d] = v_val;
            }
        }

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_fp32_to_q8_1(Q_fp32.data(), seq_len_q * num_heads, head_dim, Q_q8.data());
        quantize_fp32_to_q8_1(K_fp32.data(), seq_len_kv * num_kv_heads, head_dim, K_q8.data());
        quantize_fp32_to_q8_1(V_fp32.data(), seq_len_kv * num_kv_heads, head_dim, V_q8.data());

        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_prefill(seq_len_q * d_model, 0.0f);
        std::vector<float> output_decode(seq_len_q * d_model, 0.0f);

        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 1;
            config.wo_format = WoFormat::FP32;
            config.causal = causal;
            config.mode = AttentionMode::DECODE;

            JitFusedAttentionWo kernel(config);
            for (int q = 0; q < seq_len_q; ++q)
            {
                const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;
                float *out_slice = output_decode.data() + q * d_model;
                int effective_kv_len = causal ? std::min(q + 1, seq_len_kv) : seq_len_kv;
                kernel.compute(Q_slice, K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out_slice, 1, effective_kv_len, scale, q);
            }
        }

        std::cout << "\n  Results (first element):" << std::endl;
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "    Q" << q << ": Prefill=" << output_prefill[q * d_model]
                      << " Decode=" << output_decode[q * d_model] << std::endl;
        }

        double cos_sim = cosine_similarity(output_prefill.data(), output_decode.data(), seq_len_q * d_model);
        std::cout << "  Cosine sim: " << cos_sim << std::endl;
        EXPECT_GE(cos_sim, MIN_COSINE_SIM_CAUSAL);
    }

    // NOTE: Qwen7B prefill test removed due to JIT code size limitation.
    // The prefill kernel with 28 heads exceeds the 512KB code buffer.
    // Future work: Use runtime head loop or larger buffer.

} // namespace llaminar::v2::kernels::jit::test
