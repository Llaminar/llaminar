/**
 * @file Test__JitFusedAttentionWo_Debug.cpp
 * @brief Debug test to isolate prefill alternating zeros bug
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstdlib>

#include "kernels/cpu/attention/q8_1/jit/JitFusedAttentionWo.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"

namespace llaminar::v2::kernels::jit::test
{

    using llaminar2::fp16_to_fp32;
    using llaminar2::fp32_to_fp16;
    using microkernels::Q8_1Block;

    class Test__JitFusedAttentionWo_Debug : public ::testing::Test
    {
    protected:
        void quantize_constant_q8(Q8_1Block *blocks, int num_rows, int num_blocks_per_row, float value)
        {
            for (int row = 0; row < num_rows; ++row)
            {
                for (int b = 0; b < num_blocks_per_row; ++b)
                {
                    Q8_1Block &blk = blocks[row * num_blocks_per_row + b];

                    float scale = std::fabs(value) / 127.0f;
                    if (scale < 1e-10f)
                        scale = 1e-10f;

                    int8_t q_val = static_cast<int8_t>(std::round(value / scale));
                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        blk.qs[i] = q_val;
                        sum_qs += q_val;
                    }

                    blk.d = fp32_to_fp16(scale);
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        }
    };

    /**
     * Minimal test with constant inputs to debug alternating zeros.
     * With constant Q, K, V, the output should be the same for all queries.
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, ConstantInputs_TwoQueries)
    {
        // Minimal configuration
        const int seq_len_q = 2;  // Just 2 queries
        const int seq_len_kv = 1; // Just 1 KV position
        const int num_heads = 4;
        const int num_kv_heads = 2;
        const int head_dim = 64;
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Debug Test: Constant Inputs ===\n";
        std::cout << "seq_len_q=" << seq_len_q << ", seq_len_kv=" << seq_len_kv << "\n";
        std::cout << "heads=" << num_heads << "/" << num_kv_heads << ", head_dim=" << head_dim << "\n";
        std::cout << "scale=" << scale << "\n";
        std::cout << "blocks_per_head=" << blocks_per_head << "\n";
        std::cout << "q_tile_size should be: 4 / " << blocks_per_head << " = " << (4 / blocks_per_head) << "\n\n";

        // Allocate Q, K, V with constant values
        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        // Fill with constant value (1.0)
        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, 1.0f);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, 1.0f);
        quantize_constant_q8(V_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, 1.0f);

        // Debug: Print Q block info
        std::cout << "Q block 0 (Q[0], H[0]): scale=" << fp16_to_fp32(Q_q8[0].d)
                  << ", sum_qs=" << Q_q8[0].sum_qs << "\n";
        std::cout << "Q block " << (num_heads * blocks_per_head) << " (Q[1], H[0]): scale="
                  << fp16_to_fp32(Q_q8[num_heads * blocks_per_head].d)
                  << ", sum_qs=" << Q_q8[num_heads * blocks_per_head].sum_qs << "\n";

        // Identity Wo projection
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        // Output buffer - initialize with sentinel
        std::vector<float> output_prefill(seq_len_q * d_model, -999.0f);

        // Create and run prefill kernel
        JitAttentionConfig config;
        config.head_dim = head_dim;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.batch_size = seq_len_q;
        config.wo_format = WoFormat::FP32;
        config.causal = false;
        config.mode = AttentionMode::PREFILL;

        std::cout << "\nCreating prefill kernel...\n";
        JitFusedAttentionWo kernel(config);

        std::cout << "Running prefill kernel...\n";
        kernel.compute(
            Q_q8.data(),
            K_q8.data(),
            V_q8.data(),
            Wo_fp32.data(),
            output_prefill.data(),
            seq_len_q,
            seq_len_kv,
            scale,
            0); // position_offset

        // Check outputs
        std::cout << "\n=== Output Analysis ===\n";

        for (int q = 0; q < seq_len_q; ++q)
        {
            float sum = 0.0f;
            float max_val = -1e30f;
            float min_val = 1e30f;
            int num_zeros = 0;
            int num_sentinel = 0;

            for (int i = 0; i < d_model; ++i)
            {
                float v = output_prefill[q * d_model + i];
                sum += v;
                max_val = std::max(max_val, v);
                min_val = std::min(min_val, v);
                if (std::fabs(v) < 1e-10f)
                    num_zeros++;
                if (std::fabs(v - (-999.0f)) < 1e-6f)
                    num_sentinel++;
            }

            std::cout << "Query " << q << ":\n";
            std::cout << "  sum=" << sum << ", min=" << min_val << ", max=" << max_val << "\n";
            std::cout << "  zeros=" << num_zeros << "/" << d_model << "\n";
            std::cout << "  sentinel (-999)=" << num_sentinel << "/" << d_model << "\n";
            std::cout << "  first 8: ";
            for (int i = 0; i < 8; ++i)
            {
                std::cout << output_prefill[q * d_model + i] << " ";
            }
            std::cout << "\n\n";

            // Query should have non-zero, non-sentinel output
            bool all_sentinel = (num_sentinel == d_model);
            bool all_zeros = (num_zeros == d_model);

            EXPECT_FALSE(all_sentinel)
                << "Query " << q << " output was never written (all -999)!";
            EXPECT_FALSE(all_zeros)
                << "Query " << q << " has all zero output!";
        }
    }

    /**
     * Run decode on same inputs to compare
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, ConstantInputs_DecodeBaseline)
    {
        const int seq_len_kv = 1;
        const int num_heads = 4;
        const int num_kv_heads = 2;
        const int head_dim = 64;
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Decode Baseline Test ===\n";

        // Allocate Q, K, V with constant values
        std::vector<Q8_1Block> Q_q8(num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), num_heads, blocks_per_head, 1.0f);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, 1.0f);
        quantize_constant_q8(V_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, 1.0f);

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_decode(d_model, -999.0f);

        JitAttentionConfig config;
        config.head_dim = head_dim;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.batch_size = 1;
        config.wo_format = WoFormat::FP32;
        config.causal = false;
        config.mode = AttentionMode::DECODE;

        JitFusedAttentionWo kernel(config);

        kernel.compute(
            Q_q8.data(),
            K_q8.data(),
            V_q8.data(),
            Wo_fp32.data(),
            output_decode.data(),
            1,
            seq_len_kv,
            scale,
            0);

        float sum = 0.0f;
        for (int i = 0; i < d_model; ++i)
        {
            sum += output_decode[i];
        }

        std::cout << "Decode output sum: " << sum << "\n";
        std::cout << "First 8: ";
        for (int i = 0; i < 8; ++i)
        {
            std::cout << output_decode[i] << " ";
        }
        std::cout << "\n";

        EXPECT_GT(std::fabs(sum), 1e-6f) << "Decode output is all zeros!";
    }

    /**
     * Debug test with simple values to trace causal mask bug.
     * Use V values that identify each KV position (V[k] = k+1).
     * With uniform attention weights, Q2 should get (1+2+3)/3 = 2.0, not (1+2+3+4)/4 = 2.5
     *
     * Test with head_dim=32 -> q_tile_size=4, all queries in ONE tile
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, CausalMaskTrace_SingleTile)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 32; // q_tile_size = 4, ONE tile for all queries
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Causal Mask Trace (Single Tile) ===\n";
        std::cout << "q_tile_size = 4 / (32/32) = 4\n";
        std::cout << "Single tile: Q0, Q1, Q2, Q3 (tile_start=0)\n\n";

        // Uniform Q and K
        float uniform_val = 1.0f / std::sqrt(32.0f);

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, uniform_val);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, uniform_val);

        // V values that identify each position
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = static_cast<float>(k + 1);
            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q8_1Block &blk = V_q8[k * blocks_per_head + b];
                float blk_scale = v_val / 127.0f;
                blk.d = fp32_to_fp16(blk_scale);
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    blk.qs[i] = 127;
                    sum_qs += 127;
                }
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        // Run PREFILL
        std::vector<float> output_prefill(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        // Run DECODE for comparison
        std::vector<float> output_decode(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 1;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::DECODE;

            JitFusedAttentionWo kernel(config);
            for (int q = 0; q < seq_len_q; ++q)
            {
                const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;
                float *out_slice = output_decode.data() + q * d_model;
                int effective_kv_len = std::min(q + 1, seq_len_kv);
                kernel.compute(Q_slice, K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out_slice, 1, effective_kv_len, scale, q);
            }
        }

        std::cout << "Results (first element):\n";
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ": Prefill=" << output_prefill[q * d_model]
                      << " Decode=" << output_decode[q * d_model];
            if (std::fabs(output_prefill[q * d_model] - output_decode[q * d_model]) > 0.01f)
            {
                std::cout << " *** MISMATCH ***";
            }
            std::cout << "\n";
        }

        // Verify Q2 is correct
        float q2_prefill = output_prefill[2 * d_model];
        float q2_decode = output_decode[2 * d_model];

        EXPECT_NEAR(q2_prefill, q2_decode, 0.05f) << "Q2 prefill vs decode mismatch!";
    }

    /**
     * Debug test with simple values to trace causal mask bug.
     * Use V values that identify each KV position (V[k] = k+1).
     * With uniform attention weights, Q2 should get (1+2+3)/3 = 2.0, not (1+2+3+4)/4 = 2.5
     *
     * Test with head_dim=64 -> q_tile_size=2, TWO tiles
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, CausalMaskTrace_SingleHead)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1; // Single head to simplify
        const int num_kv_heads = 1;
        const int head_dim = 64; // q_tile_size = 2
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Causal Mask Trace (Single Head) ===\n";
        std::cout << "q_tile_size = 4 / (64/32) = 2\n";
        std::cout << "Tile 0: Q0, Q1 (tile_start=0)\n";
        std::cout << "Tile 1: Q2, Q3 (tile_start=2)\n";
        std::cout << "\nExpected q_local_start values for Tile 1:\n";
        std::cout << "  kv_idx=0: q_local_start = max(0, 0 - 2) = 0 -> both Q2, Q3 process\n";
        std::cout << "  kv_idx=1: q_local_start = max(0, 1 - 2) = 0 -> both Q2, Q3 process\n";
        std::cout << "  kv_idx=2: q_local_start = max(0, 2 - 2) = 0 -> both Q2, Q3 process\n";
        std::cout << "  kv_idx=3: q_local_start = max(0, 3 - 2) = 1 -> only Q3 (q_local=1) processes\n\n";

        // Uniform Q and K (all elements = 1/sqrt(64))
        float uniform_val = 1.0f / std::sqrt(64.0f);

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, uniform_val);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, uniform_val);

        // V values that identify each position: V[k] = k+1
        // After attention, output should reveal which KVs were attended
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = static_cast<float>(k + 1);
            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q8_1Block &blk = V_q8[k * blocks_per_head + b];
                float blk_scale = v_val / 127.0f;
                blk.d = fp32_to_fp16(blk_scale);
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    blk.qs[i] = 127; // Max value
                    sum_qs += 127;
                }
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }

        std::cout << "V values: V[0]=1, V[1]=2, V[2]=3, V[3]=4\n";
        std::cout << "Expected with causal mask:\n";
        std::cout << "  Q0: attends to K0 only → avg = 1.0\n";
        std::cout << "  Q1: attends to K0,K1 → avg = (1+2)/2 = 1.5\n";
        std::cout << "  Q2: attends to K0,K1,K2 → avg = (1+2+3)/3 ≈ 2.0\n";
        std::cout << "  Q3: attends to K0,K1,K2,K3 → avg = (1+2+3+4)/4 = 2.5\n\n";

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        // Run PREFILL
        std::vector<float> output_prefill(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        // Run DECODE for comparison
        std::vector<float> output_decode(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 1;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::DECODE;

            JitFusedAttentionWo kernel(config);
            for (int q = 0; q < seq_len_q; ++q)
            {
                const Q8_1Block *Q_slice = Q_q8.data() + q * num_heads * blocks_per_head;
                float *out_slice = output_decode.data() + q * d_model;
                int effective_kv_len = std::min(q + 1, seq_len_kv); // Causal: only up to position q
                kernel.compute(Q_slice, K_q8.data(), V_q8.data(), Wo_fp32.data(),
                               out_slice, 1, effective_kv_len, scale, q);
            }
        }

        std::cout << "Results (first element of each query):\n";
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ": Prefill=" << output_prefill[q * d_model]
                      << " Decode=" << output_decode[q * d_model];
            if (std::fabs(output_prefill[q * d_model] - output_decode[q * d_model]) > 0.01f)
            {
                std::cout << " *** MISMATCH ***";
            }
            std::cout << "\n";
        }

        // Verify Q2 is correct (should be ~2.0, not ~2.5)
        float q2_prefill = output_prefill[2 * d_model];
        float q2_decode = output_decode[2 * d_model];

        std::cout << "\nQ2 analysis:\n";
        std::cout << "  If Q2 attends to K0,K1,K2 only: expect ~2.0\n";
        std::cout << "  If Q2 attends to K0,K1,K2,K3: expect ~2.5\n";
        std::cout << "  Actual prefill: " << q2_prefill << "\n";
        std::cout << "  Actual decode:  " << q2_decode << "\n";

        // Also check ALL d_model elements to see if there's a pattern
        std::cout << "\nFull d_model output for Q2 (first 8 elements):\n";
        std::cout << "  Prefill: ";
        for (int i = 0; i < std::min(8, d_model); ++i)
        {
            std::cout << output_prefill[2 * d_model + i] << " ";
        }
        std::cout << "\n  Decode:  ";
        for (int i = 0; i < std::min(8, d_model); ++i)
        {
            std::cout << output_decode[2 * d_model + i] << " ";
        }
        std::cout << "\n";

        EXPECT_NEAR(q2_prefill, q2_decode, 0.05f) << "Q2 prefill vs decode mismatch!";
    }

    /**
     * Test to dump and examine the generated JIT code for causal mask checks.
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, DumpJitCodeForCausalMask)
    {
        auto dump_code = [](int head_dim, const std::string &suffix)
        {
            const int num_heads = 1;
            const int num_kv_heads = 1;

            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = 4;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWoGenerator generator(config);

            const uint8_t *code = generator.getCode();
            size_t code_size = generator.getSize();

            std::string tmp_file = "/tmp/jit_prefill_" + suffix + ".bin";
            std::ofstream ofs(tmp_file, std::ios::binary);
            ofs.write(reinterpret_cast<const char *>(code), code_size);
            ofs.close();

            std::cout << "\n=== " << suffix << " (head_dim=" << head_dim << ", q_tile_size=" << (4 / (head_dim / 32)) << ") ===\n";
            std::cout << "Code size: " << code_size << " bytes\n";
            std::cout << "Dumped to " << tmp_file << "\n";
        };

        dump_code(32, "single_tile"); // q_tile_size = 4
        dump_code(64, "multi_tile");  // q_tile_size = 2

        std::cout << "\nTo compare, run:\n";
        std::cout << "  objdump -D -b binary -m i386:x86-64 /tmp/jit_prefill_single_tile.bin > /tmp/single.asm\n";
        std::cout << "  objdump -D -b binary -m i386:x86-64 /tmp/jit_prefill_multi_tile.bin > /tmp/multi.asm\n";
        std::cout << "  diff /tmp/single.asm /tmp/multi.asm | less\n";

        SUCCEED();
    }

    /**
     * Test NON-CAUSAL prefill to verify basic attention works correctly.
     * If non-causal works but causal doesn't, the bug is in the causal check.
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, NonCausalBaseline)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64; // q_tile_size = 2
        const int d_model = num_heads * head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== NON-CAUSAL Baseline Test ===\n";
        std::cout << "With non-causal: all queries attend to all KVs\n";
        std::cout << "Expected output: (1+2+3+4)/4 = 2.5 for all queries\n\n";

        float uniform_val = 1.0f / std::sqrt(64.0f);

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, uniform_val);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, uniform_val);

        // V values: V[k] = k+1
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = static_cast<float>(k + 1);
            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q8_1Block &blk = V_q8[k * blocks_per_head + b];
                float blk_scale = v_val / 127.0f;
                blk.d = fp32_to_fp16(blk_scale);
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    blk.qs[i] = 127;
                    sum_qs += 127;
                }
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }

        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output_prefill(seq_len_q * d_model, -999.0f);

        JitAttentionConfig config;
        config.head_dim = head_dim;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.batch_size = seq_len_q;
        config.wo_format = WoFormat::FP32;
        config.causal = false; // Non-causal!
        config.mode = AttentionMode::PREFILL;

        JitFusedAttentionWo kernel(config);
        kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                       output_prefill.data(), seq_len_q, seq_len_kv, scale, 0);

        std::cout << "Results (first element):\n";
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ": " << output_prefill[q * d_model] << "\n";
            EXPECT_NEAR(output_prefill[q * d_model], 2.5f, 0.05f)
                << "Q" << q << " should be ~2.5 (non-causal attends to all KVs)";
        }
    }

    /**
     * Test causal mask logic directly in C++ (no JIT) to verify expected behavior.
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, CausalMaskLogic_Reference)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int q_tile_size = 2; // As in head_dim=64 case

        std::cout << "\n=== Causal Mask Logic Reference Test ===\n";
        std::cout << "seq_len_q=" << seq_len_q << ", seq_len_kv=" << seq_len_kv << "\n";
        std::cout << "q_tile_size=" << q_tile_size << "\n\n";

        // Simulate the prefill kernel's causal mask logic
        for (int tile_start = 0; tile_start < seq_len_q; tile_start += q_tile_size)
        {
            int tile_size = std::min(q_tile_size, seq_len_q - tile_start);
            std::cout << "Tile: tile_start=" << tile_start << ", tile_size=" << tile_size << "\n";

            // Track which queries attend to which KV positions
            std::vector<std::vector<bool>> attended(tile_size, std::vector<bool>(seq_len_kv, false));

            for (int kv_idx = 0; kv_idx < seq_len_kv; ++kv_idx)
            {
                // This is the causal mask calculation from the JIT code
                int q_start = std::max(0, kv_idx - 0); // position_offset=0
                int q_local_start = std::max(0, q_start - tile_start);

                std::cout << "  kv_idx=" << kv_idx << ": q_local_start=" << q_local_start << " -> ";

                for (int q_local = 0; q_local < tile_size; ++q_local)
                {
                    int q_global = tile_start + q_local;
                    // Causal check: q_local_start > q_local -> skip
                    bool should_skip = (q_local_start > q_local);

                    if (!should_skip)
                    {
                        attended[q_local][kv_idx] = true;
                        std::cout << "Q" << q_global << " ";
                    }
                }
                std::cout << "\n";
            }

            // Verify expected attendance pattern
            std::cout << "\n  Attendance matrix (rows=queries, cols=kv_pos):\n";
            for (int q_local = 0; q_local < tile_size; ++q_local)
            {
                int q_global = tile_start + q_local;
                std::cout << "    Q" << q_global << ": ";
                int count = 0;
                for (int kv = 0; kv < seq_len_kv; ++kv)
                {
                    std::cout << (attended[q_local][kv] ? "1" : "0");
                    if (attended[q_local][kv])
                        count++;
                }
                std::cout << " (attends to " << count << " positions)\n";

                // Verify expected count
                int expected_count = std::min(q_global + 1, seq_len_kv);
                EXPECT_EQ(count, expected_count) << "Q" << q_global << " should attend to " << expected_count << " positions";
            }
            std::cout << "\n";
        }
    }

    /**
     * Test with a sentinel value in V[3] to clearly identify if Q2 attends to it.
     * If the bug exists, Q2's output will be very different from expected.
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, CausalMaskSentinel)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64; // q_tile_size = 2 -> two tiles
        const int d_model = head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Causal Mask Sentinel Test ===\n";
        std::cout << "V[0]=1, V[1]=2, V[2]=3, V[3]=1000 (sentinel)\n";
        std::cout << "Q2 should attend to V[0,1,2] only -> avg ~2.0\n";
        std::cout << "If Q2 attends to V[3], output will be ~250+ (contaminated)\n\n";

        // Uniform Q and K
        float uniform_val = 1.0f / std::sqrt(64.0f);

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, uniform_val);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, uniform_val);

        // V values: V[0]=1, V[1]=2, V[2]=3, V[3]=1000 (sentinel)
        float v_values[4] = {1.0f, 2.0f, 3.0f, 1000.0f};
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = v_values[k];
            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q8_1Block &blk = V_q8[k * blocks_per_head + b];
                float blk_scale = v_val / 127.0f;
                blk.d = fp32_to_fp16(blk_scale);
                int32_t sum_qs = 0;
                for (int i = 0; i < 32; ++i)
                {
                    blk.qs[i] = 127;
                    sum_qs += 127;
                }
                blk.sum_qs = static_cast<int16_t>(sum_qs);
            }
        }

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        std::cout << "Results:\n";
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ": " << output[q * d_model];
            if (q == 2 && output[q * d_model] > 10.0f)
            {
                std::cout << " *** CONTAMINATED BY V[3] ***";
            }
            std::cout << "\n";
        }

        // Q2 should be ~2.0, not ~250
        float q2_output = output[2 * d_model];
        EXPECT_LT(q2_output, 10.0f) << "Q2 is contaminated by V[3] sentinel!";
    }

    /**
     * Test causal check in complete isolation by setting V[0..2] = 0 and V[3] = 1000
     * If Q2 gets ANY non-zero output, the causal check is failing
     */
    TEST_F(Test__JitFusedAttentionWo_Debug, CausalCheckIsolation)
    {
        const int seq_len_q = 4;
        const int seq_len_kv = 4;
        const int num_heads = 1;
        const int num_kv_heads = 1;
        const int head_dim = 64; // q_tile_size = 2 -> two tiles
        const int d_model = head_dim;
        const int blocks_per_head = head_dim / 32;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        std::cout << "\n=== Causal Check Isolation Test ===\n";
        std::cout << "V[0]=0, V[1]=0, V[2]=0, V[3]=1000\n";
        std::cout << "If causal check works, Q0,Q1,Q2 should be ~0\n";
        std::cout << "Only Q3 should have non-zero output\n\n";

        // Uniform Q and K
        float uniform_val = 1.0f / std::sqrt(64.0f);

        std::vector<Q8_1Block> Q_q8(seq_len_q * num_heads * blocks_per_head);
        std::vector<Q8_1Block> K_q8(seq_len_kv * num_kv_heads * blocks_per_head);
        std::vector<Q8_1Block> V_q8(seq_len_kv * num_kv_heads * blocks_per_head);

        quantize_constant_q8(Q_q8.data(), seq_len_q * num_heads, blocks_per_head, uniform_val);
        quantize_constant_q8(K_q8.data(), seq_len_kv * num_kv_heads, blocks_per_head, uniform_val);

        // V values: V[0]=0, V[1]=0, V[2]=0, V[3]=1000
        float v_values[4] = {0.0f, 0.0f, 0.0f, 1000.0f};
        for (int k = 0; k < seq_len_kv; ++k)
        {
            float v_val = v_values[k];
            for (int b = 0; b < blocks_per_head; ++b)
            {
                Q8_1Block &blk = V_q8[k * blocks_per_head + b];
                if (v_val == 0.0f)
                {
                    blk.d = fp32_to_fp16(0.0f);
                    blk.sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                        blk.qs[i] = 0;
                }
                else
                {
                    float blk_scale = v_val / 127.0f;
                    blk.d = fp32_to_fp16(blk_scale);
                    int32_t sum_qs = 0;
                    for (int i = 0; i < 32; ++i)
                    {
                        blk.qs[i] = 127;
                        sum_qs += 127;
                    }
                    blk.sum_qs = static_cast<int16_t>(sum_qs);
                }
            }
        }

        // Identity Wo
        std::vector<float> Wo_fp32(d_model * d_model, 0.0f);
        for (int i = 0; i < d_model; ++i)
        {
            Wo_fp32[i * d_model + i] = 1.0f;
        }

        std::vector<float> output(seq_len_q * d_model, -999.0f);
        {
            JitAttentionConfig config;
            config.head_dim = head_dim;
            config.num_heads = num_heads;
            config.num_kv_heads = num_kv_heads;
            config.batch_size = seq_len_q;
            config.wo_format = WoFormat::FP32;
            config.causal = true;
            config.mode = AttentionMode::PREFILL;

            JitFusedAttentionWo kernel(config);
            kernel.compute(Q_q8.data(), K_q8.data(), V_q8.data(), Wo_fp32.data(),
                           output.data(), seq_len_q, seq_len_kv, scale, 0);
        }

        std::cout << "Results:\n";
        for (int q = 0; q < seq_len_q; ++q)
        {
            std::cout << "  Q" << q << ": " << output[q * d_model];
            if (q < 3 && std::fabs(output[q * d_model]) > 1.0f)
            {
                std::cout << " *** CAUSAL CHECK FAILED - GOT V[3] ***";
            }
            std::cout << "\n";
        }

        // Q0, Q1, Q2 should be ~0 (only attend to V[0..2] which are 0)
        // Q3 should be ~1000 (attends to V[0..3], but V[0..2]=0, V[3]=1000)
        EXPECT_NEAR(output[0 * d_model], 0.0f, 1.0f) << "Q0 got V[3]!";
        EXPECT_NEAR(output[1 * d_model], 0.0f, 1.0f) << "Q1 got V[3]!";
        EXPECT_NEAR(output[2 * d_model], 0.0f, 1.0f) << "Q2 got V[3]!";
        EXPECT_GT(output[3 * d_model], 100.0f) << "Q3 should have V[3] influence";
    }

} // namespace
