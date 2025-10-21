// Copyright (c) 2025
// @author David Sanftenberg
//
// Tests for persistent RoPE decode state correctness.

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <cstdlib>
#include "operators/common/AttentionPrimitives.h"
#include "utils/DebugEnv.h"

using llaminar::attn::apply_rope;
using llaminar::attn::apply_rope_experimental;

namespace
{

    struct RopeFixture : public ::testing::Test
    {
        void SetUp() override
        {
            // Ensure experimental path + decode fast path enabled.
            setenv("LLAMINAR_ATTN_PRIM_ROPE_EXPERIMENTAL", "1", 1);
            setenv("LLAMINAR_ATTN_PRIM_ROPE_DECODE_FASTPATH", "1", 1);
            setenv("LLAMINAR_ATTN_PRIM_ROPE_PERSIST_STATE", "1", 1);
            // Force disable table to exercise persistent logic without table path stealing it.
            setenv("LLAMINAR_ATTN_PRIM_ROPE_TABLE_THRESHOLD", "100000000", 1); // huge threshold so table not used
        }
    };

    static void fill_seed(std::vector<float> &q, std::vector<float> &k)
    {
        for (size_t i = 0; i < q.size(); ++i)
        {
            int vq = static_cast<int>(i % 97) - 48;
            int vk = static_cast<int>((i * 131) % 113) - 56;
            q[i] = static_cast<float>(vq) / 50.f;
            k[i] = static_cast<float>(vk) / 60.f;
        }
    }

    static void rotate_single_reference(std::vector<float> &q, std::vector<float> &k, int head_dim, int q_heads, int k_heads, int n_past, float freq_base)
    {
        // Use experimental without persistence by disabling flag temporarily.
        unsetenv("LLAMINAR_ATTN_PRIM_ROPE_PERSIST_STATE");
        apply_rope_experimental(q.data(), k.data(), /*seq_len*/ 1, head_dim, q_heads, k_heads, n_past, freq_base);
        setenv("LLAMINAR_ATTN_PRIM_ROPE_PERSIST_STATE", "1", 1); // restore
    }

    static double max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
    {
        double m = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
        {
            m = std::max(m, std::abs((double)a[i] - b[i]));
        }
        return m;
    }

    TEST_F(RopeFixture, IncrementalParitySequentialSteps)
    {
        const int head_dim = 128;
        const int q_heads = 8;
        const int k_heads = 8;
        const float freq_base = 10000.f;
        const int steps = 128;
        // Seed token embedding (single token) reused each step to simulate new token with same base pattern.
        std::vector<float> q_seed(head_dim * q_heads), k_seed(head_dim * k_heads);
        fill_seed(q_seed, k_seed);
        for (int n = 0; n < steps; ++n)
        {
            // Prepare fresh copies for persistent and reference paths
            std::vector<float> q_persist = q_seed;
            std::vector<float> k_persist = k_seed;
            std::vector<float> q_ref = q_seed;
            std::vector<float> k_ref = k_seed;
            // Persistent path rotation (advances internal static state)
            apply_rope_experimental(q_persist.data(), k_persist.data(), 1, head_dim, q_heads, k_heads, n, freq_base);
            // Reference direct rotation for position n
            rotate_single_reference(q_ref, k_ref, head_dim, q_heads, k_heads, n, freq_base);
            double mdq = max_abs_diff(q_persist, q_ref);
            double mdk = max_abs_diff(k_persist, k_ref);
            ASSERT_LT(mdq, 1e-5) << "Mismatch Q at step " << n;
            ASSERT_LT(mdk, 1e-5) << "Mismatch K at step " << n;
        }
    }

    TEST_F(RopeFixture, RestartResetsPersistentState)
    {
        const int head_dim = 64;
        const int q_heads = 4;
        const int k_heads = 4;
        const float freq_base = 10000.f;
        std::vector<float> q0(head_dim * q_heads), k0(head_dim * k_heads);
        std::vector<float> q1(head_dim * q_heads), k1(head_dim * k_heads);
        fill_seed(q0, k0);
        fill_seed(q1, k1);
        // Advance some steps
        for (int n = 0; n < 32; ++n)
        {
            apply_rope_experimental(q0.data(), k0.data(), 1, head_dim, q_heads, k_heads, n, freq_base);
        }
        // Restart sequence (n_past=0) on second copy should match first copy reinitialized to step 0
        apply_rope_experimental(q1.data(), k1.data(), 1, head_dim, q_heads, k_heads, 0, freq_base);
        // Reference fresh rotation at 0
        std::vector<float> q_ref = q0;
        std::vector<float> k_ref = k0; // pull original? need fresh seed
        fill_seed(q_ref, k_ref);
        rotate_single_reference(q_ref, k_ref, head_dim, q_heads, k_heads, 0, freq_base);
        double mdq = max_abs_diff(q1, q_ref);
        double mdk = max_abs_diff(k1, k_ref);
        ASSERT_LT(mdq, 1e-6);
        ASSERT_LT(mdk, 1e-6);
    }

    TEST_F(RopeFixture, LargeJumpFastPathParity)
    {
        const int head_dim = 128;
        const int q_heads = 4;
        const int k_heads = 4;
        const float freq_base = 10000.f;
        std::vector<float> qA(head_dim * q_heads), kA(head_dim * k_heads);
        std::vector<float> qB(head_dim * q_heads), kB(head_dim * k_heads);
        fill_seed(qA, kA);
        fill_seed(qB, kB);
        // Initialize to position 0
        apply_rope_experimental(qA.data(), kA.data(), 1, head_dim, q_heads, k_heads, 0, freq_base);
        apply_rope_experimental(qB.data(), kB.data(), 1, head_dim, q_heads, k_heads, 0, freq_base);
        // Huge jump triggering large-jump recompute (threshold is 16, jump 100)
        int jump_pos = 100;
        apply_rope_experimental(qA.data(), kA.data(), 1, head_dim, q_heads, k_heads, jump_pos, freq_base);
        // Reference compute directly for position 100
        std::vector<float> q_ref(head_dim * q_heads), k_ref(head_dim * k_heads);
        fill_seed(q_ref, k_ref);
        rotate_single_reference(q_ref, k_ref, head_dim, q_heads, k_heads, jump_pos, freq_base);
        double mdq = max_abs_diff(qA, q_ref);
        double mdk = max_abs_diff(kA, k_ref);
        ASSERT_LT(mdq, 1e-5);
        ASSERT_LT(mdk, 1e-5);
    }

} // namespace
