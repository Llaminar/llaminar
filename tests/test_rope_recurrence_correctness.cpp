// Parity test comparing recurrence vs legacy RoPE paths
#include <gtest/gtest.h>
#include <vector>
#include <cstdlib>
#include <cmath>     // for std::sqrt, std::abs
#include <algorithm> // for std::max
#include "kernels/common/attention_primitives.h"
#include "utils/debug_env.h"

using namespace llaminar::attn;

static void fill(std::vector<float> &v, int seed_mul)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = float(((i * 13u) + seed_mul) % 101) / 50.0f - 1.0f; // deterministic pseudo-random
}

static void run_case(int heads, int head_dim, int seq_len)
{
    const int n_past = 7; // non-zero to exercise base angle
    size_t elems = (size_t)seq_len * heads * head_dim;
    std::vector<float> q_ref(elems), k_ref(elems), q_test(elems), k_test(elems);
    fill(q_ref, 3);
    fill(k_ref, 5);
    q_test = q_ref;
    k_test = k_ref;

    // Force legacy path
    ::setenv("LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE", "1", 1);
    llaminar::debugEnvRefresh();
    apply_rope(q_ref.data(), k_ref.data(), seq_len, head_dim, heads, heads, n_past, 10000.f);

    // Enable recurrence (and allow auto-tune); tweak threshold low so it activates
    ::setenv("LLAMINAR_ATTN_PRIM_ROPE_DISABLE_RECURRENCE", "0", 1);
    ::setenv("LLAMINAR_ATTN_PRIM_ROPE_RECURRENCE_THRESHOLD", "1", 1);
    llaminar::debugEnvRefresh();
    apply_rope(q_test.data(), k_test.data(), seq_len, head_dim, heads, heads, n_past, 10000.f);

    double max_abs_q = 0.0, max_abs_k = 0.0, rel_l2_q = 0.0, rel_l2_k = 0.0;
    double ref_q_norm = 0.0, diff_q_norm = 0.0, ref_k_norm = 0.0, diff_k_norm = 0.0;
    for (size_t i = 0; i < elems; ++i)
    {
        double dq = (double)q_test[i] - (double)q_ref[i];
        double dr = (double)k_test[i] - (double)k_ref[i];
        max_abs_q = std::max(max_abs_q, std::abs(dq));
        max_abs_k = std::max(max_abs_k, std::abs(dr));
        ref_q_norm += (double)q_ref[i] * (double)q_ref[i];
        ref_k_norm += (double)k_ref[i] * (double)k_ref[i];
        diff_q_norm += dq * dq;
        diff_k_norm += dr * dr;
    }
    rel_l2_q = std::sqrt(diff_q_norm) / std::sqrt(ref_q_norm + 1e-30);
    rel_l2_k = std::sqrt(diff_k_norm) / std::sqrt(ref_k_norm + 1e-30);

    ASSERT_LT(max_abs_q, 1e-5) << "max_abs_q exceeded";
    ASSERT_LT(max_abs_k, 1e-5) << "max_abs_k exceeded";
    ASSERT_LT(rel_l2_q, 1e-6) << "rel_l2_q exceeded";
    ASSERT_LT(rel_l2_k, 1e-6) << "rel_l2_k exceeded";
}

TEST(RoPERecurrenceParity, SmallHeads) { run_case(2, 32, 16); }
TEST(RoPERecurrenceParity, Medium) { run_case(4, 64, 48); }
TEST(RoPERecurrenceParity, Larger) { run_case(8, 64, 96); }
TEST(RoPERecurrenceParity, Large) { run_case(8, 128, 128); }
