#include <gtest/gtest.h>
#include "kernels/common/attention_primitives.h"
#include <vector>
#include <cmath>

using namespace llaminar::attn;

static void fill(std::vector<float> &v, float scale)
{
    for (size_t i = 0; i < v.size(); ++i)
        v[i] = scale * float((i % 67) + 1);
}

TEST(AttentionPrimitives, FusedMatchesStep)
{
    const int heads = 2, head_dim = 8, seq = 4;
    std::vector<float> q(seq * heads * head_dim), k(q.size()), v(q.size());
    fill(q, 0.01f);
    fill(k, 0.015f);
    fill(v, 0.02f);
    auto q2 = q, k2 = k;
    apply_rope(q.data(), k.data(), seq, head_dim, heads, 0, 10000.f);
    apply_rope(q2.data(), k2.data(), seq, head_dim, heads, 0, 10000.f);
    std::vector<float> scores(size_t(heads) * seq * seq);
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, head_dim, heads, true, true);
    std::vector<float> out_step(seq * heads * head_dim);
    apply_scores_to_v(scores.data(), v.data(), out_step.data(), seq, head_dim, heads);
    std::vector<float> out_fused(out_step.size());
    fused_attention(q2.data(), k2.data(), v.data(), out_fused.data(), seq, head_dim, heads, true);
    for (size_t i = 0; i < out_step.size(); ++i)
        ASSERT_NEAR(out_step[i], out_fused[i], 1e-6f) << "idx=" << i;
}

TEST(AttentionPrimitives, SoftmaxProperties)
{
    const int heads = 1, head_dim = 8, seq = 5;
    std::vector<float> q(seq * heads * head_dim), k(q.size()), v(q.size());
    fill(q, 0.01f);
    fill(k, 0.011f);
    fill(v, 0.012f);
    apply_rope(q.data(), k.data(), seq, head_dim, heads, 0, 10000.f);
    std::vector<float> scores(size_t(heads) * seq * seq);
    compute_qk_scores(q.data(), k.data(), scores.data(), seq, head_dim, heads, true, true);
    auto stats = validate_softmax_rows(scores.data(), seq, heads);
    EXPECT_LT(stats.max_row_deviation, 1e-6f);
    EXPECT_GE(stats.max_negative, 0.f);
    EXPECT_LE(stats.max_prob, 1.f);
}

TEST(AttentionPrimitives, RoPEPairNormInvariant)
{
    const int heads = 1, head_dim = 8, seq = 3;
    std::vector<float> q(seq * heads * head_dim), k(q.size());
    fill(q, 0.02f);
    k = q;
    std::vector<float> before;
    before.reserve(seq * head_dim / 2);
    for (int t = 0; t < seq; ++t)
    {
        float *row = q.data() + t * head_dim;
        for (int pair = 0; pair < head_dim / 2; ++pair)
        {
            float a = row[2 * pair], b = row[2 * pair + 1];
            before.push_back(std::sqrt(a * a + b * b));
        }
    }
    apply_rope(q.data(), k.data(), seq, head_dim, heads, 0, 10000.f);
    int idx = 0;
    for (int t = 0; t < seq; ++t)
    {
        float *row = q.data() + t * head_dim;
        for (int pair = 0; pair < head_dim / 2; ++pair)
        {
            float a = row[2 * pair], b = row[2 * pair + 1];
            float r = std::sqrt(a * a + b * b);
            ASSERT_NEAR(r, before[idx++], 1e-6f);
        }
    }
}
