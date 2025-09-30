/**
 * @file attention_primitives.cpp
 */
#include "attention_primitives.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace llaminar::attn
{

    static inline float *head_row(float *base, int head, int i, int seq_len)
    {
        return base + head * seq_len * seq_len + i * seq_len;
    }
    static inline const float *head_row(const float *base, int head, int i, int seq_len)
    {
        return base + head * seq_len * seq_len + i * seq_len;
    }

    void apply_rope(float *q, float *k, int seq_len, int head_dim, int heads, int n_past, float freq_base)
    {
        for (int h = 0; h < heads; ++h)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                float *qh = q + t * heads * head_dim + h * head_dim;
                float *kh = k + t * heads * head_dim + h * head_dim;
                for (int pair = 0; pair < head_dim / 2; ++pair)
                {
                    float theta = 1.f / std::pow(freq_base, (2.f * pair) / head_dim);
                    float angle = float(n_past + t) * theta;
                    float cs = std::cos(angle), sn = std::sin(angle);
                    int i0 = 2 * pair, i1 = 2 * pair + 1;
                    float q0 = qh[i0], q1 = qh[i1];
                    float k0 = kh[i0], k1 = kh[i1];
                    qh[i0] = q0 * cs - q1 * sn;
                    qh[i1] = q0 * sn + q1 * cs;
                    kh[i0] = k0 * cs - k1 * sn;
                    kh[i1] = k0 * sn + k1 * cs;
                }
            }
        }
    }

    void compute_qk_scores(const float *q, const float *k, float *scores, int seq_len, int head_dim, int heads, bool causal, bool apply_softmax)
    {
        const float scale = 1.f / std::sqrt(float(head_dim));
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                const float *qi = q + i * heads * head_dim + h * head_dim;
                float *row = head_row(scores, h, i, seq_len);
                for (int j = 0; j < seq_len; ++j)
                {
                    const float *kj = k + j * heads * head_dim + h * head_dim;
                    float dot = 0.f;
                    for (int d = 0; d < head_dim; ++d)
                        dot += qi[d] * kj[d];
                    row[j] = dot * scale;
                }
            }
        }
        if (!apply_softmax)
            return;
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                float *row = head_row(scores, h, i, seq_len);
                int valid = seq_len;
                if (causal)
                {
                    for (int j = i + 1; j < seq_len; ++j)
                        row[j] = -INFINITY;
                    valid = i + 1;
                }
                float maxv = -INFINITY;
                for (int j = 0; j < valid; ++j)
                    maxv = std::max(maxv, row[j]);
                float sum = 0.f;
                for (int j = 0; j < valid; ++j)
                {
                    row[j] = std::exp(row[j] - maxv);
                    sum += row[j];
                }
                float inv = (sum == 0.f) ? 0.f : 1.f / sum;
                for (int j = 0; j < valid; ++j)
                    row[j] *= inv;
                for (int j = valid; j < seq_len; ++j)
                    row[j] = 0.f;
            }
        }
    }

    void apply_scores_to_v(const float *scores, const float *v, float *out, int seq_len, int head_dim, int heads)
    {
        std::fill(out, out + seq_len * heads * head_dim, 0.f);
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                float *dst = out + i * heads * head_dim + h * head_dim;
                const float *score_row = head_row(scores, h, i, seq_len);
                for (int j = 0; j < seq_len; ++j)
                {
                    const float *vj = v + j * heads * head_dim + h * head_dim;
                    float w = score_row[j];
                    for (int d = 0; d < head_dim; ++d)
                        dst[d] += w * vj[d];
                }
            }
        }
    }

    void fused_attention(const float *q, const float *k, const float *v, float *out, int seq_len, int head_dim, int heads, bool causal)
    {
        std::vector<float> scores(size_t(heads) * seq_len * seq_len);
        compute_qk_scores(q, k, scores.data(), seq_len, head_dim, heads, causal, true);
        apply_scores_to_v(scores.data(), v, out, seq_len, head_dim, heads);
    }

    RowSoftmaxStats validate_softmax_rows(const float *scores, int seq_len, int heads)
    {
        RowSoftmaxStats st{0.f, 0.f, 0.f};
        for (int h = 0; h < heads; ++h)
        {
            for (int i = 0; i < seq_len; ++i)
            {
                const float *row = head_row(scores, h, i, seq_len);
                float sum = 0.f;
                for (int j = 0; j < seq_len; ++j)
                    sum += row[j];
                st.max_row_deviation = std::max(st.max_row_deviation, std::fabs(sum - 1.f));
                for (int j = 0; j < seq_len; ++j)
                {
                    st.max_negative = std::min(st.max_negative, row[j]);
                    st.max_prob = std::max(st.max_prob, row[j]);
                }
            }
        }
        return st;
    }

} // namespace llaminar::attn
