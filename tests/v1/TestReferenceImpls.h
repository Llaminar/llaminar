#pragma once
/**
 * @file TestReferenceImpls.h
 * @brief Scalar / naive reference implementations for kernel parity tests.
 */
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

namespace testref
{

    inline std::vector<float> matmul_row_major(const float *A, const float *B,
                                               int M, int K, int N)
    {
        // A: [M,K], B: [K,N], row-major
        std::vector<float> C(M * N, 0.f);
        for (int m = 0; m < M; ++m)
        {
            for (int k = 0; k < K; ++k)
            {
                float a = A[m * K + k];
                const float *brow = B + k * N;
                float *crow = C.data() + m * N;
                for (int n = 0; n < N; ++n)
                {
                    crow[n] += a * brow[n];
                }
            }
        }
        return C;
    }

    inline void swiglu(std::vector<float> &gate, const std::vector<float> &up)
    {
        // gate & up share shape; gate mutated -> gate = swish(gate) * up
        assert(gate.size() == up.size());
        for (size_t i = 0; i < gate.size(); ++i)
        {
            float x = gate[i];
            float sig = 1.f / (1.f + std::exp(-x));
            float sw = x * sig;
            gate[i] = sw * up[i];
        }
    }

    inline void rmsnorm(std::vector<float> &row, const std::vector<float> &gamma, float eps)
    {
        assert(row.size() == gamma.size());
        double ss = 0.0;
        for (float v : row)
            ss += double(v) * v;
        double mean = ss / double(row.size());
        double r = 1.0 / std::sqrt(mean + eps);
        for (size_t i = 0; i < row.size(); ++i)
            row[i] = float(row[i] * r * gamma[i]);
    }

    inline void apply_rope(std::vector<float> &vec, int head_dim, int position)
    {
        // vec: [head_dim]; even head_dim required
        assert(head_dim % 2 == 0);
        int half = head_dim / 2;
        for (int i = 0; i < half; ++i)
        {
            int even = 2 * i;
            int odd = 2 * i + 1;
            float x0 = vec[even];
            float x1 = vec[odd];
            // Simple base theta formulation; real model may use varying frequencies.
            double theta = position / std::pow(10000.0, double(i) / double(half));
            double c = std::cos(theta);
            double s = std::sin(theta);
            vec[even] = float(x0 * c - x1 * s);
            vec[odd] = float(x0 * s + x1 * c);
        }
    }

    inline void attention_single_head(const std::vector<float> &Q, const std::vector<float> &K,
                                      const std::vector<float> &V, std::vector<float> &out,
                                      int seq_q, int seq_k, int head_dim, bool causal)
    {
        // Q: [seq_q, head_dim], K/V: [seq_k, head_dim]
        const float scale = 1.f / std::sqrt(float(head_dim));
        out.assign(seq_q * head_dim, 0.f);
        std::vector<float> scores(seq_k);
        for (int qi = 0; qi < seq_q; ++qi)
        {
            // scores
            for (int kj = 0; kj < seq_k; ++kj)
            {
                if (causal && kj > qi)
                {
                    scores[kj] = -1e30f;
                    continue;
                }
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d)
                {
                    dot += double(Q[qi * head_dim + d]) * double(K[kj * head_dim + d]);
                }
                scores[kj] = float(dot * scale);
            }
            // softmax
            float max_s = -1e30f;
            for (int kj = 0; kj < seq_k; ++kj)
                max_s = std::max(max_s, scores[kj]);
            double denom = 0.0;
            for (int kj = 0; kj < seq_k; ++kj)
            {
                double e = std::exp(double(scores[kj] - max_s));
                scores[kj] = float(e);
                denom += e;
            }
            for (int kj = 0; kj < seq_k; ++kj)
                scores[kj] = float(scores[kj] / denom);
            // weighted sum
            for (int kj = 0; kj < seq_k; ++kj)
            {
                float w = scores[kj];
                for (int d = 0; d < head_dim; ++d)
                {
                    out[qi * head_dim + d] += w * V[kj * head_dim + d];
                }
            }
        }
    }

} // namespace testref
