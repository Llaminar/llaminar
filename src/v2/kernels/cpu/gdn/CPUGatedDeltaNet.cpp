/**
 * @file CPUGatedDeltaNet.cpp
 * @brief CPU implementation of delta rule recurrence for GDN linear attention
 *
 * Two execution modes:
 *
 * 1. Recurrent step (decode, seq_len=1):
 *    Direct state update following torch_recurrent_gated_delta_rule.
 *    Per-head: S = exp(g)*S, kv = S*k, delta = (v - kv)*beta, S += outer(k, delta), o = S*q
 *
 * 2. Chunk-forward (prefill, seq_len>1):
 *    Sequential per-timestep recurrence (functionally identical to chunk-parallel).
 *    The true chunk-parallel optimization is deferred to Phase F.
 *
 * The kernel owns ALL preprocessing:
 * - L2 normalization of Q and K (when use_qk_l2norm is true)
 * - Query scaling by 1/sqrt(d_k)
 * - Gate computation: g = -exp(A_log) * softplus(alpha + dt_bias)
 * - Beta sigmoid: beta_sig = sigmoid(beta_raw)
 *
 * Reference: torch_recurrent_gated_delta_rule() and torch_chunk_gated_delta_rule()
 *            from HuggingFace transformers 5.4.0
 */

#include "CPUGatedDeltaNet.h"
#include "../../../utils/OpenMPUtils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Preprocessing helpers
    // =========================================================================

    void CPUGatedDeltaNet::computeGates(
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *g_out, float *beta_sig_out,
        int seq_len, int n_heads)
    {
        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                const int idx = t * n_heads + h;

                const float x = alpha[idx] + dt_bias[h];
                const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                // GGUF stores -exp(A_log), so use it directly as the decay coefficient.
                // g = -exp(A_log) * softplus(alpha + dt_bias) = stored_value * softplus(...)
                g_out[idx] = A_log[h] * sp;

                beta_sig_out[idx] = 1.0f / (1.0f + std::exp(-beta_raw[idx]));
            }
        }
    }

    void CPUGatedDeltaNet::l2normalize(float *data, int seq_len, int n_heads, int head_dim)
    {
        constexpr float eps = 1e-6f;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                float *vec = data + t * n_heads * head_dim + h * head_dim;
                float norm_sq = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    norm_sq += vec[d] * vec[d];
                const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), eps);
                for (int d = 0; d < head_dim; ++d)
                    vec[d] *= inv_norm;
            }
        }
    }

    // =========================================================================
    // Recurrent step (decode, seq_len=1)
    // =========================================================================

    bool CPUGatedDeltaNet::recurrent_step(
        const float *q, const float *k, const float *v,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm)
    {
        // --- Preprocessing: copy Q/K, L2 norm, scale, compute gates ---
        const size_t qk_total = static_cast<size_t>(n_heads) * d_k;

        std::vector<float> q_buf(q, q + qk_total);
        std::vector<float> k_buf(k, k + qk_total);

        if (use_qk_l2norm)
        {
            l2normalize(q_buf.data(), 1, n_heads, d_k);
            l2normalize(k_buf.data(), 1, n_heads, d_k);
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (size_t i = 0; i < qk_total; ++i)
            q_buf[i] *= scale;

        std::vector<float> gate(n_heads);
        std::vector<float> beta_sig(n_heads);
        computeGates(alpha, beta_raw, A_log, dt_bias,
                     gate.data(), beta_sig.data(), 1, n_heads);

        // --- Core recurrence ---
        // Reference: torch_recurrent_gated_delta_rule() (single step)
        //
        // Per head h:
        //   S[h] *= exp(g[h])                          (decay)
        //   kv_mem = sum_j(S[h][j,:] * k[h][j])       (contract d_k)
        //   delta = (v[h] - kv_mem) * beta[h]
        //   S[h] += outer(k[h], delta)                 (rank-1 update)
        //   o[h] = sum_j(S[h][j,:] * q[h][j])         (contract d_k)
        //
        // State layout: [n_heads, d_k, d_v] row-major

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                float *S = state + static_cast<size_t>(h) * d_k * d_v;
                const float *q_h = q_buf.data() + h * d_k;
                const float *k_h = k_buf.data() + h * d_k;
                const float *v_h = v + h * d_v;
                const float g_h = gate[h];
                const float beta_h = beta_sig[h];
                float *o_h = output + h * d_v;

                // Step 1: Decay state
                const float decay = std::exp(g_h);
                for (int ij = 0; ij < d_k * d_v; ++ij)
                    S[ij] *= decay;

                // Step 2: kv_mem[v] = sum_j S[j,v] * k[j]
                std::vector<float> kv_mem(d_v, 0.0f);
                for (int j = 0; j < d_k; ++j)
                {
                    const float k_j = k_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        kv_mem[vi] += S[j * d_v + vi] * k_j;
                }

                // Step 3: delta = (v - kv_mem) * beta
                std::vector<float> delta(d_v);
                for (int vi = 0; vi < d_v; ++vi)
                    delta[vi] = (v_h[vi] - kv_mem[vi]) * beta_h;

                // Step 4: S += outer(k, delta)
                for (int j = 0; j < d_k; ++j)
                {
                    const float k_j = k_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        S[j * d_v + vi] += k_j * delta[vi];
                }

                // Step 5: output[v] = sum_j S[j,v] * q[j]
                for (int vi = 0; vi < d_v; ++vi)
                    o_h[vi] = 0.0f;
                for (int j = 0; j < d_k; ++j)
                {
                    const float q_j = q_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        o_h[vi] += S[j * d_v + vi] * q_j;
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

    // =========================================================================
    // Chunk forward (prefill, seq_len>1)
    // =========================================================================

    bool CPUGatedDeltaNet::chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int /*chunk_size*/, bool use_qk_l2norm)
    {
        // --- Preprocessing: copy Q/K, L2 norm, scale, compute gates ---
        const size_t qk_total = static_cast<size_t>(seq_len) * n_heads * d_k;

        std::vector<float> q_buf(Q, Q + qk_total);
        std::vector<float> k_buf(K, K + qk_total);

        if (use_qk_l2norm)
        {
            l2normalize(q_buf.data(), seq_len, n_heads, d_k);
            l2normalize(k_buf.data(), seq_len, n_heads, d_k);
        }

        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (size_t i = 0; i < qk_total; ++i)
            q_buf[i] *= scale_val;

        const size_t gate_size = static_cast<size_t>(seq_len) * n_heads;
        std::vector<float> gate(gate_size);
        std::vector<float> beta_sig(gate_size);
        computeGates(alpha, beta_raw, A_log, dt_bias,
                     gate.data(), beta_sig.data(), seq_len, n_heads);

        // --- Core recurrence ---
        // Sequential per-timestep recurrence over the full sequence.
        // Functionally identical to chunk-parallel; the true chunked
        // optimization is deferred to Phase F for better prefill performance.
        //
        // Q, K: [seq_len, n_heads * d_k]
        // V:    [seq_len, n_heads * d_v]
        // gate, beta: [seq_len, n_heads]
        // state: [n_heads, d_k, d_v]
        // output: [seq_len, n_heads * d_v]

        std::memset(output, 0, static_cast<size_t>(seq_len) * n_heads * d_v * sizeof(float));

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                float *S = state + static_cast<size_t>(h) * d_k * d_v;

                std::vector<float> kv_mem(d_v);
                std::vector<float> delta(d_v);

                for (int t = 0; t < seq_len; ++t)
                {
                    const float *q_t = q_buf.data() + t * n_heads * d_k + h * d_k;
                    const float *k_t = k_buf.data() + t * n_heads * d_k + h * d_k;
                    const float *v_t = V + t * n_heads * d_v + h * d_v;
                    const float g_t = gate[t * n_heads + h];
                    const float beta_t = beta_sig[t * n_heads + h];
                    float *o_t = output + t * n_heads * d_v + h * d_v;

                    // Step 1: Decay state
                    const float decay_val = std::exp(g_t);
                    for (int ij = 0; ij < d_k * d_v; ++ij)
                        S[ij] *= decay_val;

                    // Step 2: kv_mem = S^T * k (contract over d_k)
                    for (int vi = 0; vi < d_v; ++vi)
                        kv_mem[vi] = 0.0f;
                    for (int j = 0; j < d_k; ++j)
                    {
                        const float k_j = k_t[j];
                        for (int vi = 0; vi < d_v; ++vi)
                            kv_mem[vi] += S[j * d_v + vi] * k_j;
                    }

                    // Step 3: delta = (v - kv_mem) * beta
                    for (int vi = 0; vi < d_v; ++vi)
                        delta[vi] = (v_t[vi] - kv_mem[vi]) * beta_t;

                    // Step 4: S += outer(k, delta)
                    for (int j = 0; j < d_k; ++j)
                    {
                        const float k_j = k_t[j];
                        for (int vi = 0; vi < d_v; ++vi)
                            S[j * d_v + vi] += k_j * delta[vi];
                    }

                    // Step 5: output = S^T * q (contract over d_k)
                    for (int j = 0; j < d_k; ++j)
                    {
                        const float q_j = q_t[j];
                        for (int vi = 0; vi < d_v; ++vi)
                            o_t[vi] += S[j * d_v + vi] * q_j;
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

} // namespace llaminar2
