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

#if defined(__AVX512F__)
#include <immintrin.h>
#endif
#include <vector>

namespace llaminar2
{

    // =========================================================================
    // Scratch buffer management (grow-only, eliminates per-call allocations)
    // =========================================================================

    void CPUGatedDeltaNet::ensureScratch(int seq_len, int n_heads, int d_k, int /*d_v*/)
    {
        const size_t qk_total = static_cast<size_t>(seq_len) * n_heads * d_k;
        const size_t gate_total = static_cast<size_t>(seq_len) * n_heads;

        if (q_scratch_.size() < qk_total)
            q_scratch_.resize(qk_total);
        if (k_scratch_.size() < qk_total)
            k_scratch_.resize(qk_total);
        if (gate_scratch_.size() < gate_total)
            gate_scratch_.resize(gate_total);
        if (beta_sig_scratch_.size() < gate_total)
            beta_sig_scratch_.resize(gate_total);
    }

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

#if defined(__AVX512F__)
                const int hd_vec = head_dim & ~15;
                __m512 vsum = _mm512_setzero_ps();
                int d = 0;
                for (; d < hd_vec; d += 16)
                {
                    __m512 vv = _mm512_loadu_ps(vec + d);
                    vsum = _mm512_fmadd_ps(vv, vv, vsum);
                }
                float norm_sq = _mm512_reduce_add_ps(vsum);
                for (; d < head_dim; ++d)
                    norm_sq += vec[d] * vec[d];

                const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), eps);
                const __m512 vinv = _mm512_set1_ps(inv_norm);
                d = 0;
                for (; d < hd_vec; d += 16)
                    _mm512_storeu_ps(vec + d, _mm512_mul_ps(_mm512_loadu_ps(vec + d), vinv));
                for (; d < head_dim; ++d)
                    vec[d] *= inv_norm;
#else
                float norm_sq = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    norm_sq += vec[d] * vec[d];
                const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), eps);
                for (int d = 0; d < head_dim; ++d)
                    vec[d] *= inv_norm;
#endif
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
        // --- Preprocessing: reuse class scratch buffers ---
        ensureScratch(1, n_heads, d_k, d_v);

        const size_t qk_total = static_cast<size_t>(n_heads) * d_k;
        std::memcpy(q_scratch_.data(), q, qk_total * sizeof(float));
        std::memcpy(k_scratch_.data(), k, qk_total * sizeof(float));

        if (use_qk_l2norm)
        {
            l2normalize(q_scratch_.data(), 1, n_heads, d_k);
            l2normalize(k_scratch_.data(), 1, n_heads, d_k);
        }

        const float scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (size_t i = 0; i < qk_total; ++i)
            q_scratch_[i] *= scale;

        computeGates(alpha, beta_raw, A_log, dt_bias,
                     gate_scratch_.data(), beta_sig_scratch_.data(), 1, n_heads);

        // --- Core recurrence (stack-allocated per-head scratch) ---
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                float *S = state + static_cast<size_t>(h) * d_k * d_v;
                const float *q_h = q_scratch_.data() + h * d_k;
                const float *k_h = k_scratch_.data() + h * d_k;
                const float *v_h = v + h * d_v;
                const float g_h = gate_scratch_[h];
                const float beta_h = beta_sig_scratch_[h];
                float *o_h = output + h * d_v;

                const float decay = std::exp(g_h);

#if defined(__AVX512F__)
                // AVX-512 vectorized path — inner loops stride d_v,
                // process 16 floats per iteration
                const int d_v_vec = d_v & ~15; // d_v rounded down to multiple of 16

                // Step 1: Decay state — S[ij] *= decay
                {
                    const __m512 vdecay = _mm512_set1_ps(decay);
                    const int total = d_k * d_v;
                    const int total_vec = total & ~15;
                    int ij = 0;
                    for (; ij < total_vec; ij += 16)
                    {
                        __m512 s = _mm512_loadu_ps(S + ij);
                        _mm512_storeu_ps(S + ij, _mm512_mul_ps(s, vdecay));
                    }
                    for (; ij < total; ++ij)
                        S[ij] *= decay;
                }

                // Step 2: kv_mem = S^T * k  (contract over d_k)
                alignas(64) float kv_mem[512];
                {
                    // Zero kv_mem accumulators
                    int vi = 0;
                    for (; vi < d_v_vec; vi += 16)
                        _mm512_store_ps(kv_mem + vi, _mm512_setzero_ps());
                    for (; vi < d_v; ++vi)
                        kv_mem[vi] = 0.0f;

                    for (int j = 0; j < d_k; ++j)
                    {
                        const __m512 vk = _mm512_set1_ps(k_h[j]);
                        const float *S_row = S + j * d_v;
                        vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                        {
                            __m512 acc = _mm512_load_ps(kv_mem + vi);
                            __m512 sv = _mm512_loadu_ps(S_row + vi);
                            _mm512_store_ps(kv_mem + vi, _mm512_fmadd_ps(sv, vk, acc));
                        }
                        for (; vi < d_v; ++vi)
                            kv_mem[vi] += S_row[vi] * k_h[j];
                    }
                }

                // Step 3: delta = (v - kv_mem) * beta
                alignas(64) float delta[512];
                {
                    const __m512 vbeta = _mm512_set1_ps(beta_h);
                    int vi = 0;
                    for (; vi < d_v_vec; vi += 16)
                    {
                        __m512 vv = _mm512_loadu_ps(v_h + vi);
                        __m512 vkv = _mm512_load_ps(kv_mem + vi);
                        _mm512_store_ps(delta + vi, _mm512_mul_ps(_mm512_sub_ps(vv, vkv), vbeta));
                    }
                    for (; vi < d_v; ++vi)
                        delta[vi] = (v_h[vi] - kv_mem[vi]) * beta_h;
                }

                // Step 4: S += outer(k, delta)
                for (int j = 0; j < d_k; ++j)
                {
                    const __m512 vk = _mm512_set1_ps(k_h[j]);
                    float *S_row = S + j * d_v;
                    int vi = 0;
                    for (; vi < d_v_vec; vi += 16)
                    {
                        __m512 sv = _mm512_loadu_ps(S_row + vi);
                        __m512 vd = _mm512_load_ps(delta + vi);
                        _mm512_storeu_ps(S_row + vi, _mm512_fmadd_ps(vk, vd, sv));
                    }
                    for (; vi < d_v; ++vi)
                        S_row[vi] += k_h[j] * delta[vi];
                }

                // Step 5: output = S^T * q  (contract over d_k)
                {
                    int vi = 0;
                    for (; vi < d_v_vec; vi += 16)
                        _mm512_storeu_ps(o_h + vi, _mm512_setzero_ps());
                    for (; vi < d_v; ++vi)
                        o_h[vi] = 0.0f;

                    for (int j = 0; j < d_k; ++j)
                    {
                        const __m512 vq = _mm512_set1_ps(q_h[j]);
                        const float *S_row = S + j * d_v;
                        vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                        {
                            __m512 acc = _mm512_loadu_ps(o_h + vi);
                            __m512 sv = _mm512_loadu_ps(S_row + vi);
                            _mm512_storeu_ps(o_h + vi, _mm512_fmadd_ps(sv, vq, acc));
                        }
                        for (; vi < d_v; ++vi)
                            o_h[vi] += S_row[vi] * q_h[j];
                    }
                }

#else
                // Scalar fallback

                // Step 1: Decay state
                for (int ij = 0; ij < d_k * d_v; ++ij)
                    S[ij] *= decay;

                // Step 2: kv_mem = S^T * k
                alignas(64) float kv_mem[512];
                std::memset(kv_mem, 0, d_v * sizeof(float));
                for (int j = 0; j < d_k; ++j)
                {
                    const float k_j = k_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        kv_mem[vi] += S[j * d_v + vi] * k_j;
                }

                // Step 3: delta = (v - kv_mem) * beta
                alignas(64) float delta[512];
                for (int vi = 0; vi < d_v; ++vi)
                    delta[vi] = (v_h[vi] - kv_mem[vi]) * beta_h;

                // Step 4: S += outer(k, delta)
                for (int j = 0; j < d_k; ++j)
                {
                    const float k_j = k_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        S[j * d_v + vi] += k_j * delta[vi];
                }

                // Step 5: output = S^T * q
                std::memset(o_h, 0, d_v * sizeof(float));
                for (int j = 0; j < d_k; ++j)
                {
                    const float q_j = q_h[j];
                    for (int vi = 0; vi < d_v; ++vi)
                        o_h[vi] += S[j * d_v + vi] * q_j;
                }
#endif
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
        // --- Preprocessing: reuse class scratch buffers ---
        ensureScratch(seq_len, n_heads, d_k, d_v);

        const size_t qk_total = static_cast<size_t>(seq_len) * n_heads * d_k;
        std::memcpy(q_scratch_.data(), Q, qk_total * sizeof(float));
        std::memcpy(k_scratch_.data(), K, qk_total * sizeof(float));

        if (use_qk_l2norm)
        {
            l2normalize(q_scratch_.data(), seq_len, n_heads, d_k);
            l2normalize(k_scratch_.data(), seq_len, n_heads, d_k);
        }

        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        for (size_t i = 0; i < qk_total; ++i)
            q_scratch_[i] *= scale_val;

        const size_t gate_size = static_cast<size_t>(seq_len) * n_heads;
        computeGates(alpha, beta_raw, A_log, dt_bias,
                     gate_scratch_.data(), beta_sig_scratch_.data(), seq_len, n_heads);

        // --- Core recurrence (stack-allocated per-head scratch) ---
        std::memset(output, 0, static_cast<size_t>(seq_len) * n_heads * d_v * sizeof(float));

        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                float *S = state + static_cast<size_t>(h) * d_k * d_v;

                alignas(64) float kv_mem[512]; // d_v <= 512
                alignas(64) float delta[512];  // d_v <= 512

                for (int t = 0; t < seq_len; ++t)
                {
                    const float *q_t = q_scratch_.data() + t * n_heads * d_k + h * d_k;
                    const float *k_t = k_scratch_.data() + t * n_heads * d_k + h * d_k;
                    const float *v_t = V + t * n_heads * d_v + h * d_v;
                    const float g_t = gate_scratch_[t * n_heads + h];
                    const float beta_t = beta_sig_scratch_[t * n_heads + h];
                    float *o_t = output + t * n_heads * d_v + h * d_v;

                    const float decay_val = std::exp(g_t);

#if defined(__AVX512F__)
                    const int d_v_vec = d_v & ~15;

                    // Step 1: Decay state
                    {
                        const __m512 vdecay = _mm512_set1_ps(decay_val);
                        const int total = d_k * d_v;
                        const int total_vec = total & ~15;
                        int ij = 0;
                        for (; ij < total_vec; ij += 16)
                        {
                            __m512 s = _mm512_loadu_ps(S + ij);
                            _mm512_storeu_ps(S + ij, _mm512_mul_ps(s, vdecay));
                        }
                        for (; ij < total; ++ij)
                            S[ij] *= decay_val;
                    }

                    // Step 2: kv_mem = S^T * k
                    {
                        int vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                            _mm512_store_ps(kv_mem + vi, _mm512_setzero_ps());
                        for (; vi < d_v; ++vi)
                            kv_mem[vi] = 0.0f;

                        for (int j = 0; j < d_k; ++j)
                        {
                            const __m512 vk = _mm512_set1_ps(k_t[j]);
                            const float *S_row = S + j * d_v;
                            vi = 0;
                            for (; vi < d_v_vec; vi += 16)
                            {
                                __m512 acc = _mm512_load_ps(kv_mem + vi);
                                __m512 sv = _mm512_loadu_ps(S_row + vi);
                                _mm512_store_ps(kv_mem + vi, _mm512_fmadd_ps(sv, vk, acc));
                            }
                            for (; vi < d_v; ++vi)
                                kv_mem[vi] += S_row[vi] * k_t[j];
                        }
                    }

                    // Step 3: delta = (v - kv_mem) * beta
                    {
                        const __m512 vbeta = _mm512_set1_ps(beta_t);
                        int vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                        {
                            __m512 vv = _mm512_loadu_ps(v_t + vi);
                            __m512 vkv = _mm512_load_ps(kv_mem + vi);
                            _mm512_store_ps(delta + vi, _mm512_mul_ps(_mm512_sub_ps(vv, vkv), vbeta));
                        }
                        for (; vi < d_v; ++vi)
                            delta[vi] = (v_t[vi] - kv_mem[vi]) * beta_t;
                    }

                    // Step 4: S += outer(k, delta)
                    for (int j = 0; j < d_k; ++j)
                    {
                        const __m512 vk = _mm512_set1_ps(k_t[j]);
                        float *S_row = S + j * d_v;
                        int vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                        {
                            __m512 sv = _mm512_loadu_ps(S_row + vi);
                            __m512 vd = _mm512_load_ps(delta + vi);
                            _mm512_storeu_ps(S_row + vi, _mm512_fmadd_ps(vk, vd, sv));
                        }
                        for (; vi < d_v; ++vi)
                            S_row[vi] += k_t[j] * delta[vi];
                    }

                    // Step 5: output = S^T * q
                    for (int j = 0; j < d_k; ++j)
                    {
                        const __m512 vq = _mm512_set1_ps(q_t[j]);
                        const float *S_row = S + j * d_v;
                        int vi = 0;
                        for (; vi < d_v_vec; vi += 16)
                        {
                            __m512 acc = _mm512_loadu_ps(o_t + vi);
                            __m512 sv = _mm512_loadu_ps(S_row + vi);
                            _mm512_storeu_ps(o_t + vi, _mm512_fmadd_ps(sv, vq, acc));
                        }
                        for (; vi < d_v; ++vi)
                            o_t[vi] += S_row[vi] * q_t[j];
                    }
#else
                    // Scalar fallback

                    // Step 1: Decay state
                    for (int ij = 0; ij < d_k * d_v; ++ij)
                        S[ij] *= decay_val;

                    // Step 2: kv_mem = S^T * k
                    std::memset(kv_mem, 0, d_v * sizeof(float));
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

                    // Step 5: output = S^T * q
                    for (int j = 0; j < d_k; ++j)
                    {
                        const float q_j = q_t[j];
                        for (int vi = 0; vi < d_v; ++vi)
                            o_t[vi] += S[j * d_v + vi] * q_j;
                    }
#endif
                }
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        return true;
    }

} // namespace llaminar2
