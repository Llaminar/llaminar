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
#include "../../../utils/CPUFeatures.h"
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
    //
    // Optimizations over naive per-timestep recurrence:
    //   1. Preprocessing (copy+normalize+scale Q/K, gates, output zero) is
    //      parallelised across tokens — all cores contribute, not just 1.
    //   2. Steps 1+2 (decay S, kv_mem = S^T*k) fused into a single pass
    //      over S rows, eliminating one full read of the state matrix.
    //   3. Steps 4+5 (S += k⊗δ, output = S^T*q) fused similarly.
    //   4. Software prefetch: next S-row prefetched into the cache level
    //      where S resides (L1 if S fits, L2 otherwise), based on runtime
    //      cache detection via CPUFeatures.h.
    //
    // Why NO d_v tiling: S[d_k, d_v] with d_k=d_v=128 is 64 KB — too large
    // for 32 KB L1D, but fits easily in 1 MB L2.  Tiling d_v forces Q/K
    // data to be re-read once per tile, which at 296 KB per head × n_tiles
    // overflows L2 and destroys performance (~30 % regression measured).
    // Without tiling, Q/K is read once and the sequential row-access pattern
    // through S is handled efficiently by the hardware prefetcher.
    // =========================================================================

// Prefetch to L1 or L2 depending on runtime bool (GCC 14 requires
// compile-time constant for _mm_prefetch hint parameter).
#if defined(__AVX512F__)
#define GDN_PREFETCH_S(addr, to_l1)                          \
    do                                                       \
    {                                                        \
        if (to_l1)                                           \
            _mm_prefetch((const char *)(addr), _MM_HINT_T0); \
        else                                                 \
            _mm_prefetch((const char *)(addr), _MM_HINT_T1); \
    } while (0)
#endif

    bool CPUGatedDeltaNet::chunk_forward(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int /*chunk_size*/, bool use_qk_l2norm)
    {
        ensureScratch(seq_len, n_heads, d_k, d_v);

        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        constexpr float l2_eps = 1e-6f;

        // Stride between consecutive timesteps in [seq_len, n_heads * dim] layout
        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;

        // Prefetch hint: if S per head fits in L1, prefetch to L1 (T0);
        // otherwise to L2 (T1) so it doesn't evict working-set scratch.
        const auto &ci = cache_info();
        const size_t S_head_bytes = static_cast<size_t>(d_k) * d_v * sizeof(float);
        const bool pf_to_l1 = ci.fits_l1(S_head_bytes);
        // How many S-rows ahead to prefetch, scaled by cache line count per row.
        // Each row = d_v×4 bytes; at d_v=128 → 512 B = 8 cache lines.
        // Prefetch 2 rows ahead gives the hw ~16 cache-line fetches of runway.
        const int pf_rows_ahead = std::max(1, std::min(4,
                                                       static_cast<int>(ci.l2_size / (4u * S_head_bytes))));

        auto do_work = [&]()
        {
        // ── Phase 1: Parallel preprocessing (across tokens) ──────────
#pragma omp for schedule(static)
            for (int t = 0; t < seq_len; ++t)
            {
                for (int h = 0; h < n_heads; ++h)
                {
                    const int qk_off = t * qk_stride + h * d_k;
                    const int out_off = t * v_stride + h * d_v;
                    float *q_dst = q_scratch_.data() + qk_off;
                    float *k_dst = k_scratch_.data() + qk_off;
                    const float *q_src = Q + qk_off;
                    const float *k_src = K + qk_off;

                    if (use_qk_l2norm)
                    {
#if defined(__AVX512F__)
                        const int hd_vec = d_k & ~15;
                        // Q: fused L2-normalize + scale
                        {
                            __m512 vsum = _mm512_setzero_ps();
                            int d = 0;
                            for (; d < hd_vec; d += 16)
                            {
                                __m512 vv = _mm512_loadu_ps(q_src + d);
                                vsum = _mm512_fmadd_ps(vv, vv, vsum);
                            }
                            float norm_sq = _mm512_reduce_add_ps(vsum);
                            for (; d < d_k; ++d)
                                norm_sq += q_src[d] * q_src[d];
                            const float inv = scale_val / std::max(std::sqrt(norm_sq), l2_eps);
                            const __m512 vinv = _mm512_set1_ps(inv);
                            d = 0;
                            for (; d < hd_vec; d += 16)
                                _mm512_storeu_ps(q_dst + d,
                                                 _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vinv));
                            for (; d < d_k; ++d)
                                q_dst[d] = q_src[d] * inv;
                        }
                        // K: L2-normalize only
                        {
                            __m512 vsum = _mm512_setzero_ps();
                            int d = 0;
                            for (; d < hd_vec; d += 16)
                            {
                                __m512 vv = _mm512_loadu_ps(k_src + d);
                                vsum = _mm512_fmadd_ps(vv, vv, vsum);
                            }
                            float norm_sq = _mm512_reduce_add_ps(vsum);
                            for (; d < d_k; ++d)
                                norm_sq += k_src[d] * k_src[d];
                            const float inv = 1.0f / std::max(std::sqrt(norm_sq), l2_eps);
                            const __m512 vinv = _mm512_set1_ps(inv);
                            d = 0;
                            for (; d < hd_vec; d += 16)
                                _mm512_storeu_ps(k_dst + d,
                                                 _mm512_mul_ps(_mm512_loadu_ps(k_src + d), vinv));
                            for (; d < d_k; ++d)
                                k_dst[d] = k_src[d] * inv;
                        }
#else
                        {
                            float nq = 0.0f;
                            for (int d = 0; d < d_k; ++d)
                                nq += q_src[d] * q_src[d];
                            const float inv_q = scale_val / std::max(std::sqrt(nq), l2_eps);
                            for (int d = 0; d < d_k; ++d)
                                q_dst[d] = q_src[d] * inv_q;
                        }
                        {
                            float nk = 0.0f;
                            for (int d = 0; d < d_k; ++d)
                                nk += k_src[d] * k_src[d];
                            const float inv_k = 1.0f / std::max(std::sqrt(nk), l2_eps);
                            for (int d = 0; d < d_k; ++d)
                                k_dst[d] = k_src[d] * inv_k;
                        }
#endif
                    }
                    else
                    {
#if defined(__AVX512F__)
                        const int hd_vec = d_k & ~15;
                        const __m512 vscale = _mm512_set1_ps(scale_val);
                        int d = 0;
                        for (; d < hd_vec; d += 16)
                            _mm512_storeu_ps(q_dst + d,
                                             _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vscale));
                        for (; d < d_k; ++d)
                            q_dst[d] = q_src[d] * scale_val;
#else
                        for (int d = 0; d < d_k; ++d)
                            q_dst[d] = q_src[d] * scale_val;
#endif
                        std::memcpy(k_dst, k_src, d_k * sizeof(float));
                    }

                    // Gates — precompute decay = exp(g) here in the parallel
                    // phase so Phase 2 avoids 595×16 serial exp() calls.
                    const int gi = t * n_heads + h;
                    const float x = alpha[gi] + dt_bias[h];
                    const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                    gate_scratch_[gi] = std::exp(A_log[h] * sp);
                    beta_sig_scratch_[gi] = 1.0f / (1.0f + std::exp(-beta_raw[gi]));

                    // Zero output
                    std::memset(output + out_off, 0, d_v * sizeof(float));
                }
            }
            // implicit barrier between omp-for regions

            // ── Phase 2: Fused recurrence (across heads) ─────────────────
            // No d_v tiling — process all d_v columns at once so Q/K data
            // is only read once per head.  S[d_k, d_v] lives in L2; the
            // sequential row-stride access pattern is hw-prefetcher friendly.
            //
            // For d_v=128, the inner vi loop is fully unrolled with kv_mem,
            // delta, and output kept in ZMM registers.  The generic loop
            // loads/stores these from stack every j-iteration, causing ~47%
            // L1 miss rate (perf measured).  Register-resident eliminates
            // those accesses entirely.
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                float *S = state + static_cast<size_t>(h) * d_k * d_v;

#if defined(__AVX512F__)
                if (d_v == 128)
                {
                    // ══════════════════════════════════════════════════════
                    // Specialized path: d_v=128 (8 ZMMs), register-resident
                    // accumulators.  Eliminates ~5000 L1 load/stores per
                    // timestep vs the generic loop.
                    // ══════════════════════════════════════════════════════

                    for (int t = 0; t < seq_len; ++t)
                    {
                        const float *q_t = q_scratch_.data() + t * qk_stride + h * d_k;
                        const float *k_t = k_scratch_.data() + t * qk_stride + h * d_k;
                        const float *v_t = V + t * v_stride + h * d_v;
                        const float decay_val = gate_scratch_[t * n_heads + h];
                        const float beta_t = beta_sig_scratch_[t * n_heads + h];
                        float *o_t = output + t * v_stride + h * d_v;

                        // ── Fused step 1+2: Decay S + kv_mem = S^T * k ──
                        // kv_mem in 8 ZMM registers (not stack)
                        __m512 kv0 = _mm512_setzero_ps();
                        __m512 kv1 = _mm512_setzero_ps();
                        __m512 kv2 = _mm512_setzero_ps();
                        __m512 kv3 = _mm512_setzero_ps();
                        __m512 kv4 = _mm512_setzero_ps();
                        __m512 kv5 = _mm512_setzero_ps();
                        __m512 kv6 = _mm512_setzero_ps();
                        __m512 kv7 = _mm512_setzero_ps();

                        const __m512 vdecay = _mm512_set1_ps(decay_val);

                        for (int j = 0; j < d_k; ++j)
                        {
                            if (j + pf_rows_ahead < d_k)
                                GDN_PREFETCH_S(S + (j + pf_rows_ahead) * 128, pf_to_l1);

                            const __m512 vk = _mm512_set1_ps(k_t[j]);
                            float *S_row = S + j * 128;

                            __m512 s0 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 0), vdecay);
                            __m512 s1 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 16), vdecay);
                            __m512 s2 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 32), vdecay);
                            __m512 s3 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 48), vdecay);
                            __m512 s4 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 64), vdecay);
                            __m512 s5 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 80), vdecay);
                            __m512 s6 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 96), vdecay);
                            __m512 s7 = _mm512_mul_ps(_mm512_loadu_ps(S_row + 112), vdecay);

                            _mm512_storeu_ps(S_row + 0, s0);
                            _mm512_storeu_ps(S_row + 16, s1);
                            _mm512_storeu_ps(S_row + 32, s2);
                            _mm512_storeu_ps(S_row + 48, s3);
                            _mm512_storeu_ps(S_row + 64, s4);
                            _mm512_storeu_ps(S_row + 80, s5);
                            _mm512_storeu_ps(S_row + 96, s6);
                            _mm512_storeu_ps(S_row + 112, s7);

                            kv0 = _mm512_fmadd_ps(s0, vk, kv0);
                            kv1 = _mm512_fmadd_ps(s1, vk, kv1);
                            kv2 = _mm512_fmadd_ps(s2, vk, kv2);
                            kv3 = _mm512_fmadd_ps(s3, vk, kv3);
                            kv4 = _mm512_fmadd_ps(s4, vk, kv4);
                            kv5 = _mm512_fmadd_ps(s5, vk, kv5);
                            kv6 = _mm512_fmadd_ps(s6, vk, kv6);
                            kv7 = _mm512_fmadd_ps(s7, vk, kv7);
                        }

                        // ── Step 3: delta = (v - kv_mem) * beta ──
                        // delta stays in registers (reuse kv0-kv7)
                        const __m512 vbeta = _mm512_set1_ps(beta_t);
                        kv0 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 0), kv0), vbeta);
                        kv1 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 16), kv1), vbeta);
                        kv2 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 32), kv2), vbeta);
                        kv3 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 48), kv3), vbeta);
                        kv4 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 64), kv4), vbeta);
                        kv5 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 80), kv5), vbeta);
                        kv6 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 96), kv6), vbeta);
                        kv7 = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(v_t + 112), kv7), vbeta);
                        // kv0-kv7 now hold delta[0..127]

                        // ── Fused step 4+5: S += k⊗δ, output += S^T * q ──
                        // output in 8 ZMM registers
                        __m512 o0 = _mm512_setzero_ps();
                        __m512 o1 = _mm512_setzero_ps();
                        __m512 o2 = _mm512_setzero_ps();
                        __m512 o3 = _mm512_setzero_ps();
                        __m512 o4 = _mm512_setzero_ps();
                        __m512 o5 = _mm512_setzero_ps();
                        __m512 o6 = _mm512_setzero_ps();
                        __m512 o7 = _mm512_setzero_ps();

                        for (int j = 0; j < d_k; ++j)
                        {
                            if (j + pf_rows_ahead < d_k)
                                GDN_PREFETCH_S(S + (j + pf_rows_ahead) * 128, pf_to_l1);

                            const __m512 vk = _mm512_set1_ps(k_t[j]);
                            const __m512 vq = _mm512_set1_ps(q_t[j]);
                            float *S_row = S + j * 128;

                            // S update: s = S_row + k * delta  (all 8 segments)
                            __m512 s0 = _mm512_fmadd_ps(vk, kv0, _mm512_loadu_ps(S_row + 0));
                            __m512 s1 = _mm512_fmadd_ps(vk, kv1, _mm512_loadu_ps(S_row + 16));
                            __m512 s2 = _mm512_fmadd_ps(vk, kv2, _mm512_loadu_ps(S_row + 32));
                            __m512 s3 = _mm512_fmadd_ps(vk, kv3, _mm512_loadu_ps(S_row + 48));
                            __m512 s4 = _mm512_fmadd_ps(vk, kv4, _mm512_loadu_ps(S_row + 64));
                            __m512 s5 = _mm512_fmadd_ps(vk, kv5, _mm512_loadu_ps(S_row + 80));
                            __m512 s6 = _mm512_fmadd_ps(vk, kv6, _mm512_loadu_ps(S_row + 96));
                            __m512 s7 = _mm512_fmadd_ps(vk, kv7, _mm512_loadu_ps(S_row + 112));

                            _mm512_storeu_ps(S_row + 0, s0);
                            _mm512_storeu_ps(S_row + 16, s1);
                            _mm512_storeu_ps(S_row + 32, s2);
                            _mm512_storeu_ps(S_row + 48, s3);
                            _mm512_storeu_ps(S_row + 64, s4);
                            _mm512_storeu_ps(S_row + 80, s5);
                            _mm512_storeu_ps(S_row + 96, s6);
                            _mm512_storeu_ps(S_row + 112, s7);

                            // output accumulation: o += S * q  (all 8 segments)
                            o0 = _mm512_fmadd_ps(s0, vq, o0);
                            o1 = _mm512_fmadd_ps(s1, vq, o1);
                            o2 = _mm512_fmadd_ps(s2, vq, o2);
                            o3 = _mm512_fmadd_ps(s3, vq, o3);
                            o4 = _mm512_fmadd_ps(s4, vq, o4);
                            o5 = _mm512_fmadd_ps(s5, vq, o5);
                            o6 = _mm512_fmadd_ps(s6, vq, o6);
                            o7 = _mm512_fmadd_ps(s7, vq, o7);
                        }

                        // Store output (single write, no read-modify-write)
                        _mm512_storeu_ps(o_t + 0, o0);
                        _mm512_storeu_ps(o_t + 16, o1);
                        _mm512_storeu_ps(o_t + 32, o2);
                        _mm512_storeu_ps(o_t + 48, o3);
                        _mm512_storeu_ps(o_t + 64, o4);
                        _mm512_storeu_ps(o_t + 80, o5);
                        _mm512_storeu_ps(o_t + 96, o6);
                        _mm512_storeu_ps(o_t + 112, o7);

                    } // timesteps
                }
                else
#endif // __AVX512F__
                {
                    // ── Generic fallback (non-128 d_v or no AVX-512) ──────
                    alignas(64) float kv_mem[512]; // d_v <= 512
                    alignas(64) float delta[512];

                    for (int t = 0; t < seq_len; ++t)
                    {
                        const float *q_t = q_scratch_.data() + t * qk_stride + h * d_k;
                        const float *k_t = k_scratch_.data() + t * qk_stride + h * d_k;
                        const float *v_t = V + t * v_stride + h * d_v;
                        const float decay_val = gate_scratch_[t * n_heads + h];
                        const float beta_t = beta_sig_scratch_[t * n_heads + h];
                        float *o_t = output + t * v_stride + h * d_v;

                        // Fused step 1+2: Decay S + kv_mem = S^T * k
                        std::memset(kv_mem, 0, d_v * sizeof(float));
                        for (int j = 0; j < d_k; ++j)
                        {
                            float *S_row = S + j * d_v;
                            const float k_j = k_t[j];
                            for (int vi = 0; vi < d_v; ++vi)
                            {
                                S_row[vi] *= decay_val;
                                kv_mem[vi] += S_row[vi] * k_j;
                            }
                        }

                        // Step 3: delta = (v - kv_mem) * beta
                        for (int vi = 0; vi < d_v; ++vi)
                            delta[vi] = (v_t[vi] - kv_mem[vi]) * beta_t;

                        // Fused step 4+5: S += k⊗δ, output += S^T * q
                        std::memset(o_t, 0, d_v * sizeof(float));
                        for (int j = 0; j < d_k; ++j)
                        {
                            float *S_row = S + j * d_v;
                            const float k_j = k_t[j];
                            const float q_j = q_t[j];
                            for (int vi = 0; vi < d_v; ++vi)
                            {
                                S_row[vi] += k_j * delta[vi];
                                o_t[vi] += S_row[vi] * q_j;
                            }
                        }
                    } // timesteps
                }
            } // heads
        };
        OMP_WORKSHARE_REGION(do_work);

#if defined(__AVX512F__)
#undef GDN_PREFETCH_S
#endif

        return true;
    }

} // namespace llaminar2
