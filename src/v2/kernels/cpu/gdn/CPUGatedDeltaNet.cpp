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

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__AVX512F__)
// =========================================================================
// Fast AVX-512 exp/sigmoid approximations
//
// exp(x) via range reduction: exp(x) = 2^n · P(f)
//   n = round(x · log2(e)),  f = x·log2(e) - n ∈ [-0.5, 0.5]
//   P(f) = degree-5 Horner polynomial for 2^f
//   2^n via IEEE-754 exponent bit-shift (scalef)
//
// Same polynomial used in CPUShortConvolution and GatedRMSNormStage.
// =========================================================================

static inline __m512 avx512_fast_exp(__m512 vx)
{
    const __m512 vlog2e = _mm512_set1_ps(1.4426950408889634f);
    const __m512 vln2 = _mm512_set1_ps(0.6931471805599453f);
    const __m512 vc0 = _mm512_set1_ps(1.0f);
    const __m512 vc1 = _mm512_set1_ps(0.693147180559945f);
    const __m512 vc2 = _mm512_set1_ps(0.240226506959101f);
    const __m512 vc3 = _mm512_set1_ps(0.055504108664822f);
    const __m512 vc4 = _mm512_set1_ps(0.009618129107629f);
    const __m512 vc5 = _mm512_set1_ps(0.001333355814642f);

    __m512 vt = _mm512_mul_ps(vx, vlog2e);
    __m512 vn = _mm512_roundscale_ps(vt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m512 vf = _mm512_sub_ps(vt, vn);

    __m512 vpoly = _mm512_fmadd_ps(vc5, vf, vc4);
    vpoly = _mm512_fmadd_ps(vpoly, vf, vc3);
    vpoly = _mm512_fmadd_ps(vpoly, vf, vc2);
    vpoly = _mm512_fmadd_ps(vpoly, vf, vc1);
    vpoly = _mm512_fmadd_ps(vpoly, vf, vc0);

    return _mm512_scalef_ps(vpoly, vn);
}

static inline __m512 avx512_fast_sigmoid(__m512 vx)
{
    __m512 vneg = _mm512_sub_ps(_mm512_setzero_ps(), vx);
    __m512 vexp_neg = avx512_fast_exp(vneg);
    __m512 vone = _mm512_set1_ps(1.0f);
    return _mm512_div_ps(vone, _mm512_add_ps(vone, vexp_neg));
}
#endif

#if defined(__AVX2__)
// =========================================================================
// Fast AVX2 exp/sigmoid approximations (8-wide YMM)
// Same polynomial as AVX-512 version but processes 8 floats instead of 16.
// =========================================================================

static inline __m256 avx2_fast_exp(__m256 vx)
{
    const __m256 vlog2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 vc0 = _mm256_set1_ps(1.0f);
    const __m256 vc1 = _mm256_set1_ps(0.693147180559945f);
    const __m256 vc2 = _mm256_set1_ps(0.240226506959101f);
    const __m256 vc3 = _mm256_set1_ps(0.055504108664822f);
    const __m256 vc4 = _mm256_set1_ps(0.009618129107629f);
    const __m256 vc5 = _mm256_set1_ps(0.001333355814642f);

    __m256 vt = _mm256_mul_ps(vx, vlog2e);
    __m256 vn = _mm256_round_ps(vt, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    __m256 vf = _mm256_sub_ps(vt, vn);

    __m256 vpoly = _mm256_fmadd_ps(vc5, vf, vc4);
    vpoly = _mm256_fmadd_ps(vpoly, vf, vc3);
    vpoly = _mm256_fmadd_ps(vpoly, vf, vc2);
    vpoly = _mm256_fmadd_ps(vpoly, vf, vc1);
    vpoly = _mm256_fmadd_ps(vpoly, vf, vc0);

    // Reconstruct 2^n via IEEE754 exponent field
    __m256i vi_n = _mm256_cvtps_epi32(vn);
    vi_n = _mm256_add_epi32(vi_n, _mm256_set1_epi32(127));
    __m256 v2n = _mm256_castsi256_ps(_mm256_slli_epi32(vi_n, 23));
    return _mm256_mul_ps(vpoly, v2n);
}

static inline __m256 avx2_fast_sigmoid(__m256 vx)
{
    __m256 vneg = _mm256_sub_ps(_mm256_setzero_ps(), vx);
    __m256 vexp_neg = avx2_fast_exp(vneg);
    __m256 vone = _mm256_set1_ps(1.0f);
    return _mm256_div_ps(vone, _mm256_add_ps(vone, vexp_neg));
}
#endif

#include <vector>

#include "../simd/AVX2Helpers.h"

namespace llaminar2
{
    void CPUGatedDeltaNet::bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
    {
        verifier_state_capture_ = workspace;
        verifier_state_capture_rows_ = rows;
        verifier_state_capture_size_ = state_size;
    }

    bool CPUGatedDeltaNet::restoreVerifierStateCaptureRow(float *dst_state, int row, void *stream)
    {
        if (row < 0 || row >= verifier_state_capture_rows_)
            return false;
        return restoreStateFromSnapshot(
            dst_state,
            verifier_state_capture_,
            row,
            verifier_state_capture_size_,
            verifier_state_capture_size_,
            stream);
    }

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

    // Named ISA implementations: l2normalize_vec (single vector)
    static void l2normalize_vec_scalar(float *vec, int head_dim, float eps)
    {
        float norm_sq = 0.0f;
        for (int d = 0; d < head_dim; ++d)
            norm_sq += vec[d] * vec[d];
        const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), eps);
        for (int d = 0; d < head_dim; ++d)
            vec[d] *= inv_norm;
    }

#if defined(__AVX2__)
    static void l2normalize_vec_avx2(float *vec, int head_dim, float eps)
    {
        const int hd_vec = head_dim & ~7;
        __m256 vsum = _mm256_setzero_ps();
        int d = 0;
        for (; d < hd_vec; d += 8)
        {
            __m256 vv = _mm256_loadu_ps(vec + d);
            vsum = _mm256_fmadd_ps(vv, vv, vsum);
        }
        __m128 hi = _mm256_extractf128_ps(vsum, 1);
        __m128 lo = _mm256_castps256_ps128(vsum);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float norm_sq = _mm_cvtss_f32(lo);
        for (; d < head_dim; ++d)
            norm_sq += vec[d] * vec[d];

        const float inv_norm = 1.0f / std::max(std::sqrt(norm_sq), eps);
        const __m256 vinv = _mm256_set1_ps(inv_norm);
        d = 0;
        for (; d < hd_vec; d += 8)
            _mm256_storeu_ps(vec + d, _mm256_mul_ps(_mm256_loadu_ps(vec + d), vinv));
        for (; d < head_dim; ++d)
            vec[d] *= inv_norm;
    }
#endif

#if defined(__AVX512F__)
    static void l2normalize_vec_avx512(float *vec, int head_dim, float eps)
    {
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
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void l2normalize_vec_avx2(float *vec, int head_dim, float eps)
    {
        l2normalize_vec_scalar(vec, head_dim, eps);
    }
#endif
#if !defined(__AVX512F__)
    static void l2normalize_vec_avx512(float *vec, int head_dim, float eps)
    {
        l2normalize_vec_avx2(vec, head_dim, eps);
    }
#endif

    static inline void l2normalize_vec(float *vec, int head_dim, float eps)
    {
        ISA_DISPATCH_VOID(l2normalize_vec, vec, head_dim, eps);
    }

    void CPUGatedDeltaNet::l2normalize(float *data, int seq_len, int n_heads, int head_dim)
    {
        constexpr float eps = 1e-6f;

        for (int t = 0; t < seq_len; ++t)
        {
            for (int h = 0; h < n_heads; ++h)
            {
                float *vec = data + t * n_heads * head_dim + h * head_dim;
                l2normalize_vec(vec, head_dim, eps);
            }
        }
    }

    // =========================================================================
    // Named ISA implementations: QK preprocessing helpers
    // (shared between recurrent_step and chunk_forward)
    // =========================================================================

    // L2-normalize Q with fused scale, L2-normalize K (no scale)
    static void gdn_preprocess_qk_l2norm_scalar(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        float nq = 0.0f;
        for (int d = 0; d < dim; ++d)
            nq += q_src[d] * q_src[d];
        const float inv_q = scale / std::max(std::sqrt(nq), eps);
        for (int d = 0; d < dim; ++d)
            q_dst[d] = q_src[d] * inv_q;

        float nk = 0.0f;
        for (int d = 0; d < dim; ++d)
            nk += k_src[d] * k_src[d];
        const float inv_k = 1.0f / std::max(std::sqrt(nk), eps);
        for (int d = 0; d < dim; ++d)
            k_dst[d] = k_src[d] * inv_k;
    }

#if defined(__AVX2__)
    static void gdn_preprocess_qk_l2norm_avx2(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        avx2::l2norm_scale(q_src, q_dst, dim, scale, eps);
        avx2::l2norm_scale(k_src, k_dst, dim, 1.0f, eps);
    }
#endif

#if defined(__AVX512F__)
    static void gdn_preprocess_qk_l2norm_avx512(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        const int hd_vec = dim & ~15;
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
            for (; d < dim; ++d)
                norm_sq += q_src[d] * q_src[d];
            const float inv = scale / std::max(std::sqrt(norm_sq), eps);
            const __m512 vinv = _mm512_set1_ps(inv);
            d = 0;
            for (; d < hd_vec; d += 16)
                _mm512_storeu_ps(q_dst + d,
                                 _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vinv));
            for (; d < dim; ++d)
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
            for (; d < dim; ++d)
                norm_sq += k_src[d] * k_src[d];
            const float inv = 1.0f / std::max(std::sqrt(norm_sq), eps);
            const __m512 vinv = _mm512_set1_ps(inv);
            d = 0;
            for (; d < hd_vec; d += 16)
                _mm512_storeu_ps(k_dst + d,
                                 _mm512_mul_ps(_mm512_loadu_ps(k_src + d), vinv));
            for (; d < dim; ++d)
                k_dst[d] = k_src[d] * inv;
        }
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_preprocess_qk_l2norm_avx2(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        gdn_preprocess_qk_l2norm_scalar(q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_preprocess_qk_l2norm_avx512(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        gdn_preprocess_qk_l2norm_avx2(q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }
#endif

    static inline void gdn_preprocess_qk_l2norm(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale, float eps)
    {
        ISA_DISPATCH_VOID(gdn_preprocess_qk_l2norm, q_src, k_src, q_dst, k_dst, dim, scale, eps);
    }

    // Scale Q, copy K unchanged
    static void gdn_preprocess_qk_scale_scalar(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        for (int d = 0; d < dim; ++d)
            q_dst[d] = q_src[d] * scale;
        std::memcpy(k_dst, k_src, dim * sizeof(float));
    }

#if defined(__AVX2__)
    static void gdn_preprocess_qk_scale_avx2(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        avx2::copy_scale(q_dst, q_src, scale, dim);
        // k_dst filled by caller via memcpy (passed through)
        (void)k_dst;
    }
#endif

#if defined(__AVX512F__)
    static void gdn_preprocess_qk_scale_avx512(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float * /*k_dst*/,
        int dim, float scale)
    {
        const int hd_vec = dim & ~15;
        const __m512 vscale = _mm512_set1_ps(scale);
        int d = 0;
        for (; d < hd_vec; d += 16)
            _mm512_storeu_ps(q_dst + d,
                             _mm512_mul_ps(_mm512_loadu_ps(q_src + d), vscale));
        for (; d < dim; ++d)
            q_dst[d] = q_src[d] * scale;
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_preprocess_qk_scale_avx2(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        (void)k_dst;
        gdn_preprocess_qk_scale_scalar(q_src, nullptr, q_dst, k_dst, dim, scale);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_preprocess_qk_scale_avx512(
        const float *q_src, const float * /*k_src*/,
        float *q_dst, float * /*k_dst*/,
        int dim, float scale)
    {
        gdn_preprocess_qk_scale_avx2(q_src, nullptr, q_dst, nullptr, dim, scale);
    }
#endif

    static inline void gdn_preprocess_qk_scale(
        const float *q_src, const float *k_src,
        float *q_dst, float *k_dst,
        int dim, float scale)
    {
        switch (activeISALevel())
        {
        case ISALevel::AVX512:
            gdn_preprocess_qk_scale_avx512(q_src, k_src, q_dst, k_dst, dim, scale);
            std::memcpy(k_dst, k_src, dim * sizeof(float));
            break;
        case ISALevel::AVX2:
            gdn_preprocess_qk_scale_avx2(q_src, k_src, q_dst, k_dst, dim, scale);
            std::memcpy(k_dst, k_src, dim * sizeof(float));
            break;
        default:
            gdn_preprocess_qk_scale_scalar(q_src, k_src, q_dst, k_dst, dim, scale);
            break;
        }
    }

    // =========================================================================
    // Named ISA implementations: core delta recurrence (5-step)
    // =========================================================================

    static void gdn_delta_recurrence_scalar(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        // Step 1: Decay state
        for (int ij = 0; ij < d_k * d_v; ++ij)
            S[ij] *= decay;

        // Step 2: kv_mem = S^T * k
        alignas(64) float kv_mem[512];
        std::memset(kv_mem, 0, d_v * sizeof(float));
        for (int j = 0; j < d_k; ++j)
        {
            const float k_j = k[j];
            for (int vi = 0; vi < d_v; ++vi)
                kv_mem[vi] += S[j * d_v + vi] * k_j;
        }

        // Step 3: delta = (v - kv_mem) * beta
        alignas(64) float delta[512];
        for (int vi = 0; vi < d_v; ++vi)
            delta[vi] = (v[vi] - kv_mem[vi]) * beta;

        // Step 4: S += outer(k, delta)
        for (int j = 0; j < d_k; ++j)
        {
            const float k_j = k[j];
            for (int vi = 0; vi < d_v; ++vi)
                S[j * d_v + vi] += k_j * delta[vi];
        }

        // Step 5: output = S^T * q
        std::memset(o, 0, d_v * sizeof(float));
        for (int j = 0; j < d_k; ++j)
        {
            const float q_j = q[j];
            for (int vi = 0; vi < d_v; ++vi)
                o[vi] += S[j * d_v + vi] * q_j;
        }
    }

#if defined(__AVX2__)
    static void gdn_delta_recurrence_avx2(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        avx2::scale(S, d_k * d_v, decay);

        alignas(64) float kv_mem[512];
        avx2::zero(kv_mem, d_v);
        for (int j = 0; j < d_k; ++j)
            avx2::axpy(kv_mem, S + j * d_v, k[j], d_v);

        alignas(64) float delta[512];
        avx2::sub_mul(delta, v, kv_mem, beta, d_v);

        for (int j = 0; j < d_k; ++j)
            avx2::axpy(S + j * d_v, delta, k[j], d_v);

        avx2::zero(o, d_v);
        for (int j = 0; j < d_k; ++j)
            avx2::axpy(o, S + j * d_v, q[j], d_v);
    }
#endif

#if defined(__AVX512F__)
    static void gdn_delta_recurrence_avx512(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        const int d_v_vec = d_v & ~15;

        // Step 1: Decay state
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

        // Step 2: kv_mem = S^T * k
        alignas(64) float kv_mem[512];
        {
            int vi = 0;
            for (; vi < d_v_vec; vi += 16)
                _mm512_store_ps(kv_mem + vi, _mm512_setzero_ps());
            for (; vi < d_v; ++vi)
                kv_mem[vi] = 0.0f;

            for (int j = 0; j < d_k; ++j)
            {
                const __m512 vk = _mm512_set1_ps(k[j]);
                const float *S_row = S + j * d_v;
                vi = 0;
                for (; vi < d_v_vec; vi += 16)
                {
                    __m512 acc = _mm512_load_ps(kv_mem + vi);
                    __m512 sv = _mm512_loadu_ps(S_row + vi);
                    _mm512_store_ps(kv_mem + vi, _mm512_fmadd_ps(sv, vk, acc));
                }
                for (; vi < d_v; ++vi)
                    kv_mem[vi] += S_row[vi] * k[j];
            }
        }

        // Step 3: delta = (v - kv_mem) * beta
        alignas(64) float delta[512];
        {
            const __m512 vbeta = _mm512_set1_ps(beta);
            int vi = 0;
            for (; vi < d_v_vec; vi += 16)
            {
                __m512 vv = _mm512_loadu_ps(v + vi);
                __m512 vkv = _mm512_load_ps(kv_mem + vi);
                _mm512_store_ps(delta + vi, _mm512_mul_ps(_mm512_sub_ps(vv, vkv), vbeta));
            }
            for (; vi < d_v; ++vi)
                delta[vi] = (v[vi] - kv_mem[vi]) * beta;
        }

        // Step 4: S += outer(k, delta)
        for (int j = 0; j < d_k; ++j)
        {
            const __m512 vk = _mm512_set1_ps(k[j]);
            float *S_row = S + j * d_v;
            int vi = 0;
            for (; vi < d_v_vec; vi += 16)
            {
                __m512 sv = _mm512_loadu_ps(S_row + vi);
                __m512 vd = _mm512_load_ps(delta + vi);
                _mm512_storeu_ps(S_row + vi, _mm512_fmadd_ps(vk, vd, sv));
            }
            for (; vi < d_v; ++vi)
                S_row[vi] += k[j] * delta[vi];
        }

        // Step 5: output = S^T * q
        {
            int vi = 0;
            for (; vi < d_v_vec; vi += 16)
                _mm512_storeu_ps(o + vi, _mm512_setzero_ps());
            for (; vi < d_v; ++vi)
                o[vi] = 0.0f;

            for (int j = 0; j < d_k; ++j)
            {
                const __m512 vq = _mm512_set1_ps(q[j]);
                const float *S_row = S + j * d_v;
                vi = 0;
                for (; vi < d_v_vec; vi += 16)
                {
                    __m512 acc = _mm512_loadu_ps(o + vi);
                    __m512 sv = _mm512_loadu_ps(S_row + vi);
                    _mm512_storeu_ps(o + vi, _mm512_fmadd_ps(sv, vq, acc));
                }
                for (; vi < d_v; ++vi)
                    o[vi] += S_row[vi] * q[j];
            }
        }
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_delta_recurrence_avx2(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        gdn_delta_recurrence_scalar(S, q, k, v, o, decay, beta, d_k, d_v);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_delta_recurrence_avx512(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        gdn_delta_recurrence_avx2(S, q, k, v, o, decay, beta, d_k, d_v);
    }
#endif

    static inline void gdn_delta_recurrence(
        float *S, const float *q, const float *k, const float *v,
        float *o, float decay, float beta, int d_k, int d_v)
    {
        ISA_DISPATCH_VOID(gdn_delta_recurrence, S, q, k, v, o, decay, beta, d_k, d_v);
    }

    // =========================================================================
    // Named ISA implementations: batch exp + sigmoid
    // =========================================================================

    static void gdn_batch_exp_sigmoid_scalar(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        for (int hh = 0; hh < n_heads; ++hh)
        {
            gate[base + hh] = std::exp(gate[base + hh]);
            beta_sig[base + hh] = 1.0f / (1.0f + std::exp(-beta_raw[base + hh]));
        }
    }

#if defined(__AVX2__)
    static void gdn_batch_exp_sigmoid_avx2(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        int hh = 0;
        const int hh_vec = n_heads & ~7;
        for (; hh < hh_vec; hh += 8)
        {
            __m256 vg = _mm256_loadu_ps(gate + base + hh);
            _mm256_storeu_ps(gate + base + hh, avx2::fast_exp(vg));
            __m256 vb = _mm256_loadu_ps(beta_raw + base + hh);
            _mm256_storeu_ps(beta_sig + base + hh, avx2::fast_sigmoid(vb));
        }
        for (; hh < n_heads; ++hh)
        {
            gate[base + hh] = std::exp(gate[base + hh]);
            beta_sig[base + hh] = 1.0f / (1.0f + std::exp(-beta_raw[base + hh]));
        }
    }
#endif

#if defined(__AVX512F__)
    static void gdn_batch_exp_sigmoid_avx512(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        int hh = 0;
        const int hh_vec = n_heads & ~15;
        for (; hh < hh_vec; hh += 16)
        {
            __m512 vg = _mm512_loadu_ps(gate + base + hh);
            _mm512_storeu_ps(gate + base + hh, avx512_fast_exp(vg));
            __m512 vb = _mm512_loadu_ps(beta_raw + base + hh);
            _mm512_storeu_ps(beta_sig + base + hh, avx512_fast_sigmoid(vb));
        }
        for (; hh < n_heads; ++hh)
        {
            gate[base + hh] = std::exp(gate[base + hh]);
            beta_sig[base + hh] = 1.0f / (1.0f + std::exp(-beta_raw[base + hh]));
        }
    }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
    static void gdn_batch_exp_sigmoid_avx2(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        gdn_batch_exp_sigmoid_scalar(gate, beta_raw, beta_sig, base, n_heads);
    }
#endif
#if !defined(__AVX512F__)
    static void gdn_batch_exp_sigmoid_avx512(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        gdn_batch_exp_sigmoid_avx2(gate, beta_raw, beta_sig, base, n_heads);
    }
#endif

    static inline void gdn_batch_exp_sigmoid(
        float *gate, const float *beta_raw, float *beta_sig,
        int base, int n_heads)
    {
        ISA_DISPATCH_VOID(gdn_batch_exp_sigmoid, gate, beta_raw, beta_sig, base, n_heads);
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
        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        constexpr float l2_eps = 1e-6f;

        // Fold ALL preprocessing into the per-head parallel loop.
        // Each thread uses stack-local Q/K — no shared scratch, no sequential
        // bottleneck, no OMP barrier between preprocessing and recurrence.
        auto do_work = [&]()
        {
#pragma omp for schedule(static)
            for (int h = 0; h < n_heads; ++h)
            {
                // ── Per-head preprocessing (stack-local buffers) ──
                alignas(64) float q_local[512]; // d_k <= 512
                alignas(64) float k_local[512];

                const float *q_src = q + h * d_k;
                const float *k_src = k + h * d_k;

                if (use_qk_l2norm)
                {
                    gdn_preprocess_qk_l2norm(q_src, k_src, q_local, k_local, d_k, scale_val, l2_eps);
                }
                else
                {
                    gdn_preprocess_qk_scale(q_src, k_src, q_local, k_local, d_k, scale_val);
                }

                // ── Gate + beta (combined, saves one exp() call) ──
                const float x = alpha[h] + dt_bias[h];
                const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                const float decay = std::exp(A_log[h] * sp);
                const float beta_h = 1.0f / (1.0f + std::exp(-beta_raw[h]));

                // ── Core recurrence ──
                float *S = state + static_cast<size_t>(h) * d_k * d_v;
                const float *q_h = q_local;
                const float *k_h = k_local;
                const float *v_h = v + h * d_v;
                float *o_h = output + h * d_v;

                gdn_delta_recurrence(S, q_h, k_h, v_h, o_h, decay, beta_h, d_k, d_v);
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
#if defined(__AVX512F__) || defined(__AVX2__)
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
        int chunk_size, bool use_qk_l2norm)
    {
        const int state_floats = n_heads * d_k * d_v;
        float *state_snapshots = nullptr;
        int snapshot_stride_floats = 0;
        int max_snapshot_rows = 0;
        if (verifier_state_capture_ &&
            verifier_state_capture_rows_ > 0 &&
            verifier_state_capture_size_ >= state_floats)
        {
            state_snapshots = verifier_state_capture_;
            snapshot_stride_floats = verifier_state_capture_size_;
            max_snapshot_rows = verifier_state_capture_rows_;
        }
        return chunkForwardImpl(
            Q, K, V, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows);
    }

    bool CPUGatedDeltaNet::chunkForwardWithStateSnapshots(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int chunk_size, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        return chunkForwardImpl(
            Q, K, V, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows);
    }

    bool CPUGatedDeltaNet::restoreStateFromSnapshot(
        float *state, const float *state_snapshots,
        int snapshot_row, int snapshot_stride_floats,
        int state_floats, void *stream)
    {
        (void)stream;
        if (!state || !state_snapshots || snapshot_row < 0 ||
            snapshot_stride_floats < state_floats || state_floats < 0)
            return false;

        std::memcpy(state,
                    state_snapshots + static_cast<size_t>(snapshot_row) * snapshot_stride_floats,
                    static_cast<size_t>(state_floats) * sizeof(float));
        return true;
    }

    bool CPUGatedDeltaNet::chunkForwardImpl(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int /*chunk_size*/, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        const int state_floats = n_heads * d_k * d_v;
        const bool capture_state_snapshots = state_snapshots != nullptr;
        if (capture_state_snapshots &&
            (snapshot_stride_floats < state_floats || max_snapshot_rows <= 0))
        {
            return false;
        }

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
                        gdn_preprocess_qk_l2norm(q_src, k_src, q_dst, k_dst, d_k, scale_val, l2_eps);
                    }
                    else
                    {
                        gdn_preprocess_qk_scale(q_src, k_src, q_dst, k_dst, d_k, scale_val);
                    }

                    // Gates — precompute decay = exp(g) here in the parallel
                    // phase so Phase 2 avoids 595×16 serial exp() calls.
                    const int gi = t * n_heads + h;
                    const float x = alpha[gi] + dt_bias[h];
                    const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                    gate_scratch_[gi] = A_log[h] * sp; // store g, NOT exp(g) yet

                    // Zero output
                    std::memset(output + out_off, 0, d_v * sizeof(float));
                }

                // Batch exp(g) and sigmoid across all heads for this token.
                gdn_batch_exp_sigmoid(gate_scratch_.data(), beta_raw, beta_sig_scratch_.data(),
                                      t * n_heads, n_heads);
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
                const size_t head_state_floats = static_cast<size_t>(d_k) * d_v;
                float *S = state + static_cast<size_t>(h) * head_state_floats;

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

                        if (capture_state_snapshots && t < max_snapshot_rows)
                        {
                            float *snapshot_head =
                                state_snapshots + static_cast<size_t>(t) * snapshot_stride_floats +
                                static_cast<size_t>(h) * head_state_floats;
                            std::memcpy(snapshot_head, S, head_state_floats * sizeof(float));
                        }

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

#if defined(__AVX2__)
                        // Fused step 1+2: Decay S + kv_mem = S^T * k
                        avx2::zero(kv_mem, d_v);
                        for (int j = 0; j < d_k; ++j)
                        {
#if defined(__AVX512F__) || defined(__AVX2__)
                            if (j + pf_rows_ahead < d_k)
                                GDN_PREFETCH_S(S + (j + pf_rows_ahead) * d_v, pf_to_l1);
#endif
                            float *S_row = S + j * d_v;
                            avx2::scale(S_row, d_v, decay_val);
                            avx2::axpy(kv_mem, S_row, k_t[j], d_v);
                        }

                        // Step 3: delta = (v - kv_mem) * beta
                        avx2::sub_mul(delta, v_t, kv_mem, beta_t, d_v);

                        // Fused step 4+5: S += k⊗δ, output += S^T * q
                        avx2::zero(o_t, d_v);
                        for (int j = 0; j < d_k; ++j)
                        {
#if defined(__AVX512F__) || defined(__AVX2__)
                            if (j + pf_rows_ahead < d_k)
                                GDN_PREFETCH_S(S + (j + pf_rows_ahead) * d_v, pf_to_l1);
#endif
                            float *S_row = S + j * d_v;
                            avx2::axpy(S_row, delta, k_t[j], d_v);
                            avx2::axpy(o_t, S_row, q_t[j], d_v);
                        }
#else
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
#endif
                        if (capture_state_snapshots && t < max_snapshot_rows)
                        {
                            float *snapshot_head =
                                state_snapshots + static_cast<size_t>(t) * snapshot_stride_floats +
                                static_cast<size_t>(h) * head_state_floats;
                            std::memcpy(snapshot_head, S, head_state_floats * sizeof(float));
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
