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
 *    Strictly ordered recurrence parallelized across independent heads. The
 *    useful team is capped by the local head count because splitting one head's
 *    value columns duplicates Q/K traffic and is slower on measured hardware.
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
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <vector>

#include "../simd/AVX2Helpers.h"

namespace llaminar2
{
    /**
     * @brief Test two byte ranges for overlap and fail closed on address overflow.
     */
    static bool byteRangesOverlap(
        const void *left,
        size_t left_bytes,
        const void *right,
        size_t right_bytes)
    {
        const uintptr_t left_begin = reinterpret_cast<uintptr_t>(left);
        const uintptr_t right_begin = reinterpret_cast<uintptr_t>(right);
        if (left_bytes > std::numeric_limits<uintptr_t>::max() - left_begin ||
            right_bytes > std::numeric_limits<uintptr_t>::max() - right_begin)
        {
            return true;
        }

        const uintptr_t left_end = left_begin + left_bytes;
        const uintptr_t right_end = right_begin + right_bytes;
        return left_begin < right_end && right_begin < left_end;
    }

    bool CPUGatedDeltaNet::resetGPUState(void *stream)
    {
        (void)stream;
        request_state_bank_.clear();
        request_state_size_ = 0;
        request_state_capacity_ = 0;
        owned_speculative_state_work_.clear();
        return true;
    }

    bool CPUGatedDeltaNet::ensureRequestStateBank(
        int request_count,
        int state_floats,
        const float *request_zero_state)
    {
        if (request_count <= 0 || state_floats <= 0 || !request_zero_state)
            return false;

        if (request_state_size_ != state_floats)
        {
            request_state_bank_.assign(
                static_cast<size_t>(request_count) * state_floats,
                0.0f);
            request_state_size_ = state_floats;
            request_state_capacity_ = request_count;
        }
        else if (request_state_capacity_ < request_count)
        {
            request_state_bank_.resize(
                static_cast<size_t>(request_count) * state_floats,
                0.0f);
            request_state_capacity_ = request_count;
        }

        /* Request zero remains the public host prefix-state representation. */
        std::memcpy(
            request_state_bank_.data(),
            request_zero_state,
            static_cast<size_t>(state_floats) * sizeof(float));
        return true;
    }

    void CPUGatedDeltaNet::bindVerifierStateCaptureWorkspace(float *workspace, int rows, int state_size)
    {
        verifier_state_capture_ = workspace;
        verifier_state_capture_rows_ = rows;
        verifier_state_capture_size_ = state_size;
    }

    void CPUGatedDeltaNet::bindSpeculativeStateWorkspace(float *workspace, int state_size)
    {
        speculative_state_work_ = workspace;
        speculative_state_work_size_ = state_size;
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

    bool CPUGatedDeltaNet::restoreVerifierStateCaptureRows(
        float *dst_state,
        const int *host_row_indices,
        int request_count,
        void *stream)
    {
        (void)stream;
        if (!dst_state || !host_row_indices || request_count <= 0 ||
            !verifier_state_capture_ || verifier_state_capture_size_ <= 0)
        {
            return false;
        }

        /*
         * Publication is one logical transaction across the request bank.
         * Reject every invalid row before copying any slot so callers never
         * inherit a partially committed speculative timeline.
         */
        for (int request = 0; request < request_count; ++request)
        {
            const int row = host_row_indices[request];
            if (row >= verifier_state_capture_rows_)
                return false;
        }

        if (request_count == 1)
        {
            /*
             * A one-request transaction still arrives through the grouped row
             * vector API, but its live recurrence state is the public state
             * tensor rather than an internal request bank.  Commit the selected
             * post-row snapshot directly.  This is a native byte copy only; it
             * never replays a recurrent row or changes arithmetic ordering.
             */
            const int row = host_row_indices[0];
            if (row >= 0)
            {
                std::memcpy(
                    dst_state,
                    verifier_state_capture_ +
                        static_cast<size_t>(row) * verifier_state_capture_size_,
                    static_cast<size_t>(verifier_state_capture_size_) * sizeof(float));
            }

            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_request_batched_state_publications",
                1.0,
                "decode",
                "cpu",
                {{"request_count", "1"},
                 {"publication_policy", "single_request_snapshot_copy"}});
            return true;
        }

        if (request_count > request_state_capacity_ ||
            request_state_size_ <= 0 ||
            verifier_state_capture_size_ < request_state_size_)
        {
            return false;
        }

        for (int request = 0; request < request_count; ++request)
        {
            const int row = host_row_indices[request];
            if (row < 0)
                continue;
            float *destination =
                request_state_bank_.data() +
                static_cast<size_t>(request) * request_state_size_;
            const float *source =
                verifier_state_capture_ +
                static_cast<size_t>(row) * verifier_state_capture_size_;
            std::memcpy(
                destination,
                source,
                static_cast<size_t>(request_state_size_) * sizeof(float));
            if (request == 0)
            {
                std::memcpy(
                    dst_state,
                    destination,
                    static_cast<size_t>(request_state_size_) * sizeof(float));
            }
        }

        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_gdn_request_batched_state_publications",
            1.0,
            "decode",
            "cpu",
            {{"request_count", std::to_string(request_count)},
             {"publication_policy", "request_bank_snapshot_copy"}});
        return true;
    }

    float *CPUGatedDeltaNet::prepareSpeculativeState(float *live_state, int state_floats)
    {
        if (!live_state || state_floats <= 0)
            return nullptr;

        float *work = nullptr;
        if (speculative_state_work_ && speculative_state_work_size_ >= state_floats)
        {
            work = speculative_state_work_;
        }
        else
        {
            owned_speculative_state_work_.resize(static_cast<size_t>(state_floats));
            work = owned_speculative_state_work_.data();
            speculative_state_work_size_ = std::max(speculative_state_work_size_, state_floats);
        }

        std::memcpy(work, live_state, static_cast<size_t>(state_floats) * sizeof(float));
        return work;
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
        const int state_floats = n_heads * d_k * d_v;
        const bool verifier_capture_active =
            verifier_state_capture_ &&
            verifier_state_capture_rows_ > 0 &&
            verifier_state_capture_size_ >= state_floats &&
            state_floats > 0;
        float *state_for_compute = state;
        if (verifier_capture_active)
        {
            state_for_compute = prepareSpeculativeState(state, state_floats);
            if (!state_for_compute)
                return false;
        }

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
                float *S = state_for_compute + static_cast<size_t>(h) * d_k * d_v;
                const float *q_h = q_local;
                const float *k_h = k_local;
                const float *v_h = v + h * d_v;
                float *o_h = output + h * d_v;

                gdn_delta_recurrence(S, q_h, k_h, v_h, o_h, decay, beta_h, d_k, d_v);
            }
        };
        OMP_WORKSHARE_REGION(do_work);

        if (verifier_capture_active)
        {
            std::memcpy(verifier_state_capture_,
                        state_for_compute,
                        static_cast<size_t>(state_floats) * sizeof(float));
        }

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
    // Why NO d_v tiling: each output column is mathematically independent, but
    // splitting a head duplicates its Q/K stream and makes multiple cores walk
    // adjacent state rows. Measured Qwen 3.6 d_k=d_v=128 latency regresses as
    // soon as a second worker shares one head. Whole-head ownership therefore
    // remains both byte-exact and economical; the launch team is capped to the
    // useful local head count instead of manufacturing dominated work.
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

#if defined(__AVX512F__)
    /**
     * @brief Advance one complete d_v=128 GDN head with ZMM-resident vectors.
     *
     * @param q_scratch Canonically normalized/scaled Q rows.
     * @param k_scratch Canonically normalized K rows.
     * @param values Original V rows.
     * @param gate_scratch Per-row decay factors.
     * @param beta_scratch Per-row sigmoid(beta) factors.
     * @param output Complete output tensor.
     * @param state Initial recurrence state, or mutable state without snapshots.
     * @param seq_len Number of rows advanced in strict ascending order.
     * @param n_heads Number of local value heads.
     * @param d_k Key width and state-row count.
     * @param head Local head owned by this task.
     * @param value_row_stride Distance between source V rows in FP32 elements.
     * @param prefetch_rows Number of future state rows to prefetch.
     * @param prefetch_to_l1 Whether state geometry fits the detected L1 policy.
     * @param state_snapshots Optional full-head snapshot storage.
     * @param snapshot_stride_floats Distance between snapshot rows.
     *
     * The eight value vectors remain in ZMM registers through both reductions,
     * while every lane preserves the serial-decode `j=0..d_k-1` accumulation
     * order. When snapshots are requested, row zero is written directly from
     * @p state into snapshot zero and every later row advances its predecessor
     * into the next snapshot. This makes snapshots the recurrence destination,
     * eliminating state clones and post-row copies without changing arithmetic.
     */
    static void gdnChunkForwardAVX512DV128(
        const float *q_scratch,
        const float *k_scratch,
        const float *values,
        const float *gate_scratch,
        const float *beta_scratch,
        float *output,
        float *state,
        int seq_len,
        int n_heads,
        int d_k,
        int head,
        int value_row_stride,
        int prefetch_rows,
        bool prefetch_to_l1,
        float *state_snapshots,
        int snapshot_stride_floats)
    {
        constexpr int VectorCount = 8;
        constexpr int kVectorWidth = 16;
        constexpr int kValueWidth = 128;
        const int qk_stride = n_heads * d_k;
        const int output_stride = n_heads * kValueWidth;
        const size_t head_state_floats =
            static_cast<size_t>(d_k) * kValueWidth;
        float *input_head_state =
            state + static_cast<size_t>(head) * head_state_floats;

        for (int token = 0; token < seq_len; ++token)
        {
            const float *q = q_scratch +
                             static_cast<size_t>(token) * qk_stride +
                             static_cast<size_t>(head) * d_k;
            const float *k = k_scratch +
                             static_cast<size_t>(token) * qk_stride +
                             static_cast<size_t>(head) * d_k;
            const float *v = values +
                             static_cast<size_t>(token) * value_row_stride +
                             static_cast<size_t>(head) * kValueWidth;
            float *o = output +
                       static_cast<size_t>(token) * output_stride +
                       static_cast<size_t>(head) * kValueWidth;
            const float *source_head_state = input_head_state;
            float *destination_head_state = input_head_state;
            if (state_snapshots)
            {
                if (token > 0)
                {
                    source_head_state =
                        state_snapshots +
                        static_cast<size_t>(token - 1) * snapshot_stride_floats +
                        static_cast<size_t>(head) * head_state_floats;
                }
                destination_head_state =
                    state_snapshots +
                    static_cast<size_t>(token) * snapshot_stride_floats +
                    static_cast<size_t>(head) * head_state_floats;
            }

            __m512 kv[VectorCount];
#pragma GCC unroll 8
            for (int vector = 0; vector < VectorCount; ++vector)
                kv[vector] = _mm512_setzero_ps();

            const __m512 decay = _mm512_set1_ps(
                gate_scratch[static_cast<size_t>(token) * n_heads + head]);
            for (int row = 0; row < d_k; ++row)
            {
                if (row + prefetch_rows < d_k)
                {
                    GDN_PREFETCH_S(
                        source_head_state +
                            static_cast<size_t>(row + prefetch_rows) * kValueWidth,
                        prefetch_to_l1);
                }

                const __m512 key = _mm512_set1_ps(k[row]);
                const float *source_state_row =
                    source_head_state + static_cast<size_t>(row) * kValueWidth;
                float *destination_state_row =
                    destination_head_state +
                    static_cast<size_t>(row) * kValueWidth;
#pragma GCC unroll 8
                for (int vector = 0; vector < VectorCount; ++vector)
                {
                    const float *source_segment =
                        source_state_row + vector * kVectorWidth;
                    float *destination_segment =
                        destination_state_row + vector * kVectorWidth;
                    const __m512 decayed = _mm512_mul_ps(
                        _mm512_loadu_ps(source_segment), decay);
                    _mm512_storeu_ps(destination_segment, decayed);
                    kv[vector] = _mm512_fmadd_ps(decayed, key, kv[vector]);
                }
            }

            const __m512 beta = _mm512_set1_ps(
                beta_scratch[static_cast<size_t>(token) * n_heads + head]);
#pragma GCC unroll 8
            for (int vector = 0; vector < VectorCount; ++vector)
            {
                kv[vector] = _mm512_mul_ps(
                    _mm512_sub_ps(
                        _mm512_loadu_ps(v + vector * kVectorWidth),
                        kv[vector]),
                    beta);
            }

            __m512 accumulated_output[VectorCount];
#pragma GCC unroll 8
            for (int vector = 0; vector < VectorCount; ++vector)
                accumulated_output[vector] = _mm512_setzero_ps();

            for (int row = 0; row < d_k; ++row)
            {
                if (row + prefetch_rows < d_k)
                {
                    GDN_PREFETCH_S(
                        destination_head_state +
                            static_cast<size_t>(row + prefetch_rows) * kValueWidth,
                        prefetch_to_l1);
                }

                const __m512 key = _mm512_set1_ps(k[row]);
                const __m512 query = _mm512_set1_ps(q[row]);
                float *state_row =
                    destination_head_state +
                    static_cast<size_t>(row) * kValueWidth;
#pragma GCC unroll 8
                for (int vector = 0; vector < VectorCount; ++vector)
                {
                    float *segment = state_row + vector * kVectorWidth;
                    const __m512 updated = _mm512_fmadd_ps(
                        key,
                        kv[vector],
                        _mm512_loadu_ps(segment));
                    _mm512_storeu_ps(segment, updated);
                    accumulated_output[vector] = _mm512_fmadd_ps(
                        updated,
                        query,
                        accumulated_output[vector]);
                }
            }

#pragma GCC unroll 8
            for (int vector = 0; vector < VectorCount; ++vector)
            {
                _mm512_storeu_ps(
                    o + vector * kVectorWidth,
                    accumulated_output[vector]);
            }

        }
    }
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
        const ChunkInputView input{
            .q = Q,
            .k = K,
            .v = V,
            .q_row_stride = n_heads * d_k,
            .k_row_stride = n_heads * d_k,
            .v_row_stride = n_heads * d_v,
            .n_k_heads = n_heads,
            .global_v_head_offset = 0,
            .layout_name = "separate_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            state_snapshots
                ? InputStateDisposition::PreserveInitialState
                : InputStateDisposition::PublishTerminalState);
    }

    bool CPUGatedDeltaNet::chunkForwardMergedQKV(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, int chunk_size,
        bool use_qk_l2norm)
    {
        if (!merged_qkv || seq_len <= 0 || n_k_heads <= 0 ||
            n_heads <= 0 || d_k <= 0 || d_v <= 0)
        {
            return false;
        }

        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

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

        if (state_snapshots)
        {
            return chunkForwardMergedQKVWithStateSnapshots(
                merged_qkv, qkv_stride,
                alpha, beta_raw, A_log, dt_bias,
                output, state,
                seq_len, n_k_heads, n_heads, d_k, d_v,
                global_v_head_offset, chunk_size, use_qk_l2norm,
                state_snapshots, snapshot_stride_floats,
                max_snapshot_rows);
        }

        const ChunkInputView input{
            .q = merged_qkv,
            .k = merged_qkv + q_src_dim,
            .v = merged_qkv + q_src_dim + k_src_dim,
            .q_row_stride = qkv_stride,
            .k_row_stride = qkv_stride,
            .v_row_stride = qkv_stride,
            .n_k_heads = n_k_heads,
            .global_v_head_offset = global_v_head_offset,
            .layout_name = "merged_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PublishTerminalState);
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
        const ChunkInputView input{
            .q = Q,
            .k = K,
            .v = V,
            .q_row_stride = n_heads * d_k,
            .k_row_stride = n_heads * d_k,
            .v_row_stride = n_heads * d_v,
            .n_k_heads = n_heads,
            .global_v_head_offset = 0,
            .layout_name = "separate_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PublishTerminalState);
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

    bool CPUGatedDeltaNet::chunkForwardMergedQKVWithStateSnapshots(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, int chunk_size, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows)
    {
        if (!merged_qkv || seq_len <= 0 || n_k_heads <= 0 ||
            n_heads <= 0 || d_k <= 0 || d_v <= 0)
        {
            return false;
        }

        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

        /*
         * Snapshot publication is an output contract, not a distinct recurrence
         * algorithm.  Use the same source view and kernel as ordinary M=1 decode
         * so grouped verifier rows cannot drift into a second arithmetic path.
         */
        const ChunkInputView input{
            .q = merged_qkv,
            .k = merged_qkv + q_src_dim,
            .v = merged_qkv + q_src_dim + k_src_dim,
            .q_row_stride = qkv_stride,
            .k_row_stride = qkv_stride,
            .v_row_stride = qkv_stride,
            .n_k_heads = n_k_heads,
            .global_v_head_offset = global_v_head_offset,
            .layout_name = "merged_qkv"};
        return chunkForwardImpl(
            input, alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, n_heads, d_k, d_v, chunk_size, use_qk_l2norm,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            InputStateDisposition::PreserveInitialState);
    }

    template <typename RowAccessor>
    bool CPUGatedDeltaNet::chunkForwardBatchedDecodeEquivalentRows(
        RowAccessor &&row_accessor,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        if (!alpha || !beta_raw || !A_log || !dt_bias || !output || !state ||
            !host_request_seq_lens || seq_len <= 0 || request_count <= 0 ||
            request_seq_len <= 0 || n_heads <= 0 || d_k <= 0 || d_v <= 0 ||
            request_seq_len > std::numeric_limits<int>::max() / request_count ||
            seq_len != request_count * request_seq_len ||
            d_k > 512 || d_v > 512)
        {
            return false;
        }

        const size_t state_floats_wide =
            static_cast<size_t>(n_heads) * d_k * d_v;
        if (state_floats_wide > static_cast<size_t>(std::numeric_limits<int>::max()))
            return false;
        const int state_floats = static_cast<int>(state_floats_wide);
        if (state_floats <= 0 ||
            state_floats > std::numeric_limits<int>::max() / request_count ||
            n_heads > std::numeric_limits<int>::max() / request_count)
        {
            return false;
        }
        const int total_state_floats = request_count * state_floats;
        if (!ensureRequestStateBank(request_count, state_floats, state))
            return false;

        const bool capture_active =
            verifier_state_capture_ != nullptr &&
            verifier_state_capture_rows_ >= seq_len &&
            verifier_state_capture_size_ >= state_floats;
        float *effective_states = request_state_bank_.data();
        if (capture_active)
        {
            if (speculative_state_work_)
            {
                if (speculative_state_work_size_ < total_state_floats)
                    return false;
                effective_states = speculative_state_work_;
            }
            else
            {
                owned_speculative_state_work_.resize(
                    static_cast<size_t>(total_state_floats));
                effective_states = owned_speculative_state_work_.data();
            }
            std::memcpy(
                effective_states,
                request_state_bank_.data(),
                static_cast<size_t>(total_state_floats) * sizeof(float));
        }

        PerfStatsCollector::addCounter(
            "kernel",
            "cpu_gdn_request_batched_grouped_calls",
            1.0,
            "prefill",
            "cpu",
            {{"request_count", std::to_string(request_count)},
             {"request_row_width", std::to_string(request_seq_len)},
             {"n_heads", std::to_string(n_heads)},
             {"execution_policy", "request_head_grouped_recurrence"}});

        const float scale_val = 1.0f / std::sqrt(static_cast<float>(d_k));
        constexpr float l2_eps = 1e-6f;
        const int v_stride = n_heads * d_v;
        const size_t head_state_floats =
            static_cast<size_t>(d_k) * d_v;
        std::atomic<bool> row_binding_failed{false};

        auto grouped_requests = [&]()
        {
#pragma omp for schedule(static)
            for (int work = 0; work < request_count * n_heads; ++work)
            {
                const int request = work / n_heads;
                const int head = work % n_heads;
                const int real_rows = std::clamp(
                    host_request_seq_lens[request], 0, request_seq_len);
                float *S =
                    effective_states +
                    static_cast<size_t>(request) * state_floats +
                    static_cast<size_t>(head) * head_state_floats;
                alignas(64) float q_local[512];
                alignas(64) float k_local[512];

                for (int row = 0; row < real_rows; ++row)
                {
                    const int flat_row = request * request_seq_len + row;
                    const float *q_row = nullptr;
                    const float *k_row = nullptr;
                    const float *v_row = nullptr;
                    row_accessor(request, row, head, q_row, k_row, v_row);
                    if (!q_row || !k_row || !v_row)
                    {
                        /*
                         * Accessors are private, shape-checked production
                         * adapters.  Keep this fail-closed guard nevertheless:
                         * silently skipping one head would publish plausible
                         * but mathematically incomplete verifier state.
                         */
                        row_binding_failed.store(true, std::memory_order_relaxed);
                        break;
                    }

                    if (use_qk_l2norm)
                    {
                        gdn_preprocess_qk_l2norm(
                            q_row, k_row, q_local, k_local,
                            d_k, scale_val, l2_eps);
                    }
                    else
                    {
                        gdn_preprocess_qk_scale(
                            q_row, k_row, q_local, k_local,
                            d_k, scale_val);
                    }

                    const size_t gate_index =
                        static_cast<size_t>(flat_row) * n_heads + head;
                    const float x = alpha[gate_index] + dt_bias[head];
                    const float softplus =
                        x > 20.0f ? x : std::log1p(std::exp(x));
                    const float decay = std::exp(A_log[head] * softplus);
                    const float beta =
                        1.0f / (1.0f + std::exp(-beta_raw[gate_index]));
                    float *output_row =
                        output + static_cast<size_t>(flat_row) * v_stride +
                        head * d_v;
                    gdn_delta_recurrence(
                        S, q_local, k_local, v_row, output_row,
                        decay, beta, d_k, d_v);

                    if (capture_active)
                    {
                        float *snapshot =
                            verifier_state_capture_ +
                            static_cast<size_t>(flat_row) * verifier_state_capture_size_ +
                            static_cast<size_t>(head) * head_state_floats;
                        std::memcpy(
                            snapshot, S,
                            head_state_floats * sizeof(float));
                    }
                }

                for (int row = real_rows; row < request_seq_len; ++row)
                {
                    const int flat_row = request * request_seq_len + row;
                    std::memset(
                        output + static_cast<size_t>(flat_row) * v_stride +
                            head * d_v,
                        0,
                        static_cast<size_t>(d_v) * sizeof(float));
                    if (capture_active)
                    {
                        std::memset(
                            verifier_state_capture_ +
                                static_cast<size_t>(flat_row) * verifier_state_capture_size_ +
                                static_cast<size_t>(head) * head_state_floats,
                            0,
                            head_state_floats * sizeof(float));
                    }
                }
            }
        };
        OMP_WORKSHARE_REGION(grouped_requests);

        if (row_binding_failed.load(std::memory_order_relaxed))
            return false;

        if (!capture_active)
        {
            std::memcpy(
                state,
                request_state_bank_.data(),
                static_cast<size_t>(state_floats) * sizeof(float));
        }
        return true;
    }

    bool CPUGatedDeltaNet::chunkForwardBatchedRequestsWithHostSeqLens(
        const float *Q, const float *K, const float *V,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_heads, int d_k, int d_v,
        int chunk_size, bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        (void)chunk_size;
        if (!Q || !K || !V)
            return false;

        const int qk_stride = n_heads * d_k;
        const int v_stride = n_heads * d_v;
        auto contiguous_rows =
            [=](int request, int row, int head,
                const float *&q_row,
                const float *&k_row,
                const float *&v_row)
        {
            const int flat_row = request * request_seq_len + row;
            q_row = Q + static_cast<size_t>(flat_row) * qk_stride + head * d_k;
            k_row = K + static_cast<size_t>(flat_row) * qk_stride + head * d_k;
            v_row = V + static_cast<size_t>(flat_row) * v_stride + head * d_v;
        };
        return chunkForwardBatchedDecodeEquivalentRows(
            contiguous_rows,
            alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, request_count, request_seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            host_request_seq_lens);
    }

    bool CPUGatedDeltaNet::chunkForwardBatchedMergedQKVWithHostSeqLens(
        const float *merged_qkv, int qkv_stride,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int request_count, int request_seq_len,
        int n_k_heads, int n_heads, int d_k, int d_v,
        int global_v_head_offset, bool use_qk_l2norm,
        const int *host_request_seq_lens)
    {
        if (!merged_qkv || n_k_heads <= 0 || n_heads <= 0 ||
            d_k <= 0 || d_v <= 0)
        {
            return false;
        }
        const int q_src_dim = n_k_heads * d_k;
        const int k_src_dim = n_k_heads * d_k;
        const int v_dim = n_heads * d_v;
        if (qkv_stride < q_src_dim + k_src_dim + v_dim)
            return false;

        auto merged_rows =
            [=](int request, int row, int head,
                const float *&q_row,
                const float *&k_row,
                const float *&v_row)
        {
            int qk_head = (head + global_v_head_offset) % n_k_heads;
            if (qk_head < 0)
                qk_head += n_k_heads;
            const int flat_row = request * request_seq_len + row;
            const float *source =
                merged_qkv + static_cast<size_t>(flat_row) * qkv_stride;
            q_row = source + static_cast<size_t>(qk_head) * d_k;
            k_row = source + q_src_dim + static_cast<size_t>(qk_head) * d_k;
            v_row = source + q_src_dim + k_src_dim +
                    static_cast<size_t>(head) * d_v;
        };
        return chunkForwardBatchedDecodeEquivalentRows(
            merged_rows,
            alpha, beta_raw, A_log, dt_bias, output, state,
            seq_len, request_count, request_seq_len,
            n_heads, d_k, d_v, use_qk_l2norm,
            host_request_seq_lens);
    }

    bool CPUGatedDeltaNet::chunkForwardImpl(
        const ChunkInputView &input,
        const float *alpha, const float *beta_raw,
        const float *A_log, const float *dt_bias,
        float *output, float *state,
        int seq_len, int n_heads, int d_k, int d_v,
        int /*chunk_size*/, bool use_qk_l2norm,
        float *state_snapshots, int snapshot_stride_floats,
        int max_snapshot_rows,
        InputStateDisposition input_state_disposition)
    {
        if (!input.q || !input.k || !input.v || !input.layout_name ||
            input.q_row_stride <= 0 || input.k_row_stride <= 0 ||
            input.v_row_stride <= 0 || input.n_k_heads <= 0 ||
            !alpha || !beta_raw || !A_log || !dt_bias || !output || !state ||
            seq_len <= 0 || n_heads <= 0 || d_k <= 0 || d_v <= 0 ||
            d_k > 512 || d_v > 512)
        {
            return false;
        }

        const size_t state_floats_wide =
            static_cast<size_t>(n_heads) * d_k * d_v;
        if (state_floats_wide >
            static_cast<size_t>(std::numeric_limits<int>::max()))
        {
            return false;
        }
        const int state_floats = static_cast<int>(state_floats_wide);
        const bool capture_state_snapshots = state_snapshots != nullptr;
        if (capture_state_snapshots &&
            (snapshot_stride_floats < state_floats ||
             max_snapshot_rows < seq_len))
        {
            return false;
        }
        if (!capture_state_snapshots &&
            input_state_disposition == InputStateDisposition::PreserveInitialState)
        {
            return false;
        }
        if (capture_state_snapshots)
        {
            const size_t preceding_snapshot_rows =
                static_cast<size_t>(seq_len - 1);
            const size_t snapshot_stride =
                static_cast<size_t>(snapshot_stride_floats);
            if (preceding_snapshot_rows >
                (std::numeric_limits<size_t>::max() - state_floats_wide) /
                    snapshot_stride)
            {
                return false;
            }
            const size_t snapshot_span_floats =
                preceding_snapshot_rows * snapshot_stride + state_floats_wide;
            if (state_floats_wide >
                    std::numeric_limits<size_t>::max() / sizeof(float) ||
                snapshot_span_floats >
                    std::numeric_limits<size_t>::max() / sizeof(float) ||
                byteRangesOverlap(
                    state,
                    state_floats_wide * sizeof(float),
                    state_snapshots,
                    snapshot_span_floats * sizeof(float)))
            {
                return false;
            }
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
        const int active_or_requested_workers = omp_in_parallel()
                                                    ? omp_get_num_threads()
                                                    : omp_get_max_threads();
        const int useful_worker_count = std::max(
            1,
            std::min(active_or_requested_workers, n_heads));

        if (capture_state_snapshots)
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_grouped_verifier_recurrence_calls",
                1.0,
                "verifier",
                "cpu",
                {{"verifier_rows", std::to_string(seq_len)},
                 {"n_heads", std::to_string(n_heads)},
                 {"d_k", std::to_string(d_k)},
                 {"d_v", std::to_string(d_v)},
                 {"snapshot_rows", std::to_string(seq_len)},
                 {"input_layout", input.layout_name},
                 {"parallel_axis", "head"},
                 {"active_or_requested_workers", std::to_string(active_or_requested_workers)},
                 {"useful_worker_count", std::to_string(useful_worker_count)},
                 {"execution_policy", "head_grouped_recurrence"},
                 {"kernel_variant", "canonical_chunk_forward"},
                 {"snapshot_materialization", "direct_row_destination"},
                 {"arithmetic_order", "serial_decode_per_head"}});
        }
        else
        {
            PerfStatsCollector::addCounter(
                "kernel",
                "cpu_gdn_prefill_recurrence_calls",
                1.0,
                seq_len == 1 ? "decode" : "prefill",
                "cpu",
                {{"rows", std::to_string(seq_len)},
                 {"n_heads", std::to_string(n_heads)},
                 {"d_k", std::to_string(d_k)},
                 {"d_v", std::to_string(d_v)},
                 {"input_layout", input.layout_name},
                 {"parallel_axis", "head"},
                 {"active_or_requested_workers", std::to_string(active_or_requested_workers)},
                 {"useful_worker_count", std::to_string(useful_worker_count)},
                 {"arithmetic_order", "serial_decode_per_head"}});
        }

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
                    int source_head =
                        (h + input.global_v_head_offset) % input.n_k_heads;
                    if (source_head < 0)
                        source_head += input.n_k_heads;
                    const float *q_src =
                        input.q + static_cast<size_t>(t) * input.q_row_stride +
                        static_cast<size_t>(source_head) * d_k;
                    const float *k_src =
                        input.k + static_cast<size_t>(t) * input.k_row_stride +
                        static_cast<size_t>(source_head) * d_k;

                    if (use_qk_l2norm)
                    {
                        gdn_preprocess_qk_l2norm(q_src, k_src, q_dst, k_dst, d_k, scale_val, l2_eps);
                    }
                    else
                    {
                        gdn_preprocess_qk_scale(q_src, k_src, q_dst, k_dst, d_k, scale_val);
                    }

                    // Compute the exact decode gate expressions while tokens
                    // are already distributed across the OpenMP team. The old
                    // AVX fast-exp batch changed arithmetic between prefill and
                    // decode; keeping these canonical expressions here both
                    // preserves parallelism and removes a second head traversal.
                    const int gi = t * n_heads + h;
                    const float x = alpha[gi] + dt_bias[h];
                    const float sp = (x > 20.0f) ? x : std::log1p(std::exp(x));
                    gate_scratch_[gi] = std::exp(A_log[h] * sp);
                    beta_sig_scratch_[gi] =
                        1.0f / (1.0f + std::exp(-beta_raw[gi]));

                    // Zero output
                    std::memset(output + out_off, 0, d_v * sizeof(float));
                }

            }
            // implicit barrier between omp-for regions

            // Phase 2: fused recurrence across independent heads. Token order
            // remains serial inside each head. Whole-head ownership avoids the
            // measured Q/K duplication and adjacent-state contention of value
            // column splitting.
#pragma omp for schedule(static)
            for (int head = 0; head < n_heads; ++head)
            {
                const int column_count = d_v;
                const size_t head_state_floats =
                    static_cast<size_t>(d_k) * d_v;
                float *input_head_state =
                    state + static_cast<size_t>(head) * head_state_floats;

#if defined(__AVX512F__)
                if (d_v == 128)
                {
                    gdnChunkForwardAVX512DV128(
                        q_scratch_.data(), k_scratch_.data(), input.v,
                        gate_scratch_.data(), beta_sig_scratch_.data(),
                        output, state, seq_len, n_heads, d_k, head,
                        input.v_row_stride,
                        pf_rows_ahead, pf_to_l1,
                        state_snapshots,
                        snapshot_stride_floats);
                    continue;
                }
#endif

                // Generic AVX2/scalar whole-head implementation. Scratch is
                // private to the OpenMP head owner.
                alignas(64) float kv_mem[512];
                alignas(64) float delta[512];
                for (int token = 0; token < seq_len; ++token)
                {
                    const float *q = q_scratch_.data() +
                                     static_cast<size_t>(token) * qk_stride +
                                     static_cast<size_t>(head) * d_k;
                    const float *k = k_scratch_.data() +
                                     static_cast<size_t>(token) * qk_stride +
                                     static_cast<size_t>(head) * d_k;
                    const float *v = input.v +
                                     static_cast<size_t>(token) * input.v_row_stride +
                                     static_cast<size_t>(head) * d_v;
                    float *o = output +
                               static_cast<size_t>(token) * v_stride +
                               static_cast<size_t>(head) * d_v;
                    const float decay = gate_scratch_[
                        static_cast<size_t>(token) * n_heads + head];
                    const float beta = beta_sig_scratch_[
                        static_cast<size_t>(token) * n_heads + head];
                    const float *source_head_state = input_head_state;
                    float *destination_head_state = input_head_state;
                    if (capture_state_snapshots)
                    {
                        if (token > 0)
                        {
                            source_head_state =
                                state_snapshots +
                                static_cast<size_t>(token - 1) *
                                    snapshot_stride_floats +
                                static_cast<size_t>(head) * head_state_floats;
                        }
                        destination_head_state =
                            state_snapshots +
                            static_cast<size_t>(token) * snapshot_stride_floats +
                            static_cast<size_t>(head) * head_state_floats;
                    }

#if defined(__AVX2__)
                    avx2::zero(kv_mem, column_count);
                    for (int row = 0; row < d_k; ++row)
                    {
#if defined(__AVX512F__) || defined(__AVX2__)
                        if (row + pf_rows_ahead < d_k)
                        {
                            GDN_PREFETCH_S(
                                source_head_state +
                                    static_cast<size_t>(row + pf_rows_ahead) * d_v,
                                pf_to_l1);
                        }
#endif
                        const float *source_state_range =
                            source_head_state + static_cast<size_t>(row) * d_v;
                        float *destination_state_range =
                            destination_head_state +
                            static_cast<size_t>(row) * d_v;
                        avx2::copy_scale(
                            destination_state_range,
                            source_state_range,
                            decay,
                            column_count);
                        avx2::axpy(
                            kv_mem,
                            destination_state_range,
                            k[row],
                            column_count);
                    }
                    avx2::sub_mul(delta, v, kv_mem, beta, column_count);
                    avx2::zero(o, column_count);
                    for (int row = 0; row < d_k; ++row)
                    {
#if defined(__AVX512F__) || defined(__AVX2__)
                        if (row + pf_rows_ahead < d_k)
                        {
                            GDN_PREFETCH_S(
                                destination_head_state +
                                    static_cast<size_t>(row + pf_rows_ahead) * d_v,
                                pf_to_l1);
                        }
#endif
                        float *state_range =
                            destination_head_state +
                            static_cast<size_t>(row) * d_v;
                        avx2::axpy(state_range, delta, k[row], column_count);
                        avx2::axpy(o, state_range, q[row], column_count);
                    }
#else
                    std::memset(
                        kv_mem,
                        0,
                        static_cast<size_t>(column_count) * sizeof(float));
                    for (int row = 0; row < d_k; ++row)
                    {
                        const float *source_state_range =
                            source_head_state + static_cast<size_t>(row) * d_v;
                        float *destination_state_range =
                            destination_head_state +
                            static_cast<size_t>(row) * d_v;
                        for (int column = 0; column < column_count; ++column)
                        {
                            destination_state_range[column] =
                                source_state_range[column] * decay;
                            kv_mem[column] +=
                                destination_state_range[column] * k[row];
                        }
                    }
                    for (int column = 0; column < column_count; ++column)
                        delta[column] = (v[column] - kv_mem[column]) * beta;

                    std::memset(
                        o,
                        0,
                        static_cast<size_t>(column_count) * sizeof(float));
                    for (int row = 0; row < d_k; ++row)
                    {
                        float *state_range =
                            destination_head_state +
                            static_cast<size_t>(row) * d_v;
                        for (int column = 0; column < column_count; ++column)
                        {
                            state_range[column] += k[row] * delta[column];
                            o[column] += state_range[column] * q[row];
                        }
                    }
#endif

                }
            }

            if (capture_state_snapshots &&
                input_state_disposition ==
                    InputStateDisposition::PublishTerminalState)
            {
                const float *terminal_snapshot =
                    state_snapshots +
                    static_cast<size_t>(seq_len - 1) * snapshot_stride_floats;
#pragma omp for schedule(static)
                for (int head = 0; head < n_heads; ++head)
                {
                    const size_t head_state_floats =
                        static_cast<size_t>(d_k) * d_v;
                    std::memcpy(
                        state + static_cast<size_t>(head) * head_state_floats,
                        terminal_snapshot +
                            static_cast<size_t>(head) * head_state_floats,
                        head_state_floats * sizeof(float));
                }
            }
        };
        /*
         * Use the stable process-wide team even when the number of recurrent
         * heads is smaller than the socket width. A bounded team looks cheaper
         * in an isolated kernel sample, but libgomp may retire every omitted
         * worker and make the following full-width projection recreate it.
         * The omp-for loops naturally leave workers without a head idle while
         * preserving the process-wide team for the next captured CPU stage.
         */
        OMP_WORKSHARE_REGION(do_work);

#if defined(__AVX512F__)
#undef GDN_PREFETCH_S
#endif

        return true;
    }

} // namespace llaminar2
