/**
 * @file attention_primitives.cpp
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <limits>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <chrono>
#include <array>
#include <sstream>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils/debug_env.h"
#include "kernels/common/attention_primitives.h"
#include "kernels/common/softmax_core.h"
#include "logger.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace llaminar::attn
{

    // Internal helper: pointer to row (h,i) of scores (seq_len x seq_len per head)
    static inline float *head_row(float *scores, int h, int i, int seq_len)
    {
        return scores + (std::size_t)h * seq_len * seq_len + (std::size_t)i * seq_len;
    }
    static inline const float *head_row(const float *scores, int h, int i, int seq_len)
    {
        return scores + (std::size_t)h * seq_len * seq_len + (std::size_t)i * seq_len;
    }

    // Forward declaration (defined later)
    void compute_qk_scores(const float *q, const float *k, float *scores, int seq_len,
                           int head_dim, int heads, bool causal, bool apply_softmax);

    void apply_rope(float *q, float *k, int seq_len, int head_dim, int heads, int n_past, float freq_base)
    {
        // Unconditional lightweight trace (first few invocations) to verify this path executes in parity tests.
        static int rope_call_count = 0;
        if (rope_call_count < 8)
        {
            LOG_WARN("[RoPETraceCall] idx=" << rope_call_count << " seq_len=" << seq_len << " heads=" << heads << " head_dim=" << head_dim << " n_past=" << n_past << " freq_base=" << freq_base);
        }
        ++rope_call_count;
        const auto &env = llaminar::debugEnv().attention;
        const int pairs = head_dim / 2;
        std::vector<float> theta(pairs);
        for (int p = 0; p < pairs; ++p)
            theta[p] = 1.f / std::pow(freq_base, (2.f * p) / head_dim);

        bool diag = llaminar::debugEnv().attention.internal_diff && llaminar::debugEnv().pipeline.layer_token_diff;
        int preview = std::min(head_dim * heads, 8);
        std::array<float, 8> q_before{};
        std::array<float, 8> k_before{};
        q_before.fill(0.f);
        k_before.fill(0.f);
        if (diag && seq_len > 0 && preview > 0)
        {
            size_t row_offset = (size_t)(seq_len - 1) * heads * head_dim; // last token row
            for (int i = 0; i < preview; ++i)
            {
                q_before[i] = q[row_offset + i];
                k_before[i] = k[row_offset + i];
            }
        }

        const size_t total_elems = (size_t)heads * seq_len * head_dim;
        // Auto-tune cache: decide best path per (heads, head_dim, seq_len) once per process.
        struct Key
        {
            int h;
            int d;
            int s;
        };
        struct KeyHash
        {
            std::size_t operator()(const Key &k) const noexcept { return ((std::size_t)k.h * 1315423911u) ^ ((std::size_t)k.d << 1) ^ ((std::size_t)k.s << 16); }
        };
        struct KeyEq
        {
            bool operator()(const Key &a, const Key &b) const noexcept { return a.h == b.h && a.d == b.d && a.s == b.s; }
        };
        static std::unordered_map<Key, bool, KeyHash, KeyEq> g_choice; // cached decision: true => recurrence
        bool use_recurrence = false;
        if (!env.prim_rope_disable_recurrence)
        {
            use_recurrence = (total_elems >= (size_t)env.prim_rope_recurrence_threshold);
            Key key{heads, head_dim, seq_len};
            auto it = g_choice.find(key);
            if (it == g_choice.end())
            {
                // Micro-probe only if near threshold or heuristic would enable, to avoid overhead on tiny shapes.
                bool probe = use_recurrence || total_elems >= (size_t)(0.8 * env.prim_rope_recurrence_threshold);
                if (probe)
                {
                    // Copy small slices of q,k (first token per head) and time one iteration of both paths on a reduced view.
                    const int probe_tokens = std::min(seq_len, 16); // limited tokens for quick probe
                    std::vector<float> q_copy(q, q + (size_t)probe_tokens * heads * head_dim);
                    std::vector<float> k_copy(k, k + (size_t)probe_tokens * heads * head_dim);
                    auto run_legacy = [&](std::vector<float> &Q, std::vector<float> &K)
                    {
                        // Minimal legacy rotation (single pass trig per position) using existing code path by forcing disable flags.
                        for (int tt = 0; tt < probe_tokens; ++tt)
                        {
                            float pos = float(n_past + tt);
                            for (int h = 0; h < heads; ++h)
                            {
                                float *qh = Q.data() + (size_t)tt * heads * head_dim + h * head_dim;
                                float *kh = K.data() + (size_t)tt * heads * head_dim + h * head_dim;
                                for (int p2 = 0; p2 < pairs; ++p2)
                                {
                                    float angle = pos * theta[p2];
                                    float cs = std::cos(angle);
                                    float sn = std::sin(angle);
                                    int i0 = 2 * p2;
                                    int i1 = i0 + 1;
                                    if (i1 >= head_dim)
                                        break;
                                    float q0 = qh[i0], q1 = qh[i1];
                                    float k0 = kh[i0], k1 = kh[i1];
                                    qh[i0] = q0 * cs - q1 * sn;
                                    qh[i1] = q0 * sn + q1 * cs;
                                    kh[i0] = k0 * cs - k1 * sn;
                                    kh[i1] = k0 * sn + k1 * cs;
                                }
                            }
                        }
                    };
                    auto run_recur = [&](std::vector<float> &Q, std::vector<float> &K)
                    {
                        std::vector<float> sin_base(pairs), cos_base(pairs), sin_delta(pairs), cos_delta(pairs);
                        for (int p2 = 0; p2 < pairs; ++p2)
                        {
                            float base_angle = (float)n_past * theta[p2];
                            sin_base[p2] = std::sin(base_angle);
                            cos_base[p2] = std::cos(base_angle);
                            float delta = theta[p2];
                            sin_delta[p2] = std::sin(delta);
                            cos_delta[p2] = std::cos(delta);
                        }
                        for (int h = 0; h < heads; ++h)
                        {
                            std::vector<float> s_cur(sin_base), c_cur(cos_base);
                            for (int tt = 0; tt < probe_tokens; ++tt)
                            {
                                float *qh = Q.data() + (size_t)tt * heads * head_dim + h * head_dim;
                                float *kh = K.data() + (size_t)tt * heads * head_dim + h * head_dim;
                                for (int p3 = 0; p3 < pairs; ++p3)
                                {
                                    int i0 = 2 * p3, i1 = i0 + 1;
                                    if (i1 >= head_dim)
                                        break;
                                    float cs = c_cur[p3], sn = s_cur[p3];
                                    float q0 = qh[i0], q1 = qh[i1];
                                    float k0 = kh[i0], k1 = kh[i1];
                                    qh[i0] = q0 * cs - q1 * sn;
                                    qh[i1] = q0 * sn + q1 * cs;
                                    kh[i0] = k0 * cs - k1 * sn;
                                    kh[i1] = k0 * sn + k1 * cs;
                                }
                                // advance
                                for (int p3 = 0; p3 < pairs; ++p3)
                                {
                                    float s0 = s_cur[p3], c0 = c_cur[p3];
                                    float sd = sin_delta[p3], cd = cos_delta[p3];
                                    float sn = s0 * cd + c0 * sd;
                                    float cs = c0 * cd - s0 * sd;
                                    s_cur[p3] = sn;
                                    c_cur[p3] = cs;
                                }
                            }
                        }
                    };
                    auto t0 = std::chrono::high_resolution_clock::now();
                    run_recur(q_copy, k_copy);
                    auto t1 = std::chrono::high_resolution_clock::now();
                    run_legacy(q_copy, k_copy); // reuse buffers; sequence independent for timing delta
                    auto t2 = std::chrono::high_resolution_clock::now();
                    double recur_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
                    double legacy_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
                    bool faster = recur_ms <= legacy_ms * 0.995; // require at least slight win (0.5%) to choose recurrence
                    if (env.prim_rope_trace)
                    {
                        fprintf(stderr, "[RoPE][tune] h=%d d=%d s=%d recur=%g legacy=%g choose=%s\n", heads, head_dim, seq_len, recur_ms, legacy_ms, faster ? "recurrence" : "legacy");
                    }
                    if (faster)
                        use_recurrence = true;
                    else
                        use_recurrence = false;
                }
                g_choice.emplace(key, use_recurrence);
            }
            else
            {
                use_recurrence = it->second;
            }
        }

        if (use_recurrence)
        {
            std::vector<float> sin_base(pairs), cos_base(pairs), sin_delta(pairs), cos_delta(pairs);
            for (int p = 0; p < pairs; ++p)
            {
                float base_angle = (float)n_past * theta[p];
#if defined(LLAMINAR_HAVE_LIBMVEC)
                if (env.prim_rope_fused_sincos)
                {
                    float s0, c0;
                    ::sincosf(base_angle, &s0, &c0);
                    sin_base[p] = s0;
                    cos_base[p] = c0;
                }
                else
                {
                    sin_base[p] = std::sin(base_angle);
                    cos_base[p] = std::cos(base_angle);
                }
#else
                sin_base[p] = std::sin(base_angle);
                cos_base[p] = std::cos(base_angle);
#endif
                float delta = theta[p];
#if defined(LLAMINAR_HAVE_LIBMVEC)
                if (env.prim_rope_fused_sincos)
                {
                    float sd, cd;
                    ::sincosf(delta, &sd, &cd);
                    sin_delta[p] = sd;
                    cos_delta[p] = cd;
                }
                else
                {
                    sin_delta[p] = std::sin(delta);
                    cos_delta[p] = std::cos(delta);
                }
#else
                sin_delta[p] = std::sin(delta);
                cos_delta[p] = std::cos(delta);
#endif
            }

            struct RopeCounters
            {
                size_t rotations = 0;
                size_t recurrence_updates = 0;
            } global_ctrs{0, 0};
#pragma omp parallel if ((size_t)heads * seq_len * (size_t)pairs > 4096 && !env.prim_force_scalar)
            {
                std::vector<float> s_cur(sin_base), c_cur(cos_base);
                RopeCounters local{0, 0};
#pragma omp for nowait
                for (int h = 0; h < heads; ++h)
                {
                    for (int t = 0; t < seq_len; ++t)
                    {
                        float *qh = q + t * heads * head_dim + h * head_dim;
                        float *kh = k + t * heads * head_dim + h * head_dim;
                        int p = 0;
#if defined(__AVX512F__)
                        if (env.prim_rope_vectorize && head_dim >= 64)
                        {
                            // Vectorized complex rotate of 16 pairs (32 floats) at a time.
                            for (; p + 16 <= pairs; p += 16)
                            {
                                int base = 2 * p; // starting float index
                                if (base + 32 > head_dim)
                                    break; // safety
                                // Load 32 q and 32 k floats (interleaved pairs). Process in two 16-float chunks for AVX512 gather simplicity.
                                // We separate even (real) and odd (imag) components using permutexvar.
                                __m512 qblock = _mm512_loadu_ps(qh + base);       // q0 q1 q2 q3 ... q31
                                __m512 qblock2 = _mm512_loadu_ps(qh + base + 16); // q32 .. q63 (may overshoot but guarded by head_dim >=64 assumption for typical head_dim multiple)
                                __m512 kblock = _mm512_loadu_ps(kh + base);
                                __m512 kblock2 = _mm512_loadu_ps(kh + base + 16);

                                // Index vectors for even / odd extraction inside each 16-float lane segment
                                static const int idx_even_arr[16] = {0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0};
                                static const int idx_odd_arr[16] = {1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0};
                                __m512i idx_even = _mm512_loadu_si512((const void *)idx_even_arr);
                                __m512i idx_odd = _mm512_loadu_si512((const void *)idx_odd_arr);

                                __m512 q_even = _mm512_permutexvar_ps(idx_even, qblock); // first 8 valid lanes
                                __m512 q_odd = _mm512_permutexvar_ps(idx_odd, qblock);
                                __m512 k_even = _mm512_permutexvar_ps(idx_even, kblock);
                                __m512 k_odd = _mm512_permutexvar_ps(idx_odd, kblock);

                                // Load sin/cos for 16 pairs (only first 8 used in this block, but we process 16 to keep logic simple)
                                __m512 sn = _mm512_loadu_ps(s_cur.data() + p); // s_cur[p..p+15]
                                __m512 cs = _mm512_loadu_ps(c_cur.data() + p);

                                // new_q_even = q_even * cs - q_odd * sn
                                __m512 new_q_even = _mm512_fmsub_ps(q_even, cs, _mm512_mul_ps(q_odd, sn));
                                // new_q_odd  = q_even * sn + q_odd * cs
                                __m512 new_q_odd = _mm512_fmadd_ps(q_even, sn, _mm512_mul_ps(q_odd, cs));
                                __m512 new_k_even = _mm512_fmsub_ps(k_even, cs, _mm512_mul_ps(k_odd, sn));
                                __m512 new_k_odd = _mm512_fmadd_ps(k_even, sn, _mm512_mul_ps(k_odd, cs));

                                // Interleave back: we write the first 16 floats (8 pairs). We'll fallback to scalar for remainder anyway.
                                float tmp_q[16];
                                float tmp_k[16];
                                _mm512_storeu_ps(tmp_q, new_q_even);
                                float tmp_qi[16];
                                _mm512_storeu_ps(tmp_qi, new_q_odd);
                                _mm512_storeu_ps(tmp_k, new_k_even);
                                float tmp_ki[16];
                                _mm512_storeu_ps(tmp_ki, new_k_odd);
                                for (int lane = 0; lane < 8; ++lane)
                                {
                                    int i0 = base + 2 * lane;
                                    int i1 = i0 + 1;
                                    if (i1 >= head_dim)
                                        break;
                                    qh[i0] = tmp_q[lane];
                                    qh[i1] = tmp_qi[lane];
                                    kh[i0] = tmp_k[lane];
                                    kh[i1] = tmp_ki[lane];
                                    local.rotations++;
                                }
                            }
                        }
#elif defined(__AVX2__)
                        if (env.prim_rope_vectorize && head_dim >= 32)
                        {
                            for (; p + 8 <= pairs; p += 8)
                            {
                                int base = 2 * p;
                                if (base + 16 > head_dim)
                                    break; // ensure full 8 pairs present
                                float csf[8];
                                float snf[8];
                                for (int lane = 0; lane < 8; ++lane)
                                {
                                    csf[lane] = c_cur[p + lane];
                                    snf[lane] = s_cur[p + lane];
                                }
                                // Gather real/imag parts into contiguous arrays
                                float qr[8], qi[8], kr[8], ki[8];
                                bool partial = false;
                                for (int lane = 0; lane < 8; ++lane)
                                {
                                    int i0 = base + 2 * lane;
                                    int i1 = i0 + 1;
                                    if (i1 >= head_dim)
                                    {
                                        partial = true;
                                        break;
                                    }
                                    qr[lane] = qh[i0];
                                    qi[lane] = qh[i1];
                                    kr[lane] = kh[i0];
                                    ki[lane] = kh[i1];
                                }
                                if (partial)
                                    break; // fallback to scalar remainder below if tail
                                __m256 v_qr = _mm256_loadu_ps(qr);
                                __m256 v_qi = _mm256_loadu_ps(qi);
                                __m256 v_kr = _mm256_loadu_ps(kr);
                                __m256 v_ki = _mm256_loadu_ps(ki);
                                __m256 v_cs = _mm256_loadu_ps(csf);
                                __m256 v_sn = _mm256_loadu_ps(snf);
                                // new_real = real*cs - imag*sn ; new_imag = real*sn + imag*cs
                                __m256 new_qr = _mm256_fmsub_ps(v_qr, v_cs, _mm256_mul_ps(v_qi, v_sn));
                                __m256 new_qi = _mm256_fmadd_ps(v_qr, v_sn, _mm256_mul_ps(v_qi, v_cs));
                                __m256 new_kr = _mm256_fmsub_ps(v_kr, v_cs, _mm256_mul_ps(v_ki, v_sn));
                                __m256 new_ki = _mm256_fmadd_ps(v_kr, v_sn, _mm256_mul_ps(v_ki, v_cs));
                                _mm256_storeu_ps(qr, new_qr);
                                _mm256_storeu_ps(qi, new_qi);
                                _mm256_storeu_ps(kr, new_kr);
                                _mm256_storeu_ps(ki, new_ki);
                                for (int lane = 0; lane < 8; ++lane)
                                {
                                    int i0 = base + 2 * lane;
                                    int i1 = i0 + 1;
                                    if (i1 >= head_dim)
                                        break;
                                    qh[i0] = qr[lane];
                                    qh[i1] = qi[lane];
                                    kh[i0] = kr[lane];
                                    kh[i1] = ki[lane];
                                }
                                local.rotations += 8;
                            }
                        }
#endif
                        for (; p < pairs; ++p)
                        {
                            int i0 = 2 * p, i1 = i0 + 1;
                            if (i1 >= head_dim)
                                break;
                            float q0 = qh[i0], q1 = qh[i1];
                            float k0 = kh[i0], k1 = kh[i1];
                            float cs = c_cur[p], sn = s_cur[p];
                            qh[i0] = q0 * cs - q1 * sn;
                            qh[i1] = q0 * sn + q1 * cs;
                            kh[i0] = k0 * cs - k1 * sn;
                            kh[i1] = k0 * sn + k1 * cs;
                            local.rotations++;
                        }
                        // Advance sin/cos to next position
                        int p2 = 0;
#if defined(__AVX512F__)
                        for (; p2 + 16 <= pairs; p2 += 16)
                        {
                            __m512 s0 = _mm512_loadu_ps(s_cur.data() + p2);
                            __m512 c0 = _mm512_loadu_ps(c_cur.data() + p2);
                            __m512 sd = _mm512_loadu_ps(sin_delta.data() + p2);
                            __m512 cd = _mm512_loadu_ps(cos_delta.data() + p2);
                            __m512 sn = _mm512_fmadd_ps(c0, sd, _mm512_mul_ps(s0, cd));
                            __m512 cs = _mm512_fnmadd_ps(s0, sd, _mm512_mul_ps(c0, cd));
                            _mm512_storeu_ps(s_cur.data() + p2, sn);
                            _mm512_storeu_ps(c_cur.data() + p2, cs);
                            local.recurrence_updates += 16;
                        }
#elif defined(__AVX2__)
                        for (; p2 + 8 <= pairs; p2 += 8)
                        {
                            __m256 s0 = _mm256_loadu_ps(s_cur.data() + p2);
                            __m256 c0 = _mm256_loadu_ps(c_cur.data() + p2);
                            __m256 sd = _mm256_loadu_ps(sin_delta.data() + p2);
                            __m256 cd = _mm256_loadu_ps(cos_delta.data() + p2);
                            __m256 sn = _mm256_fmadd_ps(c0, sd, _mm256_mul_ps(s0, cd));
                            __m256 cs = _mm256_fnmadd_ps(s0, sd, _mm256_mul_ps(c0, cd));
                            _mm256_storeu_ps(s_cur.data() + p2, sn);
                            _mm256_storeu_ps(c_cur.data() + p2, cs);
                            local.recurrence_updates += 8;
                        }
#endif
                        for (; p2 < pairs; ++p2)
                        {
                            float s0 = s_cur[p2], c0 = c_cur[p2];
                            float sd = sin_delta[p2], cd = cos_delta[p2];
                            float sn = s0 * cd + c0 * sd;
                            float cs = c0 * cd - s0 * sd;
                            s_cur[p2] = sn;
                            c_cur[p2] = cs;
                            local.recurrence_updates++;
                        }
                    }
                }
#pragma omp critical
                {
                    global_ctrs.rotations += local.rotations;
                    global_ctrs.recurrence_updates += local.recurrence_updates;
                }
            }
            if (env.prim_rope_trace)
            {
                fprintf(stderr, "[RoPE] recurrence elems=%zu rotations=%zu recurrence_updates=%zu updates_per_rotation=%.3f\n",
                        total_elems, global_ctrs.rotations, global_ctrs.recurrence_updates,
                        global_ctrs.rotations ? (double)global_ctrs.recurrence_updates / (double)global_ctrs.rotations : 0.0);
            }
            return;
        }

#pragma omp parallel for collapse(2) if (seq_len * heads * pairs > 4096 && !env.prim_force_scalar)
        for (int h = 0; h < heads; ++h)
        {
            for (int t = 0; t < seq_len; ++t)
            {
                float *qh = q + t * heads * head_dim + h * head_dim;
                float *kh = k + t * heads * head_dim + h * head_dim;
                float pos = float(n_past + t);
                int p = 0;
#if defined(__AVX512F__)
                if (env.prim_rope_vectorize)
                {
                    for (; p + 16 <= pairs; p += 16)
                    {
                        __m512 th = _mm512_loadu_ps(theta.data() + p);
                        __m512 ang = _mm512_mul_ps(_mm512_set1_ps(pos), th);
                        alignas(64) float angf[16];
                        _mm512_store_ps(angf, ang);
                        float csf[16], snf[16];
#if defined(LLAMINAR_HAVE_LIBMVEC)
                        if (env.prim_rope_fused_sincos)
                        {
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                float s, c;
                                ::sincosf(angf[lane], &s, &c);
                                snf[lane] = s;
                                csf[lane] = c;
                            }
                        }
                        else
                        {
                            for (int lane = 0; lane < 16; ++lane)
                            {
                                csf[lane] = std::cos(angf[lane]);
                                snf[lane] = std::sin(angf[lane]);
                            }
                        }
#else
                        for (int lane = 0; lane < 16; ++lane)
                        {
                            csf[lane] = std::cos(angf[lane]);
                            snf[lane] = std::sin(angf[lane]);
                        }
#endif
                        for (int lane = 0; lane < 16; ++lane)
                        {
                            int i0 = 2 * (p + lane), i1 = i0 + 1;
                            if (i1 >= head_dim)
                                break;
                            float q0 = qh[i0], q1 = qh[i1];
                            float k0 = kh[i0], k1 = kh[i1];
                            float cs = csf[lane], sn = snf[lane];
                            qh[i0] = q0 * cs - q1 * sn;
                            qh[i1] = q0 * sn + q1 * cs;
                            kh[i0] = k0 * cs - k1 * sn;
                            kh[i1] = k0 * sn + k1 * cs;
                        }
                    }
                }
#elif defined(__AVX2__)
                if (env.prim_rope_vectorize)
                {
                    for (; p + 8 <= pairs; p += 8)
                    {
                        __m256 th = _mm256_loadu_ps(theta.data() + p);
                        __m256 ang = _mm256_mul_ps(_mm256_set1_ps(pos), th);
                        float angf[8];
                        _mm256_storeu_ps(angf, ang);
                        float csf[8], snf[8];
#if defined(LLAMINAR_HAVE_LIBMVEC)
                        if (env.prim_rope_fused_sincos)
                        {
                            for (int lane = 0; lane < 8; ++lane)
                            {
                                float s, c;
                                ::sincosf(angf[lane], &s, &c);
                                snf[lane] = s;
                                csf[lane] = c;
                            }
                        }
                        else
                        {
                            for (int lane = 0; lane < 8; ++lane)
                            {
                                csf[lane] = std::cos(angf[lane]);
                                snf[lane] = std::sin(angf[lane]);
                            }
                        }
#else
                        for (int lane = 0; lane < 8; ++lane)
                        {
                            csf[lane] = std::cos(angf[lane]);
                            snf[lane] = std::sin(angf[lane]);
                        }
#endif
                        for (int lane = 0; lane < 8; ++lane)
                        {
                            int i0 = 2 * (p + lane), i1 = i0 + 1;
                            if (i1 >= head_dim)
                                break;
                            float q0 = qh[i0], q1 = qh[i1];
                            float k0 = kh[i0], k1 = kh[i1];
                            float cs = csf[lane], sn = snf[lane];
                            qh[i0] = q0 * cs - q1 * sn;
                            qh[i1] = q0 * sn + q1 * cs;
                            kh[i0] = k0 * cs - k1 * sn;
                            kh[i1] = k0 * sn + k1 * cs;
                        }
                    }
                }
#endif
                for (; p < pairs; ++p)
                {
                    float angle = pos * theta[p];
#if defined(LLAMINAR_HAVE_LIBMVEC)
                    float sn, cs;
                    if (env.prim_rope_fused_sincos)
                    {
                        ::sincosf(angle, &sn, &cs);
                    }
                    else
                    {
                        cs = std::cos(angle);
                        sn = std::sin(angle);
                    }
#else
                    float cs = std::cos(angle), sn = std::sin(angle);
#endif
                    int i0 = 2 * p, i1 = i0 + 1;
                    if (i1 >= head_dim)
                        break;
                    float q0 = qh[i0], q1 = qh[i1];
                    float k0 = kh[i0], k1 = kh[i1];
                    qh[i0] = q0 * cs - q1 * sn;
                    qh[i1] = q0 * sn + q1 * cs;
                    kh[i0] = k0 * cs - k1 * sn;
                    kh[i1] = k0 * sn + k1 * cs;
                }
            }
        }
        if (diag && seq_len > 0 && preview > 0)
        {
            size_t row_offset = (size_t)(seq_len - 1) * heads * head_dim;
            std::array<float, 8> q_after{};
            std::array<float, 8> k_after{};
            q_after.fill(0.f);
            k_after.fill(0.f);
            for (int i = 0; i < preview; ++i)
            {
                q_after[i] = q[row_offset + i];
                k_after[i] = k[row_offset + i];
            }
            long double l2b = 0, l2d = 0;
            for (int i = 0; i < preview; ++i)
            {
                long double b = q_before[i];
                long double d = q_after[i] - b;
                l2b += b * b;
                l2d += d * d;
            }
            double rel_move = (l2b > 0) ? std::sqrt((double)l2d / (double)l2b) : 0.0;
            int pos_last = n_past + seq_len - 1;
            float theta0 = (pairs > 0) ? theta[0] : 0.f;
            float angle0 = theta0 * pos_last;
            float cs = std::cos(angle0), sn = std::sin(angle0);

            // Rotation correctness classification (only emit once per mode to reduce noise)
            bool is_prefill = (n_past == 0 && seq_len > 1);
            bool is_incremental = (n_past > 0 && seq_len == 1);
            static bool emitted_prefill = false;
            static bool emitted_incremental = false;
            std::string classification = "skipped";
            double best_rel_err = 0.0;
            if ((is_prefill && !emitted_prefill) || (is_incremental && !emitted_incremental))
            {
                // Use first complex pair (indices 0,1) if available.
                if (head_dim >= 2)
                {
                    float qb0 = q_before[0], qb1 = q_before[1];
                    // Expected single rotation
                    float exp_q0 = qb0 * cs - qb1 * sn;
                    float exp_q1 = qb0 * sn + qb1 * cs;
                    float qa0 = q_after[0], qa1 = q_after[1];
                    auto rel_err = [&](float a0, float a1)
                    {
                        double nb0 = (double)exp_q0; double nb1=(double)exp_q1; double da0=a0-nb0; double da1=a1-nb1; double denom = nb0*nb0+nb1*nb1; if(denom==0.0) denom=1.0; return std::sqrt( (da0*da0+da1*da1)/denom ); };
                    double err_expected = rel_err(qa0, qa1);
                    // Off-by-one angle checks
                    float angle_plus = theta0 * (pos_last + 1);
                    float csp = std::cos(angle_plus), snp = std::sin(angle_plus);
                    float exp_p0 = qb0 * csp - qb1 * snp;
                    float exp_p1 = qb0 * snp + qb1 * csp;
                    double err_plus = 0.0;
                    {
                        double nb0 = (double)exp_p0;
                        double nb1 = (double)exp_p1;
                        double da0 = qa0 - nb0;
                        double da1 = qa1 - nb1;
                        double denom = nb0 * nb0 + nb1 * nb1;
                        if (denom == 0.0)
                            denom = 1.0;
                        err_plus = std::sqrt((da0 * da0 + da1 * da1) / denom);
                    }
                    float angle_minus = theta0 * (pos_last - 1);
                    float csm = std::cos(angle_minus), snm = std::sin(angle_minus);
                    float exp_m0 = qb0 * csm - qb1 * snm;
                    float exp_m1 = qb0 * snm + qb1 * csm;
                    double err_minus = 0.0;
                    {
                        double nb0 = (double)exp_m0;
                        double nb1 = (double)exp_m1;
                        double da0 = qa0 - nb0;
                        double da1 = qa1 - nb1;
                        double denom = nb0 * nb0 + nb1 * nb1;
                        if (denom == 0.0)
                            denom = 1.0;
                        err_minus = std::sqrt((da0 * da0 + da1 * da1) / denom);
                    }
                    // Double rotation (angle *2)
                    float angle2 = angle0 * 2.f;
                    float cs2 = std::cos(angle2), sn2 = std::sin(angle2);
                    float exp2_q0 = qb0 * cs2 - qb1 * sn2;
                    float exp2_q1 = qb0 * sn2 + qb1 * cs2;
                    double err_double = 0.0;
                    {
                        double nb0 = (double)exp2_q0;
                        double nb1 = (double)exp2_q1;
                        double da0 = qa0 - nb0;
                        double da1 = qa1 - nb1;
                        double denom = nb0 * nb0 + nb1 * nb1;
                        if (denom == 0.0)
                            denom = 1.0;
                        err_double = std::sqrt((da0 * da0 + da1 * da1) / denom);
                    }
                    // Select classification by smallest error
                    best_rel_err = err_expected;
                    classification = "expected";
                    double best = err_expected;
                    auto consider = [&](double err, const char *tag)
                    { if(err < best*0.8 && err < best){ best=err; best_rel_err=err; classification=tag; } };
                    consider(err_double, "double_rotation");
                    consider(err_plus, "off_by_one_plus");
                    consider(err_minus, "off_by_one_minus");
                }
                if (is_prefill)
                    emitted_prefill = true;
                if (is_incremental)
                    emitted_incremental = true;
            }

            // Stream directly to logger (previous string log appeared empty in some builds)
            LOG_INFO("[RoPEDiagPrim] seq_len=" << seq_len << " heads=" << heads << " head_dim=" << head_dim
                                               << " n_past=" << n_past << " pos_last=" << pos_last << " theta0=" << std::setprecision(6) << theta0
                                               << " angle0=" << angle0 << " cos0=" << cs << " sin0=" << sn
                                               << " rel_move_q=" << rel_move << " classification=" << classification << " class_rel_err=" << best_rel_err);
            if (head_dim >= 2)
            {
                // Emit per-hypothesis relative errors for first complex pair (Q)
                float qb0 = q_before[0], qb1 = q_before[1];
                float qa0 = q_after[0], qa1 = q_after[1];
                auto rel_err_pair = [&](float e0, float e1)
                { double nb0=e0, nb1=e1; double da0=qa0-nb0; double da1=qa1-nb1; double denom=nb0*nb0+nb1*nb1; if(denom==0.0) denom=1.0; return std::sqrt((da0*da0+da1*da1)/denom); };
                float angle_plus = theta0 * (pos_last + 1);
                float angle_minus = theta0 * (pos_last - 1);
                float angle2 = angle0 * 2.f;
                float cs_plus = std::cos(angle_plus), sn_plus = std::sin(angle_plus);
                float cs_minus = std::cos(angle_minus), sn_minus = std::sin(angle_minus);
                float cs2 = std::cos(angle2), sn2 = std::sin(angle2);
                float exp_q0 = qb0 * cs - qb1 * sn;
                float exp_q1 = qb0 * sn + qb1 * cs;
                float exp_p0 = qb0 * cs_plus - qb1 * sn_plus;
                float exp_p1 = qb0 * sn_plus + qb1 * cs_plus;
                float exp_m0 = qb0 * cs_minus - qb1 * sn_minus;
                float exp_m1 = qb0 * sn_minus + qb1 * cs_minus;
                float exp2_q0 = qb0 * cs2 - qb1 * sn2;
                float exp2_q1 = qb0 * sn2 + qb1 * cs2;
                double err_exp = rel_err_pair(exp_q0, exp_q1);
                double err_plus = rel_err_pair(exp_p0, exp_p1);
                double err_minus = rel_err_pair(exp_m0, exp_m1);
                double err_double = rel_err_pair(exp2_q0, exp2_q1);
                LOG_INFO("[RoPEDiagPrimDetail] pair=q classification=" << classification
                                                                       << " err_expected=" << err_exp << " err_double=" << err_double
                                                                       << " err_plus=" << err_plus << " err_minus=" << err_minus
                                                                       << " q_before_pair=" << qb0 << "," << qb1 << " q_after_pair=" << qa0 << "," << qa1);
            }
        }
    }

    void compute_qk_scores(const float *q, const float *k, float *scores, int seq_len,
                           int head_dim, int heads, bool causal, bool apply_softmax)
    {
        const auto &env = llaminar::debugEnv().attention;
        const float scale = 1.f / std::sqrt((float)head_dim);
        std::size_t work = (std::size_t)heads * seq_len;
        bool parallel = !env.prim_force_scalar && work * (std::size_t)seq_len * head_dim >= (std::size_t)env.prim_parallel_elems_threshold;
#pragma omp parallel for if (parallel)
        for (long long h_i = 0; h_i < (long long)heads * seq_len; ++h_i)
        {
            int h = (int)(h_i / seq_len);
            int i = (int)(h_i % seq_len);
            float *row = head_row(scores, h, i, seq_len);
            const float *qi = q + (std::size_t)i * heads * head_dim + (std::size_t)h * head_dim;
            for (int j = 0; j < seq_len; ++j)
            {
                if (causal && j > i)
                {
                    row[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const float *kj = k + (std::size_t)j * heads * head_dim + (std::size_t)h * head_dim;
                float dot = 0.f;
                int d = 0;
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                for (; d + 16 <= head_dim; d += 16)
                {
                    __m512 qv = _mm512_loadu_ps(qi + d);
                    __m512 kv = _mm512_loadu_ps(kj + d);
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                }
                alignas(64) float buf[16];
                _mm512_store_ps(buf, acc);
                for (int t = 0; t < 16; ++t)
                    dot += buf[t];
#elif defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (; d + 8 <= head_dim; d += 8)
                {
                    __m256 qv = _mm256_loadu_ps(qi + d);
                    __m256 kv = _mm256_loadu_ps(kj + d);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float buf[8];
                _mm256_storeu_ps(buf, acc);
                for (int t = 0; t < 8; ++t)
                    dot += buf[t];
#endif
                for (; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                row[j] = dot * scale;
            }
        }
        if (!apply_softmax)
            return;
        for (int h = 0; h < heads; ++h)
        {
            llaminar::kernels::SoftmaxRowArgs a;
            a.scores = scores + (std::size_t)h * seq_len * seq_len;
            a.rows = seq_len;
            a.cols = seq_len;
            a.causal = causal;
            a.scale = 1.0f;
            llaminar::kernels::softmax_row_major(a);
        }
    }

    void apply_scores_to_v(const float *scores, const float *v, float *out, int seq_len, int head_dim, int heads)
    {
        const auto &env = llaminar::debugEnv().attention;
        std::fill(out, out + (size_t)seq_len * heads * head_dim, 0.f);
        std::size_t work = (std::size_t)heads * seq_len;
        bool parallel = !env.prim_force_scalar && work * (std::size_t)head_dim >= (std::size_t)env.prim_parallel_elems_threshold;
#pragma omp parallel for if (parallel)
        for (long long h_i = 0; h_i < (long long)heads * seq_len; ++h_i)
        {
            int h = (int)(h_i / seq_len);
            int i = (int)(h_i % seq_len);
            float *dst = out + i * heads * head_dim + h * head_dim;
            const float *score_row = head_row(scores, h, i, seq_len);
            for (int j = 0; j < seq_len; ++j)
            {
                float w = score_row[j];
                const float *vj = v + j * heads * head_dim + h * head_dim;
                int d = 0;
#if defined(__AVX512F__)
                __m512 wv = _mm512_set1_ps(w);
                for (; d + 16 <= head_dim; d += 16)
                {
                    __m512 acc = _mm512_loadu_ps(dst + d);
                    __m512 vv = _mm512_loadu_ps(vj + d);
                    acc = _mm512_fmadd_ps(wv, vv, acc);
                    _mm512_storeu_ps(dst + d, acc);
                }
#elif defined(__AVX2__)
                __m256 wv = _mm256_set1_ps(w);
                for (; d + 8 <= head_dim; d += 8)
                {
                    __m256 acc = _mm256_loadu_ps(dst + d);
                    __m256 vv = _mm256_loadu_ps(vj + d);
                    acc = _mm256_fmadd_ps(wv, vv, acc);
                    _mm256_storeu_ps(dst + d, acc);
                }
#endif
                for (; d < head_dim; ++d)
                    dst[d] += w * vj[d];
            }
        }
    }

    // Fused recompute attention: two-pass softmax + accumulation without materializing full scores (optional)
    static void fused_attention_recompute(const float *q, const float *k, const float *v, float *out,
                                          int seq_len, int head_dim, int heads, bool causal)
    {
        const float scale = 1.f / std::sqrt((float)head_dim);
        std::fill(out, out + (size_t)seq_len * heads * head_dim, 0.f);
        // Temporary buffer for denominators per (h,i)
        std::vector<float> row_max((size_t)heads * seq_len, -std::numeric_limits<float>::infinity());
        std::vector<double> denom((size_t)heads * seq_len, 0.0);
        const auto process_row_max = [&](int h, int i)
        {
            const float *qi = q + i * heads * head_dim + h * head_dim;
            float m = -std::numeric_limits<float>::infinity();
            for (int j=0;j<seq_len;++j){ if(causal && j>i) break; const float *kj = k + j * heads * head_dim + h * head_dim; float dot=0.f; int d=0;
#if defined(__AVX512F__)
                __m512 acc=_mm512_setzero_ps(); for(; d+16<=head_dim; d+=16){ __m512 qv=_mm512_loadu_ps(qi+d); __m512 kv=_mm512_loadu_ps(kj+d); acc=_mm512_fmadd_ps(qv,kv,acc);} alignas(64) float buf[16]; _mm512_store_ps(buf,acc); for(int t=0;t<16;++t) dot+=buf[t];
#elif defined(__AVX2__)
                __m256 acc=_mm256_setzero_ps(); for(; d+8<=head_dim; d+=8){ __m256 qv=_mm256_loadu_ps(qi+d); __m256 kv=_mm256_loadu_ps(kj+d); acc=_mm256_fmadd_ps(qv,kv,acc);} float buf[8]; _mm256_storeu_ps(buf,acc); for(int t=0;t<8;++t) dot+=buf[t];
#endif
                for(; d<head_dim; ++d) dot += qi[d]*kj[d];
                float val = dot*scale; if(val>m) m=val; }
            row_max[h*seq_len + i] = std::isfinite(m)? m:0.f; };
        const auto process_denom_accum = [&](int h, int i)
        {
            const float *qi = q + i * heads * head_dim + h * head_dim;
            float m = row_max[h * seq_len + i];
            double s = 0.0;
            float *dst = out + i * heads * head_dim + h * head_dim;
            for (int j = 0; j < seq_len; ++j)
            {
                if (causal && j > i)
                    break;
                const float *kj = k + j * heads * head_dim + h * head_dim;
                float dot = 0.f;
                int d = 0;
#if defined(__AVX512F__)
                __m512 acc = _mm512_setzero_ps();
                for (; d + 16 <= head_dim; d += 16)
                {
                    __m512 qv = _mm512_loadu_ps(qi + d);
                    __m512 kv = _mm512_loadu_ps(kj + d);
                    acc = _mm512_fmadd_ps(qv, kv, acc);
                }
                alignas(64) float buf[16];
                _mm512_store_ps(buf, acc);
                for (int t = 0; t < 16; ++t)
                    dot += buf[t];
#elif defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (; d + 8 <= head_dim; d += 8)
                {
                    __m256 qv = _mm256_loadu_ps(qi + d);
                    __m256 kv = _mm256_loadu_ps(kj + d);
                    acc = _mm256_fmadd_ps(qv, kv, acc);
                }
                float buf[8];
                _mm256_storeu_ps(buf, acc);
                for (int t = 0; t < 8; ++t)
                    dot += buf[t];
#endif
                for (; d < head_dim; ++d)
                    dot += qi[d] * kj[d];
                float val = dot * scale;
                float w = std::exp(val - m);
                s += w; // accumulate weight
                // dst += w * v_j
                const float *vj = v + j * heads * head_dim + h * head_dim;
                int dd = 0;
#if defined(__AVX512F__)
                __m512 wv = _mm512_set1_ps(w);
                for (; dd + 16 <= head_dim; dd += 16)
                {
                    __m512 accv = _mm512_loadu_ps(dst + dd);
                    __m512 vv = _mm512_loadu_ps(vj + dd);
                    accv = _mm512_fmadd_ps(wv, vv, accv);
                    _mm512_storeu_ps(dst + dd, accv);
                }
#elif defined(__AVX2__)
                __m256 wv = _mm256_set1_ps(w);
                for (; dd + 8 <= head_dim; dd += 8)
                {
                    __m256 accv = _mm256_loadu_ps(dst + dd);
                    __m256 vv = _mm256_loadu_ps(vj + dd);
                    accv = _mm256_fmadd_ps(wv, vv, accv);
                    _mm256_storeu_ps(dst + dd, accv);
                }
#endif
                for (; dd < head_dim; ++dd)
                    dst[dd] += w * vj[dd];
            }
            denom[h * seq_len + i] = s;
        };

        // Parallel over heads*rows in both passes
        std::size_t work = (size_t)heads * seq_len;
        bool parallel = work * (size_t)seq_len >= 32768; // TODO: expose
#pragma omp parallel for if (parallel)
        for (long long h_i = 0; h_i < (long long)heads * seq_len; ++h_i)
        {
            int h = (int)(h_i / seq_len);
            int i = (int)(h_i % seq_len);
            process_row_max(h, i);
        }
#pragma omp parallel for if (parallel)
        for (long long h_i = 0; h_i < (long long)heads * seq_len; ++h_i)
        {
            int h = (int)(h_i / seq_len);
            int i = (int)(h_i % seq_len);
            process_denom_accum(h, i);
        }

        // Normalize outputs
#pragma omp parallel for if (parallel)
        for (long long h_i = 0; h_i < (long long)heads * seq_len; ++h_i)
        {
            int h = (int)(h_i / seq_len);
            int i = (int)(h_i % seq_len);
            double s = denom[h * seq_len + i];
            if (s <= 0.0)
                s = 1.0;
            float inv = (float)(1.0 / s);
            float *dst = out + i * heads * head_dim + h * head_dim;
            int d = 0;
#if defined(__AVX512F__)
            __m512 invv = _mm512_set1_ps(inv);
            for (; d + 16 <= head_dim; d += 16)
            {
                __m512 vcur = _mm512_loadu_ps(dst + d);
                vcur = _mm512_mul_ps(vcur, invv);
                _mm512_storeu_ps(dst + d, vcur);
            }
#elif defined(__AVX2__)
            __m256 invv = _mm256_set1_ps(inv);
            for (; d + 8 <= head_dim; d += 8)
            {
                __m256 vcur = _mm256_loadu_ps(dst + d);
                vcur = _mm256_mul_ps(vcur, invv);
                _mm256_storeu_ps(dst + d, vcur);
            }
#endif
            for (; d < head_dim; ++d)
                dst[d] *= inv;
        }
    }

    void fused_attention(const float *q, const float *k, const float *v, float *out, int seq_len, int head_dim, int heads, bool causal)
    {
        const auto &env = llaminar::debugEnv().attention;
        bool allow_fused = !env.prim_disable_fused;
        bool force_fused = env.prim_force_fused;
        bool large = env.prim_fused_recompute_threshold > 0 && seq_len >= env.prim_fused_recompute_threshold;
        if ((force_fused || (allow_fused && large)) && !force_fused && env.prim_disable_fused)
        {
            // conflicting flags; fallback to baseline
            large = false;
        }
        if (force_fused || (allow_fused && large))
        {
            fused_attention_recompute(q, k, v, out, seq_len, head_dim, heads, causal);
            return;
        }
        std::vector<float> scores((size_t)heads * seq_len * seq_len);
        compute_qk_scores(q, k, scores.data(), seq_len, head_dim, heads, causal, true);
        apply_scores_to_v(scores.data(), v, out, seq_len, head_dim, heads);
    }

    RowSoftmaxStats validate_softmax_rows(const float *scores, int seq_len, int heads)
    {
        RowSoftmaxStats st{0.f, 0.f, 0.f};
        if (seq_len <= 0 || heads <= 0)
            return st;

        const auto &env = llaminar::debugEnv().softmax;
        const long long total_rows = (long long)heads * seq_len;
        const long long scalar_threshold = (long long)env.validate_scalar_row_threshold;
        if (total_rows < scalar_threshold)
        {
            for (int h = 0; h < heads; ++h)
            {
                for (int i = 0; i < seq_len; ++i)
                {
                    const float *row = head_row(scores, h, i, seq_len);
                    float sum = 0.f;
                    float row_min = 0.f;
                    float row_max = 0.f;
                    for (int j = 0; j < seq_len; ++j)
                    {
                        float v = row[j];
                        sum += v;
                        row_min = std::min(row_min, v);
                        row_max = std::max(row_max, v);
                    }
                    st.max_row_deviation = std::max(st.max_row_deviation, std::fabs(sum - 1.f));
                    st.max_negative = std::min(st.max_negative, row_min);
                    st.max_prob = std::max(st.max_prob, row_max);
                }
            }
            return st;
        }

        int max_threads = 1;
#ifdef _OPENMP
        max_threads = omp_get_max_threads();
#endif
        std::vector<float> max_dev(max_threads, 0.f);
        std::vector<float> min_val(max_threads, 0.f); // track most negative value
        std::vector<float> max_prob(max_threads, 0.f);

#pragma omp parallel if (max_threads > 1)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            float local_max_dev = 0.f;
            float local_min_val = 0.f;
            float local_max_prob = 0.f;
#pragma omp for schedule(static)
            for (long long r = 0; r < total_rows; ++r)
            {
                int h_idx = (int)(r / seq_len);
                int row_idx = (int)(r % seq_len);
                const float *row = head_row(scores, h_idx, row_idx, seq_len);
                float sum = 0.f;
                float row_min = 0.f;
                float row_max = 0.f;
#if defined(__AVX512F__)
                int j = 0;
                __m512 vsum = _mm512_setzero_ps();
                __m512 vmin = _mm512_setzero_ps(); // semantics: track most negative (start 0)
                __m512 vmax = _mm512_setzero_ps();
                for (; j + 16 <= seq_len; j += 16)
                {
                    __m512 v = _mm512_loadu_ps(row + j);
                    vsum = _mm512_add_ps(vsum, v);
                    vmin = _mm512_min_ps(vmin, v);
                    vmax = _mm512_max_ps(vmax, v);
                }
                alignas(64) float buf_sum[16];
                alignas(64) float buf_min[16];
                alignas(64) float buf_max[16];
                _mm512_store_ps(buf_sum, vsum);
                _mm512_store_ps(buf_min, vmin);
                _mm512_store_ps(buf_max, vmax);
                for (int t = 0; t < 16; ++t)
                {
                    sum += buf_sum[t];
                    row_min = std::min(row_min, buf_min[t]);
                    row_max = std::max(row_max, buf_max[t]);
                }
                // tail
                for (; j < seq_len; ++j)
                {
                    float v = row[j];
                    sum += v;
                    row_min = std::min(row_min, v);
                    row_max = std::max(row_max, v);
                }
#elif defined(__AVX2__)
                int j = 0;
                __m256 vsum = _mm256_setzero_ps();
                __m256 vmin = _mm256_setzero_ps();
                __m256 vmax = _mm256_setzero_ps();
                for (; j + 8 <= seq_len; j += 8)
                {
                    __m256 v = _mm256_loadu_ps(row + j);
                    vsum = _mm256_add_ps(vsum, v);
                    vmin = _mm256_min_ps(vmin, v);
                    vmax = _mm256_max_ps(vmax, v);
                }
                alignas(32) float buf_sum[8];
                alignas(32) float buf_min[8];
                alignas(32) float buf_max[8];
                _mm256_store_ps(buf_sum, vsum);
                _mm256_store_ps(buf_min, vmin);
                _mm256_store_ps(buf_max, vmax);
                for (int t = 0; t < 8; ++t)
                {
                    sum += buf_sum[t];
                    row_min = std::min(row_min, buf_min[t]);
                    row_max = std::max(row_max, buf_max[t]);
                }
                for (; j < seq_len; ++j)
                {
                    float v = row[j];
                    sum += v;
                    row_min = std::min(row_min, v);
                    row_max = std::max(row_max, v);
                }
#else
                for (int j = 0; j < seq_len; ++j)
                {
                    float v = row[j];
                    sum += v;
                    row_min = std::min(row_min, v);
                    row_max = std::max(row_max, v);
                }
#endif
                local_max_dev = std::max(local_max_dev, std::fabs(sum - 1.f));
                local_min_val = std::min(local_min_val, row_min);
                local_max_prob = std::max(local_max_prob, row_max);
            }
            max_dev[tid] = local_max_dev;
            min_val[tid] = local_min_val;
            max_prob[tid] = local_max_prob;
        }

        float g_max_dev = 0.f;
        float g_min_val = 0.f;
        float g_max_prob = 0.f;
        for (int t = 0; t < max_threads; ++t)
        {
            g_max_dev = std::max(g_max_dev, max_dev[t]);
            g_min_val = std::min(g_min_val, min_val[t]);
            g_max_prob = std::max(g_max_prob, max_prob[t]);
        }
        st.max_row_deviation = g_max_dev;
        st.max_negative = g_min_val;
        st.max_prob = g_max_prob;
        return st;
    }

    void expand_kv_for_gqa(
        const float *k_compact,
        const float *v_compact,
        float *k_expanded,
        float *v_expanded,
        int seq_len,
        int head_dim,
        int n_heads,
        int n_kv_heads)
    {
        const int kv_head_dim = n_kv_heads * head_dim;
        const int total_head_dim = n_heads * head_dim;

        // Parallelize over sequence tokens (outer loop)
        // This gives good work distribution for long sequences
#pragma omp parallel for schedule(static)
        for (int row = 0; row < seq_len; ++row)
        {
            const float *k_row_src = k_compact + (size_t)row * kv_head_dim;
            const float *v_row_src = v_compact + (size_t)row * kv_head_dim;
            float *k_row_dst = k_expanded + (size_t)row * total_head_dim;
            float *v_row_dst = v_expanded + (size_t)row * total_head_dim;

            // For each query head, map to corresponding KV head group
            for (int h = 0; h < n_heads; ++h)
            {
                int kv_h = h % n_kv_heads;
                const float *k_src = k_row_src + kv_h * head_dim;
                const float *v_src = v_row_src + kv_h * head_dim;
                float *k_dst = k_row_dst + h * head_dim;
                float *v_dst = v_row_dst + h * head_dim;

                // Use vectorized copy (memcpy is often SIMD-optimized by compiler)
                std::memcpy(k_dst, k_src, head_dim * sizeof(float));
                std::memcpy(v_dst, v_src, head_dim * sizeof(float));
            }
        }
    }

    void expand_kv_for_mha(
        const float *k_compact,
        const float *v_compact,
        float *k_expanded,
        float *v_expanded,
        int seq_len,
        int kv_head_dim,
        int total_head_dim)
    {
        // Simple parallel copy when dimensions differ
        // (typically kv_head_dim < total_head_dim, need zero-padding)
#pragma omp parallel for schedule(static)
        for (int row = 0; row < seq_len; ++row)
        {
            const float *k_src = k_compact + (size_t)row * kv_head_dim;
            const float *v_src = v_compact + (size_t)row * kv_head_dim;
            float *k_dst = k_expanded + (size_t)row * total_head_dim;
            float *v_dst = v_expanded + (size_t)row * total_head_dim;

            std::memcpy(k_dst, k_src, kv_head_dim * sizeof(float));
            std::memcpy(v_dst, v_src, kv_head_dim * sizeof(float));

            // Zero-pad the remaining dimensions if needed
            if (total_head_dim > kv_head_dim)
            {
                std::memset(k_dst + kv_head_dim, 0, (total_head_dim - kv_head_dim) * sizeof(float));
                std::memset(v_dst + kv_head_dim, 0, (total_head_dim - kv_head_dim) * sizeof(float));
            }
        }
    }

} // namespace llaminar::attn
