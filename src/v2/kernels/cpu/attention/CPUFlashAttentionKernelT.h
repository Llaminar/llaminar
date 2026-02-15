#pragma once

#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/OpenMPUtils.h"
#include "../primitives/ActivationTraits.h"
#include "CPUAttentionKernelT.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar2
{
    namespace detail
    {
        template <int DecodeTargetPct,
                  int PrefillTargetPct,
                  int DecodeMinTile,
                  int DecodeMaxTile,
                  int PrefillMinTile,
                  int PrefillMaxTile>
        struct CacheAwareFlashKVTilePolicy
        {
            static_assert(DecodeTargetPct > 0 && DecodeTargetPct <= 100, "DecodeTargetPct must be in (0, 100]");
            static_assert(PrefillTargetPct > 0 && PrefillTargetPct <= 100, "PrefillTargetPct must be in (0, 100]");

            static int choose(int head_dim, int n_kv_heads, int kv_len, bool is_decode)
            {
                const int override_tile = tile_override(is_decode);
                if (override_tile > 0)
                {
                    return clamp_to_power2(override_tile,
                                           is_decode ? DecodeMinTile : PrefillMinTile,
                                           is_decode ? DecodeMaxTile : PrefillMaxTile);
                }

                AttentionCacheConfig cfg(head_dim, n_kv_heads, kv_len);
                const size_t kv_row_bytes = static_cast<size_t>(head_dim) * sizeof(float) * 2ULL;
                const size_t score_bytes = sizeof(float);
                const size_t bytes_per_kv_position = kv_row_bytes + score_bytes;

                size_t target_cache_bytes = 0;
                switch (cfg.work_size())
                {
                case AttentionWorkSize::SMALL:
                    target_cache_bytes = static_cast<size_t>(cfg.l1_size);
                    break;
                case AttentionWorkSize::LARGE:
                    target_cache_bytes = static_cast<size_t>(cfg.l2_size);
                    break;
                case AttentionWorkSize::XL:
                default:
                    target_cache_bytes = std::max<size_t>(cfg.l3_size / 8, cfg.l2_size);
                    break;
                }

                const int target_pct = is_decode ? DecodeTargetPct : PrefillTargetPct;
                target_cache_bytes = (target_cache_bytes * static_cast<size_t>(target_pct)) / 100ULL;

                size_t raw_tile = target_cache_bytes > 0
                                      ? (target_cache_bytes / std::max<size_t>(1, bytes_per_kv_position))
                                      : static_cast<size_t>(is_decode ? DecodeMinTile : PrefillMinTile);

                if (!is_decode && cfg.prefer_kv8_tile())
                {
                    raw_tile = std::max<size_t>(raw_tile, 8);
                }

                const int min_tile = is_decode ? DecodeMinTile : PrefillMinTile;
                const int max_tile = is_decode ? DecodeMaxTile : PrefillMaxTile;
                return clamp_to_power2(static_cast<int>(raw_tile), min_tile, max_tile);
            }

        private:
            static int clamp_to_power2(int value, int min_tile, int max_tile)
            {
                if (value <= 0)
                {
                    value = min_tile;
                }

                int p2 = 1;
                while ((p2 << 1) > 0 && (p2 << 1) <= value)
                {
                    p2 <<= 1;
                }

                if (p2 < min_tile)
                {
                    p2 = min_tile;
                }
                if (p2 > max_tile)
                {
                    p2 = max_tile;
                }
                return std::max(1, p2);
            }

            static int tile_override(bool is_decode)
            {
                const auto &env = debugEnv();
                const int decode_override = env.attention.flash_kv_tile_decode;
                const int prefill_override = env.attention.flash_kv_tile_prefill;
                return is_decode ? decode_override : prefill_override;
            }
        };

        using DefaultFlashKVTilePolicy = CacheAwareFlashKVTilePolicy<50, 37, 8, 32, 4, 32>;

        template <ActivationPrecision P>
        struct FlashAttentionPrecisionToTensor;

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP32>
        {
            using Type = FP32Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::BF16>
        {
            using Type = BF16Tensor;
        };

        template <>
        struct FlashAttentionPrecisionToTensor<ActivationPrecision::FP16>
        {
            using Type = FP16Tensor;
        };
    }

    template <ActivationPrecision Precision>
    class CPUFlashAttentionKernelT : public ITensorAttention
    {
    public:
        using TensorT = typename detail::FlashAttentionPrecisionToTensor<Precision>::Type;
        using ElementType = typename primitives::ActivationTraits<TensorT>::ElementType;

        CPUFlashAttentionKernelT() = default;
        ~CPUFlashAttentionKernelT() override = default;

        bool supports_device(int device_idx) const override
        {
            return device_idx == -1;
        }

        bool compute(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)workspace_scores;
            (void)workspace_buffer;
            (void)workspace_context;
            (void)use_bf16;
            (void)mpi_ctx;
            (void)device_idx;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute(Q, K, V, output,
                                         seq_len, n_heads, n_kv_heads, head_dim,
                                         causal, window_size,
                                         workspace_scores, workspace_buffer,
                                         workspace_context, workspace_mask,
                                         use_bf16, mpi_ctx, device_idx);
            }

            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;
            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, seq_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, window_size, 0,
                                      mask);
        }

        bool compute_batch(
            const float *Q, const float *K, const float *V, float *output,
            int batch_size, int seq_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = false,
            int window_size = -1,
            TensorBase *workspace_scores = nullptr,
            TensorBase *workspace_buffer = nullptr,
            TensorBase *workspace_context = nullptr,
            TensorBase *workspace_mask = nullptr,
            bool use_bf16 = false,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1) override
        {
            (void)workspace_buffer;
            (void)workspace_context;

            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_batch(Q, K, V, output,
                                               batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                               causal, window_size,
                                               workspace_scores, workspace_buffer,
                                               workspace_context, workspace_mask,
                                               use_bf16, mpi_ctx, device_idx);
            }

            const size_t q_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_heads) * static_cast<size_t>(head_dim);
            const size_t kv_stride = static_cast<size_t>(seq_len) * static_cast<size_t>(n_kv_heads) * static_cast<size_t>(head_dim);
            const float *mask = workspace_mask ? workspace_mask->data() : nullptr;

            for (int b = 0; b < batch_size; ++b)
            {
                const float *Q_b = Q + static_cast<size_t>(b) * q_stride;
                const float *K_b = K + static_cast<size_t>(b) * kv_stride;
                const float *V_b = V + static_cast<size_t>(b) * kv_stride;
                float *O_b = output + static_cast<size_t>(b) * q_stride;
                if (!compute_flash_fp32(Q_b, K_b, V_b, O_b,
                                        seq_len, seq_len,
                                        n_heads, n_kv_heads, head_dim,
                                        causal, window_size, 0,
                                        mask))
                {
                    return false;
                }
            }
            return true;
        }

        bool compute_decode(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len, int n_heads, int n_kv_heads, int head_dim,
            bool causal = true,
            int position_offset = 0) override
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_decode(Q, K, V, output,
                                                seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                                causal, position_offset);
            }

            return compute_flash_fp32(Q, K, V, output,
                                      seq_len, kv_len,
                                      n_heads, n_kv_heads, head_dim,
                                      causal, -1, position_offset,
                                      nullptr);
        }

        bool compute_tensor(
            const ITensor *Q,
            const ITensor *K,
            const ITensor *V,
            ITensor *output,
            int batch_size,
            int seq_len,
            int kv_len,
            int n_heads,
            int n_kv_heads,
            int head_dim,
            bool causal = false,
            int window_size = -1,
            ITensor *workspace_scores = nullptr,
            ITensor *workspace_mask = nullptr,
            const MPIContext *mpi_ctx = nullptr,
            int device_idx = -1,
            int head_start = 0,
            int local_n_heads = -1,
            int local_n_kv_heads = -1) override
        {
            if constexpr (!std::is_same_v<ElementType, float>)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (head_start != 0 || local_n_heads != -1 || local_n_kv_heads != -1)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (Q->native_type() != TensorType::FP32 ||
                output->native_type() != TensorType::FP32)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            const auto *Q_base = dynamic_cast<const TensorBase *>(Q);
            const auto *K_base = dynamic_cast<const TensorBase *>(K);
            const auto *V_base = dynamic_cast<const TensorBase *>(V);
            auto *O_base = dynamic_cast<TensorBase *>(output);
            auto *scores_base = dynamic_cast<TensorBase *>(workspace_scores);
            auto *mask_base = dynamic_cast<TensorBase *>(workspace_mask);

            if (!Q_base || !K_base || !V_base || !O_base)
            {
                return false;
            }

            const float *Q_ptr = Q_base->fp32_data();
            const float *K_ptr = K_base->fp32_data();
            const float *V_ptr = V_base->fp32_data();
            float *O_ptr = O_base->mutable_data();

            if (!Q_ptr || !K_ptr || !V_ptr || !O_ptr)
            {
                return fallback_.compute_tensor(Q, K, V, output,
                                                batch_size, seq_len, kv_len,
                                                n_heads, n_kv_heads, head_dim,
                                                causal, window_size,
                                                workspace_scores, workspace_mask,
                                                mpi_ctx, device_idx,
                                                head_start, local_n_heads, local_n_kv_heads);
            }

            if (batch_size > 1)
            {
                return compute_batch(Q_ptr, K_ptr, V_ptr, O_ptr,
                                     batch_size, seq_len, n_heads, n_kv_heads, head_dim,
                                     causal, window_size,
                                     scores_base, nullptr, nullptr, mask_base,
                                     false, mpi_ctx, device_idx);
            }

            if (kv_len != seq_len)
            {
                const int position_offset = (kv_len > seq_len) ? (kv_len - seq_len) : 0;
                return compute_decode(Q_ptr, K_ptr, V_ptr, O_ptr,
                                      seq_len, kv_len, n_heads, n_kv_heads, head_dim,
                                      causal, position_offset);
            }

            return compute(Q_ptr, K_ptr, V_ptr, O_ptr,
                           seq_len, n_heads, n_kv_heads, head_dim,
                           causal, window_size,
                           scores_base, nullptr, nullptr, mask_base,
                           false, mpi_ctx, device_idx);
        }

        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::attention()
                .withInput("Q", "query tensor [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("K", "key tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withInput("V", "value tensor [kv_len, n_kv_heads * head_dim]", KernelBufferDtype::FP32)
                .withOutput("output", "attention output [seq_len, n_heads * head_dim]", KernelBufferDtype::FP32)
                .withScalar("seq_len", "query sequence length", KernelBufferDtype::INT32)
                .withScalar("kv_len", "key/value sequence length", KernelBufferDtype::INT32)
                .withScalar("n_heads", "number of query heads", KernelBufferDtype::INT32)
                .withScalar("n_kv_heads", "number of key/value heads", KernelBufferDtype::INT32)
                .withScalar("head_dim", "dimension per head", KernelBufferDtype::INT32)
                .withScalar("causal", "apply causal masking", KernelBufferDtype::INT32);
        }

    private:
        CPUAttentionKernelT<Precision> fallback_;

        static float dot_fp32_scalar(const float *a, const float *b, int n)
        {
            float sum = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
        }

        static float dot_fp32_avx512(const float *a, const float *b, int n)
        {
#if defined(__AVX512F__)
            __m512 acc = _mm512_setzero_ps();
            int i = 0;
            for (; i + 15 < n; i += 16)
            {
                __m512 va = _mm512_loadu_ps(a + i);
                __m512 vb = _mm512_loadu_ps(b + i);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }
            float sum = _mm512_reduce_add_ps(acc);
            for (; i < n; ++i)
            {
                sum += a[i] * b[i];
            }
            return sum;
#else
            return dot_fp32_scalar(a, b, n);
#endif
        }

        static void accum_weighted_v(float *out, const float *v, float weight, int head_dim, bool use_avx512)
        {
#if defined(__AVX512F__)
            if (use_avx512)
            {
                __m512 w = _mm512_set1_ps(weight);
                int d = 0;
                for (; d + 15 < head_dim; d += 16)
                {
                    __m512 o = _mm512_loadu_ps(out + d);
                    __m512 vv = _mm512_loadu_ps(v + d);
                    o = _mm512_fmadd_ps(vv, w, o);
                    _mm512_storeu_ps(out + d, o);
                }
                for (; d < head_dim; ++d)
                {
                    out[d] += weight * v[d];
                }
                return;
            }
#endif
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] += weight * v[d];
            }
        }

        static void scale_vec(float *out, float alpha, int head_dim, bool use_avx512)
        {
#if defined(__AVX512F__)
            if (use_avx512)
            {
                __m512 a = _mm512_set1_ps(alpha);
                int d = 0;
                for (; d + 15 < head_dim; d += 16)
                {
                    __m512 o = _mm512_loadu_ps(out + d);
                    o = _mm512_mul_ps(o, a);
                    _mm512_storeu_ps(out + d, o);
                }
                for (; d < head_dim; ++d)
                {
                    out[d] *= alpha;
                }
                return;
            }
#endif
            for (int d = 0; d < head_dim; ++d)
            {
                out[d] *= alpha;
            }
        }

        static void div_vec(float *out, float denom, int head_dim, bool use_avx512)
        {
            if (denom <= 0.0f)
            {
                return;
            }
            scale_vec(out, 1.0f / denom, head_dim, use_avx512);
        }

        static float quantize_row_i16_i12(const float *src, int16_t *dst, int n, int qmax)
        {
            float max_abs = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            if (max_abs <= 1e-12f)
            {
                std::memset(dst, 0, static_cast<size_t>(n) * sizeof(int16_t));
                return 0.0f;
            }

            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            for (int i = 0; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                dst[i] = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
            }
            return scale;
        }

        static float quantize_row_i16_i12_padded(const float *src, int16_t *dst, int n, int padded_n, int qmax)
        {
            const float scale = quantize_row_i16_i12(src, dst, n, qmax);
            if (padded_n > n)
            {
                std::memset(dst + n, 0, static_cast<size_t>(padded_n - n) * sizeof(int16_t));
            }
            return scale;
        }

        static float quantize_row_i16_i12_to_packedpair(
            const float *src,
            int16_t *pair_dst,
            int n,
            int padded_n,
            int qmax,
            int row_sel)
        {
            float max_abs = 0.0f;
            for (int i = 0; i < n; ++i)
            {
                max_abs = std::max(max_abs, std::abs(src[i]));
            }

            if (max_abs <= 1e-12f)
            {
                for (int i = 0; i < padded_n; ++i)
                {
                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                    const size_t lane = static_cast<size_t>(i % 32);
                    pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = 0;
                }
                return 0.0f;
            }

            const float scale = max_abs / static_cast<float>(qmax);
            const float inv_scale = 1.0f / scale;
            for (int i = 0; i < n; ++i)
            {
                const int q = static_cast<int>(std::lrint(src[i] * inv_scale));
                const int16_t v = static_cast<int16_t>(std::max(-qmax, std::min(q, qmax)));
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = v;
            }
            for (int i = n; i < padded_n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                pair_dst[block + static_cast<size_t>(row_sel) * 32ULL + lane] = 0;
            }
            return scale;
        }

        static int32_t dot_i16_i16_i32_scalar(const int16_t *a, const int16_t *b, int n)
        {
            int32_t sum = 0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
        }

        static int32_t dot_i16_i16_i32_vnni(const int16_t *a, const int16_t *b, int n)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            for (; i + 31 < n; i += 32)
            {
                const __m512i va = _mm512_loadu_si512(reinterpret_cast<const void *>(a + i));
                const __m512i vb = _mm512_loadu_si512(reinterpret_cast<const void *>(b + i));
                acc = _mm512_dpwssd_epi32(acc, va, vb);
            }

            alignas(64) int32_t lanes[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), acc);
            int32_t sum = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum += lanes[lane];
            }

            for (; i < n; ++i)
            {
                sum += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
            }
            return sum;
#else
            return dot_i16_i16_i32_scalar(a, b, n);
#endif
        }

        static void dot_i16_i16_i32_vnni_2row(
            const int16_t *q,
            const int16_t *k0,
            const int16_t *k1,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();

            int i = 0;
            for (; i + 63 < n; i += 64)
            {
                const __m512i q0 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i q1 = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i + 32));

                const __m512i k00 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k01 = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i + 32));
                const __m512i k10 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                const __m512i k11 = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i + 32));

                acc0 = _mm512_dpwssd_epi32(acc0, q0, k00);
                acc0 = _mm512_dpwssd_epi32(acc0, q1, k01);
                acc1 = _mm512_dpwssd_epi32(acc1, q0, k10);
                acc1 = _mm512_dpwssd_epi32(acc1, q1, k11);
            }

            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k0v = _mm512_loadu_si512(reinterpret_cast<const void *>(k0 + i));
                const __m512i k1v = _mm512_loadu_si512(reinterpret_cast<const void *>(k1 + i));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k0v);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k1v);
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
            }

            for (; i < n; ++i)
            {
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k0[i]);
                sum1 += qv * static_cast<int32_t>(k1[i]);
            }

            out0 = sum0;
            out1 = sum1;
#else
            out0 = dot_i16_i16_i32_scalar(q, k0, n);
            out1 = dot_i16_i16_i32_scalar(q, k1, n);
#endif
        }

        static void dot_i16_i16_i32_vnni_2row_packedpair(
            const int16_t *q,
            const int16_t *k_pair,
            int n,
            int32_t &out0,
            int32_t &out1)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();

            int i = 0;
            const int16_t *pair_ptr = k_pair;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k0v = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr));
                const __m512i k1v = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr + 32));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k0v);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k1v);
                pair_ptr += 64;
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
            }

            for (; i < n; ++i)
            {
                sum0 += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + (i % 32)]);
                sum1 += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + 32 + (i % 32)]);
            }

            out0 = sum0;
            out1 = sum1;
#else
            out0 = 0;
            out1 = 0;
            for (int i = 0; i < n; ++i)
            {
                const int pair_block = i / 32;
                const int lane = i % 32;
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k_pair[static_cast<size_t>(pair_block) * 64 + lane]);
                out1 += qv * static_cast<int32_t>(k_pair[static_cast<size_t>(pair_block) * 64 + 32 + lane]);
            }
#endif
        }

        static void dot_i16_i16_i32_vnni_4row_packedpair(
            const int16_t *q,
            const int16_t *k_pair0,
            const int16_t *k_pair1,
            int n,
            int32_t &out0,
            int32_t &out1,
            int32_t &out2,
            int32_t &out3)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc0 = _mm512_setzero_si512();
            __m512i acc1 = _mm512_setzero_si512();
            __m512i acc2 = _mm512_setzero_si512();
            __m512i acc3 = _mm512_setzero_si512();

            int i = 0;
            const int16_t *p0 = k_pair0;
            const int16_t *p1 = k_pair1;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i k00 = _mm512_loadu_si512(reinterpret_cast<const void *>(p0));
                const __m512i k01 = _mm512_loadu_si512(reinterpret_cast<const void *>(p0 + 32));
                const __m512i k10 = _mm512_loadu_si512(reinterpret_cast<const void *>(p1));
                const __m512i k11 = _mm512_loadu_si512(reinterpret_cast<const void *>(p1 + 32));
                acc0 = _mm512_dpwssd_epi32(acc0, qv, k00);
                acc1 = _mm512_dpwssd_epi32(acc1, qv, k01);
                acc2 = _mm512_dpwssd_epi32(acc2, qv, k10);
                acc3 = _mm512_dpwssd_epi32(acc3, qv, k11);
                p0 += 64;
                p1 += 64;
            }

            alignas(64) int32_t lanes0[16];
            alignas(64) int32_t lanes1[16];
            alignas(64) int32_t lanes2[16];
            alignas(64) int32_t lanes3[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes0), acc0);
            _mm512_store_si512(reinterpret_cast<void *>(lanes1), acc1);
            _mm512_store_si512(reinterpret_cast<void *>(lanes2), acc2);
            _mm512_store_si512(reinterpret_cast<void *>(lanes3), acc3);

            int32_t sum0 = 0;
            int32_t sum1 = 0;
            int32_t sum2 = 0;
            int32_t sum3 = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum0 += lanes0[lane];
                sum1 += lanes1[lane];
                sum2 += lanes2[lane];
                sum3 += lanes3[lane];
            }

            for (; i < n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                const int32_t qv = static_cast<int32_t>(q[i]);
                sum0 += qv * static_cast<int32_t>(k_pair0[block + lane]);
                sum1 += qv * static_cast<int32_t>(k_pair0[block + 32ULL + lane]);
                sum2 += qv * static_cast<int32_t>(k_pair1[block + lane]);
                sum3 += qv * static_cast<int32_t>(k_pair1[block + 32ULL + lane]);
            }

            out0 = sum0;
            out1 = sum1;
            out2 = sum2;
            out3 = sum3;
#else
            out0 = 0;
            out1 = 0;
            out2 = 0;
            out3 = 0;
            for (int i = 0; i < n; ++i)
            {
                const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                const size_t lane = static_cast<size_t>(i % 32);
                const int32_t qv = static_cast<int32_t>(q[i]);
                out0 += qv * static_cast<int32_t>(k_pair0[block + lane]);
                out1 += qv * static_cast<int32_t>(k_pair0[block + 32ULL + lane]);
                out2 += qv * static_cast<int32_t>(k_pair1[block + lane]);
                out3 += qv * static_cast<int32_t>(k_pair1[block + 32ULL + lane]);
            }
#endif
        }

        static int32_t dot_i16_i16_i32_vnni_single_from_packedpair(
            const int16_t *q,
            const int16_t *k_pair,
            int n,
            int row_sel)
        {
#if defined(__AVX512F__) && defined(__AVX512VNNI__)
            __m512i acc = _mm512_setzero_si512();
            int i = 0;
            const int16_t *pair_ptr = k_pair;
            const int row_off = (row_sel != 0) ? 32 : 0;
            for (; i + 31 < n; i += 32)
            {
                const __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void *>(q + i));
                const __m512i kv = _mm512_loadu_si512(reinterpret_cast<const void *>(pair_ptr + row_off));
                acc = _mm512_dpwssd_epi32(acc, qv, kv);
                pair_ptr += 64;
            }

            alignas(64) int32_t lanes[16];
            _mm512_store_si512(reinterpret_cast<void *>(lanes), acc);
            int32_t sum = 0;
            for (int lane = 0; lane < 16; ++lane)
            {
                sum += lanes[lane];
            }

            for (; i < n; ++i)
            {
                sum += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + row_off + (i % 32)]);
            }
            return sum;
#else
            int32_t sum = 0;
            const int row_off = (row_sel != 0) ? 32 : 0;
            for (int i = 0; i < n; ++i)
            {
                sum += static_cast<int32_t>(q[i]) * static_cast<int32_t>(k_pair[(static_cast<size_t>(i) / 32) * 64 + row_off + (i % 32)]);
            }
            return sum;
#endif
        }

        static bool compute_flash_fp32(
            const float *Q, const float *K, const float *V, float *output,
            int seq_len, int kv_len,
            int n_heads, int n_kv_heads, int head_dim,
            bool causal, int window_size, int position_offset,
            const float *mask)
        {
            KERNEL_PROFILE_SCOPE(KernelType::ATTENTION);

            if (!Q || !K || !V || !output)
            {
                return false;
            }
            if (seq_len <= 0 || kv_len <= 0 || n_heads <= 0 || n_kv_heads <= 0 || head_dim <= 0)
            {
                return false;
            }
            if (n_heads % n_kv_heads != 0)
            {
                return false;
            }

            const bool use_avx512 = cpu_supports_avx512();
            const bool is_decode = (seq_len == 1 && kv_len >= 1);
            const int kv_tile = detail::DefaultFlashKVTilePolicy::choose(head_dim, n_kv_heads, kv_len, is_decode);
            const int heads_per_kv = n_heads / n_kv_heads;
            const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            const int q_stride = n_heads * head_dim;
            const int kv_stride = n_kv_heads * head_dim;
            const int i16_row_stride = ((head_dim + 31) / 32) * 32;
            const int i16_chunks = i16_row_stride / 32;
            const int kv_pair_count = (kv_len + 1) / 2;
            const int k_pair_stride = i16_chunks * 64;
            const bool profiling_enabled = KernelProfiler::isEnabled();
            uint64_t qk_duration_ns = 0;
            uint64_t v_duration_ns = 0;

            const auto &env = debugEnv();
            const int64_t prefill_work = static_cast<int64_t>(seq_len) * static_cast<int64_t>(kv_len);
            const bool use_i16_i12_prefill =
                !is_decode &&
                env.attention.flash_prefill_i16_i12 &&
                cpu_supports_avx512_vnni() &&
                seq_len >= env.attention.flash_prefill_i16_i12_min_seq &&
                kv_len >= env.attention.flash_prefill_i16_i12_min_kv &&
                prefill_work >= env.attention.flash_prefill_i16_i12_min_work &&
                head_dim <= env.attention.flash_prefill_i16_i12_max_head_dim;

            const int qmax = std::max(1, std::min(env.attention.flash_prefill_i16_i12_qmax, 32767));
            std::vector<int16_t> packed_k_pairs_i16;
            std::vector<float> packed_k_pair_scales;

            if (use_i16_i12_prefill)
            {
                packed_k_pairs_i16.resize(static_cast<size_t>(n_kv_heads) * static_cast<size_t>(kv_pair_count) * static_cast<size_t>(k_pair_stride));
                packed_k_pair_scales.resize(static_cast<size_t>(n_kv_heads) * static_cast<size_t>(kv_pair_count) * 2ULL, 0.0f);

                auto do_pack = [&]()
                {
#pragma omp for collapse(2) schedule(static)
                    for (int kv_h = 0; kv_h < n_kv_heads; ++kv_h)
                    {
                        for (int pair_idx = 0; pair_idx < kv_pair_count; ++pair_idx)
                        {
                            const int k0 = pair_idx * 2;
                            const int k1 = k0 + 1;

                            const size_t pair_global = static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) + static_cast<size_t>(pair_idx);
                            int16_t *pair_dst = packed_k_pairs_i16.data() + pair_global * static_cast<size_t>(k_pair_stride);

                            float s0 = 0.0f;
                            float s1 = 0.0f;
                            if (k0 < kv_len)
                            {
                                const float *k_src0 = K + static_cast<size_t>(k0) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                                s0 = quantize_row_i16_i12_to_packedpair(k_src0, pair_dst, head_dim, i16_row_stride, qmax, 0);
                            }
                            else
                            {
                                for (int i = 0; i < i16_row_stride; ++i)
                                {
                                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                                    const size_t lane = static_cast<size_t>(i % 32);
                                    pair_dst[block + lane] = 0;
                                }
                            }

                            if (k1 < kv_len)
                            {
                                const float *k_src1 = K + static_cast<size_t>(k1) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                                s1 = quantize_row_i16_i12_to_packedpair(k_src1, pair_dst, head_dim, i16_row_stride, qmax, 1);
                            }
                            else
                            {
                                for (int i = 0; i < i16_row_stride; ++i)
                                {
                                    const size_t block = static_cast<size_t>(i / 32) * 64ULL;
                                    const size_t lane = static_cast<size_t>(i % 32);
                                    pair_dst[block + 32ULL + lane] = 0;
                                }
                            }

                            packed_k_pair_scales[pair_global * 2ULL + 0ULL] = s0;
                            packed_k_pair_scales[pair_global * 2ULL + 1ULL] = s1;
                        }
                    }
                };
                OMP_WORKSHARE_REGION(do_pack);
            }

            auto work = [&]()
            {
#pragma omp for schedule(static) reduction(+ : qk_duration_ns, v_duration_ns)
                for (int h = 0; h < n_heads; ++h)
                {
                    const int kv_h = h / heads_per_kv;
                    const int16_t *k_head_pairs_i16 = use_i16_i12_prefill
                                                       ? packed_k_pairs_i16.data() + static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) * static_cast<size_t>(k_pair_stride)
                                                       : nullptr;
                    const float *k_head_pair_scales = use_i16_i12_prefill
                                                     ? packed_k_pair_scales.data() + static_cast<size_t>(kv_h) * static_cast<size_t>(kv_pair_count) * 2ULL
                                                     : nullptr;
                    for (int q_pos = 0; q_pos < seq_len; ++q_pos)
                    {
                        float *out = output + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        std::fill(out, out + head_dim, 0.0f);

                        float running_m = -std::numeric_limits<float>::infinity();
                        float running_l = 0.0f;

                        const float *q_ptr = Q + static_cast<size_t>(q_pos) * q_stride + static_cast<size_t>(h) * head_dim;
                        const int q_abs = position_offset + q_pos;
                        const float *mask_row = mask ? (mask + static_cast<size_t>(q_pos) * kv_len) : nullptr;
                        thread_local std::vector<int16_t> q_i16;
                        float q_scale_i16 = 0.0f;
                        if (use_i16_i12_prefill)
                        {
                            if (static_cast<int>(q_i16.size()) < i16_row_stride)
                            {
                                q_i16.resize(static_cast<size_t>(i16_row_stride));
                            }
                            q_scale_i16 = quantize_row_i16_i12_padded(q_ptr, q_i16.data(), head_dim, i16_row_stride, qmax);
                        }

                        for (int k0 = 0; k0 < kv_len; k0 += kv_tile)
                        {
                            const int k1 = std::min(k0 + kv_tile, kv_len);
                            float block_max = -std::numeric_limits<float>::infinity();
                            thread_local std::vector<float> block_scores;
                            const int blk = k1 - k0;
                            if (static_cast<int>(block_scores.size()) < blk)
                            {
                                block_scores.resize(blk);
                            }

                            const auto qk_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            int valid_start = k0;
                            int valid_end = k1;
                            if (window_size > 0)
                            {
                                valid_start = std::max(valid_start, q_abs - window_size + 1);
                            }
                            if (causal)
                            {
                                valid_end = std::min(valid_end, q_abs + 1);
                            }
                            valid_start = std::max(k0, std::min(valid_start, k1));
                            valid_end = std::max(k0, std::min(valid_end, k1));

                            for (int k = k0; k < valid_start; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }
                            for (int k = valid_end; k < k1; ++k)
                            {
                                block_scores[static_cast<size_t>(k - k0)] = -std::numeric_limits<float>::infinity();
                            }

                            for (int k = valid_start; k < valid_end;)
                            {
                                if (use_i16_i12_prefill)
                                {
                                    if (k + 3 < valid_end)
                                    {
                                        const int pair_idx0 = k / 2;
                                        const int pair_idx1 = pair_idx0 + 1;
                                        const int16_t *k_pair0 = k_head_pairs_i16 + static_cast<size_t>(pair_idx0) * static_cast<size_t>(k_pair_stride);
                                        const int16_t *k_pair1 = k_head_pairs_i16 + static_cast<size_t>(pair_idx1) * static_cast<size_t>(k_pair_stride);
                                        const float k_scale0 = k_head_pair_scales[static_cast<size_t>(pair_idx0) * 2ULL + 0ULL];
                                        const float k_scale1 = k_head_pair_scales[static_cast<size_t>(pair_idx0) * 2ULL + 1ULL];
                                        const float k_scale2 = k_head_pair_scales[static_cast<size_t>(pair_idx1) * 2ULL + 0ULL];
                                        const float k_scale3 = k_head_pair_scales[static_cast<size_t>(pair_idx1) * 2ULL + 1ULL];

                                        int32_t dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
                                        dot_i16_i16_i32_vnni_4row_packedpair(q_i16.data(), k_pair0, k_pair1, i16_row_stride, dot0, dot1, dot2, dot3);

                                        float s0 = static_cast<float>(dot0) * (q_scale_i16 * k_scale0) * scale;
                                        float s1 = static_cast<float>(dot1) * (q_scale_i16 * k_scale1) * scale;
                                        float s2 = static_cast<float>(dot2) * (q_scale_i16 * k_scale2) * scale;
                                        float s3 = static_cast<float>(dot3) * (q_scale_i16 * k_scale3) * scale;

                                        if (mask_row)
                                        {
                                            s0 += mask_row[k + 0];
                                            s1 += mask_row[k + 1];
                                            s2 += mask_row[k + 2];
                                            s3 += mask_row[k + 3];
                                        }

                                        block_scores[static_cast<size_t>(k - k0)] = s0;
                                        block_scores[static_cast<size_t>(k + 1 - k0)] = s1;
                                        block_scores[static_cast<size_t>(k + 2 - k0)] = s2;
                                        block_scores[static_cast<size_t>(k + 3 - k0)] = s3;
                                        block_max = std::max(block_max, s0);
                                        block_max = std::max(block_max, s1);
                                        block_max = std::max(block_max, s2);
                                        block_max = std::max(block_max, s3);

                                        k += 4;
                                        continue;
                                    }

                                    if (k + 1 < valid_end)
                                    {
                                        const int pair_idx = k / 2;
                                        const int16_t *k_pair = k_head_pairs_i16 + static_cast<size_t>(pair_idx) * static_cast<size_t>(k_pair_stride);
                                        const float k_scale0 = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + 0ULL];
                                        const float k_scale1 = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + 1ULL];
                                        int32_t dot_i32_0 = 0;
                                        int32_t dot_i32_1 = 0;
                                        dot_i16_i16_i32_vnni_2row_packedpair(q_i16.data(), k_pair, i16_row_stride, dot_i32_0, dot_i32_1);

                                        float s0 = static_cast<float>(dot_i32_0) * (q_scale_i16 * k_scale0);
                                        float s1 = static_cast<float>(dot_i32_1) * (q_scale_i16 * k_scale1);
                                        s0 *= scale;
                                        s1 *= scale;

                                        if (mask_row)
                                        {
                                            s0 += mask_row[k];
                                            s1 += mask_row[k + 1];
                                        }

                                        block_scores[static_cast<size_t>(k - k0)] = s0;
                                        block_scores[static_cast<size_t>(k + 1 - k0)] = s1;
                                        block_max = std::max(block_max, s0);
                                        block_max = std::max(block_max, s1);
                                        k += 2;
                                        continue;
                                    }

                                    const int pair_idx = k / 2;
                                    const int row_sel = (k & 1);
                                    const int16_t *k_pair = k_head_pairs_i16 + static_cast<size_t>(pair_idx) * static_cast<size_t>(k_pair_stride);
                                    const float k_scale = k_head_pair_scales[static_cast<size_t>(pair_idx) * 2ULL + static_cast<size_t>(row_sel)];
                                    const int32_t dot_i32 = dot_i16_i16_i32_vnni_single_from_packedpair(q_i16.data(), k_pair, i16_row_stride, row_sel);
                                    float s = static_cast<float>(dot_i32) * (q_scale_i16 * k_scale);
                                    s *= scale;

                                    if (mask_row)
                                    {
                                        s += mask_row[k];
                                    }

                                    block_scores[static_cast<size_t>(k - k0)] = s;
                                    block_max = std::max(block_max, s);
                                    ++k;
                                    continue;
                                }

                                float s = dot_fp32_avx512(
                                    q_ptr,
                                    K + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim,
                                    head_dim);
                                s *= scale;

                                if (mask_row)
                                {
                                    s += mask_row[k];
                                }

                                block_scores[static_cast<size_t>(k - k0)] = s;
                                block_max = std::max(block_max, s);
                                ++k;
                            }

                            if (profiling_enabled)
                            {
                                qk_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - qk_start).count());
                            }

                            const float new_m = std::max(running_m, block_max);
                            const float alpha = std::isfinite(running_m) ? std::exp(running_m - new_m) : 0.0f;
                            scale_vec(out, alpha, head_dim, use_avx512);
                            float new_l = running_l * alpha;

                            const auto v_start = profiling_enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point();

                            for (int k = k0; k < k1; ++k)
                            {
                                const float s = block_scores[static_cast<size_t>(k - k0)];
                                if (!std::isfinite(s))
                                {
                                    continue;
                                }

                                const float p = std::exp(s - new_m);
                                new_l += p;
                                const float *v_ptr = V + static_cast<size_t>(k) * kv_stride + static_cast<size_t>(kv_h) * head_dim;
                                accum_weighted_v(out, v_ptr, p, head_dim, use_avx512);
                            }

                            if (profiling_enabled)
                            {
                                v_duration_ns += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - v_start).count());
                            }

                            running_m = new_m;
                            running_l = new_l;
                        }

                        div_vec(out, running_l, head_dim, use_avx512);
                    }
                }
            };

            OMP_WORKSHARE_REGION(work);
            if (profiling_enabled)
            {
                KernelProfiler::record(KernelType::ATTENTION_QK, qk_duration_ns);
                KernelProfiler::record(KernelType::ATTENTION_V, v_duration_ns);
            }
            return true;
        }
    };

    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP32>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::BF16>;
    extern template class CPUFlashAttentionKernelT<ActivationPrecision::FP16>;

} // namespace llaminar2
