/**
 * @file IntegerGemmMicroKernel.h
 * @brief Clean micro-kernel for INT8 GEMM with optional AVX512 VNNI path.
 *
 * This rewrite removes duplicated method bodies introduced by prior partial
 * edits. It preserves dual 32-byte sub-block behavior for k_panel==64 and
 * selectable bias correction strategies for dpbusd (offset vs negmask).
 */

#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <cstdio>

#include "../../../../tensors/FP16Utils.h"
#include "../../../../tensors/BlockStructures.h"
#include "../../../cpu/SimdTraits.h"

namespace llaminar2
{
    namespace kernels
    {
        namespace gemm
        {

            using ::llaminar2::Q8_0Block;

            // Provide defaulted tiling parameters so legacy 3-parameter instantiations
            // (ISA, TILE_M, TILE_N) used in older tests remain valid.
            template <typename ISA,
                      int TILE_M,
                      int TILE_N,
                      int UNROLL_K = 1,
                      int PREFETCH_DIST = 0,
                      int MC = 0,
                      int KC = 0,
                      int NC = 0>
            class IntegerGemmMicroKernel
            {
            public:
                IntegerGemmMicroKernel() { zero(); }

                inline void zero()
                {
                    std::memset(int32_acc_, 0, sizeof(int32_acc_));
                    std::memset(fp_accumulators_, 0, sizeof(fp_accumulators_));
                    num_k_blocks_ = 0;
                }

                // Primary accumulate (dual-scale aware)
                inline void accumulate(const int8_t *A_panel,
                                       const int8_t *B_panel,
                                       int k_panel,
                                       const double *a_scales,
                                       const double *b_scales,
                                       const double *a_scales2,
                                       const double *b_scales2)
                {
                    if (use_simd_path_.load(std::memory_order_relaxed))
                    {
                        accumulateSimd(A_panel, B_panel, k_panel, a_scales, b_scales, a_scales2, b_scales2);
                    }
                    else
                    {
                        accumulateScalar(A_panel, B_panel, k_panel, a_scales, b_scales, a_scales2, b_scales2);
                    }
                    ++num_k_blocks_;
                }

                // Legacy caller overload (single block scales)
                inline void accumulate(const int8_t *A_panel,
                                       const int8_t *B_panel,
                                       int k_panel,
                                       const double *a_scales,
                                       const double *b_scales)
                {
                    accumulate(A_panel, B_panel, k_panel, a_scales, b_scales, nullptr, nullptr);
                }

                inline void reduce(Q8_0Block *C_blocks)
                {
                    static_assert(TILE_N == 32, "reduce() requires TILE_N==32");
                    if (num_k_blocks_ == 0)
                    {
                        std::memset(C_blocks, 0, TILE_M * sizeof(Q8_0Block));
                        return;
                    }
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        quantizeFP32ToQ8_0(&fp_accumulators_[i * TILE_N], &C_blocks[i], TILE_N);
                    }
                }

                inline void reduce_with_scales(Q8_0Block *C_blocks, const double *scales)
                {
                    static_assert(TILE_N == 32, "reduce_with_scales() requires TILE_N==32");
                    alignas(64) float tmp[TILE_M * TILE_N];
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            int idx = i * TILE_N + j;
                            tmp[idx] = static_cast<float>(int32_acc_[idx]) * static_cast<float>(scales[idx]);
                        }
                        quantizeFP32ToQ8_0(&tmp[i * TILE_N], &C_blocks[i], TILE_N);
                    }
                }

                inline int32_t accumulator(int i, int j) const { return int32_acc_[i * TILE_N + j]; }
                inline const float *raw_fp_accumulators() const { return fp_accumulators_; }

                static void setUseSimd(bool enabled) { use_simd_path_.store(enabled, std::memory_order_relaxed); }
                static bool useSimd() { return use_simd_path_.load(std::memory_order_relaxed); }

            private:
                // Scalar fallback (also used for AVX512 off or unsupported panel sizes)
                inline void accumulateScalar(const int8_t *A_panel,
                                             const int8_t *B_panel,
                                             int k_panel,
                                             const double *a_scales,
                                             const double *b_scales,
                                             const double *a_scales2,
                                             const double *b_scales2)
                {
                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * k_panel;
                        float a_scale0 = static_cast<float>(a_scales[i]);
                        float a_scale1 = (k_panel == 64 && a_scales2) ? static_cast<float>(a_scales2[i]) : a_scale0;
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * k_panel;
                            if (k_panel == 32)
                            {
                                int32_t dot = 0;
                                for (int kk = 0; kk < 32; ++kk)
                                    dot += (int32_t)a_row[kk] * (int32_t)b_col[kk];
                                int32_acc_[i * TILE_N + j] += dot;
                                float b_scale0 = static_cast<float>(b_scales[j]);
                                fp_accumulators_[i * TILE_N + j] += (float)dot * a_scale0 * b_scale0;
                            }
                            else
                            { // 64
                                int32_t dot0 = 0, dot1 = 0;
                                for (int kk = 0; kk < 32; ++kk)
                                    dot0 += (int32_t)a_row[kk] * (int32_t)b_col[kk];
                                for (int kk = 32; kk < 64; ++kk)
                                    dot1 += (int32_t)a_row[kk] * (int32_t)b_col[kk];
                                int32_acc_[i * TILE_N + j] += (dot0 + dot1);
                                float b_scale0 = static_cast<float>(b_scales[j]);
                                float b_scale1 = b_scales2 ? static_cast<float>(b_scales2[j]) : b_scale0;
                                fp_accumulators_[i * TILE_N + j] += (float)dot0 * a_scale0 * b_scale0 + (float)dot1 * a_scale1 * b_scale1;
                            }
                        }
                    }
                }

                inline void accumulateSimd(const int8_t *A_panel,
                                           const int8_t *B_panel,
                                           int k_panel,
                                           const double *a_scales,
                                           const double *b_scales,
                                           const double *a_scales2,
                                           const double *b_scales2)
                {
#if !defined(__AVX512VNNI__)
                    // No VNNI support: fallback
                    accumulateScalar(A_panel, B_panel, k_panel, a_scales, b_scales, a_scales2, b_scales2);
                    return;
#else
                    if (!(k_panel == 32 || k_panel == 64))
                    {
                        accumulateScalar(A_panel, B_panel, k_panel, a_scales, b_scales, a_scales2, b_scales2);
                        return;
                    }

                    static bool diag_basic = (std::getenv("LLAMINAR_INT8_DIAG_BLOCKS") != nullptr);
                    static bool diag_ext = (std::getenv("LLAMINAR_INT8_DIAG_EXT") != nullptr);
                    static std::atomic<int> diag_count{0};
                    constexpr int kDiagMax = 16;
                    static int bias_mode = []()
                    {
                        const char *v = std::getenv("LLAMINAR_INT8_BIAS_MODE");
                        if (!v)
                            return 1; // negmask default
                        if (std::strcmp(v, "offset") == 0)
                            return 0;
                        return 1;
                    }();

                    const __mmask64 mask32 = (static_cast<__mmask64>(1ULL << 32) - 1);
                    alignas(64) int32_t lane_buf[16];
                    alignas(64) int8_t tmp_bytes[64];

                    auto corrected_dot_32 = [&](const int8_t *a_ptr, const int8_t *b_ptr) -> int32_t
                    {
                        __m512i a_vec = _mm512_maskz_loadu_epi8(mask32, a_ptr);
                        __m512i b_vec = _mm512_maskz_loadu_epi8(mask32, b_ptr);
                        __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), a_vec, b_vec);
                        _mm512_store_si512(lane_buf, acc);
                        int32_t unsigned_sum = 0;
                        for (int l = 0; l < 8; ++l)
                            unsigned_sum += lane_buf[l];
                        _mm512_store_si512(tmp_bytes, b_vec);
                        int32_t sum_b = 0;
                        for (int i = 0; i < 32; ++i)
                            sum_b += (int32_t)tmp_bytes[i];
                        alignas(64) int8_t a_bytes[64];
                        _mm512_store_si512(a_bytes, a_vec);
                        int32_t sum_neg_b = 0;
                        for (int i = 0; i < 32; ++i)
                            if (a_bytes[i] < 0)
                                sum_neg_b += (int32_t)tmp_bytes[i];
                        int32_t corr_offset = unsigned_sum - 128 * sum_b;
                        int32_t corr_negmask = unsigned_sum - 256 * sum_neg_b;
                        int32_t chosen = (bias_mode == 0) ? corr_offset : corr_negmask;
                        if ((diag_basic || diag_ext) && diag_count.load() < kDiagMax)
                        {
                            int32_t scalar_dot = 0;
                            for (int kk = 0; kk < 32; ++kk)
                                scalar_dot += (int32_t)a_ptr[kk] * (int32_t)b_ptr[kk];
                            if (diag_basic)
                            {
                                fprintf(stderr,
                                        "[INT8-DIAG k32] scalar=%d unsigned=%d sum_b=%d sum_neg_b=%d off=%d neg=%d chosen=%d diff=%d mode=%s\n",
                                        scalar_dot, unsigned_sum, sum_b, sum_neg_b, corr_offset, corr_negmask, chosen,
                                        (chosen - scalar_dot), (bias_mode == 0 ? "offset" : "negmask"));
                            }
                            else
                            {
                                fprintf(stderr,
                                        "[INT8-DIAG-EXT k32] scalar=%d unsigned=%d sum_b=%d sum_neg_b=%d off=%d neg=%d mode=%d d_off=%d d_neg=%d\n",
                                        scalar_dot, unsigned_sum, sum_b, sum_neg_b, corr_offset, corr_negmask, bias_mode,
                                        (corr_offset - scalar_dot), (corr_negmask - scalar_dot));
                            }
                            diag_count.fetch_add(1);
                        }
                        return chosen;
                    };

                    for (int i = 0; i < TILE_M; ++i)
                    {
                        const int8_t *a_row = A_panel + i * k_panel;
                        float a_scale0 = (float)a_scales[i];
                        float a_scale1 = (k_panel == 64 && a_scales2) ? (float)a_scales2[i] : a_scale0;
                        for (int j = 0; j < TILE_N; ++j)
                        {
                            const int8_t *b_col = B_panel + j * k_panel;
                            if (k_panel == 32)
                            {
                                int32_t dot = corrected_dot_32(a_row, b_col);
                                int32_acc_[i * TILE_N + j] += dot;
                                float b_scale0 = (float)b_scales[j];
                                fp_accumulators_[i * TILE_N + j] += (float)dot * a_scale0 * b_scale0;
                            }
                            else
                            { // 64
                                int32_t dot0 = corrected_dot_32(a_row, b_col);
                                int32_t dot1 = corrected_dot_32(a_row + 32, b_col + 32);
                                int32_acc_[i * TILE_N + j] += (dot0 + dot1);
                                float b_scale0 = (float)b_scales[j];
                                float b_scale1 = b_scales2 ? (float)b_scales2[j] : b_scale0;
                                fp_accumulators_[i * TILE_N + j] += (float)dot0 * a_scale0 * b_scale0 + (float)dot1 * a_scale1 * b_scale1;
                            }
                        }
                    }
#endif
                }

                static void quantizeFP32ToQ8_0(const float *x, Q8_0Block *block, int n)
                {
                    float amax = 0.f;
                    for (int i = 0; i < n; ++i)
                        amax = std::max(amax, std::fabs(x[i]));
                    float scale = (amax > 0.f) ? (amax / 127.f) : 1.f;
                    float inv = 1.f / scale;
                    block->d = fp32_to_fp16(scale);
                    for (int i = 0; i < n; ++i)
                    {
                        float s = x[i] * inv;
                        int32_t q = (int32_t)std::lrintf(s);
                        q = std::max(-127, std::min(127, q));
                        block->qs[i] = (int8_t)q;
                    }
                }

                alignas(64) int32_t int32_acc_[TILE_M * TILE_N];
                alignas(64) float fp_accumulators_[TILE_M * TILE_N];
                int num_k_blocks_ = 0;
                static inline std::atomic<bool> use_simd_path_{true};
            };

        } // namespace gemm
    } // namespace kernels
} // namespace llaminar2
