/**
 * @file ActivationRotation.h
 * @brief Block-diagonal randomized Hadamard rotation for activation kurtosis reduction
 *
 * Applies a deterministic block-diagonal randomized Hadamard transform to FP32
 * activation vectors before quantization (Q8_1 or Q16_1). The transform spreads
 * outlier energy uniformly across quantization blocks, dramatically reducing
 * kurtosis and improving quantization fidelity.
 *
 * Algorithm: R = HD/√d where H is the Walsh-Hadamard matrix and D = diag(±1)
 * is a random sign-flip matrix. This is an orthogonal transform (RR^T = I).
 *
 * Forward:  y = R·x = H·(D·x) / √d   →  sign-flip → FWHT → scale
 * Inverse:  x = R^T·y = D·(H·y) / √d  →  FWHT → scale → sign-flip
 *
 * Complexity: O(d log d) per block vs O(d²) for dense matrix rotation.
 * For d=128: 896 FLOPs vs 16,384 — an 18× reduction.
 *
 * AVX-512 FWHT keeps all data in ZMM registers for the full transform,
 * using intra-register shuffles for small strides and inter-register
 * arithmetic for large strides.
 *
 * Mathematical basis:
 *   For GEMM y = X @ W^T:
 *     X' = X @ R     (rotate activations before quantization)
 *     W' = W @ R     (pre-rotate weights once at load time)
 *     X' @ W'^T = XR(WR)^T = XRR^TW^T = XW^T = y  (R orthogonal ⟹ RR^T = I)
 *
 * Usage:
 *   auto rot = std::make_shared<ActivationRotation>(hidden_dim, 128);
 *   rot->rotate_inplace(fp32_row, hidden_dim);         // Hot path
 *   rot->inverse_rotate_inplace(fp32_row, hidden_dim); // Hot path
 *   rot->rotate_rows_inplace(data, rows, dim);         // OpenMP parallel
 */

#pragma once

#include "tensors/TensorClasses.h"
#include "tensors/FP16Utils.h"
#include "utils/OpenMPUtils.h"
#include "utils/CPUFeatures.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#if defined(__AVX512F__) || defined(__AVX2__)
#include <immintrin.h>
#endif

#include <omp.h>

namespace llaminar2
{

    class ActivationRotation
    {
    public:
        /**
         * @brief Construct a block-diagonal randomized Hadamard rotation.
         *
         * @param total_dim   Total vector dimension (e.g., hidden_dim or head_dim)
         * @param block_dim   Block size (must be power of 2 and divide total_dim)
         * @param seed        Seed for random sign generation (default: 31)
         */
        explicit ActivationRotation(int total_dim, int block_dim, uint64_t seed = 31)
            : total_dim_(total_dim),
              block_dim_(block_dim),
              n_blocks_(total_dim / block_dim),
              inv_sqrt_d_(1.0f / std::sqrt(static_cast<float>(block_dim)))
        {
            assert(total_dim % block_dim == 0 &&
                   "total_dim must be divisible by block_dim");
            assert(block_dim > 0 && (block_dim & (block_dim - 1)) == 0 &&
                   "block_dim must be a power of 2 for FWHT");

            // Generate deterministic random sign flips: ±1.0f
            sign_flips_.resize(block_dim);
            std::mt19937_64 rng(seed);
            for (int i = 0; i < block_dim; ++i)
                sign_flips_[i] = (rng() & 1) ? 1.0f : -1.0f;
        }

        /// Total vector dimension
        int total_dim() const { return total_dim_; }

        /// Block dimension for rotation
        int block_dim() const { return block_dim_; }

        /// Number of blocks
        int n_blocks() const { return n_blocks_; }

        /**
         * @brief Apply block-diagonal rotation to a single FP32 row in-place.
         *
         * Forward: y = HD·x / √d  →  sign-flip → FWHT → scale
         */
        void rotate_inplace(float *row, int dim) const
        {
            assert(dim % block_dim_ == 0);
            const int d = block_dim_;
            const int nb = dim / d;

            for (int b = 0; b < nb; ++b)
            {
                float *chunk = row + b * d;
                apply_sign_flips(chunk, d);
                fwht_inplace(chunk, d);
                scale_block(chunk, d, inv_sqrt_d_);
            }
        }

        /**
         * @brief Apply inverse block-diagonal rotation to a single FP32 row in-place.
         *
         * Inverse: x = D·H·y / √d  →  FWHT → scale → sign-flip
         */
        void inverse_rotate_inplace(float *row, int dim) const
        {
            assert(dim % block_dim_ == 0);
            const int d = block_dim_;
            const int nb = dim / d;

            for (int b = 0; b < nb; ++b)
            {
                float *chunk = row + b * d;
                fwht_inplace(chunk, d);
                scale_block(chunk, d, inv_sqrt_d_);
                apply_sign_flips(chunk, d);
            }
        }

        /**
         * @brief Apply block-diagonal rotation to multiple FP32 rows in-place.
         *
         * Uses OMP_WORKSHARE_REGION for nested-safe OpenMP parallelism.
         */
        void rotate_rows_inplace(float *data, int rows, int dim) const
        {
            auto work = [&]()
            {
#pragma omp for schedule(static)
                for (int r = 0; r < rows; ++r)
                    rotate_inplace(data + r * dim, dim);
            };
            OMP_WORKSHARE_REGION(work);
        }

        /**
         * @brief Apply inverse block-diagonal rotation to multiple FP32 rows in-place.
         *
         * Uses OMP_WORKSHARE_REGION for nested-safe OpenMP parallelism.
         */
        void inverse_rotate_rows_inplace(float *data, int rows, int dim) const
        {
            auto work = [&]()
            {
#pragma omp for schedule(static)
                for (int r = 0; r < rows; ++r)
                    inverse_rotate_inplace(data + r * dim, dim);
            };
            OMP_WORKSHARE_REGION(work);
        }

        /**
         * @brief Pre-rotate weight matrix rows for GEMM compatibility.
         *
         * Called once during weight preparation (not hot path).
         */
        void rotate_weight_rows(float *weight_fp32, int n_rows, int k_dim) const
        {
            assert(k_dim == total_dim_);
            auto work = [&]()
            {
#pragma omp for schedule(static)
                for (int r = 0; r < n_rows; ++r)
                    rotate_inplace(weight_fp32 + r * k_dim, k_dim);
            };
            OMP_WORKSHARE_REGION(work);
        }

    private:
        // ====================================================================
        // Core FWHT primitives
        // ====================================================================

        /**
         * @brief Apply random sign flips element-wise: data[i] *= sign_flips_[i]
         */
        // Named ISA implementations: apply_sign_flips

        void apply_sign_flips_scalar(float *data, int n) const
        {
            const float *signs = sign_flips_.data();
            for (int i = 0; i < n; ++i)
                data[i] *= signs[i];
        }

#if defined(__AVX2__)
        void apply_sign_flips_avx2(float *data, int n) const
        {
            const float *signs = sign_flips_.data();
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                __m256 v = _mm256_loadu_ps(data + i);
                __m256 s = _mm256_loadu_ps(signs + i);
                _mm256_storeu_ps(data + i, _mm256_mul_ps(v, s));
            }
            for (; i < n; ++i)
                data[i] *= signs[i];
        }
#endif

#if defined(__AVX512F__)
        void apply_sign_flips_avx512(float *data, int n) const
        {
            const float *signs = sign_flips_.data();
            int i = 0;
            for (; i + 16 <= n; i += 16)
            {
                __m512 v = _mm512_loadu_ps(data + i);
                __m512 s = _mm512_loadu_ps(signs + i);
                _mm512_storeu_ps(data + i, _mm512_mul_ps(v, s));
            }
            for (; i < n; ++i)
                data[i] *= signs[i];
        }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
        void apply_sign_flips_avx2(float *data, int n) const { apply_sign_flips_scalar(data, n); }
#endif
#if !defined(__AVX512F__)
        void apply_sign_flips_avx512(float *data, int n) const { apply_sign_flips_avx2(data, n); }
#endif

        void apply_sign_flips(float *data, int n) const
        {
            switch (activeISALevel())
            {
            case ISALevel::AVX512:
                apply_sign_flips_avx512(data, n);
                break;
            case ISALevel::AVX2:
                apply_sign_flips_avx2(data, n);
                break;
            default:
                apply_sign_flips_scalar(data, n);
                break;
            }
        }

        /**
         * @brief Scale block by scalar: data[i] *= s
         */
        // Named ISA implementations: scale_block

        static void scale_block_scalar(float *data, int n, float s)
        {
            for (int i = 0; i < n; ++i)
                data[i] *= s;
        }

#if defined(__AVX2__)
        static void scale_block_avx2(float *data, int n, float s)
        {
            __m256 vs = _mm256_set1_ps(s);
            int i = 0;
            for (; i + 8 <= n; i += 8)
                _mm256_storeu_ps(data + i, _mm256_mul_ps(_mm256_loadu_ps(data + i), vs));
            for (; i < n; ++i)
                data[i] *= s;
        }
#endif

#if defined(__AVX512F__)
        static void scale_block_avx512(float *data, int n, float s)
        {
            __m512 vs = _mm512_set1_ps(s);
            int i = 0;
            for (; i + 16 <= n; i += 16)
                _mm512_storeu_ps(data + i, _mm512_mul_ps(_mm512_loadu_ps(data + i), vs));
            for (; i < n; ++i)
                data[i] *= s;
        }
#endif

// Stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
        static void scale_block_avx2(float *data, int n, float s) { scale_block_scalar(data, n, s); }
#endif
#if !defined(__AVX512F__)
        static void scale_block_avx512(float *data, int n, float s) { scale_block_avx2(data, n, s); }
#endif

        static void scale_block(float *data, int n, float s)
        {
            switch (activeISALevel())
            {
            case ISALevel::AVX512:
                scale_block_avx512(data, n, s);
                break;
            case ISALevel::AVX2:
                scale_block_avx2(data, n, s);
                break;
            default:
                scale_block_scalar(data, n, s);
                break;
            }
        }

        /**
         * @brief In-place Fast Walsh-Hadamard Transform.
         *
         * For n=64,128: uses AVX-512 register-resident transform.
         * Otherwise: scalar fallback for any power-of-2 size.
         */
        static void fwht_inplace(float *data, int n)
        {
            const auto isa = activeISALevel();
            if (n == 128)
            {
                if (isa >= ISALevel::AVX512)
                {
                    fwht_128_avx512(data);
                    return;
                }
                if (isa >= ISALevel::AVX2)
                {
                    fwht_128_avx2(data);
                    return;
                }
            }
            if (n == 64)
            {
                if (isa >= ISALevel::AVX512)
                {
                    fwht_64_avx512(data);
                    return;
                }
                if (isa >= ISALevel::AVX2)
                {
                    fwht_64_avx2(data);
                    return;
                }
            }
            fwht_scalar(data, n);
        }

        /**
         * @brief Scalar FWHT for any power-of-2 dimension.
         */
        static void fwht_scalar(float *data, int n)
        {
            for (int len = 1; len < n; len <<= 1)
            {
                for (int i = 0; i < n; i += len << 1)
                {
                    for (int j = 0; j < len; ++j)
                    {
                        float u = data[i + j];
                        float v = data[i + j + len];
                        data[i + j] = u + v;
                        data[i + j + len] = u - v;
                    }
                }
            }
        }

#if defined(__AVX512F__)
        /**
         * @brief Intra-register butterfly: stride 1 (swap adjacent pairs)
         *
         * [a0,a1,a2,a3,...] → [a0+a1, a0-a1, a2+a3, a2-a3, ...]
         */
        static inline __m512 butterfly_stride1(__m512 z)
        {
            __m512 shuffled = _mm512_permute_ps(z, 0b10110001); // swap adjacent: [a1,a0,a3,a2,...]
            __m512 add = _mm512_add_ps(z, shuffled);
            __m512 sub = _mm512_sub_ps(shuffled, z);       // shuffled-z so upper positions get lower-upper
            return _mm512_mask_blend_ps(0xAAAA, add, sub); // even from add, odd from sub
        }

        /**
         * @brief Intra-register butterfly: stride 2
         *
         * Pairs at distance 2 within each group of 4.
         */
        static inline __m512 butterfly_stride2(__m512 z)
        {
            __m512 shuffled = _mm512_permute_ps(z, 0b01001110); // swap pairs: [a2,a3,a0,a1,...]
            __m512 add = _mm512_add_ps(z, shuffled);
            __m512 sub = _mm512_sub_ps(shuffled, z);       // shuffled-z so upper positions get lower-upper
            return _mm512_mask_blend_ps(0xCCCC, add, sub); // bits 2,3,6,7,... from sub
        }

        /**
         * @brief Intra-register butterfly: stride 4
         *
         * Pairs at distance 4: swap 128-bit lanes within each 256-bit half.
         */
        static inline __m512 butterfly_stride4(__m512 z)
        {
            __m512 shuffled = _mm512_shuffle_f32x4(z, z, 0b10110001); // swap adjacent 128-bit lanes
            __m512 add = _mm512_add_ps(z, shuffled);
            __m512 sub = _mm512_sub_ps(shuffled, z); // shuffled-z so upper positions get lower-upper
            return _mm512_mask_blend_ps(0xF0F0, add, sub);
        }

        /**
         * @brief Intra-register butterfly: stride 8
         *
         * Pairs at distance 8: swap 256-bit halves.
         */
        static inline __m512 butterfly_stride8(__m512 z)
        {
            __m512 shuffled = _mm512_shuffle_f32x4(z, z, 0b01001110); // swap 256-bit halves
            __m512 add = _mm512_add_ps(z, shuffled);
            __m512 sub = _mm512_sub_ps(shuffled, z); // shuffled-z so upper positions get lower-upper
            return _mm512_mask_blend_ps(0xFF00, add, sub);
        }

        /**
         * @brief AVX-512 FWHT for 128-element blocks (head_dim=128).
         *
         * Loads all 128 floats into 8 ZMM registers, performs the full 7-stage
         * FWHT in-register, then stores back. Zero memory traffic during transform.
         */
        static void fwht_128_avx512(float *data)
        {
            // Load 128 floats into 8 ZMM registers
            __m512 z0 = _mm512_loadu_ps(data);
            __m512 z1 = _mm512_loadu_ps(data + 16);
            __m512 z2 = _mm512_loadu_ps(data + 32);
            __m512 z3 = _mm512_loadu_ps(data + 48);
            __m512 z4 = _mm512_loadu_ps(data + 64);
            __m512 z5 = _mm512_loadu_ps(data + 80);
            __m512 z6 = _mm512_loadu_ps(data + 96);
            __m512 z7 = _mm512_loadu_ps(data + 112);

            // Stage 0 (len=1): butterfly pairs at stride 1 within each ZMM
            z0 = butterfly_stride1(z0);
            z1 = butterfly_stride1(z1);
            z2 = butterfly_stride1(z2);
            z3 = butterfly_stride1(z3);
            z4 = butterfly_stride1(z4);
            z5 = butterfly_stride1(z5);
            z6 = butterfly_stride1(z6);
            z7 = butterfly_stride1(z7);

            // Stage 1 (len=2): butterfly pairs at stride 2
            z0 = butterfly_stride2(z0);
            z1 = butterfly_stride2(z1);
            z2 = butterfly_stride2(z2);
            z3 = butterfly_stride2(z3);
            z4 = butterfly_stride2(z4);
            z5 = butterfly_stride2(z5);
            z6 = butterfly_stride2(z6);
            z7 = butterfly_stride2(z7);

            // Stage 2 (len=4): butterfly pairs at stride 4
            z0 = butterfly_stride4(z0);
            z1 = butterfly_stride4(z1);
            z2 = butterfly_stride4(z2);
            z3 = butterfly_stride4(z3);
            z4 = butterfly_stride4(z4);
            z5 = butterfly_stride4(z5);
            z6 = butterfly_stride4(z6);
            z7 = butterfly_stride4(z7);

            // Stage 3 (len=8): butterfly pairs at stride 8
            z0 = butterfly_stride8(z0);
            z1 = butterfly_stride8(z1);
            z2 = butterfly_stride8(z2);
            z3 = butterfly_stride8(z3);
            z4 = butterfly_stride8(z4);
            z5 = butterfly_stride8(z5);
            z6 = butterfly_stride8(z6);
            z7 = butterfly_stride8(z7);

            // Stage 4 (len=16): butterfly between adjacent ZMM pairs
            {
                __m512 a, b;
                a = _mm512_add_ps(z0, z1);
                b = _mm512_sub_ps(z0, z1);
                z0 = a;
                z1 = b;
                a = _mm512_add_ps(z2, z3);
                b = _mm512_sub_ps(z2, z3);
                z2 = a;
                z3 = b;
                a = _mm512_add_ps(z4, z5);
                b = _mm512_sub_ps(z4, z5);
                z4 = a;
                z5 = b;
                a = _mm512_add_ps(z6, z7);
                b = _mm512_sub_ps(z6, z7);
                z6 = a;
                z7 = b;
            }

            // Stage 5 (len=32): butterfly between ZMM groups of 2
            {
                __m512 a0, b0, a1, b1;
                a0 = _mm512_add_ps(z0, z2);
                b0 = _mm512_sub_ps(z0, z2);
                a1 = _mm512_add_ps(z1, z3);
                b1 = _mm512_sub_ps(z1, z3);
                z0 = a0;
                z2 = b0;
                z1 = a1;
                z3 = b1;

                a0 = _mm512_add_ps(z4, z6);
                b0 = _mm512_sub_ps(z4, z6);
                a1 = _mm512_add_ps(z5, z7);
                b1 = _mm512_sub_ps(z5, z7);
                z4 = a0;
                z6 = b0;
                z5 = a1;
                z7 = b1;
            }

            // Stage 6 (len=64): butterfly between ZMM halves (0-3 vs 4-7)
            {
                __m512 a, b;
                a = _mm512_add_ps(z0, z4);
                b = _mm512_sub_ps(z0, z4);
                z0 = a;
                z4 = b;
                a = _mm512_add_ps(z1, z5);
                b = _mm512_sub_ps(z1, z5);
                z1 = a;
                z5 = b;
                a = _mm512_add_ps(z2, z6);
                b = _mm512_sub_ps(z2, z6);
                z2 = a;
                z6 = b;
                a = _mm512_add_ps(z3, z7);
                b = _mm512_sub_ps(z3, z7);
                z3 = a;
                z7 = b;
            }

            // Store back
            _mm512_storeu_ps(data, z0);
            _mm512_storeu_ps(data + 16, z1);
            _mm512_storeu_ps(data + 32, z2);
            _mm512_storeu_ps(data + 48, z3);
            _mm512_storeu_ps(data + 64, z4);
            _mm512_storeu_ps(data + 80, z5);
            _mm512_storeu_ps(data + 96, z6);
            _mm512_storeu_ps(data + 112, z7);
        }

        /**
         * @brief AVX-512 FWHT for 64-element blocks (head_dim=64).
         *
         * Loads 64 floats into 4 ZMM registers.
         */
        static void fwht_64_avx512(float *data)
        {
            __m512 z0 = _mm512_loadu_ps(data);
            __m512 z1 = _mm512_loadu_ps(data + 16);
            __m512 z2 = _mm512_loadu_ps(data + 32);
            __m512 z3 = _mm512_loadu_ps(data + 48);

            // Stage 0-3: intra-register butterflies
            z0 = butterfly_stride1(z0);
            z1 = butterfly_stride1(z1);
            z2 = butterfly_stride1(z2);
            z3 = butterfly_stride1(z3);

            z0 = butterfly_stride2(z0);
            z1 = butterfly_stride2(z1);
            z2 = butterfly_stride2(z2);
            z3 = butterfly_stride2(z3);

            z0 = butterfly_stride4(z0);
            z1 = butterfly_stride4(z1);
            z2 = butterfly_stride4(z2);
            z3 = butterfly_stride4(z3);

            z0 = butterfly_stride8(z0);
            z1 = butterfly_stride8(z1);
            z2 = butterfly_stride8(z2);
            z3 = butterfly_stride8(z3);

            // Stage 4 (len=16): between adjacent pairs
            {
                __m512 a, b;
                a = _mm512_add_ps(z0, z1);
                b = _mm512_sub_ps(z0, z1);
                z0 = a;
                z1 = b;
                a = _mm512_add_ps(z2, z3);
                b = _mm512_sub_ps(z2, z3);
                z2 = a;
                z3 = b;
            }

            // Stage 5 (len=32): between groups
            {
                __m512 a0, b0, a1, b1;
                a0 = _mm512_add_ps(z0, z2);
                b0 = _mm512_sub_ps(z0, z2);
                a1 = _mm512_add_ps(z1, z3);
                b1 = _mm512_sub_ps(z1, z3);
                z0 = a0;
                z2 = b0;
                z1 = a1;
                z3 = b1;
            }

            // Store back
            _mm512_storeu_ps(data, z0);
            _mm512_storeu_ps(data + 16, z1);
            _mm512_storeu_ps(data + 32, z2);
            _mm512_storeu_ps(data + 48, z3);
        }
#endif // __AVX512F__

#if defined(__AVX2__)
        // ================================================================
        // AVX2 butterfly functions (8-wide YMM)
        // ================================================================

        /** @brief Stride 1: swap adjacent pairs [a0,a1,a2,a3,...] */
        static inline __m256 butterfly_stride1_avx2(__m256 y)
        {
            __m256 shuffled = _mm256_permute_ps(y, 0b10110001); // swap adjacent
            __m256 add = _mm256_add_ps(y, shuffled);
            __m256 sub = _mm256_sub_ps(shuffled, y);
            return _mm256_blend_ps(add, sub, 0xAA); // 0xAA = 10101010
        }

        /** @brief Stride 2: swap pairs at distance 2 */
        static inline __m256 butterfly_stride2_avx2(__m256 y)
        {
            __m256 shuffled = _mm256_permute_ps(y, 0b01001110); // swap pairs
            __m256 add = _mm256_add_ps(y, shuffled);
            __m256 sub = _mm256_sub_ps(shuffled, y);
            return _mm256_blend_ps(add, sub, 0xCC); // 0xCC = 11001100
        }

        /** @brief Stride 4: swap 128-bit lanes */
        static inline __m256 butterfly_stride4_avx2(__m256 y)
        {
            __m256 shuffled = _mm256_permute2f128_ps(y, y, 0x01); // swap 128-bit halves
            __m256 add = _mm256_add_ps(y, shuffled);
            __m256 sub = _mm256_sub_ps(shuffled, y);
            return _mm256_blend_ps(add, sub, 0xF0); // 0xF0 = 11110000
        }

        /**
         * @brief AVX2 FWHT for 64-element blocks.
         * Uses 8 YMM registers (y0..y7). 6 stages.
         */
        static void fwht_64_avx2(float *data)
        {
            __m256 y0 = _mm256_loadu_ps(data);
            __m256 y1 = _mm256_loadu_ps(data + 8);
            __m256 y2 = _mm256_loadu_ps(data + 16);
            __m256 y3 = _mm256_loadu_ps(data + 24);
            __m256 y4 = _mm256_loadu_ps(data + 32);
            __m256 y5 = _mm256_loadu_ps(data + 40);
            __m256 y6 = _mm256_loadu_ps(data + 48);
            __m256 y7 = _mm256_loadu_ps(data + 56);

            // Stage 0 (stride 1)
            y0 = butterfly_stride1_avx2(y0);
            y1 = butterfly_stride1_avx2(y1);
            y2 = butterfly_stride1_avx2(y2);
            y3 = butterfly_stride1_avx2(y3);
            y4 = butterfly_stride1_avx2(y4);
            y5 = butterfly_stride1_avx2(y5);
            y6 = butterfly_stride1_avx2(y6);
            y7 = butterfly_stride1_avx2(y7);

            // Stage 1 (stride 2)
            y0 = butterfly_stride2_avx2(y0);
            y1 = butterfly_stride2_avx2(y1);
            y2 = butterfly_stride2_avx2(y2);
            y3 = butterfly_stride2_avx2(y3);
            y4 = butterfly_stride2_avx2(y4);
            y5 = butterfly_stride2_avx2(y5);
            y6 = butterfly_stride2_avx2(y6);
            y7 = butterfly_stride2_avx2(y7);

            // Stage 2 (stride 4)
            y0 = butterfly_stride4_avx2(y0);
            y1 = butterfly_stride4_avx2(y1);
            y2 = butterfly_stride4_avx2(y2);
            y3 = butterfly_stride4_avx2(y3);
            y4 = butterfly_stride4_avx2(y4);
            y5 = butterfly_stride4_avx2(y5);
            y6 = butterfly_stride4_avx2(y6);
            y7 = butterfly_stride4_avx2(y7);

            // Stage 3 (stride 8): between adjacent YMM pairs
            {
                __m256 a, b;
                a = _mm256_add_ps(y0, y1);
                b = _mm256_sub_ps(y0, y1);
                y0 = a;
                y1 = b;
                a = _mm256_add_ps(y2, y3);
                b = _mm256_sub_ps(y2, y3);
                y2 = a;
                y3 = b;
                a = _mm256_add_ps(y4, y5);
                b = _mm256_sub_ps(y4, y5);
                y4 = a;
                y5 = b;
                a = _mm256_add_ps(y6, y7);
                b = _mm256_sub_ps(y6, y7);
                y6 = a;
                y7 = b;
            }

            // Stage 4 (stride 16): between groups of 2
            {
                __m256 a0, b0, a1, b1;
                a0 = _mm256_add_ps(y0, y2);
                b0 = _mm256_sub_ps(y0, y2);
                a1 = _mm256_add_ps(y1, y3);
                b1 = _mm256_sub_ps(y1, y3);
                y0 = a0;
                y2 = b0;
                y1 = a1;
                y3 = b1;
                a0 = _mm256_add_ps(y4, y6);
                b0 = _mm256_sub_ps(y4, y6);
                a1 = _mm256_add_ps(y5, y7);
                b1 = _mm256_sub_ps(y5, y7);
                y4 = a0;
                y6 = b0;
                y5 = a1;
                y7 = b1;
            }

            // Stage 5 (stride 32): between halves (0-3 vs 4-7)
            {
                __m256 a, b;
                a = _mm256_add_ps(y0, y4);
                b = _mm256_sub_ps(y0, y4);
                y0 = a;
                y4 = b;
                a = _mm256_add_ps(y1, y5);
                b = _mm256_sub_ps(y1, y5);
                y1 = a;
                y5 = b;
                a = _mm256_add_ps(y2, y6);
                b = _mm256_sub_ps(y2, y6);
                y2 = a;
                y6 = b;
                a = _mm256_add_ps(y3, y7);
                b = _mm256_sub_ps(y3, y7);
                y3 = a;
                y7 = b;
            }

            _mm256_storeu_ps(data, y0);
            _mm256_storeu_ps(data + 8, y1);
            _mm256_storeu_ps(data + 16, y2);
            _mm256_storeu_ps(data + 24, y3);
            _mm256_storeu_ps(data + 32, y4);
            _mm256_storeu_ps(data + 40, y5);
            _mm256_storeu_ps(data + 48, y6);
            _mm256_storeu_ps(data + 56, y7);
        }

        /**
         * @brief AVX2 FWHT for 128-element blocks.
         * Uses 16 YMM registers (y0..y15). 7 stages.
         */
        static void fwht_128_avx2(float *data)
        {
            __m256 y0 = _mm256_loadu_ps(data);
            __m256 y1 = _mm256_loadu_ps(data + 8);
            __m256 y2 = _mm256_loadu_ps(data + 16);
            __m256 y3 = _mm256_loadu_ps(data + 24);
            __m256 y4 = _mm256_loadu_ps(data + 32);
            __m256 y5 = _mm256_loadu_ps(data + 40);
            __m256 y6 = _mm256_loadu_ps(data + 48);
            __m256 y7 = _mm256_loadu_ps(data + 56);
            __m256 y8 = _mm256_loadu_ps(data + 64);
            __m256 y9 = _mm256_loadu_ps(data + 72);
            __m256 yA = _mm256_loadu_ps(data + 80);
            __m256 yB = _mm256_loadu_ps(data + 88);
            __m256 yC = _mm256_loadu_ps(data + 96);
            __m256 yD = _mm256_loadu_ps(data + 104);
            __m256 yE = _mm256_loadu_ps(data + 112);
            __m256 yF = _mm256_loadu_ps(data + 120);

            // Stage 0 (stride 1)
            y0 = butterfly_stride1_avx2(y0);
            y1 = butterfly_stride1_avx2(y1);
            y2 = butterfly_stride1_avx2(y2);
            y3 = butterfly_stride1_avx2(y3);
            y4 = butterfly_stride1_avx2(y4);
            y5 = butterfly_stride1_avx2(y5);
            y6 = butterfly_stride1_avx2(y6);
            y7 = butterfly_stride1_avx2(y7);
            y8 = butterfly_stride1_avx2(y8);
            y9 = butterfly_stride1_avx2(y9);
            yA = butterfly_stride1_avx2(yA);
            yB = butterfly_stride1_avx2(yB);
            yC = butterfly_stride1_avx2(yC);
            yD = butterfly_stride1_avx2(yD);
            yE = butterfly_stride1_avx2(yE);
            yF = butterfly_stride1_avx2(yF);

            // Stage 1 (stride 2)
            y0 = butterfly_stride2_avx2(y0);
            y1 = butterfly_stride2_avx2(y1);
            y2 = butterfly_stride2_avx2(y2);
            y3 = butterfly_stride2_avx2(y3);
            y4 = butterfly_stride2_avx2(y4);
            y5 = butterfly_stride2_avx2(y5);
            y6 = butterfly_stride2_avx2(y6);
            y7 = butterfly_stride2_avx2(y7);
            y8 = butterfly_stride2_avx2(y8);
            y9 = butterfly_stride2_avx2(y9);
            yA = butterfly_stride2_avx2(yA);
            yB = butterfly_stride2_avx2(yB);
            yC = butterfly_stride2_avx2(yC);
            yD = butterfly_stride2_avx2(yD);
            yE = butterfly_stride2_avx2(yE);
            yF = butterfly_stride2_avx2(yF);

            // Stage 2 (stride 4)
            y0 = butterfly_stride4_avx2(y0);
            y1 = butterfly_stride4_avx2(y1);
            y2 = butterfly_stride4_avx2(y2);
            y3 = butterfly_stride4_avx2(y3);
            y4 = butterfly_stride4_avx2(y4);
            y5 = butterfly_stride4_avx2(y5);
            y6 = butterfly_stride4_avx2(y6);
            y7 = butterfly_stride4_avx2(y7);
            y8 = butterfly_stride4_avx2(y8);
            y9 = butterfly_stride4_avx2(y9);
            yA = butterfly_stride4_avx2(yA);
            yB = butterfly_stride4_avx2(yB);
            yC = butterfly_stride4_avx2(yC);
            yD = butterfly_stride4_avx2(yD);
            yE = butterfly_stride4_avx2(yE);
            yF = butterfly_stride4_avx2(yF);

            // Stage 3 (stride 8): adjacent pairs
            {
                __m256 a, b;
                a = _mm256_add_ps(y0, y1);
                b = _mm256_sub_ps(y0, y1);
                y0 = a;
                y1 = b;
                a = _mm256_add_ps(y2, y3);
                b = _mm256_sub_ps(y2, y3);
                y2 = a;
                y3 = b;
                a = _mm256_add_ps(y4, y5);
                b = _mm256_sub_ps(y4, y5);
                y4 = a;
                y5 = b;
                a = _mm256_add_ps(y6, y7);
                b = _mm256_sub_ps(y6, y7);
                y6 = a;
                y7 = b;
                a = _mm256_add_ps(y8, y9);
                b = _mm256_sub_ps(y8, y9);
                y8 = a;
                y9 = b;
                a = _mm256_add_ps(yA, yB);
                b = _mm256_sub_ps(yA, yB);
                yA = a;
                yB = b;
                a = _mm256_add_ps(yC, yD);
                b = _mm256_sub_ps(yC, yD);
                yC = a;
                yD = b;
                a = _mm256_add_ps(yE, yF);
                b = _mm256_sub_ps(yE, yF);
                yE = a;
                yF = b;
            }

            // Stage 4 (stride 16): groups of 2
            {
                __m256 a0, b0, a1, b1;
                a0 = _mm256_add_ps(y0, y2);
                b0 = _mm256_sub_ps(y0, y2);
                a1 = _mm256_add_ps(y1, y3);
                b1 = _mm256_sub_ps(y1, y3);
                y0 = a0;
                y2 = b0;
                y1 = a1;
                y3 = b1;
                a0 = _mm256_add_ps(y4, y6);
                b0 = _mm256_sub_ps(y4, y6);
                a1 = _mm256_add_ps(y5, y7);
                b1 = _mm256_sub_ps(y5, y7);
                y4 = a0;
                y6 = b0;
                y5 = a1;
                y7 = b1;
                a0 = _mm256_add_ps(y8, yA);
                b0 = _mm256_sub_ps(y8, yA);
                a1 = _mm256_add_ps(y9, yB);
                b1 = _mm256_sub_ps(y9, yB);
                y8 = a0;
                yA = b0;
                y9 = a1;
                yB = b1;
                a0 = _mm256_add_ps(yC, yE);
                b0 = _mm256_sub_ps(yC, yE);
                a1 = _mm256_add_ps(yD, yF);
                b1 = _mm256_sub_ps(yD, yF);
                yC = a0;
                yE = b0;
                yD = a1;
                yF = b1;
            }

            // Stage 5 (stride 32): groups of 4
            {
                __m256 a, b;
                a = _mm256_add_ps(y0, y4);
                b = _mm256_sub_ps(y0, y4);
                y0 = a;
                y4 = b;
                a = _mm256_add_ps(y1, y5);
                b = _mm256_sub_ps(y1, y5);
                y1 = a;
                y5 = b;
                a = _mm256_add_ps(y2, y6);
                b = _mm256_sub_ps(y2, y6);
                y2 = a;
                y6 = b;
                a = _mm256_add_ps(y3, y7);
                b = _mm256_sub_ps(y3, y7);
                y3 = a;
                y7 = b;
                a = _mm256_add_ps(y8, yC);
                b = _mm256_sub_ps(y8, yC);
                y8 = a;
                yC = b;
                a = _mm256_add_ps(y9, yD);
                b = _mm256_sub_ps(y9, yD);
                y9 = a;
                yD = b;
                a = _mm256_add_ps(yA, yE);
                b = _mm256_sub_ps(yA, yE);
                yA = a;
                yE = b;
                a = _mm256_add_ps(yB, yF);
                b = _mm256_sub_ps(yB, yF);
                yB = a;
                yF = b;
            }

            // Stage 6 (stride 64): halves (0-7 vs 8-F)
            {
                __m256 a, b;
                a = _mm256_add_ps(y0, y8);
                b = _mm256_sub_ps(y0, y8);
                y0 = a;
                y8 = b;
                a = _mm256_add_ps(y1, y9);
                b = _mm256_sub_ps(y1, y9);
                y1 = a;
                y9 = b;
                a = _mm256_add_ps(y2, yA);
                b = _mm256_sub_ps(y2, yA);
                y2 = a;
                yA = b;
                a = _mm256_add_ps(y3, yB);
                b = _mm256_sub_ps(y3, yB);
                y3 = a;
                yB = b;
                a = _mm256_add_ps(y4, yC);
                b = _mm256_sub_ps(y4, yC);
                y4 = a;
                yC = b;
                a = _mm256_add_ps(y5, yD);
                b = _mm256_sub_ps(y5, yD);
                y5 = a;
                yD = b;
                a = _mm256_add_ps(y6, yE);
                b = _mm256_sub_ps(y6, yE);
                y6 = a;
                yE = b;
                a = _mm256_add_ps(y7, yF);
                b = _mm256_sub_ps(y7, yF);
                y7 = a;
                yF = b;
            }

            _mm256_storeu_ps(data, y0);
            _mm256_storeu_ps(data + 8, y1);
            _mm256_storeu_ps(data + 16, y2);
            _mm256_storeu_ps(data + 24, y3);
            _mm256_storeu_ps(data + 32, y4);
            _mm256_storeu_ps(data + 40, y5);
            _mm256_storeu_ps(data + 48, y6);
            _mm256_storeu_ps(data + 56, y7);
            _mm256_storeu_ps(data + 64, y8);
            _mm256_storeu_ps(data + 72, y9);
            _mm256_storeu_ps(data + 80, yA);
            _mm256_storeu_ps(data + 88, yB);
            _mm256_storeu_ps(data + 96, yC);
            _mm256_storeu_ps(data + 104, yD);
            _mm256_storeu_ps(data + 112, yE);
            _mm256_storeu_ps(data + 120, yF);
        }
#endif // __AVX2__

// FWHT stubs for when ISA is unavailable at compile time
#if !defined(__AVX2__)
        static void fwht_64_avx2(float *data) { fwht_scalar(data, 64); }
        static void fwht_128_avx2(float *data) { fwht_scalar(data, 128); }
#endif
#if !defined(__AVX512F__)
        static void fwht_64_avx512(float *data) { fwht_64_avx2(data); }
        static void fwht_128_avx512(float *data) { fwht_128_avx2(data); }
#endif

        int total_dim_;
        int block_dim_;
        int n_blocks_;
        float inv_sqrt_d_;
        std::vector<float> sign_flips_; ///< Random ±1.0f signs (length block_dim)
    };

} // namespace llaminar2
