/**
 * @file VNNIGemm_Complete.h
 * @brief VNNI-optimized INT8 GEMM kernel with pre-packed panel layout (Header-Only)
 * @author David Sanftenberg
 *
 * This is a complete header-only implementation following the gemm_v2 pattern.
 * All template implementations are included for explicit instantiation.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <omp.h>

namespace llaminar2
{

    // Packed A layout: 4x4-grouped for VNNI
    struct PackedA
    {
        int8_t *data;
        int ld_tile;
        int M_R;
        int K_BLK;

        inline int groups() const { return M_R / 4; }
        inline int k_chunks() const { return K_BLK / 4; }
        inline int group_stride() const { return k_chunks() * 16; }
    };

    // Packed B layout: Column-major K-contiguous
    struct PackedB
    {
        int8_t *data;
        int ld_block;
        int ld_col;
        int N;
        int K_BLK;

        inline const int8_t *block_ptr(int t) const
        {
            return data + t * ld_block;
        }
    };

    // ---------- A PACKING (4x4 GROUPED) - VECTORIZED ----------

    template <int M_R, int K_BLK>
    void pack_A_tile_4x4_grouped(
        const int8_t *__restrict A,
        int M, int K,
        int M0, int k0,
        int mr, int kblk,
        int8_t *__restrict A_tile_packed)
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int K_chunks = kblk / 4;
        const int group_stride = K_chunks * 16;

        // Pack real rows with vectorization and ILP
        for (int m_base = 0; m_base < mr; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;

            // Prefetch next group if available
            if (m_base + 4 < mr)
            {
                _mm_prefetch(reinterpret_cast<const char *>(A + (M0 + m_base + 4) * K + k0), _MM_HINT_T0);
            }

            // Check if all 4 rows are valid
            const bool all_valid = (m_base + 3 < mr) && (M0 + m_base + 3 < M);

            if (all_valid)
            {
                // Fast path: All 4 rows valid, use vectorized loads
                // Process K chunks with unrolling for ILP (2-way)
                int kk = 0;
                for (; kk + 1 < K_chunks; kk += 2)
                {
                    // Chunk 0
                    int8_t *dst0 = group_ptr + kk * 16;
                    alignas(16) int8_t temp0[16];

                    // Chunk 1 (ILP - independent of chunk 0)
                    int8_t *dst1 = group_ptr + (kk + 1) * 16;
                    alignas(16) int8_t temp1[16];

                    // Interleave loads for chunks 0 and 1 to exploit dual load ports (2 lanes at a time)
                    for (int lane = 0; lane < 4; lane += 2)
                    {
                        // Lane 0 of chunk 0 and lane 0 of chunk 1 (parallel load port exploitation)
                        const int m0 = M0 + m_base + lane;
                        const int m1 = M0 + m_base + lane;
                        const int8_t *src0_chunk0 = A + m0 * K + (k0 + kk * 4);
                        const int8_t *src0_chunk1 = A + m1 * K + (k0 + (kk + 1) * 4);
                        *reinterpret_cast<int32_t *>(&temp0[lane * 4]) = *reinterpret_cast<const int32_t *>(src0_chunk0);
                        *reinterpret_cast<int32_t *>(&temp1[lane * 4]) = *reinterpret_cast<const int32_t *>(src0_chunk1);

                        // Lane 1 of chunk 0 and lane 1 of chunk 1 (parallel load port exploitation)
                        const int m2 = M0 + m_base + lane + 1;
                        const int m3 = M0 + m_base + lane + 1;
                        const int8_t *src1_chunk0 = A + m2 * K + (k0 + kk * 4);
                        const int8_t *src1_chunk1 = A + m3 * K + (k0 + (kk + 1) * 4);
                        *reinterpret_cast<int32_t *>(&temp0[(lane + 1) * 4]) = *reinterpret_cast<const int32_t *>(src1_chunk0);
                        *reinterpret_cast<int32_t *>(&temp1[(lane + 1) * 4]) = *reinterpret_cast<const int32_t *>(src1_chunk1);
                    }

                    _mm_store_si128(reinterpret_cast<__m128i *>(dst0), _mm_load_si128(reinterpret_cast<const __m128i *>(temp0)));
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst1), _mm_load_si128(reinterpret_cast<const __m128i *>(temp1)));
                }

                // Tail handling for odd K_chunks
                for (; kk < K_chunks; ++kk)
                {
                    int8_t *dst = group_ptr + kk * 16;
                    alignas(16) int8_t temp[16];

                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const int m = M0 + m_base + lane;
                        const int8_t *src = A + m * K + (k0 + kk * 4);
                        *reinterpret_cast<int32_t *>(&temp[lane * 4]) = *reinterpret_cast<const int32_t *>(src);
                    }
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst), _mm_load_si128(reinterpret_cast<const __m128i *>(temp)));
                }
            }
            else
            {
                // Slow path: Handle partial rows (boundary case)
                for (int kk = 0; kk < K_chunks; ++kk)
                {
                    int8_t *dst = group_ptr + kk * 16;

                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const int m = M0 + m_base + lane;
                        const int row_in_tile = m_base + lane;

                        if (row_in_tile < mr && m < M)
                        {
                            const int8_t *src = A + m * K + (k0 + kk * 4);
                            *reinterpret_cast<int32_t *>(&dst[lane * 4]) = *reinterpret_cast<const int32_t *>(src);
                        }
                        else
                        {
                            // Zero-pad partial rows
                            *reinterpret_cast<int32_t *>(&dst[lane * 4]) = 0;
                        }
                    }
                }
            }
        }

        // Zero-pad remaining groups if mr < M_R (vectorized)
        for (int m_base = mr; m_base < M_R; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;

            // Use vector stores for zeroing (faster than memset for aligned data)
            const __m128i zero = _mm_setzero_si128();
            for (int i = 0; i < group_stride; i += 16)
            {
                _mm_store_si128(reinterpret_cast<__m128i *>(group_ptr + i), zero);
            }
        }
    }

    // Explicit instantiations can be added as needed, or keep it header-only via templates.

    // ---------- B PACKING (VNNI-FRIENDLY) - VECTORIZED ----------

    template <int K_BLK>
    void pack_B_panel_vnni(
        const int8_t *__restrict B,
        int K, int N,
        int k0,
        int n0, int nr,
        int8_t *__restrict B_packed_panel,
        int &ld_block_B_out,
        int &ld_col_B_out)
    {
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        // Set strides:
        ld_col_B_out = K_BLK;
        ld_block_B_out = nr * ld_col_B_out;

        // Process columns with unrolling for ILP (4-way)
        int n = 0;
        for (; n + 3 < nr; n += 4)
        {
            const bool all_valid = (n0 + n + 3 < N) && (k0 + K_BLK - 1 < K);

            if (all_valid)
            {
                // Fast path: All 4 columns and all K elements valid
                for (int col = 0; col < 4; ++col)
                {
                    const int col_idx = n0 + n + col;
                    const int8_t *src_col = B + col_idx; // B[K x N], row-major
                    int8_t *dst_col = B_packed_panel + (n + col) * ld_col_B_out;

                    // Vectorized gather: process K_BLK elements
                    // For strided access with stride N, we need scalar loop or gather
                    // Use 64-byte chunks with AVX-512 gather when K_BLK >= 64
                    int kk = 0;

                    if constexpr (K_BLK >= 64)
                    {
                        // AVX-512 path: 16-element gather
                        for (; kk + 15 < K_BLK; kk += 16)
                        {
                            // Build index vector for gather (offsets in bytes)
                            alignas(64) int32_t indices[16];
                            for (int i = 0; i < 16; ++i)
                            {
                                indices[i] = (k0 + kk + i) * N;
                            }
                            __m512i idx = _mm512_load_si512(reinterpret_cast<const __m512i *>(indices));

                            // Gather 16 bytes
                            __m128i gathered = _mm512_cvtepi32_epi8(
                                _mm512_i32gather_epi32(idx, src_col, 1));
                            _mm_store_si128(reinterpret_cast<__m128i *>(dst_col + kk), gathered);
                        }
                    }

                    // AVX2 8-way gather tail
                    for (; kk + 7 < K_BLK; kk += 8)
                    {
                        alignas(32) int32_t indices[8];
                        for (int i = 0; i < 8; ++i)
                        {
                            indices[i] = (k0 + kk + i) * N;
                        }
                        __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i *>(indices));

                        // Gather 8 int32, extract low bytes
                        __m256i gathered32 = _mm256_i32gather_epi32(
                            reinterpret_cast<const int *>(src_col), idx, 1);

                        // Extract bytes: gather returns 8 x int32, we want the low byte of each
                        // Pack 8 x int32 -> 8 x int8
                        __m128i low128 = _mm256_castsi256_si128(gathered32);       // Extract low 4
                        __m128i high128 = _mm256_extracti128_si256(gathered32, 1); // Extract high 4

                        // Shuffle to get low bytes: 0,4,8,12 from each 128-bit lane
                        const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m128i low_bytes = _mm_shuffle_epi8(low128, shuffle);
                        __m128i high_bytes = _mm_shuffle_epi8(high128, shuffle);

                        // Combine: low bytes in positions 0-3, high bytes in 4-7
                        int32_t result_low = _mm_extract_epi32(low_bytes, 0);
                        int32_t result_high = _mm_extract_epi32(high_bytes, 0);

                        *reinterpret_cast<int32_t *>(dst_col + kk) = result_low;
                        *reinterpret_cast<int32_t *>(dst_col + kk + 4) = result_high;
                    }

                    // AVX2 4-way gather tail (vectorized with 128-bit)
                    for (; kk + 3 < K_BLK; kk += 4)
                    {
                        alignas(16) int32_t indices[4];
                        for (int i = 0; i < 4; ++i)
                        {
                            indices[i] = (k0 + kk + i) * N;
                        }
                        __m128i idx = _mm_load_si128(reinterpret_cast<const __m128i *>(indices));
                        __m128i gathered32 = _mm_i32gather_epi32(
                            reinterpret_cast<const int *>(src_col), idx, 1);

                        // Extract low bytes: 0, 4, 8, 12
                        const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m128i bytes = _mm_shuffle_epi8(gathered32, shuffle);
                        int32_t result = _mm_extract_epi32(bytes, 0);
                        *reinterpret_cast<int32_t *>(dst_col + kk) = result;
                    }

                    // Scalar tail (1-3 elements)
                    for (; kk < K_BLK; ++kk)
                    {
                        dst_col[kk] = src_col[(k0 + kk) * N];
                    }
                }
            }
            else
            {
                // Slow path: Boundary handling
                for (int col = 0; col < 4; ++col)
                {
                    const int col_idx = n0 + n + col;
                    int8_t *dst_col = B_packed_panel + (n + col) * ld_col_B_out;

                    if (col_idx < N)
                    {
                        const int8_t *src_col = B + col_idx;

                        // Vectorized K loop with same hierarchical pattern
                        int kk = 0;

                        // AVX-512 16-way
                        if constexpr (K_BLK >= 64)
                        {
                            for (; kk + 15 < K_BLK; kk += 16)
                            {
                                if (k0 + kk + 15 < K)
                                {
                                    alignas(64) int32_t indices[16];
                                    for (int i = 0; i < 16; ++i)
                                    {
                                        indices[i] = (k0 + kk + i) * N;
                                    }
                                    __m512i idx = _mm512_load_si512(reinterpret_cast<const __m512i *>(indices));
                                    __m128i gathered = _mm512_cvtepi32_epi8(
                                        _mm512_i32gather_epi32(idx, src_col, 1));
                                    _mm_store_si128(reinterpret_cast<__m128i *>(dst_col + kk), gathered);
                                }
                                else
                                {
                                    break;
                                }
                            }
                        }

                        // AVX2 8-way
                        for (; kk + 7 < K_BLK && k0 + kk + 7 < K; kk += 8)
                        {
                            alignas(32) int32_t indices[8];
                            for (int i = 0; i < 8; ++i)
                            {
                                indices[i] = (k0 + kk + i) * N;
                            }
                            __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i *>(indices));
                            __m256i gathered32 = _mm256_i32gather_epi32(
                                reinterpret_cast<const int *>(src_col), idx, 1);

                            __m128i low128 = _mm256_castsi256_si128(gathered32);
                            __m128i high128 = _mm256_extracti128_si256(gathered32, 1);

                            const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                            __m128i low_bytes = _mm_shuffle_epi8(low128, shuffle);
                            __m128i high_bytes = _mm_shuffle_epi8(high128, shuffle);

                            int32_t result_low = _mm_extract_epi32(low_bytes, 0);
                            int32_t result_high = _mm_extract_epi32(high_bytes, 0);

                            *reinterpret_cast<int32_t *>(dst_col + kk) = result_low;
                            *reinterpret_cast<int32_t *>(dst_col + kk + 4) = result_high;
                        }

                        // AVX2 4-way gather (vectorized with 128-bit)
                        for (; kk + 3 < K_BLK; kk += 4)
                        {
                            const int k_idx = k0 + kk;
                            if (k_idx + 3 < K)
                            {
                                alignas(16) int32_t indices[4];
                                for (int i = 0; i < 4; ++i)
                                {
                                    indices[i] = (k_idx + i) * N;
                                }
                                __m128i idx = _mm_load_si128(reinterpret_cast<const __m128i *>(indices));
                                __m128i gathered32 = _mm_i32gather_epi32(
                                    reinterpret_cast<const int *>(src_col), idx, 1);

                                // Extract low bytes: 0, 4, 8, 12
                                const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                                __m128i bytes = _mm_shuffle_epi8(gathered32, shuffle);
                                int32_t result = _mm_extract_epi32(bytes, 0);
                                *reinterpret_cast<int32_t *>(dst_col + kk) = result;
                            }
                            else
                            {
                                break;
                            }
                        }

                        // Scalar tail
                        for (; kk < K_BLK; ++kk)
                        {
                            const int k_idx = k0 + kk;
                            dst_col[kk] = (k_idx < K) ? src_col[k_idx * N] : 0;
                        }
                    }
                    else
                    {
                        std::memset(dst_col, 0, K_BLK);
                    }
                }
            }
        }

        // Tail columns (1-3 remaining)
        for (; n < nr; ++n)
        {
            const int col_idx = n0 + n;
            int8_t *dst_col = B_packed_panel + n * ld_col_B_out;

            if (col_idx < N)
            {
                const int8_t *src_col = B + col_idx;

                // Apply same hierarchical tail handling as fast path
                int kk = 0;

                // AVX-512 16-way gather (if K_BLK >= 64)
                if constexpr (K_BLK >= 64)
                {
                    for (; kk + 15 < K_BLK; kk += 16)
                    {
                        if (k0 + kk + 15 < K)
                        {
                            alignas(64) int32_t indices[16];
                            for (int i = 0; i < 16; ++i)
                            {
                                indices[i] = (k0 + kk + i) * N;
                            }
                            __m512i idx = _mm512_load_si512(reinterpret_cast<const __m512i *>(indices));
                            __m128i gathered = _mm512_cvtepi32_epi8(
                                _mm512_i32gather_epi32(idx, src_col, 1));
                            _mm_store_si128(reinterpret_cast<__m128i *>(dst_col + kk), gathered);
                        }
                        else
                        {
                            break; // Boundary case, fall through to smaller chunks
                        }
                    }
                }

                // AVX2 8-way gather tail
                for (; kk + 7 < K_BLK && k0 + kk + 7 < K; kk += 8)
                {
                    alignas(32) int32_t indices[8];
                    for (int i = 0; i < 8; ++i)
                    {
                        indices[i] = (k0 + kk + i) * N;
                    }
                    __m256i idx = _mm256_load_si256(reinterpret_cast<const __m256i *>(indices));
                    __m256i gathered32 = _mm256_i32gather_epi32(
                        reinterpret_cast<const int *>(src_col), idx, 1);

                    __m128i low128 = _mm256_castsi256_si128(gathered32);
                    __m128i high128 = _mm256_extracti128_si256(gathered32, 1);

                    const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                    __m128i low_bytes = _mm_shuffle_epi8(low128, shuffle);
                    __m128i high_bytes = _mm_shuffle_epi8(high128, shuffle);

                    int32_t result_low = _mm_extract_epi32(low_bytes, 0);
                    int32_t result_high = _mm_extract_epi32(high_bytes, 0);

                    *reinterpret_cast<int32_t *>(dst_col + kk) = result_low;
                    *reinterpret_cast<int32_t *>(dst_col + kk + 4) = result_high;
                }

                // AVX2 4-way gather (vectorized with 128-bit)
                for (; kk + 3 < K_BLK; kk += 4)
                {
                    const int k_idx = k0 + kk;
                    if (k_idx + 3 < K)
                    {
                        alignas(16) int32_t indices[4];
                        for (int i = 0; i < 4; ++i)
                        {
                            indices[i] = (k_idx + i) * N;
                        }
                        __m128i idx = _mm_load_si128(reinterpret_cast<const __m128i *>(indices));
                        __m128i gathered32 = _mm_i32gather_epi32(
                            reinterpret_cast<const int *>(src_col), idx, 1);

                        // Extract low bytes: 0, 4, 8, 12
                        const __m128i shuffle = _mm_setr_epi8(0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
                        __m128i bytes = _mm_shuffle_epi8(gathered32, shuffle);
                        int32_t result = _mm_extract_epi32(bytes, 0);
                        *reinterpret_cast<int32_t *>(dst_col + kk) = result;
                    }
                    else
                    {
                        break; // Fall through to final scalar tail
                    }
                }

                // Scalar tail (1-3 elements)
                for (; kk < K_BLK; ++kk)
                {
                    const int k_idx = k0 + kk;
                    dst_col[kk] = (k_idx < K) ? src_col[k_idx * N] : 0;
                }
            }
            else
            {
                std::memset(dst_col, 0, K_BLK);
            }
        }
    }

    // ---------- MICROKERNEL (NEW A PACKING) ----------

    template <
        int M_R,
        int N_R,
        int K_BLK,
        int UNROLL_K,
        int PREFETCH_B_L1,
        int PREFETCH_B_L2,
        bool ACCUM_INT32,
        bool USE_L2_PREFETCH,
        bool USE_VNNI>
    inline void microkernel_int8_vnni_tile(
        const int8_t *__restrict A_tile_packed,
        const PackedB &Bp,
        float *__restrict C_tile,
        const float *__restrict bias_tile,
        const float *__restrict act_scales,
        const float *__restrict wgt_scales,
        int N0,
        int T)
    {
        static_assert(N_R % 16 == 0, "N_R must be multiple of 16");
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int K_chunks = K_BLK / 4;
        const int num_groups = M_R / 4;
        const int group_stride = K_chunks * 16;
        const int A_block_bytes = num_groups * group_stride;

        // 1. Initialize C tile with bias
        for (int m = 0; m < M_R; ++m)
        {
            float *row = C_tile + m * N_R;
            std::memcpy(row, bias_tile, sizeof(float) * N_R);
        }

        // 2. Precompute sB vectors (vectorized)
        __m512 sB_vecs[N_R / 16];
        for (int j = 0; j < N_R / 16; ++j)
        {
            // Direct vectorized load instead of scalar loop
            sB_vecs[j] = _mm512_loadu_ps(wgt_scales + j * 16);
        }

        // 3. Loop over K blocks
        for (int t = 0; t < T; ++t)
        {
            // Accumulators
            __m512i acc[M_R][N_R / 16];
            for (int m = 0; m < M_R; ++m)
                for (int j = 0; j < N_R / 16; ++j)
                    acc[m][j] = _mm512_setzero_si512();

            const int8_t *base_A_block = A_tile_packed + t * A_block_bytes;
            const int8_t *base_B_block = Bp.block_ptr(t);

            // Prefetch next B block if desired
            if (t + 1 < T)
            {
                const int8_t *b_next = Bp.block_ptr(t + 1) + N0 * Bp.ld_col;
                if constexpr (PREFETCH_B_L1 > 0)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(b_next + PREFETCH_B_L1), _MM_HINT_T0);
                }
                if constexpr (USE_L2_PREFETCH && PREFETCH_B_L2 > 0)
                {
                    _mm_prefetch(reinterpret_cast<const char *>(b_next + PREFETCH_B_L2), _MM_HINT_T1);
                }
            }

            // Inner K loop with unroll and interleaving
            for (int kk = 0; kk < K_BLK; kk += 4 * UNROLL_K)
            {
                __m128i a_groups[UNROLL_K][M_R / 4];
                __m512i b_vecs_u[UNROLL_K][N_R / 16];

                // Stage u=0 with interleaved A/B loads for dual load port exploitation
                int k_off0 = kk;
                if (k_off0 < K_BLK)
                {
                    const int kk_idx0 = k_off0 / 4;

                    // Interleave A group loads and B loads (2-way: load 1 A group, 1 B vec, repeat)
                    const int num_pairs = std::min(num_groups, N_R / 16);

                    // Load pairs of A groups and B vecs to exploit dual load ports
                    int g = 0, j = 0;
                    for (; g + 1 < num_groups && j + 1 < N_R / 16; g += 2, j += 2)
                    {
                        // Load A group 0 and B vec 0 in parallel
                        const int8_t *src_a0 = base_A_block + g * group_stride + kk_idx0 * 16;
                        const int8_t *b_ptr0 = base_B_block + (N0 + j * 16) * Bp.ld_col + k_off0;
                        a_groups[0][g] = _mm_load_si128(reinterpret_cast<const __m128i *>(src_a0));
                        b_vecs_u[0][j] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr0));

                        // Load A group 1 and B vec 1 in parallel
                        const int8_t *src_a1 = base_A_block + (g + 1) * group_stride + kk_idx0 * 16;
                        const int8_t *b_ptr1 = base_B_block + (N0 + (j + 1) * 16) * Bp.ld_col + k_off0;
                        a_groups[0][g + 1] = _mm_load_si128(reinterpret_cast<const __m128i *>(src_a1));
                        b_vecs_u[0][j + 1] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr1));
                    }

                    // Handle remaining A groups
                    for (; g < num_groups; ++g)
                    {
                        const int8_t *src = base_A_block + g * group_stride + kk_idx0 * 16;
                        a_groups[0][g] = _mm_load_si128(reinterpret_cast<const __m128i *>(src));
                    }

                    // Handle remaining B vecs
                    for (; j < N_R / 16; ++j)
                    {
                        const int8_t *b_ptr = base_B_block + (N0 + j * 16) * Bp.ld_col + k_off0;
                        b_vecs_u[0][j] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr));
                    }
                }

// Unrolled pipeline for u=1..UNROLL_K-1
#pragma unroll
                for (int u = 1; u < UNROLL_K; ++u)
                {
                    const int k_off = kk + 4 * u;
                    if (k_off >= K_BLK)
                        break;
                    const int kk_idx = k_off / 4;

                    // Stage loads for current u with interleaved A/B to exploit dual load ports
                    int g = 0, j = 0;
                    for (; g + 1 < num_groups && j + 1 < N_R / 16; g += 2, j += 2)
                    {
                        // Load A group 0 and B vec 0 in parallel
                        const int8_t *src_a0 = base_A_block + g * group_stride + kk_idx * 16;
                        const int8_t *b_ptr0 = base_B_block + (N0 + j * 16) * Bp.ld_col + k_off;
                        a_groups[u][g] = _mm_load_si128(reinterpret_cast<const __m128i *>(src_a0));
                        b_vecs_u[u][j] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr0));

                        // Load A group 1 and B vec 1 in parallel
                        const int8_t *src_a1 = base_A_block + (g + 1) * group_stride + kk_idx * 16;
                        const int8_t *b_ptr1 = base_B_block + (N0 + (j + 1) * 16) * Bp.ld_col + k_off;
                        a_groups[u][g + 1] = _mm_load_si128(reinterpret_cast<const __m128i *>(src_a1));
                        b_vecs_u[u][j + 1] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr1));
                    }

                    // Handle remaining A groups
                    for (; g < num_groups; ++g)
                    {
                        const int8_t *src = base_A_block + g * group_stride + kk_idx * 16;
                        a_groups[u][g] = _mm_load_si128(reinterpret_cast<const __m128i *>(src));
                    }

                    // Handle remaining B vecs
                    for (; j < N_R / 16; ++j)
                    {
                        const int8_t *b_ptr = base_B_block + (N0 + j * 16) * Bp.ld_col + k_off;
                        b_vecs_u[u][j] = _mm512_loadu_si512(reinterpret_cast<const void *>(b_ptr));
                    }

                    // Compute with previous u-1
                    const int prev_u = u - 1;
                    if constexpr (USE_VNNI)
                    {
                        for (int g = 0; g < num_groups; ++g)
                        {
                            const __m128i a32 = a_groups[prev_u][g];
                            for (int r = 0; r < 4; ++r)
                            {
                                const int m = g * 4 + r;
                                const __m128i a_row_32 =
                                    _mm_shuffle_epi32(a32, _MM_SHUFFLE(r, r, r, r));
                                const __m512i a_vec = _mm512_broadcast_i32x4(a_row_32);
                                for (int j = 0; j < N_R / 16; ++j)
                                {
                                    acc[m][j] = _mm512_dpbusd_epi32(acc[m][j], a_vec, b_vecs_u[prev_u][j]);
                                }
                            }
                        }
                    }
                }

                // Final compute for last staged u
                int max_u = std::min(UNROLL_K - 1, (K_BLK - kk) / 4 - 1);
                if (max_u >= 0)
                {
                    if constexpr (USE_VNNI)
                    {
                        for (int g = 0; g < num_groups; ++g)
                        {
                            const __m128i a32 = a_groups[max_u][g];
                            for (int r = 0; r < 4; ++r)
                            {
                                const int m = g * 4 + r;
                                const __m128i a_row_32 =
                                    _mm_shuffle_epi32(a32, _MM_SHUFFLE(r, r, r, r));
                                const __m512i a_vec = _mm512_broadcast_i32x4(a_row_32);
                                for (int j = 0; j < N_R / 16; ++j)
                                {
                                    acc[m][j] = _mm512_dpbusd_epi32(acc[m][j], a_vec, b_vecs_u[max_u][j]);
                                }
                            }
                        }
                    }
                }
            } // kk

            // 3.3 Scale and accumulate into C_tile
            const float sA_t = act_scales[t];
            const __m512 sA_vec = _mm512_set1_ps(sA_t);

            for (int j = 0; j < N_R / 16; ++j)
            {
                const __m512 sAB_vec = _mm512_mul_ps(sA_vec, sB_vecs[j]);

                for (int m = 0; m < M_R; ++m)
                {
                    const __m512i acc_i32 = acc[m][j];
                    const __m512 acc_f32 = _mm512_cvtepi32_ps(acc_i32);
                    const __m512 contrib = _mm512_mul_ps(acc_f32, sAB_vec);

                    float *c_row = C_tile + m * N_R;
                    const __m512 c_old = _mm512_loadu_ps(c_row + j * 16);
                    const __m512 c_new = _mm512_add_ps(c_old, contrib);
                    _mm512_storeu_ps(c_row + j * 16, c_new);
                }
            }
        } // t
    }

    // ---------- OUTER GEMM KERNEL ----------

    template <
        int M_R,
        int N_R,
        int K_BLK,
        int UNROLL_K,
        int PREFETCH_B_L1,
        int PREFETCH_B_L2,
        bool ACCUM_INT32,
        bool USE_L2_PREFETCH,
        bool USE_VNNI>
    void gemm_int8_vnni_kernel(
        const int8_t *__restrict A,
        const PackedB &Bp,
        float *__restrict C,
        const float *__restrict bias,
        const float *__restrict act_scales,
        const float *__restrict wgt_scales,
        int M, int N, int K)
    {
        static_assert(M_R % 4 == 0, "M_R must be multiple of 4");
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int T = K / K_BLK;

        const int K_chunks = K_BLK / 4;
        const int num_groups = M_R / 4;
        const int group_stride = K_chunks * 16;
        const int A_block_bytes = num_groups * group_stride;
        const int A_tile_total_bytes = A_block_bytes * T;

// OpenMP parallel over M dimension with dynamic scheduling for load balancing
#pragma omp parallel
        {
            // Thread-private scratch buffers
            alignas(64) int8_t A_tile_packed[A_tile_total_bytes];
            alignas(64) float C_tile[M_R * N_R];
            alignas(64) float bias_tile[N_R];
            alignas(64) float wgt_scales_tile[N_R];

#pragma omp for schedule(dynamic, 1)
            for (int M0 = 0; M0 < M; M0 += M_R)
            {
                const int mr = std::min(M_R, M - M0);

                // Pack A for all K blocks for this M_R tile
                for (int t = 0; t < T; ++t)
                {
                    const int k0 = t * K_BLK;
                    int8_t *A_block_tile = A_tile_packed + t * A_block_bytes;

                    pack_A_tile_4x4_grouped<M_R, K_BLK>(
                        A, M, K,
                        M0, k0,
                        mr, K_BLK,
                        A_block_tile);
                }

                for (int N0 = 0; N0 < N; N0 += N_R)
                {
                    const int nr = std::min(N_R, N - N0);

                    // Prepare bias and wgt_scales for this N tile (vectorized)
                    int n_idx = 0;

                    // AVX-512 16-way vector copy
                    for (; n_idx + 15 < nr; n_idx += 16)
                    {
                        __m512 b = _mm512_loadu_ps(bias + N0 + n_idx);
                        __m512 w = _mm512_loadu_ps(wgt_scales + N0 + n_idx);
                        _mm512_store_ps(bias_tile + n_idx, b);
                        _mm512_store_ps(wgt_scales_tile + n_idx, w);
                    }

                    // AVX2 8-way vector copy
                    for (; n_idx + 7 < nr; n_idx += 8)
                    {
                        __m256 b = _mm256_loadu_ps(bias + N0 + n_idx);
                        __m256 w = _mm256_loadu_ps(wgt_scales + N0 + n_idx);
                        _mm256_store_ps(bias_tile + n_idx, b);
                        _mm256_store_ps(wgt_scales_tile + n_idx, w);
                    }

                    // AVX2 4-way vector copy (128-bit)
                    for (; n_idx + 3 < nr; n_idx += 4)
                    {
                        __m128 b = _mm_loadu_ps(bias + N0 + n_idx);
                        __m128 w = _mm_loadu_ps(wgt_scales + N0 + n_idx);
                        _mm_store_ps(bias_tile + n_idx, b);
                        _mm_store_ps(wgt_scales_tile + n_idx, w);
                    }

                    // Scalar tail (1-3 elements)
                    for (; n_idx < nr; ++n_idx)
                    {
                        bias_tile[n_idx] = bias[N0 + n_idx];
                        wgt_scales_tile[n_idx] = wgt_scales[N0 + n_idx];
                    }

                    // Zero-pad remaining with vectorization
                    for (; n_idx + 15 < N_R; n_idx += 16)
                    {
                        _mm512_store_ps(bias_tile + n_idx, _mm512_setzero_ps());
                        _mm512_store_ps(wgt_scales_tile + n_idx, _mm512_setzero_ps());
                    }
                    for (; n_idx < N_R; ++n_idx)
                    {
                        bias_tile[n_idx] = 0.0f;
                        wgt_scales_tile[n_idx] = 0.0f;
                    }

                    // Call microkernel
                    microkernel_int8_vnni_tile<
                        M_R, N_R, K_BLK, UNROLL_K,
                        PREFETCH_B_L1, PREFETCH_B_L2,
                        ACCUM_INT32, USE_L2_PREFETCH, USE_VNNI>(
                        A_tile_packed,
                        Bp,
                        C_tile,
                        bias_tile,
                        act_scales,
                        wgt_scales_tile,
                        N0,
                        T);

                    // Store back valid part of C_tile
                    for (int m = 0; m < mr; ++m)
                    {
                        float *dst = C + (M0 + m) * N + N0;
                        const float *src = C_tile + m * N_R;
                        std::memcpy(dst, src, sizeof(float) * nr);
                    }
                }
            } // M0 loop
        } // omp parallel
    }

    // ---------- EXPLICIT TEMPLATE INSTANTIATIONS ----------
    // Instantiate configurations used by benchmarks and tests

    // Benchmark configuration: M_R=16, N_R=64, K_BLK=64, UNROLL_K=2
    template void pack_A_tile_4x4_grouped<16, 64>(
        const int8_t *, int, int, int, int, int, int, int8_t *);

    template void pack_B_panel_vnni<64>(
        const int8_t *, int, int, int, int, int, int8_t *, int &, int &);

    // Template instantiations are now generated by generate_vnni_gemm_instantiations.py
    // See kernels/cpu/gemm_v3/generated/VNNIGemmInstantiations_*.cpp

    // Additional common configurations can be added here as needed

} // namespace llaminar2
