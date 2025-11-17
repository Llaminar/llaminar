/**
 * @file WeightPackingAdapters.h
 * @brief Adapters to convert Q8_0Tensor to packed VNNI column-major format
 * @author David Sanftenberg
 *
 * Converts Q8_0 quantized weights to the packed B format expected by VNNI GEMM:
 * - Column-major K-contiguous layout
 * - Divided into K_BLK-sized blocks along K dimension
 * - Each column stored contiguously within a block
 *
 * Layout:
 *   For each K block t (k_start = t * K_BLK):
 *     For each column n in [0..N):
 *       For each k in [0..K_BLK):
 *         B_packed[t*ld_block + n*ld_col + k] = B[k_global][n]
 *   where:
 *     ld_col = K_BLK
 *     ld_block = N * ld_col
 */

#pragma once

#include "VNNIGemm.h"
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include "../../../utils/CPUFeatures.h"
#include <algorithm>
#include <cstdint>
#include <vector>
#include <cstring>
#include <immintrin.h>

namespace llaminar2
{
    template <int K_BLK>
    void pack_int8_weights_to_vnni_format(
        const int8_t *__restrict B_int8,
        int K,
        int N,
        std::vector<int8_t> &B_packed_storage,
        PackedB &Bp);

    /**
     * @brief Pack Q8_0 weights to VNNI column-major format
     *
     * Extracts int8 quantized data from Q8_0 blocks and packs to column-major
     * K-contiguous layout for VNNI GEMM kernel.
     *
     * @tparam K_BLK K block size (must be multiple of 4)
     * @param B Q8_0 weight tensor
     * @param K Number of rows in weight matrix
     * @param N Number of columns in weight matrix
     * @param B_packed_storage Output vector to own packed data
     * @param Bp Output PackedB view structure
     * @param wgt_scales Output per-column weight scales [N] (extracted from Q8_0 blocks)
     */
    template <int K_BLK>
    void pack_q8_0_weights_to_vnni_format(
        const Q8_0Tensor &B,
        int K,
        int N,
        std::vector<int8_t> &B_packed_storage,
        PackedB &Bp,
        std::vector<float> &wgt_scales)
    {
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int T = (K + K_BLK - 1) / K_BLK;
        const int ld_col = 4;
        const int chunk_count = K_BLK / 4;
        const int ld_chunk = N * ld_col;
        const int ld_block = chunk_count * ld_chunk;

        B_packed_storage.resize(T * ld_block, 0);
        wgt_scales.resize(N, 1.0f);

        const size_t block_size = Q8_0Block::BLOCK_SIZE;
        const size_t blocks_per_col = (K + block_size - 1) / block_size;

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n)
        {
            float scale_sum = 0.0f;
            size_t count = 0;

            for (size_t k_block_idx = 0; k_block_idx < blocks_per_col; ++k_block_idx)
            {
                const void *raw_block = B.get_raw_block_at(n, k_block_idx);
                if (raw_block)
                {
                    const Q8_0Block *block = static_cast<const Q8_0Block *>(raw_block);
                    scale_sum += fp16_to_fp32(block->d);
                    count++;
                }
            }

            wgt_scales[n] = count > 0 ? scale_sum / count : 1.0f;
        }

        std::vector<int8_t> int8_row_major(static_cast<size_t>(K) * N, 0);

#pragma omp parallel for schedule(static)
        for (int n = 0; n < N; ++n)
        {
            for (size_t block_idx = 0; block_idx < blocks_per_col; ++block_idx)
            {
                const Q8_0Block *block = static_cast<const Q8_0Block *>(B.get_raw_block_at(n, block_idx));
                if (!block)
                {
                    continue;
                }

                _mm_prefetch(reinterpret_cast<const char *>(block), _MM_HINT_T0);
                const int k_base = static_cast<int>(block_idx * block_size);
                const int limit = std::min<int>(block_size, K - k_base);

                for (int offset = 0; offset < limit; ++offset)
                {
                    const int global_k = k_base + offset;
                    int8_row_major[static_cast<size_t>(global_k) * N + n] = block->qs[offset];
                }
            }
        }

        pack_int8_weights_to_vnni_format<K_BLK>(
            int8_row_major.data(),
            K,
            N,
            B_packed_storage,
            Bp);

        Bp.ld_col = ld_col;
        Bp.ld_chunk = ld_chunk;
        Bp.ld_block = ld_block;
        Bp.N = N;
        Bp.K_BLK = K_BLK;
    }

    /**
     * @brief Pack raw int8 weights to VNNI column-major format
     *
     * For pre-dequantized or raw int8 weights (e.g., from testing),
     * pack directly to VNNI format without Q8_0 block extraction.
     *
     * @tparam K_BLK K block size (must be multiple of 4)
     * @param B_int8 Raw int8 weights [K x N], row-major
     * @param K Number of rows
     * @param N Number of columns
     * @param B_packed_storage Output vector to own packed data
     * @param Bp Output PackedB view structure
     */
    namespace detail
    {

        inline void store_columns_scalar(
            const int8_t *__restrict B_int8,
            int8_t *__restrict chunk_base,
            int n,
            int N,
            int k_global)
        {
            for (int lane = 0; lane < 4; ++lane)
            {
                chunk_base[n * 4 + lane] = B_int8[(k_global + lane) * N + n];
            }
        }

#if defined(__AVX512F__)
        inline void store_columns_avx512(
            const int8_t *__restrict B_int8,
            int8_t *__restrict chunk_base,
            int n_start,
            int N,
            int k_global)
        {
            const int offset0 = (k_global + 0) * N + n_start;
            const int offset1 = (k_global + 1) * N + n_start;
            const int offset2 = (k_global + 2) * N + n_start;
            const int offset3 = (k_global + 3) * N + n_start;

            __m128i lane0_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_int8 + offset0));
            __m128i lane1_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_int8 + offset1));
            __m128i lane2_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_int8 + offset2));
            __m128i lane3_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(B_int8 + offset3));

            __m512i lane0 = _mm512_cvtepu8_epi32(lane0_bytes);
            __m512i lane1 = _mm512_cvtepu8_epi32(lane1_bytes);
            __m512i lane2 = _mm512_cvtepu8_epi32(lane2_bytes);
            __m512i lane3 = _mm512_cvtepu8_epi32(lane3_bytes);

            __m512i packed = _mm512_or_si512(
                lane0,
                _mm512_or_si512(
                    _mm512_slli_epi32(lane1, 8),
                    _mm512_or_si512(
                        _mm512_slli_epi32(lane2, 16),
                        _mm512_slli_epi32(lane3, 24))));

            _mm512_storeu_si512(reinterpret_cast<__m512i *>(chunk_base + n_start * 4), packed);
        }
#endif

#if defined(__AVX2__)
        inline void store_columns_avx2_8(
            const int8_t *__restrict B_int8,
            int8_t *__restrict chunk_base,
            int n_start,
            int N,
            int k_global)
        {
            const int offset0 = (k_global + 0) * N + n_start;
            const int offset1 = (k_global + 1) * N + n_start;
            const int offset2 = (k_global + 2) * N + n_start;
            const int offset3 = (k_global + 3) * N + n_start;

            __m128i lane0_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(B_int8 + offset0));
            __m128i lane1_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(B_int8 + offset1));
            __m128i lane2_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(B_int8 + offset2));
            __m128i lane3_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(B_int8 + offset3));

            __m256i lane0 = _mm256_cvtepu8_epi32(lane0_bytes);
            __m256i lane1 = _mm256_cvtepu8_epi32(lane1_bytes);
            __m256i lane2 = _mm256_cvtepu8_epi32(lane2_bytes);
            __m256i lane3 = _mm256_cvtepu8_epi32(lane3_bytes);

            __m256i packed = _mm256_or_si256(
                lane0,
                _mm256_or_si256(
                    _mm256_slli_epi32(lane1, 8),
                    _mm256_or_si256(
                        _mm256_slli_epi32(lane2, 16),
                        _mm256_slli_epi32(lane3, 24))));

            _mm256_storeu_si256(reinterpret_cast<__m256i *>(chunk_base + n_start * 4), packed);
        }

        inline void store_columns_avx2_4(
            const int8_t *__restrict B_int8,
            int8_t *__restrict chunk_base,
            int n_start,
            int N,
            int k_global)
        {
            const int offset0 = (k_global + 0) * N + n_start;
            const int offset1 = (k_global + 1) * N + n_start;
            const int offset2 = (k_global + 2) * N + n_start;
            const int offset3 = (k_global + 3) * N + n_start;

            int lane0_val = *reinterpret_cast<const int32_t *>(B_int8 + offset0);
            int lane1_val = *reinterpret_cast<const int32_t *>(B_int8 + offset1);
            int lane2_val = *reinterpret_cast<const int32_t *>(B_int8 + offset2);
            int lane3_val = *reinterpret_cast<const int32_t *>(B_int8 + offset3);

            __m128i lane0 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(lane0_val));
            __m128i lane1 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(lane1_val));
            __m128i lane2 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(lane2_val));
            __m128i lane3 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(lane3_val));

            __m128i packed = _mm_or_si128(
                lane0,
                _mm_or_si128(
                    _mm_slli_epi32(lane1, 8),
                    _mm_or_si128(
                        _mm_slli_epi32(lane2, 16),
                        _mm_slli_epi32(lane3, 24))));

            _mm_storeu_si128(reinterpret_cast<__m128i *>(chunk_base + n_start * 4), packed);
        }
#endif

    } // namespace detail

    template <int K_BLK>
    void pack_int8_weights_to_vnni_format(
        const int8_t *__restrict B_int8,
        int K,
        int N,
        std::vector<int8_t> &B_packed_storage,
        PackedB &Bp)
    {
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        const int T = (K + K_BLK - 1) / K_BLK;
        const int ld_col = 4;
        const int chunk_count = K_BLK / 4;
        const int ld_chunk = N * ld_col;
        const int ld_block = chunk_count * ld_chunk;

        B_packed_storage.resize(T * ld_block, 0);

        const bool has_avx512 = cpu_supports_avx512();
        const bool has_avx2 = cpu_supports_avx2();

#pragma omp parallel for schedule(static)
        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            int8_t *block_base = B_packed_storage.data() + t * ld_block;

            for (int kk = 0; kk < K_BLK; kk += 4)
            {
                int8_t *chunk_base = block_base + (kk / 4) * ld_chunk;
                const int k_global = k0 + kk;

                int n = 0;
                if (has_avx512)
                {
                    for (; n + 15 < N; n += 16)
                    {
                        detail::store_columns_avx512(B_int8, chunk_base, n, N, k_global);
                        _mm_prefetch(reinterpret_cast<const char *>(B_int8 + (k_global)*N + n + 64), _MM_HINT_T0);
                    }
                }

                if (has_avx2)
                {
                    for (; n + 7 < N; n += 8)
                    {
                        detail::store_columns_avx2_8(B_int8, chunk_base, n, N, k_global);
                    }

                    for (; n + 3 < N; n += 4)
                    {
                        detail::store_columns_avx2_4(B_int8, chunk_base, n, N, k_global);
                    }
                }

                for (; n < N; ++n)
                {
                    detail::store_columns_scalar(B_int8, chunk_base, n, N, k_global);
                }
            }
        }

        Bp.data = B_packed_storage.data();
        Bp.ld_block = ld_block;
        Bp.ld_chunk = ld_chunk;
        Bp.ld_col = ld_col;
        Bp.N = N;
        Bp.K_BLK = K_BLK;
    }

} // namespace llaminar2
