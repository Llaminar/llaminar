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
#include "utils/FP16.h"
#include <cstdint>
#include <vector>

namespace llaminar2
{

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

        const int T = (K + K_BLK - 1) / K_BLK; // Number of K blocks
        const int ld_col = K_BLK;
        const int ld_block = N * ld_col;

        // Allocate storage
        B_packed_storage.resize(T * ld_block, 0);
        wgt_scales.resize(N, 1.0f);

        // Get Q8_0 block structure
        const size_t block_size = Q8_0Block::BLOCK_SIZE; // Typically 32
        const size_t blocks_per_col = (K + block_size - 1) / block_size;

        // Extract weight scales (average across K dimension for each column)
        for (int n = 0; n < N; ++n)
        {
            float scale_sum = 0.0f;
            size_t count = 0;

            for (size_t k_block_idx = 0; k_block_idx < blocks_per_col; ++k_block_idx)
            {
                // Q8_0 is stored column-major in blocks
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

        // Pack weights: Extract int8 data from Q8_0 blocks and repack to column-major
        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            int8_t *block_base = B_packed_storage.data() + t * ld_block;

            for (int n = 0; n < N; ++n)
            {
                int8_t *dst_col = block_base + n * ld_col;

                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    const int k_global = k0 + kk;

                    if (k_global < K)
                    {
                        // Determine which Q8_0 block this element belongs to
                        const size_t q8_block_idx = k_global / block_size;
                        const size_t offset_in_block = k_global % block_size;

                        // Get the Q8_0 block for column n, K-block q8_block_idx
                        const void *raw_block = B.get_raw_block_at(n, q8_block_idx);
                        if (raw_block)
                        {
                            const Q8_0Block *block = static_cast<const Q8_0Block *>(raw_block);
                            dst_col[kk] = block->qs[offset_in_block];
                        }
                        else
                        {
                            dst_col[kk] = 0;
                        }
                    }
                    else
                    {
                        // Zero-pad if K is not multiple of K_BLK
                        dst_col[kk] = 0;
                    }
                }
            }
        }

        // Fill PackedB view
        Bp.data = B_packed_storage.data();
        Bp.ld_block = ld_block;
        Bp.ld_col = ld_col;
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
        const int ld_col = K_BLK;
        const int ld_block = N * ld_col;

        B_packed_storage.resize(T * ld_block, 0);

        // Pack from row-major [K x N] to column-major K-contiguous blocks
        for (int t = 0; t < T; ++t)
        {
            const int k0 = t * K_BLK;
            int8_t *block_base = B_packed_storage.data() + t * ld_block;

            for (int n = 0; n < N; ++n)
            {
                int8_t *dst_col = block_base + n * ld_col;

                for (int kk = 0; kk < K_BLK; ++kk)
                {
                    const int k_global = k0 + kk;

                    if (k_global < K)
                    {
                        // B_int8 is row-major: B_int8[k * N + n]
                        dst_col[kk] = B_int8[k_global * N + n];
                    }
                    else
                    {
                        dst_col[kk] = 0;
                    }
                }
            }
        }

        Bp.data = B_packed_storage.data();
        Bp.ld_block = ld_block;
        Bp.ld_col = ld_col;
        Bp.N = N;
        Bp.K_BLK = K_BLK;
    }

} // namespace llaminar2
