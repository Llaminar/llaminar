/**
 * @file VNNIGemm_Complete.h
 * @brief VNNI-optimized INT8 GEMM kernel with pre-packed panel layout
 * @author David Sanftenberg
 *
 * Design Philosophy (gemm_v3):
 * - Pre-pack both A and B into VNNI-friendly layouts ONCE
 * - Maximize time spent in _mm512_dpbusd_epi32 inner loop
 * - Minimize on-the-fly format conversion overhead
 * - 4x4-grouped A packing for efficient broadcast + dpbusd
 * - Column-major K-contiguous B packing for sequential VNNI loads
 *
 * This is a header-only template implementation following the gemm_v2 pattern.
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

    // ============================================================================
    // DATA STRUCTURES
    // ============================================================================

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

    // ============================================================================
    // FUNCTION DECLARATIONS
    // ============================================================================

    template <int M_R, int K_BLK>
    void pack_A_tile_4x4_grouped(
        const int8_t *__restrict A,
        int M, int K,
        int M0, int k0,
        int mr, int kblk,
        int8_t *__restrict A_tile_packed);

    template <int K_BLK>
    void pack_B_panel_vnni(
        const int8_t *__restrict B,
        int K, int N,
        int k0,
        int n0, int nr,
        int8_t *__restrict B_packed_panel,
        int &ld_block_B_out,
        int &ld_col_B_out);

    template <
        int M_R, int N_R, int K_BLK, int UNROLL_K,
        int PREFETCH_B_L1, int PREFETCH_B_L2,
        bool ACCUM_INT32, bool USE_L2_PREFETCH, bool USE_VNNI>
    inline void microkernel_int8_vnni_tile(
        const int8_t *__restrict A_tile_packed,
        const PackedB &Bp,
        float *__restrict C_tile,
        const float *__restrict bias_tile,
        const float *__restrict act_scales,
        const float *__restrict wgt_scales,
        int N0, int T);

    template <
        int M_R, int N_R, int K_BLK, int UNROLL_K,
        int PREFETCH_B_L1, int PREFETCH_B_L2,
        bool ACCUM_INT32, bool USE_L2_PREFETCH, bool USE_VNNI>
    void gemm_int8_vnni_kernel(
        const int8_t *__restrict A,
        const PackedB &Bp,
        float *__restrict C,
        const float *__restrict bias,
        const float *__restrict act_scales,
        const float *__restrict wgt_scales,
        int M, int N, int K);

} // namespace llaminar2
