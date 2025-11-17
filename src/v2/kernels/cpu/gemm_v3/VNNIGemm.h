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
    namespace detail
    {
        [[gnu::cold]] inline __m512i load_b_vec_tail(const int8_t *src, int copy_cols, int ld_col, int8_t *buffer)
        {
            std::memset(buffer, 0, 64);
            if (copy_cols > 0 && src != nullptr)
            {
                std::memcpy(buffer, src, static_cast<size_t>(copy_cols) * ld_col);
            }
            return _mm512_loadu_si512(reinterpret_cast<const void *>(buffer));
        }
    }

    template <int Lane>
    inline __m512i broadcast_row_vec(const __m128i &src)
    {
        static_assert(Lane >= 0 && Lane < 4, "lane must be in [0,3]");
        return _mm512_broadcast_i32x4(
            _mm_shuffle_epi32(src, _MM_SHUFFLE(Lane, Lane, Lane, Lane)));
    }

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

    // Packed B layout: VNNI-interleaved 4-wide depth chunks
    struct PackedB
    {
        int8_t *data;
        int ld_block;
        int ld_chunk;
        int ld_col;
        int N;
        int K_BLK;

        inline const int8_t *block_ptr(int t) const
        {
            return data + t * ld_block;
        }

        inline const int8_t *chunk_ptr(int t, int chunk_idx) const
        {
            return block_ptr(t) + chunk_idx * ld_chunk;
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
        const __m128i xor_mask = _mm_set1_epi8(static_cast<char>(0x80));

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

                    __m128i vec0 = _mm_load_si128(reinterpret_cast<const __m128i *>(temp0));
                    __m128i vec1 = _mm_load_si128(reinterpret_cast<const __m128i *>(temp1));
                    vec0 = _mm_xor_si128(vec0, xor_mask);
                    vec1 = _mm_xor_si128(vec1, xor_mask);
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst0), vec0);
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst1), vec1);
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
                    __m128i vec = _mm_load_si128(reinterpret_cast<const __m128i *>(temp));
                    vec = _mm_xor_si128(vec, xor_mask);
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst), vec);
                }
            }
            else
            {
                // Slow path: Handle partial rows (boundary case)
                for (int kk = 0; kk < K_chunks; ++kk)
                {
                    int8_t *dst = group_ptr + kk * 16;
                    alignas(16) int8_t temp[16];

                    for (int lane = 0; lane < 4; ++lane)
                    {
                        const int m = M0 + m_base + lane;
                        const int row_in_tile = m_base + lane;

                        if (row_in_tile < mr && m < M)
                        {
                            const int8_t *src = A + m * K + (k0 + kk * 4);
                            *reinterpret_cast<int32_t *>(&temp[lane * 4]) = *reinterpret_cast<const int32_t *>(src);
                        }
                        else
                        {
                            // Zero-pad partial rows
                            *reinterpret_cast<int32_t *>(&temp[lane * 4]) = 0;
                        }
                    }

                    __m128i vec = _mm_load_si128(reinterpret_cast<const __m128i *>(temp));
                    vec = _mm_xor_si128(vec, xor_mask);
                    _mm_store_si128(reinterpret_cast<__m128i *>(dst), vec);
                }
            }
        }

        // Zero-pad remaining groups if mr < M_R (vectorized)
        const int padded_mr = ((mr + 3) / 4) * 4;
        for (int m_base = padded_mr; m_base < M_R; m_base += 4)
        {
            const int group_idx = m_base / 4;
            int8_t *group_ptr = A_tile_packed + group_idx * group_stride;

            // Use vector stores for biased zero initialization
            for (int i = 0; i < group_stride; i += 16)
            {
                _mm_store_si128(reinterpret_cast<__m128i *>(group_ptr + i), xor_mask);
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
        int &ld_chunk_B_out,
        int &ld_col_B_out)
    {
        static_assert(K_BLK % 4 == 0, "K_BLK must be multiple of 4");

        ld_col_B_out = 4;                              // 4 bytes per column chunk
        ld_chunk_B_out = nr * ld_col_B_out;            // bytes between successive column chunks
        const int chunk_count = K_BLK / 4;             // number of 4-wide depth chunks
        ld_block_B_out = chunk_count * ld_chunk_B_out; // bytes per K block

        constexpr int PREFETCH_DISTANCE = 64;

        const auto make_mask = [](int cols) -> __mmask16
        {
            if (cols >= 16)
            {
                return 0xFFFF;
            }
            if (cols <= 0)
            {
                return 0;
            }
            return static_cast<__mmask16>((1u << cols) - 1u);
        };

        const auto transpose_store = [](const __m128i &row0,
                                        const __m128i &row1,
                                        const __m128i &row2,
                                        const __m128i &row3,
                                        int block_cols,
                                        int8_t *dst_base)
        {
            const __m128i t0 = _mm_unpacklo_epi8(row0, row1);
            const __m128i t1 = _mm_unpackhi_epi8(row0, row1);
            const __m128i t2 = _mm_unpacklo_epi8(row2, row3);
            const __m128i t3 = _mm_unpackhi_epi8(row2, row3);

            const __m128i u0 = _mm_unpacklo_epi16(t0, t2); // cols 0-3
            const __m128i u1 = _mm_unpackhi_epi16(t0, t2); // cols 4-7
            const __m128i u2 = _mm_unpacklo_epi16(t1, t3); // cols 8-11
            const __m128i u3 = _mm_unpackhi_epi16(t1, t3); // cols 12-15

            if (block_cols >= 4)
            {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_base), u0);
            }
            if (block_cols >= 8)
            {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_base + 16), u1);
            }
            if (block_cols >= 12)
            {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_base + 32), u2);
            }
            if (block_cols >= 16)
            {
                _mm_storeu_si128(reinterpret_cast<__m128i *>(dst_base + 48), u3);
            }
        };

        for (int kk = 0; kk < K_BLK; kk += 4)
        {
            const int chunk_idx = kk / 4;
            int8_t *chunk_base = B_packed_panel + chunk_idx * ld_chunk_B_out;

            const int8_t *lane_ptrs[4];
            for (int lane = 0; lane < 4; ++lane)
            {
                const int k_idx = k0 + kk + lane;
                if (k_idx < K)
                {
                    lane_ptrs[lane] = B + static_cast<size_t>(k_idx) * N + n0;
                }
                else
                {
                    lane_ptrs[lane] = nullptr;
                }
            }

            const auto prefetch_lanes = [&](int col_offset)
            {
                const int rel_col = col_offset + PREFETCH_DISTANCE;
                const int absolute_col = n0 + rel_col;
                if (rel_col < 0 || rel_col >= nr || absolute_col >= N)
                {
                    return;
                }
                for (int lane = 0; lane < 4; ++lane)
                {
                    if (lane_ptrs[lane])
                    {
                        _mm_prefetch(reinterpret_cast<const char *>(lane_ptrs[lane] + rel_col), _MM_HINT_T0);
                    }
                }
            };

            const auto load_lane_vec = [&](int lane, int col_offset, __mmask16 mask) -> __m128i
            {
                if (!lane_ptrs[lane] || mask == 0)
                {
                    return _mm_setzero_si128();
                }
                return _mm_maskz_loadu_epi8(mask, lane_ptrs[lane] + col_offset);
            };

            const auto load_rows = [&](int col_offset, int block_cols,
                                       __m128i &row0, __m128i &row1,
                                       __m128i &row2, __m128i &row3)
            {
                const int absolute_col = n0 + col_offset;
                const int valid_cols = std::min(block_cols, std::max(0, N - absolute_col));
                const __mmask16 col_mask = make_mask(valid_cols);

                // Dual-load pairs to keep both load ports busy.
                row0 = load_lane_vec(0, col_offset, col_mask);
                row1 = load_lane_vec(1, col_offset, col_mask);
                row2 = load_lane_vec(2, col_offset, col_mask);
                row3 = load_lane_vec(3, col_offset, col_mask);
            };

            const auto store_block = [&](int col_offset, int block_cols)
            {
                prefetch_lanes(col_offset);
                __m128i row0, row1, row2, row3;
                load_rows(col_offset, block_cols, row0, row1, row2, row3);
                int8_t *dst = chunk_base + col_offset * ld_col_B_out;
                transpose_store(row0, row1, row2, row3, block_cols, dst);
            };

            int n = 0;
            while (n + 15 < nr)
            {
                store_block(n, 16);
                n += 16;
            }

            if (n + 7 < nr)
            {
                store_block(n, 8);
                n += 8;
            }

            if (n + 3 < nr)
            {
                store_block(n, 4);
                n += 4;
            }

            for (; n < nr; ++n)
            {
                int8_t *dst = chunk_base + n * ld_col_B_out;
                const int absolute_col = n0 + n;

                for (int lane = 0; lane < 4; ++lane)
                {
                    if (lane_ptrs[lane] && absolute_col < N)
                    {
                        dst[lane] = lane_ptrs[lane][n];
                    }
                    else
                    {
                        dst[lane] = 0;
                    }
                }
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

        constexpr int K_chunks = K_BLK / 4;
        constexpr int num_groups = M_R / 4;
        constexpr int group_stride = K_chunks * 16;
        constexpr int A_block_bytes = num_groups * group_stride;
        constexpr int nr_chunks = N_R / 16;

        const __m512 zero_ps = _mm512_setzero_ps();
        const __m512i zero_i32 = _mm512_setzero_si512();
        const __m512i ones_byte = _mm512_set1_epi8(1);
        const __m512i correction_scale = _mm512_set1_epi32(128);
        alignas(64) int8_t b_tail_buffer[64];

        const auto load_b_vec = [&](const int8_t *chunk_base, int col_base) -> __m512i
        {
            if (col_base >= Bp.N)
            {
                return _mm512_setzero_si512();
            }

            const int remaining_cols = Bp.N - col_base;
            const int8_t *src = chunk_base + static_cast<size_t>(col_base) * Bp.ld_col;

            if (remaining_cols >= 16)
            {
                return _mm512_loadu_si512(reinterpret_cast<const void *>(src));
            }

            const int copy_cols = std::max(0, remaining_cols);
            return detail::load_b_vec_tail(src, copy_cols, Bp.ld_col, b_tail_buffer);
        };

        // 1. Initialize accumulation tile to zero (bias is added after accumulation)
        for (int idx = 0; idx < M_R * N_R; idx += 16)
        {
            _mm512_store_ps(C_tile + idx, zero_ps);
        }

        // 2. Precompute sB vectors (vectorized)
        __m512 sB_vecs[nr_chunks];
        for (int j = 0; j < nr_chunks; ++j)
        {
            sB_vecs[j] = _mm512_load_ps(wgt_scales + j * 16);
        }

        // 3. Loop over K blocks
        for (int t = 0; t < T; ++t)
        {
            // Accumulators
            __m512i acc[M_R][nr_chunks];
            for (int m = 0; m < M_R; ++m)
                for (int j = 0; j < nr_chunks; ++j)
                    acc[m][j] = _mm512_setzero_si512();

            const int8_t *base_A_block = A_tile_packed + t * A_block_bytes;
            const int8_t *base_B_block = Bp.block_ptr(t);

            alignas(64) __m512i stage_a_vecs[UNROLL_K][num_groups][4];
            alignas(64) __m512i stage_b_vecs[UNROLL_K][nr_chunks];
            alignas(64) __m512i stage_corrections[UNROLL_K][nr_chunks];

            const auto load_a_group = [&](int stage_idx, int group_idx, int kk_idx)
            {
                const int8_t *src = base_A_block + group_idx * group_stride + kk_idx * 16;
                const __m128i a32 = _mm_load_si128(reinterpret_cast<const __m128i *>(src));
                stage_a_vecs[stage_idx][group_idx][0] = broadcast_row_vec<0>(a32);
                stage_a_vecs[stage_idx][group_idx][1] = broadcast_row_vec<1>(a32);
                stage_a_vecs[stage_idx][group_idx][2] = broadcast_row_vec<2>(a32);
                stage_a_vecs[stage_idx][group_idx][3] = broadcast_row_vec<3>(a32);
            };

            const auto load_b_chunk = [&](int stage_idx, int chunk_idx, const int8_t *chunk_base)
            {
                const int col_base = N0 + chunk_idx * 16;
                const __m512i vec = load_b_vec(chunk_base, col_base);
                stage_b_vecs[stage_idx][chunk_idx] = vec;
                const __m512i sum = _mm512_dpbusd_epi32(zero_i32, ones_byte, vec);
                stage_corrections[stage_idx][chunk_idx] = _mm512_mullo_epi32(sum, correction_scale);
            };

            const auto compute_stage = [&](int stage_idx)
            {
                if constexpr (USE_VNNI)
                {
                    for (int g = 0; g < num_groups; ++g)
                    {
                        const int m_base = g * 4;
                        const __m512i *a_vec = stage_a_vecs[stage_idx][g];
                        for (int j = 0; j < nr_chunks; ++j)
                        {
                            const __m512i b_vec = stage_b_vecs[stage_idx][j];
                            const __m512i correction = stage_corrections[stage_idx][j];
                            for (int lane = 0; lane < 4; ++lane)
                            {
                                const int row = m_base + lane;
                                acc[row][j] = _mm512_dpbusd_epi32(acc[row][j], a_vec[lane], b_vec);
                                acc[row][j] = _mm512_sub_epi32(acc[row][j], correction);
                            }
                        }
                    }
                }
            };

            const auto load_stage = [&](int stage_idx, int kk_idx)
            {
                const int8_t *chunk_base = base_B_block + kk_idx * Bp.ld_chunk;
                int g = 0, j = 0;
                for (; g + 1 < num_groups && j + 1 < nr_chunks; g += 2, j += 2)
                {
                    load_a_group(stage_idx, g, kk_idx);
                    load_b_chunk(stage_idx, j, chunk_base);
                    load_a_group(stage_idx, g + 1, kk_idx);
                    load_b_chunk(stage_idx, j + 1, chunk_base);
                }
                for (; g < num_groups; ++g)
                {
                    load_a_group(stage_idx, g, kk_idx);
                }
                for (; j < nr_chunks; ++j)
                {
                    load_b_chunk(stage_idx, j, chunk_base);
                }
            };

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
                int stages_loaded = 0;
                for (int u = 0; u < UNROLL_K; ++u)
                {
                    const int k_off = kk + 4 * u;
                    if (k_off >= K_BLK)
                    {
                        break;
                    }
                    const int kk_idx = k_off / 4;
                    load_stage(u, kk_idx);
                    ++stages_loaded;
                    if (u > 0)
                    {
                        compute_stage(u - 1);
                    }
                }

                if (stages_loaded > 0)
                {
                    compute_stage(stages_loaded - 1);
                }
            } // kk

            // 3.3 Scale and accumulate into C_tile
            const float sA_t = act_scales[t];
            const __m512 sA_vec = _mm512_set1_ps(sA_t);

            for (int j = 0; j < nr_chunks; ++j)
            {
                const __m512 sAB_vec = _mm512_mul_ps(sA_vec, sB_vecs[j]);

                for (int m = 0; m < M_R; ++m)
                {
                    const __m512i acc_i32 = acc[m][j];
                    const __m512 acc_f32 = _mm512_cvtepi32_ps(acc_i32);
                    const __m512 contrib = _mm512_mul_ps(acc_f32, sAB_vec);

                    float *c_row = C_tile + m * N_R;
                    const __m512 c_old = _mm512_load_ps(c_row + j * 16);
                    const __m512 c_new = _mm512_add_ps(c_old, contrib);
                    _mm512_store_ps(c_row + j * 16, c_new);
                }
            }
        } // t

        // 4. Add bias once after accumulation
        for (int m = 0; m < M_R; ++m)
        {
            float *row = C_tile + m * N_R;
            for (int j = 0; j < nr_chunks; ++j)
            {
                const __m512 bias_vec = _mm512_load_ps(bias_tile + j * 16);
                const __m512 acc_vec = _mm512_load_ps(row + j * 16);
                _mm512_store_ps(row + j * 16, _mm512_add_ps(acc_vec, bias_vec));
            }
        }
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

// OpenMP parallel over M dimension with static scheduling for predictable work distribution
#pragma omp parallel
        {
            // Thread-private scratch buffers
            alignas(64) int8_t A_tile_packed[A_tile_total_bytes];
            alignas(64) float C_tile[M_R * N_R];
            alignas(64) float bias_tile[N_R];
            alignas(64) float wgt_scales_tile[N_R];

#pragma omp for schedule(static, 1)
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
                        _mm512_storeu_ps(bias_tile + n_idx, _mm512_setzero_ps());
                        _mm512_storeu_ps(wgt_scales_tile + n_idx, _mm512_setzero_ps());
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
        const int8_t *, int, int, int, int, int, int8_t *, int &, int &, int &);

    // Template instantiations are now generated by generate_vnni_gemm_instantiations.py
    // See kernels/cpu/gemm_v3/generated/VNNIGemmInstantiations_*.cpp

    // Additional common configurations can be added here as needed

} // namespace llaminar2
