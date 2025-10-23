#include "QuantLinearKernel.h"
#include "../Logger.h"
#include <algorithm>
#include <cstring>
#include <omp.h>
#include "QuantLinearInstrumentation.h"
#include <immintrin.h>

namespace llaminar
{

    bool quant_linear_fused(const QuantizedTensor &Wq, const QuantLinearParams &p)
    {
        if (!debugEnv().quant.fused_enable)
        {
            LOG_WARN("quant_linear_fused invoked while LLAMINAR_QUANT_FUSED_ENABLE not set; skipping (returns true no-op).");
            // Behave as a no-op; callers should migrate to slab path.
            if (p.zero_C && p.C)
                std::fill(p.C, p.C + (size_t)p.M * p.N, 0.0f);
            return true;
        }
        // Basic validation
        if (!p.A || !p.C)
        {
            LOG_ERROR("quant_linear_fused: null A or C pointer");
            return false;
        }
        if (p.M <= 0 || p.K <= 0 || p.N <= 0)
        {
            LOG_ERROR("quant_linear_fused: invalid dims M=" << p.M << " K=" << p.K << " N=" << p.N);
            return false;
        }
        const auto &layout = Wq.layout();
        if (layout.original_shape.size() != 2)
        {
            LOG_ERROR("quant_linear_fused: weight tensor not 2D");
            return false;
        }
        int WK = layout.original_shape[0];
        int WN = layout.original_shape[1];
        if (WK != p.K || WN != p.N)
        {
            LOG_ERROR("quant_linear_fused: shape mismatch W[K=" << WK << ",N=" << WN << "] vs params K=" << p.K << " N=" << p.N);
            return false;
        }
        if (p.zero_C)
        {
            std::fill(p.C, p.C + (size_t)p.M * p.N, 0.0f);
        }

        const int block_elems = layout.block_desc.elements_per_block;
        const int blocks_per_row = (p.N + block_elems - 1) / block_elems;

        // Tiling heuristics
        int tile_n = p.tile_n;
        if (tile_n <= 0)
        {
            // Choose a multiple of block_elems; aim for ~256 or next multiple
            tile_n = (block_elems >= 256) ? block_elems : 256;
            if (tile_n % block_elems)
                tile_n = ((tile_n / block_elems) + 1) * block_elems;
            if (tile_n > p.N)
                tile_n = p.N;
        }
        else
        {
            // Align to block boundary
            if (tile_n % block_elems)
                tile_n = ((tile_n / block_elems) + 1) * block_elems;
            tile_n = std::min(tile_n, p.N);
        }
        int tile_k = p.tile_k;
        if (tile_k <= 0)
        {
            tile_k = (p.M >= 64 && p.K >= 64) ? 4 : 1; // decode several K rows together if workload large
        }
        tile_k = std::max(1, std::min(tile_k, p.K));

        // Weight tile cache: for current (k0,n0) tile we decode each (k,block) once and reuse for all M rows.
        // Layout: [k_span][block_count][block_elems]
        // To avoid repeated allocation inside loops, allocate max possible per tile.
        std::vector<float> weight_tile;
        weight_tile.reserve((size_t)tile_k * (size_t)((tile_n + block_elems - 1) / block_elems + 4) * block_elems);

        for (int n0 = 0; n0 < p.N; n0 += tile_n)
        {
            int n_span = std::min(tile_n, p.N - n0);
            int first_block = n0 / block_elems;
            int last_block = (n0 + n_span - 1) / block_elems;
            int block_count = last_block - first_block + 1;

            for (int k0 = 0; k0 < p.K; k0 += tile_k)
            {
                int k_span = std::min(tile_k, p.K - k0);
                // 1. Decode weight tile once
                weight_tile.resize((size_t)k_span * (size_t)block_count * block_elems);
                for (int kk = 0; kk < k_span; ++kk)
                {
                    int k = k0 + kk;
                    for (int bb = 0; bb < block_count; ++bb)
                    {
                        int b = first_block + bb;
                        size_t block_index = (size_t)k * (size_t)blocks_per_row + b;
                        float *dst_full = weight_tile.data() + ((size_t)kk * block_count + bb) * block_elems;
                        // Compute overlap for potential partial decode
                        int col0 = b * block_elems;
                        int col_block_span = std::min(block_elems, p.N - col0);
                        int rel_start = std::max(0, n0 - col0);
                        int rel_end = std::min(col_block_span, n0 + n_span - col0);
                        int span = rel_end - rel_start;
                        bool k_fmt = layout.block_desc.is_k_quant;
                        if (span > 0 && span < block_elems / 2 && !k_fmt)
                        {
                            // Decode whole then compact subset to front
                            Wq.decodeBlock(block_index, dst_full);
                            quantLinearInstr().decoded_blocks.fetch_add(1, std::memory_order_relaxed);
                            quantLinearInstr().partial_blocks.fetch_add(1, std::memory_order_relaxed);
                            if (rel_start > 0)
                            {
                                std::memmove(dst_full, dst_full + rel_start, span * sizeof(float));
                            }
                            // Optionally zero remaining tail for debugging clarity (left intact for now)
                        }
                        else
                        {
                            Wq.decodeBlock(block_index, dst_full);
                            quantLinearInstr().decoded_blocks.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                }
                // 2. Apply decoded weights to all rows of A
                for (int bb = 0; bb < block_count; ++bb)
                {
                    int b = first_block + bb;
                    int col0 = b * block_elems;
                    int col_block_span = std::min(block_elems, p.N - col0);
                    int rel_start = std::max(0, n0 - col0);
                    if (rel_start >= col_block_span)
                        continue;
                    int rel_end = std::min(col_block_span, n0 + n_span - col0);
                    int span = rel_end - rel_start;
                    if (span <= 0)
                        continue;
                    int global_col_start = col0 + rel_start;
                    // For each k in tile accumulate into C
                    quantLinearInstr().applied_blocks.fetch_add(1, std::memory_order_relaxed);
                    bool used_simd = false;
                    bool small_case = (span < 8 && k_span <= 2 && p.M <= 16);
                    if (small_case)
                    {
                        // Scalar fast path: tight nested loops, no OpenMP overhead
                        for (int m = 0; m < p.M; ++m)
                        {
                            float *c_row = p.C + (size_t)m * p.N + global_col_start;
                            for (int kk = 0; kk < k_span; ++kk)
                            {
                                int k = k0 + kk;
                                float a_val = p.A[(size_t)m * p.K + k];
                                const float *w_base = weight_tile.data() + ((size_t)kk * block_count + bb) * block_elems;
                                const float *w_block = w_base + rel_start;
                                for (int j = 0; j < span; ++j)
                                {
                                    c_row[j] += a_val * w_block[j];
                                }
                            }
                        }
                        quantLinearInstr().scalar_path.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                    {
#pragma omp parallel for if (p.M * span * k_span > 4096) reduction(|| : used_simd)
                        for (int m = 0; m < p.M; ++m)
                        {
                            float *c_row = p.C + (size_t)m * p.N + global_col_start;
                            for (int kk = 0; kk < k_span; ++kk)
                            {
                                int k = k0 + kk;
                                float a_val = p.A[(size_t)m * p.K + k];
                                const float *w_base = weight_tile.data() + ((size_t)kk * block_count + bb) * block_elems;
                                const float *w_block = w_base + rel_start;
                                int j = 0;
#if defined(__AVX2__)
                                if (span >= 8)
                                {
                                    __m256 a_vec = _mm256_set1_ps(a_val);
                                    for (; j + 8 <= span; j += 8)
                                    {
                                        __m256 wv = _mm256_loadu_ps(w_block + j);
                                        __m256 cv = _mm256_loadu_ps(c_row + j);
                                        cv = _mm256_fmadd_ps(a_vec, wv, cv);
                                        _mm256_storeu_ps(c_row + j, cv);
                                    }
                                    used_simd = true;
                                }
#endif
                                for (; j < span; ++j)
                                {
                                    c_row[j] += a_val * w_block[j];
                                }
                            }
                        }
                        if (used_simd)
                            quantLinearInstr().simd_path.fetch_add(1, std::memory_order_relaxed);
                        else
                            quantLinearInstr().scalar_path.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        return true;
    }

} // namespace llaminar
