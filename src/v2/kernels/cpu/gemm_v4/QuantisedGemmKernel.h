#pragma once

#include <immintrin.h>
#include "QuantisedGemmJit_M1.h"
#include "QuantisedGemmJit_M2.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/FP16Utils.h"
#include "../../../utils/CPUFeatures.h"
#include "../../../utils/DebugEnv.h"
#include <vector>
#include <memory>
#include <mutex>
#include <omp.h>
#include <cstring>
#include <iostream>

namespace llaminar2
{
    namespace gemm_v4
    {

        class QuantisedGemmKernel : public ITensorGemm
        {
        public:
            /**
             * @brief Generic constructor for any quantized tensor type
             *
             * The tensor must implement IINT8Unpackable interface. If it doesn't,
             * pack_weights_generic will print an error and the kernel will not work.
             * This allows gradual rollout as more tensor types implement IINT8Unpackable.
             */
            QuantisedGemmKernel(const TensorBase *weights)
            {
                pack_weights_generic(weights);
            }

            // Legacy type-specific constructors (deprecated, use generic constructor)
            QuantisedGemmKernel(const Q8_1Tensor *weights)
            {
                pack_weights_generic(weights);
            }

            QuantisedGemmKernel(const Q8_0Tensor *weights)
            {
                pack_weights_generic(weights);
            }

            QuantisedGemmKernel(const Q4_0Tensor *weights)
            {
                pack_weights_generic(weights);
            }

        private:
            // Helper to pack a single 32-element block
            __attribute__((always_inline)) void pack_single_block(
                int n,
                int k_blk,
                size_t row_base_offset,
                const int8_t *temp_vals,
                float scale,
                float min_val,
                int N_padded)
            {
                // Vectorized sum
                int32_t sum = 0;
#pragma omp simd reduction(+ : sum)
                for (int i = 0; i < 32; ++i)
                {
                    sum += temp_vals[i];
                }

                // Pack 32 bytes into 8 groups of 4 bytes, strided by 256
                // k_blk corresponds to 32 columns.
                // k starts at k_blk * 32.
                // k/4 starts at k_blk * 8.
                size_t block_offset = row_base_offset + (size_t)(k_blk * 8) * 256;

                // Ensure we don't write out of bounds if K is weirdly small/unaligned
                // But packed_data is resized to K * N_padded.
                // If K=33, K_blocks=2.
                // k_blk=1. block_offset = ... + 2048.
                // We write 32 bytes.
                // If K=33, packed_data size is 33*64 = 2112.
                // 2048 + 32 = 2080 < 2112. Safe.

                int8_t *dst_ptr = packed_weights_.packed_data.data() + block_offset;

                // Unroll the 8 writes
                for (int j = 0; j < 8; ++j)
                {
                    std::memcpy(dst_ptr + j * 256, &temp_vals[j * 4], 4);
                }

                packed_weights_.compensation[k_blk * N_padded + n] = sum;
                packed_weights_.scales[k_blk * N_padded + n] = scale;
                packed_weights_.mins[k_blk * N_padded + n] = min_val;
            }

            void pack_weights_generic(const TensorBase *weights)
            {
                // Weights are typically [N, K] (out_features, in_features)
                int N = weights->shape()[0];
                int K = weights->shape()[1];

                packed_weights_.K = K;
                packed_weights_.N = N;

                // Pad N to multiple of 64 for blocking
                int N_padded = (N + 63) / 64 * 64;
                // K_blocks covers all K, including tail
                int K_blocks = (K + 31) / 32;

                packed_weights_.packed_data.resize(K * N_padded);
                packed_weights_.compensation.resize(K_blocks * N_padded);
                packed_weights_.scales.resize(K_blocks * N_padded);
                packed_weights_.mins.resize(K_blocks * N_padded);

                // Check for IINT8Unpackable interface
                const IINT8Unpackable *unpackable = dynamic_cast<const IINT8Unpackable *>(weights);
                if (!unpackable)
                {
                    std::cerr << "Tensor type " << (int)weights->native_type() << " does not implement IINT8Unpackable" << std::endl;
                    return;
                }

                bool use_superblock = (unpackable->superblock_size() == 256);

// Iterate over N rows of weights (Parallelized)
#pragma omp parallel for schedule(static)
                for (int n = 0; n < N; ++n)
                {
                    int n_blk = n / 64;
                    int n_rem = n % 64;
                    // Base offset for this row n in the packed layout
                    // Layout: [N/64][K/4][64][4]
                    // Stride for K/4 is 256 bytes (64 * 4)
                    size_t row_base_offset = (size_t)n_blk * (K * 64) + n_rem * 4;

                    int k_blk = 0;

                    if (use_superblock)
                    {
                        // Process superblocks (8 blocks at a time)
                        int K_superblocks = K_blocks / 8;
                        for (int k_sb = 0; k_sb < K_superblocks; ++k_sb)
                        {
                            int8_t sb_vals[256];
                            float sb_scales[8];
                            float sb_mins[8];

                            unpackable->unpack_superblock_to_int8(n, k_sb, sb_vals, sb_scales, sb_mins);

                            for (int i = 0; i < 8; ++i)
                            {
                                pack_single_block(n, k_blk + i, row_base_offset, sb_vals + i * 32, sb_scales[i], sb_mins[i], N_padded);
                            }
                            k_blk += 8;
                        }
                    }

                    for (; k_blk < K_blocks; ++k_blk)
                    {
                        // Unpack block using generic interface
                        int8_t temp_vals[32];
                        unpackable->unpack_block_to_int8(n, k_blk, temp_vals);
                        float scale = unpackable->get_block_scale(n, k_blk);
                        float min_val = unpackable->get_block_min(n, k_blk);

                        pack_single_block(n, k_blk, row_base_offset, temp_vals, scale, min_val, N_padded);
                    }
                }
            }

        public:
            bool supports_device(int device_idx) const override
            {
                return device_idx == -1;
            }

            /**
             * @brief Matrix multiply with packed quantized weights: C = alpha * A @ B + beta * C
             *
             * @param A Input activations [m, k]
             * @param C Output [m, n], will be overwritten or accumulated based on beta
             * @param m Batch size (sequence length)
             * @param n Output features (must match packed_weights_.N)
             * @param k Input features (must match packed_weights_.K)
             * @param transpose_B Ignored - weights are pre-transposed during packing
             * @param alpha Scale factor for A @ B
             * @param beta If 0, overwrite C. If non-zero, accumulate: C = alpha*A@B + beta*C
             * @param ctx MPI context (unused for local compute)
             * @param device_idx Device index (-1 = CPU)
             */
            bool multiply(const float *A, float *C, int m, int n, int k, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                // Note: transpose_B is ignored - quantized weights are always pre-transposed during packing
                // Accumulate is determined by beta: beta > 0 means add to existing C
                (void)transpose_B;
                bool accumulate = (beta != 0.0f);
                return multiply_fused(A, C, m, n, k, nullptr, nullptr, false, nullptr, nullptr, accumulate, alpha, beta, ctx, device_idx);
            }

            bool multiply_fused(const float *A, float *C, int m, int n, int k,
                                const float *bias, const float *mask, bool do_softmax,
                                float *local_max, float *local_sum,
                                bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx,
                                const float *gate_input = nullptr, bool do_swiglu = false)
            {
                // Check dimensions
                if (n != packed_weights_.N || k != packed_weights_.K)
                {
                    std::cerr << "Dimension mismatch in QuantisedGemmKernel" << std::endl;
                    return false;
                }

                // Get JIT kernels
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                int k_blocks = (k + 31) / 32;
                int blocks_per_row = (n + 63) / 64; // Added for Softmax offset calculation

                // Calculate N_padded (must match packing)
                int N_padded = (n + 63) / 64 * 64;
                bool is_padded = (n != N_padded);

                // Determine beta handling mode
                bool needs_zero = (!accumulate || beta == 0.0f);
                bool needs_scale = (!needs_zero && beta != 1.0f);

                // Shared buffer for quantized A
                std::vector<uint8_t> shared_quantized_a(m * k_blocks * sizeof(Q8_1Block) + 64);
                Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(shared_quantized_a.data());

#pragma omp parallel
                {
                    // FUSED: Zero/scale C inside the same parallel region as quantize+GEMM
                    // This eliminates one parallel region entry per GEMM call
                    if (needs_zero)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            std::memset(C + i * n, 0, n * sizeof(float));
                        }
                    }
                    else if (needs_scale)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            float *row = C + i * n;
                            for (int j = 0; j < n; ++j)
                                row[j] *= beta;
                        }
                    }

                    // 1. Quantize A
                    // If M is small, parallelize over K to utilize threads
                    int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
                    if (quant_thresh == 0)
                        quant_thresh = omp_get_num_threads();

                    if (m < quant_thresh)
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                            {
                                const float *a_row = A + i * k;
                                Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                                float max_abs = 0.0f;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? std::abs(a_row[col_idx]) : 0.0f;
                                    if (val > max_abs)
                                        max_abs = val;
                                }

                                float d = max_abs / 127.0f;
                                if (d < 1e-10f)
                                    d = 1e-10f;
                                float id = 1.0f / d;

                                row_blocks[k_blk].d = fp32_to_fp16(d);

                                int32_t sum_qs = 0;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? a_row[col_idx] : 0.0f;
                                    int8_t q = static_cast<int8_t>(std::round(val * id));
                                    row_blocks[k_blk].qs[j] = q;
                                    sum_qs += q;
                                }

                                row_blocks[k_blk].sum_qs = sum_qs;
                            }
                        }
                    }
                    else
                    {
#pragma omp for schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            const float *a_row = A + i * k;
                            Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                            for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                            {
                                float max_abs = 0.0f;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? std::abs(a_row[col_idx]) : 0.0f;
                                    if (val > max_abs)
                                        max_abs = val;
                                }

                                float d = max_abs / 127.0f;
                                if (d < 1e-10f)
                                    d = 1e-10f;
                                float id = 1.0f / d;

                                row_blocks[k_blk].d = fp32_to_fp16(d);

                                int32_t sum_qs = 0;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? a_row[col_idx] : 0.0f;
                                    int8_t q = static_cast<int8_t>(std::round(val * id));
                                    row_blocks[k_blk].qs[j] = q;
                                    sum_qs += q;
                                }

                                row_blocks[k_blk].sum_qs = sum_qs;
                            }
                        }
                    }

                    // Implicit barrier here

                    // 2. GEMM (Parallel over N blocks and M)
                    // We want to parallelize over N blocks to keep B in L2 cache (B-stationary)
                    // when M is large.

                    // Detect cache sizes
                    int num_threads = omp_get_max_threads(); // Use max threads for planning
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024; // Fallback 1MB

                    long long l3_size = cpu_l3_cache_size();
                    // If L3 is unknown, assume it's large enough to not be the bottleneck compared to L2
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    // Calculate block size limit
                    // 1. L2 constraint: Block must fit in L2 (use configured % to leave room for A/C)
                    long long l2_limit = (long long)(l2_size * debugEnv().gemm.gemm_l2_limit_pct);

                    // 2. L3 constraint: All threads' blocks must fit in L3
                    // Use configured % of L3 shared among threads
                    long long l3_limit_per_thread = (long long)(l3_size * debugEnv().gemm.gemm_l3_share_pct / num_threads);

                    // Take the tighter constraint
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);

                    // Ensure we have at least some reasonable block size (e.g. 64KB)
                    int min_block_size = debugEnv().gemm.gemm_min_block_size;
                    if (block_size_limit < min_block_size)
                        block_size_limit = min_block_size;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * debugEnv().gemm.gemm_oversubscription_factor;
                    int m_granularity = debugEnv().gemm.gemm_m_task_granularity;
                    int m_tasks = (m + m_granularity - 1) / m_granularity;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    // Align to 64
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    // Clamp to max_n_block (L2 cache constraint)
                    // Ensure even splitting if we clamp to avoid load imbalance (e.g. 832 vs 64)
                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        // Align to 64
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    // Check if we need K-tiling (when B-block spills L2 cache)
                    // l2_size is already detected above

                    // Use configured % of L2 cache as threshold for K-tiling
                    // This ensures we tile even for moderate block sizes when M is large
                    const long long L2_CACHE_SIZE = (long long)(l2_size * debugEnv().gemm.gemm_k_tile_threshold_pct);
                    bool needs_k_tiling = ((long long)n_task_block * k > L2_CACHE_SIZE);

                    // Check if we have enough parallelism to avoid collapsing M
                    int num_n_tasks = (n + n_task_block - 1) / n_task_block;
                    bool enough_parallelism = (num_n_tasks >= omp_get_num_threads());

                    if (needs_k_tiling && enough_parallelism)
                    {
                        // K-tiling path: Parallelize N only, tile K inside to reuse B in L2
                        // Calculate dynamic tile size based on L2 cache and n_task_block
                        // We want n_task_block * (k_tile_blocks * 32) * sizeof(int8_t) <= L2_CACHE_SIZE * fraction
                        // Let's use configured % of L2 for B tile
                        long long target_b_size = (long long)(l2_size * debugEnv().gemm.gemm_target_b_size_pct);
                        int k_tile_elements = (int)(target_b_size / n_task_block);
                        int k_tile_blocks = k_tile_elements / 32;

                        // Clamp to reasonable limits
                        if (k_tile_blocks < debugEnv().gemm.gemm_k_tile_min_blocks)
                            k_tile_blocks = debugEnv().gemm.gemm_k_tile_min_blocks;
                        if (k_tile_blocks > debugEnv().gemm.gemm_k_tile_max_blocks)
                            k_tile_blocks = debugEnv().gemm.gemm_k_tile_max_blocks;

#pragma omp for schedule(dynamic)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            int n_end = std::min(n, n_task + n_task_block);

                            // Iterate over K tiles
                            for (int k_start = 0; k_start < k_blocks; k_start += k_tile_blocks)
                            {
                                int k_count = std::min(k_tile_blocks, k_blocks - k_start);

                                // Iterate over M (reuse B-tile for all M)
                                int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                                if (unroll != 1 && unroll != 2)
                                    unroll = 2;

                                for (int i = 0; i < m; i += unroll)
                                {
                                    int rows_left = m - i;
                                    int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;
                                    if (is_padded && rows_to_process > 1)
                                        rows_to_process = 1;

                                    Q8_1Block *blocks = all_blocks + i * k_blocks + k_start;

                                    for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                    {
                                        // Calculate packed weights offset
                                        // Base offset for N-block + Offset for K-tile
                                        // Note: packed_weights_ is [N_blocks][K_rows][64]
                                        // So offset = (n_blk/64) * (K*64) + (k_start*32)*64
                                        size_t weights_offset = (size_t)(n_blk / 64) * (k * 64) + (size_t)k_start * 32 * 64;
                                        const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                        // Fix: Offset compensation and scales by k_start * N_padded
                                        const int32_t *comp_ptr = packed_weights_.compensation.data() + (size_t)k_start * N_padded + n_blk;
                                        const float *scales_ptr = packed_weights_.scales.data() + (size_t)k_start * N_padded + n_blk;
                                        const float *mins_ptr = packed_weights_.mins.data() + (size_t)k_start * N_padded + n_blk;

                                        QuantisedGemmParams params;
                                        params.A = blocks;
                                        params.B_packed = b_ptr;
                                        params.comp = comp_ptr;
                                        params.scales = scales_ptr;
                                        params.mins = mins_ptr;
                                        params.K_blocks = k_count;
                                        params.N = 64;
                                        params.ldc = N_padded;
                                        params.bias = bias ? bias + n_blk : nullptr;
                                        params.mask = mask ? mask + i * n + n_blk : nullptr;
                                        params.A_stride = k_blocks * sizeof(Q8_1Block);

                                        bool is_tail = (n_blk + 64 > n);
                                        float C_temp[64];
                                        if (is_tail)
                                        {
                                            int valid_n = n - n_blk;
                                            std::memcpy(C_temp, C + i * n + n_blk, valid_n * sizeof(float));
                                            std::memset(C_temp + valid_n, 0, (64 - valid_n) * sizeof(float));
                                            params.C = C_temp;
                                        }
                                        else
                                        {
                                            params.C = C + i * n + n_blk;
                                        }

                                        bool is_last_k_tile = (k_start + k_count == k_blocks);
                                        bool current_do_softmax = do_softmax && is_last_k_tile;
                                        bool current_do_swiglu = do_swiglu && is_last_k_tile;

                                        float tmp_max[2], tmp_sum[2];
                                        if (current_do_softmax)
                                        {
                                            if (rows_to_process == 2)
                                            {
                                                params.local_max = tmp_max;
                                                params.local_sum = tmp_sum;
                                            }
                                            else
                                            {
                                                int block_idx = n_blk / 64;
                                                params.local_max = local_max + i * blocks_per_row + block_idx;
                                                params.local_sum = local_sum + i * blocks_per_row + block_idx;
                                            }
                                        }
                                        else
                                        {
                                            params.local_max = nullptr;
                                            params.local_sum = nullptr;
                                        }
                                        params.do_softmax = current_do_softmax;

                                        if (current_do_swiglu)
                                        {
                                            params.gate_input = gate_input + i * n + n_blk;
                                            params.do_swiglu = true;
                                        }
                                        else
                                        {
                                            params.gate_input = nullptr;
                                            params.do_swiglu = false;
                                        }

                                        if (rows_to_process == 2)
                                        {
                                            kernel_m2(&params);
                                            if (current_do_softmax)
                                            {
                                                int block_idx = n_blk / 64;
                                                local_max[i * blocks_per_row + block_idx] = tmp_max[0];
                                                local_sum[i * blocks_per_row + block_idx] = tmp_sum[0];
                                                local_max[(i + 1) * blocks_per_row + block_idx] = tmp_max[1];
                                                local_sum[(i + 1) * blocks_per_row + block_idx] = tmp_sum[1];
                                            }
                                        }
                                        else
                                            kernel(&params);

                                        if (is_tail)
                                        {
                                            int valid_n = n - n_blk;
                                            std::memcpy(C + i * n + n_blk, C_temp, valid_n * sizeof(float));
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                        if (unroll != 1 && unroll != 2)
                            unroll = 2;

                        // Standard path: Collapse M and N for maximum parallelism
#pragma omp for collapse(2) schedule(static)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            for (int i = 0; i < m; i += unroll)
                            {
                                int rows_left = m - i;
                                int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;
                                if (is_padded && rows_to_process > 1)
                                    rows_to_process = 1;

                                Q8_1Block *blocks = all_blocks + i * k_blocks;

                                int n_end = std::min(n, n_task + n_task_block);
                                for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                {
                                    // Calculate packed weights offset
                                    size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                    const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                    const int32_t *comp_ptr = packed_weights_.compensation.data() + n_blk;
                                    const float *scales_ptr = packed_weights_.scales.data() + n_blk;
                                    const float *mins_ptr = packed_weights_.mins.data() + n_blk;

                                    QuantisedGemmParams params;
                                    params.A = blocks;
                                    params.B_packed = b_ptr;
                                    params.comp = comp_ptr;
                                    params.scales = scales_ptr;
                                    params.mins = mins_ptr;
                                    params.K_blocks = k_blocks;
                                    params.N = 64;
                                    params.ldc = N_padded;
                                    params.bias = bias ? bias + n_blk : nullptr;
                                    params.mask = mask ? mask + i * n + n_blk : nullptr;
                                    params.A_stride = k_blocks * sizeof(Q8_1Block);

                                    bool is_tail = (n_blk + 64 > n);
                                    float C_temp[64];
                                    if (is_tail)
                                    {
                                        int valid_n = n - n_blk;
                                        std::memcpy(C_temp, C + i * n + n_blk, valid_n * sizeof(float));
                                        std::memset(C_temp + valid_n, 0, (64 - valid_n) * sizeof(float));
                                        params.C = C_temp;
                                    }
                                    else
                                    {
                                        params.C = C + i * n + n_blk;
                                    }

                                    float tmp_max[2], tmp_sum[2];
                                    if (do_softmax)
                                    {
                                        if (rows_to_process == 2)
                                        {
                                            params.local_max = tmp_max;
                                            params.local_sum = tmp_sum;
                                        }
                                        else
                                        {
                                            int block_idx = n_blk / 64;
                                            params.local_max = local_max + i * blocks_per_row + block_idx;
                                            params.local_sum = local_sum + i * blocks_per_row + block_idx;
                                        }
                                    }
                                    else
                                    {
                                        params.local_max = nullptr;
                                        params.local_sum = nullptr;
                                    }
                                    params.do_softmax = do_softmax;

                                    if (do_swiglu)
                                    {
                                        params.gate_input = gate_input + i * n + n_blk;
                                        params.do_swiglu = true;
                                    }
                                    else
                                    {
                                        params.gate_input = nullptr;
                                        params.do_swiglu = false;
                                    }

                                    if (rows_to_process == 2)
                                    {
                                        kernel_m2(&params);
                                        if (do_softmax)
                                        {
                                            int block_idx = n_blk / 64;
                                            local_max[i * blocks_per_row + block_idx] = tmp_max[0];
                                            local_sum[i * blocks_per_row + block_idx] = tmp_sum[0];
                                            local_max[(i + 1) * blocks_per_row + block_idx] = tmp_max[1];
                                            local_sum[(i + 1) * blocks_per_row + block_idx] = tmp_sum[1];
                                        }
                                    }
                                    else
                                        kernel(&params);

                                    if (is_tail)
                                    {
                                        int valid_n = n - n_blk;
                                        std::memcpy(C + i * n + n_blk, C_temp, valid_n * sizeof(float));
                                    }
                                }
                            }
                        }
                    }
                }

                return true;
            }

            /**
             * @brief Activation-activation multiply with packed weight fallback
             *
             * When B is nullptr, falls back to multiply() which uses packed weights.
             * This allows the pipeline to use multiply_activations uniformly for both
             * weight projections (Q/K/V) and attention (Q@K^T, scores@V).
             *
             * @note The `transpose_B` parameter is ignored for weight projections
             *       (B=nullptr) since weights are always pre-transposed during packing.
             */
            bool multiply_activations(const float *A, const float *B, float *C, int m, int n, int k, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                // When B is nullptr, use packed weights via multiply()
                if (!B)
                {
                    // Note: transpose_B is ignored - packed weights are always in the correct layout
                    // beta > 0 means accumulate into existing C
                    bool accumulate = (beta > 0.0f);
                    return multiply(A, C, m, n, k, accumulate, alpha, beta, ctx, device_idx);
                }

                // Activation-activation GEMM not supported - use FloatingPointGemmKernel
                std::cerr << "[QuantisedGemmKernel] multiply_activations with two activation matrices not supported" << std::endl;
                return false;
            }

            /**
             * @brief Strided activation-activation multiply (not supported for quantized kernel)
             *
             * QuantisedGemmKernel is designed for weight projections (packed INT8 weights).
             * For activation-activation GEMM (attention Q@K^T, scores@V), use FloatingPointGemmKernel.
             */
            bool multiply_activations_strided(const float *A, const float *B, float *C, int m, int n, int k, int stride_a, int stride_b, int stride_c, bool transpose_B, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                (void)A;
                (void)B;
                (void)C;
                (void)m;
                (void)n;
                (void)k;
                (void)stride_a;
                (void)stride_b;
                (void)stride_c;
                (void)transpose_B;
                (void)alpha;
                (void)beta;
                (void)ctx;
                (void)device_idx;
                std::cerr << "[QuantisedGemmKernel] multiply_activations_strided not supported - use FloatingPointGemmKernel" << std::endl;
                return false;
            }

            // =============================================================================
            // Fused Multi-GEMM Interface Implementation (Activation Sharing)
            // =============================================================================

            /**
             * @brief Check if this kernel supports activation sharing
             */
            bool supports_activation_sharing() const override
            {
                return true; // QuantisedGemmKernel supports Q8_1 activation sharing
            }

            /**
             * @brief Get buffer size needed for quantized activations
             */
            size_t get_quantized_activation_buffer_size(int m, int k) const override
            {
                int k_blocks = (k + 31) / 32;
                return static_cast<size_t>(m) * k_blocks * sizeof(Q8_1Block);
            }

            /**
             * @brief Quantize FP32 activations to Q8_1 for reuse across multiple GEMMs
             *
             * This is the first step in the fused multi-GEMM workflow.
             * The resulting Q8_1 blocks can be passed to multiply_with_precomputed_q8_1()
             * multiple times without redundant quantization.
             */
            bool quantize_activations(
                const float *A,
                void *q8_1_buffer,
                int m, int k) override
            {
                if (!A || !q8_1_buffer)
                {
                    return false;
                }

                int k_blocks = (k + 31) / 32;
                Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(q8_1_buffer);

#pragma omp parallel
                {
                    // Parallelize over rows for large M, or collapse for small M
                    int quant_thresh = debugEnv().gemm.gemm_quant_parallel_threshold;
                    if (quant_thresh == 0)
                        quant_thresh = omp_get_num_threads();

                    if (m < quant_thresh)
                    {
#pragma omp for collapse(2) schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                            {
                                const float *a_row = A + i * k;
                                Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                                float max_abs = 0.0f;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? std::abs(a_row[col_idx]) : 0.0f;
                                    if (val > max_abs)
                                        max_abs = val;
                                }

                                float d = max_abs / 127.0f;
                                if (d < 1e-10f)
                                    d = 1e-10f;
                                float id = 1.0f / d;

                                row_blocks[k_blk].d = fp32_to_fp16(d);

                                int32_t sum_qs = 0;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? a_row[col_idx] : 0.0f;
                                    int8_t q = static_cast<int8_t>(std::round(val * id));
                                    row_blocks[k_blk].qs[j] = q;
                                    sum_qs += q;
                                }

                                row_blocks[k_blk].sum_qs = sum_qs;
                            }
                        }
                    }
                    else
                    {
#pragma omp for schedule(static)
                        for (int i = 0; i < m; ++i)
                        {
                            const float *a_row = A + i * k;
                            Q8_1Block *row_blocks = all_blocks + i * k_blocks;

                            for (int k_blk = 0; k_blk < k_blocks; ++k_blk)
                            {
                                float max_abs = 0.0f;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? std::abs(a_row[col_idx]) : 0.0f;
                                    if (val > max_abs)
                                        max_abs = val;
                                }

                                float d = max_abs / 127.0f;
                                if (d < 1e-10f)
                                    d = 1e-10f;
                                float id = 1.0f / d;

                                row_blocks[k_blk].d = fp32_to_fp16(d);

                                int32_t sum_qs = 0;
                                for (int j = 0; j < 32; ++j)
                                {
                                    int col_idx = k_blk * 32 + j;
                                    float val = (col_idx < k) ? a_row[col_idx] : 0.0f;
                                    int8_t q = static_cast<int8_t>(std::round(val * id));
                                    row_blocks[k_blk].qs[j] = q;
                                    sum_qs += q;
                                }

                                row_blocks[k_blk].sum_qs = sum_qs;
                            }
                        }
                    }
                }

                return true;
            }

            /**
             * @brief GEMM with pre-quantized Q8_1 activations and fused post-ops
             *
             * This is the second step in the fused multi-GEMM workflow.
             * Uses pre-quantized activations from quantize_activations(),
             * eliminating redundant FP32→Q8_1 conversion.
             *
             * Supports fused operations via GemmFusedOps:
             * - SwiGLU: output *= swish(gate)
             * - Softmax: output = softmax(output * scale + mask)
             */
            bool multiply_with_precomputed_q8_1(
                const void *q8_1_activations,
                float *C,
                int m, int n, int k,
                const float *bias,
                bool accumulate,
                float alpha, float beta,
                const MPIContext *mpi_ctx,
                int device_idx,
                const GemmFusedOps &fused_ops) override
            {
                (void)mpi_ctx;
                (void)device_idx;
                (void)alpha; // Currently assumed 1.0

                if (!q8_1_activations || !C)
                    return false;

                // Check dimensions against packed weights
                if (n != packed_weights_.N || k != packed_weights_.K)
                {
                    std::cerr << "Dimension mismatch in multiply_with_precomputed_q8_1: "
                              << "expected n=" << packed_weights_.N << ", k=" << packed_weights_.K
                              << " got n=" << n << ", k=" << k << std::endl;
                    return false;
                }

                // Extract fused op parameters
                const float *gate_input = fused_ops.is_swiglu() ? fused_ops.gate_input : nullptr;
                bool do_swiglu = fused_ops.is_swiglu();
                bool do_softmax = fused_ops.is_softmax();
                const float *softmax_mask = fused_ops.softmax_mask;
                float *local_max = fused_ops.online_max;
                float *local_sum = fused_ops.online_sum;

                // Get JIT kernels
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                // Determine beta handling mode
                bool needs_zero = (!accumulate || beta == 0.0f);
                bool needs_scale = (!needs_zero && beta != 1.0f);

                int k_blocks = k / 32;
                const Q8_1Block *all_blocks = static_cast<const Q8_1Block *>(q8_1_activations);

#pragma omp parallel
                {
                    // FUSED: Zero/scale C inside the same parallel region as GEMM
                    // This eliminates one parallel region entry per GEMM call
                    if (needs_zero)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            std::memset(C + i * n, 0, n * sizeof(float));
                        }
                    }
                    else if (needs_scale)
                    {
#pragma omp for schedule(static) nowait
                        for (int i = 0; i < m; ++i)
                        {
                            float *row = C + i * n;
                            for (int j = 0; j < n; ++j)
                                row[j] *= beta;
                        }
                    }

                    // Barrier to ensure zeroing complete before GEMM writes
#pragma omp barrier

                    // Cache-aware blocking (same logic as other paths)
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            // Pointer to Q8_1 blocks for this row
                            const Q8_1Block *blocks = all_blocks + i * k_blocks;

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = packed_weights_.compensation.data() + n_blk;
                                const float *scales_ptr = packed_weights_.scales.data() + n_blk;
                                const float *mins_ptr = packed_weights_.mins.data() + n_blk;

                                QuantisedGemmParams params;
                                params.A = blocks;
                                params.B_packed = b_ptr;
                                params.comp = comp_ptr;
                                params.scales = scales_ptr;
                                params.mins = mins_ptr;
                                params.C = C + i * n + n_blk;
                                params.K_blocks = k_blocks;
                                params.N = 64;
                                params.ldc = n;
                                params.bias = bias ? bias + n_blk : nullptr;
                                params.mask = softmax_mask ? (softmax_mask + i * n + n_blk) : nullptr;
                                params.A_stride = k_blocks * sizeof(Q8_1Block);
                                params.local_max = local_max;
                                params.local_sum = local_sum;
                                params.do_softmax = do_softmax;
                                params.gate_input = gate_input ? (gate_input + i * n + n_blk) : nullptr;
                                params.do_swiglu = do_swiglu;

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);
                            }
                        }
                    }
                }

                return true;
            }

            /**
             * @brief Tensor-based GEMM with type introspection
             *
             * Optimized paths:
             * - Q8_1Tensor activations: Skip FP32→Q8_1 quantization (zero-copy blocks)
             * - FP32Tensor activations: Standard path with online quantization
             *
             * @param A Input activations tensor [m, k]
             * @param C Output tensor [m, n] (must be FP32 for now)
             * @param transpose_B Whether B is transposed (ignored, weights pre-packed)
             * @param alpha Scale factor (must be 1.0 for now)
             * @param beta Accumulate factor (0.0 = overwrite, 1.0 = add)
             * @param mpi_ctx MPI context
             * @param device_idx Device index
             *
             * @return true on success
             */
            bool multiply_tensor(
                const TensorBase *A, TensorBase *C,
                bool transpose_B = true,
                float alpha = 1.0f, float beta = 0.0f,
                const MPIContext *mpi_ctx = nullptr,
                int device_idx = -1) override
            {
                (void)transpose_B; // Weights are pre-packed

                const auto &a_shape = A->shape();
                const auto &c_shape = C->shape();
                int m = static_cast<int>(a_shape[0]);
                int k = static_cast<int>(a_shape.size() > 1 ? a_shape[1] : 1);
                int n = static_cast<int>(c_shape.size() > 1 ? c_shape[1] : c_shape[0]);

                // Check for Q8_1 activation fast path
                if (A->native_type() == TensorType::Q8_1)
                {
                    return multiply_q8_1_direct(
                        static_cast<const Q8_1Tensor *>(A),
                        C->mutable_data(), m, n, k,
                        beta != 0.0f, alpha, beta, mpi_ctx, device_idx);
                }

                // Fallback: FP32 path (with online quantization)
                return multiply(A->data(), C->mutable_data(), m, n, k, transpose_B, alpha, beta, mpi_ctx, device_idx);
            }

        private:
            /**
             * @brief Direct Q8_1 activation path (no float conversion)
             *
             * Uses pre-quantized blocks directly from Q8_1Tensor.
             * Eliminates the float→Q8_1 conversion overhead.
             */
            bool multiply_q8_1_direct(
                const Q8_1Tensor *A_tensor, float *C,
                int m, int n, int k,
                bool accumulate, float alpha, float beta,
                const MPIContext *ctx, int device_idx)
            {
                (void)ctx;
                (void)device_idx;
                (void)alpha; // Currently ignored (assumed 1.0)

                // Check dimensions
                if (n != packed_weights_.N || k != packed_weights_.K)
                {
                    std::cerr << "Dimension mismatch in QuantisedGemmKernel::multiply_q8_1_direct" << std::endl;
                    return false;
                }

                // Get JIT kernels
                static QuantisedGemmJit_M1 jit;
                static QuantisedGemmJit_M2 jit_m2;
                auto kernel = jit.get_kernel();
                auto kernel_m2 = jit_m2.get_kernel();

                // Handle beta scaling / zeroing
                if (!accumulate || beta == 0.0f)
                {
#pragma omp parallel for
                    for (size_t i = 0; i < (size_t)m * n; ++i)
                        C[i] = 0.0f;
                }
                else if (beta != 1.0f)
                {
#pragma omp parallel for
                    for (size_t i = 0; i < (size_t)m * n; ++i)
                        C[i] *= beta;
                }

                int k_blocks = k / 32;
                int blocks_per_row_output = (n + 63) / 64;

                // Get direct access to Q8_1 blocks (zero-copy!)
                // A_tensor stores blocks in row-major order: [m, k_blocks] of Q8_1Block
                const size_t blocks_per_row_input = (k + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

#pragma omp parallel
                {
                    // Detect cache sizes (same as FP32 path)
                    int num_threads = omp_get_num_threads();
                    long long l2_size = cpu_l2_cache_size();
                    if (l2_size == 0)
                        l2_size = 1024 * 1024;

                    long long l3_size = cpu_l3_cache_size();
                    if (l3_size == 0)
                        l3_size = l2_size * num_threads;

                    long long l2_limit = (long long)(l2_size * 0.9);
                    long long l3_limit_per_thread = (long long)(l3_size * 0.9 / num_threads);
                    long long block_size_limit = std::min(l2_limit, l3_limit_per_thread);
                    if (block_size_limit < 65536)
                        block_size_limit = 65536;

                    int max_n_block = block_size_limit / k;
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    int target_tasks = num_threads * 4;
                    int m_tasks = (m + 1) / 2;
                    if (m_tasks < 1)
                        m_tasks = 1;

                    int needed_n_tasks = (target_tasks + m_tasks - 1) / m_tasks;
                    if (needed_n_tasks < 1)
                        needed_n_tasks = 1;

                    int calc_block = (n + needed_n_tasks - 1) / needed_n_tasks;
                    calc_block = (calc_block + 63) / 64 * 64;
                    if (calc_block < 64)
                        calc_block = 64;

                    int n_task_block;
                    if (calc_block > max_n_block)
                    {
                        int num_splits = (n + max_n_block - 1) / max_n_block;
                        int even_block = (n + num_splits - 1) / num_splits;
                        even_block = (even_block + 63) / 64 * 64;
                        n_task_block = even_block;
                    }
                    else
                    {
                        n_task_block = calc_block;
                    }

                    // Standard path: Collapse M and N for maximum parallelism
                    // (K-tiling path would need similar adaptation)
                    int unroll = debugEnv().gemm.gemm_m_unroll_factor;
                    if (unroll != 1 && unroll != 2)
                        unroll = 2;

#pragma omp for collapse(2) schedule(static)
                    for (int n_task = 0; n_task < n; n_task += n_task_block)
                    {
                        for (int i = 0; i < m; i += unroll)
                        {
                            int rows_left = m - i;
                            int rows_to_process = (rows_left >= unroll) ? unroll : rows_left;

                            // Get pointer to first block for row i
                            // Use get_raw_block_at to get the Q8_1Block* directly
                            const Q8_1Block *blocks = static_cast<const Q8_1Block *>(
                                A_tensor->get_raw_block_at(i, 0));

                            int n_end = std::min(n, n_task + n_task_block);
                            for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                            {
                                size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                const int32_t *comp_ptr = packed_weights_.compensation.data() + n_blk;
                                const float *scales_ptr = packed_weights_.scales.data() + n_blk;
                                const float *mins_ptr = packed_weights_.mins.data() + n_blk;

                                QuantisedGemmParams params;
                                params.A = blocks;
                                params.B_packed = b_ptr;
                                params.comp = comp_ptr;
                                params.scales = scales_ptr;
                                params.mins = mins_ptr;
                                params.C = C + i * n + n_blk;
                                params.K_blocks = k_blocks;
                                params.N = 64;
                                params.ldc = n;
                                params.bias = nullptr;
                                params.mask = nullptr;
                                params.A_stride = blocks_per_row_input * sizeof(Q8_1Block);
                                params.local_max = nullptr;
                                params.local_sum = nullptr;
                                params.do_softmax = false;

                                if (rows_to_process == 2)
                                    kernel_m2(&params);
                                else
                                    kernel(&params);
                            }
                        }
                    }
                }

                return true;
            }

            QuantisedPackedWeights packed_weights_;
        };

    }
}
