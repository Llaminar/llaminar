#pragma once

#include <immintrin.h>
#include "Q8_1GemmJit_M1.h"
#include "Q8_1GemmJit_M2.h"
#include "../../../tensors/Tensors.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/FP16Utils.h"
#include <vector>
#include <memory>
#include <mutex>
#include <omp.h>

namespace llaminar2
{
    namespace gemm_v4
    {

        class Q8_1GemmKernel : public ITensorGemm
        {
        public:
            Q8_1GemmKernel(const Q8_1Tensor *weights) : weights_(weights)
            {
                // Weights are typically [N, K] (out_features, in_features)
                // We need to pack them as K x N for the kernel
                int N = weights->shape()[0];
                int K = weights->shape()[1];

                // Pack weights
                // We need to read from weights tensor.
                // It supports get_raw_block_at(row, k_offset).
                // Here row is n (0..N), k_offset is k (0..K, step 32).

                packed_weights_.K = K;
                packed_weights_.N = N;

                // Pad N to multiple of 64 for blocking
                int N_padded = (N + 63) / 64 * 64;
                packed_weights_.packed_data.resize(K * N_padded);
                packed_weights_.compensation.resize((K / 32) * N);
                packed_weights_.scales.resize((K / 32) * N);

                // Iterate over N rows of weights
                for (int n = 0; n < N; ++n)
                {
                    for (int k_blk = 0; k_blk < K / 32; ++k_blk)
                    {
                        // Get block from weights
                        const void *raw_block = weights->get_raw_block_at(n, k_blk);
                        const Q8_1Block *block = static_cast<const Q8_1Block *>(raw_block);

                        int32_t sum = 0;
                        for (int i = 0; i < 32; ++i)
                        {
                            int k = k_blk * 32 + i;
                            int8_t val = block->qs[i];
                            sum += val;

                            // Pack into [N/64][K/4][64][4]
                            // Index: (n / 64) * (K * 64) + (k / 4) * 256 + (n % 64) * 4 + (k % 4)
                            int n_blk = n / 64;
                            int n_rem = n % 64;
                            int k_blk_4 = k / 4;
                            int k_rem = k % 4;
                            size_t packed_idx = (size_t)n_blk * (K * 64) + (size_t)k_blk_4 * 256 + n_rem * 4 + k_rem;
                            packed_weights_.packed_data[packed_idx] = val;
                        }
                        packed_weights_.compensation[k_blk * N + n] = sum;
                        packed_weights_.scales[k_blk * N + n] = fp16_to_fp32(block->d);
                    }
                }
            }

            bool supports_device(int device_idx) const override
            {
                return device_idx == -1;
            }

            bool multiply(const float *A, float *C, int m, int n, int k, bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                return multiply_fused(A, C, m, n, k, nullptr, nullptr, false, nullptr, nullptr, accumulate, alpha, beta, ctx, device_idx);
            }

            bool multiply_fused(const float *A, float *C, int m, int n, int k,
                                const float *bias, const float *mask, bool do_softmax,
                                float *local_max, float *local_sum,
                                bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx)
            {
                // Check dimensions
                if (n != packed_weights_.N || k != packed_weights_.K)
                {
                    std::cerr << "Dimension mismatch in Q8_1GemmKernel" << std::endl;
                    return false;
                }

                // Get JIT kernels
                static Q8_1GemmJit_M1 jit;
                static Q8_1GemmJit_M2 jit_m2;
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
                int blocks_per_row = (n + 63) / 64; // Added for Softmax offset calculation

                // Shared buffer for quantized A
                // We need to allocate this outside parallel region or use a shared vector
                // Since m can be large, we should be careful.
                // But for typical batch sizes (up to 512), it's fine.
                std::vector<uint8_t> shared_quantized_a(m * k_blocks * sizeof(Q8_1Block) + 64);
                Q8_1Block *all_blocks = reinterpret_cast<Q8_1Block *>(shared_quantized_a.data());

#pragma omp parallel
                {
                    // 1. Quantize A
                    // If M is small, parallelize over K to utilize threads
                    if (m < omp_get_num_threads())
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
                                    float val = std::abs(a_row[k_blk * 32 + j]);
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
                                    float val = a_row[k_blk * 32 + j];
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
                                    float val = std::abs(a_row[k_blk * 32 + j]);
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
                                    float val = a_row[k_blk * 32 + j];
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
                    // L2 cache is typically 1MB.
                    // B block size = n_block * K * 1 byte.
                    // We want n_block * K <= 1MB.
                    // Use 768KB (0.75MB) to leave room for A, C and overhead
                    int max_n_block = 786432 / k;
                    // Align to 64 (kernel block size)
                    max_n_block = (max_n_block / 64) * 64;
                    if (max_n_block < 64)
                        max_n_block = 64;

                    // Adaptive block sizing
                    // We want enough tasks to saturate threads.
                    // Total tasks = (m_tasks) * (n_tasks)
                    // m_tasks = (m + 1) / 2
                    // n_tasks = n / n_task_block
                    int num_threads = omp_get_num_threads();
                    int target_tasks = num_threads * 4; // Heuristic: 4x oversubscription for load balancing
                    int m_tasks = (m + 1) / 2;
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
                    // L2 cache size estimate: 1MB. Use 90% to be safe.
                    const long long L2_CACHE_SIZE = 900 * 1024;
                    bool needs_k_tiling = ((long long)n_task_block * k > L2_CACHE_SIZE);

                    // Check if we have enough parallelism to avoid collapsing M
                    int num_n_tasks = (n + n_task_block - 1) / n_task_block;
                    bool enough_parallelism = (num_n_tasks >= omp_get_num_threads());

                    if (needs_k_tiling && enough_parallelism)
                    {
                        // K-tiling path: Parallelize N only, tile K inside to reuse B in L2
                        // Reduce tile size to 128 (256KB) to leave room for A and C in L2
                        const int k_tile_blocks = 128; // 128 * 32 = 4096 rows. 4096 * 64 = 256KB.

#pragma omp for schedule(dynamic)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            int n_end = std::min(n, n_task + n_task_block);

                            // Iterate over K tiles
                            for (int k_start = 0; k_start < k_blocks; k_start += k_tile_blocks)
                            {
                                int k_count = std::min(k_tile_blocks, k_blocks - k_start);

                                // Iterate over M (reuse B-tile for all M)
                                for (int i = 0; i < m; i += 2)
                                {
                                    int rows_left = m - i;
                                    int rows_to_process = (rows_left >= 2) ? 2 : 1;

                                    Q8_1Block *blocks = all_blocks + i * k_blocks + k_start;

                                    for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                    {
                                        // Calculate packed weights offset
                                        // Base offset for N-block + Offset for K-tile
                                        // Note: packed_weights_ is [N_blocks][K_rows][64]
                                        // So offset = (n_blk/64) * (K*64) + (k_start*32)*64
                                        size_t weights_offset = (size_t)(n_blk / 64) * (k * 64) + (size_t)k_start * 32 * 64;
                                        const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                        // Fix: Offset compensation and scales by k_start * N
                                        const int32_t *comp_ptr = packed_weights_.compensation.data() + (size_t)k_start * n + n_blk;
                                        const float *scales_ptr = packed_weights_.scales.data() + (size_t)k_start * n + n_blk;

                                        Q8_1GemmParams params;
                                        params.A = blocks;
                                        params.B_packed = b_ptr;
                                        params.comp = comp_ptr;
                                        params.scales = scales_ptr;
                                        params.C = C + i * n + n_blk;
                                        params.K_blocks = k_count;
                                        params.N = 64;
                                        params.ldc = n;
                                        params.bias = bias ? bias + n_blk : nullptr;
                                        params.mask = mask ? mask + i * n + n_blk : nullptr;

                                        bool is_last_k_tile = (k_start + k_count == k_blocks);
                                        bool current_do_softmax = do_softmax && is_last_k_tile;

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
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        // Standard path: Collapse M and N for maximum parallelism
#pragma omp for collapse(2) schedule(static)
                        for (int n_task = 0; n_task < n; n_task += n_task_block)
                        {
                            for (int i = 0; i < m; i += 2)
                            {
                                int rows_left = m - i;
                                int rows_to_process = (rows_left >= 2) ? 2 : 1;

                                Q8_1Block *blocks = all_blocks + i * k_blocks;

                                int n_end = std::min(n, n_task + n_task_block);
                                for (int n_blk = n_task; n_blk < n_end; n_blk += 64)
                                {
                                    // Calculate packed weights offset
                                    size_t weights_offset = (size_t)(n_blk / 64) * (k * 64);
                                    const int8_t *b_ptr = packed_weights_.packed_data.data() + weights_offset;

                                    const int32_t *comp_ptr = packed_weights_.compensation.data() + n_blk;
                                    const float *scales_ptr = packed_weights_.scales.data() + n_blk;

                                    Q8_1GemmParams params;
                                    params.A = blocks;
                                    params.B_packed = b_ptr;
                                    params.comp = comp_ptr;
                                    params.scales = scales_ptr;
                                    params.C = C + i * n + n_blk;
                                    params.K_blocks = k_blocks;
                                    params.N = 64;
                                    params.ldc = n;
                                    params.bias = bias ? bias + n_blk : nullptr;
                                    params.mask = mask ? mask + i * n + n_blk : nullptr;

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
                                }
                            }
                        }
                    }
                }

                return true;
            }

            bool multiply_activations(const float *A, const float *B, float *C, int m, int n, int k, bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                return false;
            }

            bool multiply_activations_strided(const float *A, const float *B, float *C, int m, int n, int k, int stride_a, int stride_b, int stride_c, bool accumulate, float alpha, float beta, const MPIContext *ctx, int device_idx) override
            {
                return false;
            }

        private:
            const Q8_1Tensor *weights_;
            Q8_1PackedWeights packed_weights_;
        };

    }
}
