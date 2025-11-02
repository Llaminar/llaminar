/**
 * @file CudaGemmVariantsBaseline.cu
 * @brief Baseline CUDA GEMM kernel variants (standard implementation)
 *
 * Implements parameterized kernels across the configuration space:
 * - Multiple tile sizes (16x16 to 128x128)
 * - Register tiling strategies (1x1 to 8x8 work per thread)
 * - Memory optimizations (prefetch, transpose, vectorization)
 * - Generic quantized weight decoding via IBlockDecoder interface
 *
 * Each variant is instantiated as a separate kernel function for optimal
 * performance (no runtime branching on configuration parameters).
 *
 * Design: Follows CPU GemmKernelTemplate pattern with compile-time polymorphism.
 * The decoder type is a template parameter, eliminating vtable overhead while
 * allowing format-specific dequantization (IQ4_NL, Q6_K, Q8_0, etc.).
 *
 * Performance: ~3,000 GFLOPS baseline (single-token decode on A100).
 *
 * @author David Sanftenberg
 * @date October 31, 2025
 */

#include "CudaGemmVariantsBaseline.h"
#include "CudaGemmConfig.h"
#include "IQ4_NL_BlockDecoder.h" // IQ4_NL block structure and decoder
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief Parameterized GEMM kernel template with fused dequantization
         *
         * Generic GEMM kernel that works with any block-quantized format.
         * The Decoder type provides format-specific dequantization via:
         *   decoder.decode_block(block_ptr, output_buffer)
         *
         * Template parameters control kernel configuration at compile time,
         * eliminating runtime branching overhead.
         *
         * @tparam Decoder Block decoder type (e.g., IQ4_NL_Decoder<IQ4_NLBlock>)
         * @tparam TILE_M Output tile rows (must equal THREADS_M * WORK_M)
         * @tparam TILE_N Output tile columns (must equal THREADS_N * WORK_N)
         * @tparam TILE_K K-dimension tile size (must be multiple of decoder block size)
         * @tparam THREADS_M Threads in M dimension
         * @tparam THREADS_N Threads in N dimension
         * @tparam WORK_M Elements per thread in M dimension
         * @tparam WORK_N Elements per thread in N dimension
         * @tparam PREFETCH_STAGES 0=no prefetch, 1=double-buffer, 2=triple-buffer
         * @tparam TRANSPOSE_SMEM Transpose shared memory layout to reduce bank conflicts
         * @tparam VECTORIZE_LOAD Use vectorized (float4) loads when aligned
         */
        template <
            typename Decoder,
            int TILE_M, int TILE_N, int TILE_K,
            int THREADS_M, int THREADS_N,
            int WORK_M, int WORK_N,
            int PREFETCH_STAGES,
            bool TRANSPOSE_SMEM,
            int VECTORIZE_LOAD // ← Fixed: was bool, now int (1, 2, or 4 for float/float2/float4)
            >
        __global__ void quantized_gemm_kernel_variant(
            const float *__restrict__ A,
            float *__restrict__ C,
            int m, int n, int k,
            Decoder decoder // Decoder passed by value (small struct with pointer)
        )
        {
            // Static assertions to catch configuration errors at compile time
            static_assert(TILE_M == THREADS_M * WORK_M, "TILE_M must equal THREADS_M * WORK_M");
            static_assert(TILE_N == THREADS_N * WORK_N, "TILE_N must equal THREADS_N * WORK_N");
            // Note: TILE_K divisibility check moved to runtime (decoder.block_size() is not constexpr)
            static_assert(THREADS_M * THREADS_N <= 1024, "Thread block size exceeds limit");
            static_assert(PREFETCH_STAGES >= 0 && PREFETCH_STAGES <= 2, "PREFETCH_STAGES must be 0-2");

            // Shared memory allocation (sized for prefetch stages)
            constexpr int NUM_BUFFERS = 1 + PREFETCH_STAGES;
            __shared__ float s_A[NUM_BUFFERS][TILE_M][TILE_K];
            __shared__ float s_B[NUM_BUFFERS][TILE_N][TILE_K];

            // Thread position within block
            const int tid_m = threadIdx.y; // 0 to THREADS_M-1
            const int tid_n = threadIdx.x; // 0 to THREADS_N-1
            const int tid = tid_m * THREADS_N + tid_n;

            // Output tile position
            const int block_row = blockIdx.y * TILE_M;
            const int block_col = blockIdx.x * TILE_N;

            // Decoder parameters
            const int BLOCK_SIZE = decoder.block_size();
            const int num_k_blocks = decoder.k_blocks();

            // Register accumulators (WORK_M × WORK_N elements)
            float acc[WORK_M][WORK_N];
#pragma unroll
            for (int wm = 0; wm < WORK_M; ++wm)
            {
#pragma unroll
                for (int wn = 0; wn < WORK_N; ++wn)
                {
                    acc[wm][wn] = 0.0f;
                }
            }

            // Number of K tiles
            const int num_k_tiles = (k + TILE_K - 1) / TILE_K;

            // Main GEMM loop over K dimension
            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
            {
                const int k_offset = k_tile * TILE_K;
                const int buffer_idx = PREFETCH_STAGES > 0 ? (k_tile % NUM_BUFFERS) : 0;

                // ==================== Load A tile ====================
                // Each thread loads WORK_M elements in M, cycling through K
                // Total: THREADS_M * THREADS_N threads load TILE_M * TILE_K elements

                constexpr int TOTAL_A_ELEMENTS = TILE_M * TILE_K;
                constexpr int TOTAL_THREADS = THREADS_M * THREADS_N;
                constexpr int A_LOADS_PER_THREAD = (TOTAL_A_ELEMENTS + TOTAL_THREADS - 1) / TOTAL_THREADS; // Ceiling division
#pragma unroll
                for (int load_idx = 0; load_idx < A_LOADS_PER_THREAD; ++load_idx)
                {
                    const int flat_idx = tid * A_LOADS_PER_THREAD + load_idx;
                    if (flat_idx >= TOTAL_A_ELEMENTS)
                        break; // Guard against excess iterations

                    const int a_row = flat_idx / TILE_K;
                    const int a_col = flat_idx % TILE_K;

                    const int global_row = block_row + a_row;
                    const int global_col = k_offset + a_col;

                    float val = 0.0f;
                    if (global_row < m && global_col < k)
                    {
                        val = A[global_row * k + global_col];
                    }

                    if constexpr (TRANSPOSE_SMEM)
                    {
                        s_A[buffer_idx][a_row][a_col] = val;
                    }
                    else
                    {
                        s_A[buffer_idx][a_row][a_col] = val;
                    }
                }

                // ==================== Load and decode B tile ====================
                // CRITICAL FIX: Handle TILE_K < BLOCK_SIZE and unaligned k dimensions
                //
                // Problems addressed:
                // 1. TILE_K < BLOCK_SIZE: Need to load partial blocks
                // 2. k not aligned to TILE_K: Last tile has partial elements
                // 3. k not aligned to BLOCK_SIZE: Last block has partial elements
                //
                // Solution: Load blocks that intersect [k_offset, k_offset+TILE_K) range

                // Compute actual k range for this tile (may be less than TILE_K at the end)
                const int k_tile_start = k_offset;
                const int k_tile_end = min(k_offset + TILE_K, k);
                const int k_tile_size = k_tile_end - k_tile_start;

                // Compute which blocks intersect this k range
                const int first_k_block = k_tile_start / BLOCK_SIZE;
                const int last_k_block = (k_tile_end - 1) / BLOCK_SIZE; // Inclusive
                const int num_blocks_this_tile = last_k_block - first_k_block + 1;

                // Total blocks to load: TILE_N columns × num_blocks_this_tile
                const int TOTAL_B_BLOCKS = TILE_N * num_blocks_this_tile;
                // TOTAL_THREADS already defined earlier
                const int B_BLOCKS_PER_THREAD = (TOTAL_B_BLOCKS + TOTAL_THREADS - 1) / TOTAL_THREADS;

                for (int block_idx = 0; block_idx < B_BLOCKS_PER_THREAD; ++block_idx)
                {
                    const int flat_idx = tid * B_BLOCKS_PER_THREAD + block_idx;
                    if (flat_idx >= TOTAL_B_BLOCKS)
                        break;

                    const int b_row = flat_idx / num_blocks_this_tile;            // Which column of B
                    const int b_k_block_offset = flat_idx % num_blocks_this_tile; // Which block in this tile

                    const int global_col = block_col + b_row;
                    const int global_k_block = first_k_block + b_k_block_offset;

                    // Decode entire block
                    float decoded[64]; // Max block size
                    if (global_col < n && global_k_block < num_k_blocks)
                    {
                        const auto *block_ptr = decoder.get_block_at(global_col, global_k_block);
                        decoder.decode_block(block_ptr, decoded);
                    }
                    else
                    {
#pragma unroll
                        for (int i = 0; i < BLOCK_SIZE; ++i)
                            decoded[i] = 0.0f;
                    }

                    // Write elements that fall within [k_tile_start, k_tile_end) to shared memory
                    const int block_k_start = global_k_block * BLOCK_SIZE;
                    const int block_k_end = block_k_start + BLOCK_SIZE;

                    for (int block_elem = 0; block_elem < BLOCK_SIZE; ++block_elem)
                    {
                        const int global_k = block_k_start + block_elem;
                        // Only write if this element is within the current tile's k range
                        if (global_k >= k_tile_start && global_k < k_tile_end)
                        {
                            const int smem_k_idx = global_k - k_tile_start;
                            if constexpr (TRANSPOSE_SMEM)
                            {
                                s_B[buffer_idx][b_row][smem_k_idx] = decoded[block_elem];
                            }
                            else
                            {
                                s_B[buffer_idx][b_row][smem_k_idx] = decoded[block_elem];
                            }
                        }
                    }
                }

                __syncthreads();

                // ==================== Compute tile ====================
                // Each thread computes WORK_M × WORK_N output elements
                // Use actual k_tile_size (may be < TILE_K for last tile)

#pragma unroll
                for (int k_idx = 0; k_idx < k_tile_size; ++k_idx)
                {
                    // Load A fragment (WORK_M elements)
                    float a_frag[WORK_M];
#pragma unroll
                    for (int wm = 0; wm < WORK_M; ++wm)
                    {
                        const int a_row = tid_m * WORK_M + wm;
                        a_frag[wm] = s_A[buffer_idx][a_row][k_idx];
                    }

                    // Load B fragment (WORK_N elements)
                    float b_frag[WORK_N];
#pragma unroll
                    for (int wn = 0; wn < WORK_N; ++wn)
                    {
                        const int b_row = tid_n * WORK_N + wn;
                        b_frag[wn] = s_B[buffer_idx][b_row][k_idx];
                    }

// Outer product: acc += a_frag * b_frag^T
#pragma unroll
                    for (int wm = 0; wm < WORK_M; ++wm)
                    {
#pragma unroll
                        for (int wn = 0; wn < WORK_N; ++wn)
                        {
                            acc[wm][wn] += a_frag[wm] * b_frag[wn];
                        }
                    }
                }

                __syncthreads();
            }

// ==================== Write output ====================
#pragma unroll
            for (int wm = 0; wm < WORK_M; ++wm)
            {
#pragma unroll
                for (int wn = 0; wn < WORK_N; ++wn)
                {
                    const int out_row = block_row + tid_m * WORK_M + wm;
                    const int out_col = block_col + tid_n * WORK_N + wn;

                    if (out_row < m && out_col < n)
                    {
                        C[out_row * n + out_col] = acc[wm][wn];
                    }
                }
            }
        }

        /**
         * @brief Launch dispatcher for parameterized kernel variants (IQ4_NL)
         *
         * Instantiates and launches the appropriate kernel based on runtime configuration.
         * This is a convenience wrapper that creates an IQ4_NL_Decoder and calls the generic launcher.
         */
        cudaError_t launchIQ4NLGemmVariant(
            const float *A,
            const IQ4_NLBlock *B_blocks,
            float *C,
            int m, int n, int k,
            const CudaGemmConfig &config,
            cudaStream_t stream)
        {
            // Validate configuration
            if (!config.isValid())
            {
                return cudaErrorInvalidValue;
            }

            if (k % 32 != 0)
            {
                return cudaErrorInvalidValue;
            }

            // Create IQ4_NL decoder
            const int num_k_blocks = k / 32;
            IQ4_NL_Decoder<IQ4_NLBlock> decoder(B_blocks, n, num_k_blocks);

            // Compute grid dimensions
            dim3 blockDim(config.threads_n, config.threads_m);
            dim3 gridDim(
                (n + config.tile_n - 1) / config.tile_n,
                (m + config.tile_m - 1) / config.tile_m);

// Dispatch to appropriate kernel variant
// This macro reduces boilerplate for template instantiation
#define LAUNCH_VARIANT(TM, TN, TK, THM, THN, WM, WN, PREFETCH, TRANSPOSE, VEC) \
    if (config.tile_m == TM && config.tile_n == TN && config.tile_k == TK &&   \
        config.threads_m == THM && config.threads_n == THN &&                  \
        config.work_per_thread_m == WM && config.work_per_thread_n == WN &&    \
        config.prefetch_stages == PREFETCH &&                                  \
        config.transpose_smem == TRANSPOSE && config.vectorize_load == VEC)    \
    {                                                                          \
        quantized_gemm_kernel_variant<                                         \
            IQ4_NL_Decoder<IQ4_NLBlock>,                                       \
            TM, TN, TK, THM, THN, WM, WN, PREFETCH, TRANSPOSE, VEC>            \
            <<<gridDim, blockDim, 0, stream>>>(A, C, m, n, k, decoder);        \
        return cudaGetLastError();                                             \
    }

            // Instantiate common variants (expand as needed)
            // Format: TILE_M, TILE_N, TILE_K, THREADS_M, THREADS_N, WORK_M, WORK_N, PREFETCH, TRANSPOSE, VEC

            // Small tile variants (16x16 output)
            LAUNCH_VARIANT(16, 16, 32, 8, 8, 2, 2, 0, false, true);
            LAUNCH_VARIANT(16, 16, 32, 8, 8, 2, 2, 1, false, true);
            LAUNCH_VARIANT(16, 16, 64, 8, 8, 2, 2, 0, false, true);
            LAUNCH_VARIANT(16, 16, 64, 8, 8, 2, 2, 1, true, true);

            // Medium tile variants (32x32 output)
            LAUNCH_VARIANT(32, 32, 32, 8, 8, 4, 4, 0, false, true);
            LAUNCH_VARIANT(32, 32, 32, 8, 8, 4, 4, 1, true, true);
            LAUNCH_VARIANT(32, 32, 64, 8, 8, 4, 4, 1, true, true);
            LAUNCH_VARIANT(32, 32, 64, 8, 8, 4, 4, 2, true, true);
            LAUNCH_VARIANT(32, 32, 32, 16, 16, 2, 2, 1, true, true);

            // Large tile variants (64x64 output)
            LAUNCH_VARIANT(64, 64, 32, 16, 16, 4, 4, 1, true, true);
            LAUNCH_VARIANT(64, 64, 32, 16, 16, 4, 4, 2, true, true);
            // LAUNCH_VARIANT(64, 64, 64, 16, 16, 4, 4, 1, true, true);  // 64KB shared mem (exceeds sm_70 limit)
            // LAUNCH_VARIANT(64, 64, 64, 16, 16, 4, 4, 2, true, true);  // 96KB shared mem (exceeds sm_70 limit)
            LAUNCH_VARIANT(64, 64, 32, 8, 8, 8, 8, 1, true, true);

            // Tall tile variants (64x16 output)
            LAUNCH_VARIANT(64, 16, 32, 16, 8, 4, 2, 0, false, true);
            LAUNCH_VARIANT(64, 16, 32, 16, 8, 4, 2, 1, false, true);
            LAUNCH_VARIANT(64, 16, 64, 16, 8, 4, 2, 1, true, true);

            // Wide tile variants (16x64 output)
            LAUNCH_VARIANT(16, 64, 32, 8, 16, 2, 4, 0, false, true);
            LAUNCH_VARIANT(16, 64, 32, 8, 16, 2, 4, 1, true, true);
            LAUNCH_VARIANT(16, 64, 64, 8, 16, 2, 4, 1, true, true);

// XL tile variants (128x128 output) - disabled for sm_70 (exceeds 48KB shared memory)
// LAUNCH_VARIANT(128, 128, 32, 16, 16, 8, 8, 1, true, true);  // 64KB shared mem
// LAUNCH_VARIANT(128, 128, 32, 16, 16, 8, 8, 2, true, true);  // 96KB shared mem

// Generated variant dispatches (200 additional configurations)
#include "generated/CudaGemmVariants_00.inc"
#include "generated/CudaGemmVariants_01.inc"
#include "generated/CudaGemmVariants_02.inc"
#include "generated/CudaGemmVariants_03.inc"
#include "generated/CudaGemmVariants_04.inc"
#include "generated/CudaGemmVariants_05.inc"
#include "generated/CudaGemmVariants_06.inc"
#include "generated/CudaGemmVariants_07.inc"
#include "generated/CudaGemmVariants_08.inc"
#include "generated/CudaGemmVariants_09.inc"

#undef LAUNCH_VARIANT

            // If no variant matched, return error
            return cudaErrorInvalidConfiguration;
        }

    } // namespace cuda
} // namespace llaminar2
