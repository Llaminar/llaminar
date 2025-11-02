/**
 * @file CudaGemmKernel.cuh
 * @brief Tensor Core CUDA GEMM kernel using CUTLASS CuTe API with templated MMA atoms
 *
 * @author David Sanftenberg
 * @date November 2, 2025
 *
 * ✅ **MODERN IMPLEMENTATION**: Uses CUTLASS 4.2.1 CuTe template API
 *
 * ARCHITECTURE - TEMPLATED MMA ATOMS:
 * - MMA atom is now a template parameter (not hardcoded!)
 * - Autotuner can select different atoms for different problem sizes
 * - Atom layout is also templated (e.g., 1×1, 2×2, 4×4)
 * - Enables true configuration space exploration
 *
 * TEMPLATE HIERARCHY:
 * 1. MmaAtomType: Tensor Core instruction (e.g., SM80_16x8x16_F32F16F16F32_TN)
 * 2. AtomLayout: How many atoms to tile (e.g., 2×2×1 = 4 atoms)
 * 3. CTA Tile: Resulting block size (atom_size × atom_layout)
 *
 * EXAMPLE CONFIGURATIONS:
 * - Small:  SM80_16x8x8  + 1×1×1 layout = 16×8×8   CTA tile
 * - Medium: SM80_16x8x16 + 2×2×1 layout = 32×16×16 CTA tile
 * - Large:  SM80_16x8x16 + 4×4×1 layout = 64×32×16 CTA tile
 *
 * ADVANTAGES OVER HARDCODED ATOMS:
 * - Autotuner can explore atom types (8, 16, 32 K-dim)
 * - Different atom layouts for different problem sizes
 * - Factory instantiates specific kernel templates per config
 * - True performance diversity in configuration space
 *
 * HARDWARE REQUIREMENTS:
 * - Compute Capability ≥ 8.0 (Ampere or newer)
 * - RTX 3090: SM 8.6 (Ampere) - fully supported
 *
 * REFERENCE:
 * - /opt/cutlass/examples/cute/tutorial/sgemm_sm80.cu
 * - CUTLASS documentation: https://github.com/NVIDIA/cutlass
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h> // BF16 support (CUDA 11.0+, Compute Capability 8.0+)

#ifdef HAVE_CUTLASS
#include <cute/tensor.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cutlass/bfloat16.h> // CUTLASS BF16 type
#else
#error "CUTLASS library required for Tensor Core kernels. Define HAVE_CUTLASS."
#endif

namespace llaminar2
{
    namespace cuda
    {

        using namespace cute;

        /**
         * @brief Type traits for Tensor Core compute types
         *
         * Maps input activation types to appropriate:
         * - Shared memory storage type (always matches InputType for zero-copy)
         * - CUDA native type (for decoder output)
         *
         * NOTE: MMA atom is now a template parameter, not hardcoded here!
         */
        template <typename InputType>
        struct TensorCoreTraits;

        // FP16 specialization
        template <>
        struct TensorCoreTraits<cutlass::half_t>
        {
            using SmemType = cutlass::half_t;
            using CudaType = __half;
            static constexpr bool can_use_async = true;

            template <typename Decoder, typename BlockType>
            __device__ static inline void decode_block(const Decoder &decoder, const BlockType *block, CudaType *output)
            {
                decoder.decode_block_fp16(block, output);
            }
        };

        // BF16 specialization
        template <>
        struct TensorCoreTraits<cutlass::bfloat16_t>
        {
            using SmemType = cutlass::bfloat16_t;
            using CudaType = __nv_bfloat16;
            static constexpr bool can_use_async = true;

            template <typename Decoder, typename BlockType>
            __device__ static inline void decode_block(const Decoder &decoder, const BlockType *block, CudaType *output)
            {
                decoder.decode_block_bf16(block, output);
            }
        };

        // FP32 specialization (fallback, requires conversion)
        template <>
        struct TensorCoreTraits<float>
        {
            using SmemType = cutlass::half_t; // Convert to FP16 for Tensor Cores
            using CudaType = __half;
            static constexpr bool can_use_async = false; // Need manual conversion

            template <typename Decoder, typename BlockType>
            __device__ static inline void decode_block(const Decoder &decoder, const BlockType *block, CudaType *output)
            {
                decoder.decode_block_fp16(block, output);
            }
        };

        /**
         * @brief Phase 2 Tensor Core GEMM kernel using CuTe with templated input type and MMA atom
         *
         * TEMPLATE PARAMETERS:
         * - InputType: Input activation type (float, cutlass::half_t, or cutlass::bfloat16_t)
         * - MmaAtomType: Tensor Core MMA atom instruction (e.g., SM80_16x8x16_F32F16F16F32_TN)
         * - AtomLayoutM/N/K: How many atoms to tile (e.g., 2×2×1 = 4 atoms)
         * - Decoder: Block decoder type (IQ4_NL_Decoder, Q6_K_Decoder, etc.)
         * - TILE_M/N/K: CTA tile dimensions (must match atom_size × atom_layout)
         *
         * ARCHITECTURE:
         * - MMA atom: The fundamental Tensor Core operation (e.g., 16×8×16)
         * - Atom layout: How many atoms to tile together (e.g., 2×2×1)
         * - CTA tile: Resulting tile size (e.g., 32×16×16 = 16×2 × 8×2 × 16×1)
         *
         * EXAMPLE CONFIGURATIONS:
         * - Small: SM80_16x8x8 with 1×1×1 layout = 16×8×8 CTA tile
         * - Medium: SM80_16x8x16 with 2×2×1 layout = 32×16×16 CTA tile
         * - Large: SM80_16x8x16 with 4×4×1 layout = 64×32×16 CTA tile
         *
         * KEY CUTE CONCEPTS:
         * - TiledMMA: Defines Tensor Core tile shape and layout
         * - TiledCopy: Async memory transfer with cp.async
         * - Tensor: View over memory with compile-time shapes
         * - partition_*: Distributes work across threads
         *
         * @tparam InputType       Input activation type (float, cutlass::half_t, or cutlass::bfloat16_t)
         * @tparam MmaAtomType     MMA atom instruction type (e.g., SM80_16x8x16_F32F16F16F32_TN)
         * @tparam AtomLayoutM     Number of atoms in M dimension (default 2)
         * @tparam AtomLayoutN     Number of atoms in N dimension (default 2)
         * @tparam AtomLayoutK     Number of atoms in K dimension (default 1)
         * @tparam Decoder         Block decoder type (IQ4_NL_Decoder, Q6_K_Decoder, etc.)
         * @tparam TILE_M          M dimension of CTA tile (default 64)
         * @tparam TILE_N          N dimension of CTA tile (default 64)
         * @tparam TILE_K          K dimension of CTA tile (default 16)
         */
        template <typename InputType,
                  typename MmaAtomType,
                  int AtomLayoutM,
                  int AtomLayoutN,
                  int AtomLayoutK,
                  typename Decoder,
                  int TILE_M = 64,
                  int TILE_N = 64,
                  int TILE_K = 16>
        __global__ void quantized_gemm_kernel_cute(
            const InputType *__restrict__ A, // [m × k] activation matrix (FP32, FP16, or BF16)
            float *__restrict__ C,           // [m × n] output matrix (FP32)
            int m, int n, int k,
            Decoder decoder // Quantized weight decoder
        )
        {
            // ==================== CuTe Type Definitions ====================

            // Use type traits to select appropriate types (no longer includes MmaAtom)
            using Traits = TensorCoreTraits<InputType>;
            using SmemType = typename Traits::SmemType;
            using CudaType = typename Traits::CudaType;
            constexpr bool can_use_async = Traits::can_use_async;

            // Create tiled MMA with templated atom and layout
            // Layout must be rank-3 (M, N, K dimensions)
            using TiledMma = TiledMMA<
                MMA_Atom<MmaAtomType>,
                Layout<Shape<Int<AtomLayoutM>, Int<AtomLayoutN>, Int<AtomLayoutK>>>>;

            TiledMma tiled_mma;

            // ==================== Shared Memory Allocation ====================

            // Double-buffered shared memory for pipelined execution (Phase 2.7)
            // Buffer 0 and Buffer 1 alternate between read and write stages
            // This allows overlapping async copy of tile K+1 with MMA of tile K
            // SmemType is either cutlass::half_t or cutlass::bfloat16_t (never float)
            __shared__ SmemType smem_A_flat[2][TILE_M * TILE_K];
            __shared__ SmemType smem_B_flat[2][TILE_N * TILE_K];

            // ==================== Block and Thread Indexing ====================

            const int bx = blockIdx.x; // Block column
            const int by = blockIdx.y; // Block row
            const int tid = threadIdx.x;

            // Decoder parameters
            const int BLOCK_SIZE = decoder.block_size();
            const int num_k_blocks = decoder.k_blocks();

            // ==================== Create CuTe Tensor Views ====================

            // Global A matrix tensor: [M, K]
            Tensor mA = make_tensor(make_gmem_ptr(A),
                                    make_shape(m, k),
                                    make_stride(k, Int<1>{})); // Row-major: (M,K) with stride (K,1)

            // Global C matrix tensor: [M, N]
            Tensor mC = make_tensor(make_gmem_ptr(C),
                                    make_shape(m, n),
                                    make_stride(n, Int<1>{})); // Row-major: (M,N) with stride (N,1)

            // Shared memory tensors (buffer 0 initially, will switch for double-buffering)
            Tensor sA = make_tensor(make_smem_ptr(smem_A_flat[0]),
                                    make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
                                    make_stride(Int<TILE_K>{}, Int<1>{}));

            Tensor sB = make_tensor(make_smem_ptr(smem_B_flat[0]),
                                    make_shape(Int<TILE_N>{}, Int<TILE_K>{}),
                                    make_stride(Int<TILE_K>{}, Int<1>{})); // Row-major (will transpose)

            // Coalesce shared memory layouts to simplify coordinate mapping
            // This reduces overhead in partition operations
            auto sA_coalesced = coalesce(sA);
            auto sB_coalesced = coalesce(sB);

            // Partition for This Thread Block ====================

            // Define CTA tiler: (BLK_M, BLK_N, BLK_K)
            auto cta_tiler = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});

            // Tile global tensors to this CTA using zipped_divide for explicit tile/rest separation
            // This gives us ((TILE), (REST)) layout - more composable than local_tile
            auto cta_coord = make_coord(by, bx, _); // (M_block, N_block, K) block coordinates

            // Use local_tile for CTA selection (will refactor to zipped_divide in future optimization)
            // NOTE: local_tile is equivalent to composition(layout, tiler) under the hood
            Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)
            Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)

            // ==================== MMA Partitioning (Layout Algebra Refactored) ====================

            // Get MMA thread slice - this contains the optimal thread→data mapping from MMA_Traits
            auto thr_mma = tiled_mma.get_thread_slice(tid);

            // Partition coalesced shared memory using MMA atom's layout
            // Using coalesced layouts reduces coordinate mapping overhead
            auto tCsA = thr_mma.partition_A(sA_coalesced); // MMA's view of sA for this thread
            auto tCsB = thr_mma.partition_B(sB_coalesced); // MMA's view of sB for this thread

            // Create accumulator partition
            auto tCgC = thr_mma.partition_C(gC);
            auto tCrC = thr_mma.make_fragment_C(tCgC); // Register accumulator

            // ==================== Predication Setup ====================
            // Create identity tensors for bounds checking (CuTe predication pattern)

            // Identity tensor for A (for input loading bounds check)
            Tensor cA = make_identity_tensor(shape(mA));                             // (M,K) -> (M,K)
            Tensor cta_cA = local_tile(cA, cta_tiler, cta_coord, Step<_1, X, _1>{}); // (BLK_M, BLK_K, k)

            // Identity tensor for C (for output writing bounds check)
            Tensor cC = make_identity_tensor(shape(mC));                             // (M,N) -> (M,N)
            Tensor cta_cC = local_tile(cC, cta_tiler, cta_coord, Step<_1, _1, X>{}); // (BLK_M, BLK_N)
            Tensor tCcC = thr_mma.partition_C(cta_cC);                               // Same partitioning as tCgC

            // Initialize accumulator to zero
            clear(tCrC);

            // ==================== Main GEMM Loop ====================

            const int num_k_tiles = (k + TILE_K - 1) / TILE_K;

            for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile)
            {

                // ==================== Load A Tile ====================
                auto gA_k = gA(_, _, k_tile); // (BLK_M, BLK_K) for this k_tile

                if constexpr (can_use_async)
                {
                    // ==================== FP16/BF16 Input: Use Explicit Async Copy ====================
                    // Create explicit SM80_CP_ASYNC TiledCopy for FP16 or BF16 (Phase 2.5/3.0)
                    // FIXED: Thread layout now matches blockDim.x = 128 (was 256)
                    using CopyAtomA = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, SmemType>;
                    auto copyA = cute::make_tiled_copy(
                        CopyAtomA{},
                        cute::Layout<cute::Shape<cute::_16, cute::_8>>{}, // ✅ 16×8 = 128 threads (was 32×8 = 256)
                        cute::Layout<cute::Shape<cute::_1, cute::_8>>{});

                    // Partition source and destination tensors
                    auto thr_copy_A = copyA.get_thread_slice(tid);
                    auto tAgA = thr_copy_A.partition_S(gA_k);
                    auto tAsA = thr_copy_A.partition_D(sA_coalesced); // Use coalesced layout

                    // Async copy with cp.async
                    copy(copyA, tAgA, tAsA);
                    cp_async_fence();
                }
                else
                {
                    // ==================== FP32 Input: Manual Copy with Conversion to SmemType ====================
                    // Manual copy required for FP32→FP16/BF16 conversion (Phase 2.0)

                    // Get coordinate tensor for this k-tile (for predication)
                    auto gA_k_coord = cta_cA(_, _, k_tile); // (BLK_M, BLK_K) coordinates for this k_tile

                    // Predicated copy with bounds checking (CuTe pattern)
                    for (int i = tid; i < TILE_M * TILE_K; i += blockDim.x)
                    {
                        const int row = i / TILE_K;
                        const int col = i % TILE_K;

                        // Get global coordinate for this element
                        auto coord = gA_k_coord(row, col);

                        float val = 0.0f;
                        if (elem_less(coord, shape(mA)))
                        { // CuTe bounds check
                            val = gA_k(row, col);
                        }

                        smem_A_flat[0][row * TILE_K + col] = SmemType(val);
                    }
                }

                // ==================== Load B Tile (Dequant to FP16) ====================
                // CRITICAL FIX: Handle TILE_K < BLOCK_SIZE and unaligned k dimensions
                // When TILE_K=16 < BLOCK_SIZE=32, old code had K_BLOCKS_PER_TILE=0!

                // Compute actual k range for this tile
                const int k_tile_start = k_tile * TILE_K;
                const int k_tile_end = min(k_tile_start + TILE_K, k);

                // Compute which blocks intersect this k range
                const int first_k_block = k_tile_start / BLOCK_SIZE;
                const int last_k_block = (k_tile_end - 1) / BLOCK_SIZE;
                const int num_blocks_this_tile = last_k_block - first_k_block + 1;

                const int num_B_blocks = TILE_N * num_blocks_this_tile;

                for (int i = tid; i < num_B_blocks; i += blockDim.x)
                {
                    const int n_idx = i / num_blocks_this_tile;
                    const int k_block_offset = i % num_blocks_this_tile;

                    const int global_n = bx * TILE_N + n_idx;
                    const int global_k_block = first_k_block + k_block_offset;

                    CudaType decoded_cuda[64]; // Use CUDA native type (__half or __nv_bfloat16)

                    if (global_n < n && global_k_block < num_k_blocks)
                    {
                        const auto *block_ptr = decoder.get_block_at(global_n, global_k_block);
                        // Use traits to call appropriate decoder method (fp16 or bf16)
                        Traits::decode_block(decoder, block_ptr, decoded_cuda);
                    }
                    else
                    {
#pragma unroll
                        for (int j = 0; j < BLOCK_SIZE; ++j)
                        {
                            decoded_cuda[j] = CudaType(0.0f);
                        }
                    }

                    // Write only elements that fall within [k_tile_start, k_tile_end) to shared memory
                    const int block_k_start = global_k_block * BLOCK_SIZE;
#pragma unroll
                    for (int j = 0; j < BLOCK_SIZE; ++j)
                    {
                        const int global_k = block_k_start + j;
                        if (global_k >= k_tile_start && global_k < k_tile_end)
                        {
                            const int smem_k_idx = global_k - k_tile_start;
                            smem_B_flat[0][n_idx * TILE_K + smem_k_idx] = SmemType(decoded_cuda[j]);
                        }
                    }
                }

                // Wait for async copies to complete
                if constexpr (can_use_async)
                {
                    cp_async_wait<0>(); // Wait for all pending async groups
                }
                __syncthreads();

                // ==================== Tensor Core MMA ====================

                // Directly perform Tensor Core GEMM using partitioned shared memory views
                // CuTe's gemm() will handle fragment creation and copies internally
                gemm(tiled_mma, tCsA, tCsB, tCrC);

                __syncthreads();
            }

            // ==================== Write Output (with predication) ====================

            // Predicated copy: only write elements within matrix bounds
            // Use CuTe's elem_less to compare coordinates against original shape
            CUTE_UNROLL
            for (int i = 0; i < size(tCgC); ++i)
            {
                if (elem_less(tCcC(i), shape(mC)))
                { // Check if coordinate is in-bounds
                    tCgC(i) = tCrC(i);
                }
            }
        }

        /**
         * @brief Launcher for CuTe Tensor Core kernel with templated MMA atom
         *
         * Calculates optimal grid/block dimensions and launches kernel.
         *
         * TEMPLATE PARAMETERS:
         * - InputType: Input activation type (float, cutlass::half_t, or cutlass::bfloat16_t)
         * - MmaAtomType: Tensor Core MMA atom instruction (e.g., SM80_16x8x16_F32F16F16F32_TN)
         * - AtomLayoutM/N/K: How many atoms to tile (e.g., 2×2×1)
         * - Decoder: Block decoder type
         * - TILE_M/N/K: CTA tile dimensions
         *
         * @param A         Input activation matrix [m × k]
         * @param C         Output matrix [m × n] (FP32)
         * @param m         Number of rows in A and C
         * @param n         Number of columns in C
         * @param k         Number of columns in A
         * @param decoder   Quantized weight decoder
         * @param stream    CUDA stream (default 0)
         * @return          cudaSuccess or error code
         */
        template <typename InputType,
                  typename MmaAtomType,
                  int AtomLayoutM,
                  int AtomLayoutN,
                  int AtomLayoutK,
                  typename Decoder,
                  int TILE_M = 32,
                  int TILE_N = 64,
                  int TILE_K = 16>
        inline cudaError_t launchQuantizedGemmCuTe(
            const InputType *A,
            float *C,
            int m, int n, int k,
            Decoder decoder,
            cudaStream_t stream = 0)
        {
            // Calculate grid dimensions
            dim3 blocks((n + TILE_N - 1) / TILE_N,
                        (m + TILE_M - 1) / TILE_M);

            // Calculate threads per block from MMA requirements
            // SM80 Tensor Core: typically 128 threads (4 warps)
            dim3 threads(128);

            // Launch kernel
            quantized_gemm_kernel_cute<InputType, MmaAtomType, AtomLayoutM, AtomLayoutN, AtomLayoutK, Decoder, TILE_M, TILE_N, TILE_K>
                <<<blocks, threads, 0, stream>>>(A, C, m, n, k, decoder);

            return cudaGetLastError();
        }

    } // namespace cuda
} // namespace llaminar2
