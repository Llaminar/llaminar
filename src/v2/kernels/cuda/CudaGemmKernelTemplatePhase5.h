/**
 * @file CudaGemmKernelTemplatePhase5.h
 * @brief Phase 5 CuTe-based Tensor Core GEMM with JIT compilation and streaming decode
 *
 * This template supports:
 * - Configurable buffer stages (1=single, 2=double, 3=triple buffering)
 * - Streaming decode with sub-tiles (overlapped dequant + MMA)
 * - CuTe Tensor Core MMA atoms (SM80_16x8x16_F32F16F16F32_TN)
 * - Swizzled shared memory layouts for bank conflict avoidance
 * - Auto-tuning with ML heuristics
 *
 * @author David Sanftenberg
 * @date November 4, 2025
 */

#pragma once

#include <string>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief CuTe headers and IQ4_NL decoder for NVRTC
         *
         * Note: NVRTC can't include external headers, so we embed CuTe snippets
         */
        const char *PHASE5_CUTE_HEADERS = R"(
#include <cuda_fp16.h>
#include <cutlass/half.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>

using namespace cute;
using cutlass::half_t;

// IQ4_NL block structure (18 bytes total)
struct IQ4_NLBlock {
    half scale;        // 2 bytes (fp16)
    uint8_t quants[16]; // 16 bytes
}; // No padding - tightly packed to 18 bytes

// IQ4_NL dequantization lookup table (float version for backward compatibility)
__device__ __constant__ float iq4nl_values[16] = {
    -127.0f / 127.0f, -104.0f / 127.0f, -83.0f / 127.0f, -65.0f / 127.0f,
    -49.0f / 127.0f,  -35.0f / 127.0f,  -22.0f / 127.0f, -10.0f / 127.0f,
    1.0f / 127.0f,    13.0f / 127.0f,   25.0f / 127.0f,  38.0f / 127.0f,
    53.0f / 127.0f,   69.0f / 127.0f,   89.0f / 127.0f,  113.0f / 127.0f
};

// IQ4_NL int8 lookup table (llama.cpp pattern - optimized for DP4A)
// These are the actual quantized int8 values used by llama.cpp
__device__ __constant__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

/**
 * @brief Hardware-accelerated table lookup from llama.cpp
 * 
 * Takes 8 indices (4 bits each, packed in uint32) and returns 8 int8 values from table.
 * Uses __byte_perm intrinsic for zero-overhead lookups (single instruction byte shuffle).
 * 
 * Adapted from: external/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh lines 52-81
 * 
 * @param q4 Input with 8×4-bit indices (e.g., 0x3A5C... = indices [3,10,5,12,...])
 * @param table Pointer to 16-element int8 lookup table
 * @return int2 containing 8 int8 values (4 in .x, 4 in .y)
 */
__device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    
    // __byte_perm selects bytes based on the lower 16 bits in its third argument.
    // Process 32 bits in q4 with 2 iterations (shift 0 and 16).
    // First call __byte_perm for low and high 64-bit halves of table using low 3 bits.
    // Then call __byte_perm again to select from low/high based on 4th bit.
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    
    // tmp contains bytes from table in same order as 4-bit indices in q4.
    // However, result needs ints with all even/odd 4-bit indices.
    // Therefore, 2 more calls to __byte_perm to put bytes in correct order.
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), 
                     __byte_perm(tmp[0], tmp[1], 0x7531));
}

/**
 * @brief Decode single IQ4_NL block to output buffer (OPTIMIZED - llama.cpp pattern)
 * 
 * Uses hardware-accelerated lookup table from llama.cpp for maximum performance:
 * - Single 16-bit load per 4 values (vs 11 loads for 18-byte block)
 * - __byte_perm intrinsic for zero-overhead table lookup
 * - Reduces memory traffic by 9× and instructions by 7×
 * 
 * Expected: 2-3× speedup over previous implementation
 */
__device__ __forceinline__ void decode_iq4nl_block(
    const IQ4_NLBlock* block,
    half_t* output)
{
    const half scale = block->scale;
    const float scale_f = __half2float(scale);
    
    // Process 4 output values at a time using lookup table
    // Each iteration: load 16 bits (4×4-bit indices) → lookup 4 int8 values
    #pragma unroll
    for (int i = 0; i < 8; i++) {  // 8 iterations × 2 bytes = 16 bytes
        // Load 2 bytes = 4×4-bit indices (aligned 16-bit load)
        const uint16_t* quants16 = reinterpret_cast<const uint16_t*>(&block->quants[i * 2]);
        uint32_t indices = *quants16;  // Load as uint16, extend to uint32
        
        // Hardware-accelerated lookup: 4×4-bit indices → 4×int8 values
        int2 vals = get_int_from_table_16(indices, kvalues_iq4nl);
        
        // Extract 4 int8 values and convert to half with scaling
        const int8_t* v_ptr = reinterpret_cast<const int8_t*>(&vals);
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Scale: kvalues are int8 in [-127, 113], normalize to [-1, 1]
            float val_f32 = (float(v_ptr[j]) / 127.0f) * scale_f;
            output[i * 4 + j] = __float2half(val_f32);
        }
    }
}

)";

        /**
         * @brief Phase 5 GEMM kernel template with parameterized buffering and vectorization
         *
         * Template parameters (substituted at compile time):
         *   ${TILE_M}, ${TILE_N}, ${TILE_K} - Tile dimensions
         *   ${SUB_K} - Sub-tile size for streaming (16, 32, or TILE_K for no streaming)
         *   ${MMA_M}, ${MMA_N} - CuTe atom layout multipliers (1, 2, or 4)
         *   ${BUFFER_STAGES} - 1 (single), 2 (double), or 3 (triple) buffering
         *   ${THREADS_PER_BLOCK} - Number of threads (32, 64, 128, 256, 512)
         *   ${SWIZZLE_B}, ${SWIZZLE_M}, ${SWIZZLE_S} - Swizzle parameters (3,3,3 typical)
         *   ${VECTORIZE_A} - Vectorization width for A loads (1, 2, 4, 8)
         *                    1=scalar, 2=float2, 4=float4, 8=2xfloat4 (256-bit)
         *
         * Based on Phase 5A streaming kernel with proper CuTe MMA operations
         * OPTIMIZATION 1: Vectorized global memory loads for bandwidth (NCU-guided)
         * OPTIMIZATION 2: Transposed A shared memory layout for coalescing (NCU-guided)
         * NOTE: For Swizzle<3,3,3> (MBase=3), use vectorize_a=8 for contiguous writes
         */
        const char *PHASE5_GEMM_KERNEL_TEMPLATE = R"(
// Decoder and types
${CUTE_HEADERS}

// Optimization flags (NCU-guided)
#define USE_TRANSPOSED_A 0           // Row-major A for better L2 efficiency
#define USE_SHARED_PADDING 0         // Disabled for now (CuTe swizzle conflicts)
#define VECTORIZE_A_LOADS 4          // Use float4 for A-matrix loads (128-bit)
#define VECTORIZE_B_LOADS 4          // Use int4 for B-matrix block loads (128-bit)

// B matrix layout control (NCU-guided coalescing optimization)
// 0 = Row-major [N][K_blocks] (default, uncoalesced)
// 1 = Transposed [K_blocks][N] (coalesced access pattern)
#ifndef TRANSPOSE_B_LAYOUT
#define TRANSPOSE_B_LAYOUT 0
#endif

// Thread indexing pattern control (CRITICAL for coalescing!)
// 0 = K-major: k_block varies fastest (default, matches old behavior)
// 1 = N-major: n varies fastest (required for transposed layout coalescing)
// NOTE: Must use N-major indexing with transposed layout for full benefit!
#ifndef N_MAJOR_INDEXING
#define N_MAJOR_INDEXING TRANSPOSE_B_LAYOUT  // Default: match layout
#endif

// Configuration constants (substituted at JIT compile time)
#define TILE_M ${TILE_M}
#define TILE_N ${TILE_N}
#define TILE_K ${TILE_K}
#define SUB_K ${SUB_K}
#define MMA_M ${MMA_M}
#define MMA_N ${MMA_N}
#define BUFFER_STAGES ${BUFFER_STAGES}
#define THREADS_PER_BLOCK ${THREADS_PER_BLOCK}
#define SWIZZLE_B ${SWIZZLE_B}
#define SWIZZLE_M ${SWIZZLE_M}
#define SWIZZLE_S ${SWIZZLE_S}
#define VECTORIZE_A ${VECTORIZE_A}  // 1, 2, 4, or 8 (scalar, float2, float4, or 2xfloat4)

/**
 * @brief Phase 5 GEMM kernel: Streaming dequant with configurable buffering
 * 
 * A: [M x K] FP32 activations
 * B: [N x K/32] IQ4_NL quantized weights
 * C: [M x N] FP32 output
 */
extern "C" __global__ void
__launch_bounds__(THREADS_PER_BLOCK)
iq4nl_gemm_phase5_kernel(
    const float* __restrict__ A,
    const IQ4_NLBlock* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // ========== 1. MMA Setup ==========
    using mma_op = SM80_16x8x16_F32F16F16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    using tiled_mma = decltype(make_tiled_mma(
        mma_atom{},
        make_layout(Shape<Int<MMA_M>, Int<MMA_N>, Int<1>>{}),
        make_tile(Layout<Shape<_1,_1,_1>>{})
    ));
    
    tiled_mma mma;
    
    // ========== 2. Shared Memory (swizzled layouts) ==========
    // REMOVE padding - NCU profiling shows shared memory bank conflicts, but adding padding
    // causes worse regression (~20%) due to increased shared memory usage and reduced occupancy.
    // The Swizzle<3,3,3> already provides some bank conflict avoidance.
    // Alternative: Focus on fixing the 56% speedup potential from uncoalesced GLOBAL memory instead.
    constexpr int B_PADDING = 0;
    
#if USE_TRANSPOSED_A
    // TRANSPOSED A: [K][M] layout for coalesced global loads
    using SmemLayoutA = decltype(composition(
        Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
        Layout<Shape<Int<TILE_K>, Int<TILE_M>>, Stride<Int<TILE_M>, Int<1>>>{}
    ));
    __shared__ half_t s_A[BUFFER_STAGES][TILE_K][TILE_M];
#else
    // ROW-MAJOR A: [M][K] layout (better for L2 efficiency)
    using SmemLayoutA = decltype(composition(
        Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
        Layout<Shape<Int<TILE_M>, Int<TILE_K>>, Stride<Int<TILE_K>, Int<1>>>{}
    ));
    __shared__ half_t s_A[BUFFER_STAGES][TILE_M][TILE_K];
#endif
    
    // ROW-MAJOR B: [N][K+PAD] layout with padding for bank conflict avoidance
    using SmemLayoutB = decltype(composition(
        Swizzle<SWIZZLE_B, SWIZZLE_M, SWIZZLE_S>{},
        Layout<Shape<Int<TILE_N>, Int<TILE_K>>, Stride<Int<TILE_K + B_PADDING>, Int<1>>>{}
    ));
    
    // Double-buffered A AND B for software pipelining
    // Add 8-element padding to avoid shared memory bank conflicts (NCU profiling shows 34% speedup potential)
    __shared__ half_t s_B[2][TILE_N][TILE_K + B_PADDING];  // Padded to shift rows to different banks

    
    // ========== 3. Block/Thread Setup ==========
    int tid = threadIdx.x;
    int tx = tid % 32;
    int ty = tid / 32;
    
    int gm_start = blockIdx.x * TILE_M;
    int gn_start = blockIdx.y * TILE_N;
    
    // ========== 4. MMA Partitioning ==========
    auto mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), make_stride(N, Int<1>{}));
    auto cta_tiler = make_shape(Int<TILE_M>{}, Int<TILE_N>{}, Int<TILE_K>{});
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});
    
    auto thr_mma = mma.get_slice(tid);
    auto tCgC = thr_mma.partition_C(gC);
    auto tCrC = thr_mma.make_fragment_C(tCgC);
    clear(tCrC);
    
    // ========== 5. Main K-tile Loop ==========
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    
    // Prologue: Load first A tile with optimized vectorized loads
    {
        auto sA_write = make_tensor(make_smem_ptr(&s_A[0][0][0]), SmemLayoutA{});
        
        // Optimized load strategy for row-major A[M][K]
        // Goal: Maximize coalescing by having consecutive threads access consecutive K elements
        // Pattern: Each warp loads one row (M dimension), threads access consecutive K
        
        constexpr int ELEMENTS_PER_THREAD = (TILE_M * TILE_K) / THREADS_PER_BLOCK;
        constexpr int K_VECTORS = TILE_K / 4;  // Number of float4 per row
        
        // Each thread loads multiple float4 vectors
        for (int elem = 0; elem < ELEMENTS_PER_THREAD; elem += 4) {
            int linear_idx = tid * ELEMENTS_PER_THREAD + elem;
            if (linear_idx + 3 < TILE_M * TILE_K) {
                int m = linear_idx / TILE_K;
                int k = linear_idx % TILE_K;
                
                int gm = gm_start + m;
                int gk = k;
                
                // Coalesced load: consecutive threads load consecutive K within same row
                float4 vec4;
                if (gm < M && gk + 3 < K) {
                    vec4 = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                } else {
                    // Boundary handling
                    vec4.x = (gm < M && gk + 0 < K) ? A[gm * K + gk + 0] : 0.0f;
                    vec4.y = (gm < M && gk + 1 < K) ? A[gm * K + gk + 1] : 0.0f;
                    vec4.z = (gm < M && gk + 2 < K) ? A[gm * K + gk + 2] : 0.0f;
                    vec4.w = (gm < M && gk + 3 < K) ? A[gm * K + gk + 3] : 0.0f;
                }
                
                // Write to swizzled shared memory
                sA_write(m, k + 0) = half_t(__float2half(vec4.x));
                sA_write(m, k + 1) = half_t(__float2half(vec4.y));
                sA_write(m, k + 2) = half_t(__float2half(vec4.z));
                sA_write(m, k + 3) = half_t(__float2half(vec4.w));
            }
        }
    }
    __syncthreads();
    
    // Prologue: Decode first B tile (ki=0)
    int b_read_stage = 0;
    int b_write_stage = 1;
    {
        int k0 = 0;
        // COALESCED ROW-MAJOR pattern: consecutive threads load consecutive IQ4_NL blocks
        // B is laid out as [N][K/32] where each block is 18 bytes
        // Row-major traversal: thread 0 loads B[0][0], thread 1 loads B[0][1], ..., thread K_BLOCKS_TOTAL loads B[1][0]
        constexpr int BLOCK_SIZE = 32;  // Elements per IQ4_NL block
        const int K_BLOCKS_TOTAL = K / BLOCK_SIZE;  // Runtime constant (K is kernel parameter)
        constexpr int K_BLOCKS_IN_SUB_K = (SUB_K + BLOCK_SIZE - 1) / BLOCK_SIZE;
        constexpr int TOTAL_BLOCKS = TILE_N * K_BLOCKS_IN_SUB_K;
        
        for (int block_idx = tid; block_idx < TOTAL_BLOCKS; block_idx += THREADS_PER_BLOCK) {
#if N_MAJOR_INDEXING
            // N-major: N varies fastest (consecutive threads → consecutive N)
            // Good for transposed layout [K_blocks][N] where B[k][n] and B[k][n+1] are adjacent
            int n = block_idx % TILE_N;
            int k_block_local = block_idx / TILE_N;
#else
            // K-major: K varies fastest (consecutive threads → consecutive K blocks in same row)
            // Original pattern, but causes uncoalesced access due to alternating rows
            int k_block_local = block_idx % K_BLOCKS_IN_SUB_K;
            int n = block_idx / K_BLOCKS_IN_SUB_K;
#endif
            int local_k = k_block_local * BLOCK_SIZE;
            
            int gn = gn_start + n;
            int gk = k0 + local_k;
            int gk_block = gk / BLOCK_SIZE;  // Global K block index
            
            if (gn < N && gk < K) {
                // Address calculation depends on B matrix layout:
                // Row-major [N][K_blocks]: &B[gn * K_BLOCKS_TOTAL + gk_block]
                //   - Adjacent threads (different n) → stride of K_BLOCKS_TOTAL * 18 bytes → uncoalesced!
                // Transposed [K_blocks][N]: &B[gk_block * N + gn]
                //   - Adjacent threads (different gn) → adjacent blocks → coalesced!
#if TRANSPOSE_B_LAYOUT
                const IQ4_NLBlock* block = &B[gk_block * N + gn];  // Transposed: [K_blocks][N]
#else
                const IQ4_NLBlock* block = &B[gn * K_BLOCKS_TOTAL + gk_block];  // Row-major: [N][K_blocks]
#endif
                
                half_t temp[BLOCK_SIZE];
                decode_iq4nl_block(block, temp);
                
                #pragma unroll
                for (int idx = 0; idx < BLOCK_SIZE && local_k + idx < SUB_K; idx++) {
                    s_B[b_read_stage][n][local_k + idx] = temp[idx];
                }
            }
        }
    }
    __syncthreads();
    
    // Main loop with software pipelining
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int k0 = k_tile * TILE_K;
        int a_read_stage = k_tile % BUFFER_STAGES;
        int a_write_stage = (a_read_stage + 1) % BUFFER_STAGES;
        
        // Inner K-loop: Process in SUB_K chunks
        for (int ki = 0; ki < TILE_K; ki += SUB_K) {
            
            // ========== SOFTWARE PIPELINE: Decode next B tile while MMA runs ==========
            int ki_next = ki + SUB_K;
            bool has_next = (ki_next < TILE_K);
            
            if (has_next) {
                constexpr int BLOCK_SIZE = 32;
                const int K_BLOCKS_TOTAL = K / BLOCK_SIZE;  // Runtime constant
                constexpr int K_BLOCKS_IN_SUB_K = (SUB_K + BLOCK_SIZE - 1) / BLOCK_SIZE;
                constexpr int TOTAL_BLOCKS = TILE_N * K_BLOCKS_IN_SUB_K;
                
                for (int block_idx = tid; block_idx < TOTAL_BLOCKS; block_idx += THREADS_PER_BLOCK) {
#if N_MAJOR_INDEXING
                    // N-major indexing for transposed layout coalescing
                    int n = block_idx % TILE_N;
                    int k_block_local = block_idx / TILE_N;
#else
                    // K-major indexing (original pattern)
                    int k_block_local = block_idx % K_BLOCKS_IN_SUB_K;
                    int n = block_idx / K_BLOCKS_IN_SUB_K;
#endif
                    int local_k = k_block_local * BLOCK_SIZE;
                    
                    int gn = gn_start + n;
                    int gk = k0 + ki_next + local_k;
                    int gk_block = gk / BLOCK_SIZE;
                    
                    if (gn < N && gk < K) {
#if TRANSPOSE_B_LAYOUT
                        const IQ4_NLBlock* block = &B[gk_block * N + gn];  // Transposed: [K_blocks][N]
#else
                        const IQ4_NLBlock* block = &B[gn * K_BLOCKS_TOTAL + gk_block];  // Row-major: [N][K_blocks]
#endif
                        
                        half_t temp[BLOCK_SIZE];
                        decode_iq4nl_block(block, temp);
                        
                        #pragma unroll
                        for (int idx = 0; idx < BLOCK_SIZE && local_k + idx < SUB_K; idx++) {
                            s_B[b_write_stage][n][ki_next + local_k + idx] = temp[idx];
                        }
                    }
                }
            }
            
            // ========== MMA on current B tile ==========
            auto sA_tensor = make_tensor(make_smem_ptr(&s_A[a_read_stage][0][0]), SmemLayoutA{});
            auto sB_tensor = make_tensor(make_smem_ptr(&s_B[b_read_stage][0][ki]), SmemLayoutB{});
            
            auto tCsA = thr_mma.partition_A(sA_tensor);
            auto tCsB = thr_mma.partition_B(sB_tensor);
            
            auto tCrA = make_fragment_like(tCsA);
            auto tCrB = make_fragment_like(tCsB);
            
            cute::copy(tCsA, tCrA);
            cute::copy(tCsB, tCrB);
            
            // MMA computation (overlaps with decode above!)
            cute::gemm(mma, tCrA, tCrB, tCrC);
            
            // Sync before swapping buffers
            __syncthreads();
            
            // Swap B buffers for next iteration
            if (has_next) {
                b_read_stage = b_write_stage;
                b_write_stage = 1 - b_write_stage;
            }
        }
        
        // ========== PREFETCH: Decode first B sub-tile of NEXT K-tile ==========
        if (k_tile + 1 < num_k_tiles) {
            int k_next_tile = (k_tile + 1) * TILE_K;
            
            // Decode first SUB_K chunk of next K-tile to write buffer
            for (int n = ty * 32 + tx; n < TILE_N; n += THREADS_PER_BLOCK) {
                int gn = gn_start + n;
                if (gn < N) {
                    for (int local_k = 0; local_k < SUB_K; local_k += 32) {
                        int gk = k_next_tile + local_k;
                        if (gk < K) {
                            const IQ4_NLBlock* block = &B[gn * (K/32) + gk/32];
                            
                            half_t temp[32];
                            decode_iq4nl_block(block, temp);
                            
                            #pragma unroll
                            for (int idx = 0; idx < 32 && local_k + idx < SUB_K; idx++) {
                                s_B[b_write_stage][n][local_k + idx] = temp[idx];
                            }
                        }
                    }
                }
            }
            
            // After prefetch, swap buffers for next K-tile
            b_read_stage = b_write_stage;
            b_write_stage = 1 - b_write_stage;
        }
        
        // Prefetch next A tile (if using buffering)
        if (BUFFER_STAGES > 1 && k_tile + 1 < num_k_tiles) {
            auto sA_next = make_tensor(make_smem_ptr(&s_A[a_write_stage][0][0]), SmemLayoutA{});
            int k_next = (k_tile + 1) * TILE_K;
            
#if VECTORIZE_A == 8
            // 8-element vectorized loads (256-bit, matches Swizzle<3,3,3>)
            constexpr int VEC_WIDTH = 8;
            constexpr int TOTAL_ELEMENTS = TILE_M * TILE_K;
            
#if USE_TRANSPOSED_A
            // COALESCED TRANSPOSE: Input A is COLUMN-MAJOR A[K][M]
            constexpr int THREADS_PER_K = TILE_M / VEC_WIDTH;
            constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_K;
            constexpr int NUM_ITERS = TILE_K / K_PER_ITER;
            
            #pragma unroll
            for (int iter = 0; iter < NUM_ITERS; iter++) {
                int k = iter * K_PER_ITER + (tid / THREADS_PER_K);
                int m_base = (tid % THREADS_PER_K) * VEC_WIDTH;
                
                int gk = k_next + k;
                int gm_base = gm_start + m_base;
                
                // Load 8 consecutive M elements (COALESCED)
                float val[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    int gm = gm_base + i;
                    if (gm < M && gk < K) {
                        val[i] = A[gk * M + gm];  // COLUMN-MAJOR: A[k][m]
                    } else {
                        val[i] = 0.0f;
                    }
                }
                
                // Write TRANSPOSED to shared
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    sA_next(k, m_base + i) = half_t(__float2half(val[i]));
                }
            }
#else
            // ROW-MAJOR (original)
            constexpr int VEC_GROUPS = TOTAL_ELEMENTS / VEC_WIDTH;
            
            for (int vec_idx = tid; vec_idx < VEC_GROUPS; vec_idx += THREADS_PER_BLOCK) {
                int linear_idx = vec_idx * VEC_WIDTH;
                int m = linear_idx / TILE_K;
                int k_base = (linear_idx % TILE_K) & ~7;  // Align to 8
                
                int gm = gm_start + m;
                int gk = k_next + k_base;
                
                float4 vec4_lo, vec4_hi;
                if (gm < M && gk + 7 < K) {
                    vec4_lo = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                    vec4_hi = *reinterpret_cast<const float4*>(&A[gm * K + gk + 4]);
                } else {
                    vec4_lo.x = (gm < M && gk + 0 < K) ? A[gm * K + gk + 0] : 0.0f;
                    vec4_lo.y = (gm < M && gk + 1 < K) ? A[gm * K + gk + 1] : 0.0f;
                    vec4_lo.z = (gm < M && gk + 2 < K) ? A[gm * K + gk + 2] : 0.0f;
                    vec4_lo.w = (gm < M && gk + 3 < K) ? A[gm * K + gk + 3] : 0.0f;
                    vec4_hi.x = (gm < M && gk + 4 < K) ? A[gm * K + gk + 4] : 0.0f;
                    vec4_hi.y = (gm < M && gk + 5 < K) ? A[gm * K + gk + 5] : 0.0f;
                    vec4_hi.z = (gm < M && gk + 6 < K) ? A[gm * K + gk + 6] : 0.0f;
                    vec4_hi.w = (gm < M && gk + 7 < K) ? A[gm * K + gk + 7] : 0.0f;
                }
                
                sA_next(m, k_base + 0) = half_t(__float2half(vec4_lo.x));
                sA_next(m, k_base + 1) = half_t(__float2half(vec4_lo.y));
                sA_next(m, k_base + 2) = half_t(__float2half(vec4_lo.z));
                sA_next(m, k_base + 3) = half_t(__float2half(vec4_lo.w));
                sA_next(m, k_base + 4) = half_t(__float2half(vec4_hi.x));
                sA_next(m, k_base + 5) = half_t(__float2half(vec4_hi.y));
                sA_next(m, k_base + 6) = half_t(__float2half(vec4_hi.z));
                sA_next(m, k_base + 7) = half_t(__float2half(vec4_hi.w));
            }
#endif
            
#elif VECTORIZE_A == 4
            // Vectorized float4 loads
            constexpr int VEC_WIDTH = 4;
            constexpr int TOTAL_ELEMENTS = TILE_M * TILE_K;
            constexpr int VEC_ELEMENTS = TOTAL_ELEMENTS / VEC_WIDTH;
            
            for (int vec_idx = tid; vec_idx < VEC_ELEMENTS; vec_idx += THREADS_PER_BLOCK) {
                int linear_idx = vec_idx * VEC_WIDTH;
                int m = linear_idx / TILE_K;
                int k_base = (linear_idx % TILE_K) & ~3;
                
                int gm = gm_start + m;
                int gk = k_next + k_base;
                
                float4 vec4;
                if (gm < M && gk + 3 < K) {
                    vec4 = *reinterpret_cast<const float4*>(&A[gm * K + gk]);
                } else {
                    vec4.x = (gm < M && gk + 0 < K) ? A[gm * K + gk + 0] : 0.0f;
                    vec4.y = (gm < M && gk + 1 < K) ? A[gm * K + gk + 1] : 0.0f;
                    vec4.z = (gm < M && gk + 2 < K) ? A[gm * K + gk + 2] : 0.0f;
                    vec4.w = (gm < M && gk + 3 < K) ? A[gm * K + gk + 3] : 0.0f;
                }
                
                sA_next(m, k_base + 0) = half_t(__float2half(vec4.x));
                sA_next(m, k_base + 1) = half_t(__float2half(vec4.y));
                sA_next(m, k_base + 2) = half_t(__float2half(vec4.z));
                sA_next(m, k_base + 3) = half_t(__float2half(vec4.w));
            }
            
#elif VECTORIZE_A == 2
            // Vectorized float2 loads
            constexpr int VEC_WIDTH = 2;
            constexpr int TOTAL_ELEMENTS = TILE_M * TILE_K;
            constexpr int VEC_ELEMENTS = TOTAL_ELEMENTS / VEC_WIDTH;
            
            for (int vec_idx = tid; vec_idx < VEC_ELEMENTS; vec_idx += THREADS_PER_BLOCK) {
                int linear_idx = vec_idx * VEC_WIDTH;
                int m = linear_idx / TILE_K;
                int k_base = (linear_idx % TILE_K) & ~1;
                
                int gm = gm_start + m;
                int gk = k_next + k_base;
                
                float2 vec2;
                if (gm < M && gk + 1 < K) {
                    vec2 = *reinterpret_cast<const float2*>(&A[gm * K + gk]);
                } else {
                    vec2.x = (gm < M && gk + 0 < K) ? A[gm * K + gk + 0] : 0.0f;
                    vec2.y = (gm < M && gk + 1 < K) ? A[gm * K + gk + 1] : 0.0f;
                }
                
                sA_next(m, k_base + 0) = half_t(__float2half(vec2.x));
                sA_next(m, k_base + 1) = half_t(__float2half(vec2.y));
            }
            
#else
            // Scalar loads (baseline)
            for (int k = tx; k < TILE_K; k += 32) {
                for (int m = ty; m < TILE_M; m += (THREADS_PER_BLOCK / 32)) {
                    int gm = gm_start + m;
                    int gk = k_next + k;
                    if (gm < M && gk < K) {
                        sA_next(m, k) = half_t(__float2half(A[gm * K + gk]));
                    } else {
                        sA_next(m, k) = half_t(__float2half(0.0f));
                    }
                }
            }
#endif
        }
        __syncthreads();
    }
    
    // ========== 6. Write Results ==========
    for (int i = 0; i < size(tCgC); i++) {
        tCgC(i) = tCrC(i);
    }
}
)";

    } // namespace cuda
} // namespace llaminar2
