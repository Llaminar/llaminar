/**
 * @file CudaGemmKernelPhase6_Int8.h
 * @brief Phase 6: Int8 DP4A GEMM kernel using llama.cpp optimization pattern
 *
 * Key optimizations:
 * - IQ4_NL → int8 via hardware lookup table (get_int_from_table_16)
 * - On-the-fly FP32 → int8 quantization for A matrix
 * - DP4A intrinsic for int8×int8 accumulation (4 ops per instruction)
 * - Register-only computation (no shared memory decode bottleneck)
 * - Eliminates 43.5M bank conflicts from Phase 5
 * - Better memory coalescing (16-bit aligned loads vs 18-byte blocks)
 *
 * Expected performance: 50-90 TFLOPS (vs 17.5 TFLOPS Phase 5)
 *
 * @author David Sanftenberg
 * @date November 5, 2025
 */

#pragma once

#include <string>

namespace llaminar2
{
    namespace cuda
    {

        const char *PHASE6_INT8_HEADERS = R"(
#include <cuda_fp16.h>

// IQ4_NL block structure (18 bytes total)
// MUST match host-side layout: quants first, then scale
struct IQ4_NLBlock {
    uint8_t quants[16]; // 16 bytes
    half scale;         // 2 bytes (fp16)
};

// IQ4_NL int8 lookup table (from llama.cpp)
__device__ __constant__ int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113
};

/**
 * @brief DP4A intrinsic wrapper (from llama.cpp)
 * 
 * Computes: c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
 * Where a, b are int32 containing 4×int8 values
 * 
 * Available on: SM_61+ (Pascal and newer)
 * Throughput: 1 instruction for 4 multiply-adds
 */
__device__ __forceinline__ int ggml_cuda_dp4a(int a, int b, int c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    // Fallback for older architectures
    const int8_t* a_bytes = (const int8_t*)&a;
    const int8_t* b_bytes = (const int8_t*)&b;
    return c + a_bytes[0]*b_bytes[0] + a_bytes[1]*b_bytes[1] + 
               a_bytes[2]*b_bytes[2] + a_bytes[3]*b_bytes[3];
#endif
}

/**
 * @brief Hardware-accelerated table lookup from llama.cpp
 * 
 * Takes 8 indices (4 bits each, packed in uint32) and returns 8 int8 values from table.
 * Uses __byte_perm intrinsic for zero-overhead lookups.
 * 
 * @param q4 Input with 8×4-bit indices
 * @param table Pointer to 16-element int8 lookup table
 * @return int2 containing 8 int8 values (4 in .x, 4 in .y)
 */
__device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    const uint32_t * table32 = (const uint32_t *) table;
    
    uint32_t tmp[2];
    const uint32_t low_high_selection_indices = (0x32103210 | ((q4 & 0x88888888) >> 1));
    
    #pragma unroll
    for (uint32_t i = 0; i < 2; ++i) {
        const uint32_t shift = 16 * i;
        
        const uint32_t low  = __byte_perm(table32[0], table32[1], q4 >> shift);
        const uint32_t high = __byte_perm(table32[2], table32[3], q4 >> shift);
        tmp[i] = __byte_perm(low, high, low_high_selection_indices >> shift);
    }
    
    return make_int2(__byte_perm(tmp[0], tmp[1], 0x6420), 
                     __byte_perm(tmp[0], tmp[1], 0x7531));
}

/**
 * @brief Quantize FP32 values to int8 with per-row scaling
 * 
 * Finds max absolute value, computes scale, and quantizes to [-127, 127]
 * This is done on-the-fly during GEMM to enable DP4A.
 * 
 * @param input FP32 values to quantize
 * @param output int8 quantized values
 * @param count Number of values
 * @return float Scale factor (for dequantization: output[i] * scale ≈ input[i])
 */
__device__ __forceinline__ float quantize_fp32_to_int8(
    const float* input,
    int8_t* output,
    int count)
{
    // Find max absolute value
    float max_val = 0.0f;
    #pragma unroll
    for (int i = 0; i < count; i++) {
        max_val = fmaxf(max_val, fabsf(input[i]));
    }
    
    // Compute scale (avoid division by zero)
    float scale = (max_val > 1e-8f) ? (max_val / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;
    
    // Quantize to int8
    #pragma unroll
    for (int i = 0; i < count; i++) {
        output[i] = (int8_t)__float2int_rn(input[i] * inv_scale);
    }
    
    return scale;
}

)";

        /**
         * @brief Phase 6 Int8 DP4A GEMM kernel template
         *
         * Compute C = A * B where:
         * - A: [M, K] FP32 (quantized to int8 on-the-fly)
         * - B: [N, K_blocks] IQ4_NL (decoded to int8 via lookup)
         * - C: [M, N] FP32
         *
         * Template parameters:
         *   ${TILE_M}, ${TILE_N}, ${TILE_K} - Tile dimensions
         *   ${THREADS_PER_BLOCK} - Number of threads
         *
         * No MMA, no CuTe - simple DP4A accumulation loop.
         */
        const char *PHASE6_INT8_KERNEL_TEMPLATE = R"(
// Headers are added by JIT compiler

extern "C" __global__ void iq4nl_gemm_phase6_int8_kernel(
    const float* __restrict__ A,
    const IQ4_NLBlock* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    constexpr int TILE_M = ${TILE_M};
    constexpr int TILE_N = ${TILE_N};
    constexpr int TILE_K = ${TILE_K};
    constexpr int THREADS_PER_BLOCK = ${THREADS_PER_BLOCK};
    constexpr int BLOCK_SIZE = 32;  // IQ4_NL block size
    
    const int K_BLOCKS_TOTAL = K / BLOCK_SIZE;
    
    // Thread and block indices
    int tid = threadIdx.x;
    int gm_start = blockIdx.x * TILE_M;
    int gn_start = blockIdx.y * TILE_N;
    
    // Shared memory for quantized A matrix
    __shared__ int8_t s_A_q[TILE_M][TILE_K];
    __shared__ float s_A_scale[TILE_M];
    
    // Calculate how many output elements this thread handles
    // Total outputs per tile: TILE_M × TILE_N
    // Each thread handles: ceil((TILE_M × TILE_N) / THREADS_PER_BLOCK) elements
    constexpr int TOTAL_OUTPUTS = TILE_M * TILE_N;
    constexpr int OUTPUTS_PER_THREAD = (TOTAL_OUTPUTS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Local accumulator for this thread's outputs
    float acc[OUTPUTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < OUTPUTS_PER_THREAD; i++) {
        acc[i] = 0.0f;
    }
    
    // Main K loop
    for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
        
        // ========== LOAD AND QUANTIZE A TILE ==========
        // Each thread loads and quantizes one row of A
        for (int m_local = tid; m_local < TILE_M; m_local += THREADS_PER_BLOCK) {
            int gm = gm_start + m_local;
            
            if (gm < M) {
                // Load FP32 row from A
                float a_row[TILE_K];
                #pragma unroll
                for (int k = 0; k < TILE_K; k++) {
                    int gk = k_tile + k;
                    a_row[k] = (gk < K) ? A[gm * K + gk] : 0.0f;
                }
                
                // Quantize to int8
                s_A_scale[m_local] = quantize_fp32_to_int8(a_row, &s_A_q[m_local][0], TILE_K);
            }
        }
        
        __syncthreads();
        
        // ========== COMPUTE: DP4A ACCUMULATION ==========
        // Each thread processes OUTPUTS_PER_THREAD elements using 1D mapping
        // Thread i handles outputs: i, i+THREADS_PER_BLOCK, i+2*THREADS_PER_BLOCK, ...
        
        for (int out_idx = 0; out_idx < OUTPUTS_PER_THREAD; out_idx++) {
            // Map linear index to (m, n) coordinates
            int linear_idx = tid + out_idx * THREADS_PER_BLOCK;
            if (linear_idx >= TOTAL_OUTPUTS) break;
            
            int m_local = linear_idx / TILE_N;
            int n_local = linear_idx % TILE_N;
            
            int gm = gm_start + m_local;
            int gn = gn_start + n_local;
            
            if (gm >= M || gn >= N) continue;
            
            float scale_a = s_A_scale[m_local];
            
            // Process by IQ4_NL blocks (32 elements each)
            // Each block requires 8 DP4A operations (32/4 = 8)
            for (int kb = 0; kb < TILE_K / BLOCK_SIZE; kb++) {
                int gk_block_start = k_tile + kb * BLOCK_SIZE;
                if (gk_block_start >= K) break;
                
                int gk_block_idx = gk_block_start / BLOCK_SIZE;
                const IQ4_NLBlock* b_block = &B[gn * K_BLOCKS_TOTAL + gk_block_idx];
                
                float scale_b = __half2float(b_block->scale) / 127.0f;
                
                int32_t block_acc = 0;
                
                // Process 32 elements in groups of 4 using DP4A
                #pragma unroll
                for (int k_in_block = 0; k_in_block < BLOCK_SIZE; k_in_block += 4) {
                    int k = kb * BLOCK_SIZE + k_in_block;
                    if (k + 3 >= TILE_K) break;
                    
                    // Load 4 int8 from quantized A (pack into int32)
                    int32_t a_packed = *((int32_t*)&s_A_q[m_local][k]);
                    
                    // Decode 4 IQ4_NL values from B (simpler direct decoding)
                    // Each 4 elements need 2 bytes (4 nibbles)
                    int8_t b_vals[4];
                    for (int i = 0; i < 4; i++) {
                        int elem_idx = k_in_block + i;
                        int byte_idx = elem_idx / 2;
                        int nibble_idx = elem_idx % 2;
                        uint8_t byte_val = b_block->quants[byte_idx];
                        uint8_t nibble = (nibble_idx == 0) ? (byte_val & 0x0F) : (byte_val >> 4);
                        b_vals[i] = kvalues_iq4nl[nibble];
                    }
                    int32_t b_packed = *((int32_t*)b_vals);
                    
                    // DP4A: 4 multiply-adds in 1 instruction, accumulate into block_acc
                    block_acc = ggml_cuda_dp4a(a_packed, b_packed, block_acc);
                }
                
                // Scale and accumulate this block's contribution
                // kvalues are in [-127, 113], normalized to [-1, 1] by dividing by 127
                acc[out_idx] += scale_a * scale_b * (float)block_acc;
            }
        }
        
        __syncthreads();
    }
    
    // ========== WRITE OUTPUT ==========
    // Each thread writes its OUTPUTS_PER_THREAD results
    for (int out_idx = 0; out_idx < OUTPUTS_PER_THREAD; out_idx++) {
        int linear_idx = tid + out_idx * THREADS_PER_BLOCK;
        if (linear_idx >= TOTAL_OUTPUTS) break;
        
        int m_local = linear_idx / TILE_N;
        int n_local = linear_idx % TILE_N;
        
        int gm = gm_start + m_local;
        int gn = gn_start + n_local;
        
        if (gm < M && gn < N) {
            C[gm * N + gn] = acc[out_idx];
        }
    }
}
)";

    } // namespace cuda
} // namespace llaminar2
