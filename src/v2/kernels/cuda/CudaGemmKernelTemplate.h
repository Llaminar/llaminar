/**
 * @file CudaGemmKernelTemplate.h
 * @brief CUDA GEMM kernel template for JIT compilation via NVRTC
 *
 * This file contains the kernel source code as a string constant for runtime compilation.
 * The template parameters are substituted at runtime before compilation.
 */

#pragma once

#include <string>

namespace llaminar2
{
    namespace cuda
    {

        /**
         * @brief IQ4_NL decoder implementation (embedded in generated kernel)
         */
        const char *IQ4_NL_DECODER_SOURCE = R"(
// IQ4_NL quantization format structure
struct IQ4_NLBlock {
    uint8_t qs[16];     // 32 4-bit quantized values (packed 2 per byte)
    uint16_t d;         // FP16 scale factor (stored as raw bits)
};

// FP16 to FP32 conversion (manual implementation for NVRTC compatibility)
__device__ inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    uint32_t f;
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f = sign << 31;
        } else {
            // Subnormal
            exponent = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FF;
            f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Inf or NaN
        f = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normal
        f = (sign << 31) | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    
    return *reinterpret_cast<float*>(&f);
}

// IQ4_NL dequantization lookup table (global constant)
__device__ __constant__ float iq4nl_values[16] = {
    -127.f/128.f, -104.f/128.f, -83.f/128.f, -65.f/128.f,
    -49.f/128.f, -35.f/128.f, -22.f/128.f, -10.f/128.f,
    1.f/128.f, 13.f/128.f, 25.f/128.f, 38.f/128.f,
    53.f/128.f, 69.f/128.f, 89.f/128.f, 113.f/128.f
};

// Decoder class for IQ4_NL format
class IQ4_NL_Decoder {
private:
    const IQ4_NLBlock* blocks_;
    int n_;
    int k_blocks_;
    
public:
    __device__ IQ4_NL_Decoder(const IQ4_NLBlock* blocks, int n, int k_blocks)
        : blocks_(blocks), n_(n), k_blocks_(k_blocks) {}
    
    __device__ int block_size() const { return 32; }
    __device__ int k_blocks() const { return k_blocks_; }
    
    __device__ const IQ4_NLBlock* get_block_at(int col, int k_block) const {
        return &blocks_[col * k_blocks_ + k_block];
    }
    
    __device__ void decode_block(const IQ4_NLBlock* block, float* output) const {
        const float scale = fp16_to_fp32(block->d);
        
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            uint8_t packed = block->qs[i];
            uint8_t q0 = packed & 0xF;        // Lower 4 bits
            uint8_t q1 = (packed >> 4) & 0xF; // Upper 4 bits
            
            output[i * 2 + 0] = scale * iq4nl_values[q0];
            output[i * 2 + 1] = scale * iq4nl_values[q1];
        }
    }
};
)";

        /**
         * CUDA GEMM kernel template with placeholders for JIT compilation
         *
         * Placeholders (replaced at runtime):
         *   ${TILE_M}, ${TILE_N}, ${TILE_K}
         *   ${THREADS_M}, ${THREADS_N}
         *   ${WORK_M}, ${WORK_N}
         *   ${PREFETCH_STAGES}
         *   ${TRANSPOSE_SMEM}
         *   ${VECTORIZE_LOAD}
         */
        const char *GEMM_KERNEL_TEMPLATE = R"(
// JIT-compiled CUDA GEMM kernel for IQ4_NL quantization
// No external includes needed - NVRTC provides built-in CUDA support

// Basic type definitions
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;

// FP16 support - use native half precision type
// __half is automatically defined by NVRTC when compiling for device code
// __half2float is a built-in NVRTC intrinsic

// Decoder implementation (IQ4_NL)
${DECODER_SOURCE}

// Kernel implementation
// __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
// Hint to compiler: optimize for 4 blocks/SM to maximize occupancy
extern "C" __global__ void
__launch_bounds__(${THREADS_M} * ${THREADS_N}, 4)
quantized_gemm_kernel_iq4nl(
    const float* __restrict__ A,
    const IQ4_NLBlock* __restrict__ B_blocks,
    float* __restrict__ C,
    int m, int n, int k)
{
    // Configuration from template parameters
    constexpr int TILE_M = ${TILE_M};
    constexpr int TILE_N = ${TILE_N};
    constexpr int TILE_K = ${TILE_K};
    constexpr int THREADS_M = ${THREADS_M};
    constexpr int THREADS_N = ${THREADS_N};
    constexpr int WORK_M = ${WORK_M};
    constexpr int WORK_N = ${WORK_N};
    constexpr int PREFETCH_STAGES = ${PREFETCH_STAGES};
    constexpr bool TRANSPOSE_SMEM = ${TRANSPOSE_SMEM};
    constexpr int VECTORIZE_LOAD = ${VECTORIZE_LOAD};
    
    // Static assertions
    static_assert(TILE_M == THREADS_M * WORK_M, "TILE_M must equal THREADS_M * WORK_M");
    static_assert(TILE_N == THREADS_N * WORK_N, "TILE_N must equal THREADS_N * WORK_N");
    static_assert(THREADS_M * THREADS_N <= 1024, "Thread block size exceeds limit");
    static_assert(PREFETCH_STAGES >= 0 && PREFETCH_STAGES <= 2, "PREFETCH_STAGES must be 0-2");
    
    // Create decoder
    const int num_k_blocks = k / 32;
    IQ4_NL_Decoder decoder(B_blocks, n, num_k_blocks);
    const int BLOCK_SIZE = 32;
    
    // Shared memory with bank conflict padding
    constexpr int NUM_BUFFERS = 1 + PREFETCH_STAGES;
    constexpr int TILE_K_PADDED = TILE_K + 1;
    
    __shared__ float s_A[NUM_BUFFERS][TILE_M][TILE_K_PADDED];
    __shared__ float s_B[NUM_BUFFERS][TILE_N][TILE_K_PADDED];
    
    // Thread position
    const int tid_m = threadIdx.y;
    const int tid_n = threadIdx.x;
    const int tid = tid_m * THREADS_N + tid_n;
    
    // Output tile position
    const int block_row = blockIdx.y * TILE_M;
    const int block_col = blockIdx.x * TILE_N;
    
    // Register accumulators
    float acc[WORK_M][WORK_N];
    #pragma unroll
    for (int wm = 0; wm < WORK_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WORK_N; ++wn) {
            acc[wm][wn] = 0.0f;
        }
    }
    
    // Number of K tiles
    const int num_k_tiles = (k + TILE_K - 1) / TILE_K;
    
    // Main GEMM loop
    for (int k_tile = 0; k_tile < num_k_tiles; ++k_tile) {
        const int k_offset = k_tile * TILE_K;
        const int buffer_idx = PREFETCH_STAGES > 0 ? (k_tile % NUM_BUFFERS) : 0;
        
        // Load A tile
        constexpr int TOTAL_A_ELEMENTS = TILE_M * TILE_K;
        constexpr int TOTAL_THREADS = THREADS_M * THREADS_N;
        constexpr int A_LOADS_PER_THREAD = (TOTAL_A_ELEMENTS + TOTAL_THREADS - 1) / TOTAL_THREADS;
        
        #pragma unroll
        for (int load_idx = 0; load_idx < A_LOADS_PER_THREAD; ++load_idx) {
            const int flat_idx = tid * A_LOADS_PER_THREAD + load_idx;
            if (flat_idx >= TOTAL_A_ELEMENTS) break;
            
            const int a_row = flat_idx / TILE_K;
            const int a_col = flat_idx % TILE_K;
            const int global_row = block_row + a_row;
            const int global_col = k_offset + a_col;
            
            float val = 0.0f;
            if (global_row < m && global_col < k) {
                val = A[global_row * k + global_col];
            }
            s_A[buffer_idx][a_row][a_col] = val;
        }
        
        // Load and decode B tile
        const int k_tile_start = k_offset;
        const int k_tile_end = min(k_offset + TILE_K, k);
        const int k_tile_size = k_tile_end - k_tile_start;
        
        const int first_k_block = k_tile_start / BLOCK_SIZE;
        const int last_k_block = (k_tile_end - 1) / BLOCK_SIZE;
        const int num_blocks_this_tile = last_k_block - first_k_block + 1;
        const int TOTAL_B_BLOCKS = TILE_N * num_blocks_this_tile;
        const int B_BLOCKS_PER_THREAD = (TOTAL_B_BLOCKS + TOTAL_THREADS - 1) / TOTAL_THREADS;
        
        for (int block_idx = 0; block_idx < B_BLOCKS_PER_THREAD; ++block_idx) {
            const int flat_idx = tid * B_BLOCKS_PER_THREAD + block_idx;
            if (flat_idx >= TOTAL_B_BLOCKS) break;
            
            const int b_row = flat_idx / num_blocks_this_tile;
            const int b_k_block_offset = flat_idx % num_blocks_this_tile;
            const int global_col = block_col + b_row;
            const int global_k_block = first_k_block + b_k_block_offset;
            
            float decoded[64];
            if (global_col < n && global_k_block < num_k_blocks) {
                const IQ4_NLBlock* block_ptr = decoder.get_block_at(global_col, global_k_block);
                decoder.decode_block(block_ptr, decoded);
            } else {
                #pragma unroll
                for (int i = 0; i < BLOCK_SIZE; ++i) decoded[i] = 0.0f;
            }
            
            const int block_k_start = global_k_block * BLOCK_SIZE;
            for (int block_elem = 0; block_elem < BLOCK_SIZE; ++block_elem) {
                const int global_k = block_k_start + block_elem;
                if (global_k >= k_tile_start && global_k < k_tile_end) {
                    const int smem_k_idx = global_k - k_tile_start;
                    s_B[buffer_idx][b_row][smem_k_idx] = decoded[block_elem];
                }
            }
        }
        
        __syncthreads();
        
        // Compute tile
        #pragma unroll
        for (int k_idx = 0; k_idx < k_tile_size; ++k_idx) {
            float a_frag[WORK_M];
            #pragma unroll
            for (int wm = 0; wm < WORK_M; ++wm) {
                const int a_row = tid_m * WORK_M + wm;
                a_frag[wm] = s_A[buffer_idx][a_row][k_idx];
            }
            
            float b_frag[WORK_N];
            #pragma unroll
            for (int wn = 0; wn < WORK_N; ++wn) {
                const int b_row = tid_n * WORK_N + wn;
                b_frag[wn] = s_B[buffer_idx][b_row][k_idx];
            }
            
            #pragma unroll
            for (int wm = 0; wm < WORK_M; ++wm) {
                #pragma unroll
                for (int wn = 0; wn < WORK_N; ++wn) {
                    acc[wm][wn] += a_frag[wm] * b_frag[wn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write output
    #pragma unroll
    for (int wm = 0; wm < WORK_M; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WORK_N; ++wn) {
            const int out_row = block_row + tid_m * WORK_M + wm;
            const int out_col = block_col + tid_n * WORK_N + wn;
            if (out_row < m && out_col < n) {
                C[out_row * n + out_col] = acc[wm][wn];
            }
        }
    }
}
)";

    } // namespace cuda
} // namespace llaminar2
