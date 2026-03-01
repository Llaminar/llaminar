/**
 * @file ROCmGemvKernel_native_VNNI.hip
 * @brief Native-VNNI GEMV kernels for lossless Q4/IQ4 decode on AMD MI50/MI60 (gfx906)
 *
 * This file contains native-VNNI kernels that decode 4-bit quantized weights
 * (Q4_0, IQ4_NL) losslessly and compute GEMV using v_dot4_i32_i8 with
 * per-block FP16 scales. Future native-VNNI kernels for other bit widths
 * should also be placed here.
 *
 * Unlike ratio-VNNI (which bakes scales into INT8 weights with truncation),
 * native-VNNI keeps per-block FP16 scales separate and decodes nibbles
 * directly to INT8:
 *   Q4_0:   w_int8 = q - 8     (exact)
 *   IQ4_NL: w_int8 = lut[q]    (exact, via device constant LUT)
 *
 * Split from ROCmGemvKernel.hip.
 *
 * @see ROCmGemvKernel_INT8_VNNI.hip for INT8 VNNI kernels
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstdlib>

#include "../../backends/rocm/HipDeviceGuard.h"

constexpr uint8_t CODEBOOK_IQ4 = 4;

__device__ __constant__ int8_t k_iq4_codebook_i8[16] = {
    -127, -104, -83, -65,
    -49, -35, -22, -10,
    1, 13, 25, 38,
    53, 69, 89, 113};


// ============================================================================
// Native-VNNI GEMV kernel: lossless Q4/IQ4 decode with per-block FP16 scales
//
// Eliminates ratio-VNNI's (ratio * d + 64) >> 7 truncation by keeping
// per-block scales separate and decoding nibbles directly to INT8:
//   Q4_0:   w_int8 = q - 8     (exact)
//   IQ4_NL: w_int8 = lut[q]    (exact)
//
// Compute flow per block (32 elements):
//   int32_block = Σ sdot4(a_packed, w_packed)   // 8 sdot4 calls
//   f_acc += (float)int32_block * __half2float(block_scale)
//
// Output is FP32 with scale_A applied inline — no epilogue kernel needed.
// ============================================================================
template <int TILE_N, int CPT, bool IQ4>
__global__ void gemv_native_vnni_kernel_t(
    const int8_t*    __restrict__ d_A_int8,       // [K] quantized activations (INT8)
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × 16] nibble payload
    const uint16_t*  __restrict__ d_block_scales,  // [blocks_per_row × N] FP16 as uint16_t
    float*           __restrict__ d_C_fp32,        // [N] output (FP32)
    const float*     __restrict__ d_scale_A,       // [1] activation scale
    int N, int K, int kblocks)
{
    static_assert(CPT == 1 || CPT == 2, "native-VNNI supports CPT in {1,2}");

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int kb = blockIdx.y;

    const int blocks_per_row = K / 32;
    const int b_per = (blocks_per_row + kblocks - 1) / kblocks;
    const int b_start = kb * b_per;
    const int b_end = min(b_start + b_per, blocks_per_row);

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
        f_acc[c] = 0.0f;

    if (active)
    {
        for (int b = b_start; b < b_end; ++b)
        {
            // A pointer for this block's 32 INT8 activation values
            const int32_t* a4 = reinterpret_cast<const int32_t*>(d_A_int8 + b * 32);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const size_t linear = static_cast<size_t>(b) * static_cast<size_t>(N) + static_cast<size_t>(n);
                    const uint8_t* payload16 = d_payload + linear * 16;
                    const uint16_t scale_bits = d_block_scales[linear];

                    // Decode 32 nibbles to 8 packed INT32 words (4 INT8 each)
                    // Q4_0 nibble layout: qs[j] = lo_nibble(q_j) | hi_nibble(q_{j+16})
                    // We must decode in sequential element order [q0,q1,...,q31] to match
                    // the sequential activation layout [a0,a1,...,a31] for sdot4.
                    //
                    // Groups 0-3: elements [0..15] from low nibbles of qs[0..15]
                    // Groups 4-7: elements [16..31] from high nibbles of qs[0..15]
                    int32_t packed_groups[8];

                    if constexpr (IQ4)
                    {
                        // IQ4_NL: nibble → LUT lookup → INT8
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint8_t b0 = payload16[g * 4 + 0];
                            const uint8_t b1 = payload16[g * 4 + 1];
                            const uint8_t b2 = payload16[g * 4 + 2];
                            const uint8_t b3 = payload16[g * 4 + 3];

                            // Group g: elements [4g..4g+3] via low nibbles
                            const int8_t lo0 = k_iq4_codebook_i8[b0 & 0x0F];
                            const int8_t lo1 = k_iq4_codebook_i8[b1 & 0x0F];
                            const int8_t lo2 = k_iq4_codebook_i8[b2 & 0x0F];
                            const int8_t lo3 = k_iq4_codebook_i8[b3 & 0x0F];
                            packed_groups[g] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo3)) << 24));

                            // Group g+4: elements [4g+16..4g+19] via high nibbles
                            const int8_t hi0 = k_iq4_codebook_i8[b0 >> 4];
                            const int8_t hi1 = k_iq4_codebook_i8[b1 >> 4];
                            const int8_t hi2 = k_iq4_codebook_i8[b2 >> 4];
                            const int8_t hi3 = k_iq4_codebook_i8[b3 >> 4];
                            packed_groups[g + 4] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi3)) << 24));
                        }
                    }
                    else
                    {
                        // Q4_0: nibble → (q - 8) → INT8 (exact, sequential order)
                        #pragma unroll
                        for (int g = 0; g < 4; ++g)
                        {
                            const uint8_t b0 = payload16[g * 4 + 0];
                            const uint8_t b1 = payload16[g * 4 + 1];
                            const uint8_t b2 = payload16[g * 4 + 2];
                            const uint8_t b3 = payload16[g * 4 + 3];

                            // Group g: elements [4g..4g+3] via low nibbles
                            const int8_t lo0 = static_cast<int8_t>((b0 & 0x0F) - 8);
                            const int8_t lo1 = static_cast<int8_t>((b1 & 0x0F) - 8);
                            const int8_t lo2 = static_cast<int8_t>((b2 & 0x0F) - 8);
                            const int8_t lo3 = static_cast<int8_t>((b3 & 0x0F) - 8);
                            packed_groups[g] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(lo3)) << 24));

                            // Group g+4: elements [4g+16..4g+19] via high nibbles
                            const int8_t hi0 = static_cast<int8_t>((b0 >> 4) - 8);
                            const int8_t hi1 = static_cast<int8_t>((b1 >> 4) - 8);
                            const int8_t hi2 = static_cast<int8_t>((b2 >> 4) - 8);
                            const int8_t hi3 = static_cast<int8_t>((b3 >> 4) - 8);
                            packed_groups[g + 4] = static_cast<int32_t>(
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi0))      ) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi1)) <<  8) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi2)) << 16) |
                                (static_cast<uint32_t>(static_cast<uint8_t>(hi3)) << 24));
                        }
                    }

                    // 8× sdot4: INT8 dot product of 32 activation × weight elements
                    int32_t block_acc = 0;
                    #pragma unroll
                    for (int g = 0; g < 8; ++g)
                    {
                        block_acc = __builtin_amdgcn_sdot4(a4[g], packed_groups[g], block_acc, false);
                    }

                    // Convert FP16 scale to FP32 and accumulate
                    const float block_scale = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
                    f_acc[c] += static_cast<float>(block_acc) * block_scale;
                }
            }
        }

        // Apply activation scale and write output
        const float scale_a = d_scale_A[0];
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                const float result = f_acc[c] * scale_a;
                if (kblocks == 1)
                {
                    d_C_fp32[n] = result;
                }
                else
                {
                    atomicAdd(&d_C_fp32[n], result);
                }
            }
        }
    }
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {

// ============================================================================
// Native-VNNI GEMV dispatch: lossless Q4/IQ4 decode with FP16 per-block scales
// ============================================================================
bool rocmGemv_native_vnni_fp32(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const void* d_block_scales,  // __half* (FP16)
    float* d_C_fp32,
    const float* d_scale_A,
    int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 32) != 0)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] K must be multiple of 32 (N=%d K=%d)\n", N, K);
        return false;
    }

    const int blocks_per_row = K / 32;
    const bool is_iq4 = (codebook_id == CODEBOOK_IQ4);

    // Ensure IQ4 LUT is loaded (when needed)
    // Note: for native-VNNI we use the k_iq4_codebook_i8 device constant directly
    // which is statically initialized — no runtime init needed.

    constexpr int TN = 128;   // TILE_N: 128 output columns per block
    constexpr int CPT = 2;    // Columns per thread: 64 threads = 1 full wavefront (wavefront64)

    // K-partition heuristic: split K across grid.y for occupancy
    int kb;
    if (blocks_per_row > 256)
        kb = min(20, max(4, blocks_per_row / 16));
    else
        kb = min(12, max(2, blocks_per_row / 8));

    // Zero output for atomicAdd accumulation
    hipMemsetAsync(d_C_fp32, 0, static_cast<size_t>(N) * sizeof(float), static_cast<hipStream_t>(stream));

    const int logical_tn = TN * CPT;
    const int grid_n = (N + logical_tn - 1) / logical_tn;
    dim3 grid(grid_n, kb);
    dim3 block(TN, 1, 1);

    const uint16_t* d_scales_u16 = static_cast<const uint16_t*>(d_block_scales);

    if (is_iq4)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN * CPT, CPT, true>),
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_payload, d_scales_u16, d_C_fp32, d_scale_A,
            N, K, kb);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_native_vnni_kernel_t<TN * CPT, CPT, false>),
            grid, block, 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_payload, d_scales_u16, d_C_fp32, d_scale_A,
            N, K, kb);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_native_vnni_fp32] Kernel launch failed: %s (N=%d K=%d kb=%d codebook=%u)\n",
                hipGetErrorString(err), N, K, kb, static_cast<unsigned>(codebook_id));
        return false;
    }

    return true;
}

} // extern "C"
