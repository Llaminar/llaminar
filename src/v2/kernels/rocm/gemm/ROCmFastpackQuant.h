#pragma once
/**
 * @file ROCmFastpackQuant.h
 * @brief Device-side helpers for GFX906 v_cvt_pk_u8_f32 "fast-pack" INT8 quantization.
 *
 * Replaces the standard per-element chain of:
 *   mul → rintf → cvt_i32 → clamp [-127,127] → cast<uint8_t> → shift/OR pack
 * with:
 *   fma(val, inv_scale, 128.0f) → v_cvt_pk_u8_f32 ×4 → XOR 0x80808080
 *
 * The v_cvt_pk_u8_f32 instruction performs FP32→UINT8 conversion with
 * round-to-nearest-even and saturation to [0,255] in a single VALU cycle,
 * placing the result in a selected byte lane of a 32-bit register.
 *
 * By FMA-shifting the signed range [-127,127] to unsigned [1,255], we let
 * the hardware saturation handle clamping and the byte-lane selector handle
 * packing. A final XOR flips back to signed representation.
 */

#include <hip/hip_runtime.h>

namespace llaminar2
{
    namespace rocm
    {

        // ============================================================================
        // Core: pack 4 FP32 values into a signed INT8x4 dword
        // ============================================================================

        /**
         * Pack 4 FP32 values into a single uint32 of 4 signed INT8 bytes.
         *
         * @param v0,v1,v2,v3  FP32 values (raw, NOT pre-scaled)
         * @param inv_scale     Reciprocal of the quantization scale (127.0f / absmax)
         * @return              Packed int8x4 as uint32 (byte 0 = v0, byte 3 = v3)
         */
        __device__ __forceinline__ unsigned fastpack_q8_4x(float v0, float v1, float v2, float v3, float inv_scale)
        {
            // FMA: val * inv_scale + 128.0f → maps [-127,127] to [1,255]
            float u0 = __fmaf_rn(v0, inv_scale, 128.0f);
            float u1 = __fmaf_rn(v1, inv_scale, 128.0f);
            float u2 = __fmaf_rn(v2, inv_scale, 128.0f);
            float u3 = __fmaf_rn(v3, inv_scale, 128.0f);

            // v_cvt_pk_u8_f32: saturate [0,255] + pack into byte lane
            unsigned pack = __builtin_amdgcn_cvt_pk_u8_f32(u0, 0, 0);
            pack = __builtin_amdgcn_cvt_pk_u8_f32(u1, 1, pack);
            pack = __builtin_amdgcn_cvt_pk_u8_f32(u2, 2, pack);
            pack = __builtin_amdgcn_cvt_pk_u8_f32(u3, 3, pack);

            // XOR: unsigned [0,255] → signed [-128,127]
            return pack ^ 0x80808080u;
        }

        /**
         * Scalar fallback: quantize a single FP32 value to INT8.
         * Used for tail elements when count is not a multiple of 4.
         */
        __device__ __forceinline__
            int8_t
            fastpack_q8_scalar(float val, float inv_scale)
        {
            float u = __fmaf_rn(val, inv_scale, 128.0f);
            unsigned pack = __builtin_amdgcn_cvt_pk_u8_f32(u, 0, 0);
            pack ^= 0x00000080u;
            return static_cast<int8_t>(pack & 0xFF);
        }

        // ============================================================================
        // Pattern A: Quantize from shared memory (GEMV fused kernels)
        //   Reads from s_A_fp32[], writes to s_A_int8[] in LDS.
        //   Each thread processes 4 contiguous elements at base = tid*4.
        // ============================================================================

        /**
         * Cooperative LDS quantization: each thread packs 4 elements from shared FP32
         * to shared INT8.  Call with __syncthreads() after.
         *
         * @param s_A_fp32   Shared memory FP32 input array
         * @param s_A_int8   Shared memory INT8 output array
         * @param tid        Thread ID within the workgroup
         * @param tile_len   Number of valid elements in the tile
         * @param inv_scale  Reciprocal scale factor
         */
        __device__ __forceinline__ void fastpack_lds_tile(const float *s_A_fp32, int8_t *s_A_int8,
                                                          int tid, int tile_len, float inv_scale)
        {
            const int base = tid * 4;
            if (base + 3 < tile_len)
            {
                unsigned pack = fastpack_q8_4x(
                    s_A_fp32[base + 0], s_A_fp32[base + 1],
                    s_A_fp32[base + 2], s_A_fp32[base + 3],
                    inv_scale);
                *reinterpret_cast<unsigned *>(s_A_int8 + base) = pack;
            }
            else if (base < tile_len)
            {
                for (int i = base; i < tile_len; ++i)
                {
                    s_A_int8[i] = fastpack_q8_scalar(s_A_fp32[i], inv_scale);
                }
            }
        }

        // ============================================================================
        // Pattern B: Quantize from global memory (standalone GEMM quant kernels)
        //   Reads from global FP32, writes packed dword to global INT8.
        //   Used in V3/V5 vectorized kernels with float4 loads.
        // ============================================================================

        /**
         * Pack a float4 from global memory to a packed int8x4 dword.
         * Writes directly to the output location.
         *
         * @param v           float4 input (already loaded from global memory)
         * @param inv_scale   Reciprocal scale
         * @param row_out     Global INT8 output pointer
         * @param k           Offset into row_out for the dword store
         */
        __device__ __forceinline__ void fastpack_global_vec4(float4 v, float inv_scale, int8_t *row_out, int k)
        {
            unsigned pack = fastpack_q8_4x(v.x, v.y, v.z, v.w, inv_scale);
            *reinterpret_cast<uint32_t *>(row_out + k) = pack;
        }

    } // namespace rocm
} // namespace llaminar2
