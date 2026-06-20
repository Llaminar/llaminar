/**
 * @file VnniRepackKernels.hip
 * @brief HIP kernels for repacking raw GGUF blocks into VNNI-interleaved layout
 *
 * Contains GPU kernels for Q4_0, IQ4_NL, and Q4_K formats.
 * Each kernel maps one thread to one (n, b) block position, performing
 * the same nibble repack and scale extraction as the CPU reference path.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

#include "kernels/rocm/repack/VnniRepackKernels.h"
#include "tensors/BlockStructures.h"

namespace llaminar2 {

// ============================================================================
// Device helper: extract scale and min from Q4_K packed scale array
// ============================================================================

__device__ inline void device_get_scale_min_k4(int j, const uint8_t* q,
                                                uint8_t* d, uint8_t* m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

// ============================================================================
// Lookup table for IQ2_XS / IQ2_XXS / IQ3_XXS sign decoding
// ============================================================================

__constant__ uint8_t d_ksigns_iq2xs[128] = {
    0, 129, 130, 3, 132, 5, 6, 135,
    136, 9, 10, 139, 12, 141, 142, 15,
    144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159,
    160, 33, 34, 163, 36, 165, 166, 39,
    40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183,
    184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207,
    80, 209, 210, 83, 212, 85, 86, 215,
    216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231,
    232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255,
};

// ============================================================================
// Kernel 1: Q4_0 / IQ4_NL repack
// ============================================================================
// One thread per (n, b) pair. Simple 16-byte payload copy + FP16 scale copy.
// Grid: (ceil(N / 256), blocks_per_row)
// Block: (256, 1)

__global__ void repack_q4_0_to_vnni(
    const Q4_0Block* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const auto& blk = d_raw[n * blocks_per_row + b];
    const size_t linear = static_cast<size_t>(b) * N + n;

    // Copy 16 bytes of packed nibbles via 4 × uint32 stores
    const uint32_t* src = reinterpret_cast<const uint32_t*>(blk.qs);
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];

    // Copy FP16 scale (already stored as FP16 in GGUF block)
    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 2: Q4_1 repack (asymmetric 4-bit with min)
// ============================================================================
// One thread per (n, b) pair. 16-byte payload copy + FP16 scale + FP16 min.
// Grid: (ceil(N / 256), blocks_per_row)
// Block: (256, 1)

__global__ void repack_q4_1_to_vnni(
    const Q4_1Block* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const auto& blk = d_raw[n * blocks_per_row + b];
    const size_t linear = static_cast<size_t>(b) * N + n;

    // Copy 16 bytes of packed nibbles via 4 × uint32 stores
    const uint32_t* src = reinterpret_cast<const uint32_t*>(blk.qs);
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];

    // Copy FP16 scale and min
    d_scales[linear] = blk.d;
    d_mins[linear]   = blk.m;
}

// ============================================================================
// Kernel 3: Q5_0 repack (symmetric 5-bit)
// ============================================================================
// One thread per (n, b) pair. 20-byte payload (16B qs + 4B qh) + FP16 scale.
// Grid: (ceil(N / 256), blocks_per_row)
// Block: (256, 1)

__global__ void repack_q5_0_to_vnni(
    const Q5_0Block* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const auto& blk = d_raw[n * blocks_per_row + b];
    const size_t linear = static_cast<size_t>(b) * N + n;

    // Copy 16 bytes of packed nibbles (qs)
    uint8_t* dst = d_payload + linear * 20;
    const uint32_t* src_qs = reinterpret_cast<const uint32_t*>(blk.qs);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    dst32[0] = src_qs[0];
    dst32[1] = src_qs[1];
    dst32[2] = src_qs[2];
    dst32[3] = src_qs[3];

    // Copy 4 bytes of high bits (qh)
    const uint32_t* src_qh = reinterpret_cast<const uint32_t*>(blk.qh);
    uint32_t* dst_qh = reinterpret_cast<uint32_t*>(dst + 16);
    dst_qh[0] = src_qh[0];

    // Copy FP16 scale
    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 4: Q5_1 repack (asymmetric 5-bit with min)
// ============================================================================
// One thread per (n, b) pair. 20-byte payload (16B qs + 4B qh) + FP16 scale + FP16 min.
// Grid: (ceil(N / 256), blocks_per_row)
// Block: (256, 1)

__global__ void repack_q5_1_to_vnni(
    const Q5_1Block* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const auto& blk = d_raw[n * blocks_per_row + b];
    const size_t linear = static_cast<size_t>(b) * N + n;

    // Copy 16 bytes of packed nibbles (qs)
    uint8_t* dst = d_payload + linear * 20;
    const uint32_t* src_qs = reinterpret_cast<const uint32_t*>(blk.qs);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    dst32[0] = src_qs[0];
    dst32[1] = src_qs[1];
    dst32[2] = src_qs[2];
    dst32[3] = src_qs[3];

    // Copy 4 bytes of high bits (qh)
    const uint32_t* src_qh = reinterpret_cast<const uint32_t*>(blk.qh);
    uint32_t* dst_qh = reinterpret_cast<uint32_t*>(dst + 16);
    dst_qh[0] = src_qh[0];

    // Copy FP16 scale and min
    d_scales[linear] = blk.d;
    d_mins[linear]   = blk.m;
}

// ============================================================================
// Kernel 4b: Q8_0 repack (symmetric 8-bit)
// ============================================================================
// One thread per (n, b) pair. 32-byte payload copy + FP16 scale copy.
// Grid: (ceil(N / 256), blocks_per_row)
// Block: (256, 1)

__global__ void repack_q8_0_to_vnni(
    const Q8_0Block* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    int N, int blocks_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y;
    if (n >= N || b >= blocks_per_row) return;

    const auto& blk = d_raw[n * blocks_per_row + b];
    const size_t linear = static_cast<size_t>(b) * N + n;

    // Copy 32 bytes of int8 payload via 8 × uint32 stores
    const uint32_t* src = reinterpret_cast<const uint32_t*>(blk.qs);
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 32);
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];

    // Copy FP16 scale (already stored as FP16 in GGUF block)
    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 5: Q4_K repack
// ============================================================================
// One thread per (n, b) pair where b is a 32-element sub-block index.
// blocks_per_row = K / 32. Each 256-element super-block contains 8 sub-blocks.
// Grid: (ceil(N / 64), ceil(blocks_per_row / 4))
// Block: (64, 4)

__global__ void repack_q4k_to_vnni(
    const Q4_KBlock* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const Q4_KBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const int group_idx = sub_idx / 2;
    const int is_high   = sub_idx & 1;
    const uint8_t* src32 = blk.qs + group_idx * 32;

    // Repack nibbles from the 64-byte group into a 16-byte sub-block
    uint8_t repacked[16];
    if (is_high) {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
    } else {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
    }

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Store repacked payload
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    const uint32_t* rp = reinterpret_cast<const uint32_t*>(repacked);
    dst[0] = rp[0];
    dst[1] = rp[1];
    dst[2] = rp[2];
    dst[3] = rp[3];

    // Extract sub-block scale and min from packed 6-bit array
    uint8_t sc, m_val;
    device_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);

    // Convert super-block scale/min to per-sub-block FP16 values
    float d_f    = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    float dmin_f = __half2float(*reinterpret_cast<const __half*>(&blk.dmin));

    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(sc)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(m_val)));
}

// ============================================================================
// Kernel 6: Q5_K repack
// ============================================================================
// One thread per (n, b) sub-block. 20-byte payload (16B repacked qs + 4B qh).
// Same grid/block pattern as Q4_K.

__global__ void repack_q5k_to_vnni(
    const Q5_KBlock* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const Q5_KBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const int group_idx = sub_idx / 2;
    const int is_high   = sub_idx & 1;
    const uint8_t* src32 = blk.qs + group_idx * 32;

    // Repack nibbles (same as Q4_K)
    uint8_t repacked_qs[16];
    if (is_high) {
        for (int i = 0; i < 16; ++i)
            repacked_qs[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
    } else {
        for (int i = 0; i < 16; ++i)
            repacked_qs[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
    }

    // Extract high bits for this sub-block
    uint8_t repacked_qh[4] = {0, 0, 0, 0};
    for (int i = 0; i < 32; ++i) {
        const int bit_val = (blk.qh[i] >> sub_idx) & 1;
        repacked_qh[i / 8] |= static_cast<uint8_t>(bit_val << (i % 8));
    }

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Store 20-byte payload: [16B qs] [4B qh]
    uint8_t* dst = d_payload + linear * 20;
    const uint32_t* rp = reinterpret_cast<const uint32_t*>(repacked_qs);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    dst32[0] = rp[0];
    dst32[1] = rp[1];
    dst32[2] = rp[2];
    dst32[3] = rp[3];
    const uint32_t* rqh = reinterpret_cast<const uint32_t*>(repacked_qh);
    uint32_t* dst_qh = reinterpret_cast<uint32_t*>(dst + 16);
    dst_qh[0] = rqh[0];

    // Scale/min: same as Q4_K (uses get_scale_min_k4)
    uint8_t sc, m_val;
    device_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);
    float d_f    = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    float dmin_f = __half2float(*reinterpret_cast<const __half*>(&blk.dmin));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(sc)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(m_val)));
}

// ============================================================================
// Kernel 7: Q6_K repack
// ============================================================================
// One thread per (n, b) sub-block. 24-byte payload (16B low4 + 8B high2).
// Uses signed int8_t scales.

__global__ void repack_q6k_to_vnni(
    const Q6_KBlock* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const Q6_KBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const int half = (sub_idx * 32) / 128;            // 0 or 1
    const int sub_in_half = (sub_idx * 32 % 128) / 32; // 0-3
    const uint8_t* ql = blk.ql + half * 64;
    const uint8_t* qh = blk.qh + half * 32;

    // Extract 6-bit raw values
    uint8_t raw6[32];
    for (int l = 0; l < 32; ++l) {
        switch (sub_in_half) {
        case 0: raw6[l] = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4); break;
        case 1: raw6[l] = (ql[l + 32] & 0xF)  | (((qh[l] >> 2) & 3) << 4); break;
        case 2: raw6[l] = (ql[l] >> 4)         | (((qh[l] >> 4) & 3) << 4); break;
        case 3: raw6[l] = (ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4); break;
        }
    }

    // Pack into 24-byte payload: [16B low4] [8B high2]
    uint8_t payload[24];
    for (int i = 0; i < 16; ++i)
        payload[i] = (raw6[i] & 0xF) | ((raw6[i + 16] & 0xF) << 4);
    for (int i = 0; i < 8; ++i)
        payload[16 + i] = ((raw6[4*i+0] >> 4) & 3)
                        | (((raw6[4*i+1] >> 4) & 3) << 2)
                        | (((raw6[4*i+2] >> 4) & 3) << 4)
                        | (((raw6[4*i+3] >> 4) & 3) << 6);

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Store payload via uint32 stores
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 24);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload);
    dst32[0] = pp[0]; dst32[1] = pp[1]; dst32[2] = pp[2]; dst32[3] = pp[3];
    dst32[4] = pp[4]; dst32[5] = pp[5];

    // Signed int8_t scales — TWO per sub-block
    const int8_t* signed_scales = reinterpret_cast<const int8_t*>(blk.scales);
    const int sc_lo_idx = half * 8 + sub_in_half * 2;
    const int sc_hi_idx = sc_lo_idx + 1;
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(signed_scales[sc_lo_idx])));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(signed_scales[sc_hi_idx])));
}

// ============================================================================
// Kernel 8: Q3_K repack
// ============================================================================
// One thread per (n, b) sub-block. 12-byte payload (8B low2 + 4B high bits).
// Uses complex 6-bit packed scale format with -32 offset.

__global__ void repack_q3k_to_vnni(
    const Q3_KBlock* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const Q3_KBlock& blk = d_raw[n * sb_per_row + sb_idx];

    // Extract 3-bit raw values
    const int base = sub_idx * 32;
    uint8_t raw3[32];
    for (int e = 0; e < 32; ++e) {
        const int i = base + e;
        const int qs_byte = (i / 128) * 32 + (i % 32);
        const int shift = ((i % 128) / 32) * 2;
        const int low2 = (blk.qs[qs_byte] >> shift) & 3;
        const int hbit = (blk.hmask[i % 32] >> (i / 32)) & 1;
        raw3[e] = static_cast<uint8_t>(low2 | (hbit << 2));
    }

    // Pack low 2 bits into 8 bytes
    uint8_t payload_buf[12];
    for (int h = 0; h < 2; ++h) {
        const int hbase = h * 16;
        for (int j = 0; j < 4; ++j) {
            payload_buf[h * 4 + j] = static_cast<uint8_t>(
                (raw3[hbase + j] & 3)
              | ((raw3[hbase + j + 4] & 3) << 2)
              | ((raw3[hbase + j + 8] & 3) << 4)
              | ((raw3[hbase + j + 12] & 3) << 6));
        }
    }
    // Pack high bits into 4 bytes
    uint32_t hbits_u32 = 0;
    for (int e = 0; e < 32; ++e)
        hbits_u32 |= static_cast<uint32_t>((raw3[e] >> 2) & 1) << e;
    *reinterpret_cast<uint32_t*>(payload_buf + 8) = hbits_u32;

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Store 12-byte payload
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 12);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload_buf);
    dst32[0] = pp[0]; dst32[1] = pp[1]; dst32[2] = pp[2];

    // Unpack 6-bit scales using kmask pattern
    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    // Copy 12 bytes of scales into aux[0..2], aux[3] is zero-init
    aux[0] = *reinterpret_cast<const uint32_t*>(blk.scales);
    aux[1] = *reinterpret_cast<const uint32_t*>(blk.scales + 4);
    aux[2] = *reinterpret_cast<const uint32_t*>(blk.scales + 8);
    aux[3] = 0;
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    int8_t unpacked_scales[16];
    *reinterpret_cast<uint32_t*>(unpacked_scales)      = aux[0];
    *reinterpret_cast<uint32_t*>(unpacked_scales + 4)  = aux[1];
    *reinterpret_cast<uint32_t*>(unpacked_scales + 8)  = aux[2];
    *reinterpret_cast<uint32_t*>(unpacked_scales + 12) = aux[3];

    const int sc_lo_idx = sub_idx * 2;
    const int sc_hi_idx = sub_idx * 2 + 1;
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(unpacked_scales[sc_lo_idx] - 32)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(unpacked_scales[sc_hi_idx] - 32)));
}

// ============================================================================
// Kernel 9: Q2_K repack
// ============================================================================
// One thread per (n, b) sub-block. 8-byte payload. Uses d_emins.

__global__ void repack_q2k_to_vnni(
    const Q2_KBlock* __restrict__ d_raw,
    uint8_t*         __restrict__ d_payload,
    uint16_t*        __restrict__ d_scales,
    uint16_t*        __restrict__ d_mins,
    uint32_t*        __restrict__ d_emins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const Q2_KBlock& blk = d_raw[n * sb_per_row + sb_idx];

    // Extract 2-bit raw values
    const int base_elem = sub_idx * 32;
    uint8_t raw2[32];
    for (int e = 0; e < 32; ++e) {
        const int i = base_elem + e;
        const int qs_byte = (i / 128) * 32 + (i % 32);
        const int shift = ((i % 128) / 32) * 2;
        raw2[e] = (blk.qs[qs_byte] >> shift) & 3;
    }

    // Pack 2-bit values into 8 bytes
    uint8_t payload_buf[8];
    for (int half = 0; half < 2; ++half) {
        const int hbase = half * 16;
        for (int j = 0; j < 4; ++j) {
            payload_buf[half * 4 + j] = static_cast<uint8_t>(
                raw2[hbase + j]
              | (raw2[hbase + j + 4] << 2)
              | (raw2[hbase + j + 8] << 4)
              | (raw2[hbase + j + 12] << 6));
        }
    }

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Store 8-byte payload
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 8);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload_buf);
    dst32[0] = pp[0]; dst32[1] = pp[1];

    // Scale extraction from packed nibbles
    const int sc_lo_idx = sub_idx * 2;
    const int sc_hi_idx = sub_idx * 2 + 1;
    float d_val  = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    float dmin_f = __half2float(*reinterpret_cast<const __half*>(&blk.dmin));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_val * static_cast<float>(blk.scales[sc_lo_idx] & 0xF)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_val * static_cast<float>(blk.scales[sc_hi_idx] & 0xF)));

    // Emins: TWO FP16 values packed into one uint32_t
    uint16_t emb_min_lo = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(blk.scales[sc_lo_idx] >> 4)));
    uint16_t emb_min_hi = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(blk.scales[sc_hi_idx] >> 4)));
    d_emins[linear] = static_cast<uint32_t>(emb_min_lo) | (static_cast<uint32_t>(emb_min_hi) << 16);
}

// ============================================================================
// Kernel 10: IQ4_XS repack
// ============================================================================

__global__ void repack_iq4xs_to_vnni(
    const IQ4_XSBlock* __restrict__ d_raw,
    uint8_t*           __restrict__ d_payload,
    uint16_t*          __restrict__ d_scales,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ4_XSBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Payload: 16 bytes from qs[sub_idx*16]
    const uint32_t* src = reinterpret_cast<const uint32_t*>(blk.qs + sub_idx * 16);
    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];

    // Scale: 6-bit from split hi/lo
    const int ls = ((blk.scales_l[sub_idx / 2] >> (4 * (sub_idx % 2))) & 0xf)
                 | (((blk.scales_h >> (2 * sub_idx)) & 3) << 4);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(ls - 32)));
}

// ============================================================================
// Kernel 11: IQ3_S repack
// ============================================================================

__global__ void repack_iq3s_to_vnni(
    const IQ3_SBlock* __restrict__ d_raw,
    uint8_t*          __restrict__ d_payload,
    uint16_t*         __restrict__ d_scales,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ3_SBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Payload: 8B qs + 1B qh + 4B signs = 13B
    uint8_t* dst = d_payload + linear * 13;
    const uint32_t* qs_src = reinterpret_cast<const uint32_t*>(blk.qs + sub_idx * 8);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    dst32[0] = qs_src[0]; dst32[1] = qs_src[1];
    dst[8] = blk.qh[sub_idx];
    const uint32_t* signs_src = reinterpret_cast<const uint32_t*>(blk.signs + sub_idx * 4);
    *reinterpret_cast<uint32_t*>(dst + 9) = signs_src[0];

    // Scale: nibble from scales
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc_byte = blk.scales[sub_idx / 2];
    int nibble = (sub_idx & 1) ? (sc_byte >> 4) : (sc_byte & 0xF);
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(1 + 2 * nibble)));
}

// ============================================================================
// Kernel 12: IQ3_XXS repack (uses d_ksigns_iq2xs)
// ============================================================================

__global__ void repack_iq3xxs_to_vnni(
    const IQ3_XXSBlock* __restrict__ d_raw,
    uint8_t*            __restrict__ d_payload,
    uint16_t*           __restrict__ d_scales,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ3_XXSBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Payload bytes 0-7: copy 8 bytes from qs[sub_idx*8]
    uint8_t payload_buf[12];
    const uint32_t* src = reinterpret_cast<const uint32_t*>(blk.qs + sub_idx * 8);
    *reinterpret_cast<uint32_t*>(payload_buf) = src[0];
    *reinterpret_cast<uint32_t*>(payload_buf + 4) = src[1];

    // aux32 from qs[64 + sub_idx*4]
    const uint32_t* aux_src = reinterpret_cast<const uint32_t*>(blk.qs + 64 + sub_idx * 4);
    uint32_t aux32 = aux_src[0];

    // Payload bytes 8-11: sign bytes via ksigns_iq2xs lookup
    payload_buf[8]  = d_ksigns_iq2xs[(aux32 >>  0) & 127];
    payload_buf[9]  = d_ksigns_iq2xs[(aux32 >>  7) & 127];
    payload_buf[10] = d_ksigns_iq2xs[(aux32 >> 14) & 127];
    payload_buf[11] = d_ksigns_iq2xs[(aux32 >> 21) & 127];

    // Store 12-byte payload
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 12);
    dst32[0] = *reinterpret_cast<uint32_t*>(payload_buf);
    dst32[1] = *reinterpret_cast<uint32_t*>(payload_buf + 4);
    dst32[2] = *reinterpret_cast<uint32_t*>(payload_buf + 8);

    // Scale: nibble from aux32 bits [28:31]
    int nibble = static_cast<int>(aux32 >> 28);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(nibble)) * 0.5f));
}

// ============================================================================
// Kernel 13: IQ2_S repack (asymmetric)
// ============================================================================

__global__ void repack_iq2s_to_vnni(
    const IQ2_SBlock* __restrict__ d_raw,
    uint8_t*          __restrict__ d_payload,
    uint16_t*         __restrict__ d_scales,
    uint16_t*         __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ2_SBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Payload: 4B qs[sub_idx*4] + 1B qh[sub_idx] + 4B qs[32+sub_idx*4] = 9B
    uint8_t* dst = d_payload + linear * 9;
    const uint32_t* src1 = reinterpret_cast<const uint32_t*>(blk.qs + sub_idx * 4);
    *reinterpret_cast<uint32_t*>(dst) = src1[0];
    dst[4] = blk.qh[sub_idx];
    const uint32_t* src2 = reinterpret_cast<const uint32_t*>(blk.qs + 32 + sub_idx * 4);
    *reinterpret_cast<uint32_t*>(dst + 5) = src2[0];

    // Dual scale from scales[sub_idx]
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc = blk.scales[sub_idx];
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f));
}

// ============================================================================
// Kernel 14: IQ2_XS repack (asymmetric, uses d_ksigns_iq2xs)
// ============================================================================

__global__ void repack_iq2xs_to_vnni(
    const IQ2_XSBlock* __restrict__ d_raw,
    uint8_t*           __restrict__ d_payload,
    uint16_t*          __restrict__ d_scales,
    uint16_t*          __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ2_XSBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Unpack 4 uint16_t entries from blk.qs[sub_idx*4 .. sub_idx*4+3]
    uint8_t payload_buf[9];
    uint8_t qh_byte = 0;
    for (int l = 0; l < 4; ++l) {
        uint16_t entry = blk.qs[sub_idx * 4 + l];
        payload_buf[l] = static_cast<uint8_t>(entry & 0xFF);
        qh_byte |= static_cast<uint8_t>(((entry >> 8) & 1) << l);
        payload_buf[5 + l] = d_ksigns_iq2xs[entry >> 9];
    }
    payload_buf[4] = qh_byte;

    // Store 9 bytes
    uint8_t* dst = d_payload + linear * 9;
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t*>(payload_buf);
    *reinterpret_cast<uint32_t*>(dst + 4) = *reinterpret_cast<uint32_t*>(payload_buf + 4);
    dst[8] = payload_buf[8];

    // Dual scale
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc = blk.scales[sub_idx];
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f));
}

// ============================================================================
// Kernel 15: IQ2_XXS repack (symmetric, uses d_ksigns_iq2xs)
// ============================================================================

__global__ void repack_iq2xxs_to_vnni(
    const IQ2_XXSBlock* __restrict__ d_raw,
    uint8_t*            __restrict__ d_payload,
    uint16_t*           __restrict__ d_scales,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ2_XXSBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    // Read 8 bytes as 2x uint32_t from qs (note: qs is uint16_t*, recast)
    const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(blk.qs);
    const uint32_t* aux_src = reinterpret_cast<const uint32_t*>(qs_bytes + sub_idx * 8);
    uint32_t aux32_0 = aux_src[0];
    uint32_t aux32_1 = aux_src[1];

    // Payload bytes 0-3: grid indices from aux32_0
    uint8_t payload_buf[8];
    payload_buf[0] = static_cast<uint8_t>(aux32_0);
    payload_buf[1] = static_cast<uint8_t>(aux32_0 >> 8);
    payload_buf[2] = static_cast<uint8_t>(aux32_0 >> 16);
    payload_buf[3] = static_cast<uint8_t>(aux32_0 >> 24);
    // Payload bytes 4-7: sign bytes via ksigns_iq2xs
    payload_buf[4] = d_ksigns_iq2xs[(aux32_1 >>  0) & 127];
    payload_buf[5] = d_ksigns_iq2xs[(aux32_1 >>  7) & 127];
    payload_buf[6] = d_ksigns_iq2xs[(aux32_1 >> 14) & 127];
    payload_buf[7] = d_ksigns_iq2xs[(aux32_1 >> 21) & 127];

    // Store 8 bytes via 2x uint32
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 8);
    dst32[0] = *reinterpret_cast<uint32_t*>(payload_buf);
    dst32[1] = *reinterpret_cast<uint32_t*>(payload_buf + 4);

    // Scale: nibble from aux32_1 bits [28:31]
    int nibble = static_cast<int>(aux32_1 >> 28);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(nibble)) * 0.25f));
}

// ============================================================================
// Kernel 16: IQ1_S repack (asymmetric)
// ============================================================================

__global__ void repack_iq1s_to_vnni(
    const IQ1_SBlock* __restrict__ d_raw,
    uint8_t*          __restrict__ d_payload,
    uint16_t*         __restrict__ d_scales,
    uint16_t*         __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ1_SBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    const uint8_t* qs = blk.qs + sub_idx * 4;
    uint16_t qh_word = blk.qh[sub_idx];

    // Payload: 4B qs + 2B qh_word = 6B
    uint8_t payload_buf[6];
    payload_buf[0] = qs[0];
    payload_buf[1] = qs[1];
    payload_buf[2] = qs[2];
    payload_buf[3] = qs[3];
    payload_buf[4] = static_cast<uint8_t>(qh_word & 0xFF);
    payload_buf[5] = static_cast<uint8_t>((qh_word >> 8) & 0xFF);

    *reinterpret_cast<uint32_t*>(d_payload + linear * 6) = *reinterpret_cast<uint32_t*>(payload_buf);
    *reinterpret_cast<uint16_t*>(d_payload + linear * 6 + 4) = *reinterpret_cast<uint16_t*>(payload_buf + 4);

    // Scale: scale_sel from bits [12:14], delta sign from bit 15
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    int scale_sel = (qh_word >> 12) & 7;
    float dl = d_f * (2.0f * static_cast<float>(scale_sel) + 1.0f);

    constexpr float IQ1S_DELTA = 0.125f;
    float delta = (qh_word & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

    d_scales[linear] = __half_as_ushort(__float2half_rn(dl));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(dl * delta));
}

// ============================================================================
// Kernel 17: IQ1_M repack (asymmetric)
// ============================================================================

__global__ void repack_iq1m_to_vnni(
    const IQ1_MBlock* __restrict__ d_raw,
    uint8_t*          __restrict__ d_payload,
    uint16_t*         __restrict__ d_scales,
    uint16_t*         __restrict__ d_mins,
    int N, int K, int sb_per_row)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int b = blockIdx.y * blockDim.y + threadIdx.y;
    const int blocks_per_row = K / 32;
    if (n >= N || b >= blocks_per_row) return;

    const int sb_idx  = b / 8;
    const int sub_idx = b % 8;
    const IQ1_MBlock& blk = d_raw[n * sb_per_row + sb_idx];

    const size_t linear = static_cast<size_t>(b) * N + n;

    const uint8_t* qs = blk.qs + sub_idx * 4;
    const uint8_t* qh = blk.qh + sub_idx * 2;

    // Reconstruct master scale from blk->scales as uint16_t[4]
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
    uint16_t scale_u16 = static_cast<uint16_t>(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d_val = __half2float(*reinterpret_cast<const __half*>(&scale_u16));

    // Extract two 3-bit scale values
    int sc_word_idx = sub_idx / 2;
    int sc_bit_offset = 6 * (sub_idx % 2);
    int sc3_lo = (sc[sc_word_idx] >> (sc_bit_offset + 0)) & 0x7;
    int sc3_hi = (sc[sc_word_idx] >> (sc_bit_offset + 3)) & 0x7;
    float dl1 = d_val * (2.0f * static_cast<float>(sc3_lo) + 1.0f);
    float dl2 = d_val * (2.0f * static_cast<float>(sc3_hi) + 1.0f);

    // Payload: 4B qs + 2B qh = 6B
    uint8_t payload_buf[6];
    payload_buf[0] = qs[0];
    payload_buf[1] = qs[1];
    payload_buf[2] = qs[2];
    payload_buf[3] = qs[3];
    payload_buf[4] = qh[0];
    payload_buf[5] = qh[1];

    *reinterpret_cast<uint32_t*>(d_payload + linear * 6) = *reinterpret_cast<uint32_t*>(payload_buf);
    *reinterpret_cast<uint16_t*>(d_payload + linear * 6 + 4) = *reinterpret_cast<uint16_t*>(payload_buf + 4);

    d_scales[linear] = __half_as_ushort(__float2half_rn(dl1));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(dl2));
}

// ============================================================================
// Host dispatch (8-parameter overload — canonical implementation)
// ============================================================================

bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    void* stream)
{
    hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
    const int blocks_per_row = K / 32;

    switch (format) {
    case RepackFormat::Q4_0:
    case RepackFormat::IQ4_NL: {
        // Q4_0 and IQ4_NL have identical 18-byte block layout (d + qs[16])
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        repack_q4_0_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q4_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q4_1: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        repack_q4_1_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q4_1Block*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q5_0: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        repack_q5_0_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q5_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q5_1: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        repack_q5_1_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q5_1Block*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q8_0: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        repack_q8_0_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q8_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q4_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_q4k_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q4_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q5_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_q5k_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q5_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q6_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_q6k_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q6_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q3_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_q3k_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q3_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q2_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_q2k_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const Q2_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, d_emins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ4_XS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq4xs_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ4_XSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ3_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq3s_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ3_SBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ3_XXS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq3xxs_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ3_XXSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq2s_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ2_SBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_XS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq2xs_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ2_XSBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_XXS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq2xxs_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ2_XXSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ1_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq1s_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ1_SBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ1_M: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        repack_iq1m_to_vnni<<<grid, block, 0, hip_stream>>>(
            static_cast<const IQ1_MBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    default:
        return false;
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[launchVnniRepack] Kernel launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// Host dispatch (7-parameter backward-compat wrapper)
// ============================================================================

bool launchVnniRepack(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    int N, int K,
    void* stream)
{
    return launchVnniRepack(format, d_raw_blocks, d_payload, d_scales, d_mins,
                            nullptr, N, K, stream);
}

} // namespace llaminar2
