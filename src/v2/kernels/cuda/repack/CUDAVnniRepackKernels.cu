/**
 * @file CUDAVnniRepackKernels.cu
 * @brief CUDA kernels for repacking raw GGUF blocks into VNNI-interleaved layout
 *
 * CUDA counterpart of kernels/rocm/repack/VnniRepackKernels.hip.
 * Contains GPU kernels for all 18 RepackFormat values (Q4_0 through IQ1_M).
 * Each kernel maps one thread to one (n, b) block position, performing
 * the same nibble repack and scale extraction as the CPU reference path.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>

#include "kernels/cuda/repack/CUDAVnniRepackKernels.h"
#include "tensors/BlockStructures.h"

namespace llaminar2 {

// ============================================================================
// Alignment-safe memory access helpers
// ============================================================================
// Block struct fields (e.g. Q4_0Block::qs at offset 2) may not be aligned
// for 4-byte reads. On NVIDIA GPUs, misaligned uint32_t loads via
// reinterpret_cast can trigger cudaErrorMisalignedAddress. Using memcpy
// is well-defined C++ and the CUDA compiler optimises small fixed-size
// copies into efficient device code.

__device__ __forceinline__ uint32_t load_u32(const void* ptr)
{
    uint32_t v;
    memcpy(&v, ptr, 4);
    return v;
}

__device__ __forceinline__ void copy16(void* __restrict__ dst, const void* __restrict__ src)
{
    // Copy 16 bytes — compiler will vectorise for aligned dst
    memcpy(dst, src, 16);
}

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

__constant__ uint8_t d_ksigns_iq2xs_cuda[128] = {
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

__global__ void cuda_repack_q4_0_to_vnni(
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

    copy16(d_payload + linear * 16, blk.qs);

    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 2: Q4_1 repack (asymmetric 4-bit with min)
// ============================================================================

__global__ void cuda_repack_q4_1_to_vnni(
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

    copy16(d_payload + linear * 16, blk.qs);

    d_scales[linear] = blk.d;
    d_mins[linear]   = blk.m;
}

// ============================================================================
// Kernel 3: Q5_0 repack (symmetric 5-bit)
// ============================================================================

__global__ void cuda_repack_q5_0_to_vnni(
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

    uint8_t* dst = d_payload + linear * 20;
    copy16(dst, blk.qs);
    memcpy(dst + 16, blk.qh, 4);

    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 4: Q5_1 repack (asymmetric 5-bit with min)
// ============================================================================

__global__ void cuda_repack_q5_1_to_vnni(
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

    uint8_t* dst = d_payload + linear * 20;
    copy16(dst, blk.qs);
    memcpy(dst + 16, blk.qh, 4);

    d_scales[linear] = blk.d;
    d_mins[linear]   = blk.m;
}

// ============================================================================
// Kernel 4b: Q8_0 repack (symmetric 8-bit)
// ============================================================================

__global__ void cuda_repack_q8_0_to_vnni(
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

    // Copy 32 bytes of int8 payload (two 16-byte copies for alignment safety)
    uint8_t* dst = d_payload + linear * 32;
    copy16(dst, blk.qs);
    copy16(dst + 16, blk.qs + 16);

    // Copy FP16 scale
    d_scales[linear] = blk.d;
}

// ============================================================================
// Kernel 5: Q4_K repack
// ============================================================================

__global__ void cuda_repack_q4k_to_vnni(
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

    uint8_t repacked[16];
    if (is_high) {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
    } else {
        for (int i = 0; i < 16; ++i)
            repacked[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
    }

    const size_t linear = static_cast<size_t>(b) * N + n;

    uint32_t* dst = reinterpret_cast<uint32_t*>(d_payload + linear * 16);
    const uint32_t* rp = reinterpret_cast<const uint32_t*>(repacked);
    dst[0] = rp[0]; dst[1] = rp[1]; dst[2] = rp[2]; dst[3] = rp[3];

    uint8_t sc, m_val;
    device_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);

    float d_f    = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    float dmin_f = __half2float(*reinterpret_cast<const __half*>(&blk.dmin));

    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(sc)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(m_val)));
}

// ============================================================================
// Kernel 6: Q5_K repack
// ============================================================================

__global__ void cuda_repack_q5k_to_vnni(
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

    uint8_t repacked_qs[16];
    if (is_high) {
        for (int i = 0; i < 16; ++i)
            repacked_qs[i] = (src32[i] >> 4) | (src32[i + 16] & 0xF0);
    } else {
        for (int i = 0; i < 16; ++i)
            repacked_qs[i] = (src32[i] & 0xF) | ((src32[i + 16] & 0xF) << 4);
    }

    uint8_t repacked_qh[4] = {0, 0, 0, 0};
    for (int i = 0; i < 32; ++i) {
        const int bit_val = (blk.qh[i] >> sub_idx) & 1;
        repacked_qh[i / 8] |= static_cast<uint8_t>(bit_val << (i % 8));
    }

    const size_t linear = static_cast<size_t>(b) * N + n;

    uint8_t* dst = d_payload + linear * 20;
    const uint32_t* rp = reinterpret_cast<const uint32_t*>(repacked_qs);
    uint32_t* dst32 = reinterpret_cast<uint32_t*>(dst);
    dst32[0] = rp[0]; dst32[1] = rp[1]; dst32[2] = rp[2]; dst32[3] = rp[3];
    const uint32_t* rqh = reinterpret_cast<const uint32_t*>(repacked_qh);
    uint32_t* dst_qh = reinterpret_cast<uint32_t*>(dst + 16);
    dst_qh[0] = rqh[0];

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

__global__ void cuda_repack_q6k_to_vnni(
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

    const int half = (sub_idx * 32) / 128;
    const int sub_in_half = (sub_idx * 32 % 128) / 32;
    const uint8_t* ql = blk.ql + half * 64;
    const uint8_t* qh = blk.qh + half * 32;

    uint8_t raw6[32];
    for (int l = 0; l < 32; ++l) {
        switch (sub_in_half) {
        case 0: raw6[l] = (ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4); break;
        case 1: raw6[l] = (ql[l + 32] & 0xF)  | (((qh[l] >> 2) & 3) << 4); break;
        case 2: raw6[l] = (ql[l] >> 4)         | (((qh[l] >> 4) & 3) << 4); break;
        case 3: raw6[l] = (ql[l + 32] >> 4)    | (((qh[l] >> 6) & 3) << 4); break;
        }
    }

    uint8_t payload[24];
    for (int i = 0; i < 16; ++i)
        payload[i] = (raw6[i] & 0xF) | ((raw6[i + 16] & 0xF) << 4);
    for (int i = 0; i < 8; ++i)
        payload[16 + i] = ((raw6[4*i+0] >> 4) & 3)
                        | (((raw6[4*i+1] >> 4) & 3) << 2)
                        | (((raw6[4*i+2] >> 4) & 3) << 4)
                        | (((raw6[4*i+3] >> 4) & 3) << 6);

    const size_t linear = static_cast<size_t>(b) * N + n;

    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 24);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload);
    dst32[0] = pp[0]; dst32[1] = pp[1]; dst32[2] = pp[2]; dst32[3] = pp[3];
    dst32[4] = pp[4]; dst32[5] = pp[5];

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

__global__ void cuda_repack_q3k_to_vnni(
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
    uint32_t hbits_u32 = 0;
    for (int e = 0; e < 32; ++e)
        hbits_u32 |= static_cast<uint32_t>((raw3[e] >> 2) & 1) << e;
    *reinterpret_cast<uint32_t*>(payload_buf + 8) = hbits_u32;

    const size_t linear = static_cast<size_t>(b) * N + n;

    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 12);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload_buf);
    dst32[0] = pp[0]; dst32[1] = pp[1]; dst32[2] = pp[2];

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;
    uint32_t aux[4];
    aux[0] = load_u32(blk.scales);
    aux[1] = load_u32(blk.scales + 4);
    aux[2] = load_u32(blk.scales + 8);
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

__global__ void cuda_repack_q2k_to_vnni(
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

    const int base_elem = sub_idx * 32;
    uint8_t raw2[32];
    for (int e = 0; e < 32; ++e) {
        const int i = base_elem + e;
        const int qs_byte = (i / 128) * 32 + (i % 32);
        const int shift = ((i % 128) / 32) * 2;
        raw2[e] = (blk.qs[qs_byte] >> shift) & 3;
    }

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

    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 8);
    const uint32_t* pp = reinterpret_cast<const uint32_t*>(payload_buf);
    dst32[0] = pp[0]; dst32[1] = pp[1];

    const int sc_lo_idx = sub_idx * 2;
    const int sc_hi_idx = sub_idx * 2 + 1;
    float d_val  = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    float dmin_f = __half2float(*reinterpret_cast<const __half*>(&blk.dmin));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_val * static_cast<float>(blk.scales[sc_lo_idx] & 0xF)));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_val * static_cast<float>(blk.scales[sc_hi_idx] & 0xF)));

    uint16_t emb_min_lo = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(blk.scales[sc_lo_idx] >> 4)));
    uint16_t emb_min_hi = __half_as_ushort(__float2half_rn(-dmin_f * static_cast<float>(blk.scales[sc_hi_idx] >> 4)));
    d_emins[linear] = static_cast<uint32_t>(emb_min_lo) | (static_cast<uint32_t>(emb_min_hi) << 16);
}

// ============================================================================
// Kernel 10: IQ4_XS repack
// ============================================================================

__global__ void cuda_repack_iq4xs_to_vnni(
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

    copy16(d_payload + linear * 16, blk.qs + sub_idx * 16);

    const int ls = ((blk.scales_l[sub_idx / 2] >> (4 * (sub_idx % 2))) & 0xf)
                 | (((blk.scales_h >> (2 * sub_idx)) & 3) << 4);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(ls - 32)));
}

// ============================================================================
// Kernel 11: IQ3_S repack
// ============================================================================

__global__ void cuda_repack_iq3s_to_vnni(
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

    uint8_t* dst = d_payload + linear * 13;
    memcpy(dst, blk.qs + sub_idx * 8, 8);
    dst[8] = blk.qh[sub_idx];
    memcpy(dst + 9, blk.signs + sub_idx * 4, 4);

    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc_byte = blk.scales[sub_idx / 2];
    int nibble = (sub_idx & 1) ? (sc_byte >> 4) : (sc_byte & 0xF);
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * static_cast<float>(1 + 2 * nibble)));
}

// ============================================================================
// Kernel 12: IQ3_XXS repack (uses d_ksigns_iq2xs_cuda)
// ============================================================================

__global__ void cuda_repack_iq3xxs_to_vnni(
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

    uint8_t payload_buf[12];
    memcpy(payload_buf, blk.qs + sub_idx * 8, 8);

    uint32_t aux32 = load_u32(blk.qs + 64 + sub_idx * 4);

    payload_buf[8]  = d_ksigns_iq2xs_cuda[(aux32 >>  0) & 127];
    payload_buf[9]  = d_ksigns_iq2xs_cuda[(aux32 >>  7) & 127];
    payload_buf[10] = d_ksigns_iq2xs_cuda[(aux32 >> 14) & 127];
    payload_buf[11] = d_ksigns_iq2xs_cuda[(aux32 >> 21) & 127];

    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 12);
    dst32[0] = *reinterpret_cast<uint32_t*>(payload_buf);
    dst32[1] = *reinterpret_cast<uint32_t*>(payload_buf + 4);
    dst32[2] = *reinterpret_cast<uint32_t*>(payload_buf + 8);

    int nibble = static_cast<int>(aux32 >> 28);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(nibble)) * 0.5f));
}

// ============================================================================
// Kernel 13: IQ2_S repack (asymmetric)
// ============================================================================

__global__ void cuda_repack_iq2s_to_vnni(
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

    uint8_t* dst = d_payload + linear * 9;
    memcpy(dst, blk.qs + sub_idx * 4, 4);
    dst[4] = blk.qh[sub_idx];
    memcpy(dst + 5, blk.qs + 32 + sub_idx * 4, 4);

    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc = blk.scales[sub_idx];
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f));
}

// ============================================================================
// Kernel 14: IQ2_XS repack (asymmetric, uses d_ksigns_iq2xs_cuda)
// ============================================================================

__global__ void cuda_repack_iq2xs_to_vnni(
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

    uint8_t payload_buf[9];
    uint8_t qh_byte = 0;
    for (int l = 0; l < 4; ++l) {
        uint16_t entry = blk.qs[sub_idx * 4 + l];
        payload_buf[l] = static_cast<uint8_t>(entry & 0xFF);
        qh_byte |= static_cast<uint8_t>(((entry >> 8) & 1) << l);
        payload_buf[5 + l] = d_ksigns_iq2xs_cuda[entry >> 9];
    }
    payload_buf[4] = qh_byte;

    uint8_t* dst = d_payload + linear * 9;
    *reinterpret_cast<uint32_t*>(dst) = *reinterpret_cast<uint32_t*>(payload_buf);
    *reinterpret_cast<uint32_t*>(dst + 4) = *reinterpret_cast<uint32_t*>(payload_buf + 4);
    dst[8] = payload_buf[8];

    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    uint8_t sc = blk.scales[sub_idx];
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f));
    d_mins[linear]   = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f));
}

// ============================================================================
// Kernel 15: IQ2_XXS repack (symmetric, uses d_ksigns_iq2xs_cuda)
// ============================================================================

__global__ void cuda_repack_iq2xxs_to_vnni(
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

    const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(blk.qs);
    uint32_t aux32_0 = load_u32(qs_bytes + sub_idx * 8);
    uint32_t aux32_1 = load_u32(qs_bytes + sub_idx * 8 + 4);

    uint8_t payload_buf[8];
    payload_buf[0] = static_cast<uint8_t>(aux32_0);
    payload_buf[1] = static_cast<uint8_t>(aux32_0 >> 8);
    payload_buf[2] = static_cast<uint8_t>(aux32_0 >> 16);
    payload_buf[3] = static_cast<uint8_t>(aux32_0 >> 24);
    payload_buf[4] = d_ksigns_iq2xs_cuda[(aux32_1 >>  0) & 127];
    payload_buf[5] = d_ksigns_iq2xs_cuda[(aux32_1 >>  7) & 127];
    payload_buf[6] = d_ksigns_iq2xs_cuda[(aux32_1 >> 14) & 127];
    payload_buf[7] = d_ksigns_iq2xs_cuda[(aux32_1 >> 21) & 127];

    uint32_t* dst32 = reinterpret_cast<uint32_t*>(d_payload + linear * 8);
    dst32[0] = *reinterpret_cast<uint32_t*>(payload_buf);
    dst32[1] = *reinterpret_cast<uint32_t*>(payload_buf + 4);

    int nibble = static_cast<int>(aux32_1 >> 28);
    float d_f = __half2float(*reinterpret_cast<const __half*>(&blk.d));
    d_scales[linear] = __half_as_ushort(__float2half_rn(d_f * (0.5f + static_cast<float>(nibble)) * 0.25f));
}

// ============================================================================
// Kernel 16: IQ1_S repack (asymmetric)
// ============================================================================

__global__ void cuda_repack_iq1s_to_vnni(
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

    uint8_t payload_buf[6];
    payload_buf[0] = qs[0];
    payload_buf[1] = qs[1];
    payload_buf[2] = qs[2];
    payload_buf[3] = qs[3];
    payload_buf[4] = static_cast<uint8_t>(qh_word & 0xFF);
    payload_buf[5] = static_cast<uint8_t>((qh_word >> 8) & 0xFF);

    *reinterpret_cast<uint32_t*>(d_payload + linear * 6) = *reinterpret_cast<uint32_t*>(payload_buf);
    *reinterpret_cast<uint16_t*>(d_payload + linear * 6 + 4) = *reinterpret_cast<uint16_t*>(payload_buf + 4);

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

__global__ void cuda_repack_iq1m_to_vnni(
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

    uint16_t sc[4];
    memcpy(sc, blk.scales, 8);
    uint16_t scale_u16 = static_cast<uint16_t>(
        (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
        ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
    float d_val = __half2float(*reinterpret_cast<const __half*>(&scale_u16));

    int sc_word_idx = sub_idx / 2;
    int sc_bit_offset = 6 * (sub_idx % 2);
    int sc3_lo = (sc[sc_word_idx] >> (sc_bit_offset + 0)) & 0x7;
    int sc3_hi = (sc[sc_word_idx] >> (sc_bit_offset + 3)) & 0x7;
    float dl1 = d_val * (2.0f * static_cast<float>(sc3_lo) + 1.0f);
    float dl2 = d_val * (2.0f * static_cast<float>(sc3_hi) + 1.0f);

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

bool launchVnniRepackCUDA(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    uint32_t* d_emins,
    int N, int K,
    void* stream)
{
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int blocks_per_row = K / 32;

    switch (format) {
    case RepackFormat::Q4_0:
    case RepackFormat::IQ4_NL: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        cuda_repack_q4_0_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q4_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q4_1: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        cuda_repack_q4_1_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q4_1Block*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q5_0: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        cuda_repack_q5_0_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q5_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q5_1: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        cuda_repack_q5_1_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q5_1Block*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q8_0: {
        dim3 block(256, 1);
        dim3 grid((N + 255) / 256, blocks_per_row);
        cuda_repack_q8_0_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q8_0Block*>(d_raw_blocks),
            d_payload, d_scales, N, blocks_per_row);
        break;
    }
    case RepackFormat::Q4_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_q4k_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q4_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q5_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_q5k_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q5_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q6_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_q6k_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q6_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q3_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_q3k_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q3_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::Q2_K: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_q2k_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const Q2_KBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, d_emins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ4_XS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq4xs_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ4_XSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ3_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq3s_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ3_SBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ3_XXS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq3xxs_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ3_XXSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq2s_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ2_SBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_XS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq2xs_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ2_XSBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ2_XXS: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq2xxs_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ2_XXSBlock*>(d_raw_blocks),
            d_payload, d_scales, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ1_S: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq1s_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ1_SBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    case RepackFormat::IQ1_M: {
        const int sb_per_row = (K + 255) / 256;
        dim3 block(64, 4);
        dim3 grid((N + 63) / 64, (blocks_per_row + 3) / 4);
        cuda_repack_iq1m_to_vnni<<<grid, block, 0, cuda_stream>>>(
            static_cast<const IQ1_MBlock*>(d_raw_blocks),
            d_payload, d_scales, d_mins, N, K, sb_per_row);
        break;
    }
    default:
        return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[launchVnniRepackCUDA] Kernel launch failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    return true;
}

// ============================================================================
// Host dispatch (7-parameter backward-compat wrapper)
// ============================================================================

bool launchVnniRepackCUDA(
    RepackFormat format,
    const void* d_raw_blocks,
    uint8_t* d_payload,
    uint16_t* d_scales,
    uint16_t* d_mins,
    int N, int K,
    void* stream)
{
    return launchVnniRepackCUDA(format, d_raw_blocks, d_payload, d_scales, d_mins,
                                nullptr, N, K, stream);
}

} // namespace llaminar2
