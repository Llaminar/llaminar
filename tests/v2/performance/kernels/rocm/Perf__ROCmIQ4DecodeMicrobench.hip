/**
 * @file Perf__ROCmIQ4DecodeMicrobench.hip
 * @brief Native IQ4 decoding timing with the production signed integer contract.
 *
 * Packed codebook values retain their original lane ownership and arithmetic.
 * The compiler selects the native signed dot4 encoding without changing the
 * benchmark's decomposition, accumulation order or timing boundaries.
 */
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <string>

#include "fort.hpp"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace
{
#ifdef HAVE_ROCM
    __device__ __constant__ int8_t kIq4LutI8[16] = {
        -127, -104, -83, -65,
        -49, -35, -22, -10,
        1, 13, 25, 38,
        53, 69, 89, 113};

    __device__ __forceinline__ int32_t dot8_groups(const int32_t *a4, const int32_t *w8)
    {
        int32_t acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], w8[g], acc);
        }
        return acc;
    }

    __device__ __forceinline__ int32_t decode_q4_pack4_low(uint8_t q0, uint8_t q1, uint8_t q2, uint8_t q3, int8_t ratio)
    {
        const uint8_t i0 = static_cast<uint8_t>(q0 & 0x0F);
        const uint8_t i1 = static_cast<uint8_t>(q1 & 0x0F);
        const uint8_t i2 = static_cast<uint8_t>(q2 & 0x0F);
        const uint8_t i3 = static_cast<uint8_t>(q3 & 0x0F);

        const int d0 = static_cast<int>(kIq4LutI8[i0]);
        const int d1 = static_cast<int>(kIq4LutI8[i1]);
        const int d2 = static_cast<int>(kIq4LutI8[i2]);
        const int d3 = static_cast<int>(kIq4LutI8[i3]);

        const int8_t w0 = static_cast<int8_t>((static_cast<int>(ratio) * d0 + 64) >> 7);
        const int8_t w1 = static_cast<int8_t>((static_cast<int>(ratio) * d1 + 64) >> 7);
        const int8_t w2 = static_cast<int8_t>((static_cast<int>(ratio) * d2 + 64) >> 7);
        const int8_t w3 = static_cast<int8_t>((static_cast<int>(ratio) * d3 + 64) >> 7);

        uint32_t packed = 0;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(w0));
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w1)) << 8);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w2)) << 16);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w3)) << 24);
        return static_cast<int32_t>(packed);
    }

    __device__ __forceinline__ int32_t decode_q4_pack4_high(uint8_t q0, uint8_t q1, uint8_t q2, uint8_t q3, int8_t ratio)
    {
        const uint8_t i0 = static_cast<uint8_t>(q0 >> 4);
        const uint8_t i1 = static_cast<uint8_t>(q1 >> 4);
        const uint8_t i2 = static_cast<uint8_t>(q2 >> 4);
        const uint8_t i3 = static_cast<uint8_t>(q3 >> 4);

        const int d0 = static_cast<int>(kIq4LutI8[i0]);
        const int d1 = static_cast<int>(kIq4LutI8[i1]);
        const int d2 = static_cast<int>(kIq4LutI8[i2]);
        const int d3 = static_cast<int>(kIq4LutI8[i3]);

        const int8_t w0 = static_cast<int8_t>((static_cast<int>(ratio) * d0 + 64) >> 7);
        const int8_t w1 = static_cast<int8_t>((static_cast<int>(ratio) * d1 + 64) >> 7);
        const int8_t w2 = static_cast<int8_t>((static_cast<int>(ratio) * d2 + 64) >> 7);
        const int8_t w3 = static_cast<int8_t>((static_cast<int>(ratio) * d3 + 64) >> 7);

        uint32_t packed = 0;
        packed |= static_cast<uint32_t>(static_cast<uint8_t>(w0));
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w1)) << 8);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w2)) << 16);
        packed |= (static_cast<uint32_t>(static_cast<uint8_t>(w3)) << 24);
        return static_cast<int32_t>(packed);
    }

    __device__ __forceinline__ int32_t decode_q4_group_to_packed_i8(const uint8_t *payload16, int group, int8_t ratio)
    {
        const int byte_base = (group & 3) * 4;
        const bool high = (group >= 4);

        const uint8_t q0 = payload16[byte_base + 0];
        const uint8_t q1 = payload16[byte_base + 1];
        const uint8_t q2 = payload16[byte_base + 2];
        const uint8_t q3 = payload16[byte_base + 3];

        if (high)
        {
            return decode_q4_pack4_high(q0, q1, q2, q3, ratio);
        }
        return decode_q4_pack4_low(q0, q1, q2, q3, ratio);
    }

    __device__ __forceinline__ void decode_q4_block8_to_packed_i8_with_lut64(
        const uint8_t *payload16,
        uint64_t lut_lo,
        uint64_t lut_hi,
        int32_t out[8])
    {
        const uint4 packed = *reinterpret_cast<const uint4 *>(payload16);
        const uint32_t p0 = packed.x;
        const uint32_t p1 = packed.y;
        const uint32_t p2 = packed.z;
        const uint32_t p3 = packed.w;

        auto lut_lookup = [&](uint8_t idx) -> int8_t {
            const uint64_t lut = (idx < 8) ? lut_lo : lut_hi;
            const uint32_t sh = static_cast<uint32_t>(idx & 7u) * 8u;
            return static_cast<int8_t>((lut >> sh) & 0xFFu);
        };

        auto pack4 = [&](uint8_t q0, uint8_t q1, uint8_t q2, uint8_t q3, bool high) -> int32_t {
            const uint8_t i0 = high ? static_cast<uint8_t>(q0 >> 4) : static_cast<uint8_t>(q0 & 0x0Fu);
            const uint8_t i1 = high ? static_cast<uint8_t>(q1 >> 4) : static_cast<uint8_t>(q1 & 0x0Fu);
            const uint8_t i2 = high ? static_cast<uint8_t>(q2 >> 4) : static_cast<uint8_t>(q2 & 0x0Fu);
            const uint8_t i3 = high ? static_cast<uint8_t>(q3 >> 4) : static_cast<uint8_t>(q3 & 0x0Fu);

            const uint8_t w0 = static_cast<uint8_t>(lut_lookup(i0));
            const uint8_t w1 = static_cast<uint8_t>(lut_lookup(i1));
            const uint8_t w2 = static_cast<uint8_t>(lut_lookup(i2));
            const uint8_t w3 = static_cast<uint8_t>(lut_lookup(i3));

            uint32_t packed4 = 0;
            packed4 |= static_cast<uint32_t>(w0);
            packed4 |= (static_cast<uint32_t>(w1) << 8);
            packed4 |= (static_cast<uint32_t>(w2) << 16);
            packed4 |= (static_cast<uint32_t>(w3) << 24);
            return static_cast<int32_t>(packed4);
        };

        const uint8_t q0 = static_cast<uint8_t>(p0 & 0xFFu);
        const uint8_t q1 = static_cast<uint8_t>((p0 >> 8) & 0xFFu);
        const uint8_t q2 = static_cast<uint8_t>((p0 >> 16) & 0xFFu);
        const uint8_t q3 = static_cast<uint8_t>((p0 >> 24) & 0xFFu);
        const uint8_t q4 = static_cast<uint8_t>(p1 & 0xFFu);
        const uint8_t q5 = static_cast<uint8_t>((p1 >> 8) & 0xFFu);
        const uint8_t q6 = static_cast<uint8_t>((p1 >> 16) & 0xFFu);
        const uint8_t q7 = static_cast<uint8_t>((p1 >> 24) & 0xFFu);
        const uint8_t q8 = static_cast<uint8_t>(p2 & 0xFFu);
        const uint8_t q9 = static_cast<uint8_t>((p2 >> 8) & 0xFFu);
        const uint8_t q10 = static_cast<uint8_t>((p2 >> 16) & 0xFFu);
        const uint8_t q11 = static_cast<uint8_t>((p2 >> 24) & 0xFFu);
        const uint8_t q12 = static_cast<uint8_t>(p3 & 0xFFu);
        const uint8_t q13 = static_cast<uint8_t>((p3 >> 8) & 0xFFu);
        const uint8_t q14 = static_cast<uint8_t>((p3 >> 16) & 0xFFu);
        const uint8_t q15 = static_cast<uint8_t>((p3 >> 24) & 0xFFu);

        out[0] = pack4(q0, q1, q2, q3, false);
        out[1] = pack4(q4, q5, q6, q7, false);
        out[2] = pack4(q8, q9, q10, q11, false);
        out[3] = pack4(q12, q13, q14, q15, false);
        out[4] = pack4(q0, q1, q2, q3, true);
        out[5] = pack4(q4, q5, q6, q7, true);
        out[6] = pack4(q8, q9, q10, q11, true);
        out[7] = pack4(q12, q13, q14, q15, true);
    }

    __device__ __forceinline__ void decode_q4_block8_to_packed_i8_with_map(
        const uint8_t *payload16,
        const int8_t map16[16],
        int32_t out[8])
    {
        const uint4 packed = *reinterpret_cast<const uint4 *>(payload16);
        const uint32_t p0 = packed.x;
        const uint32_t p1 = packed.y;
        const uint32_t p2 = packed.z;
        const uint32_t p3 = packed.w;

        const uint8_t q0 = static_cast<uint8_t>(p0 & 0xFFu);
        const uint8_t q1 = static_cast<uint8_t>((p0 >> 8) & 0xFFu);
        const uint8_t q2 = static_cast<uint8_t>((p0 >> 16) & 0xFFu);
        const uint8_t q3 = static_cast<uint8_t>((p0 >> 24) & 0xFFu);
        const uint8_t q4 = static_cast<uint8_t>(p1 & 0xFFu);
        const uint8_t q5 = static_cast<uint8_t>((p1 >> 8) & 0xFFu);
        const uint8_t q6 = static_cast<uint8_t>((p1 >> 16) & 0xFFu);
        const uint8_t q7 = static_cast<uint8_t>((p1 >> 24) & 0xFFu);
        const uint8_t q8 = static_cast<uint8_t>(p2 & 0xFFu);
        const uint8_t q9 = static_cast<uint8_t>((p2 >> 8) & 0xFFu);
        const uint8_t q10 = static_cast<uint8_t>((p2 >> 16) & 0xFFu);
        const uint8_t q11 = static_cast<uint8_t>((p2 >> 24) & 0xFFu);
        const uint8_t q12 = static_cast<uint8_t>(p3 & 0xFFu);
        const uint8_t q13 = static_cast<uint8_t>((p3 >> 8) & 0xFFu);
        const uint8_t q14 = static_cast<uint8_t>((p3 >> 16) & 0xFFu);
        const uint8_t q15 = static_cast<uint8_t>((p3 >> 24) & 0xFFu);

        auto pack4 = [&](int a, int b, int c, int d) -> int32_t {
            uint32_t packed4 = 0;
            packed4 |= static_cast<uint32_t>(static_cast<uint8_t>(a));
            packed4 |= (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8);
            packed4 |= (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16);
            packed4 |= (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
            return static_cast<int32_t>(packed4);
        };

        out[0] = pack4(map16[q0 & 0x0Fu], map16[q1 & 0x0Fu], map16[q2 & 0x0Fu], map16[q3 & 0x0Fu]);
        out[1] = pack4(map16[q4 & 0x0Fu], map16[q5 & 0x0Fu], map16[q6 & 0x0Fu], map16[q7 & 0x0Fu]);
        out[2] = pack4(map16[q8 & 0x0Fu], map16[q9 & 0x0Fu], map16[q10 & 0x0Fu], map16[q11 & 0x0Fu]);
        out[3] = pack4(map16[q12 & 0x0Fu], map16[q13 & 0x0Fu], map16[q14 & 0x0Fu], map16[q15 & 0x0Fu]);

        out[4] = pack4(map16[q0 >> 4], map16[q1 >> 4], map16[q2 >> 4], map16[q3 >> 4]);
        out[5] = pack4(map16[q4 >> 4], map16[q5 >> 4], map16[q6 >> 4], map16[q7 >> 4]);
        out[6] = pack4(map16[q8 >> 4], map16[q9 >> 4], map16[q10 >> 4], map16[q11 >> 4]);
        out[7] = pack4(map16[q12 >> 4], map16[q13 >> 4], map16[q14 >> 4], map16[q15 >> 4]);
    }

    __device__ __forceinline__ int32_t decode_q4_group_to_packed_i8_with_lut64(
        const uint8_t *payload16,
        int group,
        uint64_t lut_lo,
        uint64_t lut_hi)
    {
        const int byte_base = (group & 3) * 4;
        const bool high = (group >= 4);

        auto lut_lookup = [&](uint8_t idx) -> int8_t {
            const uint64_t lut = (idx < 8) ? lut_lo : lut_hi;
            const uint32_t sh = static_cast<uint32_t>(idx & 7u) * 8u;
            return static_cast<int8_t>((lut >> sh) & 0xFFu);
        };

        const uint8_t q0 = payload16[byte_base + 0];
        const uint8_t q1 = payload16[byte_base + 1];
        const uint8_t q2 = payload16[byte_base + 2];
        const uint8_t q3 = payload16[byte_base + 3];

        const uint8_t i0 = high ? static_cast<uint8_t>(q0 >> 4) : static_cast<uint8_t>(q0 & 0x0Fu);
        const uint8_t i1 = high ? static_cast<uint8_t>(q1 >> 4) : static_cast<uint8_t>(q1 & 0x0Fu);
        const uint8_t i2 = high ? static_cast<uint8_t>(q2 >> 4) : static_cast<uint8_t>(q2 & 0x0Fu);
        const uint8_t i3 = high ? static_cast<uint8_t>(q3 >> 4) : static_cast<uint8_t>(q3 & 0x0Fu);

        const uint8_t w0 = static_cast<uint8_t>(lut_lookup(i0));
        const uint8_t w1 = static_cast<uint8_t>(lut_lookup(i1));
        const uint8_t w2 = static_cast<uint8_t>(lut_lookup(i2));
        const uint8_t w3 = static_cast<uint8_t>(lut_lookup(i3));

        uint32_t packed4 = 0;
        packed4 |= static_cast<uint32_t>(w0);
        packed4 |= (static_cast<uint32_t>(w1) << 8);
        packed4 |= (static_cast<uint32_t>(w2) << 16);
        packed4 |= (static_cast<uint32_t>(w3) << 24);
        return static_cast<int32_t>(packed4);
    }

    __device__ __forceinline__ void decode_q4_low_high_pair_to_packed_i8_with_lut64(
        uint8_t q0,
        uint8_t q1,
        uint8_t q2,
        uint8_t q3,
        uint64_t lut_lo,
        uint64_t lut_hi,
        int32_t &packed_low,
        int32_t &packed_high)
    {
        auto lut_lookup = [&](uint8_t idx) -> uint8_t {
            const uint64_t lut = (idx < 8) ? lut_lo : lut_hi;
            const uint32_t sh = static_cast<uint32_t>(idx & 7u) * 8u;
            return static_cast<uint8_t>((lut >> sh) & 0xFFu);
        };

        const uint8_t i0l = static_cast<uint8_t>(q0 & 0x0Fu);
        const uint8_t i1l = static_cast<uint8_t>(q1 & 0x0Fu);
        const uint8_t i2l = static_cast<uint8_t>(q2 & 0x0Fu);
        const uint8_t i3l = static_cast<uint8_t>(q3 & 0x0Fu);

        const uint8_t i0h = static_cast<uint8_t>(q0 >> 4);
        const uint8_t i1h = static_cast<uint8_t>(q1 >> 4);
        const uint8_t i2h = static_cast<uint8_t>(q2 >> 4);
        const uint8_t i3h = static_cast<uint8_t>(q3 >> 4);

        uint32_t low = 0;
        low |= static_cast<uint32_t>(lut_lookup(i0l));
        low |= (static_cast<uint32_t>(lut_lookup(i1l)) << 8);
        low |= (static_cast<uint32_t>(lut_lookup(i2l)) << 16);
        low |= (static_cast<uint32_t>(lut_lookup(i3l)) << 24);

        uint32_t high = 0;
        high |= static_cast<uint32_t>(lut_lookup(i0h));
        high |= (static_cast<uint32_t>(lut_lookup(i1h)) << 8);
        high |= (static_cast<uint32_t>(lut_lookup(i2h)) << 16);
        high |= (static_cast<uint32_t>(lut_lookup(i3h)) << 24);

        packed_low = static_cast<int32_t>(low);
        packed_high = static_cast<int32_t>(high);
    }

    __global__ void iq4_decode_dot_lut64_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        uint64_t lut_lo = 0;
        uint64_t lut_hi = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int8_t wi_lo = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i]) + 64) >> 7);
            const int8_t wi_hi = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i + 8]) + 64) >> 7);
            lut_lo |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_lo)) << (i * 8));
            lut_hi |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_hi)) << (i * 8));
        }

        int32_t w8[8];
        decode_q4_block8_to_packed_i8_with_lut64(payload16, lut_lo, lut_hi, w8);
        out[tid] = dot8_groups(a4, w8);
    }

    __global__ void iq4_decode_dot_group_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        int32_t acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            const int32_t w = decode_q4_group_to_packed_i8(payload16, g, r);
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], w, acc);
        }
        out[tid] = acc;
    }

    __global__ void iq4_decode_dot_map16_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        int8_t map16[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            map16[i] = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i]) + 64) >> 7);
        }

        int32_t w8[8];
        decode_q4_block8_to_packed_i8_with_map(payload16, map16, w8);
        out[tid] = dot8_groups(a4, w8);
    }

    __global__ void iq4_decode_dot_lut64_fused_group_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        uint64_t lut_lo = 0;
        uint64_t lut_hi = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int8_t wi_lo = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i]) + 64) >> 7);
            const int8_t wi_hi = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i + 8]) + 64) >> 7);
            lut_lo |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_lo)) << (i * 8));
            lut_hi |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_hi)) << (i * 8));
        }

        int32_t acc = 0;
        #pragma unroll
        for (int g = 0; g < 8; ++g)
        {
            const int32_t w = decode_q4_group_to_packed_i8_with_lut64(payload16, g, lut_lo, lut_hi);
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], w, acc);
        }

        out[tid] = acc;
    }

    __global__ void iq4_decode_dot_split_low_high_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        int32_t acc = 0;
        #pragma unroll
        for (int g = 0; g < 4; ++g)
        {
            const int32_t w = decode_q4_group_to_packed_i8(payload16, g, r);
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], w, acc);
        }
        #pragma unroll
        for (int g = 4; g < 8; ++g)
        {
            const int32_t w = decode_q4_group_to_packed_i8(payload16, g, r);
            acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[g], w, acc);
        }
        out[tid] = acc;
    }

    __global__ void iq4_decode_dot_lut64_fused_pair_kernel(const uint8_t *payload, const int8_t *ratio, const int32_t *a4, int32_t *out, int total)
    {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        if (tid >= total) return;

        const uint8_t *payload16 = payload + static_cast<size_t>(tid) * 16;
        const int8_t r = ratio[tid];

        uint64_t lut_lo = 0;
        uint64_t lut_hi = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            const int8_t wi_lo = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i]) + 64) >> 7);
            const int8_t wi_hi = static_cast<int8_t>((static_cast<int>(r) * static_cast<int>(kIq4LutI8[i + 8]) + 64) >> 7);
            lut_lo |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_lo)) << (i * 8));
            lut_hi |= (static_cast<uint64_t>(static_cast<uint8_t>(wi_hi)) << (i * 8));
        }

        const uint4 packed = *reinterpret_cast<const uint4 *>(payload16);

        const uint8_t q0 = static_cast<uint8_t>(packed.x & 0xFFu);
        const uint8_t q1 = static_cast<uint8_t>((packed.x >> 8) & 0xFFu);
        const uint8_t q2 = static_cast<uint8_t>((packed.x >> 16) & 0xFFu);
        const uint8_t q3 = static_cast<uint8_t>((packed.x >> 24) & 0xFFu);

        const uint8_t q4 = static_cast<uint8_t>(packed.y & 0xFFu);
        const uint8_t q5 = static_cast<uint8_t>((packed.y >> 8) & 0xFFu);
        const uint8_t q6 = static_cast<uint8_t>((packed.y >> 16) & 0xFFu);
        const uint8_t q7 = static_cast<uint8_t>((packed.y >> 24) & 0xFFu);

        const uint8_t q8 = static_cast<uint8_t>(packed.z & 0xFFu);
        const uint8_t q9 = static_cast<uint8_t>((packed.z >> 8) & 0xFFu);
        const uint8_t q10 = static_cast<uint8_t>((packed.z >> 16) & 0xFFu);
        const uint8_t q11 = static_cast<uint8_t>((packed.z >> 24) & 0xFFu);

        const uint8_t q12 = static_cast<uint8_t>(packed.w & 0xFFu);
        const uint8_t q13 = static_cast<uint8_t>((packed.w >> 8) & 0xFFu);
        const uint8_t q14 = static_cast<uint8_t>((packed.w >> 16) & 0xFFu);
        const uint8_t q15 = static_cast<uint8_t>((packed.w >> 24) & 0xFFu);

        int32_t acc = 0;
        int32_t w_low = 0;
        int32_t w_high = 0;

        decode_q4_low_high_pair_to_packed_i8_with_lut64(q0, q1, q2, q3, lut_lo, lut_hi, w_low, w_high);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[0], w_low, acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[4], w_high, acc);

        decode_q4_low_high_pair_to_packed_i8_with_lut64(q4, q5, q6, q7, lut_lo, lut_hi, w_low, w_high);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[1], w_low, acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[5], w_high, acc);

        decode_q4_low_high_pair_to_packed_i8_with_lut64(q8, q9, q10, q11, lut_lo, lut_hi, w_low, w_high);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[2], w_low, acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[6], w_high, acc);

        decode_q4_low_high_pair_to_packed_i8_with_lut64(q12, q13, q14, q15, lut_lo, lut_hi, w_low, w_high);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[3], w_low, acc);
        acc = ::llaminar2::rocm_integer_contract::signedDot4(a4[7], w_high, acc);

        out[tid] = acc;
    }

    double time_kernel_ms(void (*kernel)(const uint8_t *, const int8_t *, const int32_t *, int32_t *, int),
                          const uint8_t *d_payload,
                          const int8_t *d_ratio,
                          const int32_t *d_a4,
                          int32_t *d_out,
                          int total,
                          int warmup,
                          int runs)
    {
        constexpr int BLOCK = 256;
        const int GRID = (total + BLOCK - 1) / BLOCK;

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        (void)hipEventCreate(&start);
        (void)hipEventCreate(&stop);

        for (int i = 0; i < warmup; ++i)
        {
            hipLaunchKernelGGL(kernel, dim3(GRID), dim3(BLOCK), 0, 0, d_payload, d_ratio, d_a4, d_out, total);
        }
        (void)hipDeviceSynchronize();

        std::vector<float> samples;
        samples.reserve(static_cast<size_t>(runs));
        for (int i = 0; i < runs; ++i)
        {
            (void)hipEventRecord(start);
            hipLaunchKernelGGL(kernel, dim3(GRID), dim3(BLOCK), 0, 0, d_payload, d_ratio, d_a4, d_out, total);
            (void)hipEventRecord(stop);
            (void)hipEventSynchronize(stop);
            float ms = 0.0f;
            (void)hipEventElapsedTime(&ms, start, stop);
            samples.push_back(ms);
        }

        (void)hipEventDestroy(start);
        (void)hipEventDestroy(stop);

        return std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(samples.size());
    }
#endif

    class ROCmIQ4DecodeMicrobenchTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
#ifdef HAVE_ROCM
            int count = 0;
            if (hipGetDeviceCount(&count) != hipSuccess || count <= 0)
            {
                GTEST_SKIP() << "No ROCm device";
            }
            (void)hipSetDevice(0);
#else
            GTEST_SKIP() << "ROCm not enabled";
#endif
        }
    };

    TEST_F(ROCmIQ4DecodeMicrobenchTest, DecodeDotApproachComparison)
    {
#ifdef HAVE_ROCM
        constexpr int total_blocks = 1 << 20;
        constexpr int warmup = 10;
        constexpr int runs = 50;

        std::mt19937 rng(1337);
        std::uniform_int_distribution<int> qdist(0, 255);
        std::uniform_int_distribution<int> rdist(-127, 127);
        std::uniform_int_distribution<int> adist(-127, 127);

        std::vector<uint8_t> h_payload(static_cast<size_t>(total_blocks) * 16);
        std::vector<int8_t> h_ratio(static_cast<size_t>(total_blocks));
        std::vector<int32_t> h_a4(8);

        for (auto &v : h_payload) v = static_cast<uint8_t>(qdist(rng));
        for (auto &v : h_ratio)
        {
            int r = rdist(rng);
            if (r == 0) r = 1;
            v = static_cast<int8_t>(r);
        }
        for (auto &v : h_a4)
        {
            const int8_t a0 = static_cast<int8_t>(adist(rng));
            const int8_t a1 = static_cast<int8_t>(adist(rng));
            const int8_t a2 = static_cast<int8_t>(adist(rng));
            const int8_t a3 = static_cast<int8_t>(adist(rng));
            uint32_t packed = 0;
            packed |= static_cast<uint32_t>(static_cast<uint8_t>(a0));
            packed |= (static_cast<uint32_t>(static_cast<uint8_t>(a1)) << 8);
            packed |= (static_cast<uint32_t>(static_cast<uint8_t>(a2)) << 16);
            packed |= (static_cast<uint32_t>(static_cast<uint8_t>(a3)) << 24);
            v = static_cast<int32_t>(packed);
        }

        uint8_t *d_payload = nullptr;
        int8_t *d_ratio = nullptr;
        int32_t *d_a4 = nullptr;
        int32_t *d_out = nullptr;

        ASSERT_EQ(hipMalloc(&d_payload, h_payload.size() * sizeof(uint8_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_ratio, h_ratio.size() * sizeof(int8_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_a4, h_a4.size() * sizeof(int32_t)), hipSuccess);
        ASSERT_EQ(hipMalloc(&d_out, static_cast<size_t>(total_blocks) * sizeof(int32_t)), hipSuccess);

        ASSERT_EQ(hipMemcpy(d_payload, h_payload.data(), h_payload.size() * sizeof(uint8_t), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(d_ratio, h_ratio.data(), h_ratio.size() * sizeof(int8_t), hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(d_a4, h_a4.data(), h_a4.size() * sizeof(int32_t), hipMemcpyHostToDevice), hipSuccess);

        const double lut64_ms = time_kernel_ms(iq4_decode_dot_lut64_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);
        const double group_ms = time_kernel_ms(iq4_decode_dot_group_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);
        const double map16_ms = time_kernel_ms(iq4_decode_dot_map16_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);
        const double lut64_fused_ms = time_kernel_ms(iq4_decode_dot_lut64_fused_group_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);
        const double split_low_high_ms = time_kernel_ms(iq4_decode_dot_split_low_high_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);
        const double lut64_fused_pair_ms = time_kernel_ms(iq4_decode_dot_lut64_fused_pair_kernel, d_payload, d_ratio, d_a4, d_out, total_blocks, warmup, runs);

        std::vector<int32_t> h_out(1024);
        ASSERT_EQ(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(int32_t), hipMemcpyDeviceToHost), hipSuccess);
        const int64_t checksum = std::accumulate(h_out.begin(), h_out.end(), int64_t{0});

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Variant" << "Mean ms" << "Blocks/s (M)" << "Rel vs LUT64" << fort::endr;

        const double lut64_mblocks = (static_cast<double>(total_blocks) / std::max(lut64_ms, 1e-9)) / 1000.0;
        const double group_mblocks = (static_cast<double>(total_blocks) / std::max(group_ms, 1e-9)) / 1000.0;
        const double map16_mblocks = (static_cast<double>(total_blocks) / std::max(map16_ms, 1e-9)) / 1000.0;
        const double lut64_fused_mblocks = (static_cast<double>(total_blocks) / std::max(lut64_fused_ms, 1e-9)) / 1000.0;
        const double split_low_high_mblocks = (static_cast<double>(total_blocks) / std::max(split_low_high_ms, 1e-9)) / 1000.0;
        const double lut64_fused_pair_mblocks = (static_cast<double>(total_blocks) / std::max(lut64_fused_pair_ms, 1e-9)) / 1000.0;

        table << "LUT64 precompute + block decode" << lut64_ms << lut64_mblocks << 1.0 << fort::endr;
        table << "Per-group decode" << group_ms << group_mblocks << (lut64_ms / std::max(group_ms, 1e-9)) << fort::endr;
        table << "map16 staging + block decode" << map16_ms << map16_mblocks << (lut64_ms / std::max(map16_ms, 1e-9)) << fort::endr;
        table << "LUT64 fused group decode+dot" << lut64_fused_ms << lut64_fused_mblocks << (lut64_ms / std::max(lut64_fused_ms, 1e-9)) << fort::endr;
        table << "Split low/high group loops" << split_low_high_ms << split_low_high_mblocks << (lut64_ms / std::max(split_low_high_ms, 1e-9)) << fort::endr;
        table << "LUT64 fused low/high pair" << lut64_fused_pair_ms << lut64_fused_pair_mblocks << (lut64_ms / std::max(lut64_fused_pair_ms, 1e-9)) << fort::endr;

        std::cout << "\n" << table.to_string() << std::endl;
        std::cout << "Checksum(sample): " << checksum << std::endl;

        EXPECT_GT(lut64_ms, 0.0);
        EXPECT_GT(group_ms, 0.0);
        EXPECT_GT(map16_ms, 0.0);
        EXPECT_GT(lut64_fused_ms, 0.0);
        EXPECT_GT(split_low_high_ms, 0.0);
        EXPECT_GT(lut64_fused_pair_ms, 0.0);

        (void)hipFree(d_payload);
        (void)hipFree(d_ratio);
        (void)hipFree(d_a4);
        (void)hipFree(d_out);
#endif
    }
}
