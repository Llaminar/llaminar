/**
 * @file Test__VnniRepackKernels.cpp
 * @brief Parity tests for GPU VNNI repack kernels vs CPU reference
 *
 * Creates synthetic GGUF blocks with known values, packs them via both
 * the CPU reference path and the GPU kernel, then compares byte-for-byte
 * (payload) and value-for-value (scales/mins with FP16 tolerance).
 */

#include <gtest/gtest.h>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>

#include "kernels/rocm/repack/VnniRepackKernels.h"
#include "tensors/BlockStructures.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2 {

// ============================================================================
// CPU reference helpers (inline, no tensor class dependency)
// ============================================================================

namespace {

/// CPU FP16→FP32 conversion via bit manipulation
inline float cpu_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            uint32_t bits = sign;
            float f;
            std::memcpy(&f, &bits, sizeof(f));
            return f;
        }
        // Subnormal — normalize
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 31) {
        // Inf/NaN
        uint32_t bits = sign | 0x7F800000u | (mant << 13);
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }

    uint32_t bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

/// CPU FP32→FP16 conversion (round-to-nearest-even)
inline uint16_t cpu_fp32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint16_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) {
        if (exp < -10) return sign;
        mant = (mant | 0x800000) >> (1 - exp);
        // Round to nearest even
        if ((mant & 0x1FFF) > 0x1000 || ((mant & 0x1FFF) == 0x1000 && (mant & 0x2000)))
            mant += 0x2000;
        return sign | static_cast<uint16_t>(mant >> 13);
    }
    if (exp >= 31) return sign | 0x7C00; // Inf

    // Round to nearest even
    if ((mant & 0x1FFF) > 0x1000 || ((mant & 0x1FFF) == 0x1000 && (mant & 0x2000))) {
        mant += 0x2000;
        if (mant & 0x800000) { mant = 0; exp++; }
    }
    if (exp >= 31) return sign | 0x7C00;

    return sign | static_cast<uint16_t>(exp << 10) | static_cast<uint16_t>(mant >> 13);
}

/// CPU get_scale_min_k4 — extract 6-bit scale and min from packed array
inline void cpu_get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j - 0] >> 6) << 4);
    }
}

/// CPU reference: pack Q4_0/IQ4_NL blocks into VNNI layout
void cpu_pack_q4_0(const Q4_0Block* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const auto& blk = blocks[n * blocks_per_row + b];
            const size_t linear = static_cast<size_t>(b) * N + n;
            std::memcpy(payload.data() + linear * 16, blk.qs, 16);
            scales[linear] = blk.d;
        }
    }
}

/// CPU reference: pack Q4_K blocks into VNNI layout
void cpu_pack_q4k(const Q4_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload,
                  std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

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

            std::memcpy(payload.data() + linear * 16, repacked, 16);

            uint8_t sc, m_val;
            cpu_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);
            const float d    = cpu_fp16_to_fp32(blk.d);
            const float dmin = cpu_fp16_to_fp32(blk.dmin);
            scales[linear] = cpu_fp32_to_fp16(d * static_cast<float>(sc));
            mins[linear]   = cpu_fp32_to_fp16(-dmin * static_cast<float>(m_val));
        }
    }
}

/// CPU reference: pack Q4_1 blocks into VNNI layout (16B payload + scale + min)
void cpu_pack_q4_1(const Q4_1Block* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales,
                   std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const auto& blk = blocks[n * blocks_per_row + b];
            const size_t linear = static_cast<size_t>(b) * N + n;
            std::memcpy(payload.data() + linear * 16, blk.qs, 16);
            scales[linear] = blk.d;
            mins[linear]   = blk.m;
        }
    }
}

/// CPU reference: pack Q5_0 blocks into VNNI layout (20B payload + scale)
void cpu_pack_q5_0(const Q5_0Block* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 20);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const auto& blk = blocks[n * blocks_per_row + b];
            const size_t linear = static_cast<size_t>(b) * N + n;
            uint8_t* dst = payload.data() + linear * 20;
            std::memcpy(dst, blk.qs, 16);
            std::memcpy(dst + 16, blk.qh, 4);
            scales[linear] = blk.d;
        }
    }
}

/// CPU reference: pack Q5_1 blocks into VNNI layout (20B payload + scale + min)
void cpu_pack_q5_1(const Q5_1Block* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales,
                   std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 20);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const auto& blk = blocks[n * blocks_per_row + b];
            const size_t linear = static_cast<size_t>(b) * N + n;
            uint8_t* dst = payload.data() + linear * 20;
            std::memcpy(dst, blk.qs, 16);
            std::memcpy(dst + 16, blk.qh, 4);
            scales[linear] = blk.d;
            mins[linear]   = blk.m;
        }
    }
}

/// CPU reference: pack Q5_K blocks into VNNI layout (20B payload + scale + min)
void cpu_pack_q5k(const Q5_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload,
                  std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 20);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

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

            uint8_t* dst = payload.data() + linear * 20;
            std::memcpy(dst, repacked_qs, 16);
            std::memcpy(dst + 16, repacked_qh, 4);

            uint8_t sc, m_val;
            cpu_get_scale_min_k4(sub_idx, blk.scales, &sc, &m_val);
            const float d    = cpu_fp16_to_fp32(blk.d);
            const float dmin = cpu_fp16_to_fp32(blk.dmin);
            scales[linear] = cpu_fp32_to_fp16(d * static_cast<float>(sc));
            mins[linear]   = cpu_fp32_to_fp16(-dmin * static_cast<float>(m_val));
        }
    }
}

/// CPU reference: pack Q6_K blocks into VNNI layout (24B payload + scale + min)
void cpu_pack_q6k(const Q6_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload,
                  std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 24);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

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

            uint8_t pbuf[24];
            for (int i = 0; i < 16; ++i)
                pbuf[i] = (raw6[i] & 0xF) | ((raw6[i + 16] & 0xF) << 4);
            for (int i = 0; i < 8; ++i)
                pbuf[16 + i] = ((raw6[4*i+0] >> 4) & 3)
                             | (((raw6[4*i+1] >> 4) & 3) << 2)
                             | (((raw6[4*i+2] >> 4) & 3) << 4)
                             | (((raw6[4*i+3] >> 4) & 3) << 6);

            std::memcpy(payload.data() + linear * 24, pbuf, 24);

            const int8_t* signed_scales = reinterpret_cast<const int8_t*>(blk.scales);
            const int sc_lo_idx = half * 8 + sub_in_half * 2;
            const int sc_hi_idx = sc_lo_idx + 1;
            const float d = cpu_fp16_to_fp32(blk.d);
            scales[linear] = cpu_fp32_to_fp16(d * static_cast<float>(signed_scales[sc_lo_idx]));
            mins[linear]   = cpu_fp32_to_fp16(d * static_cast<float>(signed_scales[sc_hi_idx]));
        }
    }
}

/// CPU reference: pack Q3_K blocks into VNNI layout (12B payload + scale + min)
void cpu_pack_q3k(const Q3_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload,
                  std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 12);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

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
            std::memcpy(payload_buf + 8, &hbits_u32, 4);

            std::memcpy(payload.data() + linear * 12, payload_buf, 12);

            // Unpack 6-bit scales
            int8_t unpacked_scales[16];
            {
                const uint32_t kmask1 = 0x03030303;
                const uint32_t kmask2 = 0x0f0f0f0f;
                uint32_t aux[4];
                std::memcpy(aux, blk.scales, 12);
                aux[3] = 0;
                uint32_t tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                std::memcpy(unpacked_scales, aux, 16);
            }

            const int sc_lo_idx = sub_idx * 2;
            const int sc_hi_idx = sub_idx * 2 + 1;
            const float d = cpu_fp16_to_fp32(blk.d);
            scales[linear] = cpu_fp32_to_fp16(d * static_cast<float>(unpacked_scales[sc_lo_idx] - 32));
            mins[linear]   = cpu_fp32_to_fp16(d * static_cast<float>(unpacked_scales[sc_hi_idx] - 32));
        }
    }
}

/// CPU reference: pack Q2_K blocks into VNNI layout (8B payload + scale + min + emins)
void cpu_pack_q2k(const Q2_KBlock* blocks, int N, int K,
                  std::vector<uint8_t>& payload,
                  std::vector<uint16_t>& scales,
                  std::vector<uint16_t>& mins,
                  std::vector<uint32_t>& emins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 8);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);
    emins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

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

            std::memcpy(payload.data() + linear * 8, payload_buf, 8);

            const int sc_lo_idx = sub_idx * 2;
            const int sc_hi_idx = sub_idx * 2 + 1;
            const float d_val = cpu_fp16_to_fp32(blk.d);
            const float dmin  = cpu_fp16_to_fp32(blk.dmin);
            scales[linear] = cpu_fp32_to_fp16(d_val * static_cast<float>(blk.scales[sc_lo_idx] & 0xF));
            mins[linear]   = cpu_fp32_to_fp16(d_val * static_cast<float>(blk.scales[sc_hi_idx] & 0xF));

            uint16_t emb_min_lo = cpu_fp32_to_fp16(-dmin * static_cast<float>(blk.scales[sc_lo_idx] >> 4));
            uint16_t emb_min_hi = cpu_fp32_to_fp16(-dmin * static_cast<float>(blk.scales[sc_hi_idx] >> 4));
            emins[linear] = static_cast<uint32_t>(emb_min_lo) | (static_cast<uint32_t>(emb_min_hi) << 16);
        }
    }
}

/// Fill Q4_0 blocks with deterministic patterns
void fill_q4_0_blocks(Q4_0Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 16; ++j) {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        }
    }
}

/// Fill Q4_K super-blocks with deterministic patterns
void fill_q4k_blocks(Q4_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d    = 0x3800; // 0.5 in FP16
        blocks[i].dmin = 0x3400; // 0.25 in FP16
        // Fill packed scales with known values (6-bit values packed)
        for (int j = 0; j < 12; ++j) {
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 7 + 3) & 0x3F);
        }
        // Fill quantized values with sequential pattern
        for (int j = 0; j < 128; ++j) {
            blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
        }
    }
}

/// Fill Q4_1 blocks with deterministic patterns
void fill_q4_1_blocks(Q4_1Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        blocks[i].m = 0x3800; // 0.5 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
    }
}

/// Fill Q5_0 blocks with deterministic patterns
void fill_q5_0_blocks(Q5_0Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 37) & 0xFF);
    }
}

/// Fill Q5_1 blocks with deterministic patterns
void fill_q5_1_blocks(Q5_1Block* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3C00; // 1.0 in FP16
        blocks[i].m = 0x3400; // 0.25 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 16 + j) & 0xFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 37) & 0xFF);
    }
}

/// Fill Q5_K super-blocks with deterministic patterns
void fill_q5k_blocks(Q5_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d    = 0x3800; // 0.5 in FP16
        blocks[i].dmin = 0x3400; // 0.25 in FP16
        for (int j = 0; j < 12; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 7 + 3) & 0x3F);
        for (int j = 0; j < 32; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 32 + j * 13) & 0xFF);
        for (int j = 0; j < 128; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
    }
}

/// Fill Q6_K super-blocks with deterministic patterns
void fill_q6k_blocks(Q6_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800; // 0.5 in FP16
        for (int j = 0; j < 128; ++j)
            blocks[i].ql[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
        for (int j = 0; j < 64; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i * 64 + j * 7) & 0xFF);
        // Signed int8_t scales — use values in [-50, 50] range
        int8_t* signed_scales = reinterpret_cast<int8_t*>(blocks[i].scales);
        for (int j = 0; j < 16; ++j)
            signed_scales[j] = static_cast<int8_t>(((i + j) * 11 + 5) % 101 - 50);
    }
}

/// Fill Q3_K super-blocks with deterministic patterns
void fill_q3k_blocks(Q3_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800; // 0.5 in FP16
        for (int j = 0; j < 32; ++j)
            blocks[i].hmask[j] = static_cast<uint8_t>((i * 32 + j * 17) & 0xFF);
        for (int j = 0; j < 64; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) & 0xFF);
        for (int j = 0; j < 12; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 7 + 3) & 0xFF);
    }
}

/// Fill Q2_K super-blocks with deterministic patterns
void fill_q2k_blocks(Q2_KBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d    = 0x3800; // 0.5 in FP16
        blocks[i].dmin = 0x3400; // 0.25 in FP16
        for (int j = 0; j < 16; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 11 + 3) & 0xFF);
        for (int j = 0; j < 64; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) & 0xFF);
    }
}

// ============================================================================
// ksigns_iq2xs lookup table (CPU version for reference)
// ============================================================================

static constexpr uint8_t ksigns_iq2xs[128] = {
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
// IQ format fill helpers
// ============================================================================

void fill_iq4xs_blocks(IQ4_XSBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800; // 0.5 in FP16
        blocks[i].scales_h = static_cast<uint16_t>((i * 7 + 1) & 0xFFFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].scales_l[j] = static_cast<uint8_t>(((i + j) * 13 + 5) & 0xFF);
        for (int j = 0; j < 128; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 128 + j) & 0xFF);
    }
}

void fill_iq3s_blocks(IQ3_SBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 64; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) & 0xFF);
        for (int j = 0; j < 8; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 17) & 0xFF);
        for (int j = 0; j < 32; ++j)
            blocks[i].signs[j] = static_cast<uint8_t>((i * 32 + j * 11) & 0xFF);
        for (int j = 0; j < 4; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 19 + 3) & 0xFF);
    }
}

void fill_iq3xxs_blocks(IQ3_XXSBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 96; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 96 + j) & 0xFF);
    }
}

void fill_iq2s_blocks(IQ2_SBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 64; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 64 + j) & 0xFF);
        for (int j = 0; j < 8; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 23) & 0xFF);
        for (int j = 0; j < 8; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 11 + 7) & 0xFF);
    }
}

void fill_iq2xs_blocks(IQ2_XSBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 32; ++j)
            blocks[i].qs[j] = static_cast<uint16_t>(((i * 32 + j) * 257 + 13) & 0xFFFF);
        for (int j = 0; j < 8; ++j)
            blocks[i].scales[j] = static_cast<uint8_t>(((i + j) * 11 + 7) & 0xFF);
    }
}

void fill_iq2xxs_blocks(IQ2_XXSBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 32; ++j)
            blocks[i].qs[j] = static_cast<uint16_t>(((i * 32 + j) * 257 + 11) & 0xFFFF);
    }
}

void fill_iq1s_blocks(IQ1_SBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        blocks[i].d = 0x3800;
        for (int j = 0; j < 32; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 32 + j) & 0xFF);
        for (int j = 0; j < 8; ++j)
            blocks[i].qh[j] = static_cast<uint16_t>(((i + j) * 1031 + 17) & 0xFFFF);
    }
}

void fill_iq1m_blocks(IQ1_MBlock* blocks, int count) {
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < 32; ++j)
            blocks[i].qs[j] = static_cast<uint8_t>((i * 32 + j) & 0xFF);
        for (int j = 0; j < 16; ++j)
            blocks[i].qh[j] = static_cast<uint8_t>((i + j * 19) & 0xFF);
        // scales[8] interpreted as uint16_t[4] — fill so master scale is valid FP16
        // Mask to 0x3FFF to ensure reconstructed scale_u16 ≤ 0x3FFF (no NaN/Inf)
        uint16_t* sc16 = reinterpret_cast<uint16_t*>(blocks[i].scales);
        for (int j = 0; j < 4; ++j)
            sc16[j] = static_cast<uint16_t>(((i + j) * 1031 + 37) & 0x3FFF);
    }
}

// ============================================================================
// IQ format CPU reference pack functions
// ============================================================================

void cpu_pack_iq4xs(const IQ4_XSBlock* blocks, int N, int K,
                    std::vector<uint8_t>& payload,
                    std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 16);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            std::memcpy(payload.data() + linear * 16, blk.qs + sub_idx * 16, 16);

            const int ls = ((blk.scales_l[sub_idx / 2] >> (4 * (sub_idx % 2))) & 0xf)
                         | (((blk.scales_h >> (2 * sub_idx)) & 3) << 4);
            float d_f = cpu_fp16_to_fp32(blk.d);
            scales[linear] = cpu_fp32_to_fp16(d_f * static_cast<float>(ls - 32));
        }
    }
}

void cpu_pack_iq3s(const IQ3_SBlock* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 13);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            uint8_t* dst = payload.data() + linear * 13;
            std::memcpy(dst, blk.qs + sub_idx * 8, 8);
            dst[8] = blk.qh[sub_idx];
            std::memcpy(dst + 9, blk.signs + sub_idx * 4, 4);

            float d_f = cpu_fp16_to_fp32(blk.d);
            uint8_t sc_byte = blk.scales[sub_idx / 2];
            int nibble = (sub_idx & 1) ? (sc_byte >> 4) : (sc_byte & 0xF);
            scales[linear] = cpu_fp32_to_fp16(d_f * static_cast<float>(1 + 2 * nibble));
        }
    }
}

void cpu_pack_iq3xxs(const IQ3_XXSBlock* blocks, int N, int K,
                     std::vector<uint8_t>& payload,
                     std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 12);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            uint8_t pbuf[12];
            std::memcpy(pbuf, blk.qs + sub_idx * 8, 8);

            uint32_t aux32;
            std::memcpy(&aux32, blk.qs + 64 + sub_idx * 4, 4);

            pbuf[8]  = ksigns_iq2xs[(aux32 >>  0) & 127];
            pbuf[9]  = ksigns_iq2xs[(aux32 >>  7) & 127];
            pbuf[10] = ksigns_iq2xs[(aux32 >> 14) & 127];
            pbuf[11] = ksigns_iq2xs[(aux32 >> 21) & 127];

            std::memcpy(payload.data() + linear * 12, pbuf, 12);

            int nibble = static_cast<int>(aux32 >> 28);
            float d_f = cpu_fp16_to_fp32(blk.d);
            scales[linear] = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(nibble)) * 0.5f);
        }
    }
}

void cpu_pack_iq2s(const IQ2_SBlock* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales,
                   std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 9);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            uint8_t* dst = payload.data() + linear * 9;
            std::memcpy(dst, blk.qs + sub_idx * 4, 4);
            dst[4] = blk.qh[sub_idx];
            std::memcpy(dst + 5, blk.qs + 32 + sub_idx * 4, 4);

            float d_f = cpu_fp16_to_fp32(blk.d);
            uint8_t sc = blk.scales[sub_idx];
            scales[linear] = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f);
            mins[linear]   = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f);
        }
    }
}

void cpu_pack_iq2xs(const IQ2_XSBlock* blocks, int N, int K,
                    std::vector<uint8_t>& payload,
                    std::vector<uint16_t>& scales,
                    std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 9);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            uint8_t pbuf[9];
            uint8_t qh_byte = 0;
            for (int l = 0; l < 4; ++l) {
                uint16_t entry = blk.qs[sub_idx * 4 + l];
                pbuf[l] = static_cast<uint8_t>(entry & 0xFF);
                qh_byte |= static_cast<uint8_t>(((entry >> 8) & 1) << l);
                pbuf[5 + l] = ksigns_iq2xs[entry >> 9];
            }
            pbuf[4] = qh_byte;

            std::memcpy(payload.data() + linear * 9, pbuf, 9);

            float d_f = cpu_fp16_to_fp32(blk.d);
            uint8_t sc = blk.scales[sub_idx];
            scales[linear] = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(sc & 0xF)) * 0.25f);
            mins[linear]   = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(sc >> 4)) * 0.25f);
        }
    }
}

void cpu_pack_iq2xxs(const IQ2_XXSBlock* blocks, int N, int K,
                     std::vector<uint8_t>& payload,
                     std::vector<uint16_t>& scales)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 8);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            const uint8_t* qs_bytes = reinterpret_cast<const uint8_t*>(blk.qs);
            uint32_t aux32_0, aux32_1;
            std::memcpy(&aux32_0, qs_bytes + sub_idx * 8, 4);
            std::memcpy(&aux32_1, qs_bytes + sub_idx * 8 + 4, 4);

            uint8_t pbuf[8];
            pbuf[0] = static_cast<uint8_t>(aux32_0);
            pbuf[1] = static_cast<uint8_t>(aux32_0 >> 8);
            pbuf[2] = static_cast<uint8_t>(aux32_0 >> 16);
            pbuf[3] = static_cast<uint8_t>(aux32_0 >> 24);
            pbuf[4] = ksigns_iq2xs[(aux32_1 >>  0) & 127];
            pbuf[5] = ksigns_iq2xs[(aux32_1 >>  7) & 127];
            pbuf[6] = ksigns_iq2xs[(aux32_1 >> 14) & 127];
            pbuf[7] = ksigns_iq2xs[(aux32_1 >> 21) & 127];

            std::memcpy(payload.data() + linear * 8, pbuf, 8);

            int nibble = static_cast<int>(aux32_1 >> 28);
            float d_f = cpu_fp16_to_fp32(blk.d);
            scales[linear] = cpu_fp32_to_fp16(d_f * (0.5f + static_cast<float>(nibble)) * 0.25f);
        }
    }
}

void cpu_pack_iq1s(const IQ1_SBlock* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales,
                   std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 6);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            const uint8_t* qs = blk.qs + sub_idx * 4;
            uint16_t qh_word = blk.qh[sub_idx];

            uint8_t pbuf[6];
            pbuf[0] = qs[0]; pbuf[1] = qs[1]; pbuf[2] = qs[2]; pbuf[3] = qs[3];
            pbuf[4] = static_cast<uint8_t>(qh_word & 0xFF);
            pbuf[5] = static_cast<uint8_t>((qh_word >> 8) & 0xFF);

            std::memcpy(payload.data() + linear * 6, pbuf, 6);

            float d_f = cpu_fp16_to_fp32(blk.d);
            int scale_sel = (qh_word >> 12) & 7;
            float dl = d_f * (2.0f * static_cast<float>(scale_sel) + 1.0f);

            constexpr float IQ1S_DELTA = 0.125f;
            float delta = (qh_word & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;

            scales[linear] = cpu_fp32_to_fp16(dl);
            mins[linear]   = cpu_fp32_to_fp16(dl * delta);
        }
    }
}

void cpu_pack_iq1m(const IQ1_MBlock* blocks, int N, int K,
                   std::vector<uint8_t>& payload,
                   std::vector<uint16_t>& scales,
                   std::vector<uint16_t>& mins)
{
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    payload.resize(static_cast<size_t>(blocks_per_row) * N * 6);
    scales.resize(static_cast<size_t>(blocks_per_row) * N);
    mins.resize(static_cast<size_t>(blocks_per_row) * N);

    for (int n = 0; n < N; ++n) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const size_t linear = static_cast<size_t>(b) * N + n;
            const int sb_idx  = b / 8;
            const int sub_idx = b % 8;
            const auto& blk = blocks[n * sb_per_row + sb_idx];

            const uint8_t* qs = blk.qs + sub_idx * 4;
            const uint8_t* qh = blk.qh + sub_idx * 2;

            uint8_t pbuf[6];
            pbuf[0] = qs[0]; pbuf[1] = qs[1]; pbuf[2] = qs[2]; pbuf[3] = qs[3];
            pbuf[4] = qh[0]; pbuf[5] = qh[1];

            std::memcpy(payload.data() + linear * 6, pbuf, 6);

            const uint16_t* sc = reinterpret_cast<const uint16_t*>(blk.scales);
            uint16_t scale_u16 = static_cast<uint16_t>(
                (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000));
            float d_val = cpu_fp16_to_fp32(scale_u16);

            int sc_word_idx = sub_idx / 2;
            int sc_bit_offset = 6 * (sub_idx % 2);
            int sc3_lo = (sc[sc_word_idx] >> (sc_bit_offset + 0)) & 0x7;
            int sc3_hi = (sc[sc_word_idx] >> (sc_bit_offset + 3)) & 0x7;
            float dl1 = d_val * (2.0f * static_cast<float>(sc3_lo) + 1.0f);
            float dl2 = d_val * (2.0f * static_cast<float>(sc3_hi) + 1.0f);

            scales[linear] = cpu_fp32_to_fp16(dl1);
            mins[linear]   = cpu_fp32_to_fp16(dl2);
        }
    }
}

/// Compare two FP16 values with 1-ULP tolerance, treating ±0 as equal
inline bool fp16_approx_equal(uint16_t a, uint16_t b) {
    // Both are zero (±0.0 in FP16: 0x0000 and 0x8000 are both zero)
    if ((a & 0x7FFF) == 0 && (b & 0x7FFF) == 0) return true;
    int diff = std::abs(static_cast<int>(a) - static_cast<int>(b));
    return diff <= 1;
}

} // anonymous namespace

// ============================================================================
// Test fixture
// ============================================================================

class Test__VnniRepackKernels : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        if (err != hipSuccess || count == 0) {
            GTEST_SKIP() << "No ROCm devices available";
        }
        (void)hipSetDevice(0);
#else
        GTEST_SKIP() << "HAVE_ROCM not defined";
#endif
    }
};

// ============================================================================
// Helper: GPU alloc/upload/download/free RAII wrapper
// ============================================================================

#ifdef HAVE_ROCM

template<typename T>
struct GpuBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    GpuBuffer() = default;

    explicit GpuBuffer(size_t n) : count(n) {
        if (n > 0) {
            hipError_t err = hipMalloc(&ptr, n * sizeof(T));
            EXPECT_EQ(err, hipSuccess) << "hipMalloc failed: " << hipGetErrorString(err);
        }
    }

    ~GpuBuffer() {
        if (ptr) (void)hipFree(ptr);
    }

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    void upload(const T* host_data) {
        (void)hipMemcpy(ptr, host_data, count * sizeof(T), hipMemcpyHostToDevice);
    }

    void download(T* host_data) const {
        (void)hipMemcpy(host_data, ptr, count * sizeof(T), hipMemcpyDeviceToHost);
    }

    void upload(const std::vector<T>& v) { upload(v.data()); }

    void download(std::vector<T>& v) const {
        v.resize(count);
        download(v.data());
    }
};

#endif // HAVE_ROCM

// ============================================================================
// Test: Q4_0 parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q4_0_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32; // 4
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // Create synthetic blocks
    std::vector<Q4_0Block> host_blocks(total_blocks);
    fill_q4_0_blocks(host_blocks.data(), total_blocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_q4_0(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    // GPU repack
    GpuBuffer<Q4_0Block> d_blocks(total_blocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 16);
    GpuBuffer<uint16_t>  d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q4_0, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    // Download and compare
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_0 payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    EXPECT_EQ(gpu_scales, cpu_scales) << "Q4_0 scales mismatch";
#endif
}

// ============================================================================
// Test: IQ4_NL parity (identical block layout to Q4_0)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ4_NL_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32;
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // IQ4_NLBlock has identical layout to Q4_0Block (18 bytes: d + qs[16])
    // Create as Q4_0Block, cast to IQ4_NLBlock for the upload
    std::vector<Q4_0Block> host_blocks(total_blocks);
    fill_q4_0_blocks(host_blocks.data(), total_blocks);

    // CPU reference (same packing logic as Q4_0)
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_q4_0(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    // GPU repack with IQ4_NL format enum
    GpuBuffer<Q4_0Block> d_blocks(total_blocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 16);
    GpuBuffer<uint16_t>  d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ4_NL, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ4_NL payload mismatch";
    EXPECT_EQ(gpu_scales, cpu_scales) << "IQ4_NL scales mismatch";
#endif
}

// ============================================================================
// Test: Q4_K parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q4_K_Parity) {
#ifdef HAVE_ROCM
    const int N = 32;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // Create synthetic super-blocks
    std::vector<Q4_KBlock> host_blocks(total_superblocks);
    fill_q4k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q4k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q4_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 16);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q4_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    // Payload should match byte-for-byte (integer nibble manipulation)
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_K payload mismatch";

    // Scales and mins: allow 1 ULP FP16 difference due to GPU vs CPU rounding
    // Also treats ±0.0 as equal (GPU may fold -0.0 → +0.0)
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0;
    int min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5) {
                ADD_FAILURE() << "Q4_K scale mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
            }
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total Q4_K scale mismatches: " << scale_mismatches;

    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5) {
                ADD_FAILURE() << "Q4_K min mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
            }
        }
    }
    EXPECT_EQ(min_mismatches, 0) << "Total Q4_K min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: Q4_K with larger (realistic) dimensions
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q4_K_LargerDimensions) {
#ifdef HAVE_ROCM
    const int N = 256;
    const int K = 2048;
    const int blocks_per_row = K / 32; // 64
    const int sb_per_row = (K + 255) / 256; // 8
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<Q4_KBlock> host_blocks(total_superblocks);
    fill_q4k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q4k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q4_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 16);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q4_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    // Payload: byte-for-byte match
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_K large payload mismatch";

    // Scales/mins: allow 1 ULP FP16 tolerance, ±0 treated as equal
    int mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) mismatches++;
    }
    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) mismatches++;
    }
    EXPECT_EQ(mismatches, 0) << "Q4_K large dimension scale/min mismatches: " << mismatches;
#endif
}

// ============================================================================
// Test: Q4_1 parity (asymmetric 4-bit with min)
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q4_1_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32; // 4
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // Create synthetic blocks
    std::vector<Q4_1Block> host_blocks(total_blocks);
    fill_q4_1_blocks(host_blocks.data(), total_blocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q4_1(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q4_1Block> d_blocks(total_blocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 16);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q4_1, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    // Download and compare
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q4_1 payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    EXPECT_EQ(gpu_scales, cpu_scales) << "Q4_1 scales mismatch";

    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());
    EXPECT_EQ(gpu_mins, cpu_mins) << "Q4_1 mins mismatch";
#endif
}

// ============================================================================
// Test: Q5_0 parity (symmetric 5-bit)
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q5_0_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32; // 4
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // Create synthetic blocks
    std::vector<Q5_0Block> host_blocks(total_blocks);
    fill_q5_0_blocks(host_blocks.data(), total_blocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_q5_0(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    // GPU repack
    GpuBuffer<Q5_0Block> d_blocks(total_blocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 20);
    GpuBuffer<uint16_t>  d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q5_0, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    // Download and compare
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q5_0 payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    EXPECT_EQ(gpu_scales, cpu_scales) << "Q5_0 scales mismatch";
#endif
}

// ============================================================================
// Test: Q5_1 parity (asymmetric 5-bit with min)
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q5_1_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 128;
    const int blocks_per_row = K / 32; // 4
    const int total_blocks = N * blocks_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    // Create synthetic blocks
    std::vector<Q5_1Block> host_blocks(total_blocks);
    fill_q5_1_blocks(host_blocks.data(), total_blocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q5_1(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q5_1Block> d_blocks(total_blocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 20);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q5_1, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    // Download and compare
    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q5_1 payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    EXPECT_EQ(gpu_scales, cpu_scales) << "Q5_1 scales mismatch";

    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());
    EXPECT_EQ(gpu_mins, cpu_mins) << "Q5_1 mins mismatch";
#endif
}

// ============================================================================
// Test: Q5_K parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q5_K_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<Q5_KBlock> host_blocks(total_superblocks);
    fill_q5k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q5k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q5_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 20);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q5_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    // Payload: byte-for-byte match
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q5_K payload mismatch";

    // Scales/mins: allow 1 ULP FP16 tolerance
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0;
    int min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "Q5_K scale mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total Q5_K scale mismatches: " << scale_mismatches;

    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "Q5_K min mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(min_mismatches, 0) << "Total Q5_K min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: Q6_K parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q6_K_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<Q6_KBlock> host_blocks(total_superblocks);
    fill_q6k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q6k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q6_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 24);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q6_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    // Payload: byte-for-byte match
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q6_K payload mismatch";

    // Scales/mins: allow 1 ULP FP16 tolerance
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0;
    int min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "Q6_K scale mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total Q6_K scale mismatches: " << scale_mismatches;

    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "Q6_K min mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(min_mismatches, 0) << "Total Q6_K min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: Q3_K parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q3_K_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<Q3_KBlock> host_blocks(total_superblocks);
    fill_q3k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_q3k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    // GPU repack
    GpuBuffer<Q3_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 12);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q3_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    // Payload: byte-for-byte match
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q3_K payload mismatch";

    // Scales/mins: allow 1 ULP FP16 tolerance
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0;
    int min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "Q3_K scale mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total Q3_K scale mismatches: " << scale_mismatches;

    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "Q3_K min mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(min_mismatches, 0) << "Total Q3_K min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: Q2_K parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, Q2_K_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<Q2_KBlock> host_blocks(total_superblocks);
    fill_q2k_blocks(host_blocks.data(), total_superblocks);

    // CPU reference
    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    std::vector<uint32_t> cpu_emins;
    cpu_pack_q2k(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins, cpu_emins);

    // GPU repack — uses 8-param overload with d_emins
    GpuBuffer<Q2_KBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>   d_payload(total_output * 8);
    GpuBuffer<uint16_t>  d_scales(total_output);
    GpuBuffer<uint16_t>  d_mins(total_output);
    GpuBuffer<uint32_t>  d_emins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::Q2_K, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               d_emins.ptr, N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    std::vector<uint32_t> gpu_emins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);
    d_emins.download(gpu_emins);

    // Payload: byte-for-byte match
    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "Q2_K payload mismatch";

    // Scales/mins/emins: allow 1 ULP FP16 tolerance
    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());
    ASSERT_EQ(gpu_emins.size(), cpu_emins.size());

    int scale_mismatches = 0;
    int min_mismatches = 0;
    int emin_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "Q2_K scale mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total Q2_K scale mismatches: " << scale_mismatches;

    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "Q2_K min mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(min_mismatches, 0) << "Total Q2_K min mismatches: " << min_mismatches;

    for (size_t i = 0; i < cpu_emins.size(); ++i) {
        // Compare two packed FP16 values in each uint32_t
        uint16_t gpu_lo = static_cast<uint16_t>(gpu_emins[i] & 0xFFFF);
        uint16_t gpu_hi = static_cast<uint16_t>(gpu_emins[i] >> 16);
        uint16_t cpu_lo = static_cast<uint16_t>(cpu_emins[i] & 0xFFFF);
        uint16_t cpu_hi = static_cast<uint16_t>(cpu_emins[i] >> 16);
        if (!fp16_approx_equal(gpu_lo, cpu_lo) || !fp16_approx_equal(gpu_hi, cpu_hi)) {
            emin_mismatches++;
            if (emin_mismatches <= 5)
                ADD_FAILURE() << "Q2_K emin mismatch at index " << i
                              << ": GPU=0x" << std::hex << gpu_emins[i]
                              << " CPU=0x" << cpu_emins[i] << std::dec;
        }
    }
    EXPECT_EQ(emin_mismatches, 0) << "Total Q2_K emin mismatches: " << emin_mismatches;
#endif
}

// ============================================================================
// Test: IQ4_XS parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ4_XS_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32; // 8
    const int sb_per_row = (K + 255) / 256; // 1
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ4_XSBlock> host_blocks(total_superblocks);
    fill_iq4xs_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_iq4xs(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    GpuBuffer<IQ4_XSBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>     d_payload(total_output * 16);
    GpuBuffer<uint16_t>    d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ4_XS, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ4_XS payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    int mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            mismatches++;
            if (mismatches <= 5)
                ADD_FAILURE() << "IQ4_XS scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Total IQ4_XS scale mismatches: " << mismatches;
#endif
}

// ============================================================================
// Test: IQ3_S parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ3_S_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ3_SBlock> host_blocks(total_superblocks);
    fill_iq3s_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_iq3s(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    GpuBuffer<IQ3_SBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>    d_payload(total_output * 13);
    GpuBuffer<uint16_t>   d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ3_S, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ3_S payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    int mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            mismatches++;
            if (mismatches <= 5)
                ADD_FAILURE() << "IQ3_S scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Total IQ3_S scale mismatches: " << mismatches;
#endif
}

// ============================================================================
// Test: IQ3_XXS parity
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ3_XXS_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ3_XXSBlock> host_blocks(total_superblocks);
    fill_iq3xxs_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_iq3xxs(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    GpuBuffer<IQ3_XXSBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>      d_payload(total_output * 12);
    GpuBuffer<uint16_t>     d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ3_XXS, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ3_XXS payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    int mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            mismatches++;
            if (mismatches <= 5)
                ADD_FAILURE() << "IQ3_XXS scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Total IQ3_XXS scale mismatches: " << mismatches;
#endif
}

// ============================================================================
// Test: IQ2_S parity (asymmetric)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ2_S_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ2_SBlock> host_blocks(total_superblocks);
    fill_iq2s_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_iq2s(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    GpuBuffer<IQ2_SBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>    d_payload(total_output * 9);
    GpuBuffer<uint16_t>   d_scales(total_output);
    GpuBuffer<uint16_t>   d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ2_S, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ2_S payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0, min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "IQ2_S scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "IQ2_S min mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total IQ2_S scale mismatches: " << scale_mismatches;
    EXPECT_EQ(min_mismatches, 0) << "Total IQ2_S min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: IQ2_XS parity (asymmetric, uses ksigns)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ2_XS_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ2_XSBlock> host_blocks(total_superblocks);
    fill_iq2xs_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_iq2xs(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    GpuBuffer<IQ2_XSBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>     d_payload(total_output * 9);
    GpuBuffer<uint16_t>    d_scales(total_output);
    GpuBuffer<uint16_t>    d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ2_XS, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ2_XS payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0, min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "IQ2_XS scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "IQ2_XS min mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total IQ2_XS scale mismatches: " << scale_mismatches;
    EXPECT_EQ(min_mismatches, 0) << "Total IQ2_XS min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: IQ2_XXS parity (symmetric, uses ksigns)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ2_XXS_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ2_XXSBlock> host_blocks(total_superblocks);
    fill_iq2xxs_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    cpu_pack_iq2xxs(host_blocks.data(), N, K, cpu_payload, cpu_scales);

    GpuBuffer<IQ2_XXSBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>      d_payload(total_output * 8);
    GpuBuffer<uint16_t>     d_scales(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ2_XXS, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, nullptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ2_XXS payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    int mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            mismatches++;
            if (mismatches <= 5)
                ADD_FAILURE() << "IQ2_XXS scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    EXPECT_EQ(mismatches, 0) << "Total IQ2_XXS scale mismatches: " << mismatches;
#endif
}

// ============================================================================
// Test: IQ1_S parity (asymmetric)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ1_S_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ1_SBlock> host_blocks(total_superblocks);
    fill_iq1s_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_iq1s(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    GpuBuffer<IQ1_SBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>    d_payload(total_output * 6);
    GpuBuffer<uint16_t>   d_scales(total_output);
    GpuBuffer<uint16_t>   d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ1_S, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ1_S payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0, min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "IQ1_S scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "IQ1_S min mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total IQ1_S scale mismatches: " << scale_mismatches;
    EXPECT_EQ(min_mismatches, 0) << "Total IQ1_S min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: IQ1_M parity (asymmetric)
// ============================================================================

TEST_F(Test__VnniRepackKernels, IQ1_M_Parity) {
#ifdef HAVE_ROCM
    const int N = 64;
    const int K = 256;
    const int blocks_per_row = K / 32;
    const int sb_per_row = (K + 255) / 256;
    const int total_superblocks = N * sb_per_row;
    const size_t total_output = static_cast<size_t>(blocks_per_row) * N;

    std::vector<IQ1_MBlock> host_blocks(total_superblocks);
    fill_iq1m_blocks(host_blocks.data(), total_superblocks);

    std::vector<uint8_t> cpu_payload;
    std::vector<uint16_t> cpu_scales;
    std::vector<uint16_t> cpu_mins;
    cpu_pack_iq1m(host_blocks.data(), N, K, cpu_payload, cpu_scales, cpu_mins);

    GpuBuffer<IQ1_MBlock> d_blocks(total_superblocks);
    GpuBuffer<uint8_t>    d_payload(total_output * 6);
    GpuBuffer<uint16_t>   d_scales(total_output);
    GpuBuffer<uint16_t>   d_mins(total_output);
    d_blocks.upload(host_blocks.data());

    bool ok = launchVnniRepack(RepackFormat::IQ1_M, d_blocks.ptr,
                               d_payload.ptr, d_scales.ptr, d_mins.ptr,
                               N, K, nullptr);
    ASSERT_TRUE(ok);
    (void)hipDeviceSynchronize();

    std::vector<uint8_t> gpu_payload;
    std::vector<uint16_t> gpu_scales;
    std::vector<uint16_t> gpu_mins;
    d_payload.download(gpu_payload);
    d_scales.download(gpu_scales);
    d_mins.download(gpu_mins);

    ASSERT_EQ(gpu_payload.size(), cpu_payload.size());
    EXPECT_EQ(gpu_payload, cpu_payload) << "IQ1_M payload mismatch";

    ASSERT_EQ(gpu_scales.size(), cpu_scales.size());
    ASSERT_EQ(gpu_mins.size(), cpu_mins.size());

    int scale_mismatches = 0, min_mismatches = 0;
    for (size_t i = 0; i < cpu_scales.size(); ++i) {
        if (!fp16_approx_equal(gpu_scales[i], cpu_scales[i])) {
            scale_mismatches++;
            if (scale_mismatches <= 5)
                ADD_FAILURE() << "IQ1_M scale mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_scales[i]
                              << " CPU=0x" << cpu_scales[i] << std::dec;
        }
    }
    for (size_t i = 0; i < cpu_mins.size(); ++i) {
        if (!fp16_approx_equal(gpu_mins[i], cpu_mins[i])) {
            min_mismatches++;
            if (min_mismatches <= 5)
                ADD_FAILURE() << "IQ1_M min mismatch at " << i
                              << ": GPU=0x" << std::hex << gpu_mins[i]
                              << " CPU=0x" << cpu_mins[i] << std::dec;
        }
    }
    EXPECT_EQ(scale_mismatches, 0) << "Total IQ1_M scale mismatches: " << scale_mismatches;
    EXPECT_EQ(min_mismatches, 0) << "Total IQ1_M min mismatches: " << min_mismatches;
#endif
}

// ============================================================================
// Test: Invalid format returns false
// ============================================================================

TEST_F(Test__VnniRepackKernels, InvalidFormatReturnsFalse) {
#ifdef HAVE_ROCM
    // Use an unrecognized format value (cast from an unused enum value)
    auto bad_format = static_cast<RepackFormat>(255);
    bool ok = launchVnniRepack(bad_format, nullptr, nullptr, nullptr, nullptr,
                               1, 32, nullptr);
    EXPECT_FALSE(ok);
#endif
}

} // namespace llaminar2
