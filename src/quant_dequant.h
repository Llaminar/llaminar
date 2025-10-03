// Reusable inline dequantization helpers for fused COSMA weight streaming.
// Extracted from ModelLoader implementations to avoid duplication when populating
// distributed COSMA buffers directly from quantized GGUF blocks.
// NOTE: These are performance-oriented and assume validated input sizes.

#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>

#ifndef GGML_COMMON_DECL
#define GGML_COMMON_DECL_CPP
#endif
#include "../llama.cpp/ggml/src/ggml-common.h"
#undef GGML_COMMON_DECL_CPP
#include "../llama.cpp/ggml/src/ggml-quants.h"

extern "C"
{
    void dequantize_row_q2_K(const block_q2_K *x, float *y, int64_t k);
}

namespace llaminar
{

    static inline float qd_fp32_from_bits(uint32_t w)
    {
        union
        {
            uint32_t as_bits;
            float as_value;
        } fp32;
        fp32.as_bits = w;
        return fp32.as_value;
    }

    static inline uint32_t qd_fp32_to_bits(float f)
    {
        union
        {
            float as_value;
            uint32_t as_bits;
        } fp32;
        fp32.as_value = f;
        return fp32.as_bits;
    }

    // Minimal half -> float conversion aligned with ggml implementation
    static inline float qd_fp16_to_fp32(uint16_t h)
    {
        const uint32_t w = static_cast<uint32_t>(h) << 16;
        const uint32_t sign = w & UINT32_C(0x80000000);
        const uint32_t two_w = w + w;

        const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if (defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)) && (!defined(__cplusplus) || __cplusplus >= 201703L)
        const float exp_scale = 0x1.0p-112f;
#else
        const float exp_scale = qd_fp32_from_bits(UINT32_C(0x7800000));
#endif
        const float normalized_value = qd_fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

        const uint32_t magic_mask = UINT32_C(126) << 23;
        const float magic_bias = 0.5f;
        const float denormalized_value = qd_fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

        const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
        const uint32_t result = sign |
                                (two_w < denormalized_cutoff ? qd_fp32_to_bits(denormalized_value) : qd_fp32_to_bits(normalized_value));
        return qd_fp32_from_bits(result);
    }

    // Q4_0: block size 32 values; layout: uint16_t d; uint8_t qs[16] (packed low/high nibbles)
    inline void dequant_block_q4_0(const uint8_t *block, float *dst, int values = 32)
    {
        constexpr int QK = 32;
        if (!block || !dst || values <= 0)
            return;
        uint16_t scale_bits = 0;
        std::memcpy(&scale_bits, block, sizeof(uint16_t));
        // Use upstream macro GGML_FP16_TO_FP32 for exact parity with ggml's dequantize_row_q4_0
        const float d = ggml_fp16_to_fp32(scale_bits);
        const uint8_t *qs = block + sizeof(uint16_t);
        const int half = QK / 2;
        for (int j = 0; j < half; ++j)
        {
            const uint8_t packed = qs[j];
            const int x0i = j;        // first half position
            const int x1i = j + half; // second half position
            if (x0i < values)
                dst[x0i] = (float)((packed & 0x0F) - 8) * d;
            if (x1i < values)
                dst[x1i] = (float)((packed >> 4) - 8) * d;
        }
    }

    inline void dequant_q4_0_rows(const uint8_t *data, float *dst, size_t n_elements)
    {
        if (!data || !dst || n_elements == 0)
            return;
        constexpr size_t QK = 32;
        constexpr size_t BLOCK_BYTES = sizeof(uint16_t) + 16;
        const size_t blocks = (n_elements + QK - 1) / QK;
        for (size_t b = 0; b < blocks; ++b)
        {
            const uint8_t *block = data + b * BLOCK_BYTES;
            float *row = dst + b * QK;
            const size_t remain = std::min<size_t>(QK, n_elements - b * QK);
            dequant_block_q4_0(block, row, static_cast<int>(remain));
        }
    }

    // Q5_0: block size 32 values; layout: uint16_t d; uint8_t qh[4]; uint8_t qs[16]
    // Reconstruction mirrors ModelLoader::dequantizeQ5_0 (ggml parity):
    // raw5 = (low_nibble | ( (qh_bit)<<4)); signed_val = raw5 - 16; value = signed_val * d
    inline void dequant_block_q5_0(const uint8_t *block, float *dst, int n_vals = 32)
    {
        const int QK = 32;
        if (n_vals != QK)
            ; // allow alternative but assume 32 for indexing
        uint16_t hd;
        std::memcpy(&hd, block, 2);
        float d = qd_fp16_to_fp32(hd);
        uint32_t qh;
        std::memcpy(&qh, block + 2, 4); // qh[4] packed into 32 bits
        const uint8_t *qs = block + 6;
        for (int j = 0; j < QK / 2; ++j)
        {
            // Match ggml high-bit placement (second half uses +12 offset)
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const uint8_t q = qs[j];
            const int32_t x0 = ((q & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((q >> 4) | xh_1) - 16;
            dst[j + 0] = x0 * d;
            dst[j + QK / 2] = x1 * d;
        }
    }

    // Q2_K (256 vals) simplified dequant replication for fused streaming.
    inline void dequant_block_q2_K(const uint8_t *block, float *dst, int values = 256)
    {
        if (!block || !dst || values <= 0)
            return;

        block_q2_K blk{};
        std::memcpy(&blk, block, sizeof(block_q2_K));

        if (values >= QK_K)
        {
            dequantize_row_q2_K(&blk, dst, QK_K);
        }
        else
        {
            float tmp[QK_K];
            dequantize_row_q2_K(&blk, tmp, QK_K);
            std::memcpy(dst, tmp, sizeof(float) * static_cast<size_t>(values));
        }
    }

    // Q3_K, Q5_K, Q6_K are more complex; to keep initial fused path safe we will
    // only support Q5_0 and direct F32/F16 in first iteration unless explicitly enabled later.

} // namespace llaminar
