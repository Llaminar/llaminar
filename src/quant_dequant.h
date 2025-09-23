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

namespace llaminar
{

    // Minimal half -> float conversion (delegated in model_loader but replicated here for header-only use)
    static inline float qd_fp16_to_fp32(uint16_t h)
    {
        uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
        uint32_t exp = (h & 0x7C00u) >> 10;
        uint32_t mant = (h & 0x03FFu);
        uint32_t bits;
        if (exp == 0)
        {
            if (mant == 0)
            {
                bits = sign;
            }
            else
            {
                while ((mant & 0x0400u) == 0)
                {
                    mant <<= 1;
                    --exp;
                }
                ++exp;
                mant &= 0x03FFu;
                exp = exp + (127 - 15);
                bits = sign | (exp << 23) | (mant << 13);
            }
        }
        else if (exp == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (mant << 13);
        }
        else
        {
            exp = exp + (127 - 15);
            bits = sign | (exp << 23) | (mant << 13);
        }
        union
        {
            uint32_t u;
            float f;
        } u = {bits};
        return u.f;
    }

    // Q5_0: block size 32 values; layout: uint16_t d; uint8_t qh[4]; uint8_t qs[16]
    // Reconstruction mirrors ModelLoader::dequantizeQ5_0 (ggml parity):
    // raw5 = (low_nibble | ( (qh_bit)<<4)); signed_val = raw5 - 16; value = signed_val * d
    inline void dequant_block_q5_0(const uint8_t *block, float *dst, int n_vals = 32)
    {
        const int QK = 32;
        if (n_vals != QK) ; // allow alternative but assume 32 for indexing
        uint16_t hd; std::memcpy(&hd, block, 2);
        float d = qd_fp16_to_fp32(hd);
        uint32_t qh; std::memcpy(&qh, block + 2, 4); // qh[4] packed into 32 bits
        const uint8_t *qs = block + 6;
        for (int j = 0; j < QK/2; ++j) {
            // Match ggml high-bit placement (second half uses +12 offset)
            const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
            const uint8_t xh_1 = ((qh >> (j + 12))) & 0x10;
            const uint8_t q = qs[j];
            const int32_t x0 = ((q & 0x0F) | xh_0) - 16;
            const int32_t x1 = ((q >> 4) | xh_1) - 16;
            dst[j + 0] = x0 * d;
            dst[j + QK/2] = x1 * d;
        }
    }

    // Q2_K (256 vals) simplified dequant replication for fused streaming.
    inline void dequant_block_q2_K(const uint8_t *block, float *dst)
    {
        // Layout aligns with model_loader logic; reuse that by calling into existing path would copy.
        // For now, call through a minimal subset duplication not to depend on internal static lambdas.
        // We can fallback to constructing temp vector via model loader if needed; omitted for brevity.
        // (Phase 2 optimization placeholder) -- for Phase 1 fused path we may initially skip Q2_K.
        (void)block;
        (void)dst; // TODO: full inline implementation if required.
    }

    // Q3_K, Q5_K, Q6_K are more complex; to keep initial fused path safe we will
    // only support Q5_0 and direct F32/F16 in first iteration unless explicitly enabled later.

} // namespace llaminar
