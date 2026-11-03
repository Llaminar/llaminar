/**
 * @file DeepSeekV4Math.h
 * @brief Host/device scalar arithmetic shared by every DeepSeek-V4 backend.
 *
 * Keeping the RoPE frequency, Sinkhorn, softplus and Q8_0 decode definitions in
 * one header gives CPU and CUDA the same operation order. Functions are
 * `inline` for host builds and additionally `__device__` when compiled by a
 * native GPU compiler.
 */
#pragma once

#include "DeepSeekV4Kernels.h"
#include <cmath>
#include <cstdint>
#include <cstring>


namespace llaminar2::deepseek4::math
{
/** @brief Frequency of rotation pair @p pair under @p policy (DeepSeek precompute_freqs_cis). */
DSV4_HD float ropeFrequency(const RopePolicy &policy, int pair)
{
    const float exponent = -2.0f * static_cast<float>(pair) / static_cast<float>(policy.ropeDim);
    float freq = powf(policy.base, exponent);
    if (policy.yarn)
    {
        float ramp = (static_cast<float>(pair) - policy.rampLow) / (policy.rampHigh - policy.rampLow);
        ramp = fminf(fmaxf(ramp, 0.0f), 1.0f);
        const float smooth = 1.0f - ramp;
        freq = freq / policy.factor * (1.0f - smooth) + freq * smooth;
    }
    return freq;
}

/**
 * @brief Rotate the trailing rope channels of one vector in place.
 * @param data Vector of @p width channels; the last policy.ropeDim are rotated.
 * @param position Absolute token (or block) position.
 * @param inverse Rotate by -angle.
 */
DSV4_HD void rotateTail(float *data, int width, const RopePolicy &policy, int position, bool inverse)
{
    float *tail = data + (width - policy.ropeDim);
    for (int pair = 0; pair < policy.ropeDim / 2; ++pair)
    {
        const float angle = static_cast<float>(position) * ropeFrequency(policy, pair);
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const float x0 = tail[2 * pair];
        const float x1 = tail[2 * pair + 1];
        tail[2 * pair] = x0 * c - x1 * s;
        tail[2 * pair + 1] = x0 * s + x1 * c;
    }
}

/** @brief Logistic function in FP32. */
DSV4_HD float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

/** @brief sqrt(softplus(x)) with the PyTorch softplus threshold (20). */
DSV4_HD float sqrtSoftplus(float value)
{
    const float sp = value > 20.0f ? value : log1pf(expf(value));
    return sqrtf(sp);
}

/** @brief IEEE half to float (Q8_0 block scale). */
DSV4_HD float halfToFloat(std::uint16_t h)
{
    const std::uint32_t sign = (h & 0x8000u) << 16;
    std::uint32_t exponent = (h >> 10) & 0x1Fu;
    std::uint32_t mantissa = h & 0x3FFu;
    std::uint32_t bits;
    if (exponent == 0)
    {
        if (mantissa == 0)
            bits = sign;
        else
        {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0)
            {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 31)
        bits = sign | 0x7F800000u | (mantissa << 13);
    else
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/** @brief BF16 payload to float. */
DSV4_HD float bf16ToFloat(std::uint16_t v)
{
    const std::uint32_t bits = static_cast<std::uint32_t>(v) << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

/** @brief Bytes of one Q8_0 block (fp16 scale + 32 int8). */
inline constexpr int kQ8BlockBytes = 34;

/** @brief Divide every column of [src j][dst k] by (its sum + eps). */
DSV4_HD void normalizeColumns(float *comb, int hc, float eps)
{
    for (int k = 0; k < hc; ++k)
    {
        float sum = 0.0f;
        for (int j = 0; j < hc; ++j) sum += comb[j * hc + k];
        const float d = sum + eps;
        for (int j = 0; j < hc; ++j) comb[j * hc + k] /= d;
    }
}

/**
 * @brief Sinkhorn-normalize one hc x hc combination in place (DeepSeek hc_split_sinkhorn).
 * @param comb Row-major [src j][dst k] logits already affine-transformed.
 * @details Row softmax over k, + eps, then column normalization over j, then
 * (iters - 1) rounds of row then column normalization, each with + eps.
 */
DSV4_HD void sinkhorn(float *comb, int hc, int iters, float eps)
{
    for (int j = 0; j < hc; ++j)
    {
        float *row = comb + j * hc;
        float m = row[0];
        for (int k = 1; k < hc; ++k) m = fmaxf(m, row[k]);
        float sum = 0.0f;
        for (int k = 0; k < hc; ++k)
        {
            row[k] = expf(row[k] - m);
            sum += row[k];
        }
        for (int k = 0; k < hc; ++k) row[k] = row[k] / sum + eps;
    }
    normalizeColumns(comb, hc, eps);
    for (int it = 1; it < iters; ++it)
    {
        for (int j = 0; j < hc; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < hc; ++k) sum += comb[j * hc + k];
            const float d = sum + eps;
            for (int k = 0; k < hc; ++k) comb[j * hc + k] /= d;
        }
        normalizeColumns(comb, hc, eps);
    }
}
} // namespace llaminar2::deepseek4::math
