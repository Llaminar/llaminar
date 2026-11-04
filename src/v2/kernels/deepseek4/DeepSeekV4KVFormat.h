/**
 * @file DeepSeekV4KVFormat.h
 * @brief Packed DeepSeek-V4 persistent KV formats and the official QAT rounding.
 *
 * DeepSeek-V4 is quantization-aware trained: inference/model.py rounds the
 * stored attention state through low-precision formats before any reader
 * sees it (kernel.py act_quant / fp4_act_quant with inplace=True):
 *
 *  - Window kv and the main compressor output: the leading
 *    headDim - ropeDim ("nope") channels pass through FP8 E4M3 in blocks of 64
 *    with a power-of-two (UE8M0) scale 2^ceil(log2(max(amax, 1e-4) / 448));
 *    the trailing rope channels stay in BF16 (the official activation dtype).
 *  - Indexer queries and indexer compressor output: a Hadamard rotation scaled
 *    by d^-0.5, then FP4 E2M1 in blocks of 32 with the E8M0 scale
 *    2^ceil(log2(max(amax, 6 * 2^-126) / 6)).
 *
 * Because the QAT values are exactly representable, the persistent banks can
 * store them losslessly in the compact vLLM-style records described here:
 *
 *   KV entry (fp8_ds_mla):  [nope FP8 bytes][rope BF16 x ropeDim][UE8M0 x nope/64][pad to 8]
 *                           head_dim 512, rope 64 -> 448 + 128 + 7 + 1 = 584 bytes
 *   Indexer key (MXFP4):    [E2M1 nibbles, element 2i low / 2i+1 high][E8M0 x dim/32][pad to 4]
 *                           index_head_dim 128 -> 64 + 4 = 68 bytes
 *
 * Packing quantizes from FP32; decoding is exact. Packing an already-QAT
 * vector reproduces exactly the same values (the scale may be re-expressed
 * as half the original when a block's maximum rounds down to 224, but every
 * value is unchanged), so a QAT activation and its packed record agree bitwise.
 *
 * Every function is FP32/integer arithmetic with explicit round-to-nearest-
 * even, shared by CPU and CUDA so both backends produce identical bytes,
 * including IEEE subnormals (the device side is immune to CUDA FTZ; see mulRn). Scale exponents follow kernel.py fast_round_scale bit-for-bit,
 * including the FP32 rounding of the constants 1/448 and 1/6.
 */
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DSV4_KV_HD __host__ __device__ inline
#else
#define DSV4_KV_HD inline
#endif

namespace llaminar2::deepseek4
{
/** @brief FP8 activation-quantization block (kernel.py act_quant block_size for kv). */
inline constexpr int kFP8BlockSize = 64;
/** @brief FP4 activation-quantization block (model.py fp4_block_size). */
inline constexpr int kFP4BlockSize = 32;

/** @brief Round @p value up to a multiple of @p unit. */
DSV4_KV_HD constexpr int roundUpTo(int value, int unit) { return (value + unit - 1) / unit * unit; }

/**
 * @brief Byte layout of one packed attention KV entry (raw window or compressed).
 * @details The single descriptor consumed by packers, readers and the planner.
 */
struct PackedKVEntryFormat
{
    int headDim = 0; ///< Entry width (512 in the release model).
    int ropeDim = 0; ///< Trailing BF16 channels (64).

    /** @brief FP8 channels. */
    DSV4_KV_HD constexpr int nopeDim() const { return headDim - ropeDim; }
    /** @brief UE8M0 scale bytes, one per 64 FP8 channels. */
    DSV4_KV_HD constexpr int scaleCount() const { return nopeDim() / kFP8BlockSize; }
    /** @brief Byte offset of the BF16 rope channels. */
    DSV4_KV_HD constexpr int ropeOffset() const { return nopeDim(); }
    /** @brief Byte offset of the scale bytes. */
    DSV4_KV_HD constexpr int scaleOffset() const { return nopeDim() + 2 * ropeDim; }
    /** @brief Record size, padded to 8 bytes (584 for the release geometry). */
    DSV4_KV_HD constexpr int entryBytes() const { return roundUpTo(scaleOffset() + scaleCount(), 8); }
    /** @brief Record size in INT32 arena words. */
    DSV4_KV_HD constexpr int entryWords() const { return entryBytes() / 4; }
    /** @brief Whether the geometry is representable. */
    DSV4_KV_HD constexpr bool valid() const
    {
        return headDim > 0 && ropeDim >= 0 && ropeDim % 2 == 0 && ropeDim < headDim &&
               nopeDim() % kFP8BlockSize == 0;
    }
};

/** @brief Byte layout of one packed MXFP4 indexer key. */
struct PackedIndexerKeyFormat
{
    int headDim = 0; ///< Key width (128 in the release model); a power of two, multiple of 32.

    /** @brief Byte offset of the E8M0 scales (after the nibbles). */
    DSV4_KV_HD constexpr int scaleOffset() const { return headDim / 2; }
    /** @brief Scale bytes. */
    DSV4_KV_HD constexpr int scaleCount() const { return headDim / kFP4BlockSize; }
    /** @brief Record size, padded to 4 bytes (68 for the release geometry). */
    DSV4_KV_HD constexpr int entryBytes() const { return roundUpTo(scaleOffset() + scaleCount(), 4); }
    /** @brief Record size in INT32 arena words. */
    DSV4_KV_HD constexpr int entryWords() const { return entryBytes() / 4; }
    /** @brief Whether the geometry is representable (Hadamard requires a power of two). */
    DSV4_KV_HD constexpr bool valid() const
    {
        return headDim >= kFP4BlockSize && headDim % kFP4BlockSize == 0 && (headDim & (headDim - 1)) == 0;
    }
};

static_assert(PackedKVEntryFormat{512, 64}.entryBytes() == 584, "release fp8_ds_mla record is 584 bytes");
static_assert(PackedIndexerKeyFormat{128}.entryBytes() == 68, "release MXFP4 indexer key is 68 bytes");

/** @brief Activation-quantization post-operation of a DeepSeek-V4 head vector. */
enum class ActivationQuant : std::uint8_t
{
    None,            ///< Leave FP32 values unchanged.
    FP8NopeBF16Rope, ///< act_quant(nope, 64, ue8m0) + BF16 rope channels (window kv, main compressor).
    HadamardFP4,     ///< rotate_activation + fp4_act_quant(32) (indexer q and indexer compressor).
};

namespace qat
{
/** @brief Reinterpret FP32 bits. */
DSV4_KV_HD std::uint32_t bitsOf(float value)
{
    std::uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/** @brief Reinterpret bits as FP32. */
DSV4_KV_HD float floatOf(std::uint32_t bits)
{
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

/**
 * @name IEEE binary32 arithmetic immune to the CUDA fast-math FTZ policy
 * @details Production CUDA compiles with `--use_fast_math --ftz=true`, which
 * turns ordinary float operations into flushing instructions. The QAT, Hadamard,
 * record-decode and indexer-score arithmetic must keep IEEE subnormal operands
 * and results (an FP4 code 0.5 at the E8M0 floor is 2^-127), so device code
 * emits explicit `.rn` PTX without `.ftz` (which also forbids contraction);
 * host code uses the identical IEEE operations, with `std::fma` for the fused
 * form. Every op is one correctly rounded binary32 result on both backends.
 * @{
 */
DSV4_KV_HD float mulRn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a * b;
#endif
}
DSV4_KV_HD float addRn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a + b;
#endif
}
DSV4_KV_HD float subRn(float a, float b)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
#else
    return a - b;
#endif
}
/** @brief Single-rounding a * b + c. */
DSV4_KV_HD float fmaRn(float a, float b, float c)
{
#if defined(__CUDA_ARCH__)
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
#else
    return std::fma(a, b, c);
#endif
}
/** @brief max(x, +0) from the sign bit (FTZ comparisons would zero a subnormal). */
DSV4_KV_HD float reluBits(float x) { return (bitsOf(x) >> 31) ? 0.0f : x; }
/** @} */

/** @brief 2^k for normal exponents k in [-126, 127], built from bits (exact). */
DSV4_KV_HD float exp2i(int k) { return floatOf(static_cast<std::uint32_t>(k + 127) << 23); }

/** @brief kernel.py fast_log2_ceil of a positive normal FP32 value. */
DSV4_KV_HD int ceilLog2(float value)
{
    const std::uint32_t bits = bitsOf(value);
    const int exponent = static_cast<int>((bits >> 23) & 0xFFu);
    return exponent - 127 + ((bits & 0x7FFFFFu) != 0 ? 1 : 0);
}

/** @brief UE8M0 scale exponent of an FP8 block: fast_round_scale(max(amax, 1e-4), 1/448). */
DSV4_KV_HD int fp8ScaleExponent(float amax)
{
    const float clamped = fmaxf(amax, 1e-4f);
    return ceilLog2(clamped * (1.0f / 448.0f));
}

/** @brief E8M0 scale exponent of an FP4 block: fast_round_scale(max(amax, 6 * 2^-126), 1/6). */
DSV4_KV_HD int fp4ScaleExponent(float amax)
{
    const float clamped = fmaxf(amax, 6.0f * 1.17549435e-38f);
    return ceilLog2(clamped * (1.0f / 6.0f));
}

/**
 * @brief Round-to-nearest-even FP8 E4M3 (fn, satfinite) code of a value.
 * @param value FP32 input; magnitudes above 448 saturate (kernel.py clamps first).
 */
DSV4_KV_HD std::uint8_t encodeE4M3(float value)
{
    const std::uint8_t sign = (bitsOf(value) >> 31) ? 0x80u : 0x00u;
    const float a = fminf(fabsf(value), 448.0f);
    if (a < 0.015625f) // below the smallest normal 2^-6: subnormal grid 2^-9
    {
        const int q = static_cast<int>(rintf(a * 512.0f)); // 0..8; 8 encodes exactly 2^-6
        return static_cast<std::uint8_t>(sign | q);
    }
    int e = static_cast<int>((bitsOf(a) >> 23) & 0xFFu) - 127; // floor(log2 a), a normal
    int m = static_cast<int>(rintf(a * exp2i(3 - e))) - 8;      // 0..8 mantissa eighths
    if (m == 8)
    {
        m = 0;
        ++e;
    }
    return static_cast<std::uint8_t>(sign | ((e + 7) << 3) | m);
}

/** @brief Exact FP32 value of an FP8 E4M3 code. */
DSV4_KV_HD float decodeE4M3(std::uint8_t code)
{
    const int e = (code >> 3) & 0xF;
    const int m = code & 0x7;
    const float magnitude = e == 0 ? static_cast<float>(m) * 0.001953125f /* 2^-9 */
                                   : floatOf(static_cast<std::uint32_t>(e - 7 + 127) << 23 |
                                             static_cast<std::uint32_t>(m) << 20);
    return (code & 0x80u) ? -magnitude : magnitude;
}

/** @brief Round-to-nearest-even FP4 E2M1 (satfinite) nibble of a value. */
DSV4_KV_HD std::uint8_t encodeE2M1(float value)
{
    const std::uint8_t sign = (bitsOf(value) >> 31) ? 0x8u : 0x0u;
    const float a = fminf(fabsf(value), 6.0f);
    std::uint8_t code;
    if (a < 2.0f)
        code = static_cast<std::uint8_t>(rintf(a * 2.0f)); // 0, 0.5, 1, 1.5 -> codes 0..3, 4 = 2.0
    else if (a < 4.0f)
        code = static_cast<std::uint8_t>(static_cast<int>(rintf(a)) + 2); // 2, 3, 4 -> 4, 5, 6
    else
        code = static_cast<std::uint8_t>(static_cast<int>(rintf(a * 0.5f)) + 4); // 4, 6 -> 6, 7
    return static_cast<std::uint8_t>(sign | code);
}

/** @brief Exact FP32 value of an FP4 E2M1 nibble. */
DSV4_KV_HD float decodeE2M1(std::uint8_t nibble)
{
    const int e = (nibble >> 1) & 0x3;
    const int m = nibble & 0x1;
    const float magnitude = e == 0 ? static_cast<float>(m) * 0.5f
                                   : floatOf(static_cast<std::uint32_t>(e - 1 + 127) << 23 |
                                             static_cast<std::uint32_t>(m) << 22);
    return (nibble & 0x8u) ? -magnitude : magnitude;
}

/** @brief Round-to-nearest-even BF16 payload of a finite FP32 value. */
DSV4_KV_HD std::uint16_t encodeBF16(float value)
{
    const std::uint32_t bits = bitsOf(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/** @brief Exact FP32 value of a BF16 payload. */
DSV4_KV_HD float decodeBF16(std::uint16_t payload) { return floatOf(static_cast<std::uint32_t>(payload) << 16); }

/** @brief Maximum magnitude of @p count values. */
DSV4_KV_HD float absMax(const float *values, int count)
{
    float amax = 0.0f;
    for (int i = 0; i < count; ++i) amax = fmaxf(amax, fabsf(values[i]));
    return amax;
}

/** @brief In-place FP8 QAT of one 64-value block (act_quant inplace with ue8m0 scale). */
DSV4_KV_HD void fakeQuantFP8Block(float *values, int count)
{
    const int k = fp8ScaleExponent(absMax(values, count));
    const float inverse = exp2i(-k), scale = exp2i(k);
    for (int i = 0; i < count; ++i) values[i] = mulRn(decodeE4M3(encodeE4M3(mulRn(values[i], inverse))), scale);
}

/** @brief In-place FP4 QAT of one 32-value block (fp4_act_quant inplace). */
DSV4_KV_HD void fakeQuantFP4Block(float *values, int count)
{
    const int k = fp4ScaleExponent(absMax(values, count));
    const float inverse = exp2i(-k), scale = exp2i(k);
    for (int i = 0; i < count; ++i) values[i] = mulRn(decodeE2M1(encodeE2M1(mulRn(values[i], inverse))), scale);
}

/** @brief FP32 rounding of n^-0.5 (Python's float `n ** -0.5` passed to the CUDA kernel). */
DSV4_KV_HD float hadamardScale(int n) { return static_cast<float>(1.0 / sqrt(static_cast<double>(n))); }

/**
 * @brief In-place Walsh-Hadamard transform (Sylvester order) scaled by n^-0.5.
 * @details Butterfly stages h = 1, 2, 4, ...: (a, b) -> (a + b, a - b), then
 * one multiplication by 1/sqrt(n) (fast_hadamard_transform semantics).
 */
DSV4_KV_HD void hadamard(float *values, int n)
{
    for (int h = 1; h < n; h *= 2)
        for (int i = 0; i < n; i += 2 * h)
            for (int j = i; j < i + h; ++j)
            {
                const float a = values[j], b = values[j + h];
                values[j] = addRn(a, b);
                values[j + h] = subRn(a, b);
            }
    const float scale = hadamardScale(n);
    for (int i = 0; i < n; ++i) values[i] = mulRn(values[i], scale);
}

/** @brief Apply @p quant to one head vector in place. */
DSV4_KV_HD void apply(ActivationQuant quant, float *values, int headDim, int ropeDim)
{
    if (quant == ActivationQuant::FP8NopeBF16Rope)
    {
        const int nope = headDim - ropeDim;
        for (int b = 0; b < nope; b += kFP8BlockSize) fakeQuantFP8Block(values + b, kFP8BlockSize);
        for (int c = nope; c < headDim; ++c) values[c] = decodeBF16(encodeBF16(values[c]));
    }
    else if (quant == ActivationQuant::HadamardFP4)
    {
        hadamard(values, headDim);
        for (int b = 0; b < headDim; b += kFP4BlockSize) fakeQuantFP4Block(values + b, kFP4BlockSize);
    }
}

/** @brief Quantize and pack one FP32 KV vector into its record (exact for QAT inputs). */
DSV4_KV_HD void packKVEntry(const PackedKVEntryFormat &format, const float *values, std::uint8_t *entry)
{
    const int nope = format.nopeDim();
    for (int b = 0; b < format.scaleCount(); ++b)
    {
        const float *block = values + b * kFP8BlockSize;
        const int k = fp8ScaleExponent(absMax(block, kFP8BlockSize));
        const float inverse = exp2i(-k);
        for (int i = 0; i < kFP8BlockSize; ++i) entry[b * kFP8BlockSize + i] = encodeE4M3(mulRn(block[i], inverse));
        entry[format.scaleOffset() + b] = static_cast<std::uint8_t>(k + 127);
    }
    for (int c = 0; c < format.ropeDim; ++c)
    {
        const std::uint16_t payload = encodeBF16(values[nope + c]);
        entry[format.ropeOffset() + 2 * c] = static_cast<std::uint8_t>(payload & 0xFFu);
        entry[format.ropeOffset() + 2 * c + 1] = static_cast<std::uint8_t>(payload >> 8);
    }
    for (int p = format.scaleOffset() + format.scaleCount(); p < format.entryBytes(); ++p) entry[p] = 0;
}

/** @brief Exact FP32 value of channel @p c of a packed KV record. */
DSV4_KV_HD float decodeKVChannel(const PackedKVEntryFormat &format, const std::uint8_t *entry, int c)
{
    if (c < format.nopeDim())
        return mulRn(decodeE4M3(entry[c]), exp2i(static_cast<int>(entry[format.scaleOffset() + c / kFP8BlockSize]) - 127));
    const int r = format.ropeOffset() + 2 * (c - format.nopeDim());
    return decodeBF16(static_cast<std::uint16_t>(entry[r] | (entry[r + 1] << 8)));
}

/** @brief Quantize and pack one already-rotated FP32 indexer key (exact for QAT inputs). */
DSV4_KV_HD void packIndexerKey(const PackedIndexerKeyFormat &format, const float *values, std::uint8_t *entry)
{
    for (int b = 0; b < format.scaleCount(); ++b)
    {
        const float *block = values + b * kFP4BlockSize;
        const int k = fp4ScaleExponent(absMax(block, kFP4BlockSize));
        const float inverse = exp2i(-k);
        for (int i = 0; i < kFP4BlockSize; i += 2)
            entry[(b * kFP4BlockSize + i) / 2] = static_cast<std::uint8_t>(
                encodeE2M1(mulRn(block[i], inverse)) | (encodeE2M1(mulRn(block[i + 1], inverse)) << 4));
        entry[format.scaleOffset() + b] = static_cast<std::uint8_t>(k + 127);
    }
    for (int p = format.scaleOffset() + format.scaleCount(); p < format.entryBytes(); ++p) entry[p] = 0;
}

/** @brief Exact FP32 value of channel @p c of a packed indexer key. */
DSV4_KV_HD float decodeIndexerChannel(const PackedIndexerKeyFormat &format, const std::uint8_t *entry, int c)
{
    const std::uint8_t pair = entry[c / 2];
    const std::uint8_t nibble = static_cast<std::uint8_t>((c & 1) ? (pair >> 4) : (pair & 0xFu));
    return mulRn(decodeE2M1(nibble), exp2i(static_cast<int>(entry[format.scaleOffset() + c / kFP4BlockSize]) - 127));
}

/**
 * @brief Lightning-indexer q . k for one head against a packed key.
 * @details Each 32-channel block dot of FP4-representable q and unscaled E2M1
 * k values is exact in FP32 (products need <= 4 significant bits and the block
 * sum stays below 2^12 granules); the block sums are scaled and accumulated by
 * fused multiply-adds in ascending block order. CPU (on pre-decoded keys) and
 * GPU execute exactly this operation sequence.
 */
DSV4_KV_HD float indexerDot(const PackedIndexerKeyFormat &format, const float *query, const std::uint8_t *entry)
{
    float dot = 0.0f;
    for (int b = 0; b < format.scaleCount(); ++b)
    {
        float block = 0.0f;
        for (int i = 0; i < kFP4BlockSize; i += 2)
        {
            const std::uint8_t pair = entry[(b * kFP4BlockSize + i) / 2];
            block = fmaRn(query[b * kFP4BlockSize + i], decodeE2M1(static_cast<std::uint8_t>(pair & 0xFu)), block);
            block = fmaRn(query[b * kFP4BlockSize + i + 1], decodeE2M1(static_cast<std::uint8_t>(pair >> 4)), block);
        }
        dot = fmaRn(block, exp2i(static_cast<int>(entry[format.scaleOffset() + b]) - 127), dot);
    }
    return dot;
}

/** @brief One head's contribution: score' = relu(dot) * (weight * weightScale) + score, one rounding. */
DSV4_KV_HD float indexerAccumulate(float score, float dot, float weight, float weightScale)
{
    return fmaRn(reluBits(dot), mulRn(weight, weightScale), score);
}

} // namespace qat
} // namespace llaminar2::deepseek4
