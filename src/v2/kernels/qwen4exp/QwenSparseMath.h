/**
 * @file QwenSparseMath.h
 * @brief Shared exact representation boundaries for Qwen sparse indexer kernels.
 *
 * Arithmetic between casts is FP32. FP16/BF16 round to nearest even at pooling,
 * RMSNorm output, rotary operands, and rotary output, matching the independent
 * Transformers 28034545c35a reference rather than treating FP32 storage as an
 * implicit precision policy. There is no backend-owned persistent state here.
 */
#pragma once
#include "QwenSparseAttention.h"
#include "tensors/BlockStructures.h"
#include <cstdint>
#include <cstring>
#include <cmath>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define QWEN_SPARSE_HD __host__ __device__ __forceinline__
#else
#define QWEN_SPARSE_HD inline
#endif

#if defined(__CUDACC__) && !defined(__HIPCC__)
// Direct libdevice entrypoints retain faithful argument reduction even when
// the production CUDA target enables --use_fast_math. Ordinary core RoPE keeps
// its own canonical fast arithmetic; only the HF indexer selects these calls.
extern "C" __device__ float __nv_cosf(float);
extern "C" __device__ float __nv_sinf(float);
#endif

namespace llaminar2::qwen4exp::detail
{
/** @brief Resolve FP32 checkpoint rotary factors with explicit partial rotary width. */
QWEN_SPARSE_HD void rotaryFactors(QwenSparseRotarySource source,const float *cosine,const float *sine,
    int position,int pair,int rotaryDim,float theta,float &c,float &s,
    const float *inverseFrequencies=nullptr,float negativeLogTheta=0.0f)
{
    if(source==QwenSparseRotarySource::TablePlanes) {
        const auto index=std::size_t(position)*(rotaryDim/2)+pair;
        c=cosine[index];s=sine[index];
    } else {
        // HF forms FP32 inverse frequencies first, then FP32 position products.
        const float exponent=float(2*pair)/float(rotaryDim);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        // Every device power source carries setup-baked invariants. The native
        // core permits only tables or the ordinary engine's exponential source;
        // retaining a dead powf branch here needlessly raises all codec VGPRs.
        const float frequency=source==QwenSparseRotarySource::CanonicalEngineTheta ? ::expf(negativeLogTheta*exponent) : inverseFrequencies[pair];
#else
        const float frequency=source==QwenSparseRotarySource::CanonicalEngineTheta ? ::expf(negativeLogTheta*exponent) :
            inverseFrequencies ? inverseFrequencies[pair] : 1.0f/::powf(theta,exponent);
#endif
        const float angle=float(position)*frequency;
#if defined(__CUDA_ARCH__)
        if(source==QwenSparseRotarySource::AnalyticTheta) { c=::__nv_cosf(angle);s=::__nv_sinf(angle); }
        else { ::__sincosf(angle,&s,&c); } // Exact ordinary CUDA RoPE intrinsic policy.
#elif defined(__HIP_DEVICE_COMPILE__)
        if(source==QwenSparseRotarySource::CanonicalEngineTheta) { c=::__cosf(angle);s=::__sinf(angle); }
        else { c=::cosf(angle);s=::sinf(angle); }
#else
        c=::cosf(angle);s=::sinf(angle);
#endif
    }
}
/** @brief Canonical TQ key estimator and reconstructed value use different radial scales. */
enum class BankRole : std::uint8_t { Key, Value };
/** @brief Recover the IEEE bit pattern without pointer aliasing. */
QWEN_SPARSE_HD std::uint32_t floatBits(float value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(value);
#else
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
#endif
}
/** @brief Recover an IEEE float from its bit pattern without pointer aliasing. */
QWEN_SPARSE_HD float bitsFloat(std::uint32_t bits)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(bits);
#else
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
#endif
}
/** @brief Decode one IEEE FP16 value, including subnormals. */
QWEN_SPARSE_HD float decodeHalf(std::uint16_t value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __half2float(__ushort_as_half(value));
#else
    const std::uint32_t sign = (std::uint32_t(value) & 0x8000u) << 16;
    std::uint32_t fraction = value & 1023u;
    int exponent = (value >> 10) & 31;
    if (exponent == 0)
    {
        if (fraction == 0) return bitsFloat(sign);
        exponent = -14;
        while ((fraction & 1024u) == 0) { fraction <<= 1; --exponent; }
        return bitsFloat(sign | (std::uint32_t(exponent + 127) << 23) | ((fraction & 1023u) << 13));
    }
    if (exponent == 31) return bitsFloat(sign | 0x7f800000u | (fraction << 13));
    return bitsFloat(sign | (std::uint32_t(exponent + 112) << 23) | (fraction << 13));
#endif
}
/** @brief Encode FP16 with ties-to-even; preserve signed zero and non-finite classes. */
QWEN_SPARSE_HD std::uint16_t encodeHalf(float value)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __half_as_ushort(__float2half_rn(value));
#else
    const std::uint32_t bits = floatBits(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const std::uint32_t magnitude = bits & 0x7fffffffu;
    if (magnitude >= 0x7f800000u)
        return std::uint16_t(sign | 0x7c00u | (magnitude > 0x7f800000u ? 0x0200u : 0u));
    const int exponent = int(magnitude >> 23) - 127;
    if (exponent > 15) return std::uint16_t(sign | 0x7c00u);
    if (exponent < -25) return std::uint16_t(sign);
    if (exponent < -14)
    {
        const std::uint32_t mantissa = (magnitude & 0x7fffffu) | 0x800000u;
        const int shift = -exponent - 1;
        const std::uint32_t rounded = (mantissa + ((1u << (shift - 1)) - 1u) + ((mantissa >> shift) & 1u)) >> shift;
        return std::uint16_t(sign | rounded);
    }
    const std::uint32_t rounded = magnitude + 0xfffu + ((magnitude >> 13) & 1u);
    return std::uint16_t(sign | ((rounded - 0x38000000u) >> 13));
#endif
}
/** @brief Round one FP32 container value at an explicit reference cast boundary. */
QWEN_SPARSE_HD float roundRepresentation(float value, QwenSparseScalar representation)
{
    if (representation == QwenSparseScalar::FP16) return decodeHalf(encodeHalf(value));
    if (representation == QwenSparseScalar::BF16)
    {
        std::uint32_t bits = floatBits(value);
        if ((bits & 0x7fffffffu) > 0x7f800000u) return bitsFloat((bits & 0xffff0000u) | 0x00400000u);
        bits += 0x7fffu + ((bits >> 16) & 1u);
        return bitsFloat(bits & 0xffff0000u);
    }
    return value;
}
/** @brief Read exactly one native coordinate from a selected cache head. */
QWEN_SPARSE_HD float readNative(const void *base, std::size_t offset, int coordinate,
                                QwenSparseScalar representation)
{
    const auto *bytes = static_cast<const unsigned char *>(base) + offset;
    if (representation == QwenSparseScalar::FP32)
        return reinterpret_cast<const float *>(bytes)[coordinate];
    const auto bits = reinterpret_cast<const std::uint16_t *>(bytes)[coordinate];
    return representation == QwenSparseScalar::FP16 ? decodeHalf(bits) : bitsFloat(std::uint32_t(bits) << 16);
}
/** @brief Whether one cache bank uses a rotated TurboQuant coordinate basis. */
QWEN_SPARSE_HD bool isRotated(const QwenSparseNativeBank &bank)
{
    return bank.codec == TensorType::TQ4 || bank.codec == TensorType::TQ8;
}
/** @brief Native scalar policy for an authenticated floating bank. */
QWEN_SPARSE_HD QwenSparseScalar scalarRepresentation(const QwenSparseNativeBank &bank)
{
    return bank.codec == TensorType::FP16 ? QwenSparseScalar::FP16 :
           bank.codec == TensorType::BF16 ? QwenSparseScalar::BF16 : QwenSparseScalar::FP32;
}
/**
 * @brief Decode exactly one selected coordinate in the bank's native basis.
 *
 * TurboQuant values remain rotated until the final weighted sum is transformed
 * once. AQ8 uses its canonical signed-square multiplication order before adding
 * the sole immutable request-local anchor. No cache-wide materialization exists.
 */
QWEN_SPARSE_HD float readBank(const QwenSparseNativeBank &bank, int physical_token, int head, int coordinate,
                             BankRole role = BankRole::Value)
{
    const std::size_t offset = std::size_t(physical_token) * bank.token_stride_bytes + std::size_t(head) * bank.head_stride_bytes;
    const auto *bytes = static_cast<const unsigned char *>(bank.data) + offset;
    if (bank.codec == TensorType::FP32 || bank.codec == TensorType::FP16 || bank.codec == TensorType::BF16)
        return readNative(bank.data, offset, coordinate, scalarRepresentation(bank));
    if (bank.codec == TensorType::Q8_1)
    {
        const auto &block = reinterpret_cast<const Q8_1Block *>(bytes)[coordinate / 32];
        return decodeHalf(block.d) * float(block.qs[coordinate % 32]);
    }
    if (bank.codec == TensorType::Q16_1)
    {
        const int block_size = bank.q16_block_size;
        const auto *block = bytes + std::size_t(coordinate / block_size) * (8 + block_size * 2);
        return *reinterpret_cast<const float *>(block) * float(reinterpret_cast<const std::int16_t *>(block + 8)[coordinate % block_size]);
    }
    if (bank.codec == TensorType::AQ8)
    {
        const auto &block = *reinterpret_cast<const AttentionKeyQ8Block<256> *>(bytes);
        const float code = float(block.codes[coordinate]);
        const float magnitude = code < 0.0f ? -code : code;
        const float residual = (block.quadratic_scale * code) * magnitude;
        return residual + bank.anchor[head * 256 + coordinate];
    }
    if (bank.codec == TensorType::TQ8)
    {
        const auto &block = *reinterpret_cast<const TQ8Block<256> *>(bytes);
        const float radius = role == BankRole::Key ? block.norm : block.reconstruction_norm;
        return block.norm < 1e-30f ? 0.0f : bank.centroids[block.indices[coordinate]] * (radius * 0.0625f);
    }
    // Static admission validates TQ4 as the remaining native codec.
    const auto &block = *reinterpret_cast<const TQ4Block<256> *>(bytes);
    if (block.norm < 1e-30f) return 0.0f;
    const int bit = coordinate * 3, byte = bit / 8, shift = bit % 8;
    unsigned low = unsigned(block.mse_indices[byte]) >> shift;
    if (shift > 5) low |= unsigned(block.mse_indices[byte + 1]) << (8 - shift);
    const unsigned high = (unsigned(block.high_bits[coordinate / 8]) >> (coordinate % 8)) & 1u;
    const float radius = role == BankRole::Key ? block.norm : block.reconstruction_norm;
    return bank.centroids[(low & 7u) | (high << 3)] * (radius * 0.0625f);
}
/** @brief Authenticate and resolve the sole physical-ring mapping for a row. */
QWEN_SPARSE_HD bool resolveMapping(const QwenSparseKV &kv, const QwenSparseReplay &r,
                                  int row, int token_capacity, int &origin, int &capacity)
{
    origin = capacity = 0;
    if (kv.addressing == QwenSparseKVAddressing::FixedLogicalView)
    {
        origin = kv.logical_view.logical_row_origin;
        capacity = kv.logical_view.physical_row_capacity;
    }
    else if (kv.addressing == QwenSparseKVAddressing::RowParameters)
    {
        const auto &params = kv.row_parameters[row];
        if (params.kv_len != r.prefix_tokens + row + 1 ||
            params.position_offset != r.position_origin + r.prefix_tokens + row) return false;
        origin = params.ring_row_origin;
        capacity = params.ring_row_capacity;
    }
    return capacity == 0 ? origin == 0 :
        capacity > 0 && capacity <= token_capacity && origin >= 0 && origin < capacity && r.prefix_tokens + row + 1 <= capacity;
}
/** @brief Translate one validated selected logical index into the authoritative ring. */
QWEN_SPARSE_HD int physicalRow(int logical, int origin, int capacity)
{
    if (capacity == 0) return logical;
    const auto sum = static_cast<std::uint32_t>(origin) + static_cast<std::uint32_t>(logical);
    return int(sum >= static_cast<std::uint32_t>(capacity) ? sum - static_cast<std::uint32_t>(capacity) : sum);
}
/** @brief Read a selected key and, only for declared floating pre-RoPE storage, rotate it. */
QWEN_SPARSE_HD float readKey(const QwenSparseKV &kv, int physical, int logical, int head,
                             int coordinate, int position_origin)
{
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
    const float value = readBank(kv.keys, physical, head, coordinate, BankRole::Key);
    if (kv.key_policy.encoding == attention::AttentionKeyCacheEncoding::PostRotary || coordinate >= kv.key_rotary_dim)
        return value;
    const int half = kv.key_rotary_dim / 2;
    const int partner = coordinate < half ? coordinate + half : coordinate - half;
    float other = readBank(kv.keys, physical, head, partner, BankRole::Key);
    if (coordinate < half) other = -other;
    const auto representation = scalarRepresentation(kv.keys);
    float cosine,sine;
    rotaryFactors(kv.key_rotary_source,kv.key_cosine,kv.key_sine,position_origin+logical,coordinate%half,
        kv.key_rotary_dim,kv.key_policy.rope_theta,cosine,sine,nullptr,kv.key_negative_log_theta);
    const float c = roundRepresentation(cosine, representation);
    const float s = roundRepresentation(sine, representation);
    return roundRepresentation(roundRepresentation(value * c, representation) + roundRepresentation(other * s, representation), representation);
}
/** @brief Bind the immutable codec in the compiled native inner loop. */
template<TensorType Codec,BankRole Role=BankRole::Value>
QWEN_SPARSE_HD float readBankAs(const QwenSparseNativeBank &bank, int physical, int head, int coordinate)
{
    auto native = bank;
    native.codec = Codec;
    return readBank(native, physical, head, coordinate,Role);
}
/** @brief Bind the immutable key codec before selected-only optional RoPE. */
template<TensorType Codec>
QWEN_SPARSE_HD float readKeyAs(const QwenSparseKV &kv, int physical, int logical, int head,
                               int coordinate, int position_origin)
{
    auto native = kv;
    native.keys.codec = Codec;
    return readKey(native, physical, logical, head, coordinate, position_origin);
}
} // namespace llaminar2::qwen4exp::detail
#undef QWEN_SPARSE_HD
