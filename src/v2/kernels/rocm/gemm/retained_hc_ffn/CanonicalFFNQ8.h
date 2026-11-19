/**
 * @file CanonicalFFNQ8.h
 * @brief Current production Q8 bytes in the tuned kernel's interleaved activation storage.
 *
 * Eight physical lanes virtualize the original32-value maximum and preserve
 * its exact integer sum. Production scale and prepared-inverse conversion
 * helpers own all scalar arithmetic. The actual scalar quantizer remains the
 * native test control; HC's distinct q81 and reciprocal policy are unchanged.
 */
#pragma once
#include "ROCmRetainedM1QuietMagnitude.h"

#include "kernels/rocm/gemm/ROCmPersistentMoEFFNQuantize.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace qfn_castagna_native_impl {
/**
 * @brief One original32-value production activation block, without precision loss.
 * @details Scale is the production FP32 value (already half-representable),
 * sum is the exact signed-code sum, and qs retains the original32 code bytes.
 * The type owns no allocation, publication event or source policy.
 */
struct CanonicalFFNQ8Block {
    float scale;
    std::int32_t sum;
    std::int32_t qs[8];
};
static_assert(std::is_standard_layout_v<CanonicalFFNQ8Block> &&
    std::is_trivially_copyable_v<CanonicalFFNQ8Block> &&
    sizeof(CanonicalFFNQ8Block)==40 && alignof(CanonicalFFNQ8Block)==4 &&
    offsetof(CanonicalFFNQ8Block,scale)==0 &&
    offsetof(CanonicalFFNQ8Block,sum)==4 &&
    offsetof(CanonicalFFNQ8Block,qs)==8);

/**
 * @brief Preserve the original Newton reciprocal for an already prepared Q8 scale.
 * @param scale Exact result of device_q8_activation_contract::scale(), unchanged.
 * @return The same binary32 word as reciprocalPositive(scale).
 * @pre The caller passes the positive finite half-representable scale in
 * [2^-24,65504] directly from the original clamp/round publisher. This is not
 * the general reciprocal API and cannot accept arbitrary caller metadata.
 * @details Every input is binary32-normal with stored exponent103..142.
 * The unchanged normalized five-Newton result has stored exponent126or127,
 * so the restored exponent is111..151 and also normal. Only unreachable sign,
 * zero, NaN, infinity and subnormal branches are omitted. The actual shared
 * normalized routine retains every rounded multiply/add and iteration count.
 */
__host__ __device__ __forceinline__ float canonicalFFNPreparedScaleReciprocal(float scale) {
    namespace fp=llaminar2::device_fp32_contract;
    constexpr std::uint32_t mantissa_mask=0x007fffffu;
    constexpr std::uint32_t exponent_mask=0x7f800000u;
    const std::uint32_t input_bits=fp::bitsFromFloat(scale);
    const int unbiased_exponent=int((input_bits&exponent_mask)>>23u)-127;
    const float normalized=fp::floatFromBits((127u<<23u)|(input_bits&mantissa_mask));
    const float inverse=fp::reciprocalOneToTwo(normalized);
    const std::uint32_t inverse_bits=fp::bitsFromFloat(inverse);
    const int target_exponent=int((inverse_bits&exponent_mask)>>23u)-unbiased_exponent;
    return fp::floatFromBits((std::uint32_t(target_exponent)<<23u)|(inverse_bits&mantissa_mask));
}

/**
 * @brief Publish four adjacent values through the original logical32-value Q8 contract.
 * @param values Original adjacent FP32 values, loaded from a proven16-byte-aligned row.
 * @param linear First original scalar coordinate, a multiple4 within a complete32-value block.
 * @param blocks Original40-byte block bank; no allocation or extra publication exists.
 * @return Packed four signed-code bytes, for optional same-owner observations.
 * @pre Eight physical lanes enter convergently and own one complete32-value block.
 * linear%32 equals4*(threadIdx.x%8), and every caller stride is a multiple8 quads.
 * @details Four component states virtualize original XOR16/8/4 as physical
 * XOR4/2/1, then local component XOR2/1 restores the original maximum root.
 * fabs produces positive zero. Nonfinite roots enter the identical scale clamp;
 * quantizePreparedInverse owns all defined per-value NaN/Inf conversion behavior.
 * The five-Newton inverse is computed once and reused unchanged for four values.
 * Integer sum regrouping is exact: every partial is bounded by32*127=4064,
 * so it preserves the original XOR32 signed sum without any overflow or FP add.
 */
template<bool NativeHalfScale=false>
__device__ __forceinline__ std::uint32_t canonicalFFNQuantizeFour(
    float4 values,std::size_t linear,CanonicalFFNQ8Block* blocks)
{
    using namespace llaminar2::rocm::moe_wave_contract;
    float maximum;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    if constexpr(NativeHalfScale) {
        maximum=::qfn_retained_m1_quantizer::quietMagnitudeMax32(values);
    } else
#endif
    {
        float m0=fabsf(values.x),m1=fabsf(values.y),m2=fabsf(values.z),m3=fabsf(values.w);
#pragma unroll
        for(int mask=4;mask>0;mask>>=1) {
            m0=fmaxf(m0,quantizerXor32<QuantizerXorPolicy::TargetNative>(m0,mask));
            m1=fmaxf(m1,quantizerXor32<QuantizerXorPolicy::TargetNative>(m1,mask));
            m2=fmaxf(m2,quantizerXor32<QuantizerXorPolicy::TargetNative>(m2,mask));
            m3=fmaxf(m3,quantizerXor32<QuantizerXorPolicy::TargetNative>(m3,mask));
        }
        maximum=fmaxf(fmaxf(m0,m2),fmaxf(m1,m3));
    }
    namespace q8=llaminar2::device_q8_activation_contract;
    const float scale=NativeHalfScale?q8::publishPositiveFiniteHalfScale(q8::boundedScale(maximum)):
        q8::scale(maximum);
    const float inverse=canonicalFFNPreparedScaleReciprocal(scale);
    const int q0=llaminar2::device_q8_activation_contract::quantizePreparedInverse(values.x,inverse);
    const int q1=llaminar2::device_q8_activation_contract::quantizePreparedInverse(values.y,inverse);
    const int q2=llaminar2::device_q8_activation_contract::quantizePreparedInverse(values.z,inverse);
    const int q3=llaminar2::device_q8_activation_contract::quantizePreparedInverse(values.w,inverse);
    const std::uint32_t packed=std::uint32_t(std::uint8_t(q0))|
        (std::uint32_t(std::uint8_t(q1))<<8)|
        (std::uint32_t(std::uint8_t(q2))<<16)|(std::uint32_t(std::uint8_t(q3))<<24);
    auto& block=blocks[linear/32];
    block.qs[(linear%32)/4]=__builtin_bit_cast(std::int32_t,packed);
    int sum=(q0+q2)+(q1+q3);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    if constexpr(NativeHalfScale) {
        // Only cohort lane 0 consumes the integer result. A descending down
        // tree retains that lane's complete group and exact signed sum.
        asm volatile("s_nop 1\n"
                     "v_add_u32_dpp %0, %0, %0 row_shl:4 row_mask:0xf bank_mask:0xf\n"
                     "s_nop 1\n"
                     "v_add_u32_dpp %0, %0, %0 row_shl:2 row_mask:0xf bank_mask:0xf\n"
                     "s_nop 1\n"
                     "v_add_u32_dpp %0, %0, %0 row_shl:1 row_mask:0xf bank_mask:0xf"
                     : "+v"(sum));
    } else
#endif
    {
#pragma unroll
        for(int mask=4;mask>0;mask>>=1)
            sum+=quantizerXor32<QuantizerXorPolicy::TargetNative>(sum,mask);
    }
    if((threadIdx.x%8)==0){block.scale=scale;block.sum=sum;}
    return packed;
}
} // namespace qfn_castagna_native_impl
