/**
 * @file Q8ScaleUnderflowFixture.h
 * @brief Fixed FP32 operands reproducing a nonzero half-scale underflow.
 *
 * Captured by the independent CPU diagnostic at Qwen3.6 MoE layer2 out_proj,
 * reset-start Prefill row0, columns1472..1503. This is a changed CPU diagnostic
 * trajectory operand, not a saved native GPU publication. Bit patterns preserve
 * the original32 values. Expected bytes describe the explicit min-half policy.
 */
#pragma once
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
namespace llaminar2::test {
inline constexpr std::array<std::uint32_t,32> q8ScaleUnderflowSourceBits{
    0x3196cb03u,0xb3feeea0u,0x33636f93u,0xb21017eeu,
    0xb67175ecu,0x3085d23du,0x32f0d514u,0x3248d75fu,
    0x31230a7du,0x3197a073u,0x34b25b47u,0xb312f19au,
    0xaf12292bu,0xb10dd033u,0x332d7accu,0xb5005221u,
    0xb3897f40u,0xb3b4268cu,0x34278b39u,0x322f3598u,
    0xb47f912fu,0xb445e6d1u,0x346abf4du,0xb10a0cc1u,
    0xb5a31a7cu,0xb3dba884u,0xb5c6349bu,0xb375b02du,
    0xb57d7c8du,0x3505160cu,0x352fbbb2u,0xb41bfa41u};
inline constexpr std::array<std::int8_t,32> q8ScaleUnderflowExpectedBytes{
    0,-2,1,0,-60,0,0,0,0,0,6,-1,0,0,1,-8,-1,-1,3,0,-4,-3,4,0,-20,-2,-25,-1,-16,8,11,-2};
/** @brief Return one unchanged diagnostic source value.
 * @param lane Original block lane in[0,31]; invalid indices fail via array::at.
 * @return FP32 value with its captured bits preserved.
 */
inline float q8ScaleUnderflowSourceValue(std::size_t lane) {
    return std::bit_cast<float>(q8ScaleUnderflowSourceBits.at(lane));
}
} // namespace llaminar2::test
