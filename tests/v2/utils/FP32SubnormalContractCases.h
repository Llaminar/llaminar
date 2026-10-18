/**
 * @file FP32SubnormalContractCases.h
 * @brief Independent fixed IEEE words for canonical FP32 arithmetic boundaries.
 *
 * Exact powers of two cover subnormal inputs/results, cancellation, signed
 * zeros and normal controls. The final FMA witness is exactly -2^-46, while
 * its separately rounded product is1. No host floating-point calculation or
 * device compiler mode supplies the expected words.
 */
#pragma once
#include <array>
#include <cstdint>
namespace llaminar2::test
{
/** @brief Dynamic source triple and expected multiply/add/FMA result words. */
struct FP32SubnormalCase
{
    std::uint32_t lhs, rhs, accumulator;
    std::array<std::uint32_t, 3> expected;
};
/** @brief Source and expected words, shared verbatim across all three backends. */
inline constexpr std::array<FP32SubnormalCase, 13> fp32SubnormalCases{{
    {0x00000001u,0x43000000u,0x00000000u,{0x00000080u,0x43000000u,0x00000080u}},
    {0x00800000u,0x3f000000u,0x00000000u,{0x00400000u,0x3f000000u,0x00400000u}},
    {0x80800000u,0x3f000000u,0x00000000u,{0x80400000u,0x3f000000u,0x80400000u}},
    {0x00800000u,0x807fffffu,0x00000000u,{0x80000000u,0x00000001u,0x80000000u}},
    {0x00000001u,0x3f800000u,0x00000001u,{0x00000001u,0x3f800000u,0x00000002u}},
    {0x00800000u,0x3f800000u,0x807fffffu,{0x00800000u,0x3f800000u,0x00000001u}},
    {0x80000000u,0x40000000u,0x80000000u,{0x80000000u,0x40000000u,0x80000000u}},
    {0x00000000u,0xc0000000u,0x80000000u,{0x80000000u,0xc0000000u,0x80000000u}},
    {0x80000000u,0x00000000u,0x00000000u,{0x80000000u,0x00000000u,0x00000000u}},
    {0x3fc00000u,0x3f000000u,0x3e800000u,{0x3f400000u,0x40000000u,0x3f800000u}},
    {0x007fffffu,0x40000000u,0x00000000u,{0x00fffffeu,0x40000000u,0x00fffffeu}},
    {0x00800000u,0x3f800001u,0x80800000u,{0x00800001u,0x3f800001u,0x00000001u}},
    {0x3f800001u,0x3f7ffffeu,0xbf800000u,{0x3f800000u,0x40000000u,0xa8800000u}},
}};
} // namespace llaminar2::test
