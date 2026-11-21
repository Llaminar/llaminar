/**
 * @file ROCmSelectedHeadGolden.h
 * @brief Independent tensor-format golden masks from the complete forced head sweep.
 *
 * These literals replay original measurements through host admission and actual
 * captured engine census. They never import the production policy table.
 * Seed SHA256 69d3e89177fc843e696a5f13ff80aefa2b38304f322a1cfadc18a31719fc30ed.
 */
#pragma once
#include "tensors/TensorType.h"
#include <array>
#include <cstdint>
namespace llaminar2::test {
/** @brief Original tensor source plus K2048/K5120 measured width masks. */
struct SelectedHeadGolden { TensorType source; std::array<uint16_t,2> rows; };
constexpr std::array<SelectedHeadGolden,21> selectedHeadGolden{{
    {TensorType::Q4_0,{0x7ff8,0x7ff8}},
    {TensorType::IQ4_NL,{0x7ff8,0x7ff8}},
    {TensorType::IQ4_XS,{0x7ff8,0x7ff8}},
    {TensorType::Q4_1,{0x7ff8,0x7ff8}},
    {TensorType::Q4_K,{0x7ff8,0x7ff8}},
    {TensorType::Q5_0,{0x7ff8,0x7ff8}},
    {TensorType::Q5_1,{0x7ff8,0x7ff8}},
    {TensorType::Q5_K,{0x7ff8,0x7ff8}},
    {TensorType::Q6_K,{0x7ff8,0x7ff8}},
    {TensorType::Q3_K,{0x7ff8,0x7ff8}},
    {TensorType::Q2_K,{0x7ff8,0x7ff8}},
    {TensorType::IQ3_S,{0x7ff8,0x7ff8}},
    {TensorType::IQ3_XXS,{0x7ff8,0x7ff8}},
    {TensorType::IQ2_S,{0x7ff8,0x7ff8}},
    {TensorType::IQ2_XS,{0x7ff8,0x7ff8}},
    {TensorType::IQ2_XXS,{0x7ff8,0x7ff8}},
    {TensorType::IQ1_S,{0x7ff0,0x7f80}},
    {TensorType::IQ1_M,{0x7f80,0x7e80}},
    {TensorType::Q8_0,{0x7ff8,0x7ff8}},
    {TensorType::Q8_1,{0x7ff8,0x7ff8}},
    {TensorType::Q8_K,{0x7ff8,0x7ff8}},
}};
/** @param source Original tensor format. @param n Original full vocabulary.
 * @param k Original contraction width. @param m Exact physical selected rows.
 * @return Expected Automatic engine at this measured key, without interpolation. */
constexpr bool selectedHeadExpectsWmma(TensorType source,int n,int k,int m) noexcept
{
    if(n!=248320 || (k!=2048 && k!=5120) || m<2 || m>16)return false;
    for(const auto &entry:selectedHeadGolden)if(entry.source==source)
        return (entry.rows[k==2048 ? 0 : 1]&(uint16_t{1}<<(m-2)))!=0;
    return false;
}
}
