/**
 * @file ROCmWoVerifierExpectations.h
 * @brief Independent expected native engines from frozen complete Wo observations.
 *
 * Enabled stage capture and selector replay share this small golden fixture.
 * It includes no production dispatch table or selector implementation. Every
 * source alias retains its own tensor identity and nonmonotone width mask.
 * The first actual-Automatic panel further removes keys below5% median gain.
 * Seed receipt SHA256 1c9e1bbbb442aa318809f0a167883963683ce744a2161ac789d39c136754ad0c.
 */
#pragma once
#include "QuantizedVerifierFormats.h"
#include <array>
#include <bit>
#include <utility>

namespace llaminar2::test
{
/** @brief Original tensor-format identity and six independently measured width masks. */
struct WoVerifierExpectedRow {TensorType source;std::array<uint16_t,6> rows;};
/** @brief N/K order from actual full and TP-sized local model projection extents. */
inline constexpr std::array<std::pair<int,int>,6> kWoVerifierExpectedShapes{{
    {5120,6144},{5120,3072},{5120,1536},{2048,4096},{2048,2048},{2048,1024}}};
/** @brief Frozen all31-pair plus5% seed decisions, independent of installed policy source. */
inline constexpr std::array<WoVerifierExpectedRow,21> kWoVerifierExpectedRows{{
    {TensorType::Q4_0,{0x4000,0x3400,0x3800,0x7f80,0x7800,0x0000}}, // Q4_0
    {TensorType::IQ4_NL,{0x7800,0x7b00,0x7e00,0x7f80,0x7f00,0x7800}}, // IQ4_NL
    {TensorType::IQ4_XS,{0x7800,0x6f00,0x7f00,0x7f80,0x7f00,0x7000}}, // IQ4_XS
    {TensorType::Q4_1,{0x6000,0x5800,0x7f00,0x7fc0,0x7c00,0x7800}}, // Q4_1
    {TensorType::Q4_K,{0x0000,0x2600,0x7e00,0x7f80,0x7c00,0x7800}}, // Q4_K
    {TensorType::Q5_0,{0x7f80,0x7fe8,0x7f80,0x7ff8,0x7fc0,0x7f00}}, // Q5_0
    {TensorType::Q5_1,{0x7f80,0x7f80,0x7f80,0x7f80,0x7f80,0x7f80}}, // Q5_1
    {TensorType::Q5_K,{0x7f80,0x7f80,0x7f80,0x7ff0,0x7f80,0x7d80}}, // Q5_K
    {TensorType::Q6_K,{0x7f80,0x7f80,0x7fe0,0x7ff8,0x7f80,0x7f80}}, // Q6_K
    {TensorType::Q3_K,{0x7f00,0x7b00,0x7800,0x7f80,0x7e00,0x7800}}, // Q3_K
    {TensorType::Q2_K,{0x7c00,0x7c00,0x7800,0x7f00,0x7800,0x3000}}, // Q2_K
    {TensorType::IQ3_S,{0x7f80,0x7f80,0x7c00,0x7fc0,0x7f80,0x7f00}}, // IQ3_S
    {TensorType::IQ3_XXS,{0x7f80,0x7f80,0x7800,0x7fc0,0x7f80,0x7e00}}, // IQ3_XXS
    {TensorType::IQ2_S,{0x7f80,0x7f80,0x7800,0x7f80,0x7f80,0x7f00}}, // IQ2_S
    {TensorType::IQ2_XS,{0x7f80,0x7f80,0x6800,0x7f80,0x7f80,0x7e00}}, // IQ2_XS
    {TensorType::IQ2_XXS,{0x7f80,0x7f80,0x0000,0x7f80,0x7f80,0x7000}}, // IQ2_XXS
    {TensorType::IQ1_S,{0x7d00,0x7b80,0x0000,0x7fc0,0x7f80,0x7e00}}, // IQ1_S
    {TensorType::IQ1_M,{0x7800,0x7c00,0x0000,0x7e00,0x7800,0x3000}}, // IQ1_M
    {TensorType::Q8_0,{0x7800,0x0000,0x7e80,0x7f80,0x7800,0x0000}}, // Q8_0
    {TensorType::Q8_1,{0x7800,0x0800,0x3e00,0x7f00,0x7800,0x0000}}, // Q8_1
    {TensorType::Q8_K,{0x7a00,0x2000,0x7f00,0x7f00,0x7800,0x0000}}, // Q8_K
}};
static_assert([]{size_t selected=0;for(const auto &entry:kWoVerifierExpectedRows)for(auto rows:entry.rows)selected+=std::popcount(rows);return selected==746;}());
/** @brief Resolve the expected engine for one ordinary original projection.
 * @param source Original tensor identity, never the prepared physical decoder alias.
 * @param n Physical output columns. @param k Local reduction width.
 * @param m Exact hosted physical rows; guarded capacities are excluded by the caller.
 * @return Whether frozen independent complete-operation evidence requires WMMA. */
inline constexpr bool expectedROCmWoVerifierNative(TensorType source,int n,int k,int m) noexcept
{
    if(m<2 || m>16)return false;
    for(size_t shape=0;shape<kWoVerifierExpectedShapes.size();++shape)
        if(kWoVerifierExpectedShapes[shape]==std::pair{n,k})
            for(const auto &entry:kWoVerifierExpectedRows)
                if(entry.source==source)return (entry.rows[shape]&(uint16_t{1}<<(m-2)))!=0;
    return false;
}
}
