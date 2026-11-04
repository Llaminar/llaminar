/**
 * @file ExpertTierWeightWireTestOracle.h
 * @brief Compact native wire bytes from independently prepared CPU oracle roots.
 *
 * Native integration tests still compare against the production CPU packer's
 * exact bytes, while transport tails now exclude inactive stored N columns.
 * This helper never decodes, requantizes or reads an original model payload.
 */
#pragma once
#include "execution/moe/ExpertTierWeightDeviceLayout.h"
#include <span>
#include <stdexcept>
#include <vector>
namespace llaminar2::test {
/**
 * @brief Gather one exact wire prefix from a genuine CPU reference root.
 * @param layout Original endpoint/intersection layout.
 * @param root Independently prepared complete local CPU oracle allocation.
 * @param first Intersection-relative first unit.
 * @param count Exact live unit count.
 * @return Native payload and metadata bytes for those actual columns only.
 */
inline std::vector<uint8_t> expertTierWireOracle(const ExpertTierWeightDeviceLayout& layout,
    std::span<const uint8_t> root,uint32_t first,uint32_t count)
{
    std::vector<uint8_t> result(layout.chunkBytes(count,first));
    if(!gatherExpertTierCPUWeightChunk(layout,root,first,count,result))
        throw std::invalid_argument("CPU reference root cannot form exact native transfer chunk");
    return result;
}
}
