/**
 * @file SnapshotSelectedIds.h
 * @brief Exact comparison of canonical diagnostic selected-ID rows and counts.
 *
 * Legacy NumPy ID snapshots use FP32 containers. They remain identities: every
 * authored value must be an exact representable nonnegative INT32, strictly
 * increasing within its row, followed only by canonical -1 sentinels. Thousands
 * of padding cells cannot dilute a single wrong selected ID through cosine.
 */
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2
{
/** @brief Exact discrepancy counts, with native-prefix count mismatches explicit. */
struct SnapshotSelectedIdComparison
{
    std::size_t identityMismatches=0;
    std::size_t countMismatches=0;
    std::size_t malformedRows=0;
    /** @return Whether both exact IDs and authored row counts agree. */
    bool equal() const noexcept{return !identityMismatches&&!countMismatches&&!malformedRows;}
};
/** @brief Compare IDs and sentinel-derived counts without numerical tolerances.
 * @param actual Canonical host diagnostic rows, after captured native-count validation.
 * @param expected Canonical reference IDs with trailing -1, in the same exact shape.
 * @param columns Exact selection-row capacity; both payloads must contain whole rows.
 * @return Exact scalar/count discrepancies and malformed row count.
 */
inline SnapshotSelectedIdComparison compareSnapshotSelectedIds(std::span<const float> actual,
    std::span<const float> expected,std::size_t columns)
{
    if(!columns||actual.size()!=expected.size()||actual.size()%columns)
        throw std::invalid_argument("Exact selected-ID comparison requires identical complete row geometry");
    SnapshotSelectedIdComparison result;
    for(std::size_t base=0;base<actual.size();base+=columns)
    {
        std::size_t counts[2]={0,0};bool malformed=false;
        for(unsigned side=0;side<2;++side)
        {
            const auto values=side?expected:actual;bool ended=false;double prior=-1;
            for(std::size_t column=0;column<columns;++column)
            {
                const double value=values[base+column];
                if(value==-1){ended=true;continue;}
                if(ended||!std::isfinite(value)||value<0||value>std::numeric_limits<std::int32_t>::max()||
                   value!=std::floor(value)||value<=prior) malformed=true;
                prior=value;++counts[side];
            }
        }
        result.malformedRows+=malformed;result.countMismatches+=counts[0]!=counts[1];
        for(std::size_t column=0;column<columns;++column)
            result.identityMismatches+=actual[base+column]!=expected[base+column];
    }
    return result;
}
} // namespace llaminar2
