/**
 * @file ExpertTierProjectionByteRange.h
 * @brief Compact original-coordinate byte ranges shared by local and MPI expert transfers.
 *
 * Native GPU planes are [K/32][N] arrays; floating projections are [N][K].
 * A wire cursor must respect both independent allocation strides and stop at a
 * noncontiguous boundary. Full inner extents coalesce across outer rows. This
 * value owns no memory, transport, residency state or execution authority.
 */
#pragma once

#include "ExpertTierWeightDeviceLayout.h"
#include <optional>

namespace llaminar2 {
/** @brief Storage order of one byte-preserving projection plane. */
enum class ExpertTierProjectionByteOrder { FloatingRows, NativeGPUBlocks };

/** @brief One compact run in independent original source/destination allocations. */
struct ExpertTierProjectionByteRange {
    std::size_t source_offset = 0;
    std::size_t destination_offset = 0;
    std::size_t bytes = 0;
};

/**
 * @brief Map a live wire prefix into both original roots without staging padding.
 * @param intersection Checked original source and destination ownership.
 * @param order Floating [N][K] elements or native [K/32][N] blocks.
 * @param element_bytes Size of one floating element or one native plane block.
 * @param wire_offset Already submitted compact bytes in this plane.
 * @param capacity Positive maximum bytes this submission may carry.
 * @return Exact run, or no value for empty, misaligned, overflowing or exhausted input.
 */
inline std::optional<ExpertTierProjectionByteRange> expertTierProjectionByteRange(
    const ExpertTierWeightProjectionIntersection& intersection,
    ExpertTierProjectionByteOrder order, std::size_t element_bytes,
    std::size_t wire_offset, std::size_t capacity) noexcept
{
    if(intersection.empty() || !element_bytes || !capacity)return std::nullopt;
    const auto& source=intersection.source();const auto& destination=intersection.destination();
    const bool native=order==ExpertTierProjectionByteOrder::NativeGPUBlocks;
    if(!native && order!=ExpertTierProjectionByteOrder::FloatingRows)return std::nullopt;
    if(native && (source.columns%32 || destination.columns%32 ||
        intersection.columns()%32 || intersection.sourceColumnOrigin()%32 ||
        intersection.destinationColumnOrigin()%32))return std::nullopt;
    const std::size_t inner=native?intersection.rows():intersection.columns();
    const std::size_t outer=native?intersection.columns()/32:intersection.rows();
    const std::size_t source_inner=native?source.rows:source.columns;
    const std::size_t destination_inner=native?destination.rows:destination.columns;
    const std::size_t source_outer=native?source.columns/32:source.rows;
    const std::size_t destination_outer=native?destination.columns/32:destination.rows;
    constexpr auto maximum=std::numeric_limits<std::size_t>::max();
    if(source_inner>maximum/element_bytes/source_outer ||
       destination_inner>maximum/element_bytes/destination_outer)return std::nullopt;
    const auto run=inner*element_bytes,total=run*outer;
    if(wire_offset>=total)return std::nullopt;
    const auto index=wire_offset/run,within=wire_offset%run;
    auto bytes=std::min(capacity,total-wire_offset);
    if(source_inner!=inner || destination_inner!=inner)bytes=std::min(bytes,run-within);
    const auto offset=[&](bool from_source) {
        const auto outer_origin=native?
            (from_source?intersection.sourceColumnOrigin():intersection.destinationColumnOrigin())/32:
            (from_source?intersection.sourceRowOrigin():intersection.destinationRowOrigin());
        const auto inner_origin=native?
            (from_source?intersection.sourceRowOrigin():intersection.destinationRowOrigin()):
            (from_source?intersection.sourceColumnOrigin():intersection.destinationColumnOrigin());
        return ((std::size_t(outer_origin)+index)*(from_source?source_inner:destination_inner)+
            std::size_t(inner_origin))*element_bytes+within;
    };
    return ExpertTierProjectionByteRange{offset(true),offset(false),bytes};
}
} // namespace llaminar2
