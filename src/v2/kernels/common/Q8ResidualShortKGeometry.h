/**
 * @file Q8ResidualShortKGeometry.h
 * @brief Production geometry for the ordered short-K signed-Q8 contribution tile.
 *
 * The immutable prepared geometry selects the same bounded kernel on CUDA and
 * HIP. Live row counts remain device-owned and never choose a launch family.
 * The original increasing-K contribution/fold arithmetic is unchanged.
 */
#pragma once
#include "Q8ResidualGroupedGeometry.h"
#include <optional>
#include <cstddef>

namespace llaminar2 {
/** @brief Shared launch/LDS column extent; one CUDA warp or half a gfx906 wave. */
inline constexpr int kQ8ResidualShortKColumns = 32;
/** @brief All launch/storage extents derive from immutable prepared geometry. */
struct Q8ResidualShortKGeometry {
    int groups; ///< Actual complete K32 contributions, never padding groups.
    int columns; ///< Adjacent output columns per contribution cohort.
    int threads; ///< One column/group contributor per physical thread.
    std::uint64_t blocks; ///< Complete and partial output-column tiles.
    std::size_t shared_bytes; ///< Two FP32 contribution planes, no global bank.
    bool operator==(const Q8ResidualShortKGeometry&) const = default;
};
/**
 * @brief Resolve ordered short-K production geometry without inspecting live device state.
 * @param backend Exact compiled CUDA/HIP engine; both use the same production geometry.
 * @param rows Actual retained physical rows; the ordered short-K family admits M1 only.
 * @param columns Actual prepared N, including a partial final tile.
 * @param reduction Actual prepared K in complete32 groups.
 * @param codebook Physical execution codebook; only unchanged signed Q8 accepts.
 * @return Geometry for K64..512 and N>=4K, otherwise explicit non-applicability.
 * @throws std::invalid_argument For malformed positive/native block geometry.
 */
inline std::optional<Q8ResidualShortKGeometry> q8ResidualShortKGeometry(
    Q8ResidualGroupedBackend backend,int rows,int columns,int reduction,std::uint8_t codebook) {
    if ((backend!=Q8ResidualGroupedBackend::CUDA&&backend!=Q8ResidualGroupedBackend::ROCm)||
        rows<=0||columns<=0||reduction<=0||reduction%32)
        throw std::invalid_argument("Ordered short-K requires positive native K32 geometry and an exact backend");
    if(rows!=1||codebook!=19||reduction<64||reduction>512||
        static_cast<std::uint64_t>(columns)<4ull*static_cast<unsigned>(reduction))
        return std::nullopt;
    const int groups=reduction/32;
    return Q8ResidualShortKGeometry{groups,kQ8ResidualShortKColumns,groups*kQ8ResidualShortKColumns,
        (static_cast<std::uint64_t>(columns)+kQ8ResidualShortKColumns-1)/kQ8ResidualShortKColumns,
        static_cast<std::size_t>(groups)*kQ8ResidualShortKColumns*2*sizeof(float)};
}
} // namespace llaminar2
