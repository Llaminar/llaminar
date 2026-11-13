/**
 * @file SnapshotContiguousRows.h
 * @brief Shared host/device interpretation of authored contiguous diagnostic rows.
 *
 * The resolver consumes already acquired native words. It never reads execution
 * state itself, publishes a count to inference, or treats malformed metadata as
 * empty. Callers retain the original frontier owner and exact producer ordering.
 */
#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SNAPSHOT_ROWS_HD __host__ __device__
#else
#define LLAMINAR_SNAPSHOT_ROWS_HD
#endif

namespace llaminar2
{
/** @brief Contiguous axes; packed selections require their separate offset contract. */
enum class SnapshotContiguousRowKind : std::uint8_t
{
    Physical,
    TokenRows,
    StatePrefixRows
};

/** @brief Frozen geometry, independent of current native count values. */
struct SnapshotContiguousRowGeometry
{
    SnapshotContiguousRowKind kind = SnapshotContiguousRowKind::Physical;
    std::size_t capacity = 0;
    std::int32_t sourceCapacity = 0;
    std::int32_t sourceRowsPerStateRow = 1;
};

/** @brief Invalid and valid-empty observations are distinct diagnostic outcomes. */
struct SnapshotContiguousRowObservation
{
    bool valid = false;
    std::size_t rows = 0;
};

/**
 * @brief Resolve one acquired frontier with identical CPU/CUDA/HIP arithmetic.
 * @param geometry Frozen output capacity and authored mathematical axis.
 * @param prior Acquired prior source rows (zero for token/physical axes).
 * @param live Acquired produced/token rows; unused for physical geometry.
 * @return Checked live rows, or invalid without any payload access.
 */
LLAMINAR_SNAPSHOT_ROWS_HD inline SnapshotContiguousRowObservation
resolveSnapshotContiguousRows(SnapshotContiguousRowGeometry geometry,
                              std::int32_t prior, std::int32_t live) noexcept
{
    if (!geometry.capacity)
        return {};
    if (geometry.kind == SnapshotContiguousRowKind::Physical)
        return {true, geometry.capacity};
    if (prior < 0 || live < 0)
        return {};
    std::uint64_t rows = static_cast<std::uint32_t>(live);
    if (geometry.kind == SnapshotContiguousRowKind::StatePrefixRows)
    {
        const auto total = static_cast<std::uint64_t>(prior) + rows;
        if (geometry.sourceCapacity <= 0 || geometry.sourceRowsPerStateRow <= 0 ||
            total > static_cast<std::uint64_t>(geometry.sourceCapacity))
            return {};
        rows = total / static_cast<std::uint32_t>(geometry.sourceRowsPerStateRow);
    }
    else if (geometry.kind != SnapshotContiguousRowKind::TokenRows || prior != 0)
        return {};
    if (rows > geometry.capacity)
        return {};
    return {true, static_cast<std::size_t>(rows)};
}
} // namespace llaminar2
#undef LLAMINAR_SNAPSHOT_ROWS_HD
