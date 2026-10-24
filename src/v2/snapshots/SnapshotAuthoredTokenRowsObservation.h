/**
 * @file SnapshotAuthoredTokenRowsObservation.h
 * @brief Immutable acquired original-row provenance for authored diagnostic payloads.
 *
 * These values describe one completed diagnostic acquisition. They contain no
 * live tensor pointers and never select inference work. Packed destination rows
 * remain distinct from original physical source coordinates and logical replay.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace llaminar2
{
/** @brief Typed cause for an owned empty diagnostic observation, distinct from absence. */
enum class SnapshotEmptyReason : std::uint32_t {None=0,RankingNotRequired=1};
/** @brief One acquired original request and its exact packed destination extent. */
struct SnapshotAuthoredTokenRowRecord
{
    std::int32_t prefixRows=0; ///< Original logical history preceding this invocation.
    std::int32_t liveRows=0; ///< Original logical live rows, before query elision.
    std::int32_t rowOffset=0; ///< Original request physical base minus source-slice origin; may be negative.
    std::int32_t rowCapacity=0; ///< Original request logical capacity before head repetition.
    std::int32_t sourceCapacity=0; ///< Admitted bound on original logical prefix plus live.
    std::int32_t firstRequiredLogicalRow=0; ///< Frozen producer policy, separately authenticated by consumers.
    std::int32_t firstRow=0; ///< Slice-relative original physical first authored row.
    std::int32_t endRow=0; ///< Slice-relative original physical exclusive authored end.
    SnapshotEmptyReason emptyReason=SnapshotEmptyReason::None;
    std::uint32_t reserved=0; ///< Explicit zero protocol word; never uninitialized alignment bytes.
    std::uint64_t payloadRowOffset=0; ///< First packed destination row for this request.
    bool operator==(const SnapshotAuthoredTokenRowRecord &) const=default;
};
static_assert(std::is_trivially_copyable_v<SnapshotAuthoredTokenRowRecord>);
static_assert(sizeof(SnapshotAuthoredTokenRowRecord)==48);
static_assert(offsetof(SnapshotAuthoredTokenRowRecord,payloadRowOffset)==40);
/** @brief Complete immutable source geometry and acquired request-order packed row records. */
struct SnapshotAuthoredTokenRowsObservation
{
    std::vector<SnapshotAuthoredTokenRowRecord> requests; ///< Only original requests intersecting the source slice, in original order.
    std::size_t physicalRows=0; ///< Original admitted physical source-slice row count.
    int repeatsPerRow=1; ///< Physical rows per original logical token, for example prepared heads.
    int sliceFirstRow=0; ///< Original flattened physical origin of the admitted source slice.
    bool operator==(const SnapshotAuthoredTokenRowsObservation &) const=default;
};
} // namespace llaminar2
