/**
 * @file RootedFloatRecordRange.h
 * @brief Borrowed source boundaries from the original rooted FP32 record gather.
 *
 * The transport authenticates participant IDs from its original communicator.
 * These offsets describe the actual compact receive prefix, never admission or
 * residency. Empty members retain an explicit zero-length source record.
 */
#pragma once
#include <cstddef>
#include <span>
#include <limits>
namespace llaminar2 {
/** @brief Original domain participant and its actual contiguous record prefix. */
struct RootedFloatRecordRange {
    int participant = -1;
    size_t first_record = 0;
    size_t record_count = 0;
};
/**
 * @brief Validate complete disjoint source coverage without allocating scratch.
 * @param ranges One range indexed by each original participant.
 * @param records Actual completed receive count, excluding allocation tail.
 * @return True only for complete, unique, nonoverlapping live coverage.
 */
inline bool validRootedFloatRecordRanges(std::span<const RootedFloatRecordRange> ranges,
                                        size_t records) noexcept {
    if (ranges.size()<2 || ranges.size()>static_cast<size_t>(std::numeric_limits<int>::max())) return false;
    size_t total=0;
    for (size_t i=0;i<ranges.size();++i) {
        const auto &a=ranges[i];
        if (a.participant!=static_cast<int>(i) || a.first_record>records ||
            a.record_count>records-a.first_record || a.record_count>records-total) return false;
        total+=a.record_count;
        for (size_t j=0;j<i;++j) {
            const auto &b=ranges[j];
            if (a.record_count && b.record_count && a.first_record<b.first_record+b.record_count &&
                b.first_record<a.first_record+a.record_count) return false;
        }
    }
    return total==records;
}
} // namespace llaminar2
