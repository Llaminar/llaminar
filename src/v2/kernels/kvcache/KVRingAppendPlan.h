/**
 * @file KVRingAppendPlan.h
 * @brief Allocation-free, immutable ring-append geometry and one final commit.
 *
 * Every destination derives from the original cursor, even when publication
 * contains more rows than the physical ring. Workers may write the retained
 * rows concurrently because those destinations are disjoint. The cache owner
 * publishes headAfter()/sizeAfter() only after all row producers finish.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
/** @brief Validated append transaction; does not own or mutate live cache state. */
class KVRingAppendPlan
{
public:
    /**
     * @brief Resolve a positive append against one valid immutable ring cursor.
     * @param capacity Number of physical token slots.
     * @param head Physical slot containing the oldest visible token.
     * @param size Number of visible tokens before this append.
     * @param rows Number of source token rows, including any eventually evicted.
     */
    constexpr KVRingAppendPlan(int capacity, int head, int size, int rows)
        : capacity_(capacity), head_(head), size_(size), rows_(rows)
    {
        if (capacity <= 0 || head < 0 || head >= capacity || size < 0 ||
            size > capacity || rows <= 0)
            throw std::invalid_argument("invalid KV ring append geometry");
    }

    /** @brief First source row that survives the complete append. */
    constexpr int firstRetainedSourceRow() const { return std::max(0, rows_ - capacity_); }
    /** @brief Number of disjoint physical destinations that must be written. */
    constexpr int retainedRows() const { return std::min(rows_, capacity_); }
    /** @brief Resolve one retained source row without observing mutable cursor state. */
    constexpr int destination(int source_row) const
    {
        if (source_row < firstRetainedSourceRow() || source_row >= rows_)
            throw std::out_of_range("KV append source row is outside the retained transaction");
        return static_cast<int>((static_cast<std::int64_t>(head_) + size_ + source_row) % capacity_);
    }
    /** @brief Number of old or newly appended rows evicted by the transaction. */
    constexpr std::int64_t overwrittenRows() const
    {
        return std::max<std::int64_t>(0, static_cast<std::int64_t>(size_) + rows_ - capacity_);
    }
    /** @brief Final oldest-token slot, published by the cache owner after writes. */
    constexpr int headAfter() const
    {
        return static_cast<int>((head_ + overwrittenRows()) % capacity_);
    }
    /** @brief Final visible length, published alongside headAfter(). */
    constexpr int sizeAfter() const
    {
        return static_cast<int>(std::min<std::int64_t>(capacity_, static_cast<std::int64_t>(size_) + rows_));
    }

private:
    int capacity_; ///< Immutable physical extent.
    int head_;     ///< Borrowed pre-transaction cursor value, not a live shadow.
    int size_;     ///< Borrowed pre-transaction visible length.
    int rows_;     ///< Complete source extent.
};
} // namespace llaminar2
