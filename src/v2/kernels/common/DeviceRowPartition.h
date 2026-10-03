/**
 * @file DeviceRowPartition.h
 * @brief Immutable participant coordinates for balanced device-owned live rows.
 *
 * A captured launch retains its maximum grid, but divides the current live
 * prefix, not allocation capacity. Complete rows have exactly one owner;
 * arithmetic within a row is never repartitioned. The descriptor owns neither
 * the live count nor storage and introduces no host shadow or publication word.
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_ROW_PARTITION_HD __host__ __device__
#else
#define LLAMINAR_ROW_PARTITION_HD
#endif

namespace llaminar2
{
    /** @brief Resolved contiguous source rows; a negative count rejects corruption. */
    struct DeviceOwnedRowSpan
    {
        int first = 0; ///< Original token coordinate, not a compact output offset.
        int count = -1; ///< Live rows; zero is a valid empty participant.
        /** @return Whether resolution accepted the complete count publication. */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD constexpr bool valid() const noexcept
        { return count >= 0; }
    };

    /**
     * @brief Frozen logical membership, independent of vendor or physical ordinal.
     *
     * The default is a single owner for shared kernel templates. Distributed
     * callers construct checked membership with balanced(). Resolution uses
     * quotient/remainder arithmetic, so INT_MAX capacities cannot overflow a
     * participant-times-live-count intermediate.
     */
    class DeviceRowPartition final
    {
    public:
        /** @brief Describe a single row owner without introducing a second mode. */
        constexpr DeviceRowPartition() noexcept = default;

        /** @brief Freeze one balanced participant in an ordered domain.
         * @param participant Logical coordinate in [0, participants).
         * @param participants Positive domain size; empty row owners are permitted.
         * @throws std::invalid_argument For malformed domain membership. */
        [[nodiscard]] static DeviceRowPartition balanced(int participant, int participants)
        {
            if (participants <= 0 || participant < 0 || participant >= participants)
                throw std::invalid_argument("DeviceRowPartition requires valid ordered membership");
            return DeviceRowPartition(participant, participants);
        }

        /** @return Frozen logical participant coordinate. */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD constexpr int participant() const noexcept
        { return participant_; }
        /** @return Frozen positive domain size. */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD constexpr int participants() const noexcept
        { return participants_; }

        /** @brief Resolve immutable geometry against an already observed count.
         * @param capacity Positive captured whole-matrix capacity.
         * @param live Device-authored count; tests may supply a value without a device.
         * @return Exact contiguous ownership or an invalid span, never a clamped count.
         *
         * Earlier participants own the remainder. This makes every participant's
         * maximum count monotone in capacity and keeps imbalance to at most one.
         */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD constexpr DeviceOwnedRowSpan resolveFor(
            int capacity, int live) const noexcept
        {
            return resolveMember(capacity, live, participant_, participants_);
        }

        /** @brief Resolve a peer's range using the same immutable domain geometry.
         * @param capacity Positive whole-matrix capacity.
         * @param live Authenticated count of live rows.
         * @param participant Source coordinate to inspect.
         * @param participants Frozen positive domain degree.
         * @return Exact peer range, or an invalid span for malformed geometry. */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD static constexpr DeviceOwnedRowSpan resolveMember(
            int capacity, int live, int participant, int participants) noexcept
        {
            if (capacity <= 0 || live < 0 || live > capacity || participants <= 0 ||
                participant < 0 || participant >= participants)
                return {};
            const int quotient = live / participants;
            const int remainder = live % participants;
            const int preceding_extra = participant < remainder ? participant : remainder;
            return {participant * quotient + preceding_extra,
                    quotient + (participant < remainder ? 1 : 0)};
        }

        /** @brief Invert the balanced partition without scanning participants.
         * @param capacity Positive whole-matrix capacity.
         * @param live Authenticated count of live rows.
         * @param row Original live row coordinate.
         * @return Unique owner, or -1 for padding/malformed geometry. */
        [[nodiscard]] LLAMINAR_ROW_PARTITION_HD constexpr int ownerFor(
            int capacity, int live, int row) const noexcept
        {
            if (!resolveFor(capacity, live).valid() || row < 0 || row >= live)
                return -1;
            const int quotient = live / participants_;
            const int remainder = live % participants_;
            // This product cannot exceed live. If quotient is zero, every
            // live row is in this first interval, so division by zero is absent.
            const int extended_rows = (quotient + (remainder != 0)) * remainder;
            return row < extended_rows ? row / (quotient + 1)
                                       : remainder + (row - extended_rows) / quotient;
        }

        /** @return Maximum local rows, not the next replay's live count.
         * @param capacity Positive admitted whole-matrix capacity.
         * @throws std::invalid_argument For invalid physical geometry. */
        [[nodiscard]] int capacityFor(int capacity) const
        {
            const auto span = resolveFor(capacity, capacity);
            if (!span.valid()) throw std::invalid_argument("DeviceRowPartition requires positive capacity");
            return span.count;
        }

#if defined(__CUDACC__) || defined(__HIPCC__)
        /** @brief Borrow the canonical device count after its ordered producer.
         * @param capacity Whole-matrix captured capacity.
         * @param count Non-null immutable address of the live INT32 authority.
         * @return Current source interval; malformed state traps before row access. */
        [[nodiscard]] __device__ __forceinline__ DeviceOwnedRowSpan resolve(
            int capacity, const std::int32_t *count) const
        {
            int live = count ? *count : -1;
#if defined(__HIP_DEVICE_COMPILE__)
            live = __builtin_amdgcn_readfirstlane(live);
#endif
            const auto span = resolveFor(capacity, live);
            if (!span.valid())
            {
#if defined(__CUDA_ARCH__)
                __trap();
#else
                __builtin_trap();
#endif
            }
            return span;
        }
#endif

    private:
        /** @brief Factories alone admit non-default membership. */
        constexpr DeviceRowPartition(int participant, int participants) noexcept
            : participant_(participant), participants_(participants) {}
        int participant_ = 0;
        int participants_ = 1;
    };
    static_assert(std::is_trivially_copyable_v<DeviceRowPartition>);
    static_assert(std::is_standard_layout_v<DeviceRowPartition>);
}
#undef LLAMINAR_ROW_PARTITION_HD
