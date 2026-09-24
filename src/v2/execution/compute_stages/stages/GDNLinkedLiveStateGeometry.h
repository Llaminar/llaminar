/**
 * @file GDNLinkedLiveStateGeometry.h
 * @brief Typed geometry for modulo-linked GDN live-state publication.
 *
 * GDN tensor parallelism owns one contiguous query/key interval per
 * participant and every value-head interval linked to it.  Participant-local
 * banks are therefore group-major within a rank, while a raw allgather is
 * rank-major across participants.  This value type is the single accounting
 * authority for the permutation between those layouts.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace llaminar2
{
    /** @brief Stateful GDN tensor being transferred between TP layouts. */
    enum class GDNLinkedLiveStateKind
    {
        ConvHistory, ///< Fused `[Q | K | V repeats]` short-conv history.
        Recurrence,  ///< Value-head recurrence matrices only.
    };

    /**
     * @brief Fully resolved equal-TP layout for one GDN live-state tensor.
     *
     * A raw allgather requires the same send count on every participant, so
     * resolution succeeds only when key-head ownership is uniform.  Uneven TP
     * remains a valid production projection layout, but it deliberately does
     * not advertise this equal-count live-state collective.
     */
    struct GDNLinkedLiveStateShape
    {
        int degree = 0;                    ///< Number of LocalTP participants.
        int global_key_heads = 0;          ///< Global query/key head count.
        int local_key_heads = 0;           ///< Query/key heads per participant.
        int repeat_factor = 0;             ///< Value repeats per key head.
        int prefix_group_count = 0;        ///< Two for Q/K, zero for recurrence.
        int key_elements_per_head = 0;     ///< Elements in one Q or K head group.
        int value_elements_per_head = 0;   ///< Elements in one V head group.
        int local_state_floats = 0;        ///< Participant-local bank size.
        int full_state_floats = 0;         ///< Global group-major bank size.

        /** @brief Return the local width of one Q or K semantic group. */
        constexpr int localKeyGroupFloats() const noexcept
        {
            return local_key_heads * key_elements_per_head;
        }

        /** @brief Return the global width of one Q or K semantic group. */
        constexpr int fullKeyGroupFloats() const noexcept
        {
            return global_key_heads * key_elements_per_head;
        }

        /** @brief Return the local width of one value-repeat group. */
        constexpr int localValueGroupFloats() const noexcept
        {
            return local_key_heads * value_elements_per_head;
        }

        /** @brief Return the global width of one value-repeat group. */
        constexpr int fullValueGroupFloats() const noexcept
        {
            return global_key_heads * value_elements_per_head;
        }
    };

    /**
     * @brief Model-owned geometry for every GDN live-state handoff.
     *
     * The geometry contains model facts only.  Participant-local and full bank
     * sizes are derived for a requested tensor kind and TP degree, preventing
     * callers from passing a contradictory collection of sizes and booleans.
     */
    struct GDNLinkedLiveStateGeometry
    {
        int global_key_heads = 0;       ///< Global GDN query/key heads.
        int global_value_heads = 0;     ///< Global GDN value heads.
        int key_width = 0;              ///< Elements in one query/key vector.
        int value_width = 0;            ///< Elements in one value vector.
        int conv_history_length = 0;    ///< Persistent short-conv history rows.

        /**
         * @brief Resolve exact equal-participant sizes for one state kind.
         * @param kind Short-conv history or recurrence state.
         * @param degree LocalTP participant count.
         * @return A complete shape, or `nullopt` when the geometry cannot be
         *         represented by an equal-count raw allgather or an `int`
         *         backend state size.
         */
        std::optional<GDNLinkedLiveStateShape> resolve(
            GDNLinkedLiveStateKind kind,
            int degree) const noexcept
        {
            if (degree <= 0 || global_key_heads <= 0 ||
                global_value_heads <= 0 || key_width <= 0 ||
                value_width <= 0 ||
                global_value_heads % global_key_heads != 0 ||
                global_key_heads % degree != 0)
            {
                return std::nullopt;
            }

            const int repeat_factor = global_value_heads / global_key_heads;
            const int local_key_heads = global_key_heads / degree;
            const int prefix_group_count =
                kind == GDNLinkedLiveStateKind::ConvHistory ? 2 : 0;
            if (kind == GDNLinkedLiveStateKind::ConvHistory &&
                conv_history_length <= 0)
            {
                return std::nullopt;
            }

            const std::int64_t key_elements_per_head =
                kind == GDNLinkedLiveStateKind::ConvHistory
                    ? static_cast<std::int64_t>(key_width) * conv_history_length
                    : 0;
            const std::int64_t value_elements_per_head =
                kind == GDNLinkedLiveStateKind::ConvHistory
                    ? static_cast<std::int64_t>(value_width) * conv_history_length
                    : static_cast<std::int64_t>(key_width) * value_width;
            const std::int64_t local_state_floats =
                static_cast<std::int64_t>(prefix_group_count) *
                    local_key_heads * key_elements_per_head +
                static_cast<std::int64_t>(repeat_factor) *
                    local_key_heads * value_elements_per_head;
            const std::int64_t full_state_floats =
                static_cast<std::int64_t>(prefix_group_count) *
                    global_key_heads * key_elements_per_head +
                static_cast<std::int64_t>(repeat_factor) *
                    global_key_heads * value_elements_per_head;
            constexpr std::int64_t max_int =
                std::numeric_limits<int>::max();
            if (key_elements_per_head < 0 || key_elements_per_head > max_int ||
                value_elements_per_head <= 0 || value_elements_per_head > max_int ||
                local_state_floats <= 0 || local_state_floats > max_int ||
                full_state_floats <= 0 || full_state_floats > max_int ||
                full_state_floats != local_state_floats * degree)
            {
                return std::nullopt;
            }

            return GDNLinkedLiveStateShape{
                .degree = degree,
                .global_key_heads = global_key_heads,
                .local_key_heads = local_key_heads,
                .repeat_factor = repeat_factor,
                .prefix_group_count = prefix_group_count,
                .key_elements_per_head =
                    static_cast<int>(key_elements_per_head),
                .value_elements_per_head =
                    static_cast<int>(value_elements_per_head),
                .local_state_floats = static_cast<int>(local_state_floats),
                .full_state_floats = static_cast<int>(full_state_floats),
            };
        }

        /**
         * @brief Map a group-major full offset to rank-major allgather storage.
         *
         * This pure reference is used by CPU-only protocol tests.  GPU kernels
         * implement the same arithmetic on the exact capture stream.
         *
         * @param kind State tensor kind.
         * @param degree LocalTP degree.
         * @param full_offset Offset in the full group-major state bank.
         * @return Source offset in the raw rank-major allgather result.
         */
        std::optional<std::size_t> gatheredOffsetForFullOffset(
            GDNLinkedLiveStateKind kind,
            int degree,
            std::size_t full_offset) const noexcept
        {
            const auto shape = resolve(kind, degree);
            if (!shape || full_offset >=
                              static_cast<std::size_t>(shape->full_state_floats))
            {
                return std::nullopt;
            }

            const std::size_t full_key_group =
                static_cast<std::size_t>(shape->fullKeyGroupFloats());
            const std::size_t local_key_group =
                static_cast<std::size_t>(shape->localKeyGroupFloats());
            const std::size_t full_value_group =
                static_cast<std::size_t>(shape->fullValueGroupFloats());
            const std::size_t local_value_group =
                static_cast<std::size_t>(shape->localValueGroupFloats());
            const std::size_t full_prefix =
                static_cast<std::size_t>(shape->prefix_group_count) *
                full_key_group;
            const std::size_t local_prefix =
                static_cast<std::size_t>(shape->prefix_group_count) *
                local_key_group;

            std::size_t participant = 0;
            std::size_t participant_offset = 0;
            if (full_offset < full_prefix)
            {
                const std::size_t group = full_offset / full_key_group;
                const std::size_t within_group = full_offset % full_key_group;
                participant = within_group / local_key_group;
                participant_offset =
                    group * local_key_group + within_group % local_key_group;
            }
            else
            {
                const std::size_t value_offset = full_offset - full_prefix;
                const std::size_t repeat = value_offset / full_value_group;
                const std::size_t within_group = value_offset % full_value_group;
                participant = within_group / local_value_group;
                participant_offset =
                    local_prefix + repeat * local_value_group +
                    within_group % local_value_group;
            }
            return participant *
                       static_cast<std::size_t>(shape->local_state_floats) +
                   participant_offset;
        }

    };
} // namespace llaminar2
