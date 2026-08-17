/**
 * @file GDNHeadAssignment.h
 * @brief Typed tensor-parallel ownership for modulo-linked GDN heads.
 *
 * Qwen Gated Delta Net layers can expose fewer query/key heads than value
 * heads.  Value head @c v consumes query/key head @c v % n_k_heads, so a
 * conventional contiguous value-head shard either crosses that dependency or
 * forces every participant to replicate all query/key projection rows.
 *
 * This value type makes the economical ownership rule explicit.  Each
 * participant owns a contiguous query/key-head interval and every value head
 * linked to that interval.  Local value heads are packed repeat-major, which
 * preserves the inexpensive local relation
 * @code
 * local_key_head = local_value_head % local_key_head_count
 * @endcode
 * without communication, lookup tables, or runtime host participation.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /**
     * @brief One contiguous interval of global GDN heads.
     */
    struct GDNHeadSpan
    {
        int start = 0; ///< First global head in the interval.
        int count = 0; ///< Number of heads in the interval.

        /** @brief Return the exclusive interval end. */
        constexpr int end() const noexcept { return start + count; }

        /** @brief Return whether this interval contains no heads. */
        constexpr bool empty() const noexcept { return count == 0; }

        constexpr bool operator==(const GDNHeadSpan &) const noexcept = default;
    };

    /**
     * @brief Complete participant-local assignment for modulo-linked GDN heads.
     *
     * The assignment is geometry-driven and independent of tensor format,
     * backend, model name, prompt, and measured router skew.  It can therefore
     * be shared by weight loading, graph buffer sizing, recurrent-state layout,
     * and correctness tests without duplicating sharding arithmetic.
     */
    class GDNHeadAssignment
    {
    public:
        /**
         * @brief Construct an assignment from a participant's proportional range.
         *
         * @param global_key_heads Global query/key head count.
         * @param global_value_heads Global value head count.  The GDN modular
         *        repeat contract requires this to be an integer multiple of the
         *        key-head count.
         * @param partition_start Start of this participant's range in an
         *        arbitrary integral partition space, normally attention heads.
         * @param partition_count Width of this participant's partition range.
         * @param partition_total Total width of the partition space.
         * @return A validated assignment whose local value ordering is
         *         repeat-major over the participant's local key heads.
         * @throws std::invalid_argument when the geometry cannot be represented
         *         exactly.  Rounding a head boundary would silently change the
         *         model's arithmetic and is therefore forbidden.
         */
        static GDNHeadAssignment fromPartition(
            int global_key_heads,
            int global_value_heads,
            int partition_start,
            int partition_count,
            int partition_total)
        {
            if (global_key_heads <= 0 || global_value_heads <= 0)
                throw std::invalid_argument("GDN head counts must be positive");
            if (partition_total <= 0 || partition_start < 0 || partition_count <= 0 ||
                partition_start > partition_total - partition_count)
            {
                throw std::invalid_argument("GDN partition range is invalid");
            }
            if (global_value_heads % global_key_heads != 0)
            {
                throw std::invalid_argument(
                    "GDN value-head count must be an integer multiple of key-head count");
            }

            const long long scaled_start =
                static_cast<long long>(global_key_heads) * partition_start;
            const long long scaled_end =
                static_cast<long long>(global_key_heads) *
                (partition_start + partition_count);
            if (scaled_start % partition_total != 0 ||
                scaled_end % partition_total != 0)
            {
                throw std::invalid_argument(
                    "GDN partition boundary does not align to an integral key head");
            }

            const int key_start = static_cast<int>(scaled_start / partition_total);
            const int key_end = static_cast<int>(scaled_end / partition_total);
            if (key_end <= key_start)
            {
                throw std::invalid_argument(
                    "GDN partition would assign no key heads to a participant");
            }

            return GDNHeadAssignment(
                global_key_heads,
                global_value_heads,
                key_start,
                key_end - key_start);
        }

        /**
         * @brief Construct an equal-rank assignment.
         *
         * @param global_key_heads Global query/key head count.
         * @param global_value_heads Global value head count.
         * @param rank Zero-based participant rank.
         * @param world_size Number of participants.
         */
        static GDNHeadAssignment forEqualRank(
            int global_key_heads,
            int global_value_heads,
            int rank,
            int world_size)
        {
            if (world_size <= 0 || rank < 0 || rank >= world_size)
                throw std::invalid_argument("GDN rank/world-size pair is invalid");
            return fromPartition(
                global_key_heads,
                global_value_heads,
                rank,
                1,
                world_size);
        }

        /** @brief Global query/key head count. */
        constexpr int globalKeyHeads() const noexcept { return global_key_heads_; }

        /** @brief Global value head count. */
        constexpr int globalValueHeads() const noexcept { return global_value_heads_; }

        /** @brief Participant-local contiguous query/key interval. */
        constexpr GDNHeadSpan keyHeads() const noexcept { return key_heads_; }

        /** @brief Number of value repeats per query/key head. */
        constexpr int repeatFactor() const noexcept { return repeat_factor_; }

        /** @brief Number of participant-local query/key heads. */
        constexpr int localKeyHeads() const noexcept { return key_heads_.count; }

        /** @brief Number of participant-local value heads. */
        constexpr int localValueHeads() const noexcept
        {
            return key_heads_.count * repeat_factor_;
        }

        /**
         * @brief Global value-head intervals in local packed order.
         *
         * One interval exists per repeat round.  Concatenating these intervals
         * produces the local V/Z/alpha/beta/A/dt and output-projection input
         * ordering consumed by the graph.
         */
        const std::vector<GDNHeadSpan> &valueHeadSpans() const noexcept
        {
            return value_head_spans_;
        }

        /**
         * @brief Convert one local value-head index to its global head index.
         *
         * @throws std::out_of_range when @p local_value_head is not local.
         */
        int globalValueHead(int local_value_head) const
        {
            if (local_value_head < 0 || local_value_head >= localValueHeads())
                throw std::out_of_range("Local GDN value-head index is out of range");
            const int repeat = local_value_head / key_heads_.count;
            const int local_key = local_value_head % key_heads_.count;
            return repeat * global_key_heads_ + key_heads_.start + local_key;
        }

        /**
         * @brief Resolve the local query/key head consumed by a local value head.
         */
        int localKeyHeadForValue(int local_value_head) const
        {
            if (local_value_head < 0 || local_value_head >= localValueHeads())
                throw std::out_of_range("Local GDN value-head index is out of range");
            return local_value_head % key_heads_.count;
        }

        /**
         * @brief Scale the key-head interval into tensor elements.
         *
         * @param elements_per_head Number of tensor rows/columns per key head.
         */
        GDNHeadSpan keyElementSpan(int elements_per_head) const
        {
            validateElementsPerHead(elements_per_head);
            return {.start = key_heads_.start * elements_per_head,
                    .count = key_heads_.count * elements_per_head};
        }

        /**
         * @brief Scale value-head intervals into tensor element intervals.
         *
         * @param elements_per_head Number of tensor rows/columns per value head.
         */
        std::vector<GDNHeadSpan> valueElementSpans(int elements_per_head) const
        {
            validateElementsPerHead(elements_per_head);
            std::vector<GDNHeadSpan> spans;
            spans.reserve(value_head_spans_.size());
            for (const GDNHeadSpan span : value_head_spans_)
            {
                spans.push_back({.start = span.start * elements_per_head,
                                 .count = span.count * elements_per_head});
            }
            return spans;
        }

        /**
         * @brief Number of rows in a packed local fused [Q|K|V] tensor.
         */
        std::size_t localFusedRows(int elements_per_head) const
        {
            validateElementsPerHead(elements_per_head);
            return static_cast<std::size_t>(
                (2 * localKeyHeads() + localValueHeads()) * elements_per_head);
        }

    private:
        GDNHeadAssignment(
            int global_key_heads,
            int global_value_heads,
            int key_head_start,
            int key_head_count)
            : global_key_heads_(global_key_heads),
              global_value_heads_(global_value_heads),
              key_heads_{.start = key_head_start, .count = key_head_count},
              repeat_factor_(global_value_heads / global_key_heads)
        {
            value_head_spans_.reserve(static_cast<std::size_t>(repeat_factor_));
            for (int repeat = 0; repeat < repeat_factor_; ++repeat)
            {
                value_head_spans_.push_back(
                    {.start = repeat * global_key_heads + key_head_start,
                     .count = key_head_count});
            }
        }

        static void validateElementsPerHead(int elements_per_head)
        {
            if (elements_per_head <= 0)
                throw std::invalid_argument("GDN elements-per-head must be positive");
        }

        int global_key_heads_ = 0;
        int global_value_heads_ = 0;
        GDNHeadSpan key_heads_;
        int repeat_factor_ = 0;
        std::vector<GDNHeadSpan> value_head_spans_;
    };

} // namespace llaminar2
