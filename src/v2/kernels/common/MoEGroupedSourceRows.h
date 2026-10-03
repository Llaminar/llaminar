/**
 * @file MoEGroupedSourceRows.h
 * @brief Checked borrowed route-to-token addressing for grouped expert inputs.
 *
 * A router publishes one canonical quantized hidden row per token. Grouping
 * reorders routes, not those activation bytes: every selected expert may read
 * the same original row directly. This immutable capture descriptor keeps the
 * route map's encoding explicit without materializing a top-k-expanded copy.
 * It owns neither storage nor placement and never reads a GPU map on the host.
 */
#pragma once

#include <stdexcept>
#include <type_traits>
#include <limits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_SOURCE_HD __host__ __device__
#else
#define LLAMINAR_MOE_SOURCE_HD
#endif

namespace llaminar2
{
    /** @brief Sealed source-row geometry borrowed by one captured gate/up launch. */
    class MoEGroupedSourceRows final
    {
    public:
        /**
         * @brief Bind a grouping map whose entries are original token indices.
         * @param indices Stable device map, ordered by grouped route slot.
         * @param rows Positive original activation-row capacity.
         * @throws std::invalid_argument If either required owner is absent.
         */
        [[nodiscard]] static MoEGroupedSourceRows tokenIndices(const int *indices, int rows)
        {
            return bind(indices, rows, 1);
        }

        /**
         * @brief Bind original route IDs, with token = route / top_k.
         * @param indices Stable device map, ordered by grouped route slot.
         * @param rows Original activation-row capacity, not the route capacity.
         * @param top_k Positive number of routes per original token.
         * @throws std::invalid_argument On missing storage or invalid geometry.
         */
        [[nodiscard]] static MoEGroupedSourceRows routeIndices(const int *indices, int rows, int top_k)
        {
            return bind(indices, rows, top_k);
        }

        /**
         * @brief Decode one published map entry without dereferencing its owner.
         * @return The source token or -1 for an invalid publication; never clamps.
         */
        [[nodiscard]] LLAMINAR_MOE_SOURCE_HD constexpr int decode(int index) const noexcept
        {
            if (index < 0) return -1;
            const int row = index / divisor_;
            return row < rows_ ? row : -1;
        }

        /** @return Original token-row capacity retained in capture identity. */
        [[nodiscard]] LLAMINAR_MOE_SOURCE_HD constexpr int capacity() const noexcept { return rows_; }

#if defined(__CUDACC__) || defined(__HIPCC__)
        /**
         * @brief Resolve a live grouped route after the grouping producer edge.
         * @param grouped_slot A live slot admitted by the expert count/offset.
         * @return Original token row; a corrupt publication traps before access.
         *
         * Resolve before the K loop, then retain the row in registers or the
         * CTA's bounded shared map, according to the projection's register budget.
         * Inactive routes must exit at their existing count/membership predicate.
         */
        [[nodiscard]] __device__ __forceinline__ int sourceRow(int grouped_slot) const
        {
            const int row = decode(indices_[grouped_slot]);
            if (row < 0)
            {
#if defined(__CUDA_ARCH__)
                __trap();
#else
                __builtin_trap();
#endif
            }
            return row;
        }
#endif

    private:
        /** @brief Validate immutable geometry without observing any device bytes. */
        [[nodiscard]] static MoEGroupedSourceRows bind(const int *indices, int rows, int divisor)
        {
            if (!indices || rows <= 0 || divisor <= 0 ||
                rows > std::numeric_limits<int>::max() / divisor)
                throw std::invalid_argument("MoE grouped source rows need a map and positive geometry");
            return MoEGroupedSourceRows(indices, rows, divisor);
        }

        /** @brief Construct only a validated, trivially copyable kernel argument. */
        constexpr MoEGroupedSourceRows(const int *indices, int rows, int divisor) noexcept
            : indices_(indices), rows_(rows), divisor_(divisor) {}
        const int *indices_; ///< Existing device-owned grouping publication.
        int rows_; ///< Physical source rows; never the mutable routed-row count.
        int divisor_; ///< One for token IDs, top-k for original route IDs.
    };

    static_assert(std::is_trivially_copyable_v<MoEGroupedSourceRows>);
    static_assert(std::is_standard_layout_v<MoEGroupedSourceRows>);
}

#undef LLAMINAR_MOE_SOURCE_HD
