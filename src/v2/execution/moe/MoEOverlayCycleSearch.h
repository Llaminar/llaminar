/**
 * @file MoEOverlayCycleSearch.h
 * @brief Bounded deterministic cycle search shared by CPU and GPU placement.
 *
 * Experts are directed edges between their current and desired participants.
 * Many experts can name the same edge. Reachability must visit participants,
 * not enumerate every path through those parallel edges. Scratch belongs to
 * the caller's persistent policy workspace; this operation never allocates.
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_CYCLE_HD __host__ __device__
#else
#define LLAMINAR_MOE_CYCLE_HD
#endif

namespace llaminar2
{
    /** Search outcome; edge visits make the algorithmic work bound testable. */
    struct MoEOverlayCycleSearchResult
    {
        std::uint32_t length = 0u;
        std::uint64_t edge_visits = 0u;
    };

    /**
     * @brief Find the first lexicographic simple owner-to-target cycle.
     * @param current Valid participant ids for every current owner.
     * @param desired Valid participant ids for every desired owner.
     * @param excluded Indexed predicate/container marking unavailable experts.
     * @param participants Number of participant vertices.
     * @param experts Number of expert edges.
     * @param visited Caller-owned scratch with at least `participants` bytes.
     * @param sources Caller-owned DFS vertex stack of `participants` entries.
     * @param next_edges Caller-owned DFS edge cursors of `participants` entries.
     * @param cycle Output expert ids, capacity at least `participants` entries.
     * @return Cycle length and inspected-edge count; zero length means no cycle.
     *
     * Within one root search, an exhausted vertex stays visited. Revisiting it
     * via another parallel edge cannot reveal a new route to the root: any
     * route through an ancestor is explored by that ancestor before it exits.
     * Keeping the ascending edge order therefore preserves the canonical first
     * cycle while bounding work to participants * experts for each root edge.
     * In the balanced placement multigraph the first available edge belongs to
     * a cycle, so only one root search is needed. Diagnostics can inspect work
     * counts; callers that consume only length compile that bookkeeping away.
     */
    template <typename Excluded>
    LLAMINAR_MOE_CYCLE_HD inline MoEOverlayCycleSearchResult
    findMoEOverlayPlacementCycle(
        const std::int32_t *current, const std::int32_t *desired,
        const Excluded &excluded, std::uint32_t participants,
        std::uint32_t experts, std::uint8_t *visited,
        std::uint32_t *sources, std::uint32_t *next_edges,
        std::uint32_t *cycle) noexcept
    {
        MoEOverlayCycleSearchResult result;
        for (std::uint32_t first = 0u; first < experts; ++first)
        {
            if (excluded[first] || current[first] == desired[first]) continue;
            const auto start = static_cast<std::uint32_t>(current[first]);
            const auto next = static_cast<std::uint32_t>(desired[first]);
            cycle[0] = first;
            for (std::uint32_t p = 0u; p < participants; ++p) visited[p] = 0u;
            visited[start] = visited[next] = 1u;
            std::uint32_t depth = 1u;
            sources[depth] = next;
            next_edges[depth] = 0u;
            while (true)
            {
                bool descended = false;
                const auto source = sources[depth];
                for (auto expert = next_edges[depth]; expert < experts; ++expert)
                {
                    ++result.edge_visits;
                    next_edges[depth] = expert + 1u;
                    if (excluded[expert] || expert == first ||
                        current[expert] != static_cast<std::int32_t>(source) ||
                        current[expert] == desired[expert]) continue;
                    const auto destination = static_cast<std::uint32_t>(desired[expert]);
                    cycle[depth] = expert;
                    if (destination == start)
                    {
                        result.length = depth + 1u;
                        return result;
                    }
                    if (!visited[destination] && depth + 1u < participants)
                    {
                        visited[destination] = 1u;
                        sources[++depth] = destination;
                        next_edges[depth] = 0u;
                        descended = true;
                        break;
                    }
                }
                if (descended) continue;
                if (depth == 1u) break;
                // This vertex has exhausted its edges. Do not reopen it when
                // another expert has the same destination later in the DFS.
                --depth;
            }
        }
        return result;
    }
}

#undef LLAMINAR_MOE_CYCLE_HD
