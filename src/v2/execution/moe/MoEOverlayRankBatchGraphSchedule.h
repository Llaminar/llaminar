/**
 * @file MoEOverlayRankBatchGraphSchedule.h
 * @brief Graph-only fork/join wiring for independent sparse rank-pair channels.
 *
 * Each channel owns its persistent MPI send/receive buffers. Dispatch therefore
 * need not await another channel's numerical return. The shared FP32 destination
 * still has one ordered writer at a time: all sends precede the first return,
 * and subsequent returns retain the declaration's canonical arithmetic order.
 * This setup helper adds dependencies, not threads, buffers, or runtime barriers.
 */
#pragma once

#include "execution/local_execution/graph/ComputeGraph.h"

#include <span>
#include <stdexcept>
#include <string>
#include <unordered_set>

namespace llaminar2
{
    /** @brief One independent channel's dispatch and ordered return graph nodes. */
    struct MoEOverlayRankBatchGraphLane
    {
        std::string dispatch; ///< Publishes one immutable rank-pair payload asynchronously.
        std::string returned; ///< Consumes that channel and writes the shared reduction destination.
    };

    /**
     * @brief Fork every independent dispatch before joining ordered numerical returns.
     * @param graph Owner of the already declared dispatch/return stages.
     * @param lanes Independent channels in canonical return-accumulation order.
     * @throws std::invalid_argument For an empty, missing, aliased or mistyped lane.
     *
     * The caller declares the common producer dependency and any preceding
     * shared-buffer owner. A single-threaded host executor must enqueue all
     * sends before it enters a blocking receive, regardless of topological-sort
     * tie breaking. Only the first return needs every dispatch as a predecessor;
     * the ordered return chain carries that proof in O(lanes) graph edges.
     */
    inline void wireMoEOverlayRankBatchForkJoin(
        ComputeGraph &graph, std::span<const MoEOverlayRankBatchGraphLane> lanes)
    {
        if (lanes.empty())
            throw std::invalid_argument("Sparse rank-batch fork requires at least one channel");
        std::unordered_set<std::string> names;
        for (const auto &lane : lanes)
        {
            const auto *dispatch = graph.getNode(lane.dispatch);
            const auto *returned = graph.getNode(lane.returned);
            if (!dispatch || !returned || !dispatch->stage || !returned->stage ||
                !names.insert(lane.dispatch).second || !names.insert(lane.returned).second ||
                dispatch->stage->type() != ComputeStageType::MOE_RANK_BATCH_DISPATCH ||
                returned->stage->type() != ComputeStageType::MOE_RANK_BATCH_RETURN_REDUCE)
                throw std::invalid_argument("Sparse rank-batch fork requires distinct typed dispatch/return nodes");
        }
        for (const auto &lane : lanes)
            graph.addDependency(lanes.front().returned, lane.dispatch);
        for (size_t index = 1; index < lanes.size(); ++index)
            graph.addDependency(lanes[index].returned, lanes[index - 1].returned);
    }
}
