/**
 * @file MoEOverlayRankBatchGraphSchedule.h
 * @brief Graph-only fork/join wiring for independent sparse rank-pair channels.
 *
 * Each channel owns its persistent MPI send/receive buffers. Dispatch therefore
 * need not await another channel's numerical return. The shared FP32 destination
 * still has one ordered writer at a time: all sends precede local expert work,
 * then returns retain the declaration's canonical arithmetic order. Waiting for
 * a remote return before starting local work would serialize CPU sockets even
 * though the transport already supports asynchronous dispatch. This setup
 * helper adds dependencies, not threads, buffers, or runtime barriers.
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

    /** @brief Participant-local work that can run while remote peers compute. */
    struct MoEOverlayLocalExpertGraphLane
    {
        std::string dispatch; ///< Consumes immutable routing into participant-owned storage.
        std::string compute;  ///< Produces private expert rows, not the shared reduction output.
        std::string returned; ///< Publishes those rows after the ordered remote returns.
    };

    /**
     * @brief Fork every independent dispatch before joining ordered numerical returns.
     * @param graph Owner of the already declared dispatch/return stages.
     * @param lanes Independent channels in canonical return-accumulation order.
     * @param local_lanes Independent local work, with returns ordered after remote lanes.
     * @throws std::invalid_argument For an empty, missing, aliased or mistyped lane.
     *
     * The caller declares the common producer dependency and any preceding
     * shared-buffer owner. A single-threaded host executor must enqueue all
     * sends and execute local work before it enters a blocking receive, regardless
     * of topological-sort tie breaking. Local compute writes participant-private
     * storage; only its return touches the shared destination. The ordered return
     * chain preserves arithmetic and lease retirement in O(lanes) graph edges.
     */
    inline void wireMoEOverlayRankBatchForkJoin(
        ComputeGraph &graph, std::span<const MoEOverlayRankBatchGraphLane> lanes,
        std::span<const MoEOverlayLocalExpertGraphLane> local_lanes = {})
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
        // Validate the complete declaration before publishing any dependency.
        // A local numerical return is not independent work: it still belongs
        // to the ordered join, after every remote return.
        for (const auto &lane : local_lanes)
        {
            const auto *dispatch = graph.getNode(lane.dispatch);
            const auto *compute = graph.getNode(lane.compute);
            const auto *returned = graph.getNode(lane.returned);
            if (!dispatch || !compute || !returned || !dispatch->stage ||
                !compute->stage || !returned->stage ||
                !names.insert(lane.dispatch).second ||
                !names.insert(lane.compute).second ||
                !names.insert(lane.returned).second ||
                dispatch->stage->type() != ComputeStageType::MOE_SPARSE_DISPATCH ||
                compute->stage->type() != ComputeStageType::MOE_LOCAL_EXPERT ||
                returned->stage->type() != ComputeStageType::MOE_SPARSE_RETURN_REDUCE)
                throw std::invalid_argument("Sparse local fork requires distinct typed dispatch/compute/return nodes");
        }
        for (const auto &lane : lanes)
            graph.addDependency(lanes.front().returned, lane.dispatch);
        for (size_t index = 1; index < lanes.size(); ++index)
            graph.addDependency(lanes[index].returned, lanes[index - 1].returned);
        if (!local_lanes.empty())
        {
            for (const auto &lane : lanes)
                graph.addDependency(local_lanes.front().dispatch, lane.dispatch);
            for (size_t index = 0; index < local_lanes.size(); ++index)
            {
                const auto &local = local_lanes[index];
                graph.addDependency(local.compute, local.dispatch);
                graph.addDependency(local.returned, local.compute);
                graph.addDependency(local.returned, index == 0
                    ? lanes.back().returned : local_lanes[index - 1].returned);
                if (index != 0)
                    graph.addDependency(local.dispatch, local_lanes[index - 1].compute);
            }
            // The host executor cannot wait on a peer before using its own
            // socket. Private compute may overlap; numerical publication may not.
            graph.addDependency(lanes.front().returned, local_lanes.back().compute);
        }
    }
}
