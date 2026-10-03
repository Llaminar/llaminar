/**
 * @file TPLocalReduceOverlap.h
 * @brief Declarative captured collectives overlapping independent compute.
 *
 * The builder owns both ends of the event DAG. Callers name a tensor producer
 * and one independent participant-local compute stage, never an unpaired event
 * or stream. The reduction retains its native backend and arithmetic order.
 */
#pragma once

#include "TPAllreduceStage.h"

namespace llaminar2
{
    class ComputeGraph;

    /** @brief The two existing nodes defining one bounded overlap opportunity. */
    struct TPLocalReduceOverlapWindow
    {
        std::string producer; ///< Produces the complete participant-local partial.
        std::string independent_compute; ///< Does not consume or overwrite that partial.
    };

    /**
     * @brief Add a paired native reduce submission and completion to a graph.
     *
     * The returned completion identity is params.stage_name; submission is
     * named with a `_submit` suffix. The builder orders producer -> submission
     * -> independent compute -> completion. Native events fork the reduction
     * onto one persistent auxiliary stream and join it at completion, so the
     * middle compute is not ordered after the reduction on the device.
     *
     * @param graph Participant-local graph containing both window nodes.
     * @param params Native ReduceSum contract, without sidebands or mapped transport.
     * @param window Existing producer and independent compute identities.
     * @throws std::invalid_argument for an unsafe or incomplete overlap contract.
     * @note A heterogeneous ticket boundary is not an admissible overlap window.
     *       Both event edges must remain inside the same native executable.
     */
    void addTPLocalReduceOverlap(
        ComputeGraph &graph,
        TPLocalRootedCollectiveStage::Params params,
        const TPLocalReduceOverlapWindow &window);

    /**
     * @brief Overlap an existing captured allgather with independent local compute.
     * @param graph Complete participant-local declaration, before preparation.
     * @param gather_node Existing CapturedAllGatherStage with exactly one producer.
     * @param independent_compute Existing stage that touches neither collective bank.
     * @throws std::invalid_argument For aliases, cycles or non-native boundaries.
     *
     * The gather node becomes the completion publication; its captured submission
     * is inserted after its producer and before the independent compute. The
     * same checked fork/join transaction used by rooted reduction owns both
     * events. Bytes, communicator, arithmetic and admitted storage are unchanged.
     * Native NCCL/RCCL and device-counted channels retain their own transport;
     * neither is substituted merely to make an overlap window capturable.
     */
    void overlapTPLocalAllGather(
        ComputeGraph &graph,
        const std::string &gather_node,
        const std::string &independent_compute);

    /**
     * @brief Fork an existing native allreduce around independent local compute.
     * @param graph Participant-local graph before native preparation.
     * @param allreduce_node Existing TPAllreduceStage with one producer dependency.
     * @param independent_compute Existing non-collective stage touching no partial bytes.
     * @throws std::invalid_argument For aliases, cycles, sidebands or invalid arithmetic bindings.
     *
     * The completion keeps the original identity and inout contract. Transport
     * precision and scratch remain owned by TPAllreduceStage/LocalTPContext.
     * Canonical rank-order sums retain their native allgather plus device fold;
     * their scratch is declared at both event edges and admitted before capture.
     * Callers must declare subsequent collective dependencies on this completion,
     * just as they would for any other graph-visible collective operation.
     */
    void overlapTPLocalAllreduce(
        ComputeGraph &graph,
        const std::string &allreduce_node,
        const std::string &independent_compute);

    /**
     * @brief Fork a column reduce-scatter around independent participant-local work.
     * @param graph Participant-local graph before capture preparation.
     * @param scatter_node Existing TPColumnReduceScatterStage with one producer.
     * @param independent_compute Local compute touching neither partial nor packing bank.
     * @throws std::invalid_argument For aliases, cycles or a non-native capture boundary.
     *
     * Packing, optional precision conversion and native reduction stay together
     * on the existing auxiliary stream. Both written banks remain live until
     * the same paired event join publishes them on the graph stream.
     */
    void overlapTPLocalColumnReduceScatter(
        ComputeGraph &graph,
        const std::string &scatter_node,
        const std::string &independent_compute);
}
