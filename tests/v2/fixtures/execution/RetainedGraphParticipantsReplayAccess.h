/**
 * @file RetainedGraphParticipantsReplayAccess.h
 * @brief Closed access to ordinary source-graph lifecycle for device-free replay.
 *
 * Tests retain the actual executor, preparation, revocation and enqueue bodies.
 * This helper cannot construct raw native resources or mint their qualification;
 * those diagnostics remain confined to the separate backend owner fixtures.
 */
#pragma once
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/IRetainedGraphParticipants.h"

namespace llaminar2::test {
/** @brief Replay the actual executor and source-cache lifecycle without native diagnostics. */
struct RetainedGraphParticipantsReplayAccess final {
    /**
     * @brief Attach through the original executor's cold receiver adapter.
     * @param executor Actual executor containing the production adapter.
     * @param graph Original graph carrying the declarative source plan.
     * @param cache Original retained cache that owns the resulting borrow.
     * @param receiver Exact device of the native parent.
     * @param parent Original sealed parent before instantiation.
     * @return The production attachment result, preserving its failures.
     */
    static bool attachStorage(DeviceGraphExecutor& executor, ComputeGraph& graph,
        DeviceGraphExecutor::GraphSegmentCache& cache, DeviceId receiver, IGPUGraphCapture& parent)
    { return executor.attachRetainedParticipantStorageParent(graph, cache, receiver, parent); }

    /** @brief Authenticate the recorded owner through the actual first-capture body.
     * @param executor Actual executor. @param graph Original source graph.
     * @param cache Original retained cache. @param worker Original source worker. */
    static void authenticateRecording(DeviceGraphExecutor& executor, ComputeGraph& graph,
        DeviceGraphExecutor::GraphSegmentCache& cache, IWorkerGPUContext* worker)
    { executor.authenticateRetainedParticipantRecordingOwner(graph, cache, worker); }

    /** @brief Bind the existing cache stream through the production executor.
     * @param executor Actual executor. @param graph Original source graph.
     * @param cache Original retained cache. @param worker Original source worker. */
    static void bindPreparation(DeviceGraphExecutor& executor, ComputeGraph& graph,
        DeviceGraphExecutor::GraphSegmentCache& cache, IWorkerGPUContext* worker)
    { executor.bindRetainedParticipantPreparationStream(graph, cache, worker); }

    /** @brief Construct an actual preparation token with recorded source ownership.
     * @param original Original admitted source permit. @param stream Exact recording stream.
     * @return Token governed by the same production validation and revocation. */
    static std::shared_ptr<RetainedGraphParticipantPreparationStream> preparation(
        PreparedGPUContextUsePermit original, void* stream)
    { return std::shared_ptr<RetainedGraphParticipantPreparationStream>(
        new RetainedGraphParticipantPreparationStream(std::move(original), stream)); }

    /** @brief Construct a token for recorded immutable graph and arena provenance.
     * @param original Original admitted source permit. @param stream Exact recording stream.
     * @param definition Recorded original graph, arena, policy and generation.
     * @return Actual source-cache token; this grants no executable qualification. */
    static std::shared_ptr<RetainedGraphParticipantPreparationStream> definition(
        PreparedGPUContextUsePermit original, void* stream,
        RetainedGraphParticipantPreparationStream::Definition definition)
    { return std::shared_ptr<RetainedGraphParticipantPreparationStream>(
        new RetainedGraphParticipantPreparationStream(std::move(original), stream, std::move(definition))); }

    /** @brief Revoke before original graph teardown, using the production lifecycle.
     * @param original Actual token retained by the recorded source cache. */
    static void retire(RetainedGraphParticipantPreparationStream& original) { original.retire(); }

    /** @brief Build the actual enqueue owner around recorded asynchronous worker jobs.
     * @param jobs Original participant workers and enqueue bodies.
     * @return Actual group with production ordering and partial-failure behavior. */
    static std::unique_ptr<RetainedGraphParticipantEnqueueGroup> group(
        std::vector<RetainedGraphParticipantEnqueueGroup::Job> jobs)
    { return std::unique_ptr<RetainedGraphParticipantEnqueueGroup>(
        new RetainedGraphParticipantEnqueueGroup(std::move(jobs))); }

    /** @brief Replay the executor's source identity seal without native creation.
     * @param source Complete recorded original cache identity.
     * @return Read-only seal checked by the actual source plan. */
    static RetainedGraphParticipantSourceSeal sourceSeal(RetainedGraphParticipantSourceIdentity source)
    { return RetainedGraphParticipantSourceSeal(source); }

    /** @brief Replay the executor's assignment of its matched cold source definition.
     * @param progress Original auxiliary branch being attached.
     * @param value Original revocable source-cache token. */
    static void bindSource(IGraphCaptureAuxiliaryBranch& progress,
        std::shared_ptr<const RetainedGraphParticipantPreparationStream> value)
    { progress.bindOriginalSourceDefinition(std::move(value)); }
};
} // namespace llaminar2::test
