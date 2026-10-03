/**
 * @file ForwardGraphEntryOrdering.h
 * @brief Typed incoming completion dependency for reused participant storage.
 *
 * Retained executables may own different streams while sharing activation,
 * recurrent, KV and chunk-metadata addresses. Their geometry is immutable;
 * their shared bytes are not independent. The participant prelude must acquire
 * the preceding output publication before the next graph can overwrite them.
 * Native parent graphs carry that dependency internally; this policy governs
 * externally submitted forwards only and never asks the host to await a GPU.
 */
#pragma once

#include "../graph/IGraphBuilder.h"

namespace llaminar2
{
    /** @brief Whether this participant owns terminal output or follows a PP edge. */
    enum class ForwardParticipantOutputRole
    {
        TerminalOwner,
        PipelineFollower,
    };

    /** @brief The incoming producer frontier required by one graph submission. */
    enum class ForwardGraphEntryDependency
    {
        PublishedRequestState,
        PreviousForwardCompletion,
    };

    /**
     * @brief Classify storage reuse without consulting mutable device values.
     * @param input Complete participant-local forward declaration.
     * @param owner_device Canonical physical owner of the reused execution storage.
     * @param role The topology-declared owner of terminal output.
     * @return A preceding forward edge for followers and resident chunk views.
     *
     * Same bucket width does not prove same stream. Conversely, setup merely
     * records a graph and must not depend on an earlier inference publication.
     * CPU execution is host-owned and has no native event dependency here.
     * The request's placement hint is not the storage authority: callers can
     * supply their exact execution device separately from that declaration.
     */
    [[nodiscard]] inline ForwardGraphEntryDependency forwardGraphEntryDependency(
        const ForwardInput &input, DeviceId owner_device,
        ForwardParticipantOutputRole role) noexcept
    {
        if (!owner_device.is_gpu() || input.graph_submission_intent ==
            ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch)
            return ForwardGraphEntryDependency::PublishedRequestState;
        if (role == ForwardParticipantOutputRole::PipelineFollower || input.device_prefill_chunk)
            return ForwardGraphEntryDependency::PreviousForwardCompletion;
        return ForwardGraphEntryDependency::PublishedRequestState;
    }
}
