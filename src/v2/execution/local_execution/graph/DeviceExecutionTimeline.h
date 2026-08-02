/**
 * @file DeviceExecutionTimeline.h
 * @brief Declarative cross-graph GPU event ordering for device-owned inference.
 *
 * `ComputeGraph` is the source of truth for dependencies between stages inside
 * one executable graph. Device-owned inference also has a smaller set of
 * dependencies that cross graph boundaries: request admission must precede the
 * forward graph, verifier completion must precede accepted-state publication,
 * and publication must precede the next decode. Historically those edges were
 * expressed by unrelated event fields and backend calls spread through the
 * orchestrators.
 *
 * This file is the single policy surface for those cross-graph edges. The
 * constexpr manifest makes the complete producer/consumer topology reviewable
 * in one place. `DeviceEventEdge` validates fluent runtime declarations against
 * that manifest before it records an event or queues a stream wait.
 *
 * The class intentionally exposes no blocking host-wait operation. True host
 * result and archive boundaries remain explicit, separately reviewed APIs; an
 * execution edge can never turn into a synchronization fallback.
 */

#pragma once

#include "../../../backends/DeviceId.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace llaminar2
{
    class IBackend;

    /**
     * @brief Named publication points that connect independently launched GPU work.
     *
     * Additions require a matching entry in @ref deviceExecutionTimelineManifest.
     * `Count` is a sentinel used by totality tests and is never a valid point.
     */
    enum class DeviceTimelinePoint : uint8_t
    {
        RequestStateResetReady,
        GraphBuildDeviceStateReady,
        RequestInputAdmission,
        RequestInputReuseReady,
        StochasticDraftSampleReady,
        StochasticTargetSampleReady,
        ShiftedMTPKVReady,
        AllPositionVerifierReady,
        AcceptedSpecPublicationReady,
        LogicalSequenceStateReady,
        MTPTransactionReady,
        MTPPrefillTerminalArchiveReady,
        LivePrefixCheckpointReady,
        LivePrefixMutationReady,
        PrefixPayloadReady,
        MoERebalanceMaintenanceReady,
        ForwardGraphOutputReady,
        CompactSpeculativeResponseReady,
        Count,
    };

    /**
     * @brief Semantic GPU work owners used by cross-graph dependency declarations.
     *
     * Roles describe ownership, not a concrete CUDA/HIP stream identity. Several
     * roles may resolve to the same stream for one execution; in that case the
     * runtime edge validates normally and elides the redundant backend wait.
     */
    enum class DeviceTimelineRole : uint8_t
    {
        RequestStateReset,
        GraphBuildDeviceStatePublication,
        RequestAdmissionTransfer,
        MainForwardGraph,
        MTPSidecarGraph,
        DraftSampler,
        TargetSampler,
        AllPositionVerifier,
        VerifierSummary,
        AcceptedStatePublication,
        PrefixCheckpointArchive,
        PrefixRestoreMutation,
        PrefixPayloadTransfer,
        MoERebalanceMaintenance,
        RankCollective,
        HostResultBridge,
        HostArchiveBoundary,
        Diagnostics,
        Count,
    };

    using DeviceTimelineRoleMask = uint64_t;

    /** @return A mask containing exactly @p role. */
    constexpr DeviceTimelineRoleMask deviceTimelineRoleBit(
        DeviceTimelineRole role) noexcept
    {
        return DeviceTimelineRoleMask{1}
               << static_cast<uint8_t>(role);
    }

    /**
     * @brief One declarative producer-to-consumer topology entry.
     */
    struct DeviceTimelineDependencySpec
    {
        DeviceTimelinePoint point;
        std::string_view name;
        DeviceTimelineRole producer;
        DeviceTimelineRoleMask consumers;

        /** @return Whether @p role may consume this publication. */
        constexpr bool allowsConsumer(DeviceTimelineRole role) const noexcept
        {
            return (consumers & deviceTimelineRoleBit(role)) != 0;
        }
    };

    /**
     * @brief Return the complete cross-graph device execution timeline.
     *
     * Entries are ordered by `DeviceTimelinePoint`, enabling constant-time
     * lookup and a simple totality proof in unit tests.
     */
    const std::array<
        DeviceTimelineDependencySpec,
        static_cast<size_t>(DeviceTimelinePoint::Count)> &
    deviceExecutionTimelineManifest() noexcept;

    /** @return Human-readable stable name for diagnostics and PerfStats. */
    std::string_view deviceTimelinePointName(
        DeviceTimelinePoint point) noexcept;

    /** @return Human-readable stable role name for diagnostics. */
    std::string_view deviceTimelineRoleName(
        DeviceTimelineRole role) noexcept;

    /**
     * @brief Fluent, manifest-validated runtime representation of one event edge.
     *
     * Example:
     * @code
     * DeviceEventEdge::at(DeviceTimelinePoint::CompactSpeculativeResponseReady)
     *     .from(DeviceTimelineRole::VerifierSummary)
     *     .to(DeviceTimelineRole::HostResultBridge)
     *     .enqueueWait(backend, device, event, producer_stream, copy_stream);
     * @endcode
     */
    class DeviceEventEdge final
    {
    public:
        /** @brief Start a dependency declaration at a named timeline point. */
        static constexpr DeviceEventEdge at(
            DeviceTimelinePoint point) noexcept
        {
            return DeviceEventEdge(point);
        }

        /** @brief Declare the semantic producer role. */
        [[nodiscard]] constexpr DeviceEventEdge from(
            DeviceTimelineRole producer) const noexcept
        {
            DeviceEventEdge result = *this;
            result.producer_ = producer;
            return result;
        }

        /** @brief Declare the semantic consumer role. */
        [[nodiscard]] constexpr DeviceEventEdge to(
            DeviceTimelineRole consumer) const noexcept
        {
            DeviceEventEdge result = *this;
            result.consumer_ = consumer;
            return result;
        }

        /**
         * @brief Record publication after prior work on @p producer_stream.
         *
         * Event ownership and allocation remain with the containing lifecycle
         * object so hot paths can use preallocated or shared events.
         */
        bool publish(
            IBackend &backend,
            DeviceId device,
            void *event,
            void *producer_stream) const;

        /**
         * @brief Queue a nonblocking dependency on @p consumer_stream.
         *
         * Same-stream dependencies are validated and then elided because stream
         * order already supplies the required happens-before relationship.
         */
        bool enqueueWait(
            IBackend &backend,
            DeviceId device,
            void *event,
            void *producer_stream,
            void *consumer_stream) const;

        /**
         * @brief Queue a wait for an event whose producer stream is no longer live.
         *
         * Some publications deliberately outlive the graph-cache stream on which
         * they were recorded. In that case the event is the durable ownership
         * token and retaining the raw producer stream merely to compare stream
         * identities would create a dangling-pointer hazard. This operation
         * validates the same manifest edge as @ref enqueueWait, requires an
         * explicit consumer stream, and always lowers to `streamWaitEvent()`.
         *
         * @param backend Backend that owns @p event and @p consumer_stream.
         * @param device GPU device on which the event was published.
         * @param event Previously recorded, lifecycle-owned completion event.
         * @param consumer_stream Explicit stream that will consume the publication.
         * @return true when the backend accepted the nonblocking stream wait.
         */
        bool enqueuePublishedWait(
            IBackend &backend,
            DeviceId device,
            void *event,
            void *consumer_stream) const;

        /** @return Whether producer and consumer declarations match the manifest. */
        bool validForPublication() const noexcept;
        bool validForConsumption() const noexcept;

        DeviceTimelinePoint point() const noexcept { return point_; }
        DeviceTimelineRole producer() const noexcept { return producer_; }
        DeviceTimelineRole consumer() const noexcept { return consumer_; }

    private:
        explicit constexpr DeviceEventEdge(
            DeviceTimelinePoint point) noexcept
            : point_(point)
        {
        }

        DeviceTimelinePoint point_ = DeviceTimelinePoint::Count;
        DeviceTimelineRole producer_ = DeviceTimelineRole::Count;
        DeviceTimelineRole consumer_ = DeviceTimelineRole::Count;
    };

} // namespace llaminar2
