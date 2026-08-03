/**
 * @file DeviceExecutionTimeline.cpp
 * @brief Runtime validation and backend lowering for declarative event edges.
 */

#include "DeviceExecutionTimeline.h"

#include "../../../backends/IBackend.h"
#include "../../../utils/Logger.h"

#include <initializer_list>

namespace llaminar2
{
    namespace
    {
        constexpr DeviceTimelineRoleMask roles(
            std::initializer_list<DeviceTimelineRole> values) noexcept
        {
            DeviceTimelineRoleMask result = 0;
            for (const DeviceTimelineRole value : values)
                result |= deviceTimelineRoleBit(value);
            return result;
        }

        constexpr std::array<
            DeviceTimelineDependencySpec,
            static_cast<size_t>(DeviceTimelinePoint::Count)>
            kDeviceExecutionTimeline = {{
                {
                    .point = DeviceTimelinePoint::RequestStateResetReady,
                    .name = "request_state_reset_ready",
                    .producer = DeviceTimelineRole::RequestStateReset,
                    .consumers = roles({
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::RequestAdmissionTransfer,
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::PrefixRestoreMutation,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::GraphBuildDeviceStateReady,
                    .name = "graph_build_device_state_ready",
                    .producer =
                        DeviceTimelineRole::GraphBuildDeviceStatePublication,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::RequestStateReset,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::RequestInputAdmission,
                    .name = "request_input_admission",
                    .producer = DeviceTimelineRole::RequestAdmissionTransfer,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::RequestInputReuseReady,
                    .name = "request_input_reuse_ready",
                    .producer = DeviceTimelineRole::MainForwardGraph,
                    .consumers = roles({
                        DeviceTimelineRole::RequestAdmissionTransfer,
                        DeviceTimelineRole::MainForwardGraph,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::DeviceGenerationStateReady,
                    .name = "device_generation_state_ready",
                    .producer = DeviceTimelineRole::DeviceGenerationController,
                    .consumers = roles({
                        DeviceTimelineRole::DeviceGenerationController,
                        DeviceTimelineRole::AllPositionVerifier,
                        DeviceTimelineRole::HostResultBridge,
                        DeviceTimelineRole::RequestStateReset,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::StochasticDraftSampleReady,
                    .name = "stochastic_draft_sample_ready",
                    .producer = DeviceTimelineRole::DraftSampler,
                    .consumers = roles({
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::AllPositionVerifier,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::StochasticTargetSampleReady,
                    .name = "stochastic_target_sample_ready",
                    .producer = DeviceTimelineRole::TargetSampler,
                    .consumers = roles({
                        DeviceTimelineRole::AllPositionVerifier,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::ShiftedMTPKVReady,
                    .name = "shifted_mtp_kv_ready",
                    .producer = DeviceTimelineRole::MTPSidecarGraph,
                    .consumers = roles({
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::AllPositionVerifier,
                        DeviceTimelineRole::AcceptedStatePublication,
                        DeviceTimelineRole::PrefixCheckpointArchive,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::AllPositionVerifierReady,
                    .name = "all_position_verifier_ready",
                    .producer = DeviceTimelineRole::AllPositionVerifier,
                    .consumers = roles({
                        DeviceTimelineRole::VerifierSummary,
                        DeviceTimelineRole::AcceptedStatePublication,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::AcceptedSpecPublicationReady,
                    .name = "accepted_spec_publication_ready",
                    .producer = DeviceTimelineRole::AcceptedStatePublication,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::PrefixCheckpointArchive,
                        DeviceTimelineRole::PrefixRestoreMutation,
                        DeviceTimelineRole::MoERebalanceMaintenance,
                        DeviceTimelineRole::TargetSampler,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::LogicalSequenceStateReady,
                    .name = "logical_sequence_state_ready",
                    .producer = DeviceTimelineRole::AcceptedStatePublication,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::RankCollective,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::MTPTransactionReady,
                    .name = "mtp_transaction_ready",
                    .producer = DeviceTimelineRole::AcceptedStatePublication,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::RankCollective,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::MTPPrefillTerminalArchiveReady,
                    .name = "mtp_prefill_terminal_archive_ready",
                    .producer = DeviceTimelineRole::MainForwardGraph,
                    .consumers = roles({
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::LivePrefixCheckpointReady,
                    .name = "live_prefix_checkpoint_ready",
                    .producer = DeviceTimelineRole::PrefixCheckpointArchive,
                    .consumers = roles({
                        DeviceTimelineRole::PrefixRestoreMutation,
                        DeviceTimelineRole::HostArchiveBoundary,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::LivePrefixMutationReady,
                    .name = "live_prefix_mutation_ready",
                    .producer = DeviceTimelineRole::PrefixRestoreMutation,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::PrefixCheckpointArchive,
                        DeviceTimelineRole::PrefixRestoreMutation,
                        DeviceTimelineRole::MoERebalanceMaintenance,
                        DeviceTimelineRole::TargetSampler,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::PrefixPayloadReady,
                    .name = "prefix_payload_ready",
                    .producer = DeviceTimelineRole::PrefixPayloadTransfer,
                    .consumers = roles({
                        DeviceTimelineRole::PrefixRestoreMutation,
                        DeviceTimelineRole::HostArchiveBoundary,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::MoERebalanceMaintenanceReady,
                    .name = "moe_rebalance_maintenance_ready",
                    .producer = DeviceTimelineRole::MoERebalanceMaintenance,
                    .consumers = roles({
                        DeviceTimelineRole::MainForwardGraph,
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point = DeviceTimelinePoint::ForwardGraphOutputReady,
                    .name = "forward_graph_output_ready",
                    .producer = DeviceTimelineRole::MainForwardGraph,
                    .consumers = roles({
                        DeviceTimelineRole::MTPSidecarGraph,
                        DeviceTimelineRole::TargetSampler,
                        DeviceTimelineRole::VerifierSummary,
                        DeviceTimelineRole::AcceptedStatePublication,
                        DeviceTimelineRole::PrefixCheckpointArchive,
                        DeviceTimelineRole::MoERebalanceMaintenance,
                        DeviceTimelineRole::RankCollective,
                        DeviceTimelineRole::HostResultBridge,
                        DeviceTimelineRole::HostArchiveBoundary,
                        DeviceTimelineRole::RequestStateReset,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
                {
                    .point =
                        DeviceTimelinePoint::CompactSpeculativeResponseReady,
                    .name = "compact_speculative_response_ready",
                    .producer = DeviceTimelineRole::VerifierSummary,
                    .consumers = roles({
                        DeviceTimelineRole::AcceptedStatePublication,
                        DeviceTimelineRole::HostResultBridge,
                        DeviceTimelineRole::Diagnostics,
                    }),
                },
            }};

        const DeviceTimelineDependencySpec *specFor(
            DeviceTimelinePoint point) noexcept
        {
            const size_t index = static_cast<size_t>(point);
            if (index >= kDeviceExecutionTimeline.size())
                return nullptr;
            const auto &spec = kDeviceExecutionTimeline[index];
            return spec.point == point ? &spec : nullptr;
        }

        bool validDeviceEdgeArguments(
            DeviceId device,
            void *event,
            void *producer_stream) noexcept
        {
            return device.is_gpu() &&
                   event != nullptr &&
                   producer_stream != nullptr;
        }
    } // namespace

    const std::array<
        DeviceTimelineDependencySpec,
        static_cast<size_t>(DeviceTimelinePoint::Count)> &
    deviceExecutionTimelineManifest() noexcept
    {
        return kDeviceExecutionTimeline;
    }

    std::string_view deviceTimelinePointName(
        DeviceTimelinePoint point) noexcept
    {
        const auto *spec = specFor(point);
        return spec ? spec->name : "invalid";
    }

    std::string_view deviceTimelineRoleName(
        DeviceTimelineRole role) noexcept
    {
        switch (role)
        {
        case DeviceTimelineRole::RequestStateReset:
            return "request_state_reset";
        case DeviceTimelineRole::GraphBuildDeviceStatePublication:
            return "graph_build_device_state_publication";
        case DeviceTimelineRole::RequestAdmissionTransfer:
            return "request_admission_transfer";
        case DeviceTimelineRole::DeviceGenerationController:
            return "device_generation_controller";
        case DeviceTimelineRole::MainForwardGraph:
            return "main_forward_graph";
        case DeviceTimelineRole::MTPSidecarGraph:
            return "mtp_sidecar_graph";
        case DeviceTimelineRole::DraftSampler:
            return "draft_sampler";
        case DeviceTimelineRole::TargetSampler:
            return "target_sampler";
        case DeviceTimelineRole::AllPositionVerifier:
            return "all_position_verifier";
        case DeviceTimelineRole::VerifierSummary:
            return "verifier_summary";
        case DeviceTimelineRole::AcceptedStatePublication:
            return "accepted_state_publication";
        case DeviceTimelineRole::PrefixCheckpointArchive:
            return "prefix_checkpoint_archive";
        case DeviceTimelineRole::PrefixRestoreMutation:
            return "prefix_restore_mutation";
        case DeviceTimelineRole::PrefixPayloadTransfer:
            return "prefix_payload_transfer";
        case DeviceTimelineRole::MoERebalanceMaintenance:
            return "moe_rebalance_maintenance";
        case DeviceTimelineRole::RankCollective:
            return "rank_collective";
        case DeviceTimelineRole::HostResultBridge:
            return "host_result_bridge";
        case DeviceTimelineRole::HostArchiveBoundary:
            return "host_archive_boundary";
        case DeviceTimelineRole::Diagnostics:
            return "diagnostics";
        case DeviceTimelineRole::Count:
            break;
        }
        return "invalid";
    }

    bool DeviceEventEdge::validForPublication() const noexcept
    {
        const auto *spec = specFor(point_);
        return spec &&
               producer_ != DeviceTimelineRole::Count &&
               producer_ == spec->producer &&
               consumer_ == DeviceTimelineRole::Count;
    }

    bool DeviceEventEdge::validForConsumption() const noexcept
    {
        const auto *spec = specFor(point_);
        return spec &&
               producer_ != DeviceTimelineRole::Count &&
               producer_ == spec->producer &&
               consumer_ != DeviceTimelineRole::Count &&
               spec->allowsConsumer(consumer_);
    }

    bool DeviceEventEdge::publish(
        IBackend &backend,
        DeviceId device,
        void *event,
        void *producer_stream) const
    {
        if (!validForPublication() ||
            !validDeviceEdgeArguments(device, event, producer_stream))
        {
            LOG_ERROR("[DeviceExecutionTimeline] Invalid publication edge point="
                      << deviceTimelinePointName(point_)
                      << " producer=" << deviceTimelineRoleName(producer_)
                      << " device=" << device.toString()
                      << " event=" << event
                      << " stream=" << producer_stream);
            return false;
        }

        return backend.recordEvent(
            event,
            device.gpu_ordinal(),
            producer_stream);
    }

    bool DeviceEventEdge::enqueueWait(
        IBackend &backend,
        DeviceId device,
        void *event,
        void *producer_stream,
        void *consumer_stream) const
    {
        if (!validForConsumption() ||
            !validDeviceEdgeArguments(device, event, producer_stream) ||
            consumer_stream == nullptr)
        {
            LOG_ERROR("[DeviceExecutionTimeline] Invalid consumption edge point="
                      << deviceTimelinePointName(point_)
                      << " producer=" << deviceTimelineRoleName(producer_)
                      << " consumer=" << deviceTimelineRoleName(consumer_)
                      << " device=" << device.toString()
                      << " event=" << event
                      << " producer_stream=" << producer_stream
                      << " consumer_stream=" << consumer_stream);
            return false;
        }

        if (producer_stream == consumer_stream)
            return true;

        return backend.streamWaitEvent(
            consumer_stream,
            event,
            device.gpu_ordinal());
    }

    bool DeviceEventEdge::enqueuePublishedWait(
        IBackend &backend,
        DeviceId device,
        void *event,
        void *consumer_stream) const
    {
        if (!validForConsumption() ||
            !device.is_gpu() ||
            event == nullptr ||
            consumer_stream == nullptr)
        {
            LOG_ERROR("[DeviceExecutionTimeline] Invalid durable consumption edge point="
                      << deviceTimelinePointName(point_)
                      << " producer=" << deviceTimelineRoleName(producer_)
                      << " consumer=" << deviceTimelineRoleName(consumer_)
                      << " device=" << device.toString()
                      << " event=" << event
                      << " consumer_stream=" << consumer_stream);
            return false;
        }

        /*
         * The publication event, rather than a cache-owned stream pointer, is
         * the durable happens-before token. Always queue the wait: attempting a
         * same-stream elision would require retaining and dereferencing producer
         * identity beyond the producer stream's valid lifetime.
         */
        return backend.streamWaitEvent(
            consumer_stream,
            event,
            device.gpu_ordinal());
    }

} // namespace llaminar2
