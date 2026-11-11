/**
 * @file Test__DeviceExecutionTimeline.cpp
 * @brief Structural tests for declarative cross-graph GPU event ordering.
 *
 * These tests are intentionally device-free. GPU integration tests prove that
 * backend event publication and waits execute correctly; this suite proves the
 * central policy itself is total, unique, and rejects undeclared edges before
 * any backend operation can be attempted.
 */

#include "execution/local_execution/graph/DeviceExecutionTimeline.h"
#include "execution/local_execution/orchestrators/PipelineDeviceGeneration.h"
#include "mocks/MockBackend.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string_view>

namespace llaminar2::test
{
    /** @test Native original-rank ACKs retain exact producer identity without model-state consumption. */
    TEST(Test__DeviceExecutionTimeline, OriginalPipelineProducerReceiptRecordsOnlyOriginalNativeCompletion)
    {
        using Receipt = PipelineDeviceGeneration::RankTransactionTerminal;
        using Role = MoEOverlayInferenceGraphRole;
        for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(2)})
        {
            MockBackend backend(device.type);
            void *const publication_stream = reinterpret_cast<void *>(0x81000);
            void *const completion_event = reinterpret_cast<void *>(0x82000);
            for (const auto role : {Role::MainDecode, Role::MTPConditionSpeculativeContinuation,
                                   Role::MTPConditionCommittedSerialToken, Role::MTPGroupedVerifier})
                for (const uint64_t request : {9u, 10u})
                {
                    backend.resetAll();
                    const Receipt receipt{.stream = publication_stream, .workspace = 7, .session = request, .role = role};
                    ASSERT_TRUE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                        backend, device, completion_event, receipt, 7, request));
                    const auto records = backend.getEventRecords();
                    ASSERT_EQ(records.size(), 1u);
                    EXPECT_EQ(records[0].type, MockBackend::EventRecord::RECORD);
                    EXPECT_EQ(records[0].event, completion_event);
                    EXPECT_EQ(records[0].stream, publication_stream);
                    EXPECT_EQ(records[0].device_id, device.ordinal);
                    EXPECT_EQ(backend.getEventWaitCount(), 0u);
                    EXPECT_EQ(backend.getH2DCount(), 0u);
                    EXPECT_EQ(backend.getD2HCount(), 0u);
                    EXPECT_EQ(backend.getSyncCount(), 0u);
                    EXPECT_EQ(backend.getStreamSyncCount(), 0u);
                }
            backend.resetAll();
            const Receipt original{.stream = publication_stream, .workspace = 7, .session = 9,
                .role = Role::MTPGroupedVerifier};
            const auto reject = [&](Receipt receipt) {
                std::string error;
                EXPECT_FALSE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                    backend, device, completion_event, receipt, 7, 9, &error));
                EXPECT_FALSE(error.empty());
                EXPECT_TRUE(backend.getEventRecords().empty());
            };
            for (void *stream : std::array<void *, 3>{nullptr, reinterpret_cast<void *>(1), reinterpret_cast<void *>(2)})
            {
                auto invalid = original; invalid.stream = stream; reject(invalid);
            }
            auto stale_workspace = original; ++stale_workspace.workspace; reject(stale_workspace);
            auto stale_request = original; ++stale_request.session; reject(stale_request);
            auto duplicate = original; duplicate.recorded = true; reject(duplicate);
            for (const auto role : {Role::None, Role::GenerationAdmission, Role::MainPrefill,
                                   Role::MTPDraft, static_cast<Role>(255)})
            {
                auto invalid = original; invalid.role = role; reject(invalid);
            }
            EXPECT_FALSE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                backend, device, nullptr, original, 7, 9));
            const auto foreign = device.is_cuda() ? DeviceId::rocm(device.ordinal) : DeviceId::cuda(device.ordinal);
            EXPECT_FALSE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                backend, foreign, completion_event, original, 7, 9));
            EXPECT_FALSE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                backend, device, completion_event, original, 0, 9));
            EXPECT_FALSE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                backend, device, completion_event, original, 7, 0));
            EXPECT_TRUE(backend.getEventRecords().empty());
        }
    }

    /** @test Accepted state flows to its real controller; ACK and final host observation remain distinct. */
    TEST(Test__DeviceExecutionTimeline, OriginalPipelinePublicationAckAndControllerKeepExactRoles)
    {
        const auto accepted = DeviceEventEdge::at(DeviceTimelinePoint::AcceptedSpecPublicationReady)
            .from(DeviceTimelineRole::AcceptedStatePublication);
        EXPECT_FALSE(accepted.to(DeviceTimelineRole::HostResultBridge).validForConsumption());
        ASSERT_TRUE(accepted.to(DeviceTimelineRole::DeviceGenerationController).validForConsumption());
        for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(2)})
        {
            MockBackend backend(device.type);
            void *const producer = reinterpret_cast<void *>(0x91000);
            void *const controller = reinterpret_cast<void *>(0x92000);
            void *const bridge = reinterpret_cast<void *>(0x93000);
            void *const publication = reinterpret_cast<void *>(0x94000);
            void *const ack = reinterpret_cast<void *>(0x95000);
            void *const terminal = reinterpret_cast<void *>(0x96000);
            ASSERT_TRUE(accepted.publish(backend, device, publication, producer));
            const PipelineDeviceGeneration::RankTransactionTerminal receipt{
                .stream = producer, .workspace = 7, .session = 9,
                .role = MoEOverlayInferenceGraphRole::MTPGroupedVerifier};
            ASSERT_TRUE(PipelineDeviceGeneration::recordOriginalRankProducerCompletion(
                backend, device, ack, receipt, 7, 9));
            EXPECT_EQ(backend.getEventWaitCount(), 0u) << "ACK observes its original producer only";
            ASSERT_TRUE(accepted.to(DeviceTimelineRole::DeviceGenerationController)
                .enqueuePublishedWait(backend, device, publication, controller));
            const auto completed = DeviceEventEdge::at(DeviceTimelinePoint::DeviceGenerationStateReady)
                .from(DeviceTimelineRole::DeviceGenerationController);
            ASSERT_TRUE(completed.publish(backend, device, terminal, controller));
            ASSERT_TRUE(completed.to(DeviceTimelineRole::HostResultBridge)
                .enqueuePublishedWait(backend, device, terminal, bridge));
            EXPECT_EQ(backend.getEventRecordCount(), 3u);
            EXPECT_EQ(backend.getEventWaitCount(), 2u);
            EXPECT_EQ(backend.getSyncCount(), 0u);
            EXPECT_EQ(backend.getStreamSyncCount(), 0u);
        }
    }

    TEST(Test__DeviceExecutionTimeline, ManifestCoversEveryTimelinePointExactlyOnce)
    {
        const auto &manifest = deviceExecutionTimelineManifest();
        ASSERT_EQ(
            manifest.size(),
            static_cast<size_t>(DeviceTimelinePoint::Count));

        std::array<bool, static_cast<size_t>(DeviceTimelinePoint::Count)>
            observed{};
        for (const auto &spec : manifest)
        {
            const size_t index = static_cast<size_t>(spec.point);
            ASSERT_LT(index, observed.size());
            EXPECT_FALSE(observed[index])
                << "duplicate timeline point " << spec.name;
            observed[index] = true;
            EXPECT_FALSE(spec.name.empty());
            EXPECT_NE(spec.producer, DeviceTimelineRole::Count);
            EXPECT_NE(spec.consumers, DeviceTimelineRoleMask{0});
            EXPECT_EQ(deviceTimelinePointName(spec.point), spec.name);
        }

        for (size_t index = 0; index < observed.size(); ++index)
        {
            EXPECT_TRUE(observed[index])
                << "missing timeline point at enum index " << index;
        }
    }

    TEST(Test__DeviceExecutionTimeline, CompactResponseDeclaresExactProductionFlow)
    {
        const auto publication =
            DeviceEventEdge::at(
                DeviceTimelinePoint::CompactSpeculativeResponseReady)
                .from(DeviceTimelineRole::VerifierSummary);
        EXPECT_TRUE(publication.validForPublication());

        const auto host_result =
            publication.to(DeviceTimelineRole::HostResultBridge);
        const auto accepted_state =
            publication.to(DeviceTimelineRole::AcceptedStatePublication);
        EXPECT_TRUE(host_result.validForConsumption());
        EXPECT_TRUE(accepted_state.validForConsumption());

        EXPECT_FALSE(
            publication.to(DeviceTimelineRole::RankCollective)
                .validForConsumption())
            << "A mirrored verifier outcome is already participant-local and "
               "must not acquire a second rank-owned publication phase.";

        EXPECT_FALSE(
            publication.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption())
            << "Compact host response bytes are not live forward state.";
        EXPECT_FALSE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::CompactSpeculativeResponseReady)
                .from(DeviceTimelineRole::MainForwardGraph)
                .validForPublication())
            << "The forward graph cannot impersonate the verifier summary.";
    }

    TEST(Test__DeviceExecutionTimeline, RequestInputBankDeclaresBidirectionalLifetime)
    {
        const auto admission =
            DeviceEventEdge::at(DeviceTimelinePoint::RequestInputAdmission)
                .from(DeviceTimelineRole::RequestAdmissionTransfer);
        EXPECT_TRUE(admission.validForPublication());
        EXPECT_TRUE(
            admission.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption());
        EXPECT_TRUE(
            admission.to(DeviceTimelineRole::MTPSidecarGraph)
                .validForConsumption());

        const auto reuse =
            DeviceEventEdge::at(DeviceTimelinePoint::RequestInputReuseReady)
                .from(DeviceTimelineRole::MainForwardGraph);
        EXPECT_TRUE(reuse.validForPublication());
        EXPECT_TRUE(
            reuse.to(DeviceTimelineRole::RequestAdmissionTransfer)
                .validForConsumption());
        EXPECT_TRUE(
            reuse.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption())
            << "A device-owned serial-decode position snapshot must wait for "
               "the previous graph before rewriting the stable position row.";
        EXPECT_FALSE(
            reuse.to(DeviceTimelineRole::MTPSidecarGraph)
                .validForConsumption())
            << "Sidecars release the bank through the main transaction stream; "
               "they never consume the bank-reuse publication.";
    }

    /**
     * @brief Restrict generation-controller ownership to the exact transaction DAG.
     *
     * Admission publishes the controller to verifier preparation. Accepted-state
     * publication returns the same exclusive token for either the next native
     * parent iteration, the terminal result bridge, or request reset. The
     * verifier summary is deliberately absent: allowing it to consume this edge
     * would recreate the former stochastic-only handoff after preparation had
     * already read the controller.
     */
    TEST(Test__DeviceExecutionTimeline,
         DeviceGenerationControllerDeclaresExactTransactionFlow)
    {
        const auto publication =
            DeviceEventEdge::at(
                DeviceTimelinePoint::DeviceGenerationStateReady)
                .from(DeviceTimelineRole::DeviceGenerationController);
        ASSERT_TRUE(publication.validForPublication());

        for (const DeviceTimelineRole role : {
                 DeviceTimelineRole::DeviceGenerationController,
                 DeviceTimelineRole::AllPositionVerifier,
                 DeviceTimelineRole::HostResultBridge,
                 DeviceTimelineRole::RequestStateReset,
             })
        {
            EXPECT_TRUE(publication.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
        }

        for (const DeviceTimelineRole role : {
                 DeviceTimelineRole::VerifierSummary,
                 DeviceTimelineRole::MainForwardGraph,
                 DeviceTimelineRole::RankCollective,
                 DeviceTimelineRole::Diagnostics,
             })
        {
            EXPECT_FALSE(publication.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
        }

        /*
         * Parent materialization transfers authority from the externally
         * scheduled first transaction to the captured generation family. Its
         * exact stream must therefore be able to join every predecessor that
         * may own live KV/GDN/MTP state before the first internal sidecar.
         */
        EXPECT_TRUE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::AcceptedSpecPublicationReady)
                .from(DeviceTimelineRole::AcceptedStatePublication)
                .to(DeviceTimelineRole::DeviceGenerationController)
                .validForConsumption());
        EXPECT_TRUE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::LivePrefixMutationReady)
                .from(DeviceTimelineRole::PrefixRestoreMutation)
                .to(DeviceTimelineRole::DeviceGenerationController)
                .validForConsumption());
        EXPECT_TRUE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::ForwardGraphOutputReady)
                .from(DeviceTimelineRole::MainForwardGraph)
                .to(DeviceTimelineRole::DeviceGenerationController)
                .validForConsumption());
    }

    /** @test Retained generation parents join the exact durable maintenance event without host synchronization. */
    TEST(Test__DeviceExecutionTimeline, RetainedPipelineParentConsumesMaintenancePublication)
    {
        const auto edge = DeviceEventEdge::at(DeviceTimelinePoint::MoERebalanceMaintenanceReady)
            .from(DeviceTimelineRole::MoERebalanceMaintenance)
            .to(DeviceTimelineRole::DeviceGenerationController);
        ASSERT_TRUE(edge.validForConsumption());
        for (const auto device : {DeviceId::cuda(1), DeviceId::rocm(2)})
        {
            MockBackend backend(device.type);
            void *const event = reinterpret_cast<void *>(0x82000);
            void *const stream = reinterpret_cast<void *>(0x83000);
            for (int replay = 0; replay < 20; ++replay)
            {
                backend.resetAll();
                ASSERT_TRUE(edge.enqueuePublishedWait(backend, device, event, stream));
                const auto records = backend.getEventRecords();
                ASSERT_EQ(records.size(), 1u);
                EXPECT_EQ(records.front().event, event);
                EXPECT_EQ(records.front().stream, stream);
                EXPECT_EQ(records.front().device_id, device.ordinal);
                EXPECT_EQ(backend.getEventWaitCount(), 1u);
                EXPECT_EQ(backend.getH2DCount(), 0u);
                EXPECT_EQ(backend.getD2HCount(), 0u);
                EXPECT_EQ(backend.getSyncCount(), 0u);
                EXPECT_EQ(backend.getStreamSyncCount(), 0u);
            }
        }
    }

    /** @test Headless pipeline graphs order their own state without a sampler. */
    TEST(Test__DeviceExecutionTimeline, PipelineForwardConsumesItsLocalPredecessor)
    {
        EXPECT_TRUE(DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
            .from(DeviceTimelineRole::MainForwardGraph)
            .to(DeviceTimelineRole::MainForwardGraph)
            .validForConsumption());
    }

    TEST(Test__DeviceExecutionTimeline, RequestResetMustPrecedeEveryGpuGraphFamily)
    {
        const auto reset =
            DeviceEventEdge::at(DeviceTimelinePoint::RequestStateResetReady)
                .from(DeviceTimelineRole::RequestStateReset);
        EXPECT_TRUE(reset.validForPublication());
        EXPECT_TRUE(
            reset.to(DeviceTimelineRole::RequestStateReset)
                .validForConsumption())
            << "An adjacent request boundary must consume the previous reset "
               "generation before republishing the lifecycle-owned event.";
        EXPECT_TRUE(
            reset.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption());
        EXPECT_TRUE(
            reset.to(DeviceTimelineRole::MTPSidecarGraph)
                .validForConsumption());
        EXPECT_TRUE(
            reset.to(DeviceTimelineRole::PrefixRestoreMutation)
                .validForConsumption());
        EXPECT_TRUE(
            reset.to(DeviceTimelineRole::RequestAdmissionTransfer)
                .validForConsumption())
            << "Request admission publishes request-constant GPU controls as well "
               "as token rows, so it must inherit reset before publishing the "
               "single transitive admission edge consumed by graph execution.";

        const auto prior_forward =
            DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
                .from(DeviceTimelineRole::MainForwardGraph)
                .to(DeviceTimelineRole::RequestStateReset);
        EXPECT_TRUE(prior_forward.validForConsumption())
            << "Reset zeroing must first join the previous request's final "
               "forward publication.";
    }

    TEST(Test__DeviceExecutionTimeline, GraphBuildPublicationPrecedesEveryGraphFamily)
    {
        const auto publication =
            DeviceEventEdge::at(
                DeviceTimelinePoint::GraphBuildDeviceStateReady)
                .from(
                    DeviceTimelineRole::
                        GraphBuildDeviceStatePublication);

        EXPECT_TRUE(publication.validForPublication());
        EXPECT_TRUE(
            publication.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption());
        EXPECT_TRUE(
            publication.to(DeviceTimelineRole::MTPSidecarGraph)
                .validForConsumption());
        EXPECT_TRUE(
            publication.to(DeviceTimelineRole::RequestStateReset)
                .validForConsumption())
            << "A request that ends before first execution must retire cold "
               "graph-build publication before resetting model state.";
        EXPECT_FALSE(
            publication.to(DeviceTimelineRole::RequestAdmissionTransfer)
                .validForConsumption())
            << "Graph-build descriptor publication is consumed by graph "
               "execution, not by independent request admission.";
    }

    TEST(Test__DeviceExecutionTimeline, DurableForwardPublicationNeedsNoProducerStream)
    {
        MockBackend backend(DeviceType::CUDA);
        void *event = reinterpret_cast<void *>(0xE001);
        void *consumer_stream = reinterpret_cast<void *>(0xC001);

        const auto archive_edge =
            DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
                .from(DeviceTimelineRole::MainForwardGraph)
                .to(DeviceTimelineRole::PrefixCheckpointArchive);
        ASSERT_TRUE(archive_edge.validForConsumption());
        EXPECT_TRUE(archive_edge.enqueuePublishedWait(
            backend,
            DeviceId::cuda(0),
            event,
            consumer_stream));
        EXPECT_EQ(backend.getEventWaitCount(), 1u);

        const auto records = backend.getEventRecordsForStream(consumer_stream);
        ASSERT_EQ(records.size(), 1u);
        EXPECT_EQ(records.front().type, MockBackend::EventRecord::WAIT);
        EXPECT_EQ(records.front().event, event);

        EXPECT_FALSE(archive_edge.enqueuePublishedWait(
            backend,
            DeviceId::cuda(0),
            event,
            nullptr))
            << "A durable event never licenses an implicit/default consumer stream.";
        EXPECT_EQ(backend.getEventWaitCount(), 1u);
    }

    TEST(
        Test__DeviceExecutionTimeline,
        PublishedLiveStateEdgesDeclareEveryPermittedOwner)
    {
        const std::array<DeviceTimelineRole, 9> common_permitted = {
            DeviceTimelineRole::DeviceGenerationController,
            DeviceTimelineRole::MainForwardGraph,
            DeviceTimelineRole::MTPSidecarGraph,
            DeviceTimelineRole::PrefixCheckpointArchive,
            DeviceTimelineRole::PrefixRestoreMutation,
            DeviceTimelineRole::MoERebalanceMaintenance,
            DeviceTimelineRole::TargetSampler,
            DeviceTimelineRole::RequestStateReset,
            DeviceTimelineRole::Diagnostics,
        };

        const auto accepted =
            DeviceEventEdge::at(
                DeviceTimelinePoint::AcceptedSpecPublicationReady)
                .from(DeviceTimelineRole::AcceptedStatePublication);
        const auto prefix_mutation =
            DeviceEventEdge::at(
                DeviceTimelinePoint::LivePrefixMutationReady)
                .from(DeviceTimelineRole::PrefixRestoreMutation);
        ASSERT_TRUE(accepted.validForPublication());
        ASSERT_TRUE(prefix_mutation.validForPublication());

        for (const DeviceTimelineRole role : common_permitted)
        {
            EXPECT_TRUE(accepted.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
            EXPECT_TRUE(prefix_mutation.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
        }

        EXPECT_FALSE(
            accepted.to(DeviceTimelineRole::HostResultBridge)
                .validForConsumption())
            << "Accepted speculative state reaches the host through the typed "
               "compact-response publication, not through its mutation event.";
        EXPECT_TRUE(
            prefix_mutation.to(DeviceTimelineRole::HostResultBridge)
                .validForConsumption())
            << "A host parity/result bridge must be able to observe terminal "
               "logits restored without a forward graph.";

        for (const DeviceTimelineRole role : {
                 DeviceTimelineRole::AllPositionVerifier,
                 DeviceTimelineRole::AcceptedStatePublication,
                 DeviceTimelineRole::HostArchiveBoundary,
             })
        {
            EXPECT_FALSE(accepted.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
            EXPECT_FALSE(prefix_mutation.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
        }

        EXPECT_FALSE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::AcceptedSpecPublicationReady)
                .from(DeviceTimelineRole::PrefixRestoreMutation)
                .validForPublication())
            << "A prefix restore cannot impersonate accepted-state publication.";
        EXPECT_FALSE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::LivePrefixMutationReady)
                .from(DeviceTimelineRole::AcceptedStatePublication)
                .validForPublication())
            << "Accepted-state publication cannot impersonate a prefix mutation.";
    }

    TEST(Test__DeviceExecutionTimeline, MainLogitsSamplerConsumesDurableForwardPublication)
    {
        MockBackend backend(DeviceType::CUDA);
        void *event = reinterpret_cast<void *>(0xE101);
        void *sampler_stream = reinterpret_cast<void *>(0xC101);

        const auto sampler_edge =
            DeviceEventEdge::at(DeviceTimelinePoint::ForwardGraphOutputReady)
                .from(DeviceTimelineRole::MainForwardGraph)
                .to(DeviceTimelineRole::TargetSampler);
        ASSERT_TRUE(sampler_edge.validForConsumption())
            << "Every main-logits sampler must be ordered after the exact "
               "forward invocation, including preserved graph replay.";
        EXPECT_TRUE(sampler_edge.enqueuePublishedWait(
            backend,
            DeviceId::cuda(0),
            event,
            sampler_stream));
        EXPECT_EQ(backend.getEventWaitCount(), 1u);

        const auto records = backend.getEventRecordsForStream(sampler_stream);
        ASSERT_EQ(records.size(), 1u);
        EXPECT_EQ(records.front().type, MockBackend::EventRecord::WAIT);
        EXPECT_EQ(records.front().event, event);
    }

    TEST(Test__DeviceExecutionTimeline, HostRolesNeverProduceExecutionState)
    {
        const auto &manifest = deviceExecutionTimelineManifest();
        for (const auto &spec : manifest)
        {
            EXPECT_NE(
                spec.producer,
                DeviceTimelineRole::HostResultBridge)
                << spec.name;
            EXPECT_NE(
                spec.producer,
                DeviceTimelineRole::HostArchiveBoundary)
                << spec.name;
        }
    }

} // namespace llaminar2::test
