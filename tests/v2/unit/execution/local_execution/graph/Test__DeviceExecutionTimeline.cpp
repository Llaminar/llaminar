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
#include "mocks/MockBackend.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string_view>

namespace llaminar2::test
{
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
        const auto rank_collective =
            publication.to(DeviceTimelineRole::RankCollective);
        EXPECT_TRUE(host_result.validForConsumption());
        EXPECT_TRUE(rank_collective.validForConsumption());

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

    TEST(Test__DeviceExecutionTimeline, RankCompactResponseOwnsPostCollectiveFlow)
    {
        const auto publication =
            DeviceEventEdge::at(
                DeviceTimelinePoint::RankCompactSpeculativeResponseReady)
                .from(DeviceTimelineRole::RankCollective);
        EXPECT_TRUE(publication.validForPublication());
        EXPECT_TRUE(
            publication.to(DeviceTimelineRole::HostResultBridge)
                .validForConsumption());
        EXPECT_TRUE(
            publication.to(DeviceTimelineRole::AcceptedStatePublication)
                .validForConsumption());
        EXPECT_FALSE(
            publication.to(DeviceTimelineRole::MainForwardGraph)
                .validForConsumption());
        EXPECT_FALSE(
            DeviceEventEdge::at(
                DeviceTimelinePoint::RankCompactSpeculativeResponseReady)
                .from(DeviceTimelineRole::VerifierSummary)
                .validForPublication())
            << "A child-local verifier summary cannot impersonate completed "
               "rank publication.";
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
        EXPECT_FALSE(
            reuse.to(DeviceTimelineRole::MTPSidecarGraph)
                .validForConsumption())
            << "Sidecars release the bank through the main transaction stream; "
               "they never consume the bank-reuse publication.";
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
        const std::array<DeviceTimelineRole, 7> permitted = {
            DeviceTimelineRole::MainForwardGraph,
            DeviceTimelineRole::MTPSidecarGraph,
            DeviceTimelineRole::PrefixCheckpointArchive,
            DeviceTimelineRole::PrefixRestoreMutation,
            DeviceTimelineRole::MoERebalanceMaintenance,
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

        for (const DeviceTimelineRole role : permitted)
        {
            EXPECT_TRUE(accepted.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
            EXPECT_TRUE(prefix_mutation.to(role).validForConsumption())
                << deviceTimelineRoleName(role);
        }

        for (const DeviceTimelineRole role : {
                 DeviceTimelineRole::AllPositionVerifier,
                 DeviceTimelineRole::AcceptedStatePublication,
                 DeviceTimelineRole::HostResultBridge,
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
