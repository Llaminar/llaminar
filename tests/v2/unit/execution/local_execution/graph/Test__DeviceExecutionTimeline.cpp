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
