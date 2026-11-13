/**
 * @file Test__MoEOverlayActivationEpochProtocol.cpp
 * @brief Adversarial CPU tests for device-owned ExpertOverlay activation epochs.
 *
 * Replay uses the actual shared publication helpers and original host protocol.
 * Recorded delayed dispatch, reset and absorbing cancellation observations prove
 * ordering without a GPU, host execution mirror or invented native receipt.
 */

#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlayActivationAdmissionPublication.h"
#include "execution/moe/MoEOverlayRouteTarget.h"
#include "execution/moe/MoEOverlayInferenceOriginalSlot.h"
#include "execution/moe/MoEOverlayNativeInvocationABI.h"
#include <array>

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        MoEOverlayActivationEpochConfig makeConfig(
            std::vector<std::int32_t> layers = {2, 7, 19, 33})
        {
            return {
                .channel_nonce = 0x91a77c02u,
                .topology_fingerprint_low = 0x1234567812345678ull,
                .topology_fingerprint_high = 0x8765432187654321ull,
                .workspace_generation = 17u,
                .source = {
                    .world_rank = 0,
                    .participant_id = 12,
                    .tier_priority = 80,
                    .domain_ordinal = 4,
                },
                .target = {
                    .world_rank = 1,
                    .participant_id = 27,
                    .tier_priority = 30,
                    .domain_ordinal = 9,
                },
                .lane_ordinal = 3u,
                .graph_role_mask =
                    std::uint32_t{1}
                    << static_cast<std::uint32_t>(
                           MoEOverlayInferenceGraphRole::MainDecode),
                .model_layer_indices = std::move(layers),
            };
        }

        MoEOverlayInferenceTransactionTicket makeTicket(
            const MoEOverlayActivationEpochConfig &config,
            std::uint64_t transaction_ordinal = 1u,
            std::uint64_t request_generation = 1u)
        {
            const MoEOverlayInferenceTopologyIdentity topology{
                .workspace_generation = config.workspace_generation,
                .topology_fingerprint_low = config.topology_fingerprint_low,
                .topology_fingerprint_high = config.topology_fingerprint_high,
                .source_world_rank = config.source.world_rank,
                .target_world_rank = config.target.world_rank,
            };
            const MoEOverlayInferenceCommandIdentity command{
                .request_generation = request_generation,
                .command_id = transaction_ordinal,
                .initial_placement_epoch = 41u,
            };
            return makeMoEOverlayInferenceExecutionTicket(
                topology,
                command,
                transaction_ordinal,
                /*logical_step_id=*/transaction_ordinal * 3u,
                /*placement_epoch=*/41u,
                MoEOverlayInferenceGraphRole::MainDecode,
                /*request_count=*/1,
                /*logical_rows_per_request=*/1,
                /*physical_rows_per_request=*/1);
        }

        void activateBoth(
            MoEOverlayActivationEpochProtocol &protocol,
            const MoEOverlayActivationEpochIdentity &identity)
        {
            std::string error;
            ASSERT_TRUE(protocol.activate(
                MoEOverlayActivationEndpoint::Continuation,
                identity,
                &error))
                << error;
            ASSERT_TRUE(protocol.activate(
                MoEOverlayActivationEndpoint::Follower,
                identity,
                &error))
                << error;
        }

        void runOneStage(
            MoEOverlayActivationEpochProtocol &protocol,
            const MoEOverlayActivationEpochIdentity &identity,
            std::uint32_t stage)
        {
            std::string error;
            ASSERT_TRUE(protocol.publishDispatch(
                identity,
                stage,
                /*live_rows=*/3u,
                /*live_entries=*/6u,
                /*payload_bytes=*/768u,
                &error))
                << error;
            const auto dispatch = protocol.consumeDispatch(
                identity, stage, &error);
            ASSERT_TRUE(dispatch.has_value()) << error;
            EXPECT_EQ(dispatch->stage_ordinal, stage);
            ASSERT_TRUE(protocol.publishReturn(
                identity,
                stage,
                /*payload_bytes=*/384u,
                &error))
                << error;
            const auto returned = protocol.consumeReturn(
                identity, stage, &error);
            ASSERT_TRUE(returned.has_value()) << error;
            EXPECT_EQ(returned->live_rows, dispatch->live_rows);
            EXPECT_EQ(returned->live_entries, dispatch->live_entries);
        }

        void completeBoth(
            MoEOverlayActivationEpochProtocol &protocol,
            const MoEOverlayActivationEpochIdentity &identity)
        {
            std::string error;
            ASSERT_TRUE(protocol.complete(
                MoEOverlayActivationEndpoint::Follower,
                identity,
                &error))
                << error;
            ASSERT_TRUE(protocol.complete(
                MoEOverlayActivationEndpoint::Continuation,
                identity,
                &error))
                << error;
        }
    } // namespace

    TEST(Test__MoEOverlayActivationEpochProtocol,
         FixedABIAndTwoBanksPreservePipelinedRoundTripsAcrossReset)
    {
        EXPECT_EQ(kMoEOverlayActivationABIVersion, 10u);
        EXPECT_EQ(sizeof(MoEOverlayActivationEndpointStatus), 128u);
        EXPECT_EQ(sizeof(MoEOverlayActivationEpochControl), 1152u);
        EXPECT_EQ(alignof(MoEOverlayActivationEpochControl), 64u);

        const auto config = makeConfig();
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);

        EXPECT_EQ(protocol.channelHeader().source_tier_priority, 80);
        EXPECT_EQ(protocol.channelHeader().target_tier_priority, 30);
        EXPECT_EQ(protocol.channelHeader().source_domain_ordinal, 4);
        EXPECT_EQ(protocol.channelHeader().target_domain_ordinal, 9);
        EXPECT_EQ(protocol.channelHeader().source_participant_id, 12);
        EXPECT_EQ(protocol.channelHeader().target_participant_id, 27);
        EXPECT_EQ(protocol.channelHeader().lane_ordinal, 3u);
        EXPECT_EQ(protocol.channelHeader().stage_count, 4u);
        EXPECT_EQ(protocol.admissionState(),
                  MoEOverlayActivationAdmissionState::Idle);
        EXPECT_EQ(control.admission.ready_signal, 0u);

        std::string error;
        const auto armed = protocol.arm(
            makeTicket(config),
            /*epoch_generation=*/1u,
            /*timeout_not_before_ns=*/10'000u,
            &error);
        ASSERT_TRUE(armed.has_value()) << error;
        EXPECT_EQ(armed->digest, moeOverlayActivationIdentityDigest(*armed));
        EXPECT_EQ(
            control.admission.ready_signal,
            kMoEOverlayActivationAdmissionTimeline);
        activateBoth(protocol, *armed);

        /*
         * Stage zero and one occupy different banks and may be dispatched
         * together. Stage two cannot overwrite bank zero until its exact return
         * descriptor has been consumed by the continuation.
         */
        ASSERT_TRUE(protocol.publishDispatch(
            *armed, 0u, 3u, 6u, 768u, &error))
            << error;
        ASSERT_TRUE(protocol.publishDispatch(
            *armed, 1u, 4u, 8u, 1024u, &error))
            << error;
        EXPECT_FALSE(protocol.publishDispatch(
            *armed, 2u, 2u, 4u, 512u, &error));
        EXPECT_EQ(
            protocol.endpointStatus(
                MoEOverlayActivationEndpoint::Continuation).typedCode(),
            MoEOverlayActivationStatusCode::BufferBusy);
        EXPECT_EQ(
            protocol.endpointState(MoEOverlayActivationEndpoint::Continuation),
            MoEOverlayActivationEndpointState::Active)
            << "Backpressure is retryable and must not corrupt the transaction";

        ASSERT_TRUE(protocol.consumeDispatch(*armed, 0u, &error).has_value())
            << error;
        ASSERT_TRUE(protocol.publishReturn(
            *armed, 0u, 384u, &error))
            << error;
        ASSERT_TRUE(protocol.consumeReturn(*armed, 0u, &error).has_value())
            << error;
        ASSERT_TRUE(protocol.publishDispatch(
            *armed, 2u, 2u, 4u, 512u, &error))
            << error;

        ASSERT_TRUE(protocol.consumeDispatch(*armed, 1u, &error).has_value())
            << error;
        ASSERT_TRUE(protocol.publishReturn(
            *armed, 1u, 512u, &error))
            << error;
        ASSERT_TRUE(protocol.consumeReturn(*armed, 1u, &error).has_value())
            << error;
        ASSERT_TRUE(protocol.consumeDispatch(*armed, 2u, &error).has_value())
            << error;
        ASSERT_TRUE(protocol.publishReturn(
            *armed, 2u, 256u, &error))
            << error;
        ASSERT_TRUE(protocol.consumeReturn(*armed, 2u, &error).has_value())
            << error;
        runOneStage(protocol, *armed, 3u);
        EXPECT_FALSE(protocol.completedTraffic(*armed, &error).has_value());
        EXPECT_NE(error.find("two complete endpoint-owned"), std::string::npos);
        completeBoth(protocol, *armed);

        const auto traffic = protocol.completedTraffic(*armed, &error);
        ASSERT_TRUE(traffic.has_value()) << error;
        EXPECT_EQ(traffic->dispatch_payload_bytes, 3072u);
        EXPECT_EQ(traffic->return_payload_bytes, 1536u);
        EXPECT_EQ(traffic->dispatch_live_rows, 12u);
        EXPECT_EQ(traffic->return_live_rows, 12u);
        EXPECT_EQ(traffic->dispatch_live_entries, 24u);
        EXPECT_EQ(traffic->return_live_entries, 24u);
        EXPECT_EQ(traffic->dispatch_stage_count, 4u);
        EXPECT_EQ(traffic->return_stage_count, 4u);

        ASSERT_TRUE(protocol.reset(*armed, &error)) << error;
        EXPECT_EQ(protocol.admissionState(),
                  MoEOverlayActivationAdmissionState::Idle);
        EXPECT_EQ(control.admission.ready_signal, 0u);
        for (std::uint32_t bank = 0u;
             bank < kMoEOverlayActivationBufferCount;
             ++bank)
        {
            EXPECT_EQ(protocol.dispatchTimeline(bank), 0u);
            EXPECT_EQ(protocol.returnTimeline(bank), 0u);
        }
        EXPECT_EQ(
            protocol.endpointStatus(
                MoEOverlayActivationEndpoint::Continuation)
                .published_payload_bytes,
            0u);
        EXPECT_EQ(
            protocol.endpointStatus(
                MoEOverlayActivationEndpoint::Follower)
                .published_stage_count,
            0u);
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         EmptySparseLayerAdvancesBothDeviceOwnedTimelines)
    {
        const auto config = makeConfig({6});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(
            makeTicket(config), 1u, 10'000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        activateBoth(protocol, *identity);

        ASSERT_TRUE(protocol.publishDispatch(
            *identity,
            /*stage_ordinal=*/0u,
            /*live_rows=*/0u,
            /*live_entries=*/0u,
            /*payload_bytes=*/0u,
            &error))
            << error;
        const auto dispatch = protocol.consumeDispatch(
            *identity, 0u, &error);
        ASSERT_TRUE(dispatch.has_value()) << error;
        EXPECT_EQ(dispatch->live_rows, 0u);
        EXPECT_EQ(dispatch->live_entries, 0u);
        EXPECT_EQ(dispatch->payload_bytes, 0u);

        ASSERT_TRUE(protocol.publishReturn(
            *identity, 0u, /*payload_bytes=*/0u, &error))
            << error;
        const auto returned = protocol.consumeReturn(
            *identity, 0u, &error);
        ASSERT_TRUE(returned.has_value()) << error;
        EXPECT_EQ(returned->live_rows, 0u);
        EXPECT_EQ(returned->payload_bytes, 0u);
        completeBoth(protocol, *identity);
    }

    /**
     * @brief Preserve the device failure witness in an acquired host snapshot.
     *
     * A watchdog diagnoses retained-graph admission without a diagnostic D2H
     * copy.  The endpoint accessor must therefore carry both compact witness
     * words after acquiring the endpoint-owned publication state.
     */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         EndpointSnapshotPreservesDeviceFailureWitness)
    {
        const auto config = makeConfig();
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);

        control.follower_status.failure_diagnostic = 0x0015a55au;
        control.follower_status.failure_auxiliary = 0x003f0004u;
        control.follower_status.code = static_cast<std::uint32_t>(
            MoEOverlayActivationStatusCode::InvalidControl);
        control.follower_status.state = static_cast<std::uint32_t>(
            MoEOverlayActivationEndpointState::Aborted);

        const auto snapshot = protocol.endpointStatus(
            MoEOverlayActivationEndpoint::Follower);
        EXPECT_EQ(snapshot.typedState(),
                  MoEOverlayActivationEndpointState::Aborted);
        EXPECT_EQ(snapshot.typedCode(),
                  MoEOverlayActivationStatusCode::InvalidControl);
        EXPECT_EQ(snapshot.failure_diagnostic, 0x0015a55au);
        EXPECT_EQ(snapshot.failure_auxiliary, 0x003f0004u);
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         DevicePublishedPlacementEpochMayAdvancePastImmutableSchedulerFloor)
    {
        const auto config = makeConfig({6});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(
            makeTicket(config), 1u, 10'000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        ASSERT_EQ(identity->placement_epoch_floor, 41u);
        activateBoth(protocol, *identity);

        ASSERT_TRUE(protocol.publishDispatch(
            *identity, 0u, 2u, 4u, 512u, &error))
            << error;
        /* A device-resident controller published epoch 42 after the host sent
         * its immutable floor. The real packet kernel writes this exact value
         * before its release timeline; this sequential oracle isolates the
         * corresponding protocol validation rule. */
        control.buffers[0].dispatch_descriptor.placement_epoch = 42u;
        const auto dispatch = protocol.consumeDispatch(
            *identity, 0u, &error);
        ASSERT_TRUE(dispatch.has_value()) << error;
        EXPECT_EQ(dispatch->placement_epoch, 42u);

        ASSERT_TRUE(protocol.publishReturn(
            *identity, 0u, 256u, &error))
            << error;
        EXPECT_EQ(
            control.buffers[0].return_descriptor.placement_epoch,
            42u);
        const auto returned = protocol.consumeReturn(
            *identity, 0u, &error);
        ASSERT_TRUE(returned.has_value()) << error;
        EXPECT_EQ(returned->placement_epoch, 42u);
        completeBoth(protocol, *identity);
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         TwoByFourPlusCpuTierIsExpressedOnlyAsPlannerSelectedParticipantLanes)
    {
        const std::vector<MoEOverlayActivationLaneEndpoint> continuation{
            {.world_rank = 0,
             .participant_id = 100,
             .tier_priority = 90,
             .domain_ordinal = 8},
            {.world_rank = 0,
             .participant_id = 101,
             .tier_priority = 90,
             .domain_ordinal = 8},
        };
        const std::vector<MoEOverlayActivationLaneEndpoint> secondary{
            {.world_rank = 1,
             .participant_id = 200,
             .tier_priority = 40,
             .domain_ordinal = 5},
            {.world_rank = 1,
             .participant_id = 201,
             .tier_priority = 40,
             .domain_ordinal = 5},
            {.world_rank = 1,
             .participant_id = 202,
             .tier_priority = 40,
             .domain_ordinal = 5},
            {.world_rank = 1,
             .participant_id = 203,
             .tier_priority = 40,
             .domain_ordinal = 5},
        };
        const std::vector<MoEOverlayActivationLaneEndpoint> tertiary_cpu{
            {.world_rank = 2,
             .participant_id = 300,
             .tier_priority = 10,
             .domain_ordinal = 2},
            {.world_rank = 3,
             .participant_id = 301,
             .tier_priority = 10,
             .domain_ordinal = 2},
        };

        std::uint32_t lane_ordinal = 0u;
        for (const auto &source : continuation)
        {
            for (const auto &target_group : {secondary, tertiary_cpu})
            {
                for (const auto &target : target_group)
                {
                    auto config = makeConfig({0, 1});
                    config.channel_nonce += lane_ordinal;
                    config.source = source;
                    config.target = target;
                    config.lane_ordinal = lane_ordinal++;

                    MoEOverlayActivationEpochControl control;
                    ASSERT_NO_THROW(
                        MoEOverlayActivationEpochProtocol::initialize(
                            control, config));
                    EXPECT_EQ(control.channel.source_participant_id,
                              source.participant_id);
                    EXPECT_EQ(control.channel.target_participant_id,
                              target.participant_id);
                    EXPECT_EQ(control.channel.source_tier_priority,
                              source.tier_priority);
                    EXPECT_EQ(control.channel.target_tier_priority,
                              target.tier_priority);
                    EXPECT_EQ(control.channel.lane_ordinal,
                              config.lane_ordinal);
                }
            }
        }
        EXPECT_EQ(lane_ordinal, 12u);
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         ArmRejectsStaleGenerationAndActivationRejectsAliasingIdentity)
    {
        const auto config = makeConfig({3, 5});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto first = protocol.arm(
            makeTicket(config), 9u, 1000u, &error);
        ASSERT_TRUE(first.has_value()) << error;
        activateBoth(protocol, *first);
        runOneStage(protocol, *first, 0u);
        runOneStage(protocol, *first, 1u);
        completeBoth(protocol, *first);
        ASSERT_TRUE(protocol.reset(*first, &error)) << error;

        EXPECT_FALSE(protocol.arm(
            makeTicket(config, 2u), 9u, 2000u, &error).has_value());
        const auto second = protocol.arm(
            makeTicket(config, 2u), 10u, 2000u, &error);
        ASSERT_TRUE(second.has_value()) << error;

        auto alias = *second;
        ++alias.logical_step_id;
        EXPECT_FALSE(protocol.activate(
            MoEOverlayActivationEndpoint::Follower,
            alias,
            &error));
        EXPECT_EQ(
            protocol.endpointState(MoEOverlayActivationEndpoint::Follower),
            MoEOverlayActivationEndpointState::Aborted);
        EXPECT_EQ(protocol.returnTimeline(0u),
                  kMoEOverlayActivationAbortTimeline);
        EXPECT_FALSE(protocol.reset(*second, &error))
            << "An authenticated protocol failure is process-terminal";
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         ArmAuthenticatesWorkspaceGenerationAndRetainedGraphRole)
    {
        const auto config = makeConfig({3, 5});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;

        auto wrong_workspace = makeTicket(config);
        ++wrong_workspace.workspace_generation;
        EXPECT_TRUE(wrong_workspace.valid());
        EXPECT_FALSE(protocol.arm(
            wrong_workspace, 1u, 1000u, &error).has_value());
        EXPECT_NE(error.find("authenticate"), std::string::npos);

        auto wrong_family = makeTicket(config);
        wrong_family.graph_role =
            MoEOverlayInferenceGraphRole::MainPrefill;
        EXPECT_TRUE(wrong_family.valid());
        EXPECT_FALSE(protocol.arm(
            wrong_family, 1u, 1000u, &error).has_value());
        EXPECT_NE(error.find("authenticate"), std::string::npos);

        EXPECT_TRUE(protocol.arm(
            makeTicket(config), 1u, 1000u, &error).has_value())
            << error;
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         CorruptedDescriptorAbortsFollowerAndSentinelReleasesContinuation)
    {
        const auto config = makeConfig({11});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(
            makeTicket(config), 1u, 1000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        activateBoth(protocol, *identity);
        ASSERT_TRUE(protocol.publishDispatch(
            *identity, 0u, 2u, 4u, 512u, &error))
            << error;

        /* Simulate bytes changed after the producer's authenticated descriptor. */
        control.buffers[0].dispatch_descriptor.live_rows = 99u;
        EXPECT_FALSE(protocol.consumeDispatch(*identity, 0u, &error).has_value());
        EXPECT_EQ(
            protocol.endpointStatus(
                MoEOverlayActivationEndpoint::Follower).typedCode(),
            MoEOverlayActivationStatusCode::PayloadMismatch);
        EXPECT_EQ(protocol.returnTimeline(0u),
                  kMoEOverlayActivationAbortTimeline);

        EXPECT_FALSE(protocol.consumeReturn(*identity, 0u, &error).has_value());
        EXPECT_EQ(
            protocol.endpointStatus(
                MoEOverlayActivationEndpoint::Continuation).typedCode(),
            MoEOverlayActivationStatusCode::PeerAborted);
        EXPECT_EQ(protocol.dispatchTimeline(1u),
                  kMoEOverlayActivationAbortTimeline);
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         WatchdogIsGenerationExactAndTimeoutCannotBeReset)
    {
        const auto config = makeConfig({1});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(
            makeTicket(config), 44u, 5000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;

        EXPECT_FALSE(protocol.markTimedOut(43u, 6000u, &error));
        EXPECT_FALSE(protocol.markTimedOut(44u, 4999u, &error));
        EXPECT_EQ(protocol.admissionState(),
                  MoEOverlayActivationAdmissionState::Armed);
        EXPECT_TRUE(protocol.markTimedOut(44u, 5000u, &error)) << error;
        EXPECT_EQ(protocol.admissionState(),
                  MoEOverlayActivationAdmissionState::Failed);
        EXPECT_EQ(
            static_cast<MoEOverlayActivationStatusCode>(
                control.admission.code),
            MoEOverlayActivationStatusCode::TimedOut);
        EXPECT_FALSE(protocol.reset(*identity, &error));
        EXPECT_FALSE(protocol.activate(
            MoEOverlayActivationEndpoint::Continuation,
            *identity,
            &error));
    }

    TEST(Test__MoEOverlayActivationEpochProtocol,
         TwentyConcurrentGenerationsNeverAliasOrOverwriteEitherBank)
    {
        std::vector<std::int32_t> layers;
        for (std::int32_t layer = 0; layer < 48; ++layer)
            layers.push_back(layer);
        const auto config = makeConfig(std::move(layers));
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);

        for (std::uint64_t generation = 1u; generation <= 20u; ++generation)
        {
            std::string arm_error;
            const auto identity = protocol.arm(
                makeTicket(config, generation, generation),
                generation,
                generation * 1000u,
                &arm_error);
            ASSERT_TRUE(identity.has_value()) << arm_error;
            activateBoth(protocol, *identity);

            std::atomic<std::uint64_t> failures{0u};
            std::thread continuation(
                [&]
                {
                    std::string error;
                    for (std::uint32_t stage = 0u;
                         stage < identity->stage_count;
                         ++stage)
                    {
                        if (stage >= kMoEOverlayActivationBufferCount)
                        {
                            const std::uint32_t returned_stage =
                                stage - kMoEOverlayActivationBufferCount;
                            const std::uint32_t bank =
                                moeOverlayActivationBufferIndex(returned_stage);
                            const std::uint64_t expected =
                                moeOverlayActivationLeasedTimelineValue(
                                    moeOverlayActivationBufferVisit(
                                        returned_stage));
                            while (protocol.returnTimeline(bank) < expected)
                                std::this_thread::yield();
                            if (!protocol.consumeReturn(
                                    *identity, returned_stage, &error))
                            {
                                failures.fetch_add(1u);
                                return;
                            }
                        }
                        if (!protocol.publishDispatch(
                                *identity,
                                stage,
                                /*live_rows=*/2u,
                                /*live_entries=*/4u,
                                /*payload_bytes=*/512u,
                                &error))
                        {
                            failures.fetch_add(1u);
                            return;
                        }
                    }

                    for (std::uint32_t stage =
                             identity->stage_count -
                             kMoEOverlayActivationBufferCount;
                         stage < identity->stage_count;
                         ++stage)
                    {
                        const std::uint32_t bank =
                            moeOverlayActivationBufferIndex(stage);
                        const std::uint64_t expected =
                            moeOverlayActivationLeasedTimelineValue(
                                moeOverlayActivationBufferVisit(stage));
                        while (protocol.returnTimeline(bank) < expected)
                            std::this_thread::yield();
                        if (!protocol.consumeReturn(*identity, stage, &error))
                        {
                            failures.fetch_add(1u);
                            return;
                        }
                    }
                    if (!protocol.complete(
                            MoEOverlayActivationEndpoint::Continuation,
                            *identity,
                            &error))
                    {
                        failures.fetch_add(1u);
                    }
                });

            std::thread follower(
                [&]
                {
                    std::string error;
                    for (std::uint32_t stage = 0u;
                         stage < identity->stage_count;
                         ++stage)
                    {
                        const std::uint32_t bank =
                            moeOverlayActivationBufferIndex(stage);
                        const std::uint64_t expected =
                            moeOverlayActivationLeasedTimelineValue(
                                moeOverlayActivationBufferVisit(stage));
                        while (protocol.dispatchTimeline(bank) < expected)
                            std::this_thread::yield();
                        if (!protocol.consumeDispatch(*identity, stage, &error) ||
                            !protocol.publishReturn(
                                *identity, stage, 256u, &error))
                        {
                            failures.fetch_add(1u);
                            return;
                        }
                    }
                    if (!protocol.complete(
                            MoEOverlayActivationEndpoint::Follower,
                            *identity,
                            &error))
                    {
                        failures.fetch_add(1u);
                    }
                });

            continuation.join();
            follower.join();
            ASSERT_EQ(failures.load(), 0u) << "generation=" << generation;
            std::string reset_error;
            ASSERT_TRUE(protocol.reset(*identity, &reset_error))
                << reset_error << ", generation=" << generation;
        }
    }
    /** Scheduler failure never becomes a second writer of either device endpoint. */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         SchedulerCancellationPreservesOriginalEndpointErrorsAndRejectsForeignIdentity)
    {
        for (const bool same_rank : {false, true})
        {
            auto config = makeConfig({1, 2});
            if (same_rank) config.target.world_rank = config.source.world_rank;
            MoEOverlayActivationEpochControl control;
            MoEOverlayActivationEpochProtocol::initialize(control, config);
            MoEOverlayActivationEpochProtocol protocol(control, config);
            std::string error;
            ASSERT_FALSE(protocol.cancelOriginalInvocation({}, MoEOverlayActivationStatusCode::ExplicitAbort, &error));
            const auto identity = protocol.arm(makeTicket(config), 44u, 5000u, &error);
            ASSERT_TRUE(identity.has_value()) << error;
            auto foreign = *identity;
            ++foreign.epoch_generation;
            const auto untouched = control;
            EXPECT_FALSE(protocol.cancelOriginalInvocation(foreign, MoEOverlayActivationStatusCode::ExplicitAbort, &error));
            EXPECT_FALSE(protocol.cancelOriginalInvocation(*identity, MoEOverlayActivationStatusCode::Success, &error));
            EXPECT_EQ(std::memcmp(&untouched, &control, sizeof(control)), 0);
            activateBoth(protocol, *identity);
            // Replay the documented native fail() publication window: the GPU
            // writes its first diagnostic before the final release-store of
            // Aborted. Scheduler cancellation must preserve that in-flight
            // record, not become another writer because state is still Active.
            control.follower_status.operation = static_cast<std::uint32_t>(MoEOverlayActivationOperation::ConsumeDispatch);
            control.follower_status.code = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::PayloadMismatch);
            const auto original_continuation = control.continuation_status;
            const auto original_follower = control.follower_status;
            ASSERT_TRUE(protocol.cancelOriginalInvocation(*identity, MoEOverlayActivationStatusCode::ExplicitAbort, &error));
            EXPECT_EQ(protocol.admissionState(), MoEOverlayActivationAdmissionState::Failed);
            EXPECT_EQ(control.admission.ready_signal, kMoEOverlayActivationAbortTimeline);
            for (const auto &buffer : control.buffers)
            {
                EXPECT_EQ(buffer.dispatch_signal.value, kMoEOverlayActivationAbortTimeline);
                EXPECT_EQ(buffer.return_signal.value, kMoEOverlayActivationAbortTimeline);
            }
            EXPECT_EQ(std::memcmp(&original_continuation, &control.continuation_status,
                                  sizeof(original_continuation)), 0);
            EXPECT_EQ(std::memcmp(&original_follower, &control.follower_status, sizeof(original_follower)), 0);
            EXPECT_TRUE(protocol.cancelOriginalInvocation(*identity, MoEOverlayActivationStatusCode::OutOfOrder, &error));
            EXPECT_EQ(control.admission.code, static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::ExplicitAbort));
            EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
            EXPECT_FALSE(protocol.reset(*identity, &error));
            EXPECT_FALSE(protocol.arm(makeTicket(config, 2u), 45u, 6000u, &error).has_value());
        }
    }

    /** Multiple original worker failures retain the already published watchdog reason. */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         SchedulerCancellationIsConcurrentAbsorbingAndPreservesWatchdogReason)
    {
        auto config = makeConfig({1});
        config.target.world_rank = config.source.world_rank;
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(makeTicket(config), 44u, 5000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        ASSERT_TRUE(protocol.markTimedOut(44u, 5000u, &error));
        const auto original_continuation = control.continuation_status;
        const auto original_follower = control.follower_status;
        std::atomic<int> failed{0};
        std::vector<std::thread> workers;
        for (int worker = 0; worker < 4; ++worker)
            workers.emplace_back([&]
            {
                std::string local_error;
                if (!protocol.cancelOriginalInvocation(*identity, MoEOverlayActivationStatusCode::ExplicitAbort, &local_error))
                    ++failed;
            });
        for (auto &worker : workers) worker.join();
        EXPECT_EQ(failed.load(), 0);
        EXPECT_EQ(control.admission.code, static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::TimedOut));
        EXPECT_EQ(control.admission.ready_signal, kMoEOverlayActivationAbortTimeline);
        EXPECT_EQ(std::memcmp(&original_continuation, &control.continuation_status, sizeof(original_continuation)), 0);
        EXPECT_EQ(std::memcmp(&original_follower, &control.follower_status, sizeof(original_follower)), 0);
        EXPECT_FALSE(protocol.markTimedOut(44u, 6000u, &error));
        EXPECT_FALSE(protocol.reset(*identity, &error));
        EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
    }

    /** A queued-but-unlaunched endpoint stays device-owned; cancellation proves no native completion. */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         SchedulerCancellationBeforeActivationDoesNotForgeGPUCompletion)
    {
        auto config = makeConfig({1});
        config.target.world_rank = config.source.world_rank;
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(makeTicket(config), 44u, 5000u, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        const auto continuation = control.continuation_status;
        const auto follower = control.follower_status;
        ASSERT_TRUE(protocol.cancelOriginalInvocation(*identity, MoEOverlayActivationStatusCode::ExplicitAbort, &error));
        EXPECT_EQ(std::memcmp(&continuation, &control.continuation_status, sizeof(continuation)), 0);
        EXPECT_EQ(std::memcmp(&follower, &control.follower_status, sizeof(follower)), 0);
        EXPECT_EQ(protocol.admissionState(), MoEOverlayActivationAdmissionState::Failed);
        EXPECT_EQ(control.admission.ready_signal, kMoEOverlayActivationAbortTimeline);
        EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
        EXPECT_FALSE(protocol.reset(*identity, &error));
    }

    /**
     * @test Shared physical bytes are an acquired terminal observation and reset
     * with their original epoch. A publisher may copy rows needed only by sibling
     * lanes even when this lane's own logical descriptor is empty.
     */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         SharedHiddenTrafficIsTerminalOwnedAndResetWithOriginalEpoch)
    {
        const auto config = makeConfig({7});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        for (std::uint64_t generation = 1; generation <= 20; ++generation)
        {
            const auto identity = protocol.arm(makeTicket(config, generation), generation, 10000, &error);
            ASSERT_TRUE(identity.has_value()) << error;
            activateBoth(protocol, *identity);
            ASSERT_TRUE(protocol.publishDispatch(*identity, 0, 0, 0, 0, &error)) << error;
            ASSERT_TRUE(protocol.consumeDispatch(*identity, 0, &error).has_value()) << error;
            // This is a recorded producer observation; no GPU execution is
            // claimed by the passive protocol/ownership replay.
            const std::uint64_t observed = generation % 2 ? 1024 : 0;
            control.continuation_status.published_shared_hidden_bytes = observed;
            EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
            ASSERT_TRUE(protocol.publishReturn(*identity, 0, 0, &error)) << error;
            ASSERT_TRUE(protocol.consumeReturn(*identity, 0, &error).has_value()) << error;
            completeBoth(protocol, *identity);
            const auto traffic = protocol.completedTraffic(*identity, &error);
            ASSERT_TRUE(traffic.has_value()) << error;
            EXPECT_EQ(traffic->dispatch_shared_hidden_bytes, observed);
            EXPECT_EQ(traffic->dispatch_payload_bytes, 0u);
            EXPECT_EQ(traffic->return_payload_bytes, 0u);
            EXPECT_EQ(protocol.endpointStatus(MoEOverlayActivationEndpoint::Continuation)
                .published_shared_hidden_bytes, observed);
            ASSERT_TRUE(protocol.reset(*identity, &error)) << error;
            EXPECT_EQ(control.continuation_status.published_shared_hidden_bytes, 0u);
            EXPECT_EQ(control.follower_status.published_shared_hidden_bytes, 0u);
            EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
        }
    }

    /** @test A follower cannot author source-owned shared hidden-copy totals. */
    TEST(Test__MoEOverlayActivationEpochProtocol,
         SharedHiddenTrafficRejectsForeignPublisher)
    {
        const auto config = makeConfig({7});
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(makeTicket(config), 1, 10000, &error);
        ASSERT_TRUE(identity.has_value()) << error;
        activateBoth(protocol, *identity);
        runOneStage(protocol, *identity, 0);
        control.follower_status.published_shared_hidden_bytes = 4;
        completeBoth(protocol, *identity);
        EXPECT_FALSE(protocol.completedTraffic(*identity, &error).has_value());
        EXPECT_NE(error.find("source-owned shared hidden"), std::string::npos);
    }

    /** @test Fixed cohort identity is in channel semantics and every invocation digest. */
    TEST(Test__MoEOverlayActivationEpochProtocol, FixedCohortTargetIsAuthenticatedAcrossReplay)
    {
        auto config = makeConfig({7});
        config.return_role = MoEOverlayActivationReturnRole::JointRoot;
        EXPECT_FALSE(config.valid());
        config.cohort_target = MoEOverlayRouteTarget::cpuCohort(4).word();
        ASSERT_TRUE(config.valid());
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        auto foreign = config;
        foreign.cohort_target = MoEOverlayRouteTarget::cpuCohort(5).word();
        EXPECT_THROW(MoEOverlayActivationEpochProtocol(control, foreign), std::logic_error);
        EXPECT_EQ(control.channel.abi_version, 10u);
        // Size compatibility is insufficient to establish the native roster
        // slot authority in this version. Reject old and future records.
        for (const std::uint32_t foreign_version : {9u, 11u})
        {
            control.channel.abi_version = foreign_version;
            EXPECT_THROW(MoEOverlayActivationEpochProtocol(control, config), std::logic_error);
        }
        control.channel.abi_version = kMoEOverlayActivationABIVersion;
        MoEOverlayActivationEpochProtocol protocol(control, config);
        std::string error;
        const auto identity = protocol.arm(makeTicket(config), 1, 10000, &error);
        ASSERT_TRUE(identity) << error;
        EXPECT_EQ(identity->cohort_target, config.cohort_target);
        auto changed = *identity;
        changed.cohort_target = foreign.cohort_target;
        EXPECT_NE(moeOverlayActivationIdentityDigest(changed), identity->digest);
        changed.digest = moeOverlayActivationIdentityDigest(changed);
        EXPECT_FALSE(
            protocol.activate(MoEOverlayActivationEndpoint::Continuation, changed, &error));
        EXPECT_TRUE(protocol.cancelOriginalInvocation(
            *identity, MoEOverlayActivationStatusCode::ExplicitAbort, &error));
    }

} // namespace llaminar2::test

namespace llaminar2::test {
/** Actual original submission identity is constructed without reading/resetting live epoch state. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionDescribesWithoutTouchingLiveControl)
{
    const auto config=makeConfig({3,5});
    MoEOverlayActivationEpochControl control{};MoEOverlayActivationEpochProtocol::initialize(control,config);
    MoEOverlayActivationEpochProtocol lane(control,config);
    const auto first=makeTicket(config);
    const MoEOverlayInferenceTopologyIdentity topology{.workspace_generation=config.workspace_generation,
        .topology_fingerprint_low=config.topology_fingerprint_low,.topology_fingerprint_high=config.topology_fingerprint_high,
        .source_world_rank=config.source.world_rank,.target_world_rank=config.target.world_rank};
    MoEOverlayInferenceTransactionProtocol original({.topology=topology,
        .slot_count=moeOverlayInferenceTransactionSlotCount(3),.max_request_count=1,.max_rows_per_request=1,.max_mtp_draft_depth=3});
    ASSERT_TRUE(original.beginCommand({.request_generation=1,.command_id=1,.initial_placement_epoch=41}));
    const auto a=original.accept(first);ASSERT_TRUE(a.accepted());ASSERT_TRUE(original.markSubmitted(a.slot_index,first));
    const auto issued=original.originalSubmittedSlot(first);
    const auto armed=lane.arm(first,1u,10'000u);ASSERT_TRUE(armed.has_value());activateBoth(lane,*armed);
    std::array<std::byte,sizeof(control)> before{};std::memcpy(before.data(),&control,sizeof(control));
    auto expected_first=*armed;expected_first.original_slot_index=static_cast<std::uint32_t>(issued.index());
    expected_first.digest=moeOverlayActivationIdentityDigest(expected_first);
    EXPECT_EQ(lane.describeOriginalNativeSubmission(issued,1u),expected_first);
    auto second=makeTicket(config,2u);second.command_id=1u;
    const auto b=original.accept(second);ASSERT_TRUE(b.accepted());ASSERT_TRUE(original.markSubmitted(b.slot_index,second));
    const auto next=original.originalSubmittedSlot(second);
    const auto described=lane.describeOriginalNativeSubmission(next,2u);
    MoEOverlayActivationEpochControl reference{};MoEOverlayActivationEpochProtocol::initialize(reference,config);
    MoEOverlayActivationEpochProtocol reference_lane(reference,config);
    const auto expected=reference_lane.arm(second,2u,20'000u);ASSERT_TRUE(expected.has_value());
    auto expected_native=*expected;expected_native.original_slot_index=static_cast<std::uint32_t>(next.index());
    expected_native.digest=moeOverlayActivationIdentityDigest(expected_native);
    EXPECT_EQ(described,expected_native);
    EXPECT_EQ(std::memcmp(before.data(),&control,sizeof(control)),0);
    EXPECT_THROW((void)lane.describeOriginalNativeSubmission(next,0u),std::logic_error);
    const auto abort=makeMoEOverlayInferenceTerminalTicket(topology,{.request_generation=1,.command_id=1,.initial_placement_epoch=41},
        3u,41u,MoEOverlayInferenceTransactionAction::Abort,17);
    EXPECT_EQ(original.accept(abort).status,MoEOverlayInferenceAdmissionStatus::Aborted);
    EXPECT_THROW((void)lane.describeOriginalNativeSubmission(next,2u),std::logic_error);
    EXPECT_EQ(std::memcmp(before.data(),&control,sizeof(control)),0);
}

/** A semantic Complete status alone is insufficient for exact post-payload retirement. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionRejectsStaleAndPrematurePacketRetirement)
{
    const auto config=makeConfig({3,5});MoEOverlayActivationEpochControl control{};
    MoEOverlayActivationEpochProtocol::initialize(control,config);MoEOverlayActivationEpochProtocol lane(control,config);
    const auto identity=lane.arm(makeTicket(config),1u,10'000u);ASSERT_TRUE(identity.has_value());
    MoEOverlayNativeInvocationSlot record{.identity=*identity,.original_slot_index=1u,.published_generation=1u,.timeout_not_before_ns=10'000u};
    record.identity.original_slot_index=1u;record.identity.digest=moeOverlayActivationIdentityDigest(record.identity);
    EXPECT_TRUE(moeOverlayNativeInvocationSlotMatches(record,1u,5u,control.channel,1));
    auto changed=record;changed.published_generation=2u;EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(changed,1u,5u,control.channel,1));
    changed=record;changed.abi_version=0u;EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(changed,1u,5u,control.channel,1));
    changed=record;changed.cancelled_generation=1u;EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(changed,1u,5u,control.channel,1));
    changed=record;changed.timeout_not_before_ns=0u;EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(changed,1u,5u,control.channel,1));
    EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(record,0u,5u,control.channel,1));
    EXPECT_FALSE(moeOverlayNativeInvocationSlotMatches(record,1u,5u,control.channel,3));
    auto &receipt=record.continuation;
    receipt.status.digest=record.identity.digest;receipt.status.generation=1u;
    receipt.status.state=static_cast<std::uint32_t>(MoEOverlayActivationEndpointState::Complete);
    receipt.status.code=static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::Success);
    receipt.status.last_consumed_stage=1;receipt.status.last_published_stage=1;
    EXPECT_FALSE(moeOverlayNativeInvocationEndpointRetired(receipt,record.identity));
    receipt.retired_generation=1u;EXPECT_FALSE(moeOverlayNativeInvocationEndpointRetired(receipt,record.identity));
    receipt.status.payload_retired_generation=1u;EXPECT_TRUE(moeOverlayNativeInvocationEndpointRetired(receipt,record.identity));
    receipt.retired_generation=2u;EXPECT_FALSE(moeOverlayNativeInvocationEndpointRetired(receipt,record.identity));
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test A prior ready acquire cannot authorize a newly installed native identity. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionRequiresCurrentPublicationReplay)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        const auto ready = kMoEOverlayActivationAdmissionTimeline;
        std::uint64_t generation = 41;
        std::uint64_t publication = ready;
        EXPECT_TRUE(moeOverlayActivationAdmissionPublicationCurrent(41,
            [&]() { return generation; }, [&]() { return publication; }));
        // Closing admission can precede the generation rewrite. The original
        // ready release cannot authorize this interval. A ready-first helper
        // returns true here, even though no current publication exists.
        int acquire_count = 0;
        const auto acquire = [&](const std::uint64_t* address) {
            const auto value = *address;
            if (++acquire_count == 1) publication = 0;
            return value;
        };
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationCurrent(41,
            [&]() { return acquire(&generation); },
            [&]() { return acquire(&publication); }));
        EXPECT_EQ(acquire_count, 2);
        // A newer identity remains unadmitted until its final ready release.
        generation = 42;
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationCurrent(42,
            [&]() { return generation; }, [&]() { return publication; }));
        publication = ready;
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationCurrent(41,
            [&]() { return generation; }, [&]() { return publication; }));
        EXPECT_TRUE(moeOverlayActivationAdmissionPublicationCurrent(42,
            [&]() { return generation; }, [&]() { return publication; }));
        EXPECT_TRUE(moeOverlayActivationAdmissionPublicationMatches(41, 41, ready));
        // The first ready acquire saw 41; source closed ready and installed 42.
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationMatches(42, 42, 0));
        // Generation must be read before the final ready acquire, like the peer
        // epoch helper: a replacement publication cannot authenticate old bytes.
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationMatches(41, 42, ready));
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationMatches(0, 0, ready));
        EXPECT_FALSE(moeOverlayActivationAdmissionPublicationMatches(
            42, 42, kMoEOverlayActivationAbortTimeline));
        EXPECT_TRUE(moeOverlayActivationAdmissionPublicationMatches(42, 42, ready));
    }
}
}

namespace llaminar2::test
{
/** @test The original native rearm does not normalize unknown lifecycle bytes. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionRejectsUnknownRearmStatesReplay)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        for (std::uint32_t state = 0; state < 32; ++state)
        {
            const bool expected =
                state == static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Idle) ||
                state == static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Armed);
            EXPECT_EQ(moeOverlayNativeInvocationRearmStateAllowed(state), expected) << state;
        }
        EXPECT_FALSE(moeOverlayNativeInvocationRearmStateAllowed(~std::uint32_t{0}));
    }
}
}

namespace llaminar2::test
{
/** @test Cancellation between the original state decision and ready CAS cannot be overwritten. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionReadyCASPreservesCancellationReplay)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        for (const bool closing : {false, true})
        {
            std::atomic<std::uint64_t> ready{closing ? kMoEOverlayActivationAdmissionTimeline : 0u};
            std::atomic<std::uint32_t> state{
                static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Armed)};
            const auto first_error = static_cast<std::uint32_t>(MoEOverlayActivationStatusCode::OutOfOrder);
            std::atomic<std::uint32_t> error{first_error};
            unsigned acquires = 0, compares = 0;
            const auto acquire = [&]() {
                ++acquires;
                return ready.load(std::memory_order_acquire);
            };
            const auto compare = [&](std::uint64_t expected, std::uint64_t desired) {
                ++compares;
                // The native state CAS already won. Original cancellation now
                // wins just before the ready CAS, preserving its earlier error.
                state.store(static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Failed),
                            std::memory_order_release);
                ready.store(kMoEOverlayActivationAbortTimeline, std::memory_order_release);
                return ready.compare_exchange_strong(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire);
            };
            const bool published = closing
                ? closeMoEOverlayActivationAdmissionPublication(acquire, compare)
                : publishMoEOverlayActivationAdmissionPublication(compare);
            EXPECT_FALSE(published);
            EXPECT_EQ(acquires, closing ? 1u : 0u);
            EXPECT_EQ(compares, 1u);
            EXPECT_EQ(ready.load(std::memory_order_acquire), kMoEOverlayActivationAbortTimeline);
            EXPECT_EQ(state.load(std::memory_order_acquire),
                      static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Failed));
            EXPECT_EQ(error.load(std::memory_order_acquire), first_error);
        }
    }
}

/** @test Successful original close/open uses one CAS each and never polls for progress. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionReadyCASHasExactOperationsReplay)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        for (const std::uint64_t initial : {std::uint64_t{0}, kMoEOverlayActivationAdmissionTimeline})
        {
            std::atomic<std::uint64_t> ready{initial};
            unsigned acquires = 0, compares = 0;
            const auto acquire = [&]() { ++acquires; return ready.load(std::memory_order_acquire); };
            const auto compare = [&](std::uint64_t expected, std::uint64_t desired) {
                ++compares;
                return ready.compare_exchange_strong(expected, desired,
                    std::memory_order_acq_rel, std::memory_order_acquire);
            };
            EXPECT_TRUE(closeMoEOverlayActivationAdmissionPublication(acquire, compare));
            EXPECT_EQ(ready.load(std::memory_order_acquire), 0u);
            EXPECT_EQ(acquires, 1u);
            EXPECT_EQ(compares, 1u);
            EXPECT_TRUE(publishMoEOverlayActivationAdmissionPublication(compare));
            EXPECT_EQ(ready.load(std::memory_order_acquire), kMoEOverlayActivationAdmissionTimeline);
            EXPECT_EQ(acquires, 1u);
            EXPECT_EQ(compares, 2u);
        }
        for (const auto initial : {std::uint64_t{2}, kMoEOverlayActivationAbortTimeline})
        {
            std::atomic<std::uint64_t> ready{initial};
            unsigned compares = 0;
            EXPECT_FALSE(closeMoEOverlayActivationAdmissionPublication(
                [&]() { return ready.load(std::memory_order_acquire); },
                [&](std::uint64_t, std::uint64_t) { ++compares; return true; }));
            EXPECT_EQ(compares, 0u);
            EXPECT_EQ(ready.load(std::memory_order_acquire), initial);
        }
    }
}

/** @test The unsuccessful grant poll recognizes Failed while preserving ready-Abort short circuit. */
TEST(Test__MoEOverlayActivationEpochProtocol, OriginalNativeAdmissionFailedWaitUsesOnlyRequiredLoadsReplay)
{
    for (int replay = 0; replay < 20; ++replay)
    {
        unsigned loads = 0;
        const auto failed = [&]() { ++loads; return static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Failed); };
        EXPECT_TRUE(moeOverlayActivationAdmissionWaitAborted(kMoEOverlayActivationAbortTimeline, failed));
        EXPECT_EQ(loads, 0u);
        EXPECT_TRUE(moeOverlayActivationAdmissionWaitAborted(kMoEOverlayActivationAdmissionTimeline, failed));
        EXPECT_EQ(loads, 1u);
        EXPECT_FALSE(moeOverlayActivationAdmissionWaitAborted(kMoEOverlayActivationAdmissionTimeline,
            [&]() { ++loads; return static_cast<std::uint32_t>(MoEOverlayActivationAdmissionState::Armed); }));
        EXPECT_EQ(loads, 2u);
    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test The actual shared device helper re-waits after a pre-grant edge passed old work. */
TEST(Test__MoEOverlayActivationEpochProtocol, GrantedDispatchWaitsForCurrentPublicationReplay) {
    for(int replay=0;replay<20;++replay)for(const std::uint64_t required:{1u,2u,5u,17u}) {
        const auto old_stream_wait=required;
        ASSERT_GE(old_stream_wait,required); // Prior generation's retained scheduling edge passed.
        std::uint64_t current=0;int reads=0,pauses=0,abort_reads=0;
        const auto acquire=[&] { ++reads;return current; };
        // Original validator's single acquired load rejected the newly reset
        // descriptor even though its grant was current and dispatch delayed.
        EXPECT_NE(acquire(),required);
        const auto observed=awaitMoEOverlayGrantedDispatchPublication(required,acquire,
            [&] { ++abort_reads;return false; },[&] { if(++pauses==2)current=required; });
        EXPECT_EQ(observed,required);EXPECT_EQ(pauses,2);EXPECT_EQ(abort_reads,2);EXPECT_EQ(reads,4);
        int unexpected=0;
        EXPECT_EQ(awaitMoEOverlayGrantedDispatchPublication(required,[&] { return required; },
            [&] { ++unexpected;return false; },[&] { ++unexpected; }),required);
        EXPECT_EQ(unexpected,0);
    }
}
/** @test Cancellation or an invalid timeline cannot strand a newly granted consumer. */
TEST(Test__MoEOverlayActivationEpochProtocol, GrantedDispatchAbortAndOvershootRemainExactReplay) {
    for(int replay=0;replay<20;++replay) {
        int polls=0,pauses=0;
        EXPECT_EQ(awaitMoEOverlayGrantedDispatchPublication(1,[&] { ++polls;return std::uint64_t{0}; },
            [&] { return polls==2; },[&] { ++pauses; }),kMoEOverlayActivationAbortTimeline);
        EXPECT_EQ(polls,2);EXPECT_EQ(pauses,1);
        EXPECT_EQ(awaitMoEOverlayGrantedDispatchPublication(1,[] { return kMoEOverlayActivationAbortTimeline; },
            [] { ADD_FAILURE()<<"Published Abort must not require another read";return false; },[] {}),kMoEOverlayActivationAbortTimeline);
        // Overshoot remains a rejected exact descriptor; no loop repairs it to the requested word.
        EXPECT_EQ(awaitMoEOverlayGrantedDispatchPublication(1,[] { return std::uint64_t{2}; },[] { return false; },[] {}),2u);
        EXPECT_EQ(awaitMoEOverlayGrantedDispatchPublication(0,[] { ADD_FAILURE();return std::uint64_t{0}; },
            [] { return false; },[] {}),kMoEOverlayActivationAbortTimeline);
    }
}
/** @test Actual protocol reset closes previous dispatch words before the next identity is published. */
TEST(Test__MoEOverlayActivationEpochProtocol, GrantedDispatchResetStartsUnpublishedBeforeRearmReplay) {
    const auto config=makeConfig();MoEOverlayActivationEpochControl control{};
    MoEOverlayActivationEpochProtocol::initialize(control,config);
    MoEOverlayActivationEpochProtocol protocol(control,config);std::string error;
    for(std::uint64_t generation=1;generation<=20;++generation) {
        const auto identity=protocol.arm(makeTicket(config,generation,generation),generation,1000,&error);
        ASSERT_TRUE(identity)<<error;
        EXPECT_EQ(control.admission.ready_signal,kMoEOverlayActivationAdmissionTimeline);
        for(const auto& bank:control.buffers) { EXPECT_EQ(bank.dispatch_signal.value,0u);EXPECT_EQ(bank.dispatch_descriptor.timeline,0u); }
        activateBoth(protocol,*identity);
        for(std::uint32_t stage=0;stage<config.model_layer_indices.size();++stage)runOneStage(protocol,*identity,stage);
        completeBoth(protocol,*identity);
        ASSERT_TRUE(protocol.reset(*identity,&error))<<error;
        EXPECT_EQ(control.admission.ready_signal,0u);
        for(const auto& bank:control.buffers) { EXPECT_EQ(bank.dispatch_signal.value,0u);EXPECT_EQ(bank.dispatch_descriptor.timeline,0u); }
    }
}
}
