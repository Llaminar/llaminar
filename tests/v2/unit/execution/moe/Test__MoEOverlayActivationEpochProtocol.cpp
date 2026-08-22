/**
 * @file Test__MoEOverlayActivationEpochProtocol.cpp
 * @brief Adversarial CPU tests for device-owned ExpertOverlay activation epochs.
 */

#include "execution/moe/MoEOverlayActivationEpochProtocol.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
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
            EXPECT_EQ(returned->live_entries, 0u);
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
        EXPECT_EQ(kMoEOverlayActivationABIVersion, 4u);
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
            /*deadline_ns=*/10'000u,
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
        EXPECT_EQ(traffic->return_live_entries, 0u);
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
} // namespace llaminar2::test
