/**
 * @file Test__MoEOverlayActivationProgressLifecycle.cpp
 * @brief Model-free integration proof for per-rendezvous activation deadlines.
 *
 * A real 48-layer CPU follower may spend longer than one collective timeout
 * completing the whole request while every peer publication remains healthy.
 * This test drives the production activation protocol with a deterministic
 * synthetic clock: each individual dispatch arrives inside its own deadline,
 * the aggregate transaction exceeds that interval many times over, and a
 * genuinely stalled next generation still becomes terminally timed out.
 */

#include "execution/moe/MoEOverlayActivationEpochProtocol.h"
#include "execution/moe/MoEOverlayActivationRendezvousDeadline.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        /** @return A production-width ordered main-model stage manifest. */
        MoEOverlayActivationEpochConfig progressConfig()
        {
            std::vector<std::int32_t> layers;
            layers.reserve(48u);
            for (std::int32_t layer = 0; layer < 48; ++layer)
                layers.push_back(layer);
            return {
                .channel_nonce = 0x7a9c1301u,
                .topology_fingerprint_low = 0x2134567812345678ull,
                .topology_fingerprint_high = 0x9765432187654321ull,
                .workspace_generation = 9u,
                .source = {
                    .world_rank = 1,
                    .participant_id = 0,
                    .tier_priority = 0,
                    .domain_ordinal = 0,
                },
                .target = {
                    .world_rank = 0,
                    .participant_id = 2,
                    .tier_priority = 1,
                    .domain_ordinal = 1,
                },
                .lane_ordinal = 0u,
                .graph_role_mask =
                    std::uint32_t{1}
                    << static_cast<std::uint32_t>(
                           MoEOverlayInferenceGraphRole::MainPrefill),
                .model_layer_indices = std::move(layers),
            };
        }

        /** @return One exact single-row prefill ticket for the configured lane. */
        MoEOverlayInferenceTransactionTicket progressTicket(
            const MoEOverlayActivationEpochConfig &config,
            std::uint64_t ordinal)
        {
            const MoEOverlayInferenceTopologyIdentity topology{
                .workspace_generation = config.workspace_generation,
                .topology_fingerprint_low = config.topology_fingerprint_low,
                .topology_fingerprint_high = config.topology_fingerprint_high,
                .source_world_rank = config.source.world_rank,
                .target_world_rank = config.target.world_rank,
            };
            const MoEOverlayInferenceCommandIdentity command{
                .request_generation = ordinal,
                .command_id = ordinal,
                .initial_placement_epoch = 3u,
            };
            return makeMoEOverlayInferenceExecutionTicket(
                topology,
                command,
                ordinal,
                /*logical_step_id=*/ordinal,
                /*placement_epoch=*/3u,
                MoEOverlayInferenceGraphRole::MainPrefill,
                /*request_count=*/1,
                /*logical_rows_per_request=*/1,
                /*physical_rows_per_request=*/1);
        }

        /** Activate both single-writer endpoints for one armed identity. */
        void activateProgressEndpoints(
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
    } // namespace

    TEST(Test__MoEOverlayActivationProgressLifecycle,
         FortyEightHealthyRendezvousOutliveOneTimeoutAndAStallStillFails)
    {
        using Deadline = MoEOverlayActivationRendezvousDeadline;
        constexpr auto timeout = std::chrono::milliseconds(10);
        constexpr auto healthy_edge = std::chrono::milliseconds(9);
        auto now = Deadline::TimePoint{std::chrono::nanoseconds{1}};

        const auto config = progressConfig();
        MoEOverlayActivationEpochControl control;
        MoEOverlayActivationEpochProtocol::initialize(control, config);
        MoEOverlayActivationEpochProtocol protocol(control, config);

        const auto initial_window = Deadline::begin(
            MoEOverlayActivationRendezvousKind::EndpointCompletion,
            timeout,
            now);
        const auto timeout_not_before =
            initial_window.deadlineNanoseconds();
        ASSERT_TRUE(timeout_not_before.has_value());

        std::string error;
        const auto identity = protocol.arm(
            progressTicket(config, 1u),
            /*epoch_generation=*/1u,
            *timeout_not_before,
            &error);
        ASSERT_TRUE(identity.has_value()) << error;
        activateProgressEndpoints(protocol, *identity);

        const auto transaction_started = now;
        for (std::uint32_t stage = 0u; stage < 48u; ++stage)
        {
            const auto dispatch_window = Deadline::begin(
                MoEOverlayActivationRendezvousKind::DispatchPublication,
                timeout,
                now);
            now += healthy_edge;
            ASSERT_TRUE(dispatch_window.waitingAllowed(now))
                << "stage=" << stage;

            ASSERT_TRUE(protocol.publishDispatch(
                *identity,
                stage,
                /*live_rows=*/1u,
                /*live_entries=*/1u,
                /*payload_bytes=*/64u,
                &error))
                << "stage=" << stage << " " << error;
            ASSERT_TRUE(protocol.consumeDispatch(
                *identity, stage, &error).has_value())
                << "stage=" << stage << " " << error;
            ASSERT_TRUE(protocol.publishReturn(
                *identity, stage, /*payload_bytes=*/32u, &error))
                << "stage=" << stage << " " << error;
            ASSERT_TRUE(protocol.consumeReturn(
                *identity, stage, &error).has_value())
                << "stage=" << stage << " " << error;
        }

        EXPECT_GT(now - transaction_started, timeout)
            << "The proof must exceed one aggregate timeout while progressing";
        ASSERT_TRUE(protocol.complete(
            MoEOverlayActivationEndpoint::Follower, *identity, &error))
            << error;
        ASSERT_TRUE(protocol.complete(
            MoEOverlayActivationEndpoint::Continuation, *identity, &error))
            << error;
        ASSERT_TRUE(protocol.reset(*identity, &error)) << error;

        const auto stalled_window = Deadline::begin(
            MoEOverlayActivationRendezvousKind::DispatchPublication,
            timeout,
            now);
        const auto second_timeout_not_before =
            stalled_window.deadlineNanoseconds();
        ASSERT_TRUE(second_timeout_not_before.has_value());
        const auto stalled_identity = protocol.arm(
            progressTicket(config, 2u),
            /*epoch_generation=*/2u,
            *second_timeout_not_before,
            &error);
        ASSERT_TRUE(stalled_identity.has_value()) << error;
        activateProgressEndpoints(protocol, *stalled_identity);

        now += timeout;
        EXPECT_FALSE(stalled_window.waitingAllowed(now));
        const auto timeout_observation =
            stalled_window.timeoutObservationNanoseconds(now);
        ASSERT_TRUE(timeout_observation.has_value());
        ASSERT_TRUE(protocol.markTimedOut(
            stalled_identity->epoch_generation,
            *timeout_observation,
            &error))
            << error;
        EXPECT_EQ(
            protocol.admissionState(),
            MoEOverlayActivationAdmissionState::Failed);
        EXPECT_FALSE(protocol.reset(*stalled_identity, &error));
    }
} // namespace llaminar2::test
