/**
 * @file Test__MoEOverlayNodeLocalRankBatchTransport.cpp
 * @brief CPU-only protocol tests for the node-local shared activation channel.
 *
 * These tests model two MPI ranks in one process while retaining distinct
 * topology and transport objects. They prove that same-node rank pairs exchange
 * dispatch and return payloads through the same POSIX shared pages without one
 * MPI operation, and that cross-node pairs can never construct that transport.
 */

#include "execution/moe/MoEOverlayNodeLocalRankBatchTransport.h"
#include "mocks/MockMPIContext.h"
#include "mocks/MockMPITopology.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    constexpr size_t kRowsPerParticipant = 3;
    constexpr size_t kEntriesPerParticipant = 6;
    constexpr int kDModel = 4;
    constexpr int kTopK = 2;
    const std::vector<int> kParticipants{4, 7};

    /** @return ABI role bit used by one retained activation graph family. */
    constexpr std::uint32_t roleBit(MoEOverlayInferenceGraphRole role)
    {
        return std::uint32_t{1} << static_cast<std::uint32_t>(role);
    }

    /** @brief Construct one rank context with authoritative physical locality. */
    std::shared_ptr<MockMPIContext> makeContext(
        int rank,
        int ranks_per_node)
    {
        auto context = std::make_shared<MockMPIContext>(rank, 2);
        context->set_topology(
            MockMPITopology::createSimple(rank, 2, ranks_per_node));
        return context;
    }

    /** @brief Allocate the MPI codec workspace retained by the typed factory. */
    std::shared_ptr<MoEOverlayRankBatchWireWorkspace> makeWireWorkspace()
    {
        return std::make_shared<MoEOverlayRankBatchWireWorkspace>(
            MoEOverlayRankBatchWireWorkspace::Config{
                .participant_ids = kParticipants,
                .max_total_rows =
                    kRowsPerParticipant * kParticipants.size(),
                .max_total_entries =
                    kEntriesPerParticipant * kParticipants.size(),
                .d_model = kDModel,
                .top_k = kTopK,
            });
    }

    /** @brief Build one endpoint's complete immutable channel contract. */
    MoEOverlayRankBatchTransportConfig makeConfig(
        std::shared_ptr<MockMPIContext> context,
        const std::string &identity)
    {
        return MoEOverlayRankBatchTransportConfig{
            .mpi_ctx = std::move(context),
            .source_world_rank = 0,
            .target_world_rank = 1,
            .workspace = makeWireWorkspace(),
            .max_rows_per_participant = kRowsPerParticipant,
            .max_entries_per_participant = kEntriesPerParticipant,
            .d_model = kDModel,
            .top_k = kTopK,
            .tier_index = 1,
            .domain_ordinal = 2,
            .channel_identity = identity,
            .transaction_slot_count = 32,
            .transaction_topology = {
                .workspace_generation = 17u,
                .topology_fingerprint_low = 0x1234567812345678ull,
                .topology_fingerprint_high = 0x8765432187654321ull,
                .source_world_rank = 0,
                .target_world_rank = 1,
            },
            .source_endpoint = {
                .world_rank = 0,
                .participant_id = 2,
                .tier_priority = 0,
                .domain_ordinal = 0,
            },
            .target_tier_priority = 10,
            .activation_graph_families = {
                {
                    .graph_role_mask =
                        roleBit(MoEOverlayInferenceGraphRole::MainPrefill) |
                        roleBit(MoEOverlayInferenceGraphRole::MainDecode) |
                        roleBit(
                            MoEOverlayInferenceGraphRole::MTPGroupedVerifier),
                    .model_layer_indices = {0, 1, 2, 3},
                },
                {
                    .graph_role_mask =
                        roleBit(MoEOverlayInferenceGraphRole::MTPDraft),
                    .model_layer_indices = {4},
                },
            },
            .activation_layout =
                planMoEOverlayNodeLocalActivationLayout({
                    .participant_count = kParticipants.size(),
                    .max_rows_per_participant = kRowsPerParticipant,
                    .max_entries_per_participant = kEntriesPerParticipant,
                    .d_model = kDModel,
                    .activation_graph_family_count = 2u,
                }),
            .local_lanes = {
                {.participant_id = kParticipants[0],
                 .device = DeviceId::cpu()},
                {.participant_id = kParticipants[1],
                 .device = DeviceId::cpu()},
            },
        };
    }

    /** @brief Derive a collision-free identity for parallel unit-test processes. */
    std::string uniqueIdentity(const char *suffix)
    {
        return std::string("node_local_rank_batch_") + suffix + "_pid" +
               std::to_string(static_cast<unsigned long>(::getpid()));
    }

    /** @brief Populate one in-place dispatch view with deterministic live rows. */
    void populateDispatch(
        MoEOverlaySparseRows &rows,
        int source_participant,
        int first_row,
        size_t live_rows,
        size_t live_entries,
        float value_base)
    {
        ASSERT_LE(live_rows, rows.row_capacity);
        ASSERT_LE(live_entries, rows.entry_capacity);
        rows.residency_epoch = 44;
        rows.source_participant = source_participant;
        rows.live_row_count = live_rows;
        rows.live_entry_count = live_entries;

        size_t entry = 0;
        for (size_t row = 0; row < live_rows; ++row)
        {
            rows.row_ids_host[row] = first_row + static_cast<int>(row);
            rows.entry_offsets_host[row] = static_cast<int32_t>(entry);
            const size_t remaining_rows = live_rows - row;
            const size_t remaining_entries = live_entries - entry;
            const size_t row_entries =
                (remaining_entries + remaining_rows - 1u) / remaining_rows;
            for (size_t local = 0; local < row_entries; ++local, ++entry)
            {
                rows.expert_ids_host[entry] =
                    rows.target_participant * 10 + static_cast<int>(entry);
                rows.route_weights_host[entry] =
                    value_base + static_cast<float>(entry) * 0.125f;
            }
            for (int column = 0; column < kDModel; ++column)
            {
                rows.hidden_rows_fp32[row * kDModel + column] =
                    value_base + static_cast<float>(row * kDModel + column);
            }
        }
        ASSERT_EQ(entry, live_entries);
        rows.entry_offsets_host[live_rows] =
            static_cast<int32_t>(live_entries);
    }

    /** @brief Populate one in-place return view as if a local expert completed. */
    void populateReturn(
        MoEOverlayReturnRows &rows,
        int continuation_participant,
        int first_row,
        size_t live_rows,
        float value_base)
    {
        ASSERT_LE(live_rows, rows.row_capacity);
        rows.residency_epoch = 44;
        rows.target_participant = continuation_participant;
        rows.live_row_count = live_rows;
        for (size_t row = 0; row < live_rows; ++row)
        {
            rows.row_ids_host[row] = first_row + static_cast<int>(row);
            for (int column = 0; column < kDModel; ++column)
            {
                rows.output_rows_fp32[row * kDModel + column] =
                    value_base + static_cast<float>(row * kDModel + column);
            }
        }
    }
} // namespace

TEST(Test__MoEOverlayNodeLocalRankBatchTransport,
     EmptyDispatchAccountingMatchesDevicePacketABI)
{
    const MoEOverlaySparseRows empty_rows{
        .d_model = kDModel,
        .top_k = kTopK,
    };

    EXPECT_EQ(compactMoEOverlayDispatchBytes(empty_rows), 0u);
    EXPECT_EQ(
        moeOverlayDispatchPayloadBytes(
            /*live_rows=*/0u,
            /*live_entries=*/0u,
            /*d_model=*/kDModel),
        0u);
}

TEST(Test__MoEOverlayNodeLocalRankBatchTransport,
     SelectionIsStrictlyPhysicalNodeScoped)
{
    auto same_node = makeContext(/*rank=*/0, /*ranks_per_node=*/2);
    EXPECT_EQ(
        resolveMoEOverlayRankBatchTransportKind(*same_node, 0, 1),
        MoEOverlayRankBatchTransportKind::NodeLocalSharedRows);

    auto cross_node = makeContext(/*rank=*/0, /*ranks_per_node=*/1);
    EXPECT_EQ(
        resolveMoEOverlayRankBatchTransportKind(*cross_node, 0, 1),
        MoEOverlayRankBatchTransportKind::MPI);

    auto portable = createMoEOverlayRankBatchTransport(makeConfig(
        cross_node,
        uniqueIdentity("portable_inter_node")));
    ASSERT_NE(portable, nullptr);
    EXPECT_EQ(portable->kind(), MoEOverlayRankBatchTransportKind::MPI);
    EXPECT_FALSE(portable->hasSharedRowStorage());
    EXPECT_EQ(
        dynamic_cast<IMoEOverlayMappedActivationTransport *>(portable.get()),
        nullptr);

    EXPECT_THROW(
        MoEOverlayNodeLocalRankBatchTransport(makeConfig(
            cross_node,
            uniqueIdentity("forbidden_inter_node"))),
        std::runtime_error);
}

TEST(Test__MoEOverlayNodeLocalRankBatchTransport,
     ChannelIdentityRequiresCanonicalParticipantOrder)
{
    EXPECT_EQ(
        makeMoEOverlayRankBatchChannelIdentity(
            /*tier_index=*/1,
            /*domain_ordinal=*/2,
            /*source_world_rank=*/0,
            /*target_world_rank=*/1,
            kParticipants),
        "tier1#domain2#rank0to1#p4,7,");

    const std::vector<int> reversed{7, 4};
    EXPECT_THROW(
        static_cast<void>(
            makeMoEOverlayRankBatchChannelIdentity(1, 2, 0, 1, reversed)),
        std::invalid_argument);
    EXPECT_THROW(
        static_cast<void>(
            makeMoEOverlayRankBatchChannelIdentity(
                1, 2, 0, 0, kParticipants)),
        std::invalid_argument);
}

TEST(Test__MoEOverlayNodeLocalRankBatchTransport,
     DispatchAndReturnUseSharedPagesWithoutMPI)
{
    auto source_context = makeContext(/*rank=*/0, /*ranks_per_node=*/2);
    auto target_context = makeContext(/*rank=*/1, /*ranks_per_node=*/2);
    const std::string identity = uniqueIdentity("round_trip");

    /* Construction includes the production pre-registration first-touch
     * rendezvous. Model the two MPI ranks concurrently so neither endpoint can
     * pin the complete mapping before its peer places the opposite direction. */
    std::shared_ptr<IMoEOverlayRankBatchTransport> source;
    std::shared_ptr<IMoEOverlayRankBatchTransport> target;
    std::exception_ptr source_error;
    std::exception_ptr target_error;
    std::thread source_builder([&]
                               {
                                   try
                                   {
                                       source = createMoEOverlayRankBatchTransport(
                                           makeConfig(source_context, identity));
                                   }
                                   catch (...)
                                   {
                                       source_error = std::current_exception();
                                   }
                               });
    std::thread target_builder([&]
                               {
                                   try
                                   {
                                       target = createMoEOverlayRankBatchTransport(
                                           makeConfig(target_context, identity));
                                   }
                                   catch (...)
                                   {
                                       target_error = std::current_exception();
                                   }
                               });
    source_builder.join();
    target_builder.join();
    if (source_error)
        std::rethrow_exception(source_error);
    if (target_error)
        std::rethrow_exception(target_error);
    ASSERT_EQ(source->kind(),
              MoEOverlayRankBatchTransportKind::NodeLocalSharedRows);
    ASSERT_EQ(target->kind(),
              MoEOverlayRankBatchTransportKind::NodeLocalSharedRows);

    const std::string canonical_identity =
        makeMoEOverlayRankBatchChannelIdentity(1, 2, 0, 1, kParticipants);
    MoEOverlayRankBatchTransportRegistry registry;
    registry.install(canonical_identity, source);
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_EQ(
        registry.require(
            canonical_identity, 0, 1, kParticipants),
        source);
    EXPECT_THROW(
        registry.install(canonical_identity, source),
        std::logic_error);
    EXPECT_THROW(
        static_cast<void>(registry.require(
            canonical_identity, 1, 0, kParticipants)),
        std::logic_error);

    auto *const source_channel = dynamic_cast<
        IMoEOverlayMappedActivationTransport *>(source.get());
    auto *const target_channel = dynamic_cast<
        IMoEOverlayMappedActivationTransport *>(target.get());
    auto *const source_mapping = dynamic_cast<
        MoEOverlayNodeLocalRankBatchTransport *>(source.get());
    auto *const target_mapping = dynamic_cast<
        MoEOverlayNodeLocalRankBatchTransport *>(target.get());
    ASSERT_NE(source_channel, nullptr);
    ASSERT_NE(target_channel, nullptr);
    ASSERT_NE(source_mapping, nullptr);
    ASSERT_NE(target_mapping, nullptr);
    ASSERT_EQ(source_channel->activationGraphFamilyCount(), 2u);
    ASSERT_EQ(target_channel->activationGraphFamilyCount(), 2u);
    EXPECT_EQ(source_channel->activationStageOrdinal(0u, 0), 0u);
    EXPECT_EQ(source_channel->activationStageOrdinal(0u, 3), 3u);
    EXPECT_EQ(source_channel->activationStageOrdinal(1u, 4), 0u);
    EXPECT_EQ(target_channel->activationStageOrdinal(0u, 2), 2u);
    EXPECT_THROW(
        source_channel->activationStageOrdinal(1u, 3),
        std::out_of_range);
    EXPECT_THROW(
        source_channel->activationStageOrdinal(2u, 0),
        std::out_of_range);

    auto &source_main_control =
        source_channel->activationEpochControl(kParticipants[0], 0u);
    auto &target_main_control =
        target_channel->activationEpochControl(kParticipants[0], 0u);
    EXPECT_EQ(
        source_main_control.channel.channel_nonce,
        target_main_control.channel.channel_nonce);
    EXPECT_EQ(
        source_main_control.channel.stage_manifest_digest,
        target_main_control.channel.stage_manifest_digest);
    EXPECT_EQ(source_main_control.channel.source_participant_id, 2);
    EXPECT_EQ(source_main_control.channel.target_participant_id, 4);
    EXPECT_EQ(source_main_control.channel.source_tier_priority, 0);
    EXPECT_EQ(source_main_control.channel.target_tier_priority, 10);
    EXPECT_EQ(source_main_control.channel.stage_count, 4u);
    EXPECT_EQ(
        source_main_control.channel.graph_role_mask,
        roleBit(MoEOverlayInferenceGraphRole::MainPrefill) |
            roleBit(MoEOverlayInferenceGraphRole::MainDecode) |
            roleBit(MoEOverlayInferenceGraphRole::MTPGroupedVerifier));

    const auto source_main_scheduler =
        source_channel->activationEpochConfig(kParticipants[0], 0u);
    const auto target_main_scheduler =
        target_channel->activationEpochConfig(kParticipants[0], 0u);
    EXPECT_EQ(source_main_scheduler, target_main_scheduler);
    EXPECT_TRUE(source_main_scheduler.valid());
    EXPECT_EQ(
        source_main_scheduler.channel_nonce,
        source_main_control.channel.channel_nonce);
    EXPECT_EQ(source_main_scheduler.source.participant_id, 2);
    EXPECT_EQ(
        source_main_scheduler.target.participant_id,
        kParticipants[0]);
    EXPECT_EQ(source_main_scheduler.lane_ordinal, 0u);
    EXPECT_EQ(
        source_main_scheduler.model_layer_indices,
        (std::vector<std::int32_t>{0, 1, 2, 3}));
    EXPECT_THROW(
        (void)source_channel->activationEpochConfig(99, 0u),
        std::out_of_range);
    EXPECT_THROW(
        (void)source_channel->activationEpochConfig(
            kParticipants[0], 2u),
        std::out_of_range);

    auto &source_mtp_control =
        source_channel->activationEpochControl(kParticipants[1], 1u);
    EXPECT_EQ(source_mtp_control.channel.target_participant_id, 7);
    EXPECT_EQ(source_mtp_control.channel.stage_count, 1u);
    EXPECT_EQ(source_mtp_control.channel.lane_ordinal, 3u);
    const auto source_mtp_scheduler =
        source_channel->activationEpochConfig(kParticipants[1], 1u);
    EXPECT_EQ(source_mtp_scheduler.lane_ordinal, 3u);
    EXPECT_EQ(
        source_mtp_scheduler.graph_role_mask,
        roleBit(MoEOverlayInferenceGraphRole::MTPDraft));
    EXPECT_EQ(
        source_mtp_scheduler.model_layer_indices,
        (std::vector<std::int32_t>{4}));
    EXPECT_EQ(
        source_mapping->mappedDeviceAlias(
            DeviceId::cpu(),
            &source_mtp_control,
            sizeof(source_mtp_control)),
        &source_mtp_control);
    EXPECT_EQ(
        target_mapping->mappedDeviceAlias(
            DeviceId::cpu(),
            &target_main_control,
            sizeof(target_main_control)),
        &target_main_control);

    const auto source_device_lane = source_channel->activationDeviceLane(
        kParticipants[0], 0u, DeviceId::cpu());
    const auto target_device_lane = target_channel->activationDeviceLane(
        kParticipants[0], 0u, DeviceId::cpu());
    const auto source_second_device_lane =
        source_channel->activationDeviceLane(
            kParticipants[1], 0u, DeviceId::cpu());
    const auto target_second_device_lane =
        target_channel->activationDeviceLane(
            kParticipants[1], 0u, DeviceId::cpu());
    ASSERT_TRUE(source_device_lane.valid());
    ASSERT_TRUE(target_device_lane.valid());
    ASSERT_TRUE(source_second_device_lane.valid());
    ASSERT_TRUE(target_second_device_lane.valid());
    EXPECT_EQ(source_device_lane.control_device, &source_main_control);
    EXPECT_EQ(target_device_lane.control_device, &target_main_control);
    EXPECT_EQ(
        source_device_lane.dispatch.row_ids,
        source->sharedDispatchRows(kParticipants[0]).row_ids_host);
    EXPECT_EQ(
        target_device_lane.returned.canonical_route_contributions_fp32,
        target_second_device_lane.returned.
            canonical_route_contributions_fp32);
    EXPECT_EQ(
        source_device_lane.returned.canonical_route_contributions_fp32,
        source_second_device_lane.returned.
            canonical_route_contributions_fp32);
    EXPECT_EQ(
        target_device_lane.returned.route_slot_capacity,
        kEntriesPerParticipant);
    EXPECT_NE(
        target_device_lane.returned.canonical_route_contributions_fp32,
        target->sharedReturnRows(kParticipants[0]).output_rows_fp32);
    /* Participant-local compact packet matrices remain disjoint for the CPU
     * rank-batch codec, while activation graphs bind one physical hidden page
     * and one canonical route-return matrix per rank-pair mapping. This is the
     * accounting invariant that removes N duplicate bulk publications without
     * conflating compact codec storage with device route-slot authority. */
    EXPECT_NE(
        source_device_lane.dispatch.hidden_rows_fp32,
        source_second_device_lane.dispatch.hidden_rows_fp32);
    EXPECT_EQ(
        source_device_lane.shared_dispatch_hidden_rows_fp32,
        source_second_device_lane.shared_dispatch_hidden_rows_fp32);
    EXPECT_EQ(
        source_device_lane.shared_dispatch_hidden_offset,
        source_second_device_lane.shared_dispatch_hidden_offset);
    EXPECT_NE(
        target_device_lane.dispatch.hidden_rows_fp32,
        target_second_device_lane.dispatch.hidden_rows_fp32);
    EXPECT_EQ(
        target_device_lane.shared_dispatch_hidden_rows_fp32,
        target_second_device_lane.shared_dispatch_hidden_rows_fp32);
    EXPECT_EQ(
        target_device_lane.shared_dispatch_hidden_offset,
        target_second_device_lane.shared_dispatch_hidden_offset);
    EXPECT_EQ(
        source_device_lane.shared_dispatch_hidden_offset,
        target_device_lane.shared_dispatch_hidden_offset);
    EXPECT_EQ(
        target_device_lane.return_output_offset,
        target_second_device_lane.return_output_offset);
    EXPECT_EQ(
        source_device_lane.return_output_offset,
        target_device_lane.return_output_offset);
    EXPECT_EQ(
        source_device_lane.admission_signal_offset,
        source_mapping->mappedOffset(
            &source_main_control.admission.ready_signal,
            sizeof(std::uint64_t)));
    EXPECT_EQ(
        source_device_lane.dispatch_signal_offsets[0],
        source_mapping->mappedOffset(
            &source_main_control.buffers[0].dispatch_signal.value,
            sizeof(std::uint64_t)));
    EXPECT_EQ(
        target_device_lane.return_signal_offsets[1],
        target_mapping->mappedOffset(
            &target_main_control.buffers[1].return_signal.value,
            sizeof(std::uint64_t)));
    EXPECT_THROW(
        (void)source_channel->activationDeviceLane(
            kParticipants[0], 0u, DeviceId::cuda(0)),
        std::invalid_argument);

    std::array<MoEOverlaySparseRows, 2> source_dispatch{
        source->sharedDispatchRows(kParticipants[0]),
        source->sharedDispatchRows(kParticipants[1])};
    std::array<MoEOverlaySparseRows, 2> target_dispatch{
        target->sharedDispatchRows(kParticipants[0]),
        target->sharedDispatchRows(kParticipants[1])};
    populateDispatch(source_dispatch[0], 2, 10, 2, 3, 100.0f);
    populateDispatch(source_dispatch[1], 2, 20, 1, 2, 200.0f);

    std::array<const MoEOverlaySparseRows *, 2> dispatch_out{
        &source_dispatch[0], &source_dispatch[1]};
    std::array<MoEOverlaySparseRows *, 2> dispatch_in{
        &target_dispatch[0], &target_dispatch[1]};
    const auto dispatch_key = makeMoEOverlayRankBatchKey(
        7,
        9,
        ExpertHistogramSource::DecodeToken,
        3,
        1,
        2,
        0,
        1,
        MoEOverlayCollectiveDirection::Dispatch);

    const auto dispatch_publish =
        source->exchangeDispatch(dispatch_key, dispatch_out, {});
    ASSERT_TRUE(dispatch_publish.ok) << dispatch_publish.error;
    const auto dispatch_consume =
        target->exchangeDispatch(dispatch_key, {}, dispatch_in);
    ASSERT_TRUE(dispatch_consume.ok) << dispatch_consume.error;

    EXPECT_EQ(target_dispatch[0].source_participant, 2);
    EXPECT_EQ(target_dispatch[0].target_participant, kParticipants[0]);
    EXPECT_EQ(target_dispatch[0].live_row_count, 2u);
    EXPECT_EQ(target_dispatch[0].live_entry_count, 3u);
    EXPECT_EQ(target_dispatch[0].row_ids_host[1], 11);
    EXPECT_FLOAT_EQ(target_dispatch[0].hidden_rows_fp32[7], 107.0f);
    EXPECT_FLOAT_EQ(target_dispatch[1].route_weights_host[1], 200.125f);

    std::array<MoEOverlayReturnRows, 2> target_return{
        target->sharedReturnRows(kParticipants[0]),
        target->sharedReturnRows(kParticipants[1])};
    std::array<MoEOverlayReturnRows, 2> source_return{
        source->sharedReturnRows(kParticipants[0]),
        source->sharedReturnRows(kParticipants[1])};
    populateReturn(target_return[0], 2, 10, 2, 300.0f);
    populateReturn(target_return[1], 2, 20, 1, 400.0f);

    std::array<const MoEOverlayReturnRows *, 2> return_out{
        &target_return[0], &target_return[1]};
    std::array<MoEOverlayReturnRows *, 2> return_in{
        &source_return[0], &source_return[1]};
    const auto return_key = makeMoEOverlayRankBatchKey(
        7,
        9,
        ExpertHistogramSource::DecodeToken,
        3,
        1,
        2,
        0,
        1,
        MoEOverlayCollectiveDirection::ReturnReduce);

    const auto return_publish =
        target->exchangeReturn(return_key, return_out, {});
    ASSERT_TRUE(return_publish.ok) << return_publish.error;
    const auto return_consume =
        source->exchangeReturn(return_key, {}, return_in);
    ASSERT_TRUE(return_consume.ok) << return_consume.error;

    EXPECT_EQ(source_return[0].source_participant, kParticipants[0]);
    EXPECT_EQ(source_return[0].target_participant, 2);
    EXPECT_EQ(source_return[0].live_row_count, 2u);
    EXPECT_EQ(source_return[0].row_ids_host[1], 11);
    EXPECT_FLOAT_EQ(source_return[0].output_rows_fp32[7], 307.0f);
    EXPECT_FLOAT_EQ(source_return[1].output_rows_fp32[3], 403.0f);

    /* Replaying the exact authenticated identity is a stale transaction. */
    const auto stale =
        source->exchangeDispatch(dispatch_key, dispatch_out, {});
    EXPECT_FALSE(stale.ok);
    EXPECT_NE(stale.error.find("stale"), std::string::npos);

    EXPECT_EQ(source_context->total_p2p_calls(), 0u);
    EXPECT_EQ(target_context->total_p2p_calls(), 0u);
    EXPECT_EQ(source_context->total_collective_calls(), 0u);
    EXPECT_EQ(target_context->total_collective_calls(), 0u);
}

TEST(Test__MoEOverlayNodeLocalRankBatchTransport,
     CPUFollowerRunsAuthenticatedMappedEpochWithoutMediatingGPULanes)
{
    auto source_context = makeContext(/*rank=*/0, /*ranks_per_node=*/2);
    auto target_context = makeContext(/*rank=*/1, /*ranks_per_node=*/2);
    const std::string identity = uniqueIdentity("cpu_follower_epoch");
    std::shared_ptr<IMoEOverlayRankBatchTransport> source;
    std::shared_ptr<IMoEOverlayRankBatchTransport> target;
    std::exception_ptr source_error;
    std::exception_ptr target_error;
    std::thread source_builder([&]
                               {
                                   try
                                   {
                                       source = createMoEOverlayRankBatchTransport(
                                           makeConfig(source_context, identity));
                                   }
                                   catch (...)
                                   {
                                       source_error = std::current_exception();
                                   }
                               });
    std::thread target_builder([&]
                               {
                                   try
                                   {
                                       target = createMoEOverlayRankBatchTransport(
                                           makeConfig(target_context, identity));
                                   }
                                   catch (...)
                                   {
                                       target_error = std::current_exception();
                                   }
                               });
    source_builder.join();
    target_builder.join();
    if (source_error)
        std::rethrow_exception(source_error);
    if (target_error)
        std::rethrow_exception(target_error);
    auto *const source_channel = dynamic_cast<
        IMoEOverlayMappedActivationTransport *>(source.get());
    auto *const target_channel = dynamic_cast<
        IMoEOverlayMappedActivationTransport *>(target.get());
    ASSERT_NE(source_channel, nullptr);
    ASSERT_NE(target_channel, nullptr);

    constexpr int participant = 4;
    constexpr size_t family = 0u;
    const auto source_lane = source_channel->activationDeviceLane(
        participant, family, DeviceId::cpu());
    const auto target_lane = target_channel->activationDeviceLane(
        participant, family, DeviceId::cpu());
    ASSERT_TRUE(source_lane.valid());
    ASSERT_TRUE(target_lane.valid());

    /* Payload selection is one geometry-bound authority for both endpoints.
     * A decode view names the lane-local compact matrix. A wider prefill view
     * names the rank-pair physical matrix and resolves compact rows through
     * their non-contiguous row ids. */
    const auto direct_source = source_lane.dispatchPayload(1);
    const auto direct_target = target_lane.hostDispatchPayload(1);
    ASSERT_TRUE(direct_source.valid());
    EXPECT_TRUE(direct_source.selection.usesCompactRows());
    EXPECT_EQ(
        direct_source.packet.hidden_rows_fp32,
        source_lane.dispatch.hidden_rows_fp32);
    EXPECT_EQ(
        direct_target.hidden_payload_layout,
        MoEOverlayActivationHiddenPayloadLayout::CompactRows);
    EXPECT_EQ(direct_target.hidden_row_capacity, 1u);

    const auto shared_source = source_lane.dispatchPayload(3);
    ASSERT_TRUE(shared_source.valid());
    EXPECT_TRUE(shared_source.requiresBulkPublication());
    EXPECT_EQ(
        shared_source.packet.hidden_rows_fp32,
        source_lane.shared_dispatch_hidden_rows_fp32);
    EXPECT_NE(
        shared_source.packet.hidden_rows_fp32,
        source_lane.dispatch.hidden_rows_fp32);

    const auto scheduler_config =
        target_channel->activationEpochConfig(participant, family);
    MoEOverlayActivationEpochProtocol source_protocol(
        source_channel->activationEpochControl(participant, family),
        source_channel->activationEpochConfig(participant, family));
    MoEOverlayActivationEpochProtocol target_protocol(
        target_channel->activationEpochControl(participant, family),
        scheduler_config);

    const MoEOverlayInferenceTopologyIdentity topology{
        .workspace_generation = scheduler_config.workspace_generation,
        .topology_fingerprint_low =
            scheduler_config.topology_fingerprint_low,
        .topology_fingerprint_high =
            scheduler_config.topology_fingerprint_high,
        .source_world_rank = scheduler_config.source.world_rank,
        .target_world_rank = scheduler_config.target.world_rank,
    };
    const MoEOverlayInferenceCommandIdentity command{
        .request_generation = 9u,
        .command_id = 11u,
        .initial_placement_epoch = 44u,
    };
    const auto ticket = makeMoEOverlayInferenceExecutionTicket(
        topology,
        command,
        /*transaction_ordinal=*/1u,
        /*logical_step_id=*/5u,
        /*placement_epoch=*/44u,
        MoEOverlayInferenceGraphRole::MainPrefill,
        /*request_count=*/1,
        /*logical_rows_per_request=*/1,
        /*physical_rows_per_request=*/3);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(1);
    const auto deadline_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            deadline.time_since_epoch())
            .count());

    std::string error;
    const auto epoch = target_protocol.arm(
        ticket,
        /*epoch_generation=*/1u,
        deadline_ns,
        &error);
    ASSERT_TRUE(epoch.has_value()) << error;
    ASSERT_TRUE(source_protocol.activate(
        MoEOverlayActivationEndpoint::Continuation,
        *epoch,
        &error))
        << error;
    ASSERT_TRUE(target_protocol.activate(
        MoEOverlayActivationEndpoint::Follower,
        *epoch,
        &error))
        << error;

    auto source_rows = source->sharedDispatchRows(participant);
    auto target_rows = target->sharedDispatchRows(participant);
    auto target_return = target->sharedReturnRows(participant);
    auto source_return = source->sharedReturnRows(participant);
    for (std::uint32_t stage = 0u; stage < 4u; ++stage)
    {
        const float base = 100.0f + static_cast<float>(stage) * 10.0f;
        populateDispatch(
            source_rows,
            scheduler_config.source.participant_id,
            /*first_row=*/0,
            /*live_rows=*/2u,
            /*live_entries=*/3u,
            base);
        /* The compact matrix is stable mapped storage but is not authoritative
         * for this three-row transaction. Poison it, publish non-contiguous
         * row ids, and populate only the selected shared physical matrix. */
        std::fill_n(
            source_rows.hidden_rows_fp32,
            source_rows.row_capacity * static_cast<size_t>(kDModel),
            -1000.0f - base);
        source_rows.row_ids_host[0] = 0;
        source_rows.row_ids_host[1] = 2;
        for (size_t physical_row = 0u;
             physical_row < 3u;
             ++physical_row)
        {
            for (int column = 0; column < kDModel; ++column)
            {
                shared_source.packet.hidden_rows_fp32[
                    physical_row * static_cast<size_t>(kDModel) +
                    static_cast<size_t>(column)] =
                    base + static_cast<float>(physical_row * 100u) +
                    static_cast<float>(column);
            }
        }
        ASSERT_TRUE(source_protocol.publishDispatch(
            *epoch,
            stage,
            source_rows.live_row_count,
            source_rows.live_entry_count,
            compactMoEOverlayDispatchBytes(source_rows),
            &error))
            << error;

        const auto dispatch = target_protocol.consumeDispatch(
            *epoch, stage, &error);
        ASSERT_TRUE(dispatch.has_value()) << error;
        EXPECT_EQ(dispatch->model_layer_index, static_cast<int>(stage));
        EXPECT_EQ(dispatch->live_rows, 2u);
        EXPECT_EQ(target_rows.row_ids_host[1], 2);

        auto selected_target = target_lane.hostDispatchPayload(3);
        selected_target.live_row_count =
            static_cast<size_t>(dispatch->live_rows);
        selected_target.live_entry_count =
            static_cast<size_t>(dispatch->live_entries);
        ASSERT_EQ(
            selected_target.hidden_payload_layout,
            MoEOverlayActivationHiddenPayloadLayout::SharedPhysicalRows);
        const float *const first_hidden =
            selected_target.hiddenRowForCompactIndex(0u);
        const float *const second_hidden =
            selected_target.hiddenRowForCompactIndex(1u);
        ASSERT_NE(first_hidden, nullptr);
        ASSERT_NE(second_hidden, nullptr);
        EXPECT_FLOAT_EQ(first_hidden[3], base + 3.0f);
        EXPECT_FLOAT_EQ(second_hidden[3], base + 203.0f);
        EXPECT_LT(target_rows.hidden_rows_fp32[kDModel + 3], -1000.0f);

        populateReturn(
            target_return,
            scheduler_config.source.participant_id,
            /*first_row=*/0,
            /*live_rows=*/2u,
            base * 2.0f);
        ASSERT_TRUE(target_protocol.publishReturn(
            *epoch,
            stage,
            compactMoEOverlayReturnBytes(target_return),
            &error))
            << error;
        const auto returned = source_protocol.consumeReturn(
            *epoch, stage, &error);
        ASSERT_TRUE(returned.has_value()) << error;
        EXPECT_EQ(returned->live_rows, 2u);
        EXPECT_EQ(source_return.row_ids_host[1], 1);
        EXPECT_FLOAT_EQ(
            source_return.output_rows_fp32[kDModel + 3],
            base * 2.0f + 7.0f);
    }

    ASSERT_TRUE(target_protocol.complete(
        MoEOverlayActivationEndpoint::Follower,
        *epoch,
        &error))
        << error;
    ASSERT_TRUE(source_protocol.complete(
        MoEOverlayActivationEndpoint::Continuation,
        *epoch,
        &error))
        << error;
    ASSERT_TRUE(target_protocol.reset(*epoch, &error)) << error;

    EXPECT_EQ(source_context->total_p2p_calls(), 0u);
    EXPECT_EQ(target_context->total_p2p_calls(), 0u);
    EXPECT_EQ(source_context->total_collective_calls(), 0u);
    EXPECT_EQ(target_context->total_collective_calls(), 0u);
}
