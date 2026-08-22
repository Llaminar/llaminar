/**
 * @file Test__MoEOverlayCollectiveWorkspace.cpp
 * @brief CPU protocol tests for fixed-capacity ExpertOverlay transport.
 *
 * These tests lock down immutable ticket identity, sparse collective keys and
 * participant-local transport adaptation without requiring an accelerator.
 */

#include "execution/moe/MoEOverlaySparseCollective.h"
#include "execution/moe/MoEOverlayRankBatchTransport.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "collective/ITPContext.h"
#include "kernels/cpu/moe/CPUMoEKernel.h"
#include "tensors/Tensors.h"
#include "mocks/MockComputeStage.h"
#include "mocks/MockMPIContext.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    MoEOverlayCollectiveKey dispatchKey(uint64_t sequence)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = 1;
        key.step_id = 2;
        key.layer_idx = 3;
        key.tier_idx = 1;
        key.domain_id = 7;
        key.direction = MoEOverlayCollectiveDirection::Dispatch;
        key.sequence = sequence;
        return key;
    }

    MoEOverlayCollectiveKey returnKey(uint64_t sequence)
    {
        auto key = dispatchKey(sequence);
        key.direction = MoEOverlayCollectiveDirection::ReturnReduce;
        return key;
    }

    class CountingTPContext final : public ITPContext
    {
    public:
        TPScope scope() const override { return TPScope::RANK_LOCAL; }
        int degree() const override { return 2; }
        int myIndex() const override { return 0; }
        CollectiveBackendType backend() const override { return CollectiveBackendType::HOST; }
        bool allreduce(TensorBase *tensor) override
        {
            (void)tensor;
            return true;
        }
        bool broadcast(TensorBase *tensor, int source_index = 0) override
        {
            (void)tensor;
            last_source_index = source_index;
            ++broadcast_calls;
            return true;
        }
        bool allgather(const TensorBase *local_shard, TensorBase *global_tensor) override
        {
            (void)local_shard;
            (void)global_tensor;
            return true;
        }

        int broadcast_calls = 0;
        int last_source_index = -1;
    };

    bool hasInputBinding(const StageBufferContract &contract, BufferId id)
    {
        return std::any_of(contract.inputs.begin(), contract.inputs.end(),
                           [id](const BufferBinding &binding)
                           {
                               return binding.id == id && binding.access == BufferAccess::READ;
                           });
    }

    std::shared_ptr<MoEOverlayResidencyAuthority> staticResidencyAuthority()
    {
        RoutedExpertDomain hot_domain;
        hot_domain.name = "gpu_hot";
        hot_domain.scope = ExecutionDomainScope::SINGLE;
        hot_domain.backend = CollectiveBackendType::NCCL;
        hot_domain.participants = {GlobalDeviceAddress::cuda(0, 0)};
        hot_domain.world_ranks = {0};
        hot_domain.owner_rank = 0;

        RoutedExpertDomain cold_domain;
        cold_domain.name = "cpu_cold";
        cold_domain.scope = ExecutionDomainScope::SINGLE;
        cold_domain.backend = CollectiveBackendType::MPI;
        cold_domain.participants = {GlobalDeviceAddress::cpu(1)};
        cold_domain.world_ranks = {1};
        cold_domain.owner_rank = 1;

        RoutedExpertTier hot_tier;
        hot_tier.name = "hot";
        hot_tier.domain = "gpu_hot";
        hot_tier.priority = 0;
        hot_tier.max_experts_per_layer = 1;

        RoutedExpertTier cold_tier;
        cold_tier.name = "cold";
        cold_tier.domain = "cpu_cold";
        cold_tier.priority = 1;
        cold_tier.fallback = true;

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "gpu_hot";
        plan.shared_expert_domain = "gpu_hot";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.domains = {std::move(hot_domain), std::move(cold_domain)};
        plan.routed_tiers = {std::move(hot_tier), std::move(cold_tier)};

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = 4;
        metadata.num_experts = 2;
        metadata.d_model = 4;
        metadata.routed_intermediate_size = 2;
        metadata.routed_quant_type = "F32";
        return std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = std::move(plan),
                .model_metadata = std::move(metadata),
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "CPU",
            });
    }

} // namespace

TEST(Test__MoEOverlayCollectiveWorkspace,
     RankBatchTransportCannotRegressToBlockingSendWaits)
{
    std::ifstream input(LLAMINAR_MOE_OVERLAY_RANK_BATCH_TRANSPORT_SOURCE);
    ASSERT_TRUE(input.good());
    const std::string source{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};

    EXPECT_EQ(source.find("config_.mpi_ctx->wait("), std::string::npos);
    EXPECT_EQ(source.find("MPI_Wait("), std::string::npos);
    EXPECT_NE(source.find("config_.mpi_ctx->test("), std::string::npos);
    EXPECT_NE(
        source.find("rank_batch_async_send_submissions"),
        std::string::npos);

    const size_t preposted_return = source.find(
        "pending_return_receive_ = config_.mpi_ctx->irecv(");
    ASSERT_NE(preposted_return, std::string::npos);
    const size_t dispatch_submission = source.find(
        "slot->request = config_.mpi_ctx->isend(", preposted_return);
    ASSERT_NE(dispatch_submission, std::string::npos);
    EXPECT_LT(preposted_return, dispatch_submission)
        << "The return receive must be visible before dispatch publication";
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     TransactionControlChannelCannotRegressToBlockingMPIWaits)
{
    std::ifstream input(LLAMINAR_MOE_OVERLAY_TRANSACTION_SERVICE_SOURCE);
    ASSERT_TRUE(input.good());
    const std::string source{
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>()};

    EXPECT_EQ(source.find("config_.mpi_ctx->wait("), std::string::npos);
    EXPECT_EQ(source.find("MPI_Wait("), std::string::npos);
    EXPECT_NE(source.find("config_.mpi_ctx->test("), std::string::npos);
    EXPECT_NE(source.find("control_tickets"), std::string::npos);
}

TEST(Test__MoEOverlayCollectiveWorkspace, EnsureCapacityAndResetReuseStoragePointers)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 8, 2, DeviceId::cpu());

    auto before = workspace.dispatchReceive(5, 1);
    const int32_t *row_ids_ptr = before.row_ids_host;
    const int32_t *entry_offsets_ptr = before.entry_offsets_host;
    const int32_t *expert_ids_ptr = before.expert_ids_host;
    const float *route_weights_ptr = before.route_weights_host;
    const float *hidden_ptr = before.hidden_rows_fp32;
    const size_t row_capacity = before.row_capacity;
    const size_t entry_capacity = before.entry_capacity;

    workspace.resetForStep(4, 9);

    auto after = workspace.dispatchReceive(5, 1);
    EXPECT_EQ(after.live_row_count, 0u);
    EXPECT_EQ(after.live_entry_count, 0u);
    EXPECT_EQ(after.row_capacity, row_capacity);
    EXPECT_EQ(after.entry_capacity, entry_capacity);
    EXPECT_EQ(after.row_ids_host, row_ids_ptr);
    EXPECT_EQ(after.entry_offsets_host, entry_offsets_ptr);
    EXPECT_EQ(after.expert_ids_host, expert_ids_ptr);
    EXPECT_EQ(after.route_weights_host, route_weights_ptr);
    EXPECT_EQ(after.hidden_rows_fp32, hidden_ptr);
    EXPECT_EQ(after.entry_offsets_host[0], 0);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     FixedSerialGraphFamilyAliasesKeysButRejectsCapacityDrift)
{
    MoEOverlayCollectiveWorkspace workspace(
        MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
            .max_rows = 16,
            .max_entries = 32,
            .d_model = 8,
            .top_k = 2,
            .device = DeviceId::cpu(),
            .reuse_policy = MoEOverlayCollectiveWorkspace::
                StorageReusePolicy::SerialGraphFamily,
        });

    const auto first_dispatch = workspace.dispatchReceive(0, 0);
    const auto distant_dispatch = workspace.dispatchReceive(39, 2);
    const auto first_output = workspace.localExpertOutput(0, 0);
    const auto distant_output = workspace.localExpertOutput(39, 2);

    EXPECT_TRUE(workspace.hasFixedCapacity());
    EXPECT_EQ(
        workspace.storageReusePolicy(),
        MoEOverlayCollectiveWorkspace::StorageReusePolicy::SerialGraphFamily);
    EXPECT_EQ(first_dispatch.row_ids_host, distant_dispatch.row_ids_host);
    EXPECT_EQ(first_dispatch.hidden_rows_fp32, distant_dispatch.hidden_rows_fp32);
    EXPECT_EQ(first_output.row_ids_host, distant_output.row_ids_host);
    EXPECT_EQ(first_output.output_rows_fp32, distant_output.output_rows_fp32);

    EXPECT_NO_THROW(workspace.ensureCapacity(
        16, 32, 8, 2, DeviceId::cpu()));
    EXPECT_THROW(
        workspace.ensureCapacity(17, 34, 8, 2, DeviceId::cpu()),
        std::logic_error);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     FixedDistinctGraphFamilyRetainsIndependentProtocolKeys)
{
    MoEOverlayCollectiveWorkspace workspace(
        MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
            .max_rows = 4,
            .max_entries = 8,
            .d_model = 8,
            .top_k = 2,
            .device = DeviceId::cpu(),
            .reuse_policy = MoEOverlayCollectiveWorkspace::
                StorageReusePolicy::DistinctLayerTier,
        });

    const auto first = workspace.dispatchReceive(0, 0);
    const auto second = workspace.dispatchReceive(1, 0);
    EXPECT_NE(first.row_ids_host, second.row_ids_host);
    EXPECT_NE(first.hidden_rows_fp32, second.hidden_rows_fp32);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     MPIRankMembershipOwnsSeveralHeterogeneousParticipantIds)
{
    auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 2);
    MoEOverlayMPISparseCollectiveContext collective(
        MoEOverlayMPISparseCollectiveContext::Config{
            .mpi_ctx = mpi,
            .local_participant_ids = {7, 2, 11},
        });

    EXPECT_EQ(collective.localParticipantIds(),
              (std::vector<int>{2, 7, 11}));
    EXPECT_TRUE(collective.ownsLocalParticipant(2));
    EXPECT_TRUE(collective.ownsLocalParticipant(7));
    EXPECT_TRUE(collective.ownsLocalParticipant(11));
    EXPECT_FALSE(collective.ownsLocalParticipant(0));
    EXPECT_FALSE(collective.ownsLocalParticipant(10));
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     MPIRankMembershipAllowsRelayButRejectsAmbiguousIdentity)
{
    auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(1, 2);
    MoEOverlayMPISparseCollectiveContext relay(
        MoEOverlayMPISparseCollectiveContext::Config{
            .mpi_ctx = mpi,
            .local_participant_ids = {},
        });
    EXPECT_TRUE(relay.localParticipantIds().empty());

    EXPECT_THROW(
        MoEOverlayMPISparseCollectiveContext(
            MoEOverlayMPISparseCollectiveContext::Config{
                .mpi_ctx = nullptr,
                .local_participant_ids = {},
            }),
        std::invalid_argument);
    EXPECT_THROW(
        MoEOverlayMPISparseCollectiveContext(
            MoEOverlayMPISparseCollectiveContext::Config{
                .mpi_ctx = mpi,
                .local_participant_ids = {3, 3},
            }),
        std::invalid_argument);
    EXPECT_THROW(
        MoEOverlayMPISparseCollectiveContext(
            MoEOverlayMPISparseCollectiveContext::Config{
                .mpi_ctx = mpi,
                .local_participant_ids = {-1},
            }),
        std::invalid_argument);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     RankBatchCodecPreservesCanonicalParticipantsEpochsAndExactRows)
{
    constexpr int kDModel = 4;
    constexpr int kTopK = 2;
    constexpr int kLayer = 3;
    constexpr int kTier = 1;

    MoEOverlayRankBatchWireWorkspace wire(
        MoEOverlayRankBatchWireWorkspace::Config{
            /* Deliberately adversarial input order: construction normalizes it. */
            .participant_ids = {11, 7},
            .max_total_rows = 4,
            .max_total_entries = 8,
            .d_model = kDModel,
            .top_k = kTopK,
        });
    ASSERT_EQ(wire.participantIds(), (std::vector<int>{7, 11}));
    const std::byte *const send_address = wire.sendBufferData();
    const std::byte *const receive_address = wire.receiveBufferData();

    std::array<MoEOverlayCollectiveWorkspace, 2> participant_storage;
    for (auto &workspace : participant_storage)
        workspace.ensureCapacity(4, 8, kDModel, kTopK, DeviceId::cpu());

    std::array<MoEOverlaySparseRows, 2> outbound{
        participant_storage[0].localExpertInput(kLayer, kTier),
        participant_storage[1].localExpertInput(kLayer, kTier),
    };
    for (size_t index = 0; index < outbound.size(); ++index)
    {
        outbound[index].source_participant = 2;
        outbound[index].target_participant = wire.participantIds()[index];
        outbound[index].residency_epoch = 41u + index;
    }

    outbound[0].live_row_count = 1;
    outbound[0].live_entry_count = 2;
    outbound[0].row_ids_host[0] = 5;
    outbound[0].entry_offsets_host[0] = 0;
    outbound[0].entry_offsets_host[1] = 2;
    outbound[0].expert_ids_host[0] = 3;
    outbound[0].expert_ids_host[1] = 9;
    outbound[0].route_weights_host[0] = 0.75f;
    outbound[0].route_weights_host[1] = 0.25f;
    for (int column = 0; column < kDModel; ++column)
        outbound[0].hidden_rows_fp32[column] = 10.0f + column;

    outbound[1].live_row_count = 2;
    outbound[1].live_entry_count = 3;
    outbound[1].row_ids_host[0] = 1;
    outbound[1].row_ids_host[1] = 8;
    outbound[1].entry_offsets_host[0] = 0;
    outbound[1].entry_offsets_host[1] = 1;
    outbound[1].entry_offsets_host[2] = 3;
    outbound[1].expert_ids_host[0] = 4;
    outbound[1].expert_ids_host[1] = 6;
    outbound[1].expert_ids_host[2] = 12;
    outbound[1].route_weights_host[0] = 0.5f;
    outbound[1].route_weights_host[1] = 0.125f;
    outbound[1].route_weights_host[2] = 0.375f;
    for (int element = 0; element < 2 * kDModel; ++element)
        outbound[1].hidden_rows_fp32[element] = 20.0f + element;

    const auto dispatch_key = makeMoEOverlayRankBatchKey(
        /*generation_id=*/17,
        /*step_id=*/23,
        ExpertHistogramSource::PrefillChunk,
        kLayer,
        kTier,
        /*domain_ordinal=*/4,
        /*source_world_rank=*/0,
        /*target_world_rank=*/1,
        MoEOverlayCollectiveDirection::Dispatch);
    ASSERT_TRUE(dispatch_key.isValid());

    std::array<const MoEOverlaySparseRows *, 2> outbound_views{
        &outbound[0], &outbound[1]};
    size_t dispatch_bytes = 0;
    std::string error;
    ASSERT_TRUE(wire.encodeDispatch(
        dispatch_key, outbound_views, &dispatch_bytes, &error))
        << error;
    ASSERT_GT(dispatch_bytes, 0u);
    ASSERT_LT(dispatch_bytes, wire.wireCapacityBytes());

    std::array<MoEOverlaySparseRows, 2> inbound{
        participant_storage[0].dispatchReceive(kLayer, kTier),
        participant_storage[1].dispatchReceive(kLayer, kTier),
    };
    std::array<MoEOverlaySparseRows *, 2> inbound_views{
        &inbound[0], &inbound[1]};
    ASSERT_TRUE(wire.decodeDispatch(
        dispatch_key,
        wire.encodedPayload(dispatch_bytes),
        inbound_views,
        &error))
        << error;

    EXPECT_EQ(inbound[0].key.histogram_source,
              ExpertHistogramSource::PrefillChunk);
    EXPECT_EQ(inbound[0].key.participant_id, 7);
    EXPECT_EQ(inbound[0].residency_epoch, 41u);
    EXPECT_EQ(inbound[0].source_participant, 2);
    EXPECT_EQ(inbound[0].target_participant, 7);
    EXPECT_EQ(inbound[0].live_row_count, 1u);
    EXPECT_EQ(inbound[0].live_entry_count, 2u);
    EXPECT_EQ(inbound[0].row_ids_host[0], 5);
    EXPECT_EQ(inbound[0].expert_ids_host[1], 9);
    EXPECT_FLOAT_EQ(inbound[0].route_weights_host[1], 0.25f);
    EXPECT_FLOAT_EQ(inbound[0].hidden_rows_fp32[3], 13.0f);

    EXPECT_EQ(inbound[1].key.participant_id, 11);
    EXPECT_EQ(inbound[1].residency_epoch, 42u);
    EXPECT_EQ(inbound[1].live_row_count, 2u);
    EXPECT_EQ(inbound[1].live_entry_count, 3u);
    EXPECT_EQ(inbound[1].row_ids_host[1], 8);
    EXPECT_EQ(inbound[1].entry_offsets_host[2], 3);
    EXPECT_EQ(inbound[1].expert_ids_host[2], 12);
    EXPECT_FLOAT_EQ(inbound[1].route_weights_host[2], 0.375f);
    EXPECT_FLOAT_EQ(inbound[1].hidden_rows_fp32[7], 27.0f);

    std::array<MoEOverlayReturnRows, 2> returned{
        participant_storage[0].localExpertOutput(kLayer, kTier),
        participant_storage[1].localExpertOutput(kLayer, kTier),
    };
    for (size_t index = 0; index < returned.size(); ++index)
    {
        returned[index].source_participant = wire.participantIds()[index];
        returned[index].target_participant = 2;
        returned[index].residency_epoch = 41u + index;
        returned[index].live_row_count = index + 1u;
        for (size_t row = 0; row < returned[index].live_row_count; ++row)
        {
            returned[index].row_ids_host[row] =
                static_cast<int32_t>(30u + index * 10u + row);
            for (int column = 0; column < kDModel; ++column)
            {
                returned[index].output_rows_fp32[
                    row * static_cast<size_t>(kDModel) + column] =
                    static_cast<float>(100u + index * 20u + row * 4u + column);
            }
        }
    }

    auto return_key = dispatch_key;
    return_key.direction = MoEOverlayCollectiveDirection::ReturnReduce;
    return_key.sequence = makeMoEOverlayRankBatchKey(
                              return_key.generation_id,
                              return_key.step_id,
                              return_key.histogram_source,
                              return_key.layer_idx,
                              return_key.tier_idx,
                              return_key.domain_ordinal,
                              return_key.source_world_rank,
                              return_key.target_world_rank,
                              return_key.direction)
                              .sequence;
    std::array<const MoEOverlayReturnRows *, 2> return_views{
        &returned[0], &returned[1]};
    size_t return_bytes = 0;
    ASSERT_TRUE(wire.encodeReturn(
        return_key, return_views, &return_bytes, &error))
        << error;

    std::array<MoEOverlayReturnRows, 2> return_inbound{
        participant_storage[0].returnReceive(kLayer, kTier),
        participant_storage[1].returnReceive(kLayer, kTier),
    };
    std::array<MoEOverlayReturnRows *, 2> return_inbound_views{
        &return_inbound[0], &return_inbound[1]};
    ASSERT_TRUE(wire.decodeReturn(
        return_key,
        wire.encodedPayload(return_bytes),
        return_inbound_views,
        &error))
        << error;
    EXPECT_EQ(return_inbound[0].source_participant, 7);
    EXPECT_EQ(return_inbound[0].residency_epoch, 41u);
    EXPECT_EQ(return_inbound[0].live_row_count, 1u);
    EXPECT_FLOAT_EQ(return_inbound[0].output_rows_fp32[3], 103.0f);
    EXPECT_EQ(return_inbound[1].source_participant, 11);
    EXPECT_EQ(return_inbound[1].residency_epoch, 42u);
    EXPECT_EQ(return_inbound[1].live_row_count, 2u);
    EXPECT_FLOAT_EQ(return_inbound[1].output_rows_fp32[7], 127.0f);

    /* Repeated successful codecs must retain their model-lifetime addresses. */
    for (int iteration = 0; iteration < 64; ++iteration)
    {
        ASSERT_TRUE(wire.encodeDispatch(
            dispatch_key, outbound_views, &dispatch_bytes, &error));
        ASSERT_TRUE(wire.decodeDispatch(
            dispatch_key,
            wire.encodedPayload(dispatch_bytes),
            inbound_views,
            &error));
    }
    EXPECT_EQ(wire.sendBufferData(), send_address);
    EXPECT_EQ(wire.receiveBufferData(), receive_address);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     RankBatchDecoderRejectsStaleIdentityBeforePublishingAnyParticipant)
{
    MoEOverlayRankBatchWireWorkspace wire(
        {.participant_ids = {3, 4},
         .max_total_rows = 2,
         .max_total_entries = 2,
         .d_model = 2,
         .top_k = 1});
    std::array<MoEOverlayCollectiveWorkspace, 2> storage;
    for (auto &workspace : storage)
        workspace.ensureCapacity(1, 1, 2, 1, DeviceId::cpu());

    std::array<MoEOverlaySparseRows, 2> outbound{
        storage[0].localExpertInput(0, 0),
        storage[1].localExpertInput(0, 0)};
    for (size_t index = 0; index < outbound.size(); ++index)
    {
        outbound[index].source_participant = 1;
        outbound[index].target_participant = 3 + static_cast<int>(index);
        outbound[index].residency_epoch = 9;
        outbound[index].live_row_count = 1;
        outbound[index].live_entry_count = 1;
        outbound[index].row_ids_host[0] = static_cast<int32_t>(index);
        outbound[index].entry_offsets_host[0] = 0;
        outbound[index].entry_offsets_host[1] = 1;
        outbound[index].expert_ids_host[0] = static_cast<int32_t>(index);
        outbound[index].route_weights_host[0] = 1.0f;
        outbound[index].hidden_rows_fp32[0] = 1.0f;
        outbound[index].hidden_rows_fp32[1] = 2.0f;
    }

    const auto encoded_key = makeMoEOverlayRankBatchKey(
        2,
        5,
        ExpertHistogramSource::DecodeToken,
        0,
        0,
        0,
        0,
        1,
        MoEOverlayCollectiveDirection::Dispatch);
    std::array<const MoEOverlaySparseRows *, 2> outbound_views{
        &outbound[0], &outbound[1]};
    size_t encoded_bytes = 0;
    std::string error;
    ASSERT_TRUE(wire.encodeDispatch(
        encoded_key, outbound_views, &encoded_bytes, &error));

    std::array<MoEOverlaySparseRows, 2> inbound{
        storage[0].dispatchReceive(0, 0),
        storage[1].dispatchReceive(0, 0)};
    inbound[0].live_row_count = 1;
    inbound[1].live_row_count = 1;
    inbound[0].row_ids_host[0] = 101;
    inbound[1].row_ids_host[0] = 202;
    std::array<MoEOverlaySparseRows *, 2> inbound_views{
        &inbound[0], &inbound[1]};

    auto stale_expected_key = encoded_key;
    ++stale_expected_key.step_id;
    EXPECT_FALSE(wire.decodeDispatch(
        stale_expected_key,
        wire.encodedPayload(encoded_bytes),
        inbound_views,
        &error));
    EXPECT_FALSE(error.empty());
    EXPECT_EQ(inbound[0].live_row_count, 1u);
    EXPECT_EQ(inbound[1].live_row_count, 1u);
    EXPECT_EQ(inbound[0].row_ids_host[0], 101);
    EXPECT_EQ(inbound[1].row_ids_host[0], 202);
}

TEST(Test__MoEOverlayCollectiveWorkspace, DispatchTicketBindsOneImmutableCapacityAndStablePointers)
{
    MoEOverlayDispatchTicketStorage storage;
    storage.bindFixedCapacity(
        /*layer_idx=*/7,
        /*bucket_rows=*/16,
        /*top_k=*/4,
        /*d_model=*/32,
        DeviceId::cpu(),
        /*workspace_generation=*/11);

    ASSERT_TRUE(storage.isBound());
    ASSERT_TRUE(storage.ticket().isValid());
    EXPECT_EQ(storage.ticket().header->layer_idx, 7);
    EXPECT_EQ(storage.ticket().header->bucket_row_capacity, 16);
    EXPECT_EQ(storage.ticket().header->route_capacity, 64);
    EXPECT_EQ(storage.ticket().header->logical_row_count, 16);
    EXPECT_EQ(storage.ticket().header->workspace_generation, 11u);

    const auto *header = storage.ticket().header;
    const auto *indices = storage.ticket().routing_indices_fp32;
    const auto *weights = storage.ticket().routing_weights_fp32;
    const auto *hidden = storage.ticket().hidden_rows_fp32;
    const auto *return_rows = storage.ticket().return_rows_fp32;
    EXPECT_LT(
        reinterpret_cast<uintptr_t>(header),
        reinterpret_cast<uintptr_t>(indices));
    EXPECT_LT(
        reinterpret_cast<uintptr_t>(indices),
        reinterpret_cast<uintptr_t>(weights));
    EXPECT_LT(
        reinterpret_cast<uintptr_t>(weights),
        reinterpret_cast<uintptr_t>(hidden));
    EXPECT_LT(
        reinterpret_cast<uintptr_t>(hidden),
        reinterpret_cast<uintptr_t>(return_rows));

    EXPECT_THROW(
        storage.bindFixedCapacity(7, 32, 4, 32, DeviceId::cpu(), 12),
        std::logic_error);
    EXPECT_EQ(storage.ticket().header, header);
    EXPECT_EQ(storage.ticket().routing_indices_fp32, indices);
    EXPECT_EQ(storage.ticket().routing_weights_fp32, weights);
    EXPECT_EQ(storage.ticket().hidden_rows_fp32, hidden);
    EXPECT_EQ(storage.ticket().return_rows_fp32, return_rows);
    EXPECT_TRUE(storage.hasValidBoundIdentity());
}

TEST(Test__MoEOverlayCollectiveWorkspace, DispatchTicketRejectsInvalidLogicalPrefixWithoutChangingCapacity)
{
    MoEOverlayDispatchTicketStorage storage;
    storage.bindFixedCapacity(2, 8, 2, 4, DeviceId::cpu(), 9);
    ASSERT_TRUE(storage.ticket().isValid());

    storage.ticket().header->logical_row_count = 0;
    EXPECT_FALSE(storage.ticket().isValid());
    storage.ticket().header->logical_row_count = 9;
    EXPECT_FALSE(storage.ticket().isValid());
    storage.ticket().header->logical_row_count = 3;
    EXPECT_TRUE(storage.ticket().isValid());
    storage.ticket().header->return_logical_row_count = 3;
    EXPECT_TRUE(storage.ticket().returnPayloadReady());
    storage.ticket().header->workspace_generation = 10;
    EXPECT_FALSE(storage.hasValidBoundIdentity());
    EXPECT_EQ(storage.ticket().header->bucket_row_capacity, 8);
    EXPECT_EQ(storage.ticket().header->route_capacity, 16);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     TransportedHiddenRowsPublishCanonicalRouterQ8Bytes)
{
    constexpr int rows = 3;
    constexpr int d_model = 65;
    constexpr int experts = 4;
    constexpr int top_k = 2;
    constexpr int blocks_per_row =
        (d_model + Q8_1Block::BLOCK_SIZE - 1) / Q8_1Block::BLOCK_SIZE;

    std::vector<float> hidden(
        static_cast<size_t>(rows) * static_cast<size_t>(d_model));
    std::vector<float> gate(
        static_cast<size_t>(experts) * static_cast<size_t>(d_model));
    for (size_t index = 0; index < hidden.size(); ++index)
        hidden[index] = static_cast<float>(static_cast<int>(index % 31u) - 15) * 0.03125f;
    for (size_t index = 0; index < gate.size(); ++index)
        gate[index] = static_cast<float>(static_cast<int>(index % 17u) - 8) * 0.015625f;

    CPUMoEKernel transported_kernel;
    ASSERT_TRUE(transported_kernel.publishTransportedRouterQ8Hidden(
        hidden.data(), rows, d_model));
    const Q8_1Block *const transported =
        transported_kernel.publishedRouterQ8Hidden(
            hidden.data(), rows, d_model);
    ASSERT_NE(transported, nullptr);

    CPUMoEKernel routed_kernel;
    MoERoutingResult routing;
    ASSERT_TRUE(routed_kernel.route(
        hidden.data(),
        gate.data(),
        rows,
        d_model,
        experts,
        top_k,
        /*normalize_weights=*/true,
        routing));
    const Q8_1Block *const routed =
        routed_kernel.publishedRouterQ8Hidden(
            hidden.data(), rows, d_model);
    ASSERT_NE(routed, nullptr);

    const size_t publication_bytes =
        static_cast<size_t>(rows) * static_cast<size_t>(blocks_per_row) *
        sizeof(Q8_1Block);
    EXPECT_EQ(std::memcmp(transported, routed, publication_bytes), 0)
        << "The heterogeneous CPU boundary must publish the exact Q8_1 bytes "
           "owned by the ordinary CPU router path.";
    EXPECT_EQ(
        transported_kernel.publishedRouterQ8Hidden(
            hidden.data() + 1, rows, d_model),
        nullptr)
        << "Publication provenance must retain the exact transported FP32 base.";
    EXPECT_EQ(
        transported_kernel.publishedRouterQ8Hidden(
            hidden.data(), rows, d_model - 1),
        nullptr)
        << "Publication provenance must reject a mismatched hidden width.";
}

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeysDoNotAliasMainGraphKeys)
{
    const auto main_key = makeMoEOverlayCollectiveKey(
        5,
        8,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto mtp_key = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);

    EXPECT_TRUE(main_key.isValid());
    EXPECT_TRUE(mtp_key.isValid());
    EXPECT_EQ(main_key.key_namespace, MoEOverlayCollectiveNamespace::Main);
    EXPECT_EQ(mtp_key.key_namespace, MoEOverlayCollectiveNamespace::MTP);
    EXPECT_NE(main_key, mtp_key);
    EXPECT_NE(main_key.sequence, mtp_key.sequence);
    EXPECT_NE(main_key.toString(), mtp_key.toString());
    EXPECT_STREQ(toString(main_key.key_namespace), "Main");
    EXPECT_STREQ(toString(mtp_key.key_namespace), "MTP");
}

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeysSeparateDepthParticipantAndDirection)
{
    const auto depth0 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto depth1 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        1,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto participant3 = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        3,
        MoEOverlayCollectiveDirection::Dispatch);
    const auto return_key = makeMTPMoEOverlayCollectiveKey(
        5,
        8,
        0,
        3,
        1,
        7,
        2,
        MoEOverlayCollectiveDirection::ReturnReduce);

    std::set<MoEOverlayCollectiveKey> keys{depth0, depth1, participant3, return_key};
    EXPECT_EQ(keys.size(), 4u);
    EXPECT_TRUE(depth0.isValid());
    EXPECT_TRUE(depth1.isValid());
    EXPECT_TRUE(participant3.isValid());
    EXPECT_TRUE(return_key.isValid());
    EXPECT_NE(depth0.sequence, depth1.sequence);
    EXPECT_NE(depth0.sequence, participant3.sequence);
    EXPECT_NE(depth0.sequence, return_key.sequence);
}

TEST(Test__MoEOverlayCollectiveWorkspace, MTPCollectiveKeyRequiresDepth)
{
    auto key = dispatchKey(77);
    key.key_namespace = MoEOverlayCollectiveNamespace::MTP;
    key.histogram_source = ExpertHistogramSource::GroupedVerifier;
    key.participant_id = 0;
    EXPECT_FALSE(key.isValid());

    key.mtp_depth = 0;
    EXPECT_TRUE(key.isValid());
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     RankLocalSparseCrossParticipantEdgeCopiesExactPacketsAndRejectsReplay)
{
    constexpr int kLayer = 3;
    constexpr int kTier = 1;
    constexpr int kTargetParticipant = 3;
    constexpr int kContinuationParticipant = 0;
    constexpr int kDModel = 4;
    constexpr int kTopK = 2;

    MoEOverlayCollectiveWorkspace continuation;
    MoEOverlayCollectiveWorkspace local_endpoint;
    continuation.ensureCapacity(4, 8, kDModel, kTopK, DeviceId::cpu());
    local_endpoint.ensureCapacity(4, 8, kDModel, kTopK, DeviceId::cpu());
    MoEOverlayRankLocalSparseCollectiveContext rank_local(
        {.slot_count = 8});

    const auto dispatch_key = makeMoEOverlayCollectiveKey(
        /*generation_id=*/7,
        /*step_id=*/11,
        kLayer,
        kTier,
        /*domain_id=*/kTargetParticipant,
        /*participant_id=*/kTargetParticipant,
        MoEOverlayCollectiveDirection::Dispatch);
    auto outbound = continuation.localExpertInput(kLayer, kTier);
    outbound.key = dispatch_key;
    outbound.residency_epoch = 29;
    outbound.source_participant = kContinuationParticipant;
    outbound.target_participant = kTargetParticipant;
    outbound.live_row_count = 2;
    outbound.live_entry_count = 3;
    outbound.row_ids_host[0] = 2;
    outbound.row_ids_host[1] = 5;
    outbound.entry_offsets_host[0] = 0;
    outbound.entry_offsets_host[1] = 1;
    outbound.entry_offsets_host[2] = 3;
    outbound.expert_ids_host[0] = 9;
    outbound.expert_ids_host[1] = 4;
    outbound.expert_ids_host[2] = 12;
    outbound.route_weights_host[0] = 0.75f;
    outbound.route_weights_host[1] = 0.125f;
    outbound.route_weights_host[2] = 0.875f;
    for (size_t element = 0;
         element < outbound.live_row_count * static_cast<size_t>(kDModel);
         ++element)
    {
        outbound.hidden_rows_fp32[element] =
            20.0f + static_cast<float>(element);
    }

    auto inbound = local_endpoint.dispatchReceive(kLayer, kTier);
    const auto dispatch_result =
        rank_local.dispatch(dispatch_key, outbound, &inbound, nullptr);
    ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
    EXPECT_TRUE(dispatch_result.collective_complete);
    EXPECT_EQ(inbound.key, dispatch_key);
    EXPECT_EQ(inbound.residency_epoch, 29u);
    EXPECT_EQ(inbound.source_participant, kContinuationParticipant);
    EXPECT_EQ(inbound.target_participant, kTargetParticipant);
    EXPECT_EQ(inbound.live_row_count, 2u);
    EXPECT_EQ(inbound.live_entry_count, 3u);
    EXPECT_EQ(inbound.row_ids_host[1], 5);
    EXPECT_EQ(inbound.entry_offsets_host[2], 3);
    EXPECT_EQ(inbound.expert_ids_host[2], 12);
    EXPECT_FLOAT_EQ(inbound.route_weights_host[1], 0.125f);
    EXPECT_FLOAT_EQ(inbound.hidden_rows_fp32[7], 27.0f);

    const auto repeated_dispatch =
        rank_local.dispatch(dispatch_key, outbound, &inbound, nullptr);
    EXPECT_FALSE(repeated_dispatch.ok);
    EXPECT_EQ(repeated_dispatch.error_code, 3);

    const auto return_key = makeMoEOverlayCollectiveKey(
        /*generation_id=*/7,
        /*step_id=*/11,
        kLayer,
        kTier,
        /*domain_id=*/kTargetParticipant,
        /*participant_id=*/kTargetParticipant,
        MoEOverlayCollectiveDirection::ReturnReduce);
    auto returned = local_endpoint.localExpertOutput(kLayer, kTier);
    returned.key = return_key;
    returned.residency_epoch = inbound.residency_epoch;
    returned.source_participant = kTargetParticipant;
    returned.target_participant = kContinuationParticipant;
    returned.live_row_count = 2;
    returned.row_ids_host[0] = 2;
    returned.row_ids_host[1] = 5;
    for (size_t element = 0;
         element < returned.live_row_count * static_cast<size_t>(kDModel);
         ++element)
    {
        returned.output_rows_fp32[element] =
            100.0f + static_cast<float>(element);
    }

    auto return_inbound = continuation.returnReceive(kLayer, kTier);
    const auto return_result =
        rank_local.returnReduce(
            return_key, returned, &return_inbound, nullptr);
    ASSERT_TRUE(return_result.ok) << return_result.error;
    EXPECT_TRUE(return_result.collective_complete);
    EXPECT_EQ(return_inbound.key, return_key);
    EXPECT_EQ(return_inbound.residency_epoch, 29u);
    EXPECT_EQ(return_inbound.source_participant, kTargetParticipant);
    EXPECT_EQ(return_inbound.target_participant, kContinuationParticipant);
    EXPECT_EQ(return_inbound.live_row_count, 2u);
    EXPECT_EQ(return_inbound.row_ids_host[1], 5);
    EXPECT_FLOAT_EQ(return_inbound.output_rows_fp32[7], 107.0f);

    auto aborted_key = makeMoEOverlayCollectiveKey(
        /*generation_id=*/7,
        /*step_id=*/12,
        kLayer,
        kTier,
        /*domain_id=*/kTargetParticipant,
        /*participant_id=*/kTargetParticipant,
        MoEOverlayCollectiveDirection::Dispatch);
    rank_local.abort(aborted_key, /*reason_code=*/41);
    outbound.key = aborted_key;
    const auto aborted =
        rank_local.dispatch(aborted_key, outbound, &inbound, nullptr);
    EXPECT_FALSE(aborted.ok);
    EXPECT_EQ(aborted.error_code, 41);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     RankLocalSparseContinuationLoopbackCopiesExactPacketsAndRejectsReplay)
{
    constexpr int kParticipant = 0;
    constexpr int kDModel = 4;
    constexpr int kTopK = 1;

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(
        /*max_rows=*/2,
        /*max_entries=*/2,
        kDModel,
        kTopK,
        DeviceId::cpu());
    MoEOverlayRankLocalSparseCollectiveContext rank_local(
        {.slot_count = 8});

    const auto dispatch_key = makeMoEOverlayCollectiveKey(
        /*generation_id=*/13,
        /*step_id=*/17,
        /*layer_idx=*/0,
        /*tier_idx=*/0,
        /*domain_id=*/0,
        kParticipant,
        MoEOverlayCollectiveDirection::Dispatch);
    auto outbound = workspace.localExpertInput(0, 0);
    outbound.key = dispatch_key;
    outbound.residency_epoch = 5;
    outbound.source_participant = kParticipant;
    outbound.target_participant = kParticipant;
    outbound.live_row_count = 1;
    outbound.live_entry_count = 1;
    outbound.row_ids_host[0] = 1;
    outbound.entry_offsets_host[0] = 0;
    outbound.entry_offsets_host[1] = 1;
    outbound.expert_ids_host[0] = 7;
    outbound.route_weights_host[0] = 0.75f;
    for (int column = 0; column < kDModel; ++column)
    {
        outbound.hidden_rows_fp32[column] =
            10.0f + static_cast<float>(column);
    }

    auto inbound = workspace.dispatchReceive(0, 0);
    const auto dispatch_result =
        rank_local.dispatch(dispatch_key, outbound, &inbound, nullptr);
    ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
    EXPECT_TRUE(dispatch_result.collective_complete);
    EXPECT_EQ(inbound.source_participant, kParticipant);
    EXPECT_EQ(inbound.target_participant, kParticipant);
    EXPECT_EQ(inbound.live_row_count, 1u);
    EXPECT_EQ(inbound.expert_ids_host[0], 7);
    EXPECT_FLOAT_EQ(inbound.hidden_rows_fp32[3], 13.0f);

    const auto replay =
        rank_local.dispatch(dispatch_key, outbound, &inbound, nullptr);
    EXPECT_FALSE(replay.ok);
    EXPECT_EQ(replay.error_code, 3);

    const auto return_key = makeMoEOverlayCollectiveKey(
        /*generation_id=*/13,
        /*step_id=*/17,
        /*layer_idx=*/0,
        /*tier_idx=*/0,
        /*domain_id=*/0,
        kParticipant,
        MoEOverlayCollectiveDirection::ReturnReduce);
    auto returned = workspace.localExpertOutput(0, 0);
    returned.key = return_key;
    returned.residency_epoch = 5;
    returned.source_participant = kParticipant;
    returned.target_participant = kParticipant;
    returned.live_row_count = 1;
    returned.row_ids_host[0] = 1;
    for (int column = 0; column < kDModel; ++column)
    {
        returned.output_rows_fp32[column] =
            20.0f + static_cast<float>(column);
    }

    auto return_inbound = workspace.returnReceive(0, 0);
    const auto return_result = rank_local.returnReduce(
        return_key, returned, &return_inbound, nullptr);
    ASSERT_TRUE(return_result.ok) << return_result.error;
    EXPECT_TRUE(return_result.collective_complete);
    EXPECT_EQ(return_inbound.source_participant, kParticipant);
    EXPECT_EQ(return_inbound.target_participant, kParticipant);
    EXPECT_EQ(return_inbound.live_row_count, 1u);
    EXPECT_FLOAT_EQ(return_inbound.output_rows_fp32[3], 23.0f);
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalSparseCollectiveSeparatesMainAndMTPNamespaces)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 1, .slot_count = 1});
    const auto main_key = dispatchKey(33);
    const auto mtp_key = makeMTPMoEOverlayCollectiveKey(
        main_key.generation_id,
        main_key.step_id,
        0,
        main_key.layer_idx,
        main_key.tier_idx,
        main_key.domain_id,
        0,
        MoEOverlayCollectiveDirection::Dispatch);

    auto main_outbound = workspace.localExpertInput(3, 1);
    main_outbound.key = main_key;
    main_outbound.source_participant = 0;
    main_outbound.target_participant = 0;
    main_outbound.live_row_count = 0;
    main_outbound.live_entry_count = 0;
    main_outbound.entry_offsets_host[0] = 0;

    auto main_inbound = workspace.dispatchReceive(3, 1);
    const auto main_result = collective.dispatch(main_key, main_outbound, &main_inbound, nullptr);
    ASSERT_TRUE(main_result.ok) << main_result.error;
    EXPECT_TRUE(main_result.collective_complete);

    auto mtp_outbound = workspace.localExpertInput(3, 1);
    mtp_outbound.key = mtp_key;
    mtp_outbound.source_participant = 0;
    mtp_outbound.target_participant = 0;
    mtp_outbound.live_row_count = 0;
    mtp_outbound.live_entry_count = 0;
    mtp_outbound.entry_offsets_host[0] = 0;

    auto mtp_inbound = workspace.dispatchReceive(3, 1);
    const auto mtp_result = collective.dispatch(mtp_key, mtp_outbound, &mtp_inbound, nullptr);
    EXPECT_TRUE(mtp_result.ok) << mtp_result.error;
    EXPECT_TRUE(mtp_result.collective_complete);
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalSparseDispatchMovesPayloadAndNoOpCompletesKey)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    auto key = dispatchKey(33);

    auto outbound0 = workspace.localExpertInput(3, 1);
    outbound0.key = key;
    outbound0.residency_epoch = 71;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 2;
    outbound0.live_entry_count = 4;
    outbound0.row_ids_host[0] = 10;
    outbound0.row_ids_host[1] = 14;
    outbound0.entry_offsets_host[0] = 0;
    outbound0.entry_offsets_host[1] = 2;
    outbound0.entry_offsets_host[2] = 4;
    outbound0.expert_ids_host[0] = 5;
    outbound0.expert_ids_host[1] = 7;
    outbound0.expert_ids_host[2] = 9;
    outbound0.expert_ids_host[3] = 11;
    outbound0.route_weights_host[0] = 0.5f;
    outbound0.route_weights_host[1] = 0.5f;
    outbound0.route_weights_host[2] = 0.25f;
    outbound0.route_weights_host[3] = 0.75f;
    for (size_t index = 0; index < outbound0.live_row_count * static_cast<size_t>(outbound0.d_model); ++index)
        outbound0.hidden_rows_fp32[index] = static_cast<float>(100 + index);

    auto inbound0 = workspace.dispatchReceive(3, 1);
    auto first = collective.dispatch(key, outbound0, &inbound0, nullptr);
    ASSERT_TRUE(first.ok) << first.error;
    EXPECT_FALSE(first.collective_complete);

    auto no_op = workspace.localExpertInput(3, 1);
    no_op.key = key;
    no_op.source_participant = 1;
    no_op.target_participant = 0;
    no_op.live_row_count = 0;
    no_op.live_entry_count = 0;
    no_op.entry_offsets_host[0] = 0;

    auto inbound1 = workspace.dispatchReceive(3, 1);
    auto second = collective.dispatch(key, no_op, &inbound1, nullptr);
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(second.collective_complete);

    EXPECT_EQ(inbound1.live_row_count, 2u);
    EXPECT_EQ(inbound1.live_entry_count, 4u);
    EXPECT_EQ(inbound1.residency_epoch, 71u);
    EXPECT_EQ(inbound1.row_ids_host[0], 10);
    EXPECT_EQ(inbound1.row_ids_host[1], 14);
    EXPECT_EQ(inbound1.entry_offsets_host[0], 0);
    EXPECT_EQ(inbound1.entry_offsets_host[1], 2);
    EXPECT_EQ(inbound1.entry_offsets_host[2], 4);
    EXPECT_EQ(inbound1.expert_ids_host[0], 5);
    EXPECT_EQ(inbound1.expert_ids_host[3], 11);
    EXPECT_FLOAT_EQ(inbound1.route_weights_host[2], 0.25f);
    EXPECT_FLOAT_EQ(inbound1.hidden_rows_fp32[0], 100.0f);
    EXPECT_FLOAT_EQ(inbound1.hidden_rows_fp32[7], 107.0f);

    auto stale = collective.dispatch(key, no_op, &inbound1, nullptr);
    EXPECT_FALSE(stale.ok);
    EXPECT_EQ(stale.error_code, 4);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     LocalSparseDispatchRejectsMixedResidencyEpochs)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());
    MoEOverlayLocalSparseCollectiveContext collective(
        {.participant_count = 2, .slot_count = 4});
    const auto key = dispatchKey(133);

    auto first_outbound = workspace.localExpertInput(3, 1);
    first_outbound.key = key;
    first_outbound.residency_epoch = 17;
    first_outbound.source_participant = 0;
    first_outbound.target_participant = 1;
    first_outbound.live_row_count = 1;
    first_outbound.live_entry_count = 1;
    first_outbound.row_ids_host[0] = 0;
    first_outbound.entry_offsets_host[0] = 0;
    first_outbound.entry_offsets_host[1] = 1;
    first_outbound.expert_ids_host[0] = 0;
    first_outbound.route_weights_host[0] = 1.0f;
    std::fill_n(first_outbound.hidden_rows_fp32, 4, 1.0f);
    auto first_inbound = workspace.dispatchReceive(3, 1);
    const auto first =
        collective.dispatch(key, first_outbound, &first_inbound, nullptr);
    ASSERT_TRUE(first.ok) << first.error;
    ASSERT_FALSE(first.collective_complete);

    /*
     * Even an empty contribution is part of the same transaction. Giving it
     * a different non-zero epoch must fail before the receiver can execute an
     * ambiguous mixture of old and newly published expert banks.
     */
    auto second_outbound = workspace.localExpertInput(3, 1);
    second_outbound.key = key;
    second_outbound.residency_epoch = 18;
    second_outbound.source_participant = 1;
    second_outbound.target_participant = 1;
    second_outbound.live_row_count = 0;
    second_outbound.live_entry_count = 0;
    second_outbound.entry_offsets_host[0] = 0;
    auto second_inbound = workspace.dispatchReceive(3, 1);
    const auto second =
        collective.dispatch(key, second_outbound, &second_inbound, nullptr);
    EXPECT_FALSE(second.ok);
    EXPECT_EQ(second.error, "inbound sparse payload mixes residency epochs");
}

TEST(Test__MoEOverlayCollectiveWorkspace, LocalReturnReduceMovesCompactRowsByKey)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    auto key = returnKey(34);

    auto outbound0 = workspace.localExpertOutput(3, 1);
    outbound0.key = key;
    outbound0.residency_epoch = 72;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 2;
    outbound0.row_ids_host[0] = 3;
    outbound0.row_ids_host[1] = 4;
    for (size_t index = 0; index < outbound0.live_row_count * static_cast<size_t>(outbound0.d_model); ++index)
        outbound0.output_rows_fp32[index] = static_cast<float>(200 + index);

    auto inbound0 = workspace.returnReceive(3, 1);
    auto first = collective.returnReduce(key, outbound0, &inbound0, nullptr);
    ASSERT_TRUE(first.ok) << first.error;
    EXPECT_FALSE(first.collective_complete);

    auto no_op = workspace.localExpertOutput(3, 1);
    no_op.key = key;
    no_op.source_participant = 1;
    no_op.target_participant = 0;
    no_op.live_row_count = 0;

    auto inbound1 = workspace.returnReceive(3, 1);
    auto second = collective.returnReduce(key, no_op, &inbound1, nullptr);
    ASSERT_TRUE(second.ok) << second.error;
    EXPECT_TRUE(second.collective_complete);

    EXPECT_EQ(inbound1.live_row_count, 2u);
    EXPECT_EQ(inbound1.residency_epoch, 72u);
    EXPECT_EQ(inbound1.row_ids_host[0], 3);
    EXPECT_EQ(inbound1.row_ids_host[1], 4);
    EXPECT_FLOAT_EQ(inbound1.output_rows_fp32[0], 200.0f);
    EXPECT_FLOAT_EQ(inbound1.output_rows_fp32[7], 207.0f);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     MPISparseWirePreservesDispatchAndReturnResidencyEpoch)
{
    auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 1);
    MoEOverlayMPISparseCollectiveContext collective(
        {.mpi_ctx = mpi, .local_participant_ids = {0}});
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());

    const auto dispatch_key = dispatchKey(134);
    auto dispatch_outbound = workspace.localExpertInput(3, 1);
    dispatch_outbound.key = dispatch_key;
    dispatch_outbound.residency_epoch = 99;
    dispatch_outbound.source_participant = 0;
    dispatch_outbound.target_participant = 0;
    dispatch_outbound.live_row_count = 1;
    dispatch_outbound.live_entry_count = 1;
    dispatch_outbound.row_ids_host[0] = 2;
    dispatch_outbound.entry_offsets_host[0] = 0;
    dispatch_outbound.entry_offsets_host[1] = 1;
    dispatch_outbound.expert_ids_host[0] = 3;
    dispatch_outbound.route_weights_host[0] = 0.75f;
    std::fill_n(dispatch_outbound.hidden_rows_fp32, 4, 2.0f);
    auto dispatch_inbound = workspace.dispatchReceive(3, 1);
    const auto dispatch_result = collective.dispatch(
        dispatch_key, dispatch_outbound, &dispatch_inbound, nullptr);
    ASSERT_TRUE(dispatch_result.ok) << dispatch_result.error;
    EXPECT_TRUE(dispatch_result.collective_complete);
    EXPECT_EQ(dispatch_inbound.residency_epoch, 99u);
    EXPECT_EQ(dispatch_inbound.live_row_count, 1u);

    const auto return_key = returnKey(135);
    auto return_outbound = workspace.localExpertOutput(3, 1);
    return_outbound.key = return_key;
    return_outbound.residency_epoch = dispatch_inbound.residency_epoch;
    return_outbound.source_participant = 0;
    return_outbound.target_participant = 0;
    return_outbound.live_row_count = 1;
    return_outbound.row_ids_host[0] = 2;
    std::fill_n(return_outbound.output_rows_fp32, 4, 3.0f);
    auto return_inbound = workspace.returnReceive(3, 1);
    const auto return_result = collective.returnReduce(
        return_key, return_outbound, &return_inbound, nullptr);
    ASSERT_TRUE(return_result.ok) << return_result.error;
    EXPECT_TRUE(return_result.collective_complete);
    EXPECT_EQ(return_inbound.residency_epoch, 99u);
    EXPECT_EQ(return_inbound.live_row_count, 1u);
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     ProtocolOnlyReturnParticipantNeedsNoDummyDenseTensor)
{
    auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 1);
    MoEOverlayMPISparseCollectiveContext collective(
        {.mpi_ctx = mpi, .local_participant_ids = {1}});
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);

    const auto key = returnKey(136);
    auto outbound = workspace.localExpertOutput(3, 1);
    outbound.key = key;
    outbound.source_participant = 1;
    outbound.target_participant = 0;
    outbound.live_row_count = 0;
    auto inbound = workspace.returnReceive(3, 1);

    MoESparseReturnReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.key = key;
    params.source_participant = 1;
    params.target_participant = 0;
    params.outbound_rows = &outbound;
    params.inbound_rows = &inbound;
    params.inbound_consumer_role = MoESparseReturnReduceStage::
        InboundConsumerRole::ProtocolParticipant;
    params.seq_len = 4;
    params.d_model = 4;

    MoESparseReturnReduceStage stage(std::move(params));
    EXPECT_TRUE(stage.execute(&ctx));
    EXPECT_TRUE(stage.manualGraphBoundaryComplete());
    EXPECT_TRUE(stage.getBufferRequirements().buffers.empty());
}

TEST(
    Test__MoEOverlayCollectiveWorkspace,
    DistributedPeerPreservesLogicalSourceWithoutPayloadAuthority)
{
    auto mpi = std::make_shared<llaminar2::test::MockMPIContext>(0, 1);
    MoEOverlayMPISparseCollectiveContext collective(
        {.mpi_ctx = mpi, .local_participant_ids = {1}});
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);

    auto inbound = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.workspace = &workspace;
    params.key = dispatchKey(137);
    params.source_participant = 0;
    params.target_participant = 1;
    params.seq_len = 4;
    params.top_k = 2;
    params.d_model = 4;
    params.payload_publication_role =
        MoESparseDispatchStage::PayloadPublicationRole::
            EmptyCollectiveParticipant;
    params.replicated_hidden_export = true;
    params.logical_continuation_root_participant = 0;
    params.inbound_rows = &inbound;

    MoESparseDispatchStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_TRUE(stage.manualGraphBoundaryComplete());
    EXPECT_EQ(inbound.source_participant, 0);
    EXPECT_EQ(inbound.target_participant, 1);
    EXPECT_EQ(inbound.residency_epoch, 0u);
    EXPECT_EQ(inbound.live_row_count, 0u);
    EXPECT_EQ(inbound.live_entry_count, 0u);

    auto forbidden_output = std::make_shared<MoEExpertDispatchOutput>();
    auto forbidden_inbound = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params forbidden_params;
    forbidden_params.device_id = DeviceId::cpu();
    forbidden_params.collective_context = &collective;
    forbidden_params.workspace = &workspace;
    forbidden_params.key = dispatchKey(138);
    forbidden_params.source_participant = 0;
    forbidden_params.target_participant = 1;
    forbidden_params.seq_len = 4;
    forbidden_params.top_k = 2;
    forbidden_params.d_model = 4;
    forbidden_params.payload_publication_role =
        MoESparseDispatchStage::PayloadPublicationRole::
            EmptyCollectiveParticipant;
    forbidden_params.replicated_hidden_export = true;
    forbidden_params.logical_continuation_root_participant = 0;
    forbidden_params.dispatch_output_lifetime = forbidden_output;
    forbidden_params.inbound_rows = &forbidden_inbound;

    MoESparseDispatchStage forbidden_stage(std::move(forbidden_params));
    EXPECT_FALSE(forbidden_stage.execute(&ctx));
}

TEST(
    Test__MoEOverlayCollectiveWorkspace,
    CurrentBatchTargetedDispatchPacksOnlyRowsAssignedToItsParticipant)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(3, 6, 4, 2, DeviceId::cpu());
    MoEOverlayLocalSparseCollectiveContext collective(
        {.participant_count = 1, .slot_count = 2});
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);

    FP32Tensor hidden({3, 4});
    FP32Tensor routing_indices({3, 2});
    FP32Tensor routing_weights({3, 2});
    for (size_t element = 0; element < hidden.numel(); ++element)
        hidden.mutable_data()[element] = static_cast<float>(element + 1u);
    std::fill_n(routing_indices.mutable_data(), routing_indices.numel(), 0.0f);
    std::fill_n(routing_weights.mutable_data(), routing_weights.numel(), 1.0f);

    MoEExpertTierDispatch tier_dispatch;
    tier_dispatch.tier_index = 1;
    tier_dispatch.token_rows = {0, 1, 2};
    tier_dispatch.entries = {
        {.token_row = 0, .route_slot = 0, .expert_id = 0,
         .route_weight = 0.6f, .destination_participant = 0},
        {.token_row = 0, .route_slot = 1, .expert_id = 1,
         .route_weight = 0.4f, .destination_participant = 1},
        {.token_row = 1, .route_slot = 0, .expert_id = 2,
         .route_weight = 0.7f, .destination_participant = 1},
        {.token_row = 1, .route_slot = 1, .expert_id = 3,
         .route_weight = 0.3f, .destination_participant = 1},
        {.token_row = 2, .route_slot = 0, .expert_id = 4,
         .route_weight = 0.8f, .destination_participant = 0},
        {.token_row = 2, .route_slot = 1, .expert_id = 5,
         .route_weight = 0.2f, .destination_participant = 1},
    };

    auto inbound = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.workspace = &workspace;
    params.key = dispatchKey(139);
    params.source_participant = 0;
    params.target_participant = 0;
    params.hidden = &hidden;
    params.routing_indices = &routing_indices;
    params.routing_weights = &routing_weights;
    params.seq_len = 3;
    params.top_k = 2;
    params.d_model = 4;
    params.tier_dispatch = &tier_dispatch;
    params.fixed_residency_epoch = 41;
    params.tier_index = 1;
    params.inbound_rows = &inbound;

    MoESparseDispatchStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(&ctx));
    ASSERT_TRUE(stage.manualGraphBoundaryComplete());
    ASSERT_EQ(inbound.live_row_count, 2u);
    ASSERT_EQ(inbound.live_entry_count, 2u);
    EXPECT_EQ(inbound.residency_epoch, 41u);
    EXPECT_EQ(inbound.row_ids_host[0], 0);
    EXPECT_EQ(inbound.row_ids_host[1], 2);
    EXPECT_EQ(inbound.entry_offsets_host[0], 0);
    EXPECT_EQ(inbound.entry_offsets_host[1], 1);
    EXPECT_EQ(inbound.entry_offsets_host[2], 2);
    EXPECT_EQ(inbound.expert_ids_host[0], 0);
    EXPECT_EQ(inbound.expert_ids_host[1], 4);
    EXPECT_FLOAT_EQ(inbound.route_weights_host[0], 0.6f);
    EXPECT_FLOAT_EQ(inbound.route_weights_host[1], 0.8f);
    EXPECT_FLOAT_EQ(inbound.hidden_rows_fp32[0], 1.0f);
    EXPECT_FLOAT_EQ(inbound.hidden_rows_fp32[4], 9.0f);
}

TEST(
    Test__MoEOverlayCollectiveWorkspace,
    SparseDispatchRequiresAndSharesRunnerStampedTransactionIdentity)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    const auto base_key = dispatchKey(39);

    auto make_stage = [&workspace, &base_key](
                          MoEOverlayLocalSparseCollectiveContext &collective,
                          int source_participant,
                          int target_participant,
                          bool requires_completion,
                          MoEOverlaySparseRows *inbound_rows)
    {
        MoESparseDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.collective_context = &collective;
        params.workspace = &workspace;
        params.key = base_key;
        params.source_participant = source_participant;
        params.target_participant = target_participant;
        params.seq_len = 4;
        params.top_k = 2;
        params.d_model = 4;
        params.manual_boundary_requires_collective_completion = requires_completion;
        params.require_explicit_transaction_identity = true;
        params.inbound_rows = inbound_rows;
        return MoESparseDispatchStage(std::move(params));
    };

    // A distributed graph must never use its object-local execution counter
    // as a wire identity.  It has no valid transaction until the runner has
    // published the request generation and logical chunk offset.
    MoEOverlayLocalSparseCollectiveContext missing_identity_collective(
        {.participant_count = 2, .slot_count = 8});
    auto missing_inbound = workspace.dispatchReceive(3, 1);
    auto missing_identity_stage = make_stage(
        missing_identity_collective, 0, 1, false, &missing_inbound);
    EXPECT_TRUE(missing_identity_stage.hasMoEOverlayCollectiveRuntimeParams());
    EXPECT_FALSE(missing_identity_stage.execute(&ctx));

    // Independently materialized participant stages reject different runtime
    // identities even though their static graph construction key is identical.
    MoEOverlayLocalSparseCollectiveContext mismatch_collective(
        {.participant_count = 2, .slot_count = 8});
    auto mismatch_inbound0 = workspace.dispatchReceive(3, 1);
    auto mismatch_inbound1 = workspace.dispatchReceive(3, 1);
    auto mismatch_first = make_stage(
        mismatch_collective, 0, 1, false, &mismatch_inbound0);
    auto mismatch_second = make_stage(
        mismatch_collective, 1, 0, true, &mismatch_inbound1);
    mismatch_first.updateMoEOverlayCollectiveRuntimeParams(
        {.generation_id = 41, .step_id = 12});
    mismatch_second.updateMoEOverlayCollectiveRuntimeParams(
        {.generation_id = 41, .step_id = 13});
    ASSERT_TRUE(mismatch_first.execute(&ctx));
    EXPECT_FALSE(mismatch_second.execute(&ctx));

    // Matching runner-stamped values complete the same two-participant
    // protocol even when the stage objects were constructed separately.
    MoEOverlayLocalSparseCollectiveContext matching_collective(
        {.participant_count = 2, .slot_count = 8});
    auto matching_inbound0 = workspace.dispatchReceive(3, 1);
    auto matching_inbound1 = workspace.dispatchReceive(3, 1);
    auto matching_first = make_stage(
        matching_collective, 0, 1, false, &matching_inbound0);
    auto matching_second = make_stage(
        matching_collective, 1, 0, true, &matching_inbound1);
    const IComputeStage::MoEOverlayCollectiveRuntimeParams shared_identity{
        .generation_id = 42,
        .step_id = 14,
    };
    matching_first.updateMoEOverlayCollectiveRuntimeParams(shared_identity);
    matching_second.updateMoEOverlayCollectiveRuntimeParams(shared_identity);
    ASSERT_TRUE(matching_first.execute(&ctx));
    ASSERT_TRUE(matching_second.execute(&ctx));
    EXPECT_TRUE(matching_second.manualGraphBoundaryComplete());
}

TEST(Test__MoEOverlayCollectiveWorkspace, SparseDispatchStageReportsManualBoundaryCompletion)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = dispatchKey(36);

    auto inbound0 = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params first_params;
    first_params.device_id = DeviceId::cpu();
    first_params.collective_context = &collective;
    first_params.workspace = &workspace;
    first_params.key = key;
    first_params.source_participant = 0;
    first_params.target_participant = 1;
    first_params.seq_len = 4;
    first_params.top_k = 2;
    first_params.d_model = 4;
    first_params.manual_boundary_requires_collective_completion = false;
    first_params.inbound_rows = &inbound0;

    MoESparseDispatchStage first_stage(std::move(first_params));
    ASSERT_TRUE(first_stage.execute(&ctx));
    EXPECT_TRUE(first_stage.isManualGraphBoundary());
    EXPECT_TRUE(first_stage.manualGraphBoundaryComplete());

    auto inbound1 = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params second_params;
    second_params.device_id = DeviceId::cpu();
    second_params.collective_context = &collective;
    second_params.workspace = &workspace;
    second_params.key = key;
    second_params.source_participant = 1;
    second_params.target_participant = 0;
    second_params.seq_len = 4;
    second_params.top_k = 2;
    second_params.d_model = 4;
    second_params.manual_boundary_requires_collective_completion = true;
    second_params.inbound_rows = &inbound1;

    MoESparseDispatchStage second_stage(std::move(second_params));
    ASSERT_TRUE(second_stage.execute(&ctx));
    EXPECT_TRUE(second_stage.isManualGraphBoundary());
    EXPECT_TRUE(second_stage.manualGraphBoundaryComplete());
}

TEST(Test__MoEOverlayCollectiveWorkspace,
     SparseDispatchRequiresAndPublishesTypedGraphSemantics)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(1, 1, 4, 1, DeviceId::cpu());
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(), ComputeBackendType::CPU);

    const auto build_stage = [&](
                                 MoEOverlayLocalSparseCollectiveContext &collective,
                                 MoEOverlaySparseRows *inbound,
                                 MoEOverlayCollectiveKey base_key)
    {
        MoESparseDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.collective_context = &collective;
        params.workspace = &workspace;
        params.key = std::move(base_key);
        params.source_participant = 0;
        params.target_participant = 0;
        params.seq_len = 1;
        params.top_k = 1;
        params.d_model = 4;
        params.require_explicit_transaction_identity = true;
        params.require_explicit_execution_semantics = true;
        params.inbound_rows = inbound;
        return MoESparseDispatchStage(std::move(params));
    };

    MoEOverlayLocalSparseCollectiveContext missing_context(
        {.participant_count = 1, .slot_count = 2});
    auto missing_inbound = workspace.dispatchReceive(0, 0);
    auto missing = build_stage(
        missing_context,
        &missing_inbound,
        makeMoEOverlayCollectiveKey(
            1, 0, 0, 0, 0, 0,
            MoEOverlayCollectiveDirection::Dispatch));
    missing.updateMoEOverlayCollectiveRuntimeParams(
        {.generation_id = 9, .step_id = 7});
    EXPECT_FALSE(missing.execute(&ctx));

    MoEOverlayLocalSparseCollectiveContext prefill_context(
        {.participant_count = 1, .slot_count = 2});
    auto prefill_inbound = workspace.dispatchReceive(0, 0);
    auto prefill = build_stage(
        prefill_context,
        &prefill_inbound,
        makeMoEOverlayCollectiveKey(
            1, 0, 0, 0, 0, 0,
            MoEOverlayCollectiveDirection::Dispatch));
    prefill.updateMoEOverlayCollectiveRuntimeParams({
        .generation_id = 9,
        .step_id = 7,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::Prefill,
    });
    ASSERT_TRUE(prefill.execute(&ctx));
    EXPECT_EQ(
        prefill_inbound.key.histogram_source,
        ExpertHistogramSource::PrefillChunk);
    EXPECT_EQ(prefill_inbound.key.generation_id, 9u);
    EXPECT_EQ(prefill_inbound.key.step_id, 7u);

    auto decode_key = prefill_inbound.key;
    decode_key.histogram_source = ExpertHistogramSource::DecodeToken;
    EXPECT_NE(decode_key, prefill_inbound.key);
    EXPECT_NE(decode_key.toString(), prefill_inbound.key.toString());

    const auto mtp_key = makeMTPMoEOverlayCollectiveKey(
        9,
        7,
        2,
        0,
        0,
        0,
        0,
        MoEOverlayCollectiveDirection::Dispatch);
    EXPECT_EQ(
        mtp_key.histogram_source,
        ExpertHistogramSource::GroupedVerifier);
    EXPECT_TRUE(mtp_key.isValid());

    MoEOverlayLocalSparseCollectiveContext verifier_context(
        {.participant_count = 1, .slot_count = 2});
    auto verifier_inbound = workspace.dispatchReceive(0, 0);
    auto verifier = build_stage(
        verifier_context,
        &verifier_inbound,
        makeMoEOverlayCollectiveKey(
            1, 0, 0, 0, 0, 0,
            MoEOverlayCollectiveDirection::Dispatch));
    verifier.updateMoEOverlayCollectiveRuntimeParams({
        .generation_id = 9,
        .step_id = 7,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::GroupedVerifier,
        .mtp_depth = 2,
    });
    ASSERT_TRUE(verifier.execute(&ctx));
    EXPECT_EQ(verifier_inbound.key, mtp_key);

    MoEOverlayLocalSparseCollectiveContext sidecar_context(
        {.participant_count = 1, .slot_count = 2});
    auto sidecar_inbound = workspace.dispatchReceive(0, 0);
    auto sidecar = build_stage(
        sidecar_context,
        &sidecar_inbound,
        makeMTPMoEOverlayCollectiveKey(
            1, 0, 0, 0, 0, 0, 0,
            MoEOverlayCollectiveDirection::Dispatch));
    sidecar.updateMoEOverlayCollectiveRuntimeParams({
        .generation_id = 9,
        .step_id = 8,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::MTPDraft,
        .mtp_depth = 0,
    });
    ASSERT_TRUE(sidecar.execute(&ctx));
    EXPECT_EQ(sidecar_inbound.key.key_namespace,
              MoEOverlayCollectiveNamespace::MTP);
    EXPECT_EQ(sidecar_inbound.key.mtp_depth, 0);
    EXPECT_EQ(sidecar_inbound.key.step_id, 8u);
}

TEST(Test__MoEOverlayCollectiveWorkspace, SparseDispatchAdvertisesArenaInputContractForRootHostExport)
{
    FP32Tensor hidden({1, 4});
    FP32Tensor routing_indices({1, 2});
    FP32Tensor routing_weights({1, 2});

    MoESparseDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.hidden = &hidden;
    params.routing_indices = &routing_indices;
    params.routing_weights = &routing_weights;
    params.hidden_buffer_id = BufferId::NORMALIZED;
    params.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
    params.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;

    MoESparseDispatchStage stage(std::move(params));
    const auto contract = stage.bufferContract();

    EXPECT_TRUE(hasInputBinding(contract, BufferId::NORMALIZED));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_INDICES));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_WEIGHTS));
}

TEST(Test__MoEOverlayCollectiveWorkspace, SparseDispatchFinalBoundaryRequiresCollectiveCompletion)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = dispatchKey(37);

    auto inbound0 = workspace.dispatchReceive(3, 1);
    MoESparseDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.workspace = &workspace;
    params.key = key;
    params.source_participant = 0;
    params.target_participant = 0;
    params.seq_len = 4;
    params.top_k = 2;
    params.d_model = 4;
    params.manual_boundary_requires_collective_completion = true;
    params.inbound_rows = &inbound0;

    MoESparseDispatchStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_TRUE(stage.isManualGraphBoundary());
    EXPECT_FALSE(stage.manualGraphBoundaryComplete());
}

TEST(Test__MoEOverlayCollectiveWorkspace, ReturnReduceBroadcastWaitsForCollectiveComplete)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    CountingTPContext tp_context;
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = returnKey(35);

    auto outbound0 = workspace.localExpertOutput(3, 1);
    outbound0.key = key;
    outbound0.residency_epoch = 73;
    outbound0.source_participant = 0;
    outbound0.target_participant = 1;
    outbound0.live_row_count = 1;
    outbound0.row_ids_host[0] = 2;
    for (int col = 0; col < outbound0.d_model; ++col)
        outbound0.output_rows_fp32[col] = static_cast<float>(10 + col);

    FP32Tensor dense0(std::vector<size_t>{4, 4});
    auto inbound0 = workspace.returnReceive(3, 1);
    MoESparseReturnReduceStage::Params first_params;
    first_params.device_id = DeviceId::cpu();
    first_params.collective_context = &collective;
    first_params.key = key;
    first_params.source_participant = 0;
    first_params.target_participant = 1;
    first_params.outbound_rows = &outbound0;
    first_params.inbound_rows = &inbound0;
    first_params.dense_output = &dense0;
    first_params.seq_len = 4;
    first_params.d_model = 4;
    first_params.manual_boundary_requires_collective_completion = false;
    first_params.broadcast_after_scatter = true;
    first_params.continuation_tp_context = &tp_context;
    first_params.continuation_root_tp_index = 1;

    MoESparseReturnReduceStage first_stage(std::move(first_params));
    ASSERT_TRUE(first_stage.execute(&ctx));
    EXPECT_TRUE(first_stage.isManualGraphBoundary());
    EXPECT_TRUE(first_stage.manualGraphBoundaryComplete());
    EXPECT_EQ(tp_context.broadcast_calls, 0);

    auto outbound1 = workspace.localExpertOutput(3, 1);
    outbound1.key = key;
    outbound1.source_participant = 1;
    outbound1.target_participant = 0;
    outbound1.live_row_count = 0;

    FP32Tensor dense1(std::vector<size_t>{4, 4});
    auto inbound1 = workspace.returnReceive(3, 1);
    MoESparseReturnReduceStage::Params second_params;
    second_params.device_id = DeviceId::cpu();
    second_params.collective_context = &collective;
    second_params.key = key;
    second_params.source_participant = 1;
    second_params.target_participant = 1;
    second_params.outbound_rows = &outbound1;
    second_params.inbound_rows = &inbound1;
    second_params.dense_output = &dense1;
    second_params.seq_len = 4;
    second_params.d_model = 4;
    second_params.manual_boundary_requires_collective_completion = true;
    second_params.broadcast_after_scatter = true;
    second_params.continuation_tp_context = &tp_context;
    second_params.continuation_root_tp_index = 1;

    MoESparseReturnReduceStage second_stage(std::move(second_params));
    ASSERT_TRUE(second_stage.execute(&ctx));
    EXPECT_TRUE(second_stage.isManualGraphBoundary());
    EXPECT_TRUE(second_stage.manualGraphBoundaryComplete());
    EXPECT_EQ(tp_context.broadcast_calls, 1);
    EXPECT_EQ(tp_context.last_source_index, 1);
}

TEST(Test__MoEOverlayCollectiveWorkspace, SparseReturnFinalBoundaryRequiresCollectiveCompletion)
{
    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(16, 32, 4, 2, DeviceId::cpu());

    MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 8});
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
    auto key = returnKey(38);

    auto outbound0 = workspace.localExpertOutput(3, 1);
    outbound0.key = key;
    outbound0.source_participant = 0;
    outbound0.target_participant = 0;
    outbound0.live_row_count = 0;

    FP32Tensor dense0(std::vector<size_t>{4, 4});
    auto inbound0 = workspace.returnReceive(3, 1);
    MoESparseReturnReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.key = key;
    params.source_participant = 0;
    params.target_participant = 0;
    params.outbound_rows = &outbound0;
    params.inbound_rows = &inbound0;
    params.dense_output = &dense0;
    params.seq_len = 4;
    params.d_model = 4;
    params.manual_boundary_requires_collective_completion = true;

    MoESparseReturnReduceStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_TRUE(stage.isManualGraphBoundary());
    EXPECT_FALSE(stage.manualGraphBoundaryComplete());
}

TEST(
    Test__MoEOverlayCollectiveWorkspace,
    FinalContinuationReturnReleasesExactResidencyEpochLease)
{
    auto authority = staticResidencyAuthority();
    auto acquired = authority->tryAcquireTicketSnapshot();
    ASSERT_TRUE(acquired.has_value());

    auto dispatch_output = std::make_shared<MoEExpertDispatchOutput>();
    dispatch_output->residency_epoch = (*acquired)->epoch;
    dispatch_output->residency_lease =
        std::make_shared<MoEOverlayResidencyAuthority::TicketLease>(
            std::move(*acquired));
    ASSERT_EQ(authority->activeTicketCount(), 1u);

    MoEOverlayCollectiveWorkspace workspace;
    workspace.ensureCapacity(4, 8, 4, 2, DeviceId::cpu());
    MoEOverlayLocalSparseCollectiveContext collective(
        {.participant_count = 1, .slot_count = 4});
    llaminar2::testing::MockDeviceContext ctx(
        DeviceId::cpu(),
        ComputeBackendType::CPU);

    auto outbound = workspace.localExpertOutput(3, 1);
    outbound.key = returnKey(91);
    outbound.residency_epoch = dispatch_output->residency_epoch;
    outbound.source_participant = 0;
    outbound.target_participant = 0;
    outbound.live_row_count = 0;
    auto inbound = workspace.returnReceive(3, 1);
    FP32Tensor dense(std::vector<size_t>{4, 4});

    MoESparseReturnReduceStage::Params params;
    params.device_id = DeviceId::cpu();
    params.collective_context = &collective;
    params.key = returnKey(91);
    params.source_participant = 0;
    params.target_participant = 0;
    params.outbound_rows = &outbound;
    params.inbound_rows = &inbound;
    params.dense_output = &dense;
    params.seq_len = 4;
    params.d_model = 4;
    params.manual_boundary_requires_collective_completion = true;
    params.dispatch_output_lifetime = dispatch_output;
    params.residency_lease_terminal =
        MoEOverlayHostDispatchLeaseTerminal::Release;

    MoESparseReturnReduceStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(&ctx));
    EXPECT_TRUE(stage.manualGraphBoundaryComplete());
    EXPECT_EQ(dispatch_output->residency_epoch, 1u);
    EXPECT_EQ(dispatch_output->residency_lease, nullptr);
    EXPECT_EQ(authority->activeTicketCount(), 0u);
}
