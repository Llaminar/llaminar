/**
 * @file Test__MoEExpertDispatchStage.cpp
 * @brief Device-free sparse-dispatch and captured-ticket protocol regressions.
 *
 * The suite proves route validation, immutable residency leases, sparse tier
 * descriptors, and exact phase-specific histogram publication at the explicit
 * heterogeneous boundary.  Ticket tests retain padded poison rows so any
 * accidental use of physical rather than logical geometry fails loudly.
 */

#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
namespace
{

    TEST(Test__MoEExpertDispatchLeaseLifecycle,
         MappedParentCannotStealHostLeaseTerminalOwnership)
    {
        using Owner = MoEOverlayHostDispatchLeaseOwner;
        using Terminal = MoEOverlayHostDispatchLeaseTerminal;

        EXPECT_EQ(
            moeOverlayHostDispatchLeaseTerminal(
                Owner::FinalSparseReturn,
                /*final_ordered_return=*/true),
            Terminal::Release);
        EXPECT_EQ(
            moeOverlayHostDispatchLeaseTerminal(
                Owner::FinalSparseReturn,
                /*final_ordered_return=*/false),
            Terminal::Retain);
        EXPECT_EQ(
            moeOverlayHostDispatchLeaseTerminal(
                Owner::CurrentBatchLLEP,
                /*final_ordered_return=*/true),
            Terminal::Retain);
        EXPECT_EQ(
            moeOverlayHostDispatchLeaseTerminal(
                Owner::None,
                /*final_ordered_return=*/true),
            Terminal::Retain);
    }

    RoutedExpertTier tier(const std::string &name, const std::string &domain, bool fallback = false)
    {
        RoutedExpertTier result;
        result.name = name;
        result.domain = domain;
        result.fallback = fallback;
        return result;
    }

    RoutedExpertLayerPlacement placement(std::vector<int> routed_expert_tier)
    {
        RoutedExpertLayerPlacement result;
        result.layer = 4;
        result.routed_expert_tier = std::move(routed_expert_tier);
        return result;
    }

    std::unique_ptr<FP32Tensor> routingTensor(const std::vector<size_t> &shape, const std::vector<float> &values)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    }

    MoEExpertDispatchStage::Params paramsFor(
        const ITensor *indices,
        const ITensor *weights,
        int seq_len,
        int top_k,
        std::optional<RoutedExpertLayerPlacement> expert_placement,
        std::vector<RoutedExpertTier> routed_tiers,
        MoEExpertDispatchOutput *output)
    {
        MoEExpertDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.routing_indices = indices;
        params.routing_weights = weights;
        params.seq_len = seq_len;
        params.top_k = top_k;
        params.d_model = 8;
        params.placement = std::move(expert_placement);
        params.routed_tiers = std::move(routed_tiers);
        params.output = output;
        return params;
    }

    int countEntry(
        const MoEExpertDispatchOutput &output,
        int tier_index,
        int token_row,
        int route_slot,
        int expert_id,
        float route_weight)
    {
        const auto &entries = output.tiers.at(static_cast<size_t>(tier_index)).entries;
        return static_cast<int>(std::count_if(entries.begin(), entries.end(), [&](const auto &entry) {
            return entry.token_row == token_row &&
                   entry.route_slot == route_slot &&
                   entry.expert_id == expert_id &&
                   entry.route_weight == route_weight;
        }));
    }

    void expectContributionExactlyOnce(
        const MoEExpertDispatchOutput &output,
        int tier_index,
        int token_row,
        int route_slot,
        int expert_id,
        float route_weight)
    {
        EXPECT_EQ(countEntry(output, tier_index, token_row, route_slot, expert_id, route_weight), 1)
            << "token=" << token_row << " slot=" << route_slot << " expert=" << expert_id
            << " tier=" << tier_index;
    }

    bool hasInputBinding(const StageBufferContract &contract, BufferId id)
    {
        return std::any_of(contract.inputs.begin(), contract.inputs.end(),
                           [id](const BufferBinding &binding)
                           {
                               return binding.id == id && binding.access == BufferAccess::READ;
                           });
    }

    RoutedExpertDomain residencyDomain(
        std::string name,
        GlobalDeviceAddress participant,
        int world_rank,
        CollectiveBackendType backend)
    {
        RoutedExpertDomain result;
        result.name = std::move(name);
        result.scope = ExecutionDomainScope::SINGLE;
        result.backend = backend;
        result.participants = {participant};
        result.world_ranks = {world_rank};
        result.owner_rank = world_rank;
        result.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        return result;
    }

    MoERoutedExpertPlacementPlan dynamicResidencyPlan()
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "gpu_hot";
        plan.shared_expert_domain = "gpu_hot";
        plan.residency_policy =
            RoutedExpertResidencyPolicy::HistogramTieredCache;
        plan.domains = {
            residencyDomain(
                "gpu_hot",
                GlobalDeviceAddress::cuda(0, 0),
                0,
                CollectiveBackendType::NCCL),
            residencyDomain(
                "cpu_cold",
                GlobalDeviceAddress::cpu(1),
                1,
                CollectiveBackendType::MPI),
        };
        auto hot = tier("hot", "gpu_hot");
        hot.priority = 0;
        hot.max_experts_per_layer = 2;
        auto cold = tier("cold", "cpu_cold", true);
        cold.priority = 1;
        plan.routed_tiers = {std::move(hot), std::move(cold)};
        return plan;
    }

    class AcceptingResidencyTransport final
        : public IMoEOverlayResidencyTransport
    {
    public:
        class Wave final : public IMoEOverlayResidencyWave
        {
        public:
            explicit Wave(AcceptingResidencyTransport *owner) : owner_(owner) {}

            MoEOverlayResidencyWaveProgress pollStage(
                std::string *) noexcept override
            {
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            bool beginPrepare(std::string *) noexcept override
            {
                ++owner_->prepare_calls;
                return true;
            }

            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *) noexcept override
            {
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            bool beginPublication(std::string *) noexcept override
            {
                ++owner_->publication_calls;
                return true;
            }

            MoEOverlayResidencyWaveProgress pollPublication(
                std::string *) noexcept override
            {
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            void abortStaged() noexcept override {}

            void retirePrevious() noexcept override
            {
                ++owner_->retire_calls;
            }

        private:
            AcceptingResidencyTransport *owner_ = nullptr;
        };

        MoEOverlayResidencyStageStart beginStage(
            const MoEOverlayResidencyTransaction &) override
        {
            ++stage_calls;
            return {
                .status = MoEOverlayResidencyStageStartStatus::Started,
                .wave = std::make_unique<Wave>(this),
            };
        }

        int stage_calls = 0;
        int prepare_calls = 0;
        int publication_calls = 0;
        int retire_calls = 0;
    };

} // namespace

class Test__MoEExpertDispatchStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    std::unique_ptr<llaminar2::testing::MockDeviceContext> ctx_;
};

TEST_F(Test__MoEExpertDispatchStage, PrefillTwoTiersRoutesEveryContributionOnceAndStableTokenRows)
{
    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 3.0f,
                                                    2.0f, 1.0f,
                                                    1.0f, 0.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.25f, 0.75f,
                                                    0.40f, 0.60f,
                                                    0.55f, 0.45f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);
    MoEExpertDispatchStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_DISPATCH);
    EXPECT_EQ(output.seq_len, seq_len);
    EXPECT_EQ(output.top_k, top_k);
    EXPECT_EQ(output.d_model, 8);
    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].tier_name, "hot");
    EXPECT_EQ(output.tiers[0].domain, "gpu_hot");
    EXPECT_EQ(output.tiers[1].tier_name, "cold");
    EXPECT_EQ(output.tiers[1].domain, "cpu_cold");

    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[0].entries.size(), 3u);
    EXPECT_EQ(output.tiers[1].entries.size(), 3u);

    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.25f);
    expectContributionExactlyOnce(output, 1, 0, 1, 3, 0.75f);
    expectContributionExactlyOnce(output, 0, 1, 0, 2, 0.40f);
    expectContributionExactlyOnce(output, 1, 1, 1, 1, 0.60f);
    expectContributionExactlyOnce(output, 1, 2, 0, 1, 0.55f);
    expectContributionExactlyOnce(output, 0, 2, 1, 0, 0.45f);
}

TEST_F(
    Test__MoEExpertDispatchStage,
    GraphFrozenOwnerMapPublishesExactParticipantTargetsForRankBatching)
{
    auto plan = dynamicResidencyPlan();
    plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
    plan.placements = {placement({0, 0, 1, 1})};
    auto owner_map = std::make_shared<const MoEExpertOwnerMap>(
        MoEExpertOwnerMap::build(plan));

    auto indices = routingTensor({1, 2}, {0.0f, 2.0f});
    auto weights = routingTensor({1, 2}, {0.25f, 0.75f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        indices.get(),
        weights.get(),
        1,
        2,
        plan.placements.front(),
        plan.routed_tiers,
        &output);
    params.owner_map = owner_map;
    MoEExpertDispatchStage stage(std::move(params));

    ASSERT_TRUE(stage.execute(ctx_.get()));
    ASSERT_EQ(output.tiers.size(), 2u);
    ASSERT_EQ(output.tiers[0].entries.size(), 1u);
    ASSERT_EQ(output.tiers[1].entries.size(), 1u);
    const auto *expert_zero_owner = owner_map->ownerFor(4, 0);
    const auto *expert_two_owner = owner_map->ownerFor(4, 2);
    ASSERT_NE(expert_zero_owner, nullptr);
    ASSERT_NE(expert_two_owner, nullptr);
    EXPECT_EQ(
        output.tiers[0].entries[0].destination_participant,
        expert_zero_owner->owner_participant);
    EXPECT_EQ(
        output.tiers[1].entries[0].destination_participant,
        expert_two_owner->owner_participant);
    EXPECT_GE(output.tiers[0].entries[0].destination_participant, 0);
    EXPECT_GE(output.tiers[1].entries[0].destination_participant, 0);
}

TEST_F(
    Test__MoEExpertDispatchStage,
    LiveResidencyLeaseRoutesOneEpochWhileMigrationPublishesNext)
{
    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 5;
    histogram_config.num_experts = 4;
    histogram_config.top_k = 1;
    histogram_config.window_size = 4;
    histogram_config.sockets = {DeviceId::cuda(0), DeviceId::cpu()};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        5,
        2,
        {0, 0, 1, 1});
    auto histogram =
        std::make_unique<DecodeExpertHistogram>(histogram_config);
    const std::vector<uint64_t> counts{1, 100, 90, 0};
    histogram->mergeLayerCounts(
        4,
        counts.data(),
        static_cast<int>(counts.size()),
        false);

    MoERoutedExpertModelMetadata metadata;
    metadata.num_layers = 5;
    metadata.num_experts = 4;
    metadata.d_model = 8;
    metadata.routed_intermediate_size = 4;
    metadata.routed_quant_type = "F32";
    auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
        MoEOverlayResidencyAuthority::Config{
            .initial_plan = dynamicResidencyPlan(),
            .model_metadata = metadata,
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .perf_device = "CUDA:0",
        });

    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        /*layer_idx=*/4,
        /*bucket_rows=*/1,
        /*top_k=*/1,
        /*d_model=*/8,
        DeviceId::cpu(),
        /*workspace_generation=*/41);
    auto &ticket = ticket_storage->ticket();
    ticket.header->logical_row_count = 1;
    ticket.routing_indices_fp32[0] = 2.0f;
    ticket.routing_weights_fp32[0] = 1.0f;

    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        nullptr,
        nullptr,
        1,
        1,
        placement({0, 0, 1, 1}),
        dynamicResidencyPlan().routed_tiers,
        &output);
    params.continuation_domain = "gpu_hot";
    params.residency_authority = authority;
    params.ticket_storage = ticket_storage;
    MoEExpertDispatchStage stage(params);
    EXPECT_TRUE(stage.hasMoEOverlayCollectiveRuntimeParams());
    stage.updateMoEOverlayCollectiveRuntimeParams({
        .generation_id = 7u,
        .step_id = 11u,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::GroupedVerifier,
        .mtp_depth = 2,
        .placement_epoch = 1u,
    });

    ASSERT_TRUE(stage.execute(ctx_.get()));
    ASSERT_EQ(output.residency_epoch, 1u);
    EXPECT_EQ(ticket.header->residency_epoch, 1u);
    ASSERT_NE(output.residency_lease, nullptr);
    EXPECT_EQ(authority->activeTicketCount(), 1u);
    expectContributionExactlyOnce(output, 1, 0, 0, 2, 1.0f);
    ASSERT_EQ(output.tiers[1].entries.size(), 1u);
    const int epoch_one_destination =
        output.tiers[1].entries[0].destination_participant;
    EXPECT_GE(epoch_one_destination, 0);

    AcceptingResidencyTransport transport;
    const auto transaction = authority->proposeFromHistogram();
    ASSERT_EQ(transaction.migrations.size(), 2u);
    const auto started = authority->beginApply(transaction, transport);
    EXPECT_EQ(started.status, MoEOverlayResidencyApplyStatus::Started);
    EXPECT_EQ(transport.stage_calls, 1);
    EXPECT_EQ(authority->activeTicketCount(), 1u);
    EXPECT_EQ(output.residency_epoch, 1u);

    const auto preparing = authority->advanceBackground();
    EXPECT_EQ(
        preparing.status,
        MoEOverlayResidencyApplyStatus::Preparing);
    EXPECT_EQ(authority->snapshot()->epoch, 1u);

    const auto publishing = authority->advanceBackground();
    ASSERT_EQ(
        publishing.status,
        MoEOverlayResidencyApplyStatus::Publishing)
        << publishing.error;
    EXPECT_EQ(authority->snapshot()->epoch, 1u);

    const auto published = authority->advanceBackground();
    ASSERT_EQ(
        published.status,
        MoEOverlayResidencyApplyStatus::Published)
        << published.error;
    EXPECT_EQ(authority->snapshot()->epoch, 2u);
    EXPECT_EQ(output.residency_epoch, 1u)
        << "The in-flight dispatch keeps its exact immutable owner map";
    EXPECT_EQ(ticket.header->residency_epoch, 1u);
    EXPECT_EQ(authority->activeTicketCount(), 1u);
    EXPECT_EQ(transport.stage_calls, 1);
    EXPECT_EQ(transport.prepare_calls, 1);
    EXPECT_EQ(transport.publication_calls, 1);
    EXPECT_EQ(transport.retire_calls, 0);

    ASSERT_TRUE(stage.execute(ctx_.get()));
    EXPECT_EQ(output.residency_epoch, 1u)
        << "Every graph in one MTP sequence must retain its admitted epoch";
    EXPECT_EQ(ticket.header->residency_epoch, 1u);
    expectContributionExactlyOnce(output, 1, 0, 0, 2, 1.0f);

    output.residency_lease.reset();
    ASSERT_EQ(authority->activeTicketCount(), 0u);
    EXPECT_EQ(transport.retire_calls, 0)
        << "Inference return must not run maintenance cleanup";
    EXPECT_EQ(
        authority->advanceBackground().status,
        MoEOverlayResidencyApplyStatus::Idle);
    EXPECT_EQ(transport.retire_calls, 1);

    EXPECT_FALSE(stage.execute(ctx_.get()))
        << "A retired pinned epoch must fail instead of falling forward to current";
    EXPECT_EQ(authority->activeTicketCount(), 0u);

    stage.updateMoEOverlayCollectiveRuntimeParams({
        .generation_id = 7u,
        .step_id = 12u,
        .execution_semantics =
            IComputeStage::MoEOverlayCollectiveRuntimeParams::
                ExecutionSemantics::Decode,
        .mtp_depth = -1,
        .placement_epoch = 2u,
    });
    ASSERT_TRUE(stage.execute(ctx_.get()));
    EXPECT_EQ(output.residency_epoch, 2u);
    EXPECT_EQ(ticket.header->residency_epoch, 2u);
    EXPECT_EQ(authority->activeTicketCount(), 1u);
    expectContributionExactlyOnce(output, 0, 0, 0, 2, 1.0f);
    ASSERT_EQ(output.tiers[0].entries.size(), 1u);
    EXPECT_GE(output.tiers[0].entries[0].destination_participant, 0);
    EXPECT_NE(
        output.tiers[0].entries[0].destination_participant,
        epoch_one_destination)
        << "The next dispatch must consume the newly published epoch owner map";
    output.residency_lease.reset();
    EXPECT_EQ(authority->activeTicketCount(), 0u);
}

TEST_F(Test__MoEExpertDispatchStage, AdvertisesArenaInputContractForHostDispatch)
{
    auto indices = routingTensor({1, 2}, {0.0f, 1.0f});
    auto weights = routingTensor({1, 2}, {0.75f, 0.25f});
    auto hidden = TestTensorFactory::createFP32({1, 8});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 1, 2,
                            placement({0, 1}),
                            {tier("hot", "gpu"), tier("cold", "cpu", true)},
                            &output);
    params.hidden = hidden.get();
    params.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
    params.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
    params.hidden_buffer_id = BufferId::NORMALIZED;

    MoEExpertDispatchStage stage(params);
    const auto contract = stage.bufferContract();

    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_INDICES));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_WEIGHTS));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::NORMALIZED));
}

TEST_F(Test__MoEExpertDispatchStage, ArenaCapacityRoutingBuffersUseLivePrefixRows)
{
    constexpr int seq_len = 2;
    constexpr int capacity = 4;
    constexpr int top_k = 2;
    auto indices = routingTensor({capacity, top_k}, {0.0f, 1.0f,
                                                     2.0f, 3.0f,
                                                     0.0f, 0.0f,
                                                     0.0f, 0.0f});
    auto weights = routingTensor({capacity, top_k}, {0.25f, 0.75f,
                                                     0.40f, 0.60f,
                                                     0.0f, 0.0f,
                                                     0.0f, 0.0f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);
    MoEExpertDispatchStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 2u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1}));
}

TEST_F(Test__MoEExpertDispatchStage, CapturedTicketRoutesOnlyLogicalPrefixAndDeclaresNoTensorCoherenceInputs)
{
    constexpr int bucket_rows = 4;
    constexpr int logical_rows = 2;
    constexpr int top_k = 2;
    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        /*layer_idx=*/4,
        bucket_rows,
        top_k,
        /*d_model=*/8,
        DeviceId::cpu(),
        /*workspace_generation=*/17);
    auto &ticket = ticket_storage->ticket();
    ticket.header->logical_row_count = logical_rows;

    const std::vector<float> indices{
        0.0f, 1.0f,
        2.0f, 3.0f,
        99.0f, 99.0f,
        99.0f, 99.0f,
    };
    const std::vector<float> weights{
        0.25f, 0.75f,
        0.40f, 0.60f,
        123.0f, 123.0f,
        123.0f, 123.0f,
    };
    std::copy(
        indices.begin(), indices.end(), ticket.routing_indices_fp32);
    std::copy(
        weights.begin(), weights.end(), ticket.routing_weights_fp32);

    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        nullptr,
        nullptr,
        bucket_rows,
        top_k,
        placement({0, 1, 0, 1}),
        {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
        &output);
    params.ticket_storage = ticket_storage;
    params.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
    params.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
    params.hidden_buffer_id = BufferId::NORMALIZED;

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.hasMoEOverlayCollectiveRuntimeParams());
    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(output.seq_len, bucket_rows);
    EXPECT_EQ(output.logical_seq_len, logical_rows);
    EXPECT_EQ(output.ticket_lifetime, ticket_storage);
    EXPECT_EQ(output.residency_epoch, 1u);
    EXPECT_EQ(ticket.header->residency_epoch, 1u);
    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 2u);
    EXPECT_TRUE(stage.supportsPaddedPrefillGraphCapturePreflight());
    EXPECT_TRUE(stage.supportsPaddedPrefillRealLengthContract());
    EXPECT_TRUE(stage.bufferContract().inputs.empty());
}

TEST_F(
    Test__MoEExpertDispatchStage,
    CapturedTicketPublishesExactRealPrefillRowsWithoutPaddingContamination)
{
    constexpr int bucket_rows = 4;
    constexpr int logical_rows = 2;
    constexpr int top_k = 2;
    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        /*layer_idx=*/4,
        bucket_rows,
        top_k,
        /*d_model=*/8,
        DeviceId::cpu(),
        /*workspace_generation=*/29);
    auto &ticket = ticket_storage->ticket();
    ticket.header->logical_row_count = logical_rows;

    const std::vector<float> indices{
        3.0f, 1.0f,
        3.0f, 2.0f,
        99.0f, 99.0f,
        99.0f, 99.0f,
    };
    const std::vector<float> weights{
        0.75f, 0.25f,
        0.60f, 0.40f,
        123.0f, 123.0f,
        123.0f, 123.0f,
    };
    std::copy(
        indices.begin(), indices.end(), ticket.routing_indices_fp32);
    std::copy(
        weights.begin(), weights.end(), ticket.routing_weights_fp32);

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 5;
    histogram_config.num_experts = 4;
    histogram_config.top_k = top_k;
    histogram_config.window_size = 8;
    histogram_config.sockets = {DeviceId::cpu()};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        /*num_layers=*/5,
        /*participant_count=*/1,
        {0, 0, 0, 0});
    DecodeExpertHistogram histogram(std::move(histogram_config));

    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        nullptr,
        nullptr,
        bucket_rows,
        top_k,
        placement({0, 1, 0, 1}),
        {tier("tier_0", "gpu_domain"),
         tier("tier_1", "cpu_domain")},
        &output);
    params.ticket_storage = ticket_storage;
    params.routing_evidence_publication =
        MoEExpertDispatchStage::RoutingEvidencePublication{
            .histogram = &histogram,
            .source = ExpertHistogramSource::PrefillChunk,
        };

    MoEExpertDispatchStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(output.residency_epoch, 1u);
    EXPECT_EQ(ticket.header->residency_epoch, 1u);
    EXPECT_EQ(histogram.windowTokenCount(), 2u);
    EXPECT_EQ(
        histogram.layerHistogram(ExpertHistogramSource::PrefillChunk, 4),
        (std::vector<uint64_t>{0, 1, 1, 2}));
    EXPECT_EQ(
        histogram.layerHistogram(ExpertHistogramSource::DecodeToken, 4),
        (std::vector<uint64_t>{0, 0, 0, 0}));
    EXPECT_EQ(
        histogram.layerHistogram(ExpertHistogramSource::GroupedVerifier, 4),
        (std::vector<uint64_t>{0, 0, 0, 0}));
    EXPECT_EQ(
        histogram.layerHistogram(4),
        (std::vector<uint64_t>{0, 1, 1, 2}));

    const auto frozen = histogram.freezeAndRotateWindow();
    ASSERT_TRUE(frozen.valid());
    EXPECT_EQ(frozen.token_count, 2u);
    EXPECT_EQ(frozen.source_token_counts[1], 2u);
    EXPECT_EQ(frozen.source_token_counts[0], 0u);
    EXPECT_EQ(frozen.source_token_counts[2], 0u);
}

TEST_F(
    Test__MoEExpertDispatchStage,
    CapturedTicketRejectsSpeculativeVerifierEvidenceBeforeAcceptedCommit)
{
    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        /*layer_idx=*/4,
        /*bucket_rows=*/2,
        /*top_k=*/1,
        /*d_model=*/8,
        DeviceId::cpu(),
        /*workspace_generation=*/31);

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 5;
    histogram_config.num_experts = 2;
    histogram_config.top_k = 1;
    histogram_config.window_size = 4;
    histogram_config.sockets = {DeviceId::cpu()};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        5,
        1,
        {0, 0});
    DecodeExpertHistogram histogram(std::move(histogram_config));

    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        nullptr,
        nullptr,
        2,
        1,
        placement({0, 1}),
        {tier("tier_0", "domain_0"), tier("tier_1", "domain_1")},
        &output);
    params.ticket_storage = ticket_storage;
    params.routing_evidence_publication =
        MoEExpertDispatchStage::RoutingEvidencePublication{
            .histogram = &histogram,
            .source = ExpertHistogramSource::GroupedVerifier,
        };

    EXPECT_THROW(
        MoEExpertDispatchStage(std::move(params)),
        std::invalid_argument);
}

TEST_F(Test__MoEExpertDispatchStage, CapturedTicketRejectsOutOfRangeLogicalRowCount)
{
    auto ticket_storage =
        std::make_shared<MoEOverlayDispatchTicketStorage>();
    ticket_storage->bindFixedCapacity(
        /*layer_idx=*/4,
        /*bucket_rows=*/4,
        /*top_k=*/2,
        /*d_model=*/8,
        DeviceId::cpu(),
        /*workspace_generation=*/3);
    ticket_storage->ticket().header->logical_row_count = 5;

    MoEExpertDispatchOutput output;
    auto params = paramsFor(
        nullptr,
        nullptr,
        4,
        2,
        placement({0, 1}),
        {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
        &output);
    params.ticket_storage = ticket_storage;

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, DecodeOneTokenWorksNaturally)
{
    constexpr int seq_len = 1;
    constexpr int top_k = 3;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 2.0f, 3.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.10f, 0.20f, 0.70f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 1u);
    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.10f);
    expectContributionExactlyOnce(output, 0, 0, 1, 2, 0.20f);
    expectContributionExactlyOnce(output, 1, 0, 2, 3, 0.70f);
}

TEST_F(Test__MoEExpertDispatchStage, PrefillTransferMetadataMarksNonContinuationTiersSparse)
{
    constexpr int seq_len = 4;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 1.0f,
                                                    2.0f, 3.0f,
                                                    4.0f, 5.0f,
                                                    0.0f, 5.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 0, 1, 1, 2, 2}),
                            {tier("hottest", "cuda_fast"),
                             tier("hot", "rocm_hot"),
                             tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "cuda_fast";

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 3u);
    EXPECT_FALSE(output.tiers[0].transfer_required);
    EXPECT_EQ(output.tiers[0].transfer_mode, MoEExpertTransferMode::None);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{1}));
    EXPECT_TRUE(output.tiers[2].transfer_required);
    EXPECT_EQ(output.tiers[2].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(output.tiers[2].token_rows, (std::vector<int>{2, 3}));

    const size_t hidden_row_bytes = 8u * sizeof(float);
    const size_t routing_row_bytes = top_k * 2u * sizeof(float);
    EXPECT_EQ(output.tiers[2].transfer_volume.outbound_bytes,
              2u * (hidden_row_bytes + routing_row_bytes));
    EXPECT_EQ(output.tiers[2].transfer_volume.return_bytes,
              2u * hidden_row_bytes);
    EXPECT_LT(output.estimatedTransferBytes(), output.denseTransferBytes());
}

TEST_F(Test__MoEExpertDispatchStage, DecodeTransferMetadataUsesOneTokenFastPath)
{
    constexpr int seq_len = 1;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 3.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.25f, 0.75f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 0, 1, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "gpu_hot";

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_FALSE(output.tiers[0].transfer_required);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::DecodeOneToken);
    EXPECT_EQ(output.tiers[1].transfer_volume.outbound_bytes,
              8u * sizeof(float) + top_k * 2u * sizeof(float));
    EXPECT_EQ(output.tiers[1].transfer_volume.return_bytes,
              8u * sizeof(float));
}

TEST_F(Test__MoEExpertDispatchStage, DenseTransferCompatibilityModeReportsFullSequenceBytes)
{
    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 1.0f,
                                                    0.0f, 1.0f,
                                                    0.0f, 1.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "gpu_hot";
    params.transfer_mode = MoEExpertTransferMode::DenseFullSequence;

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::DenseFullSequence);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[1].transfer_volume.totalBytes(),
              output.tiers[1].transfer_volume.denseTotalBytes());
}

TEST_F(Test__MoEExpertDispatchStage, FlatPrefillThreeTiersRoutesEveryContributionOnce)
{
    constexpr int seq_len = 2;
    constexpr int top_k = 3;
    auto indices = routingTensor({seq_len * top_k}, {0.0f, 1.0f, 2.0f,
                                                     3.0f, 4.0f, 5.0f});
    auto weights = routingTensor({seq_len * top_k}, {0.11f, 0.12f, 0.13f,
                                                     0.21f, 0.22f, 0.23f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 2, 0, 1, 2}),
                            {tier("hottest", "cuda_fast"),
                             tier("warm", "rocm_warm"),
                             tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 3u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[2].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 2u);
    EXPECT_EQ(output.tiers[2].entries.size(), 2u);

    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.11f);
    expectContributionExactlyOnce(output, 1, 0, 1, 1, 0.12f);
    expectContributionExactlyOnce(output, 2, 0, 2, 2, 0.13f);
    expectContributionExactlyOnce(output, 0, 1, 0, 3, 0.21f);
    expectContributionExactlyOnce(output, 1, 1, 1, 4, 0.22f);
    expectContributionExactlyOnce(output, 2, 1, 2, 5, 0.23f);
}

TEST_F(Test__MoEExpertDispatchStage, InvalidExpertIdFailsClearly)
{
    auto indices = routingTensor({2, 2}, {0.0f, 1.0f,
                                          99.0f, 2.0f});
    auto weights = routingTensor({2, 2}, {0.1f, 0.2f,
                                          0.3f, 0.4f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 2, 2,
                            placement({0, 1, 0}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, MissingPlacementFailsClearly)
{
    auto indices = routingTensor({1, 2}, {0.0f, 1.0f});
    auto weights = routingTensor({1, 2}, {0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 1, 2,
                            std::nullopt,
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, RejectsTransposedMatrixRoutingShape)
{
    auto indices = routingTensor({2, 3}, {0.0f, 1.0f, 2.0f,
                                          0.0f, 1.0f, 2.0f});
    auto weights = routingTensor({2, 3}, {0.1f, 0.2f, 0.3f,
                                          0.4f, 0.5f, 0.6f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 3, 2,
                            placement({0, 1, 0}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, AdvertisesCpuOnlyBackendSupport)
{
    MoEExpertDispatchOutput output;
    auto params = paramsFor(nullptr, nullptr, 1, 1,
                            placement({0}),
                            {tier("hot", "gpu_hot")},
                            &output);
    MoEExpertDispatchStage stage(params);

    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
}

TEST_F(Test__MoEExpertDispatchStage, FactoryCreatesDispatchStage)
{
    MoEExpertDispatchOutput output;
    MoEExpertDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 1;
    params.top_k = 1;
    params.d_model = 8;
    params.placement = placement({0});
    params.routed_tiers = {tier("hot", "gpu_hot")};
    params.output = &output;

    auto stage = ComputeStageFactory::createMoEExpertDispatch(params);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::MOE_EXPERT_DISPATCH);
    EXPECT_STREQ(computeStageTypeName(stage->type()), "MOE_EXPERT_DISPATCH");
}

} // namespace llaminar2::test
