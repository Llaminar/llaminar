/**
 * @file Test__CPUCurrentBatchLLEP.cpp
 * @brief Lifecycle and publication regressions for production CPU LLEP.
 *
 * These tests exercise the exact begin/restore graph stage used by CPU
 * NodeTP inference.  The collective is scripted so the suite remains a
 * fast, device-free unit gate while still proving deterministic cross-rank
 * agreement, route-slot totality, bounded graph-owned storage, and transient
 * expert residency lifecycle.
 */

#include <gtest/gtest.h>

#include "collective/IGlobalTPContext.h"
#include "execution/compute_stages/stages/MoECPUCurrentBatchLLEPStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "../../utils/TestTensorFactory.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Deterministic two-participant control collective for stage tests. */
    class ScriptedGlobalTPContext final : public IGlobalTPContext
    {
    public:
        explicit ScriptedGlobalTPContext(int participant_id,
                                         int disagree_on_gather_call = 0)
            : participant_id_(participant_id),
              disagree_on_gather_call_(disagree_on_gather_call)
        {
        }

        int degree() const override { return 2; }
        int myIndex() const override { return participant_id_; }
        CollectiveBackendType backend() const override
        {
            return CollectiveBackendType::MPI;
        }
        MPI_Comm communicator() const override { return MPI_COMM_SELF; }
        int domainId() const override { return 17; }
        const std::vector<int> &worldRanks() const override
        {
            return world_ranks_;
        }
        GlobalDeviceAddress localDevice() const override
        {
            return GlobalDeviceAddress::cpu(100 + participant_id_);
        }
        void barrier() const override {}

        bool allgatherBytes(const void *send_data,
                            void *recv_data,
                            size_t byte_count) const override
        {
            if (!send_data || !recv_data || byte_count == 0)
                return false;
            ++gather_calls_;
            auto *output = static_cast<std::byte *>(recv_data);
            std::memcpy(output, send_data, byte_count);
            std::memcpy(output + byte_count, send_data, byte_count);
            if (gather_calls_ == disagree_on_gather_call_)
                output[byte_count] ^= std::byte{1};
            return true;
        }

        bool allreduce(TensorBase *) override { return false; }
        bool broadcast(TensorBase *, int) override { return false; }
        bool allgather(const TensorBase *, TensorBase *) override { return false; }
        bool gatherVariableFloatRecordsToRoot(
            const float *, size_t, float *, size_t, size_t, int,
            size_t &, const std::string &) override
        {
            return false;
        }
        bool broadcastFloatElements(
            TensorBase *, size_t, int, const std::string &) override
        {
            return false;
        }
        bool send(const TensorBase *, int) override { return false; }
        bool recv(TensorBase *, int) override { return false; }

        [[nodiscard]] int gatherCalls() const noexcept { return gather_calls_; }

    private:
        int participant_id_ = 0;
        int disagree_on_gather_call_ = 0;
        mutable int gather_calls_ = 0;
        std::vector<int> world_ranks_{100, 101};
    };

    /** @brief Minimal participant-local endpoint used by the lifecycle fixture. */
    class RecordingExpertConsumer final
        : public ICPUCurrentBatchLLEPExpertConsumer
    {
    public:
        explicit RecordingExpertConsumer(int participant_id)
            : participant_id_(participant_id)
        {
        }

        int cpuCurrentBatchLLEPLayerIndex() const noexcept override
        {
            return 3;
        }

        int cpuCurrentBatchLLEPParticipantId() const noexcept override
        {
            return participant_id_;
        }

        bool cpuCurrentBatchLLEPUsesCPU() const noexcept override
        {
            return true;
        }

        ExpertPackedWeights cloneCPUCurrentBatchLLEPPreparedExpert(
            int,
            uint64_t) const override
        {
            throw std::logic_error(
                "The recording executor does not request packed weights");
        }

        bool installCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state,
            const std::unordered_map<int, PreparedExpertEngines> *) override
        {
            if (state.active || state.layer_idx != 3 ||
                state.participant_id != participant_id_ || installed)
            {
                return false;
            }
            installed = true;
            ++install_calls;
            return true;
        }

        bool discardCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state) noexcept override
        {
            if (!installed || state.layer_idx != 3 ||
                state.participant_id != participant_id_)
            {
                return false;
            }
            installed = false;
            ++discard_calls;
            return true;
        }

        bool installed = false;
        int install_calls = 0;
        int discard_calls = 0;

    private:
        int participant_id_ = -1;
    };

    /** @brief Records the exact state presented to the physical executor. */
    class RecordingPhysicalExecutor final
        : public ICPUCurrentBatchLLEPPhysicalExecutor
    {
    public:
        bool materializeCPUCurrentBatchLLEPTransaction(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) override
        {
            ++begin_calls;
            observed_participant = tp_ctx.myIndex();
            observed_transfer_count = state.status.weight_transfer_count;
            observed_transient_mask = state.transient_resident_mask;
            if (state.active ||
                !expert_consumer
                     .installCPUCurrentBatchLLEPTransientResidency(
                         state, nullptr))
                return false;
            state.active = true;
            ++state.transaction_epoch;
            return true;
        }

        bool restoreCPUCurrentBatchLLEPPhysicalState(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) override
        {
            ++restore_calls;
            observed_participant = tp_ctx.myIndex();
            if (!state.active ||
                !expert_consumer
                     .discardCPUCurrentBatchLLEPTransientResidency(state))
                return false;
            state.transient_resident_mask = state.owner_mask;
            state.active = false;
            return true;
        }

        int begin_calls = 0;
        int restore_calls = 0;
        int observed_participant = -1;
        uint32_t observed_transfer_count = 0;
        std::vector<bool> observed_transient_mask;
    };

    /** @brief Build the one-domain authority used by the CPU child protocol. */
    std::shared_ptr<MoEOverlayResidencyAuthority> makeResidencyAuthority(
        RoutedExpertOwnerOrder owner_order)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        plan.continuation_domain = "cpu_domain";
        plan.base_model_domain = "cpu_domain";
        plan.shared_expert_domain = "cpu_domain";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.owner_order = owner_order;

        RoutedExpertDomain domain;
        domain.name = "cpu_domain";
        domain.scope = ExecutionDomainScope::NODE_LOCAL;
        domain.backend = CollectiveBackendType::MPI;
        domain.participants = {
            GlobalDeviceAddress::cpu(100),
            GlobalDeviceAddress::cpu(101),
        };
        domain.world_ranks = {100, 101};
        domain.owner_rank = 100;
        domain.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        plan.domains = {std::move(domain)};

        RoutedExpertTier tier;
        tier.name = "priority_0";
        tier.domain = "cpu_domain";
        tier.priority = 0;
        tier.fallback = true;
        plan.routed_tiers = {std::move(tier)};

        return std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = std::move(plan),
                .model_metadata = {
                    .num_layers = 4,
                    .num_experts = 4,
                    .d_model = 8,
                    .routed_intermediate_size = 8,
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "cpu_llep_unit",
            });
    }

    /** @brief Own all objects whose addresses are bound into one graph stage. */
    struct StageFixture
    {
        static constexpr int kSeqLen = 50;
        static constexpr int kTopK = 2;
        static constexpr int kExpertCount = 4;

        StageFixture(std::vector<uint32_t> owners,
                     int participant_id,
                     int disagree_on_gather_call = 0)
            : context(participant_id, disagree_on_gather_call),
              residency_authority(makeResidencyAuthority(
                  owners == std::vector<uint32_t>{0, 0, 1, 1}
                      ? RoutedExpertOwnerOrder::Ordinal
                      : RoutedExpertOwnerOrder::Random)),
              indices(TestTensorFactory::createFP32(
                  {kSeqLen, kTopK})),
              weights(TestTensorFactory::createFP32(
                  {kSeqLen, kTopK})),
              expert_consumer(participant_id)
        {
            for (int route = 0; route < kSeqLen * kTopK; ++route)
            {
                indices->mutable_data()[route] = route < 80 ? 0.0f : 1.0f;
                weights->mutable_data()[route] = 1.0f;
            }

            least_loaded_ep::LeastLoadedExpertAssignmentConfig planner;
            planner.enable_balanced_skip = false;
            planner.lambda_numerator = 13;
            planner.lambda_denominator = 10;
            planner.alpha_numerator = 1;
            planner.alpha_denominator = 1;
            planner.min_chunk_tokens = 0;

            MoECPUCurrentBatchLLEPStage::Params begin_params;
            begin_params.device_id = DeviceId::cpu();
            begin_params.phase = CPUCurrentBatchLLEPPhase::Begin;
            begin_params.physical_executor = &physical_executor;
            begin_params.residency_authority = residency_authority;
            begin_params.residency_domain = "cpu_domain";
            begin_params.tp_ctx = &context;
            begin_params.expert_consumer = &expert_consumer;
            begin_params.routing_indices = indices.get();
            begin_params.routing_weights = weights.get();
            begin_params.layer_idx = 3;
            begin_params.seq_len = kSeqLen;
            begin_params.top_k = kTopK;
            begin_params.num_experts = kExpertCount;
            begin_params.participant_count = 2;
            begin_params.participant_id = participant_id;
            begin_params.planner_config = planner;
            begin = std::make_unique<MoECPUCurrentBatchLLEPStage>(
                std::move(begin_params));

            MoECPUCurrentBatchLLEPStage::Params restore_params;
            restore_params.device_id = DeviceId::cpu();
            restore_params.phase = CPUCurrentBatchLLEPPhase::Restore;
            restore_params.physical_executor = &physical_executor;
            restore_params.residency_authority = residency_authority;
            restore_params.residency_domain = "cpu_domain";
            restore_params.tp_ctx = &context;
            restore_params.expert_consumer = &expert_consumer;
            restore_params.state = begin->state();
            restore_params.layer_idx = 3;
            restore_params.participant_count = 2;
            restore_params.participant_id = participant_id;
            restore = std::make_unique<MoECPUCurrentBatchLLEPStage>(
                std::move(restore_params));
        }

        CPUDeviceContext device_context{DeviceId::cpu(), 1};
        ScriptedGlobalTPContext context;
        std::shared_ptr<MoEOverlayResidencyAuthority> residency_authority;
        RecordingPhysicalExecutor physical_executor;
        std::unique_ptr<FP32Tensor> indices;
        std::unique_ptr<FP32Tensor> weights;
        RecordingExpertConsumer expert_consumer;
        std::unique_ptr<MoECPUCurrentBatchLLEPStage> begin;
        std::unique_ptr<MoECPUCurrentBatchLLEPStage> restore;
    };

    void expectExactBalancedPublication(StageFixture &fixture,
                                        uint32_t transferred_expert)
    {
        const auto state = fixture.begin->state();
        ASSERT_NE(state, nullptr);

        const auto *destination_storage =
            state->destination_by_flat_route.data();
        const auto *span_storage = state->spans.data();
        const auto *transfer_storage = state->transfers.data();
        const size_t destination_capacity =
            state->destination_by_flat_route.size();
        const size_t span_capacity = state->spans.size();
        const size_t transfer_capacity = state->transfers.size();

        ASSERT_TRUE(fixture.begin->execute(&fixture.device_context));
        EXPECT_TRUE(state->active);
        EXPECT_EQ(state->transaction_epoch, 1u);
        EXPECT_EQ(fixture.context.gatherCalls(), 2);
        EXPECT_EQ(fixture.physical_executor.begin_calls, 1);
        ASSERT_TRUE(state->durable_epoch_lease.has_value());
        EXPECT_EQ(state->durable_parent_epoch, 1u);
        EXPECT_EQ(state->durable_domain, "cpu_domain");
        ASSERT_EQ(state->status.weight_transfer_count, 1u);
        EXPECT_EQ(state->status.critical_path_transfer_slots, 1u);
        EXPECT_EQ(state->transfers[0].expert, transferred_expert);
        EXPECT_EQ(state->assigned_load[0], 50u);
        EXPECT_EQ(state->assigned_load[1], 50u);

        EXPECT_EQ(state->destination_by_flat_route.data(), destination_storage);
        EXPECT_EQ(state->spans.data(), span_storage);
        EXPECT_EQ(state->transfers.data(), transfer_storage);
        EXPECT_EQ(state->destination_by_flat_route.size(),
                  destination_capacity);
        EXPECT_EQ(state->spans.size(), span_capacity);
        EXPECT_EQ(state->transfers.size(), transfer_capacity);

        EXPECT_EQ(
            std::count(state->destination_by_flat_route.begin(),
                       state->destination_by_flat_route.end(), 0u),
            50);
        EXPECT_EQ(
            std::count(state->destination_by_flat_route.begin(),
                       state->destination_by_flat_route.end(), 1u),
            50);
        for (size_t route = 0; route < state->routeSlotCount(); ++route)
            EXPECT_LT(state->destinationForFlatRoute(route), 2u);

        EXPECT_FALSE(fixture.begin->execute(&fixture.device_context))
            << "A graph may not overwrite an active route publication";
        EXPECT_EQ(fixture.physical_executor.begin_calls, 1);

        ASSERT_TRUE(fixture.restore->execute(&fixture.device_context));
        EXPECT_FALSE(state->active);
        EXPECT_EQ(state->transient_resident_mask, state->owner_mask);
        EXPECT_EQ(fixture.physical_executor.restore_calls, 1);
        EXPECT_EQ(fixture.expert_consumer.install_calls, 1);
        EXPECT_EQ(fixture.expert_consumer.discard_calls, 1);
        EXPECT_FALSE(fixture.expert_consumer.installed);
        EXPECT_FALSE(state->durable_epoch_lease.has_value());
        EXPECT_EQ(state->durable_parent_epoch, 0u);
        EXPECT_TRUE(state->durable_domain.empty());
        EXPECT_EQ(
            fixture.residency_authority->stats()
                .current_batch_llep_leases_acquired,
            1u);
        EXPECT_EQ(
            fixture.residency_authority->stats()
                .current_batch_llep_leases_released,
            1u);
        EXPECT_THROW((void)state->destinationForFlatRoute(0),
                     std::logic_error);

        EXPECT_FALSE(fixture.restore->execute(&fixture.device_context))
            << "Every begin transaction has exactly one restore";
        EXPECT_EQ(fixture.physical_executor.restore_calls, 1);
    }
} // namespace

TEST(Test__CPUCurrentBatchLLEP,
     OrdinalOwnersPublishEveryRouteAndRestoreOwnerOnlyResidency)
{
    StageFixture fixture({0, 0, 1, 1}, /*participant_id=*/1);
    expectExactBalancedPublication(fixture, /*transferred_expert=*/0);
    EXPECT_TRUE(fixture.physical_executor.observed_transient_mask[0]);
}

TEST(Test__CPUCurrentBatchLLEP,
     RandomOwnersUseTheSameDeterministicTransactionContract)
{
    StageFixture fixture({1, 0, 1, 0}, /*participant_id=*/0);
    expectExactBalancedPublication(fixture, /*transferred_expert=*/0);
    EXPECT_TRUE(fixture.physical_executor.observed_transient_mask[0]);
}

TEST(Test__CPUCurrentBatchLLEP,
     RouterPublicationDisagreementFailsBeforePlanningOrTransfer)
{
    StageFixture fixture({0, 0, 1, 1}, /*participant_id=*/0,
                         /*disagree_on_gather_call=*/1);

    EXPECT_FALSE(fixture.begin->execute(&fixture.device_context));
    EXPECT_EQ(fixture.context.gatherCalls(), 1);
    EXPECT_EQ(fixture.physical_executor.begin_calls, 0);
    EXPECT_FALSE(fixture.begin->state()->active);
}

TEST(Test__CPUCurrentBatchLLEP,
     AssignmentPlanDisagreementFailsBeforePhysicalPublication)
{
    StageFixture fixture({0, 0, 1, 1}, /*participant_id=*/0,
                         /*disagree_on_gather_call=*/2);

    EXPECT_FALSE(fixture.begin->execute(&fixture.device_context));
    EXPECT_EQ(fixture.context.gatherCalls(), 2);
    EXPECT_EQ(fixture.physical_executor.begin_calls, 0);
    EXPECT_FALSE(fixture.begin->state()->active);
}

TEST(Test__CPUCurrentBatchLLEP,
     RootDispatchConsumesTheExactActiveChildPublication)
{
    StageFixture fixture({0, 0, 1, 1}, /*participant_id=*/0);
    ASSERT_TRUE(fixture.begin->execute(&fixture.device_context));
    const auto state = fixture.begin->state();
    ASSERT_NE(state, nullptr);
    ASSERT_TRUE(state->durable_epoch_lease.has_value());
    const auto &snapshot = **state->durable_epoch_lease;
    ASSERT_NE(snapshot.placement_plan, nullptr);
    const auto placement = std::find_if(
        snapshot.placement_plan->placements.begin(),
        snapshot.placement_plan->placements.end(),
        [](const RoutedExpertLayerPlacement &candidate)
        { return candidate.layer == 3; });
    ASSERT_NE(placement, snapshot.placement_plan->placements.end());

    MoEExpertDispatchOutput output;
    MoEExpertDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.routing_indices = fixture.indices.get();
    params.routing_weights = fixture.weights.get();
    params.hidden = fixture.indices.get();
    params.seq_len = StageFixture::kSeqLen;
    params.top_k = StageFixture::kTopK;
    params.d_model = 1;
    params.continuation_domain = "cpu_domain";
    params.placement = *placement;
    params.routed_tiers = snapshot.placement_plan->routed_tiers;
    params.residency_authority = fixture.residency_authority;
    params.cpu_current_batch_llep_state = state;
    params.output = &output;
    MoEExpertDispatchStage dispatch(std::move(params));

    ASSERT_TRUE(dispatch.execute(&fixture.device_context));
    ASSERT_EQ(output.tiers.size(), 1u);
    ASSERT_EQ(output.tiers.front().entries.size(), state->routeSlotCount());
    for (const auto &entry : output.tiers.front().entries)
    {
        const size_t flat_route =
            static_cast<size_t>(entry.token_row) *
                static_cast<size_t>(StageFixture::kTopK) +
            static_cast<size_t>(entry.route_slot);
        EXPECT_EQ(
            entry.destination_participant,
            static_cast<int>(state->destinationForFlatRoute(flat_route)));
    }
    EXPECT_EQ(output.residency_epoch, state->durable_parent_epoch);
    EXPECT_EQ(output.residency_lease, nullptr)
        << "The child state, not a second dispatch lease, owns the epoch";

    ASSERT_TRUE(fixture.restore->execute(&fixture.device_context));
}
