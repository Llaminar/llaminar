/**
 * @file Test__MoEGraphNative_CPUCold_MPI.cpp
 * @brief Graph-native CPU-cold sparse MoE overlay MPI integration test.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kSeqLen = 3;
        constexpr int kDModel = 8;
        constexpr int kIntermediate = 4;
        constexpr int kNumExperts = 4;
        constexpr int kTopK = 2;
        constexpr int kLayer = 0;
        constexpr int kDomain = 7505;

        std::shared_ptr<FP32Tensor> fp32(std::vector<size_t> shape)
        {
            return std::make_shared<FP32Tensor>(std::move(shape));
        }

        void fillHidden(FP32Tensor *tensor)
        {
            float *data = tensor->mutable_data();
            for (int row = 0; row < kSeqLen; ++row)
            {
                for (int col = 0; col < kDModel; ++col)
                {
                    data[static_cast<size_t>(row) * kDModel + col] =
                        0.05f * static_cast<float>(row + 1) +
                        0.01f * static_cast<float>(col + 1);
                }
            }
        }

        void fillRouting(FP32Tensor *indices, FP32Tensor *weights)
        {
            const float route_indices[] = {
                0.0f,
                1.0f,
                2.0f,
                3.0f,
                1.0f,
                2.0f,
            };
            const float route_weights[] = {
                0.60f,
                0.40f,
                0.25f,
                0.75f,
                0.55f,
                0.45f,
            };

            std::copy(std::begin(route_indices), std::end(route_indices), indices->mutable_data());
            std::copy(std::begin(route_weights), std::end(route_weights), weights->mutable_data());
        }

        void fillExpertTensor(FP32Tensor *tensor, float scale)
        {
            const auto &shape = tensor->shape();
            ASSERT_EQ(shape.size(), 3u);
            const size_t cols = shape[0];
            const size_t rows = shape[1];
            const size_t experts = shape[2];
            float *data = tensor->mutable_data();

            for (size_t expert = 0; expert < experts; ++expert)
            {
                for (size_t row = 0; row < rows; ++row)
                {
                    for (size_t col = 0; col < cols; ++col)
                    {
                        const size_t offset = expert * rows * cols + row * cols + col;
                        data[offset] = scale * static_cast<float>(expert + 1) +
                                       0.003f * static_cast<float>(row + 1) +
                                       0.0007f * static_cast<float>(col + 1);
                    }
                }
            }
        }

        struct ExpertWeights
        {
            std::shared_ptr<FP32Tensor> gate;
            std::shared_ptr<FP32Tensor> up;
            std::shared_ptr<FP32Tensor> down;
        };

        ExpertWeights makeWeights()
        {
            ExpertWeights weights;
            weights.gate = fp32({kDModel, kIntermediate, kNumExperts});
            weights.up = fp32({kDModel, kIntermediate, kNumExperts});
            weights.down = fp32({kIntermediate, kDModel, kNumExperts});
            fillExpertTensor(weights.gate.get(), 0.010f);
            fillExpertTensor(weights.up.get(), 0.012f);
            fillExpertTensor(weights.down.get(), 0.008f);
            return weights;
        }

        ExpertComputeDomain hotDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_hot";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {0};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuColdDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::NodeLocalTP;
            domain.backend = CollectiveBackendType::UPI;
            domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
            domain.world_ranks = {1, 2};
            domain.owner_rank = 1;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(const std::string &name,
                                    const std::string &domain,
                                    int priority,
                                    bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.fallback = fallback;
            return tier;
        }

        MoEExpertParallelPlan makeCpuColdPlan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "cpu_hot";
            plan.shared_expert_domain = "cpu_hot";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {hotDomain(), cpuColdDomain()};
            plan.routed_tiers = {
                routedTier("hot", "cpu_hot", 0),
                routedTier("cold", "cpu_cold", 1, true),
            };
            plan.placements = {
                ExpertLayerPlacement{.layer = kLayer,
                                     .routed_expert_tier = {0, 0, 1, 1}},
            };
            return plan;
        }

        MoEOverlayCollectiveKey keyFor(int tier,
                                       int target_participant,
                                       MoEOverlayCollectiveDirection direction,
                                       uint64_t base_sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 51;
            key.step_id = 6;
            key.layer_idx = kLayer;
            key.tier_idx = tier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = base_sequence + static_cast<uint64_t>(tier * 10 + target_participant);
            return key;
        }

        bool runReference(IDeviceContext *ctx,
                          TensorBase *input,
                          TensorBase *routing_indices,
                          TensorBase *routing_weights,
                          const ExpertWeights &weights,
                          TensorBase *output,
                          std::vector<bool> expert_mask)
        {
            MoEExpertComputeStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = input;
            params.seq_len = kSeqLen;
            params.d_model = kDModel;
            params.num_experts = kNumExperts;
            params.top_k = kTopK;
            params.gate_exps = weights.gate.get();
            params.up_exps = weights.up.get();
            params.down_exps = weights.down.get();
            params.expert_intermediate = kIntermediate;
            params.layer_idx = kLayer;
            params.expert_mask = std::move(expert_mask);
            params.routing_indices = routing_indices;
            params.routing_weights = routing_weights;
            params.output = output;
            params.output_registered_in_arena = false;

            if (!MoEExpertComputeStage::extractExpertViews(params) ||
                !MoEExpertComputeStage::prepareExpertGemmEngines(params))
            {
                return false;
            }

            MoEExpertComputeStage stage(std::move(params));
            return stage.execute(ctx);
        }

        MoEExpertDispatchOutput buildDispatch(IDeviceContext *ctx,
                                              TensorBase *hidden,
                                              TensorBase *routing_indices,
                                              TensorBase *routing_weights,
                                              const MoEExpertParallelPlan &plan)
        {
            MoEExpertDispatchOutput dispatch;
            MoEExpertDispatchStage::Params params;
            params.device_id = DeviceId::cpu();
            params.routing_indices = routing_indices;
            params.routing_weights = routing_weights;
            params.hidden = hidden;
            params.seq_len = kSeqLen;
            params.top_k = kTopK;
            params.d_model = kDModel;
            params.continuation_domain = plan.continuation_domain;
            params.placement = plan.placements.front();
            params.routed_tiers = plan.routed_tiers;
            params.output = &dispatch;

            MoEExpertDispatchStage stage(std::move(params));
            EXPECT_TRUE(stage.execute(ctx));
            return dispatch;
        }

        MoEExpertTierDispatch filterTierForParticipant(const MoEExpertTierDispatch &tier,
                                                       const MoEExpertOwnerMap &owner_map,
                                                       int owner_participant)
        {
            MoEExpertTierDispatch filtered;
            filtered.tier_index = tier.tier_index;
            filtered.tier_name = tier.tier_name;
            filtered.domain = tier.domain;
            filtered.fallback = tier.fallback;
            filtered.transfer_required = tier.transfer_required;
            filtered.transfer_mode = tier.transfer_mode;

            std::vector<unsigned char> seen_rows(static_cast<size_t>(kSeqLen), 0);
            for (const auto &entry : tier.entries)
            {
                const auto *owner = owner_map.ownerFor(kLayer, entry.expert_id);
                if (!owner || owner->owner_participant != owner_participant)
                    continue;

                filtered.entries.push_back(entry);
                auto &seen = seen_rows[static_cast<size_t>(entry.token_row)];
                if (!seen)
                {
                    filtered.token_rows.push_back(entry.token_row);
                    seen = 1;
                }
            }

            filtered.transfer_volume = MoEExpertTokenRowTransfer::estimateVolume(
                kSeqLen,
                kTopK,
                kDModel,
                filtered.token_rows.size(),
                filtered.transfer_mode);
            return filtered;
        }

        void expectDispatchTargetsOwners(const MoEExpertDispatchOutput &dispatch,
                                         const MoEExpertOwnerMap &owner_map)
        {
            ASSERT_EQ(dispatch.tiers.size(), 2u);
            for (size_t tier = 0; tier < dispatch.tiers.size(); ++tier)
            {
                const auto participant_ids = owner_map.participantIdsForTier(static_cast<int>(tier));
                ASSERT_FALSE(participant_ids.empty());
                for (const auto &entry : dispatch.tiers[tier].entries)
                {
                    const auto *owner = owner_map.ownerFor(kLayer, entry.expert_id);
                    ASSERT_NE(owner, nullptr);
                    EXPECT_EQ(owner->tier_idx, static_cast<int>(tier));
                    EXPECT_NE(std::find(participant_ids.begin(), participant_ids.end(), owner->owner_participant),
                              participant_ids.end());
                }
            }
        }

        void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
        {
            ASSERT_EQ(actual->numel(), expected->numel());
            const float *actual_data = actual->data();
            const float *expected_data = expected->data();
            for (size_t i = 0; i < actual->numel(); ++i)
                EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
        }

        void addTensor(const TensorBase *lhs, const TensorBase *rhs, TensorBase *out)
        {
            ASSERT_EQ(lhs->numel(), rhs->numel());
            ASSERT_EQ(lhs->numel(), out->numel());
            const float *lhs_data = lhs->data();
            const float *rhs_data = rhs->data();
            float *out_data = out->mutable_data();
            for (size_t index = 0; index < out->numel(); ++index)
                out_data[index] = lhs_data[index] + rhs_data[index];
        }

    } // namespace

    class Test__MoEGraphNative_CPUCold_MPI : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            if (world_size_ != 3)
            {
                GTEST_SKIP() << "Test requires exactly 3 MPI ranks: one continuation root plus two CPU cold participants (got "
                             << world_size_ << ")";
            }

            cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            collective_ = std::make_unique<MoEOverlayMPISparseCollectiveContext>(
                MoEOverlayMPISparseCollectiveContext::Config{.mpi_ctx = mpi_ctx_, .local_participant_id = rank_});
            workspace_.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
            workspace_.resetForStep(51, 6);
        }

        void TearDown() override
        {
            if (world_size_ == 3)
                MPI_Barrier(MPI_COMM_WORLD);
        }

        int rank_ = -1;
        int world_size_ = 0;
        std::unique_ptr<llaminar2::testing::MockDeviceContext> cpu_ctx_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        MoEOverlayCollectiveWorkspace workspace_;
        std::unique_ptr<MoEOverlayMPISparseCollectiveContext> collective_;
    };

    TEST_F(Test__MoEGraphNative_CPUCold_MPI, SparseDispatchLocalExpertReturnReduceMatchesMaskedReference)
    {
        const auto plan = makeCpuColdPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        const auto *hot_owner = owner_map.ownerFor(kLayer, 0);
        ASSERT_NE(hot_owner, nullptr);
        EXPECT_EQ(hot_owner->owner_participant, 0);
        EXPECT_EQ(hot_owner->owner_world_rank, 0);

        const auto *cold_owner_2 = owner_map.ownerFor(kLayer, 2);
        ASSERT_NE(cold_owner_2, nullptr);
        EXPECT_EQ(cold_owner_2->owner_participant, 1);
        EXPECT_EQ(cold_owner_2->owner_world_rank, 1);

        const auto *cold_owner_3 = owner_map.ownerFor(kLayer, 3);
        ASSERT_NE(cold_owner_3, nullptr);
        EXPECT_EQ(cold_owner_3->owner_participant, 2);
        EXPECT_EQ(cold_owner_3->owner_world_rank, 2);

        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        auto full_reference = fp32({kSeqLen, kDModel});
        auto hot_reference = fp32({kSeqLen, kDModel});
        auto cold_reference = fp32({kSeqLen, kDModel});
        auto hot_overlay = fp32({kSeqLen, kDModel});
        auto cold_overlay = fp32({kSeqLen, kDModel});
        auto combined_overlay = fp32({kSeqLen, kDModel});
        std::fill_n(hot_overlay->mutable_data(), hot_overlay->numel(), 0.0f);
        std::fill_n(cold_overlay->mutable_data(), cold_overlay->numel(), 0.0f);
        std::fill_n(combined_overlay->mutable_data(), combined_overlay->numel(), 0.0f);

        ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                 full_reference.get(), {true, true, true, true}));
        ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                 hot_reference.get(), {true, true, false, false}));
        ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights,
                                 cold_reference.get(), {false, false, true, true}));

        const auto dispatch = buildDispatch(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan);
        expectDispatchTargetsOwners(dispatch, owner_map);
        EXPECT_EQ(dispatch.tiers[0].token_rows, (std::vector<int>{0, 2}));
        EXPECT_EQ(dispatch.tiers[1].token_rows, (std::vector<int>{1, 2}));
        EXPECT_LT(dispatch.tiers[1].transfer_volume.totalBytes(), dispatch.tiers[1].transfer_volume.denseTotalBytes());

        const auto local_expert_mask = owner_map.expertMaskForParticipant(kLayer, rank_, kNumExperts);
        ASSERT_EQ(local_expert_mask.size(), static_cast<size_t>(kNumExperts));

        for (int tier = 0; tier < 2; ++tier)
        {
            const auto participant_ids = owner_map.participantIdsForTier(tier);
            ASSERT_FALSE(participant_ids.empty());

            for (int target_participant : participant_ids)
            {
                const auto filtered_dispatch = filterTierForParticipant(
                    dispatch.tiers[static_cast<size_t>(tier)],
                    owner_map,
                    target_participant);

                if (tier == 1 && target_participant == 1)
                {
                    EXPECT_EQ(filtered_dispatch.token_rows, (std::vector<int>{1, 2}));
                    ASSERT_EQ(filtered_dispatch.entries.size(), 2u);
                    EXPECT_EQ(filtered_dispatch.entries[0].expert_id, 2);
                    EXPECT_EQ(filtered_dispatch.entries[1].expert_id, 2);
                }
                if (tier == 1 && target_participant == 2)
                {
                    EXPECT_EQ(filtered_dispatch.token_rows, (std::vector<int>{1}));
                    ASSERT_EQ(filtered_dispatch.entries.size(), 1u);
                    EXPECT_EQ(filtered_dispatch.entries[0].expert_id, 3);
                }

                const auto dispatch_key = keyFor(tier,
                                                 target_participant,
                                                 MoEOverlayCollectiveDirection::Dispatch,
                                                 100);
                auto inbound_dispatch = workspace_.dispatchReceive(kLayer, tier);

                MoESparseDispatchStage::Params dispatch_params;
                dispatch_params.device_id = DeviceId::cpu();
                dispatch_params.collective_context = collective_.get();
                dispatch_params.workspace = &workspace_;
                dispatch_params.key = dispatch_key;
                dispatch_params.source_participant = rank_;
                dispatch_params.target_participant = target_participant;
                dispatch_params.hidden = rank_ == 0 ? hidden.get() : nullptr;
                dispatch_params.routing_indices = rank_ == 0 ? routing_indices.get() : nullptr;
                dispatch_params.routing_weights = rank_ == 0 ? routing_weights.get() : nullptr;
                dispatch_params.seq_len = kSeqLen;
                dispatch_params.top_k = kTopK;
                dispatch_params.d_model = kDModel;
                dispatch_params.tier_dispatch = rank_ == 0 ? &filtered_dispatch : nullptr;
                dispatch_params.inbound_rows = &inbound_dispatch;
                MoESparseDispatchStage dispatch_stage(std::move(dispatch_params));
                ASSERT_TRUE(dispatch_stage.execute(cpu_ctx_.get()));

                if (rank_ == target_participant)
                {
                    EXPECT_EQ(inbound_dispatch.live_row_count, filtered_dispatch.token_rows.size());
                    EXPECT_EQ(inbound_dispatch.live_entry_count, filtered_dispatch.entries.size());
                }
                else
                {
                    EXPECT_EQ(inbound_dispatch.live_row_count, 0u);
                    EXPECT_EQ(inbound_dispatch.live_entry_count, 0u);
                }

                auto local_output = workspace_.localExpertOutput(kLayer, tier);
                MoELocalExpertStage::Params local_params;
                local_params.device_id = DeviceId::cpu();
                local_params.input_rows = &inbound_dispatch;
                local_params.output_rows = &local_output;
                local_params.gate_exps = weights.gate.get();
                local_params.up_exps = weights.up.get();
                local_params.down_exps = weights.down.get();
                local_params.num_experts = kNumExperts;
                local_params.top_k = kTopK;
                local_params.d_model = kDModel;
                local_params.expert_intermediate = kIntermediate;
                local_params.layer_idx = kLayer;
                local_params.expert_mask = local_expert_mask;
                MoELocalExpertStage local_stage(std::move(local_params));
                ASSERT_TRUE(local_stage.execute(cpu_ctx_.get()));

                if (rank_ == target_participant)
                    EXPECT_EQ(local_output.live_row_count, filtered_dispatch.token_rows.size());
                else
                    EXPECT_EQ(local_output.live_row_count, 0u);

                const auto return_key = keyFor(tier,
                                               target_participant,
                                               MoEOverlayCollectiveDirection::ReturnReduce,
                                               200);
                auto inbound_return = workspace_.returnReceive(kLayer, tier);
                MoESparseReturnReduceStage::Params return_params;
                return_params.device_id = DeviceId::cpu();
                return_params.collective_context = collective_.get();
                return_params.key = return_key;
                return_params.source_participant = rank_;
                return_params.target_participant = 0;
                return_params.outbound_rows = &local_output;
                return_params.inbound_rows = &inbound_return;
                return_params.dense_output = tier == 0 ? hot_overlay.get() : cold_overlay.get();
                return_params.seq_len = kSeqLen;
                return_params.d_model = kDModel;
                MoESparseReturnReduceStage return_stage(std::move(return_params));
                ASSERT_TRUE(return_stage.execute(cpu_ctx_.get()));
            }
        }

        if (rank_ == 0)
        {
            expectTensorNear(hot_overlay.get(), hot_reference.get());
            expectTensorNear(cold_overlay.get(), cold_reference.get());
            addTensor(hot_overlay.get(), cold_overlay.get(), combined_overlay.get());
            expectTensorNear(combined_overlay.get(), full_reference.get());
        }
    }

} // namespace llaminar2::test