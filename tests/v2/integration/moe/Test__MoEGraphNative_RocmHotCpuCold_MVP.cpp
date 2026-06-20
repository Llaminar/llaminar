/**
 * @file Test__MoEGraphNative_RocmHotCpuCold_MVP.cpp
 * @brief ROCm-hot plus CPU-cold graph-native sparse MoE overlay MVP integration test.
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

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

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
        constexpr int kDomain = 7404;

        bool rocmDeviceVisible()
        {
#ifdef HAVE_ROCM
            int count = 0;
            const hipError_t status = hipGetDeviceCount(&count);
            return status == hipSuccess && count > 0;
#else
            return false;
#endif
        }

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

        ExpertComputeDomain rocmHotDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "rocm_hot";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {GlobalDeviceAddress::rocm(0, 0)};
            domain.world_ranks = {0};
            domain.owner_rank = 0;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertComputeDomain cpuColdDomain()
        {
            ExpertComputeDomain domain;
            domain.name = "cpu_cold";
            domain.kind = ExpertDomainKind::SingleDevice;
            domain.backend = CollectiveBackendType::MPI;
            domain.participants = {GlobalDeviceAddress::cpu(0)};
            domain.world_ranks = {1};
            domain.owner_rank = 1;
            domain.compute_kind = ExpertDomainComputeKind::ReplicatedExperts;
            return domain;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, int priority, bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.priority = priority;
            tier.fallback = fallback;
            return tier;
        }

        MoEExpertParallelPlan makeRocmHotCpuColdPlan()
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "rocm_hot";
            plan.shared_expert_domain = "rocm_hot";
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {rocmHotDomain(), cpuColdDomain()};
            plan.routed_tiers = {
                routedTier("hot", "rocm_hot", 0),
                routedTier("cold", "cpu_cold", 1, true),
            };
            plan.placements = {
                ExpertLayerPlacement{.layer = kLayer,
                                     .routed_expert_tier = {0, 0, 1, 1}},
            };
            return plan;
        }

        MoEOverlayCollectiveKey keyFor(int tier, MoEOverlayCollectiveDirection direction, uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 41;
            key.step_id = 5;
            key.layer_idx = kLayer;
            key.tier_idx = tier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = sequence;
            return key;
        }

        bool runReference(IDeviceContext *ctx,
                          TensorBase *input,
                          TensorBase *routing_indices,
                          TensorBase *routing_weights,
                          const ExpertWeights &weights,
                          TensorBase *output)
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
            params.expert_mask = {true, true, true, true};
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

        void expectDispatchTargetsOwners(
            const MoEExpertDispatchOutput &dispatch,
            const MoEExpertOwnerMap &owner_map)
        {
            ASSERT_EQ(dispatch.tiers.size(), 2u);
            for (size_t tier = 0; tier < dispatch.tiers.size(); ++tier)
            {
                const auto participant_ids = owner_map.participantIdsForTier(static_cast<int>(tier));
                ASSERT_EQ(participant_ids.size(), 1u);
                for (const auto &entry : dispatch.tiers[tier].entries)
                {
                    const auto *owner = owner_map.ownerFor(kLayer, entry.expert_id);
                    ASSERT_NE(owner, nullptr);
                    EXPECT_EQ(owner->owner_participant, participant_ids.front());
                    if (tier == 0)
                        EXPECT_TRUE(owner->device.is_rocm());
                    else
                        EXPECT_TRUE(owner->device.is_cpu());
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

    } // namespace

    class Test__MoEGraphNative_RocmHotCpuCold_MVP : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
            MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
            if (world_size_ != 2)
                GTEST_SKIP() << "Test requires exactly 2 MPI ranks (got " << world_size_ << ")";

            int root_has_rocm = rank_ == 0 && rocmDeviceVisible() ? 1 : 0;
            MPI_Bcast(&root_has_rocm, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (root_has_rocm == 0)
                GTEST_SKIP() << "ROCm MVP requires HAVE_ROCM and a visible ROCm device on rank 0";

            cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
            mpi_ctx_ = std::make_shared<MPIContext>(rank_, world_size_, MPI_COMM_WORLD);
            collective_ = std::make_unique<MoEOverlayMPISparseCollectiveContext>(
                MoEOverlayMPISparseCollectiveContext::Config{.mpi_ctx = mpi_ctx_, .local_participant_id = rank_});
            workspace_.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
            workspace_.resetForStep(41, 5);
            ready_ = true;
        }

        void TearDown() override
        {
            if (ready_)
                MPI_Barrier(MPI_COMM_WORLD);
        }

        int rank_ = -1;
        int world_size_ = 0;
        bool ready_ = false;
        std::unique_ptr<llaminar2::testing::MockDeviceContext> cpu_ctx_;
        std::shared_ptr<IMPIContext> mpi_ctx_;
        MoEOverlayCollectiveWorkspace workspace_;
        std::unique_ptr<MoEOverlayMPISparseCollectiveContext> collective_;
    };

    TEST_F(Test__MoEGraphNative_RocmHotCpuCold_MVP, SparseDispatchLocalExpertReturnReduceMatchesFullReference)
    {
        const auto plan = makeRocmHotCpuColdPlan();
        const auto owner_map = MoEExpertOwnerMap::build(plan);

        const auto *hot_owner = owner_map.ownerFor(kLayer, 0);
        ASSERT_NE(hot_owner, nullptr);
        EXPECT_EQ(hot_owner->owner_participant, 0);
        EXPECT_EQ(hot_owner->owner_world_rank, 0);
        EXPECT_TRUE(hot_owner->device.is_rocm());

        const auto *cold_owner = owner_map.ownerFor(kLayer, 2);
        ASSERT_NE(cold_owner, nullptr);
        EXPECT_EQ(cold_owner->owner_participant, 1);
        EXPECT_EQ(cold_owner->owner_world_rank, 1);
        EXPECT_TRUE(cold_owner->device.is_cpu());

        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        auto reference_output = fp32({kSeqLen, kDModel});
        auto overlay_output = fp32({kSeqLen, kDModel});
        std::fill_n(overlay_output->mutable_data(), overlay_output->numel(), 0.0f);

        MoEExpertDispatchOutput dispatch;
        if (rank_ == 0)
        {
            ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights, reference_output.get()));
            dispatch = buildDispatch(cpu_ctx_.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan);
            expectDispatchTargetsOwners(dispatch, owner_map);
            EXPECT_EQ(dispatch.tiers[0].token_rows, (std::vector<int>{0, 2}));
            EXPECT_EQ(dispatch.tiers[1].token_rows, (std::vector<int>{1, 2}));
        }

        const auto local_expert_mask = owner_map.expertMaskForParticipant(kLayer, rank_, kNumExperts);
        ASSERT_EQ(local_expert_mask.size(), static_cast<size_t>(kNumExperts));

        for (int tier = 0; tier < 2; ++tier)
        {
            const auto participant_ids = owner_map.participantIdsForTier(tier);
            ASSERT_EQ(participant_ids.size(), 1u);
            const int owner_participant = participant_ids.front();

            const auto dispatch_key = keyFor(tier, MoEOverlayCollectiveDirection::Dispatch, 100 + static_cast<uint64_t>(tier));
            auto inbound_dispatch = workspace_.dispatchReceive(kLayer, tier);

            MoESparseDispatchStage::Params dispatch_params;
            dispatch_params.device_id = DeviceId::cpu();
            dispatch_params.collective_context = collective_.get();
            dispatch_params.workspace = &workspace_;
            dispatch_params.key = dispatch_key;
            dispatch_params.source_participant = rank_;
            dispatch_params.target_participant = owner_participant;
            dispatch_params.hidden = rank_ == 0 ? hidden.get() : nullptr;
            dispatch_params.routing_indices = rank_ == 0 ? routing_indices.get() : nullptr;
            dispatch_params.routing_weights = rank_ == 0 ? routing_weights.get() : nullptr;
            dispatch_params.seq_len = kSeqLen;
            dispatch_params.top_k = kTopK;
            dispatch_params.d_model = kDModel;
            dispatch_params.tier_dispatch = rank_ == 0 ? &dispatch.tiers[static_cast<size_t>(tier)] : nullptr;
            dispatch_params.inbound_rows = &inbound_dispatch;
            MoESparseDispatchStage dispatch_stage(std::move(dispatch_params));
            ASSERT_TRUE(dispatch_stage.execute(cpu_ctx_.get()));

            if (rank_ != owner_participant)
            {
                EXPECT_EQ(inbound_dispatch.live_row_count, 0u);
                EXPECT_EQ(inbound_dispatch.live_entry_count, 0u);
            }
            else
            {
                EXPECT_GT(inbound_dispatch.live_row_count, 0u);
                for (size_t entry = 0; entry < inbound_dispatch.live_entry_count; ++entry)
                {
                    const auto *owner = owner_map.ownerFor(kLayer, inbound_dispatch.expert_ids_host[entry]);
                    ASSERT_NE(owner, nullptr);
                    EXPECT_EQ(owner->owner_participant, rank_);
                }
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

            if (rank_ != owner_participant)
                EXPECT_EQ(local_output.live_row_count, 0u);

            const auto return_key = keyFor(tier, MoEOverlayCollectiveDirection::ReturnReduce, 200 + static_cast<uint64_t>(tier));
            auto inbound_return = workspace_.returnReceive(kLayer, tier);
            MoESparseReturnReduceStage::Params return_params;
            return_params.device_id = DeviceId::cpu();
            return_params.collective_context = collective_.get();
            return_params.key = return_key;
            return_params.source_participant = rank_;
            return_params.target_participant = 0;
            return_params.outbound_rows = &local_output;
            return_params.inbound_rows = &inbound_return;
            return_params.dense_output = overlay_output.get();
            return_params.seq_len = kSeqLen;
            return_params.d_model = kDModel;
            MoESparseReturnReduceStage return_stage(std::move(return_params));
            ASSERT_TRUE(return_stage.execute(cpu_ctx_.get()));
        }

        if (rank_ == 0)
            expectTensorNear(overlay_output.get(), reference_output.get());
    }

} // namespace llaminar2::test