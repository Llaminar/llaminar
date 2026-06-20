/**
 * @file Test__MoEGraphNative_LocalTPContinuation_RocmHotCpuCold_MVP.cpp
 * @brief LocalTP replicated-hidden continuation boundary for graph-native MoE sparse dispatch.
 */

#include <gtest/gtest.h>

#include "collective/ITPContext.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <array>
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
        constexpr int kDomain = 7808;
        constexpr int kHotTier = 0;
        constexpr int kColdTier = 1;
        constexpr int kContinuationRoot = 0;
        constexpr int kContinuationNonRoot = 1;
        constexpr int kColdExpertParticipant = 2;
        constexpr int kSparseParticipantCount = 3;

        int rocmDeviceCount()
        {
#ifdef HAVE_ROCM
            int count = 0;
            const hipError_t status = hipGetDeviceCount(&count);
            return status == hipSuccess ? count : 0;
#else
            return 0;
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

        ExpertComputeDomain rocmHotDomain(int rocm_count)
        {
            ExpertComputeDomain domain;
            domain.name = "rocm_hot";
            domain.kind = ExpertDomainKind::LocalTP;
            domain.backend = CollectiveBackendType::RCCL;
            domain.participants = {
                GlobalDeviceAddress::rocm(0, 0),
                GlobalDeviceAddress::rocm(0, rocm_count > 1 ? 1 : 0),
            };
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
            domain.owner_rank = 0;
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

        MoEExpertParallelPlan makeLocalTPPlan(int rocm_count)
        {
            MoEExpertParallelPlan plan;
            plan.enabled = true;
            plan.execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
            plan.continuation_domain = "rocm_hot";
            plan.shared_expert_domain = "rocm_hot";
            plan.continuation_domain_spec.domain = "rocm_hot";
            plan.continuation_domain_spec.logical_root_participant = kContinuationRoot;
            plan.continuation_domain_spec.dense_tp_enabled = true;
            plan.continuation_domain_spec.hidden_layout = MoEContinuationActivationLayout::ReplicatedHidden;
            plan.residency_policy = ExpertResidencyPolicy::StaticById;
            plan.domains = {rocmHotDomain(rocm_count), cpuColdDomain()};
            plan.routed_tiers = {
                routedTier("hot", "rocm_hot", 0),
                routedTier("cold", "cpu_cold", 1, true),
            };
            plan.placements = {
                ExpertLayerPlacement{.layer = kLayer,
                                     .routed_expert_tier = {kColdTier, kColdTier, kColdTier, kColdTier}},
            };
            return plan;
        }

        MoEOverlayCollectiveKey keyFor(MoEOverlayCollectiveDirection direction, uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 81;
            key.step_id = 8;
            key.layer_idx = kLayer;
            key.tier_idx = kColdTier;
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

        void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
        {
            ASSERT_EQ(actual->numel(), expected->numel());
            const float *actual_data = actual->data();
            const float *expected_data = expected->data();
            for (size_t i = 0; i < actual->numel(); ++i)
                EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
        }

        void copyTensor(const TensorBase *source, TensorBase *target)
        {
            ASSERT_NE(source, nullptr);
            ASSERT_NE(target, nullptr);
            ASSERT_EQ(source->numel(), target->numel());
            std::copy(source->data(), source->data() + source->numel(), target->mutable_data());
        }

        class FakeLocalTPContinuationContext final : public ITPContext
        {
        public:
            TPScope scope() const override { return TPScope::LOCAL; }
            int degree() const override { return static_cast<int>(participant_tensors_.size()); }
            int myIndex() const override { return 0; }
            CollectiveBackendType backend() const override { return CollectiveBackendType::HOST; }

            bool allreduce(TensorBase *) override { return false; }
            bool allgather(const TensorBase *, TensorBase *) override { return false; }

            void registerParticipant(int index, TensorBase *tensor)
            {
                ASSERT_GE(index, 0);
                ASSERT_LT(index, degree());
                participant_tensors_[static_cast<size_t>(index)] = tensor;
            }

            bool broadcast(TensorBase *tensor, int source_index = 0) override
            {
                ++broadcast_calls_;
                if (source_index < 0 || source_index >= degree())
                    return false;

                TensorBase *source = participant_tensors_[static_cast<size_t>(source_index)];
                if (!source || !tensor)
                    return false;

                if (tensor == source)
                {
                    source_ready_ = true;
                    for (TensorBase *participant_tensor : participant_tensors_)
                    {
                        if (participant_tensor && participant_tensor != source)
                            copyTensor(source, participant_tensor);
                    }
                    return true;
                }

                if (source_ready_)
                    copyTensor(source, tensor);
                return true;
            }

            int broadcastCalls() const { return broadcast_calls_; }

        private:
            std::array<TensorBase *, 2> participant_tensors_ = {nullptr, nullptr};
            bool source_ready_ = false;
            int broadcast_calls_ = 0;
        };

        struct ParticipantState
        {
            MoEOverlayCollectiveWorkspace workspace;
            std::shared_ptr<FP32Tensor> combined_output;
        };

        void initializeParticipant(ParticipantState *participant)
        {
            participant->workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
            participant->workspace.resetForStep(81, 8);
            participant->combined_output = fp32({kSeqLen, kDModel});
            std::fill_n(participant->combined_output->mutable_data(), participant->combined_output->numel(), 0.0f);
        }

    } // namespace

    TEST(Test__MoEGraphNative_LocalTPContinuation_RocmHotCpuCold_MVP,
         ReplicatedHiddenDispatchesOnceAndBroadcastsImportedOutput)
    {
        const int rocm_count = rocmDeviceCount();
        if (rocm_count <= 0)
            GTEST_SKIP() << "LocalTP continuation ROCm MVP requires HAVE_ROCM and a visible ROCm device";

        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = kSparseParticipantCount, .slot_count = 8});
        std::array<ParticipantState, kSparseParticipantCount> participants;
        for (auto &participant : participants)
            initializeParticipant(&participant);

        const auto plan = makeLocalTPPlan(rocm_count);
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        const auto hot_participants = owner_map.participantIdsForTier(kHotTier);
        ASSERT_EQ(hot_participants.size(), 2u);
        EXPECT_EQ(hot_participants[0], kContinuationRoot);
        EXPECT_EQ(hot_participants[1], kContinuationNonRoot);
        for (int participant_id : hot_participants)
        {
            const auto *participant = owner_map.participantForId(participant_id);
            ASSERT_NE(participant, nullptr);
            EXPECT_TRUE(participant->device.is_rocm());
        }

        const auto cold_participants = owner_map.participantIdsForTier(kColdTier);
        ASSERT_EQ(cold_participants.size(), 1u);
        ASSERT_EQ(cold_participants.front(), kColdExpertParticipant);
        const auto cold_expert_mask = owner_map.expertMaskForParticipant(kLayer, kColdExpertParticipant, kNumExperts);
        EXPECT_EQ(cold_expert_mask, (std::vector<bool>{true, true, true, true}));

        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        auto reference_output = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runReference(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), weights, reference_output.get()));

        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get(), plan);
        ASSERT_EQ(dispatch.tiers.size(), 2u);
        EXPECT_TRUE(dispatch.tiers[kHotTier].token_rows.empty());
        EXPECT_TRUE(dispatch.tiers[kHotTier].entries.empty());
        EXPECT_EQ(dispatch.tiers[kColdTier].token_rows, (std::vector<int>{0, 1, 2}));
        EXPECT_EQ(dispatch.tiers[kColdTier].entries.size(), static_cast<size_t>(kSeqLen * kTopK));

        auto bad_non_root_inbound = participants[kContinuationNonRoot].workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params bad_non_root_params;
        bad_non_root_params.device_id = DeviceId::cpu();
        bad_non_root_params.collective_context = &collective;
        bad_non_root_params.workspace = &participants[kContinuationNonRoot].workspace;
        bad_non_root_params.key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 99);
        bad_non_root_params.source_participant = kContinuationNonRoot;
        bad_non_root_params.target_participant = kColdExpertParticipant;
        bad_non_root_params.hidden = hidden.get();
        bad_non_root_params.routing_indices = routing_indices.get();
        bad_non_root_params.routing_weights = routing_weights.get();
        bad_non_root_params.seq_len = kSeqLen;
        bad_non_root_params.top_k = kTopK;
        bad_non_root_params.d_model = kDModel;
        bad_non_root_params.tier_dispatch = &dispatch.tiers[kColdTier];
        bad_non_root_params.replicated_hidden_export = true;
        bad_non_root_params.logical_continuation_root_participant = kContinuationRoot;
        bad_non_root_params.inbound_rows = &bad_non_root_inbound;
        MoESparseDispatchStage bad_non_root_stage(std::move(bad_non_root_params));
        EXPECT_FALSE(bad_non_root_stage.execute(cpu_ctx.get()));

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = participants[kContinuationRoot].workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params root_dispatch_params;
        root_dispatch_params.device_id = DeviceId::cpu();
        root_dispatch_params.collective_context = &collective;
        root_dispatch_params.workspace = &participants[kContinuationRoot].workspace;
        root_dispatch_params.key = dispatch_key;
        root_dispatch_params.source_participant = kContinuationRoot;
        root_dispatch_params.target_participant = kColdExpertParticipant;
        root_dispatch_params.hidden = hidden.get();
        root_dispatch_params.routing_indices = routing_indices.get();
        root_dispatch_params.routing_weights = routing_weights.get();
        root_dispatch_params.seq_len = kSeqLen;
        root_dispatch_params.top_k = kTopK;
        root_dispatch_params.d_model = kDModel;
        root_dispatch_params.tier_dispatch = &dispatch.tiers[kColdTier];
        root_dispatch_params.replicated_hidden_export = true;
        root_dispatch_params.logical_continuation_root_participant = kContinuationRoot;
        root_dispatch_params.inbound_rows = &root_inbound;
        MoESparseDispatchStage root_dispatch_stage(std::move(root_dispatch_params));
        ASSERT_TRUE(root_dispatch_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(root_inbound.live_row_count, 0u);
        EXPECT_EQ(root_inbound.live_entry_count, 0u);

        auto non_root_inbound = participants[kContinuationNonRoot].workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params non_root_dispatch_params;
        non_root_dispatch_params.device_id = DeviceId::cpu();
        non_root_dispatch_params.collective_context = &collective;
        non_root_dispatch_params.workspace = &participants[kContinuationNonRoot].workspace;
        non_root_dispatch_params.key = dispatch_key;
        non_root_dispatch_params.source_participant = kContinuationNonRoot;
        non_root_dispatch_params.target_participant = kColdExpertParticipant;
        non_root_dispatch_params.seq_len = kSeqLen;
        non_root_dispatch_params.top_k = kTopK;
        non_root_dispatch_params.d_model = kDModel;
        non_root_dispatch_params.replicated_hidden_export = true;
        non_root_dispatch_params.logical_continuation_root_participant = kContinuationRoot;
        non_root_dispatch_params.inbound_rows = &non_root_inbound;
        MoESparseDispatchStage non_root_dispatch_stage(std::move(non_root_dispatch_params));
        ASSERT_TRUE(non_root_dispatch_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(non_root_inbound.live_row_count, 0u);
        EXPECT_EQ(non_root_inbound.live_entry_count, 0u);

        auto cold_inbound = participants[kColdExpertParticipant].workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params cold_noop_dispatch_params;
        cold_noop_dispatch_params.device_id = DeviceId::cpu();
        cold_noop_dispatch_params.collective_context = &collective;
        cold_noop_dispatch_params.workspace = &participants[kColdExpertParticipant].workspace;
        cold_noop_dispatch_params.key = dispatch_key;
        cold_noop_dispatch_params.source_participant = kColdExpertParticipant;
        cold_noop_dispatch_params.target_participant = kColdExpertParticipant;
        cold_noop_dispatch_params.seq_len = kSeqLen;
        cold_noop_dispatch_params.top_k = kTopK;
        cold_noop_dispatch_params.d_model = kDModel;
        cold_noop_dispatch_params.inbound_rows = &cold_inbound;
        MoESparseDispatchStage cold_noop_dispatch_stage(std::move(cold_noop_dispatch_params));
        ASSERT_TRUE(cold_noop_dispatch_stage.execute(cpu_ctx.get()));

        EXPECT_EQ(cold_inbound.live_row_count, dispatch.tiers[kColdTier].token_rows.size());
        EXPECT_EQ(cold_inbound.live_entry_count, dispatch.tiers[kColdTier].entries.size());
        EXPECT_EQ(cold_inbound.live_row_count, static_cast<size_t>(kSeqLen));
        EXPECT_NE(cold_inbound.live_row_count, dispatch.tiers[kColdTier].token_rows.size() * 2u);

        auto local_output = participants[kColdExpertParticipant].workspace.localExpertOutput(kLayer, kColdTier);
        MoELocalExpertStage::Params local_params;
        local_params.device_id = DeviceId::cpu();
        local_params.input_rows = &cold_inbound;
        local_params.output_rows = &local_output;
        local_params.gate_exps = weights.gate.get();
        local_params.up_exps = weights.up.get();
        local_params.down_exps = weights.down.get();
        local_params.num_experts = kNumExperts;
        local_params.top_k = kTopK;
        local_params.d_model = kDModel;
        local_params.expert_intermediate = kIntermediate;
        local_params.layer_idx = kLayer;
        local_params.expert_mask = cold_expert_mask;
        MoELocalExpertStage local_stage(std::move(local_params));
        ASSERT_TRUE(local_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(local_output.live_row_count, dispatch.tiers[kColdTier].token_rows.size());

        FakeLocalTPContinuationContext continuation_tp;
        continuation_tp.registerParticipant(kContinuationRoot, participants[kContinuationRoot].combined_output.get());
        continuation_tp.registerParticipant(kContinuationNonRoot, participants[kContinuationNonRoot].combined_output.get());

        const auto return_key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 200);
        auto cold_return_inbound = participants[kColdExpertParticipant].workspace.returnReceive(kLayer, kColdTier);
        MoESparseReturnReduceStage::Params cold_return_params;
        cold_return_params.device_id = DeviceId::cpu();
        cold_return_params.collective_context = &collective;
        cold_return_params.key = return_key;
        cold_return_params.source_participant = kColdExpertParticipant;
        cold_return_params.target_participant = kContinuationRoot;
        cold_return_params.outbound_rows = &local_output;
        cold_return_params.inbound_rows = &cold_return_inbound;
        cold_return_params.dense_output = participants[kColdExpertParticipant].combined_output.get();
        cold_return_params.seq_len = kSeqLen;
        cold_return_params.d_model = kDModel;
        MoESparseReturnReduceStage cold_return_stage(std::move(cold_return_params));
        ASSERT_TRUE(cold_return_stage.execute(cpu_ctx.get()));

        auto non_root_noop_return = participants[kContinuationNonRoot].workspace.localExpertOutput(kLayer, kColdTier);
        auto non_root_return_inbound = participants[kContinuationNonRoot].workspace.returnReceive(kLayer, kColdTier);
        MoESparseReturnReduceStage::Params non_root_return_params;
        non_root_return_params.device_id = DeviceId::cpu();
        non_root_return_params.collective_context = &collective;
        non_root_return_params.key = return_key;
        non_root_return_params.source_participant = kContinuationNonRoot;
        non_root_return_params.target_participant = kContinuationRoot;
        non_root_return_params.outbound_rows = &non_root_noop_return;
        non_root_return_params.inbound_rows = &non_root_return_inbound;
        non_root_return_params.dense_output = participants[kContinuationNonRoot].combined_output.get();
        non_root_return_params.seq_len = kSeqLen;
        non_root_return_params.d_model = kDModel;
        non_root_return_params.clear_output_before_scatter = true;
        non_root_return_params.continuation_tp_context = &continuation_tp;
        non_root_return_params.broadcast_after_scatter = true;
        non_root_return_params.continuation_root_tp_index = kContinuationRoot;
        MoESparseReturnReduceStage non_root_return_stage(std::move(non_root_return_params));
        ASSERT_TRUE(non_root_return_stage.execute(cpu_ctx.get()));

        auto root_noop_return = participants[kContinuationRoot].workspace.localExpertOutput(kLayer, kColdTier);
        auto root_return_inbound = participants[kContinuationRoot].workspace.returnReceive(kLayer, kColdTier);
        MoESparseReturnReduceStage::Params root_return_params;
        root_return_params.device_id = DeviceId::cpu();
        root_return_params.collective_context = &collective;
        root_return_params.key = return_key;
        root_return_params.source_participant = kContinuationRoot;
        root_return_params.target_participant = kContinuationRoot;
        root_return_params.outbound_rows = &root_noop_return;
        root_return_params.inbound_rows = &root_return_inbound;
        root_return_params.dense_output = participants[kContinuationRoot].combined_output.get();
        root_return_params.seq_len = kSeqLen;
        root_return_params.d_model = kDModel;
        root_return_params.clear_output_before_scatter = true;
        root_return_params.continuation_tp_context = &continuation_tp;
        root_return_params.broadcast_after_scatter = true;
        root_return_params.continuation_root_tp_index = kContinuationRoot;
        MoESparseReturnReduceStage root_return_stage(std::move(root_return_params));
        ASSERT_TRUE(root_return_stage.execute(cpu_ctx.get()));

        EXPECT_EQ(root_return_inbound.live_row_count, dispatch.tiers[kColdTier].token_rows.size());
        EXPECT_EQ(continuation_tp.broadcastCalls(), 1);
        expectTensorNear(participants[kContinuationRoot].combined_output.get(), reference_output.get());
        expectTensorNear(participants[kContinuationNonRoot].combined_output.get(), reference_output.get());
    }

} // namespace llaminar2::test