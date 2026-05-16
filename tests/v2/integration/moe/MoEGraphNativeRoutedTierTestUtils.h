#pragma once

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOwnerMap.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

namespace llaminar2::test::moe_graph_native_routed_tier
{
    inline constexpr int kSeqLen = 3;
    inline constexpr int kDModel = 8;
    inline constexpr int kIntermediate = 4;
    inline constexpr int kNumExperts = 4;
    inline constexpr int kTopK = 2;
    inline constexpr int kLayer = 0;
    inline constexpr int kRootParticipant = 0;

    inline std::shared_ptr<FP32Tensor> fp32(std::vector<size_t> shape)
    {
        return std::make_shared<FP32Tensor>(std::move(shape));
    }

    inline void fillHidden(FP32Tensor *tensor)
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

    inline void fillRouting(FP32Tensor *indices, FP32Tensor *weights)
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

    inline void fillExpertTensor(FP32Tensor *tensor, float scale)
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

    inline ExpertWeights makeWeights()
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

    inline bool runReference(IDeviceContext *ctx,
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

    inline MoEExpertDispatchOutput buildDispatch(IDeviceContext *ctx,
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

    inline void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
    {
        ASSERT_EQ(actual->numel(), expected->numel());
        const float *actual_data = actual->data();
        const float *expected_data = expected->data();
        for (size_t i = 0; i < actual->numel(); ++i)
            EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
    }

    inline MoEOverlayCollectiveKey keyFor(
        uint64_t generation_id,
        uint64_t step_id,
        int domain_id,
        int tier,
        MoEOverlayCollectiveDirection direction,
        uint64_t sequence)
    {
        MoEOverlayCollectiveKey key;
        key.generation_id = generation_id;
        key.step_id = step_id;
        key.layer_idx = kLayer;
        key.tier_idx = tier;
        key.domain_id = domain_id;
        key.direction = direction;
        key.sequence = sequence;
        return key;
    }

    struct ParticipantState
    {
        MoEOverlayCollectiveWorkspace workspace;
        std::shared_ptr<FP32Tensor> combined_output;
    };

    inline void initializeParticipant(ParticipantState *participant, uint64_t generation_id, uint64_t step_id)
    {
        participant->workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        participant->workspace.resetForStep(generation_id, step_id);
        participant->combined_output = fp32({kSeqLen, kDModel});
        std::fill_n(participant->combined_output->mutable_data(), participant->combined_output->numel(), 0.0f);
    }

    inline std::vector<int> participantsWithLast(int participant_count, int last_participant)
    {
        std::vector<int> order;
        order.reserve(static_cast<size_t>(participant_count));
        for (int participant = 0; participant < participant_count; ++participant)
        {
            if (participant != last_participant)
                order.push_back(participant);
        }
        order.push_back(last_participant);
        return order;
    }

    inline void runTierThroughLocalSparseStages(
        IDeviceContext *ctx,
        MoEOverlayLocalSparseCollectiveContext *collective,
        std::vector<ParticipantState> *participants,
        int participant_count,
        int domain_id,
        uint64_t generation_id,
        uint64_t step_id,
        int tier_idx,
        int target_participant,
        const MoEExpertDispatchOutput &dispatch,
        const MoEExpertOwnerMap &owner_map,
        TensorBase *hidden,
        TensorBase *routing_indices,
        TensorBase *routing_weights,
        const ExpertWeights &weights)
    {
        ASSERT_GE(target_participant, 0);
        ASSERT_LT(target_participant, participant_count);
        ASSERT_EQ(participants->size(), static_cast<size_t>(participant_count));
        ASSERT_LT(static_cast<size_t>(tier_idx), dispatch.tiers.size());

        const auto dispatch_key = keyFor(generation_id, step_id, domain_id, tier_idx,
                                         MoEOverlayCollectiveDirection::Dispatch,
                                         100 + static_cast<uint64_t>(tier_idx));
        MoEOverlaySparseRows target_inbound;
        for (const int participant : participantsWithLast(participant_count, target_participant))
        {
            auto inbound = (*participants)[static_cast<size_t>(participant)].workspace.dispatchReceive(kLayer, tier_idx);
            MoESparseDispatchStage::Params params;
            params.device_id = DeviceId::cpu();
            params.collective_context = collective;
            params.workspace = &(*participants)[static_cast<size_t>(participant)].workspace;
            params.key = dispatch_key;
            params.source_participant = participant;
            params.target_participant = target_participant;
            params.hidden = participant == kRootParticipant ? hidden : nullptr;
            params.routing_indices = participant == kRootParticipant ? routing_indices : nullptr;
            params.routing_weights = participant == kRootParticipant ? routing_weights : nullptr;
            params.seq_len = kSeqLen;
            params.top_k = kTopK;
            params.d_model = kDModel;
            params.tier_dispatch = participant == kRootParticipant ? &dispatch.tiers[static_cast<size_t>(tier_idx)] : nullptr;
            params.inbound_rows = &inbound;
            MoESparseDispatchStage stage(std::move(params));
            ASSERT_TRUE(stage.execute(ctx));
            if (participant == target_participant)
                target_inbound = inbound;
        }

        EXPECT_EQ(target_inbound.live_row_count, dispatch.tiers[static_cast<size_t>(tier_idx)].token_rows.size());
        EXPECT_EQ(target_inbound.live_entry_count, dispatch.tiers[static_cast<size_t>(tier_idx)].entries.size());

        auto local_output = (*participants)[static_cast<size_t>(target_participant)].workspace.localExpertOutput(kLayer, tier_idx);
        MoELocalExpertStage::Params local_params;
        local_params.device_id = DeviceId::cpu();
        local_params.input_rows = &target_inbound;
        local_params.output_rows = &local_output;
        local_params.gate_exps = weights.gate.get();
        local_params.up_exps = weights.up.get();
        local_params.down_exps = weights.down.get();
        local_params.num_experts = kNumExperts;
        local_params.top_k = kTopK;
        local_params.d_model = kDModel;
        local_params.expert_intermediate = kIntermediate;
        local_params.layer_idx = kLayer;
        local_params.expert_mask = owner_map.expertMaskForParticipant(kLayer, target_participant, kNumExperts);
        MoELocalExpertStage local_stage(std::move(local_params));
        ASSERT_TRUE(local_stage.execute(ctx));
        EXPECT_EQ(local_output.live_row_count, target_inbound.live_row_count);

        const auto return_key = keyFor(generation_id, step_id, domain_id, tier_idx,
                                       MoEOverlayCollectiveDirection::ReturnReduce,
                                       200 + static_cast<uint64_t>(tier_idx));
        for (const int participant : participantsWithLast(participant_count, kRootParticipant))
        {
            auto noop_output = (*participants)[static_cast<size_t>(participant)].workspace.localExpertOutput(kLayer, tier_idx);
            const MoEOverlayReturnRows outbound = participant == target_participant ? local_output : noop_output;
            auto inbound = (*participants)[static_cast<size_t>(participant)].workspace.returnReceive(kLayer, tier_idx);
            MoESparseReturnReduceStage::Params params;
            params.device_id = DeviceId::cpu();
            params.collective_context = collective;
            params.key = return_key;
            params.source_participant = participant;
            params.target_participant = kRootParticipant;
            params.outbound_rows = &outbound;
            params.inbound_rows = &inbound;
            params.dense_output = (*participants)[static_cast<size_t>(participant)].combined_output.get();
            params.seq_len = kSeqLen;
            params.d_model = kDModel;
            MoESparseReturnReduceStage stage(std::move(params));
            ASSERT_TRUE(stage.execute(ctx));

            if (participant == kRootParticipant)
                EXPECT_EQ(inbound.live_row_count, local_output.live_row_count);
        }
    }

} // namespace llaminar2::test::moe_graph_native_routed_tier