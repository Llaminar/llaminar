/**
 * @file Test__MoEGraphNative_TransferCounters.cpp
 * @brief Graph-native sparse MoE transfer byte counter integration test.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

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
        constexpr int kSeqLen = 8;
        constexpr int kDModel = 64;
        constexpr int kIntermediate = 8;
        constexpr int kNumExperts = 4;
        constexpr int kTopK = 2;
        constexpr int kLayer = 0;
        constexpr int kDomain = 7606;
        constexpr int kColdTier = 1;

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
                        0.02f * static_cast<float>(row + 1) +
                        0.001f * static_cast<float>(col + 1);
                }
            }
        }

        void fillRouting(FP32Tensor *indices, FP32Tensor *weights)
        {
            const float route_indices[] = {
                0.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                1.0f,
                0.0f,
                1.0f,
                2.0f,
                3.0f,
                0.0f,
                1.0f,
                2.0f,
                3.0f,
                0.0f,
                1.0f,
            };
            const float route_weights[] = {
                0.70f,
                0.30f,
                0.55f,
                0.45f,
                0.80f,
                0.20f,
                0.65f,
                0.35f,
                0.60f,
                0.40f,
                0.52f,
                0.48f,
                0.25f,
                0.75f,
                0.90f,
                0.10f,
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
                                       0.0009f * static_cast<float>(row + 1) +
                                       0.00003f * static_cast<float>(col + 1);
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
            fillExpertTensor(weights.gate.get(), 0.006f);
            fillExpertTensor(weights.up.get(), 0.008f);
            fillExpertTensor(weights.down.get(), 0.004f);
            return weights;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.fallback = fallback;
            return tier;
        }

        MoEOverlayCollectiveKey keyFor(MoEOverlayCollectiveDirection direction, uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 61;
            key.step_id = 7;
            key.layer_idx = kLayer;
            key.tier_idx = kColdTier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = sequence;
            return key;
        }

        MoEExpertDispatchOutput buildDispatch(IDeviceContext *ctx,
                                              TensorBase *hidden,
                                              TensorBase *routing_indices,
                                              TensorBase *routing_weights)
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
            params.continuation_domain = "hot";
            params.placement = ExpertLayerPlacement{.layer = kLayer, .routed_expert_tier = {0, 0, 1, 1}};
            params.routed_tiers = {routedTier("hot", "hot"), routedTier("cold", "cold", true)};
            params.output = &dispatch;

            MoEExpertDispatchStage stage(std::move(params));
            EXPECT_TRUE(stage.execute(ctx));
            return dispatch;
        }

    } // namespace

    TEST(Test__MoEGraphNative_TransferCounters, CompactGraphNativePayloadMovesLessThanDenseForSparseRows)
    {
        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        const auto weights = makeWeights();
        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get());
        ASSERT_EQ(dispatch.tiers.size(), 2u);
        const auto &cold_tier = dispatch.tiers[kColdTier];
        EXPECT_EQ(cold_tier.token_rows, (std::vector<int>{4, 6}));
        ASSERT_EQ(cold_tier.entries.size(), 4u);
        EXPECT_TRUE(cold_tier.transfer_required);
        EXPECT_LT(cold_tier.transfer_volume.totalBytes(), cold_tier.transfer_volume.denseTotalBytes());

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        workspace.resetForStep(61, 7);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 4});

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params root_dispatch_params;
        root_dispatch_params.device_id = DeviceId::cpu();
        root_dispatch_params.collective_context = &collective;
        root_dispatch_params.workspace = &workspace;
        root_dispatch_params.key = dispatch_key;
        root_dispatch_params.source_participant = 0;
        root_dispatch_params.target_participant = 1;
        root_dispatch_params.hidden = hidden.get();
        root_dispatch_params.routing_indices = routing_indices.get();
        root_dispatch_params.routing_weights = routing_weights.get();
        root_dispatch_params.seq_len = kSeqLen;
        root_dispatch_params.top_k = kTopK;
        root_dispatch_params.d_model = kDModel;
        root_dispatch_params.tier_dispatch = &cold_tier;
        root_dispatch_params.inbound_rows = &root_inbound;
        MoESparseDispatchStage root_dispatch_stage(std::move(root_dispatch_params));
        ASSERT_TRUE(root_dispatch_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(root_inbound.live_row_count, 0u);

        auto cold_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        MoESparseDispatchStage::Params cold_noop_dispatch_params;
        cold_noop_dispatch_params.device_id = DeviceId::cpu();
        cold_noop_dispatch_params.collective_context = &collective;
        cold_noop_dispatch_params.workspace = &workspace;
        cold_noop_dispatch_params.key = dispatch_key;
        cold_noop_dispatch_params.source_participant = 1;
        cold_noop_dispatch_params.target_participant = 0;
        cold_noop_dispatch_params.seq_len = kSeqLen;
        cold_noop_dispatch_params.top_k = kTopK;
        cold_noop_dispatch_params.d_model = kDModel;
        cold_noop_dispatch_params.inbound_rows = &cold_inbound;
        MoESparseDispatchStage cold_noop_dispatch_stage(std::move(cold_noop_dispatch_params));
        ASSERT_TRUE(cold_noop_dispatch_stage.execute(cpu_ctx.get()));

        EXPECT_EQ(cold_inbound.live_row_count, cold_tier.token_rows.size());
        EXPECT_EQ(cold_inbound.live_entry_count, cold_tier.entries.size());
        EXPECT_EQ(cold_inbound.row_ids_host[0], 4);
        EXPECT_EQ(cold_inbound.row_ids_host[1], 6);

        auto local_output = workspace.localExpertOutput(kLayer, kColdTier);
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
        local_params.expert_mask = {false, false, true, true};
        MoELocalExpertStage local_stage(std::move(local_params));
        ASSERT_TRUE(local_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(local_output.live_row_count, cold_tier.token_rows.size());

        const auto return_key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 200);
        auto cold_return_inbound = workspace.returnReceive(kLayer, kColdTier);
        auto dense_output = fp32({kSeqLen, kDModel});
        std::fill_n(dense_output->mutable_data(), dense_output->numel(), 0.0f);
        MoESparseReturnReduceStage::Params cold_return_params;
        cold_return_params.device_id = DeviceId::cpu();
        cold_return_params.collective_context = &collective;
        cold_return_params.key = return_key;
        cold_return_params.source_participant = 1;
        cold_return_params.target_participant = 0;
        cold_return_params.outbound_rows = &local_output;
        cold_return_params.inbound_rows = &cold_return_inbound;
        cold_return_params.dense_output = dense_output.get();
        cold_return_params.seq_len = kSeqLen;
        cold_return_params.d_model = kDModel;
        MoESparseReturnReduceStage cold_return_stage(std::move(cold_return_params));
        ASSERT_TRUE(cold_return_stage.execute(cpu_ctx.get()));

        auto root_noop_return = workspace.localExpertOutput(kLayer, kColdTier);
        auto root_return_inbound = workspace.returnReceive(kLayer, kColdTier);
        MoESparseReturnReduceStage::Params root_return_params;
        root_return_params.device_id = DeviceId::cpu();
        root_return_params.collective_context = &collective;
        root_return_params.key = return_key;
        root_return_params.source_participant = 0;
        root_return_params.target_participant = 0;
        root_return_params.outbound_rows = &root_noop_return;
        root_return_params.inbound_rows = &root_return_inbound;
        root_return_params.dense_output = dense_output.get();
        root_return_params.seq_len = kSeqLen;
        root_return_params.d_model = kDModel;
        MoESparseReturnReduceStage root_return_stage(std::move(root_return_params));
        ASSERT_TRUE(root_return_stage.execute(cpu_ctx.get()));
        EXPECT_EQ(root_return_inbound.live_row_count, cold_tier.token_rows.size());

        const auto counters = measureMoEOverlaySparseTransferCounters(
            kSeqLen,
            kTopK,
            kDModel,
            &cold_inbound,
            &root_return_inbound);

        EXPECT_EQ(counters.compact_row_count, cold_tier.token_rows.size());
        EXPECT_EQ(counters.compact_entry_count, cold_tier.entries.size());
        EXPECT_EQ(counters.dense_dispatch_bytes, denseMoEOverlayDispatchBytes(kSeqLen, kTopK, kDModel));
        EXPECT_EQ(counters.dense_return_bytes, denseMoEOverlayReturnBytes(kSeqLen, kDModel));
        EXPECT_EQ(counters.compact_dispatch_bytes, compactMoEOverlayDispatchBytes(cold_inbound));
        EXPECT_EQ(counters.compact_return_bytes, compactMoEOverlayReturnBytes(root_return_inbound));
        EXPECT_LT(counters.compactTotalBytes(), counters.denseTotalBytes());
        EXPECT_GT(counters.denseBytesAvoided(), 0u);
    }

} // namespace llaminar2::test