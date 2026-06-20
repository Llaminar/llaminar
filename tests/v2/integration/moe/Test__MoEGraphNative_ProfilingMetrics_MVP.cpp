/**
 * @file Test__MoEGraphNative_ProfilingMetrics_MVP.cpp
 * @brief Phase 14 integration: verifies that LLAMINAR_PROFILING=1 causes the
 *        graph-native MoE overlay stages to emit profiling rows with the three
 *        new phase names and correct data.
 *
 * Reuses the same minimal CPU pipeline as Test__MoEGraphNative_TransferCounters
 * and checks that after execution the profiler holds rows for:
 *   gn_sparse_dispatch, gn_local_expert, gn_return_reduce
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEExpertOverlayProfiler.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"
#include "utils/DebugEnv.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

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
                for (int col = 0; col < kDModel; ++col)
                    data[static_cast<size_t>(row) * kDModel + col] =
                        0.02f * static_cast<float>(row + 1) +
                        0.001f * static_cast<float>(col + 1);
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
            for (size_t e = 0; e < experts; ++e)
                for (size_t r = 0; r < rows; ++r)
                    for (size_t c = 0; c < cols; ++c)
                        data[e * rows * cols + r * cols + c] =
                            scale * static_cast<float>(e + 1) +
                            0.0009f * static_cast<float>(r + 1) +
                            0.00003f * static_cast<float>(c + 1);
        }

        struct ExpertWeights
        {
            std::shared_ptr<FP32Tensor> gate;
            std::shared_ptr<FP32Tensor> up;
            std::shared_ptr<FP32Tensor> down;
        };

        ExpertWeights makeWeights()
        {
            ExpertWeights w;
            w.gate = fp32({kDModel, kIntermediate, kNumExperts});
            w.up = fp32({kDModel, kIntermediate, kNumExperts});
            w.down = fp32({kIntermediate, kDModel, kNumExperts});
            fillExpertTensor(w.gate.get(), 0.006f);
            fillExpertTensor(w.up.get(), 0.008f);
            fillExpertTensor(w.down.get(), 0.004f);
            return w;
        }

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, bool fallback = false)
        {
            ExpertRoutedTier tier;
            tier.name = name;
            tier.domain = domain;
            tier.fallback = fallback;
            return tier;
        }

        MoEOverlayCollectiveKey keyFor(MoEOverlayCollectiveDirection direction, uint64_t seq)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 61;
            key.step_id = 7;
            key.layer_idx = kLayer;
            key.tier_idx = kColdTier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = seq;
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

    // ─────────────────────────────────────────────────────────────────────────────
    // Fixture
    // ─────────────────────────────────────────────────────────────────────────────
    class Test__MoEGraphNative_ProfilingMetrics_MVP : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            mutableDebugEnv().profile.enabled = true;
            MoEExpertOverlayProfiler::reset();
        }

        void TearDown() override
        {
            MoEExpertOverlayProfiler::reset();
            mutableDebugEnv().profile.enabled = false;
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // Test: all three graph-native phases appear after a full dispatch/compute/return
    // ─────────────────────────────────────────────────────────────────────────────
    TEST_F(Test__MoEGraphNative_ProfilingMetrics_MVP, AllThreePhasesEmitted)
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

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        workspace.resetForStep(61, 7);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 4});

        // ── Stage 1: root participant sends rows to cold participant ──
        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 0;
            p.target_participant = 1;
            p.hidden = hidden.get();
            p.routing_indices = routing_indices.get();
            p.routing_weights = routing_weights.get();
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.tier_dispatch = &cold_tier;
            p.inbound_rows = &root_inbound;
            MoESparseDispatchStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        // ── Stage 2: cold participant receives rows ──
        auto cold_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 1;
            p.target_participant = 0;
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.inbound_rows = &cold_inbound;
            MoESparseDispatchStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }
        EXPECT_EQ(cold_inbound.live_row_count, cold_tier.token_rows.size());

        // ── Stage 3: local expert compute ──
        auto local_output = workspace.localExpertOutput(kLayer, kColdTier);
        {
            MoELocalExpertStage::Params p;
            p.device_id = DeviceId::cpu();
            p.input_rows = &cold_inbound;
            p.output_rows = &local_output;
            p.gate_exps = weights.gate.get();
            p.up_exps = weights.up.get();
            p.down_exps = weights.down.get();
            p.num_experts = kNumExperts;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.expert_intermediate = kIntermediate;
            p.layer_idx = kLayer;
            p.expert_mask = {false, false, true, true};
            MoELocalExpertStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        // ── Stage 4: cold participant returns results ──
        auto dense_output = fp32({kSeqLen, kDModel});
        auto cold_return_inbound = workspace.returnReceive(kLayer, kColdTier);
        {
            MoESparseReturnReduceStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 200);
            p.source_participant = 1;
            p.target_participant = 0;
            p.outbound_rows = &local_output;
            p.inbound_rows = &cold_return_inbound;
            p.dense_output = dense_output.get();
            p.seq_len = kSeqLen;
            p.d_model = kDModel;
            MoESparseReturnReduceStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        // ── Stage 5: root participant receives returns ──
        auto root_noop_return = workspace.localExpertOutput(kLayer, kColdTier);
        auto root_return_inbound = workspace.returnReceive(kLayer, kColdTier);
        {
            MoESparseReturnReduceStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.key = keyFor(MoEOverlayCollectiveDirection::ReturnReduce, 200);
            p.source_participant = 0;
            p.target_participant = 0;
            p.outbound_rows = &root_noop_return;
            p.inbound_rows = &root_return_inbound;
            p.dense_output = dense_output.get();
            p.seq_len = kSeqLen;
            p.d_model = kDModel;
            MoESparseReturnReduceStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        // ── Assertions ──
        const auto rows = MoEExpertOverlayProfiler::rows();
        EXPECT_FALSE(rows.empty());

        auto hasPhase = [&](const std::string &phase)
        {
            return std::any_of(rows.begin(), rows.end(),
                               [&](const MoEExpertOverlayProfileRow &r)
                               { return r.phase == phase; });
        };

        EXPECT_TRUE(hasPhase("gn_sparse_dispatch")) << "Expected gn_sparse_dispatch row";
        EXPECT_TRUE(hasPhase("gn_local_expert")) << "Expected gn_local_expert row";
        EXPECT_TRUE(hasPhase("gn_return_reduce")) << "Expected gn_return_reduce row";
    }

    TEST_F(Test__MoEGraphNative_ProfilingMetrics_MVP, DispatchRow_CompactBytesSmallerThanDense)
    {
        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        const auto weights = makeWeights();
        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get());
        const auto &cold_tier = dispatch.tiers[kColdTier];

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        workspace.resetForStep(61, 7);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 4});

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 0;
            p.target_participant = 1;
            p.hidden = hidden.get();
            p.routing_indices = routing_indices.get();
            p.routing_weights = routing_weights.get();
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.tier_dispatch = &cold_tier;
            p.inbound_rows = &root_inbound;
            MoESparseDispatchStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        const auto rows = MoEExpertOverlayProfiler::rows();
        const auto it = std::find_if(rows.begin(), rows.end(),
                                     [](const MoEExpertOverlayProfileRow &r)
                                     {
                                         return r.phase == "gn_sparse_dispatch";
                                     });
        ASSERT_NE(it, rows.end()) << "gn_sparse_dispatch row not found";
        // Compact bytes stored in outbound_bytes; dense bytes saved > 0 for sparse routing
        EXPECT_GT(it->dense_bytes_avoided, 0u);
        EXPECT_EQ(it->transport_mode, "compact");
    }

    TEST_F(Test__MoEGraphNative_ProfilingMetrics_MVP, LocalExpertRow_HasComputeMs)
    {
        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        const auto weights = makeWeights();
        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get());
        const auto &cold_tier = dispatch.tiers[kColdTier];

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        workspace.resetForStep(61, 7);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 4});

        // Run dispatch to populate cold_inbound
        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        auto cold_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 0;
            p.target_participant = 1;
            p.hidden = hidden.get();
            p.routing_indices = routing_indices.get();
            p.routing_weights = routing_weights.get();
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.tier_dispatch = &cold_tier;
            p.inbound_rows = &root_inbound;
            MoESparseDispatchStage(std::move(p)).execute(cpu_ctx.get());
        }
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 1;
            p.target_participant = 0;
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.inbound_rows = &cold_inbound;
            MoESparseDispatchStage(std::move(p)).execute(cpu_ctx.get());
        }

        // Clear dispatch profiling rows — keep only local expert
        MoEExpertOverlayProfiler::reset();

        auto local_output = workspace.localExpertOutput(kLayer, kColdTier);
        {
            MoELocalExpertStage::Params p;
            p.device_id = DeviceId::cpu();
            p.input_rows = &cold_inbound;
            p.output_rows = &local_output;
            p.gate_exps = weights.gate.get();
            p.up_exps = weights.up.get();
            p.down_exps = weights.down.get();
            p.num_experts = kNumExperts;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.expert_intermediate = kIntermediate;
            p.layer_idx = kLayer;
            p.expert_mask = {false, false, true, true};
            MoELocalExpertStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        const auto rows = MoEExpertOverlayProfiler::rows();
        const auto it = std::find_if(rows.begin(), rows.end(),
                                     [](const MoEExpertOverlayProfileRow &r)
                                     {
                                         return r.phase == "gn_local_expert";
                                     });
        ASSERT_NE(it, rows.end()) << "gn_local_expert row not found";
        EXPECT_GE(it->compute_ms, 0.0); // timing must be non-negative
        EXPECT_EQ(it->domain_kind, "CPU");
        EXPECT_FALSE(it->executed_experts.empty());
    }

    TEST_F(Test__MoEGraphNative_ProfilingMetrics_MVP, WhenProfilingDisabled_NoRowsEmitted)
    {
        mutableDebugEnv().profile.enabled = false;

        auto cpu_ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        const auto weights = makeWeights();
        const auto dispatch = buildDispatch(cpu_ctx.get(), hidden.get(), routing_indices.get(), routing_weights.get());
        const auto &cold_tier = dispatch.tiers[kColdTier];

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
        workspace.resetForStep(61, 7);
        MoEOverlayLocalSparseCollectiveContext collective({.participant_count = 2, .slot_count = 4});

        const auto dispatch_key = keyFor(MoEOverlayCollectiveDirection::Dispatch, 100);
        auto root_inbound = workspace.dispatchReceive(kLayer, kColdTier);
        {
            MoESparseDispatchStage::Params p;
            p.device_id = DeviceId::cpu();
            p.collective_context = &collective;
            p.workspace = &workspace;
            p.key = dispatch_key;
            p.source_participant = 0;
            p.target_participant = 1;
            p.hidden = hidden.get();
            p.routing_indices = routing_indices.get();
            p.routing_weights = routing_weights.get();
            p.seq_len = kSeqLen;
            p.top_k = kTopK;
            p.d_model = kDModel;
            p.tier_dispatch = &cold_tier;
            p.inbound_rows = &root_inbound;
            MoESparseDispatchStage stage(std::move(p));
            ASSERT_TRUE(stage.execute(cpu_ctx.get()));
        }

        EXPECT_TRUE(MoEExpertOverlayProfiler::rows().empty());
    }

} // namespace llaminar2::test
