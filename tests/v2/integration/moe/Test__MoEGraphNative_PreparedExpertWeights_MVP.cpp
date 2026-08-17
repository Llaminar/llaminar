/**
 * @file Test__MoEGraphNative_PreparedExpertWeights_MVP.cpp
 * @brief Phase 12 integration test — MoELocalExpertStage with prepared expert engines.
 *
 * Verifies that MoELocalExpertStage executes correctly when supplied with
 * pre-prepared GEMM engine vectors (prepared_gate_gemm / prepared_up_gemm /
 * prepared_down_gemm) instead of raw 3D packed tensors.
 *
 * The raw expert tensors (gate_exps / up_exps / down_exps) are explicitly set
 * to nullptr in MoELocalExpertStage::Params to prove the prepared path is used.
 *
 * Single MPI rank; no collective communication required.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseReturnReduceStage.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#include <algorithm>
#include <iterator>
#include <limits>
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
        constexpr int kDomain = 7007;

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
                        0.05f * static_cast<float>(row + 1) +
                        0.01f * static_cast<float>(col + 1);
        }

        void fillRouting(FP32Tensor *indices, FP32Tensor *weights)
        {
            const float route_indices[] = {0.0f, 1.0f, 2.0f, 3.0f, 1.0f, 2.0f};
            const float route_weights[] = {0.60f, 0.40f, 0.25f, 0.75f, 0.55f, 0.45f};
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
                for (size_t row = 0; row < rows; ++row)
                    for (size_t col = 0; col < cols; ++col)
                    {
                        const size_t offset = expert * rows * cols + row * cols + col;
                        data[offset] = scale * static_cast<float>(expert + 1) +
                                       0.003f * static_cast<float>(row + 1) +
                                       0.0007f * static_cast<float>(col + 1);
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
            ExpertWeights w;
            w.gate = fp32({kDModel, kIntermediate, kNumExperts});
            w.up = fp32({kDModel, kIntermediate, kNumExperts});
            w.down = fp32({kIntermediate, kDModel, kNumExperts});
            fillExpertTensor(w.gate.get(), 0.010f);
            fillExpertTensor(w.up.get(), 0.012f);
            fillExpertTensor(w.down.get(), 0.008f);
            return w;
        }

        // -----------------------------------------------------------------------
        // Full-reference run using MoEExpertComputeStage with raw tensors.
        // -----------------------------------------------------------------------
        bool runReference(IDeviceContext *ctx,
                          TensorBase *input,
                          TensorBase *routing_indices,
                          TensorBase *routing_weights,
                          const ExpertWeights &weights,
                          TensorBase *output,
                          std::vector<bool> expert_mask = {true, true, true, true},
                          int seq_len = kSeqLen)
        {
            MoEExpertComputeStage::Params params;
            params.device_id = DeviceId::cpu();
            params.input = input;
            params.seq_len = seq_len;
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

        RoutedExpertTier routedTier(const std::string &name, const std::string &domain)
        {
            RoutedExpertTier tier;
            tier.name = name;
            tier.domain = domain;
            return tier;
        }

        MoEOverlayCollectiveKey keyFor(int tier, MoEOverlayCollectiveDirection direction, uint64_t sequence)
        {
            MoEOverlayCollectiveKey key;
            key.generation_id = 51;
            key.step_id = 7;
            key.layer_idx = kLayer;
            key.tier_idx = tier;
            key.domain_id = kDomain;
            key.direction = direction;
            key.sequence = sequence;
            return key;
        }

        void expectTensorNear(const TensorBase *actual, const TensorBase *expected,
                              float tolerance = 1e-4f)
        {
            ASSERT_EQ(actual->numel(), expected->numel());
            const float *a = actual->data();
            const float *e = expected->data();
            for (size_t i = 0; i < actual->numel(); ++i)
                EXPECT_NEAR(a[i], e[i], tolerance) << "index=" << i;
        }

    } // namespace

    // ===========================================================================
    // Test fixture — single MPI rank
    // ===========================================================================

    class Test__MoEGraphNative_PreparedExpertWeights_MVP : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int world_size = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            if (world_size != 1)
                GTEST_SKIP() << "Test requires exactly 1 MPI rank (got " << world_size << ")";

            cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(
                DeviceId::cpu(), ComputeBackendType::CPU);

            // Single-participant local collective (no MPI communication required).
            collective_ = std::make_unique<MoEOverlayLocalSparseCollectiveContext>(
                MoEOverlayLocalSparseCollectiveContext::Config{.participant_count = 1, .slot_count = 4});
            workspace_.ensureCapacity(kSeqLen, kSeqLen * kTopK, kDModel, kTopK, DeviceId::cpu());
            workspace_.resetForStep(51, 7);
        }

        std::unique_ptr<llaminar2::testing::MockDeviceContext> cpu_ctx_;
        MoEOverlayCollectiveWorkspace workspace_;
        std::unique_ptr<MoEOverlayLocalSparseCollectiveContext> collective_;
    };

    // ---------------------------------------------------------------------------
    // Main test: full pipeline using prepared engine vectors, null raw tensors.
    // ---------------------------------------------------------------------------

    TEST_F(Test__MoEGraphNative_PreparedExpertWeights_MVP,
           PreparedLocalExpertStageMatchesFullReference)
    {
        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());
        fillRouting(routing_indices.get(), routing_weights.get());

        // Reference result using full MoEExpertComputeStage.
        auto reference_output = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(),
                                 routing_indices.get(), routing_weights.get(),
                                 weights, reference_output.get()));

        auto overlay_output = fp32({kSeqLen, kDModel});
        std::fill_n(overlay_output->mutable_data(), overlay_output->numel(), 0.0f);

        // ------------------------------------------------------------------ //
        // MoEExpertDispatchStage: route all 4 experts to single tier 0.      //
        // ------------------------------------------------------------------ //
        MoEExpertDispatchOutput dispatch;
        {
            MoEExpertDispatchStage::Params dp;
            dp.device_id = DeviceId::cpu();
            dp.routing_indices = routing_indices.get();
            dp.routing_weights = routing_weights.get();
            dp.hidden = hidden.get();
            dp.seq_len = kSeqLen;
            dp.top_k = kTopK;
            dp.d_model = kDModel;
            dp.continuation_domain = "local";
            dp.placement = RoutedExpertLayerPlacement{.layer = kLayer, .routed_expert_tier = {0, 0, 0, 0}};
            dp.routed_tiers = {routedTier("local", "local")};
            dp.output = &dispatch;
            MoEExpertDispatchStage ds(std::move(dp));
            ASSERT_TRUE(ds.execute(cpu_ctx_.get()));
        }
        ASSERT_EQ(dispatch.tiers.size(), 1u);

        // ------------------------------------------------------------------ //
        // MoESparseDispatchStage: participant 0 sends to itself.              //
        // ------------------------------------------------------------------ //
        const auto dispatch_key = keyFor(0, MoEOverlayCollectiveDirection::Dispatch, 100);
        auto inbound_dispatch = workspace_.dispatchReceive(kLayer, 0);

        {
            MoESparseDispatchStage::Params sp;
            sp.device_id = DeviceId::cpu();
            sp.collective_context = collective_.get();
            sp.workspace = &workspace_;
            sp.key = dispatch_key;
            sp.source_participant = 0;
            sp.target_participant = 0;
            sp.hidden = hidden.get();
            sp.routing_indices = routing_indices.get();
            sp.routing_weights = routing_weights.get();
            sp.seq_len = kSeqLen;
            sp.top_k = kTopK;
            sp.d_model = kDModel;
            sp.tier_dispatch = &dispatch.tiers[0];
            sp.fixed_residency_epoch = dispatch.residency_epoch;
            sp.inbound_rows = &inbound_dispatch;
            MoESparseDispatchStage ss(std::move(sp));
            ASSERT_TRUE(ss.execute(cpu_ctx_.get()));
        }
        EXPECT_GT(inbound_dispatch.live_row_count, 0u)
            << "Dispatch produced no rows — check routing fixture";

        // ------------------------------------------------------------------ //
        // Prepare GEMM engines from raw 3D tensors.                           //
        // Raw tensors set to nullptr in local stage params to prove prepared  //
        // path is used.                                                        //
        // ------------------------------------------------------------------ //
        MoEExpertComputeStage::Params prep;
        prep.device_id = DeviceId::cpu();
        prep.num_experts = kNumExperts;
        prep.top_k = kTopK;
        prep.d_model = kDModel;
        prep.expert_intermediate = kIntermediate;
        prep.layer_idx = kLayer;
        prep.gate_exps = weights.gate.get();
        prep.up_exps = weights.up.get();
        prep.down_exps = weights.down.get();
        prep.expert_mask = {true, true, true, true};

        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(prep))
            << "extractExpertViews failed";
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(prep))
            << "prepareExpertGemmEngines failed";

        ASSERT_EQ(static_cast<int>(prep.prepared_gate_gemm.size()), kNumExperts);
        for (int e = 0; e < kNumExperts; ++e)
            EXPECT_NE(prep.prepared_gate_gemm[static_cast<size_t>(e)], nullptr)
                << "expert " << e << " gate engine is null";

        // ------------------------------------------------------------------ //
        // MoELocalExpertStage with prepared engines; raw tensors null.        //
        // ------------------------------------------------------------------ //
        auto local_output = workspace_.localExpertOutput(kLayer, 0);

        MoELocalExpertStage::Params local_params;
        local_params.device_id = DeviceId::cpu();
        local_params.num_experts = kNumExperts;
        local_params.top_k = kTopK;
        local_params.d_model = kDModel;
        local_params.expert_intermediate = kIntermediate;
        local_params.layer_idx = kLayer;
        local_params.expert_mask = {true, true, true, true};

        // Raw tensors intentionally null — must not be used by execute().
        local_params.gate_exps = nullptr;
        local_params.up_exps = nullptr;
        local_params.down_exps = nullptr;

        // Move prepared engine state from MoEExpertComputeStage::Params.
        local_params.prepared_gate_gemm = std::move(prep.prepared_gate_gemm);
        local_params.prepared_up_gemm = std::move(prep.prepared_up_gemm);
        local_params.prepared_down_gemm = std::move(prep.prepared_down_gemm);
        local_params.moe_owned_kernels = std::move(prep.moe_owned_kernels);

        local_params.input_rows = &inbound_dispatch;
        local_params.output_rows = &local_output;

        // Validate prepared weights before execution.
        {
            std::string validate_err;
            MoELocalExpertStage probe(local_params);
            EXPECT_TRUE(probe.validatePreparedWeights(&validate_err))
                << "validatePreparedWeights: " << validate_err;
        }
        EXPECT_EQ(local_params.gate_exps, nullptr) << "gate_exps must be null (prepared path)";
        EXPECT_EQ(local_params.up_exps, nullptr) << "up_exps must be null (prepared path)";
        EXPECT_EQ(local_params.down_exps, nullptr) << "down_exps must be null (prepared path)";

        // Execute.
        {
            MoELocalExpertStage local_stage(std::move(local_params));
            ASSERT_TRUE(local_stage.execute(cpu_ctx_.get()));
        }
        EXPECT_GT(local_output.live_row_count, 0u)
            << "Local expert produced no output rows";

        // ------------------------------------------------------------------ //
        // MoESparseReturnReduceStage: gather results back to dense output.   //
        // ------------------------------------------------------------------ //
        {
            auto inbound_return = workspace_.returnReceive(kLayer, 0);
            const auto return_key = keyFor(0, MoEOverlayCollectiveDirection::ReturnReduce, 200);

            MoESparseReturnReduceStage::Params rp;
            rp.device_id = DeviceId::cpu();
            rp.collective_context = collective_.get();
            rp.key = return_key;
            rp.source_participant = 0;
            rp.target_participant = 0;
            rp.outbound_rows = &local_output;
            rp.inbound_rows = &inbound_return;
            rp.dense_output = overlay_output.get();
            rp.seq_len = kSeqLen;
            rp.d_model = kDModel;
            MoESparseReturnReduceStage rs(std::move(rp));
            ASSERT_TRUE(rs.execute(cpu_ctx_.get()));
        }

        // ------------------------------------------------------------------ //
        // Prepared path must produce the same output as the raw-tensor path. //
        // ------------------------------------------------------------------ //
        expectTensorNear(overlay_output.get(), reference_output.get());
    }

    TEST_F(Test__MoEGraphNative_PreparedExpertWeights_MVP,
           PreparedLocalExpertStageHandlesPartialTopKRows)
    {
        auto weights = makeWeights();
        auto hidden = fp32({kSeqLen, kDModel});
        auto routing_indices = fp32({kSeqLen, kTopK});
        auto routing_weights = fp32({kSeqLen, kTopK});
        fillHidden(hidden.get());

        const float route_indices[] = {0.0f, 2.0f, 1.0f, 3.0f, 2.0f, 1.0f};
        const float route_weights[] = {0.60f, 0.40f, 0.25f, 0.75f, 0.55f, 0.45f};
        std::copy(std::begin(route_indices), std::end(route_indices), routing_indices->mutable_data());
        std::copy(std::begin(route_weights), std::end(route_weights), routing_weights->mutable_data());

        const std::vector<bool> local_mask = {true, false, true, false};

        auto reference_output = fp32({kSeqLen, kDModel});
        ASSERT_TRUE(runReference(cpu_ctx_.get(), hidden.get(),
                                 routing_indices.get(), routing_weights.get(),
                                 weights, reference_output.get(), local_mask));

        MoEExpertDispatchOutput dispatch;
        {
            MoEExpertDispatchStage::Params dp;
            dp.device_id = DeviceId::cpu();
            dp.routing_indices = routing_indices.get();
            dp.routing_weights = routing_weights.get();
            dp.hidden = hidden.get();
            dp.seq_len = kSeqLen;
            dp.top_k = kTopK;
            dp.d_model = kDModel;
            dp.continuation_domain = "local";
            dp.placement = RoutedExpertLayerPlacement{.layer = kLayer, .routed_expert_tier = {0, 1, 0, 1}};
            dp.routed_tiers = {routedTier("local", "local"), routedTier("remote", "remote")};
            dp.output = &dispatch;
            MoEExpertDispatchStage ds(std::move(dp));
            ASSERT_TRUE(ds.execute(cpu_ctx_.get()));
        }
        ASSERT_EQ(dispatch.tiers.size(), 2u);
        ASSERT_EQ(dispatch.tiers[0].entries.size(), 3u);
        ASSERT_EQ(dispatch.tiers[0].token_rows.size(), 2u);

        const auto dispatch_key = keyFor(0, MoEOverlayCollectiveDirection::Dispatch, 300);
        auto inbound_dispatch = workspace_.dispatchReceive(kLayer, 0);
        {
            MoESparseDispatchStage::Params sp;
            sp.device_id = DeviceId::cpu();
            sp.collective_context = collective_.get();
            sp.workspace = &workspace_;
            sp.key = dispatch_key;
            sp.source_participant = 0;
            sp.target_participant = 0;
            sp.hidden = hidden.get();
            sp.routing_indices = routing_indices.get();
            sp.routing_weights = routing_weights.get();
            sp.seq_len = kSeqLen;
            sp.top_k = kTopK;
            sp.d_model = kDModel;
            sp.tier_dispatch = &dispatch.tiers[0];
            sp.fixed_residency_epoch = dispatch.residency_epoch;
            sp.inbound_rows = &inbound_dispatch;
            MoESparseDispatchStage ss(std::move(sp));
            ASSERT_TRUE(ss.execute(cpu_ctx_.get()));
        }
        ASSERT_EQ(inbound_dispatch.live_row_count, 2u);
        ASSERT_EQ(inbound_dispatch.live_entry_count, 3u);

        MoEExpertComputeStage::Params prep;
        prep.device_id = DeviceId::cpu();
        prep.num_experts = kNumExperts;
        prep.top_k = kTopK;
        prep.d_model = kDModel;
        prep.expert_intermediate = kIntermediate;
        prep.layer_idx = kLayer;
        prep.gate_exps = weights.gate.get();
        prep.up_exps = weights.up.get();
        prep.down_exps = weights.down.get();
        prep.expert_mask = local_mask;
        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(prep));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(prep));

        auto local_output = workspace_.localExpertOutput(kLayer, 0);
        MoELocalExpertStage::Params local_params;
        local_params.device_id = DeviceId::cpu();
        local_params.num_experts = kNumExperts;
        local_params.top_k = kTopK;
        local_params.d_model = kDModel;
        local_params.expert_intermediate = kIntermediate;
        local_params.layer_idx = kLayer;
        local_params.expert_mask = local_mask;
        local_params.prepared_gate_gemm = std::move(prep.prepared_gate_gemm);
        local_params.prepared_up_gemm = std::move(prep.prepared_up_gemm);
        local_params.prepared_down_gemm = std::move(prep.prepared_down_gemm);
        local_params.moe_owned_kernels = std::move(prep.moe_owned_kernels);
        local_params.input_rows = &inbound_dispatch;
        local_params.output_rows = &local_output;

        {
            MoELocalExpertStage local_stage(std::move(local_params));
            ASSERT_TRUE(local_stage.execute(cpu_ctx_.get()));
        }
        ASSERT_EQ(local_output.live_row_count, 2u);

        auto overlay_output = fp32({kSeqLen, kDModel});
        std::fill_n(overlay_output->mutable_data(), overlay_output->numel(), 0.0f);
        {
            auto inbound_return = workspace_.returnReceive(kLayer, 0);
            const auto return_key = keyFor(0, MoEOverlayCollectiveDirection::ReturnReduce, 400);

            MoESparseReturnReduceStage::Params rp;
            rp.device_id = DeviceId::cpu();
            rp.collective_context = collective_.get();
            rp.key = return_key;
            rp.source_participant = 0;
            rp.target_participant = 0;
            rp.outbound_rows = &local_output;
            rp.inbound_rows = &inbound_return;
            rp.dense_output = overlay_output.get();
            rp.seq_len = kSeqLen;
            rp.d_model = kDModel;
            MoESparseReturnReduceStage rs(std::move(rp));
            ASSERT_TRUE(rs.execute(cpu_ctx_.get()));
        }

        expectTensorNear(overlay_output.get(), reference_output.get());
    }

    TEST_F(Test__MoEGraphNative_PreparedExpertWeights_MVP,
           FixedCapacityTicketProtocolReusesCPUStagesAcrossLogicalPrefixes)
    {
        constexpr int bucket_rows = 5;
        auto weights = makeWeights();
        auto ticket_storage =
            std::make_shared<MoEOverlayDispatchTicketStorage>();
        ticket_storage->bindFixedCapacity(
            kLayer,
            bucket_rows,
            kTopK,
            kDModel,
            DeviceId::cpu(),
            /*workspace_generation=*/61);

        auto &ticket = ticket_storage->ticket();
        auto *const header_address = ticket.header;
        auto *const hidden_address = ticket.hidden_rows_fp32;
        auto *const return_address = ticket.return_rows_fp32;

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(
            bucket_rows,
            bucket_rows * kTopK,
            kDModel,
            kTopK,
            DeviceId::cpu());
        workspace.resetForStep(61, 0);
        MoEOverlayLocalSparseCollectiveContext collective(
            {.participant_count = 1, .slot_count = 8});

        MoEExpertDispatchOutput dispatch;
        MoEExpertDispatchStage::Params dispatch_params;
        dispatch_params.device_id = DeviceId::cpu();
        dispatch_params.ticket_storage = ticket_storage;
        dispatch_params.seq_len = bucket_rows;
        dispatch_params.top_k = kTopK;
        dispatch_params.d_model = kDModel;
        dispatch_params.continuation_domain = "local";
        dispatch_params.placement = RoutedExpertLayerPlacement{
            .layer = kLayer,
            .routed_expert_tier = {0, 0, 0, 0}};
        dispatch_params.routed_tiers = {routedTier("local", "local")};
        dispatch_params.output = &dispatch;
        MoEExpertDispatchStage dispatch_stage(std::move(dispatch_params));
        ASSERT_TRUE(dispatch_stage.requiresHostGraphTicketFence());
        ASSERT_TRUE(dispatch_stage.bufferContract().inputs.empty());

        auto inbound_dispatch = workspace.dispatchReceive(kLayer, 0);
        MoESparseDispatchStage::Params sparse_params;
        sparse_params.device_id = DeviceId::cpu();
        sparse_params.collective_context = &collective;
        sparse_params.workspace = &workspace;
        sparse_params.key = keyFor(
            0,
            MoEOverlayCollectiveDirection::Dispatch,
            501);
        sparse_params.source_participant = 0;
        sparse_params.target_participant = 0;
        sparse_params.seq_len = bucket_rows;
        sparse_params.top_k = kTopK;
        sparse_params.d_model = kDModel;
        sparse_params.tier_index = 0;
        sparse_params.ticket_storage = ticket_storage;
        sparse_params.dispatch_output = &dispatch;
        sparse_params.inbound_rows = &inbound_dispatch;
        MoESparseDispatchStage sparse_stage(std::move(sparse_params));
        ASSERT_TRUE(sparse_stage.requiresHostGraphTicketFence());
        ASSERT_TRUE(sparse_stage.bufferContract().inputs.empty());

        MoEExpertComputeStage::Params prep;
        prep.device_id = DeviceId::cpu();
        prep.num_experts = kNumExperts;
        prep.top_k = kTopK;
        prep.d_model = kDModel;
        prep.expert_intermediate = kIntermediate;
        prep.layer_idx = kLayer;
        prep.gate_exps = weights.gate.get();
        prep.up_exps = weights.up.get();
        prep.down_exps = weights.down.get();
        prep.expert_mask = {true, true, true, true};
        ASSERT_TRUE(MoEExpertComputeStage::extractExpertViews(prep));
        ASSERT_TRUE(MoEExpertComputeStage::prepareExpertGemmEngines(prep));

        auto local_output = workspace.localExpertOutput(kLayer, 0);
        MoELocalExpertStage::Params local_params;
        local_params.device_id = DeviceId::cpu();
        local_params.input_rows = &inbound_dispatch;
        local_params.output_rows = &local_output;
        local_params.num_experts = kNumExperts;
        local_params.top_k = kTopK;
        local_params.d_model = kDModel;
        local_params.expert_intermediate = kIntermediate;
        local_params.layer_idx = kLayer;
        local_params.expert_mask = {true, true, true, true};
        local_params.prepared_gate_gemm = std::move(prep.prepared_gate_gemm);
        local_params.prepared_up_gemm = std::move(prep.prepared_up_gemm);
        local_params.prepared_down_gemm = std::move(prep.prepared_down_gemm);
        local_params.moe_owned_kernels = std::move(prep.moe_owned_kernels);
        MoELocalExpertStage local_stage(std::move(local_params));

        auto inbound_return = workspace.returnReceive(kLayer, 0);
        MoESparseReturnReduceStage::Params return_params;
        return_params.device_id = DeviceId::cpu();
        return_params.collective_context = &collective;
        return_params.key = keyFor(
            0,
            MoEOverlayCollectiveDirection::ReturnReduce,
            601);
        return_params.source_participant = 0;
        return_params.target_participant = 0;
        return_params.outbound_rows = &local_output;
        return_params.inbound_rows = &inbound_return;
        return_params.ticket_storage = ticket_storage;
        return_params.publish_ticket_completion = true;
        return_params.seq_len = bucket_rows;
        return_params.d_model = kDModel;
        return_params.clear_output_before_scatter = true;
        MoESparseReturnReduceStage return_stage(std::move(return_params));
        ASSERT_TRUE(return_stage.bufferContract().outputs.empty());

        const auto run_transaction = [&](int logical_rows, float offset)
        {
            auto hidden = fp32(
                {static_cast<size_t>(logical_rows), kDModel});
            auto routing_indices = fp32(
                {static_cast<size_t>(logical_rows), kTopK});
            auto routing_weights = fp32(
                {static_cast<size_t>(logical_rows), kTopK});
            for (int row = 0; row < logical_rows; ++row)
            {
                for (int col = 0; col < kDModel; ++col)
                {
                    hidden->mutable_data()[
                        static_cast<size_t>(row) * kDModel + col] =
                        offset + 0.05f * static_cast<float>(row + 1) +
                        0.01f * static_cast<float>(col + 1);
                }
                routing_indices->mutable_data()[
                    static_cast<size_t>(row) * kTopK] =
                    static_cast<float>((row * 2) % kNumExperts);
                routing_indices->mutable_data()[
                    static_cast<size_t>(row) * kTopK + 1] =
                    static_cast<float>((row * 2 + 1) % kNumExperts);
                routing_weights->mutable_data()[
                    static_cast<size_t>(row) * kTopK] = 0.65f;
                routing_weights->mutable_data()[
                    static_cast<size_t>(row) * kTopK + 1] = 0.35f;
            }

            std::fill_n(
                ticket.routing_indices_fp32,
                bucket_rows * kTopK,
                99.0f);
            std::fill_n(
                ticket.routing_weights_fp32,
                bucket_rows * kTopK,
                std::numeric_limits<float>::quiet_NaN());
            std::fill_n(
                ticket.hidden_rows_fp32,
                bucket_rows * kDModel,
                std::numeric_limits<float>::quiet_NaN());
            std::copy_n(
                routing_indices->data(),
                logical_rows * kTopK,
                ticket.routing_indices_fp32);
            std::copy_n(
                routing_weights->data(),
                logical_rows * kTopK,
                ticket.routing_weights_fp32);
            std::copy_n(
                hidden->data(),
                logical_rows * kDModel,
                ticket.hidden_rows_fp32);
            ticket.header->logical_row_count = logical_rows;

            auto reference = fp32(
                {static_cast<size_t>(logical_rows), kDModel});
            ASSERT_TRUE(runReference(
                cpu_ctx_.get(),
                hidden.get(),
                routing_indices.get(),
                routing_weights.get(),
                weights,
                reference.get(),
                {true, true, true, true},
                logical_rows));

            ASSERT_TRUE(dispatch_stage.execute(cpu_ctx_.get()));
            ASSERT_EQ(dispatch.logical_seq_len, logical_rows);
            ASSERT_TRUE(sparse_stage.execute(cpu_ctx_.get()));
            ASSERT_EQ(
                inbound_dispatch.live_row_count,
                static_cast<size_t>(logical_rows));
            ASSERT_TRUE(local_stage.execute(cpu_ctx_.get()));
            ASSERT_TRUE(return_stage.execute(cpu_ctx_.get()));
            ASSERT_TRUE(return_stage.manualGraphBoundaryComplete());
            ASSERT_TRUE(ticket.returnPayloadReady());

            for (int row = 0; row < logical_rows; ++row)
            {
                for (int col = 0; col < kDModel; ++col)
                {
                    const size_t index =
                        static_cast<size_t>(row) * kDModel + col;
                    EXPECT_NEAR(
                        ticket.return_rows_fp32[index],
                        reference->data()[index],
                        1e-4f)
                        << "row=" << row << " col=" << col;
                }
            }
            for (int row = logical_rows; row < bucket_rows; ++row)
            {
                for (int col = 0; col < kDModel; ++col)
                {
                    EXPECT_FLOAT_EQ(
                        ticket.return_rows_fp32[
                            static_cast<size_t>(row) * kDModel + col],
                        0.0f)
                        << "padding row=" << row << " col=" << col;
                }
            }
        };

        run_transaction(/*logical_rows=*/3, /*offset=*/0.0f);
        workspace.resetForStep(61, 1);
        run_transaction(/*logical_rows=*/2, /*offset=*/0.4f);

        EXPECT_EQ(ticket.header, header_address);
        EXPECT_EQ(ticket.hidden_rows_fp32, hidden_address);
        EXPECT_EQ(ticket.return_rows_fp32, return_address);
        EXPECT_TRUE(ticket_storage->hasValidBoundIdentity());

        ticket.header->workspace_generation = 62;
        EXPECT_FALSE(dispatch_stage.execute(cpu_ctx_.get()));
        EXPECT_FALSE(ticket_storage->hasValidBoundIdentity());
    }

} // namespace llaminar2::test
