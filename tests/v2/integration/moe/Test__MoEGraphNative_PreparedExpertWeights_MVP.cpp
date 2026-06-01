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
                          std::vector<bool> expert_mask = {true, true, true, true})
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

        ExpertRoutedTier routedTier(const std::string &name, const std::string &domain)
        {
            ExpertRoutedTier tier;
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
            dp.placement = ExpertLayerPlacement{.layer = kLayer, .routed_expert_tier = {0, 0, 0, 0}};
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
            dp.placement = ExpertLayerPlacement{.layer = kLayer, .routed_expert_tier = {0, 1, 0, 1}};
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

} // namespace llaminar2::test
