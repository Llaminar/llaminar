/**
 * @file Test__MoEExpertOverlay_CPUFallback_MPI.cpp
 * @brief MPI integration test for MoE expert-overlay CPU fallback NodeLocalTP execution.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/moe/MoEExpertOverlayCPUFallback.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cmath>
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
            0.0f, 1.0f,
            2.0f, 3.0f,
            1.0f, 2.0f,
        };
        const float route_weights[] = {
            0.60f, 0.40f,
            0.25f, 0.75f,
            0.55f, 0.45f,
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

    bool runReference(
        IDeviceContext *ctx,
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
        params.layer_idx = 0;
        params.expert_mask = std::move(expert_mask);
        params.routing_indices = routing_indices;
        params.routing_weights = routing_weights;
        params.output = output;

        if (!MoEExpertComputeStage::extractExpertViews(params) ||
            !MoEExpertComputeStage::prepareExpertGemmEngines(params))
        {
            return false;
        }

        MoEExpertComputeStage stage(std::move(params));
        return stage.execute(ctx);
    }

    ExpertRoutedTier routedTier(const std::string &name, const std::string &domain, bool fallback = false)
    {
        ExpertRoutedTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.fallback = fallback;
        return tier;
    }

    std::vector<int> fallbackTokenRows(
        IDeviceContext *ctx,
        TensorBase *input,
        TensorBase *routing_indices,
        TensorBase *routing_weights)
    {
        MoEExpertDispatchOutput dispatch;
        MoEExpertDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.routing_indices = routing_indices;
        params.routing_weights = routing_weights;
        params.hidden = input;
        params.seq_len = kSeqLen;
        params.top_k = kTopK;
        params.d_model = kDModel;
        params.continuation_domain = "gpu_hot";
        params.placement = ExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 0, 1, 1}};
        params.routed_tiers = {
            routedTier("hot", "gpu_hot"),
            routedTier("cold", "cpu_cold", true),
        };
        params.output = &dispatch;

        MoEExpertDispatchStage stage(std::move(params));
        EXPECT_TRUE(stage.execute(ctx));
        if (dispatch.tiers.size() < 2)
            return {};
        EXPECT_EQ(dispatch.tiers[1].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
        return dispatch.tiers[1].token_rows;
    }

    void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
    {
        ASSERT_EQ(actual->numel(), expected->numel());
        const float *actual_data = actual->data();
        const float *expected_data = expected->data();
        for (size_t i = 0; i < actual->numel(); ++i)
        {
            EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
        }
    }

    ExpertComputeDomain cpuFallbackDomain()
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

} // namespace

class Test__MoEExpertOverlay_CPUFallback_MPI : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        if (world_size_ != 3)
        {
            GTEST_SKIP() << "Test requires exactly 3 MPI ranks: one continuation root plus two CPU fallback ranks (got " << world_size_ << ")";
        }
        cpu_ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    void TearDown() override
    {
        if (world_size_ == 3)
            MPI_Barrier(MPI_COMM_WORLD);
    }

    int rank_ = -1;
    int world_size_ = 0;
    std::unique_ptr<llaminar2::testing::MockDeviceContext> cpu_ctx_;
};

TEST_F(Test__MoEExpertOverlay_CPUFallback_MPI, CreatesDomainScopedNodeLocalTPCommunicator)
{
    MoECPUFallbackDomainConfig config;
    config.domain = cpuFallbackDomain();
    config.root_world_rank = 0;
    config.domain_id = 6106;

    auto domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(config);

    EXPECT_EQ(domain.world_ranks, (std::vector<int>{1, 2}));
    EXPECT_EQ(domain.root_domain_index, -1);
    if (rank_ == 0)
    {
        EXPECT_FALSE(domain.participates());
        EXPECT_EQ(domain.tp_context, nullptr);
    }
    else
    {
        ASSERT_TRUE(domain.participates());
        ASSERT_NE(domain.tp_context, nullptr);
        EXPECT_EQ(domain.tp_context->degree(), 2);
        EXPECT_EQ(domain.tp_context->myIndex(), rank_ - 1);
        EXPECT_EQ(domain.tp_context->scope(), TPScope::NODE_LOCAL);
        EXPECT_EQ(domain.tp_context->domainId(), 6106);
        EXPECT_NE(domain.tp_context->communicator(), MPI_COMM_WORLD);
    }
}

TEST_F(Test__MoEExpertOverlay_CPUFallback_MPI, BroadcastsComputesReducesAndMatchesSingleDomainReference)
{
    MoECPUFallbackDomainConfig config;
    config.domain = cpuFallbackDomain();
    config.root_world_rank = 0;
    config.domain_id = 6107;
    auto domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(config);

    auto weights = makeWeights();

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    auto fallback_input = fp32({kSeqLen, kDModel});
    auto fallback_indices = fp32({kSeqLen, kTopK});
    auto fallback_route_weights = fp32({kSeqLen, kTopK});
    if (rank_ == 0)
    {
        std::copy(root_input->data(), root_input->data() + root_input->numel(), fallback_input->mutable_data());
        std::copy(root_indices->data(), root_indices->data() + root_indices->numel(), fallback_indices->mutable_data());
        std::copy(root_route_weights->data(), root_route_weights->data() + root_route_weights->numel(), fallback_route_weights->mutable_data());
    }
    else
    {
        std::fill_n(fallback_input->mutable_data(), fallback_input->numel(), -99.0f);
        std::fill_n(fallback_indices->mutable_data(), fallback_indices->numel(), -1.0f);
        std::fill_n(fallback_route_weights->mutable_data(), fallback_route_weights->numel(), -1.0f);
    }

    auto fallback_output = fp32({kSeqLen, kDModel});
    auto reference_output = fp32({kSeqLen, kDModel});
    ASSERT_TRUE(runReference(cpu_ctx_.get(), root_input.get(), root_indices.get(), root_route_weights.get(), weights, reference_output.get()));

    const std::vector<bool> all_fallback_experts{true, true, true, true};
    auto local_mask = MoEExpertOverlayCPUFallback::participantExpertMask(
        all_fallback_experts, kNumExperts, /*domain_degree=*/2, rank_ - 1);
    if (rank_ == 1)
        EXPECT_EQ(local_mask, (std::vector<bool>{true, false, true, false}));
    else if (rank_ == 2)
        EXPECT_EQ(local_mask, (std::vector<bool>{false, true, false, true}));
    else
        EXPECT_EQ(local_mask, (std::vector<bool>{false, false, false, false}));

    MoECPUFallbackRunParams run;
    run.input = fallback_input.get();
    run.routing_indices = fallback_indices.get();
    run.routing_weights = fallback_route_weights.get();
    run.gate_exps = weights.gate.get();
    run.up_exps = weights.up.get();
    run.down_exps = weights.down.get();
    run.output = fallback_output.get();
    run.seq_len = kSeqLen;
    run.d_model = kDModel;
    run.num_experts = kNumExperts;
    run.top_k = kTopK;
    run.expert_intermediate = kIntermediate;
    run.layer_idx = 0;
    run.fallback_expert_mask = all_fallback_experts;

    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runReplicatedExpertFallback(domain, run, cpu_ctx_.get()));

    expectTensorNear(fallback_input.get(), root_input.get());
    expectTensorNear(fallback_indices.get(), root_indices.get());
    expectTensorNear(fallback_route_weights.get(), root_route_weights.get());
    expectTensorNear(fallback_output.get(), reference_output.get());
}

TEST_F(Test__MoEExpertOverlay_CPUFallback_MPI, GraphIntegratedStageBroadcastsComputesAndReduces)
{
    auto weights = makeWeights();

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    auto stage_input = fp32({kSeqLen, kDModel});
    auto stage_indices = fp32({kSeqLen, kTopK});
    auto stage_route_weights = fp32({kSeqLen, kTopK});
    if (rank_ == 0)
    {
        std::copy(root_input->data(), root_input->data() + root_input->numel(), stage_input->mutable_data());
        std::copy(root_indices->data(), root_indices->data() + root_indices->numel(), stage_indices->mutable_data());
        std::copy(root_route_weights->data(), root_route_weights->data() + root_route_weights->numel(), stage_route_weights->mutable_data());
    }
    else
    {
        std::fill_n(stage_input->mutable_data(), stage_input->numel(), -99.0f);
        std::fill_n(stage_indices->mutable_data(), stage_indices->numel(), -1.0f);
        std::fill_n(stage_route_weights->mutable_data(), stage_route_weights->numel(), -1.0f);
    }

    auto stage_output = fp32({kSeqLen, kDModel});
    auto reference_output = fp32({kSeqLen, kDModel});
    ASSERT_TRUE(runReference(cpu_ctx_.get(), root_input.get(), root_indices.get(), root_route_weights.get(), weights, reference_output.get()));

    MoEExpertOverlayCPUFallbackStage::Params params;
    params.device_id = DeviceId::cpu();
    params.domain = cpuFallbackDomain();
    params.root_world_rank = 0;
    params.domain_id = 6111;
    params.input = stage_input.get();
    params.routing_indices = stage_indices.get();
    params.routing_weights = stage_route_weights.get();
    params.gate_exps = weights.gate.get();
    params.up_exps = weights.up.get();
    params.down_exps = weights.down.get();
    params.output = stage_output.get();
    params.seq_len = kSeqLen;
    params.d_model = kDModel;
    params.num_experts = kNumExperts;
    params.top_k = kTopK;
    params.expert_intermediate = kIntermediate;
    params.layer_idx = 0;
    params.expert_mask = {true, true, true, true};

    MoEExpertOverlayCPUFallbackStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    expectTensorNear(stage_input.get(), root_input.get());
    expectTensorNear(stage_indices.get(), root_indices.get());
    expectTensorNear(stage_route_weights.get(), root_route_weights.get());
    expectTensorNear(stage_output.get(), reference_output.get());
}

TEST_F(Test__MoEExpertOverlay_CPUFallback_MPI, SparseTransferSendsOnlyFallbackRowsAndMatchesMaskedReference)
{
    MoECPUFallbackDomainConfig config;
    config.domain = cpuFallbackDomain();
    config.root_world_rank = 0;
    config.domain_id = 6108;
    auto domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(config);

    auto weights = makeWeights();

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    const auto token_rows = fallbackTokenRows(cpu_ctx_.get(),
                                             root_input.get(),
                                             root_indices.get(),
                                             root_route_weights.get());
    EXPECT_EQ(token_rows, (std::vector<int>{1, 2}));

    auto fallback_input = fp32({kSeqLen, kDModel});
    auto fallback_indices = fp32({kSeqLen, kTopK});
    auto fallback_route_weights = fp32({kSeqLen, kTopK});
    if (rank_ == 0)
    {
        std::copy(root_input->data(), root_input->data() + root_input->numel(), fallback_input->mutable_data());
        std::copy(root_indices->data(), root_indices->data() + root_indices->numel(), fallback_indices->mutable_data());
        std::copy(root_route_weights->data(), root_route_weights->data() + root_route_weights->numel(), fallback_route_weights->mutable_data());
    }
    else
    {
        std::fill_n(fallback_input->mutable_data(), fallback_input->numel(), -99.0f);
        std::fill_n(fallback_indices->mutable_data(), fallback_indices->numel(), -1.0f);
        std::fill_n(fallback_route_weights->mutable_data(), fallback_route_weights->numel(), -1.0f);
    }

    auto fallback_output = fp32({kSeqLen, kDModel});
    std::fill_n(fallback_output->mutable_data(), fallback_output->numel(), 0.0f);
    auto reference_output = fp32({kSeqLen, kDModel});
    ASSERT_TRUE(runReference(cpu_ctx_.get(),
                             root_input.get(),
                             root_indices.get(),
                             root_route_weights.get(),
                             weights,
                             reference_output.get(),
                             {false, false, true, true}));

    const std::vector<bool> cold_fallback_experts{false, false, true, true};
    MoECPUFallbackTransferStats transfer_stats;
    MoECPUFallbackRunParams run;
    run.input = fallback_input.get();
    run.routing_indices = fallback_indices.get();
    run.routing_weights = fallback_route_weights.get();
    run.gate_exps = weights.gate.get();
    run.up_exps = weights.up.get();
    run.down_exps = weights.down.get();
    run.output = fallback_output.get();
    run.seq_len = kSeqLen;
    run.d_model = kDModel;
    run.num_experts = kNumExperts;
    run.top_k = kTopK;
    run.expert_intermediate = kIntermediate;
    run.layer_idx = 0;
    run.fallback_expert_mask = cold_fallback_experts;
    run.transfer_token_rows = rank_ == 0 ? token_rows : std::vector<int>{};
    run.transfer_stats = &transfer_stats;

    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runReplicatedExpertFallback(domain, run, cpu_ctx_.get()));

    if (rank_ == 0)
        expectTensorNear(fallback_output.get(), reference_output.get());
    EXPECT_EQ(transfer_stats.mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(transfer_stats.token_rows, token_rows);
    EXPECT_EQ(transfer_stats.volume.selected_rows, 2u);
    EXPECT_EQ(transfer_stats.volume.outbound_bytes,
              2u * (static_cast<size_t>(kDModel) * sizeof(float) + static_cast<size_t>(kTopK) * 2u * sizeof(float)));
    EXPECT_LT(transfer_stats.volume.totalBytes(), transfer_stats.volume.denseTotalBytes());
}

} // namespace llaminar2::test