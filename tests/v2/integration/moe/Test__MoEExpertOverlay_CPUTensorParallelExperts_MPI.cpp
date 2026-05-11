/**
 * @file Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI.cpp
 * @brief MPI integration test for CPU TensorParallelExperts MoE fallback execution.
 */

#include <gtest/gtest.h>
#include <mpi.h>

#include "execution/moe/MoEExpertOverlayCPUFallback.h"
#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "mocks/MockComputeStage.h"
#include "tensors/FP16Utils.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <random>
#include <stdexcept>
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
        std::shared_ptr<TensorBase> gate;
        std::shared_ptr<TensorBase> up;
        std::shared_ptr<TensorBase> down;
    };

    ExpertWeights makeWeights()
    {
        ExpertWeights weights;
        auto gate = fp32({kDModel, kIntermediate, kNumExperts});
        auto up = fp32({kDModel, kIntermediate, kNumExperts});
        auto down = fp32({kIntermediate, kDModel, kNumExperts});
        fillExpertTensor(gate.get(), 0.010f);
        fillExpertTensor(up.get(), 0.012f);
        fillExpertTensor(down.get(), 0.008f);
        weights.gate = std::move(gate);
        weights.up = std::move(up);
        weights.down = std::move(down);
        return weights;
    }

    std::shared_ptr<Q4_KTensor> makeQ4KExpertTensor3D(
        const std::vector<size_t> &shape,
        uint32_t seed)
    {
        if (shape.size() != 3)
            throw std::invalid_argument("makeQ4KExpertTensor3D requires a 3D GGUF expert shape");

        constexpr size_t kBlockSize = Q4_KBlock::BLOCK_SIZE;
        const size_t cols = shape[0];
        const size_t rows_per_expert = shape[1];
        const size_t expert_count = shape[2];
        const size_t blocks_per_row = (cols + kBlockSize - 1) / kBlockSize;
        const size_t total_blocks = rows_per_expert * expert_count * blocks_per_row;

        std::vector<uint8_t> raw_data(total_blocks * sizeof(Q4_KBlock));
        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> qdist(0, 15);

        for (size_t block_index = 0; block_index < total_blocks; ++block_index)
        {
            blocks[block_index].d = fp32_to_fp16(0.002f + 0.0002f * static_cast<float>(block_index % 7));
            blocks[block_index].dmin = fp32_to_fp16(0.0f);
            std::memset(blocks[block_index].scales, 0x44, sizeof(blocks[block_index].scales));
            for (size_t byte_index = 0; byte_index < sizeof(blocks[block_index].qs); ++byte_index)
            {
                const uint8_t lo = static_cast<uint8_t>(qdist(rng));
                const uint8_t hi = static_cast<uint8_t>(qdist(rng));
                blocks[block_index].qs[byte_index] = static_cast<uint8_t>((hi << 4) | lo);
            }
        }

        return std::make_shared<Q4_KTensor>(shape, raw_data);
    }

    std::shared_ptr<FP32Tensor> dequantizeExpertTensor3D(
        TensorBase *source,
        int rows,
        int cols)
    {
        auto dequantized = fp32(source->shape());
        float *dst = dequantized->mutable_data();
        const size_t expert_elements = static_cast<size_t>(rows) * static_cast<size_t>(cols);
        const int expert_count = static_cast<int>(source->shape()[2]);
        for (int expert = 0; expert < expert_count; ++expert)
        {
            const size_t offset = static_cast<size_t>(expert) * expert_elements;
            auto view = source->create_view({static_cast<size_t>(rows), static_cast<size_t>(cols)}, offset);
            if (!view)
                throw std::runtime_error("failed to create expert dequantization view");
            const float *view_data = view->data();
            if (!view_data)
                throw std::runtime_error("expert dequantization view returned null data");
            std::copy_n(view_data, expert_elements, dst + offset);
        }
        return dequantized;
    }

    ExpertWeights makeQ4KWeights()
    {
        ExpertWeights weights;
        weights.gate = makeQ4KExpertTensor3D({kDModel, kIntermediate, kNumExperts}, 101);
        weights.up = makeQ4KExpertTensor3D({kDModel, kIntermediate, kNumExperts}, 211);
        weights.down = makeQ4KExpertTensor3D({kIntermediate, kDModel, kNumExperts}, 307);
        return weights;
    }

    ExpertWeights dequantizeExpertWeights(const ExpertWeights &source)
    {
        ExpertWeights weights;
        weights.gate = dequantizeExpertTensor3D(source.gate.get(), kIntermediate, kDModel);
        weights.up = dequantizeExpertTensor3D(source.up.get(), kIntermediate, kDModel);
        weights.down = dequantizeExpertTensor3D(source.down.get(), kDModel, kIntermediate);
        return weights;
    }

    void seedFallbackInputs(FP32Tensor *input,
                            FP32Tensor *indices,
                            FP32Tensor *weights,
                            const FP32Tensor *root_input,
                            const FP32Tensor *root_indices,
                            const FP32Tensor *root_weights,
                            int rank)
    {
        if (rank == 0)
        {
            std::copy(root_input->data(), root_input->data() + root_input->numel(), input->mutable_data());
            std::copy(root_indices->data(), root_indices->data() + root_indices->numel(), indices->mutable_data());
            std::copy(root_weights->data(), root_weights->data() + root_weights->numel(), weights->mutable_data());
        }
        else
        {
            std::fill_n(input->mutable_data(), input->numel(), -99.0f);
            std::fill_n(indices->mutable_data(), indices->numel(), -1.0f);
            std::fill_n(weights->mutable_data(), weights->numel(), -1.0f);
        }
    }

    MoECPUFallbackRunParams makeRunParams(
        TensorBase *input,
        TensorBase *routing_indices,
        TensorBase *routing_weights,
        const ExpertWeights &weights,
        TensorBase *output,
        const std::vector<bool> &fallback_experts)
    {
        MoECPUFallbackRunParams run;
        run.input = input;
        run.routing_indices = routing_indices;
        run.routing_weights = routing_weights;
        run.gate_exps = weights.gate.get();
        run.up_exps = weights.up.get();
        run.down_exps = weights.down.get();
        run.output = output;
        run.seq_len = kSeqLen;
        run.d_model = kDModel;
        run.num_experts = kNumExperts;
        run.top_k = kTopK;
        run.expert_intermediate = kIntermediate;
        run.layer_idx = 0;
        run.fallback_expert_mask = fallback_experts;
        return run;
    }

    void expectTensorNear(const TensorBase *actual, const TensorBase *expected, float tolerance = 1e-4f)
    {
        ASSERT_EQ(actual->numel(), expected->numel());
        const float *actual_data = actual->data();
        const float *expected_data = expected->data();
        for (size_t i = 0; i < actual->numel(); ++i)
            EXPECT_NEAR(actual_data[i], expected_data[i], tolerance) << "index=" << i;
    }

    ExpertComputeDomain cpuFallbackDomain(ExpertDomainComputeKind compute_kind)
    {
        ExpertComputeDomain domain;
        domain.name = compute_kind == ExpertDomainComputeKind::TensorParallelExperts
                          ? "cpu_cold_tp"
                          : "cpu_cold_replicated";
        domain.kind = ExpertDomainKind::NodeLocalTP;
        domain.backend = CollectiveBackendType::UPI;
        domain.participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
        domain.world_ranks = {1, 2};
        domain.owner_rank = 1;
        domain.compute_kind = compute_kind;
        return domain;
    }

    void runQ4KTensorParallelBridgeCase(
        int rank,
        IDeviceContext *cpu_ctx,
        bool sparse_transfer)
    {
        MoECPUFallbackDomainConfig q4_config;
        q4_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::TensorParallelExperts);
        q4_config.root_world_rank = 0;
        q4_config.domain_id = sparse_transfer ? 6224 : 6220;
        auto q4_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(q4_config);

        MoECPUFallbackDomainConfig fp32_config;
        fp32_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::TensorParallelExperts);
        fp32_config.root_world_rank = 0;
        fp32_config.domain_id = sparse_transfer ? 6225 : 6221;
        auto fp32_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(fp32_config);

        auto q4_weights = makeQ4KWeights();
        auto fp32_weights = dequantizeExpertWeights(q4_weights);
        const std::vector<bool> fallback_experts = sparse_transfer
                                                       ? std::vector<bool>{false, false, true, true}
                                                       : std::vector<bool>{true, true, true, true};
        const std::vector<int> token_rows = sparse_transfer
                                                ? std::vector<int>{1, 2}
                                                : std::vector<int>{};

        auto root_input = fp32({kSeqLen, kDModel});
        auto root_indices = fp32({kSeqLen, kTopK});
        auto root_route_weights = fp32({kSeqLen, kTopK});
        fillHidden(root_input.get());
        fillRouting(root_indices.get(), root_route_weights.get());

        auto q4_input = fp32({kSeqLen, kDModel});
        auto q4_indices = fp32({kSeqLen, kTopK});
        auto q4_route_weights = fp32({kSeqLen, kTopK});
        seedFallbackInputs(q4_input.get(),
                           q4_indices.get(),
                           q4_route_weights.get(),
                           root_input.get(),
                           root_indices.get(),
                           root_route_weights.get(),
                           rank);

        auto fp32_input = fp32({kSeqLen, kDModel});
        auto fp32_indices = fp32({kSeqLen, kTopK});
        auto fp32_route_weights = fp32({kSeqLen, kTopK});
        seedFallbackInputs(fp32_input.get(),
                           fp32_indices.get(),
                           fp32_route_weights.get(),
                           root_input.get(),
                           root_indices.get(),
                           root_route_weights.get(),
                           rank);

        auto q4_output = fp32({kSeqLen, kDModel});
        auto fp32_output = fp32({kSeqLen, kDModel});
        std::fill_n(q4_output->mutable_data(), q4_output->numel(), 0.0f);
        std::fill_n(fp32_output->mutable_data(), fp32_output->numel(), 0.0f);

        MoECPUFallbackTensorParallelStats q4_stats;
        MoECPUFallbackTransferStats q4_transfer_stats;
        auto q4_run = makeRunParams(q4_input.get(),
                                    q4_indices.get(),
                                    q4_route_weights.get(),
                                    q4_weights,
                                    q4_output.get(),
                                    fallback_experts);
        q4_run.tensor_parallel_stats = &q4_stats;
        q4_run.transfer_stats = &q4_transfer_stats;
        if (sparse_transfer)
            q4_run.transfer_token_rows = rank == 0 ? token_rows : std::vector<int>{};
        ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
            q4_domain, q4_run, cpu_ctx));

        MoECPUFallbackTensorParallelStats fp32_stats;
        auto fp32_run = makeRunParams(fp32_input.get(),
                                      fp32_indices.get(),
                                      fp32_route_weights.get(),
                                      fp32_weights,
                                      fp32_output.get(),
                                      fallback_experts);
        fp32_run.tensor_parallel_stats = &fp32_stats;
        if (sparse_transfer)
            fp32_run.transfer_token_rows = rank == 0 ? token_rows : std::vector<int>{};
        ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
            fp32_domain, fp32_run, cpu_ctx));

        if (rank == 0)
            expectTensorNear(q4_output.get(), fp32_output.get());

        if (sparse_transfer)
        {
            EXPECT_EQ(q4_transfer_stats.mode, MoEExpertTransferMode::SparseTokenRows);
            EXPECT_EQ(q4_transfer_stats.token_rows, token_rows);
        }

        if (rank == 1 || rank == 2)
        {
            const std::vector<int> expected_experts = sparse_transfer
                                                          ? std::vector<int>{2, 3}
                                                          : std::vector<int>{0, 1, 2, 3};
            EXPECT_EQ(q4_stats.processed_expert_ids, expected_experts);
            EXPECT_EQ(fp32_stats.processed_expert_ids, expected_experts);
        }
    }

} // namespace

class Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI : public ::testing::Test
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

TEST_F(Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI, MatchesReplicatedFallbackAndUsesBothRanksForSameExperts)
{
    MoECPUFallbackDomainConfig replicated_config;
    replicated_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::ReplicatedExperts);
    replicated_config.root_world_rank = 0;
    replicated_config.domain_id = 6207;
    auto replicated_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(replicated_config);

    MoECPUFallbackDomainConfig tp_config;
    tp_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::TensorParallelExperts);
    tp_config.root_world_rank = 0;
    tp_config.domain_id = 6208;
    auto tp_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(tp_config);

    auto weights = makeWeights();
    const std::vector<bool> all_fallback_experts{true, true, true, true};

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    auto replicated_input = fp32({kSeqLen, kDModel});
    auto replicated_indices = fp32({kSeqLen, kTopK});
    auto replicated_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(replicated_input.get(),
                       replicated_indices.get(),
                       replicated_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto tp_input = fp32({kSeqLen, kDModel});
    auto tp_indices = fp32({kSeqLen, kTopK});
    auto tp_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(tp_input.get(),
                       tp_indices.get(),
                       tp_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto replicated_output = fp32({kSeqLen, kDModel});
    auto tp_output = fp32({kSeqLen, kDModel});

    auto replicated_run = makeRunParams(replicated_input.get(),
                                        replicated_indices.get(),
                                        replicated_route_weights.get(),
                                        weights,
                                        replicated_output.get(),
                                        all_fallback_experts);
    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
        replicated_domain, replicated_run, cpu_ctx_.get()));

    MoECPUFallbackTensorParallelStats tp_stats;
    auto tp_run = makeRunParams(tp_input.get(),
                                tp_indices.get(),
                                tp_route_weights.get(),
                                weights,
                                tp_output.get(),
                                all_fallback_experts);
    tp_run.tensor_parallel_stats = &tp_stats;
    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
        tp_domain, tp_run, cpu_ctx_.get()));

    expectTensorNear(tp_input.get(), root_input.get());
    expectTensorNear(tp_indices.get(), root_indices.get());
    expectTensorNear(tp_route_weights.get(), root_route_weights.get());
    expectTensorNear(tp_output.get(), replicated_output.get());

    if (rank_ == 1 || rank_ == 2)
    {
        EXPECT_EQ(tp_stats.domain_degree, 2);
        EXPECT_EQ(tp_stats.domain_index, rank_ - 1);
        EXPECT_EQ(tp_stats.processed_expert_ids, (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(tp_stats.expert_allreduce_count, 4);

        if (rank_ == 1)
        {
            EXPECT_EQ(tp_stats.intermediate_start, 0);
            EXPECT_EQ(tp_stats.intermediate_end, 2);
        }
        else
        {
            EXPECT_EQ(tp_stats.intermediate_start, 2);
            EXPECT_EQ(tp_stats.intermediate_end, 4);
        }
    }
    else
    {
        EXPECT_TRUE(tp_stats.processed_expert_ids.empty());
        EXPECT_EQ(tp_stats.expert_allreduce_count, 0);
    }
}

TEST_F(Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI, TensorParallelFallbackConsumesQ4KExpertWeightsViaHostDequantBridge)
{
    runQ4KTensorParallelBridgeCase(rank_, cpu_ctx_.get(), false);
}

TEST_F(Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI, SparseTensorParallelFallbackConsumesQ4KExpertWeightsViaHostDequantBridge)
{
    runQ4KTensorParallelBridgeCase(rank_, cpu_ctx_.get(), true);
}

TEST_F(Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI, GraphIntegratedStageUsesBothRanksForTensorParallelExperts)
{
    MoECPUFallbackDomainConfig replicated_config;
    replicated_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::ReplicatedExperts);
    replicated_config.root_world_rank = 0;
    replicated_config.domain_id = 6211;
    auto replicated_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(replicated_config);

    auto weights = makeWeights();
    const std::vector<bool> all_fallback_experts{true, true, true, true};

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    auto replicated_input = fp32({kSeqLen, kDModel});
    auto replicated_indices = fp32({kSeqLen, kTopK});
    auto replicated_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(replicated_input.get(),
                       replicated_indices.get(),
                       replicated_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto stage_input = fp32({kSeqLen, kDModel});
    auto stage_indices = fp32({kSeqLen, kTopK});
    auto stage_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(stage_input.get(),
                       stage_indices.get(),
                       stage_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto replicated_output = fp32({kSeqLen, kDModel});
    auto stage_output = fp32({kSeqLen, kDModel});

    auto replicated_run = makeRunParams(replicated_input.get(),
                                        replicated_indices.get(),
                                        replicated_route_weights.get(),
                                        weights,
                                        replicated_output.get(),
                                        all_fallback_experts);
    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
        replicated_domain, replicated_run, cpu_ctx_.get()));

    MoECPUFallbackTensorParallelStats tp_stats;
    MoEExpertOverlayCPUFallbackStage::Params params;
    params.device_id = DeviceId::cpu();
    params.domain = cpuFallbackDomain(ExpertDomainComputeKind::TensorParallelExperts);
    params.root_world_rank = 0;
    params.domain_id = 6212;
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
    params.expert_mask = all_fallback_experts;
    params.tensor_parallel_stats = &tp_stats;

    MoEExpertOverlayCPUFallbackStage stage(std::move(params));
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    expectTensorNear(stage_input.get(), root_input.get());
    expectTensorNear(stage_indices.get(), root_indices.get());
    expectTensorNear(stage_route_weights.get(), root_route_weights.get());
    expectTensorNear(stage_output.get(), replicated_output.get());

    if (rank_ == 1 || rank_ == 2)
    {
        EXPECT_EQ(tp_stats.domain_degree, 2);
        EXPECT_EQ(tp_stats.domain_index, rank_ - 1);
        EXPECT_EQ(tp_stats.processed_expert_ids, (std::vector<int>{0, 1, 2, 3}));
        EXPECT_EQ(tp_stats.expert_allreduce_count, 4);
    }
}

TEST_F(Test__MoEExpertOverlay_CPUTensorParallelExperts_MPI, SparseTransferMatchesDenseReplicatedFallbackForColdExperts)
{
    MoECPUFallbackDomainConfig replicated_config;
    replicated_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::ReplicatedExperts);
    replicated_config.root_world_rank = 0;
    replicated_config.domain_id = 6209;
    auto replicated_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(replicated_config);

    MoECPUFallbackDomainConfig tp_config;
    tp_config.domain = cpuFallbackDomain(ExpertDomainComputeKind::TensorParallelExperts);
    tp_config.root_world_rank = 0;
    tp_config.domain_id = 6210;
    auto tp_domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(tp_config);

    auto weights = makeWeights();
    const std::vector<bool> cold_fallback_experts{false, false, true, true};
    const std::vector<int> cold_token_rows{1, 2};

    auto root_input = fp32({kSeqLen, kDModel});
    auto root_indices = fp32({kSeqLen, kTopK});
    auto root_route_weights = fp32({kSeqLen, kTopK});
    fillHidden(root_input.get());
    fillRouting(root_indices.get(), root_route_weights.get());

    auto dense_input = fp32({kSeqLen, kDModel});
    auto dense_indices = fp32({kSeqLen, kTopK});
    auto dense_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(dense_input.get(),
                       dense_indices.get(),
                       dense_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto sparse_input = fp32({kSeqLen, kDModel});
    auto sparse_indices = fp32({kSeqLen, kTopK});
    auto sparse_route_weights = fp32({kSeqLen, kTopK});
    seedFallbackInputs(sparse_input.get(),
                       sparse_indices.get(),
                       sparse_route_weights.get(),
                       root_input.get(),
                       root_indices.get(),
                       root_route_weights.get(),
                       rank_);

    auto dense_output = fp32({kSeqLen, kDModel});
    auto sparse_output = fp32({kSeqLen, kDModel});
    std::fill_n(dense_output->mutable_data(), dense_output->numel(), 0.0f);
    std::fill_n(sparse_output->mutable_data(), sparse_output->numel(), 0.0f);

    auto dense_run = makeRunParams(dense_input.get(),
                                   dense_indices.get(),
                                   dense_route_weights.get(),
                                   weights,
                                   dense_output.get(),
                                   cold_fallback_experts);
    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
        replicated_domain, dense_run, cpu_ctx_.get()));

    MoECPUFallbackTensorParallelStats tp_stats;
    MoECPUFallbackTransferStats transfer_stats;
    auto sparse_run = makeRunParams(sparse_input.get(),
                                    sparse_indices.get(),
                                    sparse_route_weights.get(),
                                    weights,
                                    sparse_output.get(),
                                    cold_fallback_experts);
    sparse_run.transfer_token_rows = rank_ == 0 ? cold_token_rows : std::vector<int>{};
    sparse_run.tensor_parallel_stats = &tp_stats;
    sparse_run.transfer_stats = &transfer_stats;
    ASSERT_TRUE(MoEExpertOverlayCPUFallback::runExpertFallback(
        tp_domain, sparse_run, cpu_ctx_.get()));

    if (rank_ == 0)
        expectTensorNear(sparse_output.get(), dense_output.get());

    EXPECT_EQ(transfer_stats.mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(transfer_stats.token_rows, cold_token_rows);
    EXPECT_EQ(transfer_stats.volume.selected_rows, 2u);
    EXPECT_LT(transfer_stats.volume.totalBytes(), transfer_stats.volume.denseTotalBytes());

    if (rank_ == 1 || rank_ == 2)
    {
        EXPECT_EQ(tp_stats.processed_expert_ids, (std::vector<int>{2, 3}));
        EXPECT_EQ(tp_stats.expert_allreduce_count, 2);
    }
}

} // namespace llaminar2::test