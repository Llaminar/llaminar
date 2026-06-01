#include "execution/compute_stages/ComputeStageFactory.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2::test
{
namespace
{

    ExpertRoutedTier tier(const std::string &name, const std::string &domain, bool fallback = false)
    {
        ExpertRoutedTier result;
        result.name = name;
        result.domain = domain;
        result.fallback = fallback;
        return result;
    }

    ExpertLayerPlacement placement(std::vector<int> routed_expert_tier)
    {
        ExpertLayerPlacement result;
        result.layer = 4;
        result.routed_expert_tier = std::move(routed_expert_tier);
        return result;
    }

    std::unique_ptr<FP32Tensor> routingTensor(const std::vector<size_t> &shape, const std::vector<float> &values)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        std::copy(values.begin(), values.end(), tensor->mutable_data());
        return tensor;
    }

    MoEExpertDispatchStage::Params paramsFor(
        const ITensor *indices,
        const ITensor *weights,
        int seq_len,
        int top_k,
        std::optional<ExpertLayerPlacement> expert_placement,
        std::vector<ExpertRoutedTier> routed_tiers,
        MoEExpertDispatchOutput *output)
    {
        MoEExpertDispatchStage::Params params;
        params.device_id = DeviceId::cpu();
        params.routing_indices = indices;
        params.routing_weights = weights;
        params.seq_len = seq_len;
        params.top_k = top_k;
        params.d_model = 8;
        params.placement = std::move(expert_placement);
        params.routed_tiers = std::move(routed_tiers);
        params.output = output;
        return params;
    }

    int countEntry(
        const MoEExpertDispatchOutput &output,
        int tier_index,
        int token_row,
        int route_slot,
        int expert_id,
        float route_weight)
    {
        const auto &entries = output.tiers.at(static_cast<size_t>(tier_index)).entries;
        return static_cast<int>(std::count_if(entries.begin(), entries.end(), [&](const auto &entry) {
            return entry.token_row == token_row &&
                   entry.route_slot == route_slot &&
                   entry.expert_id == expert_id &&
                   entry.route_weight == route_weight;
        }));
    }

    void expectContributionExactlyOnce(
        const MoEExpertDispatchOutput &output,
        int tier_index,
        int token_row,
        int route_slot,
        int expert_id,
        float route_weight)
    {
        EXPECT_EQ(countEntry(output, tier_index, token_row, route_slot, expert_id, route_weight), 1)
            << "token=" << token_row << " slot=" << route_slot << " expert=" << expert_id
            << " tier=" << tier_index;
    }

    bool hasInputBinding(const StageBufferContract &contract, BufferId id)
    {
        return std::any_of(contract.inputs.begin(), contract.inputs.end(),
                           [id](const BufferBinding &binding)
                           {
                               return binding.id == id && binding.access == BufferAccess::READ;
                           });
    }

} // namespace

class Test__MoEExpertDispatchStage : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<llaminar2::testing::MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    std::unique_ptr<llaminar2::testing::MockDeviceContext> ctx_;
};

TEST_F(Test__MoEExpertDispatchStage, PrefillTwoTiersRoutesEveryContributionOnceAndStableTokenRows)
{
    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 3.0f,
                                                    2.0f, 1.0f,
                                                    1.0f, 0.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.25f, 0.75f,
                                                    0.40f, 0.60f,
                                                    0.55f, 0.45f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);
    MoEExpertDispatchStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_DISPATCH);
    EXPECT_EQ(output.seq_len, seq_len);
    EXPECT_EQ(output.top_k, top_k);
    EXPECT_EQ(output.d_model, 8);
    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].tier_name, "hot");
    EXPECT_EQ(output.tiers[0].domain, "gpu_hot");
    EXPECT_EQ(output.tiers[1].tier_name, "cold");
    EXPECT_EQ(output.tiers[1].domain, "cpu_cold");

    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[0].entries.size(), 3u);
    EXPECT_EQ(output.tiers[1].entries.size(), 3u);

    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.25f);
    expectContributionExactlyOnce(output, 1, 0, 1, 3, 0.75f);
    expectContributionExactlyOnce(output, 0, 1, 0, 2, 0.40f);
    expectContributionExactlyOnce(output, 1, 1, 1, 1, 0.60f);
    expectContributionExactlyOnce(output, 1, 2, 0, 1, 0.55f);
    expectContributionExactlyOnce(output, 0, 2, 1, 0, 0.45f);
}

TEST_F(Test__MoEExpertDispatchStage, AdvertisesArenaInputContractForHostDispatch)
{
    auto indices = routingTensor({1, 2}, {0.0f, 1.0f});
    auto weights = routingTensor({1, 2}, {0.75f, 0.25f});
    auto hidden = TestTensorFactory::createFP32({1, 8});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 1, 2,
                            placement({0, 1}),
                            {tier("hot", "gpu"), tier("cold", "cpu", true)},
                            &output);
    params.hidden = hidden.get();
    params.routing_indices_buffer_id = BufferId::MOE_EXPERT_INDICES;
    params.routing_weights_buffer_id = BufferId::MOE_EXPERT_WEIGHTS;
    params.hidden_buffer_id = BufferId::NORMALIZED;

    MoEExpertDispatchStage stage(params);
    const auto contract = stage.bufferContract();

    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_INDICES));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::MOE_EXPERT_WEIGHTS));
    EXPECT_TRUE(hasInputBinding(contract, BufferId::NORMALIZED));
}

TEST_F(Test__MoEExpertDispatchStage, ArenaCapacityRoutingBuffersUseLivePrefixRows)
{
    constexpr int seq_len = 2;
    constexpr int capacity = 4;
    constexpr int top_k = 2;
    auto indices = routingTensor({capacity, top_k}, {0.0f, 1.0f,
                                                     2.0f, 3.0f,
                                                     0.0f, 0.0f,
                                                     0.0f, 0.0f});
    auto weights = routingTensor({capacity, top_k}, {0.25f, 0.75f,
                                                     0.40f, 0.60f,
                                                     0.0f, 0.0f,
                                                     0.0f, 0.0f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);
    MoEExpertDispatchStage stage(params);

    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 2u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1}));
}

TEST_F(Test__MoEExpertDispatchStage, DecodeOneTokenWorksNaturally)
{
    constexpr int seq_len = 1;
    constexpr int top_k = 3;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 2.0f, 3.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.10f, 0.20f, 0.70f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 1u);
    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.10f);
    expectContributionExactlyOnce(output, 0, 0, 1, 2, 0.20f);
    expectContributionExactlyOnce(output, 1, 0, 2, 3, 0.70f);
}

TEST_F(Test__MoEExpertDispatchStage, PrefillTransferMetadataMarksNonContinuationTiersSparse)
{
    constexpr int seq_len = 4;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 1.0f,
                                                    2.0f, 3.0f,
                                                    4.0f, 5.0f,
                                                    0.0f, 5.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 0, 1, 1, 2, 2}),
                            {tier("hottest", "cuda_fast"),
                             tier("hot", "rocm_hot"),
                             tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "cuda_fast";

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 3u);
    EXPECT_FALSE(output.tiers[0].transfer_required);
    EXPECT_EQ(output.tiers[0].transfer_mode, MoEExpertTransferMode::None);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{1}));
    EXPECT_TRUE(output.tiers[2].transfer_required);
    EXPECT_EQ(output.tiers[2].transfer_mode, MoEExpertTransferMode::SparseTokenRows);
    EXPECT_EQ(output.tiers[2].token_rows, (std::vector<int>{2, 3}));

    const size_t hidden_row_bytes = 8u * sizeof(float);
    const size_t routing_row_bytes = top_k * 2u * sizeof(float);
    EXPECT_EQ(output.tiers[2].transfer_volume.outbound_bytes,
              2u * (hidden_row_bytes + routing_row_bytes));
    EXPECT_EQ(output.tiers[2].transfer_volume.return_bytes,
              2u * hidden_row_bytes);
    EXPECT_LT(output.estimatedTransferBytes(), output.denseTransferBytes());
}

TEST_F(Test__MoEExpertDispatchStage, DecodeTransferMetadataUsesOneTokenFastPath)
{
    constexpr int seq_len = 1;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 3.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.25f, 0.75f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 0, 1, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "gpu_hot";

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_FALSE(output.tiers[0].transfer_required);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0}));
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::DecodeOneToken);
    EXPECT_EQ(output.tiers[1].transfer_volume.outbound_bytes,
              8u * sizeof(float) + top_k * 2u * sizeof(float));
    EXPECT_EQ(output.tiers[1].transfer_volume.return_bytes,
              8u * sizeof(float));
}

TEST_F(Test__MoEExpertDispatchStage, DenseTransferCompatibilityModeReportsFullSequenceBytes)
{
    constexpr int seq_len = 3;
    constexpr int top_k = 2;
    auto indices = routingTensor({seq_len, top_k}, {0.0f, 1.0f,
                                                    0.0f, 1.0f,
                                                    0.0f, 1.0f});
    auto weights = routingTensor({seq_len, top_k}, {0.5f, 0.5f,
                                                    0.5f, 0.5f,
                                                    0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold", true)},
                            &output);
    params.continuation_domain = "gpu_hot";
    params.transfer_mode = MoEExpertTransferMode::DenseFullSequence;

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 2u);
    EXPECT_TRUE(output.tiers[1].transfer_required);
    EXPECT_EQ(output.tiers[1].transfer_mode, MoEExpertTransferMode::DenseFullSequence);
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1, 2}));
    EXPECT_EQ(output.tiers[1].transfer_volume.totalBytes(),
              output.tiers[1].transfer_volume.denseTotalBytes());
}

TEST_F(Test__MoEExpertDispatchStage, FlatPrefillThreeTiersRoutesEveryContributionOnce)
{
    constexpr int seq_len = 2;
    constexpr int top_k = 3;
    auto indices = routingTensor({seq_len * top_k}, {0.0f, 1.0f, 2.0f,
                                                     3.0f, 4.0f, 5.0f});
    auto weights = routingTensor({seq_len * top_k}, {0.11f, 0.12f, 0.13f,
                                                     0.21f, 0.22f, 0.23f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), seq_len, top_k,
                            placement({0, 1, 2, 0, 1, 2}),
                            {tier("hottest", "cuda_fast"),
                             tier("warm", "rocm_warm"),
                             tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    ASSERT_EQ(output.tiers.size(), 3u);
    EXPECT_EQ(output.tiers[0].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[1].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[2].token_rows, (std::vector<int>{0, 1}));
    EXPECT_EQ(output.tiers[0].entries.size(), 2u);
    EXPECT_EQ(output.tiers[1].entries.size(), 2u);
    EXPECT_EQ(output.tiers[2].entries.size(), 2u);

    expectContributionExactlyOnce(output, 0, 0, 0, 0, 0.11f);
    expectContributionExactlyOnce(output, 1, 0, 1, 1, 0.12f);
    expectContributionExactlyOnce(output, 2, 0, 2, 2, 0.13f);
    expectContributionExactlyOnce(output, 0, 1, 0, 3, 0.21f);
    expectContributionExactlyOnce(output, 1, 1, 1, 4, 0.22f);
    expectContributionExactlyOnce(output, 2, 1, 2, 5, 0.23f);
}

TEST_F(Test__MoEExpertDispatchStage, InvalidExpertIdFailsClearly)
{
    auto indices = routingTensor({2, 2}, {0.0f, 1.0f,
                                          99.0f, 2.0f});
    auto weights = routingTensor({2, 2}, {0.1f, 0.2f,
                                          0.3f, 0.4f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 2, 2,
                            placement({0, 1, 0}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, MissingPlacementFailsClearly)
{
    auto indices = routingTensor({1, 2}, {0.0f, 1.0f});
    auto weights = routingTensor({1, 2}, {0.5f, 0.5f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 1, 2,
                            std::nullopt,
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, RejectsTransposedMatrixRoutingShape)
{
    auto indices = routingTensor({2, 3}, {0.0f, 1.0f, 2.0f,
                                          0.0f, 1.0f, 2.0f});
    auto weights = routingTensor({2, 3}, {0.1f, 0.2f, 0.3f,
                                          0.4f, 0.5f, 0.6f});
    MoEExpertDispatchOutput output;
    auto params = paramsFor(indices.get(), weights.get(), 3, 2,
                            placement({0, 1, 0}),
                            {tier("hot", "gpu_hot"), tier("cold", "cpu_cold")},
                            &output);

    MoEExpertDispatchStage stage(params);
    EXPECT_FALSE(stage.execute(ctx_.get()));
}

TEST_F(Test__MoEExpertDispatchStage, AdvertisesCpuOnlyBackendSupport)
{
    MoEExpertDispatchOutput output;
    auto params = paramsFor(nullptr, nullptr, 1, 1,
                            placement({0}),
                            {tier("hot", "gpu_hot")},
                            &output);
    MoEExpertDispatchStage stage(params);

    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_FALSE(stage.supportsBackend(ComputeBackendType::GPU_ROCM));
}

TEST_F(Test__MoEExpertDispatchStage, FactoryCreatesDispatchStage)
{
    MoEExpertDispatchOutput output;
    MoEExpertDispatchStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 1;
    params.top_k = 1;
    params.d_model = 8;
    params.placement = placement({0});
    params.routed_tiers = {tier("hot", "gpu_hot")};
    params.output = &output;

    auto stage = ComputeStageFactory::createMoEExpertDispatch(params);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::MOE_EXPERT_DISPATCH);
    EXPECT_STREQ(computeStageTypeName(stage->type()), "MOE_EXPERT_DISPATCH");
}

} // namespace llaminar2::test
