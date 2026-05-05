/**
 * @file Test__MoERoutingToCompute_Integration.cpp
 * @brief End-to-end integration test for MoERoutingStage → MoEExpertComputeStage pipeline
 *
 * Exercises the full two-stage MoE pipeline:
 * 1. MoERoutingStage produces routing_indices and routing_weights
 * 2. MoEExpertComputeStage consumes routing results + expert weights → combined output
 *
 * Tests prefill (multi-token), decode (single-token), expert-parallel partial output,
 * and normalized weight flow.
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "tensors/Tensors.h"
#include "tensors/BlockStructures.h"
#include "tensors/FP16Utils.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"

#include <cmath>
#include <numeric>
#include <algorithm>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

// =========================================================================
// Test Fixture
// =========================================================================

class MoERoutingToComputeTest : public ::testing::Test
{
protected:
    std::unique_ptr<MockDeviceContext> cpu_ctx_;

    // Dimensions must be multiples of 256 for Q4_K block size
    static constexpr int D_MODEL = 256;
    static constexpr int INTERMEDIATE = 256;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;

    void SetUp() override
    {
        cpu_ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }

    /// Create a 3D Q4_K expert tensor in GGUF layout [cols, rows, num_experts]
    std::shared_ptr<Q4_KTensor> createExpertQ4K(int num_experts, int rows, int cols, uint32_t seed = 42)
    {
        std::vector<size_t> shape = {static_cast<size_t>(cols),
                                     static_cast<size_t>(rows),
                                     static_cast<size_t>(num_experts)};
        size_t blocks_per_row = cols / Q4_KBlock::BLOCK_SIZE;
        size_t total_blocks = num_experts * rows * blocks_per_row;
        std::vector<uint8_t> raw(total_blocks * sizeof(Q4_KBlock));

        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        auto *blocks = reinterpret_cast<Q4_KBlock *>(raw.data());
        for (size_t i = 0; i < total_blocks; ++i)
        {
            float d_val = dist(rng) * 0.01f;
            float dmin_val = std::abs(dist(rng)) * 0.001f;
            blocks[i].d = fp32_to_fp16(d_val);
            blocks[i].dmin = fp32_to_fp16(dmin_val);
            for (size_t j = 0; j < sizeof(blocks[i].qs); ++j)
                blocks[i].qs[j] = static_cast<uint8_t>(rng());
            for (size_t j = 0; j < sizeof(blocks[i].scales); ++j)
                blocks[i].scales[j] = static_cast<uint8_t>(rng());
        }

        return std::make_shared<Q4_KTensor>(shape, raw);
    }

    /// Run the MoERoutingStage and return true on success
    bool runRouting(TensorBase *input, TensorBase *gate_weights,
                    TensorBase *output_indices, TensorBase *output_weights,
                    int seq_len, int d_model, int num_experts, int top_k,
                    bool norm_topk_prob = true)
    {
        MoERoutingStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input;
        params.gate_weights = gate_weights;
        params.output_indices = output_indices;
        params.output_weights = output_weights;
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.norm_topk_prob = norm_topk_prob;
        params.layer_idx = 0;

        MoERoutingStage stage(params);
        return stage.execute(cpu_ctx_.get());
    }

    /// Run MoEExpertComputeStage with given routing results and expert weights
    bool runExpertCompute(TensorBase *input, TensorBase *routing_indices,
                          TensorBase *routing_weights,
                          TensorBase *gate_exps, TensorBase *up_exps,
                          TensorBase *down_exps, TensorBase *output,
                          int seq_len, int d_model, int num_experts, int top_k,
                          int intermediate, int local_expert_start = 0,
                          int local_expert_count = -1)
    {
        MoEExpertComputeStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input;
        params.routing_indices = routing_indices;
        params.routing_weights = routing_weights;
        params.gate_exps = gate_exps;
        params.up_exps = up_exps;
        params.down_exps = down_exps;
        params.output = output;
        params.seq_len = seq_len;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.expert_intermediate = intermediate;
        params.local_expert_start = local_expert_start;
        params.local_expert_count = local_expert_count;

        if (!MoEExpertComputeStage::extractExpertViews(params))
            return false;
        if (!MoEExpertComputeStage::prepareExpertGemmEngines(params))
            return false;

        MoEExpertComputeStage stage(params);
        return stage.execute(cpu_ctx_.get());
    }

    /// Check that a float buffer has no NaN/Inf and at least one non-zero value
    void verifyNonZeroNoNaN(const float *data, int count, const std::string &label)
    {
        bool any_nonzero = false;
        for (int i = 0; i < count; ++i)
        {
            EXPECT_FALSE(std::isnan(data[i])) << label << ": NaN at index " << i;
            EXPECT_FALSE(std::isinf(data[i])) << label << ": Inf at index " << i;
            if (data[i] != 0.0f)
                any_nonzero = true;
        }
        EXPECT_TRUE(any_nonzero) << label << ": output is all zeros";
    }

    /// Check no NaN/Inf (but allow all-zeros for EP partial output)
    void verifyNoNaN(const float *data, int count, const std::string &label)
    {
        for (int i = 0; i < count; ++i)
        {
            EXPECT_FALSE(std::isnan(data[i])) << label << ": NaN at index " << i;
            EXPECT_FALSE(std::isinf(data[i])) << label << ": Inf at index " << i;
        }
    }
};

// =========================================================================
// Pipeline Tests
// =========================================================================

TEST_F(MoERoutingToComputeTest, PrefillPipeline)
{
    const int seq_len = 4;

    // Stage 1 inputs: hidden states + gate weights
    auto input = TestTensorFactory::createFP32Random({seq_len, D_MODEL}, -0.5f, 0.5f, 1000);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 1001);

    // Stage 1 outputs / Stage 2 inputs: routing results
    auto routing_indices = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({seq_len * TOP_K, 1});

    // Stage 2 inputs: expert weights (Q4_K 3D packed)
    auto gate_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 1010);
    auto up_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 1011);
    auto down_exps = createExpertQ4K(NUM_EXPERTS, D_MODEL, INTERMEDIATE, 1012);

    // Stage 2 output
    auto output = TestTensorFactory::createFP32({seq_len, D_MODEL});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // === Run Stage 1: MoERoutingStage ===
    ASSERT_TRUE(runRouting(input.get(), gate_weights.get(),
                           routing_indices.get(), routing_weights.get(),
                           seq_len, D_MODEL, NUM_EXPERTS, TOP_K));

    // Verify routing indices are valid expert IDs
    const float *idx = routing_indices->data();
    for (int i = 0; i < seq_len * TOP_K; ++i)
    {
        int expert_id = static_cast<int>(idx[i]);
        EXPECT_GE(expert_id, 0) << "Expert index " << i << " is negative";
        EXPECT_LT(expert_id, NUM_EXPERTS) << "Expert index " << i << " >= num_experts";
    }

    // === Run Stage 2: MoEExpertComputeStage ===
    ASSERT_TRUE(runExpertCompute(input.get(), routing_indices.get(), routing_weights.get(),
                                 gate_exps.get(), up_exps.get(), down_exps.get(),
                                 output.get(), seq_len, D_MODEL, NUM_EXPERTS, TOP_K, INTERMEDIATE));

    // Verify each token has non-zero output
    const float *out = output->data();
    for (int t = 0; t < seq_len; ++t)
    {
        bool token_nonzero = false;
        for (int d = 0; d < D_MODEL; ++d)
        {
            float v = out[t * D_MODEL + d];
            EXPECT_FALSE(std::isnan(v)) << "NaN at token " << t << " dim " << d;
            EXPECT_FALSE(std::isinf(v)) << "Inf at token " << t << " dim " << d;
            if (v != 0.0f)
                token_nonzero = true;
        }
        EXPECT_TRUE(token_nonzero) << "Token " << t << " output is all zeros";
    }
}

TEST_F(MoERoutingToComputeTest, DecodePipeline)
{
    const int seq_len = 1; // Single token decode

    auto input = TestTensorFactory::createFP32Random({seq_len, D_MODEL}, -0.5f, 0.5f, 2000);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 2001);
    auto routing_indices = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto gate_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 2010);
    auto up_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 2011);
    auto down_exps = createExpertQ4K(NUM_EXPERTS, D_MODEL, INTERMEDIATE, 2012);
    auto output = TestTensorFactory::createFP32({seq_len, D_MODEL});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // === Stage 1: Routing ===
    ASSERT_TRUE(runRouting(input.get(), gate_weights.get(),
                           routing_indices.get(), routing_weights.get(),
                           seq_len, D_MODEL, NUM_EXPERTS, TOP_K));

    // Verify routing
    const float *idx = routing_indices->data();
    for (int k = 0; k < TOP_K; ++k)
    {
        int expert_id = static_cast<int>(idx[k]);
        EXPECT_GE(expert_id, 0);
        EXPECT_LT(expert_id, NUM_EXPERTS);
    }

    // === Stage 2: Expert compute ===
    ASSERT_TRUE(runExpertCompute(input.get(), routing_indices.get(), routing_weights.get(),
                                 gate_exps.get(), up_exps.get(), down_exps.get(),
                                 output.get(), seq_len, D_MODEL, NUM_EXPERTS, TOP_K, INTERMEDIATE));

    verifyNonZeroNoNaN(output->data(), seq_len * D_MODEL, "DecodePipeline output");
}

TEST_F(MoERoutingToComputeTest, EPPartialOutput)
{
    const int seq_len = 4;
    const int local_expert_start = 2;
    const int local_expert_count = 2; // Only experts 2,3 are local

    auto input = TestTensorFactory::createFP32Random({seq_len, D_MODEL}, -0.5f, 0.5f, 3000);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 3001);
    auto routing_indices = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto gate_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 3010);
    auto up_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 3011);
    auto down_exps = createExpertQ4K(NUM_EXPERTS, D_MODEL, INTERMEDIATE, 3012);
    auto output = TestTensorFactory::createFP32({seq_len, D_MODEL});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    // === Stage 1: Routing (full routing, all experts visible) ===
    ASSERT_TRUE(runRouting(input.get(), gate_weights.get(),
                           routing_indices.get(), routing_weights.get(),
                           seq_len, D_MODEL, NUM_EXPERTS, TOP_K));

    // === Stage 2: Expert compute with EP (only experts 2,3) ===
    ASSERT_TRUE(runExpertCompute(input.get(), routing_indices.get(), routing_weights.get(),
                                 gate_exps.get(), up_exps.get(), down_exps.get(),
                                 output.get(), seq_len, D_MODEL, NUM_EXPERTS, TOP_K, INTERMEDIATE,
                                 local_expert_start, local_expert_count));

    // Output may be partially zero (tokens routed only to experts 0,1 get zero output).
    // But it must not contain NaN/Inf.
    verifyNoNaN(output->data(), seq_len * D_MODEL, "EPPartialOutput");

    // Verify allowsZeroOutput() returns true for EP configuration
    MoEExpertComputeStage::Params ep_params;
    ep_params.device_id = DeviceId::cpu();
    ep_params.local_expert_count = local_expert_count;
    ep_params.num_experts = NUM_EXPERTS;
    ep_params.top_k = TOP_K;
    ep_params.expert_intermediate = INTERMEDIATE;
    ep_params.seq_len = seq_len;
    ep_params.d_model = D_MODEL;
    MoEExpertComputeStage ep_stage(ep_params);
    EXPECT_TRUE(ep_stage.allowsZeroOutput());
}

TEST_F(MoERoutingToComputeTest, RoutingWeightsSumToOne)
{
    const int seq_len = 4;

    auto input = TestTensorFactory::createFP32Random({seq_len, D_MODEL}, -0.5f, 0.5f, 4000);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 4001);
    auto routing_indices = TestTensorFactory::createFP32({seq_len * TOP_K, 1});
    auto routing_weights = TestTensorFactory::createFP32({seq_len * TOP_K, 1});

    // Run routing with norm_topk_prob=true
    ASSERT_TRUE(runRouting(input.get(), gate_weights.get(),
                           routing_indices.get(), routing_weights.get(),
                           seq_len, D_MODEL, NUM_EXPERTS, TOP_K,
                           /*norm_topk_prob=*/true));

    // Verify weights sum to ~1.0 per token
    const float *wt = routing_weights->data();
    for (int t = 0; t < seq_len; ++t)
    {
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k)
        {
            float w = wt[t * TOP_K + k];
            EXPECT_GT(w, 0.0f) << "Token " << t << " weight " << k << " is not positive";
            sum += w;
        }
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Token " << t << " weights don't sum to 1.0";
    }

    // Now feed these normalized weights through the compute stage and verify non-zero output
    auto gate_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 4010);
    auto up_exps = createExpertQ4K(NUM_EXPERTS, INTERMEDIATE, D_MODEL, 4011);
    auto down_exps = createExpertQ4K(NUM_EXPERTS, D_MODEL, INTERMEDIATE, 4012);
    auto output = TestTensorFactory::createFP32({seq_len, D_MODEL});
    std::memset(output->mutable_data(), 0, output->numel() * sizeof(float));

    ASSERT_TRUE(runExpertCompute(input.get(), routing_indices.get(), routing_weights.get(),
                                 gate_exps.get(), up_exps.get(), down_exps.get(),
                                 output.get(), seq_len, D_MODEL, NUM_EXPERTS, TOP_K, INTERMEDIATE));

    verifyNonZeroNoNaN(output->data(), seq_len * D_MODEL, "RoutingWeightsSumToOne output");
}
