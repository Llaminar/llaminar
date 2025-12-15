/**
 * @file Test__ComputeStage.cpp
 * @brief Unit tests for ComputeStage abstraction
 * @author David Sanftenberg
 * @date December 2025
 */

#include <gtest/gtest.h>
#include "execution/ComputeStage.h"
#include "execution/DeviceContext.h"
#include <cmath>
#include <vector>
#include <numeric>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class ComputeStageTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Construct CPUDeviceContext directly (bypasses DeviceManager check)
        // This is the same pattern used in Test__DeviceContext.cpp
        ctx_ = std::make_unique<CPUDeviceContext>(0, 4);
        ASSERT_NE(ctx_, nullptr);
    }

    std::unique_ptr<CPUDeviceContext> ctx_;
};

// =============================================================================
// ComputeStageType Tests
// =============================================================================

TEST_F(ComputeStageTest, StageTypeNames)
{
    // Test that all stage types have human-readable names
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GEMM), "GEMM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::RMS_NORM), "RMS_NORM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ROPE), "ROPE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ATTENTION), "ATTENTION");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::SWIGLU), "SWIGLU");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ADD_RESIDUAL), "ADD_RESIDUAL");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_ROUTER), "MOE_ROUTER");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_EXPERT_FFN), "MOE_EXPERT_FFN");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_COMBINE), "MOE_COMBINE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ALLREDUCE), "ALLREDUCE");
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(ComputeStageTest, RMSNormBasic)
{
    // Test basic RMS normalization
    const int seq_len = 2;
    const int hidden_dim = 4;
    const float eps = 1e-6f;

    // Input: two rows of [1, 2, 3, 4]
    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f,
                                1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma(hidden_dim, 1.0f); // Scale = 1

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = eps;

    RMSNormStage stage(params);

    // Check type
    EXPECT_EQ(stage.type(), ComputeStageType::RMS_NORM);

    // Execute
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // Verify: RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
    // Each element should be divided by RMS
    float expected_rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    EXPECT_NEAR(input[0], 1.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(input[1], 2.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(input[2], 3.0f / expected_rms, 1e-5f);
    EXPECT_NEAR(input[3], 4.0f / expected_rms, 1e-5f);
}

TEST_F(ComputeStageTest, RMSNormWithGamma)
{
    // Test RMS normalization with non-unit gamma
    const int seq_len = 1;
    const int hidden_dim = 4;
    const float eps = 1e-6f;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> gamma = {2.0f, 0.5f, 1.0f, 0.0f}; // Different scales

    RMSNormStage::Params params;
    params.input = input.data();
    params.gamma = gamma.data();
    params.seq_len = seq_len;
    params.hidden_dim = hidden_dim;
    params.eps = eps;

    RMSNormStage stage(params);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    float rms = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    EXPECT_NEAR(input[0], (1.0f / rms) * 2.0f, 1e-5f); // gamma = 2
    EXPECT_NEAR(input[1], (2.0f / rms) * 0.5f, 1e-5f); // gamma = 0.5
    EXPECT_NEAR(input[3], 0.0f, 1e-5f);                // gamma = 0
}

TEST_F(ComputeStageTest, RMSNormFlopEstimate)
{
    RMSNormStage::Params params;
    params.seq_len = 512;
    params.hidden_dim = 896;

    RMSNormStage stage(params);

    // ~4 FLOPs per element: square, add, normalize (mul + div)
    size_t expected_flops = 4ULL * 512 * 896;
    EXPECT_EQ(stage.estimatedFlops(), expected_flops);
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(ComputeStageTest, RoPEBasic)
{
    // Test basic rotary position encoding
    const int seq_len = 1;
    const int n_heads = 1;
    const int head_dim = 4; // Must be even for RoPE

    // Input: [1, 0, 1, 0] (pairs: (1,0), (1,0))
    std::vector<float> tensor = {1.0f, 0.0f, 1.0f, 0.0f};

    RoPEStage::Params params;
    params.tensor = tensor.data();
    params.seq_len = seq_len;
    params.n_heads = n_heads;
    params.head_dim = head_dim;
    params.pos_offset = 0;
    params.theta_base = 10000.0f;

    RoPEStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ROPE);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // At position 0, cos(0) = 1, sin(0) = 0
    // So rotation should be identity
    EXPECT_NEAR(tensor[0], 1.0f, 1e-5f);
    EXPECT_NEAR(tensor[1], 0.0f, 1e-5f);
}

TEST_F(ComputeStageTest, RoPEPositionOffset)
{
    // Test that position offset affects rotation
    const int seq_len = 1;
    const int n_heads = 1;
    const int head_dim = 4;

    std::vector<float> tensor_pos0 = {1.0f, 0.0f, 1.0f, 0.0f};
    std::vector<float> tensor_pos1 = {1.0f, 0.0f, 1.0f, 0.0f};

    RoPEStage::Params params0, params1;
    params0.tensor = tensor_pos0.data();
    params0.seq_len = seq_len;
    params0.n_heads = n_heads;
    params0.head_dim = head_dim;
    params0.pos_offset = 0;
    params0.theta_base = 10000.0f;

    params1 = params0;
    params1.tensor = tensor_pos1.data();
    params1.pos_offset = 10; // Different position

    RoPEStage stage0(params0);
    RoPEStage stage1(params1);

    EXPECT_TRUE(stage0.execute(ctx_.get()));
    EXPECT_TRUE(stage1.execute(ctx_.get()));

    // Results should be different due to position offset
    bool different = false;
    for (int i = 0; i < head_dim; ++i)
    {
        if (std::abs(tensor_pos0[i] - tensor_pos1[i]) > 1e-5f)
        {
            different = true;
            break;
        }
    }
    EXPECT_TRUE(different);
}

// =============================================================================
// SwiGLUStage Tests
// =============================================================================

TEST_F(ComputeStageTest, SwiGLUBasic)
{
    const int seq_len = 2;
    const int intermediate_dim = 4;

    // SwiGLU: silu(gate) * up
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    std::vector<float> gate = {0.0f, 1.0f, -1.0f, 2.0f,
                               0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> up = {1.0f, 1.0f, 1.0f, 1.0f,
                             2.0f, 2.0f, 2.0f, 2.0f};
    std::vector<float> output(seq_len * intermediate_dim, 0.0f);

    SwiGLUStage::Params params;
    params.gate = gate.data();
    params.up = up.data();
    params.output = output.data();
    params.seq_len = seq_len;
    params.intermediate_dim = intermediate_dim;

    SwiGLUStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::SWIGLU);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // silu(0) = 0 / (1 + 1) = 0
    EXPECT_NEAR(output[0], 0.0f, 1e-5f);

    // silu(1) ≈ 0.731
    float silu_1 = 1.0f / (1.0f + std::exp(-1.0f));
    EXPECT_NEAR(output[1], silu_1 * 1.0f, 1e-5f);

    // silu(-1) ≈ -0.269
    float silu_neg1 = -1.0f / (1.0f + std::exp(1.0f));
    EXPECT_NEAR(output[2], silu_neg1 * 1.0f, 1e-5f);
}

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(ComputeStageTest, ResidualAddBasic)
{
    const size_t n = 8;

    std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> output(n, 0.0f);

    ResidualAddStage::Params params;
    params.input = input.data();
    params.residual = residual.data();
    params.output = output.data();
    params.num_elements = n;

    ResidualAddStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(output[i], input[i] + residual[i], 1e-5f);
    }
}

TEST_F(ComputeStageTest, ResidualAddInPlace)
{
    // Test in-place addition (output = input)
    const size_t n = 4;

    std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> residual = {0.5f, 0.5f, 0.5f, 0.5f};

    ResidualAddStage::Params params;
    params.input = buffer.data();
    params.residual = residual.data();
    params.output = buffer.data(); // In-place
    params.num_elements = n;

    ResidualAddStage stage(params);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    EXPECT_NEAR(buffer[0], 1.5f, 1e-5f);
    EXPECT_NEAR(buffer[1], 2.5f, 1e-5f);
    EXPECT_NEAR(buffer[2], 3.5f, 1e-5f);
    EXPECT_NEAR(buffer[3], 4.5f, 1e-5f);
}

// =============================================================================
// Backend Support Tests
// =============================================================================

TEST_F(ComputeStageTest, BackendSupport)
{
    // All CPU stages should support OpenBLAS and MKL
    RMSNormStage::Params rms_params;
    rms_params.seq_len = 1;
    rms_params.hidden_dim = 4;

    RMSNormStage rms_stage(rms_params);

    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::CPU_OPENBLAS));
    EXPECT_TRUE(rms_stage.supportsBackend(ComputeBackendType::CPU_MKL));
    EXPECT_FALSE(rms_stage.supportsBackend(ComputeBackendType::GPU_CUDA));
    EXPECT_FALSE(rms_stage.supportsBackend(ComputeBackendType::GPU_ROCM));
}

// =============================================================================
// MoE Stage Tests
// =============================================================================

TEST_F(ComputeStageTest, MoERouterBasic)
{
    const int seq_len = 2;
    const int d_model = 4;
    const int num_experts = 3;

    // Simple hidden states
    std::vector<float> hidden = {1.0f, 0.0f, 0.0f, 0.0f,  // Token 0
                                 0.0f, 1.0f, 0.0f, 0.0f}; // Token 1

    // Gate weights: each expert is a one-hot
    std::vector<float> gate_weights = {
        1.0f, 0.0f, 0.0f, 0.0f, // Expert 0 responds to dim 0
        0.0f, 1.0f, 0.0f, 0.0f, // Expert 1 responds to dim 1
        0.0f, 0.0f, 1.0f, 0.0f  // Expert 2 responds to dim 2
    };

    std::vector<float> router_logits(seq_len * num_experts, 0.0f);

    MoERouterStage::Params params;
    params.hidden = hidden.data();
    params.gate_weights = gate_weights.data();
    params.router_logits = router_logits.data();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.num_experts = num_experts;

    MoERouterStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_ROUTER);
    EXPECT_TRUE(stage.execute(ctx_.get()));

    // Token 0: [1,0,0,0] should have logit 1 for expert 0
    EXPECT_NEAR(router_logits[0], 1.0f, 1e-5f); // Expert 0
    EXPECT_NEAR(router_logits[1], 0.0f, 1e-5f); // Expert 1
    EXPECT_NEAR(router_logits[2], 0.0f, 1e-5f); // Expert 2

    // Token 1: [0,1,0,0] should have logit 1 for expert 1
    EXPECT_NEAR(router_logits[3], 0.0f, 1e-5f); // Expert 0
    EXPECT_NEAR(router_logits[4], 1.0f, 1e-5f); // Expert 1
    EXPECT_NEAR(router_logits[5], 0.0f, 1e-5f); // Expert 2
}

TEST_F(ComputeStageTest, MoEExpertNoTokens)
{
    // Expert with no routed tokens should be a no-op
    std::vector<int> empty_tokens;

    MoEExpertStage::Params params;
    params.expert_id = 5;
    params.input = nullptr;
    params.output = nullptr;
    params.gate_w = nullptr;
    params.up_w = nullptr;
    params.down_w = nullptr;
    params.token_indices = &empty_tokens;
    params.d_model = 896;
    params.intermediate_dim = 4864;

    MoEExpertStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::MOE_EXPERT_FFN);
    EXPECT_EQ(stage.name(), "MOE_EXPERT_5");
    EXPECT_TRUE(stage.execute(ctx_.get()));  // Should succeed (no-op)
    EXPECT_EQ(stage.estimatedFlops(), 0ULL); // No tokens = no work
}

// =============================================================================
// ComputeStageFactory Tests
// =============================================================================

TEST_F(ComputeStageTest, FactoryCreateRMSNorm)
{
    RMSNormStage::Params params;
    params.input = nullptr;
    params.gamma = nullptr;
    params.seq_len = 1;
    params.hidden_dim = 896;
    params.eps = 1e-6f;

    auto stage = ComputeStageFactory::createRMSNorm(params, ComputeBackendType::CPU_OPENBLAS);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::RMS_NORM);
}

TEST_F(ComputeStageTest, FactoryCreateRoPE)
{
    RoPEStage::Params params;
    params.tensor = nullptr;
    params.seq_len = 1;
    params.n_heads = 14;
    params.head_dim = 64;
    params.pos_offset = 0;
    params.theta_base = 10000.0f;

    auto stage = ComputeStageFactory::createRoPE(params, ComputeBackendType::CPU_MKL);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::ROPE);
}

TEST_F(ComputeStageTest, FactoryCreateSwiGLU)
{
    SwiGLUStage::Params params;
    params.gate = nullptr;
    params.up = nullptr;
    params.output = nullptr;
    params.seq_len = 1;
    params.intermediate_dim = 4864;

    auto stage = ComputeStageFactory::createSwiGLU(params, ComputeBackendType::CPU_OPENBLAS);
    ASSERT_NE(stage, nullptr);
    EXPECT_EQ(stage->type(), ComputeStageType::SWIGLU);
}

// =============================================================================
// Null Context Handling Tests
// =============================================================================

TEST_F(ComputeStageTest, NullContextHandling)
{
    // All stages should gracefully handle null context
    RMSNormStage::Params rms_params;
    rms_params.seq_len = 1;
    rms_params.hidden_dim = 4;

    RMSNormStage rms_stage(rms_params);
    EXPECT_FALSE(rms_stage.execute(nullptr));

    RoPEStage::Params rope_params;
    rope_params.seq_len = 1;
    rope_params.n_heads = 1;
    rope_params.head_dim = 4;

    RoPEStage rope_stage(rope_params);
    EXPECT_FALSE(rope_stage.execute(nullptr));

    SwiGLUStage::Params swiglu_params;
    swiglu_params.seq_len = 1;
    swiglu_params.intermediate_dim = 4;

    SwiGLUStage swiglu_stage(swiglu_params);
    EXPECT_FALSE(swiglu_stage.execute(nullptr));
}
