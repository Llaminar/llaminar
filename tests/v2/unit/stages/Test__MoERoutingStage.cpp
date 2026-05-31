/**
 * @file Test__MoERoutingStage.cpp
 * @brief Unit tests for MoERoutingStage (extracted router from MoEExpertComputeStage)
 *
 * Tests that MoERoutingStage correctly:
 * 1. Routes tokens to top-k experts via softmax
 * 2. Outputs float-cast expert indices and normalized weights
 * 3. Reports correct metadata (type, name, flops)
 * 4. Handles edge cases (null inputs, single token)
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/stages/MoERoutingStage.h"
#include "tensors/Tensors.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"
#include "utils/DebugEnv.h"

#include <cmath>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <vector>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

using namespace llaminar2;
using namespace llaminar2::test;
using namespace llaminar2::testing;

namespace
{
#ifdef HAVE_CUDA
    struct ScopedCudaStream
    {
        cudaStream_t stream = nullptr;

        ~ScopedCudaStream()
        {
            if (stream)
                cudaStreamDestroy(stream);
        }
    };
#endif

    class ScopedRocmMoEFlags
    {
    public:
        ScopedRocmMoEFlags(bool grouped_decode, bool device_routed_decode)
            : old_grouped_(mutableDebugEnv().rocm.moe_grouped_decode),
              old_device_routed_(mutableDebugEnv().rocm.moe_device_routed_decode)
        {
            mutableDebugEnv().rocm.moe_grouped_decode = grouped_decode;
            mutableDebugEnv().rocm.moe_device_routed_decode = device_routed_decode;
        }

        ~ScopedRocmMoEFlags()
        {
            mutableDebugEnv().rocm.moe_grouped_decode = old_grouped_;
            mutableDebugEnv().rocm.moe_device_routed_decode = old_device_routed_;
        }

    private:
        bool old_grouped_;
        bool old_device_routed_;
    };

    DeviceNativeVNNIMatrixDesc runtimeDesc(uintptr_t base, int n, int k)
    {
        DeviceNativeVNNIMatrixDesc desc;
        desc.payload = reinterpret_cast<const uint8_t *>(base);
        desc.scales = reinterpret_cast<const void *>(base + 0x1000u);
        desc.mins = reinterpret_cast<const void *>(base + 0x2000u);
        desc.n = n;
        desc.k = k;
        desc.blocks_per_row = static_cast<uint32_t>(k / 32);
        desc.codebook_id = 4;
        return desc;
    }

    MoEPlacementUpdate routingRuntimeUpdate(uint32_t epoch, int num_experts, int d_model)
    {
        MoEPlacementUpdate update;
        update.epoch = epoch;
        update.expert_count = static_cast<uint32_t>(num_experts);
        update.experts.resize(static_cast<size_t>(num_experts));
        update.local_compute_mask.assign(static_cast<size_t>(num_experts), 1u);
        update.replica_role.assign(static_cast<size_t>(num_experts),
                                   static_cast<uint8_t>(DeviceMoEReplicaRole::Primary));

        for (int expert = 0; expert < num_experts; ++expert)
        {
            const uintptr_t base = 0x70000000u + static_cast<uintptr_t>(expert) * 0x10000u;
            auto &desc = update.experts[static_cast<size_t>(expert)];
            desc.gate = runtimeDesc(base + 0x0100u, d_model, d_model);
            desc.up = runtimeDesc(base + 0x0200u, d_model, d_model);
            desc.down = runtimeDesc(base + 0x0300u, d_model, d_model);
            desc.logical_expert_id = expert;
            desc.owner_participant = 0;
            desc.local_slot = expert;
            desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                          DeviceMoEExpertFlags::Resident |
                                          DeviceMoEExpertFlags::LocalCompute);
        }

        return update;
    }
}

// =========================================================================
// Test Fixture
// =========================================================================

class MoERoutingStageTest : public ::testing::Test
{
protected:
    std::unique_ptr<MockDeviceContext> cpu_ctx_;

    static constexpr int D_MODEL = 64;
    static constexpr int NUM_EXPERTS = 4;
    static constexpr int TOP_K = 2;
    static constexpr int SEQ_LEN = 2;

    void SetUp() override
    {
        cpu_ctx_ = std::make_unique<MockDeviceContext>(DeviceId::cpu(), ComputeBackendType::CPU);
    }
};

// =========================================================================
// Routing Tests
// =========================================================================

TEST_F(MoERoutingStageTest, BasicRouting)
{
    auto input = TestTensorFactory::createFP32Random({SEQ_LEN, D_MODEL}, -0.5f, 0.5f, 100);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 101);
    auto output_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.norm_topk_prob = true;
    params.layer_idx = 0;

    MoERoutingStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Verify indices are valid expert IDs
    const float *idx = output_indices->data();
    for (int i = 0; i < SEQ_LEN * TOP_K; ++i)
    {
        int expert_id = static_cast<int>(idx[i]);
        EXPECT_GE(expert_id, 0) << "Expert index " << i << " is negative";
        EXPECT_LT(expert_id, NUM_EXPERTS) << "Expert index " << i << " >= num_experts";
    }

    // Verify weights are positive
    const float *wt = output_weights->data();
    for (int i = 0; i < SEQ_LEN * TOP_K; ++i)
    {
        EXPECT_GT(wt[i], 0.0f) << "Weight " << i << " is not positive";
    }

    // Verify weights sum ~1.0 per token (norm_topk_prob=true)
    for (int t = 0; t < SEQ_LEN; ++t)
    {
        float sum = 0.0f;
        for (int k = 0; k < TOP_K; ++k)
            sum += wt[t * TOP_K + k];
        EXPECT_NEAR(sum, 1.0f, 0.01f) << "Token " << t << " weights don't sum to 1";
    }
}

TEST_F(MoERoutingStageTest, SingleToken)
{
    const int seq = 1;
    auto input = TestTensorFactory::createFP32Random({seq, D_MODEL}, -0.5f, 0.5f, 200);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 201);
    auto output_indices = TestTensorFactory::createFP32({seq * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({seq * TOP_K, 1});

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = seq;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.norm_topk_prob = true;
    params.layer_idx = 0;

    MoERoutingStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Verify output dimensions
    const float *idx = output_indices->data();
    const float *wt = output_weights->data();

    // Check valid indices
    for (int k = 0; k < TOP_K; ++k)
    {
        EXPECT_GE(static_cast<int>(idx[k]), 0);
        EXPECT_LT(static_cast<int>(idx[k]), NUM_EXPERTS);
        EXPECT_GT(wt[k], 0.0f);
    }

    // Weights sum to 1
    float sum = 0.0f;
    for (int k = 0; k < TOP_K; ++k)
        sum += wt[k];
    EXPECT_NEAR(sum, 1.0f, 0.01f);
}

TEST_F(MoERoutingStageTest, NullInputsReturnError)
{
    auto output_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = nullptr;
    params.gate_weights = nullptr;
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;

    MoERoutingStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
}

// =========================================================================
// Stage Metadata Tests
// =========================================================================

TEST_F(MoERoutingStageTest, StageMetadata)
{
    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;

    MoERoutingStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::MOE_ROUTER);
    EXPECT_EQ(stage.name(), "moe_router");
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_GT(stage.estimatedFlops(), 0u);
}

TEST_F(MoERoutingStageTest, GraphCapturableRejectsCPUAndPrefill)
{
    ScopedRocmMoEFlags flags(true, true);

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;

    MoERoutingStage cpu_stage(params);
    EXPECT_FALSE(cpu_stage.isGraphCapturable());

    params.device_id = DeviceId::rocm(0);
    params.seq_len = SEQ_LEN;
    MoERoutingStage prefill_stage(params);
    EXPECT_FALSE(prefill_stage.isGraphCapturable());
}

TEST_F(MoERoutingStageTest, GraphCapturableRejectsHistogramWithoutRuntimeTable)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto gate_weights = TestTensorFactory::createFP32({NUM_EXPERTS, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    DecodeExpertHistogram histogram(histogram_config);

    MoERoutingStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.decode_histogram = &histogram;

    MoERoutingStage stage(params);
    EXPECT_FALSE(stage.isGraphCapturable());
}

TEST_F(MoERoutingStageTest, GraphCapturableAllowsHistogramWithInitializedRuntimeTable)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto gate_weights = TestTensorFactory::createFP32({NUM_EXPERTS, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    DecodeExpertHistogram histogram(histogram_config);

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    MoERoutingStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.layer_idx = 0;
    params.decode_histogram = &histogram;
    params.moe_runtime_table = &runtime_table;

    MoERoutingStage stage(params);
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(stage.isGraphCapturable());
    EXPECT_TRUE(stage.needsOnGraphReplayed());
#else
    EXPECT_FALSE(stage.isGraphCapturable());
    EXPECT_FALSE(stage.needsOnGraphReplayed());
#endif
}

TEST_F(MoERoutingStageTest, GraphCapturableRocmDecodeHonorsReleaseOnlyGuardAndFlags)
{
    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto gate_weights = TestTensorFactory::createFP32({NUM_EXPERTS, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoERoutingStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;

    {
        ScopedRocmMoEFlags flags(false, true);
        MoERoutingStage stage(params);
        EXPECT_FALSE(stage.isGraphCapturable());
    }

    {
        ScopedRocmMoEFlags flags(true, true);
        MoERoutingStage stage(params);
        EXPECT_FALSE(stage.isGraphCapturable());

        MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
        ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
        ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));
        params.layer_idx = 0;
        params.moe_runtime_table = &runtime_table;

        MoERoutingStage runtime_stage(params);
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
        EXPECT_TRUE(runtime_stage.isGraphCapturable());
#else
        EXPECT_FALSE(runtime_stage.isGraphCapturable());
#endif
    }
}

TEST_F(MoERoutingStageTest, GraphCapturableRuntimeHookRequiresInitializedStateWhenProvided)
{
    ScopedRocmMoEFlags flags(true, true);

    auto input = TestTensorFactory::createFP32({1, D_MODEL});
    auto gate_weights = TestTensorFactory::createFP32({NUM_EXPERTS, D_MODEL});
    auto output_indices = TestTensorFactory::createFP32({TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({TOP_K, 1});

    MoERuntimeTable runtime_table(DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);

    MoERoutingStage::Params params;
    params.device_id = DeviceId::rocm(0);
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = 1;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.layer_idx = 0;
    params.moe_runtime_table = &runtime_table;

    MoERoutingStage unprepared_stage(params);
    EXPECT_FALSE(unprepared_stage.isGraphCapturable());

    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    MoERoutingStage prepared_stage(params);
#if defined(HAVE_ROCM) && !defined(ENABLE_PIPELINE_SNAPSHOTS)
    EXPECT_TRUE(prepared_stage.isGraphCapturable());
#else
    EXPECT_FALSE(prepared_stage.isGraphCapturable());
#endif
}

TEST_F(MoERoutingStageTest, OutputDimensions)
{
    const int seq = 3;
    const int experts = 8;
    const int topk = 4;

    auto input = TestTensorFactory::createFP32Random({seq, D_MODEL}, -0.5f, 0.5f, 300);
    auto gate_weights = TestTensorFactory::createFP32Random({experts, D_MODEL}, -0.1f, 0.1f, 301);
    auto output_indices = TestTensorFactory::createFP32({seq * topk, 1});
    auto output_weights = TestTensorFactory::createFP32({seq * topk, 1});

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = seq;
    params.d_model = D_MODEL;
    params.num_experts = experts;
    params.top_k = topk;
    params.norm_topk_prob = true;
    params.layer_idx = 0;

    MoERoutingStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    // Verify all seq*topk entries are populated
    const float *idx = output_indices->data();
    const float *wt = output_weights->data();

    for (int i = 0; i < seq * topk; ++i)
    {
        int expert_id = static_cast<int>(idx[i]);
        EXPECT_GE(expert_id, 0);
        EXPECT_LT(expert_id, experts);
        EXPECT_GT(wt[i], 0.0f);
    }

    // Each token's experts should be distinct
    for (int t = 0; t < seq; ++t)
    {
        std::vector<int> token_experts;
        for (int k = 0; k < topk; ++k)
            token_experts.push_back(static_cast<int>(idx[t * topk + k]));
        std::sort(token_experts.begin(), token_experts.end());
        auto last = std::unique(token_experts.begin(), token_experts.end());
        EXPECT_EQ(std::distance(token_experts.begin(), last), topk)
            << "Token " << t << " has duplicate expert assignments";
    }
}

TEST_F(MoERoutingStageTest, CUDAFallbackRoutingOutputsRemainDeviceCoherent)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "Built without CUDA support";
#else
    int device_count = 0;
    cudaError_t count_err = cudaGetDeviceCount(&device_count);
    if (count_err != cudaSuccess || device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";

    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    DeviceId cuda_device = DeviceId::cuda(0);
    MockDeviceContext cuda_ctx(cuda_device, ComputeBackendType::GPU_CUDA);
    ScopedCudaStream stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream.stream, cudaStreamNonBlocking), cudaSuccess);

    auto input = TestTensorFactory::createFP32Random({SEQ_LEN, D_MODEL}, -0.5f, 0.5f, 400);
    auto gate_weights = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 401);
    auto output_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    ASSERT_TRUE(input->ensureOnDevice(cuda_device));
    ASSERT_TRUE(gate_weights->ensureOnDevice(cuda_device));
    ASSERT_TRUE(output_indices->ensureOnDevice(cuda_device));
    ASSERT_TRUE(output_weights->ensureOnDevice(cuda_device));

    MoERoutingStage::Params params;
    params.device_id = cuda_device;
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.norm_topk_prob = true;
    params.layer_idx = 0;

    MoERoutingStage stage(params);
    stage.setGPUStream(stream.stream);
    ASSERT_TRUE(stage.execute(&cuda_ctx));
    ASSERT_TRUE(output_indices->is_on_device(cuda_device));
    ASSERT_TRUE(output_weights->is_on_device(cuda_device));

    ASSERT_TRUE(output_indices->ensureOnHost());
    ASSERT_TRUE(output_weights->ensureOnHost());

    const float *idx = output_indices->data();
    const float *wt = output_weights->data();

    bool any_non_zero_index = false;
    for (int i = 0; i < SEQ_LEN * TOP_K; ++i)
    {
        int expert_id = static_cast<int>(idx[i]);
        EXPECT_GE(expert_id, 0);
        EXPECT_LT(expert_id, NUM_EXPERTS);
        any_non_zero_index = any_non_zero_index || expert_id != 0;
        EXPECT_GT(wt[i], 0.0f) << "Weight " << i << " was not uploaded to CUDA output storage";
    }
    EXPECT_TRUE(any_non_zero_index) << "Routing indices read back from CUDA remained all zero";
#endif
}
