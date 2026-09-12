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
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/moe/MoEWorkspaceRequirements.h"
#include "tensors/Tensors.h"
#include "mocks/MockComputeStage.h"
#include "utils/TestTensorFactory.h"
#include "../../utils/ObservedExpertDemandFixture.h"
#include "utils/DebugEnv.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

/**
 * @brief Prove that CPU prefill contributes only its logical routed rows.
 *
 * The router computes four physical rows while the evidence contract exposes
 * only the leading two as request-owned.  Expected counts are derived from the
 * production routing tensors so this test checks publication boundaries rather
 * than duplicating the router's numerical implementation.  The total count is
 * especially important: even when a suffix chooses the same experts as the
 * prefix, admitting it would double the observed activation total.
 */
TEST_F(MoERoutingStageTest, CPUGroupedRoutingEvidenceExcludesPhysicalSuffix)
{
    constexpr int kPhysicalRows = 4;
    constexpr int kLogicalRows = 2;

    auto input = TestTensorFactory::createFP32Random(
        {kPhysicalRows, D_MODEL}, -0.5f, 0.5f, 150);
    auto gate_weights = TestTensorFactory::createFP32Random(
        {NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 151);
    auto output_indices = TestTensorFactory::createFP32(
        {kPhysicalRows * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32(
        {kPhysicalRows * TOP_K, 1});

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    histogram_config.window_size = 32;
    histogram_config.token_boundary_layer_idx = 0;
    histogram_config.sockets = {DeviceId::cpu()};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        1, 1, std::vector<int>(NUM_EXPERTS, 0));
    DecodeExpertHistogram histogram(histogram_config);

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = output_indices.get();
    params.output_weights = output_weights.get();
    params.seq_len = kPhysicalRows;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.norm_topk_prob = true;
    params.layer_idx = 0;
    params.decode_histogram = &histogram;
    params.host_logical_row_count = kLogicalRows;
    params.host_routing_source = ExpertHistogramSource::PrefillChunk;

    MoERoutingStage stage(params);
    ASSERT_TRUE(stage.execute(cpu_ctx_.get()));

    std::vector<uint64_t> expected(static_cast<size_t>(NUM_EXPERTS), 0);
    const float *indices = output_indices->data();
    for (int route = 0; route < kLogicalRows * TOP_K; ++route)
    {
        const int expert = static_cast<int>(indices[route]);
        ASSERT_GE(expert, 0);
        ASSERT_LT(expert, NUM_EXPERTS);
        ++expected[static_cast<size_t>(expert)];
    }

    uint64_t published_total = 0;
    for (int expert = 0; expert < NUM_EXPERTS; ++expert)
    {
        const uint64_t actual = histogram.activationCount(0, expert);
        EXPECT_EQ(actual, expected[static_cast<size_t>(expert)]);
        published_total += actual;
    }
    EXPECT_EQ(published_total, static_cast<uint64_t>(kLogicalRows * TOP_K));
    EXPECT_EQ(histogram.windowTokenCount(), static_cast<uint64_t>(kLogicalRows));
}

/**
 * @brief A tracked multi-row CPU graph must declare its logical row prefix.
 *
 * Silently treating physical capacity as request geometry would make padding
 * part of Dynamic placement evidence.  The stage therefore rejects an omitted
 * contract instead of guessing that every row is live.
 */
TEST_F(MoERoutingStageTest, CPUGroupedRoutingEvidenceRequiresLogicalGeometry)
{
    auto input = TestTensorFactory::createFP32Random(
        {SEQ_LEN, D_MODEL}, -0.5f, 0.5f, 160);
    auto gate_weights = TestTensorFactory::createFP32Random(
        {NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 161);
    auto output_indices = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});
    auto output_weights = TestTensorFactory::createFP32({SEQ_LEN * TOP_K, 1});

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = NUM_EXPERTS;
    histogram_config.top_k = TOP_K;
    histogram_config.window_size = 32;
    histogram_config.token_boundary_layer_idx = 0;
    histogram_config.sockets = {DeviceId::cpu()};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        1, 1, std::vector<int>(NUM_EXPERTS, 0));
    DecodeExpertHistogram histogram(histogram_config);

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
    params.decode_histogram = &histogram;

    params.host_routing_source = ExpertHistogramSource::PrefillChunk;
    MoERoutingStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
    EXPECT_EQ(histogram.windowTokenCount(), 0u);
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

/**
 * @brief Prove that routing workspace follows the graph-family row envelope.
 *
 * The live server regression first built a 39-row Qwen3.6 MoE graph and then
 * admitted a 68-row multi-turn prompt.  The old router ignored the allocator's
 * family-wide `m` argument, leaving `moe_route_logits` permanently sized for
 * the first request.  Exercise both GPU backends without launching device work
 * and verify that a larger planner envelope, as well as a larger concrete
 * stage, can never be weakened by the other value.
 */
TEST_F(MoERoutingStageTest, GPUWorkspaceRowsCoverConcreteAndFamilyEnvelope)
{
    constexpr int kConcreteRows = 39;
    constexpr int kLaterRequestRows = 68;
    constexpr int kFamilyRows = 4096;
    constexpr int kModelWidth = 2048;
    constexpr int kExperts = 256;

    const auto expectedRouteLogitsBytes = [](int rows)
    {
        return static_cast<size_t>(rows) *
               static_cast<size_t>(kExperts) * sizeof(float);
    };

    const auto verifyBackend = [&](DeviceId device)
    {
        MoERoutingStage::Params params;
        params.device_id = device;
        params.seq_len = kConcreteRows;
        params.d_model = kModelWidth;
        params.num_experts = kExperts;
        params.top_k = 8;

        MoERoutingStage stage(params);

        const WorkspaceRequirements concrete =
            stage.getWorkspaceRequirements(/*m=*/1);
        const WorkspaceDescriptor *concrete_logits =
            concrete.find(MoEWorkspaceBuffers::ROUTE_LOGITS);
        ASSERT_NE(concrete_logits, nullptr);
        EXPECT_EQ(
            concrete_logits->size_bytes,
            expectedRouteLogitsBytes(kConcreteRows));

        const WorkspaceRequirements later_request =
            stage.getWorkspaceRequirements(kLaterRequestRows);
        const WorkspaceDescriptor *later_logits =
            later_request.find(MoEWorkspaceBuffers::ROUTE_LOGITS);
        ASSERT_NE(later_logits, nullptr);
        EXPECT_EQ(
            later_logits->size_bytes,
            expectedRouteLogitsBytes(kLaterRequestRows));

        const WorkspaceRequirements family =
            stage.getWorkspaceRequirements(kFamilyRows);
        const WorkspaceDescriptor *family_logits =
            family.find(MoEWorkspaceBuffers::ROUTE_LOGITS);
        ASSERT_NE(family_logits, nullptr);
        EXPECT_EQ(
            family_logits->size_bytes,
            expectedRouteLogitsBytes(kFamilyRows));
        EXPECT_GT(family_logits->size_bytes, later_logits->size_bytes);
    };

    verifyBackend(DeviceId::cuda(0));
    verifyBackend(DeviceId::rocm(0));
}

TEST_F(MoERoutingStageTest, CPUVerifierTwoRowsMatchSplitDecodeRoutes)
{
    const int seq = 2;
    const int d_model = 512;
    const int num_experts = 16;
    const int top_k = 4;
    auto input = TestTensorFactory::createFP32Random({seq, d_model}, -0.5f, 0.5f, 1200);
    auto gate_weights = TestTensorFactory::createFP32Random({num_experts, d_model}, -0.1f, 0.1f, 1201);
    auto multi_indices = TestTensorFactory::createFP32({seq, top_k});
    auto multi_weights = TestTensorFactory::createFP32({seq, top_k});
    auto split_indices = TestTensorFactory::createFP32({seq, top_k});
    auto split_weights = TestTensorFactory::createFP32({seq, top_k});

    auto run_route = [&](TensorBase *run_input,
                         TensorBase *run_indices,
                         TensorBase *run_weights,
                         int run_seq) -> bool
    {
        MoERoutingStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.gate_weights = gate_weights.get();
        params.output_indices = run_indices;
        params.output_weights = run_weights;
        params.seq_len = run_seq;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.norm_topk_prob = true;
        params.layer_idx = 0;

        MoERoutingStage stage(params);
        return stage.execute(cpu_ctx_.get());
    };

    ASSERT_TRUE(run_route(input.get(), multi_indices.get(), multi_weights.get(), seq));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d_model)});
        FP32Tensor row_indices({1, static_cast<size_t>(top_k)});
        FP32Tensor row_weights({1, static_cast<size_t>(top_k)});
        std::copy_n(input->data() + static_cast<size_t>(row) * d_model,
                    d_model,
                    row_input.mutable_data());
        ASSERT_TRUE(run_route(&row_input, &row_indices, &row_weights, 1));

        std::copy_n(row_indices.data(),
                    top_k,
                    split_indices->mutable_data() + static_cast<size_t>(row) * top_k);
        std::copy_n(row_weights.data(),
                    top_k,
                    split_weights->mutable_data() + static_cast<size_t>(row) * top_k);
    }

    const size_t route_bytes =
        static_cast<size_t>(seq) * top_k * sizeof(float);
    EXPECT_EQ(
        std::memcmp(multi_indices->data(), split_indices->data(), route_bytes),
        0)
        << "grouped router indices must match serial M=1 bytes";
    EXPECT_EQ(
        std::memcmp(multi_weights->data(), split_weights->data(), route_bytes),
        0)
        << "grouped router weights must match serial M=1 bytes";
}

TEST_F(MoERoutingStageTest, CPUDecodeEquivalentVerifierFlagSplitsRows)
{
    const int seq = 3;
    const int d_model = 512;
    const int num_experts = 16;
    const int top_k = 4;
    auto input = TestTensorFactory::createFP32Random({seq, d_model}, -0.5f, 0.5f, 1210);
    auto gate_weights = TestTensorFactory::createFP32Random({num_experts, d_model}, -0.1f, 0.1f, 1211);
    auto flagged_indices = TestTensorFactory::createFP32({seq, top_k});
    auto flagged_weights = TestTensorFactory::createFP32({seq, top_k});
    auto split_indices = TestTensorFactory::createFP32({seq, top_k});
    auto split_weights = TestTensorFactory::createFP32({seq, top_k});

    auto make_params = [&](TensorBase *run_input,
                           TensorBase *run_indices,
                           TensorBase *run_weights,
                           int run_seq)
    {
        MoERoutingStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = run_input;
        params.gate_weights = gate_weights.get();
        params.output_indices = run_indices;
        params.output_weights = run_weights;
        params.seq_len = run_seq;
        params.d_model = d_model;
        params.num_experts = num_experts;
        params.top_k = top_k;
        params.norm_topk_prob = true;
        params.layer_idx = 0;
        return params;
    };

    auto flagged_params = make_params(input.get(), flagged_indices.get(), flagged_weights.get(), seq);
    flagged_params.force_decode_equivalent_verifier_prefill = true;
    MoERoutingStage flagged_stage(flagged_params);
    ASSERT_TRUE(flagged_stage.execute(cpu_ctx_.get()));

    for (int row = 0; row < seq; ++row)
    {
        FP32Tensor row_input({1, static_cast<size_t>(d_model)});
        FP32Tensor row_indices({1, static_cast<size_t>(top_k)});
        FP32Tensor row_weights({1, static_cast<size_t>(top_k)});
        std::copy_n(input->data() + static_cast<size_t>(row) * d_model,
                    d_model,
                    row_input.mutable_data());

        MoERoutingStage row_stage(make_params(&row_input, &row_indices, &row_weights, 1));
        ASSERT_TRUE(row_stage.execute(cpu_ctx_.get())) << "row " << row;
        std::copy_n(row_indices.data(),
                    top_k,
                    split_indices->mutable_data() + static_cast<size_t>(row) * top_k);
        std::copy_n(row_weights.data(),
                    top_k,
                    split_weights->mutable_data() + static_cast<size_t>(row) * top_k);
    }

    const size_t route_bytes =
        static_cast<size_t>(seq) * top_k * sizeof(float);
    EXPECT_EQ(
        std::memcmp(flagged_indices->data(), split_indices->data(), route_bytes),
        0)
        << "decode-equivalent verifier indices must match serial M=1 bytes";
    EXPECT_EQ(
        std::memcmp(flagged_weights->data(), split_weights->data(), route_bytes),
        0)
        << "decode-equivalent verifier weights must match serial M=1 bytes";
}

/**
 * @brief Retain actual verifier work as whole batches, independent of acceptance.
 *
 * Includes depth-15 capacity and two request-major groups. The real CPU router
 * produces every top-k row once. A zero/partial acceptance cannot erase that
 * service cost; reset retires cached graph data without retiring frozen evidence.
 */
TEST_F(MoERoutingStageTest, CPUGroupedVerifierEvidenceRetainsExecutedBatches)
{
    for (const int rows : {1, 2, 3, 15, 16, 32})
    {
        SCOPED_TRACE(rows);
        auto input = TestTensorFactory::createFP32Random(
            {rows, D_MODEL}, -0.5f, 0.5f, 1220);
        auto gate_weights = TestTensorFactory::createFP32Random(
            {NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 1221);
        auto output_indices = TestTensorFactory::createFP32({rows, TOP_K});
        auto output_weights = TestTensorFactory::createFP32({rows, TOP_K});

        DecodeExpertHistogramConfig config;
        config.num_layers = 1;
        config.num_experts = NUM_EXPERTS;
        config.top_k = TOP_K;
        config.window_size = 64;
        config.token_boundary_layer_idx = 0;
        config.sockets = {DeviceId::cpu()};
        config.ownership = MoELayeredExpertOwnership::uniform(
            1, 1, std::vector<int>(NUM_EXPERTS, 0));
        admitObservedExpertDemand(config, {.target_rows = 64,
            .max_transaction_rows = 32, .top_k = TOP_K}, 1);
        DecodeExpertHistogram histogram(config);

        MoERoutingStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.gate_weights = gate_weights.get();
        params.output_indices = output_indices.get();
        params.output_weights = output_weights.get();
        params.seq_len = rows;
        params.d_model = D_MODEL;
        params.num_experts = NUM_EXPERTS;
        params.top_k = TOP_K;
        params.norm_topk_prob = true;
        params.layer_idx = 0;
        params.decode_histogram = &histogram;
        params.host_logical_row_count = rows;
        params.host_routing_source = ExpertHistogramSource::GroupedVerifier;
        params.force_decode_equivalent_verifier_prefill = true;

        MoERoutingStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
        EXPECT_EQ(histogram.windowTokenCount(), rows)
            << "execution evidence must not wait for accepted-state publication";
        std::vector<int> expected_routes(rows * TOP_K);
        std::transform(output_indices->data(), output_indices->data() + rows * TOP_K,
            expected_routes.begin(), [](float id) { return static_cast<int>(id); });

        // A second call is another actual invocation, not another publication
        // of the first one. This also checks retained-bank reuse after reset.
        stage.resetSessionState();
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
        const auto frozen = histogram.freezeAndRotateWindow();
        ASSERT_EQ(frozen.token_count, 2u * rows);
        const auto &demand = frozen.validatedView().transactionDemand();
        ASSERT_EQ(demand.layerTransactions(0).size(), 2u);
        for (std::size_t batch = 0; batch < 2; ++batch)
        {
            const auto route = demand.routes(0, batch);
            EXPECT_EQ(demand.layerTransactions(0)[batch].phase,
                ExpertHistogramSource::GroupedVerifier);
            EXPECT_EQ(route.logical_rows, rows);
            for (int slot = 0; slot < rows * TOP_K; ++slot)
                EXPECT_EQ(route.expert_ids[slot], expected_routes[slot]);
        }
        stage.resetSessionState();
        EXPECT_EQ(demand.layerTransactions(0).size(), 2u);
        EXPECT_EQ(histogram.windowTokenCount(), 0u);
    }
}

/** @brief Invalid executed-row geometry cannot publish a partial demand batch. */
TEST_F(MoERoutingStageTest, CPUGroupedVerifierEvidenceRejectsInvalidGeometry)
{
    constexpr int rows = 2;
    auto input = TestTensorFactory::createFP32Random(
        {rows, D_MODEL}, -0.5f, 0.5f, 1230);
    auto gate_weights = TestTensorFactory::createFP32Random(
        {NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 1231);
    auto indices = TestTensorFactory::createFP32({rows, TOP_K});
    auto weights = TestTensorFactory::createFP32({rows, TOP_K});

    DecodeExpertHistogramConfig config;
    config.num_layers = 1;
    config.num_experts = NUM_EXPERTS;
    config.top_k = TOP_K;
    config.window_size = 32;
    config.token_boundary_layer_idx = 0;
    config.sockets = {DeviceId::cpu()};
    config.ownership = MoELayeredExpertOwnership::uniform(
        1, 1, std::vector<int>(NUM_EXPERTS, 0));
    admitObservedExpertDemand(config, {.target_rows = 32,
        .max_transaction_rows = rows, .top_k = TOP_K}, 1);
    DecodeExpertHistogram histogram(config);

    MoERoutingStage::Params params;
    params.device_id = DeviceId::cpu();
    params.input = input.get();
    params.gate_weights = gate_weights.get();
    params.output_indices = indices.get();
    params.output_weights = weights.get();
    params.seq_len = rows;
    params.d_model = D_MODEL;
    params.num_experts = NUM_EXPERTS;
    params.top_k = TOP_K;
    params.layer_idx = 0;
    params.decode_histogram = &histogram;
    params.host_logical_row_count = rows + 1;
    params.host_routing_source = ExpertHistogramSource::GroupedVerifier;
    params.force_decode_equivalent_verifier_prefill = true;
    MoERoutingStage stage(params);
    EXPECT_FALSE(stage.execute(cpu_ctx_.get()));
    EXPECT_EQ(histogram.windowTokenCount(), 0u);
    for (int expert = 0; expert < NUM_EXPERTS; ++expert)
        EXPECT_EQ(histogram.activationCount(0, expert), 0u);
}

/** @brief One row does not imply serial decode: phase is a graph-owned fact. */
TEST_F(MoERoutingStageTest, CPUOneRowRoutingEvidenceHasExplicitPhase)
{
    auto input = TestTensorFactory::createFP32Random({1, D_MODEL}, -0.5f, 0.5f, 1240);
    auto gate = TestTensorFactory::createFP32Random({NUM_EXPERTS, D_MODEL}, -0.1f, 0.1f, 1241);
    auto indices = TestTensorFactory::createFP32({1, TOP_K});
    auto weights = TestTensorFactory::createFP32({1, TOP_K});
    for (const auto phase : {ExpertHistogramSource::DecodeToken,
         ExpertHistogramSource::PrefillChunk, ExpertHistogramSource::GroupedVerifier})
    {
        SCOPED_TRACE(static_cast<int>(phase));
        DecodeExpertHistogramConfig config;
        config.num_layers = 1;
        config.num_experts = NUM_EXPERTS;
        config.top_k = TOP_K;
        config.window_size = 4;
        config.sockets = {DeviceId::cpu()};
        config.ownership = MoELayeredExpertOwnership::uniform(1, 1, std::vector<int>(NUM_EXPERTS, 0));
        admitObservedExpertDemand(config, {.target_rows = 4, .max_transaction_rows = 1, .top_k = TOP_K}, 1);
        DecodeExpertHistogram histogram(config);
        MoERoutingStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input = input.get();
        params.gate_weights = gate.get();
        params.output_indices = indices.get();
        params.output_weights = weights.get();
        params.seq_len = 1;
        params.d_model = D_MODEL;
        params.num_experts = NUM_EXPERTS;
        params.top_k = TOP_K;
        params.layer_idx = 0;
        params.decode_histogram = &histogram;
        EXPECT_THROW((void)MoERoutingStage(params), std::invalid_argument);
        params.host_routing_source = phase;
        MoERoutingStage stage(params);
        ASSERT_TRUE(stage.execute(cpu_ctx_.get()));
        const auto frozen = histogram.freezeAndRotateWindow();
        ASSERT_EQ(frozen.token_count, 1u);
        const auto &demand = frozen.validatedView().transactionDemand();
        ASSERT_EQ(demand.layerTransactions(0).size(), 1u);
        EXPECT_EQ(demand.layerTransactions(0)[0].phase, phase);
        EXPECT_EQ(demand.routes(0, 0).logical_rows, 1u);
    }
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
    histogram_config.sockets = {DeviceId::rocm(0)};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        1, 1, std::vector<int>(NUM_EXPERTS, 0));
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
    histogram_config.sockets = {DeviceId::rocm(0)};
    histogram_config.ownership = MoELayeredExpertOwnership::uniform(
        1, 1, std::vector<int>(NUM_EXPERTS, 0));
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
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.isGraphCapturable());
#else
    EXPECT_FALSE(stage.isGraphCapturable());
#endif
}

TEST_F(MoERoutingStageTest, GraphCapturableRocmDecodeHonorsRuntimeTableFlags)
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
#if defined(HAVE_ROCM)
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
#if defined(HAVE_ROCM)
    EXPECT_TRUE(unprepared_stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(unprepared_stage.supportsGraphCaptureAfterLaunchPreparation());
#endif

    ASSERT_TRUE(runtime_table.prepareInactiveBank(0, routingRuntimeUpdate(1, NUM_EXPERTS, D_MODEL)));
    ASSERT_TRUE(runtime_table.flipActiveBank(0, 1, nullptr));

    MoERoutingStage prepared_stage(params);
#if defined(HAVE_ROCM)
    EXPECT_TRUE(prepared_stage.isGraphCapturable());
    EXPECT_TRUE(prepared_stage.supportsGraphCaptureAfterLaunchPreparation());
#else
    EXPECT_FALSE(prepared_stage.isGraphCapturable());
    EXPECT_FALSE(prepared_stage.supportsGraphCaptureAfterLaunchPreparation());
#endif
}

TEST_F(MoERoutingStageTest, OverlayTicketDecodeIsAColdCaptureContractWithoutRuntimeTable)
{
    ScopedRocmMoEFlags flags(true, true);

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
    params.layer_idx = 0;
    params.decode_route_publication =
        MoEDecodeRoutePublicationPolicy::FixedCapacityOverlayTicket;

    MoERoutingStage stage(params);
    EXPECT_EQ(stage.decodeRoutePublicationPolicyForTesting(),
              MoEDecodeRoutePublicationPolicy::FixedCapacityOverlayTicket);
    EXPECT_FALSE(stage.isGraphCapturable())
        << "Cold preflight must not pretend the backend kernel is prepared";
#if defined(HAVE_ROCM)
    EXPECT_TRUE(stage.supportsGraphCaptureAfterLaunchPreparation())
        << "The explicit fixed-capacity ticket is the decode publication owner";
#else
    EXPECT_FALSE(stage.supportsGraphCaptureAfterLaunchPreparation());
#endif

    MoERuntimeTable conflicting_runtime_table(
        DeviceId::cpu(), 1, NUM_EXPERTS, TOP_K);
    params.moe_runtime_table = &conflicting_runtime_table;
    MoERoutingStage conflicting_authorities(params);
    EXPECT_FALSE(conflicting_authorities.supportsGraphCaptureAfterLaunchPreparation())
        << "A decode graph cannot publish through both a runtime table and a ticket";
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

TEST_F(MoERoutingStageTest, CUDARoutingOutputsRemainDeviceCoherentWithWorkspace)
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
	auto reqs = stage.getWorkspaceRequirements(0, 0, 0);
	DeviceWorkspaceManager workspace(cuda_device, reqs.total_bytes_with_alignment() + 1024 * 1024);
	ASSERT_TRUE(workspace.allocate(reqs));
	stage.bindWorkspace(&workspace);
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
