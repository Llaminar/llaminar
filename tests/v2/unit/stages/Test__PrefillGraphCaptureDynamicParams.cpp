/**
 * @file Test__PrefillGraphCaptureDynamicParams.cpp
 * @brief Phase 3 tests: Dynamic parameter mechanism for prefill graph capture.
 *
 * Validates:
 * 1. EmbeddingStage::isPrefillGraphCaptureReady() gating
 * 2. EmbeddingStage::setStableTokenPointer() contract
 * 3. Prelaunch token upload succeeds under graph capture (ROCm)
 * 4. Missing preload fails under graph capture (ROCm)
 * 5. KVCacheAppendStage replay callback contract
 * 6. ForwardGraphCache replay_callback_stages caching
 */

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <vector>

#include "execution/compute_stages/stages/EmbeddingStage.h"
#include "execution/compute_stages/stages/KVCacheAppendStage.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "mocks/MockComputeStage.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/KernelFactory.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#endif

using namespace llaminar2;
using namespace llaminar2::testing;

namespace
{

// =========================================================================
// Mock stage with configurable needsOnGraphReplayed()
// =========================================================================

class MockReplayCallbackStage : public MockComputeStage
{
public:
    explicit MockReplayCallbackStage(bool needs_replay, std::string name = "MockReplay")
        : MockComputeStage(ComputeStageType::GEMM, std::move(name), DeviceId::cpu()),
          needs_replay_(needs_replay) {}

    bool needsOnGraphReplayed() const override { return needs_replay_; }

    void onGraphReplayed() override { replay_count_++; }

    int replayCount() const { return replay_count_; }

private:
    bool needs_replay_ = false;
    int replay_count_ = 0;
};

// =========================================================================
// Test Fixture
// =========================================================================

class Test__PrefillGraphCaptureDynamicParams : public ::testing::Test
{
protected:
    void SetUp() override
    {
        vocab_size_ = 512;
        d_model_ = 64;
    }

    std::unique_ptr<FP32Tensor> createEmbeddingTable() const
    {
        auto table = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(vocab_size_), static_cast<size_t>(d_model_)});
        float *data = table->mutable_data();
        for (int i = 0; i < vocab_size_ * d_model_; ++i)
            data[i] = static_cast<float>(i % 100) * 0.01f;
        return table;
    }

    EmbeddingStage::Params makeParams(const ITensor *embed_table,
                                      const int *token_ids,
                                      ITensor *output,
                                      int num_tokens,
                                      DeviceId device) const
    {
        EmbeddingStage::Params p{};
        p.embed_table = embed_table;
        p.token_ids = token_ids;
        p.output = output;
        p.num_tokens = num_tokens;
        p.d_model = d_model_;
        p.vocab_size = vocab_size_;
        p.device_id = device;
        return p;
    }

    int vocab_size_ = 0;
    int d_model_ = 0;
};

// =========================================================================
// Test 1: isPrefillGraphCaptureReady returns false on CPU
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, EmbeddingReadiness_CPU_ReturnsFalse)
{
    auto embed_table = createEmbeddingTable();
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(d_model_)});
    std::vector<int> tokens = {1, 2, 3, 4};

    auto params = makeParams(embed_table.get(), tokens.data(), output.get(), 4, DeviceId::cpu());
    EmbeddingStage stage(params);

    // CPU device → not ready for graph capture
    EXPECT_FALSE(stage.isPrefillGraphCaptureReady());
}

// =========================================================================
// Test 2: isPrefillGraphCaptureReady returns true on ROCm with workspace
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, EmbeddingReadiness_ROCm_ReturnsTrue)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "No ROCm support compiled";
#else
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    hipSetDevice(0);

    auto embed_table = createEmbeddingTable();
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(d_model_)});
    std::vector<int> tokens = {1, 2, 3, 4};

    embed_table->ensureOnDevice(DeviceId::rocm(0));
    output->ensureOnDevice(DeviceId::rocm(0));

    auto params = makeParams(embed_table.get(), tokens.data(), output.get(), 4, DeviceId::rocm(0));
    EmbeddingStage stage(params);

    // getKernelAsWorkspaceConsumer triggers kernel creation without executing
    auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
    if (!ws_consumer)
        GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

    auto reqs = ws_consumer->getWorkspaceRequirements(4, d_model_, 0);
    DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
    ASSERT_TRUE(wsm.allocate(reqs));
    ws_consumer->bindWorkspace(&wsm);

    EXPECT_TRUE(stage.isPrefillGraphCaptureReady());
#endif
}

// =========================================================================
// Test 3: setStableTokenPointer updates the params token_ids pointer
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, StableTokenPointer_Updates)
{
    auto embed_table = createEmbeddingTable();
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, static_cast<size_t>(d_model_)});
    std::vector<int> tokens_a = {1, 2, 3, 4};
    std::vector<int> tokens_b = {10, 20, 30, 40};

    auto params = makeParams(embed_table.get(), tokens_a.data(), output.get(), 4, DeviceId::cpu());
    EmbeddingStage stage(params);

    // Execute with original pointer
    ASSERT_TRUE(stage.execute(nullptr));
    float first_val = output->data()[0];

    // Update to stable pointer
    stage.setStableTokenPointer(tokens_b.data());

    // Execute with new pointer — should use tokens_b
    ASSERT_TRUE(stage.execute(nullptr));
    float second_val = output->data()[0];

    // Verify different tokens produce different outputs
    // tokens_b[0]=10, tokens_a[0]=1, so row 10 vs row 1 of embed table
    const float *row1 = embed_table->data() + 1 * d_model_;
    const float *row10 = embed_table->data() + 10 * d_model_;
    EXPECT_FLOAT_EQ(first_val, row1[0]);
    EXPECT_FLOAT_EQ(second_val, row10[0]);
}

// =========================================================================
// Test 4: Prelaunch token upload succeeds under graph capture (ROCm)
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, PrelaunchTokenUpload_SucceedsUnderCapture)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "No ROCm support compiled";
#else
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    hipSetDevice(0);

    // Prefill with multiple tokens (seq_len > 1)
    const int seq_len = 8;
    auto embed_table = createEmbeddingTable();
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
    std::vector<int> tokens = {5, 10, 15, 20, 25, 30, 35, 40};

    embed_table->ensureOnDevice(DeviceId::rocm(0));
    output->ensureOnDevice(DeviceId::rocm(0));

    auto params = makeParams(embed_table.get(), tokens.data(), output.get(), seq_len, DeviceId::rocm(0));
    EmbeddingStage stage(params);

    // Create kernel and bind workspace BEFORE first execute
    auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
    if (!ws_consumer)
        GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

    auto reqs = ws_consumer->getWorkspaceRequirements(seq_len, d_model_, 0);
    DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
    ASSERT_TRUE(wsm.allocate(reqs));
    ws_consumer->bindWorkspace(&wsm);

    // Verify basic execution works
    ASSERT_TRUE(stage.execute(nullptr));

    // Prelaunch: updateDynamicParams uploads token_ids to workspace buffer
    stage.updateDynamicParams(/*pos_offset=*/0, /*seq_len=*/seq_len);

    // Now execute under graph capture guard — should succeed because tokens are preloaded
    {
        GraphCaptureGuard guard;
        EXPECT_TRUE(stage.execute(nullptr));
    }
#endif
}

// =========================================================================
// Test 5: Missing preload fails under graph capture (ROCm)
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, NoPreload_FailsUnderCapture)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "No ROCm support compiled";
#else
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    hipSetDevice(0);

    const int seq_len = 4;
    auto embed_table = createEmbeddingTable();
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
    std::vector<int> tokens = {1, 2, 3, 4};

    embed_table->ensureOnDevice(DeviceId::rocm(0));
    output->ensureOnDevice(DeviceId::rocm(0));

    auto params = makeParams(embed_table.get(), tokens.data(), output.get(), seq_len, DeviceId::rocm(0));
    EmbeddingStage stage(params);

    // Create kernel and bind workspace BEFORE execution
    auto *ws_consumer = stage.getKernelAsWorkspaceConsumer();
    if (!ws_consumer)
        GTEST_SKIP() << "Embedding kernel doesn't implement IWorkspaceConsumer";

    auto reqs = ws_consumer->getWorkspaceRequirements(seq_len, d_model_, 0);
    DeviceWorkspaceManager wsm(DeviceId::rocm(0), 64 * 1024 * 1024);
    ASSERT_TRUE(wsm.allocate(reqs));
    ws_consumer->bindWorkspace(&wsm);

    // Execute once outside capture (succeeds with inline H2D)
    ASSERT_TRUE(stage.execute(nullptr));

    // Change tokens WITHOUT calling updateDynamicParams — preloaded data won't match
    std::vector<int> new_tokens = {99, 98, 97, 96};
    stage.setStableTokenPointer(new_tokens.data());

    // Under graph capture: tokens not preloaded → should fail with guard error
    {
        GraphCaptureGuard guard;
        EXPECT_FALSE(stage.execute(nullptr));
    }
#endif
}

// =========================================================================
// Test 6: KVCacheAppendStage advertises replay callback need
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, KVCacheAppend_NeedsReplayCallback)
{
    // KVCacheAppendStage must report needsOnGraphReplayed() == true
    KVCacheAppendStage::Params kv_params{};
    kv_params.device_id = DeviceId::cpu();
    kv_params.layer_idx = 0;
    kv_params.num_tokens = 1;

    KVCacheAppendStage stage(kv_params);
    EXPECT_TRUE(stage.needsOnGraphReplayed());
}

// =========================================================================
// Test 7: ForwardGraphCache caches replay callback stages
// =========================================================================

TEST_F(Test__PrefillGraphCaptureDynamicParams, ForwardGraphCache_CachesReplayCallbacks)
{
    // Build a compute graph with a mix of stages:
    // - 2 stages that need replay callbacks
    // - 2 stages that don't
    ComputeGraph graph;

    auto replay_stage_1 = std::make_unique<MockReplayCallbackStage>(true, "kv_append_0");
    auto replay_stage_2 = std::make_unique<MockReplayCallbackStage>(true, "kv_append_1");
    auto normal_stage_1 = std::make_unique<MockReplayCallbackStage>(false, "gemm_0");
    auto normal_stage_2 = std::make_unique<MockReplayCallbackStage>(false, "norm_0");

    graph.addNode("kv_append_0", std::move(replay_stage_1));
    graph.addNode("gemm_0", std::move(normal_stage_1));
    graph.addNode("kv_append_1", std::move(replay_stage_2));
    graph.addNode("norm_0", std::move(normal_stage_2));

    // Create ForwardGraphCache and populate replay_callback_stages
    ForwardGraphCache cache;
    cache.graph = std::make_unique<ComputeGraph>(std::move(graph));
    cache.valid = true;

    ASSERT_FALSE(cache.replay_callback_stages_cached);

    // Simulate caching logic (same as ForwardExecutionEngine)
    {
        cache.replay_callback_stages.clear();
        const auto &order = cache.graph->getExecutionOrder();
        for (const auto &node_name : order)
        {
            ComputeNode *node = cache.graph->getNode(node_name);
            if (node && node->stage && node->stage->needsOnGraphReplayed())
                cache.replay_callback_stages.push_back(node->stage.get());
        }
        cache.replay_callback_stages_cached = true;
    }

    EXPECT_TRUE(cache.replay_callback_stages_cached);
    EXPECT_EQ(cache.replay_callback_stages.size(), 2u);

    // Verify the correct stages were collected
    for (auto *stage : cache.replay_callback_stages)
    {
        EXPECT_TRUE(stage->needsOnGraphReplayed());
    }

    // Verify onGraphReplayed() can be called
    for (auto *stage : cache.replay_callback_stages)
    {
        stage->onGraphReplayed();
    }

    // Verify invalidation clears the cache
    cache.invalidate();
    EXPECT_FALSE(cache.replay_callback_stages_cached);
    EXPECT_TRUE(cache.replay_callback_stages.empty());
}

} // anonymous namespace

