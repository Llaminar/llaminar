/**
 * @file Test__GraphSnapshotCallbackInvocation.cpp
 * @brief Integration tests verifying snapshot callbacks are invoked for ALL stages during graph execution
 * @author David Sanftenberg
 * @date December 2025
 *
 * These tests ensure that when a snapshot callback is set on DeviceGraphExecutor,
 * it is invoked for EVERY stage after successful execution. This is critical
 * for E2E parity testing and debugging.
 *
 * Test coverage:
 * - Basic stage types (RMSNorm, SwiGLU, ResidualAdd)
 * - Multi-stage graphs with dependencies
 * - Edge cases (no callback, callback change)
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <mutex>
#include <algorithm>

// Core execution components
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/graph/IGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"

// Tensors and utilities
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

// Test utilities
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @class GraphSnapshotCallbackTest
 * @brief Test fixture for verifying snapshot callback invocation
 */
class GraphSnapshotCallbackTest : public ::testing::Test
{
protected:
    std::shared_ptr<MPIContext> mpi_ctx_;
    std::unique_ptr<CPUDeviceContext> ctx_;
    int rank_;
    int world_size_;

    // Callback tracking
    std::mutex callback_mutex_;
    std::unordered_set<std::string> invoked_stages_;
    std::unordered_map<std::string, StageDumpInfo> captured_dumps_;

    // Tensor storage (to keep tensors alive)
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;

    void SetUp() override
    {
        int rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        rank_ = rank;
        world_size_ = world_size;
        mpi_ctx_ = std::make_shared<MPIContext>(rank, world_size, MPI_COMM_WORLD);
        ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4); // device 0, 4 threads

        // Clear tracking state
        invoked_stages_.clear();
        captured_dumps_.clear();
        tensor_storage_.clear();
    }

    void TearDown() override
    {
        tensor_storage_.clear();
        ctx_.reset();
        mpi_ctx_->barrier();
    }

    /**
     * @brief Create a tracking callback that records all invocations
     */
    StageSnapshotCallback createTrackingCallback()
    {
        return [this](const std::string &stage_name, const StageDumpInfo &dump_info)
        {
            std::lock_guard<std::mutex> lock(callback_mutex_);
            invoked_stages_.insert(stage_name);
            captured_dumps_[stage_name] = dump_info;
            LOG_DEBUG("[SnapshotCallback] Invoked for stage: " << stage_name
                                                               << " outputs=" << dump_info.outputs.size());
        };
    }

    /**
     * @brief Verify all expected stages had their callbacks invoked
     */
    void verifyAllStagesInvoked(const std::vector<std::string> &expected_stages)
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);

        for (const auto &stage : expected_stages)
        {
            EXPECT_TRUE(invoked_stages_.count(stage) > 0)
                << "Snapshot callback NOT invoked for stage: " << stage;
        }

        // Also check we didn't miss any stages in the graph
        EXPECT_EQ(invoked_stages_.size(), expected_stages.size())
            << "Invoked " << invoked_stages_.size() << " stages, expected " << expected_stages.size();
    }

    /**
     * @brief Create and store an FP32 tensor for testing
     */
    FP32Tensor *createFP32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<FP32Tensor *>(ptr);
    }

    /**
     * @brief Create and store a Q16_1 tensor for testing
     */
    Q16_1Tensor *createQ16_1Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = std::make_unique<Q16_1Tensor>(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<Q16_1Tensor *>(ptr);
    }
};

// =============================================================================
// Basic Stage Callback Tests
// =============================================================================

/**
 * @brief Test that RMSNormStage triggers snapshot callback
 */
TEST_F(GraphSnapshotCallbackTest, RMSNormStage_CallbackInvoked)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    // Create input/output buffers
    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    // Initialize gamma to 1.0 for identity scaling
    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    // Create stage params
    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    // Build graph
    ComputeGraph graph;
    graph.addNode("test_rmsnorm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    // Create executor with snapshot callback
    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // Execute with graph
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    // Verify callback was invoked
    verifyAllStagesInvoked({"test_rmsnorm"});
}

/**
 * @brief Test that SwiGLUStage triggers snapshot callback
 */
TEST_F(GraphSnapshotCallbackTest, SwiGLUStage_CallbackInvoked)
{
    const size_t d_ff = 128;
    const size_t seq_len = 4;

    auto *gate = createFP32Tensor({seq_len, d_ff});
    auto *up = createFP32Tensor({seq_len, d_ff});
    auto *output = createFP32Tensor({seq_len, d_ff});

    // Initialize with non-zero values
    for (size_t i = 0; i < seq_len * d_ff; ++i)
    {
        gate->mutable_data()[i] = 0.5f;
        up->mutable_data()[i] = 0.5f;
    }

    SwiGLUStage::Params params;
    params.gate = gate;
    params.up = up;
    params.output = output;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_swiglu", ComputeStageFactory::createSwiGLU(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    verifyAllStagesInvoked({"test_swiglu"});
}

/**
 * @brief Test that ResidualAddStage triggers snapshot callback
 */
TEST_F(GraphSnapshotCallbackTest, ResidualAddStage_CallbackInvoked)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;
    const size_t num_elements = seq_len * d_model;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *residual = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});

    // Initialize with non-zero values
    for (size_t i = 0; i < num_elements; ++i)
    {
        input->mutable_data()[i] = 0.5f;
        residual->mutable_data()[i] = 0.5f;
    }

    ResidualAddStage::Params params;
    params.input = input;
    params.residual = residual;
    params.output = output;
    params.num_elements = num_elements;

    ComputeGraph graph;
    graph.addNode("test_residual", ComputeStageFactory::createResidualAdd(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    verifyAllStagesInvoked({"test_residual"});
}

/**
 * @brief Test that FusedAttentionWoStage triggers snapshot callback
 *
 * This is a critical test because FusedAttentionWoStage is a complex fused
 * operation that combines attention computation with Wo projection.
 * Verifies that snapshot callback is invoked after successful execution.
 *
 * Uses REFERENCE backend with Q8_1 tensors for simplicity.
 */
TEST_F(GraphSnapshotCallbackTest, FusedAttentionWoStage_CallbackInvoked)
{
    // Test dimensions (small for unit tests)
    const size_t seq_len = 4;
    const size_t kv_len = 8;
    const size_t n_heads = 2;
    const size_t n_kv_heads = 2;
    const size_t head_dim = 32;
    const size_t d_model = n_heads * head_dim; // 64

    // Create Q8_1 tensors for Q/K/V (using TestTensorFactory)
    auto Q_unique = TestTensorFactory::createQ8_1Random({seq_len, d_model}, -1.0f, 1.0f);
    auto K_unique = TestTensorFactory::createQ8_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f);
    auto V_unique = TestTensorFactory::createQ8_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f);

    // Keep raw pointers before moving
    auto *Q = Q_unique.get();
    auto *K = K_unique.get();
    auto *V = V_unique.get();

    // Move to storage to keep alive
    tensor_storage_.push_back(std::move(Q_unique));
    tensor_storage_.push_back(std::move(K_unique));
    tensor_storage_.push_back(std::move(V_unique));

    // Wo weights (FP32)
    auto *Wo = createFP32Tensor({d_model, d_model});

    // Output (FP32 for REFERENCE backend)
    auto *output = createFP32Tensor({seq_len, d_model});

    // Optional context_snapshot buffer for debugging
    auto *context_snapshot = createFP32Tensor({seq_len, d_model});

    // Initialize Wo weights to identity-ish values (for numerical stability)
    for (size_t i = 0; i < d_model * d_model; ++i)
    {
        Wo->mutable_data()[i] = (i % (d_model + 1) == 0) ? 0.1f : 0.0f;
    }

    FusedAttentionWoStage::Params params;
    params.Q = Q;
    params.K = K;
    params.V = V;
    params.Wo = Wo;
    params.output = output;
    params.batch_size = 1;
    params.seq_len = static_cast<int>(seq_len);
    params.kv_len = static_cast<int>(kv_len);
    params.n_heads = static_cast<int>(n_heads);
    params.n_kv_heads = static_cast<int>(n_kv_heads);
    params.head_dim = static_cast<int>(head_dim);
    params.d_model = static_cast<int>(d_model);
    params.causal = true;
    params.backend = FusedAttentionBackend::REFERENCE; // Simple C++ reference implementation
    params.fuse_residual_add = false;                  // Not needed for REFERENCE backend
    params.context_snapshot = context_snapshot;

    ComputeGraph graph;
    graph.addNode("test_fused_attn_wo", ComputeStageFactory::createFusedAttentionWo(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // Execute the graph
    bool success = executor.execute(graph, ctx_.get());
    ASSERT_TRUE(success) << "FusedAttentionWoStage execution failed";

    // Verify callback was invoked
    verifyAllStagesInvoked({"test_fused_attn_wo"});

    // Additionally verify the dump info contains expected fields
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        ASSERT_TRUE(captured_dumps_.count("test_fused_attn_wo") > 0);

        const auto &dump_info = captured_dumps_["test_fused_attn_wo"];

        // Should have scalars for dimensions
        EXPECT_GE(dump_info.scalars.size(), 5u)
            << "FusedAttentionWoStage dump info should have dimension scalars";

        // Should have at least the output buffer (and optionally context_snapshot)
        EXPECT_GE(dump_info.outputs.size(), 1u)
            << "FusedAttentionWoStage dump info should have output buffer";
    }
}

/**
 * @brief Test FusedAttentionWoStage with Q16_INTEGER backend (HybridQ16 mode)
 *
 * Q16_INTEGER backend requirements:
 * - fuse_residual_add=true (HybridQ16 mode)
 * - Q16_1 output tensor (serves as residual_in and residual_out)
 * - Q8_1 tensors for Q/K/V
 * - Q8_1 Wo tensor (auto-packed to VNNI by the stage)
 *
 * Uses TestTensorFactory's Q16_1 and Q8_1FromFP32Random helpers.
 *
 * DISABLED: HybridQ16 project on hold - Q16_INTEGER backend tests skipped.
 */
TEST_F(GraphSnapshotCallbackTest, DISABLED_FusedAttentionWoStage_Q16IntegerBackend_CallbackInvoked)
{
    // Test dimensions (small for unit tests)
    const size_t seq_len = 4;
    const size_t kv_len = 8;
    const size_t n_heads = 2;
    const size_t n_kv_heads = 2;
    const size_t head_dim = 32;
    const size_t d_model = n_heads * head_dim; // 64

    // Create Q16_1 tensors for Q/K/V (Q16_INTEGER kernel expects Q16_1 format)
    auto Q_unique = test::TestTensorFactory::createQ16_1Random({seq_len, d_model}, -1.0f, 1.0f, 42);
    auto K_unique = test::TestTensorFactory::createQ16_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 43);
    auto V_unique = test::TestTensorFactory::createQ16_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 44);

    auto *Q = Q_unique.get();
    auto *K = K_unique.get();
    auto *V = V_unique.get();

    tensor_storage_.push_back(std::move(Q_unique));
    tensor_storage_.push_back(std::move(K_unique));
    tensor_storage_.push_back(std::move(V_unique));

    // Create Q8_1 Wo weight tensor (will be auto-packed to VNNI by the stage)
    auto Wo_q8 = test::TestTensorFactory::createQ8_1FromFP32Random(
        {d_model, d_model},
        /*mean=*/0.0f, /*stddev=*/0.1f, /*seed=*/123);
    TensorBase *Wo = Wo_q8.get();

    // Create Q16_1 output tensor (serves as residual for fused mode)
    // Initialize with small random values to simulate residual input
    auto output_unique = test::TestTensorFactory::createQ16_1Random(
        {seq_len, d_model}, -0.5f, 0.5f, 99);
    auto *output = output_unique.get();
    tensor_storage_.push_back(std::move(output_unique));

    // Context snapshot buffer for debugging (FP32)
    auto *context_snapshot = createFP32Tensor({seq_len, d_model});

    FusedAttentionWoStage::Params params;
    params.Q = Q;
    params.K = K;
    params.V = V;
    params.Wo = Wo;         // Q8_1, will be auto-packed to VNNI
    params.output = output; // Q16_1 tensor (serves as residual)
    params.batch_size = 1;
    params.seq_len = static_cast<int>(seq_len);
    params.kv_len = static_cast<int>(kv_len);
    params.n_heads = static_cast<int>(n_heads);
    params.n_kv_heads = static_cast<int>(n_kv_heads);
    params.head_dim = static_cast<int>(head_dim);
    params.d_model = static_cast<int>(d_model);
    params.causal = true;
    params.backend = FusedAttentionBackend::Q16_INTEGER;
    params.fuse_residual_add = true; // Required for Q16_INTEGER
    params.context_snapshot = context_snapshot;

    ComputeGraph graph;
    graph.addNode("test_fused_attn_wo_q16", ComputeStageFactory::createFusedAttentionWo(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // Execute the graph
    bool success = executor.execute(graph, ctx_.get());
    ASSERT_TRUE(success) << "FusedAttentionWoStage (Q16_INTEGER) execution failed";

    // Verify callback was invoked
    verifyAllStagesInvoked({"test_fused_attn_wo_q16"});

    // Additionally verify the dump info contains expected fields
    {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        ASSERT_TRUE(captured_dumps_.count("test_fused_attn_wo_q16") > 0);

        const auto &dump_info = captured_dumps_["test_fused_attn_wo_q16"];

        // Should have scalars for dimensions
        EXPECT_GE(dump_info.scalars.size(), 5u)
            << "FusedAttentionWoStage (Q16) dump info should have dimension scalars";

        // Should have at least the output buffer
        EXPECT_GE(dump_info.outputs.size(), 1u)
            << "FusedAttentionWoStage (Q16) dump info should have output buffer";
    }
}

/**
 * @brief Test Q16_INTEGER backend snapshot output keys and data validity
 *
 * This test verifies that:
 * 1. The "context" snapshot key is present (maps to ATTENTION_CONTEXT in pipeline)
 * 2. The context snapshot data has correct dimensions [seq_len × (n_heads * head_dim)]
 * 3. The context snapshot contains valid FP32 values (no NaN/Inf)
 * 4. The context snapshot is not all zeros (kernel actually wrote data)
 * 5. All expected scalar keys are present with correct values
 *
 * DISABLED: HybridQ16 project on hold - Q16_INTEGER backend tests skipped.
 */
TEST_F(GraphSnapshotCallbackTest, DISABLED_Q16IntegerBackend_SnapshotOutputsCorrectKeysAndData)
{
    // Test dimensions
    const size_t seq_len = 4;
    const size_t kv_len = 8;
    const size_t n_heads = 2;
    const size_t n_kv_heads = 2;
    const size_t head_dim = 32;
    const size_t d_model = n_heads * head_dim; // 64

    // Create Q16_1 tensors for Q/K/V (Q16_INTEGER kernel expects Q16_1 format)
    auto Q_unique = test::TestTensorFactory::createQ16_1Random({seq_len, d_model}, -1.0f, 1.0f, 100);
    auto K_unique = test::TestTensorFactory::createQ16_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 101);
    auto V_unique = test::TestTensorFactory::createQ16_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 102);

    auto *Q = Q_unique.get();
    auto *K = K_unique.get();
    auto *V = V_unique.get();

    tensor_storage_.push_back(std::move(Q_unique));
    tensor_storage_.push_back(std::move(K_unique));
    tensor_storage_.push_back(std::move(V_unique));

    // Create Q8_1 Wo weight tensor
    auto Wo_q8 = test::TestTensorFactory::createQ8_1FromFP32Random(
        {d_model, d_model}, 0.0f, 0.1f, 200);
    TensorBase *Wo = Wo_q8.get();

    // Create Q16_1 output tensor (residual)
    auto output_unique = test::TestTensorFactory::createQ16_1Random(
        {seq_len, d_model}, -0.5f, 0.5f, 300);
    auto *output = output_unique.get();
    tensor_storage_.push_back(std::move(output_unique));

    // Context snapshot buffer (FP32) - this is what we're testing!
    auto *context_snapshot = createFP32Tensor({seq_len, d_model});

    // Initialize to sentinel value to verify kernel writes data
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        context_snapshot->mutable_data()[i] = -999.0f;
    }

    FusedAttentionWoStage::Params params;
    params.Q = Q;
    params.K = K;
    params.V = V;
    params.Wo = Wo;
    params.output = output;
    params.batch_size = 1;
    params.seq_len = static_cast<int>(seq_len);
    params.kv_len = static_cast<int>(kv_len);
    params.n_heads = static_cast<int>(n_heads);
    params.n_kv_heads = static_cast<int>(n_kv_heads);
    params.head_dim = static_cast<int>(head_dim);
    params.d_model = static_cast<int>(d_model);
    params.causal = true;
    params.backend = FusedAttentionBackend::Q16_INTEGER;
    params.fuse_residual_add = true;
    params.context_snapshot = context_snapshot;

    ComputeGraph graph;
    graph.addNode("attn_q16", ComputeStageFactory::createFusedAttentionWo(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);
    bool success = executor.execute(graph, ctx_.get());
    ASSERT_TRUE(success) << "Q16_INTEGER attention execution failed";

    // Verify the dump info
    std::lock_guard<std::mutex> lock(callback_mutex_);
    ASSERT_TRUE(captured_dumps_.count("attn_q16") > 0) << "Stage dump not captured";

    const auto &dump_info = captured_dumps_["attn_q16"];

    // =========================================================================
    // Test 1: Verify "context" output key is present (maps to ATTENTION_CONTEXT)
    // =========================================================================
    bool has_context_key = false;
    const StageDumpInfo::OutputBuffer *context_output = nullptr;
    for (const auto &out : dump_info.outputs)
    {
        if (std::string(out.name) == "context")
        {
            has_context_key = true;
            context_output = &out;
            break;
        }
    }
    ASSERT_TRUE(has_context_key)
        << "Q16_INTEGER backend must provide 'context' snapshot (ATTENTION_CONTEXT)";
    ASSERT_NE(context_output, nullptr);

    // =========================================================================
    // Test 2: Verify context snapshot dimensions
    // =========================================================================
    ASSERT_NE(context_output->data, nullptr) << "Context snapshot data is null";
    EXPECT_EQ(context_output->rows, seq_len)
        << "Context snapshot rows should be seq_len=" << seq_len;
    EXPECT_EQ(context_output->cols, d_model) // n_heads * head_dim
        << "Context snapshot cols should be d_model=" << d_model;

    // =========================================================================
    // Test 3: Verify context snapshot contains valid FP32 values
    // =========================================================================
    const float *ctx_data = static_cast<const float *>(context_output->data);
    size_t context_size = context_output->rows * context_output->cols;
    for (size_t i = 0; i < context_size; ++i)
    {
        float val = ctx_data[i];
        ASSERT_FALSE(std::isnan(val)) << "Context snapshot contains NaN at index " << i;
        ASSERT_FALSE(std::isinf(val)) << "Context snapshot contains Inf at index " << i;
    }

    // =========================================================================
    // Test 4: Verify context snapshot is not all sentinel values (kernel wrote data)
    // =========================================================================
    bool has_non_sentinel = false;
    for (size_t i = 0; i < context_size; ++i)
    {
        if (ctx_data[i] != -999.0f)
        {
            has_non_sentinel = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_sentinel)
        << "Context snapshot still contains sentinel values - kernel may not have written data";

    // =========================================================================
    // Test 5: Verify all expected scalar keys with correct values
    // =========================================================================
    std::unordered_map<std::string, int64_t> expected_scalars = {
        {"seq_len", static_cast<int64_t>(seq_len)},
        {"kv_len", static_cast<int64_t>(kv_len)},
        {"n_heads", static_cast<int64_t>(n_heads)},
        {"n_kv_heads", static_cast<int64_t>(n_kv_heads)},
        {"head_dim", static_cast<int64_t>(head_dim)},
        {"d_model", static_cast<int64_t>(d_model)},
    };

    for (const auto &[name, expected_val] : expected_scalars)
    {
        bool found = false;
        for (const auto &scalar : dump_info.scalars)
        {
            if (std::string(scalar.name) == name)
            {
                found = true;
                EXPECT_EQ(static_cast<int64_t>(scalar.value), expected_val)
                    << "Scalar '" << name << "' has wrong value";
                break;
            }
        }
        EXPECT_TRUE(found) << "Expected scalar '" << name << "' not found in dump info";
    }

    // =========================================================================
    // Test 6: Verify backend scalar is Q16_INTEGER (enum value)
    // =========================================================================
    bool found_backend = false;
    for (const auto &scalar : dump_info.scalars)
    {
        if (std::string(scalar.name) == "backend")
        {
            found_backend = true;
            EXPECT_EQ(static_cast<int>(scalar.value), static_cast<int>(FusedAttentionBackend::Q16_INTEGER))
                << "Backend scalar should be Q16_INTEGER";
            break;
        }
    }
    EXPECT_TRUE(found_backend) << "Expected scalar 'backend' not found in dump info";
}

/**
 * @brief Verify Q16_INTEGER backend rejects wrong tensor types with clear error
 *
 * Q16_INTEGER backend requires Q16_1 tensors for Q/K/V. This test verifies that
 * passing Q8_1 tensors (wrong type) is properly rejected with a descriptive error.
 *
 * DISABLED: HybridQ16 project on hold - Q16_INTEGER backend tests skipped.
 */
TEST_F(GraphSnapshotCallbackTest, DISABLED_Q16IntegerBackend_RejectsWrongTensorType)
{
    const size_t seq_len = 4;
    const size_t kv_len = 8;
    const size_t n_heads = 2;
    const size_t n_kv_heads = 2;
    const size_t head_dim = 32;
    const size_t d_model = n_heads * head_dim;

    // Intentionally create Q8_1 tensors (WRONG type for Q16_INTEGER backend)
    auto Q_unique = TestTensorFactory::createQ8_1Random({seq_len, d_model}, -1.0f, 1.0f, 42);
    auto K_unique = TestTensorFactory::createQ8_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 43);
    auto V_unique = TestTensorFactory::createQ8_1Random({kv_len, n_kv_heads * head_dim}, -1.0f, 1.0f, 44);

    auto *Q = Q_unique.get();
    auto *K = K_unique.get();
    auto *V = V_unique.get();

    tensor_storage_.push_back(std::move(Q_unique));
    tensor_storage_.push_back(std::move(K_unique));
    tensor_storage_.push_back(std::move(V_unique));

    auto Wo_q8 = test::TestTensorFactory::createQ8_1FromFP32Random({d_model, d_model}, 0.0f, 0.1f, 123);
    TensorBase *Wo = Wo_q8.get();

    auto output_unique = test::TestTensorFactory::createQ16_1Random({seq_len, d_model}, -0.5f, 0.5f, 99);
    auto *output = output_unique.get();
    tensor_storage_.push_back(std::move(output_unique));

    FusedAttentionWoStage::Params params;
    params.Q = Q;
    params.K = K;
    params.V = V;
    params.Wo = Wo;
    params.output = output;
    params.batch_size = 1;
    params.seq_len = static_cast<int>(seq_len);
    params.kv_len = static_cast<int>(kv_len);
    params.n_heads = static_cast<int>(n_heads);
    params.n_kv_heads = static_cast<int>(n_kv_heads);
    params.head_dim = static_cast<int>(head_dim);
    params.d_model = static_cast<int>(d_model);
    params.causal = true;
    params.backend = FusedAttentionBackend::Q16_INTEGER;
    params.fuse_residual_add = true;

    ComputeGraph graph;
    graph.addNode("wrong_type_test", ComputeStageFactory::createFusedAttentionWo(params), DeviceId::cpu());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // Execution should fail due to wrong tensor types
    bool success = executor.execute(graph, ctx_.get());
    EXPECT_FALSE(success) << "Q16_INTEGER backend should reject Q8_1 tensors for Q/K/V";
}

// =============================================================================
// Multi-Stage Graph Callback Tests
// =============================================================================

/**
 * @brief Test that ALL stages in a multi-stage graph trigger callbacks
 */
TEST_F(GraphSnapshotCallbackTest, MultiStageGraph_AllCallbacksInvoked)
{
    const size_t d_model = 64;
    const size_t d_ff = 128;
    const size_t seq_len = 4;

    // Create shared buffers
    auto *input = createFP32Tensor({seq_len, d_model});
    auto *norm_out = createFP32Tensor({seq_len, d_model});
    auto *gate = createFP32Tensor({seq_len, d_ff});
    auto *up = createFP32Tensor({seq_len, d_ff});
    auto *swiglu_out = createFP32Tensor({seq_len, d_ff});
    auto *gamma = createFP32Tensor({d_model});
    auto *residual_out = createFP32Tensor({seq_len, d_model});

    // Initialize gamma to 1.0
    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    // Initialize other tensors with non-zero values
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f;
    for (size_t i = 0; i < seq_len * d_ff; ++i)
    {
        gate->mutable_data()[i] = 0.5f;
        up->mutable_data()[i] = 0.5f;
    }

    // Build multi-stage graph
    ComputeGraph graph;

    // Stage 1: RMSNorm
    {
        RMSNormStage::Params params;
        params.input = input;
        params.output = norm_out;
        params.gamma = gamma;
        params.eps = 1e-5f;
        params.seq_len = static_cast<int>(seq_len);
        graph.addNode("stage1_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());
    }

    // Stage 2: SwiGLU (simulating FFN activation)
    {
        SwiGLUStage::Params params;
        params.gate = gate;
        params.up = up;
        params.output = swiglu_out;
        params.seq_len = static_cast<int>(seq_len);
        graph.addNode("stage2_swiglu", ComputeStageFactory::createSwiGLU(params), DeviceId::cpu());
        graph.addDependency("stage2_swiglu", "stage1_norm");
    }

    // Stage 3: Residual add
    {
        ResidualAddStage::Params params;
        params.input = norm_out;
        params.residual = input;
        params.output = residual_out;
        params.num_elements = seq_len * d_model;
        graph.addNode("stage3_residual", ComputeStageFactory::createResidualAdd(params), DeviceId::cpu());
        graph.addDependency("stage3_residual", "stage2_swiglu");
    }

    // Create executor with tracking callback
    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    // Verify ALL three stages had callbacks invoked
    verifyAllStagesInvoked({"stage1_norm", "stage2_swiglu", "stage3_residual"});
}

// =============================================================================
// Edge Cases
// =============================================================================

/**
 * @brief Test that callback not set = no crash
 */
TEST_F(GraphSnapshotCallbackTest, NoCallback_NoFailure)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    // NO callback set
    GraphExecutorConfig config;
    // config.snapshot_callback is nullptr by default

    DeviceGraphExecutor executor(config);

    // Should not crash even without callback
    EXPECT_TRUE(executor.execute(graph, ctx_.get()));
}

/**
 * @brief Test that callback can be changed between executions
 */
TEST_F(GraphSnapshotCallbackTest, CallbackCanBeChanged)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    GraphExecutorConfig config;
    config.snapshot_callback = createTrackingCallback();

    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(invoked_stages_.size(), 1u);

    // Reset graph for re-execution
    graph.reset();

    // Change callback to a different one that uses a counter
    int invoke_count = 0;
    executor.setSnapshotCallback([&invoke_count](const std::string &, const StageDumpInfo &)
                                 { invoke_count++; });

    // Second execution
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));
    EXPECT_EQ(invoke_count, 1);
}

/**
 * @brief Test that StageDumpInfo contains valid output data when callback is invoked
 */
TEST_F(GraphSnapshotCallbackTest, CallbackReceivesValidDumpInfo)
{
    const size_t d_model = 64;
    const size_t seq_len = 4;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

    // Initialize input with non-zero values (required for buffer validation)
    for (size_t i = 0; i < seq_len * d_model; ++i)
        input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

    for (size_t i = 0; i < d_model; ++i)
        gamma->mutable_data()[i] = 1.0f;

    RMSNormStage::Params params;
    params.input = input;
    params.output = output;
    params.gamma = gamma;
    params.eps = 1e-5f;
    params.seq_len = static_cast<int>(seq_len);

    ComputeGraph graph;
    graph.addNode("test_norm", ComputeStageFactory::createRMSNorm(params), DeviceId::cpu());

    bool callback_invoked = false;
    StageDumpInfo captured_info;

    GraphExecutorConfig config;
    config.snapshot_callback = [&](const std::string &name, const StageDumpInfo &info)
    {
        callback_invoked = true;
        captured_info = info;
    };

    DeviceGraphExecutor executor(config);
    ASSERT_TRUE(executor.execute(graph, ctx_.get()));

    EXPECT_TRUE(callback_invoked);
    // RMSNorm should have at least one output
    EXPECT_GE(captured_info.outputs.size(), 1u)
        << "RMSNormStage should report at least one output in dump info";
}

// =============================================================================
// MPI-aware main
// =============================================================================

int main(int argc, char **argv)
{
    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Initialize GTest
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize MPI
    MPI_Finalize();

    return result;
}
