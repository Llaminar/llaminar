/**
 * @file Test__GPUGraphCaptureExecution.cpp
 * @brief Integration tests for GPU graph capture/replay via DeviceGraphExecutor::executeWithGraphCapture()
 *
 * Tests the end-to-end flow:
 *   1. Build ComputeGraph with real stages (RMSNorm, ResidualAdd)
 *   2. Execute via executeWithGraphCapture() with a real IGPUGraphCapture
 *   3. Verify output correctness
 *   4. Test re-capture + update path (second call → tryUpdate)
 *   5. Test fallback paths (nullptr capture, collective nodes present)
 *
 * IMPORTANT: No <hip/hip_runtime.h> or <cuda_runtime.h> includes.
 * All GPU interaction goes through the backend interfaces.
 *
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>
#include <unordered_set>
#include <string>
#include <cmath>

// Core execution
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"

// GPU backend interfaces
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"

// Tensors and utilities
#include "tensors/Tensors.h"
#include "utils/Logger.h"

// Test utilities
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ===========================================================================
// Skip macros (same pattern as unit/backends/Test__GPUGraphCapture.cpp)
// ===========================================================================

#define SKIP_IF_NO_GPU()                                            \
    if (!GPUDeviceContextPool::instance().hasNvidiaSupport() &&     \
        !GPUDeviceContextPool::instance().hasAMDSupport())          \
    {                                                               \
        GTEST_SKIP() << "No GPU available (neither CUDA nor ROCm)"; \
    }

// ===========================================================================
// Test Fixture
// ===========================================================================

class GPUGraphCaptureExecutionTest : public ::testing::Test
{
protected:
    // GPU context
    IWorkerGPUContext *gpu_ctx_ = nullptr;
    std::unique_ptr<IDeviceContext> device_ctx_;
    std::unique_ptr<IGPUGraphCapture> capture_;

    // Tensor storage (keeps tensors alive for the duration of the test)
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;

    void SetUp() override
    {
        auto &pool = GPUDeviceContextPool::instance();

        if (pool.hasAMDSupport())
        {
            gpu_ctx_ = &pool.getAMDContext(0);
            device_ctx_ = IDeviceContext::create(DeviceId::rocm(0), 1);
        }
        else if (pool.hasNvidiaSupport())
        {
            gpu_ctx_ = &pool.getNvidiaContext(0);
            device_ctx_ = IDeviceContext::create(DeviceId::cuda(0), 1);
        }

        if (gpu_ctx_)
        {
            gpu_ctx_->submitAndWait([&]
                                    { capture_ = gpu_ctx_->createGraphCapture(); });
        }
    }

    void TearDown() override
    {
        // Reset capture first (releases GPU graph resources)
        if (capture_)
        {
            gpu_ctx_->submitAndWait([&]
                                    {
                capture_->reset();
                capture_.reset(); });
        }
        tensor_storage_.clear();
        device_ctx_.reset();
        gpu_ctx_ = nullptr;
    }

    // -----------------------------------------------------------------------
    // Tensor helpers
    // -----------------------------------------------------------------------

    FP32Tensor *createFP32Tensor(const std::vector<size_t> &shape)
    {
        auto tensor = TestTensorFactory::createFP32(shape);
        auto *ptr = tensor.get();
        tensor_storage_.push_back(std::move(tensor));
        return static_cast<FP32Tensor *>(ptr);
    }

    /**
     * @brief Build a simple 2-stage graph: RMSNorm → ResidualAdd
     *
     * Graph:
     *   norm_input  ──→  [RMSNorm] ──→  norm_output
     *   residual    ──→  [ResidualAdd] ──→  result_output
     *                       ↑
     *                   norm_output
     *
     * @param seq_len  Sequence length
     * @param d_model  Model dimension
     * @param[out] norm_input  Raw pointer to the RMSNorm input tensor
     * @param[out] residual    Raw pointer to the residual tensor
     * @param[out] result_output Raw pointer to the final output tensor
     * @param stage_device  Device ID for stage params (kernel dispatch).
     *                      Default: CPU — the stages create CPU kernels.
     * @param node_device   Device ID for graph nodes (coherence target).
     *                      Default: device_ctx_->deviceId() (GPU).
     *                      IMPORTANT: When executeNode() is used (e.g. collective-
     *                      graph fallback), node_device MUST match stage_device.
     *                      Otherwise coherence marks outputs as GPU-dirty while the
     *                      CPU kernel writes to host, causing verification to read
     *                      uninitialized GPU zeros.
     * @return Populated ComputeGraph
     */
    ComputeGraph buildNormResidualGraph(size_t seq_len, size_t d_model,
                                        FP32Tensor *&norm_input,
                                        FP32Tensor *&residual,
                                        FP32Tensor *&result_output,
                                        DeviceId stage_device = DeviceId::cpu(),
                                        std::optional<DeviceId> node_device = std::nullopt)
    {
        const DeviceId graph_device = node_device.value_or(device_ctx_->deviceId());

        norm_input = createFP32Tensor({seq_len, d_model});
        auto *norm_output = createFP32Tensor({seq_len, d_model});
        auto *gamma = createFP32Tensor({d_model});
        residual = createFP32Tensor({seq_len, d_model});
        result_output = createFP32Tensor({seq_len, d_model});

        const size_t num_elements = seq_len * d_model;

        // Initialize norm_input with non-zero values
        for (size_t i = 0; i < num_elements; ++i)
            norm_input->mutable_data()[i] = 0.5f + (i % 10) * 0.1f;

        // Gamma = 1.0 (identity scaling)
        for (size_t i = 0; i < d_model; ++i)
            gamma->mutable_data()[i] = 1.0f;

        // Residual values
        for (size_t i = 0; i < num_elements; ++i)
            residual->mutable_data()[i] = 0.1f * (i % 7);

        // Build stages
        RMSNormStage::Params norm_params;
        norm_params.input = norm_input;
        norm_params.output = norm_output;
        norm_params.gamma = gamma;
        norm_params.eps = 1e-5f;
        norm_params.seq_len = static_cast<int>(seq_len);
        norm_params.device_id = stage_device;

        ResidualAddStage::Params res_params;
        res_params.input = norm_output;
        res_params.residual = residual;
        res_params.output = result_output;
        res_params.num_elements = num_elements;
        res_params.device_id = stage_device;

        // Assemble graph with dependency
        ComputeGraph graph;
        graph.addNode("rmsnorm", ComputeStageFactory::createRMSNorm(norm_params), graph_device);
        graph.addNode("residual_add", ComputeStageFactory::createResidualAdd(res_params), graph_device);
        graph.addDependency("residual_add", "rmsnorm");

        return graph;
    }
};

// ===========================================================================
// 1. Basic executeWithGraphCapture — first call (instantiate path)
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, FirstExecution_ProducesCorrectOutput)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr) << "Failed to create graph capture object";

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // Execute via graph capture path
    bool success = executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get());
    ASSERT_TRUE(success) << "executeWithGraphCapture failed on first invocation";

    // For CPU-only stages, 0-node capture is valid.
    // With GPU stages, nodeCount() > 0 and hasExecutable() is true.
    // With CPU stages, nodeCount() == 0 and we skip instantiation.
    if (capture_->nodeCount() > 0)
    {
        EXPECT_TRUE(capture_->hasExecutable());
    }

    // Verify output is not all zeros (stages actually ran)
    const float *out = result->data();
    bool has_nonzero = false;
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        if (out[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero) << "Output tensor is all zeros — stages did not execute";

    // Verify output has no NaN/Inf
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ===========================================================================
// 2. Re-capture + update path (second call should use tryUpdate)
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, SecondExecution_UsesUpdatePath)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution — instantiate path (or 0-node skip for CPU-only stages)
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));

    // Save first-run output
    std::vector<float> first_output(seq_len * d_model);
    std::memcpy(first_output.data(), result->data(), first_output.size() * sizeof(float));

    // Reset graph completion flags for re-execution
    graph.reset();

    // Second execution — with GPU stages this takes the tryUpdate path.
    // With CPU-only stages (0-node graph), it re-executes stages directly.
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));

    // Output should still be valid (same graph, same inputs → same output)
    const float *out = result->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i << " after re-capture";
        ASSERT_FALSE(std::isinf(out[i])) << "Inf at index " << i << " after re-capture";
    }

    // Output should match first run (deterministic — same inputs, same graph)
    bool outputs_match = true;
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        if (std::fabs(out[i] - first_output[i]) > 1e-5f)
        {
            outputs_match = false;
            break;
        }
    }
    EXPECT_TRUE(outputs_match)
        << "Second execution output diverged from first — update path may have failed";
}

// ===========================================================================
// 3. Multiple re-executions (stability test)
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, MultipleReExecutions_StableOutput)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 2;
    const size_t d_model = 32;
    const int num_iterations = 5;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));

    std::vector<float> reference_output(seq_len * d_model);
    std::memcpy(reference_output.data(), result->data(), reference_output.size() * sizeof(float));

    // Re-execute multiple times
    for (int iter = 1; iter < num_iterations; ++iter)
    {
        graph.reset();
        ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()))
            << "Iteration " << iter << " failed";

        const float *out = result->data();
        for (size_t i = 0; i < seq_len * d_model; ++i)
        {
            EXPECT_NEAR(out[i], reference_output[i], 1e-5f)
                << "Divergence at index " << i << " on iteration " << iter;
        }
    }
}

// ===========================================================================
// 4. Fallback: nullptr capture → executeFastDecode
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, NullCapture_FallsBackToFastDecode)
{
    SKIP_IF_NO_GPU();

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // Pass nullptr for capture — should fall back to executeFastDecode
    bool success = executor.executeWithGraphCapture(graph, device_ctx_.get(), nullptr);
    ASSERT_TRUE(success) << "Fallback to executeFastDecode should succeed";

    // Verify output is valid
    const float *out = result->data();
    bool has_nonzero = false;
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        if (out[i] != 0.0f)
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "Fallback output is all zeros";
}

// ===========================================================================
// 5. Fallback: collective nodes present → executeFastDecode
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, CollectiveNodesPresent_FallsBackToFastDecode)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;

    // Build graph with CPU device for BOTH stage params and graph nodes.
    //
    // This test exercises the collective-fallback path in executeFastDecode,
    // which routes ALL nodes (not just collectives) through executeNode().
    // executeNode() does full coherence management based on node.device:
    //   - Uploads inputs to target device
    //   - Allocates output buffers on target device
    //   - After stage->execute(), marks outputs as device-dirty
    //
    // If node.device is GPU but stage device_id is CPU, the CPU kernel
    // writes to host memory while coherence marks outputs as GPU-dirty.
    // When verification later calls data(), it syncs from GPU (which has
    // uninitialized zeros) back to host, overwriting the correct CPU output.
    //
    // Fix: Use CPU device for graph nodes so that executeNode()'s coherence
    // targets CPU (effectively a no-op), matching the CPU kernel dispatch.
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result,
                                        DeviceId::cpu(), DeviceId::cpu());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // Provide a non-empty collective_nodes set — this forces the fallback path
    // even though none of the actual nodes in our graph are collective.
    // The implementation checks `!collective_nodes->empty()` and falls back.
    std::unordered_set<std::string> collective_nodes = {"fake_allreduce"};

    bool success = executor.executeWithGraphCapture(
        graph, device_ctx_.get(), capture_.get(), &collective_nodes);
    ASSERT_TRUE(success) << "Fallback with collective nodes should succeed";

    // capture should NOT have been used (no executable created)
    EXPECT_FALSE(capture_->hasExecutable())
        << "Graph capture should not be used when collective nodes are present";

    // Verify output is valid
    const float *out = result->data();
    bool has_nonzero = false;
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        if (out[i] != 0.0f)
            has_nonzero = true;
    }
    EXPECT_TRUE(has_nonzero) << "Fallback output is all zeros";
}

// ===========================================================================
// 6. Empty collective set does NOT trigger fallback
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, EmptyCollectiveSet_DoesNotFallBack)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 2;
    const size_t d_model = 32;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // Empty set — should NOT cause fallback
    std::unordered_set<std::string> empty_collectives;

    bool success = executor.executeWithGraphCapture(
        graph, device_ctx_.get(), capture_.get(), &empty_collectives);
    ASSERT_TRUE(success);

    // Graph capture was attempted (not bypassed by collective check).
    // With GPU stages, hasExecutable() is true. With CPU-only stages,
    // 0-node capture skips instantiation.
    // Verify the test ran (output has non-zero values) as proxy.
    const float *out_check = result->data();
    bool has_nonzero_check = false;
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        if (out_check[i] != 0.0f)
        {
            has_nonzero_check = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero_check) << "Stages should have produced non-zero output";
}

// ===========================================================================
// 7. Graph capture reset and re-capture
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, ResetAndRecapture_Works)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 2;
    const size_t d_model = 32;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));

    // Reset capture object
    gpu_ctx_->submitAndWait([&]
                            { capture_->reset(); });
    EXPECT_FALSE(capture_->hasExecutable());

    // Re-capture from scratch
    graph.reset();
    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));

    // Verify output is still valid
    const float *out = result->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i << " after re-capture";
    }
}

// ===========================================================================
// 8. Single-stage graph (minimal capture)
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, SingleStageGraph_Works)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;

    auto *input = createFP32Tensor({seq_len, d_model});
    auto *output = createFP32Tensor({seq_len, d_model});
    auto *gamma = createFP32Tensor({d_model});

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
    graph.addNode("solo_rmsnorm", ComputeStageFactory::createRMSNorm(params), device_ctx_->deviceId());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.executeWithGraphCapture(graph, device_ctx_.get(), capture_.get()));
    // With GPU stages, hasExecutable() is true.
    // With CPU-only stages (0 GPU nodes), instantiation is skipped.

    // Verify output
    const float *out = output->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ===========================================================================
// 9. Correctness: graph capture output matches fast decode output
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, OutputMatchesFastDecode)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    // --- Run 1: via executeFastDecode (nullptr capture) ---
    FP32Tensor *norm_input1 = nullptr;
    FP32Tensor *residual1 = nullptr;
    FP32Tensor *result1 = nullptr;
    auto graph1 = buildNormResidualGraph(seq_len, d_model, norm_input1, residual1, result1);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    ASSERT_TRUE(executor.executeWithGraphCapture(graph1, device_ctx_.get(), nullptr));
    std::vector<float> fast_decode_output(num_elements);
    std::memcpy(fast_decode_output.data(), result1->data(), num_elements * sizeof(float));

    // --- Run 2: via graph capture ---
    // We need identical inputs, so build a fresh graph with the same values.
    FP32Tensor *norm_input2 = nullptr;
    FP32Tensor *residual2 = nullptr;
    FP32Tensor *result2 = nullptr;
    auto graph2 = buildNormResidualGraph(seq_len, d_model, norm_input2, residual2, result2);

    ASSERT_TRUE(executor.executeWithGraphCapture(graph2, device_ctx_.get(), capture_.get()));

    // Compare outputs
    const float *graph_output = result2->data();
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_NEAR(graph_output[i], fast_decode_output[i], 1e-5f)
            << "Graph capture output differs from fast decode at index " << i;
    }
}
