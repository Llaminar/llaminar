/**
 * @file Test__GPUGraphCaptureExecution.cpp
 * @brief Integration tests for GPU graph capture/replay via DeviceGraphExecutor::executeWithGraphCapture()
 *
 * Tests the end-to-end flow:
 *   1. Build ComputeGraph with real stages (RMSNorm, ResidualAdd)
 *   2. Execute via executeWithGraphCapture() with a real IGPUGraphCapture
 *   3. Verify output correctness
 *   4. Record a production child graph without launching it
 *   5. Test backend-specific executable publication after re-capture
 *   6. Prove invalid capture ownership fails closed
 *
 * IMPORTANT: No <hip/hip_runtime.h> or <cuda_runtime.h> includes.
 * All GPU interaction goes through the backend interfaces.
 *
 * @date February 2026
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>
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
#include "planning/PhysicalMemoryAuthority.h"

// Tensors and utilities
#include "tensors/Tensors.h"
#include "utils/Logger.h"

// Test utilities
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ===========================================================================
// Skip macros for the backend linked into this binary
// ===========================================================================

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define ENSURE_LINKED_GPU_BACKEND() ensureNvidiaFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasNvidiaSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getNvidiaContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cuda(0)
#define LINKED_GPU_SKIP_MESSAGE "CUDA not available"
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define ENSURE_LINKED_GPU_BACKEND() ensureAMDFactoryRegistered()
#define HAS_LINKED_GPU_SUPPORT() GPUDeviceContextPool::instance().hasAMDSupport()
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getAMDContext(0)
#define LINKED_GPU_DEVICE_ID() DeviceId::rocm(0)
#define LINKED_GPU_SKIP_MESSAGE "ROCm not available"
#else
#define ENSURE_LINKED_GPU_BACKEND() ((void)0)
#define HAS_LINKED_GPU_SUPPORT() false
#define LINKED_GPU_CONTEXT() GPUDeviceContextPool::instance().getContext("", 0)
#define LINKED_GPU_DEVICE_ID() DeviceId::cpu()
#define LINKED_GPU_SKIP_MESSAGE "No GPU backend linked in this test binary"
#endif

#define SKIP_IF_NO_GPU()                                      \
    do                                                        \
    {                                                         \
        ENSURE_LINKED_GPU_BACKEND();                          \
        if (!HAS_LINKED_GPU_SUPPORT())                        \
            GTEST_SKIP() << LINKED_GPU_SKIP_MESSAGE;          \
    } while (false)

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
    BufferArena arena_;

    // Tensor storage (keeps tensors alive for the duration of the test)
    std::vector<std::unique_ptr<TensorBase>> tensor_storage_;

    void SetUp() override
    {
        ENSURE_LINKED_GPU_BACKEND();
        if (HAS_LINKED_GPU_SUPPORT())
        {
            gpu_ctx_ = &LINKED_GPU_CONTEXT();
            device_ctx_ = IDeviceContext::create(LINKED_GPU_DEVICE_ID(), 1);
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
     * @brief Make all synthetic tensor parameters resident before capture.
     *
     * These integration stages bind raw tensor pointers rather than arena
     * BufferIds. The fixture must therefore allocate and upload every tensor
     * before beginCapture(), then wait once on the upload stream. No allocation
     * or H2D transfer is allowed inside the captured graph.
     *
     * @param excluded_tensor Optional internal output whose cold device-storage
     *        lifecycle should be left for capture preflight to exercise.
     * @return true when every included tensor has a stable device address and
     *         the context-owned stream has completed all uploads.
     */
    bool prepareFixtureTensorsForGPUExecution(
        ITensor *excluded_tensor = nullptr)
    {
        if (!gpu_ctx_ || !device_ctx_)
            return false;

        void *stream = gpu_ctx_->defaultStream();
        if (!stream)
            return false;

        const DeviceId device = device_ctx_->deviceId();
        for (const auto &tensor : tensor_storage_)
        {
            if (tensor.get() == excluded_tensor)
                continue;
            if (!tensor || !tensor->ensureOnDevice(device, stream))
                return false;
        }
        return gpu_ctx_->synchronizeStreamChecked(stream);
    }

    /**
     * @brief Execute one mandatory capture on the context-owned stream.
     *
     * @param graph GPU-only graph whose tensor parameters are already resident.
     * @param executor Executor under test.
     * @param externally_visible_output Output whose post-launch completion
     *        becomes observable to the test.
     * @return true only when capture, instantiation/update, and launch succeed.
     */
    bool executeCapturedGraph(
        ComputeGraph &graph,
        DeviceGraphExecutor &executor,
        ITensor *externally_visible_output)
    {
        executor.setArena(&arena_);
        const std::array<ITensor *, 1> outputs = {
            externally_visible_output};
        return executor.executeWithGraphCapture(
            graph,
            device_ctx_.get(),
            capture_.get(),
            outputs);
    }

    /**
     * @brief Require evidence that the selected path owns a real GPU graph.
     */
    void expectExecutableGraph() const
    {
        ASSERT_NE(capture_, nullptr);
        EXPECT_GT(capture_->nodeCount(), 0u)
            << "Positive graph-capture integration cases must capture device nodes";
        EXPECT_TRUE(capture_->hasExecutable())
            << "Positive graph-capture integration cases must instantiate an executable";
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
     * @param[out] norm_output_out Optional raw pointer to the internal edge.
     * @param strict_copy_consumer When true, the second stage performs a D2D
     *        copy and invokes StageGPUExecution::requirePreparedInput(), directly
     *        exercising capture-time internal-edge authority.
     * @return Populated ComputeGraph
     */
    ComputeGraph buildNormResidualGraph(size_t seq_len, size_t d_model,
                                        FP32Tensor *&norm_input,
                                        FP32Tensor *&residual,
                                        FP32Tensor *&result_output,
                                        FP32Tensor **norm_output_out = nullptr,
                                        bool strict_copy_consumer = false)
    {
        const DeviceId device = device_ctx_->deviceId();

        norm_input = createFP32Tensor({seq_len, d_model});
        auto *norm_output = createFP32Tensor({seq_len, d_model});
        if (norm_output_out)
            *norm_output_out = norm_output;
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
        norm_params.device_id = device;
        norm_params.input_buffer_id = BufferId::HIDDEN_STATE;
        norm_params.output_buffer_id = BufferId::NORMALIZED;

        ResidualAddStage::Params res_params;
        res_params.input = norm_output;
        res_params.residual = strict_copy_consumer ? nullptr : residual;
        res_params.output = result_output;
        res_params.num_elements = num_elements;
        res_params.device_id = device;
        res_params.input_buffer_id = BufferId::NORMALIZED;
        if (!strict_copy_consumer)
            res_params.residual_buffer_id = BufferId::RESIDUAL;
        res_params.output_buffer_id = BufferId::ATTN_PROJ;

        /*
         * Bind the same typed arena identities used by production graphs. The
         * fixture owns the tensors; BufferArena owns coherence and ordering.
         * Rebinding is intentional because several tests build a fresh graph
         * with the same logical BufferIds after completing an earlier launch.
         */
        EXPECT_TRUE(arena_.bindExternalBuffer(BufferId::HIDDEN_STATE, norm_input));
        EXPECT_TRUE(arena_.bindExternalBuffer(BufferId::NORMALIZED, norm_output));
        EXPECT_TRUE(arena_.bindExternalBuffer(BufferId::RESIDUAL, residual));
        EXPECT_TRUE(arena_.bindExternalBuffer(BufferId::ATTN_PROJ, result_output));

        // Assemble graph with dependency
        ComputeGraph graph;
        graph.addNode("rmsnorm", ComputeStageFactory::createRMSNorm(norm_params), device);
        graph.addNode("residual_add", ComputeStageFactory::createResidualAdd(res_params), device);
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
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);

    // Execute via graph capture path
    bool success = executeCapturedGraph(graph, executor, result);
    ASSERT_TRUE(success) << "executeWithGraphCapture failed on first invocation";
    expectExecutableGraph();

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

TEST_F(GPUGraphCaptureExecutionTest,
       InternalHostAuthoritativeIntermediateIsNotAnExternalCaptureInput)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    FP32Tensor *internal_norm_output = nullptr;
    auto graph = buildNormResidualGraph(
        /*seq_len=*/2,
        /*d_model=*/32,
        norm_input,
        residual,
        result,
        &internal_norm_output,
        /*strict_copy_consumer=*/true);
    ASSERT_NE(internal_norm_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());
    ASSERT_NE(internal_norm_output->gpu_data_ptr(), nullptr);

    /*
     * Reproduce the lazy MTP sidecar state exactly: arena storage has a stable
     * device address, but its value is not valid until the first captured stage
     * produces it. Preflight must not demand pre-existing device authority for
     * this internal edge.
     */
    internal_norm_output->mutable_data()[0] = -123.0f;
    ASSERT_EQ(
        internal_norm_output->coherenceState(),
        TensorCoherenceState::HOST_AUTHORITATIVE);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    const float *output = result->data();
    for (size_t index = 0; index < result->numel(); ++index)
    {
        EXPECT_TRUE(std::isfinite(output[index]))
            << "Captured producer/consumer chain emitted invalid output at " << index;
    }
}

TEST_F(GPUGraphCaptureExecutionTest,
       ColdInternalOutputStorageIsAllocatedBeforeCapture)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    FP32Tensor *internal_norm_output = nullptr;
    auto graph = buildNormResidualGraph(
        /*seq_len=*/2,
        /*d_model=*/32,
        norm_input,
        residual,
        result,
        &internal_norm_output,
        /*strict_copy_consumer=*/true);
    ASSERT_NE(internal_norm_output, nullptr);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution(internal_norm_output));
    ASSERT_EQ(internal_norm_output->gpu_data_ptr(), nullptr)
        << "The regression requires a genuinely cold internal output";

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    ASSERT_NE(internal_norm_output->gpu_data_ptr(), nullptr);
    ASSERT_EQ(
        internal_norm_output->current_device(),
        std::optional<DeviceId>{device_ctx_->deviceId()});
    const float *output = result->data();
    for (size_t index = 0; index < result->numel(); ++index)
    {
        EXPECT_TRUE(std::isfinite(output[index]))
            << "Cold-output capture emitted invalid output at " << index;
    }
}

// ===========================================================================
// 2. Re-capture + backend-specific executable publication
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, SecondExecution_UsesBackendPublicationPath)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
    EXPECT_TRUE(capture_->supportsExecutableUpdate())
        << "CUDA must exercise the in-place executable-update lifecycle";
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    EXPECT_FALSE(capture_->supportsExecutableUpdate())
        << "ROCm must replace the executable directly after recapture";
#endif

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution owns a real graph and instantiates it.
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    // Save first-run output
    std::vector<float> first_output(seq_len * d_model);
    std::memcpy(first_output.data(), result->data(), first_output.size() * sizeof(float));

    // Reset graph completion flags for re-execution
    graph.reset();

    // Second execution captures the same topology and takes the update path.
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

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
        << "Second execution output diverged from first after executable publication";
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
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    std::vector<float> reference_output(seq_len * d_model);
    std::memcpy(reference_output.data(), result->data(), reference_output.size() * sizeof(float));

    // Re-execute multiple times
    for (int iter = 1; iter < num_iterations; ++iter)
    {
        graph.reset();
        ASSERT_TRUE(executeCapturedGraph(graph, executor, result))
            << "Iteration " << iter << " failed";
        expectExecutableGraph();

        const float *out = result->data();
        for (size_t i = 0; i < seq_len * d_model; ++i)
        {
            EXPECT_NEAR(out[i], reference_output[i], 1e-5f)
                << "Divergence at index " << i << " on iteration " << iter;
        }
    }
}

// ===========================================================================
// 4. Hard failure: a capture policy requires a capture owner
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, NullCaptureFailsClosed)
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

    EXPECT_FALSE(
        executor.executeWithGraphCapture(graph, device_ctx_.get(), nullptr, {}))
        << "A selected graph-capture path must not silently change execution modes";
}

// ===========================================================================
// 5. Hard failure: graph outputs require an explicit publication contract
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, MissingOutputPublicationFailsBeforeCapture)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(
        2,
        32,
        norm_input,
        residual,
        result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    EXPECT_FALSE(executor.executeWithGraphCapture(
        graph,
        device_ctx_.get(),
        capture_.get(),
        {}));
    EXPECT_FALSE(capture_->hasExecutable())
        << "An incomplete graph output contract must fail before capture begins";
    EXPECT_EQ(capture_->nodeCount(), 0u);
}

// ===========================================================================
// 6. Hard failure: this legacy single-device API cannot own collectives
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, CollectiveNodesPresentFailClosed)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;

    auto graph = buildNormResidualGraph(
        seq_len,
        d_model,
        norm_input,
        residual,
        result);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // This API is deliberately single-device. Fully captured collective graphs
    // use the cached graph controller and may not be rewritten as eager work.
    std::unordered_set<std::string> collective_nodes = {"fake_allreduce"};

    EXPECT_FALSE(executor.executeWithGraphCapture(
        graph, device_ctx_.get(), capture_.get(), {}, &collective_nodes));

    EXPECT_FALSE(capture_->hasExecutable())
        << "The unsupported single-device collective graph must not begin capture";
}

// ===========================================================================
// 7. Empty collective set does NOT trigger fallback
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
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);

    // Empty set still requires one complete captured graph.
    std::unordered_set<std::string> empty_collectives;

    const std::array<ITensor *, 1> outputs = {result};
    bool success = executor.executeWithGraphCapture(
        graph,
        device_ctx_.get(),
        capture_.get(),
        outputs,
        &empty_collectives);
    ASSERT_TRUE(success);
    expectExecutableGraph();

    // Verify the captured launch produced output as a second, independent
    // correctness assertion.
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
// 8. Retained composition records a child graph without launching it
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest,
       RetainedGraphFragmentIsRecordedButNotInstantiated)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(
        2,
        32,
        norm_input,
        residual,
        result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);

    ASSERT_TRUE(executor.captureRetainedGraphFragment(
        graph,
        device_ctx_.get(),
        capture_.get(),
        "integration_retained_child"));
    EXPECT_GT(capture_->nodeCount(), 0u);
    EXPECT_FALSE(capture_->hasExecutable())
        << "A retained child must remain an uninstantiated native graph for its parent to import";

    graph.reset();
    EXPECT_FALSE(executor.captureRetainedGraphFragment(
        graph,
        device_ctx_.get(),
        capture_.get(),
        "integration_reused_retained_child"))
        << "Retained composition must reject an owner that already contains graph topology";
}

// ===========================================================================
// 9. Graph capture reset and re-capture
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
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);

    // First execution
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    // Reset capture object
    gpu_ctx_->submitAndWait([&]
                            { capture_->reset(); });
    EXPECT_FALSE(capture_->hasExecutable());

    // Re-capture from scratch
    graph.reset();
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();

    // Verify output is still valid
    const float *out = result->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i << " after re-capture";
    }
}

// ===========================================================================
// 9. Single-stage graph (minimal capture)
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
    params.device_id = device_ctx_->deviceId();
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::NORMALIZED;

    ASSERT_TRUE(arena_.bindExternalBuffer(BufferId::HIDDEN_STATE, input));
    ASSERT_TRUE(arena_.bindExternalBuffer(BufferId::NORMALIZED, output));

    ComputeGraph graph;
    graph.addNode("solo_rmsnorm", ComputeStageFactory::createRMSNorm(params), device_ctx_->deviceId());

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);

    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());
    ASSERT_TRUE(executeCapturedGraph(graph, executor, output));
    expectExecutableGraph();

    // Verify output
    const float *out = output->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        ASSERT_FALSE(std::isnan(out[i])) << "NaN at index " << i;
        ASSERT_FALSE(std::isinf(out[i])) << "Inf at index " << i;
    }
}

// ===========================================================================
// 10. Correctness: graph capture output matches fast decode output
// ===========================================================================

TEST_F(GPUGraphCaptureExecutionTest, OutputMatchesFastDecode)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    const size_t seq_len = 4;
    const size_t d_model = 64;
    const size_t num_elements = seq_len * d_model;

    // --- Run 1: via explicitly selected eager execution ---
    FP32Tensor *norm_input1 = nullptr;
    FP32Tensor *residual1 = nullptr;
    FP32Tensor *result1 = nullptr;
    auto graph1 = buildNormResidualGraph(seq_len, d_model, norm_input1, residual1, result1);

    GraphExecutorConfig config;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);

    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());
    ASSERT_TRUE(executor.executeFastDecode(graph1, device_ctx_.get()));
    std::vector<float> fast_decode_output(num_elements);
    std::memcpy(fast_decode_output.data(), result1->data(), num_elements * sizeof(float));

    // --- Run 2: via graph capture ---
    // We need identical inputs, so build a fresh graph with the same values.
    FP32Tensor *norm_input2 = nullptr;
    FP32Tensor *residual2 = nullptr;
    FP32Tensor *result2 = nullptr;
    auto graph2 = buildNormResidualGraph(seq_len, d_model, norm_input2, residual2, result2);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    ASSERT_TRUE(executeCapturedGraph(graph2, executor, result2));
    expectExecutableGraph();

    // Compare outputs
    const float *graph_output = result2->data();
    for (size_t i = 0; i < num_elements; ++i)
    {
        EXPECT_NEAR(graph_output[i], fast_decode_output[i], 1e-5f)
            << "Graph capture output differs from fast decode at index " << i;
    }
}

/**
 * @brief One VRAM arena retains every graph-captured diagnostic checkpoint.
 *
 * A parity graph may publish hundreds of stage outputs. Captured writes to
 * mapped host memory multiply PCIe packets and completion traffic; on ROCm a
 * production-sized manifest can overflow the interrupt ring and degrade later
 * RCCL capture. This regression proves that descriptor preparation performs one
 * exact device allocation, reuses it without pointer drift, and that one bulk
 * download still publishes byte-exact stage values.
 */
TEST_F(GPUGraphCaptureExecutionTest,
       SnapshotManifestUsesOneArenaAndPublishesCapturedBytes)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    constexpr size_t seq_len = 4;
    constexpr size_t d_model = 64;
    constexpr size_t element_count = seq_len * d_model;
    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(
        seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    std::unordered_map<std::string, std::vector<float>> captured_outputs;
    GraphExecutorConfig config;
    config.snapshot_callback =
        [&](const std::string &stage_name, const StageDumpInfo &dump_info)
        {
            ASSERT_FALSE(dump_info.outputs.empty());
            const auto &output = dump_info.outputs.front();
            ASSERT_NE(output.data, nullptr);
            ASSERT_STREQ(output.dtype, "FP32");
            const auto *values = static_cast<const float *>(output.data);
            captured_outputs[stage_name] = std::vector<float>(
                values, values + output.rows * output.cols);
        };

    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);
    void *const stream = capture_->executionStream();
    ASSERT_NE(stream, nullptr);

    DeviceGraphExecutor::GraphSnapshotManifest layout_manifest;
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        graph,
        device_ctx_.get(),
        stream,
        "snapshot_arena_layout_regression",
        &layout_manifest));
    ASSERT_TRUE(layout_manifest.storageBound());
    ASSERT_NE(layout_manifest.storage_arena, nullptr);
    EXPECT_FALSE(layout_manifest.storage_arena->isMapped());
    EXPECT_EQ(layout_manifest.storage_allocation_count, 1u);
    EXPECT_EQ(layout_manifest.bound_slot_count, 2u);
    EXPECT_GE(
        layout_manifest.storage_capacity_bytes,
        layout_manifest.storage_required_bytes);

    std::vector<std::pair<size_t, size_t>> ranges;
    for (const auto &[stage_name, stage_copies] :
         layout_manifest.stage_copies)
    {
        SCOPED_TRACE(stage_name);
        ASSERT_EQ(stage_copies.outputs.size(), 1u);
        const auto &copy = stage_copies.outputs.front();
        ASSERT_GT(copy.byte_size, 0u);
        ASSERT_GE(copy.storage_bytes, copy.byte_size);
        ASSERT_LE(
            copy.storage_offset_bytes + copy.storage_bytes,
            layout_manifest.storage_capacity_bytes);
        ranges.emplace_back(
            copy.storage_offset_bytes,
            copy.storage_offset_bytes + copy.storage_bytes);
    }
    ASSERT_EQ(ranges.size(), 2u);
    std::sort(ranges.begin(), ranges.end());
    EXPECT_LE(ranges[0].second, ranges[1].first)
        << "Distinct checkpoints must own disjoint arena subranges";

    void *const original_arena_pointer =
        layout_manifest.storage_arena->gpu_data_ptr();
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        graph,
        device_ctx_.get(),
        stream,
        "snapshot_arena_identity_reuse",
        &layout_manifest));
    EXPECT_EQ(layout_manifest.storage_allocation_count, 1u);
    EXPECT_EQ(
        layout_manifest.storage_arena->gpu_data_ptr(),
        original_arena_pointer)
        << "An unchanged capture identity must reuse its device arena";

    /*
     * The direct capture API owns the executor's transient manifest. Prepare
     * that manifest before beginCapture(), then publish it through the same
     * post-launch callback path used by cached production graphs.
     */
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        graph,
        device_ctx_.get(),
        stream,
        "snapshot_arena_capture_regression"));
    ASSERT_TRUE(executeCapturedGraph(graph, executor, result));
    expectExecutableGraph();
    ASSERT_TRUE(executor.publishSnapshotsAfterGraphExecution(
        graph,
        stream,
        "snapshot_arena_capture_regression"));

    ASSERT_EQ(captured_outputs.size(), 2u);
    ASSERT_EQ(captured_outputs["rmsnorm"].size(), element_count);
    ASSERT_EQ(captured_outputs["residual_add"].size(), element_count);

    const float *const final_values = result->data();
    EXPECT_EQ(
        std::memcmp(
            captured_outputs["residual_add"].data(),
            final_values,
            element_count * sizeof(float)),
        0)
        << "The final captured checkpoint must be byte-identical to its graph output";
}

/**
 * @brief Production snapshot allocation is bounded by one admitted GPU owner.
 *
 * This is intentionally exercised by both backend-specialized binaries. The
 * executor must claim the arena before calling CUDA/HIP allocation, publish
 * the exact materialized byte count, and fail closed when production mode has
 * no authority instead of discovering exhaustion in the backend allocator.
 */
TEST_F(GPUGraphCaptureExecutionTest,
       SnapshotManifestMaterializesThroughPhysicalMemoryAuthority)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    constexpr size_t seq_len = 4;
    constexpr size_t d_model = 64;
    FP32Tensor *norm_input = nullptr;
    FP32Tensor *residual = nullptr;
    FP32Tensor *result = nullptr;
    auto graph = buildNormResidualGraph(
        seq_len, d_model, norm_input, residual, result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    constexpr size_t kSnapshotEnvelopeBytes = 1024u * 1024u;
    const DeviceId device = LINKED_GPU_DEVICE_ID();
    PhysicalMemoryPlanBuilder plan_builder;
    plan_builder.add(
        PhysicalMemoryResource{
            .world_rank = 0,
            .device = device,
            .total_bytes = 2u * kSnapshotEnvelopeBytes,
            .admission_available_bytes = 2u * kSnapshotEnvelopeBytes,
        },
        PhysicalMemoryOwner::GraphSnapshotArena,
        kSnapshotEnvelopeBytes);
    const auto admission = std::make_shared<
        const PhysicalMemoryPlanAdmissionCertificate>(
        plan_builder.build());
    PhysicalMemoryAuthority authority(admission, 0);
    auto reservation = std::make_shared<PhysicalMemoryOwnerReservation>(
        authority.reserveNewAllocations(
            device,
            PhysicalMemoryOwner::GraphSnapshotArena,
            kSnapshotEnvelopeBytes));

    GraphExecutorConfig config;
    config.snapshot_callback =
        [](const std::string &, const StageDumpInfo &) {};
    config.require_snapshot_memory_authority = true;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);
    executor.setGraphSnapshotMemoryReservation(reservation);

    DeviceGraphExecutor::GraphSnapshotManifest manifest;
    void *const stream = capture_->executionStream();
    ASSERT_NE(stream, nullptr);
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        graph,
        device_ctx_.get(),
        stream,
        "snapshot_physical_authority_regression",
        &manifest));
    ASSERT_NE(manifest.storage_arena, nullptr);
    ASSERT_NE(manifest.storage_allocation_lease, nullptr);
    EXPECT_TRUE(manifest.storage_allocation_lease->valid());
    EXPECT_EQ(
        manifest.storage_allocation_lease->bytes(),
        manifest.storage_capacity_bytes);
    EXPECT_EQ(
        authority.claimedBytes(
            device,
            PhysicalMemoryOwner::GraphSnapshotArena,
            PhysicalMemoryMaterializationKind::NewAllocation),
        manifest.storage_capacity_bytes);
    EXPECT_EQ(
        reservation->remainingBytes(),
        kSnapshotEnvelopeBytes - manifest.storage_capacity_bytes);

    DeviceGraphExecutor missing_authority_executor(config);
    missing_authority_executor.setArena(&arena_);
    DeviceGraphExecutor::GraphSnapshotManifest rejected_manifest;
    EXPECT_FALSE(missing_authority_executor.prepareSnapshotsForGraphCapture(
        graph,
        device_ctx_.get(),
        stream,
        "snapshot_missing_physical_authority_regression",
        &rejected_manifest));
    EXPECT_EQ(rejected_manifest.storage_arena, nullptr);
    EXPECT_EQ(rejected_manifest.storage_allocation_lease, nullptr);
}

/**
 * @brief Mutually exclusive graph roles reuse only their typed device pool.
 *
 * Production serving setup retains several prefill buckets and two verifier
 * outcome graphs. Capturing each checkpoint manifest into an independent VRAM
 * tensor needlessly multiplies diagnostic storage. This regression proves that
 * a largest-first alternative family aliases one stable address, a concurrent
 * MTP role remains disjoint, and an undersized frozen pool fails instead of
 * reallocating behind an already captured pointer.
 */
TEST_F(GPUGraphCaptureExecutionTest,
       SnapshotAlternativeArenasReuseFrozenTypedCapacity)
{
    SKIP_IF_NO_GPU();
    ASSERT_NE(capture_, nullptr);

    FP32Tensor *large_input = nullptr;
    FP32Tensor *large_residual = nullptr;
    FP32Tensor *large_result = nullptr;
    auto large_graph = buildNormResidualGraph(
        /*seq_len=*/8,
        /*d_model=*/64,
        large_input,
        large_residual,
        large_result);

    FP32Tensor *small_input = nullptr;
    FP32Tensor *small_residual = nullptr;
    FP32Tensor *small_result = nullptr;
    auto small_graph = buildNormResidualGraph(
        /*seq_len=*/4,
        /*d_model=*/64,
        small_input,
        small_residual,
        small_result);
    ASSERT_TRUE(prepareFixtureTensorsForGPUExecution());

    GraphExecutorConfig config;
    config.snapshot_callback =
        [](const std::string &, const StageDumpInfo &) {};
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena_);
    void *const stream = capture_->executionStream();
    ASSERT_NE(stream, nullptr);

    using ReuseClass =
        DeviceGraphExecutor::GraphSnapshotArenaReuseClass;
    using ReusePolicy =
        DeviceGraphExecutor::GraphSnapshotArenaReusePolicy;
    const ReusePolicy prefill_policy{
        .reuse_class =
            ReuseClass::PrefillOrMTPVerifierAlternative,
        .configuration_identity = UINT64_C(91),
    };

    DeviceGraphExecutor::GraphSnapshotManifest large_manifest;
    large_manifest.bindStorageReusePolicy(prefill_policy);
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        large_graph,
        device_ctx_.get(),
        stream,
        "largest_prefill_alternative",
        &large_manifest));
    ASSERT_NE(large_manifest.storage_arena, nullptr);
    EXPECT_EQ(large_manifest.storage_allocation_count, 1u);
    void *const shared_prefill_address =
        large_manifest.storage_arena->gpu_data_ptr();
    const size_t shared_prefill_capacity =
        large_manifest.storage_capacity_bytes;

    DeviceGraphExecutor::GraphSnapshotManifest small_manifest;
    small_manifest.bindStorageReusePolicy(prefill_policy);
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        small_graph,
        device_ctx_.get(),
        stream,
        "smaller_prefill_alternative",
        &small_manifest));
    ASSERT_NE(small_manifest.storage_arena, nullptr);
    EXPECT_EQ(small_manifest.storage_allocation_count, 0u)
        << "The second alternative must bind, not allocate";
    EXPECT_EQ(
        small_manifest.storage_arena->gpu_data_ptr(),
        shared_prefill_address);
    EXPECT_EQ(
        small_manifest.storage_capacity_bytes,
        shared_prefill_capacity);
    EXPECT_LT(
        small_manifest.storage_required_bytes,
        small_manifest.storage_capacity_bytes);

    DeviceGraphExecutor::GraphSnapshotManifest verifier_manifest;
    verifier_manifest.bindStorageReusePolicy(ReusePolicy{
        .reuse_class =
            ReuseClass::PrefillOrMTPVerifierAlternative,
        .configuration_identity = UINT64_C(91),
    });
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        small_graph,
        device_ctx_.get(),
        stream,
        "grouped_verifier_after_published_prefill",
        &verifier_manifest));
    ASSERT_NE(verifier_manifest.storage_arena, nullptr);
    EXPECT_EQ(verifier_manifest.storage_allocation_count, 0u);
    EXPECT_EQ(
        verifier_manifest.storage_arena->gpu_data_ptr(),
        shared_prefill_address)
        << "Prefill publication precedes grouped verification, so their wide "
           "checkpoint graphs must reuse one frozen lane";

    DeviceGraphExecutor::GraphSnapshotManifest condition_manifest;
    condition_manifest.bindStorageReusePolicy(ReusePolicy{
        .reuse_class = ReuseClass::MTPConditionAlternative,
        .configuration_identity = UINT64_C(91),
    });
    ASSERT_TRUE(executor.prepareSnapshotsForGraphCapture(
        small_graph,
        device_ctx_.get(),
        stream,
        "concurrent_mtp_condition_role",
        &condition_manifest));
    ASSERT_NE(condition_manifest.storage_arena, nullptr);
    EXPECT_NE(
        condition_manifest.storage_arena->gpu_data_ptr(),
        shared_prefill_address)
        << "MTP condition and verifier snapshots may coexist and cannot alias";

    DeviceGraphExecutor undersized_executor(config);
    undersized_executor.setArena(&arena_);
    const ReusePolicy undersized_policy{
        .reuse_class =
            ReuseClass::PrefillOrMTPVerifierAlternative,
        .configuration_identity = UINT64_C(92),
    };
    DeviceGraphExecutor::GraphSnapshotManifest undersized_manifest;
    undersized_manifest.bindStorageReusePolicy(undersized_policy);
    ASSERT_TRUE(undersized_executor.prepareSnapshotsForGraphCapture(
        small_graph,
        device_ctx_.get(),
        stream,
        "undersized_verifier_pool",
        &undersized_manifest));
    const void *const frozen_address =
        undersized_manifest.storage_arena->gpu_data_ptr();

    DeviceGraphExecutor::GraphSnapshotManifest oversized_manifest;
    oversized_manifest.bindStorageReusePolicy(undersized_policy);
    EXPECT_FALSE(undersized_executor.prepareSnapshotsForGraphCapture(
        large_graph,
        device_ctx_.get(),
        stream,
        "illegal_verifier_pool_growth",
        &oversized_manifest));
    EXPECT_EQ(
        undersized_manifest.storage_arena->gpu_data_ptr(),
        frozen_address)
        << "Rejecting a larger alternative must preserve every captured address";
}
