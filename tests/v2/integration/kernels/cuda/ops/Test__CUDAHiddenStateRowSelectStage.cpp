/**
 * @file Test__CUDAHiddenStateRowSelectStage.cpp
 * @brief CUDA integration tests for graph-capturable hidden-state row selection.
 *
 * Covers dynamic stage-owned rows, immutable device-only checkpoint rows,
 * external verifier metadata, and request-terminal rows derived directly from
 * device-resident unequal request lengths. Captured graph replays prove each
 * ownership policy executes without host work inside capture.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "kernels/cuda/ops/CUDARowSelectKernels.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

using namespace llaminar2;

namespace
{
#ifdef HAVE_CUDA
    /**
     * @brief Simulate a preceding library stage that temporarily binds another GPU.
     *
     * This is the CUDA twin of the HIP handoff regression. The row-selection
     * stage must not rely on whichever device a preceding collective or helper
     * happened to leave current on the host thread.
     */
    class ForeignCUDADeviceStage final : public IComputeStage
    {
    public:
        ForeignCUDADeviceStage(DeviceId graph_device, int foreign_device)
            : IComputeStage(graph_device),
              foreign_device_(foreign_device)
        {
        }

        bool execute(IDeviceContext *) override
        {
            return cudaSetDevice(foreign_device_) == cudaSuccess;
        }

        ComputeStageType type() const override
        {
            return ComputeStageType::COPY;
        }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::GPU_CUDA;
        }

        CoherencePolicy coherencePolicy() const override
        {
            return CoherencePolicy::NONE;
        }

    private:
        StageDumpInfo buildDumpInfoImpl() const override
        {
            return {};
        }

        int foreign_device_ = -1;
    };
#endif

    /// @brief Fill hidden rows with deterministic values that identify the row.
    std::unique_ptr<FP32Tensor> makeHiddenStates(int seq_len, int d_model, DeviceId device, void *stream)
    {
        auto hidden = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model)},
            DeviceId::cpu());
        float *hidden_data = hidden->mutable_data();
        for (int row = 0; row < seq_len; ++row)
        {
            for (int column = 0; column < d_model; ++column)
            {
                hidden_data[static_cast<size_t>(row) * d_model + column] =
                    10.0f * static_cast<float>(row + 1) + 0.125f * static_cast<float>(column);
            }
        }
        hidden->ensureOnDevice(device, stream);
        return hidden;
    }

#ifdef HAVE_CUDA
    /// @brief Copy scratch row from CUDA device memory for assertion.
    std::vector<float> downloadScratchRow(FP32Tensor &scratch, int d_model, cudaStream_t stream)
    {
        std::vector<float> row(static_cast<size_t>(d_model), 0.0f);
        EXPECT_EQ(cudaMemcpyAsync(
                      row.data(),
                      scratch.gpu_data_ptr(),
                      static_cast<size_t>(d_model) * sizeof(float),
                      cudaMemcpyDeviceToHost,
                      stream),
                  cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return row;
    }

    /// @brief Copy compact scratch rows from CUDA device memory for assertion.
    std::vector<float> downloadScratchRows(FP32Tensor &scratch, int row_count, int d_model, cudaStream_t stream)
    {
        std::vector<float> rows(static_cast<size_t>(row_count) * static_cast<size_t>(d_model), 0.0f);
        cudaMemcpyAsync(
            rows.data(),
            scratch.gpu_data_ptr(),
            rows.size() * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream);
        cudaStreamSynchronize(stream);
        return rows;
    }

    /// @brief Assert that a downloaded row equals the selected source row.
    void expectRow(const std::vector<float> &row, const FP32Tensor &hidden, int selected_row, int d_model)
    {
        const float *hidden_data = hidden.data();
        for (int column = 0; column < d_model; ++column)
        {
            EXPECT_FLOAT_EQ(row[static_cast<size_t>(column)],
                            hidden_data[static_cast<size_t>(selected_row) * d_model + column])
                << "column=" << column;
        }
    }

    /// @brief Assert that compact rows equal the requested source rows in order.
    void expectRows(const std::vector<float> &rows, const FP32Tensor &hidden, const std::vector<int> &selected_rows, int d_model)
    {
        const float *hidden_data = hidden.data();
        for (size_t output_row = 0; output_row < selected_rows.size(); ++output_row)
        {
            for (int column = 0; column < d_model; ++column)
            {
                EXPECT_FLOAT_EQ(
                    rows[output_row * static_cast<size_t>(d_model) + static_cast<size_t>(column)],
                    hidden_data[static_cast<size_t>(selected_rows[output_row]) * static_cast<size_t>(d_model) + static_cast<size_t>(column)])
                    << "output_row=" << output_row << " column=" << column;
            }
        }
    }
#endif
}

/**
 * @brief Prove low-level CUDA row helpers cannot adopt the default stream.
 *
 * All non-stream arguments are valid device-backed storage, so a missing
 * stream guard would enqueue real work on stream zero and make these calls
 * succeed. Rejection therefore tests the API contract rather than an unrelated
 * invalid-pointer path.
 */
TEST(Test__CUDAHiddenStateRowSelectStage,
     LowLevelLaunchContractsRejectNullStream)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    cudaStream_t setup_stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(
                  &setup_stream,
                  cudaStreamNonBlocking),
              cudaSuccess);
    FP32Tensor values({1, 1}, DeviceId::cpu());
    INT32Tensor selected_row(std::vector<size_t>{1});
    ASSERT_TRUE(values.allocateOnDevice(DeviceId::cuda(0), setup_stream));
    ASSERT_TRUE(selected_row.allocateOnDevice(
        DeviceId::cuda(0),
        setup_stream));

    int host_selected_row = 0;
    auto *device_value = static_cast<float *>(values.gpu_data_ptr());
    auto *device_selected_row =
        static_cast<int *>(selected_row.gpu_data_ptr());
    EXPECT_FALSE(llaminar2::cuda::uploadRowSelectParam(
        device_selected_row,
        &host_selected_row,
        nullptr));
    EXPECT_FALSE(llaminar2::cuda::launchRowSelectFP32(
        device_value,
        device_value,
        device_selected_row,
        /*seq_len=*/1,
        /*d_model=*/1,
        nullptr));
    EXPECT_FALSE(llaminar2::cuda::launchMTPConcatFP32(
        device_value,
        device_value,
        device_value,
        /*rows=*/1,
        /*hidden_dim=*/1,
        nullptr));

    ASSERT_EQ(cudaStreamSynchronize(setup_stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(setup_stream), cudaSuccess);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage, CapturedGraphReplayUsesUpdatedSelectedRow)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const DeviceId device = DeviceId::cuda(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    HiddenStateRowSelectStage stage(params);
    DeviceWorkspaceManager workspace(device, 1024);
    ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(bucket_seq_len, d_model, 0)));
    stage.bindWorkspace(&workspace);

    stage.setGPUStream(stream);

    // Warmup performs scalar allocation before capture and proves the stage path works.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{2, bucket_seq_len, 0});
    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 1, d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 1, d_model);

    // No recapture: update host intent, then let the graph-launch preparation
    // hook upload the scalar on the same explicit stream used for graph launch.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{6, bucket_seq_len, 0});
    ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, stream));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 5, d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
#endif
}

/**
 * @brief Prove fixed diagnostic rows capture without scalar workspace or H2D.
 */
TEST(Test__CUDAHiddenStateRowSelectStage, CapturedFixedDeviceRowUsesDeviceOnlyCopy)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    const int seq_len = 8;
    const int d_model = 32;
    const int selected_row = 5;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.selected_row_idx = selected_row;
    params.selection_policy =
        HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow;
    HiddenStateRowSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.getWorkspaceRequirements(seq_len, d_model, 0).buffers.empty());
    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        selected_row,
        d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        selected_row,
        d_model);

    EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

/**
 * @brief Reproduce padded M=512 checkpoint selection from resident metadata.
 *
 * A single captured executable first observes a 443-row request and then a
 * 257-row request. Only the persistent device length changes between launches;
 * no stage replay setter or graph-launch preparation hook participates.
 */
TEST(Test__CUDAHiddenStateRowSelectStage, CapturedCheckpointReadsResidentPrefillLength)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int bucket_seq_len = 512;
    constexpr int d_model = 32;
    int32_t initial_length = 443;
    int32_t replay_length = 257;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto hidden = makeHiddenStates(
        bucket_seq_len,
        d_model,
        device,
        stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *length_device = nullptr;
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&length_device),
            sizeof(int32_t)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            length_device,
            &initial_length,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_idx = bucket_seq_len - 1;
    params.selection_policy =
        HiddenStateRowSelectStage::SelectionPolicy::
            DeviceResidentRequestLength;
    params.request_sequence_length_device = length_device;
    HiddenStateRowSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(
        stage.getWorkspaceRequirements(
                 bucket_seq_len,
                 d_model,
                 0)
            .buffers.empty());
    ASSERT_EQ(
        stage.graphLaunchPreparationPolicy(),
        GraphLaunchPreparationPolicy::None);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(
            &graph_exec,
            graph,
            nullptr,
            nullptr,
            0),
        cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        /*selected_row=*/442,
        d_model);

    ASSERT_EQ(
        cudaMemcpyAsync(
            length_device,
            &replay_length,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        /*selected_row=*/256,
        d_model);

    EXPECT_EQ(cudaGraphExecDestroy(graph_exec), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    EXPECT_EQ(cudaFree(length_device), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
#endif
}

/**
 * @brief Reproduce the first production M=39 mirrored checkpoint through the executor.
 *
 * This is the CUDA twin of the ROCm long-context regression. It exercises the
 * exact 39x2048 graph-managed checkpoint geometry and delegates stream binding
 * to DeviceGraphExecutor, proving that both GPU backends enforce the same
 * explicit-stream and device-resident-length contract.
 */
TEST(Test__CUDAHiddenStateRowSelectStage, ExecutorRunsProductionM39ResidentCheckpoint)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int seq_len = 39;
    constexpr int d_model = 2048;
    const int32_t request_length = seq_len;

    cudaStream_t setup_stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&setup_stream), cudaSuccess);
    auto hidden = makeHiddenStates(seq_len, d_model, device, setup_stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());

    int32_t *length_device = nullptr;
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&length_device),
            sizeof(int32_t)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            length_device,
            &request_length,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            setup_stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(setup_stream), cudaSuccess);

    BufferArena arena;
    ASSERT_TRUE(
        arena.registerExternalBuffer(
            BufferId::HIDDEN_STATE,
            hidden.get()));
    ASSERT_TRUE(
        arena.registerExternalBuffer(
            BufferId::PREFIX_TERMINAL_HIDDEN,
            scratch.get()));

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.selected_row_idx = seq_len - 1;
    params.selection_policy =
        HiddenStateRowSelectStage::SelectionPolicy::
            DeviceResidentRequestLength;
    params.request_sequence_length_device = length_device;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;

    auto stage = std::make_unique<HiddenStateRowSelectStage>(params);
    auto *stage_ptr = stage.get();
    ComputeGraph graph;
    if (device_count > 1)
    {
        graph.addNode(
            "foreign_cuda_device_binding",
            std::make_unique<ForeignCUDADeviceStage>(device, 1),
            device);
    }
    graph.addNode("production_m39_resident_checkpoint", std::move(stage), device);
    if (device_count > 1)
    {
        graph.addDependency(
            "production_m39_resident_checkpoint",
            "foreign_cuda_device_binding");
    }

    GraphExecutorConfig config;
    config.enable_profiling = false;
    config.enable_validation = false;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena);
    auto ctx = IDeviceContext::create(device, 1);
    ASSERT_NE(ctx, nullptr);

    ASSERT_TRUE(executor.execute(graph, ctx.get()));
    ASSERT_NE(stage_ptr->gpuStream(), nullptr);
    int active_device = -1;
    ASSERT_EQ(cudaGetDevice(&active_device), cudaSuccess);
    EXPECT_EQ(active_device, 0)
        << "The checkpoint handoff must restore its stream-owning device";
    auto execution_stream =
        static_cast<cudaStream_t>(stage_ptr->gpuStream());
    ASSERT_EQ(cudaStreamSynchronize(execution_stream), cudaSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, execution_stream),
        *hidden,
        /*selected_row=*/seq_len - 1,
        d_model);

    EXPECT_EQ(cudaFree(length_device), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(setup_stream), cudaSuccess);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage, GraphManagedExecutionOwnsCoherenceThroughArena)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const DeviceId device = DeviceId::cuda(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());

    BufferArena arena;
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::HIDDEN_STATE, hidden.get()));
    ASSERT_TRUE(arena.registerExternalBuffer(BufferId::PREFIX_TERMINAL_HIDDEN, scratch.get()));

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_idx = 1;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::PREFIX_TERMINAL_HIDDEN;
    params.workspace_buffer_name = std::string(HiddenStateRowSelectStage::WS_SELECTED_ROW_SCALAR) +
                                   "_cuda_graph_managed_regression";

    auto stage = std::make_unique<HiddenStateRowSelectStage>(params);
    auto *stage_ptr = stage.get();

    DeviceWorkspaceManager workspace(device, 1024);
    ASSERT_TRUE(workspace.allocate(stage_ptr->getWorkspaceRequirements(bucket_seq_len, d_model, 0)));
    stage_ptr->bindWorkspace(&workspace);

    ComputeGraph graph;
    graph.addNode("row_select", std::move(stage), device);

    GraphExecutorConfig config;
    config.enable_profiling = false;
    config.enable_validation = false;
    DeviceGraphExecutor executor(config);
    executor.setArena(&arena);
    auto ctx = IDeviceContext::create(device, 1);
    ASSERT_NE(ctx, nullptr);

    stage_ptr->setSelectedRowForReplay(1);
    ASSERT_TRUE(executor.execute(graph, ctx.get()));
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stage_ptr->gpuStream())), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 1, d_model);

    stage_ptr->setSelectedRowForReplay(5);
    ASSERT_TRUE(executor.execute(graph, ctx.get()));
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stage_ptr->gpuStream())), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 5, d_model);

    ASSERT_TRUE(scratch->ensureOnHost(stage_ptr->gpuStream()))
        << "Graph-managed row-select output must be readable after executor-owned coherence";
    expectRow(std::vector<float>(
                  scratch->data(),
                  scratch->data() + static_cast<size_t>(d_model)),
              *hidden,
              5,
              d_model);

    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage, CapturedGraphReplayUsesUpdatedSelectedRows)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const DeviceId device = DeviceId::cuda(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;
    const std::vector<int> initial_rows{1, 3, 6};
    const std::vector<int> replay_rows{7, 0, 4};

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{initial_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_count = static_cast<int>(initial_rows.size());
    params.selected_row_indices = initial_rows;
    HiddenStateRowsSelectStage stage(params);
    DeviceWorkspaceManager workspace(device, 1024);
    ASSERT_TRUE(workspace.allocate(stage.getWorkspaceRequirements(bucket_seq_len, d_model, 0)));
    stage.bindWorkspace(&workspace);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    ASSERT_TRUE(stage.setSelectedRowsForReplay(replay_rows));
    ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, stream));
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(replay_rows.size()), d_model, stream),
               *hidden,
               replay_rows,
               d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage, CapturedFixedContiguousRowsAreByteExactWithoutMetadata)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int bucket_seq_len = 8;
    constexpr int d_model = 32;
    constexpr int first_row = 2;
    constexpr int row_count = 3;
    const std::vector<int> expected_rows{2, 3, 4};

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{row_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_count = row_count;
    params.selected_row_indices = expected_rows;
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::
            FixedContiguousRange;
    params.fixed_contiguous_row_start = first_row;
    HiddenStateRowsSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.getWorkspaceRequirements(bucket_seq_len, d_model, 0).buffers.empty());
    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(
        downloadScratchRows(*scratch, row_count, d_model, stream),
        *hidden,
        expected_rows,
        d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(
        downloadScratchRows(*scratch, row_count, d_model, stream),
        *hidden,
        expected_rows,
        d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage, CapturedGraphReplayReadsExternalMetadataRows)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    cudaSetDevice(0);

    const DeviceId device = DeviceId::cuda(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;
    const std::vector<int> initial_rows{2, 4, 6};
    const std::vector<int> replay_rows{7, 1, 0};
    constexpr const char *kExternalRows = "mtp_spec_decode_verifier_rows";

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{initial_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_count = static_cast<int>(initial_rows.size());
    params.selected_row_indices = {0, 1, 2};
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::ExternalDeviceIndices;

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({
        kExternalRows,
        initial_rows.size() * sizeof(int),
        alignof(int),
        true});
    DeviceWorkspaceManager workspace(device, 1024);
    ASSERT_TRUE(workspace.allocate(reqs));
    params.external_device_row_indices =
        static_cast<const int32_t *>(workspace.getBuffer(kExternalRows));
    ASSERT_NE(params.external_device_row_indices, nullptr);
    HiddenStateRowsSelectStage stage(params);
    stage.setGPUStream(stream);

    auto upload_rows = [&](const std::vector<int> &rows)
    {
        ASSERT_EQ(cudaMemcpyAsync(
                      workspace.getBuffer(kExternalRows),
                      rows.data(),
                      rows.size() * sizeof(int),
                      cudaMemcpyHostToDevice,
                      stream),
                  cudaSuccess);
    };

    upload_rows(initial_rows);
    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    upload_rows(replay_rows);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(replay_rows.size()), d_model, stream),
               *hidden,
               replay_rows,
               d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage,
     CapturedGraphReplaysAcrossResidentRequestWidths)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int request_count = 2;
    constexpr int seq_capacity = 16;
    const int32_t initial_row_stride = 8;
    const int32_t replay_row_stride = 6;
    constexpr int d_model = 32;
    const std::vector<int32_t> initial_lengths{8, 3};
    const std::vector<int32_t> replay_lengths{4, 6};
    const std::vector<int> initial_terminal_rows{7, 10};
    const std::vector<int> replay_terminal_rows{3, 11};

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(seq_capacity, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{request_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *lengths_device = nullptr;
    int32_t *row_stride_device = nullptr;
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&lengths_device),
            initial_lengths.size() * sizeof(int32_t)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&row_stride_device),
            sizeof(int32_t)),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            lengths_device,
            initial_lengths.data(),
            initial_lengths.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            row_stride_device,
            &initial_row_stride,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = seq_capacity;
    params.d_model = d_model;
    params.selected_row_count = request_count;
    params.selected_row_indices = initial_terminal_rows;
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::RequestTerminalLengths;
    params.request_sequence_lengths_device = lengths_device;
    params.request_row_stride_source =
        HiddenStateRowsSelectStage::RequestRowStrideSource::
            ExternalDeviceScalar;
    params.request_row_stride_device = row_stride_device;
    HiddenStateRowsSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.getWorkspaceRequirements(seq_capacity, d_model, 0).buffers.empty())
        << "Resident request lengths must not allocate or alias verifier-row metadata";
    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(
        downloadScratchRows(*scratch, request_count, d_model, stream),
        *hidden,
        initial_terminal_rows,
        d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(
        downloadScratchRows(*scratch, request_count, d_model, stream),
        *hidden,
        initial_terminal_rows,
        d_model);

    /*
     * Both values change inside the same logical device geometry record. The
     * captured kernel must derive new terminal rows without a prompt-width
     * graph identity or host row-plan update.
     */
    ASSERT_EQ(
        cudaMemcpyAsync(
            lengths_device,
            replay_lengths.data(),
            replay_lengths.size() * sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(
            row_stride_device,
            &replay_row_stride,
            sizeof(int32_t),
            cudaMemcpyHostToDevice,
            stream),
        cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(
        downloadScratchRows(*scratch, request_count, d_model, stream),
        *hidden,
        replay_terminal_rows,
        d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFree(lengths_device);
    cudaFree(row_stride_device);
    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage,
     CapturedGraphReplaysAcrossShiftedPrefillKVProgress)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int seq_capacity = 24;
    constexpr int d_model = 32;
    constexpr int request_index = 1;
    constexpr int selected_row_count = 3;
    const int32_t initial_stride = 10;
    const std::vector<int32_t> initial_lengths{7, 8};
    const int32_t initial_main_count = 20;
    const int32_t initial_shifted_count = 14;
    const std::vector<int> initial_rows{12, 13, 14};
    const int32_t replay_stride = 8;
    const std::vector<int32_t> replay_lengths{5, 6};
    const int32_t replay_main_count = 31;
    const int32_t replay_shifted_count = 28;
    const std::vector<int> replay_rows{11, 12, 13};

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    auto hidden = makeHiddenStates(seq_capacity, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{selected_row_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *lengths_device = nullptr;
    int32_t *stride_device = nullptr;
    int32_t *main_count_device = nullptr;
    int32_t *shifted_count_device = nullptr;
    ASSERT_EQ(cudaMalloc(
                  reinterpret_cast<void **>(&lengths_device),
                  initial_lengths.size() * sizeof(int32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(
                  reinterpret_cast<void **>(&stride_device),
                  sizeof(int32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(
                  reinterpret_cast<void **>(&main_count_device),
                  sizeof(int32_t)),
              cudaSuccess);
    ASSERT_EQ(cudaMalloc(
                  reinterpret_cast<void **>(&shifted_count_device),
                  sizeof(int32_t)),
              cudaSuccess);

    const auto publish_progress =
        [&](const std::vector<int32_t> &lengths,
            const int32_t stride,
            const int32_t main_count,
            const int32_t shifted_count)
    {
        ASSERT_EQ(cudaMemcpyAsync(
                      lengths_device,
                      lengths.data(),
                      lengths.size() * sizeof(int32_t),
                      cudaMemcpyHostToDevice,
                      stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      stride_device,
                      &stride,
                      sizeof(int32_t),
                      cudaMemcpyHostToDevice,
                      stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      main_count_device,
                      &main_count,
                      sizeof(int32_t),
                      cudaMemcpyHostToDevice,
                      stream),
                  cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(
                      shifted_count_device,
                      &shifted_count,
                      sizeof(int32_t),
                      cudaMemcpyHostToDevice,
                      stream),
                  cudaSuccess);
    };
    publish_progress(
        initial_lengths,
        initial_stride,
        initial_main_count,
        initial_shifted_count);

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = seq_capacity;
    params.d_model = d_model;
    params.selected_row_count = selected_row_count;
    params.selected_row_indices = initial_rows;
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::
            ShiftedPrefillKVProgress;
    params.request_sequence_lengths_device = lengths_device;
    params.request_row_stride_source =
        HiddenStateRowsSelectStage::RequestRowStrideSource::
            ExternalDeviceScalar;
    params.request_row_stride_device = stride_device;
    params.main_cached_tokens_device = main_count_device;
    params.shifted_cached_tokens_device = shifted_count_device;
    params.request_index = request_index;
    HiddenStateRowsSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.getWorkspaceRequirements(seq_capacity, d_model, 0).buffers.empty())
        << "Canonical KV progress must not allocate or upload a host row cursor";
    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(
        downloadScratchRows(
            *scratch,
            selected_row_count,
            d_model,
            stream),
        *hidden,
        initial_rows,
        d_model);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
            cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        cudaSuccess);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(
        downloadScratchRows(
            *scratch,
            selected_row_count,
            d_model,
            stream),
        *hidden,
        initial_rows,
        d_model);

    publish_progress(
        replay_lengths,
        replay_stride,
        replay_main_count,
        replay_shifted_count);
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    expectRows(
        downloadScratchRows(
            *scratch,
            selected_row_count,
            d_model,
            stream),
        *hidden,
        replay_rows,
        d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFree(lengths_device);
    cudaFree(stride_device);
    cudaFree(main_count_device);
    cudaFree(shifted_count_device);
    cudaStreamDestroy(stream);
#endif
}

TEST(Test__CUDAHiddenStateRowSelectStage,
     CapturedShiftedPrefillTransactionIsExactAcrossInitialAndBridgeSegments)
{
#ifndef HAVE_CUDA
    GTEST_SKIP() << "CUDA support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No CUDA device available";
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    const DeviceId device = DeviceId::cuda(0);
    constexpr int captured_rows = 8;
    constexpr int d_model = 32;
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    auto hidden = makeHiddenStates(captured_rows, d_model, device, stream);
    auto packed = std::make_unique<FP32Tensor>(
        std::vector<size_t>{captured_rows, d_model},
        DeviceId::cpu());
    auto archive = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, d_model},
        DeviceId::cpu());
    std::fill(
        packed->mutable_data(),
        packed->mutable_data() + packed->numel(),
        0.0f);
    std::fill(
        archive->mutable_data(),
        archive->mutable_data() + archive->numel(),
        -1.0f);
    ASSERT_TRUE(packed->allocateOnDevice(device, stream));
    ASSERT_TRUE(archive->ensureOnDevice(device, stream));

    int32_t *input_tokens = nullptr;
    int32_t *input_positions = nullptr;
    int32_t *shifted_tokens = nullptr;
    int32_t *shifted_positions = nullptr;
    int32_t *append_lengths = nullptr;
    int32_t *request_lengths = nullptr;
    int32_t *request_stride = nullptr;
    int32_t *main_count = nullptr;
    int32_t *shifted_count = nullptr;
    const size_t row_bytes = captured_rows * sizeof(int32_t);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&input_tokens), row_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&input_positions), row_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&shifted_tokens), row_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&shifted_positions), row_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&append_lengths), sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&request_lengths), sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&request_stride), sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&main_count), sizeof(int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(&shifted_count), sizeof(int32_t)), cudaSuccess);

    const auto upload_rows = [&](const std::array<int32_t, captured_rows> &tokens,
                                 const std::array<int32_t, captured_rows> &positions,
                                 int32_t length,
                                 int32_t stride,
                                 int32_t main_tokens,
                                 int32_t shifted_tokens_count)
    {
        ASSERT_EQ(cudaMemcpyAsync(input_tokens, tokens.data(), row_bytes,
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(input_positions, positions.data(), row_bytes,
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(request_lengths, &length, sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(request_stride, &stride, sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(main_count, &main_tokens, sizeof(int32_t),
                                  cudaMemcpyHostToDevice, stream), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(shifted_count, &shifted_tokens_count,
                                  sizeof(int32_t), cudaMemcpyHostToDevice,
                                  stream), cudaSuccess);
    };

    const std::array<int32_t, captured_rows> initial_tokens{1, 2, 3, 4, 5, 0, 0, 0};
    const std::array<int32_t, captured_rows> initial_positions{10, 11, 12, 13, 14, 0, 0, 0};
    upload_rows(initial_tokens, initial_positions,
                /*length=*/5, /*stride=*/5,
                /*main_tokens=*/5, /*shifted_tokens_count=*/0);

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = packed.get();
    params.seq_len = captured_rows;
    params.d_model = d_model;
    params.selected_row_count = captured_rows;
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::
            ShiftedPrefillTransaction;
    params.request_sequence_lengths_device = request_lengths;
    params.request_row_stride_source =
        HiddenStateRowsSelectStage::RequestRowStrideSource::
            ExternalDeviceScalar;
    params.request_row_stride_device = request_stride;
    params.input_token_ids_device = input_tokens;
    params.input_position_ids_device = input_positions;
    params.shifted_token_ids_output_device = shifted_tokens;
    params.shifted_position_ids_output_device = shifted_positions;
    params.shifted_append_lengths_output_device = append_lengths;
    params.terminal_hidden_archive = archive.get();
    params.main_cached_tokens_by_request = {main_count};
    params.shifted_cached_tokens_by_request = {shifted_count};
    params.request_count = 1;
    params.terminal_hidden_archive_buffer_id =
        BufferId::PREFIX_TERMINAL_HIDDEN;
    HiddenStateRowsSelectStage stage(std::move(params));
    stage.setGPUStream(stream);

    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal),
                  cudaSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(cudaStreamEndCapture(stream, &graph), cudaSuccess);
    }
    ASSERT_NE(graph, nullptr);
    size_t graph_node_count = 0;
    ASSERT_EQ(
        cudaGraphGetNodes(graph, nullptr, &graph_node_count),
        cudaSuccess);
    EXPECT_EQ(graph_node_count, 1U)
        << "The shifted-prefill payload and terminal archive must remain one fused transaction";
    ASSERT_EQ(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
              cudaSuccess);

    const auto download_transaction = [&]()
    {
        struct Result
        {
            std::vector<float> hidden;
            std::array<int32_t, captured_rows> tokens{};
            std::array<int32_t, captured_rows> positions{};
            std::array<float, d_model> archive{};
            int32_t append_length = -1;
        } result;
        result.hidden.resize(
            static_cast<size_t>(captured_rows) * d_model);
        EXPECT_EQ(cudaMemcpyAsync(result.hidden.data(), packed->gpu_data_ptr(),
                                  result.hidden.size() * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(result.tokens.data(), shifted_tokens, row_bytes,
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(result.positions.data(), shifted_positions, row_bytes,
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(&result.append_length, append_lengths,
                                  sizeof(int32_t), cudaMemcpyDeviceToHost,
                                  stream), cudaSuccess);
        EXPECT_EQ(cudaMemcpyAsync(result.archive.data(), archive->gpu_data_ptr(),
                                  d_model * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream), cudaSuccess);
        EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        return result;
    };

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    const auto initial = download_transaction();
    EXPECT_EQ(initial.append_length, 4);
    EXPECT_EQ(initial.tokens,
              (std::array<int32_t, captured_rows>{2, 3, 4, 5, 0, 0, 0, 0}));
    EXPECT_EQ(initial.positions,
              (std::array<int32_t, captured_rows>{11, 12, 13, 14, 0, 0, 0, 0}));
    for (int row = 0; row < captured_rows; ++row)
    {
        for (int column = 0; column < d_model; ++column)
        {
            const float expected = row < 4
                                       ? hidden->data()[static_cast<size_t>(row) * d_model + column]
                                       : 0.0f;
            EXPECT_FLOAT_EQ(initial.hidden[static_cast<size_t>(row) * d_model + column], expected);
        }
    }
    for (int column = 0; column < d_model; ++column)
    {
        EXPECT_FLOAT_EQ(initial.archive[static_cast<size_t>(column)],
                        hidden->data()[static_cast<size_t>(4) * d_model + column]);
    }

    std::vector<float> continuation_hidden(
        static_cast<size_t>(captured_rows) * d_model,
        0.0f);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < d_model; ++column)
        {
            continuation_hidden[static_cast<size_t>(row) * d_model + column] =
                1000.0f + 10.0f * row + 0.125f * column;
        }
    }
    ASSERT_EQ(cudaMemcpyAsync(hidden->gpu_data_ptr(), continuation_hidden.data(),
                              continuation_hidden.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream), cudaSuccess);
    const std::array<int32_t, captured_rows> bridge_tokens{6, 7, 8, 0, 0, 0, 0, 0};
    const std::array<int32_t, captured_rows> bridge_positions{15, 16, 17, 0, 0, 0, 0, 0};
    upload_rows(bridge_tokens, bridge_positions,
                /*length=*/3, /*stride=*/3,
                /*main_tokens=*/8, /*shifted_tokens_count=*/4);

    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    const auto bridge = download_transaction();
    EXPECT_EQ(bridge.append_length, 3);
    EXPECT_EQ(bridge.tokens,
              (std::array<int32_t, captured_rows>{6, 7, 8, 0, 0, 0, 0, 0}));
    EXPECT_EQ(bridge.positions,
              (std::array<int32_t, captured_rows>{15, 16, 17, 0, 0, 0, 0, 0}));
    for (int column = 0; column < d_model; ++column)
    {
        EXPECT_FLOAT_EQ(
            bridge.hidden[static_cast<size_t>(column)],
            initial.archive[static_cast<size_t>(column)]);
        EXPECT_FLOAT_EQ(
            bridge.hidden[static_cast<size_t>(d_model) + column],
            continuation_hidden[static_cast<size_t>(column)]);
        EXPECT_FLOAT_EQ(
            bridge.hidden[static_cast<size_t>(2 * d_model) + column],
            continuation_hidden[static_cast<size_t>(d_model) + column]);
        EXPECT_FLOAT_EQ(
            bridge.archive[static_cast<size_t>(column)],
            continuation_hidden[static_cast<size_t>(2 * d_model) + column]);
    }
    for (int row = 3; row < captured_rows; ++row)
    {
        for (int column = 0; column < d_model; ++column)
        {
            EXPECT_FLOAT_EQ(
                bridge.hidden[static_cast<size_t>(row) * d_model + column],
                0.0f);
        }
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaFree(input_tokens);
    cudaFree(input_positions);
    cudaFree(shifted_tokens);
    cudaFree(shifted_positions);
    cudaFree(append_lengths);
    cudaFree(request_lengths);
    cudaFree(request_stride);
    cudaFree(main_count);
    cudaFree(shifted_count);
    cudaStreamDestroy(stream);
#endif
}
