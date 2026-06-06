/**
 * @file Test__CUDAHiddenStateRowSelectStage.cpp
 * @brief CUDA integration tests for graph-capturable hidden-state row selection.
 *
 * Captures one row-select stage into a CUDA graph, replays it twice, and verifies
 * that changing PrefillReplayParams changes the selected row without recapture.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include <memory>
#include <vector>

using namespace llaminar2;

namespace
{
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
    std::vector<float> downloadScratchRow(FP32Tensor &scratch, int d_model)
    {
        std::vector<float> row(static_cast<size_t>(d_model), 0.0f);
        cudaMemcpy(row.data(), scratch.gpu_data_ptr(), static_cast<size_t>(d_model) * sizeof(float), cudaMemcpyDeviceToHost);
        return row;
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
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 1, d_model);

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
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 1, d_model);

    // No recapture: update only the pinned scalar and replay the same graph exec.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{6, bucket_seq_len, 0});
    ASSERT_EQ(cudaGraphLaunch(graph_exec, stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 5, d_model);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
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
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 1, d_model);

    stage_ptr->setSelectedRowForReplay(5);
    ASSERT_TRUE(executor.execute(graph, ctx.get()));
    ASSERT_EQ(cudaStreamSynchronize(static_cast<cudaStream_t>(stage_ptr->gpuStream())), cudaSuccess);
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 5, d_model);

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
