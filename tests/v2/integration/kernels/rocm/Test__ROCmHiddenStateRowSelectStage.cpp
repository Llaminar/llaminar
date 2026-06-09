/**
 * @file Test__ROCmHiddenStateRowSelectStage.cpp
 * @brief ROCm integration tests for graph-capturable hidden-state row selection.
 *
 * Captures one row-select stage into a HIP graph, replays it twice, and verifies
 * that changing PrefillReplayParams changes the selected row without recapture.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "tensors/Tensors.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <memory>
#include <vector>

using namespace llaminar2;

namespace
{
    /// @brief Fill hidden rows with deterministic values that identify the row.
    std::unique_ptr<FP32Tensor> makeHiddenStates(int seq_len, int d_model, DeviceId device)
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
                    20.0f * static_cast<float>(row + 1) + 0.0625f * static_cast<float>(column);
            }
        }
        hidden->ensureOnDevice(device);
        return hidden;
    }

#ifdef HAVE_ROCM
    /// @brief Copy scratch row from HIP device memory for assertion.
    std::vector<float> downloadScratchRow(FP32Tensor &scratch, int d_model)
    {
        std::vector<float> row(static_cast<size_t>(d_model), 0.0f);
        EXPECT_EQ(hipMemcpy(row.data(),
                            scratch.gpu_data_ptr(),
                            static_cast<size_t>(d_model) * sizeof(float),
                            hipMemcpyDeviceToHost),
                  hipSuccess);
        return row;
    }

    /// @brief Copy compact scratch rows from HIP device memory for assertion.
    std::vector<float> downloadScratchRows(FP32Tensor &scratch, int row_count, int d_model, hipStream_t stream)
    {
        std::vector<float> rows(static_cast<size_t>(row_count) * static_cast<size_t>(d_model), 0.0f);
        EXPECT_EQ(hipMemcpyAsync(
                      rows.data(),
                      scratch.gpu_data_ptr(),
                      rows.size() * sizeof(float),
                      hipMemcpyDeviceToHost,
                      stream),
                  hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
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

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedGraphReplayUsesUpdatedSelectedRow)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const DeviceId device = DeviceId::rocm(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;
    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device);

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

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    stage.setGPUStream(stream);

    // Warmup performs scalar allocation before capture and proves the stage path works.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{2, bucket_seq_len, 0});
    ASSERT_TRUE(stage.execute(nullptr));
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 1, d_model);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 1, d_model);

    // No recapture: update only the pinned scalar and replay the same graph exec.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{6, bucket_seq_len, 0});
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(downloadScratchRow(*scratch, d_model), *hidden, 5, d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmHiddenStateRowSelectStage, GpuExecutionRequiresBoundWorkspace)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const DeviceId device = DeviceId::rocm(0);
    const int bucket_seq_len = 4;
    const int d_model = 16;
    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device);

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    HiddenStateRowSelectStage stage(params);

    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{2, bucket_seq_len, 0});
    EXPECT_FALSE(stage.execute(nullptr));
#endif
}

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedGraphReplayUsesUpdatedSelectedRows)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const DeviceId device = DeviceId::rocm(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;
    const std::vector<int> initial_rows{1, 3, 6};
    const std::vector<int> replay_rows{7, 0, 4};
    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{initial_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device);

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

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    ASSERT_TRUE(stage.setSelectedRowsForReplay(replay_rows));
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(replay_rows.size()), d_model, stream),
               *hidden,
               replay_rows,
               d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedGraphReplayReadsExternalMetadataRows)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    ASSERT_EQ(hipGetDeviceCount(&device_count), hipSuccess);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    const DeviceId device = DeviceId::rocm(0);
    const int bucket_seq_len = 8;
    const int d_model = 32;
    const std::vector<int> initial_rows{2, 4, 6};
    const std::vector<int> replay_rows{7, 1, 0};
    constexpr const char *kExternalRows = "mtp_spec_decode_verifier_rows";

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{initial_rows.size(), static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device, stream);

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    params.selected_row_count = static_cast<int>(initial_rows.size());
    params.selected_row_indices = {0, 1, 2};
    params.workspace_buffer_name = kExternalRows;
    params.declare_selected_rows_workspace = false;
    params.upload_selected_rows_to_workspace = false;
    HiddenStateRowsSelectStage stage(params);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({
        kExternalRows,
        initial_rows.size() * sizeof(int),
        alignof(int),
        true});
    DeviceWorkspaceManager workspace(device, 1024);
    ASSERT_TRUE(workspace.allocate(reqs));
    stage.bindWorkspace(&workspace);
    stage.setGPUStream(stream);

    auto upload_rows = [&](const std::vector<int> &rows)
    {
        ASSERT_EQ(hipMemcpyAsync(
                      workspace.getBuffer(kExternalRows),
                      rows.data(),
                      rows.size() * sizeof(int),
                      hipMemcpyHostToDevice,
                      stream),
                  hipSuccess);
    };

    upload_rows(initial_rows);
    ASSERT_TRUE(stage.execute(nullptr));
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(initial_rows.size()), d_model, stream),
               *hidden,
               initial_rows,
               d_model);

    upload_rows(replay_rows);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(downloadScratchRows(*scratch, static_cast<int>(replay_rows.size()), d_model, stream),
               *hidden,
               replay_rows,
               d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}
