/**
 * @file Test__ROCmHiddenStateRowSelectStage.cpp
 * @brief ROCm integration tests for graph-capturable hidden-state row selection.
 *
 * Captures one row-select stage into a HIP graph, replays it twice, and verifies
 * that changing PrefillReplayParams changes the selected row without recapture.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
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
        hipMemcpy(row.data(), scratch.gpu_data_ptr(), static_cast<size_t>(d_model) * sizeof(float), hipMemcpyDeviceToHost);
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

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedGraphReplayUsesUpdatedSelectedRow)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm support not compiled";
#else
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    if (device_count <= 0)
        GTEST_SKIP() << "No ROCm device available";
    hipSetDevice(0);

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

    hipGraphExecDestroy(graph_exec);
    hipGraphDestroy(graph);
    hipStreamDestroy(stream);
#endif
}