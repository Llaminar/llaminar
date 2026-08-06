/**
 * @file Test__ROCmHiddenStateRowSelectStage.cpp
 * @brief ROCm integration tests for graph-capturable hidden-state row selection.
 *
 * Covers dynamic stage-owned rows, immutable device-only checkpoint rows,
 * external verifier metadata, and request-terminal rows derived directly from
 * device-resident unequal request lengths. Captured graph replays prove each
 * ownership policy executes without host work inside capture.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/HiddenStateRowSelectStage.h"
#include "execution/compute_stages/stages/HiddenStateRowsSelectStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/ComputeGraph.h"
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "memory/BufferArena.h"
#include "tensors/Tensors.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

using namespace llaminar2;

namespace
{
#ifdef HAVE_ROCM
    /**
     * @brief Simulate a preceding library stage that temporarily binds another GPU.
     *
     * Real LocalTP graphs interleave compute stages with RCCL and backend helper
     * calls. This deliberately adversarial stage reproduces the relevant
     * execution boundary without loading a model: it leaves the calling thread
     * on another HIP device immediately before the production M=39 checkpoint.
     */
    class ForeignHIPDeviceStage final : public IComputeStage
    {
    public:
        ForeignHIPDeviceStage(DeviceId graph_device, int foreign_device)
            : IComputeStage(graph_device),
              foreign_device_(foreign_device)
        {
        }

        bool execute(IDeviceContext *) override
        {
            return hipSetDevice(foreign_device_) == hipSuccess;
        }

        ComputeStageType type() const override
        {
            return ComputeStageType::COPY;
        }

        bool supportsBackend(ComputeBackendType backend) const override
        {
            return backend == ComputeBackendType::GPU_ROCM;
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

    /// @brief Fill hidden rows with deterministic values that identify the row.
    std::unique_ptr<FP32Tensor> makeHiddenStates(int seq_len, int d_model, DeviceId device, hipStream_t stream)
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
        hidden->ensureOnDevice(device, stream);
        return hidden;
    }

    /// @brief Copy scratch row from HIP device memory for assertion.
    std::vector<float> downloadScratchRow(FP32Tensor &scratch, int d_model, hipStream_t stream)
    {
        std::vector<float> row(static_cast<size_t>(d_model), 0.0f);
        EXPECT_EQ(hipMemcpyAsync(row.data(),
                                 scratch.gpu_data_ptr(),
                                 static_cast<size_t>(d_model) * sizeof(float),
                                 hipMemcpyDeviceToHost,
                                 stream),
                  hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device, stream);

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
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 1, d_model);

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
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 1, d_model);

    // No recapture: update host intent, then let the graph-launch preparation
    // hook upload the scalar on the same explicit stream used for graph launch.
    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{6, bucket_seq_len, 0});
    ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, stream));
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(downloadScratchRow(*scratch, d_model, stream), *hidden, 5, d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @brief Prove fixed diagnostic rows capture without scalar workspace or H2D.
 */
TEST(Test__ROCmHiddenStateRowSelectStage, CapturedFixedDeviceRowUsesDeviceOnlyCopy)
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
    const int seq_len = 8;
    const int d_model = 32;
    const int selected_row = 5;
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

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
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        selected_row,
        d_model);

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        hipSuccess);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        selected_row,
        d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @brief Reproduce padded M=512 checkpoint selection from resident metadata.
 *
 * A single captured executable first observes a 443-row request and then a
 * 257-row request. Only the persistent device length changes between launches;
 * no stage replay setter or graph-launch preparation hook participates.
 */
TEST(Test__ROCmHiddenStateRowSelectStage, CapturedCheckpointReadsResidentPrefillLength)
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
    constexpr int bucket_seq_len = 512;
    constexpr int d_model = 32;
    int32_t initial_length = 443;
    int32_t replay_length = 257;

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
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
        hipMalloc(
            reinterpret_cast<void **>(&length_device),
            sizeof(int32_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            length_device,
            &initial_length,
            sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);

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

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipStreamBeginCapture(
                stream,
                hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        hipGraphInstantiate(
            &graph_exec,
            graph,
            nullptr,
            nullptr,
            0),
        hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        /*selected_row=*/442,
        d_model);

    ASSERT_EQ(
        hipMemcpyAsync(
            length_device,
            &replay_length,
            sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, stream),
        *hidden,
        /*selected_row=*/256,
        d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipFree(length_device), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @brief Reproduce the first production M=39 mirrored checkpoint through the executor.
 *
 * The long-context ROCm2 E2E first builds an exact-shape, 39-row prefill graph
 * with a 2048-wide hidden state. This test deliberately uses graph-managed
 * arena tensors and leaves stream assignment to DeviceGraphExecutor. It
 * therefore proves the production ownership contract in addition to the raw
 * HIP kernel geometry: the executor must bind a non-null worker stream and the
 * stage must read the terminal row from the persistent device length.
 */
TEST(Test__ROCmHiddenStateRowSelectStage, ExecutorRunsProductionM39ResidentCheckpoint)
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
    constexpr int seq_len = 39;
    constexpr int d_model = 2048;
    const int32_t request_length = seq_len;

    hipStream_t setup_stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&setup_stream), hipSuccess);
    auto hidden = makeHiddenStates(seq_len, d_model, device, setup_stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());

    int32_t *length_device = nullptr;
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&length_device),
            sizeof(int32_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            length_device,
            &request_length,
            sizeof(int32_t),
            hipMemcpyHostToDevice,
            setup_stream),
        hipSuccess);
    ASSERT_EQ(hipStreamSynchronize(setup_stream), hipSuccess);

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
            "foreign_hip_device_binding",
            std::make_unique<ForeignHIPDeviceStage>(device, 1),
            device);
    }
    graph.addNode("production_m39_resident_checkpoint", std::move(stage), device);
    if (device_count > 1)
    {
        graph.addDependency(
            "production_m39_resident_checkpoint",
            "foreign_hip_device_binding");
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
    ASSERT_EQ(hipGetDevice(&active_device), hipSuccess);
    EXPECT_EQ(active_device, 0)
        << "The checkpoint handoff must restore its stream-owning device";
    auto execution_stream =
        static_cast<hipStream_t>(stage_ptr->gpuStream());
    ASSERT_EQ(hipStreamSynchronize(execution_stream), hipSuccess);
    expectRow(
        downloadScratchRow(*scratch, d_model, execution_stream),
        *hidden,
        /*selected_row=*/seq_len - 1,
        d_model);

    EXPECT_EQ(hipFree(length_device), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(setup_stream), hipSuccess);
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{1, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device, stream);

    HiddenStateRowSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = bucket_seq_len;
    params.d_model = d_model;
    HiddenStateRowSelectStage stage(params);
    stage.setGPUStream(stream);

    stage.updatePrefillReplayParams(IComputeStage::PrefillReplayParams{2, bucket_seq_len, 0});
    EXPECT_FALSE(stage.execute(nullptr));

    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
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
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
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
    ASSERT_TRUE(stage.prepareGraphLaunch(nullptr, stream));
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

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedFixedContiguousRowsAreByteExactWithoutMetadata)
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
    constexpr int bucket_seq_len = 8;
    constexpr int d_model = 32;
    constexpr int first_row = 2;
    constexpr int row_count = 3;
    const std::vector<int> expected_rows{2, 3, 4};

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{row_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    scratch->ensureOnDevice(device, stream);

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

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        hipSuccess);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(
        downloadScratchRows(*scratch, row_count, d_model, stream),
        *hidden,
        expected_rows,
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

    auto hidden = makeHiddenStates(bucket_seq_len, d_model, device, stream);
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

TEST(Test__ROCmHiddenStateRowSelectStage,
     CapturedGraphReplaysAcrossResidentRequestWidths)
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
    constexpr int request_count = 2;
    constexpr int seq_capacity = 16;
    const int32_t initial_row_stride = 8;
    const int32_t replay_row_stride = 6;
    constexpr int d_model = 32;
    const std::vector<int32_t> initial_lengths{8, 3};
    const std::vector<int32_t> replay_lengths{4, 6};
    const std::vector<int> initial_terminal_rows{7, 10};
    const std::vector<int> replay_terminal_rows{3, 11};

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(seq_capacity, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{request_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *lengths_device = nullptr;
    int32_t *row_stride_device = nullptr;
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&lengths_device),
            initial_lengths.size() * sizeof(int32_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&row_stride_device),
            sizeof(int32_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            lengths_device,
            initial_lengths.data(),
            initial_lengths.size() * sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            row_stride_device,
            &initial_row_stride,
            sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);

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

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
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
        hipMemcpyAsync(
            lengths_device,
            replay_lengths.data(),
            replay_lengths.size() * sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            row_stride_device,
            &replay_row_stride,
            sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(
        downloadScratchRows(*scratch, request_count, d_model, stream),
        *hidden,
        replay_terminal_rows,
        d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipFree(lengths_device), hipSuccess);
    EXPECT_EQ(hipFree(row_stride_device), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

TEST(Test__ROCmHiddenStateRowSelectStage,
     CapturedGraphReplaysAcrossShiftedPrefillKVProgress)
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

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    auto hidden = makeHiddenStates(seq_capacity, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{selected_row_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *lengths_device = nullptr;
    int32_t *stride_device = nullptr;
    int32_t *main_count_device = nullptr;
    int32_t *shifted_count_device = nullptr;
    ASSERT_EQ(hipMalloc(
                  reinterpret_cast<void **>(&lengths_device),
                  initial_lengths.size() * sizeof(int32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(
                  reinterpret_cast<void **>(&stride_device),
                  sizeof(int32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(
                  reinterpret_cast<void **>(&main_count_device),
                  sizeof(int32_t)),
              hipSuccess);
    ASSERT_EQ(hipMalloc(
                  reinterpret_cast<void **>(&shifted_count_device),
                  sizeof(int32_t)),
              hipSuccess);

    const auto publish_progress =
        [&](const std::vector<int32_t> &lengths,
            const int32_t stride,
            const int32_t main_count,
            const int32_t shifted_count)
    {
        ASSERT_EQ(hipMemcpyAsync(
                      lengths_device,
                      lengths.data(),
                      lengths.size() * sizeof(int32_t),
                      hipMemcpyHostToDevice,
                      stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                      stride_device,
                      &stride,
                      sizeof(int32_t),
                      hipMemcpyHostToDevice,
                      stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                      main_count_device,
                      &main_count,
                      sizeof(int32_t),
                      hipMemcpyHostToDevice,
                      stream),
                  hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(
                      shifted_count_device,
                      &shifted_count,
                      sizeof(int32_t),
                      hipMemcpyHostToDevice,
                      stream),
                  hipSuccess);
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

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
            hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    ASSERT_EQ(
        hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
        hipSuccess);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
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
    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
    expectRows(
        downloadScratchRows(
            *scratch,
            selected_row_count,
            d_model,
            stream),
        *hidden,
        replay_rows,
        d_model);

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipFree(lengths_device), hipSuccess);
    EXPECT_EQ(hipFree(stride_device), hipSuccess);
    EXPECT_EQ(hipFree(main_count_device), hipSuccess);
    EXPECT_EQ(hipFree(shifted_count_device), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}

/**
 * @brief Prove the graph-integrated shifted-prefill transaction on HIP.
 *
 * The first replay covers a fresh request, where the shifted cache consumes
 * every main-prefill row except the terminal row. The second replay covers a
 * continuation segment, where the archived terminal row bridges the prior and
 * current segments. Keeping both transitions in one captured graph verifies
 * that live cache counters and admitted request geometry remain device-owned.
 */
TEST(Test__ROCmHiddenStateRowSelectStage,
     CapturedShiftedPrefillTransactionIsExactAcrossInitialAndBridgeSegments)
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
    constexpr int captured_rows = 8;
    constexpr int d_model = 32;
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

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
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&input_tokens), row_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&input_positions), row_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&shifted_tokens), row_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&shifted_positions), row_bytes), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&append_lengths), sizeof(int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&request_lengths), sizeof(int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&request_stride), sizeof(int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&main_count), sizeof(int32_t)), hipSuccess);
    ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(&shifted_count), sizeof(int32_t)), hipSuccess);

    const auto upload_rows = [&](const std::array<int32_t, captured_rows> &tokens,
                                 const std::array<int32_t, captured_rows> &positions,
                                 int32_t length,
                                 int32_t stride,
                                 int32_t main_tokens,
                                 int32_t shifted_tokens_count)
    {
        ASSERT_EQ(hipMemcpyAsync(input_tokens, tokens.data(), row_bytes,
                                 hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(input_positions, positions.data(), row_bytes,
                                 hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(request_lengths, &length, sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(request_stride, &stride, sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(main_count, &main_tokens, sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream), hipSuccess);
        ASSERT_EQ(hipMemcpyAsync(shifted_count, &shifted_tokens_count,
                                 sizeof(int32_t), hipMemcpyHostToDevice,
                                 stream), hipSuccess);
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

    hipGraph_t graph = nullptr;
    hipGraphExec_t graph_exec = nullptr;
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal),
                  hipSuccess);
        ASSERT_TRUE(stage.execute(nullptr));
        ASSERT_EQ(hipStreamEndCapture(stream, &graph), hipSuccess);
    }
    ASSERT_NE(graph, nullptr);
    size_t graph_node_count = 0;
    ASSERT_EQ(
        hipGraphGetNodes(graph, nullptr, &graph_node_count),
        hipSuccess);
    EXPECT_EQ(graph_node_count, 1U)
        << "The shifted-prefill payload and terminal archive must remain one fused transaction";
    ASSERT_EQ(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0),
              hipSuccess);

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
        EXPECT_EQ(hipMemcpyAsync(result.hidden.data(), packed->gpu_data_ptr(),
                                 result.hidden.size() * sizeof(float),
                                 hipMemcpyDeviceToHost, stream), hipSuccess);
        EXPECT_EQ(hipMemcpyAsync(result.tokens.data(), shifted_tokens, row_bytes,
                                 hipMemcpyDeviceToHost, stream), hipSuccess);
        EXPECT_EQ(hipMemcpyAsync(result.positions.data(), shifted_positions, row_bytes,
                                 hipMemcpyDeviceToHost, stream), hipSuccess);
        EXPECT_EQ(hipMemcpyAsync(&result.append_length, append_lengths,
                                 sizeof(int32_t), hipMemcpyDeviceToHost,
                                 stream), hipSuccess);
        EXPECT_EQ(hipMemcpyAsync(result.archive.data(), archive->gpu_data_ptr(),
                                 d_model * sizeof(float),
                                 hipMemcpyDeviceToHost, stream), hipSuccess);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        return result;
    };

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
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
    ASSERT_EQ(hipMemcpyAsync(hidden->gpu_data_ptr(), continuation_hidden.data(),
                             continuation_hidden.size() * sizeof(float),
                             hipMemcpyHostToDevice, stream), hipSuccess);
    const std::array<int32_t, captured_rows> bridge_tokens{6, 7, 8, 0, 0, 0, 0, 0};
    const std::array<int32_t, captured_rows> bridge_positions{15, 16, 17, 0, 0, 0, 0, 0};
    upload_rows(bridge_tokens, bridge_positions,
                /*length=*/3, /*stride=*/3,
                /*main_tokens=*/8, /*shifted_tokens_count=*/4);

    ASSERT_EQ(hipGraphLaunch(graph_exec, stream), hipSuccess);
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

    EXPECT_EQ(hipGraphExecDestroy(graph_exec), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(graph), hipSuccess);
    EXPECT_EQ(hipFree(input_tokens), hipSuccess);
    EXPECT_EQ(hipFree(input_positions), hipSuccess);
    EXPECT_EQ(hipFree(shifted_tokens), hipSuccess);
    EXPECT_EQ(hipFree(shifted_positions), hipSuccess);
    EXPECT_EQ(hipFree(append_lengths), hipSuccess);
    EXPECT_EQ(hipFree(request_lengths), hipSuccess);
    EXPECT_EQ(hipFree(request_stride), hipSuccess);
    EXPECT_EQ(hipFree(main_count), hipSuccess);
    EXPECT_EQ(hipFree(shifted_count), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}
