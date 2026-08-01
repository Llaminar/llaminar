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
    ASSERT_FALSE(stage.needsGraphLaunchPreparation());

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
    params.workspace_buffer_name = kExternalRows;
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

TEST(Test__ROCmHiddenStateRowSelectStage, CapturedGraphReadsResidentUnequalRequestLengths)
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
    constexpr int request_row_stride = 8;
    constexpr int total_rows = request_count * request_row_stride;
    constexpr int d_model = 32;
    const std::vector<int32_t> initial_lengths{8, 3};
    const std::vector<int32_t> replay_lengths{4, 8};
    const std::vector<int> initial_terminal_rows{7, 10};
    const std::vector<int> replay_terminal_rows{3, 15};

    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

    auto hidden = makeHiddenStates(total_rows, d_model, device, stream);
    auto scratch = std::make_unique<FP32Tensor>(
        std::vector<size_t>{request_count, static_cast<size_t>(d_model)},
        DeviceId::cpu());
    ASSERT_TRUE(scratch->allocateOnDevice(device, stream));

    int32_t *lengths_device = nullptr;
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&lengths_device),
            initial_lengths.size() * sizeof(int32_t)),
        hipSuccess);
    ASSERT_EQ(
        hipMemcpyAsync(
            lengths_device,
            initial_lengths.data(),
            initial_lengths.size() * sizeof(int32_t),
            hipMemcpyHostToDevice,
            stream),
        hipSuccess);

    HiddenStateRowsSelectStage::Params params;
    params.device_id = device;
    params.input = hidden.get();
    params.output = scratch.get();
    params.seq_len = total_rows;
    params.d_model = d_model;
    params.selected_row_count = request_count;
    params.selected_row_indices = initial_terminal_rows;
    params.device_row_index_source =
        HiddenStateRowsSelectStage::DeviceRowIndexSource::RequestTerminalLengths;
    params.request_sequence_lengths_device = lengths_device;
    params.request_row_stride = request_row_stride;
    HiddenStateRowsSelectStage stage(params);
    stage.setGPUStream(stream);

    ASSERT_TRUE(stage.getWorkspaceRequirements(total_rows, d_model, 0).buffers.empty())
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
     * Only the device length array changes. The captured kernel must derive a
     * new terminal row for each request without a host row-plan update.
     */
    ASSERT_EQ(
        hipMemcpyAsync(
            lengths_device,
            replay_lengths.data(),
            replay_lengths.size() * sizeof(int32_t),
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
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
#endif
}
