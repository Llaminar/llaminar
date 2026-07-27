/**
 * @file Test__RoPEGraphCaptureNoH2D.cpp
 * @brief GPU graph-capture regressions for RoPE dynamic position metadata.
 *
 * RoPE graph replay must not record host-to-device copies. Contiguous positions
 * use a pre-uploaded device scalar for pos_offset; explicit position IDs use a
 * pre-uploaded workspace row buffer so request-batched verifier rows can replay
 * without rebuilding the graph.
 */

#include <gtest/gtest.h>

#include "execution/compute_stages/stages/RoPEStage.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/rope/RoPEDeviceParams.h"
#include "mocks/MockComputeStage.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#include <cuda_runtime.h>
#elif defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#include <hip/hip_runtime.h>
#endif

using namespace llaminar2;

namespace
{
    constexpr int kSeqLen = 6;
    constexpr int kQHeads = 2;
    constexpr int kKVHeads = 1;
    constexpr int kHeadDim = 8;
    constexpr float kThetaBase = 10000.0f;

    std::unique_ptr<FP32Tensor> makeTensor(const std::vector<float> &values, int rows, int cols)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
            DeviceId::cpu());
        std::memcpy(tensor->mutable_data(), values.data(), values.size() * sizeof(float));
        return tensor;
    }

    std::vector<float> makeValues(int rows, int cols, float base)
    {
        std::vector<float> values(static_cast<size_t>(rows) * cols);
        for (int row = 0; row < rows; ++row)
        {
            for (int col = 0; col < cols; ++col)
            {
                values[static_cast<size_t>(row) * cols + col] =
                    base + 0.03125f * static_cast<float>(row + 1) +
                    0.0078125f * static_cast<float>((col % 11) - 5);
            }
        }
        return values;
    }

    void expectNear(const std::vector<float> &actual, const std::vector<float> &expected)
    {
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i)
        {
            EXPECT_NEAR(actual[i], expected[i], 2e-4f) << "i=" << i;
        }
    }

    struct CpuReference
    {
        std::vector<float> q;
        std::vector<float> k;
    };

    CpuReference computeCpuReference(
        const std::vector<float> &q_input,
        const std::vector<float> &k_input,
        int pos_offset,
        const std::vector<int> *position_ids = nullptr)
    {
        auto q = makeTensor(q_input, kSeqLen, kQHeads * kHeadDim);
        auto k = makeTensor(k_input, kSeqLen, kKVHeads * kHeadDim);

        RoPEStage::Params params{};
        params.device_id = DeviceId::cpu();
        params.Q = q.get();
        params.K = k.get();
        params.n_heads = kQHeads;
        params.n_kv_heads = kKVHeads;
        params.head_dim = kHeadDim;
        params.seq_len = kSeqLen;
        params.pos_offset = pos_offset;
        params.theta_base = kThetaBase;
        if (position_ids)
        {
            params.position_ids = position_ids->data();
        }

        RoPEStage stage(params);
        llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(), ComputeBackendType::CPU);
        EXPECT_TRUE(stage.execute(&ctx));

        CpuReference ref;
        ref.q.assign(q->data(), q->data() + q_input.size());
        ref.k.assign(k->data(), k->data() + k_input.size());
        return ref;
    }

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA) || defined(GPU_CONTEXT_TEST_BACKEND_ROCM)

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
    using StreamT = cudaStream_t;
    using EventT = cudaEvent_t;
    using GraphT = cudaGraph_t;
    using GraphExecT = cudaGraphExec_t;

    constexpr auto kBackendType = ComputeBackendType::GPU_CUDA;

    DeviceId testDevice() { return DeviceId::cuda(0); }
    const char *backendName() { return "CUDA"; }

    bool hasDevice()
    {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }

    void setDevice() { ASSERT_EQ(cudaSetDevice(0), cudaSuccess); }
    void createStream(StreamT *stream) { ASSERT_EQ(cudaStreamCreate(stream), cudaSuccess); }
    void destroyStream(StreamT stream) { ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess); }
    void synchronize(StreamT stream) { ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess); }
    void createEvent(EventT *event) { ASSERT_EQ(cudaEventCreateWithFlags(event, cudaEventDisableTiming), cudaSuccess); }
    void destroyEvent(EventT event) { ASSERT_EQ(cudaEventDestroy(event), cudaSuccess); }
    void recordEvent(EventT event, StreamT stream) { ASSERT_EQ(cudaEventRecord(event, stream), cudaSuccess); }
    void waitEvent(StreamT stream, EventT event) { ASSERT_EQ(cudaStreamWaitEvent(stream, event, 0), cudaSuccess); }
    bool eventComplete(EventT event)
    {
        const cudaError_t result = cudaEventQuery(event);
        EXPECT_TRUE(result == cudaSuccess || result == cudaErrorNotReady);
        return result == cudaSuccess;
    }
    void allocateDevice(void **ptr, size_t bytes) { ASSERT_EQ(cudaMalloc(ptr, bytes), cudaSuccess); }
    void freeDevice(void *ptr) { ASSERT_EQ(cudaFree(ptr), cudaSuccess); }
    void fillDevice(void *ptr, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(cudaMemsetAsync(ptr, 0x5a, bytes, stream), cudaSuccess);
    }
    void copyDevice(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream), cudaSuccess);
    }
    void upload(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), cudaSuccess);
    }
    void download(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), cudaSuccess);
    }
    void beginCapture(StreamT stream) { ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess); }
    void endCapture(StreamT stream, GraphT *graph) { ASSERT_EQ(cudaStreamEndCapture(stream, graph), cudaSuccess); }
    void instantiate(GraphExecT *exec, GraphT graph) { ASSERT_EQ(cudaGraphInstantiate(exec, graph, nullptr, nullptr, 0), cudaSuccess); }
    void launch(GraphExecT exec, StreamT stream) { ASSERT_EQ(cudaGraphLaunch(exec, stream), cudaSuccess); }
    void destroyGraphExec(GraphExecT exec) { ASSERT_EQ(cudaGraphExecDestroy(exec), cudaSuccess); }
    void destroyGraph(GraphT graph) { ASSERT_EQ(cudaGraphDestroy(graph), cudaSuccess); }
#else
    using StreamT = hipStream_t;
    using EventT = hipEvent_t;
    using GraphT = hipGraph_t;
    using GraphExecT = hipGraphExec_t;

    constexpr auto kBackendType = ComputeBackendType::GPU_ROCM;

    DeviceId testDevice() { return DeviceId::rocm(0); }
    const char *backendName() { return "ROCm"; }

    bool hasDevice()
    {
        int count = 0;
        return hipGetDeviceCount(&count) == hipSuccess && count > 0;
    }

    void setDevice() { ASSERT_EQ(hipSetDevice(0), hipSuccess); }
    void createStream(StreamT *stream) { ASSERT_EQ(hipStreamCreate(stream), hipSuccess); }
    void destroyStream(StreamT stream) { ASSERT_EQ(hipStreamDestroy(stream), hipSuccess); }
    void synchronize(StreamT stream) { ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess); }
    void createEvent(EventT *event) { ASSERT_EQ(hipEventCreateWithFlags(event, hipEventDisableTiming), hipSuccess); }
    void destroyEvent(EventT event) { ASSERT_EQ(hipEventDestroy(event), hipSuccess); }
    void recordEvent(EventT event, StreamT stream) { ASSERT_EQ(hipEventRecord(event, stream), hipSuccess); }
    void waitEvent(StreamT stream, EventT event) { ASSERT_EQ(hipStreamWaitEvent(stream, event, 0), hipSuccess); }
    bool eventComplete(EventT event)
    {
        const hipError_t result = hipEventQuery(event);
        EXPECT_TRUE(result == hipSuccess || result == hipErrorNotReady);
        return result == hipSuccess;
    }
    void allocateDevice(void **ptr, size_t bytes) { ASSERT_EQ(hipMalloc(ptr, bytes), hipSuccess); }
    void freeDevice(void *ptr) { ASSERT_EQ(hipFree(ptr), hipSuccess); }
    void fillDevice(void *ptr, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(hipMemsetAsync(ptr, 0x5a, bytes, stream), hipSuccess);
    }
    void copyDevice(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream), hipSuccess);
    }
    void upload(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream), hipSuccess);
    }
    void download(void *dst, const void *src, size_t bytes, StreamT stream)
    {
        ASSERT_EQ(hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, stream), hipSuccess);
    }
    void beginCapture(StreamT stream) { ASSERT_EQ(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal), hipSuccess); }
    void endCapture(StreamT stream, GraphT *graph) { ASSERT_EQ(hipStreamEndCapture(stream, graph), hipSuccess); }
    void instantiate(GraphExecT *exec, GraphT graph) { ASSERT_EQ(hipGraphInstantiate(exec, graph, nullptr, nullptr, 0), hipSuccess); }
    void launch(GraphExecT exec, StreamT stream) { ASSERT_EQ(hipGraphLaunch(exec, stream), hipSuccess); }
    void destroyGraphExec(GraphExecT exec) { ASSERT_EQ(hipGraphExecDestroy(exec), hipSuccess); }
    void destroyGraph(GraphT graph) { ASSERT_EQ(hipGraphDestroy(graph), hipSuccess); }
#endif

    void uploadTensor(FP32Tensor &tensor, const std::vector<float> &values, StreamT stream)
    {
        ASSERT_NE(tensor.gpu_data_ptr(), nullptr);
        upload(tensor.gpu_data_ptr(), values.data(), values.size() * sizeof(float), stream);
        synchronize(stream);
    }

    std::vector<float> downloadTensor(FP32Tensor &tensor, size_t count, StreamT stream)
    {
        std::vector<float> values(count, 0.0f);
        if (!tensor.gpu_data_ptr())
        {
            ADD_FAILURE() << "Tensor has no GPU buffer";
            return values;
        }
        download(values.data(), tensor.gpu_data_ptr(), count * sizeof(float), stream);
        synchronize(stream);
        return values;
    }

    struct StageBundle
    {
        std::unique_ptr<FP32Tensor> q;
        std::unique_ptr<FP32Tensor> k;
        std::unique_ptr<DeviceWorkspaceManager> workspace;
        std::unique_ptr<RoPEStage> stage;
        std::unique_ptr<llaminar2::testing::MockDeviceContext> ctx;
    };

    StageBundle makeGpuStage(
        const std::vector<float> &q_input,
        const std::vector<float> &k_input,
        StreamT stream,
        const int *position_ids = nullptr)
    {
        const DeviceId device = testDevice();
        auto q = makeTensor(q_input, kSeqLen, kQHeads * kHeadDim);
        auto k = makeTensor(k_input, kSeqLen, kKVHeads * kHeadDim);
        if (!q->ensureOnDevice(device, stream) || !k->ensureOnDevice(device, stream))
            throw std::runtime_error("Failed to upload RoPE graph-capture tensors");
        synchronize(stream);

        RoPEStage::Params params{};
        params.device_id = device;
        params.Q = q.get();
        params.K = k.get();
        params.n_heads = kQHeads;
        params.n_kv_heads = kKVHeads;
        params.head_dim = kHeadDim;
        params.seq_len = kSeqLen;
        params.theta_base = kThetaBase;
        params.position_ids = position_ids;

        auto stage = std::make_unique<RoPEStage>(params);
        stage->setGPUStream(stream);

        auto *consumer = stage->getKernelAsWorkspaceConsumer();
        if (!consumer)
            throw std::runtime_error("RoPE kernel does not expose workspace binding");
        const auto requirements = consumer->getWorkspaceRequirements(kSeqLen, kQHeads * kHeadDim, 0);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            device,
            requirements.total_bytes_with_alignment() + 4096);
        if (!workspace->allocate(requirements))
            throw std::runtime_error("Failed to allocate RoPE graph-capture workspace");
        consumer->bindWorkspace(workspace.get());

        auto ctx = std::make_unique<llaminar2::testing::MockDeviceContext>(device, kBackendType);
        return StageBundle{std::move(q), std::move(k), std::move(workspace), std::move(stage), std::move(ctx)};
    }

#endif
}

TEST(Test__RoPEGraphCaptureNoH2D, BackToBackScalarPublicationsPreserveEachQueuedValue)
{
#if !defined(GPU_CONTEXT_TEST_BACKEND_CUDA) && !defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    GTEST_SKIP() << "No GPU graph-capture backend selected";
#else
    if (!hasDevice())
        GTEST_SKIP() << "No " << backendName() << " device available";
    setDevice();

    const auto q_input = makeValues(kSeqLen, kQHeads * kHeadDim, 0.25f);
    const auto k_input = makeValues(kSeqLen, kKVHeads * kHeadDim, -0.125f);
    constexpr int first_pos_offset = 7;
    constexpr int second_pos_offset = 19;

    StreamT graph_stream{};
    StreamT blocker_stream{};
    EventT release_event{};
    createStream(&graph_stream);
    createStream(&blocker_stream);
    createEvent(&release_event);

    auto bundle = makeGpuStage(q_input, k_input, graph_stream);
    auto *device_params = bundle.workspace->getBuffer(RoPEWorkspaceBuffers::DEVICE_PARAMS);
    ASSERT_NE(device_params, nullptr);

    void *first_snapshot = nullptr;
    void *second_snapshot = nullptr;
    void *blocker_buffer = nullptr;
    allocateDevice(&first_snapshot, sizeof(rope::RoPEDeviceParams));
    allocateDevice(&second_snapshot, sizeof(rope::RoPEDeviceParams));
    constexpr size_t blocker_bytes = 128ULL * 1024ULL * 1024ULL;
    constexpr int blocker_passes = 2048;
    allocateDevice(&blocker_buffer, blocker_bytes);

    /*
     * Hold the graph stream behind a finite device workload before either
     * publication can execute. Both publications are therefore completely
     * queued before device work starts.
     *
     * A mutable pinned H2D source collapses both queued copies to the second
     * value; launch-by-value device publication preserves the two values.
     */
    for (int pass = 0; pass < blocker_passes; ++pass)
    {
        fillDevice(blocker_buffer, blocker_bytes, blocker_stream);
    }
    recordEvent(release_event, blocker_stream);
    ASSERT_FALSE(eventComplete(release_event))
        << "The publication gate completed before both updates could be queued";
    waitEvent(graph_stream, release_event);

    bundle.stage->updateDynamicParams(first_pos_offset, kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), graph_stream));
    copyDevice(first_snapshot, device_params, sizeof(rope::RoPEDeviceParams), graph_stream);

    bundle.stage->updateDynamicParams(second_pos_offset, kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), graph_stream));
    copyDevice(second_snapshot, device_params, sizeof(rope::RoPEDeviceParams), graph_stream);

    rope::RoPEDeviceParams first_result{};
    rope::RoPEDeviceParams second_result{};
    download(&first_result, first_snapshot, sizeof(first_result), graph_stream);
    download(&second_result, second_snapshot, sizeof(second_result), graph_stream);
    synchronize(graph_stream);

    EXPECT_EQ(first_result.pos_offset, first_pos_offset);
    EXPECT_EQ(second_result.pos_offset, second_pos_offset);

    freeDevice(blocker_buffer);
    freeDevice(second_snapshot);
    freeDevice(first_snapshot);
    destroyEvent(release_event);
    destroyStream(blocker_stream);
    destroyStream(graph_stream);
#endif
}

TEST(Test__RoPEGraphCaptureNoH2D, ContiguousPositionsReplayWithUpdatedDeviceScalar)
{
#if !defined(GPU_CONTEXT_TEST_BACKEND_CUDA) && !defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    GTEST_SKIP() << "No GPU graph-capture backend selected";
#else
    if (!hasDevice())
        GTEST_SKIP() << "No " << backendName() << " device available";
    setDevice();

    const auto q_input = makeValues(kSeqLen, kQHeads * kHeadDim, 0.25f);
    const auto k_input = makeValues(kSeqLen, kKVHeads * kHeadDim, -0.125f);
    constexpr int first_pos_offset = 7;
    constexpr int second_pos_offset = 19;

    StreamT stream{};
    createStream(&stream);

    auto bundle = makeGpuStage(q_input, k_input, stream);
    bundle.stage->updateDynamicParams(first_pos_offset, kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), stream));

    // Warmup initializes invariant device tables outside graph capture.
    ASSERT_TRUE(bundle.stage->execute(bundle.ctx.get()));
    synchronize(stream);
    uploadTensor(*bundle.q, q_input, stream);
    uploadTensor(*bundle.k, k_input, stream);

    GraphT graph{};
    GraphExecT graph_exec{};
    {
        GraphCaptureGuard guard;
        beginCapture(stream);
        ASSERT_TRUE(bundle.stage->execute(bundle.ctx.get()));
        endCapture(stream, &graph);
    }
    ASSERT_NE(graph, nullptr);
    instantiate(&graph_exec, graph);

    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), stream));
    launch(graph_exec, stream);
    synchronize(stream);
    const auto first_ref = computeCpuReference(q_input, k_input, first_pos_offset);
    expectNear(downloadTensor(*bundle.q, q_input.size(), stream), first_ref.q);
    expectNear(downloadTensor(*bundle.k, k_input.size(), stream), first_ref.k);

    uploadTensor(*bundle.q, q_input, stream);
    uploadTensor(*bundle.k, k_input, stream);
    bundle.stage->updateDynamicParams(second_pos_offset, kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), stream));
    launch(graph_exec, stream);
    synchronize(stream);
    const auto second_ref = computeCpuReference(q_input, k_input, second_pos_offset);
    expectNear(downloadTensor(*bundle.q, q_input.size(), stream), second_ref.q);
    expectNear(downloadTensor(*bundle.k, k_input.size(), stream), second_ref.k);

    destroyGraphExec(graph_exec);
    destroyGraph(graph);
    destroyStream(stream);
#endif
}

TEST(Test__RoPEGraphCaptureNoH2D, ExplicitPositionIdsReplayWithUpdatedDeviceRows)
{
#if !defined(GPU_CONTEXT_TEST_BACKEND_CUDA) && !defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
    GTEST_SKIP() << "No GPU graph-capture backend selected";
#else
    if (!hasDevice())
        GTEST_SKIP() << "No " << backendName() << " device available";
    setDevice();

    const auto q_input = makeValues(kSeqLen, kQHeads * kHeadDim, 0.5f);
    const auto k_input = makeValues(kSeqLen, kKVHeads * kHeadDim, 0.125f);
    const std::vector<int> first_positions{7, 8, 9, 10, 11, 12};
    const std::vector<int> second_positions{19, 19, 23, 23, 31, 32};

    StreamT stream{};
    createStream(&stream);

    auto bundle = makeGpuStage(q_input, k_input, stream);

    /*
     * Request-batched MTP may pass explicit rows that are numerically
     * contiguous for one launch and non-contiguous for the next.  Once a stage
     * receives explicit position IDs, graph capture must keep using the
     * workspace row-buffer path instead of silently switching back to scalar
     * pos_offset metadata.
     */
    bundle.stage->updateDynamicPositionIds(first_positions.data(), kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), stream));
    synchronize(stream);

    // Warmup initializes invariant device tables outside graph capture.
    ASSERT_TRUE(bundle.stage->execute(bundle.ctx.get()));
    synchronize(stream);
    uploadTensor(*bundle.q, q_input, stream);
    uploadTensor(*bundle.k, k_input, stream);

    GraphT graph{};
    GraphExecT graph_exec{};
    {
        GraphCaptureGuard guard;
        beginCapture(stream);
        ASSERT_TRUE(bundle.stage->execute(bundle.ctx.get()));
        endCapture(stream, &graph);
    }
    ASSERT_NE(graph, nullptr);
    instantiate(&graph_exec, graph);

    launch(graph_exec, stream);
    synchronize(stream);
    const auto first_ref = computeCpuReference(q_input, k_input, first_positions.front(), &first_positions);
    expectNear(downloadTensor(*bundle.q, q_input.size(), stream), first_ref.q);
    expectNear(downloadTensor(*bundle.k, k_input.size(), stream), first_ref.k);

    uploadTensor(*bundle.q, q_input, stream);
    uploadTensor(*bundle.k, k_input, stream);
    bundle.stage->updateDynamicPositionIds(second_positions.data(), kSeqLen);
    ASSERT_TRUE(bundle.stage->prepareGraphLaunch(bundle.ctx.get(), stream));
    launch(graph_exec, stream);
    synchronize(stream);
    const auto second_ref = computeCpuReference(q_input, k_input, second_positions.front(), &second_positions);
    expectNear(downloadTensor(*bundle.q, q_input.size(), stream), second_ref.q);
    expectNear(downloadTensor(*bundle.k, k_input.size(), stream), second_ref.k);

    destroyGraphExec(graph_exec);
    destroyGraph(graph);
    destroyStream(stream);
#endif
}
