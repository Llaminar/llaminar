/**
 * @file Test__LocalTPNCCLGraphCapture.cpp
 * @brief CUDA-only LocalTP graph capture regression and timing probes.
 *
 * Keep this in a separate translation unit from ROCm tests: cuda_runtime.h and
 * hip_runtime.h expose overlapping vector types and are intentionally not mixed.
 */

#include <gtest/gtest.h>

#ifdef HAVE_CUDA

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/IBackend.h"
#include "collective/ICollectiveBackend.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"

using namespace llaminar2;

namespace
{
    class Barrier
    {
    public:
        explicit Barrier(int expected) : expected_(expected) {}

        void arriveAndWait()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const int generation = generation_;
            if (++arrived_ == expected_)
            {
                arrived_ = 0;
                ++generation_;
                cv_.notify_all();
                return;
            }
            cv_.wait(lock, [&] { return generation_ != generation; });
        }

    private:
        const int expected_;
        int arrived_ = 0;
        int generation_ = 0;
        std::mutex mutex_;
        std::condition_variable cv_;
    };

    struct CaptureResult
    {
        bool collective_ok = false;
        cudaError_t begin_status = cudaSuccess;
        cudaError_t launch_status = cudaSuccess;
        cudaError_t end_status = cudaSuccess;
        cudaError_t instantiate_status = cudaSuccess;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
    };

    struct CapturedP2PGraph
    {
        CaptureResult result0;
        CaptureResult result1;
        cudaStream_t capture_stream0 = nullptr;
        cudaStream_t capture_stream1 = nullptr;
        cudaStream_t transfer_stream0 = nullptr;
        cudaStream_t transfer_stream1 = nullptr;
        cudaEvent_t transfer_ready0 = nullptr;
        cudaEvent_t transfer_ready1 = nullptr;
        cudaEvent_t transfer_done0 = nullptr;
        cudaEvent_t transfer_done1 = nullptr;
    };

    struct P2PTimingResult
    {
        bool ok = true;
        double wall_ms = 0.0;
        cudaError_t first_error = cudaSuccess;
        std::string step;

        void record(cudaError_t status, std::string where)
        {
            if (status == cudaSuccess || !ok)
                return;
            ok = false;
            first_error = status;
            step = std::move(where);
        }
    };

    template <typename T>
    void allocateAndUpload(int device, const std::vector<T> &host_values, T **device_ptr)
    {
        ASSERT_NE(device_ptr, nullptr);
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        ASSERT_EQ(cudaMalloc(reinterpret_cast<void **>(device_ptr),
                             host_values.size() * sizeof(T)),
                  cudaSuccess);
        ASSERT_NE(*device_ptr, nullptr);

        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(*device_ptr,
                                  host_values.data(),
                                  host_values.size() * sizeof(T),
                                  cudaMemcpyHostToDevice,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    void freeDevicePtr(int device, void *ptr)
    {
        if (!ptr)
            return;
        EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
        EXPECT_EQ(cudaFree(ptr), cudaSuccess);
    }

    template <typename T>
    void downloadDeviceVector(int device, const T *device_ptr, std::vector<T> *host_values)
    {
        ASSERT_NE(device_ptr, nullptr);
        ASSERT_NE(host_values, nullptr);
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaMemcpyAsync(host_values->data(),
                                  device_ptr,
                                  host_values->size() * sizeof(T),
                                  cudaMemcpyDeviceToHost,
                                  stream),
                  cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    void destroyCaptureResult(CaptureResult &result)
    {
        if (result.exec)
        {
            EXPECT_EQ(cudaGraphExecDestroy(result.exec), cudaSuccess);
            result.exec = nullptr;
        }
        if (result.graph)
        {
            EXPECT_EQ(cudaGraphDestroy(result.graph), cudaSuccess);
            result.graph = nullptr;
        }
    }

    void createP2PStreamsAndEvents(CapturedP2PGraph &graph)
    {
        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&graph.capture_stream0, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&graph.transfer_stream0, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&graph.transfer_ready0, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&graph.transfer_done0, cudaEventDisableTiming), cudaSuccess);

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&graph.capture_stream1, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&graph.transfer_stream1, cudaStreamNonBlocking), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&graph.transfer_ready1, cudaEventDisableTiming), cudaSuccess);
        ASSERT_EQ(cudaEventCreateWithFlags(&graph.transfer_done1, cudaEventDisableTiming), cudaSuccess);
    }

    void destroyP2PStreamsAndEvents(CapturedP2PGraph &graph)
    {
        destroyCaptureResult(graph.result0);
        destroyCaptureResult(graph.result1);

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        if (graph.transfer_ready0)
        {
            EXPECT_EQ(cudaEventDestroy(graph.transfer_ready0), cudaSuccess);
            graph.transfer_ready0 = nullptr;
        }
        if (graph.transfer_done0)
        {
            EXPECT_EQ(cudaEventDestroy(graph.transfer_done0), cudaSuccess);
            graph.transfer_done0 = nullptr;
        }
        if (graph.capture_stream0)
        {
            EXPECT_EQ(cudaStreamDestroy(graph.capture_stream0), cudaSuccess);
            graph.capture_stream0 = nullptr;
        }
        if (graph.transfer_stream0)
        {
            EXPECT_EQ(cudaStreamDestroy(graph.transfer_stream0), cudaSuccess);
            graph.transfer_stream0 = nullptr;
        }

        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        if (graph.transfer_ready1)
        {
            EXPECT_EQ(cudaEventDestroy(graph.transfer_ready1), cudaSuccess);
            graph.transfer_ready1 = nullptr;
        }
        if (graph.transfer_done1)
        {
            EXPECT_EQ(cudaEventDestroy(graph.transfer_done1), cudaSuccess);
            graph.transfer_done1 = nullptr;
        }
        if (graph.capture_stream1)
        {
            EXPECT_EQ(cudaStreamDestroy(graph.capture_stream1), cudaSuccess);
            graph.capture_stream1 = nullptr;
        }
        if (graph.transfer_stream1)
        {
            EXPECT_EQ(cudaStreamDestroy(graph.transfer_stream1), cudaSuccess);
            graph.transfer_stream1 = nullptr;
        }
    }

    void captureMaintenanceGroupedP2PGraph(
        ILocalTPContext &ctx,
        const std::vector<CollectiveP2POp> &ops0,
        const std::vector<CollectiveP2POp> &ops1,
        CapturedP2PGraph &graph)
    {
        Barrier ready_to_capture(2);
        Barrier captured_collective(2);

        auto capture_worker = [&](int device,
                                  const std::vector<CollectiveP2POp> *ops,
                                  cudaStream_t capture_stream,
                                  cudaStream_t transfer_stream,
                                  cudaEvent_t transfer_ready,
                                  cudaEvent_t transfer_done,
                                  CaptureResult *result)
        {
            result->begin_status = cudaSetDevice(device);
            if (result->begin_status == cudaSuccess)
            {
                result->begin_status = cudaStreamBeginCapture(
                    capture_stream,
                    cudaStreamCaptureModeRelaxed);
            }

            ready_to_capture.arriveAndWait();

            if (result->begin_status == cudaSuccess)
            {
                GraphCaptureGuard guard;
                result->launch_status = cudaEventRecord(
                    transfer_ready,
                    capture_stream);
                if (result->launch_status == cudaSuccess)
                {
                    result->launch_status = cudaStreamWaitEvent(
                        transfer_stream,
                        transfer_ready,
                        0);
                }

                if (result->launch_status == cudaSuccess)
                {
                    result->collective_ok = ctx.groupedP2PRawOnStream(
                        *ops,
                        device,
                        transfer_stream,
                        "nccl_p2p_aux_stream_capture_probe");
                }

                if (result->launch_status == cudaSuccess &&
                    result->collective_ok)
                {
                    result->launch_status = cudaEventRecord(
                        transfer_done,
                        transfer_stream);
                    if (result->launch_status == cudaSuccess)
                    {
                        result->launch_status = cudaStreamWaitEvent(
                            capture_stream,
                            transfer_done,
                            0);
                    }
                }
            }

            captured_collective.arriveAndWait();

            if (result->begin_status == cudaSuccess &&
                result->launch_status == cudaSuccess &&
                result->collective_ok)
            {
                result->end_status = cudaSetDevice(device);
                if (result->end_status == cudaSuccess)
                {
                    result->end_status = cudaStreamEndCapture(
                        capture_stream,
                        &result->graph);
                }
            }

            if (result->end_status == cudaSuccess && result->graph)
            {
                result->instantiate_status = cudaSetDevice(device);
                if (result->instantiate_status == cudaSuccess)
                {
                    result->instantiate_status = cudaGraphInstantiate(
                        &result->exec,
                        result->graph,
                        nullptr,
                        nullptr,
                        0);
                }
            }
        };

        std::thread t0(capture_worker,
                       0,
                       &ops0,
                       graph.capture_stream0,
                       graph.transfer_stream0,
                       graph.transfer_ready0,
                       graph.transfer_done0,
                       &graph.result0);
        std::thread t1(capture_worker,
                       1,
                       &ops1,
                       graph.capture_stream1,
                       graph.transfer_stream1,
                       graph.transfer_ready1,
                       graph.transfer_done1,
                       &graph.result1);
        t0.join();
        t1.join();
    }

    P2PTimingResult replayP2PGraphPair(
        CapturedP2PGraph &graph,
        int iterations)
    {
        P2PTimingResult result;
        Barrier replay_iteration(2);

        auto replay_worker = [&](int device,
                                 cudaStream_t capture_stream,
                                 cudaStream_t transfer_stream,
                                 const CaptureResult *capture,
                                 P2PTimingResult *local_result)
        {
            auto record = [&](cudaError_t status, const std::string &where)
            {
                local_result->record(status, "device " + std::to_string(device) + " " + where);
                return status == cudaSuccess;
            };

            record(cudaSetDevice(device), "set device");
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
            {
                replay_iteration.arriveAndWait();
                if (local_result->ok)
                {
                    record(cudaGraphLaunch(capture->exec, capture_stream),
                           "launch p2p graph");
                }
            }
            if (local_result->ok)
                record(cudaStreamSynchronize(capture_stream), "sync capture stream");
            if (local_result->ok)
                record(cudaStreamSynchronize(transfer_stream), "sync transfer stream");
            const auto end = std::chrono::steady_clock::now();
            local_result->wall_ms =
                std::chrono::duration<double, std::milli>(end - start).count();
        };

        P2PTimingResult result0;
        P2PTimingResult result1;
        std::thread t0(replay_worker,
                       0,
                       graph.capture_stream0,
                       graph.transfer_stream0,
                       &graph.result0,
                       &result0);
        std::thread t1(replay_worker,
                       1,
                       graph.capture_stream1,
                       graph.transfer_stream1,
                       &graph.result1,
                       &result1);
        t0.join();
        t1.join();

        if (!result0.ok)
            return result0;
        if (!result1.ok)
            return result1;
        result.wall_ms = std::max(result0.wall_ms, result1.wall_ms);
        return result;
    }

    void expectCapturedGraphReady(const CaptureResult &result, const char *name)
    {
        EXPECT_EQ(result.begin_status, cudaSuccess) << name;
        EXPECT_TRUE(result.collective_ok) << name;
        EXPECT_EQ(result.end_status, cudaSuccess) << name;
        EXPECT_NE(result.graph, nullptr) << name;
        EXPECT_EQ(result.instantiate_status, cudaSuccess) << name;
        EXPECT_NE(result.exec, nullptr) << name;
    }
} // namespace

TEST(Test__LocalTPNCCLGraphCapture, NCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    constexpr size_t payload_bytes = 4 * 1024 * 1024;
    constexpr int replay_iterations = 6;

    int8_t *send0 = nullptr;
    int8_t *send1 = nullptr;
    int8_t *recv0 = nullptr;
    int8_t *recv1 = nullptr;
    allocateAndUpload<int8_t>(0, std::vector<int8_t>(payload_bytes, 3), &send0);
    allocateAndUpload<int8_t>(1, std::vector<int8_t>(payload_bytes, 5), &send1);
    allocateAndUpload<int8_t>(0, std::vector<int8_t>(payload_bytes, -1), &recv0);
    allocateAndUpload<int8_t>(1, std::vector<int8_t>(payload_bytes, -1), &recv1);

    const std::vector<CollectiveP2POp> ops0 = {
        CollectiveP2POp{CollectiveP2POpKind::Send, send0, nullptr, payload_bytes, CollectiveDataType::INT8, 1},
        CollectiveP2POp{CollectiveP2POpKind::Recv, nullptr, recv0, payload_bytes, CollectiveDataType::INT8, 1},
    };
    const std::vector<CollectiveP2POp> ops1 = {
        CollectiveP2POp{CollectiveP2POpKind::Send, send1, nullptr, payload_bytes, CollectiveDataType::INT8, 0},
        CollectiveP2POp{CollectiveP2POpKind::Recv, nullptr, recv1, payload_bytes, CollectiveDataType::INT8, 0},
    };

    CapturedP2PGraph aux_graph;
    createP2PStreamsAndEvents(aux_graph);

    captureMaintenanceGroupedP2PGraph(
        *ctx,
        ops0,
        ops1,
        aux_graph);

    expectCapturedGraphReady(aux_graph.result0, "aux_graph0");
    expectCapturedGraphReady(aux_graph.result1, "aux_graph1");
    ASSERT_FALSE(::testing::Test::HasFailure());

    const P2PTimingResult aux_timing =
        replayP2PGraphPair(aux_graph, replay_iterations);

    EXPECT_TRUE(aux_timing.ok) << aux_timing.step
                               << " cudaError=" << cudaGetErrorString(aux_timing.first_error);
    std::cout << "[NCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe]"
              << " payload_bytes=" << payload_bytes
              << " iterations=" << replay_iterations
              << " aux_wall_ms=" << aux_timing.wall_ms
              << std::endl;

    std::vector<int8_t> recv_host0(payload_bytes);
    std::vector<int8_t> recv_host1(payload_bytes);
    cudaStream_t verify_stream0 = nullptr;
    cudaStream_t verify_stream1 = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&verify_stream0, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(recv_host0.data(),
                              recv0,
                              recv_host0.size() * sizeof(int8_t),
                              cudaMemcpyDeviceToHost,
                              verify_stream0),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(verify_stream0), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(verify_stream0), cudaSuccess);

    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&verify_stream1, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(recv_host1.data(),
                              recv1,
                              recv_host1.size() * sizeof(int8_t),
                              cudaMemcpyDeviceToHost,
                              verify_stream1),
              cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(verify_stream1), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(verify_stream1), cudaSuccess);

    EXPECT_TRUE(std::all_of(recv_host0.begin(), recv_host0.end(), [](int8_t v) { return v == 5; }));
    EXPECT_TRUE(std::all_of(recv_host1.begin(), recv_host1.end(), [](int8_t v) { return v == 3; }));

    destroyP2PStreamsAndEvents(aux_graph);
    freeDevicePtr(0, send0);
    freeDevicePtr(1, send1);
    freeDevicePtr(0, recv0);
    freeDevicePtr(1, recv1);
}

TEST(Test__LocalTPNCCLGraphCapture, NCCLRawAllgather_OnStreamGraphCapture_ReplaysCorrectly)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found " << cuda_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    constexpr size_t int32_count = 128;
    constexpr size_t fp32_count = 257;
    constexpr size_t histogram_u64_count = 4096;
    static_assert(sizeof(uint64_t) == 2 * sizeof(int32_t));
    auto make_histogram_payload = [](uint64_t base)
    {
        std::vector<uint64_t> values(histogram_u64_count);
        for (size_t i = 0; i < values.size(); ++i)
        {
            values[i] = base + static_cast<uint64_t>(i);
        }
        return values;
    };
    const std::vector<uint64_t> send_hist_host0 =
        make_histogram_payload(0x100000000ULL + 17ULL);
    const std::vector<uint64_t> send_hist_host1 =
        make_histogram_payload(0x200000000ULL + 29ULL);
    int32_t *send_i32_0 = nullptr;
    int32_t *send_i32_1 = nullptr;
    int32_t *recv_i32_0 = nullptr;
    int32_t *recv_i32_1 = nullptr;
    float *send_fp32_0 = nullptr;
    float *send_fp32_1 = nullptr;
    float *recv_fp32_0 = nullptr;
    float *recv_fp32_1 = nullptr;
    uint64_t *send_hist_0 = nullptr;
    uint64_t *send_hist_1 = nullptr;
    uint64_t *recv_hist_0 = nullptr;
    uint64_t *recv_hist_1 = nullptr;

    allocateAndUpload<int32_t>(0, std::vector<int32_t>(int32_count, 11), &send_i32_0);
    allocateAndUpload<int32_t>(1, std::vector<int32_t>(int32_count, 22), &send_i32_1);
    allocateAndUpload<int32_t>(0, std::vector<int32_t>(int32_count * 2, -1), &recv_i32_0);
    allocateAndUpload<int32_t>(1, std::vector<int32_t>(int32_count * 2, -1), &recv_i32_1);
    allocateAndUpload<float>(0, std::vector<float>(fp32_count, 1.25f), &send_fp32_0);
    allocateAndUpload<float>(1, std::vector<float>(fp32_count, 2.5f), &send_fp32_1);
    allocateAndUpload<float>(0, std::vector<float>(fp32_count * 2, -1.0f), &recv_fp32_0);
    allocateAndUpload<float>(1, std::vector<float>(fp32_count * 2, -1.0f), &recv_fp32_1);
    allocateAndUpload<uint64_t>(0, send_hist_host0, &send_hist_0);
    allocateAndUpload<uint64_t>(1, send_hist_host1, &send_hist_1);
    allocateAndUpload<uint64_t>(
        0,
        std::vector<uint64_t>(histogram_u64_count * 2, 0xffffffffffffffffULL),
        &recv_hist_0);
    allocateAndUpload<uint64_t>(
        1,
        std::vector<uint64_t>(histogram_u64_count * 2, 0xffffffffffffffffULL),
        &recv_hist_1);

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking), cudaSuccess);

    Barrier ready_to_capture(2);
    Barrier captured_collective(2);
    Barrier ready_to_launch(2);
    CaptureResult result0;
    CaptureResult result1;

    auto capture_worker = [&](int device,
                              const int32_t *send_i32,
                              int32_t *recv_i32,
                              const float *send_fp32,
                              float *recv_fp32,
                              const uint64_t *send_hist,
                              uint64_t *recv_hist,
                              cudaStream_t stream,
                              CaptureResult *result)
    {
        result->begin_status = cudaSetDevice(device);
        if (result->begin_status == cudaSuccess)
        {
            result->begin_status = cudaStreamBeginCapture(
                stream,
                cudaStreamCaptureModeRelaxed);
        }

        ready_to_capture.arriveAndWait();

        if (result->begin_status == cudaSuccess)
        {
            GraphCaptureGuard guard;
            const bool i32_ok = ctx->allgatherRawOnStream(
                send_i32,
                recv_i32,
                int32_count,
                CollectiveDataType::INT32,
                device,
                stream,
                "nccl_raw_allgather_graph_capture_i32");
            const bool fp32_ok = ctx->allgatherRawOnStream(
                send_fp32,
                recv_fp32,
                fp32_count,
                CollectiveDataType::FLOAT32,
                device,
                stream,
                "nccl_raw_allgather_graph_capture_fp32");
            const bool hist_ok = ctx->allgatherRawOnStream(
                send_hist,
                recv_hist,
                histogram_u64_count * 2u,
                CollectiveDataType::INT32,
                device,
                stream,
                "nccl_raw_allgather_graph_capture_histogram_u64_as_i32");
            result->collective_ok = i32_ok && fp32_ok && hist_ok;
        }

        captured_collective.arriveAndWait();

        if (result->begin_status == cudaSuccess &&
            result->collective_ok)
        {
            result->end_status = cudaSetDevice(device);
            if (result->end_status == cudaSuccess)
            {
                result->end_status = cudaStreamEndCapture(
                    stream,
                    &result->graph);
            }
        }

        if (result->end_status == cudaSuccess && result->graph)
        {
            result->instantiate_status = cudaSetDevice(device);
            if (result->instantiate_status == cudaSuccess)
            {
                result->instantiate_status = cudaGraphInstantiate(
                    &result->exec,
                    result->graph,
                    nullptr,
                    nullptr,
                    0);
            }
        }

        for (int replay = 0; replay < 2; ++replay)
        {
            ready_to_launch.arriveAndWait();
            if (result->instantiate_status == cudaSuccess && result->exec)
            {
                result->launch_status = cudaSetDevice(device);
                if (result->launch_status == cudaSuccess)
                    result->launch_status = cudaGraphLaunch(result->exec, stream);
                if (result->launch_status == cudaSuccess)
                    result->launch_status = cudaStreamSynchronize(stream);
            }
        }
    };

    std::thread t0(capture_worker,
                   0,
                   send_i32_0,
                   recv_i32_0,
                   send_fp32_0,
                   recv_fp32_0,
                   send_hist_0,
                   recv_hist_0,
                   stream0,
                   &result0);
    std::thread t1(capture_worker,
                   1,
                   send_i32_1,
                   recv_i32_1,
                   send_fp32_1,
                   recv_fp32_1,
                   send_hist_1,
                   recv_hist_1,
                   stream1,
                   &result1);
    t0.join();
    t1.join();

    expectCapturedGraphReady(result0, "nccl_raw_allgather_graph0");
    expectCapturedGraphReady(result1, "nccl_raw_allgather_graph1");
    EXPECT_EQ(result0.launch_status, cudaSuccess);
    EXPECT_EQ(result1.launch_status, cudaSuccess);

    std::vector<int32_t> recv_i32_host0(int32_count * 2);
    std::vector<int32_t> recv_i32_host1(int32_count * 2);
    std::vector<float> recv_fp32_host0(fp32_count * 2);
    std::vector<float> recv_fp32_host1(fp32_count * 2);
    std::vector<uint64_t> recv_hist_host0(histogram_u64_count * 2);
    std::vector<uint64_t> recv_hist_host1(histogram_u64_count * 2);
    downloadDeviceVector(0, recv_i32_0, &recv_i32_host0);
    downloadDeviceVector(1, recv_i32_1, &recv_i32_host1);
    downloadDeviceVector(0, recv_fp32_0, &recv_fp32_host0);
    downloadDeviceVector(1, recv_fp32_1, &recv_fp32_host1);
    downloadDeviceVector(0, recv_hist_0, &recv_hist_host0);
    downloadDeviceVector(1, recv_hist_1, &recv_hist_host1);

    auto expect_i32_allgather = [&](const std::vector<int32_t> &values, const char *name)
    {
        ASSERT_EQ(values.size(), int32_count * 2) << name;
        EXPECT_TRUE(std::all_of(values.begin(),
                                values.begin() + static_cast<std::ptrdiff_t>(int32_count),
                                [](int32_t v) { return v == 11; }))
            << name;
        EXPECT_TRUE(std::all_of(values.begin() + static_cast<std::ptrdiff_t>(int32_count),
                                values.end(),
                                [](int32_t v) { return v == 22; }))
            << name;
    };
    auto expect_fp32_allgather = [&](const std::vector<float> &values, const char *name)
    {
        ASSERT_EQ(values.size(), fp32_count * 2) << name;
        EXPECT_TRUE(std::all_of(values.begin(),
                                values.begin() + static_cast<std::ptrdiff_t>(fp32_count),
                                [](float v) { return v == 1.25f; }))
            << name;
        EXPECT_TRUE(std::all_of(values.begin() + static_cast<std::ptrdiff_t>(fp32_count),
                                values.end(),
                                [](float v) { return v == 2.5f; }))
            << name;
    };
    expect_i32_allgather(recv_i32_host0, "device0_i32");
    expect_i32_allgather(recv_i32_host1, "device1_i32");
    expect_fp32_allgather(recv_fp32_host0, "device0_fp32");
    expect_fp32_allgather(recv_fp32_host1, "device1_fp32");
    auto expect_histogram_allgather = [&](const std::vector<uint64_t> &values, const char *name)
    {
        ASSERT_EQ(values.size(), histogram_u64_count * 2) << name;
        EXPECT_TRUE(std::equal(send_hist_host0.begin(),
                               send_hist_host0.end(),
                               values.begin()))
            << name;
        EXPECT_TRUE(std::equal(send_hist_host1.begin(),
                               send_hist_host1.end(),
                               values.begin() + static_cast<std::ptrdiff_t>(histogram_u64_count)))
            << name;
    };
    expect_histogram_allgather(recv_hist_host0, "device0_histogram");
    expect_histogram_allgather(recv_hist_host1, "device1_histogram");

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    freeDevicePtr(0, send_i32_0);
    freeDevicePtr(1, send_i32_1);
    freeDevicePtr(0, recv_i32_0);
    freeDevicePtr(1, recv_i32_1);
    freeDevicePtr(0, send_fp32_0);
    freeDevicePtr(1, send_fp32_1);
    freeDevicePtr(0, recv_fp32_0);
    freeDevicePtr(1, recv_fp32_1);
    freeDevicePtr(0, send_hist_0);
    freeDevicePtr(1, send_hist_1);
    freeDevicePtr(0, recv_hist_0);
    freeDevicePtr(1, recv_hist_1);
}

#endif // HAVE_CUDA
