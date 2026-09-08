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
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/IBackend.h"
#include "backends/cuda/CUDAGraphCapture.h"
#include "collective/ICollectiveBackend.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "kernels/common/TPRankOrderedReductionKernels.h"
#include "kernels/cuda/moe/CUDAMoEKernel.h"
#include "tensors/TensorClasses.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

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

    /**
     * @brief Capture one FP16-transport allreduce followed by its first consumer.
     *
     * Qwen3.6 MoE's deferred LocalTP combine writes a full FP32 hidden-state
     * row, transports it through FP16 NCCL scratch, casts the reduced result
     * back to FP32, and immediately lets the next layer consume that FP32
     * buffer. A collective-only test can miss an ordering defect if it
     * synchronizes before inspecting the result. This helper records the
     * immediate device-to-device consumer in the same graph so the copied
     * bytes prove that stream ordering survived capture and replay.
     *
     * All storage must already exist before this function is called. In
     * particular, callers must execute one eager FP16 allreduce first so
     * LocalTPContext's persistent FP16 transport scratch is allocated outside
     * graph capture.
     *
     * @param ctx Shared two-device LocalTP NCCL context.
     * @param tensor0 Device-0 FP32 collective input and output tensor.
     * @param tensor1 Device-1 FP32 collective input and output tensor.
     * @param stream0 Explicit capture stream owned by CUDA device 0.
     * @param stream1 Explicit capture stream owned by CUDA device 1.
     * @param consumer0 Preallocated device-0 destination for the first consumer.
     * @param consumer1 Preallocated device-1 destination for the first consumer.
     * @param count Number of FP32 elements reduced and copied.
     * @param result0 Device-0 captured graph resources and status.
     * @param result1 Device-1 captured graph resources and status.
     */
    void captureFP16AllreduceWithImmediateConsumer(
        ILocalTPContext &ctx,
        TensorBase *tensor0,
        TensorBase *tensor1,
        cudaStream_t stream0,
        cudaStream_t stream1,
        float *consumer0,
        float *consumer1,
        size_t count,
        CaptureResult &result0,
        CaptureResult &result1,
        const std::vector<LocalTPCollectiveSidebandBuffer> *sidebands0 = nullptr,
        const std::vector<LocalTPCollectiveSidebandBuffer> *sidebands1 = nullptr)
    {
        ASSERT_NE(tensor0, nullptr);
        ASSERT_NE(tensor1, nullptr);
        ASSERT_NE(consumer0, nullptr);
        ASSERT_NE(consumer1, nullptr);
        ASSERT_EQ(sidebands0 == nullptr, sidebands1 == nullptr)
            << "Both LocalTP participants must either publish the same sideband "
               "descriptor sequence or omit sidebands together.";

        Barrier capture_started(2);
        Barrier collective_recorded(2);

        auto capture_worker =
            [&](int device,
                TensorBase *tensor,
                cudaStream_t stream,
                float *consumer,
                CaptureResult *result)
        {
            result->begin_status = cudaSetDevice(device);
            if (result->begin_status == cudaSuccess)
            {
                result->begin_status = cudaStreamBeginCapture(
                    stream,
                    cudaStreamCaptureModeRelaxed);
            }

            capture_started.arriveAndWait();

            if (result->begin_status == cudaSuccess)
            {
                GraphCaptureGuard guard;
                if (sidebands0)
                {
                    const auto &participant_sidebands =
                        device == 0 ? *sidebands0 : *sidebands1;
                    result->collective_ok = ctx.allreduceWithSidebandsOnStream(
                        tensor,
                        "layer0_moe_combined_allreduce_with_rebalance_sidebands",
                        count,
                        stream,
                        "fp16",
                        participant_sidebands,
                        device);
                }
                else
                {
                    result->collective_ok = ctx.allreduceOnStream(
                        tensor,
                        "layer0_moe_combined_allreduce",
                        count,
                        stream,
                        "fp16");
                }
                if (result->collective_ok)
                {
                    result->launch_status = cudaMemcpyAsync(
                        consumer,
                        tensor->gpu_data_ptr(),
                        count * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream);
                }
            }

            /*
             * CUDA's participant-local NCCL capture records one operation on
             * each stream. Neither worker may end capture while its peer is
             * still inside the LocalTP grouped-launch rendezvous.
             */
            collective_recorded.arriveAndWait();

            if (result->begin_status == cudaSuccess &&
                result->launch_status == cudaSuccess &&
                result->collective_ok)
            {
                result->end_status = cudaStreamEndCapture(
                    stream,
                    &result->graph);
            }

            if (result->end_status == cudaSuccess && result->graph)
            {
                result->instantiate_status = cudaGraphInstantiate(
                    &result->exec,
                    result->graph,
                    nullptr,
                    nullptr,
                    0);
            }
        };

        std::thread worker0(
            capture_worker,
            0,
            tensor0,
            stream0,
            consumer0,
            &result0);
        std::thread worker1(
            capture_worker,
            1,
            tensor1,
            stream1,
            consumer1,
            &result1);
        worker0.join();
        worker1.join();
    }

    /**
     * @brief Launch one captured LocalTP graph per CUDA participant.
     *
     * RankOrchestrator launches participant graphs from one worker thread per
     * device. The launch barrier below reproduces that ownership and avoids
     * accidentally validating only a sequential host-launch schedule.
     *
     * @param result0 Device-0 captured graph.
     * @param result1 Device-1 captured graph.
     * @param stream0 Device-0 replay stream.
     * @param stream1 Device-1 replay stream.
     */
    void replayCapturedGraphPair(
        CaptureResult &result0,
        CaptureResult &result1,
        cudaStream_t stream0,
        cudaStream_t stream1)
    {
        Barrier launch_ready(2);

        auto replay_worker =
            [&](int device,
                CaptureResult *result,
                cudaStream_t stream)
        {
            result->launch_status = cudaSetDevice(device);
            launch_ready.arriveAndWait();
            if (result->launch_status == cudaSuccess)
                result->launch_status = cudaGraphLaunch(result->exec, stream);
            if (result->launch_status == cudaSuccess)
                result->launch_status = cudaStreamSynchronize(stream);
        };

        std::thread worker0(replay_worker, 0, &result0, stream0);
        std::thread worker1(replay_worker, 1, &result1, stream1);
        worker0.join();
        worker1.join();
    }
} // namespace

/**
 * @brief Inventory graph-captured NCCL kernels through their driver identities.
 *
 * NCCL may publish CUDA graph kernel nodes as `CUkernel` objects rather than
 * legacy runtime `cudaFunction_t` handles. Runtime-only parameter inspection
 * rejects those valid nodes with `cudaErrorInvalidDeviceFunction`, which once
 * made opt-in production graph inventory abort before the first MTP replay.
 * This test captures one real participant-local NCCL operation on each GPU and
 * proves the backend inspector accepts and names every resulting kernel.
 */
TEST(
    Test__LocalTPNCCLGraphCapture,
    KernelInventoryAcceptsDriverOriginNCCLNodes)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found "
                     << cuda_backend->deviceCount();
    }

    constexpr size_t kElementCount = 2048;
    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({kElementCount});
    auto tensor1 = TestTensorFactory::createFP32({kElementCount});
    TestTensorFactory::fillValue(tensor0.get(), 1.0f);
    TestTensorFactory::fillValue(tensor1.get(), 2.0f);
    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    std::array<cudaStream_t, 2> streams{nullptr, nullptr};
    for (int device = 0; device < 2; ++device)
    {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        ASSERT_EQ(
            cudaStreamCreateWithFlags(
                &streams[static_cast<size_t>(device)],
                cudaStreamNonBlocking),
            cudaSuccess);
    }

    /*
     * The eager operation provisions LocalTP's persistent transport scratch.
     * The subsequent capture must contain no allocation or workspace binding.
     */
    std::array<bool, 2> warmup_ok{false, false};
    std::thread warmup0([&]
                        {
                            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
                            warmup_ok[0] = ctx->allreduceOnStream(
                                tensor0.get(),
                                "inventory_nccl_warmup",
                                kElementCount,
                                streams[0],
                                "fp32");
                        });
    std::thread warmup1([&]
                        {
                            ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
                            warmup_ok[1] = ctx->allreduceOnStream(
                                tensor1.get(),
                                "inventory_nccl_warmup",
                                kElementCount,
                                streams[1],
                                "fp32");
                        });
    warmup0.join();
    warmup1.join();
    ASSERT_TRUE(warmup_ok[0]);
    ASSERT_TRUE(warmup_ok[1]);
    for (int device = 0; device < 2; ++device)
    {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        ASSERT_EQ(
            cudaStreamSynchronize(streams[static_cast<size_t>(device)]),
            cudaSuccess);
    }

    std::array<std::unique_ptr<CUDAGraphCapture>, 2> captures{
        std::make_unique<CUDAGraphCapture>(streams[0], 0),
        std::make_unique<CUDAGraphCapture>(streams[1], 1)};
    std::array<bool, 2> begin_ok{false, false};
    std::array<bool, 2> collective_ok{false, false};
    std::array<bool, 2> end_ok{false, false};
    Barrier capture_started(2);
    Barrier collective_recorded(2);

    auto capture_participant = [&](int device, TensorBase *tensor)
    {
        const size_t participant = static_cast<size_t>(device);
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        begin_ok[participant] = captures[participant]->beginCapture();
        capture_started.arriveAndWait();
        if (begin_ok[participant])
        {
            GraphCaptureGuard guard;
            collective_ok[participant] = ctx->allreduceOnStream(
                tensor,
                "inventory_nccl_captured",
                kElementCount,
                streams[participant],
                "fp32");
        }
        collective_recorded.arriveAndWait();
        if (begin_ok[participant] && collective_ok[participant])
            end_ok[participant] = captures[participant]->endCapture();
    };

    std::thread capture0(capture_participant, 0, tensor0.get());
    std::thread capture1(capture_participant, 1, tensor1.get());
    capture0.join();
    capture1.join();

    for (size_t participant = 0; participant < captures.size(); ++participant)
    {
        ASSERT_TRUE(begin_ok[participant]);
        ASSERT_TRUE(collective_ok[participant]);
        ASSERT_TRUE(end_ok[participant]);

        std::vector<GPUGraphKernelNodeInfo> kernels;
        std::string error;
        ASSERT_TRUE(captures[participant]->inspectKernelNodes(kernels, &error))
            << error;
        ASSERT_FALSE(kernels.empty());
        for (const GPUGraphKernelNodeInfo &kernel : kernels)
        {
            EXPECT_TRUE(kernel.valid()) << kernel.graph_path;
            EXPECT_TRUE(kernel.name_resolved)
                << "NCCL kernel identity was not resolved at "
                << kernel.graph_path;
            EXPECT_GT(kernel.registers_per_thread, 0U);
            EXPECT_GT(kernel.max_threads_per_block, 0U);
            EXPECT_GT(kernel.max_active_blocks_per_sm, 0U);
        }
    }

    captures = {};
    for (int device = 0; device < 2; ++device)
    {
        EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
        EXPECT_EQ(
            cudaStreamDestroy(streams[static_cast<size_t>(device)]),
            cudaSuccess);
    }
}

/**
 * @brief Captured FP16 NCCL output must be live for its immediate consumer.
 *
 * This regression models the deferred Qwen3.6 MoE branch topology that first
 * exposed stale continuation state:
 *
 * `local routed+shared FP32 -> FP16 NCCL allreduce -> FP32 -> next layer`
 *
 * Two replays use different integer-valued inputs. Integer values are exactly
 * representable in FP16, so any mismatch is lifecycle or ordering corruption,
 * not an allowed transport-rounding difference. The second replay is
 * essential: a graph that accidentally republishes warmup bytes can pass a
 * one-shot test while still poisoning production continuation state.
 */
TEST(
    Test__LocalTPNCCLGraphCapture,
    NCCLFP16AllreduceGraph_ImmediateConsumerObservesEveryReplay)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found "
                     << cuda_backend->deviceCount();
    }

    constexpr size_t kPromptRows = 9;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    constexpr size_t kElementCount =
        kPromptRows * kQwen36MoEHiddenDim;

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    auto tensor0 = TestTensorFactory::createFP32({kElementCount});
    auto tensor1 = TestTensorFactory::createFP32({kElementCount});
    TestTensorFactory::fillValue(tensor0.get(), 1.0f);
    TestTensorFactory::fillValue(tensor1.get(), 2.0f);
    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    float *consumer0 = nullptr;
    float *consumer1 = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&consumer0),
            kElementCount * sizeof(float)),
        cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&consumer1),
            kElementCount * sizeof(float)),
        cudaSuccess);

    /*
     * Allocate the LocalTPContext FP16 scratch before capture. Production does
     * the same during graph warmup; no allocation or deallocation is permitted
     * in the captured hot path.
     */
    bool warmup0 = false;
    bool warmup1 = false;
    std::thread warmup_worker0(
        [&]
        {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            warmup0 = ctx->allreduceOnStream(
                tensor0.get(),
                "warmup_moe_combined_allreduce",
                kElementCount,
                stream0,
                "fp16");
        });
    std::thread warmup_worker1(
        [&]
        {
            ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
            warmup1 = ctx->allreduceOnStream(
                tensor1.get(),
                "warmup_moe_combined_allreduce",
                kElementCount,
                stream1,
                "fp16");
        });
    warmup_worker0.join();
    warmup_worker1.join();
    ASSERT_TRUE(warmup0);
    ASSERT_TRUE(warmup1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);

    CaptureResult result0;
    CaptureResult result1;
    captureFP16AllreduceWithImmediateConsumer(
        *ctx,
        tensor0.get(),
        tensor1.get(),
        stream0,
        stream1,
        consumer0,
        consumer1,
        kElementCount,
        result0,
        result1);
    expectCapturedGraphReady(result0, "fp16_allreduce_graph0");
    expectCapturedGraphReady(result1, "fp16_allreduce_graph1");
    ASSERT_FALSE(::testing::Test::HasFailure());

    auto replay_and_expect =
        [&](float value0, float value1)
    {
        const std::vector<float> input0(kElementCount, value0);
        const std::vector<float> input1(kElementCount, value1);
        std::vector<float> output0(kElementCount, 0.0f);
        std::vector<float> output1(kElementCount, 0.0f);

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                tensor0->gpu_data_ptr(),
                input0.data(),
                kElementCount * sizeof(float),
                cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                tensor1->gpu_data_ptr(),
                input1.data(),
                kElementCount * sizeof(float),
                cudaMemcpyHostToDevice),
            cudaSuccess);

        replayCapturedGraphPair(
            result0,
            result1,
            stream0,
            stream1);
        ASSERT_EQ(result0.launch_status, cudaSuccess)
            << cudaGetErrorString(result0.launch_status);
        ASSERT_EQ(result1.launch_status, cudaSuccess)
            << cudaGetErrorString(result1.launch_status);

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                output0.data(),
                consumer0,
                kElementCount * sizeof(float),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                output1.data(),
                consumer1,
                kElementCount * sizeof(float),
                cudaMemcpyDeviceToHost),
            cudaSuccess);

        const float expected = value0 + value1;
        for (size_t element = 0; element < kElementCount; ++element)
        {
            ASSERT_FLOAT_EQ(output0[element], expected)
                << "device0 immediate consumer mismatch at element "
                << element;
            ASSERT_FLOAT_EQ(output1[element], expected)
                << "device1 immediate consumer mismatch at element "
                << element;
        }
    };

    replay_and_expect(4.0f, 5.0f);
    replay_and_expect(7.0f, 8.0f);

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_EQ(cudaFree(consumer0), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_EQ(cudaFree(consumer1), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
}

/**
 * @brief Captured Dynamic-MoE sidebands must remain in the anchor NCCL bundle.
 *
 * Dynamic expert publication attaches small device-resident control payloads
 * to a normal activation allreduce. The production contract is one grouped
 * NCCL launch containing both the FP16 activation allreduce and every sideband;
 * a participant-local anchor launch is not sufficient because it silently
 * leaves the control payload outside the captured graph.
 *
 * This test uses the same INT32 allgather semantic as the rebalance histogram
 * and command sidebands. It checks the activation's immediate consumer and the
 * gathered sideband after two graph replays with different inputs. The second
 * replay proves that neither output is merely the eager warmup result.
 */
TEST(
    Test__LocalTPNCCLGraphCapture,
    NCCLFP16AllreduceWithAllgatherSidebandGraph_EveryReplayPublishesWholeBundle)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found "
                     << cuda_backend->deviceCount();
    }

    constexpr size_t kPromptRows = 9;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    constexpr size_t kElementCount =
        kPromptRows * kQwen36MoEHiddenDim;
    constexpr size_t kSidebandElementCount = 128;
    constexpr size_t kGatheredSidebandElementCount =
        2 * kSidebandElementCount;

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsCollectiveSidebandOnStreamGraphCapture());

    auto tensor0 = TestTensorFactory::createFP32({kElementCount});
    auto tensor1 = TestTensorFactory::createFP32({kElementCount});
    TestTensorFactory::fillValue(tensor0.get(), 1.0f);
    TestTensorFactory::fillValue(tensor1.get(), 2.0f);
    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::cuda(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::cuda(1)));

    int32_t *sideband_send0 = nullptr;
    int32_t *sideband_send1 = nullptr;
    int32_t *sideband_recv0 = nullptr;
    int32_t *sideband_recv1 = nullptr;
    allocateAndUpload<int32_t>(
        0,
        std::vector<int32_t>(kSidebandElementCount, 11),
        &sideband_send0);
    allocateAndUpload<int32_t>(
        1,
        std::vector<int32_t>(kSidebandElementCount, 22),
        &sideband_send1);
    allocateAndUpload<int32_t>(
        0,
        std::vector<int32_t>(kGatheredSidebandElementCount, -1),
        &sideband_recv0);
    allocateAndUpload<int32_t>(
        1,
        std::vector<int32_t>(kGatheredSidebandElementCount, -1),
        &sideband_recv1);

    const std::vector<LocalTPCollectiveSidebandBuffer> sidebands0 = {
        LocalTPCollectiveSidebandBuffer{
            .kind = LocalTPCollectiveSidebandKind::Allgather,
            .send_buffer = sideband_send0,
            .recv_buffer = sideband_recv0,
            .element_count = kSidebandElementCount,
            .dtype = CollectiveDataType::INT32,
            .root_device_index = 0,
            .name = "moe_rebalance_histogram_sideband"}};
    const std::vector<LocalTPCollectiveSidebandBuffer> sidebands1 = {
        LocalTPCollectiveSidebandBuffer{
            .kind = LocalTPCollectiveSidebandKind::Allgather,
            .send_buffer = sideband_send1,
            .recv_buffer = sideband_recv1,
            .element_count = kSidebandElementCount,
            .dtype = CollectiveDataType::INT32,
            .root_device_index = 0,
            .name = "moe_rebalance_histogram_sideband"}};

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    float *consumer0 = nullptr;
    float *consumer1 = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&consumer0),
            kElementCount * sizeof(float)),
        cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(
        cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking),
        cudaSuccess);
    ASSERT_EQ(
        cudaMalloc(
            reinterpret_cast<void **>(&consumer1),
            kElementCount * sizeof(float)),
        cudaSuccess);

    /*
     * Warmup creates the persistent FP16 transport scratch and exercises the
     * exact eager grouped bundle before capture. No allocation, host transfer,
     * or stream synchronization is then needed inside the captured operation.
     */
    bool warmup0 = false;
    bool warmup1 = false;
    std::thread warmup_worker0(
        [&]
        {
            ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
            warmup0 = ctx->allreduceWithSidebandsOnStream(
                tensor0.get(),
                "warmup_moe_combined_allreduce_with_rebalance_sidebands",
                kElementCount,
                stream0,
                "fp16",
                sidebands0,
                0);
        });
    std::thread warmup_worker1(
        [&]
        {
            ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
            warmup1 = ctx->allreduceWithSidebandsOnStream(
                tensor1.get(),
                "warmup_moe_combined_allreduce_with_rebalance_sidebands",
                kElementCount,
                stream1,
                "fp16",
                sidebands1,
                1);
        });
    warmup_worker0.join();
    warmup_worker1.join();
    ASSERT_TRUE(warmup0);
    ASSERT_TRUE(warmup1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream1), cudaSuccess);

    CaptureResult result0;
    CaptureResult result1;
    captureFP16AllreduceWithImmediateConsumer(
        *ctx,
        tensor0.get(),
        tensor1.get(),
        stream0,
        stream1,
        consumer0,
        consumer1,
        kElementCount,
        result0,
        result1,
        &sidebands0,
        &sidebands1);
    expectCapturedGraphReady(result0, "fp16_sideband_bundle_graph0");
    expectCapturedGraphReady(result1, "fp16_sideband_bundle_graph1");
    ASSERT_FALSE(::testing::Test::HasFailure());

    auto replay_and_expect =
        [&](float value0,
            float value1,
            int32_t sideband_value0,
            int32_t sideband_value1)
    {
        const std::vector<float> input0(kElementCount, value0);
        const std::vector<float> input1(kElementCount, value1);
        const std::vector<int32_t> sideband_input0(
            kSidebandElementCount,
            sideband_value0);
        const std::vector<int32_t> sideband_input1(
            kSidebandElementCount,
            sideband_value1);
        std::vector<float> output0(kElementCount, 0.0f);
        std::vector<float> output1(kElementCount, 0.0f);
        std::vector<int32_t> gathered0(kGatheredSidebandElementCount, 0);
        std::vector<int32_t> gathered1(kGatheredSidebandElementCount, 0);

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                tensor0->gpu_data_ptr(),
                input0.data(),
                kElementCount * sizeof(float),
                cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                sideband_send0,
                sideband_input0.data(),
                kSidebandElementCount * sizeof(int32_t),
                cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                tensor1->gpu_data_ptr(),
                input1.data(),
                kElementCount * sizeof(float),
                cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                sideband_send1,
                sideband_input1.data(),
                kSidebandElementCount * sizeof(int32_t),
                cudaMemcpyHostToDevice),
            cudaSuccess);

        replayCapturedGraphPair(
            result0,
            result1,
            stream0,
            stream1);
        ASSERT_EQ(result0.launch_status, cudaSuccess)
            << cudaGetErrorString(result0.launch_status);
        ASSERT_EQ(result1.launch_status, cudaSuccess)
            << cudaGetErrorString(result1.launch_status);

        ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                output0.data(),
                consumer0,
                kElementCount * sizeof(float),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                gathered0.data(),
                sideband_recv0,
                kGatheredSidebandElementCount * sizeof(int32_t),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                output1.data(),
                consumer1,
                kElementCount * sizeof(float),
                cudaMemcpyDeviceToHost),
            cudaSuccess);
        ASSERT_EQ(
            cudaMemcpy(
                gathered1.data(),
                sideband_recv1,
                kGatheredSidebandElementCount * sizeof(int32_t),
                cudaMemcpyDeviceToHost),
            cudaSuccess);

        const float expected_activation = value0 + value1;
        for (size_t element = 0; element < kElementCount; ++element)
        {
            ASSERT_FLOAT_EQ(output0[element], expected_activation)
                << "device0 immediate consumer mismatch at element "
                << element;
            ASSERT_FLOAT_EQ(output1[element], expected_activation)
                << "device1 immediate consumer mismatch at element "
                << element;
        }
        for (size_t element = 0; element < kSidebandElementCount; ++element)
        {
            ASSERT_EQ(gathered0[element], sideband_value0);
            ASSERT_EQ(
                gathered0[kSidebandElementCount + element],
                sideband_value1);
            ASSERT_EQ(gathered1[element], sideband_value0);
            ASSERT_EQ(
                gathered1[kSidebandElementCount + element],
                sideband_value1);
        }
    };

    replay_and_expect(4.0f, 5.0f, 101, 202);
    replay_and_expect(7.0f, 8.0f, 303, 404);

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_EQ(cudaFree(consumer0), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_EQ(cudaFree(consumer1), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    freeDevicePtr(0, sideband_send0);
    freeDevicePtr(1, sideband_send1);
    freeDevicePtr(0, sideband_recv0);
    freeDevicePtr(1, sideband_recv1);
}

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

/**
 * @test Rooted canonical-route publication is graph-captured and byte exact.
 *
 * Every route slot has exactly one device owner, matching the production
 * apportioned-expert contract. The test reduces those independently rounded
 * slots to participant zero, invokes the production router-order CUDA reducer
 * only on that root, and broadcasts only the compact output. A separate launch
 * of the same production reducer over the complete serial slot bank supplies
 * the byte oracle. Sweeping every verifier row count through depth fifteen,
 * plus M=31, catches count-dependent collective and launch-geometry holes.
 */
TEST(Test__LocalTPNCCLGraphCapture,
     NCCLCanonicalRouteRootedPublication_GraphCaptured_MTotal_ByteExact)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found "
                     << cuda_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices{
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);

    constexpr int kTopK = 8;
    constexpr int kDModel = 257;
    constexpr int kRoot = 0;
    constexpr int kMaximumM = 31;

    std::array<std::unique_ptr<FP32Tensor>, 2> route_slots;
    std::array<std::unique_ptr<FP32Tensor>, 2> compact_outputs;
    std::array<cudaStream_t, 2> streams{nullptr, nullptr};

    for (int participant = 0; participant < 2; ++participant)
    {
        ASSERT_EQ(cudaSetDevice(participant), cudaSuccess);
        ASSERT_EQ(
            cudaStreamCreateWithFlags(
                &streams[static_cast<size_t>(participant)],
                cudaStreamNonBlocking),
            cudaSuccess);
        route_slots[static_cast<size_t>(participant)] =
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(kMaximumM * kTopK),
                    static_cast<size_t>(kDModel)},
                DeviceId::cuda(participant));
        compact_outputs[static_cast<size_t>(participant)] =
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(kMaximumM),
                    static_cast<size_t>(kDModel)},
                DeviceId::cuda(participant));
        ASSERT_TRUE(route_slots[static_cast<size_t>(participant)]
                        ->ensureOnDevice(
                            DeviceId::cuda(participant),
                            streams[static_cast<size_t>(participant)]));
        ASSERT_TRUE(compact_outputs[static_cast<size_t>(participant)]
                        ->ensureOnDevice(
                            DeviceId::cuda(participant),
                            streams[static_cast<size_t>(participant)]));
        ASSERT_EQ(
            cudaStreamSynchronize(
                streams[static_cast<size_t>(participant)]),
            cudaSuccess);
    }
    auto serial_route_slots = std::make_unique<FP32Tensor>(
        std::vector<size_t>{
            static_cast<size_t>(kMaximumM * kTopK),
            static_cast<size_t>(kDModel)},
        DeviceId::cuda(kRoot));
    auto serial_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{
            static_cast<size_t>(kMaximumM),
            static_cast<size_t>(kDModel)},
        DeviceId::cuda(kRoot));
    ASSERT_TRUE(serial_route_slots->ensureOnDevice(
        DeviceId::cuda(kRoot),
        streams[0]));
    ASSERT_TRUE(serial_output->ensureOnDevice(
        DeviceId::cuda(kRoot),
        streams[0]));
    ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);

    CUDAMoEKernel root_reducer(kRoot);
    root_reducer.bindGPUStream(
        ExplicitGPUStream(static_cast<void *>(streams[0])));

    std::vector<int> verifier_rows;
    for (int m = 1; m <= 16; ++m)
        verifier_rows.push_back(m);
    verifier_rows.push_back(31);

    for (const int m : verifier_rows)
    {
        const size_t route_elements =
            static_cast<size_t>(m) * kTopK * kDModel;
        const size_t output_elements =
            static_cast<size_t>(m) * kDModel;
        std::vector<float> serial_host(route_elements);
        std::array<std::vector<float>, 2> participant_host{
            std::vector<float>(route_elements, 0.0f),
            std::vector<float>(route_elements, 0.0f)};

        for (int row = 0; row < m; ++row)
        {
            for (int route = 0; route < kTopK; ++route)
            {
                const int owner = (row + route) & 1;
                for (int column = 0; column < kDModel; ++column)
                {
                    const size_t index =
                        (static_cast<size_t>(row) * kTopK + route) *
                            kDModel +
                        column;
                    /*
                     * Mixed signs and non-power-of-two divisors make each
                     * ordered FP32 addition observable while avoiding NaNs,
                     * infinities, and owner values that are exactly zero.
                     */
                    const int numerator =
                        ((row + 3) * 97 +
                         (route + 5) * 53 +
                         (column + 7) * 29) %
                            4093 -
                        2046;
                    const float value =
                        static_cast<float>(numerator) / 37.0f +
                        (owner == 0 ? 0.03125f : -0.046875f);
                    serial_host[index] = value;
                    participant_host[static_cast<size_t>(owner)][index] =
                        value;
                }
            }
        }

        for (int participant = 0; participant < 2; ++participant)
        {
            ASSERT_EQ(cudaSetDevice(participant), cudaSuccess);
            const cudaStream_t stream =
                streams[static_cast<size_t>(participant)];
            FP32Tensor *const route_tensor =
                route_slots[static_cast<size_t>(participant)].get();
            std::copy(
                participant_host[static_cast<size_t>(participant)].begin(),
                participant_host[static_cast<size_t>(participant)].end(),
                route_tensor->mutable_data());
            ASSERT_TRUE(route_tensor->ensureOnDevice(
                DeviceId::cuda(participant),
                stream));
            ASSERT_EQ(
                cudaMemsetAsync(
                    compact_outputs[static_cast<size_t>(participant)]
                        ->gpu_data_ptr(),
                    0xA5,
                    output_elements * sizeof(float),
                    stream),
                cudaSuccess);
            ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
        }

        /* Build the serial-row oracle with the production reducer launcher. */
        ASSERT_EQ(cudaSetDevice(kRoot), cudaSuccess);
        std::copy(
            serial_host.begin(),
            serial_host.end(),
            serial_route_slots->mutable_data());
        ASSERT_TRUE(serial_route_slots->ensureOnDevice(
            DeviceId::cuda(kRoot),
            streams[0]));
        ASSERT_TRUE(root_reducer.reduceCanonicalRouteContributions(
            serial_route_slots.get(),
            serial_output.get(),
            m,
            kTopK,
            kDModel));
        ASSERT_EQ(cudaStreamSynchronize(streams[0]), cudaSuccess);

        Barrier capture_started(2);
        Barrier capture_finished(2);
        std::array<CaptureResult, 2> capture_results{};

        auto capture_participant = [&](int participant)
        {
            CaptureResult &result =
                capture_results[static_cast<size_t>(participant)];
            const cudaStream_t stream =
                streams[static_cast<size_t>(participant)];
            result.begin_status = cudaSetDevice(participant);
            if (result.begin_status == cudaSuccess)
            {
                result.begin_status = cudaStreamBeginCapture(
                    stream,
                    cudaStreamCaptureModeRelaxed);
            }
            capture_started.arriveAndWait();

            if (result.begin_status == cudaSuccess)
            {
                GraphCaptureGuard guard;
                bool ok = ctx->reduceRawOnStream(
                    route_slots[static_cast<size_t>(participant)]
                        ->gpu_data_ptr(),
                    route_slots[static_cast<size_t>(participant)]
                        ->gpu_data_ptr(),
                    route_elements,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM,
                    kRoot,
                    participant,
                    stream,
                    "canonical_routes_reduce_to_root");
                if (ok && participant == kRoot)
                {
                    ok = root_reducer.reduceCanonicalRouteContributions(
                        route_slots[0].get(),
                        compact_outputs[0].get(),
                        m,
                        kTopK,
                        kDModel);
                }
                if (ok)
                {
                    ok = ctx->broadcastRawOnStream(
                        compact_outputs[static_cast<size_t>(participant)]
                            ->gpu_data_ptr(),
                        compact_outputs[static_cast<size_t>(participant)]
                            ->gpu_data_ptr(),
                        output_elements,
                        CollectiveDataType::FLOAT32,
                        kRoot,
                        participant,
                        stream,
                        "canonical_routes_broadcast");
                }
                result.collective_ok = ok;
            }

            capture_finished.arriveAndWait();
            if (result.begin_status == cudaSuccess && result.collective_ok)
            {
                result.end_status = cudaStreamEndCapture(
                    stream,
                    &result.graph);
            }
            if (result.end_status == cudaSuccess && result.graph)
            {
                result.instantiate_status = cudaGraphInstantiate(
                    &result.exec,
                    result.graph,
                    nullptr,
                    nullptr,
                    0);
            }
        };

        std::thread capture0(capture_participant, 0);
        std::thread capture1(capture_participant, 1);
        capture0.join();
        capture1.join();

        for (int participant = 0; participant < 2; ++participant)
        {
            const CaptureResult &result =
                capture_results[static_cast<size_t>(participant)];
            ASSERT_EQ(result.begin_status, cudaSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_TRUE(result.collective_ok)
                << "M=" << m << " participant=" << participant;
            ASSERT_EQ(result.end_status, cudaSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_EQ(result.instantiate_status, cudaSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_NE(result.exec, nullptr);
        }

        Barrier launch_ready(2);
        std::array<cudaError_t, 2> replay_status{
            cudaSuccess,
            cudaSuccess};
        auto replay_participant = [&](int participant)
        {
            cudaError_t &status =
                replay_status[static_cast<size_t>(participant)];
            status = cudaSetDevice(participant);
            launch_ready.arriveAndWait();
            if (status == cudaSuccess)
            {
                status = cudaGraphLaunch(
                    capture_results[static_cast<size_t>(participant)].exec,
                    streams[static_cast<size_t>(participant)]);
            }
            if (status == cudaSuccess)
            {
                status = cudaStreamSynchronize(
                    streams[static_cast<size_t>(participant)]);
            }
        };
        std::thread replay0(replay_participant, 0);
        std::thread replay1(replay_participant, 1);
        replay0.join();
        replay1.join();
        ASSERT_EQ(replay_status[0], cudaSuccess) << "M=" << m;
        ASSERT_EQ(replay_status[1], cudaSuccess) << "M=" << m;

        std::vector<float> expected(output_elements);
        std::array<std::vector<float>, 2> actual{
            std::vector<float>(output_elements),
            std::vector<float>(output_elements)};
        downloadDeviceVector<float>(
            kRoot,
            static_cast<const float *>(serial_output->gpu_data_ptr()),
            &expected);
        for (int participant = 0; participant < 2; ++participant)
        {
            downloadDeviceVector<float>(
                participant,
                static_cast<const float *>(
                    compact_outputs[static_cast<size_t>(participant)]
                        ->gpu_data_ptr()),
                &actual[static_cast<size_t>(participant)]);
            EXPECT_EQ(
                std::memcmp(
                    expected.data(),
                    actual[static_cast<size_t>(participant)].data(),
                    output_elements * sizeof(float)),
                0)
                << "Rooted canonical publication drifted from serial-row bytes"
                << " M=" << m << " participant=" << participant;
        }

        destroyCaptureResult(capture_results[0]);
        destroyCaptureResult(capture_results[1]);
    }

    root_reducer.clearGPUStreamBinding();
    ASSERT_EQ(cudaSetDevice(kRoot), cudaSuccess);
    serial_output.reset();
    serial_route_slots.reset();
    for (int participant = 0; participant < 2; ++participant)
    {
        ASSERT_EQ(cudaSetDevice(participant), cudaSuccess);
        compact_outputs[static_cast<size_t>(participant)].reset();
        route_slots[static_cast<size_t>(participant)].reset();
        EXPECT_EQ(
            cudaStreamDestroy(streams[static_cast<size_t>(participant)]),
            cudaSuccess);
    }
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

/**
 * @brief Proves production-sized back-to-back K/V allgathers survive graph replay.
 *
 * Phase-split LocalTP prefill compacts one K shard and one V shard, gathers
 * both payloads on the graph stream, and only then deinterleaves them into the
 * replicated decode-cache rows.  Small sideband collectives do not exercise
 * the same launch geometry or expose accidental overlap between the adjacent
 * receive buffers.  Keep this case symmetric with the RCCL regression and use
 * the Qwen3.6 long-context size class that originally exposed K-only drift.
 */
TEST(Test__LocalTPNCCLGraphCapture, NCCLRawAllgather_GraphCapturedLargeBackToBackKVPayloads_ReplaysCorrectly)
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

    constexpr size_t tokens = 640;
    constexpr size_t local_kv_dim = 256;
    constexpr size_t shard_count = tokens * local_kv_dim;

    float *send_k0 = nullptr;
    float *send_k1 = nullptr;
    float *send_v0 = nullptr;
    float *send_v1 = nullptr;
    float *recv_k0 = nullptr;
    float *recv_k1 = nullptr;
    float *recv_v0 = nullptr;
    float *recv_v1 = nullptr;
    allocateAndUpload<float>(0, std::vector<float>(shard_count, 1.0f), &send_k0);
    allocateAndUpload<float>(1, std::vector<float>(shard_count, 2.0f), &send_k1);
    allocateAndUpload<float>(0, std::vector<float>(shard_count, 3.0f), &send_v0);
    allocateAndUpload<float>(1, std::vector<float>(shard_count, 4.0f), &send_v1);
    allocateAndUpload<float>(0, std::vector<float>(shard_count * 2, -1.0f), &recv_k0);
    allocateAndUpload<float>(1, std::vector<float>(shard_count * 2, -1.0f), &recv_k1);
    allocateAndUpload<float>(0, std::vector<float>(shard_count * 2, -1.0f), &recv_v0);
    allocateAndUpload<float>(1, std::vector<float>(shard_count * 2, -1.0f), &recv_v1);

    cudaStream_t stream0 = nullptr;
    cudaStream_t stream1 = nullptr;
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream0, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking), cudaSuccess);

    Barrier ready_to_capture(2);
    Barrier captured_collectives(2);
    Barrier ready_to_launch(2);
    CaptureResult result0;
    CaptureResult result1;

    auto capture_worker = [&](int device,
                              const float *send_k,
                              const float *send_v,
                              float *recv_k,
                              float *recv_v,
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
            const bool k_ok = ctx->allgatherRawOnStream(
                send_k,
                recv_k,
                shard_count,
                CollectiveDataType::FLOAT32,
                device,
                stream,
                "large_kv_raw_allgather_K");
            const bool v_ok = ctx->allgatherRawOnStream(
                send_v,
                recv_v,
                shard_count,
                CollectiveDataType::FLOAT32,
                device,
                stream,
                "large_kv_raw_allgather_V");
            result->collective_ok = k_ok && v_ok;
        }

        captured_collectives.arriveAndWait();
        if (result->begin_status == cudaSuccess && result->collective_ok)
        {
            result->end_status = cudaSetDevice(device);
            if (result->end_status == cudaSuccess)
                result->end_status = cudaStreamEndCapture(stream, &result->graph);
        }
        if (result->end_status == cudaSuccess && result->graph)
        {
            result->instantiate_status = cudaGraphInstantiate(
                &result->exec,
                result->graph,
                nullptr,
                nullptr,
                0);
        }

        for (int replay = 0; replay < 2; ++replay)
        {
            ready_to_launch.arriveAndWait();
            if (result->instantiate_status == cudaSuccess && result->exec)
            {
                result->launch_status = cudaGraphLaunch(result->exec, stream);
                if (result->launch_status == cudaSuccess)
                    result->launch_status = cudaStreamSynchronize(stream);
            }
        }
    };

    std::thread t0(
        capture_worker,
        0,
        send_k0,
        send_v0,
        recv_k0,
        recv_v0,
        stream0,
        &result0);
    std::thread t1(
        capture_worker,
        1,
        send_k1,
        send_v1,
        recv_k1,
        recv_v1,
        stream1,
        &result1);
    t0.join();
    t1.join();

    expectCapturedGraphReady(result0, "large_kv_graph0");
    expectCapturedGraphReady(result1, "large_kv_graph1");
    EXPECT_EQ(result0.launch_status, cudaSuccess);
    EXPECT_EQ(result1.launch_status, cudaSuccess);

    std::vector<float> k0(shard_count * 2);
    std::vector<float> k1(shard_count * 2);
    std::vector<float> v0(shard_count * 2);
    std::vector<float> v1(shard_count * 2);
    downloadDeviceVector(0, recv_k0, &k0);
    downloadDeviceVector(1, recv_k1, &k1);
    downloadDeviceVector(0, recv_v0, &v0);
    downloadDeviceVector(1, recv_v1, &v1);

    auto expect_shards = [&](const std::vector<float> &actual,
                             float first,
                             float second,
                             const char *label)
    {
        ASSERT_EQ(actual.size(), shard_count * 2) << label;
        EXPECT_TRUE(std::all_of(
            actual.begin(),
            actual.begin() + static_cast<std::ptrdiff_t>(shard_count),
            [first](float value) { return value == first; }))
            << label << " first shard";
        EXPECT_TRUE(std::all_of(
            actual.begin() + static_cast<std::ptrdiff_t>(shard_count),
            actual.end(),
            [second](float value) { return value == second; }))
            << label << " second shard";
    };
    expect_shards(k0, 1.0f, 2.0f, "device0 K");
    expect_shards(k1, 1.0f, 2.0f, "device1 K");
    expect_shards(v0, 3.0f, 4.0f, "device0 V");
    expect_shards(v1, 3.0f, 4.0f, "device1 V");

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream0), cudaSuccess);
    ASSERT_EQ(cudaSetDevice(1), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(stream1), cudaSuccess);
    freeDevicePtr(0, send_k0);
    freeDevicePtr(1, send_k1);
    freeDevicePtr(0, send_v0);
    freeDevicePtr(1, send_v1);
    freeDevicePtr(0, recv_k0);
    freeDevicePtr(1, recv_k1);
    freeDevicePtr(0, recv_v0);
    freeDevicePtr(1, recv_v1);
}

/**
 * @test A captured NCCL byte gather followed by the device-owned rank fold is
 *       invariant to verifier row count.
 *
 * Production MTP compares grouped row zero with an independently captured M=1
 * decode. Native collective reduction trees may change with message size, so
 * the canonical protocol transports immutable rank banks and performs its only
 * arithmetic in ascending rank order. This test binds that complete protocol
 * into retained CUDA graphs for M=1, M=2, and M=3 and requires byte-identical
 * row-zero results.
 */
TEST(Test__LocalTPNCCLGraphCapture,
     NCCLCanonicalRankOrderReduction_GraphCapturedRowZeroIsBatchInvariant)
{
    auto *cuda_backend = getCUDABackend();
    ASSERT_NE(cuda_backend, nullptr);
    if (cuda_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ CUDA GPUs, found "
                     << cuda_backend->deviceCount();
    }

    constexpr int kParticipants = 2;
    constexpr size_t kColumns = 3072;
    auto ctx = createLocalTPContext(
        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
        {},
        CollectiveBackendType::NCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    const auto run_shape = [&](int rows, std::vector<float> *row_zero)
    {
        ASSERT_GT(rows, 0);
        ASSERT_NE(row_zero, nullptr);
        const size_t element_count = static_cast<size_t>(rows) * kColumns;
        std::array<float *, kParticipants> values{};
        std::array<float *, kParticipants> rank_banks{};
        std::array<cudaStream_t, kParticipants> streams{};
        std::array<CaptureResult, kParticipants> results{};

        std::array<std::vector<float>, kParticipants> host_values;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            auto &host = host_values[static_cast<size_t>(participant)];
            host.resize(element_count);
            for (int row = 0; row < rows; ++row)
            {
                for (size_t column = 0; column < kColumns; ++column)
                {
                    const float row_term = static_cast<float>(row) * 0.25F;
                    const float column_term =
                        static_cast<float>(column % 17u) * 0.03125F;
                    host[static_cast<size_t>(row) * kColumns + column] =
                        participant == 0
                            ? 1.0F + row_term + column_term
                            : 2.0F - row_term - column_term * 0.5F;
                }
            }
            allocateAndUpload<float>(participant, host, &values[participant]);
            allocateAndUpload<float>(
                participant,
                std::vector<float>(
                    element_count * static_cast<size_t>(kParticipants),
                    -1.0F),
                &rank_banks[participant]);
            ASSERT_EQ(cudaSetDevice(participant), cudaSuccess);
            ASSERT_EQ(
                cudaStreamCreateWithFlags(
                    &streams[participant], cudaStreamNonBlocking),
                cudaSuccess);
        }

        Barrier ready_to_capture(kParticipants);
        Barrier captured_protocol(kParticipants);
        std::array<std::thread, kParticipants> capture_threads;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            capture_threads[participant] = std::thread(
                [&, participant]
                {
                    auto &result = results[participant];
                    result.begin_status = cudaSetDevice(participant);
                    if (result.begin_status == cudaSuccess)
                    {
                        result.begin_status = cudaStreamBeginCapture(
                            streams[participant],
                            cudaStreamCaptureModeRelaxed);
                    }
                    ready_to_capture.arriveAndWait();

                    if (result.begin_status == cudaSuccess)
                    {
                        GraphCaptureGuard guard;
                        const bool gathered = ctx->allgatherRawOnStream(
                            values[participant],
                            rank_banks[participant],
                            element_count,
                            CollectiveDataType::FLOAT32,
                            participant,
                            streams[participant],
                            "nccl_canonical_rank_order_m" +
                                std::to_string(rows));
                        const bool folded = gathered &&
                            launchCUDATPRankOrderedSumFP32(
                                rank_banks[participant],
                                values[participant],
                                element_count,
                                kParticipants,
                                participant,
                                streams[participant]);
                        result.collective_ok = gathered && folded;
                    }
                    captured_protocol.arriveAndWait();

                    if (result.begin_status == cudaSuccess &&
                        result.collective_ok)
                    {
                        result.end_status = cudaSetDevice(participant);
                        if (result.end_status == cudaSuccess)
                        {
                            result.end_status = cudaStreamEndCapture(
                                streams[participant], &result.graph);
                        }
                    }
                    if (result.end_status == cudaSuccess && result.graph)
                    {
                        result.instantiate_status = cudaGraphInstantiate(
                            &result.exec,
                            result.graph,
                            nullptr,
                            nullptr,
                            0);
                    }
                });
        }
        for (auto &thread : capture_threads)
            thread.join();

        for (int participant = 0; participant < kParticipants; ++participant)
        {
            const auto &result = results[participant];
            ASSERT_EQ(result.begin_status, cudaSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_TRUE(result.collective_ok)
                << "participant=" << participant << " M=" << rows;
            ASSERT_EQ(result.end_status, cudaSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_EQ(result.instantiate_status, cudaSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_NE(result.exec, nullptr);
        }

        Barrier ready_to_replay(kParticipants);
        std::array<cudaError_t, kParticipants> replay_status{};
        std::array<std::thread, kParticipants> replay_threads;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            replay_threads[participant] = std::thread(
                [&, participant]
                {
                    replay_status[participant] = cudaSetDevice(participant);
                    ready_to_replay.arriveAndWait();
                    if (replay_status[participant] == cudaSuccess)
                    {
                        replay_status[participant] = cudaGraphLaunch(
                            results[participant].exec,
                            streams[participant]);
                    }
                    if (replay_status[participant] == cudaSuccess)
                    {
                        replay_status[participant] = cudaStreamSynchronize(
                            streams[participant]);
                    }
                });
        }
        for (auto &thread : replay_threads)
            thread.join();

        std::array<std::vector<float>, kParticipants> actual;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            ASSERT_EQ(replay_status[participant], cudaSuccess)
                << "participant=" << participant << " M=" << rows;
            actual[participant].resize(element_count);
            downloadDeviceVector<float>(
                participant, values[participant], &actual[participant]);
        }
        for (int participant = 1; participant < kParticipants; ++participant)
        {
            EXPECT_EQ(
                std::memcmp(
                    actual[0].data(),
                    actual[participant].data(),
                    element_count * sizeof(float)),
                0)
                << "Canonical TP publication differs by participant, M="
                << rows;
        }

        row_zero->assign(actual[0].begin(), actual[0].begin() + kColumns);
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            destroyCaptureResult(results[participant]);
            ASSERT_EQ(cudaSetDevice(participant), cudaSuccess);
            EXPECT_EQ(cudaStreamDestroy(streams[participant]), cudaSuccess);
            freeDevicePtr(participant, values[participant]);
            freeDevicePtr(participant, rank_banks[participant]);
        }
    };

    std::vector<float> serial_row;
    std::vector<float> grouped_m2_row_zero;
    std::vector<float> grouped_m3_row_zero;
    ASSERT_NO_FATAL_FAILURE(run_shape(1, &serial_row));
    ASSERT_NO_FATAL_FAILURE(run_shape(2, &grouped_m2_row_zero));
    ASSERT_NO_FATAL_FAILURE(run_shape(3, &grouped_m3_row_zero));
    ASSERT_EQ(serial_row.size(), kColumns);
    ASSERT_EQ(grouped_m2_row_zero.size(), kColumns);
    ASSERT_EQ(grouped_m3_row_zero.size(), kColumns);
    EXPECT_EQ(
        std::memcmp(
            serial_row.data(),
            grouped_m2_row_zero.data(),
            kColumns * sizeof(float)),
        0)
        << "M=2 verifier row zero must be byte-identical to M=1 decode";
    EXPECT_EQ(
        std::memcmp(
            serial_row.data(),
            grouped_m3_row_zero.data(),
            kColumns * sizeof(float)),
        0)
        << "M=3 verifier row zero must be byte-identical to M=1 decode";
}

/**
 * @test Production NCCL rejoins each recording of the same retained parent.
 *
 * CUDA reuses a graph's capture ID across begin/end recording sessions. NCCL
 * must not mistake that ID for proof that its cached internal stream is still
 * capturing. Exercise three sessions, a native conditional, and repeated
 * execution with changed inputs; independent child graphs miss this defect.
 */
TEST(Test__LocalTPNCCLGraphCapture, NCCLRetainedParentFragmentReentry)
{
    auto *backend = getCUDABackend();
    ASSERT_NE(backend, nullptr);
    if (backend->deviceCount() < 2)
        GTEST_SKIP() << "Requires two CUDA participants";
    auto context = createLocalTPContext(
        {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)},
        {}, CollectiveBackendType::NCCL);
    ASSERT_NE(context, nullptr);
    constexpr size_t count = 257u;
    std::array<float *, 2> inputs{}, outputs{};
    std::array<cudaStream_t, 2> streams{};
    std::array<std::unique_ptr<CUDAGraphCapture>, 2> parents;
    for (int rank = 0; rank < 2; ++rank)
    {
        allocateAndUpload(rank, std::vector<float>(count, float(rank + 1)), &inputs[rank]);
        allocateAndUpload(rank, std::vector<float>(count, 0.0f), &outputs[rank]);
        ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
        ASSERT_EQ(cudaStreamCreateWithFlags(&streams[rank], cudaStreamNonBlocking), cudaSuccess);
        parents[rank] = std::make_unique<CUDAGraphCapture>(streams[rank], rank);
    }
    Barrier boundary(2);
    std::atomic<bool> failed{false};
    const auto record = [&](int rank) {
        EXPECT_EQ(cudaSetDevice(rank), cudaSuccess);
        std::array<std::unique_ptr<IGPUGraphCapture>, 3> fragments;
        std::array<GPUOrderedTimelineStep, 3> steps;
        for (size_t part = 0; part < fragments.size(); ++part)
        {
            fragments[part] = parents[rank]->createOrderedTimelineFragment();
            const bool began = fragments[part] && fragments[part]->beginCapture();
            if (!began) failed.store(true);
            // Both participants must enter/leave each collective together,
            // including the error path. No one skips a peer's rendezvous.
            boundary.arriveAndWait();
            const bool admitted = !failed.load();
            boundary.arriveAndWait();
            if (admitted)
            {
                const bool reduced = context->reduceRawOnStream(
                    inputs[rank], outputs[rank], count,
                    CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM,
                    0, rank, streams[rank], "retained_fragment_reduce");
                EXPECT_TRUE(reduced) << "fragment=" << part << " rank=" << rank;
                if (!reduced) failed.store(true);
            }
            boundary.arriveAndWait();
            // Include the conditional shape that forbids cloning fragments.
            if (began && !failed.load())
            {
                cudaGraph_t graph{};
                cudaStreamCaptureStatus status{};
                const cudaGraphNode_t *dependencies = nullptr;
                size_t dependency_count = 0u;
                EXPECT_EQ(cudaStreamGetCaptureInfo(streams[rank], &status, nullptr,
                    &graph, &dependencies, nullptr, &dependency_count), cudaSuccess);
                cudaGraphConditionalHandle handle{};
                EXPECT_EQ(cudaGraphConditionalHandleCreate(&handle, graph, 1u,
                    cudaGraphCondAssignDefault), cudaSuccess);
                cudaGraphNodeParams params{};
                params.type = cudaGraphNodeTypeConditional;
                params.conditional.handle = handle;
                params.conditional.type = cudaGraphCondTypeIf;
                params.conditional.size = 1u;
                cudaGraphNode_t branch{};
                EXPECT_EQ(cudaGraphAddNode(&branch, graph, dependencies, nullptr,
                    dependency_count, &params), cudaSuccess);
                cudaGraphNode_t body{};
                EXPECT_EQ(cudaGraphAddEmptyNode(&body,
                    params.conditional.phGraph_out[0], nullptr, 0u), cudaSuccess);
                EXPECT_EQ(cudaStreamUpdateCaptureDependencies(streams[rank], &branch,
                    nullptr, 1u, cudaStreamSetCaptureDependencies), cudaSuccess);
            }
            if (began && !fragments[part]->endCapture()) failed.store(true);
            boundary.arriveAndWait();
            if (failed.load()) break;
            steps[part] = {.name = "NCCL fragment",
                .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                .capture = fragments[part].get()};
        }
        if (!failed.load())
        {
            EXPECT_TRUE(parents[rank]->buildOrderedTimelineTransaction(steps));
            EXPECT_TRUE(parents[rank]->instantiate());
        }
    };
    std::thread first(record, 0), second(record, 1);
    first.join();
    second.join();
    if (!failed.load())
    {
        for (int replay = 0; replay < 5; ++replay)
        {
            // Replays use fresh values, not capture-time contents. Submit both
            // participants before waiting for terminal test-only completion.
            std::array<std::vector<float>, 2> values{
                std::vector<float>(count, float(replay + 1)),
                std::vector<float>(count, float(replay + 2))};
            for (int rank = 0; rank < 2; ++rank)
            {
                ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
                ASSERT_EQ(cudaMemcpyAsync(inputs[rank], values[rank].data(),
                    count * sizeof(float), cudaMemcpyHostToDevice, streams[rank]), cudaSuccess);
                ASSERT_TRUE(parents[rank]->launch());
            }
            for (int rank = 0; rank < 2; ++rank)
            {
                ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
                ASSERT_EQ(cudaStreamSynchronize(streams[rank]), cudaSuccess);
            }
            std::vector<float> actual(count);
            downloadDeviceVector(0, outputs[0], &actual);
            for (const float value : actual) EXPECT_EQ(value, float(2 * replay + 3));
        }
    }
    EXPECT_FALSE(failed.load());
    for (int rank = 0; rank < 2; ++rank)
    {
        ASSERT_EQ(cudaSetDevice(rank), cudaSuccess);
        parents[rank].reset();
        freeDevicePtr(rank, inputs[rank]);
        freeDevicePtr(rank, outputs[rank]);
        EXPECT_EQ(cudaStreamDestroy(streams[rank]), cudaSuccess);
    }
}

#endif // HAVE_CUDA
