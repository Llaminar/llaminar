/**
 * @file Test__LocalTPRCCLGraphCapture.cpp
 * @brief ROCm-only LocalTP graph capture regression tests.
 *
 * Keep this in a separate translation unit from CUDA tests: cuda_runtime.h and
 * hip_runtime.h expose overlapping vector types and are intentionally not mixed.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <hip/hip_runtime.h>

#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"
#include "backends/IBackend.h"
#include "collective/ICollectiveBackend.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
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
        bool fp32_allgather_ok = false;
        bool int32_allgather_ok = false;
        bool int8_allgather_ok = false;
        hipError_t begin_status = hipSuccess;
        hipError_t end_status = hipSuccess;
        hipError_t instantiate_status = hipSuccess;
        hipError_t launch_status = hipSuccess;
        hipGraph_t graph = nullptr;
        hipGraphExec_t exec = nullptr;
    };

    struct CapturedAllgatherResources
    {
        void *capture_stream = nullptr;
    };

    struct CapturedP2PGraph
    {
        CaptureResult result0;
        CaptureResult result1;
        void *capture_stream0 = nullptr;
        void *capture_stream1 = nullptr;
        void *transfer_stream0 = nullptr;
        void *transfer_stream1 = nullptr;
        hipEvent_t transfer_ready0 = nullptr;
        hipEvent_t transfer_ready1 = nullptr;
        hipEvent_t transfer_done0 = nullptr;
        hipEvent_t transfer_done1 = nullptr;
    };

    struct P2PTimingResult
    {
        bool ok = true;
        double wall_ms = 0.0;
        hipError_t first_error = hipSuccess;
        std::string step;

        void record(hipError_t status, std::string where)
        {
            if (status == hipSuccess || !ok)
                return;
            ok = false;
            first_error = status;
            step = std::move(where);
        }
    };

    enum class RcclOverlapPattern
    {
        NoBarrierAfterMaintenanceLaunch,
        BarrierAfterMaintenanceLaunch,
        DecodeWaitsForMaintenanceCompletion
    };

    struct OverlapReplayResult
    {
        bool ok = true;
        double wall_ms = 0.0;
        hipError_t first_error = hipSuccess;
        std::string step;

        void record(hipError_t status, std::string where)
        {
            if (status == hipSuccess || !ok)
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
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        ASSERT_EQ(hipMalloc(reinterpret_cast<void **>(device_ptr),
                            host_values.size() * sizeof(T)),
                  hipSuccess);
        ASSERT_NE(*device_ptr, nullptr);
        ASSERT_EQ(hipMemcpy(*device_ptr,
                            host_values.data(),
                            host_values.size() * sizeof(T),
                            hipMemcpyHostToDevice),
                  hipSuccess);
    }

    void freeDevicePtr(int device, void *ptr)
    {
        if (!ptr)
            return;
        EXPECT_EQ(hipSetDevice(device), hipSuccess);
        EXPECT_EQ(hipFree(ptr), hipSuccess);
    }

    void destroyCaptureResult(CaptureResult &result)
    {
        if (result.exec)
        {
            EXPECT_EQ(hipGraphExecDestroy(result.exec), hipSuccess);
            result.exec = nullptr;
        }
        if (result.graph)
        {
            EXPECT_EQ(hipGraphDestroy(result.graph), hipSuccess);
            result.graph = nullptr;
        }
    }

    void destroyAllgatherResources(IBackend *backend, int device, CapturedAllgatherResources &resources)
    {
        if (resources.capture_stream)
        {
            backend->destroyStream(resources.capture_stream, device);
            resources.capture_stream = nullptr;
        }
    }

    void captureDecodeAllreduceGraph(
        ILocalTPContext &ctx,
        TensorBase *tensor0,
        TensorBase *tensor1,
        void *stream0,
        void *stream1,
        size_t count,
        CaptureResult &result0,
        CaptureResult &result1)
    {
        Barrier ready_to_capture(2);
        Barrier captured_collective(2);

        auto capture_worker = [&](int device,
                                  TensorBase *tensor,
                                  void *stream,
                                  CaptureResult *result)
        {
            result->begin_status = hipSetDevice(device);
            if (result->begin_status == hipSuccess)
            {
                result->begin_status = hipStreamBeginCapture(
                    static_cast<hipStream_t>(stream),
                    hipStreamCaptureModeRelaxed);
            }

            ready_to_capture.arriveAndWait();

            if (result->begin_status == hipSuccess)
            {
                GraphCaptureGuard guard;
                result->collective_ok = ctx.allreduceOnStream(
                    tensor,
                    "rccl_overlap_lab_decode_allreduce",
                    count,
                    stream,
                    "fp32");
            }

            captured_collective.arriveAndWait();

            if (result->begin_status == hipSuccess)
            {
                result->end_status = hipSetDevice(device);
                if (result->end_status == hipSuccess)
                {
                    result->end_status = hipStreamEndCapture(
                        static_cast<hipStream_t>(stream),
                        &result->graph);
                }
            }

            if (result->end_status == hipSuccess && result->graph)
            {
                result->instantiate_status = hipSetDevice(device);
                if (result->instantiate_status == hipSuccess)
                {
                    result->instantiate_status = hipGraphInstantiate(
                        &result->exec,
                        result->graph,
                        nullptr,
                        nullptr,
                        0);
                }
            }
        };

        std::thread t0(capture_worker, 0, tensor0, stream0, &result0);
        std::thread t1(capture_worker, 1, tensor1, stream1, &result1);
        t0.join();
        t1.join();
    }

    void captureMaintenanceRawAllgatherGraph(
        ILocalTPContext &ctx,
        TensorBase *send0,
        TensorBase *recv0,
        TensorBase *send1,
        TensorBase *recv1,
        void *stream0,
        void *stream1,
        size_t count,
        CaptureResult &result0,
        CaptureResult &result1)
    {
        Barrier ready_to_capture(2);
        Barrier captured_collective(2);

        auto capture_worker = [&](int device,
                                  TensorBase *send,
                                  TensorBase *recv,
                                  void *stream,
                                  CaptureResult *result)
        {
            result->begin_status = hipSetDevice(device);
            if (result->begin_status == hipSuccess)
            {
                result->begin_status = hipStreamBeginCapture(
                    static_cast<hipStream_t>(stream),
                    hipStreamCaptureModeRelaxed);
            }

            ready_to_capture.arriveAndWait();

            if (result->begin_status == hipSuccess)
            {
                GraphCaptureGuard guard;
                result->fp32_allgather_ok = ctx.allgatherRawOnStream(
                    send->gpu_data_ptr(),
                    recv->gpu_data_ptr(),
                    count,
                    CollectiveDataType::FLOAT32,
                    device,
                    stream,
                    "rccl_overlap_lab_maintenance_raw_allgather");
                result->collective_ok = result->fp32_allgather_ok;
            }

            captured_collective.arriveAndWait();

            if (result->begin_status == hipSuccess &&
                result->end_status == hipSuccess &&
                result->collective_ok)
            {
                result->end_status = hipSetDevice(device);
                if (result->end_status == hipSuccess)
                {
                    result->end_status = hipStreamEndCapture(
                        static_cast<hipStream_t>(stream),
                        &result->graph);
                }
            }

            if (result->end_status == hipSuccess && result->graph)
            {
                result->instantiate_status = hipSetDevice(device);
                if (result->instantiate_status == hipSuccess)
                {
                    result->instantiate_status = hipGraphInstantiate(
                        &result->exec,
                        result->graph,
                        nullptr,
                        nullptr,
                        0);
                }
            }
        };

        std::thread t0(capture_worker, 0, send0, recv0, stream0, &result0);
        std::thread t1(capture_worker, 1, send1, recv1, stream1, &result1);
        t0.join();
        t1.join();
    }

    void createP2PStreamsAndEvents(IBackend *backend, CapturedP2PGraph &graph)
    {
        graph.capture_stream0 = backend->createStream(0);
        graph.capture_stream1 = backend->createStream(1);
        graph.transfer_stream0 = backend->createStream(0);
        graph.transfer_stream1 = backend->createStream(1);
        ASSERT_NE(graph.capture_stream0, nullptr);
        ASSERT_NE(graph.capture_stream1, nullptr);
        ASSERT_NE(graph.transfer_stream0, nullptr);
        ASSERT_NE(graph.transfer_stream1, nullptr);

        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&graph.transfer_ready0, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&graph.transfer_done0, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&graph.transfer_ready1, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&graph.transfer_done1, hipEventDisableTiming), hipSuccess);
    }

    void destroyP2PGraph(IBackend *backend, CapturedP2PGraph &graph)
    {
        destroyCaptureResult(graph.result0);
        destroyCaptureResult(graph.result1);
        if (graph.transfer_ready0)
        {
            EXPECT_EQ(hipSetDevice(0), hipSuccess);
            EXPECT_EQ(hipEventDestroy(graph.transfer_ready0), hipSuccess);
            graph.transfer_ready0 = nullptr;
        }
        if (graph.transfer_done0)
        {
            EXPECT_EQ(hipSetDevice(0), hipSuccess);
            EXPECT_EQ(hipEventDestroy(graph.transfer_done0), hipSuccess);
            graph.transfer_done0 = nullptr;
        }
        if (graph.transfer_ready1)
        {
            EXPECT_EQ(hipSetDevice(1), hipSuccess);
            EXPECT_EQ(hipEventDestroy(graph.transfer_ready1), hipSuccess);
            graph.transfer_ready1 = nullptr;
        }
        if (graph.transfer_done1)
        {
            EXPECT_EQ(hipSetDevice(1), hipSuccess);
            EXPECT_EQ(hipEventDestroy(graph.transfer_done1), hipSuccess);
            graph.transfer_done1 = nullptr;
        }
        if (graph.capture_stream0)
        {
            backend->destroyStream(graph.capture_stream0, 0);
            graph.capture_stream0 = nullptr;
        }
        if (graph.capture_stream1)
        {
            backend->destroyStream(graph.capture_stream1, 1);
            graph.capture_stream1 = nullptr;
        }
        if (graph.transfer_stream0)
        {
            backend->destroyStream(graph.transfer_stream0, 0);
            graph.transfer_stream0 = nullptr;
        }
        if (graph.transfer_stream1)
        {
            backend->destroyStream(graph.transfer_stream1, 1);
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
                                  void *capture_stream,
                                  void *transfer_stream,
                                  hipEvent_t transfer_ready,
                                  hipEvent_t transfer_done,
                                  CaptureResult *result)
        {
            result->begin_status = hipSetDevice(device);
            if (result->begin_status == hipSuccess)
            {
                result->begin_status = hipStreamBeginCapture(
                    static_cast<hipStream_t>(capture_stream),
                    hipStreamCaptureModeRelaxed);
            }

            ready_to_capture.arriveAndWait();

            if (result->begin_status == hipSuccess)
            {
                GraphCaptureGuard guard;
                result->launch_status = hipEventRecord(
                    transfer_ready,
                    static_cast<hipStream_t>(capture_stream));
                if (result->launch_status == hipSuccess)
                {
                    result->launch_status = hipStreamWaitEvent(
                        static_cast<hipStream_t>(transfer_stream),
                        transfer_ready,
                        0);
                }

                if (result->launch_status == hipSuccess)
                {
                    result->collective_ok = ctx.groupedP2PRawOnStream(
                        *ops,
                        device,
                        transfer_stream,
                        "rccl_p2p_aux_stream_capture_probe");
                }

                if (result->launch_status == hipSuccess &&
                    result->collective_ok)
                {
                    result->launch_status = hipEventRecord(
                        transfer_done,
                        static_cast<hipStream_t>(transfer_stream));
                    if (result->launch_status == hipSuccess)
                    {
                        result->launch_status = hipStreamWaitEvent(
                            static_cast<hipStream_t>(capture_stream),
                            transfer_done,
                            0);
                    }
                }
            }

            captured_collective.arriveAndWait();

            if (result->begin_status == hipSuccess &&
                result->launch_status == hipSuccess &&
                result->collective_ok)
            {
                result->end_status = hipSetDevice(device);
                if (result->end_status == hipSuccess)
                {
                    result->end_status = hipStreamEndCapture(
                        static_cast<hipStream_t>(capture_stream),
                        &result->graph);
                }
            }

            if (result->end_status == hipSuccess && result->graph)
            {
                result->instantiate_status = hipSetDevice(device);
                if (result->instantiate_status == hipSuccess)
                {
                    result->instantiate_status = hipGraphInstantiate(
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
                                 void *capture_stream,
                                 void *transfer_stream,
                                 const CaptureResult *capture,
                                 P2PTimingResult *local_result)
        {
            auto record = [&](hipError_t status, const std::string &where)
            {
                local_result->record(status, "device " + std::to_string(device) + " " + where);
                return status == hipSuccess;
            };

            record(hipSetDevice(device), "set device");
            const auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
            {
                replay_iteration.arriveAndWait();
                if (local_result->ok)
                {
                    record(hipGraphLaunch(
                               capture->exec,
                               static_cast<hipStream_t>(capture_stream)),
                           "launch p2p graph");
                }
            }
            if (local_result->ok)
                record(hipStreamSynchronize(static_cast<hipStream_t>(capture_stream)), "sync capture stream");
            if (local_result->ok)
                record(hipStreamSynchronize(static_cast<hipStream_t>(transfer_stream)), "sync transfer stream");
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
        EXPECT_EQ(result.begin_status, hipSuccess) << name;
        EXPECT_TRUE(result.collective_ok) << name;
        EXPECT_EQ(result.end_status, hipSuccess) << name;
        EXPECT_NE(result.graph, nullptr) << name;
        EXPECT_EQ(result.instantiate_status, hipSuccess) << name;
        EXPECT_NE(result.exec, nullptr) << name;
    }

    void runRcclDecodeMaintenanceOverlapLab(RcclOverlapPattern pattern)
    {
        auto *rocm_backend = getROCmBackend();
        ASSERT_NE(rocm_backend, nullptr);
        if (rocm_backend->deviceCount() < 2)
        {
            GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_backend->deviceCount();
        }

        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1)};

        auto decode_ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
        auto maintenance_ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
        ASSERT_NE(decode_ctx, nullptr);
        ASSERT_NE(maintenance_ctx, nullptr);
        ASSERT_TRUE(maintenance_ctx->supportsRawAllgatherOnStreamGraphCapture());

        constexpr size_t decode_count = 18432;
        constexpr size_t maintenance_count = 4096;
        constexpr int iterations = 4;

        auto decode0 = TestTensorFactory::createFP32({decode_count});
        auto decode1 = TestTensorFactory::createFP32({decode_count});
        auto send0 = TestTensorFactory::createFP32({maintenance_count});
        auto send1 = TestTensorFactory::createFP32({maintenance_count});
        auto recv0 = TestTensorFactory::createFP32({maintenance_count * 2});
        auto recv1 = TestTensorFactory::createFP32({maintenance_count * 2});
        TestTensorFactory::fillValue(decode0.get(), 1.0f);
        TestTensorFactory::fillValue(decode1.get(), 2.0f);
        TestTensorFactory::fillValue(send0.get(), 11.0f);
        TestTensorFactory::fillValue(send1.get(), 22.0f);
        TestTensorFactory::fillValue(recv0.get(), -1.0f);
        TestTensorFactory::fillValue(recv1.get(), -1.0f);

        ASSERT_TRUE(decode0->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(decode1->ensureOnDevice(DeviceId::rocm(1)));
        ASSERT_TRUE(send0->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(send1->ensureOnDevice(DeviceId::rocm(1)));
        ASSERT_TRUE(recv0->ensureOnDevice(DeviceId::rocm(0)));
        ASSERT_TRUE(recv1->ensureOnDevice(DeviceId::rocm(1)));

        void *decode_stream0 = rocm_backend->createStream(0);
        void *decode_stream1 = rocm_backend->createStream(1);
        void *maintenance_stream0 = rocm_backend->createStream(0);
        void *maintenance_stream1 = rocm_backend->createStream(1);
        ASSERT_NE(decode_stream0, nullptr);
        ASSERT_NE(decode_stream1, nullptr);
        ASSERT_NE(maintenance_stream0, nullptr);
        ASSERT_NE(maintenance_stream1, nullptr);

        CaptureResult decode_result0;
        CaptureResult decode_result1;
        CaptureResult maintenance_result0;
        CaptureResult maintenance_result1;

        captureDecodeAllreduceGraph(
            *decode_ctx,
            decode0.get(),
            decode1.get(),
            decode_stream0,
            decode_stream1,
            decode_count,
            decode_result0,
            decode_result1);
        captureMaintenanceRawAllgatherGraph(
            *maintenance_ctx,
            send0.get(),
            recv0.get(),
            send1.get(),
            recv1.get(),
            maintenance_stream0,
            maintenance_stream1,
            maintenance_count,
            maintenance_result0,
            maintenance_result1);

        expectCapturedGraphReady(decode_result0, "decode0");
        expectCapturedGraphReady(decode_result1, "decode1");
        expectCapturedGraphReady(maintenance_result0, "maintenance0");
        expectCapturedGraphReady(maintenance_result1, "maintenance1");
        ASSERT_FALSE(::testing::Test::HasFailure());

        hipEvent_t decode_done0 = nullptr;
        hipEvent_t decode_done1 = nullptr;
        hipEvent_t maintenance_done0 = nullptr;
        hipEvent_t maintenance_done1 = nullptr;
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&decode_done0, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&maintenance_done0, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&decode_done1, hipEventDisableTiming), hipSuccess);
        ASSERT_EQ(hipEventCreateWithFlags(&maintenance_done1, hipEventDisableTiming), hipSuccess);

        Barrier iteration_start(2);
        Barrier maintenance_ready(2);
        Barrier maintenance_launched(2);
        OverlapReplayResult replay0;
        OverlapReplayResult replay1;

        auto replay_worker = [&](int device,
                                 void *decode_stream,
                                 void *maintenance_stream,
                                 hipEvent_t decode_done,
                                 hipEvent_t maintenance_done,
                                 const CaptureResult *decode_result,
                                 const CaptureResult *maintenance_result,
                                 OverlapReplayResult *replay_result)
        {
            auto record = [&](hipError_t status, const std::string &where)
            {
                replay_result->record(status, "device " + std::to_string(device) + " " + where);
                return status == hipSuccess;
            };

            if (!record(hipSetDevice(device), "set device"))
            {
                replay_result->ok = false;
            }

            const auto replay_start = std::chrono::steady_clock::now();
            for (int i = 0; i < iterations; ++i)
            {
                iteration_start.arriveAndWait();
                if (replay_result->ok)
                {
                    record(hipGraphLaunch(
                               decode_result->exec,
                               static_cast<hipStream_t>(decode_stream)),
                           "launch prior decode graph");
                }
                if (replay_result->ok)
                {
                    record(hipEventRecord(
                               decode_done,
                               static_cast<hipStream_t>(decode_stream)),
                           "record decode event");
                }
                if (replay_result->ok)
                {
                    record(hipStreamWaitEvent(
                               static_cast<hipStream_t>(maintenance_stream),
                               decode_done,
                               0),
                           "maintenance waits for decode event");
                }

                maintenance_ready.arriveAndWait();
                if (replay_result->ok)
                {
                    record(hipGraphLaunch(
                               maintenance_result->exec,
                               static_cast<hipStream_t>(maintenance_stream)),
                           "launch maintenance graph");
                }

                if (pattern == RcclOverlapPattern::BarrierAfterMaintenanceLaunch ||
                    pattern == RcclOverlapPattern::DecodeWaitsForMaintenanceCompletion)
                {
                    maintenance_launched.arriveAndWait();
                }

                if (pattern == RcclOverlapPattern::DecodeWaitsForMaintenanceCompletion &&
                    replay_result->ok)
                {
                    record(hipEventRecord(
                               maintenance_done,
                               static_cast<hipStream_t>(maintenance_stream)),
                           "record maintenance event");
                }
                if (pattern == RcclOverlapPattern::DecodeWaitsForMaintenanceCompletion &&
                    replay_result->ok)
                {
                    record(hipStreamWaitEvent(
                               static_cast<hipStream_t>(decode_stream),
                               maintenance_done,
                               0),
                           "decode waits for maintenance event");
                }

                if (replay_result->ok)
                {
                    record(hipGraphLaunch(
                               decode_result->exec,
                               static_cast<hipStream_t>(decode_stream)),
                           "launch overlapping decode graph");
                }
            }

            record(hipStreamSynchronize(static_cast<hipStream_t>(decode_stream)),
                   "sync decode stream");
            record(hipStreamSynchronize(static_cast<hipStream_t>(maintenance_stream)),
                   "sync maintenance stream");
            const auto replay_end = std::chrono::steady_clock::now();
            replay_result->wall_ms =
                std::chrono::duration<double, std::milli>(replay_end - replay_start).count();
        };

        std::thread t0(replay_worker,
                       0,
                       decode_stream0,
                       maintenance_stream0,
                       decode_done0,
                       maintenance_done0,
                       &decode_result0,
                       &maintenance_result0,
                       &replay0);
        std::thread t1(replay_worker,
                       1,
                       decode_stream1,
                       maintenance_stream1,
                       decode_done1,
                       maintenance_done1,
                       &decode_result1,
                       &maintenance_result1,
                       &replay1);
        t0.join();
        t1.join();

        EXPECT_TRUE(replay0.ok) << replay0.step << " hipError=" << hipGetErrorString(replay0.first_error);
        EXPECT_TRUE(replay1.ok) << replay1.step << " hipError=" << hipGetErrorString(replay1.first_error);
        const char *pattern_name =
            pattern == RcclOverlapPattern::NoBarrierAfterMaintenanceLaunch
                ? "no_barrier_after_maintenance_launch"
                : (pattern == RcclOverlapPattern::BarrierAfterMaintenanceLaunch
                       ? "barrier_after_maintenance_launch"
                       : "decode_waits_for_maintenance_completion");
        std::cout << "[RCCLDecodeMaintenanceOverlap]"
                  << " pattern=" << pattern_name
                  << " iterations=" << iterations
                  << " wall_ms=" << std::max(replay0.wall_ms, replay1.wall_ms)
                  << " device0_wall_ms=" << replay0.wall_ms
                  << " device1_wall_ms=" << replay1.wall_ms
                  << std::endl;

        if (decode_done0)
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            EXPECT_EQ(hipEventDestroy(decode_done0), hipSuccess);
        }
        if (maintenance_done0)
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
            EXPECT_EQ(hipEventDestroy(maintenance_done0), hipSuccess);
        }
        if (decode_done1)
        {
            ASSERT_EQ(hipSetDevice(1), hipSuccess);
            EXPECT_EQ(hipEventDestroy(decode_done1), hipSuccess);
        }
        if (maintenance_done1)
        {
            ASSERT_EQ(hipSetDevice(1), hipSuccess);
            EXPECT_EQ(hipEventDestroy(maintenance_done1), hipSuccess);
        }

        destroyCaptureResult(decode_result0);
        destroyCaptureResult(decode_result1);
        destroyCaptureResult(maintenance_result0);
        destroyCaptureResult(maintenance_result1);
        rocm_backend->destroyStream(decode_stream0, 0);
        rocm_backend->destroyStream(decode_stream1, 1);
        rocm_backend->destroyStream(maintenance_stream0, 0);
        rocm_backend->destroyStream(maintenance_stream1, 1);
    }
} // namespace

TEST(Test__LocalTPRCCLGraphCapture, RCCLAllreduce_OnStreamGraphCapture_Completes)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);

    constexpr size_t count = 18432;
    auto tensor0 = TestTensorFactory::createFP32({count});
    auto tensor1 = TestTensorFactory::createFP32({count});
    TestTensorFactory::fillValue(tensor0.get(), 1.0f);
    TestTensorFactory::fillValue(tensor1.get(), 2.0f);

    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::rocm(1)));

    void *stream0 = rocm_backend->createStream(0);
    void *stream1 = rocm_backend->createStream(1);
    ASSERT_NE(stream0, nullptr);
    ASSERT_NE(stream1, nullptr);

    Barrier ready_to_capture(2);
    Barrier captured_collective(2);
    CaptureResult result0;
    CaptureResult result1;

    auto capture_worker = [&](int device,
                              TensorBase *tensor,
                              void *stream,
                              CaptureResult *result)
    {
        result->begin_status = hipSetDevice(device);
        if (result->begin_status == hipSuccess)
        {
            result->begin_status = hipStreamBeginCapture(
                static_cast<hipStream_t>(stream),
                hipStreamCaptureModeRelaxed);
        }

        ready_to_capture.arriveAndWait();

        if (result->begin_status == hipSuccess)
        {
            GraphCaptureGuard guard;
            result->collective_ok = ctx->allreduceOnStream(
                tensor,
                "on_stream_graph_capture_regression_rccl",
                count,
                stream,
                "fp32");
        }

        captured_collective.arriveAndWait();

        if (result->begin_status == hipSuccess)
        {
            result->end_status = hipSetDevice(device);
            if (result->end_status == hipSuccess)
            {
                result->end_status = hipStreamEndCapture(
                    static_cast<hipStream_t>(stream),
                    &result->graph);
            }
        }

        if (result->end_status == hipSuccess && result->graph)
        {
            result->instantiate_status = hipSetDevice(device);
            if (result->instantiate_status == hipSuccess)
            {
                result->instantiate_status = hipGraphInstantiate(
                    &result->exec,
                    result->graph,
                    nullptr,
                    nullptr,
                    0);
            }
        }

        if (result->instantiate_status == hipSuccess && result->exec)
        {
            result->launch_status = hipSetDevice(device);
            if (result->launch_status == hipSuccess)
            {
                result->launch_status = hipGraphLaunch(
                    result->exec,
                    static_cast<hipStream_t>(stream));
            }
        }
    };

    std::thread t0(capture_worker, 0, tensor0.get(), stream0, &result0);
    std::thread t1(capture_worker, 1, tensor1.get(), stream1, &result1);
    t0.join();
    t1.join();

    EXPECT_EQ(result0.begin_status, hipSuccess);
    EXPECT_EQ(result1.begin_status, hipSuccess);
    EXPECT_TRUE(result0.collective_ok);
    EXPECT_TRUE(result1.collective_ok);
    EXPECT_EQ(result0.end_status, hipSuccess);
    EXPECT_EQ(result1.end_status, hipSuccess);
    EXPECT_NE(result0.graph, nullptr);
    EXPECT_NE(result1.graph, nullptr);
    EXPECT_EQ(result0.instantiate_status, hipSuccess);
    EXPECT_EQ(result1.instantiate_status, hipSuccess);
    EXPECT_EQ(result0.launch_status, hipSuccess);
    EXPECT_EQ(result1.launch_status, hipSuccess);

    ASSERT_TRUE(rocm_backend->synchronizeStream(stream0, 0));
    ASSERT_TRUE(rocm_backend->synchronizeStream(stream1, 1));

    const float *data0 = tensor0->data();
    const float *data1 = tensor1->data();
    ASSERT_NE(data0, nullptr);
    ASSERT_NE(data1, nullptr);
    for (size_t i = 0; i < count; ++i)
    {
        EXPECT_FLOAT_EQ(data0[i], 3.0f) << "ROCm graph capture:0 mismatch at index " << i;
        EXPECT_FLOAT_EQ(data1[i], 3.0f) << "ROCm graph capture:1 mismatch at index " << i;
    }

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    rocm_backend->destroyStream(stream0, 0);
    rocm_backend->destroyStream(stream1, 1);
}

TEST(Test__LocalTPRCCLGraphCapture, RCCLRawAllgather_OnStreamGraphCapture_Completes)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    constexpr size_t count = 4096;
    constexpr size_t int32_count = 128;
    constexpr size_t int8_count = 257;
    auto send0 = TestTensorFactory::createFP32({count});
    auto send1 = TestTensorFactory::createFP32({count});
    auto recv0 = TestTensorFactory::createFP32({count * 2});
    auto recv1 = TestTensorFactory::createFP32({count * 2});
    TestTensorFactory::fillValue(send0.get(), 1.0f);
    TestTensorFactory::fillValue(send1.get(), 2.0f);
    TestTensorFactory::fillValue(recv0.get(), -1.0f);
    TestTensorFactory::fillValue(recv1.get(), -1.0f);

    ASSERT_TRUE(send0->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(send1->ensureOnDevice(DeviceId::rocm(1)));
    ASSERT_TRUE(recv0->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(recv1->ensureOnDevice(DeviceId::rocm(1)));

    int32_t *send_i32_0 = nullptr;
    int32_t *send_i32_1 = nullptr;
    int32_t *recv_i32_0 = nullptr;
    int32_t *recv_i32_1 = nullptr;
    int8_t *send_i8_0 = nullptr;
    int8_t *send_i8_1 = nullptr;
    int8_t *recv_i8_0 = nullptr;
    int8_t *recv_i8_1 = nullptr;
    allocateAndUpload<int32_t>(0, std::vector<int32_t>(int32_count, 11), &send_i32_0);
    allocateAndUpload<int32_t>(1, std::vector<int32_t>(int32_count, 22), &send_i32_1);
    allocateAndUpload<int32_t>(0, std::vector<int32_t>(int32_count * 2, -1), &recv_i32_0);
    allocateAndUpload<int32_t>(1, std::vector<int32_t>(int32_count * 2, -1), &recv_i32_1);
    allocateAndUpload<int8_t>(0, std::vector<int8_t>(int8_count, 3), &send_i8_0);
    allocateAndUpload<int8_t>(1, std::vector<int8_t>(int8_count, 5), &send_i8_1);
    allocateAndUpload<int8_t>(0, std::vector<int8_t>(int8_count * 2, -1), &recv_i8_0);
    allocateAndUpload<int8_t>(1, std::vector<int8_t>(int8_count * 2, -1), &recv_i8_1);

    CapturedAllgatherResources resources0;
    CapturedAllgatherResources resources1;
    resources0.capture_stream = rocm_backend->createStream(0);
    resources1.capture_stream = rocm_backend->createStream(1);
    ASSERT_NE(resources0.capture_stream, nullptr);
    ASSERT_NE(resources1.capture_stream, nullptr);

    Barrier ready_to_capture(2);
    Barrier captured_collective(2);
    Barrier ready_to_launch(2);
    CaptureResult result0;
    CaptureResult result1;

    auto capture_worker = [&](int device,
                              TensorBase *send,
                              TensorBase *recv,
                              int32_t *send_i32,
                              int32_t *recv_i32,
                              int8_t *send_i8,
                              int8_t *recv_i8,
                              CapturedAllgatherResources *resources,
                              CaptureResult *result)
    {
        result->begin_status = hipSetDevice(device);
        if (result->begin_status == hipSuccess)
        {
            result->begin_status = hipStreamBeginCapture(
                static_cast<hipStream_t>(resources->capture_stream),
                hipStreamCaptureModeRelaxed);
        }

        ready_to_capture.arriveAndWait();

        if (result->begin_status == hipSuccess)
        {
            GraphCaptureGuard guard;
            result->fp32_allgather_ok = ctx->allgatherRawOnStream(
                send->gpu_data_ptr(),
                recv->gpu_data_ptr(),
                count,
                CollectiveDataType::FLOAT32,
                device,
                resources->capture_stream,
                "on_stream_graph_capture_regression_rccl");
            result->int32_allgather_ok = ctx->allgatherRawOnStream(
                send_i32,
                recv_i32,
                int32_count,
                CollectiveDataType::INT32,
                device,
                resources->capture_stream,
                "on_stream_graph_capture_regression_rccl_i32");
            result->int8_allgather_ok = ctx->allgatherRawOnStream(
                send_i8,
                recv_i8,
                int8_count,
                CollectiveDataType::INT8,
                device,
                resources->capture_stream,
                "on_stream_graph_capture_regression_rccl_i8");
            result->collective_ok =
                result->fp32_allgather_ok &&
                result->int32_allgather_ok &&
                result->int8_allgather_ok;
        }

        captured_collective.arriveAndWait();

        if (result->begin_status == hipSuccess &&
            result->end_status == hipSuccess &&
            result->collective_ok)
        {
            result->end_status = hipSetDevice(device);
            if (result->end_status == hipSuccess)
            {
                result->end_status = hipStreamEndCapture(
                    static_cast<hipStream_t>(resources->capture_stream),
                    &result->graph);
            }
        }

        if (result->end_status == hipSuccess && result->graph)
        {
            result->instantiate_status = hipSetDevice(device);
            if (result->instantiate_status == hipSuccess)
            {
                result->instantiate_status = hipGraphInstantiate(
                    &result->exec,
                    result->graph,
                    nullptr,
                    nullptr,
                    0);
            }
        }

        ready_to_launch.arriveAndWait();

        if (result->instantiate_status == hipSuccess && result->exec)
        {
            result->launch_status = hipSetDevice(device);
            if (result->launch_status == hipSuccess)
            {
                result->launch_status = hipGraphLaunch(
                    result->exec,
                    static_cast<hipStream_t>(resources->capture_stream));
            }
        }
    };

    std::thread t0(capture_worker,
                   0,
                   send0.get(),
                   recv0.get(),
                   send_i32_0,
                   recv_i32_0,
                   send_i8_0,
                   recv_i8_0,
                   &resources0,
                   &result0);
    std::thread t1(capture_worker,
                   1,
                   send1.get(),
                   recv1.get(),
                   send_i32_1,
                   recv_i32_1,
                   send_i8_1,
                   recv_i8_1,
                   &resources1,
                   &result1);
    t0.join();
    t1.join();

    EXPECT_EQ(result0.begin_status, hipSuccess);
    EXPECT_EQ(result1.begin_status, hipSuccess);
    EXPECT_TRUE(result0.fp32_allgather_ok);
    EXPECT_TRUE(result1.fp32_allgather_ok);
    EXPECT_TRUE(result0.int32_allgather_ok);
    EXPECT_TRUE(result1.int32_allgather_ok);
    EXPECT_TRUE(result0.int8_allgather_ok);
    EXPECT_TRUE(result1.int8_allgather_ok);
    EXPECT_TRUE(result0.collective_ok);
    EXPECT_TRUE(result1.collective_ok);
    EXPECT_EQ(result0.end_status, hipSuccess);
    EXPECT_EQ(result1.end_status, hipSuccess);
    EXPECT_NE(result0.graph, nullptr);
    EXPECT_NE(result1.graph, nullptr);
    EXPECT_EQ(result0.instantiate_status, hipSuccess);
    EXPECT_EQ(result1.instantiate_status, hipSuccess);
    EXPECT_EQ(result0.launch_status, hipSuccess);
    EXPECT_EQ(result1.launch_status, hipSuccess);

    ASSERT_TRUE(rocm_backend->synchronizeStream(resources0.capture_stream, 0));
    ASSERT_TRUE(rocm_backend->synchronizeStream(resources1.capture_stream, 1));

    std::vector<float> data0(count * 2);
    std::vector<float> data1(count * 2);
    ASSERT_TRUE(rocm_backend->deviceToHost(
        data0.data(),
        recv0->gpu_data_ptr(),
        data0.size() * sizeof(float),
        0));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        data1.data(),
        recv1->gpu_data_ptr(),
        data1.size() * sizeof(float),
        1));
    for (size_t i = 0; i < count; ++i)
    {
        ASSERT_FLOAT_EQ(data0[i], 1.0f) << "ROCm graph capture:0 first shard mismatch at index " << i;
        ASSERT_FLOAT_EQ(data0[count + i], 2.0f) << "ROCm graph capture:0 second shard mismatch at index " << i;
        ASSERT_FLOAT_EQ(data1[i], 1.0f) << "ROCm graph capture:1 first shard mismatch at index " << i;
        ASSERT_FLOAT_EQ(data1[count + i], 2.0f) << "ROCm graph capture:1 second shard mismatch at index " << i;
    }

    std::vector<int32_t> i32_data0(int32_count * 2);
    std::vector<int32_t> i32_data1(int32_count * 2);
    std::vector<int8_t> i8_data0(int8_count * 2);
    std::vector<int8_t> i8_data1(int8_count * 2);
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i32_data0.data(),
        recv_i32_0,
        i32_data0.size() * sizeof(int32_t),
        0));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i32_data1.data(),
        recv_i32_1,
        i32_data1.size() * sizeof(int32_t),
        1));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i8_data0.data(),
        recv_i8_0,
        i8_data0.size() * sizeof(int8_t),
        0));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i8_data1.data(),
        recv_i8_1,
        i8_data1.size() * sizeof(int8_t),
        1));
    for (size_t i = 0; i < int32_count; ++i)
    {
        ASSERT_EQ(i32_data0[i], 11) << "ROCm graph capture:0 int32 first shard mismatch at index " << i;
        ASSERT_EQ(i32_data0[int32_count + i], 22) << "ROCm graph capture:0 int32 second shard mismatch at index " << i;
        ASSERT_EQ(i32_data1[i], 11) << "ROCm graph capture:1 int32 first shard mismatch at index " << i;
        ASSERT_EQ(i32_data1[int32_count + i], 22) << "ROCm graph capture:1 int32 second shard mismatch at index " << i;
    }
    for (size_t i = 0; i < int8_count; ++i)
    {
        ASSERT_EQ(i8_data0[i], 3) << "ROCm graph capture:0 int8 first shard mismatch at index " << i;
        ASSERT_EQ(i8_data0[int8_count + i], 5) << "ROCm graph capture:0 int8 second shard mismatch at index " << i;
        ASSERT_EQ(i8_data1[i], 3) << "ROCm graph capture:1 int8 first shard mismatch at index " << i;
        ASSERT_EQ(i8_data1[int8_count + i], 5) << "ROCm graph capture:1 int8 second shard mismatch at index " << i;
    }

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    destroyAllgatherResources(rocm_backend, 0, resources0);
    destroyAllgatherResources(rocm_backend, 1, resources1);
    freeDevicePtr(0, send_i32_0);
    freeDevicePtr(1, send_i32_1);
    freeDevicePtr(0, recv_i32_0);
    freeDevicePtr(1, recv_i32_1);
    freeDevicePtr(0, send_i8_0);
    freeDevicePtr(1, send_i8_1);
    freeDevicePtr(0, recv_i8_0);
    freeDevicePtr(1, recv_i8_1);
}

TEST(Test__LocalTPRCCLGraphCapture, RCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found " << rocm_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};

    auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::RCCL);
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
    createP2PStreamsAndEvents(rocm_backend, aux_graph);

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
                               << " hipError=" << hipGetErrorString(aux_timing.first_error);
    std::cout << "[RCCLGroupedP2PMaintenanceGraph_AuxiliaryStream_TimingProbe]"
              << " payload_bytes=" << payload_bytes
              << " iterations=" << replay_iterations
              << " aux_wall_ms=" << aux_timing.wall_ms
              << std::endl;

    std::vector<int8_t> recv_host0(payload_bytes);
    std::vector<int8_t> recv_host1(payload_bytes);
    ASSERT_TRUE(rocm_backend->deviceToHost(
        recv_host0.data(),
        recv0,
        recv_host0.size() * sizeof(int8_t),
        0));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        recv_host1.data(),
        recv1,
        recv_host1.size() * sizeof(int8_t),
        1));
    for (size_t i = 0; i < recv_host0.size(); i += 4096)
    {
        EXPECT_EQ(recv_host0[i], 5) << "device0 recv mismatch at byte " << i;
        EXPECT_EQ(recv_host1[i], 3) << "device1 recv mismatch at byte " << i;
    }

    destroyP2PGraph(rocm_backend, aux_graph);
    freeDevicePtr(0, send0);
    freeDevicePtr(1, send1);
    freeDevicePtr(0, recv0);
    freeDevicePtr(1, recv1);
}

TEST(Test__LocalTPRCCLGraphCapture, DISABLED_RCCLDecodeMaintenanceOverlap_NoBarrierAfterMaintenanceLaunch)
{
    runRcclDecodeMaintenanceOverlapLab(
        RcclOverlapPattern::NoBarrierAfterMaintenanceLaunch);
}

TEST(Test__LocalTPRCCLGraphCapture, RCCLDecodeMaintenanceOverlap_BarrierAfterMaintenanceLaunch_Completes)
{
    runRcclDecodeMaintenanceOverlapLab(
        RcclOverlapPattern::BarrierAfterMaintenanceLaunch);
}

TEST(Test__LocalTPRCCLGraphCapture, DISABLED_RCCLDecodeMaintenanceOverlap_DecodeWaitsForMaintenanceCompletion)
{
    runRcclDecodeMaintenanceOverlapLab(
        RcclOverlapPattern::DecodeWaitsForMaintenanceCompletion);
}

#else

TEST(Test__LocalTPRCCLGraphCapture, SkipsWithoutROCm)
{
    GTEST_SKIP() << "ROCm not enabled";
}

#endif
