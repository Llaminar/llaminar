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
#include <condition_variable>
#include <cstdint>
#include <mutex>
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

#else

TEST(Test__LocalTPRCCLGraphCapture, SkipsWithoutROCm)
{
    GTEST_SKIP() << "ROCm not enabled";
}

#endif
