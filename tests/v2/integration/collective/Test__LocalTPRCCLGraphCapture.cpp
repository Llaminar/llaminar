/**
 * @file Test__LocalTPRCCLGraphCapture.cpp
 * @brief ROCm-only LocalTP graph capture regression tests.
 *
 * Keep this in a separate translation unit from CUDA tests: cuda_runtime.h and
 * hip_runtime.h expose overlapping vector types and are intentionally not mixed.
 */

#include <gtest/gtest.h>

#ifdef HAVE_ROCM

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
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
#include "backends/rocm/HIPGraphCapture.h"
#include "collective/ICollectiveBackend.h"
#include "collective/LocalTPContext.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "kernels/common/TPRankOrderedReductionKernels.h"
#include "kernels/rocm/moe/ROCmMoEKernel.h"
#include "tensors/TensorClasses.h"
#include "transfer/TransferEngine.h"
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
        bool k_allgather_ok = false;
        bool v_allgather_ok = false;
        hipError_t begin_status = hipSuccess;
        hipError_t pre_collective_status = hipSuccess;
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

    /**
     * @brief Persistent resources for a production-shaped layered allgather graph.
     *
     * Qwen3.6 prefill LLEP records one compute-to-transfer event, one compact
     * payload allgather, and one transfer-to-compute event for every routed
     * layer. Adjacent layers use two rolling transfer lanes, while every layer
     * owns distinct event objects so captured dependencies cannot alias a later
     * record. This fixture mirrors that topology without loading model weights.
     */
    struct LayeredAllgatherGraph
    {
        std::array<CaptureResult, 2> results{};
        std::array<void *, 2> compute_streams{};
        std::array<std::array<void *, 2>, 2> transfer_streams{};
        std::array<std::vector<hipEvent_t>, 2> compute_ready_events{};
        std::array<std::vector<hipEvent_t>, 2> transfer_done_events{};
        std::array<hipEvent_t, 2> start_events{};
        std::array<hipEvent_t, 2> stop_events{};
    };

    /**
     * @brief GPU-event distribution for one captured graph topology.
     */
    struct LayeredAllgatherTiming
    {
        double minimum_us = 0.0;
        double median_us = 0.0;
        double p95_us = 0.0;
        double maximum_us = 0.0;
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

    /**
     * @brief Copy one device vector to host at the terminal verification boundary.
     *
     * Synchronization is deliberately confined to this integration-test result
     * observation. The captured production transaction being tested contains
     * no host transfer or host wait.
     */
    template <typename T>
    void downloadDeviceVector(
        int device,
        const T *device_ptr,
        std::vector<T> *host_values)
    {
        ASSERT_NE(device_ptr, nullptr);
        ASSERT_NE(host_values, nullptr);
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        hipStream_t stream = nullptr;
        ASSERT_EQ(
            hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpyAsync(
                host_values->data(),
                device_ptr,
                host_values->size() * sizeof(T),
                hipMemcpyDeviceToHost,
                stream),
            hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        ASSERT_EQ(hipStreamDestroy(stream), hipSuccess);
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

    /**
     * @brief Copies a rank-major two-shard FP32 allgather result to host and verifies both shards.
     *
     * LocalTP raw allgather writes shard 0 followed by shard 1 on every
     * participating device.  This helper intentionally checks the complete
     * payload rather than sampling so late-row K/V corruption cannot hide in
     * the tail of a large prefix handoff.
     *
     * @param backend ROCm backend used to copy device memory back to the host.
     * @param device ROCm ordinal that owns @p device_buffer.
     * @param stream Exact consumer stream ordered after the captured graph.
     * @param device_buffer Device pointer containing two rank-major shards.
     * @param shard_count Number of FP32 elements in each rank-local shard.
     * @param first_value Expected value for rank 0's shard.
     * @param second_value Expected value for rank 1's shard.
     * @param label Diagnostic label included in assertion messages.
     */
    void expectGatheredFloatShards(
        IBackend *backend,
        int device,
        void *stream,
        const float *device_buffer,
        size_t shard_count,
        float first_value,
        float second_value,
        const char *label)
    {
        ASSERT_NE(backend, nullptr);
        ASSERT_NE(stream, nullptr);
        ASSERT_NE(device_buffer, nullptr);

        std::vector<float> host(shard_count * 2);
        ASSERT_TRUE(backend->deviceToHost(
            host.data(),
            device_buffer,
            host.size() * sizeof(float),
            device,
            stream));
        for (size_t i = 0; i < shard_count; ++i)
        {
            ASSERT_FLOAT_EQ(host[i], first_value)
                << label << ": device " << device
                << " rank0 shard mismatch at element " << i;
            ASSERT_FLOAT_EQ(host[shard_count + i], second_value)
                << label << ": device " << device
                << " rank1 shard mismatch at element " << i;
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

    /**
     * @brief Capture a prefill-like stream segment with work before RCCL allreduce.
     *
     * The E2E prefix prefill graph records an embedding kernel before the
     * embedding_allreduce collective. A collective-only graph can succeed even
     * when RCCL rejects a grouped collective appended after existing captured
     * nodes, so this helper records a simple HIP memset before the same LocalTP
     * on-stream allreduce.  The memset targets a separate scratch allocation so
     * the test still proves the captured RCCL payload preserves nonzero
     * per-rank activation values and not merely that an all-zero tensor can be
     * replayed.
     *
     * @param ctx LocalTP context under test.
     * @param tensor0 Device-0 tensor to reduce in place.
     * @param tensor1 Device-1 tensor to reduce in place.
     * @param stream0 Capture stream for ROCm device 0.
     * @param stream1 Capture stream for ROCm device 1.
     * @param count FP32 element count reduced by the collective.
     * @param result0 Capture status for device 0.
     * @param result1 Capture status for device 1.
     */
    void capturePrefillLikeAllreduceGraph(
        ILocalTPContext &ctx,
        TensorBase *tensor0,
        TensorBase *tensor1,
        void *stream0,
        void *stream1,
        void *scratch0,
        void *scratch1,
        size_t count,
        CaptureResult &result0,
        CaptureResult &result1)
    {
        Barrier ready_to_capture(2);
        Barrier captured_collective(2);

        auto capture_worker = [&](int device,
                                  TensorBase *tensor,
                                  void *stream,
                                  void *scratch,
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
                result->pre_collective_status = hipMemsetAsync(
                    scratch,
                    0,
                    sizeof(uint32_t),
                    static_cast<hipStream_t>(stream));
                if (result->pre_collective_status == hipSuccess)
                {
                    GraphCaptureGuard guard;
                    result->collective_ok = ctx.allreduceOnStream(
                        tensor,
                        "embedding_allreduce",
                        count,
                        stream,
                        "fp32");
                }
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

        std::thread t0(capture_worker, 0, tensor0, stream0, scratch0, &result0);
        std::thread t1(capture_worker, 1, tensor1, stream1, scratch1, &result1);
        t0.join();
        t1.join();
    }

    /**
     * @brief Capture one FP16 activation allreduce, one allgather sideband,
     *        and the activation's immediate consumer in one RCCL group.
     *
     * Dynamic maintenance already attaches future-epoch control traffic to an
     * activation collective.  Current-batch LLEP may use the same mechanism
     * only if graph construction can expose a causally valid pre-routed-FFN
     * anchor.  This helper proves the backend primitive independently of that
     * model-graph decision: both participant graphs must publish the reduced
     * activation and the complete rank-major sideband on every replay.
     *
     * The caller performs one eager grouped launch before capture so FP16
     * transport scratch is persistent.  No allocation, host transfer, or host
     * synchronization occurs inside the captured transaction.
     *
     * @param ctx Shared two-device LocalTP RCCL context.
     * @param tensor0 Device-0 FP32 activation buffer.
     * @param tensor1 Device-1 FP32 activation buffer.
     * @param stream0 Explicit capture stream for ROCm device 0.
     * @param stream1 Explicit capture stream for ROCm device 1.
     * @param consumer0 Persistent device-0 immediate-consumer buffer.
     * @param consumer1 Persistent device-1 immediate-consumer buffer.
     * @param count Number of FP32 activation elements.
     * @param sidebands0 Device-0 sideband descriptors.
     * @param sidebands1 Device-1 sideband descriptors.
     * @param result0 Device-0 capture result.
     * @param result1 Device-1 capture result.
     */
    void captureFP16AllreduceWithSidebandAndConsumer(
        ILocalTPContext &ctx,
        TensorBase *tensor0,
        TensorBase *tensor1,
        void *stream0,
        void *stream1,
        float *consumer0,
        float *consumer1,
        size_t count,
        const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands0,
        const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands1,
        CaptureResult &result0,
        CaptureResult &result1)
    {
        ASSERT_NE(tensor0, nullptr);
        ASSERT_NE(tensor1, nullptr);
        ASSERT_NE(stream0, nullptr);
        ASSERT_NE(stream1, nullptr);
        ASSERT_NE(consumer0, nullptr);
        ASSERT_NE(consumer1, nullptr);
        ASSERT_FALSE(sidebands0.empty());
        ASSERT_EQ(sidebands0.size(), sidebands1.size());

        Barrier ready_to_capture(2);
        Barrier collective_recorded(2);

        auto capture_worker =
            [&](int device,
                TensorBase *tensor,
                void *stream,
                float *consumer,
                const std::vector<LocalTPCollectiveSidebandBuffer> *sidebands,
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
                result->collective_ok = ctx.allreduceWithSidebandsOnStream(
                    tensor,
                    "layer0_moe_combined_allreduce_with_rebalance_sidebands",
                    count,
                    stream,
                    "fp16",
                    *sidebands,
                    device);
                if (result->collective_ok)
                {
                    result->launch_status = hipMemcpyAsync(
                        consumer,
                        tensor->gpu_data_ptr(),
                        count * sizeof(float),
                        hipMemcpyDeviceToDevice,
                        static_cast<hipStream_t>(stream));
                }
            }

            /*
             * The final participant submits the complete multi-stream RCCL
             * group.  Keep both relaxed captures live until that rendezvous
             * has returned so neither graph ends with a partial collective.
             */
            collective_recorded.arriveAndWait();

            if (result->begin_status == hipSuccess &&
                result->launch_status == hipSuccess &&
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

        std::thread worker0(
            capture_worker,
            0,
            tensor0,
            stream0,
            consumer0,
            &sidebands0,
            &result0);
        std::thread worker1(
            capture_worker,
            1,
            tensor1,
            stream1,
            consumer1,
            &sidebands1,
            &result1);
        worker0.join();
        worker1.join();
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

    /**
     * @brief Allocate streams and per-layer event identities before capture.
     *
     * @param backend ROCm backend that owns the explicit streams.
     * @param layer_count Number of MoE layer transactions recorded in the graph.
     * @param graph Resource bundle to populate.
     */
    void createLayeredAllgatherResources(
        IBackend *backend,
        size_t layer_count,
        LayeredAllgatherGraph &graph)
    {
        ASSERT_NE(backend, nullptr);
        ASSERT_GT(layer_count, 0u);

        for (int participant = 0; participant < 2; ++participant)
        {
            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            graph.compute_streams[static_cast<size_t>(participant)] =
                backend->createStream(participant);
            ASSERT_NE(
                graph.compute_streams[static_cast<size_t>(participant)],
                nullptr);

            for (size_t lane = 0; lane < 2; ++lane)
            {
                graph.transfer_streams[static_cast<size_t>(participant)][lane] =
                    backend->createStream(participant);
                ASSERT_NE(
                    graph.transfer_streams[static_cast<size_t>(participant)][lane],
                    nullptr);
            }

            auto &ready =
                graph.compute_ready_events[static_cast<size_t>(participant)];
            auto &done =
                graph.transfer_done_events[static_cast<size_t>(participant)];
            ready.resize(layer_count, nullptr);
            done.resize(layer_count, nullptr);
            for (size_t layer = 0; layer < layer_count; ++layer)
            {
                ASSERT_EQ(
                    hipEventCreateWithFlags(
                        &ready[layer],
                        hipEventDisableTiming),
                    hipSuccess);
                ASSERT_EQ(
                    hipEventCreateWithFlags(
                        &done[layer],
                        hipEventDisableTiming),
                    hipSuccess);
            }

            ASSERT_EQ(
                hipEventCreateWithFlags(
                    &graph.start_events[static_cast<size_t>(participant)],
                    hipEventDefault),
                hipSuccess);
            ASSERT_EQ(
                hipEventCreateWithFlags(
                    &graph.stop_events[static_cast<size_t>(participant)],
                    hipEventDefault),
                hipSuccess);
        }
    }

    /**
     * @brief Release a layered graph after all replay and observation ends.
     *
     * Graph executables are destroyed before events, streams, and the RCCL
     * context because their nodes retain those resource identities.
     */
    void destroyLayeredAllgatherResources(
        IBackend *backend,
        LayeredAllgatherGraph &graph)
    {
        ASSERT_NE(backend, nullptr);
        for (int participant = 0; participant < 2; ++participant)
        {
            const size_t index = static_cast<size_t>(participant);
            EXPECT_EQ(hipSetDevice(participant), hipSuccess);
            destroyCaptureResult(graph.results[index]);

            for (hipEvent_t event : graph.compute_ready_events[index])
            {
                if (event)
                    EXPECT_EQ(hipEventDestroy(event), hipSuccess);
            }
            for (hipEvent_t event : graph.transfer_done_events[index])
            {
                if (event)
                    EXPECT_EQ(hipEventDestroy(event), hipSuccess);
            }
            graph.compute_ready_events[index].clear();
            graph.transfer_done_events[index].clear();

            if (graph.start_events[index])
            {
                EXPECT_EQ(hipEventDestroy(graph.start_events[index]), hipSuccess);
                graph.start_events[index] = nullptr;
            }
            if (graph.stop_events[index])
            {
                EXPECT_EQ(hipEventDestroy(graph.stop_events[index]), hipSuccess);
                graph.stop_events[index] = nullptr;
            }
            for (size_t lane = 0; lane < 2; ++lane)
            {
                if (graph.transfer_streams[index][lane])
                {
                    backend->destroyStream(
                        graph.transfer_streams[index][lane],
                        participant);
                    graph.transfer_streams[index][lane] = nullptr;
                }
            }
            if (graph.compute_streams[index])
            {
                backend->destroyStream(
                    graph.compute_streams[index],
                    participant);
                graph.compute_streams[index] = nullptr;
            }
        }
    }

    /**
     * @brief Capture the same fixed-size allgather once per modeled MoE layer.
     *
     * With @p auxiliary_lanes false, all collectives are recorded directly on
     * the graph's compute stream. With it true, layer N records the production
     * compute -> lane(N % 2) -> compute event chain around its allgather. The
     * latter is the topology used by transfer-backed prefill LLEP.
     *
     * @param ctx Production LocalTP collective authority.
     * @param send_buffers Participant-local compact payloads.
     * @param recv_buffers Participant-local rank-major gathered payloads.
     * @param send_count Number of INT8 payload bytes contributed per participant.
     * @param layer_count Number of repeated layer transactions to capture.
     * @param auxiliary_lanes Whether to route collectives through two transfer lanes.
     * @param graph Preallocated graph resource bundle.
     */
    void captureLayeredAllgatherGraph(
        ILocalTPContext &ctx,
        const std::array<const void *, 2> &send_buffers,
        const std::array<void *, 2> &recv_buffers,
        size_t send_count,
        size_t layer_count,
        bool auxiliary_lanes,
        LayeredAllgatherGraph &graph)
    {
        Barrier capture_started(2);
        Barrier all_collectives_recorded(2);

        auto capture_worker = [&](int participant)
        {
            const size_t index = static_cast<size_t>(participant);
            CaptureResult &result = graph.results[index];
            auto compute_stream =
                static_cast<hipStream_t>(graph.compute_streams[index]);

            result.begin_status = hipSetDevice(participant);
            if (result.begin_status == hipSuccess)
            {
                result.begin_status = hipStreamBeginCapture(
                    compute_stream,
                    hipStreamCaptureModeRelaxed);
            }

            capture_started.arriveAndWait();
            result.collective_ok = result.begin_status == hipSuccess;
            if (result.collective_ok)
            {
                GraphCaptureGuard guard;
                for (size_t layer = 0; layer < layer_count; ++layer)
                {
                    void *collective_stream = graph.compute_streams[index];
                    if (auxiliary_lanes)
                    {
                        const size_t lane = layer % 2u;
                        collective_stream = graph.transfer_streams[index][lane];
                        result.launch_status = hipEventRecord(
                            graph.compute_ready_events[index][layer],
                            compute_stream);
                        if (result.launch_status == hipSuccess)
                        {
                            result.launch_status = hipStreamWaitEvent(
                                static_cast<hipStream_t>(collective_stream),
                                graph.compute_ready_events[index][layer],
                                0);
                        }
                    }

                    const bool gathered = ctx.allgatherRawOnStream(
                        send_buffers[index],
                        recv_buffers[index],
                        send_count,
                        CollectiveDataType::INT8,
                        participant,
                        collective_stream,
                        auxiliary_lanes
                            ? "rccl_llep_layered_auxiliary_allgather"
                            : "rccl_llep_layered_compute_allgather");
                    result.collective_ok = result.collective_ok && gathered;

                    if (auxiliary_lanes &&
                        result.launch_status == hipSuccess)
                    {
                        result.launch_status = hipEventRecord(
                            graph.transfer_done_events[index][layer],
                            static_cast<hipStream_t>(collective_stream));
                        if (result.launch_status == hipSuccess)
                        {
                            result.launch_status = hipStreamWaitEvent(
                                compute_stream,
                                graph.transfer_done_events[index][layer],
                                0);
                        }
                    }
                }
            }

            all_collectives_recorded.arriveAndWait();
            if (result.begin_status == hipSuccess &&
                result.launch_status == hipSuccess &&
                result.collective_ok)
            {
                result.end_status = hipStreamEndCapture(
                    compute_stream,
                    &result.graph);
            }
            if (result.end_status == hipSuccess && result.graph)
            {
                result.instantiate_status = hipGraphInstantiate(
                    &result.exec,
                    result.graph,
                    nullptr,
                    nullptr,
                    0);
            }
        };

        std::thread worker0(capture_worker, 0);
        std::thread worker1(capture_worker, 1);
        worker0.join();
        worker1.join();
    }

    /**
     * @brief Measure one paired graph replay with events on both compute streams.
     *
     * The stop event follows the graph launch on the compute stream. Auxiliary
     * work is included because every transfer lane explicitly rejoins that
     * stream before graph completion.
     */
    double measureLayeredAllgatherGraphOnce(LayeredAllgatherGraph &graph)
    {
        Barrier replay_started(2);
        std::array<hipError_t, 2> statuses{hipSuccess, hipSuccess};
        std::array<double, 2> elapsed_us{0.0, 0.0};

        auto replay_worker = [&](int participant)
        {
            const size_t index = static_cast<size_t>(participant);
            auto compute_stream =
                static_cast<hipStream_t>(graph.compute_streams[index]);
            hipError_t &status = statuses[index];
            status = hipSetDevice(participant);
            replay_started.arriveAndWait();
            if (status == hipSuccess)
            {
                status = hipEventRecord(
                    graph.start_events[index],
                    compute_stream);
            }
            if (status == hipSuccess)
            {
                status = hipGraphLaunch(
                    graph.results[index].exec,
                    compute_stream);
            }
            if (status == hipSuccess)
            {
                status = hipEventRecord(
                    graph.stop_events[index],
                    compute_stream);
            }
            if (status == hipSuccess)
                status = hipEventSynchronize(graph.stop_events[index]);
            if (status == hipSuccess)
            {
                float elapsed_ms = 0.0f;
                status = hipEventElapsedTime(
                    &elapsed_ms,
                    graph.start_events[index],
                    graph.stop_events[index]);
                elapsed_us[index] = static_cast<double>(elapsed_ms) * 1000.0;
            }
        };

        std::thread worker0(replay_worker, 0);
        std::thread worker1(replay_worker, 1);
        worker0.join();
        worker1.join();

        EXPECT_EQ(statuses[0], hipSuccess)
            << "participant 0 replay: " << hipGetErrorString(statuses[0]);
        EXPECT_EQ(statuses[1], hipSuccess)
            << "participant 1 replay: " << hipGetErrorString(statuses[1]);
        return std::max(elapsed_us[0], elapsed_us[1]);
    }

    /**
     * @brief Warm and summarize repeated paired graph replays.
     */
    LayeredAllgatherTiming benchmarkLayeredAllgatherGraph(
        LayeredAllgatherGraph &graph,
        int warmup_iterations,
        int measured_iterations)
    {
        EXPECT_GE(warmup_iterations, 0);
        EXPECT_GT(measured_iterations, 0);
        for (int iteration = 0; iteration < warmup_iterations; ++iteration)
            (void)measureLayeredAllgatherGraphOnce(graph);

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>(measured_iterations));
        for (int iteration = 0; iteration < measured_iterations; ++iteration)
            samples.push_back(measureLayeredAllgatherGraphOnce(graph));
        std::sort(samples.begin(), samples.end());

        LayeredAllgatherTiming timing;
        if (samples.empty())
            return timing;
        timing.minimum_us = samples.front();
        timing.median_us = samples[samples.size() / 2u];
        const size_t p95_index = std::min(
            samples.size() - 1u,
            (samples.size() * 95u) / 100u);
        timing.p95_us = samples[p95_index];
        timing.maximum_us = samples.back();
        return timing;
    }

    /**
     * @brief Return the number of native nodes retained by one HIP graph.
     */
    size_t layeredAllgatherNodeCount(const CaptureResult &result)
    {
        size_t node_count = 0;
        EXPECT_NE(result.graph, nullptr);
        if (result.graph)
            EXPECT_EQ(hipGraphGetNodes(result.graph, nullptr, &node_count), hipSuccess);
        return node_count;
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

    /**
     * @brief Capture, instantiate, launch, and verify a two-device RCCL allreduce graph.
     *
     * This helper mirrors the inference graph-capture shape used by LocalTP
     * workers: each ROCm device captures its own stream on its own host thread,
     * and LocalTPContext coordinates one grouped RCCL on-stream collective
     * across those participant streams. Keeping the payload size configurable
     * lets tests cover both small decode collectives and prefill-sized
     * embedding/hidden-state allreduces.
     *
     * @param count FP32 elements per device buffer.
     * @param label Diagnostic label for assertion messages.
     * @param capture_repetitions Number of capture/launch cycles to run on the same context and streams.
     * @param pre_collective_work Record a HIP operation before the allreduce, matching prefill segments.
     */
    void runCapturedAllreduceGraphPayload(
        size_t count,
        const char *label,
        int capture_repetitions = 1,
        bool pre_collective_work = false)
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
        void *scratch0 = nullptr;
        void *scratch1 = nullptr;
        if (pre_collective_work)
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess) << label;
            ASSERT_EQ(hipMalloc(&scratch0, sizeof(uint32_t)), hipSuccess)
                << label << " device0 scratch allocation failed";
            ASSERT_EQ(hipSetDevice(1), hipSuccess) << label;
            ASSERT_EQ(hipMalloc(&scratch1, sizeof(uint32_t)), hipSuccess)
                << label << " device1 scratch allocation failed";
        }

        ASSERT_GT(capture_repetitions, 0) << label;
        float expected_value = 0.0f;
        for (int repetition = 0; repetition < capture_repetitions; ++repetition)
        {
            CaptureResult result0;
            CaptureResult result1;
            if (pre_collective_work)
            {
                capturePrefillLikeAllreduceGraph(
                    *ctx,
                    tensor0.get(),
                    tensor1.get(),
                    stream0,
                    stream1,
                    scratch0,
                    scratch1,
                    count,
                    result0,
                    result1);
            }
            else
            {
                captureDecodeAllreduceGraph(
                    *ctx,
                    tensor0.get(),
                    tensor1.get(),
                    stream0,
                    stream1,
                    count,
                    result0,
                    result1);
            }

            expectCapturedGraphReady(result0, "allreduce0");
            expectCapturedGraphReady(result1, "allreduce1");
            EXPECT_EQ(result0.pre_collective_status, hipSuccess) << label << " repetition " << repetition;
            EXPECT_EQ(result1.pre_collective_status, hipSuccess) << label << " repetition " << repetition;
            ASSERT_FALSE(::testing::Test::HasFailure()) << label << " repetition " << repetition;

            ASSERT_EQ(hipSetDevice(0), hipSuccess) << label;
            result0.launch_status = hipGraphLaunch(
                result0.exec,
                static_cast<hipStream_t>(stream0));
            ASSERT_EQ(hipSetDevice(1), hipSuccess) << label;
            result1.launch_status = hipGraphLaunch(
                result1.exec,
                static_cast<hipStream_t>(stream1));
            EXPECT_EQ(result0.launch_status, hipSuccess) << label << " repetition " << repetition;
            EXPECT_EQ(result1.launch_status, hipSuccess) << label << " repetition " << repetition;

            ASSERT_TRUE(rocm_backend->synchronizeStream(stream0, 0)) << label << " repetition " << repetition;
            ASSERT_TRUE(rocm_backend->synchronizeStream(stream1, 1)) << label << " repetition " << repetition;

            expected_value = repetition == 0 ? 3.0f : expected_value * 2.0f;
            destroyCaptureResult(result0);
            destroyCaptureResult(result1);
        }

        const float *data0 = tensor0->data();
        const float *data1 = tensor1->data();
        ASSERT_NE(data0, nullptr) << label;
        ASSERT_NE(data1, nullptr) << label;
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(data0[i], expected_value) << label << " device0 mismatch at index " << i;
            EXPECT_FLOAT_EQ(data1[i], expected_value) << label << " device1 mismatch at index " << i;
        }
        freeDevicePtr(0, scratch0);
        freeDevicePtr(1, scratch1);
        rocm_backend->destroyStream(stream0, 0);
        rocm_backend->destroyStream(stream1, 1);
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

/**
 * @brief Captured RCCL activation and allgather sidebands publish atomically.
 *
 * This is the ROCm peer of the NCCL grouped-sideband regression.  It proves
 * the exact primitive needed to carry an LLEP expert payload on a causally
 * independent activation collective: FP16 activation transport, an INT32
 * rank-major allgather, and the activation's first consumer all remain inside
 * each participant's complete captured graph.  Two replays with different
 * values reject an eager-warmup or stale-buffer false positive.
 */
TEST(
    Test__LocalTPRCCLGraphCapture,
    RCCLFP16AllreduceWithAllgatherSidebandGraph_EveryReplayPublishesWholeBundle)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found "
                     << rocm_backend->deviceCount();
    }

    constexpr size_t kPromptRows = 9;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    constexpr size_t kElementCount =
        kPromptRows * kQwen36MoEHiddenDim;
    constexpr size_t kSidebandElementCount = 128;
    constexpr size_t kGatheredSidebandElementCount =
        2 * kSidebandElementCount;

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsCollectiveSidebandOnStreamGraphCapture());

    auto tensor0 = TestTensorFactory::createFP32({kElementCount});
    auto tensor1 = TestTensorFactory::createFP32({kElementCount});
    TestTensorFactory::fillValue(tensor0.get(), 1.0f);
    TestTensorFactory::fillValue(tensor1.get(), 2.0f);
    ASSERT_TRUE(tensor0->ensureOnDevice(DeviceId::rocm(0)));
    ASSERT_TRUE(tensor1->ensureOnDevice(DeviceId::rocm(1)));

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

    void *stream0 = rocm_backend->createStream(0);
    void *stream1 = rocm_backend->createStream(1);
    ASSERT_NE(stream0, nullptr);
    ASSERT_NE(stream1, nullptr);
    float *consumer0 = nullptr;
    float *consumer1 = nullptr;
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&consumer0),
            kElementCount * sizeof(float)),
        hipSuccess);
    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&consumer1),
            kElementCount * sizeof(float)),
        hipSuccess);

    /*
     * LocalTP owns persistent FP16 conversion buffers.  Materialize them and
     * validate the same grouped RCCL call before capture so capture itself is
     * allocation-free.
     */
    bool warmup0 = false;
    bool warmup1 = false;
    std::thread warmup_worker0(
        [&]
        {
            ASSERT_EQ(hipSetDevice(0), hipSuccess);
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
            ASSERT_EQ(hipSetDevice(1), hipSuccess);
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
    ASSERT_TRUE(rocm_backend->synchronizeStream(stream0, 0));
    ASSERT_TRUE(rocm_backend->synchronizeStream(stream1, 1));

    CaptureResult result0;
    CaptureResult result1;
    captureFP16AllreduceWithSidebandAndConsumer(
        *ctx,
        tensor0.get(),
        tensor1.get(),
        stream0,
        stream1,
        consumer0,
        consumer1,
        kElementCount,
        sidebands0,
        sidebands1,
        result0,
        result1);
    expectCapturedGraphReady(result0, "fp16_sideband_bundle_graph0");
    expectCapturedGraphReady(result1, "fp16_sideband_bundle_graph1");
    ASSERT_FALSE(::testing::Test::HasFailure());

    std::cout
        << "[RCCL_FP16_ALLREDUCE_ALLGATHER_SIDEBAND]"
        << " graph_nodes="
        << layeredAllgatherNodeCount(result0) << ','
        << layeredAllgatherNodeCount(result1)
        << std::endl;

    const auto replay_and_expect =
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

        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                tensor0->gpu_data_ptr(),
                input0.data(),
                kElementCount * sizeof(float),
                hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                sideband_send0,
                sideband_input0.data(),
                kSidebandElementCount * sizeof(int32_t),
                hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                tensor1->gpu_data_ptr(),
                input1.data(),
                kElementCount * sizeof(float),
                hipMemcpyHostToDevice),
            hipSuccess);
        ASSERT_EQ(
            hipMemcpy(
                sideband_send1,
                sideband_input1.data(),
                kSidebandElementCount * sizeof(int32_t),
                hipMemcpyHostToDevice),
            hipSuccess);

        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        result0.launch_status = hipGraphLaunch(
            result0.exec,
            static_cast<hipStream_t>(stream0));
        ASSERT_EQ(result0.launch_status, hipSuccess)
            << hipGetErrorString(result0.launch_status);
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        result1.launch_status = hipGraphLaunch(
            result1.exec,
            static_cast<hipStream_t>(stream1));
        ASSERT_EQ(result1.launch_status, hipSuccess)
            << hipGetErrorString(result1.launch_status);
        ASSERT_TRUE(rocm_backend->synchronizeStream(stream0, 0));
        ASSERT_TRUE(rocm_backend->synchronizeStream(stream1, 1));

        std::vector<float> output0(kElementCount, 0.0f);
        std::vector<float> output1(kElementCount, 0.0f);
        std::vector<int32_t> gathered0(
            kGatheredSidebandElementCount,
            0);
        std::vector<int32_t> gathered1(
            kGatheredSidebandElementCount,
            0);
        downloadDeviceVector(0, consumer0, &output0);
        downloadDeviceVector(1, consumer1, &output1);
        downloadDeviceVector(0, sideband_recv0, &gathered0);
        downloadDeviceVector(1, sideband_recv1, &gathered1);

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
        for (size_t element = 0;
             element < kSidebandElementCount;
             ++element)
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
    freeDevicePtr(0, consumer0);
    freeDevicePtr(1, consumer1);
    freeDevicePtr(0, sideband_send0);
    freeDevicePtr(1, sideband_send1);
    freeDevicePtr(0, sideband_recv0);
    freeDevicePtr(1, sideband_recv1);
    rocm_backend->destroyStream(stream0, 0);
    rocm_backend->destroyStream(stream1, 1);
}

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

/**
 * @brief Prefill-sized RCCL on-stream graph capture must be legal.
 *
 * The Qwen3.6 MoE ROCm2TP prefix tests capture an embedding_allreduce over a
 * full prefill bucket. A small decode-sized captured allreduce can pass while
 * the larger prefill payload still trips RCCL/HIP capture constraints, so this
 * regression locks the payload class that failed in the E2E suite.
 */
TEST(Test__LocalTPRCCLGraphCapture, RCCLAllreduce_OnStreamGraphCapture_PrefillPayloadCompletes)
{
    constexpr size_t kPrefillSeqLen = 256;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    runCapturedAllreduceGraphPayload(
        kPrefillSeqLen * kQwen36MoEHiddenDim,
        "prefill-sized captured RCCL allreduce");
}

/**
 * @brief Repeated prefill-sized capture must not poison later RCCL capture.
 *
 * Prefix-cache validation captures and resets a prefill graph before capturing
 * another prefill graph on the same LocalTP/RCCL machinery. This regression
 * keeps the second capture honest so stale grouped-launch or stream-capture
 * state cannot hide behind a one-shot success.
 */
TEST(Test__LocalTPRCCLGraphCapture, RCCLAllreduce_OnStreamGraphCapture_RepeatedPrefillPayloadCompletes)
{
    constexpr size_t kPrefillSeqLen = 256;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    runCapturedAllreduceGraphPayload(
        kPrefillSeqLen * kQwen36MoEHiddenDim,
        "repeated prefill-sized captured RCCL allreduce",
        2);
}

/**
 * @brief Prefill-like captured stream work before RCCL must remain legal.
 *
 * This is the closest local reproduction of the E2E embedding_allreduce
 * failure: each participant stream already contains captured GPU work before
 * the final-arrival thread records the grouped RCCL allreduce.
 */
TEST(Test__LocalTPRCCLGraphCapture, RCCLAllreduce_OnStreamGraphCapture_PrefillLikeSegmentCompletes)
{
    constexpr size_t kPrefillSeqLen = 256;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    runCapturedAllreduceGraphPayload(
        kPrefillSeqLen * kQwen36MoEHiddenDim,
        "prefill-like captured RCCL allreduce",
        1,
        true);
}

/**
 * @brief Regress repeated prefill-like captures on the same RCCL context.
 *
 * Long-context bucketed prefill warms a bucket, captures it on the next chunk,
 * launches the captured graph, and later captures the same prefill-shaped
 * collective pattern again.  A single pre-collective work item is not enough
 * coverage for that lifecycle because stale stream/capture bookkeeping can
 * survive the first graph launch and only poison the following capture.
 */
TEST(Test__LocalTPRCCLGraphCapture, RCCLAllreduce_OnStreamGraphCapture_RepeatedPrefillLikeSegmentCompletes)
{
    constexpr size_t kPrefillSeqLen = 256;
    constexpr size_t kQwen36MoEHiddenDim = 2048;
    runCapturedAllreduceGraphPayload(
        kPrefillSeqLen * kQwen36MoEHiddenDim,
        "repeated prefill-like captured RCCL allreduce",
        2,
        true);
}

/**
 * @brief LocalTP prefill graph-capture boundaries must be domain-wide barriers.
 *
 * The ROCm2TP dynamic prefix path warms a prefill bucket and then captures the
 * same bucket on the following chunk.  Before this regression, one participant
 * could enter HIP graph capture while its sibling was still finishing the eager
 * warmup tail, leaving a sticky capture-implicit stream dependency error that
 * later surfaced as rcclGroupEnd invalid usage.  This test locks the ownership
 * rule at the LocalTP boundary: the early participant cannot cross the boundary
 * until every sibling arrives, and the same context can reuse the barrier for
 * the next capture lifecycle boundary.
 */
TEST(Test__LocalTPRCCLGraphCapture, PrefillGraphCaptureBoundaryRendezvousBlocksUntilAllParticipantsArrive)
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

    constexpr int kTimeoutMs = 30000;
    const std::string first_boundary =
        "prefill_graph:before_begin:seq=256:bucket=256:test";
    std::atomic<bool> early_entered{false};
    std::atomic<bool> early_returned{false};
    bool early_ok = false;

    std::thread early([&]()
                      {
                          early_entered.store(true, std::memory_order_release);
                          early_ok = ctx->graphCaptureBoundaryRendezvous(
                              first_boundary,
                              0,
                              kTimeoutMs);
                          early_returned.store(true, std::memory_order_release);
                      });

    while (!early_entered.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(early_returned.load(std::memory_order_acquire))
        << "slot 0 crossed the capture boundary before slot 1 arrived";

    const bool late_ok = ctx->graphCaptureBoundaryRendezvous(
        first_boundary,
        1,
        kTimeoutMs);
    early.join();

    EXPECT_TRUE(early_ok);
    EXPECT_TRUE(late_ok);

    const std::string second_boundary =
        "prefill_graph:before_launch_after_capture:seq=256:bucket=256:test";
    std::atomic<bool> second_early_entered{false};
    std::atomic<bool> second_early_returned{false};
    bool second_early_ok = false;

    std::thread second_early([&]()
                             {
                                 second_early_entered.store(true, std::memory_order_release);
                                 second_early_ok = ctx->graphCaptureBoundaryRendezvous(
                                     second_boundary,
                                     1,
                                     kTimeoutMs);
                                 second_early_returned.store(true, std::memory_order_release);
                             });

    while (!second_early_entered.load(std::memory_order_acquire))
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_FALSE(second_early_returned.load(std::memory_order_acquire))
        << "slot 1 crossed the reused capture boundary before slot 0 arrived";

    const bool second_late_ok = ctx->graphCaptureBoundaryRendezvous(
        second_boundary,
        0,
        kTimeoutMs);
    second_early.join();

    EXPECT_TRUE(second_early_ok);
    EXPECT_TRUE(second_late_ok);
}

/**
 * @test Rooted canonical-route publication is graph-captured and byte exact.
 *
 * Every route slot has exactly one device owner, matching the production
 * apportioned-expert contract. The test reduces those independently rounded
 * slots to participant zero, invokes the production router-order ROCm reducer
 * only on that root, and broadcasts only the compact output. A separate launch
 * of the same production reducer over the complete serial slot bank supplies
 * the byte oracle. Sweeping every verifier row count through depth fifteen,
 * plus M=31, catches count-dependent collective and launch-geometry holes.
 */
TEST(Test__LocalTPRCCLGraphCapture,
     RCCLCanonicalRouteRootedPublication_GraphCaptured_MTotal_ByteExact)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found "
                     << rocm_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices{
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);

    constexpr int kTopK = 8;
    constexpr int kDModel = 257;
    constexpr int kRoot = 0;
    constexpr int kMaximumM = 31;

    std::array<std::unique_ptr<FP32Tensor>, 2> route_slots;
    std::array<std::unique_ptr<FP32Tensor>, 2> compact_outputs;
    std::array<hipStream_t, 2> streams{nullptr, nullptr};

    for (int participant = 0; participant < 2; ++participant)
    {
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        ASSERT_EQ(
            hipStreamCreateWithFlags(
                &streams[static_cast<size_t>(participant)],
                hipStreamNonBlocking),
            hipSuccess);
        route_slots[static_cast<size_t>(participant)] =
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(kMaximumM * kTopK),
                    static_cast<size_t>(kDModel)},
                DeviceId::rocm(participant));
        compact_outputs[static_cast<size_t>(participant)] =
            std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    static_cast<size_t>(kMaximumM),
                    static_cast<size_t>(kDModel)},
                DeviceId::rocm(participant));
        ASSERT_TRUE(route_slots[static_cast<size_t>(participant)]
                        ->ensureOnDevice(
                            DeviceId::rocm(participant),
                            streams[static_cast<size_t>(participant)]));
        ASSERT_TRUE(compact_outputs[static_cast<size_t>(participant)]
                        ->ensureOnDevice(
                            DeviceId::rocm(participant),
                            streams[static_cast<size_t>(participant)]));
        ASSERT_EQ(
            hipStreamSynchronize(
                streams[static_cast<size_t>(participant)]),
            hipSuccess);
    }
    auto serial_route_slots = std::make_unique<FP32Tensor>(
        std::vector<size_t>{
            static_cast<size_t>(kMaximumM * kTopK),
            static_cast<size_t>(kDModel)},
        DeviceId::rocm(kRoot));
    auto serial_output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{
            static_cast<size_t>(kMaximumM),
            static_cast<size_t>(kDModel)},
        DeviceId::rocm(kRoot));
    ASSERT_TRUE(serial_route_slots->ensureOnDevice(
        DeviceId::rocm(kRoot),
        streams[0]));
    ASSERT_TRUE(serial_output->ensureOnDevice(
        DeviceId::rocm(kRoot),
        streams[0]));
    ASSERT_EQ(hipStreamSynchronize(streams[0]), hipSuccess);

    ROCmMoEKernel root_reducer(kRoot);
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
            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            const hipStream_t stream =
                streams[static_cast<size_t>(participant)];
            FP32Tensor *const route_tensor =
                route_slots[static_cast<size_t>(participant)].get();
            std::copy(
                participant_host[static_cast<size_t>(participant)].begin(),
                participant_host[static_cast<size_t>(participant)].end(),
                route_tensor->mutable_data());
            ASSERT_TRUE(route_tensor->ensureOnDevice(
                DeviceId::rocm(participant),
                stream));
            ASSERT_EQ(
                hipMemsetAsync(
                    compact_outputs[static_cast<size_t>(participant)]
                        ->gpu_data_ptr(),
                    0xA5,
                    output_elements * sizeof(float),
                    stream),
                hipSuccess);
            ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
        }

        /* Build the serial-row oracle with the production reducer launcher. */
        ASSERT_EQ(hipSetDevice(kRoot), hipSuccess);
        std::copy(
            serial_host.begin(),
            serial_host.end(),
            serial_route_slots->mutable_data());
        ASSERT_TRUE(serial_route_slots->ensureOnDevice(
            DeviceId::rocm(kRoot),
            streams[0]));
        ASSERT_TRUE(root_reducer.reduceCanonicalRouteContributions(
            serial_route_slots.get(),
            serial_output.get(),
            m,
            kTopK,
            kDModel));
        ASSERT_EQ(hipStreamSynchronize(streams[0]), hipSuccess);

        Barrier capture_started(2);
        Barrier capture_finished(2);
        std::array<CaptureResult, 2> capture_results{};

        auto capture_participant = [&](int participant)
        {
            CaptureResult &result =
                capture_results[static_cast<size_t>(participant)];
            const hipStream_t stream =
                streams[static_cast<size_t>(participant)];
            result.begin_status = hipSetDevice(participant);
            if (result.begin_status == hipSuccess)
            {
                result.begin_status = hipStreamBeginCapture(
                    stream,
                    hipStreamCaptureModeRelaxed);
            }
            capture_started.arriveAndWait();

            if (result.begin_status == hipSuccess)
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
            if (result.begin_status == hipSuccess && result.collective_ok)
            {
                result.end_status = hipSetDevice(participant);
                if (result.end_status == hipSuccess)
                {
                    result.end_status = hipStreamEndCapture(
                        stream,
                        &result.graph);
                }
            }
            if (result.end_status == hipSuccess && result.graph)
            {
                result.instantiate_status = hipSetDevice(participant);
                if (result.instantiate_status == hipSuccess)
                {
                    result.instantiate_status = hipGraphInstantiate(
                        &result.exec,
                        result.graph,
                        nullptr,
                        nullptr,
                        0);
                }
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
            ASSERT_EQ(result.begin_status, hipSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_TRUE(result.collective_ok)
                << "M=" << m << " participant=" << participant;
            ASSERT_EQ(result.end_status, hipSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_EQ(result.instantiate_status, hipSuccess)
                << "M=" << m << " participant=" << participant;
            ASSERT_NE(result.exec, nullptr);
        }

        Barrier launch_ready(2);
        std::array<hipError_t, 2> replay_status{
            hipSuccess,
            hipSuccess};
        auto replay_participant = [&](int participant)
        {
            hipError_t &status =
                replay_status[static_cast<size_t>(participant)];
            status = hipSetDevice(participant);
            launch_ready.arriveAndWait();
            if (status == hipSuccess)
            {
                status = hipGraphLaunch(
                    capture_results[static_cast<size_t>(participant)].exec,
                    streams[static_cast<size_t>(participant)]);
            }
            if (status == hipSuccess)
            {
                status = hipStreamSynchronize(
                    streams[static_cast<size_t>(participant)]);
            }
        };
        std::thread replay0(replay_participant, 0);
        std::thread replay1(replay_participant, 1);
        replay0.join();
        replay1.join();
        ASSERT_EQ(replay_status[0], hipSuccess) << "M=" << m;
        ASSERT_EQ(replay_status[1], hipSuccess) << "M=" << m;

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
    ASSERT_EQ(hipSetDevice(kRoot), hipSuccess);
    serial_output.reset();
    serial_route_slots.reset();
    for (int participant = 0; participant < 2; ++participant)
    {
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        compact_outputs[static_cast<size_t>(participant)].reset();
        route_slots[static_cast<size_t>(participant)].reset();
        EXPECT_EQ(
            hipStreamDestroy(streams[static_cast<size_t>(participant)]),
            hipSuccess);
    }
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
        0,
        resources0.capture_stream));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        data1.data(),
        recv1->gpu_data_ptr(),
        data1.size() * sizeof(float),
        1,
        resources1.capture_stream));
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
        0,
        resources0.capture_stream));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i32_data1.data(),
        recv_i32_1,
        i32_data1.size() * sizeof(int32_t),
        1,
        resources1.capture_stream));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i8_data0.data(),
        recv_i8_0,
        i8_data0.size() * sizeof(int8_t),
        0,
        resources0.capture_stream));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        i8_data1.data(),
        recv_i8_1,
        i8_data1.size() * sizeof(int8_t),
        1,
        resources1.capture_stream));
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

/**
 * @brief Regresses large back-to-back K/V raw allgathers inside RCCL graph capture.
 *
 * Qwen phase-split prefill publishes full K and V state by capturing two
 * adjacent LocalTP raw allgathers. The payload here matches the order and size
 * class of that handoff so native graph-captured RCCL allgather must replay
 * correct bytes for both K and V, including the late rows restored by
 * prefix-cache partial hits.
 */
TEST(Test__LocalTPRCCLGraphCapture, RCCLRawAllgather_GraphCapturedLargeBackToBackKVPayloads_ReplaysCorrectly)
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
                              const float *send_k,
                              const float *send_v,
                              float *recv_k,
                              float *recv_v,
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
            result->k_allgather_ok = ctx->allgatherRawOnStream(
                send_k,
                recv_k,
                shard_count,
                CollectiveDataType::FLOAT32,
                device,
                resources->capture_stream,
                "large_kv_raw_allgather_K");
            result->v_allgather_ok = ctx->allgatherRawOnStream(
                send_v,
                recv_v,
                shard_count,
                CollectiveDataType::FLOAT32,
                device,
                resources->capture_stream,
                "large_kv_raw_allgather_V");
            result->collective_ok =
                result->k_allgather_ok &&
                result->v_allgather_ok;
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
                   send_k0,
                   send_v0,
                   recv_k0,
                   recv_v0,
                   &resources0,
                   &result0);
    std::thread t1(capture_worker,
                   1,
                   send_k1,
                   send_v1,
                   recv_k1,
                   recv_v1,
                   &resources1,
                   &result1);
    t0.join();
    t1.join();

    EXPECT_EQ(result0.begin_status, hipSuccess);
    EXPECT_EQ(result1.begin_status, hipSuccess);
    EXPECT_TRUE(result0.k_allgather_ok);
    EXPECT_TRUE(result1.k_allgather_ok);
    EXPECT_TRUE(result0.v_allgather_ok);
    EXPECT_TRUE(result1.v_allgather_ok);
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

    expectGatheredFloatShards(
        rocm_backend,
        0,
        resources0.capture_stream,
        recv_k0,
        shard_count,
        1.0f,
        2.0f,
        "large K allgather on device 0");
    expectGatheredFloatShards(
        rocm_backend,
        1,
        resources1.capture_stream,
        recv_k1,
        shard_count,
        1.0f,
        2.0f,
        "large K allgather on device 1");
    expectGatheredFloatShards(
        rocm_backend,
        0,
        resources0.capture_stream,
        recv_v0,
        shard_count,
        3.0f,
        4.0f,
        "large V allgather on device 0");
    expectGatheredFloatShards(
        rocm_backend,
        1,
        resources1.capture_stream,
        recv_v1,
        shard_count,
        3.0f,
        4.0f,
        "large V allgather on device 1");

    destroyCaptureResult(result0);
    destroyCaptureResult(result1);
    destroyAllgatherResources(rocm_backend, 0, resources0);
    destroyAllgatherResources(rocm_backend, 1, resources1);
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
 * @brief Measure the exact repeated RCCL topology used by Qwen3.6 prefill LLEP.
 *
 * A forced real-weight Qwen3.6-35B run moves one 1,900,800-byte prepared expert
 * slot in each of forty routed layers. Isolated payload kernels and one raw
 * allgather are much faster than the observed whole-graph regression, so this
 * probe separates two possible transport costs while retaining production
 * capture semantics:
 *
 *  - forty allgathers recorded directly on each participant compute stream;
 *  - forty allgathers on two rolling transfer lanes, with distinct per-layer
 *    compute-ready and transfer-done events.
 *
 * Both variants are complete participant-local HIP graphs over the production
 * LocalTPContext/RCCL API. Timing uses device events, while terminal host reads
 * validate the rank-major gathered bytes. Results are diagnostic rather than a
 * fixed hardware threshold; the model-level economy gate owns the speed target.
 */
TEST(Test__LocalTPRCCLGraphCapture,
     RCCLRawAllgather_GraphCapturedQwen36LLEPFortyLayerScalingProbe)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 2)
    {
        GTEST_SKIP() << "Requires 2+ ROCm GPUs, found "
                     << rocm_backend->deviceCount();
    }

    std::vector<GlobalDeviceAddress> devices = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1)};
    auto ctx = createLocalTPContext(
        devices,
        {},
        CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    constexpr size_t kPreparedExpertPayloadBytes = 1'900'800u;
    constexpr size_t kRoutedLayerCount = 40u;
    constexpr int kWarmupIterations = 3;
    constexpr int kMeasuredIterations = 12;

    int8_t *send0 = nullptr;
    int8_t *send1 = nullptr;
    int8_t *recv0 = nullptr;
    int8_t *recv1 = nullptr;
    allocateAndUpload<int8_t>(
        0,
        std::vector<int8_t>(kPreparedExpertPayloadBytes, 3),
        &send0);
    allocateAndUpload<int8_t>(
        1,
        std::vector<int8_t>(kPreparedExpertPayloadBytes, 5),
        &send1);
    allocateAndUpload<int8_t>(
        0,
        std::vector<int8_t>(kPreparedExpertPayloadBytes * 2u, -1),
        &recv0);
    allocateAndUpload<int8_t>(
        1,
        std::vector<int8_t>(kPreparedExpertPayloadBytes * 2u, -1),
        &recv1);

    const std::array<const void *, 2> send_buffers{send0, send1};
    const std::array<void *, 2> recv_buffers{recv0, recv1};
    LayeredAllgatherTiming direct_timing;
    LayeredAllgatherTiming auxiliary_timing;
    std::chrono::nanoseconds direct_capture_time{};
    std::chrono::nanoseconds auxiliary_capture_time{};
    std::array<size_t, 2> direct_nodes{};
    std::array<size_t, 2> auxiliary_nodes{};

    auto run_variant = [&](bool auxiliary_lanes,
                           LayeredAllgatherTiming &timing,
                           std::chrono::nanoseconds &capture_time,
                           std::array<size_t, 2> &node_counts)
    {
        LayeredAllgatherGraph graph;
        createLayeredAllgatherResources(
            rocm_backend,
            kRoutedLayerCount,
            graph);
        const auto capture_begin = std::chrono::steady_clock::now();
        captureLayeredAllgatherGraph(
            *ctx,
            send_buffers,
            recv_buffers,
            kPreparedExpertPayloadBytes,
            kRoutedLayerCount,
            auxiliary_lanes,
            graph);
        capture_time = std::chrono::steady_clock::now() - capture_begin;

        expectCapturedGraphReady(graph.results[0], "layered allgather graph 0");
        expectCapturedGraphReady(graph.results[1], "layered allgather graph 1");
        if (!::testing::Test::HasFailure())
        {
            node_counts[0] = layeredAllgatherNodeCount(graph.results[0]);
            node_counts[1] = layeredAllgatherNodeCount(graph.results[1]);
            timing = benchmarkLayeredAllgatherGraph(
                graph,
                kWarmupIterations,
                kMeasuredIterations);
        }
        destroyLayeredAllgatherResources(rocm_backend, graph);
    };

    run_variant(false, direct_timing, direct_capture_time, direct_nodes);
    if (!::testing::Test::HasFailure())
        run_variant(
            true,
            auxiliary_timing,
            auxiliary_capture_time,
            auxiliary_nodes);

    if (!::testing::Test::HasFailure())
    {
        EXPECT_GT(direct_timing.median_us, 0.0);
        EXPECT_GT(auxiliary_timing.median_us, 0.0);
        std::cout
            << "[RCCL_QWEN36_LLEP_LAYER_SCALING]"
            << " payload_bytes=" << kPreparedExpertPayloadBytes
            << " layers=" << kRoutedLayerCount
            << " direct_capture_ms="
            << std::chrono::duration<double, std::milli>(
                   direct_capture_time)
                   .count()
            << " direct_nodes=" << direct_nodes[0] << ',' << direct_nodes[1]
            << " direct_median_us=" << direct_timing.median_us
            << " direct_per_layer_us="
            << direct_timing.median_us / static_cast<double>(kRoutedLayerCount)
            << " direct_p95_us=" << direct_timing.p95_us
            << " auxiliary_capture_ms="
            << std::chrono::duration<double, std::milli>(
                   auxiliary_capture_time)
                   .count()
            << " auxiliary_nodes=" << auxiliary_nodes[0] << ',' << auxiliary_nodes[1]
            << " auxiliary_median_us=" << auxiliary_timing.median_us
            << " auxiliary_per_layer_us="
            << auxiliary_timing.median_us / static_cast<double>(kRoutedLayerCount)
            << " auxiliary_p95_us=" << auxiliary_timing.p95_us
            << std::endl;

        std::vector<int8_t> recv_host0(kPreparedExpertPayloadBytes * 2u);
        std::vector<int8_t> recv_host1(kPreparedExpertPayloadBytes * 2u);
        downloadDeviceVector(0, recv0, &recv_host0);
        downloadDeviceVector(1, recv1, &recv_host1);
        for (size_t byte = 0; byte < kPreparedExpertPayloadBytes; byte += 4096u)
        {
            EXPECT_EQ(recv_host0[byte], 3)
                << "device 0 rank-0 shard mismatch at byte " << byte;
            EXPECT_EQ(recv_host0[kPreparedExpertPayloadBytes + byte], 5)
                << "device 0 rank-1 shard mismatch at byte " << byte;
            EXPECT_EQ(recv_host1[byte], 3)
                << "device 1 rank-0 shard mismatch at byte " << byte;
            EXPECT_EQ(recv_host1[kPreparedExpertPayloadBytes + byte], 5)
                << "device 1 rank-1 shard mismatch at byte " << byte;
        }
    }

    freeDevicePtr(0, send0);
    freeDevicePtr(1, send1);
    freeDevicePtr(0, recv0);
    freeDevicePtr(1, recv1);
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
        0,
        aux_graph.capture_stream0));
    ASSERT_TRUE(rocm_backend->deviceToHost(
        recv_host1.data(),
        recv1,
        recv_host1.size() * sizeof(int8_t),
        1,
        aux_graph.capture_stream1));
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

/**
 * @test A captured RCCL byte gather followed by an ascending-rank device fold
 *       publishes identical row-zero bytes for M=1, M=2, and M=3.
 *
 * Four participants make the reduction-order defect observable: the rank-zero
 * row uses cancellation-sensitive values for which a balanced native tree and
 * the required ascending-rank fold produce different answers. The protocol is
 * captured and replayed as one retained graph per participant, matching the
 * production MTP transaction boundary without any host arithmetic authority.
 */
TEST(Test__LocalTPRCCLGraphCapture,
     RCCLCanonicalRankOrderReduction_GraphCapturedRowZeroIsBatchInvariant)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 4)
    {
        GTEST_SKIP() << "Requires 4+ ROCm GPUs, found "
                     << rocm_backend->deviceCount();
    }

    constexpr int kParticipants = 4;
    constexpr size_t kColumns = 3072;
    auto ctx = createLocalTPContext(
        {GlobalDeviceAddress::rocm(0),
         GlobalDeviceAddress::rocm(1),
         GlobalDeviceAddress::rocm(2),
         GlobalDeviceAddress::rocm(3)},
        {},
        CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    const auto run_shape = [&](int rows, std::vector<float> *row_zero)
    {
        ASSERT_GT(rows, 0);
        ASSERT_NE(row_zero, nullptr);
        const size_t element_count = static_cast<size_t>(rows) * kColumns;
        std::array<float *, kParticipants> values{};
        std::array<float *, kParticipants> rank_banks{};
        std::array<hipStream_t, kParticipants> streams{};
        std::array<CaptureResult, kParticipants> results{};

        constexpr std::array<float, kParticipants> kAdversarialRow{
            16777216.0F,
            1.0F,
            -16777216.0F,
            0.5F};
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            std::vector<float> host_values(element_count);
            for (int row = 0; row < rows; ++row)
            {
                for (size_t column = 0; column < kColumns; ++column)
                {
                    host_values[static_cast<size_t>(row) * kColumns + column] =
                        row == 0
                            ? kAdversarialRow[participant]
                            : static_cast<float>(participant + 1) * 0.25F +
                                  static_cast<float>(column % 11u) * 0.03125F;
                }
            }
            allocateAndUpload<float>(
                participant, host_values, &values[participant]);
            allocateAndUpload<float>(
                participant,
                std::vector<float>(
                    element_count * static_cast<size_t>(kParticipants),
                    -1.0F),
                &rank_banks[participant]);
            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            ASSERT_EQ(
                hipStreamCreateWithFlags(
                    &streams[participant], hipStreamNonBlocking),
                hipSuccess);
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
                    result.begin_status = hipSetDevice(participant);
                    if (result.begin_status == hipSuccess)
                    {
                        result.begin_status = hipStreamBeginCapture(
                            streams[participant],
                            hipStreamCaptureModeRelaxed);
                    }
                    ready_to_capture.arriveAndWait();

                    if (result.begin_status == hipSuccess)
                    {
                        GraphCaptureGuard guard;
                        const bool gathered = ctx->allgatherRawOnStream(
                            values[participant],
                            rank_banks[participant],
                            element_count,
                            CollectiveDataType::FLOAT32,
                            participant,
                            streams[participant],
                            "rccl_canonical_rank_order_m" +
                                std::to_string(rows));
                        const bool folded = gathered &&
                            launchROCmTPRankOrderedSumFP32(
                                rank_banks[participant],
                                values[participant],
                                element_count,
                                kParticipants,
                                participant,
                                streams[participant]);
                        result.collective_ok = gathered && folded;
                    }
                    captured_protocol.arriveAndWait();

                    if (result.begin_status == hipSuccess &&
                        result.collective_ok)
                    {
                        result.end_status = hipSetDevice(participant);
                        if (result.end_status == hipSuccess)
                        {
                            result.end_status = hipStreamEndCapture(
                                streams[participant], &result.graph);
                        }
                    }
                    if (result.end_status == hipSuccess && result.graph)
                    {
                        result.instantiate_status = hipGraphInstantiate(
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
            ASSERT_EQ(result.begin_status, hipSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_TRUE(result.collective_ok)
                << "participant=" << participant << " M=" << rows;
            ASSERT_EQ(result.end_status, hipSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_EQ(result.instantiate_status, hipSuccess)
                << "participant=" << participant << " M=" << rows;
            ASSERT_NE(result.exec, nullptr);
        }

        Barrier ready_to_replay(kParticipants);
        std::array<hipError_t, kParticipants> replay_status{};
        std::array<std::thread, kParticipants> replay_threads;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            replay_threads[participant] = std::thread(
                [&, participant]
                {
                    replay_status[participant] = hipSetDevice(participant);
                    ready_to_replay.arriveAndWait();
                    if (replay_status[participant] == hipSuccess)
                    {
                        replay_status[participant] = hipGraphLaunch(
                            results[participant].exec,
                            streams[participant]);
                    }
                    if (replay_status[participant] == hipSuccess)
                    {
                        replay_status[participant] = hipStreamSynchronize(
                            streams[participant]);
                    }
                });
        }
        for (auto &thread : replay_threads)
            thread.join();

        std::array<std::vector<float>, kParticipants> actual;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            ASSERT_EQ(replay_status[participant], hipSuccess)
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
        EXPECT_TRUE(std::all_of(
            actual[0].begin(),
            actual[0].begin() + static_cast<std::ptrdiff_t>(kColumns),
            [](float value) { return value == 0.5F; }))
            << "Ascending rank order must be the sole arithmetic authority";

        row_zero->assign(actual[0].begin(), actual[0].begin() + kColumns);
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            destroyCaptureResult(results[participant]);
            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            EXPECT_EQ(hipStreamDestroy(streams[participant]), hipSuccess);
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
 * @test A mapped linear activation copy remains executable after retained-parent import.
 *
 * HIP stream capture represents `hipMemcpyAsync` as a specialized 1D graph
 * node whose runtime classification accounts for mapped and peer memory. The
 * retained-parent importer must preserve that representation. Recreating the
 * node through HIP's generic 3D API makes ROCm 7.1 on gfx906 advertise direct
 * packet capture while producing no packet; graph instantiation and launch
 * then report success even though the complete parent executes no work.
 *
 * The 12,288-byte payload is the Qwen 3.5 122B decode activation geometry that
 * exposed the defect. Verification synchronizes only at the terminal test
 * observation boundary, after the retained graph has been launched normally.
 */
TEST(Test__LocalTPRCCLGraphCapture,
     MappedLinearMemcpy_RetainedParentImportExecutes)
{
    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 1)
        GTEST_SKIP() << "Requires one ROCm GPU";

    constexpr int kDeviceOrdinal = 0;
    constexpr std::size_t kActivationBytes = 12'288u;
    const DeviceId device = DeviceId::rocm(kDeviceOrdinal);
    ASSERT_EQ(hipSetDevice(kDeviceOrdinal), hipSuccess);

    hipStream_t stream = nullptr;
    ASSERT_EQ(
        hipStreamCreateWithFlags(&stream, hipStreamNonBlocking),
        hipSuccess);

    const std::array<DeviceId, 1> mapped_devices{device};
    auto source_region =
        TransferEngine::instance().allocateMappedHostRegion(
            kActivationBytes, mapped_devices);
    ASSERT_NE(source_region, nullptr);
    ASSERT_TRUE(source_region->isBound());
    auto *const source = static_cast<std::uint8_t *>(
        source_region->mutableHostData());
    for (std::size_t index = 0u; index < kActivationBytes; ++index)
        source[index] = static_cast<std::uint8_t>((index * 17u + 29u) & 0xffu);

    std::uint8_t *destination = nullptr;
    ASSERT_EQ(
        hipMalloc(
            reinterpret_cast<void **>(&destination),
            kActivationBytes),
        hipSuccess);
    ASSERT_NE(destination, nullptr);
    ASSERT_EQ(hipMemset(destination, 0, kActivationBytes), hipSuccess);

    auto child = std::make_unique<HIPGraphCapture>(
        stream, kDeviceOrdinal);
    ASSERT_TRUE(child->beginCapture());
    {
        GraphCaptureGuard guard;
        ASSERT_EQ(
            hipMemcpyAsync(
                destination,
                source_region->deviceAlias(device),
                kActivationBytes,
                hipMemcpyDeviceToDevice,
                stream),
            hipSuccess);
    }
    ASSERT_TRUE(child->endCapture());
    ASSERT_EQ(child->nodeCount(), 1u);

    auto parent = std::make_unique<HIPGraphCapture>(
        stream, kDeviceOrdinal);
    const std::array<GPUOrderedTimelineStep, 1> ordered_steps{{
        {
            .name = "mapped_activation_copy",
            .kind = GPUOrderedTimelineStepKind::CapturedFragment,
            .capture = child.get(),
            .signal = nullptr,
            .value = 0u,
        },
    }};
    ASSERT_TRUE(parent->buildOrderedTimelineTransaction(ordered_steps));
    ASSERT_EQ(parent->nodeCount(), 1u);
    ASSERT_TRUE(parent->instantiate());
    ASSERT_TRUE(parent->launch());

    /* This synchronization is the test's terminal observation boundary, not
     * part of the retained production transaction. */
    ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);
    std::vector<std::uint8_t> observed(kActivationBytes, 0u);
    ASSERT_EQ(
        hipMemcpy(
            observed.data(),
            destination,
            kActivationBytes,
            hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_TRUE(std::equal(observed.begin(), observed.end(), source));

    parent.reset();
    child.reset();
    EXPECT_EQ(hipFree(destination), hipSuccess);
    EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
}

/**
 * @test A production-shaped captured epoch transaction remains launchable
 *       after native parent composition on four ROCm participants.
 *
 * Qwen 3.5 122B decode records an embedding/dense rank-order reduction, an
 * attention rank-order reduction, a mapped CPU ExpertOverlay ticket, and a
 * rooted route reduce/broadcast in one child before flattening 49 children
 * into its retained endpoint parent. A direct child launch does not exercise
 * RCCL's persistent graph-plan lifetime after node import, while a single
 * collective or tiny payload can select a different RCCL protocol. This
 * focused regression therefore preserves the production order, decode payload
 * classes, 49 layer fragments, and four independently instantiated retained
 * parent families. The immutable child fragments are recorded once and
 * imported into every parent, matching production's graph-cache ownership.
 * Parent validation uses exact source-to-parent node conservation rather than
 * duplicating a model graph's transient node census: real-weight parity owns
 * the complete live graph-shape proof, while this model-free test owns the
 * RCCL graph-plan, import, epoch, and ticket lifetime proof.
 * The epoch acquire is the first node of fragment zero and its release is the
 * terminal of fragment 48, proving that the ordinary main graph owns one
 * self-contained residency transaction. Participant zero's mapped ticket is
 * serviced concurrently by the CPU exactly between the dense and rooted
 * collective families.
 */
TEST(Test__LocalTPRCCLGraphCapture,
     RCCLCanonicalRankOrderReduction_TP4CapturedEpochTransactionParentCompletes)
{
    const auto test_started_at = std::chrono::steady_clock::now();
    const auto report_phase = [&](const char *phase,
                                  int participant = -1,
                                  int ordinal = -1)
    {
        const auto elapsed_ms = std::chrono::duration_cast<
            std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - test_started_at)
                                    .count();
        std::cerr << "[RCCL retained-parent regression] phase=" << phase;
        if (participant >= 0)
            std::cerr << " participant=" << participant;
        if (ordinal >= 0)
            std::cerr << " ordinal=" << ordinal;
        std::cerr << " elapsed_ms=" << elapsed_ms << std::endl;
    };

    auto *rocm_backend = getROCmBackend();
    ASSERT_NE(rocm_backend, nullptr);
    if (rocm_backend->deviceCount() < 4)
    {
        GTEST_SKIP() << "Requires 4+ ROCm GPUs, found "
                     << rocm_backend->deviceCount();
    }

    constexpr int kParticipants = 4;
    constexpr int kFragments = 49;
    constexpr int kRetainedFamilies = 4;
    constexpr std::size_t kDenseElements = 3072u;
    constexpr std::size_t kRouteElements = 8u * kDenseElements;
    constexpr std::size_t kProductionMemcpyBytes = 993280u;
    auto ctx = createLocalTPContext(
        {GlobalDeviceAddress::rocm(0),
         GlobalDeviceAddress::rocm(1),
         GlobalDeviceAddress::rocm(2),
         GlobalDeviceAddress::rocm(3)},
        {},
        CollectiveBackendType::RCCL);
    ASSERT_NE(ctx, nullptr);
    ASSERT_TRUE(ctx->supportsRawAllgatherOnStreamGraphCapture());

    constexpr std::size_t kTicketStride = 2u * sizeof(std::uint64_t);
    const std::array<DeviceId, 1> ticket_devices{DeviceId::rocm(0)};
    constexpr std::size_t kEntryTicketOffset =
        kFragments * kTicketStride;
    auto ticket_region =
        TransferEngine::instance().allocateMappedHostRegion(
            (kFragments + 1u) * kTicketStride, ticket_devices);
    ASSERT_NE(ticket_region, nullptr);
    ASSERT_TRUE(ticket_region->isBound());

    /*
     * The production continuation GPUs do not acquire their placement banks
     * independently.  They first raise disjoint guards in one mapped barrier,
     * then participant zero freezes the exact topology-wide admission epoch.
     * Keep the scalar and record on separate cache-line boundaries so the test
     * exercises the same system-scope memory protocol without false sharing.
     */
    constexpr std::size_t kAdmissionEpochOffset = 0u;
    constexpr std::size_t kAdmissionBarrierOffset = 64u;
    const std::array<DeviceId, kParticipants> admission_devices{
        DeviceId::rocm(0),
        DeviceId::rocm(1),
        DeviceId::rocm(2),
        DeviceId::rocm(3)};
    auto admission_region =
        TransferEngine::instance().allocateMappedHostRegion(
            kAdmissionBarrierOffset +
                sizeof(MoEOverlayDeviceControllerInferenceEpochRecord),
            admission_devices);
    ASSERT_NE(admission_region, nullptr);
    ASSERT_TRUE(admission_region->isBound());
    *static_cast<std::uint64_t *>(admission_region->mutableHostData(
        kAdmissionEpochOffset)) = 1u;
    auto *const admission_record = static_cast<
        MoEOverlayDeviceControllerInferenceEpochRecord *>(
        admission_region->mutableHostData(kAdmissionBarrierOffset));
    *admission_record = MoEOverlayDeviceControllerInferenceEpochRecord{};
    admission_record->participant_mask = 0b1111u;
    admission_record->publisher_participant_id = 0u;
    admission_record->topology_fingerprint = 1u;
    admission_record->epoch = 1u;

    std::array<float *, kParticipants> values{};
    std::array<float *, kParticipants> rank_banks{};
    std::array<float *, kParticipants> route_values{};
    std::array<float *, kParticipants> compact_values{};
    std::array<std::uint8_t *, kParticipants> copy_sources{};
    std::array<std::uint8_t *, kParticipants> copy_destinations{};
    std::array<std::unique_ptr<FP32Tensor>, kParticipants> consensus_values{};
    std::array<
        std::shared_ptr<DeviceMoEOverlayEpochArena>,
        kParticipants>
        epoch_arenas;
    std::array<std::unique_ptr<ROCmMoEKernel>, kParticipants> epoch_kernels;
    std::array<hipStream_t, kParticipants> streams{};
    std::array<
        std::vector<std::unique_ptr<HIPGraphCapture>>,
        kParticipants>
        children;
    std::array<std::vector<void *>, kParticipants> residency_pressure;
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        report_phase("participant_setup_begin", participant);
        allocateAndUpload<float>(
            participant,
            std::vector<float>(
                kDenseElements,
                static_cast<float>(participant + 1)),
            &values[participant]);
        allocateAndUpload<float>(
            participant,
            std::vector<float>(
                kDenseElements * kParticipants, -1.0F),
            &rank_banks[participant]);
        allocateAndUpload<float>(
            participant,
            std::vector<float>(
                kRouteElements,
                static_cast<float>(participant + 1)),
            &route_values[participant]);
        allocateAndUpload<float>(
            participant,
            std::vector<float>(kDenseElements, -1.0F),
            &compact_values[participant]);
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        ASSERT_EQ(
            hipMalloc(
                reinterpret_cast<void **>(&copy_sources[participant]),
                kProductionMemcpyBytes),
            hipSuccess);
        ASSERT_EQ(
            hipMalloc(
                reinterpret_cast<void **>(&copy_destinations[participant]),
                kProductionMemcpyBytes),
            hipSuccess);
        ASSERT_NE(copy_sources[participant], nullptr);
        ASSERT_NE(copy_destinations[participant], nullptr);
        ASSERT_EQ(
            hipMemset(copy_sources[participant], 0x5a, kProductionMemcpyBytes),
            hipSuccess);
        ASSERT_EQ(
            hipMemset(
                copy_destinations[participant],
                0xa5,
                kProductionMemcpyBytes),
            hipSuccess);
        consensus_values[participant] = TestTensorFactory::createFP32({1u});
        TestTensorFactory::fillValue(
            consensus_values[participant].get(),
            static_cast<float>(participant + 1));
        ASSERT_TRUE(consensus_values[participant]->ensureOnDevice(
            DeviceId::rocm(participant)));
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        ASSERT_EQ(
            hipStreamCreateWithFlags(
                &streams[participant], hipStreamNonBlocking),
            hipSuccess);
        epoch_arenas[participant] =
            std::make_shared<DeviceMoEOverlayEpochArena>(
                DeviceMoEOverlayEpochArena::Config{
                    .device_id = DeviceId::rocm(participant),
                    .initial_epoch = 1u,
                    .initial_bank = 0u,
                    .request_slot_capacity = 1u,
                    .external_admission_epoch = static_cast<
                        const std::uint64_t *>(
                        admission_region->deviceAlias(
                            DeviceId::rocm(participant),
                            kAdmissionEpochOffset)),
                    .external_admission_lifetime = admission_region,
                    .admission_barrier = {
                        .record = static_cast<
                            MoEOverlayDeviceControllerInferenceEpochRecord *>(
                            admission_region->deviceAlias(
                                DeviceId::rocm(participant),
                                kAdmissionBarrierOffset)),
                        .participant_id = static_cast<std::uint32_t>(
                            participant),
                    },
                });
        epoch_kernels[participant] =
            std::make_unique<ROCmMoEKernel>(participant);
        children[participant].reserve(kFragments);
        report_phase("participant_setup_complete", participant);
    }

    report_phase("child_capture_begin");
    Barrier capture_begin(kParticipants);
    Barrier collective_recorded(kParticipants);
    std::array<bool, kParticipants> capture_ok{};
    std::array<std::thread, kParticipants> capture_threads;
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        capture_threads[participant] = std::thread(
            [&, participant]
            {
                capture_ok[participant] =
                    hipSetDevice(participant) == hipSuccess;
                for (int fragment = 0;
                     fragment < kFragments && capture_ok[participant];
                     ++fragment)
                {
                    const auto fragment_begin =
                        std::chrono::steady_clock::now();
                    auto capture_begin_complete = fragment_begin;
                    auto memcpy_capture_complete = fragment_begin;
                    auto attention_capture_complete = fragment_begin;
                    auto collective_capture_complete = fragment_begin;
                    auto child = std::make_unique<HIPGraphCapture>(
                        streams[participant], participant);
                    capture_ok[participant] = child->beginCapture();
                    capture_begin_complete =
                        std::chrono::steady_clock::now();
                    capture_begin.arriveAndWait();
                    if (capture_ok[participant])
                    {
                        GraphCaptureGuard guard;
                        const MoEKernelLaunchContext epoch_launch{
                            .stream = streams[participant],
                            .workspace = nullptr,
                        };
                        /* Production's continuation authority publishes its
                         * mapped ticket-service entry as the parent root.  The
                         * sibling parents begin at acquireEpochKernel.  Keep
                         * this before participant zero's acquire so the native
                         * root topology, not just eventual behavior, agrees. */
                        if (participant == 0 && fragment == 0)
                        {
                            TransferEngine::instance().
                                enqueueMappedTimelinePublish64(
                                    *ticket_region,
                                    kEntryTicketOffset,
                                    1u,
                                    DeviceId::rocm(0),
                                    streams[participant]);
                        }
                        if (fragment == 0)
                        {
                            capture_ok[participant] =
                                epoch_kernels[participant]
                                    ->acquireMoEOverlayEpoch(
                                        epoch_launch,
                                        epoch_arenas[participant]->control(),
                                        epoch_arenas[participant]
                                            ->requestTicket(0u),
                                        epoch_arenas[participant]
                                            ->requestStatus(0u),
                                        epoch_arenas[participant]
                                            ->externalAdmissionEpoch(),
                                        epoch_arenas[participant]
                                            ->admissionBarrier());
                        }
                        /* Retain one production-maximum D2D payload in every
                         * layer fragment. This selects the same HIP copy-node
                         * class whose native import previously lost graph-plan
                         * lifetime, without repeating thousands of identical
                         * nodes from a stale model-specific census. */
                        capture_ok[participant] = hipMemcpyAsync(
                            copy_destinations[participant],
                            copy_sources[participant],
                            kProductionMemcpyBytes,
                            hipMemcpyDeviceToDevice,
                            streams[participant]) == hipSuccess;
                        memcpy_capture_complete =
                            std::chrono::steady_clock::now();
                        const bool attention_gathered =
                            capture_ok[participant] &&
                            ctx->allgatherRawOnStream(
                                values[participant],
                                rank_banks[participant],
                                kDenseElements,
                                CollectiveDataType::FLOAT32,
                                participant,
                                streams[participant],
                                "rccl_retained_tp4_attention_fragment_" +
                                    std::to_string(fragment));
                        const bool attention_folded = attention_gathered &&
                            launchROCmTPRankOrderedSumFP32(
                                rank_banks[participant],
                                values[participant],
                                kDenseElements,
                                kParticipants,
                                participant,
                                streams[participant]);
                        attention_capture_complete =
                            std::chrono::steady_clock::now();
                        const bool admission_consensus =
                            attention_folded &&
                            ctx->allreduceOnStream(
                                consensus_values[participant].get(),
                                "rccl_retained_tp4_admission_fragment_" +
                                    std::to_string(fragment),
                                1u,
                                streams[participant],
                                "fp32");
                        const bool completion_consensus =
                            admission_consensus &&
                            ctx->allreduceOnStream(
                                consensus_values[participant].get(),
                                "rccl_retained_tp4_completion_fragment_" +
                                    std::to_string(fragment),
                                1u,
                                streams[participant],
                                "fp32");

                        if (completion_consensus && participant == 0)
                        {
                            const std::uint64_t timeline =
                                static_cast<std::uint64_t>(fragment + 1);
                            TransferEngine::instance().
                                enqueueMappedTimelinePublish64(
                                    *ticket_region,
                                    static_cast<std::size_t>(fragment) *
                                        kTicketStride,
                                    timeline,
                                    DeviceId::rocm(0),
                                    streams[participant]);
                            TransferEngine::instance().
                                enqueueMappedTimelineWait64(
                                    *ticket_region,
                                    static_cast<std::size_t>(fragment) *
                                            kTicketStride +
                                        sizeof(std::uint64_t),
                                    timeline,
                                    DeviceId::rocm(0),
                                    streams[participant]);
                        }

                        const bool routes_reduced = completion_consensus &&
                            ctx->reduceRawOnStream(
                                route_values[participant],
                                route_values[participant],
                                kRouteElements,
                                CollectiveDataType::FLOAT32,
                                CollectiveOp::ALLREDUCE_SUM,
                                /*root_device_index=*/0,
                                participant,
                                streams[participant],
                                "rccl_retained_tp4_routes_reduce_fragment_" +
                                    std::to_string(fragment));
                        const bool routes_broadcast = routes_reduced &&
                            ctx->broadcastRawOnStream(
                                compact_values[participant],
                                compact_values[participant],
                                kDenseElements,
                                CollectiveDataType::FLOAT32,
                                /*root_device_index=*/0,
                                participant,
                                streams[participant],
                                "rccl_retained_tp4_routes_broadcast_fragment_" +
                                    std::to_string(fragment));
                        capture_ok[participant] = routes_broadcast;
                        if (capture_ok[participant] &&
                            fragment == kFragments - 1)
                        {
                            capture_ok[participant] =
                                epoch_kernels[participant]
                                    ->releaseMoEOverlayEpoch(
                                        epoch_launch,
                                        epoch_arenas[participant]->control(),
                                        epoch_arenas[participant]
                                            ->requestTicket(0u),
                                        epoch_arenas[participant]
                                            ->requestStatus(0u));
                        }
                        collective_capture_complete =
                            std::chrono::steady_clock::now();
                    }
                    collective_recorded.arriveAndWait();
                    if (capture_ok[participant])
                        capture_ok[participant] = child->endCapture();
                    const auto fragment_complete =
                        std::chrono::steady_clock::now();
                    const auto fragment_ms = std::chrono::duration_cast<
                        std::chrono::milliseconds>(
                        fragment_complete - fragment_begin)
                                                 .count();
                    if (participant == 0 && fragment_ms >= 500)
                    {
                        const auto phase_ms = [](auto begin, auto end)
                        {
                            return std::chrono::duration_cast<
                                std::chrono::milliseconds>(end - begin)
                                .count();
                        };
                        std::cerr
                            << "[RCCL retained-parent regression]"
                            << " phase=slow_child_capture"
                            << " participant=" << participant
                            << " ordinal=" << fragment
                            << " total_ms=" << fragment_ms
                            << " begin_ms="
                            << phase_ms(
                                   fragment_begin,
                                   capture_begin_complete)
                            << " memcpy_ms="
                            << phase_ms(
                                   capture_begin_complete,
                                   memcpy_capture_complete)
                            << " attention_ms="
                            << phase_ms(
                                   memcpy_capture_complete,
                                   attention_capture_complete)
                            << " collective_ms="
                            << phase_ms(
                                   attention_capture_complete,
                                   collective_capture_complete)
                            << " end_ms="
                            << phase_ms(
                                   collective_capture_complete,
                                   fragment_complete)
                            << std::endl;
                    }
                    if (capture_ok[participant])
                        children[participant].push_back(std::move(child));
                    if (participant == 0 &&
                        (fragment == 0 || fragment == 12 ||
                         fragment == 24 || fragment == 36 ||
                         fragment == kFragments - 1))
                    {
                        report_phase(
                            "child_capture_checkpoint",
                            participant,
                            fragment);
                    }
                }
            });
    }
    for (auto &thread : capture_threads)
        thread.join();
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        ASSERT_TRUE(capture_ok[participant])
            << "participant=" << participant;
        ASSERT_EQ(children[participant].size(), kFragments)
            << "participant=" << participant;
    }
    std::array<std::size_t, kParticipants> expected_parent_nodes{};
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        for (const auto &child : children[participant])
            expected_parent_nodes[participant] += child->nodeCount();
        ASSERT_GT(expected_parent_nodes[participant], 0u)
            << "participant=" << participant;
    }
    report_phase("child_capture_complete");

    /*
     * Apply the model-residency envelope at the operation whose behavior it is
     * intended to certify: native parent composition, AQL packet capture,
     * executable instantiation, and replay. Artificially consuming nearly all
     * VRAM before recording 49 otherwise identical RCCL children forces the
     * runtime to repeat low-memory registration work for every distinct source
     * graph. Real-model parity owns capture-time weight/arena residency and
     * measures that path directly; multiplying it into this model-free
     * lifecycle proof both duplicates that authority and obscures the packet
     * path under test. These setup-only allocations remain live through every
     * parent replay and are released only after all executables retire.
     */
    constexpr std::size_t kTargetFreeBytes =
        3ull * 1024ull * 1024ull * 1024ull;
    constexpr std::size_t kPressureChunkBytes =
        512ull * 1024ull * 1024ull;
    report_phase("parent_residency_pressure_begin");
    for (int participant = 0; participant < kParticipants; ++participant)
    {
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        for (;;)
        {
            std::size_t free_bytes = 0u;
            std::size_t total_bytes = 0u;
            ASSERT_EQ(
                hipMemGetInfo(&free_bytes, &total_bytes), hipSuccess)
                << "participant=" << participant;
            if (free_bytes <= kTargetFreeBytes + kPressureChunkBytes)
                break;
            const std::size_t allocation_bytes = std::min(
                kPressureChunkBytes,
                free_bytes - kTargetFreeBytes);
            void *allocation = nullptr;
            ASSERT_EQ(
                hipMalloc(&allocation, allocation_bytes), hipSuccess)
                << "participant=" << participant
                << " free_bytes=" << free_bytes
                << " allocation_bytes=" << allocation_bytes;
            ASSERT_NE(allocation, nullptr);
            residency_pressure[participant].push_back(allocation);
        }
    }
    report_phase("parent_residency_pressure_complete");

    std::array<
        std::array<std::unique_ptr<HIPGraphCapture>, kParticipants>,
        kRetainedFamilies>
        parents;
    for (int family = 0; family < kRetainedFamilies; ++family)
    {
        report_phase("parent_family_begin", -1, family);
        /*
         * RankOrchestrator owns one persistent setup worker per LocalTP
         * participant. Each worker composes and instantiates the same serving
         * family concurrently on its exact device; serial parent construction
         * here would omit the production runtime boundary that this regression
         * exists to certify. The barrier removes incidental composition skew
         * and makes hipGraphInstantiate concurrency explicit and repeatable.
         */
        Barrier ready_to_instantiate(kParticipants);
        std::array<bool, kParticipants> build_ok{};
        std::array<bool, kParticipants> instantiate_ok{};
        std::array<std::thread, kParticipants> parent_threads;
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            parent_threads[participant] = std::thread(
                [&, family, participant]
                {
                    build_ok[participant] =
                        hipSetDevice(participant) == hipSuccess;
                    if (build_ok[participant])
                    {
                        parents[family][participant] =
                            std::make_unique<HIPGraphCapture>(
                                streams[participant], participant);
                        std::vector<std::string> names;
                        std::vector<GPUOrderedTimelineStep> steps;
                        names.reserve(kFragments);
                        steps.reserve(kFragments);
                        for (int fragment = 0; fragment < kFragments; ++fragment)
                        {
                            names.push_back(
                                "canonical_rank_order_family_" +
                                std::to_string(family) + "_fragment_" +
                                std::to_string(fragment));
                            steps.push_back({
                                .name = names.back().c_str(),
                                .kind = GPUOrderedTimelineStepKind::CapturedFragment,
                                .capture =
                                    children[participant][fragment].get(),
                                .signal = nullptr,
                                .value = 0u,
                            });
                        }
                        build_ok[participant] =
                            parents[family][participant]
                                ->buildOrderedTimelineTransaction(steps);
                    }

                    ready_to_instantiate.arriveAndWait();
                    instantiate_ok[participant] =
                        build_ok[participant] &&
                        parents[family][participant]->instantiate();
                });
        }
        for (auto &thread : parent_threads)
            thread.join();

        for (int participant = 0; participant < kParticipants; ++participant)
        {
            ASSERT_TRUE(build_ok[participant])
                << "family=" << family << " participant=" << participant;
            ASSERT_TRUE(instantiate_ok[participant])
                << "family=" << family << " participant=" << participant;
            ASSERT_EQ(
                parents[family][participant]->nodeCount(),
                expected_parent_nodes[participant])
                << "retained parent lost one or more immutable child nodes, "
                   "family="
                << family << " participant=" << participant;
        }
        report_phase("parent_family_complete", -1, family);
    }

    for (int participant = 0; participant < kParticipants; ++participant)
    {
        for (int family = 0; family < kRetainedFamilies; ++family)
        {
            ASSERT_TRUE(parents[family][participant]->hasExecutable())
                << "family=" << family
                << " participant=" << participant;
            if (family > 0)
            {
                ASSERT_NE(
                    parents[family - 1][participant]->graph(),
                    parents[family][participant]->graph())
                    << "retained families must own distinct parent graphs, "
                       "family="
                    << family << " participant=" << participant;
            }
        }
    }

    const auto replay_family = [&](int family)
    {
        report_phase("replay_family_begin", -1, family);
        ASSERT_GE(family, 0);
        ASSERT_LT(family, kRetainedFamilies);
        std::memset(
            ticket_region->mutableHostData(),
            0,
            (kFragments + 1u) * kTicketStride);

        Barrier replay_begin(kParticipants);
        std::array<bool, kParticipants> replay_ok{};
        std::array<std::thread, kParticipants> replay_threads;
        std::atomic<bool> ticket_service_ok{true};
        std::atomic<bool> entry_ticket_observed{false};
        std::thread ticket_service(
            [&]
            {
                auto *const entry = static_cast<std::uint64_t *>(
                    ticket_region->mutableHostData(kEntryTicketOffset));
                const auto entry_deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(30);
                while (std::atomic_ref<std::uint64_t>(*entry).load(
                           std::memory_order_acquire) < 1u)
                {
                    if (std::chrono::steady_clock::now() >= entry_deadline)
                    {
                        ticket_service_ok.store(
                            false, std::memory_order_release);
                        return;
                    }
                    std::this_thread::yield();
                }
                entry_ticket_observed.store(true, std::memory_order_release);
                for (int fragment = 0; fragment < kFragments; ++fragment)
                {
                    const std::uint64_t timeline =
                        static_cast<std::uint64_t>(fragment + 1);
                    auto *const dispatch = static_cast<std::uint64_t *>(
                        ticket_region->mutableHostData(
                            static_cast<std::size_t>(fragment) *
                            kTicketStride));
                    auto *const returned = static_cast<std::uint64_t *>(
                        ticket_region->mutableHostData(
                            static_cast<std::size_t>(fragment) *
                                kTicketStride +
                            sizeof(std::uint64_t)));
                    const auto deadline =
                        std::chrono::steady_clock::now() +
                        std::chrono::seconds(30);
                    while (std::atomic_ref<std::uint64_t>(*dispatch).load(
                               std::memory_order_acquire) < timeline)
                    {
                        if (std::chrono::steady_clock::now() >= deadline)
                        {
                            ticket_service_ok.store(
                                false, std::memory_order_release);
                            return;
                        }
                        std::this_thread::yield();
                    }
                    std::atomic_ref<std::uint64_t>(*returned).store(
                        timeline, std::memory_order_release);
                }
            });
        for (int participant = 0; participant < kParticipants; ++participant)
        {
            replay_threads[participant] = std::thread(
                [&, participant]
                {
                    replay_ok[participant] =
                        hipSetDevice(participant) == hipSuccess;
                    replay_begin.arriveAndWait();
                    if (family == kRetainedFamilies - 1)
                    {
                        /* Match the live LocalTP worker skew observed on the
                         * 122B retained decode parent: the highest participant
                         * submits first, followed by authority zero, then the
                         * remaining peers. RCCL graph replay must not require
                         * an accidental host-side simultaneous launch. */
                        /* Cell 73's retained decode workers submitted
                         * participant 2 first, then 3, then 1, with the
                         * continuation authority (participant 0) last.  The
                         * earlier fixture accidentally exercised the opposite
                         * authority-first ordering and therefore could not
                         * expose a packet-submission queue that stalls while
                         * the leading RCCL participants await the authority. */
                        constexpr std::array<int, kParticipants>
                            kModelLikeLaunchDelayMs{30, 15, 0, 5};
                        std::this_thread::sleep_for(
                            std::chrono::milliseconds(
                                kModelLikeLaunchDelayMs[participant]));
                    }
                    if (replay_ok[participant])
                    {
                        replay_ok[participant] =
                            parents[family][participant]->launch();
                    }
                });
        }
        for (auto &thread : replay_threads)
            thread.join();

        /* Production decode does not synchronize after graph submission: the
         * mapped ticket service must observe native progress while every
         * capture stream remains asynchronous. A test that synchronizes here
         * can accidentally make host-driven HIP submission progress look like
         * a valid device-owned transaction. */
        const auto asynchronous_entry_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!entry_ticket_observed.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < asynchronous_entry_deadline)
        {
            std::this_thread::yield();
        }
        EXPECT_TRUE(entry_ticket_observed.load(std::memory_order_acquire))
            << "retained parent made no mapped-ticket progress before host "
               "stream synchronization, family="
            << family;

        for (int participant = 0; participant < kParticipants; ++participant)
        {
            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            if (replay_ok[participant])
            {
                replay_ok[participant] =
                    hipStreamSynchronize(streams[participant]) == hipSuccess;
            }
        }
        ticket_service.join();
        EXPECT_TRUE(ticket_service_ok.load(std::memory_order_acquire))
            << "family=" << family;
        for (int participant = 0;
             participant < kParticipants;
             ++participant)
        {
            EXPECT_TRUE(replay_ok[participant])
                << "family=" << family
                << " participant=" << participant;
            if (!replay_ok[participant])
                continue;

            ASSERT_EQ(hipSetDevice(participant), hipSuccess);
            DeviceMoEOverlayEpochControl control{};
            DeviceMoEOverlayEpochTicket ticket{};
            DeviceMoEOverlayEpochStatus status{};
            ASSERT_EQ(
                hipMemcpy(
                    &control,
                    epoch_arenas[participant]->control(),
                    sizeof(control),
                    hipMemcpyDeviceToHost),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(
                    &ticket,
                    epoch_arenas[participant]->requestTicket(0u),
                    sizeof(ticket),
                    hipMemcpyDeviceToHost),
                hipSuccess);
            ASSERT_EQ(
                hipMemcpy(
                    &status,
                    epoch_arenas[participant]->requestStatus(0u),
                    sizeof(status),
                    hipMemcpyDeviceToHost),
                hipSuccess);
            EXPECT_EQ(control.bank_readers[0], 0u)
                << "family=" << family
                << " participant=" << participant;
            EXPECT_EQ(control.bank_readers[1], 0u)
                << "family=" << family
                << " participant=" << participant;
            EXPECT_EQ(control.acquisitions_in_flight, 0u)
                << "family=" << family
                << " participant=" << participant;
            EXPECT_FALSE(ticket.valid())
                << "family=" << family
                << " participant=" << participant;
            EXPECT_EQ(
                status.typedOperation(),
                DeviceMoEOverlayEpochOperation::Release)
                << "family=" << family
                << " participant=" << participant;
            EXPECT_EQ(
                status.typedCode(),
                DeviceMoEOverlayEpochStatusCode::Success)
                << "family=" << family
                << " participant=" << participant;
        }
        report_phase("replay_family_complete", -1, family);
    };
    ASSERT_NO_FATAL_FAILURE(replay_family(0));
    ASSERT_NO_FATAL_FAILURE(replay_family(kRetainedFamilies - 1));

    for (int participant = 0; participant < kParticipants; ++participant)
    {
        for (auto &family : parents)
            family[participant].reset();
        children[participant].clear();
        epoch_kernels[participant].reset();
        epoch_arenas[participant].reset();
        ASSERT_EQ(hipSetDevice(participant), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(streams[participant]), hipSuccess);
        freeDevicePtr(participant, values[participant]);
        freeDevicePtr(participant, rank_banks[participant]);
        freeDevicePtr(participant, route_values[participant]);
        freeDevicePtr(participant, compact_values[participant]);
        freeDevicePtr(participant, copy_sources[participant]);
        freeDevicePtr(participant, copy_destinations[participant]);
        for (void *allocation : residency_pressure[participant])
            EXPECT_EQ(hipFree(allocation), hipSuccess);
        residency_pressure[participant].clear();
    }
}

#else

TEST(Test__LocalTPRCCLGraphCapture, SkipsWithoutROCm)
{
    GTEST_SKIP() << "ROCm not enabled";
}

#endif
