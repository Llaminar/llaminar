/**
 * @file Perf__NCCLCollectiveLatency.cu
 * @brief Measures CUDA LocalTP collective latency at production decode shapes.
 *
 * Qwen3.6 MoE decode issues one small expert-output reduction per routed layer,
 * while phase-split GDN state publication gathers larger recurrent-state rows.
 * A whole-model profile can show that NCCL dominates a request, but it cannot
 * distinguish the irreducible transport floor from graph scheduling overhead.
 * This benchmark therefore drives the same explicit-stream NCCLCoordinator
 * APIs used by captured LocalTP graphs and reports their isolated GPU latency.
 *
 * The measured region contains no allocation, host/device transfer, device-wide
 * synchronization, or default-stream work. Persistent device buffers, streams,
 * and timing events are created by SetUp(); every sample records events on the
 * two producer streams and waits only after the complete collective is queued.
 *
 * The message matrix includes both Qwen3.6-35B MoE reductions: the shared
 * expert publishes M*d_model FP32 values, while serial-row-equivalent routed
 * experts preserve every top-k contribution and publish
 * M*top_k*d_model FP32 values before their canonical local reduction. The
 * latter is eight times larger for Qwen3.6-35B-A3B and is the production
 * transport floor that a plain M*d_model benchmark would hide. The sweep also
 * retains the one-megabyte recurrent-state allgather. A separate production
 * prefill matrix covers the Qwen3.5 canonical publication layout, whose
 * physical row contains `top_k + continuation_participants` FP32 banks. This
 * distinction matters at M=600 for Qwen3.5-122B-A10B: the captured rooted
 * reduction moves 70.31 MiB per layer even though the final replicated hidden
 * row is only 7.03 MiB.
 * Peer-access capability is printed so SHM/proxy measurements are never
 * mistaken for a CUDA P2P floor.
 */

#include "collective/coordinators/NCCLCoordinator.h"
#include "execution/local_execution/graph/GraphCaptureGuard.h"

#include <gtest/gtest.h>

#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2
{
namespace
{
    constexpr int kDeviceCount = 2;
    constexpr int kWarmupIterations = 12;
    constexpr int kMeasuredIterations = 100;
    constexpr size_t kQwen36MoEHidden = 2048u;
    constexpr size_t kQwen36MoETopK = 8u;
    constexpr size_t kMaximumVerifierRows = 16u;
    constexpr size_t kVerifierMaximumSendBytes =
        kMaximumVerifierRows * kQwen36MoEHidden * kQwen36MoETopK *
        sizeof(float);
    constexpr size_t kQwen35MoEHidden = 3072u;
    constexpr size_t kQwen35MaximumPrefillRows = 768u;
    constexpr size_t kQwen35CanonicalPublicationSlots =
        kQwen36MoETopK + kDeviceCount;
    constexpr size_t kQwen35MaximumPrefillPublicationBytes =
        kQwen35MaximumPrefillRows * kQwen35MoEHidden *
        kQwen35CanonicalPublicationSlots * sizeof(float);
    constexpr size_t kMaximumSendBytes =
        std::max(
            kVerifierMaximumSendBytes,
            kQwen35MaximumPrefillPublicationBytes);
    constexpr size_t kQwen35ProductionPrefillRows = 600u;
    constexpr size_t kQwen35ProductionAttentionBytes =
        kQwen35ProductionPrefillRows * kQwen35MoEHidden * sizeof(float);

    /**
     * @brief One production-relevant collective message geometry.
     */
    struct MessageShape
    {
        const char *label = nullptr;
        size_t send_bytes = 0;
        size_t canonical_output_bytes = 0;
    };

    constexpr std::array<MessageShape, 13> kMessageShapes{{
        {"shared M=1", 1u * kQwen36MoEHidden * sizeof(float)},
        {"shared M=4", 4u * kQwen36MoEHidden * sizeof(float)},
        {"shared M=8", 8u * kQwen36MoEHidden * sizeof(float)},
        {"shared M=16", 16u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=2", 2u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 2u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=4", 4u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 4u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=5", 5u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 5u * kQwen36MoEHidden * sizeof(float)},
        {"GDN recurrence shard", 1024u * 1024u},
        {"canonical M=8", 8u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 8u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=10", 10u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 10u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=12", 12u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 12u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=15", 15u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 15u * kQwen36MoEHidden * sizeof(float)},
        {"canonical M=16", 16u * kQwen36MoEHidden * kQwen36MoETopK * sizeof(float), 16u * kQwen36MoEHidden * sizeof(float)},
    }};

    /**
     * @brief Qwen3.5 production prefill publication geometries.
     *
     * Every physical row retains the eight original router slots followed by
     * one shared-expert bank per continuation participant. The current graph
     * reduces that entire fixed representation to its logical root before it
     * broadcasts one compact hidden row. Keeping this as a separate matrix
     * prevents verifier-only shapes from concealing the bulk prefill floor.
     */
    constexpr std::array<MessageShape, 5> kQwen35PrefillPublicationShapes{{
        {"Qwen35 publication M=64",
         64u * kQwen35MoEHidden * kQwen35CanonicalPublicationSlots *
             sizeof(float),
         64u * kQwen35MoEHidden * sizeof(float)},
        {"Qwen35 publication M=256",
         256u * kQwen35MoEHidden * kQwen35CanonicalPublicationSlots *
             sizeof(float),
         256u * kQwen35MoEHidden * sizeof(float)},
        {"Qwen35 publication M=512",
         512u * kQwen35MoEHidden * kQwen35CanonicalPublicationSlots *
             sizeof(float),
         512u * kQwen35MoEHidden * sizeof(float)},
        {"Qwen35 publication M=600",
         600u * kQwen35MoEHidden * kQwen35CanonicalPublicationSlots *
             sizeof(float),
         600u * kQwen35MoEHidden * sizeof(float)},
        {"Qwen35 publication M=768",
         768u * kQwen35MoEHidden * kQwen35CanonicalPublicationSlots *
             sizeof(float),
         768u * kQwen35MoEHidden * sizeof(float)},
    }};

    /**
     * @brief Reusable two-party rendezvous for capture and replay workers.
     *
     * Each CUDA participant owns one persistent host worker in production.
     * The benchmark uses the same participant-local enqueue contract and this
     * barrier prevents host scheduling skew from being mistaken for NCCL
     * transport latency.
     */
    class Barrier
    {
    public:
        explicit Barrier(int expected)
            : expected_(expected)
        {
        }

        void arriveAndWait()
        {
            std::unique_lock<std::mutex> lock(mutex_);
            const int generation = generation_;
            if (++arrived_ == expected_)
            {
                arrived_ = 0;
                ++generation_;
                condition_.notify_all();
                return;
            }
            condition_.wait(
                lock,
                [&]() { return generation_ != generation; });
        }

    private:
        const int expected_;
        int arrived_ = 0;
        int generation_ = 0;
        std::mutex mutex_;
        std::condition_variable condition_;
    };

    /**
     * @brief Distribution summary for repeated GPU-event measurements.
     */
    struct LatencySummary
    {
        double median_us = 0.0;
        double p95_us = 0.0;
        double minimum_us = 0.0;
        double maximum_us = 0.0;
    };

    /**
     * @brief Return one nearest-rank percentile from an already sorted sample.
     */
    double percentile(const std::vector<double> &sorted, double quantile)
    {
        if (sorted.empty())
            return 0.0;
        const double scaled =
            quantile * static_cast<double>(sorted.size() - 1u);
        const size_t index = static_cast<size_t>(std::ceil(scaled));
        return sorted[std::min(index, sorted.size() - 1u)];
    }

    /**
     * @brief Summarize a vector without changing the caller's sample order.
     */
    LatencySummary summarize(std::vector<double> samples)
    {
        std::sort(samples.begin(), samples.end());
        LatencySummary summary;
        if (samples.empty())
            return summary;
        summary.minimum_us = samples.front();
        summary.median_us = percentile(samples, 0.50);
        summary.p95_us = percentile(samples, 0.95);
        summary.maximum_us = samples.back();
        return summary;
    }

    /**
     * @brief Fail the current test with the exact CUDA operation and error.
     */
    void requireCudaSuccess(cudaError_t status, const char *operation)
    {
        if (status == cudaSuccess)
            return;

        throw std::runtime_error(
            std::string(operation) + " failed: " + cudaGetErrorString(status));
    }

    /** @return `value` rounded up to the next `alignment` boundary. */
    constexpr size_t alignUp(size_t value, size_t alignment) noexcept
    {
        return ((value + alignment - 1u) / alignment) * alignment;
    }

    /** @brief System-acquire load used by graph-resident epoch waiters. */
    __device__ __forceinline__ std::uint64_t loadSystemAcquire(
        std::uint64_t *address)
    {
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *address);
        return reference.load(cuda::memory_order_acquire);
    }

    /** @brief Release one completed mapped payload chunk to its peer GPU. */
    static __global__ void publishMappedEpochKernel(std::uint64_t *epoch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        __threadfence_system();
        cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system> reference(
            *epoch);
        reference.fetch_add(1u, cuda::memory_order_release);
    }

    /** @brief Root-side device wait for one fresh non-root input chunk. */
    static __global__ void waitForMappedInputKernel(
        std::uint64_t *input_epoch,
        std::uint64_t *output_epoch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        while (loadSystemAcquire(input_epoch) <=
               loadSystemAcquire(output_epoch))
        {
            __nanosleep(64u);
        }
    }

    /** @brief Non-root device wait for the root's matching output chunk. */
    static __global__ void waitForMappedOutputKernel(
        std::uint64_t *input_epoch,
        std::uint64_t *output_epoch)
    {
        if (blockIdx.x != 0u || threadIdx.x != 0u)
            return;
        const std::uint64_t expected = loadSystemAcquire(input_epoch);
        while (loadSystemAcquire(output_epoch) < expected)
            __nanosleep(64u);
    }

    /**
     * @brief Add one peer chunk in fixed participant order and publish it.
     *
     * The root reads its own input from VRAM, then the rank-one input from the
     * mapped aperture. Writing the same rounded FP32 result to root VRAM and
     * the mapped output preserves two-participant arithmetic exactly while
     * eliminating a separate root D2H copy from the critical path.
     */
    static __global__ void reduceAndPublishMappedChunkKernel(
        const float *root_input,
        const float *peer_input,
        float *root_output,
        float *mapped_output,
        size_t count)
    {
        for (size_t index =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             index < count;
             index += static_cast<size_t>(gridDim.x) * blockDim.x)
        {
            const float reduced = root_input[index] + peer_input[index];
            root_output[index] = reduced;
            mapped_output[index] = reduced;
        }
    }

    /**
     * @brief Retained mapped pages and endpoint-correct CUDA aliases.
     *
     * This is deliberately test infrastructure. Production installation must
     * obtain the same ownership through TransferEngine and expose it through a
     * typed LocalTP collective authority. The isolated benchmark uses raw CUDA
     * only so it can reject an uneconomical protocol before adding production
     * surface area.
     */
    class MappedAllreduceRegion final
    {
    public:
        MappedAllreduceRegion(size_t payload_bytes, size_t chunk_count)
            : payload_bytes_(payload_bytes),
              chunk_count_(chunk_count),
              output_offset_(alignUp(payload_bytes_, 256u)),
              input_epoch_offset_(
                  alignUp(output_offset_ + payload_bytes_, 256u)),
              output_epoch_offset_(
                  input_epoch_offset_ +
                  chunk_count_ * sizeof(std::uint64_t)),
              total_bytes_(
                  alignUp(
                      output_epoch_offset_ +
                          chunk_count_ * sizeof(std::uint64_t),
                      4096u))
        {
            requireCudaSuccess(
                cudaHostAlloc(
                    &host_base_,
                    total_bytes_,
                    cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc(mapped allreduce region)");
            std::memset(host_base_, 0, total_bytes_);
            for (int participant = 0;
                 participant < kDeviceCount;
                 ++participant)
            {
                requireCudaSuccess(
                    cudaSetDevice(participant),
                    "cudaSetDevice(mapped allreduce alias)");
                requireCudaSuccess(
                    cudaHostGetDevicePointer(
                        &device_bases_[static_cast<size_t>(participant)],
                        host_base_,
                        0u),
                    "cudaHostGetDevicePointer(mapped allreduce alias)");
            }
        }

        ~MappedAllreduceRegion()
        {
            if (host_base_)
                cudaFreeHost(host_base_);
        }

        MappedAllreduceRegion(const MappedAllreduceRegion &) = delete;
        MappedAllreduceRegion &operator=(
            const MappedAllreduceRegion &) = delete;

        /** @return CPU pointer used only as an asynchronous copy endpoint. */
        [[nodiscard]] void *hostInput(size_t byte_offset) const
        {
            return static_cast<std::byte *>(host_base_) + byte_offset;
        }

        /** @return CPU pointer to one root-published output chunk. */
        [[nodiscard]] void *hostOutput(size_t byte_offset) const
        {
            return static_cast<std::byte *>(host_base_) + output_offset_ +
                   byte_offset;
        }

        /** @return GPU alias for the non-root input payload. */
        [[nodiscard]] float *deviceInput(
            int participant,
            size_t byte_offset) const
        {
            return reinterpret_cast<float *>(
                static_cast<std::byte *>(
                    device_bases_[static_cast<size_t>(participant)]) +
                byte_offset);
        }

        /** @return GPU alias for the root-published output payload. */
        [[nodiscard]] float *deviceOutput(
            int participant,
            size_t byte_offset) const
        {
            return reinterpret_cast<float *>(
                static_cast<std::byte *>(
                    device_bases_[static_cast<size_t>(participant)]) +
                output_offset_ + byte_offset);
        }

        /** @return GPU alias for one non-root publication epoch. */
        [[nodiscard]] std::uint64_t *deviceInputEpoch(
            int participant,
            size_t chunk) const
        {
            return reinterpret_cast<std::uint64_t *>(
                static_cast<std::byte *>(
                    device_bases_[static_cast<size_t>(participant)]) +
                input_epoch_offset_) + chunk;
        }

        /** @return GPU alias for one root output epoch. */
        [[nodiscard]] std::uint64_t *deviceOutputEpoch(
            int participant,
            size_t chunk) const
        {
            return reinterpret_cast<std::uint64_t *>(
                static_cast<std::byte *>(
                    device_bases_[static_cast<size_t>(participant)]) +
                output_epoch_offset_) + chunk;
        }

    private:
        size_t payload_bytes_ = 0u;
        size_t chunk_count_ = 0u;
        size_t output_offset_ = 0u;
        size_t input_epoch_offset_ = 0u;
        size_t output_epoch_offset_ = 0u;
        size_t total_bytes_ = 0u;
        void *host_base_ = nullptr;
        std::array<void *, kDeviceCount> device_bases_{};
    };

    /**
     * @brief Persistent resources for one CUDA collective participant.
     */
    struct ParticipantResources
    {
        int ordinal = -1;
        void *buffer = nullptr;
        void *send_buffer = nullptr;
        cudaStream_t stream = nullptr;
        cudaEvent_t start_event = nullptr;
        cudaEvent_t stop_event = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t graph_executable = nullptr;
    };

    /**
     * @brief Native statuses produced while one participant records a graph.
     */
    struct CaptureStatus
    {
        bool collective_ok = false;
        cudaError_t set_device_status = cudaSuccess;
        cudaError_t begin_status = cudaSuccess;
        cudaError_t end_status = cudaSuccess;
        cudaError_t instantiate_status = cudaSuccess;
    };

    /**
     * @brief Fixture for isolated explicit-stream NCCL collective timing.
     */
    class Perf__NCCLCollectiveLatency : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            int available_devices = 0;
            if (cudaGetDeviceCount(&available_devices) != cudaSuccess ||
                available_devices < kDeviceCount)
            {
                GTEST_SKIP() << "Two CUDA devices are required";
            }

            std::vector<int> ordinals{0, 1};
            if (!coordinator_.initialize(ordinals))
            {
                GTEST_SKIP() << "NCCL coordinator initialization failed: "
                             << coordinator_.lastError();
            }
            initialized_ = true;

            for (int participant = 0; participant < kDeviceCount; ++participant)
            {
                ParticipantResources &resources =
                    participants_[static_cast<size_t>(participant)];
                resources.ordinal = ordinals[static_cast<size_t>(participant)];
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(SetUp)");
                requireCudaSuccess(
                    cudaStreamCreateWithFlags(
                        &resources.stream,
                        cudaStreamNonBlocking),
                    "cudaStreamCreateWithFlags");
                requireCudaSuccess(
                    cudaEventCreate(&resources.start_event),
                    "cudaEventCreate(start)");
                requireCudaSuccess(
                    cudaEventCreate(&resources.stop_event),
                    "cudaEventCreate(stop)");
                requireCudaSuccess(
                    cudaMalloc(
                        &resources.buffer,
                        kMaximumSendBytes * kDeviceCount),
                    "cudaMalloc(collective receive/in-place buffer)");
                requireCudaSuccess(
                    cudaMalloc(
                        &resources.send_buffer,
                        kMaximumSendBytes),
                    "cudaMalloc(allgather send buffer)");
                requireCudaSuccess(
                    cudaMemsetAsync(
                        resources.buffer,
                        participant + 1,
                        kMaximumSendBytes * kDeviceCount,
                        resources.stream),
                    "cudaMemsetAsync(collective buffer)");
                requireCudaSuccess(
                    cudaMemsetAsync(
                        resources.send_buffer,
                        participant + 1,
                        kMaximumSendBytes,
                        resources.stream),
                    "cudaMemsetAsync(allgather send buffer)");
            }

            streams_.reserve(kDeviceCount);
            buffers_.reserve(kDeviceCount);
            send_buffers_.reserve(kDeviceCount);
            for (const ParticipantResources &resources : participants_)
            {
                streams_.push_back(static_cast<void *>(resources.stream));
                buffers_.push_back(resources.buffer);
                send_buffers_.push_back(resources.send_buffer);
            }
        }

        void TearDown() override
        {
            for (ParticipantResources &resources : participants_)
            {
                if (resources.ordinal >= 0)
                    cudaSetDevice(resources.ordinal);
                destroyCapturedGraph(resources);
            }

            /*
             * Graph executables retain NCCL nodes, so their ownership must end
             * before the communicator they reference is shut down.
             */
            if (initialized_)
                coordinator_.shutdown();

            for (ParticipantResources &resources : participants_)
            {
                if (resources.ordinal >= 0)
                    cudaSetDevice(resources.ordinal);
                if (resources.stop_event)
                    cudaEventDestroy(resources.stop_event);
                if (resources.start_event)
                    cudaEventDestroy(resources.start_event);
                if (resources.buffer)
                    cudaFree(resources.buffer);
                if (resources.send_buffer)
                    cudaFree(resources.send_buffer);
                if (resources.stream)
                    cudaStreamDestroy(resources.stream);
                resources = {};
            }
        }

        /**
         * @brief Destroy one participant's captured graph on its owning GPU.
         */
        void destroyCapturedGraph(ParticipantResources &resources)
        {
            if (resources.graph_executable)
            {
                requireCudaSuccess(
                    cudaGraphExecDestroy(resources.graph_executable),
                    "cudaGraphExecDestroy");
                resources.graph_executable = nullptr;
            }
            if (resources.graph)
            {
                requireCudaSuccess(
                    cudaGraphDestroy(resources.graph),
                    "cudaGraphDestroy");
                resources.graph = nullptr;
            }
        }

        /**
         * @brief Capture one participant-local NCCL operation per GPU.
         *
         * CUDA/NCCL capture is participant-local: each worker records the
         * collective on the exact stream that owns its surrounding compute.
         * The two workers rendezvous after begin-capture and after NCCL enqueue
         * so neither participant closes its graph while its peer is still
         * publishing the matching collective.
         */
        template <typename ParticipantLaunch>
        void captureCollective(ParticipantLaunch &&launch)
        {
            for (ParticipantResources &resources : participants_)
            {
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(destroy prior graph)");
                destroyCapturedGraph(resources);
            }

            Barrier capture_started(kDeviceCount);
            Barrier collective_recorded(kDeviceCount);
            std::array<CaptureStatus, kDeviceCount> statuses{};

            auto capture_worker =
                [&](int participant)
            {
                ParticipantResources &resources =
                    participants_[static_cast<size_t>(participant)];
                CaptureStatus &status =
                    statuses[static_cast<size_t>(participant)];

                status.set_device_status = cudaSetDevice(resources.ordinal);
                if (status.set_device_status == cudaSuccess)
                {
                    status.begin_status = cudaStreamBeginCapture(
                        resources.stream,
                        cudaStreamCaptureModeRelaxed);
                }

                capture_started.arriveAndWait();
                if (status.set_device_status == cudaSuccess &&
                    status.begin_status == cudaSuccess)
                {
                    GraphCaptureGuard guard;
                    status.collective_ok = launch(participant, resources);
                }
                collective_recorded.arriveAndWait();

                if (status.set_device_status == cudaSuccess &&
                    status.begin_status == cudaSuccess)
                {
                    status.end_status = cudaStreamEndCapture(
                        resources.stream,
                        &resources.graph);
                }
                if (status.end_status == cudaSuccess && resources.graph)
                {
                    status.instantiate_status = cudaGraphInstantiate(
                        &resources.graph_executable,
                        resources.graph,
                        nullptr,
                        nullptr,
                        0);
                }
            };

            std::thread worker0(capture_worker, 0);
            std::thread worker1(capture_worker, 1);
            worker0.join();
            worker1.join();

            for (int participant = 0;
                 participant < kDeviceCount;
                 ++participant)
            {
                const CaptureStatus &status =
                    statuses[static_cast<size_t>(participant)];
                requireCudaSuccess(
                    status.set_device_status,
                    "cudaSetDevice(capture worker)");
                requireCudaSuccess(
                    status.begin_status,
                    "cudaStreamBeginCapture");
                if (!status.collective_ok)
                {
                    throw std::runtime_error(
                        "NCCL participant-local capture failed: " +
                        coordinator_.lastError());
                }
                requireCudaSuccess(
                    status.end_status,
                    "cudaStreamEndCapture");
                requireCudaSuccess(
                    status.instantiate_status,
                    "cudaGraphInstantiate");
                if (!participants_[static_cast<size_t>(participant)]
                         .graph_executable)
                {
                    throw std::runtime_error(
                        "CUDA graph instantiation returned a null executable");
                }
            }
        }

        /**
         * @brief Replay one captured graph per participant and time GPU work.
         *
         * Start events are recorded only after both persistent-worker stand-ins
         * reach the launch rendezvous. Consequently the sample includes graph
         * and NCCL device execution but excludes host thread creation and
         * scheduling skew before the launch boundary.
         */
        double measureCapturedOne()
        {
            Barrier launch_ready(kDeviceCount);
            std::array<cudaError_t, kDeviceCount> statuses{
                cudaSuccess,
                cudaSuccess};
            std::array<double, kDeviceCount> elapsed_us{0.0, 0.0};

            auto replay_worker =
                [&](int participant)
            {
                ParticipantResources &resources =
                    participants_[static_cast<size_t>(participant)];
                cudaError_t &status =
                    statuses[static_cast<size_t>(participant)];
                status = cudaSetDevice(resources.ordinal);
                launch_ready.arriveAndWait();
                if (status == cudaSuccess)
                {
                    status = cudaEventRecord(
                        resources.start_event,
                        resources.stream);
                }
                if (status == cudaSuccess)
                {
                    status = cudaGraphLaunch(
                        resources.graph_executable,
                        resources.stream);
                }
                if (status == cudaSuccess)
                {
                    status = cudaEventRecord(
                        resources.stop_event,
                        resources.stream);
                }
                if (status == cudaSuccess)
                    status = cudaEventSynchronize(resources.stop_event);
                if (status == cudaSuccess)
                {
                    float elapsed_ms = 0.0f;
                    status = cudaEventElapsedTime(
                        &elapsed_ms,
                        resources.start_event,
                        resources.stop_event);
                    elapsed_us[static_cast<size_t>(participant)] =
                        static_cast<double>(elapsed_ms) * 1000.0;
                }
            };

            std::thread worker0(replay_worker, 0);
            std::thread worker1(replay_worker, 1);
            worker0.join();
            worker1.join();

            for (cudaError_t status : statuses)
            {
                requireCudaSuccess(
                    status,
                    "participant-local cudaGraphLaunch/replay");
            }
            return *std::max_element(elapsed_us.begin(), elapsed_us.end());
        }

        /**
         * @brief Warm and measure the currently captured graph pair.
         */
        LatencySummary benchmarkCaptured()
        {
            for (int iteration = 0;
                 iteration < kWarmupIterations;
                 ++iteration)
            {
                (void)measureCapturedOne();
            }

            std::vector<double> samples;
            samples.reserve(kMeasuredIterations);
            for (int iteration = 0;
                 iteration < kMeasuredIterations;
                 ++iteration)
            {
                samples.push_back(measureCapturedOne());
            }
            return summarize(std::move(samples));
        }

        /**
         * @brief Return true only when CUDA can directly access both peers.
         */
        bool bidirectionalPeerAccessAvailable() const
        {
            int zero_to_one = 0;
            int one_to_zero = 0;
            if (cudaDeviceCanAccessPeer(&zero_to_one, 0, 1) != cudaSuccess ||
                cudaDeviceCanAccessPeer(&one_to_zero, 1, 0) != cudaSuccess)
            {
                return false;
            }
            return zero_to_one != 0 && one_to_zero != 0;
        }

        /**
         * @brief Time one launch callback using events on every participant.
         *
         * The returned sample is the slower participant's elapsed time because
         * the collective is complete only when every device has reached its
         * stop event. The host never calls cudaDeviceSynchronize().
         */
        template <typename Launch>
        double measureOne(Launch &&launch)
        {
            for (ParticipantResources &resources : participants_)
            {
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(record start)");
                requireCudaSuccess(
                    cudaEventRecord(resources.start_event, resources.stream),
                    "cudaEventRecord(start)");
            }

            if (!launch())
            {
                throw std::runtime_error(
                    "NCCL collective launch failed: " +
                    coordinator_.lastError());
            }

            for (ParticipantResources &resources : participants_)
            {
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(record stop)");
                requireCudaSuccess(
                    cudaEventRecord(resources.stop_event, resources.stream),
                    "cudaEventRecord(stop)");
            }

            double slowest_us = 0.0;
            for (ParticipantResources &resources : participants_)
            {
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(wait stop)");
                requireCudaSuccess(
                    cudaEventSynchronize(resources.stop_event),
                    "cudaEventSynchronize(stop)");
                float elapsed_ms = 0.0f;
                requireCudaSuccess(
                    cudaEventElapsedTime(
                        &elapsed_ms,
                        resources.start_event,
                        resources.stop_event),
                    "cudaEventElapsedTime");
                slowest_us =
                    std::max(slowest_us, static_cast<double>(elapsed_ms) * 1000.0);
            }
            return slowest_us;
        }

        /**
         * @brief Warm and measure one fixed collective launch shape.
         */
        template <typename Launch>
        LatencySummary benchmark(Launch &&launch)
        {
            for (int iteration = 0;
                 iteration < kWarmupIterations;
                 ++iteration)
            {
                (void)measureOne(launch);
            }

            std::vector<double> samples;
            samples.reserve(kMeasuredIterations);
            for (int iteration = 0;
                 iteration < kMeasuredIterations;
                 ++iteration)
            {
                samples.push_back(measureOne(launch));
            }
            return summarize(std::move(samples));
        }

        /**
         * @brief Print one compact table row.
         */
        void printResult(
            const char *operation,
            const MessageShape &shape,
            const LatencySummary &summary) const
        {
            std::cout << std::left << std::setw(10) << operation
                      << std::setw(24) << shape.label
                      << std::right << std::setw(10) << shape.send_bytes
                      << std::setw(12) << std::fixed << std::setprecision(2)
                      << summary.median_us
                      << std::setw(12) << summary.p95_us
                      << std::setw(12) << summary.minimum_us
                      << std::setw(12) << summary.maximum_us << '\n';
        }

        NCCLCoordinator coordinator_;
        bool initialized_ = false;
        std::array<ParticipantResources, kDeviceCount> participants_{};
        std::vector<void *> streams_;
        std::vector<void *> buffers_;
        std::vector<const void *> send_buffers_;
    };

    /**
     * @brief Sweep production MoE allreduce message sizes.
     */
    TEST_F(Perf__NCCLCollectiveLatency, ExplicitStreamAllreduce)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        for (const MessageShape &shape : kMessageShapes)
        {
            const size_t count = shape.send_bytes / sizeof(float);
            const LatencySummary summary = benchmark([&]()
            {
                return coordinator_.allreduceMultiOnStreams(
                    buffers_,
                    count,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM,
                    streams_);
            });
            printResult("allreduce", shape, summary);
        }
    }

    /**
     * @brief Sweep explicit-stream allgather at the same production geometries.
     */
    TEST_F(Perf__NCCLCollectiveLatency, ExplicitStreamAllgather)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        for (const MessageShape &shape : kMessageShapes)
        {
            CollectiveSidebandMultiOnStreamsOp operation;
            operation.kind = CollectiveSidebandOp::Allgather;
            operation.send_buffers = send_buffers_;
            operation.recv_buffers = buffers_;
            operation.count = shape.send_bytes / sizeof(float);
            operation.dtype = CollectiveDataType::FLOAT32;

            const std::vector<CollectiveSidebandMultiOnStreamsOp> operations{
                operation};
            const LatencySummary summary = benchmark([&]()
            {
                return coordinator_.collectiveSidebandsMultiOnStreams(
                    operations,
                    streams_);
            });
            printResult("allgather", shape, summary);
        }
    }

    /**
     * @brief Sweep graph-captured participant-local allreduces.
     */
    TEST_F(Perf__NCCLCollectiveLatency, GraphCapturedAllreduce)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        for (const MessageShape &shape : kMessageShapes)
        {
            const size_t count = shape.send_bytes / sizeof(float);
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    return coordinator_.allreduceSingleDeviceOnStream(
                        resources.buffer,
                        count,
                        CollectiveDataType::FLOAT32,
                        CollectiveOp::ALLREDUCE_SUM,
                        participant,
                        resources.stream);
                });
            printResult("graph_ar", shape, benchmarkCaptured());
        }
    }

    /**
     * @brief Compare NCCL with a chunked device-epoch mapped-host allreduce.
     *
     * This candidate targets a homogeneous two-GPU continuation domain whose
     * PCIe topology exposes no CUDA P2P. The non-root graph publishes fixed
     * chunks through the D2H engine, the root consumes and adds them in fixed
     * participant order while writing the result back to mapped pages, and the
     * non-root graph imports completed chunks. All waits are device-owned
     * system-scope epochs captured in the participant graphs; the host only
     * submits the two retained graphs and observes their final timing events.
     *
     * Exact 1.0f + 2.0f output is checked on every element and participant for
     * every chunk geometry before timing. This is an isolated design gate, not
     * a production fallback: an economical result must still be installed as
     * a typed LocalTP authority using TransferEngine-owned mapped pages.
     */
    TEST_F(Perf__NCCLCollectiveLatency,
           GraphCapturedMappedHostChunkedAllreduce)
    {
        if (!initialized_)
            GTEST_SKIP();

        const size_t count =
            kQwen35ProductionAttentionBytes / sizeof(float);
        const std::array<std::vector<float>, kDeviceCount> inputs{
            std::vector<float>(count, 1.0f),
            std::vector<float>(count, 2.0f),
        };

        for (int participant = 0;
             participant < kDeviceCount;
             ++participant)
        {
            auto &resources =
                participants_[static_cast<size_t>(participant)];
            requireCudaSuccess(
                cudaSetDevice(resources.ordinal),
                "cudaSetDevice(mapped candidate input)");
            requireCudaSuccess(
                cudaMemcpyAsync(
                    resources.buffer,
                    inputs[static_cast<size_t>(participant)].data(),
                    kQwen35ProductionAttentionBytes,
                    cudaMemcpyHostToDevice,
                    resources.stream),
                "cudaMemcpyAsync(NCCL baseline input)");
            requireCudaSuccess(
                cudaMemcpyAsync(
                    resources.send_buffer,
                    inputs[static_cast<size_t>(participant)].data(),
                    kQwen35ProductionAttentionBytes,
                    cudaMemcpyHostToDevice,
                    resources.stream),
                "cudaMemcpyAsync(mapped candidate input)");
            requireCudaSuccess(
                cudaStreamSynchronize(resources.stream),
                "cudaStreamSynchronize(candidate setup)");
        }

        captureCollective(
            [&](int participant, ParticipantResources &resources)
            {
                return coordinator_.allreduceSingleDeviceOnStream(
                    resources.buffer,
                    count,
                    CollectiveDataType::FLOAT32,
                    CollectiveOp::ALLREDUCE_SUM,
                    participant,
                    resources.stream);
            });
        const LatencySummary nccl = benchmarkCaptured();

        const MessageShape attention_shape{
            "Qwen35 attention M=600",
            kQwen35ProductionAttentionBytes,
            kQwen35ProductionAttentionBytes,
        };
        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; measuring mapped-host candidate")
                  << "\n";
        std::cout << std::left << std::setw(18) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';
        std::cout << std::left << std::setw(18) << "nccl_allreduce"
                  << std::setw(24) << attention_shape.label
                  << std::right << std::setw(10)
                  << attention_shape.send_bytes
                  << std::setw(12) << std::fixed << std::setprecision(2)
                  << nccl.median_us
                  << std::setw(12) << nccl.p95_us
                  << std::setw(12) << nccl.minimum_us
                  << std::setw(12) << nccl.maximum_us << '\n';

        constexpr std::array<size_t, 5> candidate_chunk_bytes{
            64u * 1024u,
            256u * 1024u,
            1024u * 1024u,
            4u * 1024u * 1024u,
            kQwen35ProductionAttentionBytes,
        };
        for (const size_t requested_chunk_bytes :
             candidate_chunk_bytes)
        {
            const size_t chunk_bytes = std::min(
                requested_chunk_bytes,
                kQwen35ProductionAttentionBytes);
            const size_t chunk_count =
                (kQwen35ProductionAttentionBytes + chunk_bytes - 1u) /
                chunk_bytes;
            MappedAllreduceRegion region(
                kQwen35ProductionAttentionBytes,
                chunk_count);

            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    constexpr int root = 0;
                    constexpr int non_root = 1;
                    if (participant == non_root)
                    {
                        /* Publish every input chunk first. The root consumes
                         * completed early chunks while this stream's D2H
                         * engine advances through the remaining payload. */
                        for (size_t chunk = 0u;
                             chunk < chunk_count;
                             ++chunk)
                        {
                            const size_t byte_offset = chunk * chunk_bytes;
                            const size_t live_bytes = std::min(
                                chunk_bytes,
                                kQwen35ProductionAttentionBytes -
                                    byte_offset);
                            const auto *source =
                                static_cast<const std::byte *>(
                                    resources.send_buffer) + byte_offset;
                            if (cudaMemcpyAsync(
                                    region.hostInput(byte_offset),
                                    source,
                                    live_bytes,
                                    cudaMemcpyDeviceToHost,
                                    resources.stream) != cudaSuccess)
                            {
                                return false;
                            }
                            publishMappedEpochKernel<<<
                                1,
                                1,
                                0,
                                resources.stream>>>(
                                region.deviceInputEpoch(
                                    participant, chunk));
                        }

                        /* Import in the same chunk order after all outbound
                         * copies are queued. Device waits make an early root
                         * completion cheap and preserve one cyclic epoch per
                         * chunk without host intervention. */
                        for (size_t chunk = 0u;
                             chunk < chunk_count;
                             ++chunk)
                        {
                            const size_t byte_offset = chunk * chunk_bytes;
                            const size_t live_bytes = std::min(
                                chunk_bytes,
                                kQwen35ProductionAttentionBytes -
                                    byte_offset);
                            waitForMappedOutputKernel<<<
                                1,
                                1,
                                0,
                                resources.stream>>>(
                                region.deviceInputEpoch(
                                    participant, chunk),
                                region.deviceOutputEpoch(
                                    participant, chunk));
                            auto *destination =
                                static_cast<std::byte *>(resources.buffer) +
                                byte_offset;
                            if (cudaMemcpyAsync(
                                    destination,
                                    region.hostOutput(byte_offset),
                                    live_bytes,
                                    cudaMemcpyHostToDevice,
                                    resources.stream) != cudaSuccess)
                            {
                                return false;
                            }
                        }
                    }
                    else if (participant == root)
                    {
                        for (size_t chunk = 0u;
                             chunk < chunk_count;
                             ++chunk)
                        {
                            const size_t byte_offset = chunk * chunk_bytes;
                            const size_t live_bytes = std::min(
                                chunk_bytes,
                                kQwen35ProductionAttentionBytes -
                                    byte_offset);
                            const size_t live_count =
                                live_bytes / sizeof(float);
                            waitForMappedInputKernel<<<
                                1,
                                1,
                                0,
                                resources.stream>>>(
                                region.deviceInputEpoch(
                                    participant, chunk),
                                region.deviceOutputEpoch(
                                    participant, chunk));
                            const int blocks = static_cast<int>(std::min<size_t>(
                                1024u,
                                (live_count + 255u) / 256u));
                            reduceAndPublishMappedChunkKernel<<<
                                blocks,
                                256,
                                0,
                                resources.stream>>>(
                                static_cast<const float *>(
                                    resources.send_buffer) +
                                    byte_offset / sizeof(float),
                                region.deviceInput(
                                    participant, byte_offset),
                                static_cast<float *>(resources.buffer) +
                                    byte_offset / sizeof(float),
                                region.deviceOutput(
                                    participant, byte_offset),
                                live_count);
                            publishMappedEpochKernel<<<
                                1,
                                1,
                                0,
                                resources.stream>>>(
                                region.deviceOutputEpoch(
                                    participant, chunk));
                        }
                    }
                    else
                    {
                        return false;
                    }
                    return cudaPeekAtLastError() == cudaSuccess;
                });

            /* One replay proves the captured protocol before performance is
             * sampled. Blocking copies below are diagnostic-only and occur
             * after both retained graphs have completed. */
            (void)measureCapturedOne();
            for (int participant = 0;
                 participant < kDeviceCount;
                 ++participant)
            {
                auto &resources =
                    participants_[static_cast<size_t>(participant)];
                std::vector<float> output(count, 0.0f);
                requireCudaSuccess(
                    cudaSetDevice(resources.ordinal),
                    "cudaSetDevice(mapped candidate verification)");
                requireCudaSuccess(
                    cudaMemcpy(
                        output.data(),
                        resources.buffer,
                        kQwen35ProductionAttentionBytes,
                        cudaMemcpyDeviceToHost),
                    "cudaMemcpy(mapped candidate verification)");
                const auto mismatch = std::find_if(
                    output.begin(),
                    output.end(),
                    [](float value) { return value != 3.0f; });
                ASSERT_EQ(mismatch, output.end())
                    << "participant=" << participant
                    << " chunk_bytes=" << chunk_bytes
                    << " mismatch_index="
                    << std::distance(output.begin(), mismatch)
                    << " value="
                    << (mismatch == output.end() ? 3.0f : *mismatch);
            }

            const LatencySummary mapped = benchmarkCaptured();
            const std::string operation =
                "mapped_" + std::to_string(chunk_bytes / 1024u) + "K";
            std::cout << std::left << std::setw(18) << operation
                      << std::setw(24) << attention_shape.label
                      << std::right << std::setw(10)
                      << attention_shape.send_bytes
                      << std::setw(12) << mapped.median_us
                      << std::setw(12) << mapped.p95_us
                      << std::setw(12) << mapped.minimum_us
                      << std::setw(12) << mapped.maximum_us << '\n';
        }
    }

    /**
     * @brief Sweep graph-captured participant-local allgathers.
     */
    TEST_F(Perf__NCCLCollectiveLatency, GraphCapturedAllgather)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        for (const MessageShape &shape : kMessageShapes)
        {
            const size_t count = shape.send_bytes / sizeof(float);
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    return coordinator_.allgatherSingleDeviceOnStream(
                        resources.send_buffer,
                        resources.buffer,
                        count,
                        CollectiveDataType::FLOAT32,
                        participant,
                        resources.stream);
                });
            printResult("graph_ag", shape, benchmarkCaptured());
        }
    }

    /**
     * @brief Compare canonical allreduce with root-reduce plus compact broadcast.
     *
     * Routed expert contributions are already independently rounded and stored
     * in original router order. Reducing those slots to one fixed root retains
     * the exact representation consumed by the serial-order reducer. The real
     * graph would run that reducer between the two collectives, then broadcast
     * only M*d_model final values. This transport-only benchmark omits the
     * small reducer kernel but preserves both collective payloads and their
     * stream dependency, establishing whether the architecture has enough
     * headroom to justify a production graph implementation.
     */
    TEST_F(Perf__NCCLCollectiveLatency, GraphCapturedRootReduceBroadcast)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        constexpr int root = 0;
        for (const MessageShape &shape : kMessageShapes)
        {
            if (shape.canonical_output_bytes == 0)
                continue;

            const size_t route_count = shape.send_bytes / sizeof(float);
            const size_t output_count =
                shape.canonical_output_bytes / sizeof(float);
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    if (!coordinator_.reduceSingleDeviceOnStream(
                            resources.buffer,
                            resources.buffer,
                            route_count,
                            CollectiveDataType::FLOAT32,
                            CollectiveOp::ALLREDUCE_SUM,
                            root,
                            participant,
                            resources.stream))
                    {
                        return false;
                    }

                    /*
                     * The production reducer writes a compact M*d_model row
                     * bank before this broadcast. For transport timing, the
                     * first output_count values are a stable-address stand-in;
                     * NCCL sees the identical byte count and stream ordering.
                     */
                    return coordinator_.broadcastSingleDeviceOnStream(
                        resources.buffer,
                        resources.send_buffer,
                        output_count,
                        CollectiveDataType::FLOAT32,
                        root,
                        participant,
                        resources.stream);
                });
            printResult("red+bcast", shape, benchmarkCaptured());
        }
    }

    /**
     * @brief Measure the exact bulk rooted publication used by Qwen3.5 prefill.
     *
     * Unlike the verifier matrix above, these payloads include both shared
     * expert participant banks in the reduced physical row. The capture has
     * the same ReduceSum -> compact Broadcast ordering as the production
     * continuation graph. No expert compute or remote-tier wait is present,
     * so the result is the transport floor against which the end-to-end
     * per-layer collective interval must be compared.
     */
    TEST_F(Perf__NCCLCollectiveLatency,
           GraphCapturedQwen35PrefillPublication)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(30) << "shape"
                  << std::right << std::setw(12) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        constexpr int root = 0;
        for (const MessageShape &shape :
             kQwen35PrefillPublicationShapes)
        {
            const size_t publication_count =
                shape.send_bytes / sizeof(float);
            const size_t output_count =
                shape.canonical_output_bytes / sizeof(float);
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    if (!coordinator_.reduceSingleDeviceOnStream(
                            resources.buffer,
                            resources.buffer,
                            publication_count,
                            CollectiveDataType::FLOAT32,
                            CollectiveOp::ALLREDUCE_SUM,
                            root,
                            participant,
                            resources.stream))
                    {
                        return false;
                    }
                    return coordinator_.broadcastSingleDeviceOnStream(
                        resources.buffer,
                        resources.send_buffer,
                        output_count,
                        CollectiveDataType::FLOAT32,
                        root,
                        participant,
                        resources.stream);
                });
            printResult("red+bcast", shape, benchmarkCaptured());
        }
    }

    /**
     * @brief Compare root-reduce against a directed route-slot gather.
     *
     * Canonical route slots have exactly one producing participant. Therefore
     * the collective does not mathematically need to add two useful values:
     * it only has to make every non-root participant's slots visible to the
     * fixed root. This candidate uses graph-captured NCCL send/recv for that
     * directed handoff, after which the production reducer could read the
     * root-local and received banks in deterministic participant and router
     * order. The compact M*d_model broadcast remains unchanged.
     *
     * This transport benchmark deliberately omits the tiny deterministic merge
     * kernel, just as GraphCapturedQwen35PrefillPublication omits the canonical
     * reducer. Every participant nevertheless records the exact production
     * ordering on its explicit stream: send/recv, reducer position, then
     * broadcast. No peer memcpy, host staging, allocation, synchronization, or
     * default-stream operation is introduced by this candidate. The exact
     * Qwen3.5 prefill matrix prevents a verifier-sized result from being used
     * to justify a bulk production lowering.
     *
     * The current fixture has two participants, so the root receives one full
     * route-slot bank. General degree-N lowering would provide one persistent
     * receive bank per non-root and issue all matching receives in the same
     * NCCL group before running the deterministic merge.
     */
    TEST_F(Perf__NCCLCollectiveLatency, GraphCapturedRootGatherBroadcast)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(10) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        constexpr int root = 0;
        constexpr int non_root = 1;
        for (const MessageShape &shape :
             kQwen35PrefillPublicationShapes)
        {
            if (shape.canonical_output_bytes == 0)
                continue;

            const size_t route_count = shape.send_bytes / sizeof(float);
            const size_t output_count =
                shape.canonical_output_bytes / sizeof(float);
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    CollectiveP2POp route_handoff;
                    route_handoff.count = route_count;
                    route_handoff.dtype = CollectiveDataType::FLOAT32;
                    if (participant == root)
                    {
                        route_handoff.kind = CollectiveP2POpKind::Recv;
                        route_handoff.recv_buffer = resources.send_buffer;
                        route_handoff.peer = non_root;
                    }
                    else
                    {
                        route_handoff.kind = CollectiveP2POpKind::Send;
                        route_handoff.send_buffer = resources.buffer;
                        route_handoff.peer = root;
                    }

                    if (!coordinator_.groupedP2PSingleDeviceOnStream(
                            {route_handoff},
                            participant,
                            resources.stream))
                    {
                        return false;
                    }

                    /*
                     * The production graph inserts its deterministic
                     * participant/router-order merge here. The root's first
                     * output_count values are a stable-address stand-in for
                     * the merged row bank in this transport-only comparison.
                     */
                    return coordinator_.broadcastSingleDeviceOnStream(
                        resources.buffer,
                        resources.send_buffer,
                        output_count,
                        CollectiveDataType::FLOAT32,
                        root,
                        participant,
                        resources.stream);
                });
            printResult("gath+bcast", shape, benchmarkCaptured());
        }
    }

    /**
     * @brief Compare the current routed/shared transaction with one rooted payload.
     *
     * The production LocalTP MoE graph currently publishes two independent
     * branch results. Routed expert slots take the byte-exact root-reduce plus
     * compact-broadcast path, while the input-parallel shared-expert down row
     * takes a separate allreduce. Both custom epilogues are intentionally
     * omitted here: the current route-order fold and the candidate fused
     * route-fold/shared-gate epilogue do comparable device-local work. This A/B
     * therefore measures only the communication architecture we are deciding
     * whether to lower into the production graph.
     *
     * The candidate appends one M*d_model shared-partial bank per participant
     * after the M*top_k*d_model route slots. Each participant writes only its
     * own bank and zeroes its peers' banks, so NCCL transports the shared
     * evidence without choosing its floating-point reduction order. The fixed
     * root can then fold participant banks in ascending rank order, preserving
     * serial-row arithmetic independently of NCCL algorithm selection. One
     * compact M*d_model broadcast publishes the final FFN delta. Every address
     * is persistent and both variants execute inside one participant-local
     * CUDA graph with the same explicit stream contract as production.
     */
    TEST_F(Perf__NCCLCollectiveLatency, GraphCapturedCombinedMoETransaction)
    {
        if (!initialized_)
            GTEST_SKIP();

        std::cout << "\nCUDA peer access: "
                  << (bidirectionalPeerAccessAvailable()
                          ? "bidirectional P2P"
                          : "unavailable; NCCL will use a non-P2P transport")
                  << "\n";
        std::cout << std::left << std::setw(16) << "operation"
                  << std::setw(24) << "shape"
                  << std::right << std::setw(10) << "bytes"
                  << std::setw(12) << "median_us"
                  << std::setw(12) << "p95_us"
                  << std::setw(12) << "min_us"
                  << std::setw(12) << "max_us" << '\n';

        constexpr int root = 0;
        for (const MessageShape &shape : kMessageShapes)
        {
            if (shape.canonical_output_bytes == 0)
                continue;

            const size_t route_count = shape.send_bytes / sizeof(float);
            const size_t output_count =
                shape.canonical_output_bytes / sizeof(float);

            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    if (!coordinator_.reduceSingleDeviceOnStream(
                            resources.buffer,
                            resources.buffer,
                            route_count,
                            CollectiveDataType::FLOAT32,
                            CollectiveOp::ALLREDUCE_SUM,
                            root,
                            participant,
                            resources.stream))
                    {
                        return false;
                    }
                    if (!coordinator_.broadcastSingleDeviceOnStream(
                            resources.buffer,
                            resources.send_buffer,
                            output_count,
                            CollectiveDataType::FLOAT32,
                            root,
                            participant,
                            resources.stream))
                    {
                        return false;
                    }

                    float *shared_partial =
                        static_cast<float *>(resources.buffer) + route_count;
                    return coordinator_.allreduceSingleDeviceOnStream(
                        shared_partial,
                        output_count,
                        CollectiveDataType::FLOAT32,
                        CollectiveOp::ALLREDUCE_SUM,
                        participant,
                        resources.stream);
                });
            const LatencySummary split = benchmarkCaptured();

            const size_t deterministic_publication_count =
                route_count +
                static_cast<size_t>(kDeviceCount) * output_count;
            captureCollective(
                [&](int participant, ParticipantResources &resources)
                {
                    if (!coordinator_.reduceSingleDeviceOnStream(
                            resources.buffer,
                            resources.buffer,
                            deterministic_publication_count,
                            CollectiveDataType::FLOAT32,
                            CollectiveOp::ALLREDUCE_SUM,
                            root,
                            participant,
                            resources.stream))
                    {
                        return false;
                    }
                    return coordinator_.broadcastSingleDeviceOnStream(
                        resources.buffer,
                        resources.send_buffer,
                        output_count,
                        CollectiveDataType::FLOAT32,
                        root,
                        participant,
                        resources.stream);
                });
            const LatencySummary combined = benchmarkCaptured();

            const MessageShape split_shape{
                shape.label,
                shape.send_bytes + shape.canonical_output_bytes,
                shape.canonical_output_bytes};
            const MessageShape deterministic_shape{
                shape.label,
                shape.send_bytes +
                    static_cast<size_t>(kDeviceCount) *
                        shape.canonical_output_bytes,
                shape.canonical_output_bytes};
            printResult("split_moe", split_shape, split);
            printResult("combined_rank_banks", deterministic_shape, combined);
            std::cout << "  speedup=" << std::fixed << std::setprecision(3)
                      << (combined.median_us > 0.0
                              ? split.median_us / combined.median_us
                              : 0.0)
                      << "x\n";
        }
    }

} // namespace
} // namespace llaminar2
