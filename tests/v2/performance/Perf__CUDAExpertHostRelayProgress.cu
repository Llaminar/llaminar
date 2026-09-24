/**
 * @file Perf__CUDAExpertHostRelayProgress.cu
 * @brief Economy proof for CUDA expert D2H progress under a captured epoch.
 *
 * A CUDA host-relay lane normally uses asynchronous D2H DMA. NVIDIA documents
 * that stream priorities do not influence H2D/D2H scheduling, so a saturated
 * continuation device can leave that copy queued for an entire inference
 * segment. A separately launched high-priority copy kernel also cannot preempt
 * a resident mapped-ticket wait kernel. This benchmark therefore compares both
 * launch-after-admission approaches with a bounded progress branch captured as
 * part of the inference graph and driven by fixed-size mapped commands.
 *
 * The synthetic inference epoch is a retained CUDA graph containing many
 * production-sized memory-pressure kernels. Each candidate starts only after a
 * lead kernel proves the inference stream is live. Its pressure nodes read and
 * write mapped system memory, matching the node-local activation channel that
 * distinguishes the production continuation graph from an ordinary VRAM-only
 * workload. The fixture measures the
 * candidate's submit-to-event wall time, its device time, and the concurrent
 * graph duration. Every copied byte is compared with the source pattern.
 *
 * The timed region contains no allocation, free, synchronization, host
 * callback, or default-stream operation. Event queries are the only host-side
 * completion mechanism. Set `LLAMINAR_CUDA_EXPERT_RELAY_PROFILE_ONLY=1` to
 * reduce the test to the mapped-copy kernel for an isolated Nsight Compute run.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "transfer/TransferEngine.h"

#if defined(HAVE_CUDA)

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2
{
namespace
{
    constexpr std::size_t kExpertChunkBytes = 4u * 1024u * 1024u;
    constexpr std::size_t kPressureBytes = 1u * 1024u * 1024u;
    constexpr int kPressureGraphNodes = 1024;
    constexpr int kDefaultCapturedProgressBlocks = 1;
    constexpr int kCapturedWaitMilliseconds = 40;
    constexpr int kWarmupSamples = 1;
    constexpr int kMeasuredSamples = 5;
    constexpr auto kCompletionDeadline = std::chrono::seconds(10);

    /** @brief Throw one exact CUDA diagnostic during fixture setup. */
    void requireCuda(cudaError_t status, const char *operation)
    {
        if (status != cudaSuccess)
        {
            throw std::runtime_error(
                std::string(operation) + ": " + cudaGetErrorString(status));
        }
    }

    /**
     * @brief Keep the inference epoch memory-bandwidth active between graph nodes.
     *
     * The grid-stride pass reads and writes the complete mapped system-memory
     * allocation. Distinct salts prevent graph nodes from becoming
     * semantically redundant.
     */
    __global__ void capturedMemoryPressureKernel(
        std::uint32_t *__restrict__ words,
        std::size_t word_count,
        std::uint32_t salt)
    {
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t index =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x;
             index < word_count;
             index += stride)
        {
            const std::uint32_t value = words[index];
            words[index] = (value ^ salt) * 1664525u + 1013904223u;
        }
    }

    /**
     * @brief Model a production mapped-ticket wait that temporarily owns all SMs.
     *
     * ExpertOverlay participant graphs contain device-side waits on mapped
     * node-local tickets. A later high-priority launch cannot preempt a kernel
     * that is already resident, which was the missing condition in the original
     * synthetic benchmark. Every thread performs a volatile mapped load so this
     * interval also retains the relevant PCIe pressure.
     */
    __global__ void capturedMappedTicketWaitKernel(
        volatile const std::uint32_t *__restrict__ signal,
        unsigned long long wait_cycles)
    {
        const unsigned long long begin = clock64();
        std::uint32_t witness = 0;
        while (clock64() - begin < wait_cycles)
        {
            witness ^= signal[
                (static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x) %
                (kPressureBytes / sizeof(std::uint32_t))];
        }
        if (witness == 0x6d5a56a5u)
            const_cast<std::uint32_t *>(signal)[0] = witness;
    }

    /**
     * @brief Host/device command page for the captured progress branch.
     *
     * The host fills every immutable command field and publishes `generation`
     * last. The device publishes `completed_generation` only after all branch
     * blocks have copied the bytes and issued a system fence. Keeping the
     * command page mapped avoids a transfer-stream dependency merely to submit
     * transfer metadata.
     */
    struct alignas(64) CapturedCopyCommand
    {
        std::uint64_t generation = 0;
        std::uint64_t completed_generation = 0;
        std::uint64_t source_address = 0;
        std::uint64_t destination_address = 0;
        std::uint64_t bytes = 0;
    };

    /**
     * @brief Device-resident immutable snapshot consumed by one graph replay.
     *
     * A tiny claim node copies the published mapped command into this storage
     * before the inference and transfer branches fork. The copy branch therefore
     * never observes a partially replaced host command, and the host remains
     * free to poll only the completion cache line while the replay is active.
     */
    struct alignas(64) CapturedCopyClaim
    {
        std::uint64_t generation = 0;
        std::uint64_t source_address = 0;
        std::uint64_t destination_address = 0;
        std::uint64_t bytes = 0;
    };

    /**
     * @brief Snapshot one mapped command before graph branches execute.
     *
     * This single-thread node is the only mapped-command reader on the device.
     * Its graph dependency is the publication/acquire edge for both following
     * branches; the transfer branch can then use ordinary device memory.
     */
    __global__ void capturedCopyCommandClaimKernel(
        volatile const CapturedCopyCommand *__restrict__ command,
        CapturedCopyClaim *__restrict__ claim)
    {
        if (blockIdx.x != 0 || threadIdx.x != 0)
            return;

        const std::uint64_t generation = command->generation;
        claim->source_address = command->source_address;
        claim->destination_address = command->destination_address;
        claim->bytes = command->bytes;
        __threadfence();
        claim->generation = generation;
    }

    /**
     * @brief Finite source-to-mapped-host branch embedded in inference graphs.
     *
     * Every retained graph contains this branch from capture time, so its launch
     * cannot sit behind an already-resident mapped-ticket wait. An idle replay
     * exits immediately. An admitted generation uses the fixed branch grid to
     * copy disjoint vector lanes while the main inference branch runs. The
     * device-only counter supplies the grid join and the last block publishes
     * one system-visible generation to the CPU.
     */
    __global__ void capturedMappedHostCopyBranchKernel(
        const CapturedCopyClaim *__restrict__ claim,
        volatile CapturedCopyCommand *__restrict__ command,
        unsigned int *__restrict__ completed_blocks)
    {
        const std::uint64_t generation = claim->generation;
        if (generation == 0 ||
            generation == command->completed_generation)
        {
            return;
        }

        const auto *const source = reinterpret_cast<const uint4 *>(
            static_cast<std::uintptr_t>(claim->source_address));
        auto *const destination = reinterpret_cast<uint4 *>(
            static_cast<std::uintptr_t>(claim->destination_address));
        const std::size_t vector_count =
            static_cast<std::size_t>(claim->bytes) / sizeof(uint4);
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t index =
                 static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                 threadIdx.x;
             index < vector_count;
             index += stride)
        {
            destination[index] = source[index];
        }
        __threadfence_system();
        __syncthreads();

        if (threadIdx.x == 0)
        {
            const unsigned int prior = atomicAdd(completed_blocks, 1u);
            if (prior + 1u == gridDim.x)
            {
                atomicExch(completed_blocks, 0u);
                __threadfence_system();
                command->completed_generation = generation;
                __threadfence_system();
            }
        }
    }

    /** @brief One concurrent epoch/copy observation. */
    struct Observation
    {
        double inference_ms = 0.0;
        double copy_device_ms = 0.0;
        double copy_wall_ms = 0.0;
    };

    /** @brief Median of a small already-materialized sample vector. */
    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2u;
        return values.size() % 2u == 0u
                   ? (values[middle - 1u] + values[middle]) * 0.5
                   : values[middle];
    }

    /** @brief Host observation times for two independently completing events. */
    struct EventReadiness
    {
        std::chrono::steady_clock::time_point first;
        std::chrono::steady_clock::time_point second;
    };

    /**
     * @brief Poll two CUDA events and retain when each one became observable.
     *
     * Recording the two timestamps independently is important here: inference
     * may outlive the transfer, but that later inference completion must not be
     * charged to the transfer's submit-to-publication latency.
     */
    EventReadiness awaitEvents(cudaEvent_t first, cudaEvent_t second)
    {
        const auto entered = std::chrono::steady_clock::now();
        bool first_ready = first == nullptr;
        bool second_ready = second == nullptr;
        EventReadiness readiness{
            .first = first_ready
                         ? entered
                         : std::chrono::steady_clock::time_point{},
            .second = second_ready
                          ? entered
                          : std::chrono::steady_clock::time_point{},
        };
        const auto deadline = std::chrono::steady_clock::now() +
                              kCompletionDeadline;
        while ((!first_ready || !second_ready) &&
               std::chrono::steady_clock::now() < deadline)
        {
            if (!first_ready)
            {
                const cudaError_t status = cudaEventQuery(first);
                if (status == cudaSuccess)
                {
                    first_ready = true;
                    readiness.first = std::chrono::steady_clock::now();
                }
                else if (status != cudaErrorNotReady)
                    requireCuda(status, "cudaEventQuery(first)");
            }
            if (!second_ready)
            {
                const cudaError_t status = cudaEventQuery(second);
                if (status == cudaSuccess)
                {
                    second_ready = true;
                    readiness.second = std::chrono::steady_clock::now();
                }
                else if (status != cudaErrorNotReady)
                    requireCuda(status, "cudaEventQuery(second)");
            }
            if (!first_ready || !second_ready)
                std::this_thread::yield();
        }
        if (!first_ready || !second_ready)
            throw std::runtime_error("CUDA expert relay perf event timed out");
        return readiness;
    }

    /** @brief Persistent allocations, streams, events, and captured workload. */
    class CUDAExpertHostRelayProgressFixture : public ::testing::Test
    {
    protected:
        /** @brief Materialize every resource before any timed observation. */
        void SetUp() override
        {
            int devices = 0;
            requireCuda(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
            if (devices < 1)
                GTEST_SKIP() << "A CUDA device is required";

            requireCuda(cudaSetDevice(0), "cudaSetDevice");
            requireCuda(
                cudaDeviceGetAttribute(
                    &multiprocessors_, cudaDevAttrMultiProcessorCount, 0),
                "cudaDeviceGetAttribute(multiprocessors)");
            requireCuda(
                cudaDeviceGetAttribute(
                    &clock_rate_khz_, cudaDevAttrClockRate, 0),
                "cudaDeviceGetAttribute(clock rate)");

            requireCuda(
                cudaStreamCreateWithFlags(
                    &inference_stream_, cudaStreamNonBlocking),
                "cudaStreamCreate(inference)");
            requireCuda(
                cudaStreamCreateWithFlags(&dma_stream_, cudaStreamNonBlocking),
                "cudaStreamCreate(dma)");
            int least_priority = 0;
            int greatest_priority = 0;
            requireCuda(
                cudaDeviceGetStreamPriorityRange(
                    &least_priority, &greatest_priority),
                "cudaDeviceGetStreamPriorityRange");
            requireCuda(
                cudaStreamCreateWithPriority(
                    &kernel_stream_,
                    cudaStreamNonBlocking,
                    greatest_priority),
                "cudaStreamCreateWithPriority(mapped kernel)");

            requireCuda(
                cudaMalloc(&device_source_, kExpertChunkBytes),
                "cudaMalloc(device source)");
            requireCuda(
                cudaHostAlloc(
                    &pressure_host_,
                    kPressureBytes,
                    cudaHostAllocMapped | cudaHostAllocPortable),
                "cudaHostAlloc(mapped pressure words)");
            void *pressure_device_alias = nullptr;
            requireCuda(
                cudaHostGetDevicePointer(
                    &pressure_device_alias, pressure_host_, 0),
                "cudaHostGetDevicePointer(mapped pressure words)");
            pressure_words_ =
                static_cast<std::uint32_t *>(pressure_device_alias);
            const std::array<DeviceId, 1> copy_devices{DeviceId::cuda(0)};
            copy_region_ = transfer_engine_.allocateMappedHostRegion(
                kExpertChunkBytes,
                copy_devices);
            mapped_host_ = copy_region_->mutableHostData();
            command_region_ = transfer_engine_.allocateMappedHostRegion(
                sizeof(CapturedCopyCommand),
                copy_devices);
            command_ = static_cast<CapturedCopyCommand *>(
                command_region_->mutableHostData());
            std::memset(command_, 0, sizeof(*command_));

            expected_.resize(kExpertChunkBytes);
            for (std::size_t index = 0; index < expected_.size(); ++index)
            {
                expected_[index] = static_cast<std::uint8_t>(
                    (index * 43u + (index >> 5u) + 0x5du) & 0xffu);
            }
            requireCuda(
                cudaMemcpy(
                    device_source_,
                    expected_.data(),
                    expected_.size(),
                    cudaMemcpyHostToDevice),
                "cudaMemcpy(setup source)");
            std::memset(pressure_host_, 0x5a, kPressureBytes);
            std::memset(mapped_host_, 0, kExpertChunkBytes);

            createEvents();
            materializeCapturedProgressBranch();
            capturePressureGraph(/*include_progress_branch=*/false);
            capturePressureGraph(/*include_progress_branch=*/true);
            captureStandaloneProgressGraph();
        }

        /** @brief Destroy only quiescent resources after every event was joined. */
        void TearDown() override
        {
            if (standalone_progress_graph_exec_)
                (void)cudaGraphExecDestroy(standalone_progress_graph_exec_);
            if (standalone_progress_graph_)
                (void)cudaGraphDestroy(standalone_progress_graph_);
            if (captured_progress_graph_exec_)
                (void)cudaGraphExecDestroy(captured_progress_graph_exec_);
            if (captured_progress_graph_)
                (void)cudaGraphDestroy(captured_progress_graph_);
            if (pressure_graph_exec_)
                (void)cudaGraphExecDestroy(pressure_graph_exec_);
            if (pressure_graph_)
                (void)cudaGraphDestroy(pressure_graph_);
            for (cudaEvent_t event : events_)
            {
                if (event)
                    (void)cudaEventDestroy(event);
            }
            if (capture_fork_)
                (void)cudaEventDestroy(capture_fork_);
            if (capture_join_)
                (void)cudaEventDestroy(capture_join_);
            if (kernel_stream_)
                (void)cudaStreamDestroy(kernel_stream_);
            if (dma_stream_)
                (void)cudaStreamDestroy(dma_stream_);
            if (inference_stream_)
                (void)cudaStreamDestroy(inference_stream_);
            if (captured_completed_blocks_)
                (void)cudaFree(captured_completed_blocks_);
            if (device_claim_)
                (void)cudaFree(device_claim_);
            command_ = nullptr;
            command_region_.reset();
            copy_region_.reset();
            mapped_host_ = nullptr;
            if (pressure_host_)
                (void)cudaFreeHost(pressure_host_);
            if (device_source_)
                (void)cudaFree(device_source_);
        }

        /** @brief Candidate transport selected for one observation. */
        enum class CopyMode : std::uint8_t
        {
            None,         ///< Baseline captured inference only.
            Dma,          ///< Ordinary asynchronous D2H copy.
            MappedKernel, ///< High-priority kernel writing mapped host pages.
            PrelaunchedGraph, ///< Retained progress graph submitted before inference.
            CapturedBranch, ///< Command handled by an inference-graph root branch.
        };

        /**
         * @brief Poll inference and one mapped completion generation independently.
         *
         * The captured branch publishes completion independently of the main
         * inference terminal, so the mapped generation is its exact edge.
         */
        EventReadiness awaitInferenceAndGeneration(
            cudaEvent_t inference,
            std::uint64_t generation)
        {
            bool inference_ready = false;
            bool command_ready = false;
            EventReadiness readiness{};
            const auto deadline = std::chrono::steady_clock::now() +
                                  kCompletionDeadline;
            while ((!inference_ready || !command_ready) &&
                   std::chrono::steady_clock::now() < deadline)
            {
                if (!inference_ready)
                {
                    const cudaError_t status = cudaEventQuery(inference);
                    if (status == cudaSuccess)
                    {
                        inference_ready = true;
                        readiness.first = std::chrono::steady_clock::now();
                    }
                    else if (status != cudaErrorNotReady)
                    {
                        requireCuda(
                            status,
                            "cudaEventQuery(captured progress inference)");
                    }
                }
                if (!command_ready &&
                    std::atomic_ref<std::uint64_t>(
                        command_->completed_generation)
                            .load(std::memory_order_acquire) == generation)
                {
                    command_ready = true;
                    readiness.second = std::chrono::steady_clock::now();
                }
                if (!inference_ready || !command_ready)
                    std::this_thread::yield();
            }
            if (!inference_ready || !command_ready)
            {
                throw std::runtime_error(
                    "CUDA captured expert relay command timed out");
            }
            return readiness;
        }

        /** @brief Publish one immutable command after all payload fields exist. */
        std::uint64_t publishCapturedCommand()
        {
            ++next_command_generation_;
            if (next_command_generation_ == 0)
                throw std::overflow_error(
                    "CUDA captured expert relay generation overflowed");
            command_->source_address =
                reinterpret_cast<std::uintptr_t>(device_source_);
            command_->destination_address = reinterpret_cast<std::uintptr_t>(
                copy_region_->deviceAlias(DeviceId::cuda(0)));
            command_->bytes = kExpertChunkBytes;
            std::atomic_thread_fence(std::memory_order_release);
            std::atomic_ref<std::uint64_t>(command_->generation).store(
                next_command_generation_, std::memory_order_release);
            return next_command_generation_;
        }

        /** @brief Run one captured epoch with the selected concurrent copy. */
        Observation observe(CopyMode mode)
        {
            std::memset(mapped_host_, 0, kExpertChunkBytes);

            std::uint64_t captured_generation = 0;
            std::chrono::steady_clock::time_point wall_start{};
            if (mode == CopyMode::CapturedBranch ||
                mode == CopyMode::PrelaunchedGraph)
            {
                wall_start = std::chrono::steady_clock::now();
                captured_generation = publishCapturedCommand();
            }

            if (mode == CopyMode::PrelaunchedGraph)
            {
                requireCuda(
                    cudaGraphLaunch(
                        standalone_progress_graph_exec_, kernel_stream_),
                    "cudaGraphLaunch(prelaunched progress epoch)");
            }

            requireCuda(
                cudaEventRecord(inference_start_, inference_stream_),
                "cudaEventRecord(inference start)");
            capturedMemoryPressureKernel<<<
                pressureBlocks(), 256, 0, inference_stream_>>>(
                pressure_words_,
                kPressureBytes / sizeof(std::uint32_t),
                0x9e3779b9u);
            requireCuda(cudaGetLastError(), "lead pressure kernel");
            requireCuda(
                cudaEventRecord(lead_ready_, inference_stream_),
                "cudaEventRecord(lead ready)");
            requireCuda(
                cudaGraphLaunch(
                    mode == CopyMode::CapturedBranch
                        ? captured_progress_graph_exec_
                        : pressure_graph_exec_,
                    inference_stream_),
                "cudaGraphLaunch(pressure epoch)");
            requireCuda(
                cudaEventRecord(inference_stop_, inference_stream_),
                "cudaEventRecord(inference stop)");

            /*
             * Join only the lead event so the retained graph is already live
             * when the transfer is submitted. This setup wait is outside the
             * measured copy interval and never enters production inference.
             */
            awaitEvents(lead_ready_, nullptr);

            cudaStream_t copy_stream = nullptr;
            if (mode != CopyMode::CapturedBranch &&
                mode != CopyMode::PrelaunchedGraph)
                wall_start = std::chrono::steady_clock::now();
            if (mode == CopyMode::Dma)
            {
                copy_stream = dma_stream_;
                requireCuda(
                    cudaEventRecord(copy_start_, copy_stream),
                    "cudaEventRecord(DMA start)");
                requireCuda(
                    cudaMemcpyAsync(
                        mapped_host_,
                        device_source_,
                        kExpertChunkBytes,
                        cudaMemcpyDeviceToHost,
                        copy_stream),
                    "cudaMemcpyAsync(D2H candidate)");
            }
            else if (mode == CopyMode::MappedKernel)
            {
                copy_stream = kernel_stream_;
                requireCuda(
                    cudaEventRecord(copy_start_, copy_stream),
                    "cudaEventRecord(mapped kernel start)");
                transfer_engine_.
                    enqueuePersistentDeviceRegionToMappedHostByKernel(
                        device_source_,
                        kExpertChunkBytes,
                        0u,
                        *copy_region_,
                        0u,
                        kExpertChunkBytes,
                        DeviceId::cuda(0),
                        copy_stream);
            }

            if (copy_stream)
            {
                requireCuda(
                    cudaEventRecord(copy_stop_, copy_stream),
                    "cudaEventRecord(copy stop)");
            }
            const EventReadiness readiness =
                mode == CopyMode::CapturedBranch ||
                        mode == CopyMode::PrelaunchedGraph
                    ? awaitInferenceAndGeneration(
                          inference_stop_, captured_generation)
                    : awaitEvents(
                          inference_stop_,
                          copy_stream ? copy_stop_ : nullptr);

            float inference_ms = 0.0F;
            requireCuda(
                cudaEventElapsedTime(
                    &inference_ms, inference_start_, inference_stop_),
                "cudaEventElapsedTime(inference)");
            float copy_ms = 0.0F;
            if (copy_stream || mode == CopyMode::CapturedBranch ||
                mode == CopyMode::PrelaunchedGraph)
            {
                if (copy_stream)
                {
                    requireCuda(
                        cudaEventElapsedTime(
                            &copy_ms, copy_start_, copy_stop_),
                        "cudaEventElapsedTime(copy)");
                }
                if (std::memcmp(
                        mapped_host_, expected_.data(), expected_.size()) != 0)
                {
                    throw std::runtime_error(
                        "CUDA expert host-relay candidate changed payload bytes");
                }
            }

            return {
                .inference_ms = static_cast<double>(inference_ms),
                .copy_device_ms = static_cast<double>(copy_ms),
                .copy_wall_ms =
                    copy_stream || mode == CopyMode::CapturedBranch ||
                            mode == CopyMode::PrelaunchedGraph
                                    ? std::chrono::duration<double, std::milli>(
                                          readiness.second - wall_start).count()
                                    : 0.0,
            };
        }

        /** @brief Run warmup and return measured observations only. */
        std::vector<Observation> measure(CopyMode mode)
        {
            for (int sample = 0; sample < kWarmupSamples; ++sample)
                (void)observe(mode);
            std::vector<Observation> observations;
            observations.reserve(kMeasuredSamples);
            for (int sample = 0; sample < kMeasuredSamples; ++sample)
                observations.push_back(observe(mode));
            return observations;
        }

        /** @brief Return a device-populating grid without hardcoded SM count. */
        int pressureBlocks() const noexcept
        {
            return std::max(1, multiprocessors_ * 8);
        }

        /** @return Captured progress-branch block count for this process. */
        int capturedProgressBlocks() const noexcept
        {
            return captured_progress_blocks_;
        }

        /** @brief Materialize fixed counter storage before graph capture. */
        void materializeCapturedProgressBranch()
        {
            int progress_blocks = kDefaultCapturedProgressBlocks;
            if (const char *configured = std::getenv(
                "LLAMINAR_CUDA_EXPERT_RELAY_CAPTURED_PROGRESS_BLOCKS"))
            {
                progress_blocks = std::atoi(configured);
            }
            if (progress_blocks <= 0 || progress_blocks > multiprocessors_)
            {
                throw std::invalid_argument(
                    "CUDA captured relay progress blocks must be in [1, SM count]");
            }
            captured_progress_blocks_ = progress_blocks;
            requireCuda(
                cudaMalloc(
                    &captured_completed_blocks_,
                    sizeof(*captured_completed_blocks_)),
                "cudaMalloc(captured progress counter)");
            requireCuda(
                cudaMalloc(&device_claim_, sizeof(*device_claim_)),
                "cudaMalloc(captured command claim)");
            requireCuda(
                cudaMemsetAsync(
                    captured_completed_blocks_,
                    0,
                    sizeof(*captured_completed_blocks_),
                    kernel_stream_),
                "cudaMemsetAsync(captured progress counter)");
            requireCuda(
                cudaMemsetAsync(
                    device_claim_,
                    0,
                    sizeof(*device_claim_),
                    kernel_stream_),
                "cudaMemsetAsync(captured command claim)");
            requireCuda(
                cudaEventRecord(copy_stop_, kernel_stream_),
                "cudaEventRecord(captured progress setup)");
            (void)awaitEvents(copy_stop_, nullptr);
        }

    private:
        /** @brief Create reusable timing and readiness events. */
        void createEvents()
        {
            for (cudaEvent_t &event : events_)
                requireCuda(cudaEventCreate(&event), "cudaEventCreate");
            inference_start_ = events_[0];
            lead_ready_ = events_[1];
            inference_stop_ = events_[2];
            copy_start_ = events_[3];
            copy_stop_ = events_[4];
            requireCuda(
                cudaEventCreateWithFlags(
                    &capture_fork_, cudaEventDisableTiming),
                "cudaEventCreate(capture fork)");
            requireCuda(
                cudaEventCreateWithFlags(
                    &capture_join_, cudaEventDisableTiming),
                "cudaEventCreate(capture join)");
        }

        /**
         * @brief Capture one pressure epoch with or without its parallel branch.
         * @param include_progress_branch Whether this graph owns the transfer fork.
         */
        void capturePressureGraph(bool include_progress_branch)
        {
            cudaGraph_t *const graph = include_progress_branch
                                           ? &captured_progress_graph_
                                           : &pressure_graph_;
            cudaGraphExec_t *const graph_exec =
                include_progress_branch
                    ? &captured_progress_graph_exec_
                    : &pressure_graph_exec_;
            requireCuda(
                cudaStreamBeginCapture(
                    inference_stream_, cudaStreamCaptureModeGlobal),
                "cudaStreamBeginCapture");
            if (include_progress_branch)
            {
                auto *const device_command =
                    static_cast<CapturedCopyCommand *>(
                        command_region_->deviceAlias(DeviceId::cuda(0)));
                capturedCopyCommandClaimKernel<<<1, 1, 0, inference_stream_>>>(
                    device_command, device_claim_);
                requireCuda(
                    cudaGetLastError(), "capturedCopyCommandClaimKernel");
                requireCuda(
                    cudaEventRecord(capture_fork_, inference_stream_),
                    "cudaEventRecord(captured progress fork)");
                requireCuda(
                    cudaStreamWaitEvent(kernel_stream_, capture_fork_, 0),
                    "cudaStreamWaitEvent(captured progress fork)");
                capturedMappedHostCopyBranchKernel<<<
                    captured_progress_blocks_, 256, 0, kernel_stream_>>>(
                    device_claim_,
                    device_command,
                    captured_completed_blocks_);
                requireCuda(
                    cudaGetLastError(),
                    "capturedMappedHostCopyBranchKernel");
            }
            capturedMappedTicketWaitKernel<<<
                pressureBlocks(), 256, 0, inference_stream_>>>(
                pressure_words_,
                static_cast<unsigned long long>(clock_rate_khz_) *
                    kCapturedWaitMilliseconds);
            for (int node = 0; node < kPressureGraphNodes; ++node)
            {
                capturedMemoryPressureKernel<<<
                    pressureBlocks(), 256, 0, inference_stream_>>>(
                    pressure_words_,
                    kPressureBytes / sizeof(std::uint32_t),
                    0x85ebca6bu + static_cast<std::uint32_t>(node));
            }
            requireCuda(cudaGetLastError(), "captured pressure kernels");
            if (include_progress_branch)
            {
                requireCuda(
                    cudaEventRecord(capture_join_, kernel_stream_),
                    "cudaEventRecord(captured progress join)");
                requireCuda(
                    cudaStreamWaitEvent(inference_stream_, capture_join_, 0),
                    "cudaStreamWaitEvent(captured progress join)");
            }
            requireCuda(
                cudaStreamEndCapture(inference_stream_, graph),
                "cudaStreamEndCapture");
            if (include_progress_branch)
            {
                requireCuda(
                    cudaGraphInstantiateWithFlags(
                        graph_exec,
                        *graph,
                        cudaGraphInstantiateFlagUseNodePriority),
                    "cudaGraphInstantiateWithFlags(node priority)");
            }
            else
            {
                requireCuda(
                    cudaGraphInstantiate(graph_exec, *graph, nullptr, nullptr, 0),
                    "cudaGraphInstantiate(pressure only)");
            }
        }

        /** @brief Capture the finite transfer epoch replayed before inference. */
        void captureStandaloneProgressGraph()
        {
            auto *const device_command =
                static_cast<CapturedCopyCommand *>(
                    command_region_->deviceAlias(DeviceId::cuda(0)));
            requireCuda(
                cudaStreamBeginCapture(
                    kernel_stream_, cudaStreamCaptureModeGlobal),
                "cudaStreamBeginCapture(standalone progress)");
            capturedCopyCommandClaimKernel<<<1, 1, 0, kernel_stream_>>>(
                device_command, device_claim_);
            capturedMappedHostCopyBranchKernel<<<
                captured_progress_blocks_, 256, 0, kernel_stream_>>>(
                device_claim_, device_command, captured_completed_blocks_);
            requireCuda(
                cudaGetLastError(),
                "captured standalone progress kernels");
            requireCuda(
                cudaStreamEndCapture(
                    kernel_stream_, &standalone_progress_graph_),
                "cudaStreamEndCapture(standalone progress)");
            requireCuda(
                cudaGraphInstantiate(
                    &standalone_progress_graph_exec_,
                    standalone_progress_graph_,
                    nullptr,
                    nullptr,
                    0),
                "cudaGraphInstantiate(standalone progress)");
        }

        int multiprocessors_ = 0;
        int clock_rate_khz_ = 0;
        int captured_progress_blocks_ = 0;
        std::uint64_t next_command_generation_ = 0;
        TransferEngine transfer_engine_;
        std::shared_ptr<MappedHostTransferRegion> copy_region_;
        std::shared_ptr<MappedHostTransferRegion> command_region_;
        CapturedCopyCommand *command_ = nullptr;
        CapturedCopyClaim *device_claim_ = nullptr;
        unsigned int *captured_completed_blocks_ = nullptr;
        void *device_source_ = nullptr;
        void *pressure_host_ = nullptr;
        std::uint32_t *pressure_words_ = nullptr;
        void *mapped_host_ = nullptr;
        std::vector<std::uint8_t> expected_;
        cudaStream_t inference_stream_ = nullptr;
        cudaStream_t dma_stream_ = nullptr;
        cudaStream_t kernel_stream_ = nullptr;
        cudaGraph_t pressure_graph_ = nullptr;
        cudaGraphExec_t pressure_graph_exec_ = nullptr;
        cudaGraph_t captured_progress_graph_ = nullptr;
        cudaGraphExec_t captured_progress_graph_exec_ = nullptr;
        cudaGraph_t standalone_progress_graph_ = nullptr;
        cudaGraphExec_t standalone_progress_graph_exec_ = nullptr;
        std::array<cudaEvent_t, 5> events_{};
        cudaEvent_t inference_start_ = nullptr;
        cudaEvent_t lead_ready_ = nullptr;
        cudaEvent_t inference_stop_ = nullptr;
        cudaEvent_t copy_start_ = nullptr;
        cudaEvent_t copy_stop_ = nullptr;
        cudaEvent_t capture_fork_ = nullptr;
        cudaEvent_t capture_join_ = nullptr;
    };

    TEST_F(
        CUDAExpertHostRelayProgressFixture,
        RetainedProgressEpochsMakeBoundedByteExactProgressWithoutInterference)
    {
        if (std::getenv("LLAMINAR_CUDA_EXPERT_RELAY_PROFILE_ONLY"))
        {
            const Observation profile = observe(CopyMode::CapturedBranch);
            EXPECT_GT(profile.copy_wall_ms, 0.0);
            return;
        }

        const auto baseline = measure(CopyMode::None);
        const auto dma = measure(CopyMode::Dma);
        const auto mapped = measure(CopyMode::MappedKernel);
        const auto prelaunched = measure(CopyMode::PrelaunchedGraph);
        const auto captured = measure(CopyMode::CapturedBranch);
        const auto collect = [](
                                 const std::vector<Observation> &observations,
                                 auto member)
        {
            std::vector<double> values;
            values.reserve(observations.size());
            for (const Observation &observation : observations)
                values.push_back(observation.*member);
            return median(std::move(values));
        };

        const double baseline_inference_ms =
            collect(baseline, &Observation::inference_ms);
        const double dma_inference_ms =
            collect(dma, &Observation::inference_ms);
        const double dma_wall_ms = collect(dma, &Observation::copy_wall_ms);
        const double dma_device_ms =
            collect(dma, &Observation::copy_device_ms);
        const double mapped_inference_ms =
            collect(mapped, &Observation::inference_ms);
        const double mapped_wall_ms =
            collect(mapped, &Observation::copy_wall_ms);
        const double mapped_device_ms =
            collect(mapped, &Observation::copy_device_ms);
        const double mapped_interference_ms =
            std::max(0.0, mapped_inference_ms - baseline_inference_ms);
        const double prelaunched_inference_ms =
            collect(prelaunched, &Observation::inference_ms);
        const double prelaunched_wall_ms =
            collect(prelaunched, &Observation::copy_wall_ms);
        const double prelaunched_interference_ms =
            std::max(0.0, prelaunched_inference_ms - baseline_inference_ms);
        const double captured_inference_ms =
            collect(captured, &Observation::inference_ms);
        const double captured_wall_ms =
            collect(captured, &Observation::copy_wall_ms);
        const double captured_interference_ms =
            std::max(0.0, captured_inference_ms - baseline_inference_ms);

        std::cout << std::fixed << std::setprecision(3)
                  << "CUDA expert host-relay progress certificate"
                  << " bytes=" << kExpertChunkBytes
                  << " graph_nodes=" << kPressureGraphNodes
                  << " mapped_wait_ms=" << kCapturedWaitMilliseconds
                  << " captured_progress_blocks="
                  << capturedProgressBlocks() << '\n'
                  << "  baseline inference_ms=" << baseline_inference_ms
                  << '\n'
                  << "  dma inference_ms=" << dma_inference_ms
                  << " copy_wall_ms=" << dma_wall_ms
                  << " copy_device_ms=" << dma_device_ms << '\n'
                  << "  mapped_kernel inference_ms=" << mapped_inference_ms
                  << " interference_ms=" << mapped_interference_ms
                  << " copy_wall_ms=" << mapped_wall_ms
                  << " copy_device_ms=" << mapped_device_ms << '\n'
                  << "  prelaunched_progress inference_ms="
                  << prelaunched_inference_ms
                  << " interference_ms=" << prelaunched_interference_ms
                  << " copy_wall_ms=" << prelaunched_wall_ms << '\n'
                  << "  captured_progress inference_ms="
                  << captured_inference_ms
                  << " interference_ms=" << captured_interference_ms
                  << " copy_wall_ms=" << captured_wall_ms << '\n';

        EXPECT_GT(dma_wall_ms, 0.0);
        EXPECT_GT(mapped_wall_ms, 0.0);
        EXPECT_GT(prelaunched_wall_ms, 0.0);
        EXPECT_GT(captured_wall_ms, 0.0);
        EXPECT_LE(prelaunched_wall_ms, 10.0)
            << "A prelaunched retained progress epoch must publish one expert chunk promptly";
        EXPECT_LE(prelaunched_wall_ms, mapped_wall_ms * 0.25)
            << "Prelaunch must eliminate submission behind an already-live mapped wait";
        EXPECT_LE(
            prelaunched_interference_ms,
            std::max(5.0, baseline_inference_ms * 0.05))
            << "Prelaunched progress must not impose uneconomical inference interference";
        EXPECT_LE(captured_wall_ms, 10.0)
            << "A captured progress branch must publish one expert chunk promptly";
        EXPECT_LE(captured_wall_ms, mapped_wall_ms * 0.25)
            << "A graph-owned branch must eliminate launch starvation behind a mapped wait";
        EXPECT_LE(
            captured_interference_ms,
            std::max(5.0, baseline_inference_ms * 0.05))
            << "Captured progress must not impose uneconomical inference interference";
    }
} // namespace
} // namespace llaminar2

#endif // HAVE_CUDA
