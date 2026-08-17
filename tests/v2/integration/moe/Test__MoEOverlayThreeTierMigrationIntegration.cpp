/**
 * @file Test__MoEOverlayThreeTierMigrationIntegration.cpp
 * @brief Real CUDA/ROCm/CPU residency rotation certificates.
 *
 * A frozen decode histogram rotates three experts through three compute tiers:
 * CPU->CUDA promotion, CUDA->ROCm packed-blob transfer, and ROCm->CPU demotion.
 * Gate, up, and down projections use their production dimensional orientation,
 * so one publication wave owns nine independently event-polled operations. The
 * test keeps an old epoch ticket live, proves unrelated work completes on both
 * GPUs while migration remains pending, publishes only after every byte is
 * ready, and checks PerfStats for the actual physical movement. A symmetric
 * CUDA-hot/CPU-cold case additionally swaps complete experts in both directions.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/ExpertTierMigrationOperations.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "utils/DebugEnv.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kProjectionCount = 3;

        /** @brief Stable geometry and semantic identity for one expert matrix. */
        struct ProjectionSpec
        {
            ExpertTierWeightProjection projection;
            int n;
            int k;
            const char *name;
        };

        constexpr std::array<ProjectionSpec, kProjectionCount> kProjections{{
            {ExpertTierWeightProjection::Gate, 192, 256, "gate"},
            {ExpertTierWeightProjection::Up, 192, 256, "up"},
            {ExpertTierWeightProjection::Down, 256, 192, "down"},
        }};

        /** @brief Enable movement evidence and restore process configuration. */
        class ScopedThreeTierPerfStats final
        {
        public:
            /** @brief Enable PerfStats and clear records before the certificate. */
            ScopedThreeTierPerfStats()
            {
                if (const char *value =
                        std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))
                {
                    had_value_ = true;
                    value_ = value;
                }
                setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            /** @brief Restore the caller's setting and discard scoped records. */
            ~ScopedThreeTierPerfStats()
            {
                if (had_value_)
                    setenv(
                        "LLAMINAR_PERF_STATS_SUMMARY",
                        value_.c_str(),
                        1);
                else
                    unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
                mutableDebugEnv().reload();
                PerfStatsCollector::reset();
            }

            ScopedThreeTierPerfStats(const ScopedThreeTierPerfStats &) = delete;
            ScopedThreeTierPerfStats &operator=(
                const ScopedThreeTierPerfStats &) = delete;

        private:
            bool had_value_ = false;
            std::string value_;
        };

        /** @brief Own one separated asymmetric Q4_K projection on one GPU. */
        class DevicePackedProjection final
        {
        public:
            /**
             * @brief Allocate every non-empty descriptor region.
             * @param backend Exact CUDA or ROCm allocation authority.
             * @param device GPU endpoint owning the allocation.
             * @param shape Host projection defining exact region capacities.
             */
            DevicePackedProjection(
                IBackend &backend,
                DeviceId device,
                const HostGpuExpertPackedProjection &shape)
                : backend_(backend),
                  device_(device),
                  n_(shape.N),
                  k_(shape.K),
                  blocks_per_row_(shape.blocks_per_row),
                  payload_bytes_per_block_(shape.payload_bytes_per_block),
                  payload_bytes_(shape.payload.size()),
                  scales_bytes_(shape.scales.size() * sizeof(std::uint16_t)),
                  mins_bytes_(shape.mins.size() * sizeof(std::uint16_t))
            {
                const int ordinal = device_.gpu_ordinal();
                payload_ = static_cast<std::uint8_t *>(
                    backend_.allocate(payload_bytes_, ordinal));
                scales_ = static_cast<std::uint16_t *>(
                    backend_.allocate(scales_bytes_, ordinal));
                mins_ = static_cast<std::uint16_t *>(
                    backend_.allocate(mins_bytes_, ordinal));
                if (!payload_ || !scales_ || !mins_)
                    throw std::runtime_error(
                        "Could not allocate a three-tier packed projection");
            }

            /** @brief Release all regions through their exact backend. */
            ~DevicePackedProjection()
            {
                const int ordinal = device_.gpu_ordinal();
                if (payload_)
                    backend_.free(payload_, ordinal);
                if (scales_)
                    backend_.free(scales_, ordinal);
                if (mins_)
                    backend_.free(mins_, ordinal);
            }

            DevicePackedProjection(const DevicePackedProjection &) = delete;
            DevicePackedProjection &operator=(
                const DevicePackedProjection &) = delete;

            /** @brief Enqueue a complete host projection on an explicit stream. */
            [[nodiscard]] bool upload(
                const HostGpuExpertPackedProjection &source,
                void *stream) noexcept
            {
                if (!stream || source.payload.size() != payload_bytes_ ||
                    source.scales.size() * sizeof(std::uint16_t) !=
                        scales_bytes_ ||
                    source.mins.size() * sizeof(std::uint16_t) != mins_bytes_)
                {
                    return false;
                }
                const int ordinal = device_.gpu_ordinal();
                return backend_.hostToDeviceOnStream(
                           payload_,
                           source.payload.data(),
                           payload_bytes_,
                           ordinal,
                           stream) &&
                       backend_.hostToDeviceOnStream(
                           scales_,
                           source.scales.data(),
                           scales_bytes_,
                           ordinal,
                           stream) &&
                       backend_.hostToDeviceOnStream(
                           mins_,
                           source.mins.data(),
                           mins_bytes_,
                           ordinal,
                           stream);
            }

            /** @brief Describe immutable arrays for GPU-to-CPU conversion. */
            [[nodiscard]] ExpertTierGpuConstProjectionView constView()
                const noexcept
            {
                return {
                    .payload = payload_,
                    .scales = scales_,
                    .mins = mins_,
                    .emins = nullptr,
                    .payload_bytes = payload_bytes_,
                    .scales_bytes = scales_bytes_,
                    .mins_bytes = mins_bytes_,
                    .emins_bytes = 0,
                };
            }

            /** @brief Describe writable arrays for CPU-to-GPU conversion. */
            [[nodiscard]] ExpertTierGpuMutableProjectionView mutableView()
                noexcept
            {
                return {
                    .payload = payload_,
                    .scales = scales_,
                    .mins = mins_,
                    .emins = nullptr,
                    .payload_bytes = payload_bytes_,
                    .scales_bytes = scales_bytes_,
                    .mins_bytes = mins_bytes_,
                    .emins_bytes = 0,
                };
            }

            /** @brief Describe the common CUDA/ROCm packed blob. */
            [[nodiscard]] GpuExpertPackedDescriptor descriptor() const noexcept
            {
                return {
                    .ptrs = {
                        .d_vnni = payload_,
                        .d_scales = scales_,
                        .d_mins = mins_,
                    },
                    .n = n_,
                    .k = k_,
                    .blocks_per_row = blocks_per_row_,
                    .codebook_id = 5,
                    .payload_bytes_per_block = payload_bytes_per_block_,
                    .is_asymmetric = true,
                    .has_emins = false,
                    .vnni_bytes = payload_bytes_,
                    .scales_bytes = scales_bytes_,
                    .mins_bytes = mins_bytes_,
                    .emins_bytes = 0,
                };
            }

            /**
             * @brief Compare all destination bytes at a diagnostic-only edge.
             * @param expected CPU-owned exact oracle.
             * @param stream Explicit stream used by the test observation copy.
             * @param error Receives the first mismatched separated region.
             * @return Whether payload, scales, and minima all match exactly.
             */
            [[nodiscard]] bool matches(
                const HostGpuExpertPackedProjection &expected,
                void *stream,
                std::string *error) noexcept
            {
                std::vector<std::uint8_t> payload(payload_bytes_);
                std::vector<std::uint16_t> scales(
                    scales_bytes_ / sizeof(std::uint16_t));
                std::vector<std::uint16_t> mins(
                    mins_bytes_ / sizeof(std::uint16_t));
                const int ordinal = device_.gpu_ordinal();
                if (!backend_.deviceToHost(
                        payload.data(), payload_, payload_bytes_, ordinal, stream) ||
                    !backend_.deviceToHost(
                        scales.data(), scales_, scales_bytes_, ordinal, stream) ||
                    !backend_.deviceToHost(
                        mins.data(), mins_, mins_bytes_, ordinal, stream))
                {
                    if (error)
                        *error = "Diagnostic destination download failed";
                    return false;
                }
                if (payload != expected.payload)
                {
                    if (error)
                        *error = "Packed payload bytes differ";
                    return false;
                }
                if (scales != expected.scales)
                {
                    if (error)
                        *error = "Packed scale bytes differ";
                    return false;
                }
                if (mins != expected.mins)
                {
                    if (error)
                        *error = "Packed minimum bytes differ";
                    return false;
                }
                return true;
            }

        private:
            IBackend &backend_;
            DeviceId device_;
            int n_ = 0;
            int k_ = 0;
            std::uint32_t blocks_per_row_ = 0;
            std::uint8_t payload_bytes_per_block_ = 0;
            std::size_t payload_bytes_ = 0;
            std::size_t scales_bytes_ = 0;
            std::size_t mins_bytes_ = 0;
            std::uint8_t *payload_ = nullptr;
            std::uint16_t *scales_ = nullptr;
            std::uint16_t *mins_ = nullptr;
        };

        /** @brief Own independent device work used as a no-wait witness. */
        class DeviceCopyWitness final
        {
        public:
            /** @brief Allocate a source/destination pair before inference starts. */
            DeviceCopyWitness(IBackend &backend, DeviceId device)
                : backend_(backend), device_(device)
            {
                source_ = backend_.allocate(kBytes, device_.gpu_ordinal());
                destination_ = backend_.allocate(kBytes, device_.gpu_ordinal());
                if (!source_ || !destination_)
                    throw std::runtime_error(
                        "Could not allocate a three-tier inference witness");
            }

            /** @brief Release witness storage after the completion event is ready. */
            ~DeviceCopyWitness()
            {
                const int ordinal = device_.gpu_ordinal();
                if (source_)
                    backend_.free(source_, ordinal);
                if (destination_)
                    backend_.free(destination_, ordinal);
            }

            DeviceCopyWitness(const DeviceCopyWitness &) = delete;
            DeviceCopyWitness &operator=(const DeviceCopyWitness &) = delete;

            /** @brief Enqueue independent device work on an exact hot-path stream. */
            [[nodiscard]] bool enqueue(void *stream) noexcept
            {
                return stream && backend_.deviceCopyAsync(
                                     destination_,
                                     source_,
                                     kBytes,
                                     device_.gpu_ordinal(),
                                     stream);
            }

        private:
            static constexpr std::size_t kBytes = 8 * 1024 * 1024;
            IBackend &backend_;
            DeviceId device_;
            void *source_ = nullptr;
            void *destination_ = nullptr;
        };

        /** @brief Build deterministic asymmetric Q4_K bytes for one projection. */
        HostGpuExpertPackedProjection makeProjection(
            const ProjectionSpec &spec,
            std::uint32_t seed)
        {
            HostGpuExpertPackedProjection projection;
            projection.N = spec.n;
            projection.K = spec.k;
            projection.blocks_per_row =
                static_cast<std::uint32_t>(spec.k / 32);
            projection.source_codebook_id = 5;
            projection.codebook_id = 5;
            projection.payload_bytes_per_block = 16;
            projection.is_asymmetric = true;
            projection.is_superblock = true;
            projection.has_emins = false;

            const std::size_t blocks =
                static_cast<std::size_t>(spec.n) *
                projection.blocks_per_row;
            projection.payload.resize(blocks * 16);
            projection.scales.resize(blocks);
            projection.mins.resize(blocks);
            for (std::size_t block = 0; block < blocks; ++block)
            {
                for (std::size_t byte = 0; byte < 16; ++byte)
                {
                    projection.payload[block * 16 + byte] =
                        static_cast<std::uint8_t>(
                            (seed + block * 37u + byte * 19u) & 0xffu);
                }
                projection.scales[block] = static_cast<std::uint16_t>(
                    0x2800u + ((seed + block * 11u) & 0x03ffu));
                projection.mins[block] = static_cast<std::uint16_t>(
                    0xa800u + ((seed + block * 7u) & 0x03ffu));
            }
            return projection;
        }

        /** @brief Shared bank observations retained after retirement. */
        struct PhysicalBankObservations
        {
            std::size_t commits_started = 0;
            std::size_t commit_polls = 0;
            std::size_t aborts = 0;
            std::size_t retires = 0;
        };

        /** @brief Minimal inactive-bank transaction around real transferred data. */
        class PhysicalInactiveBank final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /** @brief Retain evidence independently of the bank object. */
            explicit PhysicalInactiveBank(
                std::shared_ptr<PhysicalBankObservations> observations)
                : observations_(std::move(observations))
            {
            }

            /** @brief Accept commit only after the composite transfer barrier. */
            bool beginCommit(std::string *error) noexcept override
            {
                if (aborted_ || begun_)
                {
                    if (error)
                        *error = "Physical inactive bank has invalid commit lifecycle";
                    return false;
                }
                begun_ = true;
                ++observations_->commits_started;
                return true;
            }

            /** @brief Preserve one asynchronous maintenance poll before Ready. */
            MoEOverlayResidencyWaveProgress pollCommit(
                std::string *error) noexcept override
            {
                ++observations_->commit_polls;
                if (!begun_ || aborted_)
                {
                    if (error)
                        *error = "Physical inactive bank commit was not active";
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (!pending_observed_)
                {
                    pending_observed_ = true;
                    return MoEOverlayResidencyWaveProgress::Pending;
                }
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Mark every unpublished destination reservation discarded. */
            void abort() noexcept override
            {
                if (!aborted_)
                {
                    aborted_ = true;
                    ++observations_->aborts;
                }
            }

            /** @brief This host-only bank has no additional abort event edge. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Record old-bank retirement after the ticket lease drains. */
            void retirePrevious() noexcept override
            {
                ++observations_->retires;
            }

        private:
            std::shared_ptr<PhysicalBankObservations> observations_;
            bool begun_ = false;
            bool pending_observed_ = false;
            bool aborted_ = false;
        };

        /** @brief Physical storage and lane for one expert/projection edge. */
        struct PhysicalProjectionFlow
        {
            HostGpuExpertPackedProjection expected_gpu;
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            ExpertTierWeightDeviceLayout gpu_to_cpu_layout;
            ExpertTierWeightDeviceLayout cpu_to_gpu_layout;
            std::unique_ptr<DevicePackedProjection> gpu_source;
            std::unique_ptr<DevicePackedProjection> gpu_destination;
            std::vector<std::uint8_t> cpu_destination;
            std::shared_ptr<ExpertTierWeightTransferLane> cpu_edge_lane;
            std::shared_ptr<ExpertTierGpuBlobTransferLane> blob_lane;
        };

        /**
         * @brief Pre-materialized physical factory for one three-tier rotation.
         *
         * Expert 0 starts on CUDA and moves to ROCm, expert 1 starts on ROCm
         * and moves to CPU, and expert 2 starts on CPU and moves to CUDA. All
         * buffers, streams, events, and lanes exist before `prepare()` is called.
         */
        class RealThreeTierWaveFactory final
            : public IMoEOverlayTierPreparedWaveFactory
        {
        public:
            /** @brief Bind exact backend authorities without allocating yet. */
            RealThreeTierWaveFactory(IBackend &cuda, IBackend &rocm)
                : cuda_(cuda), rocm_(rocm),
                  bank_observations(
                      std::make_shared<PhysicalBankObservations>())
            {
            }

            /** @brief Destroy source-ready events after every lane is quiescent. */
            ~RealThreeTierWaveFactory()
            {
                if (cuda_source_ready_)
                    cuda_.destroyEvent(cuda_source_ready_, 0);
                if (rocm_source_ready_)
                    rocm_.destroyEvent(rocm_source_ready_, 0);
            }

            RealThreeTierWaveFactory(const RealThreeTierWaveFactory &) = delete;
            RealThreeTierWaveFactory &operator=(
                const RealThreeTierWaveFactory &) = delete;

            /**
             * @brief Allocate/upload every source, destination, and persistent lane.
             * @param error Receives the first topology/materialization failure.
             * @return Whether the factory is ready before ticket admission.
             */
            [[nodiscard]] bool initialize(std::string *error) noexcept
            {
                try
                {
                    auto &cuda_context =
                        GPUDeviceContextPool::instance().getContext(
                            DeviceId::cuda(0));
                    auto &rocm_context =
                        GPUDeviceContextPool::instance().getContext(
                            DeviceId::rocm(0));
                    cuda_producer_stream_ =
                        cuda_context.getOrCreateAuxiliaryStream(
                            "three_tier_rotation:cuda_source_producer");
                    rocm_producer_stream_ =
                        rocm_context.getOrCreateAuxiliaryStream(
                            "three_tier_rotation:rocm_source_producer");
                    cuda_observation_stream_ =
                        cuda_context.getOrCreateAuxiliaryStream(
                            "three_tier_rotation:cuda_observation");
                    rocm_observation_stream_ =
                        rocm_context.getOrCreateAuxiliaryStream(
                            "three_tier_rotation:rocm_observation");
                    if (!cuda_producer_stream_ || !rocm_producer_stream_ ||
                        !cuda_observation_stream_ || !rocm_observation_stream_)
                    {
                        throw std::runtime_error(
                            "Could not create all explicit three-tier streams");
                    }

                    const auto *source_format =
                        native_vnni_formats::forSourceIdentity(5, true);
                    if (!source_format)
                        throw std::runtime_error(
                            "Q4_K source format is absent from NativeVNNI catalog");

                    for (int expert = 0; expert < 3; ++expert)
                    {
                        for (std::size_t projection_index = 0;
                             projection_index < kProjectionCount;
                             ++projection_index)
                        {
                            const auto &spec = kProjections[projection_index];
                            auto flow = std::make_unique<PhysicalProjectionFlow>();
                            flow->expected_gpu = makeProjection(
                                spec,
                                static_cast<std::uint32_t>(
                                    41000 + expert * 100 + projection_index));
                            std::string conversion_error;
                            if (!flow->expected_gpu.valid(&conversion_error) ||
                                !gpuToCpuExpertPackedReference(
                                    flow->expected_gpu,
                                    flow->expected_cpu,
                                    &conversion_error))
                            {
                                throw std::runtime_error(conversion_error);
                            }

                            flow->gpu_to_cpu_layout =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    spec.n,
                                    spec.k,
                                    2,
                                    0,
                                    expert,
                                    spec.projection,
                                    1)
                                    .deviceLayout();
                            flow->cpu_to_gpu_layout =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    flow->expected_cpu,
                                    2,
                                    0,
                                    expert,
                                    spec.projection,
                                    1)
                                    .deviceLayout();

                            const std::string identity =
                                "expert" + std::to_string(expert) + ":" +
                                spec.name;
                            if (expert == 0)
                            {
                                flow->gpu_source =
                                    std::make_unique<DevicePackedProjection>(
                                        cuda_,
                                        DeviceId::cuda(0),
                                        flow->expected_gpu);
                                flow->gpu_destination =
                                    std::make_unique<DevicePackedProjection>(
                                        rocm_,
                                        DeviceId::rocm(0),
                                        flow->expected_gpu);
                                if (!flow->gpu_source->upload(
                                        flow->expected_gpu,
                                        cuda_producer_stream_))
                                {
                                    throw std::runtime_error(
                                        "CUDA source upload failed for " + identity);
                                }
                                flow->blob_lane = std::make_shared<
                                    ExpertTierGpuBlobTransferLane>(
                                    ExpertTierGpuBlobTransferLane::Config{
                                        .source_device = DeviceId::cuda(0),
                                        .destination_device = DeviceId::rocm(0),
                                        .staging_capacity_bytes = 1024,
                                        .lane_name = "three_tier:" + identity,
                                        .perf_device = "cuda-hot/rocm-warm/cpu-cold",
                                        .collect_timing_measurements = true,
                                    });
                                if (!flow->blob_lane->materialize(
                                        &conversion_error))
                                    throw std::runtime_error(conversion_error);
                            }
                            else if (expert == 1)
                            {
                                flow->gpu_source =
                                    std::make_unique<DevicePackedProjection>(
                                        rocm_,
                                        DeviceId::rocm(0),
                                        flow->expected_gpu);
                                if (!flow->gpu_source->upload(
                                        flow->expected_gpu,
                                        rocm_producer_stream_))
                                {
                                    throw std::runtime_error(
                                        "ROCm source upload failed for " + identity);
                                }
                                flow->cpu_destination.resize(
                                    flow->expected_cpu.native_interleaved.size());
                                flow->cpu_edge_lane = std::make_shared<
                                    ExpertTierWeightTransferLane>(
                                    ExpertTierWeightTransferLane::Config{
                                        .device = DeviceId::rocm(0),
                                        .staging_capacity_bytes =
                                            flow->gpu_to_cpu_layout.chunkBytes(1),
                                        .lane_name = "three_tier:" + identity,
                                        .perf_device = "cuda-hot/rocm-warm/cpu-cold",
                                        .collect_timing_measurements = true,
                                    });
                                if (!flow->cpu_edge_lane->materialize(
                                        &conversion_error))
                                    throw std::runtime_error(conversion_error);
                            }
                            else
                            {
                                flow->gpu_destination =
                                    std::make_unique<DevicePackedProjection>(
                                        cuda_,
                                        DeviceId::cuda(0),
                                        flow->expected_gpu);
                                flow->cpu_edge_lane = std::make_shared<
                                    ExpertTierWeightTransferLane>(
                                    ExpertTierWeightTransferLane::Config{
                                        .device = DeviceId::cuda(0),
                                        .staging_capacity_bytes =
                                            flow->cpu_to_gpu_layout.chunkBytes(1),
                                        .lane_name = "three_tier:" + identity,
                                        .perf_device = "cuda-hot/rocm-warm/cpu-cold",
                                        .collect_timing_measurements = true,
                                    });
                                if (!flow->cpu_edge_lane->materialize(
                                        &conversion_error))
                                    throw std::runtime_error(conversion_error);
                            }
                            flows_[static_cast<std::size_t>(expert)]
                                  [projection_index] = std::move(flow);
                        }
                    }

                    cuda_source_ready_ = cuda_.createEvent(0);
                    rocm_source_ready_ = rocm_.createEvent(0);
                    if (!cuda_source_ready_ || !rocm_source_ready_ ||
                        !cuda_.recordEvent(
                            cuda_source_ready_, 0, cuda_producer_stream_) ||
                        !rocm_.recordEvent(
                            rocm_source_ready_, 0, rocm_producer_stream_))
                    {
                        throw std::runtime_error(
                            "Could not publish exact source-ready events");
                    }
                    initialized_ = true;
                    return true;
                }
                catch (const std::exception &exception)
                {
                    if (error)
                        *error = exception.what();
                    return false;
                }
                catch (...)
                {
                    if (error)
                        *error = "Three-tier factory initialization failed";
                    return false;
                }
            }

            /** @brief Start all nine physical operations from the histogram plan. */
            MoEOverlayTierPreparedWave prepare(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                MoEOverlayTierPreparedWave wave;
                if (!initialized_ || started_ ||
                    transaction.migrations.size() != 3)
                {
                    wave.status = MoEOverlayResidencyStageStartStatus::Failed;
                    wave.error = "Three-tier physical factory has invalid lifecycle";
                    return wave;
                }

                started_ = true;
                wave.status = MoEOverlayResidencyStageStartStatus::Started;
                wave.inactive_bank =
                    std::make_unique<PhysicalInactiveBank>(bank_observations);
                for (const auto &migration : transaction.migrations)
                {
                    if (!expectedEdge(migration))
                    {
                        wave.status = MoEOverlayResidencyStageStartStatus::Failed;
                        wave.error = "Histogram proposal did not form the expected three-tier rotation";
                        return wave;
                    }

                    for (std::size_t projection_index = 0;
                         projection_index < kProjectionCount;
                         ++projection_index)
                    {
                        auto &flow = *flows_[
                            static_cast<std::size_t>(migration.expert_id)]
                            [projection_index];
                        std::string start_error;
                        bool started = false;
                        if (migration.expert_id == 0)
                        {
                            started = flow.blob_lane->start(
                                flow.gpu_source->descriptor(),
                                flow.gpu_destination->descriptor(),
                                ExpertTierSourceReadiness::producerEvent(
                                    cuda_source_ready_),
                                &start_error);
                            if (started ||
                                flow.blob_lane->progress() ==
                                    ExpertTierGpuBlobTransferProgress::Pending)
                            {
                                wave.transfers.push_back(
                                    std::make_unique<
                                        ExpertTierGpuBlobLaneOperation>(
                                        flow.blob_lane));
                            }
                        }
                        else if (migration.expert_id == 1)
                        {
                            started = flow.cpu_edge_lane->startGpuToCpu(
                                flow.gpu_to_cpu_layout,
                                flow.gpu_source->constView(),
                                flow.cpu_destination,
                                ExpertTierSourceReadiness::producerEvent(
                                    rocm_source_ready_),
                                &start_error);
                            if (started ||
                                flow.cpu_edge_lane->progress() ==
                                    ExpertTierWeightTransferProgress::Pending)
                            {
                                wave.transfers.push_back(
                                    std::make_unique<
                                        ExpertTierWeightLaneOperation>(
                                        flow.cpu_edge_lane));
                            }
                        }
                        else
                        {
                            started = flow.cpu_edge_lane->startCpuToGpu(
                                flow.cpu_to_gpu_layout,
                                std::span<const std::uint8_t>(
                                    flow.expected_cpu.native_interleaved.data(),
                                    flow.expected_cpu.native_interleaved.size()),
                                flow.gpu_destination->mutableView(),
                                &start_error);
                            if (started ||
                                flow.cpu_edge_lane->progress() ==
                                    ExpertTierWeightTransferProgress::Pending)
                            {
                                wave.transfers.push_back(
                                    std::make_unique<
                                        ExpertTierWeightLaneOperation>(
                                        flow.cpu_edge_lane));
                            }
                        }

                        if (!started)
                        {
                            wave.status =
                                MoEOverlayResidencyStageStartStatus::Failed;
                            wave.error = start_error.empty()
                                             ? "A physical tier lane failed to start"
                                             : std::move(start_error);
                            return wave;
                        }
                    }
                }
                return wave;
            }

            /** @brief Return whether at least one physical lane is still pending. */
            [[nodiscard]] bool anyPending() const noexcept
            {
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (!flow)
                            continue;
                        if (flow->cpu_edge_lane &&
                            flow->cpu_edge_lane->progress() ==
                                ExpertTierWeightTransferProgress::Pending)
                            return true;
                        if (flow->blob_lane &&
                            flow->blob_lane->progress() ==
                                ExpertTierGpuBlobTransferProgress::Pending)
                            return true;
                    }
                }
                return false;
            }

            /** @brief Verify all nine final projection representations exactly. */
            [[nodiscard]] bool verifyDestinations(std::string *error) noexcept
            {
                for (std::size_t projection_index = 0;
                     projection_index < kProjectionCount;
                     ++projection_index)
                {
                    auto &cuda_to_rocm = *flows_[0][projection_index];
                    if (!cuda_to_rocm.gpu_destination->matches(
                            cuda_to_rocm.expected_gpu,
                            rocm_observation_stream_,
                            error))
                        return false;

                    auto &rocm_to_cpu = *flows_[1][projection_index];
                    if (rocm_to_cpu.cpu_destination.size() !=
                            rocm_to_cpu.expected_cpu.native_interleaved.size() ||
                        !std::equal(
                            rocm_to_cpu.cpu_destination.begin(),
                            rocm_to_cpu.cpu_destination.end(),
                            rocm_to_cpu.expected_cpu.native_interleaved.begin()))
                    {
                        if (error)
                            *error = "ROCm-to-CPU final execution bytes differ";
                        return false;
                    }

                    auto &cpu_to_cuda = *flows_[2][projection_index];
                    if (!cpu_to_cuda.gpu_destination->matches(
                            cpu_to_cuda.expected_gpu,
                            cuda_observation_stream_,
                            error))
                        return false;
                }
                return true;
            }

            /** @brief Sum CPU-edge counters across six physical lanes. */
            [[nodiscard]] ExpertTierWeightTransferLaneStats cpuEdgeStats()
                const noexcept
            {
                ExpertTierWeightTransferLaneStats total;
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (!flow || !flow->cpu_edge_lane)
                            continue;
                        const auto stats = flow->cpu_edge_lane->stats();
                        total.transfers_started += stats.transfers_started;
                        total.transfers_completed += stats.transfers_completed;
                        total.chunks_submitted += stats.chunks_submitted;
                        total.bytes_submitted += stats.bytes_submitted;
                        total.pending_event_polls += stats.pending_event_polls;
                        total.failed_transfers += stats.failed_transfers;
                        total.timing_measurement_failures +=
                            stats.timing_measurement_failures;
                        total.inference_stream_waits += stats.inference_stream_waits;
                        total.blocking_synchronizations +=
                            stats.blocking_synchronizations;
                    }
                }
                return total;
            }

            /** @return Whether every completed CPU-edge lane retained exact timing. */
            [[nodiscard]] bool allCpuEdgeMeasurementsValid() const noexcept
            {
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (!flow || !flow->cpu_edge_lane)
                            continue;
                        const auto stats = flow->cpu_edge_lane->stats();
                        if (!stats.last_measurement.valid() ||
                            stats.last_measurement.device_nanoseconds == 0u ||
                            stats.last_measurement.host_nanoseconds == 0u)
                            return false;
                    }
                }
                return true;
            }

            /** @brief Sum heterogeneous blob counters across three lanes. */
            [[nodiscard]] ExpertTierGpuBlobTransferLaneStats blobStats()
                const noexcept
            {
                ExpertTierGpuBlobTransferLaneStats total;
                for (const auto &flow : flows_[0])
                {
                    if (!flow || !flow->blob_lane)
                        continue;
                    const auto stats = flow->blob_lane->stats();
                    total.transfers_started += stats.transfers_started;
                    total.transfers_completed += stats.transfers_completed;
                    total.chunks_submitted += stats.chunks_submitted;
                    total.chunks_completed += stats.chunks_completed;
                    total.bytes_submitted += stats.bytes_submitted;
                    total.source_d2h_submissions +=
                        stats.source_d2h_submissions;
                    total.destination_h2d_submissions +=
                        stats.destination_h2d_submissions;
                    total.host_relay_copies += stats.host_relay_copies;
                    total.host_relay_bytes += stats.host_relay_bytes;
                    total.pending_event_polls += stats.pending_event_polls;
                    total.failed_transfers += stats.failed_transfers;
                    total.timing_measurement_failures +=
                        stats.timing_measurement_failures;
                    total.maximum_in_flight_chunks = std::max(
                        total.maximum_in_flight_chunks,
                        stats.maximum_in_flight_chunks);
                    total.inference_stream_waits += stats.inference_stream_waits;
                    total.blocking_synchronizations +=
                        stats.blocking_synchronizations;
                }
                return total;
            }

            /** @return Whether every blob lane retained device and relay timing. */
            [[nodiscard]] bool allBlobMeasurementsValid() const noexcept
            {
                for (const auto &flow : flows_[0])
                {
                    if (!flow || !flow->blob_lane)
                        continue;
                    const auto stats = flow->blob_lane->stats();
                    if (!stats.last_measurement.valid() ||
                        stats.last_measurement.device_nanoseconds == 0u ||
                        stats.last_measurement.host_nanoseconds == 0u)
                        return false;
                }
                return true;
            }

            std::shared_ptr<PhysicalBankObservations> bank_observations;

        private:
            /** @brief Validate exact source/destination backend for one expert. */
            [[nodiscard]] static bool expectedEdge(
                const MoEOverlayTierMigration &migration) noexcept
            {
                if (migration.expert_id == 0)
                    return migration.source.device.is_cuda() &&
                           migration.destination.device.is_rocm();
                if (migration.expert_id == 1)
                    return migration.source.device.is_rocm() &&
                           migration.destination.device.is_cpu();
                if (migration.expert_id == 2)
                    return migration.source.device.is_cpu() &&
                           migration.destination.device.is_cuda();
                return false;
            }

            IBackend &cuda_;
            IBackend &rocm_;
            std::array<
                std::array<
                    std::unique_ptr<PhysicalProjectionFlow>,
                    kProjectionCount>,
                3>
                flows_{};
            void *cuda_producer_stream_ = nullptr;
            void *rocm_producer_stream_ = nullptr;
            void *cuda_observation_stream_ = nullptr;
            void *rocm_observation_stream_ = nullptr;
            void *cuda_source_ready_ = nullptr;
            void *rocm_source_ready_ = nullptr;
            bool initialized_ = false;
            bool started_ = false;
        };

        /** @brief Physical storage and lane for one CUDA/CPU expert projection. */
        struct CudaCpuProjectionFlow
        {
            HostGpuExpertPackedProjection expected_gpu;
            cpu::native_vnni::CPUNativeVNNIPackedWeights expected_cpu;
            ExpertTierWeightDeviceLayout gpu_to_cpu_layout;
            ExpertTierWeightDeviceLayout cpu_to_gpu_layout;
            std::unique_ptr<DevicePackedProjection> gpu_source;
            std::unique_ptr<DevicePackedProjection> gpu_destination;
            std::vector<std::uint8_t> cpu_destination;
            std::shared_ptr<ExpertTierWeightTransferLane> lane;
        };

        /**
         * @brief Pre-materialized factory for a CUDA-hot/CPU-cold expert swap.
         *
         * Expert 0 starts hot on CUDA and is demoted to CPU. Expert 1 starts
         * cold on CPU and is promoted to CUDA. Each expert owns independent
         * gate, up, and down lanes so all six transfers may progress together.
         */
        class RealCudaCpuWaveFactory final
            : public IMoEOverlayTierPreparedWaveFactory
        {
        public:
            /** @brief Bind the exact CUDA allocation and event authority. */
            explicit RealCudaCpuWaveFactory(IBackend &cuda)
                : cuda_(cuda),
                  bank_observations(
                      std::make_shared<PhysicalBankObservations>())
            {
            }

            /** @brief Destroy the source-ready event after every lane drains. */
            ~RealCudaCpuWaveFactory()
            {
                if (cuda_source_ready_)
                    cuda_.destroyEvent(cuda_source_ready_, 0);
            }

            RealCudaCpuWaveFactory(const RealCudaCpuWaveFactory &) = delete;
            RealCudaCpuWaveFactory &operator=(
                const RealCudaCpuWaveFactory &) = delete;

            /**
             * @brief Allocate and upload every persistent transfer resource.
             * @param error Receives the first materialization failure.
             * @return Whether the factory is ready before ticket admission.
             */
            [[nodiscard]] bool initialize(std::string *error) noexcept
            {
                try
                {
                    auto &cuda_context =
                        GPUDeviceContextPool::instance().getContext(
                            DeviceId::cuda(0));
                    cuda_producer_stream_ =
                        cuda_context.getOrCreateAuxiliaryStream(
                            "cuda_cpu_rotation:cuda_source_producer");
                    cuda_observation_stream_ =
                        cuda_context.getOrCreateAuxiliaryStream(
                            "cuda_cpu_rotation:cuda_observation");
                    if (!cuda_producer_stream_ || !cuda_observation_stream_)
                    {
                        throw std::runtime_error(
                            "Could not create explicit CUDA/CPU streams");
                    }

                    const auto *source_format =
                        native_vnni_formats::forSourceIdentity(5, true);
                    if (!source_format)
                    {
                        throw std::runtime_error(
                            "Q4_K source format is absent from NativeVNNI catalog");
                    }

                    for (int expert = 0; expert < 2; ++expert)
                    {
                        for (std::size_t projection_index = 0;
                             projection_index < kProjectionCount;
                             ++projection_index)
                        {
                            const auto &spec = kProjections[projection_index];
                            auto flow =
                                std::make_unique<CudaCpuProjectionFlow>();
                            flow->expected_gpu = makeProjection(
                                spec,
                                static_cast<std::uint32_t>(
                                    51000 + expert * 100 + projection_index));
                            std::string conversion_error;
                            if (!flow->expected_gpu.valid(&conversion_error) ||
                                !gpuToCpuExpertPackedReference(
                                    flow->expected_gpu,
                                    flow->expected_cpu,
                                    &conversion_error))
                            {
                                throw std::runtime_error(conversion_error);
                            }

                            flow->gpu_to_cpu_layout =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    spec.n,
                                    spec.k,
                                    2,
                                    0,
                                    expert,
                                    spec.projection,
                                    1)
                                    .deviceLayout();
                            flow->cpu_to_gpu_layout =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    flow->expected_cpu,
                                    2,
                                    0,
                                    expert,
                                    spec.projection,
                                    1)
                                    .deviceLayout();

                            const std::string identity =
                                "expert" + std::to_string(expert) + ":" +
                                spec.name;
                            if (expert == 0)
                            {
                                flow->gpu_source =
                                    std::make_unique<DevicePackedProjection>(
                                        cuda_,
                                        DeviceId::cuda(0),
                                        flow->expected_gpu);
                                if (!flow->gpu_source->upload(
                                        flow->expected_gpu,
                                        cuda_producer_stream_))
                                {
                                    throw std::runtime_error(
                                        "CUDA source upload failed for " +
                                        identity);
                                }
                                flow->cpu_destination.resize(
                                    flow->expected_cpu.native_interleaved.size());
                            }
                            else
                            {
                                flow->gpu_destination =
                                    std::make_unique<DevicePackedProjection>(
                                        cuda_,
                                        DeviceId::cuda(0),
                                        flow->expected_gpu);
                            }

                            const auto &layout = expert == 0
                                                     ? flow->gpu_to_cpu_layout
                                                     : flow->cpu_to_gpu_layout;
                            flow->lane = std::make_shared<
                                ExpertTierWeightTransferLane>(
                                ExpertTierWeightTransferLane::Config{
                                    .device = DeviceId::cuda(0),
                                    .staging_capacity_bytes =
                                        layout.chunkBytes(1),
                                    .lane_name =
                                        "cuda_cpu_rotation:" + identity,
                                    .perf_device = "cuda-hot/cpu-cold",
                                    .collect_timing_measurements = true,
                                });
                            if (!flow->lane->materialize(&conversion_error))
                                throw std::runtime_error(conversion_error);
                            flows_[static_cast<std::size_t>(expert)]
                                  [projection_index] = std::move(flow);
                        }
                    }

                    cuda_source_ready_ = cuda_.createEvent(0);
                    if (!cuda_source_ready_ ||
                        !cuda_.recordEvent(
                            cuda_source_ready_,
                            0,
                            cuda_producer_stream_))
                    {
                        throw std::runtime_error(
                            "Could not publish the CUDA source-ready event");
                    }
                    initialized_ = true;
                    return true;
                }
                catch (const std::exception &exception)
                {
                    if (error)
                        *error = exception.what();
                    return false;
                }
                catch (...)
                {
                    if (error)
                        *error = "CUDA/CPU factory initialization failed";
                    return false;
                }
            }

            /** @brief Start all six directions selected by the histogram swap. */
            MoEOverlayTierPreparedWave prepare(
                const MoEOverlayResidencyTransaction &transaction) override
            {
                MoEOverlayTierPreparedWave wave;
                if (!initialized_ || started_ ||
                    transaction.migrations.size() != 2)
                {
                    wave.status =
                        MoEOverlayResidencyStageStartStatus::Failed;
                    wave.error =
                        "CUDA/CPU physical factory has invalid lifecycle";
                    return wave;
                }

                started_ = true;
                wave.status = MoEOverlayResidencyStageStartStatus::Started;
                wave.inactive_bank =
                    std::make_unique<PhysicalInactiveBank>(bank_observations);
                for (const auto &migration : transaction.migrations)
                {
                    if (!expectedEdge(migration))
                    {
                        wave.status =
                            MoEOverlayResidencyStageStartStatus::Failed;
                        wave.error =
                            "Histogram proposal did not form the CUDA/CPU swap";
                        return wave;
                    }

                    for (std::size_t projection_index = 0;
                         projection_index < kProjectionCount;
                         ++projection_index)
                    {
                        auto &flow = *flows_[
                            static_cast<std::size_t>(migration.expert_id)]
                            [projection_index];
                        std::string start_error;
                        bool started = false;
                        if (migration.expert_id == 0)
                        {
                            started = flow.lane->startGpuToCpu(
                                flow.gpu_to_cpu_layout,
                                flow.gpu_source->constView(),
                                flow.cpu_destination,
                                ExpertTierSourceReadiness::producerEvent(
                                    cuda_source_ready_),
                                &start_error);
                        }
                        else
                        {
                            started = flow.lane->startCpuToGpu(
                                flow.cpu_to_gpu_layout,
                                std::span<const std::uint8_t>(
                                    flow.expected_cpu.native_interleaved.data(),
                                    flow.expected_cpu.native_interleaved.size()),
                                flow.gpu_destination->mutableView(),
                                &start_error);
                        }

                        if (started ||
                            flow.lane->progress() ==
                                ExpertTierWeightTransferProgress::Pending)
                        {
                            wave.transfers.push_back(
                                std::make_unique<
                                    ExpertTierWeightLaneOperation>(flow.lane));
                        }
                        if (!started)
                        {
                            wave.status =
                                MoEOverlayResidencyStageStartStatus::Failed;
                            wave.error = start_error.empty()
                                             ? "A CUDA/CPU tier lane failed to start"
                                             : std::move(start_error);
                            return wave;
                        }
                    }
                }
                return wave;
            }

            /** @brief Return whether any of the six physical lanes is pending. */
            [[nodiscard]] bool anyPending() const noexcept
            {
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (flow &&
                            flow->lane->progress() ==
                                ExpertTierWeightTransferProgress::Pending)
                        {
                            return true;
                        }
                    }
                }
                return false;
            }

            /** @brief Compare both final execution representations bytewise. */
            [[nodiscard]] bool verifyDestinations(std::string *error) noexcept
            {
                for (std::size_t projection_index = 0;
                     projection_index < kProjectionCount;
                     ++projection_index)
                {
                    const auto &cuda_to_cpu = *flows_[0][projection_index];
                    if (cuda_to_cpu.cpu_destination.size() !=
                            cuda_to_cpu.expected_cpu.native_interleaved.size() ||
                        !std::equal(
                            cuda_to_cpu.cpu_destination.begin(),
                            cuda_to_cpu.cpu_destination.end(),
                            cuda_to_cpu.expected_cpu.native_interleaved.begin()))
                    {
                        if (error)
                            *error = "CUDA-to-CPU final execution bytes differ";
                        return false;
                    }

                    auto &cpu_to_cuda = *flows_[1][projection_index];
                    if (!cpu_to_cuda.gpu_destination->matches(
                            cpu_to_cuda.expected_gpu,
                            cuda_observation_stream_,
                            error))
                    {
                        return false;
                    }
                }
                return true;
            }

            /** @brief Sum proof counters across the six CUDA-owned lanes. */
            [[nodiscard]] ExpertTierWeightTransferLaneStats stats()
                const noexcept
            {
                ExpertTierWeightTransferLaneStats total;
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (!flow)
                            continue;
                        const auto lane_stats = flow->lane->stats();
                        total.transfers_started +=
                            lane_stats.transfers_started;
                        total.transfers_completed +=
                            lane_stats.transfers_completed;
                        total.chunks_submitted += lane_stats.chunks_submitted;
                        total.bytes_submitted += lane_stats.bytes_submitted;
                        total.pending_event_polls +=
                            lane_stats.pending_event_polls;
                        total.failed_transfers += lane_stats.failed_transfers;
                        total.timing_measurement_failures +=
                            lane_stats.timing_measurement_failures;
                        total.inference_stream_waits +=
                            lane_stats.inference_stream_waits;
                        total.blocking_synchronizations +=
                            lane_stats.blocking_synchronizations;
                    }
                }
                return total;
            }

            /** @return Whether all six bidirectional CPU-edge observations exist. */
            [[nodiscard]] bool allMeasurementsValid() const noexcept
            {
                for (const auto &expert : flows_)
                {
                    for (const auto &flow : expert)
                    {
                        if (!flow)
                            continue;
                        const auto stats = flow->lane->stats();
                        if (!stats.last_measurement.valid() ||
                            stats.last_measurement.device_nanoseconds == 0u ||
                            stats.last_measurement.host_nanoseconds == 0u)
                            return false;
                    }
                }
                return true;
            }

            std::shared_ptr<PhysicalBankObservations> bank_observations;

        private:
            /** @brief Validate both directed edges in the capacity swap. */
            [[nodiscard]] static bool expectedEdge(
                const MoEOverlayTierMigration &migration) noexcept
            {
                if (migration.expert_id == 0)
                {
                    return migration.source.device.is_cuda() &&
                           migration.destination.device.is_cpu();
                }
                if (migration.expert_id == 1)
                {
                    return migration.source.device.is_cpu() &&
                           migration.destination.device.is_cuda();
                }
                return false;
            }

            IBackend &cuda_;
            std::array<
                std::array<
                    std::unique_ptr<CudaCpuProjectionFlow>,
                    kProjectionCount>,
                2>
                flows_{};
            void *cuda_producer_stream_ = nullptr;
            void *cuda_observation_stream_ = nullptr;
            void *cuda_source_ready_ = nullptr;
            bool initialized_ = false;
            bool started_ = false;
        };

        /** @brief Construct one single-participant physical execution domain. */
        RoutedExpertDomain makeDomain(
            std::string name,
            GlobalDeviceAddress participant,
            int world_rank,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {participant};
            result.world_ranks = {world_rank};
            result.owner_rank = world_rank;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Construct one fixed-capacity thermal tier. */
        RoutedExpertTier makeTier(
            std::string name,
            std::string domain,
            int priority,
            int capacity,
            bool fallback = false)
        {
            return {
                .name = std::move(name),
                .domain = std::move(domain),
                .priority = priority,
                .max_experts_per_layer = capacity,
                .fallback = fallback,
            };
        }

        /** @brief Declare CUDA-hot, ROCm-warm, and CPU-cold one-slot tiers. */
        MoERoutedExpertPlacementPlan threeTierPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                makeDomain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
                makeDomain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(1, 0),
                    1,
                    CollectiveBackendType::RCCL),
                makeDomain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(2),
                    2,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                makeTier("hot", "cuda_hot", 0, 1),
                makeTier("warm", "rocm_warm", 1, 1),
                makeTier("cold", "cpu_cold", 2, 0, true),
            };
            return plan;
        }

        /** @brief Declare one CUDA-hot slot and one CPU-cold fallback slot. */
        MoERoutedExpertPlacementPlan cudaCpuTierPlan()
        {
            MoERoutedExpertPlacementPlan plan;
            plan.enabled = true;
            plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
            plan.continuation_domain = "cuda_hot";
            plan.shared_expert_domain = "cuda_hot";
            plan.residency_policy =
                RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
            plan.domains = {
                makeDomain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0, 0),
                    0,
                    CollectiveBackendType::NCCL),
                makeDomain(
                    "cpu_cold",
                    GlobalDeviceAddress::cpu(1),
                    1,
                    CollectiveBackendType::MPI),
            };
            plan.routed_tiers = {
                makeTier("hot", "cuda_hot", 0, 1),
                makeTier("cold", "cpu_cold", 1, 0, true),
            };
            return plan;
        }

        /** @brief Qwen-shaped metadata sufficient for the three expert proof. */
        MoERoutedExpertModelMetadata threeTierMetadata()
        {
            return {
                .num_layers = 1,
                .num_experts = 3,
                .d_model = 256,
                .routed_intermediate_size = 192,
                .routed_quant_type = "Q4_K",
            };
        }

        /** @brief Qwen-shaped metadata sufficient for a two-expert swap. */
        MoERoutedExpertModelMetadata cudaCpuTierMetadata()
        {
            return {
                .num_layers = 1,
                .num_experts = 2,
                .d_model = 256,
                .routed_intermediate_size = 192,
                .routed_quant_type = "Q4_K",
            };
        }

        /** @brief Freeze counts rotating expert 2 into hot and expert 1 cold. */
        std::unique_ptr<DecodeExpertHistogram> rotationHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 3;
            config.top_k = 1;
            config.window_size = 3;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::rocm(0),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                3,
                {0, 1, 2});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::uint64_t counts[] = {80, 70, 100};
            histogram->mergeLayerCounts(0, counts, 3, false);
            return histogram;
        }

        /** @brief Freeze counts that swap the CPU and CUDA resident experts. */
        std::unique_ptr<DecodeExpertHistogram> cudaCpuSwapHistogram()
        {
            DecodeExpertHistogramConfig config;
            config.num_layers = 1;
            config.num_experts = 2;
            config.top_k = 1;
            config.window_size = 2;
            config.sockets = {
                DeviceId::cuda(0),
                DeviceId::cpu(),
            };
            config.ownership = MoELayeredExpertOwnership::uniform(
                1,
                2,
                {0, 1});
            auto histogram = std::make_unique<DecodeExpertHistogram>(config);
            const std::uint64_t counts[] = {10, 100};
            histogram->mergeLayerCounts(0, counts, 2, false);
            return histogram;
        }

        /** @brief Sum one exported movement counter across tag variants. */
        double movementCounter(const std::string &name)
        {
            double total = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name == name)
                    total += record.value;
            }
            return total;
        }

        TEST(
            MoEOverlayCudaCpuMigrationIntegration,
            HistogramSwapMovesBothDirectionsWithoutBlockingCudaInference)
        {
#if !defined(HAVE_CUDA)
            GTEST_SKIP() << "CUDA is required";
#else
            IBackend *cuda = getCUDABackend();
            if (!cuda || cuda->deviceCount() < 1)
                GTEST_SKIP() << "One CUDA device is required";

            ScopedThreeTierPerfStats perf_stats;
            RealCudaCpuWaveFactory factory(*cuda);
            std::string error;
            ASSERT_TRUE(factory.initialize(&error)) << error;

            /*
             * The inference witness and completion event exist before ticket
             * admission. Its stream is independent from all six migration
             * lanes, matching the production ownership boundary.
             */
            auto &cuda_context =
                GPUDeviceContextPool::instance().getContext(
                    DeviceId::cuda(0));
            void *cuda_inference_stream =
                cuda_context.getOrCreateAuxiliaryStream(
                    "cuda_cpu_rotation:cuda_hot_path");
            ASSERT_NE(cuda_inference_stream, nullptr);
            DeviceCopyWitness cuda_witness(*cuda, DeviceId::cuda(0));
            void *cuda_done = cuda->createEvent(0);
            ASSERT_NE(cuda_done, nullptr);

            auto histogram = cudaCpuSwapHistogram();
            MoEOverlayResidencyAuthority authority({
                .initial_plan = cudaCpuTierPlan(),
                .model_metadata = cudaCpuTierMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "cuda-hot/cpu-cold",
            });
            const auto transaction = authority.proposeFromHistogram();
            ASSERT_TRUE(transaction.valid());
            ASSERT_EQ(transaction.migrations.size(), 2u);
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);

            auto old_ticket = authority.tryAcquireTicketSnapshot();
            ASSERT_TRUE(old_ticket.has_value());
            ASSERT_EQ((*old_ticket)->epoch, 1u);

            MoEOverlayTierMigrationTransport transport({
                .factory = &factory,
                .projections_per_expert = kProjectionCount,
                .perf_device = "cuda-hot/cpu-cold",
            });
            const auto started = authority.beginApply(transaction, transport);
            if (started.status != MoEOverlayResidencyApplyStatus::Started)
            {
                EXPECT_EQ(
                    started.status,
                    MoEOverlayResidencyApplyStatus::Started)
                    << started.error;
                cuda->destroyEvent(cuda_done, 0);
                return;
            }

            const bool cuda_work_enqueued =
                cuda_witness.enqueue(cuda_inference_stream);
            const bool cuda_event_recorded =
                cuda->recordEvent(cuda_done, 0, cuda_inference_stream);

            bool cuda_ready = false;
            bool cuda_ready_during_migration = false;
            MoEOverlayResidencyApplyResult progress = started;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (std::chrono::steady_clock::now() < deadline &&
                   progress.status !=
                       MoEOverlayResidencyApplyStatus::Committed &&
                   progress.ok())
            {
                if (cuda_event_recorded)
                    (void)cuda_context.queryEventChecked(cuda_done, cuda_ready);
                if (cuda_ready && factory.anyPending() &&
                    authority.snapshot()->epoch == 1)
                {
                    cuda_ready_during_migration = true;
                }
                progress = authority.advanceBackground();
                std::this_thread::yield();
            }

            /* Drain only event-polled abort cleanup before lane destruction. */
            while (authority.pendingAbortCount() != 0 &&
                   std::chrono::steady_clock::now() < deadline)
            {
                (void)authority.advanceBackground();
                std::this_thread::yield();
            }

            EXPECT_TRUE(cuda_work_enqueued);
            EXPECT_TRUE(cuda_event_recorded);
            EXPECT_TRUE(cuda_ready);
            EXPECT_TRUE(cuda_ready_during_migration)
                << "Independent CUDA work did not finish during the two-tier wave";
            ASSERT_EQ(
                progress.status,
                MoEOverlayResidencyApplyStatus::Committed)
                << progress.error;
            ASSERT_EQ(authority.snapshot()->epoch, 2u);
            ASSERT_TRUE(old_ticket.has_value());
            EXPECT_EQ((*old_ticket)->epoch, 1u);
            EXPECT_EQ(authority.pendingRetirementCount(), 1u);

            ASSERT_TRUE(factory.verifyDestinations(&error)) << error;
            const auto authority_stats = authority.stats();
            EXPECT_EQ(authority_stats.committed_migrations, 2u);
            EXPECT_EQ(authority_stats.committed_cycles, 1u);
            EXPECT_EQ(authority_stats.promotions, 1u);
            EXPECT_EQ(authority_stats.demotions, 1u);
            EXPECT_EQ(authority_stats.cross_domain_migrations, 2u);
            EXPECT_EQ(authority_stats.cross_rank_migrations, 2u);
            EXPECT_EQ(authority_stats.cross_backend_migrations, 2u);
            EXPECT_EQ(authority_stats.published_with_old_tickets, 1u);

            const auto transport_stats = transport.stats();
            EXPECT_EQ(transport_stats.transfer_operations_started, 6u);
            EXPECT_EQ(transport_stats.transfer_operations_completed, 6u);
            EXPECT_EQ(transport_stats.commits_started, 1u);
            EXPECT_EQ(transport_stats.commits_completed, 1u);
            EXPECT_EQ(transport_stats.inference_stream_waits, 0u);
            EXPECT_EQ(transport_stats.blocking_synchronizations, 0u);

            const auto lane_stats = factory.stats();
            EXPECT_EQ(lane_stats.transfers_started, 6u);
            EXPECT_EQ(lane_stats.transfers_completed, 6u);
            EXPECT_GT(lane_stats.chunks_submitted, 6u);
            EXPECT_EQ(lane_stats.failed_transfers, 0u);
            EXPECT_EQ(lane_stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(factory.allMeasurementsValid());
            EXPECT_EQ(lane_stats.inference_stream_waits, 0u);
            EXPECT_EQ(lane_stats.blocking_synchronizations, 0u);

            EXPECT_EQ(movementCounter("committed_expert_migrations"), 2.0);
            EXPECT_EQ(movementCounter("promotions"), 1.0);
            EXPECT_EQ(movementCounter("demotions"), 1.0);
            EXPECT_EQ(movementCounter("cross_domain_migrations"), 2.0);
            EXPECT_EQ(movementCounter("cross_backend_migrations"), 2.0);
            EXPECT_EQ(movementCounter("tier_transfers_completed"), 6.0);

            old_ticket.reset();
            EXPECT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Idle);
            EXPECT_EQ(authority.pendingRetirementCount(), 0u);
            EXPECT_EQ(factory.bank_observations->commits_started, 1u);
            EXPECT_EQ(factory.bank_observations->retires, 1u);

            cuda->destroyEvent(cuda_done, 0);
#endif
        }

        TEST(
            MoEOverlayThreeTierMigrationIntegration,
            HistogramRotationMovesAllProjectionsWithoutBlockingEitherGPU)
        {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
            GTEST_SKIP() << "CUDA and ROCm are both required";
#else
            IBackend *cuda = getCUDABackend();
            IBackend *rocm = getROCmBackend();
            if (!cuda || !rocm ||
                cuda->deviceCount() < 1 || rocm->deviceCount() < 1)
            {
                GTEST_SKIP() << "One CUDA and one ROCm device are required";
            }

            ScopedThreeTierPerfStats perf_stats;
            RealThreeTierWaveFactory factory(*cuda, *rocm);
            std::string error;
            ASSERT_TRUE(factory.initialize(&error)) << error;

            /*
             * Hot-path witnesses, streams, and events are topology resources:
             * materialize them before the first inference ticket and before a
             * migration wave can enqueue any work.
             */
            auto &cuda_context =
                GPUDeviceContextPool::instance().getContext(
                    DeviceId::cuda(0));
            auto &rocm_context =
                GPUDeviceContextPool::instance().getContext(
                    DeviceId::rocm(0));
            void *cuda_inference_stream =
                cuda_context.getOrCreateAuxiliaryStream(
                    "three_tier_rotation:cuda_hot_path");
            void *rocm_inference_stream =
                rocm_context.getOrCreateAuxiliaryStream(
                    "three_tier_rotation:rocm_hot_path");
            ASSERT_NE(cuda_inference_stream, nullptr);
            ASSERT_NE(rocm_inference_stream, nullptr);
            DeviceCopyWitness cuda_witness(*cuda, DeviceId::cuda(0));
            DeviceCopyWitness rocm_witness(*rocm, DeviceId::rocm(0));
            void *cuda_done = cuda->createEvent(0);
            void *rocm_done = rocm->createEvent(0);
            ASSERT_NE(cuda_done, nullptr);
            ASSERT_NE(rocm_done, nullptr);

            auto histogram = rotationHistogram();
            MoEOverlayResidencyAuthority authority({
                .initial_plan = threeTierPlan(),
                .model_metadata = threeTierMetadata(),
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .perf_device = "cuda-hot/rocm-warm/cpu-cold",
            });
            const auto transaction = authority.proposeFromHistogram();
            ASSERT_TRUE(transaction.valid());
            ASSERT_EQ(transaction.migrations.size(), 3u);
            ASSERT_EQ(transaction.migration_cycles.size(), 1u);

            auto old_ticket = authority.tryAcquireTicketSnapshot();
            ASSERT_TRUE(old_ticket.has_value());
            ASSERT_EQ((*old_ticket)->epoch, 1u);

            MoEOverlayTierMigrationTransport transport({
                .factory = &factory,
                .projections_per_expert = kProjectionCount,
                .perf_device = "cuda-hot/rocm-warm/cpu-cold",
            });
            const auto started = authority.beginApply(transaction, transport);
            if (started.status != MoEOverlayResidencyApplyStatus::Started)
            {
                EXPECT_EQ(
                    started.status,
                    MoEOverlayResidencyApplyStatus::Started)
                    << started.error;
                return;
            }

            const bool cuda_work_enqueued =
                cuda_witness.enqueue(cuda_inference_stream);
            const bool rocm_work_enqueued =
                rocm_witness.enqueue(rocm_inference_stream);
            const bool cuda_event_recorded =
                cuda->recordEvent(cuda_done, 0, cuda_inference_stream);
            const bool rocm_event_recorded =
                rocm->recordEvent(rocm_done, 0, rocm_inference_stream);

            bool cuda_ready = false;
            bool rocm_ready = false;
            bool both_devices_ready_during_migration = false;
            MoEOverlayResidencyApplyResult progress = started;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(30);
            while (std::chrono::steady_clock::now() < deadline &&
                   progress.status != MoEOverlayResidencyApplyStatus::Committed &&
                   progress.ok())
            {
                if (cuda_event_recorded)
                    (void)cuda_context.queryEventChecked(cuda_done, cuda_ready);
                if (rocm_event_recorded)
                    (void)rocm_context.queryEventChecked(rocm_done, rocm_ready);
                if (cuda_ready && rocm_ready && factory.anyPending() &&
                    authority.snapshot()->epoch == 1)
                {
                    both_devices_ready_during_migration = true;
                }
                progress = authority.advanceBackground();
                std::this_thread::yield();
            }

            /*
             * If a normal assertion below fails, first finish any event-polled
             * cleanup so lane destructors never mask the original diagnostic.
             */
            while (authority.pendingAbortCount() != 0 &&
                   std::chrono::steady_clock::now() < deadline)
            {
                (void)authority.advanceBackground();
                std::this_thread::yield();
            }

            EXPECT_TRUE(cuda_work_enqueued);
            EXPECT_TRUE(rocm_work_enqueued);
            EXPECT_TRUE(cuda_event_recorded);
            EXPECT_TRUE(rocm_event_recorded);
            EXPECT_TRUE(cuda_ready);
            EXPECT_TRUE(rocm_ready);
            EXPECT_TRUE(both_devices_ready_during_migration)
                << "Independent CUDA/ROCm work did not finish while migration remained pending";
            ASSERT_EQ(
                progress.status,
                MoEOverlayResidencyApplyStatus::Committed)
                << progress.error;
            ASSERT_EQ(authority.snapshot()->epoch, 2u);
            ASSERT_TRUE(old_ticket.has_value());
            EXPECT_EQ((*old_ticket)->epoch, 1u);
            EXPECT_EQ(authority.pendingRetirementCount(), 1u);

            ASSERT_TRUE(factory.verifyDestinations(&error)) << error;
            const auto authority_stats = authority.stats();
            EXPECT_EQ(authority_stats.committed_migrations, 3u);
            EXPECT_EQ(authority_stats.committed_cycles, 1u);
            EXPECT_EQ(authority_stats.promotions, 1u);
            EXPECT_EQ(authority_stats.demotions, 2u);
            EXPECT_EQ(authority_stats.cross_domain_migrations, 3u);
            EXPECT_EQ(authority_stats.cross_rank_migrations, 3u);
            EXPECT_EQ(authority_stats.cross_backend_migrations, 3u);

            const auto transport_stats = transport.stats();
            EXPECT_EQ(transport_stats.transfer_operations_started, 9u);
            EXPECT_EQ(transport_stats.transfer_operations_completed, 9u);
            EXPECT_EQ(transport_stats.commits_started, 1u);
            EXPECT_EQ(transport_stats.commits_completed, 1u);
            EXPECT_EQ(transport_stats.inference_stream_waits, 0u);
            EXPECT_EQ(transport_stats.blocking_synchronizations, 0u);

            const auto cpu_edge_stats = factory.cpuEdgeStats();
            EXPECT_EQ(cpu_edge_stats.transfers_started, 6u);
            EXPECT_EQ(cpu_edge_stats.transfers_completed, 6u);
            EXPECT_GT(cpu_edge_stats.chunks_submitted, 6u);
            EXPECT_EQ(cpu_edge_stats.failed_transfers, 0u);
            EXPECT_EQ(cpu_edge_stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(factory.allCpuEdgeMeasurementsValid());
            EXPECT_EQ(cpu_edge_stats.inference_stream_waits, 0u);
            EXPECT_EQ(cpu_edge_stats.blocking_synchronizations, 0u);

            const auto blob_stats = factory.blobStats();
            EXPECT_EQ(blob_stats.transfers_started, 3u);
            EXPECT_EQ(blob_stats.transfers_completed, 3u);
            EXPECT_EQ(blob_stats.chunks_submitted, blob_stats.chunks_completed);
            EXPECT_GT(blob_stats.host_relay_bytes, 0u);
            EXPECT_EQ(blob_stats.failed_transfers, 0u);
            EXPECT_EQ(blob_stats.timing_measurement_failures, 0u);
            EXPECT_TRUE(factory.allBlobMeasurementsValid());
            EXPECT_EQ(blob_stats.inference_stream_waits, 0u);
            EXPECT_EQ(blob_stats.blocking_synchronizations, 0u);

            EXPECT_EQ(movementCounter("committed_expert_migrations"), 3.0);
            EXPECT_EQ(movementCounter("promotions"), 1.0);
            EXPECT_EQ(movementCounter("demotions"), 2.0);
            EXPECT_EQ(movementCounter("cross_domain_migrations"), 3.0);
            EXPECT_EQ(movementCounter("tier_transfers_completed"), 6.0);
            EXPECT_EQ(
                movementCounter(
                    "heterogeneous_gpu_blob_transfers_completed"),
                3.0);

            old_ticket.reset();
            EXPECT_EQ(
                authority.advanceBackground().status,
                MoEOverlayResidencyApplyStatus::Idle);
            EXPECT_EQ(authority.pendingRetirementCount(), 0u);
            EXPECT_EQ(factory.bank_observations->commits_started, 1u);
            EXPECT_EQ(factory.bank_observations->retires, 1u);

            cuda->destroyEvent(cuda_done, 0);
            rocm->destroyEvent(rocm_done, 0);
#endif
        }
    } // namespace
} // namespace llaminar2
