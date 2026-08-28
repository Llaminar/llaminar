/**
 * @file Test__MoEOverlayROCmRemoteProjection_MPI.cpp
 * @brief Real-MPI proof for secondary-ROCm ExpertOverlay weight movement.
 *
 * A two-rank production composition streams one Q8_0 projection in both
 * directions between a CPU rank and ROCm device one.  Every wave deliberately
 * leaves the maintenance thread's current device set to ROCm device zero, then
 * exercises the production GPU endpoint and MPI transport.  This locks down
 * device selection, chunk ownership, event-polled progress, exact final bytes,
 * lane reuse, and teardown without loading a model.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "backends/IWorkerGPUContext.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "execution/moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "utils/MPIContext.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kGpuOwnerRank = 1;
        constexpr int kCpuOwnerRank = 0;
        constexpr int kSecondaryRocmOrdinal = 1;
        constexpr std::uint32_t kUnitsPerChunk = 2;
        constexpr std::size_t kWaveCount = 32;

        /** @brief Bind the two-rank world communicator used by the data plane. */
        std::shared_ptr<MPIContext> worldContext()
        {
            int rank = -1;
            int world_size = 0;
            if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
                MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Could not inspect the ROCm remote-projection test world");
            }
            return std::make_shared<MPIContext>(
                rank, world_size, MPI_COMM_WORLD);
        }

        /** @brief Create deterministic, structurally valid separated Q8_0 bytes. */
        HostGpuExpertPackedProjection q8GpuProjection()
        {
            HostGpuExpertPackedProjection projection;
            projection.N = 128;
            projection.K = 1024;
            projection.blocks_per_row = projection.K / 32;
            projection.source_codebook_id = 19;
            projection.codebook_id = 19;
            projection.payload_bytes_per_block = 32;
            projection.is_asymmetric = false;
            projection.is_superblock = false;
            projection.has_emins = false;
            const std::size_t blocks =
                static_cast<std::size_t>(projection.N) *
                projection.blocks_per_row;
            projection.payload.resize(blocks * 32u);
            projection.scales.resize(blocks);
            for (std::size_t index = 0; index < projection.payload.size(); ++index)
            {
                projection.payload[index] =
                    static_cast<std::uint8_t>(
                        (index * 131u + (index >> 3u) * 17u + 29u) & 0xffu);
            }
            for (std::size_t index = 0; index < projection.scales.size(); ++index)
            {
                projection.scales[index] = static_cast<std::uint16_t>(
                    0x2400u + ((index * 19u + 7u) & 0x01ffu));
            }
            return projection;
        }

        /** @brief Present the one final CPU allocation as immutable regions. */
        std::array<std::span<const std::uint8_t>, 4> cpuSourceRegions(
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &weights)
        {
            return {
                std::span<const std::uint8_t>(
                    weights.native_interleaved.data(),
                    weights.native_interleaved.size()),
                {},
                {},
                {},
            };
        }

        /** @brief Present preallocated CPU arrival storage as writable regions. */
        std::array<std::span<std::uint8_t>, 4> cpuDestinationRegions(
            std::vector<std::uint8_t> &bytes)
        {
            return {
                std::span<std::uint8_t>(bytes.data(), bytes.size()),
                {},
                {},
                {},
            };
        }

        /**
         * @brief Own exact separated Q8_0 storage on one ROCm device.
         *
         * Allocation, upload, observation, and release all name the same device
         * and explicit stream.  This fixture deliberately contains no fallback
         * host representation beyond the immutable byte oracle supplied by the
         * caller.
         */
        class DeviceQ8Projection final
        {
        public:
            /** @brief Allocate every non-empty region in @p reference. */
            DeviceQ8Projection(
                IBackend &backend,
                DeviceId device,
                const HostGpuExpertPackedProjection &reference)
                : backend_(backend), device_(device), reference_(reference)
            {
                std::string error;
                if (!device_.is_rocm() || !reference_.valid(&error) ||
                    reference_.is_asymmetric || reference_.has_emins)
                {
                    throw std::invalid_argument(
                        error.empty()
                            ? "ROCm MPI Q8 fixture requires symmetric storage"
                            : std::move(error));
                }
                payload_ = static_cast<std::uint8_t *>(backend_.allocate(
                    reference_.payload.size(), device_.gpu_ordinal()));
                scales_ = static_cast<std::uint16_t *>(backend_.allocate(
                    reference_.scales.size() * sizeof(std::uint16_t),
                    device_.gpu_ordinal()));
                if (!payload_ || !scales_)
                {
                    throw std::runtime_error(
                        "Could not allocate secondary-ROCm Q8 projection");
                }
            }

            /** @brief Release device storage only after all endpoint owners die. */
            ~DeviceQ8Projection()
            {
                if (payload_)
                    backend_.free(payload_, device_.gpu_ordinal());
                if (scales_)
                    backend_.free(scales_, device_.gpu_ordinal());
            }

            DeviceQ8Projection(const DeviceQ8Projection &) = delete;
            DeviceQ8Projection &operator=(const DeviceQ8Projection &) = delete;

            /** @return Production descriptor for the exact allocation. */
            [[nodiscard]] GpuExpertPackedDescriptor descriptor() const noexcept
            {
                return {
                    .ptrs = {
                        .d_vnni = payload_,
                        .d_scales = scales_,
                    },
                    .n = reference_.N,
                    .k = reference_.K,
                    .blocks_per_row = reference_.blocks_per_row,
                    .codebook_id = reference_.codebook_id,
                    .payload_bytes_per_block =
                        reference_.payload_bytes_per_block,
                    .is_asymmetric = false,
                    .has_emins = false,
                    .vnni_bytes = reference_.payload.size(),
                    .scales_bytes =
                        reference_.scales.size() * sizeof(std::uint16_t),
                    .mins_bytes = 0,
                    .emins_bytes = 0,
                };
            }

            /** @brief Enqueue a complete host representation on @p stream. */
            bool upload(
                const HostGpuExpertPackedProjection &source,
                void *stream) noexcept
            {
                if (!stream || source.payload.size() != reference_.payload.size() ||
                    source.scales.size() != reference_.scales.size())
                {
                    return false;
                }
                return backend_.hostToDeviceOnStream(
                           payload_, source.payload.data(),
                           source.payload.size(), device_.gpu_ordinal(), stream) &&
                       backend_.hostToDeviceOnStream(
                           scales_, source.scales.data(),
                           source.scales.size() * sizeof(std::uint16_t),
                           device_.gpu_ordinal(), stream);
            }

            /** @brief Observe every final byte at a test-only boundary. */
            bool download(
                void *stream,
                HostGpuExpertPackedProjection *destination) noexcept
            {
                if (!stream || !destination)
                    return false;
                *destination = reference_;
                return backend_.deviceToHost(
                           destination->payload.data(), payload_,
                           destination->payload.size(), device_.gpu_ordinal(),
                           stream) &&
                       backend_.deviceToHost(
                           destination->scales.data(), scales_,
                           destination->scales.size() * sizeof(std::uint16_t),
                           device_.gpu_ordinal(), stream);
            }

        private:
            IBackend &backend_;
            DeviceId device_;
            HostGpuExpertPackedProjection reference_;
            std::uint8_t *payload_ = nullptr;
            std::uint16_t *scales_ = nullptr;
        };

        /** @brief Poll one explicit setup event without synchronizing a stream. */
        bool awaitStreamEdge(
            IBackend &backend,
            IWorkerGPUContext &context,
            DeviceId device,
            void *stream)
        {
            void *event = backend.createEvent(device.gpu_ordinal());
            if (!event || !backend.recordEvent(
                              event, device.gpu_ordinal(), stream))
            {
                if (event)
                    backend.destroyEvent(event, device.gpu_ordinal());
                return false;
            }
            bool ready = false;
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (!ready && std::chrono::steady_clock::now() < deadline)
            {
                if (!context.queryEventChecked(event, ready))
                    break;
                if (!ready)
                    std::this_thread::yield();
            }
            backend.destroyEvent(event, device.gpu_ordinal());
            return ready;
        }

        /** @brief Compare every separated Q8_0 region exactly. */
        void expectSameProjection(
            const HostGpuExpertPackedProjection &actual,
            const HostGpuExpertPackedProjection &expected)
        {
            EXPECT_EQ(actual.payload, expected.payload);
            EXPECT_EQ(actual.scales, expected.scales);
            EXPECT_EQ(actual.mins, expected.mins);
            EXPECT_EQ(actual.emins, expected.emins);
        }

        /** @brief Construct one direction of the repeated two-rank wave. */
        MoEOverlayRemoteProjectionIdentity projectionIdentity(
            std::uint64_t epoch,
            std::uint64_t migration_index,
            int source_rank,
            int destination_rank,
            DeviceId source_device,
            DeviceId destination_device)
        {
            return {
                .expected_epoch = epoch,
                .candidate_epoch = epoch + 1u,
                .execution_fingerprint = {
                    .low = 0x7100000000000000ull + epoch,
                    .high = 0x8200000000000000ull + epoch,
                },
                .migration_index = migration_index,
                .layer_idx = 7,
                .expert_id = static_cast<int>(11u + migration_index),
                .projection = migration_index == 0u
                    ? ExpertTierWeightProjection::Up
                    : ExpertTierWeightProjection::Down,
                .source_participant = source_rank,
                .destination_participant = destination_rank,
                .source_world_rank = source_rank,
                .destination_world_rank = destination_rank,
                .source_device = source_device,
                .destination_device = destination_device,
            };
        }
    } // namespace

    /**
     * @test Repeated bidirectional Q8 movement composes real MPI with ROCm:1.
     *
     * Both operations are live in each wave, matching a migration cycle that
     * demotes one expert while promoting another.  Exact comparisons and lane
     * counters prove work happened; zero wait/synchronization counters prove it
     * retained the asynchronous maintenance contract.
     */
    TEST(
        Test__MoEOverlayROCmRemoteProjectionMPI,
        SecondaryDeviceBidirectionalQ8WavesAreByteExactAndTeardownCleanly)
    {
        auto context = worldContext();
        if (context->world_size() != 2)
            GTEST_SKIP() << "ROCm remote projection requires exactly two ranks";

        IBackend *rocm = getROCmBackend();
        const int local_device_count = rocm ? rocm->deviceCount() : 0;
        int available_device_count = 0;
        ASSERT_EQ(
            MPI_Allreduce(
                &local_device_count, &available_device_count, 1, MPI_INT,
                MPI_MIN, MPI_COMM_WORLD),
            MPI_SUCCESS);
        if (available_device_count <= kSecondaryRocmOrdinal)
            GTEST_SKIP() << "The regression requires at least two ROCm devices";
        ASSERT_NE(rocm, nullptr);

        const HostGpuExpertPackedProjection expected_gpu = q8GpuProjection();
        cpu::native_vnni::CPUNativeVNNIPackedWeights cpu_weights;
        std::string error;
        ASSERT_TRUE(gpuToCpuExpertPackedReference(
            expected_gpu, cpu_weights, &error)) << error;
        HostGpuExpertPackedProjection round_trip_gpu;
        ASSERT_TRUE(cpuToGpuExpertPackedReference(
            cpu_weights, round_trip_gpu, &error)) << error;
        expectSameProjection(round_trip_gpu, expected_gpu);

        const DeviceId gpu_device = DeviceId::rocm(kSecondaryRocmOrdinal);
        std::unique_ptr<DeviceQ8Projection> gpu_source;
        std::unique_ptr<DeviceQ8Projection> gpu_destination;
        std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> source_lane;
        std::shared_ptr<MoEOverlayGpuRemoteProjectionLane> destination_lane;
        std::shared_ptr<ITensorGemm> destination_engine;
        std::shared_ptr<void> gpu_slot_lifetime;
        void *observation_stream = nullptr;

        const std::size_t cpu_block_stride = static_cast<std::size_t>(
            cpu_weights.interleaved_block_stride);
        const std::size_t staging_bytes =
            cpu_block_stride * kUnitsPerChunk;
        ASSERT_LE(staging_bytes, static_cast<std::size_t>(UINT32_MAX));

        if (context->rank() == kGpuOwnerRank)
        {
            gpu_source = std::make_unique<DeviceQ8Projection>(
                *rocm, gpu_device, expected_gpu);
            gpu_destination = std::make_unique<DeviceQ8Projection>(
                *rocm, gpu_device, expected_gpu);
            IWorkerGPUContext &gpu_context =
                GPUDeviceContextPool::instance().getContext(gpu_device);
            observation_stream = gpu_context.getOrCreateAuxiliaryStream(
                "rocm_mpi_remote_projection_observation");
            ASSERT_NE(observation_stream, nullptr);
            ASSERT_TRUE(gpu_source->upload(expected_gpu, observation_stream));
            ASSERT_TRUE(gpu_destination->upload(
                HostGpuExpertPackedProjection{
                    .N = expected_gpu.N,
                    .K = expected_gpu.K,
                    .blocks_per_row = expected_gpu.blocks_per_row,
                    .source_codebook_id = expected_gpu.source_codebook_id,
                    .codebook_id = expected_gpu.codebook_id,
                    .payload_bytes_per_block =
                        expected_gpu.payload_bytes_per_block,
                    .is_asymmetric = false,
                    .is_superblock = expected_gpu.is_superblock,
                    .has_emins = false,
                    .payload = std::vector<std::uint8_t>(
                        expected_gpu.payload.size(), 0xa5u),
                    .scales = std::vector<std::uint16_t>(
                        expected_gpu.scales.size(), 0x7bffu),
                },
                observation_stream));
            ASSERT_TRUE(awaitStreamEdge(
                *rocm, gpu_context, gpu_device, observation_stream));

            source_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = gpu_device,
                    .staging_capacity_bytes = staging_bytes,
                    .lane_name = "rocm1_mpi_q8_demotion",
                    .perf_device = "rocm:1",
                });
            destination_lane = std::make_shared<
                MoEOverlayGpuRemoteProjectionLane>(
                MoEOverlayGpuRemoteProjectionLane::Config{
                    .device = gpu_device,
                    .staging_capacity_bytes = staging_bytes,
                    .lane_name = "rocm1_mpi_q8_promotion",
                    .perf_device = "rocm:1",
                });
            ASSERT_TRUE(source_lane->materialize(&error)) << error;
            ASSERT_TRUE(destination_lane->materialize(&error)) << error;

            gpu_slot_lifetime = std::make_shared<int>(91);
            const auto descriptor = gpu_destination->descriptor();
            destination_engine = std::make_shared<
                rocm::ROCmQuantisedGemmKernel>(
                descriptor.n,
                descriptor.k,
                kSecondaryRocmOrdinal,
                descriptor.ptrs.d_vnni,
                descriptor.ptrs.d_scales,
                descriptor.ptrs.d_mins,
                descriptor.ptrs.d_emins,
                descriptor.codebook_id,
                descriptor.blocks_per_row,
                gpu_slot_lifetime,
                NativeVnniSourceIdentity{
                    .codebook_id = expected_gpu.source_codebook_id,
                    .is_superblock = expected_gpu.is_superblock,
                    .present = true,
                });
        }

        std::vector<std::uint8_t> demoted_cpu_bytes(
            cpu_weights.native_interleaved.size(), 0xa5u);
        const auto cpu_source_lifetime = std::make_shared<int>(37);
        const auto cpu_destination_lifetime = std::make_shared<int>(43);

        {
            MoEOverlayMPIRemoteProjectionTransport transport({
                .mpi_context = context,
                .lane_budget = {
                    .maximum_participants_per_cycle = 2,
                    .maximum_concurrent_cycles = 1,
                    .projections_per_expert = 1,
                },
                .staging_capacity_bytes = staging_bytes,
                .perf_device = "rocm1_mpi_remote_projection",
            });

            for (std::size_t wave_index = 0;
                 wave_index < kWaveCount;
                 ++wave_index)
            {
                const std::uint64_t epoch = 1000u + wave_index * 2u;
                const auto demotion_identity = projectionIdentity(
                    epoch,
                    0u,
                    kGpuOwnerRank,
                    kCpuOwnerRank,
                    gpu_device,
                    DeviceId::cpu());
                const auto promotion_identity = projectionIdentity(
                    epoch,
                    1u,
                    kCpuOwnerRank,
                    kGpuOwnerRank,
                    DeviceId::cpu(),
                    gpu_device);
                const auto *source_format =
                    native_vnni_formats::forSourceIdentity(
                        expected_gpu.source_codebook_id,
                        expected_gpu.is_superblock);
                ASSERT_NE(source_format, nullptr);
                const auto demotion_stream =
                    makeGpuToCpuExpertTierWeightStreamManifest(
                        *source_format,
                        expected_gpu.N,
                        expected_gpu.K,
                        demotion_identity.expected_epoch,
                        demotion_identity.layer_idx,
                        demotion_identity.expert_id,
                        demotion_identity.projection,
                        kUnitsPerChunk);
                const auto promotion_stream =
                    makeCpuToGpuExpertTierWeightStreamManifest(
                        cpu_weights,
                        promotion_identity.candidate_epoch,
                        promotion_identity.layer_idx,
                        promotion_identity.expert_id,
                        promotion_identity.projection,
                        kUnitsPerChunk);
                const auto demotion_manifest =
                    makeMoEOverlayRemoteCpuProjectionManifest(
                        demotion_identity,
                        demotion_stream,
                        static_cast<std::uint32_t>(staging_bytes));
                const auto promotion_manifest =
                    makeMoEOverlayRemoteCpuProjectionManifest(
                        promotion_identity,
                        promotion_stream,
                        static_cast<std::uint32_t>(staging_bytes));

                std::vector<MoEOverlayMPIRemoteProjectionBinding> bindings;
                bindings.reserve(2);
                if (context->rank() == kGpuOwnerRank)
                {
                    ASSERT_TRUE(rocm->setDevice(0));
                    auto gpu_source_endpoint = std::make_shared<
                        MoEOverlayGpuRemoteProjectionSource>(
                        demotion_manifest,
                        source_lane,
                        gpu_source->descriptor(),
                        ExpertTierSourceReadiness::publishedResidencyBank(
                            demotion_identity.expected_epoch),
                        gpu_slot_lifetime);
                    auto gpu_destination_endpoint = std::make_shared<
                        MoEOverlayGpuRemoteProjectionDestination>(
                        promotion_identity,
                        destination_lane,
                        [&, promotion_manifest](
                            const MoEOverlayRemoteProjectionManifest &received,
                            MoEOverlayGpuRemoteProjectionDestinationBinding *binding,
                            std::string *factory_error) -> bool
                        {
                            if (!binding || received != promotion_manifest)
                            {
                                if (factory_error)
                                {
                                    *factory_error =
                                        "ROCm MPI promotion factory received a different manifest";
                                }
                                return false;
                            }
                            *binding = {
                                .descriptor = gpu_destination->descriptor(),
                                .engine = destination_engine,
                            };
                            if (factory_error)
                                factory_error->clear();
                            return true;
                        },
                        gpu_slot_lifetime);
                    bindings.push_back({
                        .lane_index = 0,
                        .identity = demotion_identity,
                        .source = std::move(gpu_source_endpoint),
                    });
                    bindings.push_back({
                        .lane_index = 1,
                        .identity = promotion_identity,
                        .destination =
                            std::move(gpu_destination_endpoint),
                    });
                }
                else
                {
                    std::fill(
                        demoted_cpu_bytes.begin(),
                        demoted_cpu_bytes.end(),
                        0xa5u);
                    bindings.push_back({
                        .lane_index = 0,
                        .identity = demotion_identity,
                        .destination = std::make_shared<
                            MoEOverlayHostRemoteProjectionDestination>(
                            demotion_identity,
                            cpuDestinationRegions(demoted_cpu_bytes),
                            cpu_destination_lifetime,
                            demotion_manifest),
                    });
                    bindings.push_back({
                        .lane_index = 1,
                        .identity = promotion_identity,
                        .source = std::make_shared<
                            MoEOverlayHostRemoteProjectionSource>(
                            promotion_manifest,
                            cpuSourceRegions(cpu_weights),
                            cpu_source_lifetime),
                    });
                }

                auto wave = transport.reserveWave(std::move(bindings));
                ASSERT_EQ(
                    wave.status,
                    MoEOverlayResidencyStageStartStatus::Started) << wave.error;
                ASSERT_EQ(wave.operations.size(), 2u);

                std::array<bool, 2> ready{false, false};
                const auto deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(10);
                while ((!ready[0] || !ready[1]) &&
                       std::chrono::steady_clock::now() < deadline)
                {
                    for (std::size_t operation = 0;
                         operation < wave.operations.size();
                         ++operation)
                    {
                        if (ready[operation])
                            continue;
                        const auto progress =
                            wave.operations[operation]->poll(&error);
                        ASSERT_NE(
                            progress,
                            MoEOverlayResidencyWaveProgress::Failed) << error;
                        ready[operation] =
                            progress == MoEOverlayResidencyWaveProgress::Ready;
                    }
                    std::this_thread::yield();
                }
                ASSERT_TRUE(ready[0]);
                ASSERT_TRUE(ready[1]);
                wave.operations.clear();

                ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
                if (context->rank() == kCpuOwnerRank)
                {
                    EXPECT_TRUE(std::equal(
                        demoted_cpu_bytes.begin(),
                        demoted_cpu_bytes.end(),
                        cpu_weights.native_interleaved.begin(),
                        cpu_weights.native_interleaved.end()));
                }
                else
                {
                    HostGpuExpertPackedProjection observed;
                    ASSERT_TRUE(gpu_destination->download(
                        observation_stream, &observed));
                    expectSameProjection(observed, expected_gpu);
                }
            }

            const auto stats = transport.stats();
            EXPECT_EQ(stats.wave_reservations_started, kWaveCount);
            EXPECT_EQ(stats.operations_completed, kWaveCount * 2u);
            EXPECT_EQ(stats.active_mpi_requests, 0u);
            EXPECT_EQ(stats.mpi_failures, 0u);
            EXPECT_EQ(stats.protocol_failures, 0u);
            EXPECT_EQ(stats.inference_stream_waits, 0u);
            EXPECT_EQ(stats.blocking_synchronizations, 0u);
            ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
        }

        if (context->rank() == kGpuOwnerRank)
        {
            const auto source_stats = source_lane->stats();
            const auto destination_stats = destination_lane->stats();
            EXPECT_EQ(source_stats.gpu_to_cpu_repack_chunks,
                      destination_stats.cpu_to_gpu_repack_chunks);
            EXPECT_GT(source_stats.gpu_to_cpu_repack_chunks, kWaveCount);
            EXPECT_EQ(source_stats.inference_stream_waits, 0u);
            EXPECT_EQ(destination_stats.inference_stream_waits, 0u);
            EXPECT_EQ(source_stats.blocking_synchronizations, 0u);
            EXPECT_EQ(destination_stats.blocking_synchronizations, 0u);
        }

        /* Retire executable aliases, lanes, then their backing allocations. */
        destination_engine.reset();
        source_lane.reset();
        destination_lane.reset();
        gpu_destination.reset();
        gpu_source.reset();
        ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
    }
} // namespace llaminar2::test
