/**
 * @file Test__MoEOverlayHeterogeneousResidencyMPI.cpp
 * @brief Two-rank production-fabric proof for CUDA/ROCm/CPU tier movement.
 *
 * The test discovers the actual socket owning CUDA:0 and ROCm:0 instead of
 * assigning either accelerator to a fixed MPI rank.  One CPU participant is
 * bound to each socket, matching the production NodeTP cold tier.  A
 * coordinator-owned histogram then drives a four-participant rotation through
 * the real RCU authority, MPI histogram lane, two-phase consensus, remote
 * projection data plane, CPU/GPU repackers, prepared GEMM engines, and
 * background maintenance service.  Every published gate/up/down projection is
 * converted back to the same CPU execution representation and compared byte
 * exactly with its original expert oracle.
 */

#include "backends/BackendManager.h"
#include "backends/GPUDeviceContextPool.h"
#include "backends/IBackend.h"
#include "execution/moe/CpuExpertSlotPool.h"
#include "execution/moe/ExpertTierWeightStream.h"
#include "execution/moe/GPUExpertTransfer.h"
#include "execution/moe/GpuExpertSlotPool.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"
#include "execution/moe/MoEOverlayDistributedResidencyTransport.h"
#include "execution/moe/MoEOverlayMPIHistogramPublisher.h"
#include "execution/moe/MoEOverlayMPIRemoteProjectionTransport.h"
#include "execution/moe/MoEOverlayMPIResidencyConsensus.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayResidencyMaintenanceService.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#include "planning/ClusterInventoryGatherer.h"
#include "tensors/TensorKernels.h"
#include "utils/DebugEnv.h"
#include "utils/MPIContext.h"
#include "utils/PerfStatsCollector.h"

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace llaminar2::test
{
    namespace
    {
        constexpr int kLayer = 0;
        constexpr int kExperts = 4;
        constexpr std::size_t kProjectionCount = 3;
        constexpr std::size_t kStagingBytes = 4096;

        /** @brief Production MLP geometry for one compact integration expert. */
        struct ProjectionGeometry
        {
            ExpertTierWeightProjection projection;
            const char *label;
            int N;
            int K;
        };

        constexpr std::array<ProjectionGeometry, kProjectionCount>
            kProjectionGeometry{{
                {ExpertTierWeightProjection::Gate, "gate", 32, 64},
                {ExpertTierWeightProjection::Up, "up", 32, 64},
                {ExpertTierWeightProjection::Down, "down", 64, 32},
            }};

        using CpuExpertOracle = std::array<
            cpu::native_vnni::CPUNativeVNNIPackedWeights,
            kProjectionCount>;

        /** @brief Enable isolated migration counters for one MPI process. */
        class ScopedPerfStats final
        {
        public:
            /** @brief Save process state, enable collection, and clear records. */
            ScopedPerfStats()
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

            /** @brief Restore process state after all background work stops. */
            ~ScopedPerfStats()
            {
                PerfStatsCollector::reset();
                if (had_value_)
                    setenv(
                        "LLAMINAR_PERF_STATS_SUMMARY",
                        value_.c_str(),
                        1);
                else
                    unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
                mutableDebugEnv().reload();
            }

            ScopedPerfStats(const ScopedPerfStats &) = delete;
            ScopedPerfStats &operator=(const ScopedPerfStats &) = delete;

        private:
            bool had_value_ = false;
            std::string value_;
        };

        /** @brief Independent exact-stream copy used as an inference witness. */
        class GpuInferenceWitness final
        {
        public:
            /** @brief Allocate persistent buffers, stream, and completion event. */
            GpuInferenceWitness(DeviceId device, int world_rank)
                : device_(device), backend_(getBackendFor(device))
            {
                if (!backend_ || !device_.is_gpu())
                    throw std::invalid_argument(
                        "GPU inference witness requires a live GPU backend");
                auto &context =
                    GPUDeviceContextPool::instance().getContext(device_);
                stream_ = context.getOrCreateAuxiliaryStream(
                    "heterogeneous_overlay_mpi:inference:" +
                    std::to_string(world_rank) + ":" +
                    device_.to_string());
                source_ = backend_->allocate(kBytes, device_.gpu_ordinal());
                destination_ =
                    backend_->allocate(kBytes, device_.gpu_ordinal());
                event_ = backend_->createEvent(device_.gpu_ordinal());
                if (!stream_ || !source_ || !destination_ || !event_)
                    throw std::runtime_error(
                        "Could not materialize GPU inference witness resources");
            }

            /** @brief Release resources after the test has observed completion. */
            ~GpuInferenceWitness()
            {
                const int ordinal = device_.gpu_ordinal();
                if (event_)
                    backend_->destroyEvent(event_, ordinal);
                if (source_)
                    backend_->free(source_, ordinal);
                if (destination_)
                    backend_->free(destination_, ordinal);
            }

            GpuInferenceWitness(const GpuInferenceWitness &) = delete;
            GpuInferenceWitness &operator=(
                const GpuInferenceWitness &) = delete;

            /** @brief Enqueue witness work without touching a migration stream. */
            [[nodiscard]] bool start() noexcept
            {
                if (started_)
                    return false;
                started_ = backend_->deviceCopyAsync(
                               destination_,
                               source_,
                               kBytes,
                               device_.gpu_ordinal(),
                               stream_) &&
                           backend_->recordEvent(
                               event_, device_.gpu_ordinal(), stream_);
                return started_;
            }

            /** @brief Query the exact event without a host or stream wait. */
            [[nodiscard]] bool poll(bool &ready) noexcept
            {
                ready = false;
                return started_ && backend_->queryEvent(
                                       event_,
                                       device_.gpu_ordinal(),
                                       &ready);
            }

        private:
            static constexpr std::size_t kBytes = 256u * 1024u;
            DeviceId device_;
            IBackend *backend_ = nullptr;
            void *stream_ = nullptr;
            void *source_ = nullptr;
            void *destination_ = nullptr;
            void *event_ = nullptr;
            bool started_ = false;
        };

        /** @brief Bind the test to the exact MPI world supplied by CTest. */
        std::shared_ptr<MPIContext> worldContext()
        {
            int rank = -1;
            int world_size = 0;
            if (MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS ||
                MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Could not resolve the heterogeneous overlay MPI world");
            }
            return std::make_shared<MPIContext>(
                rank, world_size, MPI_COMM_WORLD);
        }

        /** @brief Return whether the gathered inventory exposes both GPU families. */
        bool hasRequiredAccelerators(const ClusterInventory &inventory)
        {
            bool cuda = false;
            bool rocm = false;
            for (const auto &rank : inventory.ranks)
            {
                for (const auto &gpu : rank.gpus)
                {
                    cuda = cuda || gpu.type == DeviceType::CUDA;
                    rocm = rocm || gpu.type == DeviceType::ROCm;
                }
            }
            return cuda && rocm;
        }

        /** @brief Build one rank-agnostic accelerator domain selector. */
        RoutedExpertDomain gpuDomain(
            std::string name,
            GlobalDeviceAddress participant,
            CollectiveBackendType backend)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.backend = backend;
            result.participants = {std::move(participant)};
            result.owner_rank = -1;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Build the two-socket NodeTP CPU cold domain. */
        RoutedExpertDomain cpuColdDomain()
        {
            RoutedExpertDomain result;
            result.name = "cpu_cold";
            result.scope = ExecutionDomainScope::NODE_LOCAL;
            result.backend = CollectiveBackendType::UPI;
            result.participants = {
                GlobalDeviceAddress::cpu(0),
                GlobalDeviceAddress::cpu(1),
            };
            result.owner_rank = -1;
            result.routed_compute_policy =
                RoutedExpertComputePolicy::Apportioned;
            return result;
        }

        /** @brief Construct one thermal tier with fixed or fallback capacity. */
        RoutedExpertTier tier(
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

        /** @brief Declare portable CUDA-hot/ROCm-warm/CPU-cold intent. */
        MoERoutedExpertPlacementPlan requestedPlan()
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
                gpuDomain(
                    "cuda_hot",
                    GlobalDeviceAddress::cuda(0),
                    CollectiveBackendType::NCCL),
                gpuDomain(
                    "rocm_warm",
                    GlobalDeviceAddress::rocm(0),
                    CollectiveBackendType::RCCL),
                cpuColdDomain(),
            };
            plan.routed_tiers = {
                tier("hot", "cuda_hot", 0, 1),
                tier("warm", "rocm_warm", 1, 1),
                tier("cold", "cpu_cold", 2, 0, true),
            };
            plan.placements = {{
                .layer = kLayer,
                .routed_expert_tier = {0, 1, 2, 2},
            }};
            return plan;
        }

        /** @brief Complete tier service fixture for physical protocol coverage. */
        std::shared_ptr<const MoERoutedTierServiceProfile> serviceProfile()
        {
            auto profile = std::make_shared<MoERoutedTierServiceProfile>();
            profile->identity = "heterogeneous-mpi-service-profile-v1";
            profile->costs = {
                {.tier_index = 0,
                 .layer = kLayer,
                 .nanoseconds_per_activation = {10, 20, 30}},
                {.tier_index = 1,
                 .layer = kLayer,
                 .nanoseconds_per_activation = {50, 100, 150}},
                {.tier_index = 2,
                 .layer = kLayer,
                 .nanoseconds_per_activation = {100, 200, 300}},
            };
            return profile;
        }

        /** @brief Complete cheap directed profile for all resolved participants. */
        std::shared_ptr<const MoEOverlayMigrationCostProfile>
        migrationProfile(const MoEExpertOwnerMap &owner_map)
        {
            auto profile =
                std::make_shared<MoEOverlayMigrationCostProfile>();
            profile->identity =
                "heterogeneous-mpi-migration-profile-v1";
            for (const auto &source : owner_map.participants())
            {
                for (const auto &destination : owner_map.participants())
                {
                    if (source.participant_id == destination.participant_id)
                        continue;
                    profile->costs.push_back({
                        .source_participant = source.participant_id,
                        .destination_participant =
                            destination.participant_id,
                        .layer = kLayer,
                        .transfer_and_repack_ns = 1,
                        .inference_interference_ns = 0,
                    });
                }
            }
            return profile;
        }

        /** @brief Return the source identity used by all three projections. */
        NativeVnniSourceIdentity sourceIdentity()
        {
            return {
                .codebook_id = native_vnni_formats::Q4_0.codebook_id,
                .is_superblock =
                    native_vnni_formats::Q4_0.is_superblock,
                .present = true,
            };
        }

        /** @brief Generate deterministic canonical GPU bytes for one projection. */
        HostGpuExpertPackedProjection canonicalGpuProjection(
            int expert_id,
            std::size_t projection_index)
        {
            const auto &geometry = kProjectionGeometry[projection_index];
            const auto &format = native_vnni_formats::Q4_0;
            HostGpuExpertPackedProjection result;
            result.N = geometry.N;
            result.K = geometry.K;
            result.blocks_per_row =
                static_cast<std::uint32_t>(geometry.K / 32);
            result.source_codebook_id = format.codebook_id;
            result.codebook_id =
                canonicalDeviceVnniCodebookId(format.codebook_id);
            result.payload_bytes_per_block =
                static_cast<std::uint8_t>(format.payload_bytes);
            result.is_asymmetric = format.is_asymmetric;
            result.is_superblock = format.is_superblock;
            result.has_emins = format.has_emins;

            const std::size_t blocks =
                static_cast<std::size_t>(result.N) *
                result.blocks_per_row;
            result.payload.resize(
                blocks * result.payload_bytes_per_block);
            result.scales.resize(blocks);
            if (result.is_asymmetric)
                result.mins.resize(blocks);
            if (result.has_emins)
                result.emins.resize(blocks);

            const std::uint32_t seed = static_cast<std::uint32_t>(
                17011 + expert_id * 997 + projection_index * 131);
            for (std::size_t block = 0; block < blocks; ++block)
            {
                for (std::size_t byte = 0;
                     byte < result.payload_bytes_per_block;
                     ++byte)
                {
                    result.payload[
                        block * result.payload_bytes_per_block + byte] =
                        static_cast<std::uint8_t>(
                            (seed + block * 29u + byte * 17u) & 0xffu);
                }
                result.scales[block] = static_cast<std::uint16_t>(
                    0x3000u + ((seed + block * 7u) & 0x03ffu));
                if (result.is_asymmetric)
                {
                    result.mins[block] = static_cast<std::uint16_t>(
                        0xb000u + ((seed + block * 11u) & 0x03ffu));
                }
                if (result.has_emins)
                    result.emins[block] = seed + block;
            }
            std::string error;
            if (!result.valid(&error))
                throw std::runtime_error(error);
            return result;
        }

        /** @brief Build the device-independent exact CPU oracle for every expert. */
        std::vector<CpuExpertOracle> buildOracles()
        {
            std::vector<CpuExpertOracle> result(kExperts);
            for (int expert = 0; expert < kExperts; ++expert)
            {
                for (std::size_t projection = 0;
                     projection < kProjectionCount;
                     ++projection)
                {
                    std::string error;
                    if (!gpuToCpuExpertPackedReference(
                            canonicalGpuProjection(expert, projection),
                            result[static_cast<std::size_t>(expert)]
                                  [projection],
                            &error))
                    {
                        throw std::runtime_error(error);
                    }
                }
            }
            return result;
        }

        /** @brief Publish one engine into its canonical gate/up/down field. */
        void setProjection(
            MoEOverlayPreparedExpertTriplet &triplet,
            ExpertTierWeightProjection projection,
            std::shared_ptr<ITensorGemm> engine)
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                triplet.gate = std::move(engine);
                return;
            case ExpertTierWeightProjection::Up:
                triplet.up = std::move(engine);
                return;
            case ExpertTierWeightProjection::Down:
                triplet.down = std::move(engine);
                return;
            }
            throw std::invalid_argument(
                "Unknown heterogeneous overlay projection role");
        }

        /** @brief Materialize one exact CPU source expert on its NUMA owner. */
        MoEOverlayPreparedExpertTriplet makeCpuExpert(
            const MoEExpertOwnerParticipant &participant,
            int expert_id,
            const CpuExpertOracle &oracle)
        {
            std::vector<CpuExpertSlotPool::ProjectionSpec> specs;
            for (const auto &geometry : kProjectionGeometry)
            {
                specs.push_back({
                    .projection = geometry.projection,
                    .N = geometry.N,
                    .K = geometry.K,
                    .format =
                        ExpertWeightFormat::nativeVnni(sourceIdentity()),
                });
            }
            auto pool = CpuExpertSlotPool::create({
                .participant_id = participant.participant_id,
                .layer_idx = kLayer,
                .capacity = 1,
                .projections = std::move(specs),
                .memory_placement =
                    CpuExpertSlotPool::MemoryPlacement::boundNode(
                        participant.address.numa_node),
                .perf_device = "heterogeneous-overlay-mpi",
            });
            auto lease = pool->acquire(expert_id, 1);
            if (!lease)
                throw std::runtime_error(
                    "Could not acquire initial CPU expert slot");

            MoEOverlayPreparedExpertTriplet result;
            for (const auto &projection : lease->projections)
            {
                const auto role = static_cast<std::size_t>(
                    projection.projection);
                if (role >= oracle.size() ||
                    projection.destination_bytes.size() !=
                        oracle[role].native_interleaved.size())
                {
                    throw std::runtime_error(
                        "Initial CPU expert oracle has incompatible storage");
                }
                std::copy(
                    oracle[role].native_interleaved.begin(),
                    oracle[role].native_interleaved.end(),
                    projection.destination_bytes.begin());
                setProjection(
                    result,
                    projection.projection,
                    projection.engine);
            }
            if (!result.complete())
                throw std::runtime_error(
                    "Initial CPU expert is incomplete");
            return result;
        }

        /** @brief Poll one setup upload event without a blocking synchronization. */
        void finishUpload(IBackend &backend, DeviceId device, void *stream)
        {
            void *event = backend.createEvent(device.gpu_ordinal());
            if (!event ||
                !backend.recordEvent(event, device.gpu_ordinal(), stream))
            {
                if (event)
                    backend.destroyEvent(event, device.gpu_ordinal());
                throw std::runtime_error(
                    "Could not record initial GPU expert upload event");
            }
            bool ready = false;
            const auto deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(5);
            while (!ready && std::chrono::steady_clock::now() < deadline)
            {
                if (!backend.queryEvent(
                        event, device.gpu_ordinal(), &ready))
                {
                    backend.destroyEvent(event, device.gpu_ordinal());
                    throw std::runtime_error(
                        "Initial GPU expert upload event query failed");
                }
                if (!ready)
                    std::this_thread::yield();
            }
            backend.destroyEvent(event, device.gpu_ordinal());
            if (!ready)
                throw std::runtime_error(
                    "Initial GPU expert upload did not complete");
        }

        /** @brief Materialize one actual CUDA or ROCm prepared source expert. */
        MoEOverlayPreparedExpertTriplet makeGpuExpert(
            const MoEExpertOwnerParticipant &participant,
            int expert_id,
            int world_rank)
        {
            IBackend *backend = getBackendFor(participant.device);
            if (!backend)
                throw std::runtime_error(
                    "Initial GPU expert has no backend");
            auto &context = GPUDeviceContextPool::instance().getContext(
                participant.device);
            void *stream = context.getOrCreateAuxiliaryStream(
                "heterogeneous_overlay_mpi:initial_upload:" +
                std::to_string(world_rank) + ":" +
                participant.device.to_string());
            if (!stream)
                throw std::runtime_error(
                    "Initial GPU expert has no upload stream");

            const auto identity = sourceIdentity();
            const auto &format = native_vnni_formats::Q4_0;
            const auto allocation =
                reusableDeviceVnniAllocationFormat(format);
            std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
            for (const auto &geometry : kProjectionGeometry)
            {
                specs.push_back({
                    .label = geometry.label,
                    .N = geometry.N,
                    .K = geometry.K,
                    .payload_bytes_per_block =
                        allocation.payload_bytes_per_block,
                    .is_asymmetric = allocation.has_mins,
                    .has_emins = allocation.has_emins,
                    .codebook_id =
                        canonicalDeviceVnniCodebookId(format.codebook_id),
                    .format = ExpertWeightFormat::nativeVnni(identity),
                });
            }
            auto pool = GpuExpertSlotPool::create(
                backend,
                participant.device,
                participant.device.gpu_ordinal(),
                kLayer,
                1,
                std::move(specs),
                0,
                0);
            auto lease = pool->acquire(expert_id, 1);
            if (!lease)
                throw std::runtime_error(
                    "Could not acquire initial GPU expert slot");

            MoEOverlayPreparedExpertTriplet result;
            for (std::size_t projection_index = 0;
                 projection_index < kProjectionCount;
                 ++projection_index)
            {
                const auto host = canonicalGpuProjection(
                    expert_id, projection_index);
                const auto &slot = lease->projections[projection_index];
                DeviceNativeVNNIMatrixDesc descriptor;
                descriptor.payload = slot.slot.d_native_vnni_payload;
                descriptor.scales = slot.slot.d_native_vnni_scales;
                descriptor.mins = slot.slot.d_native_vnni_mins;
                descriptor.emins = slot.slot.d_native_vnni_emins;
                descriptor.n = host.N;
                descriptor.k = host.K;
                descriptor.blocks_per_row = host.blocks_per_row;
                descriptor.codebook_id = host.codebook_id;
                descriptor.allocation_payload_bytes_per_block =
                    allocation.payload_bytes_per_block;
                descriptor.allocation_has_mins =
                    allocation.has_mins ? 1u : 0u;
                descriptor.allocation_has_emins =
                    allocation.has_emins ? 1u : 0u;
                descriptor.source_codebook_id = identity.codebook_id;
                descriptor.source_is_superblock =
                    identity.is_superblock ? 1u : 0u;
                descriptor.source_identity_present = 1u;
                const auto packed = makeGpuExpertPackedDescriptor(
                    descriptor,
                    host.payload_bytes_per_block,
                    host.is_asymmetric,
                    host.has_emins);
                if (!packed.valid() ||
                    !backend->hostToDeviceOnStream(
                        packed.ptrs.d_vnni,
                        host.payload.data(),
                        packed.vnni_bytes,
                        participant.device.gpu_ordinal(),
                        stream) ||
                    !backend->hostToDeviceOnStream(
                        packed.ptrs.d_scales,
                        host.scales.data(),
                        packed.scales_bytes,
                        participant.device.gpu_ordinal(),
                        stream))
                {
                    throw std::runtime_error(
                        "Initial GPU expert upload submission failed");
                }

                std::shared_ptr<ITensorGemm> engine;
#ifdef HAVE_CUDA
                if (participant.device.is_cuda())
                {
                    engine = std::make_shared<
                        cuda::CUDAQuantisedGemmKernel>(
                        packed.n,
                        packed.k,
                        participant.device.cuda_ordinal(),
                        packed.ptrs.d_vnni,
                        static_cast<std::uint16_t *>(
                            packed.ptrs.d_scales),
                        static_cast<std::uint16_t *>(packed.ptrs.d_mins),
                        static_cast<std::uint32_t *>(packed.ptrs.d_emins),
                        packed.codebook_id,
                        packed.blocks_per_row,
                        lease->lifetime,
                        identity,
                        allocation);
                }
#endif
#ifdef HAVE_ROCM
                if (participant.device.is_rocm())
                {
                    engine = std::make_shared<
                        rocm::ROCmQuantisedGemmKernel>(
                        packed.n,
                        packed.k,
                        participant.device.rocm_ordinal(),
                        packed.ptrs.d_vnni,
                        packed.ptrs.d_scales,
                        packed.ptrs.d_mins,
                        packed.ptrs.d_emins,
                        packed.codebook_id,
                        packed.blocks_per_row,
                        lease->lifetime,
                        identity,
                        allocation);
                }
#endif
                if (!engine)
                    throw std::runtime_error(
                        "Initial GPU expert backend was not built");
                setProjection(
                    result,
                    kProjectionGeometry[projection_index].projection,
                    std::move(engine));
            }
            finishUpload(*backend, participant.device, stream);
            if (!result.complete())
                throw std::runtime_error(
                    "Initial GPU expert is incomplete");
            return result;
        }

        /** @brief Register every initial process-local participant bank. */
        std::shared_ptr<MoEOverlayParticipantResidencyRegistry>
        makeRegistry(
            const std::shared_ptr<const MoEOverlayResidencySnapshot> &snapshot,
            int world_rank,
            const std::vector<CpuExpertOracle> &oracles)
        {
            std::vector<int> local_ids;
            for (const auto &participant : snapshot->owner_map.participants())
            {
                if (participant.world_rank_known &&
                    participant.world_rank == world_rank)
                {
                    local_ids.push_back(participant.participant_id);
                }
            }
            auto registry = std::make_shared<
                MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = local_ids,
                    .num_layers = 1,
                    .num_experts = kExperts,
                    .initial_epoch = snapshot->epoch,
                    .retained_epoch_capacity = 2,
                });

            for (const int participant_id : local_ids)
            {
                const auto *participant =
                    snapshot->owner_map.participantForId(participant_id);
                if (!participant)
                    throw std::logic_error(
                        "Initial registry lost a local participant");
                const auto mask = snapshot->owner_map.expertMaskForParticipant(
                    kLayer, participant_id, kExperts);
                std::vector<MoEOverlayPreparedExpertTriplet> engines(
                    kExperts);
                for (int expert = 0; expert < kExperts; ++expert)
                {
                    if (!mask[static_cast<std::size_t>(expert)])
                        continue;
                    engines[static_cast<std::size_t>(expert)] =
                        participant->device.is_cpu()
                            ? makeCpuExpert(
                                  *participant,
                                  expert,
                                  oracles[static_cast<std::size_t>(expert)])
                            : makeGpuExpert(
                                  *participant, expert, world_rank);
                }
                std::string error;
                if (!registry->registerInitialLayer(
                        participant_id,
                        kLayer,
                        mask,
                        engines,
                        &error))
                {
                    throw std::runtime_error(error);
                }
            }
            if (!registry->allInitialBanksReady())
                throw std::runtime_error(
                    "Initial heterogeneous registry is incomplete");
            return registry;
        }

        /** @brief Publish exact model geometry for initially remote destinations. */
        std::vector<MoEOverlayLayerWeightManifest> layerManifest()
        {
            MoEOverlayLayerWeightManifest layer;
            layer.layer_idx = kLayer;
            for (std::size_t projection = 0;
                 projection < kProjectionCount;
                 ++projection)
            {
                const auto &geometry = kProjectionGeometry[projection];
                layer.projections[projection] = {
                    .projection = geometry.projection,
                    .N = geometry.N,
                    .K = geometry.K,
                    .format =
                        ExpertWeightFormat::nativeVnni(sourceIdentity()),
                };
            }
            return {std::move(layer)};
        }

        /** @brief Download one GPU engine and normalize it to final CPU bytes. */
        cpu::native_vnni::CPUNativeVNNIPackedWeights engineAsCpu(
            const std::shared_ptr<ITensorGemm> &engine,
            DeviceId device,
            int world_rank)
        {
            if (!engine)
                throw std::invalid_argument(
                    "Cannot inspect a null prepared engine");
            if (device.is_cpu())
            {
                const auto *packed =
                    engine->exportCPUNativeVNNIPackedWeights();
                if (!packed)
                    throw std::runtime_error(
                        "CPU destination cannot export final weights");
                return *packed;
            }

            DeviceNativeVNNIMatrixDesc descriptor;
            NativeVnniSourceIdentity identity;
            if (!engine->exportNativeVNNIMatrixDesc(descriptor) ||
                !engine->exportNativeVNNISourceIdentity(identity))
            {
                throw std::runtime_error(
                    "GPU destination cannot export its physical contract");
            }
            const auto *source = native_vnni_formats::forSourceIdentity(
                identity.codebook_id, identity.is_superblock);
            if (!source)
                throw std::runtime_error(
                    "GPU destination lost source provenance");

            std::uint8_t payload_bytes = 0;
            bool asymmetric = false;
            bool has_emins = false;
            if (descriptor.codebook_id ==
                canonicalDeviceVnniCodebookId(source->codebook_id))
            {
                payload_bytes =
                    static_cast<std::uint8_t>(source->payload_bytes);
                asymmetric = source->is_asymmetric;
                has_emins = source->has_emins;
            }
            else
            {
                payload_bytes = 32;
                asymmetric = source->is_asymmetric;
                has_emins = false;
            }
            const auto packed = makeGpuExpertPackedDescriptor(
                descriptor, payload_bytes, asymmetric, has_emins);
            if (!packed.valid())
                throw std::runtime_error(
                    "GPU destination descriptor is invalid");

            HostGpuExpertPackedProjection host;
            host.N = packed.n;
            host.K = packed.k;
            host.blocks_per_row = packed.blocks_per_row;
            host.source_codebook_id = identity.codebook_id;
            host.codebook_id = packed.codebook_id;
            host.payload_bytes_per_block = packed.payload_bytes_per_block;
            host.is_asymmetric = packed.is_asymmetric;
            host.is_superblock = identity.is_superblock;
            host.has_emins = packed.has_emins;
            host.payload.resize(packed.vnni_bytes);
            host.scales.resize(
                packed.scales_bytes / sizeof(std::uint16_t));
            host.mins.resize(
                packed.mins_bytes / sizeof(std::uint16_t));
            host.emins.resize(
                packed.emins_bytes / sizeof(std::uint32_t));

            IBackend *backend = getBackendFor(device);
            auto &context =
                GPUDeviceContextPool::instance().getContext(device);
            void *stream = context.getOrCreateAuxiliaryStream(
                "heterogeneous_overlay_mpi:diagnostic:" +
                std::to_string(world_rank) + ":" + device.to_string());
            if (!backend || !stream ||
                !backend->deviceToHost(
                    host.payload.data(),
                    packed.ptrs.d_vnni,
                    packed.vnni_bytes,
                    device.gpu_ordinal(),
                    stream) ||
                !backend->deviceToHost(
                    host.scales.data(),
                    packed.ptrs.d_scales,
                    packed.scales_bytes,
                    device.gpu_ordinal(),
                    stream) ||
                (packed.mins_bytes != 0 &&
                 !backend->deviceToHost(
                     host.mins.data(),
                     packed.ptrs.d_mins,
                     packed.mins_bytes,
                     device.gpu_ordinal(),
                     stream)) ||
                (packed.emins_bytes != 0 &&
                 !backend->deviceToHost(
                     host.emins.data(),
                     packed.ptrs.d_emins,
                     packed.emins_bytes,
                     device.gpu_ordinal(),
                     stream)))
            {
                throw std::runtime_error(
                    "Diagnostic GPU destination download failed");
            }

            cpu::native_vnni::CPUNativeVNNIPackedWeights result;
            std::string error;
            if (!gpuToCpuExpertPackedReference(host, result, &error))
                throw std::runtime_error(error);
            return result;
        }

        /** @brief Compare complete CPU execution layouts and every final byte. */
        bool exactCpuWeights(
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &actual,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &expected)
        {
            return actual.N == expected.N && actual.K == expected.K &&
                   actual.N_padded == expected.N_padded &&
                   actual.blocks_per_row == expected.blocks_per_row &&
                   actual.codebook_id == expected.codebook_id &&
                   actual.is_asymmetric == expected.is_asymmetric &&
                   actual.is_superblock == expected.is_superblock &&
                   actual.encoding == expected.encoding &&
                   actual.data_stride == expected.data_stride &&
                   actual.interleaved_block_stride ==
                       expected.interleaved_block_stride &&
                   actual.native_interleaved.size() ==
                       expected.native_interleaved.size() &&
                   std::equal(
                       actual.native_interleaved.begin(),
                       actual.native_interleaved.end(),
                       expected.native_interleaved.begin());
        }

        /** @brief Sum one unsigned counter over the exact two-rank world. */
        std::uint64_t globalSum(
            std::uint64_t local,
            MPI_Comm communicator)
        {
            std::uint64_t result = 0;
            if (MPI_Allreduce(
                    &local,
                    &result,
                    1,
                    MPI_UINT64_T,
                    MPI_SUM,
                    communicator) != MPI_SUCCESS)
            {
                throw std::runtime_error(
                    "Heterogeneous overlay evidence reduction failed");
            }
            return result;
        }

        /** @brief Sum matching process-local PerfStats counter records. */
        double localPerfCounter(const std::string &name)
        {
            double result = 0.0;
            for (const auto &record :
                 PerfStatsCollector::snapshot({"moe_overlay_residency"}))
            {
                if (record.kind == PerfStatRecord::Kind::Counter &&
                    record.name == name)
                {
                    result += record.value;
                }
            }
            return result;
        }
    } // namespace

    TEST(
        Test__MoEOverlayHeterogeneousResidencyMPI,
        TopologyBoundHistogramRotationPublishesExactWeightsWithoutInferenceWaits)
    {
#if !defined(HAVE_CUDA) || !defined(HAVE_ROCM)
        GTEST_SKIP() << "CUDA and ROCm builds are both required";
#else
        auto context = worldContext();
        if (context->world_size() != 2)
            GTEST_SKIP() << "The production cold tier requires two socket ranks";

        ScopedPerfStats perf_stats;
        const auto inventory = gatherClusterInventory(context);
        if (!hasRequiredAccelerators(inventory))
            GTEST_SKIP() << "One topology-visible CUDA and ROCm device are required";

        auto plan = bindMoEExpertOverlayPlanToClusterInventory(
            requestedPlan(), inventory);
        ASSERT_NE(plan, nullptr);
        ASSERT_EQ(plan->domains.size(), 3u);
        ASSERT_EQ(plan->domains[0].world_ranks.size(), 1u);
        ASSERT_EQ(plan->domains[1].world_ranks.size(), 1u);
        EXPECT_GE(plan->domains[0].world_ranks.front(), 0);
        EXPECT_LT(
            plan->domains[0].world_ranks.front(), context->world_size());
        EXPECT_GE(plan->domains[1].world_ranks.front(), 0);
        EXPECT_LT(
            plan->domains[1].world_ranks.front(), context->world_size());
        EXPECT_EQ(plan->domains[2].world_ranks, (std::vector<int>{0, 1}));

        const MoEExpertOwnerMap initial_owner_map =
            MoEExpertOwnerMap::build(*plan);
        DecodeExpertHistogramConfig histogram_config;
        histogram_config.num_layers = 1;
        histogram_config.num_experts = kExperts;
        histogram_config.top_k = 1;
        histogram_config.window_size = 4;
        histogram_config.token_boundary_layer_idx = kLayer;
        for (const auto &participant : initial_owner_map.participants())
            histogram_config.sockets.push_back(participant.device);
        histogram_config.ownership = initial_owner_map.layeredOwnership(
            1, kExperts);
        auto histogram = std::make_shared<DecodeExpertHistogram>(
            std::move(histogram_config));

        const int coordinator_rank =
            plan->domains[0].world_ranks.front();
        if (context->rank() == coordinator_rank)
        {
            const std::uint64_t counts[]{70, 60, 50, 100};
            histogram->mergeLayerCounts(0, counts, kExperts, false);
            histogram->recordTokenBoundary(kLayer, 4);
            ASSERT_TRUE(histogram->windowFull());
        }
        else
        {
            ASSERT_FALSE(histogram->windowFull());
        }

        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *plan,
                .model_metadata = {
                    .num_layers = 1,
                    .num_experts = kExperts,
                    .d_model = 64,
                    .routed_intermediate_size = 32,
                    .routed_quant_type = "Q4_0",
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
                .histogram = histogram.get(),
                .phase_service_profile = serviceProfile(),
                .migration_cost_profile =
                    migrationProfile(initial_owner_map),
                .migration_economy_policy =
                    MoEOverlayMigrationEconomyPolicy{
                        .historical_window_weight = 0,
                        .current_window_weight = 1,
                        .payoff_horizon_tokens = 4,
                        .minimum_residency_generations = 0,
                    },
                .shadow_slots_per_endpoint_layer = 1,
                .max_concurrent_cycles = 1,
                .perf_device =
                    "cuda-hot/rocm-warm/cpu-nodelocal-cold",
            });
        const auto initial_snapshot = authority->snapshot();
        ASSERT_NE(initial_snapshot, nullptr);
        ASSERT_EQ(initial_snapshot->epoch, 1u);

        const auto oracles = buildOracles();
        auto registry = makeRegistry(
            initial_snapshot, context->rank(), oracles);

        /* Private communicator creation order matches production runner setup. */
        auto consensus = std::make_shared<MoEOverlayMPIResidencyConsensus>(
            MoEOverlayMPIResidencyConsensus::Config{
                .mpi_context = context,
                .perf_device =
                    "cuda-hot/rocm-warm/cpu-nodelocal-cold",
            });
        auto remote =
            std::make_shared<MoEOverlayMPIRemoteProjectionTransport>(
                MoEOverlayMPIRemoteProjectionTransport::Config{
                    .mpi_context = context,
                    .lane_budget = {
                        .maximum_participants_per_cycle =
                            initial_snapshot->owner_map
                                .participants().size(),
                        .maximum_concurrent_cycles = 1,
                        .projections_per_expert = kProjectionCount,
                    },
                    .staging_capacity_bytes = kStagingBytes,
                    .perf_device =
                        "cuda-hot/rocm-warm/cpu-nodelocal-cold",
                });
        auto publisher = std::make_shared<MoEOverlayMPIHistogramPublisher>(
            MoEOverlayMPIHistogramPublisher::Config{
                .mpi_context = context,
                .coordinator_world_rank = coordinator_rank,
                .num_layers = 1,
                .num_experts = kExperts,
                .perf_device =
                    "cuda-hot/rocm-warm/cpu-nodelocal-cold",
            });
        auto fabric = MoEOverlayPhysicalResidencyFabric::create({
            .registry = registry,
            .initial_snapshot = initial_snapshot,
            .layer_weight_manifest = layerManifest(),
            .remote_projection_transport = remote,
            .shadow_slots_per_endpoint_layer = 1,
            .staging_capacity_bytes = kStagingBytes,
            .gpu_vram_safety_margin_bytes = 0,
            .perf_device =
                "cuda-hot/rocm-warm/cpu-nodelocal-cold",
        });
        auto factory = std::make_unique<
            MoEOverlayParticipantPreparedWaveFactory>(
            MoEOverlayParticipantPreparedWaveFactory::Config{
                .registry = registry,
                .transfer_provider = fabric,
                .perf_device =
                    "cuda-hot/rocm-warm/cpu-nodelocal-cold",
            });
        auto local_transport =
            std::make_shared<MoEOverlayTierMigrationTransport>(
                MoEOverlayTierMigrationTransport::Config{
                    .factory = factory.get(),
                    .projections_per_expert = kProjectionCount,
                    .perf_device =
                        "cuda-hot/rocm-warm/cpu-nodelocal-cold",
                });
        auto distributed =
            std::make_shared<MoEOverlayDistributedResidencyTransport>(
                MoEOverlayDistributedResidencyTransport::Config{
                    .local_transport = local_transport.get(),
                    .consensus = consensus,
                    .perf_device =
                        "cuda-hot/rocm-warm/cpu-nodelocal-cold",
                });

        std::vector<std::unique_ptr<GpuInferenceWitness>> witnesses;
        for (const auto &participant :
             initial_snapshot->owner_map.participants())
        {
            if (participant.world_rank == context->rank() &&
                participant.device.is_gpu())
            {
                witnesses.push_back(
                    std::make_unique<GpuInferenceWitness>(
                        participant.device, context->rank()));
            }
        }

        context->barrier();
        auto service = std::make_unique<
            MoEOverlayResidencyMaintenanceService>(
            MoEOverlayResidencyMaintenanceService::Config{
                .authority = authority,
                .transport = distributed,
                .histogram_publisher = publisher,
                .idle_poll_interval = std::chrono::microseconds(100),
                .perf_device =
                    "cuda-hot/rocm-warm/cpu-nodelocal-cold",
            });
        for (auto &witness : witnesses)
            ASSERT_TRUE(witness->start());

        std::vector<bool> witness_ready(witnesses.size(), false);
        bool local_overlap_observed = witnesses.empty();
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (std::chrono::steady_clock::now() < deadline &&
               service->healthy() &&
               authority->snapshot()->epoch != 2u)
        {
            bool every_local_witness_ready = true;
            for (std::size_t index = 0;
                 index < witnesses.size();
                 ++index)
            {
                bool ready = witness_ready[index];
                ASSERT_TRUE(witnesses[index]->poll(ready));
                witness_ready[index] = ready;
                every_local_witness_ready =
                    every_local_witness_ready && ready;
            }
            if (every_local_witness_ready &&
                authority->snapshot()->epoch == 1u)
            {
                local_overlap_observed = true;
            }
            std::this_thread::yield();
        }

        ASSERT_TRUE(service->healthy()) << service->failureMessage();
        ASSERT_EQ(authority->snapshot()->epoch, 2u)
            << service->failureMessage();
        for (std::size_t index = 0; index < witnesses.size(); ++index)
        {
            bool ready = witness_ready[index];
            while (!ready && std::chrono::steady_clock::now() < deadline)
            {
                ASSERT_TRUE(witnesses[index]->poll(ready));
                if (!ready)
                    std::this_thread::yield();
            }
            EXPECT_TRUE(ready);
        }
        EXPECT_TRUE(local_overlap_observed)
            << "Local inference-stream work did not complete before residency publication";

        service->stopAndDrain();
        ASSERT_TRUE(service->healthy()) << service->failureMessage();
        const auto service_stats = service->stats();
        EXPECT_EQ(service_stats.proposals, 1u);
        EXPECT_EQ(service_stats.committed_waves, 1u);
        service.reset();
        publisher->stopAndDrain();

        const auto candidate = authority->snapshot();
        ASSERT_NE(candidate, nullptr);
        ASSERT_EQ(candidate->epoch, 2u);
        EXPECT_EQ(candidate->owner_map.ownerFor(0, 3)->tier_name, "hot");
        EXPECT_EQ(candidate->owner_map.ownerFor(0, 0)->tier_name, "warm");

        for (const int participant_id : registry->localParticipantIds())
        {
            const auto *participant =
                candidate->owner_map.participantForId(participant_id);
            ASSERT_NE(participant, nullptr);
            const auto bank = registry->endpoint(participant_id)->acquire(2);
            ASSERT_NE(bank, nullptr);
            const auto expected_mask =
                candidate->owner_map.expertMaskForParticipant(
                    kLayer, participant_id, kExperts);
            EXPECT_EQ(bank->layers[0].resident_mask, expected_mask);
            for (int expert = 0; expert < kExperts; ++expert)
            {
                if (!expected_mask[static_cast<std::size_t>(expert)])
                    continue;
                const auto &arrived = bank->layers[0]
                                          .experts[static_cast<std::size_t>(
                                              expert)];
                ASSERT_TRUE(arrived.complete());
                const std::array<std::shared_ptr<ITensorGemm>, 3> engines{
                    arrived.gate,
                    arrived.up,
                    arrived.down,
                };
                for (std::size_t projection = 0;
                     projection < engines.size();
                     ++projection)
                {
                    const auto actual = engineAsCpu(
                        engines[projection],
                        participant->device,
                        context->rank());
                    EXPECT_TRUE(exactCpuWeights(
                        actual,
                        oracles[static_cast<std::size_t>(expert)]
                               [projection]))
                        << "participant=" << participant_id
                        << " expert=" << expert
                        << " projection=" << projection;
                }
            }
        }

        const auto authority_stats = authority->stats();
        EXPECT_EQ(authority_stats.committed_waves, 1u);
        EXPECT_GE(authority_stats.committed_migrations, 3u);
        EXPECT_EQ(
            authority_stats.committed_migrations,
            authority_stats.promotions + authority_stats.demotions +
                authority_stats.same_priority_moves);
        EXPECT_EQ(authority_stats.committed_cycles, 1u);
        EXPECT_EQ(authority_stats.promotions, 1u);
        EXPECT_EQ(authority_stats.demotions, 2u);
        EXPECT_EQ(authority_stats.cross_domain_migrations, 3u);
        EXPECT_GE(authority_stats.cross_rank_migrations, 2u);
        EXPECT_EQ(authority_stats.cross_backend_migrations, 3u);

        const auto fabric_stats = fabric->stats();
        EXPECT_EQ(fabric_stats.waves_prepared, 1u);
        EXPECT_EQ(
            fabric_stats.projection_operations_prepared,
            authority_stats.committed_migrations * kProjectionCount);
        EXPECT_EQ(fabric_stats.inference_stream_waits, 0u);
        EXPECT_EQ(fabric_stats.blocking_synchronizations, 0u);
        const auto remote_stats = remote->stats();
        EXPECT_EQ(remote_stats.wave_reservations_started, 1u);
        EXPECT_GT(remote_stats.operations_completed, 0u);
        EXPECT_EQ(remote_stats.mpi_failures, 0u);
        EXPECT_EQ(remote_stats.protocol_failures, 0u);
        EXPECT_EQ(remote_stats.inference_stream_waits, 0u);
        EXPECT_EQ(remote_stats.blocking_synchronizations, 0u);
        const auto distributed_stats = distributed->stats();
        EXPECT_EQ(distributed_stats.waves_published, 1u);
        EXPECT_EQ(distributed_stats.reservation_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.stage_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.commit_consensus_ready, 1u);
        EXPECT_EQ(distributed_stats.inference_thread_waits, 0u);
        EXPECT_EQ(distributed_stats.blocking_synchronizations, 0u);

        const auto global_bytes_sent = globalSum(
            remote_stats.bytes_sent, context->communicator());
        const auto global_bytes_received = globalSum(
            remote_stats.bytes_received, context->communicator());
        EXPECT_GT(global_bytes_sent, 0u);
        EXPECT_EQ(global_bytes_sent, global_bytes_received);
        EXPECT_EQ(
            localPerfCounter(
                "remote_projection_payload_bytes_completed"),
            static_cast<double>(
                remote_stats.bytes_sent + remote_stats.bytes_received));
        EXPECT_GT(localPerfCounter("remote_projection_wall_ns"), 0.0);
        EXPECT_GT(
            localPerfCounter(
                "remote_projection_admission_to_dispatch_ns"),
            0.0);
        EXPECT_GT(localPerfCounter("remote_projection_host_work_ns"), 0.0);
        EXPECT_GT(
            localPerfCounter(
                "remote_projection_mpi_request_visible_ns"),
            0.0);
        EXPECT_EQ(
            globalSum(
                fabric_stats.inference_stream_waits,
                context->communicator()),
            0u);
        EXPECT_EQ(
            globalSum(
                fabric_stats.blocking_synchronizations,
                context->communicator()),
            0u);
        EXPECT_EQ(
            localPerfCounter("committed_expert_migrations"),
            static_cast<double>(authority_stats.committed_migrations));
        EXPECT_EQ(localPerfCounter("promotions"), 1.0);
        EXPECT_EQ(localPerfCounter("demotions"), 2.0);
        EXPECT_EQ(localPerfCounter("cross_domain_migrations"), 3.0);

        EXPECT_TRUE(consensus->idle());
#endif
    }
} // namespace llaminar2::test
