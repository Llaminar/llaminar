/**
 * @file MoEOverlayPhysicalResidencyFabricInternal.h
 * @brief Shared internal lane pools, copy operations, and Impl definition for MoEOverlayPhysicalResidencyFabric shards.
 */
#pragma once
/**
 * @file MoEOverlayPhysicalResidencyFabric.cpp
 * @brief Local slot reservation and parallel physical movement for ExpertOverlay.
 *
 * One maintenance worker polls every operation in a wave. Each directed
 * device edge owns an admission-sized persistent lane pool for gate, up, and
 * down. Every independent operation reserves its own lane before the first
 * poll, so the worker submits the complete wave without software queueing or
 * runtime stream/staging allocation.
 * Completed lifecycle diagnostics fold epoch identities into bounded evidence;
 * an indefinitely reused fabric cannot accumulate one statistics key per wave.
 * Device-owned whole/cohort waves retain member-typed payloads in one physical
 * ledger. Terminal compaction borrows that inventory instead of stale host banks.
 */

#include "MoEOverlayPhysicalResidencyFabric.h"
#include "backends/NativeContextPreparation.h"
#include "MoEOverlayPhysicalProjectionPieces.h"
#include "MoEOverlayPhysicalLaneDemand.h"
#include "MoEOverlayPhysicalRemotePieceOrdinals.h"
#include "MoEExpertOverlayPreparationPlan.h"

#include "CpuExpertSlotPool.h"
#include "ExpertPreparedMemoryGeometry.h"
#include "ExpertTierGpuBlobTransferLane.h"
#include "ExpertTierGpuPeerTransferLane.h"
#include "ExpertTierWeightTransferLane.h"
#include "GpuExpertSlotPool.h"
#include "MoEOverlayDevicePhysicalSlotLedger.h"
#include "MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "MoEOverlayMPIRemoteProjectionTransport.h"
#include "MoEOverlayPreparedWeightSource.h"
#include "MoEPhysicalResidencyPerfStats.h"
#include "loaders/ModelLoader.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/CPUNativeQuantisedGemmKernel.h"
#include "memory/NUMAAllocator.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "utils/PerfStatsCollector.h"

#ifdef HAVE_CUDA
#include "kernels/cuda/gemm/CUDAFloatingPointGemmKernel.h"
#include "kernels/cuda/gemm/CUDAQuantisedGemmKernel.h"
#endif
#ifdef HAVE_ROCM
#include "kernels/rocm/gemm/ROCmFloatingPointGemmKernel.h"
#include "kernels/rocm/gemm/ROCmQuantisedGemmKernel.h"
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <span>
#include <stdexcept>
#include <tuple>
#include <thread>
#include <utility>
#include <variant>

namespace llaminar2
{
    namespace
    {
        /** Gate/up/down roles in the canonical wire and bank order. */
        constexpr std::array<ExpertTierWeightProjection, 3> kProjections{
            ExpertTierWeightProjection::Gate,
            ExpertTierWeightProjection::Up,
            ExpertTierWeightProjection::Down,
        };

        /** @brief Stable lowercase label used for pool and lane identities. */
        const char *projectionName(
            ExpertTierWeightProjection projection) noexcept
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return "gate";
            case ExpertTierWeightProjection::Up:
                return "up";
            case ExpertTierWeightProjection::Down:
                return "down";
            }
            return "unknown";
        }

        /** @brief Select one retained projection from a complete expert payload. */
        std::shared_ptr<ITensorGemm> projectionEngine(
            const MoEOverlayPreparedExpertPayload &payload,
            ExpertTierWeightProjection projection)
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return payload.gate();
            case ExpertTierWeightProjection::Up:
                return payload.up();
            case ExpertTierWeightProjection::Down:
                return payload.down();
            }
            throw std::invalid_argument(
                "ExpertOverlay physical fabric received an unknown projection role");
        }

        /**
         * @brief Select a genuine engine from either closed participant-bank arm.
         * @param layer Existing immutable member bank, never a raw source mirror.
         * @param expert Original full-model logical expert coordinate.
         * @param projection Original semantic role.
         * @return Actual prepared engine retaining its original physical root.
         */
        std::shared_ptr<ITensorGemm> projectionEngine(
            const MoEOverlayParticipantLayerBank &layer, std::size_t expert,
            ExpertTierWeightProjection projection)
        {
            if (!layer.fragment_family)
                return projectionEngine(layer.experts.at(expert), projection);
            const auto &fragment = layer.fragments.at(expert);
            if (!layer.fragment_family->accepts(fragment, static_cast<int>(expert)))
                throw std::invalid_argument("Physical source fragment changed its original member family");
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate: return fragment.gate();
            case ExpertTierWeightProjection::Up: return fragment.up();
            case ExpertTierWeightProjection::Down: return fragment.down();
            }
            throw std::invalid_argument("Unknown original fragment projection role");
        }

        /** @brief Original authored full-model dimensions; no scalar owner shadow. */
        struct OriginalFabricGeometry
        {
            int first_layer, layers, experts;
            int firstModelLayer() const noexcept { return first_layer; }
            int layerCount() const noexcept { return layers; }
            int expertCount() const noexcept { return experts; }
        };
        /**
         * @brief Read the constructor-authenticated original geometry arm.
         * @param config Same fabric source, preparation and snapshot authorities.
         * @return Original full E and layer interval, including an empty tier quota.
         * @throws std::invalid_argument For missing or inconsistent original preparation/plan.
         */
        OriginalFabricGeometry originalFabricGeometry(
            const MoEOverlayPhysicalResidencyFabric::Config &config)
        {
            const auto &snapshot = *config.initial_snapshot;
            if (!snapshot.valid())
                throw std::invalid_argument("Physical fabric requires an authenticated source snapshot");
            const OriginalFabricGeometry geometry{
                snapshot.firstModelLayer(), snapshot.layerCount(), snapshot.expertCount()};
            if (snapshot.ownershipKind() == MoEOverlayResidencyOwnershipKind::WholeExperts)
                return geometry;
            const auto source = snapshot.originalProjectionPreparation();
            const auto &registered = config.registry->projectionPreparation();
            if (!source || !registered || source.get() != registered.get() ||
                source.owner_before(registered) || registered.owner_before(source) ||
                !source->jointOwnerMap())
                throw std::invalid_argument("Physical cohort fabric lost its same original preparation or full source plan");
            // The snapshot's sealed original definition owns geometry; the
            // fabric authenticates its registry binding rather than reparsing
            // a second live placement into a parallel shape authority.
            source->jointOwnerMap()->requireLayerGeometry(
                geometry.layers, geometry.experts, geometry.first_layer);
            return geometry;
        }

        /** @brief Participant/layer identity for one inactive destination pool. */
        struct EndpointLayerKey
        {
            int participant_id = -1;
            int layer_idx = -1;

            bool operator<(const EndpointLayerKey &other) const noexcept
            {
                return std::tie(participant_id, layer_idx) <
                       std::tie(other.participant_id, other.layer_idx);
            }
        };

        /** @brief Directed physical lane and projection identity. */
        struct EdgeKey
        {
            DeviceId source = DeviceId::invalid();
            DeviceId destination = DeviceId::invalid();
            ExpertTierWeightProjection projection =
                ExpertTierWeightProjection::Gate;

            bool operator<(const EdgeKey &other) const noexcept
            {
                return std::tie(
                           source.type,
                           source.ordinal,
                           destination.type,
                           destination.ordinal,
                           projection) <
                       std::tie(
                           other.source.type,
                           other.source.ordinal,
                           other.destination.type,
                           other.destination.ordinal,
                           other.projection);
            }
        };

        /** @brief Directed CPU participant-address and projection identity. */
        struct CpuAddressEdgeKey
        {
            GlobalDeviceAddress source;
            GlobalDeviceAddress destination;
            ExpertTierWeightProjection projection =
                ExpertTierWeightProjection::Gate;

            /** @brief Order CPU lanes by complete host, NUMA, and role identity. */
            bool operator<(const CpuAddressEdgeKey &other) const noexcept
            {
                return std::tie(source, destination, projection) <
                       std::tie(
                           other.source,
                           other.destination,
                           other.projection);
            }
        };

        /** Whether a persistent cross-rank GPU lane prepares or consumes bytes. */
        enum class RemoteGpuLaneRole : std::uint8_t
        {
            Source,
            Destination,
        };

        /** GPU/projection/role identity independent of remote rank placement. */
        struct RemoteGpuLaneKey
        {
            DeviceId device = DeviceId::invalid();
            ExpertTierWeightProjection projection =
                ExpertTierWeightProjection::Gate;
            RemoteGpuLaneRole role = RemoteGpuLaneRole::Source;

            /** @brief Provide deterministic map ordering over the full key. */
            bool operator<(const RemoteGpuLaneKey &other) const noexcept
            {
                return std::tie(device.type, device.ordinal, projection, role) <
                       std::tie(
                           other.device.type,
                           other.device.ordinal,
                           other.projection,
                           other.role);
            }
        };

        /**
         * @brief One reusable lane with explicit single-operation ownership.
         *
         * The maintenance worker is normally the only caller, but a mutex makes
         * abort/teardown races structurally safe and keeps future distributed
         * progress threads from relying on that scheduling assumption.
         */
        template <typename Lane>
        class SharedLane final
        {
        public:
            /** @brief Retain one completely materialized physical lane. */
            explicit SharedLane(std::shared_ptr<Lane> lane)
                : lane_(std::move(lane))
            {
                if (!lane_)
                    throw std::invalid_argument(
                        "ExpertOverlay shared lane requires physical ownership");
            }

            /** @brief Try to assign the lane to one exact admitted operation. */
            bool tryAcquire(const void *owner) noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (!owner || owner_)
                    return false;
                owner_ = owner;
                return true;
            }

            /** @brief Release only the operation that currently owns the lane. */
            void release(const void *owner) noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (owner_ != owner)
                    std::terminate();
                owner_ = nullptr;
            }

            /** @return Retained physical lane. */
            [[nodiscard]] const std::shared_ptr<Lane> &lane() const noexcept
            {
                return lane_;
            }

        private:
            std::shared_ptr<Lane> lane_;
            std::mutex mutex_;
            const void *owner_ = nullptr;
        };

        /**
         * @brief Setup-owned pool that reserves one distinct lane per operation.
         *
         * Pool exhaustion after the residency authority admitted a wave is a
         * configuration/accounting defect, never ordinary backpressure. The
         * operation constructors therefore reserve from this pool before any
         * transfer can start and fail the complete wave if no lane is free.
         */
        template <typename Lane>
        class SharedLanePool final
        {
        public:
            /** @brief Append one fully materialized lane during model setup. */
            void add(std::shared_ptr<Lane> lane)
            {
                lanes_.push_back(
                    std::make_shared<SharedLane<Lane>>(std::move(lane)));
            }

            /** @brief Reserve a unique lane for an admitted operation. */
            [[nodiscard]] std::shared_ptr<SharedLane<Lane>> tryAcquire(
                const void *owner) noexcept
            {
                for (const auto &lane : lanes_)
                {
                    if (lane->tryAcquire(owner))
                    {
                        reservations_.fetch_add(1, std::memory_order_relaxed);
                        return lane;
                    }
                }
                exhaustions_.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }

            /** @return Exact setup-time lane capacity. */
            [[nodiscard]] std::size_t capacity() const noexcept
            {
                return lanes_.size();
            }

            /** @return Successful distinct-lane reservations. */
            [[nodiscard]] std::uint64_t reservations() const noexcept
            {
                return reservations_.load(std::memory_order_relaxed);
            }

            /** @return Failed reservations, each a fatal accounting defect. */
            [[nodiscard]] std::uint64_t exhaustions() const noexcept
            {
                return exhaustions_.load(std::memory_order_relaxed);
            }

        private:
            std::vector<std::shared_ptr<SharedLane<Lane>>> lanes_;
            std::atomic<std::uint64_t> reservations_{0};
            std::atomic<std::uint64_t> exhaustions_{0};
        };

        /** @brief Barrier that releases every CPU copy in one wave together. */
        class CpuCopyWaveGate final
        {
        public:
            /** @brief Create a gate for the exact admitted CPU operation count. */
            explicit CpuCopyWaveGate(std::size_t expected_jobs)
                : expected_jobs_(expected_jobs)
            {
                if (expected_jobs_ == 0)
                    throw std::invalid_argument(
                        "ExpertOverlay CPU copy wave gate requires positive fan-out");
            }

            /**
             * @brief Record one operation's immutable job before waking its lane.
             * @param error Optional exact lifecycle diagnostic.
             * @return False if the wave was cancelled or over-armed.
             */
            bool armJob(std::string *error = nullptr) noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (cancelled_ || released_ || armed_jobs_ >= expected_jobs_)
                {
                    if (error)
                        *error =
                            "ExpertOverlay CPU copy wave gate rejected an extra or cancelled job";
                    return false;
                }
                ++armed_jobs_;
                releaseIfComplete();
                cv_.notify_all();
                return true;
            }

            /**
             * @brief Park one dedicated worker until the complete fan-out is ready.
             * @return True only when every admitted job and worker reached the gate.
             */
            bool workerArriveAndWait() noexcept
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (cancelled_ || workers_ready_ >= expected_jobs_ ||
                    workers_ready_ >= armed_jobs_)
                {
                    return false;
                }
                ++workers_ready_;
                releaseIfComplete();
                if (released_)
                    cv_.notify_all();
                cv_.wait(
                    lock,
                    [&] { return released_ || cancelled_; });
                return released_ && !cancelled_;
            }

            /** @brief Release parked workers without copying after wave abort. */
            void cancel() noexcept
            {
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    cancelled_ = true;
                }
                cv_.notify_all();
            }

            /** @return Exact operation count used for admission and release. */
            [[nodiscard]] std::size_t expectedJobs() const noexcept
            {
                return expected_jobs_;
            }

        private:
            /** @brief Open the barrier only after jobs and workers are complete. */
            void releaseIfComplete() noexcept
            {
                if (armed_jobs_ == expected_jobs_ &&
                    workers_ready_ == expected_jobs_)
                {
                    released_ = true;
                }
            }

            const std::size_t expected_jobs_;
            std::mutex mutex_;
            std::condition_variable cv_;
            std::size_t armed_jobs_ = 0;
            std::size_t workers_ready_ = 0;
            bool released_ = false;
            bool cancelled_ = false;
        };

        /** @brief Shared proof of real simultaneous CPU worker activity. */
        class CpuCopyConcurrencyTracker final
        {
        public:
            /** @brief Enter the active set and retain its model-lifetime peak. */
            void begin() noexcept
            {
                const std::uint64_t active =
                    active_.fetch_add(1, std::memory_order_acq_rel) + 1;
                std::uint64_t observed = peak_.load(std::memory_order_relaxed);
                while (observed < active &&
                       !peak_.compare_exchange_weak(
                           observed,
                           active,
                           std::memory_order_relaxed,
                           std::memory_order_relaxed))
                {
                }
            }

            /** @brief Leave the active set after copy or cancellation. */
            void end() noexcept
            {
                if (active_.fetch_sub(1, std::memory_order_acq_rel) == 0)
                    std::terminate();
            }

            /** @return Workers currently armed or copying. */
            [[nodiscard]] std::uint64_t active() const noexcept
            {
                return active_.load(std::memory_order_acquire);
            }

            /** @return Largest simultaneously armed worker set observed. */
            [[nodiscard]] std::uint64_t peak() const noexcept
            {
                return peak_.load(std::memory_order_relaxed);
            }

        private:
            std::atomic<std::uint64_t> active_{0};
            std::atomic<std::uint64_t> peak_{0};
        };

        /** @brief Full contiguous final storage, with no representation padding. */
        struct CompleteCPUCopy {};
        /** @brief Same original native intersection in two independent final roots. */
        struct NativeCPUCopy
        {
            ExpertTierWeightCPUStorageLayout source, destination;
            std::shared_ptr<std::mutex> encoded_write_mutex;
        };
        /** @brief Exact floating row spans, retaining each root's original stride. */
        struct FloatingCPUCopy
        {
            ExpertTierWeightProjectionIntersection intersection;
            size_t element_bytes;
        };
        using CPUCopyGeometry = std::variant<CompleteCPUCopy, NativeCPUCopy, FloatingCPUCopy>;

        /** @brief Quote only actual logical/encoded bytes for one existing worker job. */
        size_t cpuCopyBytes(const CPUCopyGeometry &shape, size_t complete_bytes)
        {
            return std::visit([&](const auto &copy) -> size_t {
                using T = std::decay_t<decltype(copy)>;
                if constexpr (std::is_same_v<T, CompleteCPUCopy>) return complete_bytes;
                else if constexpr (std::is_same_v<T, NativeCPUCopy>)
                    return copy.source.chunkBytes(copy.source.unit_count);
                else return size_t(copy.intersection.rows()) * copy.intersection.columns() * copy.element_bytes;
            }, shape);
        }

        /** @brief Validate final root extents before reserving or releasing a worker wave. */
        bool validCPUCopy(const CPUCopyGeometry &shape, std::span<const uint8_t> source,
            std::span<uint8_t> destination)
        {
            if (source.empty() || destination.empty()) return false;
            const auto x = reinterpret_cast<uintptr_t>(source.data()), y = reinterpret_cast<uintptr_t>(destination.data());
            if (!x || !y || source.size() > std::numeric_limits<uintptr_t>::max() - x ||
                destination.size() > std::numeric_limits<uintptr_t>::max() - y ||
                (x < y + destination.size() && y < x + source.size())) return false;
            return std::visit([&](const auto &copy) {
                using T = std::decay_t<decltype(copy)>;
                if constexpr (std::is_same_v<T, CompleteCPUCopy>) return source.size() == destination.size();
                else if constexpr (std::is_same_v<T, NativeCPUCopy>)
                    return copy.source.valid() && copy.destination.valid() && copy.encoded_write_mutex &&
                        copy.source.N == copy.destination.N && copy.source.K == copy.destination.K &&
                        copy.source.cpu_codebook_id == copy.destination.cpu_codebook_id &&
                        copy.source.cpu_is_asymmetric == copy.destination.cpu_is_asymmetric &&
                        copy.source.cpu_is_superblock == copy.destination.cpu_is_superblock &&
                        copy.source.cpu_encoding == copy.destination.cpu_encoding &&
                        copy.source.wire_columns_per_unit == copy.destination.wire_columns_per_unit &&
                        source.size() == copy.source.cpuStorageBytes() && destination.size() == copy.destination.cpuStorageBytes();
                else
                {
                    const auto &a = copy.intersection.source(), &b = copy.intersection.destination();
                    return !copy.intersection.empty() && (copy.element_bytes == 2 || copy.element_bytes == 4) &&
                        size_t(a.rows) * a.columns * copy.element_bytes == source.size() &&
                        size_t(b.rows) * b.columns * copy.element_bytes == destination.size();
                }
            }, shape);
        }

        /**
         * @brief Copy one bounded live chunk with no intermediate payload allocation.
         * @param shape Typed original layout, shared by admission and this actual job.
         * @param source Immutable original final source root.
         * @param destination Existing final candidate root.
         * @param offset Previously copied LIVE bytes, not a resident-root offset.
         * @param capacity Maximum useful bytes this worker may copy before cancellation.
         * @return Actual copied bytes; native chunk boundaries remain complete encoded units.
         */
        size_t copyCPUChunk(const CPUCopyGeometry &shape, std::span<const uint8_t> source,
            std::span<uint8_t> destination, size_t offset, size_t capacity)
        {
            return std::visit([&](const auto &copy) -> size_t {
                using T = std::decay_t<decltype(copy)>;
                if constexpr (std::is_same_v<T, CompleteCPUCopy>)
                {
                    const auto bytes = std::min(capacity, source.size() - offset);
                    std::memcpy(destination.data() + offset, source.data() + offset, bytes);
                    return bytes;
                }
                else if constexpr (std::is_same_v<T, NativeCPUCopy>)
                {
                    const auto first = expertTierWireUnitAt(copy.source, offset);
                    const auto units = std::min({copy.source.unit_count - first,
                        copy.source.maximum_units_per_chunk, copy.destination.maximum_units_per_chunk});
                    const auto bytes = copy.source.chunkBytes(units, first);
                    if (units == 0 || bytes > capacity ||
                        !copyExpertTierCPUWeightChunk(copy.source, source, copy.destination, destination,
                            first, units, copy.encoded_write_mutex.get()))
                        throw std::runtime_error("Physical CPU worker rejected its original native intersection");
                    return bytes;
                }
                else
                {
                    const auto &overlap = copy.intersection;
                    const size_t row_bytes = size_t(overlap.columns()) * copy.element_bytes;
                    const size_t row = offset / row_bytes, column_byte = offset % row_bytes;
                    const size_t bytes = std::min(capacity, row_bytes - column_byte);
                    const size_t from = ((size_t(overlap.sourceRowOrigin()) + row) * overlap.source().columns +
                        overlap.sourceColumnOrigin()) * copy.element_bytes + column_byte;
                    const size_t to = ((size_t(overlap.destinationRowOrigin()) + row) * overlap.destination().columns +
                        overlap.destinationColumnOrigin()) * copy.element_bytes + column_byte;
                    std::memcpy(destination.data() + to, source.data() + from, bytes);
                    return bytes;
                }
            }, shape);
        }

        /**
         * @brief Bind actual prepared native roots to the canonical CPU-only plane mapper.
         * @param source Actual immutable CPU prepared source metadata.
         * @param destination Actual final inactive CPU engine metadata.
         * @param intersection Same original overlap used by the wave's piece record.
         * @param capacity Existing worker bounded-copy quantum.
         * @param write_mutex Same final-role native write authority shared by all pieces.
         * @return Validated independent source/destination strides without staging.
         */
        NativeCPUCopy nativeCPUCopyGeometry(
            const cpu::native_quantised::CPUNativeQuantisedPackedWeights &source,
            const cpu::native_quantised::CPUNativeQuantisedPackedWeights &destination,
            const ExpertTierWeightProjectionIntersection &intersection, size_t capacity,
            std::shared_ptr<std::mutex> write_mutex)
        {
            const auto format = [&](const auto &root) {
                ExpertTierWeightCPUStorageLayout traits{
                    .cpu_codebook_id = root.codebook_id,
                    .cpu_is_asymmetric = uint8_t(root.is_asymmetric),
                    .cpu_is_superblock = uint8_t(root.is_superblock),
                    .cpu_encoding = root.encoding, .cpu_data_stride = uint32_t(root.data_stride),
                    .cpu_block_stride = uint32_t(root.interleaved_block_stride)};
                const size_t column_bytes = traits.cpu_block_stride / 64;
                if (!column_bytes || capacity < column_bytes || capacity > UINT32_MAX)
                    throw std::invalid_argument("CPU native worker capacity cannot contain one exact encoded column");
                traits.wire_columns_per_unit = unsigned(std::min(size_t(64), capacity / column_bytes));
                return traits;
            };
            if (source.N != intersection.source().rows || source.K != intersection.source().columns ||
                destination.N != intersection.destination().rows || destination.K != intersection.destination().columns)
                throw std::invalid_argument("CPU native worker roots changed original local dimensions");
            return {cpuExpertTierStorageLayoutForIntersection(format(source), intersection.source(), intersection, uint32_t(capacity)),
                cpuExpertTierStorageLayoutForIntersection(format(destination), intersection.destination(), intersection, uint32_t(capacity)),
                std::move(write_mutex)};
        }

        /** @brief One setup-owned asynchronous CPU/NUMA projection-copy worker. */
        class CpuCopyLane final
        {
        public:
            /**
             * @brief Immutable worker placement and copy geometry.
             *
             * Every lane owns a dedicated `SCHED_IDLE` thread. It never borrows
             * an inference executor or a CPU GEMM/OpenMP worker, so runnable
             * inference work always has scheduler priority over maintenance.
             */
            struct Config
            {
                int destination_numa_node = -1;
                std::size_t chunk_bytes = 0;
                std::string lane_name;
                std::string perf_device;
                std::shared_ptr<CpuCopyConcurrencyTracker> concurrency;
            };

            /** @brief Validate the lane identity without starting its worker. */
            explicit CpuCopyLane(Config config)
                : config_(std::move(config))
            {
                if (config_.chunk_bytes == 0 || config_.lane_name.empty() ||
                    config_.perf_device.empty() || !config_.concurrency)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU copy lane requires complete setup ownership");
                }
            }

            /** @brief Stop and join only after the last asynchronous job drained. */
            ~CpuCopyLane()
            {
                if (!quiescent())
                    std::terminate();
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    stopping_ = true;
                }
                cv_.notify_all();
                if (worker_.joinable())
                {
                    worker_.request_stop();
                    worker_.join();
                }
            }

            CpuCopyLane(const CpuCopyLane &) = delete;
            CpuCopyLane &operator=(const CpuCopyLane &) = delete;

            /**
             * @brief Start the persistent worker and prove its NUMA affinity.
             * @param error Optional exact setup diagnostic.
             * @return True only when the worker is ready to accept jobs.
             */
            bool materialize(std::string *error = nullptr) noexcept
            {
                std::unique_lock<std::mutex> lock(mutex_);
                if (initialized_)
                {
                    if (!initialization_ok_ && error)
                        *error = failure_;
                    return initialization_ok_;
                }
                if (worker_.joinable())
                {
                    if (error)
                        *error =
                            "ExpertOverlay CPU copy lane has a partial worker lifecycle";
                    return false;
                }
                try
                {
                    worker_ = std::jthread(
                        [this](std::stop_token stop_token)
                        { workerLoop(stop_token); });
                    initialized_cv_.wait(
                        lock,
                        [&] { return initialized_; });
                }
                catch (const std::exception &exception)
                {
                    failure_ = exception.what();
                    initialized_ = true;
                    initialization_ok_ = false;
                }
                catch (...)
                {
                    failure_ =
                        "ExpertOverlay CPU copy worker creation threw a non-standard exception";
                    initialized_ = true;
                    initialization_ok_ = false;
                }
                if (!initialization_ok_ && error)
                    *error = failure_;
                return initialization_ok_;
            }

            /**
             * @brief Submit one immutable source/final-destination copy job.
             * @param source Exact source bytes retained by @p source_engine.
             * @param destination Exact untouched candidate-slot bytes.
             * @param gate Whole-wave launch barrier shared by every CPU job.
             * @param source_engine Source physical-lifetime pin.
             * @param destination_engine Destination physical-lifetime pin.
             * @param error Optional exact submission diagnostic.
             * @return True when the dedicated worker owns the complete job.
             */
            bool start(
                std::span<const std::uint8_t> source,
                std::span<std::uint8_t> destination,
                std::shared_ptr<CpuCopyWaveGate> gate,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                std::string *error = nullptr, CPUCopyGeometry geometry = CompleteCPUCopy{}) noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const State state = state_.load(std::memory_order_acquire);
                if (!initialization_ok_ || stopping_ || !validCPUCopy(geometry, source, destination) || !gate ||
                    !source_engine || !destination_engine ||
                    state == State::Pending || state == State::Failed)
                {
                    if (error)
                        *error =
                            "ExpertOverlay CPU copy lane rejected incomplete or overlapping work";
                    return false;
                }

                std::string gate_error;
                if (!gate->armJob(&gate_error))
                {
                    failure_ = std::move(gate_error);
                    state_.store(State::Failed, std::memory_order_release);
                    if (error)
                        *error = failure_;
                    return false;
                }

                pending_job_.emplace(Job{
                    .source = source,
                    .destination = destination,
                    .geometry = std::move(geometry),
                    .gate = std::move(gate),
                    .source_engine = std::move(source_engine),
                    .destination_engine = std::move(destination_engine),
                });
                active_gate_ = pending_job_->gate;
                cancel_requested_.store(false, std::memory_order_release);
                failure_.clear();
                measurement_.reset();
                state_.store(State::Pending, std::memory_order_release);
                cv_.notify_one();
                return true;
            }

            /** @brief Query worker state without joining or waiting. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error = nullptr) noexcept
            {
                const State state = state_.load(std::memory_order_acquire);
                if (state == State::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (state == State::Ready)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (error)
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    *error = failure_.empty()
                        ? "ExpertOverlay CPU copy lane is not ready"
                        : failure_;
                }
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Request cooperative cancellation at the next chunk edge. */
            void requestAbort() noexcept
            {
                cancel_requested_.store(true, std::memory_order_release);
                std::shared_ptr<CpuCopyWaveGate> gate;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    gate = active_gate_;
                }
                if (gate)
                    gate->cancel();
            }

            /** @return Whether no worker can still access the current buffers. */
            [[nodiscard]] bool quiescent() const noexcept
            {
                return state_.load(std::memory_order_acquire) != State::Pending;
            }

            /** @return Exact completed host-copy timing, when available. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return measurement_;
            }

        private:
            /** @brief Host-visible lifecycle for one persistent CPU worker. */
            enum class State : std::uint8_t
            {
                Idle,
                Pending,
                Ready,
                Cancelled,
                Failed,
            };

            /** @brief Pointer-stable copy job retained until worker completion. */
            struct Job
            {
                std::span<const std::uint8_t> source;
                std::span<std::uint8_t> destination;
                CPUCopyGeometry geometry;
                std::shared_ptr<CpuCopyWaveGate> gate;
                std::shared_ptr<ITensorGemm> source_engine;
                std::shared_ptr<ITensorGemm> destination_engine;
            };

            /** @brief Initialize affinity, then execute one admitted job at a time. */
            void workerLoop(std::stop_token stop_token) noexcept
            {
                sched_param background_priority{};
                const int scheduler_error = pthread_setschedparam(
                    pthread_self(), SCHED_IDLE, &background_priority);
                bool affinity_ok = scheduler_error == 0;
                if (config_.destination_numa_node >= 0)
                {
                    affinity_ok = affinity_ok &&
                        NUMAAllocator::instance().bindThreadToNode(
                            config_.destination_numa_node);
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    initialized_ = true;
                    initialization_ok_ = affinity_ok;
                    if (!affinity_ok)
                    {
                        failure_ = scheduler_error != 0
                            ? "ExpertOverlay CPU copy worker could not enter SCHED_IDLE: " +
                                  std::string(std::strerror(scheduler_error))
                            : "ExpertOverlay CPU copy worker could not bind to destination NUMA node " +
                                  std::to_string(
                                      config_.destination_numa_node);
                        state_.store(State::Failed, std::memory_order_release);
                    }
                }
                initialized_cv_.notify_all();
                if (!affinity_ok)
                    return;

                while (!stop_token.stop_requested())
                {
                    std::optional<Job> job;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        cv_.wait(
                            lock,
                            [&]
                            {
                                return stopping_ ||
                                       pending_job_.has_value();
                            });
                        if (stopping_ || stop_token.stop_requested())
                            return;
                        job = std::move(pending_job_);
                        pending_job_.reset();
                    }

                    config_.concurrency->begin();
                    bool concurrency_active = true;
                    const auto end_concurrency = [&]() noexcept
                    {
                        if (!concurrency_active)
                            return;
                        config_.concurrency->end();
                        concurrency_active = false;
                    };
                    const bool released = job->gate->workerArriveAndWait();
                    const auto started_at = std::chrono::steady_clock::now();
                    std::uint64_t host_nanoseconds = 0;
                    std::size_t offset = 0;
                    const auto live_bytes = cpuCopyBytes(job->geometry, job->source.size());
                    bool cancelled = !released;
                    try
                    {
                        while (!cancelled && offset < live_bytes)
                        {
                            cancelled = cancel_requested_.load(
                                std::memory_order_acquire);
                            if (cancelled)
                                break;
                            const auto copy_started_at = std::chrono::steady_clock::now();
                            const auto bytes = copyCPUChunk(job->geometry, job->source, job->destination,
                                offset, config_.chunk_bytes);
                            if (bytes == 0 || bytes > live_bytes - offset)
                                throw std::runtime_error("CPU projection worker failed to advance exact live bytes");
                            const auto copy_elapsed =
                                std::chrono::duration_cast<
                                    std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() -
                                    copy_started_at)
                                    .count();
                            host_nanoseconds =
                                saturatingExpertTierMeasurementAdd(
                                    host_nanoseconds,
                                    static_cast<std::uint64_t>(
                                        std::max<std::int64_t>(
                                            1, copy_elapsed)));
                            offset += bytes;
                        }

                        const auto wall_elapsed =
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - started_at)
                                .count();
                        if (!cancelled)
                        {
                            PerfStatsCollector::addCounter(
                                "moe_overlay_residency",
                                "cpu_native_copy_bytes",
                                static_cast<double>(live_bytes),
                                "maintenance",
                                config_.perf_device,
                                {{"lane", config_.lane_name},
                                 {"destination_numa",
                                  std::to_string(
                                      config_.destination_numa_node)}});
                        }
                        /* Ready publication is the proof that no worker remains. */
                        end_concurrency();
                        {
                            std::lock_guard<std::mutex> lock(mutex_);
                            active_gate_.reset();
                            if (cancelled)
                            {
                                failure_ =
                                    "ExpertOverlay CPU copy was cancelled before publication";
                                state_.store(
                                    State::Cancelled,
                                    std::memory_order_release);
                            }
                            else
                            {
                                measurement_ = {
                                    .sequence =
                                        sequence_.fetch_add(
                                            1,
                                            std::memory_order_relaxed) +
                                        1,
                                    .bytes = static_cast<std::uint64_t>(
                                        live_bytes),
                                    .wall_nanoseconds =
                                        static_cast<std::uint64_t>(
                                            std::max<std::int64_t>(
                                                1, wall_elapsed)),
                                    .device_nanoseconds = 0,
                                    .host_nanoseconds = host_nanoseconds,
                                };
                                state_.store(
                                    State::Ready,
                                    std::memory_order_release);
                            }
                        }
                    }
                    catch (const std::exception &exception)
                    {
                        end_concurrency();
                        std::lock_guard<std::mutex> lock(mutex_);
                        active_gate_.reset();
                        failure_ = exception.what();
                        state_.store(State::Failed, std::memory_order_release);
                    }
                    catch (...)
                    {
                        end_concurrency();
                        std::lock_guard<std::mutex> lock(mutex_);
                        active_gate_.reset();
                        failure_ =
                            "ExpertOverlay CPU copy worker threw a non-standard exception";
                        state_.store(State::Failed, std::memory_order_release);
                    }
                }
            }

            Config config_;
            mutable std::mutex mutex_;
            std::condition_variable cv_;
            std::condition_variable initialized_cv_;
            std::jthread worker_;
            std::optional<Job> pending_job_;
            std::shared_ptr<CpuCopyWaveGate> active_gate_;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::string failure_;
            std::atomic<State> state_{State::Idle};
            std::atomic<bool> cancel_requested_{false};
            std::atomic<std::uint64_t> sequence_{0};
            bool initialized_ = false;
            bool initialization_ok_ = false;
            bool stopping_ = false;
        };

        using SharedWeightLane = SharedLane<ExpertTierWeightTransferLane>;
        using SharedBlobLane = SharedLane<ExpertTierGpuBlobTransferLane>;
        using SharedPeerLane = SharedLane<ExpertTierGpuPeerTransferLane>;
        using SharedCpuCopyLane = SharedLane<CpuCopyLane>;
        using SharedWeightLanePool =
            SharedLanePool<ExpertTierWeightTransferLane>;
        using SharedBlobLanePool =
            SharedLanePool<ExpertTierGpuBlobTransferLane>;
        using SharedPeerLanePool =
            SharedLanePool<ExpertTierGpuPeerTransferLane>;
        using SharedCpuCopyLanePool = SharedLanePool<CpuCopyLane>;

        /** @brief Store an operation failure once and return the failed state. */
        MoEOverlayResidencyWaveProgress failOperation(
            std::string &failure,
            std::string message,
            std::string *error) noexcept
        {
            if (failure.empty())
                failure = std::move(message);
            if (error)
                *error = failure;
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        /**
         * @brief Fixed-cardinality operation for a rank outside a local edge.
         *
         * A globally fingerprinted wave contains every projection on every
         * rank. When both endpoints share some other rank, this explicit no-op
         * preserves composite ordering without creating a fake MPI endpoint.
         */
        class ReadyPhysicalProjectionOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Report readiness without touching device or network state. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                return aborted_ ? MoEOverlayResidencyWaveProgress::Failed
                                : MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Mark this unpublished no-op as discarded. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief No physical cleanup edge remains for an explicit no-op. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

        private:
            bool aborted_ = false;
        };

        /**
         * @brief Destination endpoint that finalizes storage before original operation completion.
         *
         * Final bytes remain owned and validated by the wrapped device-specific
         * endpoint. Storage finalization occurs only after that endpoint reports complete
         * and accepts its own finalization edge, so the candidate RCU bank can
         * never observe a partially received projection.
         */
        class FinalizingRemoteProjectionDestination final
            : public IMoEOverlayRemoteProjectionDestinationEndpoint
        {
        public:
            /**
             * @brief Retain original final storage and its reserved engine alias.
             * @param storage Original destination endpoint owning physical finalization.
             * @param engine Genuine engine alias over the reserved destination bytes.
             */
            FinalizingRemoteProjectionDestination(
                std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint> storage,
                std::shared_ptr<ITensorGemm> engine)
                : storage_(std::move(storage)), engine_(std::move(engine))
            {
                if (!storage_ || !engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay remote finalization endpoint requires complete ownership");
                }
            }

            /**
             * @brief Retain a deferred engine provider for manifest-shaped GPU arrivals.
             * @param storage Original endpoint owning manifest validation and final storage.
             * @param engine_provider Returns that endpoint's actual finalized engine.
             *
             * A remote GPU sender's live physical format is known only after
             * manifest authentication. The wrapped GPU endpoint creates the
             * exact engine alias then; this provider retrieves it at the original
             * finalization edge without guessing a canonical representation.
             */
            FinalizingRemoteProjectionDestination(
                std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint> storage,
                std::function<std::shared_ptr<ITensorGemm>()> engine_provider)
                : storage_(std::move(storage)), engine_provider_(std::move(engine_provider))
            {
                if (!storage_ || !engine_provider_)
                {
                    throw std::invalid_argument("ExpertOverlay deferred remote finalization "
                                                "endpoint requires complete ownership");
                }
            }

            /** @brief Delegate authenticated manifest binding to final storage. */
            bool beginManifest(
                const MoEOverlayRemoteProjectionManifest &manifest,
                std::string *error) noexcept override
            {
                return storage_->beginManifest(manifest, error);
            }

            /** @brief Delegate one validated staging chunk to final storage. */
            MoEOverlayResidencyWaveProgress beginChunk(
                const MoEOverlayRemoteProjectionChunkHeader &header,
                std::span<const std::uint8_t> payload,
                std::string *error) noexcept override
            {
                return storage_->beginChunk(header, payload, error);
            }

            /** @brief Poll the wrapped destination's exact completion event. */
            MoEOverlayResidencyWaveProgress pollChunk(
                std::string *error) noexcept override
            {
                return storage_->pollChunk(error);
            }

            /** @return Whether all authenticated bytes reached final storage. */
            [[nodiscard]] bool complete() const noexcept override
            {
                return storage_->complete();
            }

            /** @brief Finalize original storage so the original transport operation may finish; do not publish
             * arrival. */
            bool publishFinal(std::string *error) noexcept override
            {
                if (finalized_)
                {
                    if (error)
                        error->clear();
                    return true;
                }
                if (!storage_->publishFinal(error))
                    return false;
                if (!engine_)
                {
                    try
                    {
                        engine_ = engine_provider_();
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
                            *error =
                                "ExpertOverlay remote engine provider threw a non-standard exception";
                        return false;
                    }
                }
                if (!engine_)
                {
                    if (error)
                        *error =
                            "ExpertOverlay remote destination completed without an executable engine";
                    return false;
                }
                finalized_ = true;
                return true;
            }

            /** @brief Abort exact storage work; the outer operation poisons candidate arrival. */
            void abort() noexcept override
            {
                storage_->abort();
            }

            /** @brief Delegate event-aware storage reclamation after abort. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                return storage_->pollAbort(error);
            }

            /** @return Actual finalized engine, retained without granting candidate visibility. */
            std::shared_ptr<ITensorGemm> preparedEngine() const noexcept
            {
                return finalized_ ? engine_ : nullptr;
            }

        private:
            std::shared_ptr<
                IMoEOverlayRemoteProjectionDestinationEndpoint> storage_;
            std::shared_ptr<ITensorGemm> engine_;
            std::function<std::shared_ptr<ITensorGemm>()> engine_provider_;
            bool finalized_ = false;
        };

        /**
         * @brief Admitted GPU/CPU conversion over one exclusively reserved lane.
         *
         * Construction reserves one member of the edge's setup-owned pool.
         * The first poll therefore submits immediately; it can never wait for
         * another operation in the same wave to release a software lane.
         */
        class AdmittedWeightOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** Direction-specific data retained until completion or abort. */
            enum class Direction
            {
                GpuToCpuRepacked,
                CpuToGpuRepacked,
                GpuToCpuContiguous,
                CpuToGpuContiguous,
            };

            /**
             * @brief Bind an admitted operation to persistent source/destination storage.
             * @param lane_pool Admission-sized pool for the directed edge.
             * @param direction Conversion and DMA direction.
             * @param layout Complete scalar conversion contract.
             * @param gpu_source Read-only source for GPU-to-CPU.
             * @param cpu_destination Final CPU bytes for GPU-to-CPU.
             * @param cpu_source Final CPU bytes for CPU-to-GPU.
             * @param gpu_destination Inactive arrays for CPU-to-GPU.
             * @param source_readiness Installed old-epoch ordering proof.
             * @param source_engine Pins the immutable source slot.
             * @param destination_engine Pins the inactive destination slot.
             */
            AdmittedWeightOperation(
                std::shared_ptr<SharedWeightLanePool> lane_pool,
                Direction direction,
                ExpertTierWeightDeviceLayout layout,
                ExpertTierGpuConstProjectionView gpu_source,
                std::span<std::uint8_t> cpu_destination,
                std::span<const std::uint8_t> cpu_source,
                ExpertTierGpuMutableProjectionView gpu_destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                std::shared_ptr<std::mutex> encoded_write_mutex = {},
                ExpertTierDeviceTimingCapture timing_capture =
                    ExpertTierDeviceTimingCapture::LaneDefault)
                : direction_(direction),
                  layout_(layout),
                  gpu_source_(gpu_source),
                  cpu_destination_(cpu_destination),
                  cpu_source_(cpu_source),
                  gpu_destination_(gpu_destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  timing_capture_(timing_capture)
            {
                const bool repacked_direction =
                    direction_ == Direction::GpuToCpuRepacked ||
                    direction_ == Direction::CpuToGpuRepacked;
                if (!lane_pool || !repacked_direction || !layout_.valid() ||
                    !source_engine_ ||
                    !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted GPU/CPU operation has incomplete ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted GPU/CPU wave exhausted its pre-materialized parallel lane pool");
                }
                try { lane_->lane()->bindCPUEncodedWriteOwnership(std::move(encoded_write_mutex)); }
                catch (...) { lane_->release(this); throw; }
                owns_lane_ = true;
            }

            /**
             * @brief Bind one raw floating matrix to an admitted GPU/CPU lane.
             * @param lane_pool Admission-sized pool for the directed edge.
             * @param direction Raw floating DMA direction.
             * @param gpu_floating Exact GPU source or inactive destination.
             * @param cpu_destination Final CPU bytes for GPU-to-CPU.
             * @param cpu_source Final CPU bytes for CPU-to-GPU.
             * @param source_readiness Installed old-epoch ordering proof.
             * @param source_engine Pins the immutable source slot.
             * @param destination_engine Pins the inactive destination slot.
             * @param intersection Original-coordinate overlap between independently strided roots.
             *
             * Floating representations are byte-identical on CPU and GPU, so
             * this operation streams bounded chunks without a repack kernel.
             * It still uses the same pre-materialized lane, event lifecycle,
             * and publication rules as quantized transfers.
             */
            AdmittedWeightOperation(
                std::shared_ptr<SharedWeightLanePool> lane_pool,
                Direction direction,
                ContiguousFloatingPointWeightDescriptor gpu_floating,
                std::span<std::uint8_t> cpu_destination,
                std::span<const std::uint8_t> cpu_source,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                std::optional<ExpertTierWeightProjectionIntersection> intersection = std::nullopt,
                ExpertTierDeviceTimingCapture timing_capture =
                    ExpertTierDeviceTimingCapture::LaneDefault)
                : direction_(direction),
                  cpu_destination_(cpu_destination),
                  cpu_source_(cpu_source),
                  gpu_floating_(gpu_floating),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_intersection_(std::move(intersection)),
                  timing_capture_(timing_capture)
            {
                const bool contiguous_direction =
                    direction_ == Direction::GpuToCpuContiguous ||
                    direction_ == Direction::CpuToGpuContiguous;
                const auto cpu_bytes =
                    direction_ == Direction::GpuToCpuContiguous
                        ? cpu_destination_.size()
                        : cpu_source_.size();
                if (!lane_pool || !contiguous_direction ||
                    !gpu_floating_.valid() ||
                    (!floating_intersection_ && cpu_bytes != gpu_floating_.bytes) || !source_engine_ ||
                    !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted floating GPU/CPU operation has incomplete ownership");
                }
                if (floating_intersection_)
                {
                    const auto &overlap = *floating_intersection_;
                    const auto &gpu = direction_ == Direction::GpuToCpuContiguous ? overlap.source() : overlap.destination();
                    const auto &cpu = direction_ == Direction::GpuToCpuContiguous ? overlap.destination() : overlap.source();
                    const auto element = ExpertWeightFormat::floating(gpu_floating_.type).floatingElementBytes();
                    if (overlap.empty() || gpu.rows != gpu_floating_.n || gpu.columns != gpu_floating_.k ||
                        size_t(cpu.rows) * cpu.columns * element != cpu_bytes)
                        throw std::invalid_argument("Floating physical operation changed original independent root geometry");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted floating GPU/CPU wave exhausted its pre-materialized parallel lane pool");
                }
                try { lane_->lane()->bindCPUEncodedWriteOwnership({}); }
                catch (...) { lane_->release(this); throw; }
                owns_lane_ = true;
            }

            /** @brief Release only an unsubmitted or already-quiescent lease. */
            ~AdmittedWeightOperation() override
            {
                if (!owns_lane_)
                    return;
                if (started_ && !lane_->lane()->quiescent())
                    std::terminate();
                releaseLane();
            }

            /** @brief Acquire, submit, and event-poll without waiting. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU/CPU operation was aborted",
                        error);
                if (ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (failed_)
                    return failOperation(failure_, failure_, error);

                if (!started_)
                {
                    std::string start_error;
                    bool started = false;
                    switch (direction_)
                    {
                    case Direction::GpuToCpuRepacked:
                        started = lane_->lane()->startGpuToCpu(
                            layout_, gpu_source_, cpu_destination_,
                            source_readiness_, &start_error, timing_capture_);
                        break;
                    case Direction::CpuToGpuRepacked:
                        started = lane_->lane()->startCpuToGpu(
                            layout_, cpu_source_, gpu_destination_,
                            &start_error, timing_capture_);
                        break;
                    case Direction::GpuToCpuContiguous:
                        // The whole GPU root and CPU shard retain different
                        // strides. Forward the same checked overlap used to
                        // admit this operation instead of copying whole storage.
                        started = floating_intersection_
                            ? lane_->lane()->startGpuToCpuContiguous(
                                  gpu_floating_, cpu_destination_, source_readiness_,
                                  *floating_intersection_, &start_error, timing_capture_)
                            : lane_->lane()->startGpuToCpuContiguous(
                                  gpu_floating_, cpu_destination_, source_readiness_,
                                  &start_error, timing_capture_);
                        break;
                    case Direction::CpuToGpuContiguous:
                        started = floating_intersection_
                            ? lane_->lane()->startCpuToGpuContiguous(cpu_source_, gpu_floating_, *floating_intersection_, &start_error, timing_capture_)
                            : lane_->lane()->startCpuToGpuContiguous(cpu_source_, gpu_floating_, &start_error, timing_capture_);
                        break;
                    }
                    if (!started)
                    {
                        failed_ = true;
                        releaseLane();
                        return failOperation(
                            failure_,
                            start_error.empty()
                                ? "ExpertOverlay GPU/CPU lane failed to start"
                                : std::move(start_error),
                            error);
                    }
                    // Capture the protocol quote before polling completion or releasing this lane.
                    quoted_bytes_ = lane_->lane()->quotedPayloadBytes();
                    started_ = true;
                }

                std::string poll_error;
                const auto progress = lane_->lane()->poll(&poll_error);
                if (progress == ExpertTierWeightTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (progress == ExpertTierWeightTransferProgress::Ready)
                {
                    /*
                     * Copy the pointer-free observation before releasing the
                     * shared lane. A later wave may reuse that lane in
                     * the very next maintenance pass and replace its stats.
                     */
                    const auto observed =
                        lane_->lane()->stats().last_measurement;
                    if (observed.valid())
                        measurement_ = observed;
                    ready_ = true;
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                failed_ = true;
                if (lane_->lane()->quiescent())
                    releaseLane();
                return failOperation(
                    failure_,
                    poll_error.empty()
                        ? "ExpertOverlay GPU/CPU lane failed"
                        : std::move(poll_error),
                    error);
            }

            /** @brief Discard reserved work or mark submitted work for draining. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief Event-poll submitted work until lane reuse is safe. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU/CPU abort was not requested",
                        error);
                if (!owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (!started_)
                {
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }

                auto progress = lane_->lane()->progress();
                if (progress == ExpertTierWeightTransferProgress::Pending)
                    progress = lane_->lane()->poll(error);
                if (progress == ExpertTierWeightTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (!lane_->lane()->quiescent())
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU/CPU abort lost its completion fence",
                        error);
                releaseLane();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return the immutable observation captured at readiness. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? measurement_ : std::nullopt;
            }

            /** @return Actual bound lane quote captured before completion, independent of measurement_. */
            std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override { return quoted_bytes_; }


        private:
            std::optional<std::uint64_t> quoted_bytes_;
            /** @brief Relinquish exact shared-lane ownership once. */
            void releaseLane() noexcept
            {
                if (!owns_lane_)
                    return;
                lane_->release(this);
                owns_lane_ = false;
            }

            std::shared_ptr<SharedWeightLane> lane_;
            Direction direction_;
            ExpertTierWeightDeviceLayout layout_;
            ExpertTierGpuConstProjectionView gpu_source_;
            std::span<std::uint8_t> cpu_destination_;
            std::span<const std::uint8_t> cpu_source_;
            ExpertTierGpuMutableProjectionView gpu_destination_;
            ContiguousFloatingPointWeightDescriptor gpu_floating_;
            std::optional<ExpertTierWeightProjectionIntersection> floating_intersection_;
            ExpertTierSourceReadiness source_readiness_;
            std::shared_ptr<ITensorGemm> source_engine_;
            std::shared_ptr<ITensorGemm> destination_engine_;
            ExpertTierDeviceTimingCapture timing_capture_ =
                ExpertTierDeviceTimingCapture::LaneDefault;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::string failure_;
            bool started_ = false;
            bool owns_lane_ = false;
            bool ready_ = false;
            bool failed_ = false;
            bool aborted_ = false;
        };

        /** @brief Admitted byte-preserving CUDA/ROCm operation. */
        class AdmittedBlobOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain both descriptors, slots, and one shared blob lane. */
            AdmittedBlobOperation(
                std::shared_ptr<SharedBlobLanePool> lane_pool,
                GpuExpertPackedDescriptor source,
                GpuExpertPackedDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                ExpertTierWeightProjectionIntersection intersection)
                : source_(source),
                  destination_(destination),
                  intersection_(intersection),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
            {
                if (!lane_pool || !source_.valid() || !destination_.valid() ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted GPU host-relay operation has incomplete ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted GPU host-relay wave exhausted its pre-materialized parallel lane pool");
                }
                owns_lane_ = true;
            }

            /** @brief Retain one contiguous floating source/destination pair. */
            AdmittedBlobOperation(
                std::shared_ptr<SharedBlobLanePool> lane_pool,
                ContiguousFloatingPointWeightDescriptor source,
                ContiguousFloatingPointWeightDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                ExpertTierWeightProjectionIntersection intersection)
                : floating_source_(source),
                  floating_destination_(destination),
                  intersection_(intersection),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_(true)
            {
                if (!lane_pool || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != intersection_.source().rows ||
                    floating_source_.k != intersection_.source().columns ||
                    floating_destination_.n != intersection_.destination().rows ||
                    floating_destination_.k != intersection_.destination().columns ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted floating GPU host-relay operation has incompatible storage or ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted floating GPU host-relay wave exhausted its pre-materialized parallel lane pool");
                }
                owns_lane_ = true;
            }

            /** @brief Release only an unsubmitted or already-quiescent lease. */
            ~AdmittedBlobOperation() override
            {
                if (!owns_lane_)
                    return;
                if (started_ && !lane_->lane()->quiescent())
                    std::terminate();
                releaseLane();
            }

            /** @brief Acquire the edge and poll both runtimes' exact events. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU host-relay operation was aborted",
                        error);
                if (ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (failed_)
                    return failOperation(failure_, failure_, error);
                if (!started_)
                {
                    std::string start_error;
                    const bool started = floating_
                        ? lane_->lane()->startFloating(
                              floating_source_,
                              floating_destination_,
                              intersection_,
                              source_readiness_,
                              &start_error)
                        : lane_->lane()->start(
                              source_,
                              destination_,
                              intersection_,
                              source_readiness_,
                              &start_error);
                    if (!started)
                    {
                        failed_ = true;
                        releaseLane();
                        return failOperation(
                            failure_,
                            start_error.empty()
                                ? "ExpertOverlay GPU host-relay lane failed to start"
                                : std::move(start_error),
                            error);
                    }
                    // Capture the protocol quote before polling completion or releasing this lane.
                    quoted_bytes_ = lane_->lane()->quotedPayloadBytes();
                    started_ = true;
                }

                std::string poll_error;
                const auto progress = lane_->lane()->poll(&poll_error);
                if (progress == ExpertTierGpuBlobTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (progress == ExpertTierGpuBlobTransferProgress::Ready)
                {
                    /* Preserve evidence before a later wave reuses this lane. */
                    const auto observed =
                        lane_->lane()->stats().last_measurement;
                    if (observed.valid())
                        measurement_ = observed;
                    ready_ = true;
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                failed_ = true;
                if (lane_->lane()->quiescent())
                    releaseLane();
                return failOperation(
                    failure_,
                    poll_error.empty()
                        ? "ExpertOverlay GPU host-relay lane failed"
                        : std::move(poll_error),
                    error);
            }

            /** @brief Mark an unpublished reserved or submitted blob discarded. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief Drain both runtime event sets before releasing the edge. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU host-relay abort was not requested",
                        error);
                if (!owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (!started_)
                {
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                auto progress = lane_->lane()->progress();
                if (progress == ExpertTierGpuBlobTransferProgress::Pending)
                    progress = lane_->lane()->poll(error);
                if (progress == ExpertTierGpuBlobTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (!lane_->lane()->quiescent())
                    return failOperation(
                        failure_,
                        "ExpertOverlay heterogeneous GPU abort lost a completion fence",
                        error);
                releaseLane();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return the completed heterogeneous transfer evidence. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? measurement_ : std::nullopt;
            }

            /** @return Actual bound lane quote captured before completion, independent of measurement_. */
            std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override { return quoted_bytes_; }


        private:
            std::optional<std::uint64_t> quoted_bytes_;
            /** @brief Return the shared blob lane to its directed edge. */
            void releaseLane() noexcept
            {
                if (!owns_lane_)
                    return;
                lane_->release(this);
                owns_lane_ = false;
            }

            std::shared_ptr<SharedBlobLane> lane_;
            GpuExpertPackedDescriptor source_;
            GpuExpertPackedDescriptor destination_;
            ContiguousFloatingPointWeightDescriptor floating_source_;
            ContiguousFloatingPointWeightDescriptor floating_destination_;
            ExpertTierWeightProjectionIntersection intersection_;
            ExpertTierSourceReadiness source_readiness_;
            std::shared_ptr<ITensorGemm> source_engine_;
            std::shared_ptr<ITensorGemm> destination_engine_;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::string failure_;
            bool started_ = false;
            bool owns_lane_ = false;
            bool ready_ = false;
            bool failed_ = false;
            bool aborted_ = false;
            bool floating_ = false;
        };

        /** @brief Admitted same-backend GPU peer-copy operation. */
        class AdmittedPeerOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain compatible packed descriptors and their slots. */
            AdmittedPeerOperation(
                std::shared_ptr<SharedPeerLanePool> lane_pool,
                GpuExpertPackedDescriptor source,
                GpuExpertPackedDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                ExpertTierWeightProjectionIntersection intersection,
                ExpertTierDeviceTimingCapture timing_capture =
                    ExpertTierDeviceTimingCapture::LaneDefault)
                : source_(source),
                  destination_(destination),
                  intersection_(intersection),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  timing_capture_(timing_capture)
            {
                if (!lane_pool || !source_.valid() || !destination_.valid() ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted GPU peer operation has incomplete ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted GPU peer wave exhausted its pre-materialized parallel lane pool");
                }
                owns_lane_ = true;
            }

            /** @brief Retain one contiguous floating peer-copy pair. */
            AdmittedPeerOperation(
                std::shared_ptr<SharedPeerLanePool> lane_pool,
                ContiguousFloatingPointWeightDescriptor source,
                ContiguousFloatingPointWeightDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                ExpertTierWeightProjectionIntersection intersection,
                ExpertTierDeviceTimingCapture timing_capture =
                    ExpertTierDeviceTimingCapture::LaneDefault)
                : floating_source_(source),
                  floating_destination_(destination),
                  intersection_(intersection),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  timing_capture_(timing_capture),
                  floating_(true)
            {
                if (!lane_pool || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != intersection_.source().rows ||
                    floating_source_.k != intersection_.source().columns ||
                    floating_destination_.n != intersection_.destination().rows ||
                    floating_destination_.k != intersection_.destination().columns ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted floating GPU peer operation has incompatible storage or ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted floating GPU peer wave exhausted its pre-materialized parallel lane pool");
                }
                owns_lane_ = true;
            }

            /** @brief Release only an unsubmitted or already-quiescent lease. */
            ~AdmittedPeerOperation() override
            {
                if (!owns_lane_)
                    return;
                if (started_ && !lane_->lane()->quiescent())
                    std::terminate();
                releaseLane();
            }

            /** @brief Acquire, submit, and query the destination event. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU peer operation was aborted",
                        error);
                if (ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (failed_)
                    return failOperation(failure_, failure_, error);
                if (!started_)
                {
                    std::string start_error;
                    const bool started = floating_
                        ? lane_->lane()->startFloating(
                              floating_source_,
                              floating_destination_,
                              intersection_,
                              source_readiness_,
                              &start_error,
                              timing_capture_)
                        : lane_->lane()->start(
                              source_,
                              destination_,
                              intersection_,
                              source_readiness_,
                              &start_error,
                              timing_capture_);
                    if (!started)
                    {
                        failed_ = true;
                        releaseLane();
                        return failOperation(
                            failure_,
                            start_error.empty()
                                ? "ExpertOverlay GPU peer lane failed to start"
                                : std::move(start_error),
                            error);
                    }
                    // Capture the protocol quote before polling completion or releasing this lane.
                    quoted_bytes_ = lane_->lane()->quotedPayloadBytes();
                    started_ = true;
                }

                std::string poll_error;
                const auto progress = lane_->lane()->poll(&poll_error);
                if (progress == ExpertTierGpuPeerTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (progress == ExpertTierGpuPeerTransferProgress::Ready)
                {
                    /* Preserve evidence before a later wave reuses this lane. */
                    const auto observed =
                        lane_->lane()->stats().last_measurement;
                    if (observed.valid())
                        measurement_ = observed;
                    ready_ = true;
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                failed_ = true;
                if (lane_->lane()->quiescent())
                    releaseLane();
                return failOperation(
                    failure_,
                    poll_error.empty()
                        ? "ExpertOverlay GPU peer lane failed"
                        : std::move(poll_error),
                    error);
            }

            /** @brief Mark reserved work discarded without cancelling DMA. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief Drain the exact destination event before edge reuse. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU peer abort was not requested",
                        error);
                if (!owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (!started_)
                {
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                auto progress = lane_->lane()->progress();
                if (progress == ExpertTierGpuPeerTransferProgress::Pending)
                    progress = lane_->lane()->poll(error);
                if (progress == ExpertTierGpuPeerTransferProgress::Pending)
                    return MoEOverlayResidencyWaveProgress::Pending;
                if (!lane_->lane()->quiescent())
                    return failOperation(
                        failure_,
                        "ExpertOverlay GPU peer abort lost its completion fence",
                        error);
                releaseLane();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return the completed same-backend peer-copy evidence. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? measurement_ : std::nullopt;
            }

            /** @return Actual bound lane quote captured before completion, independent of measurement_. */
            std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override { return quoted_bytes_; }


        private:
            std::optional<std::uint64_t> quoted_bytes_;
            /** @brief Return the shared peer lane to its directed edge. */
            void releaseLane() noexcept
            {
                if (!owns_lane_)
                    return;
                lane_->release(this);
                owns_lane_ = false;
            }

            std::shared_ptr<SharedPeerLane> lane_;
            GpuExpertPackedDescriptor source_;
            GpuExpertPackedDescriptor destination_;
            ContiguousFloatingPointWeightDescriptor floating_source_;
            ContiguousFloatingPointWeightDescriptor floating_destination_;
            ExpertTierWeightProjectionIntersection intersection_;
            ExpertTierSourceReadiness source_readiness_;
            std::shared_ptr<ITensorGemm> source_engine_;
            std::shared_ptr<ITensorGemm> destination_engine_;
            ExpertTierDeviceTimingCapture timing_capture_ =
                ExpertTierDeviceTimingCapture::LaneDefault;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::string failure_;
            bool started_ = false;
            bool owns_lane_ = false;
            bool ready_ = false;
            bool failed_ = false;
            bool aborted_ = false;
            bool floating_ = false;
        };

        /** @brief Admitted CPU/NUMA copy over one dedicated background worker. */
        class AdmittedCpuCopyOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /**
             * @brief Reserve one setup-owned worker for immutable final storage.
             * @param lane_pool Exact CPU address-edge/projection worker pool.
             * @param source Final prepared source bytes.
             * @param destination Candidate-slot destination bytes.
             * @param wave_gate Barrier releasing the complete CPU wave together.
             * @param source_engine Physical source-lifetime pin.
             * @param destination_engine Physical destination-lifetime pin.
             */
            AdmittedCpuCopyOperation(
                std::shared_ptr<SharedCpuCopyLanePool> lane_pool,
                std::span<const std::uint8_t> source,
                std::span<std::uint8_t> destination,
                std::shared_ptr<CpuCopyWaveGate> wave_gate,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                CPUCopyGeometry geometry = CompleteCPUCopy{})
                : source_(source),
                  destination_(destination),
                  wave_gate_(std::move(wave_gate)),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)), geometry_(std::move(geometry))
            {
                if (!lane_pool || !validCPUCopy(geometry_, source_, destination_) || !wave_gate_ ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU copy requires exact final storage and ownership");
                }
                quoted_bytes_ = cpuCopyBytes(geometry_, source_.size());
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay CPU copy wave exhausted its pre-materialized parallel worker pool");
                }
                owns_lane_ = true;
            }

            /** @brief Reject destruction while its background worker owns bytes. */
            ~AdmittedCpuCopyOperation() override
            {
                if (!owns_lane_)
                    return;
                if (started_ && !lane_->lane()->quiescent())
                    std::terminate();
                releaseLane();
            }

            /** @brief Submit once, then observe the worker without waiting. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay CPU copy was aborted",
                        error);
                if (ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;

                if (!started_)
                {
                    std::string start_error;
                    if (!lane_->lane()->start(
                            source_,
                            destination_,
                            wave_gate_,
                            source_engine_,
                            destination_engine_,
                            &start_error, geometry_))
                    {
                        if (lane_->lane()->quiescent())
                            releaseLane();
                        return failOperation(
                            failure_,
                            start_error.empty()
                                ? "ExpertOverlay CPU copy worker rejected admitted work"
                                : std::move(start_error),
                            error);
                    }
                    started_ = true;
                }

                std::string lane_error;
                const auto progress = lane_->lane()->poll(&lane_error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    return progress;
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                {
                    if (lane_->lane()->quiescent())
                        releaseLane();
                    return failOperation(
                        failure_,
                        lane_error.empty()
                            ? "ExpertOverlay CPU copy worker failed"
                            : std::move(lane_error),
                        error);
                }

                measurement_ = lane_->lane()->completedMeasurement();
                if (!measurement_ || !measurement_->valid())
                {
                    releaseLane();
                    return failOperation(
                        failure_,
                        "ExpertOverlay CPU copy worker omitted exact timing evidence",
                        error);
                }
                ready_ = true;
                releaseLane();
                return progress;
            }

            /** @brief Cancel before submission or at the next worker chunk edge. */
            void abort() noexcept override
            {
                aborted_ = true;
                if (started_ && owns_lane_)
                    lane_->lane()->requestAbort();
            }

            /** @brief Event-poll cooperative worker cancellation without joining. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay CPU copy abort was not requested",
                        error);
                if (!owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (!started_)
                {
                    releaseLane();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (!lane_->lane()->quiescent())
                    return MoEOverlayResidencyWaveProgress::Pending;
                releaseLane();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return exact asynchronous host-copy timing after readiness. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return ready_ ? measurement_ : std::nullopt;
            }

            /** @return Same exact native-plane/floating-row quote used by the CPU worker, not observed bytes. */
            std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override
            { return quoted_bytes_; }


        private:
            std::optional<std::uint64_t> quoted_bytes_;
            /** @brief Return the dedicated worker to its exact address edge. */
            void releaseLane() noexcept
            {
                if (!owns_lane_)
                    return;
                lane_->release(this);
                owns_lane_ = false;
            }

            std::shared_ptr<SharedCpuCopyLane> lane_;
            std::span<const std::uint8_t> source_;
            std::span<std::uint8_t> destination_;
            std::shared_ptr<CpuCopyWaveGate> wave_gate_;
            std::shared_ptr<ITensorGemm> source_engine_;
            std::shared_ptr<ITensorGemm> destination_engine_;
            CPUCopyGeometry geometry_;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::string failure_;
            bool started_ = false;
            bool owns_lane_ = false;
            bool ready_ = false;
            bool aborted_ = false;
        };

        /** Source-independent exact projection identity shared with admission. */
        using ProjectionSignature = ExpertPreparedProjectionIdentity;

        /** Gate/up/down identity shared by admission and physical arenas. */
        using ExpertGeometryKey = ExpertPreparedTripletGeometryKey;

        /**
         * @brief Recyclable view over loader-owned initial expert allocations.
         *
         * Initial prepared engines predate the migration fabric, so they do not
         * carry the lease token used by ordinary arrival slots. This arena
         * adopts their stable storage without copying it, marks a departed
         * bootstrap assignment free only after the old ticket barrier, and
         * gives every later assignment the normal aliasing lease. The backing
         * engine remains model-owned; only its logical expert/epoch assignment
         * changes.
         */
        class AdoptedInitialExpertSlotRecycler final
            : public std::enable_shared_from_this<
                  AdoptedInitialExpertSlotRecycler>
        {
        public:
            /** @brief One role's stable engine plus CPU or GPU destination view. */
            struct Projection
            {
                ExpertTierWeightProjection projection =
                    ExpertTierWeightProjection::Gate;
                std::shared_ptr<ITensorGemm> engine;
                std::span<std::uint8_t> cpu_destination;
                std::optional<GpuExpertSlotPool::ProjectionSlot>
                    gpu_destination;
            };

            /** @brief One initially occupied physical expert allocation. */
            struct InitialSlot
            {
                int layer_idx = -1;
                int expert_id = -1;
                std::uint64_t residency_epoch = 0;
                std::vector<Projection> projections;
            };

            /**
             * @brief Validate and adopt every resident allocation in one geometry.
             * @param device Exact endpoint device.
             * @param participant_id Logical endpoint identity.
             * @param layer_indices Sorted layers sharing the projection geometry.
             * @param initial_slots Loader-owned resident payloads.
             * @param perf_device Stable topology label for evidence.
             */
            AdoptedInitialExpertSlotRecycler(
                DeviceId device,
                int participant_id,
                std::vector<int> layer_indices,
                std::vector<InitialSlot> initial_slots,
                std::string perf_device)
                : device_(device),
                  participant_id_(participant_id),
                  layer_indices_(std::move(layer_indices)),
                  perf_device_(std::move(perf_device))
            {
                if ((!device_.is_cpu() && !device_.is_gpu()) ||
                    participant_id_ < 0 || layer_indices_.empty() ||
                    !std::is_sorted(
                        layer_indices_.begin(), layer_indices_.end()) ||
                    std::adjacent_find(
                        layer_indices_.begin(), layer_indices_.end()) !=
                        layer_indices_.end() ||
                    layer_indices_.front() < 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay adopted-slot arena has invalid endpoint identity");
                }

                slots_.reserve(initial_slots.size());
                for (auto &initial : initial_slots)
                {
                    if (!std::binary_search(
                            layer_indices_.begin(),
                            layer_indices_.end(),
                            initial.layer_idx) ||
                        initial.expert_id < 0 ||
                        initial.residency_epoch == 0 ||
                        initial.projections.size() != kProjections.size())
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay adopted initial slot has invalid identity or projection count");
                    }
                    std::array<bool, 3> seen{};
                    for (const auto &projection : initial.projections)
                    {
                        const auto role = static_cast<std::size_t>(
                            projection.projection);
                        if (role >= seen.size() || seen[role] ||
                            !projection.engine)
                        {
                            throw std::invalid_argument(
                                "ExpertOverlay adopted initial slot has duplicate or absent projection ownership");
                        }
                        seen[role] = true;
                        const bool exact_storage = device_.is_cpu()
                            ? (!projection.cpu_destination.empty() &&
                               !projection.gpu_destination.has_value())
                            : (projection.cpu_destination.empty() &&
                               projection.gpu_destination.has_value());
                        if (!exact_storage)
                        {
                            throw std::invalid_argument(
                                "ExpertOverlay adopted initial slot storage disagrees with its endpoint device");
                        }
                    }
                    slots_.push_back({
                        .projections = std::move(initial.projections),
                        .layer_idx = initial.layer_idx,
                        .expert_id = initial.expert_id,
                        .residency_epoch = initial.residency_epoch,
                        .bootstrap_assignment = true,
                    });
                    ++initial_capacity_by_layer_[initial.layer_idx];
                }
            }

            /**
             * @brief Acquire a free adopted CPU allocation for a candidate epoch.
             * @return Complete aliasing lease, or no value when none is free.
             */
            [[nodiscard]] std::optional<CpuExpertSlotPool::Lease> acquireCpu(
                int layer_idx,
                int expert_id,
                std::uint64_t residency_epoch)
            {
                if (!device_.is_cpu())
                    return std::nullopt;
                const auto assignment = acquireAssignment(
                    layer_idx, expert_id, residency_epoch);
                if (!assignment)
                    return std::nullopt;

                CpuExpertSlotPool::Lease lease;
                lease.slot_index = assignment->slot_index;
                lease.layer_idx = layer_idx;
                lease.expert_id = expert_id;
                lease.residency_epoch = residency_epoch;
                lease.lifetime = assignment->lifetime;
                const auto &slot = slots_[static_cast<std::size_t>(
                    assignment->slot_index)];
                lease.projections.reserve(slot.projections.size());
                for (const auto &projection : slot.projections)
                {
                    /* The alias, not the permanent engine owner, pins assignment. */
                    std::shared_ptr<ITensorGemm> engine(
                        assignment->lifetime,
                        projection.engine.get());
                    std::span<std::uint8_t> destination =
                        projection.cpu_destination;
                    if (!destination.empty() &&
                        projection.engine->exportCPUNativeQuantisedPackedWeights())
                    {
                        /* The epoch rewrites these bytes: unadmitted until arrival. */
                        destination =
                            projection.engine->beginRetiredCPUNativeQuantisedRewrite();
                        if (destination.data() !=
                                projection.cpu_destination.data() ||
                            destination.size() !=
                                projection.cpu_destination.size())
                        {
                            throw std::logic_error(
                                "ExpertOverlay adopted CPU engine no longer owns its recyclable final storage");
                        }
                    }
                    lease.projections.push_back({
                        .projection = projection.projection,
                        .destination_bytes = destination,
                        .engine = std::move(engine),
                    });
                }
                recordAcquisition(
                    layer_idx, assignment->slot_index, residency_epoch);
                return lease;
            }

            /**
             * @brief Acquire a free adopted GPU allocation for a candidate epoch.
             * @return Stable descriptor storage plus assignment lease, or no value.
             */
            [[nodiscard]] std::optional<GpuExpertSlotPool::AcquiredSlot>
            acquireGpu(
                int layer_idx,
                int expert_id,
                std::uint64_t residency_epoch)
            {
                if (!device_.is_gpu())
                    return std::nullopt;
                const auto assignment = acquireAssignment(
                    layer_idx, expert_id, residency_epoch);
                if (!assignment)
                    return std::nullopt;

                GpuExpertSlotPool::AcquiredSlot lease;
                lease.slot_index = assignment->slot_index;
                lease.layer_idx = layer_idx;
                lease.expert_id = expert_id;
                lease.residency_epoch = residency_epoch;
                lease.lifetime = assignment->lifetime;
                const auto &slot = slots_[static_cast<std::size_t>(
                    assignment->slot_index)];
                lease.projections.reserve(slot.projections.size());
                for (const auto &projection : slot.projections)
                    lease.projections.push_back(*projection.gpu_destination);
                recordAcquisition(
                    layer_idx, assignment->slot_index, residency_epoch);
                return lease;
            }

            /** @return Immediately reusable adopted allocations. */
            [[nodiscard]] std::size_t availableSlots() const noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return static_cast<std::size_t>(std::count_if(
                    slots_.begin(),
                    slots_.end(),
                    [](const Slot &slot) { return slot.expert_id < 0; }));
            }

            /** @return Number of loader allocations enrolled in the arena. */
            [[nodiscard]] std::size_t capacity() const noexcept
            {
                return slots_.size();
            }

            /** @return Loader-owned allocations originating in one layer. */
            [[nodiscard]] std::size_t initialCapacityForLayer(
                int layer_idx) const noexcept
            {
                const auto found = initial_capacity_by_layer_.find(layer_idx);
                return found == initial_capacity_by_layer_.end()
                    ? 0u
                    : found->second;
            }

            /**
             * @brief Test whether one resident already occupies this exact arena.
             * @param expert_id Logical expert expected in the current bank.
             * @return True when exactly one adopted slot owns that expert.
             *
             * Assignment identity is stronger than allocation shape.  A generic
             * GPU shadow slot can expose the same payload size as a model format
             * while still retaining unused minima arrays, so format comparison
             * alone cannot decide whether the old shadow arena may be destroyed.
             */
            [[nodiscard]] bool ownsAssignment(
                int layer_idx,
                int expert_id) const noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                return std::count_if(
                           slots_.begin(),
                           slots_.end(),
                           [layer_idx, expert_id](const Slot &slot)
                           {
                               return slot.layer_idx == layer_idx &&
                                   slot.expert_id == expert_id;
                           }) == 1;
            }

            /**
             * @brief Release an initial non-lease assignment after ticket drain.
             * @param expert_id Expert that has departed this endpoint/layer.
             * @param retired_bank_epoch Globally drained bank that last exposed
             *        the expert; it may be newer than the physical assignment.
             * @return True only when an exact bootstrap identity became free.
             *
             * Later assignments are lease-managed and deliberately ignored;
             * destruction of their final bank/operation alias performs release.
             */
            bool retireBootstrapAssignment(
                int layer_idx,
                int expert_id,
                std::uint64_t retired_bank_epoch) noexcept
            {
                int released_slot = -1;
                std::uint64_t assignment_epoch = 0;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (std::size_t index = 0; index < slots_.size(); ++index)
                    {
                        auto &slot = slots_[index];
                        if (slot.layer_idx != layer_idx ||
                            slot.expert_id != expert_id ||
                            !slot.bootstrap_assignment ||
                            slot.residency_epoch > retired_bank_epoch)
                        {
                            continue;
                        }
                        /*
                         * A loader-owned assignment keeps the epoch at which
                         * its physical storage entered the recycler.  The
                         * expert may survive through any number of immutable
                         * RCU banks before it departs, so equality with the
                         * departing bank epoch would strand every bootstrap
                         * slot except one removed in the first wave.  The
                         * bootstrap bit plus expert identity is unique in this
                         * endpoint/layer; the monotonic check prevents an
                         * impossible future assignment from being reclaimed.
                         */
                        assignment_epoch = slot.residency_epoch;
                        slot.layer_idx = -1;
                        slot.expert_id = -1;
                        slot.residency_epoch = 0;
                        slot.bootstrap_assignment = false;
                        released_slot = static_cast<int>(index);
                        break;
                    }
                }
                if (released_slot < 0)
                    return false;
                recordRelease(
                    "bootstrap_slot_retired",
                    layer_idx,
                    released_slot,
                    assignment_epoch);
                return true;
            }

        private:
            /** @brief Physical slot plus its current logical assignment. */
            struct Slot
            {
                std::vector<Projection> projections;
                /** Current logical layer, or -1 while this storage is free. */
                int layer_idx = -1;
                int expert_id = -1;
                std::uint64_t residency_epoch = 0;
                bool bootstrap_assignment = false;
            };

            /** @brief Token whose final alias returns one non-bootstrap slot. */
            struct LeaseToken
            {
                /**
                 * Keep the recycler and its loader-owned engine objects alive.
                 *
                 * A restored model registry may retain this alias after the
                 * runner destroys its physical fabric. Weak ownership would
                 * leave the alias pointing at an engine object owned only by
                 * the recycler. The strong edge is acyclic: the recycler never
                 * owns lease tokens, and final token destruction returns the
                 * logical slot before releasing the recycler.
                 */
                std::shared_ptr<AdoptedInitialExpertSlotRecycler> pool;
                int slot_index = -1;
                int layer_idx = -1;
                int expert_id = -1;
                std::uint64_t residency_epoch = 0;
            };

            /** @brief Index and shared lifetime returned by atomic reservation. */
            struct Assignment
            {
                int slot_index = -1;
                std::shared_ptr<void> lifetime;
            };

            /** @brief Atomically reserve one free slot and create its lease token. */
            [[nodiscard]] std::optional<Assignment> acquireAssignment(
                int layer_idx,
                int expert_id,
                std::uint64_t residency_epoch)
            {
                if (!std::binary_search(
                        layer_indices_.begin(),
                        layer_indices_.end(),
                        layer_idx) ||
                    expert_id < 0 || residency_epoch == 0)
                    return std::nullopt;

                int selected = -1;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    for (std::size_t index = 0; index < slots_.size(); ++index)
                    {
                        if (slots_[index].expert_id < 0)
                        {
                            selected = static_cast<int>(index);
                            break;
                        }
                    }
                    if (selected < 0)
                        return std::nullopt;
                    auto &slot = slots_[static_cast<std::size_t>(selected)];
                    slot.layer_idx = layer_idx;
                    slot.expert_id = expert_id;
                    slot.residency_epoch = residency_epoch;
                    slot.bootstrap_assignment = false;
                }

                auto token = std::shared_ptr<LeaseToken>(
                    new LeaseToken{
                        .pool = shared_from_this(),
                        .slot_index = selected,
                        .layer_idx = layer_idx,
                        .expert_id = expert_id,
                        .residency_epoch = residency_epoch,
                    },
                    [](LeaseToken *lease) noexcept
                    {
                        if (lease)
                        {
                            if (lease->pool)
                            {
                                lease->pool->releaseLease(
                                    lease->slot_index,
                                    lease->layer_idx,
                                    lease->expert_id,
                                    lease->residency_epoch);
                            }
                        }
                        delete lease;
                    });
                return Assignment{
                    .slot_index = selected,
                    .lifetime = std::shared_ptr<void>(token, token.get()),
                };
            }

            /** @brief Release only the exact lease-managed assignment. */
            void releaseLease(
                int slot_index,
                int layer_idx,
                int expert_id,
                std::uint64_t residency_epoch) noexcept
            {
                bool released = false;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    if (slot_index < 0 ||
                        slot_index >= static_cast<int>(slots_.size()))
                    {
                        return;
                    }
                    auto &slot = slots_[static_cast<std::size_t>(slot_index)];
                    if (slot.bootstrap_assignment ||
                        slot.layer_idx != layer_idx ||
                        slot.expert_id != expert_id ||
                        slot.residency_epoch != residency_epoch)
                    {
                        return;
                    }
                    slot.layer_idx = -1;
                    slot.expert_id = -1;
                    slot.residency_epoch = 0;
                    released = true;
                }
                if (released)
                    recordRelease(
                        "adopted_slot_lease_released",
                        layer_idx,
                        slot_index,
                        residency_epoch);
            }

            /** @brief Emit one acquisition proof outside the pool mutex. */
            void recordAcquisition(
                int layer_idx,
                int slot_index,
                std::uint64_t residency_epoch) const
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "adopted_live_slot_acquired",
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"participant", std::to_string(participant_id_)},
                     {"layer", std::to_string(layer_idx)},
                     {"slot", std::to_string(slot_index)},
                     {"epoch", std::to_string(residency_epoch)},
                     {"device", device_.to_string()}});
            }

            /** @brief Emit one exact retirement/release proof. */
            void recordRelease(
                const char *counter,
                int layer_idx,
                int slot_index,
                std::uint64_t residency_epoch) const noexcept
            {
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    counter,
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"participant", std::to_string(participant_id_)},
                     {"layer", std::to_string(layer_idx)},
                     {"slot", std::to_string(slot_index)},
                     {"epoch", std::to_string(residency_epoch)},
                     {"device", device_.to_string()}});
            }

            DeviceId device_ = DeviceId::invalid();
            int participant_id_ = -1;
            std::vector<int> layer_indices_;
            std::map<int, std::size_t> initial_capacity_by_layer_;
            std::string perf_device_;
            std::vector<Slot> slots_;
            mutable std::mutex mutex_;
        };

        /** @brief One participant's recyclable arena for an exact geometry. */
        struct EndpointGeometryPool
        {
            DeviceId device = DeviceId::invalid();
            int participant_id = -1;
            std::vector<int> layer_indices;
            /** Globally bounded inactive overlap capacity charged in the BOM. */
            std::size_t shadow_capacity = 0;
            /** Loader allocations shared by every compatible layer after retire. */
            std::shared_ptr<AdoptedInitialExpertSlotRecycler> adopted_slots;
            std::variant<
                std::shared_ptr<CpuExpertSlotPool>,
                std::shared_ptr<GpuExpertSlotPool>>
                pool;
        };

        /** @brief Layer binding into one exact-geometry destination arena. */
        struct EndpointLayerPool
        {
            std::shared_ptr<EndpointGeometryPool> geometry_pool;
            /** Maximum simultaneous arrivals for this one logical layer. */
            std::size_t layer_arrival_capacity = 0;
        };

        /**
         * @brief Represent existing whole-owner work through checked intersection coverage.
         * @param rows Actual original projection N, without storage padding.
         * @param columns Actual original projection K.
         * @return One full original overlap, not capacity-sized wire work.
         */
        ExpertTierWeightProjectionIntersection wholeProjectionIntersection(int rows, int columns)
        {
            const MoEExpertProjectionOwnership::Projection whole{
                MoEProjectionResidence::ExpertOwner, rows, columns, 0, rows, 0, columns};
            return ExpertTierWeightProjectionIntersection::between(whole, whole);
        }

        /**
         * @brief Same original member bank retained until every real source piece drains.
         *
         * The piece count is completion protocol state, never a live byte or
         * capacity ledger. Releasing at the last Ready/abort-drained edge lets
         * the existing RCU old-reader retirement proceed before wave destruction.
         */
        struct RetainedPhysicalSourceMember
        {
            std::shared_ptr<MoEOverlayParticipantResidency> endpoint;
            MoEOverlayParticipantBankLease bank;
            std::array<std::shared_ptr<ITensorGemm>, 3> engines;
            std::atomic<std::size_t> pending{0};
            /** @brief Release the original read lease exactly at the last quiescent piece. */
            void completePiece() noexcept
            {
                const auto previous = pending.fetch_sub(1, std::memory_order_acq_rel);
                if (previous == 0) std::terminate();
                if (previous == 1) bank = {};
            }
        };

        /** @brief Preserve one original source member across its actual lane/MPI operation. */
        class SourceMemberRetainedOperation final : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @param operation Original real operation, including its abort/drain semantics.
             * @param source SAME source bank state shared by all of this member's pieces. */
            SourceMemberRetainedOperation(std::unique_ptr<IMoEOverlayTierTransferOperation> operation,
                std::shared_ptr<RetainedPhysicalSourceMember> source)
                : operation_(std::move(operation)), source_(std::move(source))
            {
                if (!operation_ || !source_ || source_->pending.load() == 0)
                    throw std::invalid_argument("Physical source retention requires its actual operation and original member");
            }
            /** @brief Delegate normal progress and release only after real Ready. */
            MoEOverlayResidencyWaveProgress poll(std::string *error) noexcept override
            { return observe(operation_->poll(error)); }
            /** @brief Preserve original asynchronous cancellation. */
            void abort() noexcept override { operation_->abort(); }
            /** @brief Delegate abort drain and release only after genuine quiescence. */
            MoEOverlayResidencyWaveProgress pollAbort(std::string *error) noexcept override
            { return observe(operation_->pollAbort(error)); }
            /** @return Original per-lane measurement without aggregation or changed attribution. */
            std::optional<ExpertTierProjectionTransferMeasurement> completedMeasurement() const noexcept override
            { return operation_->completedMeasurement(); }
            /** @return Same underlying bound quote; source retention does not replace byte authority. */
            std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override
            { return operation_->quotedPayloadBytes(); }
        private:
            /** @brief Consume this operation's one terminal completion edge. */
            MoEOverlayResidencyWaveProgress observe(MoEOverlayResidencyWaveProgress progress) noexcept
            {
                if (progress == MoEOverlayResidencyWaveProgress::Ready && !complete_)
                { complete_ = true; source_->completePiece(); }
                return progress;
            }
            std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
            std::shared_ptr<RetainedPhysicalSourceMember> source_;
            bool complete_ = false;
        };

        /** @brief Actual final-role storage shared by all of its independent physical pieces. */
        struct DestinationProjectionStorage
        {
            std::shared_ptr<ITensorGemm> engine;
            std::shared_ptr<std::mutex> encoded_write_mutex = std::make_shared<std::mutex>();
        };
        /** @brief Coverage completion and storage are separate to avoid a shared-owner cycle. */
        struct DestinationProjectionGroup
        {
            std::shared_ptr<DestinationProjectionStorage> storage;
            std::shared_ptr<MoEOverlayPreparedProjectionCompletion> completion;
        };

        /** @brief Complete local destination reservation for one migration. */
        struct ReservedDestination
        {
            std::optional<CpuExpertSlotPool::Lease> cpu;
            std::optional<GpuExpertSlotPool::AcquiredSlot> gpu;
        };

        /** @brief Return exact resident count in one immutable layer bank. */
        std::size_t residentCount(
            const MoEOverlayParticipantLayerBank &layer) noexcept
        {
            return static_cast<std::size_t>(std::count(
                layer.resident_mask.begin(),
                layer.resident_mask.end(),
                true));
        }

        /** @brief Convert a GPU descriptor to the repack launcher's read view. */
        ExpertTierGpuConstProjectionView gpuConstView(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return {
                .payload = descriptor.ptrs.d_vnni,
                .scales = static_cast<const std::uint16_t *>(
                    descriptor.ptrs.d_scales),
                .mins = static_cast<const std::uint16_t *>(
                    descriptor.ptrs.d_mins),
                .emins = static_cast<const std::uint32_t *>(
                    descriptor.ptrs.d_emins),
                .payload_bytes = descriptor.vnni_bytes,
                .scales_bytes = descriptor.scales_bytes,
                .mins_bytes = descriptor.mins_bytes,
                .emins_bytes = descriptor.emins_bytes,
            };
        }

        /** @brief Convert a GPU descriptor to the repack launcher's write view. */
        ExpertTierGpuMutableProjectionView gpuMutableView(
            const GpuExpertPackedDescriptor &descriptor) noexcept
        {
            return {
                .payload = descriptor.ptrs.d_vnni,
                .scales = static_cast<std::uint16_t *>(
                    descriptor.ptrs.d_scales),
                .mins = static_cast<std::uint16_t *>(
                    descriptor.ptrs.d_mins),
                .emins = static_cast<std::uint32_t *>(
                    descriptor.ptrs.d_emins),
                .payload_bytes = descriptor.vnni_bytes,
                .scales_bytes = descriptor.scales_bytes,
                .mins_bytes = descriptor.mins_bytes,
                .emins_bytes = descriptor.emins_bytes,
            };
        }

        /** @brief Locate one role in a CPU slot lease. */
        const CpuExpertSlotPool::ProjectionLease &cpuProjection(
            const CpuExpertSlotPool::Lease &lease,
            ExpertTierWeightProjection projection)
        {
            const auto found = std::find_if(
                lease.projections.begin(),
                lease.projections.end(),
                [projection](const auto &candidate)
                { return candidate.projection == projection; });
            if (found == lease.projections.end())
                throw std::logic_error(
                    "ExpertOverlay CPU destination lease lost a projection");
            return *found;
        }

        /** @brief Locate one role in a GPU slot lease by its stable label. */
        const GpuExpertSlotPool::ProjectionSlot &gpuProjection(
            const GpuExpertSlotPool::AcquiredSlot &lease,
            ExpertTierWeightProjection projection)
        {
            const std::string label = projectionName(projection);
            const auto found = std::find_if(
                lease.projections.begin(),
                lease.projections.end(),
                [&label](const auto &candidate)
                { return candidate.spec.label == label; });
            if (found == lease.projections.end())
                throw std::logic_error(
                    "ExpertOverlay GPU destination lease lost a projection");
            return *found;
        }

        /**
         * @brief Determine the largest whole CPU unit fitting one lane chunk.
         * @throws std::runtime_error when even one indivisible unit cannot fit.
         */
        std::uint32_t maximumUnitsPerChunk(
            const ExpertTierWeightStreamManifest &probe,
            std::size_t staging_capacity_bytes)
        {
            const auto layout = probe.deviceLayout();
            if (!layout.valid() || layout.cpu_block_stride == 0 ||
                staging_capacity_bytes < layout.cpu_block_stride)
            {
                throw std::runtime_error(
                    "ExpertOverlay staging capacity cannot hold one CPU NativeQuantised unit");
            }
            return static_cast<std::uint32_t>(std::min<std::size_t>(
                layout.unit_count,
                staging_capacity_bytes / layout.cpu_block_stride));
        }

        /** @brief Require identical final CPU representations before NUMA copy. */
        void requireCompatibleCpuWeights(
            const cpu::native_quantised::CPUNativeQuantisedPackedWeights &source,
            const cpu::native_quantised::CPUNativeQuantisedPackedWeights &destination)
        {
            if (source.N != destination.N || source.K != destination.K ||
                source.N_padded != destination.N_padded ||
                source.blocks_per_row != destination.blocks_per_row ||
                source.codebook_id != destination.codebook_id ||
                source.is_asymmetric != destination.is_asymmetric ||
                source.is_superblock != destination.is_superblock ||
                source.encoding != destination.encoding ||
                source.data_stride != destination.data_stride ||
                source.interleaved_block_stride !=
                    destination.interleaved_block_stride ||
                source.native_interleaved.size() !=
                    destination.native_interleaved.size())
            {
                throw std::runtime_error(
                    "ExpertOverlay CPU participants have incompatible final packed representations");
            }
        }

        /** @brief Build one capacity-aware destination descriptor over a GPU slot. */
        GpuExpertPackedDescriptor makeGpuDestinationDescriptor(
            const GpuExpertSlotPool::ProjectionSlot &projection,
            int N,
            int K,
            std::uint8_t execution_codebook,
            std::uint8_t payload_bytes,
            bool is_asymmetric,
            bool has_emins,
            NativeQuantisedSourceIdentity source_identity)
        {
            if (projection.spec.N != N || projection.spec.K != K ||
                projection.spec.payload_bytes_per_block < payload_bytes ||
                (is_asymmetric && !projection.spec.is_asymmetric) ||
                (has_emins && !projection.spec.has_emins))
            {
                throw std::runtime_error(
                    "ExpertOverlay GPU shadow slot cannot represent an arriving projection");
            }

            DeviceNativeQuantisedMatrixDesc descriptor;
            descriptor.payload = projection.slot.d_native_quantised_payload;
            descriptor.scales = projection.slot.d_native_quantised_scales;
            descriptor.mins = is_asymmetric
                                  ? projection.slot.d_native_quantised_mins
                                  : nullptr;
            descriptor.emins = has_emins
                                   ? projection.slot.d_native_quantised_emins
                                   : nullptr;
            descriptor.n = N;
            descriptor.k = K;
            descriptor.blocks_per_row = static_cast<std::uint32_t>(K / 32);
            descriptor.codebook_id = execution_codebook;
            descriptor.allocation_payload_bytes_per_block =
                static_cast<std::uint8_t>(
                    projection.spec.payload_bytes_per_block);
            descriptor.allocation_has_mins =
                projection.spec.is_asymmetric ? 1u : 0u;
            descriptor.allocation_has_emins =
                projection.spec.has_emins ? 1u : 0u;
            descriptor.source_codebook_id = source_identity.codebook_id;
            descriptor.source_is_superblock = static_cast<std::uint8_t>(
                source_identity.is_superblock);
            descriptor.source_identity_present = static_cast<std::uint8_t>(
                source_identity.present);

            auto packed = makeGpuExpertPackedDescriptor(
                descriptor,
                payload_bytes,
                is_asymmetric,
                has_emins);
            if (!packed.valid() ||
                packed.vnni_bytes > projection.slot.payload_bytes ||
                packed.scales_bytes > projection.slot.scales_bytes ||
                packed.mins_bytes > projection.slot.mins_bytes ||
                packed.emins_bytes > projection.slot.emins_bytes)
            {
                throw std::runtime_error(
                    "ExpertOverlay GPU shadow descriptor exceeds its persistent allocation");
            }
            return packed;
        }

        /** @brief Construct the backend GEMM alias over one inactive GPU slot. */
        std::shared_ptr<ITensorGemm> makeGpuDestinationEngine(
            DeviceId device,
            const GpuExpertPackedDescriptor &descriptor,
            std::shared_ptr<void> lifetime,
            NativeQuantisedSourceIdentity source_identity)
        {
            if (!lifetime || !source_identity.present)
                throw std::invalid_argument(
                    "ExpertOverlay GPU destination engine requires slot and provenance ownership");
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                return std::make_shared<cuda::CUDAQuantisedGemmKernel>(
                    descriptor.n,
                    descriptor.k,
                    device.cuda_ordinal(),
                    descriptor.ptrs.d_vnni,
                    static_cast<std::uint16_t *>(descriptor.ptrs.d_scales),
                    static_cast<std::uint16_t *>(descriptor.ptrs.d_mins),
                    static_cast<std::uint32_t *>(descriptor.ptrs.d_emins),
                    descriptor.codebook_id,
                    descriptor.blocks_per_row,
                    std::move(lifetime),
                    source_identity,
                    NativeQuantisedReusableDeviceAllocationFormat{
                        .payload_bytes_per_block =
                            descriptor.allocation_payload_bytes_per_block,
                        .has_mins = descriptor.allocation_has_mins,
                        .has_emins = descriptor.allocation_has_emins,
                    });
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                return std::make_shared<rocm::ROCmQuantisedGemmKernel>(
                    descriptor.n,
                    descriptor.k,
                    device.rocm_ordinal(),
                    descriptor.ptrs.d_vnni,
                    descriptor.ptrs.d_scales,
                    descriptor.ptrs.d_mins,
                    descriptor.ptrs.d_emins,
                    descriptor.codebook_id,
                    descriptor.blocks_per_row,
                    std::move(lifetime),
                    source_identity,
                    NativeQuantisedReusableDeviceAllocationFormat{
                        .payload_bytes_per_block =
                            descriptor.allocation_payload_bytes_per_block,
                        .has_mins = descriptor.allocation_has_mins,
                        .has_emins = descriptor.allocation_has_emins,
                    });
            }
#endif
            throw std::runtime_error(
                "ExpertOverlay GPU destination backend was not built");
        }

        /** @brief Bind one inactive raw slot as a floating projection view. */
        ContiguousFloatingPointWeightDescriptor
        makeGpuFloatingDestinationDescriptor(
            const GpuExpertSlotPool::ProjectionSlot &projection,
            const ExpertWeightFormat &format)
        {
            const auto type = format.floatingTensorType();
            const std::size_t element_bytes =
                format.floatingElementBytes();
            if (!type || projection.spec.format != format ||
                projection.spec.N <= 0 || projection.spec.K <= 0 ||
                !projection.slot.d_native_quantised_payload ||
                element_bytes == 0)
            {
                throw std::runtime_error(
                    "ExpertOverlay GPU floating shadow slot has incompatible format or storage");
            }
            const std::size_t bytes =
                static_cast<std::size_t>(projection.spec.N) *
                static_cast<std::size_t>(projection.spec.K) *
                element_bytes;
            ContiguousFloatingPointWeightDescriptor descriptor{
                .data = projection.slot.d_native_quantised_payload,
                .type = *type,
                .n = projection.spec.N,
                .k = projection.spec.K,
                .bytes = bytes,
            };
            if (!descriptor.valid() ||
                descriptor.bytes > projection.slot.payload_bytes)
            {
                throw std::runtime_error(
                    "ExpertOverlay GPU floating descriptor exceeds its persistent allocation");
            }
            return descriptor;
        }

        /** @brief Construct a backend floating GEMM alias over an inactive slot. */
        std::shared_ptr<ITensorGemm> makeGpuFloatingDestinationEngine(
            DeviceId device,
            const ContiguousFloatingPointWeightDescriptor &descriptor,
            std::shared_ptr<void> lifetime)
        {
            if (!descriptor.valid() || !lifetime)
                throw std::invalid_argument(
                    "ExpertOverlay GPU floating destination requires valid storage and lifetime ownership");
#ifdef HAVE_CUDA
            if (device.is_cuda())
            {
                cuda::CUDAFloatingPointGemmKernel::Precision precision;
                switch (descriptor.type)
                {
                case TensorType::FP16:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP16;
                    break;
                case TensorType::BF16:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::BF16;
                    break;
                case TensorType::FP32:
                    precision = cuda::CUDAFloatingPointGemmKernel::Precision::FP32;
                    break;
                default:
                    throw std::invalid_argument(
                        "ExpertOverlay CUDA floating destination precision is unsupported");
                }
                return std::make_shared<cuda::CUDAFloatingPointGemmKernel>(
                    descriptor.data,
                    descriptor.n,
                    descriptor.k,
                    device.cuda_ordinal(),
                    precision,
                    std::move(lifetime));
            }
#endif
#ifdef HAVE_ROCM
            if (device.is_rocm())
            {
                rocm::ROCmFloatingPointGemmKernel::Precision precision;
                switch (descriptor.type)
                {
                case TensorType::FP16:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP16;
                    break;
                case TensorType::BF16:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::BF16;
                    break;
                case TensorType::FP32:
                    precision = rocm::ROCmFloatingPointGemmKernel::Precision::FP32;
                    break;
                default:
                    throw std::invalid_argument(
                        "ExpertOverlay ROCm floating destination precision is unsupported");
                }
                return std::make_shared<rocm::ROCmFloatingPointGemmKernel>(
                    descriptor.data,
                    descriptor.n,
                    descriptor.k,
                    device.rocm_ordinal(),
                    precision,
                    std::move(lifetime));
            }
#endif
            throw std::runtime_error(
                "ExpertOverlay GPU floating destination backend was not built");
        }

        /**
         * @brief Create exactly one final native GPU role owner and verify every later piece agrees.
         * @param storage Same destination-member/role storage, never an independent per-piece engine.
         * @param device Original exact destination GPU.
         * @param descriptor Complete final local root descriptor, not compact wire dimensions.
         * @param lifetime Existing reserved final-slot authority.
         * @param source Original mathematical codebook identity.
         * @return Same genuine final engine control block for all pieces.
         */
        std::shared_ptr<ITensorGemm> retainNativeGpuDestination(
            DestinationProjectionStorage &storage, DeviceId device,
            const GpuExpertPackedDescriptor &descriptor, std::shared_ptr<void> lifetime,
            NativeQuantisedSourceIdentity source)
        {
            if (!storage.engine)
                storage.engine = makeGpuDestinationEngine(device, descriptor, std::move(lifetime), source);
            MoEOverlayPreparedWeightSource actual;
            std::string error;
            if (!resolveMoEOverlayPreparedWeightSource(storage.engine, device, actual, &error) ||
                actual.kind != MoEOverlayPreparedWeightSourceKind::GpuSeparatedNativeQuantised ||
                actual.format.native_quantised != source)
                throw std::invalid_argument("Final GPU role owner changed its original native source identity");
            const auto &a = actual.gpu_packed, &b = descriptor;
            if (a.ptrs.d_vnni != b.ptrs.d_vnni || a.ptrs.d_scales != b.ptrs.d_scales ||
                a.ptrs.d_mins != b.ptrs.d_mins || a.ptrs.d_emins != b.ptrs.d_emins || a.n != b.n || a.k != b.k ||
                a.blocks_per_row != b.blocks_per_row || a.codebook_id != b.codebook_id ||
                a.payload_bytes_per_block != b.payload_bytes_per_block || a.is_asymmetric != b.is_asymmetric ||
                a.has_emins != b.has_emins || a.vnni_bytes != b.vnni_bytes || a.scales_bytes != b.scales_bytes ||
                a.mins_bytes != b.mins_bytes || a.emins_bytes != b.emins_bytes)
                throw std::invalid_argument("Physical pieces disagree on one final GPU role allocation/encoding");
            return storage.engine;
        }

        /** @brief Retain exactly one final floating GPU root; compact pieces never alter its stride. */
        std::shared_ptr<ITensorGemm> retainFloatingGpuDestination(
            DestinationProjectionStorage &storage, DeviceId device,
            const ContiguousFloatingPointWeightDescriptor &descriptor, std::shared_ptr<void> lifetime)
        {
            if (!storage.engine)
                storage.engine = makeGpuFloatingDestinationEngine(device, descriptor, std::move(lifetime));
            MoEOverlayPreparedWeightSource actual;
            std::string error;
            if (!resolveMoEOverlayPreparedWeightSource(storage.engine, device, actual, &error) ||
                !actual.format.isFloating() || actual.floating.data != descriptor.data ||
                actual.floating.n != descriptor.n || actual.floating.k != descriptor.k ||
                actual.floating.type != descriptor.type || actual.floating.bytes != descriptor.bytes)
                throw std::invalid_argument("Physical pieces disagree on one final floating GPU role allocation");
            return storage.engine;
        }

        /** @brief Compare source geometry/provenance independent of execution layout. */
        ProjectionSignature signatureFor(
            const MoEOverlayPreparedWeightSource &source)
        {
            ProjectionSignature signature;
            signature.format = source.format;
            if (source.kind ==
                    MoEOverlayPreparedWeightSourceKind::CpuNativeQuantised)
            {
                signature.N = source.cpu_packed->N;
                signature.K = source.cpu_packed->K;
            }
            else if (source.kind ==
                     MoEOverlayPreparedWeightSourceKind::
                         GpuSeparatedNativeQuantised)
            {
                signature.N = source.gpu_packed.n;
                signature.K = source.gpu_packed.k;
            }
            else
            {
                signature.N = source.floating.n;
                signature.K = source.floating.k;
            }
            return signature;
        }

        /** @brief Convert projection role to an array index after validation. */
        std::size_t projectionIndex(
            ExpertTierWeightProjection projection)
        {
            const auto index = static_cast<std::size_t>(projection);
            if (index >= kProjections.size())
                throw std::invalid_argument(
                    "ExpertOverlay projection role is outside gate/up/down");
            return index;
        }

        /**
         * @brief Convert one immutable initial layer bank into recyclable slots.
         * @param layer Initial participant-local prepared engines.
         * @param device Exact endpoint device.
         * @param epoch Initial residency epoch.
         * @return One adopted physical slot for every resident expert.
         */
        std::vector<AdoptedInitialExpertSlotRecycler::InitialSlot>
        makeAdoptedInitialSlots(
            const MoEOverlayParticipantLayerBank &layer,
            int layer_idx,
            DeviceId device,
            std::uint64_t epoch)
        {
            std::vector<AdoptedInitialExpertSlotRecycler::InitialSlot> result;
            result.reserve(residentCount(layer));
            for (std::size_t expert = 0; expert < layer.resident_mask.size(); ++expert)
            {
                if (!layer.resident_mask[expert])
                    continue;

                AdoptedInitialExpertSlotRecycler::InitialSlot slot;
                slot.layer_idx = layer_idx;
                slot.expert_id = static_cast<int>(expert);
                slot.residency_epoch = epoch;
                slot.projections.reserve(kProjections.size());
                for (const auto projection : kProjections)
                {
                    MoEOverlayPreparedWeightSource source;
                    std::string source_error;
                    if (!resolveMoEOverlayPreparedWeightSource(
                            projectionEngine(layer, expert, projection),
                            device,
                            source,
                            &source_error))
                    {
                        throw std::runtime_error(
                            source_error.empty()
                                ? "ExpertOverlay could not adopt an initial prepared projection"
                                : std::move(source_error));
                    }

                    AdoptedInitialExpertSlotRecycler::Projection adopted;
                    adopted.projection = projection;
                    adopted.engine = source.engine;
                    if (device.is_cpu())
                    {
                        if (source.format.isFloating())
                        {
                            adopted.cpu_destination =
                                source.engine
                                    ->exportRetiredCPUFloatingPointStorage();
                            if (!source.floating.valid() ||
                                adopted.cpu_destination.empty() ||
                                adopted.cpu_destination.data() !=
                                    source.floating.data ||
                                adopted.cpu_destination.size() !=
                                    source.floating.bytes)
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay initial CPU floating engine cannot expose its exact recyclable final storage");
                            }
                        }
                        else
                        {
                            adopted.cpu_destination =
                                source.engine
                                    ->exportRetiredCPUNativeQuantisedStorage();
                            if (!source.cpu_packed ||
                                adopted.cpu_destination.empty() ||
                                adopted.cpu_destination.data() !=
                                    source.cpu_packed->native_interleaved.data() ||
                                adopted.cpu_destination.size() !=
                                    source.cpu_packed->native_interleaved.size())
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay initial CPU NativeQuantised engine cannot expose its exact recyclable final storage");
                            }
                        }
                    }
                    else
                    {
                        if (source.format.isFloating())
                        {
                            GpuExpertSlotPool::ProjectionSlot projection_slot;
                            projection_slot.spec = {
                                .label = projectionName(projection),
                                .N = source.floating.n,
                                .K = source.floating.k,
                                .format = source.format,
                            };
                            projection_slot.slot = {
                                .d_native_quantised_payload =
                                    static_cast<std::uint8_t *>(
                                        const_cast<void *>(
                                            source.floating.data)),
                                .payload_bytes = source.floating.bytes,
                                .staging_bytes = 0,
                            };
                            projection_slot.blocks_per_row = 0;
                            adopted.gpu_destination =
                                std::move(projection_slot);
                            slot.projections.push_back(std::move(adopted));
                            continue;
                        }
                        const auto *source_format =
                            native_quantised_formats::forSourceIdentity(
                                source.format.native_quantised.codebook_id,
                                source.format.native_quantised.is_superblock);
                        if (!source_format)
                        {
                            throw std::runtime_error(
                                "ExpertOverlay initial GPU expert lacks catalogued source provenance");
                        }
                        const auto allocation =
                            reusableDeviceVnniAllocationFormat(*source_format);
                        if (!source.gpu_packed.valid() ||
                            source.gpu_packed
                                    .allocation_payload_bytes_per_block !=
                                allocation.payload_bytes_per_block ||
                            source.gpu_packed.allocation_has_mins !=
                                allocation.has_mins ||
                            source.gpu_packed.allocation_has_emins !=
                                allocation.has_emins)
                        {
                            throw std::runtime_error(
                                "ExpertOverlay initial GPU expert lacks the exact reusable allocation charged by capacity admission: device=" +
                                device.to_string() +
                                " expert=" + std::to_string(expert) +
                                " projection=" + projectionName(projection) +
                                " source_codebook=" +
                                std::to_string(source.format.native_quantised.codebook_id) +
                                " observed_payload_bytes_per_block=" +
                                std::to_string(source.gpu_packed.allocation_payload_bytes_per_block) +
                                " expected_payload_bytes_per_block=" +
                                std::to_string(allocation.payload_bytes_per_block) +
                                " observed_has_mins=" +
                                std::to_string(source.gpu_packed.allocation_has_mins) +
                                " expected_has_mins=" +
                                std::to_string(allocation.has_mins) +
                                " observed_has_emins=" +
                                std::to_string(source.gpu_packed.allocation_has_emins) +
                                " expected_has_emins=" +
                                std::to_string(allocation.has_emins));
                        }

                        GpuExpertSlotPool::ProjectionSlot projection_slot;
                        projection_slot.spec = {
                            .label = projectionName(projection),
                            .N = source.gpu_packed.n,
                            .K = source.gpu_packed.k,
                            .payload_bytes_per_block =
                                allocation.payload_bytes_per_block,
                            .is_asymmetric =
                                allocation.has_mins,
                            .has_emins = allocation.has_emins,
                            .codebook_id = source.gpu_packed.codebook_id,
                            .format = source.format,
                        };
                        projection_slot.slot = {
                            .d_native_quantised_payload =
                                source.gpu_packed.ptrs.d_vnni,
                            .d_native_quantised_scales =
                                source.gpu_packed.ptrs.d_scales,
                            .d_native_quantised_mins =
                                source.gpu_packed.ptrs.d_mins,
                            .d_native_quantised_emins =
                                source.gpu_packed.ptrs.d_emins,
                            .payload_bytes =
                                source.gpu_packed.allocationPayloadBytes(),
                            .scales_bytes = source.gpu_packed.scales_bytes,
                            .mins_bytes =
                                source.gpu_packed.allocationMinsBytes(),
                            .emins_bytes =
                                source.gpu_packed.allocationEminsBytes(),
                            .staging_bytes = 0,
                        };
                        projection_slot.blocks_per_row =
                            source.gpu_packed.blocks_per_row;
                        adopted.gpu_destination =
                            std::move(projection_slot);
                    }
                    slot.projections.push_back(std::move(adopted));
                }
                result.push_back(std::move(slot));
            }
            return result;
        }

        /** @brief Present one final CPU packed allocation as a wire source. */
        std::array<
            std::span<const std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount>
        remoteCpuSourceRegions(
            const cpu::native_quantised::CPUNativeQuantisedPackedWeights &packed)
        {
            return {
                std::span<const std::uint8_t>(
                    packed.native_interleaved.data(),
                    packed.native_interleaved.size()),
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
            };
        }

        /** @brief Present one contiguous host matrix as a wire source. */
        std::array<
            std::span<const std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount>
        remoteContiguousSourceRegions(
            std::span<const std::uint8_t> source)
        {
            return {
                source,
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
                std::span<const std::uint8_t>{},
            };
        }

        /** @brief Present one final CPU slot allocation as a wire destination. */
        std::array<
            std::span<std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount>
        remoteCpuDestinationRegions(std::span<std::uint8_t> destination)
        {
            return {
                destination,
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
            };
        }

        /** @brief Present one contiguous host matrix as a wire destination. */
        std::array<
            std::span<std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount>
        remoteContiguousDestinationRegions(
            std::span<std::uint8_t> destination)
        {
            return {
                destination,
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
                std::span<std::uint8_t>{},
            };
        }

        /** @brief Derive one canonical cross-rank identity from the transaction. */
        MoEOverlayRemoteProjectionIdentity remoteProjectionIdentity(
            std::uint64_t expected_epoch,
            std::uint64_t candidate_epoch,
            const MoEOverlayResidencyExecutionFingerprint &
                execution_fingerprint,
            const std::vector<MoEOverlayTierMigration> &migrations,
            std::size_t migration_index,
            const MoEOverlayPhysicalProjectionPiece &piece,
            const MoEExpertOwnerMap &topology)
        {
            const auto &migration = migrations[migration_index];
            const auto &source = *topology.participantForId(piece.source_participant);
            const auto &destination = *topology.participantForId(piece.destination_participant);
            return {
                .expected_epoch = expected_epoch,
                .candidate_epoch = candidate_epoch,
                .execution_fingerprint = execution_fingerprint,
                .migration_index = migration_index,
                .layer_idx = migration.layer_idx,
                .expert_id = migration.expert_id,
                .projection = piece.projection,
                .source_participant =
                    piece.source_participant,
                .destination_participant =
                    piece.destination_participant,
                .source_world_rank = source.world_rank,
                .destination_world_rank =
                    destination.world_rank,
                .source_device = source.device,
                .destination_device = destination.device,
            };
        }
    } // namespace

    /** @brief Private materialized maps and lock-free evidence counters. */
    struct MoEOverlayPhysicalResidencyFabric::Impl
    {
        /** Typed one-way lifecycle for terminal physical canonicalization. */
        enum class ReusableSealState
        {
            Open,    ///< Ordinary migration may still use the fabric.
            Sealing, ///< One terminal caller owns canonicalization.
            Sealed,  ///< cached_reusable_seal is complete and immutable.
            Failed,  ///< First terminal failure is sticky and unrecoverable.
        };

        /** Original authenticated layer metadata retained from model-time pool construction. */
        std::vector<MoEOverlayLayerWeightManifest> original_layer_manifests;
        MoEOverlayPhysicalLaneDemandPlan lane_demand_plan;
        std::map<EndpointLayerKey, EndpointLayerPool> endpoint_pools;
        std::map<EdgeKey, std::shared_ptr<SharedWeightLanePool>> weight_lanes;
        std::map<EdgeKey, std::shared_ptr<SharedBlobLanePool>> blob_lanes;
        std::map<EdgeKey, std::shared_ptr<SharedPeerLanePool>> peer_lanes;
        /** One topology-sized retained relay epoch for each participating GPU. */
        std::map<DeviceId, std::shared_ptr<MappedTransferProgressEpoch>>
            transfer_progress_epochs;
        std::map<
            CpuAddressEdgeKey,
            std::shared_ptr<SharedCpuCopyLanePool>>
            cpu_copy_lanes;
        std::map<
            RemoteGpuLaneKey,
            std::vector<std::shared_ptr<MoEOverlayGpuRemoteProjectionLane>>>
            remote_gpu_lanes;
        std::shared_ptr<CpuCopyConcurrencyTracker> cpu_copy_concurrency =
            std::make_shared<CpuCopyConcurrencyTracker>();
        /**
         * Physical-only lifetime state for device-authored epochs. It is seeded
         * once from loader banks and never stores a histogram or owner map.
         */
        std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger>
            device_slot_ledger;

        /** Serializes the sole Open -> Sealing -> Sealed/Failed transition. */
        std::mutex reusable_seal_mutex;
        ReusableSealState reusable_seal_state = ReusableSealState::Open;
        std::optional<MoEOverlayReusableContextSeal> cached_reusable_seal;
        std::string reusable_seal_failure;

        std::atomic<std::uint64_t> endpoint_layer_pools{0};
        std::atomic<std::uint64_t> endpoint_geometry_pools{0};
        std::atomic<std::uint64_t> adopted_initial_slots{0};
        std::atomic<std::uint64_t> adopted_initial_slots_recycled{0};
        std::atomic<std::uint64_t> cpu_shadow_slots{0};
        std::atomic<std::uint64_t> gpu_shadow_slots{0};
        std::atomic<std::uint64_t> persistent_transfer_lanes{0};
        std::atomic<std::uint64_t> direct_gpu_peer_lanes{0};
        std::atomic<std::uint64_t> same_backend_no_peer_relay_lanes{0};
        std::atomic<std::uint64_t> cross_backend_gpu_relay_lanes{0};
        std::atomic<std::uint64_t> maximum_parallel_edge_lanes{0};
        std::atomic<std::uint64_t> maximum_parallel_cpu_copy_lanes{0};
        std::atomic<std::uint64_t> waves_prepared{0};
        std::atomic<std::uint64_t> waves_deferred{0};
        std::atomic<std::uint64_t> waves_failed{0};
        std::atomic<std::uint64_t> projection_operations_prepared{0};
        std::atomic<std::uint64_t> cpu_copy_operations{0};
        std::atomic<std::uint64_t> remote_cpu_operations{0};
        std::atomic<std::uint64_t> remote_gpu_cpu_operations{0};
        std::atomic<std::uint64_t> remote_gpu_blob_operations{0};
        std::atomic<std::uint64_t> gpu_cpu_operations{0};
        std::atomic<std::uint64_t> same_backend_gpu_operations{0};
        std::atomic<std::uint64_t> heterogeneous_gpu_operations{0};
    };

    namespace
    {
        /** @brief One possibly-unresolved projection contract per model layer. */
        using LayerProjectionSignatures =
            std::array<std::optional<ProjectionSignature>, 3>;

        /** @brief Authenticate or install one device-independent projection. */
        void mergeProjectionSignature(
            std::map<int, LayerProjectionSignatures> &signatures,
            int layer_idx,
            ExpertTierWeightProjection projection,
            const ProjectionSignature &candidate)
        {
            auto &slot = signatures[layer_idx][projectionIndex(projection)];
            if (slot && *slot != candidate)
            {
                throw std::runtime_error(
                    "ExpertOverlay participants or published manifest disagree on model projection geometry or provenance");
            }
            slot = candidate;
        }

        /** @brief Resolve the exact payload contract for one model layer. */
        std::array<ProjectionSignature, 3> requireLayerSignatures(
            const std::map<int, LayerProjectionSignatures> &signatures,
            int layer_idx)
        {
            const auto found = signatures.find(layer_idx);
            if (found == signatures.end() ||
                std::any_of(
                    found->second.begin(),
                    found->second.end(),
                    [](const auto &entry) { return !entry.has_value(); }))
            {
                throw std::runtime_error(
                    "ExpertOverlay empty receiving tier requires a complete globally-published layer weight manifest");
            }
            return {
                *found->second[0],
                *found->second[1],
                *found->second[2],
            };
        }

        /** @return Existing immutable cached directed fact; this never probes a native backend. */
        std::optional<bool> observedGpuPeerAccess(DeviceId destination,DeviceId source)
        {
            return DeviceManager::instance().peerAccessAvailable(destination,source);
        }

        /**
         * @brief Authenticate the complete original source/choice declaration before pool creators.
         * @param config Original source/PMA setup, with no live ownership or byte decisions.
         * @param impl Retains the exact validated declaration for all later creators.
         * @throws std::exception For missing/foreign source geometry or changed directed evidence.
         */
        void authenticatePhysicalLaneDemand(const MoEOverlayPhysicalResidencyFabric::Config& config,
            MoEOverlayPhysicalResidencyFabric::Impl& impl)
        {
            if(!config.initial_snapshot->placement_plan)
                throw std::logic_error("Physical lane setup lost the original declared topology");
            const int world_rank=config.memory_authority->worldRank();
            const auto lane_scope=config.remote_projection_transport
                ?MoEOverlayPhysicalLaneScope::communicator(config.remote_projection_transport->worldSize())
                :MoEOverlayPhysicalLaneScope::rankLocal(world_rank);
            impl.lane_demand_plan=configuredMoEOverlayLaneDemand(*config.initial_snapshot->placement_plan,
                impl.original_layer_manifests,originalFabricGeometry(config).expertCount(),lane_scope,
                config.maximum_concurrent_cycles,config.maximum_cycles_per_layer);
            const auto local_ids=config.registry->localParticipantIds();
            std::map<DeviceId,size_t> actual_gpu_streams;
            for(const auto id:local_ids) {
                const auto endpoint=config.registry->endpoint(id);
                if(!endpoint)throw std::logic_error("Physical stream owner is missing its original local endpoint");
                if(endpoint->device().is_gpu()) {
                    auto& count=actual_gpu_streams[endpoint->device()];
                    count=checkedMoEOverlayLaneAdd(count,config.maximum_execution_streams);
                }
            }
            const auto declared_streams=MoEOverlayCapacityAdmission::executionStreamCounts(
                *config.initial_snapshot->placement_plan,world_rank,config.maximum_execution_streams);
            if(actual_gpu_streams!=declared_streams)
                throw std::invalid_argument("Physical movement GPU cohort differs from its original bound participant roster");
            impl.lane_demand_plan.original_execution_streams=declared_streams;
            const bool local_gpu=!actual_gpu_streams.empty();
            if(local_gpu) {
                if(declared_streams.empty())throw std::invalid_argument("Physical GPU stream owners are not bound to their original PMA rank");
                requireOriginalMoEOverlayGpuDemand(config.physical_gpu_lane_demand.get(),impl.lane_demand_plan,
                    world_rank,observedGpuPeerAccess);
                impl.lane_demand_plan=*config.physical_gpu_lane_demand;
            }
        }

        /** @brief Materialize every local endpoint/layer from one model manifest. */
        void materializeEndpointPools(
            const MoEOverlayPhysicalResidencyFabric::Config &config,
            MoEOverlayPhysicalResidencyFabric::Impl &impl)
        {

            const auto local_ids=config.registry->localParticipantIds();
            std::map<int, LayerProjectionSignatures> model_signatures;
            std::map<
                int,
                MoEOverlayParticipantBankLease>
                initial_banks;
            std::vector<MoEOverlayDeviceInitialPhysicalSlot> initial_device_slots;
            std::vector<MoEOverlayDeviceFragmentPhysicalSlot> initial_fragment_slots;
            std::vector<MoEOverlayDevicePhysicalSlotLedger::FragmentFamily> original_fragment_families;

            const auto ownership = originalFabricGeometry(config);
            if (!config.layer_weight_manifest.empty())
            {
                if (config.layer_weight_manifest.size() !=
                    static_cast<std::size_t>(ownership.layerCount()))
                    throw std::invalid_argument(
                        "ExpertOverlay layer weight manifest does not cover the owned stage");
                for (std::size_t row = 0; row < config.layer_weight_manifest.size(); ++row)
                    if (config.layer_weight_manifest[row].layer_idx !=
                        ownership.firstModelLayer() + static_cast<int>(row))
                        throw std::invalid_argument(
                            "ExpertOverlay layer weight manifest belongs to another stage");
            }

            /*
             * Distributed setup may publish the model contract before this
             * process has any local source for a layer.  Install those entries
             * first, then authenticate every live local engine against them.
             */
            for (const auto &layer : config.layer_weight_manifest)
            {
                if (!layer.valid())
                    throw std::invalid_argument(
                        "ExpertOverlay layer weight manifest is invalid or incomplete");
                if (model_signatures.count(layer.layer_idx) != 0)
                    throw std::invalid_argument(
                        "ExpertOverlay layer weight manifest contains a duplicate layer");
                for (const auto &projection : layer.projections)
                {
                    mergeProjectionSignature(
                        model_signatures,
                        layer.layer_idx,
                        projection.projection,
                        ProjectionSignature{
                            .N = projection.N,
                            .K = projection.K,
                            .format = projection.format,
                        });
                }
            }

            /* Pass one discovers and validates the process-local live sources. */
            for (const int participant_id : local_ids)
            {
                const auto endpoint = config.registry->endpoint(participant_id);
                const auto *participant =
                    config.initial_snapshot->placementOwnerMap().participantForId(
                        participant_id);
                const auto bank = endpoint
                                      ? endpoint->acquire(
                                            config.initial_snapshot->epoch)
                                      : MoEOverlayParticipantBankLease{};
                if (!endpoint || !participant || !bank ||
                    bank->device != endpoint->device() ||
                    endpoint->firstModelLayer() != ownership.firstModelLayer() ||
                    endpoint->numLayers() != ownership.layerCount() ||
                    endpoint->numExperts() != ownership.expertCount() ||
                    !bank->valid(participant_id, endpoint->device(),
                                 endpoint->numLayers(), endpoint->numExperts(),
                                 endpoint->firstModelLayer()))
                {
                    throw std::runtime_error(
                        "ExpertOverlay physical fabric requires every local initial bank with matching stage geometry");
                }
                initial_banks.emplace(participant_id, bank);

                for (int layer_idx = endpoint->firstModelLayer(); layer_idx < endpoint->endModelLayer(); ++layer_idx)
                {
                    const auto &layer = bank->layerForModelLayer(layer_idx);
                    std::optional<MoEOverlayLayerWeightManifest> local_fragment;
                    if (layer.fragment_family)
                    {
                        const auto &family = *layer.fragment_family;
                        const auto source_preparation = config.initial_snapshot->originalProjectionPreparation();
                        const auto &prepared = family.preparation();
                        if (!source_preparation ||
                            prepared.get() != source_preparation.get() ||
                            prepared.owner_before(source_preparation) || source_preparation.owner_before(prepared) ||
                            !family.cohort() || family.participantIndex() >= family.cohort()->participants().size() ||
                            family.cohort()->participants()[family.participantIndex()].participant_id != participant_id ||
                            family.cohort()->source().layer_idx != layer_idx ||
                            family.cohort()->ownership(family.participantIndex()).geometry().experts != ownership.expertCount())
                            throw std::invalid_argument("Physical initial fragment bank lost its original preparation/member geometry");
                        for (const auto &projection : family.cohort()->source().projections)
                            mergeProjectionSignature(model_signatures, layer_idx, projection.projection,
                                {.N = projection.N, .K = projection.K, .format = projection.format});
                        local_fragment = family.cohort()->localManifest(family.participantIndex());
                        original_fragment_families.push_back({.participant_id = participant_id,
                            .layer_idx = layer_idx, .family = layer.fragment_family});
                    }
                    for (std::size_t expert = 0; expert < layer.resident_mask.size(); ++expert)
                    {
                        if (!layer.resident_mask[expert]) continue;
                        if (!local_fragment)
                            initial_device_slots.push_back({
                                .key = {.participant_id = participant_id, .layer_idx = layer_idx,
                                    .expert_id = static_cast<int>(expert)},
                                .entered_epoch = config.initial_snapshot->epoch,
                                .bootstrap_allocation = true, .payload = layer.experts.at(expert)});
                        else
                            initial_fragment_slots.push_back({
                                .key = {.participant_id = participant_id, .layer_idx = layer_idx,
                                    .expert_id = static_cast<int>(expert)},
                                .entered_epoch = config.initial_snapshot->epoch,
                                .bootstrap_allocation = true, .payload = layer.fragments.at(expert)});
                        for (const auto projection : kProjections)
                        {
                            MoEOverlayPreparedWeightSource source;
                            std::string source_error;
                            if (!resolveMoEOverlayPreparedWeightSource(
                                    projectionEngine(layer, expert, projection), endpoint->device(), source, &source_error))
                                throw std::runtime_error(source_error.empty()
                                    ? "ExpertOverlay initial engine cannot source migration" : source_error);
                            const auto signature = signatureFor(source);
                            if (local_fragment)
                            {
                                const auto local = std::find_if(local_fragment->projections.begin(), local_fragment->projections.end(),
                                    [projection](const auto &entry) { return entry.projection == projection; });
                                if (local == local_fragment->projections.end() || signature.N != local->N ||
                                    signature.K != local->K || signature.format != local->format)
                                    throw std::invalid_argument("Physical CPU source engine differs from original member-local manifest");
                            }
                            else mergeProjectionSignature(model_signatures, layer_idx, projection, signature);
                        }
                    }
                }
            }

            // Retain the already authenticated source contract once. This is
            // immutable geometry, not a second live residency or byte ledger.
            for (const auto &[layer, _] : model_signatures)
            {
                const auto signatures = requireLayerSignatures(model_signatures, layer);
                MoEOverlayLayerWeightManifest source{.layer_idx = layer};
                for (const auto role : kProjections)
                {
                    const auto index = projectionIndex(role);
                    source.projections[index] = {.projection = role,
                        .N = signatures[index].N, .K = signatures[index].K,
                        .format = signatures[index].format};
                }
                impl.original_layer_manifests.push_back(std::move(source));
            }

            // The preceding pass is declarative source authentication only.
            // Reject changed GPU incidence before any shadow/event/stream creator.
            authenticatePhysicalLaneDemand(config,impl);

            /*
             * From this point onward the device path resolves sources through
             * shared prepared-engine lifetimes, not through the setup registry.
             * The registry remains available solely to the distinct host-RCU
             * authority path.
             */
            impl.device_slot_ledger = MoEOverlayDevicePhysicalSlotLedger::createMixed({
                .whole = {.initial_epoch = config.initial_snapshot->epoch,
                    .local_participant_ids = local_ids,
                    .initial_slots = std::move(initial_device_slots),
                    .first_model_layer = ownership.firstModelLayer(),
                    .num_layers = static_cast<std::uint32_t>(ownership.layerCount()),
                    .num_experts = static_cast<std::uint32_t>(ownership.expertCount())},
                .families = std::move(original_fragment_families),
                .initial_fragments = std::move(initial_fragment_slots)});

            /*
             * Pass two groups layers by their exact gate/up/down allocation
             * contract. A closed migration cycle can visit one participant
             * only once, so layers with identical geometry share the bounded
             * inactive arena instead of each retaining an allocation that can
             * never be used concurrently. Loader-owned slots join that same
             * arena: after publication, the departed source from any compatible
             * layer replaces the inactive slot consumed by the arrival.
             */
            for (const int participant_id : local_ids)
            {
                const auto endpoint = config.registry->endpoint(participant_id);
                const auto *participant =
                    config.initial_snapshot->placementOwnerMap().participantForId(
                        participant_id);
                const auto bank_found = initial_banks.find(participant_id);
                if (!endpoint || !participant ||
                    bank_found == initial_banks.end())
                {
                    throw std::logic_error(
                        "ExpertOverlay physical fabric lost a validated local endpoint");
                }
                if (config.shadow_slots_per_endpoint_layer >
                    static_cast<std::size_t>(endpoint->numExperts()))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay shadow-slot capacity exceeds the model expert count");
                }

                std::map<ExpertGeometryKey, std::vector<int>>
                    layers_by_geometry;
                for (int layer_idx = endpoint->firstModelLayer();
                     layer_idx < endpoint->endModelLayer();
                     ++layer_idx)
                {
                    auto signatures = requireLayerSignatures(model_signatures, layer_idx);
                    const auto &member_layer = bank_found->second->layerForModelLayer(layer_idx);
                    if (member_layer.fragment_family)
                    {
                        const auto &family = *member_layer.fragment_family;
                        const auto local = family.cohort()->localManifest(family.participantIndex());
                        for (const auto &projection : local.projections)
                            signatures[projectionIndex(projection.projection)] = {
                                .N = projection.N, .K = projection.K, .format = projection.format};
                    }
                    layers_by_geometry[ExpertGeometryKey{signatures}]
                        .push_back(layer_idx);
                }

                for (const auto &[geometry, layer_indices] :
                     layers_by_geometry)
                {
                    if (layer_indices.empty() ||
                        layer_indices.size() >
                            std::numeric_limits<std::size_t>::max() /
                                config.shadow_slots_per_endpoint_layer)
                    {
                        throw std::overflow_error(
                            "ExpertOverlay shared shadow-slot capacity overflows size_t");
                    }
                    const std::size_t geometry_layer_capacity =
                        layer_indices.size() *
                        config.shadow_slots_per_endpoint_layer;
                    const std::size_t capacity = std::min(
                        config.maximum_concurrent_cycles,
                        geometry_layer_capacity);
                    if (capacity == 0 ||
                        capacity > static_cast<std::size_t>(
                            std::numeric_limits<int>::max()))
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay shared shadow-slot arena has invalid capacity");
                    }

                    std::vector<
                        AdoptedInitialExpertSlotRecycler::InitialSlot>
                        adopted_initial_slots;
                    for (const int layer_idx : layer_indices)
                    {
                        const auto &initial_layer =
                            bank_found->second->layerForModelLayer(layer_idx);
                        auto layer_slots = makeAdoptedInitialSlots(
                            initial_layer,
                            layer_idx,
                            endpoint->device(),
                            config.initial_snapshot->epoch);
                        adopted_initial_slots.insert(
                            adopted_initial_slots.end(),
                            std::make_move_iterator(layer_slots.begin()),
                            std::make_move_iterator(layer_slots.end()));
                    }
                    const std::size_t adopted_count =
                        adopted_initial_slots.size();
                    auto destination = std::make_shared<EndpointGeometryPool>();
                    destination->device = endpoint->device();
                    destination->participant_id = participant_id;
                    destination->layer_indices = layer_indices;
                    destination->shadow_capacity = capacity;
                    destination->adopted_slots = std::make_shared<
                        AdoptedInitialExpertSlotRecycler>(
                        endpoint->device(),
                        participant_id,
                        layer_indices,
                        std::move(adopted_initial_slots),
                        config.perf_device);
                    impl.adopted_initial_slots.fetch_add(
                        adopted_count, std::memory_order_relaxed);
                    if (endpoint->device().is_cpu())
                    {
                        std::vector<CpuExpertSlotPool::ProjectionSpec> specs;
                        specs.reserve(kProjections.size());
                        for (const auto projection : kProjections)
                        {
                            const auto &signature =
                                geometry.projections[
                                    projectionIndex(projection)];
                            specs.push_back({
                                .projection = projection,
                                .N = signature.N,
                                .K = signature.K,
                                .format = signature.format,
                            });
                        }

                        const auto memory_placement =
                            participant->address.hasValidNuma()
                                ? CpuExpertSlotPool::MemoryPlacement::boundNode(
                                      participant->address.numa_node)
                                : CpuExpertSlotPool::MemoryPlacement::aggregateDomain();
                        auto pool = CpuExpertSlotPool::create(
                            {
                                .participant_id = participant_id,
                                .layer_idx = layer_indices.front(),
                                .capacity = static_cast<int>(capacity),
                                .projections = std::move(specs),
                                .memory_placement = memory_placement,
                                .perf_device = config.perf_device,
                            },
                            config.memory_authority,
                            PhysicalMemoryOwner::ExpertShadowSlots);
                        destination->pool = std::move(pool);
                        impl.cpu_shadow_slots.fetch_add(
                            capacity, std::memory_order_relaxed);
                    }
                    else if (endpoint->device().is_gpu())
                    {
                        IBackend *backend = getBackendFor(endpoint->device());
                        if (!backend)
                            throw std::runtime_error(
                                "ExpertOverlay GPU participant has no backend authority");

                        std::vector<GpuExpertSlotPool::ProjectionSpec> specs;
                        specs.reserve(kProjections.size());
                        for (const auto projection : kProjections)
                        {
                            const auto &signature =
                                geometry.projections[
                                    projectionIndex(projection)];
                            GpuExpertSlotPool::ProjectionSpec spec{
                                .label = projectionName(projection),
                                .N = signature.N,
                                .K = signature.K,
                                .format = signature.format,
                            };
                            if (signature.format.isNativeQuantised())
                            {
                                /*
                                 * Expanded CPU arrivals need 32 payload bytes;
                                 * minima and extended minima cover every one of
                                 * the catalogued formats. Actual descriptors
                                 * expose only the regions their decoder uses.
                                 */
                                spec.payload_bytes_per_block = 32;
                                spec.is_asymmetric = true;
                                spec.has_emins = true;
                                spec.codebook_id =
                                    canonicalDeviceVnniCodebookId(signature.format.native_quantised);
                            }
                            specs.push_back(std::move(spec));
                        }
                        auto pool = GpuExpertSlotPool::create(
                            backend,
                            endpoint->device(),
                            endpoint->device().gpu_ordinal(),
                            layer_indices.front(),
                            static_cast<int>(capacity),
                            std::move(specs),
                            config.memory_authority,
                            PhysicalMemoryOwner::ExpertShadowSlots,
                            /*transfer_capacity=*/0);
                        destination->pool = std::move(pool);
                        impl.gpu_shadow_slots.fetch_add(
                            capacity, std::memory_order_relaxed);
                    }
                    else
                    {
                        throw std::runtime_error(
                            "ExpertOverlay physical fabric encountered an unsupported participant device");
                    }

                    impl.endpoint_geometry_pools.fetch_add(
                        1, std::memory_order_relaxed);
                    for (const int layer_idx : layer_indices)
                    {
                        const auto [_, inserted] = impl.endpoint_pools.emplace(
                            EndpointLayerKey{participant_id, layer_idx},
                            EndpointLayerPool{
                                .geometry_pool = destination,
                                .layer_arrival_capacity =
                                    config.shadow_slots_per_endpoint_layer,
                            });
                        if (!inserted)
                        {
                            throw std::logic_error(
                                "ExpertOverlay physical fabric created a duplicate endpoint/layer binding");
                        }
                        impl.endpoint_layer_pools.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }
        }

        /** @brief Produce a deterministic lane name without topology aliases. */
        std::string edgeLaneName(const EdgeKey &key)
        {
            return key.source.to_string() + "_to_" +
                   key.destination.to_string() + "_" +
                   projectionName(key.projection);
        }



        /** @brief Materialize every independently runnable local/remote GPU lane. */
        void materializeTransferLanes(
            const MoEOverlayPhysicalResidencyFabric::Config &config,
            MoEOverlayPhysicalResidencyFabric::Impl &impl)
        {
            const int world_rank=config.memory_authority->worldRank();
            const auto gpuEdgeUsesDirectPeer=[&](DeviceId source,DeviceId destination){
                if(!source.is_gpu() || !destination.is_gpu())return false;
                return admittedMoEOverlayDirectGpuPeer(impl.lane_demand_plan,source,destination);
            };
            const auto lane_counts=impl.lane_demand_plan.materializedCounts();
            const auto count=[&](MoEOverlayPhysicalLaneKey key){const auto found=lane_counts.find(key);
                return found==lane_counts.end()?size_t{0}:found->second;};
            const auto local_count=[&](DeviceId source,DeviceId destination,ExpertTierWeightProjection role){
                return count({.kind=MoEOverlayPhysicalLaneKind::LocalDevice,.world_rank=world_rank,
                    .source=source,.destination=destination,.projection=role});};
            const auto local_total=[&](DeviceId source,DeviceId destination){size_t total=0;
                for(const auto role:kProjections)total=checkedMoEOverlayLaneAdd(total,local_count(source,destination,role));return total;};
            const auto remote_count=[&](DeviceId device,RemoteGpuLaneRole direction,ExpertTierWeightProjection role){
                return count({.kind=direction==RemoteGpuLaneRole::Source?MoEOverlayPhysicalLaneKind::RemoteSource:MoEOverlayPhysicalLaneKind::RemoteDestination,
                    .world_rank=world_rank,.source=direction==RemoteGpuLaneRole::Source?device:DeviceId::invalid(),
                    .destination=direction==RemoteGpuLaneRole::Destination?device:DeviceId::invalid(),.projection=role});};
            const auto local_ids = config.registry->localParticipantIds();
            std::map<DeviceId, std::size_t> participant_multiplicity;
            std::map<GlobalDeviceAddress, std::size_t>
                cpu_address_multiplicity;
            for (const int participant_id : local_ids)
            {
                const auto endpoint = config.registry->endpoint(participant_id);
                const auto *participant =
                    config.initial_snapshot->placementOwnerMap().participantForId(
                        participant_id);
                if (!endpoint || !participant)
                    throw std::logic_error(
                        "ExpertOverlay lane materialization lost a local endpoint");
                ++participant_multiplicity[endpoint->device()];
                if (endpoint->device().is_cpu())
                    ++cpu_address_multiplicity[participant->address];
            }

            const auto parallelLaneCount = [&](std::size_t multiplicity)
            {
                if (multiplicity == 0 ||
                    config.maximum_concurrent_cycles == 0 ||
                    multiplicity >
                        std::numeric_limits<std::size_t>::max() /
                            config.maximum_concurrent_cycles)
                {
                    throw std::overflow_error(
                        "ExpertOverlay parallel lane BOM has zero or overflowing geometry");
                }
                return multiplicity * config.maximum_concurrent_cycles;
            };

            const auto& execution_stream_counts=impl.lane_demand_plan.original_execution_streams;

            /*
             * A complete placement target can take part once in a canonical
             * cycle, but its original members may require several real pieces.
             * Staging and command identities consume the shared piece plan.
             * Execution streams retain the declared physical stream policy;
             * multiplying pieces does not create additional hardware queues.
             */
            std::map<
                DeviceId,
                std::vector<PersistentTransferExecutionLane>>
                execution_lanes;
            for (const auto &[device, multiplicity] :
                 participant_multiplicity)
            {
                if (!device.is_gpu())
                    continue;
                execution_lanes.emplace(
                    device,
                    TransferEngine::instance()
                        .allocatePersistentTransferExecutionLanes(
                            execution_stream_counts.at(device),
                            device,
                            "moe_overlay_physical_fabric",
                            GPUDeviceContextPool::instance().retainExistingPreparedContext(device,config.memory_authority)));
            }
            const auto executionLane = [&execution_lanes](
                                           DeviceId device,
                                           std::size_t lane_index)
                -> const PersistentTransferExecutionLane &
            {
                const auto found = execution_lanes.find(device);
                if (found == execution_lanes.end() ||
                    found->second.empty())
                {
                    throw std::logic_error(
                        "ExpertOverlay physical lane exceeded its typed GPU execution-stream pool");
                }
                /* Logical participant/cycle lanes keep independent storage and
                 * terminal events. Only their background stream is shared, so
                 * every operation can be enqueued in one maintenance pass
                 * while finite GPU queues remain a separately tunable BOM. */
                return found->second[lane_index % found->second.size()];
            };
            std::size_t maximum_parallel_lanes = 0;

            /*
             * Derive retained command capacity from the exact lane BOM before
             * constructing any lane. A physical blob lane owns two pipeline
             * slots in each endpoint epoch for every projection and directed
             * edge. This is the same multiplicity/cycle calculation used below,
             * so setup cannot admit lanes that preflight forgot to budget.
             */
            std::map<DeviceId, std::size_t> progress_slot_demand;
            const auto addProgressSlots = [&progress_slot_demand](
                                              DeviceId device,
                                              std::size_t increment)
            {
                std::size_t &current = progress_slot_demand[device];
                if (increment >
                    std::numeric_limits<std::size_t>::max() - current)
                {
                    throw std::overflow_error(
                        "ExpertOverlay retained progress slot BOM overflowed");
                }
                current += increment;
            };
            for (const auto &[source_device, source_multiplicity] :
                 participant_multiplicity)
            {
                for (const auto &[destination_device,
                                  destination_multiplicity] :
                     participant_multiplicity)
                {
                    if(!local_total(source_device,destination_device))continue;
                    if (!source_device.is_gpu() ||
                        !destination_device.is_gpu() ||
                        (source_device == destination_device &&
                         source_multiplicity < 2) ||
                        gpuEdgeUsesDirectPeer(
                            source_device, destination_device))
                    {
                        continue;
                    }
                    const std::size_t projection_lanes=local_total(source_device,destination_device);
                    constexpr std::size_t slots_per_lane = 2u;
                    if (projection_lanes >
                        std::numeric_limits<std::size_t>::max() / slots_per_lane)
                    {
                        throw std::overflow_error(
                            "ExpertOverlay retained progress edge geometry overflowed");
                    }
                    const std::size_t edge_slots =
                        projection_lanes * slots_per_lane;
                    addProgressSlots(source_device, edge_slots);
                    addProgressSlots(destination_device, edge_slots);
                }
            }
            // Relay payload storage and service commands have different BOMs.
            // CPU/remote lanes already own mapped staging, but must reserve
            // their own command identities in the same CUDA progress authority.
            const auto mapped_relay_slot_demand = progress_slot_demand;
            for (const auto &[source, source_count] : participant_multiplicity)
            {
                for (const auto &[destination, destination_count] : participant_multiplicity)
                {
                    if (source.is_cpu() == destination.is_cpu())
                        continue;
                    const auto gpu = source.is_gpu() ? source : destination;
                    if (!gpu.is_cuda())
                        continue; // HIP uses the independently progressing SDMA mechanism.
                    const auto operations=local_total(source,destination);
                    if(operations>std::numeric_limits<size_t>::max()/2u)throw std::overflow_error("Local mapped lane demand overflow");
                    addProgressSlots(gpu,operations*2u);
                }
            }
            if (config.remote_projection_transport)
                for (const auto &[device, count] : participant_multiplicity)
                    if (device.is_cuda()) {
                        size_t operations=0;
                        for(const auto role:kProjections)for(const auto direction:{RemoteGpuLaneRole::Source,RemoteGpuLaneRole::Destination})
                            operations=checkedMoEOverlayLaneAdd(operations,remote_count(device,direction,role));
                        if(operations>std::numeric_limits<size_t>::max()/2u)throw std::overflow_error("Remote mapped lane demand overflow");
                        addProgressSlots(device,operations*2u);
                    }

            for (const auto &[device, slot_capacity] : progress_slot_demand)
            {
                /* The command directory scales with directed edges,
                 * projections, and double buffering. Physical GPU submission
                 * concurrency instead follows the configured migration-cycle
                 * budget. Commands beyond that bounded stream/event pool stay
                 * queued under their permanent slot identities. */
                const std::size_t execution_lane_capacity = std::min(slot_capacity,
                    std::max(execution_lanes.at(device).size(),
                        device.is_cuda()
                            ? parallelLaneCount(participant_multiplicity.at(device)) * kProjections.size() * 2u
                            : config.maximum_concurrent_cycles));
                auto epoch = MappedTransferProgressEpoch::create({
                    .device = device,
                    .slot_capacity = slot_capacity,
                    .execution_lane_capacity = execution_lane_capacity,
                    .execution_streams = execution_lanes.at(device),
                    .maximum_bytes = config.staging_capacity_bytes,
                    .name = "expert_overlay_physical_relay:" +
                            device.to_string(),
                    .perf_device = config.perf_device,
                });
                impl.transfer_progress_epochs.emplace(
                    device, std::move(epoch));
            }

            /*
             * The progress-slot BOM is also the exact mapped-payload BOM: one
             * source or destination region belongs to every retained command
             * slot. Allocate one native mapped slab per physical GPU, then
             * distribute immutable child regions as lanes are constructed.
             * This keeps setup proportional to devices rather than directed
             * edges, projections, cycle width, and double-buffer slots.
             */
            std::map<
                DeviceId,
                std::vector<std::shared_ptr<MappedHostTransferRegion>>>
                mapped_relay_staging;
            std::map<DeviceId, std::size_t> mapped_relay_staging_cursor;
            for (const auto &[device, slot_capacity] : mapped_relay_slot_demand)
            {
                mapped_relay_staging.emplace(
                    device,
                    TransferEngine::instance()
                        .allocateMappedHostTransferSlices(
                            config.staging_capacity_bytes,
                            slot_capacity,
                            device));
                mapped_relay_staging_cursor.emplace(device, 0u);
            }
            const auto takeMappedRelayStaging =
                [&](DeviceId device)
                -> std::shared_ptr<MappedHostTransferRegion>
            {
                const auto pool = mapped_relay_staging.find(device);
                auto cursor = mapped_relay_staging_cursor.find(device);
                if (pool == mapped_relay_staging.end() ||
                    cursor == mapped_relay_staging_cursor.end() ||
                    cursor->second >= pool->second.size())
                {
                    throw std::logic_error(
                        "ExpertOverlay mapped relay staging exceeded its exact progress-slot BOM for " +
                        device.toString());
                }
                return pool->second[cursor->second++];
            };

            for (const auto &[source_device, source_multiplicity] :
                 participant_multiplicity)
            {
                for (const auto &[destination_device,
                                  destination_multiplicity] :
                     participant_multiplicity)
                {
                    if (source_device.is_cpu() && destination_device.is_cpu())
                    {
                        continue;
                    }
                    /*
                     * A same-device lane is required even for one logical
                     * participant.  Ordinary migration never targets itself,
                     * but terminal reusable-context sealing must compact a
                     * returned expert from a shadow slot into a loader-era
                     * allocation without synchronizing the device.
                     */
                    std::size_t maximum_role_lanes=0;
                    for(const auto role:kProjections)maximum_role_lanes=std::max(maximum_role_lanes,local_count(source_device,destination_device,role));
                    maximum_parallel_lanes = std::max(
                        maximum_parallel_lanes, maximum_role_lanes);

                    for (const auto projection : kProjections)
                    {
                        const size_t lane_count=local_count(source_device,destination_device,projection);
                        if(!lane_count)continue;
                        const EdgeKey key{
                            .source = source_device,
                            .destination = destination_device,
                            .projection = projection,
                        };
                        const std::string name_prefix = edgeLaneName(key);
                        if (source_device.is_gpu() &&
                            destination_device.is_gpu())
                        {
                            const bool direct_peer = gpuEdgeUsesDirectPeer(
                                source_device,
                                destination_device);
                            if (direct_peer)
                            {
                                auto pool =
                                    std::make_shared<SharedPeerLanePool>();
                                for (std::size_t lane_index = 0;
                                     lane_index < lane_count;
                                     ++lane_index)
                                {
                                    auto lane = std::make_shared<
                                        ExpertTierGpuPeerTransferLane>(
                                        ExpertTierGpuPeerTransferLane::Config{
                                            .source_device = source_device,
                                            .destination_device =
                                                destination_device,
                                            .execution = executionLane(
                                                destination_device,
                                                lane_index),
                                            .lane_name =
                                                name_prefix + "_lane_" +
                                                std::to_string(lane_index),
                                            .perf_device = config.perf_device,
                                            .collect_timing_measurements =
                                                config.collect_economy_measurements,
                                        });
                                    std::string error;
                                    if (!lane->materialize(&error))
                                        throw std::runtime_error(error);
                                    pool->add(std::move(lane));
                                    impl.persistent_transfer_lanes.fetch_add(
                                        1, std::memory_order_relaxed);
                                    impl.direct_gpu_peer_lanes.fetch_add(
                                        1, std::memory_order_relaxed);
                                }
                                impl.peer_lanes.emplace(key, std::move(pool));
                            }
                            else
                            {
                                auto pool =
                                    std::make_shared<SharedBlobLanePool>();
                                for (std::size_t lane_index = 0;
                                     lane_index < lane_count;
                                     ++lane_index)
                                {
                                    std::array<
                                        std::shared_ptr<
                                            MappedHostTransferRegion>,
                                        2>
                                        source_staging;
                                    std::array<
                                        std::shared_ptr<
                                            MappedHostTransferRegion>,
                                        2>
                                        destination_staging;
                                    for (std::size_t slot_index = 0u;
                                         slot_index < source_staging.size();
                                         ++slot_index)
                                    {
                                        source_staging[slot_index] =
                                            takeMappedRelayStaging(
                                                source_device);
                                        destination_staging[slot_index] =
                                            takeMappedRelayStaging(
                                                destination_device);
                                    }
                                    auto lane = std::make_shared<
                                        ExpertTierGpuBlobTransferLane>(
                                        ExpertTierGpuBlobTransferLane::Config{
                                            .source_device = source_device,
                                            .destination_device =
                                                destination_device,
                                            .relay_kind =
                                                source_device.type ==
                                                        destination_device.type
                                                    ? ExpertTierGpuBlobRelayKind::
                                                          SameBackendWithoutPeerAccess
                                                    : ExpertTierGpuBlobRelayKind::
                                                          CrossBackend,
                                            .staging_capacity_bytes =
                                                config.staging_capacity_bytes,
                                            .source_mapped_staging =
                                                std::move(source_staging),
                                            .destination_mapped_staging =
                                                std::move(destination_staging),
                                            .source_progress_epoch =
                                                impl.transfer_progress_epochs.at(
                                                    source_device),
                                            .destination_progress_epoch =
                                                impl.transfer_progress_epochs.at(
                                                    destination_device),
                                            .lane_name =
                                                name_prefix + "_lane_" +
                                                std::to_string(lane_index),
                                            .perf_device = config.perf_device,
                                            .collect_timing_measurements =
                                                config.collect_economy_measurements,
                                        });
                                    std::string error;
                                    if (!lane->materialize(&error))
                                        throw std::runtime_error(error);
                                    pool->add(std::move(lane));
                                    impl.persistent_transfer_lanes.fetch_add(
                                        1, std::memory_order_relaxed);
                                    if (source_device.type ==
                                        destination_device.type)
                                    {
                                        impl.same_backend_no_peer_relay_lanes
                                            .fetch_add(
                                                1,
                                                std::memory_order_relaxed);
                                    }
                                    else
                                    {
                                        impl.cross_backend_gpu_relay_lanes
                                            .fetch_add(
                                                1,
                                                std::memory_order_relaxed);
                                    }
                                }
                                impl.blob_lanes.emplace(key, std::move(pool));
                            }
                        }
                        else
                        {
                            const DeviceId gpu = source_device.is_gpu()
                                                     ? source_device
                                                     : destination_device;
                            auto pool =
                                std::make_shared<SharedWeightLanePool>();
                            /*
                             * Preserve one exclusive slice per physical lane,
                             * while registering the complete pool as one host
                             * slab and allocating it as one device slab. ROCm
                             * host registration has substantial fixed and
                             * growing per-allocation cost; exposing every 4 MiB
                             * slice as a separate allocation made setup scale
                             * quadratically with the public wave width.
                             */
                            const auto staging_slices =
                                TransferEngine::instance()
                                    .allocatePersistentTransferStagingSlices(
                                        config.staging_capacity_bytes,
                                        lane_count,
                                        gpu);
                            for (std::size_t lane_index = 0;
                                 lane_index < lane_count;
                                 ++lane_index)
                            {
                                auto lane = std::make_shared<
                                    ExpertTierWeightTransferLane>(
                                    ExpertTierWeightTransferLane::Config{
                                        .device = gpu,
                                        .staging =
                                            staging_slices[lane_index],
                                        .execution = executionLane(
                                            gpu, lane_index),
                                        .progress = gpu.is_cuda()
                                            ? BackgroundTransferProgressBinding::graphService(
                                                impl.transfer_progress_epochs.at(gpu))
                                            : BackgroundTransferProgressBinding::nativeStream(),
                                        .lane_name =
                                            name_prefix + "_lane_" +
                                            std::to_string(lane_index),
                                        .perf_device = config.perf_device,
                                        .collect_timing_measurements =
                                            config.collect_economy_measurements,
                                    });
                                std::string error;
                                if (!lane->materialize(&error))
                                    throw std::runtime_error(error);
                                pool->add(std::move(lane));
                                impl.persistent_transfer_lanes.fetch_add(
                                    1, std::memory_order_relaxed);
                            }
                            impl.weight_lanes.emplace(key, std::move(pool));
                        }
                    }
                }
            }

            for (const auto &[device, slices] : mapped_relay_staging)
            {
                const auto cursor =
                    mapped_relay_staging_cursor.find(device);
                if (cursor == mapped_relay_staging_cursor.end() ||
                    cursor->second != slices.size())
                {
                    throw std::logic_error(
                        "ExpertOverlay mapped relay staging did not consume its exact progress-slot BOM for " +
                        device.toString());
                }
            }

            /*
             * DeviceId intentionally collapses every CPU NUMA participant to
             * cpu:0. CPU movement therefore keys workers by the complete
             * topology address. Each lane owns one persistent destination-
             * NUMA worker, and a same-address pool is needed only when two
             * logical participants share that physical CPU endpoint.
             */
            std::size_t maximum_parallel_cpu_lanes = 0;
            for (const auto &[source_address, source_multiplicity] :
                 cpu_address_multiplicity)
            {
                for (const auto &[destination_address,
                                  destination_multiplicity] :
                     cpu_address_multiplicity)
                {
                    /*
                     * Preserve one same-address worker family for terminal
                     * context compaction.  It is setup-owned and remains idle
                     * during inference unless two logical participants share
                     * this address and an ordinary cycle legitimately uses it.
                     */
                    size_t maximum_role_lanes=0;
                    const auto cpu_count=[&](ExpertTierWeightProjection role){return count({
                        .kind=MoEOverlayPhysicalLaneKind::LocalCPUWorker,.world_rank=world_rank,
                        .source=source_address.toLocalDeviceId(),.destination=destination_address.toLocalDeviceId(),
                        .source_address=source_address,.destination_address=destination_address,.projection=role});};
                    for(const auto role:kProjections)maximum_role_lanes=std::max(maximum_role_lanes,cpu_count(role));
                    maximum_parallel_cpu_lanes = std::max(
                        maximum_parallel_cpu_lanes, maximum_role_lanes);
                    maximum_parallel_lanes = std::max(
                        maximum_parallel_lanes, maximum_role_lanes);

                    for (const auto projection : kProjections)
                    {
                        const size_t lane_count=cpu_count(projection);
                        if(!lane_count)continue;
                        const CpuAddressEdgeKey key{
                            .source = source_address,
                            .destination = destination_address,
                            .projection = projection,
                        };
                        auto pool =
                            std::make_shared<SharedCpuCopyLanePool>();
                        const std::string name_prefix =
                            source_address.toString() + "_to_" +
                            destination_address.toString() + "_" +
                            projectionName(projection);
                        for (std::size_t lane_index = 0;
                             lane_index < lane_count;
                             ++lane_index)
                        {
                            auto lane = std::make_shared<CpuCopyLane>(
                                CpuCopyLane::Config{
                                    .destination_numa_node =
                                        destination_address.hasValidNuma()
                                            ? destination_address.numa_node
                                            : -1,
                                    .chunk_bytes =
                                        config.staging_capacity_bytes,
                                    .lane_name =
                                        name_prefix + "_lane_" +
                                        std::to_string(lane_index),
                                    .perf_device = config.perf_device,
                                    .concurrency =
                                        impl.cpu_copy_concurrency,
                                });
                            std::string error;
                            if (!lane->materialize(&error))
                            {
                                throw std::runtime_error(
                                    error.empty()
                                        ? "Could not materialize an ExpertOverlay CPU copy worker"
                                        : std::move(error));
                            }
                            pool->add(std::move(lane));
                            impl.persistent_transfer_lanes.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                        impl.cpu_copy_lanes.emplace(key, std::move(pool));
                    }
                }
            }
            impl.maximum_parallel_cpu_copy_lanes.store(
                maximum_parallel_cpu_lanes, std::memory_order_relaxed);

            impl.maximum_parallel_edge_lanes.store(
                maximum_parallel_lanes, std::memory_order_relaxed);
            if (!config.remote_projection_transport)
                return;

            /*
             * Actual cross-rank fanout and the existing retained role ports
             * are separate requirements in the same immutable setup plan.
             * Every independently writable piece gets a real lane; neither a
             * cohort's absent scalar owner nor storage capacity is its extent.
             */
            for (const auto &[device, multiplicity] : participant_multiplicity)
            {
                if (!device.is_gpu())
                    continue;
                for (const auto projection : kProjections)
                {
                    for (const auto role : {
                             RemoteGpuLaneRole::Source,
                             RemoteGpuLaneRole::Destination})
                    {
                        const size_t lane_count=remote_count(device,role,projection);
                        if(!lane_count)continue;
                        maximum_parallel_lanes=std::max(maximum_parallel_lanes,lane_count);
                        const RemoteGpuLaneKey key{
                            .device = device,
                            .projection = projection,
                            .role = role,
                        };
                        const std::string role_name =
                            role == RemoteGpuLaneRole::Source
                                ? "remote_source"
                                : "remote_destination";
                        auto &lanes = impl.remote_gpu_lanes[key];
                        lanes.reserve(lane_count);
                        /* Cross-rank endpoints retain the same independently
                         * writable lane geometry but share two pool-wide
                         * TransferEngine allocation authorities. */
                        const auto staging_slices =
                            TransferEngine::instance()
                                .allocatePersistentTransferStagingSlices(
                                    config.staging_capacity_bytes,
                                    lane_count,
                                    device,
                                    HostTransferAccess::CrossProcessReadable);
                        for (std::size_t lane_index = 0;
                             lane_index < lane_count;
                             ++lane_index)
                        {
                            auto lane = std::make_shared<
                                MoEOverlayGpuRemoteProjectionLane>(
                                MoEOverlayGpuRemoteProjectionLane::Config{
                                    .device = device,
                                    .staging = staging_slices[lane_index],
                                    .execution = executionLane(
                                        device, lane_index),
                                    .progress = device.is_cuda()
                                        ? BackgroundTransferProgressBinding::graphService(
                                            impl.transfer_progress_epochs.at(device))
                                        : BackgroundTransferProgressBinding::nativeStream(),
                                    .lane_name =
                                        device.to_string() + "_" + role_name +
                                        "_" + projectionName(projection) +
                                        "_lane_" +
                                        std::to_string(lane_index),
                                    .perf_device = config.perf_device,
                                });
                            std::string error;
                            if (!lane->materialize(&error))
                                throw std::runtime_error(error);
                            lanes.push_back(std::move(lane));
                            impl.persistent_transfer_lanes.fetch_add(
                                1, std::memory_order_relaxed);
                        }
                    }
                }
            }
            impl.maximum_parallel_edge_lanes.store(
                maximum_parallel_lanes, std::memory_order_relaxed);
        }

        /** @brief Find a destination pool or fail with one exact identity. */
        EndpointLayerPool &requireEndpointPool(
            MoEOverlayPhysicalResidencyFabric::Impl &impl,
            int participant_id,
            int layer_idx)
        {
            const auto found = impl.endpoint_pools.find(
                EndpointLayerKey{participant_id, layer_idx});
            if (found == impl.endpoint_pools.end())
            {
                throw std::runtime_error(
                    "ExpertOverlay migration targets an endpoint/layer with no preallocated shadow pool");
            }
            return found->second;
        }

        /** @brief Acquire one CPU or GPU slot without starting physical work. */
        ReservedDestination reserveDestination(
            EndpointLayerPool &binding,
            int layer_idx,
            int expert_id,
            std::uint64_t candidate_epoch)
        {
            if (!binding.geometry_pool)
            {
                throw std::logic_error(
                    "ExpertOverlay endpoint/layer binding lost its geometry arena");
            }
            auto &pool = *binding.geometry_pool;
            ReservedDestination reserved;
            if (pool.device.is_cpu())
            {
                if (pool.adopted_slots)
                {
                    reserved.cpu = pool.adopted_slots->acquireCpu(
                        layer_idx, expert_id, candidate_epoch);
                    if (reserved.cpu)
                        return reserved;
                }
                auto cpu_pool =
                    std::get<std::shared_ptr<CpuExpertSlotPool>>(pool.pool);
                reserved.cpu = cpu_pool->acquireForLayer(
                    layer_idx, expert_id, candidate_epoch);
                if (!reserved.cpu)
                    throw std::runtime_error(
                        "ExpertOverlay CPU shadow-slot preflight changed before reservation");
            }
            else
            {
                if (pool.adopted_slots)
                {
                    reserved.gpu = pool.adopted_slots->acquireGpu(
                        layer_idx, expert_id, candidate_epoch);
                    if (reserved.gpu)
                        return reserved;
                }
                auto gpu_pool =
                    std::get<std::shared_ptr<GpuExpertSlotPool>>(pool.pool);
                reserved.gpu = gpu_pool->acquireForLayer(
                    layer_idx, expert_id, candidate_epoch);
                if (!reserved.gpu)
                    throw std::runtime_error(
                        "ExpertOverlay GPU shadow-slot preflight changed before reservation");
            }
            return reserved;
        }

        /** @brief Return currently available slots through the typed pool. */
        std::size_t availableSlots(const EndpointLayerPool &pool)
        {
            if (!pool.geometry_pool)
                return 0;
            const auto &geometry = *pool.geometry_pool;
            const std::size_t adopted = geometry.adopted_slots
                                            ? geometry.adopted_slots
                                                  ->availableSlots()
                                            : 0;
            if (geometry.device.is_cpu())
            {
                return adopted +
                    std::get<std::shared_ptr<CpuExpertSlotPool>>(geometry.pool)
                        ->availableSlots();
            }
            return adopted +
                std::get<std::shared_ptr<GpuExpertSlotPool>>(geometry.pool)
                    ->availableSlots();
        }
    } // namespace

} // namespace llaminar2
