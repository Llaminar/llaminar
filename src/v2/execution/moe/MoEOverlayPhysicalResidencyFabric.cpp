/**
 * @file MoEOverlayPhysicalResidencyFabric.cpp
 * @brief Local slot reservation and parallel physical movement for ExpertOverlay.
 *
 * One maintenance worker polls every operation in a wave. Each directed
 * device edge owns an admission-sized persistent lane pool for gate, up, and
 * down. Every independent operation reserves its own lane before the first
 * poll, so the worker submits the complete wave without software queueing or
 * runtime stream/staging allocation.
 */

#include "MoEOverlayPhysicalResidencyFabric.h"

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
#include "loaders/ModelLoader.h"
#include "backends/BackendManager.h"
#include "backends/ComputeBackend.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
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
#include <pthread.h>
#include <sched.h>
#include <span>
#include <stdexcept>
#include <tuple>
#include <thread>
#include <utility>
#include <variant>

namespace llaminar2
{
    bool MoEOverlayProjectionWeightManifest::valid() const noexcept
    {
        const auto role = static_cast<std::size_t>(projection);
        return role < 3 && N > 0 && K > 0 && format.valid() &&
               (format.isFloating() || (K % 32) == 0);
    }

    bool MoEOverlayLayerWeightManifest::valid() const noexcept
    {
        if (layer_idx < 0)
            return false;
        std::array<bool, 3> seen{};
        for (const auto &projection : projections)
        {
            const auto role = static_cast<std::size_t>(projection.projection);
            if (!projection.valid() || role >= seen.size() || seen[role])
                return false;
            seen[role] = true;
        }
        return std::all_of(
            seen.begin(), seen.end(), [](bool value) { return value; });
    }

    bool MoEOverlayReusableContextSeal::valid() const noexcept
    {
        if (source_epoch == 0 || canonical_owner_map.owners().empty() ||
            canonical_owner_map.participants().empty() || local_banks.empty())
            return false;

        try
        {
            for (std::size_t bank_index = 0;
                 bank_index < local_banks.size();
                 ++bank_index)
            {
                const auto &bank = local_banks[bank_index];
                const auto *participant =
                    canonical_owner_map.participantForId(bank.participant_id);
                if (!participant || bank.epoch != source_epoch ||
                    bank.device != participant->device || bank.layers.empty() ||
                    bank.layers.front().resident_mask.empty() ||
                    !bank.valid(
                        bank.participant_id,
                        participant->device,
                        static_cast<int>(bank.layers.size()),
                        static_cast<int>(
                            bank.layers.front().resident_mask.size())))
                {
                    return false;
                }
                for (std::size_t previous = 0;
                     previous < bank_index;
                     ++previous)
                {
                    if (local_banks[previous].participant_id ==
                        bank.participant_id)
                    {
                        return false;
                    }
                }

                for (std::size_t layer = 0;
                     layer < bank.layers.size();
                     ++layer)
                {
                    const auto &resident_mask =
                        bank.layers[layer].resident_mask;
                    for (std::size_t expert = 0;
                         expert < resident_mask.size();
                         ++expert)
                    {
                        const auto *owner = canonical_owner_map.ownerFor(
                            static_cast<int>(layer),
                            static_cast<int>(expert));
                        if (!owner ||
                            resident_mask[expert] !=
                                (owner->owner_participant ==
                                 bank.participant_id))
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        catch (...)
        {
            /* Validation is a failure predicate at an noexcept lifecycle edge. */
            return false;
        }
    }

    std::vector<MoEOverlayLayerWeightManifest>
    buildMoEOverlayLayerWeightManifestFromGGUF(
        const GGUFModel &model,
        int num_layers,
        int num_experts)
    {
        if (num_layers <= 0 || num_experts <= 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay GGUF weight manifest requires positive model geometry");
        }

        const auto source_format = [](GGUFTensorType type)
            -> ExpertWeightFormat
        {
            const auto native = [](const NativeVnniFormatInfo &format)
            {
                return ExpertWeightFormat::nativeVnni({
                    .codebook_id = format.codebook_id,
                    .is_superblock = format.is_superblock,
                    .present = true,
                });
            };
            switch (type)
            {
            case GGUFTensorType::Q4_0:
                return native(native_vnni_formats::Q4_0);
            case GGUFTensorType::Q4_1:
                return native(native_vnni_formats::Q4_1);
            case GGUFTensorType::Q5_0:
                return native(native_vnni_formats::Q5_0);
            case GGUFTensorType::Q5_1:
                return native(native_vnni_formats::Q5_1);
            case GGUFTensorType::Q8_0:
                return native(native_vnni_formats::Q8_0);
            case GGUFTensorType::Q2_K:
                return native(native_vnni_formats::Q2_K);
            case GGUFTensorType::Q3_K:
                return native(native_vnni_formats::Q3_K);
            case GGUFTensorType::Q4_K:
                return native(native_vnni_formats::Q4_K);
            case GGUFTensorType::Q5_K:
                return native(native_vnni_formats::Q5_K);
            case GGUFTensorType::Q6_K:
                return native(native_vnni_formats::Q6_K);
            case GGUFTensorType::Q8_K:
                return native(native_vnni_formats::Q8_K);
            case GGUFTensorType::IQ2_XXS:
                return native(native_vnni_formats::IQ2_XXS);
            case GGUFTensorType::IQ2_XS:
                return native(native_vnni_formats::IQ2_XS);
            case GGUFTensorType::IQ3_XXS:
                return native(native_vnni_formats::IQ3_XXS);
            case GGUFTensorType::IQ1_S:
                return native(native_vnni_formats::IQ1_S);
            case GGUFTensorType::IQ4_NL:
                return native(native_vnni_formats::IQ4_NL);
            case GGUFTensorType::IQ3_S:
                return native(native_vnni_formats::IQ3_S);
            case GGUFTensorType::IQ2_S:
                return native(native_vnni_formats::IQ2_S);
            case GGUFTensorType::IQ4_XS:
                return native(native_vnni_formats::IQ4_XS);
            case GGUFTensorType::IQ1_M:
                return native(native_vnni_formats::IQ1_M);
            case GGUFTensorType::F32:
                return ExpertWeightFormat::floating(TensorType::FP32);
            case GGUFTensorType::F16:
                return ExpertWeightFormat::floating(TensorType::FP16);
            case GGUFTensorType::BF16:
                return ExpertWeightFormat::floating(TensorType::BF16);
            }
            return {};
        };

        struct ProjectionTensor
        {
            ExpertTierWeightProjection projection;
            const char *suffix;
        };
        constexpr std::array<ProjectionTensor, 3> projections{{
            {ExpertTierWeightProjection::Gate, "ffn_gate_exps.weight"},
            {ExpertTierWeightProjection::Up, "ffn_up_exps.weight"},
            {ExpertTierWeightProjection::Down, "ffn_down_exps.weight"},
        }};

        std::vector<MoEOverlayLayerWeightManifest> result;
        result.reserve(static_cast<std::size_t>(num_layers));
        for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
        {
            MoEOverlayLayerWeightManifest layer;
            layer.layer_idx = layer_idx;
            for (std::size_t role = 0; role < projections.size(); ++role)
            {
                const std::string name =
                    "blk." + std::to_string(layer_idx) + "." +
                    projections[role].suffix;
                const GGUFTensorInfo *tensor = model.findTensor(name);
                const ExpertWeightFormat format =
                    tensor ? source_format(tensor->type)
                           : ExpertWeightFormat{};
                if (!tensor || tensor->dimensions.size() != 3 ||
                    tensor->dimensions[0] == 0 ||
                    tensor->dimensions[1] == 0 ||
                    tensor->dimensions[2] !=
                        static_cast<std::uint64_t>(num_experts) ||
                    (!format.isFloating() &&
                     tensor->dimensions[0] % 32u != 0) ||
                    !format.valid() ||
                    tensor->dimensions[0] >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
                    tensor->dimensions[1] >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay GGUF tensor '" + name +
                        "' has absent, unsupported, or incompatible metadata");
                }

                /*
                 * ModelLoader canonicalizes GGUF dimensions to
                 * [K, N, experts]. Every destination engine consumes one
                 * expert as an ordinary [N, K] projection.
                 */
                layer.projections[role] = {
                    .projection = projections[role].projection,
                    .N = static_cast<int>(tensor->dimensions[1]),
                    .K = static_cast<int>(tensor->dimensions[0]),
                    .format = format,
                };
            }
            if (!layer.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay GGUF metadata produced an invalid layer manifest");
            }
            result.push_back(std::move(layer));
        }
        return result;
    }

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

        /** @brief Select one retained projection from a complete expert triplet. */
        std::shared_ptr<ITensorGemm> projectionEngine(
            const MoEOverlayPreparedExpertTriplet &triplet,
            ExpertTierWeightProjection projection)
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return triplet.gate;
            case ExpertTierWeightProjection::Up:
                return triplet.up;
            case ExpertTierWeightProjection::Down:
                return triplet.down;
            }
            throw std::invalid_argument(
                "ExpertOverlay physical fabric received an unknown projection role");
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
                std::string *error = nullptr) noexcept
            {
                std::lock_guard<std::mutex> lock(mutex_);
                const State state = state_.load(std::memory_order_acquire);
                if (!initialization_ok_ || stopping_ || source.empty() ||
                    source.size() != destination.size() || !gate ||
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
                    bool cancelled = !released;
                    try
                    {
                        while (!cancelled && offset < job->source.size())
                        {
                            cancelled = cancel_requested_.load(
                                std::memory_order_acquire);
                            if (cancelled)
                                break;
                            const std::size_t bytes = std::min(
                                config_.chunk_bytes,
                                job->source.size() - offset);
                            const auto copy_started_at =
                                std::chrono::steady_clock::now();
                            std::memcpy(
                                job->destination.data() + offset,
                                job->source.data() + offset,
                                bytes);
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
                                static_cast<double>(job->source.size()),
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
                                        job->source.size()),
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
         * @brief Destination endpoint that publishes one exact prepared engine.
         *
         * Final bytes remain owned and validated by the wrapped device-specific
         * endpoint. Publication occurs only after that endpoint reports complete
         * and accepts its own finalization edge, so the candidate RCU bank can
         * never observe a partially received projection.
         */
        class PublishingRemoteProjectionDestination final
            : public IMoEOverlayRemoteProjectionDestinationEndpoint
        {
        public:
            /** @brief Retain storage, arrival authority, role, and engine alias. */
            PublishingRemoteProjectionDestination(
                std::shared_ptr<
                    IMoEOverlayRemoteProjectionDestinationEndpoint> storage,
                std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
                ExpertTierWeightProjection projection,
                std::shared_ptr<ITensorGemm> engine)
                : storage_(std::move(storage)),
                  arrival_(std::move(arrival)),
                  projection_(projection),
                  engine_(std::move(engine))
            {
                if (!storage_ || !arrival_ || !engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay remote publication endpoint requires complete ownership");
                }
            }

            /**
             * @brief Retain a deferred engine provider for manifest-shaped GPU arrivals.
             *
             * A remote GPU sender's live physical format is known only after
             * manifest authentication. The wrapped GPU endpoint creates the
             * exact engine alias then; this provider retrieves it at the final
             * publication edge without guessing a canonical representation.
             */
            PublishingRemoteProjectionDestination(
                std::shared_ptr<
                    IMoEOverlayRemoteProjectionDestinationEndpoint> storage,
                std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
                ExpertTierWeightProjection projection,
                std::function<std::shared_ptr<ITensorGemm>()> engine_provider)
                : storage_(std::move(storage)),
                  arrival_(std::move(arrival)),
                  projection_(projection),
                  engine_provider_(std::move(engine_provider))
            {
                if (!storage_ || !arrival_ || !engine_provider_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay deferred remote publication endpoint requires complete ownership");
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

            /** @brief Finalize storage, then publish the engine exactly once. */
            bool publishFinal(std::string *error) noexcept override
            {
                if (published_)
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
                if (!arrival_->publish(projection_, engine_, error))
                    return false;
                published_ = true;
                return true;
            }

            /** @brief Poison an unpublished arrival and abort exact storage work. */
            void abort() noexcept override
            {
                if (!published_)
                {
                    (void)arrival_->fail(
                        "ExpertOverlay remote projection was aborted before publication");
                }
                storage_->abort();
            }

            /** @brief Delegate event-aware storage reclamation after abort. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                return storage_->pollAbort(error);
            }

        private:
            std::shared_ptr<
                IMoEOverlayRemoteProjectionDestinationEndpoint> storage_;
            std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival_;
            ExpertTierWeightProjection projection_ =
                ExpertTierWeightProjection::Gate;
            std::shared_ptr<ITensorGemm> engine_;
            std::function<std::shared_ptr<ITensorGemm>()> engine_provider_;
            bool published_ = false;
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : direction_(direction),
                  layout_(layout),
                  gpu_source_(gpu_source),
                  cpu_destination_(cpu_destination),
                  cpu_source_(cpu_source),
                  gpu_destination_(gpu_destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : direction_(direction),
                  cpu_destination_(cpu_destination),
                  cpu_source_(cpu_source),
                  gpu_floating_(gpu_floating),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
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
                    cpu_bytes != gpu_floating_.bytes || !source_engine_ ||
                    !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay admitted floating GPU/CPU operation has incomplete ownership");
                }
                lane_ = lane_pool->tryAcquire(this);
                if (!lane_)
                {
                    throw std::runtime_error(
                        "ExpertOverlay admitted floating GPU/CPU wave exhausted its pre-materialized parallel lane pool");
                }
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
                            source_readiness_, &start_error);
                        break;
                    case Direction::CpuToGpuRepacked:
                        started = lane_->lane()->startCpuToGpu(
                            layout_, cpu_source_, gpu_destination_,
                            &start_error);
                        break;
                    case Direction::GpuToCpuContiguous:
                        started = lane_->lane()->startGpuToCpuContiguous(
                            gpu_floating_, cpu_destination_,
                            source_readiness_, &start_error);
                        break;
                    case Direction::CpuToGpuContiguous:
                        started = lane_->lane()->startCpuToGpuContiguous(
                            cpu_source_, gpu_floating_, &start_error);
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

        private:
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : source_(source),
                  destination_(destination),
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : floating_source_(source),
                  floating_destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_(true)
            {
                if (!lane_pool || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != floating_destination_.n ||
                    floating_source_.k != floating_destination_.k ||
                    floating_source_.bytes != floating_destination_.bytes ||
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
                        ? lane_->lane()->startContiguous(
                              floating_source_.data,
                              const_cast<void *>(
                                  floating_destination_.data),
                              floating_source_.bytes,
                              source_readiness_,
                              &start_error)
                        : lane_->lane()->start(
                              source_,
                              destination_,
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

        private:
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : source_(source),
                  destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : floating_source_(source),
                  floating_destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_(true)
            {
                if (!lane_pool || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != floating_destination_.n ||
                    floating_source_.k != floating_destination_.k ||
                    floating_source_.bytes != floating_destination_.bytes ||
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
                        ? lane_->lane()->startContiguous(
                              floating_source_.data,
                              const_cast<void *>(
                                  floating_destination_.data),
                              floating_source_.bytes,
                              source_readiness_,
                              &start_error)
                        : lane_->lane()->start(
                              source_,
                              destination_,
                              source_readiness_,
                              &start_error);
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

        private:
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
                std::shared_ptr<ITensorGemm> destination_engine)
                : source_(source),
                  destination_(destination),
                  wave_gate_(std::move(wave_gate)),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
            {
                if (!lane_pool || source_.empty() ||
                    source_.size() != destination_.size() || !wave_gate_ ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU copy requires exact final storage and ownership");
                }
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
                            &start_error))
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

        private:
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
             * @param initial_slots Loader-owned resident triplets.
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
                    lease.projections.push_back({
                        .projection = projection.projection,
                        .destination_bytes = projection.cpu_destination,
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
                    "ExpertOverlay staging capacity cannot hold one CPU NativeVNNI unit");
            }
            return static_cast<std::uint32_t>(std::min<std::size_t>(
                layout.unit_count,
                staging_capacity_bytes / layout.cpu_block_stride));
        }

        /** @brief Require identical final CPU representations before NUMA copy. */
        void requireCompatibleCpuWeights(
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &source,
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &destination)
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
            NativeVnniSourceIdentity source_identity)
        {
            if (projection.spec.N != N || projection.spec.K != K ||
                projection.spec.payload_bytes_per_block < payload_bytes ||
                (is_asymmetric && !projection.spec.is_asymmetric) ||
                (has_emins && !projection.spec.has_emins))
            {
                throw std::runtime_error(
                    "ExpertOverlay GPU shadow slot cannot represent an arriving projection");
            }

            DeviceNativeVNNIMatrixDesc descriptor;
            descriptor.payload = projection.slot.d_native_vnni_payload;
            descriptor.scales = projection.slot.d_native_vnni_scales;
            descriptor.mins = is_asymmetric
                                  ? projection.slot.d_native_vnni_mins
                                  : nullptr;
            descriptor.emins = has_emins
                                   ? projection.slot.d_native_vnni_emins
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
            NativeVnniSourceIdentity source_identity)
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
                    NativeVnniReusableDeviceAllocationFormat{
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
                    NativeVnniReusableDeviceAllocationFormat{
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
                !projection.slot.d_native_vnni_payload ||
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
                .data = projection.slot.d_native_vnni_payload,
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

        /** @brief Compare source geometry/provenance independent of execution layout. */
        ProjectionSignature signatureFor(
            const MoEOverlayPreparedWeightSource &source)
        {
            ProjectionSignature signature;
            signature.format = source.format;
            if (source.kind ==
                    MoEOverlayPreparedWeightSourceKind::CpuNativeVnni)
            {
                signature.N = source.cpu_packed->N;
                signature.K = source.cpu_packed->K;
            }
            else if (source.kind ==
                     MoEOverlayPreparedWeightSourceKind::
                         GpuSeparatedNativeVnni)
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
            for (std::size_t expert = 0; expert < layer.experts.size(); ++expert)
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
                            projectionEngine(layer.experts[expert], projection),
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
                                    ->exportRetiredCPUNativeVNNIStorage();
                            if (!source.cpu_packed ||
                                adopted.cpu_destination.empty() ||
                                adopted.cpu_destination.data() !=
                                    source.cpu_packed->native_interleaved.data() ||
                                adopted.cpu_destination.size() !=
                                    source.cpu_packed->native_interleaved.size())
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay initial CPU NativeVNNI engine cannot expose its exact recyclable final storage");
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
                                .d_native_vnni_payload =
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
                            native_vnni_formats::forSourceIdentity(
                                source.format.native_vnni.codebook_id,
                                source.format.native_vnni.is_superblock);
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
                                std::to_string(source.format.native_vnni.codebook_id) +
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
                            .d_native_vnni_payload =
                                source.gpu_packed.ptrs.d_vnni,
                            .d_native_vnni_scales =
                                source.gpu_packed.ptrs.d_scales,
                            .d_native_vnni_mins =
                                source.gpu_packed.ptrs.d_mins,
                            .d_native_vnni_emins =
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
            const cpu::native_vnni::CPUNativeVNNIPackedWeights &packed)
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
            ExpertTierWeightProjection projection)
        {
            const auto &migration = migrations[migration_index];
            return {
                .expected_epoch = expected_epoch,
                .candidate_epoch = candidate_epoch,
                .execution_fingerprint = execution_fingerprint,
                .migration_index = migration_index,
                .layer_idx = migration.layer_idx,
                .expert_id = migration.expert_id,
                .projection = projection,
                .source_participant =
                    migration.source.owner_participant,
                .destination_participant =
                    migration.destination.owner_participant,
                .source_world_rank = migration.source.owner_world_rank,
                .destination_world_rank =
                    migration.destination.owner_world_rank,
                .source_device = migration.source.device,
                .destination_device = migration.destination.device,
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

        /** @brief Resolve the exact triplet contract for one model layer. */
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

        /** @brief Materialize every local endpoint/layer from one model manifest. */
        void materializeEndpointPools(
            const MoEOverlayPhysicalResidencyFabric::Config &config,
            MoEOverlayPhysicalResidencyFabric::Impl &impl)
        {
            const auto local_ids = config.registry->localParticipantIds();
            std::map<int, LayerProjectionSignatures> model_signatures;
            std::map<
                int,
                MoEOverlayParticipantBankLease>
                initial_banks;
            std::vector<MoEOverlayDeviceInitialPhysicalSlot>
                initial_device_slots;

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
                    config.initial_snapshot->owner_map.participantForId(
                        participant_id);
                const auto bank = endpoint
                                      ? endpoint->acquire(
                                            config.initial_snapshot->epoch)
                                      : MoEOverlayParticipantBankLease{};
                if (!endpoint || !participant || !bank ||
                    bank->device != endpoint->device())
                {
                    throw std::runtime_error(
                        "ExpertOverlay physical fabric requires every local initial bank");
                }
                initial_banks.emplace(participant_id, bank);

                for (int layer_idx = 0;
                     layer_idx < endpoint->numLayers();
                     ++layer_idx)
                {
                    const auto &layer =
                        bank->layers[static_cast<std::size_t>(layer_idx)];
                    for (std::size_t expert = 0;
                         expert < layer.experts.size();
                         ++expert)
                    {
                        if (!layer.resident_mask[expert])
                            continue;
                        initial_device_slots.push_back({
                            .key = {
                                .participant_id = participant_id,
                                .layer_idx = layer_idx,
                                .expert_id = static_cast<int>(expert),
                            },
                            .entered_epoch =
                                config.initial_snapshot->epoch,
                            .bootstrap_allocation = true,
                            .triplet = layer.experts[expert],
                        });
                        for (const auto projection : kProjections)
                        {
                            MoEOverlayPreparedWeightSource source;
                            std::string source_error;
                            if (!resolveMoEOverlayPreparedWeightSource(
                                    projectionEngine(
                                        layer.experts[expert], projection),
                                    endpoint->device(),
                                    source,
                                    &source_error))
                            {
                                throw std::runtime_error(
                                    source_error.empty()
                                        ? "ExpertOverlay initial engine cannot source migration"
                                        : source_error);
                            }
                            mergeProjectionSignature(
                                model_signatures,
                                layer_idx,
                                projection,
                                signatureFor(source));
                        }
                    }
                }
            }

            /*
             * From this point onward the device path resolves sources through
             * shared prepared-engine lifetimes, not through the setup registry.
             * The registry remains available solely to the distinct host-RCU
             * authority path.
             */
            impl.device_slot_ledger =
                std::make_unique<MoEOverlayDevicePhysicalSlotLedger>(
                    MoEOverlayDevicePhysicalSlotLedger::Config{
                        .initial_epoch = config.initial_snapshot->epoch,
                        .local_participant_ids = local_ids,
                        .initial_slots = std::move(initial_device_slots),
                    });

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
                    config.initial_snapshot->owner_map.participantForId(
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
                for (int layer_idx = 0; layer_idx < endpoint->numLayers();
                     ++layer_idx)
                {
                    const auto signatures = requireLayerSignatures(
                        model_signatures, layer_idx);
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
                            bank_found->second->layers[
                                static_cast<std::size_t>(layer_idx)];
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
                            if (signature.format.isNativeVnni())
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
                                    canonicalDeviceVnniCodebookId(
                                        signature.format.native_vnni
                                            .codebook_id);
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

        /**
         * @brief Decide whether one exact GPU edge may use direct device access.
         *
         * Same-device copies are ordinary D2D operations.  A different-device
         * edge is direct only when the destination (which owns the transfer
         * stream) can address the source allocation.  Missing topology is a
         * setup error rather than permission to discover an implicit runtime
         * fallback while inference is live.
         */
        bool gpuEdgeUsesDirectPeer(DeviceId source, DeviceId destination)
        {
            if (!source.is_gpu() || !destination.is_gpu() ||
                source.type != destination.type)
            {
                return false;
            }
            if (source == destination)
                return true;
            const auto access = DeviceManager::instance().peerAccessAvailable(
                destination,
                source);
            if (!access.has_value())
            {
                throw std::runtime_error(
                    "ExpertOverlay could not resolve directed peer access for " +
                    destination.to_string() + " -> " + source.to_string());
            }
            return *access;
        }

        /** @brief Materialize every independently runnable local/remote GPU lane. */
        void materializeTransferLanes(
            const MoEOverlayPhysicalResidencyFabric::Config &config,
            MoEOverlayPhysicalResidencyFabric::Impl &impl)
        {
            const auto local_ids = config.registry->localParticipantIds();
            std::map<DeviceId, std::size_t> participant_multiplicity;
            std::map<GlobalDeviceAddress, std::size_t>
                cpu_address_multiplicity;
            for (const int participant_id : local_ids)
            {
                const auto endpoint = config.registry->endpoint(participant_id);
                const auto *participant =
                    config.initial_snapshot->owner_map.participantForId(
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

            const auto physicalStreamCount = [&](std::size_t multiplicity)
            {
                if (multiplicity == 0u ||
                    config.maximum_execution_streams == 0u ||
                    multiplicity >
                        std::numeric_limits<std::size_t>::max() /
                            config.maximum_execution_streams)
                {
                    throw std::overflow_error(
                        "ExpertOverlay physical stream-pool geometry overflowed");
                }
                return multiplicity * config.maximum_execution_streams;
            };

            /*
             * One physical participant can take part once in every admitted
             * cycle. Gate/up/down projections and source/destination roles are
             * separate logical operations, but multiplying HIP/CUDA streams by
             * those roles does not create additional hardware queues. Bind all
             * compatible operations to the same typed participant/cycle lane;
             * their staging and completion events remain independent.
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
                            physicalStreamCount(multiplicity),
                            device,
                            "moe_overlay_physical_fabric"));
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
                    if (!source_device.is_gpu() ||
                        !destination_device.is_gpu() ||
                        (source_device == destination_device &&
                         source_multiplicity < 2) ||
                        gpuEdgeUsesDirectPeer(
                            source_device, destination_device))
                    {
                        continue;
                    }
                    const std::size_t lane_count = parallelLaneCount(
                        source_device == destination_device
                            ? source_multiplicity
                            : std::min(
                                  source_multiplicity,
                                  destination_multiplicity));
                    constexpr std::size_t slots_per_lane = 2u;
                    if (lane_count >
                        std::numeric_limits<std::size_t>::max() /
                            slots_per_lane /
                            kProjections.size())
                    {
                        throw std::overflow_error(
                            "ExpertOverlay retained progress edge geometry overflowed");
                    }
                    const std::size_t edge_slots =
                        lane_count * slots_per_lane * kProjections.size();
                    addProgressSlots(source_device, edge_slots);
                    addProgressSlots(destination_device, edge_slots);
                }
            }
            for (const auto &[device, slot_capacity] : progress_slot_demand)
            {
                /* The command directory scales with directed edges,
                 * projections, and double buffering. Physical GPU submission
                 * concurrency instead follows the configured migration-cycle
                 * budget. Commands beyond that bounded stream/event pool stay
                 * queued under their permanent slot identities. */
                const std::size_t execution_lane_capacity = std::min(
                    slot_capacity, config.maximum_concurrent_cycles);
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
            for (const auto &[device, slot_capacity] : progress_slot_demand)
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
                    const std::size_t lane_count = parallelLaneCount(
                        source_device == destination_device
                            ? source_multiplicity
                            : std::min(
                                  source_multiplicity,
                                  destination_multiplicity));
                    maximum_parallel_lanes = std::max(
                        maximum_parallel_lanes, lane_count);

                    for (const auto projection : kProjections)
                    {
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
                    const std::size_t lane_count = parallelLaneCount(
                        source_address == destination_address
                            ? source_multiplicity
                            : std::min(
                                  source_multiplicity,
                                  destination_multiplicity));
                    maximum_parallel_cpu_lanes = std::max(
                        maximum_parallel_cpu_lanes, lane_count);
                    maximum_parallel_lanes = std::max(
                        maximum_parallel_lanes, lane_count);

                    for (const auto projection : kProjections)
                    {
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
             * Cross-rank edges are not known until histogram-driven placement
             * chooses a transaction. Each logical participant on a local GPU
             * can appear once per closed cycle, so every role owns that exact
             * multiplicity times the concurrent-cycle cap.
             */
            for (const auto &[device, multiplicity] : participant_multiplicity)
            {
                if (!device.is_gpu())
                    continue;
                const std::size_t lane_count = parallelLaneCount(multiplicity);
                maximum_parallel_lanes = std::max(
                    maximum_parallel_lanes, lane_count);
                for (const auto projection : kProjections)
                {
                    for (const auto role : {
                             RemoteGpuLaneRole::Source,
                             RemoteGpuLaneRole::Destination})
                    {
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
                                    device);
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

    std::shared_ptr<MoEOverlayPhysicalResidencyFabric>
    MoEOverlayPhysicalResidencyFabric::create(Config config)
    {
        if (!config.memory_authority || !config.registry ||
            !config.initial_snapshot ||
            !config.initial_snapshot->valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires the rank-bound memory authority, registry, and valid initial snapshot");
        }
        if (config.registry->initialEpoch() !=
                config.initial_snapshot->epoch ||
            !config.registry->allInitialBanksReady())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires complete matching initial banks");
        }
        if (config.shadow_slots_per_endpoint_layer == 0 ||
            config.staging_capacity_bytes == 0 ||
            config.maximum_concurrent_cycles == 0 ||
            config.maximum_execution_streams == 0 ||
            config.maximum_execution_streams >
                config.maximum_concurrent_cycles)
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires positive shadow, staging, cycle, and execution-stream capacities, with streams no greater than cycles");
        }
        if (config.remote_projection_transport)
        {
            if (config.staging_capacity_bytes >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::invalid_argument(
                    "ExpertOverlay remote staging capacity exceeds the canonical 32-bit chunk ABI");
            }
            const int local_world_rank =
                config.remote_projection_transport->worldRank();
            for (const int participant_id :
                 config.registry->localParticipantIds())
            {
                const auto *participant =
                    config.initial_snapshot->owner_map.participantForId(
                        participant_id);
                if (!participant || !participant->world_rank_known ||
                    participant->world_rank != local_world_rank)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay distributed physical fabric registry contains a participant assigned to another or unknown rank");
                }
            }
        }
        if (config.perf_device.empty())
            config.perf_device = "expert_overlay";

        auto impl = std::make_unique<Impl>();
        materializeEndpointPools(config, *impl);
        materializeTransferLanes(config, *impl);

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "physical_fabrics_materialized",
            1.0,
            "model_setup",
            config.perf_device,
            {{"endpoint_layer_pools",
              std::to_string(impl->endpoint_pools.size())},
             {"endpoint_geometry_pools",
              std::to_string(
                  impl->endpoint_geometry_pools.load(
                      std::memory_order_relaxed))},
             {"adopted_initial_slots",
              std::to_string(
                  impl->adopted_initial_slots.load(
                      std::memory_order_relaxed))},
             {"persistent_lanes",
              std::to_string(
                  impl->persistent_transfer_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_parallel_edge_lanes",
              std::to_string(
                  impl->maximum_parallel_edge_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_parallel_cpu_copy_lanes",
              std::to_string(
                  impl->maximum_parallel_cpu_copy_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_concurrent_cycles",
              std::to_string(config.maximum_concurrent_cycles)},
             {"maximum_execution_streams",
              std::to_string(config.maximum_execution_streams)}});
        return std::shared_ptr<MoEOverlayPhysicalResidencyFabric>(
            new MoEOverlayPhysicalResidencyFabric(
                std::move(config), std::move(impl)));
    }

    MoEOverlayPhysicalResidencyFabric::MoEOverlayPhysicalResidencyFabric(
        Config config,
        std::unique_ptr<Impl> impl) noexcept
        : config_(std::move(config)), impl_(std::move(impl))
    {
    }

    MoEOverlayPhysicalResidencyFabric::~MoEOverlayPhysicalResidencyFabric() =
        default;

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::prepareTransfers(
        const MoEOverlayResidencyTransaction &transaction,
        const std::vector<int> &local_destination_participants)
    {
        if (!transaction.valid() || transaction.empty() ||
            !transaction.candidate)
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric requires a valid non-empty transaction",
            };
        }
        return preparePhysicalTransfers(
            transaction.purpose,
            transaction.expected_epoch,
            transaction.candidate->epoch,
            fingerprintMoEOverlayResidencyExecutionPlan(transaction),
            transaction.migrations,
            transaction.migration_cycles,
            transaction.shadow_requirements,
            nullptr,
            local_destination_participants);
    }

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::prepareDeviceTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const std::vector<int> &local_destination_participants)
    {
        if (!batch.valid() || !batch.movesWeights() ||
            (batch.kind !=
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
             batch.kind != MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore))
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric requires non-empty device-authored durable movement",
            };
        }
        if (!impl_->device_slot_ledger)
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric lost its device slot ledger",
            };
        }
        std::string ledger_error;
        if (!impl_->device_slot_ledger->begin(batch, &ledger_error))
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error = std::move(ledger_error),
            };
        }

        auto prepared = preparePhysicalTransfers(
            MoEOverlayResidencyTransactionPurpose::PlacementChange,
            batch.base_epoch,
            batch.candidate_epoch,
            batch.execution_fingerprint,
            batch.migrations,
            batch.migration_cycles,
            batch.shadow_requirements,
            &batch,
            local_destination_participants);
        if (prepared.status != MoEOverlayResidencyStageStartStatus::Started)
        {
            std::string abort_error;
            if (!impl_->device_slot_ledger->abort(batch, &abort_error) &&
                prepared.error.empty())
            {
                prepared.error = std::move(abort_error);
                prepared.status = MoEOverlayResidencyStageStartStatus::Failed;
            }
        }
        return prepared;
    }

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::preparePhysicalTransfers(
        MoEOverlayResidencyTransactionPurpose purpose,
        std::uint64_t expected_epoch,
        std::uint64_t candidate_epoch,
        const MoEOverlayResidencyExecutionFingerprint &execution_fingerprint,
        const std::vector<MoEOverlayTierMigration> &migrations,
        const std::vector<MoEOverlayTierMigrationCycle> &migration_cycles,
        const std::vector<MoEOverlayTierShadowRequirement> &shadow_requirements,
        const MoEOverlayDevicePhysicalMovementBatch *device_batch,
        const std::vector<int> &local_destination_participants)
    {
        MoEOverlayParticipantPreparedTransfers result;
        auto fail = [&](std::string message)
            -> MoEOverlayParticipantPreparedTransfers
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            result.status = MoEOverlayResidencyStageStartStatus::Failed;
            result.error = std::move(message);
            return std::move(result);
        };

        {
            std::lock_guard<std::mutex> seal_lock(
                impl_->reusable_seal_mutex);
            if (impl_->reusable_seal_state !=
                Impl::ReusableSealState::Open)
            {
                return fail(
                    "ExpertOverlay physical movement is forbidden after reusable-context sealing begins");
            }
        }

        if (migrations.empty() || expected_epoch == 0u ||
            candidate_epoch == 0u || !execution_fingerprint.valid())
            return fail(
                "ExpertOverlay physical fabric received incomplete physical movement identity");

        const auto configured_local = config_.registry->localParticipantIds();
        if (configured_local != local_destination_participants)
            return fail(
                "ExpertOverlay physical fabric received a different local participant set");

        const auto is_local_participant = [&](int participant_id)
        {
            return std::binary_search(
                configured_local.begin(),
                configured_local.end(),
                participant_id);
        };
        const auto remote_transport = config_.remote_projection_transport;
        const int world_rank = remote_transport
                                   ? remote_transport->worldRank()
                                   : -1;
        const int world_size = remote_transport
                                   ? remote_transport->worldSize()
                                   : 0;
        bool has_remote_migrations = false;

        /*
         * A distributed fabric requires exact participant-to-rank membership.
         * This prevents a moved GPU or CPU domain from being inferred from an
         * ordinal: topology intent decides ownership, and the registry must
         * materialize every participant assigned to this rank and no others.
         */
        for (const auto &migration : migrations)
        {
            const bool source_local = is_local_participant(
                migration.source.owner_participant);
            const bool destination_local = is_local_participant(
                migration.destination.owner_participant);
            if (!remote_transport)
            {
                if (!source_local || !destination_local)
                {
                    return fail(
                        "ExpertOverlay transaction contains a remote endpoint but no remote projection transport is installed");
                }
                continue;
            }

            if (!migration.source.owner_world_rank_known ||
                !migration.destination.owner_world_rank_known ||
                migration.source.owner_world_rank < 0 ||
                migration.source.owner_world_rank >= world_size ||
                migration.destination.owner_world_rank < 0 ||
                migration.destination.owner_world_rank >= world_size)
            {
                return fail(
                    "ExpertOverlay distributed movement requires exact in-range world-rank ownership");
            }
            if (source_local !=
                    (migration.source.owner_world_rank == world_rank) ||
                destination_local !=
                    (migration.destination.owner_world_rank == world_rank))
            {
                return fail(
                    "ExpertOverlay local participant registry disagrees with distributed rank ownership");
            }
            has_remote_migrations = has_remote_migrations ||
                migration.source.owner_world_rank !=
                    migration.destination.owner_world_rank;
        }

        if (migration_cycles.size() >
            config_.maximum_concurrent_cycles)
        {
            return fail(
                "ExpertOverlay transaction exceeds the physical fabric's concurrent-cycle lane BOM");
        }

        /*
         * Prove the complete software-dispatch fan-out before reserving a
         * destination slot. A Started wave is therefore guaranteed to own one
         * distinct physical lane for every non-CPU projection; no operation is
         * admitted on the promise that an earlier operation will finish first.
         */
        std::map<EdgeKey, std::size_t> local_lane_demand;
        std::map<CpuAddressEdgeKey, std::size_t> cpu_copy_lane_demand;
        std::map<RemoteGpuLaneKey, std::size_t> remote_gpu_lane_demand;
        for (const auto &migration : migrations)
        {
            const bool source_local = is_local_participant(
                migration.source.owner_participant);
            const bool destination_local = is_local_participant(
                migration.destination.owner_participant);
            for (const auto projection : kProjections)
            {
                if (source_local && destination_local &&
                    migration.source.device.is_cpu() &&
                    migration.destination.device.is_cpu())
                {
                    ++cpu_copy_lane_demand[CpuAddressEdgeKey{
                        .source = migration.source.address,
                        .destination = migration.destination.address,
                        .projection = projection,
                    }];
                }
                else if (source_local && destination_local)
                {
                    ++local_lane_demand[EdgeKey{
                        .source = migration.source.device,
                        .destination = migration.destination.device,
                        .projection = projection,
                    }];
                }
                if (has_remote_migrations && source_local &&
                    !destination_local && migration.source.device.is_gpu())
                {
                    ++remote_gpu_lane_demand[RemoteGpuLaneKey{
                        .device = migration.source.device,
                        .projection = projection,
                        .role = RemoteGpuLaneRole::Source,
                    }];
                }
                if (has_remote_migrations && !source_local &&
                    destination_local &&
                    migration.destination.device.is_gpu())
                {
                    ++remote_gpu_lane_demand[RemoteGpuLaneKey{
                        .device = migration.destination.device,
                        .projection = projection,
                        .role = RemoteGpuLaneRole::Destination,
                    }];
                }
            }
        }
        for (const auto &[edge, demand] : local_lane_demand)
        {
            std::size_t capacity = 0;
            if (const auto peer_pool = impl_->peer_lanes.find(edge);
                peer_pool != impl_->peer_lanes.end())
            {
                capacity = peer_pool->second->capacity();
            }
            else if (const auto blob_pool = impl_->blob_lanes.find(edge);
                     blob_pool != impl_->blob_lanes.end())
            {
                capacity = blob_pool->second->capacity();
            }
            else
            {
                const auto weight_pool = impl_->weight_lanes.find(edge);
                if (weight_pool != impl_->weight_lanes.end())
                    capacity = weight_pool->second->capacity();
            }
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized local parallel lane pool");
            }
        }
        std::size_t local_cpu_copy_operation_count = 0;
        for (const auto &[edge, demand] : cpu_copy_lane_demand)
        {
            const auto pool = impl_->cpu_copy_lanes.find(edge);
            const std::size_t capacity =
                pool == impl_->cpu_copy_lanes.end()
                    ? 0
                    : pool->second->capacity();
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized CPU parallel worker pool");
            }
            if (demand >
                std::numeric_limits<std::size_t>::max() -
                    local_cpu_copy_operation_count)
            {
                return fail(
                    "ExpertOverlay CPU parallel worker demand overflows size_t");
            }
            local_cpu_copy_operation_count += demand;
        }
        for (const auto &[key, demand] : remote_gpu_lane_demand)
        {
            const auto pool = impl_->remote_gpu_lanes.find(key);
            const std::size_t capacity = pool == impl_->remote_gpu_lanes.end()
                ? 0
                : pool->second.size();
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized remote GPU parallel lane pool");
            }
        }

        /*
         * Validate the immutable BOM and transient availability for the whole
         * cycle set before acquiring one slot.  Exceeding the model-time BOM is
         * fatal; an old ticket retaining a planned slot is ordinary deferral.
         */
        struct GeometryShadowDemand
        {
            EndpointLayerPool *binding = nullptr;
            std::size_t slot_count = 0;
            int destination_participant = -1;
        };
        std::map<const EndpointGeometryPool *, GeometryShadowDemand>
            demand_by_geometry;
        for (const auto &requirement : shadow_requirements)
        {
            if (!std::binary_search(
                    configured_local.begin(),
                    configured_local.end(),
                    requirement.destination_participant))
            {
                continue;
            }
            EndpointLayerPool *pool = nullptr;
            try
            {
                pool = &requireEndpointPool(
                    *impl_,
                    requirement.destination_participant,
                    requirement.layer_idx);
            }
            catch (const std::exception &error)
            {
                return fail(error.what());
            }
            if (!pool->geometry_pool ||
                requirement.slot_count > pool->layer_arrival_capacity)
            {
                return fail(
                    "ExpertOverlay migration wave exceeds the per-layer shadow arrival bound: participant=" +
                    std::to_string(requirement.destination_participant) +
                    " layer=" + std::to_string(requirement.layer_idx) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " capacity=" +
                    std::to_string(pool->layer_arrival_capacity));
            }
            auto &geometry_demand = demand_by_geometry[
                pool->geometry_pool.get()];
            if (!geometry_demand.binding)
            {
                geometry_demand.binding = pool;
                geometry_demand.destination_participant =
                    requirement.destination_participant;
            }
            if (requirement.slot_count >
                std::numeric_limits<std::size_t>::max() -
                    geometry_demand.slot_count)
            {
                return fail(
                    "ExpertOverlay geometry shadow demand overflows size_t");
            }
            geometry_demand.slot_count += requirement.slot_count;
        }

        for (const auto &[_, requirement] : demand_by_geometry)
        {
            if (!requirement.binding ||
                !requirement.binding->geometry_pool)
            {
                return fail(
                    "ExpertOverlay geometry shadow demand lost its typed arena binding");
            }
            const auto &geometry = *requirement.binding->geometry_pool;
            if (requirement.slot_count > geometry.shadow_capacity)
            {
                return fail(
                    "ExpertOverlay migration wave exceeds the shared exact-geometry shadow-slot BOM: participant=" +
                    std::to_string(requirement.destination_participant) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " capacity=" +
                    std::to_string(geometry.shadow_capacity));
            }
            const std::size_t available = availableSlots(
                *requirement.binding);
            if (requirement.slot_count > available)
            {
                impl_->waves_deferred.fetch_add(
                    1, std::memory_order_relaxed);
                result.status = MoEOverlayResidencyStageStartStatus::Deferred;
                result.error =
                    "ExpertOverlay shadow slots are retained by an older epoch: "
                    "participant=" +
                    std::to_string(requirement.destination_participant) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " available=" + std::to_string(available) +
                    " planned_capacity=" +
                    std::to_string(geometry.shadow_capacity) +
                    " expected_epoch=" +
                    std::to_string(expected_epoch) +
                    " candidate_epoch=" +
                    std::to_string(candidate_epoch);
                return result;
            }
        }

        std::vector<ReservedDestination> destinations(
            migrations.size());
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    migrations[migration_index];
                if (!is_local_participant(
                        migration.destination.owner_participant))
                {
                    continue;
                }
                auto &pool = requireEndpointPool(
                    *impl_,
                    migration.destination.owner_participant,
                    migration.layer_idx);
                destinations[migration_index] = reserveDestination(
                    pool,
                    migration.layer_idx,
                    migration.expert_id,
                    candidate_epoch);
            }
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        result.migrations.resize(migrations.size());
        struct RemoteOperationTarget
        {
            std::size_t migration_index = 0;
            std::size_t projection_index = 0;
        };
        std::vector<MoEOverlayMPIRemoteProjectionBinding> remote_bindings;
        std::vector<RemoteOperationTarget> remote_targets;
        if (has_remote_migrations)
        {
            const auto remote_projection_count =
                migrations.size() * kProjections.size();
            remote_bindings.reserve(remote_projection_count);
            remote_targets.reserve(remote_projection_count);
        }

        const auto remote_execution_fingerprint = remote_transport
            ? std::optional<MoEOverlayResidencyExecutionFingerprint>(
                  execution_fingerprint)
            : std::nullopt;
        const auto cpu_copy_wave_gate = local_cpu_copy_operation_count == 0
            ? std::shared_ptr<CpuCopyWaveGate>{}
            : std::make_shared<CpuCopyWaveGate>(
                  local_cpu_copy_operation_count);
        std::uint64_t local_cpu_copy_operations = 0;
        std::uint64_t local_remote_cpu_operations = 0;
        std::uint64_t local_remote_gpu_cpu_operations = 0;
        std::uint64_t local_remote_gpu_blob_operations = 0;
        std::uint64_t local_gpu_cpu_operations = 0;
        std::uint64_t local_same_backend_gpu_operations = 0;
        std::uint64_t local_heterogeneous_gpu_operations = 0;
        std::map<RemoteGpuLaneKey, std::size_t> remote_gpu_lane_cursor;
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    migrations[migration_index];
                auto &prepared = result.migrations[migration_index];
                const bool source_local = is_local_participant(
                    migration.source.owner_participant);
                const bool destination_local = is_local_participant(
                    migration.destination.owner_participant);
                const bool crosses_world_rank = remote_transport &&
                    migration.source.owner_world_rank !=
                        migration.destination.owner_world_rank;

                std::shared_ptr<MoEOverlayParticipantResidency> source_endpoint;
                MoEOverlayParticipantBankLease source_bank;
                std::optional<MoEOverlayPreparedExpertTriplet>
                    device_source_triplet;
                const MoEOverlayPreparedExpertTriplet *source_triplet = nullptr;
                if (source_local)
                {
                    if (device_batch)
                    {
                        std::string source_error;
                        device_source_triplet =
                            impl_->device_slot_ledger->sourceTriplet(
                                *device_batch,
                                {
                                    .participant_id =
                                        migration.source.owner_participant,
                                    .layer_idx = migration.layer_idx,
                                    .expert_id = migration.expert_id,
                                },
                                &source_error);
                        if (!device_source_triplet)
                        {
                            throw std::runtime_error(
                                source_error.empty()
                                    ? "ExpertOverlay device slot ledger cannot resolve the local source"
                                    : std::move(source_error));
                        }
                        source_triplet = &*device_source_triplet;
                    }
                    else
                    {
                        source_endpoint = config_.registry->endpoint(
                            migration.source.owner_participant);
                        source_bank = source_endpoint
                                          ? source_endpoint->acquire(
                                                expected_epoch)
                                          : MoEOverlayParticipantBankLease{};
                        if (!source_endpoint || !source_bank ||
                            migration.layer_idx < 0 ||
                            migration.layer_idx >=
                                source_endpoint->numLayers() ||
                            migration.expert_id < 0 ||
                            migration.expert_id >=
                                source_endpoint->numExperts())
                        {
                            throw std::runtime_error(
                                "ExpertOverlay physical fabric cannot acquire the exact local source epoch");
                        }
                        source_triplet = &source_bank
                                              ->layers[static_cast<std::size_t>(
                                                  migration.layer_idx)]
                                              .experts[static_cast<std::size_t>(
                                                  migration.expert_id)];
                    }
                    if (!source_triplet->complete())
                    {
                        throw std::runtime_error(
                            "ExpertOverlay migration source is not resident in its old epoch bank");
                    }
                }

                if (destination_local)
                {
                    prepared.destination_arrival =
                        std::make_shared<MoEOverlayPreparedExpertArrival>();
                }

                /* A same-rank edge owned elsewhere still occupies global shape. */
                if (!crosses_world_rank && !source_local)
                {
                    for (const auto projection : kProjections)
                    {
                        prepared.projections[projectionIndex(projection)] =
                            std::make_unique<
                                ReadyPhysicalProjectionOperation>();
                    }
                    continue;
                }

                for (const auto projection : kProjections)
                {
                    const auto projection_index = projectionIndex(projection);

                    if (crosses_world_rank)
                    {
                        if (!remote_execution_fingerprint)
                            throw std::logic_error(
                                "ExpertOverlay remote projection lost transaction fingerprint ownership");

                        const auto identity = remoteProjectionIdentity(
                            expected_epoch,
                            candidate_epoch,
                            *remote_execution_fingerprint,
                            migrations,
                            migration_index,
                            projection);
                        MoEOverlayMPIRemoteProjectionBinding binding{
                            .lane_index =
                                migration_index * kProjections.size() +
                                projection_index,
                            .identity = identity,
                            .purpose = purpose,
                        };

                        if (source_local)
                        {
                            auto source_engine = projectionEngine(
                                *source_triplet, projection);
                            MoEOverlayPreparedWeightSource source;
                            std::string source_error;
                            if (!resolveMoEOverlayPreparedWeightSource(
                                    source_engine,
                                    migration.source.device,
                                    source,
                                    &source_error))
                            {
                                throw std::runtime_error(
                                    source_error.empty()
                                        ? "ExpertOverlay remote source does not expose prepared weights"
                                        : std::move(source_error));
                            }

                            if (migration.source.device.is_cpu())
                            {
                                if (source.format.isFloating())
                                {
                                    const auto manifest =
                                        makeMoEOverlayRemoteFloatingProjectionManifest(
                                            identity,
                                            source.floating,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteContiguousSourceRegions(
                                            std::span<const std::uint8_t>(
                                                static_cast<const std::uint8_t *>(
                                                    source.floating.data),
                                                source.floating.bytes)),
                                        source_engine);
                                    if (migration.destination.device.is_cpu())
                                        ++local_remote_cpu_operations;
                                    else
                                        ++local_remote_gpu_cpu_operations;
                                }
                                else if (!source.cpu_packed)
                                {
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU source lost final NativeVNNI storage");
                                }
                                else if (migration.destination.device.is_cpu())
                                {
                                    const auto manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            *source.cpu_packed,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteCpuSourceRegions(
                                            *source.cpu_packed),
                                        source_engine);
                                    ++local_remote_cpu_operations;
                                }
                                else
                                {
                                    const auto probe =
                                        makeCpuToGpuExpertTierWeightStreamManifest(
                                            *source.cpu_packed,
                                            candidate_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeCpuToGpuExpertTierWeightStreamManifest(
                                            *source.cpu_packed,
                                            candidate_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units);
                                    const auto manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteCpuSourceRegions(
                                            *source.cpu_packed),
                                        source_engine);
                                    ++local_remote_gpu_cpu_operations;
                                }
                            }
                            else
                            {
                                const RemoteGpuLaneKey lane_key{
                                    .device = migration.source.device,
                                    .projection = projection,
                                    .role = RemoteGpuLaneRole::Source,
                                };
                                const auto lane =
                                    impl_->remote_gpu_lanes.find(lane_key);
                                if (lane == impl_->remote_gpu_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay remote GPU source has no pre-materialized lane");
                                const std::size_t lane_index =
                                    remote_gpu_lane_cursor[lane_key]++;
                                if (lane_index >= lane->second.size())
                                    throw std::logic_error(
                                        "ExpertOverlay remote GPU source exceeded its preflighted parallel lane pool");

                                MoEOverlayRemoteProjectionManifest manifest;
                                if (source.format.isFloating())
                                {
                                    manifest =
                                        makeMoEOverlayRemoteFloatingProjectionManifest(
                                            identity,
                                            source.floating,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    if (migration.destination.device.is_cpu())
                                        ++local_remote_gpu_cpu_operations;
                                    else
                                        ++local_remote_gpu_blob_operations;
                                }
                                else if (migration.destination.device.is_cpu())
                                {
                                    const auto *source_format =
                                        native_vnni_formats::forSourceIdentity(
                                            source.format.native_vnni
                                                .codebook_id,
                                            source.format.native_vnni
                                                .is_superblock);
                                    if (!source_format)
                                        throw std::runtime_error(
                                            "ExpertOverlay remote GPU source lost catalogued provenance");
                                    const auto probe =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            source.gpu_packed,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            source.gpu_packed,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units);
                                    manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    ++local_remote_gpu_cpu_operations;
                                }
                                else
                                {
                                    manifest =
                                        makeMoEOverlayRemoteGpuProjectionManifest(
                                            identity,
                                            source.gpu_packed,
                                            source.format.native_vnni,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    ++local_remote_gpu_blob_operations;
                                }
                                const auto readiness =
                                    ExpertTierSourceReadiness::
                                        publishedResidencyBank(
                                            expected_epoch);
                                if (source.format.isFloating())
                                {
                                    binding.source = std::make_shared<
                                        MoEOverlayGpuRemoteProjectionSource>(
                                        std::move(manifest),
                                        lane->second[lane_index],
                                        source.floating,
                                        readiness,
                                        source_engine);
                                }
                                else
                                {
                                    binding.source = std::make_shared<
                                        MoEOverlayGpuRemoteProjectionSource>(
                                        std::move(manifest),
                                        lane->second[lane_index],
                                        source.gpu_packed,
                                        readiness,
                                        source_engine);
                                }
                            }
                        }

                        if (destination_local)
                        {
                            if (migration.destination.device.is_cpu())
                            {
                                auto &lease =
                                    *destinations[migration_index].cpu;
                                const auto &destination = cpuProjection(
                                    lease, projection);
                                auto destination_engine = destination.engine;
                                ContiguousFloatingPointWeightDescriptor
                                    destination_floating;
                                const bool destination_is_floating =
                                    destination_engine
                                        ->exportContiguousFloatingPointWeights(
                                            destination_floating);
                                const auto *destination_packed =
                                    destination_engine
                                        ->exportCPUNativeVNNIPackedWeights();
                                if (destination_is_floating ==
                                    (destination_packed != nullptr))
                                {
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU destination must expose exactly one physical representation");
                                }

                                MoEOverlayRemoteProjectionManifest
                                    expected_manifest;
                                std::array<
                                    std::span<std::uint8_t>,
                                    kMoEOverlayRemoteProjectionRegionCount>
                                    destination_regions;
                                auto match_policy =
                                    MoEOverlayRemoteProjectionManifestMatchPolicy::
                                        Exact;
                                if (destination_is_floating)
                                {
                                    if (!destination_floating.valid() ||
                                        destination.destination_bytes.data() !=
                                            destination_floating.data ||
                                        destination.destination_bytes.size() !=
                                            destination_floating.bytes)
                                    {
                                        throw std::runtime_error(
                                            "ExpertOverlay remote floating CPU destination disagrees with its final slot");
                                    }
                                    expected_manifest =
                                        makeMoEOverlayRemoteFloatingProjectionManifest(
                                            identity,
                                            destination_floating,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    destination_regions =
                                        remoteContiguousDestinationRegions(
                                            destination.destination_bytes);
                                    if (migration.source.device.is_cpu())
                                        ++local_remote_cpu_operations;
                                    else
                                        ++local_remote_gpu_cpu_operations;
                                }
                                else if (migration.source.device.is_cpu())
                                {
                                    expected_manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            *destination_packed,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    destination_regions =
                                        remoteCpuDestinationRegions(
                                            destination.destination_bytes);
                                    ++local_remote_cpu_operations;
                                }
                                else
                                {
                                    const auto *source_format =
                                        native_vnni_formats::forSourceIdentity(
                                            destination_packed->codebook_id,
                                            destination_packed->is_superblock);
                                    if (!source_format)
                                        throw std::runtime_error(
                                            "ExpertOverlay remote CPU destination lost source provenance");
                                    const auto probe =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            destination_packed->N,
                                            destination_packed->K,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            destination_packed->N,
                                            destination_packed->K,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units);
                                    expected_manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    match_policy =
                                        MoEOverlayRemoteProjectionManifestMatchPolicy::
                                            EquivalentGpuSourceForFinalCpuStorage;
                                    destination_regions =
                                        remoteCpuDestinationRegions(
                                            destination.destination_bytes);
                                    ++local_remote_gpu_cpu_operations;
                                }

                                auto storage = std::make_shared<
                                    MoEOverlayHostRemoteProjectionDestination>(
                                    identity,
                                    destination_regions,
                                    destination_engine,
                                    expected_manifest,
                                    match_policy);
                                binding.destination = std::make_shared<
                                    PublishingRemoteProjectionDestination>(
                                    std::move(storage),
                                    prepared.destination_arrival,
                                    projection,
                                    std::move(destination_engine));
                            }
                            else
                            {
                                auto &lease =
                                    *destinations[migration_index].gpu;
                                const auto destination_slot = gpuProjection(
                                    lease, projection);
                                const auto slot_lifetime = lease.lifetime;
                                const auto destination_format =
                                    destination_slot.spec.format;
                                const auto destination_device =
                                    migration.destination.device;
                                const RemoteGpuLaneKey lane_key{
                                    .device = destination_device,
                                    .projection = projection,
                                    .role = RemoteGpuLaneRole::Destination,
                                };
                                const auto lane =
                                    impl_->remote_gpu_lanes.find(lane_key);
                                if (lane == impl_->remote_gpu_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay remote GPU destination has no pre-materialized lane");
                                const std::size_t lane_index =
                                    remote_gpu_lane_cursor[lane_key]++;
                                if (lane_index >= lane->second.size())
                                    throw std::logic_error(
                                        "ExpertOverlay remote GPU destination exceeded its preflighted parallel lane pool");

                                MoEOverlayGpuRemoteProjectionDestinationFactory
                                    factory =
                                        [destination_slot,
                                         destination_device,
                                         slot_lifetime,
                                         destination_format](
                                            const MoEOverlayRemoteProjectionManifest &
                                                manifest,
                                            MoEOverlayGpuRemoteProjectionDestinationBinding *
                                                output,
                                            std::string *error) -> bool
                                {
                                    if (destination_format.isFloating())
                                    {
                                        if (!output ||
                                            !manifest
                                                 .carriesFloatingBytes() ||
                                            manifest.N !=
                                                destination_slot.spec.N ||
                                            manifest.K !=
                                                destination_slot.spec.K ||
                                            manifest.format_kind !=
                                                destination_format.kind)
                                        {
                                            if (error)
                                                *error =
                                                    "Remote floating ExpertOverlay GPU arrival differs from the model projection contract";
                                            return false;
                                        }
                                        auto descriptor =
                                            makeGpuFloatingDestinationDescriptor(
                                                destination_slot,
                                                destination_format);
                                        auto engine =
                                            makeGpuFloatingDestinationEngine(
                                                destination_device,
                                                descriptor,
                                                slot_lifetime);
                                        *output = {
                                            .floating_descriptor = descriptor,
                                            .engine = std::move(engine),
                                        };
                                        if (error)
                                            error->clear();
                                        return true;
                                    }

                                    const auto source_identity =
                                        destination_format.native_vnni;
                                    if (!output ||
                                        (!manifest.carriesGpuBytes() &&
                                         !manifest.carriesCpuBytes()) ||
                                        manifest.N !=
                                            destination_slot.spec.N ||
                                        manifest.K !=
                                            destination_slot.spec.K ||
                                        manifest.source_codebook_id !=
                                            source_identity.codebook_id ||
                                        (manifest.source_is_superblock != 0) !=
                                            source_identity.is_superblock)
                                    {
                                        if (error)
                                            *error =
                                                "Remote ExpertOverlay GPU arrival differs from the model projection contract";
                                        return false;
                                    }
                                    auto descriptor =
                                        makeGpuDestinationDescriptor(
                                            destination_slot,
                                            manifest.N,
                                            manifest.K,
                                            manifest.gpu_codebook_id,
                                            manifest
                                                .gpu_payload_bytes_per_block,
                                            manifest.gpu_is_asymmetric != 0,
                                            manifest.gpu_has_emins != 0,
                                            source_identity);
                                    auto engine = makeGpuDestinationEngine(
                                        destination_device,
                                        descriptor,
                                        slot_lifetime,
                                        source_identity);
                                    *output = {
                                        .descriptor = descriptor,
                                        .engine = std::move(engine),
                                    };
                                    if (error)
                                        error->clear();
                                    return true;
                                };
                                auto gpu_storage = std::make_shared<
                                    MoEOverlayGpuRemoteProjectionDestination>(
                                    identity,
                                    lane->second[lane_index],
                                    std::move(factory),
                                    slot_lifetime);
                                binding.destination = std::make_shared<
                                    PublishingRemoteProjectionDestination>(
                                    gpu_storage,
                                    prepared.destination_arrival,
                                    projection,
                                    [gpu_storage]()
                                    {
                                        return gpu_storage->preparedEngine();
                                    });
                                if (migration.source.device.is_cpu())
                                    ++local_remote_gpu_cpu_operations;
                                else
                                    ++local_remote_gpu_blob_operations;
                            }
                        }

                        remote_targets.push_back({
                            .migration_index = migration_index,
                            .projection_index = projection_index,
                        });
                        remote_bindings.push_back(std::move(binding));
                        continue;
                    }

                    auto source_engine = projectionEngine(
                        *source_triplet, projection);
                    MoEOverlayPreparedWeightSource source;
                    std::string source_error;
                    if (!resolveMoEOverlayPreparedWeightSource(
                            source_engine,
                            migration.source.device,
                            source,
                            &source_error))
                    {
                        throw std::runtime_error(source_error);
                    }

                    std::shared_ptr<ITensorGemm> destination_engine;
                    std::unique_ptr<IMoEOverlayTierTransferOperation>
                        physical_operation;
                    const EdgeKey edge{
                        .source = migration.source.device,
                        .destination = migration.destination.device,
                        .projection = projection,
                    };
                    std::shared_ptr<SharedCpuCopyLanePool> cpu_copy_lane;
                    if (migration.source.device.is_cpu() &&
                        migration.destination.device.is_cpu())
                    {
                        const CpuAddressEdgeKey cpu_edge{
                            .source = migration.source.address,
                            .destination = migration.destination.address,
                            .projection = projection,
                        };
                        const auto found = impl_->cpu_copy_lanes.find(cpu_edge);
                        if (found == impl_->cpu_copy_lanes.end())
                            throw std::runtime_error(
                                "ExpertOverlay CPU address edge has no persistent parallel worker pool");
                        cpu_copy_lane = found->second;
                    }

                    if (migration.destination.device.is_cpu())
                    {
                        auto &lease = *destinations[migration_index].cpu;
                        const auto &destination = cpuProjection(
                            lease, projection);
                        destination_engine = destination.engine;
                        if (source.format.isFloating())
                        {
                            ContiguousFloatingPointWeightDescriptor
                                destination_floating;
                            if (!destination_engine
                                     ->exportContiguousFloatingPointWeights(
                                         destination_floating) ||
                                !destination_floating.valid() ||
                                source.floating.type !=
                                    destination_floating.type ||
                                source.floating.n != destination_floating.n ||
                                source.floating.k != destination_floating.k ||
                                source.floating.bytes !=
                                    destination_floating.bytes ||
                                destination.destination_bytes.data() !=
                                    destination_floating.data)
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay CPU floating destination disagrees with the model projection contract");
                            }
                            if (migration.source.device.is_cpu())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedCpuCopyOperation>(
                                        cpu_copy_lane,
                                        std::span<const std::uint8_t>(
                                            static_cast<const std::uint8_t *>(
                                                source.floating.data),
                                            source.floating.bytes),
                                        destination.destination_bytes,
                                        cpu_copy_wave_gate,
                                        source_engine,
                                        destination_engine);
                                ++local_cpu_copy_operations;
                            }
                            else
                            {
                                const auto lane = impl_->weight_lanes.find(edge);
                                if (lane == impl_->weight_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating GPU-to-CPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<AdmittedWeightOperation>(
                                        lane->second,
                                        AdmittedWeightOperation::Direction::
                                            GpuToCpuContiguous,
                                        source.floating,
                                        destination.destination_bytes,
                                        std::span<const std::uint8_t>{},
                                        ExpertTierSourceReadiness::
                                            publishedResidencyBank(
                                                expected_epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_gpu_cpu_operations;
                            }
                        }
                        else
                        {
                        const auto *destination_packed =
                            destination_engine
                                ->exportCPUNativeVNNIPackedWeights();
                        if (!destination_packed)
                            throw std::runtime_error(
                                "ExpertOverlay CPU destination engine cannot export final storage");

                        if (migration.source.device.is_cpu())
                        {
                            requireCompatibleCpuWeights(
                                *source.cpu_packed, *destination_packed);
                            physical_operation =
                                std::make_unique<AdmittedCpuCopyOperation>(
                                    cpu_copy_lane,
                                    std::span<const std::uint8_t>(
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size()),
                                    destination.destination_bytes,
                                    cpu_copy_wave_gate,
                                    source_engine,
                                    destination_engine);
                            ++local_cpu_copy_operations;
                        }
                        else
                        {
                            const auto *source_format =
                                native_vnni_formats::forSourceIdentity(
                                    source.format.native_vnni.codebook_id,
                                    source.format.native_vnni.is_superblock);
                            if (!source_format)
                                throw std::runtime_error(
                                    "ExpertOverlay GPU source lost catalogued provenance");
                            const auto probe =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    source.gpu_packed,
                                    expected_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    1);
                            const auto maximum_units = maximumUnitsPerChunk(
                                probe,
                                config_.staging_capacity_bytes);
                            const auto manifest =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    source.gpu_packed,
                                    expected_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    maximum_units);
                            const auto lane = impl_->weight_lanes.find(edge);
                            if (lane == impl_->weight_lanes.end())
                                throw std::runtime_error(
                                    "ExpertOverlay GPU-to-CPU edge has no persistent lane");
                            physical_operation =
                                std::make_unique<AdmittedWeightOperation>(
                                    lane->second,
                                    AdmittedWeightOperation::Direction::
                                        GpuToCpuRepacked,
                                    manifest.deviceLayout(),
                                    gpuConstView(source.gpu_packed),
                                    destination.destination_bytes,
                                    std::span<const std::uint8_t>{},
                                    ExpertTierGpuMutableProjectionView{},
                                    ExpertTierSourceReadiness::publishedResidencyBank(
                                        expected_epoch),
                                    source_engine,
                                    destination_engine);
                            ++local_gpu_cpu_operations;
                        }
                        }
                    }
                    else
                    {
                        auto &lease = *destinations[migration_index].gpu;
                        const auto &destination_slot = gpuProjection(
                            lease, projection);
                        GpuExpertPackedDescriptor destination_descriptor;

                        if (source.format.isFloating())
                        {
                            const auto destination_floating =
                                makeGpuFloatingDestinationDescriptor(
                                    destination_slot,
                                    source.format);
                            destination_engine =
                                makeGpuFloatingDestinationEngine(
                                    migration.destination.device,
                                    destination_floating,
                                    lease.lifetime);
                            if (migration.source.device.is_cpu())
                            {
                                const auto cpu_source =
                                    std::span<const std::uint8_t>(
                                        static_cast<const std::uint8_t *>(
                                            source.floating.data),
                                        source.floating.bytes);
                                const auto lane = impl_->weight_lanes.find(edge);
                                if (lane == impl_->weight_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating CPU-to-GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<AdmittedWeightOperation>(
                                        lane->second,
                                        AdmittedWeightOperation::Direction::
                                            CpuToGpuContiguous,
                                        destination_floating,
                                        std::span<std::uint8_t>{},
                                        cpu_source,
                                        ExpertTierSourceReadiness::
                                            publishedResidencyBank(
                                                expected_epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_gpu_cpu_operations;
                            }
                            else if (const auto peer_lane =
                                    impl_->peer_lanes.find(edge);
                                peer_lane != impl_->peer_lanes.end())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedPeerOperation>(
                                        peer_lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto blob_lane =
                                    impl_->blob_lanes.find(edge);
                                if (blob_lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating GPU edge has no persistent direct or host-relay lane");
                                physical_operation =
                                    std::make_unique<AdmittedBlobOperation>(
                                        blob_lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine);
                                if (migration.source.device.type ==
                                    migration.destination.device.type)
                                {
                                    ++local_same_backend_gpu_operations;
                                }
                                else
                                {
                                    ++local_heterogeneous_gpu_operations;
                                }
                            }
                        }
                        else if (migration.source.device.is_cpu())
                        {
                            const auto probe =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    *source.cpu_packed,
                                    candidate_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    1);
                            const auto maximum_units = maximumUnitsPerChunk(
                                probe,
                                config_.staging_capacity_bytes);
                            const auto manifest =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    *source.cpu_packed,
                                    candidate_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    maximum_units);
                            const auto layout = manifest.deviceLayout();
                            destination_descriptor =
                                makeGpuDestinationDescriptor(
                                    destination_slot,
                                    layout.N,
                                    layout.K,
                                    layout.gpu_codebook_id,
                                    layout.gpu_payload_bytes_per_block,
                                    layout.gpu_is_asymmetric != 0,
                                    layout.gpu_has_emins != 0,
                                    source.format.native_vnni);
                            destination_engine = makeGpuDestinationEngine(
                                migration.destination.device,
                                destination_descriptor,
                                lease.lifetime,
                                source.format.native_vnni);
                            const auto lane = impl_->weight_lanes.find(edge);
                            if (lane == impl_->weight_lanes.end())
                                throw std::runtime_error(
                                    "ExpertOverlay CPU-to-GPU edge has no persistent lane");
                            physical_operation =
                                std::make_unique<AdmittedWeightOperation>(
                                    lane->second,
                                    AdmittedWeightOperation::Direction::
                                        CpuToGpuRepacked,
                                    layout,
                                    ExpertTierGpuConstProjectionView{},
                                    std::span<std::uint8_t>{},
                                    std::span<const std::uint8_t>(
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size()),
                                    gpuMutableView(destination_descriptor),
                                    ExpertTierSourceReadiness::publishedResidencyBank(
                                        expected_epoch),
                                    source_engine,
                                    destination_engine);
                            ++local_gpu_cpu_operations;
                        }
                        else
                        {
                            destination_descriptor =
                                makeGpuDestinationDescriptor(
                                    destination_slot,
                                    source.gpu_packed.n,
                                    source.gpu_packed.k,
                                    source.gpu_packed.codebook_id,
                                    source.gpu_packed
                                        .payload_bytes_per_block,
                                    source.gpu_packed.is_asymmetric,
                                    source.gpu_packed.has_emins,
                                    source.format.native_vnni);
                            destination_engine = makeGpuDestinationEngine(
                                migration.destination.device,
                                destination_descriptor,
                                lease.lifetime,
                                source.format.native_vnni);

                            if (const auto peer_lane =
                                    impl_->peer_lanes.find(edge);
                                peer_lane != impl_->peer_lanes.end())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedPeerOperation>(
                                        peer_lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto blob_lane =
                                    impl_->blob_lanes.find(edge);
                                if (blob_lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay GPU edge has no persistent direct or host-relay lane");
                                physical_operation =
                                    std::make_unique<AdmittedBlobOperation>(
                                        blob_lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine);
                                if (migration.source.device.type ==
                                    migration.destination.device.type)
                                {
                                    ++local_same_backend_gpu_operations;
                                }
                                else
                                {
                                    ++local_heterogeneous_gpu_operations;
                                }
                            }
                        }
                    }

                    prepared.projections[projection_index] =
                        std::make_unique<
                            MoEOverlayPreparedProjectionOperation>(
                            std::move(physical_operation),
                            prepared.destination_arrival,
                            projection,
                            std::move(destination_engine));
                }
            }

            if (!remote_bindings.empty())
            {
                auto remote_wave = remote_transport->reserveWave(
                    std::move(remote_bindings));
                if (remote_wave.status ==
                    MoEOverlayResidencyStageStartStatus::Deferred)
                {
                    result.migrations.clear();
                    result.status = remote_wave.status;
                    result.error = std::move(remote_wave.error);
                    impl_->waves_deferred.fetch_add(
                        1, std::memory_order_relaxed);
                    return result;
                }
                if (remote_wave.status !=
                        MoEOverlayResidencyStageStartStatus::Started ||
                    remote_wave.operations.size() != remote_targets.size())
                {
                    result.migrations.clear();
                    return fail(
                        remote_wave.error.empty()
                            ? "ExpertOverlay remote projection transport returned an invalid wave shape"
                            : std::move(remote_wave.error));
                }
                for (std::size_t remote_index = 0;
                     remote_index < remote_targets.size();
                     ++remote_index)
                {
                    const auto target = remote_targets[remote_index];
                    result.migrations[target.migration_index]
                        .projections[target.projection_index] =
                        std::move(remote_wave.operations[remote_index]);
                }
            }
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        result.status = MoEOverlayResidencyStageStartStatus::Started;
        impl_->projection_operations_prepared.fetch_add(
            migrations.size() * kProjections.size(),
            std::memory_order_relaxed);
        impl_->cpu_copy_operations.fetch_add(
            local_cpu_copy_operations, std::memory_order_relaxed);
        impl_->remote_cpu_operations.fetch_add(
            local_remote_cpu_operations, std::memory_order_relaxed);
        impl_->remote_gpu_cpu_operations.fetch_add(
            local_remote_gpu_cpu_operations, std::memory_order_relaxed);
        impl_->remote_gpu_blob_operations.fetch_add(
            local_remote_gpu_blob_operations, std::memory_order_relaxed);
        impl_->gpu_cpu_operations.fetch_add(
            local_gpu_cpu_operations, std::memory_order_relaxed);
        impl_->same_backend_gpu_operations.fetch_add(
            local_same_backend_gpu_operations, std::memory_order_relaxed);
        impl_->heterogeneous_gpu_operations.fetch_add(
            local_heterogeneous_gpu_operations,
            std::memory_order_relaxed);
        impl_->waves_prepared.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "physical_waves_prepared",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"migrations", std::to_string(migrations.size())},
             {"projections",
              std::to_string(
                  migrations.size() * kProjections.size())},
             {"cpu_parallel_copy_operations",
              std::to_string(local_cpu_copy_operations)},
             {"remote_cpu_endpoints",
              std::to_string(local_remote_cpu_operations)},
             {"remote_gpu_cpu_endpoints",
              std::to_string(local_remote_gpu_cpu_operations)},
             {"remote_gpu_blob_endpoints",
              std::to_string(local_remote_gpu_blob_operations)}});
        return result;
    }

    bool MoEOverlayPhysicalResidencyFabric::stageDevicePreparedTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayParticipantPreparedTransfers &prepared,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!impl_->device_slot_ledger || !batch.valid() ||
            prepared.status != MoEOverlayResidencyStageStartStatus::Started ||
            prepared.migrations.size() != batch.migrations.size())
        {
            if (error)
                *error =
                    "ExpertOverlay device staging requires the exact completed physical wave";
            return false;
        }

        try
        {
            const auto local_ids = config_.registry->localParticipantIds();
            std::vector<MoEOverlayDeviceStagedPhysicalArrival> arrivals;
            arrivals.reserve(batch.migrations.size());
            for (std::size_t index = 0u;
                 index < batch.migrations.size();
                 ++index)
            {
                const auto &migration = batch.migrations[index];
                const bool local_destination = std::binary_search(
                    local_ids.begin(),
                    local_ids.end(),
                    migration.destination.owner_participant);
                const auto &arrival =
                    prepared.migrations[index].destination_arrival;
                if (!local_destination)
                {
                    if (arrival)
                    {
                        if (error)
                            *error =
                                "ExpertOverlay device staging received a local lifetime for a remote destination";
                        return false;
                    }
                    continue;
                }
                if (!arrival)
                {
                    if (error)
                        *error =
                            "ExpertOverlay device staging lost a local destination arrival";
                    return false;
                }

                MoEOverlayPreparedExpertTriplet triplet;
                std::string arrival_error;
                if (!arrival->completeTriplet(triplet, &arrival_error))
                {
                    if (error)
                    {
                        *error = arrival_error.empty()
                            ? "ExpertOverlay device staging observed an incomplete destination triplet"
                            : std::move(arrival_error);
                    }
                    return false;
                }
                arrivals.push_back({
                    .key = {
                        .participant_id =
                            migration.destination.owner_participant,
                        .layer_idx = migration.layer_idx,
                        .expert_id = migration.expert_id,
                    },
                    .triplet = std::move(triplet),
                });
            }

            if (!impl_->device_slot_ledger->stage(
                    batch, std::move(arrivals), error))
            {
                return false;
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "device_physical_destinations_staged",
                static_cast<double>(batch.migrations.size()),
                "maintenance",
                config_.perf_device,
                {{"transaction", std::to_string(batch.transaction_id)},
                 {"base_epoch", std::to_string(batch.base_epoch)},
                 {"candidate_epoch",
                  std::to_string(batch.candidate_epoch)}});
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
                *error =
                    "ExpertOverlay device staging failed with a non-standard exception";
            return false;
        }
    }

    bool MoEOverlayPhysicalResidencyFabric::publishDevicePreparedTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger ||
            !impl_->device_slot_ledger->publish(batch, error))
        {
            if (error && error->empty())
                *error =
                    "ExpertOverlay device physical publication lost its slot ledger";
            return false;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_physical_epoch_published",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"transaction", std::to_string(batch.transaction_id)},
             {"candidate_epoch", std::to_string(batch.candidate_epoch)}});
        return true;
    }

    bool MoEOverlayPhysicalResidencyFabric::retireDevicePreviousSources(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger)
        {
            if (error)
                *error =
                    "ExpertOverlay device retirement lost its slot ledger";
            return false;
        }
        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> retired;
        if (!impl_->device_slot_ledger->retire(batch, &retired, error))
            return false;

        /*
         * Bootstrap storage lacks an aliasing lease, so enroll its departed
         * assignment in the recyclable arena explicitly. Later-arrival slots
         * are released when `retired` destroys their final triplet aliases at
         * method exit. Neither path synchronizes a device or inference stream.
         */
        retirePreviousSources(batch.base_epoch, batch.migrations);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_physical_sources_retired",
            static_cast<double>(retired.size()),
            "maintenance",
            config_.perf_device,
            {{"transaction", std::to_string(batch.transaction_id)},
             {"retired_epoch", std::to_string(batch.base_epoch)}});
        return true;
    }

    bool MoEOverlayPhysicalResidencyFabric::abortDeviceTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger ||
            !impl_->device_slot_ledger->abort(batch, error))
        {
            if (error && error->empty())
                *error =
                    "ExpertOverlay device physical abort lost its slot ledger";
            return false;
        }
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "device_physical_wave_aborted",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"transaction", std::to_string(batch.transaction_id)}});
        return true;
    }

    void MoEOverlayPhysicalResidencyFabric::retirePreviousSources(
        std::uint64_t retired_epoch,
        const std::vector<MoEOverlayTierMigration> &migrations) noexcept
    {
        std::uint64_t recycled = 0;
        for (const auto &migration : migrations)
        {
            /* A missing endpoint means this rank does not own the source. */
            if (!config_.registry->endpoint(
                    migration.source.owner_participant))
            {
                continue;
            }

            const auto found = impl_->endpoint_pools.find({
                migration.source.owner_participant,
                migration.layer_idx,
            });
            if (found == impl_->endpoint_pools.end() ||
                !found->second.geometry_pool ||
                !found->second.geometry_pool->adopted_slots)
            {
                /* Local topology disappearing after publication is unrecoverable. */
                std::terminate();
            }
            if (found->second.geometry_pool->adopted_slots
                    ->retireBootstrapAssignment(
                        migration.layer_idx,
                        migration.expert_id,
                        retired_epoch))
            {
                ++recycled;
            }
        }

        if (recycled == 0)
            return;
        impl_->adopted_initial_slots_recycled.fetch_add(
            recycled, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "bootstrap_live_slots_recycled",
            static_cast<double>(recycled),
            "maintenance",
            config_.perf_device,
            {{"retired_epoch", std::to_string(retired_epoch)}});
    }

    std::shared_ptr<MappedTransferProgressEpoch>
    MoEOverlayPhysicalResidencyFabric::transferProgressEpoch(
        DeviceId device) const noexcept
    {
        const auto found = impl_->transfer_progress_epochs.find(device);
        return found == impl_->transfer_progress_epochs.end()
                   ? nullptr
                   : found->second;
    }

    std::vector<DeviceId>
    MoEOverlayPhysicalResidencyFabric::transferProgressDevices() const
    {
        std::vector<DeviceId> devices;
        devices.reserve(impl_->transfer_progress_epochs.size());
        for (const auto &[device, epoch] :
             impl_->transfer_progress_epochs)
        {
            if (!epoch || epoch->device() != device)
            {
                throw std::logic_error(
                    "ExpertOverlay physical fabric retained an invalid transfer-progress inventory entry");
            }
            devices.push_back(device);
        }
        return devices;
    }

    bool MoEOverlayPhysicalResidencyFabric::
        submitOutstandingTransferProgress(std::string *error) noexcept
    {
        for (const auto &[device, epoch] : impl_->transfer_progress_epochs)
        {
            if (!epoch || epoch->device() != device)
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay physical fabric retained an invalid transfer-progress epoch for " +
                        device.toString();
                }
                return false;
            }
            if (!epoch->submitOutstandingProgress())
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay could not enqueue mapped transfer progress on " +
                        device.toString();
                }
                return false;
            }
        }
        return true;
    }

    std::optional<MoEOverlayReusableContextSeal>
    MoEOverlayPhysicalResidencyFabric::sealReusableInitialPlacement(
        std::uint64_t published_epoch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();

        std::unique_lock<std::mutex> seal_lock(impl_->reusable_seal_mutex);
        const auto fail = [&](std::string message)
            -> std::optional<MoEOverlayReusableContextSeal>
        {
            impl_->reusable_seal_failure = std::move(message);
            impl_->reusable_seal_state = Impl::ReusableSealState::Failed;
            if (error)
                *error = impl_->reusable_seal_failure;
            return std::nullopt;
        };

        if (impl_->reusable_seal_state == Impl::ReusableSealState::Sealed)
            return impl_->cached_reusable_seal;
        if (impl_->reusable_seal_state == Impl::ReusableSealState::Failed)
        {
            if (error)
                *error = impl_->reusable_seal_failure;
            return std::nullopt;
        }
        if (impl_->reusable_seal_state != Impl::ReusableSealState::Open)
        {
            return fail(
                "ExpertOverlay reusable-context seal was entered concurrently");
        }
        impl_->reusable_seal_state = Impl::ReusableSealState::Sealing;

        try
        {
            if (published_epoch == 0 || !config_.registry ||
                !config_.initial_snapshot ||
                !config_.initial_snapshot->valid())
            {
                return fail(
                    "ExpertOverlay reusable-context seal requires a positive published epoch and complete initial placement");
            }

            MoEOverlayReusableContextSeal result;
            result.source_epoch = published_epoch;
            result.canonical_owner_map =
                config_.initial_snapshot->owner_map;
            const auto participant_ids =
                config_.registry->localParticipantIds();
            result.local_banks.reserve(participant_ids.size());

            /*
             * Materialize the exact terminal inventory from its typed owner.
             * Host RCU keeps participant banks current. Device RCU intentionally
             * does not: its controller owns selectors and its physical ledger
             * owns engine lifetimes. Reading the registry in that mode would be
             * an informal host mirror and, after movement, names the wrong
             * epoch. Both authorities produce the same immutable bank value for
             * the common capacity/compaction proof below.
             */
            if (config_.inventory_authority ==
                MoEOverlayPhysicalInventoryAuthority::ParticipantRegistry)
            {
                for (const int participant_id : participant_ids)
                {
                    const auto endpoint =
                        config_.registry->endpoint(participant_id);
                    auto bank = endpoint
                        ? endpoint->acquire(published_epoch)
                        : MoEOverlayParticipantBankLease{};
                    if (!endpoint || !bank ||
                        !bank->valid(
                            participant_id,
                            endpoint->device(),
                            endpoint->numLayers(),
                            endpoint->numExperts()))
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal cannot acquire participant " +
                            std::to_string(participant_id) + " at epoch " +
                            std::to_string(published_epoch) +
                            " from the host participant-bank authority");
                    }
                    result.local_banks.push_back(*bank);
                }
            }
            else if (config_.inventory_authority ==
                     MoEOverlayPhysicalInventoryAuthority::DeviceSlotLedger)
            {
                if (!impl_->device_slot_ledger)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal lost its device physical inventory authority");
                }
                std::string inventory_error;
                const auto inventory = impl_->device_slot_ledger->snapshot(
                    published_epoch, &inventory_error);
                if (!inventory || !inventory->valid())
                {
                    return fail(
                        inventory_error.empty()
                            ? "ExpertOverlay reusable-context seal could not snapshot the device physical inventory"
                            : std::move(inventory_error));
                }

                std::map<int, std::size_t> bank_index_by_participant;
                for (const int participant_id : participant_ids)
                {
                    const auto endpoint =
                        config_.registry->endpoint(participant_id);
                    const auto *participant =
                        config_.initial_snapshot->owner_map.participantForId(
                            participant_id);
                    if (!endpoint || !participant ||
                        endpoint->device() != participant->device)
                    {
                        return fail(
                            "ExpertOverlay device physical inventory cannot resolve local participant " +
                            std::to_string(participant_id));
                    }

                    MoEOverlayParticipantResidencyBank bank{
                        .epoch = published_epoch,
                        .participant_id = participant_id,
                        .device = endpoint->device(),
                    };
                    bank.layers.resize(
                        static_cast<std::size_t>(endpoint->numLayers()));
                    for (auto &layer : bank.layers)
                    {
                        layer.resident_mask.assign(
                            static_cast<std::size_t>(endpoint->numExperts()),
                            false);
                        layer.experts.resize(
                            static_cast<std::size_t>(endpoint->numExperts()));
                    }
                    bank_index_by_participant.emplace(
                        participant_id, result.local_banks.size());
                    result.local_banks.push_back(std::move(bank));
                }

                for (const auto &slot : inventory->slots)
                {
                    const auto bank_index = bank_index_by_participant.find(
                        slot.key.participant_id);
                    if (bank_index == bank_index_by_participant.end())
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains a non-local participant slot");
                    }
                    auto &bank = result.local_banks.at(bank_index->second);
                    if (slot.key.layer_idx < 0 ||
                        slot.key.expert_id < 0 ||
                        static_cast<std::size_t>(slot.key.layer_idx) >=
                            bank.layers.size() ||
                        static_cast<std::size_t>(slot.key.expert_id) >=
                            bank.layers.at(
                                static_cast<std::size_t>(slot.key.layer_idx))
                                .experts.size())
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains an out-of-range expert coordinate");
                    }
                    auto &layer = bank.layers.at(
                        static_cast<std::size_t>(slot.key.layer_idx));
                    if (layer.resident_mask.at(
                            static_cast<std::size_t>(slot.key.expert_id)))
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains a duplicate expert coordinate");
                    }
                    layer.setResidentExpert(
                        slot.key.expert_id, slot.triplet);
                }
            }
            else
            {
                return fail(
                    "ExpertOverlay reusable-context seal has an unknown physical inventory authority");
            }

            /*
             * Prove the complete prepared placement before submitting a single
             * byte copy. This makes a partial physical seal impossible for an
             * owner-map or adopted-capacity defect.
             */
            std::map<
                const AdoptedInitialExpertSlotRecycler *,
                std::pair<
                    std::shared_ptr<AdoptedInitialExpertSlotRecycler>,
                    std::size_t>>
                compaction_demand_by_geometry;
            for (auto &canonical : result.local_banks)
            {
                const int participant_id = canonical.participant_id;
                const auto endpoint = config_.registry->endpoint(participant_id);
                const auto *participant =
                    config_.initial_snapshot->owner_map.participantForId(
                        participant_id);
                if (!endpoint || !participant ||
                    !canonical.valid(
                        participant_id,
                        endpoint->device(),
                        endpoint->numLayers(),
                        endpoint->numExperts()))
                {
                    return fail(
                        "ExpertOverlay reusable-context seal materialized an incomplete local participant bank");
                }
                for (int layer_idx = 0;
                     layer_idx < endpoint->numLayers();
                     ++layer_idx)
                {
                    auto &layer = canonical.layers.at(
                        static_cast<std::size_t>(layer_idx));
                    const auto expected_mask =
                        config_.initial_snapshot->owner_map
                            .expertMaskForParticipant(
                                layer_idx,
                                participant_id,
                                endpoint->numExperts());
                    if (layer.resident_mask != expected_mask)
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal was requested before the initial prepared owner map was restored");
                    }

                    auto &pool = requireEndpointPool(
                        *impl_, participant_id, layer_idx);
                    if (!pool.geometry_pool ||
                        !pool.geometry_pool->adopted_slots ||
                        pool.geometry_pool->adopted_slots
                                ->initialCapacityForLayer(layer_idx) !=
                            residentCount(layer))
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal found an adopted-slot capacity that disagrees with the prepared placement");
                    }

                    std::size_t needs_compaction = 0;
                    for (int expert_id = 0;
                         expert_id < endpoint->numExperts();
                         ++expert_id)
                    {
                        if (!layer.resident_mask.at(
                                static_cast<std::size_t>(expert_id)))
                        {
                            continue;
                        }
                        const auto &triplet = layer.experts.at(
                            static_cast<std::size_t>(expert_id));
                        if (!triplet.complete())
                        {
                            return fail(
                                "ExpertOverlay reusable-context seal found an incomplete resident triplet");
                        }
                        if (!pool.geometry_pool->adopted_slots
                                 ->ownsAssignment(layer_idx, expert_id))
                            ++needs_compaction;
                    }
                    auto &geometry_demand =
                        compaction_demand_by_geometry[
                            pool.geometry_pool->adopted_slots.get()];
                    if (!geometry_demand.first)
                    {
                        geometry_demand.first =
                            pool.geometry_pool->adopted_slots;
                    }
                    if (needs_compaction >
                        std::numeric_limits<std::size_t>::max() -
                            geometry_demand.second)
                    {
                        return fail(
                            "ExpertOverlay reusable-context compaction demand overflows size_t");
                    }
                    geometry_demand.second += needs_compaction;
                }
            }
            for (const auto &[_, demand] : compaction_demand_by_geometry)
            {
                if (!demand.first ||
                    demand.first->availableSlots() < demand.second)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal cannot fit every shadow resident into the shared prepared allocation arena");
                }
            }

            const auto completion_deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(30);

            /*
             * All operations below own setup-time lanes and persistent source/
             * destination engines. Polling observes events or worker state only;
             * it never synchronizes a device or an inference stream.
             */
            const auto complete_operations = [completion_deadline](
                std::vector<std::unique_ptr<
                    IMoEOverlayTierTransferOperation>> &operations,
                std::string *operation_error) -> bool
            {
                std::vector<bool> ready(operations.size(), false);
                std::size_t ready_count = 0;
                std::string first_failure;
                while (ready_count != operations.size())
                {
                    for (std::size_t index = 0;
                         index < operations.size();
                         ++index)
                    {
                        if (ready[index])
                            continue;
                        std::string poll_error;
                        const auto progress =
                            operations[index]->poll(&poll_error);
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Ready)
                        {
                            ready[index] = true;
                            ++ready_count;
                        }
                        else if (progress ==
                                 MoEOverlayResidencyWaveProgress::Failed)
                        {
                            first_failure = poll_error.empty()
                                ? "ExpertOverlay reusable-context physical copy failed"
                                : std::move(poll_error);
                            break;
                        }
                    }
                    if (!first_failure.empty() ||
                        std::chrono::steady_clock::now() >=
                            completion_deadline)
                    {
                        if (first_failure.empty())
                        {
                            first_failure =
                                "ExpertOverlay reusable-context physical copy exceeded the canonical 30-second protocol deadline";
                        }
                        break;
                    }
                    if (ready_count != operations.size())
                        std::this_thread::yield();
                }
                if (first_failure.empty())
                    return true;

                for (auto &operation : operations)
                    operation->abort();
                const auto abort_deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(30);
                for (;;)
                {
                    bool drained = true;
                    for (auto &operation : operations)
                    {
                        std::string ignored;
                        if (operation->pollAbort(&ignored) ==
                            MoEOverlayResidencyWaveProgress::Pending)
                        {
                            drained = false;
                        }
                    }
                    if (drained)
                        break;
                    if (std::chrono::steady_clock::now() >= abort_deadline)
                    {
                        /* Destruction with DMA or a CPU worker live is illegal. */
                        std::terminate();
                    }
                    std::this_thread::yield();
                }
                if (operation_error)
                    *operation_error = std::move(first_failure);
                return false;
            };

            const auto set_projection = [](
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
                throw std::logic_error(
                    "ExpertOverlay reusable-context seal received an unknown projection role");
            };

            for (auto &bank : result.local_banks)
            {
                const auto endpoint =
                    config_.registry->endpoint(bank.participant_id);
                const auto *participant =
                    config_.initial_snapshot->owner_map.participantForId(
                        bank.participant_id);
                if (!endpoint || !participant)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal lost a preflighted participant");
                }

                for (int layer_idx = 0;
                     layer_idx < endpoint->numLayers();
                     ++layer_idx)
                {
                    auto &layer = bank.layers.at(
                        static_cast<std::size_t>(layer_idx));
                    auto &pool = requireEndpointPool(
                        *impl_, bank.participant_id, layer_idx);
                    if (!pool.geometry_pool ||
                        !pool.geometry_pool->adopted_slots)
                    {
                        return fail(
                            "ExpertOverlay reusable-context compaction lost its geometry arena");
                    }
                    for (int expert_id = 0;
                         expert_id < endpoint->numExperts();
                         ++expert_id)
                    {
                        if (!layer.resident_mask.at(
                                static_cast<std::size_t>(expert_id)))
                        {
                            continue;
                        }
                        if (pool.geometry_pool->adopted_slots
                                ->ownsAssignment(layer_idx, expert_id))
                        {
                            ++result.retained_canonical_experts;
                            continue;
                        }

                        const auto source_triplet = layer.experts.at(
                            static_cast<std::size_t>(expert_id));
                        MoEOverlayPreparedExpertTriplet destination_triplet;
                        std::vector<std::unique_ptr<
                            IMoEOverlayTierTransferOperation>> operations;
                        operations.reserve(kProjections.size());

                        if (bank.device.is_cpu())
                        {
                            auto destination =
                                pool.geometry_pool->adopted_slots->acquireCpu(
                                    layer_idx,
                                    expert_id,
                                    published_epoch);
                            if (!destination)
                            {
                                return fail(
                                    "ExpertOverlay reusable-context seal lost a preflighted CPU adopted slot");
                            }
                            auto gate = std::make_shared<CpuCopyWaveGate>(
                                kProjections.size());
                            for (const auto projection : kProjections)
                            {
                                auto source_engine = projectionEngine(
                                    source_triplet, projection);
                                MoEOverlayPreparedWeightSource source;
                                std::string source_error;
                                if (!resolveMoEOverlayPreparedWeightSource(
                                        source_engine,
                                        bank.device,
                                        source,
                                        &source_error))
                                {
                                    return fail(std::move(source_error));
                                }
                                const auto &target = cpuProjection(
                                    *destination, projection);
                                std::span<const std::uint8_t> source_bytes;
                                if (source.format.isFloating())
                                {
                                    source_bytes = {
                                        static_cast<const std::uint8_t *>(
                                            source.floating.data),
                                        source.floating.bytes,
                                    };
                                }
                                else if (source.cpu_packed)
                                {
                                    source_bytes = {
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size(),
                                    };
                                }
                                if (source_bytes.empty() ||
                                    source_bytes.size() !=
                                        target.destination_bytes.size())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context CPU source and prepared allocation disagree on byte size");
                                }
                                const CpuAddressEdgeKey edge{
                                    .source = participant->address,
                                    .destination = participant->address,
                                    .projection = projection,
                                };
                                const auto lane =
                                    impl_->cpu_copy_lanes.find(edge);
                                if (lane == impl_->cpu_copy_lanes.end())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context CPU compaction lane was not materialized at setup");
                                }
                                operations.push_back(
                                    std::make_unique<
                                        AdmittedCpuCopyOperation>(
                                        lane->second,
                                        source_bytes,
                                        target.destination_bytes,
                                        gate,
                                        source_engine,
                                        target.engine));
                                set_projection(
                                    destination_triplet,
                                    projection,
                                    target.engine);
                            }
                        }
                        else
                        {
                            auto destination =
                                pool.geometry_pool->adopted_slots->acquireGpu(
                                    layer_idx,
                                    expert_id,
                                    published_epoch);
                            if (!destination)
                            {
                                return fail(
                                    "ExpertOverlay reusable-context seal lost a preflighted GPU adopted slot");
                            }
                            for (const auto projection : kProjections)
                            {
                                auto source_engine = projectionEngine(
                                    source_triplet, projection);
                                MoEOverlayPreparedWeightSource source;
                                std::string source_error;
                                if (!resolveMoEOverlayPreparedWeightSource(
                                        source_engine,
                                        bank.device,
                                        source,
                                        &source_error))
                                {
                                    return fail(std::move(source_error));
                                }
                                const auto &target = gpuProjection(
                                    *destination, projection);
                                const EdgeKey edge{
                                    .source = bank.device,
                                    .destination = bank.device,
                                    .projection = projection,
                                };
                                const auto lane = impl_->peer_lanes.find(edge);
                                if (lane == impl_->peer_lanes.end())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context GPU compaction lane was not materialized at setup");
                                }

                                std::shared_ptr<ITensorGemm> target_engine;
                                if (source.format.isFloating())
                                {
                                    const auto descriptor =
                                        makeGpuFloatingDestinationDescriptor(
                                            target, source.format);
                                    target_engine =
                                        makeGpuFloatingDestinationEngine(
                                            bank.device,
                                            descriptor,
                                            destination->lifetime);
                                    operations.push_back(
                                        std::make_unique<
                                            AdmittedPeerOperation>(
                                            lane->second,
                                            source.floating,
                                            descriptor,
                                            ExpertTierSourceReadiness::
                                                publishedResidencyBank(
                                                    published_epoch),
                                            source_engine,
                                            target_engine));
                                }
                                else
                                {
                                    const auto descriptor =
                                        makeGpuDestinationDescriptor(
                                            target,
                                            source.gpu_packed.n,
                                            source.gpu_packed.k,
                                            source.gpu_packed.codebook_id,
                                            source.gpu_packed
                                                .payload_bytes_per_block,
                                            source.gpu_packed.is_asymmetric,
                                            source.gpu_packed.has_emins,
                                            source.format.native_vnni);
                                    target_engine = makeGpuDestinationEngine(
                                        bank.device,
                                        descriptor,
                                        destination->lifetime,
                                        source.format.native_vnni);
                                    operations.push_back(
                                        std::make_unique<
                                            AdmittedPeerOperation>(
                                            lane->second,
                                            source.gpu_packed,
                                            descriptor,
                                            ExpertTierSourceReadiness::
                                                publishedResidencyBank(
                                                    published_epoch),
                                            source_engine,
                                            target_engine));
                                }
                                set_projection(
                                    destination_triplet,
                                    projection,
                                    std::move(target_engine));
                            }
                        }

                        std::string operation_error;
                        if (!destination_triplet.complete() ||
                            !complete_operations(
                                operations, &operation_error))
                        {
                            return fail(
                                operation_error.empty()
                                    ? "ExpertOverlay reusable-context compaction produced an incomplete destination triplet"
                                    : std::move(operation_error));
                        }
                        layer.experts.at(
                            static_cast<std::size_t>(expert_id)) =
                            std::move(destination_triplet);
                        ++result.compacted_shadow_experts;
                    }
                }
                if (!bank.valid(
                        bank.participant_id,
                        bank.device,
                        endpoint->numLayers(),
                        endpoint->numExperts()))
                {
                    return fail(
                        "ExpertOverlay reusable-context seal produced an invalid canonical bank");
                }
            }

            if (!result.valid())
            {
                return fail(
                    "ExpertOverlay reusable-context seal produced an incomplete terminal value");
            }
            impl_->cached_reusable_seal = std::move(result);
            impl_->reusable_seal_state = Impl::ReusableSealState::Sealed;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "reusable_context_physical_seals",
                1.0,
                "model_teardown",
                config_.perf_device,
                {{"epoch", std::to_string(published_epoch)},
                 {"canonical_experts",
                  std::to_string(
                      impl_->cached_reusable_seal
                          ->retained_canonical_experts)},
                 {"compacted_shadow_experts",
                  std::to_string(
                      impl_->cached_reusable_seal
                          ->compacted_shadow_experts)}});
            return impl_->cached_reusable_seal;
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }
        catch (...)
        {
            return fail(
                "ExpertOverlay reusable-context seal raised a non-standard exception");
        }
    }

    MoEOverlayPhysicalResidencyFabricStats
    MoEOverlayPhysicalResidencyFabric::stats() const noexcept
    {
        std::uint64_t parallel_lane_reservations = 0;
        std::uint64_t parallel_lane_pool_exhaustions = 0;
        const auto accumulate_pool_stats = [&](const auto &pools)
        {
            for (const auto &[_, pool] : pools)
            {
                parallel_lane_reservations += pool->reservations();
                parallel_lane_pool_exhaustions += pool->exhaustions();
            }
        };
        accumulate_pool_stats(impl_->weight_lanes);
        accumulate_pool_stats(impl_->blob_lanes);
        accumulate_pool_stats(impl_->peer_lanes);
        std::uint64_t parallel_cpu_copy_lane_reservations = 0;
        std::uint64_t parallel_cpu_copy_lane_pool_exhaustions = 0;
        for (const auto &[_, pool] : impl_->cpu_copy_lanes)
        {
            parallel_cpu_copy_lane_reservations += pool->reservations();
            parallel_cpu_copy_lane_pool_exhaustions += pool->exhaustions();
        }

        return {
            .endpoint_layer_pools =
                impl_->endpoint_layer_pools.load(std::memory_order_relaxed),
            .endpoint_geometry_pools =
                impl_->endpoint_geometry_pools.load(
                    std::memory_order_relaxed),
            .adopted_initial_slots =
                impl_->adopted_initial_slots.load(
                    std::memory_order_relaxed),
            .adopted_initial_slots_recycled =
                impl_->adopted_initial_slots_recycled.load(
                    std::memory_order_relaxed),
            .cpu_shadow_slots =
                impl_->cpu_shadow_slots.load(std::memory_order_relaxed),
            .gpu_shadow_slots =
                impl_->gpu_shadow_slots.load(std::memory_order_relaxed),
            .persistent_transfer_lanes =
                impl_->persistent_transfer_lanes.load(
                    std::memory_order_relaxed),
            .direct_gpu_peer_lanes =
                impl_->direct_gpu_peer_lanes.load(
                    std::memory_order_relaxed),
            .same_backend_no_peer_relay_lanes =
                impl_->same_backend_no_peer_relay_lanes.load(
                    std::memory_order_relaxed),
            .cross_backend_gpu_relay_lanes =
                impl_->cross_backend_gpu_relay_lanes.load(
                    std::memory_order_relaxed),
            .maximum_parallel_edge_lanes =
                impl_->maximum_parallel_edge_lanes.load(
                    std::memory_order_relaxed),
            .parallel_lane_reservations = parallel_lane_reservations,
            .parallel_lane_pool_exhaustions =
                parallel_lane_pool_exhaustions,
            .serialized_lane_deferrals = 0,
            .maximum_parallel_cpu_copy_lanes =
                impl_->maximum_parallel_cpu_copy_lanes.load(
                    std::memory_order_relaxed),
            .parallel_cpu_copy_lane_reservations =
                parallel_cpu_copy_lane_reservations,
            .parallel_cpu_copy_lane_pool_exhaustions =
                parallel_cpu_copy_lane_pool_exhaustions,
            .maximum_concurrent_cpu_copy_operations =
                impl_->cpu_copy_concurrency->peak(),
            .active_cpu_copy_operations =
                impl_->cpu_copy_concurrency->active(),
            .waves_prepared =
                impl_->waves_prepared.load(std::memory_order_relaxed),
            .waves_deferred =
                impl_->waves_deferred.load(std::memory_order_relaxed),
            .waves_failed =
                impl_->waves_failed.load(std::memory_order_relaxed),
            .projection_operations_prepared =
                impl_->projection_operations_prepared.load(
                    std::memory_order_relaxed),
            .cpu_copy_operations =
                impl_->cpu_copy_operations.load(std::memory_order_relaxed),
            .remote_cpu_operations =
                impl_->remote_cpu_operations.load(
                    std::memory_order_relaxed),
            .remote_gpu_cpu_operations =
                impl_->remote_gpu_cpu_operations.load(
                    std::memory_order_relaxed),
            .remote_gpu_blob_operations =
                impl_->remote_gpu_blob_operations.load(
                    std::memory_order_relaxed),
            .gpu_cpu_operations =
                impl_->gpu_cpu_operations.load(std::memory_order_relaxed),
            .same_backend_gpu_operations =
                impl_->same_backend_gpu_operations.load(
                    std::memory_order_relaxed),
            .heterogeneous_gpu_operations =
                impl_->heterogeneous_gpu_operations.load(
                    std::memory_order_relaxed),
            .inference_stream_waits = 0,
            .blocking_synchronizations = 0,
        };
    }
} // namespace llaminar2
