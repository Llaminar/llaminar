/**
 * @file MoEOverlayPhysicalResidencyFabric.cpp
 * @brief Local slot reservation and queued physical movement for ExpertOverlay.
 *
 * One maintenance worker polls every operation in a wave.  Each directed
 * device edge owns one persistent lane for gate, up, and down, so the three
 * projections overlap while additional experts queue without creating more
 * streams or staging allocations.  Queue admission and destination-slot
 * reservation are complete before the first poll can submit bytes.
 */

#include "MoEOverlayPhysicalResidencyFabric.h"

#include "CpuExpertSlotPool.h"
#include "ExpertTierGpuBlobTransferLane.h"
#include "ExpertTierGpuPeerTransferLane.h"
#include "ExpertTierWeightTransferLane.h"
#include "GpuExpertSlotPool.h"
#include "MoEOverlayGpuRemoteProjectionEndpoint.h"
#include "MoEOverlayMPIRemoteProjectionTransport.h"
#include "MoEOverlayPreparedWeightSource.h"
#include "loaders/ModelLoader.h"
#include "backends/BackendManager.h"
#include "kernels/cpu/gemm/CPUNativeVNNIGemmKernel.h"
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
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <tuple>
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

            /** @brief Try to assign the lane to one exact queued operation. */
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

        using SharedWeightLane = SharedLane<ExpertTierWeightTransferLane>;
        using SharedBlobLane = SharedLane<ExpertTierGpuBlobTransferLane>;
        using SharedPeerLane = SharedLane<ExpertTierGpuPeerTransferLane>;

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
         * @brief Queued GPU/CPU conversion operation over one reusable lane.
         *
         * Queueing is important: a 16-expert wave still owns only three streams
         * per GPU/CPU edge.  The first poll that obtains the lane submits the
         * first chunk; later polls advance only event-ready work.
         */
        class QueuedWeightOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** Direction-specific data retained until completion or abort. */
            enum class Direction
            {
                GpuToCpu,
                CpuToGpu,
            };

            /**
             * @brief Bind an admitted operation to persistent source/destination storage.
             * @param lane Shared physical lane for the exact directed edge.
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
            QueuedWeightOperation(
                std::shared_ptr<SharedWeightLane> lane,
                Direction direction,
                ExpertTierWeightDeviceLayout layout,
                ExpertTierGpuConstProjectionView gpu_source,
                std::span<std::uint8_t> cpu_destination,
                std::span<const std::uint8_t> cpu_source,
                ExpertTierGpuMutableProjectionView gpu_destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine)
                : lane_(std::move(lane)),
                  direction_(direction),
                  layout_(layout),
                  gpu_source_(gpu_source),
                  cpu_destination_(cpu_destination),
                  cpu_source_(cpu_source),
                  gpu_destination_(gpu_destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
            {
                if (!lane_ || !layout_.valid() || !source_engine_ ||
                    !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay queued GPU/CPU operation has incomplete ownership");
                }
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
                    if (!lane_->tryAcquire(this))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    owns_lane_ = true;

                    std::string start_error;
                    const bool started =
                        direction_ == Direction::GpuToCpu
                            ? lane_->lane()->startGpuToCpu(
                                  layout_,
                                  gpu_source_,
                                  cpu_destination_,
                                  source_readiness_,
                                  &start_error)
                            : lane_->lane()->startCpuToGpu(
                                  layout_,
                                  cpu_source_,
                                  gpu_destination_,
                                  &start_error);
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
                     * shared lane. A queued successor may reuse that lane in
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

            /** @brief Discard queued work or mark submitted work for draining. */
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
                if (!started_ || !owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;

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

        /** @brief Queued byte-preserving CUDA/ROCm operation. */
        class QueuedBlobOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain both descriptors, slots, and one shared blob lane. */
            QueuedBlobOperation(
                std::shared_ptr<SharedBlobLane> lane,
                GpuExpertPackedDescriptor source,
                GpuExpertPackedDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine)
                : lane_(std::move(lane)),
                  source_(source),
                  destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
            {
                if (!lane_ || !source_.valid() || !destination_.valid() ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay queued heterogeneous GPU operation has incomplete ownership");
                }
            }

            /** @brief Retain one contiguous floating source/destination pair. */
            QueuedBlobOperation(
                std::shared_ptr<SharedBlobLane> lane,
                ContiguousFloatingPointWeightDescriptor source,
                ContiguousFloatingPointWeightDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine)
                : lane_(std::move(lane)),
                  floating_source_(source),
                  floating_destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_(true)
            {
                if (!lane_ || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != floating_destination_.n ||
                    floating_source_.k != floating_destination_.k ||
                    floating_source_.bytes != floating_destination_.bytes ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay queued heterogeneous floating GPU operation has incompatible storage or ownership");
                }
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
                        "ExpertOverlay heterogeneous GPU operation was aborted",
                        error);
                if (ready_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (failed_)
                    return failOperation(failure_, failure_, error);
                if (!started_)
                {
                    if (!lane_->tryAcquire(this))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    owns_lane_ = true;
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
                                ? "ExpertOverlay heterogeneous GPU lane failed to start"
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
                    /* Preserve evidence before another queued blob reuses it. */
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
                        ? "ExpertOverlay heterogeneous GPU lane failed"
                        : std::move(poll_error),
                    error);
            }

            /** @brief Mark an unpublished queued or submitted blob discarded. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief Drain both runtime event sets before releasing the edge. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (!aborted_)
                    return failOperation(
                        failure_,
                        "ExpertOverlay heterogeneous GPU abort was not requested",
                        error);
                if (!started_ || !owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
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

        /** @brief Queued same-backend GPU peer-copy operation. */
        class QueuedPeerOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain compatible packed descriptors and their slots. */
            QueuedPeerOperation(
                std::shared_ptr<SharedPeerLane> lane,
                GpuExpertPackedDescriptor source,
                GpuExpertPackedDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine)
                : lane_(std::move(lane)),
                  source_(source),
                  destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine))
            {
                if (!lane_ || !source_.valid() || !destination_.valid() ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay queued GPU peer operation has incomplete ownership");
                }
            }

            /** @brief Retain one contiguous floating peer-copy pair. */
            QueuedPeerOperation(
                std::shared_ptr<SharedPeerLane> lane,
                ContiguousFloatingPointWeightDescriptor source,
                ContiguousFloatingPointWeightDescriptor destination,
                ExpertTierSourceReadiness source_readiness,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine)
                : lane_(std::move(lane)),
                  floating_source_(source),
                  floating_destination_(destination),
                  source_readiness_(source_readiness),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  floating_(true)
            {
                if (!lane_ || !floating_source_.valid() ||
                    !floating_destination_.valid() ||
                    floating_source_.type != floating_destination_.type ||
                    floating_source_.n != floating_destination_.n ||
                    floating_source_.k != floating_destination_.k ||
                    floating_source_.bytes != floating_destination_.bytes ||
                    !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay queued floating GPU peer operation has incompatible storage or ownership");
                }
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
                    if (!lane_->tryAcquire(this))
                        return MoEOverlayResidencyWaveProgress::Pending;
                    owns_lane_ = true;
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
                    /* Preserve evidence before another queued peer copy starts. */
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

            /** @brief Mark queued work discarded without cancelling DMA. */
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
                if (!started_ || !owns_lane_)
                    return MoEOverlayResidencyWaveProgress::Ready;
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

        /**
         * @brief Bounded host-native copy between CPU/NUMA participants.
         *
         * A poll copies at most one staging-capacity chunk.  The maintenance
         * thread performs the copy, so inference never joins it and large CPU
         * experts cannot monopolize one maintenance iteration.
         */
        class QueuedCpuCopyOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Retain final source/destination storage and chunk budget. */
            QueuedCpuCopyOperation(
                std::span<const std::uint8_t> source,
                std::span<std::uint8_t> destination,
                std::size_t chunk_bytes,
                std::shared_ptr<ITensorGemm> source_engine,
                std::shared_ptr<ITensorGemm> destination_engine,
                std::string perf_device)
                : source_(source),
                  destination_(destination),
                  chunk_bytes_(chunk_bytes),
                  source_engine_(std::move(source_engine)),
                  destination_engine_(std::move(destination_engine)),
                  perf_device_(std::move(perf_device))
            {
                if (source_.empty() || source_.size() != destination_.size() ||
                    chunk_bytes_ == 0 || !source_engine_ || !destination_engine_)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay CPU copy requires exact final storage and ownership");
                }
            }

            /** @brief Copy one bounded range and report readiness exactly. */
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
                if (offset_ == source_.size())
                    return MoEOverlayResidencyWaveProgress::Ready;

                if (!started_)
                {
                    transfer_started_at_ = std::chrono::steady_clock::now();
                    started_ = true;
                }

                const std::size_t bytes = std::min(
                    chunk_bytes_, source_.size() - offset_);
                const auto copy_started_at =
                    std::chrono::steady_clock::now();
                std::memcpy(
                    destination_.data() + offset_,
                    source_.data() + offset_,
                    bytes);
                const auto copy_elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - copy_started_at)
                        .count();
                host_nanoseconds_ = saturatingExpertTierMeasurementAdd(
                    host_nanoseconds_,
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, copy_elapsed)));
                offset_ += bytes;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "cpu_native_copy_bytes",
                    static_cast<double>(bytes),
                    "maintenance",
                    perf_device_);
                if (offset_ != source_.size())
                    return MoEOverlayResidencyWaveProgress::Pending;

                const auto wall_elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - transfer_started_at_)
                        .count();
                measurement_ = {
                    .sequence = 1,
                    .bytes = static_cast<std::uint64_t>(source_.size()),
                    .wall_nanoseconds = static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, wall_elapsed)),
                    .device_nanoseconds = 0,
                    .host_nanoseconds = host_nanoseconds_,
                };
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Stop before the next bounded CPU copy range. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief Host memcpy has no outstanding runtime ownership. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *) noexcept override
            {
                return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                                : MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Return exact bounded host-copy timing after completion. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return offset_ == source_.size() ? measurement_ : std::nullopt;
            }

        private:
            std::span<const std::uint8_t> source_;
            std::span<std::uint8_t> destination_;
            std::size_t chunk_bytes_ = 0;
            std::size_t offset_ = 0;
            std::shared_ptr<ITensorGemm> source_engine_;
            std::shared_ptr<ITensorGemm> destination_engine_;
            std::string perf_device_;
            std::chrono::steady_clock::time_point transfer_started_at_{};
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::uint64_t host_nanoseconds_ = 0;
            std::string failure_;
            bool started_ = false;
            bool aborted_ = false;
        };

        /** @brief Source-independent geometry/provenance expected per layer role. */
        struct ProjectionSignature
        {
            int N = 0;
            int K = 0;
            ExpertWeightFormat format;

            bool operator==(const ProjectionSignature &) const = default;
        };

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
                int expert_id = -1;
                std::uint64_t residency_epoch = 0;
                std::vector<Projection> projections;
            };

            /**
             * @brief Validate and adopt every resident allocation in one layer.
             * @param device Exact endpoint device.
             * @param participant_id Logical endpoint identity.
             * @param layer_idx Transformer layer identity.
             * @param initial_slots Loader-owned resident triplets.
             * @param perf_device Stable topology label for evidence.
             */
            AdoptedInitialExpertSlotRecycler(
                DeviceId device,
                int participant_id,
                int layer_idx,
                std::vector<InitialSlot> initial_slots,
                std::string perf_device)
                : device_(device),
                  participant_id_(participant_id),
                  layer_idx_(layer_idx),
                  perf_device_(std::move(perf_device))
            {
                if ((!device_.is_cpu() && !device_.is_gpu()) ||
                    participant_id_ < 0 || layer_idx_ < 0)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay adopted-slot arena has invalid endpoint identity");
                }

                slots_.reserve(initial_slots.size());
                for (auto &initial : initial_slots)
                {
                    if (initial.expert_id < 0 ||
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
                        .expert_id = initial.expert_id,
                        .residency_epoch = initial.residency_epoch,
                        .bootstrap_assignment = true,
                    });
                }
            }

            /**
             * @brief Acquire a free adopted CPU allocation for a candidate epoch.
             * @return Complete aliasing lease, or no value when none is free.
             */
            [[nodiscard]] std::optional<CpuExpertSlotPool::Lease> acquireCpu(
                int expert_id,
                std::uint64_t residency_epoch)
            {
                if (!device_.is_cpu())
                    return std::nullopt;
                const auto assignment = acquireAssignment(
                    expert_id, residency_epoch);
                if (!assignment)
                    return std::nullopt;

                CpuExpertSlotPool::Lease lease;
                lease.slot_index = assignment->slot_index;
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
                recordAcquisition(assignment->slot_index, residency_epoch);
                return lease;
            }

            /**
             * @brief Acquire a free adopted GPU allocation for a candidate epoch.
             * @return Stable descriptor storage plus assignment lease, or no value.
             */
            [[nodiscard]] std::optional<GpuExpertSlotPool::AcquiredSlot>
            acquireGpu(int expert_id, std::uint64_t residency_epoch)
            {
                if (!device_.is_gpu())
                    return std::nullopt;
                const auto assignment = acquireAssignment(
                    expert_id, residency_epoch);
                if (!assignment)
                    return std::nullopt;

                GpuExpertSlotPool::AcquiredSlot lease;
                lease.slot_index = assignment->slot_index;
                lease.expert_id = expert_id;
                lease.residency_epoch = residency_epoch;
                lease.lifetime = assignment->lifetime;
                const auto &slot = slots_[static_cast<std::size_t>(
                    assignment->slot_index)];
                lease.projections.reserve(slot.projections.size());
                for (const auto &projection : slot.projections)
                    lease.projections.push_back(*projection.gpu_destination);
                recordAcquisition(assignment->slot_index, residency_epoch);
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
                        if (slot.expert_id != expert_id ||
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
                    released_slot,
                    assignment_epoch);
                return true;
            }

        private:
            /** @brief Physical slot plus its current logical assignment. */
            struct Slot
            {
                std::vector<Projection> projections;
                int expert_id = -1;
                std::uint64_t residency_epoch = 0;
                bool bootstrap_assignment = false;
            };

            /** @brief Token whose final alias returns one non-bootstrap slot. */
            struct LeaseToken
            {
                std::weak_ptr<AdoptedInitialExpertSlotRecycler> pool;
                int slot_index = -1;
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
                int expert_id,
                std::uint64_t residency_epoch)
            {
                if (expert_id < 0 || residency_epoch == 0)
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
                    slot.expert_id = expert_id;
                    slot.residency_epoch = residency_epoch;
                    slot.bootstrap_assignment = false;
                }

                auto token = std::shared_ptr<LeaseToken>(
                    new LeaseToken{
                        .pool = weak_from_this(),
                        .slot_index = selected,
                        .expert_id = expert_id,
                        .residency_epoch = residency_epoch,
                    },
                    [](LeaseToken *lease) noexcept
                    {
                        if (lease)
                        {
                            if (auto pool = lease->pool.lock())
                            {
                                pool->releaseLease(
                                    lease->slot_index,
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
                        slot.expert_id != expert_id ||
                        slot.residency_epoch != residency_epoch)
                    {
                        return;
                    }
                    slot.expert_id = -1;
                    slot.residency_epoch = 0;
                    released = true;
                }
                if (released)
                    recordRelease(
                        "adopted_slot_lease_released",
                        slot_index,
                        residency_epoch);
            }

            /** @brief Emit one acquisition proof outside the pool mutex. */
            void recordAcquisition(
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
                     {"layer", std::to_string(layer_idx_)},
                     {"slot", std::to_string(slot_index)},
                     {"epoch", std::to_string(residency_epoch)},
                     {"device", device_.to_string()}});
            }

            /** @brief Emit one exact retirement/release proof. */
            void recordRelease(
                const char *counter,
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
                     {"layer", std::to_string(layer_idx_)},
                     {"slot", std::to_string(slot_index)},
                     {"epoch", std::to_string(residency_epoch)},
                     {"device", device_.to_string()}});
            }

            DeviceId device_ = DeviceId::invalid();
            int participant_id_ = -1;
            int layer_idx_ = -1;
            std::string perf_device_;
            std::vector<Slot> slots_;
            mutable std::mutex mutex_;
        };

        /** @brief CPU or GPU inactive slots for one local endpoint/layer. */
        struct EndpointLayerPool
        {
            DeviceId device = DeviceId::invalid();
            /** Guaranteed inactive overlap capacity charged in the BOM. */
            std::size_t capacity = 0;
            /** Initial live allocations that rotate into free slots on retire. */
            std::shared_ptr<AdoptedInitialExpertSlotRecycler> adopted_slots;
            std::variant<
                std::shared_ptr<CpuExpertSlotPool>,
                std::shared_ptr<GpuExpertSlotPool>>
                pool;
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
                                "ExpertOverlay initial GPU expert lacks the exact reusable allocation charged by capacity admission");
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

        /** @brief Derive one canonical cross-rank identity from the transaction. */
        MoEOverlayRemoteProjectionIdentity remoteProjectionIdentity(
            const MoEOverlayResidencyTransaction &transaction,
            const MoEOverlayResidencyTransactionFingerprint &fingerprint,
            std::size_t migration_index,
            ExpertTierWeightProjection projection)
        {
            const auto &migration = transaction.migrations[migration_index];
            return {
                .expected_epoch = transaction.expected_epoch,
                .candidate_epoch = transaction.candidate->epoch,
                .transaction_fingerprint = fingerprint,
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
        std::map<EndpointLayerKey, EndpointLayerPool> endpoint_pools;
        std::map<EdgeKey, std::shared_ptr<SharedWeightLane>> weight_lanes;
        std::map<EdgeKey, std::shared_ptr<SharedBlobLane>> blob_lanes;
        std::map<EdgeKey, std::shared_ptr<SharedPeerLane>> peer_lanes;
        std::map<
            RemoteGpuLaneKey,
            std::shared_ptr<MoEOverlayGpuRemoteProjectionLane>>
            remote_gpu_lanes;

        std::atomic<std::uint64_t> endpoint_layer_pools{0};
        std::atomic<std::uint64_t> adopted_initial_slots{0};
        std::atomic<std::uint64_t> adopted_initial_slots_recycled{0};
        std::atomic<std::uint64_t> cpu_shadow_slots{0};
        std::atomic<std::uint64_t> gpu_shadow_slots{0};
        std::atomic<std::uint64_t> persistent_transfer_lanes{0};
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
                std::shared_ptr<const MoEOverlayParticipantResidencyBank>>
                initial_banks;

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
                                      : nullptr;
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
             * Pass two allocates every endpoint/layer, including a completely
             * empty cold tier.  Shadow capacity is a runtime wave bound, not a
             * function of current occupancy, so it must remain exact.
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

                for (int layer_idx = 0;
                     layer_idx < endpoint->numLayers();
                     ++layer_idx)
                {
                    const auto signatures = requireLayerSignatures(
                        model_signatures, layer_idx);
                    const auto &initial_layer =
                        bank_found->second->layers[
                            static_cast<std::size_t>(layer_idx)];
                    const std::size_t capacity =
                        config.shadow_slots_per_endpoint_layer;
                    EndpointLayerPool destination;
                    destination.device = endpoint->device();
                    destination.capacity = capacity;
                    auto adopted_initial_slots = makeAdoptedInitialSlots(
                        initial_layer,
                        endpoint->device(),
                        config.initial_snapshot->epoch);
                    const std::size_t adopted_count =
                        adopted_initial_slots.size();
                    destination.adopted_slots = std::make_shared<
                        AdoptedInitialExpertSlotRecycler>(
                        endpoint->device(),
                        participant_id,
                        layer_idx,
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
                                signatures[projectionIndex(projection)];
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
                        auto pool = CpuExpertSlotPool::create({
                            .participant_id = participant_id,
                            .layer_idx = layer_idx,
                            .capacity = static_cast<int>(capacity),
                            .projections = std::move(specs),
                            .memory_placement = memory_placement,
                            .perf_device = config.perf_device,
                        });
                        destination.pool = std::move(pool);
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
                                signatures[projectionIndex(projection)];
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
                            layer_idx,
                            static_cast<int>(capacity),
                            std::move(specs),
                            config.gpu_vram_safety_margin_bytes,
                            /*transfer_capacity=*/0);
                        destination.pool = std::move(pool);
                        impl.gpu_shadow_slots.fetch_add(
                            capacity, std::memory_order_relaxed);
                    }
                    else
                    {
                        throw std::runtime_error(
                            "ExpertOverlay physical fabric encountered an unsupported participant device");
                    }

                    const auto [_, inserted] = impl.endpoint_pools.emplace(
                        EndpointLayerKey{participant_id, layer_idx},
                        std::move(destination));
                    if (!inserted)
                        throw std::logic_error(
                            "ExpertOverlay physical fabric created a duplicate endpoint pool");
                    impl.endpoint_layer_pools.fetch_add(
                        1, std::memory_order_relaxed);
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

        /** @brief Materialize each unique local directed GPU edge exactly once. */
        void materializeTransferLanes(
            const MoEOverlayPhysicalResidencyFabric::Config &config,
            MoEOverlayPhysicalResidencyFabric::Impl &impl)
        {
            const auto local_ids = config.registry->localParticipantIds();
            for (const int source_id : local_ids)
            {
                const auto source = config.registry->endpoint(source_id);
                if (!source)
                    throw std::logic_error(
                        "ExpertOverlay lane materialization lost a source endpoint");
                for (const int destination_id : local_ids)
                {
                    if (source_id == destination_id)
                        continue;
                    const auto destination =
                        config.registry->endpoint(destination_id);
                    if (!destination)
                        throw std::logic_error(
                            "ExpertOverlay lane materialization lost a destination endpoint");
                    if (source->device().is_cpu() &&
                        destination->device().is_cpu())
                    {
                        continue;
                    }

                    for (const auto projection : kProjections)
                    {
                        const EdgeKey key{
                            .source = source->device(),
                            .destination = destination->device(),
                            .projection = projection,
                        };
                        const std::string name = edgeLaneName(key);
                        std::string error;
                        if (source->device().is_gpu() &&
                            destination->device().is_gpu())
                        {
                            if (source->device().type ==
                                destination->device().type)
                            {
                                if (impl.peer_lanes.count(key) != 0)
                                    continue;
                                auto lane = std::make_shared<
                                    ExpertTierGpuPeerTransferLane>(
                                    ExpertTierGpuPeerTransferLane::Config{
                                        .source_device = source->device(),
                                        .destination_device =
                                            destination->device(),
                                        .lane_name = name,
                                        .perf_device = config.perf_device,
                                        .collect_timing_measurements =
                                            config.collect_economy_measurements,
                                    });
                                if (!lane->materialize(&error))
                                    throw std::runtime_error(error);
                                impl.peer_lanes.emplace(
                                    key,
                                    std::make_shared<SharedPeerLane>(
                                        std::move(lane)));
                            }
                            else
                            {
                                if (impl.blob_lanes.count(key) != 0)
                                    continue;
                                auto lane = std::make_shared<
                                    ExpertTierGpuBlobTransferLane>(
                                    ExpertTierGpuBlobTransferLane::Config{
                                        .source_device = source->device(),
                                        .destination_device =
                                            destination->device(),
                                        .staging_capacity_bytes =
                                            config.staging_capacity_bytes,
                                        .lane_name = name,
                                        .perf_device = config.perf_device,
                                        .collect_timing_measurements =
                                            config.collect_economy_measurements,
                                    });
                                if (!lane->materialize(&error))
                                    throw std::runtime_error(error);
                                impl.blob_lanes.emplace(
                                    key,
                                    std::make_shared<SharedBlobLane>(
                                        std::move(lane)));
                            }
                        }
                        else
                        {
                            if (impl.weight_lanes.count(key) != 0)
                                continue;
                            const DeviceId gpu = source->device().is_gpu()
                                                     ? source->device()
                                                     : destination->device();
                            auto lane = std::make_shared<
                                ExpertTierWeightTransferLane>(
                                ExpertTierWeightTransferLane::Config{
                                    .device = gpu,
                                    .staging_capacity_bytes =
                                        config.staging_capacity_bytes,
                                    .lane_name = name,
                                    .perf_device = config.perf_device,
                                    .collect_timing_measurements =
                                        config.collect_economy_measurements,
                                });
                            if (!lane->materialize(&error))
                                throw std::runtime_error(error);
                            impl.weight_lanes.emplace(
                                key,
                                std::make_shared<SharedWeightLane>(
                                    std::move(lane)));
                        }
                        impl.persistent_transfer_lanes.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }

            if (!config.remote_projection_transport)
                return;

            /*
             * Cross-rank edges are not known until histogram-driven placement
             * chooses a transaction. Materialize one fair source and one fair
             * destination lane for each local GPU/projection now, independent
             * of which remote rank or device becomes the other endpoint.
             */
            for (const int participant_id : local_ids)
            {
                const auto endpoint = config.registry->endpoint(participant_id);
                if (!endpoint || !endpoint->device().is_gpu())
                    continue;
                for (const auto projection : kProjections)
                {
                    for (const auto role : {
                             RemoteGpuLaneRole::Source,
                             RemoteGpuLaneRole::Destination})
                    {
                        const RemoteGpuLaneKey key{
                            .device = endpoint->device(),
                            .projection = projection,
                            .role = role,
                        };
                        if (impl.remote_gpu_lanes.count(key) != 0)
                            continue;
                        const std::string role_name =
                            role == RemoteGpuLaneRole::Source
                                ? "remote_source"
                                : "remote_destination";
                        auto lane = std::make_shared<
                            MoEOverlayGpuRemoteProjectionLane>(
                            MoEOverlayGpuRemoteProjectionLane::Config{
                                .device = endpoint->device(),
                                .staging_capacity_bytes =
                                    config.staging_capacity_bytes,
                                .lane_name = endpoint->device().to_string() +
                                             "_" + role_name + "_" +
                                             projectionName(projection),
                                .perf_device = config.perf_device,
                            });
                        std::string error;
                        if (!lane->materialize(&error))
                            throw std::runtime_error(error);
                        impl.remote_gpu_lanes.emplace(key, std::move(lane));
                        impl.persistent_transfer_lanes.fetch_add(
                            1, std::memory_order_relaxed);
                    }
                }
            }
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
            EndpointLayerPool &pool,
            int expert_id,
            std::uint64_t candidate_epoch)
        {
            ReservedDestination reserved;
            if (pool.device.is_cpu())
            {
                if (pool.adopted_slots)
                {
                    reserved.cpu = pool.adopted_slots->acquireCpu(
                        expert_id, candidate_epoch);
                    if (reserved.cpu)
                        return reserved;
                }
                auto cpu_pool =
                    std::get<std::shared_ptr<CpuExpertSlotPool>>(pool.pool);
                reserved.cpu = cpu_pool->acquire(
                    expert_id, candidate_epoch);
                if (!reserved.cpu)
                    throw std::runtime_error(
                        "ExpertOverlay CPU shadow-slot preflight changed before reservation");
            }
            else
            {
                if (pool.adopted_slots)
                {
                    reserved.gpu = pool.adopted_slots->acquireGpu(
                        expert_id, candidate_epoch);
                    if (reserved.gpu)
                        return reserved;
                }
                auto gpu_pool =
                    std::get<std::shared_ptr<GpuExpertSlotPool>>(pool.pool);
                reserved.gpu = gpu_pool->acquire(
                    expert_id, candidate_epoch);
                if (!reserved.gpu)
                    throw std::runtime_error(
                        "ExpertOverlay GPU shadow-slot preflight changed before reservation");
            }
            return reserved;
        }

        /** @brief Return currently available slots through the typed pool. */
        std::size_t availableSlots(const EndpointLayerPool &pool)
        {
            const std::size_t adopted = pool.adopted_slots
                                            ? pool.adopted_slots
                                                  ->availableSlots()
                                            : 0;
            if (pool.device.is_cpu())
            {
                return adopted +
                    std::get<std::shared_ptr<CpuExpertSlotPool>>(pool.pool)
                        ->availableSlots();
            }
            return adopted +
                std::get<std::shared_ptr<GpuExpertSlotPool>>(pool.pool)
                    ->availableSlots();
        }
    } // namespace

    std::shared_ptr<MoEOverlayPhysicalResidencyFabric>
    MoEOverlayPhysicalResidencyFabric::create(Config config)
    {
        if (!config.registry || !config.initial_snapshot ||
            !config.initial_snapshot->valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires registry and valid initial snapshot");
        }
        if (config.registry->initialEpoch() !=
                config.initial_snapshot->epoch ||
            !config.registry->allInitialBanksReady())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires complete matching initial banks");
        }
        if (config.shadow_slots_per_endpoint_layer == 0 ||
            config.staging_capacity_bytes == 0)
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires positive shadow and staging capacities");
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
             {"adopted_initial_slots",
              std::to_string(
                  impl->adopted_initial_slots.load(
                      std::memory_order_relaxed))},
             {"persistent_lanes",
              std::to_string(
                  impl->persistent_transfer_lanes.load(
                      std::memory_order_relaxed))}});
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
        MoEOverlayParticipantPreparedTransfers result;
        auto fail = [&](std::string message)
            -> MoEOverlayParticipantPreparedTransfers
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            result.status = MoEOverlayResidencyStageStartStatus::Failed;
            result.error = std::move(message);
            return std::move(result);
        };

        if (!transaction.valid() || transaction.empty() || !transaction.candidate)
            return fail(
                "ExpertOverlay physical fabric requires a valid non-empty transaction");

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
        for (const auto &migration : transaction.migrations)
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

        /*
         * Validate the immutable BOM and transient availability for the whole
         * cycle set before acquiring one slot.  Exceeding the model-time BOM is
         * fatal; an old ticket retaining a planned slot is ordinary deferral.
         */
        for (const auto &requirement : transaction.shadow_requirements)
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
            if (requirement.slot_count > pool->capacity)
            {
                return fail(
                    "ExpertOverlay migration wave exceeds the preallocated shadow-slot BOM");
            }
            const std::size_t available = availableSlots(*pool);
            if (requirement.slot_count > available)
            {
                impl_->waves_deferred.fetch_add(
                    1, std::memory_order_relaxed);
                result.status = MoEOverlayResidencyStageStartStatus::Deferred;
                result.error =
                    "ExpertOverlay shadow slots are retained by an older epoch: "
                    "participant=" +
                    std::to_string(requirement.destination_participant) +
                    " layer=" + std::to_string(requirement.layer_idx) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " available=" + std::to_string(available) +
                    " planned_capacity=" + std::to_string(pool->capacity) +
                    " expected_epoch=" +
                    std::to_string(transaction.expected_epoch) +
                    " candidate_epoch=" +
                    std::to_string(transaction.candidate->epoch);
                return result;
            }
        }

        std::vector<ReservedDestination> destinations(
            transaction.migrations.size());
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < transaction.migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    transaction.migrations[migration_index];
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
                    migration.expert_id,
                    transaction.candidate->epoch);
            }
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        result.migrations.resize(transaction.migrations.size());
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
                transaction.migrations.size() * kProjections.size();
            remote_bindings.reserve(remote_projection_count);
            remote_targets.reserve(remote_projection_count);
        }

        const auto transaction_fingerprint = remote_transport
            ? std::optional<MoEOverlayResidencyTransactionFingerprint>(
                  fingerprintMoEOverlayResidencyTransaction(transaction))
            : std::nullopt;
        std::uint64_t local_cpu_copy_operations = 0;
        std::uint64_t local_remote_cpu_operations = 0;
        std::uint64_t local_remote_gpu_cpu_operations = 0;
        std::uint64_t local_remote_gpu_blob_operations = 0;
        std::uint64_t local_gpu_cpu_operations = 0;
        std::uint64_t local_same_backend_gpu_operations = 0;
        std::uint64_t local_heterogeneous_gpu_operations = 0;
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < transaction.migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    transaction.migrations[migration_index];
                auto &prepared = result.migrations[migration_index];
                const bool source_local = is_local_participant(
                    migration.source.owner_participant);
                const bool destination_local = is_local_participant(
                    migration.destination.owner_participant);
                const bool crosses_world_rank = remote_transport &&
                    migration.source.owner_world_rank !=
                        migration.destination.owner_world_rank;

                std::shared_ptr<MoEOverlayParticipantResidency> source_endpoint;
                std::shared_ptr<const MoEOverlayParticipantResidencyBank>
                    source_bank;
                const MoEOverlayPreparedExpertTriplet *source_triplet = nullptr;
                if (source_local)
                {
                    source_endpoint = config_.registry->endpoint(
                        migration.source.owner_participant);
                    source_bank = source_endpoint
                                      ? source_endpoint->acquire(
                                            transaction.previous->epoch)
                                      : nullptr;
                    if (!source_endpoint || !source_bank ||
                        migration.layer_idx < 0 ||
                        migration.layer_idx >= source_endpoint->numLayers() ||
                        migration.expert_id < 0 ||
                        migration.expert_id >= source_endpoint->numExperts())
                    {
                        throw std::runtime_error(
                            "ExpertOverlay physical fabric cannot acquire the exact local source epoch");
                    }
                    source_triplet = &source_bank
                                          ->layers[static_cast<std::size_t>(
                                              migration.layer_idx)]
                                          .experts[static_cast<std::size_t>(
                                              migration.expert_id)];
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
                        if (!transaction_fingerprint)
                            throw std::logic_error(
                                "ExpertOverlay remote projection lost transaction fingerprint ownership");

                        const auto identity = remoteProjectionIdentity(
                            transaction,
                            *transaction_fingerprint,
                            migration_index,
                            projection);
                        MoEOverlayMPIRemoteProjectionBinding binding{
                            .lane_index =
                                migration_index * kProjections.size() +
                                projection_index,
                            .identity = identity,
                            .purpose = transaction.purpose,
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
                                if (!source.cpu_packed)
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU source lost final NativeVNNI storage");
                                if (migration.destination.device.is_cpu())
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
                                            transaction.candidate->epoch,
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
                                            transaction.candidate->epoch,
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
                                if (source.format.isFloating())
                                {
                                    throw std::runtime_error(
                                        "Remote ExpertOverlay floating GPU source protocol is not yet materialized");
                                }
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

                                MoEOverlayRemoteProjectionManifest manifest;
                                if (migration.destination.device.is_cpu())
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
                                            transaction.previous->epoch,
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
                                            transaction.previous->epoch,
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
                                binding.source = std::make_shared<
                                    MoEOverlayGpuRemoteProjectionSource>(
                                    std::move(manifest),
                                    lane->second,
                                    source.gpu_packed,
                                    ExpertTierSourceReadiness::
                                        publishedResidencyBank(
                                            transaction.previous->epoch),
                                    source_engine);
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
                                const auto *destination_packed =
                                    destination_engine
                                        ->exportCPUNativeVNNIPackedWeights();
                                if (!destination_packed)
                                {
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU destination cannot export final storage metadata");
                                }

                                MoEOverlayRemoteProjectionManifest
                                    expected_manifest;
                                auto match_policy =
                                    MoEOverlayRemoteProjectionManifestMatchPolicy::
                                        Exact;
                                if (migration.source.device.is_cpu())
                                {
                                    expected_manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            *destination_packed,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
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
                                            transaction.previous->epoch,
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
                                            transaction.previous->epoch,
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
                                    ++local_remote_gpu_cpu_operations;
                                }

                                auto storage = std::make_shared<
                                    MoEOverlayHostRemoteProjectionDestination>(
                                    identity,
                                    remoteCpuDestinationRegions(
                                        destination.destination_bytes),
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
                                if (destination_slot.spec.format.isFloating())
                                {
                                    throw std::runtime_error(
                                        "Remote ExpertOverlay floating GPU destination protocol is not yet materialized");
                                }
                                const auto source_identity =
                                    destination_slot.spec.format.native_vnni;
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

                                MoEOverlayGpuRemoteProjectionDestinationFactory
                                    factory =
                                        [destination_slot,
                                         destination_device,
                                         slot_lifetime,
                                         source_identity](
                                            const MoEOverlayRemoteProjectionManifest &
                                                manifest,
                                            MoEOverlayGpuRemoteProjectionDestinationBinding *
                                                output,
                                            std::string *error) -> bool
                                {
                                    if (!output ||
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
                                    lane->second,
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
                                    std::make_unique<QueuedCpuCopyOperation>(
                                        std::span<const std::uint8_t>(
                                            static_cast<const std::uint8_t *>(
                                                source.floating.data),
                                            source.floating.bytes),
                                        destination.destination_bytes,
                                        config_.staging_capacity_bytes,
                                        source_engine,
                                        destination_engine,
                                        config_.perf_device);
                                ++local_cpu_copy_operations;
                            }
                            else
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay GPU-to-CPU floating transfer lane is not yet materialized");
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
                                std::make_unique<QueuedCpuCopyOperation>(
                                    std::span<const std::uint8_t>(
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size()),
                                    destination.destination_bytes,
                                    config_.staging_capacity_bytes,
                                    source_engine,
                                    destination_engine,
                                    config_.perf_device);
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
                                    transaction.previous->epoch,
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
                                    transaction.previous->epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    maximum_units);
                            const auto lane = impl_->weight_lanes.find(edge);
                            if (lane == impl_->weight_lanes.end())
                                throw std::runtime_error(
                                    "ExpertOverlay GPU-to-CPU edge has no persistent lane");
                            physical_operation =
                                std::make_unique<QueuedWeightOperation>(
                                    lane->second,
                                    QueuedWeightOperation::Direction::GpuToCpu,
                                    manifest.deviceLayout(),
                                    gpuConstView(source.gpu_packed),
                                    destination.destination_bytes,
                                    std::span<const std::uint8_t>{},
                                    ExpertTierGpuMutableProjectionView{},
                                    ExpertTierSourceReadiness::publishedResidencyBank(
                                        transaction.previous->epoch),
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
                                throw std::runtime_error(
                                    "ExpertOverlay CPU-to-GPU floating transfer lane is not yet materialized");
                            }
                            if (migration.source.device.type ==
                                migration.destination.device.type)
                            {
                                const auto lane = impl_->peer_lanes.find(edge);
                                if (lane == impl_->peer_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay same-backend floating GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<QueuedPeerOperation>(
                                        lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            transaction.previous->epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto lane = impl_->blob_lanes.find(edge);
                                if (lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay heterogeneous floating GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<QueuedBlobOperation>(
                                        lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            transaction.previous->epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_heterogeneous_gpu_operations;
                            }
                        }
                        else if (migration.source.device.is_cpu())
                        {
                            const auto probe =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    *source.cpu_packed,
                                    transaction.candidate->epoch,
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
                                    transaction.candidate->epoch,
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
                                std::make_unique<QueuedWeightOperation>(
                                    lane->second,
                                    QueuedWeightOperation::Direction::CpuToGpu,
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
                                        transaction.previous->epoch),
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

                            if (migration.source.device.type ==
                                migration.destination.device.type)
                            {
                                const auto lane = impl_->peer_lanes.find(edge);
                                if (lane == impl_->peer_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay same-backend GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<QueuedPeerOperation>(
                                        lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            transaction.previous->epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto lane = impl_->blob_lanes.find(edge);
                                if (lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay heterogeneous GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<QueuedBlobOperation>(
                                        lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            transaction.previous->epoch),
                                        source_engine,
                                        destination_engine);
                                ++local_heterogeneous_gpu_operations;
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
            transaction.migrations.size() * kProjections.size(),
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
            {{"migrations", std::to_string(transaction.migrations.size())},
             {"projections",
              std::to_string(
                  transaction.migrations.size() * kProjections.size())},
             {"remote_cpu_endpoints",
              std::to_string(local_remote_cpu_operations)},
             {"remote_gpu_cpu_endpoints",
              std::to_string(local_remote_gpu_cpu_operations)},
             {"remote_gpu_blob_endpoints",
              std::to_string(local_remote_gpu_blob_operations)}});
        return result;
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
                !found->second.adopted_slots)
            {
                /* Local topology disappearing after publication is unrecoverable. */
                std::terminate();
            }
            if (found->second.adopted_slots->retireBootstrapAssignment(
                    migration.expert_id, retired_epoch))
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

    MoEOverlayPhysicalResidencyFabricStats
    MoEOverlayPhysicalResidencyFabric::stats() const noexcept
    {
        return {
            .endpoint_layer_pools =
                impl_->endpoint_layer_pools.load(std::memory_order_relaxed),
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
