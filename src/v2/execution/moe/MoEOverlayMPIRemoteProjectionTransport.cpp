/**
 * @file MoEOverlayMPIRemoteProjectionTransport.cpp
 * @brief Event-free CPU endpoints and non-blocking MPI projection progression.
 */

#include "MoEOverlayMPIRemoteProjectionTransport.h"

#include "MoEOverlayMPIFatal.h"

#include "collective/CollectiveTimeoutPolicy.h"
#include "interfaces/IMPIContext.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Store one optional caller-facing diagnostic. */
        void setError(std::string *error, std::string message)
        {
            if (error)
                *error = std::move(message);
        }

        /** @brief Convert an MPI status code into a stable operation diagnostic. */
        std::string mpiError(const char *operation, int mpi_error)
        {
            char buffer[MPI_MAX_ERROR_STRING]{};
            int length = 0;
            const int describe_result =
                MPI_Error_string(mpi_error, buffer, &length);
            std::ostringstream message;
            message << "ExpertOverlay remote projection " << operation
                    << " failed with MPI code " << mpi_error;
            if (describe_result == MPI_SUCCESS && length > 0)
                message << ": " << std::string(buffer, buffer + length);
            return message.str();
        }

        /** @brief Return an exact mutable region by authenticated index. */
        std::span<std::uint8_t> mutableRegion(
            std::array<
                std::span<std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> &regions,
            std::uint8_t region) noexcept
        {
            if (region >= regions.size())
                return {};
            return regions[region];
        }

        /** @brief Render typed transaction intent as a stable PerfStats tag. */
        const char *transactionPurposeName(
            MoEOverlayResidencyTransactionPurpose purpose) noexcept
        {
            switch (purpose)
            {
            case MoEOverlayResidencyTransactionPurpose::PlacementChange:
                return "placement_change";
            case MoEOverlayResidencyTransactionPurpose::EconomyCalibration:
                return "economy_calibration";
            case MoEOverlayResidencyTransactionPurpose::
                PreparedContextRestoration:
                return "prepared_context_restoration";
            }
            return "invalid";
        }

        /** @brief Reject enum corruption before a lane can own MPI work. */
        bool validTransactionPurpose(
            MoEOverlayResidencyTransactionPurpose purpose) noexcept
        {
            return purpose ==
                       MoEOverlayResidencyTransactionPurpose::PlacementChange ||
                   purpose ==
                       MoEOverlayResidencyTransactionPurpose::EconomyCalibration ||
                   purpose == MoEOverlayResidencyTransactionPurpose::
                                  PreparedContextRestoration;
        }

        /**
         * @brief Operation used by ranks with no endpoint in one global move.
         *
         * It remains a real operation so every rank's composite wave has the
         * globally fixed `migration_count * 3` shape.
         */
        class UninvolvedRemoteProjectionOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** @brief Report immediate readiness after reservation consensus. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                return aborted_ ? MoEOverlayResidencyWaveProgress::Failed
                                : MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Mark the no-op discarded. */
            void abort() noexcept override { aborted_ = true; }

            /** @brief No physical resource needs asynchronous reclamation. */
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
    } // namespace

    MoEOverlayHostRemoteProjectionSource::
        MoEOverlayHostRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::array<
                std::span<const std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions,
            std::shared_ptr<void> lifetime)
        : manifest_(std::move(manifest)),
          cursor_(manifest_, regions),
          lifetime_(std::move(lifetime))
    {
        if (!lifetime_)
            throw std::invalid_argument(
                "Host remote projection source requires retained slot ownership");
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayHostRemoteProjectionSource::pollNextChunk(
        MoEOverlayRemoteProjectionChunkView *chunk,
        std::string *error) noexcept
    {
        if (!chunk || aborted_ || outstanding_header_)
        {
            setError(
                error,
                aborted_
                    ? "Host remote projection source is aborted"
                    : outstanding_header_
                          ? "Host remote projection source payload is still owned by MPI"
                    : "Host remote projection source requires a chunk destination");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        const auto next = cursor_.takeNext();
        if (!next)
        {
            setError(
                error,
                "Host remote projection source was polled beyond its final chunk");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        *chunk = *next;
        outstanding_header_ = next->header;
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    bool MoEOverlayHostRemoteProjectionSource::acknowledgeChunkSent(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::string *error) noexcept
    {
        if (aborted_ || !outstanding_header_ ||
            header.manifest_hash != outstanding_header_->manifest_hash ||
            header.payload_hash != outstanding_header_->payload_hash ||
            header.sequence != outstanding_header_->sequence ||
            header.region != outstanding_header_->region ||
            header.region_offset != outstanding_header_->region_offset ||
            header.payload_bytes != outstanding_header_->payload_bytes ||
            header.final_chunk != outstanding_header_->final_chunk)
        {
            setError(
                error,
                aborted_
                    ? "Host remote projection source is aborted"
                    : "Host remote projection source received an unexpected MPI payload acknowledgement");
            return false;
        }
        outstanding_header_.reset();
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayHostRemoteProjectionSource::pollAbort(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                        : MoEOverlayResidencyWaveProgress::Failed;
    }

    MoEOverlayHostRemoteProjectionDestination::
        MoEOverlayHostRemoteProjectionDestination(
            MoEOverlayRemoteProjectionIdentity expected_identity,
            std::array<
                std::span<std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions,
            std::shared_ptr<void> lifetime,
            std::optional<MoEOverlayRemoteProjectionManifest>
                expected_manifest,
            MoEOverlayRemoteProjectionManifestMatchPolicy match_policy)
        : expected_identity_(expected_identity),
          regions_(regions),
          lifetime_(std::move(lifetime)),
          expected_manifest_(std::move(expected_manifest)),
          match_policy_(match_policy)
    {
        if (!expected_identity_.valid() || !lifetime_)
            throw std::invalid_argument(
                "Host remote projection destination requires identity and slot ownership");
        std::string manifest_error;
        if (expected_manifest_ &&
            (!expected_manifest_->valid(&manifest_error) ||
             expected_manifest_->identity != expected_identity_))
        {
            throw std::invalid_argument(
                manifest_error.empty()
                    ? "Host remote projection destination expected manifest disagrees with identity"
                    : std::move(manifest_error));
        }
        if (match_policy_ ==
                MoEOverlayRemoteProjectionManifestMatchPolicy::
                    EquivalentGpuSourceForFinalCpuStorage &&
            (!expected_manifest_ || !expected_manifest_->carriesCpuBytes() ||
             !expected_identity_.source_device.is_gpu() ||
             !expected_identity_.destination_device.is_cpu()))
        {
            throw std::invalid_argument(
                "Equivalent remote GPU source matching is valid only for a GPU-to-CPU final-storage contract");
        }
    }

    bool MoEOverlayHostRemoteProjectionDestination::beginManifest(
        const MoEOverlayRemoteProjectionManifest &manifest,
        std::string *error) noexcept
    {
        if (aborted_ || validator_ || !manifest.valid(error) ||
            manifest.identity != expected_identity_)
        {
            if (aborted_)
                setError(error, "Host remote projection destination is aborted");
            else if (validator_)
                setError(error, "Host remote projection manifest was already bound");
            else if (manifest.identity != expected_identity_)
                setError(error, "Host remote projection manifest identity is unexpected");
            return false;
        }
        bool manifest_matches = true;
        if (expected_manifest_)
        {
            if (match_policy_ ==
                MoEOverlayRemoteProjectionManifestMatchPolicy::Exact)
            {
                manifest_matches = manifest == *expected_manifest_;
            }
            else
            {
                /*
                 * Both manifests have independently passed full validation.
                 * Substitute only the four live GPU representation fields and
                 * digest in the expected value; equality then proves every CPU
                 * storage, topology, geometry, capacity, and provenance field.
                 */
                auto equivalent = *expected_manifest_;
                equivalent.gpu_codebook_id = manifest.gpu_codebook_id;
                equivalent.gpu_payload_bytes_per_block =
                    manifest.gpu_payload_bytes_per_block;
                equivalent.gpu_is_asymmetric = manifest.gpu_is_asymmetric;
                equivalent.gpu_has_emins = manifest.gpu_has_emins;
                equivalent.manifest_hash = manifest.manifest_hash;
                manifest_matches = equivalent == manifest;
            }
        }
        if (!manifest_matches)
        {
            setError(
                error,
                "Host remote projection physical format differs from destination contract");
            return false;
        }
        for (std::size_t region = 0; region < regions_.size(); ++region)
        {
            if (regions_[region].size() != manifest.region_bytes[region])
            {
                setError(
                    error,
                    "Host remote projection final regions disagree with sender manifest");
                return false;
            }
        }
        try
        {
            validator_.emplace(manifest);
        }
        catch (const std::exception &exception)
        {
            setError(error, exception.what());
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayHostRemoteProjectionDestination::beginChunk(
        const MoEOverlayRemoteProjectionChunkHeader &header,
        std::span<const std::uint8_t> payload,
        std::string *error) noexcept
    {
        if (aborted_ || !validator_ ||
            !validator_->accept(header, payload, error))
        {
            if (aborted_)
                setError(error, "Host remote projection destination is aborted");
            else if (!validator_)
                setError(error, "Host remote projection has no accepted manifest");
            return MoEOverlayResidencyWaveProgress::Failed;
        }

        auto destination = mutableRegion(regions_, header.region);
        if (header.region_offset > destination.size() ||
            payload.size() > destination.size() -
                                 static_cast<std::size_t>(
                                     header.region_offset))
        {
            validator_->abort();
            setError(
                error,
                "Host remote projection validated a range outside final storage");
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        /* Authentication above is complete before final execution bytes mutate. */
        std::memcpy(
            destination.data() +
                static_cast<std::size_t>(header.region_offset),
            payload.data(),
            payload.size());
        if (error)
            error->clear();
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayHostRemoteProjectionDestination::pollChunk(
        std::string *error) noexcept
    {
        setError(
            error,
            "Host remote projection commits synchronously and must not be polled");
        return MoEOverlayResidencyWaveProgress::Failed;
    }

    bool MoEOverlayHostRemoteProjectionDestination::complete() const noexcept
    {
        return validator_ && validator_->complete();
    }

    bool MoEOverlayHostRemoteProjectionDestination::publishFinal(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (complete())
            return true;
        setError(
            error,
            "Host remote projection cannot publish before final-byte readiness");
        return false;
    }

    void MoEOverlayHostRemoteProjectionDestination::abort() noexcept
    {
        aborted_ = true;
        if (validator_)
            validator_->abort();
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayHostRemoteProjectionDestination::pollAbort(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        return aborted_ ? MoEOverlayResidencyWaveProgress::Ready
                        : MoEOverlayResidencyWaveProgress::Failed;
    }

    /** @brief Atomic counters shared with operations that outlive the facade. */
    struct MoEOverlayMPIRemoteProjectionTransportSharedStats
    {
        std::atomic<std::uint64_t> lanes_materialized{0};
        std::atomic<std::uint64_t> wave_reservations_started{0};
        std::atomic<std::uint64_t> wave_reservations_deferred{0};
        std::atomic<std::uint64_t> wave_reservations_failed{0};
        std::atomic<std::uint64_t> reservations_started{0};
        std::atomic<std::uint64_t> reservations_deferred{0};
        std::atomic<std::uint64_t> source_operations{0};
        std::atomic<std::uint64_t> destination_operations{0};
        std::atomic<std::uint64_t> uninvolved_operations{0};
        std::atomic<std::uint64_t> manifests_sent{0};
        std::atomic<std::uint64_t> manifests_received{0};
        std::atomic<std::uint64_t> chunks_sent{0};
        std::atomic<std::uint64_t> chunks_received{0};
        std::atomic<std::uint64_t> bytes_sent{0};
        std::atomic<std::uint64_t> bytes_received{0};
        std::atomic<std::uint64_t> pending_mpi_polls{0};
        std::atomic<std::uint64_t> pending_endpoint_polls{0};
        std::atomic<std::uint64_t> active_mpi_requests{0};
        std::atomic<std::uint64_t> maximum_concurrent_mpi_requests{0};
        std::atomic<std::uint64_t> operations_completed{0};
        std::atomic<std::uint64_t> operations_aborted{0};
        std::atomic<std::uint64_t> mpi_failures{0};
        std::atomic<std::uint64_t> protocol_failures{0};
    };

    /** @brief One model-time payload buffer and single-owner reservation bit. */
    struct MoEOverlayMPIRemoteProjectionLaneStorage
    {
        explicit MoEOverlayMPIRemoteProjectionLaneStorage(
            std::size_t staging_capacity)
            : payload(staging_capacity)
        {
        }

        std::mutex mutex;
        bool occupied = false;
        std::array<
            std::uint8_t,
            MoEOverlayRemoteProjectionManifest::kWireBytes>
            manifest_packet{};
        std::array<
            std::uint8_t,
            MoEOverlayRemoteProjectionChunkHeader::kWireBytes>
            header_packet{};
        std::vector<std::uint8_t> payload;
    };

    struct MoEOverlayMPIRemoteProjectionTransport::Impl
    {
        MPI_Comm communicator = MPI_COMM_NULL;
        int world_rank = -1;
        int world_size = 0;
        int tag_upper_bound = -1;
        std::size_t staging_capacity = 0;
        std::string perf_device;
        std::vector<std::unique_ptr<
            MoEOverlayMPIRemoteProjectionLaneStorage>> lanes;
        std::shared_ptr<
            MoEOverlayMPIRemoteProjectionTransportSharedStats> stats =
            std::make_shared<
                MoEOverlayMPIRemoteProjectionTransportSharedStats>();

        /** @brief Export one low-frequency proof event. */
        void recordCounter(
            const char *name,
            double value = 1.0,
            PerfStatsCollector::Tags extra_tags = {}) const
        {
            auto tags = extra_tags;
            tags.emplace("world_rank", std::to_string(world_rank));
            tags.emplace("world_size", std::to_string(world_size));
            tags.emplace("background", "true");
            tags.emplace("blocking", "false");
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                name,
                value,
                "maintenance",
                perf_device,
                tags);
        }

        /** @brief Release exactly one reserved lane after MPI is quiescent. */
        void releaseLane(std::size_t lane_index) noexcept
        {
            auto &lane = *lanes[lane_index];
            std::lock_guard<std::mutex> lock(lane.mutex);
            if (!lane.occupied)
                std::terminate();
            lane.occupied = false;
        }

        /** @return Whether every model-time lane is unreserved. */
        [[nodiscard]] bool allLanesIdle() const noexcept
        {
            for (const auto &lane : lanes)
            {
                std::lock_guard<std::mutex> lock(lane->mutex);
                if (lane->occupied)
                    return false;
            }
            return true;
        }

        /**
         * @brief Owned state machine for one source, destination, or relay rank.
         *
         * Exactly one MPI request is active at a time.  This conservative first
         * implementation makes message ordering trivial and supplies the
         * correctness oracle; double-buffered GPU endpoint overlap is installed
         * behind the same interface and measured before production admission.
         */
        class NetworkOperation final
            : public IMoEOverlayTierTransferOperation
        {
        public:
            /** Role of this MPI rank in the point-to-point projection edge. */
            enum class Role
            {
                Source,
                Destination,
            };

            /** Fine-grained request/endpoint lifecycle. */
            enum class State
            {
                Bound,
                SendingManifest,
                PreparingSourceChunk,
                SendingHeader,
                SendingPayload,
                ReceivingManifest,
                ReceivingHeader,
                ReceivingPayload,
                CommittingDestinationChunk,
                Ready,
                Failed,
                Aborting,
                AbortReady,
            };

            /**
             * @brief Build an inert operation before atomic lane admission.
             *
             * Construction owns endpoints but not the lane. @ref
             * adoptReservedLane is called only after every lane in the wave has
             * been proved free and marked occupied under its mutex.
             */
            NetworkOperation(
                std::shared_ptr<Impl> impl,
                std::size_t lane_index,
                MoEOverlayRemoteProjectionIdentity identity,
                MoEOverlayResidencyTransactionPurpose purpose,
                Role role,
                std::shared_ptr<
                    IMoEOverlayRemoteProjectionSourceEndpoint> source,
                std::shared_ptr<
                    IMoEOverlayRemoteProjectionDestinationEndpoint>
                    destination)
                : impl_(std::move(impl)),
                  lane_index_(lane_index),
                  identity_(identity),
                  purpose_(purpose),
                  role_(role),
                  source_(std::move(source)),
                  destination_(std::move(destination))
            {
            }

            /** @brief Adopt the lane already marked occupied by batch admission. */
            void adoptReservedLane() noexcept
            {
                if (state_ != State::Bound || !lane_released_)
                    std::terminate();
                lane_released_ = false;
                lane_admitted_at_ = std::chrono::steady_clock::now();
            }

            /** @brief Enforce explicit quiescence and lane release. */
            ~NetworkOperation() override
            {
                if (request_ != MPI_REQUEST_NULL || request_counted_ ||
                    !lane_released_)
                {
                    LOG_ERROR(
                        "[MoEOverlayMPIRemoteProjectionTransport] Operation destroyed before MPI/lane quiescence"
                        << " lane=" << lane_index_
                        << " state=" << static_cast<int>(state_));
                    std::terminate();
                }
            }

            /** @brief Begin or progress exactly one non-blocking state edge. */
            MoEOverlayResidencyWaveProgress poll(
                std::string *error) noexcept override
            {
                if (lane_released_)
                {
                    return failProtocol(
                        "Remote projection was polled before atomic lane admission",
                        error);
                }
                if (state_ == State::Ready)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (state_ == State::Failed || state_ == State::Aborting ||
                    state_ == State::AbortReady)
                {
                    setError(
                        error,
                        failure_.empty()
                            ? "Remote projection operation is not pollable"
                            : failure_);
                    return MoEOverlayResidencyWaveProgress::Failed;
                }

                if (state_ == State::Bound)
                {
                    /*
                     * Reservation may deliberately precede dispatch while a
                     * calibration probe waits for its exact live-inference
                     * ticket or distributed ranks vote on capacity.  Start
                     * transfer timing only when maintenance releases bytes;
                     * retaining admission delay separately prevents policy
                     * economics and transport diagnostics from pricing idle
                     * ownership as UPI or network work.
                     */
                    operation_started_at_ =
                        std::chrono::steady_clock::now();
                    const auto admission_delay =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            *operation_started_at_ - lane_admitted_at_)
                            .count();
                    admission_to_dispatch_nanoseconds_ =
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, admission_delay));
                    return startFirstRequest(error);
                }
                if (state_ == State::PreparingSourceChunk)
                    return prepareSourceChunk(error);
                if (state_ == State::CommittingDestinationChunk)
                    return pollDestinationChunk(error);

                int complete = 0;
                MPI_Status status{};
                const int mpi_result =
                    MPI_Test(&request_, &complete, &status);
                if (mpi_result != MPI_SUCCESS)
                    return failMPI("MPI_Test", mpi_result, error);
                if (!complete)
                {
                    impl_->stats->pending_mpi_polls.fetch_add(
                        1, std::memory_order_relaxed);
                    if (std::chrono::steady_clock::now() -
                            request_started_at_ >=
                        std::chrono::milliseconds(
                            collective_timeout_policy::
                                kDefaultCollectiveTimeoutMs))
                    {
                        std::ostringstream diagnostic;
                        diagnostic
                            << "MPI request exceeded the canonical 30-second timeout"
                            << " lane_index=" << lane_index_
                            << " state=" << static_cast<int>(state_)
                            << " role=" << static_cast<int>(role_)
                            << " source_rank=" << identity_.source_world_rank
                            << " destination_rank="
                            << identity_.destination_world_rank
                            << " migration_index="
                            << identity_.migration_index
                            << " projection_index="
                            << static_cast<int>(identity_.projection);
                        failure_ = diagnostic.str();
                        setError(error, failure_);
                        abortMoEOverlayMPI(
                            impl_->communicator,
                            impl_->world_rank,
                            "remote_projection",
                            failure_);
                    }
                    return MoEOverlayResidencyWaveProgress::Pending;
                }

                /*
                 * This is request-visible latency: it intentionally includes
                 * time until the maintenance worker next observed completion.
                 * Keeping it separate from active endpoint work exposes CPU
                 * scheduling starvation without pretending it is raw link
                 * time.
                 */
                recordCompletedMPIRequest();

                switch (state_)
                {
                case State::SendingManifest:
                    impl_->stats->manifests_sent.fetch_add(
                        1, std::memory_order_relaxed);
                    state_ = State::PreparingSourceChunk;
                    return prepareSourceChunk(error);
                case State::SendingHeader:
                    return startPayloadSend(error);
                case State::SendingPayload:
                    return completePayloadSend(error);
                case State::ReceivingManifest:
                    return completeManifestReceive(status, error);
                case State::ReceivingHeader:
                    return completeHeaderReceive(status, error);
                case State::ReceivingPayload:
                    return completePayloadReceive(status, error);
                default:
                    return failProtocol(
                        "Remote projection completed MPI in an invalid state",
                        error);
                }
            }

            /** @brief Cancel/drain the current point-to-point request and endpoints. */
            void abort() noexcept override
            {
                if (state_ == State::AbortReady)
                    return;
                if (source_)
                    source_->abort();
                if (destination_)
                    destination_->abort();
                if (request_ != MPI_REQUEST_NULL)
                {
                    const int cancel_result = MPI_Cancel(&request_);
                    if (cancel_result != MPI_SUCCESS && failure_.empty())
                        failure_ = mpiError("MPI_Cancel", cancel_result);
                }
                state_ = State::Aborting;
                impl_->stats->operations_aborted.fetch_add(
                    1, std::memory_order_relaxed);
            }

            /** @brief Test cancellation and endpoint events until lane reuse is safe. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (state_ == State::AbortReady)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (state_ != State::Aborting)
                {
                    setError(
                        error,
                        "Remote projection abort cleanup was polled before abort");
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                if (request_ != MPI_REQUEST_NULL)
                {
                    int complete = 0;
                    const int mpi_result = MPI_Test(
                        &request_, &complete, MPI_STATUS_IGNORE);
                    if (mpi_result != MPI_SUCCESS)
                        return failMPI("MPI_Test(cancel)", mpi_result, error);
                    if (!complete)
                        return MoEOverlayResidencyWaveProgress::Pending;
                    recordCompletedMPIRequest();
                }

                bool endpoint_pending = false;
                if (source_)
                {
                    const auto progress = source_->pollAbort(error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                        return progress;
                    endpoint_pending =
                        progress == MoEOverlayResidencyWaveProgress::Pending;
                }
                if (destination_)
                {
                    const auto progress = destination_->pollAbort(error);
                    if (progress == MoEOverlayResidencyWaveProgress::Failed)
                        return progress;
                    endpoint_pending = endpoint_pending ||
                        progress == MoEOverlayResidencyWaveProgress::Pending;
                }
                if (endpoint_pending)
                    return MoEOverlayResidencyWaveProgress::Pending;

                releaseLane();
                state_ = State::AbortReady;
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Return exact end-to-end MPI data-plane evidence at Ready. */
            [[nodiscard]] std::optional<
                ExpertTierProjectionTransferMeasurement>
            completedMeasurement() const noexcept override
            {
                return state_ == State::Ready ? measurement_ : std::nullopt;
            }

        private:
            /** @return Persistent lane storage reserved by this operation. */
            MoEOverlayMPIRemoteProjectionLaneStorage &lane() noexcept
            {
                return *impl_->lanes[lane_index_];
            }

            /** @return MPI tag derived solely from deterministic lane index. */
            int tag() const noexcept
            {
                return static_cast<int>(lane_index_);
            }

            /** @brief Start source manifest send or destination manifest receive. */
            MoEOverlayResidencyWaveProgress startFirstRequest(
                std::string *error) noexcept
            {
                int mpi_result = MPI_SUCCESS;
                if (role_ == Role::Source)
                {
                    if (!encodeMoEOverlayRemoteProjectionManifest(
                            source_->manifest(),
                            lane().manifest_packet,
                            error))
                    {
                        return failProtocol(
                            error && !error->empty()
                                ? *error
                                : "Remote source manifest could not be encoded",
                            error);
                    }
                    mpi_result = MPI_Isend(
                        lane().manifest_packet.data(),
                        static_cast<int>(lane().manifest_packet.size()),
                        MPI_BYTE,
                        identity_.destination_world_rank,
                        tag(),
                        impl_->communicator,
                        &request_);
                    state_ = State::SendingManifest;
                }
                else
                {
                    mpi_result = MPI_Irecv(
                        lane().manifest_packet.data(),
                        static_cast<int>(lane().manifest_packet.size()),
                        MPI_BYTE,
                        identity_.source_world_rank,
                        tag(),
                        impl_->communicator,
                        &request_);
                    state_ = State::ReceivingManifest;
                }
                if (mpi_result != MPI_SUCCESS ||
                    request_ == MPI_REQUEST_NULL)
                {
                    return failMPI(
                        role_ == Role::Source ? "MPI_Isend(manifest)"
                                             : "MPI_Irecv(manifest)",
                        mpi_result,
                        error);
                }
                recordStartedMPIRequest();
                return MoEOverlayResidencyWaveProgress::Pending;
            }

            /** @brief Obtain one ready endpoint chunk and start its header send. */
            MoEOverlayResidencyWaveProgress prepareSourceChunk(
                std::string *error) noexcept
            {
                MoEOverlayRemoteProjectionChunkView chunk;
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                const auto progress = source_->pollNextChunk(&chunk, error);
                recordHostWorkSince(endpoint_started_at);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                {
                    impl_->stats->pending_endpoint_polls.fetch_add(
                        1, std::memory_order_relaxed);
                    return progress;
                }
                if (progress != MoEOverlayResidencyWaveProgress::Ready ||
                    chunk.payload.empty() ||
                    chunk.header.manifest_hash !=
                        source_->manifest().manifest_hash ||
                    chunk.header.payload_bytes != chunk.payload.size() ||
                    !encodeMoEOverlayRemoteProjectionChunkHeader(
                        chunk.header, lane().header_packet, error))
                {
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote source produced an invalid next chunk",
                        error);
                }
                active_chunk_ = chunk;
                const int mpi_result = MPI_Isend(
                    lane().header_packet.data(),
                    static_cast<int>(lane().header_packet.size()),
                    MPI_BYTE,
                    identity_.destination_world_rank,
                    tag(),
                    impl_->communicator,
                    &request_);
                if (mpi_result != MPI_SUCCESS ||
                    request_ == MPI_REQUEST_NULL)
                    return failMPI("MPI_Isend(chunk header)", mpi_result, error);
                state_ = State::SendingHeader;
                recordStartedMPIRequest();
                return MoEOverlayResidencyWaveProgress::Pending;
            }

            /** @brief Send exact chunk bytes after its header buffer is reusable. */
            MoEOverlayResidencyWaveProgress startPayloadSend(
                std::string *error) noexcept
            {
                if (active_chunk_.payload.empty() ||
                    active_chunk_.payload.size() >
                        static_cast<std::size_t>(INT_MAX))
                {
                    return failProtocol(
                        "Remote source payload is empty or exceeds MPI count ABI",
                        error);
                }
                const int mpi_result = MPI_Isend(
                    active_chunk_.payload.data(),
                    static_cast<int>(active_chunk_.payload.size()),
                    MPI_BYTE,
                    identity_.destination_world_rank,
                    tag(),
                    impl_->communicator,
                    &request_);
                if (mpi_result != MPI_SUCCESS ||
                    request_ == MPI_REQUEST_NULL)
                    return failMPI("MPI_Isend(chunk payload)", mpi_result, error);
                state_ = State::SendingPayload;
                recordStartedMPIRequest();
                return MoEOverlayResidencyWaveProgress::Pending;
            }

            /** @brief Account a sent payload and either finish or request another. */
            MoEOverlayResidencyWaveProgress completePayloadSend(
                std::string *error) noexcept
            {
                impl_->stats->chunks_sent.fetch_add(
                    1, std::memory_order_relaxed);
                impl_->stats->bytes_sent.fetch_add(
                    active_chunk_.payload.size(), std::memory_order_relaxed);
                completed_payload_bytes_ =
                    saturatingExpertTierMeasurementAdd(
                        completed_payload_bytes_,
                        static_cast<std::uint64_t>(
                            active_chunk_.payload.size()));
                const bool final = active_chunk_.header.final_chunk != 0;
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                if (!source_->acknowledgeChunkSent(
                        active_chunk_.header, error))
                {
                    recordHostWorkSince(endpoint_started_at);
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote source rejected its completed MPI payload acknowledgement",
                        error);
                }
                recordHostWorkSince(endpoint_started_at);
                active_chunk_ = {};
                if (final)
                    return finish(error);
                state_ = State::PreparingSourceChunk;
                return prepareSourceChunk(error);
            }

            /** @brief Decode/authenticate manifest and post the first header receive. */
            MoEOverlayResidencyWaveProgress completeManifestReceive(
                const MPI_Status &status,
                std::string *error) noexcept
            {
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                int count = 0;
                const bool complete_manifest =
                    MPI_Get_count(&status, MPI_BYTE, &count) == MPI_SUCCESS &&
                    count == static_cast<int>(lane().manifest_packet.size());
                if (!complete_manifest)
                {
                    recordHostWorkSince(endpoint_started_at);
                    return failProtocol(
                        "Remote destination received a truncated manifest",
                        error);
                }
                MoEOverlayRemoteProjectionManifest manifest;
                const bool accepted =
                    decodeMoEOverlayRemoteProjectionManifest(
                        lane().manifest_packet, &manifest, error) &&
                    manifest.identity == identity_ &&
                    manifest.maximum_chunk_bytes <= impl_->staging_capacity &&
                    destination_->beginManifest(manifest, error);
                recordHostWorkSince(endpoint_started_at);
                if (!accepted)
                {
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote destination rejected the source manifest",
                        error);
                }
                received_manifest_ = manifest;
                impl_->stats->manifests_received.fetch_add(
                    1, std::memory_order_relaxed);
                return postHeaderReceive(error);
            }

            /** @brief Post one exact fixed-size header receive. */
            MoEOverlayResidencyWaveProgress postHeaderReceive(
                std::string *error) noexcept
            {
                const int mpi_result = MPI_Irecv(
                    lane().header_packet.data(),
                    static_cast<int>(lane().header_packet.size()),
                    MPI_BYTE,
                    identity_.source_world_rank,
                    tag(),
                    impl_->communicator,
                    &request_);
                if (mpi_result != MPI_SUCCESS ||
                    request_ == MPI_REQUEST_NULL)
                    return failMPI("MPI_Irecv(chunk header)", mpi_result, error);
                state_ = State::ReceivingHeader;
                recordStartedMPIRequest();
                return MoEOverlayResidencyWaveProgress::Pending;
            }

            /** @brief Decode a header and post an exact bounded payload receive. */
            MoEOverlayResidencyWaveProgress completeHeaderReceive(
                const MPI_Status &status,
                std::string *error) noexcept
            {
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                int count = 0;
                const bool accepted =
                    MPI_Get_count(&status, MPI_BYTE, &count) == MPI_SUCCESS &&
                    count == static_cast<int>(lane().header_packet.size()) &&
                    decodeMoEOverlayRemoteProjectionChunkHeader(
                        lane().header_packet, &received_header_, error) &&
                    received_header_.manifest_hash ==
                        received_manifest_.manifest_hash &&
                    received_header_.payload_bytes <= impl_->staging_capacity;
                recordHostWorkSince(endpoint_started_at);
                if (!accepted)
                {
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote destination received an invalid chunk header",
                        error);
                }
                const int mpi_result = MPI_Irecv(
                    lane().payload.data(),
                    static_cast<int>(received_header_.payload_bytes),
                    MPI_BYTE,
                    identity_.source_world_rank,
                    tag(),
                    impl_->communicator,
                    &request_);
                if (mpi_result != MPI_SUCCESS ||
                    request_ == MPI_REQUEST_NULL)
                    return failMPI("MPI_Irecv(chunk payload)", mpi_result, error);
                state_ = State::ReceivingPayload;
                recordStartedMPIRequest();
                return MoEOverlayResidencyWaveProgress::Pending;
            }

            /** @brief Hand authenticated-length staging to the destination endpoint. */
            MoEOverlayResidencyWaveProgress completePayloadReceive(
                const MPI_Status &status,
                std::string *error) noexcept
            {
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                int count = 0;
                if (MPI_Get_count(&status, MPI_BYTE, &count) != MPI_SUCCESS ||
                    count != static_cast<int>(received_header_.payload_bytes))
                {
                    recordHostWorkSince(endpoint_started_at);
                    return failProtocol(
                        "Remote destination received a truncated chunk payload",
                        error);
                }
                const auto payload = std::span<const std::uint8_t>(
                    lane().payload.data(), received_header_.payload_bytes);
                const auto progress = destination_->beginChunk(
                    received_header_, payload, error);
                recordHostWorkSince(endpoint_started_at);
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote destination rejected a chunk",
                        error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                {
                    state_ = State::CommittingDestinationChunk;
                    return progress;
                }
                return completeDestinationChunk(error);
            }

            /** @brief Event-poll one GPU destination commit when required. */
            MoEOverlayResidencyWaveProgress pollDestinationChunk(
                std::string *error) noexcept
            {
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                const auto progress = destination_->pollChunk(error);
                recordHostWorkSince(endpoint_started_at);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                {
                    impl_->stats->pending_endpoint_polls.fetch_add(
                        1, std::memory_order_relaxed);
                    return progress;
                }
                if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    return failProtocol(
                        error && !error->empty()
                            ? *error
                            : "Remote destination chunk commit failed",
                        error);
                return completeDestinationChunk(error);
            }

            /** @brief Account committed bytes, then finish or post next header. */
            MoEOverlayResidencyWaveProgress completeDestinationChunk(
                std::string *error) noexcept
            {
                const auto endpoint_started_at =
                    std::chrono::steady_clock::now();
                impl_->stats->chunks_received.fetch_add(
                    1, std::memory_order_relaxed);
                impl_->stats->bytes_received.fetch_add(
                    received_header_.payload_bytes,
                    std::memory_order_relaxed);
                completed_payload_bytes_ =
                    saturatingExpertTierMeasurementAdd(
                        completed_payload_bytes_,
                        static_cast<std::uint64_t>(
                            received_header_.payload_bytes));
                const bool final = received_header_.final_chunk != 0;
                received_header_ = {};
                if (final)
                {
                    if (!destination_->complete())
                    {
                        recordHostWorkSince(endpoint_started_at);
                        return failProtocol(
                            "Remote destination final marker preceded complete storage",
                            error);
                    }
                    if (!destination_->publishFinal(error))
                    {
                        recordHostWorkSince(endpoint_started_at);
                        return failProtocol(
                            error && !error->empty()
                                ? *error
                                : "Remote destination could not publish its prepared projection",
                            error);
                    }
                    recordHostWorkSince(endpoint_started_at);
                    return finish(error);
                }
                if (destination_->complete())
                {
                    recordHostWorkSince(endpoint_started_at);
                    return failProtocol(
                        "Remote destination completed before the sender final marker",
                        error);
                }
                recordHostWorkSince(endpoint_started_at);
                return postHeaderReceive(error);
            }

            /** @brief Release the persistent lane after successful completion. */
            MoEOverlayResidencyWaveProgress finish(
                std::string *error) noexcept
            {
                if (!operation_started_at_)
                {
                    return failProtocol(
                        "Remote projection reached completion before dispatch timing began",
                        error);
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() -
                        *operation_started_at_)
                        .count();
                const std::uint64_t wall_nanoseconds =
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, elapsed));
                measurement_ = {
                    .sequence = identity_.candidate_epoch,
                    .bytes = completed_payload_bytes_,
                    .wall_nanoseconds = wall_nanoseconds,
                    .device_nanoseconds = 0,
                    .host_nanoseconds = host_work_nanoseconds_,
                    .transport_nanoseconds = mpi_visible_nanoseconds_,
                };
                releaseLane();
                state_ = State::Ready;
                impl_->stats->operations_completed.fetch_add(
                    1, std::memory_order_relaxed);
                const PerfStatsCollector::Tags tags{
                    {"lane", std::to_string(lane_index_)},
                    {"role", role_ == Role::Source ? "source"
                                                    : "destination"},
                    {"purpose", transactionPurposeName(purpose_)},
                    {"candidate_epoch",
                     std::to_string(identity_.candidate_epoch)},
                    {"migration_index",
                     std::to_string(identity_.migration_index)},
                    {"layer", std::to_string(identity_.layer_idx)},
                    {"expert", std::to_string(identity_.expert_id)},
                    {"source_participant",
                     std::to_string(identity_.source_participant)},
                    {"destination_participant",
                     std::to_string(identity_.destination_participant)},
                };
                impl_->recordCounter(
                    "remote_projection_operations_completed",
                    1.0,
                    tags);
                /*
                 * Count final authenticated payload bytes, not MPI headers or
                 * manifests.  Both endpoints publish their own completion so
                 * a distributed report can prove send and receive progress;
                 * `role` keeps those quantities independently auditable.
                 */
                impl_->recordCounter(
                    "remote_projection_payload_bytes_completed",
                    static_cast<double>(completed_payload_bytes_),
                    tags);
                impl_->recordCounter(
                    "remote_projection_wall_ns",
                    static_cast<double>(wall_nanoseconds),
                    tags);
                impl_->recordCounter(
                    "remote_projection_admission_to_dispatch_ns",
                    static_cast<double>(
                        admission_to_dispatch_nanoseconds_),
                    tags);
                impl_->recordCounter(
                    "remote_projection_host_work_ns",
                    static_cast<double>(host_work_nanoseconds_),
                    tags);
                impl_->recordCounter(
                    "remote_projection_mpi_request_visible_ns",
                    static_cast<double>(mpi_visible_nanoseconds_),
                    tags);
                const std::uint64_t attributed_nanoseconds =
                    saturatingExpertTierMeasurementAdd(
                        host_work_nanoseconds_, mpi_visible_nanoseconds_);
                impl_->recordCounter(
                    "remote_projection_unattributed_poll_ns",
                    static_cast<double>(
                        wall_nanoseconds > attributed_nanoseconds
                            ? wall_nanoseconds - attributed_nanoseconds
                            : 0u),
                    tags);
                if (error)
                    error->clear();
                return MoEOverlayResidencyWaveProgress::Ready;
            }

            /** @brief Record one recoverable protocol failure before global abort. */
            MoEOverlayResidencyWaveProgress failProtocol(
                std::string message,
                std::string *error) noexcept
            {
                if (failure_.empty())
                    failure_ = std::move(message);
                state_ = State::Failed;
                impl_->stats->protocol_failures.fetch_add(
                    1, std::memory_order_relaxed);
                setError(error, failure_);
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Record MPI failure; request ownership may still require abort. */
            MoEOverlayResidencyWaveProgress failMPI(
                const char *operation,
                int mpi_result,
                std::string *error) noexcept
            {
                impl_->stats->mpi_failures.fetch_add(
                    1, std::memory_order_relaxed);
                return failProtocol(mpiError(operation, mpi_result), error);
            }

            /** @brief Return exactly this operation's reservation once. */
            void releaseLane() noexcept
            {
                if (lane_released_)
                    return;
                impl_->releaseLane(lane_index_);
                lane_released_ = true;
            }

            /** @brief Accumulate active endpoint/protocol CPU work. */
            void recordHostWorkSince(
                std::chrono::steady_clock::time_point started_at) noexcept
            {
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - started_at)
                        .count();
                host_work_nanoseconds_ = saturatingExpertTierMeasurementAdd(
                    host_work_nanoseconds_,
                    static_cast<std::uint64_t>(
                        std::max<std::int64_t>(1, elapsed)));
            }

            /** @brief Publish one newly outstanding request and update peak overlap. */
            void recordStartedMPIRequest() noexcept
            {
                if (request_ == MPI_REQUEST_NULL || request_counted_)
                    std::terminate();
                request_counted_ = true;
                request_started_at_ = std::chrono::steady_clock::now();
                const std::uint64_t active =
                    impl_->stats->active_mpi_requests.fetch_add(
                        1, std::memory_order_acq_rel) +
                    1;
                auto peak = impl_->stats->maximum_concurrent_mpi_requests.load(
                    std::memory_order_relaxed);
                bool raised_peak = false;
                while (peak < active &&
                       !impl_->stats->maximum_concurrent_mpi_requests
                            .compare_exchange_weak(
                                peak,
                                active,
                                std::memory_order_relaxed,
                                std::memory_order_relaxed))
                {
                }
                raised_peak = peak < active;
                if (raised_peak)
                {
                    impl_->recordCounter(
                        "remote_projection_concurrent_mpi_request_peak",
                        static_cast<double>(active),
                        {{"candidate_epoch",
                          std::to_string(identity_.candidate_epoch)},
                         {"layer", std::to_string(identity_.layer_idx)}});
                }
            }

            /** @brief Retire one request and accumulate observed MPI latency. */
            void recordCompletedMPIRequest() noexcept
            {
                if (!request_counted_)
                    std::terminate();
                request_counted_ = false;
                if (impl_->stats->active_mpi_requests.fetch_sub(
                        1, std::memory_order_acq_rel) == 0)
                {
                    std::terminate();
                }
                const auto elapsed =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - request_started_at_)
                        .count();
                mpi_visible_nanoseconds_ =
                    saturatingExpertTierMeasurementAdd(
                        mpi_visible_nanoseconds_,
                        static_cast<std::uint64_t>(
                            std::max<std::int64_t>(1, elapsed)));
            }

            std::shared_ptr<Impl> impl_;
            std::size_t lane_index_ = 0;
            MoEOverlayRemoteProjectionIdentity identity_;
            MoEOverlayResidencyTransactionPurpose purpose_ =
                MoEOverlayResidencyTransactionPurpose::PlacementChange;
            Role role_ = Role::Source;
            std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> source_;
            std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
                destination_;
            State state_ = State::Bound;
            MPI_Request request_ = MPI_REQUEST_NULL;
            /** True exactly while the shared active-request gauge owns this request. */
            bool request_counted_ = false;
            std::chrono::steady_clock::time_point request_started_at_{};
            /** Lane ownership begins here but no payload may have been released. */
            std::chrono::steady_clock::time_point lane_admitted_at_{};
            /** First maintenance dispatch poll; active wall timing begins here. */
            std::optional<std::chrono::steady_clock::time_point>
                operation_started_at_;
            MoEOverlayRemoteProjectionChunkView active_chunk_;
            MoEOverlayRemoteProjectionManifest received_manifest_;
            MoEOverlayRemoteProjectionChunkHeader received_header_;
            std::string failure_;
            std::optional<ExpertTierProjectionTransferMeasurement>
                measurement_;
            std::uint64_t completed_payload_bytes_ = 0;
            /** Active source hashing, destination validation/copy, and publication. */
            std::uint64_t host_work_nanoseconds_ = 0;
            /** MPI request latency including maintenance observation cadence. */
            std::uint64_t mpi_visible_nanoseconds_ = 0;
            /** Reservation/consensus or calibration delay before first dispatch. */
            std::uint64_t admission_to_dispatch_nanoseconds_ = 0;
            /* True until atomic wave admission transfers lane ownership here. */
            bool lane_released_ = true;
        };
    };

    MoEOverlayMPIRemoteProjectionTransport::
        MoEOverlayMPIRemoteProjectionTransport(Config config)
        : config_(std::move(config)), impl_(std::make_shared<Impl>())
    {
        const auto projection_lane_count =
            config_.lane_budget.projectionOperationCount();
        if (!config_.mpi_context ||
            config_.mpi_context->world_size() < 2 ||
            config_.mpi_context->rank() < 0 ||
            config_.mpi_context->rank() >=
                config_.mpi_context->world_size() ||
            config_.mpi_context->communicator() == MPI_COMM_NULL)
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay projection transport requires valid multi-rank MPI membership");
        }
        if (!projection_lane_count ||
            config_.staging_capacity_bytes == 0 ||
            config_.staging_capacity_bytes > static_cast<std::size_t>(INT_MAX))
        {
            throw std::invalid_argument(
                "Remote ExpertOverlay projection transport requires positive MPI-bounded lane capacity");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay_remote_projection";

        int initialized = 0;
        int finalized = 0;
        int thread_support = MPI_THREAD_SINGLE;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS || !initialized ||
            MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Query_thread(&thread_support) != MPI_SUCCESS ||
            thread_support < MPI_THREAD_MULTIPLE)
        {
            throw std::runtime_error(
                "Remote ExpertOverlay projection transport requires live MPI_THREAD_MULTIPLE support");
        }

        impl_->world_rank = config_.mpi_context->rank();
        impl_->world_size = config_.mpi_context->world_size();
        impl_->staging_capacity = config_.staging_capacity_bytes;
        impl_->perf_device = config_.perf_device;

        int flag = 0;
        int *tag_upper_bound = nullptr;
        const int attr_result = MPI_Comm_get_attr(
            config_.mpi_context->communicator(),
            MPI_TAG_UB,
            &tag_upper_bound,
            &flag);
        if (attr_result != MPI_SUCCESS || !flag || !tag_upper_bound)
            throw std::runtime_error(
                mpiError("MPI_Comm_get_attr(MPI_TAG_UB)", attr_result));
        impl_->tag_upper_bound = *tag_upper_bound;
        if (*projection_lane_count - 1u >
            static_cast<std::size_t>(impl_->tag_upper_bound))
        {
            throw std::runtime_error(
                "Remote ExpertOverlay projection lane BOM exceeds MPI_TAG_UB");
        }

        /*
         * Allocate every fixed receive lane before the collective duplication.
         * Allocation failure is then rank-local startup failure rather than a
         * peer deadlock in the next private-communicator constructor.
         */
        impl_->lanes.reserve(
            *projection_lane_count);
        for (std::size_t lane = 0;
             lane < *projection_lane_count;
             ++lane)
        {
            impl_->lanes.push_back(std::make_unique<
                MoEOverlayMPIRemoteProjectionLaneStorage>(
                config_.staging_capacity_bytes));
        }

        const int duplicate_result = MPI_Comm_dup(
            config_.mpi_context->communicator(), &impl_->communicator);
        if (duplicate_result != MPI_SUCCESS ||
            impl_->communicator == MPI_COMM_NULL)
        {
            throw std::runtime_error(
                mpiError("MPI_Comm_dup", duplicate_result));
        }

        impl_->stats->lanes_materialized.store(
            impl_->lanes.size(), std::memory_order_relaxed);
        impl_->recordCounter(
            "remote_projection_lanes_materialized",
            static_cast<double>(impl_->lanes.size()),
            {{"staging_capacity_bytes",
              std::to_string(config_.staging_capacity_bytes)},
             {"maximum_participants_per_cycle",
              std::to_string(
                  config_.lane_budget.maximum_participants_per_cycle)},
             {"maximum_concurrent_cycles",
              std::to_string(
                  config_.lane_budget.maximum_concurrent_cycles)},
             {"projections_per_expert",
              std::to_string(
                  config_.lane_budget.projections_per_expert)}});
    }

    MoEOverlayMPIRemoteProjectionTransport::
        ~MoEOverlayMPIRemoteProjectionTransport()
    {
        if (!impl_)
            return;
        if (!impl_.unique() || !impl_->allLanesIdle())
        {
            LOG_ERROR(
                "[MoEOverlayMPIRemoteProjectionTransport] Destroyed with live operations or reservations");
            std::terminate();
        }
        int finalized = 0;
        if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized ||
            MPI_Comm_free(&impl_->communicator) != MPI_SUCCESS)
        {
            LOG_ERROR(
                "[MoEOverlayMPIRemoteProjectionTransport] Communicator outlived MPI or failed to free");
            std::terminate();
        }
    }

    MoEOverlayMPIRemoteProjectionReservation
    MoEOverlayMPIRemoteProjectionTransport::reserve(
        std::size_t lane_index,
        const MoEOverlayRemoteProjectionIdentity &identity,
        std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> source,
        std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
            destination)
    {
        MoEOverlayMPIRemoteProjectionReservation result;
        auto wave = reserveWave({MoEOverlayMPIRemoteProjectionBinding{
            .lane_index = lane_index,
            .identity = identity,
            .source = std::move(source),
            .destination = std::move(destination),
        }});
        result.status = wave.status;
        result.error = std::move(wave.error);
        if (wave.status == MoEOverlayResidencyStageStartStatus::Started)
        {
            if (wave.operations.size() != 1 || !wave.operations.front())
                std::terminate();
            result.operation = std::move(wave.operations.front());
        }
        return result;
    }

    MoEOverlayMPIRemoteProjectionWaveReservation
    MoEOverlayMPIRemoteProjectionTransport::reserveWave(
        std::vector<MoEOverlayMPIRemoteProjectionBinding> bindings)
    {
        MoEOverlayMPIRemoteProjectionWaveReservation result;
        auto fail = [&](std::string message)
            -> MoEOverlayMPIRemoteProjectionWaveReservation
        {
            impl_->stats->wave_reservations_failed.fetch_add(
                1, std::memory_order_relaxed);
            result.status = MoEOverlayResidencyStageStartStatus::Failed;
            result.operations.clear();
            result.error = std::move(message);
            return std::move(result);
        };

        if (bindings.empty())
            return fail(
                "Remote ExpertOverlay wave reservation requires at least one projection");

        /*
         * Validate every endpoint and allocate every inert operation before
         * taking a lane mutex. Allocation failure can therefore unwind without
         * leaking a process-local reservation or requiring asynchronous abort.
         */
        std::vector<std::size_t> involved_binding_indices;
        std::vector<std::size_t> involved_lane_indices;
        std::vector<Impl::NetworkOperation *> network_operations;
        involved_binding_indices.reserve(bindings.size());
        involved_lane_indices.reserve(bindings.size());
        network_operations.reserve(bindings.size());
        result.operations.reserve(bindings.size());

        for (std::size_t binding_index = 0;
             binding_index < bindings.size();
             ++binding_index)
        {
            auto &binding = bindings[binding_index];
            if (!binding.identity.valid())
            {
                return fail(
                    "Remote ExpertOverlay wave contains an invalid projection identity at binding " +
                    std::to_string(binding_index));
            }
            if (!validTransactionPurpose(binding.purpose))
            {
                return fail(
                    "Remote ExpertOverlay wave contains an invalid transaction purpose at binding " +
                    std::to_string(binding_index));
            }
            if (binding.lane_index >= impl_->lanes.size())
            {
                return fail(
                    "Remote ExpertOverlay wave lane exceeds its setup-time BOM: binding=" +
                    std::to_string(binding_index) +
                    " lane=" + std::to_string(binding.lane_index) +
                    " lanes=" + std::to_string(impl_->lanes.size()));
            }

            const bool is_source =
                impl_->world_rank == binding.identity.source_world_rank;
            const bool is_destination =
                impl_->world_rank == binding.identity.destination_world_rank;
            const bool is_uninvolved = !is_source && !is_destination;
            if ((!is_uninvolved && is_source == is_destination) ||
                static_cast<bool>(binding.source) != is_source ||
                static_cast<bool>(binding.destination) != is_destination)
            {
                return fail(
                    "Remote ExpertOverlay projection endpoint ownership disagrees with world rank");
            }

            if (binding.source)
            {
                std::string manifest_error;
                const auto &manifest = binding.source->manifest();
                if (!manifest.valid(&manifest_error) ||
                    manifest.identity != binding.identity ||
                    manifest.maximum_chunk_bytes > impl_->staging_capacity)
                {
                    return fail(
                        manifest_error.empty()
                            ? "Remote source manifest disagrees with its wave reservation"
                            : std::move(manifest_error));
                }
            }

            if (is_uninvolved)
            {
                result.operations.push_back(std::make_unique<
                    UninvolvedRemoteProjectionOperation>());
                continue;
            }

            involved_binding_indices.push_back(binding_index);
            involved_lane_indices.push_back(binding.lane_index);
            auto operation = std::make_unique<Impl::NetworkOperation>(
                impl_,
                binding.lane_index,
                binding.identity,
                binding.purpose,
                is_source ? Impl::NetworkOperation::Role::Source
                          : Impl::NetworkOperation::Role::Destination,
                std::move(binding.source),
                std::move(binding.destination));
            network_operations.push_back(operation.get());
            result.operations.push_back(std::move(operation));
        }

        auto sorted_lanes = involved_lane_indices;
        std::sort(sorted_lanes.begin(), sorted_lanes.end());
        if (std::adjacent_find(sorted_lanes.begin(), sorted_lanes.end()) !=
            sorted_lanes.end())
        {
            return fail(
                "Remote ExpertOverlay wave assigns one MPI lane more than once");
        }

        /*
         * Deterministic lock order permits independent maintenance callers
         * without lock inversion. No operation owns a lane until all locks are
         * held and every occupied bit has been checked.
         */
        std::vector<std::unique_lock<std::mutex>> lane_locks;
        lane_locks.reserve(sorted_lanes.size());
        for (const auto lane_index_value : sorted_lanes)
        {
            lane_locks.emplace_back(
                impl_->lanes[lane_index_value]->mutex);
        }
        const auto retained = std::find_if(
            sorted_lanes.begin(),
            sorted_lanes.end(),
            [&](std::size_t lane_index_value)
            {
                return impl_->lanes[lane_index_value]->occupied;
            });
        if (retained != sorted_lanes.end())
        {
            result.operations.clear();
            result.status = MoEOverlayResidencyStageStartStatus::Deferred;
            result.error =
                "Remote ExpertOverlay projection wave has a lane retained by an older epoch";
            impl_->stats->reservations_deferred.fetch_add(
                1, std::memory_order_relaxed);
            impl_->stats->wave_reservations_deferred.fetch_add(
                1, std::memory_order_relaxed);
            return result;
        }

        for (const auto lane_index_value : involved_lane_indices)
            impl_->lanes[lane_index_value]->occupied = true;
        for (auto *operation : network_operations)
            operation->adoptReservedLane();

        result.status = MoEOverlayResidencyStageStartStatus::Started;
        impl_->stats->wave_reservations_started.fetch_add(
            1, std::memory_order_relaxed);
        impl_->stats->reservations_started.fetch_add(
            involved_binding_indices.size(), std::memory_order_relaxed);
        impl_->stats->uninvolved_operations.fetch_add(
            bindings.size() - involved_binding_indices.size(),
            std::memory_order_relaxed);
        for (const auto binding_index : involved_binding_indices)
        {
            const auto &identity = bindings[binding_index].identity;
            (impl_->world_rank == identity.source_world_rank
                 ? impl_->stats->source_operations
                 : impl_->stats->destination_operations)
                .fetch_add(1, std::memory_order_relaxed);
        }
        return result;
    }

    MoEOverlayMPIRemoteProjectionTransportStats
    MoEOverlayMPIRemoteProjectionTransport::stats() const noexcept
    {
        const auto &stats = *impl_->stats;
        return {
            .lanes_materialized =
                stats.lanes_materialized.load(std::memory_order_relaxed),
            .wave_reservations_started =
                stats.wave_reservations_started.load(
                    std::memory_order_relaxed),
            .wave_reservations_deferred =
                stats.wave_reservations_deferred.load(
                    std::memory_order_relaxed),
            .wave_reservations_failed =
                stats.wave_reservations_failed.load(
                    std::memory_order_relaxed),
            .reservations_started =
                stats.reservations_started.load(std::memory_order_relaxed),
            .reservations_deferred =
                stats.reservations_deferred.load(std::memory_order_relaxed),
            .source_operations =
                stats.source_operations.load(std::memory_order_relaxed),
            .destination_operations =
                stats.destination_operations.load(std::memory_order_relaxed),
            .uninvolved_operations =
                stats.uninvolved_operations.load(std::memory_order_relaxed),
            .manifests_sent =
                stats.manifests_sent.load(std::memory_order_relaxed),
            .manifests_received =
                stats.manifests_received.load(std::memory_order_relaxed),
            .chunks_sent = stats.chunks_sent.load(std::memory_order_relaxed),
            .chunks_received =
                stats.chunks_received.load(std::memory_order_relaxed),
            .bytes_sent = stats.bytes_sent.load(std::memory_order_relaxed),
            .bytes_received =
                stats.bytes_received.load(std::memory_order_relaxed),
            .pending_mpi_polls =
                stats.pending_mpi_polls.load(std::memory_order_relaxed),
            .pending_endpoint_polls =
                stats.pending_endpoint_polls.load(std::memory_order_relaxed),
            .maximum_concurrent_mpi_requests =
                stats.maximum_concurrent_mpi_requests.load(
                    std::memory_order_relaxed),
            .active_mpi_requests =
                stats.active_mpi_requests.load(std::memory_order_relaxed),
            .operations_completed =
                stats.operations_completed.load(std::memory_order_relaxed),
            .operations_aborted =
                stats.operations_aborted.load(std::memory_order_relaxed),
            .mpi_failures =
                stats.mpi_failures.load(std::memory_order_relaxed),
            .protocol_failures =
                stats.protocol_failures.load(std::memory_order_relaxed),
            .inference_stream_waits = 0,
            .blocking_synchronizations = 0,
        };
    }

    int MoEOverlayMPIRemoteProjectionTransport::worldRank() const noexcept
    {
        return impl_->world_rank;
    }

    int MoEOverlayMPIRemoteProjectionTransport::worldSize() const noexcept
    {
        return impl_->world_size;
    }
} // namespace llaminar2
