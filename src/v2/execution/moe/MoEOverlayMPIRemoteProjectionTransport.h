/**
 * @file MoEOverlayMPIRemoteProjectionTransport.h
 * @brief Asynchronous MPI data plane for remote ExpertOverlay projections.
 *
 * The distributed residency vote lane proves when a whole placement epoch is
 * publishable; this transport moves the actual gate/up/down bytes that precede
 * that vote.  One model-lifetime private communicator owns a bounded set of
 * deterministic tags and persistent receive buffers.  Projection operations
 * reserve one lane before the global reservation barrier, then begin MPI and
 * device work only when their first maintenance poll occurs.
 *
 * Endpoint interfaces are chunk-oriented so GPU implementations can expose
 * event-ready pinned chunks without constructing a full host mirror.  CPU
 * endpoints supplied here use the identical contract and provide a device-free
 * oracle for protocol and MPI integration tests.
 */

#pragma once

#include "MoEOverlayRemoteProjectionProtocol.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    class IMPIContext;

    /**
     * @brief Source-side producer of authenticated host-visible network chunks.
     *
     * A CPU implementation returns spans into immutable final execution bytes.
     * A GPU implementation first enqueues conversion or D2H on its exact
     * auxiliary stream, returns Pending while its event is incomplete, and then
     * returns a span into model-lifetime pinned staging storage.
     */
    class IMoEOverlayRemoteProjectionSourceEndpoint
    {
    public:
        virtual ~IMoEOverlayRemoteProjectionSourceEndpoint() = default;

        /** @return Immutable authenticated manifest sent before any chunk. */
        [[nodiscard]] virtual const MoEOverlayRemoteProjectionManifest &
        manifest() const noexcept = 0;

        /**
         * @brief Produce the next event-ready chunk without waiting.
         * @param chunk Receives a header and stable payload span on Ready.
         * @param error Receives an exact conversion/event failure.
         * @return Pending while device preparation runs, Ready for one chunk,
         *         or Failed. Ready never means end-of-stream without a chunk.
         */
        virtual MoEOverlayResidencyWaveProgress pollNextChunk(
            MoEOverlayRemoteProjectionChunkView *chunk,
            std::string *error = nullptr) noexcept = 0;

        /**
         * @brief Acknowledge that MPI no longer reads the last returned payload.
         * @param header Exact header returned by the preceding successful poll.
         * @param error Optional lifecycle diagnostic.
         * @return True when endpoint staging may be recycled or released.
         *
         * GPU endpoints retain a pinned chunk until the non-blocking payload
         * send completes.  Making that ownership edge explicit prevents a
         * later repack/D2H submission from overwriting bytes still owned by MPI.
         */
        virtual bool acknowledgeChunkSent(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::string *error = nullptr) noexcept = 0;

        /** @brief Discard unpublished source work without blocking. */
        virtual void abort() noexcept = 0;

        /**
         * @brief Poll source-side event cleanup after abort.
         * @param error Optional cleanup diagnostic.
         * @return Ready only when its payload storage may be reused.
         */
        virtual MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief Destination-side consumer of authenticated network chunks.
     *
     * MPI always receives into persistent lane staging first.  The destination
     * validates a chunk before copying it to final CPU storage or enqueueing
     * H2D/repack into an inactive GPU slot.  At most one chunk may be active.
     */
    class IMoEOverlayRemoteProjectionDestinationEndpoint
    {
    public:
        virtual ~IMoEOverlayRemoteProjectionDestinationEndpoint() = default;

        /**
         * @brief Validate the received manifest and bind final storage.
         * @param manifest Authenticated sender format and identity.
         * @param error Optional topology/format/capacity diagnostic.
         * @return True before the first payload receive may be posted.
         */
        virtual bool beginManifest(
            const MoEOverlayRemoteProjectionManifest &manifest,
            std::string *error = nullptr) noexcept = 0;

        /**
         * @brief Validate and begin committing one received staging chunk.
         * @param header Exact ordered range already decoded by the MPI lane.
         * @param payload Authenticated-length persistent receive staging.
         * @param error Optional validation or submission diagnostic.
         * @return Pending for asynchronous device commit, Ready when the chunk
         *         is fully in final storage, or Failed.
         */
        virtual MoEOverlayResidencyWaveProgress beginChunk(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept = 0;

        /**
         * @brief Poll an asynchronously committing destination chunk.
         * @param error Optional device-event diagnostic.
         * @return Pending, Ready, or Failed without synchronization.
         */
        virtual MoEOverlayResidencyWaveProgress pollChunk(
            std::string *error = nullptr) noexcept = 0;

        /** @return Whether the unique final chunk is in final destination storage. */
        [[nodiscard]] virtual bool complete() const noexcept = 0;

        /**
         * @brief Publish the prepared execution object after final-byte readiness.
         * @param error Optional exact publication diagnostic.
         * @return True only when the destination may participate in bank commit.
         *
         * The MPI operation calls this once after @ref complete becomes true and
         * before releasing its lane. Device-free storage oracles may implement
         * an explicit no-op; production endpoints use it to publish the exact
         * prepared engine into their candidate expert arrival authority.
         */
        virtual bool publishFinal(std::string *error = nullptr) noexcept = 0;

        /** @brief Discard unpublished destination work without waiting. */
        virtual void abort() noexcept = 0;

        /**
         * @brief Poll destination-side event cleanup after abort.
         * @param error Optional cleanup diagnostic.
         * @return Ready only when final/staging storage may be recycled.
         */
        virtual MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief Device-free source over immutable final host-visible regions.
     *
     * The retained lifetime pins the prepared source engine or equivalent slot
     * owner.  No bytes are copied: MPI sends each cursor span directly.
     */
    class MoEOverlayHostRemoteProjectionSource final
        : public IMoEOverlayRemoteProjectionSourceEndpoint
    {
    public:
        /**
         * @brief Bind one manifest to exact source regions and ownership.
         * @param manifest Valid authenticated CPU or host-staged GPU contract.
         * @param regions Exact immutable regions named by @p manifest.
         * @param lifetime Non-null source-slot/engine lifetime.
         * @throws std::invalid_argument For incomplete storage or ownership.
         */
        MoEOverlayHostRemoteProjectionSource(
            MoEOverlayRemoteProjectionManifest manifest,
            std::array<
                std::span<const std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions,
            std::shared_ptr<void> lifetime);

        /** @return Retained immutable manifest. */
        [[nodiscard]] const MoEOverlayRemoteProjectionManifest &manifest()
            const noexcept override
        {
            return manifest_;
        }

        /** @brief Return the next allocation-free cursor chunk immediately. */
        MoEOverlayResidencyWaveProgress pollNextChunk(
            MoEOverlayRemoteProjectionChunkView *chunk,
            std::string *error = nullptr) noexcept override;

        /** @brief Release the cursor span after MPI completes its payload send. */
        bool acknowledgeChunkSent(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::string *error = nullptr) noexcept override;

        /** @brief Mark later cursor reads illegal. */
        void abort() noexcept override { aborted_ = true; }

        /** @brief Host spans have no asynchronous cleanup edge. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept override;

    private:
        MoEOverlayRemoteProjectionManifest manifest_;
        MoEOverlayRemoteProjectionChunkCursor cursor_;
        std::shared_ptr<void> lifetime_;
        std::optional<MoEOverlayRemoteProjectionChunkHeader>
            outstanding_header_;
        bool aborted_ = false;
    };

    /** Policy for matching a received physical format to final host storage. */
    enum class MoEOverlayRemoteProjectionManifestMatchPolicy : std::uint8_t
    {
        Exact, ///< Every authenticated physical-format field must match.
        /**
         * The final CPU representation must match exactly, while the sender's
         * live GPU representation may be canonical or its valid normalized
         * CPU-promotion form.
         */
        EquivalentGpuSourceForFinalCpuStorage,
    };

    /**
     * @brief Device-free destination writing validated bytes to final regions.
     *
     * Validation occurs before every memcpy, so corrupt, reordered, or replayed
     * input cannot mutate final storage.  The lifetime pins the inactive slot
     * and its prepared engine until the enclosing residency bank adopts it.
     */
    class MoEOverlayHostRemoteProjectionDestination final
        : public IMoEOverlayRemoteProjectionDestinationEndpoint
    {
    public:
        /**
         * @brief Bind expected identity, final regions, and slot ownership.
         * @param expected_identity Exact transaction projection expected here.
         * @param regions Preallocated final writable regions.
         * @param lifetime Non-null destination-slot/engine lifetime.
         * @param expected_manifest Optional complete physical format contract.
         *        Production destinations supply it; protocol-only adversarial
         *        tests may omit it when testing identity/range validation alone.
         * @param match_policy Whether the live sender GPU representation must
         *        equal the expected physical fields or may be an equivalent
         *        valid representation producing the same final CPU storage.
         * @throws std::invalid_argument For invalid identity or ownership.
         */
        MoEOverlayHostRemoteProjectionDestination(
            MoEOverlayRemoteProjectionIdentity expected_identity,
            std::array<
                std::span<std::uint8_t>,
                kMoEOverlayRemoteProjectionRegionCount> regions,
            std::shared_ptr<void> lifetime,
            std::optional<MoEOverlayRemoteProjectionManifest>
                expected_manifest = std::nullopt,
            MoEOverlayRemoteProjectionManifestMatchPolicy match_policy =
                MoEOverlayRemoteProjectionManifestMatchPolicy::Exact);

        /** @brief Authenticate format, identity, and all final-region sizes. */
        bool beginManifest(
            const MoEOverlayRemoteProjectionManifest &manifest,
            std::string *error = nullptr) noexcept override;

        /** @brief Validate then copy one bounded chunk into its exact final range. */
        MoEOverlayResidencyWaveProgress beginChunk(
            const MoEOverlayRemoteProjectionChunkHeader &header,
            std::span<const std::uint8_t> payload,
            std::string *error = nullptr) noexcept override;

        /** @brief Host commit is synchronous and has no pending event. */
        MoEOverlayResidencyWaveProgress pollChunk(
            std::string *error = nullptr) noexcept override;

        /** @return Whether the strict validator accepted the final marker. */
        [[nodiscard]] bool complete() const noexcept override;

        /** @brief Explicit device-free no-op publication for protocol oracles. */
        bool publishFinal(std::string *error = nullptr) noexcept override;

        /** @brief Poison the validator and prohibit later final-memory writes. */
        void abort() noexcept override;

        /** @brief Host destination writes have no asynchronous cleanup edge. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error = nullptr) noexcept override;

    private:
        MoEOverlayRemoteProjectionIdentity expected_identity_;
        std::array<
            std::span<std::uint8_t>,
            kMoEOverlayRemoteProjectionRegionCount> regions_;
        std::shared_ptr<void> lifetime_;
        std::optional<MoEOverlayRemoteProjectionManifest> expected_manifest_;
        MoEOverlayRemoteProjectionManifestMatchPolicy match_policy_ =
            MoEOverlayRemoteProjectionManifestMatchPolicy::Exact;
        std::optional<MoEOverlayRemoteProjectionChunkValidator> validator_;
        bool aborted_ = false;
    };

    /** Process-local evidence for the persistent MPI projection data plane. */
    struct MoEOverlayMPIRemoteProjectionTransportStats
    {
        std::uint64_t lanes_materialized = 0;
        std::uint64_t wave_reservations_started = 0;
        std::uint64_t wave_reservations_deferred = 0;
        std::uint64_t wave_reservations_failed = 0;
        std::uint64_t reservations_started = 0;
        std::uint64_t reservations_deferred = 0;
        std::uint64_t source_operations = 0;
        std::uint64_t destination_operations = 0;
        std::uint64_t uninvolved_operations = 0;
        std::uint64_t manifests_sent = 0;
        std::uint64_t manifests_received = 0;
        std::uint64_t chunks_sent = 0;
        std::uint64_t chunks_received = 0;
        std::uint64_t bytes_sent = 0;
        std::uint64_t bytes_received = 0;
        std::uint64_t pending_mpi_polls = 0;
        std::uint64_t pending_endpoint_polls = 0;
        std::uint64_t operations_completed = 0;
        std::uint64_t operations_aborted = 0;
        std::uint64_t mpi_failures = 0;
        std::uint64_t protocol_failures = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
    };

    /** Result of reserving one deterministic projection lane. */
    struct MoEOverlayMPIRemoteProjectionReservation
    {
        MoEOverlayResidencyStageStartStatus status =
            MoEOverlayResidencyStageStartStatus::Failed;
        std::unique_ptr<IMoEOverlayTierTransferOperation> operation;
        std::string error;
    };

    /**
     * @brief One transaction-local projection endpoint presented for admission.
     *
     * The lane index is globally deterministic even on an uninvolved rank. Only
     * the source rank supplies @ref source, only the destination rank supplies
     * @ref destination, and every other rank supplies neither endpoint.
     */
    struct MoEOverlayMPIRemoteProjectionBinding
    {
        std::size_t lane_index = 0;
        MoEOverlayRemoteProjectionIdentity identity;
        /** Typed intent used to separate aborted calibration traffic from live moves. */
        MoEOverlayResidencyTransactionPurpose purpose =
            MoEOverlayResidencyTransactionPurpose::PlacementChange;
        std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> source;
        std::shared_ptr<IMoEOverlayRemoteProjectionDestinationEndpoint>
            destination;
    };

    /**
     * @brief Exact persistent MPI lane budget for bounded migration cycles.
     *
     * A capacity-preserving cycle never visits one physical participant more
     * than once, so `maximum_participants_per_cycle` also bounds the number of
     * expert migrations in one cycle. Multiplying by the public concurrent
     * cycle policy and the expert projection count yields the complete
     * setup-time lane count. Keeping the factors typed prevents a caller from
     * materializing a one-cycle pool while the residency authority admits a
     * wider wave.
     */
    struct MoEOverlayRemoteProjectionLaneBudget
    {
        std::size_t maximum_participants_per_cycle = 0;
        std::size_t maximum_concurrent_cycles = 0;
        std::size_t projections_per_expert = 0;

        /**
         * @brief Derive the complete projection-operation lane count.
         * @return Positive count, or `nullopt` for zero/overflowing geometry.
         */
        [[nodiscard]] constexpr std::optional<std::size_t>
        projectionOperationCount() const noexcept
        {
            if (maximum_participants_per_cycle == 0 ||
                maximum_concurrent_cycles == 0 ||
                projections_per_expert == 0 ||
                maximum_participants_per_cycle >
                    std::numeric_limits<std::size_t>::max() /
                        maximum_concurrent_cycles)
            {
                return std::nullopt;
            }
            const std::size_t migrations =
                maximum_participants_per_cycle * maximum_concurrent_cycles;
            if (migrations > std::numeric_limits<std::size_t>::max() /
                                 projections_per_expert)
            {
                return std::nullopt;
            }
            return migrations * projections_per_expert;
        }

        /** @brief Compare all setup-time capacity factors. */
        bool operator==(
            const MoEOverlayRemoteProjectionLaneBudget &) const = default;
    };

    /** @brief Atomic result of admitting one complete remote projection wave. */
    struct MoEOverlayMPIRemoteProjectionWaveReservation
    {
        MoEOverlayResidencyStageStartStatus status =
            MoEOverlayResidencyStageStartStatus::Failed;
        std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
            operations;
        std::string error;
    };

    /**
     * @brief Private-communicator pool of persistent remote projection lanes.
     *
     * Lane index is `migration_index * projections_per_expert + projection` in
     * the enclosing transaction.  Every rank therefore derives the same MPI tag
     * without assuming where CUDA, ROCm, or CPU participants live.  Only source
     * and destination ranks reserve storage; uninvolved ranks receive a real
     * no-op operation so the composite wave keeps identical cardinality.
     */
    class MoEOverlayMPIRemoteProjectionTransport final
    {
    public:
        /** @brief Immutable MPI membership and model-time capacity BOM. */
        struct Config
        {
            std::shared_ptr<IMPIContext> mpi_context;
            MoEOverlayRemoteProjectionLaneBudget lane_budget;
            std::size_t staging_capacity_bytes = 0;
            std::string perf_device;
        };

        /**
         * @brief Duplicate one communicator and allocate every lane buffer.
         * @param config Multi-rank MPI_THREAD_MULTIPLE context and positive BOM.
         * @throws std::invalid_argument For invalid membership or capacity.
         * @throws std::runtime_error For MPI setup/thread/tag failures.
         */
        explicit MoEOverlayMPIRemoteProjectionTransport(Config config);

        /**
         * @brief Free the private communicator only after all lanes are idle.
         *
         * Destruction never drains or waits. A live request or reservation is a
         * fatal runner ownership error.
         */
        ~MoEOverlayMPIRemoteProjectionTransport();

        MoEOverlayMPIRemoteProjectionTransport(
            const MoEOverlayMPIRemoteProjectionTransport &) = delete;
        MoEOverlayMPIRemoteProjectionTransport &operator=(
            const MoEOverlayMPIRemoteProjectionTransport &) = delete;

        /**
         * @brief Reserve one projection without starting MPI or device work.
         * @param lane_index Deterministic transaction-local operation index.
         * @param identity Exact source/destination topology and epoch identity.
         * @param source Non-null only on `identity.source_world_rank`.
         * @param destination Non-null only on destination world rank.
         * @return Started owned operation, typed lane backpressure, or failure.
         *
         * The returned operation begins work on its first poll, which occurs
         * only after the distributed reservation vote is unanimously Ready.
         */
        MoEOverlayMPIRemoteProjectionReservation reserve(
            std::size_t lane_index,
            const MoEOverlayRemoteProjectionIdentity &identity,
            std::shared_ptr<IMoEOverlayRemoteProjectionSourceEndpoint> source,
            std::shared_ptr<
                IMoEOverlayRemoteProjectionDestinationEndpoint> destination);

        /**
         * @brief Atomically reserve every process-local lane in a closed wave.
         * @param bindings Complete remote projection set in composite order.
         * @return One operation per binding on Started, and no ownership on
         *         Deferred or structural failure.
         *
         * Operations are allocated while inert. The method then locks involved
         * lanes in ascending index order, verifies all are free, and adopts all
         * reservations as one state transition. Consequently a later retained
         * lane cannot leave an earlier lane partially reserved, and no MPI or
         * device work begins before the outer all-rank reservation consensus.
         */
        MoEOverlayMPIRemoteProjectionWaveReservation reserveWave(
            std::vector<MoEOverlayMPIRemoteProjectionBinding> bindings);

        /** @return World rank bound to the private transport communicator. */
        [[nodiscard]] int worldRank() const noexcept;

        /** @return Fixed communicator membership used by projection identities. */
        [[nodiscard]] int worldSize() const noexcept;

        /** @return Race-safe cumulative proof counters. */
        [[nodiscard]] MoEOverlayMPIRemoteProjectionTransportStats stats()
            const noexcept;

    private:
        struct Impl;

        Config config_;
        std::shared_ptr<Impl> impl_;
    };
} // namespace llaminar2
