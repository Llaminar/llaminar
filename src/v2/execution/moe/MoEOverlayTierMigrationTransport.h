/**
 * @file MoEOverlayTierMigrationTransport.h
 * @brief Composite asynchronous transport for arbitrary-tier residency waves.
 *
 * A histogram proposal may contain several closed migration cycles and every
 * moved expert has gate, up, and down projections. This file provides the
 * production control-plane composition: endpoint code reserves/pins resources
 * and returns independently pollable transfer operations plus one inactive-bank
 * transaction. The composite wave advances all operations without waiting and
 * exposes commit readiness to `MoEOverlayResidencyAuthority` only after every
 * projection is complete.
 */

#pragma once

#include "ExpertTierTransferMeasurement.h"
#include "MoEOverlayResidencyAuthority.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    struct MoEOverlayTierMigrationTransportSharedStats;

    /**
     * @brief Device-progress authority shared by every operation in one wave.
     *
     * Heterogeneous GPU relay lanes publish commands into topology-sized mapped
     * epochs. The physical fabric implements this interface so a composite wave
     * can advance each unique device epoch once per maintenance poll, rather
     * than giving every projection lane a duplicate submission authority.
     */
    class IMoEOverlayTransferProgressAuthority
    {
    public:
        virtual ~IMoEOverlayTransferProgressAuthority() = default;

        /**
         * @brief Enqueue all outstanding device progress without a device wait.
         * @param error Optional exact worker/launch failure diagnostic.
         * @return True after every required graph enqueue or idle no-op.
         */
        [[nodiscard]] virtual bool submitOutstandingTransferProgress(
            std::string *error = nullptr) noexcept = 0;
    };

    /**
     * @brief Completed physical evidence for one expert in a migration wave.
     *
     * A rank may own none, some, or all projection observations for a
     * cross-rank movement. Missing optionals are explicit and are merged by the
     * distributed certification owner; they are never replaced with a sibling
     * backend estimate. `wave_wall_nanoseconds` is shared by every migration in
     * the concurrently executed closed wave and conservatively captures its
     * actual event-polled critical path.
     */
    struct MoEOverlayCompletedMigrationMeasurement
    {
        std::uint64_t expected_epoch = 0;
        std::uint64_t candidate_epoch = 0;
        int source_participant = -1;
        int destination_participant = -1;
        int layer = -1;
        int expert = -1;
        std::array<
            std::optional<ExpertTierProjectionTransferMeasurement>,
            3>
            projections;
        std::uint64_t wave_wall_nanoseconds = 0;

        /** @return Whether identity, wall time, and optional values are coherent. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Authority receiving one complete event-polled wave observation.
     *
     * Implementations run on the maintenance worker after all projection
     * events are ready and before inactive-bank commit. They may copy bounded
     * pointer-free evidence into preallocated or setup-owned storage, but must
     * not wait for inference, a device, or a collective.
     */
    class IMoEOverlayMigrationMeasurementSink
    {
    public:
        virtual ~IMoEOverlayMigrationMeasurementSink() = default;

        /**
         * @brief Retain one locally observed wave without blocking.
         * @param measurements Transaction-ordered expert observations.
         * @param error Exact rejection diagnostic.
         * @return Whether the complete local observation was accepted.
         */
        virtual bool recordCompletedWave(
            const std::vector<MoEOverlayCompletedMigrationMeasurement> &
                measurements,
            std::string *error) noexcept = 0;
    };

    /**
     * @brief One already-enqueued projection transfer in a residency wave.
     *
     * Implementations wrap a persistent CPU-edge conversion lane, a same-backend
     * peer-copy lane, a heterogeneous GPU blob lane, or an explicit network
     * transport. `poll()` is non-blocking and `abort()` only marks/recycles work
     * asynchronously after its own events make that safe.
     */
    class IMoEOverlayTierTransferOperation
    {
    public:
        virtual ~IMoEOverlayTierTransferOperation() = default;

        /**
         * @brief Query operation readiness without synchronizing any stream.
         * @param error Exact failure diagnostic when `Failed` is returned.
         * @return Pending, Ready, or Failed for this projection.
         */
        virtual MoEOverlayResidencyWaveProgress poll(
            std::string *error) noexcept = 0;

        /** @brief Abort unpublished work and arrange asynchronous recycling. */
        virtual void abort() noexcept = 0;

        /**
         * @brief Query whether this operation's abort cleanup may be destroyed.
         * @param error Exact event/transport failure diagnostic.
         * @return Pending, Ready, or Failed without waiting.
         */
        virtual MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept
        {
            (void)error;
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        /**
         * @brief Return pointer-free timing evidence after `poll()` returns Ready.
         *
         * An empty result means this rank performed no physical projection
         * work, as on a non-endpoint rank in a distributed wave. A configured
         * certification sink decides whether that absence is expected.
         */
        [[nodiscard]] virtual std::optional<
            ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept
        {
            return std::nullopt;
        }
    };

    /**
     * @brief Inactive runtime-bank transaction associated with a transfer wave.
     *
     * Endpoint reservations, prepared-engine handles, and candidate descriptor
     * banks remain owned here until abort or old-epoch retirement. Preparation
     * may be enqueued only after all transfer operations report Ready; selector
     * publication is a separate irreversible phase.
     */
    class IMoEOverlayInactiveBankTransaction
    {
    public:
        virtual ~IMoEOverlayInactiveBankTransaction() = default;

        /**
         * @brief Validate all candidate engines and enqueue inactive-bank build.
         * @param error Exact enqueue or completeness failure.
         * @return Whether preparation work was enqueued without blocking.
         */
        virtual bool beginPrepare(std::string *error) noexcept = 0;

        /**
         * @brief Query candidate-bank preparation readiness without waiting.
         * @param error Exact failure diagnostic when `Failed` is returned.
         * @return Pending, Ready, or Failed for bank construction.
         */
        virtual MoEOverlayResidencyWaveProgress pollPrepare(
            std::string *error) noexcept = 0;

        /**
         * @brief Enqueue process-local inference-visible selector publication.
         * @param error Exact enqueue or lifecycle failure.
         * @return Whether publication was submitted without blocking.
         */
        virtual bool beginPublication(std::string *error) noexcept = 0;

        /**
         * @brief Query process-local selector publication without waiting.
         * @param error Exact failure diagnostic when `Failed` is returned.
         * @return Pending, Ready, or Failed for selector publication.
         */
        virtual MoEOverlayResidencyWaveProgress pollPublication(
            std::string *error) noexcept = 0;

        /** @brief Abort and recycle every unpublished destination reservation. */
        virtual void abort() noexcept = 0;

        /**
         * @brief Query whether aborted bank resources may now be destroyed.
         * @param error Exact cleanup failure diagnostic.
         * @return Pending, Ready, or Failed without waiting.
         */
        virtual MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept
        {
            (void)error;
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        /**
         * @brief Poll participant-local device readers for the prior epoch.
         * @param error Exact grace-period failure diagnostic.
         * @return Pending while a device reader remains, Ready when local
         *         runtime storage may enter the distributed retirement vote,
         *         or Failed for a broken publication lifecycle.
         *
         * Host-only banks have no second reader domain and therefore complete
         * immediately. GPU-backed implementations override this method and
         * drive their exact device RCU retirement kernels by events.
         */
        virtual MoEOverlayResidencyWaveProgress pollRetirementFence(
            std::string *error) noexcept
        {
            if (error)
                error->clear();
            return MoEOverlayResidencyWaveProgress::Ready;
        }

        /** @brief Retire source exclusions after old-epoch tickets have drained. */
        virtual void retirePrevious() noexcept = 0;
    };

    /** @brief Resources returned after atomic source-pin/shadow-slot reservation. */
    struct MoEOverlayTierPreparedWave
    {
        MoEOverlayResidencyStageStartStatus status =
            MoEOverlayResidencyStageStartStatus::Failed;
        std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>> transfers;
        std::unique_ptr<IMoEOverlayInactiveBankTransaction> inactive_bank;
        std::string error;

        /**
         * @brief Validate ownership shape for a requested operation count.
         * @param expected_transfer_count Migrations times projections per expert.
         * @return Whether status and resource ownership are self-consistent.
         */
        [[nodiscard]] bool valid(
            std::size_t expected_transfer_count) const noexcept;
    };

    /**
     * @brief Endpoint-owned factory that atomically reserves one physical wave.
     *
     * The factory resolves every migration to its live source and inactive
     * destination, pins all sources, reserves every shadow requirement, and
     * starts one operation per projection. It returns `Deferred` before moving
     * bytes when the complete closed-cycle reservation cannot fit.
     */
    class IMoEOverlayTierPreparedWaveFactory
    {
    public:
        virtual ~IMoEOverlayTierPreparedWaveFactory() = default;

        /**
         * @brief Reserve and enqueue all physical work for one transaction.
         * @param transaction Immutable capacity-preserving candidate wave.
         * @return Started resources, typed deferral, or fatal failure.
         */
        virtual MoEOverlayTierPreparedWave prepare(
            const MoEOverlayResidencyTransaction &transaction) = 0;
    };

    /** @brief Process-local proof counters for composite residency waves. */
    struct MoEOverlayTierMigrationTransportStats
    {
        std::uint64_t waves_started = 0;
        std::uint64_t waves_deferred = 0;
        std::uint64_t waves_failed_to_prepare = 0;
        std::uint64_t transfer_operations_started = 0;
        std::uint64_t transfer_operations_completed = 0;
        /** Ready projection operations belonging to publishable placement waves. */
        std::uint64_t placement_transfer_operations_completed = 0;
        /** Exact payload bytes observed for publishable placement waves. */
        std::uint64_t placement_transfer_payload_bytes_completed = 0;
        std::uint64_t stage_pending_polls = 0;
        std::uint64_t commits_started = 0;
        std::uint64_t commits_completed = 0;
        std::uint64_t commit_pending_polls = 0;
        std::uint64_t waves_aborted = 0;
        std::uint64_t abort_pending_polls = 0;
        std::uint64_t abort_cleanups_completed = 0;
        std::uint64_t old_banks_retired = 0;
        std::uint64_t inference_stream_waits = 0;
        std::uint64_t blocking_synchronizations = 0;
    };

    /** @brief Typed lifetime scope for optional completed-wave measurement. */
    enum class MoEOverlayMigrationMeasurementScope : std::uint8_t
    {
        AllWaves, ///< Observe calibration and publishable placement waves.
        EconomyCalibrationOnly, ///< Stop journaling after setup certification.
    };

    /**
     * @brief Compose endpoint transfer operations into one publishable wave.
     *
     * Publication remains serialized by `MoEOverlayResidencyAuthority`, while
     * all transfer operations in this object may progress concurrently. The
     * default projection count is three (gate/up/down); it is explicit in the
     * configuration so model families with a different typed projection set
     * cannot silently under-stage an expert.
     */
    class MoEOverlayTierMigrationTransport final
        : public IMoEOverlayResidencyTransport
    {
    public:
        /** @brief Immutable factory, projection cardinality, and evidence label. */
        struct Config
        {
            IMoEOverlayTierPreparedWaveFactory *factory = nullptr;
            /** Optional physical authority for mapped GPU relay progress. */
            std::shared_ptr<IMoEOverlayTransferProgressAuthority>
                transfer_progress_authority;
            std::size_t projections_per_expert = 3;
            /** Optional setup/runtime owner for exact movement observations. */
            std::shared_ptr<IMoEOverlayMigrationMeasurementSink>
                measurement_sink;
            /** Select which typed transaction purpose reaches the sink. */
            MoEOverlayMigrationMeasurementScope measurement_scope =
                MoEOverlayMigrationMeasurementScope::AllWaves;
            /** Reject a ready local operation that omitted timing evidence. */
            bool require_complete_local_measurements = false;
            std::string perf_device;
        };

        /**
         * @brief Bind one endpoint factory for the transport lifetime.
         * @param config Non-null factory and positive projection cardinality.
         * @throws std::invalid_argument For an incomplete configuration.
         */
        explicit MoEOverlayTierMigrationTransport(Config config);

        /**
         * @brief Reserve and start one non-blocking composite wave.
         * @param transaction Valid candidate from the residency authority.
         * @return Started wave, exact deferral, or fatal preparation failure.
         */
        MoEOverlayResidencyStageStart beginStage(
            const MoEOverlayResidencyTransaction &transaction) override;

        /** @brief Return cumulative control-plane proof counters. */
        [[nodiscard]] MoEOverlayTierMigrationTransportStats stats() const noexcept;

        /** @return Exact payload bytes in completed publishable waves. */
        [[nodiscard]] std::uint64_t
        completedPlacementPayloadBytes() const noexcept override;

    private:
        Config config_;
        std::shared_ptr<MoEOverlayTierMigrationTransportSharedStats> stats_;
    };
} // namespace llaminar2
