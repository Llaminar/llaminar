/**
 * @file MoEOverlayParticipantMigration.h
 * @brief Production RCU bank composition for ExpertOverlay migration waves.
 *
 * Physical transfer lanes move one gate, up, or down projection and eventually
 * publish a prepared destination engine.  Sparse inference, however, consumes
 * one immutable participant bank containing complete expert triplets.  This
 * file joins those two ownership domains without making inference wait: a
 * topology-owned provider reserves and enqueues the physical work, while the
 * participant factory clones the old RCU banks, applies the complete movement
 * transaction off to the side, and installs the candidate epoch only after all
 * projection operations report readiness.
 */

#pragma once

#include "ExpertTierWeightStream.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Number of independently transferred projections in one routed expert. */
    inline constexpr std::size_t kMoEOverlayExpertProjectionCount = 3;

    /**
     * @brief Thread-safe handoff from projection transport to bank construction.
     *
     * A physical operation may construct its prepared GEMM before enqueue, from
     * a progress callback, or on a background worker.  It publishes exactly one
     * shared lifetime under the matching projection role.  The bank transaction
     * reads the triplet only after the composite transfer barrier reports Ready,
     * but the mutex also makes publication safe when callbacks run concurrently.
     */
    class MoEOverlayPreparedExpertArrival final
    {
    public:
        /**
         * @brief Publish one prepared projection lifetime exactly once.
         * @param projection Gate, up, or down role represented by @p engine.
         * @param engine Non-null destination execution engine.
         * @param error Optional exact rejection diagnostic.
         * @return True for the first publication or an identical idempotent one.
         */
        bool publish(
            ExpertTierWeightProjection projection,
            std::shared_ptr<ITensorGemm> engine,
            std::string *error = nullptr);

        /**
         * @brief Mark asynchronous preparation as irrecoverably failed.
         * @param error Non-empty physical preparation diagnostic.
         * @return True only when this call installed the first failure.
         */
        bool fail(std::string error);

        /**
         * @brief Snapshot the complete destination triplet after transfer readiness.
         * @param triplet Receives the three retained prepared-engine lifetimes.
         * @param error Optional incomplete/failure diagnostic.
         * @return True only when all projections are present and no failure exists.
         */
        [[nodiscard]] bool completeTriplet(
            MoEOverlayPreparedExpertTriplet &triplet,
            std::string *error = nullptr) const;

    private:
        mutable std::mutex mutex_;
        MoEOverlayPreparedExpertTriplet triplet_;
        std::string failure_;
    };

    /**
     * @brief Publish one prepared engine only after its physical transfer is ready.
     *
     * The wrapper retains the destination engine, its arrival authority, and
     * the exact underlying lane operation.  A successful lane poll publishes
     * the engine lifetime before returning Ready to the composite barrier.  A
     * lane or publication failure poisons the arrival and fails the wave.  No
     * stream wait, host synchronization, or engine allocation occurs here.
     */
    class MoEOverlayPreparedProjectionOperation final
        : public IMoEOverlayTierTransferOperation
    {
    public:
        /**
         * @brief Bind an already-enqueued operation to one destination engine.
         * @param operation Exact physical lane operation.
         * @param arrival Shared expert-level gate/up/down arrival authority.
         * @param projection Projection role published on readiness.
         * @param engine Fully constructed engine over reserved destination storage.
         * @throws std::invalid_argument When any ownership is absent.
         */
        MoEOverlayPreparedProjectionOperation(
            std::unique_ptr<IMoEOverlayTierTransferOperation> operation,
            std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
            ExpertTierWeightProjection projection,
            std::shared_ptr<ITensorGemm> engine);

        /** @brief Poll the lane and publish the retained engine at exact readiness. */
        MoEOverlayResidencyWaveProgress poll(
            std::string *error) noexcept override;

        /** @brief Abort the physical operation and poison this unpublished arrival. */
        void abort() noexcept override;

        /** @brief Delegate asynchronous reclamation to the exact physical operation. */
        MoEOverlayResidencyWaveProgress pollAbort(
            std::string *error) noexcept override;

        /** @brief Delegate completed timing evidence to the physical operation. */
        [[nodiscard]] std::optional<
            ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override;

    private:
        std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
        std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival_;
        ExpertTierWeightProjection projection_ =
            ExpertTierWeightProjection::Gate;
        std::shared_ptr<ITensorGemm> engine_;
        bool published_ = false;
        bool aborted_ = false;
    };

    /**
     * @brief Physical work and optional local arrival for one migration record.
     *
     * Records are ordered exactly like `MoEOverlayResidencyTransaction::migrations`.
     * Every record owns three operations.  `destination_arrival` is present only
     * on the process that owns the destination participant; other ranks still
     * carry their topology protocol operations but do not create a host shadow
     * of a remote prepared engine.
     */
    struct MoEOverlayParticipantExpertTransfer
    {
        std::array<
            std::unique_ptr<IMoEOverlayTierTransferOperation>,
            kMoEOverlayExpertProjectionCount>
            projections;
        std::shared_ptr<MoEOverlayPreparedExpertArrival> destination_arrival;
    };

    /** @brief Atomic result of reserving and enqueueing one physical wave. */
    struct MoEOverlayParticipantPreparedTransfers
    {
        MoEOverlayResidencyStageStartStatus status =
            MoEOverlayResidencyStageStartStatus::Failed;
        std::vector<MoEOverlayParticipantExpertTransfer> migrations;
        std::string error;
    };

    /**
     * @brief Topology-specific authority for projection slots, lanes, and engines.
     *
     * Implementations must reserve the complete closed-cycle transaction before
     * enqueueing its first byte.  A `Deferred` result owns no operation.  A
     * `Failed` result may return a fenced prefix so the generic wave can retain
     * and asynchronously abort it.  CUDA/CPU, CUDA/ROCm, same-backend peer, and
     * cross-rank providers all implement this one contract.
     */
    class IMoEOverlayParticipantTransferProvider
    {
    public:
        virtual ~IMoEOverlayParticipantTransferProvider() = default;

        /**
         * @brief Reserve and enqueue every projection in one residency transaction.
         * @param transaction Immutable global migration and epoch identity.
         * @param local_destination_participants Sorted participants hosted here.
         * @return Exact started work, typed backpressure, or retained failure work.
         */
        virtual MoEOverlayParticipantPreparedTransfers prepareTransfers(
            const MoEOverlayResidencyTransaction &transaction,
            const std::vector<int> &local_destination_participants) = 0;

        /**
         * @brief Recycle process-local source slots after the old ticket barrier.
         * @param retired_epoch Exact residency epoch no longer reachable by a
         *        live inference ticket.
         * @param migrations Complete globally ordered movement set that was
         *        published in the successor epoch.
         *
         * This callback is part of successful commit, never abort. Providers
         * must ignore non-local sources and must not synchronize a device;
         * destination lease aliases perform their own deferred recycling.
         */
        virtual void retirePreviousSources(
            std::uint64_t retired_epoch,
            const std::vector<MoEOverlayTierMigration> &migrations) noexcept = 0;
    };

    /**
     * @brief Compose physical projection work with participant-local RCU banks.
     *
     * `prepare()` runs only on the maintenance domain.  It clones every local
     * endpoint's previous bank and checks shadow capacity before asking the
     * physical provider to enqueue work.  Its returned inactive-bank transaction
     * installs all local candidate banks before the global authority can publish
     * the new epoch, rolls back partial installation on failure, and retires the
     * old epoch only after ticket leases drain.
     */
    class MoEOverlayParticipantPreparedWaveFactory final
        : public IMoEOverlayTierPreparedWaveFactory
    {
    public:
        /** @brief Stable process-local ownership and physical transport authority. */
        struct Config
        {
            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry;
            std::shared_ptr<IMoEOverlayParticipantTransferProvider>
                transfer_provider;
            std::string perf_device;
        };

        /**
         * @brief Bind the process-local registry and topology transfer provider.
         * @param config Non-null authorities retained for every outstanding wave.
         * @throws std::invalid_argument When either authority is absent.
         */
        explicit MoEOverlayParticipantPreparedWaveFactory(Config config);

        /**
         * @brief Clone candidate banks and start the complete physical wave.
         * @param transaction Valid non-empty authority proposal.
         * @return Composite-ready transfers and one RCU bank transaction.
         */
        MoEOverlayTierPreparedWave prepare(
            const MoEOverlayResidencyTransaction &transaction) override;

    private:
        Config config_;
    };
} // namespace llaminar2
