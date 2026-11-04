/**
 * @file MoEOverlayParticipantMigration.h
 * @brief Production RCU bank composition for ExpertOverlay migration waves.
 *
 * Physical transfer lanes move one gate, up, or down projection and eventually
 * publish a prepared destination engine.  Sparse inference, however, consumes
 * one immutable participant bank containing complete expert payloads.  This
 * file joins those two ownership domains without making inference wait: a
 * topology-owned provider reserves and enqueues the physical work, while the
 * participant factory clones the old RCU banks, applies the complete movement
 * transaction off to the side, and installs the candidate epoch only after all
 * projection operations report readiness.
 */

#pragma once

#include "ExpertTierWeightStream.h"
#include "ExpertTierWeightDeviceLayout.h"

#include <functional>
#include "MoEOverlayHostAuthorityDeviceBankPublisher.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayTierMigrationTransport.h"

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace test { struct MoEOverlayParticipantMigrationTestAccess; }

    /** Number of independently transferred projections in one routed expert. */
    inline constexpr std::size_t kMoEOverlayExpertProjectionCount = 3;

    /**
     * @brief Thread-safe handoff from projection transport to bank construction.
     *
     * A physical operation may construct its prepared GEMM before enqueue, from
     * a progress callback, or on a background worker.  It publishes exactly one
     * shared lifetime under the matching projection role.  The bank transaction
     * reads the payload only after the composite transfer barrier reports Ready,
     * but the mutex also makes publication safe when callbacks run concurrently.
     */
    class MoEOverlayPreparedExpertArrival final
    {
    public:
        /**
         * @brief Bind the immutable projection contract before any arrival.
         * @throws std::invalid_argument For an unknown projection family.
         */
        explicit MoEOverlayPreparedExpertArrival(
            DeviceMoEProjectionSet projections = DeviceMoEProjectionSet::CompleteExpert);

        /** @return Whether this arrival has registry-authenticated CPU fragment provenance. */
        bool isCPUJointFragment() const noexcept
        {
            return fragment_destination_.has_value();
        }
        /** @return Exact candidate epoch for a fragment arrival; zero for an ordinary arrival. */
        uint64_t destinationEpoch() const noexcept
        {
            return fragment_destination_ ? fragment_destination_->epoch : 0;
        }
        /**
         * @brief Seal a completed local fragment without representing it as a whole expert.
         * @param payload Receives genuine completed engines and fixed original cohort provenance.
         * @param error Precise wrong-arm, incomplete or failed-arrival diagnostic.
         * @return True only for a complete registry-issued CPU fragment arrival.
         */
        bool completeCPUFragment(MoEOverlayPreparedCPUFragmentPayload &payload,
                                 std::string *error = nullptr) const;

        /**
         * @return Required physical role set for this transfer transaction.
         * @note CompleteExpert names gate/up/down roles only for a fragment arrival;
         * it grants no whole-owner payload. isCPUJointFragment() distinguishes ownership,
         * and completePayload() rejects that fragment arm.
         */
        [[nodiscard]] DeviceMoEProjectionSet projections() const noexcept
        {
            return projections_;
        }
        /**
         * @brief Publish one prepared projection lifetime exactly once.
         * @param projection Gate, up, or down role represented by @p engine.
         * @param engine Non-null destination execution engine.
         * @param error Optional exact rejection diagnostic.
         * @return True for the first publication or an identical idempotent one.
         */
        bool publish(ExpertTierWeightProjection projection, std::shared_ptr<ITensorGemm> engine,
                     std::string *error = nullptr);

        /**
         * @brief Mark asynchronous preparation as irrecoverably failed.
         * @param error Non-empty physical preparation diagnostic.
         * @return True only when this call installed the first failure.
         */
        bool fail(std::string error);

        /**
         * @brief Snapshot the complete destination payload after transfer readiness.
         * @param payload Receives exactly the declared prepared-engine family.
         * @param error Optional incomplete/failure diagnostic.
         * @return True only when all projections are present and no failure exists.
         */
        [[nodiscard]] bool completePayload(MoEOverlayPreparedExpertPayload &payload,
                                           std::string *error = nullptr) const;

    private:
        friend class MoEOverlayParticipantResidencyRegistry;
        /** @brief Immutable candidate identity issued only by the original source-bound registry.
         */
        struct CPUFragmentDestination
        {
            ModelContextId model;
            std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
            std::shared_ptr<const MoEExpertShardCohort> cohort;
            std::size_t member;
            uint64_t epoch;
            int expert;
        };
        /**
         * @brief Admit only the registry's checked original fixed-cohort candidate destination.
         * @param model Original source-bound model identity.
         * @param preparation SAME original preparation control block.
         * @param cohort SAME original fixed source/cohort/cut authority.
         * @param member Original cohort member coordinate.
         * @param candidate_epoch Exact unpublished candidate bank epoch.
         * @param expert Original candidate logical expert, not initial selected ordinal.
         * @throws std::invalid_argument For absent or malformed destination identity.
         */
        MoEOverlayPreparedExpertArrival(
            ModelContextId model,
            std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation,
            std::shared_ptr<const MoEExpertShardCohort> cohort, std::size_t member,
            uint64_t candidate_epoch, int expert);
        const std::optional<CPUFragmentDestination> fragment_destination_;
        mutable std::mutex mutex_;
        const DeviceMoEProjectionSet projections_;
        std::array<std::shared_ptr<ITensorGemm>, 3> engines_{};
        std::string failure_;
    };

    /**
     * @brief Checked destination coverage shared by its original physical operations.
     *
     * Construction proves a complete disjoint original-coordinate tiling before
     * any operation can claim a piece. Readiness comes only from the associated
     * physical operation, including any ACK it owns, never recipient storage finalization.
     * This object owns no allocation or placement ledger.
     */
    class MoEOverlayPreparedProjectionCompletion final
    {
    public:
        /**
         * @brief Bind a complete nonoverlapping destination and its final engine provider.
         * @param intersections Exact original source/destination overlap for every physical piece.
         * @param arrival Existing expert arrival receiving the completed projection.
         * @param projection Semantic gate/up/down role.
         * @param engine_provider Returns the actual reserved destination engine after all pieces
         * finish.
         * @throws std::invalid_argument For empty, overlapping, incomplete or foreign destination
         * coverage.
         * @note A multi-piece provider must construct and validate this complete coverage before
         * enqueueing any physical operation in its wave. No invalid prefix may begin transfer.
         */
        MoEOverlayPreparedProjectionCompletion(
            std::vector<ExpertTierWeightProjectionIntersection> intersections,
            std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
            ExpertTierWeightProjection projection,
            std::function<std::shared_ptr<ITensorGemm>()> engine_provider);

    private:
        friend class MoEOverlayPreparedProjectionOperation;
        /**
         * @brief Assign a coverage piece to its unique physical operation.
         * @param piece Original piece index.
         * @throws std::invalid_argument For duplicate or out-of-range ownership.
         */
        void claim(std::size_t piece);
        /**
         * @brief Accept actual physical completion and publish only the complete tiling.
         * @param piece Claimed physical piece.
         * @param error Failure diagnostic.
         * @return Whether this piece's readiness was accepted.
         */
        bool ready(std::size_t piece, std::string *error) noexcept;
        /**
         * @brief Preserve the first physical failure across every destination piece.
         * @param error Physical operation failure; poisons unpublished arrival.
         */
        void fail(std::string error) noexcept;
        /**
         * @brief Read the shared destination failure without changing physical operations.
         * @param error Receives the original failure.
         * @return Whether any piece failed or aborted.
         */
        bool failed(std::string *error) const noexcept;

        const std::vector<ExpertTierWeightProjectionIntersection> intersections_;
        std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival_;
        ExpertTierWeightProjection projection_;
        std::function<std::shared_ptr<ITensorGemm>()> engine_provider_;
        mutable std::mutex mutex_;
        std::vector<bool> claimed_, ready_;
        std::string failure_;
        bool published_ = false;
    };

    /**
     * @brief Retain one original physical operation and its exact destination piece.
     * Each operation keeps its own timing, ACK, source/destination owners and
     * asynchronous drain. Only the checked shared completion can publish the
     * destination engine, after every claimed intersection reports Ready.
     */
    class MoEOverlayPreparedProjectionOperation final : public IMoEOverlayTierTransferOperation
    {
    public:
        /**
         * @brief Bind one original physical operation to exactly one checked piece.
         * @param operation Already-enqueued original lane/ACK operation.
         * @param completion Complete immutable destination coverage.
         * @param piece Exact original index in that coverage; may be claimed only once.
         * @throws std::invalid_argument For absent ownership or an invalid/duplicate piece.
         */
        MoEOverlayPreparedProjectionOperation(
            std::unique_ptr<IMoEOverlayTierTransferOperation> operation,
            std::shared_ptr<MoEOverlayPreparedProjectionCompletion> completion, std::size_t piece);
        /** @brief Poll the original operation and report its piece only after actual Ready. */
        MoEOverlayResidencyWaveProgress poll(std::string *error) noexcept override;
        /** @brief Poison unpublished completion and asynchronously abort the original operation. */
        void abort() noexcept override;
        /** @brief Drain the original operation without releasing its retained owners early. */
        MoEOverlayResidencyWaveProgress pollAbort(std::string *error) noexcept override;
        /** @return Unmodified timing of this physical piece after successful completion. */
        std::optional<ExpertTierProjectionTransferMeasurement>
        completedMeasurement() const noexcept override;
        /** @return The underlying actual operation quote, with unchanged source lifetime. */
        std::optional<std::uint64_t> quotedPayloadBytes() const noexcept override;

    private:
        std::unique_ptr<IMoEOverlayTierTransferOperation> operation_;
        std::shared_ptr<MoEOverlayPreparedProjectionCompletion> completion_;
        std::size_t piece_;
        bool ready_ = false;
        bool aborted_ = false;
    };

    /** @brief One real physical piece, in role/destination-member/source-member order. */
    struct MoEOverlayParticipantProjectionTransfer
    {
        ExpertTierWeightProjection projection=ExpertTierWeightProjection::Gate;
        int destination_participant=-1; ///< Canonical original target member, never allocation ordinal.
        int source_participant=-1; ///< Canonical original source member.
        std::unique_ptr<IMoEOverlayTierTransferOperation> operation;
        bool requires_local_measurement = false;
    };

    /** @brief One actual local destination's complete role arrival authority. */
    struct MoEOverlayParticipantDestinationArrival
    {
        int participant_id=-1;
        std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival;
    };

    /**
     * @brief All physical work and participant-keyed arrivals for one logical migration.
     *
     * Records retain logical migration order. Operations contain only actual
     * physical/protocol work, in role, original destination-member, original
     * source-member order. Local arrivals match the exact local destination
     * target members; source-only and empty-local ranks fabricate no arrival.
     * Storage capacity never determines these live vector extents.
     */
    struct MoEOverlayParticipantExpertTransfer
    {
        std::vector<MoEOverlayParticipantProjectionTransfer> operations;
        std::vector<MoEOverlayParticipantDestinationArrival> destination_arrivals;
        std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry> measurement_geometry;
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
     * enqueueing its first byte. Complete multi-piece destination coverage must
     * also be checked before that enqueue edge. A `Deferred` result owns no operation.  A
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
            /**
             * Process-local GPU bank authority for heterogeneous host policy.
             * Null is valid only when every local participant is CPU-owned.
             */
            std::shared_ptr<IMoEOverlayHostAuthorityDeviceBankPublisher>
                device_bank_publisher;
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
        friend struct test::MoEOverlayParticipantMigrationTestAccess;
        /** Explicit recorded component boundary, inaccessible to production callers. */
        struct RecordedComponentSource {};
        /**
         * @brief Preserve component protocol tests without pretending to own a model source.
         * @param config Existing test registry and recorded transfer provider.
         * @param source Private test-access tag, never a public runtime policy.
         */
        MoEOverlayParticipantPreparedWaveFactory(Config config,RecordedComponentSource source);
        Config config_;
        bool requires_original_source_=true; ///< Constructor-only; public runtime construction always authenticates.
    };
} // namespace llaminar2
