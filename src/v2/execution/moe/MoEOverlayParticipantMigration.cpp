/**
 * @file MoEOverlayParticipantMigration.cpp
 * @brief RCU candidate-bank implementation for production ExpertOverlay moves.
 *
 * The code below deliberately performs no device operation.  Exact physical
 * providers own transfer streams and readiness events; this layer owns only
 * immutable prepared-engine lifetimes and the epoch transaction.  Keeping that
 * boundary narrow lets CPU-only tests exhaust the publication protocol while
 * real-device integration tests prove each provider's event DAG independently.
 */

#include "MoEOverlayParticipantMigration.h"
#include "MoEOverlayPhysicalProjectionPieces.h"
#include "MoEExpertShardCohort.h"
#include "MoEOverlayPreparedWeightSource.h"

#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>
#include <tuple>

namespace llaminar2
{
    namespace
    {
        /** @brief Select the shared-pointer field named by one wire projection. */
        std::shared_ptr<ITensorGemm> &projectionField(
            std::array<std::shared_ptr<ITensorGemm>, 3> &engines,
            ExpertTierWeightProjection projection)
        {
            switch (projection)
            {
            case ExpertTierWeightProjection::Gate:
                return engines[0];
            case ExpertTierWeightProjection::Up:
                return engines[1];
            case ExpertTierWeightProjection::Down:
                return engines[2];
            }
            throw std::invalid_argument(
                "ExpertOverlay arrival references an unknown projection role");
        }

        /** @brief Return whether a sorted participant vector contains an id. */
        bool containsParticipant(
            const std::vector<int> &participants,
            int participant_id) noexcept
        {
            return std::binary_search(
                participants.begin(), participants.end(), participant_id);
        }

        /** @brief Mutable candidate plus the exact endpoint that will own it. */
        struct LocalCandidateBank
        {
            int participant_id = -1;
            std::shared_ptr<MoEOverlayParticipantResidency> endpoint;
            MoEOverlayParticipantResidencyBank bank;
            std::optional<MoEOverlayPreparedParticipantBank> prepared_bank;
            bool installed_by_this_transaction = false;
        };

        /**
         * @brief Build every local epoch clone before any physical work starts.
         *
         * This is the shadow-capacity preflight.  A closed cycle is deferred as
         * a whole if even one endpoint cannot retain old plus candidate banks.
         */
        MoEOverlayResidencyStageStartStatus buildLocalCandidates(
            const std::shared_ptr<MoEOverlayParticipantResidencyRegistry>
                &registry,
            const MoEOverlayResidencyTransaction &transaction,
            std::vector<LocalCandidateBank> &candidates,
            std::string *error)
        {
            candidates.clear();
            const auto local_ids = registry->localParticipantIds();
            candidates.reserve(local_ids.size());
            for (const int participant_id : local_ids)
            {
                auto endpoint = registry->endpoint(participant_id);
                const auto *previous_participant =
                    transaction.previous->placementOwnerMap().participantForId(
                        participant_id);
                const auto *candidate_participant =
                    transaction.candidate->placementOwnerMap().participantForId(
                        participant_id);
                if (!endpoint || !previous_participant ||
                    !candidate_participant ||
                    previous_participant->device != endpoint->device() ||
                    candidate_participant->device != endpoint->device())
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay candidate references a missing or "
                            "changed local participant p" +
                            std::to_string(participant_id);
                    }
                    return MoEOverlayResidencyStageStartStatus::Failed;
                }
                if (!endpoint->acquire(transaction.previous->epoch))
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay local participant p" +
                            std::to_string(participant_id) +
                            " does not retain previous epoch " +
                            std::to_string(transaction.previous->epoch);
                    }
                    return MoEOverlayResidencyStageStartStatus::Failed;
                }
                if (!endpoint->hasCandidateCapacity())
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay local participant p" +
                            std::to_string(participant_id) +
                            " has no inactive RCU bank capacity";
                    }
                    candidates.clear();
                    return MoEOverlayResidencyStageStartStatus::Deferred;
                }

                try
                {
                    // Validate both snapshots before preparing any physical
                    // transfers. Equal row counts do not identify a PP stage.
                    for (const auto *snapshot :
                         {transaction.previous.get(), transaction.candidate.get()})
                    {
                        snapshot->placementOwnerMap().requireLayerGeometry(
                            endpoint->numLayers(), endpoint->numExperts(),
                            endpoint->firstModelLayer());
                        if (snapshot->ownershipKind()==MoEOverlayResidencyOwnershipKind::WholeExperts &&
                            (snapshot->layered_ownership.firstModelLayer() != endpoint->firstModelLayer() ||
                            snapshot->layered_ownership.layerCount() != endpoint->numLayers() ||
                            snapshot->layered_ownership.expertCount() != endpoint->numExperts()))
                            throw std::logic_error(
                                "ExpertOverlay migration snapshot belongs to another model stage");
                    }
                    auto bank = endpoint->cloneCandidate(
                        transaction.previous->epoch,
                        transaction.candidate->epoch);
                    candidates.push_back({
                        .participant_id = participant_id,
                        .endpoint = std::move(endpoint),
                        .bank = std::move(bank),
                    });
                }
                catch (const std::exception &exception)
                {
                    if (error)
                        *error = exception.what();
                    candidates.clear();
                    return MoEOverlayResidencyStageStartStatus::Failed;
                }
            }

            /*
             * Remove departures only from unpublished clones.  Arrivals are
             * applied later, after their three physical operations are Ready.
             */
            try {
                for(const auto& migration:transaction.migrations) {
                    const auto target=MoEExpertPlacementTarget::fromPlacement(migration.source);
                    if(!target)throw std::logic_error("Candidate departure has no checked original source target");
                    for(size_t member=0;member<target->participantCount();++member) {
                        const int id=target->participantId(member);
                        const auto found=std::find_if(candidates.begin(),candidates.end(),[&](const auto& local){return local.participant_id==id;});
                        if(found==candidates.end())continue;
                        auto& layer=found->bank.layerForModelLayer(migration.layer_idx);
                        if(migration.expert_id<0 || size_t(migration.expert_id)>=layer.resident_mask.size() || !layer.resident_mask[size_t(migration.expert_id)])
                            throw std::logic_error("Candidate departure lost an original source bank resident");
                        layer.clearExpert(migration.expert_id);
                    }
                }
            } catch(const std::exception& exception) {
                if(error)*error=exception.what();candidates.clear();return MoEOverlayResidencyStageStartStatus::Failed;
            }

            return MoEOverlayResidencyStageStartStatus::Started;
        }

        /**
         * @brief Local inactive-bank owner retained by the composite wave.
         *
         * Installation is atomic relative to global publication: if any local
         * endpoint rejects its candidate, every bank installed earlier by this
         * transaction is removed before failure is reported.
         */
        class ParticipantInactiveBankTransaction final
            : public IMoEOverlayInactiveBankTransaction
        {
        public:
            /**
             * @brief Exact participant-bank lifecycle owned by this adapter.
             *
             * The phase is deliberately singular: combinations such as
             * "published and aborted" or "retired while preparation is still
             * pending" cannot be represented.  Per-device fan-out has its own
             * typed endpoint phases inside the device publisher.
             */
            enum class Phase
            {
                Created,
                Preparing,
                Prepared,
                Publishing,
                Published,
                RetirementFencing,
                RetirementReady,
                Aborting,
                Aborted,
                Retired,
                FailedBeforePublication,
                FailedAfterPublication,
            };

            /** @brief Take candidate clones and transfer-arrival lifetimes. */
            ParticipantInactiveBankTransaction(
                std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry,
                std::shared_ptr<IMoEOverlayParticipantTransferProvider>
                    transfer_provider,
                std::shared_ptr<
                    IMoEOverlayHostAuthorityDeviceBankPublisher>
                    device_bank_publisher,
                std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
                std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
                std::vector<MoEOverlayTierMigration> migrations,
                std::vector<LocalCandidateBank> local_candidates,
                std::vector<std::vector<MoEOverlayParticipantDestinationArrival>> arrivals,
                std::string perf_device)
                : registry_(std::move(registry)),
                  transfer_provider_(std::move(transfer_provider)),
                  device_bank_publisher_(
                      std::move(device_bank_publisher)),
                  previous_(std::move(previous)),
                  candidate_(std::move(candidate)),
                  migrations_(std::move(migrations)),
                  local_candidates_(std::move(local_candidates)),
                  arrivals_(std::move(arrivals)),
                  perf_device_(std::move(perf_device))
            {
            }

            /** @brief Complete, validate, and install every local candidate bank. */
            bool beginPrepare(std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ != Phase::Created)
                {
                    if (error)
                        *error = "ExpertOverlay participant bank has invalid preparation lifecycle";
                    return false;
                }
                phase_ = Phase::Preparing;

                try
                {
                    if (!registry_ || !transfer_provider_ || !previous_ ||
                        !candidate_ ||
                        arrivals_.size() != migrations_.size())
                    {
                        throw std::logic_error(
                            "ExpertOverlay participant bank lost its transaction authorities");
                    }

                    const bool owns_gpu_endpoint = std::any_of(
                        local_candidates_.begin(),
                        local_candidates_.end(),
                        [](const LocalCandidateBank &local)
                        {
                            return local.endpoint &&
                                   local.endpoint->device().is_gpu();
                        });
                    if (owns_gpu_endpoint && !device_bank_publisher_)
                    {
                        throw std::logic_error(
                            "ExpertOverlay GPU candidate has no host-authority device-bank publisher");
                    }

                    for (std::size_t index = 0;
                         index < migrations_.size();
                         ++index)
                    {
                        const auto &migration = migrations_[index];
                        for(const auto& destination:arrivals_[index]) {
                            const auto local=std::find_if(local_candidates_.begin(),local_candidates_.end(),[&](const auto& entry){return entry.participant_id==destination.participant_id;});
                            if(local==local_candidates_.end() || !destination.arrival)
                                throw std::logic_error("Candidate arrival has no exact local participant owner");
                            auto& layer=local->bank.layerForModelLayer(migration.layer_idx);
                            std::string arrival_error;
                            if(layer.cpu_fragment_family) {
                                MoEOverlayPreparedCPUFragmentPayload fragment;
                                if(destination.arrival->destinationEpoch()!=candidate_->epoch ||
                                   !destination.arrival->completeCPUFragment(fragment,&arrival_error))
                                    throw std::runtime_error(arrival_error.empty()?"Candidate CPU fragment arrival has no exact completed epoch":arrival_error);
                                layer.setResidentCPUFragment(migration.expert_id,std::move(fragment));
                            } else {
                                MoEOverlayPreparedExpertPayload payload;
                                if(!destination.arrival->completePayload(payload,&arrival_error))
                                    throw std::runtime_error(arrival_error.empty()?"Candidate whole payload remains incomplete":arrival_error);
                                layer.setResidentExpert(migration.expert_id,std::move(payload));
                            }
                        }

                    }

                    /*
                     * Validate the physical bank against the candidate owner
                     * map, not merely against its own internally consistent
                     * mask. This prevents a missed movement from publishing.
                     */
                    for (const auto &local : local_candidates_)
                    {
                        if (!local.bank.valid(
                                local.participant_id,
                                local.endpoint->device(),
                                local.endpoint->numLayers(),
                                local.endpoint->numExperts(),
                                local.endpoint->firstModelLayer()))
                        {
                            throw std::logic_error(
                                "ExpertOverlay candidate participant bank is incomplete");
                        }
                        for (int layer = local.endpoint->firstModelLayer();
                             layer < local.endpoint->endModelLayer();
                             ++layer)
                        {
                            const auto expected =
                                candidate_->placementOwnerMap().expertMaskForParticipant(
                                    layer,
                                    local.participant_id,
                                    local.endpoint->numExperts());
                            if (local.bank.layerForModelLayer(layer)
                                    .resident_mask != expected)
                            {
                                throw std::logic_error(
                                    "ExpertOverlay candidate bank differs from candidate owner map");
                            }
                        }
                    }

                    /*
                     * Seal and allocate every immutable node before publishing
                     * the first endpoint. This is maintenance-side work and can
                     * be arbitrarily heavier than the fixed-slot pointer stores
                     * below without extending a partially visible commit.
                     */
                    for (auto &local : local_candidates_)
                    {
                        std::string prepare_error;
                        local.prepared_bank =
                            local.endpoint->prepareReadyBank(
                                std::move(local.bank), &prepare_error);
                        if (!local.prepared_bank)
                        {
                            throw std::runtime_error(
                                prepare_error.empty()
                                    ? "ExpertOverlay local candidate bank preparation failed"
                                    : prepare_error);
                        }
                    }

                    const auto publication_start =
                        std::chrono::steady_clock::now();
                    for (auto &local : local_candidates_)
                    {
                        std::string install_error;
                        const auto status = local.endpoint->installReadyBank(
                            std::move(*local.prepared_bank), &install_error);
                        if (status !=
                            MoEOverlayParticipantBankInstallStatus::Installed)
                        {
                            throw std::runtime_error(
                                install_error.empty()
                                    ? "ExpertOverlay local candidate bank installation was not unique"
                                    : install_error);
                        }
                        local.installed_by_this_transaction = true;
                    }
                    const auto publication_elapsed =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() -
                            publication_start)
                            .count();

                    if (owns_gpu_endpoint)
                    {
                        std::string device_error;
                        device_bank_transaction_ =
                            device_bank_publisher_->createTransaction(
                                previous_,
                                candidate_,
                                &device_error);
                        if (!device_bank_transaction_ ||
                            !device_bank_transaction_->beginPrepare(
                                &device_error))
                        {
                            throw std::runtime_error(
                                device_error.empty()
                                    ? "ExpertOverlay GPU inactive-bank preparation could not start"
                                    : device_error);
                        }
                    }
                    else
                    {
                        /* CPU banks are already immutable and exact-addressable
                         * once installed; there is no second device bank build. */
                        phase_ = Phase::Prepared;
                    }
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "participant_candidate_banks_installed",
                        static_cast<double>(local_candidates_.size()),
                        "maintenance",
                        perf_device_,
                        {{"previous_epoch", std::to_string(previous_->epoch)},
                         {"candidate_epoch", std::to_string(candidate_->epoch)}});
                    PerfStatsCollector::recordTimingNs(
                        "moe_overlay_residency",
                        "participant_candidate_bank_preparation",
                        static_cast<uint64_t>(publication_elapsed),
                        "maintenance",
                        perf_device_,
                        {{"previous_epoch", std::to_string(previous_->epoch)},
                         {"candidate_epoch", std::to_string(candidate_->epoch)},
                         {"host_installation", "fixed_slot_atomic"},
                         {"device_publication", owns_gpu_endpoint
                                                    ? "event_driven"
                                                    : "not_applicable"}});
                    return true;
                }
                catch (const std::exception &exception)
                {
                    rollbackInstalledCandidates();
                    failure_ = exception.what();
                    phase_ = Phase::FailedBeforePublication;
                }
                catch (...)
                {
                    rollbackInstalledCandidates();
                    failure_ =
                        "ExpertOverlay participant bank preparation failed with a non-standard exception";
                    phase_ = Phase::FailedBeforePublication;
                }

                if (error)
                    *error = failure_;
                return false;
            }

            /** @brief Host candidate preparation is ready after installation. */
            MoEOverlayResidencyWaveProgress pollPrepare(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::Prepared)
                {
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (phase_ == Phase::Preparing && device_bank_transaction_)
                {
                    const auto progress =
                        device_bank_transaction_->pollPrepare(error);
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                        phase_ = Phase::Prepared;
                    else if (progress == MoEOverlayResidencyWaveProgress::Failed)
                        phase_ = Phase::FailedBeforePublication;
                    return progress;
                }
                if (error)
                {
                    *error = failure_.empty()
                                 ? "ExpertOverlay participant bank was not prepared"
                                 : failure_;
                }
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Publish a host-only candidate after all banks are installed. */
            bool beginPublication(std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ != Phase::Prepared)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay participant bank has invalid publication lifecycle";
                    }
                    return false;
                }

                /* Publication is the irreversible edge. Record it before the
                 * first endpoint submit so a partial fan-out cannot be aborted. */
                phase_ = Phase::Publishing;
                if (device_bank_transaction_ &&
                    !device_bank_transaction_->beginPublication(error))
                {
                    phase_ = Phase::FailedAfterPublication;
                    return false;
                }
                if (!device_bank_transaction_)
                    phase_ = Phase::Published;
                return true;
            }

            /** @brief Return the explicit host-only publication terminal. */
            MoEOverlayResidencyWaveProgress pollPublication(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::Published)
                {
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (phase_ == Phase::Publishing && device_bank_transaction_)
                {
                    const auto progress =
                        device_bank_transaction_->pollPublication(error);
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                        phase_ = Phase::Published;
                    else if (progress == MoEOverlayResidencyWaveProgress::Failed)
                        phase_ = Phase::FailedAfterPublication;
                    return progress;
                }
                if (error)
                {
                    *error =
                        "ExpertOverlay participant publication was not completed";
                }
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Remove only candidate banks installed by this transaction. */
            void abort() noexcept override
            {
                if (phase_ == Phase::Aborting || phase_ == Phase::Aborted ||
                    phase_ == Phase::Retired)
                    return;
                if (phase_ == Phase::Publishing ||
                    phase_ == Phase::Published ||
                    phase_ == Phase::RetirementFencing ||
                    phase_ == Phase::RetirementReady ||
                    phase_ == Phase::FailedAfterPublication)
                {
                    LOG_ERROR(
                        "[ExpertOverlay] Attempted to abort participant banks after publication began");
                    std::terminate();
                }
                if (device_bank_transaction_)
                    device_bank_transaction_->abort();
                rollbackInstalledCandidates();
                phase_ = Phase::Aborting;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "participant_candidate_bank_aborts",
                    1.0,
                    "maintenance",
                    perf_device_,
                    {{"candidate_epoch",
                      candidate_ ? std::to_string(candidate_->epoch) : "0"}});
            }

            /** @brief Host metadata cleanup is safe once transfer aborts are ready. */
            MoEOverlayResidencyWaveProgress pollAbort(
                std::string *error) noexcept override
            {
                if (phase_ == Phase::Aborted)
                {
                    if (error)
                        error->clear();
                    return MoEOverlayResidencyWaveProgress::Ready;
                }
                if (phase_ == Phase::Aborting)
                {
                    if (!device_bank_transaction_)
                    {
                        phase_ = Phase::Aborted;
                        return MoEOverlayResidencyWaveProgress::Ready;
                    }
                    const auto progress =
                        device_bank_transaction_->pollAbort(error);
                    if (progress == MoEOverlayResidencyWaveProgress::Ready)
                        phase_ = Phase::Aborted;
                    return progress;
                }
                if (error)
                    *error = "ExpertOverlay participant abort was not requested";
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Poll every local GPU reader before distributed retirement. */
            MoEOverlayResidencyWaveProgress pollRetirementFence(
                std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (phase_ == Phase::RetirementReady)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (phase_ != Phase::Published &&
                    phase_ != Phase::RetirementFencing)
                {
                    if (error)
                    {
                        *error =
                            "ExpertOverlay participant retirement fence has an invalid lifecycle";
                    }
                    return MoEOverlayResidencyWaveProgress::Failed;
                }
                phase_ = Phase::RetirementFencing;
                const auto progress = device_bank_transaction_
                                          ? device_bank_transaction_
                                                ->pollRetirementFence(error)
                                          : MoEOverlayResidencyWaveProgress::Ready;
                if (progress == MoEOverlayResidencyWaveProgress::Ready)
                    phase_ = Phase::RetirementReady;
                else if (progress == MoEOverlayResidencyWaveProgress::Failed)
                    phase_ = Phase::FailedAfterPublication;
                return progress;
            }

            /** @brief Remove old endpoint banks after the global lease barrier. */
            void retirePrevious() noexcept override
            {
                if (phase_ == Phase::Retired)
                    return;
                if (phase_ != Phase::RetirementReady || !previous_ ||
                    !candidate_)
                    std::terminate();

                if (device_bank_transaction_)
                    device_bank_transaction_->retirePrevious();

                for (auto &local : local_candidates_)
                {
                    if (!local.endpoint->retire(previous_->epoch))
                    {
                        /*
                         * Missing old storage after the ticket barrier means a
                         * second authority mutated this endpoint. There is no
                         * safe recovery that preserves exact epoch ownership.
                         */
                        std::terminate();
                    }
                }


                /*
                 * The participant maps are gone and the outer authority has
                 * already drained every ticket for `previous_`. Bootstrap
                 * slots can therefore join the same recyclable arena used by
                 * arrived lease-backed slots without exposing an overwritten
                 * pointer to inference.
                 */
                transfer_provider_->retirePreviousSources(
                    previous_->epoch, migrations_);
                phase_ = Phase::Retired;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "participant_old_banks_retired",
                    static_cast<double>(local_candidates_.size()),
                    "maintenance",
                    perf_device_,
                    {{"retired_epoch", std::to_string(previous_->epoch)},
                     {"live_epoch", std::to_string(candidate_->epoch)}});
            }

        private:
            /** @brief Roll back the unpublished prefix installed by this owner. */
            void rollbackInstalledCandidates() noexcept
            {
                if (!candidate_)
                    return;
                for (auto &local : local_candidates_)
                {
                    if (!local.installed_by_this_transaction)
                        continue;
                    (void)local.endpoint->abortUnpublished(candidate_->epoch);
                    local.installed_by_this_transaction = false;
                }
            }

            std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry_;
            std::shared_ptr<IMoEOverlayParticipantTransferProvider>
                transfer_provider_;
            std::shared_ptr<IMoEOverlayHostAuthorityDeviceBankPublisher>
                device_bank_publisher_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate_;
            std::vector<MoEOverlayTierMigration> migrations_;
            std::vector<LocalCandidateBank> local_candidates_;
            std::vector<std::vector<MoEOverlayParticipantDestinationArrival>> arrivals_;
            std::unique_ptr<IMoEOverlayInactiveBankTransaction>
                device_bank_transaction_;
            std::string perf_device_;
            std::string failure_;
            Phase phase_ = Phase::Created;
        };

        /** @brief Move every non-null physical operation into composite order. */
        std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
        flattenOperations(MoEOverlayParticipantPreparedTransfers &prepared)
        {
            std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>> flat;
            size_t count=0;
            for(const auto& migration:prepared.migrations)count+=migration.operations.size();
            flat.reserve(count);
            for (auto &migration : prepared.migrations)
            {
                for (auto &piece : migration.operations)
                {
                    if (piece.operation)
                        flat.push_back(std::move(piece.operation));
                }
            }
            return flat;
        }
    } // namespace

    bool MoEOverlayPreparedCPUFragmentPayload::validateRole(
        const MoEExpertShardCohort &cohort, std::size_t member, ExpertTierWeightProjection role,
        const std::shared_ptr<ITensorGemm> &engine, std::string *error)
    {
        if (error)
            error->clear();
        const auto reject = [&](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };
        if (member >= cohort.participants().size() || !engine || static_cast<unsigned>(role) > 2)
            return reject("CPU fragment role requires its original member and actual local engine");
        const auto local = cohort.localManifest(member);
        const auto expected =
            std::find_if(local.projections.begin(), local.projections.end(),
                         [role](const auto &projection) { return projection.projection == role; });
        if (expected == local.projections.end())
            return reject("CPU fragment role is absent from its original source manifest");
        MoEOverlayPreparedWeightSource source;
        if (!resolveMoEOverlayPreparedWeightSource(
                engine, cohort.participants()[member].address.toLocalDeviceId(), source,
                error))
            return false;
        int rows = 0, columns = 0;
        if (source.kind == MoEOverlayPreparedWeightSourceKind::CpuNativeQuantised)
        {
            rows = source.cpu_packed->N;
            columns = source.cpu_packed->K;
        }
        else if (source.kind == MoEOverlayPreparedWeightSourceKind::CpuContiguousFloating)
        {
            rows = source.floating.n;
            columns = source.floating.k;
        }
        else
            return reject("CPU fragment arrival cannot adopt a foreign GPU prepared engine");
        if (rows != expected->N || columns != expected->K || source.format != expected->format)
            return reject("CPU fragment engine differs from its original local projection "
                          "geometry/source format");
        return true;
    }

    MoEOverlayPreparedCPUFragmentPayload::MoEOverlayPreparedCPUFragmentPayload(
        ModelContextId model, std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation,
        std::shared_ptr<const MoEExpertShardCohort> cohort, std::size_t member, int expert,
        std::array<std::shared_ptr<ITensorGemm>, 3> engines,std::uint64_t arrival_epoch)
        : model_(model), preparation_(std::move(preparation)), cohort_(std::move(cohort)),
          member_(member), expert_(expert), arrival_epoch_(arrival_epoch), engines_(std::move(engines))
    {
        if (!model_.value || !preparation_ || !cohort_ ||
            member_ >= cohort_->participants().size() || expert_ < 0 ||
            expert_ >= cohort_->ownership(member_).geometry().experts)
            throw std::invalid_argument("CPU fragment payload requires original "
                                        "model/preparation/cohort/member/expert provenance");
        for (std::size_t role = 0; role < engines_.size(); ++role)
        {
            std::string error;
            if (!validateRole(*cohort_, member_, static_cast<ExpertTierWeightProjection>(role),
                              engines_[role], &error))
                throw std::invalid_argument(error);
        }
        if(arrival_epoch_){
            const auto& member=cohort_->participants()[member_];
            execution_evidence_.emplace(PerfStatsCollector::bindCounter("moe_overlay_residency",
                "completed_cpu_fragment_execution","inference",member.address.toLocalDeviceId().toString(),
                {{"layer",std::to_string(cohort_->source().layer_idx)}, {"expert",std::to_string(expert_)},
                 {"participant",std::to_string(member.participant_id)}, {"world_rank",std::to_string(member.world_rank)},
                 {"member",std::to_string(member_)},{"schema_version","1"}}));
        }
    }

    void MoEOverlayPreparedCPUFragmentPayload::recordCompletedExecution(std::uint64_t invocation_epoch,std::uint64_t routes)const
    {
        if(!arrival_epoch_ || !routes || !PerfStatsCollector::isDomainEnabled("moe_overlay_residency"))return;
        if(!valid() || !execution_evidence_ || invocation_epoch<arrival_epoch_)
            throw std::logic_error("CPU fragment execution must use its original completed arrival in an exact successor bank");
        PerfStatsCollector::addCounterWithSequence(*execution_evidence_,static_cast<double>(routes),
            {arrival_epoch_,invocation_epoch});
    }

    bool MoEOverlayPreparedCPUFragmentPayload::valid() const noexcept
    {
        return model_.value && preparation_ && cohort_ && engines_[0] && engines_[1] && engines_[2];
    }

    bool MoEOverlayPreparedCPUFragmentPayload::sameIdentity(
        const MoEOverlayPreparedCPUFragmentPayload &other) const noexcept
    {
        const auto same = [](const auto &a, const auto &b)
        { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
        if (empty() || other.empty())
            return empty() && other.empty();
        return model_.value == other.model_.value && member_ == other.member_ &&
               expert_ == other.expert_ && arrival_epoch_==other.arrival_epoch_ && same(cohort_, other.cohort_) &&
               same(preparation_, other.preparation_) && same(engines_[0], other.engines_[0]) &&
               same(engines_[1], other.engines_[1]) && same(engines_[2], other.engines_[2]);
    }

    MoEOverlayPreparedExpertArrival::MoEOverlayPreparedExpertArrival(
        ModelContextId model, std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation,
        std::shared_ptr<const MoEExpertShardCohort> cohort, std::size_t member,
        uint64_t candidate_epoch, int expert)
        : fragment_destination_(CPUFragmentDestination{
              model, std::move(preparation), std::move(cohort), member, candidate_epoch, expert}),
          projections_(DeviceMoEProjectionSet::CompleteExpert)
    {
        const auto &destination = *fragment_destination_;
        if (!destination.model.value || !destination.preparation || !destination.cohort ||
            !destination.epoch || destination.member >= destination.cohort->participants().size() ||
            destination.expert < 0 ||
            destination.expert >=
                destination.cohort->ownership(destination.member).geometry().experts)
            throw std::invalid_argument(
                "CPU fragment arrival requires its original fixed-cohort candidate destination");
    }

    bool MoEOverlayPreparedExpertArrival::completeCPUFragment(
        MoEOverlayPreparedCPUFragmentPayload &payload, std::string *error) const
    {
        if (error)
            error->clear();
        std::lock_guard lock(mutex_);
        payload = {};
        if (!fragment_destination_ || !failure_.empty() || !engines_[0] || !engines_[1] ||
            !engines_[2])
        {
            if (error)
                *error =
                    !failure_.empty()
                        ? failure_
                        : "CPU fragment arrival is wrong-arm or lacks its complete local triplet";
            return false;
        }
        const auto &destination = *fragment_destination_;
        try
        {
            payload = MoEOverlayPreparedCPUFragmentPayload(
                destination.model, destination.preparation, destination.cohort, destination.member,
                destination.expert, engines_,destination.epoch);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
    }

    MoEOverlayPreparedExpertArrival::MoEOverlayPreparedExpertArrival(DeviceMoEProjectionSet projections)
        : projections_(projections)
    {
        if (projections != DeviceMoEProjectionSet::CompleteExpert && projections != DeviceMoEProjectionSet::GateUp)
            throw std::invalid_argument("ExpertOverlay arrival requires a known projection family");
    }

    bool MoEOverlayPreparedExpertArrival::publish(
        ExpertTierWeightProjection projection,
        std::shared_ptr<ITensorGemm> engine,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!engine || (projections_ == DeviceMoEProjectionSet::GateUp &&
                        projection == ExpertTierWeightProjection::Down))
        {
            if (error)
                *error = "ExpertOverlay cannot publish an absent or non-movable projection";
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty())
        {
            if (error)
                *error = failure_;
            return false;
        }
        try
        {
            if (fragment_destination_ && !MoEOverlayPreparedCPUFragmentPayload::validateRole(
                *fragment_destination_->cohort, fragment_destination_->member, projection, engine, error))
                return false;
            auto &field = projectionField(engines_, projection);
            if (field && (field.get() != engine.get() || (fragment_destination_ &&
                (field.owner_before(engine) || engine.owner_before(field)))))
            {
                if (error)
                    *error = "ExpertOverlay projection was published with a different engine identity";
                return false;
            }
            field = std::move(engine);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
    }

    bool MoEOverlayPreparedExpertArrival::fail(std::string error)
    {
        if (error.empty())
            error = "ExpertOverlay projection preparation failed";
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty())
            return false;
        failure_ = std::move(error);
        return true;
    }

    bool MoEOverlayPreparedExpertArrival::completePayload(
        MoEOverlayPreparedExpertPayload &payload,
        std::string *error) const
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty())
        {
            if (error)
                *error = failure_;
            payload = {};
            return false;
        }
        if (fragment_destination_)
        {
            if (error) *error = "CPU fragment arrival cannot produce a whole-expert payload";
            payload = {};
            return false;
        }
        if (!engines_[0] || !engines_[1] ||
            (projections_ == DeviceMoEProjectionSet::CompleteExpert && !engines_[2]))
        {
            if (error)
                *error = projections_ == DeviceMoEProjectionSet::CompleteExpert
                    ? "ExpertOverlay destination does not have its complete gate/up/down payload"
                    : "ExpertOverlay destination does not have its complete gate/up payload";
            payload = {};
            return false;
        }
        payload = MoEOverlayPreparedExpertPayload::fromProjections(projections_, engines_);
        return true;
    }

    MoEOverlayPreparedProjectionCompletion::MoEOverlayPreparedProjectionCompletion(
        std::vector<ExpertTierWeightProjectionIntersection> intersections,
        std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
        ExpertTierWeightProjection projection,
        std::function<std::shared_ptr<ITensorGemm>()> engine_provider)
        : intersections_(std::move(intersections)), arrival_(std::move(arrival)),
          projection_(projection), engine_provider_(std::move(engine_provider)),
          claimed_(intersections_.size(), false), ready_(intersections_.size(), false)
    {
        if (!arrival_ || !engine_provider_ || intersections_.empty() ||
            static_cast<unsigned>(projection_) >
                static_cast<unsigned>(ExpertTierWeightProjection::Down) ||
            (projection_ == ExpertTierWeightProjection::Down &&
             arrival_->projections() != DeviceMoEProjectionSet::CompleteExpert))
            throw std::invalid_argument(
                "ExpertOverlay completion requires a known role, coverage and final engine owner");
        const auto &destination = intersections_.front().destination();
        uint64_t covered = 0;
        for (std::size_t i = 0; i < intersections_.size(); ++i)
        {
            const auto &piece = intersections_[i];
            if (piece.empty() || piece.destination() != destination)
                throw std::invalid_argument(
                    "ExpertOverlay completion has empty or foreign destination coverage");
            for (std::size_t j = 0; j < i; ++j)
            {
                const auto &other = intersections_[j];
                if (std::max(piece.firstRow(), other.firstRow()) <
                        std::min(piece.firstRow() + piece.rows(),
                                 other.firstRow() + other.rows()) &&
                    std::max(piece.firstColumn(), other.firstColumn()) <
                        std::min(piece.firstColumn() + piece.columns(),
                                 other.firstColumn() + other.columns()))
                    throw std::invalid_argument("ExpertOverlay completion intersections overlap");
            }
            // All disjoint pieces are contained in this positive int-sized
            // destination, so their area sum is bounded by its uint64 area.
            covered += uint64_t(piece.rows()) * uint64_t(piece.columns());
        }
        if (covered != uint64_t(destination.rows) * uint64_t(destination.columns))
            throw std::invalid_argument(
                "ExpertOverlay completion intersections leave a destination gap");
    }

    void MoEOverlayPreparedProjectionCompletion::claim(std::size_t piece)
    {
        std::lock_guard lock(mutex_);
        if (piece >= claimed_.size() || claimed_[piece])
            throw std::invalid_argument(
                "ExpertOverlay completion piece requires one original operation owner");
        claimed_[piece] = true;
    }

    bool MoEOverlayPreparedProjectionCompletion::failed(std::string *error) const noexcept
    {
        std::lock_guard lock(mutex_);
        if (failure_.empty())
            return false;
        if (error)
            *error = failure_;
        return true;
    }

    void MoEOverlayPreparedProjectionCompletion::fail(std::string error) noexcept
    {
        std::lock_guard lock(mutex_);
        if (failure_.empty())
            failure_ = std::move(error);
        (void)arrival_->fail(failure_);
    }

    bool MoEOverlayPreparedProjectionCompletion::ready(std::size_t piece,
                                                       std::string *error) noexcept
    {
        std::lock_guard lock(mutex_);
        if (!failure_.empty())
        {
            if (error)
                *error = failure_;
            return false;
        }
        ready_[piece] = true; // Only a uniquely claimed operation can call this private edge.
        if (published_ ||
            !std::all_of(ready_.begin(), ready_.end(), [](bool value) { return value; }))
            return true;
        try
        {
            auto engine = engine_provider_();
            if (!engine || !arrival_->publish(projection_, std::move(engine), &failure_))
            {
                if (failure_.empty())
                    failure_ =
                        "ExpertOverlay completed intersections lack the final destination engine";
            }
            else
            {
                published_ = true;
                return true;
            }
        }
        catch (const std::exception &exception)
        {
            failure_ = exception.what();
        }
        catch (...)
        {
            failure_ = "ExpertOverlay final destination engine provider failed";
        }
        (void)arrival_->fail(failure_);
        if (error)
            *error = failure_;
        return false;
    }

    MoEOverlayPreparedProjectionOperation::MoEOverlayPreparedProjectionOperation(
        std::unique_ptr<IMoEOverlayTierTransferOperation> operation,
        std::shared_ptr<MoEOverlayPreparedProjectionCompletion> completion, std::size_t piece)
        : operation_(std::move(operation)), completion_(std::move(completion)), piece_(piece)
    {
        if (!operation_ || !completion_)
            throw std::invalid_argument(
                "ExpertOverlay prepared piece requires physical operation and checked completion");
        completion_->claim(piece_);
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayPreparedProjectionOperation::poll(std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (aborted_ || completion_->failed(error))
        {
            if (aborted_ && error)
                *error = "ExpertOverlay prepared projection was aborted";
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (ready_)
            return MoEOverlayResidencyWaveProgress::Ready;
        std::string operation_error;
        const auto progress = operation_->poll(&operation_error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return progress;
        if (progress == MoEOverlayResidencyWaveProgress::Failed)
        {
            if (operation_error.empty())
                operation_error = "ExpertOverlay physical projection transfer failed";
            completion_->fail(operation_error);
            if (error)
                *error = operation_error;
            return progress;
        }
        if (!completion_->ready(piece_, error))
            return MoEOverlayResidencyWaveProgress::Failed;
        ready_ = true;
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    void MoEOverlayPreparedProjectionOperation::abort() noexcept
    {
        if (aborted_)
            return;
        aborted_ = true;
        completion_->fail("ExpertOverlay prepared projection was discarded before publication");
        operation_->abort();
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayPreparedProjectionOperation::pollAbort(std::string *error) noexcept
    {
        if (!aborted_)
        {
            if (error)
                *error = "ExpertOverlay prepared projection abort was not requested";
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        return operation_->pollAbort(error);
    }

    std::optional<ExpertTierProjectionTransferMeasurement>
    MoEOverlayPreparedProjectionOperation::completedMeasurement() const noexcept
    {
        if (!ready_ || aborted_ || completion_->failed(nullptr))
            return std::nullopt;
        return operation_->completedMeasurement();
    }

    std::optional<std::uint64_t> MoEOverlayPreparedProjectionOperation::quotedPayloadBytes() const noexcept
    { return operation_->quotedPayloadBytes(); }


    MoEOverlayParticipantPreparedWaveFactory::
        MoEOverlayParticipantPreparedWaveFactory(Config config)
        : config_(std::move(config))
    {
        if (!config_.registry || !config_.transfer_provider)
        {
            throw std::invalid_argument(
                "ExpertOverlay participant wave factory requires registry and transfer provider authorities");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";
    }

    MoEOverlayParticipantPreparedWaveFactory::MoEOverlayParticipantPreparedWaveFactory(
        Config config, RecordedComponentSource)
        : MoEOverlayParticipantPreparedWaveFactory(std::move(config))
    {
        requires_original_source_ = false;
    }

    MoEOverlayTierPreparedWave MoEOverlayParticipantPreparedWaveFactory::prepare(
        const MoEOverlayResidencyTransaction &transaction)
    {
        MoEOverlayTierPreparedWave result;
        if (!transaction.valid() || transaction.empty())
        {
            result.error =
                "ExpertOverlay participant factory requires a valid non-empty transaction";
            return result;
        }

        // Public runtime construction always requires the real source owner,
        // including zero-ID misuse. Recorded component tests have a private
        // constructor boundary and cannot mint a model-source capability.
        std::optional<MoEOverlayParticipantResidencyRegistry::PreparedSourceUse> original_source;
        if (requires_original_source_)
        {
            original_source = config_.registry->acquireOriginalPreparedSource(&result.error);
            if (!original_source)
                return result;
        }
        std::vector<LocalCandidateBank> local_candidates;
        std::string candidate_error;
        const auto candidate_status = buildLocalCandidates(
            config_.registry,
            transaction,
            local_candidates,
            &candidate_error);
        if (candidate_status != MoEOverlayResidencyStageStartStatus::Started)
        {
            result.status = candidate_status;
            result.error = std::move(candidate_error);
            return result;
        }

        const auto local_ids = config_.registry->localParticipantIds();
        MoEOverlayParticipantPreparedTransfers prepared;
        try
        {
            prepared = config_.transfer_provider->prepareTransfers(
                transaction, local_ids);
        }
        catch (const std::exception &exception)
        {
            result.error = exception.what();
            return result;
        }
        catch (...)
        {
            result.error =
                "ExpertOverlay transfer provider threw a non-standard exception";
            return result;
        }

        std::vector<std::vector<MoEOverlayParticipantDestinationArrival>> arrivals(transaction.migrations.size());
        for (std::size_t index = 0;
             index < std::min(
                 prepared.migrations.size(), transaction.migrations.size());
             ++index)
        {
            arrivals[index]=prepared.migrations[index].destination_arrivals;
        }

        auto inactive_bank = std::make_unique<ParticipantInactiveBankTransaction>(
            config_.registry,
            config_.transfer_provider,
            config_.device_bank_publisher,
            transaction.previous,
            transaction.candidate,
            transaction.migrations,
            std::move(local_candidates),
            std::move(arrivals),
            config_.perf_device);

        const bool deferred_shape_ok =
            prepared.status != MoEOverlayResidencyStageStartStatus::Deferred ||
            prepared.migrations.empty();
        if (prepared.status == MoEOverlayResidencyStageStartStatus::Deferred &&
            deferred_shape_ok)
        {
            result.status = prepared.status;
            result.error = std::move(prepared.error);
            return result;
        }

        bool started_shape_ok =
            prepared.status == MoEOverlayResidencyStageStartStatus::Started &&
            prepared.migrations.size() == transaction.migrations.size();
        if (started_shape_ok)
        {
            for (std::size_t index = 0;
                 index < prepared.migrations.size();
                 ++index)
            {
                const auto& migration=transaction.migrations[index];
                const auto source=MoEExpertPlacementTarget::fromPlacement(migration.source);
                const auto destination=MoEExpertPlacementTarget::fromPlacement(migration.destination);
                if(!source || !destination){started_shape_ok=false;break;}
                const auto memberIndex=[](const MoEExpertPlacementTarget& target,int id)->std::optional<size_t>{
                    for(size_t member=0;member<target.participantCount();++member)if(target.participantId(member)==id)return member;
                    return std::nullopt;
                };
                std::optional<std::tuple<unsigned,size_t,size_t>> prior;
                const auto& record=prepared.migrations[index];
                if (migration.source.shard_cohort || migration.destination.shard_cohort)
                {
                    const auto &cohort = migration.source.shard_cohort ? migration.source.shard_cohort : migration.destination.shard_cohort;
                    const auto &manifest = cohort->source();
                    const auto expected = physicalMoEOverlayProjectionLayouts(migration, manifest,
                        cohort->ownership(0).geometry().experts, transaction.previous->placementOwnerMap());
                    if (!record.measurement_geometry || record.measurement_geometry->targets() != expected)
                    { started_shape_ok = false; break; }
                    for (std::size_t role = 0; role < 3; ++role)
                        if (record.measurement_geometry->formats()[role] != manifest.projectionForRole(static_cast<ExpertTierWeightProjection>(role)).format)
                            started_shape_ok = false;
                    if (!started_shape_ok) break;
                }
                for(const auto& piece:record.operations) {
                    const auto src=memberIndex(*source,piece.source_participant),dst=memberIndex(*destination,piece.destination_participant);
                    const auto role=static_cast<unsigned>(piece.projection);
                    if(!piece.operation || !src || !dst || role>=kMoEOverlayExpertProjectionCount){started_shape_ok=false;break;}
                    const auto order=std::tuple{role,*dst,*src};
                    if(prior && !(*prior<order)){started_shape_ok=false;break;}prior=order;
                }
                if(!started_shape_ok)break;
                size_t local_arrival=0;
                for(size_t member=0;member<destination->participantCount();++member) {
                    const int id=destination->participantId(member);
                    if(!containsParticipant(config_.registry->localParticipantIds(),id))continue;
                    if(local_arrival>=record.destination_arrivals.size() || record.destination_arrivals[local_arrival].participant_id!=id ||
                       !record.destination_arrivals[local_arrival].arrival){started_shape_ok=false;break;}
                    const auto& arrival=record.destination_arrivals[local_arrival++].arrival;
                    if(arrival->isCPUJointFragment()!=config_.registry->isCPUJointParticipant(id) ||
                       (arrival->isCPUJointFragment() && arrival->destinationEpoch()!=transaction.candidate->epoch)) {started_shape_ok=false;break;}
                    for(size_t previous=0;previous+1<local_arrival;++previous)
                        if(record.destination_arrivals[previous].arrival.get()==arrival.get()){started_shape_ok=false;break;}
                    // Each required role has real/protocol work. Exact native
                    // intersection coverage remains the all-piece completion's proof.
                    for(unsigned role=0;role<kMoEOverlayExpertProjectionCount;++role) {
                        if(arrival->projections()==DeviceMoEProjectionSet::GateUp && role==unsigned(ExpertTierWeightProjection::Down))continue;
                        if(std::none_of(record.operations.begin(),record.operations.end(),[&](const auto& piece){return piece.destination_participant==id && unsigned(piece.projection)==role;}))started_shape_ok=false;
                    }
                }
                if(local_arrival!=record.destination_arrivals.size()){started_shape_ok=false;}
                const auto local_ids=config_.registry->localParticipantIds();
                const bool local_source=std::any_of(local_ids.begin(),local_ids.end(),[&](int id){return bool(memberIndex(*source,id));});
                if((local_source || local_arrival) && record.operations.empty())started_shape_ok=false;
                if(!started_shape_ok)break;

            }
        }

        // The fabric retains every canonical expected global piece, including
        // ranks with only protocol work. Preserve identity before moving operations.
        const bool described = std::any_of(prepared.migrations.begin(), prepared.migrations.end(),
            [](const auto &migration) { return bool(migration.measurement_geometry); });
        if (described)
        {
            result.measurement_geometries.reserve(prepared.migrations.size());
            for (std::size_t migration = 0; migration < prepared.migrations.size(); ++migration)
            {
                const auto &record = prepared.migrations[migration];
                const auto &geometry = record.measurement_geometry;
                if (!geometry || geometry->pieces().size() != record.operations.size())
                { started_shape_ok = false; break; }
                result.measurement_geometries.push_back(geometry);
                for (std::size_t piece = 0; piece < record.operations.size(); ++piece)
                {
                    const auto &operation = record.operations[piece];
                    const auto &identity = geometry->pieces()[piece];
                    if (identity.projection != operation.projection ||
                        identity.source_participant != operation.source_participant ||
                        identity.destination_participant != operation.destination_participant)
                    { started_shape_ok = false; break; }
                    if (operation.operation)
                        result.measurement_operations.push_back({migration, piece,
                            operation.requires_local_measurement, operation.observer_world_rank});
                }
            }
        }
        result.transfers = flattenOperations(prepared);
        result.inactive_bank = std::move(inactive_bank);
        if (!deferred_shape_ok ||
            (prepared.status == MoEOverlayResidencyStageStartStatus::Started &&
             !started_shape_ok))
        {
            result.status = MoEOverlayResidencyStageStartStatus::Failed;
            result.error = prepared.error.empty()
                               ? "ExpertOverlay transfer provider returned an invalid ownership shape"
                               : std::move(prepared.error);
            return result;
        }

        result.status = prepared.status;
        result.error = std::move(prepared.error);
        return result;
    }
} // namespace llaminar2
