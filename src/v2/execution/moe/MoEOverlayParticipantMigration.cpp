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

#include "../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** @brief Select the shared-pointer field named by one wire projection. */
        std::shared_ptr<ITensorGemm> &projectionField(
            MoEOverlayPreparedExpertTriplet &triplet,
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
                    transaction.previous->owner_map.participantForId(
                        participant_id);
                const auto *candidate_participant =
                    transaction.candidate->owner_map.participantForId(
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
            for (const auto &migration : transaction.migrations)
            {
                const auto found = std::find_if(
                    candidates.begin(),
                    candidates.end(),
                    [&migration](const auto &candidate)
                    {
                        return candidate.participant_id ==
                               migration.source.owner_participant;
                    });
                if (found == candidates.end())
                    continue;
                if (migration.layer_idx < 0 ||
                    migration.layer_idx >= found->endpoint->numLayers() ||
                    migration.expert_id < 0 ||
                    migration.expert_id >= found->endpoint->numExperts())
                {
                    if (error)
                        *error = "ExpertOverlay migration departure is outside local bank geometry";
                    candidates.clear();
                    return MoEOverlayResidencyStageStartStatus::Failed;
                }
                found->bank.layers[static_cast<std::size_t>(migration.layer_idx)]
                    .clearExpert(migration.expert_id);
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
            /** @brief Take candidate clones and transfer-arrival lifetimes. */
            ParticipantInactiveBankTransaction(
                std::shared_ptr<MoEOverlayParticipantResidencyRegistry> registry,
                std::shared_ptr<IMoEOverlayParticipantTransferProvider>
                    transfer_provider,
                std::shared_ptr<const MoEOverlayResidencySnapshot> previous,
                std::shared_ptr<const MoEOverlayResidencySnapshot> candidate,
                std::vector<MoEOverlayTierMigration> migrations,
                std::vector<LocalCandidateBank> local_candidates,
                std::vector<std::shared_ptr<MoEOverlayPreparedExpertArrival>>
                    arrivals,
                std::string perf_device)
                : registry_(std::move(registry)),
                  transfer_provider_(std::move(transfer_provider)),
                  previous_(std::move(previous)),
                  candidate_(std::move(candidate)),
                  migrations_(std::move(migrations)),
                  local_candidates_(std::move(local_candidates)),
                  arrivals_(std::move(arrivals)),
                  perf_device_(std::move(perf_device))
            {
            }

            /** @brief Complete, validate, and install every local candidate bank. */
            bool beginCommit(std::string *error) noexcept override
            {
                if (error)
                    error->clear();
                if (commit_attempted_ || aborted_ || retired_)
                {
                    if (error)
                        *error = "ExpertOverlay participant bank has invalid commit lifecycle";
                    return false;
                }
                commit_attempted_ = true;

                try
                {
                    if (!registry_ || !transfer_provider_ || !previous_ ||
                        !candidate_ ||
                        arrivals_.size() != migrations_.size())
                    {
                        throw std::logic_error(
                            "ExpertOverlay participant bank lost its transaction authorities");
                    }

                    for (std::size_t index = 0;
                         index < migrations_.size();
                         ++index)
                    {
                        const auto &migration = migrations_[index];
                        const auto local_candidate = std::find_if(
                            local_candidates_.begin(),
                            local_candidates_.end(),
                            [&migration](const auto &entry)
                            {
                                return entry.participant_id ==
                                       migration.destination.owner_participant;
                            });
                        if (local_candidate == local_candidates_.end())
                            continue;
                        if (!arrivals_[index])
                        {
                            throw std::logic_error(
                                "ExpertOverlay local destination has no prepared arrival authority");
                        }

                        MoEOverlayPreparedExpertTriplet triplet;
                        std::string arrival_error;
                        if (!arrivals_[index]->completeTriplet(
                                triplet, &arrival_error))
                        {
                            throw std::runtime_error(
                                arrival_error.empty()
                                    ? "ExpertOverlay destination triplet is incomplete after transfer readiness"
                                    : arrival_error);
                        }
                        local_candidate
                            ->bank.layers[static_cast<std::size_t>(
                                migration.layer_idx)]
                            .setResidentExpert(
                                migration.expert_id, std::move(triplet));
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
                                local.endpoint->numExperts()))
                        {
                            throw std::logic_error(
                                "ExpertOverlay candidate participant bank is incomplete");
                        }
                        for (int layer = 0;
                             layer < local.endpoint->numLayers();
                             ++layer)
                        {
                            const auto expected =
                                candidate_->owner_map.expertMaskForParticipant(
                                    layer,
                                    local.participant_id,
                                    local.endpoint->numExperts());
                            if (local.bank.layers[static_cast<std::size_t>(layer)]
                                    .resident_mask != expected)
                            {
                                throw std::logic_error(
                                    "ExpertOverlay candidate bank differs from candidate owner map");
                            }
                        }
                    }

                    for (auto &local : local_candidates_)
                    {
                        std::string install_error;
                        const auto status = local.endpoint->installReadyBank(
                            local.bank, &install_error);
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

                    committed_ = true;
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "participant_candidate_banks_installed",
                        static_cast<double>(local_candidates_.size()),
                        "maintenance",
                        perf_device_,
                        {{"previous_epoch", std::to_string(previous_->epoch)},
                         {"candidate_epoch", std::to_string(candidate_->epoch)}});
                    return true;
                }
                catch (const std::exception &exception)
                {
                    rollbackInstalledCandidates();
                    failure_ = exception.what();
                }
                catch (...)
                {
                    rollbackInstalledCandidates();
                    failure_ =
                        "ExpertOverlay participant bank commit failed with a non-standard exception";
                }

                if (error)
                    *error = failure_;
                return false;
            }

            /** @brief Metadata publication is ready immediately after installation. */
            MoEOverlayResidencyWaveProgress pollCommit(
                std::string *error) noexcept override
            {
                if (committed_ && !aborted_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (error)
                {
                    *error = failure_.empty()
                                 ? "ExpertOverlay participant bank was not committed"
                                 : failure_;
                }
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Remove only candidate banks installed by this transaction. */
            void abort() noexcept override
            {
                if (aborted_ || retired_)
                    return;
                rollbackInstalledCandidates();
                aborted_ = true;
                committed_ = false;
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
                if (aborted_)
                    return MoEOverlayResidencyWaveProgress::Ready;
                if (error)
                    *error = "ExpertOverlay participant abort was not requested";
                return MoEOverlayResidencyWaveProgress::Failed;
            }

            /** @brief Remove old endpoint banks after the global lease barrier. */
            void retirePrevious() noexcept override
            {
                if (retired_ || aborted_)
                    return;
                if (!committed_ || !previous_ || !candidate_)
                    std::terminate();

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
                retired_ = true;
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
            std::shared_ptr<const MoEOverlayResidencySnapshot> previous_;
            std::shared_ptr<const MoEOverlayResidencySnapshot> candidate_;
            std::vector<MoEOverlayTierMigration> migrations_;
            std::vector<LocalCandidateBank> local_candidates_;
            std::vector<std::shared_ptr<MoEOverlayPreparedExpertArrival>>
                arrivals_;
            std::string perf_device_;
            std::string failure_;
            bool commit_attempted_ = false;
            bool committed_ = false;
            bool aborted_ = false;
            bool retired_ = false;
        };

        /** @brief Move every non-null physical operation into composite order. */
        std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>>
        flattenOperations(MoEOverlayParticipantPreparedTransfers &prepared)
        {
            std::vector<std::unique_ptr<IMoEOverlayTierTransferOperation>> flat;
            flat.reserve(
                prepared.migrations.size() *
                kMoEOverlayExpertProjectionCount);
            for (auto &migration : prepared.migrations)
            {
                for (auto &projection : migration.projections)
                {
                    if (projection)
                        flat.push_back(std::move(projection));
                }
            }
            return flat;
        }
    } // namespace

    bool MoEOverlayPreparedExpertArrival::publish(
        ExpertTierWeightProjection projection,
        std::shared_ptr<ITensorGemm> engine,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!engine)
        {
            if (error)
                *error = "ExpertOverlay cannot publish a null prepared projection";
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
            auto &field = projectionField(triplet_, projection);
            if (field && field.get() != engine.get())
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

    bool MoEOverlayPreparedExpertArrival::completeTriplet(
        MoEOverlayPreparedExpertTriplet &triplet,
        std::string *error) const
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(mutex_);
        if (!failure_.empty())
        {
            if (error)
                *error = failure_;
            triplet = {};
            return false;
        }
        if (!triplet_.complete())
        {
            if (error)
                *error = "ExpertOverlay destination does not have a complete gate/up/down triplet";
            triplet = {};
            return false;
        }
        triplet = triplet_;
        return true;
    }

    MoEOverlayPreparedProjectionOperation::
        MoEOverlayPreparedProjectionOperation(
            std::unique_ptr<IMoEOverlayTierTransferOperation> operation,
            std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival,
            ExpertTierWeightProjection projection,
            std::shared_ptr<ITensorGemm> engine)
        : operation_(std::move(operation)),
          arrival_(std::move(arrival)),
          projection_(projection),
          engine_(std::move(engine))
    {
        if (!operation_ || !arrival_ || !engine_)
        {
            throw std::invalid_argument(
                "ExpertOverlay prepared projection requires operation, arrival, and engine ownership");
        }
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayPreparedProjectionOperation::poll(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (aborted_)
        {
            if (error)
                *error = "ExpertOverlay prepared projection was aborted";
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        if (published_)
            return MoEOverlayResidencyWaveProgress::Ready;

        std::string operation_error;
        const auto progress = operation_->poll(&operation_error);
        if (progress == MoEOverlayResidencyWaveProgress::Pending)
            return progress;
        if (progress == MoEOverlayResidencyWaveProgress::Failed)
        {
            if (operation_error.empty())
                operation_error = "ExpertOverlay physical projection transfer failed";
            (void)arrival_->fail(operation_error);
            if (error)
                *error = std::move(operation_error);
            return progress;
        }

        std::string publication_error;
        if (!arrival_->publish(
                projection_, engine_, &publication_error))
        {
            if (publication_error.empty())
                publication_error = "ExpertOverlay prepared projection publication failed";
            (void)arrival_->fail(publication_error);
            if (error)
                *error = std::move(publication_error);
            return MoEOverlayResidencyWaveProgress::Failed;
        }
        published_ = true;
        return MoEOverlayResidencyWaveProgress::Ready;
    }

    void MoEOverlayPreparedProjectionOperation::abort() noexcept
    {
        if (aborted_)
            return;
        aborted_ = true;
        (void)arrival_->fail(
            "ExpertOverlay prepared projection was discarded before publication");
        operation_->abort();
    }

    MoEOverlayResidencyWaveProgress
    MoEOverlayPreparedProjectionOperation::pollAbort(
        std::string *error) noexcept
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
    MoEOverlayPreparedProjectionOperation::completedMeasurement()
        const noexcept
    {
        if (!published_ || aborted_ || !operation_)
            return std::nullopt;
        return operation_->completedMeasurement();
    }

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

    MoEOverlayTierPreparedWave
    MoEOverlayParticipantPreparedWaveFactory::prepare(
        const MoEOverlayResidencyTransaction &transaction)
    {
        MoEOverlayTierPreparedWave result;
        if (!transaction.valid() || transaction.empty())
        {
            result.error =
                "ExpertOverlay participant factory requires a valid non-empty transaction";
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

        std::vector<std::shared_ptr<MoEOverlayPreparedExpertArrival>> arrivals(
            transaction.migrations.size());
        for (std::size_t index = 0;
             index < std::min(
                 prepared.migrations.size(), transaction.migrations.size());
             ++index)
        {
            arrivals[index] =
                prepared.migrations[index].destination_arrival;
        }

        auto inactive_bank = std::make_unique<ParticipantInactiveBankTransaction>(
            config_.registry,
            config_.transfer_provider,
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
                const bool local_destination = containsParticipant(
                    local_ids,
                    transaction.migrations[index]
                        .destination.owner_participant);
                const bool exact_arrival =
                    static_cast<bool>(prepared.migrations[index]
                                          .destination_arrival) ==
                    local_destination;
                const bool complete_operations = std::all_of(
                    prepared.migrations[index].projections.begin(),
                    prepared.migrations[index].projections.end(),
                    [](const auto &operation)
                    {
                        return operation != nullptr;
                    });
                if (!exact_arrival || !complete_operations)
                {
                    started_shape_ok = false;
                    break;
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
