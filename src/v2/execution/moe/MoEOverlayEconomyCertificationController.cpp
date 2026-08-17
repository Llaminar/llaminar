/**
 * @file MoEOverlayEconomyCertificationController.cpp
 * @brief Local or all-rank measured-economy composition and installation.
 */

#include "MoEOverlayEconomyCertificationController.h"

#include "../../utils/FNV1a.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Hash one scalar without native padding or endianness. */
        void hashUnsigned(std::uint64_t &hash, std::uint64_t value) noexcept
        {
            std::array<unsigned char, sizeof(value)> bytes{};
            for (std::size_t index = 0; index < bytes.size(); ++index)
            {
                bytes[index] = static_cast<unsigned char>(value & 0xffu);
                value >>= 8u;
            }
            hash = fnv1a64(bytes.data(), bytes.size(), hash);
        }

        /** @brief Hash one length-delimited identity string. */
        void hashString(
            std::uint64_t &hash,
            const std::string &value) noexcept
        {
            hashUnsigned(hash, static_cast<std::uint64_t>(value.size()));
            hash = fnv1a64(value.data(), value.size(), hash);
        }

        /** @return Stable diagnostic name for one dense production phase. */
        const char *serviceSourceName(std::size_t source) noexcept
        {
            constexpr std::array<const char *,
                                 kExpertHistogramProductionSourceCount>
                names{"decode", "prefill", "grouped_verifier"};
            return source < names.size() ? names[source] : "invalid";
        }
    } // namespace

    MoEOverlayEconomyCertificationController::
        MoEOverlayEconomyCertificationController(Config config)
        : config_(std::move(config))
    {
        if (!config_.calibration || !config_.registry ||
            !config_.layer_catalog || !config_.authority ||
            !config_.economy_policy.valid() ||
            !config_.authority->migrationEnabled() ||
            config_.authority->hasEconomyCertification() ||
            !validExpertHistogramProductionSourceMask(
                config_.active_sources) ||
            config_.active_sources !=
                config_.calibration->requiredSources())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification requires complete uncertified dynamic dependencies with one consistent runtime phase mask");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";

        const auto snapshot = config_.authority->snapshot();
        if (!snapshot || !snapshot->valid() ||
            config_.model_metadata.num_layers !=
                snapshot->layered_ownership.layerCount() ||
            config_.model_metadata.num_experts !=
                snapshot->layered_ownership.expertCount() ||
            config_.layer_catalog->completeExpertBytesPerLayer().size() !=
                static_cast<std::size_t>(config_.model_metadata.num_layers))
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification geometry disagrees with the live authority");
        }

        std::vector<int> global_participants;
        global_participants.reserve(snapshot->owner_map.participants().size());
        for (const auto &participant : snapshot->owner_map.participants())
            global_participants.push_back(participant.participant_id);
        std::sort(global_participants.begin(), global_participants.end());
        if (!config_.evidence_exchange &&
            config_.registry->localParticipantIds() != global_participants)
        {
            throw std::invalid_argument(
                "Process-local ExpertOverlay certification requires every topology participant on this rank");
        }
        if (config_.evidence_exchange)
        {
            if (config_.evidence_exchange->worldSize() < 2 ||
                config_.evidence_exchange->worldRank() < 0 ||
                config_.evidence_exchange->worldRank() >=
                    config_.evidence_exchange->worldSize())
            {
                throw std::invalid_argument(
                    "Distributed ExpertOverlay certification requires valid evidence-exchange membership");
            }
            for (const auto &participant :
                 snapshot->owner_map.participants())
            {
                if (!participant.world_rank_known ||
                    participant.world_rank < 0 ||
                    participant.world_rank >=
                        config_.evidence_exchange->worldSize())
                {
                    throw std::invalid_argument(
                        "Distributed ExpertOverlay certification requires resolved participant rank ownership");
                }
            }
        }
    }

    void MoEOverlayEconomyCertificationController::poll() noexcept
    {
        polls_.fetch_add(1, std::memory_order_relaxed);
        if (!healthy() ||
            state() == MoEOverlayEconomyCertificationState::Complete ||
            state() == MoEOverlayEconomyCertificationState::Stopped)
        {
            return;
        }

        try
        {
            if (stop_requested_.load(std::memory_order_acquire))
            {
                if (state() ==
                        MoEOverlayEconomyCertificationState::
                            ExchangingServiceReadiness &&
                    config_.evidence_exchange &&
                    !config_.evidence_exchange->idle())
                {
                    bool discarded = false;
                    std::string error;
                    const auto progress =
                        config_.evidence_exchange->pollServiceReadiness(
                            &discarded, &error);
                    if (progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        return;
                    }
                    if (progress != MoEOverlayResidencyWaveProgress::Ready)
                    {
                        fail(
                            error.empty()
                                ? "ExpertOverlay certification could not drain its service readiness exchange"
                                : std::move(error));
                        return;
                    }
                }
                if (state() ==
                        MoEOverlayEconomyCertificationState::
                            ExchangingServiceEvidence &&
                    config_.evidence_exchange &&
                    !config_.evidence_exchange->idle())
                {
                    std::vector<MoEOverlayParticipantLayerServiceTotals>
                        discarded;
                    std::string error;
                    const auto progress =
                        config_.evidence_exchange->pollService(
                            &discarded, &error);
                    if (progress ==
                        MoEOverlayResidencyWaveProgress::Pending)
                    {
                        return;
                    }
                    if (progress != MoEOverlayResidencyWaveProgress::Ready)
                    {
                        fail(
                            error.empty()
                                ? "ExpertOverlay certification could not drain its service evidence exchange"
                                : std::move(error));
                        return;
                    }
                }
                config_.calibration->requestStop();
                config_.calibration->poll();
                if (!config_.calibration->healthy())
                {
                    fail(config_.calibration->failureMessage());
                    return;
                }
                const auto calibration_state = config_.calibration->state();
                if (calibration_state ==
                        MoEOverlayEconomyCalibrationState::Stopped ||
                    calibration_state ==
                        MoEOverlayEconomyCalibrationState::Complete)
                {
                    expanded_migration_measurements_.reset();
                    ready_local_service_evidence_.reset();
                    state_.store(
                        MoEOverlayEconomyCertificationState::Stopped,
                        std::memory_order_release);
                }
                return;
            }

            if (state() ==
                MoEOverlayEconomyCertificationState::
                    ExchangingServiceReadiness)
            {
                bool all_ranks_ready = false;
                std::string error;
                const auto progress =
                    config_.evidence_exchange->pollServiceReadiness(
                        &all_ranks_ready, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    return;
                if (progress != MoEOverlayResidencyWaveProgress::Ready)
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay service readiness exchange failed"
                            : std::move(error));
                }
                if (!all_ranks_ready)
                {
                    ready_local_service_evidence_.reset();
                    state_.store(
                        MoEOverlayEconomyCertificationState::
                            AwaitingServiceEvidence,
                        std::memory_order_release);
                    return;
                }
                if (!ready_local_service_evidence_)
                {
                    throw std::logic_error(
                        "ExpertOverlay all-rank service readiness succeeded without a retained local snapshot");
                }
                if (!config_.evidence_exchange->beginService(
                        *ready_local_service_evidence_, &error))
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay could not begin its service evidence exchange"
                            : std::move(error));
                }
                state_.store(
                    MoEOverlayEconomyCertificationState::
                        ExchangingServiceEvidence,
                    std::memory_order_release);
                return;
            }

            if (state() ==
                MoEOverlayEconomyCertificationState::
                    ExchangingServiceEvidence)
            {
                std::vector<MoEOverlayParticipantLayerServiceTotals>
                    complete_service;
                std::string error;
                const auto progress =
                    config_.evidence_exchange->pollService(
                        &complete_service, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    return;
                if (progress != MoEOverlayResidencyWaveProgress::Ready)
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay service evidence exchange failed"
                            : std::move(error));
                }
                const auto snapshot = config_.authority->snapshot();
                if (!snapshot || !snapshot->valid())
                {
                    throw std::logic_error(
                        "ExpertOverlay service evidence exchange lost its live authority snapshot");
                }
                std::vector<int> global_participant_ids;
                global_participant_ids.reserve(
                    snapshot->owner_map.participants().size());
                for (const auto &participant :
                     snapshot->owner_map.participants())
                {
                    global_participant_ids.push_back(
                        participant.participant_id);
                }
                std::sort(
                    global_participant_ids.begin(),
                    global_participant_ids.end());
                if (!serviceEvidenceCoverage(
                         complete_service, global_participant_ids)
                         .complete)
                {
                    throw std::logic_error(
                        "ExpertOverlay all-rank readiness produced incomplete global service evidence");
                }
                ready_local_service_evidence_.reset();
                installCompleteEvidence(std::move(complete_service));
                return;
            }
            if (state() ==
                MoEOverlayEconomyCertificationState::CalibratingMovement)
            {
                movement_calibration_polls_.fetch_add(
                    1, std::memory_order_relaxed);
                config_.calibration->poll();
                if (!config_.calibration->healthy())
                {
                    fail(config_.calibration->failureMessage());
                    return;
                }
                if (config_.calibration->state() !=
                    MoEOverlayEconomyCalibrationState::Complete)
                {
                    return;
                }
                const auto *representatives =
                    config_.calibration->sealedMeasurements();
                if (!representatives)
                {
                    throw std::logic_error(
                        "Completed ExpertOverlay calibration omitted its sealed evidence");
                }
                expanded_migration_measurements_ =
                    config_.layer_catalog->expand(*representatives);
                state_.store(
                    MoEOverlayEconomyCertificationState::
                        AwaitingServiceEvidence,
                    std::memory_order_release);
                return;
            }

            service_snapshot_attempts_.fetch_add(
                1, std::memory_order_relaxed);
            std::vector<MoEOverlayParticipantLayerServiceTotals> raw_service;
            const bool snapshot_ready =
                config_.registry->trySnapshotServiceMeasurements(
                    &raw_service);
            if (!snapshot_ready)
            {
                service_snapshot_contentions_.fetch_add(
                    1, std::memory_order_relaxed);
            }
            const auto local_coverage = snapshot_ready
                                            ? serviceEvidenceCoverage(
                                                  raw_service,
                                                  config_.registry
                                                      ->localParticipantIds())
                                            : ServiceEvidenceCoverage{};
            const bool local_ready =
                snapshot_ready && local_coverage.complete;
            if (!local_ready)
            {
                incomplete_service_snapshots_.fetch_add(
                    1, std::memory_order_relaxed);
                if (snapshot_ready)
                {
                    const std::array<int, 3> deficit{
                        local_coverage.participant_id,
                        local_coverage.representative_layer,
                        static_cast<int>(local_coverage.source_index),
                    };
                    if (!last_reported_service_deficit_ ||
                        *last_reported_service_deficit_ != deficit)
                    {
                        last_reported_service_deficit_ = deficit;
                        LOG_INFO(
                            "[ExpertOverlay][Economy] Awaiting live service "
                            "evidence participant="
                            << local_coverage.participant_id
                            << " representative_layer="
                            << local_coverage.representative_layer
                            << " source="
                            << serviceSourceName(
                                   local_coverage.source_index));
                        PerfStatsCollector::addCounter(
                            "moe_overlay_residency",
                            "economy_service_coverage_pending",
                            1.0,
                            "maintenance",
                            config_.perf_device,
                            {{"participant",
                              std::to_string(
                                  local_coverage.participant_id)},
                             {"representative_layer",
                              std::to_string(
                                  local_coverage.representative_layer)},
                             {"source",
                              serviceSourceName(
                                  local_coverage.source_index)}});
                    }
                }
            }
            else
            {
                last_reported_service_deficit_.reset();
            }
            if (config_.evidence_exchange)
            {
                if (local_ready)
                {
                    ready_local_service_evidence_ =
                        std::move(raw_service);
                }
                else
                {
                    ready_local_service_evidence_.reset();
                }
                std::string error;
                if (!config_.evidence_exchange->beginServiceReadiness(
                        local_ready, &error))
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay could not begin its service readiness exchange"
                            : std::move(error));
                }
                state_.store(
                    MoEOverlayEconomyCertificationState::
                        ExchangingServiceReadiness,
                    std::memory_order_release);
                return;
            }
            if (!local_ready)
                return;
            installCompleteEvidence(std::move(raw_service));
        }
        catch (const std::exception &error)
        {
            fail(error.what());
        }
        catch (...)
        {
            fail(
                "ExpertOverlay economy certification raised a non-standard exception");
        }
    }

    void MoEOverlayEconomyCertificationController::installCompleteEvidence(
        std::vector<MoEOverlayParticipantLayerServiceTotals> raw_service)
    {
        if (!expanded_migration_measurements_ ||
            !expanded_migration_measurements_->valid())
        {
            throw std::logic_error(
                "ExpertOverlay certification lost its expanded migration evidence");
        }

        MoEOverlayEconomyMeasurements measurements;
        measurements.service_measurement_identity =
            serviceMeasurementIdentity(raw_service);
        measurements.migration_measurement_identity =
            expanded_migration_measurements_->identity;
        measurements.active_sources = config_.active_sources;
        measurements.participant_service =
            MoEOverlayEconomyProfileComposer::
                normalizeEquivalentServiceTotals(
                    std::move(raw_service),
                    config_.active_sources,
                    *config_.layer_catalog);
        measurements.directed_migration =
            MoEOverlayEconomyProfileComposer::normalizeMigrationMeasurements(
                expanded_migration_measurements_->rows);

        const auto snapshot = config_.authority->snapshot();
        if (!snapshot || !snapshot->valid() || !snapshot->placement_plan)
        {
            throw std::logic_error(
                "ExpertOverlay authority lost its initial snapshot before economy installation");
        }
        auto profiles = MoEOverlayEconomyProfileComposer::compose(
            *snapshot->placement_plan,
            config_.model_metadata,
            snapshot->owner_map,
            std::move(measurements),
            config_.economy_policy);
        if (!profiles.valid())
        {
            throw std::logic_error(
                "ExpertOverlay economy composer returned incomplete certified profiles");
        }
        profiles_composed_.fetch_add(1, std::memory_order_relaxed);
        recordServicePriorityCrossovers(
            *profiles.service,
            *snapshot->placement_plan);
        config_.authority->installEconomyCertification(
            std::move(profiles.service),
            std::move(profiles.migration),
            profiles.policy);
        certifications_installed_.fetch_add(
            1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCertificationState::Complete,
            std::memory_order_release);

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "economy_certification_complete",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"blocking_inference", "false"},
             {"layer_equivalence", config_.layer_catalog->identity()},
             {"service_source", "live_prepared_experts"},
             {"movement_source", "real_non_publishable_waves"},
             {"distributed",
              config_.evidence_exchange ? "true" : "false"}});
    }

    void MoEOverlayEconomyCertificationController::
        recordServicePriorityCrossovers(
        const MoERoutedTierServiceProfile &profile,
        const MoERoutedExpertPlacementPlan &plan) const
    {
        const std::size_t tier_count = plan.routed_tiers.size();
        const std::size_t layer_count = static_cast<std::size_t>(
            config_.model_metadata.num_layers);
        if (profile.costs.size() != tier_count * layer_count)
        {
            throw std::invalid_argument(
                "ExpertOverlay crossover diagnostics received incomplete service geometry");
        }

        std::vector<const MoERoutedTierLayerPhaseServiceCost *> rows(
            profile.costs.size(), nullptr);
        for (const auto &row : profile.costs)
        {
            if (row.tier_index < 0 || row.layer < 0 ||
                static_cast<std::size_t>(row.tier_index) >= tier_count ||
                static_cast<std::size_t>(row.layer) >= layer_count)
            {
                throw std::invalid_argument(
                    "ExpertOverlay crossover diagnostics received an invalid service coordinate");
            }
            const std::size_t offset =
                static_cast<std::size_t>(row.tier_index) * layer_count +
                static_cast<std::size_t>(row.layer);
            if (rows[offset] != nullptr)
            {
                throw std::invalid_argument(
                    "ExpertOverlay crossover diagnostics received duplicate service coordinates");
            }
            rows[offset] = &row;
        }

        std::vector<int> priority_order(tier_count, 0);
        std::iota(priority_order.begin(), priority_order.end(), 0);
        std::sort(
            priority_order.begin(),
            priority_order.end(),
            [&plan](int lhs, int rhs)
            {
                return plan.routed_tiers[static_cast<std::size_t>(lhs)]
                           .priority <
                       plan.routed_tiers[static_cast<std::size_t>(rhs)]
                           .priority;
            });

        std::size_t crossovers = 0;
        for (std::size_t layer = 0; layer < layer_count; ++layer)
        {
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (!profile.active_sources[phase])
                    continue;
                for (std::size_t rank = 1;
                     rank < priority_order.size();
                     ++rank)
                {
                    const int preferred = priority_order[rank - 1];
                    const int candidate = priority_order[rank];
                    const auto preferred_cost =
                        rows[static_cast<std::size_t>(preferred) *
                                 layer_count +
                             layer]
                            ->nanoseconds_per_activation[phase];
                    const auto candidate_cost =
                        rows[static_cast<std::size_t>(candidate) *
                                 layer_count +
                             layer]
                            ->nanoseconds_per_activation[phase];
                    if (candidate_cost >= preferred_cost)
                        continue;
                    ++crossovers;
                    PerfStatsCollector::addCounter(
                        "moe_overlay_residency",
                        "economy_service_priority_crossovers",
                        1.0,
                        "maintenance",
                        config_.perf_device,
                        {{"layer", std::to_string(layer)},
                         {"source", serviceSourceName(phase)},
                         {"preferred_tier", std::to_string(preferred)},
                         {"preferred_priority",
                          std::to_string(
                              plan.routed_tiers[
                                  static_cast<std::size_t>(preferred)]
                                  .priority)},
                         {"preferred_ns_per_activation",
                          std::to_string(preferred_cost)},
                         {"crossover_tier", std::to_string(candidate)},
                         {"crossover_priority",
                          std::to_string(
                              plan.routed_tiers[
                                  static_cast<std::size_t>(candidate)]
                                  .priority)},
                         {"crossover_ns_per_activation",
                          std::to_string(candidate_cost)}});
                }
            }
        }
        if (crossovers != 0)
        {
            LOG_INFO(
                "[ExpertOverlay][Economy] Accepted " << crossovers
                << " measured phase-specific service crossover(s); integer "
                   "priority remains the capacity and deterministic tie authority");
        }
    }

    void MoEOverlayEconomyCertificationController::requestStop() noexcept
    {
        stop_requested_.store(true, std::memory_order_release);
        config_.calibration->requestStop();
    }

    MoEOverlayEconomyCertificationController::ServiceEvidenceCoverage
    MoEOverlayEconomyCertificationController::serviceEvidenceCoverage(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
        const std::vector<int> &participant_ids) const
    {
        const auto snapshot = config_.authority->snapshot();
        if (!snapshot || !snapshot->valid())
        {
            throw std::logic_error(
                "ExpertOverlay service certification lost its live snapshot");
        }
        const std::size_t expected =
            participant_ids.size() *
            static_cast<std::size_t>(config_.model_metadata.num_layers);
        if (rows.size() != expected)
        {
            throw std::invalid_argument(
                "ExpertOverlay service snapshot omitted a participant/layer row");
        }

        std::size_t row_index = 0;
        for (const int participant_id : participant_ids)
        {
            for (int layer = 0;
                 layer < config_.model_metadata.num_layers;
                 ++layer, ++row_index)
            {
                const auto &row = rows[row_index];
                if (!row.valid() ||
                    row.participant_id != participant_id ||
                    row.layer != layer)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service snapshot is malformed, overflowed, or non-canonical");
                }
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    if (!config_.active_sources[phase] &&
                        row.sample_count[phase] != 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay service snapshot sampled a runtime-disabled inference phase");
                    }
                }
            }
        }

        /*
         * Sparse packets may legitimately leave individual equivalent layers
         * idle. Readiness is therefore one real observation per participant,
         * authenticated manifest class, and active production phase. The
         * composer later pools exact integer totals over the same classes.
         */
        for (std::size_t participant = 0;
             participant < participant_ids.size();
             ++participant)
        {
            const std::size_t base = participant *
                static_cast<std::size_t>(
                    config_.model_metadata.num_layers);
            for (const auto &group : config_.layer_catalog->groups())
            {
                if (!group.valid())
                {
                    throw std::logic_error(
                        "ExpertOverlay service readiness received an invalid layer-equivalence class");
                }
                for (std::size_t phase = 0;
                     phase < kExpertHistogramProductionSourceCount;
                     ++phase)
                {
                    if (!config_.active_sources[phase])
                        continue;
                    const bool observed = std::any_of(
                        group.member_layers.begin(),
                        group.member_layers.end(),
                        [&rows, base, phase](int layer)
                        {
                            return rows[
                                       base + static_cast<std::size_t>(layer)]
                                       .sample_count[phase] != 0;
                        });
                    if (!observed)
                    {
                        return {
                            .complete = false,
                            .participant_id =
                                participant_ids[participant],
                            .representative_layer =
                                group.representative_layer,
                            .source_index = phase,
                        };
                    }
                }
            }
        }
        return {.complete = true};
    }

    std::string
    MoEOverlayEconomyCertificationController::serviceMeasurementIdentity(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows) const
    {
        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashString(hash, config_.layer_catalog->identity());
        for (const bool active : config_.active_sources)
            hashUnsigned(hash, active ? 1u : 0u);
        hashUnsigned(hash, static_cast<std::uint64_t>(rows.size()));
        for (const auto &row : rows)
        {
            hashUnsigned(hash, static_cast<std::uint64_t>(row.participant_id));
            hashUnsigned(hash, static_cast<std::uint64_t>(row.layer));
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                hashUnsigned(hash, row.total_nanoseconds[phase]);
                hashUnsigned(hash, row.activation_count[phase]);
                hashUnsigned(hash, row.sample_count[phase]);
            }
        }
        std::ostringstream output;
        output << "expert-overlay-live-service-v2/" << std::hex
               << std::setfill('0') << std::setw(16) << hash;
        return output.str();
    }

    void MoEOverlayEconomyCertificationController::fail(
        std::string message) noexcept
    {
        if (!healthy_.exchange(false, std::memory_order_acq_rel))
            return;
        if (message.empty())
            message = "Unknown ExpertOverlay economy certification failure";
        {
            std::lock_guard<std::mutex> lock(failure_mutex_);
            failure_message_ = std::move(message);
        }
        fatal_failures_.fetch_add(1, std::memory_order_relaxed);
        state_.store(
            MoEOverlayEconomyCertificationState::Failed,
            std::memory_order_release);
    }

    MoEOverlayEconomyCertificationState
    MoEOverlayEconomyCertificationController::state() const noexcept
    {
        return state_.load(std::memory_order_acquire);
    }

    bool MoEOverlayEconomyCertificationController::healthy() const noexcept
    {
        return healthy_.load(std::memory_order_acquire);
    }

    std::string
    MoEOverlayEconomyCertificationController::failureMessage() const
    {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        return failure_message_;
    }

    MoEOverlayEconomyCertificationStats
    MoEOverlayEconomyCertificationController::stats() const noexcept
    {
        return {
            .polls = polls_.load(std::memory_order_relaxed),
            .movement_calibration_polls =
                movement_calibration_polls_.load(std::memory_order_relaxed),
            .service_snapshot_attempts =
                service_snapshot_attempts_.load(std::memory_order_relaxed),
            .service_snapshot_contentions =
                service_snapshot_contentions_.load(std::memory_order_relaxed),
            .incomplete_service_snapshots =
                incomplete_service_snapshots_.load(
                    std::memory_order_relaxed),
            .profiles_composed =
                profiles_composed_.load(std::memory_order_relaxed),
            .certifications_installed =
                certifications_installed_.load(std::memory_order_relaxed),
            .fatal_failures =
                fatal_failures_.load(std::memory_order_relaxed),
        };
    }
} // namespace llaminar2
