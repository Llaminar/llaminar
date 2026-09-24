/**
 * @file MoEOverlayEconomyCertificationController.cpp
 * @brief Local or all-rank measured-economy composition and installation.
 *
 * The exact layer catalog owns service coverage; this controller owns the
 * ordered readiness exchange, immutable profile installation, and routing
 * evidence rebase. A cold-format price remains missing until a real
 * participant-local measurement supplies it.
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

        /** @return Stable diagnostic name for one certification lifecycle state. */
        const char *certificationStateName(
            MoEOverlayEconomyCertificationState state) noexcept
        {
            switch (state)
            {
            case MoEOverlayEconomyCertificationState::CalibratingMovement:
                return "calibrating_movement";
            case MoEOverlayEconomyCertificationState::AwaitingServiceEvidence:
                return "awaiting_service_evidence";
            case MoEOverlayEconomyCertificationState::ExchangingServiceReadiness:
                return "exchanging_service_readiness";
            case MoEOverlayEconomyCertificationState::ExchangingServiceEvidence:
                return "exchanging_service_evidence";
            case MoEOverlayEconomyCertificationState::RebasingRoutingEvidence:
                return "rebasing_routing_evidence";
            case MoEOverlayEconomyCertificationState::Complete:
                return "complete";
            case MoEOverlayEconomyCertificationState::Failed:
                return "failed";
            case MoEOverlayEconomyCertificationState::Stopped:
                return "stopped";
            }
            return "invalid";
        }

        /** @return Generic production workload corresponding to one MoE phase. */
        InferenceMeasurementWorkloadKind measurementWorkloadKind(
            ExpertHistogramSource source) noexcept
        {
            switch (source)
            {
            case ExpertHistogramSource::PrefillChunk:
                return InferenceMeasurementWorkloadKind::Prefill;
            case ExpertHistogramSource::DecodeToken:
                return InferenceMeasurementWorkloadKind::Decode;
            case ExpertHistogramSource::GroupedVerifier:
                return InferenceMeasurementWorkloadKind::GroupedVerifier;
            case ExpertHistogramSource::SyntheticTest:
                return InferenceMeasurementWorkloadKind::None;
            }
            return InferenceMeasurementWorkloadKind::None;
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
            (config_.target ==
                 MoEOverlayEconomyCertificationTarget::ResidencyAuthority &&
             config_.authority->hasEconomyCertification()) ||
            (config_.target !=
                 MoEOverlayEconomyCertificationTarget::ResidencyAuthority &&
             config_.target !=
                 MoEOverlayEconomyCertificationTarget::
                     DetachedDeviceAuthority) ||
            !config_.production_topology.valid() ||
            config_.production_topology.economyActiveSources() !=
                config_.calibration->requiredSources() ||
            config_.service_readiness_retry_interval <
                std::chrono::milliseconds::zero())
        {
            throw std::invalid_argument(
                "ExpertOverlay economy certification requires complete dynamic dependencies, a valid typed target, and one consistent production topology");
        }
        if (config_.perf_device.empty())
            config_.perf_device = "expert_overlay";

        if (!config_.prepared_service_measurements.empty())
        {
            (void)config_.layer_catalog->serviceEvidenceGaps(
                config_.prepared_service_measurements,
                config_.registry->localParticipantIds(),
                config_.production_topology);
        }

        const auto snapshot = config_.authority->snapshot();
        if (!snapshot || !snapshot->valid() ||
            config_.model_metadata.num_layers !=
                snapshot->layered_ownership.layerCount() ||
            config_.model_metadata.num_experts !=
                snapshot->layered_ownership.expertCount() ||
            config_.production_topology.layerCount() !=
                static_cast<std::size_t>(
                    config_.model_metadata.num_layers) ||
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

    void MoEOverlayEconomyCertificationController::
        beginServiceReadinessRound(
            MoEOverlayServiceEvidenceReadiness local_readiness)
    {
        if (!config_.evidence_exchange ||
            state() !=
                MoEOverlayEconomyCertificationState::AwaitingServiceEvidence ||
            !config_.evidence_exchange->idle())
        {
            throw std::logic_error(
                "ExpertOverlay service-readiness round requires an idle distributed lane in AwaitingServiceEvidence");
        }
        if (local_readiness ==
                MoEOverlayServiceEvidenceReadiness::Ready &&
            !ready_local_service_evidence_)
        {
            throw std::logic_error(
                "ExpertOverlay ready service round requires a retained local snapshot");
        }

        std::string error;
        if (!config_.evidence_exchange->beginServiceReadiness(
                local_readiness, &error))
        {
            throw std::runtime_error(
                error.empty()
                    ? "ExpertOverlay could not begin its typed service-readiness round"
                    : std::move(error));
        }
        state_.store(
            MoEOverlayEconomyCertificationState::ExchangingServiceReadiness,
            std::memory_order_release);
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
            auto current = state();
            if (current ==
                MoEOverlayEconomyCertificationState::
                    RebasingRoutingEvidence)
            {
                if (config_.target !=
                        MoEOverlayEconomyCertificationTarget::
                            ResidencyAuthority ||
                    !pending_host_profiles_ ||
                    !pending_host_profiles_->valid())
                {
                    throw std::logic_error(
                        "ExpertOverlay routing-evidence rebase lost its complete host profiles");
                }

                routing_evidence_rebase_polls_.fetch_add(
                    1, std::memory_order_relaxed);
                const auto progress =
                    config_.authority->progressEconomyEvidenceRebase();
                if (progress == MoEOverlayHistogramRebaseProgress::Pending)
                    return;

                /*
                 * The fresh histogram generation is authoritative before the
                 * economy certificate becomes visible.  Consequently the
                 * maintenance service cannot issue a proposal from traffic
                 * used only to measure prepared-expert service cost.
                 */
                auto profiles = std::move(*pending_host_profiles_);
                pending_host_profiles_.reset();
                config_.authority->installEconomyCertification(
                    std::move(profiles.service),
                    std::move(profiles.migration),
                    profiles.policy);
                routing_evidence_rebases_.fetch_add(
                    1, std::memory_order_relaxed);
                certifications_installed_.fetch_add(
                    1, std::memory_order_relaxed);
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "economy_routing_evidence_rebases",
                    1.0,
                    "maintenance",
                    config_.perf_device,
                    {{"calibration_demand_discarded", "true"},
                     {"blocking_inference", "false"},
                     {"policy_owner", "host"}});
                markComplete();
                return;
            }

            if (current ==
                MoEOverlayEconomyCertificationState::CalibratingMovement)
            {
                /*
                 * Movement calibration and service rounds share one private
                 * lane. Finish or stop calibration before the certification
                 * state machine is allowed to own that lane.
                 */
                const bool stopping =
                    stop_requested_.load(std::memory_order_acquire);
                if (stopping)
                    config_.calibration->requestStop();
                movement_calibration_polls_.fetch_add(
                    1, std::memory_order_relaxed);
                config_.calibration->poll();
                if (!config_.calibration->healthy())
                {
                    fail(config_.calibration->failureMessage());
                    return;
                }
                const auto calibration_state = config_.calibration->state();
                if (stopping)
                {
                    if (calibration_state !=
                            MoEOverlayEconomyCalibrationState::Stopped &&
                        calibration_state !=
                            MoEOverlayEconomyCalibrationState::Complete)
                    {
                        return;
                    }
                    expanded_migration_measurements_.reset();
                    ready_local_service_evidence_.reset();
                    state_.store(
                        MoEOverlayEconomyCertificationState::
                            AwaitingServiceEvidence,
                        std::memory_order_release);
                    if (config_.evidence_exchange)
                    {
                        beginServiceReadinessRound(
                            MoEOverlayServiceEvidenceReadiness::Stopping);
                    }
                    else
                    {
                        state_.store(
                            MoEOverlayEconomyCertificationState::Stopped,
                            std::memory_order_release);
                    }
                    return;
                }

                if (calibration_state !=
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
                next_service_readiness_attempt_ = {};
                state_.store(
                    MoEOverlayEconomyCertificationState::
                        AwaitingServiceEvidence,
                    std::memory_order_release);
                return;
            }

            if (current ==
                MoEOverlayEconomyCertificationState::
                    ExchangingServiceReadiness)
            {
                MoEOverlayServiceEvidenceReadiness global_readiness =
                    MoEOverlayServiceEvidenceReadiness::AwaitingEvidence;
                std::string error;
                const auto progress =
                    config_.evidence_exchange->pollServiceReadiness(
                        &global_readiness, &error);
                if (progress == MoEOverlayResidencyWaveProgress::Pending)
                    return;
                if (progress != MoEOverlayResidencyWaveProgress::Ready)
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay service-readiness round failed"
                            : std::move(error));
                }

                if (global_readiness ==
                    MoEOverlayServiceEvidenceReadiness::Stopping)
                {
                    stop_requested_.store(true, std::memory_order_release);
                    config_.calibration->requestStop();
                    ready_local_service_evidence_.reset();
                    expanded_migration_measurements_.reset();
                    state_.store(
                        MoEOverlayEconomyCertificationState::Stopped,
                        std::memory_order_release);
                    return;
                }
                if (global_readiness ==
                    MoEOverlayServiceEvidenceReadiness::AwaitingEvidence)
                {
                    ready_local_service_evidence_.reset();
                    next_service_readiness_attempt_ =
                        std::chrono::steady_clock::now() +
                        config_.service_readiness_retry_interval;
                    state_.store(
                        MoEOverlayEconomyCertificationState::
                            AwaitingServiceEvidence,
                        std::memory_order_release);
                    return;
                }
                if (global_readiness !=
                    MoEOverlayServiceEvidenceReadiness::Ready ||
                    !ready_local_service_evidence_)
                {
                    throw std::logic_error(
                        "ExpertOverlay ready service round omitted its retained local snapshot");
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

            if (current ==
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
                /*
                 * A successful Ready round commits the finite all-gather. A
                 * stop request arriving after that point cannot roll one rank
                 * back while its peers install the same immutable certificate.
                 */
                composeCompleteEvidence(std::move(complete_service));
                return;
            }

            if (current !=
                MoEOverlayEconomyCertificationState::AwaitingServiceEvidence)
            {
                throw std::logic_error(
                    "ExpertOverlay economy certification reached an invalid lifecycle state");
            }

            if (stop_requested_.load(std::memory_order_acquire))
            {
                config_.calibration->requestStop();
                ready_local_service_evidence_.reset();
                expanded_migration_measurements_.reset();
                if (config_.evidence_exchange)
                {
                    beginServiceReadinessRound(
                        MoEOverlayServiceEvidenceReadiness::Stopping);
                }
                else
                {
                    state_.store(
                        MoEOverlayEconomyCertificationState::Stopped,
                        std::memory_order_release);
                }
                return;
            }

            const auto now = std::chrono::steady_clock::now();
            if (now < next_service_readiness_attempt_)
                return;

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
            else if (!config_.prepared_service_measurements.empty())
            {
                // Certification owns the composition, not either measurement
                // producer's live ledger. A later device snapshot still imports
                // its original cumulative values without a fabricated offset.
                raw_service = config_.layer_catalog->withPreparedServiceEvidence(
                    raw_service, config_.prepared_service_measurements,
                    config_.registry->localParticipantIds(),
                    config_.production_topology);
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
                    std::ostringstream observed_layers_builder;
                    bool first_observed_layer = true;
                    std::array<std::uint64_t,
                               kExpertHistogramProductionSourceCount>
                        participant_samples{};
                    std::array<std::uint64_t,
                               kExpertHistogramProductionSourceCount>
                        participant_activations{};
                    for (const auto &row : raw_service)
                    {
                        if (row.participant_id !=
                            local_coverage.participant_id)
                        {
                            continue;
                        }
                        for (std::size_t source = 0u;
                             source <
                                 kExpertHistogramProductionSourceCount;
                             ++source)
                        {
                            participant_samples[source] +=
                                row.sample_count[source];
                            participant_activations[source] +=
                                row.activation_count[source];
                        }
                        if (row.sample_count[
                                local_coverage.source_index] == 0)
                        {
                            continue;
                        }
                        if (!first_observed_layer)
                            observed_layers_builder << ',';
                        observed_layers_builder << row.layer;
                        first_observed_layer = false;
                    }
                    const std::string observed_layers =
                        first_observed_layer
                            ? std::string("none")
                            : observed_layers_builder.str();
                    std::ostringstream phase_totals_builder;
                    for (std::size_t source = 0u;
                         source < kExpertHistogramProductionSourceCount;
                         ++source)
                    {
                        if (source != 0u)
                            phase_totals_builder << ';';
                        phase_totals_builder
                            << serviceSourceName(source)
                            << "{samples="
                            << participant_samples[source]
                            << ",activations="
                            << participant_activations[source] << '}';
                    }
                    const std::string phase_totals =
                        phase_totals_builder.str();
                    if (!last_reported_service_deficit_ ||
                        *last_reported_service_deficit_ != deficit ||
                        last_reported_service_observed_layers_ !=
                            observed_layers)
                    {
                        last_reported_service_deficit_ = deficit;
                        last_reported_service_observed_layers_ =
                            observed_layers;
                        LOG_INFO(
                            "[ExpertOverlay][Economy] Awaiting live service "
                            "evidence participant="
                            << local_coverage.participant_id
                            << " representative_layer="
                            << local_coverage.representative_layer
                            << " source="
                            << serviceSourceName(
                                   local_coverage.source_index)
                            << " observed_layers="
                            << observed_layers
                            << " participant_phase_totals="
                            << phase_totals);
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
                last_reported_service_observed_layers_.clear();
            }
            if (config_.evidence_exchange)
            {
                /*
                 * Every rank enters the same bounded round even when its local
                 * sparse routes have not covered the whole service matrix.
                 * The prior collective completion orders the next round; the
                 * retry interval limits control-plane traffic without putting
                 * ExpertOverlay policy into the inference or benchmark runner.
                 */
                if (local_ready)
                    ready_local_service_evidence_ = std::move(raw_service);
                else
                    ready_local_service_evidence_.reset();
                beginServiceReadinessRound(
                    local_ready
                        ? MoEOverlayServiceEvidenceReadiness::Ready
                        : MoEOverlayServiceEvidenceReadiness::
                              AwaitingEvidence);
                return;
            }
            if (!local_ready)
            {
                next_service_readiness_attempt_ =
                    now + config_.service_readiness_retry_interval;
                return;
            }
            composeCompleteEvidence(std::move(raw_service));
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

    void MoEOverlayEconomyCertificationController::composeCompleteEvidence(
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
        measurements.production_topology = config_.production_topology;
        measurements.participant_service =
            MoEOverlayEconomyProfileComposer::
                normalizeEquivalentServiceTotals(
                    std::move(raw_service),
                    config_.production_topology,
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
        recordCertifiedServiceEconomy(
            *profiles.service,
            *snapshot->placement_plan);
        if (config_.target ==
            MoEOverlayEconomyCertificationTarget::ResidencyAuthority)
        {
            if (pending_host_profiles_)
            {
                throw std::logic_error(
                    "ExpertOverlay host economy profiles were composed more than once");
            }
            pending_host_profiles_ = std::move(profiles);
            state_.store(
                MoEOverlayEconomyCertificationState::
                    RebasingRoutingEvidence,
                std::memory_order_release);
            return;
        }

        /* The host owns measurement evidence, not placement. Publish one
         * immutable bundle for the mapped-fabric writer before Complete is
         * release-stored; the device controller remains the only policy and
         * durable-epoch authority. */
        {
            std::lock_guard<std::mutex> lock(detached_profiles_mutex_);
            if (detached_profiles_)
            {
                throw std::logic_error(
                    "ExpertOverlay detached economy profiles were published more than once");
            }
            detached_profiles_ = std::move(profiles);
        }
        detached_profiles_completed_.fetch_add(
            1, std::memory_order_relaxed);
        markComplete();
    }

    void MoEOverlayEconomyCertificationController::markComplete()
    {
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
              config_.evidence_exchange ? "true" : "false"},
             {"target",
              config_.target ==
                      MoEOverlayEconomyCertificationTarget::ResidencyAuthority
                  ? "host_authority"
                  : "device_authority"}});
    }

    void MoEOverlayEconomyCertificationController::
        recordCertifiedServiceEconomy(
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

        const std::size_t participant_count =
            profile.participant_costs.empty()
                ? 0u
                : profile.participant_costs.size() / layer_count;
        if (participant_count == 0u ||
            profile.participant_costs.size() !=
                participant_count * layer_count)
        {
            throw std::invalid_argument(
                "ExpertOverlay service diagnostics received incomplete participant geometry");
        }
        for (const auto &row : profile.participant_costs)
        {
            if (row.participant_id < 0 || row.layer < 0 ||
                static_cast<std::size_t>(row.participant_id) >=
                    participant_count ||
                static_cast<std::size_t>(row.layer) >= layer_count)
            {
                throw std::invalid_argument(
                    "ExpertOverlay service diagnostics received an invalid participant coordinate");
            }
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                if (!profile.production_topology.requiresServiceEvidence(
                        row.layer, phase))
                    continue;
                const std::uint64_t cost =
                    row.nanoseconds_per_activation[phase];
                if (cost == 0u)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay service diagnostics received a zero active participant cost");
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "certified_participant_service_ns_per_activation",
                    static_cast<double>(cost),
                    "maintenance",
                    config_.perf_device,
                    {{"participant",
                      std::to_string(row.participant_id)},
                     {"layer", std::to_string(row.layer)},
                     {"source", serviceSourceName(phase)},
                     {"service_profile", profile.identity}});
            }
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
                if (!profile.production_topology.requiresServiceEvidence(
                        static_cast<int>(layer), phase))
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

    bool MoEOverlayEconomyCertificationController::
        importDeviceServiceMeasurements(
            int participant_id,
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            std::string *error)
    {
        if (error)
            error->clear();
        const auto current = state();
        if (!healthy() ||
            (current !=
                 MoEOverlayEconomyCertificationState::CalibratingMovement &&
             current !=
                 MoEOverlayEconomyCertificationState::AwaitingServiceEvidence))
        {
            if (error)
                *error = "device service evidence arrived after certification exchange began";
            return false;
        }
        return config_.registry->importDeviceServiceMeasurements(
            participant_id, rows, error);
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
        // The catalog owns coverage for preparation and certification alike.
        // Keep the first-gap diagnostic without a second set of coverage rules.
        const auto gaps = config_.layer_catalog->serviceEvidenceGaps(
            rows, participant_ids, config_.production_topology);
        if (gaps.empty())
            return {.complete = true};
        const auto &gap = gaps.front();
        return {
            .complete = false,
            .participant_id = gap.participant_id,
            .representative_layer = gap.representative_layer,
            .source_index = expertHistogramProductionSourceIndex(gap.source),
        };
    }

    std::string
    MoEOverlayEconomyCertificationController::serviceMeasurementIdentity(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows) const
    {
        std::uint64_t hash = kFNV1a64OffsetBasis;
        hashString(hash, config_.layer_catalog->identity());
        for (int layer = 0;
             layer < config_.model_metadata.num_layers;
             ++layer)
        {
            for (const bool reachable :
                 config_.production_topology.sources(layer))
            {
                hashUnsigned(hash, reachable ? 1u : 0u);
            }
            for (const bool priced :
                 config_.production_topology.economySources(layer))
            {
                hashUnsigned(hash, priced ? 1u : 0u);
            }
        }
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

    std::optional<MoEOverlayCertifiedEconomyProfiles>
    MoEOverlayEconomyCertificationController::detachedProfiles() const
    {
        if (config_.target !=
                MoEOverlayEconomyCertificationTarget::
                    DetachedDeviceAuthority ||
            state() != MoEOverlayEconomyCertificationState::Complete)
        {
            return std::nullopt;
        }
        std::lock_guard<std::mutex> lock(detached_profiles_mutex_);
        if (!detached_profiles_ || !detached_profiles_->valid())
        {
            throw std::logic_error(
                "Completed detached ExpertOverlay certification omitted its immutable profiles");
        }
        return detached_profiles_;
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
            .routing_evidence_rebase_polls =
                routing_evidence_rebase_polls_.load(
                    std::memory_order_relaxed),
            .routing_evidence_rebases =
                routing_evidence_rebases_.load(
                    std::memory_order_relaxed),
            .certifications_installed =
                certifications_installed_.load(std::memory_order_relaxed),
            .detached_profiles_completed =
                detached_profiles_completed_.load(
                    std::memory_order_relaxed),
            .fatal_failures =
                fatal_failures_.load(std::memory_order_relaxed),
        };
    }

    InferenceMeasurementReadiness
    MoEOverlayEconomyCertificationController::measurementReadiness() const
    {
        const auto observed_state = state();
        const auto calibration_stats = config_.calibration->stats();
        InferenceMeasurementReadiness readiness{
            .state = InferenceMeasurementReadinessState::Calibrating,
            .completed_work_units = calibration_stats.accepted_pairs,
            .required_work_units =
                config_.calibration->expectedAcceptedPairs(),
            .inference_requested = false,
            .inference_request_generation = 0u,
            .requested_workload = InferenceMeasurementWorkloadKind::None,
            .owner = "expert_overlay_economy",
            .phase = certificationStateName(observed_state),
        };

        if (observed_state !=
                MoEOverlayEconomyCertificationState::CalibratingMovement &&
            observed_state != MoEOverlayEconomyCertificationState::Failed &&
            observed_state != MoEOverlayEconomyCertificationState::Stopped)
        {
            /*
             * Physical topology profiling is the only setup gate. Prepared
             * expert service telemetry is accumulated by ordinary requests;
             * until it is complete the movement authority remains dormant,
             * but inference is fully valid and must not be delayed.
             */
            readiness.state = InferenceMeasurementReadinessState::Ready;
            return readiness;
        }
        if (observed_state ==
                MoEOverlayEconomyCertificationState::Failed ||
            observed_state ==
                MoEOverlayEconomyCertificationState::Stopped ||
            !healthy())
        {
            readiness.state = InferenceMeasurementReadinessState::Failed;
            readiness.diagnostic = failureMessage();
            if (readiness.diagnostic.empty())
            {
                readiness.diagnostic =
                    observed_state ==
                            MoEOverlayEconomyCertificationState::Stopped
                        ? "ExpertOverlay economy certification stopped before completion"
                        : "ExpertOverlay economy certification failed";
            }
        }
        return readiness;
    }
} // namespace llaminar2
