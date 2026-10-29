/**
 * @file MoEOverlayCPUHostMaintenance.cpp
 * @brief Build and retire real CPU maintenance from an already admitted stage.
 *
 * CPU sample scratch, host demand, shadow slots and transfer staging are priced
 * by MoEOverlayLocalCapacityPlanner before joint admission. Every allocation
 * below goes through those existing typed PMA owners. Global source identities
 * remain offset-aware in calibration, service evidence and reusable bank seals.
 */
#include "MoEOverlayCPUHostMaintenance.h"
#include "MoEOverlayCPUServiceMeasurement.h"
#include "MoEOverlayEconomyCalibrationController.h"
#include "MoEOverlayEconomyCertificationController.h"
#include "MoEOverlayMigrationMeasurementExchange.h"
#include "MoEOverlayMPIRemoteProjectionTransport.h"
#include "collective/CollectiveTimeoutPolicy.h"
#include <chrono>
#include <stdexcept>
#include <thread>

namespace llaminar2
{
    MoEOverlayCPUHostMaintenance::MoEOverlayCPUHostMaintenance(Config config) : config_(std::move(config))
    {
        const auto &c = config_;
        if (!c.memory || !c.admission || c.memory->admission() != c.admission || !c.authority || !c.residency || !c.histogram ||
            c.rebalance.mode != MoERebalanceRuntimeMode::Dynamic ||
            c.authority->maintenanceMode() != c.rebalance.mode || c.authority->noPeerResidency() ||
            !c.authority->retainsHistogram(c.histogram.get()) || c.storage.overlay_world_size != 1 || !c.storage.usesPhysicalResidencyFabric() ||
            c.storage.shadow_slots_per_endpoint_layer == 0 || c.storage.staging_capacity_bytes == 0 ||
            c.storage.maximum_concurrent_cycles == 0 || c.storage.maximum_execution_streams == 0)
            throw std::invalid_argument("CPU host maintenance requires one admitted mutable host authority");
        const auto snapshot = c.authority->snapshot();
        if (!snapshot || !snapshot->valid() ||
            snapshot->placement_plan->authority_execution != MoEOverlayAuthorityExecutionKind::HostResident ||
            c.residency->initialEpoch() != snapshot->epoch ||
            c.residency->initialOwnerMap().owners() != snapshot->owner_map.owners() ||
            c.residency->initialOwnerMap().participants() != snapshot->owner_map.participants())
            throw std::invalid_argument("CPU host maintenance has foreign prepared residency");
        snapshot->owner_map.requireLayerGeometry(c.metadata.num_layers, c.metadata.num_experts, c.metadata.first_model_layer);
        if (!c.topology.valid() || c.topology.firstModelLayer() != c.metadata.first_model_layer ||
            c.topology.layerCount() != static_cast<std::size_t>(c.metadata.num_layers))
            throw std::invalid_argument("CPU host maintenance topology has a foreign stage interval");
        for (const auto &participant : snapshot->owner_map.participants())
        {
            if (!participant.device.is_cpu() || !participant.world_rank_known ||
                participant.world_rank != c.memory->worldRank() || !c.memory->contains(participant.device))
                throw std::invalid_argument("CPU host maintenance requires exact local NUMA allocator identities");
        }
        const auto catalog = c.authority->economyLayerCatalog();
        if (!catalog || catalog->firstModelLayer() != c.metadata.first_model_layer ||
            catalog->layerCount() != static_cast<std::size_t>(c.metadata.num_layers))
            throw std::invalid_argument("CPU host maintenance lost its exact source catalog");
    }

    MoEOverlayCPUHostMaintenance::~MoEOverlayCPUHostMaintenance()
    {
        try { stopAndDrain(); } catch (...) { std::terminate(); }
    }

    void MoEOverlayCPUHostMaintenance::prepare()
    {
        if (service_ || drained_ || !config_.residency->allInitialBanksReady())
            throw std::logic_error("CPU host maintenance preparation requires fresh complete physical banks");
        // Build privately: any exception releases every partial root before publication.
        const auto snapshot = config_.authority->snapshot();
        const auto catalog = config_.authority->economyLayerCatalog();
        auto samples = MoEOverlayCPUServiceMeasurement::measure(*config_.residency, snapshot->epoch,
            *catalog, config_.topology,
            {.d_model = config_.metadata.d_model, .intermediate = config_.metadata.routed_intermediate_size}, config_.memory);
        auto fabric = MoEOverlayPhysicalResidencyFabric::create({
            .memory_authority = config_.memory, .registry = config_.residency, .initial_snapshot = snapshot,
            .inventory_authority = MoEOverlayPhysicalInventoryAuthority::ParticipantRegistry,
            .shadow_slots_per_endpoint_layer = config_.storage.shadow_slots_per_endpoint_layer,
            .staging_capacity_bytes = config_.storage.staging_capacity_bytes,
            .maximum_concurrent_cycles = config_.storage.maximum_concurrent_cycles,
            .maximum_execution_streams = config_.storage.maximum_execution_streams,
            .collect_economy_measurements = true, .perf_device = config_.perf_device});
        auto factory = std::make_unique<MoEOverlayParticipantPreparedWaveFactory>(MoEOverlayParticipantPreparedWaveFactory::Config{
            .registry = config_.residency, .transfer_provider = fabric, .perf_device = config_.perf_device});
        const auto count = MoEOverlayRemoteProjectionLaneBudget{
            .maximum_participants_per_cycle = snapshot->owner_map.participants().size(),
            .maximum_concurrent_cycles = config_.storage.maximum_concurrent_cycles,
            .projections_per_expert = 1}.projectionOperationCount();
        if (!count) throw std::overflow_error("CPU host migration journal geometry overflow");
        auto journal = std::make_shared<MoEOverlayMigrationMeasurementJournal>(MoEOverlayMigrationMeasurementJournal::Config{
            .maximum_migrations_per_wave = *count});
        auto transport = std::make_shared<MoEOverlayTierMigrationTransport>(MoEOverlayTierMigrationTransport::Config{
            .factory = factory.get(), .transfer_progress_authority = fabric,
            .projections_per_expert = kMoEOverlayExpertProjectionCount, .measurement_sink = journal,
            .measurement_scope = MoEOverlayMigrationMeasurementScope::EconomyCalibrationOnly,
            .require_complete_local_measurements = true, .perf_device = config_.perf_device});
        auto planner = std::make_shared<MoEOverlayEconomyCalibrationPlanner>(MoEOverlayEconomyCalibrationPlanner::Config{
            .live_snapshot = snapshot, .complete_expert_bytes_per_layer = catalog->completeExpertBytesPerLayer(),
            .calibration_layers = catalog->representativeLayers()});
        auto ledger = std::make_shared<MoEOverlayMigrationMeasurementLedger>(MoEOverlayMigrationMeasurementLedger::Config{
            .required_coordinates = planner->requiredCoordinates(), .warmup_samples_per_coordinate = 0,
            .measured_samples_per_coordinate = MoEOverlayEconomyProfileComposer::kMinimumMigrationSamples,
            .required_sources = config_.topology.economyActiveSources(), .measurement_identity = catalog->identity()});
        auto calibration = std::make_shared<MoEOverlayEconomyCalibrationController>(MoEOverlayEconomyCalibrationController::Config{
            .planner = planner, .ledger = ledger, .journal = journal, .transport = transport,
            .presealed_measurements = config_.presealed_measurements, .perf_device = config_.perf_device});
        auto certification = std::make_shared<MoEOverlayEconomyCertificationController>(MoEOverlayEconomyCertificationController::Config{
            .calibration = calibration, .registry = config_.residency, .layer_catalog = catalog,
            .authority = config_.authority, .model_metadata = config_.metadata,
            .economy_policy = {.payoff_horizon_tokens = config_.rebalance.migration_payoff_horizon_tokens},
            .production_topology = config_.topology, .prepared_service_measurements = std::move(samples),
            .perf_device = config_.perf_device});
        auto service = std::make_shared<MoEOverlayResidencyMaintenanceService>(MoEOverlayResidencyMaintenanceService::Config{
            .authority = config_.authority, .transport = transport, .economy_certification = certification,
            .idle_poll_interval = std::chrono::milliseconds(2), .perf_device = config_.perf_device});
        fabric_ = std::move(fabric); factory_ = std::move(factory); transport_ = std::move(transport);
        calibration_ = std::move(calibration); service_ = std::move(service);
    }

    void MoEOverlayCPUHostMaintenance::start()
    {
        if (!ready() || drained_ || service_->state() != MoEOverlayMaintenanceState::Prepared)
            throw std::logic_error("CPU host maintenance activation requires a prepared owner");
        service_->start();
    }

    void MoEOverlayCPUHostMaintenance::notify(MoEOverlayInferenceProgress progress) noexcept
    {
        if (service_) service_->notifyInferenceProgress(progress);
    }

    void MoEOverlayCPUHostMaintenance::stopAndDrain()
    {
        if (service_ && !drained_)
        {
            service_->stopAndDrain(MoEOverlayMaintenanceDrainScope::ProcessLocalComposition);
            drained_ = true;
        }
    }

    bool MoEOverlayCPUHostMaintenance::ready() const noexcept
    { return service_ && fabric_ && transport_ && service_->healthy() && !drained_; }

    void MoEOverlayCPUHostMaintenance::requireScope(const MoEOverlayResidencyAuthority *authority,
        const MoEOverlayParticipantResidencyRegistry *residency, const MoERoutedExpertModelMetadata &metadata) const
    {
        if (authority != config_.authority.get() || residency != config_.residency.get() ||
            metadata.first_model_layer != config_.metadata.first_model_layer || metadata.num_layers != config_.metadata.num_layers ||
            metadata.num_experts != config_.metadata.num_experts || metadata.d_model != config_.metadata.d_model ||
            metadata.routed_intermediate_size != config_.metadata.routed_intermediate_size ||
            metadata.main_inference_layer_count != config_.metadata.main_inference_layer_count)
            throw std::invalid_argument("CPU host maintenance belongs to another stage or residency authority");
    }

    MoEOverlayReusableContextSeal MoEOverlayCPUHostMaintenance::restorePreparedPlacement()
    {
        if (!service_ || !drained_ || !service_->healthy() || config_.authority->activeTicketCount() != 0)
            throw std::logic_error("CPU host reuse sealing requires healthy drained maintenance and no readers");
        const auto idle = [&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(
                collective_timeout_policy::kDefaultCollectiveTimeoutMs);
            for (;;)
            {
                const auto progress = config_.authority->advanceBackground();
                if (!progress.ok()) throw std::runtime_error(progress.error);
                if (progress.status == MoEOverlayResidencyApplyStatus::Idle) return;
                if (std::chrono::steady_clock::now() >= deadline)
                    throw std::runtime_error("CPU host reuse restoration exceeded the standard 30-second protocol");
                std::this_thread::yield();
            }
        };
        idle();
        const auto snapshot = config_.authority->snapshot();
        const auto limit = snapshot->layered_ownership.layerCount() * snapshot->layered_ownership.expertCount() + 1;
        std::size_t waves = 0;
        while (!config_.authority->initialPreparedPlacementPublished())
        {
            if (++waves > limit) throw std::logic_error("CPU host reuse restoration lost bounded monotonic progress");
            const auto transaction = config_.authority->proposeInitialPreparedPlacementRestoration();
            const auto begun = config_.authority->beginApply(transaction, *transport_);
            if (begun.status != MoEOverlayResidencyApplyStatus::Started) throw std::runtime_error(begun.error);
            idle();
        }
        std::string error;
        auto seal = fabric_->sealReusableInitialPlacement(config_.authority->snapshot()->epoch, &error);
        if (!seal) throw std::runtime_error(error);
        return std::move(*seal);
    }
}
