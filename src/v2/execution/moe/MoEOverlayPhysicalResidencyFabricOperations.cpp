/**
 * @file MoEOverlayPhysicalResidencyFabricOperations.cpp
 * @brief Physical residency fabric wave preparation, projection transfer scheduling, and reusable context sealing.
 */
#include "MoEOverlayPhysicalResidencyFabricInternal.h"

namespace llaminar2
{
    bool MoEOverlayPhysicalResidencyFabric::stageDevicePreparedTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayParticipantPreparedTransfers &prepared,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!impl_->device_slot_ledger || !batch.valid() ||
            prepared.status != MoEOverlayResidencyStageStartStatus::Started ||
            prepared.migrations.size() != batch.migrations.size())
        {
            if (error)
                *error =
                    "ExpertOverlay device staging requires the exact completed physical wave";
            return false;
        }

        try
        {
            const auto local_ids = config_.registry->localParticipantIds();
            std::vector<MoEOverlayDevicePreparedPhysicalArrival> arrivals;
            for (std::size_t index = 0u; index < batch.migrations.size(); ++index)
            {
                const auto &migration = batch.migrations[index];
                const auto target = MoEExpertPlacementTarget::fromPlacement(migration.destination);
                if (!target)
                {
                    if (error) *error = "ExpertOverlay device staging lost the complete original target";
                    return false;
                }
                std::set<int> expected;
                for (std::size_t member = 0; member < target->participantCount(); ++member)
                    if (std::binary_search(local_ids.begin(), local_ids.end(), target->participantId(member)))
                        expected.insert(target->participantId(member));
                const auto &keyed = prepared.migrations[index].destination_arrivals;
                if (keyed.size() != expected.size())
                {
                    if (error) *error = "ExpertOverlay device staging requires every exact local target member";
                    return false;
                }
                for (const auto &member : keyed)
                {
                    if (!expected.erase(member.participant_id) || !member.arrival ||
                        member.arrival->isJointFragment() != bool(migration.destination.shard_cohort))
                    {
                        if (error) *error = "ExpertOverlay device staging rejected a duplicate, foreign or wrong-arm member";
                        return false;
                    }
                    MoEOverlayDevicePreparedPhysicalArrival arrival{
                        .key = {.participant_id = member.participant_id, .layer_idx = migration.layer_idx,
                            .expert_id = migration.expert_id}};
                    std::string arrival_error;
                    if (migration.destination.shard_cohort)
                    {
                        MoEOverlayPreparedFragmentPayload fragment;
                        if (member.arrival->destinationEpoch() != batch.candidate_epoch ||
                            !member.arrival->completeFragment(fragment, &arrival_error))
                        {
                            if (error) *error = arrival_error.empty()
                                ? "ExpertOverlay device staging requires a complete candidate-epoch fragment" : arrival_error;
                            return false;
                        }
                        arrival.payload = std::move(fragment);
                    }
                    else
                    {
                        MoEOverlayPreparedExpertPayload whole;
                        if (!member.arrival->completePayload(whole, &arrival_error))
                        {
                            if (error) *error = arrival_error;
                            return false;
                        }
                        arrival.payload = std::move(whole);
                    }
                    arrivals.push_back(std::move(arrival));
                }
            }
            // One ledger checks the complete mixed member set before accepting
            // any destination. The original receipt/ACK edge still releases it.
            if (!impl_->device_slot_ledger->stagePrepared(batch, std::move(arrivals), error)) return false;
            recordMoEPhysicalWave(MoEPhysicalWaveEvent::DestinationsStaged,
                batch.transaction_id, batch.base_epoch, batch.candidate_epoch,
                batch.migrations.size(), config_.perf_device);
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
        catch (...)
        {
            if (error)
                *error =
                    "ExpertOverlay device staging failed with a non-standard exception";
            return false;
        }
    }

    bool MoEOverlayPhysicalResidencyFabric::publishDevicePreparedTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger ||
            !impl_->device_slot_ledger->publish(batch, error))
        {
            if (error && error->empty())
                *error =
                    "ExpertOverlay device physical publication lost its slot ledger";
            return false;
        }
        recordMoEPhysicalWave(MoEPhysicalWaveEvent::EpochPublished,
            batch.transaction_id, batch.base_epoch, batch.candidate_epoch,
            0, config_.perf_device);
        return true;
    }

    bool MoEOverlayPhysicalResidencyFabric::retireDevicePreviousSources(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger)
        {
            if (error)
                *error =
                    "ExpertOverlay device retirement lost its slot ledger";
            return false;
        }
        std::vector<MoEOverlayDevicePreparedPhysicalSlot> retired;
        if (!impl_->device_slot_ledger->retirePrepared(batch, &retired, error))
            return false;

        /*
         * Bootstrap storage lacks an aliasing lease, so enroll its departed
         * assignment in the recyclable arena explicitly. Later-arrival slots
         * are released when `retired` destroys their final payload aliases at
         * method exit. Neither path synchronizes a device or inference stream.
         */
        retirePreviousSources(batch.base_epoch, batch.migrations);
        recordMoEPhysicalWave(MoEPhysicalWaveEvent::SourcesRetired,
            batch.transaction_id, batch.base_epoch, batch.candidate_epoch,
            retired.size(), config_.perf_device);
        return true;
    }

    bool MoEOverlayPhysicalResidencyFabric::abortDeviceTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (!impl_->device_slot_ledger ||
            !impl_->device_slot_ledger->abort(batch, error))
        {
            if (error && error->empty())
                *error =
                    "ExpertOverlay device physical abort lost its slot ledger";
            return false;
        }
        recordMoEPhysicalWave(MoEPhysicalWaveEvent::WaveAborted,
            batch.transaction_id, batch.base_epoch, batch.candidate_epoch,
            0, config_.perf_device);
        return true;
    }

    void MoEOverlayPhysicalResidencyFabric::retirePreviousSources(
        std::uint64_t retired_epoch,
        const std::vector<MoEOverlayTierMigration> &migrations) noexcept
    {
        std::uint64_t recycled = 0;
        for (const auto &migration : migrations)
        {
            const auto target = MoEExpertPlacementTarget::fromPlacement(migration.source);
            if (!target) std::terminate();
            for (size_t member = 0; member < target->participantCount(); ++member)
            {
                const int participant = target->participantId(member);
                if (!config_.registry->endpoint(participant)) continue;
                const auto found = impl_->endpoint_pools.find({participant, migration.layer_idx});
                if (found == impl_->endpoint_pools.end() || !found->second.geometry_pool ||
                    !found->second.geometry_pool->adopted_slots) std::terminate();
                if (found->second.geometry_pool->adopted_slots->retireBootstrapAssignment(
                        migration.layer_idx, migration.expert_id, retired_epoch)) ++recycled;
            }
        }

        if (recycled == 0)
            return;
        impl_->adopted_initial_slots_recycled.fetch_add(
            recycled, std::memory_order_relaxed);
        recordMoEBootstrapSlotsRecycled(retired_epoch, recycled, config_.perf_device);
    }

    std::shared_ptr<MappedTransferProgressEpoch>
    MoEOverlayPhysicalResidencyFabric::transferProgressEpoch(
        DeviceId device) const noexcept
    {
        const auto found = impl_->transfer_progress_epochs.find(device);
        return found == impl_->transfer_progress_epochs.end()
                   ? nullptr
                   : found->second;
    }

    std::vector<DeviceId>
    MoEOverlayPhysicalResidencyFabric::transferProgressDevices() const
    {
        std::vector<DeviceId> devices;
        devices.reserve(impl_->transfer_progress_epochs.size());
        for (const auto &[device, epoch] :
             impl_->transfer_progress_epochs)
        {
            if (!epoch || epoch->device() != device)
            {
                throw std::logic_error(
                    "ExpertOverlay physical fabric retained an invalid transfer-progress inventory entry");
            }
            devices.push_back(device);
        }
        return devices;
    }

    bool MoEOverlayPhysicalResidencyFabric::
        submitOutstandingTransferProgress(std::string *error) noexcept
    {
        for (const auto &[device, epoch] : impl_->transfer_progress_epochs)
        {
            if (!epoch || epoch->device() != device)
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay physical fabric retained an invalid transfer-progress epoch for " +
                        device.toString();
                }
                return false;
            }
            if (!epoch->submitOutstandingProgress())
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay could not enqueue mapped transfer progress on " +
                        device.toString();
                }
                return false;
            }
        }
        return true;
    }

    std::optional<MoEOverlayReusableContextSeal>
    MoEOverlayPhysicalResidencyFabric::sealReusableInitialPlacement(
        std::uint64_t published_epoch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();

        std::unique_lock<std::mutex> seal_lock(impl_->reusable_seal_mutex);
        const auto fail = [&](std::string message)
            -> std::optional<MoEOverlayReusableContextSeal>
        {
            impl_->reusable_seal_failure = std::move(message);
            impl_->reusable_seal_state = Impl::ReusableSealState::Failed;
            if (error)
                *error = impl_->reusable_seal_failure;
            return std::nullopt;
        };

        if (impl_->reusable_seal_state == Impl::ReusableSealState::Sealed)
            return impl_->cached_reusable_seal;
        if (impl_->reusable_seal_state == Impl::ReusableSealState::Failed)
        {
            if (error)
                *error = impl_->reusable_seal_failure;
            return std::nullopt;
        }
        if (impl_->reusable_seal_state != Impl::ReusableSealState::Open)
        {
            return fail(
                "ExpertOverlay reusable-context seal was entered concurrently");
        }
        impl_->reusable_seal_state = Impl::ReusableSealState::Sealing;

        try
        {
            if (published_epoch == 0 || !config_.registry ||
                !config_.initial_snapshot ||
                !config_.initial_snapshot->valid())
            {
                return fail(
                    "ExpertOverlay reusable-context seal requires a positive published epoch and complete initial placement");
            }

            MoEOverlayReusableContextSeal result;
            result.source_epoch = published_epoch;
            result.canonical_owner_map =
                config_.initial_snapshot->placementOwnerMap();
            const auto participant_ids =
                config_.registry->localParticipantIds();
            result.local_banks.reserve(participant_ids.size());
            // One actual installed source borrow per participant spans every
            // physical compaction/completion/drain below. It is not a new
            // inference ticket or a host mirror of global selection.
            std::map<int, MoEOverlayParticipantBankLease> seal_source_banks;
            std::optional<MoEOverlayDevicePreparedPhysicalInventory> seal_device_inventory;

            /*
             * Materialize the exact terminal inventory from its typed owner.
             * Host RCU keeps participant banks current. Device RCU intentionally
             * does not: its controller owns selectors and its physical ledger
             * owns engine lifetimes. Reading the registry in that mode would be
             * an informal host mirror and, after movement, names the wrong
             * epoch. Both authorities produce the same immutable bank value for
             * the common capacity/compaction proof below.
             */
            if (config_.inventory_authority ==
                MoEOverlayPhysicalInventoryAuthority::ParticipantRegistry)
            {
                for (const int participant_id : participant_ids)
                {
                    const auto endpoint =
                        config_.registry->endpoint(participant_id);
                    auto bank = endpoint
                        ? endpoint->acquire(published_epoch)
                        : MoEOverlayParticipantBankLease{};
                    if (!endpoint || !bank ||
                        !bank->valid(
                            participant_id,
                            endpoint->device(),
                            endpoint->numLayers(),
                            endpoint->numExperts(),
                            endpoint->firstModelLayer()))
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal cannot acquire participant " +
                            std::to_string(participant_id) + " at epoch " +
                            std::to_string(published_epoch) +
                            " from the host participant-bank authority");
                    }
                    result.local_banks.push_back(*bank);
                    seal_source_banks.emplace(participant_id, std::move(bank));
                }
            }
            else if (config_.inventory_authority ==
                     MoEOverlayPhysicalInventoryAuthority::DeviceSlotLedger)
            {
                if (!impl_->device_slot_ledger)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal lost its device physical inventory authority");
                }
                std::string inventory_error;
                seal_device_inventory = impl_->device_slot_ledger->preparedSnapshot(
                    published_epoch, &inventory_error);
                if (!seal_device_inventory)
                {
                    return fail(
                        inventory_error.empty()
                            ? "ExpertOverlay reusable-context seal could not snapshot the device physical inventory"
                            : std::move(inventory_error));
                }

                std::map<int, std::size_t> bank_index_by_participant;
                for (const int participant_id : participant_ids)
                {
                    const auto endpoint =
                        config_.registry->endpoint(participant_id);
                    const auto *participant =
                        config_.initial_snapshot->placementOwnerMap().participantForId(
                            participant_id);
                    if (!endpoint || !participant ||
                        endpoint->device() != participant->device)
                    {
                        return fail(
                            "ExpertOverlay device physical inventory cannot resolve local participant " +
                            std::to_string(participant_id));
                    }

                    MoEOverlayParticipantResidencyBank bank{
                        .epoch = published_epoch,
                        .participant_id = participant_id,
                        .device = endpoint->device(),
                        .first_model_layer = endpoint->firstModelLayer(),
                    };
                    bank.layers.resize(
                        static_cast<std::size_t>(endpoint->numLayers()));
                    for (int layer_idx = endpoint->firstModelLayer(); layer_idx < endpoint->endModelLayer(); ++layer_idx)
                    {
                        auto &layer = bank.layerForModelLayer(layer_idx);
                        layer.resident_mask.assign(static_cast<std::size_t>(endpoint->numExperts()), false);
                        layer.fragment_family = impl_->device_slot_ledger->fragmentFamily(participant_id, layer_idx);
                        if (layer.fragment_family)
                            layer.fragments.resize(static_cast<std::size_t>(endpoint->numExperts()));
                        else layer.experts.resize(static_cast<std::size_t>(endpoint->numExperts()));
                    }
                    bank_index_by_participant.emplace(
                        participant_id, result.local_banks.size());
                    result.local_banks.push_back(std::move(bank));
                }

                for (const auto &slot : seal_device_inventory->slots)
                {
                    const auto bank_index = bank_index_by_participant.find(
                        slot.key.participant_id);
                    if (bank_index == bank_index_by_participant.end())
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains a non-local participant slot");
                    }
                    auto &bank = result.local_banks.at(bank_index->second);
                    if (!bank.containsModelLayer(slot.key.layer_idx) ||
                        slot.key.expert_id < 0 ||
                        static_cast<std::size_t>(slot.key.expert_id) >=
                            bank.layerForModelLayer(slot.key.layer_idx).resident_mask.size())
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains an out-of-range expert coordinate");
                    }
                    auto &layer = bank.layerForModelLayer(slot.key.layer_idx);
                    if (layer.resident_mask.at(
                            static_cast<std::size_t>(slot.key.expert_id)))
                    {
                        return fail(
                            "ExpertOverlay device physical inventory contains a duplicate expert coordinate");
                    }
                    if (const auto *whole = std::get_if<MoEOverlayPreparedExpertPayload>(&slot.payload))
                        layer.setResidentExpert(slot.key.expert_id, *whole);
                    else layer.setResidentFragment(slot.key.expert_id,
                        std::get<MoEOverlayPreparedFragmentPayload>(slot.payload));
                }
            }
            else
            {
                return fail(
                    "ExpertOverlay reusable-context seal has an unknown physical inventory authority");
            }

            /*
             * Prove the complete prepared placement before submitting a single
             * byte copy. This makes a partial physical seal impossible for an
             * owner-map or adopted-capacity defect.
             */
            std::map<
                const AdoptedInitialExpertSlotRecycler *,
                std::pair<
                    std::shared_ptr<AdoptedInitialExpertSlotRecycler>,
                    std::size_t>>
                compaction_demand_by_geometry;
            for (auto &canonical : result.local_banks)
            {
                const int participant_id = canonical.participant_id;
                const auto endpoint = config_.registry->endpoint(participant_id);
                const auto *participant =
                    config_.initial_snapshot->placementOwnerMap().participantForId(
                        participant_id);
                if (!endpoint || !participant ||
                    !canonical.valid(
                        participant_id,
                        endpoint->device(),
                        endpoint->numLayers(),
                        endpoint->numExperts(),
                        endpoint->firstModelLayer()))
                {
                    return fail(
                        "ExpertOverlay reusable-context seal materialized an incomplete local participant bank");
                }
                for (int layer_idx = endpoint->firstModelLayer();
                     layer_idx < endpoint->endModelLayer();
                     ++layer_idx)
                {
                    auto &layer = canonical.layerForModelLayer(layer_idx);
                    const auto expected_mask =
                        config_.initial_snapshot->placementOwnerMap()
                            .expertMaskForParticipant(
                                layer_idx,
                                participant_id,
                                endpoint->numExperts());
                    if (layer.resident_mask != expected_mask)
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal was requested before the initial prepared owner map was restored");
                    }

                    auto &pool = requireEndpointPool(
                        *impl_, participant_id, layer_idx);
                    if (!pool.geometry_pool ||
                        !pool.geometry_pool->adopted_slots ||
                        pool.geometry_pool->adopted_slots
                                ->initialCapacityForLayer(layer_idx) !=
                            residentCount(layer))
                    {
                        return fail(
                            "ExpertOverlay reusable-context seal found an adopted-slot capacity that disagrees with the prepared placement");
                    }

                    std::size_t needs_compaction = 0;
                    for (int expert_id = 0;
                         expert_id < endpoint->numExperts();
                         ++expert_id)
                    {
                        if (!layer.resident_mask.at(
                                static_cast<std::size_t>(expert_id)))
                        {
                            continue;
                        }
                        const bool complete = layer.fragment_family
                            ? layer.fragment_family->accepts(layer.fragments.at(size_t(expert_id)), expert_id)
                            : layer.experts.at(size_t(expert_id)).complete();
                        if (!complete)
                        {
                            return fail(
                                "ExpertOverlay reusable-context seal found an incomplete resident payload");
                        }
                        if (!pool.geometry_pool->adopted_slots
                                 ->ownsAssignment(layer_idx, expert_id))
                            ++needs_compaction;
                    }
                    auto &geometry_demand =
                        compaction_demand_by_geometry[
                            pool.geometry_pool->adopted_slots.get()];
                    if (!geometry_demand.first)
                    {
                        geometry_demand.first =
                            pool.geometry_pool->adopted_slots;
                    }
                    if (needs_compaction >
                        std::numeric_limits<std::size_t>::max() -
                            geometry_demand.second)
                    {
                        return fail(
                            "ExpertOverlay reusable-context compaction demand overflows size_t");
                    }
                    geometry_demand.second += needs_compaction;
                }
            }
            for (const auto &[_, demand] : compaction_demand_by_geometry)
            {
                if (!demand.first ||
                    demand.first->availableSlots() < demand.second)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal cannot fit every shadow resident into the shared prepared allocation arena");
                }
            }

            const auto completion_deadline =
                std::chrono::steady_clock::now() +
                std::chrono::seconds(30);

            /*
             * All operations below own setup-time lanes and persistent source/
             * destination engines. Polling observes events or worker state only;
             * it never synchronizes a device or an inference stream.
             */
            const auto complete_operations = [completion_deadline](
                std::vector<std::unique_ptr<
                    IMoEOverlayTierTransferOperation>> &operations,
                std::string *operation_error) -> bool
            {
                std::vector<bool> ready(operations.size(), false);
                std::size_t ready_count = 0;
                std::string first_failure;
                while (ready_count != operations.size())
                {
                    for (std::size_t index = 0;
                         index < operations.size();
                         ++index)
                    {
                        if (ready[index])
                            continue;
                        std::string poll_error;
                        const auto progress =
                            operations[index]->poll(&poll_error);
                        if (progress ==
                            MoEOverlayResidencyWaveProgress::Ready)
                        {
                            ready[index] = true;
                            ++ready_count;
                        }
                        else if (progress ==
                                 MoEOverlayResidencyWaveProgress::Failed)
                        {
                            first_failure = poll_error.empty()
                                ? "ExpertOverlay reusable-context physical copy failed"
                                : std::move(poll_error);
                            break;
                        }
                    }
                    if (!first_failure.empty() ||
                        std::chrono::steady_clock::now() >=
                            completion_deadline)
                    {
                        if (first_failure.empty())
                        {
                            first_failure =
                                "ExpertOverlay reusable-context physical copy exceeded the canonical 30-second protocol deadline";
                        }
                        break;
                    }
                    if (ready_count != operations.size())
                        std::this_thread::yield();
                }
                if (first_failure.empty())
                    return true;

                for (auto &operation : operations)
                    operation->abort();
                const auto abort_deadline =
                    std::chrono::steady_clock::now() +
                    std::chrono::seconds(30);
                for (;;)
                {
                    bool drained = true;
                    for (auto &operation : operations)
                    {
                        std::string ignored;
                        if (operation->pollAbort(&ignored) ==
                            MoEOverlayResidencyWaveProgress::Pending)
                        {
                            drained = false;
                        }
                    }
                    if (drained)
                        break;
                    if (std::chrono::steady_clock::now() >= abort_deadline)
                    {
                        /* Destruction with DMA or a CPU worker live is illegal. */
                        std::terminate();
                    }
                    std::this_thread::yield();
                }
                if (operation_error)
                    *operation_error = std::move(first_failure);
                return false;
            };

            const auto set_projection = [](
                std::array<std::shared_ptr<ITensorGemm>, 3> &engines,
                ExpertTierWeightProjection projection,
                std::shared_ptr<ITensorGemm> engine)
            {
                switch (projection)
                {
                case ExpertTierWeightProjection::Gate:
                    engines[0] = std::move(engine);
                    return;
                case ExpertTierWeightProjection::Up:
                    engines[1] = std::move(engine);
                    return;
                case ExpertTierWeightProjection::Down:
                    engines[2] = std::move(engine);
                    return;
                }
                throw std::logic_error(
                    "ExpertOverlay reusable-context seal received an unknown projection role");
            };

            for (auto &bank : result.local_banks)
            {
                const auto endpoint =
                    config_.registry->endpoint(bank.participant_id);
                const auto *participant =
                    config_.initial_snapshot->placementOwnerMap().participantForId(
                        bank.participant_id);
                if (!endpoint || !participant)
                {
                    return fail(
                        "ExpertOverlay reusable-context seal lost a preflighted participant");
                }

                for (int layer_idx = endpoint->firstModelLayer();
                     layer_idx < endpoint->endModelLayer();
                     ++layer_idx)
                {
                    auto &layer = bank.layerForModelLayer(layer_idx);
                    auto &pool = requireEndpointPool(
                        *impl_, bank.participant_id, layer_idx);
                    if (!pool.geometry_pool ||
                        !pool.geometry_pool->adopted_slots)
                    {
                        return fail(
                            "ExpertOverlay reusable-context compaction lost its geometry arena");
                    }
                    for (int expert_id = 0;
                         expert_id < endpoint->numExperts();
                         ++expert_id)
                    {
                        if (!layer.resident_mask.at(
                                static_cast<std::size_t>(expert_id)))
                        {
                            continue;
                        }
                        if (pool.geometry_pool->adopted_slots
                                ->ownsAssignment(layer_idx, expert_id))
                        {
                            ++result.retained_canonical_experts;
                            continue;
                        }

                        std::array<std::shared_ptr<ITensorGemm>, 3> source_engines;
                        for (const auto role : kProjections)
                            source_engines[projectionIndex(role)] = projectionEngine(layer, size_t(expert_id), role);
                        const auto projection_set = layer.fragment_family ? DeviceMoEProjectionSet::IntermediateShard
                            : layer.experts.at(size_t(expert_id)).projections();
                        std::array<std::shared_ptr<ITensorGemm>, 3> destination_engines{};
                        std::vector<std::unique_ptr<
                            IMoEOverlayTierTransferOperation>> operations;
                        operations.reserve(kProjections.size());

                        if (bank.device.is_cpu())
                        {
                            auto destination =
                                pool.geometry_pool->adopted_slots->acquireCpu(
                                    layer_idx,
                                    expert_id,
                                    published_epoch);
                            if (!destination)
                            {
                                return fail(
                                    "ExpertOverlay reusable-context seal lost a preflighted CPU adopted slot");
                            }
                            auto gate = std::make_shared<CpuCopyWaveGate>(
                                kProjections.size());
                            for (const auto projection : kProjections)
                            {
                                auto source_engine = source_engines[projectionIndex(projection)];
                                MoEOverlayPreparedWeightSource source;
                                std::string source_error;
                                if (!resolveMoEOverlayPreparedWeightSource(
                                        source_engine,
                                        bank.device,
                                        source,
                                        &source_error))
                                {
                                    return fail(std::move(source_error));
                                }
                                const auto &target = cpuProjection(
                                    *destination, projection);
                                std::span<const std::uint8_t> source_bytes;
                                if (source.format.isFloating())
                                {
                                    source_bytes = {
                                        static_cast<const std::uint8_t *>(
                                            source.floating.data),
                                        source.floating.bytes,
                                    };
                                }
                                else if (source.cpu_packed)
                                {
                                    source_bytes = {
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size(),
                                    };
                                }
                                if (source_bytes.empty() ||
                                    source_bytes.size() !=
                                        target.destination_bytes.size())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context CPU source and prepared allocation disagree on byte size");
                                }
                                const CpuAddressEdgeKey edge{
                                    .source = participant->address,
                                    .destination = participant->address,
                                    .projection = projection,
                                };
                                const auto lane =
                                    impl_->cpu_copy_lanes.find(edge);
                                if (lane == impl_->cpu_copy_lanes.end())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context CPU compaction lane was not materialized at setup");
                                }
                                CPUCopyGeometry copy_geometry = CompleteCPUCopy{};
                                if (source.cpu_packed)
                                {
                                    const auto *packed = target.engine->exportCPUNativeQuantisedPackedWeights();
                                    if (!packed) return fail("Canonical CPU compaction changed original native storage");
                                    const auto owned = layer.fragment_family
                                        ? layer.fragment_family->cohort()->ownership(layer.fragment_family->participantIndex())
                                            .projection(projection == ExpertTierWeightProjection::Gate ? WeightRole::MoEExpertGate
                                                : projection == ExpertTierWeightProjection::Up ? WeightRole::MoEExpertUp : WeightRole::MoEExpertDown)
                                        : wholeProjectionIntersection(source.cpu_packed->N, source.cpu_packed->K).source();
                                    copy_geometry = nativeCPUCopyGeometry(*source.cpu_packed, *packed,
                                        ExpertTierWeightProjectionIntersection::between(owned, owned),
                                        config_.staging_capacity_bytes, std::make_shared<std::mutex>());
                                }
                                operations.push_back(std::make_unique<AdmittedCpuCopyOperation>(lane->second,
                                    source_bytes, target.destination_bytes, gate, source_engine, target.engine,
                                    std::move(copy_geometry)));
                                set_projection(
                                    destination_engines,
                                    projection,
                                    target.engine);
                            }
                        }
                        else
                        {
                            auto destination =
                                pool.geometry_pool->adopted_slots->acquireGpu(
                                    layer_idx,
                                    expert_id,
                                    published_epoch);
                            if (!destination)
                            {
                                return fail(
                                    "ExpertOverlay reusable-context seal lost a preflighted GPU adopted slot");
                            }
                            for (const auto projection : kProjections)
                            {
                                auto source_engine = source_engines[projectionIndex(projection)];
                                MoEOverlayPreparedWeightSource source;
                                std::string source_error;
                                if (!resolveMoEOverlayPreparedWeightSource(
                                        source_engine,
                                        bank.device,
                                        source,
                                        &source_error))
                                {
                                    return fail(std::move(source_error));
                                }
                                const auto &target = gpuProjection(
                                    *destination, projection);
                                const EdgeKey edge{
                                    .source = bank.device,
                                    .destination = bank.device,
                                    .projection = projection,
                                };
                                const auto lane = impl_->peer_lanes.find(edge);
                                if (lane == impl_->peer_lanes.end())
                                {
                                    return fail(
                                        "ExpertOverlay reusable-context GPU compaction lane was not materialized at setup");
                                }

                                const auto signature=signatureFor(source);
                                const auto owned=layer.fragment_family
                                    ? layer.fragment_family->cohort()->ownership(layer.fragment_family->participantIndex())
                                        .projection(projection==ExpertTierWeightProjection::Gate?WeightRole::MoEExpertGate:
                                            projection==ExpertTierWeightProjection::Up?WeightRole::MoEExpertUp:WeightRole::MoEExpertDown)
                                    : wholeProjectionIntersection(signature.N,signature.K).source();
                                const auto intersection=ExpertTierWeightProjectionIntersection::between(owned,owned);
                                std::shared_ptr<ITensorGemm> target_engine;
                                if (source.format.isFloating())
                                {
                                    const auto descriptor =
                                        makeGpuFloatingDestinationDescriptor(
                                            target, source.format);
                                    target_engine =
                                        makeGpuFloatingDestinationEngine(
                                            bank.device,
                                            descriptor,
                                            destination->lifetime);
                                    operations.push_back(
                                        std::make_unique<
                                            AdmittedPeerOperation>(
                                            lane->second,
                                            source.floating,
                                            descriptor,
                                            ExpertTierSourceReadiness::
                                                publishedResidencyBank(
                                                    published_epoch),
                                            source_engine,
                                            target_engine,intersection));
                                }
                                else
                                {
                                    const auto descriptor =
                                        makeGpuDestinationDescriptor(
                                            target,
                                            source.gpu_packed.n,
                                            source.gpu_packed.k,
                                            source.gpu_packed.codebook_id,
                                            source.gpu_packed
                                                .payload_bytes_per_block,
                                            source.gpu_packed.is_asymmetric,
                                            source.gpu_packed.has_emins,
                                            source.format.native_quantised);
                                    target_engine = makeGpuDestinationEngine(
                                        bank.device,
                                        descriptor,
                                        destination->lifetime,
                                        source.format.native_quantised);
                                    operations.push_back(
                                        std::make_unique<
                                            AdmittedPeerOperation>(
                                            lane->second,
                                            source.gpu_packed,
                                            descriptor,
                                            ExpertTierSourceReadiness::
                                                publishedResidencyBank(
                                                    published_epoch),
                                            source_engine,
                                            target_engine,intersection));
                                }
                                set_projection(
                                    destination_engines,
                                    projection,
                                    std::move(target_engine));
                            }
                        }

                        std::string operation_error;
                        std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival;
                        if (layer.fragment_family)
                        {
                            if (seal_device_inventory)
                                arrival = config_.registry->prepareJointDeviceInventoryArrival(*seal_device_inventory,
                                    {.participant_id = bank.participant_id, .layer_idx = layer_idx,
                                        .expert_id = expert_id}, &operation_error);
                            else
                            {
                                const auto borrow = seal_source_banks.find(bank.participant_id);
                                if (borrow == seal_source_banks.end())
                                    return fail("Fragment terminal compaction lost its original published-bank borrow");
                                arrival = config_.registry->prepareJointArrivalFromInstalledBank(borrow->second,
                                    bank.participant_id, layer_idx, expert_id, &operation_error);
                            }
                            if (!arrival) return fail(operation_error);
                        }
                        else arrival = std::make_shared<MoEOverlayPreparedExpertArrival>(projection_set);
                        for (const auto role : kProjections)
                        {
                            const auto index = projectionIndex(role);
                            MoEOverlayPreparedWeightSource original;
                            if (!resolveMoEOverlayPreparedWeightSource(source_engines[index], bank.device,
                                    original, &operation_error)) return fail(operation_error);
                            const auto signature = signatureFor(original);
                            const auto owned = layer.fragment_family
                                ? layer.fragment_family->cohort()->ownership(layer.fragment_family->participantIndex())
                                    .projection(role == ExpertTierWeightProjection::Gate ? WeightRole::MoEExpertGate
                                        : role == ExpertTierWeightProjection::Up ? WeightRole::MoEExpertUp : WeightRole::MoEExpertDown)
                                : wholeProjectionIntersection(signature.N, signature.K).source();
                            auto completion = std::make_shared<MoEOverlayPreparedProjectionCompletion>(
                                std::vector{ExpertTierWeightProjectionIntersection::between(owned, owned)}, arrival, role,
                                [engine = destination_engines[index]] { return engine; });
                            operations[index] = std::make_unique<MoEOverlayPreparedProjectionOperation>(
                                std::move(operations[index]), std::move(completion), 0);
                        }
                        if (!complete_operations(operations, &operation_error)) return fail(operation_error);
                        if (layer.fragment_family)
                        {
                            MoEOverlayPreparedFragmentPayload fragment;
                            if (!arrival->completeFragment(fragment, &operation_error)) return fail(operation_error);
                            layer.setResidentFragment(expert_id, std::move(fragment));
                        }
                        else
                        {
                            MoEOverlayPreparedExpertPayload payload;
                            if (!arrival->completePayload(payload, &operation_error)) return fail(operation_error);
                            layer.setResidentExpert(expert_id, std::move(payload));
                        }
                        ++result.compacted_shadow_experts;
                    }
                }
                if (!bank.valid(
                        bank.participant_id,
                        bank.device,
                        endpoint->numLayers(),
                        endpoint->numExperts(),
                        endpoint->firstModelLayer()))
                {
                    return fail(
                        "ExpertOverlay reusable-context seal produced an invalid canonical bank");
                }
            }

            if (!result.valid())
            {
                return fail(
                    "ExpertOverlay reusable-context seal produced an incomplete terminal value");
            }
            impl_->cached_reusable_seal = std::move(result);
            impl_->reusable_seal_state = Impl::ReusableSealState::Sealed;
            recordMoEReusableContextSeal(published_epoch,
                impl_->cached_reusable_seal->retained_canonical_experts,
                impl_->cached_reusable_seal->compacted_shadow_experts,
                config_.perf_device);
            return impl_->cached_reusable_seal;
        }
        catch (const std::exception &exception)
        {
            return fail(exception.what());
        }
        catch (...)
        {
            return fail(
                "ExpertOverlay reusable-context seal raised a non-standard exception");
        }
    }

    MoEOverlayPhysicalResidencyFabricStats
    MoEOverlayPhysicalResidencyFabric::stats() const noexcept
    {
        std::uint64_t parallel_lane_reservations = 0;
        std::uint64_t parallel_lane_pool_exhaustions = 0;
        const auto accumulate_pool_stats = [&](const auto &pools)
        {
            for (const auto &[_, pool] : pools)
            {
                parallel_lane_reservations += pool->reservations();
                parallel_lane_pool_exhaustions += pool->exhaustions();
            }
        };
        accumulate_pool_stats(impl_->weight_lanes);
        accumulate_pool_stats(impl_->blob_lanes);
        accumulate_pool_stats(impl_->peer_lanes);
        std::uint64_t parallel_cpu_copy_lane_reservations = 0;
        std::uint64_t parallel_cpu_copy_lane_pool_exhaustions = 0;
        for (const auto &[_, pool] : impl_->cpu_copy_lanes)
        {
            parallel_cpu_copy_lane_reservations += pool->reservations();
            parallel_cpu_copy_lane_pool_exhaustions += pool->exhaustions();
        }

        return {
            .endpoint_layer_pools =
                impl_->endpoint_layer_pools.load(std::memory_order_relaxed),
            .endpoint_geometry_pools =
                impl_->endpoint_geometry_pools.load(
                    std::memory_order_relaxed),
            .adopted_initial_slots =
                impl_->adopted_initial_slots.load(
                    std::memory_order_relaxed),
            .adopted_initial_slots_recycled =
                impl_->adopted_initial_slots_recycled.load(
                    std::memory_order_relaxed),
            .cpu_shadow_slots =
                impl_->cpu_shadow_slots.load(std::memory_order_relaxed),
            .gpu_shadow_slots =
                impl_->gpu_shadow_slots.load(std::memory_order_relaxed),
            .persistent_transfer_lanes =
                impl_->persistent_transfer_lanes.load(
                    std::memory_order_relaxed),
            .direct_gpu_peer_lanes =
                impl_->direct_gpu_peer_lanes.load(
                    std::memory_order_relaxed),
            .same_backend_no_peer_relay_lanes =
                impl_->same_backend_no_peer_relay_lanes.load(
                    std::memory_order_relaxed),
            .cross_backend_gpu_relay_lanes =
                impl_->cross_backend_gpu_relay_lanes.load(
                    std::memory_order_relaxed),
            .maximum_parallel_edge_lanes =
                impl_->maximum_parallel_edge_lanes.load(
                    std::memory_order_relaxed),
            .parallel_lane_reservations = parallel_lane_reservations,
            .parallel_lane_pool_exhaustions =
                parallel_lane_pool_exhaustions,
            .serialized_lane_deferrals = 0,
            .maximum_parallel_cpu_copy_lanes =
                impl_->maximum_parallel_cpu_copy_lanes.load(
                    std::memory_order_relaxed),
            .parallel_cpu_copy_lane_reservations =
                parallel_cpu_copy_lane_reservations,
            .parallel_cpu_copy_lane_pool_exhaustions =
                parallel_cpu_copy_lane_pool_exhaustions,
            .maximum_concurrent_cpu_copy_operations =
                impl_->cpu_copy_concurrency->peak(),
            .active_cpu_copy_operations =
                impl_->cpu_copy_concurrency->active(),
            .waves_prepared =
                impl_->waves_prepared.load(std::memory_order_relaxed),
            .waves_deferred =
                impl_->waves_deferred.load(std::memory_order_relaxed),
            .waves_failed =
                impl_->waves_failed.load(std::memory_order_relaxed),
            .projection_operations_prepared =
                impl_->projection_operations_prepared.load(
                    std::memory_order_relaxed),
            .cpu_copy_operations =
                impl_->cpu_copy_operations.load(std::memory_order_relaxed),
            .remote_cpu_operations =
                impl_->remote_cpu_operations.load(
                    std::memory_order_relaxed),
            .remote_gpu_cpu_operations =
                impl_->remote_gpu_cpu_operations.load(
                    std::memory_order_relaxed),
            .remote_gpu_blob_operations =
                impl_->remote_gpu_blob_operations.load(
                    std::memory_order_relaxed),
            .gpu_cpu_operations =
                impl_->gpu_cpu_operations.load(std::memory_order_relaxed),
            .same_backend_gpu_operations =
                impl_->same_backend_gpu_operations.load(
                    std::memory_order_relaxed),
            .heterogeneous_gpu_operations =
                impl_->heterogeneous_gpu_operations.load(
                    std::memory_order_relaxed),
            .inference_stream_waits = 0,
            .blocking_synchronizations = 0,
        };
    }
} // namespace llaminar2
