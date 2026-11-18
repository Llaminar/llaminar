/**
 * @file MoEOverlayDevicePhysicalSlotLedger.cpp
 * @brief Transactional physical lifetime bookkeeping for device-owned MoE RCU.
 *
 * The ledger retains prepared allocation lifetimes within one immutable
 * pipeline stage. It never owns placement policy or a second memory budget.
 * Global layer keys and stage-bound transaction receipts prevent a same-shaped
 * peer stage from publishing or retiring these allocations. Fixed original
 * whole/fragment rows share one mixed transaction and complete member receipt
 * set; neither source borrowing nor terminal snapshots flatten fragment semantics.
 */

#include "MoEOverlayDevicePhysicalSlotLedger.h"
#include "MoEExpertShardCohort.h"
#include "MoEExpertOwnerMap.h"

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>

namespace llaminar2
{
    namespace
    {
        /** Populate an optional diagnostic and return false for guard clauses. */
        bool reject(std::string *error, std::string message) noexcept
        {
            if (error)
                *error = std::move(message);
            return false;
        }

        /** @return Actual original physical members; a cohort never acquires a scalar owner. */
        std::vector<MoEOverlayDevicePhysicalSlotKey> physicalKeys(
            const MoEOverlayTierMigration &migration, bool source)
        {
            const auto target = MoEExpertPlacementTarget::fromPlacement(
                source ? migration.source : migration.destination);
            std::vector<MoEOverlayDevicePhysicalSlotKey> keys;
            if (!target) return keys;
            keys.reserve(target->participantCount());
            for (std::size_t member = 0; member < target->participantCount(); ++member)
                keys.push_back({target->participantId(member), migration.layer_idx, migration.expert_id});
            return keys;
        }
    } // namespace

    bool MoEOverlayDevicePhysicalSlotKey::operator<(
        const MoEOverlayDevicePhysicalSlotKey &other) const noexcept
    {
        return std::tie(participant_id, layer_idx, expert_id) <
               std::tie(
                   other.participant_id,
                   other.layer_idx,
                   other.expert_id);
    }

    bool MoEOverlayDeviceFragmentPhysicalSlot::valid() const noexcept
    {
        if (!key.valid() || !entered_epoch || !payload.valid() ||
            payload.expertId() != key.expert_id || payload.arrivalEpoch() > entered_epoch ||
            payload.cohort()->source().layer_idx != key.layer_idx)
            return false;
        const auto members = payload.cohort()->participants();
        return payload.participantIndex() < members.size() &&
            members[payload.participantIndex()].participant_id == key.participant_id;
    }

    bool MoEOverlayDeviceFragmentInventorySnapshot::valid() const noexcept
    {
        return epoch != 0u && std::all_of(slots.begin(), slots.end(),
            [this](const auto &slot) { return slot.valid() && slot.entered_epoch <= epoch; });
    }

    bool MoEOverlayDeviceInitialPhysicalSlot::valid() const noexcept
    {
        return key.valid() && entered_epoch != 0u && payload.ready();
    }

    bool MoEOverlayDeviceStagedPhysicalArrival::valid() const noexcept
    {
        return key.valid() && payload.ready();
    }

    bool MoEOverlayDeviceActivePhysicalSlot::valid() const noexcept
    {
        return key.valid() && entered_epoch != 0u && payload.ready();
    }

    bool MoEOverlayDevicePhysicalInventorySnapshot::valid() const noexcept
    {
        return epoch != 0u && std::all_of(
            slots.begin(),
            slots.end(),
            [](const MoEOverlayDeviceActivePhysicalSlot &slot)
            { return slot.valid(); });
    }

    /** @brief Private active inventory and sole pending transaction. */
    struct MoEOverlayDevicePhysicalSlotLedger::Impl
    {
        /** Bind the setup namespace before enrolling any allocation lifetime. */
        explicit Impl(const Config &config)
            : first_model_layer(config.first_model_layer),
              num_layers(config.num_layers), num_experts(config.num_experts)
        {
        }

        /** @return Whether immutable command geometry names this exact stage. */
        bool matchesScope(const MoEOverlayDevicePhysicalMovementBatch &batch) const noexcept
        {
            return batch.first_model_layer == first_model_layer &&
                   batch.num_layers == num_layers && batch.num_experts == num_experts;
        }

        /** @return Whether an allocation key belongs to the admitted model rows. */
        bool contains(const MoEOverlayDevicePhysicalSlotKey &key) const noexcept
        {
            return key.layer_idx >= first_model_layer && key.expert_id >= 0 &&
                   static_cast<std::uint32_t>(key.layer_idx - first_model_layer) < num_layers &&
                   static_cast<std::uint32_t>(key.expert_id) < num_experts;
        }

        using Payload = MoEOverlayDevicePreparedPhysicalPayload;
        using Families = std::map<std::pair<int, int>, std::shared_ptr<const MoEOverlayFragmentFamily>>;

        /** @return Whether this ledger retains closed fragment families. */
        bool fragments() const noexcept { return fragments_only; }

        /** @return Whether the legacy whole-only interface describes every admitted row. */
        bool whole() const noexcept { return families.empty(); }

        /** @return Whether the exact original family, coordinate and epoch are authenticated. */
        bool accepts(const MoEOverlayDevicePhysicalSlotKey &key, const Payload &payload,
                     std::uint64_t maximum_epoch, bool arrival = false) const noexcept
        {
            if (!key.valid() || !contains(key) || !local(key.participant_id)) return false;
            if (const auto *whole = std::get_if<MoEOverlayPreparedExpertPayload>(&payload))
                return !families.contains({key.participant_id, key.layer_idx}) && whole->readyFor(whole_family);
            const auto &fragment = std::get<MoEOverlayPreparedFragmentPayload>(payload);
            const auto found = families.find({key.participant_id, key.layer_idx});
            return found != families.end() && found->second->accepts(fragment, key.expert_id) &&
                (arrival ? fragment.arrivalEpoch() == maximum_epoch : fragment.arrivalEpoch() <= maximum_epoch);
        }

        /** @return Whether a command names the SAME retained original source/cohort authority. */
        bool matchesPlacement(const MoEOverlayDevicePhysicalSlotKey &key,
                              const MoEExpertOwner &placement) const noexcept
        {
            const auto found = families.find({key.participant_id, key.layer_idx});
            if (found == families.end()) return !placement.shard_cohort;
            const auto &original = found->second->cohort();
            return original == placement.shard_cohort &&
                !original.owner_before(placement.shard_cohort) &&
                !placement.shard_cohort.owner_before(original);
        }

        /** Physical allocation retained independently of device descriptor bytes. */
        struct ActiveSlot
        {
            std::uint64_t entered_epoch = 0u;
            bool bootstrap_allocation = false;
            Payload payload;
        };

        /** Immutable identity and mutable transport phase of one Dynamic wave. */
        struct PendingWave
        {
            std::uint64_t topology_fingerprint = 0u;
            std::uint64_t transaction_id = 0u;
            std::uint64_t base_epoch = 0u;
            std::uint64_t candidate_epoch = 0u;
            std::uint64_t command_digest = 0u;
            MoEOverlayResidencyExecutionFingerprint fingerprint;
            std::vector<MoEOverlayTierMigration> migrations;
            std::map<
                MoEOverlayDevicePhysicalSlotKey,
                Payload>
                staged_arrivals;
            bool staged = false;
            bool published = false;
        };

        /** @return Whether @p batch is exactly the pending authenticated wave. */
        bool exact(
            const MoEOverlayDevicePhysicalMovementBatch &batch) const noexcept
        {
            return matchesScope(batch) && pending.has_value() &&
                   pending->topology_fingerprint ==
                       batch.topology_fingerprint &&
                   pending->transaction_id == batch.transaction_id &&
                   pending->base_epoch == batch.base_epoch &&
                   pending->candidate_epoch == batch.candidate_epoch &&
                   pending->command_digest == batch.command_digest &&
                   pending->fingerprint == batch.execution_fingerprint;
        }

        /** @return Whether this process owns @p participant_id. */
        bool local(int participant_id) const noexcept
        {
            return std::binary_search(
                local_participant_ids.begin(),
                local_participant_ids.end(),
                participant_id);
        }

        /** @brief Stage either closed payload arm through the same wave lifecycle. */
        template <class Arrival> bool stage(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<Arrival> arrivals, std::string *error) noexcept;
        /** @brief Retire either closed payload arm after the same reader barrier. */
        template <class Slot> bool retire(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<Slot> *retired, std::string *error) noexcept;

        mutable std::mutex mutex;
        const int first_model_layer;
        const std::uint32_t num_layers;
        const std::uint32_t num_experts;
        std::uint64_t current_epoch = 0u;
        std::vector<int> local_participant_ids;
        DeviceMoEProjectionSet whole_family = DeviceMoEProjectionSet::CompleteExpert;
        Families families;
        bool fragments_only = false;
        std::map<MoEOverlayDevicePhysicalSlotKey, ActiveSlot> active;
        std::optional<PendingWave> pending;
    };

    MoEOverlayDevicePhysicalSlotLedger::MoEOverlayDevicePhysicalSlotLedger(
        Config config)
        : impl_(std::make_unique<Impl>(config))
    {
        if (config.first_model_layer < 0 || config.num_layers == 0u ||
            config.num_layers > static_cast<std::uint32_t>(
                std::numeric_limits<int>::max() - config.first_model_layer) ||
            config.num_experts == 0u ||
            (config.movable_projections != DeviceMoEProjectionSet::CompleteExpert &&
             config.movable_projections != DeviceMoEProjectionSet::GateUp) || config.initial_epoch == 0u ||
            (!config.local_participant_ids.empty() && config.local_participant_ids.front() < 0) ||
            !std::is_sorted(
                config.local_participant_ids.begin(),
                config.local_participant_ids.end()) ||
            std::adjacent_find(
                config.local_participant_ids.begin(),
                config.local_participant_ids.end()) !=
                config.local_participant_ids.end())
        {
            throw std::invalid_argument(
                "device physical slot ledger requires valid stage geometry, a positive epoch and sorted unique nonnegative local participant IDs");
        }
        impl_->current_epoch = config.initial_epoch;
        impl_->whole_family = config.movable_projections;
        impl_->local_participant_ids =
            std::move(config.local_participant_ids);
        for (auto &slot : config.initial_slots)
        {
            if (!slot.valid() || !impl_->accepts(slot.key, slot.payload, slot.entered_epoch) ||
                slot.entered_epoch > config.initial_epoch ||
                !impl_->local(slot.key.participant_id) || !impl_->contains(slot.key))
            {
                throw std::invalid_argument(
                    "device physical slot ledger initial inventory is invalid or non-local");
            }
            const auto [unused, inserted] = impl_->active.emplace(
                slot.key,
                Impl::ActiveSlot{
                    .entered_epoch = slot.entered_epoch,
                    .bootstrap_allocation = slot.bootstrap_allocation,
                    .payload = std::move(slot.payload),
                });
            if (!inserted)
                throw std::invalid_argument(
                    "device physical slot ledger initial inventory contains a duplicate coordinate");
        }
    }

    std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger>
    MoEOverlayDevicePhysicalSlotLedger::createFragments(FragmentConfig config)
    {
        return std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger>(
            new MoEOverlayDevicePhysicalSlotLedger(std::move(config), FragmentEnrollment{}));
    }

    std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger>
    MoEOverlayDevicePhysicalSlotLedger::createMixed(MixedConfig config)
    {
        return std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger>(
            new MoEOverlayDevicePhysicalSlotLedger(std::move(config), MixedEnrollment{}));
    }

    MoEOverlayDevicePhysicalSlotLedger::MoEOverlayDevicePhysicalSlotLedger(FragmentConfig config, FragmentEnrollment)
        : MoEOverlayDevicePhysicalSlotLedger(MixedConfig{
              .whole = {.initial_epoch = config.initial_epoch,
                  .local_participant_ids = config.local_participant_ids,
                  .first_model_layer = config.first_model_layer,
                  .num_layers = config.num_layers, .num_experts = config.num_experts},
              .families = std::move(config.families),
              .initial_fragments = std::move(config.initial_slots)}, MixedEnrollment{})
    {
        // The legacy fragment-only factory requires every local row to be a fragment.
        for (const int participant : impl_->local_participant_ids)
            for (std::uint32_t layer = 0; layer < impl_->num_layers; ++layer)
                if (!impl_->families.contains({participant, impl_->first_model_layer + static_cast<int>(layer)}))
                    throw std::invalid_argument("device fragment slot ledger requires complete local family coverage");
        impl_->fragments_only = true;
    }

    MoEOverlayDevicePhysicalSlotLedger::MoEOverlayDevicePhysicalSlotLedger(MixedConfig config, MixedEnrollment)
        : MoEOverlayDevicePhysicalSlotLedger(std::move(config.whole))
    {
        Impl::Families families;
        ModelContextId original_model{};
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> original_preparation;
        for (auto &entry : config.families)
        {
            const MoEOverlayDevicePhysicalSlotKey key{entry.participant_id, entry.layer_idx, 0};
            if (!key.valid() || !impl_->local(entry.participant_id) || !impl_->contains(key) ||
                !entry.family || !entry.family->cohort() || !entry.family->preparation() ||
                !entry.family->model().value)
                throw std::invalid_argument("device fragment slot ledger requires original local families");
            if (!original_preparation)
            {
                original_model = entry.family->model();
                original_preparation = entry.family->preparation();
            }
            const auto &preparation = entry.family->preparation();
            if (entry.family->model() != original_model || preparation != original_preparation ||
                preparation.owner_before(original_preparation) || original_preparation.owner_before(preparation))
                throw std::invalid_argument("device fragment slot ledger cannot mix original model/preparation authorities");
            const auto &cohort = *entry.family->cohort();
            const auto member = entry.family->participantIndex();
            if (member >= cohort.participants().size() ||
                cohort.participants()[member].participant_id != entry.participant_id ||
                cohort.source().layer_idx != entry.layer_idx ||
                cohort.ownership(member).geometry().experts != static_cast<int>(impl_->num_experts) ||
                !families.emplace(std::make_pair(entry.participant_id, entry.layer_idx), std::move(entry.family)).second)
                throw std::invalid_argument("device fragment slot ledger rejected foreign member or source geometry");
        }
        impl_->families = std::move(families);
        // A row's fixed projection contract cannot change beneath a retained source.
        for (const auto &[key, slot] : impl_->active)
            if (!impl_->accepts(key, slot.payload, slot.entered_epoch))
                throw std::invalid_argument("device mixed ledger whole source conflicts with its fragment row");
        for (auto &slot : config.initial_fragments)
        {
            if (!slot.valid() || slot.entered_epoch > impl_->current_epoch ||
                !impl_->accepts(slot.key, slot.payload, slot.entered_epoch) ||
                !impl_->active.emplace(slot.key, Impl::ActiveSlot{
                    .entered_epoch = slot.entered_epoch, .bootstrap_allocation = slot.bootstrap_allocation,
                    .payload = std::move(slot.payload)}).second)
                throw std::invalid_argument("device fragment slot ledger rejected initial fragment family, epoch or coordinate");
        }
    }

    MoEOverlayDevicePhysicalSlotLedger::~MoEOverlayDevicePhysicalSlotLedger() =
        default;

    bool MoEOverlayDevicePhysicalSlotLedger::begin(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!impl_->matchesScope(batch) || !batch.valid() || !batch.movesWeights() ||
            (batch.kind !=
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
             batch.kind != MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore))
        {
            return reject(
                error,
                "device physical slot ledger requires non-empty durable placement movement for its exact stage");
        }

        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->pending)
            return reject(
                error,
                "device physical slot ledger already owns an unfinished wave");
        if (batch.base_epoch != impl_->current_epoch)
            return reject(
                error,
                "device physical slot ledger rejected a stale base epoch");

        std::set<MoEOverlayDevicePhysicalSlotKey> local_destinations;
        for (const auto &migration : batch.migrations)
        {
            for (const auto &source : physicalKeys(migration, true))
            {
                if (!impl_->local(source.participant_id)) continue;
                const auto found = impl_->active.find(source);
                if (found == impl_->active.end() || !impl_->matchesPlacement(source, migration.source) ||
                    !impl_->accepts(source, found->second.payload, found->second.entered_epoch))
                {
                    return reject(
                        error,
                        "device physical slot ledger cannot resolve a local source allocation in its original family");
                }
            }
            for (const auto &destination : physicalKeys(migration, false))
            {
                if (impl_->local(destination.participant_id) &&
                    (!impl_->matchesPlacement(destination, migration.destination) ||
                     impl_->active.contains(destination) ||
                     !local_destinations.insert(destination).second))
                {
                    return reject(
                        error,
                        "device physical slot ledger destination is foreign, already active or duplicated");
                }
            }
        }

        impl_->pending = Impl::PendingWave{
            .topology_fingerprint = batch.topology_fingerprint,
            .transaction_id = batch.transaction_id,
            .base_epoch = batch.base_epoch,
            .candidate_epoch = batch.candidate_epoch,
            .command_digest = batch.command_digest,
            .fingerprint = batch.execution_fingerprint,
            .migrations = batch.migrations,
        };
        return true;
    }

    std::optional<MoEOverlayPreparedExpertPayload>
    MoEOverlayDevicePhysicalSlotLedger::sourcePayload(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayDevicePhysicalSlotKey &key,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!key.valid() || !impl_->contains(key) || !impl_->exact(batch) ||
            !impl_->local(key.participant_id))
        {
            reject(
                error,
                "device physical slot ledger source request has invalid wave or endpoint identity");
            return std::nullopt;
        }
        const bool named = std::any_of(impl_->pending->migrations.begin(), impl_->pending->migrations.end(),
            [&](const auto &migration) {
                const auto keys = physicalKeys(migration, true);
                return std::find(keys.begin(), keys.end(), key) != keys.end();
            });
        const auto found = impl_->active.find(key);
        if (!named || found == impl_->active.end() || !std::holds_alternative<MoEOverlayPreparedExpertPayload>(found->second.payload) ||
            !impl_->accepts(key, found->second.payload, found->second.entered_epoch))
        {
            reject(
                error,
                "device physical slot ledger source is not active");
            return std::nullopt;
        }
        return std::get<MoEOverlayPreparedExpertPayload>(found->second.payload);
    }

    std::optional<MoEOverlayPreparedFragmentPayload>
    MoEOverlayDevicePhysicalSlotLedger::sourceFragmentPayload(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayDevicePhysicalSlotKey &key, std::string *error) const noexcept
    {
        if (error) error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->exact(batch))
        {
            reject(error, "device fragment slot ledger source has wrong payload arm or wave identity");
            return std::nullopt;
        }
        const bool named = std::any_of(impl_->pending->migrations.begin(), impl_->pending->migrations.end(),
            [&](const auto &migration) {
                const auto keys = physicalKeys(migration, true);
                return std::find(keys.begin(), keys.end(), key) != keys.end();
            });
        const auto found = impl_->active.find(key);
        if (!named || found == impl_->active.end() ||
            !std::holds_alternative<MoEOverlayPreparedFragmentPayload>(found->second.payload) ||
            !impl_->accepts(key, found->second.payload, found->second.entered_epoch))
        {
            reject(error, "device fragment slot ledger source is not a named original local fragment");
            return std::nullopt;
        }
        return std::get<MoEOverlayPreparedFragmentPayload>(found->second.payload);
    }

    template <class Arrival>
    bool MoEOverlayDevicePhysicalSlotLedger::Impl::stage(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<Arrival> arrivals,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(mutex);
        if ((std::is_same_v<Arrival, MoEOverlayDeviceStagedFragmentArrival> && !fragments()) ||
            (std::is_same_v<Arrival, MoEOverlayDeviceStagedPhysicalArrival> && !whole()) ||
            !exact(batch) || pending->staged ||
            pending->published)
        {
            return reject(
                error,
                "device physical slot ledger stage has invalid lifecycle identity");
        }

        std::set<MoEOverlayDevicePhysicalSlotKey> expected;
        for (const auto &migration : pending->migrations)
        {
            for (const auto &key : physicalKeys(migration, false))
                if (local(key.participant_id)) expected.insert(key);
        }

        std::map<
            MoEOverlayDevicePhysicalSlotKey,
            Payload>
            staged;
        for (auto &arrival : arrivals)
        {
            if (!accepts(arrival.key, arrival.payload, batch.candidate_epoch, true) ||
                !expected.contains(arrival.key) ||
                !staged.emplace(
                           arrival.key, std::move(arrival.payload)).second)
            {
                return reject(
                    error,
                    "device physical slot ledger received an invalid, foreign, or duplicate arrival");
            }
        }
        if (staged.size() != expected.size())
            return reject(
                error,
                "device physical slot ledger did not receive every local destination");

        pending->staged_arrivals = std::move(staged);
        pending->staged = true;
        return true;
    }

    bool MoEOverlayDevicePhysicalSlotLedger::stage(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceStagedPhysicalArrival> arrivals, std::string *error) noexcept
    {
        return impl_->stage(batch, std::move(arrivals), error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::stageFragments(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceStagedFragmentArrival> arrivals, std::string *error) noexcept
    {
        return impl_->stage(batch, std::move(arrivals), error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::stagePrepared(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDevicePreparedPhysicalArrival> arrivals, std::string *error) noexcept
    {
        return impl_->stage(batch, std::move(arrivals), error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::publish(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->exact(batch) || !impl_->pending->staged ||
            impl_->pending->published ||
            impl_->current_epoch != batch.base_epoch)
        {
            return reject(
                error,
                "device physical slot ledger publish has invalid lifecycle or epoch");
        }
        for (const auto &[key, payload] :
             impl_->pending->staged_arrivals)
        {
            if (impl_->active.contains(key) || !impl_->accepts(key, payload, batch.candidate_epoch, true))
                return reject(
                    error,
                    "device physical slot ledger destination changed before publication");
        }
        for (auto &[key, payload] : impl_->pending->staged_arrivals)
        {
            impl_->active.emplace(
                key,
                Impl::ActiveSlot{
                    .entered_epoch = batch.candidate_epoch,
                    .bootstrap_allocation = false,
                    .payload = std::move(payload),
                });
        }
        impl_->pending->staged_arrivals.clear();
        impl_->pending->published = true;
        impl_->current_epoch = batch.candidate_epoch;
        return true;
    }

    template <class Slot>
    bool MoEOverlayDevicePhysicalSlotLedger::Impl::retire(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<Slot> *retired,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!retired)
            return reject(
                error,
                "device physical slot ledger retirement requires an output owner");

        std::lock_guard<std::mutex> lock(mutex);
        if ((std::is_same_v<Slot, MoEOverlayDeviceFragmentPhysicalSlot> && !fragments()) ||
            (std::is_same_v<Slot, MoEOverlayDeviceRetiredPhysicalSlot> && !whole()) || !exact(batch) || !pending->published ||
            current_epoch != batch.candidate_epoch)
        {
            return reject(
                error,
                "device physical slot ledger retirement has invalid lifecycle or epoch");
        }

        // Validate the complete removal set before moving any lifetime. Even a
        // wrong typed projection must leave the entire published wave intact.
        for (const auto &migration : pending->migrations)
            for (const auto &key : physicalKeys(migration, true))
            {
                if (!local(key.participant_id)) continue;
                const auto found = active.find(key);
                if (found == active.end() || !accepts(key, found->second.payload, found->second.entered_epoch))
                    return reject(error, "device physical ledger lost an original source before retirement");
                if constexpr (!std::is_same_v<decltype(Slot::payload), Payload>)
                    if (!std::holds_alternative<decltype(Slot::payload)>(found->second.payload))
                        return reject(error, "device physical ledger retirement has a wrong typed payload arm");
            }
        std::vector<Slot> result;
        for (const auto &migration : pending->migrations)
        {
            for (const auto &key : physicalKeys(migration, true))
            {
                if (!local(key.participant_id)) continue;
                const auto found = active.find(key);
                if (found == active.end())
                    return reject(
                        error,
                        "device physical slot ledger lost a source before retirement");
                result.push_back({
                    .key = key,
                    .entered_epoch = found->second.entered_epoch,
                    .bootstrap_allocation = found->second.bootstrap_allocation,
                    .payload = [&]() -> decltype(Slot::payload) {
                        if constexpr (std::is_same_v<decltype(Slot::payload), Payload>)
                            return std::move(found->second.payload);
                        else
                            return std::get<decltype(Slot::payload)>(std::move(found->second.payload));
                    }(),
                });
                active.erase(found);
            }
        }
        pending.reset();
        *retired = std::move(result);
        return true;
    }

    bool MoEOverlayDevicePhysicalSlotLedger::retire(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> *retired, std::string *error) noexcept
    {
        return impl_->retire(batch, retired, error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::retireFragments(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceFragmentPhysicalSlot> *retired, std::string *error) noexcept
    {
        return impl_->retire(batch, retired, error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::retirePrepared(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDevicePreparedPhysicalSlot> *retired, std::string *error) noexcept
    {
        return impl_->retire(batch, retired, error);
    }

    bool MoEOverlayDevicePhysicalSlotLedger::abort(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->exact(batch) || impl_->pending->published)
        {
            return reject(
                error,
                "device physical slot ledger cannot abort this wave after publication or identity loss");
        }
        impl_->pending.reset();
        return true;
    }

    std::optional<MoEOverlayDevicePhysicalInventorySnapshot>
    MoEOverlayDevicePhysicalSlotLedger::snapshot(
        std::uint64_t epoch,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->whole() || epoch == 0u || epoch != impl_->current_epoch)
        {
            reject(
                error,
                "device physical inventory snapshot requires the exact current durable epoch");
            return std::nullopt;
        }
        if (impl_->pending)
        {
            reject(
                error,
                "device physical inventory snapshot is forbidden while a wave remains active");
            return std::nullopt;
        }

        MoEOverlayDevicePhysicalInventorySnapshot result;
        result.epoch = epoch;
        result.slots.reserve(impl_->active.size());
        for (const auto &[key, slot] : impl_->active)
        {
            result.slots.push_back({
                .key = key,
                .entered_epoch = slot.entered_epoch,
                .bootstrap_allocation = slot.bootstrap_allocation,
                .payload = std::get<MoEOverlayPreparedExpertPayload>(slot.payload),
            });
        }
        if (!result.valid())
        {
            reject(
                error,
                "device physical inventory snapshot found an incomplete active slot");
            return std::nullopt;
        }
        return result;
    }

    std::optional<MoEOverlayDeviceFragmentInventorySnapshot>
    MoEOverlayDevicePhysicalSlotLedger::fragmentSnapshot(std::uint64_t epoch, std::string *error) const noexcept
    {
        if (error) error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->fragments() || !epoch || epoch != impl_->current_epoch || impl_->pending)
        {
            reject(error, "device fragment inventory requires its exact quiescent epoch and payload arm");
            return std::nullopt;
        }
        MoEOverlayDeviceFragmentInventorySnapshot result{.epoch = epoch};
        result.slots.reserve(impl_->active.size());
        for (const auto &[key, slot] : impl_->active)
        {
            if (!impl_->accepts(key, slot.payload, slot.entered_epoch))
            {
                reject(error, "device fragment inventory lost its original source family");
                return std::nullopt;
            }
            result.slots.push_back({.key = key, .entered_epoch = slot.entered_epoch,
                .bootstrap_allocation = slot.bootstrap_allocation,
                .payload = std::get<MoEOverlayPreparedFragmentPayload>(slot.payload)});
        }
        return result;
    }

    std::optional<MoEOverlayDevicePreparedPhysicalInventory>
    MoEOverlayDevicePhysicalSlotLedger::preparedSnapshot(std::uint64_t epoch, std::string *error) const noexcept
    {
        if (error) error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!epoch || epoch != impl_->current_epoch || impl_->pending)
        {
            reject(error, "device mixed inventory requires its exact quiescent durable epoch");
            return std::nullopt;
        }
        MoEOverlayDevicePreparedPhysicalInventory result{.epoch = epoch};
        result.slots.reserve(impl_->active.size());
        for (const auto &[key, slot] : impl_->active)
        {
            if (!impl_->accepts(key, slot.payload, slot.entered_epoch))
            {
                reject(error, "device mixed inventory lost an original physical source family");
                return std::nullopt;
            }
            result.slots.push_back({.key = key, .entered_epoch = slot.entered_epoch,
                .bootstrap_allocation = slot.bootstrap_allocation, .payload = slot.payload});
        }
        return result;
    }

    std::shared_ptr<const MoEOverlayFragmentFamily>
    MoEOverlayDevicePhysicalSlotLedger::fragmentFamily(int participant, int layer) const noexcept
    {
        // Immutable after construction, independent of live selected-owner state.
        const auto found = impl_->families.find({participant, layer});
        return found == impl_->families.end() ? nullptr : found->second;
    }

    std::uint64_t MoEOverlayDevicePhysicalSlotLedger::currentEpoch()
        const noexcept
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->current_epoch;
    }

    std::size_t MoEOverlayDevicePhysicalSlotLedger::activeSlotCount()
        const noexcept
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->active.size();
    }

    bool MoEOverlayDevicePhysicalSlotLedger::hasPendingWave() const noexcept
    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        return impl_->pending.has_value();
    }
} // namespace llaminar2
