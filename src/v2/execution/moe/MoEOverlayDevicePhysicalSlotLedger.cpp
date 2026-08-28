/**
 * @file MoEOverlayDevicePhysicalSlotLedger.cpp
 * @brief Transactional physical lifetime bookkeeping for device-owned MoE RCU.
 */

#include "MoEOverlayDevicePhysicalSlotLedger.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
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

        /** Convert one resolved migration endpoint into a ledger coordinate. */
        MoEOverlayDevicePhysicalSlotKey sourceKey(
            const MoEOverlayTierMigration &migration) noexcept
        {
            return {
                .participant_id = migration.source.owner_participant,
                .layer_idx = migration.layer_idx,
                .expert_id = migration.expert_id,
            };
        }

        /** Convert one resolved migration destination into a ledger coordinate. */
        MoEOverlayDevicePhysicalSlotKey destinationKey(
            const MoEOverlayTierMigration &migration) noexcept
        {
            return {
                .participant_id = migration.destination.owner_participant,
                .layer_idx = migration.layer_idx,
                .expert_id = migration.expert_id,
            };
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

    bool MoEOverlayDeviceInitialPhysicalSlot::valid() const noexcept
    {
        return key.valid() && entered_epoch != 0u && triplet.complete();
    }

    bool MoEOverlayDeviceStagedPhysicalArrival::valid() const noexcept
    {
        return key.valid() && triplet.complete();
    }

    /** @brief Private active inventory and sole pending transaction. */
    struct MoEOverlayDevicePhysicalSlotLedger::Impl
    {
        /** Physical allocation retained independently of device descriptor bytes. */
        struct ActiveSlot
        {
            std::uint64_t entered_epoch = 0u;
            bool bootstrap_allocation = false;
            MoEOverlayPreparedExpertTriplet triplet;
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
                MoEOverlayPreparedExpertTriplet>
                staged_arrivals;
            bool staged = false;
            bool published = false;
        };

        /** @return Whether @p batch is exactly the pending authenticated wave. */
        bool exact(
            const MoEOverlayDevicePhysicalMovementBatch &batch) const noexcept
        {
            return pending.has_value() &&
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

        mutable std::mutex mutex;
        std::uint64_t current_epoch = 0u;
        std::vector<int> local_participant_ids;
        std::map<MoEOverlayDevicePhysicalSlotKey, ActiveSlot> active;
        std::optional<PendingWave> pending;
    };

    MoEOverlayDevicePhysicalSlotLedger::MoEOverlayDevicePhysicalSlotLedger(
        Config config)
        : impl_(std::make_unique<Impl>())
    {
        if (config.initial_epoch == 0u ||
            config.local_participant_ids.empty() ||
            !std::is_sorted(
                config.local_participant_ids.begin(),
                config.local_participant_ids.end()) ||
            std::adjacent_find(
                config.local_participant_ids.begin(),
                config.local_participant_ids.end()) !=
                config.local_participant_ids.end())
        {
            throw std::invalid_argument(
                "device physical slot ledger requires a positive epoch and sorted unique local participants");
        }
        impl_->current_epoch = config.initial_epoch;
        impl_->local_participant_ids =
            std::move(config.local_participant_ids);
        for (auto &slot : config.initial_slots)
        {
            if (!slot.valid() || slot.entered_epoch > config.initial_epoch ||
                !impl_->local(slot.key.participant_id))
            {
                throw std::invalid_argument(
                    "device physical slot ledger initial inventory is invalid or non-local");
            }
            const auto [unused, inserted] = impl_->active.emplace(
                slot.key,
                Impl::ActiveSlot{
                    .entered_epoch = slot.entered_epoch,
                    .bootstrap_allocation = slot.bootstrap_allocation,
                    .triplet = std::move(slot.triplet),
                });
            if (!inserted)
                throw std::invalid_argument(
                    "device physical slot ledger initial inventory contains a duplicate coordinate");
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
        if (!batch.valid() || !batch.movesWeights() ||
            batch.kind !=
                MoEOverlayDeviceControllerTransactionKind::DynamicPlacement)
        {
            return reject(
                error,
                "device physical slot ledger requires non-empty durable Dynamic movement");
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
            const auto source = sourceKey(migration);
            const auto destination = destinationKey(migration);
            if (impl_->local(source.participant_id))
            {
                const auto found = impl_->active.find(source);
                if (found == impl_->active.end() ||
                    !found->second.triplet.complete())
                {
                    return reject(
                        error,
                        "device physical slot ledger cannot resolve a local source allocation");
                }
            }
            if (impl_->local(destination.participant_id) &&
                (impl_->active.contains(destination) ||
                 !local_destinations.insert(destination).second))
            {
                return reject(
                    error,
                    "device physical slot ledger destination is already active or duplicated");
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

    std::optional<MoEOverlayPreparedExpertTriplet>
    MoEOverlayDevicePhysicalSlotLedger::sourceTriplet(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const MoEOverlayDevicePhysicalSlotKey &key,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!key.valid() || !impl_->exact(batch) ||
            !impl_->local(key.participant_id))
        {
            reject(
                error,
                "device physical slot ledger source request has invalid wave or endpoint identity");
            return std::nullopt;
        }
        const auto found = impl_->active.find(key);
        if (found == impl_->active.end() || !found->second.triplet.complete())
        {
            reject(
                error,
                "device physical slot ledger source is not active");
            return std::nullopt;
        }
        return found->second.triplet;
    }

    bool MoEOverlayDevicePhysicalSlotLedger::stage(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceStagedPhysicalArrival> arrivals,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->exact(batch) || impl_->pending->staged ||
            impl_->pending->published)
        {
            return reject(
                error,
                "device physical slot ledger stage has invalid lifecycle identity");
        }

        std::set<MoEOverlayDevicePhysicalSlotKey> expected;
        for (const auto &migration : impl_->pending->migrations)
        {
            const auto key = destinationKey(migration);
            if (impl_->local(key.participant_id))
                expected.insert(key);
        }

        std::map<
            MoEOverlayDevicePhysicalSlotKey,
            MoEOverlayPreparedExpertTriplet>
            staged;
        for (auto &arrival : arrivals)
        {
            if (!arrival.valid() || !expected.contains(arrival.key) ||
                !staged.emplace(
                           arrival.key, std::move(arrival.triplet)).second)
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

        impl_->pending->staged_arrivals = std::move(staged);
        impl_->pending->staged = true;
        return true;
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
        for (const auto &[key, triplet] :
             impl_->pending->staged_arrivals)
        {
            if (impl_->active.contains(key) || !triplet.complete())
                return reject(
                    error,
                    "device physical slot ledger destination changed before publication");
        }
        for (auto &[key, triplet] : impl_->pending->staged_arrivals)
        {
            impl_->active.emplace(
                key,
                Impl::ActiveSlot{
                    .entered_epoch = batch.candidate_epoch,
                    .bootstrap_allocation = false,
                    .triplet = std::move(triplet),
                });
        }
        impl_->pending->staged_arrivals.clear();
        impl_->pending->published = true;
        impl_->current_epoch = batch.candidate_epoch;
        return true;
    }

    bool MoEOverlayDevicePhysicalSlotLedger::retire(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> *retired,
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (!retired)
            return reject(
                error,
                "device physical slot ledger retirement requires an output owner");

        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (!impl_->exact(batch) || !impl_->pending->published ||
            impl_->current_epoch != batch.candidate_epoch)
        {
            return reject(
                error,
                "device physical slot ledger retirement has invalid lifecycle or epoch");
        }

        std::vector<MoEOverlayDeviceRetiredPhysicalSlot> result;
        for (const auto &migration : impl_->pending->migrations)
        {
            const auto key = sourceKey(migration);
            if (!impl_->local(key.participant_id))
                continue;
            const auto found = impl_->active.find(key);
            if (found == impl_->active.end())
                return reject(
                    error,
                    "device physical slot ledger lost a source before retirement");
            result.push_back({
                .key = key,
                .entered_epoch = found->second.entered_epoch,
                .bootstrap_allocation =
                    found->second.bootstrap_allocation,
                .triplet = std::move(found->second.triplet),
            });
            impl_->active.erase(found);
        }
        impl_->pending.reset();
        *retired = std::move(result);
        return true;
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
