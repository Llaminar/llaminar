/**
 * @file MoEOverlayParticipantResidency.cpp
 * @brief Implementation of participant-local epoch-indexed expert banks.
 */

#include "MoEOverlayParticipantResidency.h"

#include "../../loaders/ExpertGemmRegistry.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    /**
     * @brief One immutable bank plus lock-free inference-reader accounting.
     *
     * The node is allocated and validated before publication.  Its embedded
     * retired link lets maintenance defer reclamation without allocating at
     * retirement time.  Only the maintenance writer mutates `retired_next`;
     * inference touches only `reader_count` and the immutable bank.
     */
    struct MoEOverlayParticipantPublishedBank
    {
        /** @brief Adopt one validated bank for exactly one endpoint state. */
        MoEOverlayParticipantPublishedBank(
            MoEOverlayParticipantResidencyBank value,
            const MoEOverlayParticipantResidencyState *authority_value)
            : bank(std::move(value)), authority(authority_value)
        {
        }

        const MoEOverlayParticipantResidencyBank bank;
        const MoEOverlayParticipantResidencyState *authority = nullptr;
        std::atomic<uint64_t> reader_count{0};
        std::unique_ptr<MoEOverlayParticipantPublishedBank> retired_next;
    };

    /**
     * @brief Fixed publication slots and asynchronous reclamation for one endpoint.
     *
     * Publication pointers and acquire hazards are native lock-free atomics on
     * every supported 64-bit execution host.  The mutex is maintenance-only:
     * installation, abort, and retirement use it to maintain sole-writer slot
     * ownership, while inference never observes or waits on it.
     */
    class MoEOverlayParticipantResidencyState final
    {
    public:
        /** @brief Allocate the exact retained-epoch slot count at model setup. */
        explicit MoEOverlayParticipantResidencyState(std::size_t capacity)
            : capacity_(capacity),
              slots_(std::make_unique<Slot[]>(capacity))
        {
        }

    private:
        friend class MoEOverlayParticipantBankLease;
        friend class MoEOverlayParticipantResidency;

        /** @brief One raw inference publication and its maintenance owner. */
        struct Slot
        {
            std::atomic<MoEOverlayParticipantPublishedBank *> published{
                nullptr};
            std::unique_ptr<MoEOverlayParticipantPublishedBank> owner;
        };

        /**
         * @brief Reclaim retired nodes that no acquisition can still adopt.
         *
         * The caller owns `writer_mutex_`. A non-zero acquire hazard means a
         * reader may have loaded a just-cleared raw pointer but not incremented
         * that node yet, so reclamation is deferred wholesale and never waits.
         */
        void reclaimRetiredBanks() noexcept
        {
            if (acquires_in_flight_.load(std::memory_order_seq_cst) != 0)
                return;

            auto *link = &retired_head_;
            while (*link)
            {
                if ((*link)->reader_count.load(std::memory_order_acquire) != 0)
                {
                    link = &((*link)->retired_next);
                    continue;
                }

                auto reclaimed = std::move(*link);
                *link = std::move(reclaimed->retired_next);
                /* Destruction and prepared-engine release stay on maintenance. */
            }
        }

        const std::size_t capacity_;
        std::unique_ptr<Slot[]> slots_;
        std::mutex writer_mutex_;
        std::unique_ptr<MoEOverlayParticipantPublishedBank> retired_head_;
        std::atomic<uint64_t> acquires_in_flight_{0};
    };

    static_assert(
        std::atomic<MoEOverlayParticipantPublishedBank *>::is_always_lock_free,
        "ExpertOverlay inference publication requires lock-free pointer atomics");
    static_assert(
        std::atomic<uint64_t>::is_always_lock_free,
        "ExpertOverlay inference leases require lock-free 64-bit atomics");

    namespace
    {
        /** @return Dense production-phase index, or the sentinel count. */
        std::size_t serviceSourceIndex(
            ExpertHistogramSource source) noexcept
        {
            switch (source)
            {
            case ExpertHistogramSource::DecodeToken:
                return 0;
            case ExpertHistogramSource::PrefillChunk:
                return 1;
            case ExpertHistogramSource::GroupedVerifier:
                return 2;
            case ExpertHistogramSource::SyntheticTest:
                break;
            }
            return kExpertHistogramProductionSourceCount;
        }

        /** @brief Saturating checked addition inside an already-owned cell. */
        bool addWithoutOverflow(
            uint64_t &target,
            uint64_t value) noexcept
        {
            if (value > std::numeric_limits<uint64_t>::max() - target)
            {
                target = std::numeric_limits<uint64_t>::max();
                return false;
            }
            target += value;
            return true;
        }
    } // namespace

    bool MoEOverlayParticipantLayerServiceTotals::valid() const noexcept
    {
        if (participant_id < 0 || layer < 0)
            return false;
        for (std::size_t phase = 0;
             phase < kExpertHistogramProductionSourceCount;
             ++phase)
        {
            if (overflowed[phase])
                return false;
            const bool empty = sample_count[phase] == 0;
            if (empty != (total_nanoseconds[phase] == 0) ||
                empty != (activation_count[phase] == 0))
            {
                return false;
            }
        }
        return true;
    }

    bool MoEOverlayPreparedExpertTriplet::complete() const noexcept
    {
        return gate != nullptr && up != nullptr && down != nullptr;
    }

    bool MoEOverlayPreparedExpertTriplet::empty() const noexcept
    {
        return gate == nullptr && up == nullptr && down == nullptr;
    }

    bool MoEOverlayPreparedExpertTriplet::sameIdentity(
        const MoEOverlayPreparedExpertTriplet &other) const noexcept
    {
        return gate.get() == other.gate.get() &&
               up.get() == other.up.get() &&
               down.get() == other.down.get();
    }

    bool resolveMoEOverlayPreparedExpertTriplets(
        const ExpertGemmRegistry &registry,
        const MoEExpertOwnerParticipant &participant,
        int layer_idx,
        int num_experts,
        const std::vector<bool> &resident_mask,
        std::vector<MoEOverlayPreparedExpertTriplet> &output,
        std::string *error)
    {
        if (error)
            error->clear();
        output.clear();
        if (layer_idx < 0 || num_experts <= 0 ||
            !participant.device.is_valid() ||
            participant.domain_name.empty() ||
            participant.domain_participant_index < 0 ||
            resident_mask.size() != static_cast<std::size_t>(num_experts))
        {
            if (error)
            {
                *error =
                    "ExpertOverlay prepared-triplet resolution has invalid "
                    "geometry or participant identity";
            }
            return false;
        }

        output.resize(static_cast<std::size_t>(num_experts));
        const int world_rank = participant.world_rank_known
                                   ? participant.world_rank
                                   : -1;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            if (!resident_mask[static_cast<std::size_t>(expert)])
                continue;

            auto &triplet = output[static_cast<std::size_t>(expert)];
            triplet.gate = registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::GATE);
            triplet.up = registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::UP);
            triplet.down = registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::DOWN);
            if (!triplet.complete())
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay participant p" +
                        std::to_string(participant.participant_id) +
                        " has no complete owned gate/up/down lifetime for layer " +
                        std::to_string(layer_idx) + " expert " +
                        std::to_string(expert);
                }
                output.clear();
                return false;
            }
        }
        return true;
    }

    void MoEOverlayParticipantLayerBank::setResidentExpert(
        int expert_id,
        MoEOverlayPreparedExpertTriplet engines)
    {
        if (expert_id < 0 ||
            static_cast<std::size_t>(expert_id) >= experts.size() ||
            experts.size() != resident_mask.size())
        {
            throw std::out_of_range(
                "ExpertOverlay participant bank expert id is outside geometry");
        }
        if (!engines.complete())
        {
            throw std::invalid_argument(
                "ExpertOverlay resident expert requires gate, up, and down engines");
        }

        const auto index = static_cast<std::size_t>(expert_id);
        /* Install lifetimes before publishing the resident bit in this mutable candidate. */
        experts[index] = std::move(engines);
        resident_mask[index] = true;
    }

    void MoEOverlayParticipantLayerBank::clearExpert(int expert_id)
    {
        if (expert_id < 0 ||
            static_cast<std::size_t>(expert_id) >= experts.size() ||
            experts.size() != resident_mask.size())
        {
            throw std::out_of_range(
                "ExpertOverlay participant bank expert id is outside geometry");
        }

        const auto index = static_cast<std::size_t>(expert_id);
        /*
         * Only this unpublished candidate loses the lifetimes. The installed
         * old bank owns its independent shared pointers until lease retirement.
         */
        resident_mask[index] = false;
        experts[index] = {};
    }

    bool MoEOverlayParticipantLayerBank::valid(int num_experts) const noexcept
    {
        if (num_experts <= 0 ||
            resident_mask.size() != static_cast<std::size_t>(num_experts) ||
            experts.size() != static_cast<std::size_t>(num_experts))
        {
            return false;
        }

        for (std::size_t expert = 0; expert < experts.size(); ++expert)
        {
            if (resident_mask[expert] != experts[expert].complete())
                return false;
            if (!resident_mask[expert] && !experts[expert].empty())
                return false;
        }
        return true;
    }

    bool MoEOverlayParticipantLayerBank::sameIdentity(
        const MoEOverlayParticipantLayerBank &other) const noexcept
    {
        if (resident_mask != other.resident_mask ||
            experts.size() != other.experts.size())
        {
            return false;
        }
        for (std::size_t expert = 0; expert < experts.size(); ++expert)
        {
            if (!experts[expert].sameIdentity(other.experts[expert]))
                return false;
        }
        return true;
    }

    bool MoEOverlayParticipantResidencyBank::valid(
        int expected_participant_id,
        DeviceId expected_device,
        int num_layers,
        int num_experts) const noexcept
    {
        if (epoch == 0 || participant_id != expected_participant_id ||
            device != expected_device || num_layers <= 0 || num_experts <= 0 ||
            layers.size() != static_cast<std::size_t>(num_layers))
        {
            return false;
        }
        return std::all_of(
            layers.begin(),
            layers.end(),
            [num_experts](const auto &layer)
            { return layer.valid(num_experts); });
    }

    bool MoEOverlayParticipantResidencyBank::sameIdentity(
        const MoEOverlayParticipantResidencyBank &other) const noexcept
    {
        if (epoch != other.epoch ||
            participant_id != other.participant_id ||
            device != other.device || layers.size() != other.layers.size())
        {
            return false;
        }
        for (std::size_t layer = 0; layer < layers.size(); ++layer)
        {
            if (!layers[layer].sameIdentity(other.layers[layer]))
                return false;
        }
        return true;
    }

    MoEOverlayPreparedParticipantBank::MoEOverlayPreparedParticipantBank()
        noexcept = default;

    MoEOverlayPreparedParticipantBank::~MoEOverlayPreparedParticipantBank() =
        default;

    MoEOverlayPreparedParticipantBank::MoEOverlayPreparedParticipantBank(
        MoEOverlayPreparedParticipantBank &&other) noexcept = default;

    MoEOverlayPreparedParticipantBank &
    MoEOverlayPreparedParticipantBank::operator=(
        MoEOverlayPreparedParticipantBank &&other) noexcept = default;

    MoEOverlayPreparedParticipantBank::MoEOverlayPreparedParticipantBank(
        std::unique_ptr<MoEOverlayParticipantPublishedBank> node) noexcept
        : node_(std::move(node))
    {
    }

    bool MoEOverlayPreparedParticipantBank::valid() const noexcept
    {
        return node_ != nullptr;
    }

    uint64_t MoEOverlayPreparedParticipantBank::epoch() const noexcept
    {
        return node_ ? node_->bank.epoch : 0;
    }

    MoEOverlayParticipantBankLease::MoEOverlayParticipantBankLease() noexcept =
        default;

    MoEOverlayParticipantBankLease::~MoEOverlayParticipantBankLease()
    {
        reset();
    }

    MoEOverlayParticipantBankLease::MoEOverlayParticipantBankLease(
        std::shared_ptr<MoEOverlayParticipantResidencyState> state,
        MoEOverlayParticipantPublishedBank *node) noexcept
        : state_(std::move(state)), node_(node)
    {
    }

    MoEOverlayParticipantBankLease::MoEOverlayParticipantBankLease(
        const MoEOverlayParticipantBankLease &other) noexcept
        : state_(other.state_), node_(other.node_)
    {
        retain();
    }

    MoEOverlayParticipantBankLease &
    MoEOverlayParticipantBankLease::operator=(
        const MoEOverlayParticipantBankLease &other) noexcept
    {
        if (this == &other)
            return *this;
        reset();
        state_ = other.state_;
        node_ = other.node_;
        retain();
        return *this;
    }

    MoEOverlayParticipantBankLease::MoEOverlayParticipantBankLease(
        MoEOverlayParticipantBankLease &&other) noexcept
        : state_(std::move(other.state_)),
          node_(std::exchange(other.node_, nullptr))
    {
    }

    MoEOverlayParticipantBankLease &
    MoEOverlayParticipantBankLease::operator=(
        MoEOverlayParticipantBankLease &&other) noexcept
    {
        if (this == &other)
            return *this;
        reset();
        state_ = std::move(other.state_);
        node_ = std::exchange(other.node_, nullptr);
        return *this;
    }

    const MoEOverlayParticipantResidencyBank *
    MoEOverlayParticipantBankLease::get() const noexcept
    {
        return node_ ? &node_->bank : nullptr;
    }

    const MoEOverlayParticipantResidencyBank &
    MoEOverlayParticipantBankLease::operator*() const noexcept
    {
        return node_->bank;
    }

    const MoEOverlayParticipantResidencyBank *
    MoEOverlayParticipantBankLease::operator->() const noexcept
    {
        return get();
    }

    MoEOverlayParticipantBankLease::operator bool() const noexcept
    {
        return node_ != nullptr;
    }

    void MoEOverlayParticipantBankLease::retain() noexcept
    {
        if (node_)
        {
            node_->reader_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void MoEOverlayParticipantBankLease::reset() noexcept
    {
        if (node_)
        {
            node_->reader_count.fetch_sub(1, std::memory_order_release);
            node_ = nullptr;
        }
        state_.reset();
    }

    MoEOverlayParticipantResidency::MoEOverlayParticipantResidency(Config config)
        : config_(std::move(config))
    {
        if (config_.participant_id < 0 || !config_.device.is_valid() ||
            config_.num_layers <= 0 || config_.num_experts <= 0 ||
            config_.retained_epoch_capacity < 2)
        {
            throw std::invalid_argument(
                "ExpertOverlay participant residency requires valid identity, "
                "geometry, and at least two retained epoch slots");
        }
        state_ = std::make_shared<MoEOverlayParticipantResidencyState>(
            config_.retained_epoch_capacity);
        if (config_.collect_economy_service_measurements)
        {
            const std::size_t cell_count =
                static_cast<std::size_t>(config_.num_layers) *
                kExpertHistogramProductionSourceCount;
            service_measurements_ =
                std::make_unique<ServiceMeasurementCell[]>(cell_count);
        }
    }

    std::size_t MoEOverlayParticipantResidency::serviceMeasurementOffset(
        int layer,
        ExpertHistogramSource source) const noexcept
    {
        const std::size_t phase = serviceSourceIndex(source);
        if (layer < 0 || layer >= config_.num_layers ||
            phase >= kExpertHistogramProductionSourceCount)
        {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(layer) *
                   kExpertHistogramProductionSourceCount +
               phase;
    }

    MoEOverlayServiceMeasurementRecordStatus
    MoEOverlayParticipantResidency::recordServiceMeasurement(
        int layer,
        ExpertHistogramSource source,
        uint64_t elapsed_nanoseconds,
        uint64_t activations) noexcept
    {
        if (!config_.collect_economy_service_measurements)
            return MoEOverlayServiceMeasurementRecordStatus::Disabled;
        if (device_service_measurements_imported_.load(
                std::memory_order_acquire))
        {
            return MoEOverlayServiceMeasurementRecordStatus::Invalid;
        }

        const std::size_t offset =
            serviceMeasurementOffset(layer, source);
        if (!service_measurements_ ||
            offset == std::numeric_limits<std::size_t>::max() ||
            elapsed_nanoseconds == 0 || activations == 0)
        {
            return MoEOverlayServiceMeasurementRecordStatus::Invalid;
        }

        ServiceMeasurementCell &cell = service_measurements_[offset];
        if (cell.owned.test_and_set(std::memory_order_acquire))
        {
            dropped_service_measurements_.fetch_add(
                1, std::memory_order_relaxed);
            return MoEOverlayServiceMeasurementRecordStatus::Contended;
        }

        /*
         * These values form one logical sample. Keep the cell owned until all
         * three are updated, then publish the complete tuple with the release
         * clear. A snapshot can never divide a new duration by old activations.
         */
        const bool duration_ok = addWithoutOverflow(
            cell.total_nanoseconds, elapsed_nanoseconds);
        const bool activations_ok = addWithoutOverflow(
            cell.activation_count, activations);
        const bool samples_ok = addWithoutOverflow(cell.sample_count, 1);
        cell.overflowed = cell.overflowed || !duration_ok ||
                          !activations_ok || !samples_ok;
        const bool overflowed = cell.overflowed;
        cell.owned.clear(std::memory_order_release);
        return overflowed
                   ? MoEOverlayServiceMeasurementRecordStatus::Overflow
                   : MoEOverlayServiceMeasurementRecordStatus::Recorded;
    }

    bool MoEOverlayParticipantResidency::importServiceMeasurements(
        const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!config_.collect_economy_service_measurements ||
            !service_measurements_ ||
            rows.size() != static_cast<std::size_t>(config_.num_layers))
        {
            if (error)
                *error = "service snapshot does not match an enabled endpoint";
            return false;
        }
        const bool replacing_device_snapshot =
            device_service_measurements_imported_.load(
                std::memory_order_acquire);
        for (int layer = 0; layer < config_.num_layers; ++layer)
        {
            const auto &row = rows[static_cast<std::size_t>(layer)];
            if (row.participant_id != config_.participant_id ||
                row.layer != layer || !row.valid())
            {
                if (error)
                    *error = "service snapshot contains invalid participant/layer totals";
                return false;
            }
        }

        const std::size_t cell_count =
            static_cast<std::size_t>(config_.num_layers) *
            kExpertHistogramProductionSourceCount;
        std::vector<ServiceMeasurementCell *> acquired;
        acquired.reserve(cell_count);
        const auto release_all = [&acquired]() noexcept
        {
            for (auto *cell : acquired)
                cell->owned.clear(std::memory_order_release);
            acquired.clear();
        };
        for (std::size_t offset = 0u; offset < cell_count; ++offset)
        {
            auto &cell = service_measurements_[offset];
            if (cell.owned.test_and_set(std::memory_order_acquire))
            {
                release_all();
                if (error)
                    *error = "service snapshot endpoint is busy with an inference writer";
                return false;
            }
            acquired.push_back(&cell);
        }

        for (int layer = 0; layer < config_.num_layers; ++layer)
        {
            const auto &row = rows[static_cast<std::size_t>(layer)];
            for (std::size_t phase = 0u;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                auto &cell = service_measurements_[
                    static_cast<std::size_t>(layer) *
                        kExpertHistogramProductionSourceCount +
                    phase];
                const bool empty = cell.total_nanoseconds == 0u &&
                                   cell.activation_count == 0u &&
                                   cell.sample_count == 0u &&
                                   !cell.overflowed;
                const bool identical =
                    cell.total_nanoseconds == row.total_nanoseconds[phase] &&
                    cell.activation_count == row.activation_count[phase] &&
                    cell.sample_count == row.sample_count[phase] &&
                    cell.overflowed == row.overflowed[phase];
                const bool monotonic_device_update =
                    replacing_device_snapshot &&
                    row.total_nanoseconds[phase] >=
                        cell.total_nanoseconds &&
                    row.activation_count[phase] >=
                        cell.activation_count &&
                    row.sample_count[phase] >= cell.sample_count &&
                    (!cell.overflowed || row.overflowed[phase]);
                if ((!replacing_device_snapshot && !empty && !identical) ||
                    (replacing_device_snapshot &&
                     !monotonic_device_update))
                {
                    release_all();
                    if (error)
                        *error = "service snapshot conflicts with existing endpoint evidence";
                    return false;
                }
            }
        }

        for (int layer = 0; layer < config_.num_layers; ++layer)
        {
            const auto &row = rows[static_cast<std::size_t>(layer)];
            for (std::size_t phase = 0u;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                auto &cell = service_measurements_[
                    static_cast<std::size_t>(layer) *
                        kExpertHistogramProductionSourceCount +
                    phase];
                cell.total_nanoseconds = row.total_nanoseconds[phase];
                cell.activation_count = row.activation_count[phase];
                cell.sample_count = row.sample_count[phase];
                cell.overflowed = row.overflowed[phase];
            }
        }
        device_service_measurements_imported_.store(
            true, std::memory_order_release);
        release_all();
        return true;
    }

    bool MoEOverlayParticipantResidency::trySnapshotServiceMeasurements(
        std::vector<MoEOverlayParticipantLayerServiceTotals> *output) const
    {
        if (!output)
            return false;
        output->clear();
        if (!config_.collect_economy_service_measurements ||
            !service_measurements_)
        {
            return false;
        }

        output->resize(static_cast<std::size_t>(config_.num_layers));
        for (int layer = 0; layer < config_.num_layers; ++layer)
        {
            auto &row = (*output)[static_cast<std::size_t>(layer)];
            row.participant_id = config_.participant_id;
            row.layer = layer;
            for (std::size_t phase = 0;
                 phase < kExpertHistogramProductionSourceCount;
                 ++phase)
            {
                ServiceMeasurementCell &cell = service_measurements_[
                    static_cast<std::size_t>(layer) *
                        kExpertHistogramProductionSourceCount +
                    phase];
                if (cell.owned.test_and_set(std::memory_order_acquire))
                {
                    output->clear();
                    return false;
                }
                row.total_nanoseconds[phase] = cell.total_nanoseconds;
                row.activation_count[phase] = cell.activation_count;
                row.sample_count[phase] = cell.sample_count;
                row.overflowed[phase] = cell.overflowed;
                cell.owned.clear(std::memory_order_release);
            }
        }
        return true;
    }

    MoEOverlayParticipantResidencyBank
    MoEOverlayParticipantResidency::cloneCandidate(
        uint64_t previous_epoch,
        uint64_t candidate_epoch) const
    {
        if (candidate_epoch == 0 || candidate_epoch <= previous_epoch)
        {
            throw std::invalid_argument(
                "ExpertOverlay candidate epoch must be newer than its source");
        }

        const auto previous = acquire(previous_epoch);
        if (!previous)
        {
            throw std::out_of_range(
                "ExpertOverlay candidate source epoch is not retained");
        }

        MoEOverlayParticipantResidencyBank candidate = *previous;
        candidate.epoch = candidate_epoch;
        return candidate;
    }

    std::optional<MoEOverlayPreparedParticipantBank>
    MoEOverlayParticipantResidency::prepareReadyBank(
        MoEOverlayParticipantResidencyBank bank,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        if (!bank.valid(
                config_.participant_id,
                config_.device,
                config_.num_layers,
                config_.num_experts))
        {
            if (error)
            {
                *error =
                    "ExpertOverlay participant ready bank is incomplete or has wrong geometry";
            }
            return std::nullopt;
        }

        try
        {
            return MoEOverlayPreparedParticipantBank(
                std::make_unique<MoEOverlayParticipantPublishedBank>(
                    std::move(bank), state_.get()));
        }
        catch (const std::exception &exception)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay participant ready-bank preparation failed: " +
                    std::string(exception.what());
            }
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay participant ready-bank preparation failed with a non-standard exception";
            }
        }
        return std::nullopt;
    }

    MoEOverlayParticipantBankInstallStatus
    MoEOverlayParticipantResidency::installReadyBank(
        MoEOverlayPreparedParticipantBank &&prepared,
        std::string *error)
    {
        if (error)
            error->clear();
        if (!prepared.node_ ||
            prepared.node_->authority != state_.get())
        {
            if (error)
            {
                *error =
                    "ExpertOverlay prepared participant bank is empty or belongs to another endpoint";
            }
            return MoEOverlayParticipantBankInstallStatus::Invalid;
        }

        const auto &bank = prepared.node_->bank;
        std::lock_guard<std::mutex> lock(state_->writer_mutex_);
        MoEOverlayParticipantResidencyState::Slot *empty_slot = nullptr;
        for (std::size_t index = 0; index < state_->capacity_; ++index)
        {
            auto &slot = state_->slots_[index];
            if (!slot.owner)
            {
                if (!empty_slot)
                    empty_slot = &slot;
                continue;
            }
            if (slot.owner->bank.epoch != bank.epoch)
                continue;
            if (slot.owner->bank.sameIdentity(bank))
                return MoEOverlayParticipantBankInstallStatus::AlreadyInstalled;
            if (error)
            {
                *error =
                    "ExpertOverlay participant epoch already names a different bank";
            }
            return MoEOverlayParticipantBankInstallStatus::EpochConflict;
        }
        if (!empty_slot)
        {
            if (error)
            {
                *error =
                    "ExpertOverlay participant retained-bank capacity is exhausted";
            }
            return MoEOverlayParticipantBankInstallStatus::CapacityUnavailable;
        }

        /*
         * Every expensive operation happened in prepareReadyBank(). The raw
         * pointer becomes visible only after its sole owner occupies the fixed
         * slot, and the sequentially-consistent store participates in the
         * acquire-versus-retire hazard proof used below.
         */
        empty_slot->owner = std::move(prepared.node_);
        empty_slot->published.store(
            empty_slot->owner.get(), std::memory_order_seq_cst);
        return MoEOverlayParticipantBankInstallStatus::Installed;
    }

    MoEOverlayParticipantBankLease
    MoEOverlayParticipantResidency::acquire(uint64_t epoch) const noexcept
    {
        auto state = state_;
        state->acquires_in_flight_.fetch_add(
            1, std::memory_order_seq_cst);

        for (std::size_t index = 0; index < state->capacity_; ++index)
        {
            auto *node = state->slots_[index].published.load(
                std::memory_order_seq_cst);
            if (!node || node->bank.epoch != epoch)
                continue;

            /*
             * The global hazard remains set until this exact node owns its
             * reader. Retirement may unlink it meanwhile, but cannot reclaim
             * it while either counter proves this acquisition is in progress.
             */
            node->reader_count.fetch_add(1, std::memory_order_relaxed);
            state->acquires_in_flight_.fetch_sub(
                1, std::memory_order_seq_cst);
            return MoEOverlayParticipantBankLease(
                std::move(state), node);
        }

        state->acquires_in_flight_.fetch_sub(
            1, std::memory_order_seq_cst);
        return {};
    }

    bool MoEOverlayParticipantResidency::abortUnpublished(
        uint64_t epoch) noexcept
    {
        return retire(epoch);
    }

    bool MoEOverlayParticipantResidency::retire(uint64_t epoch) noexcept
    {
        std::lock_guard<std::mutex> lock(state_->writer_mutex_);
        for (std::size_t index = 0; index < state_->capacity_; ++index)
        {
            auto &slot = state_->slots_[index];
            if (!slot.owner || slot.owner->bank.epoch != epoch)
                continue;

            /*
             * Stop new readers before detaching ownership. A reader that saw
             * the old pointer first is protected by acquires_in_flight_ until
             * it increments this node's reader_count.
             */
            slot.published.store(nullptr, std::memory_order_seq_cst);
            auto retired = std::move(slot.owner);
            retired->retired_next = std::move(state_->retired_head_);
            state_->retired_head_ = std::move(retired);
            state_->reclaimRetiredBanks();
            return true;
        }
        state_->reclaimRetiredBanks();
        return false;
    }

    std::size_t
    MoEOverlayParticipantResidency::retainedEpochCount() const noexcept
    {
        std::size_t retained = 0;
        for (std::size_t index = 0; index < state_->capacity_; ++index)
        {
            retained += state_->slots_[index].published.load(
                            std::memory_order_acquire) != nullptr
                            ? 1u
                            : 0u;
        }
        return retained;
    }

    bool MoEOverlayParticipantResidency::hasCandidateCapacity() const noexcept
    {
        for (std::size_t index = 0; index < state_->capacity_; ++index)
        {
            if (!state_->slots_[index].published.load(
                    std::memory_order_acquire))
            {
                return true;
            }
        }
        return false;
    }

    MoEOverlayParticipantResidencyRegistry::
        MoEOverlayParticipantResidencyRegistry(Config config)
        : config_(std::move(config))
    {
        if (config_.num_layers <= 0 || config_.num_experts <= 0 ||
            config_.initial_epoch == 0 ||
            config_.retained_epoch_capacity < 2)
        {
            throw std::invalid_argument(
                "ExpertOverlay participant registry requires positive geometry, "
                "an initial epoch, and two-bank capacity");
        }

        /*
         * A relay-only MPI rank deliberately owns no prepared expert bank.  It
         * still retains this registry so every rank carries the same typed
         * orchestration contract and enters the matched sparse protocol.  An
         * empty endpoint set is therefore a valid, vacuously-ready registry;
         * non-relay role validation remains the participant runner's job.
         */

        std::sort(
            config_.local_participant_ids.begin(),
            config_.local_participant_ids.end());
        if (std::adjacent_find(
                config_.local_participant_ids.begin(),
                config_.local_participant_ids.end()) !=
            config_.local_participant_ids.end())
        {
            throw std::invalid_argument(
                "ExpertOverlay participant registry ids must be unique");
        }

        for (const int participant_id : config_.local_participant_ids)
        {
            const auto *participant =
                config_.owner_map.participantForId(participant_id);
            if (!participant || !participant->device.is_valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant registry references an unknown local endpoint");
            }

            EndpointAssembly assembly;
            assembly.endpoint =
                std::make_shared<MoEOverlayParticipantResidency>(
                    MoEOverlayParticipantResidency::Config{
                        .participant_id = participant_id,
                        .device = participant->device,
                        .num_layers = config_.num_layers,
                        .num_experts = config_.num_experts,
                        .retained_epoch_capacity =
                            config_.retained_epoch_capacity,
                        .collect_economy_service_measurements =
                            config_.collect_economy_service_measurements,
                    });
            assembly.initial_bank.epoch = config_.initial_epoch;
            assembly.initial_bank.participant_id = participant_id;
            assembly.initial_bank.device = participant->device;
            assembly.initial_bank.layers.resize(
                static_cast<std::size_t>(config_.num_layers));
            assembly.registered_layers.assign(
                static_cast<std::size_t>(config_.num_layers), false);

            for (int layer = 0; layer < config_.num_layers; ++layer)
            {
                auto &layer_bank =
                    assembly.initial_bank.layers[
                        static_cast<std::size_t>(layer)];
                layer_bank.resident_mask =
                    config_.owner_map.expertMaskForParticipant(
                        layer,
                        participant_id,
                        config_.num_experts);
                layer_bank.experts.resize(
                    static_cast<std::size_t>(config_.num_experts));

                /* No engines are needed for a layer with no local residents. */
                assembly.registered_layers[static_cast<std::size_t>(layer)] =
                    std::none_of(
                        layer_bank.resident_mask.begin(),
                        layer_bank.resident_mask.end(),
                        [](bool resident) { return resident; });
            }

            /*
             * A legitimate receiving tier may start with no experts in any
             * layer. No graph stage exists that could call registerInitialLayer
             * for such an endpoint, so publish its complete empty bank now.
             * This is data initialization only: topology and capacity were
             * already frozen by the constructor configuration.
             */
            if (std::all_of(
                    assembly.registered_layers.begin(),
                    assembly.registered_layers.end(),
                    [](bool registered) { return registered; }))
            {
                std::string error;
                auto prepared = assembly.endpoint->prepareReadyBank(
                    assembly.initial_bank, &error);
                if (!prepared)
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay could not prepare an initially empty participant bank"
                            : error);
                }
                const auto status = assembly.endpoint->installReadyBank(
                    std::move(*prepared), &error);
                if (status !=
                        MoEOverlayParticipantBankInstallStatus::Installed &&
                    status !=
                        MoEOverlayParticipantBankInstallStatus::AlreadyInstalled)
                {
                    throw std::runtime_error(
                        error.empty()
                            ? "ExpertOverlay could not install an initially empty participant bank"
                            : error);
                }
            }
            endpoints_.emplace(participant_id, std::move(assembly));
        }
    }

    bool MoEOverlayParticipantResidencyRegistry::registerInitialLayer(
        int participant_id,
        int layer_idx,
        const std::vector<bool> &resident_mask,
        const std::vector<MoEOverlayPreparedExpertTriplet> &experts,
        std::string *error)
    {
        if (error)
            error->clear();
        if (layer_idx < 0 || layer_idx >= config_.num_layers ||
            resident_mask.size() !=
                static_cast<std::size_t>(config_.num_experts) ||
            experts.size() != static_cast<std::size_t>(config_.num_experts))
        {
            if (error)
                *error = "ExpertOverlay initial layer registration has invalid geometry";
            return false;
        }

        std::unique_lock<std::shared_mutex> lock(mutex_);
        const auto found = endpoints_.find(participant_id);
        if (found == endpoints_.end())
        {
            if (error)
                *error = "ExpertOverlay initial layer registration targets a non-local participant";
            return false;
        }
        auto &assembly = found->second;
        auto &expected_layer =
            assembly.initial_bank.layers[static_cast<std::size_t>(layer_idx)];
        if (resident_mask != expected_layer.resident_mask)
        {
            if (error)
                *error = "ExpertOverlay initial layer mask differs from the canonical owner map";
            return false;
        }

        MoEOverlayParticipantLayerBank supplied_layer{
            .resident_mask = resident_mask,
            .experts = experts,
        };
        if (!supplied_layer.valid(config_.num_experts))
        {
            if (error)
                *error = "ExpertOverlay initial layer has incomplete or extraneous expert engines";
            return false;
        }

        const auto existing =
            assembly.endpoint->acquire(config_.initial_epoch);
        if (existing)
        {
            if (!existing->layers[static_cast<std::size_t>(layer_idx)]
                     .sameIdentity(supplied_layer))
            {
                if (error)
                    *error = "Cached graph resolved different engines for an installed initial epoch";
                return false;
            }
            return true;
        }

        const auto layer_index = static_cast<std::size_t>(layer_idx);
        if (assembly.registered_layers[layer_index] &&
            !expected_layer.sameIdentity(supplied_layer))
        {
            if (error)
                *error = "Concurrent graph construction resolved different initial engines";
            return false;
        }
        expected_layer = std::move(supplied_layer);
        assembly.registered_layers[layer_index] = true;

        if (!std::all_of(
                assembly.registered_layers.begin(),
                assembly.registered_layers.end(),
                [](bool registered) { return registered; }))
        {
            return true;
        }

        auto prepared = assembly.endpoint->prepareReadyBank(
            assembly.initial_bank, error);
        if (!prepared)
            return false;
        const auto status = assembly.endpoint->installReadyBank(
            std::move(*prepared), error);
        return status == MoEOverlayParticipantBankInstallStatus::Installed ||
               status ==
                   MoEOverlayParticipantBankInstallStatus::AlreadyInstalled;
    }

    std::shared_ptr<MoEOverlayParticipantResidency>
    MoEOverlayParticipantResidencyRegistry::endpoint(
        int participant_id) const noexcept
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        const auto found = endpoints_.find(participant_id);
        return found == endpoints_.end() ? nullptr : found->second.endpoint;
    }

    std::vector<int>
    MoEOverlayParticipantResidencyRegistry::localParticipantIds() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<int> ids;
        ids.reserve(endpoints_.size());
        for (const auto &[participant_id, _] : endpoints_)
            ids.push_back(participant_id);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    bool MoEOverlayParticipantResidencyRegistry::allInitialBanksReady()
        const noexcept
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return std::all_of(
            endpoints_.begin(),
            endpoints_.end(),
            [this](const auto &entry)
            {
                return entry.second.endpoint->acquire(
                           config_.initial_epoch) != nullptr;
            });
    }

    std::vector<
        MoEOverlayParticipantResidencyRegistry::InitialBankExpertSelection>
    MoEOverlayParticipantResidencyRegistry::initialBankExpertSelections() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<int> participant_ids;
        participant_ids.reserve(endpoints_.size());
        for (const auto &[participant_id, _] : endpoints_)
            participant_ids.push_back(participant_id);
        std::sort(participant_ids.begin(), participant_ids.end());

        std::vector<InitialBankExpertSelection> selections;
        selections.reserve(
            participant_ids.size() *
            static_cast<std::size_t>(config_.num_layers));
        for (const int participant_id : participant_ids)
        {
            const auto found = endpoints_.find(participant_id);
            if (found == endpoints_.end())
            {
                throw std::logic_error(
                    "ExpertOverlay initial-bank snapshot lost a registered participant");
            }

            const auto lease = found->second.endpoint->acquire(
                config_.initial_epoch);
            if (!lease)
            {
                throw std::logic_error(
                    "ExpertOverlay initial-bank snapshot requires every local bank to be installed");
            }
            if (lease->participant_id != participant_id ||
                lease->device != found->second.endpoint->device() ||
                lease->layers.size() !=
                    static_cast<std::size_t>(config_.num_layers))
            {
                throw std::logic_error(
                    "ExpertOverlay installed initial bank has stale identity or geometry");
            }

            for (int layer_idx = 0;
                 layer_idx < config_.num_layers;
                 ++layer_idx)
            {
                const auto &layer =
                    lease->layers[static_cast<std::size_t>(layer_idx)];
                if (layer.resident_mask.size() !=
                    static_cast<std::size_t>(config_.num_experts))
                {
                    throw std::logic_error(
                        "ExpertOverlay installed initial bank has stale expert geometry");
                }

                InitialBankExpertSelection selection{
                    .participant_id = participant_id,
                    .device = lease->device,
                    .layer_idx = layer_idx,
                };
                selection.expert_ids.reserve(
                    static_cast<std::size_t>(std::count(
                        layer.resident_mask.begin(),
                        layer.resident_mask.end(),
                        true)));
                for (int expert_id = 0;
                     expert_id < config_.num_experts;
                     ++expert_id)
                {
                    if (layer.resident_mask[
                            static_cast<std::size_t>(expert_id)])
                    {
                        selection.expert_ids.push_back(expert_id);
                    }
                }
                selections.push_back(std::move(selection));
            }
        }
        return selections;
    }

    std::vector<
        MoEOverlayParticipantResidencyRegistry::IncompleteInitialBank>
    MoEOverlayParticipantResidencyRegistry::incompleteInitialBanks() const
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<IncompleteInitialBank> incomplete;
        incomplete.reserve(endpoints_.size());
        for (const auto &[participant_id, assembly] : endpoints_)
        {
            if (assembly.endpoint->acquire(config_.initial_epoch))
                continue;

            IncompleteInitialBank deficit;
            deficit.participant_id = participant_id;
            deficit.device = assembly.initial_bank.device;
            for (std::size_t layer = 0;
                 layer < assembly.registered_layers.size();
                 ++layer)
            {
                if (!assembly.registered_layers[layer])
                    deficit.missing_layers.push_back(
                        static_cast<int>(layer));
            }
            deficit.publication_pending = deficit.missing_layers.empty();
            incomplete.push_back(std::move(deficit));
        }
        std::sort(
            incomplete.begin(),
            incomplete.end(),
            [](const auto &left, const auto &right)
            {
                return left.participant_id < right.participant_id;
            });
        return incomplete;
    }

    bool MoEOverlayParticipantResidencyRegistry::trySnapshotServiceMeasurements(
        std::vector<MoEOverlayParticipantLayerServiceTotals> *output) const
    {
        if (!output)
            return false;
        output->clear();

        std::shared_lock<std::shared_mutex> lock(mutex_);
        std::vector<int> participant_ids;
        participant_ids.reserve(endpoints_.size());
        for (const auto &[participant_id, assembly] : endpoints_)
        {
            (void)assembly;
            participant_ids.push_back(participant_id);
        }
        std::sort(participant_ids.begin(), participant_ids.end());

        output->reserve(
            participant_ids.size() *
            static_cast<std::size_t>(config_.num_layers));
        for (const int participant_id : participant_ids)
        {
            const auto found = endpoints_.find(participant_id);
            if (found == endpoints_.end() || !found->second.endpoint)
            {
                output->clear();
                return false;
            }
            std::vector<MoEOverlayParticipantLayerServiceTotals> rows;
            if (!found->second.endpoint->trySnapshotServiceMeasurements(
                    &rows))
            {
                output->clear();
                return false;
            }
            output->insert(
                output->end(),
                std::make_move_iterator(rows.begin()),
                std::make_move_iterator(rows.end()));
        }
        return true;
    }

    bool MoEOverlayParticipantResidencyRegistry::
        importDeviceServiceMeasurements(
            int participant_id,
            const std::vector<MoEOverlayParticipantLayerServiceTotals> &rows,
            std::string *error)
    {
        if (error)
            error->clear();
        std::shared_lock<std::shared_mutex> lock(mutex_);
        const auto found = endpoints_.find(participant_id);
        if (found == endpoints_.end() || !found->second.endpoint)
        {
            if (error)
                *error = "device service snapshot names a non-local participant";
            return false;
        }
        return found->second.endpoint->importServiceMeasurements(rows, error);
    }
} // namespace llaminar2
