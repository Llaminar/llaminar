/**
 * @file MoEOverlayParticipantResidency.cpp
 * @brief Implementation of participant-local epoch-indexed expert banks.
 *
 * One RCU publication protocol retains immutable movable payloads for both
 * complete experts and gate/up owners. Projection-distributed down weights
 * instead use a model-lifetime bank: they never enter a movement epoch or
 * disappear when this participant no longer owns an expert's gate/up pair.
 * Cold producer binding authenticates the original context/manager once;
 * weak retained ownership avoids cycles, while scoped source uses protect only
 * active preparation. Existing bank leases retain their own physical engines.
 * Shadow calibration arrivals authenticate the original complete-target edge
 * without assigning its incoming expert to the unchanged live placement.
 * Cold diagnostics inspect the existing retired chain under its writer mutex;
 * published-slot count is distinct from readers pinning an unpublished node.
 */

#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayFixedDownProjectionBank.h"
#include "MoEOverlayParticipantMigration.h"
#include "MoEOverlayResidencyAuthority.h"
#include "MoEExpertOverlayPreparationPlan.h"
#include "DeviceMoEExpertDescriptorBuilder.h"

#include "../../loaders/ExpertGemmRegistry.h"
#include "loaders/ModelContext.h"
#include "loaders/WeightManager.h"
#include "../../tensors/NativeVnniFormatInfo.h"

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

    bool resolveMoEOverlayPreparedExpertPayloads(
        const ExpertGemmRegistry &registry,
        const MoEExpertOwnerParticipant &participant,
        int layer_idx,
        int num_experts,
        const std::vector<bool> &resident_mask,
        std::vector<MoEOverlayPreparedExpertPayload> &output,
        std::string *error,
        const std::optional<MoEExpertProjectionOwnership> &ownership)
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
                    "ExpertOverlay prepared-payload resolution has invalid "
                    "geometry or participant identity";
            }
            return false;
        }

        if (ownership && (ownership->geometry().experts != num_experts ||
                          ownership->participant() != participant.domain_participant_index ||
                          !participant.device.is_gpu()))
        {
            if (error) *error = "ExpertOverlay prepared payload does not match its frozen projection geometry";
            return false;
        }
        const auto projections = ownership ? ownership->movableProjections() : DeviceMoEProjectionSet::CompleteExpert;
        output.resize(static_cast<std::size_t>(num_experts));
        const int world_rank = participant.world_rank_known
                                   ? participant.world_rank
                                   : -1;
        for (int expert = 0; expert < num_experts; ++expert)
        {
            if (!resident_mask[static_cast<std::size_t>(expert)])
                continue;

            auto gate = registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::GATE, ownership);
            auto up = registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::UP, ownership);
            // Never borrow a fixed slice into the movable lifetime, even when
            // the registry contains it under this same logical expert ID.
            auto down = projections == DeviceMoEProjectionSet::GateUp ? std::shared_ptr<ITensorGemm>{}
                : registry.getEngineLifetimeForParticipant(
                participant.domain_name,
                participant.device,
                world_rank,
                participant.domain_participant_index,
                layer_idx,
                expert,
                ExpertGemmRegistry::WeightRole::DOWN, ownership);
            if (!gate || !up || (projections == DeviceMoEProjectionSet::CompleteExpert && !down))
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay participant p" +
                        std::to_string(participant.participant_id) +
                        " has no complete declared movable lifetime for layer " +
                        std::to_string(layer_idx) + " expert " +
                        std::to_string(expert);
                }
                output.clear();
                return false;
            }
            output[static_cast<std::size_t>(expert)] = MoEOverlayPreparedExpertPayload::fromProjections(
                projections, {std::move(gate), std::move(up), std::move(down)});
        }
        return true;
    }

    MoEOverlayFixedDownProjectionBank::MoEOverlayFixedDownProjectionBank(
        MoEExpertOwnerParticipant participant, int layer, MoEExpertProjectionOwnership ownership,
        std::vector<std::shared_ptr<ITensorGemm>> engines)
        : participant_(std::move(participant)), layer_(layer), ownership_(std::move(ownership)), engines_(std::move(engines))
    {
    }

    std::shared_ptr<const MoEOverlayFixedDownProjectionBank> MoEOverlayFixedDownProjectionBank::resolve(
        const ExpertGemmRegistry &registry, const MoEExpertOwnerParticipant &participant,
        int layer, const MoEExpertProjectionOwnership &ownership)
    {
        if (layer < 0 || participant.participant_id < 0 || !participant.device.is_gpu() ||
            participant.domain_name.empty() || participant.domain_participant_index != ownership.participant() ||
            ownership.movableProjections() != DeviceMoEProjectionSet::GateUp)
            throw std::invalid_argument("Fixed down bank requires an exact GPU participant and output-column contract");
        std::vector<std::shared_ptr<ITensorGemm>> engines;
        engines.reserve(static_cast<std::size_t>(ownership.geometry().experts));
        const int world_rank = participant.world_rank_known ? participant.world_rank : -1;
        for (int expert = 0; expert < ownership.geometry().experts; ++expert)
        {
            // The exact preparation identity authenticates the original N/K
            // and slice interval, not just the smaller matrix dimensions.
            auto engine = registry.getEngineLifetimeForParticipant(participant.domain_name, participant.device,
                world_rank, participant.domain_participant_index, layer, expert,
                ExpertGemmRegistry::WeightRole::DOWN, ownership);
            if (!engine)
                throw std::runtime_error("Fixed down bank is missing prepared slice for layer " +
                    std::to_string(layer) + " expert " + std::to_string(expert));
            engines.push_back(std::move(engine));
        }
        return std::shared_ptr<const MoEOverlayFixedDownProjectionBank>(
            new MoEOverlayFixedDownProjectionBank(participant, layer, ownership, std::move(engines)));
    }

    bool MoEOverlayFixedDownProjectionBank::sameIdentity(const MoEOverlayFixedDownProjectionBank &other) const noexcept
    {
        return layer_ == other.layer_ && ownership_ == other.ownership_ && engines_ == other.engines_ &&
               participant_.participant_id == other.participant_.participant_id &&
               participant_.address == other.participant_.address &&
               participant_.domain_name == other.participant_.domain_name && participant_.device == other.participant_.device &&
               participant_.world_rank_known == other.participant_.world_rank_known &&
               (!participant_.world_rank_known || participant_.world_rank == other.participant_.world_rank);
    }

    MoEOverlayFixedDownProjectionBank::DescriptorTable
    MoEOverlayFixedDownProjectionBank::exportDescriptorTable() const
    {
        const auto projection = ownership_.projection(WeightRole::MoEExpertDown);
        // Build a local candidate, then return it only after every expert has
        // validated. A late failure must never expose a partially usable table.
        DescriptorTable table = NativeDescriptorTable{};
        for (std::size_t expert = 0; expert < engines_.size(); ++expert)
        {
            const auto fail = [&](const char *reason) -> void
            {
                throw std::runtime_error("Fixed down descriptor for layer " +
                    std::to_string(layer_) + " expert " + std::to_string(expert) +
                    " on " + device().to_string() + ": " + reason);
            };
            auto &engine = *engines_[expert];
            DeviceNativeVNNIMatrixDesc native{};
            if (engine.exportNativeVNNIMatrixDesc(native))
            {
                if (!native.valid() || native.n != projection.rows || native.k != projection.source_columns ||
                    native.k % 32 != 0 || native.blocks_per_row != static_cast<uint32_t>(native.k / 32))
                    fail("prepared geometry differs from the fixed full-K slice");
                NativeVnniSourceIdentity source{};
                NativeVnniExecutionFormatEnvelope envelope;
                if (!engine.exportNativeVNNISourceIdentity(source) || !source.present ||
                    !addNativeVnniExecutionFormat(envelope, native.codebook_id, source, false) ||
                    (native.source_identity_present &&
                     (native.source_codebook_id != source.codebook_id ||
                      static_cast<bool>(native.source_is_superblock) != source.is_superblock)))
                    fail("native source arithmetic identity is missing or contradictory");
                // Some prepared engines export provenance separately. Carry
                // that same identity into the immutable kernel table rather
                // than letting a normalized execution codebook replace it.
                // The graph also retains ownership()'s original N in its
                // phase contract; the descriptor's N is the physical slice.
                native.source_codebook_id = source.codebook_id;
                native.source_is_superblock = static_cast<uint8_t>(source.is_superblock);
                native.source_identity_present = 1;
                auto *entries = std::get_if<NativeDescriptorTable>(&table);
                if (!entries) fail("mixed native and floating down descriptors");
                if (expert == 0) entries->experts.reserve(engines_.size());
                entries->experts.push_back(native);
                continue;
            }

            ContiguousFloatingPointWeightDescriptor source{};
            DeviceMoEFloatingMatrixDesc floating{};
            DeviceMoEWeightFormat format{};
            if (!engine.exportContiguousFloatingPointWeights(source) ||
                !exportDeviceMoEFloatingMatrixDescriptor(source, floating, format) ||
                floating.n != projection.rows || floating.k != projection.source_columns)
                fail("floating descriptor is invalid or differs from the fixed full-K slice");
            if (expert == 0)
            {
                table = FloatingDescriptorTable{format, {}};
                std::get<FloatingDescriptorTable>(table).experts.reserve(engines_.size());
            }
            auto *entries = std::get_if<FloatingDescriptorTable>(&table);
            if (!entries || entries->format != format)
                fail("mixed down descriptor arithmetic families or floating precisions");
            entries->experts.push_back(floating);
        }
        return table;
    }

    bool MoEOverlayCPUFragmentFamily::accepts(const MoEOverlayPreparedCPUFragmentPayload& payload,int expert) const noexcept
    {
        return cohort_ && payload.valid() && payload.model()==model_ && payload.expertId()==expert &&
            payload.cohort()==cohort_ &&
            !payload.cohort().owner_before(cohort_) && !cohort_.owner_before(payload.cohort()) &&
            payload.participantIndex()==member_ &&
            payload.preparation()==preparation_ &&
            !payload.preparation().owner_before(preparation_) && !preparation_.owner_before(payload.preparation());
    }

    void MoEOverlayParticipantLayerBank::setResidentCPUFragment(int expert,MoEOverlayPreparedCPUFragmentPayload fragment)
    {
        if(!cpu_fragment_family || !experts.empty() || expert<0 || size_t(expert)>=resident_mask.size() ||
           cpu_fragments.size()!=resident_mask.size() || !cpu_fragment_family->accepts(fragment,expert))
            throw std::invalid_argument("CPU fragment candidate requires its exact original family and closed destination payload");
        const auto at=std::lower_bound(cpu_fragment_expert_ids.begin(),cpu_fragment_expert_ids.end(),expert);
        if(at==cpu_fragment_expert_ids.end() || *at!=expert)cpu_fragment_expert_ids.insert(at,expert);
        cpu_fragments[size_t(expert)]=std::move(fragment);
        resident_mask[size_t(expert)]=true;
    }

    void MoEOverlayParticipantLayerBank::setResidentExpert(
        int expert_id,
        MoEOverlayPreparedExpertPayload engines)
    {
        if (cpu_fragment_family || expert_id < 0 ||
            static_cast<std::size_t>(expert_id) >= experts.size() ||
            experts.size() != resident_mask.size())
        {
            throw std::out_of_range(
                "ExpertOverlay participant bank expert id is outside geometry");
        }
        if (!engines.readyFor(movable_projections))
        {
            throw std::invalid_argument(
                "ExpertOverlay resident expert requires the endpoint's exact movable projection family");
        }

        const auto index = static_cast<std::size_t>(expert_id);
        /* Install lifetimes before publishing the resident bit in this mutable candidate. */
        experts[index] = std::move(engines);
        resident_mask[index] = true;
    }

    void MoEOverlayParticipantLayerBank::clearExpert(int expert_id)
    {
        if(cpu_fragment_family) {
            if(expert_id<0 || size_t(expert_id)>=cpu_fragments.size() || cpu_fragments.size()!=resident_mask.size() || !experts.empty())
                throw std::out_of_range("CPU fragment candidate expert is outside original geometry");
            const auto at=std::lower_bound(cpu_fragment_expert_ids.begin(),cpu_fragment_expert_ids.end(),expert_id);
            if(at!=cpu_fragment_expert_ids.end() && *at==expert_id)cpu_fragment_expert_ids.erase(at);
            resident_mask[size_t(expert_id)]=false;cpu_fragments[size_t(expert_id)]={};return;
        }

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
        if(cpu_fragment_family) {
            if(num_experts<=0 || movable_projections!=DeviceMoEProjectionSet::CompleteExpert || !experts.empty() ||
               resident_mask.size()!=size_t(num_experts) || cpu_fragments.size()!=size_t(num_experts))return false;
            size_t selected=0;
            for(size_t expert=0;expert<cpu_fragments.size();++expert) {
                if(resident_mask[expert]) {
                    if(!cpu_fragment_family->accepts(cpu_fragments[expert],int(expert)) || selected>=cpu_fragment_expert_ids.size() ||
                       cpu_fragment_expert_ids[selected++]!=int(expert))return false;
                } else if(!cpu_fragments[expert].empty())return false;
            }
            return selected==cpu_fragment_expert_ids.size();
        }
        if(!cpu_fragments.empty() || !cpu_fragment_expert_ids.empty())return false;

        if ((movable_projections != DeviceMoEProjectionSet::CompleteExpert &&
             movable_projections != DeviceMoEProjectionSet::GateUp) || num_experts <= 0 ||
            resident_mask.size() != static_cast<std::size_t>(num_experts) ||
            experts.size() != static_cast<std::size_t>(num_experts))
        {
            return false;
        }

        for (std::size_t expert = 0; expert < experts.size(); ++expert)
        {
            if (resident_mask[expert] != experts[expert].readyFor(movable_projections))
                return false;
            if (!resident_mask[expert] && !experts[expert].empty())
                return false;
        }
        return true;
    }

    bool MoEOverlayParticipantLayerBank::sameIdentity(
        const MoEOverlayParticipantLayerBank &other) const noexcept
    {
        if(cpu_fragment_family!=other.cpu_fragment_family ||
           cpu_fragment_family.owner_before(other.cpu_fragment_family) || other.cpu_fragment_family.owner_before(cpu_fragment_family) || cpu_fragments.size()!=other.cpu_fragments.size() ||
           cpu_fragment_expert_ids!=other.cpu_fragment_expert_ids)return false;
        for(size_t expert=0;expert<cpu_fragments.size();++expert)
            if(!cpu_fragments[expert].sameIdentity(other.cpu_fragments[expert]))return false;

        if (movable_projections != other.movable_projections || resident_mask != other.resident_mask ||
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

    std::size_t MoEOverlayParticipantResidencyBank::storageIndexForModelLayer(int layer) const
    {
        if (!containsModelLayer(layer))
            throw std::out_of_range("ExpertOverlay participant bank layer is outside its owned interval");
        return static_cast<std::size_t>(layer - first_model_layer);
    }

    MoEOverlayParticipantLayerBank &MoEOverlayParticipantResidencyBank::layerForModelLayer(int layer)
    {
        return layers[storageIndexForModelLayer(layer)];
    }

    const MoEOverlayParticipantLayerBank &MoEOverlayParticipantResidencyBank::layerForModelLayer(int layer) const
    {
        return layers[storageIndexForModelLayer(layer)];
    }

    bool MoEOverlayParticipantResidencyBank::valid(
        int expected_participant_id,
        DeviceId expected_device,
        int num_layers,
        int num_experts,
        int expected_first_model_layer) const noexcept
    {
        if (epoch == 0 || participant_id != expected_participant_id ||
            device != expected_device || num_layers <= 0 || num_experts <= 0 ||
            first_model_layer < 0 || first_model_layer != expected_first_model_layer ||
            num_layers > std::numeric_limits<int>::max() - first_model_layer ||
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
            first_model_layer != other.first_model_layer ||
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
        if(!config_.cpu_fragment_families.empty() &&
           (!config_.device.is_cpu() || config_.cpu_fragment_families.size()!=size_t(config_.num_layers) ||
            std::any_of(config_.cpu_fragment_families.begin(),config_.cpu_fragment_families.end(),[](const auto& family){return !family;})))
            throw std::invalid_argument("CPU fragment endpoint requires its complete original per-layer family");

        if (config_.participant_id < 0 || !config_.device.is_valid() ||
            config_.num_layers <= 0 || config_.num_experts <= 0 ||
            config_.first_model_layer < 0 ||
            config_.num_layers > std::numeric_limits<int>::max() - config_.first_model_layer ||
            config_.retained_epoch_capacity < 2 ||
            (config_.movable_projections != DeviceMoEProjectionSet::CompleteExpert &&
             config_.movable_projections != DeviceMoEProjectionSet::GateUp) ||
            (config_.movable_projections == DeviceMoEProjectionSet::GateUp && !config_.device.is_gpu()))
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
        if (layer < firstModelLayer() || layer >= endModelLayer() ||
            phase >= kExpertHistogramProductionSourceCount)
        {
            return std::numeric_limits<std::size_t>::max();
        }
        return static_cast<std::size_t>(layer - firstModelLayer()) *
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
                row.layer != firstModelLayer() + layer || !row.valid())
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
            row.layer = firstModelLayer() + layer;
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
                config_.num_experts,
                config_.first_model_layer) ||
            std::any_of(bank.layers.begin(), bank.layers.end(), [this](const auto &layer) {
                return layer.movable_projections != config_.movable_projections;
            }))
        {
            if (error)
            {
                *error =
                    "ExpertOverlay participant ready bank is incomplete or has wrong geometry";
            }
            return std::nullopt;
        }

        for(size_t row=0;row<bank.layers.size();++row) {
            const auto expected=config_.cpu_fragment_families.empty()?nullptr:config_.cpu_fragment_families[row];
            if(bank.layers[row].cpu_fragment_family!=expected ||
               bank.layers[row].cpu_fragment_family.owner_before(expected) || expected.owner_before(bank.layers[row].cpu_fragment_family)) {
                if(error)*error="CPU fragment ready bank replaced the original endpoint family";
                return std::nullopt;
            }
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

    std::optional<uint64_t>
    MoEOverlayParticipantResidency::retiredEpochReaders(uint64_t epoch) const noexcept
    {
        std::lock_guard<std::mutex> lock(state_->writer_mutex_);
        for (const auto *node = state_->retired_head_.get(); node;
             node = node->retired_next.get())
        {
            if (node->bank.epoch == epoch)
                return node->reader_count.load(std::memory_order_acquire);
        }
        return std::nullopt;
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

    const MoEExpertOwnerMap& MoEOverlayParticipantResidencyRegistry::initialOwnerMap()const noexcept
    {
        const auto source=config_.projection_preparation?config_.projection_preparation->cpuJointOwnerMap():nullptr;
        return source?*source:config_.owner_map;
    }

    MoEOverlayParticipantResidencyRegistry::
        MoEOverlayParticipantResidencyRegistry(Config config)
        : config_(std::move(config))
    {
        if (config_.num_layers <= 0 || config_.num_experts <= 0 ||
            config_.first_model_layer < 0 ||
            config_.num_layers > std::numeric_limits<int>::max() - config_.first_model_layer ||
            config_.initial_epoch == 0 ||
            config_.retained_epoch_capacity < 2)
        {
            throw std::invalid_argument(
                "ExpertOverlay participant registry requires positive geometry, "
                "an initial epoch, and two-bank capacity");
        }
        const auto joint_map=config_.projection_preparation?config_.projection_preparation->cpuJointOwnerMap():nullptr;
        if(joint_map) {
            if(!config_.original_model.value || config_.collect_economy_service_measurements ||
               !config_.owner_map.owners().empty() || !config_.owner_map.participants().empty() ||
               joint_map->owners().size()!=size_t(config_.num_layers)*size_t(config_.num_experts))
                throw std::invalid_argument("CPU joint registry requires its original model/source map and Static service policy");
            for(size_t i=0;i<joint_map->owners().size();++i)
                if(joint_map->owners()[i].layer_idx!=config_.first_model_layer+int(i/size_t(config_.num_experts)) ||
                   joint_map->owners()[i].expert_id!=int(i%size_t(config_.num_experts)))
                    throw std::invalid_argument("CPU joint registry source belongs to another layer/expert interval");
        } else {
            initialOwnerMap().requireLayerGeometry(
                config_.num_layers,config_.num_experts,config_.first_model_layer);
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
                initialOwnerMap().participantForId(participant_id);
            if (!participant || !participant->device.is_valid())
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant registry references an unknown local endpoint");
            }

            EndpointAssembly assembly;
            const bool joint=joint_map && std::any_of(joint_map->owners().begin(),joint_map->owners().end(),[&](const auto& row){
                if(!row.shard_cohort)return false;
                const auto members=row.shard_cohort->participants();
                return std::any_of(members.begin(),members.end(),[&](const auto& member){return member.participant_id==participant_id;});
            });
            std::vector<std::shared_ptr<const MoEOverlayCPUFragmentFamily>> fragment_families;
            if(joint) {
                if(participant->device!=DeviceId::cpu())throw std::invalid_argument("Joint registry endpoint is not its original CPU member");
                assembly.joint_layers.emplace(size_t(config_.num_layers));
                for(int row=0;row<config_.num_layers;++row) {
                    const int layer=config_.first_model_layer+row;
                    const auto ids=joint_map->expertsForParticipant(layer,participant_id);
                    std::shared_ptr<const MoEExpertShardCohort> cohort;
                    if(!ids.empty()) {
                        const auto selected=config_.projection_preparation->cpuShardSourceFor(layer,*participant);
                        if(selected->expertIds().size()!=ids.size() || !std::equal(ids.begin(),ids.end(),selected->expertIds().begin()))
                            throw std::invalid_argument("Joint registry source selection changed original fragment IDs");
                        cohort=selected->cohort();
                    }
                    // A proven original empty layer retains its closed empty
                    // family; it cannot mint incoming engines without a cohort.
                    fragment_families.push_back(std::shared_ptr<const MoEOverlayCPUFragmentFamily>(
                        new MoEOverlayCPUFragmentFamily(config_.original_model,config_.projection_preparation,
                            std::move(cohort),size_t(participant->domain_participant_index))));
                }
            }
            const auto projections = config_.projection_preparation && !joint_map
                ? DeviceMoEProjectionSet::GateUp : DeviceMoEProjectionSet::CompleteExpert;
            if (config_.projection_preparation && !joint_map)
            {
                for (int layer = config_.first_model_layer;
                     layer < config_.first_model_layer + config_.num_layers; ++layer)
                {
                    const auto layout = config_.projection_preparation->requireProjectionOwnershipForParticipant(
                        *participant, layer);
                    if (layout.geometry().experts != config_.num_experts)
                        throw std::invalid_argument("ExpertOverlay registry and projection preparation disagree on expert geometry");
                }
            }
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
                        .movable_projections = projections,
                        .first_model_layer = config_.first_model_layer,
                        .cpu_fragment_families=fragment_families,
                    });
            assembly.initial_bank.epoch = config_.initial_epoch;
            assembly.initial_bank.participant_id = participant_id;
            assembly.initial_bank.device = participant->device;
            assembly.initial_bank.first_model_layer = config_.first_model_layer;
            assembly.initial_bank.layers.resize(
                static_cast<std::size_t>(config_.num_layers));
            assembly.registered_layers.assign(
                static_cast<std::size_t>(config_.num_layers), false);

            for (int layer = 0; layer < config_.num_layers; ++layer)
            {
                auto &layer_bank =
                    assembly.initial_bank.layers[
                        static_cast<std::size_t>(layer)];
                layer_bank.movable_projections = projections;
                layer_bank.resident_mask =
                    initialOwnerMap().expertMaskForParticipant(
                        config_.first_model_layer + layer,
                        participant_id,
                        config_.num_experts);
                if(joint) {
                    layer_bank.cpu_fragment_family=fragment_families[size_t(layer)];
                    layer_bank.cpu_fragments.resize(size_t(config_.num_experts));
                } else layer_bank.experts.resize(static_cast<std::size_t>(config_.num_experts));

                /* No engines are needed for a layer with no local residents. */
                assembly.registered_layers[static_cast<std::size_t>(layer)] =
                    std::none_of(
                        layer_bank.resident_mask.begin(),
                        layer_bank.resident_mask.end(),
                        [](bool resident) { return resident; });
                if(joint && assembly.registered_layers[size_t(layer)])
                    (*assembly.joint_layers)[size_t(layer)]=std::shared_ptr<const CPUJointLayer>(new CPUJointLayer(
                        config_.initial_epoch,participant_id,config_.first_model_layer+layer,{},
                        fragment_families[size_t(layer)],assembly.endpoint));

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
                assembly.publication_state =
                    InitialPublicationState::Completed;
            }
            endpoints_.emplace(participant_id, std::move(assembly));
        }
    }

    namespace {
        /** @return Exact prepared/source identity, including shared allocation control blocks. */
        bool sameJointBankSource(const MoEOverlayParticipantResidencyRegistry::CPUJointLayer& actual,
                                 const ExpertGemmRegistry::StaticProjectionLayer& expected)
        {
            const auto& source=actual.sourceSelection();
            const auto construction=actual.acquireInitialConstruction();
            if(!source || !expected.cpu_joint_source || !construction ||
               source!=expected.cpu_joint_source || source.owner_before(expected.cpu_joint_source) ||
               expected.cpu_joint_source.owner_before(source))return false;
            const auto& layer=construction->layerForModelLayer(actual.layer());
            if(layer.cpu_fragments.size()!=expected.experts.size())return false;
            const auto same=[](const auto& a,const auto& b){return a.get()==b.get() && !a.owner_before(b) && !b.owner_before(a);};
            for(size_t expert=0;expert<expected.experts.size();++expert) {
                const auto& a=layer.cpu_fragments[expert];const auto& b=expected.experts[expert];
                if(!same(a.gate(),b.gate) || !same(a.up(),b.up) || !same(a.down(),b.down))return false;
            }
            return true;
        }
    }
    bool MoEOverlayParticipantResidencyRegistry::isCPUJointParticipant(int id)const noexcept
    {
        std::shared_lock lock(mutex_);const auto found=endpoints_.find(id);
        return found!=endpoints_.end() && bool(found->second.joint_layers);
    }
    std::optional<MoEOverlayParticipantResidencyRegistry::PreparedSourceUse>
    MoEOverlayParticipantResidencyRegistry::acquireOriginalPreparedSourceLocked(
        std::string *error) const
    {
        const auto reject = [&](const char *message) -> std::optional<PreparedSourceUse>
        {
            if (error)
                *error = message;
            return std::nullopt;
        };
        if (!prepared_source_bound_)
            return reject("Prepared source has no original ModelContext binding");
        auto context = prepared_source_context_.lock();
        auto manager = prepared_source_manager_.lock();
        if (!context || !manager)
            return reject(
                "Prepared source original context/manager has expired; rebinding is forbidden");
        const auto actual = context->concreteWeightManager();
        if (actual.get() != manager.get() || actual.owner_before(manager) ||
            manager.owner_before(actual) ||
            &manager->expertGemmRegistry() != prepared_source_registry_)
            return reject("Prepared source original manager/registry was replaced");
        if (error)
            error->clear();
        return PreparedSourceUse(std::move(context), std::move(manager), prepared_source_registry_,
                                 config_.projection_preparation);
    }
    std::optional<MoEOverlayParticipantResidencyRegistry::PreparedSourceUse>
    MoEOverlayParticipantResidencyRegistry::acquireOriginalPreparedSource(std::string *error) const
    {
        std::lock_guard lock(prepared_source_mutex_);
        return acquireOriginalPreparedSourceLocked(error);
    }
    bool MoEOverlayParticipantResidencyRegistry::acceptsPreparedRegistryLocked(
        const ExpertGemmRegistry &registry, std::string *error) const
    {
        if (prepared_source_registry_ && prepared_source_registry_ != &registry)
        {
            if (error)
                *error = "Prepared source engine registry differs from its original producer";
            return false;
        }
        return true;
    }
    bool MoEOverlayParticipantResidencyRegistry::authenticatePublishedPreparedSourceLocked(
        const ExpertGemmRegistry &registry, std::string *error) const
    {
        std::shared_lock lock(mutex_);
        for (const auto &[id, assembly] : endpoints_)
        {
            if (assembly.joint_layers)
                continue; // Exact committed-source check is already in joint registration.
            const auto *participant = initialOwnerMap().participantForId(id);
            if (!participant)
            {
                if (error)
                    *error = "Prepared source lost original participant";
                return false;
            }
            for (size_t row = 0; row < assembly.registered_layers.size(); ++row)
            {
                if (!assembly.registered_layers[row])
                    continue;
                const int layer = config_.first_model_layer + int(row);
                const auto &actual = assembly.initial_bank.layers[row];
                const auto ownership =
                    config_.projection_preparation &&
                            !config_.projection_preparation->cpuJointOwnerMap()
                        ? std::optional(
                              config_.projection_preparation
                                  ->requireProjectionOwnershipForParticipant(*participant, layer))
                        : std::nullopt;
                std::vector<MoEOverlayPreparedExpertPayload> expected;
                if (!resolveMoEOverlayPreparedExpertPayloads(
                        registry, *participant, layer, config_.num_experts, actual.resident_mask,
                        expected, error, ownership))
                    return false;
                for (size_t expert = 0; expert < expected.size(); ++expert)
                {
                    if (!actual.experts[expert].sameIdentity(expected[expert]))
                    {
                        if (error)
                            *error = "Published component bank differs from original model source";
                        return false;
                    }
                    const auto sameOwner = [](const auto &a, const auto &b)
                    { return !a.owner_before(b) && !b.owner_before(a); };
                    if (!sameOwner(actual.experts[expert].gate(), expected[expert].gate()) ||
                        !sameOwner(actual.experts[expert].up(), expected[expert].up()) ||
                        !sameOwner(actual.experts[expert].down(), expected[expert].down()))
                    {
                        if (error)
                            *error = "Published component bank has an independent engine owner";
                        return false;
                    }
                }
            }
        }
        return true;
    }
    bool MoEOverlayParticipantResidencyRegistry::bindOriginalPreparedContext(
        const std::shared_ptr<ModelContext> &context,
        const std::shared_ptr<const MoEExpertOverlayPreparationPlan> &preparation,
        std::string *error)
    {
        std::lock_guard lock(prepared_source_mutex_);
        const auto reject = [&](const char *message)
        {
            if (error)
                *error = message;
            return false;
        };
        if (!context || !config_.original_model.value ||
            config_.original_model.value != reinterpret_cast<uint64_t>(context.get()))
            return reject("Prepared source requires its actual original ModelContext identity");
        const auto &expected = config_.projection_preparation;
        if (preparation.get() != expected.get() || preparation.owner_before(expected) ||
            expected.owner_before(preparation))
            return reject(
                "Prepared source preparation differs from its original admission control block");
        auto manager = context->concreteWeightManager();
        if (!manager)
            return reject("Prepared source original ModelContext has no concrete WeightManager");
        if (prepared_source_bound_)
        {
            auto original = acquireOriginalPreparedSourceLocked(error);
            if (!original)
                return false;
            if (original->context_.get() != context.get() ||
                original->context_.owner_before(context) ||
                context.owner_before(original->context_) ||
                original->manager_.get() != manager.get() ||
                original->manager_.owner_before(manager) ||
                manager.owner_before(original->manager_))
                return reject(
                    "Prepared source cannot replace its original shared context/manager owner");
            if (error)
                error->clear();
            return true;
        }
        if (!acceptsPreparedRegistryLocked(manager->expertGemmRegistry(), error))
            return false;
        try
        {
            if (!authenticatePublishedPreparedSourceLocked(manager->expertGemmRegistry(), error))
                return false;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return false;
        }
        prepared_source_context_ = context;
        prepared_source_manager_ = manager;
        prepared_source_registry_ = &manager->expertGemmRegistry();
        prepared_source_bound_ = true;
        if (error)
            error->clear();
        return true;
    }
    bool MoEOverlayParticipantResidencyRegistry::admitInitialCPUJointGraphSources(
        const std::shared_ptr<ModelContext> &context, std::string *error)
    {
        const auto participants = localParticipantIds();
        if (std::none_of(participants.begin(), participants.end(),
                         [this](int id) { return isCPUJointParticipant(id); }))
        {
            if (error) error->clear();
            return true;
        }
        // Weight preparation has completed on the original cold owner. Bind
        // that owner before any graph consumes the immutable source declaration;
        // maintenance finalization is later and cannot establish this dependency.
        if (!bindOriginalPreparedContext(context, config_.projection_preparation, error))
            return false;
        auto source = acquireOriginalPreparedSource(error);
        if (!source) return false;
        for (const int id : participants)
            if (isCPUJointParticipant(id) &&
                !registerInitialCPUJointBank(source->registry(), config_.original_model, id, error))
                return false;
        return true;
    }

    bool MoEOverlayParticipantResidencyRegistry::registerInitialCPUJointBank(
        const ExpertGemmRegistry &registry, ModelContextId model, int id, std::string *error)
    {
        if (error)
            error->clear();
        std::lock_guard lock(prepared_source_mutex_);
        auto source =
            prepared_source_bound_ ? acquireOriginalPreparedSourceLocked(error) : std::nullopt;
        if ((prepared_source_bound_ && !source) || !acceptsPreparedRegistryLocked(registry, error))
            return false;
        if (!registerInitialCPUJointBankImpl(registry, model, id, error))
            return false;
        prepared_source_registry_ = &registry;
        return true;
    }

    bool MoEOverlayParticipantResidencyRegistry::registerInitialCPUJointBankImpl(
        const ExpertGemmRegistry &registry, ModelContextId model, int id, std::string *error)
    {
        const auto reject = [&](const std::string &message)
        {
            if (error)
                *error = message;
            return false;
        };
        if (!model.value || model.value != config_.original_model.value)
            return reject("CPU joint bank requires its original model identity");
        const auto *participant = initialOwnerMap().participantForId(id);
        std::shared_ptr<MoEOverlayParticipantResidency> endpoint;
        MoEOverlayParticipantResidencyBank candidate;
        MoEOverlayParticipantResidencyBank declaration;
        {
            std::shared_lock lock(mutex_);
            const auto found = endpoints_.find(id);
            if (!participant || found == endpoints_.end() || !found->second.joint_layers ||
                !found->second.endpoint)
                return reject("CPU joint bank requires its exact original local member");
            const auto &assembly = found->second;
            endpoint = assembly.endpoint;
            if (assembly.publication_state == InitialPublicationState::Completed)
            {
                // The completed state certifies the source once. The actual
                // installed node, not mutable registry lookup entries, is the
                // immutable engine authority for every retained graph family.
                const auto installed = endpoint->acquire(config_.initial_epoch);
                if (!installed || installed->participant_id != id ||
                    installed->device != participant->device ||
                    installed->first_model_layer != config_.first_model_layer ||
                    installed->layers.size() != size_t(config_.num_layers))
                    return reject("CPU joint initial bank is retired or absent; cold registration "
                                  "cannot resurrect it");
                return true;
            }
            candidate = assembly.initial_bank;
            declaration = assembly.initial_bank;
        }
        std::vector<std::shared_ptr<const CPUJointLayer>> frames(size_t(config_.num_layers));
        std::vector<bool> registered(size_t(config_.num_layers), true);
        try
        {
            for (int index = 0; index < config_.num_layers; ++index)
            {
                const int layer = config_.first_model_layer + index;
                auto &destination = candidate.layers.at(size_t(index));
                const auto &family = destination.cpu_fragment_family;
                const auto ids = initialOwnerMap().expertsForParticipant(layer, id);
                std::shared_ptr<const CPUExpertShardSourceSelection> selection;
                if (!ids.empty())
                {
                    const auto expected =
                        config_.projection_preparation->cpuShardSourceFor(layer, *participant);
                    auto acquired = registry.acquireCPUJointLayer(model, *participant, layer);
                    if (!expected || !acquired || !acquired->cpu_joint_source ||
                        *acquired->cpu_joint_source != *expected ||
                        acquired->cpu_joint_source->cohort() != expected->cohort() ||
                        acquired->cpu_joint_source->cohort().owner_before(expected->cohort()) ||
                        expected->cohort().owner_before(acquired->cpu_joint_source->cohort()))
                        return reject("CPU joint publication has no exact original source/cohort "
                                      "lease for layer " +
                                      std::to_string(layer));
                    selection = acquired->cpu_joint_source;
                    if (!family || family->cohort() != selection->cohort() ||
                        family->cohort().owner_before(selection->cohort()) ||
                        selection->cohort().owner_before(family->cohort()) ||
                        family->participantIndex() != selection->participantIndex())
                        return reject("CPU joint initial bank changed original source family");
                    for (const int expert : ids)
                    {
                        const auto &actual = acquired->experts.at(size_t(expert));
                        MoEOverlayPreparedCPUFragmentPayload payload(
                            model, config_.projection_preparation, family->cohort(),
                            family->participantIndex(), expert,
                            {actual.gate, actual.up, actual.down},0);
                        destination.setResidentCPUFragment(expert, std::move(payload));
                    }
                }
                frames[size_t(index)] = std::shared_ptr<const CPUJointLayer>(new CPUJointLayer(
                    config_.initial_epoch, id, layer, selection, family, endpoint));
                for (auto &payload : declaration.layers[size_t(index)].cpu_fragments)
                    payload = {};
            }
            // All allocations, source checks and frame construction precede
            // the first visible store. Only this one installed bank owns the
            // engines; setup bookkeeping retains declaration metadata alone.
            auto ready = endpoint->prepareReadyBank(std::move(candidate), error);
            if (!ready)
                return false;
            const auto status = endpoint->installReadyBank(std::move(*ready), error);
            if (status != MoEOverlayParticipantBankInstallStatus::Installed)
                return false;
            std::unique_lock lock(mutex_);
            auto &assembly = endpoints_.at(id);
            assembly.initial_bank = std::move(declaration);
            assembly.joint_layers = std::move(frames);
            assembly.registered_layers = std::move(registered);
            assembly.publication_state = InitialPublicationState::Completed;
            return true;
        }
        catch (const std::exception &exception)
        {
            return reject(exception.what());
        }
    }

    std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::CPUJointLayer>
    MoEOverlayParticipantResidencyRegistry::initialCPUJointLayer(int id,int layer)const
    {return cpuJointLayer(id,layer,config_.initial_epoch);}
    std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::CPUJointLayer>
    MoEOverlayParticipantResidencyRegistry::cpuJointLayer(int id,int layer,uint64_t epoch)const
    {
        if(epoch<config_.initial_epoch || layer<config_.first_model_layer || layer-config_.first_model_layer>=config_.num_layers)
            throw std::invalid_argument("CPU joint layer received an invalid epoch or foreign model layer");
        std::shared_ptr<const CPUJointLayer> original;
        {
            std::shared_lock lock(mutex_);const auto found=endpoints_.find(id);
            if(found==endpoints_.end() || !found->second.joint_layers)
                throw std::invalid_argument("CPU joint layer requires its exact registered fragment endpoint");
            original=found->second.joint_layers->at(size_t(layer-config_.first_model_layer));
        }
        if(!original)throw std::logic_error("CPU joint layer has no published original prepared source");
        if(epoch==config_.initial_epoch)return original;
        auto acquired=original->acquireEpoch(epoch);
        if(!acquired)throw std::logic_error("CPU joint requested epoch is absent or has a foreign fragment family");
        return std::shared_ptr<const CPUJointLayer>(new CPUJointLayer(std::move(*acquired)));
    }

    std::optional<MoEOverlayParticipantResidencyRegistry::CPUJointLayer>
    MoEOverlayParticipantResidencyRegistry::CPUJointLayer::acquireEpoch(uint64_t epoch) const noexcept
    {
        if(!endpoint_ || !family_ || epoch==0)return std::nullopt;
        auto bank=endpoint_->acquire(epoch);
        if(!bank || bank->epoch!=epoch || bank->participant_id!=participant_ || !bank->containsModelLayer(layer_))return std::nullopt;
        const auto& layer=bank->layerForModelLayer(layer_);
        if(layer.cpu_fragment_family!=family_ || layer.cpu_fragment_family.owner_before(family_) || family_.owner_before(layer.cpu_fragment_family))return std::nullopt;
        auto result=*this;result.epoch_=epoch;result.bank_=std::move(bank);return result;
    }


    std::shared_ptr<MoEOverlayPreparedExpertArrival>
    MoEOverlayParticipantResidencyRegistry::prepareCPUJointArrival(
        const MoEOverlayResidencySnapshot& candidate,int id,int layer,int expert,std::string* error) const
    {
        if(error)error->clear();
        const auto reject=[&](const char* reason)->std::shared_ptr<MoEOverlayPreparedExpertArrival>{if(error)*error=reason;return {};};
        // A model ID or passive source geometry alone cannot issue this arm.
        auto original=acquireOriginalPreparedSource(error);
        if(!original)return {};
        if(original->preparation()!=config_.projection_preparation ||
           candidate.originalProjectionPreparation()!=config_.projection_preparation ||
           candidate.originalProjectionPreparation().owner_before(config_.projection_preparation) ||
           config_.projection_preparation.owner_before(candidate.originalProjectionPreparation()) ||
           !candidate.valid() || candidate.epoch<=config_.initial_epoch ||
           layer<config_.first_model_layer || layer-config_.first_model_layer>=config_.num_layers || expert<0 || expert>=config_.num_experts)
            return reject("CPU fragment arrival requires the original preparation and a later candidate epoch");
        try {
            const auto& owners=candidate.placementOwnerMap();
            const auto* selected=owners.participantForId(id);
            const auto* initial=initialOwnerMap().participantForId(id);
            const auto* row=owners.placementFor(layer,expert);
            if(!selected || !initial || !(*selected==*initial) || !row || !row->shard_cohort ||
               row->layer_idx!=layer || row->expert_id!=expert)
                return reject("CPU fragment arrival changed original participant or candidate placement");
            std::shared_lock lock(mutex_);
            const auto found=endpoints_.find(id);
            if(found==endpoints_.end() || !found->second.joint_layers || !found->second.endpoint)
                return reject("CPU fragment arrival has no original local RCU endpoint");
            const auto& family=found->second.initial_bank.layerForModelLayer(layer).cpu_fragment_family;
            if(!family || !family->cohort() || family->cohort()!=row->shard_cohort ||
               family->cohort().owner_before(row->shard_cohort) || row->shard_cohort.owner_before(family->cohort()) ||
               family->model()!=config_.original_model || family->preparation()!=original->preparation())
                return reject("CPU fragment candidate changed original fixed cohort or model family");
            const auto members=family->cohort()->participants();
            if(family->participantIndex()>=members.size() || members[family->participantIndex()].participant_id!=id ||
               members[family->participantIndex()].world_rank!=selected->world_rank ||
               members[family->participantIndex()].address!=selected->address)
                return reject("CPU fragment candidate has a foreign original cohort member");
            const auto target=MoEExpertPlacementTarget::fromPlacement(*row);
            bool receives=false;
            if(target)for(size_t member=0;member<target->participantCount();++member)receives=receives || target->participantId(member)==id;
            if(!receives)return reject("CPU fragment candidate does not assign this expert to the original member");
            return std::shared_ptr<MoEOverlayPreparedExpertArrival>(new MoEOverlayPreparedExpertArrival(
                config_.original_model,original->preparation(),family->cohort(),family->participantIndex(),candidate.epoch,expert));
        } catch(const std::exception& exception) {if(error)*error=exception.what();return {};}
    }

    std::shared_ptr<MoEOverlayPreparedExpertArrival>
    MoEOverlayParticipantResidencyRegistry::prepareCPUJointCalibrationArrival(
        const MoEOverlayResidencyTransaction &transaction, std::size_t migration_index,
        int id, std::string *error) const
    {
        if (error) error->clear();
        const auto reject = [&](const char *reason) -> std::shared_ptr<MoEOverlayPreparedExpertArrival>
        { if (error) *error = reason; return {}; };
        // Only the closed authority-produced calibration transaction can name
        // a nonresident shadow destination while retaining unchanged live owners.
        if (transaction.purpose != MoEOverlayResidencyTransactionPurpose::EconomyCalibration ||
            !transaction.valid() || migration_index >= transaction.migrations.size())
            return reject("CPU fragment shadow arrival requires a genuine calibration transaction and edge");
        auto original = acquireOriginalPreparedSource(error);
        if (!original) return {};
        const auto &preparation = config_.projection_preparation;
        const auto same_preparation = [&](const auto &value)
        { return value && value.get() == preparation.get() &&
                 !value.owner_before(preparation) && !preparation.owner_before(value); };
        if (!same_preparation(original->preparation()) ||
            !same_preparation(transaction.previous->originalProjectionPreparation()) ||
            !same_preparation(transaction.candidate->originalProjectionPreparation()))
            return reject("CPU fragment calibration changed the original prepared source owner");
        const auto &edge = transaction.migrations[migration_index];
        const int layer = edge.layer_idx, expert = edge.expert_id;
        if (!edge.destination.shard_cohort || layer < config_.first_model_layer ||
            layer - config_.first_model_layer >= config_.num_layers || expert < 0 || expert >= config_.num_experts)
            return reject("CPU fragment calibration has a foreign destination layer or expert");
        try
        {
            const auto *selected = transaction.previous->placementOwnerMap().participantForId(id);
            const auto *initial = initialOwnerMap().participantForId(id);
            if (!selected || !initial || !(*selected == *initial))
                return reject("CPU fragment calibration changed the original participant");
            std::shared_lock lock(mutex_);
            const auto found = endpoints_.find(id);
            if (found == endpoints_.end() || !found->second.joint_layers || !found->second.endpoint)
                return reject("CPU fragment calibration has no original local RCU endpoint");
            auto bank = found->second.endpoint->acquire(transaction.expected_epoch);
            if (!bank || !bank->containsModelLayer(layer))
                return reject("CPU fragment calibration lost its original installed epoch");
            const auto &family = bank->layerForModelLayer(layer).cpu_fragment_family;
            const auto &cohort = edge.destination.shard_cohort;
            if (!family || !family->cohort() || family->cohort().get() != cohort.get() ||
                family->cohort().owner_before(cohort) || cohort.owner_before(family->cohort()) ||
                family->model() != config_.original_model || !same_preparation(family->preparation()))
                return reject("CPU fragment calibration changed the original fixed family or model");
            const auto members = cohort->participants();
            const auto member = family->participantIndex();
            if (member >= members.size() || members[member].participant_id != id ||
                members[member].world_rank != selected->world_rank || members[member].address != selected->address)
                return reject("CPU fragment calibration has a foreign original cohort member");
            // The genuine transaction validator authenticates this complete
            // original destination target. The live bank intentionally does NOT
            // own the incoming expert; no resident bit or source selection changes.
            return std::shared_ptr<MoEOverlayPreparedExpertArrival>(new MoEOverlayPreparedExpertArrival(
                config_.original_model, original->preparation(), cohort, member, transaction.candidate->epoch, expert));
        }
        catch (const std::exception &exception)
        { if (error) *error = exception.what(); return {}; }
    }

    std::shared_ptr<MoEOverlayPreparedExpertArrival>
    MoEOverlayParticipantResidencyRegistry::prepareCPUJointArrivalFromInstalledBank(
        const MoEOverlayParticipantBankLease &bank, int id, int layer, int expert,
        std::string *error) const
    {
        if (error)
            error->clear();
        const auto reject =
            [&](const char *reason) -> std::shared_ptr<MoEOverlayPreparedExpertArrival>
        {
            if (error)
                *error = reason;
            return {};
        };
        auto original = acquireOriginalPreparedSource(error);
        if (!original)
            return {};
        if (!bank || bank->participant_id != id || bank->epoch < config_.initial_epoch ||
            layer < config_.first_model_layer ||
            layer - config_.first_model_layer >= config_.num_layers ||
            !bank->containsModelLayer(layer) || expert < 0 || expert >= config_.num_experts)
            return reject("Installed CPU fragment arrival has a foreign or empty bank coordinate");
        try
        {
            std::shared_lock lock(mutex_);
            const auto found = endpoints_.find(id);
            const auto *participant = initialOwnerMap().participantForId(id);
            if (found == endpoints_.end() || !found->second.endpoint ||
                !found->second.joint_layers || !participant || bank->device != participant->device)
                return reject("Installed CPU fragment arrival has no original local endpoint");
            // A retained retired node remains physically safe to read, but cannot mint new work.
            const auto installed = found->second.endpoint->acquire(bank->epoch);
            if (!installed || installed.get() != bank.get())
                return reject(
                    "Installed CPU fragment arrival names a foreign or retired bank node");
            const auto &row = bank->layerForModelLayer(layer);
            const auto &family =
                found->second.initial_bank.layerForModelLayer(layer).cpu_fragment_family;
            if (!family || row.cpu_fragment_family != family ||
                row.cpu_fragment_family.owner_before(family) ||
                family.owner_before(row.cpu_fragment_family) ||
                family->model() != config_.original_model ||
                family->preparation() != original->preparation() ||
                family->preparation().owner_before(original->preparation()) ||
                original->preparation().owner_before(family->preparation()) || !family->cohort())
                return reject(
                    "Installed CPU fragment arrival changed its original model/preparation/family");
            const auto members = family->cohort()->participants();
            if (family->participantIndex() >= members.size() ||
                members[family->participantIndex()].participant_id != id ||
                members[family->participantIndex()].world_rank != participant->world_rank ||
                members[family->participantIndex()].address != participant->address ||
                !row.resident_mask.at(size_t(expert)) ||
                !family->accepts(row.cpu_fragments.at(size_t(expert)), expert))
                return reject(
                    "Installed CPU fragment arrival lost its original member or resident expert");
            return std::shared_ptr<MoEOverlayPreparedExpertArrival>(
                new MoEOverlayPreparedExpertArrival(config_.original_model, original->preparation(),
                                                    family->cohort(), family->participantIndex(),
                                                    bank->epoch, expert));
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
            return {};
        }
    }

    bool MoEOverlayParticipantResidencyRegistry::registerInitialLayer(
        int participant_id, int layer_idx, const std::vector<bool> &resident_mask,
        const std::vector<MoEOverlayPreparedExpertPayload> &experts, std::string *error)
    {
        std::lock_guard lock(prepared_source_mutex_);
        std::optional<PreparedSourceUse> source;
        if (isCPUJointParticipant(participant_id))
            return registerInitialLayerImpl(participant_id, layer_idx, resident_mask, experts, error);
        if (prepared_source_bound_)
        {
            source = acquireOriginalPreparedSourceLocked(error);
            if (!source)
                return false;
            const auto *participant = initialOwnerMap().participantForId(participant_id);
            if (!participant)
            {
                if (error)
                    *error = "Prepared source lost original participant";
                return false;
            }
            try
            {
                const auto ownership =
                    config_.projection_preparation &&
                            !config_.projection_preparation->cpuJointOwnerMap()
                        ? std::optional(config_.projection_preparation
                                            ->requireProjectionOwnershipForParticipant(*participant,
                                                                                       layer_idx))
                        : std::nullopt;
                std::vector<MoEOverlayPreparedExpertPayload> expected;
                if (!resolveMoEOverlayPreparedExpertPayloads(
                        source->registry(), *participant, layer_idx, config_.num_experts,
                        resident_mask, expected, error, ownership) ||
                    expected.size() != experts.size())
                    return false;
                for (size_t i = 0; i < expected.size(); ++i)
                {
                    const auto same = [](const auto &a, const auto &b)
                    { return a.get() == b.get() && !a.owner_before(b) && !b.owner_before(a); };
                    if (!expected[i].sameIdentity(experts[i]) ||
                        !same(expected[i].gate(), experts[i].gate()) ||
                        !same(expected[i].up(), experts[i].up()) ||
                        !same(expected[i].down(), experts[i].down()))
                    {
                        if (error)
                            *error =
                                "Initial payload differs from bound original model engines/owners";
                        return false;
                    }
                }
            }
            catch (const std::exception &exception)
            {
                if (error)
                    *error = exception.what();
                return false;
            }
        }
        return registerInitialLayerImpl(participant_id, layer_idx, resident_mask, experts, error);
    }
    bool MoEOverlayParticipantResidencyRegistry::registerInitialLayerImpl(
        int participant_id,
        int layer_idx,
        const std::vector<bool> &resident_mask,
        const std::vector<MoEOverlayPreparedExpertPayload> &experts,
        std::string *error)
    {
        if (error)
            error->clear();
        if(isCPUJointParticipant(participant_id)) {
            if(error)*error="CPU joint fragments require source-bound publication, never whole-expert payloads";
            return false;
        }
        if (layer_idx < config_.first_model_layer ||
            layer_idx >= config_.first_model_layer + config_.num_layers ||
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
            assembly.initial_bank.layerForModelLayer(layer_idx);
        if (resident_mask != expected_layer.resident_mask)
        {
            if (error)
                *error = "ExpertOverlay initial layer mask differs from the canonical owner map";
            return false;
        }

        MoEOverlayParticipantLayerBank supplied_layer{
            .resident_mask = resident_mask,
            .experts = experts,
            .movable_projections = assembly.endpoint->movableProjections(),
        };
        if (!supplied_layer.valid(config_.num_experts))
        {
            if (error)
                *error = "ExpertOverlay initial layer has incomplete or extraneous expert engines";
            return false;
        }

        /*
         * Initial publication is a one-way setup transition, not a proxy for
         * current residency.  Dynamic maintenance legitimately retires the
         * bootstrap epoch.  A later graph-cache variant may certify the same
         * prepared identities, but allowing it to reinstall epoch one would
         * occupy the inactive RCU slot beside the live epoch and permanently
         * prevent the next maintenance wave.
         */
        if (assembly.publication_state ==
            InitialPublicationState::Completed)
        {
            if (!expected_layer.sameIdentity(supplied_layer))
            {
                if (error)
                {
                    *error =
                        "Cached graph resolved different engines after initial publication completed";
                }
                return false;
            }
            return true;
        }

        const auto layer_index = assembly.initial_bank.storageIndexForModelLayer(layer_idx);
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
        if (status != MoEOverlayParticipantBankInstallStatus::Installed &&
            status !=
                MoEOverlayParticipantBankInstallStatus::AlreadyInstalled)
        {
            return false;
        }
        assembly.publication_state = InitialPublicationState::Completed;
        return true;
    }

    bool MoEOverlayParticipantResidencyRegistry::registerInitialLayerFromPreparedRegistry(
        const ExpertGemmRegistry &registry, int layer_idx, std::string *error)
    {
        if (error) error->clear();
        if (layer_idx < config_.first_model_layer ||
            layer_idx - config_.first_model_layer >= config_.num_layers)
        {
            if (error) *error = "Prepared layer is outside the frozen residency interval: " + std::to_string(layer_idx);
            return false;
        }
        std::lock_guard source_lock(prepared_source_mutex_);
        auto source=prepared_source_bound_?acquireOriginalPreparedSourceLocked(error):std::nullopt;
        if((prepared_source_bound_ && !source) || !acceptsPreparedRegistryLocked(registry,error))return false;
        // Configuration is immutable; registerInitialLayer owns the publication
        // lock. Never hold it while retaining model registry engine lifetimes.
        const auto participant_ids = localParticipantIds();
        for (const int participant_id : participant_ids)
        {
            if(isCPUJointParticipant(participant_id)) {
                if(!registerInitialCPUJointBankImpl(registry,config_.original_model,participant_id,error))return false;
                continue;
            }
            const auto *participant =
                initialOwnerMap().participantForId(participant_id);
            if (!participant)
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay initial-bank finalization lost participant " +
                        std::to_string(participant_id) +
                        " from the frozen owner map";
                }
                return false;
            }

            const auto resident_mask =
                initialOwnerMap().expertMaskForParticipant(
                    layer_idx,
                    participant_id,
                    config_.num_experts);
            std::vector<MoEOverlayPreparedExpertPayload> experts;
            std::string detail;
            const auto ownership = config_.projection_preparation && !config_.projection_preparation->cpuJointOwnerMap()
                ? std::optional(config_.projection_preparation->requireProjectionOwnershipForParticipant(
                    *participant, layer_idx)) : std::nullopt;
            if (!resolveMoEOverlayPreparedExpertPayloads(
                    registry,
                    *participant,
                    layer_idx,
                    config_.num_experts,
                    resident_mask,
                    experts,
                    &detail,
                    ownership))
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay initial-bank finalization could not "
                        "resolve participant " +
                        std::to_string(participant_id) + " layer " +
                        std::to_string(layer_idx) + ": " + detail;
                }
                return false;
            }
            if (!registerInitialLayerImpl(
                    participant_id,
                    layer_idx,
                    resident_mask,
                    experts,
                    &detail))
            {
                if (error)
                {
                    *error =
                        "ExpertOverlay initial-bank finalization rejected "
                        "participant " +
                        std::to_string(participant_id) + " layer " +
                        std::to_string(layer_idx) + ": " + detail;
                }
                return false;
            }
        }
        prepared_source_registry_=&registry;
        return true;
    }

    bool MoEOverlayParticipantResidencyRegistry::finalizeInitialBanksFromPreparedRegistry(
        const ExpertGemmRegistry &registry, std::string *error)
    {
        for (int layer = config_.first_model_layer;
             layer < config_.first_model_layer + config_.num_layers; ++layer)
            if (!registerInitialLayerFromPreparedRegistry(registry, layer, error)) return false;
        if (!allInitialBanksReady())
        {
            if (error)
            {
                *error =
                    "ExpertOverlay initial-bank finalization completed every "
                    "prepared layer but did not publish every local bank";
            }
            return false;
        }
        return true;
    }

    std::shared_ptr<MoEOverlayParticipantResidency>
    MoEOverlayParticipantResidencyRegistry::endpoint(
        int participant_id) const
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
            [](const auto &entry)
            {
                return entry.second.publication_state ==
                       InitialPublicationState::Completed;
            });
    }

    std::size_t MoEOverlayParticipantResidencyRegistry::verifyImmutablePreparedSources(
        const ExpertGemmRegistry &registry) const
    {
        std::size_t authenticated = 0;
        for (const int id : localParticipantIds())
        {
            const auto *participant = initialOwnerMap().participantForId(id);
            if(isCPUJointParticipant(id)) {
                if(!participant)throw std::logic_error("Joint source verification lost its original member");
                for(int layer=config_.first_model_layer;layer<config_.first_model_layer+config_.num_layers;++layer) {
                    const auto actual=initialCPUJointLayer(id,layer);
                    const auto expected=registry.acquireCPUJointLayer(config_.original_model,*participant,layer);
                    const auto ids=initialOwnerMap().expertsForParticipant(layer,id);
                    if(ids.empty()) {
                        if(actual->sourceSelection() || expected)throw std::logic_error("Empty joint layer acquired foreign source storage");
                    } else if(!expected || !sameJointBankSource(*actual,*expected)) {
                        throw std::logic_error("Joint prepared source seal found changed source/engine ownership");
                    }
                    authenticated+=ids.size();
                }
                continue;
            }
            const auto owner = endpoint(id);
            const auto bank = owner ? owner->acquire(config_.initial_epoch) : MoEOverlayParticipantBankLease{};
            if (!participant || !owner || !bank ||
                !bank->valid(id, participant->device, config_.num_layers, config_.num_experts, config_.first_model_layer))
                throw std::logic_error("Immutable prepared source seal lost its original scoped bank");
            for (int layer = config_.first_model_layer; layer < config_.first_model_layer + config_.num_layers; ++layer)
            {
                const auto mask = initialOwnerMap().expertMaskForParticipant(layer, id, config_.num_experts);
                const auto &row = bank->layerForModelLayer(layer);
                if (row.resident_mask != mask)
                    throw std::logic_error("Immutable prepared source seal changed initial ownership");
                const auto ownership = config_.projection_preparation && !config_.projection_preparation->cpuJointOwnerMap()
                    ? std::optional(config_.projection_preparation->requireProjectionOwnershipForParticipant(*participant, layer))
                    : std::nullopt;
                std::vector<MoEOverlayPreparedExpertPayload> expected;
                std::string error;
                if (!resolveMoEOverlayPreparedExpertPayloads(registry, *participant, layer,
                        config_.num_experts, mask, expected, &error, ownership))
                    throw std::logic_error("Immutable prepared source seal cannot resolve original engines: " + error);
                for (std::size_t expert = 0; expert < expected.size(); ++expert)
                {
                    if (!row.experts.at(expert).sameIdentity(expected[expert]))
                        throw std::logic_error("Immutable prepared source seal found a replaced engine identity");
                    if (mask[expert]) ++authenticated;
                }
            }
        }
        return authenticated;
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
            const auto *const participant =
                initialOwnerMap().participantForId(participant_id);
            if (!participant)
            {
                throw std::logic_error(
                    "ExpertOverlay initial-bank snapshot cannot resolve its canonical participant");
            }
            const auto tier_participants =
                initialOwnerMap().participantIdsForTier(participant->tier_idx);
            if (tier_participants.empty())
            {
                throw std::logic_error(
                    "ExpertOverlay initial-bank snapshot resolved an empty ownership tier");
            }
            const auto found = endpoints_.find(participant_id);
            if (found == endpoints_.end())
            {
                throw std::logic_error(
                    "ExpertOverlay initial-bank snapshot lost a registered participant");
            }

            if(found->second.joint_layers) {
                for(int layer=config_.first_model_layer;layer<config_.first_model_layer+config_.num_layers;++layer) {
                    const auto& actual=found->second.joint_layers->at(size_t(layer-config_.first_model_layer));
                    const auto construction = actual ? actual->acquireInitialConstruction()
                                                     : MoEOverlayParticipantBankLease{};
                    if(!actual || actual->epoch()!=config_.initial_epoch || !construction ||
                       construction->participant_id!=participant_id || construction->device!=participant->device ||
                       !construction->containsModelLayer(layer))
                        throw std::logic_error("Joint initial selection requires its actual published source lease");
                    InitialBankExpertSelection selection{.participant_id=participant_id,.device=participant->device,
                        .tier_idx=participant->tier_idx,.tier_participant_count=int(tier_participants.size()),.layer_idx=layer};
                    // Original source metadata survives retirement deliberately;
                    // it cannot stand in for an installed bank observation.
                    const auto& observed = construction->layerForModelLayer(layer);
                    if(observed.cpu_fragment_family!=actual->family() || !observed.valid(config_.num_experts))
                        throw std::logic_error("Joint initial selection lost its authenticated fragment bank");
                    selection.expert_ids=observed.cpu_fragment_expert_ids;
                    if(actual->sourceSelection()) {
                        const auto ids=actual->sourceSelection()->expertIds();
                        if(!std::equal(ids.begin(),ids.end(),selection.expert_ids.begin(),selection.expert_ids.end()))
                            throw std::logic_error("Joint initial bank differs from its immutable source selection");
                    }
                    selection.matches_frozen_owner_map=selection.expert_ids==initialOwnerMap().expertsForParticipant(layer,participant_id);
                    selection.ownership_kind=InitialOwnershipKind::CPUJointFragments;
                    selections.push_back(std::move(selection));
                }
                continue;
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
                lease->first_model_layer != config_.first_model_layer ||
                lease->layers.size() !=
                    static_cast<std::size_t>(config_.num_layers))
            {
                throw std::logic_error(
                    "ExpertOverlay installed initial bank has stale identity or geometry");
            }

            for (int layer_idx = config_.first_model_layer;
                 layer_idx < config_.first_model_layer + config_.num_layers;
                 ++layer_idx)
            {
                const auto &layer =
                    lease->layerForModelLayer(layer_idx);
                if (layer.resident_mask.size() !=
                    static_cast<std::size_t>(config_.num_experts))
                {
                    throw std::logic_error(
                        "ExpertOverlay installed initial bank has stale expert geometry");
                }

                InitialBankExpertSelection selection{
                    .participant_id = participant_id,
                    .device = lease->device,
                    .tier_idx = participant->tier_idx,
                    .tier_participant_count =
                        static_cast<int>(tier_participants.size()),
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
                selection.matches_frozen_owner_map =
                    selection.expert_ids ==
                    initialOwnerMap().expertsForParticipant(
                        layer_idx, participant_id);
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
            if (assembly.publication_state ==
                InitialPublicationState::Completed)
                continue;

            IncompleteInitialBank deficit;
            deficit.participant_id = participant_id;
            const auto* participant=initialOwnerMap().participantForId(participant_id);
            deficit.device=participant?participant->device:DeviceId::invalid();
            for (std::size_t layer = 0;
                 layer < assembly.registered_layers.size();
                 ++layer)
            {
                if (!assembly.registered_layers[layer])
                    deficit.missing_layers.push_back(
                        config_.first_model_layer + static_cast<int>(layer));
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
