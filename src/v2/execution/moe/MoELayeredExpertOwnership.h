/**
 * @file MoELayeredExpertOwnership.h
 * @brief Authoritative per-layer ownership of complete routed MoE experts.
 *
 * Dynamic expert movement is planned independently for every routed layer.
 * Representing the installed state as a single expert-to-participant vector
 * therefore loses information: two layers may legitimately assign the same
 * logical expert id to different participants.  This value type makes the
 * complete `(layer, expert) -> participant` relation explicit and validates it
 * at construction time.  Rebalance planning, migration, mask publication, and
 * graph/prefix identity all consume this same object.
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /**
     * @brief One installed ownership change between two layered plans.
     *
     * A change names the exact layer and expert whose complete prepared weight
     * triplet moves.  Keeping the layer in the type prevents callers from
     * accidentally expanding one layer-local decision across every model
     * layer.
     */
    struct MoELayeredExpertOwnershipChange
    {
        int layer_idx = -1;
        int expert_id = -1;
        int previous_participant = -1;
        int current_participant = -1;

        bool operator==(const MoELayeredExpertOwnershipChange &) const = default;
    };

    /**
     * @brief Validated dense ownership table for routed experts.
     *
     * Every layer has the same positive expert cardinality and every entry is
     * a valid participant id.  The class intentionally exposes no mutable row
     * references: ownership changes must pass through `assignOwner()`, which
     * preserves participant-range validation.
     */
    class MoELayeredExpertOwnership final
    {
    public:
        /** @brief Construct an empty, unbound ownership value. */
        MoELayeredExpertOwnership() = default;

        /**
         * @brief Construct and validate an explicit per-layer ownership table.
         *
         * @param participant_count Number of participants that may own experts.
         * @param owner_participant_by_layer Dense `[layer][expert]` owner table.
         * @throws std::invalid_argument if dimensions are empty or ragged, or
         *         an owner lies outside `[0, participant_count)`.
         */
        MoELayeredExpertOwnership(
            int participant_count,
            std::vector<std::vector<int>> owner_participant_by_layer)
            : participant_count_(participant_count),
              owner_participant_by_layer_(std::move(owner_participant_by_layer))
        {
            validate();
        }

        /**
         * @brief Repeat one owner row for every model layer.
         *
         * This is the canonical conversion for a static ownership policy that
         * intentionally assigns every layer alike.  Dynamic movement begins
         * from the resulting full table and may then diverge layer by layer.
         *
         * @param layer_count Positive number of routed layers.
         * @param participant_count Positive number of ownership participants.
         * @param owner_participant_by_expert One complete expert owner row.
         * @return Validated layered ownership value.
         */
        static MoELayeredExpertOwnership uniform(
            int layer_count,
            int participant_count,
            const std::vector<int> &owner_participant_by_expert)
        {
            if (layer_count <= 0)
            {
                throw std::invalid_argument(
                    "MoE layered ownership requires a positive layer count");
            }
            return MoELayeredExpertOwnership(
                participant_count,
                std::vector<std::vector<int>>(
                    static_cast<size_t>(layer_count),
                    owner_participant_by_expert));
        }

        /** @brief Return whether this value has not yet been bound to a model. */
        bool empty() const noexcept
        {
            return owner_participant_by_layer_.empty();
        }

        /** @brief Return the number of represented routed layers. */
        int layerCount() const noexcept
        {
            return static_cast<int>(owner_participant_by_layer_.size());
        }

        /** @brief Return the number of logical routed experts per layer. */
        int expertCount() const noexcept
        {
            return owner_participant_by_layer_.empty()
                       ? 0
                       : static_cast<int>(owner_participant_by_layer_.front().size());
        }

        /** @brief Return the number of possible ownership participants. */
        int participantCount() const noexcept
        {
            return participant_count_;
        }

        /**
         * @brief Return the owner of one exact layer/expert pair.
         *
         * @throws std::out_of_range if either index is outside the table.
         */
        int owner(int layer_idx, int expert_id) const
        {
            return ownersForLayer(layer_idx).at(checkedExpertIndex(expert_id));
        }

        /**
         * @brief Return the immutable owner row for one layer.
         *
         * @throws std::out_of_range if `layer_idx` is outside the table.
         */
        const std::vector<int> &ownersForLayer(int layer_idx) const
        {
            if (layer_idx < 0 || layer_idx >= layerCount())
            {
                throw std::out_of_range(
                    "MoE layered ownership layer index is out of range");
            }
            return owner_participant_by_layer_[static_cast<size_t>(layer_idx)];
        }

        /**
         * @brief Assign one exact layer/expert pair to a participant.
         *
         * @throws std::out_of_range for an invalid layer, expert, or participant.
         */
        void assignOwner(int layer_idx, int expert_id, int participant_id)
        {
            if (participant_id < 0 || participant_id >= participant_count_)
            {
                throw std::out_of_range(
                    "MoE layered ownership participant index is out of range");
            }
            auto &layer = owner_participant_by_layer_.at(checkedLayerIndex(layer_idx));
            layer.at(checkedExpertIndex(expert_id)) = participant_id;
        }

        /**
         * @brief Build all layer masks for one ownership participant.
         *
         * The returned table is `[layer][expert]`; every expert appears in
         * exactly one participant's masks because ownership itself is total.
         *
         * @throws std::out_of_range if `participant_id` is invalid.
         */
        std::vector<std::vector<bool>> masksForParticipant(int participant_id) const
        {
            if (participant_id < 0 || participant_id >= participant_count_)
            {
                throw std::out_of_range(
                    "MoE layered ownership mask participant is out of range");
            }

            std::vector<std::vector<bool>> masks(
                static_cast<size_t>(layerCount()),
                std::vector<bool>(static_cast<size_t>(expertCount()), false));
            for (int layer_idx = 0; layer_idx < layerCount(); ++layer_idx)
            {
                for (int expert_id = 0; expert_id < expertCount(); ++expert_id)
                {
                    masks[static_cast<size_t>(layer_idx)][static_cast<size_t>(expert_id)] =
                        owner(layer_idx, expert_id) == participant_id;
                }
            }
            return masks;
        }

        /**
         * @brief Return exact ownership changes from `previous` to this plan.
         *
         * @throws std::invalid_argument if the two plans describe different
         *         model or participant geometries.
         */
        std::vector<MoELayeredExpertOwnershipChange> changesFrom(
            const MoELayeredExpertOwnership &previous) const
        {
            requireSameGeometry(previous, "compare");

            std::vector<MoELayeredExpertOwnershipChange> changes;
            for (int layer_idx = 0; layer_idx < layerCount(); ++layer_idx)
            {
                for (int expert_id = 0; expert_id < expertCount(); ++expert_id)
                {
                    const int old_owner = previous.owner(layer_idx, expert_id);
                    const int new_owner = owner(layer_idx, expert_id);
                    if (old_owner == new_owner)
                        continue;
                    changes.push_back({
                        .layer_idx = layer_idx,
                        .expert_id = expert_id,
                        .previous_participant = old_owner,
                        .current_participant = new_owner,
                    });
                }
            }
            return changes;
        }

        /**
         * @brief Verify that every layer retains the same owner capacities.
         *
         * Paired ownership swaps must not strand one participant with more
         * resident experts than its prepared capacity.  This comparison is
         * stricter than merely checking total expert count.
         */
        bool hasSameLayerCapacitiesAs(
            const MoELayeredExpertOwnership &other) const
        {
            requireSameGeometry(other, "compare capacities");
            for (int layer_idx = 0; layer_idx < layerCount(); ++layer_idx)
            {
                std::vector<int> lhs_counts(static_cast<size_t>(participant_count_), 0);
                std::vector<int> rhs_counts(static_cast<size_t>(participant_count_), 0);
                for (int expert_id = 0; expert_id < expertCount(); ++expert_id)
                {
                    ++lhs_counts[static_cast<size_t>(owner(layer_idx, expert_id))];
                    ++rhs_counts[static_cast<size_t>(other.owner(layer_idx, expert_id))];
                }
                if (lhs_counts != rhs_counts)
                    return false;
            }
            return true;
        }

        bool operator==(const MoELayeredExpertOwnership &) const = default;

    private:
        int participant_count_ = 0;
        std::vector<std::vector<int>> owner_participant_by_layer_;

        size_t checkedLayerIndex(int layer_idx) const
        {
            if (layer_idx < 0 || layer_idx >= layerCount())
            {
                throw std::out_of_range(
                    "MoE layered ownership layer index is out of range");
            }
            return static_cast<size_t>(layer_idx);
        }

        size_t checkedExpertIndex(int expert_id) const
        {
            if (expert_id < 0 || expert_id >= expertCount())
            {
                throw std::out_of_range(
                    "MoE layered ownership expert index is out of range");
            }
            return static_cast<size_t>(expert_id);
        }

        void requireSameGeometry(
            const MoELayeredExpertOwnership &other,
            const char *operation) const
        {
            if (layerCount() == other.layerCount() &&
                expertCount() == other.expertCount() &&
                participant_count_ == other.participant_count_)
            {
                return;
            }

            throw std::invalid_argument(
                std::string("Cannot ") + operation +
                " MoE layered ownership values with different geometries");
        }

        void validate() const
        {
            if (participant_count_ <= 0)
            {
                throw std::invalid_argument(
                    "MoE layered ownership requires a positive participant count");
            }
            if (owner_participant_by_layer_.empty() ||
                owner_participant_by_layer_.front().empty())
            {
                throw std::invalid_argument(
                    "MoE layered ownership requires positive layer and expert counts");
            }

            const size_t expert_count = owner_participant_by_layer_.front().size();
            for (size_t layer_idx = 0;
                 layer_idx < owner_participant_by_layer_.size();
                 ++layer_idx)
            {
                const auto &owners = owner_participant_by_layer_[layer_idx];
                if (owners.size() != expert_count)
                {
                    throw std::invalid_argument(
                        "MoE layered ownership rows must have identical expert counts");
                }
                const auto invalid_owner = std::find_if(
                    owners.begin(),
                    owners.end(),
                    [&](int participant_id)
                    {
                        return participant_id < 0 ||
                               participant_id >= participant_count_;
                    });
                if (invalid_owner != owners.end())
                {
                    throw std::invalid_argument(
                        "MoE layered ownership contains an invalid participant id");
                }
            }
        }
    };
} // namespace llaminar2
