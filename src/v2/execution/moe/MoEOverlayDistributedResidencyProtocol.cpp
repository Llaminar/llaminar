/**
 * @file MoEOverlayDistributedResidencyProtocol.cpp
 * @brief Implementation of distributed ExpertOverlay residency consensus.
 *
 * Hashing is deliberately field-by-field and endian-stable.  Process-local
 * addresses, padding bytes, STL object representations, and pointer identities
 * never enter the wire identity, so independently constructed rank state can
 * authenticate the same declarative residency transaction.
 */

#include "MoEOverlayDistributedResidencyProtocol.h"
#include "MoEOverlayWireIO.h"

#include <algorithm>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2
{
    namespace
    {
        using moe_overlay_wire::readLittleEndian;
        using moe_overlay_wire::writeLittleEndian;
        /** @brief Two independently seeded byte mixers used as a 128-bit digest. */
        class StableDigestBuilder final
        {
        public:
            /** @brief Mix one unsigned or enum scalar in canonical little-endian order. */
            template <typename Value>
            void addScalar(Value value) noexcept
            {
                if constexpr (std::is_same_v<std::remove_cv_t<Value>, bool>)
                {
                    addByte(value ? 1u : 0u);
                }
                else
                {
                    using Raw = typename std::conditional_t<
                        std::is_enum_v<Value>,
                        std::underlying_type<Value>,
                        std::type_identity<Value>>::type;
                    using Unsigned = std::make_unsigned_t<Raw>;
                    Unsigned bits = static_cast<Unsigned>(value);
                    for (std::size_t byte_idx = 0;
                         byte_idx < sizeof(Unsigned);
                         ++byte_idx)
                    {
                        addByte(
                            static_cast<std::uint8_t>(bits & 0xffu));
                        bits >>= 8u;
                    }
                }
            }

            /** @brief Mix length and bytes so concatenated strings cannot alias. */
            void addString(const std::string &value) noexcept
            {
                addScalar(static_cast<std::uint64_t>(value.size()));
                for (const unsigned char byte : value)
                    addByte(byte);
            }

            /** @brief Return non-zero lanes even for an extremely rare zero digest. */
            template <typename Fingerprint =
                          MoEOverlayResidencyTransactionFingerprint>
            [[nodiscard]] Fingerprint finish() const noexcept
            {
                return {
                    .low = low_ == 0 ? 0x9e3779b97f4a7c15ull : low_,
                    .high = high_ == 0 ? 0xd6e8feb86659fd93ull : high_,
                };
            }

        private:
            /** @brief Mix one canonical byte through independent FNV-style lanes. */
            void addByte(std::uint8_t byte) noexcept
            {
                low_ ^= static_cast<std::uint64_t>(byte);
                low_ *= 1099511628211ull;

                high_ ^= static_cast<std::uint64_t>(byte) +
                         0x9e3779b97f4a7c15ull;
                high_ *= 14029467366897019727ull;
                high_ ^= high_ >> 29u;
            }

            std::uint64_t low_ = 14695981039346656037ull;
            std::uint64_t high_ = 7809847782465536322ull;
        };

        /** @brief Mix a complete global endpoint without string formatting. */
        void addAddress(
            StableDigestBuilder &digest,
            const GlobalDeviceAddress &address) noexcept
        {
            digest.addString(address.hostname);
            digest.addScalar(address.numa_node);
            digest.addScalar(address.device_type);
            digest.addScalar(address.device_ordinal);
        }

        /** @brief Mix a complete resolved owner used by a physical migration. */
        void addOwner(
            StableDigestBuilder &digest,
            const MoEExpertOwner &owner) noexcept
        {
            digest.addScalar(owner.layer_idx);
            digest.addScalar(owner.expert_id);
            digest.addScalar(owner.tier_idx);
            digest.addScalar(owner.owner_participant);
            digest.addScalar(owner.device.type);
            digest.addScalar(owner.device.ordinal);
            digest.addScalar(owner.resident);
            digest.addString(owner.tier_name);
            digest.addString(owner.domain_name);
            digest.addScalar(owner.domain_participant_index);
            digest.addScalar(owner.owner_world_rank);
            digest.addScalar(owner.owner_world_rank_known);
            addAddress(digest, owner.address);
        }

        /** @brief Mix the complete dense layered owner table. */
        void addLayeredOwnership(
            StableDigestBuilder &digest,
            const MoELayeredExpertOwnership &ownership)
        {
            digest.addScalar(ownership.layerCount());
            digest.addScalar(ownership.expertCount());
            digest.addScalar(ownership.participantCount());
            for (int layer_idx = 0; layer_idx < ownership.layerCount();
                 ++layer_idx)
            {
                for (int expert_id = 0;
                     expert_id < ownership.expertCount();
                     ++expert_id)
                {
                    digest.addScalar(ownership.owner(layer_idx, expert_id));
                }
            }
        }

        /** @brief Mix every participant and resolved owner in one owner map. */
        void addOwnerMap(
            StableDigestBuilder &digest,
            const MoEExpertOwnerMap &owner_map) noexcept
        {
            digest.addScalar(
                static_cast<std::uint64_t>(owner_map.participants().size()));
            for (const auto &participant : owner_map.participants())
            {
                digest.addScalar(participant.participant_id);
                digest.addScalar(participant.tier_idx);
                digest.addString(participant.tier_name);
                digest.addString(participant.domain_name);
                digest.addScalar(participant.domain_participant_index);
                addAddress(digest, participant.address);
                digest.addScalar(participant.device.type);
                digest.addScalar(participant.device.ordinal);
                digest.addScalar(participant.world_rank);
                digest.addScalar(participant.world_rank_known);
            }

            digest.addScalar(
                static_cast<std::uint64_t>(owner_map.owners().size()));
            for (const auto &owner : owner_map.owners())
                addOwner(digest, owner);
        }

        /** @brief Mix tier policy that affects thermal scoring and capacity. */
        void addPlacementPolicy(
            StableDigestBuilder &digest,
            const MoERoutedExpertPlacementPlan &plan) noexcept
        {
            digest.addScalar(plan.enabled);
            digest.addScalar(plan.topology);
            digest.addScalar(plan.residency_policy);
            digest.addScalar(plan.owner_order);
            digest.addString(plan.continuation_domain);
            digest.addString(plan.base_model_domain);
            digest.addString(plan.shared_expert_domain);

            /*
             * Backend and rank membership decide which physical network lane
             * owns a remote edge.  Hash the declarative domains even when the
             * resolved owner map happens to name the same endpoint devices.
             */
            digest.addScalar(
                static_cast<std::uint64_t>(plan.domains.size()));
            for (const auto &domain : plan.domains)
            {
                digest.addString(domain.name);
                digest.addScalar(domain.scope);
                digest.addScalar(domain.backend);
                digest.addScalar(domain.owner_rank);
                digest.addScalar(domain.routed_compute_policy);
                digest.addScalar(domain.routed_phase_policy);
                digest.addScalar(
                    domain.routed_decode_assignment_policy);
                digest.addScalar(
                    domain.routed_prefill_assignment_policy);
                digest.addScalar(static_cast<std::uint64_t>(
                    domain.participants.size()));
                for (const auto &participant : domain.participants)
                    addAddress(digest, participant);
                digest.addScalar(static_cast<std::uint64_t>(
                    domain.world_ranks.size()));
                for (const int world_rank : domain.world_ranks)
                    digest.addScalar(world_rank);
            }

            digest.addScalar(plan.replica_cache_capacity.has_value());
            if (plan.replica_cache_capacity)
            {
                digest.addScalar(plan.replica_cache_capacity->requested());
                digest.addScalar(plan.replica_cache_capacity->admitted());
            }
            digest.addScalar(
                static_cast<std::uint64_t>(plan.routed_tiers.size()));
            for (const auto &tier : plan.routed_tiers)
            {
                digest.addString(tier.name);
                digest.addString(tier.domain);
                digest.addScalar(tier.priority);
                digest.addScalar(tier.max_experts_per_layer);
                digest.addScalar(
                    static_cast<std::uint64_t>(tier.memory_budget_bytes));
                digest.addScalar(tier.fallback);
                digest.addScalar(static_cast<std::uint64_t>(
                    tier.resolved_live_experts_per_layer.size()));
                for (const int quota :
                     tier.resolved_live_experts_per_layer)
                {
                    digest.addScalar(quota);
                }
            }

            digest.addScalar(
                static_cast<std::uint64_t>(plan.placements.size()));
            for (const auto &placement : plan.placements)
            {
                digest.addScalar(placement.layer);
                digest.addScalar(static_cast<std::uint64_t>(
                    placement.routed_expert_tier.size()));
                for (const int tier_idx : placement.routed_expert_tier)
                    digest.addScalar(tier_idx);
            }
        }

        /** @brief Mix one immutable epoch snapshot without pointer identity. */
        void addSnapshot(
            StableDigestBuilder &digest,
            const MoEOverlayResidencySnapshot &snapshot)
        {
            digest.addScalar(snapshot.epoch);
            addPlacementPolicy(digest, *snapshot.placement_plan);
            addOwnerMap(digest, snapshot.owner_map);
            addLayeredOwnership(digest, snapshot.layered_ownership);
        }

        /** @brief One semantic fingerprint recipe shared by windows and complete plans. */
        void addHistogramWindow(StableDigestBuilder &digest, const DecodeExpertHistogramWindow &window)
        {
            digest.addScalar(window.generation);
            digest.addScalar(window.token_count);
            for (const auto count : window.source_token_counts) digest.addScalar(count);
            digest.addScalar(window.num_layers);
            digest.addScalar(window.num_experts);
            digest.addScalar(static_cast<uint64_t>(window.expert_counts.size()));
            for (const auto count : window.expert_counts) digest.addScalar(count);
            digest.addScalar(static_cast<uint64_t>(window.source_expert_counts.size()));
            for (const auto count : window.source_expert_counts) digest.addScalar(count);
            digest.addScalar(window.transaction_demand != nullptr);
            if (!window.transaction_demand) return;
            const auto &demand = *window.transaction_demand;
            digest.addScalar(demand.topK());
            digest.addScalar(demand.tokenBoundaryLayer());
            for (int layer = 0; layer < window.num_layers; ++layer)
            {
                const auto records = demand.layerTransactions(layer);
                digest.addScalar(static_cast<uint64_t>(records.size()));
                for (size_t batch = 0; batch < records.size(); ++batch)
                {
                    digest.addScalar(records[batch].logical_rows);
                    digest.addScalar(records[batch].phase);
                    const auto routes = demand.routes(layer, batch);
                    for (uint64_t slot = 0; slot < routes.capacity_slots; ++slot)
                        digest.addScalar(routes.expert_ids[slot]);
                }
            }
        }

        /** @brief Stable diagnostic digest retained in a fixed-layout failed vote. */
        std::uint64_t diagnosticFingerprint(
            const std::string &diagnostic) noexcept
        {
            StableDigestBuilder digest;
            digest.addString("MoEOverlayDistributedResidencyDiagnostic/v1");
            digest.addString(diagnostic);
            return digest.finish().low;
        }

        /** @brief Assign an optional user-facing error without duplicating branches. */
        void setError(std::string *error, const std::string &message)
        {
            if (error)
                *error = message;
        }

    } // namespace

    namespace
    {
        /** @brief Hash common execution state and optional root policy evidence. */
        template <typename Fingerprint>
        Fingerprint fingerprintTransaction(
            const MoEOverlayResidencyTransaction &transaction,
            bool include_economy)
        {
            if (!transaction.valid())
            {
                throw std::invalid_argument(
                    "Cannot fingerprint an invalid ExpertOverlay residency transaction");
            }

            StableDigestBuilder digest;
            digest.addString(
                include_economy
                    ? "MoEOverlayResidencyTransaction/v6"
                    : "MoEOverlayResidencyExecutionPlan/v2");
            digest.addScalar(transaction.purpose);
            digest.addScalar(transaction.calibration_sequence);
            digest.addScalar(transaction.expected_epoch);
            digest.addScalar(transaction.histogram_generation);
            addSnapshot(digest, *transaction.previous);
            addSnapshot(digest, *transaction.candidate);

            digest.addScalar(transaction.histogram_window != nullptr);
            if (transaction.histogram_window)
            {
                addHistogramWindow(digest, *transaction.histogram_window);
            }

            digest.addScalar(
                static_cast<std::uint64_t>(transaction.migrations.size()));
            for (const auto &migration : transaction.migrations)
            {
                digest.addScalar(migration.layer_idx);
                digest.addScalar(migration.expert_id);
                digest.addScalar(migration.activation_count);
                digest.addScalar(static_cast<std::uint64_t>(
                    migration.estimated_weight_bytes));
                digest.addScalar(migration.direction);
                digest.addScalar(migration.axis);
                addOwner(digest, migration.source);
                addOwner(digest, migration.destination);
            }

            digest.addScalar(static_cast<std::uint64_t>(
                transaction.migration_cycles.size()));
            for (const auto &cycle : transaction.migration_cycles)
            {
                digest.addScalar(cycle.layer_idx);
                digest.addScalar(static_cast<std::uint64_t>(
                    cycle.migration_indices.size()));
                for (const std::size_t migration_idx :
                     cycle.migration_indices)
                {
                    digest.addScalar(
                        static_cast<std::uint64_t>(migration_idx));
                }
            }

            digest.addScalar(static_cast<std::uint64_t>(
                transaction.shadow_requirements.size()));
            for (const auto &requirement :
                 transaction.shadow_requirements)
            {
                digest.addScalar(requirement.layer_idx);
                digest.addScalar(requirement.tier_idx);
                digest.addScalar(requirement.destination_participant);
                digest.addScalar(
                    static_cast<std::uint64_t>(requirement.slot_count));
            }

            if (include_economy)
            {
                digest.addScalar(transaction.host_admission.has_value());
                if (transaction.host_admission)
                {
                    const auto &admission = *transaction.host_admission;
                    digest.addScalar(admission.authority);
                    digest.addScalar(admission.transaction);
                    digest.addScalar(admission.candidate_epoch);
                    digest.addScalar(admission.cycle_capacity_kind);
                    digest.addScalar(admission.maximum_concurrent_cycles);
                    digest.addScalar(admission.candidate_cycles);
                    digest.addScalar(admission.policy_eligible_cycles);
                    digest.addScalar(admission.policy_eligible_axes.tier_residency);
                    digest.addScalar(
                        admission.policy_eligible_axes.participant_placement);
                    digest.addScalar(admission.policy_eligible_axes.combined);
                    digest.addScalar(admission.admitted_candidate_cycles);
                    digest.addScalar(
                        admission.admitted_candidate_axes.tier_residency);
                    digest.addScalar(
                        admission.admitted_candidate_axes
                            .participant_placement);
                    digest.addScalar(
                        admission.admitted_candidate_axes.combined);
                    digest.addScalar(admission.admitted_physical_cycles);
                    digest.addScalar(
                        admission.admitted_physical_axes.tier_residency);
                    digest.addScalar(
                        admission.admitted_physical_axes
                            .participant_placement);
                    digest.addScalar(
                        admission.admitted_physical_axes.combined);
                    digest.addScalar(
                        admission.individual_policy_rejected_cycles);
                    digest.addScalar(
                        admission.dependent_payoff_rejected_cycles);
                    digest.addScalar(admission.capacity_rejected_cycles);
                    digest.addScalar(
                        admission.participant_axis_budget_rejected_cycles);
                    digest.addScalar(
                        admission.dependent_cohort_candidates);
                    digest.addScalar(
                        admission.dependent_cohort_payoff_rejections);
                    digest.addScalar(admission.physical_cycle_recomposition);
                    digest.addScalar(admission.capacity_bounded);
                    digest.addScalar(admission.policy_bounded);
                }
                digest.addScalar(transaction.economy.enabled);
                digest.addString(
                    transaction.economy.service_profile_identity);
                digest.addString(
                    transaction.economy.migration_profile_identity);
                digest.addScalar(
                    transaction.economy.smoothed_through_generation);
                digest.addScalar(transaction.economy.forecast_fingerprint);
                digest.addScalar(
                    transaction.economy.historical_window_weight);
                digest.addScalar(
                    transaction.economy.current_window_weight);
                digest.addScalar(
                    transaction.economy.payoff_horizon_tokens);
                digest.addScalar(
                    transaction.economy.minimum_net_benefit_ns);
                digest.addScalar(
                    transaction.economy.minimum_residency_generations);
                digest.addScalar(
                    transaction.economy.projected_service_gain_ns);
                digest.addScalar(
                    transaction.economy.projected_transfer_and_repack_ns);
                digest.addScalar(
                    transaction.economy.projected_inference_interference_ns);
                digest.addScalar(
                    transaction.economy.projected_net_benefit_ns);
                digest.addScalar(
                    transaction.economy.payoff_rejected_cycles);
                digest.addScalar(
                    transaction.economy.residency_rejected_cycles);
            }
            return digest.finish<Fingerprint>();
        }
    } // namespace

    MoEOverlayResidencyTransactionFingerprint
    fingerprintMoEOverlayResidencyTransaction(
        const MoEOverlayResidencyTransaction &transaction)
    {
        return fingerprintTransaction<
            MoEOverlayResidencyTransactionFingerprint>(transaction, true);
    }

    MoEOverlayResidencyExecutionFingerprint
    fingerprintMoEOverlayResidencyExecutionPlan(
        const MoEOverlayResidencyTransaction &transaction)
    {
        return fingerprintTransaction<
            MoEOverlayResidencyExecutionFingerprint>(transaction, false);
    }

    bool MoEOverlayDistributedHistogramHeader::valid() const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            num_layers <= 0 || num_experts <= 0 ||
            production_source_count !=
                kExpertHistogramProductionSourceCount ||
            reserved != 0 || counts_fingerprint == 0)
        {
            return false;
        }
        const auto layers = static_cast<std::uint64_t>(num_layers);
        const auto experts = static_cast<std::uint64_t>(num_experts);
        if (layers > std::numeric_limits<std::uint64_t>::max() / experts)
            return false;
        const std::uint64_t entries = layers * experts;
        return entries <=
                   std::numeric_limits<std::uint64_t>::max() /
                       kExpertHistogramProductionSourceCount &&
               expert_count_entries == entries &&
               source_expert_count_entries ==
                   entries * kExpertHistogramProductionSourceCount;
    }

    std::size_t moeOverlayDistributedHistogramWireBytes(
        int num_layers,
        int num_experts,
        const moe_overlay_economy::TransactionDemandCapacity *transactions)
    {
        if (num_layers <= 0 || num_experts <= 0)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay histogram requires positive geometry");
        }
        const auto layers = static_cast<std::size_t>(num_layers);
        const auto experts = static_cast<std::size_t>(num_experts);
        if (layers > std::numeric_limits<std::size_t>::max() / experts)
        {
            throw std::overflow_error(
                "Distributed ExpertOverlay histogram count geometry overflows size_t");
        }
        const std::size_t entries = layers * experts;
        constexpr std::size_t phase_token_bytes =
            kExpertHistogramProductionSourceCount * sizeof(std::uint64_t);
        constexpr std::size_t fixed_bytes =
            MoEOverlayDistributedHistogramHeader::kWireBytes +
            phase_token_bytes;
        constexpr std::size_t count_vectors =
            1 + kExpertHistogramProductionSourceCount;
        if (entries >
            (std::numeric_limits<std::size_t>::max() - fixed_bytes) /
                (count_vectors * sizeof(std::uint64_t)))
        {
            throw std::overflow_error(
                "Distributed ExpertOverlay histogram packet size overflows size_t");
        }
        const auto counts = fixed_bytes + entries * count_vectors * sizeof(std::uint64_t);
        const auto demand = transactions ? DecodeExpertTransactionWindow::maximumWireBytes(
            *transactions, num_layers, num_experts) : 0u;
        if (demand > std::numeric_limits<size_t>::max() - counts)
            throw std::overflow_error("ExpertOverlay histogram receive capacity overflows size_t");
        return counts + demand;
    }

    std::size_t moeOverlayDistributedHistogramWireBytes(const DecodeExpertHistogramWindow &window)
    {
        if (!window.valid()) throw std::invalid_argument("Cannot size an invalid histogram window");
        const auto counts = moeOverlayDistributedHistogramWireBytes(window.num_layers, window.num_experts);
        const auto demand = window.transaction_demand ? window.transaction_demand->wireBytes() : 0u;
        if (demand > std::numeric_limits<size_t>::max() - counts)
            throw std::overflow_error("ExpertOverlay histogram packet size overflows size_t");
        return counts + demand;
    }

    std::uint64_t fingerprintDecodeExpertHistogramWindow(
        const DecodeExpertHistogramWindow &window)
    {
        if (!window.valid())
        {
            throw std::invalid_argument(
                "Cannot fingerprint an invalid ExpertOverlay histogram window");
        }
        StableDigestBuilder digest;
        digest.addString("MoEOverlayHistogramWindow/v3");
        addHistogramWindow(digest, window);
        return digest.finish().low;
    }

    bool encodeMoEOverlayDistributedHistogramWindow(
        const DecodeExpertHistogramWindow &window,
        std::span<std::uint8_t> destination,
        std::string *error)
    {
        if (!window.valid())
        {
            setError(
                error,
                "Cannot encode an invalid ExpertOverlay histogram window");
            return false;
        }
        const std::size_t required =
            moeOverlayDistributedHistogramWireBytes(window);
        if (destination.size() != required)
        {
            setError(
                error,
                "ExpertOverlay histogram destination has the wrong compact packet size");
            return false;
        }

        const MoEOverlayDistributedHistogramHeader header{
            .generation = window.generation,
            .token_count = window.token_count,
            .num_layers = window.num_layers,
            .num_experts = window.num_experts,
            .production_source_count =
                static_cast<std::uint32_t>(
                    kExpertHistogramProductionSourceCount),
            .reserved = 0,
            .expert_count_entries = static_cast<std::uint64_t>(
                window.expert_counts.size()),
            .source_expert_count_entries = static_cast<std::uint64_t>(
                window.source_expert_counts.size()),
            .counts_fingerprint =
                fingerprintDecodeExpertHistogramWindow(window),
            .transaction_bytes = window.transaction_demand ? window.transaction_demand->wireBytes() : 0u,
        };
        if (!header.valid())
        {
            setError(
                error,
                "ExpertOverlay histogram produced an invalid wire header");
            return false;
        }

        std::size_t offset = 0;
        writeLittleEndian(destination, offset, header.magic);
        writeLittleEndian(destination, offset, header.abi_version);
        writeLittleEndian(destination, offset, header.generation);
        writeLittleEndian(destination, offset, header.token_count);
        writeLittleEndian(destination, offset, header.num_layers);
        writeLittleEndian(destination, offset, header.num_experts);
        writeLittleEndian(
            destination, offset, header.production_source_count);
        writeLittleEndian(destination, offset, header.reserved);
        writeLittleEndian(
            destination, offset, header.expert_count_entries);
        writeLittleEndian(
            destination, offset, header.source_expert_count_entries);
        writeLittleEndian(
            destination, offset, header.counts_fingerprint);
        writeLittleEndian(destination, offset, header.transaction_bytes);
        if (offset != MoEOverlayDistributedHistogramHeader::kWireBytes)
        {
            throw std::logic_error(
                "ExpertOverlay histogram header wire-size constant is stale");
        }
        for (const std::uint64_t count : window.source_token_counts)
            writeLittleEndian(destination, offset, count);
        for (const std::uint64_t count : window.expert_counts)
            writeLittleEndian(destination, offset, count);
        for (const std::uint64_t count : window.source_expert_counts)
            writeLittleEndian(destination, offset, count);
        if (window.transaction_demand)
        {
            window.transaction_demand->encodeWire(destination.subspan(offset, header.transaction_bytes));
            offset += header.transaction_bytes;
        }
        if (offset != destination.size())
        {
            throw std::logic_error(
                "ExpertOverlay histogram encoder did not fill its exact packet");
        }
        if (error)
            error->clear();
        return true;
    }

    bool decodeMoEOverlayDistributedHistogramWindow(
        std::span<const std::uint8_t> packet,
        int expected_layers,
        int expected_experts,
        DecodeExpertHistogramWindow *window,
        std::string *error,
        const ExpertHistogramTransactionConfig *transactions)
    {
        if (!window || expected_layers <= 0 || expected_experts <= 0)
        {
            setError(
                error,
                "ExpertOverlay histogram decoder requires output and positive model geometry");
            return false;
        }
        // Reuse count-vector capacity but retire any previous sample identity.
        window->generation = 0;
        window->token_count = 0;
        window->source_token_counts.fill(0);
        window->num_layers = 0;
        window->num_experts = 0;
        window->expert_counts.clear();
        window->source_expert_counts.clear();
        window->transaction_demand.reset();
        const std::size_t minimum_size =
            moeOverlayDistributedHistogramWireBytes(
                expected_layers, expected_experts);
        const std::size_t maximum_size = moeOverlayDistributedHistogramWireBytes(
            expected_layers, expected_experts, transactions ? &transactions->capacity : nullptr);
        if (packet.size() < minimum_size || packet.size() > maximum_size)
        {
            setError(
                error,
                "ExpertOverlay histogram packet is outside admitted receive size");
            return false;
        }

        std::size_t offset = 0;
        MoEOverlayDistributedHistogramHeader header;
        header.magic = readLittleEndian<std::uint32_t>(packet, offset);
        header.abi_version =
            readLittleEndian<std::uint32_t>(packet, offset);
        header.generation =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.token_count =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.num_layers =
            readLittleEndian<std::int32_t>(packet, offset);
        header.num_experts =
            readLittleEndian<std::int32_t>(packet, offset);
        header.production_source_count =
            readLittleEndian<std::uint32_t>(packet, offset);
        header.reserved =
            readLittleEndian<std::uint32_t>(packet, offset);
        header.expert_count_entries =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.source_expert_count_entries =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.counts_fingerprint =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.transaction_bytes = readLittleEndian<uint64_t>(packet, offset);
        if (!header.valid() || header.num_layers != expected_layers ||
            header.num_experts != expected_experts)
        {
            setError(
                error,
                "ExpertOverlay histogram header or model geometry is invalid");
            return false;
        }
        if (header.transaction_bytes != packet.size() - minimum_size ||
            (header.transaction_bytes != 0u) != (transactions != nullptr))
        {
            setError(error, "ExpertOverlay transaction evidence is missing, unadmitted, or has the wrong extent");
            return false;
        }

        window->generation = header.generation;
        window->token_count = header.token_count;
        for (auto &count : window->source_token_counts)
            count = readLittleEndian<std::uint64_t>(packet, offset);
        window->num_layers = header.num_layers;
        window->num_experts = header.num_experts;
        window->expert_counts.resize(
            static_cast<std::size_t>(header.expert_count_entries));
        for (auto &count : window->expert_counts)
            count = readLittleEndian<std::uint64_t>(packet, offset);
        window->source_expert_counts.resize(
            static_cast<std::size_t>(
                header.source_expert_count_entries));
        for (auto &count : window->source_expert_counts)
            count = readLittleEndian<std::uint64_t>(packet, offset);
        if (transactions)
        {
            try
            {
                window->transaction_demand = DecodeExpertTransactionWindow::decodeWire(
                    *transactions, *window, packet.subspan(offset, header.transaction_bytes));
                offset += header.transaction_bytes;
            }
            catch (const std::exception &failure)
            {
                *window = {};
                setError(error, std::string("ExpertOverlay transaction authentication failed: ") + failure.what());
                return false;
            }
        }
        if (offset != packet.size() || !window->valid() ||
            fingerprintDecodeExpertHistogramWindow(*window) !=
                header.counts_fingerprint)
        {
            *window = {};
            setError(
                error,
                "ExpertOverlay histogram packet failed count authentication");
            return false;
        }

        if (error)
            error->clear();
        return true;
    }

    namespace
    {
        constexpr std::size_t kAuthoritativeEntryWireBytes =
            sizeof(std::int32_t) + sizeof(std::int32_t) +
            sizeof(std::uint32_t) + sizeof(std::uint64_t) +
            sizeof(std::uint64_t);

        /** @brief Hash one canonical dense proposal independently of its packet. */
        std::uint64_t fingerprintAuthoritativeResidencyPlan(
            const MoEOverlayAuthoritativeResidencyPlan &plan)
        {
            if (!plan.valid())
            {
                throw std::invalid_argument(
                    "Cannot fingerprint an invalid authoritative ExpertOverlay plan");
            }
            StableDigestBuilder digest;
            digest.addString("MoEOverlayAuthoritativeResidencyPlan/v1");
            digest.addScalar(plan.expected_epoch);
            digest.addScalar(plan.num_layers);
            digest.addScalar(plan.num_experts);
            digest.addScalar(
                fingerprintDecodeExpertHistogramWindow(
                    *plan.histogram_window));
            digest.addScalar(
                static_cast<std::uint64_t>(plan.entries.size()));
            for (const auto &entry : plan.entries)
            {
                digest.addScalar(entry.candidate_tier_idx);
                digest.addScalar(entry.candidate_owner_participant);
                digest.addScalar(entry.changed);
                digest.addScalar(entry.axis);
                digest.addScalar(entry.activation_count);
                digest.addScalar(static_cast<std::uint64_t>(
                    entry.estimated_weight_bytes));
            }
            return digest.finish().low;
        }
    } // namespace

    bool MoEOverlayDistributedResidencyProposal::valid() const noexcept
    {
        return plan.valid() && execution_fingerprint.valid() &&
               policy_fingerprint.valid();
    }

    bool MoEOverlayDistributedResidencyProposalHeader::valid() const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            expected_epoch == 0 ||
            expected_epoch == std::numeric_limits<std::uint64_t>::max() ||
            candidate_epoch != expected_epoch + 1u || num_layers <= 0 ||
            num_experts <= 0 || !execution_fingerprint.valid() ||
            !policy_fingerprint.valid() || plan_fingerprint == 0 ||
            reserved != 0)
        {
            return false;
        }
        const auto layers = static_cast<std::uint64_t>(num_layers);
        const auto experts = static_cast<std::uint64_t>(num_experts);
        return layers <=
                   std::numeric_limits<std::uint64_t>::max() / experts &&
               entry_count == layers * experts &&
               changed_entry_count <= entry_count;
    }

    MoEOverlayDistributedResidencyProposal
    makeMoEOverlayDistributedResidencyProposal(
        MoEOverlayAuthoritativeResidencyPlan plan,
        const MoEOverlayResidencyTransaction &transaction)
    {
        if (!plan.valid() || !transaction.valid() ||
            transaction.purpose !=
                MoEOverlayResidencyTransactionPurpose::PlacementChange ||
            transaction.expected_epoch != plan.expected_epoch ||
            !transaction.histogram_window ||
            fingerprintDecodeExpertHistogramWindow(
                *transaction.histogram_window) !=
                fingerprintDecodeExpertHistogramWindow(
                    *plan.histogram_window))
        {
            throw std::invalid_argument(
                "Authoritative ExpertOverlay plan does not match its root transaction");
        }
        const auto changed_entries = static_cast<std::size_t>(std::count_if(
            plan.entries.begin(),
            plan.entries.end(),
            [](const auto &entry) { return entry.changed; }));
        if (changed_entries != transaction.migrations.size())
        {
            throw std::invalid_argument(
                "Authoritative ExpertOverlay plan has a different movement cardinality from its root transaction");
        }

        MoEOverlayDistributedResidencyProposal proposal{
            .plan = std::move(plan),
            .execution_fingerprint =
                fingerprintMoEOverlayResidencyExecutionPlan(transaction),
            .policy_fingerprint =
                fingerprintMoEOverlayResidencyTransaction(transaction),
        };
        if (!proposal.valid())
        {
            throw std::logic_error(
                "Root ExpertOverlay transaction produced an invalid distributed proposal");
        }
        return proposal;
    }

    std::size_t moeOverlayDistributedResidencyProposalWireBytes(
        int num_layers,
        int num_experts,
        const moe_overlay_economy::TransactionDemandCapacity *transactions)
    {
        if (num_layers <= 0 || num_experts <= 0)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay proposal requires positive geometry");
        }
        const auto layers = static_cast<std::size_t>(num_layers);
        const auto experts = static_cast<std::size_t>(num_experts);
        if (layers > std::numeric_limits<std::size_t>::max() / experts)
        {
            throw std::overflow_error(
                "Distributed ExpertOverlay proposal geometry overflows size_t");
        }
        const std::size_t entries = layers * experts;
        const std::size_t histogram_bytes =
            moeOverlayDistributedHistogramWireBytes(
                num_layers, num_experts, transactions);
        constexpr std::size_t header_bytes =
            MoEOverlayDistributedResidencyProposalHeader::kWireBytes;
        if (histogram_bytes > std::numeric_limits<std::size_t>::max() - header_bytes ||
            entries >
            (std::numeric_limits<std::size_t>::max() - header_bytes -
             histogram_bytes) /
                kAuthoritativeEntryWireBytes)
        {
            throw std::overflow_error(
                "Distributed ExpertOverlay proposal packet size overflows size_t");
        }
        return header_bytes + histogram_bytes +
               entries * kAuthoritativeEntryWireBytes;
    }

    std::size_t moeOverlayDistributedResidencyProposalWireBytes(
        const MoEOverlayDistributedResidencyProposal &proposal)
    {
        if (!proposal.valid()) throw std::invalid_argument("Cannot size an invalid residency proposal");
        const auto fixed = moeOverlayDistributedResidencyProposalWireBytes(
            proposal.plan.num_layers, proposal.plan.num_experts);
        const auto &demand = proposal.plan.histogram_window->transaction_demand;
        const auto extra = demand ? demand->wireBytes() : 0u;
        if (extra > std::numeric_limits<size_t>::max() - fixed)
            throw std::overflow_error("Residency proposal compact size overflows size_t");
        return fixed + extra;
    }

    bool encodeMoEOverlayDistributedResidencyProposal(
        const MoEOverlayDistributedResidencyProposal &proposal,
        std::span<std::uint8_t> destination,
        std::string *error)
    {
        if (!proposal.valid())
        {
            setError(
                error,
                "Cannot encode an invalid distributed ExpertOverlay proposal");
            return false;
        }
        const std::size_t required =
            moeOverlayDistributedResidencyProposalWireBytes(proposal);
        if (destination.size() != required)
        {
            setError(
                error,
                "ExpertOverlay proposal destination has the wrong fixed packet size");
            return false;
        }

        const auto changed_entries = static_cast<std::uint64_t>(std::count_if(
            proposal.plan.entries.begin(),
            proposal.plan.entries.end(),
            [](const auto &entry) { return entry.changed; }));
        const MoEOverlayDistributedResidencyProposalHeader header{
            .expected_epoch = proposal.plan.expected_epoch,
            .candidate_epoch = proposal.plan.expected_epoch + 1u,
            .num_layers = proposal.plan.num_layers,
            .num_experts = proposal.plan.num_experts,
            .entry_count = static_cast<std::uint64_t>(
                proposal.plan.entries.size()),
            .changed_entry_count = changed_entries,
            .execution_fingerprint = proposal.execution_fingerprint,
            .policy_fingerprint = proposal.policy_fingerprint,
            .plan_fingerprint =
                fingerprintAuthoritativeResidencyPlan(proposal.plan),
            .reserved = 0,
        };
        if (!header.valid())
        {
            setError(
                error,
                "ExpertOverlay proposal produced an invalid wire header");
            return false;
        }

        std::size_t offset = 0;
        writeLittleEndian(destination, offset, header.magic);
        writeLittleEndian(destination, offset, header.abi_version);
        writeLittleEndian(destination, offset, header.expected_epoch);
        writeLittleEndian(destination, offset, header.candidate_epoch);
        writeLittleEndian(destination, offset, header.num_layers);
        writeLittleEndian(destination, offset, header.num_experts);
        writeLittleEndian(destination, offset, header.entry_count);
        writeLittleEndian(destination, offset, header.changed_entry_count);
        writeLittleEndian(
            destination, offset, header.execution_fingerprint.low);
        writeLittleEndian(
            destination, offset, header.execution_fingerprint.high);
        writeLittleEndian(
            destination, offset, header.policy_fingerprint.low);
        writeLittleEndian(
            destination, offset, header.policy_fingerprint.high);
        writeLittleEndian(destination, offset, header.plan_fingerprint);
        writeLittleEndian(destination, offset, header.reserved);
        if (offset !=
            MoEOverlayDistributedResidencyProposalHeader::kWireBytes)
        {
            throw std::logic_error(
                "ExpertOverlay proposal header wire-size constant is stale");
        }

        const std::size_t histogram_bytes =
            moeOverlayDistributedHistogramWireBytes(*proposal.plan.histogram_window);
        if (!encodeMoEOverlayDistributedHistogramWindow(
                *proposal.plan.histogram_window,
                destination.subspan(offset, histogram_bytes),
                error))
        {
            return false;
        }
        offset += histogram_bytes;
        for (const auto &entry : proposal.plan.entries)
        {
            writeLittleEndian(
                destination, offset, entry.candidate_tier_idx);
            writeLittleEndian(
                destination,
                offset,
                entry.candidate_owner_participant);
            const std::uint32_t axis_code =
                entry.changed
                    ? static_cast<std::uint32_t>(entry.axis) + 1u
                    : 0u;
            writeLittleEndian(destination, offset, axis_code);
            writeLittleEndian(
                destination, offset, entry.activation_count);
            writeLittleEndian(
                destination,
                offset,
                static_cast<std::uint64_t>(
                    entry.estimated_weight_bytes));
        }
        if (offset != destination.size())
        {
            throw std::logic_error(
                "ExpertOverlay proposal encoder did not fill its exact packet");
        }
        if (error)
            error->clear();
        return true;
    }

    bool decodeMoEOverlayDistributedResidencyProposal(
        std::span<const std::uint8_t> packet,
        int expected_layers,
        int expected_experts,
        MoEOverlayDistributedResidencyProposal *proposal,
        std::string *error,
        const ExpertHistogramTransactionConfig *transactions)
    {
        if (!proposal || expected_layers <= 0 || expected_experts <= 0)
        {
            setError(
                error,
                "ExpertOverlay proposal decoder requires output and positive model geometry");
            return false;
        }
        *proposal = {};
        const std::size_t minimum_size =
            moeOverlayDistributedResidencyProposalWireBytes(
                expected_layers, expected_experts);
        const auto maximum_size = moeOverlayDistributedResidencyProposalWireBytes(
            expected_layers, expected_experts, transactions ? &transactions->capacity : nullptr);
        if (packet.size() < minimum_size || packet.size() > maximum_size)
        {
            setError(
                error,
                "ExpertOverlay proposal packet is outside admitted receive size");
            return false;
        }

        std::size_t offset = 0;
        MoEOverlayDistributedResidencyProposalHeader header;
        header.magic = readLittleEndian<std::uint32_t>(packet, offset);
        header.abi_version =
            readLittleEndian<std::uint32_t>(packet, offset);
        header.expected_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.candidate_epoch =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.num_layers =
            readLittleEndian<std::int32_t>(packet, offset);
        header.num_experts =
            readLittleEndian<std::int32_t>(packet, offset);
        header.entry_count =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.changed_entry_count =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.execution_fingerprint.low =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.execution_fingerprint.high =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.policy_fingerprint.low =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.policy_fingerprint.high =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.plan_fingerprint =
            readLittleEndian<std::uint64_t>(packet, offset);
        header.reserved =
            readLittleEndian<std::uint64_t>(packet, offset);
        if (!header.valid() || header.num_layers != expected_layers ||
            header.num_experts != expected_experts)
        {
            setError(
                error,
                "ExpertOverlay proposal header or model geometry is invalid");
            return false;
        }

        const std::size_t histogram_bytes =
            moeOverlayDistributedHistogramWireBytes(
                expected_layers, expected_experts) + packet.size() - minimum_size;
        auto histogram =
            std::make_shared<DecodeExpertHistogramWindow>();
        if (!decodeMoEOverlayDistributedHistogramWindow(
                packet.subspan(offset, histogram_bytes),
                expected_layers,
                expected_experts,
                histogram.get(),
                error, transactions))
        {
            return false;
        }
        offset += histogram_bytes;

        MoEOverlayAuthoritativeResidencyPlan plan{
            .expected_epoch = header.expected_epoch,
            .num_layers = header.num_layers,
            .num_experts = header.num_experts,
            .histogram_window = std::move(histogram),
        };
        plan.entries.resize(static_cast<std::size_t>(header.entry_count));
        std::uint64_t changed_entries = 0;
        for (auto &entry : plan.entries)
        {
            entry.candidate_tier_idx =
                readLittleEndian<std::int32_t>(packet, offset);
            entry.candidate_owner_participant =
                readLittleEndian<std::int32_t>(packet, offset);
            const std::uint32_t axis_code =
                readLittleEndian<std::uint32_t>(packet, offset);
            if (axis_code > 3u)
            {
                setError(
                    error,
                    "ExpertOverlay proposal contains an invalid movement-axis code");
                return false;
            }
            entry.changed = axis_code != 0u;
            entry.axis = entry.changed
                             ? static_cast<MoEOptimizationMovementAxis>(
                                   axis_code - 1u)
                             : MoEOptimizationMovementAxis::TierResidency;
            entry.activation_count =
                readLittleEndian<std::uint64_t>(packet, offset);
            const std::uint64_t weight_bytes =
                readLittleEndian<std::uint64_t>(packet, offset);
            if (weight_bytes >
                static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max()))
            {
                setError(
                    error,
                    "ExpertOverlay proposal weight size exceeds local size_t");
                return false;
            }
            entry.estimated_weight_bytes =
                static_cast<std::size_t>(weight_bytes);
            changed_entries += entry.changed ? 1u : 0u;
        }
        if (offset != packet.size() ||
            changed_entries != header.changed_entry_count || !plan.valid() ||
            fingerprintAuthoritativeResidencyPlan(plan) !=
                header.plan_fingerprint)
        {
            setError(
                error,
                "ExpertOverlay proposal packet failed plan authentication");
            return false;
        }

        proposal->plan = std::move(plan);
        proposal->execution_fingerprint =
            header.execution_fingerprint;
        proposal->policy_fingerprint = header.policy_fingerprint;
        if (!proposal->valid())
        {
            *proposal = {};
            setError(
                error,
                "ExpertOverlay proposal packet decoded to an invalid proposal");
            return false;
        }
        if (error)
            error->clear();
        return true;
    }

    bool MoEOverlayDistributedResidencyWaveIdentity::valid() const noexcept
    {
        return magic == kMagic && abi_version == kABIVersion &&
               expected_epoch > 0 &&
               expected_epoch < std::numeric_limits<std::uint64_t>::max() &&
               candidate_epoch == expected_epoch + 1 &&
               migration_count > 0 && cycle_count > 0 &&
               execution_fingerprint.valid();
    }

    MoEOverlayDistributedResidencyWaveIdentity
    makeMoEOverlayDistributedResidencyWaveIdentity(
        const MoEOverlayResidencyTransaction &transaction)
    {
        if (!transaction.valid() || transaction.empty())
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay residency requires one valid non-empty transaction");
        }
        if (transaction.migrations.size() >
                std::numeric_limits<std::uint64_t>::max() ||
            transaction.migration_cycles.size() >
                std::numeric_limits<std::uint64_t>::max())
        {
            throw std::overflow_error(
                "ExpertOverlay transaction cardinality exceeds the distributed wire ABI");
        }

        MoEOverlayDistributedResidencyWaveIdentity identity{
            .expected_epoch = transaction.expected_epoch,
            .candidate_epoch = transaction.candidate->epoch,
            .histogram_generation = transaction.histogram_generation,
            .migration_count = static_cast<std::uint64_t>(
                transaction.migrations.size()),
            .cycle_count = static_cast<std::uint64_t>(
                transaction.migration_cycles.size()),
            .execution_fingerprint =
                fingerprintMoEOverlayResidencyExecutionPlan(transaction),
        };
        if (!identity.valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay transaction cannot form a valid distributed wave identity");
        }
        return identity;
    }

    bool MoEOverlayDistributedResidencyVote::valid(int world_size) const noexcept
    {
        if (magic != kMagic || abi_version != kABIVersion ||
            !identity.valid() || world_size <= 0 || world_rank < 0 ||
            world_rank >= world_size)
        {
            return false;
        }

        const bool phase_valid =
            phase == MoEOverlayDistributedResidencyVotePhase::Reserved ||
            phase == MoEOverlayDistributedResidencyVotePhase::Staged ||
            phase ==
                MoEOverlayDistributedResidencyVotePhase::InactivePrepared ||
            phase ==
                MoEOverlayDistributedResidencyVotePhase::RuntimePublished ||
            phase ==
                MoEOverlayDistributedResidencyVotePhase::
                    RetirementAdmissionReady ||
            phase ==
                MoEOverlayDistributedResidencyVotePhase::LeaseDrained;
        if (!phase_valid)
            return false;

        switch (decision)
        {
        case MoEOverlayDistributedResidencyVoteDecision::Ready:
            return error_code == 0 && detail_fingerprint == 0;
        case MoEOverlayDistributedResidencyVoteDecision::Failed:
            return error_code > 0 && detail_fingerprint != 0;
        case MoEOverlayDistributedResidencyVoteDecision::Deferred:
            return (phase ==
                        MoEOverlayDistributedResidencyVotePhase::Reserved ||
                    phase ==
                        MoEOverlayDistributedResidencyVotePhase::Staged) &&
                   error_code == 0 && detail_fingerprint == 0;
        case MoEOverlayDistributedResidencyVoteDecision::Waiting:
            return (phase ==
                        MoEOverlayDistributedResidencyVotePhase::
                            RetirementAdmissionReady ||
                    phase ==
                        MoEOverlayDistributedResidencyVotePhase::LeaseDrained) &&
                   error_code == 0 && detail_fingerprint == 0;
        }
        return false;
    }

    MoEOverlayDistributedResidencyProtocol::
        MoEOverlayDistributedResidencyProtocol(Config config)
        : config_(std::move(config))
    {
        if (!config_.identity.valid())
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay protocol requires a valid wave identity");
        }
        if (config_.world_size <= 0 || config_.local_world_rank < 0 ||
            config_.local_world_rank >= config_.world_size)
        {
            throw std::invalid_argument(
                "Distributed ExpertOverlay protocol has invalid communicator membership");
        }
    }

    MoEOverlayDistributedResidencyVotePhase
    MoEOverlayDistributedResidencyProtocol::expectedPhase() const
    {
        switch (state_)
        {
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingLocalReservation:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingReservationConsensus:
            return MoEOverlayDistributedResidencyVotePhase::Reserved;
        case MoEOverlayDistributedResidencyProtocolState::AwaitingLocalStage:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingStageConsensus:
            return MoEOverlayDistributedResidencyVotePhase::Staged;
        case MoEOverlayDistributedResidencyProtocolState::AwaitingLocalPrepare:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingPrepareConsensus:
            return MoEOverlayDistributedResidencyVotePhase::InactivePrepared;
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingLocalPublication:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingPublicationConsensus:
            return MoEOverlayDistributedResidencyVotePhase::RuntimePublished;
        case MoEOverlayDistributedResidencyProtocolState::Published:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingRetirementAdmissionConsensus:
            return MoEOverlayDistributedResidencyVotePhase::
                RetirementAdmissionReady;
        case MoEOverlayDistributedResidencyProtocolState::
            RetirementAdmissionClosed:
        case MoEOverlayDistributedResidencyProtocolState::
            AwaitingRetirementConsensus:
            return MoEOverlayDistributedResidencyVotePhase::LeaseDrained;
        default:
            throw std::logic_error(
                "Distributed ExpertOverlay protocol has no vote due in its current state");
        }
    }

    MoEOverlayDistributedResidencyVote
    MoEOverlayDistributedResidencyProtocol::makeLocalVote(
        MoEOverlayDistributedResidencyVoteDecision decision,
        int error_code,
        const std::string &diagnostic)
    {
        const bool awaiting_reservation =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingLocalReservation;
        const bool awaiting_stage =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingLocalStage;
        const bool awaiting_prepare =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingLocalPrepare;
        const bool awaiting_publication =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingLocalPublication;
        const bool awaiting_retirement_admission =
            state_ ==
            MoEOverlayDistributedResidencyProtocolState::Published;
        const bool awaiting_retirement =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          RetirementAdmissionClosed;
        if (!awaiting_reservation && !awaiting_stage && !awaiting_prepare &&
            !awaiting_publication && !awaiting_retirement_admission &&
            !awaiting_retirement)
        {
            throw std::logic_error(
                "Distributed ExpertOverlay protocol is not awaiting a local vote");
        }

        switch (decision)
        {
        case MoEOverlayDistributedResidencyVoteDecision::Ready:
            if (error_code != 0 || !diagnostic.empty())
            {
                throw std::invalid_argument(
                    "Ready ExpertOverlay vote cannot carry failure diagnostics");
            }
            break;
        case MoEOverlayDistributedResidencyVoteDecision::Failed:
            if (error_code <= 0 || diagnostic.empty())
            {
                throw std::invalid_argument(
                    "Failed ExpertOverlay vote requires a positive code and precise diagnostic");
            }
            break;
        case MoEOverlayDistributedResidencyVoteDecision::Deferred:
            if ((!awaiting_reservation && !awaiting_stage) ||
                error_code != 0 || !diagnostic.empty())
            {
                throw std::invalid_argument(
                    "Deferred ExpertOverlay vote is valid only for reservation/staging and carries no failure diagnostic");
            }
            break;
        case MoEOverlayDistributedResidencyVoteDecision::Waiting:
            if ((!awaiting_retirement_admission && !awaiting_retirement) ||
                error_code != 0 || !diagnostic.empty())
            {
                throw std::invalid_argument(
                    "Waiting ExpertOverlay vote is valid only during an old-epoch retirement grace period and carries no failure diagnostic");
            }
            break;
        default:
            throw std::invalid_argument(
                "ExpertOverlay vote decision is outside the wire ABI");
        }

        MoEOverlayDistributedResidencyVote vote{
            .identity = config_.identity,
            .phase = expectedPhase(),
            .decision = decision,
            .world_rank = config_.local_world_rank,
            .error_code = error_code,
            .detail_fingerprint =
                decision ==
                        MoEOverlayDistributedResidencyVoteDecision::Failed
                    ? diagnosticFingerprint(diagnostic)
                    : 0,
        };
        if (!vote.valid(config_.world_size))
        {
            throw std::logic_error(
                "ExpertOverlay protocol constructed an invalid local vote");
        }

        local_vote_ = vote;
        state_ = awaiting_reservation
                     ? MoEOverlayDistributedResidencyProtocolState::
                           AwaitingReservationConsensus
                     : (awaiting_stage
                            ? MoEOverlayDistributedResidencyProtocolState::
                                  AwaitingStageConsensus
                            : (awaiting_prepare
                                   ? MoEOverlayDistributedResidencyProtocolState::
                                         AwaitingPrepareConsensus
                                   : (awaiting_publication
                                          ? MoEOverlayDistributedResidencyProtocolState::
                                                AwaitingPublicationConsensus
                                          : (awaiting_retirement_admission
                                                 ? MoEOverlayDistributedResidencyProtocolState::
                                                       AwaitingRetirementAdmissionConsensus
                                                 : MoEOverlayDistributedResidencyProtocolState::
                                                       AwaitingRetirementConsensus))));
        return vote;
    }

    void MoEOverlayDistributedResidencyProtocol::fail(
        MoEOverlayDistributedResidencyFailure failure,
        std::string message,
        std::string *error)
    {
        failure_ = failure;
        failure_message_ = std::move(message);
        state_ = MoEOverlayDistributedResidencyProtocolState::Failed;
        setError(error, failure_message_);
    }

    bool MoEOverlayDistributedResidencyProtocol::acceptConsensus(
        const std::vector<MoEOverlayDistributedResidencyVote> &votes,
        std::string *error)
    {
        const bool awaiting_reservation =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingReservationConsensus;
        const bool awaiting_stage =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingStageConsensus;
        const bool awaiting_prepare =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingPrepareConsensus;
        const bool awaiting_publication =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingPublicationConsensus;
        const bool awaiting_retirement_admission =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingRetirementAdmissionConsensus;
        const bool awaiting_retirement =
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingRetirementConsensus;
        if (!awaiting_reservation && !awaiting_stage && !awaiting_prepare &&
            !awaiting_publication && !awaiting_retirement_admission &&
            !awaiting_retirement)
        {
            fail(
                {.phase =
                     MoEOverlayDistributedResidencyVotePhase::Reserved,
                 .world_rank = config_.local_world_rank,
                 .error_code = 1001},
                "Distributed ExpertOverlay consensus arrived in an invalid protocol state",
                error);
            return false;
        }

        const auto phase = expectedPhase();
        if (votes.size() != static_cast<std::size_t>(config_.world_size))
        {
            fail(
                {.phase = phase,
                 .world_rank = config_.local_world_rank,
                 .error_code = 1002},
                "Distributed ExpertOverlay consensus did not contain exactly one vote per rank",
                error);
            return false;
        }

        std::vector<bool> rank_seen(
            static_cast<std::size_t>(config_.world_size), false);
        std::optional<MoEOverlayDistributedResidencyFailure> first_failure;
        bool any_deferred = false;
        bool any_waiting = false;
        bool local_vote_seen = false;
        for (const auto &vote : votes)
        {
            if (!vote.valid(config_.world_size))
            {
                fail(
                    {.phase = phase,
                     .world_rank = vote.world_rank,
                     .error_code = 1003},
                    "Distributed ExpertOverlay consensus contains a malformed vote",
                    error);
                return false;
            }
            if (vote.identity != config_.identity)
            {
                fail(
                    {.phase = phase,
                     .world_rank = vote.world_rank,
                     .error_code = 1004},
                    "Distributed ExpertOverlay ranks disagree on transaction identity",
                    error);
                return false;
            }
            if (vote.phase != phase)
            {
                fail(
                    {.phase = phase,
                     .world_rank = vote.world_rank,
                     .error_code = 1005},
                    "Distributed ExpertOverlay ranks entered different vote phases",
                    error);
                return false;
            }

            const auto rank_idx = static_cast<std::size_t>(vote.world_rank);
            if (rank_seen[rank_idx])
            {
                fail(
                    {.phase = phase,
                     .world_rank = vote.world_rank,
                     .error_code = 1006},
                    "Distributed ExpertOverlay consensus contains a duplicate rank vote",
                    error);
                return false;
            }
            rank_seen[rank_idx] = true;

            if (vote.world_rank == config_.local_world_rank)
            {
                local_vote_seen = local_vote_.has_value() &&
                                  vote == *local_vote_;
            }
            if (vote.decision ==
                MoEOverlayDistributedResidencyVoteDecision::Failed)
            {
                MoEOverlayDistributedResidencyFailure candidate{
                    .phase = phase,
                    .world_rank = vote.world_rank,
                    .error_code = vote.error_code,
                    .detail_fingerprint = vote.detail_fingerprint,
                };
                if (!first_failure ||
                    candidate.world_rank < first_failure->world_rank)
                {
                    first_failure = candidate;
                }
            }
            else if (vote.decision ==
                     MoEOverlayDistributedResidencyVoteDecision::Deferred)
            {
                any_deferred = true;
            }
            else if (vote.decision ==
                     MoEOverlayDistributedResidencyVoteDecision::Waiting)
            {
                any_waiting = true;
            }
        }

        if (!local_vote_seen)
        {
            fail(
                {.phase = phase,
                 .world_rank = config_.local_world_rank,
                 .error_code = 1007},
                "Distributed ExpertOverlay consensus omitted or altered the local rank vote",
                error);
            return false;
        }

        if (first_failure)
        {
            std::ostringstream message;
            const char *phase_name =
                phase == MoEOverlayDistributedResidencyVotePhase::Reserved
                    ? "reservation"
                    : (phase ==
                               MoEOverlayDistributedResidencyVotePhase::Staged
                           ? "staging"
                           : (phase ==
                                      MoEOverlayDistributedResidencyVotePhase::
                                          InactivePrepared
                                  ? "inactive-bank preparation"
                                  : (phase ==
                                             MoEOverlayDistributedResidencyVotePhase::
                                                 RuntimePublished
                                         ? "runtime publication"
                                         : (phase ==
                                                    MoEOverlayDistributedResidencyVotePhase::
                                                        RetirementAdmissionReady
                                                ? "old-epoch retirement admission"
                                                : "old-epoch lease drain"))));
            message << "Distributed ExpertOverlay " << phase_name
                    << " failed on world rank "
                    << first_failure->world_rank << " with code "
                    << first_failure->error_code << " (detail fingerprint "
                    << first_failure->detail_fingerprint << ')';
            fail(*first_failure, message.str(), error);
            return false;
        }

        if (any_deferred)
        {
            local_vote_.reset();
            state_ =
                MoEOverlayDistributedResidencyProtocolState::Deferred;
            if (error)
                error->clear();
            return false;
        }

        if (any_waiting)
        {
            /* A waiting grace-period generation is retryable. Return to the
             * exact source state for this phase so every rank enters the next
             * ordered collective generation with the same phase identity. */
            local_vote_.reset();
            state_ = awaiting_retirement_admission
                         ? MoEOverlayDistributedResidencyProtocolState::Published
                         : MoEOverlayDistributedResidencyProtocolState::
                               RetirementAdmissionClosed;
            if (error)
                error->clear();
            return false;
        }

        local_vote_.reset();
        state_ = awaiting_reservation
                     ? MoEOverlayDistributedResidencyProtocolState::
                           AwaitingLocalStage
                     : (awaiting_stage
                            ? MoEOverlayDistributedResidencyProtocolState::
                                  AwaitingLocalPrepare
                            : (awaiting_prepare
                                   ? MoEOverlayDistributedResidencyProtocolState::
                                         AwaitingLocalPublication
                                   : (awaiting_publication
                                          ? MoEOverlayDistributedResidencyProtocolState::
                                                ReadyForAuthorityPublication
                                          : (awaiting_retirement_admission
                                                 ? MoEOverlayDistributedResidencyProtocolState::
                                                       ReadyToCloseRetirementAdmission
                                                 : MoEOverlayDistributedResidencyProtocolState::
                                                       ReadyToRetire))));
        if (error)
            error->clear();
        return true;
    }

    void MoEOverlayDistributedResidencyProtocol::markAuthorityPublished()
    {
        if (state_ !=
            MoEOverlayDistributedResidencyProtocolState::
                ReadyForAuthorityPublication)
        {
            throw std::logic_error(
                "Distributed ExpertOverlay epoch cannot become authoritative before unanimous runtime publication");
        }
        state_ = MoEOverlayDistributedResidencyProtocolState::Published;
    }

    void MoEOverlayDistributedResidencyProtocol::
        markRetirementAdmissionClosed()
    {
        if (state_ != MoEOverlayDistributedResidencyProtocolState::
                          ReadyToCloseRetirementAdmission)
        {
            throw std::logic_error(
                "Distributed ExpertOverlay old-epoch admission cannot close before unanimous retirement quiescence");
        }
        state_ = MoEOverlayDistributedResidencyProtocolState::
            RetirementAdmissionClosed;
    }

    void MoEOverlayDistributedResidencyProtocol::markRetired()
    {
        if (state_ !=
            MoEOverlayDistributedResidencyProtocolState::ReadyToRetire)
        {
            throw std::logic_error(
                "Distributed ExpertOverlay old epoch cannot retire before unanimous lease drainage");
        }
        state_ = MoEOverlayDistributedResidencyProtocolState::Retired;
    }

    bool MoEOverlayDistributedResidencyProtocol::abort() noexcept
    {
        if (state_ == MoEOverlayDistributedResidencyProtocolState::Published ||
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingRetirementAdmissionConsensus ||
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          ReadyToCloseRetirementAdmission ||
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          RetirementAdmissionClosed ||
            state_ == MoEOverlayDistributedResidencyProtocolState::
                          AwaitingRetirementConsensus ||
            state_ ==
                MoEOverlayDistributedResidencyProtocolState::ReadyToRetire ||
            state_ == MoEOverlayDistributedResidencyProtocolState::Retired)
            return false;
        state_ = MoEOverlayDistributedResidencyProtocolState::Aborted;
        local_vote_.reset();
        return true;
    }
} // namespace llaminar2
