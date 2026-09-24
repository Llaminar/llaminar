/**
 * @file MoETransferStateMachineTestModel.h
 * @brief Shared reference model for CUDA/ROCm expert-transfer lifecycle stress.
 *
 * The production transfer path owns model-wide reusable storage while runtime
 * placement is published independently by each layer. Isolated one-wave tests
 * cannot prove that repeated cross-layer replacement leaves exactly one logical
 * occupant per physical slot. This header provides one deterministic reference
 * model used unchanged by both GPU backend integration suites.
 *
 * The stress geometry mirrors the Qwen 3.6 LLEP failure that motivated the
 * suite: 20 participant-local MoE layers, 128 experts, and 42 durable transfer
 * slots. Eight seeded rotations exercise same-layer and cross-layer reuse,
 * multi-arrival transactions, incomplete publication, inactive-bank
 * alternation, generation advance, hot-slot contention, and full-capacity
 * operation.
 */

#pragma once

#include <gtest/gtest.h>

#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <vector>

namespace llaminar2::test::moe_transfer_state_machine
{
    inline constexpr uint32_t kLayerCount = 20;
    inline constexpr uint32_t kExpertCount = 128;
    inline constexpr uint32_t kTopK = 8;
    inline constexpr uint32_t kParticipantId = 1;
    inline constexpr uint32_t kParticipantCount = 3;
    inline constexpr uint32_t kTransferSlotCount = 42;
    inline constexpr uint32_t kMaxCommandsPerWave = 4;
    inline constexpr uint32_t kStressRotationCount = 8;
    inline constexpr uint32_t kStressWaveCount =
        kStressRotationCount * kTransferSlotCount;
    inline constexpr std::array<uint32_t, kStressRotationCount>
        kStressSequenceSeeds = {
            0x6d2b79f5u,
            0x9e3779b9u,
            0x243f6a88u,
            0xb7e15162u,
            0x94d049bbu,
            0x8538ec77u,
            0xda3e39cbu,
            0xa4093822u,
        };

    /**
     * @brief Logical identity and generation currently stored in one slot.
     */
    struct Occupant
    {
        uint32_t layer = 0;
        uint32_t expert = 0;
        uint32_t owner = 0;
        uint32_t generation = 0;
    };

    using OccupantTable =
        std::array<Occupant, kTransferSlotCount>;

    /**
     * @brief Construct a valid pointer-bearing descriptor without allocating.
     *
     * Apply kernels only publish these descriptors; they do not dereference the
     * synthetic payloads in this state-machine suite. Distinct stable addresses
     * let validation prove that retirement clears all three projections.
     */
    inline DeviceNativeVNNIMatrixDesc fakeMatrix(
        uintptr_t address) noexcept
    {
        DeviceNativeVNNIMatrixDesc descriptor;
        descriptor.payload =
            reinterpret_cast<const uint8_t *>(address);
        descriptor.scales =
            reinterpret_cast<const void *>(address + 0x100u);
        descriptor.n = 16;
        descriptor.k = 16;
        descriptor.blocks_per_row = 1;
        descriptor.codebook_id = 0;
        return descriptor;
    }

    /**
     * @brief Return the stable authoritative owner used by the test domain.
     *
     * The state-machine workload models a cache of remotely owned experts.
     * Ownership therefore follows the same modulo placement used by the base
     * runtime image, and candidate selection excludes participant 1.
     */
    inline uint32_t authoritativeOwnerForExpert(
        uint32_t expert) noexcept
    {
        return expert % kParticipantCount;
    }

    inline bool occupiedByEarlierSlot(
        const OccupantTable &occupants,
        uint32_t slot_limit,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        for (uint32_t slot = 0;
             slot < slot_limit;
             ++slot)
        {
            if (occupants[slot].layer == layer &&
                occupants[slot].expert == expert)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Produce the full-capacity initial transfer directory.
     */
    inline OccupantTable initialOccupants()
    {
        OccupantTable occupants{};
        for (uint32_t slot = 0;
             slot < kTransferSlotCount;
             ++slot)
        {
            bool selected = false;
            for (uint32_t probe = 0;
                 probe < kLayerCount * kExpertCount;
                 ++probe)
            {
                const uint32_t layer =
                    (slot * 7u + probe * 3u) % kLayerCount;
                const uint32_t expert =
                    (slot * 13u + probe * 11u + 3u) %
                    kExpertCount;
                const uint32_t owner =
                    authoritativeOwnerForExpert(expert);
                if (owner == kParticipantId ||
                    occupiedByEarlierSlot(
                        occupants,
                        slot,
                        layer,
                        expert))
                {
                    continue;
                }
                occupants[slot] = {
                    .layer = layer,
                    .expert = expert,
                    .owner = owner,
                    .generation = 1u,
                };
                selected = true;
                break;
            }
            EXPECT_TRUE(selected)
                << "initial transfer state model exhausted identities";
        }
        return occupants;
    }

    inline bool occupiedByAnotherSlot(
        const OccupantTable &occupants,
        uint32_t ignored_slot,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        for (uint32_t slot = 0;
             slot < kTransferSlotCount;
             ++slot)
        {
            if (slot != ignored_slot &&
                occupants[slot].layer == layer &&
                occupants[slot].expert == expert)
            {
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Select a physical slot using several deterministic stress regimes.
     *
     * Rotations alternate complete permutations, reverse order, seven-slot hot
     * sets, and consecutive same-slot churn. The directory therefore sees both
     * fair steady-state traffic and pathological temporal locality.
     */
    inline uint32_t stressSlotForWave(
        uint32_t wave) noexcept
    {
        const uint32_t rotation =
            wave / kTransferSlotCount;
        const uint32_t offset =
            wave % kTransferSlotCount;
        switch (rotation % 4u)
        {
        case 0u:
            return (offset * 17u) %
                   kTransferSlotCount;
        case 1u:
            return kTransferSlotCount - 1u - offset;
        case 2u:
            return offset < kTransferSlotCount / 2u
                       ? offset % 7u
                       : ((offset - kTransferSlotCount / 2u) *
                          13u) %
                             kTransferSlotCount;
        default:
            return ((offset / 2u) * 19u) %
                   kTransferSlotCount;
        }
    }

    /**
     * @brief Select an unused remote expert on one required decode layer.
     */
    inline Occupant nextOccupantForLayer(
        const OccupantTable &occupants,
        uint32_t slot,
        uint32_t wave,
        uint32_t sequence_seed,
        uint32_t target_layer)
    {
        for (uint32_t probe = 0;
             probe < kExpertCount;
             ++probe)
        {
            const uint32_t expert =
                (wave * 19u +
                 probe * 11u +
                 sequence_seed * 13u +
                 17u) %
                kExpertCount;
            const uint32_t owner =
                authoritativeOwnerForExpert(expert);
            if (owner == kParticipantId ||
                occupiedByAnotherSlot(
                    occupants,
                    slot,
                    target_layer,
                    expert) ||
                (target_layer == occupants[slot].layer &&
                 expert == occupants[slot].expert))
            {
                continue;
            }
            return {
                .layer = target_layer,
                .expert = expert,
                .owner = owner,
                .generation =
                    occupants[slot].generation + 1u,
            };
        }

        ADD_FAILURE()
            << "layer-constrained transfer state model exhausted identities";
        return occupants[slot];
    }

    inline int32_t slotForIdentity(
        const OccupantTable &occupants,
        uint32_t layer,
        uint32_t expert) noexcept
    {
        for (uint32_t slot = 0;
             slot < kTransferSlotCount;
             ++slot)
        {
            if (occupants[slot].layer == layer &&
                occupants[slot].expert == expert)
            {
                return static_cast<int32_t>(slot);
            }
        }
        return -1;
    }

    /**
     * @brief Build one participant-local initial placement bank.
     */
    inline MoEPlacementUpdate makeInitialLayerUpdate(
        uint32_t layer,
        const OccupantTable &occupants)
    {
        MoEPlacementUpdate update;
        update.epoch = 1;
        update.expert_count = kExpertCount;
        update.participant_id = kParticipantId;
        update.participant_count = kParticipantCount;
        update.experts.resize(kExpertCount);
        update.local_compute_mask.assign(kExpertCount, 0u);
        update.replica_role.assign(
            kExpertCount,
            static_cast<uint8_t>(
                DeviceMoEReplicaRole::None));
        update.resident_participant_mask.resize(
            kExpertCount);
        update.transient_placement_observed = true;

        for (uint32_t expert = 0;
             expert < kExpertCount;
             ++expert)
        {
            auto &descriptor = update.experts[expert];
            const uint32_t owner =
                authoritativeOwnerForExpert(expert);
            const uintptr_t base =
                0x50000000u +
                static_cast<uintptr_t>(layer) * 0x100000u +
                static_cast<uintptr_t>(expert) * 0x1000u;
            descriptor.gate = fakeMatrix(base + 0x10u);
            descriptor.up = fakeMatrix(base + 0x20u);
            descriptor.down = fakeMatrix(base + 0x30u);
            descriptor.logical_expert_id =
                static_cast<int32_t>(expert);
            descriptor.owner_participant =
                static_cast<int32_t>(owner);
            descriptor.local_slot =
                owner == kParticipantId
                    ? static_cast<int32_t>(expert)
                    : -1;
            descriptor.flags =
                toMoEExpertFlags(
                    DeviceMoEExpertFlags::PreferredOwner);
            update.resident_participant_mask[expert] =
                1u << owner;
            if (owner == kParticipantId)
            {
                descriptor.flags |= toMoEExpertFlags(
                    DeviceMoEExpertFlags::Valid |
                    DeviceMoEExpertFlags::Resident |
                    DeviceMoEExpertFlags::LocalCompute);
                update.local_compute_mask[expert] = 1u;
                update.replica_role[expert] =
                    static_cast<uint8_t>(
                        DeviceMoEReplicaRole::Primary);
            }
            else
            {
                descriptor.gate = {};
                descriptor.up = {};
                descriptor.down = {};
            }

            const int32_t transfer_slot =
                slotForIdentity(
                    occupants,
                    layer,
                    expert);
            if (transfer_slot < 0)
                continue;

            const auto &occupant =
                occupants[static_cast<size_t>(
                    transfer_slot)];
            descriptor.gate = fakeMatrix(
                base + 0x410u);
            descriptor.up = fakeMatrix(
                base + 0x420u);
            descriptor.down = fakeMatrix(
                base + 0x430u);
            descriptor.owner_participant =
                static_cast<int32_t>(occupant.owner);
            descriptor.local_slot = transfer_slot;
            descriptor.flags |= toMoEExpertFlags(
                DeviceMoEExpertFlags::Valid |
                DeviceMoEExpertFlags::Resident |
                DeviceMoEExpertFlags::Replicated |
                DeviceMoEExpertFlags::LocalCompute |
                DeviceMoEExpertFlags::TransferSlot);
            update.local_compute_mask[expert] = 1u;
            update.replica_role[expert] =
                static_cast<uint8_t>(
                    DeviceMoEReplicaRole::Replica);
            update.resident_participant_mask[expert] =
                (1u << occupant.owner) |
                (1u << kParticipantId);
        }
        return update;
    }

    /**
     * @brief Materialize one completed destination-directory publication.
     */
    inline DeviceMoEExpertDirectoryEntry makeReadySlot(
        uint32_t slot,
        const Occupant &occupant,
        uint32_t command_epoch)
    {
        DeviceMoEExpertDirectoryEntry entry;
        const uintptr_t base =
            0x70000000u +
            static_cast<uintptr_t>(slot) * 0x100000u +
            static_cast<uintptr_t>(
                occupant.generation) *
                0x1000u;
        entry.descriptor.gate = fakeMatrix(base + 0x10u);
        entry.descriptor.up = fakeMatrix(base + 0x20u);
        entry.descriptor.down = fakeMatrix(base + 0x30u);
        entry.descriptor.logical_expert_id =
            static_cast<int32_t>(occupant.expert);
        entry.descriptor.owner_participant =
            static_cast<int32_t>(occupant.owner);
        entry.descriptor.local_slot =
            static_cast<int32_t>(slot);
        entry.descriptor.flags = toMoEExpertFlags(
            DeviceMoEExpertFlags::Valid |
            DeviceMoEExpertFlags::Resident |
            DeviceMoEExpertFlags::Replicated |
            DeviceMoEExpertFlags::LocalCompute |
            DeviceMoEExpertFlags::TransferSlot);
        entry.layer = occupant.layer;
        entry.expert = occupant.expert;
        entry.participant = kParticipantId;
        entry.resident_mask =
            (1u << occupant.owner) |
            (1u << kParticipantId);
        entry.epoch = command_epoch;
        entry.slot_index = slot;
        entry.generation = occupant.generation;
        entry.flags =
            static_cast<uint32_t>(
                DeviceMoERebalanceDirectoryFlags::Valid) |
            static_cast<uint32_t>(
                DeviceMoERebalanceDirectoryFlags::Resident) |
            static_cast<uint32_t>(
                DeviceMoERebalanceDirectoryFlags::LocalCompute) |
            static_cast<uint32_t>(
                DeviceMoERebalanceDirectoryFlags::TransferSlot) |
            static_cast<uint32_t>(
                DeviceMoERebalanceDirectoryFlags::CopyComplete);
        return entry;
    }

    /**
     * @brief Complete reference transaction for one adversarial apply wave.
     */
    struct AdversarialWave
    {
        uint32_t target_layer = 0;
        uint32_t command_count = 0;
        uint32_t expected_changed_layers = 0;
        uint32_t same_layer_replacements = 0;
        uint32_t cross_layer_replacements = 0;
        OccupantTable next_occupants{};
        std::array<uint32_t, kMaxCommandsPerWave> slot_indices{};
        std::array<DeviceMoERebalancePlanEntry,
                   kMaxCommandsPerWave>
            plans{};
        std::array<DeviceMoEExpertDirectoryEntry,
                   kMaxCommandsPerWave>
            completed_slots{};
    };

    /**
     * @brief Build one legal but deliberately contentious multi-arrival wave.
     *
     * Every command targets the same decode layer, matching graph-native
     * per-layer apply, while its physical slots may previously belong to
     * several other layers. The returned reference image is committed only
     * after the backend proves that every arrival was ready.
     */
    inline AdversarialWave makeAdversarialWave(
        const OccupantTable &occupants,
        uint32_t wave_number,
        uint32_t command_epoch)
    {
        AdversarialWave wave;
        wave.next_occupants = occupants;
        wave.command_count =
            std::array<uint32_t, 4>{1u, 2u, 4u, 3u}
                [wave_number % 4u];

        const uint32_t rotation =
            wave_number / kTransferSlotCount;
        const uint32_t sequence_seed =
            kStressSequenceSeeds[
                rotation % kStressRotationCount];
        const uint32_t first_slot =
            stressSlotForWave(wave_number);
        const uint32_t first_prior_layer =
            occupants[first_slot].layer;
        wave.target_layer =
            (wave_number % 3u) == 0u
                ? first_prior_layer
                : (first_prior_layer +
                   1u +
                   sequence_seed % (kLayerCount - 1u)) %
                      kLayerCount;
        if ((wave_number % 3u) != 0u &&
            wave.target_layer == first_prior_layer)
        {
            wave.target_layer =
                (wave.target_layer + 1u) %
                kLayerCount;
        }

        std::array<bool, kLayerCount> changed_layers{};
        changed_layers[wave.target_layer] = true;

        for (uint32_t command = 0;
             command < wave.command_count;
             ++command)
        {
            uint32_t slot = 0u;
            bool found_slot = false;
            for (uint32_t probe = 0;
                 probe < kTransferSlotCount;
                 ++probe)
            {
                const uint32_t candidate =
                    stressSlotForWave(
                        wave_number +
                        command * 13u +
                        probe);
                bool already_selected = false;
                for (uint32_t prior_command = 0;
                     prior_command < command;
                     ++prior_command)
                {
                    already_selected |=
                        wave.slot_indices[prior_command] ==
                        candidate;
                }
                if (!already_selected)
                {
                    slot = candidate;
                    found_slot = true;
                    break;
                }
            }
            EXPECT_TRUE(found_slot)
                << "adversarial wave could not select unique physical slots";
            wave.slot_indices[command] = slot;

            const Occupant prior =
                wave.next_occupants[slot];
            const Occupant next =
                nextOccupantForLayer(
                    wave.next_occupants,
                    slot,
                    wave_number * kMaxCommandsPerWave +
                        command,
                    sequence_seed,
                    wave.target_layer);

            auto &plan = wave.plans[command];
            plan.op = static_cast<uint32_t>(
                DeviceMoERebalancePlanOp::ExpertPayloadArrival);
            plan.layer = next.layer;
            plan.expert = next.expert;
            plan.source_participant = next.owner;
            plan.destination_participant = kParticipantId;
            plan.source_resident_mask = 1u << next.owner;
            plan.destination_slot = slot;
            plan.payload_slot = command;
            plan.destination_previous_layer = prior.layer;
            plan.destination_previous_expert = prior.expert;
            plan.destination_generation = prior.generation;
            wave.completed_slots[command] =
                makeReadySlot(
                    slot,
                    next,
                    command_epoch);

            changed_layers[prior.layer] = true;
            if (prior.layer == wave.target_layer)
                ++wave.same_layer_replacements;
            else
                ++wave.cross_layer_replacements;
            wave.next_occupants[slot] = next;
        }

        for (bool changed : changed_layers)
        {
            wave.expected_changed_layers +=
                changed ? 1u : 0u;
        }
        return wave;
    }

    /**
     * @brief Validate every active bank against the exact reference occupants.
     */
    inline ::testing::AssertionResult validateSnapshot(
        const std::array<DeviceMoELayerRuntime,
                         kLayerCount> &runtime,
        const OccupantTable &occupants,
        const DeviceMoERebalanceConfig &config)
    {
        const auto claims =
            deviceMoETransferSlotClaimSummary(
                runtime.data(),
                config);
        if (claims.active_claims != kTransferSlotCount ||
            claims.unique_claims != kTransferSlotCount ||
            claims.duplicate_claims != 0u ||
            claims.invalid_claims != 0u)
        {
            return ::testing::AssertionFailure()
                   << "claim summary active="
                   << claims.active_claims
                   << " unique=" << claims.unique_claims
                   << " duplicate="
                   << claims.duplicate_claims
                   << " invalid=" << claims.invalid_claims
                   << " first_invalid_layer="
                   << claims.first_invalid_layer
                   << " first_invalid_expert="
                   << claims.first_invalid_expert
                   << " first_invalid_slot="
                   << claims.first_invalid_slot
                   << " first_invalid_reasons="
                   << claims.first_invalid_reasons;
        }

        for (uint32_t slot = 0;
             slot < kTransferSlotCount;
             ++slot)
        {
            const auto &occupant = occupants[slot];
            const auto &layer = runtime[occupant.layer];
            const auto &bank =
                layer.banks[layer.active_bank];
            const auto &descriptor =
                bank.experts[occupant.expert];
            if (bank.local_compute_mask[occupant.expert] == 0u ||
                (bank.resident_participant_mask[occupant.expert] &
                 (1u << kParticipantId)) == 0u ||
                descriptor.local_slot !=
                    static_cast<int32_t>(slot) ||
                descriptor.owner_participant !=
                    static_cast<int32_t>(occupant.owner) ||
                !hasMoEExpertFlag(
                    descriptor.flags,
                    DeviceMoEExpertFlags::TransferSlot) ||
                descriptor.gate.payload == nullptr ||
                descriptor.up.payload == nullptr ||
                descriptor.down.payload == nullptr)
            {
                return ::testing::AssertionFailure()
                       << "slot " << slot
                       << " expected layer="
                       << occupant.layer
                       << " expert=" << occupant.expert
                       << " owner=" << occupant.owner
                       << " but published local_slot="
                       << descriptor.local_slot
                       << " owner="
                       << descriptor.owner_participant
                       << " flags=0x" << std::hex
                       << descriptor.flags
                       << " resident_mask=0x"
                       << bank.resident_participant_mask
                              [occupant.expert]
                       << std::dec;
            }
        }

        for (uint32_t layer_index = 0;
             layer_index < kLayerCount;
             ++layer_index)
        {
            const auto &layer = runtime[layer_index];
            const auto &bank =
                layer.banks[layer.active_bank];
            for (uint32_t expert = 0;
                 expert < kExpertCount;
                 ++expert)
            {
                const int32_t expected_slot =
                    slotForIdentity(
                        occupants,
                        layer_index,
                        expert);
                const auto &descriptor =
                    bank.experts[expert];
                const uint32_t owner =
                    authoritativeOwnerForExpert(expert);
                const bool local_resident =
                    (bank.resident_participant_mask[expert] &
                     (1u << kParticipantId)) != 0u;

                if (expected_slot >= 0)
                    continue;

                if (owner == kParticipantId)
                {
                    if (!local_resident ||
                        bank.local_compute_mask[expert] == 0u ||
                        descriptor.owner_participant !=
                            static_cast<int32_t>(owner) ||
                        hasMoEExpertFlag(
                            descriptor.flags,
                            DeviceMoEExpertFlags::TransferSlot) ||
                        descriptor.gate.payload == nullptr ||
                        descriptor.up.payload == nullptr ||
                        descriptor.down.payload == nullptr)
                    {
                        return ::testing::AssertionFailure()
                               << "authoritative local expert layer="
                               << layer_index
                               << " expert=" << expert
                               << " has incoherent publication flags=0x"
                               << std::hex << descriptor.flags
                               << " resident_mask=0x"
                               << bank.resident_participant_mask[expert]
                               << std::dec
                               << " local_slot="
                               << descriptor.local_slot;
                    }
                    continue;
                }

                if (local_resident ||
                    bank.local_compute_mask[expert] != 0u ||
                    hasMoEExpertFlag(
                        descriptor.flags,
                        DeviceMoEExpertFlags::TransferSlot) ||
                    descriptor.local_slot != -1 ||
                    descriptor.gate.payload != nullptr ||
                    descriptor.up.payload != nullptr ||
                    descriptor.down.payload != nullptr)
                {
                    return ::testing::AssertionFailure()
                           << "retired remote expert layer="
                           << layer_index
                           << " expert=" << expert
                           << " retained local payload state flags=0x"
                           << std::hex << descriptor.flags
                           << " resident_mask=0x"
                           << bank.resident_participant_mask[expert]
                           << std::dec
                           << " local_slot="
                           << descriptor.local_slot;
                }
            }
        }
        return ::testing::AssertionSuccess();
    }
} // namespace llaminar2::test::moe_transfer_state_machine
