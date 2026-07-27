#pragma once

#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace llaminar2::test::moe_llep_perf
{
    struct Shape
    {
        int seq_len = 512;
        int num_experts = 256;
        int top_k = 8;
        int participant_count = 4;
        int d_model = 2048;
        int intermediate = 512;
    };

    inline int envInt(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (!raw || !*raw)
            return fallback;
        char *end = nullptr;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || parsed <= 0)
            return fallback;
        return static_cast<int>(parsed);
    }

    inline uint64_t fnv1a64(const int32_t *data, size_t count)
    {
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < count; ++i)
        {
            uint32_t word = static_cast<uint32_t>(data[i]);
            for (int byte = 0; byte < 4; ++byte)
            {
                hash ^= static_cast<uint8_t>((word >> (byte * 8)) & 0xffu);
                hash *= 1099511628211ull;
            }
        }
        return hash;
    }

    inline uint64_t fnv1a64(const std::vector<int32_t> &data)
    {
        return fnv1a64(data.data(), data.size());
    }

    inline uint64_t fnv1a64Plan(const DeviceMoERebalancePlanEntry *data, size_t count)
    {
        const auto *bytes = reinterpret_cast<const uint8_t *>(data);
        const size_t byte_count = count * sizeof(DeviceMoERebalancePlanEntry);
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < byte_count; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    inline uint64_t fnv1a64Bytes(const void *data, size_t byte_count)
    {
        const auto *bytes = static_cast<const uint8_t *>(data);
        uint64_t hash = 1469598103934665603ull;
        for (size_t i = 0; i < byte_count; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
        return hash;
    }

    inline uint32_t participantMask(int participant_count)
    {
        return participant_count >= 32 ? 0xffffffffu : ((1u << participant_count) - 1u);
    }

    inline void configureRuntimeLayer(DeviceMoELayerRuntime &runtime,
                                      const Shape &shape,
                                      bool all_participants_resident,
                                      int participant_id = 0)
    {
        runtime.participant_id = static_cast<uint32_t>(participant_id);
        runtime.participant_count = static_cast<uint32_t>(shape.participant_count);
        runtime.expert_count = static_cast<uint32_t>(shape.num_experts);
        runtime.top_k = static_cast<uint32_t>(shape.top_k);

        auto &bank = runtime.banks[runtime.active_bank];
        bank.expert_count = static_cast<uint32_t>(shape.num_experts);
        const uint32_t all_mask = participantMask(shape.participant_count);
        for (int expert = 0; expert < shape.num_experts; ++expert)
        {
            const int owner = expert % shape.participant_count;
            bank.experts[expert] = DeviceMoEExpertDescriptor{};
            bank.experts[expert].logical_expert_id = expert;
            bank.experts[expert].owner_participant = owner;
            bank.experts[expert].local_slot = expert;
            const bool local_resident =
                all_participants_resident || owner == participant_id;
            DeviceMoEExpertFlags flags =
                DeviceMoEExpertFlags::Valid | DeviceMoEExpertFlags::Resident;
            if (local_resident)
                flags |= DeviceMoEExpertFlags::LocalCompute;
            bank.experts[expert].flags =
                toMoEExpertFlags(flags);
            bank.local_compute_mask[expert] = local_resident ? 1u : 0u;
            bank.replica_role[expert] =
                local_resident
                    ? static_cast<uint8_t>(DeviceMoEReplicaRole::Primary)
                    : static_cast<uint8_t>(DeviceMoEReplicaRole::None);
            bank.resident_participant_mask[expert] =
                all_participants_resident ? all_mask : (1u << static_cast<uint32_t>(owner));
        }
    }

    struct SyntheticPayloadSpec
    {
        int n = 128;
        int k = 1024;
        uint8_t codebook_id = 5;
    };

    inline uint64_t alignUp(uint64_t value, uint64_t alignment)
    {
        return ((value + alignment - 1u) / alignment) * alignment;
    }

    inline uint32_t syntheticBlocksPerRow(int k)
    {
        return static_cast<uint32_t>((k + 31) / 32);
    }

    inline uint64_t syntheticMatrixDataBytes(const SyntheticPayloadSpec &spec)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.n = spec.n;
        desc.k = spec.k;
        desc.blocks_per_row = syntheticBlocksPerRow(spec.k);
        desc.codebook_id = spec.codebook_id;
        DeviceMoEExpertDirectoryEntry entry{};
        return deviceMoEMatrixPayloadBytes(desc, entry) +
               deviceMoEMatrixScalesBytes(desc) +
               deviceMoEMatrixMinsBytes(desc, entry) +
               deviceMoEMatrixEminsBytes(desc, entry);
    }

    inline uint64_t syntheticExpertDataBytes(const SyntheticPayloadSpec &spec)
    {
        return syntheticMatrixDataBytes(spec) * 3u;
    }

    inline uint64_t syntheticPayloadSlotBytes(const SyntheticPayloadSpec &spec)
    {
        return alignUp(sizeof(DeviceMoEExpertDirectoryEntry) +
                           syntheticExpertDataBytes(spec),
                       256u);
    }

    inline DeviceNativeVNNIMatrixDesc makeSyntheticMatrixDesc(uint8_t *base,
                                                             uint64_t &offset,
                                                             const SyntheticPayloadSpec &spec)
    {
        DeviceNativeVNNIMatrixDesc desc{};
        desc.n = spec.n;
        desc.k = spec.k;
        desc.blocks_per_row = syntheticBlocksPerRow(spec.k);
        desc.codebook_id = spec.codebook_id;

        DeviceMoEExpertDirectoryEntry entry{};
        const uint64_t blocks =
            static_cast<uint64_t>(desc.blocks_per_row) *
            static_cast<uint64_t>(desc.n);
        const uint64_t payload_bytes = deviceMoEMatrixPayloadBytes(desc, entry);
        const uint64_t scales_bytes = blocks * sizeof(uint16_t);
        const uint64_t mins_bytes = deviceMoEMatrixMinsBytes(desc, entry);
        const uint64_t emins_bytes = deviceMoEMatrixEminsBytes(desc, entry);

        desc.payload = base + offset;
        offset += payload_bytes;
        desc.scales = base + offset;
        offset += scales_bytes;
        desc.mins = mins_bytes > 0u ? base + offset : nullptr;
        offset += mins_bytes;
        desc.emins = emins_bytes > 0u ? base + offset : nullptr;
        offset += emins_bytes;
        return desc;
    }

    inline DeviceMoEExpertDescriptor makeSyntheticExpertDescriptor(uint8_t *base,
                                                                  const SyntheticPayloadSpec &spec,
                                                                  int expert,
                                                                  int owner,
                                                                  int local_slot)
    {
        uint64_t offset = 0;
        DeviceMoEExpertDescriptor desc{};
        desc.gate = makeSyntheticMatrixDesc(base, offset, spec);
        desc.up = makeSyntheticMatrixDesc(base, offset, spec);
        desc.down = makeSyntheticMatrixDesc(base, offset, spec);
        desc.logical_expert_id = expert;
        desc.owner_participant = owner;
        desc.local_slot = local_slot;
        desc.flags = toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                      DeviceMoEExpertFlags::Resident |
                                      DeviceMoEExpertFlags::LocalCompute);
        return desc;
    }

    inline DeviceMoEExpertDirectoryEntry makeSyntheticTransferSlot(uint8_t *base,
                                                                   const SyntheticPayloadSpec &spec,
                                                                   uint32_t participant,
                                                                   uint32_t slot)
    {
        DeviceMoEExpertDirectoryEntry entry{};
        entry.descriptor = makeSyntheticExpertDescriptor(
            base, spec, -1, static_cast<int>(participant), static_cast<int>(slot));
        entry.layer = kDeviceMoEInvalidSlot;
        entry.expert = kDeviceMoEInvalidSlot;
        entry.participant = participant;
        entry.slot_index = slot;
        entry.flags = static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::Valid) |
                      static_cast<uint32_t>(DeviceMoERebalanceDirectoryFlags::TransferSlot);
        auto mark_transfer_capacity = [](DeviceNativeVNNIMatrixDesc &matrix)
        {
            uint8_t payload_bytes_per_block = 0;
            uint8_t is_asymmetric = 0;
            uint8_t has_emins = 0;
            if (deviceMoEProjectionFormat(
                    matrix,
                    payload_bytes_per_block,
                    is_asymmetric,
                    has_emins))
            {
                matrix.allocation_payload_bytes_per_block = payload_bytes_per_block;
                matrix.allocation_has_mins = is_asymmetric;
                matrix.allocation_has_emins = has_emins;
            }
        };
        mark_transfer_capacity(entry.descriptor.gate);
        mark_transfer_capacity(entry.descriptor.up);
        mark_transfer_capacity(entry.descriptor.down);
        return entry;
    }

    inline int sourceOrdinalForExpert(int expert, int participant_count)
    {
        return expert / participant_count;
    }

    inline void installSyntheticLocalExpertDescriptors(DeviceMoELayerRuntime &runtime,
                                                       const Shape &shape,
                                                       const SyntheticPayloadSpec &spec,
                                                       int participant_id,
                                                       uint8_t *source_slab,
                                                       uint64_t bytes_per_expert)
    {
        auto &bank = runtime.banks[runtime.active_bank];
        for (int expert = 0; expert < shape.num_experts; ++expert)
        {
            const int owner = expert % shape.participant_count;
            if (owner != participant_id)
                continue;

            const int ordinal = sourceOrdinalForExpert(expert, shape.participant_count);
            auto *base = source_slab + static_cast<uint64_t>(ordinal) * bytes_per_expert;
            bank.experts[expert] = makeSyntheticExpertDescriptor(
                base, spec, expert, owner, ordinal);
            bank.local_compute_mask[expert] = 1u;
            bank.replica_role[expert] =
                static_cast<uint8_t>(DeviceMoEReplicaRole::Primary);
            bank.resident_participant_mask[expert] =
                1u << static_cast<uint32_t>(participant_id);
        }
    }

    /**
     * @brief Install device-backed descriptors for every expert visible locally.
     *
     * The resident-only determinism scenarios intentionally advertise every
     * expert as resident on every participant. That promise includes local
     * executable descriptors: publishing only residency masks while leaving
     * gate/up/down pointers null describes an impossible production state and
     * must trip the grouped assignment kernel's fail-fast guard.
     *
     * @param runtime Host image of the device runtime table being prepared.
     * @param shape Synthetic MoE geometry and participant count.
     * @param spec Quantized matrix layout used for deterministic payload bytes.
     * @param participant_id Participant represented by this test process.
     * @param source_slab Device allocation containing one expert payload per slot.
     * @param bytes_per_expert Byte stride between consecutive expert payloads.
     */
    inline void installSyntheticAllLocalExpertDescriptors(
        DeviceMoELayerRuntime &runtime,
        const Shape &shape,
        const SyntheticPayloadSpec &spec,
        int participant_id,
        uint8_t *source_slab,
        uint64_t bytes_per_expert)
    {
        auto &bank = runtime.banks[runtime.active_bank];
        const uint32_t all_mask = participantMask(shape.participant_count);
        for (int expert = 0; expert < shape.num_experts; ++expert)
        {
            const int owner = expert % shape.participant_count;
            auto *base =
                source_slab + static_cast<uint64_t>(expert) * bytes_per_expert;
            bank.experts[expert] = makeSyntheticExpertDescriptor(
                base,
                spec,
                expert,
                owner,
                expert);
            bank.local_compute_mask[expert] = 1u;
            bank.replica_role[expert] =
                static_cast<uint8_t>(
                    owner == participant_id
                        ? DeviceMoEReplicaRole::Primary
                        : DeviceMoEReplicaRole::Replica);
            bank.resident_participant_mask[expert] = all_mask;
        }
    }

    inline std::vector<float> makeResidentAssignmentRouteExperts(const Shape &shape)
    {
        const int total_slots = shape.seq_len * shape.top_k;
        std::vector<float> experts(static_cast<size_t>(total_slots));
        for (int slot = 0; slot < total_slots; ++slot)
        {
            const int token = slot / shape.top_k;
            const int rank = slot % shape.top_k;
            experts[static_cast<size_t>(slot)] =
                static_cast<float>((token * 17 + rank * 31 + (token >> 2)) % shape.num_experts);
        }
        return experts;
    }

    inline std::vector<float> makeSourceZeroTransferRouteExperts(const Shape &shape)
    {
        const int total_slots = shape.seq_len * shape.top_k;
        const int source_zero_experts = std::max(1, shape.num_experts / shape.participant_count);
        std::vector<float> experts(static_cast<size_t>(total_slots));
        for (int slot = 0; slot < total_slots; ++slot)
        {
            const int source_zero_ordinal = (slot * 17 + slot / 7) % source_zero_experts;
            experts[static_cast<size_t>(slot)] =
                static_cast<float>(source_zero_ordinal * shape.participant_count);
        }
        return experts;
    }

    inline std::vector<float> makeRouteWeights(const Shape &shape)
    {
        const int total_slots = shape.seq_len * shape.top_k;
        std::vector<float> weights(static_cast<size_t>(total_slots));
        for (int slot = 0; slot < total_slots; ++slot)
        {
            const int rank = slot % shape.top_k;
            weights[static_cast<size_t>(slot)] = 1.0f / static_cast<float>(rank + 1);
        }
        return weights;
    }

    inline least_loaded_ep::LeastLoadedExpertAssignmentConfig assignmentConfig(const Shape &shape)
    {
        least_loaded_ep::LeastLoadedExpertAssignmentConfig config;
        config.expert_count = static_cast<uint32_t>(shape.num_experts);
        config.participant_count = static_cast<uint32_t>(shape.participant_count);
        config.enable_balanced_skip = false;
        config.min_foreign_rows_per_transfer = 0;
        return config;
    }

    inline DeviceMoERebalanceConfig rebalanceConfig(const Shape &shape)
    {
        DeviceMoERebalanceConfig config;
        config.num_layers = 1;
        config.num_experts = static_cast<uint32_t>(shape.num_experts);
        config.top_k = static_cast<uint32_t>(shape.top_k);
        config.participant_id = 0;
        config.participant_count = static_cast<uint32_t>(shape.participant_count);
        config.root_participant = 0;
        config.window_size_tokens = 256;
        return config;
    }

    inline DeviceMoERebalanceConfig dynamicMaintenanceConfig(const Shape &shape)
    {
        auto config = rebalanceConfig(shape);
        config.flags =
            static_cast<uint32_t>(DeviceMoERebalanceFlags::PlanMissingArrivals) |
            static_cast<uint32_t>(DeviceMoERebalanceFlags::DeferRuntimeApply) |
            static_cast<uint32_t>(DeviceMoERebalanceFlags::CollectLoadStats);
        config.max_hot_replicas_per_participant = 0;
        config.layer_window_start = 0;
        config.layer_window_count = 1;
        config.layer_wave_count = 1;
        config.routed_assignment_policy = kDeviceMoERebalanceAssignmentStaticOwner;
        return config;
    }

    inline void installSkewedDynamicHistogram(DeviceMoELayerRuntime &runtime,
                                              const Shape &shape)
    {
        for (int expert = 0; expert < kDeviceMoEMaxExperts; ++expert)
        {
            runtime.decode_histogram[expert] = 0;
            runtime.decode_local_histogram[expert] = 0;
        }

        for (int expert = 0; expert < shape.num_experts; ++expert)
        {
            const int owner = expert % shape.participant_count;
            uint64_t count = 8;
            if (owner == 0)
                count = 64;
            else if (owner == 1)
                count = 1;
            else
                count = 16;
            runtime.decode_histogram[expert] = count;
            runtime.decode_local_histogram[expert] = count;
        }
        runtime.decode_histogram[0] = 4096;
        runtime.decode_local_histogram[0] = 4096;
    }

    inline void printTiming(const char *backend,
                            const char *case_name,
                            const Shape &shape,
                            int iterations,
                            float avg_us,
                            uint64_t hash,
                            uint32_t span_count,
                            uint32_t transfer_count)
    {
        std::cout << "backend,case,seq_len,num_experts,top_k,participants,iters,avg_us,hash,spans,transfers\n"
                  << backend << ","
                  << case_name << ","
                  << shape.seq_len << ","
                  << shape.num_experts << ","
                  << shape.top_k << ","
                  << shape.participant_count << ","
                  << iterations << ","
                  << avg_us << ","
                  << hash << ","
                  << span_count << ","
                  << transfer_count << "\n";
    }
} // namespace llaminar2::test::moe_llep_perf
