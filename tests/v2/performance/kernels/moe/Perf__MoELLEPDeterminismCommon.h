#pragma once

#include "execution/moe/DeviceMoERebalanceController.h"
#include "execution/moe/MoERuntimeTable.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
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

    inline uint32_t participantMask(int participant_count)
    {
        return participant_count >= 32 ? 0xffffffffu : ((1u << participant_count) - 1u);
    }

    inline void configureRuntimeLayer(DeviceMoELayerRuntime &runtime,
                                      const Shape &shape,
                                      bool all_participants_resident)
    {
        runtime.participant_id = 0;
        runtime.participant_count = static_cast<uint32_t>(shape.participant_count);
        runtime.expert_count = static_cast<uint32_t>(shape.num_experts);
        runtime.top_k = static_cast<uint32_t>(shape.top_k);

        auto &bank = runtime.banks[runtime.active_bank];
        bank.expert_count = static_cast<uint32_t>(shape.num_experts);
        const uint32_t all_mask = participantMask(shape.participant_count);
        for (int expert = 0; expert < shape.num_experts; ++expert)
        {
            const int owner = expert % shape.participant_count;
            bank.experts[expert].logical_expert_id = expert;
            bank.experts[expert].owner_participant = owner;
            bank.experts[expert].local_slot = expert;
            bank.experts[expert].flags =
                toMoEExpertFlags(DeviceMoEExpertFlags::Valid |
                                 DeviceMoEExpertFlags::Resident);
            bank.resident_participant_mask[expert] =
                all_participants_resident ? all_mask : (1u << static_cast<uint32_t>(owner));
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
