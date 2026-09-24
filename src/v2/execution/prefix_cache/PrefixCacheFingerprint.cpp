/**
 * @file PrefixCacheFingerprint.cpp
 * @brief Stable prefix compatibility hashing and placement-bound admission.
 *
 * Fields are sorted before hashing so construction order cannot change a key.
 * The movement epoch enters through one typed argument and is returned with
 * the key; independently sampled epochs must never relabel cached state.
 */
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "execution/moe/MoERuntimeTable.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        constexpr uint64_t kFnvOffset = 14695981039346656037ull;
        constexpr uint64_t kFnvPrime = 1099511628211ull;

        /** @brief Fold one byte into the stable FNV-1a accumulator. */
        uint64_t fnvUpdate(uint64_t hash, unsigned char byte)
        {
            hash ^= static_cast<uint64_t>(byte);
            hash *= kFnvPrime;
            return hash;
        }

        /** @brief Hash string bytes and a delimiter to distinguish field boundaries. */
        uint64_t fnvUpdateString(uint64_t hash, const std::string &value)
        {
            for (unsigned char byte : value)
            {
                hash = fnvUpdate(hash, byte);
            }
            return fnvUpdate(hash, 0xffu);
        }

        /** @brief Hash a 64-bit value in fixed byte order on every CPU ISA. */
        uint64_t fnvUpdateU64(uint64_t hash, uint64_t value)
        {
            for (int i = 0; i < 8; ++i)
            {
                hash = fnvUpdate(hash, static_cast<unsigned char>((value >> (i * 8)) & 0xffu));
            }
            return hash;
        }
    } // namespace

    uint64_t hashPrefixFingerprintFields(
        const std::string &part_name,
        const std::vector<PrefixFingerprintField> &fields)
    {
        auto sorted_fields = fields;
        std::sort(sorted_fields.begin(), sorted_fields.end(),
                  [](const PrefixFingerprintField &lhs, const PrefixFingerprintField &rhs)
                  {
                      if (lhs.name != rhs.name)
                          return lhs.name < rhs.name;
                      return lhs.value < rhs.value;
                  });

        uint64_t hash = kFnvOffset;
        hash = fnvUpdateString(hash, part_name);
        for (const auto &field : sorted_fields)
        {
            hash = fnvUpdateString(hash, field.name);
            hash = fnvUpdateString(hash, field.value);
        }
        return hash;
    }

    PrefixFingerprintParts buildPrefixFingerprintParts(
        const PrefixFingerprintMaterial &material)
    {
        PrefixFingerprintParts parts;
        parts.model = hashPrefixFingerprintFields("model", material.model);
        parts.tokenizer = hashPrefixFingerprintFields("tokenizer", material.tokenizer);
        parts.runtime = hashPrefixFingerprintFields("runtime", material.runtime);
        parts.topology = hashPrefixFingerprintFields("topology", material.topology);
        parts.hybrid = hashPrefixFingerprintFields("hybrid", material.hybrid);
        parts.moe = hashPrefixFingerprintFields("moe", material.moe);
        parts.mtp = hashPrefixFingerprintFields("mtp", material.mtp);
        return parts;
    }

    uint64_t combinePrefixFingerprintParts(const PrefixFingerprintParts &parts)
    {
        uint64_t hash = kFnvOffset;
        // Arithmetic is part of restart compatibility, not just physical layout.
        // V1 could persist chunk-mean AQ8 bases (and downstream states), which
        // cannot certify the request-partition-invariant first-input contract.
        // Retain old archives on disk, but never admit them under the new key.
        hash = fnvUpdateString(hash, "prefix-cache-v2");

        const std::array<std::pair<const char *, uint64_t>, 7> ordered_parts{{
            {"model", parts.model},
            {"tokenizer", parts.tokenizer},
            {"runtime", parts.runtime},
            {"topology", parts.topology},
            {"hybrid", parts.hybrid},
            {"moe", parts.moe},
            {"mtp", parts.mtp},
        }};

        for (const auto &[name, value] : ordered_parts)
        {
            hash = fnvUpdateString(hash, name);
            hash = fnvUpdateU64(hash, value);
        }
        return hash;
    }

    PrefixCacheFingerprintResult buildPrefixCacheFingerprint(
        PrefixFingerprintMaterial material,
        bool model_is_moe,
        PrefixCacheMoEPolicy moe_policy,
        uint64_t placement_epoch)
    {
        PrefixCacheFingerprintResult result;
        result.placement_epoch = placement_epoch;
        if (model_is_moe && moe_policy == PrefixCacheMoEPolicy::Disabled)
        {
            result.bypass = true;
            result.bypass_reason = "prefix cache disabled for MoE placement policy";
            return result;
        }

        // Own this field centrally: a second stringly supplied epoch could
        // disagree with the immutable provenance returned alongside the key.
        if (std::any_of(material.moe.begin(), material.moe.end(),
                        [](const PrefixFingerprintField &field)
                        { return field.name == "runtime_movement_epoch"; }))
        {
            throw std::invalid_argument(
                "Prefix fingerprint movement epoch must use the typed argument");
        }
        if (model_is_moe && moe_policy == PrefixCacheMoEPolicy::InvalidateOnRebalance)
        {
            material.moe.push_back(
                {"runtime_movement_epoch", std::to_string(placement_epoch)});
        }
        result.parts = buildPrefixFingerprintParts(material);
        result.key = combinePrefixFingerprintParts(result.parts);
        return result;
    }

    void appendMoEPlacementFingerprintFields(
        std::vector<PrefixFingerprintField> &fields,
        const IMoERuntimeTable &table,
        int layer_count,
        const std::string &scope)
    {
        if (layer_count <= 0)
            return;

        fields.push_back({scope + ".layer_count", std::to_string(layer_count)});
        for (int layer = 0; layer < layer_count; ++layer)
        {
            if (table.decodeRuntimePublicationRequired(layer))
            {
                throw std::logic_error(
                    "[PrefixCacheFingerprint] cannot fingerprint unpublished MoE runtime layer " +
                    std::to_string(layer));
            }
            const auto &state = table.hostLayerState(layer);
            const std::string layer_prefix = scope + ".layer." + std::to_string(layer);
            const uint32_t active_bank = state.active_bank <= 1 ? state.active_bank : 0;
            const auto &bank = state.banks[active_bank];
            const uint32_t expert_count = std::min<uint32_t>(bank.expert_count, kDeviceMoEMaxExperts);

            fields.push_back({layer_prefix + ".active_bank", std::to_string(state.active_bank)});
            fields.push_back({layer_prefix + ".active_epoch", std::to_string(state.active_epoch)});
            fields.push_back({layer_prefix + ".expert_count", std::to_string(state.expert_count)});
            fields.push_back({layer_prefix + ".top_k", std::to_string(state.top_k)});
            fields.push_back({layer_prefix + ".bank.epoch", std::to_string(bank.epoch)});
            fields.push_back({layer_prefix + ".bank.expert_count", std::to_string(bank.expert_count)});

            for (uint32_t expert = 0; expert < expert_count; ++expert)
            {
                const auto &desc = bank.experts[expert];
                const std::string expert_prefix = layer_prefix + ".expert." + std::to_string(expert);
                fields.push_back({expert_prefix + ".logical_expert_id", std::to_string(desc.logical_expert_id)});
                fields.push_back({expert_prefix + ".owner_participant", std::to_string(desc.owner_participant)});
                fields.push_back({expert_prefix + ".local_slot", std::to_string(desc.local_slot)});
                fields.push_back({expert_prefix + ".flags", std::to_string(desc.flags)});
                fields.push_back({expert_prefix + ".local_compute_mask", std::to_string(bank.local_compute_mask[expert])});
                fields.push_back({expert_prefix + ".replica_role", std::to_string(bank.replica_role[expert])});
            }
        }
    }

} // namespace llaminar2
