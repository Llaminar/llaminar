/**
 * @file PrefixCacheFingerprint.cpp
 * @brief Stable prefix compatibility hashing and placement-bound admission.
 *
 * Fields are sorted before hashing so construction order cannot change a key.
 * The movement epoch enters through one typed argument and is returned with
 * the key; independently sampled epochs must never relabel cached state.
 * Arithmetic generations reject incompatible retained KV, recurrent state and
 * terminal logits using metadata before any payload is accessed.
 */
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "execution/moe/MoERuntimeTable.h"
#include "kernels/cpu/gdn/CPUAdjustedGDNReadout.h"
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

    void appendProjectionArithmeticFingerprintFields(
        std::vector<PrefixFingerprintField> &fields,
        GDNProjectionArithmetic gdn,
        MoERouterArithmetic router,
        CUDARouterPrefillArithmetic cuda_prefill)
    {
        const std::string gdn_name(gdnProjectionArithmeticName(gdn));
        const std::string router_name(moeRouterArithmeticName(router));
        const std::string prefill_name(cudaRouterPrefillArithmeticName(cuda_prefill));
        for (const auto &field : fields)
            if (field.name == "projection_arithmetic_revision" ||
                field.name == "gdn_projection_arithmetic" ||
                field.name == "moe_router_arithmetic" ||
                field.name == "cuda_router_prefill_arithmetic" ||
                field.name == "cpu_gdn_readout_arithmetic")
                throw std::invalid_argument("Duplicate projection prefix arithmetic authority");
        // This revision also distinguishes historical diagnostic FP16 decode,
        // whose accumulator order did not certify grouped verifier equivalence.
        fields.push_back({"projection_arithmetic_revision", "half-operands-v1"});
        fields.push_back({"gdn_projection_arithmetic", gdn_name});
        fields.push_back({"moe_router_arithmetic", router_name});
        fields.push_back({"cuda_router_prefill_arithmetic", prefill_name});
        // Changed readouts affect later layers' KV and recurrent state even
        // when the current layer retains the same state-update arithmetic.
        fields.push_back({"cpu_gdn_readout_arithmetic", std::string(kCPUGDNReadoutArithmetic)});
    }

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
        // V2 could also persist rank-zero terminal logits in every TP shard
        // after a full-vocabulary publication. V3 archives the owned vocabulary
        // interval. V9 authenticates the safeguarded CUDA FP16 prefill
        // arithmetic across partial/larger query buckets and saturated grids.
        // This fresh generation also fences the unvalidated V8 checkpoint,
        // whose unrelated MTP/admission changes are not installed here.
        // Earlier V6/V7 tensor dispatch covered fewer geometries, so the same
        // model/topology key could retain FP32-produced GDN state or logits.
        // ROCm register-score scheduling preserves its original FP32 bytes.
        // Retain old files, but reject old arithmetic before reading payloads.
        // V10 preserves original embedding source values instead of the lossy
        // secondary EmbedQ8 representation. Even identical model metadata can
        // now produce different KV/recurrent/terminal state on every backend;
        // reject V9 archives before examining any persisted payload.
        // V11 corrects GDN Q/K normalization to sqrt(sum(x*x) + epsilon)
        // on CPU, CUDA and ROCm. V10's denominator clamp produced different
        // recurrent state; metadata alone must reject those persisted entries.
        // V12 bounds Q8 activation scales to the positive finite half range
        // and saturates rounded values before integer conversion. Earlier
        // zero/infinite scales could produce different downstream state. It
        // also preserves subnormal values in the explicit FP32 byte contract;
        // CUDA's ordinary fast-math policy must not erase those dependencies.
        // V13 gives standalone/full/column shared gating the same separately
        // rounded FP32 product and residual sum, preserving subnormal values.
        // It also includes every ROCm residual tail element. Either correction
        // can change recurrent/KV state; reject V12 metadata before payload I/O.
        // V14 preserves original Q6_K scale factors in prepared GEMM weights.
        // Earlier half-rounded factors can produce different recurrent/KV state;
        // reject that arithmetic using metadata before accessing cached payloads.
        // V15 preserves original IQ2_S base/coefficient words instead of
        // narrowing derived factors. Identical model metadata may produce
        // different recurrent/KV state; reject V14 before payload access.
        // V16 preserves IQ4_XS source base/coefficient products instead of
        // narrowing the derived factor. Fence V15 state before payload access.
        // V17 preserves IQ3_S original factors and rejects V16 state before payload I/O.
        hash = fnvUpdateString(hash, "prefix-cache-v17");

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
