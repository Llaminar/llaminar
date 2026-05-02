#pragma once

#include "../backends/DeviceId.h"
#include "../tensors/CoherenceState.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>

namespace llaminar2
{
    struct ModelContextId
    {
        uint64_t value = 0;

        bool operator==(const ModelContextId &other) const { return value == other.value; }
    };

    enum class WeightRole
    {
        Embedding,
        LMHead,
        OutputNorm,
        AttentionQ,
        AttentionK,
        AttentionV,
        AttentionWO,
        FusedQKV,
        GDNProjection,
        GDNSsmParam,
        FFNGate,
        FFNUp,
        FFNDown,
        MoERouter,
        MoEExpertGate,
        MoEExpertUp,
        MoEExpertDown,
        SharedExpertGate,
        SharedExpertUp,
        SharedExpertDown,
        Norm,
        Bias,
        Other,
    };

    enum class WeightDerivationKind
    {
        Source,
        RowSlice,
        ColumnSlice,
        ExpertSlice,
        DeviceClone,
        TiedAlias,
        FusedSubblockConcat,
        DecodeShard,
        RebalancedExpertReplica,
    };

    enum class WeightHostPolicy
    {
        RequiredForCPUExecution,
        RequiredUntilGraphMaterialized,
        RequiredUntilPreparedOrTransferred,
        ReleasableAfterPreparation,
        Released,
    };

    struct WeightIdentity
    {
        ModelContextId model_id;
        uint64_t logical_id = 0;
        uint64_t instance_id = 0;
        std::string canonical_name;
        WeightRole role = WeightRole::Other;
        WeightDerivationKind derivation = WeightDerivationKind::Source;
        std::optional<uint64_t> source_instance_id;
        int layer = -1;
        int expert = -1;
        int pp_stage = -1;
        int tp_domain = -1;
        int tp_rank_or_device_index = 0;

        bool operator==(const WeightIdentity &other) const
        {
            return model_id == other.model_id && instance_id == other.instance_id;
        }
    };

    struct WeightSliceSpec
    {
        size_t source_rows = 0;
        size_t source_cols = 0;
        size_t row_start = 0;
        size_t row_count = 0;
        size_t col_start = 0;
        size_t col_count = 0;
        size_t expert_start = 0;
        size_t expert_count = 0;
        bool inner_is_presliced = false;
    };

    struct WeightResidency
    {
        DeviceId home_device = DeviceId::cpu();
        std::optional<DeviceId> resident_device;
        TensorCoherenceState coherence = TensorCoherenceState::HOST_ONLY;
        WeightHostPolicy host_policy = WeightHostPolicy::RequiredUntilGraphMaterialized;
        bool raw_host_data_available = true;
        bool raw_device_data_valid = false;
    };

    std::string toString(WeightRole role);
    std::string toString(WeightDerivationKind derivation);
    std::string toString(WeightHostPolicy policy);

    WeightRole inferWeightRole(const std::string &canonical_name);
    int inferWeightLayer(const std::string &canonical_name);
    int inferWeightExpert(const std::string &canonical_name);
    uint64_t stableWeightLogicalId(const std::string &canonical_name);
    WeightIdentity makeSourceWeightIdentity(
        const std::string &canonical_name,
        ModelContextId model_id = {},
        uint64_t instance_id = 0);
}

namespace std
{
    template <>
    struct hash<llaminar2::ModelContextId>
    {
        size_t operator()(const llaminar2::ModelContextId &id) const noexcept
        {
            return hash<uint64_t>{}(id.value);
        }
    };

    template <>
    struct hash<llaminar2::WeightIdentity>
    {
        size_t operator()(const llaminar2::WeightIdentity &identity) const noexcept
        {
            const size_t model_hash = hash<llaminar2::ModelContextId>{}(identity.model_id);
            const size_t instance_hash = hash<uint64_t>{}(identity.instance_id);
            return model_hash ^ (instance_hash + 0x9e3779b97f4a7c15ULL + (model_hash << 6) + (model_hash >> 2));
        }
    };
}
