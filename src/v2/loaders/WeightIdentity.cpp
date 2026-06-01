#include "WeightIdentity.h"

#include <functional>

namespace llaminar2
{
    std::string toString(WeightRole role)
    {
        switch (role)
        {
        case WeightRole::Embedding: return "Embedding";
        case WeightRole::LMHead: return "LMHead";
        case WeightRole::OutputNorm: return "OutputNorm";
        case WeightRole::AttentionQ: return "AttentionQ";
        case WeightRole::AttentionK: return "AttentionK";
        case WeightRole::AttentionV: return "AttentionV";
        case WeightRole::AttentionWO: return "AttentionWO";
        case WeightRole::FusedQKV: return "FusedQKV";
        case WeightRole::GDNProjection: return "GDNProjection";
        case WeightRole::GDNSsmParam: return "GDNSsmParam";
        case WeightRole::FFNGate: return "FFNGate";
        case WeightRole::FFNUp: return "FFNUp";
        case WeightRole::FFNDown: return "FFNDown";
        case WeightRole::MoERouter: return "MoERouter";
        case WeightRole::MoEExpertGate: return "MoEExpertGate";
        case WeightRole::MoEExpertUp: return "MoEExpertUp";
        case WeightRole::MoEExpertDown: return "MoEExpertDown";
        case WeightRole::SharedExpertGate: return "SharedExpertGate";
        case WeightRole::SharedExpertUp: return "SharedExpertUp";
        case WeightRole::SharedExpertDown: return "SharedExpertDown";
        case WeightRole::Norm: return "Norm";
        case WeightRole::Bias: return "Bias";
        case WeightRole::Other: return "Other";
        }
        return "Other";
    }

    std::string toString(WeightDerivationKind derivation)
    {
        switch (derivation)
        {
        case WeightDerivationKind::Source: return "Source";
        case WeightDerivationKind::RowSlice: return "RowSlice";
        case WeightDerivationKind::ColumnSlice: return "ColumnSlice";
        case WeightDerivationKind::ExpertSlice: return "ExpertSlice";
        case WeightDerivationKind::DeviceClone: return "DeviceClone";
        case WeightDerivationKind::TiedAlias: return "TiedAlias";
        case WeightDerivationKind::FusedSubblockConcat: return "FusedSubblockConcat";
        case WeightDerivationKind::DecodeShard: return "DecodeShard";
        case WeightDerivationKind::RebalancedExpertReplica: return "RebalancedExpertReplica";
        }
        return "Source";
    }

    std::string toString(WeightHostPolicy policy)
    {
        switch (policy)
        {
        case WeightHostPolicy::RequiredForCPUExecution: return "RequiredForCPUExecution";
        case WeightHostPolicy::RequiredUntilGraphMaterialized: return "RequiredUntilGraphMaterialized";
        case WeightHostPolicy::RequiredUntilPreparedOrTransferred: return "RequiredUntilPreparedOrTransferred";
        case WeightHostPolicy::ReleasableAfterPreparation: return "ReleasableAfterPreparation";
        case WeightHostPolicy::Released: return "Released";
        }
        return "RequiredUntilGraphMaterialized";
    }

    std::string toString(WeightResidencyCategory category)
    {
        switch (category)
        {
        case WeightResidencyCategory::Unspecified: return "Unspecified";
        case WeightResidencyCategory::RootNonExpert: return "RootNonExpert";
        case WeightResidencyCategory::SharedExpert: return "SharedExpert";
        case WeightResidencyCategory::AcceleratorRoutedExpert: return "AcceleratorRoutedExpert";
        case WeightResidencyCategory::CpuFallbackExpert: return "CpuFallbackExpert";
        case WeightResidencyCategory::WorkerFallbackExpert: return "WorkerFallbackExpert";
        }
        return "Unspecified";
    }

    const char *toString(WeightLifecycleState state)
    {
        switch (state)
        {
        case WeightLifecycleState::Planned: return "Planned";
        case WeightLifecycleState::SourceLoaded: return "SourceLoaded";
        case WeightLifecycleState::DerivedMaterialized: return "DerivedMaterialized";
        case WeightLifecycleState::DevicePrepared: return "DevicePrepared";
        case WeightLifecycleState::GraphMaterialized: return "GraphMaterialized";
        case WeightLifecycleState::Frozen: return "Frozen";
        case WeightLifecycleState::HostReleased: return "HostReleased";
        }
        return "Unknown";
    }

    WeightRole inferWeightRole(const std::string &name)
    {
        if (name == "token_embd.weight") return WeightRole::Embedding;
        if (name == "output.weight") return WeightRole::LMHead;
        if (name == "output_norm.weight") return WeightRole::OutputNorm;
        if (name.find("attn_qkv.weight") != std::string::npos) return WeightRole::FusedQKV;
        if (name.find("attn_q.weight") != std::string::npos) return WeightRole::AttentionQ;
        if (name.find("attn_k.weight") != std::string::npos) return WeightRole::AttentionK;
        if (name.find("attn_v.weight") != std::string::npos) return WeightRole::AttentionV;
        if (name.find("attn_output.weight") != std::string::npos ||
            name.find("attn_o.weight") != std::string::npos)
            return WeightRole::AttentionWO;
        if (name.find("gdn_qkv.weight") != std::string::npos ||
            name.find("ssm.qkv_proj.weight") != std::string::npos)
            return WeightRole::GDNProjection;
        if (name.find("ssm.") != std::string::npos) return WeightRole::GDNSsmParam;
        if (name.find("ffn_gate_exps.weight") != std::string::npos) return WeightRole::MoEExpertGate;
        if (name.find("ffn_up_exps.weight") != std::string::npos) return WeightRole::MoEExpertUp;
        if (name.find("ffn_down_exps.weight") != std::string::npos) return WeightRole::MoEExpertDown;
        if (name.find("ffn_gate_inp.weight") != std::string::npos) return WeightRole::MoERouter;
        if (name.find("ffn_gate_inp_shexp.weight") != std::string::npos) return WeightRole::SharedExpertGate;
        if (name.find("ffn_gate_shexp.weight") != std::string::npos) return WeightRole::SharedExpertGate;
        if (name.find("ffn_up_shexp.weight") != std::string::npos) return WeightRole::SharedExpertUp;
        if (name.find("ffn_down_shexp.weight") != std::string::npos) return WeightRole::SharedExpertDown;
        if (name.find("shared_expert_gate.weight") != std::string::npos) return WeightRole::SharedExpertGate;
        if (name.find("shared_expert_up.weight") != std::string::npos) return WeightRole::SharedExpertUp;
        if (name.find("shared_expert_down.weight") != std::string::npos) return WeightRole::SharedExpertDown;
        if (name.find("moe_gate.weight") != std::string::npos || name.find("router") != std::string::npos)
            return WeightRole::MoERouter;
        if (name.find("ffn_gate.weight") != std::string::npos) return WeightRole::FFNGate;
        if (name.find("ffn_up.weight") != std::string::npos) return WeightRole::FFNUp;
        if (name.find("ffn_down.weight") != std::string::npos) return WeightRole::FFNDown;
        if (name.find("norm.weight") != std::string::npos) return WeightRole::Norm;
        if (name.find("bias") != std::string::npos) return WeightRole::Bias;
        return WeightRole::Other;
    }

    int inferWeightLayer(const std::string &name)
    {
        const std::string prefix = "blk.";
        const auto pos = name.find(prefix);
        if (pos == std::string::npos)
            return -1;
        size_t start = pos + prefix.size();
        size_t end = start;
        while (end < name.size() && name[end] >= '0' && name[end] <= '9')
            ++end;
        if (end == start)
            return -1;
        return std::stoi(name.substr(start, end - start));
    }

    int inferWeightExpert(const std::string &name)
    {
        const std::string marker = ".experts.";
        const auto pos = name.find(marker);
        if (pos == std::string::npos)
            return -1;
        size_t start = pos + marker.size();
        size_t end = start;
        while (end < name.size() && name[end] >= '0' && name[end] <= '9')
            ++end;
        if (end == start)
            return -1;
        return std::stoi(name.substr(start, end - start));
    }

    uint64_t stableWeightLogicalId(const std::string &canonical_name)
    {
        return static_cast<uint64_t>(std::hash<std::string>{}(canonical_name));
    }

    WeightIdentity makeSourceWeightIdentity(
        const std::string &canonical_name,
        ModelContextId model_id,
        uint64_t instance_id)
    {
        WeightIdentity identity;
        identity.model_id = model_id;
        identity.logical_id = stableWeightLogicalId(canonical_name);
        identity.instance_id = instance_id;
        identity.canonical_name = canonical_name;
        identity.role = inferWeightRole(canonical_name);
        identity.derivation = WeightDerivationKind::Source;
        identity.layer = inferWeightLayer(canonical_name);
        identity.expert = inferWeightExpert(canonical_name);
        return identity;
    }
}
