#include "WeightPlan.h"

#include "WeightLifecycleTrace.h"
#include "fort.hpp"

#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        std::string layerKey(int layer, const std::string &suffix)
        {
            return std::to_string(layer) + ":" + suffix;
        }

        std::string suffixForLayerWeight(const std::string &canonical_name)
        {
            const int layer = inferWeightLayer(canonical_name);
            if (layer < 0)
                return canonical_name;
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            if (canonical_name.rfind(prefix, 0) == 0)
                return canonical_name.substr(prefix.size());
            return canonical_name;
        }
    }

    WeightPlan::WeightPlan(InferenceStrategy strategy)
        : strategy_(std::move(strategy))
    {
    }

    void WeightPlan::add(WeightRequirement requirement)
    {
        if (requirement.role == WeightRole::Other && !requirement.canonical_name.empty())
            requirement.role = inferWeightRole(requirement.canonical_name);
        if (requirement.layer < 0)
            requirement.layer = inferWeightLayer(requirement.canonical_name);
        if (requirement.expert < 0)
            requirement.expert = inferWeightExpert(requirement.canonical_name);
        requirements_.push_back(std::move(requirement));
    }

    std::string WeightPlan::renderAuditTable() const
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Name" << "Role" << "Residency" << "Domain" << "Part" << "Rank" << "Layer" << "PP" << "TP" << "Device" << "Prepared" << "Host" << fort::endr;
        for (const auto &req : requirements_)
        {
            table << req.canonical_name
                  << toString(req.role)
                  << toString(req.residency_category)
                  << (req.overlay_domain.empty() ? "-" : req.overlay_domain)
                  << req.overlay_participant_index
                  << req.overlay_participant_world_rank
                  << req.layer
                  << req.pp_stage
                  << req.tp_rank_or_device_index
                  << req.target_device.to_string()
                  << toString(req.expected_prepared_kind)
                  << toString(req.host_policy)
                  << fort::endr;
        }
        return table.to_string();
    }

    ModelWeightSetBuilder::ModelWeightSetBuilder(InferenceStrategy strategy)
        : strategy_(std::move(strategy))
    {
    }

    WeightBinding &ModelWeightSetBuilder::addBinding(WeightBinding binding)
    {
        if (binding.binding_id == 0)
            binding.binding_id = next_binding_id_++;
        if (binding.identity.instance_id == 0)
            binding.identity.instance_id = binding.binding_id;
        if (binding.prepared && binding.prepared->binding_id == 0)
            binding.prepared->binding_id = binding.binding_id;
        bindings_.push_back(std::move(binding));
        return bindings_.back();
    }

    std::vector<WeightBinding> ModelWeightSetBuilder::freezeBindings()
    {
        for (auto &binding : bindings_)
            binding.immutable = true;
        return std::move(bindings_);
    }

    FrozenModelWeightSet::FrozenModelWeightSet(InferenceStrategy strategy, std::vector<WeightBinding> bindings)
        : strategy_(std::move(strategy)), bindings_(std::move(bindings))
    {
        for (size_t index = 0; index < bindings_.size(); ++index)
            indexBinding(index, bindings_[index]);
    }

    void FrozenModelWeightSet::indexBinding(size_t index, const WeightBinding &binding)
    {
        if (binding.identity.layer < 0)
            global_index_[binding.identity.canonical_name] = index;
        else
            layer_index_[layerKey(binding.identity.layer, suffixForLayerWeight(binding.identity.canonical_name))] = index;
    }

    const WeightBinding &FrozenModelWeightSet::global(const std::string &canonical_name) const
    {
        auto it = global_index_.find(canonical_name);
        if (it == global_index_.end())
            throw std::out_of_range("Missing global weight binding: " + canonical_name);
        return bindings_[it->second];
    }

    const WeightBinding &FrozenModelWeightSet::layer(int layer_idx, const std::string &suffix) const
    {
        const WeightBinding *binding = optionalLayer(layer_idx, suffix);
        if (!binding)
            throw std::out_of_range("Missing layer weight binding: layer=" + std::to_string(layer_idx) + " suffix=" + suffix);
        return *binding;
    }

    const WeightBinding *FrozenModelWeightSet::optionalLayer(int layer_idx, const std::string &suffix) const
    {
        auto it = layer_index_.find(layerKey(layer_idx, suffix));
        if (it == layer_index_.end())
            return nullptr;
        return &bindings_[it->second];
    }

    std::vector<const WeightBinding *> FrozenModelWeightSet::forDevice(DeviceId device) const
    {
        std::vector<const WeightBinding *> result;
        for (const auto &binding : bindings_)
        {
            if (binding.residency.home_device == device ||
                (binding.residency.resident_device && *binding.residency.resident_device == device))
            {
                result.push_back(&binding);
            }
        }
        return result;
    }

    void FrozenModelWeightSet::validateForGraph() const
    {
        for (const auto &binding : bindings_)
        {
            if (!binding.immutable)
                throw std::runtime_error("Weight binding is not frozen: " + binding.identity.canonical_name);
            if (binding.identity.canonical_name.empty())
                throw std::runtime_error("Weight binding missing canonical name");
            if (binding.prepared && binding.prepared->binding_id != binding.binding_id)
                throw std::runtime_error("Prepared weight ref binding id mismatch: " + binding.identity.canonical_name);
        }
    }

    std::string FrozenModelWeightSet::renderAuditTable() const
    {
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header << "Binding" << "Name" << "Role" << "Residency" << "Domain" << "Part" << "Rank" << "Derivation" << "Layer" << "Device" << "Prepared" << "Host" << fort::endr;
        for (const auto &binding : bindings_)
        {
            table << binding.binding_id
                  << binding.identity.canonical_name
                  << toString(binding.identity.role)
                  << toString(binding.identity.residency_category)
                  << (binding.identity.overlay_domain.empty() ? "-" : binding.identity.overlay_domain)
                  << binding.identity.overlay_participant_index
                  << binding.identity.overlay_participant_world_rank
                  << toString(binding.identity.derivation)
                  << binding.identity.layer
                  << binding.residency.home_device.to_string()
                  << (binding.prepared ? toString(binding.prepared->kind) : "None")
                  << toString(binding.residency.host_policy)
                  << fort::endr;
        }
        return table.to_string();
    }

    std::string toString(PreparedWeightKind kind)
    {
        switch (kind)
        {
        case PreparedWeightKind::None: return "None";
        case PreparedWeightKind::CpuPackedGemm: return "CpuPackedGemm";
        case PreparedWeightKind::CudaInt8PackedGemm: return "CudaInt8PackedGemm";
        case PreparedWeightKind::RocmInt8PackedGemm: return "RocmInt8PackedGemm";
        case PreparedWeightKind::PreparedEmbedding: return "PreparedEmbedding";
        case PreparedWeightKind::MoeExpertSlab: return "MoeExpertSlab";
        }
        return "None";
    }
}
